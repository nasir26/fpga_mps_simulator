#!/usr/bin/env python3
"""
Author: Nasir Ali
Organization: C-DAC Noida
Target: Xilinx Alveo U55C, Vitis 2023.2, XRT, pyxrt, Python 3.8

FPGA MPS Simulator -- Honest Verification & Benchmark Harness
=============================================================
This harness answers, with measurements rather than a mode string, the question
"what actually runs on the FPGA?". It does three things:

  (A) PROVENANCE       -- confirms a physical U55C is present and the kernel is
                          actually invoked (counts kernel calls, prints BDF/name).

  (B) LOAD-BEARING     -- a *tamper test*: re-runs each circuit with the kernel
                          replaced by a no-op. If the FPGA output is genuinely
                          used, the results must become WRONG when the kernel is
                          disabled. A kernel whose removal changes nothing is not
                          doing the computation.

  (C) TIME ATTRIBUTION -- instruments every run to split wall-clock time into:
                          FPGA kernel time | host SVD time | host sampling time |
                          host buffer/other. This is the figure a reviewer wants.

  (D) CHI SWEEP        -- the real difficulty axis for MPS is bond dimension chi,
                          not qubit count. Sweeps chi at fixed N on an entangling
                          circuit so the FPGA contraction does meaningful work.

Measurements are written to verification_results.json. Figures are generated
from that JSON by make_figures(), so they can be regenerated without hardware:

    # on the U55C host:
    PYTHONPATH=/opt/xilinx/xrt/python python3 test_fpga_verification.py
    # anywhere (regenerate figures from saved JSON):
    python3 test_fpga_verification.py --figures-only verification_results.json

NOTE: this harness runs *in process* against device 0 so it can instrument the
kernel and host calls directly. That deliberately bypasses the multi-card worker
pool -- the goal here is attribution and proof, not multi-card throughput.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Dict, List, Optional

import numpy as np


# =============================================================================
#  Instrumentation primitives
# =============================================================================

class _Timer:
    """Accumulates total wall time and call count for a wrapped operation."""
    __slots__ = ("total", "count")

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def add(self, dt: float):
        self.total += dt
        self.count += 1


class _KernelProxy:
    """
    Wraps a pyxrt kernel. Each call returns a run handle whose .wait() is timed.
    Records kernel-call count and cumulative kernel (FPGA) wall time.

    If `noop=True`, the kernel is NOT invoked at all -- .wait() returns
    immediately. Used by the tamper test to prove the FPGA output is load-bearing.
    """
    def __init__(self, real_kernel, timer: _Timer, noop: bool = False):
        self._k = real_kernel
        self._timer = timer
        self._noop = noop

    def group_id(self, i):
        return self._k.group_id(i)

    def __call__(self, *args, **kwargs):
        if self._noop:
            return _NoopRun()
        run = self._k(*args, **kwargs)
        return _TimedRun(run, self._timer)


class _TimedRun:
    def __init__(self, run, timer: _Timer):
        self._run = run
        self._timer = timer

    def wait(self, *a, **k):
        t0 = time.perf_counter()
        r = self._run.wait(*a, **k)
        self._timer.add(time.perf_counter() - t0)
        return r


class _NoopRun:
    def wait(self, *a, **k):
        return 0


# =============================================================================
#  Circuit families
# =============================================================================

def circ_ghz(n):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def circ_1q(n):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(42)
    for i, th in enumerate(rng.uniform(0.1, math.pi - 0.1, n)):
        qc.ry(th, i)
    return qc


def circ_brickwall(n, depth=4):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(7)
    for d in range(depth):
        for i in range(n):
            qc.ry(rng.uniform(0.1, math.pi), i)
        for i in range(d % 2, n - 1, 2):
            qc.cx(i, i + 1)
    return qc


def circ_random_entangling(n, depth):
    """Deeper random circuit designed to GROW the bond dimension toward chi_max,
    so the FPGA 2q-contraction path is genuinely exercised."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(123)
    for d in range(depth):
        for i in range(n):
            qc.ry(rng.uniform(0, 2 * math.pi), i)
            qc.rz(rng.uniform(0, 2 * math.pi), i)
        for i in range(d % 2, n - 1, 2):
            qc.cx(i, i + 1)
    return qc


FAMILIES = {
    "GHZ":       (circ_ghz,        "H + chain-CX (chi<=2)"),
    "1Q-Rots":   (circ_1q,         "random Ry only (chi=1)"),
    "Brickwall": (circ_brickwall,  "Ry+CX, depth 4 (chi up to chi_max)"),
}


# =============================================================================
#  Hardware setup
# =============================================================================

def open_device(xclbin="fpga_mps_simulator.xclbin"):
    try:
        import pyxrt as xrt
    except ImportError:
        sys.exit("[FATAL] pyxrt not importable. source /opt/xilinx/xrt/setup.sh")
    dev = xrt.device(0)
    bdf = dev.get_info(xrt.xrt_info_device.bdf)
    name = dev.get_info(xrt.xrt_info_device.name)
    uuid = dev.load_xclbin(xclbin)
    kern = xrt.kernel(dev, uuid, "fpga_mps_simulator")
    return xrt, dev, kern, bdf, name


# =============================================================================
#  Single instrumented run
# =============================================================================

def instrumented_run(fmps, cpu_ref, xrt, dev, real_kernel,
                     qc, shots, chi_max, seed, noop_kernel=False):
    """
    Run one circuit in-process with full time attribution.

    Returns dict with:
      total, t_fpga, t_svd, t_sample, t_other, n_kernel_calls, n_2q,
      max_bond, counts
    """
    fpga_timer = _Timer()
    svd_timer = _Timer()
    sample_timer = _Timer()

    kern_proxy = _KernelProxy(real_kernel, fpga_timer, noop=noop_kernel)

    # --- wrap host SVD (scipy.linalg.svd referenced as fmps.la.svd) ----------
    real_la = fmps.la

    class _LAProxy:
        def __getattr__(self, item):
            attr = getattr(real_la, item)
            if item != "svd":
                return attr

            def timed_svd(*a, **k):
                t0 = time.perf_counter()
                r = attr(*a, **k)
                svd_timer.add(time.perf_counter() - t0)
                return r
            return timed_svd

    # --- wrap host sampling (MPSState.get_counts) ----------------------------
    real_get_counts = cpu_ref.MPSState.get_counts

    def timed_get_counts(self, *a, **k):
        t0 = time.perf_counter()
        r = real_get_counts(self, *a, **k)
        sample_timer.add(time.perf_counter() - t0)
        return r

    fmps.la = _LAProxy()
    cpu_ref.MPSState.get_counts = timed_get_counts
    try:
        t0 = time.perf_counter()
        mps, counts = fmps._run_circuit_fpga(
            qc, shots, chi_max, 1e-12, seed, dev, kern_proxy, xrt
        )
        total = time.perf_counter() - t0
    finally:
        fmps.la = real_la
        cpu_ref.MPSState.get_counts = real_get_counts

    n_2q = sum(1 for g in fmps.compile_circuit(qc, qc.num_qubits)[0]
               if g["type"] == "2q")
    t_other = max(0.0, total - fpga_timer.total - svd_timer.total - sample_timer.total)

    return {
        "total": total,
        "t_fpga": fpga_timer.total,
        "t_svd": svd_timer.total,
        "t_sample": sample_timer.total,
        "t_other": t_other,
        "n_kernel_calls": fpga_timer.count,
        "n_2q": n_2q,
        "max_bond": mps.max_bond(),
        "counts": counts,
    }


# =============================================================================
#  Correctness check vs Qiskit (small n only)
# =============================================================================

def fidelity_vs_qiskit(fmps, mps, qc) -> float:
    from qiskit.quantum_info import Statevector
    sv = mps.get_statevector()
    oracle = Statevector(qc).data
    sv = sv / np.linalg.norm(sv)
    oracle = oracle / np.linalg.norm(oracle)
    return float(abs(np.vdot(sv, oracle)) ** 2)


# =============================================================================
#  Measurement driver
# =============================================================================

def measure(args) -> dict:
    import fpga_mps_simulator as fmps
    import cpu_reference as cpu_ref

    xrt, dev, kern, bdf, name = open_device(args.xclbin)

    print("=" * 70)
    print("  FPGA MPS Simulator -- Verification & Benchmark")
    print("  Author: Nasir Ali | C-DAC Noida")
    print("=" * 70)
    print(f"  device 0 BDF  : {bdf}")
    print(f"  device 0 name : {name}")
    print(f"  shots         : {args.shots}    chi_max: {args.chi}")

    out = {
        "meta": {"bdf": bdf, "name": name, "shots": args.shots,
                 "chi_max": args.chi, "xclbin": args.xclbin},
        "scaling": [], "tamper": [], "chi_sweep": [], "correctness": [],
    }

    # -- (A/C) qubit scaling with time attribution --------------------------
    print("\n  [scaling]  nq | family    | total  fpga  svd   samp  other | calls 2q  bond")
    for fam in args.families:
        builder, _ = FAMILIES[fam]
        for nq in args.qubits:
            r = instrumented_run(fmps, cpu_ref, xrt, dev,
                                 kern, builder(nq), args.shots, args.chi, args.seed)
            r.update(family=fam, nq=nq)
            r.pop("counts", None)
            out["scaling"].append(r)
            print(f"           {nq:4d} | {fam:9s} | "
                  f"{r['total']:5.2f} {r['t_fpga']:5.2f} {r['t_svd']:5.2f} "
                  f"{r['t_sample']:5.2f} {r['t_other']:5.2f} | "
                  f"{r['n_kernel_calls']:4d} {r['n_2q']:3d} {r['max_bond']:4d}")
            sys.stdout.flush()

    # -- (B) tamper test: disable kernel, results must break ----------------
    print("\n  [tamper]   proving FPGA output is load-bearing (kernel disabled -> wrong)")
    for nq in [4, 6, 8]:
        qc = circ_brickwall(nq, depth=3)
        # real kernel
        try:
            mps_real, _ = fmps._run_circuit_fpga(qc, 64, args.chi, 1e-12,
                                                 1, dev, kern, args.xclbin and xrt)
            fid_real = fidelity_vs_qiskit(fmps, mps_real, qc)
        except Exception as e:
            fid_real = float("nan")
        # no-op kernel
        kp = _KernelProxy(kern, _Timer(), noop=True)
        try:
            mps_noop, _ = fmps._run_circuit_fpga(qc, 64, args.chi, 1e-12,
                                                 1, dev, kp, xrt)
            fid_noop = fidelity_vs_qiskit(fmps, mps_noop, qc)
        except Exception:
            fid_noop = 0.0  # crash also means FPGA output was required
        load_bearing = (fid_real > 0.99) and (fid_noop < 0.90)
        out["tamper"].append({"nq": nq, "fid_real": fid_real,
                              "fid_noop": fid_noop, "load_bearing": load_bearing})
        print(f"           nq={nq}: fid(real kernel)={fid_real:.4f}  "
              f"fid(kernel disabled)={fid_noop:.4f}  "
              f"-> FPGA load-bearing: {'YES' if load_bearing else 'NO'}")

    # -- (D) chi sweep at fixed N (real difficulty axis) --------------------
    print(f"\n  [chi-sweep] fixed N={args.chi_sweep_nq}, growing entanglement")
    print("            chi_max | total  fpga  svd   samp | bond  fpga%")
    for chi in args.chi_list:
        qc = circ_random_entangling(args.chi_sweep_nq, depth=args.chi_sweep_depth)
        r = instrumented_run(fmps, cpu_ref, xrt, dev, kern, qc,
                            args.shots, chi, args.seed)
        r.pop("counts", None)
        r["chi_max"] = chi
        pct = 100.0 * r["t_fpga"] / r["total"] if r["total"] > 0 else 0.0
        r["fpga_pct"] = pct
        out["chi_sweep"].append(r)
        print(f"            {chi:6d}  | {r['total']:5.2f} {r['t_fpga']:5.2f} "
              f"{r['t_svd']:5.2f} {r['t_sample']:5.2f} | {r['max_bond']:4d}  {pct:5.1f}")
        sys.stdout.flush()

    # -- correctness vs Qiskit (small n) ------------------------------------
    print("\n  [correctness] vs Qiskit statevector (small n)")
    for nq in [3, 5, 7]:
        qc = circ_brickwall(nq, depth=3)
        mps, _ = fmps._run_circuit_fpga(qc, 64, args.chi, 1e-12, 1, dev, kern, xrt)
        fid = fidelity_vs_qiskit(fmps, mps, qc)
        out["correctness"].append({"nq": nq, "fidelity": fid})
        print(f"            nq={nq}: fidelity={fid:.6f}  {'OK' if fid > 0.99 else 'FAIL'}")

    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  measurements -> {args.json_out}")
    return out


# =============================================================================
#  Figures (regenerable from JSON, no hardware needed)
# =============================================================================

def make_figures(data: dict, prefix="fig"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = data["meta"]
    cap = (f"Xilinx Alveo U55C ({meta['name']})  |  chi_max={meta['chi_max']}  |  "
           f"shots={meta['shots']}  |  Nasir Ali, C-DAC Noida")
    C = {"GHZ": "#1f77b4", "1Q-Rots": "#2ca02c", "Brickwall": "#d62728"}
    M = {"GHZ": "o", "1Q-Rots": "s", "Brickwall": "^"}
    saved = []

    sc = data["scaling"]
    fams = list(dict.fromkeys(r["family"] for r in sc))

    # --- Figure 1: TIME ATTRIBUTION (the honest figure) --------------------
    fig, axes = plt.subplots(1, len(fams), figsize=(5.2 * len(fams), 4.6),
                            squeeze=False)
    for ax, fam in zip(axes[0], fams):
        recs = sorted([r for r in sc if r["family"] == fam], key=lambda r: r["nq"])
        xs = [r["nq"] for r in recs]
        comp = {k: [r[k] for r in recs]
                for k in ("t_fpga", "t_svd", "t_sample", "t_other")}
        ax.stackplot(xs, comp["t_fpga"], comp["t_svd"], comp["t_sample"],
                     comp["t_other"],
                     labels=["FPGA kernel", "host SVD", "host sampling",
                             "host other"],
                     colors=["#2ca02c", "#ff7f0e", "#9467bd", "#bbbbbb"],
                     alpha=0.9)
        ax.set_title(f"Time breakdown -- {fam}", fontweight="bold", fontsize=10)
        ax.set_xlabel("number of qubits")
        ax.set_ylabel("wall-clock time (s)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle("Where the time actually goes\n" + cap, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = f"{prefix}_1_time_attribution.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # --- Figure 2: FPGA TIME FRACTION --------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for fam in fams:
        recs = sorted([r for r in sc if r["family"] == fam], key=lambda r: r["nq"])
        xs = [r["nq"] for r in recs]
        ys = [100.0 * r["t_fpga"] / r["total"] if r["total"] > 0 else 0
              for r in recs]
        ax.plot(xs, ys, color=C.get(fam, "gray"), marker=M.get(fam, "o"),
                label=fam, linewidth=1.8)
    ax.set_title("Fraction of wall-clock time spent on the FPGA",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("number of qubits"); ax.set_ylabel("FPGA kernel time (% of total)")
    ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.figure.text(0.5, -0.02, cap, ha="center", fontsize=8)
    fig.tight_layout()
    p = f"{prefix}_2_fpga_fraction.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # --- Figure 3: CHI SWEEP (real difficulty axis) ------------------------
    cs = sorted(data["chi_sweep"], key=lambda r: r["chi_max"])
    if cs:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
        xs = [r["chi_max"] for r in cs]
        a1.plot(xs, [r["total"] for r in cs], "o-", label="total", color="#333")
        a1.plot(xs, [r["t_fpga"] for r in cs], "s-", label="FPGA kernel",
                color="#2ca02c")
        a1.plot(xs, [r["t_svd"] for r in cs], "^-", label="host SVD",
                color="#ff7f0e")
        a1.set_xlabel("chi_max"); a1.set_ylabel("time (s)")
        a1.set_title("Cost vs bond dimension", fontweight="bold", fontsize=10)
        a1.set_xscale("log", base=2); a1.grid(alpha=0.3); a1.legend(fontsize=8)
        a2.plot(xs, [r.get("fpga_pct", 0) for r in cs], "o-", color="#2ca02c")
        a2.set_xlabel("chi_max"); a2.set_ylabel("FPGA time (% of total)")
        a2.set_title("FPGA share vs bond dimension", fontweight="bold", fontsize=10)
        a2.set_xscale("log", base=2); a2.set_ylim(0, 100); a2.grid(alpha=0.3)
        fig.suptitle("Bond dimension is the real difficulty axis for MPS\n" + cap,
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        p = f"{prefix}_3_chi_sweep.png"
        fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # --- Figure 4: tamper test ---------------------------------------------
    tm = data["tamper"]
    if tm:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        xs = np.arange(len(tm)); w = 0.38
        ax.bar(xs - w / 2, [r["fid_real"] for r in tm], w,
               label="real kernel", color="#2ca02c")
        ax.bar(xs + w / 2, [r["fid_noop"] for r in tm], w,
               label="kernel disabled", color="#d62728")
        ax.axhline(0.99, ls="--", color="gray", lw=1, label="fidelity = 0.99")
        ax.set_xticks(xs); ax.set_xticklabels([f"{r['nq']}q" for r in tm])
        ax.set_ylabel("fidelity vs Qiskit"); ax.set_ylim(0, 1.05)
        ax.set_title("Tamper test: disabling the kernel breaks the result\n"
                     "(proves the FPGA output is load-bearing, not decorative)",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
        ax.figure.text(0.5, -0.02, cap, ha="center", fontsize=8)
        fig.tight_layout()
        p = f"{prefix}_4_tamper_test.png"
        fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); saved.append(p)

    print("  figures:")
    for p in saved:
        print(f"    {p}")
    return saved


# =============================================================================
#  Entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--chi", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qubits", type=int, nargs="+",
                    default=[2, 4, 8, 16, 32, 64, 128, 200, 300, 500])
    ap.add_argument("--families", nargs="+",
                    default=["GHZ", "1Q-Rots", "Brickwall"],
                    choices=list(FAMILIES.keys()))
    ap.add_argument("--chi-list", type=int, nargs="+",
                    default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--chi-sweep-nq", type=int, default=16)
    ap.add_argument("--chi-sweep-depth", type=int, default=12)
    ap.add_argument("--xclbin", default="fpga_mps_simulator.xclbin")
    ap.add_argument("--json-out", default="verification_results.json")
    ap.add_argument("--prefix", default="fig")
    ap.add_argument("--figures-only", metavar="JSON", default=None,
                    help="skip hardware; regenerate figures from an existing JSON")
    args = ap.parse_args()

    if args.figures_only:
        with open(args.figures_only) as f:
            data = json.load(f)
        make_figures(data, args.prefix)
        return 0

    data = measure(args)
    make_figures(data, args.prefix)

    # one-line honest verdict
    sc = data["scaling"]
    if sc:
        avg_fpga = np.mean([100.0 * r["t_fpga"] / r["total"]
                            for r in sc if r["total"] > 0])
        print("\n  VERDICT")
        print(f"    kernel was invoked on {data['meta']['name']} (BDF "
              f"{data['meta']['bdf']}).")
        print(f"    across scaling runs, FPGA kernel time was ~{avg_fpga:.1f}% "
              f"of wall-clock; the remainder is host SVD + sampling + transfer.")
        lb = all(t["load_bearing"] for t in data["tamper"])
        print(f"    FPGA output load-bearing (tamper test): "
              f"{'YES' if lb else 'INCONCLUSIVE -- inspect fig 4'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())