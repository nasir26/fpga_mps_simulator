#!/usr/bin/env python3
"""
Author: Nasir Ali
Organization: C-DAC Noida
Date: June 2026
Target: Xilinx Alveo U55C, Vitis 2023.2, XRT, pyxrt, Python 3.8
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
#  Guard: verify XRT + FPGA before importing anything else
# ──────────────────────────────────────────────────────────────────────────────

def _require_fpga():
    try:
        import pyxrt as _xrt
    except ImportError:
        sys.exit(
            "[FATAL] pyxrt not importable.\n"
            "  source /opt/xilinx/xrt/setup.sh\n"
            "  or run with: PYTHONPATH=/opt/xilinx/xrt/python python3 test_scaling.py"
        )
    try:
        dev = _xrt.device(0)
        bdf = dev.get_info(_xrt.xrt_info_device.bdf)
        name = dev.get_info(_xrt.xrt_info_device.name)
        print(f"  Device 0  BDF  : {bdf}")
        print(f"  Device 0  name : {name}")
        return bdf
    except Exception as e:
        sys.exit(f"[FATAL] Cannot open FPGA device 0: {e}")


# ──────────────────────────────────────────────────────────────────────────────
#  Circuit builders
# ──────────────────────────────────────────────────────────────────────────────

def _circuit_ghz(n: int):
    """H(0) + CX(i, i+1) for i in 0..n-2  →  (|0…0⟩ + |1…1⟩)/√2.
    Bond dimension grows to 2 then stays flat — very HBM-efficient."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def _circuit_1q_rotations(n: int):
    """Ry(θ_i) on each qubit independently — purely 1Q, no entanglement.
    Bond dimension stays 1; tests 1Q-gate throughput at large n."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(42)
    angles = rng.uniform(0.1, math.pi - 0.1, n)
    for i, th in enumerate(angles):
        qc.ry(th, i)
    return qc


def _circuit_brickwall(n: int, depth: int = 4):
    """Alternating layers of Ry rotations + nearest-neighbour CX gates.
    Bond dimension saturates at chi_max for large n×depth; tests entanglement
    capacity and SVD truncation on the FPGA path."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(7)
    for d in range(depth):
        for i in range(n):
            qc.ry(rng.uniform(0.1, math.pi), i)
        parity = d % 2
        for i in range(parity, n - 1, 2):
            qc.cx(i, i + 1)
    return qc


CIRCUIT_FAMILIES = {
    "GHZ":       (_circuit_ghz,        "H + chain-CX  (bond≤2)"),
    "1Q-Rots":   (_circuit_1q_rotations, "random Ry only (bond=1)"),
    "Brickwall": (_circuit_brickwall,   "Ry+CX layers  (bond up to χ_max)"),
}


# ──────────────────────────────────────────────────────────────────────────────
#  Per-run timeout via SIGALRM (Linux only)
# ──────────────────────────────────────────────────────────────────────────────

class _Timeout(Exception):
    pass


@contextmanager
def _time_limit(seconds: int):
    def _handler(signum, frame):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ──────────────────────────────────────────────────────────────────────────────
#  Result record
# ──────────────────────────────────────────────────────────────────────────────

class _Record:
    __slots__ = ('nq', 'family', 'elapsed', 'max_bond', 'status', 'mode')

    def __init__(self, nq, family, elapsed, max_bond, status, mode):
        self.nq       = nq
        self.family   = family
        self.elapsed  = elapsed
        self.max_bond = max_bond
        self.status   = status   # 'ok' | 'timeout' | 'error'
        self.mode     = mode     # e.g. 'shots_fpga'


# ──────────────────────────────────────────────────────────────────────────────
#  Main benchmark loop
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(
        qubit_sizes: List[int],
        families: List[str],
        shots: int,
        chi_max: int,
        num_cards: int,
        timeout_s: int,
        seed: int,
) -> List[_Record]:
    from fpga_mps_simulator import FPGAMPSSimulator

    sim = FPGAMPSSimulator(
        num_cards=num_cards,
        chi_max=chi_max,
        svd_cutoff=1e-12,
        engine="fpga",
        mode="shots",
    )

    # Hard requirement: must be on FPGA
    assert sim._engine == "fpga", (
        f"Simulator resolved to engine={sim._engine!r}. "
        "No CPU fallback is permitted for this benchmark."
    )
    print(f"\n  Simulator engine : {sim._engine.upper()}  (no CPU fallback)")
    print(f"  chi_max          : {chi_max}")
    print(f"  shots per run    : {shots}")
    print(f"  per-run timeout  : {timeout_s}s")
    print(f"  cards available  : {num_cards}")

    records: List[_Record] = []

    # Column header
    hdr = f"  {'nq':>5}  {'circuit':<12}  {'time(s)':>9}  {'bond':>6}  {'mode':<16}  status"
    sep = "  " + "-" * (len(hdr) - 2)
    print(f"\n{sep}")
    print(hdr)
    print(sep)

    for family in families:
        builder, desc = CIRCUIT_FAMILIES[family]
        stopped = False   # once we hit a timeout/error, stop larger sizes
        for nq in qubit_sizes:
            if stopped:
                records.append(_Record(nq, family, None, None, 'skipped', ''))
                continue

            qc = builder(nq)
            elapsed = None
            max_bond = None
            status = 'ok'
            mode_str = ''

            try:
                with _time_limit(timeout_s):
                    t0 = time.perf_counter()
                    res = sim.run(qc, shots=shots, seed=seed)
                    elapsed = time.perf_counter() - t0

                mode_str = res.mode
                # Hard FPGA verification
                if not mode_str.endswith('_fpga'):
                    raise AssertionError(
                        f"mode={mode_str!r} — result did NOT come from FPGA kernel"
                    )
                max_bond = res.get_max_bond()

            except _Timeout:
                elapsed = timeout_s
                status = 'timeout'
                stopped = True
            except AssertionError as e:
                status = f'err:{e}'
                stopped = True
            except Exception as e:
                status = f'err:{type(e).__name__}:{e}'
                stopped = True

            rec = _Record(nq, family, elapsed, max_bond, status, mode_str)
            records.append(rec)

            # Console row
            t_str  = f"{elapsed:9.3f}" if elapsed is not None else f"{'':>9}"
            bd_str = f"{max_bond:6d}"  if max_bond  is not None else f"{'':>6}"
            print(f"  {nq:>5}  {family:<12}  {t_str}  {bd_str}  {mode_str:<16}  {status}")
            sys.stdout.flush()

    print(sep)

    try:
        sim.shutdown()
    except Exception:
        pass

    return records


# ──────────────────────────────────────────────────────────────────────────────
#  Visualization
# ──────────────────────────────────────────────────────────────────────────────

def _plot_results(records: List[_Record], out_path: str, chi_max: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("\n  [WARN] matplotlib not installed — skipping plot. "
              "pip install matplotlib to enable.")
        return

    families = list(dict.fromkeys(r.family for r in records))
    colors = {"GHZ": "#2196F3", "1Q-Rots": "#4CAF50", "Brickwall": "#FF5722"}
    markers = {"GHZ": "o", "1Q-Rots": "s", "Brickwall": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle(
        "FPGA MPS Simulator — Qubit Scaling Benchmark\n"
        f"Xilinx Alveo U55C  |  HLS kernel  |  χ_max = {chi_max}  |  "
        "Author: Nasir Ali, C-DAC Noida",
        fontsize=11, y=1.02,
    )

    # ── Panel 1: execution time vs qubits ────────────────────────────────────
    ax1 = axes[0]
    ax1.set_title("Execution Time vs Qubits", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Number of Qubits")
    ax1.set_ylabel("Wall-clock Time (s)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    for fam in families:
        recs = [r for r in records if r.family == fam and r.status == 'ok'
                and r.elapsed is not None]
        if not recs:
            continue
        xs = [r.nq for r in recs]
        ys = [r.elapsed for r in recs]
        ax1.plot(xs, ys, color=colors.get(fam, "gray"),
                 marker=markers.get(fam, "o"), linewidth=1.8,
                 markersize=6, label=fam)
        # Annotate the last point
        ax1.annotate(f"{ys[-1]:.2f}s",
                     xy=(xs[-1], ys[-1]), fontsize=7,
                     textcoords="offset points", xytext=(5, 2))

    # Mark timeout points
    for fam in families:
        recs = [r for r in records if r.family == fam and r.status == 'timeout']
        for r in recs:
            ax1.axvline(r.nq, color=colors.get(fam, "gray"),
                        ls=":", alpha=0.5, linewidth=1)
    ax1.legend(fontsize=8, loc="upper left")

    # ── Panel 2: max bond dimension vs qubits ────────────────────────────────
    ax2 = axes[1]
    ax2.set_title("Max Bond Dimension vs Qubits", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Number of Qubits")
    ax2.set_ylabel("Max Bond Dimension  χ")
    ax2.set_xscale("log")
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    ax2.axhline(chi_max, color="red", ls="--", linewidth=1.2,
                label=f"χ_max = {chi_max}")

    for fam in families:
        recs = [r for r in records if r.family == fam and r.status == 'ok'
                and r.max_bond is not None and r.max_bond > 0]
        if not recs:
            continue
        xs = [r.nq for r in recs]
        ys = [r.max_bond for r in recs]
        ax2.plot(xs, ys, color=colors.get(fam, "gray"),
                 marker=markers.get(fam, "o"), linewidth=1.8,
                 markersize=6, label=fam)
    ax2.legend(fontsize=8, loc="upper left")

    # ── Panel 3: throughput — qubits per second ───────────────────────────────
    ax3 = axes[2]
    ax3.set_title("Qubit Throughput vs Qubits", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Number of Qubits")
    ax3.set_ylabel("Throughput  (qubits / s)")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.grid(True, which="both", ls="--", alpha=0.4)

    for fam in families:
        recs = [r for r in records if r.family == fam and r.status == 'ok'
                and r.elapsed and r.elapsed > 0]
        if not recs:
            continue
        xs = [r.nq for r in recs]
        ys = [r.nq / r.elapsed for r in recs]
        ax3.plot(xs, ys, color=colors.get(fam, "gray"),
                 marker=markers.get(fam, "o"), linewidth=1.8,
                 markersize=6, label=fam)
    ax3.legend(fontsize=8, loc="upper left")

    # ── FPGA provenance watermark ─────────────────────────────────────────────
    for ax in axes:
        ax.text(0.99, 0.01,
                "✓ FPGA kernel  (mode=shots_fpga)",
                transform=ax.transAxes,
                fontsize=7, color="green", alpha=0.8,
                ha="right", va="bottom")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  FPGA proof summary
# ──────────────────────────────────────────────────────────────────────────────

def _print_fpga_proof(records: List[_Record]):
    ok_recs = [r for r in records if r.status == 'ok']
    if not ok_recs:
        print("\n  [WARN] No successful runs to verify.")
        return

    modes = set(r.mode for r in ok_recs)
    fpga_confirmed = all(m.endswith('_fpga') for m in modes)

    print("\n  ── FPGA Execution Proof ─────────────────────────────────────")
    print(f"  Successful runs      : {len(ok_recs)}")
    print(f"  Unique mode strings  : {sorted(modes)}")
    print(f"  All modes end _fpga  : {'YES ✓' if fpga_confirmed else 'NO ✗  (check output above)'}")
    print(f"  CPU fallback used    : {'NO ✓' if fpga_confirmed else 'POSSIBLY ✗'}")
    max_nq = max(r.nq for r in ok_recs)
    max_rec = next(r for r in ok_recs if r.nq == max_nq)
    print(f"  Largest successful   : {max_nq} qubits  "
          f"({max_rec.family}, {max_rec.elapsed:.3f}s, bond={max_rec.max_bond})")
    print("  " + "─" * 62)


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _main() -> int:
    ap = argparse.ArgumentParser(
        description="FPGA MPS Simulator qubit-scaling benchmark"
    )
    ap.add_argument("--cards",    type=int, default=4,    choices=(1, 2, 4))
    ap.add_argument("--shots",    type=int, default=1024,
                    help="measurement shots per circuit (default 1024)")
    ap.add_argument("--chi",      type=int, default=64,
                    help="max bond dimension (default 64)")
    ap.add_argument("--timeout",  type=int, default=60,
                    help="per-run timeout in seconds (default 60)")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--families", nargs="+",
                    default=["GHZ", "1Q-Rots", "Brickwall"],
                    choices=list(CIRCUIT_FAMILIES.keys()),
                    help="circuit families to benchmark")
    ap.add_argument("--max-qubits", type=int, default=500,
                    help="upper qubit limit to probe (default 500)")
    ap.add_argument("--out", default="scaling_results.png",
                    help="output plot filename (default scaling_results.png)")
    args = ap.parse_args()

    print("\n" + "=" * 66)
    print("  FPGA MPS Simulator — Qubit Scaling Benchmark")
    print("  Author: Nasir Ali  |  C-DAC Noida  |  June 2026")
    print("=" * 66)

    # ── hardware check ────────────────────────────────────────────────────────
    print("\n  Verifying FPGA hardware ...")
    bdf = _require_fpga()

    # ── qubit size sweep ──────────────────────────────────────────────────────
    # Dense at small sizes, then geometric spacing to probe large qubit counts
    small   = [2, 4, 8, 16, 32]
    medium  = [50, 64, 100, 128]
    large   = [200, 300, 400, 500]
    sizes   = [n for n in (small + medium + large) if n <= args.max_qubits]
    # Remove duplicates while preserving order
    seen = set()
    qubit_sizes = [n for n in sizes if not (n in seen or seen.add(n))]

    print(f"\n  Circuit families : {args.families}")
    print(f"  Qubit sweep      : {qubit_sizes}")

    records = run_benchmark(
        qubit_sizes  = qubit_sizes,
        families     = args.families,
        shots        = args.shots,
        chi_max      = args.chi,
        num_cards    = args.cards,
        timeout_s    = args.timeout,
        seed         = args.seed,
    )

    _print_fpga_proof(records)
    _plot_results(records, args.out, args.chi)

    # Return code: 0 if at least one family had at least one successful run
    any_ok = any(r.status == 'ok' for r in records)
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(_main())
