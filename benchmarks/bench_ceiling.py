#!/usr/bin/env python3
"""
bench_ceiling.py — per-family qubit-ceiling discovery for the FPGA MPS
simulator on a Xilinx Alveo U55C.

Author: Nasir Ali, C-DAC Noida

WHY THIS SCRIPT EXISTS
-----------------------
test_scaling.py sweeps every circuit family over the *same* fixed qubit list.
That breaks down once families have wildly different limits — e.g. 1Q-Rots
(bond=1, essentially free) keeps working past 2000 qubits while QCNN-Cluster
(heavy SWAP routing + Toffoli decomposition) tops out closer to 64. This
script instead discovers each family's own ceiling independently, so a
family that dies early never truncates the sweep for the others.

CEILING-DISCOVERY ALGORITHM (see discover_family() below)
-----------------------------------------------------------
1. Coarse growth phase — try qubit counts [2,4,8,16,32,64,128,256,512,1000],
   then keep doubling (2000, 4000, ...) until a STOP condition:
     - a run HARD-FAILS (exception, e.g. an XRT "failed to allocate userptr
       bo" style allocation error),
     - a run exceeds --timeout seconds (enforced with SIGALRM, matching
       test_scaling.py's _time_limit()),
     - or the run size would exceed the global --max-cap safety limit
       (in which case we also probe exactly at --max-cap once, then stop
       cleanly with no failure).
   Every attempt (ok / timeout / hard_fail) is recorded with its error text.
2. Bisection refine (--bisect, on by default) — once we have the last
   passing size (last_ok) and the first failing size (first_fail), binary
   search between them until the gap is <= --tol qubits. This turns a coarse
   "passed 2000, failed 4000" into a precise ceiling like "~2634".
3. Each family is discovered inside its own try/except in main(): a family
   crashing its driver logic (as opposed to a single run failing, which is
   already handled) cannot abort the other families.
4. Recovery after a timeout/hard_fail: an interrupted or failed run can
   leave the FPGA worker processes / result queue in a stale state — the
   card that was mid-job keeps running on real hardware (there is no
   host-side kernel-cancel API) and its late completion message can get
   misattributed to a later run, or a plain SIGTERM-based teardown can
   itself hang if the worker is blocked inside an uninterruptible XRT/DMA
   wait. _SimHolder.recover() (see below) handles this by SIGKILL-ing the
   worker PIDs directly, waiting up to --recovery-grace seconds for them to
   actually die, and then building a brand-new FPGAMPSSimulator rather than
   reusing the old one — bounding the worst case instead of risking an
   indefinite hang. run_attempt() calls this after any non-ok result.

CIRCUIT BUILDERS ARE REUSED, NOT REIMPLEMENTED
-------------------------------------------------
GHZ / 1Q-Rots / Brickwall / QCNN-Cluster / QCNN-Generic come straight from
test_scaling.CIRCUIT_FAMILIES (which itself merges qcnn_circuits.py). Four
new families (W-State, GHZ-Ladder, TFIM-Trotter, Random-Brickwork) live in
extra_circuits.py, built from the same engine-native gate set (single-qubit
h/x/rx/ry/rz + adjacent cx/cz — see that file's docstring).

RE-RUNNING A SINGLE FAMILY
-----------------------------
    python3 bench_ceiling.py --families TFIM-Trotter
    python3 bench_ceiling.py --families 1Q-Rots GHZ --max-cap 4096 --timeout 90

RESULTS
-------
    results/summary.csv, results/summary.json   — every attempt, every family
    results/ceilings.json                        — max_ok_qubits / first_fail / reason
    results/<family>_time.png                    — per-family time-vs-qubits (A)
    results/<family>_metrics.png                 — per-family 3-panel (B)
    results/combined_time.png / _bond.png / _throughput.png  — cross-family overlays (C)
    results/ceilings_bar.png                     — sorted ceiling bar chart (D)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
#  Reuse the existing engine call pattern, timeout helper, and circuit families
# ──────────────────────────────────────────────────────────────────────────────
from test_scaling import _require_fpga, _time_limit, _Timeout, CIRCUIT_FAMILIES
from extra_circuits import EXTRA_FAMILIES

ALL_FAMILIES: Dict[str, Tuple] = {}
ALL_FAMILIES.update(CIRCUIT_FAMILIES)
ALL_FAMILIES.update(EXTRA_FAMILIES)

COARSE_SIZES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]


# ──────────────────────────────────────────────────────────────────────────────
#  Attempt record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Attempt:
    family: str
    nq: int
    phase: str          # 'coarse' | 'bisect'
    elapsed: Optional[float]
    bond: Optional[int]
    throughput: Optional[float]
    status: str          # 'ok' | 'timeout' | 'hard_fail'
    mode: str
    error: str

    def row(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
#  Growth sequence: coarse list, then doubling, capped at --max-cap
# ──────────────────────────────────────────────────────────────────────────────

def _growth_sizes(max_cap: int):
    sizes = [s for s in COARSE_SIZES if s <= max_cap]
    if not sizes:
        yield max_cap
        return
    for s in sizes:
        yield s
    last = sizes[-1]
    while True:
        nxt = last * 2
        if nxt >= max_cap:
            if last < max_cap:
                yield max_cap
            return
        yield nxt
        last = nxt


# ──────────────────────────────────────────────────────────────────────────────
#  Simulator holder — survives a wedged worker by hard-killing and rebuilding
# ──────────────────────────────────────────────────────────────────────────────

class _SimHolder:
    """
    Mutable holder around one FPGAMPSSimulator so a wedged instance can be
    swapped out for a fresh one without threading a rebound variable through
    every call site.

    Why this exists: when a run is abandoned at the --timeout deadline, the
    FPGA card may still be genuinely executing that job in hardware -- there
    is no host-side kernel-cancel API. The library's own shutdown()/
    _teardown() sends SIGTERM and does bounded joins, but a worker blocked
    inside an uninterruptible XRT/DMA wait won't act on SIGTERM (or even
    SIGKILL) until that underlying syscall returns, which can take far
    longer than a few seconds. Observed in practice: a plain shutdown() call
    after a timeout hung the whole benchmark for 15+ minutes.

    recover() bounds this: SIGKILL every worker PID directly, wait up to
    --recovery-grace seconds for them to actually exit, then build a brand
    new simulator regardless (abandoning any still-wedged process rather
    than trying to reuse it). This trades a small amount of correctness risk
    (an abandoned worker could in the rarest case still be touching shared
    HBM/PCIe state) for a hard bound on how long recovery can take.
    """

    def __init__(self, factory, recovery_grace_s: int = 120):
        self._factory = factory
        self._grace = recovery_grace_s
        self.sim = factory()

    def recover(self):
        old = self.sim
        workers = list(getattr(old, '_workers', []) or [])
        for w in workers:
            if w.pid:
                try:
                    os.kill(w.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception:
                    pass

        deadline = time.time() + self._grace
        for w in workers:
            w.join(timeout=max(0.0, deadline - time.time()))
        stuck = [w.pid for w in workers if w.is_alive()]
        if stuck:
            print(f"  [WARN] recovery: worker PIDs {stuck} still alive after "
                  f"{self._grace}s grace period -- abandoning them and "
                  f"building a fresh simulator instance")
            sys.stdout.flush()

        try:
            old.shutdown()
        except Exception:
            pass

        self.sim = self._factory()


# ──────────────────────────────────────────────────────────────────────────────
#  Single-run attempt — identical call pattern to test_scaling.run_benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_attempt(holder: _SimHolder, family: str, builder, nq: int, shots: int,
                 timeout_s: int, seed: int, phase: str) -> Attempt:
    elapsed = None
    bond = None
    status = 'ok'
    mode_str = ''
    error = ''

    try:
        with _time_limit(timeout_s):
            qc = builder(nq)
            t0 = time.perf_counter()
            res = holder.sim.run(qc, shots=shots, seed=seed)
            elapsed = time.perf_counter() - t0

        mode_str = res.mode
        if not mode_str.endswith('_fpga'):
            raise AssertionError(
                f"mode={mode_str!r} — result did NOT come from the FPGA kernel"
            )
        bond = res.get_max_bond()

    except _Timeout:
        elapsed = float(timeout_s)
        status = 'timeout'
        error = f'exceeded {timeout_s}s per-run timeout'
    except Exception as e:
        status = 'hard_fail'
        error = f'{type(e).__name__}: {e}'

    if status != 'ok':
        # See _SimHolder.recover(): abandon the (possibly wedged) worker
        # processes and rebuild, bounded by --recovery-grace, so the next
        # attempt starts from a known-good simulator instance.
        holder.recover()

    throughput = (nq / elapsed) if (status == 'ok' and elapsed) else None
    return Attempt(family, nq, phase, elapsed, bond, throughput, status, mode_str, error)


# ──────────────────────────────────────────────────────────────────────────────
#  Per-family ceiling discovery
# ──────────────────────────────────────────────────────────────────────────────

def discover_family(holder: _SimHolder, family: str, builder, shots: int, timeout_s: int,
                     max_cap: int, seed: int, bisect: bool, tol: int,
                     progress_cb) -> Tuple[List[Attempt], dict]:
    attempts: List[Attempt] = []
    last_ok: Optional[int] = None
    first_fail: Optional[int] = None
    fail_reason = ''

    for nq in _growth_sizes(max_cap):
        a = run_attempt(holder, family, builder, nq, shots, timeout_s, seed, 'coarse')
        attempts.append(a)
        progress_cb(a)
        if a.status == 'ok':
            last_ok = nq
        else:
            first_fail = nq
            fail_reason = a.error
            break
    else:
        fail_reason = f'reached safety cap ({max_cap} qubits) with no failure'

    if bisect and last_ok is not None and first_fail is not None \
            and (first_fail - last_ok) > tol:
        lo, hi = last_ok, first_fail
        while hi - lo > tol:
            mid = (lo + hi) // 2
            if mid <= lo or mid >= hi:
                break
            a = run_attempt(holder, family, builder, mid, shots, timeout_s, seed, 'bisect')
            attempts.append(a)
            progress_cb(a)
            if a.status == 'ok':
                lo = mid
            else:
                hi = mid
        last_ok = lo
        first_fail = hi

    ceiling = {
        'max_ok_qubits': last_ok if last_ok is not None else 0,
        'first_fail_qubits': first_fail,
        'reason': fail_reason,
    }
    return attempts, ceiling


# ──────────────────────────────────────────────────────────────────────────────
#  Console progress table
# ──────────────────────────────────────────────────────────────────────────────

_HDR = f"  {'nq':>6}  {'family':<16}  {'phase':<7}  {'time(s)':>9}  {'bond':>6}  {'mode':<16}  status"
_SEP = "  " + "-" * (len(_HDR) - 2)


def _print_header():
    print(f"\n{_SEP}")
    print(_HDR)
    print(_SEP)


def _print_row(a: Attempt):
    t_str = f"{a.elapsed:9.3f}" if a.elapsed is not None else f"{'':>9}"
    b_str = f"{a.bond:6d}" if a.bond is not None else f"{'':>6}"
    print(f"  {a.nq:>6}  {a.family:<16}  {a.phase:<7}  {t_str}  {b_str}  "
          f"{a.mode:<16}  {a.status}"
          + (f"  ({a.error})" if a.error else ""))
    sys.stdout.flush()


# ──────────────────────────────────────────────────────────────────────────────
#  Color / marker palette — consistent across every plot
# ──────────────────────────────────────────────────────────────────────────────

_MARKERS = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', 'h', '<', '>']


def _build_palette(families: List[str]) -> Dict[str, dict]:
    cmap = plt.get_cmap('tab10') if len(families) <= 10 else plt.get_cmap('tab20')
    palette = {}
    for i, fam in enumerate(families):
        palette[fam] = {
            'color': cmap(i % cmap.N),
            'marker': _MARKERS[i % len(_MARKERS)],
        }
    return palette


# ──────────────────────────────────────────────────────────────────────────────
#  Visualization (A): individual time-vs-qubits per family
# ──────────────────────────────────────────────────────────────────────────────

def plot_family_time(family: str, attempts: List[Attempt], ceiling: dict,
                      chi_max: int, timeout_s: int, palette: dict, out_dir: str):
    style = palette[family]
    ok = sorted([a for a in attempts if a.status == 'ok'], key=lambda a: a.nq)
    if not ok:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    xs = [a.nq for a in ok]
    ys = [a.elapsed for a in ok]
    ax.plot(xs, ys, color=style['color'], marker=style['marker'],
            linewidth=1.8, markersize=6, label=family)

    ax.axhline(timeout_s, color='red', ls='--', linewidth=1.2,
               label=f'timeout = {timeout_s}s')

    max_ok = ceiling['max_ok_qubits']
    max_ok_attempt = next((a for a in ok if a.nq == max_ok), ok[-1])
    ax.scatter([max_ok_attempt.nq], [max_ok_attempt.elapsed], s=140,
               facecolors='none', edgecolors=style['color'], linewidths=2, zorder=5)
    ax.annotate(f"ceiling: {max_ok} qubits\n({max_ok_attempt.elapsed:.3f}s)",
                xy=(max_ok_attempt.nq, max_ok_attempt.elapsed),
                xytext=(10, 10), textcoords='offset points', fontsize=8,
                bbox=dict(boxstyle='round', fc='white', ec=style['color'], alpha=0.9))

    first_fail = ceiling['first_fail_qubits']
    caption = ceiling['reason']
    if first_fail is not None:
        fail_attempt = next((a for a in attempts
                              if a.nq == first_fail and a.status != 'ok'), None)
        fail_y = fail_attempt.elapsed if (fail_attempt and fail_attempt.elapsed) else timeout_s
        ax.scatter([first_fail], [fail_y], marker='x', s=160, color='black',
                   linewidths=2.5, zorder=6, label='first failure')
        caption = f"first failure @ {first_fail} qubits: {ceiling['reason']}"

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title(f"{family} — max = {max_ok} qubits", fontsize=11, fontweight='bold')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=8, loc='upper left')

    fig.text(0.5, -0.02, caption, ha='center', fontsize=8,
              bbox=dict(boxstyle='round', fc='#f5f5f5', ec='gray', alpha=0.9))

    plt.tight_layout()
    path = os.path.join(out_dir, f"{family}_time.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
#  Visualization (B): individual 3-panel metrics per family
# ──────────────────────────────────────────────────────────────────────────────

def plot_family_metrics(family: str, attempts: List[Attempt], ceiling: dict,
                         chi_max: int, palette: dict, out_dir: str):
    style = palette[family]
    ok = sorted([a for a in attempts if a.status == 'ok'], key=lambda a: a.nq)
    if not ok:
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle(
        f"FPGA MPS Simulator — {family} | Xilinx Alveo U55C | "
        f"chi_max={chi_max} | Author: Nasir Ali, C-DAC Noida",
        fontsize=11, y=1.03,
    )

    xs = [a.nq for a in ok]

    ax1 = axes[0]
    ax1.plot(xs, [a.elapsed for a in ok], color=style['color'],
              marker=style['marker'], linewidth=1.8, markersize=6, label=family)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel('Number of Qubits'); ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Time vs Qubits', fontsize=10, fontweight='bold')
    ax1.grid(True, which='both', ls='--', alpha=0.4)
    ax1.legend(fontsize=8, loc='upper left')

    ax2 = axes[1]
    bond_ok = [a for a in ok if a.bond is not None and a.bond > 0]
    if bond_ok:
        ax2.plot([a.nq for a in bond_ok], [a.bond for a in bond_ok],
                  color=style['color'], marker=style['marker'],
                  linewidth=1.8, markersize=6, label=family)
    ax2.axhline(chi_max, color='red', ls='--', linewidth=1.2,
                label=f'chi_max = {chi_max}')
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of Qubits'); ax2.set_ylabel('Max Bond Dimension')
    ax2.set_title('Max Bond Dimension vs Qubits', fontsize=10, fontweight='bold')
    ax2.grid(True, which='both', ls='--', alpha=0.4)
    ax2.legend(fontsize=8, loc='upper left')

    ax3 = axes[2]
    thr_ok = [a for a in ok if a.throughput]
    if thr_ok:
        ax3.plot([a.nq for a in thr_ok], [a.throughput for a in thr_ok],
                  color=style['color'], marker=style['marker'],
                  linewidth=1.8, markersize=6, label=family)
    ax3.set_xscale('log'); ax3.set_yscale('log')
    ax3.set_xlabel('Number of Qubits'); ax3.set_ylabel('Throughput (qubits/s)')
    ax3.set_title('Throughput vs Qubits', fontsize=10, fontweight='bold')
    ax3.grid(True, which='both', ls='--', alpha=0.4)
    ax3.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    path = os.path.join(out_dir, f"{family}_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
#  Visualization (C): combined overlays across all families
# ──────────────────────────────────────────────────────────────────────────────

def plot_combined(all_attempts: List[Attempt], ceilings: Dict[str, dict],
                   families: List[str], chi_max: int, out_dir: str, palette: dict):

    def _series(fam, key):
        recs = sorted([a for a in all_attempts if a.family == fam and a.status == 'ok'
                        and getattr(a, key) is not None], key=lambda a: a.nq)
        return [a.nq for a in recs], [getattr(a, key) for a in recs]

    # ---- combined_time.png ----
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for fam in families:
        xs, ys = _series(fam, 'elapsed')
        if not xs:
            continue
        style = palette[fam]
        ax.plot(xs, ys, color=style['color'], marker=style['marker'],
                linewidth=1.6, markersize=5, label=fam)
        ceil = ceilings.get(fam, {})
        max_ok = ceil.get('max_ok_qubits')
        if max_ok and max_ok in xs:
            yv = ys[xs.index(max_ok)]
            ax.scatter([max_ok], [yv], s=110, facecolors='none',
                       edgecolors=style['color'], linewidths=2, zorder=5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Number of Qubits'); ax.set_ylabel('Execution Time (s)')
    ax.set_title('FPGA MPS Simulator — Execution Time vs Qubits (all families)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=8, loc='upper left', ncol=2, title='test')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'combined_time.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- combined_bond.png ----
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for fam in families:
        xs, ys = _series(fam, 'bond')
        if not xs:
            continue
        style = palette[fam]
        ax.plot(xs, ys, color=style['color'], marker=style['marker'],
                linewidth=1.6, markersize=5, label=fam)
    ax.axhline(chi_max, color='red', ls='--', linewidth=1.2, label=f'chi_max = {chi_max}')
    ax.set_xscale('log')
    ax.set_xlabel('Number of Qubits'); ax.set_ylabel('Max Bond Dimension')
    ax.set_title('FPGA MPS Simulator — Max Bond Dimension vs Qubits (all families)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=8, loc='upper left', ncol=2, title='test')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'combined_bond.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- combined_throughput.png ----
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for fam in families:
        xs, ys = _series(fam, 'throughput')
        if not xs:
            continue
        style = palette[fam]
        ax.plot(xs, ys, color=style['color'], marker=style['marker'],
                linewidth=1.6, markersize=5, label=fam)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Number of Qubits'); ax.set_ylabel('Throughput (qubits/s)')
    ax.set_title('FPGA MPS Simulator — Throughput vs Qubits (all families)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=8, loc='upper left', ncol=2, title='test')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'combined_throughput.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
#  Visualization (D): ceilings bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_ceilings_bar(ceilings: Dict[str, dict], families: List[str],
                       palette: dict, out_dir: str):
    ranked = sorted(families, key=lambda f: ceilings.get(f, {}).get('max_ok_qubits', 0))
    values = [ceilings.get(f, {}).get('max_ok_qubits', 0) for f in ranked]
    colors = [palette[f]['color'] for f in ranked]

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.55 * len(ranked))))
    bars = ax.barh(ranked, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() * 1.01 + max(values) * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{v}", va='center', fontsize=9)

    ax.set_xlabel('Max OK Qubits (ceiling)')
    ax.set_title('FPGA MPS Simulator — Per-Family Qubit Ceiling\n'
                 'Xilinx Alveo U55C | Author: Nasir Ali, C-DAC Noida',
                 fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ceilings_bar.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
#  Data outputs
# ──────────────────────────────────────────────────────────────────────────────

def write_outputs(all_attempts: List[Attempt], ceilings: Dict[str, dict], out_dir: str):
    csv_path = os.path.join(out_dir, 'summary.csv')
    json_path = os.path.join(out_dir, 'summary.json')
    ceilings_path = os.path.join(out_dir, 'ceilings.json')

    fieldnames = ['family', 'nq', 'phase', 'elapsed', 'bond', 'throughput',
                  'status', 'mode', 'error']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for a in all_attempts:
            w.writerow(a.row())

    with open(json_path, 'w') as f:
        json.dump([a.row() for a in all_attempts], f, indent=2)

    with open(ceilings_path, 'w') as f:
        json.dump(ceilings, f, indent=2)

    return csv_path, json_path, ceilings_path


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def _main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-family qubit-ceiling discovery for the FPGA MPS simulator"
    )
    ap.add_argument("--families", nargs="+", default=list(ALL_FAMILIES.keys()),
                     choices=list(ALL_FAMILIES.keys()),
                     help="circuit families to discover ceilings for")
    ap.add_argument("--chi-max", type=int, default=64, dest="chi_max",
                     help="max bond dimension (default 64)")
    ap.add_argument("--shots", type=int, default=1024,
                     help="measurement shots per run (default 1024)")
    ap.add_argument("--timeout", type=int, default=60,
                     help="per-run timeout in seconds (default 60)")
    ap.add_argument("--max-cap", type=int, default=8192, dest="max_cap",
                     help="global safety cap on qubit count (default 8192)")
    ap.add_argument("--bisect", dest="bisect", action="store_true", default=True,
                     help="bisection-refine the ceiling (default on)")
    ap.add_argument("--no-bisect", dest="bisect", action="store_false",
                     help="disable bisection refinement")
    ap.add_argument("--tol", type=int, default=100,
                     help="bisection tolerance in qubits (default 100)")
    ap.add_argument("--out", default="results",
                     help="output directory (default results)")
    ap.add_argument("--cards", type=int, default=4, choices=(1, 2, 4),
                     help="number of FPGA cards to use (default 4)")
    ap.add_argument("--recovery-grace", type=int, default=120, dest="recovery_grace",
                     help="max seconds to wait for a wedged worker to die "
                          "after a timeout/hard_fail before rebuilding the "
                          "simulator from scratch (default 120)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("\n" + "=" * 78)
    print("  FPGA MPS Simulator — Per-Family Qubit Ceiling Discovery")
    print("  Author: Nasir Ali  |  C-DAC Noida")
    print("=" * 78)

    print("\n  Verifying FPGA hardware ...")
    _require_fpga()

    os.makedirs(args.out, exist_ok=True)

    from fpga_mps_simulator import FPGAMPSSimulator

    def _make_sim():
        s = FPGAMPSSimulator(
            num_cards=args.cards,
            chi_max=args.chi_max,
            svd_cutoff=1e-12,
            engine="fpga",
            mode="shots",
        )
        assert s._engine == "fpga", (
            f"Simulator resolved to engine={s._engine!r}. "
            "No CPU fallback is permitted for this benchmark."
        )
        return s

    holder = _SimHolder(_make_sim, recovery_grace_s=args.recovery_grace)
    print(f"\n  Simulator engine : {holder.sim._engine.upper()}  (no CPU fallback)")
    print(f"  chi_max          : {args.chi_max}")
    print(f"  shots per run    : {args.shots}")
    print(f"  per-run timeout  : {args.timeout}s")
    print(f"  global max-cap   : {args.max_cap} qubits")
    print(f"  bisection        : {'on (tol=' + str(args.tol) + ')' if args.bisect else 'off'}")
    print(f"  recovery grace   : {args.recovery_grace}s")
    print(f"  families         : {args.families}")

    all_attempts: List[Attempt] = []
    ceilings: Dict[str, dict] = {}

    _print_header()
    for family in args.families:
        builder, desc = ALL_FAMILIES[family]
        print(f"\n  >>> {family}: {desc}")
        try:
            attempts, ceiling = discover_family(
                holder, family, builder, args.shots, args.timeout, args.max_cap,
                args.seed, args.bisect, args.tol, _print_row,
            )
        except Exception as e:
            print(f"  [FAMILY DRIVER FAILED] {family}: {type(e).__name__}: {e}")
            traceback.print_exc()
            attempts = []
            ceiling = {'max_ok_qubits': 0, 'first_fail_qubits': None,
                       'reason': f'family driver crashed: {type(e).__name__}: {e}'}
        all_attempts.extend(attempts)
        ceilings[family] = ceiling
        print(f"  <<< {family} ceiling: max_ok={ceiling['max_ok_qubits']}  "
              f"first_fail={ceiling['first_fail_qubits']}  reason={ceiling['reason']}")

    print(_SEP)

    try:
        holder.sim.shutdown()
    except Exception:
        pass

    palette = _build_palette(args.families)

    for family in args.families:
        fam_attempts = [a for a in all_attempts if a.family == family]
        if not fam_attempts:
            continue
        plot_family_time(family, fam_attempts, ceilings[family], args.chi_max,
                          args.timeout, palette, args.out)
        plot_family_metrics(family, fam_attempts, ceilings[family], args.chi_max,
                             palette, args.out)

    plot_combined(all_attempts, ceilings, args.families, args.chi_max, args.out, palette)
    plot_ceilings_bar(ceilings, args.families, palette, args.out)

    csv_path, json_path, ceilings_path = write_outputs(all_attempts, ceilings, args.out)

    print("\n  ── Outputs ─────────────────────────────────────────────────")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {ceilings_path}")
    print(f"  {args.out}/<family>_time.png, <family>_metrics.png  (per family)")
    print(f"  {args.out}/combined_time.png, combined_bond.png, combined_throughput.png")
    print(f"  {args.out}/ceilings_bar.png")

    print("\n  ── Per-Family Ceilings ─────────────────────────────────────")
    hdr = f"  {'family':<18}  {'max_ok_qubits':>14}  {'first_fail':>11}  reason"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for family in args.families:
        c = ceilings[family]
        ff = c['first_fail_qubits'] if c['first_fail_qubits'] is not None else '-'
        print(f"  {family:<18}  {c['max_ok_qubits']:>14}  {str(ff):>11}  {c['reason']}")
    print()

    any_ok = any(a.status == 'ok' for a in all_attempts)
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(_main())
