"""
benchmark.py — Performance benchmarks for FPGA MPS Quantum Circuit Simulator

Compares:
  - fpga_mps_simulator (CPU mode, MPS)
  - qiskit-aer matrix_product_state (if available)
  - qiskit-aer statevector_simulator (up to n ≤ 24)

Outputs a formatted results table.

Author: Nasir Ali
Organization: C-DAC Noida
Date: June 2026
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cpu_reference import CPUMPSSimulator

try:
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("WARNING: Qiskit not installed. Some benchmarks will be skipped.")

try:
    from qiskit_aer import AerSimulator
    AER_AVAILABLE = True
except ImportError:
    try:
        from qiskit.providers.aer import AerSimulator
        AER_AVAILABLE = True
    except ImportError:
        AER_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  Circuit factories
# ─────────────────────────────────────────────────────────────────────────────

def make_ghz(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def make_random_layer(n, depth, seed=42):
    """Random circuit: alternating H/RZ and CX layers."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for d in range(depth):
        for i in range(n):
            qc.rz(float(rng.uniform(0, 2 * np.pi)), i)
            qc.h(i)
        for i in range(0, n - 1, 2):
            qc.cx(i, i + 1)
        for i in range(1, n - 1, 2):
            qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def make_qft(n):
    """Quantum Fourier Transform (linear chain routing)."""
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, min(i + 5, n)):  # limited range for MPS efficiency
            angle = np.pi / (2 ** (j - i))
            qc.cp(angle, j, i)
    qc.measure_all()
    return qc


# ─────────────────────────────────────────────────────────────────────────────
#  Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def run_mps(circuit, shots, chi_max, seed=0):
    sim = CPUMPSSimulator(chi_max=chi_max, seed=seed)
    t0 = time.perf_counter()
    mps, counts = sim.run_mps(circuit, shots)
    dt = time.perf_counter() - t0
    return dt, counts, mps.max_bond()


def run_aer_mps(circuit, shots, seed=0):
    if not AER_AVAILABLE:
        return None, None, None
    try:
        backend = AerSimulator(method='matrix_product_state')
        t0 = time.perf_counter()
        job = backend.run(circuit, shots=shots, seed_simulator=seed)
        result = job.result()
        dt = time.perf_counter() - t0
        counts = dict(result.get_counts())
        return dt, counts, None
    except Exception as e:
        print(f"  Aer MPS error: {e}")
        return None, None, None


def run_aer_sv(circuit, shots, seed=0):
    if not AER_AVAILABLE:
        return None, None
    try:
        backend = AerSimulator(method='statevector')
        t0 = time.perf_counter()
        job = backend.run(circuit, shots=shots, seed_simulator=seed)
        result = job.result()
        dt = time.perf_counter() - t0
        counts = dict(result.get_counts())
        return dt, counts
    except Exception as e:
        print(f"  Aer SV error: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  Main benchmark suite
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 90)
    print(f"{'FPGA MPS Simulator — Performance Benchmark':^90}")
    print(f"{'(CPU mode: fpga_mps_simulator vs qiskit-aer)':^90}")
    print("=" * 90)
    print()

    SHOTS = 1000
    RESULTS = []

    # ── GHZ benchmark ────────────────────────────────────────────────────────
    print("1. GHZ State (nearest-neighbor, low entanglement)")
    print("-" * 90)
    print(f"{'n':>6}  {'chi_max':>8}  {'MPS time(s)':>12}  {'max_bond':>10}  "
          f"{'Aer MPS(s)':>12}  {'Aer SV(s)':>10}  {'speedup':>8}")
    print("-" * 90)

    for n, chi_max in [(10, 4), (20, 4), (50, 4), (100, 4), (200, 4), (500, 4)]:
        if not QISKIT_AVAILABLE:
            break
        qc = make_ghz(n)
        t_mps, counts_mps, max_b = run_mps(qc, SHOTS, chi_max)

        t_aer_mps = None
        if n <= 100 and AER_AVAILABLE:
            t_aer_mps, _, _ = run_aer_mps(qc, SHOTS)

        t_aer_sv = None
        if n <= 24 and AER_AVAILABLE:
            t_aer_sv, _ = run_aer_sv(qc, SHOTS)

        speedup_str = ''
        if t_aer_mps is not None and t_mps > 0:
            speedup_str = f'{t_aer_mps / t_mps:.1f}×'

        print(f"{n:>6}  {chi_max:>8}  {t_mps:>12.3f}  {max_b:>10}  "
              f"{str(round(t_aer_mps, 3)) if t_aer_mps else 'N/A':>12}  "
              f"{str(round(t_aer_sv, 3)) if t_aer_sv else 'N/A':>10}  "
              f"{speedup_str:>8}")

        RESULTS.append({
            'circuit': f'GHZ-{n}q', 'n': n, 'chi_max': chi_max,
            't_mps': t_mps, 'max_bond': max_b, 't_aer_mps': t_aer_mps,
        })

    print()

    # ── Random circuit benchmark ─────────────────────────────────────────────
    print("2. Random Layered Circuit (depth=6, higher entanglement)")
    print("-" * 90)
    print(f"{'n':>6}  {'chi_max':>8}  {'depth':>6}  {'MPS time(s)':>12}  "
          f"{'max_bond':>10}  {'Aer MPS(s)':>12}")
    print("-" * 90)

    for n, chi_max, depth in [(10, 32, 6), (20, 64, 6), (30, 128, 6), (50, 256, 4)]:
        if not QISKIT_AVAILABLE:
            break
        qc = make_random_layer(n, depth)
        t_mps, counts_mps, max_b = run_mps(qc, SHOTS, chi_max)

        t_aer_mps = None
        if n <= 30 and AER_AVAILABLE:
            t_aer_mps, _, _ = run_aer_mps(qc, SHOTS)

        print(f"{n:>6}  {chi_max:>8}  {depth:>6}  {t_mps:>12.3f}  {max_b:>10}  "
              f"{str(round(t_aer_mps, 3)) if t_aer_mps else 'N/A':>12}")

        RESULTS.append({
            'circuit': f'Random-{n}q-d{depth}', 'n': n, 'chi_max': chi_max,
            't_mps': t_mps, 'max_bond': max_b, 't_aer_mps': t_aer_mps,
        })

    print()

    # ── QFT benchmark ────────────────────────────────────────────────────────
    print("3. Quantum Fourier Transform (limited-range CP gates)")
    print("-" * 90)
    print(f"{'n':>6}  {'chi_max':>8}  {'MPS time(s)':>12}  {'max_bond':>10}")
    print("-" * 90)

    for n, chi_max in [(10, 32), (20, 64), (50, 128)]:
        if not QISKIT_AVAILABLE:
            break
        qc = make_qft(n)
        t_mps, _, max_b = run_mps(qc, SHOTS, chi_max)
        print(f"{n:>6}  {chi_max:>8}  {t_mps:>12.3f}  {max_b:>10}")

    print()

    # ── Memory budget table ──────────────────────────────────────────────────
    print("4. HBM Memory Budget (per-card, 4-card distribution)")
    print("-" * 90)
    print(f"{'n':>6}  {'chi_max':>8}  {'Total MB':>10}  {'Per-card MB':>12}  "
          f"{'Within 8 GB HBM':>16}")
    print("-" * 90)

    for n, chi in [(50, 64), (100, 128), (200, 256), (200, 512), (500, 512)]:
        total_mb = n * chi * 2 * chi * 16 / 1e6
        per_mb = total_mb / 4
        fits = "YES" if per_mb < 8192 else "NO (too large)"
        print(f"{n:>6}  {chi:>8}  {total_mb:>10.0f}  {per_mb:>12.0f}  {fits:>16}")

    print()
    print("=" * 90)
    print("Note: 'MPS time' = cpu_reference.py (single-core Python).")
    print("FPGA execution would be ~10-100× faster for kernel-bound operations.")
    print("=" * 90)
    print()


if __name__ == '__main__':
    main()
