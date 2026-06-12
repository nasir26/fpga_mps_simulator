"""
test_scaling.py — Performance and scaling tests for MPS simulator

Tests memory scaling, runtime scaling, and chi_max impact.
All tests run in CPU mode.

Author: Nasir Ali
Organization: C-DAC Noida
Date: June 2026
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from cpu_reference import MPSState, CPUMPSSimulator, GATE_MATRICES

try:
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  Memory budget: bond dimension vs n qubits
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryScaling:

    def _mem_bytes(self, n: int, chi_max: int) -> int:
        """Upper bound on MPS memory in bytes: n × chi_max × 2 × chi_max × 16."""
        return n * chi_max * 2 * chi_max * 16  # complex128

    @pytest.mark.parametrize("n,chi_max,limit_mb", [
        (50,  64,  200),
        (100, 128, 500),
        (200, 256, 4000),
        (500, 512, 40000),
    ])
    def test_memory_budget(self, n, chi_max, limit_mb):
        mem = self._mem_bytes(n, chi_max)
        assert mem <= limit_mb * 1024 * 1024, \
            f"n={n}, chi_max={chi_max}: {mem/1e6:.0f} MB > {limit_mb} MB"

    def test_bond_dims_shape(self):
        """bond_dims() should return n-1 values."""
        mps = MPSState(10, chi_max=8)
        mps.apply_1q(0, GATE_MATRICES['h'])
        for i in range(9):
            mps.apply_2q(i, i + 1, GATE_MATRICES['cx'])
        bd = mps.bond_dims()
        assert len(bd) == 9, f"Expected 9 bond dims, got {len(bd)}"

    def test_product_state_constant_memory(self):
        """Product state tensor sizes should be O(n), not O(chi²)."""
        for n in [10, 50, 100]:
            mps = MPSState(n, chi_max=64)
            for i in range(n):
                mps.apply_1q(i, GATE_MATRICES['h'])
            # All bond dims should be 1
            total_amps = sum(
                mps.sites[i].size for i in range(n)
            )
            # Each site: (1, 2, 1) → 2 amps; total = 2n
            assert total_amps == 2 * n, \
                f"n={n}: product state has {total_amps} amps, expected {2*n}"


# ─────────────────────────────────────────────────────────────────────────────
#  Runtime scaling: sampling should be O(n × chi²) per shot
# ─────────────────────────────────────────────────────────────────────────────

class TestRuntimeScaling:

    @pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
    @pytest.mark.parametrize("n", [10, 20, 50, 100])
    def test_ghz_scales_linearly(self, n):
        """GHZ state on n qubits: sampling 100 shots must finish < 60s."""
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure_all()

        sim = CPUMPSSimulator(chi_max=4, seed=0)
        t0 = time.time()
        mps, counts = sim.run_mps(qc, shots=100)
        dt = time.time() - t0
        assert dt < 60, f"GHZ {n}q, 100 shots: {dt:.1f}s (limit 60s)"
        assert sum(counts.values()) == 100

    @pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
    def test_product_state_scaling(self):
        """Product state (chi=1): n=500 sampling should be fast."""
        n = 500
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.h(i)
        qc.measure_all()

        sim = CPUMPSSimulator(chi_max=1, seed=0)
        t0 = time.time()
        _, counts = sim.run_mps(qc, shots=100)
        dt = time.time() - t0
        assert dt < 30, f"Product {n}q, 100 shots: {dt:.1f}s (limit 30s)"


# ─────────────────────────────────────────────────────────────────────────────
#  Chi-max impact on accuracy
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
class TestChiMaxImpact:

    def _fidelity(self, sv1, sv2):
        sv1 = sv1.ravel() / np.linalg.norm(sv1)
        sv2 = sv2.ravel() / np.linalg.norm(sv2)
        return float(abs(np.dot(sv1.conj(), sv2)) ** 2)

    def test_larger_chi_higher_fidelity(self):
        """For a highly entangled circuit, larger chi_max → higher fidelity."""
        from qiskit.quantum_info import Statevector as QkSV

        n = 6
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.h(i)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        for i in range(n):
            qc.rz(0.3 * i, i)

        sv_exact = QkSV(qc).data
        fidelities = {}
        for chi in [2, 4, 8, 32]:
            sim = CPUMPSSimulator(chi_max=chi, seed=0)
            mps, _ = sim.run_mps(qc, shots=0)
            sv_mps = mps.get_statevector()
            fidelities[chi] = self._fidelity(sv_exact, sv_mps)

        # Fidelity should increase with chi_max
        for i, chi in enumerate([2, 4, 8]):
            assert fidelities[chi] <= fidelities[32] + 0.01, \
                f"chi={chi} fidelity {fidelities[chi]:.4f} > chi=32 fidelity {fidelities[32]:.4f}"

        # chi=32 should be essentially exact for n=6
        assert fidelities[32] > 0.999, \
            f"chi=32 fidelity on 6-qubit circuit: {fidelities[32]:.4f}"

    def test_chi_max_1_product_state_exact(self):
        """Product state with chi_max=1 should give exactly correct amplitudes."""
        from qiskit.quantum_info import Statevector as QkSV

        n = 5
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.h(i)

        sv_exact = QkSV(qc).data
        sim = CPUMPSSimulator(chi_max=1)
        mps, _ = sim.run_mps(qc, shots=0)
        sv_mps = mps.get_statevector()
        assert self._fidelity(sv_exact, sv_mps) > 0.9999


# ─────────────────────────────────────────────────────────────────────────────
#  Progressive scale stress test
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
@pytest.mark.parametrize("n,chi_max", [
    (5,   8),
    (10, 16),
    (16, 16),
    (20, 32),
    (24, 64),
    (32, 64),
])
def test_progressive_scale_ghz(n, chi_max):
    """GHZ state for increasing n: verify shot distribution and timing."""
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()

    sim = CPUMPSSimulator(chi_max=chi_max, seed=0)
    t0 = time.time()
    mps, counts = sim.run_mps(qc, shots=500)
    dt = time.time() - t0

    zeros = '0' * n
    ones = '1' * n
    total = sum(counts.values())
    p = (counts.get(zeros, 0) + counts.get(ones, 0)) / total
    assert p > 0.95, f"GHZ {n}q chi={chi_max}: P={p:.4f} (expected >0.95)"
    assert dt < 120, f"GHZ {n}q chi={chi_max}: {dt:.1f}s (limit 120s)"


# ─────────────────────────────────────────────────────────────────────────────
#  HBM memory budget table (print only, no assertion)
# ─────────────────────────────────────────────────────────────────────────────

def test_print_memory_budget_table():
    """Print the HBM memory budget table (informational)."""
    print("\n" + "=" * 70)
    print(f"{'HBM Memory Budget Table (per card, 4-card split)':^70}")
    print("=" * 70)
    print(f"{'n qubits':>12}  {'chi_max':>8}  {'Total MB':>10}  {'Per card MB':>12}")
    print("-" * 70)
    for n, chi in [(50, 64), (100, 128), (200, 256), (200, 512), (500, 512)]:
        total_mb = n * chi * 2 * chi * 16 / 1e6
        per_card_mb = total_mb / 4
        flag = " ⚠" if per_card_mb > 8000 else ""
        print(f"{n:>12}  {chi:>8}  {total_mb:>10.0f}  {per_card_mb:>12.0f}{flag}")
    print("=" * 70)
    print("U55C HBM capacity: ~8 GB per card (32 GB total)")
    print()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
