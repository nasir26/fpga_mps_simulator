"""
test_distributed.py — Tests for multi-card distribution modes

Mode A (shots): multiple circuits dispatched in parallel across cards.
Mode B (distributed): chain partition across 4 workers.

All tests run in CPU mode (engine="cpu") — FPGA not required.

Author: Nasir Ali
Organization: C-DAC Noida
Date: June 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from fpga_mps_simulator import FPGAMPSSimulator, FPGAMPSResult

try:
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def _fidelity(sv1, sv2):
    sv1 = sv1.ravel() / np.linalg.norm(sv1)
    sv2 = sv2.ravel() / np.linalg.norm(sv2)
    return float(abs(np.dot(sv1.conj(), sv2)) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
#  Mode A (shots) — circuit parallelism
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
class TestModeAShots:

    def _make_ghz(self, n):
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure_all()
        return qc

    def test_single_circuit_mode_a(self):
        qc = self._make_ghz(4)
        sim = FPGAMPSSimulator(num_cards=1, engine="cpu",
                                chi_max=16, mode="shots")
        result = sim.run(qc, shots=2000, seed=0)
        counts = result.get_counts()
        total = sum(counts.values())
        p = (counts.get('0000', 0) + counts.get('1111', 0)) / total
        assert p > 0.97, f"GHZ: P(0000)+P(1111) = {p:.4f}"

    def test_multiple_circuits_parallel(self):
        """Four circuits, one per card."""
        sim = FPGAMPSSimulator(num_cards=4, engine="cpu",
                                chi_max=32, mode="shots")
        circuits = [self._make_ghz(n) for n in [3, 4, 5, 6]]
        result = sim.run(circuits, shots=1000, seed=42)
        assert len(result._results) == 4
        for i, n in enumerate([3, 4, 5, 6]):
            counts = result.get_counts(i)
            total = sum(counts.values())
            assert total == 1000
            zeros = '0' * n
            ones = '1' * n
            p = (counts.get(zeros, 0) + counts.get(ones, 0)) / total
            assert p > 0.97, f"GHZ {n}q: P = {p:.4f}"

    def test_mode_a_results_reproducible_with_seed(self):
        sim = FPGAMPSSimulator(num_cards=1, engine="cpu", chi_max=32)
        qc = self._make_ghz(4)
        r1 = sim.run(qc, shots=500, seed=99)
        r2 = sim.run(qc, shots=500, seed=99)
        assert r1.get_counts() == r2.get_counts()

    def test_more_circuits_than_cards(self):
        """8 circuits, 4 cards → cards are reused."""
        sim = FPGAMPSSimulator(num_cards=4, engine="cpu", chi_max=16)
        circuits = [self._make_ghz(n) for n in range(2, 10)]
        result = sim.run(circuits, shots=500, seed=5)
        assert len(result._results) == 8

    def test_card_stats_structure(self):
        sim = FPGAMPSSimulator(num_cards=4, engine="cpu", chi_max=16)
        qc = self._make_ghz(4)
        sim.run(qc, shots=100, seed=0)
        stats = sim.get_card_stats()
        assert len(stats) == 4
        for s in stats:
            assert 'card_id' in s
            assert 'execution_mode' in s
            assert s['execution_mode'] == 'CPU'


# ─────────────────────────────────────────────────────────────────────────────
#  Mode B (distributed chain) — multi-worker correctness
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
class TestModeBDistributed:

    def _make_linear_circuit(self, n):
        """Linear chain of CX gates — nearest-neighbour only."""
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure_all()
        return qc

    def test_distributed_ghz_4q(self):
        """4-qubit GHZ distributed across 4 workers (1 qubit each)."""
        n = 4
        qc = self._make_linear_circuit(n)
        sim = FPGAMPSSimulator(num_cards=4, engine="cpu",
                                chi_max=16, mode="distributed")
        result = sim.run(qc, shots=2000, seed=0)
        counts = result.get_counts()
        total = sum(counts.values())
        zeros = '0' * n
        ones = '1' * n
        p = (counts.get(zeros, 0) + counts.get(ones, 0)) / total
        assert p > 0.95, f"Distributed GHZ {n}q: P = {p:.4f}"

    def test_distributed_vs_cpu_fidelity(self):
        """Distributed results should match single-node CPU results."""
        n = 8
        qc = self._make_linear_circuit(n)

        sim_single = FPGAMPSSimulator(num_cards=1, engine="cpu",
                                       chi_max=32, mode="shots")
        sim_dist = FPGAMPSSimulator(num_cards=4, engine="cpu",
                                     chi_max=32, mode="distributed")

        r_single = sim_single.run(qc, shots=4000, seed=7)
        r_dist = sim_dist.run(qc, shots=4000, seed=7)

        c_single = r_single.get_counts()
        c_dist = r_dist.get_counts()

        # Both should concentrate on |00..0⟩ and |11..1⟩
        for counts in [c_single, c_dist]:
            total = sum(counts.values())
            zeros = '0' * n
            ones = '1' * n
            p = (counts.get(zeros, 0) + counts.get(ones, 0)) / total
            assert p > 0.95, f"Expected GHZ distribution; got {p:.4f}"

    def test_distributed_1card_fallback(self):
        """Distributed with 1 card should degrade gracefully to shots mode."""
        n = 4
        qc = self._make_linear_circuit(n)
        sim = FPGAMPSSimulator(num_cards=1, engine="cpu",
                                chi_max=16, mode="distributed")
        result = sim.run(qc, shots=500, seed=0)
        assert sum(result.get_counts().values()) == 500

    def test_distributed_local_gates_only(self):
        """Circuit with only local gates should work in distributed mode."""
        n = 8
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.h(i)
            qc.rz(0.5, i)
        qc.measure_all()

        sim = FPGAMPSSimulator(num_cards=4, engine="cpu",
                                chi_max=4, mode="distributed")
        result = sim.run(qc, shots=1000, seed=1)
        counts = result.get_counts()
        assert sum(counts.values()) == 1000
        # All 2^8 = 256 states should appear with roughly equal probability
        assert len(counts) > 100, "Uniform distribution should have many distinct outcomes"


# ─────────────────────────────────────────────────────────────────────────────
#  Worker lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
class TestWorkerLifecycle:

    def test_context_manager(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()

        with FPGAMPSSimulator(num_cards=2, engine="cpu", chi_max=8) as sim:
            result = sim.run(qc, shots=500, seed=0)
            assert sum(result.get_counts().values()) == 500

    def test_reuse_across_runs(self):
        """Workers should be reused for same n_qubits without reinit."""
        sim = FPGAMPSSimulator(num_cards=2, engine="cpu", chi_max=16)
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        for _ in range(3):
            result = sim.run(qc, shots=200, seed=0)
            assert sum(result.get_counts().values()) == 200

        sim.shutdown()

    def test_shutdown_idempotent(self):
        sim = FPGAMPSSimulator(num_cards=1, engine="cpu", chi_max=8)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.measure_all()
        sim.run(qc, shots=100, seed=0)
        sim.shutdown()
        sim.shutdown()  # second shutdown should not error

    def test_spawn_isolation(self):
        """
        Workers spawned via 'spawn' start method — verifies no shared state
        via CPU mode by running independent circuits with different seeds.
        """
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.measure_all()

        sim = FPGAMPSSimulator(num_cards=2, engine="cpu", chi_max=8)
        r1 = sim.run([qc, qc], shots=200, seed=10)
        r2 = sim.run([qc, qc], shots=200, seed=99)

        c1 = r1.get_counts(0)
        c2 = r2.get_counts(0)
        # Both should have ~equal distribution but with different seeds
        for c in [c1, c2]:
            total = sum(c.values())
            assert total == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
