"""
test_correctness.py — Correctness tests for fpga_mps_simulator (CPU mode)

All tests run with engine="cpu" and are validated against Qiskit Statevector
where available (n ≤ 20), or against the cpu_reference golden model.

Author: Nasir Ali
Organization: C-DAC Noida
Date: June 2026
"""

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from fpga_mps_simulator import FPGAMPSSimulator, FPGAMPSResult
from cpu_reference import CPUMPSSimulator, MPSState, apply_gate

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

pytestmark = pytest.mark.usefixtures()

SIM = FPGAMPSSimulator(num_cards=1, engine="cpu", chi_max=256)

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fidelity(sv1: np.ndarray, sv2: np.ndarray) -> float:
    """State fidelity |<ψ1|ψ2>|²."""
    sv1 = sv1.ravel() / np.linalg.norm(sv1)
    sv2 = sv2.ravel() / np.linalg.norm(sv2)
    return float(abs(np.dot(sv1.conj(), sv2)) ** 2)


def _run_cpu(circuit, shots=4096, chi_max=256):
    sim = CPUMPSSimulator(chi_max=chi_max, svd_cutoff=1e-12, seed=42)
    mps, counts = sim.run_mps(circuit, shots)
    return mps, counts


@pytest.fixture
def sim():
    return FPGAMPSSimulator(num_cards=1, engine="cpu", chi_max=256)


# ─────────────────────────────────────────────────────────────────────────────
#  1-qubit gate correctness
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
class TestSingleQubitGates:
    GATES = ['h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'sx']
    PARAM_GATES = [('rx', [0.5]), ('ry', [1.2]), ('rz', [0.7]),
                   ('p', [1.0]), ('u', [0.3, 0.4, 0.5])]

    @pytest.mark.parametrize("gate_name", GATES)
    def test_1q_gate_fidelity(self, gate_name):
        qc = QuantumCircuit(3)
        qc.h(0)
        getattr(qc, gate_name)(1)
        qc.x(2)

        sv_qiskit = Statevector(qc).data
        mps, _ = _run_cpu(qc, shots=0)
        sv_mps = mps.get_statevector()
        assert _fidelity(sv_qiskit, sv_mps) > 0.9999, \
            f"{gate_name}: fidelity {_fidelity(sv_qiskit, sv_mps):.6f}"

    @pytest.mark.parametrize("gate_name,params", PARAM_GATES)
    def test_param_gate_fidelity(self, gate_name, params):
        qc = QuantumCircuit(2)
        qc.h(0)
        getattr(qc, gate_name)(*params, 1)

        sv_qiskit = Statevector(qc).data
        mps, _ = _run_cpu(qc, shots=0)
        sv_mps = mps.get_statevector()
        assert _fidelity(sv_qiskit, sv_mps) > 0.9999, \
            f"{gate_name}: fidelity {_fidelity(sv_qiskit, sv_mps):.6f}"


# ─────────────────────────────────────────────────────────────────────────────
#  2-qubit gate correctness
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
class TestTwoQubitGates:
    GATES_2Q = ['cx', 'cz', 'cy', 'ch', 'swap']

    @pytest.mark.parametrize("gate_name", GATES_2Q)
    def test_2q_gate_bell_fidelity(self, gate_name):
        qc = QuantumCircuit(2)
        qc.h(0)
        getattr(qc, gate_name)(0, 1)

        sv_qiskit = Statevector(qc).data
        mps, _ = _run_cpu(qc, shots=0)
        sv_mps = mps.get_statevector()
        f = _fidelity(sv_qiskit, sv_mps)
        assert f > 0.9999, f"{gate_name}: fidelity {f:.6f}"

    def test_cx_non_adjacent(self):
        """CX between non-adjacent qubits; must be SWAP-routed."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 3)  # non-adjacent

        sv_qiskit = Statevector(qc).data
        mps, _ = _run_cpu(qc, shots=0)
        sv_mps = mps.get_statevector()
        f = _fidelity(sv_qiskit, sv_mps)
        assert f > 0.99, f"cx(0,3): fidelity {f:.6f}"

    def test_rzz_entanglement(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rzz(0.5, 0, 1)

        sv_qiskit = Statevector(qc).data
        mps, _ = _run_cpu(qc, shots=0)
        sv_mps = mps.get_statevector()
        assert _fidelity(sv_qiskit, sv_mps) > 0.9999


# ─────────────────────────────────────────────────────────────────────────────
#  GHZ state correctness
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
@pytest.mark.parametrize("n", [2, 3, 5, 8, 10, 16, 20])
def test_ghz_state(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()

    result = SIM.run(qc, shots=8192, seed=0)
    counts = result.get_counts()
    zeros = '0' * n
    ones = '1' * n
    total = sum(counts.values())
    p_zeros = counts.get(zeros, 0) / total
    p_ones = counts.get(ones, 0) / total

    assert p_zeros + p_ones > 0.98, \
        f"GHZ {n}q: P(00..0)+P(11..1) = {p_zeros+p_ones:.4f} (expected > 0.98)"
    assert p_zeros > 0.4, f"GHZ {n}q: P(00..0) = {p_zeros:.4f}"
    assert p_ones > 0.4, f"GHZ {n}q: P(11..1) = {p_ones:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
#  Bell state fidelity (exact statevector check for small n)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
def test_bell_state_fidelity():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    sv_qiskit = Statevector(qc).data
    mps, _ = _run_cpu(qc, shots=0)
    sv_mps = mps.get_statevector()

    assert _fidelity(sv_qiskit, sv_mps) > 0.9999
    assert mps.max_bond() == 2


# ─────────────────────────────────────────────────────────────────────────────
#  W state correctness
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
def test_w_state():
    """W state = (|001⟩+|010⟩+|100⟩)/√3."""
    qc = QuantumCircuit(3)
    # Prepare W state
    qc.ry(2 * math.acos(1.0 / math.sqrt(3)), 0)
    qc.ch(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 2)
    qc.cx(0, 1)

    sv_qiskit = Statevector(qc).data
    mps, _ = _run_cpu(qc, shots=0)
    sv_mps = mps.get_statevector()
    assert _fidelity(sv_qiskit, sv_mps) > 0.999


# ─────────────────────────────────────────────────────────────────────────────
#  Toffoli (CCX) gate
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
def test_toffoli():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.x(1)
    qc.ccx(0, 1, 2)  # decomposed on host

    sv_qiskit = Statevector(qc).data
    mps, _ = _run_cpu(qc, shots=0)
    sv_mps = mps.get_statevector()
    assert _fidelity(sv_qiskit, sv_mps) > 0.999


# ─────────────────────────────────────────────────────────────────────────────
#  Measurement counts distribution
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
def test_uniform_superposition_counts():
    """H⊗n should give uniform distribution over 2^n outcomes."""
    n = 5
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
    qc.measure_all()

    result = SIM.run(qc, shots=10000, seed=42)
    counts = result.get_counts()

    expected = 10000 / (2 ** n)
    for state, count in counts.items():
        assert abs(count - expected) < 3 * math.sqrt(expected), \
            f"Uniform state {state}: count={count}, expected≈{expected:.0f}"


# ─────────────────────────────────────────────────────────────────────────────
#  Bond dimension tracking
# ─────────────────────────────────────────────────────────────────────────────

def test_bond_dim_product_state():
    """Product state (no entanglement) → bond dim = 1 everywhere."""
    n = 10
    mps = MPSState(n, chi_max=16)
    for i in range(n):
        from cpu_reference import GATE_MATRICES
        mps.apply_1q(i, GATE_MATRICES['h'])
    assert mps.max_bond() == 1, f"Product state max_bond should be 1, got {mps.max_bond()}"


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
def test_bond_dim_grows_with_entanglement():
    n = 8
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    mps, _ = _run_cpu(qc, shots=0, chi_max=32)
    assert mps.max_bond() > 1, "GHZ state should have bond > 1"


# ─────────────────────────────────────────────────────────────────────────────
#  FPGAMPSSimulator API correctness
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
def test_fpga_simulator_api_cpu_mode():
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.measure_all()

    sim = FPGAMPSSimulator(num_cards=1, engine="cpu", chi_max=64)
    result = sim.run(qc, shots=2048, seed=7)
    assert isinstance(result, FPGAMPSResult)
    counts = result.get_counts()
    total = sum(counts.values())
    assert total == 2048

    zeros = counts.get('0000', 0) / total
    ones = counts.get('1111', 0) / total
    assert zeros + ones > 0.97


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
def test_multiple_circuits_cpu():
    circuits = []
    for n in [2, 3, 4]:
        qc = QuantumCircuit(n)
        qc.h(0)
        qc.cx(0, n - 1)
        qc.measure_all()
        circuits.append(qc)

    sim = FPGAMPSSimulator(num_cards=1, engine="cpu", chi_max=64)
    result = sim.run(circuits, shots=1000, seed=0)
    assert len(result._results) == 3
    for i in range(3):
        counts = result.get_counts(i)
        assert sum(counts.values()) == 1000


def test_engine_fpga_requires_xrt():
    """engine='fpga' must raise RuntimeError if XRT is absent."""
    import fpga_mps_simulator as fms
    if not fms.XRT_AVAILABLE:
        with pytest.raises(RuntimeError, match="pyxrt"):
            FPGAMPSSimulator(num_cards=1, engine="fpga")
    else:
        pytest.skip("XRT is present — FPGA init not tested here")


def test_no_silent_cpu_fallback():
    """engine='fpga' must NOT silently fall back when FPGA unavailable."""
    import fpga_mps_simulator as fms
    if not fms.XRT_AVAILABLE:
        with pytest.raises(RuntimeError):
            FPGAMPSSimulator(num_cards=1, engine="fpga")
    else:
        pytest.skip("XRT present; test only relevant when XRT absent")


# ─────────────────────────────────────────────────────────────────────────────
#  Expectation value
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
def test_expectation_value_zz():
    """|00⟩ has ⟨ZZ⟩ = +1; |11⟩ has ⟨ZZ⟩ = +1; |01⟩ has ⟨ZZ⟩ = -1."""
    for state, expected in [('00', 1.0), ('11', 1.0), ('01', -1.0)]:
        qc = QuantumCircuit(2)
        for i, b in enumerate(state):
            if b == '1':
                qc.x(i)

        mps, _ = _run_cpu(qc, shots=0)
        sv = mps.get_statevector()

        from fpga_mps_simulator import _pauli_expectation_sv
        val = _pauli_expectation_sv(sv, 'ZZ')
        assert abs(val - expected) < 1e-10, \
            f"|{state}⟩: ⟨ZZ⟩ = {val:.6f}, expected {expected}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
