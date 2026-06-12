"""
test_mps_specific.py — MPS-specific property tests

Tests SVD truncation, entanglement entropy, canonical form,
bond spectrum, and MPS-specific correctness properties.

Author: Nasir Ali
Organization: C-DAC Noida
Date: June 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from cpu_reference import MPSState, apply_gate, GATE_MATRICES, CPUMPSSimulator

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, entropy
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def _fidelity(sv1, sv2):
    sv1 = sv1.ravel() / np.linalg.norm(sv1)
    sv2 = sv2.ravel() / np.linalg.norm(sv2)
    return float(abs(np.dot(sv1.conj(), sv2)) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
#  SVD truncation
# ─────────────────────────────────────────────────────────────────────────────

class TestSVDTruncation:

    def test_product_state_not_truncated(self):
        """Product state: SVD should not truncate (chi = 1)."""
        n = 10
        mps = MPSState(n, chi_max=16)
        for i in range(n):
            mps.apply_1q(i, GATE_MATRICES['h'])
        for i in range(n - 1):
            assert mps.chi_r(i) == 1, f"bond {i}: chi_r should be 1"

    def test_chi_max_respected(self):
        """Bond dimension should not exceed chi_max."""
        chi_max = 4
        n = 6
        mps = MPSState(n, chi_max=chi_max)
        mps.apply_1q(0, GATE_MATRICES['h'])
        for i in range(n - 1):
            G = GATE_MATRICES['cx']
            mps.apply_2q(i, i + 1, G)
        assert mps.max_bond() <= chi_max, \
            f"max_bond {mps.max_bond()} exceeds chi_max {chi_max}"

    def test_svd_cutoff_removes_small_values(self):
        """Large svd_cutoff should truncate more aggressively."""
        n = 4
        mps_loose = MPSState(n, chi_max=64, svd_cutoff=0.1)
        mps_strict = MPSState(n, chi_max=64, svd_cutoff=1e-12)

        for mps in [mps_loose, mps_strict]:
            mps.apply_1q(0, GATE_MATRICES['h'])
            for i in range(n - 1):
                mps.apply_2q(i, i + 1, GATE_MATRICES['cx'])

        # Loose cutoff should have ≤ strict cutoff's bond dim
        assert mps_loose.max_bond() <= mps_strict.max_bond() + 1

    def test_norm_preserved_after_truncation(self):
        """MPS norm should be preserved after SVD truncation."""
        n = 6
        mps = MPSState(n, chi_max=8)
        mps.apply_1q(0, GATE_MATRICES['h'])
        for i in range(n - 1):
            mps.apply_2q(i, i + 1, GATE_MATRICES['cx'])

        sv = mps.get_statevector()
        norm = np.linalg.norm(sv)
        assert abs(norm - 1.0) < 0.01, f"Norm after truncation: {norm:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
#  Entanglement entropy
# ─────────────────────────────────────────────────────────────────────────────

class TestEntanglementEntropy:

    def test_product_state_zero_entropy(self):
        """Product state has zero entanglement entropy at every bond."""
        n = 8
        mps = MPSState(n, chi_max=16)
        for i in range(n):
            mps.apply_1q(i, GATE_MATRICES['h'])
        for bond in range(n - 1):
            S = mps.entanglement_entropy(bond)
            assert abs(S) < 1e-10, f"Bond {bond}: entropy {S:.2e} should be 0"

    @pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
    def test_bell_entropy(self):
        """Bell state has entropy ln(2) ≈ 0.693 at the central bond."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        sim = CPUMPSSimulator(chi_max=8)
        mps, _ = sim.run_mps(qc, shots=0)
        S = mps.entanglement_entropy(0)
        assert abs(S - np.log(2)) < 0.01, \
            f"Bell state entropy at bond 0: {S:.4f}, expected {np.log(2):.4f}"

    @pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
    def test_ghz_entropy_grows(self):
        """GHZ entropy at middle bond increases with n."""
        entropies = []
        for n in [4, 6, 8]:
            qc = QuantumCircuit(n)
            qc.h(0)
            for i in range(n - 1):
                qc.cx(i, i + 1)
            sim = CPUMPSSimulator(chi_max=16)
            mps, _ = sim.run_mps(qc, shots=0)
            S = mps.entanglement_entropy(n // 2 - 1)
            entropies.append(S)

        # GHZ entropy at center should be ln(2) regardless of n
        for S in entropies:
            assert abs(S - np.log(2)) < 0.05, \
                f"GHZ center entropy: {S:.4f}, expected {np.log(2):.4f}"

    def test_entropy_nonnegative(self):
        """Entropy must be non-negative at all bonds."""
        n = 6
        mps = MPSState(n, chi_max=16)
        mps.apply_1q(0, GATE_MATRICES['h'])
        for i in range(n - 1):
            mps.apply_2q(i, i + 1, GATE_MATRICES['cx'])
        for bond in range(n - 1):
            S = mps.entanglement_entropy(bond)
            assert S >= -1e-12, f"Bond {bond}: negative entropy {S:.6e}"


# ─────────────────────────────────────────────────────────────────────────────
#  Bond spectrum
# ─────────────────────────────────────────────────────────────────────────────

class TestBondSpectrum:

    def test_bond_spectrum_sorted(self):
        """Singular values at each bond should be in descending order."""
        n = 6
        mps = MPSState(n, chi_max=16)
        mps.apply_1q(0, GATE_MATRICES['h'])
        for i in range(n - 1):
            mps.apply_2q(i, i + 1, GATE_MATRICES['cx'])
        for bond in range(n - 1):
            sv = mps.bond_spectrum(bond)
            for j in range(len(sv) - 1):
                assert sv[j] >= sv[j + 1] - 1e-10, \
                    f"Bond {bond}: singular values not sorted at {j}"

    def test_product_state_spectrum(self):
        """Product state has only one nonzero singular value at each bond."""
        n = 5
        mps = MPSState(n, chi_max=8)
        for i in range(n):
            mps.apply_1q(i, GATE_MATRICES['h'])
        for bond in range(n - 1):
            sv = mps.bond_spectrum(bond)
            # Only one value (should be ≈1 / sqrt(2) or 1 depending on normalization)
            assert len(sv) == 1 or sv[1] < 1e-10, \
                f"Bond {bond}: product state should have 1 significant SV, got {sv}"


# ─────────────────────────────────────────────────────────────────────────────
#  Canonical form
# ─────────────────────────────────────────────────────────────────────────────

class TestCanonicalForm:

    def test_left_canonicalize(self):
        """Left-canonical sites should satisfy A^† A = I (isometric tensors)."""
        n = 5
        mps = MPSState(n, chi_max=16)
        mps.apply_1q(0, GATE_MATRICES['h'])
        for i in range(n - 1):
            mps.apply_2q(i, i + 1, GATE_MATRICES['cx'])
        mps.left_canonicalize()

        # Check isometry: sum_sigma A[chi_l, sigma, chi_r]^† A = I
        for i in range(n - 1):
            A = mps.sites[i]  # (chi_l, 2, chi_r)
            chi_l, _, chi_r = A.shape
            # Contract: M[b, b'] = sum_{a, sigma} A*[a,sigma,b] A[a,sigma,b']
            M = np.einsum('asb,asc->bc', A.conj(), A)
            eye = np.eye(chi_r)
            assert np.allclose(M, eye, atol=1e-8), \
                f"Site {i}: left-canonical A^†A ≠ I (max err {np.max(np.abs(M - eye)):.2e})"

    def test_right_canonicalize(self):
        """Right-canonical sites should satisfy B B^† = I."""
        n = 5
        mps = MPSState(n, chi_max=16)
        mps.apply_1q(0, GATE_MATRICES['h'])
        for i in range(n - 1):
            mps.apply_2q(i, i + 1, GATE_MATRICES['cx'])
        mps.right_canonicalize()

        for i in range(1, n):
            B = mps.sites[i]  # (chi_l, 2, chi_r)
            chi_l, _, chi_r = B.shape
            # Contract: M[a, a'] = sum_{sigma, b} B[a,sigma,b] B*[a',sigma,b]
            M = np.einsum('asb,csb->ac', B, B.conj())
            eye = np.eye(chi_l)
            assert np.allclose(M, eye, atol=1e-8), \
                f"Site {i}: right-canonical BB^† ≠ I (max err {np.max(np.abs(M - eye)):.2e})"


# ─────────────────────────────────────────────────────────────────────────────
#  Large-n scalability (MPS advantage)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
@pytest.mark.parametrize("n", [50, 100, 200])
def test_large_ghz_sampling(n):
    """GHZ state on n qubits must sample correctly (can't use statevector)."""
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()

    sim = CPUMPSSimulator(chi_max=4, svd_cutoff=1e-12, seed=42)
    mps, counts = sim.run_mps(qc, shots=2000)

    zeros = '0' * n
    ones = '1' * n
    total = sum(counts.values())
    p_both = (counts.get(zeros, 0) + counts.get(ones, 0)) / total
    assert p_both > 0.97, f"GHZ {n}q: P(00..0)+P(11..1) = {p_both:.4f}"
    assert mps.max_bond() <= 4, f"GHZ {n}q: max_bond {mps.max_bond()} > 4"


def test_product_state_large_n():
    """Product state on 500 qubits should be O(n) not O(2^n) in memory."""
    n = 500
    mps = MPSState(n, chi_max=2)
    for i in range(n):
        mps.apply_1q(i, GATE_MATRICES['h'])
    assert mps.max_bond() == 1
    # Spot-check a random sample
    rng = np.random.default_rng(42)
    counts = mps.get_counts(100, list(range(min(20, n))), rng)
    assert sum(counts.values()) == 100


# ─────────────────────────────────────────────────────────────────────────────
#  Statevector reconstruction
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
def test_statevector_normalization():
    qc = QuantumCircuit(5)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(0.5, 2)
    qc.rz(1.0, 3)
    qc.cx(2, 4)

    sim = CPUMPSSimulator(chi_max=32)
    mps, _ = sim.run_mps(qc, shots=0)
    sv = mps.get_statevector()
    assert abs(np.linalg.norm(sv) - 1.0) < 1e-10

    sv_qiskit = Statevector(qc).data
    assert _fidelity(sv, sv_qiskit) > 0.9999


def test_statevector_raises_large_n():
    """get_statevector() must raise ValueError for n > 24."""
    mps = MPSState(25, chi_max=2)
    with pytest.raises(ValueError, match="24"):
        mps.get_statevector()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
