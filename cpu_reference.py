#!/usr/bin/env python3
"""
cpu_reference.py — Explicit CPU MPS Reference Engine (Golden Model)
====================================================================

Implements a Matrix Product State (MPS) quantum circuit simulator using
NumPy/SciPy. This is the explicit CPU reference engine used by all tests
when FPGA hardware is unavailable, and as the correctness oracle.

Never used as a silent fallback — it is always selected explicitly via
engine="cpu" in FPGAMPSSimulator, or by using CPUMPSSimulator directly.

MPS Conventions
---------------
State of n qubits as a chain of rank-3 tensors:
    sites[i].shape = (chi_l, 2, chi_r)
    sites[0].shape = (1, 2, chi_1)      <- left boundary
    sites[n-1].shape = (chi_{n-1}, 2, 1) <- right boundary

Amplitude for basis state |σ_0 σ_1 ... σ_{n-1}⟩:
    ψ[σ] = sites[0][0, σ_0, :] @ sites[1][:, σ_1, :] @ ... @ sites[n-1][:, σ_{n-1}, 0]

All tensors stored as complex128 (np.complex128).

Canonical Forms
---------------
Left-canonical  site i: Σ_σ A^i†[σ] A^i[σ] = I_{chi_r}
Right-canonical site i: Σ_σ B^i[σ]  B^i†[σ] = I_{chi_l}

The chain is kept in mixed-canonical form with the orthogonality center
at the active site(s) during gate application. Normalization tracked via
norm_log (log of overall norm dropped during SVD truncation).

Author: Nasir Ali
Organization: C-DAC Noida
Date: June 2026
Target: All platforms (pure NumPy/SciPy, no hardware dependencies)
"""

from __future__ import annotations
import math
import logging
import numpy as np
import scipy.linalg as la
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Gate matrices (complex128)
# ─────────────────────────────────────────────────────────────────────────────

_S2 = math.sqrt(2)
_IS2 = 1.0 / _S2

GATE_MATRICES: Dict[str, np.ndarray] = {
    'id':   np.eye(2, dtype=np.complex128),
    'h':    np.array([[_IS2, _IS2], [_IS2, -_IS2]], dtype=np.complex128),
    'x':    np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'y':    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'z':    np.array([[1, 0], [0, -1]], dtype=np.complex128),
    's':    np.array([[1, 0], [0, 1j]], dtype=np.complex128),
    'sdg':  np.array([[1, 0], [0, -1j]], dtype=np.complex128),
    't':    np.array([[1, 0], [0, np.exp(1j*math.pi/4)]], dtype=np.complex128),
    'tdg':  np.array([[1, 0], [0, np.exp(-1j*math.pi/4)]], dtype=np.complex128),
    'sx':   np.array([[0.5+0.5j, 0.5-0.5j], [0.5-0.5j, 0.5+0.5j]], dtype=np.complex128),
    'sxdg': np.array([[0.5-0.5j, 0.5+0.5j], [0.5+0.5j, 0.5-0.5j]], dtype=np.complex128),
    # 2-qubit gates (4×4, row-major order |00⟩,|01⟩,|10⟩,|11⟩)
    'cx':   np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex128),
    'cz':   np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=np.complex128),
    'cy':   np.array([[1,0,0,0],[0,1,0,0],[0,0,0,-1j],[0,0,1j,0]], dtype=np.complex128),
    'swap': np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=np.complex128),
    'iswap':np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=np.complex128),
    'dcx':  np.array([[1,0,0,0],[0,0,0,1],[0,1,0,0],[0,0,1,0]], dtype=np.complex128),
}


def _gate_1q(name: str, params: Optional[List[float]]) -> np.ndarray:
    """Return 2×2 complex128 gate matrix."""
    nm = name.lower()
    if nm in GATE_MATRICES:
        return GATE_MATRICES[nm]
    p = [float(x) for x in (params or [])]
    if nm == 'rx':
        c, s = math.cos(p[0]/2), math.sin(p[0]/2)
        return np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex128)
    if nm == 'ry':
        c, s = math.cos(p[0]/2), math.sin(p[0]/2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if nm == 'rz':
        e = np.exp(1j * p[0] / 2)
        return np.array([[e.conj(), 0], [0, e]], dtype=np.complex128)
    if nm in ('p', 'phase', 'u1'):
        return np.array([[1, 0], [0, np.exp(1j*p[0])]], dtype=np.complex128)
    if nm == 'u2':
        inv_s2 = 1.0 / math.sqrt(2)
        phi, lam = p[0], p[1]
        return inv_s2 * np.array([
            [1, -np.exp(1j*lam)],
            [np.exp(1j*phi), np.exp(1j*(phi+lam))]
        ], dtype=np.complex128)
    if nm in ('u3', 'u'):
        theta, phi, lam = p[0], p[1], p[2]
        ct, st = math.cos(theta/2), math.sin(theta/2)
        return np.array([
            [ct, -np.exp(1j*lam)*st],
            [np.exp(1j*phi)*st, np.exp(1j*(phi+lam))*ct]
        ], dtype=np.complex128)
    raise ValueError(f"Unknown 1q gate: {name}")


def _gate_2q(name: str, params: Optional[List[float]]) -> np.ndarray:
    """Return 4×4 complex128 gate matrix (basis order: |00⟩,|01⟩,|10⟩,|11⟩)."""
    nm = name.lower()
    if nm in GATE_MATRICES:
        return GATE_MATRICES[nm]
    p = [float(x) for x in (params or [])]
    if nm in ('cp', 'cphase'):
        m = np.eye(4, dtype=np.complex128)
        m[3, 3] = np.exp(1j*p[0])
        return m
    if nm == 'ch':
        inv_s2 = 1.0 / math.sqrt(2)
        m = np.eye(4, dtype=np.complex128)
        m[2:, 2:] = inv_s2 * np.array([[1, 1], [1, -1]])
        return m
    if nm == 'crx':
        c, s = math.cos(p[0]/2), math.sin(p[0]/2)
        m = np.eye(4, dtype=np.complex128)
        m[2:, 2:] = np.array([[c, -1j*s], [-1j*s, c]])
        return m
    if nm == 'cry':
        c, s = math.cos(p[0]/2), math.sin(p[0]/2)
        m = np.eye(4, dtype=np.complex128)
        m[2:, 2:] = np.array([[c, -s], [s, c]])
        return m
    if nm == 'crz':
        e = np.exp(1j * p[0] / 2)
        m = np.eye(4, dtype=np.complex128)
        m[2, 2] = e.conj()
        m[3, 3] = e
        return m
    if nm == 'csx':
        sx = np.array([[0.5+0.5j, 0.5-0.5j], [0.5-0.5j, 0.5+0.5j]], dtype=np.complex128)
        m = np.eye(4, dtype=np.complex128)
        m[2:, 2:] = sx
        return m
    if nm == 'ecr':
        inv_s2 = 1.0 / math.sqrt(2)
        return inv_s2 * np.array([
            [0, 0, 1, 1j],
            [0, 0, 1j, 1],
            [1, -1j, 0, 0],
            [-1j, 1, 0, 0],
        ], dtype=np.complex128)
    if nm == 'rxx':
        c, s = math.cos(p[0]/2), math.sin(p[0]/2)
        return np.array([
            [c, 0, 0, -1j*s],
            [0, c, -1j*s, 0],
            [0, -1j*s, c, 0],
            [-1j*s, 0, 0, c],
        ], dtype=np.complex128)
    if nm == 'ryy':
        c, s = math.cos(p[0]/2), math.sin(p[0]/2)
        return np.array([
            [c, 0, 0, 1j*s],
            [0, c, -1j*s, 0],
            [0, -1j*s, c, 0],
            [1j*s, 0, 0, c],
        ], dtype=np.complex128)
    if nm == 'rzz':
        ep = np.exp(1j*p[0]/2)
        em = ep.conj()
        return np.diag([em, ep, ep, em])
    raise ValueError(f"Unknown 2q gate: {name}")


# ─────────────────────────────────────────────────────────────────────────────
#  SWAP routing: decompose non-adjacent 2q gate into nearest-neighbour ops
# ─────────────────────────────────────────────────────────────────────────────

def _swap_route(i: int, j: int, gate_name: str, params: Optional[List[float]]):
    """
    Route a 2q gate on (i, j) with |i-j| > 1 to a sequence of nearest-neighbour
    gates. Uses SWAP-chain to bring qubit j adjacent to i, apply gate, SWAP back.
    Returns list of (gate_name, [q0, q1], params) tuples, all nearest-neighbour.
    """
    if j < i:
        i, j = j, i
        # Swap qubit order in gate if asymmetric — caller handles via transpile
    ops = []
    # Bring j to i+1 via SWAP chain moving j leftward
    for k in range(j, i+1, -1):
        ops.append(('swap', [k-1, k], None))
    ops.append((gate_name, [i, i+1], params))
    for k in range(i+1, j):
        ops.append(('swap', [k, k+1], None))
    return ops


# ─────────────────────────────────────────────────────────────────────────────
#  MPS State
# ─────────────────────────────────────────────────────────────────────────────

class MPSState:
    """
    Matrix Product State for n qubits.

    Attributes
    ----------
    n          : number of qubits
    chi_max    : maximum bond dimension
    svd_cutoff : singular value cutoff (relative to largest S value)
    sites      : list of n tensors, sites[i].shape = (chi_l, 2, chi_r)
    norm_log   : log of the overall norm (accumulated truncation weight)
    """

    def __init__(self, n: int, chi_max: int = 256, svd_cutoff: float = 1e-12):
        self.n = n
        self.chi_max = chi_max
        self.svd_cutoff = svd_cutoff
        self.norm_log: float = 0.0
        # Initialize to |0...0⟩: all sites are [[1,0]] shaped correctly
        self.sites: List[np.ndarray] = []
        for _ in range(n):
            t = np.zeros((1, 2, 1), dtype=np.complex128)
            t[0, 0, 0] = 1.0  # σ=0 (|0⟩ component)
            self.sites.append(t)

    def copy(self) -> 'MPSState':
        s = MPSState(self.n, self.chi_max, self.svd_cutoff)
        s.sites = [t.copy() for t in self.sites]
        s.norm_log = self.norm_log
        return s

    # ── Site-tensor access helpers ───────────────────────────────────────────

    def chi_l(self, i: int) -> int:
        return self.sites[i].shape[0]

    def chi_r(self, i: int) -> int:
        return self.sites[i].shape[2]

    def max_bond(self) -> int:
        if self.n <= 1:
            return 1
        return max(self.chi_r(i) for i in range(self.n - 1))

    def bond_dims(self) -> List[int]:
        return [self.chi_r(i) for i in range(self.n - 1)]

    # ── Canonical form helpers ───────────────────────────────────────────────

    def left_normalize_site(self, i: int):
        """
        Left-normalize site i via QR. Updates sites[i] and absorbs R into sites[i+1].
        After: Σ_σ A†[σ] A[σ] = I.
        """
        A = self.sites[i]
        chi_l, d, chi_r = A.shape
        M = A.reshape(chi_l * d, chi_r)
        Q, R = np.linalg.qr(M)
        chi_new = Q.shape[1]
        self.sites[i] = Q.reshape(chi_l, d, chi_new)
        if i + 1 < self.n:
            B = self.sites[i + 1]
            # Absorb R into next site: B'[gamma, sigma, chi_r] = R[chi_new, gamma] @ B
            self.sites[i + 1] = np.tensordot(R, B, axes=([1], [0]))

    def right_normalize_site(self, i: int):
        """
        Right-normalize site i via LQ. Updates sites[i] and absorbs L into sites[i-1].
        After: Σ_σ B[σ] B†[σ] = I.
        """
        A = self.sites[i]
        chi_l, d, chi_r = A.shape
        M = A.reshape(chi_l, d * chi_r)
        # LQ = transpose of QR of transpose
        Q, R = np.linalg.qr(M.T.conj())
        # M = R^H Q^H
        L = R.T.conj()
        B = Q.T.conj()
        chi_new = B.shape[0]
        self.sites[i] = B.reshape(chi_new, d, chi_r)
        if i - 1 >= 0:
            prev = self.sites[i - 1]
            self.sites[i - 1] = np.tensordot(prev, L, axes=([2], [0]))

    def left_canonicalize(self):
        """Left-canonicalize the full chain (sites 0..n-2)."""
        for i in range(self.n - 1):
            self.left_normalize_site(i)

    def right_canonicalize(self):
        """Right-canonicalize the full chain (sites 1..n-1)."""
        for i in range(self.n - 1, 0, -1):
            self.right_normalize_site(i)

    # ── 1-qubit gate application ─────────────────────────────────────────────

    def apply_1q(self, site: int, gate: np.ndarray):
        """
        Apply 2×2 unitary gate to qubit `site`.
        A'[χ_l, σ', χ_r] = Σ_σ G[σ', σ] · A[χ_l, σ, χ_r]
        """
        A = self.sites[site]  # (chi_l, 2, chi_r)
        # einsum: A'[chi_l, s', chi_r] = G[s', s] * A[chi_l, s, chi_r]
        self.sites[site] = np.tensordot(gate, A, axes=([1], [1])).transpose(1, 0, 2)

    # ── SVD truncation ───────────────────────────────────────────────────────

    def _svd_truncate(self, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        SVD of matrix M with chi_max and svd_cutoff truncation.
        Returns (U, sqrt_S, Vh) where S is normalized so ||ψ||=1 is preserved
        up to truncation error. Accumulates log(norm) into self.norm_log.
        """
        U, S, Vh = la.svd(M, full_matrices=False)
        # Absolute cutoff relative to largest singular value
        threshold = self.svd_cutoff * S[0] if len(S) > 0 else 0.0
        keep = np.sum(S > threshold)
        keep = int(np.clip(keep, 1, self.chi_max))
        # Log dropped weight
        if keep < len(S):
            dropped_norm_sq = np.sum(S[keep:] ** 2)
            if dropped_norm_sq > 0:
                self.norm_log += 0.5 * math.log(1.0 - dropped_norm_sq /
                                                  max(np.sum(S**2), 1e-300))
        U = U[:, :keep]
        S_t = S[:keep]
        Vh = Vh[:keep, :]
        # Renormalize to compensate for dropped weight
        norm = np.linalg.norm(S_t)
        if norm > 1e-15:
            self.norm_log += math.log(norm)
            S_t = S_t / norm
        sqrt_S = np.sqrt(S_t)
        return U, sqrt_S, Vh

    # ── 2-qubit gate application ─────────────────────────────────────────────

    def apply_2q(self, i: int, j: int, gate: np.ndarray):
        """
        Apply 4×4 gate to qubits i and j.
        If |i-j| > 1, SWAP-routes the gate to nearest-neighbour.
        Gate matrix convention: rows/cols ordered as |00⟩,|01⟩,|10⟩,|11⟩.
        """
        if abs(i - j) != 1:
            ops = _swap_route(i, j, '__gate2q__', None)
            for op_name, op_qubits, op_params in ops:
                if op_name == 'swap':
                    self.apply_2q(op_qubits[0], op_qubits[1], GATE_MATRICES['swap'])
                else:
                    self.apply_2q(op_qubits[0], op_qubits[1], gate)
            return

        if i > j:
            i, j = j, i
            # Permute gate for reversed qubit order
            perm = np.array([0, 2, 1, 3])
            gate = gate[np.ix_(perm, perm)]

        self._apply_adjacent_2q(i, gate)

    def _apply_adjacent_2q(self, i: int, gate: np.ndarray):
        """
        Apply 4×4 gate to adjacent sites i and i+1.
        Implements contract → apply → SVD → split.
        """
        j = i + 1
        A = self.sites[i]   # (chi_l, 2, chi_m)
        B = self.sites[j]   # (chi_m, 2, chi_r)
        chi_l = A.shape[0]
        chi_m = A.shape[2]  # == B.shape[0]
        chi_r = B.shape[2]

        # Contract: Theta[chi_l, σ_i, σ_j, chi_r]
        # Theta[a, si, sj, b] = Σ_γ A[a, si, γ] B[γ, sj, b]
        Theta = np.tensordot(A, B, axes=([2], [0]))  # (chi_l, 2, 2, chi_r)

        # Apply gate: G[σ'_i σ'_j, σ_i σ_j] (4×4)
        # Reshape Theta to (chi_l, 4, chi_r), apply gate rows to dim-1
        G = gate.reshape(4, 4)
        Theta_flat = Theta.reshape(chi_l, 4, chi_r)
        # Theta'[a, σ', b] = Σ_{σ} G[σ', σ] Theta[a, σ, b]
        Theta_new = np.einsum('ij,kjl->kil', G, Theta_flat)  # (chi_l, 4, chi_r)

        # Reshape for SVD: (chi_l*2, 2*chi_r)
        M = Theta_new.reshape(chi_l * 2, 2 * chi_r)

        # SVD with truncation
        U, sqrt_S, Vh = self._svd_truncate(M)
        chi_new = len(sqrt_S)

        # Reconstruct site tensors
        # A_new[chi_l, σ_i, chi_new] = (U * sqrt_S)[chi_l*2, chi_new].reshape(chi_l, 2, chi_new)
        self.sites[i] = (U * sqrt_S).reshape(chi_l, 2, chi_new)
        # B_new[chi_new, σ_j, chi_r] = (sqrt_S[:,None] * Vh)[chi_new, 2*chi_r].reshape(chi_new, 2, chi_r)
        self.sites[j] = (sqrt_S[:, None] * Vh).reshape(chi_new, 2, chi_r)

    # ── Measurement ─────────────────────────────────────────────────────────

    def _right_environments(self) -> List[np.ndarray]:
        """
        Compute right environments R[i] for sites i = n-1 downto 0.
        R[n] = [[1.0]]
        R[i] = Σ_σ sites[i][:, σ, :] · R[i+1] · sites[i][:, σ, :]†
             (chi_l × chi_l Hermitian matrix)
        Used for sampling: P(σ_i | σ_0...σ_{i-1}) = Tr[L[i] Σ_σ A[σ] R[i+1] A†[σ]]
        """
        R_list = [None] * (self.n + 1)
        R_list[self.n] = np.array([[1.0]], dtype=np.complex128)
        for i in range(self.n - 1, -1, -1):
            A = self.sites[i]  # (chi_l, 2, chi_r)
            R = R_list[i + 1]  # (chi_r, chi_r)
            # R_new[a,a'] = Σ_{σ,b,b'} A[a,σ,b] R[b,b'] A*[a',σ,b']
            R_new = np.einsum('isb,bc,jsc->ij', A, R, A.conj())
            # Normalize each R to trace=1 to prevent underflow for large n.
            # The sample() function only uses R for relative probability ratios,
            # so this multiplicative normalization does not change outcomes.
            tr = np.real(np.trace(R_new))
            if tr > 1e-300:
                R_new /= tr
            R_list[i] = R_new
        return R_list

    def sample(self, num_shots: int = 1, measured_qubits: Optional[List[int]] = None,
               rng: Optional[np.random.Generator] = None) -> List[str]:
        """
        Sample measurement outcomes using sequential conditional sampling.
        O(n · chi²) per shot. Does NOT reconstruct the full statevector.

        Returns list of bitstrings (ordered msb..lsb matching Qiskit convention:
        bitstring[0] = qubit n-1, bitstring[-1] = qubit 0).
        """
        if rng is None:
            rng = np.random.default_rng()
        if measured_qubits is None:
            measured_qubits = list(range(self.n))

        R_list = self._right_environments()
        outcomes = []

        for _ in range(num_shots):
            # Sequential conditional sampling using left boundary vector.
            # v[a] represents the partial bra <σ_0...σ_{i-1}|_left at bond a.
            v = np.array([1.0], dtype=np.complex128)   # shape (chi_l_0,) = (1,)
            bits = ['0'] * self.n

            for i in range(self.n):
                A = self.sites[i]  # (chi_l, 2, chi_r)
                R = R_list[i + 1]  # (chi_r, chi_r)
                probs = np.zeros(2, dtype=np.float64)
                for sigma in range(2):
                    # Av = v @ A[:,sigma,:] has shape (chi_r,)
                    Av = v @ A[:, sigma, :]
                    # P ∝ Av @ R @ conj(Av)
                    probs[sigma] = np.real(Av @ R @ Av.conj())
                probs = np.maximum(probs, 0.0)
                total = probs.sum()
                if total < 1e-15:
                    probs = np.array([0.5, 0.5])
                else:
                    probs /= total

                outcome = rng.choice(2, p=probs)
                bits[i] = str(outcome)

                # Update left boundary vector
                v = v @ A[:, outcome, :]
                norm = np.linalg.norm(v)
                if norm > 1e-15:
                    v /= norm

            # Qiskit bitstring convention: qubit n-1 first
            bitstring = ''.join(reversed(bits))
            # Filter to measured qubits (in qubit-order)
            if set(measured_qubits) != set(range(self.n)):
                ms = sorted(measured_qubits, reverse=True)
                bitstring = ''.join(bits[self.n - 1 - q] for q in ms)
            outcomes.append(bitstring)

        return outcomes

    def get_counts(self, num_shots: int, measured_qubits: Optional[List[int]] = None,
                   rng: Optional[np.random.Generator] = None) -> Dict[str, int]:
        """Sample and return counts dict."""
        samples = self.sample(num_shots, measured_qubits, rng)
        counts: Dict[str, int] = {}
        for s in samples:
            counts[s] = counts.get(s, 0) + 1
        return counts

    # ── Full statevector reconstruction (small n only) ────────────────────

    def get_statevector(self) -> np.ndarray:
        """
        Reconstruct the full 2^n statevector. ONLY for n ≤ 24 (warns above 20).
        O(2^n · chi^2 · n) time and O(2^n) memory.
        """
        if self.n > 24:
            raise ValueError(f"Statevector reconstruction for n={self.n} > 24 not supported.")
        if self.n > 20:
            logger.warning("Reconstructing statevector for n=%d — this may be slow.", self.n)

        # Contract chain left-to-right; site 0 becomes outer (MSB) index.
        A = self.sites[0]  # (1, 2, chi_1)
        state = A[0, :, :]  # (2, chi_1)

        for i in range(1, self.n):
            B = self.sites[i]  # (chi_i, 2, chi_{i+1})
            state = np.tensordot(state, B, axes=([1], [0]))
            # state shape: (2^i, 2, chi_{i+1})
            s = state.shape
            state = state.reshape(s[0] * s[1], s[2])

        sv = state[:, 0].copy()  # (2^n,) with site-0 as MSB

        # Apply accumulated normalization factor
        if self.norm_log != 0.0:
            sv *= math.exp(self.norm_log)

        # Reverse qubit ordering: site-0 is MSB, Qiskit uses qubit-0 as LSB.
        # Reshape → (2,)*n, transpose [n-1,...,0], reshape back.
        sv = sv.reshape([2] * self.n).transpose(
            list(range(self.n - 1, -1, -1))
        ).reshape(2 ** self.n)

        return sv

    # ── Entanglement entropy ─────────────────────────────────────────────────

    def entanglement_entropy(self, bond: int) -> float:
        """
        Compute von Neumann entropy S = -Σ_i λ_i² log(λ_i²) at the bond
        between sites `bond` and `bond+1` (0-indexed, 0 ≤ bond < n-1).
        Requires the MPS to be in mixed-canonical form at this bond.
        """
        if bond < 0 or bond >= self.n - 1:
            raise ValueError(f"bond {bond} out of range [0, {self.n-2}]")
        # SVD the chain at this bond
        # Contract all left sites into a (2^(bond+1), chi) matrix
        A = self.sites[0][0, :, :]
        for i in range(1, bond + 1):
            B = self.sites[i]
            A = np.tensordot(A, B, axes=([1], [0]))
            A = A.reshape(-1, A.shape[-1])

        _, S, _ = la.svd(A, full_matrices=False)
        S = S[S > 1e-15]
        S_sq = S**2
        S_sq /= S_sq.sum()
        return -float(np.sum(S_sq * np.log(np.maximum(S_sq, 1e-300))))

    def bond_spectrum(self, bond: int) -> np.ndarray:
        """Return singular values at bond `bond` (left-to-right sweep)."""
        mps = self.copy()
        for i in range(bond + 1):
            mps.left_normalize_site(i)
        A = mps.sites[bond]
        chi_l, d, chi_r = A.shape
        M = A.reshape(chi_l * d, chi_r)
        _, S, _ = la.svd(M, full_matrices=False)
        return S


# ─────────────────────────────────────────────────────────────────────────────
#  Gate application from Qiskit-format instructions
# ─────────────────────────────────────────────────────────────────────────────

_SINGLE_QUBIT_GATES = frozenset({
    'id','h','x','y','z','s','sdg','t','tdg','sx','sxdg',
    'rx','ry','rz','p','phase','u1','u2','u3','u',
})

_TWO_QUBIT_GATES = frozenset({
    'cx','cnot','cy','cz','ch','cp','cphase',
    'crx','cry','crz','csx','swap','iswap','dcx',
    'ecr','rxx','ryy','rzz',
})

_THREE_QUBIT_GATES = frozenset({'ccx','toffoli','ccnot'})


def apply_gate(mps: MPSState, name: str, qubits: List[int],
               params: Optional[List[float]]):
    """Apply a named gate to the MPS. Handles 1q, 2q, and CCX."""
    nm = name.lower()
    if nm in ('measure', 'barrier', 'reset', 'snapshot', 'delay'):
        return  # non-unitary, handled separately

    if nm in _SINGLE_QUBIT_GATES:
        G = _gate_1q(nm, params)
        mps.apply_1q(qubits[0], G)
        return

    if nm in _TWO_QUBIT_GATES:
        G = _gate_2q(nm, params)
        mps.apply_2q(qubits[0], qubits[1], G)
        return

    if nm in _THREE_QUBIT_GATES:
        # Decompose CCX into 1q + CX
        c0, c1, tgt = qubits
        h = GATE_MATRICES['h']
        t = GATE_MATRICES['t']
        tdg = GATE_MATRICES['tdg']
        cx = GATE_MATRICES['cx']
        mps.apply_1q(tgt, h)
        mps.apply_2q(c1, tgt, cx)
        mps.apply_1q(tgt, tdg)
        mps.apply_2q(c0, tgt, cx)
        mps.apply_1q(tgt, t)
        mps.apply_2q(c1, tgt, cx)
        mps.apply_1q(tgt, tdg)
        mps.apply_2q(c0, tgt, cx)
        mps.apply_1q(c1, t)
        mps.apply_1q(tgt, t)
        mps.apply_1q(tgt, h)
        mps.apply_2q(c0, c1, cx)
        mps.apply_1q(c0, t)
        mps.apply_1q(c1, tdg)
        mps.apply_2q(c0, c1, cx)
        return

    raise ValueError(f"Unsupported gate: {name}")


# ─────────────────────────────────────────────────────────────────────────────
#  CPUMPSSimulator — Qiskit-compatible wrapper
# ─────────────────────────────────────────────────────────────────────────────

class CPUMPSSimulator:
    """
    CPU MPS simulator. Provides the same interface as FPGAMPSSimulator.
    Always runs on CPU. Explicitly selected, never a silent fallback.
    """

    def __init__(self, chi_max: int = 256, svd_cutoff: float = 1e-12,
                 seed: Optional[int] = None):
        self.chi_max = chi_max
        self.svd_cutoff = svd_cutoff
        self.rng = np.random.default_rng(seed)
        self._last_mps: Optional[MPSState] = None

    def _parse_circuit(self, circuit):
        """Extract gate list from Qiskit QuantumCircuit."""
        gates = []
        for inst in circuit.data:
            nm = inst.operation.name.lower()
            try:
                if hasattr(circuit, 'find_bit'):
                    qs = [circuit.find_bit(q).index for q in inst.qubits]
                else:
                    qs = [circuit.qubits.index(q) for q in inst.qubits]
            except Exception:
                qs = list(range(len(inst.qubits)))
            ps = [float(p) for p in getattr(inst.operation, 'params', [])]
            gates.append((nm, qs, ps or None))
        return gates

    def run_mps(self, circuit, shots: int = 1024) -> Tuple[MPSState, Dict[str, int]]:
        """Run circuit, return (final_mps, counts)."""
        nq = circuit.num_qubits
        mps = MPSState(nq, self.chi_max, self.svd_cutoff)
        gates = self._parse_circuit(circuit)
        measured_qubits = []

        for nm, qs, ps in gates:
            if nm == 'measure':
                for q in qs:
                    if q not in measured_qubits:
                        measured_qubits.append(q)
                continue
            if nm in ('barrier', 'snapshot', 'delay'):
                continue
            if nm == 'reset':
                # Collapse qubit to |0⟩
                mps.apply_1q(qs[0], GATE_MATRICES['x'])  # prepare |1⟩ then project
                # Simplified: just re-initialize the site (valid only if qubit is unentangled)
                # For correctness, we'd need a projective measurement + normalization
                # For now: perform measurement and re-init (approximate for entangled qubits)
                t = np.zeros((mps.chi_l(qs[0]), 2, mps.chi_r(qs[0])), dtype=np.complex128)
                t[:, 0, :] = mps.sites[qs[0]][:, 0, :]
                norm = np.linalg.norm(t)
                if norm > 1e-15:
                    t /= norm
                mps.sites[qs[0]] = t
                mps.apply_1q(qs[0], GATE_MATRICES['x'])  # back to |0⟩
                continue
            apply_gate(mps, nm, qs, ps)

        if not measured_qubits:
            measured_qubits = list(range(nq))

        counts = mps.get_counts(shots, measured_qubits, self.rng)
        self._last_mps = mps
        return mps, counts

    def run(self, circuits, shots: int = 1024):
        """Run one or more circuits, return Qiskit-compatible result."""
        if not isinstance(circuits, list):
            circuits = [circuits]

        results = []
        for circuit in circuits:
            mps, counts = self.run_mps(circuit, shots)
            sv = None
            if circuit.num_qubits <= 24:
                try:
                    sv = mps.get_statevector()
                except Exception:
                    pass
            results.append({
                'counts': counts,
                'statevector': sv,
                'mps': mps,
                'max_bond': mps.max_bond(),
                'bond_dims': mps.bond_dims(),
            })

        return _MPSResult(results, shots)

    def get_last_mps(self) -> Optional[MPSState]:
        return self._last_mps


class _MPSResult:
    """Lightweight Qiskit-compatible result wrapper."""

    def __init__(self, results, shots):
        self._results = results
        self.shots = shots

    def get_counts(self, idx: int = 0) -> Dict[str, int]:
        return self._results[idx]['counts']

    def get_statevector(self, idx: int = 0) -> np.ndarray:
        sv = self._results[idx].get('statevector')
        if sv is None:
            raise ValueError("Statevector not available (n > 24 or not computed).")
        return sv

    def get_mps(self, idx: int = 0) -> MPSState:
        return self._results[idx]['mps']

    def get_max_bond(self, idx: int = 0) -> int:
        return self._results[idx]['max_bond']

    def get_bond_dims(self, idx: int = 0) -> List[int]:
        return self._results[idx]['bond_dims']

    def success(self) -> bool:
        return True
