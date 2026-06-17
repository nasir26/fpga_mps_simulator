#!/usr/bin/env python3
"""
Author: Nasir Ali
Organization: C-DAC Noida
Date: 26 June 2026
Target: Xilinx Alveo U55C, Vitis 2023.2, XRT, pyxrt, Python 3.8
"""

from __future__ import annotations

import argparse
import atexit
import math
import sys
import time
import traceback
from typing import Dict, Optional, Tuple

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from fpga_mps_simulator import FPGAMPSSimulator


# ─────────────────────────────────────────────────────────────────────────────
#  Test configuration (overridable from argv in __main__)
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    "cards": 4,        # number of U55C cards to spawn workers on
    "shots": 4096,     # measurement shots per circuit
    "chi":   64,       # host chi_max; kept == kernel CHI_MAX_SVD so the 2q
                       # contraction workspace is never silently truncated
    "seed":  12345,    # base RNG seed for reproducibility
}

# Statistical tolerance on a single probability for `shots` samples.
# 1-sigma for p=0.5 at 4096 shots is ~0.008; 0.05 leaves wide margin.
PROB_TOL = 0.05
MIN_FIDELITY = 0.99


# ─────────────────────────────────────────────────────────────────────────────
#  Lazily-built, shared simulator (FPGA, no fallback)
# ─────────────────────────────────────────────────────────────────────────────

_SIM: Optional[FPGAMPSSimulator] = None


def _get_sim() -> FPGAMPSSimulator:
    """Build the FPGA simulator once and reuse it (avoids reloading the
    .xclbin and respawning workers for every test)."""
    global _SIM
    if _SIM is None:
        _SIM = FPGAMPSSimulator(
            num_cards=CFG["cards"],
            chi_max=CFG["chi"],
            svd_cutoff=1e-12,
            engine="fpga",      # hard requirement: no CPU fallback
            mode="shots",       # genuine hardware kernel path
        )
    return _SIM


@atexit.register
def _shutdown_sim():
    global _SIM
    if _SIM is not None:
        try:
            _SIM.shutdown()
        except Exception:
            pass
        _SIM = None


# ─────────────────────────────────────────────────────────────────────────────
#  Comparison helpers
# ─────────────────────────────────────────────────────────────────────────────

def _state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """|<a|b>|^2 with normalization; insensitive to global phase."""
    a = np.asarray(a, dtype=np.complex128).ravel()
    b = np.asarray(b, dtype=np.complex128).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(abs(np.vdot(a / na, b / nb)) ** 2)


def _best_fidelity(fpga_sv: np.ndarray, oracle_sv: np.ndarray,
                   n: int) -> Tuple[float, str]:
    """Fidelity of fpga_sv against oracle_sv, trying both the direct and the
    qubit-reversed orderings so a pure endian convention difference is not
    flagged as a correctness failure. Returns (fidelity, ordering_label)."""
    f_direct = _state_fidelity(fpga_sv, oracle_sv)
    if n <= 1:
        return f_direct, "direct"
    rev = (np.asarray(oracle_sv, dtype=np.complex128)
           .reshape([2] * n)
           .transpose(list(range(n))[::-1])
           .ravel())
    f_rev = _state_fidelity(fpga_sv, rev)
    return (f_direct, "direct") if f_direct >= f_rev else (f_rev, "reversed")


def _probs_from_counts(counts: Dict) -> Dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {str(k).replace(" ", ""): v / total for k, v in counts.items()}


def _check_state(qc: QuantumCircuit, seed: int,
                 min_fid: float = MIN_FIDELITY):
    """Run on FPGA and assert statevector fidelity vs Qiskit's exact oracle.
    Returns (result, fidelity, ordering)."""
    sim = _get_sim()
    res = sim.run(qc, shots=CFG["shots"], seed=seed)

    # Proof we went through the hardware shots path, not a CPU branch.
    assert res.mode == "shots_fpga", \
        f"expected FPGA shots path, got mode={res.mode!r}"

    fpga_sv = np.asarray(res.get_statevector())
    oracle = Statevector(qc).data
    fid, order = _best_fidelity(fpga_sv, oracle, qc.num_qubits)
    assert fid >= min_fid, \
        f"statevector fidelity {fid:.5f} < {min_fid} (ordering={order})"
    return res, fid, order


def _assert_prob(probs: Dict[str, float], bitstring: str,
                 expected: float, tol: float = PROB_TOL):
    p = probs.get(bitstring, 0.0)
    assert abs(p - expected) <= tol, \
        f"P({bitstring!r}) = {p:.4f}, expected {expected:.4f} +/- {tol}"


# ─────────────────────────────────────────────────────────────────────────────
#  Circuit builders
# ─────────────────────────────────────────────────────────────────────────────

def _ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def _bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


# ─────────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_environment():
    """Hardware is present and the no-fallback engine resolved to FPGA.
    Fails fast (and clearly) if XRT or a device is missing."""
    try:
        import pyxrt as xrt
    except ImportError:
        raise AssertionError(
            "pyxrt not importable -- source /opt/xilinx/xrt/setup.sh. "
            "engine='fpga' has no CPU fallback, so the suite cannot run."
        )
    dev = xrt.device(0)             # raises if no card -> loud failure
    bdf = dev.get_info(xrt.xrt_info_device.bdf)
    assert bdf, "device 0 reported empty BDF"

    sim = _get_sim()
    assert sim._engine == "fpga", \
        f"engine resolved to {sim._engine!r}, expected 'fpga' (no fallback)"


def test_single_x():
    """X on a single qubit -> deterministic |1>."""
    qc = QuantumCircuit(1)
    qc.x(0)
    res, _, _ = _check_state(qc, CFG["seed"])
    probs = _probs_from_counts(res.get_counts())
    _assert_prob(probs, "1", 1.0, tol=1e-6)


def test_hadamard_superposition():
    """H on a single qubit -> balanced superposition."""
    qc = QuantumCircuit(1)
    qc.h(0)
    res, _, _ = _check_state(qc, CFG["seed"])
    probs = _probs_from_counts(res.get_counts())
    _assert_prob(probs, "0", 0.5)
    _assert_prob(probs, "1", 0.5)


def test_bell_state():
    """h(0), cx(0,1) -> (|00> + |11>)/sqrt(2); no |01> or |10>."""
    res, fid, order = _check_state(_bell(), CFG["seed"])
    probs = _probs_from_counts(res.get_counts())
    _assert_prob(probs, "00", 0.5)
    _assert_prob(probs, "11", 0.5)
    assert probs.get("01", 0.0) <= PROB_TOL
    assert probs.get("10", 0.0) <= PROB_TOL


def test_ghz3():
    res, _, _ = _check_state(_ghz(3), CFG["seed"])
    probs = _probs_from_counts(res.get_counts())
    _assert_prob(probs, "000", 0.5)
    _assert_prob(probs, "111", 0.5)


def test_ghz5():
    res, _, _ = _check_state(_ghz(5), CFG["seed"])
    probs = _probs_from_counts(res.get_counts())
    _assert_prob(probs, "00000", 0.5)
    _assert_prob(probs, "11111", 0.5)


def test_ry_rotation():
    """ry(theta) on |0>: P(1) = sin^2(theta/2)."""
    theta = math.pi / 3.0           # P(1) = sin^2(pi/6) = 0.25
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    res, _, _ = _check_state(qc, CFG["seed"])
    probs = _probs_from_counts(res.get_counts())
    _assert_prob(probs, "1", math.sin(theta / 2) ** 2)
    _assert_prob(probs, "0", math.cos(theta / 2) ** 2)


def test_nonadjacent_cx_routing():
    """h(0), cx(0,2) on 3 qubits -> exercises SWAP-based nearest-neighbour
    routing in compile_circuit. State = (|000> + |q0=1,q2=1>)/sqrt(2)."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 2)
    res, fid, order = _check_state(qc, CFG["seed"])
    probs = _probs_from_counts(res.get_counts())
    # The two non-zero outcomes are palindromic, so order-robust.
    dominant = {b for b, p in probs.items() if p > 0.05}
    assert dominant == {"000", "101"}, \
        f"unexpected dominant outcomes: {dominant}"
    _assert_prob(probs, "000", 0.5)
    _assert_prob(probs, "101", 0.5)


def test_statevector_fidelity_mixed_gates():
    """A deterministic mix of rotations + entanglers on a 4-qubit line,
    validated by full statevector fidelity against Qiskit."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.ry(0.7, 1)
    qc.rx(1.1, 2)
    qc.rz(0.4, 3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.t(0)
    qc.s(2)
    qc.cx(0, 1)
    _, fid, order = _check_state(qc, CFG["seed"])
    assert fid >= MIN_FIDELITY


def test_multi_circuit_shot_parallel():
    """Submit four circuits at once. With num_cards>1 these are dispatched
    across cards; each result is independently validated."""
    sim = _get_sim()
    circuits = [_bell(), _ghz(3), _ghz(4), _ghz(5)]
    res = sim.run(circuits, shots=CFG["shots"], seed=CFG["seed"])
    assert res.mode == "shots_fpga", \
        f"expected FPGA shots path, got mode={res.mode!r}"
    for i, qc in enumerate(circuits):
        fpga_sv = np.asarray(res.get_statevector(i))
        oracle = Statevector(qc).data
        fid, order = _best_fidelity(fpga_sv, oracle, qc.num_qubits)
        assert fid >= MIN_FIDELITY, \
            f"circuit {i} fidelity {fid:.5f} < {MIN_FIDELITY} ({order})"


def test_result_metadata():
    """Result container reports sane bookkeeping for a GHZ run."""
    sim = _get_sim()
    res = sim.run(_ghz(4), shots=CFG["shots"], seed=CFG["seed"])
    assert res.success() is True
    assert res.get_max_bond() >= 1
    bdims = res.get_bond_dims()
    assert isinstance(bdims, (list, tuple)) and len(bdims) >= 1
    # GHZ has a maximum bond dimension of 2.
    assert res.get_max_bond() <= 2, \
        f"GHZ max bond should be 2, got {res.get_max_bond()}"


def test_no_cpu_fallback_marker():
    """Defensive check: the engine never silently degrades to CPU. The result
    mode must carry the '_fpga' marker, and the resolved engine stays 'fpga'."""
    sim = _get_sim()
    assert sim._engine == "fpga"
    res = sim.run(_bell(), shots=512, seed=CFG["seed"])
    assert res.mode.endswith("_fpga"), \
        f"result mode {res.mode!r} does not indicate the FPGA path"


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone runner (also pytest-compatible: each test_* uses bare asserts)
# ─────────────────────────────────────────────────────────────────────────────

# Explicit order: environment check first, then progressively harder cases.
_ORDERED_TESTS = [
    test_environment,
    test_single_x,
    test_hadamard_superposition,
    test_bell_state,
    test_ghz3,
    test_ghz5,
    test_ry_rotation,
    test_nonadjacent_cx_routing,
    test_statevector_fidelity_mixed_gates,
    test_multi_circuit_shot_parallel,
    test_result_metadata,
    test_no_cpu_fallback_marker,
]


def _main() -> int:
    ap = argparse.ArgumentParser(description="FPGA MPS simulator test suite")
    ap.add_argument("--cards", type=int, default=CFG["cards"],
                    choices=(1, 2, 4), help="number of U55C cards")
    ap.add_argument("--shots", type=int, default=CFG["shots"])
    ap.add_argument("--chi", type=int, default=CFG["chi"],
                    help="host chi_max (keep <= kernel CHI_MAX_SVD = 64)")
    ap.add_argument("--seed", type=int, default=CFG["seed"])
    args = ap.parse_args()
    CFG.update(cards=args.cards, shots=args.shots, chi=args.chi, seed=args.seed)

    print("FPGA MPS Simulator -- Test Suite")
    print("=" * 60)
    print(f"  engine        : fpga (no CPU fallback)")
    print(f"  mode          : shots")
    print(f"  cards         : {CFG['cards']}")
    print(f"  shots         : {CFG['shots']}")
    print(f"  chi_max       : {CFG['chi']}")
    print(f"  seed          : {CFG['seed']}")
    print(f"  fidelity gate : >= {MIN_FIDELITY}")
    print("=" * 60)

    passed = failed = 0
    failures = []
    for fn in _ORDERED_TESTS:
        name = fn.__name__
        t0 = time.time()
        try:
            fn()
            dt = time.time() - t0
            print(f"  [PASS] {name:<42s} ({dt:6.2f}s)")
            passed += 1
        except AssertionError as e:
            dt = time.time() - t0
            print(f"  [FAIL] {name:<42s} ({dt:6.2f}s)  {e}")
            failed += 1
            failures.append((name, str(e)))
        except Exception as e:  # hardware / runtime errors, etc.
            dt = time.time() - t0
            print(f"  [ERR ] {name:<42s} ({dt:6.2f}s)  {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
            failures.append((name, f"{type(e).__name__}: {e}"))
            # If the very first (environment) test errors there is no point
            # continuing -- the FPGA path is unavailable and there is no
            # fallback by design.
            if fn is test_environment:
                print("\n  Environment check failed; aborting remaining tests.")
                break

    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, "
          f"{len(_ORDERED_TESTS)} total")
    if failures:
        print("  Failed:")
        for nm, msg in failures:
            print(f"    - {nm}: {msg}")
    print("=" * 60)

    _shutdown_sim()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())