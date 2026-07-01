# =============================================================================
#  fpga_mps_simulator - extra stress-test circuit families for bench_ceiling.py
#
#  Emits only gates the FPGA MPS engine natively supports (see _OP1Q/_OP2Q in
#  fpga_mps_simulator.py): single-qubit h/x/rx/ry/rz and adjacent two-qubit
#  cx/cz. Any non-adjacent two-qubit gate would be SWAP-routed by the host
#  compile_circuit(), but none of the families below need that -- every
#  two-qubit gate here already targets nearest neighbours.
#
#  Author: Nasir Ali
#  C-DAC Noida
# =============================================================================

import math
import numpy as np
from qiskit import QuantumCircuit


def circ_w_state(n: int):
    """
    Linear-staircase W-state preparation: (|100...0> + |010...0> + ... )/sqrt(n).

    Built from a controlled-RY at each step, decomposed into the engine-native
    ry + cx pair (ry(t/2), cx, ry(-t/2), cx) rather than emitting a fused
    controlled-rotation gate the kernel doesn't know. A single excitation is
    "passed along" the chain with an extra cx per step. Entanglement stays
    low (bond dimension <= 2 throughout), so this is a low-chi width stress
    test rather than a contraction stress test.
    """
    qc = QuantumCircuit(n)
    qc.x(0)
    for i in range(n - 1):
        remaining = n - i
        theta = 2.0 * math.acos(math.sqrt(1.0 / remaining))
        qc.ry(theta / 2, i + 1)
        qc.cx(i, i + 1)
        qc.ry(-theta / 2, i + 1)
        qc.cx(i, i + 1)
        qc.cx(i + 1, i)
    return qc


def circ_ghz_ladder(n: int):
    """
    GHZ-ladder / staircase-CNOT variant: a forward CX staircase (standard GHZ
    preparation) followed by a reverse CZ staircase "rung" pass across the
    same neighbour bonds. The rungs add a second full sweep of adjacent
    two-qubit gates without changing the GHZ correlations (CZ is diagonal in
    the computational basis reached after the CX staircase), so this mainly
    stress-tests sustained adjacent-gate throughput at large n while keeping
    bond dimension bounded near 2.
    """
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    for i in range(n - 2, -1, -1):
        qc.cz(i, i + 1)
    return qc


def circ_tfim_trotter(n: int, layers: int = 6, dt: float = 0.2,
                       J: float = 1.0, h: float = 1.0):
    """
    Transverse-field Ising model, first-order Trotter brickwork:
    alternating even/odd layers of ZZ(theta) couplings on neighbours,
    each decomposed as cx-rz-cx (engine-native), followed by a global
    Rx layer (transverse field). This is a physically meaningful circuit
    whose entanglement grows with Trotter depth, so bond dimension climbs
    toward chi_max as `layers` increases -- a genuine high-chi test rather
    than a synthetic one.
    """
    qc = QuantumCircuit(n)
    theta_zz = 2.0 * J * dt
    theta_x = 2.0 * h * dt
    for layer in range(layers):
        parity = layer % 2
        for i in range(parity, n - 1, 2):
            qc.cx(i, i + 1)
            qc.rz(theta_zz, i + 1)
            qc.cx(i, i + 1)
        for i in range(n):
            qc.rx(theta_x, i)
    return qc


def circ_random_brickwork(n: int, depth: int = 8, seed: int = 99):
    """
    Random single-qubit rotation + CZ brickwork (a standard random-circuit-
    sampling ansatz). Each layer applies a random Rx/Ry/Rz on every qubit,
    then a brick of CZ gates on alternating neighbour pairs. Entanglement
    grows with `depth` until it saturates chi_max, making depth the tunable
    knob for pushing SVD truncation on the FPGA contraction path.
    """
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(seed)
    gate_choices = ('rx', 'ry', 'rz')
    for d in range(depth):
        for i in range(n):
            g = gate_choices[rng.integers(0, 3)]
            theta = rng.uniform(0, 2 * math.pi)
            getattr(qc, g)(theta, i)
        parity = d % 2
        for i in range(parity, n - 1, 2):
            qc.cz(i, i + 1)
    return qc


EXTRA_FAMILIES = {
    "W-State":      (circ_w_state,
                      "linear-staircase W-state prep (low chi, ry+cx only)"),
    "GHZ-Ladder":   (circ_ghz_ladder,
                      "GHZ staircase + reverse CZ rung pass (bond~2, sustained adjacent-gate throughput)"),
    "TFIM-Trotter": (circ_tfim_trotter,
                      "transverse-field Ising Trotter brickwork (bond grows with depth)"),
    "Random-Brickwork": (circ_random_brickwork,
                      "random 1q + CZ brickwork (bond climbs toward chi_max with depth)"),
}
