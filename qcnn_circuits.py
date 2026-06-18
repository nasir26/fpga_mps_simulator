# =============================================================================
#  fpga_mps_simulator - QCNN circuit family
#
#  Quantum Convolutional Neural Network workloads for the FPGA MPS engine.
#  Emits only gates the engine natively supports:
#     - single-qubit gates (h, ry, rz, x, ...)
#     - adjacent two-qubit gates, including arbitrary 4x4 UnitaryGate
#     - Toffoli (ccx), which the host decomposes into 1q/2q gates
#  Non-adjacent two-qubit gates are SWAP-routed by the host backend.
#
#  Two families are provided:
#     qcnn_cluster  - exact-style cluster-model QCNN (low chi, width stress,
#                     heavy SWAP routing + Toffoli decomposition)
#     qcnn_generic  - trainable QCNN with random SU(4) convolution blocks;
#                     bond dimension grows with depth, so this is the
#                     high-chi workload where the FPGA contraction earns
#                     its place.
#
#  Author: Nasir Ali
#  C-DAC Noida
# =============================================================================

import math
import numpy as np
from qiskit import QuantumCircuit


# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------
def _random_su4(rng):
    """Haar-random 4x4 unitary (a generic two-qubit convolution block)."""
    z = (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) / math.sqrt(2.0)
    q, r = np.linalg.qr(z)
    ph = np.diag(r) / np.abs(np.diag(r))
    return q * ph  # column-corrected -> Haar unitary


def _random_su2(rng):
    """Haar-random 2x2 unitary (a single-qubit pooling/conv rotation)."""
    z = (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))) / math.sqrt(2.0)
    q, r = np.linalg.qr(z)
    ph = np.diag(r) / np.abs(np.diag(r))
    return q * ph


# -----------------------------------------------------------------------------
#  Family 1 : exact-style cluster-model QCNN  (low chi)
# -----------------------------------------------------------------------------
def circ_qcnn_cluster(n, depth=2):
    """
    Cluster-model QCNN recogniser, deferred-measurement (all-unitary) form.

    Structure per convolution-pooling unit, reducing the active chain L -> L/3:
      convolution : controlled-phase (cz) bonds + Toffoli with X-basis controls
      pooling     : phase-flip (z) on the survivor controlled on the
                    neighbour's X-basis state  ==  H . cz . H
    Fully-connected layer measures Z_{i-1} X_i Z_{i+1} on the central survivor;
    in unitary form we rotate that stabiliser onto a single-qubit Z readout.

    All measurements are deferred to the end (the survivors are the readout
    qubits), so the engine sees only h / cz / ccx and never a mid-circuit
    measurement. chi stays small (this is a MERA-structured state), which makes
    it a width-and-routing stress test rather than a contraction stress test.
    """
    qc = QuantumCircuit(n)

    # Prepare a 1D cluster state as the input: H on all, then cz on every bond.
    for i in range(n):
        qc.h(i)
    for i in range(n - 1):
        qc.cz(i, i + 1)

    active = list(range(n))
    for _ in range(depth):
        if len(active) < 3:
            break

        # ---- convolution: stabiliser-style entangling layer -----------------
        # cz bonds between active neighbours
        for j in range(len(active) - 1):
            qc.cz(active[j], active[j + 1])
        # Toffoli with controls in the X basis on each contiguous triple
        for j in range(0, len(active) - 2, 1):
            a, b, c = active[j], active[j + 1], active[j + 2]
            qc.h(a); qc.h(c)
            qc.ccx(a, c, b)
            qc.h(a); qc.h(c)

        # ---- pooling: keep every 3rd active qubit, fold neighbours in -------
        survivors = active[1::3] if len(active) >= 3 else active
        for s in survivors:
            for nb in (s - 1, s + 1):
                if nb in active:
                    # apply Z to survivor controlled on neighbour's |-> state
                    qc.h(nb)
                    qc.cz(nb, s)
                    qc.h(nb)
        active = survivors

    # ---- fully-connected readout: rotate Z_{i-1} X_i Z_{i+1} onto one qubit --
    if len(active) >= 3:
        c0, c1, c2 = active[len(active) // 2 - 1: len(active) // 2 + 2]
        qc.cz(c0, c1)
        qc.cz(c1, c2)
        qc.h(c1)
    return qc


# -----------------------------------------------------------------------------
#  Family 2 : trainable / generic QCNN  (high chi -- the FPGA-relevant regime)
# -----------------------------------------------------------------------------
def circ_qcnn_generic(n, depth=None, seed=20260618):
    """
    Generic trainable QCNN: stacked convolution + pooling layers that halve the
    active-qubit count each step, finished by a fully-connected block.

      convolution : translationally-invariant brick of random SU(4) blocks on
                    nearest-neighbour active pairs (the same block is reused
                    across a layer -- O(log N) distinct parameters, as in the
                    QCNN architecture).
      pooling     : a random SU(4) block on each (measured, survivor) pair,
                    in deferred-measurement form; the measured qubit is then
                    dropped from the active set.
      fc          : a random SU(4) block on the final pair of survivors.

    Random SU(4) convolution blocks drive entanglement up, so the measured
    maximum bond dimension climbs toward chi_max with depth. This is the
    workload that exercises the O(chi^3) FPGA contraction rather than the
    O(n chi^2) host sampler.
    """
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(seed)

    # light input so layer 0 has something to act on
    for i in range(n):
        qc.ry(rng.uniform(0, 2 * math.pi), i)

    active = list(range(n))
    layer = 0
    max_layers = depth if depth is not None else n  # run until 1-2 survivors
    while len(active) >= 2 and layer < max_layers:
        # one reused conv block + one reused pool block per layer (param sharing)
        u_conv = _random_su4(rng)
        u_pool = _random_su4(rng)

        # ---- convolution: brick of identical blocks on active neighbours ----
        for off in (0, 1):
            for j in range(off, len(active) - 1, 2):
                qc.unitary(u_conv, [active[j], active[j + 1]],
                           label="conv")

        # ---- pooling: fold odd-indexed active qubits into their left neighbour
        survivors, dropped = [], []
        for j in range(0, len(active) - 1, 2):
            keep, meas = active[j], active[j + 1]
            qc.unitary(u_pool, [meas, keep], label="pool")  # deferred control
            survivors.append(keep)
            dropped.append(meas)
        if len(active) % 2 == 1:
            survivors.append(active[-1])

        active = survivors
        layer += 1

    # ---- fully-connected block on the surviving qubits ----------------------
    if len(active) >= 2:
        qc.unitary(_random_su4(rng), [active[0], active[1]], label="fc")
    elif len(active) == 1:
        qc.unitary(_random_su2(rng), [active[0]], label="fc")
    return qc


# -----------------------------------------------------------------------------
#  Registry entries -- merge into the simulator's FAMILIES dict
# -----------------------------------------------------------------------------
QCNN_FAMILIES = {
    "QCNN-Cluster": (circ_qcnn_cluster,
                     "cluster-model QCNN, deferred meas (low chi, routing/Toffoli stress)"),
    "QCNN-Generic": (circ_qcnn_generic,
                     "trainable QCNN, random SU(4) conv (chi grows toward chi_max)"),
}
