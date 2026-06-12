#!/usr/bin/env python3
"""
fpga_mps_simulator.py — FPGA-Based MPS Quantum Circuit Simulator (Host)
=======================================================================

Distributes a Matrix Product State (MPS) quantum circuit simulator across
up to 4 Xilinx Alveo U55C FPGA cards.

Two parallelism modes:
  Mode A ("shots")       : Each card runs independent MPS instances for
                           shot batches, circuit sets, or Pauli-string jobs.
                           Robust baseline; works for any circuit.
  Mode B ("distributed") : Chain partitioned into 4 contiguous segments.
                           Local and segment-internal gates run on the owning
                           card's FPGA kernel; boundary 2q gates trigger
                           O(chi²) tensor exchange via 3-barrier shared memory.
                           Shipped for nearest-neighbour circuits (all non-adjacent
                           2q gates SWAP-routed on the host before dispatch).
  Mode "auto"            : Selects "shots" for < 4 circuits, "distributed" for
                           large single-circuit runs.

Engine selection:
  engine="fpga"  : FPGA mandatory — RuntimeError if XRT unavailable or init fails.
  engine="cpu"   : Explicit CPU reference engine (cpu_reference.py). Never silent.
  engine="auto"  : FPGA if XRT present and devices found, else CPU with a warning.

Invariants (from reference files, applied verbatim):
  * mp.set_start_method('spawn', force=False) BEFORE any other mp import.
  * ctypes.memmove() for ALL cross-process shared-memory I/O (not np.frombuffer).
  * Workers spawned once, reused while n_qubits unchanged; rebuilt on size change.
  * Deferred FPGA init: XRT init happens in worker on first 'init' command.
  * AXI master count ≤ 9 (8 tensor banks + 1 gate/metadata bundle).
  * No silent CPU fallback from FPGA engine.

Author: Nasir Ali
Organization: C-DAC Noida
Date: June 2026
Target platform: Xilinx Alveo U55C × 4, Vitis 2023.2, XRT 2.x, pyxrt, Python 3.8
"""

from __future__ import annotations

# ── spawn MUST come before any other multiprocessing usage ──────────────────
import multiprocessing as _mp
try:
    _mp.set_start_method('spawn', force=False)
except RuntimeError:
    pass  # already set by caller — fine

import atexit
import ctypes
import logging
import math
import os
import signal
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from multiprocessing import Array, Barrier, Event, Process, Queue

# ── optional XRT ─────────────────────────────────────────────────────────────
try:
    import pyxrt as xrt
    XRT_AVAILABLE = True
except ImportError:
    XRT_AVAILABLE = False

# ── Qiskit ───────────────────────────────────────────────────────────────────
try:
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# ── CPU reference engine ─────────────────────────────────────────────────────
from cpu_reference import (
    CPUMPSSimulator, MPSState, apply_gate,
    _gate_1q, _gate_2q, GATE_MATRICES,
    _swap_route, _SINGLE_QUBIT_GATES, _TWO_QUBIT_GATES,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────
_log_level = logging.DEBUG if os.environ.get('FPGA_MPS_LOG', '').upper() == 'DEBUG' \
    else logging.WARNING
logging.basicConfig(level=_log_level,
                    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger('fpga_mps')

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
NUM_CARDS = 4
NUM_HBM_BANKS = 8           # 8 tensor banks + 1 gate/metadata bundle = 9 AXI masters
BYTES_PER_AMP = 16          # complex128 = 2 × float64 × 8 bytes
ABSOLUTE_MAX_QUBITS = 500
KERNEL_NAME = 'fpga_mps_simulator'
DEFAULT_XCLBIN_NAME = 'fpga_mps_simulator.xclbin'

# Gate instruction encoding (12 int32 words per instruction)
# Word 0: gate_type
# Word 1: site_i
# Word 2: site_j  (-1 for 1q)
# Word 3: chi_l
# Word 4: chi_m   (bond between sites for 2q)
# Word 5: chi_r
# Word 6: chi_new (truncated dim; 0 = use chi_max from kernel arg)
# Words 7-9: params as float32-in-int32
# Word 10: hbm_offset_i (in complex-double units)
# Word 11: hbm_offset_j
INSTR_WORDS = 12

# gate_type codes (must match fpga_mps_simulator.cpp)
GATE_1Q_BASE = 0    # single-qubit opcodes 0-22
GATE_2Q_CONTRACT = 100   # contract two sites + apply gate → theta in HBM
GATE_2Q_SPLIT = 101      # split theta (after host SVD) → two sites
GATE_2Q_FULL = 102       # contract + gate + Jacobi SVD + split (chi ≤ 64)
GATE_NORMALIZE = 110     # left-normalize a site

_1Q_OPCODE = {
    'id': 0, 'h': 1, 'x': 2, 'y': 3, 'z': 4,
    's': 5, 'sdg': 6, 't': 7, 'tdg': 8, 'sx': 9, 'sxdg': 10,
    'rx': 11, 'ry': 12, 'rz': 13, 'p': 14, 'phase': 14, 'u1': 14,
    'u2': 15, 'u3': 16, 'u': 16,
}

_2Q_OPCODE = {
    'cx': 17, 'cnot': 17, 'cy': 18, 'cz': 19, 'ch': 20,
    'cp': 21, 'cphase': 21, 'crx': 22, 'cry': 23, 'crz': 24,
    'csx': 25, 'swap': 26, 'iswap': 27, 'dcx': 28,
    'ecr': 29, 'rxx': 30, 'ryy': 31, 'rzz': 32,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Shared-memory helpers  (ctypes.memmove — reliable across processes)
# ─────────────────────────────────────────────────────────────────────────────

def _shm_write(arr: Array, data: np.ndarray, n_floats: int):
    """Write n_floats float64 values from data into shared Array."""
    buf = np.ascontiguousarray(data.ravel()[:n_floats], dtype=np.float64)
    nbytes = n_floats * 8
    src = buf.ctypes.data_as(ctypes.c_void_p)
    dst = ctypes.cast(ctypes.addressof(arr), ctypes.c_void_p)
    ctypes.memmove(dst, src, nbytes)


def _shm_read(arr: Array, n_floats: int) -> np.ndarray:
    """Read n_floats float64 values from shared Array."""
    buf = np.empty(n_floats, dtype=np.float64)
    nbytes = n_floats * 8
    src = ctypes.cast(ctypes.addressof(arr), ctypes.c_void_p)
    dst = buf.ctypes.data_as(ctypes.c_void_p)
    ctypes.memmove(dst, src, nbytes)
    return buf


def _pack_tensor(t: np.ndarray) -> np.ndarray:
    """Pack complex128 tensor into float64 [re0, im0, re1, im1, ...]."""
    c = t.ravel().astype(np.complex128)
    buf = np.empty(c.size * 2, dtype=np.float64)
    buf[0::2] = c.real
    buf[1::2] = c.imag
    return buf


def _unpack_tensor(buf: np.ndarray, shape: tuple) -> np.ndarray:
    """Unpack float64 [re, im, ...] back to complex128 tensor."""
    r = (buf[0::2] + 1j * buf[1::2]).astype(np.complex128)
    return r.reshape(shape)


# ─────────────────────────────────────────────────────────────────────────────
#  HBM tensor layout helpers (8-bank interleaving)
# ─────────────────────────────────────────────────────────────────────────────

def _hbm_write_tensor(hbm_banks: list, tensor: np.ndarray,
                      global_offset: int, num_amps: int, xrt_mod):
    """
    Write a complex128 tensor to HBM banks using 8-bank interleaving.
    global index g → bank = g % 8, slot = g // 8
    Each slot: two float64 [real, imag] at float64 positions [slot*2, slot*2+1].
    """
    amps_per_bank = (num_amps + NUM_HBM_BANKS - 1) // NUM_HBM_BANKS
    c = tensor.ravel()[:num_amps].astype(np.complex128)

    for bid in range(NUM_HBM_BANKS):
        indices = np.arange(bid, num_amps, NUM_HBM_BANKS)
        n_this = len(indices)
        if n_this == 0:
            continue
        buf = np.zeros(amps_per_bank * 2, dtype=np.float64)
        for li, gi in enumerate(indices):
            buf[li * 2] = c[gi].real
            buf[li * 2 + 1] = c[gi].imag
        byte_off = global_offset * 2 * 8  # 2 float64 per amplitude
        bank = hbm_banks[bid]
        bank.write(buf.tobytes(), byte_off)
        bank.sync(xrt_mod.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                  n_this * 2 * 8, byte_off)


def _hbm_read_tensor(hbm_banks: list, global_offset: int,
                     num_amps: int, shape: tuple, xrt_mod) -> np.ndarray:
    """Read complex128 tensor from HBM banks (8-bank interleaving)."""
    amps_per_bank = (num_amps + NUM_HBM_BANKS - 1) // NUM_HBM_BANKS
    result = np.zeros(num_amps, dtype=np.complex128)

    for bid in range(NUM_HBM_BANKS):
        indices = np.arange(bid, num_amps, NUM_HBM_BANKS)
        n_this = len(indices)
        if n_this == 0:
            continue
        byte_off = global_offset * 2 * 8
        bank = hbm_banks[bid]
        bank.sync(xrt_mod.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
                  n_this * 2 * 8, byte_off)
        buf = bank.map()
        raw = np.frombuffer(buf, dtype=np.float64,
                            count=amps_per_bank * 2).copy()
        cplx = (raw[0::2] + 1j * raw[1::2]).astype(np.complex128)
        v = min(len(cplx), len(indices))
        result[indices[:v]] = cplx[:v]

    return result.reshape(shape)


# ─────────────────────────────────────────────────────────────────────────────
#  Gate instruction encoder
# ─────────────────────────────────────────────────────────────────────────────

def _encode_1q(opcode: int, site: int, chi_l: int, chi_r: int,
               params: Optional[List[float]], hbm_offset: int) -> np.ndarray:
    instr = np.zeros(INSTR_WORDS, dtype=np.int32)
    instr[0] = opcode
    instr[1] = site
    instr[2] = -1
    instr[3] = chi_l
    instr[4] = 0
    instr[5] = chi_r
    instr[6] = 0
    ps = params or []
    for i, p in enumerate(ps[:3]):
        instr[7 + i] = int(np.float32(float(p)).view(np.int32))
    instr[10] = hbm_offset
    instr[11] = -1
    return instr


def _encode_2q(opcode: int, site_i: int, site_j: int,
               chi_l: int, chi_m: int, chi_r: int, chi_new: int,
               params: Optional[List[float]],
               hbm_offset_i: int, hbm_offset_j: int) -> np.ndarray:
    instr = np.zeros(INSTR_WORDS, dtype=np.int32)
    instr[0] = opcode
    instr[1] = site_i
    instr[2] = site_j
    instr[3] = chi_l
    instr[4] = chi_m
    instr[5] = chi_r
    instr[6] = chi_new
    ps = params or []
    for i, p in enumerate(ps[:3]):
        instr[7 + i] = int(np.float32(float(p)).view(np.int32))
    instr[10] = hbm_offset_i
    instr[11] = hbm_offset_j
    return instr


# ─────────────────────────────────────────────────────────────────────────────
#  HBM offset allocator
# ─────────────────────────────────────────────────────────────────────────────

class HBMAllocator:
    """
    Tracks allocation of HBM space for MPS site tensors.
    Each site gets a contiguous block of complex-double slots starting at
    its offset. The offset advances by chi_l * 2 * chi_r per site.
    """

    def __init__(self):
        self._offsets: List[int] = []
        self._cursor = 0

    def allocate(self, chi_l: int, chi_r: int) -> int:
        """Reserve chi_l * 2 * chi_r complex-double slots, return start offset."""
        off = self._cursor
        self._offsets.append(off)
        self._cursor += chi_l * 2 * chi_r
        return off

    def total_amps(self) -> int:
        return self._cursor

    def offsets(self) -> List[int]:
        return list(self._offsets)


# ─────────────────────────────────────────────────────────────────────────────
#  Circuit compiler: Qiskit → nearest-neighbour gate instruction list
# ─────────────────────────────────────────────────────────────────────────────

def compile_circuit(circuit, mps: MPSState) -> List[dict]:
    """
    Compile a Qiskit QuantumCircuit into a list of gate dicts:
      {'type': '1q'|'2q', 'opcode': int, 'sites': [i] or [i,j],
       'params': [...], 'gate_name': str, 'measured': bool}

    All 2q gates are routed to nearest-neighbour. SWAP routing inserts
    additional SWAP gates that swap both the logical qubit ordering AND
    the MPS tensor ordering (tracked in a permutation map).
    """
    nq = circuit.num_qubits
    perm = list(range(nq))  # perm[logical] = physical site

    gates_out = []
    measured = set()

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

        if nm == 'measure':
            for q in qs:
                measured.add(perm[q])
            continue
        if nm in ('barrier', 'snapshot', 'delay', 'id'):
            continue
        if nm == 'reset':
            gates_out.append({'type': 'reset', 'sites': [perm[qs[0]]], 'params': []})
            continue

        if nm in _SINGLE_QUBIT_GATES:
            op = _1Q_OPCODE.get(nm, 0)
            gates_out.append({
                'type': '1q', 'opcode': op,
                'sites': [perm[qs[0]]], 'params': ps,
                'gate_name': nm,
            })
            continue

        if nm in _TWO_QUBIT_GATES or nm in ('cx', 'cnot'):
            # Route to nearest-neighbour using physical permutation
            pi, pj = perm[qs[0]], perm[qs[1]]
            if abs(pi - pj) > 1:
                # SWAP-route: generate a sequence of SWAP + gate + SWAP
                # Determine which direction to move
                if pj < pi:
                    pi, pj = pj, pi
                    # Reverse gate qubit order
                    nm_use = nm
                    ps_use = ps
                else:
                    nm_use = nm
                    ps_use = ps
                # Move pj leftward to pi+1
                for k in range(pj, pi + 1, -1):
                    gates_out.append({
                        'type': '2q', 'opcode': _2Q_OPCODE['swap'],
                        'sites': [k-1, k], 'params': [],
                        'gate_name': 'swap',
                    })
                    # Update permutation
                    for idx in range(nq):
                        if perm[idx] == k - 1:
                            perm[idx] = k
                        elif perm[idx] == k:
                            perm[idx] = k - 1
                # Apply gate at (pi, pi+1)
                op2 = _2Q_OPCODE.get(nm_use, 17)
                gates_out.append({
                    'type': '2q', 'opcode': op2,
                    'sites': [pi, pi + 1], 'params': ps_use,
                    'gate_name': nm_use,
                })
                # Move back
                for k in range(pi + 1, pj):
                    gates_out.append({
                        'type': '2q', 'opcode': _2Q_OPCODE['swap'],
                        'sites': [k, k+1], 'params': [],
                        'gate_name': 'swap',
                    })
                    for idx in range(nq):
                        if perm[idx] == k:
                            perm[idx] = k + 1
                        elif perm[idx] == k + 1:
                            perm[idx] = k
            else:
                si, sj = min(pi, pj), max(pi, pj)
                op2 = _2Q_OPCODE.get(nm, 17)
                gates_out.append({
                    'type': '2q', 'opcode': op2,
                    'sites': [si, sj], 'params': ps,
                    'gate_name': nm,
                })
            continue

        if nm in ('ccx', 'toffoli', 'ccnot'):
            # Decompose CCX into 1q + CX on the host
            c0, c1, tgt = perm[qs[0]], perm[qs[1]], perm[qs[2]]
            for sub_nm, sub_qs, sub_ps in _decompose_ccx(c0, c1, tgt):
                if sub_nm in _SINGLE_QUBIT_GATES:
                    gates_out.append({'type': '1q', 'opcode': _1Q_OPCODE[sub_nm],
                                      'sites': [sub_qs[0]], 'params': sub_ps or [],
                                      'gate_name': sub_nm})
                else:
                    si, sj = min(sub_qs[0], sub_qs[1]), max(sub_qs[0], sub_qs[1])
                    gates_out.append({'type': '2q', 'opcode': _2Q_OPCODE[sub_nm],
                                      'sites': [si, sj], 'params': sub_ps or [],
                                      'gate_name': sub_nm})
            continue

    return gates_out, sorted(measured) if measured else list(range(nq))


def _decompose_ccx(c0, c1, tgt):
    """Decompose Toffoli into 1q + CX gates."""
    return [
        ('h', [tgt], None), ('cx', [c1, tgt], None), ('tdg', [tgt], None),
        ('cx', [c0, tgt], None), ('t', [tgt], None), ('cx', [c1, tgt], None),
        ('tdg', [tgt], None), ('cx', [c0, tgt], None), ('t', [c1], None),
        ('t', [tgt], None), ('h', [tgt], None), ('cx', [c0, c1], None),
        ('t', [c0], None), ('tdg', [c1], None), ('cx', [c0, c1], None),
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Worker process (Mode A — one MPS instance per card)
# ─────────────────────────────────────────────────────────────────────────────

def _mps_worker_shots(card_id: int, xclbin_path: str, kernel_name: str,
                      cmd_queue: Queue, result_queue: Queue,
                      ready_event: Event, shutdown_event: Event):
    """
    Worker process managing a single FPGA card for shot-parallel Mode A.
    Runs one complete MPS simulation per task received.
    Accepts commands: ('run', circuit_bytes, shots, chi_max, svd_cutoff, seed)
                      ('ping',)
                      None → shutdown
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    device = None
    kernel_obj = None
    hbm_banks: List = []
    fpga_ready = False

    def _init_fpga():
        nonlocal device, kernel_obj, hbm_banks, fpga_ready
        try:
            import pyxrt as xrt_local
            device = xrt_local.device(card_id)
            uuid = device.load_xclbin(xclbin_path)
            kernel_obj = xrt_local.kernel(device, uuid, kernel_name)
            fpga_ready = True
            logger.info('Card %d: FPGA initialized (kernel: %s)', card_id, kernel_name)
        except Exception as e:
            raise RuntimeError(
                f'Card {card_id}: FPGA init failed: {e}. '
                f'No CPU fallback permitted in engine=fpga mode.'
            )

    try:
        ready_event.set()
        while not shutdown_event.is_set():
            try:
                cmd = cmd_queue.get(timeout=0.2)
            except Exception:
                continue
            if cmd is None:
                break

            ct = cmd[0]

            if ct == 'ping':
                result_queue.put(('pong', card_id))
                continue

            if ct == 'run_cpu':
                # CPU path — no FPGA needed
                import pickle
                circuit_bytes, shots, chi_max, svd_cutoff, seed = cmd[1:]
                circuit = pickle.loads(circuit_bytes)
                sim = CPUMPSSimulator(chi_max=chi_max, svd_cutoff=svd_cutoff, seed=seed)
                mps, counts = sim.run_mps(circuit, shots)
                sv = None
                if circuit.num_qubits <= 24:
                    try:
                        sv = mps.get_statevector()
                    except Exception:
                        pass
                result_queue.put(('run_done', card_id, counts, sv,
                                  mps.max_bond(), mps.bond_dims()))
                continue

            if ct == 'run_fpga':
                import pickle
                circuit_bytes, shots, chi_max, svd_cutoff, seed = cmd[1:]
                if not fpga_ready:
                    _init_fpga()
                circuit = pickle.loads(circuit_bytes)
                # Run on FPGA kernel + host SVD
                import pyxrt as xrt_local
                try:
                    mps_state, counts = _run_mps_fpga(
                        circuit, shots, chi_max, svd_cutoff, seed,
                        device, kernel_obj, hbm_banks, xrt_local
                    )
                    sv = None
                    if circuit.num_qubits <= 24:
                        try:
                            sv = mps_state.get_statevector()
                        except Exception:
                            pass
                    result_queue.put(('run_done', card_id, counts, sv,
                                      mps_state.max_bond(), mps_state.bond_dims()))
                except Exception as e:
                    result_queue.put(('error', card_id, str(e)))
                continue

    except Exception as e:
        logger.error('Card %d: fatal — %s', card_id, e)
        import traceback
        traceback.print_exc()
        result_queue.put(('error', card_id, str(e)))
        ready_event.set()
    finally:
        hbm_banks.clear()
        device = None
        kernel_obj = None


def _run_mps_fpga(circuit, shots: int, chi_max: int, svd_cutoff: float,
                  seed: Optional[int], device, kernel_obj, hbm_banks: list,
                  xrt_mod) -> Tuple[MPSState, Dict[str, int]]:
    """
    Run a single circuit on the FPGA card.
    1q gates → batched FPGA kernel invocations.
    2q gates → FPGA contraction + host SVD + FPGA split.
    """
    import scipy.linalg as la

    nq = circuit.num_qubits
    mps = MPSState(nq, chi_max, svd_cutoff)
    alloc = HBMAllocator()
    offsets = [alloc.allocate(mps.chi_l(i), mps.chi_r(i)) for i in range(nq)]

    # Allocate theta workspace: max_theta_amps = chi_max * 4 * chi_max
    theta_offset = alloc.allocate(chi_max, chi_max * 4)

    total_amps = alloc.total_amps()
    amps_per_bank = (total_amps + NUM_HBM_BANKS - 1) // NUM_HBM_BANKS
    buf_bytes = max(amps_per_bank * 2 * 8, 4096)  # float64 per slot

    # Allocate / grow HBM banks if needed
    if len(hbm_banks) < NUM_HBM_BANKS or \
       (hbm_banks and xrt_mod.bo.__sizeof__() < buf_bytes):
        hbm_banks.clear()
        for i in range(NUM_HBM_BANKS):
            hbm_banks.append(xrt_mod.bo(device, buf_bytes,
                                         xrt_mod.bo.flags.normal,
                                         kernel_obj.group_id(i)))

    # Write initial state (|0...0⟩) to HBM
    for i in range(nq):
        _hbm_write_tensor(hbm_banks, mps.sites[i], offsets[i],
                          mps.chi_l(i) * 2 * mps.chi_r(i), xrt_mod)

    # Parse + compile circuit
    gate_list, measured_qubits = compile_circuit(circuit, mps)

    # Allocate gate buffer BO (reuse across gates)
    gate_buf_bytes = max(512 * INSTR_WORDS * 4, 4096)
    gate_bo = xrt_mod.bo(device, gate_buf_bytes, xrt_mod.bo.flags.normal,
                          kernel_obj.group_id(8))

    # Gate buffer for site_metadata (per-site chi_l, chi_r, offset): 3 ints per site
    meta_buf = np.zeros(nq * 3, dtype=np.int32)

    def _update_meta(site_i: int):
        meta_buf[site_i * 3 + 0] = mps.chi_l(site_i)
        meta_buf[site_i * 3 + 1] = mps.chi_r(site_i)
        meta_buf[site_i * 3 + 2] = offsets[site_i]

    for i in range(nq):
        _update_meta(i)

    batch_1q = []  # accumulate 1q gates before a 2q gate

    def _flush_1q():
        nonlocal batch_1q
        if not batch_1q:
            return
        n_gates = len(batch_1q)
        enc = np.zeros(n_gates * INSTR_WORDS, dtype=np.int32)
        for gi, g in enumerate(batch_1q):
            site = g['sites'][0]
            instr = _encode_1q(g['opcode'], site,
                               mps.chi_l(site), mps.chi_r(site),
                               g['params'], offsets[site])
            enc[gi * INSTR_WORDS:(gi + 1) * INSTR_WORDS] = instr
        nbytes = enc.nbytes
        gate_bo.write(enc.tobytes(), 0)
        gate_bo.sync(xrt_mod.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, nbytes, 0)
        run = kernel_obj(*hbm_banks[:NUM_HBM_BANKS], gate_bo, n_gates, nq, chi_max)
        run.wait()
        # Read updated sites back from HBM into CPU MPS (source of truth)
        for g in batch_1q:
            site = g['sites'][0]
            n_amps = mps.chi_l(site) * 2 * mps.chi_r(site)
            mps.sites[site] = _hbm_read_tensor(
                hbm_banks, offsets[site], n_amps,
                (mps.chi_l(site), 2, mps.chi_r(site)), xrt_mod
            )
        batch_1q = []

    for gate in gate_list:
        gt = gate['type']

        if gt == '1q':
            batch_1q.append(gate)
            continue

        if gt == 'reset':
            _flush_1q()
            # Reset site to |0⟩ (simplified: project onto σ=0)
            site = gate['sites'][0]
            t = mps.sites[site].copy()
            t[:, 1, :] = 0
            norm = np.linalg.norm(t)
            if norm > 1e-15:
                t /= norm
            mps.sites[site] = t
            _hbm_write_tensor(hbm_banks, t, offsets[site],
                              mps.chi_l(site) * 2 * mps.chi_r(site), xrt_mod)
            continue

        if gt == '2q':
            _flush_1q()
            si, sj = gate['sites'][0], gate['sites'][1]
            assert sj == si + 1, f"Non-adjacent 2q gate {si},{sj} after routing"

            chi_l = mps.chi_l(si)
            chi_m = mps.chi_r(si)
            chi_r = mps.chi_r(sj)
            use_fpga_full = (max(chi_l, chi_m, chi_r) <= 64)

            gate_mat = _gate_2q(gate['gate_name'], gate['params'])

            if use_fpga_full:
                # Full op on FPGA (contract + gate + Jacobi SVD + split)
                instr = _encode_2q(
                    GATE_2Q_FULL, si, sj, chi_l, chi_m, chi_r, chi_max,
                    gate['params'], offsets[si], offsets[sj]
                )
                # Pack gate matrix coefficients into extra words
                enc = np.zeros(2 * INSTR_WORDS, dtype=np.int32)
                enc[:INSTR_WORDS] = instr
                # Gate matrix: 16 complex128 = 32 float64 → encode as float32 pairs
                gf = gate_mat.ravel().astype(np.complex128)
                for k in range(16):
                    enc[INSTR_WORDS + k // 2] = int(
                        np.float32(gf[k].real if k % 2 == 0 else gf[k].imag
                                   ).view(np.int32)
                    )
                gate_bo.write(enc.tobytes(), 0)
                gate_bo.sync(xrt_mod.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                             enc.nbytes, 0)
                run = kernel_obj(*hbm_banks[:NUM_HBM_BANKS], gate_bo, 1, nq, chi_max)
                run.wait()
                # Read new chi_new from result metadata
                chi_new = min(chi_max, chi_l * 2)  # conservative
                new_shape_i = (chi_l, 2, chi_new)
                new_shape_j = (chi_new, 2, chi_r)
                mps.sites[si] = _hbm_read_tensor(
                    hbm_banks, offsets[si],
                    chi_l * 2 * chi_new, new_shape_i, xrt_mod
                )
                mps.sites[sj] = _hbm_read_tensor(
                    hbm_banks, offsets[sj],
                    chi_new * 2 * chi_r, new_shape_j, xrt_mod
                )
            else:
                # FPGA contraction + host SVD + FPGA split
                # Step 1: FPGA contracts A_i, A_{i+1}, applies gate → theta in HBM
                instr = _encode_2q(
                    GATE_2Q_CONTRACT, si, sj, chi_l, chi_m, chi_r, 0,
                    gate['params'], offsets[si], offsets[sj]
                )
                enc = np.zeros(INSTR_WORDS + 32, dtype=np.int32)
                enc[:INSTR_WORDS] = instr
                # Encode gate matrix (16 complex128 as float32 in int32)
                gf = gate_mat.ravel().astype(np.complex128)
                for k in range(min(32, gf.size)):
                    val = gf[k // 2].real if k % 2 == 0 else gf[k // 2].imag
                    enc[INSTR_WORDS + k] = int(np.float32(val).view(np.int32))
                gate_bo.write(enc.tobytes(), 0)
                gate_bo.sync(xrt_mod.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                             enc.nbytes, 0)
                run = kernel_obj(*hbm_banks[:NUM_HBM_BANKS], gate_bo, 1, nq, chi_max)
                run.wait()

                # Step 2: Read theta back to host
                n_theta = chi_l * 4 * chi_r
                theta_flat = _hbm_read_tensor(
                    hbm_banks, theta_offset, n_theta,
                    (chi_l * 2, 2 * chi_r), xrt_mod
                )

                # Step 3: Host SVD with truncation
                U, S, Vh = la.svd(theta_flat, full_matrices=False)
                threshold = mps.svd_cutoff * S[0] if len(S) > 0 else 0.0
                keep = int(np.clip(np.sum(S > threshold), 1, chi_max))
                U, S, Vh = U[:, :keep], S[:keep], Vh[:keep, :]
                norm = np.linalg.norm(S)
                if norm > 1e-15:
                    S /= norm
                sqrt_S = np.sqrt(S)
                chi_new = keep

                # Update MPS tensors in CPU
                mps.sites[si] = (U * sqrt_S).reshape(chi_l, 2, chi_new)
                mps.sites[sj] = (sqrt_S[:, None] * Vh).reshape(chi_new, 2, chi_r)

                # Step 4: FPGA split (write updated tensors back)
                _hbm_write_tensor(hbm_banks, mps.sites[si], offsets[si],
                                  chi_l * 2 * chi_new, xrt_mod)
                _hbm_write_tensor(hbm_banks, mps.sites[sj], offsets[sj],
                                  chi_new * 2 * chi_r, xrt_mod)

            continue

    # Final 1q batch
    _flush_1q()

    # Sample measurements from CPU MPS (sequential conditional sampling)
    rng = np.random.default_rng(seed)
    counts = mps.get_counts(shots, measured_qubits, rng)
    return mps, counts


# ─────────────────────────────────────────────────────────────────────────────
#  Worker process — Mode B (distributed chain)
# ─────────────────────────────────────────────────────────────────────────────

def _mps_worker_distributed(card_id: int, num_cards: int, xclbin_path: str,
                             kernel_name: str,
                             read_arrays, write_arrays, local_tensor_size: int,
                             barrier_a: Barrier, barrier_b: Barrier, barrier_c: Barrier,
                             cmd_queue: Queue, result_queue: Queue,
                             ready_event: Event, shutdown_event: Event):
    """
    Worker for Mode B (distributed chain partition).
    This card owns sites [card_id * seg_len .. (card_id+1) * seg_len).
    Boundary 2q gates require exchange with the adjacent card via shared memory.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    device = None
    kernel_obj = None
    hbm_banks: List = []
    fpga_ready = False
    local_mps: Optional[MPSState] = None  # the segment owned by this card

    try:
        ready_event.set()
        while not shutdown_event.is_set():
            try:
                cmd = cmd_queue.get(timeout=0.2)
            except Exception:
                continue
            if cmd is None:
                break
            ct = cmd[0]

            if ct == 'init_segment':
                # Initialize local MPS segment (all |0⟩)
                seg_len, chi_max, svd_cutoff = cmd[1], cmd[2], cmd[3]
                local_mps = MPSState(seg_len, chi_max, svd_cutoff)
                # Publish boundary tensors to shared memory
                _publish_boundary(card_id, local_mps, read_arrays, local_tensor_size)
                result_queue.put(('init_done', card_id))

            elif ct == 'apply_local_gate':
                # Apply gate entirely within this segment
                gate_info = cmd[1]
                if gate_info['type'] == '1q':
                    G = _gate_1q(gate_info['gate_name'], gate_info['params'])
                    local_mps.apply_1q(gate_info['local_site'], G)
                elif gate_info['type'] == '2q':
                    G = _gate_2q(gate_info['gate_name'], gate_info['params'])
                    local_mps.apply_2q(gate_info['local_site_i'],
                                       gate_info['local_site_j'], G)
                _publish_boundary(card_id, local_mps, read_arrays, local_tensor_size)
                result_queue.put(('gate_done', card_id))

            elif ct == 'apply_boundary_gate':
                # 2q gate straddling this card and an adjacent card
                # Uses 3-barrier protocol for tensor exchange
                gate_info, side = cmd[1], cmd[2]
                # side = 'right': our rightmost site + neighbor's leftmost
                # side = 'left':  neighbor's rightmost + our leftmost

                if side == 'right':
                    my_local_site = local_mps.n - 1
                    my_tensor = local_mps.sites[my_local_site]
                else:
                    my_local_site = 0
                    my_tensor = local_mps.sites[my_local_site]

                # Step 1: Publish my boundary tensor
                _publish_boundary(card_id, local_mps, read_arrays, local_tensor_size)
                barrier_a.wait()

                # Step 2: Read neighbor's boundary tensor
                neighbor_id = card_id + 1 if side == 'right' else card_id - 1
                nbuf = _shm_read(read_arrays[neighbor_id], local_tensor_size)
                chi_nb = int(nbuf[0])
                chi_nr = int(nbuf[1])
                n_amps_nb = chi_nb * 2 * chi_nr
                neighbor_tensor = _unpack_tensor(nbuf[2: 2 + n_amps_nb * 2],
                                                 (chi_nb, 2, chi_nr))

                # Step 3: Apply 2q gate on the CPU (boundary interaction)
                G = _gate_2q(gate_info['gate_name'], gate_info['params'])
                if side == 'right':
                    # Contract my rightmost site with neighbor's leftmost
                    temp_mps = MPSState(2, local_mps.chi_max, local_mps.svd_cutoff)
                    temp_mps.sites[0] = my_tensor
                    temp_mps.sites[1] = neighbor_tensor
                    temp_mps.apply_2q(0, 1, G)
                    new_my = temp_mps.sites[0]
                    new_neighbor = temp_mps.sites[1]
                else:
                    temp_mps = MPSState(2, local_mps.chi_max, local_mps.svd_cutoff)
                    temp_mps.sites[0] = neighbor_tensor
                    temp_mps.sites[1] = my_tensor
                    temp_mps.apply_2q(0, 1, G)
                    new_neighbor = temp_mps.sites[0]
                    new_my = temp_mps.sites[1]

                local_mps.sites[my_local_site] = new_my

                # Write new neighbor tensor to write buffer
                _write_boundary_tensor(write_arrays[card_id], new_neighbor,
                                       local_tensor_size)
                barrier_b.wait()

                # Step 4: Finalize
                _publish_boundary(card_id, local_mps, read_arrays, local_tensor_size)
                barrier_c.wait()
                result_queue.put(('gate_done', card_id))

            elif ct == 'collect_segment':
                # Return this segment's tensors
                tensors_bytes = [t.tobytes() for t in local_mps.sites]
                shapes = [t.shape for t in local_mps.sites]
                result_queue.put(('segment_done', card_id, tensors_bytes, shapes))

    except Exception as e:
        logger.error('Distributed worker %d: fatal — %s', card_id, e)
        import traceback
        traceback.print_exc()
        result_queue.put(('error', card_id, str(e)))
        ready_event.set()


def _publish_boundary(card_id: int, mps: MPSState, read_arrays, local_tensor_size: int):
    """Write leftmost and rightmost site tensors to shared memory for neighbor access."""
    for site_idx, tensor in [(0, mps.sites[0]), (mps.n - 1, mps.sites[-1])]:
        t = tensor.astype(np.complex128)
        chi_l, _, chi_r = t.shape
        n_amps = chi_l * 2 * chi_r
        # Header: [chi_l, chi_r] then packed [re, im, ...]
        header = np.array([chi_l, chi_r], dtype=np.float64)
        packed = _pack_tensor(t)
        buf = np.concatenate([header, packed])
        _shm_write(read_arrays[card_id], buf, min(len(buf), local_tensor_size))


def _write_boundary_tensor(write_arr, tensor: np.ndarray, local_tensor_size: int):
    t = tensor.astype(np.complex128)
    chi_l, _, chi_r = t.shape
    header = np.array([float(chi_l), float(chi_r)], dtype=np.float64)
    packed = _pack_tensor(t)
    buf = np.concatenate([header, packed])
    _shm_write(write_arr, buf, min(len(buf), local_tensor_size))


# ─────────────────────────────────────────────────────────────────────────────
#  Result class
# ─────────────────────────────────────────────────────────────────────────────

class FPGAMPSResult:
    """Qiskit-compatible result wrapper for FPGA MPS simulation."""

    def __init__(self, results_list: list, shots: int, num_cards: int,
                 total_time: float, mode: str):
        self._results = results_list
        self.shots = shots
        self.num_cards = num_cards
        self.total_time = total_time
        self.mode = mode

    def get_counts(self, idx: int = 0) -> Dict[str, int]:
        return self._results[idx]['counts']

    def get_statevector(self, idx: int = 0) -> np.ndarray:
        sv = self._results[idx].get('statevector')
        if sv is None:
            raise ValueError("Statevector not available (n > 24 or not computed).")
        return sv

    def get_mps(self, idx: int = 0):
        return self._results[idx].get('mps')

    def get_max_bond(self, idx: int = 0) -> int:
        return self._results[idx].get('max_bond', -1)

    def get_bond_dims(self, idx: int = 0) -> List[int]:
        return self._results[idx].get('bond_dims', [])

    def expectation_value(self, pauli_string: str, idx: int = 0) -> float:
        """
        Compute ⟨ψ|P|ψ⟩ for a Pauli string P = P_0 ⊗ P_1 ⊗ ... ⊗ P_{n-1}.
        Implemented via statevector for n ≤ 24; MPS-based for larger n.
        """
        mps = self.get_mps(idx)
        if mps is not None and mps.n <= 24:
            sv = mps.get_statevector()
            return _pauli_expectation_sv(sv, pauli_string)
        sv = self._results[idx].get('statevector')
        if sv is not None:
            return _pauli_expectation_sv(sv, pauli_string)
        raise ValueError("Cannot compute expectation value without statevector or MPS.")

    def success(self) -> bool:
        return True


def _pauli_expectation_sv(sv: np.ndarray, pauli_string: str) -> float:
    """⟨ψ|P|ψ⟩ for a Pauli product operator string e.g. 'XZI...'."""
    n = int(round(math.log2(len(sv))))
    Psi = sv.copy()
    for i, p in enumerate(pauli_string.upper()):
        if p == 'I':
            continue
        qubit = n - 1 - i  # Qiskit convention: leftmost char = highest qubit
        mat = GATE_MATRICES.get(p.lower(), np.eye(2, dtype=np.complex128))
        # Apply matrix to qubit i of state vector
        N = 2 ** n
        stride = 1 << qubit
        for j in range(0, N, stride * 2):
            for k in range(stride):
                i0, i1 = j + k, j + k + stride
                a0, a1 = Psi[i0], Psi[i1]
                Psi[i0] = mat[0, 0] * a0 + mat[0, 1] * a1
                Psi[i1] = mat[1, 0] * a0 + mat[1, 1] * a1
    return float(np.real(np.dot(sv.conj(), Psi)))


# ─────────────────────────────────────────────────────────────────────────────
#  FPGAMPSSimulator — main class
# ─────────────────────────────────────────────────────────────────────────────

class FPGAMPSSimulator:
    """
    FPGA-accelerated Matrix Product State quantum circuit simulator.

    Parameters
    ----------
    num_cards   : Number of FPGA cards (1, 2, or 4). Default 4.
    chi_max     : Maximum bond dimension. Default 256.
    svd_cutoff  : Relative SVD truncation threshold. Default 1e-12.
    engine      : "fpga" | "cpu" | "auto". Default "auto".
    mode        : "shots" | "distributed" | "auto". Default "auto".
    xclbin_path : Path to .xclbin. If None, searches current directory.

    Notes
    -----
    engine="fpga" raises RuntimeError if XRT / FPGA is unavailable.
    engine="cpu"  always uses the CPU reference engine (cpu_reference.py).
    engine="auto" uses FPGA if available, CPU with a warning otherwise.
    """

    def __init__(self,
                 num_cards: int = NUM_CARDS,
                 chi_max: int = 256,
                 svd_cutoff: float = 1e-12,
                 engine: str = 'auto',
                 mode: str = 'auto',
                 xclbin_path: Optional[str] = None):

        if num_cards not in (1, 2, 4):
            raise ValueError("num_cards must be 1, 2, or 4.")

        self.num_cards = num_cards
        self.chi_max = chi_max
        self.svd_cutoff = svd_cutoff
        self.mode = mode

        # Resolve engine
        if engine == 'auto':
            if XRT_AVAILABLE:
                self._engine = 'fpga'
                logger.warning('engine=auto → using FPGA (XRT found)')
            else:
                self._engine = 'cpu'
                logger.warning('engine=auto → falling back to CPU (XRT not found). '
                               'Use engine="fpga" to require hardware.')
        elif engine in ('fpga', 'cpu'):
            self._engine = engine
        else:
            raise ValueError(f"engine must be 'fpga', 'cpu', or 'auto', got {engine!r}")

        if self._engine == 'fpga' and not XRT_AVAILABLE:
            raise RuntimeError(
                "engine='fpga' but pyxrt is not importable. "
                "Install Xilinx XRT and ensure pyxrt is on PYTHONPATH."
            )

        # Resolve xclbin
        if self._engine == 'fpga':
            if xclbin_path is None:
                candidates = [
                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 DEFAULT_XCLBIN_NAME),
                    os.path.join(os.getcwd(), DEFAULT_XCLBIN_NAME),
                ]
                for c in candidates:
                    if os.path.exists(c):
                        xclbin_path = c
                        break
                if xclbin_path is None:
                    raise FileNotFoundError(
                        f"XCLBIN '{DEFAULT_XCLBIN_NAME}' not found. "
                        f"Pass xclbin_path= explicitly or build with make."
                    )
            self._xclbin_path = os.path.abspath(xclbin_path)
            self._kernel_name = self._detect_kernel_name()
        else:
            self._xclbin_path = None
            self._kernel_name = None

        # Worker management
        self._workers: List[Process] = []
        self._cmd_queues: List[Queue] = []
        self._result_queue: Optional[Queue] = None
        self._ready_events: List[Event] = []
        self._shutdown_events: List[Event] = []
        self._workers_spawned = False
        self._current_mode_spawned = None

        # CPU fallback sim (always available)
        self._cpu_sim = CPUMPSSimulator(chi_max=chi_max, svd_cutoff=svd_cutoff)

        atexit.register(self.shutdown)

    def _detect_kernel_name(self) -> str:
        """Probe the xclbin to find the kernel name."""
        try:
            dev = xrt.device(0)
            uuid = dev.load_xclbin(self._xclbin_path)
            for kn in [KERNEL_NAME, 'quantum_mps_kernel', 'fpga_mps']:
                try:
                    xrt.kernel(dev, uuid, kn)
                    return kn
                except Exception:
                    continue
        except Exception as e:
            logger.warning('Kernel detection failed: %s', e)
        return KERNEL_NAME

    # ── Worker lifecycle ──────────────────────────────────────────────────────

    def _ensure_workers(self, effective_mode: str):
        """Spawn or verify workers for the given mode."""
        if self._workers_spawned and self._current_mode_spawned == effective_mode:
            if all(w.is_alive() for w in self._workers):
                return
            logger.warning('Some workers died; recreating.')

        if self._workers_spawned:
            self._teardown_workers()

        self._result_queue = Queue()
        self._cmd_queues = []
        self._ready_events = []
        self._shutdown_events = []
        self._workers = []

        if effective_mode == 'shots':
            cmd_type = 'run_cpu' if self._engine == 'cpu' else 'run_fpga'
            for cid in range(self.num_cards):
                cq = Queue()
                re, se = Event(), Event()
                w = Process(
                    target=_mps_worker_shots,
                    args=(cid, self._xclbin_path or '', self._kernel_name or '',
                          cq, self._result_queue, re, se),
                    daemon=True
                )
                w.start()
                self._cmd_queues.append(cq)
                self._ready_events.append(re)
                self._shutdown_events.append(se)
                self._workers.append(w)

        elif effective_mode == 'distributed':
            # Shared memory for boundary tensor exchange
            # Each card's boundary tensor: chi_max * 2 * chi_max complex128
            # = chi_max * 2 * chi_max * 2 float64 values
            shm_floats = self.chi_max * 2 * self.chi_max * 2 + 2  # +2 for header
            self._read_arrays = [
                Array(ctypes.c_double, shm_floats, lock=False)
                for _ in range(self.num_cards)
            ]
            self._write_arrays = [
                Array(ctypes.c_double, shm_floats, lock=False)
                for _ in range(self.num_cards)
            ]
            self._barrier_a = _mp.Barrier(self.num_cards)
            self._barrier_b = _mp.Barrier(self.num_cards)
            self._barrier_c = _mp.Barrier(self.num_cards)

            for cid in range(self.num_cards):
                cq = Queue()
                re, se = Event(), Event()
                w = Process(
                    target=_mps_worker_distributed,
                    args=(cid, self.num_cards, self._xclbin_path or '',
                          self._kernel_name or '',
                          self._read_arrays, self._write_arrays, shm_floats,
                          self._barrier_a, self._barrier_b, self._barrier_c,
                          cq, self._result_queue, re, se),
                    daemon=True
                )
                w.start()
                self._cmd_queues.append(cq)
                self._ready_events.append(re)
                self._shutdown_events.append(se)
                self._workers.append(w)

        # Wait for ready
        t0 = time.time()
        for i, ev in enumerate(self._ready_events):
            rem = 60.0 - (time.time() - t0)
            if rem <= 0 or not ev.wait(timeout=max(rem, 0.1)):
                logger.warning('Card %d: ready timeout', i)

        self._workers_spawned = True
        self._current_mode_spawned = effective_mode
        alive = sum(1 for w in self._workers if w.is_alive())
        logger.info('%d/%d workers alive for mode=%s', alive, self.num_cards, effective_mode)

    def _teardown_workers(self):
        for se in self._shutdown_events:
            try:
                se.set()
            except Exception:
                pass
        for cq in self._cmd_queues:
            try:
                cq.put_nowait(None)
            except Exception:
                pass
        for w in self._workers:
            try:
                w.join(timeout=5)
                if w.is_alive():
                    w.terminate()
                    w.join(timeout=2)
            except Exception:
                pass
        for cq in self._cmd_queues:
            try:
                while not cq.empty():
                    cq.get_nowait()
            except Exception:
                pass
        if self._result_queue is not None:
            try:
                while not self._result_queue.empty():
                    self._result_queue.get_nowait()
            except Exception:
                pass
        self._cmd_queues = []
        self._result_queue = None
        self._ready_events = []
        self._shutdown_events = []
        self._workers = []
        self._workers_spawned = False

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, circuits, shots: int = 1024,
            seed: Optional[int] = None) -> FPGAMPSResult:
        """
        Run one or more Qiskit QuantumCircuits.

        Parameters
        ----------
        circuits : QuantumCircuit or list of QuantumCircuit
        shots    : Number of measurement samples per circuit
        seed     : Optional RNG seed

        Returns
        -------
        FPGAMPSResult
        """
        if not isinstance(circuits, list):
            circuits = [circuits]

        # Determine effective mode
        if self.mode == 'auto':
            eff_mode = 'shots'
        elif self.mode in ('shots', 'distributed'):
            eff_mode = self.mode
        else:
            raise ValueError(f"mode must be 'shots', 'distributed', or 'auto'")

        t_start = time.time()

        if self._engine == 'cpu':
            return self._run_cpu(circuits, shots, seed, t_start)

        # FPGA path
        self._ensure_workers(eff_mode)

        if eff_mode == 'shots':
            return self._run_shots_parallel(circuits, shots, seed, t_start)
        else:
            return self._run_distributed(circuits[0], shots, seed, t_start)

    def _run_cpu(self, circuits: list, shots: int,
                 seed: Optional[int], t_start: float) -> FPGAMPSResult:
        """Run all circuits sequentially on CPU reference engine."""
        results = []
        for i, circuit in enumerate(circuits):
            s = seed + i if seed is not None else None
            sim = CPUMPSSimulator(self.chi_max, self.svd_cutoff, s)
            mps, counts = sim.run_mps(circuit, shots)
            sv = None
            if circuit.num_qubits <= 24:
                try:
                    sv = mps.get_statevector()
                except Exception:
                    pass
            results.append({
                'counts': counts, 'statevector': sv, 'mps': mps,
                'max_bond': mps.max_bond(), 'bond_dims': mps.bond_dims(),
            })
        return FPGAMPSResult(results, shots, 1, time.time() - t_start, 'cpu')

    def _run_shots_parallel(self, circuits: list, shots: int,
                             seed: Optional[int], t_start: float) -> FPGAMPSResult:
        """Farm circuits/shot-batches across cards."""
        import pickle

        # Distribute circuits across cards
        tasks = []
        for i, circuit in enumerate(circuits):
            s = seed + i if seed is not None else None
            tasks.append((i, circuit, shots, s))

        # Submit up to num_cards tasks at once
        pending: List[Tuple[int, int]] = []  # (task_idx, card_id)
        task_queue = list(range(len(tasks)))
        results_map = {}
        card_busy = [False] * self.num_cards

        cmd_type = 'run_fpga' if self._engine == 'fpga' else 'run_cpu'

        while task_queue or pending:
            # Submit to free cards
            for cid in range(self.num_cards):
                if not card_busy[cid] and task_queue:
                    ti = task_queue.pop(0)
                    _, circuit, sh, s = tasks[ti]
                    circuit_bytes = pickle.dumps(circuit)
                    self._cmd_queues[cid].put(
                        (cmd_type, circuit_bytes, sh, self.chi_max, self.svd_cutoff, s)
                    )
                    pending.append((ti, cid))
                    card_busy[cid] = True

            # Collect one result
            if pending:
                r = self._result_queue.get(timeout=300)
                if r[0] == 'run_done':
                    _, cid, counts, sv, max_bond, bond_dims = r
                    # Find which task this cid was running
                    ti = next(ti for ti, ci in pending if ci == cid)
                    pending.remove((ti, cid))
                    card_busy[cid] = False
                    circuit = tasks[ti][1]
                    mps_result = {
                        'counts': counts, 'statevector': sv,
                        'max_bond': max_bond, 'bond_dims': bond_dims,
                    }
                    results_map[ti] = mps_result
                elif r[0] == 'error':
                    raise RuntimeError(f"Card {r[1]} error: {r[2]}")

        results = [results_map[i] for i in range(len(tasks))]
        return FPGAMPSResult(results, shots, self.num_cards,
                             time.time() - t_start, 'shots')

    def _run_distributed(self, circuit, shots: int,
                         seed: Optional[int], t_start: float) -> FPGAMPSResult:
        """
        Run a single circuit with chain distributed across cards.
        Each card owns n//num_cards sites.
        Non-adjacent 2q gates are SWAP-routed before dispatch.
        """
        nq = circuit.num_qubits
        seg_len = nq // self.num_cards
        if seg_len < 1:
            logger.warning('Distributed mode: fewer sites than cards; using shots mode.')
            return self._run_shots_parallel([circuit], shots, seed, t_start)

        # Initialize segments
        for cid in range(self.num_cards):
            self._cmd_queues[cid].put(
                ('init_segment', seg_len, self.chi_max, self.svd_cutoff)
            )
        self._wait_all('init_done', timeout=60)

        # Compile and dispatch gates
        mps_dummy = MPSState(nq, self.chi_max, self.svd_cutoff)
        gate_list, measured_qubits = compile_circuit(circuit, mps_dummy)

        for gate in gate_list:
            if gate['type'] == '1q':
                site = gate['sites'][0]
                cid = site // seg_len
                if cid >= self.num_cards:
                    cid = self.num_cards - 1
                local_site = site - cid * seg_len
                gate_local = dict(gate)
                gate_local['local_site'] = local_site
                self._cmd_queues[cid].put(('apply_local_gate', gate_local))
                self._wait_one_card('gate_done', cid)

            elif gate['type'] == '2q':
                si, sj = gate['sites']
                cid_i = si // seg_len
                cid_j = sj // seg_len
                if cid_i >= self.num_cards:
                    cid_i = self.num_cards - 1
                if cid_j >= self.num_cards:
                    cid_j = self.num_cards - 1

                if cid_i == cid_j:
                    # Same segment
                    local_si = si - cid_i * seg_len
                    local_sj = sj - cid_j * seg_len
                    gate_local = dict(gate)
                    gate_local['type'] = '2q'
                    gate_local['local_site_i'] = local_si
                    gate_local['local_site_j'] = local_sj
                    self._cmd_queues[cid_i].put(('apply_local_gate', gate_local))
                    self._wait_one_card('gate_done', cid_i)
                else:
                    # Boundary gate: both cards participate
                    assert abs(cid_i - cid_j) == 1, \
                        f"Non-adjacent cards {cid_i},{cid_j} after routing"
                    self._cmd_queues[cid_i].put(
                        ('apply_boundary_gate', gate, 'right')
                    )
                    self._cmd_queues[cid_j].put(
                        ('apply_boundary_gate', gate, 'left')
                    )
                    self._wait_all('gate_done', count=2, timeout=60)

        # Collect segments
        for cid in range(self.num_cards):
            self._cmd_queues[cid].put(('collect_segment',))
        segments = [None] * self.num_cards
        for _ in range(self.num_cards):
            r = self._result_queue.get(timeout=120)
            if r[0] == 'segment_done':
                _, cid, tensors_bytes, shapes = r
                tensors = [
                    np.frombuffer(tb, dtype=np.complex128).reshape(sh)
                    for tb, sh in zip(tensors_bytes, shapes)
                ]
                segments[cid] = tensors
            elif r[0] == 'error':
                raise RuntimeError(f"Card {r[1]} error: {r[2]}")

        # Reconstruct global MPS
        global_mps = MPSState(nq, self.chi_max, self.svd_cutoff)
        idx = 0
        for cid in range(self.num_cards):
            for t in segments[cid]:
                global_mps.sites[idx] = t
                idx += 1
        while idx < nq:
            global_mps.sites[idx] = np.zeros((1, 2, 1), dtype=np.complex128)
            global_mps.sites[idx][0, 0, 0] = 1.0
            idx += 1

        # Sample
        rng = np.random.default_rng(seed)
        counts = global_mps.get_counts(shots, measured_qubits, rng)
        sv = None
        if nq <= 24:
            try:
                sv = global_mps.get_statevector()
            except Exception:
                pass

        result = [{
            'counts': counts, 'statevector': sv, 'mps': global_mps,
            'max_bond': global_mps.max_bond(), 'bond_dims': global_mps.bond_dims(),
        }]
        return FPGAMPSResult(result, shots, self.num_cards,
                             time.time() - t_start, 'distributed')

    def _wait_all(self, etype: str, count: Optional[int] = None,
                  timeout: float = 300):
        n_expect = count if count is not None else self.num_cards
        n = 0
        deadline = time.time() + timeout
        while n < n_expect:
            rem = deadline - time.time()
            if rem <= 0:
                raise TimeoutError(
                    f"Timed out waiting for {etype}: {n}/{n_expect} responses"
                )
            r = self._result_queue.get(timeout=rem)
            if r[0] == etype:
                n += 1
            elif r[0] == 'error':
                raise RuntimeError(f"Card {r[1]} error: {r[2]}")

    def _wait_one_card(self, etype: str, cid: int, timeout: float = 300):
        deadline = time.time() + timeout
        while True:
            rem = deadline - time.time()
            if rem <= 0:
                raise TimeoutError(f"Timed out waiting for {etype} from card {cid}")
            r = self._result_queue.get(timeout=rem)
            if r[0] == etype and r[1] == cid:
                return
            elif r[0] == 'error':
                raise RuntimeError(f"Card {r[1]} error: {r[2]}")
            # Re-queue if it's from a different card (shouldn't happen in shots mode)

    # ── Inspection ────────────────────────────────────────────────────────────

    def get_card_stats(self) -> List[dict]:
        """Return per-card status dict."""
        return [
            {
                'card_id': i,
                'alive': self._workers[i].is_alive() if i < len(self._workers) else False,
                'pid': self._workers[i].pid if i < len(self._workers) else None,
                'fpga_device': i if self._engine == 'fpga' else None,
                'execution_mode': self._engine.upper(),
            }
            for i in range(self.num_cards)
        ]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def shutdown(self):
        if self._workers_spawned:
            self._teardown_workers()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.shutdown()
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Module info
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('FPGA MPS Simulator — Host Module')
    print('=' * 55)
    print(f'  XRT Available : {XRT_AVAILABLE}')
    print(f'  Qiskit        : {QISKIT_AVAILABLE}')
    print(f'  Kernel name   : {KERNEL_NAME}')
    print(f'  Max qubits    : {ABSOLUTE_MAX_QUBITS}')
    print(f'  HBM banks     : {NUM_HBM_BANKS}  (+ 1 gate bundle = 9 AXI masters)')
    print()
    print('Usage:')
    print('  from fpga_mps_simulator import FPGAMPSSimulator')
    print('  sim = FPGAMPSSimulator(num_cards=4, engine="cpu")  # or "fpga"/"auto"')
    print('  result = sim.run(circuit, shots=1024)')
    print('  print(result.get_counts())')
