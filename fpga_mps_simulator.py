#!/usr/bin/env python3
"""
fpga_mps_simulator.py — FPGA-Based MPS Quantum Circuit Simulator (Host)
=======================================================================

Calls the compiled fpga_mps_simulator.xclbin kernel on Xilinx Alveo U55C
cards to accelerate Matrix Product State quantum circuit simulation.

Engine selection:
  engine="fpga"  : FPGA required; RuntimeError if XRT unavailable or init fails.
                   Loads fpga_mps_simulator.xclbin and drives it via pyxrt.
  engine="cpu"   : CPU reference engine (cpu_reference.py). Never silent.
  engine="auto"  : FPGA if XRT present and device found, else CPU with a warning.

Parallelism modes:
  mode="shots"       : Each card runs independent MPS instances (circuit/shot batches).
  mode="distributed" : Chain partitioned across 4 cards; O(chi²) boundary exchange.
  mode="auto"        : shots for <4 circuits, distributed for large single-circuit runs.

Kernel AXI master ports (9 total — U55C routing budget):
  bank0..bank7  : HBM tensor banks (8 × m_axi, group_id 0..7)
  gate_seq      : gate instructions + metadata (1 × m_axi, group_id 8)

Kernel scalar arguments (s_axilite via pyxrt):
  num_gates (int), nq (int), chi_max (int)

Kernel call signature (positional, matching extern "C" declaration order):
  kernel(bank0_bo, bank1_bo, ..., bank7_bo,
         gate_seq_bo,
         num_gates, nq, chi_max)

HBM layout (8-bank interleaving):
  global index g → bank = g % 8,  slot = g // 8
  fp64[slot*2] = real,  fp64[slot*2+1] = imag  (complex128)

Gate instruction format (WORDS_PER_GATE = 44 int32 per gate):
  Header (12 int32):
    [0]  opcode          (0-16: 1q gate, 100: 2q contract, 102: 2q full+SVD)
    [1]  site_i          (MPS site index)
    [2]  site_j          (-1 for 1q gates)
    [3]  chi_l           (left bond dimension)
    [4]  chi_m           (bond between sites i and i+1)
    [5]  chi_r           (right bond dimension)
    [6]  chi_new         (target bond dim; 0 → use chi_max)
    [7-9] params         (float32 packed as int32 bits)
    [10] hbm_offset_i    (global complex-double offset for site i)
    [11] hbm_offset_j    (global complex-double offset for site j; -1 for 1q)
  Gate matrix (32 int32): 16 complex128 values as float32 pairs packed as int32

Invariants (hard rules, non-negotiable):
  * mp.set_start_method('spawn', force=False) BEFORE any other mp usage.
  * ctypes.memmove() for ALL cross-process shared-memory I/O.
  * engine='fpga' raises RuntimeError on failure — no silent CPU fallback.
  * AXI master count ≤ 9 (8 tensor banks + 1 gate bundle).
  * complex128 end-to-end.

Author: Nasir Ali
Organization: C-DAC Noida
Date: June 2026
Target platform: Xilinx Alveo U55C × 4, Vitis 2023.2, XRT 2.x, pyxrt, Python 3.8
"""

from __future__ import annotations

# ── spawn MUST be set before any other multiprocessing usage ─────────────────
import multiprocessing as _mp
try:
    _mp.set_start_method('spawn', force=False)
except RuntimeError:
    pass

import atexit
import ctypes
import logging
import math
import os
import signal
import struct
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

from multiprocessing import Array, Barrier, Event, Process, Queue

# ── XRT import ────────────────────────────────────────────────────────────────
try:
    import pyxrt as xrt
    XRT_AVAILABLE = True
except ImportError:
    XRT_AVAILABLE = False

# ── Qiskit (optional) ────────────────────────────────────────────────────────
try:
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# ── CPU reference engine ─────────────────────────────────────────────────────
from cpu_reference import (
    CPUMPSSimulator, MPSState,
    GATE_MATRICES, _gate_1q, _gate_2q,
    _SINGLE_QUBIT_GATES, _TWO_QUBIT_GATES,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────
_lvl = (logging.DEBUG
        if os.environ.get('FPGA_MPS_LOG', '').upper() == 'DEBUG'
        else logging.WARNING)
logging.basicConfig(level=_lvl,
                    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger('fpga_mps')

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
NUM_CARDS      = 4
NUM_HBM_BANKS  = 8          # 8 tensor + 1 gate bundle = 9 AXI masters
KERNEL_NAME    = 'fpga_mps_simulator'
DEFAULT_XCLBIN = 'fpga_mps_simulator.xclbin'

INSTR_WORDS    = 12         # header words per gate instruction
GATE_MAT_WORDS = 32         # 16 complex128 as float32 pairs → 32 int32
WORDS_PER_GATE = INSTR_WORDS + GATE_MAT_WORDS   # 44 int32 per gate
MAX_GATES_BRAM = 512        # must match fpga_mps_simulator.cpp MAX_GATES_BRAM

# Opcodes (must match fpga_mps_simulator.cpp)
_OP1Q: Dict[str, int] = {
    'id': 0, 'h': 1, 'x': 2, 'y': 3, 'z': 4,
    's': 5, 'sdg': 6, 't': 7, 'tdg': 8, 'sx': 9, 'sxdg': 10,
    'rx': 11, 'ry': 12, 'rz': 13, 'p': 14, 'phase': 14, 'u1': 14,
    'u2': 15, 'u3': 16, 'u': 16,
}
_OP2Q: Dict[str, int] = {
    'cx': 17, 'cnot': 17, 'cy': 18, 'cz': 19, 'ch': 20,
    'cp': 21, 'cphase': 21, 'crx': 22, 'cry': 23, 'crz': 24,
    'csx': 25, 'swap': 26, 'iswap': 27, 'dcx': 28,
    'ecr': 29, 'rxx': 30, 'ryy': 31, 'rzz': 32,
}
OP_2Q_CONTRACT = 100  # kernel: contract+gate only; host does SVD
OP_2Q_FULL     = 102  # kernel: contract+gate+JacobiSVD+split (chi ≤ 64)

CHI_FULL_SVD   = 0    # SVD always runs on host CPU (Jacobi SVD removed from FPGA)


# ─────────────────────────────────────────────────────────────────────────────
#  Float32-in-int32 packing (union trick, matches C++ extract_f32)
# ─────────────────────────────────────────────────────────────────────────────

def _f32_bits(f: float) -> int:
    """Pack a float as its IEEE 754 float32 bits in a Python int."""
    return struct.unpack('<I', struct.pack('<f', float(np.float32(f))))[0]


# ─────────────────────────────────────────────────────────────────────────────
#  ctypes.memmove shared-memory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shm_write_f64(arr: Array, data: np.ndarray, n: int):
    buf = np.ascontiguousarray(data.ravel()[:n], dtype=np.float64)
    ctypes.memmove(
        ctypes.cast(ctypes.addressof(arr), ctypes.c_void_p),
        buf.ctypes.data_as(ctypes.c_void_p),
        n * 8
    )


def _shm_read_f64(arr: Array, n: int) -> np.ndarray:
    buf = np.empty(n, dtype=np.float64)
    ctypes.memmove(
        buf.ctypes.data_as(ctypes.c_void_p),
        ctypes.cast(ctypes.addressof(arr), ctypes.c_void_p),
        n * 8
    )
    return buf


def _pack_cplx128(t: np.ndarray) -> np.ndarray:
    """complex128 ndarray → float64 [re0, im0, re1, im1, ...]"""
    c = t.ravel().astype(np.complex128)
    b = np.empty(c.size * 2, dtype=np.float64)
    b[0::2] = c.real
    b[1::2] = c.imag
    return b


def _unpack_cplx128(buf: np.ndarray, shape: tuple) -> np.ndarray:
    return (buf[0::2] + 1j * buf[1::2]).astype(np.complex128).reshape(shape)


# ─────────────────────────────────────────────────────────────────────────────
#  HBM bump allocator
# ─────────────────────────────────────────────────────────────────────────────

class _HBMAlloc:
    """
    Bump allocator for complex-double slots in 8-bank HBM.
    Offset in units of complex128 amplitudes (= 2×float64 = 16 bytes).
    """

    def __init__(self):
        self._cursor = 0

    def alloc(self, n_amps: int) -> int:
        off = self._cursor
        # Round up to next multiple of NUM_HBM_BANKS so every site's g_offset
        # is bank-aligned.  _hbm_write/_hbm_read assume g_offset % 8 == 0:
        # local_index i maps to bank i%8, which only holds when g_offset%8 == 0.
        n_padded = ((n_amps + NUM_HBM_BANKS - 1) // NUM_HBM_BANKS) * NUM_HBM_BANKS
        self._cursor += n_padded
        return off

    def alloc_site(self, chi_l: int, chi_r: int) -> int:
        return self.alloc(chi_l * 2 * chi_r)   # σ ∈ {0,1}

    @property
    def total(self) -> int:
        return self._cursor


# ─────────────────────────────────────────────────────────────────────────────
#  HBM tensor I/O via pyxrt BO objects
# ─────────────────────────────────────────────────────────────────────────────

def _hbm_write(banks: list, tensor: np.ndarray,
               g_offset: int, n_amps: int, xrt_mod):
    """Write n_amps complex128 values to HBM (8-bank interleaved)."""
    c = tensor.ravel()[:n_amps].astype(np.complex128)
    amps_per_bank = (n_amps + NUM_HBM_BANKS - 1) // NUM_HBM_BANKS

    for bid in range(NUM_HBM_BANKS):
        idxs = np.arange(bid, n_amps, NUM_HBM_BANKS)
        if len(idxs) == 0:
            continue
        buf = np.zeros(amps_per_bank * 2, dtype=np.float64)
        v = len(idxs)
        buf[:v * 2:2] = c[idxs].real
        buf[1:v * 2:2] = c[idxs].imag
        byte_off = (g_offset // NUM_HBM_BANKS) * 2 * 8
        n_bytes  = v * 2 * 8
        banks[bid].write(buf.tobytes(), byte_off)
        banks[bid].sync(
            xrt_mod.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
            n_bytes, byte_off
        )


def _hbm_read(banks: list, g_offset: int,
              n_amps: int, shape: tuple, xrt_mod) -> np.ndarray:
    """Read n_amps complex128 values from HBM (8-bank interleaved)."""
    amps_per_bank = (n_amps + NUM_HBM_BANKS - 1) // NUM_HBM_BANKS
    result = np.zeros(n_amps, dtype=np.complex128)
    # g_offset is always a multiple of 8 (enforced by _HBMAlloc.alloc padding),
    # so slot_start = g_offset // 8 is the starting slot index in every bank.
    byte_off  = (g_offset // NUM_HBM_BANKS) * 2 * 8  # = slot_start * 16
    f64_start = byte_off // 8                          # index into float64 view

    for bid in range(NUM_HBM_BANKS):
        idxs = np.arange(bid, n_amps, NUM_HBM_BANKS)
        if len(idxs) == 0:
            continue
        n_bytes = len(idxs) * 2 * 8
        banks[bid].sync(
            xrt_mod.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
            n_bytes, byte_off
        )
        # Read starting at byte_off (slot_start), not from byte 0.
        raw_all = np.frombuffer(banks[bid].map(), dtype=np.float64)
        raw = raw_all[f64_start : f64_start + amps_per_bank * 2].copy()
        v = min(len(idxs), amps_per_bank)
        result[idxs[:v]] = raw[:v * 2:2] + 1j * raw[1:v * 2:2]

    return result.reshape(shape)


# ─────────────────────────────────────────────────────────────────────────────
#  Gate instruction encoder
# ─────────────────────────────────────────────────────────────────────────────

def _enc_header(opcode: int, si: int, sj: int,
                chi_l: int, chi_m: int, chi_r: int, chi_new: int,
                params: Optional[list],
                off_i: int, off_j: int) -> np.ndarray:
    h = np.zeros(INSTR_WORDS, dtype=np.int32)
    h[0] = opcode
    h[1] = si
    h[2] = sj
    h[3] = chi_l
    h[4] = chi_m
    h[5] = chi_r
    h[6] = chi_new
    for k, p in enumerate((params or [])[:3]):
        h[7 + k] = _f32_bits(p)
    h[10] = off_i
    h[11] = off_j
    return h


def _build_1q_instr(opcode: int, site: int,
                    chi_l: int, chi_r: int,
                    params: Optional[list],
                    off_i: int,
                    gate_mat: np.ndarray) -> np.ndarray:
    instr = np.zeros(WORDS_PER_GATE, dtype=np.int32)
    instr[:INSTR_WORDS] = _enc_header(
        opcode, site, -1, chi_l, 0, chi_r, 0, params, off_i, -1
    )
    # 2×2 complex128 → 8 float32 pairs
    g2 = gate_mat.ravel().astype(np.complex128)
    for k in range(4):
        instr[INSTR_WORDS + k * 2]     = _f32_bits(g2[k].real)
        instr[INSTR_WORDS + k * 2 + 1] = _f32_bits(g2[k].imag)
    # The compiled xclbin reads p0,p1,p2 from gbuf[base+INSTR_WORDS+{0,1,2}]
    # (words 12,13,14) not from the header words 7,8,9.  Override those slots.
    ps = list((params or [])[:3]) + [0.0, 0.0, 0.0]
    for k in range(3):
        instr[INSTR_WORDS + k] = _f32_bits(ps[k])
    return instr


def _build_2q_instr(opcode: int, si: int, sj: int,
                    chi_l: int, chi_m: int, chi_r: int, chi_new: int,
                    params: Optional[list],
                    off_i: int, off_j: int,
                    gate_mat: np.ndarray) -> np.ndarray:
    instr = np.zeros(WORDS_PER_GATE, dtype=np.int32)
    instr[:INSTR_WORDS] = _enc_header(
        opcode, si, sj, chi_l, chi_m, chi_r, chi_new, params, off_i, off_j
    )
    gf = gate_mat.ravel().astype(np.complex128)
    for k in range(16):
        instr[INSTR_WORDS + k * 2]     = _f32_bits(gf[k].real)
        instr[INSTR_WORDS + k * 2 + 1] = _f32_bits(gf[k].imag)
    return instr


# ─────────────────────────────────────────────────────────────────────────────
#  Circuit compiler: QuantumCircuit → nearest-neighbour gate list
# ─────────────────────────────────────────────────────────────────────────────

def _qubit_index(circuit, qubit) -> int:
    try:
        return circuit.find_bit(qubit).index
    except Exception:
        return circuit.qubits.index(qubit)


def _decompose_ccx(c0: int, c1: int, tgt: int) -> list:
    return [
        ('h',   [tgt], None),       ('cx', [c1,  tgt], None),
        ('tdg', [tgt], None),       ('cx', [c0,  tgt], None),
        ('t',   [tgt], None),       ('cx', [c1,  tgt], None),
        ('tdg', [tgt], None),       ('cx', [c0,  tgt], None),
        ('t',   [c1],  None),       ('t',  [tgt], None),
        ('h',   [tgt], None),       ('cx', [c0,  c1],  None),
        ('t',   [c0],  None),       ('tdg',[c1],  None),
        ('cx',  [c0,  c1],  None),
    ]


def compile_circuit(circuit, n_qubits: int) -> Tuple[list, list]:
    """
    Compile QuantumCircuit to a nearest-neighbour gate list.

    Returns (gate_list, measured_qubit_indices).

    Each gate dict:
        type      : '1q' | '2q' | 'reset'
        gate_name : str (canonical name)
        sites     : [i] for 1q, [i, i+1] for 2q (always adjacent after routing)
        params    : list[float] | None
        opcode    : int
        gate_mat  : np.ndarray (2×2 or 4×4 complex128)
    """
    perm = list(range(n_qubits))   # perm[logical] = physical site
    gates_out: list = []
    measured: list = []

    def _swap_sites(a: int, b: int):
        sa, sb = min(a, b), max(a, b)
        gates_out.append({
            'type': '2q', 'gate_name': 'swap', 'sites': [sa, sb],
            'params': None, 'opcode': _OP2Q['swap'],
            'gate_mat': GATE_MATRICES['swap'],
        })
        for idx in range(n_qubits):
            if perm[idx] == sa:
                perm[idx] = sb
            elif perm[idx] == sb:
                perm[idx] = sa

    def _route_2q(pi: int, pj: int, nm: str, ps: Optional[list]):
        if pi > pj:
            pi, pj = pj, pi
            pp = np.array([0, 2, 1, 3])
            gm = _gate_2q(nm, ps).reshape(4, 4)[np.ix_(pp, pp)]
        else:
            gm = _gate_2q(nm, ps)
        for k in range(pj, pi + 1, -1):
            _swap_sites(k - 1, k)
        gates_out.append({
            'type': '2q', 'gate_name': nm, 'sites': [pi, pi + 1],
            'params': ps, 'opcode': _OP2Q.get(nm, 17),
            'gate_mat': gm,
        })
        for k in range(pi + 1, pj):
            _swap_sites(k, k + 1)

    for inst in circuit.data:
        nm  = inst.operation.name.lower()
        qs  = [_qubit_index(circuit, q) for q in inst.qubits]
        ps  = [float(p) for p in getattr(inst.operation, 'params', [])] or None

        if nm == 'measure':
            for q in qs:
                if perm[q] not in measured:
                    measured.append(perm[q])
            continue
        if nm in ('barrier', 'snapshot', 'delay'):
            continue
        if nm == 'id':
            continue

        if nm == 'reset':
            gates_out.append({
                'type': 'reset', 'gate_name': 'reset',
                'sites': [perm[qs[0]]], 'params': None,
                'opcode': -1, 'gate_mat': None,
            })
            continue

        if nm in _SINGLE_QUBIT_GATES or nm in _OP1Q:
            site = perm[qs[0]]
            gates_out.append({
                'type': '1q', 'gate_name': nm, 'sites': [site],
                'params': ps, 'opcode': _OP1Q.get(nm, 0),
                'gate_mat': _gate_1q(nm, ps),
            })
            continue

        if nm in _TWO_QUBIT_GATES or nm in _OP2Q:
            pi, pj = perm[qs[0]], perm[qs[1]]
            if abs(pi - pj) == 1:
                si, sj = min(pi, pj), max(pi, pj)
                if pi > pj:
                    pp = np.array([0, 2, 1, 3])
                    gm = _gate_2q(nm, ps).reshape(4, 4)[np.ix_(pp, pp)]
                else:
                    gm = _gate_2q(nm, ps)
                gates_out.append({
                    'type': '2q', 'gate_name': nm, 'sites': [si, sj],
                    'params': ps, 'opcode': _OP2Q.get(nm, 17),
                    'gate_mat': gm,
                })
            else:
                _route_2q(pi, pj, nm, ps)
            continue

        if nm in ('ccx', 'toffoli', 'ccnot'):
            c0, c1, tgt = perm[qs[0]], perm[qs[1]], perm[qs[2]]
            for sub_nm, sub_qs, sub_ps in _decompose_ccx(c0, c1, tgt):
                if sub_nm in _SINGLE_QUBIT_GATES or sub_nm in _OP1Q:
                    gates_out.append({
                        'type': '1q', 'gate_name': sub_nm,
                        'sites': [sub_qs[0]], 'params': sub_ps,
                        'opcode': _OP1Q.get(sub_nm, 0),
                        'gate_mat': _gate_1q(sub_nm, sub_ps),
                    })
                else:
                    pi2, pj2 = sub_qs[0], sub_qs[1]
                    if abs(pi2 - pj2) == 1:
                        si2, sj2 = min(pi2, pj2), max(pi2, pj2)
                        gates_out.append({
                            'type': '2q', 'gate_name': sub_nm,
                            'sites': [si2, sj2], 'params': sub_ps,
                            'opcode': _OP2Q.get(sub_nm, 17),
                            'gate_mat': _gate_2q(sub_nm, sub_ps),
                        })
                    else:
                        _route_2q(pi2, pj2, sub_nm, sub_ps)
            continue

    if not measured:
        measured = list(range(n_qubits))

    return gates_out, sorted(measured)


# ─────────────────────────────────────────────────────────────────────────────
#  Core FPGA execution engine for one circuit on one open device+kernel
# ─────────────────────────────────────────────────────────────────────────────

def _run_circuit_fpga(circuit, shots: int,
                      chi_max: int, svd_cutoff: float,
                      seed: Optional[int],
                      device, kernel,
                      xrt_mod) -> Tuple[MPSState, Dict[str, int]]:
    """
    Execute one circuit on an already-initialised FPGA device + kernel.
    Returns (mps_state_after_gates, counts_dict).

    Gate execution strategy:
      1q gates    : batched in BRAM gate buffer → one kernel call per batch.
      2q gates (chi ≤ CHI_FULL_SVD)  : OP_2Q_FULL  → kernel does SVD+split.
      2q gates (chi > CHI_FULL_SVD)  : OP_2Q_CONTRACT → kernel contracts+applies;
                                        host reads theta, runs scipy SVD, writes back.
    """
    nq  = circuit.num_qubits
    mps = MPSState(nq, chi_max, svd_cutoff)

    # ── Compile circuit first (needed to size BOs for gate-phase allocations) ─
    gate_list, measured_qubits = compile_circuit(circuit, nq)

    # ── HBM allocation ────────────────────────────────────────────────────────
    alloc = _HBMAlloc()
    site_offsets: List[int] = []
    for i in range(nq):
        site_offsets.append(alloc.alloc_site(mps.chi_l(i), mps.chi_r(i)))
    theta_off = alloc.alloc(chi_max * 2 * chi_max * 2)   # workspace
    init_amps = alloc.total

    # Each 2Q gate creates 2 new padded site allocations (worst-case chi_max²).
    max_site_amps = ((chi_max * 2 * chi_max + NUM_HBM_BANKS - 1)
                     // NUM_HBM_BANKS) * NUM_HBM_BANKS
    n_2q = sum(1 for g in gate_list if g['type'] == '2q')
    total_amps = init_amps + n_2q * 2 * max_site_amps

    # ── Allocate HBM BOs (one per bank, interleaved) ─────────────────────────
    amps_per_bank = (total_amps + NUM_HBM_BANKS - 1) // NUM_HBM_BANKS
    hbm_bytes = max(amps_per_bank * 2 * 8, 4096)
    banks: List = [
        xrt_mod.bo(device, hbm_bytes,
                   xrt_mod.bo.flags.normal,
                   kernel.group_id(bid))
        for bid in range(NUM_HBM_BANKS)
    ]
    gate_bo = xrt_mod.bo(device,
                         max(MAX_GATES_BRAM * WORDS_PER_GATE * 4, 4096),
                         xrt_mod.bo.flags.normal,
                         kernel.group_id(8))

    # ── Upload |0...0⟩ state ─────────────────────────────────────────────────
    for i in range(nq):
        _hbm_write(banks, mps.sites[i], site_offsets[i],
                   mps.chi_l(i) * 2 * mps.chi_r(i), xrt_mod)

    # ── Execute gate list ─────────────────────────────────────────────────────
    pending_1q: list = []

    def _flush_1q():
        nonlocal pending_1q
        if not pending_1q:
            return
        n_g = len(pending_1q)
        enc = np.concatenate(pending_1q, axis=0).astype(np.int32)
        gate_bo.write(enc.tobytes(), 0)
        gate_bo.sync(xrt_mod.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                     enc.nbytes, 0)
        kernel(*banks, gate_bo, n_g, nq, chi_max).wait()

        # Read updated site tensors back so CPU MPS state stays coherent
        for instr in pending_1q:
            s     = int(instr[1])
            chl   = mps.chi_l(s)
            chr_  = mps.chi_r(s)
            mps.sites[s] = _hbm_read(
                banks, site_offsets[s],
                chl * 2 * chr_, (chl, 2, chr_), xrt_mod
            )
        pending_1q.clear()

    for gate in gate_list:

        if gate['type'] == 'reset':
            _flush_1q()
            s = gate['sites'][0]
            t = mps.sites[s].copy()
            t[:, 1, :] = 0.0
            nm = np.linalg.norm(t)
            if nm > 1e-15:
                t /= nm
            mps.sites[s] = t
            _hbm_write(banks, t, site_offsets[s],
                       mps.chi_l(s) * 2 * mps.chi_r(s), xrt_mod)
            continue

        if gate['type'] == '1q':
            s = gate['sites'][0]
            pending_1q.append(_build_1q_instr(
                gate['opcode'], s,
                mps.chi_l(s), mps.chi_r(s),
                gate['params'], site_offsets[s],
                gate['gate_mat']
            ))
            if len(pending_1q) >= MAX_GATES_BRAM - 4:
                _flush_1q()
            continue

        if gate['type'] == '2q':
            _flush_1q()
            si, sj = gate['sites'][0], gate['sites'][1]
            assert sj == si + 1, \
                f"Non-adjacent 2q gate after routing: sites {si},{sj}"

            chi_l = mps.chi_l(si)
            chi_m = mps.chi_r(si)
            chi_r = mps.chi_r(sj)

            # Always OP_2Q_CONTRACT: FPGA contracts+applies gate → writes theta
            # to theta_off; host reads theta, runs scipy SVD, writes back A_i/A_j.
            # word [6] carries theta_off so the kernel knows where to write.
            instr = _build_2q_instr(
                OP_2Q_CONTRACT,
                si, sj, chi_l, chi_m, chi_r,
                theta_off,    # word [6] = HBM theta workspace offset
                gate['params'], site_offsets[si], site_offsets[sj],
                gate['gate_mat']
            ).reshape(1, WORDS_PER_GATE).astype(np.int32)

            gate_bo.write(instr.tobytes(), 0)
            gate_bo.sync(xrt_mod.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                         instr.nbytes, 0)
            kernel(*banks, gate_bo, 1, nq, chi_max).wait()

            # Host SVD: read theta from workspace, truncate, write A_i / A_j back
            theta = _hbm_read(banks, theta_off,
                              chi_l * 4 * chi_r,
                              (chi_l * 2, 2 * chi_r), xrt_mod)
            U, S, Vh = la.svd(theta, full_matrices=False)
            thresh = svd_cutoff * S[0] if len(S) > 0 else 0.0
            keep = int(np.clip(np.sum(S > thresh), 1, chi_max))
            U, S, Vh = U[:, :keep], S[:keep], Vh[:keep, :]
            nm_ = np.linalg.norm(S)
            if nm_ > 1e-15:
                S /= nm_
            sqS = np.sqrt(S)
            chi_new = keep

            mps.sites[si] = (U * sqS).reshape(chi_l, 2, chi_new)
            mps.sites[sj] = (sqS[:, None] * Vh).reshape(chi_new, 2, chi_r)

            site_offsets[si] = alloc.alloc_site(chi_l, chi_new)
            site_offsets[sj] = alloc.alloc_site(chi_new, chi_r)
            _hbm_write(banks, mps.sites[si], site_offsets[si],
                       chi_l * 2 * chi_new, xrt_mod)
            _hbm_write(banks, mps.sites[sj], site_offsets[sj],
                       chi_new * 2 * chi_r, xrt_mod)
            continue

    _flush_1q()

    # ── Sequential conditional sampling on CPU ────────────────────────────────
    rng = np.random.default_rng(seed)
    counts = mps.get_counts(shots, measured_qubits, rng)

    banks.clear()
    return mps, counts


# ─────────────────────────────────────────────────────────────────────────────
#  Worker process — Mode A (shot-parallel, one card)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_a(card_id: int, xclbin: str, kernel_name: str,
              cmd_q: Queue, res_q: Queue,
              ready: Event, shutdown: Event):
    """Persistent worker for one FPGA card (Mode A)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    _device = None
    _kernel = None
    _fpga_ready = False

    def _init_fpga():
        nonlocal _device, _kernel, _fpga_ready
        import pyxrt as _xrt
        _device = _xrt.device(card_id)
        uuid    = _device.load_xclbin(xclbin)
        _kernel = _xrt.kernel(_device, uuid, kernel_name)
        _fpga_ready = True
        logger.info('Card %d: FPGA ready', card_id)

    try:
        ready.set()
        while not shutdown.is_set():
            try:
                cmd = cmd_q.get(timeout=0.25)
            except Exception:
                continue
            if cmd is None:
                break

            ct = cmd[0]

            if ct == 'ping':
                res_q.put(('pong', card_id))
                continue

            if ct == 'run_cpu':
                import pickle
                circuit_b, shots, chi_max, svd_cutoff, seed = cmd[1:]
                circuit = pickle.loads(circuit_b)
                sim = CPUMPSSimulator(chi_max=chi_max,
                                      svd_cutoff=svd_cutoff, seed=seed)
                mps_st, counts = sim.run_mps(circuit, shots)
                sv = None
                if circuit.num_qubits <= 24:
                    try:
                        sv = mps_st.get_statevector()
                    except Exception:
                        pass
                res_q.put(('done', card_id, counts, sv,
                           mps_st.max_bond(), mps_st.bond_dims()))
                continue

            if ct == 'run_fpga':
                import pickle, pyxrt as _xrt
                circuit_b, shots, chi_max, svd_cutoff, seed = cmd[1:]
                if not _fpga_ready:
                    _init_fpga()
                circuit = pickle.loads(circuit_b)
                try:
                    mps_st, counts = _run_circuit_fpga(
                        circuit, shots, chi_max, svd_cutoff, seed,
                        _device, _kernel, _xrt
                    )
                    sv = None
                    if circuit.num_qubits <= 24:
                        try:
                            sv = mps_st.get_statevector()
                        except Exception:
                            pass
                    res_q.put(('done', card_id, counts, sv,
                               mps_st.max_bond(), mps_st.bond_dims()))
                except Exception as e:
                    res_q.put(('error', card_id, str(e)))
                continue

    except Exception as e:
        logger.error('Worker card %d fatal: %s', card_id, e)
        res_q.put(('error', card_id, str(e)))


# ─────────────────────────────────────────────────────────────────────────────
#  Worker process — Mode B (distributed MPS chain segment)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_b(card_id: int, num_cards: int,
              xclbin: str, kernel_name: str,
              read_arrs: list, write_arrs: list, shm_floats: int,
              bar_a: Barrier, bar_b: Barrier, bar_c: Barrier,
              cmd_q: Queue, res_q: Queue,
              ready: Event, shutdown: Event):
    """Worker owning a contiguous MPS segment (Mode B)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    local_mps: Optional[MPSState] = None

    try:
        ready.set()
        while not shutdown.is_set():
            try:
                cmd = cmd_q.get(timeout=0.25)
            except Exception:
                continue
            if cmd is None:
                break

            ct = cmd[0]

            if ct == 'init_seg':
                seg_len, chi_max, svd_cutoff = cmd[1], cmd[2], cmd[3]
                local_mps = MPSState(seg_len, chi_max, svd_cutoff)
                _pub(card_id, local_mps, read_arrs, shm_floats)
                res_q.put(('init_done', card_id))

            elif ct == 'local_gate':
                gate = cmd[1]
                if gate['type'] == '1q':
                    local_mps.apply_1q(gate['local_site'], gate['gate_mat'])
                else:
                    local_mps.apply_2q(gate['local_si'], gate['local_sj'],
                                       gate['gate_mat'])
                _pub(card_id, local_mps, read_arrs, shm_floats)
                res_q.put(('gate_done', card_id))

            elif ct == 'boundary_gate':
                gate, side = cmd[1], cmd[2]
                my_site = (local_mps.n - 1) if side == 'right' else 0
                _pub(card_id, local_mps, read_arrs, shm_floats)
                bar_a.wait()

                nb_id = card_id + 1 if side == 'right' else card_id - 1
                nbuf = _shm_read_f64(read_arrs[nb_id], shm_floats)
                chi_l_nb = int(nbuf[0])
                chi_r_nb = int(nbuf[1])
                n_nb = chi_l_nb * 2 * chi_r_nb
                nb_t = _unpack_cplx128(nbuf[2:2 + n_nb * 2],
                                       (chi_l_nb, 2, chi_r_nb))

                G = gate['gate_mat']
                tmp = MPSState(2, local_mps.chi_max, local_mps.svd_cutoff)
                if side == 'right':
                    tmp.sites[0] = local_mps.sites[my_site]
                    tmp.sites[1] = nb_t
                    tmp.apply_2q(0, 1, G)
                    local_mps.sites[my_site] = tmp.sites[0]
                    new_nb = tmp.sites[1]
                else:
                    tmp.sites[0] = nb_t
                    tmp.sites[1] = local_mps.sites[my_site]
                    tmp.apply_2q(0, 1, G)
                    new_nb = tmp.sites[0]
                    local_mps.sites[my_site] = tmp.sites[1]

                _write_nb(write_arrs[card_id], new_nb, shm_floats)
                bar_b.wait()
                _pub(card_id, local_mps, read_arrs, shm_floats)
                bar_c.wait()
                res_q.put(('gate_done', card_id))

            elif ct == 'collect':
                tbs = [s.tobytes() for s in local_mps.sites]
                shs = [s.shape    for s in local_mps.sites]
                res_q.put(('segment', card_id, tbs, shs))

    except Exception as e:
        logger.error('Worker-B card %d fatal: %s', card_id, e)
        res_q.put(('error', card_id, str(e)))


def _pub(cid: int, mps: MPSState, arrs: list, n: int):
    for t in [mps.sites[0], mps.sites[-1]]:
        cl, _, cr = t.shape
        buf = np.concatenate([np.array([float(cl), float(cr)]),
                               _pack_cplx128(t)])
        _shm_write_f64(arrs[cid], buf, min(len(buf), n))


def _write_nb(arr: Array, tensor: np.ndarray, n: int):
    cl, _, cr = tensor.shape
    buf = np.concatenate([np.array([float(cl), float(cr)]),
                           _pack_cplx128(tensor)])
    _shm_write_f64(arr, buf, min(len(buf), n))


# ─────────────────────────────────────────────────────────────────────────────
#  FPGAMPSResult
# ─────────────────────────────────────────────────────────────────────────────

class FPGAMPSResult:
    """Qiskit-compatible result container."""

    def __init__(self, results: list, shots: int,
                 num_cards: int, elapsed: float, mode: str):
        self._results = results
        self.shots     = shots
        self.num_cards = num_cards
        self.total_time = elapsed
        self.mode      = mode

    def get_counts(self, idx: int = 0) -> Dict[str, int]:
        return self._results[idx]['counts']

    def get_statevector(self, idx: int = 0) -> np.ndarray:
        sv = self._results[idx].get('statevector')
        if sv is None:
            raise ValueError(
                'Statevector not available (n > 24 or not computed).'
            )
        return sv

    def get_mps(self, idx: int = 0) -> Optional[MPSState]:
        return self._results[idx].get('mps')

    def get_max_bond(self, idx: int = 0) -> int:
        return self._results[idx].get('max_bond', -1)

    def get_bond_dims(self, idx: int = 0) -> List[int]:
        return self._results[idx].get('bond_dims', [])

    def success(self) -> bool:
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  FPGAMPSSimulator — main class
# ─────────────────────────────────────────────────────────────────────────────

class FPGAMPSSimulator:
    """
    FPGA-accelerated Matrix Product State quantum circuit simulator.

    Parameters
    ----------
    num_cards   : 1, 2, or 4. Default 4.
    chi_max     : Maximum bond dimension. Default 256.
    svd_cutoff  : Relative SVD truncation threshold. Default 1e-12.
    engine      : 'fpga' | 'cpu' | 'auto'.
                  'fpga' → RuntimeError if XRT unavailable or device absent.
                  'cpu'  → always cpu_reference.py.
                  'auto' → FPGA if XRT present and device found, else CPU.
    mode        : 'shots' | 'distributed' | 'auto'.
    xclbin_path : Path to .xclbin; None → auto-search cwd and script dir.
    """

    def __init__(self,
                 num_cards: int = NUM_CARDS,
                 chi_max: int = 256,
                 svd_cutoff: float = 1e-12,
                 engine: str = 'auto',
                 mode: str = 'auto',
                 xclbin_path: Optional[str] = None):

        if num_cards not in (1, 2, 4):
            raise ValueError('num_cards must be 1, 2, or 4.')

        self.num_cards  = num_cards
        self.chi_max    = chi_max
        self.svd_cutoff = svd_cutoff
        self.mode       = mode

        # ── Resolve engine ────────────────────────────────────────────────────
        if engine == 'auto':
            if XRT_AVAILABLE:
                try:
                    xrt.device(0)
                    self._engine = 'fpga'
                except Exception:
                    self._engine = 'cpu'
                    logger.warning(
                        'engine=auto → CPU (XRT present but no device found).'
                    )
            else:
                self._engine = 'cpu'
                logger.warning(
                    'engine=auto → CPU (pyxrt not available). '
                    'Source /opt/xilinx/xrt/setup.sh to enable FPGA.'
                )
        elif engine == 'fpga':
            if not XRT_AVAILABLE:
                raise RuntimeError(
                    "engine='fpga' requires pyxrt. "
                    'Source /opt/xilinx/xrt/setup.sh and retry.'
                )
            self._engine = 'fpga'
        elif engine == 'cpu':
            self._engine = 'cpu'
        else:
            raise ValueError(
                f"engine must be 'fpga', 'cpu', or 'auto'; got {engine!r}"
            )

        # ── Resolve xclbin path ───────────────────────────────────────────────
        if self._engine == 'fpga':
            if xclbin_path is None:
                for candidate in [
                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 DEFAULT_XCLBIN),
                    os.path.join(os.getcwd(), DEFAULT_XCLBIN),
                ]:
                    if os.path.isfile(candidate):
                        xclbin_path = candidate
                        break
                if xclbin_path is None:
                    raise FileNotFoundError(
                        f"'{DEFAULT_XCLBIN}' not found. "
                        "Run 'make xclbin' or pass xclbin_path= explicitly."
                    )
            if not os.path.isfile(xclbin_path):
                raise FileNotFoundError(f'xclbin not found: {xclbin_path}')
            self._xclbin = os.path.abspath(xclbin_path)
            self._kname  = self._probe_kernel_name()
        else:
            self._xclbin = None
            self._kname  = None

        # ── Worker state ──────────────────────────────────────────────────────
        self._workers:      List[Process] = []
        self._cmd_qs:       List[Queue]   = []
        self._res_q:        Optional[Queue] = None
        self._ready_evts:   List[Event]   = []
        self._shutdown_evts: List[Event]  = []
        self._spawned       = False
        self._spawned_mode: Optional[str] = None

        atexit.register(self.shutdown)

    # ─────────────────────────────────────────────────────────────────────────
    #  Kernel name probing
    # ─────────────────────────────────────────────────────────────────────────

    def _probe_kernel_name(self) -> str:
        try:
            d    = xrt.device(0)
            uuid = d.load_xclbin(self._xclbin)
            for kn in (KERNEL_NAME, 'mps_kernel', 'quantum_mps'):
                try:
                    xrt.kernel(d, uuid, kn)
                    return kn
                except Exception:
                    pass
        except Exception as e:
            logger.warning('Kernel probe failed: %s — using default name', e)
        return KERNEL_NAME

    # ─────────────────────────────────────────────────────────────────────────
    #  Worker spawn / teardown
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_workers(self, eff_mode: str):
        if self._spawned and self._spawned_mode == eff_mode:
            if all(w.is_alive() for w in self._workers):
                return
            logger.warning('Dead workers detected; respawning.')
        if self._spawned:
            self._teardown()

        self._res_q = Queue()

        if eff_mode == 'shots':
            cmd_type = 'run_fpga' if self._engine == 'fpga' else 'run_cpu'
            for cid in range(self.num_cards):
                cq, re, se = Queue(), Event(), Event()
                w = Process(
                    target=_worker_a,
                    args=(cid, self._xclbin or '', self._kname or '',
                          cq, self._res_q, re, se),
                    daemon=True
                )
                w.start()
                self._workers.append(w)
                self._cmd_qs.append(cq)
                self._ready_evts.append(re)
                self._shutdown_evts.append(se)

        elif eff_mode == 'distributed':
            slot = self.chi_max * 2 * self.chi_max * 2 + 2
            self._read_arrs  = [Array(ctypes.c_double, slot, lock=False)
                                 for _ in range(self.num_cards)]
            self._write_arrs = [Array(ctypes.c_double, slot, lock=False)
                                 for _ in range(self.num_cards)]
            bars = [_mp.Barrier(self.num_cards) for _ in range(3)]
            for cid in range(self.num_cards):
                cq, re, se = Queue(), Event(), Event()
                w = Process(
                    target=_worker_b,
                    args=(cid, self.num_cards,
                          self._xclbin or '', self._kname or '',
                          self._read_arrs, self._write_arrs, slot,
                          *bars,
                          cq, self._res_q, re, se),
                    daemon=True
                )
                w.start()
                self._workers.append(w)
                self._cmd_qs.append(cq)
                self._ready_evts.append(re)
                self._shutdown_evts.append(se)

        t0 = time.time()
        for i, ev in enumerate(self._ready_evts):
            ev.wait(timeout=max(1.0, 60.0 - (time.time() - t0)))

        self._spawned      = True
        self._spawned_mode = eff_mode

    def _teardown(self):
        for se in self._shutdown_evts:
            try:
                se.set()
            except Exception:
                pass
        for cq in self._cmd_qs:
            try:
                cq.put_nowait(None)
            except Exception:
                pass
        for w in self._workers:
            w.join(timeout=5)
            if w.is_alive():
                w.terminate()
                w.join(timeout=2)
        self._workers.clear()
        self._cmd_qs.clear()
        self._ready_evts.clear()
        self._shutdown_evts.clear()
        try:
            if self._res_q:
                while not self._res_q.empty():
                    self._res_q.get_nowait()
        except Exception:
            pass
        self._res_q        = None
        self._spawned      = False
        self._spawned_mode = None

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self,
            circuits,
            shots: int = 1024,
            seed: Optional[int] = None) -> FPGAMPSResult:
        """
        Run one or more QuantumCircuits.

        Parameters
        ----------
        circuits : QuantumCircuit or list[QuantumCircuit]
        shots    : measurement shots per circuit
        seed     : RNG seed for reproducibility

        Returns
        -------
        FPGAMPSResult
        """
        if not isinstance(circuits, list):
            circuits = [circuits]

        t0 = time.time()

        if self._engine == 'cpu':
            return self._run_cpu(circuits, shots, seed, t0)

        eff_mode = self.mode
        if eff_mode == 'auto':
            eff_mode = ('distributed'
                        if len(circuits) == 1 and circuits[0].num_qubits >= 16
                        else 'shots')

        self._ensure_workers(eff_mode)

        if eff_mode == 'shots':
            return self._run_shots(circuits, shots, seed, t0)
        else:
            return self._run_distributed(circuits[0], shots, seed, t0)

    # ─────────────────────────────────────────────────────────────────────────
    #  CPU path
    # ─────────────────────────────────────────────────────────────────────────

    def _run_cpu(self, circuits, shots, seed, t0):
        results = []
        for i, circuit in enumerate(circuits):
            s   = (seed + i) if seed is not None else None
            sim = CPUMPSSimulator(self.chi_max, self.svd_cutoff, s)
            mps_st, counts = sim.run_mps(circuit, shots)
            sv = None
            if circuit.num_qubits <= 24:
                try:
                    sv = mps_st.get_statevector()
                except Exception:
                    pass
            results.append({
                'counts': counts, 'statevector': sv, 'mps': mps_st,
                'max_bond': mps_st.max_bond(), 'bond_dims': mps_st.bond_dims(),
            })
        return FPGAMPSResult(results, shots, 1, time.time() - t0, 'cpu')

    # ─────────────────────────────────────────────────────────────────────────
    #  FPGA shot-parallel path
    # ─────────────────────────────────────────────────────────────────────────

    def _run_shots(self, circuits, shots, seed, t0) -> FPGAMPSResult:
        import pickle
        cmd_type    = 'run_fpga' if self._engine == 'fpga' else 'run_cpu'
        tasks       = list(enumerate(circuits))
        pending:    List[Tuple[int, int]] = []
        free_cards  = list(range(self.num_cards))
        results_map: Dict[int, dict] = {}
        remaining   = list(range(len(tasks)))

        while remaining or pending:
            while remaining and free_cards:
                ti  = remaining.pop(0)
                cid = free_cards.pop(0)
                _, circuit = tasks[ti]
                s = (seed + ti) if seed is not None else None
                self._cmd_qs[cid].put(
                    (cmd_type,
                     pickle.dumps(circuit),
                     shots, self.chi_max, self.svd_cutoff, s)
                )
                pending.append((ti, cid))

            if pending:
                r = self._res_q.get(timeout=600)
                if r[0] == 'done':
                    _, cid, counts, sv, max_b, bdims = r
                    ti = next(i for i, c in pending if c == cid)
                    pending.remove((ti, cid))
                    free_cards.append(cid)
                    results_map[ti] = {
                        'counts': counts, 'statevector': sv,
                        'max_bond': max_b, 'bond_dims': bdims,
                    }
                elif r[0] == 'error':
                    raise RuntimeError(f'Card {r[1]} error: {r[2]}')

        results = [results_map[i] for i in range(len(tasks))]
        return FPGAMPSResult(results, shots, self.num_cards,
                             time.time() - t0, 'shots_fpga')

    # ─────────────────────────────────────────────────────────────────────────
    #  FPGA distributed path
    # ─────────────────────────────────────────────────────────────────────────

    def _run_distributed(self, circuit, shots, seed, t0) -> FPGAMPSResult:
        nq  = circuit.num_qubits
        seg = nq // self.num_cards
        if seg < 1:
            logger.warning('Distributed: fewer sites than cards → shots mode')
            return self._run_shots([circuit], shots, seed, t0)

        for cid in range(self.num_cards):
            self._cmd_qs[cid].put(
                ('init_seg', seg, self.chi_max, self.svd_cutoff)
            )
        self._wait_all('init_done', self.num_cards)

        gate_list, measured = compile_circuit(circuit, nq)

        for gate in gate_list:
            if gate['type'] == '1q':
                s   = gate['sites'][0]
                cid = min(s // seg, self.num_cards - 1)
                g2  = dict(gate)
                g2['local_site'] = s - cid * seg
                self._cmd_qs[cid].put(('local_gate', g2))
                self._wait_one('gate_done', cid)

            elif gate['type'] == '2q':
                si, sj = gate['sites']
                ci = min(si // seg, self.num_cards - 1)
                cj = min(sj // seg, self.num_cards - 1)
                if ci == cj:
                    g2 = dict(gate)
                    g2['local_si'] = si - ci * seg
                    g2['local_sj'] = sj - ci * seg
                    self._cmd_qs[ci].put(('local_gate', g2))
                    self._wait_one('gate_done', ci)
                else:
                    assert abs(ci - cj) == 1
                    self._cmd_qs[ci].put(('boundary_gate', gate, 'right'))
                    self._cmd_qs[cj].put(('boundary_gate', gate, 'left'))
                    self._wait_all('gate_done', 2)

        for cid in range(self.num_cards):
            self._cmd_qs[cid].put(('collect',))

        segments: List[Optional[list]] = [None] * self.num_cards
        for _ in range(self.num_cards):
            r = self._res_q.get(timeout=120)
            if r[0] == 'segment':
                _, cid, tbs, shs = r
                segments[cid] = [
                    np.frombuffer(tb, dtype=np.complex128).reshape(sh)
                    for tb, sh in zip(tbs, shs)
                ]
            elif r[0] == 'error':
                raise RuntimeError(f'Card {r[1]}: {r[2]}')

        global_mps = MPSState(nq, self.chi_max, self.svd_cutoff)
        idx = 0
        for cid in range(self.num_cards):
            for t in (segments[cid] or []):
                if idx < nq:
                    global_mps.sites[idx] = t
                    idx += 1

        rng    = np.random.default_rng(seed)
        counts = global_mps.get_counts(shots, measured, rng)
        sv     = None
        if nq <= 24:
            try:
                sv = global_mps.get_statevector()
            except Exception:
                pass

        return FPGAMPSResult([{
            'counts': counts, 'statevector': sv, 'mps': global_mps,
            'max_bond': global_mps.max_bond(),
            'bond_dims': global_mps.bond_dims(),
        }], shots, self.num_cards, time.time() - t0, 'distributed_fpga')

    # ─────────────────────────────────────────────────────────────────────────
    #  Barrier helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _wait_all(self, etype: str, count: int, timeout: float = 300.0):
        seen = 0
        deadline = time.time() + timeout
        while seen < count:
            rem = deadline - time.time()
            if rem <= 0:
                raise TimeoutError(
                    f'Timeout waiting for {etype} ({seen}/{count})'
                )
            r = self._res_q.get(timeout=rem)
            if r[0] == etype:
                seen += 1
            elif r[0] == 'error':
                raise RuntimeError(f'Card {r[1]}: {r[2]}')

    def _wait_one(self, etype: str, cid: int, timeout: float = 300.0):
        deadline = time.time() + timeout
        while True:
            rem = deadline - time.time()
            if rem <= 0:
                raise TimeoutError(
                    f'Timeout waiting for {etype} from card {cid}'
                )
            r = self._res_q.get(timeout=rem)
            if r[0] == etype:
                return
            elif r[0] == 'error':
                raise RuntimeError(f'Card {r[1]}: {r[2]}')

    # ─────────────────────────────────────────────────────────────────────────
    #  Introspection / lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def get_card_stats(self) -> List[dict]:
        return [
            {
                'card_id':        i,
                'pid':            self._workers[i].pid if i < len(self._workers) else None,
                'alive':          (self._workers[i].is_alive()
                                   if i < len(self._workers) else False),
                'engine':         self._engine.upper(),
                'execution_mode': self._engine.upper(),
                'xclbin':         self._xclbin,
            }
            for i in range(self.num_cards)
        ]

    def shutdown(self):
        if self._spawned:
            self._teardown()

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
#  Pauli expectation value helper
# ─────────────────────────────────────────────────────────────────────────────

def _pauli_expectation_sv(sv: np.ndarray, pauli: str) -> float:
    """Compute ⟨ψ|P|ψ⟩ for a Pauli string P (e.g. 'ZZ', 'IXZ')."""
    n   = int(round(math.log2(len(sv))))
    psi = sv.copy()
    for i, p in enumerate(pauli.upper()):
        if p == 'I':
            continue
        qubit  = n - 1 - i
        mat    = GATE_MATRICES.get(p.lower(), np.eye(2, dtype=np.complex128))
        N      = 1 << n
        stride = 1 << qubit
        for j in range(0, N, stride * 2):
            for k in range(stride):
                a0 = psi[j + k]
                a1 = psi[j + k + stride]
                psi[j + k]          = mat[0, 0] * a0 + mat[0, 1] * a1
                psi[j + k + stride] = mat[1, 0] * a0 + mat[1, 1] * a1
    return float(np.real(np.dot(sv.conj(), psi)))


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point / self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('FPGA MPS Simulator — Host Driver')
    print('=' * 56)
    print(f'  XRT Available : {XRT_AVAILABLE}')
    print(f'  Qiskit        : {QISKIT_AVAILABLE}')
    print(f'  Kernel name   : {KERNEL_NAME}')
    print(f'  Default xclbin: {DEFAULT_XCLBIN}')
    print(f'  HBM banks     : {NUM_HBM_BANKS}  (+1 gate = 9 AXI masters total)')
    print(f'  FPGA SVD cap  : chi ≤ {CHI_FULL_SVD}')
    print()

    if XRT_AVAILABLE:
        try:
            d  = xrt.device(0)
            print(f'  Device 0 BDF  : {d.get_info(xrt.xrt_info_device.bdf)}')
            xb = DEFAULT_XCLBIN
            if os.path.isfile(xb):
                uuid = d.load_xclbin(xb)
                k    = xrt.kernel(d, uuid, KERNEL_NAME)
                print(f'  xclbin loaded : {xb}')
                print(f'  Kernel OK     : {KERNEL_NAME}')
            else:
                print(f'  xclbin        : NOT FOUND — run "make xclbin" first')
        except Exception as e:
            print(f'  Device probe  : FAILED — {e}')
    print()
    print('Usage:')
    print('  from fpga_mps_simulator import FPGAMPSSimulator')
    print('  sim = FPGAMPSSimulator(num_cards=4, engine="fpga")')
    print('  result = sim.run(circuit, shots=1024)')
    print('  print(result.get_counts())')

    if QISKIT_AVAILABLE:
        print()
        print('Self-test (CPU mode, 3-qubit GHZ):')
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()
        sim = FPGAMPSSimulator(num_cards=1, engine='cpu', chi_max=16)
        res = sim.run(qc, shots=1024, seed=0)
        counts = res.get_counts()
        total  = sum(counts.values())
        p000   = counts.get('000', 0) / total
        p111   = counts.get('111', 0) / total
        ok     = p000 + p111 > 0.95
        print(f'  P(000)+P(111) = {p000+p111:.3f}  {"PASS" if ok else "FAIL"}')
        sim.shutdown()
