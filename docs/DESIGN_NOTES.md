# Design Notes — FPGA MPS Simulator

**Author:** Nasir Ali  
**Organization:** C-DAC Noida  
**Date:** June 2026

---

## 1. What Was Reused from Each Reference File

### fpga_distributed_statevector12.py — Host Engineering Baseline

**Critical fix adopted verbatim:** `mp.set_start_method('spawn', force=False)` placed before any
other multiprocessing import. The fork()-based default inherits parent XRT file descriptors and
memory-mapped HBM regions into child processes, corrupting them silently. Spawn starts clean.

**Critical fix adopted verbatim:** `ctypes.memmove()` for all cross-process shared-memory reads
and writes. Using `np.frombuffer(mp.Array(...))` returns a cached view that does not cross the
process memory fence; reads silently return stale data. memmove reads from the raw mmap address.

**Architecture adopted:** Deferred worker spawn and dynamic per-circuit memory allocation.
Workers are created once and reused while the qubit count stays the same; if the qubit count
changes, workers are torn down and rebuilt at the new size. This avoids the "MAX_QUBITS upfront"
bug that triggered SIGBUS from a multi-TB mmap reservation.

**Architecture adopted:** 3-barrier double-buffered protocol for cross-card boundary tensor
exchange. Step 1: publish local data → read_buffer, barrier_a. Step 2: compute from
read_buffer → write_buffer, barrier_b. Step 3: copy write→read, barrier_c.

**Pattern adopted:** Explicit engine selection (engine="fpga"|"cpu"); RuntimeError on FPGA init
failure — never silent fallback.

**Pattern adopted:** Clean shutdown: atexit.register + signal.SIG_IGN in workers + drain queues
before join with timeout + terminate stragglers.

---

### quantum_fpga_kernel_complex128_v06.cpp — The Routing Lesson (Most Critical)

**Lesson applied, non-negotiable:** 16 independent AXI4 master ports on the Alveo U55C caused
"partially-conflicted nets in bank0_m_axi_U/bus_read/rreq_burst_conv" — Vivado failed after 4
global rip-up/reroute iterations. The fix is ≤9 AXI masters total: 8 amplitude/tensor bank ports
+ 1 shared bundle for gate_sequence and metadata.

**Adopted:** complex128 (double) datapath throughout. Single shared FP64 multiply-accumulate
path in `apply_single_gate`; all opcodes feed matrix coefficients into it rather than duplicating
the FP datapath per opcode. This dramatically reduces DSP and LUT usage.

**Adopted:** Union trick for encoding float32 parameters as int32 words inside the gate buffer:
`union { int i; float f; } p0; p0.i = gate_buf[base+7]; double dp0 = (double)p0.f;`

**Adopted:** BRAM gate buffer with `#pragma HLS BIND_STORAGE variable=gbuf type=ram_2p impl=bram`
and a separate LOAD loop before the EXEC loop. This decouples HBM gate reads from tensor reads.

**Adopted:** Realistic `#pragma HLS PIPELINE II=16` for HBM-bound loops and `II=8` for diagonal
(read-only) loops. `#pragma HLS LOOP_TRIPCOUNT` annotations.

**Adopted:** `extern "C"` top-level kernel to prevent C++ name mangling with XRT.

**Adopted:** `#pragma HLS INTERFACE s_axilite port=return bundle=control` pattern.

---

### fpga_distributed_statevector.py + fpga_distributed_statevector11.py — Architecture

**Adopted:** HBM buffer allocation per-circuit via `_ensure_workers(nq)`: avoids pre-allocating
at MAX_QUBITS which causes "failed to allocate userptr bo" errors.

**Adopted:** Worker process receives commands via mp.Queue, responds via shared result_queue.
Command protocol: ('init', nq), ('exec_gates', gate_list), ('collect',), ('shutdown',).

**Adopted:** Double-buffered shared memory interleaved as float32 [real0, imag0, real1, imag1, ...]
written via memmove and read via memmove.

**Key difference for MPS:** The inter-card communication is O(chi²) boundary tensor exchange
rather than O(2^n) full statevector exchange. This makes distributed MPS practical for large n.

---

### quantum_gates_kernel.cpp — Gate Coverage and HBM Layout

**Adopted:** Amplitude interleaving across HBM banks: global index g → bank = g % NUM_BANKS,
slot = g / NUM_BANKS, float offset = slot * 2. This gives equal load across banks.

**Adopted:** `read_amp()` / `write_amp()` inline helpers with switch-case over bank index.

**Adopted:** 16 banks in the reference were reduced to 8 in v06 — we follow v06's 8-bank layout.

**Gate list adopted:** Full coverage of H/X/Y/Z/S/T/Sdg/Tdg/SX/SXdg/RX/RY/RZ/P/U1/U2/U3,
CX/CY/CZ/CH/CP/CRX/CRY/CRZ/CSX/SWAP/iSWAP/ECR/RXX/RYY/RZZ/DCX/CCX (via decomposition on host).

---

### connectivity.cfg — HBM Mapping Strategy

**Lesson applied:** Grouping multiple sp= lines to the same physical HBM bank reduces the number
of distinct AXI interfaces Vitis generates. The 4-group pattern (banks 0-3 → HBM[0], 4-7 → HBM[4]
etc.) worked better than 16 individual assignments.

**New mapping for this kernel (8 tensor banks + 1 metadata bundle):**
- 8 tensor banks spread across 4 distinct HBM pseudo-channels (2 banks per channel)
- gate_sequence and site_metadata share a 9th bundle → HBM[16]
- Justified: HBM channels operate independently; spreading amplitude banks across channels
  maximizes concurrent read/write bandwidth for tensor contractions.

**Vivado strategies carried forward unchanged:** `Congestion_SpreadLogic_high`,
`PHYS_OPT_DESIGN enabled`, `-tns_cleanup`, `Flow_AreaOptimized_high` for synth.

---

### fpga_simulator_v06.py — Host Gate Decomposition

**Adopted:** `_lower_to_kernel_basis()` pattern — decompose all exotic gates to
{1q, CX, CZ} on the host before sending to FPGA. For MPS, we extend this to
{1q, nearest-neighbor 2q} since MPS can apply 2q gates natively.

**Adopted:** Lazy HBM allocation (`_ensure_hbm(nq)`), collect statevector helper,
distribute statevector helper.

**Adopted:** Qiskit Result wrapper class for get_counts() / get_statevector() compatibility.

---

### btest.py / test.py / final_test22.py — Test Patterns

**Adopted:** Progressive scale-up: test 5, 10, 16, 20, 24, 32, 50, 100 qubits.
**Adopted:** GHZ structure check: P(|00...0⟩) + P(|11...1⟩) > 0.98, exactly 2 nonzero amplitudes.
**Adopted:** Fidelity validation against Qiskit Statevector for n ≤ 20.
**Adopted:** Banner table format for results.
**Adopted:** Safety cap: skip if memory would exceed available RAM.

---

## 2. Novel Contributions vs References

| Feature | Source | Change in this project |
|---|---|---|
| MPS state representation | quantum_mps_kernel.cpp | Corrected einsum conventions; boundary conditions |
| Jacobi SVD | quantum_mps_kernel.cpp | Simplified on-chip; host LAPACK for large chi |
| Distributed chain partition | Novel (Mode B) | Boundary tensor exchange via shared memory |
| Shot/circuit parallelism | Inspired by distributed SV | Mode A; straightforward across 4 cards |
| chi-adaptive mode | Novel | Auto-selects FPGA SVD vs host LAPACK based on chi |
| canonical form | Novel | Left/right normalization tracked per site |

---

## 3. Memory Budget

| n qubits | chi_max | Tensor memory | HBM per card (4-card split) | Notes |
|---|---|---|---|---|
| 50 | 64 | 50 × 64 × 2 × 64 × 16 B ≈ 52 MB | 13 MB | Fits comfortably |
| 100 | 128 | 100 × 128 × 2 × 128 × 16 B ≈ 419 MB | 105 MB | Fine |
| 200 | 256 | 200 × 256 × 2 × 256 × 16 B ≈ 3.4 GB | 850 MB | Within 8 GB HBM |
| 200 | 512 | 200 × 512 × 2 × 512 × 16 B ≈ 13.4 GB | 3.3 GB | Fine per card |
| 500 | 512 | 500 × 512 × 2 × 512 × 16 B ≈ 33 GB | 8.3 GB | Borderline; use smaller chi |

Working tensors (theta for 2q gates): chi × 4 × chi × 16 B = 4 MB at chi=256, 16 MB at chi=512.
These fit in URAM (< 34 MB available on U55C).

---

## 4. Known Limitations and Future Work

- Jacobi SVD in kernel limited to chi ≤ CHI_MAX_SVD=64; larger chi uses host LAPACK round-trip.
- Mode B (distributed chain) shipped for nearest-neighbor circuits only; arbitrary non-adjacent
  2q gates are routed to nearest-neighbor via SWAP insertion on the host.
- Measurement back-action (post-measurement state update) implemented on host, not kernel.
- No noise model in MPS (decoherence channels) — future extension.
- Full complex128 kernel; complex64 mode is a TODO (would double qubit capacity per card).
