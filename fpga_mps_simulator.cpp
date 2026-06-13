/*
 * fpga_mps_simulator.cpp
 * ======================
 * HLS Kernel: FPGA-Based Matrix Product State (MPS) Quantum Circuit Simulator
 *
 * Top-level function: fpga_mps_simulator (extern "C", matches .xclbin name)
 *
 * AXI master count: EXACTLY 9
 *   - 8 HBM tensor banks:  m_axi_bank0 .. m_axi_bank7
 *   - 1 gate+metadata:     m_axi_gate  (gate_sequence + site_metadata)
 * Total = 9.  (17 masters caused routing failure on U55C in v06 — non-negotiable)
 *
 * Operations:
 *   opcode  0-16  : 1-qubit gates (opcode matches host _1Q_OPCODE encoding)
 *   opcode 102    : 2q full — contract + gate + Jacobi SVD (chi ≤ CHI_MAX_SVD=64)
 *   opcode 100    : 2q contract + gate only; host drives SVD
 *   opcode 110    : left-normalize site i
 *
 * Gate instruction format (12 × int32):
 *   [0]  gate_type
 *   [1]  site_i
 *   [2]  site_j      (-1 for 1q)
 *   [3]  chi_l
 *   [4]  chi_m       (bond between sites for 2q)
 *   [5]  chi_r
 *   [6]  chi_new     (truncated; 0 → runtime)
 *   [7-9] params as float32-in-int32
 *  [10]  hbm_offset_i  (in complex-double units)
 *  [11]  hbm_offset_j
 * For 2q ops, the kernel reads an extra 32 int32 words (gate matrix as float32 pairs)
 * immediately after the 12-word header.
 *
 * HBM layout:
 *   global index g  →  bank = g % 8,  slot = g / 8
 *   Each slot: float64[slot*2] = real,  float64[slot*2+1] = imag
 *   (float64 = double precision, complex128)
 *
 * Author: Nasir Ali
 * Organization: C-DAC Noida
 * Date: June 2026
 * Target: Xilinx Alveo U55C, Vitis 2023.2 HLS
 */

#include <stdint.h>
#include <math.h>
#include <ap_int.h>

// ─────────────────────────────────────────────────────────────────────────────
//  Compile-time limits
// ─────────────────────────────────────────────────────────────────────────────

#define MAX_QUBITS    500
#define CHI_MAX_SVD   64        // Jacobi SVD only for chi ≤ this
#define INSTR_WORDS   12
#define MAX_GATES_BRAM 512      // BRAM gate buffer capacity

// ─────────────────────────────────────────────────────────────────────────────
//  Types
// ─────────────────────────────────────────────────────────────────────────────

typedef double fp64;
typedef long long ll;

// Complex double pair stored as two fp64 in HBM
// Index arithmetic: amp at (bank, slot) is at fp64[slot*2], fp64[slot*2+1]

// ─────────────────────────────────────────────────────────────────────────────
//  Float32-as-int32 extraction (union trick — avoids UB in C++)
// ─────────────────────────────────────────────────────────────────────────────

static inline fp64 extract_f32(int packed) {
#pragma HLS INLINE
    union { int i; float f; } u;
    u.i = packed;
    return (fp64)u.f;
}

// ─────────────────────────────────────────────────────────────────────────────
//  HBM read/write helpers (8-bank interleaving)
// ─────────────────────────────────────────────────────────────────────────────

// Each bank pointer is passed as a separate AXI master argument.
// read_amp(bank0..7, g) reads complex double at global index g.
// write_amp(bank0..7, g, re, im) writes it.

#define READ_AMP_RE(banks, bid, slot) ((banks)[(bid)][(slot)*2])
#define READ_AMP_IM(banks, bid, slot) ((banks)[(bid)][(slot)*2+1])
#define WRITE_AMP_RE(banks, bid, slot, val) ((banks)[(bid)][(slot)*2]   = (val))
#define WRITE_AMP_IM(banks, bid, slot, val) ((banks)[(bid)][(slot)*2+1] = (val))

// Helper: read real part of amplitude at global offset g
static inline fp64 read_re(
        fp64 *b0, fp64 *b1, fp64 *b2, fp64 *b3,
        fp64 *b4, fp64 *b5, fp64 *b6, fp64 *b7,
        int global_g) {
#pragma HLS INLINE
    int bid  = global_g & 7;    // global_g % 8
    int slot = global_g >> 3;   // global_g / 8
    switch (bid) {
        case 0: return b0[slot*2];
        case 1: return b1[slot*2];
        case 2: return b2[slot*2];
        case 3: return b3[slot*2];
        case 4: return b4[slot*2];
        case 5: return b5[slot*2];
        case 6: return b6[slot*2];
        default: return b7[slot*2];
    }
}

static inline fp64 read_im(
        fp64 *b0, fp64 *b1, fp64 *b2, fp64 *b3,
        fp64 *b4, fp64 *b5, fp64 *b6, fp64 *b7,
        int global_g) {
#pragma HLS INLINE
    int bid  = global_g & 7;
    int slot = global_g >> 3;
    switch (bid) {
        case 0: return b0[slot*2+1];
        case 1: return b1[slot*2+1];
        case 2: return b2[slot*2+1];
        case 3: return b3[slot*2+1];
        case 4: return b4[slot*2+1];
        case 5: return b5[slot*2+1];
        case 6: return b6[slot*2+1];
        default: return b7[slot*2+1];
    }
}

static inline void write_amp(
        fp64 *b0, fp64 *b1, fp64 *b2, fp64 *b3,
        fp64 *b4, fp64 *b5, fp64 *b6, fp64 *b7,
        int global_g, fp64 re, fp64 im) {
#pragma HLS INLINE
    int bid  = global_g & 7;
    int slot = global_g >> 3;
    switch (bid) {
        case 0: b0[slot*2] = re; b0[slot*2+1] = im; break;
        case 1: b1[slot*2] = re; b1[slot*2+1] = im; break;
        case 2: b2[slot*2] = re; b2[slot*2+1] = im; break;
        case 3: b3[slot*2] = re; b3[slot*2+1] = im; break;
        case 4: b4[slot*2] = re; b4[slot*2+1] = im; break;
        case 5: b5[slot*2] = re; b5[slot*2+1] = im; break;
        case 6: b6[slot*2] = re; b6[slot*2+1] = im; break;
        default: b7[slot*2] = re; b7[slot*2+1] = im; break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  1-qubit gate matrix computation
//  Returns 2×2 complex matrix as [m00r, m00i, m01r, m01i, m10r, m10i, m11r, m11i]
// ─────────────────────────────────────────────────────────────────────────────

static void compute_1q_matrix(int opcode, fp64 p0, fp64 p1, fp64 p2,
                               fp64 mat[8]) {
#pragma HLS INLINE
    // Defaults: identity
    mat[0]=1; mat[1]=0; mat[2]=0; mat[3]=0;
    mat[4]=0; mat[5]=0; mat[6]=1; mat[7]=0;

    fp64 c, s;
    switch (opcode) {
        case 0: // id — already identity
            break;
        case 1: // h
            mat[0]=0.7071067811865476; mat[1]=0;
            mat[2]=0.7071067811865476; mat[3]=0;
            mat[4]=0.7071067811865476; mat[5]=0;
            mat[6]=-0.7071067811865476; mat[7]=0;
            break;
        case 2: // x
            mat[0]=0;mat[1]=0; mat[2]=1;mat[3]=0;
            mat[4]=1;mat[5]=0; mat[6]=0;mat[7]=0;
            break;
        case 3: // y
            mat[0]=0;mat[1]=0;  mat[2]=0;mat[3]=-1;
            mat[4]=0;mat[5]=1;  mat[6]=0;mat[7]=0;
            break;
        case 4: // z
            mat[0]=1;mat[1]=0;  mat[2]=0;mat[3]=0;
            mat[4]=0;mat[5]=0;  mat[6]=-1;mat[7]=0;
            break;
        case 5: // s
            mat[0]=1;mat[1]=0;  mat[2]=0;mat[3]=0;
            mat[4]=0;mat[5]=0;  mat[6]=0;mat[7]=1;
            break;
        case 6: // sdg
            mat[0]=1;mat[1]=0;  mat[2]=0;mat[3]=0;
            mat[4]=0;mat[5]=0;  mat[6]=0;mat[7]=-1;
            break;
        case 7: // t
            mat[0]=1;mat[1]=0;  mat[2]=0;mat[3]=0;
            mat[4]=0;mat[5]=0;
            mat[6]=0.7071067811865476; mat[7]=0.7071067811865476;
            break;
        case 8: // tdg
            mat[0]=1;mat[1]=0;  mat[2]=0;mat[3]=0;
            mat[4]=0;mat[5]=0;
            mat[6]=0.7071067811865476; mat[7]=-0.7071067811865476;
            break;
        case 9: // sx  (= (I + iX)/sqrt(2) * global phase)
            mat[0]=0.5;mat[1]=0.5;   mat[2]=0.5;mat[3]=-0.5;
            mat[4]=0.5;mat[5]=-0.5;  mat[6]=0.5;mat[7]=0.5;
            break;
        case 10: // sxdg
            mat[0]=0.5;mat[1]=-0.5;  mat[2]=0.5;mat[3]=0.5;
            mat[4]=0.5;mat[5]=0.5;   mat[6]=0.5;mat[7]=-0.5;
            break;
        case 11: // rx(p0)
            c=cos(p0*0.5); s=sin(p0*0.5);
            mat[0]=c;mat[1]=0;    mat[2]=0;mat[3]=-s;
            mat[4]=0;mat[5]=-s;   mat[6]=c;mat[7]=0;
            break;
        case 12: // ry(p0)
            c=cos(p0*0.5); s=sin(p0*0.5);
            mat[0]=c;mat[1]=0;    mat[2]=-s;mat[3]=0;
            mat[4]=s;mat[5]=0;    mat[6]=c;mat[7]=0;
            break;
        case 13: // rz(p0)
            c=cos(p0*0.5); s=sin(p0*0.5);
            mat[0]=c;mat[1]=-s;   mat[2]=0;mat[3]=0;
            mat[4]=0;mat[5]=0;    mat[6]=c;mat[7]=s;
            break;
        case 14: // p(p0) / u1 / phase
            mat[0]=1;mat[1]=0;    mat[2]=0;mat[3]=0;
            mat[4]=0;mat[5]=0;    mat[6]=cos(p0);mat[7]=sin(p0);
            break;
        case 15: // u2(p0,p1) = u3(pi/2, p0, p1)
            p2 = 1.5707963267948966; // pi/2
            // fall through to u3
            // u3(theta, phi, lam)
            c = cos(p2*0.5);
            s = sin(p2*0.5);
            mat[0] = c;  mat[1] = 0;
            mat[2] = -cos(p1)*s; mat[3] = -sin(p1)*s;
            mat[4] = cos(p0)*s;  mat[5] = sin(p0)*s;
            {
                fp64 cr = cos(p0+p1)*c;
                fp64 ci = sin(p0+p1)*c;
                mat[6] = cr; mat[7] = ci;
            }
            break;
        case 16: // u3(theta=p0, phi=p1, lam=p2)
            c = cos(p0*0.5);
            s = sin(p0*0.5);
            mat[0] = c;  mat[1] = 0;
            mat[2] = -cos(p2)*s; mat[3] = -sin(p2)*s;
            mat[4] = cos(p1)*s;  mat[5] = sin(p1)*s;
            {
                fp64 cr = cos(p1+p2)*c;
                fp64 ci = sin(p1+p2)*c;
                mat[6] = cr; mat[7] = ci;
            }
            break;
        default:
            // identity (already set)
            break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Apply 1-qubit gate to site tensor in HBM
//  Site tensor A[chi_l, 2, chi_r]:
//    A'[a, sigma', b] = sum_sigma M[sigma', sigma] * A[a, sigma, b]
//  HBM layout for this site: complex double at (a*2*chi_r + sigma*chi_r + b + offset_i)
// ─────────────────────────────────────────────────────────────────────────────

static void apply_1q_gate(
        fp64 *b0, fp64 *b1, fp64 *b2, fp64 *b3,
        fp64 *b4, fp64 *b5, fp64 *b6, fp64 *b7,
        int chi_l, int chi_r, int hbm_offset,
        fp64 mat[8]) {
#pragma HLS INLINE off

    fp64 a0r, a0i, a1r, a1i;
    fp64 r0r, r0i, r1r, r1i;

    // mat = [m00r,m00i, m01r,m01i, m10r,m10i, m11r,m11i]
    fp64 m00r = mat[0], m00i = mat[1];
    fp64 m01r = mat[2], m01i = mat[3];
    fp64 m10r = mat[4], m10i = mat[5];
    fp64 m11r = mat[6], m11i = mat[7];

LOOP_CHI_L:
    for (int a = 0; a < chi_l; a++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512 avg=64
LOOP_CHI_R:
        for (int b = 0; b < chi_r; b++) {
#pragma HLS PIPELINE II=16
#pragma HLS LOOP_TRIPCOUNT min=1 max=512 avg=64
            // A[a, 0, b] at global offset (a*2*chi_r + 0*chi_r + b) + hbm_offset
            int g0 = hbm_offset + a * 2 * chi_r + b;
            int g1 = hbm_offset + a * 2 * chi_r + chi_r + b;

            a0r = read_re(b0,b1,b2,b3,b4,b5,b6,b7, g0);
            a0i = read_im(b0,b1,b2,b3,b4,b5,b6,b7, g0);
            a1r = read_re(b0,b1,b2,b3,b4,b5,b6,b7, g1);
            a1i = read_im(b0,b1,b2,b3,b4,b5,b6,b7, g1);

            // A'[a, 0, b] = M[0,0]*A[a,0,b] + M[0,1]*A[a,1,b]
            r0r = m00r*a0r - m00i*a0i + m01r*a1r - m01i*a1i;
            r0i = m00r*a0i + m00i*a0r + m01r*a1i + m01i*a1r;

            // A'[a, 1, b] = M[1,0]*A[a,0,b] + M[1,1]*A[a,1,b]
            r1r = m10r*a0r - m10i*a0i + m11r*a1r - m11i*a1i;
            r1i = m10r*a0i + m10i*a0r + m11r*a1i + m11i*a1r;

            write_amp(b0,b1,b2,b3,b4,b5,b6,b7, g0, r0r, r0i);
            write_amp(b0,b1,b2,b3,b4,b5,b6,b7, g1, r1r, r1i);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Contract two adjacent MPS sites + apply 2q gate → theta in HBM
//
//  A_i[chi_l, 2, chi_m] (offset_i)   A_j[chi_m, 2, chi_r] (offset_j)
//  Theta[chi_l, 2, 2, chi_r]  (written to offset_j after gate application)
//  theta[a, s0, s1, d] = sum_{b,p,q} G[s0,s1,p,q] * A_i[a,p,b] * A_j[b,q,d]
//  Stored as theta[a, s, d] where s = s0*2+s1; layout: theta_offset
// ─────────────────────────────────────────────────────────────────────────────

static void contract_and_apply_2q(
        fp64 *b0, fp64 *b1, fp64 *b2, fp64 *b3,
        fp64 *b4, fp64 *b5, fp64 *b6, fp64 *b7,
        int chi_l, int chi_m, int chi_r,
        int offset_i, int offset_j,
        int theta_off,    // dedicated HBM workspace — avoids overflowing offset_j
        fp64 gmat[32]) {  // 4x4 complex matrix as [re00,im00, re01,...] row-major
#pragma HLS INLINE off

    // Workspace in on-chip URAM (up to chi_max * 4 * chi_max complex doubles)
    // Limit to CHI_MAX_SVD * 4 * CHI_MAX_SVD = 64*4*64 = 16384 complex doubles
    static fp64 theta_re[CHI_MAX_SVD][4][CHI_MAX_SVD];
    static fp64 theta_im[CHI_MAX_SVD][4][CHI_MAX_SVD];
#pragma HLS BIND_STORAGE variable=theta_re type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=theta_im type=ram_2p impl=uram

LOOP_A_CONTRACT:
    for (int a = 0; a < chi_l && a < CHI_MAX_SVD; a++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64 avg=32
LOOP_S_CONTRACT:
        for (int s = 0; s < 4; s++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
LOOP_D_CONTRACT:
            for (int d = 0; d < chi_r && d < CHI_MAX_SVD; d++) {
#pragma HLS PIPELINE II=16
#pragma HLS LOOP_TRIPCOUNT min=1 max=64 avg=32
                int s0 = s >> 1;  // upper qubit
                int s1 = s &  1;  // lower qubit

                fp64 sum_re = 0.0, sum_im = 0.0;

LOOP_B_INNER:
                for (int b = 0; b < chi_m && b < CHI_MAX_SVD; b++) {
#pragma HLS UNROLL factor=4
#pragma HLS LOOP_TRIPCOUNT min=1 max=64 avg=32
LOOP_P_INNER:
                    for (int p = 0; p < 2; p++) {
LOOP_Q_INNER:
                        for (int q = 0; q < 2; q++) {
                            // Gate matrix entry G[s0,s1,p,q] at index (s0*2+s1)*4+(p*2+q)
                            int gidx = (s0*2+s1)*8 + (p*2+q)*2;  // *2 for re/im
                            fp64 gr = gmat[gidx];
                            fp64 gi = gmat[gidx+1];

                            // A_i[a, p, b]
                            int gi_amp = offset_i + a*2*chi_m + p*chi_m + b;
                            fp64 air = read_re(b0,b1,b2,b3,b4,b5,b6,b7, gi_amp);
                            fp64 aii = read_im(b0,b1,b2,b3,b4,b5,b6,b7, gi_amp);

                            // A_j[b, q, d]
                            int gj_amp = offset_j + b*2*chi_r + q*chi_r + d;
                            fp64 ajr = read_re(b0,b1,b2,b3,b4,b5,b6,b7, gj_amp);
                            fp64 aji = read_im(b0,b1,b2,b3,b4,b5,b6,b7, gj_amp);

                            // G * A_i * A_j  (complex triple product)
                            fp64 air_re = gr*air - gi*aii;
                            fp64 air_im = gr*aii + gi*air;
                            sum_re += air_re*ajr - air_im*aji;
                            sum_im += air_re*aji + air_im*ajr;
                        }
                    }
                }
                theta_re[a][s][d] = sum_re;
                theta_im[a][s][d] = sum_im;
            }
        }
    }

    // Write theta to dedicated HBM workspace (host reads back to run scipy SVD)
LOOP_A_WRITE:
    for (int a = 0; a < chi_l && a < CHI_MAX_SVD; a++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64 avg=32
LOOP_S_WRITE:
        for (int s = 0; s < 4; s++) {
LOOP_D_WRITE:
            for (int d = 0; d < chi_r && d < CHI_MAX_SVD; d++) {
#pragma HLS PIPELINE II=8
#pragma HLS LOOP_TRIPCOUNT min=1 max=64 avg=32
                int g = theta_off + a*4*chi_r + s*chi_r + d;
                write_amp(b0,b1,b2,b3,b4,b5,b6,b7, g,
                          theta_re[a][s][d], theta_im[a][s][d]);
            }
        }
    }
}

// SVD is handled entirely on the host (scipy.linalg.svd via LAPACK).
// The FPGA kernel only contracts + applies the gate → theta, then writes
// theta to HBM.  The host reads theta, runs SVD, and writes back A_i, A_j.
// This keeps all double-precision divide/sqrt off the FPGA critical path,
// which was the root cause of WNS = -149 ns at 300 MHz in the first build.

// (jacobi_svd and apply_2q_full removed — SVD runs on host CPU via scipy)

// ─────────────────────────────────────────────────────────────────────────────
//  Top-level kernel (extern "C" prevents C++ name mangling with XRT)
//
//  AXI master ports (= 9 total):
//    bank0 .. bank7  : tensor banks (8 × m_axi)
//    gate_seq        : gate instructions + metadata (1 × m_axi, bundle=gate_bundle)
//
//  AXI-Lite scalar arguments:
//    num_gates, nq, chi_max
// ─────────────────────────────────────────────────────────────────────────────

extern "C" void fpga_mps_simulator(
        fp64 *bank0,   //  HBM tensor bank 0
        fp64 *bank1,   //  HBM tensor bank 1
        fp64 *bank2,   //  HBM tensor bank 2
        fp64 *bank3,   //  HBM tensor bank 3
        fp64 *bank4,   //  HBM tensor bank 4
        fp64 *bank5,   //  HBM tensor bank 5
        fp64 *bank6,   //  HBM tensor bank 6
        fp64 *bank7,   //  HBM tensor bank 7
        int  *gate_seq,  //  gate instructions (int32 words)
        int   num_gates,
        int   nq,
        int   chi_max) {

#pragma HLS INTERFACE m_axi port=bank0    bundle=bank0_axi  offset=slave
#pragma HLS INTERFACE m_axi port=bank1    bundle=bank1_axi  offset=slave
#pragma HLS INTERFACE m_axi port=bank2    bundle=bank2_axi  offset=slave
#pragma HLS INTERFACE m_axi port=bank3    bundle=bank3_axi  offset=slave
#pragma HLS INTERFACE m_axi port=bank4    bundle=bank4_axi  offset=slave
#pragma HLS INTERFACE m_axi port=bank5    bundle=bank5_axi  offset=slave
#pragma HLS INTERFACE m_axi port=bank6    bundle=bank6_axi  offset=slave
#pragma HLS INTERFACE m_axi port=bank7    bundle=bank7_axi  offset=slave
#pragma HLS INTERFACE m_axi port=gate_seq bundle=gate_bundle offset=slave

// All AXI-Lite offset registers and scalars must share one bundle (Vitis rule).
// Listing each pointer here merges its base-address register into "control".
#pragma HLS INTERFACE s_axilite port=bank0    bundle=control
#pragma HLS INTERFACE s_axilite port=bank1    bundle=control
#pragma HLS INTERFACE s_axilite port=bank2    bundle=control
#pragma HLS INTERFACE s_axilite port=bank3    bundle=control
#pragma HLS INTERFACE s_axilite port=bank4    bundle=control
#pragma HLS INTERFACE s_axilite port=bank5    bundle=control
#pragma HLS INTERFACE s_axilite port=bank6    bundle=control
#pragma HLS INTERFACE s_axilite port=bank7    bundle=control
#pragma HLS INTERFACE s_axilite port=gate_seq bundle=control
#pragma HLS INTERFACE s_axilite port=num_gates bundle=control
#pragma HLS INTERFACE s_axilite port=nq        bundle=control
#pragma HLS INTERFACE s_axilite port=chi_max   bundle=control
#pragma HLS INTERFACE s_axilite port=return    bundle=control

    // BRAM gate buffer: load all instructions before execution
    static int gbuf[MAX_GATES_BRAM * (INSTR_WORDS + 32)];
#pragma HLS BIND_STORAGE variable=gbuf type=ram_2p impl=bram

    int n_load = (num_gates < MAX_GATES_BRAM) ? num_gates : MAX_GATES_BRAM;
    int words_per_gate = INSTR_WORDS + 32;  // 12 header + 32 gate-matrix words

LOAD_LOOP:
    for (int i = 0; i < n_load * words_per_gate; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=22528  // MAX_GATES_BRAM * 44
        gbuf[i] = gate_seq[i];
    }

EXEC_LOOP:
    for (int gi = 0; gi < n_load; gi++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
        int base = gi * words_per_gate;
        int opcode  = gbuf[base + 0];
        int site_i  = gbuf[base + 1];
        int site_j  = gbuf[base + 2];
        int chi_l   = gbuf[base + 3];
        int chi_m   = gbuf[base + 4];
        int chi_r     = gbuf[base + 5];
        int theta_off = gbuf[base + 6];  // dedicated HBM workspace for theta output
        fp64 p0 = extract_f32(gbuf[base + 7]);
        fp64 p1 = extract_f32(gbuf[base + 8]);
        fp64 p2 = extract_f32(gbuf[base + 9]);
        int off_i   = gbuf[base + 10];
        int off_j   = gbuf[base + 11];

        // Decode gate matrix (words 12..43, as float32 pairs: re0,im0,re1,im1...)
        fp64 gmat[32];
#pragma HLS ARRAY_PARTITION variable=gmat complete
        for (int k = 0; k < 32; k++) {
#pragma HLS UNROLL
            gmat[k] = extract_f32(gbuf[base + INSTR_WORDS + k]);
        }

        if (opcode <= 16) {
            // 1-qubit gate
            fp64 mat[8];
#pragma HLS ARRAY_PARTITION variable=mat complete
            compute_1q_matrix(opcode, p0, p1, p2, mat);
            apply_1q_gate(bank0, bank1, bank2, bank3,
                          bank4, bank5, bank6, bank7,
                          chi_l, chi_r, off_i, mat);
        }
        else if (opcode == 100 || opcode == 102) {
            // 2q: contract + apply gate → write theta to HBM workspace.
            // Host reads theta_off, runs scipy SVD, writes back A_i and A_j.
            // opcode 102 (formerly on-FPGA Jacobi SVD) is now identical to 100:
            // all SVD is handled by the host to keep FP divide off the FPGA.
            contract_and_apply_2q(bank0, bank1, bank2, bank3,
                                  bank4, bank5, bank6, bank7,
                                  chi_l, chi_m, chi_r, off_i, off_j,
                                  theta_off, gmat);
        }
        // opcode 110 (left-normalize) is handled by host LAPACK QR — no-op here
    }
}
