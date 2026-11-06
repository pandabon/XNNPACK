// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-qf8w-gemm/MRxNRv-rvvzvfofp8.c.in
//   Generator: tools/xngen
//
// Copyright 2024 SiFive, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// bf16 activations x fp8 (OFP8 E4M3) weights with one f32 scale per output channel -> f32, for Penguin's Saturn.
//
// Derived from our own src/bf16-f32-gemm/MRxNRv-rvvzvfbfa.c.in (fork pandabon/XNNPACK branch penguin @ ea4eefe79),
// itself derived from XNNPACK src/f32-gemm/MRxNRv-rvv.c.in (Copyright 2024 SiFive, Inc.) @ 234f87b88. Read the header
// of that template first: it explains the Zvfbfa vfwmacc.vf k loop, the register-form vsetvl with vtype.altfmt = 1,
// the 30-operand limit of GCC inline asm and why the row pointers come in as an array. What is new here:
//  * B is stored as fp8 (1 byte per weight). Each k step loads 32 fp8 bytes (vle8.v, e8 m1, altfmt = 0 = E4M3) and
//    widens them with vfwcvtbf16.f.f.v (zvfofp8min: OFP8 at SEW 8 -> bf16 at SEW 16, exact) into an e16 m2 group.
//    The widened group then feeds the same vfwmacc.vf as the bf16 kernel (e16 m2, altfmt = 1).
//  * Zvfofp8min is not known to GCC 14 / binutils, so the convert is written with .insn. The encoding is taken from
//    Saturn's own benchmark header: ucb-bar/saturn-vectors (as vendored in tapeout-chipyard) generators/saturn/
//    benchmarks/common/rvv_mx.h @ saturn-vectors de04946, macro VFWCVTBF16_F_F_V (lines 124-140):
//    `.insn r 0x57, 0x1, 0x25, vd, x13, vs2` (vs1 field 13 selects vfwcvtbf16). We checked it on our Spike: all 256
//    E4M3 byte values give bf16 bit-identical to PyTorch float8_e4m3fn -> bfloat16.
//    The register names in .insn must be x-numbers, so the two scratch vector registers are fixed (v28-v29 = bf16
//    group, v30 = fp8 bytes) and declared as clobbers; the MR accumulators use v0-v27 (7 * 4 = 28 registers: MR = 7
//    fits exactly).
//  * Two vsetvl per k step (e8 m1 altfmt 0 for the load + convert, e16 m2 altfmt 1 for the FMAs). Both use the saved
//    vl as AVL; vl <= 32 = VLMAX(e8, m1) at VLEN 256, so every vl is the same.
//  * The accumulators start at 0. After the k loop: acc = acc * scale + bias (vfmadd, fused), then the min/max clamp.
//    So the bias is added AFTER the scale (the contract), unlike the bf16 kernel where the bias seeds the sum.
//  * Packed weights (uint8, built offline; contract: the Executorch session's fp8-contract.md, 2026-11-06): per tile
//    of nr = 32 columns, contiguous: float bias[32]; uint8 q[K][32]; float scale[32]. N is padded to a multiple of 32
//    (pad columns: q = 0, s = 0, bias = 0). There is no K padding. Tile bytes = 128 + 32 * K + 128.
//  * Needs -march=rv64gcv_zfh_zvfbfmin (flh) and a core with Zvfbfa and Zvfofp8min; same opt-in as the Zvfbfa kernels
//    (CMake XNNPACK_ENABLE_RISCV_ZVFBFA); the ISA token is `rvvzvfofp8`.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"


void xnn_bf16_qf8w_f32_gemm_minmax_ukernel_7x4v__rvvzvfofp8(
    size_t mr,
    size_t nc,
    size_t kc,
    const xnn_bfloat16* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t unused_cn_stride,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 7);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  const uint16_t* a0 = (const uint16_t*) a;
  float* c0 = c;
  const uint16_t* a1 = (const uint16_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint16_t* a2 = (const uint16_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const uint16_t* a3 = (const uint16_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const uint16_t* a4 = (const uint16_t*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const uint16_t* a5 = (const uint16_t*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const uint16_t* a6 = (const uint16_t*) ((uintptr_t) a5 + a_stride);
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }

  // The asm reads the row pointers from here (see the bf16 template for why an array).
  const uint16_t* a_ptrs[7] = {
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
  };

  const size_t nr = __riscv_vsetvlmax_e32m4();
  // The packed layout is defined for 32 columns per tile (VLEN 256): one e8 m1 load covers a whole tile row.
  assert(nr == 32);
  const size_t k_count = kc / sizeof(uint16_t);
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
    }
    nc = nc - vl;

    // Tile layout: f32 bias[nr] | uint8 q[K][nr] | f32 scale[nr].
    const float* wbias = (const float*) w;
    const uint8_t* wq = (const uint8_t*) ((uintptr_t) w + nr * sizeof(float));
    const float* wscale = (const float*) ((uintptr_t) wq + k_count * nr);

    vfloat32m4_t vacc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t vacc1 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc2 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc3 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc4 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc5 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc6 = __riscv_vmv_v_v_f32m4(vacc0, vl);

    // k loop in inline asm. Operands: vacc* (MR f32 accumulators, LMUL 4), wq (pointer to the fp8 rows, advanced by
    // nr bytes per k), k (A bytes left, counts down by 2). Scratch: v28-v29 (bf16 B row, e16 m2), v30 (fp8 bytes,
    // e8 m1) as clobbers, t0-t6 (row pointers), ft0-ft6 (bf16 A values).
    size_t k = kc;
    size_t vl_saved, vtype_saved;
    __asm__ volatile(
        // Save vl/vtype so they can be restored: the compiler believes they are e32/m4 with this vl.
        "csrr %[vl_saved], vl\n"
        "csrr %[vtype_saved], vtype\n"
        "ld t0, 0(%[a_ptrs])\n"
        "ld t1, 8(%[a_ptrs])\n"
        "ld t2, 16(%[a_ptrs])\n"
        "ld t3, 24(%[a_ptrs])\n"
        "ld t4, 32(%[a_ptrs])\n"
        "ld t5, 40(%[a_ptrs])\n"
        "ld t6, 48(%[a_ptrs])\n"
        "1:\n"
        // Load 32 fp8 bytes (E4M3, altfmt 0): vsetvli e8, m1, ta, ma. Then widen to bf16 into v28-v29:
        // vfwcvtbf16.f.f.v v28, v30 (zvfofp8min; .insn r 0x57, 0x1, 0x25, vd, x13, vs2).
        "vsetvli x0, %[vl_saved], e8, m1, ta, ma\n"
        "vle8.v v30, (%[wq])\n"
        "add %[wq], %[wq], %[wstep]\n"
        ".insn r 0x57, 0x1, 0x25, x28, x13, x30\n"
        // vtype for the FMAs: e16, m2, ta, ma, altfmt = 1 = 0x1C9, with the immediate form of vsetvli written as
        // .insn (the assembler does not know altfmt): .insn i 0x57, 0x7, rd, rs1, vtype.
        ".insn i 0x57, 0x7, x0, %[vl_saved], 0x1C9\n"
        "flh ft0, 0(t0)\n"
        "addi t0, t0, 2\n"
        "flh ft1, 0(t1)\n"
        "addi t1, t1, 2\n"
        "flh ft2, 0(t2)\n"
        "addi t2, t2, 2\n"
        "flh ft3, 0(t3)\n"
        "addi t3, t3, 2\n"
        "flh ft4, 0(t4)\n"
        "addi t4, t4, 2\n"
        "flh ft5, 0(t5)\n"
        "addi t5, t5, 2\n"
        "flh ft6, 0(t6)\n"
        "addi t6, t6, 2\n"
        // acc_m(f32) += bf16(ft_m) * bf16(v28): the Zvfbfa widening FMA (vfwmacc.vf under altfmt = 1).
        "vfwmacc.vf %[vacc0], ft0, v28\n"
        "vfwmacc.vf %[vacc1], ft1, v28\n"
        "vfwmacc.vf %[vacc2], ft2, v28\n"
        "vfwmacc.vf %[vacc3], ft3, v28\n"
        "vfwmacc.vf %[vacc4], ft4, v28\n"
        "vfwmacc.vf %[vacc5], ft5, v28\n"
        "vfwmacc.vf %[vacc6], ft6, v28\n"
        "addi %[k], %[k], -2\n"
        "bnez %[k], 1b\n"
        // Back to the compiler's vtype/vl (rd = x0, rs1 = saved vl register).
        "vsetvl x0, %[vl_saved], %[vtype_saved]\n"
        : [vl_saved] "=&r"(vl_saved), [vtype_saved] "=&r"(vtype_saved),
          [wq] "+r"(wq), [k] "+r"(k)
          , [vacc0] "+vr"(vacc0)
          , [vacc1] "+vr"(vacc1)
          , [vacc2] "+vr"(vacc2)
          , [vacc3] "+vr"(vacc3)
          , [vacc4] "+vr"(vacc4)
          , [vacc5] "+vr"(vacc5)
          , [vacc6] "+vr"(vacc6)
        : [a_ptrs] "r"(a_ptrs), [wstep] "r"(nr)
        : "memory", "vl", "vtype", "t0", "t1", "t2", "t3", "t4", "t5", "t6",
          "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "v28", "v29", "v30");

    // acc = acc * scale + bias (per column), the bias after the scale.
    const vfloat32m4_t vscale = __riscv_vle32_v_f32m4(wscale, vl);
    const vfloat32m4_t vbias = __riscv_vle32_v_f32m4(wbias, vl);
    vacc0 = __riscv_vfmadd_vv_f32m4(vacc0, vscale, vbias, vl);
    vacc1 = __riscv_vfmadd_vv_f32m4(vacc1, vscale, vbias, vl);
    vacc2 = __riscv_vfmadd_vv_f32m4(vacc2, vscale, vbias, vl);
    vacc3 = __riscv_vfmadd_vv_f32m4(vacc3, vscale, vbias, vl);
    vacc4 = __riscv_vfmadd_vv_f32m4(vacc4, vscale, vbias, vl);
    vacc5 = __riscv_vfmadd_vv_f32m4(vacc5, vscale, vbias, vl);
    vacc6 = __riscv_vfmadd_vv_f32m4(vacc6, vscale, vbias, vl);

    // Next tile: bias row + K rows of nr fp8 + scale row.
    w = (const void*) ((uintptr_t) w + 2 * nr * sizeof(float) + k_count * nr);

    // clamp results with min & max
    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc1 = __riscv_vfmax_vf_f32m4(vacc1, vmin, vl);
    vacc2 = __riscv_vfmax_vf_f32m4(vacc2, vmin, vl);
    vacc3 = __riscv_vfmax_vf_f32m4(vacc3, vmin, vl);
    vacc4 = __riscv_vfmax_vf_f32m4(vacc4, vmin, vl);
    vacc5 = __riscv_vfmax_vf_f32m4(vacc5, vmin, vl);
    vacc6 = __riscv_vfmax_vf_f32m4(vacc6, vmin, vl);

    vacc0 = __riscv_vfmin_vf_f32m4(vacc0, vmax, vl);
    vacc1 = __riscv_vfmin_vf_f32m4(vacc1, vmax, vl);
    vacc2 = __riscv_vfmin_vf_f32m4(vacc2, vmax, vl);
    vacc3 = __riscv_vfmin_vf_f32m4(vacc3, vmax, vl);
    vacc4 = __riscv_vfmin_vf_f32m4(vacc4, vmax, vl);
    vacc5 = __riscv_vfmin_vf_f32m4(vacc5, vmax, vl);
    vacc6 = __riscv_vfmin_vf_f32m4(vacc6, vmax, vl);
    // store 7 x vl results to c, then step to the next tile of nr columns (nr floats, as the bf16 kernels do;
    // the contract's cn_stride is 32 * sizeof(float), the same).
    __riscv_vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + nr * sizeof(float));
    __riscv_vse32_v_f32m4(c1, vacc1, vl);
    c1 = (float*) ((uintptr_t) c1 + nr * sizeof(float));
    __riscv_vse32_v_f32m4(c2, vacc2, vl);
    c2 = (float*) ((uintptr_t) c2 + nr * sizeof(float));
    __riscv_vse32_v_f32m4(c3, vacc3, vl);
    c3 = (float*) ((uintptr_t) c3 + nr * sizeof(float));
    __riscv_vse32_v_f32m4(c4, vacc4, vl);
    c4 = (float*) ((uintptr_t) c4 + nr * sizeof(float));
    __riscv_vse32_v_f32m4(c5, vacc5, vl);
    c5 = (float*) ((uintptr_t) c5 + nr * sizeof(float));
    __riscv_vse32_v_f32m4(c6, vacc6, vl);
    c6 = (float*) ((uintptr_t) c6 + nr * sizeof(float));
  } while (nc != 0);
}
