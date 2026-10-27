// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-f32-gemm/MRxNRv-rvvzvfbfa.c.in
//   Generator: tools/xngen
//
// Copyright 2024 SiFive, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Derived from, and structured exactly like, our own src/bf16-f32-gemm/MRxNRv-rvv.c.in (fork pandabon/XNNPACK
// branch penguin @ 27ac2acb4), itself derived from XNNPACK src/f32-gemm/MRxNRv-rvv.c.in
// (Copyright 2024 SiFive, Inc.) @ 234f87b88. Only the k loop is new: it is one inline-asm block that uses the
// Zvfbfa widening FMA directly on bf16 data.
//
// The k-loop structure (scalar bf16 A broadcast with a .vf instruction, one LMUL-2 bf16 load of B per k, widening
// FMA into LMUL-4 f32 accumulators, K unrolled by 1) follows these references (structure only, no code copied):
//  * SiFive Kernel Library (SKL), src/gemm/gemm_bf16_bf16_f32_zvfbfwma.c @ a3cb2cf34e67
//  * IREE runtime/src/iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_zvfbfwma.c @ 69e0d66e
// Both use vfwmaccbf16.vf (Zvfbfwma). Penguin's Saturn does not implement Zvfbfwma; it implements Zvfbfa (draft
// v0.1, see below), so we use vfwmacc.vf with vtype.altfmt = 1 instead.
//
// How it works (for readers new to this):
//  * Zvfbfa adds a bit to vtype, `altfmt` (bit 8). With SEW = 16 and altfmt = 1 the floating-point vector
//    instructions read their 16-bit elements as bf16 instead of IEEE binary16. So at e16/m2/altfmt=1
//      vfwmacc.vf vd, fs1, vs2      means      vd(f32, LMUL 4) += bf16(fs1) * bf16(vs2)   (fused, f32 accumulation)
//    The scalar fs1 is a bf16 in the low 16 bits of an f register, NaN-boxed; `flh` loads it that way.
//    Spike implements this (riscv-isa-sim riscv/v_ext_macros.h VI_VFP_VF_LOOP_WIDE, `P.VU.altfmt ? bf16_to_f32`).
//  * Why this beats the plain-RVV kernel: no widening of B (vzext + vsll, an LMUL-4 temporary) is needed, so the
//    register budget is MR * 4 accumulator registers + 2 for the bf16 B row (LMUL 2) = 30 for MR = 7, no spills.
//  * The compiler (GCC 14.3) has no intrinsics for this and the assembler does not know zvfbfa, so the k loop is
//    inline asm made only of instructions the assembler does know (vsetvl, vle16.v, vfwmacc.vf, flh); the altfmt
//    bit is set through the register form of vsetvl, whose vtype value we compute ourselves:
//      vtype = vlmul(m2 = 1) | vsew(e16 = 1) << 3 | vta << 6 | vma << 7 | altfmt << 8 = 1 + 8 + 64 + 128 + 256 = 0x1C9
//    (RVV 1.0 vtype layout; altfmt at bit 8 is what Spike's vector_unit.cc decodes.)
//  * The asm saves vl and vtype on entry and restores them on exit, so the compiler's own vsetvl bookkeeping
//    (e32/m4 for the bias load, the clamp and the store) stays valid.
//  * Packed weights layout (kr = 1, sr = 1, from xnn_pack_bf16_f32_gemm_goi_w / _gio_w): per block of nr columns,
//    first nr f32 biases, then for every k one row of nr bf16 weights (the last block is zero-padded to nr).
//  * A (activations) is bf16: one scalar `flh` per row and k.
//  * Accumulation and output are f32; the clamp to [min, max] is the same as the f32 kernel.
//  * Needs -march=rv64gcv_zfh_zvfbfmin (flh) and a core with Zvfbfa; the ISA token is `rvvzvfbfa`.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"


void xnn_bf16_f32_gemm_minmax_ukernel_7x4v__rvvzvfbfa(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint16_t* restrict a,
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
  const uint16_t* a0 = a;
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

  // The asm reads the row pointers from here (see the k loop).
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
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
    }
    nc = nc - vl;

    // f32 biases come first in each packed block of nr columns.
    const float* wf = (const float*) w;
    vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(wf, vl);
    const uint16_t* wh = (const uint16_t*) ((uintptr_t) w + nr * sizeof(float));
    vfloat32m4_t vacc1 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc2 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc3 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc4 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc5 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc6 = __riscv_vmv_v_v_f32m4(vacc0, vl);

    // k loop in inline asm (see the header). GCC allows at most 30 asm operands and a "+" operand counts twice
    // (it is an output and an input), so the MR accumulators (2 * MR) leave too little room for 2 * MR more operands
    // for the A pointers and the scalar bf16 values. Therefore:
    //  * the row pointers come in through the array a_ptrs; the asm loads them into scratch registers t0-t6 and
    //    advances those copies, so a0..a6 (and the array) stay untouched and need no reset after the loop;
    //  * the scalar bf16 A values use the scratch float registers ft0-ft6 (named clobbers do not count as operands).
    // Operands:
    //  vacc*  : the MR f32 accumulators (LMUL 4), read and written ("+vr" = any vector register).
    //  vb     : early-clobber scratch for one row of nr bf16 weights (LMUL 2); it is written while the asm still
    //           reads its inputs, so it must not share a register with them.
    //  wh     : pointer to the bf16 weights, advanced by wstep = nr * 2 bytes per k.
    //  k      : bytes of A left (counts down by 2 per bf16).
    size_t k = kc;
    const size_t wstep = nr * sizeof(uint16_t);
    size_t vl_saved, vtype_saved;
    vuint16m2_t vb;
    __asm__ volatile(
        // Save vl/vtype so they can be restored: the compiler believes they are e32/m4 with this vl.
        "csrr %[vl_saved], vl\n"
        "csrr %[vtype_saved], vtype\n"
        // vtype for the loop: e16, m2, ta, ma, altfmt = 1 (see the header). In the register form of vsetvl the AVL is
        // the old vl, and vl = min(AVL, VLMAX(e16, m2)) = old vl because VLMAX(e16, m2) == VLMAX(e32, m4). So the
        // m2 bf16 row has exactly as many elements as each m4 f32 accumulator. rd = x0: the new vl is not needed.
        "li t0, 0x1C9\n"
        "vsetvl x0, %[vl_saved], t0\n"
        // Load the MR row pointers into t0-t6 (t0 is free again after the vsetvl).
        "ld t0, 0(%[a_ptrs])\n"
        "ld t1, 8(%[a_ptrs])\n"
        "ld t2, 16(%[a_ptrs])\n"
        "ld t3, 24(%[a_ptrs])\n"
        "ld t4, 32(%[a_ptrs])\n"
        "ld t5, 40(%[a_ptrs])\n"
        "ld t6, 48(%[a_ptrs])\n"
        "1:\n"
        // vb = next row of nr bf16 weights (e16, m2, vl elements); then step the weight pointer.
        "vle16.v %[vb], (%[wh])\n"
        "add %[wh], %[wh], %[wstep]\n"
        // One bf16 of A per row into a float register: flh NaN-boxes the 16-bit value, which a .vf instruction at
        // SEW=16 requires. Then step each A pointer by one bf16.
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
        // acc_m(f32) += bf16(ft_m) * bf16(vb): the Zvfbfa widening FMA (vfwmacc.vf under altfmt = 1).
        "vfwmacc.vf %[vacc0], ft0, %[vb]\n"
        "vfwmacc.vf %[vacc1], ft1, %[vb]\n"
        "vfwmacc.vf %[vacc2], ft2, %[vb]\n"
        "vfwmacc.vf %[vacc3], ft3, %[vb]\n"
        "vfwmacc.vf %[vacc4], ft4, %[vb]\n"
        "vfwmacc.vf %[vacc5], ft5, %[vb]\n"
        "vfwmacc.vf %[vacc6], ft6, %[vb]\n"
        "addi %[k], %[k], -2\n"
        "bnez %[k], 1b\n"
        // Back to the compiler's vtype/vl (rd = x0, rs1 = saved vl register: vl = min(saved vl, VLMAX)).
        "vsetvl x0, %[vl_saved], %[vtype_saved]\n"
        : [vl_saved] "=&r"(vl_saved), [vtype_saved] "=&r"(vtype_saved), [vb] "=&vr"(vb),
          [wh] "+r"(wh), [k] "+r"(k)
          , [vacc0] "+vr"(vacc0)
          , [vacc1] "+vr"(vacc1)
          , [vacc2] "+vr"(vacc2)
          , [vacc3] "+vr"(vacc3)
          , [vacc4] "+vr"(vacc4)
          , [vacc5] "+vr"(vacc5)
          , [vacc6] "+vr"(vacc6)
        : [a_ptrs] "r"(a_ptrs), [wstep] "r"(wstep)
        : "memory", "vl", "vtype", "t0", "t1", "t2", "t3", "t4", "t5", "t6",
          "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");
    // Next block of nr columns starts after this block's bias row and its kc/2 rows of nr bf16.
    w = (const void*) ((uintptr_t) w + nr * sizeof(float) + (kc / sizeof(uint16_t)) * nr * sizeof(uint16_t));
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
    // store 7 x vl results to c, then step to the next block of nr columns. Like the other bf16-f32 GEMM
    // kernels (e.g. bf16-f32-gemm-1x4c2-minmax-scalar.c: c0 += 4), the step is nr floats, NOT cn_stride:
    // upstream's tester passes cn_stride = 0 for this kernel type (gemm-microkernel-tester.cc, bf16_f32 Test).
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
