// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vsigmoid/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 UC Berkeley BAR
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vunary.h"


// f32 sigmoid via the rr1-p5-div scheme — pure polynomial, no LUT, no
// gather. Constants and skeleton mirror the non-RVV `*-rr1-p5-div`
// variants (e.g. f32-vsigmoid-aarch64-neonfma-rr1-p5-div-u4.c). RVV
// pieces:
//   abs       → vfabs
//   FMA       → vfmacc / vfmadd / vfnmsac
//   exp shift → reinterpret f32m -> i32m, vsll(., 23), reinterpret back
//   sat 0     → vmfgt + vfmerge_vfm with 0.0 (zero where |vx| > cutoff)
//   sym ext   → vmflt(vx, 0) + vmerge_vvm between vf and 1-vf
void xnn_f32_vsigmoid_ukernel__rvv_rr1_p5_div_u4v(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vmagic_bias = 0x1.8000FEp23f;
  const float vminus_log2e = -0x1.715476p0f;
  const float vc5 = -0x1.0F9F9Cp-7f;
  const float vc4 =  0x1.573A1Ap-5f;
  const float vc3 = -0x1.555A80p-3f;
  const float vc2 =  0x1.FFFDC6p-2f;
  const float vc1 = -0x1.FFFFF6p-1f;
  const float vone =  1.0f;
  const float vdenorm_cutoff = 0x1.5D589Ep+6f;
  const float vln2 = 0x1.62E430p-1f;

  batch >>= XNN_LOG2_SIZEOF_FLOAT;
  do {
    const size_t n = __riscv_vsetvl_e32m4(batch);
    vfloat32m4_t vx = __riscv_vle32_v_f32m4(input, n);
    input += n;

    vfloat32m4_t vz = __riscv_vfabs_v_f32m4(vx, n);

    // vn = magic_bias + vz * (-log2e)  (range-reduce: vn ≈ round(vz * log2e))
    vfloat32m4_t vn = __riscv_vfmv_v_f_f32m4(vmagic_bias, n);
    vn = __riscv_vfmacc_vf_f32m4(vn, vminus_log2e, vz, n);

    // vs = (int)vn << 23  reinterpreted as float — yields 2^vn.
    vint32m4_t vn_i = __riscv_vreinterpret_v_f32m4_i32m4(vn);
    vint32m4_t vs_i = __riscv_vsll_vx_i32m4(vn_i, 23, n);
    vfloat32m4_t vs = __riscv_vreinterpret_v_i32m4_f32m4(vs_i);

    // Recover vn as a regular float integer.
    vn = __riscv_vfsub_vf_f32m4(vn, vmagic_bias, n);

    // vt = vz + vn * ln2  (fractional remainder; vn here is signed
    // round(-vz/ln2), so the addition is the correct fold-back).
    vfloat32m4_t vt = __riscv_vfmacc_vf_f32m4(vz, vln2, vn, n);

    // Horner-rule polynomial: vp = c5*vt+c4, then iteratively folded.
    vfloat32m4_t vp = __riscv_vfmv_v_f_f32m4(vc4, n);
    vp = __riscv_vfmacc_vf_f32m4(vp, vc5, vt, n);
    vp = __riscv_vfmadd_vv_f32m4(
        vp, vt, __riscv_vfmv_v_f_f32m4(vc3, n), n);
    vp = __riscv_vfmadd_vv_f32m4(
        vp, vt, __riscv_vfmv_v_f_f32m4(vc2, n), n);
    vp = __riscv_vfmadd_vv_f32m4(
        vp, vt, __riscv_vfmv_v_f_f32m4(vc1, n), n);

    vt = __riscv_vfmul_vv_f32m4(vt, vs, n);

    // ve = vs + vp * vt  ≈ exp(vz)
    vfloat32m4_t ve = __riscv_vfmadd_vv_f32m4(vp, vt, vs, n);

    // sigmoid(vz) = ve / (ve + 1)
    vfloat32m4_t vd = __riscv_vfadd_vf_f32m4(ve, vone, n);
    vfloat32m4_t vf = __riscv_vfdiv_vv_f32m4(ve, vd, n);

    // Saturate to 0 for |vx| beyond the denormal cutoff (matches the
    // upstream variants — well below the point where exp underflows
    // below the f32 denorm boundary).
    vbool8_t mcutoff =
        __riscv_vmfgt_vf_f32m4_b8(vz, vdenorm_cutoff, n);
    vf = __riscv_vfmerge_vfm_f32m4(vf, 0.0f, mcutoff, n);

    // Symmetric extension: sigmoid(vx) = (vx<0) ? vf : 1-vf, where vf
    // is sigmoid(|vx|).
    vfloat32m4_t vone_minus_vf =
        __riscv_vfrsub_vf_f32m4(vf, vone, n);
    vbool8_t mneg =
        __riscv_vmflt_vf_f32m4_b8(vx, 0.0f, n);
    // vmerge_vvm: result[i] = mask[i] ? op2[i] : op1[i].
    // mneg[i]=1 when vx<0 → pick vf (sigmoid of positive |vx|);
    // mneg[i]=0 when vx>=0 → pick 1-vf.
    vf = __riscv_vmerge_vvm_f32m4(vone_minus_vf, vf, mneg, n);

    __riscv_vse32_v_f32m4(output, vf, n);
    output += n;

    batch -= n;
  } while (batch != 0);
}
