// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-vbinary/vopc-rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2023 SiFive, Inc.
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// bf16 add / multiply / subtract / reverse-subtract with a scalar second operand (the "c" kernels) for RISC-V
// Vector (plain RVV 1.0, no Zvfbf* needed).
//
// Sources (all in XNNPACK, BSD-style licence, commit 92a7ad501b95 unless noted):
//  * Loop and operations: src/f32-vbinary/vopc-rvv.c.in (Copyright 2023 SiFive, Inc., notice kept).
//  * Widen-compute-narrow structure and the f32 -> bf16 narrowing: see src/bf16-vbinary/vop-rvv.c.in (same sources:
//    google/XNNPACK 1b70146d8351 by Jake Stevens, Meta, Copyright 2026 Google LLC; Tencent ncnn 7e36d25cb153,
//    BSD-3-Clause).
//
// The scalar operand is widened once, in scalar code (bf16 bits << 16, as in xnn_bfloat16_to_float).

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vbinary.h"


void xnn_bf16_vrsubc_ukernel__rvv_u2v(
    size_t batch,
    const xnn_bfloat16* input_a,
    const xnn_bfloat16* input_b,
    xnn_bfloat16* output,
    const struct xnn_bf16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_bfloat16) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float b = xnn_bfloat16_to_float(*input_b);
  size_t n = batch >> XNN_LOG2_SIZEOF_BFLOAT16;

  const uint16_t* a = (const uint16_t*) input_a;
  uint16_t* o = (uint16_t*) output;
  do {
    const size_t vl = __riscv_vsetvl_e32m2(n);
    n -= vl;

    // Load bf16, widen to f32 (zero-extend, shift left by 16).
    const vuint16m1_t va_bf16 = __riscv_vle16_v_u16m1(a, vl);
    a += vl;
    const vfloat32m2_t va = __riscv_vreinterpret_v_u32m2_f32m2(
        __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(va_bf16, vl), 16, vl));

    const vfloat32m2_t vacc = __riscv_vfrsub_vf_f32m2(va, b, vl);

    // Narrow f32 -> bf16: round to nearest even (vnclipu), NaN lanes quieted and truncated
    // (identical to src/f32-bf16-vcvt/rvv.c.in and to the scalar xnn_bfloat16_from_float).
    const vuint32m2_t vbits = __riscv_vreinterpret_v_f32m2_u32m2(vacc);
    vuint16m1_t vy_bf16 = __riscv_vnclipu_wx_u16m1(vbits, 16, __RISCV_VXRM_RNE, vl);
    const vuint32m2_t vabs = __riscv_vand_vx_u32m2(vbits, UINT32_C(0x7FFFFFFF), vl);
    const vbool16_t vnan = __riscv_vmsgtu_vx_u32m2_b16(vabs, UINT32_C(0x7F800000), vl);
    const vuint32m2_t vquiet = __riscv_vor_vx_u32m2(vbits, UINT32_C(0x00400000), vl);
    const vuint16m1_t vnan_y = __riscv_vnsrl_wx_u16m1(vquiet, 16, vl);
    vy_bf16 = __riscv_vmerge_vvm_u16m1(vy_bf16, vnan_y, vnan, vl);

    __riscv_vse16_v_u16m1(o, vy_bf16, vl);
    o += vl;
  } while (n > 0);
}
