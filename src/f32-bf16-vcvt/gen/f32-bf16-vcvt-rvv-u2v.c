// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-bf16-vcvt/rvv.c.in
//   Generator: tools/xngen
//
// Kernel structure follows XNNPACK src/f32-f16-vcvt/rvvfp16arith.c.in (Copyright 2024 Imagination Technologies, Inc.).
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Derived from Tencent ncnn, src/layer/riscv/riscv_usability.h, function float2bfloat_rvv(),
// the integer branch (no bf16 intrinsics needed), commit 7e36d25cb153092ba87a60530455c7376fedf452,
// Copyright 2021 Tencent, SPDX-License-Identifier: BSD-3-Clause.
// The vnclipu rounding line is ncnn's. The NaN fix-up below is NOT in ncnn: it was added so the result is
// bit-exact with XNNPACK's scalar reference math_cvt_bf16_fp32() (src/xnnpack/math.h).
//
// How it works (f32 -> bf16 keeps the top 16 bits of the f32 pattern, rounded to nearest even):
//  * vnclipu.wx with shift 16 and vxrm = RNE drops the low 16 bits and rounds the kept part to nearest, ties
//    to even. This equals the scalar "bits += 0x7FFF + ((bits >> 16) & 1); bits >>= 16".
//    Finite values that round past the largest bf16 become infinity (0x7F7FFFFF -> 0x7F80), as in the scalar.
//  * NaN needs special care. The scalar sets the quiet bit (0x00400000) and then TRUNCATES. Rounding a NaN
//    could turn it into infinity (payload only in the low 16 bits) or into -0.0 (0x7FFFFFFF rounds up to
//    0x8000), so NaN lanes are recomputed with a truncating narrowing shift and the quiet bit set.
//    Infinities are not NaN and are exact under rounding.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/vcvt.h"


void xnn_f32_bf16_vcvt_ukernel__rvv_u2v(
    size_t batch,
    const float* input,
    xnn_bfloat16* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_FLOAT;

  uint16_t* o = (uint16_t*) output;
  for (; batch > 0;) {
    const size_t n = __riscv_vsetvl_e32m2(batch); batch -= n;

    // Load f32 and view the bit patterns as u32.
    const vuint32m2_t vbits = __riscv_vreinterpret_v_f32m2_u32m2(__riscv_vle32_v_f32m2(input, n)); input += n;

    // ncnn float2bfloat_rvv: narrow with shift 16, round to nearest even.
    vuint16m1_t vy = __riscv_vnclipu_wx_u16m1(vbits, 16, __RISCV_VXRM_RNE, n);

    // NaN lanes: |x| > 0x7F800000 as an unsigned integer. Quiet them and truncate (as the scalar reference).
    const vuint32m2_t vabs = __riscv_vand_vx_u32m2(vbits, UINT32_C(0x7FFFFFFF), n);
    const vbool16_t vnan = __riscv_vmsgtu_vx_u32m2_b16(vabs, UINT32_C(0x7F800000), n);
    const vuint32m2_t vquiet = __riscv_vor_vx_u32m2(vbits, UINT32_C(0x00400000), n);
    const vuint16m1_t vnan_y = __riscv_vnsrl_wx_u16m1(vquiet, 16, n);
    vy = __riscv_vmerge_vvm_u16m1(vy, vnan_y, vnan, n);

    __riscv_vse16_v_u16m1(o, vy, n); o += n;
  }
}
