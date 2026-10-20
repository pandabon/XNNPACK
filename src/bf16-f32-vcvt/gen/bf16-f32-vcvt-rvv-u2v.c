// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-f32-vcvt/rvv.c.in
//   Generator: tools/xngen
//
// Kernel structure follows XNNPACK src/f16-f32-vcvt/rvvfp16arith.c.in (Copyright 2024 Imagination Technologies, Inc.).
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Derived from Tencent ncnn, src/layer/riscv/riscv_usability.h, function bfloat2float_rvv(),
// the integer branch (no bf16 intrinsics needed), commit 7e36d25cb153092ba87a60530455c7376fedf452,
// Copyright 2021 Tencent, SPDX-License-Identifier: BSD-3-Clause.
//
// How it works: a bf16 is the top 16 bits of an f32, so widening is exact: zero-extend the 16-bit pattern to
// 32 bits (vwcvtu.x.x = vzext.vf2), shift it left by 16, and reinterpret the bits as f32.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/vcvt.h"


void xnn_bf16_f32_vcvt_ukernel__rvv_u2v(
    size_t batch,
    const xnn_bfloat16* input,
    float* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_bfloat16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_BFLOAT16;

  const uint16_t* i = (const uint16_t*) input;
  for (; batch > 0;) {
    const size_t n = __riscv_vsetvl_e32m2(batch); batch -= n;

    const vuint16m1_t vx = __riscv_vle16_v_u16m1(i, n); i += n;

    // ncnn bfloat2float_rvv: zero-extend, shift left by 16, reinterpret as f32.
    const vuint32m2_t vw = __riscv_vsll_vx_u32m2(__riscv_vwcvtu_x_x_v_u32m2(vx, n), 16, n);

    __riscv_vse32_v_f32m2(output, __riscv_vreinterpret_v_u32m2_f32m2(vw), n); output += n;
  }
}
