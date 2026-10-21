// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-vbinary/vop-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// bf16 add / multiply / subtract, scalar fallback. bf16 is a storage format: each element is widened to f32, the
// operation is done in f32, and the result is rounded back to bf16 (round to nearest even, NaNs quieted).
//
// Sources (all in XNNPACK, BSD-style licence):
//  * Loop structure and tail: src/f32-vbinary/vop-scalar.c.in (Copyright 2019 Google LLC, notice kept), commit
//    92a7ad501b95.
//  * bf16 widen / narrow with xnn_bfloat16_to_float / xnn_bfloat16_from_float, as in src/bf16-vunary/scalar.c.in
//    (Copyright 2026 Google LLC; google/XNNPACK commit 1b70146d8351, Jake Stevens, Meta).

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vbinary.h"


void xnn_bf16_vmul_ukernel__scalar_u4(
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

  for (; batch >= 4 * sizeof(xnn_bfloat16); batch -= 4 * sizeof(xnn_bfloat16)) {
    const float va0 = xnn_bfloat16_to_float(input_a[0]);
    const float va1 = xnn_bfloat16_to_float(input_a[1]);
    const float va2 = xnn_bfloat16_to_float(input_a[2]);
    const float va3 = xnn_bfloat16_to_float(input_a[3]);
    input_a += 4;

    const float vb0 = xnn_bfloat16_to_float(input_b[0]);
    const float vb1 = xnn_bfloat16_to_float(input_b[1]);
    const float vb2 = xnn_bfloat16_to_float(input_b[2]);
    const float vb3 = xnn_bfloat16_to_float(input_b[3]);
    input_b += 4;

    const float vacc0 = va0 * vb0;
    const float vacc1 = va1 * vb1;
    const float vacc2 = va2 * vb2;
    const float vacc3 = va3 * vb3;

    output[0] = xnn_bfloat16_from_float(vacc0);
    output[1] = xnn_bfloat16_from_float(vacc1);
    output[2] = xnn_bfloat16_from_float(vacc2);
    output[3] = xnn_bfloat16_from_float(vacc3);
    output += 4;
  }
  while (batch != 0) {
    const float va = xnn_bfloat16_to_float(*input_a++);
    const float vb = xnn_bfloat16_to_float(*input_b++);
    const float vacc = va * vb;
    *output++ = xnn_bfloat16_from_float(vacc);
    batch -= sizeof(xnn_bfloat16);
  }
}
