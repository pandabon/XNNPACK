// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-vbinary/vopc-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// bf16 add / multiply / subtract / reverse-subtract with a scalar second operand (the "c" kernels), scalar
// fallback. bf16 is a storage format: widen to f32, compute in f32, round back to bf16 (RNE, NaNs quieted).
//
// Sources (all in XNNPACK, BSD-style licence):
//  * Loop structure and tail: src/f32-vbinary/vopc-scalar.c.in (Copyright 2019 Google LLC, notice kept), commit
//    92a7ad501b95.
//  * bf16 widen / narrow with xnn_bfloat16_to_float / xnn_bfloat16_from_float, as in src/bf16-vunary/scalar.c.in
//    (Copyright 2026 Google LLC; google/XNNPACK commit 1b70146d8351, Jake Stevens, Meta).

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vbinary.h"


void xnn_bf16_vsubc_ukernel__scalar_u4(
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

  const float vb = xnn_bfloat16_to_float(*input_b);

  for (; batch >= 4 * sizeof(xnn_bfloat16); batch -= 4 * sizeof(xnn_bfloat16)) {
    const float va0 = xnn_bfloat16_to_float(input_a[0]);
    const float va1 = xnn_bfloat16_to_float(input_a[1]);
    const float va2 = xnn_bfloat16_to_float(input_a[2]);
    const float va3 = xnn_bfloat16_to_float(input_a[3]);
    input_a += 4;

    const float vacc0 = va0 - vb;
    const float vacc1 = va1 - vb;
    const float vacc2 = va2 - vb;
    const float vacc3 = va3 - vb;

    output[0] = xnn_bfloat16_from_float(vacc0);
    output[1] = xnn_bfloat16_from_float(vacc1);
    output[2] = xnn_bfloat16_from_float(vacc2);
    output[3] = xnn_bfloat16_from_float(vacc3);
    output += 4;
  }
  while (batch != 0) {
    const float va = xnn_bfloat16_to_float(*input_a++);
    const float vacc = va - vb;
    *output++ = xnn_bfloat16_from_float(vacc);
    batch -= sizeof(xnn_bfloat16);
  }
}
