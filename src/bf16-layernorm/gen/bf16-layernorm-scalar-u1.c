// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-layernorm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Portable layer normalization of one row, the reference the RVV kernel is tested against.
// Same algorithm and operation order as the RVV kernel in rvv.c.in. Derived from the non-vector branch of
// layernorm() in Tencent ncnn, src/layer/riscv/layernorm_riscv.cpp, commit 706c4d19646796f8d79af568d28beec47d07bc2f
// (BSD-3-Clause, Copyright 2025 Tencent): f32 statistics (mean, then the mean of the squared deviations: two passes),
//   a = 1 / sqrt(var + epsilon), b = -mean * a, y = (x * a + b) * scale + bias.
// For bf16 the row is widened to f32 on load and rounded to nearest even once, on store (xnn_bfloat16_from_float).

#include <assert.h>
#include <math.h>
#include <stddef.h>

#include "src/xnnpack/layernorm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_bf16_layernorm_ukernel__scalar_u1(
    size_t batch,
    const xnn_bfloat16* input,
    const xnn_bfloat16* scale,
    const xnn_bfloat16* bias,
    xnn_bfloat16* output,
    const struct xnn_bf16_layernorm_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_bfloat16) == 0);
  assert(input != NULL);
  assert(scale != NULL);
  assert(bias != NULL);
  assert(output != NULL);

  const size_t channels = batch / sizeof(xnn_bfloat16);
  const float vepsilon = params->scalar.epsilon;

  #define LOAD(p) xnn_bfloat16_to_float(p)
  #define STORE(v) xnn_bfloat16_from_float(v)

  // Pass 1: mean.
  float vsum = 0.0f;
  for (size_t c = 0; c < channels; c++) {
    vsum += LOAD(input[c]);
  }
  const float vmean = vsum / (float) channels;

  // Pass 2: variance (biased, centered on the mean).
  float vsqsum = 0.0f;
  for (size_t c = 0; c < channels; c++) {
    const float vd = LOAD(input[c]) - vmean;
    vsqsum += vd * vd;
  }
  const float vvar = vsqsum / (float) channels;

  const float va = 1.0f / sqrtf(vvar + vepsilon);
  const float vb = -vmean * va;

  // Pass 3: normalize, scale and shift.
  for (size_t c = 0; c < channels; c++) {
    const float vy = (LOAD(input[c]) * va + vb) * LOAD(scale[c]) + LOAD(bias[c]);
    output[c] = STORE(vy);
  }

  #undef LOAD
  #undef STORE
}
