// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_LAYERNORM_H_
#define XNNPACK_SRC_XNNPACK_LAYERNORM_H_

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"  // IWYU pragma: keep

#ifdef __cplusplus
extern "C" {
#endif

// One row per call: batch = row length in bytes, then input, scale (gamma),
// bias (beta), output and the params (epsilon).
#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype,   \
                    params_type, init_params)                                 \
  XNN_INTERNAL void ukernel(size_t batch, const datatype* input,              \
                            const datatype* scale, const datatype* bias,      \
                            datatype* output, const params_type* params);
#include "src/bf16-layernorm/bf16-layernorm.inc"
#include "src/f32-layernorm/f32-layernorm.inc"
#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_LAYERNORM_H_
