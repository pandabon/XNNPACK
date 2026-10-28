// clang-format off
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_f32_layernorm_ukernel__rvv_u8v, 8, true, float, struct xnn_f32_layernorm_params, xnn_init_f32_layernorm_scalar_params)
#endif  // XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR

XNN_UKERNEL(xnn_arch_none, xnn_f32_layernorm_ukernel__scalar_u1, 1, false, float, struct xnn_f32_layernorm_params, xnn_init_f32_layernorm_scalar_params)
