// clang-format off
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_bf16_vaddc_ukernel__rvv_u2v, 2, true, xnn_bfloat16, struct xnn_bf16_default_params, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_bf16_vaddc_ukernel__rvv_u4v, 4, true, xnn_bfloat16, struct xnn_bf16_default_params, NULL)
#endif  // XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR

XNN_UKERNEL(0, xnn_bf16_vaddc_ukernel__scalar_u4, 4, false, xnn_bfloat16, struct xnn_bf16_default_params, NULL)
