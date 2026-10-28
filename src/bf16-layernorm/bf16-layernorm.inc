// clang-format off
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// The kernels in gen/ are expanded from the templates in src/f32-layernorm/ (one template per ISA serves both
// datatypes, see scripts/generate-bf16-layernorm.sh), like the bf16-vsqr kernels come from src/bf16-vunary/.

#if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_bf16_layernorm_ukernel__rvv_u8v, 8, true, xnn_bfloat16, struct xnn_bf16_layernorm_params, xnn_init_bf16_layernorm_scalar_params)
#endif  // XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR

XNN_UKERNEL(xnn_arch_none, xnn_bf16_layernorm_ukernel__scalar_u1, 1, false, xnn_bfloat16, struct xnn_bf16_layernorm_params, xnn_init_bf16_layernorm_scalar_params)
