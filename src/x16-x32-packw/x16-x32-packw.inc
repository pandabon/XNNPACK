// clang-format off
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// arch_flags, ukernel, nr, kr, sr, kblock, nr_scale
XNN_UKERNEL(xnn_arch_none, xnn_x16_x32_packw_gemm_goi_ukernel_x32c2__scalar, 32, 2, 1, 32, 1)
XNN_GIO_UKERNEL(xnn_arch_none, xnn_x16_x32_packw_gemm_gio_ukernel_x32c2__scalar, 32, 2, 1, 32, 1)
// RVV: dynamic NR (nr=vsetvlmax_e32m4 at runtime), bf16 weights + fp32 bias
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x16_x32_packw_gemm_goi_ukernel_x4v__rvv_u8, 0, 1, 1, 1, 1)
