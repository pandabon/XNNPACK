// clang-format off
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Arguments are:
// XNN_UKERNEL(arch, ukernel, nr, kr, sr, kblock, nr_scale)
XNN_UKERNEL(xnn_arch_none, xnn_x16_x32_packw_gemm_goi_ukernel_x32c2__scalar, 32, 2, 1, 32, 1)
XNN_GIO_UKERNEL(xnn_arch_none, xnn_x16_x32_packw_gemm_gio_ukernel_x32c2__scalar, 32, 2, 1, 32, 1)

#if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x16_x32_packw_gemm_goi_ukernel_x4v__rvv_u2, 4, 1, 1, 2, xnn_init_hardware_config()->vlenb / sizeof(uint32_t))
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x16_x32_packw_gemm_goi_ukernel_x4v__rvv_u4, 4, 1, 1, 4, xnn_init_hardware_config()->vlenb / sizeof(uint32_t))
#endif  // XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
