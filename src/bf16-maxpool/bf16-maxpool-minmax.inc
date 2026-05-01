// clang-format off
// Copyright 2026 UC Berkeley BAR
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// X-macro registration list for the bf16 maxpool family. Mirrors the
// shape of src/f16-maxpool/f16-maxpool-minmax.inc and
// src/f32-maxpool/f32-maxpool-minmax.inc. Included from
// src/xnnpack/maxpool.h alongside the f16/f32/s8/u8 lists, and from
// test/maxpool-minmax.cc for the parametrized test driver.
//
// Per Plan §1C ("below-is-lenet-stats-crispy-falcon"): we ship only the
// combined Zvfbfmin+Zvfbfa variant for ladder symmetry with the
// bf16-{,f32-}gemm families. The kernel itself only literally exercises
// Zvfbfmin instructions (vfwcvtbf16.f.f.v widen + vfncvtbf16.f.f.w narrow);
// see the kernel header comment and docs-ucb-bar/bf16-rvv-kernels-report.md
// §"Cross-cutting design notes" for the rationale on the
// -zvfbfmin-zvfbfa filename suffix convention.

// arch_flags, ukernel, channel_tile, primary_tile, datatype, params_type, init_params

#if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR && XNN_ENABLE_RISCV_ZVFBFMIN && XNN_ENABLE_RISCV_ZVFBFA
XNN_UKERNEL(xnn_arch_riscv_zvfbfmin | xnn_arch_riscv_zvfbfa, xnn_bf16_maxpool_minmax_ukernel_9p__rvv_zvfbfmin_zvfbfa_u1v, (2*xnn_init_hardware_config()->vlenb/sizeof(uint16_t)), 9, xnn_bfloat16, struct xnn_bf16_minmax_params, xnn_init_bf16_minmax_scalar_params)
#endif  // XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR && XNN_ENABLE_RISCV_ZVFBFMIN && XNN_ENABLE_RISCV_ZVFBFA
