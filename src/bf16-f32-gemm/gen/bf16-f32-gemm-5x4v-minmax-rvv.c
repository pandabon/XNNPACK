// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-f32-gemm/MRxNRv-rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 SiFive, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Derived from, and structured exactly like, XNNPACK src/f32-gemm/MRxNRv-rvv.c.in
// (Copyright 2024 SiFive, Inc.), fork pandabon/XNNPACK branch penguin @ 234f87b88.
// The bf16 parts (widen the packed bf16 weights to f32 in registers, then use ordinary f32 FMA) follow the
// "widen once, then f32 math" approach of OpenBLAS kernel/riscv64/sbgemm_kernel_16x8_zvl256b.c
// @ 5137bb62b4ca (BSD-3-Clause, Copyright The OpenBLAS Project contributors), in its integer fallback
// (USE_BF16_CVT undefined), which is the same integer widening as Tencent ncnn
// src/layer/riscv/riscv_usability.h bfloat2float_rvv() @ 7e36d25cb153092ba87a60530455c7376fedf452
// (Copyright 2021 Tencent, BSD-3-Clause) that our src/bf16-f32-vcvt/rvv.c.in already uses.
//
// How it works (for readers new to this):
//  * A bf16 value is the top 16 bits of an f32, so widening is exact: zero-extend the 16-bit pattern to 32 bits
//    and shift it left by 16 (vzext.vf2 + vsll). No bf16 vector extension (Zvfbfmin/Zvfbfa) is needed: only
//    standard RVV 1.0 intrinsics, so the file keeps the plain `rvv` ISA token (-march=rv64gcv).
//  * Packed weights layout (kr = 1, sr = 1, from xnn_pack_bf16_f32_gemm_goi_w / _gio_w): per block of nr columns,
//    first nr f32 biases, then for every k one row of nr bf16 weights (the last block is zero-padded to nr).
//  * A (activations) is bf16 too: one scalar load per row and k, widened with a scalar shift (a bf16 is the top half
//    of an f32; uint32_as_float is XNNPACK's own union cast), then vfmacc.vf.
//  * Accumulation and output are f32; the clamp to [min, max] is the same as the f32 kernel.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"


void xnn_bf16_f32_gemm_minmax_ukernel_5x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint16_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t unused_cn_stride,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 5);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  const uint16_t* a0 = a;
  float* c0 = c;
  const uint16_t* a1 = (const uint16_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint16_t* a2 = (const uint16_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const uint16_t* a3 = (const uint16_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const uint16_t* a4 = (const uint16_t*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
    }
    nc = nc - vl;

    // f32 biases come first in each packed block of nr columns.
    const float* wf = (const float*) w;
    vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(wf, vl);
    const uint16_t* wh = (const uint16_t*) ((uintptr_t) w + nr * sizeof(float));
    vfloat32m4_t vacc1 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc2 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc3 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc4 = __riscv_vmv_v_v_f32m4(vacc0, vl);

    size_t k = kc;
    do {
      const float va0 = uint32_as_float((uint32_t) *a0++ << 16);
      const float va1 = uint32_as_float((uint32_t) *a1++ << 16);
      const float va2 = uint32_as_float((uint32_t) *a2++ << 16);
      const float va3 = uint32_as_float((uint32_t) *a3++ << 16);
      const float va4 = uint32_as_float((uint32_t) *a4++ << 16);
      // One row of nr packed bf16 weights, widened once and shared by all 5 rows of A.
      const vuint16m2_t vb16 = __riscv_vle16_v_u16m2(wh, vl);
      wh = wh + nr;
      const vuint32m4_t vb32 = __riscv_vsll_vx_u32m4(__riscv_vwcvtu_x_x_v_u32m4(vb16, vl), 16, vl);
      const vfloat32m4_t vb = __riscv_vreinterpret_v_u32m4_f32m4(vb32);
      vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va0, vb, vl);
      vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, va1, vb, vl);
      vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, va2, vb, vl);
      vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, va3, vb, vl);
      vacc4 = __riscv_vfmacc_vf_f32m4(vacc4, va4, vb, vl);
      k -= sizeof(uint16_t);
    } while (k != 0);
    // Next block of nr columns starts after this block's bias row and its kc/2 rows of nr bf16.
    w = (const void*) ((uintptr_t) w + nr * sizeof(float) + (kc / sizeof(uint16_t)) * nr * sizeof(uint16_t));
    // clamp results with min & max
    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc1 = __riscv_vfmax_vf_f32m4(vacc1, vmin, vl);
    vacc2 = __riscv_vfmax_vf_f32m4(vacc2, vmin, vl);
    vacc3 = __riscv_vfmax_vf_f32m4(vacc3, vmin, vl);
    vacc4 = __riscv_vfmax_vf_f32m4(vacc4, vmin, vl);

    vacc0 = __riscv_vfmin_vf_f32m4(vacc0, vmax, vl);
    vacc1 = __riscv_vfmin_vf_f32m4(vacc1, vmax, vl);
    vacc2 = __riscv_vfmin_vf_f32m4(vacc2, vmax, vl);
    vacc3 = __riscv_vfmin_vf_f32m4(vacc3, vmax, vl);
    vacc4 = __riscv_vfmin_vf_f32m4(vacc4, vmax, vl);
    // store 5 x vl results to c, then step to the next block of nr columns. Like the other bf16-f32 GEMM
    // kernels (e.g. bf16-f32-gemm-1x4c2-minmax-scalar.c: c0 += 4), the step is nr floats, NOT cn_stride:
    // upstream's tester passes cn_stride = 0 for this kernel type (gemm-microkernel-tester.cc, bf16_f32 Test).
    __riscv_vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + nr * sizeof(float));
    __riscv_vse32_v_f32m4(c1, vacc1, vl);
    c1 = (float*) ((uintptr_t) c1 + nr * sizeof(float));
    __riscv_vse32_v_f32m4(c2, vacc2, vl);
    c2 = (float*) ((uintptr_t) c2 + nr * sizeof(float));
    __riscv_vse32_v_f32m4(c3, vacc3, vl);
    c3 = (float*) ((uintptr_t) c3 + nr * sizeof(float));
    __riscv_vse32_v_f32m4(c4, vacc4, vl);
    c4 = (float*) ((uintptr_t) c4 + nr * sizeof(float));
    a0 = (const uint16_t*) ((uintptr_t) a0 - kc);
    a1 = (const uint16_t*) ((uintptr_t) a1 - kc);
    a2 = (const uint16_t*) ((uintptr_t) a2 - kc);
    a3 = (const uint16_t*) ((uintptr_t) a3 - kc);
    a4 = (const uint16_t*) ((uintptr_t) a4 - kc);
  } while (nc != 0);
}
