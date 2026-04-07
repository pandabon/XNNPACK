// Copyright 2025 UC Berkeley BAR
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// MR=4, NR=vsetvlmax_e32m4() bf16->f32 GEMM with minmax epilogue.
// Zvfbfa path: uses vfwmacc.vf with altfmt=1 SEW=16 (native bf16 widening FMA).
//
// Packed weight layout: [fp32 bias x NR][bf16 weights x NR per KC step]

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/microparams.h"

static inline double nanbox_bf16(uint16_t v) {
  uint64_t bits = 0xFFFFFFFFFFFF0000ULL | (uint64_t)v;
  double d;
  __builtin_memcpy(&d, &bits, sizeof(d));
  return d;
}

void xnn_bf16_f32_gemm_minmax_ukernel_4x4v__rvv_zvfbfa(
    size_t mr, size_t nc, size_t kc,
    const uint16_t* restrict a, size_t a_stride,
    const void* restrict w,
    float* restrict c, size_t cm_stride, size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0); assert(mr <= 4);
  assert(nc != 0); assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  const uint16_t* a0 = a;
  float*          c0 = c;
  const uint16_t* a1 = (const uint16_t*)((uintptr_t)a0 + a_stride);
  float*          c1 = (float*)((uintptr_t)c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) { a1 = a0; c1 = c0; }
  const uint16_t* a2 = (const uint16_t*)((uintptr_t)a1 + a_stride);
  float*          c2 = (float*)((uintptr_t)c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) { a2 = a1; c2 = c1; }
  const uint16_t* a3 = (const uint16_t*)((uintptr_t)a2 + a_stride);
  float*          c3 = (float*)((uintptr_t)c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) { a3 = a2; c3 = c2; }

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;
  const xnn_bfloat16* w_ptr = (const xnn_bfloat16*)w;

  do {
    if XNN_UNLIKELY(nc < nr) { vl = __riscv_vsetvl_e32m4(nc); }
    nc -= vl;

    vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4((const float*)w_ptr, vl);
    vfloat32m4_t vacc1 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc2 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc3 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    w_ptr += nr * 2;

    const uint16_t* wb   = (const uint16_t*)w_ptr;
    const uint16_t* a0k  = a0;
    const uint16_t* a1k  = a1;
    const uint16_t* a2k  = a2;
    const uint16_t* a3k  = a3;
    size_t k = kc;

    do {
      uint16_t raw_a0 = *a0k++;
      uint16_t raw_a1 = *a1k++;
      uint16_t raw_a2 = *a2k++;
      uint16_t raw_a3 = *a3k++;

      vuint16m2_t vb = __riscv_vle16_v_u16m2(wb, vl);
      wb += nr;

      double da0 = nanbox_bf16(raw_a0);
      double da1 = nanbox_bf16(raw_a1);
      double da2 = nanbox_bf16(raw_a2);
      double da3 = nanbox_bf16(raw_a3);

      // vfwmacc.vf with altfmt=1: bf16 scalar x bf16 vector -> fp32 accumulate.
      asm volatile(
          ".option push\n\t"
          ".option arch, +zvfbfa\n\t"
          "vfwmacc.vf %[a0], %[s0], %[b]\n\t"
          "vfwmacc.vf %[a1], %[s1], %[b]\n\t"
          "vfwmacc.vf %[a2], %[s2], %[b]\n\t"
          "vfwmacc.vf %[a3], %[s3], %[b]\n\t"
          ".option pop\n\t"
          : [a0] "+vr"(vacc0), [a1] "+vr"(vacc1),
            [a2] "+vr"(vacc2), [a3] "+vr"(vacc3)
          : [s0] "f"(da0), [s1] "f"(da1),
            [s2] "f"(da2), [s3] "f"(da3),
            [b]  "vr"(vb)
      );
      k -= sizeof(uint16_t);
    } while (k != 0);
    w_ptr = (const xnn_bfloat16*)wb;

    vl = __riscv_vsetvl_e32m4(vl);

    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc1 = __riscv_vfmax_vf_f32m4(vacc1, vmin, vl);
    vacc2 = __riscv_vfmax_vf_f32m4(vacc2, vmin, vl);
    vacc3 = __riscv_vfmax_vf_f32m4(vacc3, vmin, vl);
    vacc0 = __riscv_vfmin_vf_f32m4(vacc0, vmax, vl);
    vacc1 = __riscv_vfmin_vf_f32m4(vacc1, vmax, vl);
    vacc2 = __riscv_vfmin_vf_f32m4(vacc2, vmax, vl);
    vacc3 = __riscv_vfmin_vf_f32m4(vacc3, vmax, vl);

    __riscv_vse32_v_f32m4(c0, vacc0, vl); c0 = (float*)((uintptr_t)c0 + cn_stride);
    __riscv_vse32_v_f32m4(c1, vacc1, vl); c1 = (float*)((uintptr_t)c1 + cn_stride);
    __riscv_vse32_v_f32m4(c2, vacc2, vl); c2 = (float*)((uintptr_t)c2 + cn_stride);
    __riscv_vse32_v_f32m4(c3, vacc3, vl); c3 = (float*)((uintptr_t)c3 + cn_stride);

    a0 = (const uint16_t*)((uintptr_t)a0 - kc);
    a1 = (const uint16_t*)((uintptr_t)a1 - kc);
    a2 = (const uint16_t*)((uintptr_t)a2 - kc);
    a3 = (const uint16_t*)((uintptr_t)a3 - kc);
  } while (nc != 0);
}
