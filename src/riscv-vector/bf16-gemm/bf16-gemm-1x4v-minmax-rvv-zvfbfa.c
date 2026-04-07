// Copyright 2025 UC Berkeley BAR
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// MR=1, NR=vsetvlmax_e32m4() bf16 GEMM with minmax epilogue.
// Zvfbfa path: uses vfwmacc.vf with altfmt=1 SEW=16 for native bf16 widening
// FMA, then narrows fp32 result back to bf16 via vfncvtbf16 (Zvfbfmin).
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

void xnn_bf16_gemm_minmax_ukernel_1x4v__rvv_zvfbfa(
    size_t mr, size_t nc, size_t kc,
    const xnn_bfloat16* restrict a,
    size_t a_stride,
    const xnn_bfloat16* restrict w_ptr,
    xnn_bfloat16* restrict c,
    size_t cm_stride, size_t cn_stride,
    const struct xnn_bf16_minmax_params* restrict params)
{
  assert(mr != 0); assert(mr <= 1);
  assert(nc != 0); assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  const uint16_t* a0 = (const uint16_t*)a;
  uint16_t*       c0 = (uint16_t*)c;

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;

  do {
    if XNN_UNLIKELY(nc < nr) { vl = __riscv_vsetvl_e32m4(nc); }
    nc -= vl;

    // Load fp32 bias.
    const float* bias_ptr = (const float*)w_ptr;
    vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(bias_ptr, vl);
    w_ptr += nr * 2;

    const uint16_t* wb  = (const uint16_t*)w_ptr;
    const uint16_t* a0k = a0;
    size_t k = kc;

    do {
      uint16_t raw_a0 = *a0k++;

      vuint16m2_t vb = __riscv_vle16_v_u16m2(wb, vl);
      wb += nr;

      double da0 = nanbox_bf16(raw_a0);

      // vfwmacc.vf with altfmt=1: bf16 scalar x bf16 vector -> fp32 accumulate.
      asm volatile(
          ".option push\n\t"
          ".option arch, +zvfbfa\n\t"
          "vfwmacc.vf %[acc], %[s0], %[b]\n\t"
          ".option pop\n\t"
          : [acc] "+vr"(vacc0)
          : [s0] "f"(da0), [b] "vr"(vb)
      );
      k -= sizeof(uint16_t);
    } while (k != 0);
    w_ptr = (const xnn_bfloat16*)wb;

    // Reset vtype to e32m4 for fp32 epilogue.
    vl = __riscv_vsetvl_e32m4(vl);

    // Minmax clamp in fp32.
    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc0 = __riscv_vfmin_vf_f32m4(vacc0, vmax, vl);

    // Narrow fp32 -> bf16 and store (Zvfbfmin via inline asm).
    vuint16m2_t vout0;
    asm volatile(
        ".option push\n\t"
        ".option arch, +zvfbfmin\n\t"
        "vfncvtbf16.f.f.w %[dst], %[src]\n\t"
        ".option pop\n\t"
        : [dst] "=&vr"(vout0)
        : [src] "vr"(vacc0)
    );
    __riscv_vse16_v_u16m2(c0, vout0, vl);
    c0 = (uint16_t*)((uintptr_t)c0 + cn_stride);

    a0 = (const uint16_t*)((uintptr_t)a0 - kc);
  } while (nc != 0);
}
