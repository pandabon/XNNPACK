// Copyright 2025 UC Berkeley BAR
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// MR=1, NR=vsetvlmax_e32m4() bf16->f32 GEMM with minmax epilogue.
// Zvfbfmin-only fallback: widen bf16 weights to fp32, then fp32 vfmacc.
//
// Requirements: Zvfbfmin (vfwcvtbf16.f.f.v for widening bf16->fp32)
// Packed weight layout: [fp32 bias x NR][bf16 weights x NR per KC step]

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/microparams.h"

void xnn_bf16_f32_gemm_minmax_ukernel_1x4v__rvv_zvfbfmin(
    size_t mr, size_t nc, size_t kc,
    const uint16_t* restrict a, size_t a_stride,
    const void* restrict w,
    float* restrict c, size_t cm_stride, size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0); assert(mr <= 1);
  assert(nc != 0); assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  const uint16_t* a0 = a;
  float*          c0 = c;

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;
  const xnn_bfloat16* w_ptr = (const xnn_bfloat16*)w;

  do {
    if XNN_UNLIKELY(nc < nr) { vl = __riscv_vsetvl_e32m4(nc); }
    nc -= vl;

    // Load fp32 bias.
    vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4((const float*)w_ptr, vl);
    w_ptr += nr * 2;  // skip NR fp32 bias values (in bfloat16* units)

    const uint16_t* wb  = (const uint16_t*)w_ptr;
    const uint16_t* a0k = a0;
    size_t k = kc;

    do {
      // A: bf16 scalar -> fp32 via bit-shift.
      uint32_t va0_bits = (uint32_t)(*a0k++) << 16;
      float va0;
      memcpy(&va0, &va0_bits, sizeof(float));

      // B: load NR bf16 weights, widen to fp32 (Zvfbfmin).
      vuint16m2_t vb_u16 = __riscv_vle16_v_u16m2(wb, vl);
      vfloat32m4_t vb;
      asm volatile(
          ".option push\n\t"
          ".option arch, +zvfbfmin\n\t"
          "vfwcvtbf16.f.f.v %[dst], %[src]\n\t"
          ".option pop\n\t"
          : [dst] "=&vr"(vb)
          : [src] "vr"(vb_u16)
      );
      wb += nr;

      vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va0, vb, vl);
      k -= sizeof(uint16_t);
    } while (k != 0);
    w_ptr = (const xnn_bfloat16*)wb;

    // Minmax clamp.
    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc0 = __riscv_vfmin_vf_f32m4(vacc0, vmax, vl);

    // Store fp32 output.
    __riscv_vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*)((uintptr_t)c0 + cn_stride);

    a0 = (const uint16_t*)((uintptr_t)a0 - kc);
  } while (nc != 0);
}
