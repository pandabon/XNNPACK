// Copyright 2026 UC Berkeley BAR
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// MR=1, NR=vsetvlmax_e32m4() bf16->bf16 GEMM with minmax epilogue.
// Zvfbfmin path: BF16 weights are widened to FP32 with vfwcvtbf16.f.f.v,
// FMA accumulates in FP32, and the output is narrowed back to BF16 with
// vfncvtbf16.f.f.w. Every Zvfbfmin asm block sets its own vtype
// (e16, m2) so conversions are correct regardless of the surrounding
// compiler-managed vtype state.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_bf16_gemm_minmax_ukernel_1x4v__rvv_zvfbfmin(
    size_t mr,
    size_t nc,
    size_t kc,
    const xnn_bfloat16* restrict a,
    size_t a_stride,
    const xnn_bfloat16* restrict w_ptr,
    xnn_bfloat16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_bf16_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w_ptr != NULL);
  assert(c != NULL);

  const uint16_t* a0 = (const uint16_t*) a;
  uint16_t*       c0 = (uint16_t*) c;
  (void) a_stride;
  (void) cm_stride;

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;

  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
    }
    nc -= vl;

    vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4((const float*) w_ptr, vl);
    const size_t bias_u16_stride = nr * (sizeof(float) / sizeof(uint16_t));
    w_ptr += bias_u16_stride;

    const uint16_t* wb = (const uint16_t*) w_ptr;
    const uint16_t* a0k = a0;
    size_t k = kc;
    do {
      uint32_t va0b = (uint32_t) (*a0k++) << 16;
      float va0;
      memcpy(&va0, &va0b, sizeof(float));

      vuint16m2_t vb_u16 = __riscv_vle16_v_u16m2(wb, vl);
      vfloat32m4_t vb;
      __asm__ volatile(
          ".option push\n\t"
          ".option arch, +zvfbfmin\n\t"
          "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
          "vfwcvtbf16.f.f.v %[d], %[s]\n\t"
          ".option pop\n\t"
          : [d] "=&vr"(vb)
          : [s] "vr"(vb_u16), [vl] "r"(vl));
      wb += nr;

      vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va0, vb, vl);
      k -= sizeof(uint16_t);
    } while (k != 0);
    w_ptr = (const xnn_bfloat16*) wb;

    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc0 = __riscv_vfmin_vf_f32m4(vacc0, vmax, vl);

    vuint16m2_t vout0;
    __asm__ volatile(
        ".option push\n\t"
        ".option arch, +zvfbfmin\n\t"
        "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
        "vfncvtbf16.f.f.w %[d], %[s]\n\t"
        ".option pop\n\t"
        : [d] "=&vr"(vout0)
        : [s] "vr"(vacc0), [vl] "r"(vl));

    __riscv_vse16_v_u16m2(c0, vout0, vl);
    c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
  } while (nc != 0);
}
