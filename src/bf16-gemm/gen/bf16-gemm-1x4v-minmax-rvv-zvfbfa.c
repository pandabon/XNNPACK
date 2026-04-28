// Copyright 2026 UC Berkeley BAR
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// MR=1, NR=vsetvlmax_e32m4() bf16->bf16 GEMM with minmax epilogue.
// Zvfbfa path: BF16 × BF16 -> FP32 via vfwmacc.vf under altfmt=1,
// FP32 minmax clamp (base-V), then FP32 -> BF16 narrow via
// vfncvt.f.f.w under altfmt=1.
//
// Toolchain note: the target assembler doesn't recognise e16alt, so
// the altfmt-enabling vsetvli is emitted as `.insn i 0x57, 0x7, rd,
// rs1, 0x1C9`. Immediate 0x1C9 is derived from Zvfbfa v0.1 spec
// (bit 8 = altfmt, base RVV 1.0 for the rest). Both vfwmacc.vf and
// vfncvt.f.f.w use standard RVV 1.0 encodings.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_bf16_gemm_minmax_ukernel_1x4v__rvv_zvfbfa(
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
      const double da0 = xnn_nanbox_bf16(*a0k++);
      vuint16m2_t vb = __riscv_vle16_v_u16m2(wb, vl);
      wb += nr;

      size_t vl_tmp;
      __asm__ volatile(
          ".insn i 0x57, 0x7, %[vlout], %[avl], 0x1C9\n\t"  // vsetvli e16,m2,ta,ma,altfmt
          "vfwmacc.vf %[a0], %[s0], %[v]\n\t"
          : [vlout] "=&r"(vl_tmp),
            [a0] "+vr"(vacc0)
          : [avl] "r"(vl),
            [s0] "f"(da0),
            [v] "vr"(vb));
      k -= sizeof(uint16_t);
    } while (k != 0);
    w_ptr = (const xnn_bfloat16*) wb;

    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc0 = __riscv_vfmin_vf_f32m4(vacc0, vmax, vl);

    // Narrow fp32m4 -> u16m2 under altfmt via vfncvt.f.f.w.
    size_t vl_tmp;
    vuint16m2_t vout0;
    __asm__ volatile(
        ".insn i 0x57, 0x7, %[vlout], %[avl], 0x1C9\n\t"
        "vfncvt.f.f.w %[d0], %[s0]\n\t"
        : [vlout] "=&r"(vl_tmp),
          [d0] "=&vr"(vout0)
        : [avl] "r"(vl),
          [s0] "vr"(vacc0));

    __riscv_vse16_v_u16m2(c0, vout0, vl);
    c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
  } while (nc != 0);
}
