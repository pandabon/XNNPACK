// Copyright 2026 UC Berkeley BAR
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// MR=4, NR=vsetvlmax_e32m4() bf16->bf16 GEMM with minmax epilogue.
// Zvfbfa path: BF16 × BF16 -> FP32 via vfwmacc.vf under altfmt=1,
// FP32 minmax clamp (base-V), then FP32 -> BF16 narrow via
// vfncvt.f.f.w under altfmt=1.
//
// Register budget (VLEN=256, NR=32, LMUL=m4):
//   4 * vfloat32m4_t (acc) = 16 vregs
//   1 * vuint16m2_t  (vb)  =  2 vregs
//   4 * vuint16m2_t  (vout)=  8 vregs (narrow epilogue; accs are dead)
//   Peak: 18 / 32 during KC loop; 24 / 32 during narrow epilogue.
//
// Toolchain note: .insn i 0x57, 0x7, rd, rs1, 0x1C9 sets altfmt=1,
// vma=1, vta=1, SEW=16, LMUL=m2 per Zvfbfa v0.1 spec bit layout.
// vfwmacc.vf and vfncvt.f.f.w use standard RVV 1.0 encodings.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_bf16_gemm_minmax_ukernel_4x4v__rvv_zvfbfa(
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
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w_ptr != NULL);
  assert(c != NULL);

  const uint16_t* a0 = (const uint16_t*) a;
  uint16_t*       c0 = (uint16_t*) c;
  const uint16_t* a1 = (const uint16_t*) ((uintptr_t) a0 + a_stride);
  uint16_t*       c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) { a1 = a0; c1 = c0; }
  const uint16_t* a2 = (const uint16_t*) ((uintptr_t) a1 + a_stride);
  uint16_t*       c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) { a2 = a1; c2 = c1; }
  const uint16_t* a3 = (const uint16_t*) ((uintptr_t) a2 + a_stride);
  uint16_t*       c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) { a3 = a2; c3 = c2; }

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
    vfloat32m4_t vacc1 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc2 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc3 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    const size_t bias_u16_stride = nr * (sizeof(float) / sizeof(uint16_t));
    w_ptr += bias_u16_stride;

    const uint16_t* wb = (const uint16_t*) w_ptr;
    const uint16_t* a0k = a0;
    const uint16_t* a1k = a1;
    const uint16_t* a2k = a2;
    const uint16_t* a3k = a3;
    size_t k = kc;
    do {
      const double da0 = xnn_nanbox_bf16(*a0k++);
      const double da1 = xnn_nanbox_bf16(*a1k++);
      const double da2 = xnn_nanbox_bf16(*a2k++);
      const double da3 = xnn_nanbox_bf16(*a3k++);
      vuint16m2_t vb = __riscv_vle16_v_u16m2(wb, vl);
      wb += nr;

      size_t vl_tmp;
      __asm__ volatile(
          ".insn i 0x57, 0x7, %[vlout], %[avl], 0x1C9\n\t"  // vsetvli e16,m2,ta,ma,altfmt
          "vfwmacc.vf %[a0], %[s0], %[v]\n\t"
          "vfwmacc.vf %[a1], %[s1], %[v]\n\t"
          "vfwmacc.vf %[a2], %[s2], %[v]\n\t"
          "vfwmacc.vf %[a3], %[s3], %[v]\n\t"
          : [vlout] "=&r"(vl_tmp),
            [a0] "+vr"(vacc0), [a1] "+vr"(vacc1),
            [a2] "+vr"(vacc2), [a3] "+vr"(vacc3)
          : [avl] "r"(vl),
            [s0] "f"(da0), [s1] "f"(da1),
            [s2] "f"(da2), [s3] "f"(da3),
            [v] "vr"(vb));
      k -= sizeof(uint16_t);
    } while (k != 0);
    w_ptr = (const xnn_bfloat16*) wb;

    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc1 = __riscv_vfmax_vf_f32m4(vacc1, vmin, vl);
    vacc2 = __riscv_vfmax_vf_f32m4(vacc2, vmin, vl);
    vacc3 = __riscv_vfmax_vf_f32m4(vacc3, vmin, vl);
    vacc0 = __riscv_vfmin_vf_f32m4(vacc0, vmax, vl);
    vacc1 = __riscv_vfmin_vf_f32m4(vacc1, vmax, vl);
    vacc2 = __riscv_vfmin_vf_f32m4(vacc2, vmax, vl);
    vacc3 = __riscv_vfmin_vf_f32m4(vacc3, vmax, vl);

    // Narrow fp32m4 -> u16m2 under altfmt via vfncvt.f.f.w.
    size_t vl_tmp;
    vuint16m2_t vout0, vout1, vout2, vout3;
    __asm__ volatile(
        ".insn i 0x57, 0x7, %[vlout], %[avl], 0x1C9\n\t"
        "vfncvt.f.f.w %[d0], %[s0]\n\t"
        "vfncvt.f.f.w %[d1], %[s1]\n\t"
        "vfncvt.f.f.w %[d2], %[s2]\n\t"
        "vfncvt.f.f.w %[d3], %[s3]\n\t"
        : [vlout] "=&r"(vl_tmp),
          [d0] "=&vr"(vout0), [d1] "=&vr"(vout1),
          [d2] "=&vr"(vout2), [d3] "=&vr"(vout3)
        : [avl] "r"(vl),
          [s0] "vr"(vacc0), [s1] "vr"(vacc1),
          [s2] "vr"(vacc2), [s3] "vr"(vacc3));

    __riscv_vse16_v_u16m2(c0, vout0, vl); c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
    __riscv_vse16_v_u16m2(c1, vout1, vl); c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
    __riscv_vse16_v_u16m2(c2, vout2, vl); c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
    __riscv_vse16_v_u16m2(c3, vout3, vl); c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
  } while (nc != 0);
}
