// Copyright 2026 UC Berkeley BAR
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// MR=4, NR=vsetvlmax_e32m4() bf16->fp32 GEMM with minmax epilogue.
// Zvfbfa path: BF16 × BF16 -> FP32 via vfwmacc.vf (widening FMA)
// under altfmt=1 vtype. No separate widen of B is needed.
//
// Register budget (VLEN=256, NR=32, LMUL=m4):
//   4 * vfloat32m4_t (acc)    = 16 vregs
//   1 * vuint16m2_t  (vb)     =  2 vregs
//   Total: 18 / 32  (4 vregs of headroom vs Zvfbfmin's 22/32)
//
// vsetvli-with-altfmt is emitted as `.insn i 0x57, 0x7, rd, rs1, 0x1C9`
// because the target toolchain does not recognise `e16alt`. The
// immediate 0x1C9 is derived from the Zvfbfa v0.1 spec: bit 8 = altfmt,
// bit 7 = vma, bit 6 = vta, bits[5:3] = vsew=001 (SEW=16), bits[2:0] =
// vlmul=001 (m2). `vfwmacc.vf` itself uses the standard RVV 1.0
// encoding.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_bf16_f32_gemm_minmax_ukernel_4x4v__rvv_zvfbfa(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint16_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const uint16_t* a0 = a;
  float* c0 = c;
  const uint16_t* a1 = (const uint16_t*) ((uintptr_t) a0 + a_stride);
  float*          c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) { a1 = a0; c1 = c0; }
  const uint16_t* a2 = (const uint16_t*) ((uintptr_t) a1 + a_stride);
  float*          c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) { a2 = a1; c2 = c1; }
  const uint16_t* a3 = (const uint16_t*) ((uintptr_t) a2 + a_stride);
  float*          c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) { a3 = a2; c3 = c2; }

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;

  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
    }
    nc -= vl;

    // FP32 bias broadcast to all MR rows.
    vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4((const float*) w, vl);
    vfloat32m4_t vacc1 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc2 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc3 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    w = (const float*) w + nr;

    const uint16_t* wb = (const uint16_t*) w;
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
    w = (const void*) wb;

    // FP32 epilogue — next intrinsic re-issues vsetvli to e32m4.
    const float vmin = params->scalar.min;
    const float vmax = params->scalar.max;
    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc1 = __riscv_vfmax_vf_f32m4(vacc1, vmin, vl);
    vacc2 = __riscv_vfmax_vf_f32m4(vacc2, vmin, vl);
    vacc3 = __riscv_vfmax_vf_f32m4(vacc3, vmin, vl);
    vacc0 = __riscv_vfmin_vf_f32m4(vacc0, vmax, vl);
    vacc1 = __riscv_vfmin_vf_f32m4(vacc1, vmax, vl);
    vacc2 = __riscv_vfmin_vf_f32m4(vacc2, vmax, vl);
    vacc3 = __riscv_vfmin_vf_f32m4(vacc3, vmax, vl);

    __riscv_vse32_v_f32m4(c0, vacc0, vl); c0 = (float*) ((uintptr_t) c0 + cn_stride);
    __riscv_vse32_v_f32m4(c1, vacc1, vl); c1 = (float*) ((uintptr_t) c1 + cn_stride);
    __riscv_vse32_v_f32m4(c2, vacc2, vl); c2 = (float*) ((uintptr_t) c2 + cn_stride);
    __riscv_vse32_v_f32m4(c3, vacc3, vl); c3 = (float*) ((uintptr_t) c3 + cn_stride);

    a0 = (const uint16_t*) ((uintptr_t) a0 - kc);
    a1 = (const uint16_t*) ((uintptr_t) a1 - kc);
    a2 = (const uint16_t*) ((uintptr_t) a2 - kc);
    a3 = (const uint16_t*) ((uintptr_t) a3 - kc);
  } while (nc != 0);
}
