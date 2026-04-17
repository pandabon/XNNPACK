// Copyright 2025 UC Berkeley BAR
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// MR=4, NR=vsetvlmax_e32m4() bf16 GEMM with minmax epilogue.
// Hot path tile for 3-linear MNIST MLP on Saturn GENV256D128 (VLEN=256).
//
// Register budget (VLEN=256, NR=32):
//   vacc0..vacc3  : 4 x vfloat32m4_t = 4 x 4 regs = 16 regs
//   vb_u16        : vuint16m2_t       = 2 regs
//   vb            : vfloat32m4_t      = 4 regs   (distinct from vb_u16)
//   Total: 22 / 32 vector registers
//
// Packed weight layout (produced by x16-x32-packw-x4v-rvv):
//   [fp32 bias × NR]
//   [bf16 weights × NR] per KC step
//
// Zvfbfmin instructions emitted via inline asm (.option arch, +zvfbfmin)
// to bypass missing compiler intrinsic support in GCC 14.x.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/microparams.h"

void xnn_bf16_gemm_minmax_ukernel_4x4v__rvv_zvfbfmin(
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

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

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

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;

  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
    }
    nc -= vl;

    // ------------------------------------------------------------------
    // Load fp32 bias; broadcast to all MR rows.
    // ------------------------------------------------------------------
    const float* bias_ptr = (const float*) w_ptr;
    vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(bias_ptr, vl);
    vfloat32m4_t vacc1 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc2 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc3 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    w_ptr += nr * 2;  // skip NR fp32 bias values

    // ------------------------------------------------------------------
    // KC reduction loop.
    // ------------------------------------------------------------------
    const uint16_t* w = (const uint16_t*) w_ptr;
    const uint16_t* a0k = a0;
    const uint16_t* a1k = a1;
    const uint16_t* a2k = a2;
    const uint16_t* a3k = a3;
    size_t k = kc;
    do {
      // A: bf16 scalars -> fp32 via bit-shift (one per row).
      uint32_t va0_bits = (uint32_t)(*a0k++) << 16; float va0; memcpy(&va0, &va0_bits, sizeof(float));
      uint32_t va1_bits = (uint32_t)(*a1k++) << 16; float va1; memcpy(&va1, &va1_bits, sizeof(float));
      uint32_t va2_bits = (uint32_t)(*a2k++) << 16; float va2; memcpy(&va2, &va2_bits, sizeof(float));
      uint32_t va3_bits = (uint32_t)(*a3k++) << 16; float va3; memcpy(&va3, &va3_bits, sizeof(float));

      // B: load NR bf16 weights, widen to fp32 (Zvfbfmin via inline asm).
      vuint16m2_t vb_u16 = __riscv_vle16_v_u16m2(w, vl);
      vfloat32m4_t vb;
      asm volatile(
          ".option push\n\t"
          ".option arch, +zvfbfmin\n\t"
          "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
          "vfwcvtbf16.f.f.v %[dst], %[src]\n\t"
          ".option pop\n\t"
          : [dst] "=&vr"(vb)
          : [src] "vr"(vb_u16), [vl] "r"(vl)
      );
      w += nr;

      vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va0, vb, vl);
      vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, va1, vb, vl);
      vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, va2, vb, vl);
      vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, va3, vb, vl);
      k -= sizeof(uint16_t);
    } while (k != 0);

    w_ptr = (const xnn_bfloat16*) w;

    // ------------------------------------------------------------------
    // Minmax clamp.
    // ------------------------------------------------------------------
    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc1 = __riscv_vfmax_vf_f32m4(vacc1, vmin, vl);
    vacc2 = __riscv_vfmax_vf_f32m4(vacc2, vmin, vl);
    vacc3 = __riscv_vfmax_vf_f32m4(vacc3, vmin, vl);

    vacc0 = __riscv_vfmin_vf_f32m4(vacc0, vmax, vl);
    vacc1 = __riscv_vfmin_vf_f32m4(vacc1, vmax, vl);
    vacc2 = __riscv_vfmin_vf_f32m4(vacc2, vmax, vl);
    vacc3 = __riscv_vfmin_vf_f32m4(vacc3, vmax, vl);

    // Narrow fp32 -> bf16 and store (Zvfbfmin via inline asm).
    vuint16m2_t vout0, vout1, vout2, vout3;
    asm volatile(
        ".option push\n\t"
        ".option arch, +zvfbfmin\n\t"
        "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
        "vfncvtbf16.f.f.w %[d0], %[s0]\n\t"
        "vfncvtbf16.f.f.w %[d1], %[s1]\n\t"
        "vfncvtbf16.f.f.w %[d2], %[s2]\n\t"
        "vfncvtbf16.f.f.w %[d3], %[s3]\n\t"
        ".option pop\n\t"
        : [d0] "=&vr"(vout0), [d1] "=&vr"(vout1),
          [d2] "=&vr"(vout2), [d3] "=&vr"(vout3)
        : [s0] "vr"(vacc0), [s1] "vr"(vacc1),
          [s2] "vr"(vacc2), [s3] "vr"(vacc3),
          [vl] "r"(vl)
    );

    __riscv_vse16_v_u16m2(c0, vout0, vl); c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
    __riscv_vse16_v_u16m2(c1, vout1, vl); c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
    __riscv_vse16_v_u16m2(c2, vout2, vl); c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
    __riscv_vse16_v_u16m2(c3, vout3, vl); c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);

    // Rewind A pointers for the next NC tile.
    a0 = (const uint16_t*) ((uintptr_t) a0 - kc);
    a1 = (const uint16_t*) ((uintptr_t) a1 - kc);
    a2 = (const uint16_t*) ((uintptr_t) a2 - kc);
    a3 = (const uint16_t*) ((uintptr_t) a3 - kc);
  } while (nc != 0);
}
