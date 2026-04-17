// Copyright 2025 UC Berkeley BAR
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// MR=4, NR=vsetvlmax_e32m4() bf16->f32 GEMM with minmax epilogue.
// Hot-path tile for Saturn GENV256D128 (VLEN=256, Zvfbfmin).
//
// Zvfbfmin path: bf16 weights widened to fp32 via vfwcvtbf16.f.f.v;
// bf16 activations widened to fp32 via bit-shift + memcpy;
// FMA via base-V vfmacc.vf on fp32 accumulators.
// Packed weight layout: [fp32 bias x NR][bf16 weights x NR per KC step]

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/microparams.h"

void xnn_bf16_f32_gemm_minmax_ukernel_4x4v__rvv_zvfbfmin(
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

  // nr = NR = vsetvlmax_e32m4() sets vtype=e32m4. For VLEN=256: nr=32.
  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;
  const xnn_bfloat16* w_ptr = (const xnn_bfloat16*)w;

  do {
    // Tail: set vl to remaining columns. vtype stays e32m4.
    if XNN_UNLIKELY(nc < nr) { vl = __riscv_vsetvl_e32m4(nc); }
    nc -= vl;

    // Load fp32 bias for this NR-tile (vtype=e32m4).
    vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4((const float*)w_ptr, vl);
    vfloat32m4_t vacc1 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc2 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc3 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    w_ptr += nr * 2;  // skip NR fp32 bias values (in bfloat16* units: 2 per float)

    const uint16_t* wb   = (const uint16_t*)w_ptr;
    const uint16_t* a0k  = a0;
    const uint16_t* a1k  = a1;
    const uint16_t* a2k  = a2;
    const uint16_t* a3k  = a3;
    size_t k = kc;

    do {
      // A: bf16 scalars -> fp32 via bit-shift (one per row).
      uint32_t va0_bits = (uint32_t)(*a0k++) << 16;
      float va0; memcpy(&va0, &va0_bits, sizeof(float));
      uint32_t va1_bits = (uint32_t)(*a1k++) << 16;
      float va1; memcpy(&va1, &va1_bits, sizeof(float));
      uint32_t va2_bits = (uint32_t)(*a2k++) << 16;
      float va2; memcpy(&va2, &va2_bits, sizeof(float));
      uint32_t va3_bits = (uint32_t)(*a3k++) << 16;
      float va3; memcpy(&va3, &va3_bits, sizeof(float));

      // B: load NR bf16 weights, widen to fp32 (Zvfbfmin via inline asm).
      vuint16m2_t vb_u16 = __riscv_vle16_v_u16m2(wb, vl);
      vfloat32m4_t vb;
      asm volatile(
          ".option push\n\t"
          ".option arch, +zvfbfmin\n\t"
          "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
          "vfwcvtbf16.f.f.v %[d], %[s]\n\t"
          ".option pop\n\t"
          : [d] "=&vr"(vb)
          : [s] "vr"(vb_u16), [vl] "r"(vl)
      );
      wb += nr;

      vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va0, vb, vl);
      vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, va1, vb, vl);
      vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, va2, vb, vl);
      vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, va3, vb, vl);
      k -= sizeof(uint16_t);
    } while (k != 0);
    w_ptr = (const xnn_bfloat16*)wb;

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
