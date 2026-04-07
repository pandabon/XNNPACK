// Copyright 2025 UC Berkeley BAR
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// MR=1, NR=vsetvlmax_e32m4() bf16->f32 GEMM with minmax epilogue.
// Zvfbfa path: uses vfwmacc.vf with altfmt=1 SEW=16 (native bf16 widening FMA).
//
// Under Zvfbfa, setting altfmt=1 in vtype redefines e16 as bf16 format.
// vfwmacc.vf then performs bf16 scalar x bf16 vector -> fp32 accumulate
// using the standard V-extension opcode (no dedicated bf16 instruction needed).
//
// Packed weight layout: [fp32 bias x NR][bf16 weights x NR per KC step]

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/microparams.h"

// NaN-box a bf16 value into a 64-bit FP register (RISC-V D extension).
// Under Zvfbfa altfmt=1, vfwmacc.vf reads the lower 16 bits of rs1 as bf16.
static inline double nanbox_bf16(uint16_t v) {
  uint64_t bits = 0xFFFFFFFFFFFF0000ULL | (uint64_t)v;
  double d;
  __builtin_memcpy(&d, &bits, sizeof(d));
  return d;
}

void xnn_bf16_f32_gemm_minmax_ukernel_1x4v__rvv_zvfbfa(
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
  float* c0 = c;

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;
  const xnn_bfloat16* w_ptr = (const xnn_bfloat16*)w;

  do {
    if XNN_UNLIKELY(nc < nr) { vl = __riscv_vsetvl_e32m4(nc); }
    nc -= vl;

    // Load fp32 bias (vtype=e32m4).
    vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4((const float*)w_ptr, vl);
    w_ptr += nr * 2;

    const uint16_t* wb  = (const uint16_t*)w_ptr;
    const uint16_t* a0k = a0;
    size_t k = kc;

    do {
      uint16_t raw_a0 = *a0k++;

      // Load NR bf16 weights as u16m2.
      vuint16m2_t vb = __riscv_vle16_v_u16m2(wb, vl);
      wb += nr;

      double da0 = nanbox_bf16(raw_a0);

      // vfwmacc.vf with altfmt=1: bf16 scalar x bf16 vector -> fp32 accumulate.
      // We set altfmt=1 via .option arch directive; the assembler handles
      // the vtype encoding when Zvfbfa is enabled in -march.
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

    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc0 = __riscv_vfmin_vf_f32m4(vacc0, vmax, vl);
    __riscv_vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*)((uintptr_t)c0 + cn_stride);
    a0 = (const uint16_t*)((uintptr_t)a0 - kc);
  } while (nc != 0);
}
