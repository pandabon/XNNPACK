// Copyright 2025 UC Berkeley BAR
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// MR=1, NR=vsetvlmax_e32m4() bf16 GEMM with minmax epilogue.
//
// Packed weight layout (produced by x16-x32-packw-x4v-rvv):
//   [fp32 bias × NR]          -- NR * sizeof(float) bytes
//   [bf16 weights × NR]       -- NR * sizeof(uint16_t) bytes, per KC step
//
// Requirements: Zvfbfmin (vfwcvtbf16.f.f.v, vfncvtbf16.f.f.w)

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "src/xnnpack/gemm.h"
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

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  const uint16_t* a0 = (const uint16_t*) a;
  uint16_t*       c0 = (uint16_t*) c;

  // nr: maximum vector length for fp32 m4 (= vlenb for VLEN=256).
  // Same vl applies to e16m2 (half SEW, half LMUL => same element count).
  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;

  do {
    // Set vl for this NC tile (tail may be < nr).
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
    }
    nc -= vl;

    // ------------------------------------------------------------------
    // Load fp32 bias from packed buffer head. The packw writes NR fp32
    // values (= NR * 2 uint16 words) at the start of each NR-column tile.
    // ------------------------------------------------------------------
    const float* bias_ptr = (const float*) w_ptr;
    vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(bias_ptr, vl);
    w_ptr += nr * 2;  // skip NR fp32 values (each fp32 = 2 uint16 words)

    // ------------------------------------------------------------------
    // KC reduction loop: one bf16 A scalar × NR bf16 B column per step.
    // ------------------------------------------------------------------
    const uint16_t* w = (const uint16_t*) w_ptr;
    const uint16_t* a0k = a0;
    size_t k = kc;
    do {
      // A: bf16 scalar -> fp32 via bit-shift (no Zvfbfmin required for scalar).
      uint32_t va0_bits = (uint32_t)(*a0k++) << 16;
      float va0;
      memcpy(&va0, &va0_bits, sizeof(float));

      // B: load NR bf16 weights, widen to fp32 (Zvfbfmin via inline asm).
      vuint16m2_t vb_u16 = __riscv_vle16_v_u16m2(w, vl);
      vfloat32m4_t vb;
      asm volatile(
          ".option push\n\t"
          ".option arch, +zvfbfmin\n\t"
          "vfwcvtbf16.f.f.v %[dst], %[src]\n\t"
          ".option pop\n\t"
          : [dst] "=&vr"(vb)
          : [src] "vr"(vb_u16)
      );
      w += nr;  // advance by NR bf16 elements

      vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va0, vb, vl);
      k -= sizeof(uint16_t);
    } while (k != 0);

    // Advance w_ptr past all KC steps for this NC tile.
    w_ptr = (const xnn_bfloat16*) w;

    // ------------------------------------------------------------------
    // Minmax clamp.
    // ------------------------------------------------------------------
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
    c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

    // Rewind A pointer for the next NC tile.
    a0 = (const uint16_t*) ((uintptr_t) a0 - kc);
  } while (nc != 0);
}
