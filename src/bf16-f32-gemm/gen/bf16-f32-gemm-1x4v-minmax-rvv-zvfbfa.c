// Copyright 2026 UC Berkeley BAR
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// MR=1, NR=vsetvlmax_e32m4() bf16->fp32 GEMM with minmax epilogue.
// Zvfbfa path: BF16 × BF16 -> FP32 via vfwmacc.vf (widening FMA)
// under altfmt=1 vtype. No separate widen of B is needed — the FMA
// reads narrow BF16 sources and writes a wide FP32 accumulator.
//
// The assembler in the target toolchain does not recognise the
// `e16alt` SEW token or the `zvfbfa` extension name, so the vsetvli
// that enables altfmt is emitted via `.insn i` with a spec-derived
// immediate:
//
//   vtypei bits (per Zvfbfa v0.1, bit 8 = altfmt; rest per RVV 1.0):
//     [8]=altfmt=1  [7]=vma=1  [6]=vta=1  [5:3]=vsew=001 (SEW=16)
//     [2:0]=vlmul=001 (m2)  => 0x1C9
//
// `vfwmacc.vf` and `vfncvt.f.f.w` use standard RVV 1.0 encodings; the
// Zvfbfa spec is silent on encoding changes (altfmt is routed
// entirely through vtype), so these mnemonics assemble as-is under
// the base-V extension.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_bf16_f32_gemm_minmax_ukernel_1x4v__rvv_zvfbfa(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const uint16_t* a0 = a;
  float* c0 = c;
  (void) a_stride;
  (void) cm_stride;

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;

  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
    }
    nc -= vl;

    // FP32 bias broadcast.
    vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4((const float*) w, vl);
    w = (const float*) w + nr;

    const uint16_t* wb = (const uint16_t*) w;
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
    w = (const void*) wb;

    // FP32 epilogue — next intrinsic re-issues vsetvli to e32m4.
    const float vmin = params->scalar.min;
    const float vmax = params->scalar.max;
    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc0 = __riscv_vfmin_vf_f32m4(vacc0, vmax, vl);

    __riscv_vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);

    a0 = (const uint16_t*) ((uintptr_t) a0 - kc);
  } while (nc != 0);
}
