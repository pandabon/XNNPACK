// Copyright 2026 UC Berkeley BAR
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// MR=1, NR=vsetvlmax_e32m4() bf16->fp32 IGEMM with minmax epilogue.
// Combined Zvfbfmin+Zvfbfa path. Inner-loop FMA `vfwmacc.vf` under
// altfmt vtype (Zvfbfa: bf16 x bf16 -> fp32 in one instruction). No
// narrow on store (output is fp32). Indirection scaffolding lifted
// from f32-igemm-1x4v-minmax-rvv: outer ks loop over indirect
// activation pointers, per-element a_offset apply with zero-pointer
// bypass, post-loop pointer rewind.
//
// This combined-variant kernel is byte-equivalent to a hypothetical
// Zvfbfa-only `bf16-f32-igemm-*-rvv-zvfbfa.c` because the fp32 output
// path has no narrow-on-store edge for Zvfbfmin's `vfncvtbf16` to
// optimize (kernels-report.md §Scope: "vfwmaccbf16.vf variant would
// be identical code"). The combined file is emitted under the
// `zvfbfmin_zvfbfa` filename for ladder symmetry with bf16-igemm.
//
// Toolchain note: the target binutils does not recognise `e16alt`,
// the `zvfbfa` extension name, or `vfwmaccbf16.vf`. The altfmt vsetvli
// is hand-emitted as `.insn i 0x57, 0x7, rd, rs1, 0x1C9` (Zvfbfa
// v0.1: bit 8=altfmt, bits[5:3]=vsew=001, bits[2:0]=vlmul=001), and
// the K-loop FMA uses `vfwmacc.vf` (standard RVV 1.0 mnemonic) which
// under altfmt=1 vtype performs the same bf16 widening FMA as
// Zvfbfwma's `vfwmaccbf16.vf` would.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/igemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_bf16_f32_igemm_minmax_ukernel_1x4v__rvv_zvfbfmin_zvfbfa(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint16_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint16_t* zero,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  (void) cm_stride;
  (void) cn_stride;

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

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
    size_t p = ks;
    do {
      const uint16_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint16_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const double da0 = xnn_nanbox_bf16(*a0++);
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
      p -= 1 * sizeof(void*);
    } while (p != 0);
    w = (const void*) wb;

    // FP32 epilogue — next intrinsic re-issues vsetvli to e32m4.
    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc0 = __riscv_vfmin_vf_f32m4(vacc0, vmax, vl);

    __riscv_vse32_v_f32m4(c0, vacc0, vl);
    c0 += vl;

    a = (const uint16_t**) ((uintptr_t) a - ks);
  } while (nc != 0);
}
