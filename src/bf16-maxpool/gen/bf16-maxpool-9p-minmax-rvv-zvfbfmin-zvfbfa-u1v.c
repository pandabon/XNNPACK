// Copyright 2026 UC Berkeley BAR
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// 9-input BF16 maxpool with FP32-min/max clamp.
//
// Inner-loop strategy (Plan §1C of "below-is-lenet-stats-crispy-falcon"):
//   1. Load each of the 9 BF16 input rows as opaque uint16 vectors
//      (vle16_v_u16m2).
//   2. Widen each to fp32m4 via vfwcvtbf16.f.f.v (Zvfbfmin native
//      mnemonic). The convert is emitted via __asm__ volatile under
//      .option push / .option arch, +zvfbfmin so the assembler accepts
//      the mnemonic regardless of the surrounding -march, and each block
//      opens with its own `vsetvli zero, vl, e16, m2, ta, ma` to be
//      independent of whatever vtype the surrounding compiler-emitted
//      intrinsics (e32m4) leave behind. Mirrors the convention from
//      src/bf16-f32-gemm/gen/bf16-f32-gemm-1x4v-minmax-rvv-zvfbfmin.c.
//   3. Tree-reduce 9 fp32m4 vectors with base-V vfmax.vv (no inline-asm
//      needed; the standard intrinsic suffices since the surrounding
//      vtype is e32m4 from intrinsic vsetvl).
//   4. Clamp with vfmin.vf / vfmax.vf against fp32 output_min / output_max
//      from xnn_bf16_minmax_params.
//   5. Narrow back to bf16m2 via vfncvtbf16.f.f.w under another
//      __asm__ block with its own e16,m2,ta,ma vsetvli, then store as
//      vse16_v_u16m2.
//
// ISA-extension capability matrix (see docs-ucb-bar/bf16-rvv-kernels-report.md
// §"Cross-cutting design notes"):
//   - This kernel literally only requires Zvfbfmin: vfwcvtbf16.f.f.v and
//     vfncvtbf16.f.f.w are both Zvfbfmin instructions, and the tree-reduce
//     / clamp / load / store are all base-V (RVV 1.0). It does NOT use
//     any Zvfbfa-named instruction (no vfwmacc.vf altfmt, no vfwmaccbf16.vf).
//   - The "-zvfbfmin-zvfbfa" filename suffix is for ladder symmetry with
//     the bf16-{f32-,}gemm/igemm families that genuinely require both
//     extensions. The X-macro registration in
//     src/bf16-maxpool/bf16-maxpool-minmax.inc gates on both arch flags so
//     the singleton-flag rungs of the priority ladder remain empty for
//     this iteration (Plan §"Scope simplification").

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/maxpool.h"
#include "src/xnnpack/microparams.h"


void xnn_bf16_maxpool_minmax_ukernel_9p__rvv_zvfbfmin_zvfbfa_u1v(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const xnn_bfloat16** input,
    size_t input_offset,
    size_t input_pixel_stride,
    xnn_bfloat16* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_bf16_minmax_params* restrict params)
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;

  do {
    const xnn_bfloat16** i = input;
    uint16_t* o = (uint16_t*) output;
    {
      const uint16_t* i0 = (const uint16_t*) *i++;
      const uint16_t* i1 = (const uint16_t*) (1 < kernel_elements ? *i++ : (const xnn_bfloat16*) i0);
      const uint16_t* i2 = (const uint16_t*) (2 < kernel_elements ? *i++ : (const xnn_bfloat16*) i0);
      const uint16_t* i3 = (const uint16_t*) (3 < kernel_elements ? *i++ : (const xnn_bfloat16*) i0);
      const uint16_t* i4 = (const uint16_t*) (4 < kernel_elements ? *i++ : (const xnn_bfloat16*) i0);
      const uint16_t* i5 = (const uint16_t*) (5 < kernel_elements ? *i++ : (const xnn_bfloat16*) i0);
      const uint16_t* i6 = (const uint16_t*) (6 < kernel_elements ? *i++ : (const xnn_bfloat16*) i0);
      const uint16_t* i7 = (const uint16_t*) (7 < kernel_elements ? *i++ : (const xnn_bfloat16*) i0);
      const uint16_t* i8 = (const uint16_t*) (8 < kernel_elements ? *i++ : (const xnn_bfloat16*) i0);
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);

      size_t c = channels;
      do {
        size_t vl = __riscv_vsetvl_e16m2(c);

        // Load + widen + accumulate-max progressively to keep liveness low.
        // At any time at most 1 u16m2 + 2 fp32m4 vectors are live, so the
        // register-allocator never sees the simultaneous-9-widen pressure
        // that a single asm block with 9 outputs would create.
        #define XNN_BF16_MAXPOOL_WIDEN(dst_f32m4, src_u16m2) \
            __asm__ volatile( \
                ".option push\n\t" \
                ".option arch, +zvfbfmin\n\t" \
                "vsetvli zero, %[vl], e16, m2, ta, ma\n\t" \
                "vfwcvtbf16.f.f.v %[d], %[s]\n\t" \
                ".option pop\n\t" \
                : [d] "=&vr"(dst_f32m4) \
                : [s] "vr"(src_u16m2), [vl] "r"(vl))

        vuint16m2_t v_u16 = __riscv_vle16_v_u16m2(i0, vl); i0 += vl;
        vfloat32m4_t out_f32; XNN_BF16_MAXPOOL_WIDEN(out_f32, v_u16);

        vfloat32m4_t t_f32;
        v_u16 = __riscv_vle16_v_u16m2(i1, vl); i1 += vl;
        XNN_BF16_MAXPOOL_WIDEN(t_f32, v_u16);
        out_f32 = __riscv_vfmax_vv_f32m4(out_f32, t_f32, vl);
        v_u16 = __riscv_vle16_v_u16m2(i2, vl); i2 += vl;
        XNN_BF16_MAXPOOL_WIDEN(t_f32, v_u16);
        out_f32 = __riscv_vfmax_vv_f32m4(out_f32, t_f32, vl);
        v_u16 = __riscv_vle16_v_u16m2(i3, vl); i3 += vl;
        XNN_BF16_MAXPOOL_WIDEN(t_f32, v_u16);
        out_f32 = __riscv_vfmax_vv_f32m4(out_f32, t_f32, vl);
        v_u16 = __riscv_vle16_v_u16m2(i4, vl); i4 += vl;
        XNN_BF16_MAXPOOL_WIDEN(t_f32, v_u16);
        out_f32 = __riscv_vfmax_vv_f32m4(out_f32, t_f32, vl);
        v_u16 = __riscv_vle16_v_u16m2(i5, vl); i5 += vl;
        XNN_BF16_MAXPOOL_WIDEN(t_f32, v_u16);
        out_f32 = __riscv_vfmax_vv_f32m4(out_f32, t_f32, vl);
        v_u16 = __riscv_vle16_v_u16m2(i6, vl); i6 += vl;
        XNN_BF16_MAXPOOL_WIDEN(t_f32, v_u16);
        out_f32 = __riscv_vfmax_vv_f32m4(out_f32, t_f32, vl);
        v_u16 = __riscv_vle16_v_u16m2(i7, vl); i7 += vl;
        XNN_BF16_MAXPOOL_WIDEN(t_f32, v_u16);
        out_f32 = __riscv_vfmax_vv_f32m4(out_f32, t_f32, vl);
        v_u16 = __riscv_vle16_v_u16m2(i8, vl); i8 += vl;
        XNN_BF16_MAXPOOL_WIDEN(t_f32, v_u16);
        out_f32 = __riscv_vfmax_vv_f32m4(out_f32, t_f32, vl);

        out_f32 = __riscv_vfmax_vf_f32m4(out_f32, output_min, vl);
        out_f32 = __riscv_vfmin_vf_f32m4(out_f32, output_max, vl);

        vuint16m2_t out_u16;
        __asm__ volatile(
            ".option push\n\t"
            ".option arch, +zvfbfmin\n\t"
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vfncvtbf16.f.f.w %[d], %[s]\n\t"
            ".option pop\n\t"
            : [d] "=&vr"(out_u16)
            : [s] "vr"(out_f32), [vl] "r"(vl));

        __riscv_vse16_v_u16m2(o, out_u16, vl); o += vl;

        c -= vl;
        #undef XNN_BF16_MAXPOOL_WIDEN
      } while (c != 0);
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const uint16_t* i0 = (const uint16_t*) *i++;
      const uint16_t* i1 = (const uint16_t*) (1 < k ? *i++ : (const xnn_bfloat16*) i0);
      const uint16_t* i2 = (const uint16_t*) (2 < k ? *i++ : (const xnn_bfloat16*) i0);
      const uint16_t* i3 = (const uint16_t*) (3 < k ? *i++ : (const xnn_bfloat16*) i0);
      const uint16_t* i4 = (const uint16_t*) (4 < k ? *i++ : (const xnn_bfloat16*) i0);
      const uint16_t* i5 = (const uint16_t*) (5 < k ? *i++ : (const xnn_bfloat16*) i0);
      const uint16_t* i6 = (const uint16_t*) (6 < k ? *i++ : (const xnn_bfloat16*) i0);
      const uint16_t* i7 = (const uint16_t*) (7 < k ? *i++ : (const xnn_bfloat16*) i0);
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);

      o = (uint16_t*) output;
      size_t c = channels;
      do {
        size_t vl = __riscv_vsetvl_e16m2(c);

        #define XNN_BF16_MAXPOOL_WIDEN(dst_f32m4, src_u16m2) \
            __asm__ volatile( \
                ".option push\n\t" \
                ".option arch, +zvfbfmin\n\t" \
                "vsetvli zero, %[vl], e16, m2, ta, ma\n\t" \
                "vfwcvtbf16.f.f.v %[d], %[s]\n\t" \
                ".option pop\n\t" \
                : [d] "=&vr"(dst_f32m4) \
                : [s] "vr"(src_u16m2), [vl] "r"(vl))

        // Tail-loop seeds the accumulator with the partial-max already
        // staged in the output buffer (mirrors f32-maxpool i8-from-`o`
        // pattern), then folds in 8 more inputs.
        vuint16m2_t v_u16 = __riscv_vle16_v_u16m2(o, vl);
        vfloat32m4_t out_f32; XNN_BF16_MAXPOOL_WIDEN(out_f32, v_u16);

        vfloat32m4_t t_f32;
        v_u16 = __riscv_vle16_v_u16m2(i0, vl); i0 += vl;
        XNN_BF16_MAXPOOL_WIDEN(t_f32, v_u16);
        out_f32 = __riscv_vfmax_vv_f32m4(out_f32, t_f32, vl);
        v_u16 = __riscv_vle16_v_u16m2(i1, vl); i1 += vl;
        XNN_BF16_MAXPOOL_WIDEN(t_f32, v_u16);
        out_f32 = __riscv_vfmax_vv_f32m4(out_f32, t_f32, vl);
        v_u16 = __riscv_vle16_v_u16m2(i2, vl); i2 += vl;
        XNN_BF16_MAXPOOL_WIDEN(t_f32, v_u16);
        out_f32 = __riscv_vfmax_vv_f32m4(out_f32, t_f32, vl);
        v_u16 = __riscv_vle16_v_u16m2(i3, vl); i3 += vl;
        XNN_BF16_MAXPOOL_WIDEN(t_f32, v_u16);
        out_f32 = __riscv_vfmax_vv_f32m4(out_f32, t_f32, vl);
        v_u16 = __riscv_vle16_v_u16m2(i4, vl); i4 += vl;
        XNN_BF16_MAXPOOL_WIDEN(t_f32, v_u16);
        out_f32 = __riscv_vfmax_vv_f32m4(out_f32, t_f32, vl);
        v_u16 = __riscv_vle16_v_u16m2(i5, vl); i5 += vl;
        XNN_BF16_MAXPOOL_WIDEN(t_f32, v_u16);
        out_f32 = __riscv_vfmax_vv_f32m4(out_f32, t_f32, vl);
        v_u16 = __riscv_vle16_v_u16m2(i6, vl); i6 += vl;
        XNN_BF16_MAXPOOL_WIDEN(t_f32, v_u16);
        out_f32 = __riscv_vfmax_vv_f32m4(out_f32, t_f32, vl);
        v_u16 = __riscv_vle16_v_u16m2(i7, vl); i7 += vl;
        XNN_BF16_MAXPOOL_WIDEN(t_f32, v_u16);
        out_f32 = __riscv_vfmax_vv_f32m4(out_f32, t_f32, vl);

        out_f32 = __riscv_vfmax_vf_f32m4(out_f32, output_min, vl);
        out_f32 = __riscv_vfmin_vf_f32m4(out_f32, output_max, vl);

        vuint16m2_t out_u16;
        __asm__ volatile(
            ".option push\n\t"
            ".option arch, +zvfbfmin\n\t"
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vfncvtbf16.f.f.w %[d], %[s]\n\t"
            ".option pop\n\t"
            : [d] "=&vr"(out_u16)
            : [s] "vr"(out_f32), [vl] "r"(vl));

        __riscv_vse16_v_u16m2(o, out_u16, vl); o += vl;

        c -= vl;
        #undef XNN_BF16_MAXPOOL_WIDEN
      } while (c != 0);
    }
    input = (const xnn_bfloat16**) ((uintptr_t) input + input_increment);
    input_offset += input_pixel_stride;
    output = (xnn_bfloat16*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
