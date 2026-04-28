// Copyright 2026 UC Berkeley BAR
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// BF16 weight + FP32 bias packw for the RISC-V Vector GEMM kernels
// (xnn_bf16_*_gemm_minmax_ukernel_*x4v__rvv_zvfbfmin).
//
// Per NR-tile layout (NR = vsetvlmax_e32m4(), 32 fp32 lanes at VLEN=256):
//   [fp32 bias  × NR]           NR * 4 bytes
//   [bf16 weights × NR] × KC    NR * 2 bytes per KC step
//
// Inner weight pack uses vlsseg8e16 / vlsseg4e16 segmented loads at
// e16m1 / e16m2 to amortize KC iterations 8- and 4-fold respectively.
// The _e16m1 typing gives gcc latitude to emit `e8/mf2` vsetvli at
// EMUL=1 for vlsseg8e16 — this matters on Saturn where each segment
// fetch is a DLEN-aligned 16-byte access (vs 32 single-halfword
// strided requests in the simpler m2 strided alternative below).
// The strided alternative is preserved at the end of this file as
// reference and as a fallback if the segmented path needs to be
// disabled (e.g., for a DUT whose vlsseg+fractional-LMUL path
// regresses).

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/packw.h"


void xnn_x16_x32_packw_gemm_goi_ukernel_x4v__rvv_u8(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* weights,
  const uint32_t* bias,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == __riscv_vsetvlmax_e32m4());
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  uint16_t* out = packed_weights;
  const uint32_t* b = bias;
  const size_t kc_bstride = kc * sizeof(uint16_t);

  do {
    const uint16_t* w0 = weights;
    size_t n = nc;

    // NC main loop: process multiples of NR output neurons.
    for (; n >= nr; n -= nr) {
      size_t vlmax = __riscv_vsetvlmax_e32m4();
      vuint32m4_t v_bias;
      if XNN_LIKELY(b != NULL) {
        v_bias = __riscv_vle32_v_u32m4(b, vlmax); b += nr;
      } else {
        v_bias = __riscv_vmv_v_x_u32m4(0, vlmax);
      }
      __riscv_vse32_v_u32m4((uint32_t*) out, v_bias, vlmax);
      out += nr * 2;  // NR fp32 bias = 2*NR uint16_t units

      uint16_t* out0 = out;
      size_t k = kc;

      // 8 KC rows at a time (vlsseg8e16, EEW=16, EMUL=1 ⇒ gcc may emit e8/mf2).
      vlmax = __riscv_vsetvlmax_e16m1();
      for (; k >= 8; k -= 8) {
        uint16_t* out1 = out0 + nr;
        uint16_t* out2 = out1 + nr;
        uint16_t* out3 = out2 + nr;
        uint16_t* out4 = out3 + nr;
        uint16_t* out5 = out4 + nr;
        uint16_t* out6 = out5 + nr;
        uint16_t* out7 = out6 + nr;

        const uint16_t* w_ptr = w0;
        size_t remaining_n = nr;
        do {
          vuint16m1x8_t v_w = __riscv_vlsseg8e16_v_u16m1x8(w_ptr, kc_bstride, vlmax);
          w_ptr += kc * vlmax;

          __riscv_vse16_v_u16m1(out0, __riscv_vget_v_u16m1x8_u16m1(v_w, 0), vlmax); out0 += vlmax;
          __riscv_vse16_v_u16m1(out1, __riscv_vget_v_u16m1x8_u16m1(v_w, 1), vlmax); out1 += vlmax;
          __riscv_vse16_v_u16m1(out2, __riscv_vget_v_u16m1x8_u16m1(v_w, 2), vlmax); out2 += vlmax;
          __riscv_vse16_v_u16m1(out3, __riscv_vget_v_u16m1x8_u16m1(v_w, 3), vlmax); out3 += vlmax;
          __riscv_vse16_v_u16m1(out4, __riscv_vget_v_u16m1x8_u16m1(v_w, 4), vlmax); out4 += vlmax;
          __riscv_vse16_v_u16m1(out5, __riscv_vget_v_u16m1x8_u16m1(v_w, 5), vlmax); out5 += vlmax;
          __riscv_vse16_v_u16m1(out6, __riscv_vget_v_u16m1x8_u16m1(v_w, 6), vlmax); out6 += vlmax;
          __riscv_vse16_v_u16m1(out7, __riscv_vget_v_u16m1x8_u16m1(v_w, 7), vlmax); out7 += vlmax;
          remaining_n -= vlmax;
        } while (remaining_n > 0);
        out0 = out7;
        w0 += 8;
      }

      // 4 KC rows at a time (vlsseg4e16, EMUL=2).
      vlmax = __riscv_vsetvlmax_e16m2();
      for (; k >= 4; k -= 4) {
        uint16_t* out1 = out0 + nr;
        uint16_t* out2 = out1 + nr;
        uint16_t* out3 = out2 + nr;

        const uint16_t* w_ptr = w0;
        size_t remaining_n = nr;
        do {
          vuint16m2x4_t v_w = __riscv_vlsseg4e16_v_u16m2x4(w_ptr, kc_bstride, vlmax);
          w_ptr += kc * vlmax;

          __riscv_vse16_v_u16m2(out0, __riscv_vget_v_u16m2x4_u16m2(v_w, 0), vlmax); out0 += vlmax;
          __riscv_vse16_v_u16m2(out1, __riscv_vget_v_u16m2x4_u16m2(v_w, 1), vlmax); out1 += vlmax;
          __riscv_vse16_v_u16m2(out2, __riscv_vget_v_u16m2x4_u16m2(v_w, 2), vlmax); out2 += vlmax;
          __riscv_vse16_v_u16m2(out3, __riscv_vget_v_u16m2x4_u16m2(v_w, 3), vlmax); out3 += vlmax;
          remaining_n -= vlmax;
        } while (remaining_n > 0);
        out0 = out3;
        w0 += 4;
      }

      // Remaining KC rows one at a time.
      vlmax = __riscv_vsetvlmax_e16m2();
      for (; k >= 1; k -= 1) {
        vuint16m2_t v_w = __riscv_vlse16_v_u16m2(w0, kc_bstride, vlmax);
        __riscv_vse16_v_u16m2(out0, v_w, vlmax);
        out0 += vlmax;
        w0 += 1;
      }

      out = (uint16_t*) ((uintptr_t) out0 + extra_bytes);
      w0 += (nr - 1) * kc;
    }

    // NC remainder: n < NR active output neurons (output strides remain nr-wide).
    if (n > 0) {
      size_t vl = __riscv_vsetvl_e32m4(n);
      vuint32m4_t v_bias;
      if XNN_LIKELY(b != NULL) {
        v_bias = __riscv_vle32_v_u32m4(b, vl); b += vl;
      } else {
        v_bias = __riscv_vmv_v_x_u32m4(0, vl);
      }
      __riscv_vse32_v_u32m4((uint32_t*) out, v_bias, vl);
      // Zero-pad the remaining (nr - vl) fp32 bias slots.
      memset((uint8_t*) out + vl * sizeof(uint32_t), 0, (nr - vl) * sizeof(uint32_t));
      out += nr * 2;

      uint16_t* out0 = out;
      size_t k = kc;
      size_t vlmax;

      // 8 KC rows at a time.
      vlmax = __riscv_vsetvlmax_e16m1();
      for (; k >= 8; k -= 8) {
        uint16_t* out1 = out0 + nr;
        uint16_t* out2 = out1 + nr;
        uint16_t* out3 = out2 + nr;
        uint16_t* out4 = out3 + nr;
        uint16_t* out5 = out4 + nr;
        uint16_t* out6 = out5 + nr;
        uint16_t* out7 = out6 + nr;

        const uint16_t* w_ptr = w0;
        unsigned char remaining_blocks = 2;
        size_t remaining_n = n;
        do {
          size_t chunk_vl = XNN_LIKELY(remaining_n >= vlmax)
                              ? vlmax
                              : __riscv_vsetvl_e16m1(remaining_n);
          vuint16m1x8_t v_w = __riscv_vlsseg8e16_v_u16m1x8(w_ptr, kc_bstride, chunk_vl);
          w_ptr += kc * chunk_vl;

          __riscv_vse16_v_u16m1(out0, __riscv_vget_v_u16m1x8_u16m1(v_w, 0), chunk_vl); out0 += vlmax;
          __riscv_vse16_v_u16m1(out1, __riscv_vget_v_u16m1x8_u16m1(v_w, 1), chunk_vl); out1 += vlmax;
          __riscv_vse16_v_u16m1(out2, __riscv_vget_v_u16m1x8_u16m1(v_w, 2), chunk_vl); out2 += vlmax;
          __riscv_vse16_v_u16m1(out3, __riscv_vget_v_u16m1x8_u16m1(v_w, 3), chunk_vl); out3 += vlmax;
          __riscv_vse16_v_u16m1(out4, __riscv_vget_v_u16m1x8_u16m1(v_w, 4), chunk_vl); out4 += vlmax;
          __riscv_vse16_v_u16m1(out5, __riscv_vget_v_u16m1x8_u16m1(v_w, 5), chunk_vl); out5 += vlmax;
          __riscv_vse16_v_u16m1(out6, __riscv_vget_v_u16m1x8_u16m1(v_w, 6), chunk_vl); out6 += vlmax;
          __riscv_vse16_v_u16m1(out7, __riscv_vget_v_u16m1x8_u16m1(v_w, 7), chunk_vl); out7 += vlmax;
          remaining_n -= chunk_vl;
          remaining_blocks--;
        } while (remaining_n > 0);
        out0 = out7 + remaining_blocks * vlmax;
        w0 += 8;
      }

      // 4 KC rows at a time.
      vlmax = __riscv_vsetvlmax_e16m2();
      for (; k >= 4; k -= 4) {
        uint16_t* out1 = out0 + nr;
        uint16_t* out2 = out1 + nr;
        uint16_t* out3 = out2 + nr;

        const uint16_t* w_ptr = w0;
        unsigned char remaining_blocks = 1;
        size_t remaining_n = n;
        do {
          size_t chunk_vl = XNN_LIKELY(remaining_n >= vlmax)
                              ? vlmax
                              : __riscv_vsetvl_e16m2(remaining_n);
          vuint16m2x4_t v_w = __riscv_vlsseg4e16_v_u16m2x4(w_ptr, kc_bstride, chunk_vl);
          w_ptr += kc * chunk_vl;

          __riscv_vse16_v_u16m2(out0, __riscv_vget_v_u16m2x4_u16m2(v_w, 0), chunk_vl); out0 += vlmax;
          __riscv_vse16_v_u16m2(out1, __riscv_vget_v_u16m2x4_u16m2(v_w, 1), chunk_vl); out1 += vlmax;
          __riscv_vse16_v_u16m2(out2, __riscv_vget_v_u16m2x4_u16m2(v_w, 2), chunk_vl); out2 += vlmax;
          __riscv_vse16_v_u16m2(out3, __riscv_vget_v_u16m2x4_u16m2(v_w, 3), chunk_vl); out3 += vlmax;
          remaining_n -= chunk_vl;
          remaining_blocks--;
        } while (remaining_n > 0);
        out0 = out3 + remaining_blocks * vlmax;
        w0 += 4;
      }

      // Remaining KC rows one at a time.
      vl = __riscv_vsetvl_e16m2(n);
      vlmax = __riscv_vsetvlmax_e16m2();
      for (; k >= 1; k -= 1) {
        vuint16m2_t v_w = __riscv_vlse16_v_u16m2(w0, kc_bstride, vl);
        __riscv_vse16_v_u16m2(out0, v_w, vl);
        out0 += vlmax;
        w0 += 1;
      }

      out = (uint16_t*) ((uintptr_t) out0 + extra_bytes);
      w0 += (nr - 1) * kc;
    }

    weights += nc * kc;
  } while (--g != 0);
}


// ---------------------------------------------------------------------------
// Reference: simpler strided alternative (one KC row per inner iter).
//
// Kept as a fallback in case the segmented path above regresses on a
// DUT whose vlsseg or fractional-LMUL handling is broken (cf. the
// Saturn vlsseg8e16 + e8/mf2 EMUL bug — fixed; this is the workaround
// that was used before the fix). On a spec-compliant simulator both
// produce identical packed-buffer output.
//
// Why segmented is preferred for production on DLEN=128:
//   - vlse16 e16m2:  32 strided 2-byte fetches per inner iter. Each
//                    LSU request transfers 2 of the 16 bytes the
//                    DLEN beat can carry (12.5% useful).
//   - vlsseg8e16 e16m1: 16 strided 16-byte segment fetches per chunk.
//                    Each LSU request transfers a full DLEN beat.
//
// To re-enable, swap the block above for the body below.
// ---------------------------------------------------------------------------
#if 0
void xnn_x16_x32_packw_gemm_goi_ukernel_x4v__rvv_u8(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* weights,
  const uint32_t* bias,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == __riscv_vsetvlmax_e32m4());
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  uint16_t* out = packed_weights;
  const uint32_t* b = bias;
  const ptrdiff_t kc_bstride = (ptrdiff_t) (kc * sizeof(uint16_t));
  const size_t bias_u16_stride = nr * (sizeof(uint32_t) / sizeof(uint16_t));

  do {
    const uint16_t* w0 = weights;
    size_t n = nc;

    for (; n >= nr; n -= nr) {
      vuint32m4_t v_bias;
      if XNN_LIKELY(b != NULL) {
        v_bias = __riscv_vle32_v_u32m4(b, nr);
        b += nr;
      } else {
        v_bias = __riscv_vmv_v_x_u32m4(0, nr);
      }
      __riscv_vse32_v_u32m4((uint32_t*) out, v_bias, nr);
      out += bias_u16_stride;

      for (size_t k = 0; k < kc; ++k) {
        vuint16m2_t v_w = __riscv_vlse16_v_u16m2(w0 + k, kc_bstride, nr);
        __riscv_vse16_v_u16m2(out, v_w, nr);
        out += nr;
      }
      out = (uint16_t*) ((uintptr_t) out + extra_bytes);
      w0 += nr * kc;
    }

    if (n != 0) {
      vuint32m4_t v_bias;
      if XNN_LIKELY(b != NULL) {
        v_bias = __riscv_vle32_v_u32m4(b, n);
        b += n;
      } else {
        v_bias = __riscv_vmv_v_x_u32m4(0, n);
      }
      __riscv_vse32_v_u32m4((uint32_t*) out, v_bias, n);
      out += bias_u16_stride;

      for (size_t k = 0; k < kc; ++k) {
        vuint16m2_t v_w = __riscv_vlse16_v_u16m2(w0 + k, kc_bstride, n);
        __riscv_vse16_v_u16m2(out, v_w, n);
        out += nr;
      }
      out = (uint16_t*) ((uintptr_t) out + extra_bytes);
      w0 += n * kc;
    }

    weights += nc * kc;
  } while (--g != 0);
}
#endif
