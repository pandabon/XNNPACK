// Copyright 2025 UC Berkeley BAR
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Dynamic-NR packw for bf16 GEMM: packs bf16 weights + fp32 bias into
// NR-interleaved layout consumed by bf16-gemm-{1,4}x4v-rvv-zvfbfmin.
//
// NR = vsetvlmax_e32m4() at runtime (= vlenb for VLEN=256, i.e. 32).
//
// Output layout per NR-column tile:
//   [fp32 bias × NR]          -- NR * sizeof(float) bytes
//   [bf16 weights × NR]       -- NR * sizeof(uint16_t) bytes, per KC step
//
// Modeled after src/x32-packw/gen/x32-packw-x4v-gemm-goi-rvv-u8.c
// but adapted for 16-bit (bf16) weights and 32-bit (fp32) bias.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "src/xnnpack/packw.h"

void xnn_x16_x32_packw_gemm_goi_ukernel_x4v__rvv_u8(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* weights,   // bf16 weights [nc × kc] in GOI order
  const uint32_t* bias,      // fp32 bias [nc]
  const void* scale,
  uint16_t* packed_weights,  // output: NR-interleaved [bias fp32 | bf16 weights]
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
  // Byte stride between consecutive output-neuron weight rows.
  const size_t kc_bstride = kc * sizeof(uint16_t);

  do {
    const uint16_t* w0 = weights;
    size_t n = nc;

    // ------------------------------------------------------------------
    // NC main loop: process multiples of NR output neurons.
    // ------------------------------------------------------------------
    for (; n >= nr; n -= nr) {
      // Pack NR fp32 bias values.
      size_t vlmax = __riscv_vsetvlmax_e32m4();
      vuint32m4_t v_bias;
      if XNN_LIKELY(b != NULL) {
        v_bias = __riscv_vle32_v_u32m4(b, vlmax); b += nr;
      } else {
        v_bias = __riscv_vmv_v_x_u32m4(0, vlmax);
      }
      // Write as raw bytes (fp32 stored in the uint16_t* output buffer).
      memcpy(out, &v_bias, nr * sizeof(uint32_t));  // safe: __riscv_vse32 equivalent
      out += nr * 2;  // advance by NR fp32 values (each = 2 uint16 words)

      // Pack NR × KC bf16 weights, 8 KC rows at a time.
      // vlsseg8e16 requires LMUL <= 1 (vlmax_e16m1 elements per chunk).
      uint16_t* out0 = out;
      size_t k = kc;
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

      // 4 KC rows at a time (vlsseg4e16, LMUL <= 2).
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

    // ------------------------------------------------------------------
    // NC remainder: n < NR output neurons.
    // ------------------------------------------------------------------
    if (n > 0) {
      size_t vl = __riscv_vsetvl_e32m4(n);

      // Pack partial fp32 bias; zero-pad to NR slots.
      vuint32m4_t v_bias;
      if XNN_LIKELY(b != NULL) {
        v_bias = __riscv_vle32_v_u32m4(b, vl); b += vl;
      } else {
        v_bias = __riscv_vmv_v_x_u32m4(0, vl);
      }
      memcpy(out, &v_bias, vl * sizeof(uint32_t));
      memset((uint8_t*)out + vl * sizeof(uint32_t), 0,
             (nr - vl) * sizeof(uint32_t));
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
