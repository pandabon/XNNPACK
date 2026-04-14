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
// The kernel itself emits no Zvfbfmin instructions — it is plain base-V.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

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
  const ptrdiff_t kc_bstride = (ptrdiff_t) (kc * sizeof(uint16_t));
  // u16 elements per NR-wide bias slot (NR fp32 lanes = 2*NR u16 elements).
  const size_t bias_u16_stride = nr * (sizeof(uint32_t) / sizeof(uint16_t));

  do {
    const uint16_t* w0 = weights;
    size_t n = nc;

    // Main NR-aligned tiles.
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

    // NC tail: n < nr active lanes, but output strides remain nr-wide.
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
