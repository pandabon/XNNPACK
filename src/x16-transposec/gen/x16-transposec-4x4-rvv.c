// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x32-transposec/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 UC Berkeley BAR — extended for SIZE=16 (x16-transposec).
// Copyright 2023 SiFive, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <riscv_vector.h>

#include <assert.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/transpose.h"

void xnn_x16_transposec_ukernel__4x4_rvv(
  const uint16_t* input,
  uint16_t* output,
  size_t input_stride,
  size_t output_stride,
  size_t block_width,
  size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint16_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint16_t));

  const size_t tile_height = 4;
  const size_t tile_width = 4;
  const size_t tile_hbytes = tile_height * sizeof(uint16_t);
  const size_t tile_wbytes = tile_width * sizeof(uint16_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t input_offset = tile_height * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint16_t);

  const uint16_t* i0 = input;

  uint16_t* o0 = (uint16_t*) output;
  uint16_t* o1 = (uint16_t*) ((uintptr_t) o0 + output_stride);
  uint16_t* o2 = (uint16_t*) ((uintptr_t) o1 + output_stride);
  uint16_t* o3 = (uint16_t*) ((uintptr_t) o2 + output_stride);

  do {
    size_t bh = block_height;
    size_t vl = __riscv_vsetvl_e16m1(tile_height);
    // Inner-loop contract: when bh >= tile_height, vsetvl must return
    // tile_height (i.e. host vlmax >= tile_height). Below that the loop
    // would advance pointers by tile_hbytes while only writing vl rows
    // → silent under-processing. The runtime config ladder enforces this
    // by selecting kernels whose tile_height matches vlmax exactly; this
    // assert catches a hypothetical ladder regression in debug builds.
    assert(block_height < tile_height || vl == tile_height);
    for (; bh >= 4; bh -= 4) {
      if (block_width >= tile_width) {
        vuint16m1x4_t tuple = __riscv_vlsseg4e16_v_u16m1x4(i0, input_stride, vl);

        vuint16m1_t v_d0 = __riscv_vget_v_u16m1x4_u16m1(tuple, 0);
        __riscv_vse16_v_u16m1(o0, v_d0, vl);
        vuint16m1_t v_d1 = __riscv_vget_v_u16m1x4_u16m1(tuple, 1);
        __riscv_vse16_v_u16m1(o1, v_d1, vl);
        vuint16m1_t v_d2 = __riscv_vget_v_u16m1x4_u16m1(tuple, 2);
        __riscv_vse16_v_u16m1(o2, v_d2, vl);
        vuint16m1_t v_d3 = __riscv_vget_v_u16m1x4_u16m1(tuple, 3);
        __riscv_vse16_v_u16m1(o3, v_d3, vl);

      } else {
        switch (block_width) {
          case 3: {
            vuint16m1x3_t tuple = __riscv_vlsseg3e16_v_u16m1x3(i0, input_stride, vl);

            vuint16m1_t v_d0 = __riscv_vget_v_u16m1x3_u16m1(tuple, 0);
            __riscv_vse16_v_u16m1(o0, v_d0, vl);
            vuint16m1_t v_d1 = __riscv_vget_v_u16m1x3_u16m1(tuple, 1);
            __riscv_vse16_v_u16m1(o1, v_d1, vl);
            vuint16m1_t v_d2 = __riscv_vget_v_u16m1x3_u16m1(tuple, 2);
            __riscv_vse16_v_u16m1(o2, v_d2, vl);
            break;
          }

          case 2: {
            vuint16m1x2_t tuple = __riscv_vlsseg2e16_v_u16m1x2(i0, input_stride, vl);

            vuint16m1_t v_d0 = __riscv_vget_v_u16m1x2_u16m1(tuple, 0);
            __riscv_vse16_v_u16m1(o0, v_d0, vl);
            vuint16m1_t v_d1 = __riscv_vget_v_u16m1x2_u16m1(tuple, 1);
            __riscv_vse16_v_u16m1(o1, v_d1, vl);
            break;
          }

          case 1: {
            vuint16m1_t v_d0 = __riscv_vlse16_v_u16m1(i0, input_stride, vl);
            __riscv_vse16_v_u16m1(o0, v_d0, vl);
            break;
          }

          default:
            XNN_UNREACHABLE;
        }
      }

      i0 = (uint16_t*) ((uintptr_t) i0 + input_offset);
      o3 = (uint16_t*) ((uintptr_t) o3 + tile_hbytes);
      o2 = (uint16_t*) ((uintptr_t) o2 + tile_hbytes);
      o1 = (uint16_t*) ((uintptr_t) o1 + tile_hbytes);
      o0 = (uint16_t*) ((uintptr_t) o0 + tile_hbytes);
    }

    if (bh != 0) {
      const uint16_t* i = i0;
      vl = __riscv_vsetvl_e16m1(bh);
      // Trailing-path contract: vsetvl(bh) must return bh (i.e. vlmax >= bh,
      // and bh < tile_height ≤ vlmax holds when the kernel is correctly
      // selected). See the inner-loop assert above for the ladder rationale.
      assert(vl == bh);
      if (block_width >= tile_width) {
        vuint16m1x4_t tuple = __riscv_vlsseg4e16_v_u16m1x4(i, input_stride, vl);

        vuint16m1_t v_d0 = __riscv_vget_v_u16m1x4_u16m1(tuple, 0);
        __riscv_vse16_v_u16m1(o0, v_d0, vl);
        vuint16m1_t v_d1 = __riscv_vget_v_u16m1x4_u16m1(tuple, 1);
        __riscv_vse16_v_u16m1(o1, v_d1, vl);
        vuint16m1_t v_d2 = __riscv_vget_v_u16m1x4_u16m1(tuple, 2);
        __riscv_vse16_v_u16m1(o2, v_d2, vl);
        vuint16m1_t v_d3 = __riscv_vget_v_u16m1x4_u16m1(tuple, 3);
        __riscv_vse16_v_u16m1(o3, v_d3, vl);
      } else {
        switch(block_width) {
          case 3: {
            vuint16m1x3_t tuple = __riscv_vlsseg3e16_v_u16m1x3(i, input_stride, vl);

            vuint16m1_t v_d0 = __riscv_vget_v_u16m1x3_u16m1(tuple, 0);
            __riscv_vse16_v_u16m1(o0, v_d0, vl);
            vuint16m1_t v_d1 = __riscv_vget_v_u16m1x3_u16m1(tuple, 1);
            __riscv_vse16_v_u16m1(o1, v_d1, vl);
            vuint16m1_t v_d2 = __riscv_vget_v_u16m1x3_u16m1(tuple, 2);
            __riscv_vse16_v_u16m1(o2, v_d2, vl);
            break;
          }
          case 2: {
            vuint16m1x2_t tuple = __riscv_vlsseg2e16_v_u16m1x2(i, input_stride, vl);

            vuint16m1_t v_d0 = __riscv_vget_v_u16m1x2_u16m1(tuple, 0);
            __riscv_vse16_v_u16m1(o0, v_d0, vl);
            vuint16m1_t v_d1 = __riscv_vget_v_u16m1x2_u16m1(tuple, 1);
            __riscv_vse16_v_u16m1(o1, v_d1, vl);
            break;
          }

          case 1: {
            vuint16m1_t v_d0 = __riscv_vlse16_v_u16m1(i, input_stride, vl);
            __riscv_vse16_v_u16m1(o0, v_d0, vl);
            break;
          }

          default:
            XNN_UNREACHABLE;
        }
      }

      if (bh & 2) {
        o3 += 2;
        o2 += 2;
        o1 += 2;
        o0 += 2;
        i = (uint16_t*) ((uintptr_t) i + input_stride * 2);
      }
    }

    i0 = (const uint16_t*) ((uintptr_t) i0 + input_reset);

    o0 = (uint16_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint16_t*) ((uintptr_t) o1 + output_reset);
    o2 = (uint16_t*) ((uintptr_t) o2 + output_reset);
    o3 = (uint16_t*) ((uintptr_t) o3 + output_reset);

    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
