#!/bin/sh
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################

## C2 packing
tools/xngen src/x16-x32-packw/kr-scalar.c.in -D NR=32  -D KR=2 -D TYPE=uint16_t -D BTYPE=uint32_t -o src/x16-x32-packw/gen/x16-x32-packw-x32c2-gemm-goi-scalar.c &
tools/xngen src/x16-x32-packw/kr-gio-scalar.c.in -D NR=32 -D KR=2 -D TYPE=uint16_t -D BTYPE=uint32_t -o src/x16-x32-packw/gen/x16-x32-packw-x32c2-gemm-gio-scalar.c &

################################### RISC-V RVV ################################
tools/xngen src/x16-x32-packw/rvv.c.in -D NR=m4 -D KBLOCK=2 -o src/x16-x32-packw/gen/x16-x32-packw-x4v-gemm-goi-rvv-u2.c &
tools/xngen src/x16-x32-packw/rvv.c.in -D NR=m4 -D KBLOCK=4 -o src/x16-x32-packw/gen/x16-x32-packw-x4v-gemm-goi-rvv-u4.c &

wait
