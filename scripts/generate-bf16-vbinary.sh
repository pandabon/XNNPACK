#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# bf16 binary elementwise micro-kernels: add, multiply, subtract (vop: both inputs are vectors; vopc: the second input
# is one scalar; vrsubc computes b - a). All widen to f32, compute in f32 and round back to bf16.

################################### Scalar ####################################
tools/xngen src/bf16-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=4 -o src/bf16-vbinary/gen/bf16-vadd-scalar-u4.c &
tools/xngen src/bf16-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=4 -o src/bf16-vbinary/gen/bf16-vmul-scalar-u4.c &
tools/xngen src/bf16-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=4 -o src/bf16-vbinary/gen/bf16-vsub-scalar-u4.c &

tools/xngen src/bf16-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=4 -o src/bf16-vbinary/gen/bf16-vaddc-scalar-u4.c &
tools/xngen src/bf16-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=4 -o src/bf16-vbinary/gen/bf16-vmulc-scalar-u4.c &
tools/xngen src/bf16-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=4 -o src/bf16-vbinary/gen/bf16-vsubc-scalar-u4.c &
tools/xngen src/bf16-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=4 -o src/bf16-vbinary/gen/bf16-vrsubc-scalar-u4.c &

################################### RISC-V Vector #############################
tools/xngen src/bf16-vbinary/vop-rvv.c.in -D OP=ADD -D LMUL=2 -o src/bf16-vbinary/gen/bf16-vadd-rvv-u2v.c &
tools/xngen src/bf16-vbinary/vop-rvv.c.in -D OP=ADD -D LMUL=4 -o src/bf16-vbinary/gen/bf16-vadd-rvv-u4v.c &
tools/xngen src/bf16-vbinary/vop-rvv.c.in -D OP=MUL -D LMUL=2 -o src/bf16-vbinary/gen/bf16-vmul-rvv-u2v.c &
tools/xngen src/bf16-vbinary/vop-rvv.c.in -D OP=MUL -D LMUL=4 -o src/bf16-vbinary/gen/bf16-vmul-rvv-u4v.c &
tools/xngen src/bf16-vbinary/vop-rvv.c.in -D OP=SUB -D LMUL=2 -o src/bf16-vbinary/gen/bf16-vsub-rvv-u2v.c &
tools/xngen src/bf16-vbinary/vop-rvv.c.in -D OP=SUB -D LMUL=4 -o src/bf16-vbinary/gen/bf16-vsub-rvv-u4v.c &

tools/xngen src/bf16-vbinary/vopc-rvv.c.in -D OP=ADD  -D LMUL=2 -o src/bf16-vbinary/gen/bf16-vaddc-rvv-u2v.c &
tools/xngen src/bf16-vbinary/vopc-rvv.c.in -D OP=ADD  -D LMUL=4 -o src/bf16-vbinary/gen/bf16-vaddc-rvv-u4v.c &
tools/xngen src/bf16-vbinary/vopc-rvv.c.in -D OP=MUL  -D LMUL=2 -o src/bf16-vbinary/gen/bf16-vmulc-rvv-u2v.c &
tools/xngen src/bf16-vbinary/vopc-rvv.c.in -D OP=MUL  -D LMUL=4 -o src/bf16-vbinary/gen/bf16-vmulc-rvv-u4v.c &
tools/xngen src/bf16-vbinary/vopc-rvv.c.in -D OP=SUB  -D LMUL=2 -o src/bf16-vbinary/gen/bf16-vsubc-rvv-u2v.c &
tools/xngen src/bf16-vbinary/vopc-rvv.c.in -D OP=SUB  -D LMUL=4 -o src/bf16-vbinary/gen/bf16-vsubc-rvv-u4v.c &
tools/xngen src/bf16-vbinary/vopc-rvv.c.in -D OP=RSUB -D LMUL=2 -o src/bf16-vbinary/gen/bf16-vrsubc-rvv-u2v.c &
tools/xngen src/bf16-vbinary/vopc-rvv.c.in -D OP=RSUB -D LMUL=4 -o src/bf16-vbinary/gen/bf16-vrsubc-rvv-u4v.c &

wait
