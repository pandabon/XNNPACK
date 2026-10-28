#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

tools/xngen src/f32-layernorm/scalar.c.in -D DATATYPE=bf16 -o src/bf16-layernorm/gen/bf16-layernorm-scalar-u1.c &
tools/xngen src/f32-layernorm/rvv.c.in -D DATATYPE=bf16 -D LMUL=8 -o src/bf16-layernorm/gen/bf16-layernorm-rvv-u8v.c &

wait
