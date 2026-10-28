#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

tools/xngen src/f32-layernorm/scalar.c.in -D DATATYPE=f32 -o src/f32-layernorm/gen/f32-layernorm-scalar-u1.c &
tools/xngen src/f32-layernorm/rvv.c.in -D DATATYPE=f32 -D LMUL=8 -o src/f32-layernorm/gen/f32-layernorm-rvv-u8v.c &

wait
