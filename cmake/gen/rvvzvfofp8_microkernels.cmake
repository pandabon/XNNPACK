# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for rvvzvfofp8
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_RVVZVFOFP8_MICROKERNEL_SRCS)

SET(NON_PROD_RVVZVFOFP8_MICROKERNEL_SRCS
  src/bf16-qf8w-gemm/gen/bf16-qf8w-f32-gemm-1x4v-minmax-rvvzvfofp8.c
  src/bf16-qf8w-gemm/gen/bf16-qf8w-f32-gemm-5x4v-minmax-rvvzvfofp8.c
  src/bf16-qf8w-gemm/gen/bf16-qf8w-f32-gemm-7x4v-minmax-rvvzvfofp8.c)

SET(ALL_RVVZVFOFP8_MICROKERNEL_SRCS ${PROD_RVVZVFOFP8_MICROKERNEL_SRCS} ${NON_PROD_RVVZVFOFP8_MICROKERNEL_SRCS})
