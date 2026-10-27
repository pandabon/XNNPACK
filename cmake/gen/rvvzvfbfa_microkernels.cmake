# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for rvvzvfbfa
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_RVVZVFBFA_MICROKERNEL_SRCS
  src/bf16-f32-gemm/gen/bf16-f32-gemm-5x4v-minmax-rvvzvfbfa.c)

SET(NON_PROD_RVVZVFBFA_MICROKERNEL_SRCS
  src/bf16-f32-gemm/gen/bf16-f32-gemm-1x4v-minmax-rvvzvfbfa.c
  src/bf16-f32-gemm/gen/bf16-f32-gemm-7x4v-minmax-rvvzvfbfa.c)

SET(ALL_RVVZVFBFA_MICROKERNEL_SRCS ${PROD_RVVZVFBFA_MICROKERNEL_SRCS} ${NON_PROD_RVVZVFBFA_MICROKERNEL_SRCS})
