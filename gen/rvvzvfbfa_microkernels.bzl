#
# Microkernel filenames lists for rvvzvfbfa.
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py
#

PROD_RVVZVFBFA_MICROKERNEL_SRCS = [
    "src/bf16-f32-gemm/gen/bf16-f32-gemm-5x4v-minmax-rvvzvfbfa.c",
]

NON_PROD_RVVZVFBFA_MICROKERNEL_SRCS = [
    "src/bf16-f32-gemm/gen/bf16-f32-gemm-1x4v-minmax-rvvzvfbfa.c",
    "src/bf16-f32-gemm/gen/bf16-f32-gemm-7x4v-minmax-rvvzvfbfa.c",
]

ALL_RVVZVFBFA_MICROKERNEL_SRCS = PROD_RVVZVFBFA_MICROKERNEL_SRCS + NON_PROD_RVVZVFBFA_MICROKERNEL_SRCS
