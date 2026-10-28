// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Layer normalization over the channels (last) dimension of an NC tensor, in f32 and bf16.
// Plumbing modelled on src/operators/softmax-nc.c (create / reshape / setup, a 1D compute task per row).
// The scale (gamma) and bias (beta) vectors are copied at create time, so the caller may free them afterwards.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "include/xnnpack.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/compute.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/operator-type.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/params.h"
#include <pthreadpool.h>

static enum xnn_status create_layer_norm_nc(
    size_t channels,
    float epsilon,
    const void* scale,
    const void* bias,
    uint32_t flags,
    size_t log2_element_size,
    const struct xnn_layernorm_config* config,
    enum xnn_operator_type operator_type,
    xnn_operator_t* layer_norm_op_out)
{
  xnn_operator_t layer_norm_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(operator_type), channels);
    goto error;
  }

  if (!(epsilon >= 0.0f) || !isfinite(epsilon)) {
    xnn_log_error(
      "failed to create %s operator with %.7g epsilon: epsilon must be finite and non-negative",
      xnn_operator_type_to_string(operator_type), epsilon);
    goto error;
  }

  if (scale == NULL || bias == NULL) {
    xnn_log_error(
      "failed to create %s operator: scale and bias must be provided",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_unsupported_hardware;

  if (config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_out_of_memory;

  layer_norm_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (layer_norm_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  layer_norm_op->compute = xnn_allocate_zero_memory(sizeof(struct compute_parameters));
  if (layer_norm_op->compute == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct compute_parameters),
                  xnn_operator_type_to_string(operator_type));
    goto error;
  }
  layer_norm_op->num_compute_invocations = 1;

  // scale followed by bias, owned by the operator (released by xnn_destroy_operator).
  const size_t vector_bytes = channels << log2_element_size;
  layer_norm_op->packed_weights.pointer = xnn_allocate_simd_memory(2 * vector_bytes);
  if (layer_norm_op->packed_weights.pointer == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator scale and bias",
                  2 * vector_bytes, xnn_operator_type_to_string(operator_type));
    goto error;
  }
  memcpy(layer_norm_op->packed_weights.pointer, scale, vector_bytes);
  memcpy((char*) layer_norm_op->packed_weights.pointer + vector_bytes, bias, vector_bytes);

  layer_norm_op->channels = channels;
  layer_norm_op->type = operator_type;
  layer_norm_op->flags = flags;
  layer_norm_op->layernorm_config = config;
  layer_norm_op->state = xnn_run_state_invalid;

  // epsilon is kept in the compute context; fill it now so reshape only has to set the geometry.
  if (operator_type == xnn_operator_type_layer_norm_nc_f32) {
    config->init.f32(&layer_norm_op->context.layernorm.params.f32, epsilon);
  } else {
    config->init.bf16(&layer_norm_op->context.layernorm.params.bf16, epsilon);
  }

  *layer_norm_op_out = layer_norm_op;
  return xnn_status_success;

error:
  xnn_delete_operator(layer_norm_op);
  return status;
}

enum xnn_status xnn_create_layer_norm_nc_f32(
    size_t channels,
    float epsilon,
    const float* scale,
    const float* bias,
    uint32_t flags,
    xnn_operator_t* layer_norm_op_out)
{
  return create_layer_norm_nc(
    channels, epsilon, scale, bias, flags,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    xnn_init_f32_layernorm_config(),
    xnn_operator_type_layer_norm_nc_f32,
    layer_norm_op_out);
}

enum xnn_status xnn_create_layer_norm_nc_bf16(
    size_t channels,
    float epsilon,
    const void* scale,
    const void* bias,
    uint32_t flags,
    xnn_operator_t* layer_norm_op_out)
{
  return create_layer_norm_nc(
    channels, epsilon, scale, bias, flags,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_BFLOAT16,
    xnn_init_bf16_layernorm_config(),
    xnn_operator_type_layer_norm_nc_bf16,
    layer_norm_op_out);
}

static enum xnn_status reshape_layer_norm_nc(
    xnn_operator_t layer_norm_op,
    enum xnn_operator_type expected_operator_type,
    size_t batch_size,
    size_t input_stride,
    size_t output_stride,
    uint32_t log2_element_size)
{
  if (layer_norm_op->type != expected_operator_type) {
    xnn_log_error(
        "failed to reshape operator: operator type mismatch (expected %s, got %s)",
        xnn_operator_type_to_string(expected_operator_type),
        xnn_operator_type_to_string_v2(layer_norm_op));
    return xnn_status_invalid_parameter;
  }
  layer_norm_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_uninitialized;
  }

  const size_t channels = layer_norm_op->channels;
  if (input_stride < channels) {
    xnn_log_error(
        "failed to reshape %s operator with input element stride of %zu: stride must be at least as large as the "
        "number of channels (%zu)",
        xnn_operator_type_to_string(expected_operator_type), input_stride, channels);
    return xnn_status_invalid_parameter;
  }

  if (output_stride < channels) {
    xnn_log_error(
        "failed to reshape %s operator with output element stride of %zu: stride must be at least as large as the "
        "number of channels (%zu)",
        xnn_operator_type_to_string(expected_operator_type), output_stride, channels);
    return xnn_status_invalid_parameter;
  }

  layer_norm_op->batch_size = batch_size;
  layer_norm_op->input_pixel_stride = input_stride;
  layer_norm_op->output_pixel_stride = output_stride;

  if (batch_size == 0) {
    layer_norm_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t vector_bytes = channels << log2_element_size;
  struct layernorm_context* context = &layer_norm_op->context.layernorm;
  context->n = vector_bytes;
  context->x_stride = input_stride << log2_element_size;
  context->y_stride = output_stride << log2_element_size;
  context->scale = layer_norm_op->packed_weights.pointer;
  context->bias = (const char*) layer_norm_op->packed_weights.pointer + vector_bytes;
  context->ukernel = layer_norm_op->layernorm_config->ukernel;

  layer_norm_op->compute[0].type = xnn_parallelization_type_1d;
  layer_norm_op->compute[0].task_1d = (pthreadpool_task_1d_t) xnn_compute_layernorm;
  layer_norm_op->compute[0].range[0] = batch_size;
  layer_norm_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_reshape_layer_norm_nc_f32(
    xnn_operator_t layer_norm_op,
    size_t batch_size,
    size_t input_stride,
    size_t output_stride,
    pthreadpool_t threadpool)
{
  return reshape_layer_norm_nc(
    layer_norm_op, xnn_operator_type_layer_norm_nc_f32,
    batch_size, input_stride, output_stride, XNN_LOG2_SIZEOF_FLOAT);
}

enum xnn_status xnn_reshape_layer_norm_nc_bf16(
    xnn_operator_t layer_norm_op,
    size_t batch_size,
    size_t input_stride,
    size_t output_stride,
    pthreadpool_t threadpool)
{
  return reshape_layer_norm_nc(
    layer_norm_op, xnn_operator_type_layer_norm_nc_bf16,
    batch_size, input_stride, output_stride, XNN_LOG2_SIZEOF_BFLOAT16);
}

static enum xnn_status setup_layer_norm_nc(
    xnn_operator_t layer_norm_op,
    enum xnn_operator_type expected_operator_type,
    const void* input,
    void* output)
{
  if (layer_norm_op->type != expected_operator_type) {
    xnn_log_error(
        "failed to setup operator: operator type mismatch (expected %s, got %s)",
        xnn_operator_type_to_string(expected_operator_type),
        xnn_operator_type_to_string_v2(layer_norm_op));
    return xnn_status_invalid_parameter;
  }

  switch (layer_norm_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
          "failed to setup %s operator: operator has not been reshaped yet",
          xnn_operator_type_to_string_v2(layer_norm_op));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  layer_norm_op->context.layernorm.x = input;
  layer_norm_op->context.layernorm.y = output;
  layer_norm_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_layer_norm_nc_f32(
    xnn_operator_t layer_norm_op,
    const float* input,
    float* output)
{
  return setup_layer_norm_nc(layer_norm_op, xnn_operator_type_layer_norm_nc_f32, input, output);
}

enum xnn_status xnn_setup_layer_norm_nc_bf16(
    xnn_operator_t layer_norm_op,
    const void* input,
    void* output)
{
  return setup_layer_norm_nc(layer_norm_op, xnn_operator_type_layer_norm_nc_bf16, input, output);
}
