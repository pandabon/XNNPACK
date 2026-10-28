// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Layer Normalization subgraph node (last dimension, static scale and bias).
// Modelled on src/subgraph/softmax.c.

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator-type.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/reshape-helpers.h"
#include "src/xnnpack/subgraph-validation.h"
#include "src/xnnpack/subgraph.h"
#include <pthreadpool.h>

static enum xnn_status create_layer_norm_operator(
  const struct xnn_node* node,
  const struct xnn_runtime_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  xnn_weights_cache_t weights_cache)
{
  assert(node->num_inputs == 3);
  assert(node->num_outputs == 1);

  const uint32_t input_id = node->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);
  const struct xnn_runtime_value* input_value = &values[input_id];

  const uint32_t scale_id = node->inputs[1];
  assert(scale_id != XNN_INVALID_VALUE_ID);
  assert(scale_id < num_values);
  const struct xnn_runtime_value* scale_value = &values[scale_id];

  const uint32_t bias_id = node->inputs[2];
  assert(bias_id != XNN_INVALID_VALUE_ID);
  assert(bias_id < num_values);
  const struct xnn_runtime_value* bias_value = &values[bias_id];

  // The fp32 fallback rewrite may have turned bf16 inputs into fp32 values together with their static data: use
  // the datatype of the values as they are now, not as they were defined.
  if (scale_value->datatype != input_value->datatype || bias_value->datatype != input_value->datatype ||
      scale_value->data == NULL || bias_value->data == NULL) {
    xnn_log_error(
      "failed to create %s operator: scale (ID #%" PRIu32 ") and bias (ID #%" PRIu32 ") must be static tensors of "
      "the input datatype %s",
      xnn_node_type_to_string(xnn_node_type_layer_norm), scale_id, bias_id,
      xnn_datatype_to_string(input_value->datatype));
    return xnn_status_invalid_parameter;
  }

  const size_t channels = scale_value->shape.dim[0];

  enum xnn_status status;
  switch (input_value->datatype) {
    case xnn_datatype_fp32:
      status = xnn_create_layer_norm_nc_f32(
        channels, node->params.layer_norm.epsilon,
        (const float*) scale_value->data, (const float*) bias_value->data,
        node->flags, &opdata->operator_objects[0]);
      break;
    case xnn_datatype_bf16:
      status = xnn_create_layer_norm_nc_bf16(
        channels, node->params.layer_norm.epsilon,
        scale_value->data, bias_value->data,
        node->flags, &opdata->operator_objects[0]);
      break;
    default:
      XNN_UNREACHABLE;
  }
  return status;
}

static enum xnn_status reshape_layer_norm_operator(
  struct xnn_operator_data* opdata,
  struct xnn_runtime_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id < num_values);

  const size_t num_input_dims = values[input_id].shape.num_dims;
  if (num_input_dims == 0) {
    xnn_log_error(
      "failed to reshape %s operator with input ID #%" PRIu32
      ": number of dimensions (%zu) must be at least 1",
      xnn_node_type_to_string(xnn_node_type_layer_norm), input_id, num_input_dims);
    return xnn_status_invalid_parameter;
  }
  const size_t channel_dim = values[input_id].shape.dim[num_input_dims - 1];
  const size_t batch_size = xnn_shape_multiply_non_channel_dims(&values[input_id].shape);

  const uint32_t scale_id = opdata->inputs[1];
  assert(scale_id < num_values);
  if (channel_dim != values[scale_id].shape.dim[0]) {
    xnn_log_error(
      "failed to reshape %s operator with input ID #%" PRIu32
      ": the last dimension (%zu) must match the size of the scale and bias vectors (%zu)",
      xnn_node_type_to_string(xnn_node_type_layer_norm), input_id, channel_dim, values[scale_id].shape.dim[0]);
    return xnn_status_invalid_parameter;
  }

  const size_t old_workspace_size = opdata->workspace_size;
  enum xnn_status status = xnn_status_invalid_state;
  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_layer_norm_nc_f32:
      status = xnn_reshape_layer_norm_nc_f32(
        opdata->operator_objects[0],
        batch_size,
        /*input_stride=*/channel_dim,
        /*output_stride=*/channel_dim,
        threadpool);
      break;
    case xnn_operator_type_layer_norm_nc_bf16:
      status = xnn_reshape_layer_norm_nc_bf16(
        opdata->operator_objects[0],
        batch_size,
        /*input_stride=*/channel_dim,
        /*output_stride=*/channel_dim,
        threadpool);
      break;
    default:
      XNN_UNREACHABLE;
  }
  if (status != xnn_status_success) {
    return status;
  }
  return resize_unary_elementwise_output_tensor(opdata, values, num_values, old_workspace_size, threadpool);
}

static enum xnn_status setup_layer_norm_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_runtime_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const void* input_data = values[input_id].data;
  assert(input_data != NULL);

  void* output_data = values[output_id].data;
  assert(output_data != NULL);

  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_layer_norm_nc_f32:
      return xnn_setup_layer_norm_nc_f32(opdata->operator_objects[0], input_data, output_data);
    case xnn_operator_type_layer_norm_nc_bf16:
      return xnn_setup_layer_norm_nc_bf16(opdata->operator_objects[0], input_data, output_data);
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_define_layer_norm(
  xnn_subgraph_t subgraph,
  float epsilon,
  uint32_t input_id,
  uint32_t scale_id,
  uint32_t bias_id,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_layer_norm)) != xnn_status_success) {
    return status;
  }

  if (!(epsilon >= 0.0f) || !isfinite(epsilon)) {
    xnn_log_error(
      "failed to define %s operator with %.7g epsilon: epsilon must be finite and non-negative",
      xnn_node_type_to_string(xnn_node_type_layer_norm), epsilon);
    return xnn_status_invalid_parameter;
  }

  if ((status = xnn_subgraph_check_input_node_id(xnn_node_type_layer_norm, input_id, subgraph->num_values)) !=
      xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_layer_norm, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input_value->datatype) {
    case xnn_datatype_bf16:
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_layer_norm), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  // Scale (gamma) and bias (beta): static one-dimensional tensors of the input datatype.
  const uint32_t vector_ids[2] = {scale_id, bias_id};
  const char* vector_names[2] = {"scale", "bias"};
  for (size_t i = 0; i < 2; i++) {
    if (vector_ids[i] >= subgraph->num_values) {
      xnn_log_error(
        "failed to define %s operator with %s ID #%" PRIu32 ": invalid Value ID, the subgraph has %" PRIu32 " Values",
        xnn_node_type_to_string(xnn_node_type_layer_norm), vector_names[i], vector_ids[i], subgraph->num_values);
      return xnn_status_invalid_parameter;
    }
    const struct xnn_value* vector_value = &subgraph->values[vector_ids[i]];
    if (vector_value->type != xnn_value_type_dense_tensor || vector_value->datatype != input_value->datatype ||
        !xnn_value_is_static(vector_value->allocation_type) || vector_value->data == NULL ||
        vector_value->shape.num_dims != 1) {
      xnn_log_error(
        "failed to define %s operator with %s ID #%" PRIu32 ": it must be a static one-dimensional dense tensor "
        "of the input datatype %s",
        xnn_node_type_to_string(xnn_node_type_layer_norm), vector_names[i], vector_ids[i],
        xnn_datatype_to_string(input_value->datatype));
      return xnn_status_invalid_parameter;
    }
    // A rank-0 input or a dynamic last dimension (0 here) is checked when the runtime is reshaped.
    const size_t input_channels =
        input_value->shape.num_dims == 0 ? 0 : input_value->shape.dim[input_value->shape.num_dims - 1];
    if (input_channels != 0 && input_channels != vector_value->shape.dim[0]) {
      xnn_log_error(
        "failed to define %s operator with %s ID #%" PRIu32 ": its size (%zu) must match the last dimension of "
        "the input (%zu)",
        xnn_node_type_to_string(xnn_node_type_layer_norm), vector_names[i], vector_ids[i],
        vector_value->shape.dim[0], input_channels);
      return xnn_status_invalid_parameter;
    }
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_layer_norm, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_layer_norm, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_datatype_matches(
    xnn_node_type_layer_norm, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_layer_norm;
  node->params.layer_norm.epsilon = epsilon;
  node->num_inputs = 3;
  node->inputs[0] = input_id;
  node->inputs[1] = scale_id;
  node->inputs[2] = bias_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_layer_norm_operator;
  node->reshape = reshape_layer_norm_operator;
  node->setup = setup_layer_norm_operator;

  return xnn_status_success;
}
