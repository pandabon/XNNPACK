// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Modelled on test/operators/softmax-operator-tester.h.

#ifndef XNNPACK_TEST_OPERATORS_LAYER_NORM_OPERATOR_TESTER_H_
#define XNNPACK_TEST_OPERATORS_LAYER_NORM_OPERATOR_TESTER_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/math.h"
#include "test/replicable_random_device.h"

class LayerNormOperatorTester {
 public:
  LayerNormOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const { return this->channels_; }

  LayerNormOperatorTester& input_stride(size_t input_stride) {
    assert(input_stride != 0);
    this->input_stride_ = input_stride;
    return *this;
  }

  size_t input_stride() const {
    return this->input_stride_ == 0 ? this->channels_ : this->input_stride_;
  }

  LayerNormOperatorTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  size_t output_stride() const {
    return this->output_stride_ == 0 ? this->channels_ : this->output_stride_;
  }

  LayerNormOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const { return this->batch_size_; }

  LayerNormOperatorTester& epsilon(float epsilon) {
    this->epsilon_ = epsilon;
    return *this;
  }

  float epsilon() const { return this->epsilon_; }

  LayerNormOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const { return this->iterations_; }

  void TestF32() const { Test<float>(); }
  void TestBF16() const { Test<xnn_bfloat16>(); }

 private:
  static float ToFloat(float v) { return v; }
  static float ToFloat(xnn_bfloat16 v) { return xnn_bfloat16_to_float(v); }
  static void FromFloat(float v, float& out) { out = v; }
  static void FromFloat(float v, xnn_bfloat16& out) {
    out = xnn_bfloat16_from_float(v);
  }

  static xnn_status Create(size_t channels, float epsilon, const float* scale,
                           const float* bias, xnn_operator_t* op) {
    return xnn_create_layer_norm_nc_f32(channels, epsilon, scale, bias, 0, op);
  }
  static xnn_status Create(size_t channels, float epsilon,
                           const xnn_bfloat16* scale, const xnn_bfloat16* bias,
                           xnn_operator_t* op) {
    return xnn_create_layer_norm_nc_bf16(channels, epsilon, scale, bias, 0, op);
  }
  xnn_status Reshape(xnn_operator_t op, float*) const {
    return xnn_reshape_layer_norm_nc_f32(op, batch_size(), input_stride(),
                                         output_stride(), nullptr);
  }
  xnn_status Reshape(xnn_operator_t op, xnn_bfloat16*) const {
    return xnn_reshape_layer_norm_nc_bf16(op, batch_size(), input_stride(),
                                          output_stride(), nullptr);
  }
  static xnn_status Setup(xnn_operator_t op, const float* in, float* out) {
    return xnn_setup_layer_norm_nc_f32(op, in, out);
  }
  static xnn_status Setup(xnn_operator_t op, const xnn_bfloat16* in,
                          xnn_bfloat16* out) {
    return xnn_setup_layer_norm_nc_bf16(op, in, out);
  }

  template <typename T>
  void Test() const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> input_dist(-4.0f, 4.0f);
    std::uniform_real_distribution<float> scale_dist(0.5f, 2.0f);
    std::uniform_real_distribution<float> bias_dist(-1.0f, 1.0f);

    xnnpack::Buffer<T> input((batch_size() - 1) * input_stride() + channels(),
                             xnnpack::XnnExtraBytes);
    xnnpack::Buffer<T> output((batch_size() - 1) * output_stride() + channels(),
                              xnnpack::XnnExtraBytes);
    xnnpack::Buffer<T> scale(channels(), xnnpack::XnnExtraBytes);
    xnnpack::Buffer<T> bias(channels(), xnnpack::XnnExtraBytes);
    xnnpack::Buffer<double> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      for (size_t i = 0; i < input.size(); i++) {
        FromFloat(input_dist(rng), input[i]);
      }
      for (size_t c = 0; c < channels(); c++) {
        FromFloat(scale_dist(rng), scale[c]);
        FromFloat(bias_dist(rng), bias[c]);
      }

      // Reference, in double.
      for (size_t i = 0; i < batch_size(); i++) {
        double sum = 0.0;
        for (size_t c = 0; c < channels(); c++) {
          sum += ToFloat(input[i * input_stride() + c]);
        }
        const double mean = sum / channels();
        double sqsum = 0.0;
        for (size_t c = 0; c < channels(); c++) {
          const double d = ToFloat(input[i * input_stride() + c]) - mean;
          sqsum += d * d;
        }
        const double inv_std = 1.0 / std::sqrt(sqsum / channels() + epsilon());
        for (size_t c = 0; c < channels(); c++) {
          output_ref[i * channels() + c] =
              (ToFloat(input[i * input_stride() + c]) - mean) * inv_std *
                  ToFloat(scale[c]) +
              ToFloat(bias[c]);
        }
      }

      // Create, reshape, setup, run and destroy the operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t layer_norm_op = nullptr;
      ASSERT_EQ(xnn_status_success,
                Create(channels(), epsilon(), scale.data(), bias.data(),
                       &layer_norm_op));
      ASSERT_NE(nullptr, layer_norm_op);
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
          auto_layer_norm_op(layer_norm_op, xnn_delete_operator);

      // The operator copied scale and bias: scribbling over ours must not matter.
      std::fill(scale.begin(), scale.end(), T());
      std::fill(bias.begin(), bias.end(), T());

      ASSERT_EQ(xnn_status_success, Reshape(layer_norm_op, (T*)nullptr));
      ASSERT_EQ(xnn_status_success,
                Setup(layer_norm_op, input.data(), output.data()));
      ASSERT_EQ(xnn_status_success,
                xnn_run_operator(layer_norm_op, /*threadpool=*/nullptr));

      // Verify results: f32 within a few ulps of the sums' order, bf16 within one bf16 ulp.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const double expected = output_ref[i * channels() + c];
          // A single channel is a constant row (variance 0, so a = 1 / sqrt(epsilon) is large): the kernels compute
          // y = (x * a + b) * scale + bias, and the rounding of b = -mean * a then shows up in y. Allow for it.
          const double constant_row = channels() == 1 ? 1e-3 : 0.0;
          const double tolerance =
              ((sizeof(T) == 4) ? 2e-5 * std::abs(expected) + 2e-5
                                : std::ldexp(std::abs(expected), -7) + 1e-4) +
              constant_row;
          ASSERT_NEAR(ToFloat(output[i * output_stride() + c]), expected,
                      tolerance)
              << "row " << i << " / " << batch_size() << ", channel " << c
              << " / " << channels();
        }
      }
    }
  }

  size_t channels_ = 1;
  size_t input_stride_ = 0;
  size_t output_stride_ = 0;
  size_t batch_size_ = 1;
  float epsilon_ = 1e-6f;
  size_t iterations_ = 15;
};

#endif  // XNNPACK_TEST_OPERATORS_LAYER_NORM_OPERATOR_TESTER_H_
