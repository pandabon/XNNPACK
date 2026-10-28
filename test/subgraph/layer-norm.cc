// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Modelled on test/subgraph/softmax.cc.

#include <chrono>
#include <cmath>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

namespace {

float ToFloat(float v) { return v; }
float ToFloat(xnn_bfloat16 v) { return xnn_bfloat16_to_float(v); }
template <typename T> T FromFloat(float v);
template <> float FromFloat<float>(float v) { return v; }
template <> xnn_bfloat16 FromFloat<xnn_bfloat16>(float v) {
  return xnn_bfloat16_from_float(v);
}

}  // namespace

// Reference in double: y = (x - mean) / sqrt(var + epsilon) * scale + bias over the last dimension.
template <typename T>
Tensor<T> layer_norm(const Tensor<T>& x, const std::vector<T>& scale,
                     const std::vector<T>& bias, float epsilon) {
  Tensor<T> y(x.extents());
  std::vector<size_t> batch_dims = x.extents();
  const size_t channels = x.extents().back();
  batch_dims.pop_back();
  for (std::vector<size_t> i : EnumerateIndices(batch_dims)) {
    i.push_back(0);
    const T* x_i = &x(i);
    T* y_i = &y(i);
    double sum = 0.0;
    for (size_t c = 0; c < channels; c++) sum += ToFloat(x_i[c]);
    const double mean = sum / channels;
    double sqsum = 0.0;
    for (size_t c = 0; c < channels; c++) {
      const double d = ToFloat(x_i[c]) - mean;
      sqsum += d * d;
    }
    const double inv_std = 1.0 / std::sqrt(sqsum / channels + epsilon);
    for (size_t c = 0; c < channels; c++) {
      y_i[c] = FromFloat<T>((ToFloat(x_i[c]) - mean) * inv_std * ToFloat(scale[c]) + ToFloat(bias[c]));
    }
  }
  return y;
}

template <typename T>
void TestImpl(size_t rank) {
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  const size_t channels = 24;
  const float epsilon = 1e-6f;
  std::vector<T> scale(channels), bias(channels);
  for (size_t c = 0; c < channels; c++) {
    scale[c] = FromFloat<T>(0.5f + 0.1f * c);
    bias[c] = FromFloat<T>(0.05f * c - 0.5f);
  }

  // Define subgraph: external input 0, external output 1, static scale 2 and bias 3.
  SubgraphTester subgraph(4);
  subgraph.AddInputTensor(rank, xnn_datatype_of<T>(), 0)
      .AddOutputTensor(rank, xnn_datatype_of<T>(), 1)
      .AddStaticTensor(TensorShape({channels}), 2, scale.data())
      .AddStaticTensor(TensorShape({channels}), 3, bias.data())
      .AddLayerNorm(epsilon, 0, 2, 3, 1);
  xnn_status status = subgraph.CreateRuntime();
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
    return;
  }

  // At most 12 iterations: on a bare-metal target without a clock (Spike) the time limit never expires.
  for (auto _ : FuzzTest(std::chrono::milliseconds(500), /*min_iters=*/1, /*max_iters=*/12)) {
    std::vector<size_t> shape = random_shape(rng, rank);
    shape.back() = channels;

    Tensor<T> input(shape, xnnpack::XnnExtraBytes);
    DatatypeGenerator<T> generator(-4.0f, 4.0f);
    input.generate([&]() { return generator(rng); });

    Tensor<T> expected = layer_norm(input, scale, bias, epsilon);

    // Check reshaped shape is correct
    subgraph.ReshapeExternalTensor(shape, input.base(), 0).ReshapeRuntime();
    ASSERT_EQ(subgraph.GetExternalTensorShape(1), expected.extents());

    // Run subgraph
    Tensor<T> output(expected.extents(), xnnpack::XnnExtraBytes);
    subgraph.SetupExternalTensor(output.base(), 1)
        .SetupRuntime()
        .InvokeRuntime();

    // Verify results: bf16 within one bf16 ulp (relative 2^-7), f32 within a few ulps.
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
      const float e = ToFloat(expected.data()[i]);
      const float tolerance = sizeof(T) == 2 ? std::ldexp(std::abs(e), -7) + 1e-4f : 2e-5f * std::abs(e) + 2e-5f;
      ASSERT_NEAR(ToFloat(output.data()[i]), e, tolerance) << "element " << i;
    }
  }
}

template <typename T>
class LayerNorm : public ::testing::TestWithParam<int> {};

using LayerNormBF16 = LayerNorm<xnn_bfloat16>;
using LayerNormF32 = LayerNorm<float>;

TEST_P(LayerNormBF16, test) { TestImpl<xnn_bfloat16>(GetParam()); }
TEST_P(LayerNormF32, test) { TestImpl<float>(GetParam()); }

auto rank_params = testing::Range(1, XNN_MAX_TENSOR_DIMS);
INSTANTIATE_TEST_SUITE_P(LayerNorm, LayerNormBF16, rank_params);
INSTANTIATE_TEST_SUITE_P(LayerNorm, LayerNormF32, rank_params);

TEST(LayerNorm, reshape_rejects_scalar_input) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  std::vector<float> scale(4, 1.0f), bias(4, 0.0f);
  SubgraphTester subgraph(4);
  subgraph.AddInputTensor(TensorShape(), xnn_datatype_fp32, 0)
      .AddOutputTensor(1, xnn_datatype_fp32, 1)
      .AddStaticTensor(TensorShape({4}), 2, scale.data())
      .AddStaticTensor(TensorShape({4}), 3, bias.data())
      .AddLayerNorm(1e-6f, 0, 2, 3, 1);
  if (subgraph.CreateRuntime() == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  float data = 0.0f;
  subgraph.ReshapeExternalTensor(TensorShape(), &data, 0).ReshapeRuntime();
  EXPECT_EQ(subgraph.Status(), xnn_status_invalid_parameter);
}

TEST(LayerNorm, reshape_rejects_last_dim_mismatch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  std::vector<float> scale(4, 1.0f), bias(4, 0.0f);
  SubgraphTester subgraph(4);
  subgraph.AddInputTensor(2, xnn_datatype_fp32, 0)
      .AddOutputTensor(2, xnn_datatype_fp32, 1)
      .AddStaticTensor(TensorShape({4}), 2, scale.data())
      .AddStaticTensor(TensorShape({4}), 3, bias.data())
      .AddLayerNorm(1e-6f, 0, 2, 3, 1);
  if (subgraph.CreateRuntime() == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  // The scale has 4 elements, the reshaped input has 5 channels.
  std::vector<float> data(10, 0.0f);
  subgraph.ReshapeExternalTensor({2, 5}, data.data(), 0).ReshapeRuntime();
  EXPECT_EQ(subgraph.Status(), xnn_status_invalid_parameter);
}

TEST(LayerNorm, define_rejects_bad_arguments) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  std::vector<float> scale(4, 1.0f), bias(4, 0.0f);
  std::vector<float> short_scale(3, 1.0f);
  {
    // Input fp32, output bf16.
    SubgraphTester subgraph(4);
    subgraph.AddInputTensor(1, xnn_datatype_fp32, 0)
        .AddOutputTensor(1, xnn_datatype_bf16, 1)
        .AddStaticTensor(TensorShape({4}), 2, scale.data())
        .AddStaticTensor(TensorShape({4}), 3, bias.data());
    EXPECT_EQ(xnn_define_layer_norm(subgraph.Subgraph(), 1e-6f, 0, 2, 3, 1, /*flags=*/0),
              xnn_status_invalid_parameter);
  }
  {
    // Scale that is not a static tensor (dynamic).
    SubgraphTester subgraph(4);
    subgraph.AddInputTensor(1, xnn_datatype_fp32, 0)
        .AddOutputTensor(1, xnn_datatype_fp32, 1)
        .AddInputTensor(1, xnn_datatype_fp32, 2)
        .AddStaticTensor(TensorShape({4}), 3, bias.data());
    EXPECT_EQ(xnn_define_layer_norm(subgraph.Subgraph(), 1e-6f, 0, 2, 3, 1, /*flags=*/0),
              xnn_status_invalid_parameter);
  }
  {
    // Negative epsilon.
    SubgraphTester subgraph(4);
    subgraph.AddInputTensor(1, xnn_datatype_fp32, 0)
        .AddOutputTensor(1, xnn_datatype_fp32, 1)
        .AddStaticTensor(TensorShape({4}), 2, scale.data())
        .AddStaticTensor(TensorShape({4}), 3, bias.data());
    EXPECT_EQ(xnn_define_layer_norm(subgraph.Subgraph(), -1.0f, 0, 2, 3, 1, /*flags=*/0),
              xnn_status_invalid_parameter);
  }
  {
    // Static input with a known last dimension that differs from the scale size.
    SubgraphTester subgraph(4);
    subgraph.AddInputTensor(TensorShape({2, 5}), xnn_datatype_fp32, 0)
        .AddOutputTensor(2, xnn_datatype_fp32, 1)
        .AddStaticTensor(TensorShape({4}), 2, scale.data())
        .AddStaticTensor(TensorShape({4}), 3, bias.data());
    EXPECT_EQ(xnn_define_layer_norm(subgraph.Subgraph(), 1e-6f, 0, 2, 3, 1, /*flags=*/0),
              xnn_status_invalid_parameter);
  }
}

}  // namespace xnnpack
