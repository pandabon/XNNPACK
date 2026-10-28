// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Tests of the f32 and bf16 LAYERNORM microkernels (one row per call). The reference is computed in double:
//   y = (x - mean) / sqrt(var + epsilon) * scale + bias, with the biased variance of the row.
// Structure follows test/rsum.cc.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"  // IWYU pragma: keep
#include "src/xnnpack/hardware-config.h"  // IWYU pragma: keep
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/layernorm.h"  // IWYU pragma: keep
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"  // IWYU pragma: keep
#include "test/replicable_random_device.h"

namespace {

float ToFloat(float v) { return v; }
float ToFloat(xnn_bfloat16 v) { return xnn_bfloat16_to_float(v); }
template <typename T> T FromFloat(float v);
template <> float FromFloat<float>(float v) { return v; }
template <> xnn_bfloat16 FromFloat<xnn_bfloat16>(float v) {
  return xnn_bfloat16_from_float(v);
}

struct Kernel;

class Tester {
 public:
  Tester& channels(size_t channels) {
    channels_ = channels;
    return *this;
  }
  Tester& epsilon(float epsilon) {
    epsilon_ = epsilon;
    return *this;
  }
  // Rows are mean + uniform(-spread, spread) * stddev scale.
  Tester& mean(float mean) {
    mean_ = mean;
    return *this;
  }
  Tester& spread(float spread) {
    spread_ = spread;
    return *this;
  }
  // Extra absolute tolerance. The kernels compute y = (x * a + b) * scale + bias with b = -mean * a (as ncnn and
  // PyTorch's CPU kernel do), which loses digits when |mean| is much larger than the standard deviation.
  Tester& atol(float atol) {
    atol_ = atol;
    return *this;
  }
  Tester& inplace(bool inplace) {
    inplace_ = inplace;
    return *this;
  }

  template <typename T, typename Params, typename InitParams>
  void Test(void (*ukernel)(size_t, const T*, const T*, const T*, T*,
                            const Params*),
            InitParams init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> input_dist(mean_ - spread_,
                                                     mean_ + spread_);
    std::uniform_real_distribution<float> scale_dist(0.5f, 2.0f);
    std::uniform_real_distribution<float> bias_dist(-1.0f, 1.0f);

    xnnpack::Buffer<T> input(channels_, xnnpack::XnnExtraBytes);
    xnnpack::Buffer<T> scale(channels_, xnnpack::XnnExtraBytes);
    xnnpack::Buffer<T> bias(channels_, xnnpack::XnnExtraBytes);
    xnnpack::Buffer<T> output(channels_, xnnpack::XnnExtraBytes);
    for (size_t i = 0; i < channels_; i++) {
      input[i] = FromFloat<T>(input_dist(rng));
      scale[i] = FromFloat<T>(scale_dist(rng));
      bias[i] = FromFloat<T>(bias_dist(rng));
    }

    // Reference, in double, from the (already rounded) inputs.
    double sum = 0.0;
    for (size_t i = 0; i < channels_; i++) sum += ToFloat(input[i]);
    const double mean = sum / channels_;
    double sqsum = 0.0;
    for (size_t i = 0; i < channels_; i++) {
      const double d = ToFloat(input[i]) - mean;
      sqsum += d * d;
    }
    const double inv_std = 1.0 / std::sqrt(sqsum / channels_ + epsilon_);
    std::vector<double> expected(channels_);
    for (size_t i = 0; i < channels_; i++) {
      expected[i] = (ToFloat(input[i]) - mean) * inv_std * ToFloat(scale[i]) +
                    ToFloat(bias[i]);
    }

    Params params;
    init_params(&params, epsilon_);

    const T* x = input.data();
    T* y = output.data();
    if (inplace_) {
      y = input.data();
    }
    ukernel(channels_ * sizeof(T), x, scale.data(), bias.data(), y, &params);

    for (size_t i = 0; i < channels_; i++) {
      // f32: sums in a different order than the reference, so a few ulps.
      // bf16: one rounding to bf16 on store (half an ulp), and the f32 statistics can move the
      // value across a rounding boundary: allow one full bf16 ulp (2^-7 relative).
      // A single channel is a constant row (variance 0, a = 1 / sqrt(epsilon) is large): see atol().
      const double constant_row = channels_ == 1 ? 1e-3 : 0.0;
      const double tolerance = ((sizeof(T) == 4) ? 2e-5 * std::abs(expected[i]) + 2e-5
                                                 : std::ldexp(std::abs(expected[i]), -7) + 1e-4) +
                               atol_ + constant_row;
      ASSERT_NEAR(expected[i], ToFloat(y[i]), tolerance)
          << "channel " << i << " of " << channels_;
    }
  }

  void Test(const Kernel& kernel) const;

 private:
  size_t channels_ = 1;
  float epsilon_ = 1e-6f;
  float mean_ = 0.0f;
  float spread_ = 4.0f;
  float atol_ = 0.0f;
  bool inplace_ = false;
};

struct Kernel {
  explicit Kernel(xnn_f32_layernorm_ukernel_fn fn,
                  xnn_init_f32_layernorm_params_fn init_params) {
    dispatch = [=](const Tester& tester) { tester.Test(fn, init_params); };
  }
  explicit Kernel(xnn_bf16_layernorm_ukernel_fn fn,
                  xnn_init_bf16_layernorm_params_fn init_params) {
    dispatch = [=](const Tester& tester) { tester.Test(fn, init_params); };
  }
  std::function<void(const Tester&)> dispatch;
};

void Tester::Test(const Kernel& kernel) const { kernel.dispatch(*this); }

struct KernelInfo {
  const char* name;
  uint64_t arch_flags;
  Kernel kernel;
  size_t batch_tile;
  bool vector_tile;
  size_t elem_size;
};

KernelInfo kernels[] = {
#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype, \
                    params_type, init_params)                               \
  {#ukernel,   arch_flags,  Kernel{ukernel, init_params},                   \
   batch_tile, vector_tile, sizeof(datatype)},
#include "src/bf16-layernorm/bf16-layernorm.inc"
#include "src/f32-layernorm/f32-layernorm.inc"
#undef XNN_UKERNEL
};

class Test : public testing::TestWithParam<KernelInfo> {};

INSTANTIATE_TEST_SUITE_P(
    layernorm, Test, testing::ValuesIn(kernels),
    [](const testing::TestParamInfo<Test::ParamType>& info) {
      return info.param.name;
    });

// Row lengths: every length 1..batch_tile * 2 + 1 (full register groups and tails), and lengths around
// 2 and 3 register groups plus the sizes real models use (768 = SigLIP, 960 = SmolVLM connector).
std::vector<size_t> RowLengths(const KernelInfo& param) {
  const size_t tile = param.batch_tile *
                      (param.vector_tile ? get_batch_scale(param.elem_size) * param.elem_size / 4 : 1);
  std::vector<size_t> lengths;
  for (size_t c = 1; c <= 2 * tile + 1; c++) lengths.push_back(c);
  for (size_t c : {3 * tile - 1, 3 * tile, 3 * tile + 1, size_t{768}, size_t{960}, size_t{1000}}) {
    lengths.push_back(c);
  }
  return lengths;
}

TEST_P(Test, channels) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  for (size_t c : RowLengths(param)) {
    Tester().channels(c).Test(param.kernel);
  }
}

TEST_P(Test, inplace) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  for (size_t c : {size_t{1}, size_t{7}, size_t{64}, size_t{768}, size_t{1001}}) {
    Tester().channels(c).inplace(true).Test(param.kernel);
  }
}

TEST_P(Test, epsilon) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  for (float epsilon : {0.0f, 1e-12f, 1e-6f, 1e-5f, 0.1f, 10.0f}) {
    Tester().channels(97).epsilon(epsilon).Test(param.kernel);
  }
}

// A large common offset (mean / stddev about 170): the variance needs the two-pass (centered) sum to stay accurate.
TEST_P(Test, large_mean) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester().channels(768).mean(100.0f).spread(1.0f).atol(5e-4f).Test(param.kernel);
}

TEST_P(Test, small_values) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester().channels(257).spread(1e-3f).epsilon(1e-6f).Test(param.kernel);
}

TEST_P(Test, large_values) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester().channels(257).spread(1000.0f).Test(param.kernel);
}

}  // namespace
