// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Modelled on test/operators/softmax-nc.cc.

#include <cstddef>

#include <gtest/gtest.h>
#include "test/operators/layer-norm-operator-tester.h"

TEST(LAYER_NORM_NC_F32, single_channel) {
  LayerNormOperatorTester().batch_size(1).channels(1).iterations(5).TestF32();
}

TEST(LAYER_NORM_NC_F32, many_channels) {
  for (size_t channels = 2; channels < 70; channels += 3) {
    LayerNormOperatorTester().batch_size(1).channels(channels).iterations(1).TestF32();
  }
}

TEST(LAYER_NORM_NC_F32, siglip_channels) {
  LayerNormOperatorTester().batch_size(3).channels(768).iterations(3).TestF32();
}

TEST(LAYER_NORM_NC_F32, small_batch) {
  for (size_t channels = 1; channels < 100; channels += 11) {
    LayerNormOperatorTester().batch_size(3).channels(channels).iterations(2).TestF32();
  }
}

TEST(LAYER_NORM_NC_F32, large_batch) {
  LayerNormOperatorTester().batch_size(64).channels(37).iterations(2).TestF32();
}

TEST(LAYER_NORM_NC_F32, strided_input) {
  for (size_t channels = 1; channels < 100; channels += 11) {
    LayerNormOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .iterations(2)
        .TestF32();
  }
}

TEST(LAYER_NORM_NC_F32, strided_output) {
  for (size_t channels = 1; channels < 100; channels += 11) {
    LayerNormOperatorTester()
        .batch_size(3)
        .channels(channels)
        .output_stride(117)
        .iterations(2)
        .TestF32();
  }
}

TEST(LAYER_NORM_NC_F32, epsilon) {
  for (float epsilon : {0.0f, 1e-5f, 1e-2f, 1.0f}) {
    LayerNormOperatorTester().batch_size(2).channels(50).epsilon(epsilon).iterations(2).TestF32();
  }
}

TEST(LAYER_NORM_NC_BF16, single_channel) {
  LayerNormOperatorTester().batch_size(1).channels(1).iterations(5).TestBF16();
}

TEST(LAYER_NORM_NC_BF16, many_channels) {
  for (size_t channels = 2; channels < 70; channels += 3) {
    LayerNormOperatorTester().batch_size(1).channels(channels).iterations(1).TestBF16();
  }
}

TEST(LAYER_NORM_NC_BF16, siglip_channels) {
  LayerNormOperatorTester().batch_size(3).channels(768).iterations(3).TestBF16();
}

TEST(LAYER_NORM_NC_BF16, small_batch) {
  for (size_t channels = 1; channels < 100; channels += 11) {
    LayerNormOperatorTester().batch_size(3).channels(channels).iterations(2).TestBF16();
  }
}

TEST(LAYER_NORM_NC_BF16, large_batch) {
  LayerNormOperatorTester().batch_size(64).channels(37).iterations(2).TestBF16();
}

TEST(LAYER_NORM_NC_BF16, strided_input) {
  for (size_t channels = 1; channels < 100; channels += 11) {
    LayerNormOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .iterations(2)
        .TestBF16();
  }
}

TEST(LAYER_NORM_NC_BF16, strided_output) {
  for (size_t channels = 1; channels < 100; channels += 11) {
    LayerNormOperatorTester()
        .batch_size(3)
        .channels(channels)
        .output_stride(117)
        .iterations(2)
        .TestBF16();
  }
}

TEST(LAYER_NORM_NC_BF16, epsilon) {
  for (float epsilon : {0.0f, 1e-5f, 1e-2f, 1.0f}) {
    LayerNormOperatorTester().batch_size(2).channels(50).epsilon(epsilon).iterations(2).TestBF16();
  }
}
