// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/layernorm.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"

static struct xnn_layernorm_config bf16_layernorm_config = {0};
static struct xnn_layernorm_config f32_layernorm_config = {0};

XNN_INIT_ONCE_GUARD(bf16_layernorm);
XNN_INIT_ONCE_GUARD(f32_layernorm);

// Macro to log the microkernel name if and when it is registered.
#define XNN_INIT_LAYERNORM_UKERNEL(ukernel)        \
  (xnn_layernorm_ukernel_fn) ukernel;              \
  xnn_log_info("Using layernorm microkernel '%s'.", #ukernel);

static void init_bf16_layernorm_config(void) {
  bf16_layernorm_config.init.bf16 = xnn_init_bf16_layernorm_scalar_params;
  bf16_layernorm_config.ukernel =
      XNN_INIT_LAYERNORM_UKERNEL(xnn_bf16_layernorm_ukernel__scalar_u1);
  #if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    // Widen-to-f32 RVV kernel (plain RVV 1.0, no Zvfbf*); the scalar kernel above stays as the fallback.
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->arch_flags & xnn_arch_riscv_vector) {
      bf16_layernorm_config.ukernel =
          XNN_INIT_LAYERNORM_UKERNEL(xnn_bf16_layernorm_ukernel__rvv_u8v);
    }
  #endif
}

static void init_f32_layernorm_config(void) {
  f32_layernorm_config.init.f32 = xnn_init_f32_layernorm_scalar_params;
  f32_layernorm_config.ukernel =
      XNN_INIT_LAYERNORM_UKERNEL(xnn_f32_layernorm_ukernel__scalar_u1);
  #if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->arch_flags & xnn_arch_riscv_vector) {
      f32_layernorm_config.ukernel =
          XNN_INIT_LAYERNORM_UKERNEL(xnn_f32_layernorm_ukernel__rvv_u8v);
    }
  #endif
}

const struct xnn_layernorm_config* xnn_init_bf16_layernorm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(bf16_layernorm);
  return &bf16_layernorm_config;
}

const struct xnn_layernorm_config* xnn_init_f32_layernorm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_layernorm);
  return &f32_layernorm_config;
}
