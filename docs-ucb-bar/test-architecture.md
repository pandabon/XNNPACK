# XNNPack test architecture

End-to-end pipeline of how XNNPack's existing gtest-based tests work,
organised by category and layer. §1–7 walk the **GEMM-microkernel**
pipeline in detail (the most elaborate one); §0 maps that flow to the
other categories so divergences are easy to find.

## 0. Test category & layer map

XNNPack tests fall into **three layers**, each catching wiring the lower
ones can't:

1. **Microkernel layer** — direct call into a kernel symbol with hand-built
   activations / packed weights. One ELF per kernel family (or per op,
   depending on category). All driven from [test/CMakeLists.txt](../test/CMakeLists.txt).
2. **Operator layer** — exercises the public C API
   (`xnn_create_*` → `xnn_reshape_*` → `xnn_setup_*`). One source file per
   operator under [test/operators/](../test/operators/) (~60 files);
   wired via `ADD_SUBDIRECTORY(operators)` at line 88. Catches
   config-dispatch / packer / setup wiring the microkernel layer can't.
3. **Subgraph layer** — exercises `xnn_define_*` + the graph runtime. One
   source file per node type under [test/subgraph/](../test/subgraph/);
   wired via `ADD_SUBDIRECTORY(subgraph)` at line 89. Catches
   validate-datatypes / dispatch-switch wiring the operator layer can't.

A given kernel is hit by **all three layers** transitively. The categories
below are a partition of the *microkernel layer only*.

### Microkernel-layer categories (7 SETs in [test/CMakeLists.txt](../test/CMakeLists.txt))

Each `SET(...)` lists family names; a `FOREACH` makes one ELF per name.
Two driver patterns are in use:

- **YAML + per-family generator → checked-in `.cc`** — one ELF per family,
  generated `.cc` instantiates one test suite per kernel variant in the
  YAML. The pattern walked through in §1–7.
- **`.inc` X-macro register → one shared driver `.cc`** — one ELF for the
  whole category, the driver loops over the union of `.inc` lists. Used
  for maxpool/avgpool today; cheaper to add a new family (drop in an
  `.inc`), heavier to add a new test case (modify shared driver).

| SET (line) | Pattern | Generator | Notes |
|------------|---------|-----------|-------|
| `MICROKERNEL_GEMM_UNIT_TESTS` (189) | YAML + gen | [tools/generate-gemm-test.py](../tools/generate-gemm-test.py) | gemm + igemm; uses `FILE(GLOB ${TEST}*.cc)` so e.g. bf16-gemm-minmax pulls in any companion `.cc`. **§1–7 walk this in detail.** |
| `MICROKERNEL_DWCONV_UNIT_TESTS` (160) | YAML + gen | [tools/generate-dwconv-test.py](../tools/generate-dwconv-test.py) | depthwise conv |
| `MICROKERNEL_VBINARY_UNIT_TESTS` (267) | YAML + gen | [tools/generate-vbinary-test.py](../tools/generate-vbinary-test.py) | one ELF per op (`f32-vadd`, `f16-vmul`, …) |
| `MICROKERNEL_VUNARY_TESTS` (376) | YAML + gen | [tools/generate-vunary-test.py](../tools/generate-vunary-test.py) | one ELF per op (`f32-vsigmoid`, `f16-vtanh`, …) |
| `MICROKERNEL_VCVT_TESTS` (339) | YAML + gen | (no dedicated generator on disk; `.cc` checked in directly) | dtype convert |
| `MICROKERNEL_PACKQ_UNIT_TESTS` (251) | hand-written `.cc` | — | quantised packers, kleidiai-dep |
| `MICROKERNEL_UNIT_TESTS` (92) | **mixed** | various | catchall: maxpool/avgpool (`.inc` X-macro shared driver), ibilinear, packw, raddstoreexpminusmax, spmm, indirection, transpose, lut, … each with its own per-op driver |

Where `MICROKERNEL_UNIT_TESTS` uses `.inc` (maxpool, avgpool): the driver
`.cc` (e.g. [test/maxpool-minmax.cc](../test/maxpool-minmax.cc))
`#include`s every per-family `.inc` (e.g.
`src/{f16,f32,s8,u8,bf16}-maxpool/<family>-minmax.inc`) under a local
`XNN_UKERNEL` macro that expands each line into a `XnnTestParam` row.
Adding a new family = drop a `.inc` and add one `#include` to the driver.
No YAML, no generator. Per-family build-option / arch gating goes inside
the `.inc` via `#if XNN_ARCH_* && XNN_ENABLE_*`.

### Cross-cutting microkernel infra

- [test/gemm-microkernel-tester.{h,cc}](../test/gemm-microkernel-tester.h)
  — the GEMM `Test()` overloads (one per ukernel-fn-pointer typedef);
  see §3.
- Per-family tester headers next to the family driver (e.g.
  [test/maxpool-microkernel-tester.h](../test/maxpool-microkernel-tester.h),
  [test/dwconv-microkernel-tester.h](../test/dwconv-microkernel-tester.h),
  [test/conv-hwc-microkernel-tester.h](../test/conv-hwc-microkernel-tester.h)).
- [test/replicable_random_device.h](../test/replicable_random_device.h),
  [test/buffer.h](../test/buffer.h), [test/next_prime.h](../test/next_prime.h)
  — shared utilities used across all categories.
- [src/xnnpack/isa-checks.h](../src/xnnpack/isa-checks.h) —
  `TEST_REQUIRES_ARCH_FLAGS` runtime arch-gate macro (§7).

### Layer-2 / Layer-3 drivers

Operator and subgraph tests each have their own `*-operator-tester.h` /
`*-subgraph-tester.h` fixture per node type, e.g.
[test/operators/convolution-operator-tester.h](../test/operators/convolution-operator-tester.h)
and [test/subgraph/subgraph-tester.h](../test/subgraph/subgraph-tester.h).
These are out of scope for the GEMM-specific walkthrough below but follow
the same gtest fixture conventions.

## 1. Spec files (YAML) — one entry per kernel variant

Each microkernel family has a YAML spec under [test/](../test/):

- [test/bf16-f32-gemm-minmax.yaml](../test/bf16-f32-gemm-minmax.yaml)
- [test/bf16-gemm-minmax.yaml](../test/bf16-gemm-minmax.yaml)
- [test/f32-gemm-minmax.yaml](../test/f32-gemm-minmax.yaml) (etc.)

Each entry has 4 fields:

```yaml
- name: xnn_bf16_f32_gemm_minmax_ukernel_4x4v__rvv_zvfbfa
  init: xnn_init_f32_minmax_scalar_params
  pack: xnn_pack_bf16_f32_gemm_goi_w
  k-block: 1
```

The kernel `name` encodes everything:
`xnn_<datatype>_<ukernel-type>_<activation>_ukernel_<MR>x<NR>[c<KR>][s<SR>][v]__<arch>_<isa>`.

`v` = vector-tile (NR scaled by VLEN at runtime); `c<KR>` = channel-tile
size; `s<SR>` = subgroup. Parsed by `split_ukernel_name()` in
[tools/generate-gemm-test.py:43](../tools/generate-gemm-test.py#L43).

## 2. Generator (Python) — emits the `.cc` per spec

[tools/generate-gemm-test.py](../tools/generate-gemm-test.py) reads each
YAML and emits one C++ file (e.g. `test/bf16-f32-gemm-minmax.cc`).
Invocation:

```sh
python3 tools/generate-gemm-test.py \
  -s test/bf16-f32-gemm-minmax.yaml \
  -o test/bf16-f32-gemm-minmax.cc
```

Flags: `-s/--spec`, `-o/--output-test`, optional `-b/--output-bench`. The
generated `.cc` is checked in.

For each YAML entry it produces an `INSTANTIATE_TEST_SUITE_P` block:

```cpp
INSTANTIATE_TEST_SUITE_P(
    BF16_F32_GEMM_MINMAX_4X4V__RVV_ZVFBFA, GemmTest,
    testing::ValuesIn(CreateTestsN(
        /*k_block=*/1, /*adj_k_block=*/1,
        /*mr=*/4, /*nr=*/4 * vlenb / sizeof(float),  // NR scaled at runtime
        /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false, /*unsigned_inputs=*/false, /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_bf16_f32_gemm_minmax_ukernel_4x4v__rvv_zvfbfa,
                      xnn_init_f32_minmax_scalar_params,
                      xnn_pack_bf16_f32_gemm_goi_w);
        },
        /*arch_flags=*/xnn_arch_riscv_zvfbfa)),  // runtime gate
    [](const auto& info) { return info.param.test_name; });
```

Wrapped in `#if XNN_ENABLE_RISCV_ZVFBFA && XNN_ARCH_RISCV` (build-time gate).

The mapping from kernel-name suffix (e.g. `__rvv_zvfbfa`) to compile-time
macro and runtime arch flag lives in
[tools/xnncommon.py](../tools/xnncommon.py):

- `_ISA_TO_MACRO_MAP`: ISA token → `XNN_ENABLE_*` build macro
- `_ISA_TO_ARCH_MAP`: ISA token → list of arch macros
- `_ISA_TO_ARCH_FLAGS_MAP`: ISA token → `xnn_arch_*` runtime flag
- `_ISA_HIERARCHY`: ordering for benchmark output

Adding a new ISA suffix (e.g. `zvfbfmin`/`zvfbfa`) means inserting an entry
in all four. `parse_target_name()` walks the suffix tokens left-to-right;
the last-matching ISA token wins.

The generator also has an `accum_type` lookup keyed by `input_datatype` —
needs an entry for any new input type (e.g. `"bf16": "float"`), and an
`nr_scale` lookup keyed by ISA — needs entries mirroring `rvv` for any
RISC-V vector-tile ISA. There's a special-case wrap `if isa == "rvv":`
that adds another arch-guard `#if` around the `CreateTestsN` helper —
needs to be widened to include any new RISC-V ISA suffix.

## 3. The tester class (the heart of it)

[test/gemm-microkernel-tester.h](../test/gemm-microkernel-tester.h)
defines `GemmMicrokernelTester` — a builder-pattern fixture with all the
sweep knobs:

```cpp
class GemmMicrokernelTester {
  size_t mr_, nr_, kr_, sr_, m_, n_, k_;
  size_t a_stride_, cm_stride_, cn_stride_;
  uint8_t qmin_ = 0, qmax_ = 255;
  // ~30 setters: .mr(), .nr(), .k(), .a_stride(), etc.

  // Per-ukernel-signature overloads (one per fn-pointer type), e.g.:
  void Test(xnn_bf16_f32_gemm_minmax_ukernel_fn ukernel,
            xnn_init_f32_minmax_params_fn init_params,
            xnn_pack_bf16_f32_gemm_fn pack) const;

  void Test(xnn_bf16_gemm_minmax_ukernel_fn ukernel,
            xnn_init_bf16_minmax_params_fn init_params,
            xnn_pack_f16_gemm_fn pack) const;
};
```

Each `Test()` overload at
[test/gemm-microkernel-tester.cc:3204](../test/gemm-microkernel-tester.cc#L3204)
does, for `iterations()` repeats:

1. Generate random `a[m × k]`, `b[k × n]`, `bias[n]` (BF16 inputs;
   FP32 bias for `bf16_f32`, BF16 for `bf16`).
2. Call `pack(g=1, n, k, nr, kr, sr, b, bias, scale=nullptr, packed_w,
   extra_bytes=0, params=nullptr)` to lay out weights+bias the way the
   ukernel expects.
3. Compute the FP32 reference:
   `c_ref[i,j] = bias[j] + Σ_k bf16_to_f32(a[i,k]) * bf16_to_f32(b[k,j])`,
   then `c_ref = clamp(c_ref, min, max)`. For `bf16` (BF16 output), the
   reference is rounded to BF16 on writeback, then re-widened for the
   compare.
4. Call the ukernel:
   `ukernel(m, n, k * sizeof(uint16_t), a, a_stride * sizeof(uint16_t),
   packed_w, c, cm_stride * sizeof(out_elem), cn_stride, &params)`.
5. `ASSERT_NEAR(c[i,j], c_ref[i,j], max(1e-4, 3e-2 * |c_ref[i,j]|))` —
   3% relative tolerance with a 1e-4 absolute floor.

Packed buffer sizes:

- `bf16_f32`: `packed_n() * packed_k() + packed_n() * sizeof(float)` bytes
  (one FP32 bias per output channel).
- `bf16`: `packed_n() * packed_k() + packed_n() * sizeof(uint16_t)` bytes
  (one BF16 bias per output channel).
- `packed_n() = round_up(n, nr)`, `packed_k() = round_up_po2(k, kr*sr)`.

Init-params structs ([src/xnnpack/microparams.h](../src/xnnpack/microparams.h))
are trivial:

```cpp
struct xnn_f32_minmax_params  { struct { float min; float max; } scalar; };
struct xnn_bf16_minmax_params { struct { float min; float max; } scalar; };
```

## 4. The parametric sweep

The `CreateTestsN(...)` helper in the generated `.cc` is shared by every
`INSTANTIATE_TEST_SUITE_P` of the same shape. It builds a
`vector<GemmTestParams>` covering ~15 categories per kernel:

| Category                      | What it varies                                |
|-------------------------------|-----------------------------------------------|
| `k_eq_kbs`                    | exact k=k_block, m=mr, n=nr (1 case)          |
| `k_eq_kbs_strided_a`          | + a_stride = NextPrime(k+1) (GEMM only)       |
| `k_eq_kbs_subtile`            | loops m∈[1,mr], n∈[1,nr]                      |
| `k_eq_kbs_subtile_m`          | k=k_block, n=nr, loop m                       |
| `k_eq_kbs_subtile_n`          | k=k_block, m=mr, loop n                       |
| `k_lt_akbs_subtile`           | k∈[1, adj_k_block) × m × n                    |
| `k_gt_akbs[, _subtile]`       | k∈[adj+1, 2*adj) (× m × n)                    |
| `k_div_kbs[, _subtile]`       | k stepped by k_block (× m × n)                |
| `n_gt_nr[, _subtile]`         | n past nr boundary                            |
| `n_div_nr[, _subtile]`        | n at multiples of nr                          |
| `strided_cm[, _subtile]`      | cm_stride = NextPrime(nr+1)                   |
| `min`, `max`                  | minmax-clamp boundary tests                   |

Total per kernel variant: **~3000 sub-tests**. Across the four BF16
RISC-V families (1x4v / 4x4v × Zvfbfmin / Zvfbfa): ~12k sub-tests.

## 5. Build wiring

[test/CMakeLists.txt](../test/CMakeLists.txt) lists test families in
`MICROKERNEL_GEMM_UNIT_TESTS`; a `FOREACH` produces one ELF per family:

```cmake
ADD_EXECUTABLE(${TEST}-test ${TEST_SOURCES})  # bf16-gemm-minmax-test, etc.
TARGET_LINK_LIBRARIES(${TEST}-test PRIVATE
    gemm-microkernel-tester
    xnnpack-microkernels-all
    xnnpack-test          # interface lib bundling gtest+gmock+XNNPACK
    pthreadpool ...)
ADD_TEST(NAME ${TEST}-test COMMAND ${TEST}-test)
```

`xnnpack-test` (line 48) is an INTERFACE library bundling `GTest::gtest`,
`GTest::gtest_main`, and `XNNPACK`. Tests use gtest's auto-`main()` from
`gtest_main`.

**Note:** `bf16-f32-gemm-minmax` is missing from the CMake list (only
present in Bazel); add to `MICROKERNEL_GEMM_UNIT_TESTS` to wire it for
CMake builds.

## 6. Running

`ctest` invokes each ELF. For cross-compiles, `CMAKE_CROSSCOMPILING_EMULATOR`
(set by [cmake/riscv64.toolchain](../cmake/riscv64.toolchain) to
`qemu-riscv64;<options>`) wraps the invocation:

```
qemu-riscv64 -cpu rv64,v=true,vlen=512,zba=true,... \
  -L /usr/riscv64-linux-gnu \
  ./bf16-gemm-minmax-test
```

The ELF is a normal Linux glibc binary running gtest's test-runner main.
QEMU `-cpu` flags must enable the extensions the kernels need (e.g.
`zvfbfmin=true,zvfbfa=true` if those are tested) — verify with
`qemu-riscv64 -cpu help` since property names drift between QEMU versions.

## 7. Runtime arch gating (the skip path)

[src/xnnpack/isa-checks.h](../src/xnnpack/isa-checks.h) provides:

```cpp
#define TEST_REQUIRES_ARCH_FLAGS(FLAGS)                       \
  do {                                                        \
    const auto* hw = xnn_init_hardware_config();              \
    if (!hw || (hw->arch_flags & (FLAGS)) != (FLAGS))         \
      GTEST_SKIP();                                           \
  } while (0)
```

Each generated test invokes `TEST_REQUIRES_ARCH_FLAGS(arch_flags)` first.
If the simulator/CPU doesn't expose the bit, the test cleanly skips
instead of crashing on an illegal instruction. `xnn_init_hardware_config()`
populates `arch_flags` from cpuinfo / hwprobe / etc.; under QEMU and
spike, the bits are inferred from the `-cpu` / `--isa` configuration.

## Adding a new kernel family — checklist

1. Author the kernel `.c` in `src/<family>/gen/<name>.c` (use `__asm__`
   not bare `asm`; XNNPACK pins `-std=c99` strictly).
2. Add an `XNNPACK_ENABLE_RISCV_<EXT>` build option in
   [CMakeLists.txt](../CMakeLists.txt), propagate to `XNN_ENABLE_RISCV_<EXT>`
   compile def.
3. Reserve an `xnn_arch_riscv_<ext>` bit in
   [src/xnnpack/hardware-config.h](../src/xnnpack/hardware-config.h);
   add cpuinfo/hwprobe detection.
4. Wire into [src/configs/gemm-config.c](../src/configs/gemm-config.c)
   for operator-level dispatch.
5. Add `__rvv_<ext>` mapping to all four maps in
   [tools/xnncommon.py](../tools/xnncommon.py).
6. If the generator doesn't recognize a new datatype, add an `accum_type`
   entry; if it's a vector-tile ISA, add an `nr_scale` entry; widen the
   `if isa == "rvv":` guard if the new ISA needs the same `CreateTests`
   wrap.
7. Append entries to the appropriate `test/<family>.yaml`.
8. Re-run `python3 tools/generate-gemm-test.py -s … -o …` to regenerate
   the `.cc`.
9. Add the family to `MICROKERNEL_GEMM_UNIT_TESTS` in
   [test/CMakeLists.txt](../test/CMakeLists.txt) if not already present.
10. Build + run via the cross toolchain.

## Substituting the runtime (e.g. spike for qemu)

Steps 1–5 and 7 are runtime-agnostic. Only step 6 changes:

- Replace `cmake/riscv64.toolchain` with one that points compiler tools at
  the desired toolchain root (env var, e.g. `RISCV_TOOLCHAIN_ROOT`) and
  emits `CMAKE_CROSSCOMPILING_EMULATOR=spike;--isa=…` instead of
  `qemu-riscv64;…`.
- For bare-metal newlib targets (e.g. Zephyr SDK), additionally need an
  HTIF runtime (`crt.S` + `syscalls.c` + linker script) — there's a
  reusable one at
  `/home/cc/cs152/sp26/class/cs152-acs/lab4/benchmarks/common/` — and
  small upstream-aligned patches to gate `#include <pthread.h>` on
  `XNN_HAS_PTHREADS` in 3 files (`src/xnnpack/{init-once,mutex}.h`,
  `src/mutex.c`).
- pthreadpool needs a one-line sed-patch to use its `shim.c`
  single-threaded path under `CMAKE_SYSTEM_NAME=Generic` (its upstream
  CMakeLists only enables shim under `EMSCRIPTEN`).
