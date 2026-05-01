# Running XNNPack tests under spike (bare-metal HTIF)

Companion to [test-architecture.md](test-architecture.md), which covers the
upstream YAML → generator → gtest pipeline. This doc covers the
**runtime-substitution layer**: cross-compile XNNPack's existing gtest
binaries with the Zephyr SDK `riscv64-zephyr-elf-gcc` and run them under
spike using a self-contained HTIF runtime in [cmake/htif/](../cmake/htif/) —
no qemu, no glibc, no `pk` proxy kernel, no external lab/runtime sources.

## Required environment

| Variable             | Purpose                                                                       |
|----------------------|-------------------------------------------------------------------------------|
| `RISCV_TOOLCHAIN_ROOT` | Zephyr-SDK-style toolchain root (its `bin/` has `riscv64-zephyr-elf-gcc`).  |
| `SPIKE_BIN` (opt)    | spike binary (default: `spike` from `PATH`).                                  |
| `SPIKE_ISA` (opt)    | `--isa=` string (default: `rv64gcv_zicntr_zfbfmin_zvfbfmin_zvfbfa`).          |

Spike itself needs its dynamic-link dependencies. The boost-1.85 / icu-75 /
recent libstdc++ build available via conda:

```sh
conda create -n spike-build -c conda-forge boost=1.85 icu=75 libstdcxx-ng dtc -y
conda activate spike-build
```

`dtc` is needed because spike calls it on startup; the conda env provides
all four (boost_regex, boost_system, icudata, libstdc++) on `LD_LIBRARY_PATH`.

## One-time CMake configure

```sh
source <conda-base>/etc/profile.d/conda.sh
conda activate spike-build
export RISCV_TOOLCHAIN_ROOT=/path/to/zephyr-sdk/gnu/riscv64-zephyr-elf

cmake -S . -B build/spike \
  -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-zephyr-elf-spike.toolchain \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DXNNPACK_LIBRARY_TYPE=static \
  -DXNNPACK_BUILD_BENCHMARKS=OFF \
  -DXNNPACK_BUILD_TESTS=ON \
  -DXNNPACK_ENABLE_RISCV_VECTOR=ON \
  -DXNNPACK_ENABLE_RISCV_ZVFBFMIN=ON \
  -DXNNPACK_ENABLE_RISCV_ZVFBFA=ON \
  -Dgtest_disable_pthreads=ON
```

## pthreadpool sed-patch (one-time after first configure)

XNNPack pulls pthreadpool via FetchContent. Pthreadpool's upstream
`CMakeLists.txt` only takes the single-threaded `shim.c` path under
`EMSCRIPTEN`, and only links `Threads::Threads` when the system isn't
Emscripten. Our toolchain sets `CMAKE_SYSTEM_NAME=Generic`. Patch the
fetched source to extend both gates:

```sh
sed -i \
  -e 's|IF(EMSCRIPTEN)|IF(EMSCRIPTEN OR CMAKE_SYSTEM_NAME STREQUAL "Generic")|' \
  -e 's|IF(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")|IF(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Generic")|' \
  build/spike/pthreadpool-source/CMakeLists.txt

cmake -B build/spike   # re-configure to pick the patched CMakeLists up
```

The patch is idempotent across rebuilds; just re-run it after wiping
`build/spike/`.

## Build

```sh
cmake --build build/spike -j --target \
  bf16-f32-gemm-minmax-test \
  bf16-gemm-minmax-test \
  x16-x32-packw-test \
  f32-gemm-minmax-test
```

Each test ELF is statically linked, ~6–9 MB, with `_start` at `0x80000000`
(see [cmake/htif/link.ld](../cmake/htif/link.ld)). The infra is
datatype-agnostic — any RVV-emitting microkernel test target in
[test/CMakeLists.txt](../test/CMakeLists.txt) builds the same way; the
four above are the ones currently smoke-tested.

## Run

`spike` must be on `PATH` (or `SPIKE_BIN` set), and the conda env's
`bin/`/`lib/` must be on `PATH`/`LD_LIBRARY_PATH` so spike finds its
boost/icu/dtc deps. A typical session-prelude:

```sh
export PATH=$HOME/.rvt/bin:<conda-env>/bin:$PATH
export LD_LIBRARY_PATH=<conda-env>/lib:$LD_LIBRARY_PATH
```

There are three ways to run the binaries; pick the one that matches the
task.

### 1. One binary, one VLEN — fastest iteration

```sh
spike --isa=rv64gcv_zicntr_zfbfmin_zvfbfmin_zvfbfa_zvl128b \
  build/spike/test/f32-gemm-minmax-test
```

Filter to a slice with `--gtest_filter='*RVV*'` for quick smoke checks
on suites with thousands of cases (`f32-gemm-minmax-test`,
`f32-igemm-minmax-test`, ...).

### 2. ctest — multiple binaries at one VLEN

```sh
ctest --test-dir build/spike -V -R 'bf16.*(zvfbfmin|zvfbfa)'
```

`CMAKE_CROSSCOMPILING_EMULATOR` in the toolchain wires ctest to
`spike --isa=…` automatically. To change VLEN, set `SPIKE_VLEN` and
re-configure (the toolchain appends `_zvl${N}b` to `SPIKE_ISA`):

```sh
SPIKE_VLEN=512 cmake -B build/spike   # re-bakes CMAKE_CROSSCOMPILING_EMULATOR
ctest --test-dir build/spike -V -R 'bf16.*(zvfbfmin|zvfbfa)'
```

### 3. VLEN sweep on one binary

The kernels use `vsetvli` at runtime so binaries are VLEN-agnostic — one
build runs at any VLEN by appending `_zvl<N>b` to the `--isa=` string.
Spike's VLEN floor is 32 (it rejects `_zvl16b`).

```sh
TEST=build/spike/test/bf16-f32-gemm-minmax-test
for vlen in 32 64 128 256 512 1024; do
  echo "=== VLEN=$vlen ==="
  spike --isa=rv64gcv_zicntr_zfbfmin_zvfbfmin_zvfbfa_zvl${vlen}b $TEST
done
```

Verified working as of the last edit: all four targets above pass at
VLEN ∈ {32, 64, 128, 256, 512, 1024}.

### Fractional-LMUL coverage

Two complementary regimes are exercised by the test pipeline:

**Short-vector regime in the bf16 gemm kernels.** The bf16 RVV gemm
kernels use only `e32, m4` (FP32 accumulator) and `e16, m2` (BF16
widening) and don't emit fractional-LMUL vtypes themselves. But the
*short-vector* paths fractional LMULs would exercise — the `vl < vlmax`
regime — are reachable today and are worth covering as a distinct test
surface. The kernel bug fixed in `cab66200ad` (`c0 += cn_stride`
advancing past the actually-stored elements) only surfaced when
`nc % NR != 0`, i.e. exactly this regime. It decomposes into two axes:

1. **Inner-loop tail (`vl < vlmax_e32m4` in the last iteration).**
   `GemmMicrokernelTester::Test()` parametrically sweeps `n` from 1 up to
   ~2·NR, so every iteration where `n % NR != 0` runs the loop body
   under the `if XNN_UNLIKELY(nc < nr)` branch with `vl ∈ {1..NR-1}`.
2. **Small VLMAX (NR shrinks).** The VLEN sweep above bottoms out at
   VLEN=32, where `vlmax_e32m4 = 4` — the smallest NR the m4 kernels
   can be driven to. (Spike rejects `_zvl16b`, so 32 is the floor.)

The product covers `vl ∈ {1..NR_at_VLEN}` for each VLEN ∈ {32, 64, 128,
256, 512, 1024}.

**Genuine `mf*` emission in the x16-x32 packer.** The restored segmented
`xnn_x16_x32_packw_gemm_goi_ukernel_x4v__rvv_u8` (commit `c4d08926f9`)
uses `vlsseg{4,8}e16` with `_e16m1` / `_e16m2` typed intrinsics. EMUL=1
gives gcc latitude to coalesce `vsetvli` to the equivalent `e8, mf2`
encoding, which it does — confirmed both statically (`mf2` immediates in
the packer .obj) and dynamically (31 mf2 executions in a spike trace at
the packer body PC). `x16-x32-packw-test` is the canonical test target
for this.

Two one-time validations cement the regimes, in addition to the test
sweeps themselves:

```sh
# (a) Static: list all SEW/LMUL combinations the bf16 RVV objects emit.
mkdir -p /tmp/xnn_disasm && cd /tmp/xnn_disasm
$RISCV_TOOLCHAIN_ROOT/bin/riscv64-zephyr-elf-ar x \
  $XNN/build/spike/libxnnpack-microkernels-prod.a
for o in bf16*rvv*.obj x16-x32-packw*rvv*.obj; do
  $RISCV_TOOLCHAIN_ROOT/bin/riscv64-zephyr-elf-objdump -d "$o" \
    | grep -E 'vsetvli|vsetivli'
done | grep -Eo 'm[f]?[0-9]' | sort -u
# Expected: m1, m2, m4 from the gemm kernels; mf2 from the packer
# (gcc coalesces vlsseg{4,8}e16 EMUL=1 to e8, mf2).

# (b) Dynamic: confirm mf2 actually executes at runtime in the packer.
spike -l --isa=rv64gcv_zicntr_zfbfmin_zvfbfmin_zvfbfa_zvl128b \
  $XNN/build/spike/test/x16-x32-packw-test \
  --gtest_filter='*k_eq_kblock*' 2>trace.log
grep -cE 'vsetvli.*e8, *mf2' trace.log
# Expected: a positive count (we observed 31 hits).

# (c) Dynamic: confirm vl < vlmax executes in the bf16 gemm kernels.
spike -l --isa=rv64gcv_zicntr_zfbfmin_zvfbfmin_zvfbfa_zvl128b \
  $XNN/build/spike/test/bf16-f32-gemm-minmax-test \
  --gtest_filter='*ZVFBFMIN*n_eq_1*' 2>trace.log
grep -E 'vsetvli.*e32, *m4' trace.log | head
# Expected: at least one line with vl < 16 (since VLEN=128 → vlmax=16).
```

## What the substitution does

The toolchain at [cmake/riscv64-zephyr-elf-spike.toolchain](../cmake/riscv64-zephyr-elf-spike.toolchain)
+ project hook at [cmake/riscv64-zephyr-elf-spike-project-hook.cmake](../cmake/riscv64-zephyr-elf-spike-project-hook.cmake)
inject [cmake/htif/](../cmake/htif/) as a static `htif-runtime` library
linked into every executable. The runtime has three pieces:

- [cmake/htif/crt.S](../cmake/htif/crt.S) — `_start`: zero GPRs, enable
  FP+V state in `mstatus`, set `gp`/`sp`/`tp`, zero `.bss`, run
  `__libc_init_array` (so C++ static ctors fire), call `main(0,0,0)`,
  run `__libc_fini_array`, write `(rc << 1) | 1` to `tohost`. Trap
  vector points at a small handler that writes `(1337 << 1) | 1` to
  `tohost` for any unexpected exception.
- [cmake/htif/syscalls.c](../cmake/htif/syscalls.c) — POSIX I/O over the
  HTIF magic-buffer protocol: `write`/`_write` write to fd 1 via spike's
  console syscall (`tohost = &{which=64, fd, buf, len}`); `read`,
  `lseek`, `close`, `open`, `fstat`, `kill`, `getpid` are stubbed; `_exit`
  / `exit` / `abort` write `(code<<1)|1` to `tohost`; `_sbrk` bumps a
  pointer in the linker-defined `[__heap_start, __heap_end)` 64 MiB
  region; `stdin`/`stdout`/`stderr` are picolibc-tinystdio FILE structs
  whose `put` callback routes through `write`.
- [cmake/htif/link.ld](../cmake/htif/link.ld) — places `.text.init` at
  `0x80000000`, `.tohost` (with `tohost`/`fromhost` symbols) at the next
  page, `.text`, `.data`, `.bss`, then a per-binary 1 MiB stack and
  64 MiB heap. **Strong** (not `PROVIDE`) definitions of
  `__bothinit_array_start/_end` (which picolibc weak-references) so C++
  ctors actually run.

## Required upstream-aligned XNNPack patches

Several XNNPack source files unconditionally reach for pthread / POSIX
time symbols that newlib doesn't provide. The patches mirror the existing
`XNN_PLATFORM_WEB && !defined(__EMSCRIPTEN_PTHREADS__)` no-pthread
pattern:

| File                           | Change                                                                       |
|--------------------------------|------------------------------------------------------------------------------|
| `src/xnnpack/common.h`         | Detect `XNN_HAS_PTHREADS` via `__has_include(<pthread.h>)`.                  |
| `src/xnnpack/init-once.h`      | Gate `#include <pthread.h>` on `XNN_HAS_PTHREADS`.                           |
| `src/xnnpack/mutex.h`          | Add `!XNN_HAS_PTHREADS` no-mutex struct branch.                              |
| `src/mutex.c`                  | Add `XNN_HAS_PTHREADS` to the existing pthread-using `#elif`.                |
| `src/runtime.c`                | Gate `clock_gettime` on `XNN_HAS_PTHREADS`; return zeros otherwise.          |
| `src/xnnpack/subgraph.h`       | Add `XNN_HAS_PTHREADS` branch to `xnn_timestamp` typedef.                    |

These can be committed on the branch alongside the test-infrastructure
additions in `tools/` and `test/`.

## Compile-time / link-time gotchas baked into the toolchain

These are all set by [cmake/riscv64-zephyr-elf-spike.toolchain](../cmake/riscv64-zephyr-elf-spike.toolchain)
and shouldn't need user action, but listed for awareness:

- `-march=rv64gcv_zvfbfmin -mabi=lp64d -mcmodel=medany`. Zvfbfa is
  reached via hand-emitted `.insn i 0x57, 0x7, …, 0x1C9` in source
  (gcc 14.3 doesn't accept `zvfbfa` as a `-march` extension yet).
- `-fno-builtin-printf -fno-builtin-puts` so gcc doesn't transform
  `printf("hello\n")` into a call to `puts()` that drags in tinystdio's
  `stdout` buffering before our HTIF FILE shim is wired up.
- `_POSIX_C_SOURCE=200809L` and `__POSIX_VISIBLE=200809L` to expose
  `localtime_r` etc. from newlib (libstdc++ chrono uses them).
- `GTEST_HAS_PTHREAD=0`, `GTEST_HAS_FILE_SYSTEM=0`,
  `GTEST_HAS_STREAM_REDIRECTION=0`. Don't `=0`-define
  `GTEST_HAS_DEATH_TEST` — gtest derives it from
  `GTEST_HAS_FILE_SYSTEM`, and `#ifdef`-checks are sensitive to a `=0`
  definition.
- `PATH_MAX=4096` so gtest's `gtest-filepath.cc` doesn't fall through to
  the undefined `_POSIX_PATH_MAX`.
- `CMAKE_THREAD_LIBS_INIT=""` plus a pre-defined empty
  `Threads::Threads` interface library in the project hook, so
  XNNPack's `find_package(Threads)` doesn't request `-lpthreads`.
- `INSTALL_GTEST=OFF` to suppress googletest's install-export rules
  (they trip on the propagated `htif-runtime` link dep).

## Verification

A successful run:

1. `cmake --build build/spike --target bf16-f32-gemm-minmax-test
   bf16-gemm-minmax-test` produces two ELFs, no link errors.
2. `spike --isa=rv64gcv_zicntr_zfbfmin_zvfbfmin_zvfbfa
   build/spike/test/bf16-f32-gemm-minmax-test` prints gtest banner +
   per-test PASS/FAIL output via HTIF console, runs every parametrized
   instantiation in `BF16_F32_GEMM_MINMAX_*X4V__RVV_ZVFBF{MIN,A}` plus
   the scalar baseline `BF16_F32_GEMM_MINMAX_1X4C2__SCALAR`, and exits
   with status 0 on full PASS.
3. Same for `bf16-gemm-minmax-test`.
4. The Zvfbfa altfmt vsetvli round-trips through the toolchain
   correctly. From the disassembly of any `xnn_bf16_*_gemm_minmax_ukernel_*__rvv_zvfbfa`
   function:
   ```
   vsetvli xN, xM, 457    # 457 = 0x1C9 = altfmt=1, vma, vta, vsew=e16, vlmul=m2
   vfwmacc.vf v4, fX, v2  # widening BF16 FMA under altfmt=1
   ```

## Pack function choice for the new RISC-V kernels

Both `bf16_f32_gemm` and `bf16_gemm` RISC-V variants
(`__rvv_zvfbfmin` / `__rvv_zvfbfa`) read **FP32** bias from the packed
weights buffer (matching the operator-level `xnn_x16_x32_packw_x4v__rvv_u8`
runtime packer). The microkernel test must therefore use
`xnn_pack_bf16_f32_gemm_goi_w` for both YAML families — even the
BF16-output `bf16_gemm` entries. (Existing NEON BF16 kernels use BF16
bias and stay on `xnn_pack_f16_gemm_goi_w`.)

## Known caveats

- Death tests, file-system tests, and pthread-dependent gtest features
  are compiled out — only the GEMM microkernel tests are intended to
  run.
