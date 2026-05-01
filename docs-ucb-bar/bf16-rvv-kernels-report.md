# RVV BF16 kernels — design report

Companion to [bf16-rvv-fix-report.md](bf16-rvv-fix-report.md) (correctness pass) and [spike-runtime.md](spike-runtime.md) (build / run harness). This document covers the kernel-authoring side: what was added, why the shapes and instruction selections look the way they do, and what was tricky about the toolchain.

Branch: `torch-1.0.0-bump-bf16`. Phase 1 commits: `4e7bafb6..1077af6f` (bf16 gemm). Phase 2 commits: `ce803b45..f4b99f2b` (bf16-bf16-bf16 FC + bf16 igemm + bf16 maxpool + conv2d / maxpool operator-subgraph stack + tests, scoped to "make full LeNet bf16 delegate"). Phase 3 follow-ups (executorch-side workaround for the bf16 depthwise gap; per-tensor type-accept switch lesson) covered in §Known follow-ups.

## Scope

### Phase 1 — gemm

Three kernel families × two output dtypes × two MR shapes, plus shared infrastructure:

| Family | Inner FMA | bf16 → bf16 (`bf16-gemm`) | bf16 → fp32 (`bf16-f32-gemm`) |
|--------|-----------|----------------------------|--------------------------------|
| Zvfbfmin only        | widen `vfwcvtbf16` + fp32 `vfmacc.vf`    | 1x4v, 4x4v | 1x4v, 4x4v |
| Zvfbfa only          | `vfwmacc.vf` under altfmt=1 vtype        | 1x4v, 4x4v | 1x4v, 4x4v |
| Zvfbfmin + Zvfbfa    | `vfwmacc.vf` under altfmt=1 + `vfncvtbf16.f.f.w` narrow | 1x4v, 4x4v | (not built — see below) |

Shared:
- `x16-x32-packw-x4v-gemm-goi-rvv-u8` — packw kernel (FP32 bias slot, BF16 column-major weights), used by all three families.
- `init_bf16_gemm_config` / `init_bf16_f32_gemm_config` priority ladder, two new build options `XNNPACK_ENABLE_RISCV_ZVFBFMIN` / `_ZVFBFA` (default OFF), two new arch-flag bits, and `xnn_nanbox_bf16` helper in `math.h`.

Kernel shape across all three families: NR = `vsetvlmax_e32m4()` (= 16 at VLEN=128, 32 at VLEN=256), MR ∈ {1, 4}, fp32 accumulators at LMUL=m4. This matches the LMUL convention of the in-tree f32 / qd8-f32-qc8w / qs8-qc8w RVV GEMM kernels. Register budget for 4x4v is 4·4 + 6 = 22 of 32 vregs, leaving room for future KC unrolling.

The combined-extension family is `bf16-gemm` only because the `bf16-f32-gemm` Zvfbfa kernel is already a single `vfwmacc.vf` per K step — there is no widening to fold away, so the combined variant would be identical code.

### Phase 2 — igemm + maxpool + operator/subgraph stack

Adds an indirect-GEMM family (for conv2d) and a maxpool family, plus the operator and subgraph plumbing that lets the partitioner accept bf16 conv2d / max_pool2d nodes:

| Family | Inner op | bf16 → bf16 | bf16 → fp32 |
|--------|----------|-------------|-------------|
| `bf16-igemm`     | `vfwmacc.vf` altfmt + `vfncvtbf16` narrow | 1x4v, 4x4v (combined zvfbfmin+zvfbfa only) | — |
| `bf16-f32-igemm` | `vfwmacc.vf` altfmt (no narrow)            | — | 1x4v, 4x4v (combined only) |
| `bf16-maxpool`   | widen → base-V `vfmax.vv` (f32) → narrow   | 9p × `u1v` (Zvfbfmin only; ladder-symmetric `-zvfbfmin-zvfbfa` filename) | n/a |

Shared additions:
- New reference packers `xnn_pack_bf16_f32_conv_{goki,kgo}_w` produce the same FP32-bias / BF16-column-major layout the gemm packer does, but for the 4D conv weight shape (g × oc × ks × kc).
- `init_bf16_gemm_config` extended on the both-flags rung with `igemm[]` slots + `pack_igemm_{goki,kgo}` (same packed-weights layout as gemm; the operator dispatches to igemm when `ks > 1`).
- `init_bf16_f32_gemm_config` gains a new top-rung "both flags" arm with combined igemm + the existing zvfbfa gemm symbols (combined gemm would be byte-identical so isn't emitted).
- New `init_bf16_maxpool_config()` + `xnn_maxpool_config.init.bf16` slot in `config-types.h`.
- 3 new operators (`xnn_create_convolution2d_nhwc_{bf16,bf16_f32}`, `xnn_create_max_pooling2d_nhwc_bf16`) with matching `_reshape_*` / `_setup_*`, public-API decls, and operator-type enum entries.
- Subgraph wiring: `convolution-2d.c` validate fns accept bf16 (with FP32 bias to match packw layout); dispatch routes `output=bf16` to `_bf16` and `output=fp32` to `_bf16_f32`. `max-pooling-2d.c` switch gains a bf16 case.

## Phase 1 per-stage breakdown

### `4e7bafb6` patch hardware config with vsetvli

Drive-by RVV-init fix. `xnn_init_hardware_config` was reading `vlenb` without ensuring vtype/vl were established; added a `vsetvli` to put the harness into a defined state before the `csrr` so that `vlenb` reads are reproducible across spike + native runs.

### `e507fe3b` Zvfbfmin/Zvfbfa hardware detection and config plumbing

- Reserved two arch-flag bits (`xnn_arch_riscv_zvfbfmin`, `xnn_arch_riscv_zvfbfa`) on the RISC-V branch.
- Added build options `XNNPACK_ENABLE_RISCV_ZVFBFMIN` / `_ZVFBFA` and the `XNN_ENABLE_RISCV_*` macros they expand to.
- Extended `gemm-config.c` with an empty `bf16` arm of the init union and `xnn_init_bf16_gemm_config` accessor symmetric to the existing `bf16_f32` one. RISC-V branches in both `init_bf16_*_gemm_config` were stubbed (mr=0 ⇒ scalar fallback) until later commits register kernels.

### `acb9a0b4` packw + Zvfbfmin `bf16-f32-gemm` (1x4v, 4x4v)

The Zvfbfmin BF16→FP32 widen lives in a self-contained inline-asm block that opens with its own `vsetvli e16, m2, ta, ma` so that the widen is correct regardless of the surrounding `e32m4` vtype the compiler-emitted intrinsics leave behind. Outside that block, all clamp/store work is base-V intrinsics on `f32m4`.

The packw kernel layout (FP32 bias, then KC × NR BF16 column-major) is shared across all six follow-on kernels and is the single source of truth that the kernels read against.

### `776e3f38` Zvfbfmin `bf16-gemm` (1x4v, 4x4v)

Same skeleton as `acb9a0b4`, with an added `vfncvtbf16.f.f.w` narrow at the store epilogue (under its own `e16, m2` `vsetvli`). Narrow uses the dynamic rounding mode `frm`; XNNPACK leaves `frm = RNE` process-wide, which matches the BF16-spec rounding, so no explicit `fsrm` is issued.

`init_bf16_gemm_config` becomes non-stub at this commit and `xnn_init_bf16_gemm_config` is exposed to the operator layer.

### `3e8c2e43` Zvfbfa `bf16-f32-gemm` (1x4v, 4x4v)

`vfwmacc.vf` under altfmt=1 vtype: BF16 × BF16 → FP32 in one instruction per KC step, ~4 vregs cheaper than the Zvfbfmin widen-then-macc sequence and one fewer dependency chain.

**Toolchain workaround.** The Zephyr binutils in use does not recognise the `zvfbfa` extension name or the `e16alt` SEW token, so the altfmt-enabling vsetvli is hand-emitted as

```
.insn i 0x57, 0x7, %[vlout], %[avl], 0x1C9  // vsetvli e16, m2, ta, ma, altfmt
```

Immediate `0x1C9` is per Zvfbfa v0.1: bit 8 = altfmt, bit 7 = vma, bit 6 = vta, bits[5:3] = vsew=001 (SEW=16), bits[2:0] = vlmul=001 (m2). `vfwmacc.vf` itself uses the standard RVV 1.0 encoding — Zvfbfa is silent on encoding changes and routes altfmt entirely through vtype, so only the `vsetvli` needs the `.insn` form.

The scalar `.vf` operand is the activation; Zvfbfa altfmt=1 expects a NaN-boxed BF16 in an f-register (upper FLEN-16 bits all 1s). Added `xnn_nanbox_bf16(uint16_t)` to `src/xnnpack/math.h` returning `double` so the inline-asm `"f"` constraint picks an f-register without the compiler trying to round-trip through fp32.

### `fdd0b38b` Zvfbfa `bf16-gemm` (1x4v, 4x4v)

Mirrors `3e8c2e43` plus a `vfncvt.f.f.w` narrow under altfmt=1 to land back in BF16. Same `.insn i ... 0x1C9` workaround for the altfmt vsetvli; both `vfwmacc.vf` and `vfncvt.f.f.w` use standard RVV 1.0 encodings.

`init_bf16_gemm_config` / `init_bf16_f32_gemm_config` now prefer the Zvfbfa branch over the Zvfbfmin fallback.

### `809a5160` `__asm__` instead of `asm`

gcc 14.3 rejects bare `asm` under `-std=c99`. Mechanical rename in all six bf16 kernels to match in-tree convention.

### `cab66200` output-stride + stale-rewind fix

Covered in detail in [bf16-rvv-fix-report.md](bf16-rvv-fix-report.md). Two-bug commit: bf16-f32-gemm output pointer was advancing by `cn_stride` instead of `vl` (silent at NC ≤ NR), and all eight kernels had a dead `a*` pointer rewind at end-of-loop.

### `c4d08926` segmented packw

Replaced the strided `vlse16` packw inner loop with `vlsseg{4,8}e16`. The motivation is LSU efficiency on DLEN=128 hardware: segmented loads issue 16-byte-aligned segment fetches per beat, while the strided form issues a single halfword per beat. The strided implementation is preserved in an `#if 0` block as a reference / fallback for any DUT whose `vlsseg` or fractional-LMUL handling is broken — the comment block in the file calls out the Saturn `vlsseg8e16 + e8/mf2 EMUL` bug that motivated keeping the escape hatch.

The `_e16m1` typing on the inner segmented load is deliberate: it gives gcc latitude to emit `e8/mf2` `vsetvli` at EMUL=1 for `vlsseg8e16`, which is what we want on Saturn where each segment is 16 bytes.

### `9003779a` combined `zvfbfmin+zvfbfa` `bf16-gemm` (1x4v, 4x4v)

The combined variant takes the best of each extension at each pipeline stage:
- **Inner FMA**: `vfwmacc.vf` under altfmt=1 vsetvli (Zvfbfa) — same as the Zvfbfa-only kernel; one widening-FMA instruction per KC step.
- **Epilogue narrow**: `vfncvtbf16.f.f.w` under a plain base-V `vsetvli e16,m2` (Zvfbfmin native mnemonic) — no altfmt vsetvli needed for the narrow, which saves the round-trip the Zvfbfa-only kernel pays at store time.

The Zvfbfwma-named opcode `vfwmaccbf16.vf` semantically replaces "altfmt vsetvli + `vfwmacc.vf`" with a single mnemonic that does not need altfmt vtype. The Zephyr binutils on this branch does **not** recognise `vfwmaccbf16.vf` (and does not advertise Zvfbfwma at all), so we don't emit it; the FMA stays as `vfwmacc.vf` under the altfmt `.insn i ... 0x1C9` vsetvli. The combined kernel's win over Zvfbfa-only is therefore solely the Zvfbfmin-native narrow-on-store, not a fused FMA.

(An earlier draft of this report claimed the combined kernel emits `vfwmaccbf16.vf` directly. That description doesn't match the in-tree source — `bf16-gemm-{1,4}x4v-minmax-rvv-zvfbfmin-zvfbfa.c` uses `vfwmacc.vf` under altfmt. The aspirational `vfwmaccbf16.vf` path would only be reachable on a toolchain that supports Zvfbfwma; until then, the file naming is best read as "requires both Zvfbfmin and Zvfbfa flags at runtime", not "uses Zvfbfwma instructions". The Phase 2 igemm kernels follow the same convention.)

### `63eaba10` build options default OFF

`XNNPACK_ENABLE_RISCV_ZVFBFMIN` and `_ZVFBFA` default OFF in `CMakeLists.txt`. Production XNNPACK builds (and the existing CI matrix) do not assemble Zvfbfa-touching code unless the operator opts in; the build-options gate also keeps the `.insn` immediate out of any toolchain that has not been verified against the Zvfbfa v0.1 encoding.

### `1077af6f` wire combined kernels into config

`init_bf16_gemm_config` priority ladder is now:

```
both flags  → 1x4v / 4x4v __rvv_zvfbfmin_zvfbfa   (vfwmacc.vf altfmt + Zvfbfmin narrow)
zvfbfa only → 1x4v / 4x4v __rvv_zvfbfa            (vfwmacc.vf altfmt + altfmt narrow)
zvfbfmin    → 1x4v / 4x4v __rvv_zvfbfmin          (widen + vfmacc + Zvfbfmin narrow)
neither     → scalar fallback (mr stays 0)
```

`init_bf16_f32_gemm_config` keeps the two-rung ladder (no combined kernel — see §Scope). Both configs use the same `x16-x32-packw-x4v-gemm-goi-rvv-u8` packer and `mr = 4`, `nr = vlenb`, `log2_kr = 0`.

## Phase 2 per-stage breakdown

Phase 2 lands in six commits that mirror the dependency chain (kernels → config → operator → subgraph → tests → docs), plus a preceding FC commit:

| Commit | Title | Scope |
|--------|-------|-------|
| `ce803b45` | Add bf16-bf16-bf16 fully-connected operator + subgraph dispatch | FC bf16-output variant, complementing the existing bf16-f32 path |
| `152e64f8` | Add bf16-igemm + bf16-maxpool RVV kernels (zvfbfmin+zvfbfa combined) | 5 kernel sources + reference packers + microkernel headers + build list |
| `01f05893` | riscv: wire bf16-igemm + bf16-maxpool kernels into config | gemm-config.c (igemm slots) + maxpool-config.c (new) + config / config-types decls |
| `9055169d` | Add bf16 conv2d and max-pooling NHWC operators + public API | 3 new operators in convolution-nhwc.c / max-pooling-nhwc.c + decls in xnnpack.h + 3 new operator-type enums |
| `a6089491` | Add bf16 subgraph dispatch + per-tensor type-accept switch fixes | convolution-2d.c + max-pooling-2d.c, with both the new bf16 dispatch arms and the per-tensor type-accept switch fixes |
| `f4b99f2b` | test infra: bf16-igemm + bf16-maxpool test wiring | new YAMLs + generated .cc + tester overloads + CMake registration |
| (this) | docs-ucb-bar: bf16-rvv-kernels-report + supporting docs | this report + symlink-executorch-xnnpack.md + test-architecture.md updates |

Sections below describe the kernel and plumbing units in dependency order rather than per-commit, since some of the smaller commits merge naturally into one logical unit (e.g. configs and headers).

### `bf16-{f32-,}igemm-{1,4}x4v-minmax-rvv-zvfbfmin-zvfbfa` kernels

Indirect-GEMM (igemm) is the path conv2d takes when ks > 1: the activation argument is an indirection array `const uint16_t** a` pointing into im2col-style row pointers, with a per-element `a_offset` apply and a zero-pointer bypass for padded rows. The inner KC loop is identical to the gemm body; only the activation-pointer setup and the post-loop pointer rewind differ.

Structurally each new file is the corresponding gemm kernel's body wrapped in the f32-igemm RVV scaffolding (`src/f32-igemm/gen/f32-igemm-{1,7}x4v-minmax-rvv.c`):
- Outer `do { ... } while (--p != 0);` loop over `ks`.
- Per-element `if (a_i != zero) a_i = (const uint16_t*) ((uintptr_t) a_i + a_offset);` guard.
- After the KC loop, `a = (const uint16_t**) ((uintptr_t) a - ks);` to rewind for the next NC tile.
- Same `vfwmacc.vf` altfmt FMA + (for `bf16-igemm`) `vfncvtbf16.f.f.w` narrow-on-store as the combined gemm kernel.

We ship only the combined-extension variant for each shape this iteration (per the LeNet plan in `.claude/plans/below-is-lenet-stats-crispy-falcon.md`), so the priority ladder for igemm has only the both-flags rung populated; the zvfbfa-only and zvfbfmin-only rungs leave igemm slots empty and the operator falls through to scalar on those configurations.

**Reverse store ordering — 4x4v subtile correctness.** The 4x4v gemm sibling collapses output pointers `c1=c0; c2=c1; c3=c2` when the runtime `mr` parameter is < 4, so all four `vse16` stores write to the same address. For gemm, the four `vacc` registers also collapse (because `a1=a0` etc. up the activation chain) so all stores write the same value and the last-write-wins property is harmless. **Igemm doesn't get this property**: the test framework fills `im2col[ks_index*mr + m_index] = junk.data()` for `m_index >= m`, and the kernel reads junk activations into `vacc1..vacc3` regardless of `mr`. With forward-order stores `c0 = vout0; ...; c3 = vout3;`, the last write (junk) overwrites the valid row-0 result. The fix mirrors the in-tree `f32-igemm-7x4v-minmax-rvv.c` convention: write `c3 → c2 → c1 → c0` so the valid row 0 is the last write and survives the collapse. This applies to both `bf16-igemm-4x4v` and `bf16-f32-igemm-4x4v`.

### `bf16-maxpool-9p-minmax-rvv-zvfbfmin-zvfbfa-u1v` kernel

9-pointer max-pool with FP32-domain reduction (load BF16 → widen with `vfwcvtbf16.f.f.v` → tree-reduce with base-V `vfmax.vv` in fp32m4 → clamp with `vfmin.vf`/`vfmax.vf` against fp32 `output_min`/`output_max` → narrow back with `vfncvtbf16.f.f.w` → `vse16` store). Both convert ops are Zvfbfmin native mnemonics and accepted directly (under `.option arch, +zvfbfmin` push/pop), so this kernel needs no `.insn` workaround at all. The `-zvfbfmin-zvfbfa` filename suffix is for ladder symmetry with the gemm/igemm families; the source actually only exercises Zvfbfmin instructions.

**Liveness — load+widen+reduce progressively.** A first pass at the kernel batched all 9 widens into a single `__asm__` block with 9 `=&vr` outputs and 9 `vr` inputs. At LMUL=m4 each fp32m4 output occupies 4 registers and each u16m2 input occupies 2 registers, so the constraint system requested ~54 vector registers — RVV only has 32. gcc's "unable to find a register to spill" failure surfaced this immediately. The fix restructures the inner loop to load → widen → max-into-accumulator one row at a time, keeping at most one u16m2 + two fp32m4 vectors live at any moment (~10 vregs total). The progressive form also has shorter ILP chains for the reduction than a 9-input batched widen would.

The widening flow over a native bf16-direct flow (`vfmax.vv` on bf16 lanes under altfmt vtype) was chosen for simplicity: the widening flow uses only Zvfbfmin native mnemonics and avoids the altfmt-vsetvli `.insn` workaround. The bf16-direct flow would be ~3× cheaper in instructions but adds Zvfbfa dependency to the kernel. Deferred until profiling justifies the complexity.

### Reference packers — `xnn_pack_bf16_f32_conv_{goki,kgo}_w`

Conv2d operator-create needs to repack 4D source weights (`g × oc × ks × kc`) into the same FP32-bias / BF16-column-major layout the igemm kernels read. The existing `xnn_pack_bf16_f32_gemm_goi_w` only handles 2D weights (gemm). New reference packers in `src/reference/packing.cc` mirror `xnn_pack_f16_conv_{goki,kgo}_w` (uint16 weights + uint16 bias) but advance the bias slot by `nr * sizeof(float)` per NR-tile and copy the bias as fp32. Same packers serve both `bf16-igemm` and `bf16-f32-igemm` because the layout doesn't depend on output dtype. Wired into `init_bf16_{,f32-}gemm_config.pack_igemm_{goki,kgo}` on the both-flags rung.

### gemm-config.c igemm-slot wiring

`init_bf16_gemm_config` both-flags arm extended with `igemm[]` slot fills + the new `pack_igemm_{goki,kgo}`. `init_bf16_f32_gemm_config` gains a new top-rung "both flags" arm at the head of the existing zvfbfa / zvfbfmin / scalar ladder, populating both the zvfbfa-only gemm symbols (no combined gemm exists for bf16-f32 — would be byte-identical) and the new combined igemm symbols. The "neither" rung remains scalar.

```
init_bf16_gemm_config (igemm column only):
  both flags  → 1x4v / 4x4v bf16-igemm-*-zvfbfmin-zvfbfa
  zvfbfa only → (igemm slot empty)
  zvfbfmin    → (igemm slot empty)
  neither     → scalar (mr stays 0)

init_bf16_f32_gemm_config (igemm column only):
  both flags  → 1x4v / 4x4v bf16-f32-igemm-*-zvfbfmin-zvfbfa  (gemm: zvfbfa-only)
  zvfbfa only → (igemm slot empty; gemm: zvfbfa-only)
  zvfbfmin    → (igemm slot empty; gemm: zvfbfmin-only)
  neither     → scalar
```

### maxpool-config.c bf16 entry + struct extension

`struct xnn_maxpool_config.init` union (in `src/xnnpack/config-types.h`) gains a `bf16` slot of type `xnn_init_bf16_minmax_params_fn`. `src/configs/maxpool-config.c` adds `bf16_maxpool_config` static, `init_bf16_maxpool_config()` registering the new 9p kernel under the both-flags rung, and `xnn_init_bf16_maxpool_config()` accessor. Forward decl in `src/xnnpack/config.h`. `src/xnnpack/maxpool.h` `#include`s the new `src/bf16-maxpool/bf16-maxpool-minmax.inc` X-macro registration list alongside the existing f16/f32/s8/u8 lists.

### 3 new operators — conv2d_bf16, conv2d_bf16_f32, max_pooling_bf16

`src/operators/convolution-nhwc.c` gets `xnn_create_convolution2d_nhwc_bf16` (output bf16, gemm_config = `xnn_init_bf16_gemm_config()`) and `xnn_create_convolution2d_nhwc_bf16_f32` (output fp32, gemm_config = `xnn_init_bf16_f32_gemm_config()`). Both pass `dwconv_ukernel = NULL` and `vmulcaddc_config = NULL` — bf16 has no dwconv or vmulcaddc kernels, and the inner `create_convolution2d_nhwc` helper tolerates NULL on those paths (it gates each fastpath behind a non-NULL check; line 588 of the file). `bias_element_size = sizeof(float)` since the packw layout uses FP32 bias even for bf16-output conv. Matching `_reshape_*` and `_setup_*` wrappers go to the existing inner helpers with `xnn_operator_type_convolution_nhwc_bf16{,_f32}` discriminators.

`src/operators/max-pooling-nhwc.c` gets `xnn_create_max_pooling2d_nhwc_bf16` cloning the f16 path with `xnn_init_bf16_maxpool_config()` and bf16 minmax params. Reshape/setup mirror f16's structure.

`src/xnnpack/operator.h` `union xnn_params` gains a `bf16_minmax` slot (used by maxpool reshape's params storage). `src/xnnpack/operator-type-defs.inc` gains 3 enum entries (`xnn_operator_type_convolution_nhwc_bf16`, `_bf16_f32`, `xnn_operator_type_max_pooling_nhwc_bf16`).

### Subgraph plumbing — `convolution-2d.c`, `max-pooling-2d.c`

`src/subgraph/convolution-2d.c`:
- `validate_datatypes_with_bias` accepts (input=bf16, filter=bf16, bias=fp32, output=bf16) and (input=bf16, filter=bf16, bias=fp32, output=fp32). `validate_datatypes_without_bias` accepts the same minus the bias check. Both produce the FP32-bias-slot layout the packw expects.
- **Per-tensor type-accept switches** in `xnn_define_convolution_2d` (input, filter, output — three separate switch statements that run *before* `validate_datatypes_*`) gain `case xnn_datatype_bf16:` arms. **Easy to miss** because the validate functions look like the only dtype gate; the per-tensor switches return `xnn_status_invalid_parameter` from `default:` first. First symptom was an end-to-end runtime failure (executorch XNNCompiler reporting `xnn_status_invalid_parameter` from `xnn_define_convolution_2d`) after the validators were already wired, so the gap doesn't surface in microkernel or operator-layer tests — it only surfaces when something actually calls `xnn_define_*`.
- `define_convolution_2d` gains a bf16 output-dtype arm that branches on the requested output dtype: bf16 output → `xnn_create_convolution2d_nhwc_bf16`, fp32 output → `xnn_create_convolution2d_nhwc_bf16_f32`. The fp32-output arm also adds a bf16 filter-dtype case under the existing fp32 output arm so subgraphs that mix bf16 conv with fp32 downstream activations classify correctly.
- The reshape and setup dispatch switches gain matching `xnn_operator_type_convolution_nhwc_bf16{,_f32}` cases.

`src/subgraph/max-pooling-2d.c`:
- Same per-tensor type-accept switch fix in `xnn_define_max_pooling_2d` (input + output). Same gotcha as conv2d.
- The input-dtype switch in `create_max_pooling_operator` gains a `xnn_datatype_bf16` case calling `xnn_create_max_pooling2d_nhwc_bf16`.
- The reshape and setup switches gain `xnn_operator_type_max_pooling_nhwc_bf16` arms.

Public-API decls for all 3 new operators land in `include/xnnpack.h`.

### Test wiring — gemm + maxpool harnesses

igemm tests (mirrors the Phase 1 gemm test wiring):
- `test/bf16-{f32-,}igemm-minmax.yaml` lists the 4 new kernel symbols + their packer + init.
- `tools/generate-gemm-test.py` (already igemm-aware via the `igemm` token in the kernel name) generates `test/bf16-{f32-,}igemm-minmax.cc`.
- New `xnn_bf16_{f32-,}igemm_minmax_ukernel_fn` typedefs in `src/xnnpack/microfnptr.h`. New `xnn_pack_bf16_f32_igemm_fn` typedef in `src/xnnpack/pack.h` (matches the `xnn_pack_f16_igemm_fn` shape but with bf16 weights / fp32 bias).
- Two new `GemmMicrokernelTester::Test()` overloads in `test/gemm-microkernel-tester.{h,cc}` — one for bf16-f32-igemm (fp32 output buffer, fp32 minmax params) and one for bf16-igemm (bf16 output buffer, bf16 minmax params). Both accept the new packer typedef.
- `test/CMakeLists.txt` adds `bf16-f32-igemm-minmax` and `bf16-igemm-minmax` to `MICROKERNEL_GEMM_UNIT_TESTS`.

maxpool tests (mirrors the existing `.inc`-driven shared driver):
- `test/maxpool-minmax.cc` `#include`s `src/bf16-maxpool/bf16-maxpool-minmax.inc` alongside the f16/f32/s8/u8 lists. The single `XnnTest` driver picks up the new kernel from the X-macro registration.
- New `xnn_bf16_maxpool_ukernel_fn` typedef in `src/xnnpack/microfnptr.h`.
- `test/maxpool-microkernel-tester.h` gains a `Test(xnn_bf16_maxpool_ukernel_fn, xnn_init_bf16_minmax_params_fn)` overload (mirroring the f16 implementation but rounding clamps onto the bf16 grid before the comparison) + a matching `Kernel(...)` constructor.

Build-list registration: the 5 new kernel sources land in `cmake/gen/rvv_microkernels.cmake` `PROD_RVV_MICROKERNEL_SRCS`. The list header notes it's auto-generated by `tools/update-microkernels.py`; the in-tree edit was made by hand for this slice and should be regenerated upstream-side before submitting beyond this branch.

## Cross-cutting design notes

- **vtype discipline.** Every Zvfbfmin/Zvfbfa inline-asm block opens with its own `vsetvli` and assumes nothing about the surrounding vtype, because the compiler-emitted base-V intrinsics around it run at `e32m4`. The cost of the extra `vsetvli` is negligible on spike and on Saturn-class hardware where `vsetvli` is a single-cycle ALU op; the readability win (each block is self-contained) is worth more than the cycles.
- **K step is one BF16 lane.** All gemm/igemm kernels use `k -= sizeof(uint16_t)` per inner-loop iteration: one BF16 activation lane × one NR-wide BF16 weight column per step. `kc % 2 == 0` is asserted; no `kc` padding.
- **Bias slot is FP32, weights are BF16.** This is the packw layout, and it's what `xnn_pack_bf16_f32_gemm_goi_w` (gemm) and `xnn_pack_bf16_f32_conv_{goki,kgo}_w` (igemm/conv) produce. The `bf16-gemm` / `bf16-igemm` (BF16-output) kernels still load FP32 biases — they widen on the fly only inside the K loop. Documented in the kernels via `bias_u16_stride = nr * (sizeof(float)/sizeof(uint16_t))`.
- **Encoding workarounds isolated to vsetvli.** Across all Zvfbfa kernels and the combined kernels, the only `.insn` directive is the altfmt `vsetvli`. Every actual data-movement / FMA / convert instruction is a recognised RVV 1.0 or Zvfbfmin mnemonic. If/when the assembler learns `e16alt` (or `vfwmaccbf16.vf` from Zvfbfwma), the `.insn` directives can be removed mechanically without touching kernel logic. The maxpool kernel doesn't use any altfmt instructions and so has no `.insn` directive at all.
- **Filename `-zvfbfmin-zvfbfa` ≠ Zvfbfwma.** The combined-extension filename suffix gates the runtime config dispatch on both arch flags being present, but does *not* imply the kernel uses Zvfbfwma-named opcodes. All combined kernels (gemm and igemm) emit `vfwmacc.vf` under altfmt vsetvli rather than `vfwmaccbf16.vf`, because the Zephyr binutils on this branch does not recognise the latter. The maxpool kernel is named `-zvfbfmin-zvfbfa` for ladder symmetry but actually only uses Zvfbfmin instructions.
- **Reverse store order for igemm 4x4v.** The 4x4v output-pointer-collapse pattern (`c1 = c0` etc when runtime mr < 4) is harmless for gemm because activations also collapse and all `vacc_i` agree; for igemm activations don't collapse and the test fills `im2col[m_index >= m]` with junk pointers, so writing forward `c0..c3` lets the junk row-3 result overwrite c0. `bf16-{f32-,}igemm-4x4v` mirrors `f32-igemm-7x4v`'s reverse `c3..c0` store order so the valid row 0 is the last write and survives the collapse.
- **Maxpool widening register pressure.** A single `__asm__` block with 9 widening converts requests 9 × 4 (m4 outputs) + 9 × 2 (m2 inputs) = 54 vector registers for the constraint system, which exceeds RVV's 32. The kernel restructures into a load → widen → max-into-accumulator loop body where at most ~10 vregs are live simultaneously. Same pattern in the tail loop that folds the partial-max already staged in the output buffer.
- **Two dtype gates per `xnn_define_*` function.** Subgraph define functions have *both* per-tensor type-accept switches (input/filter/output, one switch each) *and* the cross-tensor `validate_datatypes_*` helper. The per-tensor switches run first; their `default:` arm returns `xnn_status_invalid_parameter` and short-circuits before `validate_datatypes_*` is consulted. Adding a new dtype to subgraph plumbing means touching *all* of these gates — for `convolution-2d.c` that's 3 switches + 2 validators (with-bias and without-bias); for `max-pooling-2d.c` that's 2 switches + 1 datatype-match check. Microkernel and operator tests can't catch a missing per-tensor switch arm because they call `xnn_create_*` directly and bypass the subgraph layer; the gap only surfaces when something runs `xnn_define_*` end-to-end (e.g. executorch's XNNCompiler).

## Files added or substantially rewritten

### Phase 1

Kernels (`src/bf16-gemm/gen/`, `src/bf16-f32-gemm/gen/`):
- `bf16-f32-gemm-{1,4}x4v-minmax-rvv-zvfbfmin.c`
- `bf16-f32-gemm-{1,4}x4v-minmax-rvv-zvfbfa.c`
- `bf16-gemm-{1,4}x4v-minmax-rvv-zvfbfmin.c`
- `bf16-gemm-{1,4}x4v-minmax-rvv-zvfbfa.c`
- `bf16-gemm-{1,4}x4v-minmax-rvv-zvfbfmin-zvfbfa.c`

Shared:
- [src/x16-x32-packw/gen/x16-x32-packw-x4v-gemm-goi-rvv-u8.c](../src/x16-x32-packw/gen/x16-x32-packw-x4v-gemm-goi-rvv-u8.c) — segmented packw (`c4d08926`).
- [src/configs/gemm-config.c](../src/configs/gemm-config.c) — `init_bf16_gemm_config`, `init_bf16_f32_gemm_config`.
- [src/configs/hardware-config.c](../src/configs/hardware-config.c) — Zvfbfmin/Zvfbfa detection + the vsetvli-before-vlenb fix.
- [src/xnnpack/math.h](../src/xnnpack/math.h) — `xnn_nanbox_bf16`.
- [src/xnnpack/gemm.h](../src/xnnpack/gemm.h), `config.h`, `config-types.h`, `hardware-config.h` — symbol forward decls + arch-flag bits.
- `CMakeLists.txt` — `XNNPACK_ENABLE_RISCV_ZVFBFMIN` / `_ZVFBFA` options.

### Phase 2

Kernels:
- `src/bf16-f32-igemm/gen/bf16-f32-igemm-{1,4}x4v-minmax-rvv-zvfbfmin-zvfbfa.c`
- `src/bf16-igemm/gen/bf16-igemm-{1,4}x4v-minmax-rvv-zvfbfmin-zvfbfa.c`
- `src/bf16-maxpool/gen/bf16-maxpool-9p-minmax-rvv-zvfbfmin-zvfbfa-u1v.c`
- `src/bf16-maxpool/bf16-maxpool-minmax.inc` — X-macro registration list

Reference packers:
- `src/reference/packing.cc` — `xnn_pack_bf16_f32_conv_{goki,kgo}_w` (BF16 weights + FP32 bias, conv layout).

Configs:
- [src/configs/gemm-config.c](../src/configs/gemm-config.c) — both-flags rung extended with `igemm[]` slots + `pack_igemm_{goki,kgo}`; new top rung in `init_bf16_f32_gemm_config`.
- [src/configs/maxpool-config.c](../src/configs/maxpool-config.c) — new `bf16_maxpool_config` + `init_bf16_maxpool_config()` + `xnn_init_bf16_maxpool_config()` accessor.

Operators:
- [src/operators/convolution-nhwc.c](../src/operators/convolution-nhwc.c) — `xnn_create_convolution2d_nhwc_bf16{,_f32}` + matching reshape/setup.
- [src/operators/max-pooling-nhwc.c](../src/operators/max-pooling-nhwc.c) — `xnn_create_max_pooling2d_nhwc_bf16` + reshape/setup.

Subgraph:
- [src/subgraph/convolution-2d.c](../src/subgraph/convolution-2d.c) — bf16 dtype validation + dispatch arms + per-tensor type-accept switches (input/filter/output).
- [src/subgraph/max-pooling-2d.c](../src/subgraph/max-pooling-2d.c) — bf16 input-dtype + operator-type dispatch + per-tensor type-accept switches (input/output).

Public API + headers:
- [include/xnnpack.h](../include/xnnpack.h) — 3 new operator decls + reshape/setup decls.
- [src/xnnpack/igemm.h](../src/xnnpack/igemm.h) — `DECLARE_BF16_{F32_,}IGEMM_MINMAX_UKERNEL_FUNCTION` macros + 4 symbol decls.
- [src/xnnpack/microfnptr.h](../src/xnnpack/microfnptr.h) — new ukernel typedefs (`xnn_bf16_{f32-,}igemm_minmax_ukernel_fn`, `xnn_bf16_maxpool_ukernel_fn`).
- [src/xnnpack/operator.h](../src/xnnpack/operator.h) — `bf16_minmax` slot in `union xnn_params`.
- [src/xnnpack/operator-type-defs.inc](../src/xnnpack/operator-type-defs.inc) — 3 new enum entries.
- [src/xnnpack/maxpool.h](../src/xnnpack/maxpool.h) — `#include` of new bf16-maxpool `.inc`.
- [src/xnnpack/config-types.h](../src/xnnpack/config-types.h) — `bf16` slot in `xnn_maxpool_config.init`.
- [src/xnnpack/config.h](../src/xnnpack/config.h) — `xnn_init_bf16_maxpool_config` decl.
- [src/xnnpack/pack.h](../src/xnnpack/pack.h) — decls for the new conv packers + `xnn_pack_bf16_f32_igemm_fn` typedef.

Tests:
- `test/bf16-{f32-,}igemm-minmax.{yaml,cc}` — generated via `tools/generate-gemm-test.py`.
- [test/gemm-microkernel-tester.h](../test/gemm-microkernel-tester.h), [test/gemm-microkernel-tester.cc](../test/gemm-microkernel-tester.cc) — two new `Test()` overloads for bf16-{f32-,}igemm.
- [test/maxpool-minmax.cc](../test/maxpool-minmax.cc) — `#include`s the new bf16-maxpool `.inc`.
- [test/maxpool-microkernel-tester.h](../test/maxpool-microkernel-tester.h) — new `Test()` overload + `Kernel()` constructor for bf16 maxpool.
- [test/CMakeLists.txt](../test/CMakeLists.txt) — registers `bf16-{f32-,}igemm-minmax` test binaries in `MICROKERNEL_GEMM_UNIT_TESTS`.

Build registration:
- [cmake/gen/rvv_microkernels.cmake](../cmake/gen/rvv_microkernels.cmake) — 5 new sources in `PROD_RVV_MICROKERNEL_SRCS` (auto-generated header notwithstanding; regenerate via `tools/update-microkernels.py` before commit).

## Verification

- Phase 1 kernels under spike at VLEN ∈ {128, 256}: see [bf16-rvv-fix-report.md §Verification](bf16-rvv-fix-report.md).
- Phase 2 kernels under spike at VLEN ∈ {128, 256, 512, 1024}: all pass.
  - `bf16-f32-igemm-minmax-test`: 42/42 PASS (21 cases × 2 MR shapes).
  - `bf16-igemm-minmax-test`: 42/42 PASS (21 cases × 2 MR shapes).
  - `maxpool-minmax-test --gtest_filter='*BF16*'`: 228/228 PASS.
- End-to-end LeNet bf16 on spike (Zephyr executor runner, VLEN=256):
  - All 7 ops delegated (2 conv2d + 2 max_pool2d + 3 linear + 4 fused-relu); only `view_copy` + `getitem` framework noise non-delegated.
  - 2 lowered modules total (conv block before flatten, FC block after); single subgraph per block.
  - XNNCompiler instantiates every node (`xnn_define_*` returns success across the board) after the per-tensor type-accept switch fixes landed.

## Known follow-ups / revisit list

### Executorch depthwise heuristic blocks bf16 conv2d delegation

**Symptom.** LeNet bf16 export delegates the conv2d nodes (lstats shows them in the delegated column) but the runtime `XNNCompiler.cpp` fails to instantiate them with `xnn_status_invalid_parameter`:

```
E [executorch:XNNCompiler.cpp:1094] Failed to create depthwise convolution node 1 with code: xnn_status_invalid_parameter
```

**Root cause.** Two layers compose into the failure:

1. `executorch/backends/xnnpack/operators/op_conv2d.py` flags any conv with `group_input_channels == 1` as depthwise — this is mathematically equivalent to a 1-input-channel regular conv (LeNet conv1: `1 → 6`, `groups=1`), so executorch serialises it as `XNNDepthwiseConv2d` to opt into XNNPACK's dwconv fast path.
2. XNNPACK's depthwise subgraph node (`src/subgraph/depthwise-convolution-2d.c`) has no bf16 case in its dtype validator. fp32/fp16 slip through because they have entries; bf16 trips `xnn_status_invalid_parameter`.

**Workaround in place (Fix 1).** Added `and not is_bf16` to the `is_depthwise_conv` heuristic in `executorch/backends/xnnpack/operators/op_conv2d.py` so bf16 convs always serialise as `XNNConv2d` (the path Phase 2 already wires up). One conditional, one file. Restores LeNet bf16 end-to-end.

**The more correct fix (Fix 2a) — revisit.** The dtype gap is in XNNPACK, not executorch. The proper fix is to extend `src/subgraph/depthwise-convolution-2d.c` validators (`validate_datatypes_with_bias` / `_without_bias`, `define_depthwise_convolution_2d`) to accept `(input=bf16, filter=bf16, bias=fp32, output=bf16)` and dispatch to `xnn_create_convolution2d_nhwc_bf16` with `dwconv_ukernel = NULL`. The inner `create_convolution2d_nhwc` helper already falls through to igemm when no dwconv kernel is registered (`convolution-nhwc.c:588`), so this lights up the `XNNDepthwiseConv2d` route at the same igemm-equivalent perf as the regular conv route — *and* the executorch workaround can be reverted, restoring the depthwise fast-path opt-in for any future bf16 dwconv kernel.

**The complete fix (Fix 2b).** Author a `bf16-dwconv` family + dwconv-config entry. Real perf win for true depthwise convs (`groups == in_channels`, `gic == 1` semantically). Out of scope for the LeNet slice (LeNet has no real depthwise convs); justifies its own follow-up if a model with bf16 depthwise lands.

**When to revisit.** Any of:
- Upstreaming this branch — Fix 1 mutates an executorch file outside the XNNPACK tree, which is awkward for a clean XNNPACK PR. Fix 2a localises the change to XNNPACK.
- Adding a real bf16 dwconv kernel (Fix 2b path) — Fix 2a is its prerequisite.
- A model that uses true depthwise bf16 convs hits the partitioner — perf will silently drop to igemm if Fix 2a is in place; without 2a it'd fail to delegate at all under Fix 1 (the workaround disables depthwise routing for *all* bf16 convs, not just gic=1-with-groups=1 misclassifications).

Tracking artifact: `executorch/backends/xnnpack/operators/op_conv2d.py` lines around the `is_depthwise_conv` definition (look for the `# BF16 has no dwconv kernel` comment).
