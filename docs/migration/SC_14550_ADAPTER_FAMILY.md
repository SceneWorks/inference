# sc-14550 — Stable Audio 3 load-time adapters: the full eight-type family

Story: sc-14550 (epic sc-14533). Branch `michaeltrefry1818/sc-14550/p4-lora-adapters-load-t`.

## What shipped

`LoadSpec::adapters` is honoured on all six registered Stable Audio 3 checkpoints, stacked in
request order and folded into the weights as they are served. `supports_lora` flips to `true`;
`supports_lokr` stays `false` and `AdapterKind::Lokr` is refused as typed `Unsupported`.

The scope is the **whole native family**, not classic LoRA:

| # | type | delta | post-transform |
|---|---|---|---|
| 1 | `lora` | `(α/r)·s·(B @ A)` | none |
| 2 | `dora-rows` | as above | row-normalize, rescale by `magnitude_r` |
| 3 | `dora-cols` | as above | column-normalize, rescale by `magnitude_c` |
| 4 | `bora` | as above | rows, then columns |
| 5 | `lora-xs` | `(α/r)·s·(U @ M @ Vᵀ)` | none |
| 6 | `dora-rows-xs` | as 5 | rows |
| 7 | `dora-cols-xs` | as 5 | columns |
| 8 | `bora-xs` | as 5 | rows, then columns |

plus the legacy `dora` / `dora-xs` aliases, resolved by sniffing which magnitude tensor the file
carries and defaulting to the row axis (upstream's training default).

`U` and `V` for the four `-xs` types are the top-`r` singular vectors of the **base weight**, not
tensors in the adapter file. See "The `-xs` half" below.

## ⚠ What is NOT claimed

**No Stable Audio 3 adapter artifact exists on this machine, in any cache, of any type.** An
exhaustive search returns only LTX / Wan / SDXL / SCAIL2 / Anima / SenseNova adapters, and no
published community SA3 adapter is cached. `sc-15347` tracks obtaining them and **blocks this
story's real-weight-artifact acceptance**.

Everything gated here uses adapters this repository *synthesizes*, in the on-disk format
`crates/audio/candle-audio-stable-audio-3/src/adapters.rs` declares. That is a genuine exercise of
the load → validate → plan → fold → serve path, including against real checkpoints on real weights.
It is **not** evidence that the key spellings in that format match an upstream file, and no
cross-framework parity claim appears anywhere in the diff.

The native key spelling — `"{target-without-.weight}.{index}.{factor}"` with
`factor ∈ {lora_A, lora_B, lora_M, magnitude_r, magnitude_c}` and `__metadata__` carrying
`adapter_type` / `rank` / `alpha` / `include` / `exclude` — is **this repository's declared format**,
derived from the described structure of the upstream family. When a real artifact lands under
sc-15347 the reader may need to grow an alias table; that is the expected outcome, not a defect.

## Where the fold happens, and why there

`adapters::AdapterBackend` is a `candle_nn::var_builder::SimpleBackend` wrapping the root
checkpoint's backend. Each weight is adapted at the moment the graph asks for it.

The alternative — materialize the checkpoint, fold, rebuild — was rejected on a number:
`stable_audio_3_medium` is 1.45B parameters, so a whole-checkpoint clone costs ~5.8 GB of host
memory before the graph exists. The wrapper keeps the fold per-module and lazy, releases temporaries
immediately, and serves every unplanned key from the inner backend untouched.

Two structural facts fall out of *where* the wrapper sits, and they are stronger than the checks
they replace:

* **T5Gemma is unreachable.** The text encoder is a different file
  (`t5gemma-b-b-ul2/model.safetensors`) with its own backend, which is never wrapped. No adapter key,
  however spelled, can reach a T5 tensor — the backend serving them has no plan to consult. Note that
  T5Gemma's own keys begin `model.encoder.…` and therefore *pass* the prefix rule; the thing that
  actually keeps them out is that they are not keys of the root checkpoint at all.
* **SAME is in the same file**, so it is excluded explicitly (`is_adaptable_target` refuses
  `pretransform.…`) and an adapter naming one fails loudly at plan time.

Adaptable targets are 2-D Linear and 3-D Conv1d `.weight` tensors under `model.` or `conditioner.`.
On the real `small-music` header that is exactly the DiT projections, `preprocess_conv` /
`postprocess_conv`, and one conditioner module:
`conditioner.conditioners.seconds_total.embedder.embedding.1.weight` (`[768, 256]`) — the learned
`seconds_total` NumberConditioner Linear. 1-D gains, biases and embeddings are never targets,
because **none of the eight types is a bias-diff type**.

## Order is load-bearing here, unlike the image lane

Classic LoRA deltas commute, which is why the image providers fold a stack into a map and do not
care. **DoRA and BoRA do not.** The strength participates *inside* the normalization —
`magnitude · (W + s·δ) / ‖W + s·δ‖` is not linear in `δ` — so swapping two adapters produces
genuinely different weights. The plan preserves `LoadSpec::adapters` order and the fold is strictly
sequential, each adapter seeing the previous one's output as its base.

`dora_stacks_do_not_commute_and_classic_lora_stacks_do` asserts **both** directions, four orders of
magnitude apart: classic LoRA's two orderings agree to `< 1e-5` (F32 addition is not associative, so
the disagreement is a last-ulp artifact), DoRA/BoRA disagree by `> 1e-3`. One direction alone would
not prove the harness measures order rather than noise.

A consequence worth recording because it is a **choice**, not a derivation: for a stacked `-xs`
adapter the SVD bases come from the weight *as it enters that adapter*, not from the pristine
checkpoint weight. That is the reading under which "fold sequentially" and "attach parametrizations
sequentially" are the same operation. It is unverified against an upstream artifact.

## `scale == 0.0` is a no-mutation fast path, not a multiply by zero

`plan_for` loads and **fully validates** every adapter — including zero-scale ones, whose mistyped
keys, wrong ranks and bad shapes are all still refused — and then drops the zero-scale members. If
none survives, the result is an **empty plan**, and an empty plan is never installed: the load takes
the ordinary mmap/packed path and produces byte-identical weights.

The distinction is not cosmetic. `a_zero_scale_stack_validates_fully_and_then_mutates_nothing` uses
an adapter whose factors are `1e30`, so `B @ A` overflows F32 to `inf`. The case asserts three
things: the zero-scale plan is empty, the weight comes back bit-identical, **and** the same adapter
at scale 1.0 really does produce non-finite values — so a multiply-by-zero implementation would
poison the checkpoint with `NaN`, and the fast path is proven load-bearing rather than assumed to be.

## The `-xs` half — implemented, deterministic, and expensive

Candle has no `linalg.svd` on any backend. `src/svd.rs` provides a one-sided Jacobi decomposition
with two deliberate properties:

* **It never runs on the accelerator.** Every rotation is host `f64` arithmetic on `Vec<f64>`. There
  is no Metal/CUDA reduction whose summation order could vary, so CPU, CUDA and Metal execute the
  identical instruction sequence on identical inputs. Cross-platform agreement is a property of the
  construction, not a bound someone measured and hoped would hold. This is the direct answer to the
  sign-drift hazard: a sign that depends on the backend is a *misapplied* adapter, which is worse
  than a failed one.
* **It is exact, not iterative-approximate.** Fixed cyclic sweep order, no random start, no seed, no
  spectral-gap dependence. A randomized or Lanczos truncated SVD would be far faster and would
  reintroduce exactly the drift being avoided.

Sign canonicalization follows upstream: each `U` column is signed by its largest-magnitude entry, and
**the matching `V` column takes the same flip**. Ties resolve to the lowest row index; an entry of
exactly `0.0` does not flip. Columns sort by singular value descending with a stable tie-break.

### The cost, measured

Apple M-series, `--release`, `f64`, `k = 8`:

| shape | wall |
|---|---|
| `128 x 128` | 0.048 s |
| `256 x 256` | 0.99 s |
| `768 x 256` | 1.90 s |
| `512 x 512` | 10.9 s |
| `1024 x 1024` | 112.7 s |

Growth is roughly `n^3.4`. Truncating to `k` does not help — one-sided Jacobi orthogonalizes every
column before any can be discarded.

**So a full-DiT `-xs` adapter is a multi-hour cold start** on `small-*` (1024-wide) and worse on
`medium` (1536×24). The math is correct and gated at every scale; only the wall clock is impractical.
The conditioner's `[768, 256]` Linear at 1.9 s is comfortably practical and is what the real-weight
`-xs` case exercises. **This is surfaced, not narrowed away**: an accelerated *deterministic*
truncated decomposition, or an explicit safe precomputed-bases contract (which is what upstream's
`--svd_bases_path` training flag caches), is the fix and is filed as a follow-up.

### `svd_bases.pt` is still never read

Each `-base` repository ships `svd_bases.pt` (1.27 GB on the smalls, 3.84 GB on medium). It is a
**startup cache** for exactly the decomposition above, not a correctness requirement — and it is a
pickle, i.e. an arbitrary-code-execution surface. It is refused by the same rule as every other
pickle, so nothing special-cases it, and the download allow-lists in `release/real-weight-models.toml`
stay exactly as they are (5.1 GB stays off the runner).

## Formats and the security boundary

Accepted: a native `.safetensors` file, or a PEFT directory (`adapter_model.safetensors` +
`adapter_config.json`). Both spellings of the same adapter are asserted to fold **identically**,
across all eight types — otherwise one of the two readers is silently transposing or dropping a
factor.

Refused with a typed `Unsupported`, naming the boundary: `.ckpt`, `.pt`, `.pth`, `.bin`, and any
unrecognized extension. Unpickling is arbitrary code execution and an adapter path is caller-supplied
data; there is no opt-in. The extension list is a denylist *plus* a catch-all refusal, so its
existence does not make an unknown format implicitly safe.

## Filters and "every tensor consumed exactly once"

`include` / `exclude` come from the adapter's own metadata (safetensors `__metadata__`, or PEFT's
`target_modules` / `exclude_modules`) and are substring matches with upstream's bracket-range
expansion: `layers.[0-3].cross_attn` becomes four substrings, multiple brackets expand as a cartesian
product, and a malformed or inverted range is an **error** rather than a silent literal match — a
filter that quietly matches nothing is exactly how an adapter no-ops.

The consumption rule is strict, and the reading is recorded because it is a decision: since the
filters come from the *file*, they describe the module set the adapter was trained over. A module
whose target the file's own filters exclude is an internal contradiction, so it is refused
(`no_target_matched`) rather than silently dropped. That is what makes "every adapter tensor consumed
exactly once" a checkable property. `AdapterSpec` carries no caller-side filter field, so no caller
capability is lost by this reading.

Four ways consumption can break, four refusals, all gated: an unknown factor segment; a factor the
declared type cannot use; a required factor missing; an `-xs` file that also ships `lora_A`/`lora_B`
(whose bases come from the weight, so those tensors have no consumer).

## Conv1d

`[out, in, k]` flattens to `[out, in*k]`, is adapted, and is restored to the original shape. The gate
uses `k = 2` deliberately: both of SA3's real Conv1d targets have `k == 1`, where a wrong flatten is
invisible on shape alone, so the case checks the delta entrywise against the flattened arithmetic.

## The contract was NOT extended — and that is a measurement, not a preference

Upstream exposes a per-adapter sigma interval and a runtime DiT layer filter. Neither is expressible
by a load-time weight fold: a sigma interval means the adapter is active only within part of the
denoise, which requires runtime hooks in the DiT rather than a folded weight. Adding
`AdapterSpec` fields for them here would ship knobs that parse and do nothing — the exact
"appears to work" failure this epic keeps finding.

The blast radius was measured before deciding, per sc-14549's precedent: **98 `AdapterSpec { … }`
struct-literal sites across the workspace, and `git grep` finds ZERO of them using
`..Default::default()`.** So a new field on `AdapterSpec` breaks 98 sites. That cost buys nothing
until a runtime application path exists.

Decision: **do not extend the contract; file the runtime capability as its own story**, with this
measurement attached. The two *existing* model-specific knobs are refused by name rather than
repurposed or ignored — `pass_scales` is LTX-2.3's per-denoise-stage schedule, `moe_expert` selects
one expert of Wan2.2's dual-expert MoE — and a spec carrying neither is accepted, so the gate
discriminates.

## Where the refusals happen

Split deliberately across two points:

* `adapters::validate_spec_shape` runs in `load_variant` **before the snapshot directory is opened**.
  A `Lokr` request, or one carrying another backend's knob, is a statement about the request that
  reading the checkpoint cannot change, and the caller should hear about it immediately.
* `resolve_adapter_plan` runs after snapshot identity, still at `load_variant` — so a malformed,
  pickle-format, or key-mismatched adapter fails at **load** rather than at first generate, minutes
  later, behind a cold start. Its result is discarded; the plan the pipeline folds is rebuilt on the
  compute device by the identical function, so the two paths cannot drift. That second bullet is
  gated by the real-weight `a_key_mismatched_adapter_is_refused_at_load_variant_not_at_first_generate`
  case; deleting the call was green in every lane before it existed.

### Target counts: the fixture's number is not the checkpoint's

Worth stating because it has already been confused once, in a story comment since corrected. The
**real** `small-music` checkpoint has **193** adaptable targets out of 685 tensors. The **8** that
appears in `tests/adapters.rs` is the count for `real_key_shapes()`, a deliberately small
representative slice of that header — not a property of the model. Any claim of the form "the
checkpoint has N adaptable targets" must be read off the pinned header, not off the fixture.

## Mutation matrix

Every site was mutated by a single-token edit and the suite re-run. See the story comment for the
run-by-run record; the summary is in `crates/audio/candle-audio-stable-audio-3/tests/adapters.rs`
and in the ci.yml audit comment.

## What a single-token edit can still do

Named rather than claimed closed:

| edit | weight-free | catcher |
|---|---|---|
| drop the `AdapterBackend` wrapper install in `full_pipeline_builders_with_adapters` | green | real-weight `real_adapters_change_the_render_…` (the render stops changing) |
| pass `None` for `adapters` in the lazy `pipeline()` path | green | same |
| swap `row_norm`'s `sum_keepdim(1)` for `(0)` | **RED** — `dora_rows_normalizes_rows_…` | weight-free |
| drop the `-xs` `v.t()` transpose | **RED** — `lora_xs_reconstructs_…` | weight-free |
| remove the sign flip's propagation to `V` | **RED** — `xs_sign_canonicalization_propagates_from_u_to_v` | weight-free |
| `filter(|a| a.scale != 0.0)` → `filter(|_| true)` | **RED** — `a_zero_scale_stack_…` | weight-free |
| fold ops in reverse order | **RED** — `dora_stacks_do_not_commute_…` | weight-free |

The first two rows are honestly green in the PR lane. Both are load-wiring, not math, and both are
caught by the real-weight lane on `sa3-base-identity-{metal,cuda}` — the only jobs provisioning all
six snapshots.

### Gates added because the review found them ungated

Adversarial review found no correctness bug in shipped code. What it found was **gates that would
not have caught their own regression** — code that was right, guarded by a test that could not tell.
Each row below was verified by applying the edit and observing the named case go RED; each was
**green before** the case existed.

| edit | before | after |
|---|---|---|
| delete `resolve_adapter_plan(…)` in `load_variant` | green everywhere, incl. the real-weight lane | **RED** — real-weight `a_key_mismatched_adapter_is_refused_at_load_variant_not_at_first_generate` |
| `is_adaptable_target`'s `.weight` suffix rule → `if false` | green | **RED** — `only_dit_and_conditioner_…`, `the_target_set_is_read_from_a_real_safetensors_header` |
| PEFT `r` absent → default to a constant | green | **RED** — `a_malformed_peft_config_is_refused_rather_than_defaulted` |
| PEFT `lora_alpha` absent → default to `1.0` | green | **RED** — same |
| PEFT `!alpha.is_finite()` → `if false` | green | **RED** — same |
| PEFT `rank == 0` refusal deleted | green | **RED** — same |
| `json_filter`'s non-string entry → coerce | green | **RED** — same |
| native `{index}` integer check → `if false` | green | **RED** — `a_native_adapter_index_segment_must_be_an_integer` |
| `has_row \|= module.magnitude_r.is_some()` → `\|= false` | green | **RED** — `the_legacy_dora_alias_…` (the both-magnitudes file) |
| `AdapterBackend::get_unchecked` body → serve every key unadapted | green | **RED** — `the_adapter_backend_serves_unplanned_keys_untouched` |
| `build`'s `matched == 0` refusal deleted | green | **RED** — `a_module_less_adapter_is_refused_by_build` |
| `adapter_index` renumbered positionally after the zero-scale filter | green | **RED** — `adapter_index_survives_the_zero_scale_filter` |
| classic-LoRA strength clamped to `[0, 1]` | green — the single-point `s = 0.5` case does not see it | **RED** — `classic_lora_delta_norm_is_exactly_linear_in_the_requested_strength` |

The last row is the one worth reading twice. `classic_lora_folds_exactly_alpha_over_rank_times_scale…`
pins the scale at **one** value, and a clamp to `[0, 1]` satisfies it exactly. Only the
`{0.25, 0.5, 1, 2}` sweep — which asserts the analytically exact `‖δ(s)‖ == s·‖δ(1)‖`, not
monotonicity — separates the two.

The `.weight` row has a fixture note attached to it: the suffix rule was unobservable because every
non-`.weight` key in `real_key_shapes()` was 1-D and thus rejected on rank alone. The real
`small-music` header carries exactly **one** 2-D non-`.weight` tensor under an allowed prefix,
`model.model.transformer.memory_tokens [64, 1024]` (verified against the pinned header: 1 of 685
tensors), and adding it is what makes the rule discriminating. Without it a regression would make a
learned parameter table an adapter target.

## CI

* **Weight-free**: `--test adapters` added to ci.yml's SA3 step. The lane now runs **eleven targets /
  90 live cases** (was ten / 56), measured by running the step. The audit comment is updated from the
  measurement, and `scripts/tests/test_sa3_ci_target_coverage.py` passes.
* **Real weights**: `--test adapters -- --ignored` on `sa3-base-identity-metal` and
  `sa3-base-identity-cuda`, verified against the jobs' actual `--test` flags rather than their
  comment blocks. Four cases: the two exactly-signed gates on all six checkpoints, the order case,
  the `-xs` case scoped to the conditioner Linear, and the load-time key-mismatch refusal.

The Metal lane matters beyond redundancy here: `packed_metal_backend` is a **different**
`SimpleBackend` from CPU/CUDA's mmap, and the wrapper composes over both. No other lane exercises
that composition.
