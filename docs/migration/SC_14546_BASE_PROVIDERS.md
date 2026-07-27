# sc-14546 — the three `-base` Stable Audio 3 providers

Registers `stable_audio_3_small_music_base`, `stable_audio_3_small_sfx_base` and
`stable_audio_3_medium_base`: the pre-trained flow-matching checkpoints the three post-trained
providers (sc-14543/14544/14545) were distilled and adversarially post-trained from.

| | value |
|---|---|
| Provider ids | `stable_audio_3_small_music_base`, `stable_audio_3_small_sfx_base`, `stable_audio_3_medium_base` |
| Repositories | `stabilityai/stable-audio-3-small-music-base@eab5ceee5ad9c1ed38800aff30a8e49d1161c539` · `…-small-sfx-base@cc5ddb990e30daa68336ac61c140c37c7033ab7c` · `…-medium-base@b32993f73c3bdc3864043a72d8032606bba737c8` |
| Root artifacts | 2,270,384,940 B (both smalls) · 9,222,116,660 B (medium-base) — the *same* lengths as their post-trained siblings |
| Architecture | identical to the post-trained sibling in every field: `1024×20` ordinary DiT over SAME-S, `1536×24` differential DiT over SAME-L |
| `diffusion_objective` | `rectified_flow` (post-trained: `rf_denoiser`) |
| `sample_size` | 5,324,800 (smalls; post-trained 5,292,032) · 16,777,216 (medium, unchanged) |
| Advertised maximum | 120 s / 120 s / 380 s — unchanged from the post-trained siblings |
| Default sampler / steps / guidance | **Euler / 50 / 7.0** (post-trained: Pingpong / 8 / 1.0) |
| Hub gating | ungated, unlike the post-trained three — same Stability Community + Gemma license files |
| Weight licenses | 9 new rows, 18 total — see `release/model-weight-licenses.json` |

## Two claims in the story that the code contradicted

**"Base variants get `supports_negative_prompt` / `supports_guidance` `true` as a divergence."**
Both flags have been `true` for all three post-trained ids since sc-14544 (`model.rs`, pinned with an
explicit comment). Implementing the story literally would have meant flipping three shipped
descriptors to `false` to manufacture a difference. The assertions were treated as the tripwire they
are and left alone.

**"The bases are the only checkpoints where `cfg_scale` and `negative_prompt` do anything."** False at
code level. `dit.rs`'s guided forward takes its batch-1 shortcut on `cfg_scale == 1.0` and on no
other condition, so at any other guidance the identical batch-2 CFG / APG / rescale path runs on a
post-trained checkpoint. Upstream's "no effect on post-trained checkpoints" is a claim about what
distillation did to the *weights*. What is true, and what this story pins, is that `1.0` is the
post-trained **default** and `7.0` is the base default — so a default post-trained request genuinely
ignores its negative prompt and a default base request genuinely does not.

The related open question from the epic — whether `rectified_flow` and `rf_denoiser` are different
prediction targets — is settled here as **cosmetic except for the sampler default**. The two are
handled identically at every branch in `dit.rs` (sigma, the shared
`denoised = x − sigma·out` arm, the guided combine, APG / `cfg_norm_threshold` / `scale_phi`); only
the unreachable `V` objective differs. The one mechanical consequence is
`SamplerKind::recommended`, upstream's `sampler_type = "pingpong" if diffusion_objective ==
"rf_denoiser" else "euler"` rule.

## The identity gate

`Variant::geometry()` previously set `expected_repo: self.hub_repo()` and `validate_layout` compared
that string to the snapshot's declared conditioner `repo_id`. **Every base `model_config.json`
declares its post-trained sibling's repository**, so carrying that forward would have rejected every
base snapshot that will ever exist; deleting the check instead would have reopened the two
architecturally identical smalls to each other.

`VariantGeometry` therefore now carries three fields where it carried two:

- `hub_repo` — **provenance**. License source URLs, the CI snapshot manifest, error messages. Never
  compared to the snapshot.
- `expected_conditioner_repo` — the **gate value**, `Variant::conditioner_repo()`. Equal to
  `hub_repo` for the post-trained ids, equal to the post-trained sibling's repository for the base
  ids. Compared as a string; never resolved over the network.
- `expected_objective` — per variant, replacing the hard-coded `!= RfDenoiser`. Parameterised rather
  than loosened: a global relaxation would have opened all three post-trained ids to their base
  siblings at once, which is exactly the failure sc-14545 avoided when `VariantShape` became
  per-variant.

What actually discriminates each pair, in both directions:

| pair | config-level discriminators | payload |
|---|---|---|
| `small_music` ↔ `small_music_base` | `sample_size` (5,292,032 / 5,324,800) **and** `diffusion_objective` | distinct root SHA-256 at identical byte length |
| `small_sfx` ↔ `small_sfx_base` | same two | same |
| `medium` ↔ `medium_base` | **`diffusion_objective` only** — identical inventory, `sample_size`, declared `repo_id` | distinct root SHA-256 at identical 9,222,116,660 B |
| `small_music_base` ↔ `small_sfx_base` | conditioner `repo_id` | distinct root SHA-256 |
| small bases ↔ `medium_base` | architecture | never reached |

`tests/variant_binding.rs` runs all of it on real weights, each direction preceded by a
load-must-succeed control, plus the mutation that isolates the hash pin: each sibling's root
`model.safetensors` under the *other's* `model_config.json`, so the objective, `sample_size` and
`repo_id` checks all pass and only the digest can catch it. Measured rejection messages are in the
PR body.

The medium pair's SHA-256 pin is load-bearing in a way nothing else in this crate is. It is verified
twice — once in `load_variant`, once again on the lazy pipeline path immediately before tensors are
mmapped — and the second pass is the only thing standing between `stable_audio_3_medium` and a
pre-trained checkpoint swapped in after load.

## Per-variant defaults, and where 50 / 7.0 come from

`pipeline::DEFAULT_STEPS` / `DEFAULT_GUIDANCE` (8 / 1.0) stay as the upstream API defaults;
`BASE_DEFAULT_STEPS` / `BASE_DEFAULT_GUIDANCE` (50 / 7.0) are new, and `synthesis_parameters`
resolves all three omitted fields — sampler, steps, guidance — through `Variant`.

**These are a product choice, not an upstream default, and the code and this document say so.**
Upstream's Python and CLI entry points default to 8 steps at guidance 1 for *every* checkpoint; only
Stability's Gradio app varies them per model. They are not invented here either: each shipped
`model_config.json` carries a `training.demo` block recording the operating point the checkpoint was
demoed at, and the two halves of the family disagree there exactly the way the shipped defaults do.

| | `training.demo.demo_steps` | `training.demo.demo_cfg_scales` |
|---|---|---|
| all three post-trained | 8 | `[1]` |
| all three `-base` | 50 | `[2, 4, 7]` |

`training.*` is never read at inference, so `tests/base_guidance.rs` reads the raw JSON out of all
six pinned snapshots and asserts the shipped defaults still match. That is what stops the two
constants from being unattributable numbers.

`SamplerKind::recommended` was **dead code on the provider path** before this story —
`synthesis_parameters` hard-coded `None | Some("pingpong") => Pingpong`, which happened to be right
for the three post-trained ids and is wrong for all three bases. It is now the source of the
resolved default, and `tests/provider.rs` proves the resolution end to end by rendering the same seed
with the sampler omitted, with `sampler: Some("euler")` and with `sampler: Some("pingpong")`: the
defaulted render must be byte-identical to the first and different from the second. Pingpong draws
one full-latent random tensor per step off the same request-local stream, so the two solvers cannot
agree by accident.

### The cost, stated plainly

A default base render is 50 Euler steps at guidance 7, i.e. a **batch-2** CFG forward per step: 100
DiT forwards against the post-trained default's 8, **12.5×** the example-work per second of audio.
On `stable_audio_3_medium_base` this compounds with the family's "an omitted `audio.target_duration`
renders the checkpoint's full length" rule: a request that omits the field asks for 380 s at 100
forwards per step-pair, on the order of **20 minutes of Metal compute**. The default is still not
special-cased — six ids in one family obeying two rules for the same missing field is a worse
contract than one expensive uniform rule — but it is documented on
`Variant::default_duration_secs` and here. `Capabilities` has no field for a default duration, step
count, guidance or sampler, so none of this is advertised; that gap is tracked with the other
additive descriptor gaps as sc-15041.

## The advertised cap stays 120 s on the small bases

The small bases' `sample_size` is `5,324,800` frames = `120.743764…` s, and an earlier note on this
story proposed advertising the larger figure. The tightness rule in `tests/conformance.rs` rules it
out from the other side: the advertised cap must be a whole second count whose successor does *not*
fit, and `121 s = 5,336,100 > 5,324,800`. `120` is still the tightest advertisable cap, and the
residual `0.74 s` is unreachable through the descriptor by the same rule that keeps medium at `380`.
What the larger `sample_size` changes is the **validator**, not the advertisement — and that is
precisely what separates a base snapshot from its post-trained sibling.

## Prompts: the `demo_cond` convention breaks here

sc-14543/14544/14545 sourced each variant's test prompt from the checkpoint's own `demo_cond`. Two of
the three bases make that impossible, so the rule is abandoned deliberately rather than applied
blindly:

- `small-music-base` ships four genuine music `demo_cond` prompts. Its first is used verbatim.
- `small-sfx-base` ships the **music-base prompt list, unchanged** — "A beautiful piano arpeggio…",
  "Amen break 174 BPM", "lofi house loop". That is copy-paste in the shipped config, not an SFX
  prompt set. Rendering and calibrating a Foley checkpoint on it would measure the wrong
  distribution entirely, so this variant uses its post-trained sibling's shipped SFX prompt and
  sweeps the post-trained SFX prompt list.
- `medium-base` ships **no `demo_cond` at all** (`training.demo` carries only `num_demos: 4`). It
  takes its post-trained sibling's prompt and the same two-domain sweep list, for the same reason.

## The CFG / negative-prompt gate, and its derived floor

`tests/base_guidance.rs` is the story's acceptance evidence and it does not invent a tolerance:

1. It replays the frozen-Torch guidance oracle
   (`docs/migration/sa3-sampler-reference/guidance.safetensors`, vanilla CFG / APG /
   blended+rescaled at `cfg_scale = 2.5`) on `small-music-base`, on the same device and dtype, and
   records the largest per-element disagreement. That is the noise floor.
2. At `cfg_scale = 1.0` it requires a negative prompt to be a **bit-for-bit no-op**. This is the
   honest control: the DiT never evaluates the negative branch there, and a test that "proved"
   otherwise would be proving a bug.
3. At `cfg_scale = 7.0` — the base default — it requires the negative-prompt divergence, measured in
   the same latent space from the same initial noise at the same single Euler step, to exceed the
   floor from step 1.

Be precise about what step 3 proves. The floor is an *agreement* quantity (`3.685e-3` on Metal /
`small-music-base`) and the divergence is a *signal* quantity (`368.9`), so the comparison
discriminates exactly one alternative: a divergence of zero, i.e. negative threading removed
outright. It says nothing about a negative branch that is threaded but wired **wrongly**. No second,
larger multiple of the floor is asserted — an earlier draft required `≥ 10×`, which was a number
chosen by taste that bought nothing at a measured margin of `100,105×`.

What covers a mis-wired-but-non-zero branch is a separate case in the same file,
`the_guided_latents_are_exactly_the_cfg_recomposition_of_their_own_two_branches`. With
`apg_scale = 0`, `cfg_norm_threshold = 0` and `scale_phi = 0`, `dit::guided_prediction` reduces
algebraically to `uncond + g·(cond − uncond)` in v-space, and one Euler step is affine in the model
output, so the guided latents are pinned to an **identity**:

```text
L_guided(P, N, g) == L(N) + g · (L(P) − L(N))
```

where `L(X)` is the same Euler step at `cfg_scale = 1.0` — the batch-1 branch — on prompt `X`, and
`N` is the negative embedding *after* the attention-mask multiply `forward_guided_impl` applies
before batching. The only difference between the two sides is batch-2 versus batch-1 numerics.

The bound is the same frozen-Torch floor, read at this comparison's own guidance scale. Both
quantities are guided-space disagreements, and `uncond + g·(cond − uncond) = g·cond − (g − 1)·uncond`
amplifies an independent per-branch error by `2g − 1`. The floor is measured at the oracle's
`g = 2.5` (factor 4) and the residual at the base default `g = 7` (factor 13), so the floor is scaled
by `13/4`.

That makes it a principled cross-scale bound calibrated on three backends — not a derivation, and the
difference matters. The two quantities do not measure the same error: the floor is this
implementation's disagreement with frozen Torch, the residual is batch-2-versus-batch-1 disagreement
inside this implementation. They are related through the shared `2g − 1` recombination by analogy,
not by identity. The floor is also a `max` over three oracle cases, two of which (`apg_scale = 1.0`
and the blended rescale) do not recombine as `2g − 1`, so attributing factor 4 to it is loose in an
unspecified direction. What the rescaling is *not* is slack bought to make the gate pass: the first
draft used the unscaled floor, which passed on Metal and **failed on every other backend measured**
(CPU residual `5.05e-3` against a floor of `4.03e-3`; CUDA `3.524780e-3` against `3.471375e-3`) — a
Metal-only bound, exactly the accident the rescaling removes — and the closest mis-wiring still
overshoots the rescaled bound by 6,690x.

| | frozen-Torch floor (`g = 2.5`) | rescaled bound (`g = 7`) | correct recomposition | swapped halves | weight as `g − 1` | negative mask dropped |
|---|---:|---:|---:|---:|---:|---:|
| Metal | 3.684998e-3 | 1.1976e-2 | **6.1035e-5** | 1140.78 | 87.75 | 454.28 |
| CUDA | 3.471375e-3 | 1.1282e-2 | **3.524780e-3** | 1140.82 | 87.76 | 454.29 |
| CPU | 4.028320e-3 | 1.3092e-2 | **5.050659e-3** | 1138.90 | 87.61 | 454.05 |

Which rows are *enforced* and which are reference matters, because sc-14546 changed it. `base_guidance`
now honours `SA3_TEST_CUDA`, so the two lanes that run this gate — `sa3-base-identity-metal` and
`sa3-base-identity-cuda` — pin the Metal and CUDA rows, and **no lane measures the CPU row any more**.
The CPU row is retained because it is the row that motivated the rescaling and is still reproducible
locally with no `SA3_TEST_*` set; the Metal and CUDA figures are from
run [30259152906](https://github.com/SceneWorks/inference/actions/runs/30259152906).

The CUDA row is also what stops "the unscaled bound was a Metal-only accident" from reading as a
CPU-only quirk. The CUDA residual, `3.524780e-3`, **exceeds its own unscaled floor**, `3.471375e-3` —
so the first draft's bound would have failed on the CUDA lane too, not just on CPU. The rescaling was
necessary on both non-Metal backends; Metal's `6.1035e-5` residual, two orders of magnitude under its
floor, is the outlier.

Every one of those mis-wirings diverges from the no-negative render just as loudly as the correct
wiring, so every one passes step 3 — and overshoots this bound by at least 6,690x (the closest,
`g − 1` on CPU; 7,328x on Metal and 7,779x on CUDA), while the correct recomposition stays under it
on all three backends.

`tests/dit_oracle.rs`'s `real_weights_detect_conditioning_mutations_and_exercise_cfg_apg` is the
third leg: it is the only case that separates *absent* negative conditioning — which takes the
`zero_cross_context_from_batch` path and zeroes the entire cross context — from an explicit
all-invalid negative prompt, which retains its conditioned duration row. It ran in no lane before
this story; sc-14546 wires it into `sa3-base-identity-{metal,cuda}` and switches it from a hardcoded
`Device::Cpu` to the shared selector, since every assertion in it is a self-comparison between two
real-weight forwards and it must certify the backend whose CFG path it covers.

A further case runs the inertness/materiality pair end to end through `provider_registry().load(…)`
on decoded PCM, which is what covers `GenerationRequest::negative_prompt` actually reaching the
conditioner.

CLAP text-audio similarity is deliberately **not** a gate. It is not guaranteed monotonic in CFG, so
a monotonicity assertion would be a flake generator rather than a correctness gate. The hard gate is
fixed-guidance frozen-Torch parity.

## Device selection

There is no blanket rule here, so this section enumerates instead of asserting one. Three mechanisms
exist in `crates/audio/candle-audio-stable-audio-3/tests/`, and all 15 targets in that directory
appear below. Several targets split across mechanisms case by case — `variant_divergence.rs`,
`base_guidance.rs`, `dit_oracle.rs`, `text_oracle.rs` and `real_snapshots.rs` each do — and those
splits are itemized rather than rounded to a single row.

The list is meant to be checked against the tree, not taken on faith:

```
cd crates/audio/candle-audio-stable-audio-3/tests
grep -c 'Device::Cpu\|Device::new_metal\|Device::new_cuda' *.rs
```

```text
base_guidance.rs:4
chunked_oracle.rs:3
conformance.rs:0
dit_oracle.rs:8
dtype_policy.rs:6
primitive_oracle.rs:6
provider.rs:0
provider_oracle.rs:1
real_snapshots.rs:1
same_oracle.rs:4
sampler_oracle.rs:15
text_oracle.rs:3
variant_binding.rs:0
variant_divergence.rs:6
variant_quality.rs:0
```

That is every site in the directory that names a device, which is a **superset** of mechanisms A and
C — not a match for them. Read it with three qualifications, all of them consequences of rows the
tables below already carry:

- Two mechanism-B rows legitimately name devices, for the reasons given in their table entries:
  `text_oracle.rs:166` asserts `!matches!(device, Device::Cpu)` — naming CPU in order to reject it —
  and `dtype_policy.rs:158-166` is a `cfg(feature = …)` ladder whose subject *is* the per-backend
  dtype policy. Neither is an env read, so neither is mechanism A.
- The weight-free cases called out at the end of this section contribute most of the volume, and are
  out of scope for the tables. 11 of `sampler_oracle.rs`'s 15 are scalar/synthetic `Device::Cpu`
  tensors (the other 4 are its selector); 5 of `primitive_oracle.rs`'s 6 likewise (the sixth,
  `:25`, is its mechanism-C loader); and `dtype_policy.rs:328-336` names all three devices because
  it asserts the policy table itself.
- Do **not** widen the pattern to `SA3_TEST_`. That alternative also matches the render-knob env
  vars, which name no device: `provider.rs`, a pure mechanism-B target, contributes 11 hits that way
  (`SA3_TEST_DURATION`/`STEPS`/`PROMPT`/`SEED`) and would read as a false mechanism-A entry. Use
  `SA3_TEST_METAL\|SA3_TEST_CUDA` if you want the selector sites specifically.

The four zeros are the load-bearing part: `conformance.rs`, `provider.rs`, `variant_binding.rs` and
`variant_quality.rs` name no device at all, which is mechanism B by construction.

**Mechanism A — the three-way env selector.** `SA3_TEST_METAL`, then `SA3_TEST_CUDA` (which panics
without `--features cuda`), then `Device::Cpu`. A requested backend that is unavailable is a hard
failure, never a fallback.

| target | real-weight cases on the selector | lanes that select the target |
|---|---|---|
| `same_oracle.rs` | 11 of 11 (`test_device`) | `same-l-{metal,cuda}` name 4 of the 11 by bare name filter rather than `--test`, and `sa3-medium-metal` re-runs 2 of those 4 for their embedded branch; the other 7 run in no lane (pre-existing) |
| `chunked_oracle.rs` | 2 of 2 (`test_device`) | `same-chunked-metal` names both |
| `sampler_oracle.rs` | 3 of 3 (`test_device`) | `sa3-base-identity-{metal,cuda}` names 2 of the 3; `real_default_sampler_resource_probe` runs in no lane |
| `dit_oracle.rs` | 4 of 6 (`device`) — `small_music_intermediates_and_frozen_v_zero_padding_match`, `real_weights_detect_conditioning_mutations_and_exercise_cfg_apg`, `selected_real_device_prediction_matches_p0`, `selected_real_device_resource_probe` | `sa3-base-identity-{metal,cuda}` names only `real_weights_detect_conditioning_mutations_and_exercise_cfg_apg`; the other 3 run in no lane (sc-15235) |
| `base_guidance.rs` | 2 of 4 (`device`) — the two that build a `StableAudio3Dit` directly | `sa3-base-identity-{metal,cuda}` |
| `variant_divergence.rs` | 1 of 2 (`test_device`) — `single_step_dit_divergence_matches_the_frozen_torch_reference` | `sa3-small-sfx-{metal,cuda}` |

sc-14546 added the selector to `base_guidance.rs`, `sampler_oracle.rs`, `dit_oracle.rs` and
`variant_divergence.rs`. The first three had branched on `SA3_TEST_METAL` alone, so on the CUDA lanes
they silently ran on `Device::Cpu` — including this story's headline gate and the frozen-Torch floor
it derives, both documented as measured "on this exact checkpoint, device and dtype".
`variant_divergence.rs` read no `SA3_TEST_*` variable at all: its DiT half hardcoded `Device::Cpu`
while `sa3-small-sfx-metal` **and** `sa3-small-sfx-cuda` both selected the whole target, so two
real-weight 3.45 GB DiT forwards ran on each runner's CPU and certified neither backend. Its
tolerance is a ±0.02 *absolute* band on a cosine the frozen reference pins into `0.5 … 0.75`, i.e.
about ±3% relative against a ~1e-3 relative accelerator matmul delta on a globally normalized
aggregate, so nothing in it was CPU-calibrated and the switch is safe.

Measured rather than argued: with the switch in place, `cargo test --release -p
candle-audio-stable-audio-3 --features metal --test variant_divergence -- --ignored` passes both
cases on an M-series Mac in 29.39 s. The DiT cosine reads **0.601294** on Metal against a frozen
Torch reference of 0.601294 — agreement to the printed six decimals, well inside the ±0.02 band. The
runtime half is unchanged and reproduces its committed envelope exactly (max `|cos|` 0.062972 against
a 0.15 threshold, min RMS delta 1.290740 against 0.9), as do both discriminating controls
(shared-weight null `cos` = 1.000000 / delta 0.000000; partial-blend control `cos` = 0.250000). The
same target with `SA3_TEST_CUDA=1` and no `--features cuda` fails closed on
`SA3_TEST_CUDA requires --features cuda`, which is the proof the env read is live rather than dead
code.

**Mechanism B — resolved for them, by the loader.** These cases never name a device. They go through
`provider_registry().load(…)` / `load_variant(…)` and inherit `candle_audio::default_device()`, which
is chosen by cargo feature — CUDA under `--features cuda`, else Metal under `--features metal`, else
CPU. On a `--features metal` lane they run on Metal, on a `--features cuda` lane on CUDA, with no env
var involved. Or they touch no tensor at all.

| target | cases | note |
|---|---|---|
| `conformance.rs` | 12 real-weight | registry `load` into `gen_core_testkit::audio_conformance` |
| `provider.rs` | 13 real-weight | registry `load` |
| `variant_quality.rs` | 1 real-weight | registry `load` |
| `variant_binding.rs` | 13 real-weight | Names no device anywhere, and needs none: `load_variant` (`model.rs:1312`) validates the pinned checkpoint identity **without reading weights**, and the remaining cases read `SnapshotLayout` config/headers. No tensor is ever constructed, so there is no device to select |
| `variant_divergence.rs` | 1 of 2 | the runtime half, `music_and_sfx_produce_materially_different_audio_from_the_same_prompt_and_seed` — the same file's DiT half is mechanism A |
| `base_guidance.rs` | 2 of 4 | `the_request_negative_prompt_is_inert_at_guidance_one_and_material_at_the_base_default` goes through the registry; `the_shipped_configs_record_the_operating_point_each_half_of_the_family_defaults_to` reads configs only |
| `text_oracle.rs` | 1 of 2 | `actual_metal_policy_matches_canonical_cpu_f32_oracle` calls `default_device()` **and asserts the result is not `Device::Cpu`** |
| `dtype_policy.rs` | 1 real-weight | its `device()` is a `cfg(feature = …)` ladder, not an env read. Deliberate and unchanged: the target's subject *is* the per-backend dtype policy, so the backend must follow the build, not an env var |
| `real_snapshots.rs` | 1 of 2 | `shipped_dit_config_fails_closed_for_every_unsupported_branch` is config validation with no tensors |

**Mechanism C — deliberately pinned to `Device::Cpu`, with the reason.** These do not honour
`SA3_TEST_*`, and must not. This is the complete list of hardcoded-CPU real-weight sites in the
crate.

| site | reason |
|---|---|
| `provider_oracle.rs` `thirty_second_eight_step_provider_matches_frozen_torch` | Bit-reproduction against a frozen Torch CPU-f32 artifact, not backend certification: portable host-LCG noise so no device RNG can enter, and bounds calibrated on CPU f32 — latent cosine `>= 0.99999`, exact-latent decode `max_abs <= 1e-3` and `mean_abs <= 1e-4`, `deltas_gt_0.1 == 0` across 1,323,000 stereo frames. A reduced-precision accelerator matmul across eight sampler steps plus a full VAE decode would break them. **`sa3-small-music-metal` selects this target while setting `SA3_TEST_METAL`, so that step does not certify Metal** — the job step and the test's doc comment now both say so. Metal small-music coverage comes from that job's conformance, stereo-width and render steps (all mechanism B). There is deliberately no CUDA counterpart. |
| `dit_oracle.rs` `all_six_cpu_f32_predictions_match_p0` | The canonical-precision leg of a deliberate pair: it sweeps all six checkpoints against the frozen P0 artifact in CPU f32, while its twin `selected_real_device_prediction_matches_p0` runs one selected checkpoint against the same artifact on `device()`. Putting the sweep on the selector would delete the CPU-f32 leg rather than add coverage. Already documented in place at `dit_oracle.rs`'s `device()` doc comment before this story. Runs in no lane (sc-15235). |
| `dit_oracle.rs` `all_six_consume_every_dit_and_number_conditioner_tensor_exactly` | A tensor-name consumption audit through a tracking `VarBuilder`. It asserts set equality over tensor names and no numerics at all, so the device is unobservable in the result; CPU is the cheapest. Runs in no lane (sc-15235). |
| `text_oracle.rs` `actual_cpu_f32_fallback_matches_transformers_oracle` | The CPU-f32 fallback path is the subject of the test, named in the test. Runs in no lane (sc-15235). |
| `primitive_oracle.rs` `actual_checkpoints_match_locked_primitive_oracle` | Per-primitive element-wise parity against the locked sc-14536 CPU-f32 `primitives.safetensors`, under per-primitive `max_abs` limits. Same class as `provider_oracle.rs`: the limits are exact-reproduction-grade and calibrated on CPU. Pre-existing; runs in no lane. |
| `real_snapshots.rs` `all_eight_configs_and_real_headers_match` | Parses configs and safetensors headers. `mmap_builders` is constructed but no forward runs, so no kernel executes on any device. Runs in no lane (sc-15235). |
| `variant_divergence.rs` `reference_divergence()` (helper, not a test) | Reads the committed frozen-Torch artifacts and reduces them to host `f64` scalars. No model runs; an accelerator round-trip would change the transport, not the quantity. |

Weight-free cases inside these targets — `sampler_oracle.rs`'s 17 scalar/synthetic cases,
`same_oracle.rs`'s `frozen_upstream_two_stage_model_locks_override_list_execution_order`,
`primitive_oracle.rs`'s `frozen_upstream_missing_branches_match`, `provider.rs`'s and
`conformance.rs`'s and `variant_quality.rs`'s non-`#[ignore]` cases, and `dtype_policy.rs`'s
`compute_policy_resolves_to_full_precision_on_every_backend` (which names all three devices because
it asserts the policy table itself) — construct their own small tensors on `Device::Cpu` and are the
weight-free PR lane in `ci.yml`. They are out of scope for this section.

## CI

- `release/real-weight-models.toml` gains `stable-audio-3-small-music-base` and
  `stable-audio-3-small-sfx-base`, and **all three** base entries gain `download_files` allow-lists.
  The pre-existing `stable-audio-3-medium-base` entry had none and pulled 3.84 GB of `svd_bases.pt`
  training pickles on every run; two more base entries without allow-lists would have taken that to
  6.4 GB per run. No shipped config references the file and `SnapshotLayout::from_dir` names every
  file it opens rather than scanning the directory, so it is unreachable from the loader — the
  exclusion is documented at `weights.rs:17` and asserted in that module's tests.
- Two new job pairs in `real-weights.yml`:
  - **`sa3-base-identity-{metal,cuda}`** provisions all six pinned snapshots and owns the full
    `--test variant_binding` matrix, `--test base_guidance`, `dit_oracle`'s
    `real_weights_detect_conditioning_mutations_and_exercise_cfg_apg`, and the two `sampler_oracle`
    real-weight cases (`all_six_real_p0_pingpong_trajectories_match_stepwise` and
    `real_sampler_cfg_apg_scale_phi_matches_frozen_upstream`) — all four of which previously ran in
    **no lane at all**, and one of which is this story's own fixed-guidance parity gate. It sets
    `SA3_REQUIRE_ALL_SNAPSHOTS: "1"`, because a silently skipped gate on the one job that has every
    snapshot is a gate that runs nowhere. That variable is read only by `variant_binding`, so
    sc-14546 also drops it from the medium jobs, which no longer run that target.
  - **`sa3-small-base-{metal,cuda}`** owns the two small-base providers' conformance, floor
    calibration and defaulted renders.
- `--test variant_binding` is **removed** from the small-sfx and medium jobs: since this story every
  gate in that target needs both sides of a post-trained/base pair, and those jobs provision two and
  five snapshots respectively. Running it there would fail on a missing snapshot — the correct
  failure, but the wrong job.

  Net effect on coverage, stated exactly rather than waved at:

  - On `schedule`, `workflow_dispatch` with `profile=all`, and `profile=audio`, coverage **increases**.
    `sa3-base-identity-*` runs the full 13-case matrix, whereas the step it replaced in
    `sa3-small-sfx-*` had been a *hard failure* since sc-14545: that job never sets
    `SA3_MEDIUM_SNAPSHOT`, and `variant_binding`'s `medium()` helper panics when it is unset.
  - On `workflow_dispatch` with `profile=sa3-small-sfx` or `profile=sa3-medium`, `variant_binding`
    now runs **nowhere**. That is a real narrowing and it is deliberate: neither profile's snapshot
    set can pass the target, so the only thing the old wiring produced on those dispatches was that
    same panic. Reach the matrix through `profile=sa3-base-identity`, `profile=audio` or
    `profile=all`. The `sa3-base-identity-*` jobs are not added to the two narrower profiles'
    conditions because they would drag all six snapshots — including the 10.4 GB medium pair — onto
    a dispatch chosen precisely to avoid them.
- The medium jobs gain `stable_audio_3_medium_base` conformance, floor calibration and its defaulted
  render, because they already provision that 10.4 GB snapshot for the identity gates.
- `sampler_oracle`'s real-weight cases are selected **by name**, not by `--test sampler_oracle --
  --ignored`: the third ignored case in that target is an operator resource probe requiring
  `SA3_RESOURCE_SNAPSHOT` / `SA3_RESOURCE_P0` / `SA3_RESOURCE_SECONDS` and fails without them.

## What is not in scope

LoRA on the base checkpoints — the other reason these registrations exist — is sc-14550. No adapter
surface is opened here: `load_variant` still rejects any `LoadSpec` carrying adapters, quantization,
components or a non-resident offload policy, on all six ids.
