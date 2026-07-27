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
   floor from step 1 by at least 10×.

A second case runs the same two facts end to end through `provider_registry().load(…)` on decoded
PCM, which is what covers `GenerationRequest::negative_prompt` actually reaching the conditioner.

CLAP text-audio similarity is deliberately **not** a gate. It is not guaranteed monotonic in CFG, so
a monotonicity assertion would be a flake generator rather than a correctness gate. The hard gate is
fixed-guidance frozen-Torch parity.

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
    `--test variant_binding` matrix, `--test base_guidance`, and the two `sampler_oracle`
    real-weight cases (`all_six_real_p0_pingpong_trajectories_match_stepwise` and
    `real_sampler_cfg_apg_scale_phi_matches_frozen_upstream`) that previously ran in **no lane at
    all** — the second of which is this story's own fixed-guidance parity gate. It sets
    `SA3_REQUIRE_ALL_SNAPSHOTS: "1"`, because a silently skipped gate on the one job that has every
    snapshot is a gate that runs nowhere.
  - **`sa3-small-base-{metal,cuda}`** owns the two small-base providers' conformance, floor
    calibration and defaulted renders.
- `--test variant_binding` is **removed** from the small-sfx and medium jobs: since this story every
  gate in that target needs both sides of a post-trained/base pair, and those jobs provision two and
  five snapshots respectively. Running it there would fail on a missing snapshot — the correct
  failure, but the wrong job.
- The medium jobs gain `stable_audio_3_medium_base` conformance, floor calibration and its defaulted
  render, because they already provision that 10.4 GB snapshot for the identity gates.
- `sampler_oracle`'s real-weight cases are selected **by name**, not by `--test sampler_oracle --
  --ignored`: the third ignored case in that target is an operator resource probe requiring
  `SA3_RESOURCE_SNAPSHOT` / `SA3_RESOURCE_P0` / `SA3_RESOURCE_SECONDS` and fails without them.

## What is not in scope

LoRA on the base checkpoints — the other reason these registrations exist — is sc-14550. No adapter
surface is opened here: `load_variant` still rejects any `LoadSpec` carrying adapters, quantization,
components or a non-resident offload policy, on all six ids.
