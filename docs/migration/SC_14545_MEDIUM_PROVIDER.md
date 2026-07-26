# sc-14545 — `stable_audio_3_medium` provider

Third registered Stable Audio 3 checkpoint, and the first that is not a checkpoint swap.
`stable_audio_3_small_music` (sc-14543) and `stable_audio_3_small_sfx` (sc-14544) share one
architecture; `stable_audio_3_medium` is a different graph — a 1.45B `1536×24` **differential** DiT
over the 852M **SAME-L** autoencoder, with a `16,777,216`-frame ceiling instead of the smalls'
`5,292,032`.

| | value |
|---|---|
| Provider id | `stable_audio_3_medium` |
| Repository | `stabilityai/stable-audio-3-medium@27b5a21b791b1b033d193a9e1e3ce78493f102f9` |
| Root artifact | `model.safetensors`, 9,222,116,660 B, 997 F32 tensors / 2,305,495,793 params |
| — DiT | `model.*`, 522 tensors / 1,453,170,192 params, `embed_dim 1536`, `depth 24`, `heads 24`, `differential: true` |
| — autoencoder | `pretransform.model.*`, 472 tensors / 852,127,457 params (SAME-L) |
| — conditioner | `conditioner.*`, 3 tensors / 198,144 params |
| Text encoder | bundled `t5gemma-b-b-ul2/`, 1,183,022,944 B, 340 BF16 tensors — byte-identical to both smalls |
| Output | 44.1 kHz stereo, `floor(seconds × 44100)` frames |
| Advertised maximum | 380 s |
| Default sampler / steps | Pingpong / 8 |
| Weight licenses | Stability AI Community (composite + root) and Gemma Terms (`t5gemma`) — see `release/model-weight-licenses.json` |

## What was already in place, and what this story had to build

The SAME-L decode path (sc-14539), the outer-chunked decode (sc-14540) and the differential DiT
config branch (sc-14534/sc-14539) all landed before this story. Medium's architecture was
implemented; it had simply never been reachable through a registered provider. What sc-14545 built:

1. **Variant-bound geometry.** `MAX_DURATION_SECS` was a crate-global `120.0` feeding `descriptor_for`,
   and the strict wrapper hard-pinned `685` root tensors and `embed_dim 1024 / depth 20 / heads 16`.
   Both are now per-variant records (`model::VariantShape`, `Variant::geometry`). This matters in
   both directions: loosening the shared check to admit medium would have silently opened the two
   small ids to medium weights.
2. **A per-variant advertised cap.** Medium's `sample_size` is `16,777,216` frames
   (`380.43573696…` s). The descriptor advertises the published `380` s, which is strictly inside
   the geometric ceiling; `tests/conformance.rs` asserts each variant's cap is both servable and
   tight (one second more must not fit).
3. **A typed compute-dtype policy** (`pipeline::ComputeDTypes`) replacing a hard-coded
   `DType::F32, DType::F32` literal, with the F16 path threaded end to end and measured — see
   [Decision 2](#decision-2--dtype-policy).
4. **The rename.** `StableAudio3SmallGenerator` / `StableAudio3SmallPipeline` /
   `validate_small_layout` are now `StableAudio3Generator` / `StableAudio3Pipeline` /
   `validate_layout`.
5. **CI that knows medium exists.** `release/real-weight-models.toml` had no medium entry despite
   five test files reading `SA3_MEDIUM_SNAPSHOT`, and `real-weights.yml` hard-coded
   `SA3_SAME_L_CASE: standalone`, so the embedded-SAME-L oracle case had never run. Both are wired.

## Identity and variant binding

`ModelRegistration::load` receives no provider id, so loading is bound at the registration site
through `model::load_variant(Variant, &LoadSpec)`. Two independent checks run before any tensor is
materialized, and the second runs again on the lazy pipeline path immediately before the tensors are
mmapped (the sc-14544 load-to-use TOCTOU fix):

- **Architecture**, from `Variant::geometry()` — `sample_size`, root/DiT/autoencoder tensor counts,
  `embed_dim`/`depth`/`num_heads`/`differential`, objective, latent and downsampling geometry.
- **Pinned payload** — byte length and SHA-256 of all six consumed files against the pinned
  revision.

Against the two smalls, medium is separated on architecture alone; the hash pin is never reached.
Against its own **base** sibling it is separated by almost nothing, and that is the case worth
recording:

| | `stable-audio-3-medium` | `stable-audio-3-medium-base` |
|---|---|---|
| root tensors | 997 (522 / 472 / 3) | 997 (522 / 472 / 3) |
| root bytes | 9,222,116,660 | 9,222,116,660 |
| `sample_size` | 16,777,216 | 16,777,216 |
| conditioner `repo_id` | `stabilityai/stable-audio-3-medium` | `stabilityai/stable-audio-3-medium` |
| `diffusion_objective` | `rf_denoiser` | `rectified_flow` |

The conditioner `repo_id` — the field that separates the two smalls from each other — discriminates
**nothing** here. The objective check and the SHA-256 pin are the whole gate.
`tests/variant_binding.rs` asserts the base snapshot is rejected under the post-trained id, and
separately that the base *root weights* under the post-trained *config* (so both the `repo_id` and
the objective read correct) are rejected too, which isolates the hash pin.

## Omitted config keys

The epic's rule is that an absent key means the upstream constructor default, never `null` or
`false`. Diffing medium's `model_config.json` against `small-music`'s, medium omits:

| key | medium | small | Rust default |
|---|---|---|---|
| DiT `timestep_features_logsnr` | absent | `false` | `false` |
| `distribution_shift_options.type` | absent | `"full"` | `Full` |
| SAME enc/dec `dyt` | absent | `true` | `true` |
| SAME enc/dec `chunk_size` | absent | `32` | `128` |
| SAME enc/dec `chunk_midpoint_shift` | absent | `true` | `false` |
| SAME enc `conv_mapping` | absent | `false` | `false` |
| SAME **dec** `conv_mapping` | absent | `true` | `false` |

The decoder `conv_mapping` row is a genuine divergence rather than a default that happens to agree:
small explicitly enables it and medium does not, so medium's decoder is structurally different, not
just differently sized. Medium additionally sets `conv_bias: false` where standalone `SAME-L` omits
it — the loader does not consume `conv_bias` (tensor presence is authoritative and both inventories
carry the same 53 biases), so it is inert here, but the two configs are **not** equal and any
"standalone ≡ embedded" claim rests on behaviour, not on config equality. That behavioural claim is
what `same_l_short_standalone_and_embedded_match_every_band_layer` carries, and sc-14545 is the first
story to actually run its embedded branch in CI.

`config.rs` contains an assertion that `DistributionShiftConfig::default() == Full`, and medium's
omitted `type` key rides on exactly that default. A test asserting a default is a false green: the
discriminating evidence is the frozen `docs/migration/sa3-reference/medium-reference.safetensors`
comparison, not the Rust default.

## The three decisions

### Decision 1 — device policy

**CPU is permitted and registered, not rejected.** Measured, not assumed — see
[Measurements](#measurements).

Context: as of sc-15074 `default_device_metal_incompatible()` and the `MetalIncompatible` arm are
deleted; `DevicePolicy` is a one-variant typed seam and `resolve_device` always returns a device.
There is no longer a per-provider device seam at all, so anything medium needed that the smalls did
not would have been a family-wide change.

Medium runs on Metal comfortably — 3–5× realtime at 8 steps across the whole duration range — so no
rejection is warranted there. CPU is slow but not unusable and, critically, is the only lane
available on a machine with no accelerator; returning `Error::Unsupported` there would remove a
working capability rather than describe one. The descriptor therefore advertises no
accelerator requirement, because there is none to advertise honestly.

What is *not* claimed: the descriptor still cannot express "runs, but slowly here". `ModelDescriptor`
has no accelerator-required or CPU-viability field, so a consumer choosing on the descriptor alone
learns nothing about the cost difference. That is a real gap; it is additive to the contract, it is
not needed for this story, and it is filed separately with this story's measured numbers attached.

### Decision 2 — dtype policy

**F32 root and F32 text compute on every backend.** Upstream's `model_half=True`-on-CUDA is
deliberately **not** adopted, on measured evidence.

sc-14545 built the machinery: the dtype boundary is threaded through `dit.rs` (one cast site for
prompt, latents, local conditioning and both Fourier feature maps), `sampler.rs` (the sigma schedule
stays F32 for every value the model sees, and a separate solver-dtype copy drives the arithmetic) and
the guidance math. The graph now runs end to end at F16, which it could not before. It was then
measured on Metal at 30 s / 8 steps, three seeds at each dtype:

| statistic | F32 range | F16 range |
|---|---|---|
| rms | 0.057639 … 0.067437 | 0.069209 … 0.075448 |
| peak | 0.519312 … 0.619903 | 0.525879 … 0.603027 |
| hf emphasis | 0.121821 … 0.150911 | 0.095454 … 0.123614 |
| side ratio | 0.502560 … 0.650129 | 0.373487 … 0.951624 |

F16 is louder on 3/3 seeds, duller on 3/3, and its stereo spread exceeds twice the F32 envelope.
Louder-and-duller is the signature of a decoder losing precision. It is **not conclusive**: at a
fixed seed the F16 and F32 waveforms sit at cosine `0.222` against `0.005` for two F32 renders at
adjacent seeds, so half precision selects a *different draw* rather than perturbing one, and a
different draw legitimately has a different brightness. Three seeds cannot separate the two
explanations — which is itself the finding, and the reason an MR-STFT or SNR "parity bound" against
the F32 render was rejected as an instrument: it would have to be loosened until it admitted an
unrelated take.

An ambiguous measurement is not a licence to ship the change on CUDA, the only backend it would apply
to and the one no hardware was available to measure on, especially when adopting it would re-open the
two already-merged small providers there. The seam therefore resolves to F32 everywhere,
`tests/dtype_policy.rs` keeps the fp16 path executable so it cannot rot, and the split policy worth
trying next — half the 1.45B DiT, keep the 852M SAME autoencoder at F32, isolating whether the
dullness comes from the decoder — is filed with these numbers.

The text side was never part of this decision. sc-14537 pinned BF16-on-disk / F32-compute / one BF16
rounding at the raw-embedding boundary, and `tests/text_oracle.rs` gates it against the frozen
Transformers 5.8.0 oracle on CPU and Metal. Moving T5Gemma to BF16 *compute* would move a surface
with a numeric parity gate behind it for 281 MB of a 10.4 GB resident set.

### Decision 3 — domain metadata

**Documentation-only, explicitly accepted.** Stability tags medium for **both** `music` and
`sound-effects`; the two smalls are single-domain specialists. `Capabilities` has no field that can
carry that and `family` is not overloaded to fake it, so the ids, the crate module doc, and
`descriptor_for`'s doc comment are the entire signal. **No typed domain coverage is claimed.** The
additive contract change is tracked as sc-15041.

Both domains *are* exercised: `tests/variant_quality.rs` renders medium beside `small_music` on a
shipped music `demo_cond` prompt and beside `small_sfx` on a shipped SFX `demo_cond` prompt, and the
side-ratio calibration sweep deliberately mixes both domains — which turned out to matter, because
medium's stereo image on `"Dog barking next to a waterfall"` collapses to `1.2e-4` at two of five
seeds while every music prompt stays above `2.6e-1`. A floor calibrated on music alone would have
gated honest sparse-SFX output on this id.

## Measurements

All Metal figures on an Apple M5 Max / 128 GB, `--release`, 8 steps, seed 42, prompt
`"Meditative lo-fi ambient piano jazz, soft acoustic drum kit"`.

### Long-form renders (Metal)

| seconds | frames | wall clock | realtime factor |
|---:|---:|---:|---:|
| 120 | 5,292,000 | 37.258 s | 3.22× |
| 300 | 13,230,000 | 56.817 s | 5.28× |
| 380 | 16,758,000 | 91.986 s | 4.13× |

Process peak resident set across all three renders plus the load: **19,380,813,824 B (18.05 GiB)**.
That covers the 10.4 GB of packed Metal weight buffers plus the 380-second activation and decode
working set. Every render satisfied exact `floor(seconds × 44100)` framing, finite and clamped PCM,
non-silent output, a genuine two-channel image, and neither the white-noise nor the pure-tone
degeneracy gate.

### CPU verdict — runnable, roughly an order of magnitude slower than Metal

Same machine, `--release`, no `metal` feature, so `candle_audio::default_device()` resolves to
`Device::Cpu`. Wall clock is the whole test, which includes the cold start (snapshot load plus the
two SHA-256 passes over 10.4 GB of pinned files — measured at ≈ 42.7 s here, against ≈ 6.9 s over the
smalls' 3.45 GB).

| duration | steps | wall clock | peak RSS |
|---:|---:|---:|---:|
| 5 s | 1 | 49.18 s | 21.02 GB |
| 20 s | 1 | 68.58 s | 20.98 GB |
| 5 s | 8 | 60.24 s | 21.02 GB |
| 30 s | 8 | 98.17 s | 21.02 GB |

Netting the cold start out of the 30 s / 8-step point leaves ≈ 55.5 s of generation, against ≈ 5.3 s
for the same configuration on Metal (the 25-render calibration sweep completes in 161.79 s including
one load). **CPU is ≈ 10× slower**, i.e. ≈ 0.54× realtime at 30 s / 8 steps, where Metal runs at
≈ 5.7× realtime. Extrapolating the ratio, a 380-second render that takes 92 s on Metal would take
≈ 16 minutes on CPU.

That is slow, and it is not unusable — so CPU stays registered and no lane returns
`Error::Unsupported`. Removing it would delete the only lane available on a machine without an
accelerator, in exchange for a number the descriptor cannot express anyway.

Two deployment facts worth stating plainly, because they are the real constraint rather than the
wall clock: CPU peak resident is **21.0 GB**, *higher* than Metal's 18.05 GB, and the cold-start
authentication is ≈ 42.7 s on this machine every time a generator is constructed.

### Cross-checkpoint divergence (Metal, 30 s / 8 steps, three seeds per domain)

| domain | pair | seed 42 | seed 7 | seed 2026 |
|---|---|---:|---:|---:|
| music | medium vs `small_music` | 0.083488 | 0.114087 | 0.036770 |
| sfx | medium vs `small_sfx` | 0.262361 | 0.281723 | 0.206281 |

Worst 0.281723; the shipped bound is `0.45`, 1.60× above it and 2.22× below the `1.0` a mis-wired
registration produces. Medium's own two seeds sit at 0.0053 (music) and 0.0039 (SFX), three orders
below the cross-checkpoint values.

Worth recording because it contradicts the obvious expectation: `variant_divergence.rs`'s `0.15`,
calibrated for the *same-architecture* music/SFX pair, rejects the entire SFX row here even though
medium against a small is a *larger* architectural gap. Both takes on
`"Dog barking next to a waterfall"` are sparse and near-mono, and their shared near-silence lifts the
cosine on its own — the metric is responding to prompt sparsity, not to the weights. Importing the
threshold rather than measuring it would have produced a red gate on honest output.

### Side-ratio calibration (Metal, 30 s / 8 steps, 5 prompts × 5 seeds)

`min_global = 1.20543e-4`, `min_median_window = 1.02879e-4`. The shipped floor is `5e-5`, a 2.06×
margin — the same order the SFX specialist uses, and for the same reason: the distribution is
bimodal and a floor inside it would flake on honest output.

## Tests

| file | what it gates |
|---|---|
| `tests/variant_binding.rs` | medium ↔ both smalls in both directions; medium-base under the medium id; base root under the medium config; standalone SAME-L under every id; medium's post-load root swap, with the restore control |
| `tests/conformance.rs` | full `gen_core` audio conformance, concurrent RNG isolation, and the per-variant advertised-cap assertion |
| `tests/provider.rs` | the 30 s render gate, the two-domain side-ratio sweep, and the timed long-form renders |
| `tests/variant_quality.rs` | both-domain coverage, the seed-swept cross-checkpoint divergence bound, and its weight-free Gram-Schmidt control |
| `tests/dtype_policy.rs` | the resolved policy, the fp16 measurement, and the envelope check's own degradation controls |

### What is deliberately not claimed

Medium is **not** claimed to be perceptually better than `small_music`. The story's original wording
asked for "audibly higher quality"; no objective metric that fits in a test can carry that, because
`MR-STFT` and SNR measure agreement with a reference and two different checkpoints rendering the same
prompt are supposed to disagree. A perceptual claim needs a pinned blinded protocol (ABX or MOS-style,
multiple listeners, held-out prompts), which is separate work and was not run. What is claimed is the
capability difference, which is objective and enforced: 380 s against 120 s, SAME-L against SAME-S,
and both domains against one.
