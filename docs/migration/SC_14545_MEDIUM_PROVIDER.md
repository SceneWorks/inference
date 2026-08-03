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

**The F16 path did not load on CUDA at all, and lighting the dark Metal lane found the same bug.**
The first real-weight run of these newly-wired jobs came back red twice, both times inside the SAME
autoencoder load, and both times for one reason: `bottleneck.noise_scaling_factor` is persisted as a
genuinely empty `[1, 0, 1]` buffer (zero bytes, present in `stabilityai/SAME-L` and in medium's
embedded copy alike), and Candle materializes every mmapped tensor by copying host bytes to the
device and then casting. Both halves are degenerate at zero elements, and each accelerator failed a
different half:

| lane | half that failed | symptom |
|---|---|---|
| Metal, embedded SAME-L, F32 | the copy — `newBufferWithBytes:length:0` returns nil | `Metal error Failed to create metal resource: Buffer` |
| CUDA, medium at `root = F16` | the cast — `grid_dim.x = 0usize.div_ceil(1024) == 0` | `DriverError(CUDA_ERROR_INVALID_VALUE)` |

The asymmetry explains why neither was caught earlier. Standalone SAME-L on Metal goes through
`packed_metal_builder`, which concatenates on the host and hands out `narrow` views, so an empty
tensor is a zero-length view into a non-empty buffer — immune. And CUDA at F32 never casts, so it
loaded fine; only F16 launched the empty kernel. `ZeroElementSafeMmap` in `weights.rs` serves
zero-element tensors with `Tensor::zeros` instead, which both backends accept and neither routes
through a copy or a kernel; the persisted shape is still verified from the safetensors header, so
the inventory check the tensor exists for is unweakened.

This matters for the record beyond the fix: **on CUDA the F16 path was not merely unmeasured, it did
not load.** "The graph runs end to end at F16" was demonstrated on Metal and, until this fix, not
demonstrated on CUDA at all. The crate's unit tests now run under `--features metal` and
`--features cuda` on the two real-weight jobs specifically so
`zero_element_tensors_load_on_the_target_device` executes on the hardware where it discriminates —
it is a no-op on the CPU lane.

**What stays F32 regardless of `root`, corrected.** An earlier draft of the `ComputeDTypes` doc
claimed `same.rs` forces F32 RMS/DyT statistics. It does not: `same.rs` contains no `DType::F32` at
all, builds its norms with `NormConfig { eps: 1e-3, ..Default::default() }` whose `force_fp32` is
`false`, and medium's SAME-L config sets `dyt: true` — and `Norm::forward` returns through the
`DynamicTanh` branch *before* the `force_fp32` upcast is reached, so DyT would not honour the flag
even if it were set. What is genuinely pinned F32 is the DiT's block norms (`config.rs` rejects any
config whose `norm_kwargs.force_fp32` is not `true`), RoPE (`inv_freq` requested at `DType::F32`
explicitly), and the sigma schedule. This matters for the follow-up: a DiT-F16 / SAME-F32 split is
not merely a memory choice, it is the only arrangement in which SAME's statistics stay F32, and
sc-15151 now records that.

**The envelope check's slack was `3.0` and is now `2.5`.** Review showed the `3.0` was chosen to
clear the widest observed excursion and its control used a synthetic reference whose envelope was far
narrower relative to level than the measured F32 envelopes — so the control's rejections did not
transfer to the shipped gate. Driving the control from the committed envelopes instead, `3.0` admits
a halved level (0.028820 against an allowed low of 0.028245) and a high end cut to a third (0.040607
against 0.034551). At `2.5` all three named degradations are rejected, and the measured F16 envelope
is still admitted (its binding `side_ratio` excursion is 2.043 widths). The control now brackets the
constant from both sides — degradations rejected, milder versions admitted — which pins it into
roughly `[2.09, 2.79)` and fails at `3.0`. The gate's actual rejection points are documented on the
constant: below 57.5% (rms), 51.6% (peak), 40.3% (hf emphasis) and 26.6% (side ratio) of the
committed F32 minimum. It is a gross-degradation gate, not a fine parity bound, and says so.

The text side was never part of this decision. sc-14537 pinned BF16-on-disk / F32-compute; CPU keeps
the raw output F32, while Metal and CUDA apply one BF16 rounding at the raw-embedding boundary before
F32 conditioning. `tests/text_oracle.rs` gates that policy against the frozen Transformers 5.8.0
oracle. Moving T5Gemma to BF16 *compute* would move a surface with a numeric parity gate behind it
for 281 MB of the exact 10,443,755,936-byte medium artifact pin set.

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

| seconds | frames | wall clock (3 runs) | realtime factor |
|---:|---:|---:|---:|
| 120 | 5,292,000 | 35.767 – 37.258 s | 3.22 – 3.36× |
| 300 | 13,230,000 | 43.388 – 56.817 s | 5.28 – 6.91× |
| 380 | 16,758,000 | 56.711 – 91.986 s | 4.13 – 6.70× |

Wall clock is reported as a range across three runs on the same machine rather than as a point. The
380-second render varied by 1.6× between the fastest and slowest run with nothing changed but
machine load, so a single figure would overstate the precision. Nothing in this PR gates on it.

Process peak resident set across all three renders plus the load: **19,380,846,592 B (18.05 GiB)**,
reproduced at 19,380,813,824 B on the previous run — a 32 KB spread. That covers the
10,443,755,936-byte pinned artifact set in packed Metal weight buffers plus the 380-second
activation and decode working set. Darwin's `peak
memory footprint`, which excludes clean file-backed pages, is **17,463,895,112 B (16.26 GiB)**.

**How this is measured, and why the first figure survived re-measurement.** Adversarial review
flagged that the original CI step timed `/usr/bin/time -l cargo test …`, whose `ru_maxrss` covers the
whole reaped child tree and would therefore conflate the release build with the render. The step now
builds with `--no-run`, resolves the `provider` target's own executable from cargo's JSON artifact
stream, and times only that binary — so the reported peak is unambiguously the render's. Re-measuring
that way reproduced the committed figure to within 32 KB, i.e. the original number was *not* in fact
inflated by the build: on Darwin `getrusage(RUSAGE_CHILDREN)` reports the maximum over children
rather than their sum, and the render's own 18.05 GiB exceeds any rustc or linker peak. The measured
value did not change; what changed is that it is now measured in a way that cannot be wrong.

Every render satisfied exact `floor(seconds × 44100)` framing, finite and clamped PCM, non-silent
output, a genuine two-channel image, and neither the white-noise nor the pure-tone degeneracy gate.

### CPU verdict — runnable, roughly an order of magnitude slower than Metal

Same machine, `--release`, no `metal` feature, so `candle_audio::default_device()` resolves to
`Device::Cpu`. Wall clock is the whole test, which includes the cold start (snapshot load plus the
two SHA-256 passes over 10,443,755,936 bytes of pinned files — measured at ≈ 42.7 s here, against
≈ 6.9 s over the smalls' 3.45 GB).

| duration | steps | wall clock | peak RSS |
|---:|---:|---:|---:|
| 5 s | 1 | 49.18 s | 21.02 GB |
| 20 s | 1 | 68.58 s | 20.98 GB |
| 5 s | 8 | 60.24 s | 21.02 GB |
| 30 s | 8 | 98.17 s | 21.02 GB |

Netting the cold start out of the 30 s / 8-step point leaves ≈ 55.5 s of generation, against ≈ 5.3 s
for the same configuration on Metal (the 25-render calibration sweep completes in 161.79 s including
one load). **CPU is ≈ 10× slower**, i.e. ≈ 0.54× realtime at 30 s / 8 steps, where Metal runs at
≈ 5.7× realtime. Extrapolating the ratio gives a **10–16 minute estimate** for a 380-second CPU
render. That number is inferred from small-model CPU throughput and the measured 57–92 s Metal
render; it is not a measurement of a 380-second CPU render.

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
margin — the same *margin ratio* the SFX specialist uses, and for the same reason: the distribution
is bimodal and a floor inside it would flake on honest output.

**The margin ratio is the same; the gate strength is not, and the difference is stated rather than
implied.** 2.06× applied to a measured minimum an order of magnitude lower puts medium's absolute
floor *below* the `1e-4` sc-14544 explicitly tightened away from as "equivalent to: the side signal
is not exactly zero". Concretely, the `near_mono` control that the SFX floor was tightened to
reject — one channel duplicated with a ~77 dB-down alternating differential, ≈ 1.4e-4 — **passes**
medium's floor. So medium's floor is a near-mono detector with a discrimination point around -86 dB,
strictly stronger than "not exactly zero" but materially weaker than the SFX specialist's.

That weakness is forced: medium renders near-mono sparse SFX and wide music under one id, and any
floor with SFX-grade strength (`2e-4`) would reject medium's own honest output on
`"Dog barking next to a waterfall"` at two of five seeds. A generalist checkpoint cannot have one
floor that is both calibrated and strong. This PR chose calibrated.
`the_medium_side_ratio_floor_is_a_near_mono_detector_not_a_width_bar` in `tests/provider.rs` brackets
the discrimination point with a weight-free control it rejects (≈ 4.2e-5) and one it admits
(≈ 6.9e-5), and separately asserts that the sc-14544 `near_mono` control (≈ 1.39e-4) passes here
while failing the SFX floor — so none of the above is prose-only.

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
prompt are supposed to disagree. What is claimed is the capability difference, which is objective and
enforced: 380 s against 120 s, SAME-L against SAME-S, and both domains against one.

**The "audibly higher quality" wording is retired** (sc-15178), not merely left unclaimed. What a
rigorous perceptual claim would take is now pinned rather than gestured at:
[`SC_15178_SA3_LISTENING_PROTOCOL.md`](SC_15178_SA3_LISTENING_PROTOCOL.md) specifies the blinded ABX
+ preference design, the held-out LUFS-level-matched stimulus set, the same-checkpoint validity
control, and a panel size and analysis pre-registered from a stated effect size. It also fixes the
rule that makes the result interpretable — ABX answers *discriminability* and gates the preference
question, so a failed ABX means the preference result is not reported.

That protocol is **designed but not executed**; the listening itself needs a human panel and is
tracked as **sc-15377**. The wording is reinstated only if that run substantiates it, and never by a
metric: no test here may be changed to assert perceptual superiority from an agreement metric.
