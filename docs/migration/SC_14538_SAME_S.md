# sc-14538 — SAME-S patched autoencoder parity

## Source of truth

The implementation is traced to frozen
[`Stability-AI/stable-audio-3`](https://github.com/Stability-AI/stable-audio-3)
commit `124e8a799f57a1f665495ecb72e547d0a62867f1`:

| Upstream file | SHA-256 | Owned behavior |
|---|---|---|
| `stable_audio_3/models/autoencoders.py` | `44939e6b1c2a72690736757cec3f46cb7f8f6b9ccf9161b738daac320bdfa171` | patching, resampling, learned tokens, 3+3 midpoint shift, encoder/decoder |
| `stable_audio_3/models/bottleneck.py` | `1d19a7713dba78de6b87fc003f74a54c40c9ac1604917c827f730363e2214c88` | SoftNorm encode/decode and evaluation-time noise |
| `stable_audio_3/models/transformer.py` | `7436d5d1f040ed74af38899ca57c1eb7b6f3ee5e5c27a3e7402f39f22a52a83a` | differential DyT transformer and RoPE |

The locked standalone snapshot is `stabilityai/SAME-S` revision
`fbeb3dcf53a326e5682f38e22e7f740202d44232`. The embedded namespace proof uses
`stabilityai/stable-audio-3-small-music` revision
`0fef1392cd842149a2b6d445e181c97608faac06`. Their exact file hashes remain in
`docs/migration/sa3-reference/snapshot-files.json`. The standalone checkpoint has exactly 244
F32 tensors: 120 encoder, 120 decoder, and four bottleneck tensors. The all-eight real-header gate
also revalidates every small/medium/base/standalone snapshot before this component is accepted.

The oracle was generated on macOS 26.5.2 arm64 with Python 3.12.13, Torch 2.7.1,
Torchaudio 2.7.1, Transformers 5.8.0, and CPU execution. Rust validation used
rustc 1.96.0 (`ac68faa20`, aarch64-apple-darwin); the Metal gate ran on an Apple M5 Max.

## Runtime surface

`same::SameAutoencoder` is config-driven and deliberately unregistered. SAME-S and SAME-L share
this implementation; sc-14539 adds efficient sliding-band attention to the resampling seam rather
than cloning the autoencoder.

SAME-S executes:

1. Stereo 44.1 kHz audio is zero-tail-patched by 256 samples (`2 -> 512` channels).
2. The encoder maps `512 -> 768` with weight-normalized Conv1d kernel 1.
3. Every 16 input tokens receive one learned token. Two 17-token subchunks make an exact
   34-token attention chunk.
4. Six dim-768/head-64 direct-subtraction differential DyT blocks run as 3+3. After block 2, the
   first and last 17-token subchunks are repeated; blocks 3–5 run on shifted 34-token chunks and
   the 17-token edges are cropped.
5. Each block consumes its persisted `rope.inv_freq`; positions reset to `0..33` for every folded
   chunk.
6. The encoder gathers the final token of each 17-token subchunk, projects `768 -> 256`, and applies
   SoftNorm encode.
7. Decode reverses SoftNorm, projects `256 -> 768`, expands one learned token to 16 outputs per
   latent, runs the same six-block fold, gathers the final 16 tokens, maps `768 -> 512` with
   weight-normalized Conv1d kernel 3, and unpatches.

Production `decode()` and `decode_with_rng()` use one host-seeded, backend-portable RNG stream.
Its observable order is fixed:

1. unit-normal SoftNorm regularization noise, scaled by `0.001` in evaluation mode (`0.05` in the
   explicit training-mode mutation seam);
2. decoder learned-token noise in stage execution order, scaled by that stage's configured `0.01`.

The captureable RNG records unit draws, semantic draw identity, and scale. Equal seeds produce
bit-identical output and captured draws; replaying the captures through the explicit-noise seam is
also bit-identical. Tests reject disabling, zeroing, rescaling, reordering, or changing
evaluation/training scale. `decode()` leaves both stochastic paths active, exactly like upstream.
The same seam preserves SAME-L encoder learned-token noise for the later sliding-attention slice.

## Length and stride policy

The small model's effective raw-audio alignment is 8,192 samples, not 4,096:

- raw patching pads to 256;
- the encoder pads patched time to 32;
- the decoder pads odd latent length to 2.

Therefore 12,288 raw samples encode to four latents and decode to 16,384 samples. A caller that
owns the source length uses `SameAutoencoder::crop_valid_prefix`; the model never guesses that
length or silently drops the padded tail. Tests lock non-256 input, 12,288 samples, odd latent
length, batch size two, and exact valid-prefix cropping.

Stride override is a real runtime path. It requires `variable_stride=true`, exactly one value per
stage, a nonzero stride, and divisibility into `chunk_size`. The real SAME-S checkpoint proves
stride 8 and rejects zero, 7, and wrong-length override lists. A frozen two-stage upstream model
also executes `[2,4]` and `[4,2]`; every mapped, folded, expanded, block, selected, latent, and
decoded tensor proves that override lists are consumed in stage execution order.

## Frozen oracle

`scripts/reference/sa3_same_reference.py` is an offline generator/verifier. It records:

- pre-transformer mapped sequences and first folded 34-token chunks;
- learned-token expansion, each raw transformer-block input, and the post-repeat/refold block-3
  input;
- all six independently named encoder and decoder block outputs;
- final selected subchunks, resampling outputs, latents, and audio;
- both injected decoder noise tensors;
- real SAME-S stride-8 encoder and controlled decoder evidence plus two frozen Torch
  post-block-0 perturbation-sensitivity decodes;
- a complete synthetic two-stage checkpoint and both override-list execution orders;
- exact upstream/runtime/snapshot/file provenance.

The standard-library verifier does not trust manifest self-consistency. It independently locks the
upstream commit and three source hashes, both snapshot revisions and model hashes, music URL,
license, offset, metric definitions and bounds, synthetic config, and immutable artifact
size/hash/count. It parses each safetensors header, requires contiguous payload offsets and exact
metadata, and verifies every tensor name, dtype, shape, and payload SHA-256. The Python mutation
suite changes every provenance class and separately corrupts binary metadata, inventory, shape,
and payload. Runtime loading is independently instrumented against actual safetensors headers:
standalone and embedded namespaces each consume all and only the 244 SAME-S tensors.

### Backend-sensitive stride-8 decoder

Stride 8 changes each folded attention chunk from 34 to 36 tokens. With exact Torch latents and
noise, Candle CPU and Torch agree before attention to `3.576e-6` max. Small backend operation-order
differences are then amplified by this frozen decoder: the recurrent final result measures cosine
`0.962210421`, max absolute error `2.380501270`, relative L2 `0.275782334`, and SNR
`11.188671 dB`. Repeating the same Candle decode is bit-identical.

This is accepted only as a split oracle, never as a replacement for local parity:

- every transformer receives its exact frozen Torch input, including block 3 after edge
  repeat/refold;
- controlled blocks 0–4 are within `1.22070e-4`, block 5 and selected segments within
  `1.045465e-3`, and final projection within `4.42743e-4`; every cosine is `>= 0.9999`;
- the recurrent result must retain cosine `>= 0.95`, max absolute error `<= 2.5`, relative L2
  `<= 0.30`, and SNR `>= 10.5 dB`.

The frozen Torch sensitivity artifact makes the amplification mutation-visible. Injecting the
same unit-normal tensor after block 0 at scales `1e-6` and `3e-6` produces respectively:

| scale | injection max | final cosine | final max abs | relative L2 | SNR |
|---:|---:|---:|---:|---:|---:|
| `1e-6` | `4.537286e-6` | `0.972588239` | `2.293635368` | `0.234568383` | `12.594611 dB` |
| `3e-6` | `1.361186e-5` | `0.948503209` | `3.662863493` | `0.331362506` | `9.593933 dB` |

The original default stride-16 shipped path keeps its `>= 0.999` parity requirement unchanged.

The 10-second stereo music fixture is a 44.1 kHz excerpt of the public-domain US Army Strings
performance of Brahms' *Hungarian Dance No. 5*, distributed by librosa under the Public Domain
Mark 1.0. Source SHA-256:
`8e93ff0182a93168b15346c497b164cb49d2a97bf1e987a1149ea579e914532e`.

Quality uses the valid 441,000-sample prefix:

- `SNR = 10*log10(sum(reference^2) / sum((reference-decoded)^2))`;
- MR-STFT is the mean across `(window, hop)` values `(512,128)`, `(1024,256)`, and
  `(2048,512)` of spectral convergence plus log-magnitude L1;
- STFT uses periodic Hann, `center=true`, reflect padding, and a `1e-7` log floor.

Frozen Torch measured SNR is `7.799532 dB` and MR-STFT is `3.053042`. Acceptance bounds are
SNR `>= 7.6 dB` and MR-STFT `<= 3.055`. The shared Rust STFT measured `7.799539 dB` and
`3.047712`; the small MR-STFT difference is host FFT accumulation, within the explicitly locked
bound.

## Validation

```text
cargo check --locked -p candle-audio-stable-audio-3
cargo test --locked -p candle-audio-stable-audio-3
python3 -m unittest scripts.tests.test_sa3_same_reference -v
SA3_SAME_S_SNAPSHOT=... cargo test --locked -p candle-audio-stable-audio-3 \
  --test same_oracle -- --ignored --nocapture
SA3_TEST_METAL=1 SA3_SAME_S_SNAPSHOT=... cargo test --locked \
  -p candle-audio-stable-audio-3 --features metal --test same_oracle \
  structural_oracle_locks_all_blocks_midpoint_selection_and_noise -- --ignored --nocapture
```

CPU compact parity reaches cosine `>= 0.999999998`; the largest structural absolute error is
`7.37429e-4` at encoder block 5, while final decoded error is `2.8536e-5`. The 10-second
roundtrip's latent and decoded maximum errors are `1.66446e-4` and `6.1899e-5`.
The real-weight Metal structural gate reaches the same cosine floor; its largest structural error
is `1.085043e-3` and final decoded error is `2.0713e-5`.
The newly expanded stride-8 Metal gate was attempted twice in fresh processes but failed while
loading the model with Candle `Failed to create metal resource: Buffer`. That is repeatable local
Metal resource-budget evidence, not a numerical result; CPU and the prior default-stride Metal
evidence remain distinct.

This story does not register a provider and does not implement the outer 128-latent/32-overlap
chunked pretransform; that separately owned composition is sc-14540.
