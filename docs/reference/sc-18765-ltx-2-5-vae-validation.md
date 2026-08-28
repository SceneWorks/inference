# sc-18765 — LTX-2.5 conv video VAE + audio VAE on the existing ports

Epic [sc-18755](https://app.shortcut.com/trefry/epic/18755) §1.2 lists
`vae/ltx-2.5-video-vae-conv-bf16` and `vae/ltx-2.5-audio-vae-bf16` as **drop-in reuse** for the
shipped `mlx-gen-ltx` / `candle-gen-ltx` ports. This document records what that turned out to mean
exactly, measured against the real checkpoints — including the structural differences, which are
real but small, and are handled rather than absorbed.

Sources: `Lightricks/LTX-2.5` snapshot `6c7e5e573ac1667efc83407806fe9b0b93730e60`; reference
implementation `Lightricks/LTX-2` @ `d151147788a9284cca791edc6ce898007e727fe6`; the shipped
`SceneWorks/ltx-2.3-mlx` split rehost as the 2.3 baseline.

---

## 1. Verdict

| Surface | MLX (`mlx-gen-ltx`) | candle (`candle-gen-ltx`) |
| --- | --- | --- |
| conv video VAE (encoder + decoder) | reuse, via the existing convert step | **reuse, no change at all** |
| audio VAE decoder | reuse, via the existing convert step | **reuse, no change at all** |
| vocoder (BigVGAN core + BWE) | reuse, via the existing convert step | **reuse, no change at all** |
| DiffVAE conv **encoder** | reuse (encoder-only) | reuse (encoder-only) |
| DiffVAE decoder (`NADiffusionDecoder`) | not this port — sc-18766 | not this port — sc-18767 |

No model code changed on either backend. What changed is: the MLX converter learned LTX-2.5's
component namespace, the VAE config learned the `latent_log_var` field and the DiffVAE's nested
encoder block, and `LtxVideoVae` gained an encoder-only constructor for the DiffVAE case.

## 2. Structural differences found

### 2.1 The video VAE moved namespace (the only key-level difference anywhere)

LTX-2.3 ships one checkpoint holding the whole stack, so its VAE tensors are namespaced `vae.*`.
LTX-2.5 ships the video VAE as its own file, so the same tensors sit at the file root:

| | LTX-2.3 all-in-one | LTX-2.5 component file |
| --- | --- | --- |
| decoder | `vae.decoder.*` | `decoder.*` |
| encoder | `vae.encoder.*` | `encoder.*` |
| statistics | `vae.per_channel_statistics.{mean,std}-of-means` | `per_channel_statistics.{mean,std}-of-means` |

Below the namespace the tensor names and shapes are identical, key for key: 170 tensors in the 2.5
file map onto exactly the 86 + 86 the shipped `vae_decoder` / `vae_encoder` components consume (the
two statistics are shared by both halves, hence 170 = 84 + 84 + 2). Asserted in
`mlx-gen-ltx/tests/ltx_2_5_vae_conformance.rs::conv_vae_component_carries_exactly_the_2_3_split_key_sets`.

**The audio VAE and vocoder do not differ even in this.** LTX-2.5 keys them `audio_vae.decoder.*`,
`audio_vae.per_channel_statistics.*`, `vocoder.vocoder.*`, `vocoder.bwe_generator.*` and
`vocoder.mel_stft.*` — byte-for-byte the 2.3 spelling, so the shipped sanitizers and the candle
VarBuilder roots consume them untouched.

The audio VAE additionally carries its **encoder** (44 tensors, `audio_vae.encoder.*`), exactly as
2.3 does. Both pipelines only ever decode audio, so those tensors go unread on both backends; the
count is asserted so a restructuring cannot pass silently.

### 2.2 Layout is PyTorch upstream, channels-last in the SceneWorks rehost

This is not a 2.5 change — it is the pre-existing difference between upstream and the
`SceneWorks/ltx-2.3-mlx` rehost — but it decides which backend needs a conversion step:

- **candle** reads PyTorch layout natively (`Conv3d [O,I,kt,kh,kw]`, `Conv2d [O,I,kH,kW]`,
  `Conv1d [O,I,k]`, `ConvTranspose1d [I,O,k]`), so it consumes the upstream 2.5 files as shipped;
- **MLX** reads the channels-last layout the rehost carries, so 2.5 goes through the same
  `convert.rs` transpose the 2.3 path already applies.

Every recorded 2.5 shape is asserted to be the exact PyTorch pre-image of the shipped channels-last
shape, over all 1457 tensors, in
`ltx_2_5_vae_conformance.rs::recorded_2_5_shapes_are_the_pytorch_preimage_of_the_shipped_channels_last_shapes`.

Corroboration: converting the 2.5 components produces split files within ~30 bytes of the shipped
2.3 rehost's file sizes (`vae_decoder` 814 348 487 vs 814 348 463; `vae_encoder` 637 884 271 vs
637 884 303; `audio_vae` 63 836 661 vs 63 836 673; `vocoder` 258 304 533 vs 258 304 501) — identical
tensor payloads, differing only in safetensors header key ordering.

### 2.3 `latent_log_var`: `uniform` (conv) vs `constant` (DiffVAE) — same means either way

The conv VAE declares `latent_log_var: "uniform"`; the DiffVAE's encoder declares
`latent_log_var: "constant"` with `latent_log_var_value: -7.824046010856292`. Read against the
reference (`video_vae.py:246-252, 301-336`):

- both modes size `conv_out` at `latent_channels + 1` — **129 channels in both files**, confirmed on
  the recorded shapes;
- `uniform` splits `[means, logvar]` and broadcasts the single log-variance channel;
- `constant` **drops** the trailing channel and substitutes a hardcoded `approx_ln_0 = -30`;
- both then keep `normalize(sample[:, :latent_channels])`.

So the normalized means — the only thing a deterministic encoder returns — are identical under both
modes. Measured on the real DiffVAE encoder: `constant`-vs-`uniform` max|Δ| = **0.0** (bit-identical).

Two things worth recording:

1. `latent_log_var_value` is **dead config** at this reference pin — the `CONSTANT` branch uses the
   hardcoded `-30`, never the declared `-7.824046010856292`. Asserted, so a future upstream change
   that starts honouring it cannot land unnoticed.
2. `latent_log_var: "none"` is **rejected** by the MLX port. It would leave no log-variance tail, at
   which point the reference's own `torch.chunk(sample, 2, dim=1)` splits the *means* in half and
   returns a half-width latent. No shipped checkpoint declares it, and guessing which reading was
   intended is not this port's call.

The mode is now a parsed, enforced field rather than an assumption: `VideoEncoder::from_weights`
checks the checkpoint's `conv_out` width against the declared mode and errors on disagreement.
Executed control on real weights: declaring `per_channel` against the 129-channel head is refused
with a message naming both widths (129 vs 256).

### 2.4 The DiffVAE reuses the conv encoder verbatim

The `CausalDiffusionVAE` file's encoder is **tensor-identical** to the conv VAE's — same 84 keys,
same shapes, same 129-channel `conv_out`, same `blocks` list, same `patch_size: 4`, same
`pixel_norm` — differing only in the declared log-variance mode. Its config spells the same fields
in a nested form (`vae.encoder.blocks` / `vae.encoder.out_channels` instead of
`vae.encoder_blocks` / `vae.latent_channels`), which `LtxVaeConfig::from_embedded_vae` now handles,
mirroring the reference's `_prepare_video_encoder_kwargs`.

`convert_vae_components` therefore emits the encoder from a DiffVAE file and **refuses to emit a
conv `vae_decoder`** from it — that decoder is an `NADiffusionDecoder` (sc-18766 / sc-18767) and a
conv decoder built from its config would load and render garbage.

### 2.5 Unchanged, and asserted so

- `SpatioTemporalScaleFactors` 32 × 32 × 8 — latent geometry identical, so no downstream tiling or
  geometry math shifts;
- `encoder_blocks` / `decoder_blocks` — identical to the shipped 2.3 block lists, entry for entry;
- `patch_size: 4`, `latent_channels: 128`, `norm_layer: pixel_norm`, `spatial_padding_mode: zeros`,
  `timestep_conditioning: false`;
- audio VAE `ddconfig` — `mel_bins: 64`, `z_channels: 8`, `ch: 128`, `ch_mult: [1,2,4]`,
  `num_res_blocks: 2`, `causality_axis: height`, `mid_block_add_attention: false`, sampling rate
  16 000, stereo;
- vocoder — BigVGAN (`snakebeta` / `AMP1`) core with `upsample_rates [5,2,2,2,2,2]`,
  `upsample_initial_channel 1536`, plus the BWE stage at 16 kHz in → 48 kHz out.

## 3. Measured evidence

### 3.1 MLX / Metal (this Mac, f32, real 2.5 weights)

Conversion of both component files → four split components: **4.9 s**.

| geometry | latent | encode | decode | round-trip PSNR |
| --- | --- | --- | --- | --- |
| 960×544×89 (LTX-2.5 trainer) | `[1,128,12,17,30]` | 11.4 s, peak 18.96 GiB | 10.9 s, peak 20.21 GiB | **58.29 dB** |
| 768×512×25 (SceneWorks bucket) | `[1,128,4,16,24]` | 2.4 s, peak 6.87 GiB | 2.4 s, peak 6.53 GiB | **53.71 dB** |

Audio, latent `T = 93` (≈3.7 s of audio):
`[1,8,93,16]` → mel `[1,2,369,64]` in 0.15 s → waveform `[1,2,177120]` in 1.08 s, peak 3.77 GiB.
The BigVGAN core stage produces **59 040 samples of stereo at 16 kHz = 3.690 s**; the BWE stage
carries the same 3.690 s to 48 kHz. Duration is preserved exactly.

DiffVAE encoder: loads at `latent_log_var: constant`, `constant`-vs-`uniform` means max|Δ| = 0.0.

### 3.2 candle (CPU on this Mac, f32, real 2.5 weights, **unconverted files**)

The candle port memory-maps the upstream 2.5 files as shipped — no conversion, no key remap, no
transpose — and runs both acceptance geometries:

| geometry | latent | encode | decode | round-trip PSNR |
| --- | --- | --- | --- | --- |
| 960×544×89 (LTX-2.5 trainer) | `[1,128,12,17,30]` | 280.0 s | 257.7 s | **58.29 dB** |
| 768×512×25 (SceneWorks bucket) | `[1,128,4,16,24]` | 85.1 s | 66.0 s | **53.71 dB** |
| 768×512×9 | `[1,128,2,16,24]` | 30.4 s | 26.0 s | 45.13 dB |
| 64×64×1 | `[1,128,1,2,2]` | 0.4 s | 0.5 s | 48.11 dB |

**The two backends agree to two decimal places at both acceptance geometries** — 58.29 dB and
53.71 dB on each — despite reaching the weights by completely different routes (MLX through the
channels-last converter on Metal; candle straight off the PyTorch-layout file on CPU). That is a
stronger statement than either number alone: a layout or statistics error on one side would not
reproduce on the other.

Audio, latent `T = 25`: mel `[1,2,97,64]` in 0.94 s → waveform `[1,2,46560]` in 11.13 s; core stage
15 520 samples of stereo at 16 kHz = 0.970 s, carried to 48 kHz by the BWE stage.

**Pending:** the candle runs above are on the **CPU** device — this Mac has no CUDA toolchain, so
none of this exercises the CUDA kernels. The tests are deliberately not CUDA-gated and carry the
acceptance geometries behind `LTX25_FULL=1`, so the CUDA lane runs them unchanged; that run is
outstanding.

### 3.3 The existing sweeps, unmodified, on the converted 2.5 components

`mlx-gen-ltx/tests/vae_decode_sweep.rs` takes `LTX_VAE_DIR` and needs only a directory holding
`vae_decoder.safetensors` + `embedded_config.json`, which is exactly what `convert_vae_components`
emits. It runs against LTX-2.5 with no change:

| run | output | peak | bytes / out-vox |
| --- | --- | --- | --- |
| `LTX_W=960 LTX_H=544 LTX_FRAMES=89` | `[1,3,89,544,960]` in 8.8 s | 19.26 GB | 444.9 |
| `LTX_W=768 LTX_H=512 LTX_FRAMES=25` | `[1,3,25,512,768]` in 1.9 s | 5.99 GB | 654.3 |
| 960×544×89, `LTX_TILE_PX=256 LTX_OVERLAP_PX=32` | same output in 11.9 s | **6.25 GB** | 144.5 |

The budgeted/tiled decode path therefore also works unchanged on 2.5, cutting peak from 19.26 GB to
6.25 GB at the trainer geometry.

### 3.4 Why the *parity* suites were not pointed at 2.5

`vae_parity.rs`, `audio_vae_parity.rs`, `vocoder_parity.rs` (MLX) and `vae_encode_parity.rs`
(candle) compare against goldens dumped from the reference running **LTX-2.3 weights**. Feeding
them LTX-2.5 weights would compare 2.5 outputs against 2.3 reference outputs — a guaranteed,
meaningless failure. They are left pointed at 2.3, where they keep doing their job.

What those suites establish is that *these port implementations* reproduce the reference. That
result carries to 2.5 unchanged, because **no model code changed**: the encoder, decoder, audio
decoder and vocoder are the same functions the 2.3 goldens gate. What 2.5 introduces is new weights
and new packaging, and that is what is validated here — the key/shape/config conformance in §2 plus
the round trips in §3.

The residual risk this leaves is a *shared misreading* of the reference that both backends inherit.
The cross-backend agreement in §3.2 does not fully retire it (both ports were written from the same
reading), but the 2.3 golden gates do: they are dumps of the reference itself, and they pass on the
same code paths. A 2.5-weight reference golden would be strictly stronger and is cheap to add if
sc-18766/18767 stand up a torch reference environment for the DiffVAE anyway.

## 4. What the tests cover

| test | needs weights | what it pins |
| --- | --- | --- |
| `mlx-gen-ltx/tests/ltx_2_5_vae_conformance.rs` (10 tests) | no | key-by-key 2.5↔2.3 diff, layout pre-image over all 1457 tensors, config/block-list equality, DiffVAE encoder identity, converter output key sets, unknown-class refusal |
| `mlx-gen-ltx/src/convert.rs` unit tests (2 added) | no | the LTX-2.3 (`Bundled`) sanitizer mapping the namespace refactor generalized — otherwise covered only by the real-checkpoint byte-parity goldens |
| `candle-gen-ltx/tests/ltx_2_5_vae_conformance.rs` (2 tests) | no | the shipped candle loaders find every 2.5 tensor at the file root, with a `vae.`-namespaced control that must fail |
| `mlx-gen-ltx/tests/ltx_2_5_vae_real_weights.rs` (3 tests, `#[ignore]`) | yes | real-weight round trip at both acceptance geometries, audio VAE + vocoder length/channel/rate, DiffVAE encoder mode equivalence + width-check control |
| `candle-gen-ltx/tests/ltx_2_5_vae_real_weights.rs` (2 tests, `#[ignore]`) | yes | the unconverted 2.5 files through the candle ports, device- and geometry-scaled |

The key-set fixture is `docs/reference/sc-18765-vae-keysets/ltx-vae-keysets.json`, regenerated by
`crates/media/mlx-gen/tools/dump_ltx_vae_keysets.py` (safetensors headers only — no payload bytes).
It is what makes a future upstream VAE change fail a cheap CI test instead of a 42 GB pipeline run.
