# SAME-L sliding-window autoencoder (`sc-14539`)

This checkpoint records the unregistered Candle-native SAME-L encoder/decoder,
its frozen-upstream parity evidence, and the maximum-duration resource gate.
The source of truth is Stability AI's `stable-audio-3` checkout at
`124e8a799f57a1f665495ecb72e547d0a62867f1`; generation and verification are
offline and accept only explicit immutable paths.

## Runtime shape

SAME-L uses a global packed attention sequence, unlike SAME-S's independent
full-attention chunks. With the default stride 16, `sliding_window=[1,1]`
expands to an inclusive 17-token halo on each side, so one query sees at most
35 keys. Every block:

1. tiles at 1,024 query rows;
2. projects Q only for those rows and K/V only for the bounded halo;
3. applies global absolute RoPE positions to both ranges;
4. evaluates ordinary and differential attention branches sequentially;
5. tiles the feed-forward intermediate before concatenating the global hidden
   state for the next block.

At the medium checkpoint's exact maximum, 16,777,216 audio samples become
4,096 latents and 69,632 packed tokens: 68 query tiles, with at most 1,058 key
rows. A dense F32 differential score branch would require about 433.5 GiB; a
bounded score tile is about 99.2 MiB. The implementation also avoids a roughly
1.99 GiB full fused-QKV projection and tiles the 4,608-wide feed-forward
intermediate to about 18 MiB per query tile.

The schedule is typed as full or band attention. A stride override changes the
padding, grouping, gather, and band together; band mode deliberately does not
inherit SAME-S's unrelated `chunk_size % stride` constraint. Encoder padding is
performed before channel mapping, so the learned mapping bias on padded frames
participates exactly as upstream does. Decoder band mode has a one-latent input
segment and no latent padding.

## Frozen parity fixture

[`sa3-same-l-reference/manifest.json`](sa3-same-l-reference/manifest.json)
pins Python 3.12.13, Torch/torchaudio 2.7.1, Transformers 5.8.0, the three
upstream source files, immutable SAME-L/medium/medium-base revisions, all three
noise paths, and seven cases: standalone and embedded at short, 10-second, and
120-second durations, plus a standalone stride-7 case. Evidence is split into
three files so each remains below GitHub's 100 MiB file limit:

| Artifact | Contents | Bytes | SHA-256 |
|---|---|---:|---|
| [`same-l.safetensors`](sa3-same-l-reference/same-l.safetensors) | 354 primary F32 boundary tensors | 72,828,184 | `b77cdc73eac861f3f8fdd6b29271caf38b5de47198ce63b4dcea7b60f2645e94` |
| [`same-l-extended.safetensors`](sa3-same-l-reference/same-l-extended.safetensors) | 278 embedded-long and stride-7 F32 boundary tensors | 57,273,040 | `256271358a45b42ce4400206874fa2673336e0802fdd0cfc7d0dc1876b053598` |
| [`same-l-outputs-f16.safetensors`](sa3-same-l-reference/same-l-outputs-f16.safetensors) | 14 complete latent/audio outputs | 47,522,248 | `fac3744ffaba6461767f6ad49105f46333e5b611e5c35a1e7b6a3cc68b7a4aab` |

Whole outputs use F16 storage only to keep the largest individual artifact
below 100 MiB; every compact diagnostic boundary remains F32. The manifest
itself is 167,176 bytes with SHA-256
`dd24d788938333ead1abc6958d96e5e257f79adf248d0190177ccdb57339c104`.
The verifier pins all four generated-file hashes; generation cannot bless its
own output.

The frozen CPU reference explicitly disables `flex_attention` and executes the
upstream `_sliding_window_chunked_halo_sdpa` fallback. Input audio and the
encoder-token, SoftNorm, and decoder-token noises use a portable integer
sequence reconstructed independently by Python and Rust.

The fixture numerically executes standalone SAME-L and the embedded medium
autoencoder at every required duration. It additionally streams every embedded
autoencoder byte from the medium and medium-base 9.2 GB checkpoints without
materializing either model:
both canonical autoencoder configs hash to
`e61e27487e452e8a83d4e6277476b4d14666b14a8d5b41a405b1693b3f2bb2bf`,
and all 472 names/shapes/dtypes plus 3,408,509,828 payload bytes hash to
`a91db184266e0a1874ebde54e53ec6c6ac25d27d6c712968ab22418e6b32b405`.

## Numeric evidence

The real F32 loader consumes exactly all 472 autoencoder tensors in standalone
and embedded layouts, including all 24 persisted RoPE buffers. Compact F32
traces cover every encoder and decoder layer at sequence edges, the 17-token
segment boundary, the 1,024-row tile boundary, and the sequence midpoint.

| Case | Layouts executed | Whole latent cosine / max abs | Whole audio cosine / max abs |
|---|---|---:|---:|
| 16,384 samples | standalone + embedded | 0.999999979 / 0.000971556 | 0.999999981 / < 0.000124 |
| 10 s / 441,000 samples | standalone + embedded | 0.999999978 / 0.000978708 | ≥ 0.999999981 / 0.000255793 |
| 120 s / 5,292,000 samples | standalone + embedded | 0.999999978 / 0.000979424 | ≥ 0.999999980 / 0.000533722 |
| stride 7 / 16,384 samples | standalone | 0.999999979 / 0.000976562 | 0.999999964 / 0.000223666 |

The whole-output maximum is dominated by the committed F16 storage floor.
Across F32 diagnostic slices, the worst observed encoder, decoder, and final
audio max-abs values were 0.000179291, 0.000140905, and 0.000009865,
respectively; every slice displayed cosine 1.000000000 at nine decimal places.
The 10-second whole-audio comparison additionally gates reference SNR at
70 dB or better and a three-resolution (512/1024/2048 sample Hann windows,
25% hop) MR-STFT distance at 0.08 or less. Observed standalone/embedded MR-STFT
is 0.073040458/0.056127347; this bound includes the committed F16 whole-output
storage floor.

The non-divisor stride-7 oracle changes the derived band to ±8 packed tokens
and mutation-tests encoder learned-token noise, SoftNorm noise, decoder
learned-token noise, padding, gather selection, packed order, and the three
production noise scales/order.

An independent small-sequence test compares the tiled implementation with a
dense additive-mask oracle for ordinary and differential attention, batch two,
asymmetric left/right bands, and a non-multiple final tile. Allocation-plan
tests cover lengths 1, 17, 18, 35, 36, 1,023, 1,024, 1,025, and the exact
69,632-token maximum.

## Maximum-duration resource evidence

The fresh release-profile F32 probe warms a short encode/decode, then constructs
procedural stereo audio and runs one request-local seeded RNG through the full
encode and decode. It synchronizes the selected backend around load, encode,
and decode; asserts exact shapes; and records process RSS, output checksum, and
PCM SHA-256. Literal 380.0 seconds and the checkpoint's exact 16,777,216-sample
limit are separate runs.

[`resource-evidence.json`](sa3-same-l-reference/resource-evidence.json) locks
the exact CPU records below in a 1,229-byte independent file with SHA-256
`f51bc9cc6fa6b245b915ec5cee8bf654e186ab47a73f42f2d6a705b3cf4d1e11`.
The repository verifier pins that fifth hash and rejects metric, shape, or
output-hash drift.

| Backend / case | Hardware | Load | Encode | Decode | Peak memory | Output evidence |
|---|---|---:|---:|---:|---:|---|
| CPU / literal 380 s | Apple M5 Max, 18 CPU cores, 128 GB | 0.369 s | 255.181 s | 253.865 s | 7,455,899,648-byte process RSS | checksum `-11548.334960938`; PCM `e453aaa90f24121fdbbe33e880be4fc2a92dd219ab6b39ab18d2feb3761cf9d1` |
| CPU / exact maximum | same | 0.355 s | 257.091 s | 296.654 s | 7,203,241,984-byte process RSS | checksum `-20443.292968750`; PCM `a357fb9f39b6dcfce24363f54127838d64087a6c93af4b6480a417f6ff80a809` |
| Metal / literal 380 s | Apple M5 Max, 40 GPU cores, 128 GB, Metal 4 | 0.497 s | 22.094 s | 23.958 s | 7,943,159,808-byte process RSS | checksum `-11548.527343750`; PCM `e1b646cb6b45982dc9d9f616bab41353c512b52b5fe3964e02227edc55cb8b0c` |
| Metal / exact maximum | same | 0.501 s | 24.424 s | 26.234 s | 7,943,143,424-byte process RSS | checksum `-20443.314453125`; PCM `b74f964a6f49aac65b111215ab07240d3c5b374c0f72554003d719af27654754` |
| CUDA / literal 380 s | NVIDIA RTX PRO 6000 Blackwell, 96 GB, driver 596.36 | 1.591 s | 5.874 s | 6.518 s | 6,153 MiB sampled VRAM delta | checksum `-11548.447265625`; PCM `756c01e9b7cd2a25b9bafb47bf2428d770793005264e8fc2ccfe0d7b360bc9a1` |
| CUDA / exact maximum | same | 1.611 s | 5.955 s | 6.525 s | 6,345 MiB sampled VRAM delta | checksum `-20443.207031250`; PCM `10f1612a2d9337097f8304064fbd61a63eeb19e9c60055e2f9317708baa2dfb1` |

CPU verdict: viable for medium decoding. Both the literal and exact-maximum
decodes complete faster than audio duration (0.67× and 0.78× respectively).
Including encode, a complete roundtrip takes 1.34–1.46× audio duration.

Accelerator verdict: viable for medium decoding. Metal completes a full
exact-maximum roundtrip in 50.7 seconds and CUDA in 12.5 seconds, each far below
the 380.4-second output duration. The immutable measurements came from
[real-weight run 30191271527](https://github.com/SceneWorks/inference/actions/runs/30191271527)
at commit `9fd36c8f3b553052a768436f3cd31dd82558881c`; that run emitted one resource
record per case and sampled CUDA memory throughout each foreground process.
Its CUDA step exposed a stale PowerShell `LASTEXITCODE` after both successful
probes, which the workflow now clears explicitly. The final-head merge gate
repeats both cases and requires successful status propagation.

The first real-weight Metal probe also found a backend-specific loader limit:
the ordinary lazy mmap path materialized hundreds of persistent Metal buffers
and exhausted the command-buffer resource table before the final bottleneck
scalar. Standalone autoencoder snapshots now use bounded 256 MiB packed Metal
weight buffers with per-tensor views. Full snapshots, text builders, and
non-Metal devices retain the lazy mmap path, so the fix does not expand
component residency.

The isolated `same-l` manual profile and weekly real-weight workflow provision
the ungated, pinned standalone SAME-L snapshot and run short, 10/120-second,
stride-7, literal-380-second, and exact-maximum gates separately on Metal and
CUDA. These accelerator jobs intentionally use the standalone layout: the same
production kernels serve both layouts, while CPU numeric evidence executes both
standalone and embedded checkpoint namespaces and the streamed identity proof
covers medium-base. The story remains unregistered.

## Regeneration

Run with the frozen upstream environment and four explicit paths:

```bash
/path/to/stable-audio-3/.venv/bin/python \
  scripts/reference/sa3_same_l_reference.py \
  --upstream /path/to/stable-audio-3 \
  --same-l /models/SAME-L/41acf79dd242877d6499a1108ca5dba5d5eecfc5 \
  --medium /models/stable-audio-3-medium/27b5a21b791b1b033d193a9e1e3ce78493f102f9 \
  --medium-base /models/stable-audio-3-medium-base/b32993f73c3bdc3864043a72d8032606bba737c8 \
  --output docs/migration/sa3-same-l-reference

python3 scripts/reference/sa3_same_l_reference.py \
  --verify --output docs/migration/sa3-same-l-reference
```
