# Stable Audio 3 small-music provider (`sc-14543`)

This checkpoint registers the complete Candle text-to-audio graph as
`stable_audio_3_small_music` in every shipped audio bundle. The implementation
is pinned to:

- upstream `Stability-AI/stable-audio-3` commit
  `124e8a799f57a1f665495ecb72e547d0a62867f1`;
- `stabilityai/stable-audio-3-small-music` revision
  `0fef1392cd842149a2b6d445e181c97608faac06`;
- the snapshot-local encoder-only T5Gemma weights, config, tokenizer JSON, and
  tokenizer model.

The shared family preparer recognizes complete Stable Audio 3 full and
standalone dense layouts. The registered small-music loader additionally
authenticates the byte length and SHA-256 of every consumed root/T5
config, weight, and tokenizer file against the revision above. It rejects a
single file, quantization, alternate precision, adapters, control inputs,
external text encoders/components, identity conditioning, and sequential
offload before inference starts.

## Registered contract

The descriptor reports family `stable_audio_3`, backend `candle`, 44.1 kHz
stereo audio, a 120-second maximum, negative prompts, and guidance. It exposes
the four native sampler paths (`pingpong`, `euler`, `rk4`, and `dpmpp`) and the
three mapped guidance methods (`cfg`, `apg`, and `cfg_rescale`).

The frozen defaults are 120 seconds, eight steps, guidance 1, Pingpong, full
APG, LogSNR schedule rate 0, six seconds of padding headroom, and SAME-S outer
decode chunking with `C=128` and `O=32`. For a 30-second request this is:

- 1,589,248 padded samples and 388 latent positions;
- 387 valid latent positions;
- chunk starts `[0, 96, 192, 260]`;
- 1,323,000 final stereo frames / 2,646,000 interleaved `f32` samples.

T5Gemma right-truncates or learned-pads to 256 tokens. Its projected prompt
sequence is concatenated with the seconds token, and local conditioning is an
all-zero `[B,257,T]` tensor. The post-trained checkpoint uses CFG/APG; it is not
the base-model pretraining path.

Request validation rejects empty prompts, unsupported BPM/key/lyrics or other
conditioning, duration/step/guidance values outside the descriptor, nonzero APG
momentum, and guidance fields attached to the wrong method. Cancellation is
checked through text layers, DiT blocks, sampling steps, and every outer SAME
chunk. Progress emits each sampling step followed by one decoding event.

One request-local seeded RNG is consumed in frozen order: initial latent noise,
every Pingpong draw including the terminal draw, then each SAME chunk's
SoftNorm and learned-token noise. Same-seed sequential and concurrent requests
are byte-identical; alternate seeds differ.

## Frozen connected oracle

[`sa3-small-music-provider-reference/manifest.json`](sa3-small-music-provider-reference/manifest.json)
locks the exact upstream source hashes, model/config hashes, Python 3.12.13,
Torch/torchaudio 2.7.1, Transformers 5.8.0, canonical CPU F32 text policy,
30-second request, and all 17 stochastic draws.

| Artifact | Contents | Bytes | SHA-256 |
|---|---|---:|---|
| [`provider-output.safetensors`](sa3-small-music-provider-reference/provider-output.safetensors) | sampled latents, first direct SAME chunk, and full 30-second stereo audio | 9,883,912 | `0b217e13ba379c714697ac3f34443d0c89c1ad3b2302d885c1c3d7911dc0376f` |
| [`manifest.json`](sa3-small-music-provider-reference/manifest.json) | authenticated provenance, request, compute policy, geometry, and draw order | 3,788 | `a893ff4040612942d217dbaa369566909db26bf88d81499ee231834a6acbc982` |

The final real-weight Candle CPU comparison is:

| Boundary | Cosine | Max abs | Mean abs |
|---|---:|---:|---:|
| sampled latents | 0.999999454 | 0.022943586 | 0.001335185 |
| first direct SAME chunk | 0.999999999 | 0.000114650 | 0.000011200 |
| exact-latent full decode | 0.999999977 | 0.000311613 | 0.000039040 |
| complete provider audio | 0.999992223 | 0.061759830 | 0.000657751 |

No exact-decode or full-provider sample exceeds an absolute delta of 0.1. The
oracle exposed and now guards the padding-mask seam: each latent validity value
must cover its complete 4,096-sample codec span.

## Runtime and bundle gates

The registered CPU production path rendered a 30-second, eight-step PCM16 WAV
with 1,323,000 stereo frames and SHA-256
`22e353663fe1d91ab133285b987cc59df83b789d6170be80d839514cde261f85`.
The real Metal path rendered the same geometry on an Apple M5 Max with SHA-256
`17fed179eccdd638eeeb96986c5272852f2a46fa9d73712b8231641669c2fc9e`.

The full graph consumes 685 root tensors plus the T5 encoder. Ordinary
component loaders remain lazy. The registered Metal full-pipeline loader alone
coalesces those used weights into bounded shared buffers, avoiding Metal's
persistent-resource ceiling without changing CPU/CUDA mmap behavior or loading
the unused T5 decoder.

`candle-audio-catalog` composes the provider, preparer, and composite license
rows into `runtime-cpu`, `runtime-macos`, and `runtime-cuda`. The
`sa3-small-music` real-weight profile runs exact-head Metal and CUDA provider
conformance, in-process concurrent RNG isolation, and a real 30-second
eight-step WAV on each accelerator. The CUDA lane is required because a machine
without `nvcc` cannot compile or execute the CUDA configuration.

The root weights are governed by the Stability AI Community License, including
its revenue threshold and prohibited-use terms. The bundled T5Gemma component
is separately attributed under the Gemma Terms and Prohibited Use Policy. Both
rows are included in `release/model-weight-licenses.json`.

## Verification

```bash
python3 scripts/reference/sa3_small_music_provider_reference.py \
  --verify --output docs/migration/sa3-small-music-provider-reference

SA3_SMALL_MUSIC_SNAPSHOT=/models/stable-audio-3-small-music/0fef1392cd842149a2b6d445e181c97608faac06 \
  cargo test --locked -p candle-audio-stable-audio-3 \
    --test provider_oracle thirty_second_eight_step_provider_matches_frozen_torch \
    -- --ignored --nocapture

SA3_SMALL_MUSIC_SNAPSHOT=/models/stable-audio-3-small-music/0fef1392cd842149a2b6d445e181c97608faac06 \
SA3_TEST_DURATION=30 SA3_TEST_STEPS=8 \
  cargo test --locked -p candle-audio-stable-audio-3 --features metal \
    --test provider connected_short_generation_is_stereo_finite_and_exact_length \
    -- --ignored --nocapture
```

Regeneration additionally requires the frozen upstream checkout and its pinned
Python environment:

```bash
/path/to/stable-audio-3/.venv/bin/python \
  scripts/reference/sa3_small_music_provider_reference.py \
  --upstream /path/to/stable-audio-3 \
  --snapshot /models/stable-audio-3-small-music/0fef1392cd842149a2b6d445e181c97608faac06 \
  --output docs/migration/sa3-small-music-provider-reference
```
