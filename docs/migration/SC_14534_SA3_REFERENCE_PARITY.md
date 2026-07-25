# Stable Audio 3 PyTorch reference parity (`sc-14534`)

This checkpoint freezes the upstream inputs and compact PyTorch reference
tensors used by the Stable Audio 3 port. It is intentionally an offline,
explicit-path workflow: the harness never resolves a repository id, downloads a
file, or derives a path from a local Hugging Face cache.

## Frozen upstream

- Repository: `https://github.com/Stability-AI/stable-audio-3.git`
- Commit: `124e8a799f57a1f665495ecb72e547d0a62867f1`
- Python: `3.12.13`
- Torch / torchaudio: `2.7.1` / `2.7.1`, exactly as declared by the upstream
  `pyproject.toml`
- Transformers: `5.8.0`

The committed run used CPU on arm64 macOS because the managed process had no
Metal device. Flash Attention was therefore unavailable and upstream used its
ordinary PyTorch fallback. This is reference evidence, not a claim about
accelerator performance.

## Immutable snapshots

The generation run verified every path, immutable revision, required
`model_config.json`, `model.safetensors`, `LICENSE.md`, `LICENSE_GEMMA.md`, and
`NOTICE`. Each of the six full SA3 snapshots also had a complete bundled
`t5gemma-b-b-ul2` tokenizer/config/model.

| Environment variable | Repository | Revision |
|---|---|---|
| `SA3_SMALL_MUSIC_SNAPSHOT` | `stabilityai/stable-audio-3-small-music` | `0fef1392cd842149a2b6d445e181c97608faac06` |
| `SA3_SMALL_SFX_SNAPSHOT` | `stabilityai/stable-audio-3-small-sfx` | `ae12755283df9d62ca39a9b050a39a0b607b8c20` |
| `SA3_MEDIUM_SNAPSHOT` | `stabilityai/stable-audio-3-medium` | `27b5a21b791b1b033d193a9e1e3ce78493f102f9` |
| `SA3_SMALL_MUSIC_BASE_SNAPSHOT` | `stabilityai/stable-audio-3-small-music-base` | `eab5ceee5ad9c1ed38800aff30a8e49d1161c539` |
| `SA3_SMALL_SFX_BASE_SNAPSHOT` | `stabilityai/stable-audio-3-small-sfx-base` | `cc5ddb990e30daa68336ac61c140c37c7033ab7c` |
| `SA3_MEDIUM_BASE_SNAPSHOT` | `stabilityai/stable-audio-3-medium-base` | `b32993f73c3bdc3864043a72d8032606bba737c8` |
| `SA3_SAME_S_SNAPSHOT` | `stabilityai/SAME-S` | `fbeb3dcf53a326e5682f38e22e7f740202d44232` |
| `SA3_SAME_L_SNAPSHOT` | `stabilityai/SAME-L` | `41acf79dd242877d6499a1108ca5dba5d5eecfc5` |

The complete file sizes, config hashes, consumed configuration, and path-variable
names are recorded in
[`sa3-reference/manifest.json`](sa3-reference/manifest.json). Absolute local
paths are deliberately not committed. The independent
[`sa3-reference/snapshot-files.json`](sa3-reference/snapshot-files.json) lock
pins the byte size and SHA-256 of all 82 required snapshot payloads. Snapshot
verification and generation re-hash every required model, config, license,
notice, and bundled tokenizer/T5 file against that lock before importing model
code.

## Actual consumed DiT configuration

Upstream `create_diffusion_cond_from_config` consumes the outer model and
diffusion fields recorded under every snapshot's `consumedConfig`: IO/sample
geometry, condition-id lists, distribution-shift options, padding/effective
length flags, the complete DiT constructor dictionary, conditioning
configuration, and autoencoder pretransform configuration. The frozen configs
establish this post-trained/base distinction:

| Checkpoints | Objective | Sample size | DiT | Attention |
|---|---:|---:|---|---|
| Small Music / Small SFX post-trained | `rf_denoiser` | 5,292,032 | width 1,024; 20 layers; 16 heads; 64 memory tokens | RMS QK norm; non-differential |
| Small Music / Small SFX base | `rectified_flow` | 5,324,800 | width 1,024; 20 layers; 16 heads; 64 memory tokens | RMS QK norm; non-differential |
| Medium post-trained | `rf_denoiser` | 16,777,216 | width 1,536; 24 layers; 24 heads; 64 memory tokens | RMS QK norm; differential |
| Medium base | `rectified_flow` | 16,777,216 | width 1,536; 24 layers; 24 heads; 64 memory tokens | RMS QK norm; differential |

All six use 256 latent IO channels, 768-dimensional text/global conditions, a
257-channel local-add condition, learned T5 padding, full/equivalent
length-shift bounds of 256–4,096, padding attention, and effective-length
schedule selection. The manifest preserves the complete dictionaries rather
than reducing future ports to this summary.

Post-trained `rf_denoiser` models select pingpong when the upstream sampler type
is omitted. Base `rectified_flow` configs select Euler by that generic default;
the harness explicitly requests pingpong for base checkpoints so downstream
ports have the same eight-step integrator oracle for every checkpoint. It does
not mislabel that explicit base run as the base model's default.

## Reference tensors

The fixed seed is `14534`, the prompt is
`Warm analog synth pulses, crisp percussion, spacious stereo field, 112 BPM`,
the compact DiT latent length is 16, and the one-step timestep is 0.5.

Each full-checkpoint safetensors file contains:

- tokenizer input ids and attention mask;
- raw T5Gemma encoder `last_hidden_state`;
- the conditioner output after projection and learned-padding handling;
- fixed noise, timestep, and one DiT velocity/prediction;
- initial sampler noise, every one of the eight pingpong step inputs,
  denoised intermediates, and sigmas, plus the final latent.

The SAME-S and SAME-L files contain a deterministic stereo input, encoded
latents, decoded audio, and the input/output of the sole encoder and decoder
`TransformerResamplingBlock`. Every file and every tensor has a SHA-256 record
in the manifest. Tensor hashes cover the exact safetensors payload bytes,
independent of header ordering. Verification requires exactly all eight
component files, their complete tensor inventories, exact safetensors metadata,
shapes/dtypes, contiguous payload ranges, and matching file/tensor hashes.
Downstream parity tests should load the safetensors values for tolerance checks
and use the hashes to detect accidental oracle drift.

## Regeneration

Check out the exact upstream commit and create the throwaway environment:

```bash
git clone https://github.com/Stability-AI/stable-audio-3.git /tmp/stable-audio-3
git -C /tmp/stable-audio-3 checkout --detach 124e8a799f57a1f665495ecb72e547d0a62867f1
UV_PYTHON=/opt/homebrew/bin/python3.12 uv sync \
  --directory /tmp/stable-audio-3 --frozen --no-dev
```

Pass all eight snapshot directories explicitly. These examples are placeholders
for immutable snapshot directories provisioned by the caller; no environment
variable may point at a moving repo root:

```bash
export SA3_SMALL_MUSIC_SNAPSHOT=/models/sa3/small-music/0fef1392cd842149a2b6d445e181c97608faac06
export SA3_SMALL_SFX_SNAPSHOT=/models/sa3/small-sfx/ae12755283df9d62ca39a9b050a39a0b607b8c20
export SA3_MEDIUM_SNAPSHOT=/models/sa3/medium/27b5a21b791b1b033d193a9e1e3ce78493f102f9
export SA3_SMALL_MUSIC_BASE_SNAPSHOT=/models/sa3/small-music-base/eab5ceee5ad9c1ed38800aff30a8e49d1161c539
export SA3_SMALL_SFX_BASE_SNAPSHOT=/models/sa3/small-sfx-base/cc5ddb990e30daa68336ac61c140c37c7033ab7c
export SA3_MEDIUM_BASE_SNAPSHOT=/models/sa3/medium-base/b32993f73c3bdc3864043a72d8032606bba737c8
export SA3_SAME_S_SNAPSHOT=/models/sa3/same-s/fbeb3dcf53a326e5682f38e22e7f740202d44232
export SA3_SAME_L_SNAPSHOT=/models/sa3/same-l/41acf79dd242877d6499a1108ca5dba5d5eecfc5
```

From this repository:

```bash
/tmp/stable-audio-3/.venv/bin/python \
  scripts/reference/sa3_reference.py verify-snapshots

/tmp/stable-audio-3/.venv/bin/python \
  scripts/reference/sa3_reference.py generate \
  --upstream-root /tmp/stable-audio-3 \
  --output docs/migration/sa3-reference \
  --device cpu \
  --components small-music small-sfx medium \
    small-music-base small-sfx-base medium-base same-s same-l

/tmp/stable-audio-3/.venv/bin/python \
  scripts/reference/sa3_reference.py verify-artifacts \
  --output docs/migration/sa3-reference
```

Generation enables deterministic Torch algorithms and forces the Transformers
and Hugging Face Hub offline modes before importing upstream. A missing path,
revision drift, incomplete license/model/T5 payload, upstream SHA drift, Torch
tracked modification or untracked file in the upstream checkout, payload
size/hash drift, exact
Python/Torch/torchaudio/Transformers version drift, truncated component/tensor
inventory, safetensors metadata drift, or artifact mutation fails closed.
