# Stable Audio 3 DiT parity evidence (`sc-14541`)

This directory locks the unregistered Candle DiT port to upstream commit
`124e8a799f57a1f665495ecb72e547d0a62867f1`. The 3,075,888-byte
`dit-intermediates.safetensors` artifact contains 20 tensors and has SHA-256
`3e91409679043ac9c825eb7a1d28ea57989796e5a701d82238a9951692ad2083`.
`manifest.json` additionally locks the five owning upstream source files, the exact
small-music snapshot/config, the P0 artifact, runtime versions, tensor metadata, and
every tensor payload.

Regenerate offline with the pinned Python environment and explicit immutable paths:

```console
python scripts/reference/sa3_dit_reference.py generate \
  --upstream /path/to/stable-audio-3-at-124e8a7 \
  --snapshot /path/to/stable-audio-3-small-music-at-0fef139 \
  --output docs/migration/sa3-dit-reference
python scripts/reference/sa3_dit_reference.py verify \
  --output docs/migration/sa3-dit-reference \
  --upstream /path/to/stable-audio-3-at-124e8a7
```

The oracle follows the shipped path exactly: direct F32 Expo timestep features;
duration as both the final cross-attention row and the AdaLN global input; no DiT
cross mask; 64 memory tokens and global RoPE; per-block biased
`[inpaint_mask,inpaint_masked_input]` local MLPs; `sigmoid(1-g)` AdaLN gates;
trained pre/post 1x1 residual convolutions; and frozen CPU padding that zeros V but
does not remove K from the softmax. The partial-padding fixture proves this behavior
changes the valid prefix.

## Real-weight parity

All measurements use F32, P0 seed 14534, prompt conditioning from the committed P0
artifact, latent shape `[1,256,16]`, `seconds_total=0.25`, and `t=0.5`.

| Snapshot | Attention | CPU cosine | CPU max abs |
|---|---:|---:|---:|
| small-music | ordinary | 1.000000000 | 0.003509521 |
| small-sfx | ordinary | 1.000000000 | 0.004028320 |
| small-music-base | ordinary | 1.000000000 | 0.002990723 |
| small-sfx-base | ordinary | 1.000000000 | 0.004150391 |
| medium | differential | 1.000000000 | 0.000022173 |
| medium-base | differential | 1.000000000 | 0.000013590 |

Real Metal short parity was also run for one ordinary and one differential
checkpoint: small-music cosine `1.000000000`, max abs `0.003280640`; medium
cosine `1.000000000`, max abs `0.000021458`. The managed sandbox cannot
enumerate a Metal device, so these tests were run as approved unsandboxed
processes.

Expo frequencies are deterministically constructed as host F32 constants in the
same order as upstream (`linspace`, affine log-frequency ramp, `exp`), then copied
to the selected device before the argument/trigonometric operations. Running
`exp` separately on CPU and Metal increased the cross-backend final envelope to
`0.038879395`; the shared F32 frequency table reduces it to the measured
`0.003509521` CPU / `0.003280640` Metal envelope while retaining the frozen
cos-then-sin oracle. This is a constant configuration table, not model state.

No CUDA runtime or `nvcc` exists on this host. Local `--features cuda` compilation
therefore stops in `cudarc` before compiling this crate; draft-PR CI is the
compile source of truth. CUDA numerical ownership remains with `sc-14552`.

## Representative resource probes

Each row is a synchronized fresh process with one model load and one raw DiT
forward. Peak RSS is Darwin `ru_maxrss` and includes memory-mapped checkpoint
pages, not only live tensor allocations.

| Device | Snapshot | Latent length | Load (s) | Forward (s) | Peak RSS (bytes) |
|---|---|---:|---:|---:|---:|
| CPU | small-music | 1292 | 0.208451 | 3.850479 | 3,685,023,744 |
| CPU | small-music-base | 1300 | 0.216077 | 4.045930 | 3,683,221,504 |
| CPU | medium | 4096 | 0.613632 | 36.188848 | 11,645,304,832 |
| Metal | small-music | 1292 | 0.220343 | 0.294032 | 3,692,462,080 |
| Metal | small-music-base | 1300 | 0.214809 | 0.295658 | 3,692,740,608 |
| Metal | medium | 4096 | 0.639552 | 3.267900 | 11,646,025,728 |

These are backbone probes, not sampler throughput claims. They do not load the T5
decoder, an SVD basis, the autoencoder, or duplicate DiT weights.
