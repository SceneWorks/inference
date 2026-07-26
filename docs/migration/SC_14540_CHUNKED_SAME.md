# SAME outer long-duration chunking (`sc-14540`)

This checkpoint records the unregistered, shared SAME-S/SAME-L outer
encode/decode chunker. Its source of truth is Stability AI's
`stable-audio-3` checkout at
`124e8a799f57a1f665495ecb72e547d0a62867f1`, specifically
`AudioAutoencoder.encode_audio` and `decode_audio`.

Despite the historical story title, frozen upstream does not use overlap-add
or a crossfade. It writes hard-trimmed chunk interiors into one output in
start order. A later right-anchored final chunk overwrites earlier data where
the final overlap is larger than the configured overlap.

## Exact planner and ownership

The shipped defaults are latent-space chunk size `C=128`, overlap `O=32`,
hop `H=C-O=96`, and half-overlap `Q=floor(O/2)=16`. For a total length `L`:

1. disabled chunking or `L<C` calls the direct path;
2. `L==C` executes a one-chunk scaffold;
3. starts are `0,H,2H,... <= L-C`, followed by the right anchor `L-C` when
   it is not already present;
4. chunk zero owns `[0,s1+Q)`, interior chunk `i` owns
   `[si+Q,s(i+1)+Q)`, and the last chunk owns `[slast+Q,L)`;
5. encode scales chunk input ranges by the 4,096-sample codec ratio and
   stitches latent units; decode slices latent units and scales every
   ownership range by 4,096 samples.

For the frozen 225-latent oracle, starts are `[0,96,97]` and final ownership
is `[0,112)`, `[112,113)`, `[113,225)`. The middle chunk therefore owns only
one latent; the final chunk overwrites its remaining upstream writes. This is
the mutation-sensitive case that distinguishes the real algorithm from
half-trim concatenation, earlier-writer ownership, final-shortening,
crossfade, or normalized overlap-add.

The pure planner validates nonzero chunk size, `O<C`, checked scaling, exact
coverage, and no gaps. Tests cover direct thresholds, equality, normal and
right-anchored remainders, shipped long lengths, `O=0`, odd overlap with
floor-half, odd chunk size, `O=C-1`, and overflow. An indexed synthetic
writer independently reconstructs the sequential upstream result.

## Policy and stochastic behavior

The policy surface preserves all three upstream callers:

- full-model encode uses only parsed `pretransform.chunked`;
- full-model generation decode resolves `Option<bool>` over that config
  default;
- standalone encode/decode defaults off and accepts an explicit bool plus
  custom `C/O`.

All six full SA3 configs in the P0 provenance manifest set `chunked=true`.
The runtime remains unregistered.

Chunk recursion deliberately does not forward inner stride or noise kwargs,
matching frozen Python. Production creates one request-local RNG and threads
it through chunks in start order. Exact parity uses one controlled noise set
per chunk: SAME-L encode token noise, then decoder SoftNorm and learned-token
noise for both variants. Owned slices are copied before the next chunk is
retained, so a view cannot keep every full decoded chunk alive.

## Frozen Torch evidence

[`sa3-chunked-reference/manifest.json`](sa3-chunked-reference/manifest.json)
pins the upstream commit/file hash, Python 3.12.13, Torch/torchaudio 2.7.1,
Transformers 5.8.0, standalone SAME-S/SAME-L revisions, the portable
per-chunk noise streams and shapes, planner ownership, raw chunk edge slices,
stitched boundary slices, full outputs, and boundary spectral metrics.

| Artifact | Contents | Bytes | SHA-256 |
|---|---|---:|---|
| [`chunked-f32.safetensors`](sa3-chunked-reference/chunked-f32.safetensors) | raw chunk edges and stitched ownership-boundary slices | 2,035,792 | `ab537ae1803d0b74834e5c32c3a3c677e16bd0d41156116971ae881e556996a3` |
| [`chunked-outputs-f16.safetensors`](sa3-chunked-reference/chunked-outputs-f16.safetensors) | full encoded, decoded, and zero-noise direct/chunked outputs | 22,365,984 | `7a1a3eaf67506d8c6e0a62d9004b2886826d97e10adf1e7b617670e17615d965` |
| [`manifest.json`](sa3-chunked-reference/manifest.json) | provenance, authenticated snapshot hashes, tensor hashes, noise order, ownership, spectral evidence | 17,568 | `8127254d720a92c74f115d92a32da67bbee33efffd913c78dd519fa686cc1a80` |
| [`resource-evidence.json`](sa3-chunked-reference/resource-evidence.json) | exact CPU/Metal measurements, hardware, toolchain, geometry, and workflow provenance | 3,177 | `32a6cbcf52d06d0e0dbb737459fc2438fbaaab9b10fc9c3bf40d8f364b4bf63d` |

Real F32 Candle execution against the two pinned standalone checkpoints gives:

| Variant | Stitched encode cosine / max abs | Stitched decode cosine / max abs |
|---|---:|---:|
| SAME-S | 0.999999979 / 0.001026392 | 0.999999977 / 0.000063226 |
| SAME-L | 0.999999979 / 0.000978708 | 0.999999978 / 0.000062883 |

Every committed F32 ownership-boundary slice is tighter than the F16 whole
output floor: worst observed SAME-S encode/decode max abs is
0.000236511/0.000011355; SAME-L is 0.000077486/0.000002503.

With every stochastic path explicitly zeroed, the 225-latent oracle measures
the two true ownership boundaries at samples 458,752 and 462,848. Frozen
Torch's windowed log-magnitude L1 differs from direct by at most 0.00154 for
SAME-S and 0.000000194 for SAME-L. The independent Rust STFT gate requires
chunked discontinuity to remain within 0.003 of direct and the single-sample
jump below 0.03; observed maximum jumps are 0.01923 and 0.00493,
respectively.

## Long-duration resource gate

The resource geometry is exactly 1,292 latents / 5,292,032 samples
(120.0007 seconds). Direct execution processes 1,292 latent-equivalents;
chunking executes 14 full chunks / 1,792 latent-equivalents, 38.7% extra
model work. Each encode/decode variant runs in a separate fresh process after
a short warmup, with synchronized load/operation time, Metal current-allocated
bytes before the operation and sampled every 5 ms through its synchronized
completion, process peak RSS, backend, dtype, shapes, and checksums. RSS is
retained as diagnostic evidence but is not the accelerator gate because F32
SAME-L weight loading dominates it.

The `same-chunked` manual/weekly real-weight profile provisions both pinned
standalone snapshots, reruns frozen parity on Metal, and executes the full
SAME-S/SAME-L × direct/chunked × encode/decode matrix. The log verifier
requires exactly eight Metal records, the exact geometry above, strictly
lower chunked Metal allocation for every comparison, and an independently
chosen 2× wall-time ceiling. Merge remains gated on the exact-head accelerator
matrix; its reviewed measurements are recorded here before closeout.

Exact-head workflow
[`30194850972`](https://github.com/SceneWorks/inference/actions/runs/30194850972)
passed on an Apple M5 Max MacBook Pro (`Mac17,6`, 18 CPU cores, 40 GPU
cores, 128 GiB, Metal 4) with Rust/Cargo 1.96.0. The measured commit was
`424e90166386d02bf95bdaa57e958e839693418a`.

| Variant | Operation | Direct peak device bytes | Chunked peak device bytes | Reduction | Direct / chunked seconds | Ratio |
|---|---|---:|---:|---:|---:|---:|
| SAME-S | encode | 3,730,702,336 | 849,215,488 | 77.24% | 1.393 / 1.604 | 1.15× |
| SAME-S | decode | 3,695,804,416 | 907,427,840 | 75.45% | 1.514 / 1.618 | 1.07× |
| SAME-L | encode | 5,581,832,192 | 4,176,330,752 | 25.18% | 6.856 / 9.515 | 1.39× |
| SAME-L | decode | 5,462,081,536 | 4,193,386,496 | 23.23% | 7.129 / 9.835 | 1.38× |

The exact committed records, CPU diagnostics, geometry, hardware, toolchain,
and workflow provenance are in
[`resource-evidence.json`](sa3-chunked-reference/resource-evidence.json).
The final amended head reruns the same fail-closed workflow before merge so
the evidence-only commit amendment cannot conceal a source regression.

## Regeneration

```bash
/path/to/stable-audio-3/.venv/bin/python \
  scripts/reference/sa3_chunked_autoencoder_reference.py \
  --upstream /path/to/stable-audio-3 \
  --same-s /models/SAME-S/fbeb3dcf53a326e5682f38e22e7f740202d44232 \
  --same-l /models/SAME-L/41acf79dd242877d6499a1108ca5dba5d5eecfc5 \
  --output docs/migration/sa3-chunked-reference

python3 scripts/reference/sa3_chunked_autoencoder_reference.py \
  --verify --output docs/migration/sa3-chunked-reference
```
