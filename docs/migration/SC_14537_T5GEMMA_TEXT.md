# sc-14537: T5Gemma-b-b-ul2 text conditioning

This slice implements the encoder-only `t5gemma-b-b-ul2` path used by the
Stable Audio 3 checkpoints. It does not register a provider or load any decoder
tensor.

## Frozen inputs

- Stable Audio 3 source: commit
  `124e8a799f57a1f665495ecb72e547d0a62867f1`
- Snapshot: `stabilityai/stable-audio-3-small-music` revision
  `0fef1392cd842149a2b6d445e181c97608faac06`
- Text checkpoint: 1,183,022,944 bytes, SHA-256
  `9b05ea5a4f211d023832f706fb2c0e83e4fc721b6da35ab69ceb0b55eb7800d3`
- Reference runtime: Python 3.12.13, Torch 2.7.1, Transformers 5.8.0,
  eager attention

The text safetensors header is checked before construction: 340 BF16 tensors
and 591,490,560 parameters, split into exactly 134 encoder tensors /
281,580,288 parameters and 206 decoder tensors / 309,910,272 parameters. An
access-tracking real-weight gate proves construction requests exactly the 134
encoder keys.

## Architecture and conditioning

The implementation locks the shipped 12-layer, hidden-768, 12-head encoder:
half-split RoPE at theta 10,000; `sqrt(64)` query normalization; logit
soft-capping `50*tanh(scores/50)`; F32 softmax; four `(1 + weight)` F32
RMSNorms per layer; biasless Q/K/V/O; and
`down(gelu_tanh(gate) * up)`. Attention is bidirectional. The 4,096-token
sliding window cannot engage at the fixed 256-token conditioner length.

`tokenizer.json` is loaded directly with tokenizers 0.22. Inputs are
right-truncated and right-padded to 256 with pad ID 0. The identity
post-processor adds no BOS/EOS tokens.

All `None`, `Zero`, and `Learned` padding modes and the optional projection
branch are implemented. The shipped path is identity projection plus learned
padding: raw encoder output is converted to the padding tensor's F32 dtype,
then padded rows are replaced by the learned vector.

## Deterministic compute policy

The artifact remains BF16 on disk and the inventory check rejects any other
dtype. An exhaustive `DeviceLocation` policy computes the encoder in F32 on
CPU, Metal, and CUDA:

- CPU returns raw F32 encoder output.
- Metal casts the completed raw encoder output once to BF16, preserving the
  shipped raw-output contract, before the learned-padding operation promotes
  the conditioned result to F32.
- CUDA uses the same one-time BF16 raw-output boundary and F32 learned-padding
  result as Metal.

This policy is necessary because Candle CPU has no BF16 matmul kernel, while
native Metal BF16 and Torch BF16 backends are not mutually deterministic at
the required per-token cosine. On the 256-token probe, independent
Torch-CPU-BF16 and Torch-MPS-BF16 outputs themselves have minimum per-token
cosine 0.995014906 and maximum absolute difference 2.4765625. The MPS-BF16
artifact is retained as backend-variance evidence, but the canonical oracle is
Torch CPU-F32.

Against that canonical oracle:

- CPU F32 minimum per-token cosine is 0.999999995; measured maximum absolute
  difference is 0.001459122 (gate: 0.002, approved in sc-14537 comment
  `#14815` after strengthening the truncation probe).
- Metal F32 compute with the final BF16 boundary has minimum per-token cosine
  0.999999715; measured maximum absolute difference is 0.0625 (gate: 0.075).
- Token IDs, attention masks, right truncation, and every learned padding row
  must match exactly.

CUDA numerical parity is not claimed by this slice because this development
host has no CUDA toolkit or device. Story `sc-14552` owns real CUDA numerical
proof. Every change under `crates/audio` selects the required
`windows-cuda-check` CI lane; that lane compiles the complete Candle audio
package and test-target set with `--features cuda --no-run`, then runs CUDA
Clippy and rustdoc without allocating a GPU context.

## Metal resource evidence

The three-prompt real-weight gate (short, medium, and a heterogeneous 394-token
input whose distinct tail is truncated at 256) measured:

- first post-rebuild process: 2.51 s total, 1.90 s test execution,
  1,726,922,752-byte peak RSS;
- compile-free warm process: 1.93 s total, 1.81 s test execution,
  1,726,988,288-byte peak RSS.

The committed artifacts and their exact hashes, shapes, dtypes, tokenizer
contract, runtime, source commit, snapshot revision, and source payload hashes
are locked in `docs/migration/sa3-text-reference/manifest.json`. Run:

```bash
python3 scripts/reference/sa3_text_reference.py verify \
  --output-dir docs/migration/sa3-text-reference
```

Generation additionally requires the frozen Python environment, clean upstream
checkout, pinned snapshot, and an MPS-capable host.
