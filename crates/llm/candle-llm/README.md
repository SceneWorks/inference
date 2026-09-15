# candle-llm

An **on-device LLM serving engine** — the [Candle](https://github.com/huggingface/candle) backend of
a backend-neutral serving contract (cross-platform, including Windows/CUDA). It implements the
[`core-llm`](https://github.com/SceneWorks/core-llm) contract and, by passing that contract's
conformance suite as a *second, independent* backend, **de-provisionalizes** it. A sibling
[`mlx-llm`](https://github.com/SceneWorks/mlx-llm) crate provides the Apple MLX backend.

## Architecture

Built bottom-up, mirroring `mlx-llm`'s structure on Candle tensors:

1. **`primitives`** — backend-owned tensor leaves: a batch-capable `KvCache`, a pluggable sampler,
   the `Rope` family (standard / Llama-3 scaled), GQA attention helpers, group-wise quantization
   (via Candle's `QTensor`/`QMatMul`), the `nn` leaves, and a safetensors `Weights` loader.
2. **`config` + `models`** — `ModelConfig` parsed from `config.json` and the generic causal decoder
   (`CausalLm`), `&self` forward + `from_weights`, with architecture dispatch (Llama / Mistral /
   Qwen3).
3. **`decode`** — the streaming, cancellable decode loop driving any `Decode` model.
4. **`provider`** — implements `core_llm::TextLlm` over the engine and registers it (`candle-llama`).
   Supports a controllable **thinking mode** for models whose chat template gates `enable_thinking`
   (e.g. Qwen3): a `ThinkingSegmenter` splits the stream into `<think>…</think>` reasoning vs answer,
   surfaced on the `Thinking` / `Content` channels and in `out.thinking`. Supports **tool (function)
   calling** for models whose chat template renders tools (e.g. Qwen3.6): a request's `tools` render
   into the prompt, and a `ToolCallSegmenter` lifts the model's `<tool_call>` blocks (Qwen3.6 XML or
   JSON/Hermes) out of the answer text into structured `out.tool_calls`.
5. **`prepare`** — implements `core_llm::SnapshotPreparer`: turn a downloaded model (an HF snapshot
   dir or a `*.gguf`) into a persisted, loadable snapshot, optionally baking in Q4/Q8. Candle's
   `QTensor` has no safetensors form, so a quantized snapshot is dense weights carrying the
   quantization rounding plus a `quantization` block in `config.json`; the loader re-quantizes the
   projections on load, so `prepare_snapshot` yields a genuinely quantized model in candle's storage
   shape.

## Backends / features

- default → CPU (builds anywhere)
- `--features cuda` → NVIDIA CUDA (the Windows target)
- `--features metal` → Apple Metal
- `--features flash-attn` → CUDA + fused **FlashAttention-2** kernels (`candle-flash-attn`) at the
  attention seam, with an eager fallback for the cases the kernel can't serve

Compute runs in `bf16` on the GPU backends and `f32` on CPU. The dense compute dtype is also
selectable per load (`CausalLm::from_weights_dtype` — e.g. **f16** vs the default bf16) for dtype
perf tuning.

With `flash-attn`, `primitives::attention::sdpa` dispatches the dense causal/bidirectional path to the
fused kernel and falls back to the eager softmax SDPA for soft-cap (Gemma-2), MLA's mismatched q/v
head dims, padded-batch additive masks, or f32/CPU. The kernel's causal masking is bottom-right
aligned, matching the eager convention, so cached decode stays correct; the two paths agree within a
few half-precision ULPs (proven by a gated parity test). `candle-flash-attn` ships sm80 kernels that
**do compile and run on sm_120** (Blackwell) under the project's `CUDA_COMPUTE_CAP=120` build.

### Multi-GPU (pipeline sharding)

`CausalLm::from_dir_sharded(dir, cfg, dtype, &[dev0, dev1, …])` splits a decoder's layers into
**contiguous blocks across multiple GPUs** — for a model too large to fit on one card (e.g. across
2×24GB consumer GPUs). Layer block `b` lives on `devices[b]`, the embeddings + first input on the
first device, the final norm + LM head on the last, and the decoder hands the hidden state across each
boundary (a no-op for a single-device load). The sharded `Weights` loader streams each file through
host memory, so **no single GPU ever holds more than its own shard** — the whole point, for cards that
can't stage the full model. This is a *capacity* feature, not a throughput one (a single sequence
gains nothing from being split). Dense only; combine with quantize-on-load by choosing one or the
other to fit, not both.

## Testing

Unit tests + the synthetic conformance run need no models and no GPU:

```sh
cargo test                       # CPU, no weights
cargo clippy --all-targets -- -D warnings
```

The real-weights checks are `#[ignore]`d and gated on environment variables pointing at on-disk
snapshots; run them with `-- --ignored` (add `--features cuda` for the GPU path). They cover the
full `core-llm-testkit` conformance suite across architectures and load formats, plus per-feature
parity tests:

| env var | points at | exercised by |
|---|---|---|
| `CANDLE_LLM_TEST_MODEL` | a Llama-family HF snapshot dir (e.g. SmolLM2-135M-Instruct) | conformance (dense + **Q8** quantize-on-load), batch decode, **prefix-cache** reuse, **paged** cache, **continuous** batching, **speculative** (prompt-lookup + draft-model), **snapshot prepare** (dense + Q8), `bench` (tokens/s) |
| `CANDLE_LLM_QWEN3_MODEL` | a Qwen3 HF snapshot dir | conformance (dense + **Q4** quantize-on-load; q/k RMSNorm, head_dim 128), **prefix-cache** reuse, **paged** cache, **continuous** batching, **speculative** (prompt-lookup + draft-model), **snapshot prepare** (Q4), **thinking** (Qwen3 gates `enable_thinking` → real `<think>` reasoning) |
| `CANDLE_LLM_QWEN35_MODEL` | a Qwen3.6 (`qwen3_5` / Qwen3-Next) snapshot dir (27B dense or 35B-A3B MoE) | `qwen35` — resolve/dispatch, coherent text, think/no-think, Q8, conformance; **tool calling** (`qwen35_tools` — the model emits a `<tool_call>` that the provider parses into `out.tool_calls`) |
| `CANDLE_LLM_GGUF` | a single `*.gguf` file | conformance + GGUF parity vs the HF load, **snapshot prepare** (GGUF → snapshot, dense + Q8) |
| `CANDLE_LLM_{PHI3,QWEN2MOE,GEMMA2,GLM4,DEEPSEEK}_MODEL` | a snapshot for that architecture family | `breadth` — coherent-text streaming per family |
| `CANDLE_LLM_GEMMA4_MODEL` | a Gemma 4 unified snapshot dir (e.g. `google/gemma-4-12B-it`) | `breadth` — coherent-text streaming; `gemma4_decoder` — the real-weight per-layer-type forward (hidden-state stack + soft-capped logits) |
| `CANDLE_LLM_VLM_MODEL` | a SigLIP-based `LlavaForConditionalGeneration` snapshot dir (small: `llava-hf/llava-interleave-qwen-0.5b-hf`; faithful: JoyCaption) | `vlm` — image captioning + the multimodal conformance check |

The `breadth` test streams a prompt through each non-Llama architecture: **Phi-3** (packed qkv/gate_up),
**Qwen2-MoE** (router + experts + shared, q/k/v bias), **Gemma-2** (sandwich norms + soft-caps + GeGLU),
**GLM-4** (sandwich + partial/interleaved RoPE), **DeepSeek-V2** (Multi-head Latent Attention +
fine-grained MoE; verified on `deepseek-ai/DeepSeek-V2-Lite-Chat`, which fits in 96GB), and
**Gemma 4 unified** (the per-layer-type decoder — see below).

**Gemma 4** is the first architecture whose layers are *not* uniform: `layer_types` alternates 40
`sliding_attention` layers (head_dim 256, θ=10 000, 1024-key window, 8 KV heads) with 8
`full_attention` ones (head_dim 512, `proportional` θ=1 000 000 with `partial_rotary_factor` 0.25, 1
KV head, and `attention_k_eq_v` — one shared key/value projection, so those layers carry no `v_proj`
at all). The decoder therefore reads `ModelConfig::layer_attention` per layer and carries one RoPE
per layer *type* (`RopeTables`); norms are plain `weight` (not Gemma-2's `1 + weight`), the attention
scale is `1.0`, the value heads take a scale-free per-head RMSNorm, and every block's output is
multiplied by a learned per-layer `layer_scalar` after both residual adds.

Two further Gemma 4 knobs are implemented even though the shipped LTX-2.5 encoder leaves them off:
`num_kv_shared_layers` (the trailing layers project no K/V at all and attend the last earlier layer
of their **own** type) and `use_double_wide_mlp` (those same layers get a 2x inner width). Their
donor keys/values are published per forward rather than carried in the cache, so a *continued*
generation on such a model is refused by name — a single forward from an empty cache, which is what
a text encoder runs, is exact.

Because the two layer types disagree on `(n_kv_heads, head_dim)`, the single-shape **paged** pool
cannot serve the stack — use the contiguous cache and the `Exact` continuous mode; both refuse
Gemma 4 by name rather than attending the wrong shape.

`tests/gemma4_decoder.rs` pins the numbers against
`crates/llm/testdata/gemma4/gemma4_decoder_goldens.json` — a backend-neutral fixture owned by
sc-18760 that `mlx-llm` asserts against too, so a divergence between the two ports fails on one of
them rather than going unnoticed. It ships mutation oracles (a skipped `layer_scalar`, V read off
the normed key, a dropped window, a leading-slice partial RoPE) which the suite requires to differ
from the truth.

The `prefix` test covers **shared-prefix KV reuse** (`generate_cached` over a `PrefixCache`): a request
sharing a leading run of tokens with a stored one (a system prompt, a few-shot preamble, a growing
chat) reuses that span's keys/values instead of recomputing prefill. A tiny synthetic CPU model proves
it is **bit-exact** — `generate_cached` is token-for-token identical to a cold `generate` — and the
`#[ignore]`d real-weights variants confirm the mechanic on a GPU snapshot (first-token logits match
within a small bf16 tolerance; reuse accounting is exact) and report the reused-token count.

The `paged` test covers the **paged KV cache** (`PagedKvCache` + `BlockPool`, gather-then-SDPA behind
the `KvCache` trait): per-sequence block tables drawn from a shared pool, so a **ragged** batch
(sequences of differing lengths) decodes bit-exactly — each attends only its own keys, no left-pad —
and divergent requests sharing a system prefix point at the *same physical blocks* (copy-on-write,
refcounted). A synthetic CPU model proves drop-in parity (paged is token-for-token identical to the
contiguous cache), ragged correctness, and bit-exact prefix sharing; the `#[ignore]`d real-weights
variants confirm the drop-in is bit-exact on a GPU snapshot and report the reservation saving vs a
naive max-context slab.

The `continuous` test covers **iteration-level continuous batching** (`generate_continuous`): up to
`max_batch` sequences decode at once over **per-sequence** `PagedKvCache`s on one shared `BlockPool`,
and the moment a sequence retires a waiting request is prefilled into the freed slot (admit-on-retire,
driven by the same `core_llm::Scheduler`) — the batch never drains, with no left-padding. It ships two
modes (`BatchExactness`): **`Exact`** runs each sequence as its own batch-1 forward (`decode_logits` on
its own cache) — byte-identical to running the request alone; **`Throughput`** batches the
projections / MLP / lm_head and runs only attention per-sequence (`decode_logits_per_seq`, an
`LlamaAttention` `project → per-seq attention → output` refactor) — throughput scales with occupancy at
the cost of a row only *tracking* its batch-1 run (the batched matmul isn't M-invariant on a GPU, the
same caveat the synchronous `generate_batch` carries). A synthetic CPU model proves `Exact` (and, since
CPU f32 reduces in a batch-invariant order, `Throughput` too) is **token-for-token identical** to each
request's batch-1 run — across differing prompt lengths and across admit-on-retire — plus exactly-one
terminal `Done` per request under mid-stream cancel and zero-budget requests. The `#[ignore]`d
real-weights variant confirms the `Exact` equality on a GPU snapshot and reports `Throughput` decode
tokens/s by occupancy. (The custom paged attention kernel that would batch the per-sequence attention
loop — the next bottleneck as occupancy grows — is the deferred story 7258 / mlx sc-7325.)

The `speculative` test covers **speculative decoding** — proposing several tokens per target forward
and verifying them in one batched pass (`decode_logits_all`), accepting the longest agreeing prefix +
a bonus and rolling back rejected drafts via the `KvCache::truncate` seam, in two flavors:
**prompt-lookup** (`generate_prompt_lookup`, n-gram proposer, no draft model) and **draft-model**
(`generate_draft_speculative`, a small/quantized model proposes, the big model verifies, accepted via
the distribution-preserving acceptance sampler). With `num_draft = 0` the verify is a single-token
forward, so both are **bit-identical** to non-speculative `generate` — the exactness gate. A synthetic
CPU model also shows draft acceptance (`forwards < tokens`) at identical greedy output (an identical
draft accepts every token); the `#[ignore]`d real-weights variants confirm the speedup on a GPU
snapshot (a dense target + **Q4** draft from the same weights), where the greedy run *tracks* (rather
than bit-matches) non-speculative because the multi-token verify kernel rounds a few bf16 ULP
differently from the single-token decode kernel.

The `vlm` test covers the **vision-language path** (`LlavaModel` + `LlavaProvider`): a SigLIP vision
tower ([`SiglipVisionTower`]) encodes the image, a two-layer GELU MLP projector lifts a chosen
penultimate hidden state into the language hidden size, and those patch rows replace the expanded
image-token placeholders in the prompt embeddings (the `decode_logits_from_embeds` splice hook) before
the reused Llama decoder generates a caption. A tiny synthetic CPU model (SigLIP tower + projector +
Llama, random weights) proves the mechanic end-to-end with no weights and no GPU: the tower is
image-sensitive, its features splice into the right rows, and the change is visible in the decoder's
first-token logits (the image actually drives the decode), with greedily deterministic generation. The
`#[ignore]`d real-weights variant loads an actual SigLIP-based LLaVA snapshot
(`CANDLE_LLM_VLM_MODEL` — e.g. `fancyfeast/llama-joycaption-beta-one-hf-llava`), captions an image on
the selected device (CUDA with `--features cuda`), and confirms the `core-llm-testkit`
`check_multimodal` check passes via the **generate** branch (and that a text-only request is rejected).
The vision tower + projector run in **f32** (the bf16 weights promoted on load) for numeric fidelity,
then the features are cast to the decoder's compute dtype before the splice.

The `bench` test (`CANDLE_LLM_TEST_MODEL`) reports prefill + decode tokens/s for the reference decoder
across compute dtypes (bf16 / **f16**) and a Q8 quantized load — dtype perf is *measured*, not assumed.
Built with `--features flash-attn` the same bench drives the fused kernel, so running it both ways
reads the flash-vs-eager speedup. On an **RTX PRO 6000 (sm_120)**, prefill 256 / decode 128 tokens:

| model (head_dim) | dtype | prefill eager → flash | decode eager → flash |
|---|---|---|---|
| Qwen3-0.6B (128) | bf16 | 5118 → 6534 tok/s (**+28%**) | 24.1 → 27.4 tok/s (**+14%**) |
| Qwen3-0.6B (128) | f16  | 3300 → 6638 tok/s (**+101%**) | 25.0 → 26.7 tok/s |
| SmolLM2-135M (64) | f16 | 6371 → 8211 tok/s (**+29%**) | 26.3 → 30.0 tok/s (**+14%**) |

(Small models are launch-overhead-bound at decode, so the prefill gain is the clearer signal; the
fused path wins on every row.) The flash kernel's numerical agreement with the eager path is asserted
by a gated CUDA unit test (`flash_attn_matches_eager_on_cuda`, full-prompt + decode shapes), and the
whole `core-llm-testkit` conformance suite passes built `--features flash-attn`.

> Q4_K's block size is 256, so Q4 quantize-on-load needs projection `in`-dims that are multiples of
> 256 (true of Qwen3's hidden 1024, not of SmolLM2's 576); Q8_0's block is 32 and applies broadly.

```sh
# Whole real-weights suite on CUDA:
CANDLE_LLM_TEST_MODEL=/path/SmolLM2-135M-Instruct \
CANDLE_LLM_QWEN3_MODEL=/path/Qwen3-0.6B \
CANDLE_LLM_GGUF=/path/Model-Q4_K_M.gguf \
  cargo test --features cuda -- --ignored --nocapture

# Tokens/s, eager vs the fused FlashAttention-2 path (run both, compare):
CANDLE_LLM_TEST_MODEL=/path/Qwen3-0.6B cargo test --features cuda       --test bench -- --ignored --nocapture
CANDLE_LLM_TEST_MODEL=/path/Qwen3-0.6B cargo test --features flash-attn --test bench -- --ignored --nocapture
```

On Windows/CUDA the build needs the VS dev environment (`vcvars64` + `CUDA_COMPUTE_CAP`); see the
helper `.bat` scripts under the workspace root.

## Status

Work in progress (epic 7153, story 7237). Not yet published.

## License

Apache-2.0.
