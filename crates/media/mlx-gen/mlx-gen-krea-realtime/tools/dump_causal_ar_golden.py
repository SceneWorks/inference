#!/usr/bin/env python3
"""Dump a **vendored torch golden** for the Krea Realtime causal AR forward (sc-15064, S16).

Why this exists
---------------
The S3 causal forward (`sc-8436`, `tests/causal_forward.rs`) is validated **torch-free** by three
non-circular invariants (cached-incremental == full-masked-recompute, single-chunk == stock
`forward_packed`, a hand-derived [4,4] mask matrix). Those checks pin our implementation against
*itself* and against the already-validated Wan attention — they cannot catch a mis-*interpretation*
of the reference. This script produces the missing EXTERNAL anchor: numbers dumped from the actual
reference `CausalWanModel`, so `tests/causal_ar_torch_golden.rs` can assert that our ≥2-chunk causal
forward agrees with the reference *implementation*, not with our reading of it.

Reference
---------
    repo      https://huggingface.co/krea/krea-realtime-video   (Apache-2.0)
    revision  6b5d204f9d14c3c3a59608e49d1da7fad90daf8d          (pinned; see REF_REVISION)
    files     transformer/{__init__,attention,causal_model,model}.py

The reference **source is NOT vendored into this repository** — this script downloads those four
`.py` files from the pinned revision into a temporary package at run time (`hf_hub_download`; no
weights, the 28.58 GB checkpoint is never touched) and imports them. Only the *numbers* it produces
are committed, as `tests/fixtures/causal_ar_golden.safetensors`. See `../NOTICE` for the
provenance/attribution record covering that fixture.

Reference patches applied (ALL of them, and why)
-----------------------------------------------
The reference is executed as published except for the four substitutions below. Each is asserted to
apply an exact number of times, so an upstream edit breaks this script loudly instead of silently
changing what the golden means. **None of them touch the two things under test — the block-causal
mask rule and the `causal_rope_apply` `start_frame` offset.**

  1. `model.py`: `device=torch.cuda.current_device()` → `device=position.device` (1x).
     PORTABILITY. `sinusoidal_embedding_1d` hard-codes a CUDA device; this host has none. Purely
     where the arange is allocated, not what it contains.
  2. `model.py`: `dtype = torch.bfloat16` → `dtype = q.dtype` — BOTH occurrences (2x): the
     cross-attention SDPA fallback AND the sageattention branch above it (the latter is dead on this
     host, which has no `sageattn`; it is patched only because the count-assert covers the whole
     file and both sites are the identical cast). PRECISION. The published code down-casts q/k/v to
     bf16 unconditionally, which both (a) crashes an fp32 model at the following fp32 `self.o(x)`
     and (b) would put a bf16 noise floor under the golden. Same math, higher precision.
  3. `causal_model.attention` / `model.attention` are rebound to the reference's own `attention`
     with `dtype=torch.float32`. PRECISION, via a kwarg the reference itself exposes (its default is
     `torch.bfloat16`); the function body is unchanged.
  4. `causal_model.py`: `frame_seqlen = 1560` → `frame_seqlen = <this golden's frame_seqlen>` (1x,
     inside `CausalWanSelfAttention.forward`'s cached branch). GEOMETRY. The reference hard-codes the
     production 14B tokens-per-latent-frame there as a `torch.compile` workaround — the line it
     replaced is still in the file directly above it:
         # frame_seqlen = math.prod(grid_sizes[0][1:]).item() # torch compile doesn't like this
     `current_start_frame = current_start // frame_seqlen` is the causal-RoPE offset, so on a small
     geometry the literal 1560 would make every chunk's offset 0 and the golden would be vacuous.
     We substitute the value the commented-out line computes. Nothing else about the offset rule is
     changed.

What is dumped
--------------
A single small safetensors (all f32; MLX rejects a file containing any f64 tensor):

    input.latent_chunk{0,1,2} [C_in, F_block, H, W]  the three AR chunks
    input.context             [text_len, text_dim]   raw T5-side text embedding
    input.timestep            [1]                    scalar diffusion timestep
    output.chunk{0,1,2}       [C_out, F_block, H, W] reference velocity per chunk
    output.sdpa_mask          [L, L]                 reference `get_sdpa_mask` (1.0 allow / 0.0 mask)
    weights.<state_dict key>                         every model parameter

`output.chunk1` / `output.chunk2` are the load-bearing tensors: they are produced with a populated KV
cache at `current_start = block_size` and `2·block_size`, i.e. the ≥2-chunk causal path where a
mask-interpretation or RoPE-offset error shows up. Three chunks (rather than the minimum two) put the
last chunk four latent frames into the clip, which widens the gap the negative control measures.
`output.sdpa_mask` anchors the mask RULE directly against the reference's own `get_sdpa_mask`, which
is fully parameterised and needs no patch.

Weights are rounded onto the bf16 grid before the reference runs and before they are dumped, so the
Rust side (which casts the transformer to bf16 on load) consumes bit-identical weights and the only
residual difference is activation precision.

Usage
-----
    uv venv && uv pip install torch safetensors diffusers einops huggingface_hub
    python dump_causal_ar_golden.py                  # rewrite the committed fixture
    python dump_causal_ar_golden.py OUT.safetensors  # write elsewhere
"""

import functools
import os
import sys
import types

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

REF_REPO = "krea/krea-realtime-video"
REF_REVISION = "6b5d204f9d14c3c3a59608e49d1da7fad90daf8d"
REF_FILES = [
    "transformer/__init__.py",
    "transformer/attention.py",
    "transformer/causal_model.py",
    "transformer/model.py",
]

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "tests", "fixtures", "causal_ar_golden.safetensors")

# --- the golden's geometry (must match `tests/causal_ar_torch_golden.rs`) --------------------------
DIM = 64
NUM_HEADS = 2  # head_dim 32
NUM_LAYERS = 2
FFN_DIM = 128
FREQ_DIM = 32
TEXT_DIM = 32
TEXT_LEN = 8
IN_DIM = 16
OUT_DIM = 16
PATCH = (1, 2, 2)
LATENT_H = 4
LATENT_W = 4
FRAMES_PER_BLOCK = 2
TIMESTEP = 500.0

FRAME_SEQLEN = (LATENT_H // PATCH[1]) * (LATENT_W // PATCH[2])  # tokens per latent frame = 4
BLOCK_TOKENS = FRAME_SEQLEN * FRAMES_PER_BLOCK  # 8
NUM_CHUNKS = 3
TOTAL_TOKENS = BLOCK_TOKENS * NUM_CHUNKS  # 24


def _patch_source(text: str, old: str, new: str, expected: int, where: str) -> str:
    """Apply a documented substitution, asserting it hits exactly `expected` sites."""
    found = text.count(old)
    if found != expected:
        raise SystemExit(
            f"reference drift in {where}: expected {expected} occurrence(s) of {old!r}, found "
            f"{found}. The pinned revision ({REF_REVISION}) changed, or the wrong file was fetched. "
            "Re-audit the patch list in this script's docstring before regenerating the golden."
        )
    return text.replace(old, new)


def load_reference() -> types.ModuleType:
    """Fetch the pinned reference `.py` files and import `causal_model` with the documented patches."""
    paths = {f: hf_hub_download(REF_REPO, f, revision=REF_REVISION) for f in REF_FILES}

    pkg = types.ModuleType("krea_ref")
    pkg.__path__ = []  # namespace-ish package so `from .attention import ...` resolves
    sys.modules["krea_ref"] = pkg

    def _exec(name: str, src: str, filename: str):
        mod = types.ModuleType(f"krea_ref.{name}")
        mod.__package__ = "krea_ref"
        sys.modules[f"krea_ref.{name}"] = mod
        setattr(pkg, name, mod)
        exec(compile(src, filename, "exec"), mod.__dict__)
        return mod

    with open(paths["transformer/attention.py"]) as fh:
        _exec("attention", fh.read(), "krea_ref/attention.py")

    with open(paths["transformer/model.py"]) as fh:
        model_src = fh.read()
    # (1) portability, (2) precision — see the module docstring.
    model_src = _patch_source(
        model_src,
        "device=torch.cuda.current_device()",
        "device=position.device",
        1,
        "model.py",
    )
    model_src = _patch_source(
        model_src, "dtype = torch.bfloat16", "dtype = q.dtype", 2, "model.py"
    )
    _exec("model", model_src, "krea_ref/model.py")

    with open(paths["transformer/causal_model.py"]) as fh:
        causal_src = fh.read()
    # (4) geometry — see the module docstring.
    causal_src = _patch_source(
        causal_src,
        "            frame_seqlen = 1560\n",
        f"            frame_seqlen = {FRAME_SEQLEN}\n",
        1,
        "causal_model.py",
    )
    causal_model = _exec("causal_model", causal_src, "krea_ref/causal_model.py")

    # (3) precision — rebind to the reference's own `attention` with its fp32 kwarg.
    fp32_attention = functools.partial(
        sys.modules["krea_ref.attention"].attention, dtype=torch.float32
    )
    causal_model.attention = fp32_attention
    sys.modules["krea_ref.model"].attention = fp32_attention
    return causal_model


def build_model(causal_model):
    torch.manual_seed(0)
    model = causal_model.CausalWanModel(
        model_type="t2v",
        patch_size=PATCH,
        text_len=TEXT_LEN,
        in_dim=IN_DIM,
        dim=DIM,
        ffn_dim=FFN_DIM,
        freq_dim=FREQ_DIM,
        text_dim=TEXT_DIM,
        out_dim=OUT_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        local_attn_size=-1,  # the shipped checkpoint's value (global attention)
        sink_size=0,  # ditto
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ).eval()

    # The constructor already ran the reference's own `init_weights()` (Xavier-uniform Linears,
    # `randn(1,6,dim)/sqrt(dim)` modulation, N(0,0.02) text/time embeddings) — keep it: a realistic
    # weight scale is what makes self-attention actually MOVE the residual stream, and therefore what
    # makes the golden discriminating. Two deliberate deviations:
    #
    #   * `init_weights()` ZEROES `head.head.weight` (a diffusion-training convention). Left alone the
    #     model would emit only `head.head.bias` — also zero — so the golden would be an all-zero
    #     tensor that any implementation reproduces. Re-init it Xavier-uniform.
    #   * All biases are zeroed by `init_weights()`. A zero bias hides a bias-handling bug, so give
    #     every bias a small non-zero value.
    gen = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(model.head.head.weight)
        for name, p in model.named_parameters():
            if name.endswith(".bias"):
                p.copy_(0.05 * (torch.rand(p.shape, generator=gen) - 0.5))
        # Round onto the bf16 grid so the Rust side (bf16 transformer) sees identical weights.
        for p in model.parameters():
            p.copy_(p.to(torch.bfloat16).to(torch.float32))
    return model.to(torch.float32)


def new_caches():
    """The per-block KV + cross-attention caches `_forward_inference` expects (allocated pipeline-side
    in the reference; reproduced here at the exact shapes and initial values it reads)."""
    head_dim = DIM // NUM_HEADS
    kv_cache = [
        {
            "k": torch.zeros(1, TOTAL_TOKENS, NUM_HEADS, head_dim, dtype=torch.float32),
            "v": torch.zeros(1, TOTAL_TOKENS, NUM_HEADS, head_dim, dtype=torch.float32),
            "global_end_index": 0,
            "local_end_index": 0,
        }
        for _ in range(NUM_LAYERS)
    ]
    crossattn_cache = [{"is_init": False} for _ in range(NUM_LAYERS)]
    return kv_cache, crossattn_cache


def main():
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)
    causal_model = load_reference()
    model = build_model(causal_model)

    torch.manual_seed(7)
    chunks = [
        torch.rand(IN_DIM, FRAMES_PER_BLOCK, LATENT_H, LATENT_W, dtype=torch.float32) * 2.0 - 1.0
        for _ in range(NUM_CHUNKS)
    ]
    context = torch.rand(TEXT_LEN, TEXT_DIM, dtype=torch.float32) * 2.0 - 1.0
    # `t` is [B, F]: `_forward_inference` unflattens the time embedding by `t.shape`, and each block
    # reads `e.shape[1]` as its frame count. A constant fill is the scalar-timestep case our
    # `CausalKreaTransformer::forward_chunk` takes.
    t = torch.full((1, FRAMES_PER_BLOCK), TIMESTEP, dtype=torch.float32)

    kv_cache, crossattn_cache = new_caches()
    outs = []
    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            y = model(
                [chunk],
                t,
                [context],
                seq_len=TOTAL_TOKENS,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=i * BLOCK_TOKENS,
                cache_start=0,
            )
            outs.append(y[0].float().contiguous())
            assert kv_cache[0]["global_end_index"] == (i + 1) * BLOCK_TOKENS, kv_cache[0]

    # The reference's own mask rule, unpatched (`get_sdpa_mask` is fully parameterised). Trimmed from
    # its 128-multiple right padding to the real sequence length.
    mask = causal_model.get_sdpa_mask(
        "cpu",
        num_frames=FRAMES_PER_BLOCK * NUM_CHUNKS,
        frame_seqlen=FRAME_SEQLEN,
        num_frame_per_block=FRAMES_PER_BLOCK,
        local_attn_size=-1,
        dtype=torch.bool,
    )[:TOTAL_TOKENS, :TOTAL_TOKENS]

    tensors = {
        "input.context": context.contiguous(),
        "input.timestep": torch.tensor([TIMESTEP], dtype=torch.float32),
        "output.sdpa_mask": mask.float().contiguous(),
    }
    for i in range(NUM_CHUNKS):
        tensors[f"input.latent_chunk{i}"] = chunks[i].contiguous()
        tensors[f"output.chunk{i}"] = outs[i]
    for k, v in model.state_dict().items():
        tensors[f"weights.{k}"] = v.detach().float().contiguous()

    for k, v in tensors.items():
        assert v.dtype == torch.float32, (k, v.dtype)  # MLX rejects a file with any f64 tensor

    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_file(
        tensors,
        out,
        metadata={
            "source": f"{REF_REPO}@{REF_REVISION} transformer/causal_model.py CausalWanModel",
            "license": "Apache-2.0",
            "generator": "tools/dump_causal_ar_golden.py (sc-15064)",
            "geometry": (
                f"dim={DIM} heads={NUM_HEADS} layers={NUM_LAYERS} frame_seqlen={FRAME_SEQLEN} "
                f"block_tokens={BLOCK_TOKENS} chunks={NUM_CHUNKS}"
            ),
        },
    )
    print(f"wrote {out} ({os.path.getsize(out)} bytes, {len(tensors)} tensors)")
    for i, o in enumerate(outs):
        print(f"  chunk{i} |max| = {o.abs().max().item():.6f}")


if __name__ == "__main__":
    main()
