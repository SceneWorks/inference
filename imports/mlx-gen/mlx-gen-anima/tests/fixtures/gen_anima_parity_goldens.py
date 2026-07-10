#!/usr/bin/env python3
"""Regenerate the Anima MLX-port parity goldens (sc-10524, epic 10512).

    Regenerate with (from the repo root, using the prepared reference venv):
        <venv>/bin/python mlx-gen-anima/tests/fixtures/gen_anima_parity_goldens.py

WHY THIS IS OK TO COMMIT
------------------------
The numeric reference is Hugging Face **diffusers** (Apache-2.0), which as of 0.39.0 ships the Anima
modular pipeline (`AnimaModularPipeline`), the `AnimaTextConditioner`, and the Cosmos-Predict2
`CosmosTransformer3DModel` (PR #13732, released). diffusers is Apache-2.0, so importing/using it from
this generator is license-compatible with this Apache-2.0 repo. This script:
  * imports diffusers/transformers (does NOT vendor their source),
  * reads model weights from the local HF cache (never committing weights), and
  * writes only the COMPUTED golden JSON files (data, not upstream copyrighted expression).

Each golden records: reference package + version, checkpoint repo + snapshot SHA, and this exact regen
command. Goldens are kept small (<200 KB total) — full activations are NOT dumped; instead a
deterministic subsample plus summary statistics (mean/std/min/max + fixed indices) are committed and
the Rust tests assert on those.

STAGES
------
  1. tokenizers        (CPU/exact, CI)     -> tokenizer_golden.json
  5. sigma schedule    (CPU/exact, CI)     -> sigma_schedule_golden.json
  2. qwen3 hidden      (real weights)       -> qwen3_hidden_golden.json
  3. conditioner       (real weights)       -> conditioner_golden.json
  4. DiT block0 + all  (real weights)       -> dit_forward_golden.json

Stages 2-4 read licensed weights; their JSON is committed but the Rust tests that consume them are
`#[ignore]`d + weights-gated (they need the snapshot + Metal to run the MLX side). Stages 1 & 5 need no
weights and run in CI. This script gracefully skips any stage whose weights are absent.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# --- reference provenance -------------------------------------------------------------------------
HF_HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))
DIFFUSERS_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
SINGLE_FILE_REPO = "circlestone-labs/Anima"


def diffusers_snapshot() -> Path | None:
    base = HF_HUB / f"models--{DIFFUSERS_REPO.replace('/', '--')}" / "snapshots"
    if not base.is_dir():
        return None
    for d in sorted(base.iterdir()):
        if (d / "transformer").is_dir():
            return d
    return None


def single_file_snapshot() -> Path | None:
    base = HF_HUB / f"models--{SINGLE_FILE_REPO.replace('/', '--')}" / "snapshots"
    if not base.is_dir():
        return None
    for d in sorted(base.iterdir()):
        if (d / "split_files" / "diffusion_models").is_dir():
            return d
    return None


def _pkg_versions() -> dict:
    import diffusers
    import transformers

    return {
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }


def _meta(reference: str, extra: dict | None = None) -> dict:
    m = {
        "story": "sc-10524",
        "epic": "10512",
        "reference": reference,
        "reference_packages": _pkg_versions(),
        "diffusers_repo": DIFFUSERS_REPO,
        "single_file_repo": SINGLE_FILE_REPO,
        "regen_command": (
            "<venv>/bin/python mlx-gen-anima/tests/fixtures/gen_anima_parity_goldens.py"
        ),
        "generator": "mlx-gen-anima/tests/fixtures/gen_anima_parity_goldens.py",
        "note": "Parity = no quality regression (Metal matmul ~1e-3; bf16 frozen-Python golden). "
        "Real-weights stages compare MLX bf16 vs a torch bf16 reference at ~1e-2 (mean-rel).",
    }
    ds = diffusers_snapshot()
    sf = single_file_snapshot()
    if ds is not None:
        m["diffusers_snapshot_sha"] = ds.name
    if sf is not None:
        m["single_file_snapshot_sha"] = sf.name
    if extra:
        m.update(extra)
    return m


# --- tensor summarization (keeps goldens tiny) ----------------------------------------------------
def summarize(arr: np.ndarray, n_samples: int = 48) -> dict:
    """A deterministic subsample + summary stats of a tensor. The Rust test recomputes the SAME
    fixed indices and stats on its own output and asserts elementwise + aggregate agreement."""
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    total = int(flat.size)
    # A deterministic, spread-out index set: evenly strided across the whole tensor.
    if total <= n_samples:
        idx = np.arange(total, dtype=np.int64)
    else:
        idx = np.linspace(0, total - 1, n_samples).astype(np.int64)
    idx = np.unique(idx)
    return {
        "shape": list(np.asarray(arr).shape),
        "count": total,
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "l2": float(np.sqrt((flat * flat).sum())),
        "sample_indices": [int(i) for i in idx],
        "sample_values": [float(flat[i]) for i in idx],
    }


def write(name: str, doc: dict) -> None:
    out = HERE / name
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {out}  ({out.stat().st_size} bytes)")


# --- deterministic LCG filler (bit-identical to the Rust test's lcg_fill) --------------------------
def lcg_fill(n: int, seed: int) -> np.ndarray:
    """Portable LCG in [-1, 1). Pure integer recurrence -> reproduces bit-exactly in Rust
    (u64 wrapping_mul/add + 0x7fffffff mask, value = s/2147483647*2-1 in f64 then cast f32)."""
    s = seed & 0x7FFFFFFF
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        vals[i] = s / 2147483647.0 * 2.0 - 1.0
    return vals.astype(np.float32)


# =================================================================================================
# Stage 1 — tokenizers (Qwen2 BPE + T5 SentencePiece) -> exact token ids. CPU/exact, CI.
# =================================================================================================
# Fixture prompts exercise the verified traps: booru @artist tags, nested parens (ComfyUI weight
# syntax is plain text to the tokenizer), CJK, emoji, digit runs, and the empty string.
FIXTURE_PROMPTS = [
    "1girl, silver hair, blue eyes, (detailed:1.2), masterpiece",
    "@artist_name, (chibi:2), simple background",
    "こんにちは世界",  # konnichiwa sekai
    "\U0001f3a8✨ 12345 67890",  # emoji + digit runs
    "((nested (parens)))",
    "",
]
MAIN_PROMPT = FIXTURE_PROMPTS[0]
MAX_SEQ = 512
QWEN_PAD_TOKEN_ID = 151643  # Qwen2 <|endoftext|> pad id (config::QWEN_PAD_TOKEN_ID)


def stage1_tokenizers() -> bool:
    snap = diffusers_snapshot()
    if snap is None:
        print("stage1: diffusers snapshot absent -> skip")
        return False
    from transformers import AutoTokenizer

    qwen = AutoTokenizer.from_pretrained(snap / "tokenizer")
    t5 = AutoTokenizer.from_pretrained(snap / "t5_tokenizer")

    cases = []
    for p in FIXTURE_PROMPTS:
        # Qwen2Tokenizer(padding="longest") — no BOS/EOS; empty -> reference replaces with a single 0.
        q = qwen(p, padding="longest", max_length=MAX_SEQ, truncation=True)
        q_ids = list(q["input_ids"])
        q_mask = list(q["attention_mask"])
        if len(q_ids) == 0:
            q_ids, q_mask = [0], [0]
        # T5TokenizerFast(padding="longest") — adds EOS (id 1).
        t = t5(p, padding="longest", max_length=MAX_SEQ, truncation=True)
        t_ids = list(t["input_ids"])
        cases.append(
            {"prompt": p, "qwen_ids": q_ids, "qwen_mask": q_mask, "t5_ids": t_ids}
        )

    doc = {
        "meta": _meta(
            "diffusers Anima tokenizers: Qwen2Tokenizer (tokenizer/) + T5TokenizerFast (t5_tokenizer/), "
            "padding='longest', truncation, max_length=512; empty prompt -> Qwen [[0]]/mask[[0]].",
            {"qwen_pad_token_id": 151643, "max_sequence_length": MAX_SEQ},
        ),
        "cases": cases,
    }
    write("tokenizer_golden.json", doc)
    return True


# =================================================================================================
# Stage 5 — sigma schedule + timestep sequence. CPU/exact, CI.
# =================================================================================================
def stage5_sigmas() -> bool:
    snap = diffusers_snapshot()
    from diffusers import FlowMatchEulerDiscreteScheduler

    if snap is not None:
        sched = FlowMatchEulerDiscreteScheduler.from_pretrained(snap / "scheduler")
    else:
        # config is fixed & known (shift=3.0, static, num_train_timesteps=1000)
        sched = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=False
        )
    schedules = {}
    for n in (1, 10, 30, 50):
        s = FlowMatchEulerDiscreteScheduler.from_config(sched.config)
        sigmas_in = np.linspace(1.0, 1.0 / n, n)  # before_denoise.py
        s.set_timesteps(sigmas=sigmas_in)
        schedules[str(n)] = {
            "sigmas": [float(x) for x in s.sigmas.tolist()],
            "timesteps": [float(x) for x in s.timesteps.tolist()],
        }
    doc = {
        "meta": _meta(
            "diffusers FlowMatchEulerDiscreteScheduler(shift=3.0, use_dynamic_shifting=False); "
            "sigmas_in = linspace(1.0, 1/N, N); shift s->3s/(1+2s); timesteps = sigma*1000; "
            "terminal 0.0 appended -> sigmas length N+1.",
            {"shift": float(sched.config.shift), "num_train_timesteps": 1000},
        ),
        "schedules": schedules,
    }
    write("sigma_schedule_golden.json", doc)
    return True


# =================================================================================================
# Real-weights stages (2, 3, 4). Compare MLX bf16 vs a torch bf16 reference at ~1e-2 (mean-rel).
# =================================================================================================
def _load_torch_bf16():
    import torch

    torch.manual_seed(0)
    return torch


def _qwen_ids_for(prompt, snap, torch):
    from transformers import AutoTokenizer

    qwen = AutoTokenizer.from_pretrained(snap / "tokenizer")
    q = qwen(prompt, padding="longest", max_length=MAX_SEQ, truncation=True, return_tensors="pt")
    ids = q["input_ids"]
    mask = q["attention_mask"]
    if ids.shape[-1] == 0:
        ids = ids.new_zeros((ids.shape[0], 1))
        mask = mask.new_zeros((mask.shape[0], 1))
    return ids, mask


def stage2_qwen3(torch) -> bool:
    snap = diffusers_snapshot()
    if snap is None:
        print("stage2: diffusers snapshot absent -> skip")
        return False
    from transformers import AutoModel

    te = AutoModel.from_pretrained(snap / "text_encoder", dtype=torch.bfloat16).eval()
    ids, mask = _qwen_ids_for(MAIN_PROMPT, snap, torch)
    real = int(ids.shape[1])
    # EXPLICITLY right-pad the batch-1 input with K Qwen2-pad tokens (id 151643) at attention-mask 0, so
    # the mask-multiply trap has real padded rows to zero. The all-ones batch-1 mask made it a no-op —
    # dropping the multiply then changed nothing (sc-10524 review). With K padded rows, dropping the
    # multiply leaves them nonzero (the causal tower still computes them) → the summary diverges.
    pad_k = 6
    pad_id = QWEN_PAD_TOKEN_ID
    ids = torch.cat([ids, ids.new_full((ids.shape[0], pad_k), pad_id)], dim=1)
    mask = torch.cat([mask, mask.new_zeros((mask.shape[0], pad_k))], dim=1)
    with torch.no_grad():
        out = te(input_ids=ids, attention_mask=mask)
        hidden = out.last_hidden_state
        hidden = hidden * mask.to(hidden.dtype).unsqueeze(-1)  # the mask-multiply trap (now non-trivial)
    arr = hidden.to(torch.float32).numpy()
    pad_abs_max = float(np.abs(arr[:, real:, :]).max())  # must be exactly 0 after the multiply
    doc = {
        "meta": _meta(
            "Qwen3-0.6B (text_encoder/) last_hidden_state AFTER the attention-mask multiply, bf16 tower. "
            f"Input = Qwen2 ids for the main fixture prompt + {pad_k} right-pad tokens (id {pad_id}) at "
            "mask 0, so the mask-multiply zeros real padded rows (not a no-op).",
            {
                "prompt": MAIN_PROMPT,
                "qwen_ids": ids[0].tolist(),
                "qwen_mask": mask[0].tolist(),
                "real_tokens": real,
                "padded_tokens": pad_k,
            },
        ),
        "last_hidden_state": summarize(arr),
        "pad_abs_max": pad_abs_max,
    }
    write("qwen3_hidden_golden.json", doc)
    return True


def stage3_conditioner(torch) -> bool:
    snap = diffusers_snapshot()
    if snap is None:
        print("stage3: diffusers snapshot absent -> skip")
        return False
    from transformers import AutoTokenizer
    from diffusers.models.condition_embedders.condition_embedder_anima import AnimaTextConditioner

    # Isolate the conditioner (not chained through Qwen3): a DETERMINISTIC LCG source (bit-reproducible
    # in Rust) + the REAL T5 ids for the main prompt. Run in fp32 both sides (the Rust conditioner takes
    # a `dtype` arg), so the golden isolates port MATH from bf16 quantization.
    t5 = AutoTokenizer.from_pretrained(snap / "t5_tokenizer")
    t5_ids_list = t5(MAIN_PROMPT, padding="longest", max_length=MAX_SEQ, truncation=True)["input_ids"]
    t5_ids = torch.tensor([t5_ids_list], dtype=torch.long)
    st = len(t5_ids_list)
    s_src = 18  # synthetic Qwen-source length
    src_shape = [1, s_src, 1024]
    src = torch.tensor(lcg_fill(int(np.prod(src_shape)), seed=3).reshape(src_shape))  # fp32

    cond = AnimaTextConditioner.from_pretrained(snap / "text_conditioner").float().eval()
    with torch.no_grad():
        out = cond(source_hidden_states=src, target_input_ids=t5_ids)  # [1, 512, 1024] fp32
    arr = out.numpy()
    active = arr[:, :st, :]  # the real conditioned tokens (rows [st:512] are zero padding)
    pad = arr[:, st:, :]
    doc = {
        "meta": _meta(
            "AnimaTextConditioner(text_conditioner/) output, fp32. Deterministic LCG source "
            "[1,18,1024] seed=3 + real T5 ids for the main prompt -> [1,512,1024], right-padded after "
            "masking. Isolates the conditioner (not chained through Qwen3).",
            {
                "prompt": MAIN_PROMPT,
                "t5_ids": t5_ids_list,
                "st": st,
                "lcg": {"source_seed": 3, "source_shape": src_shape},
                "expected_shape": [1, 512, 1024],
            },
        ),
        # Aggregate stats over the FULL tensor verify the right-pad (wrong padding shifts std/l2/count);
        # sample values are drawn from the ACTIVE region so they assert real conditioning, not zeros.
        "full": summarize(arr, n_samples=1),
        "active": summarize(active, n_samples=64),
        "pad_abs_max": float(np.abs(pad).max()) if pad.size else 0.0,
    }
    write("conditioner_golden.json", doc)
    return True


def stage4_dit(torch) -> bool:
    snap = diffusers_snapshot()
    if snap is None:
        print("stage4: diffusers snapshot absent -> skip")
        return False
    import diffusers.models.transformers.transformer_cosmos as tc
    from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel

    # The reference resizes the padding_mask via torchvision (absent, and torch 2.13.0 has no matching
    # wheel). We pass the padding_mask ALREADY at latent resolution (8x8 -> 8x8), so a NEAREST resize is
    # exactly identity — inject a tiny identity shim (no torchvision, numerically exact).
    class _ShimF:
        @staticmethod
        def resize(img, size, interpolation=None):
            assert list(img.shape[-2:]) == list(size), (
                f"identity-resize shim requires matching size, got {img.shape[-2:]} -> {size}"
            )
            return img

    class _ShimMode:
        NEAREST = "nearest"

    class _ShimT:
        functional = _ShimF
        InterpolationMode = _ShimMode

    tc.transforms = _ShimT

    # fp32 both sides (the Rust DiT.forward takes a `dtype` arg) — isolates port math (adaLN-LoRA +
    # NTK 3D RoPE) from bf16 quantization; also sidesteps diffusers' keep_in_fp32_modules cast guard.
    dit = CosmosTransformer3DModel.from_pretrained(snap / "transformer").float().eval()

    # deterministic LCG inputs (bit-reproducible in the Rust test). The latent is NON-SQUARE (h=8, w=12
    # ⇒ post-patch grid 4×6) so an h/w RoPE-axis transposition is DETECTABLE — a square fixture hides it
    # (sc-10524 review). Regenerate if you change these dims; the Rust test reads `latent_shape` from meta.
    lat_shape = [1, 16, 1, 8, 12]
    enc_shape = [1, 8, 1024]
    latent = torch.tensor(lcg_fill(int(np.prod(lat_shape)), seed=1).reshape(lat_shape))  # fp32
    encoder = torch.tensor(lcg_fill(int(np.prod(enc_shape)), seed=2).reshape(enc_shape))  # fp32
    sigma = torch.tensor([0.7], dtype=torch.float32)
    padding_mask = torch.zeros(1, 1, lat_shape[3], lat_shape[4], dtype=torch.float32)

    captured = {}

    def hook(_m, _i, o):
        captured["block0"] = (o[0] if isinstance(o, tuple) else o).detach().to(torch.float32).numpy()

    h = dit.transformer_blocks[0].register_forward_hook(hook)
    with torch.no_grad():
        out = dit(
            hidden_states=latent,
            timestep=sigma,
            encoder_hidden_states=encoder,
            padding_mask=padding_mask,
            return_dict=True,
        ).sample
    h.remove()
    full = out.to(torch.float32).numpy()

    doc = {
        "meta": _meta(
            "CosmosTransformer3DModel(transformer/) forward, fp32 (the model is cast .float() below — "
            "isolates port math from bf16 quantization). Deterministic LCG inputs: "
            "latent [1,16,1,8,12] seed=1 (NON-SQUARE ⇒ post-patch 4×6, so an h/w RoPE swap is detectable), "
            "encoder [1,8,1024] seed=2, sigma=0.7, padding_mask=zeros. "
            "block0 = hidden after transformer_blocks[0] [1,24,2048]; full = final velocity [1,16,1,8,12]. "
            "Exercises adaLN-LoRA modulation + NTK-scaled 3D RoPE.",
            {
                "lcg": {"latent_seed": 1, "encoder_seed": 2, "sigma": 0.7},
                "latent_shape": lat_shape,
                "encoder_shape": enc_shape,
            },
        ),
        "block0": summarize(captured["block0"], n_samples=64),
        "full": summarize(full, n_samples=64),
    }
    write("dit_forward_golden.json", doc)
    return True


def main():
    args = set(sys.argv[1:]) or {"1", "5", "2", "3", "4"}
    ran = []
    if "1" in args and stage1_tokenizers():
        ran.append("1")
    if "5" in args and stage5_sigmas():
        ran.append("5")
    if any(a in args for a in ("2", "3", "4")):
        torch = _load_torch_bf16()
        if "2" in args and stage2_qwen3(torch):
            ran.append("2")
        if "3" in args and stage3_conditioner(torch):
            ran.append("3")
        if "4" in args and stage4_dit(torch):
            ran.append("4")
    print("stages generated:", ",".join(ran) if ran else "(none)")


if __name__ == "__main__":
    main()
