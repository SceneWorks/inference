#!/usr/bin/env python3
"""Produce the YuE stage-2 (teacher-forced upsampler) parity fixtures (sc-19381, epic sc-19373).

`crates/audio/candle-audio-yue/src/stage2.rs` ports upstream YuE's stage-2 loop
(`inference/infer.py`: `stage2_generate` + `stage2_inference`, including the 300-frame chunking,
the batch grouping, the ragged tail chunk and the `fix_output` repair). This script runs **the
upstream functions themselves** — extracted verbatim from the pinned `infer.py` with `ast` and
executed against the upstream `CodecManipulator` / `_MMSentencePieceTokenizer` — so the goldens
are the reference's own output, not a re-implementation of it. Nothing upstream is vendored.

Two fixtures, two subcommands:

* ``mock`` → ``stage2_mock_reference.json``. The upstream loop driven by a **deterministic integer
  mock model** (`MockModel` below; `stage2::tests::MockLm` is its Rust twin). No weights, seconds to
  run. It exists for the assembly logic: multi-chunk batch groups, a partial last group, the ragged
  tail chunk, the `num_batch == 0` path, the blocked-range slice and `fix_output`'s
  most-frequent repair with its first-seen tie order. The mock is biased so most residuals land
  in their own codebook but a known fraction do not, so the repair runs on every case.
* ``real`` → ``stage2_real_reference.json``. The upstream loop driven by the real
  `m-a-p/YuE-s2-1B-general` checkpoint through `transformers` on CPU. The codebook-0 inputs are
  the xcodec encode (upstream `SoundStream`, `target_bw=0.5`, exactly as `infer.py`'s ICL path
  encodes a reference) of a synthetic, arithmetic 16 kHz clip — in-distribution codes with no
  third-party audio. ``--compute-dtype`` selects the torch compute dtype: ``float32`` is the
  committed golden (it is what the candle CPU lane computes in — bf16 weights upcast to f32);
  ``bfloat16`` is upstream's own setting and is kept for the divergence characterisation (bf16
  logits tie often, and `torch.argmax` breaks ties to the lowest id). See the fixture README.

Environment (never resolved from a Hugging Face cache — epic 13657):

* ``YUE_REF_INFERENCE_DIR`` — the upstream clone's ``inference/`` directory at YuE-v1
  ``6d4f0b1f8ce6a55fb2392e959394c46e07ee334d`` (with ``xcodec_mini_infer`` beside it for
  ``real``). Every upstream file read is SHA-256 pinned below.
* ``YUE_S2_BF16_SNAPSHOT`` (``real`` only) — the dense stage-2 snapshot (``bf16/`` of the staged
  ``yue-s2-1b-general-candle`` repo, upstream revision ``9dfa90b7``).

Run inside the reference environment (torch, transformers, numpy, einops, sentencepiece,
omegaconf)::

    python scripts/reference/yue_stage2_reference.py mock
    python scripts/reference/yue_stage2_reference.py real --compute-dtype float32
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "crates/audio/candle-audio-yue/tests/fixtures"

UPSTREAM_COMMIT = "6d4f0b1f8ce6a55fb2392e959394c46e07ee334d"
STAGE2_REVISION = "9dfa90b7013f6b5e7eb5eb2991620dca33058a0e"
UPSTREAM_SHA256 = {
    "infer.py": "9e2fa795b1556d73e9a8c7a1c6c32bcb8b053d70ba7cf67b2c5ca9a4f510d47f",
    "codecmanipulator.py": "0d0fcb26b38807ddfd7a012973b6a457258155bc7e13197320004d96fc99370e",
    "mmtokenizer.py": "1473edc703423b06ec404847f63fac4ceff0775c13d674e0d0b4d6e741c0cd05",
    "mm_tokenizer_v0.2_hf/tokenizer.model": "ee5c7cbf32da93989f14d9ba635e3e1d1ab2cc88a92908a5ed0f149375f6ee49",
}
# The functions/classes lifted verbatim out of infer.py (the rest of it is a CLI script that loads
# both models at import time).
UPSTREAM_DEFS = ("BlockTokenRangeProcessor", "stage2_generate", "stage2_inference")

# The model's embedding / lm_head width (config.json `vocab_size`); the mm tokenizer's is 83738.
MODEL_VOCAB = 83_840
MM_VOCAB = 83_738

# --- the mock model (Rust twin: `stage2::tests::MockLm`) -------------------------------------
MOCK_P = 4_194_301  # prime; 3·P < 2^24, so every score is exact in f32
MOCK_A = 2_654_435
CODEC_OFFSET = 45_334
CODEBOOK = 1_024


def mock_key(history: list[int]) -> int:
    return (len(history) * 1_000_003 + history[-1] * 7_919 + history[-2] * 104_729) % MOCK_P


def mock_expected_codebook(last: int) -> int | None:
    """The codebook the mock favours next: 1 after a codebook-0 token, j+1 after codebook j."""
    if CODEC_OFFSET <= last < CODEC_OFFSET + 7 * CODEBOOK:
        return (last - CODEC_OFFSET) // CODEBOOK + 1
    return None


def mock_scores(history: list[int], np):
    """Row scores over the whole model vocabulary: a keyed permutation of `[0, P)`, plus `2P` on
    codes `< 16` of the expected codebook unless `key % 8 == 0` (the 1-in-8 "stray" step that
    picks from the whole slice and so usually lands in a wrong codebook)."""
    key = mock_key(history)
    v = np.arange(MODEL_VOCAB, dtype=np.int64)
    s = (v * MOCK_A + key) % MOCK_P
    expected = mock_expected_codebook(history[-1])
    if expected is not None and key % 8 != 0:
        lo = CODEC_OFFSET + expected * CODEBOOK
        s[lo : lo + 16] += 2 * MOCK_P
    # The upstream processors leave `[mm_vocab 83738, model_vocab)` open, but upstream `ids2npy`
    # asserts every id is `< 57622`, so a pick there aborts the reference. The mock never picks
    # one (score 0), and the port restricts to the slice — identical wherever upstream returns.
    s[MM_VOCAB:] = 0
    return s.astype(np.float32)


class MockModel:
    """Stands in for `model_stage2` in the upstream loop: greedy `generate` over `mock_scores`,
    applying the caller's `logits_processor` exactly as `transformers` greedy search does."""

    def __init__(self, torch, np):
        self.torch, self.np = torch, np

    def generate(self, input_ids, min_new_tokens, max_new_tokens, eos_token_id, pad_token_id,
                 logits_processor):
        torch, np = self.torch, self.np
        assert min_new_tokens == max_new_tokens == 7
        ids = input_ids
        for _ in range(max_new_tokens):
            scores = torch.as_tensor(np.stack([mock_scores(row, np) for row in ids.tolist()]))
            scores = logits_processor(ids, scores)
            nxt = torch.argmax(scores, dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt.to(ids.dtype)], dim=1)
        return ids


def mock_cb0(frames: int, seed: int) -> list[int]:
    return [(t * 37 + (t * t) % 101 + seed) % CODEBOOK for t in range(frames)]


# (name, frames, batch_size): multi-group + partial group + ragged tail; one full chunk + tail;
# tail only (upstream's `num_batch == 0` path — see the patch in `load_upstream`).
MOCK_CASES = (("groups_partial_and_tail", 937, 2, 11), ("one_chunk_and_tail", 337, 4, 5),
              ("tail_only", 45, 4, 3))


# --- shared ---------------------------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upstream_dir() -> Path:
    raw = os.environ.get("YUE_REF_INFERENCE_DIR")
    if not raw:
        sys.exit("set YUE_REF_INFERENCE_DIR to the upstream YuE-v1 `inference/` directory")
    root = Path(raw).resolve()
    for rel, want in UPSTREAM_SHA256.items():
        got = sha256(root / rel)
        if got != want:
            sys.exit(f"{root / rel}: sha256 {got} != pinned {want} (not YuE-v1 {UPSTREAM_COMMIT})")
    return root


def load_upstream(root: Path, model, device):
    """Exec the pinned upstream stage-2 functions in a namespace wired like infer.py's globals."""
    import copy
    from collections import Counter

    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import LogitsProcessor, LogitsProcessorList

    sys.path[:0] = [str(root)]
    from codecmanipulator import CodecManipulator
    from mmtokenizer import _MMSentencePieceTokenizer

    tree = ast.parse((root / "infer.py").read_text(encoding="utf-8"))
    picked = [n for n in tree.body
              if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in UPSTREAM_DEFS]
    assert sorted(n.name for n in picked) == sorted(UPSTREAM_DEFS), [n.name for n in picked]
    ns = {
        "np": np, "torch": torch, "os": os, "copy": copy, "Counter": Counter, "tqdm": tqdm,
        "LogitsProcessor": LogitsProcessor, "LogitsProcessorList": LogitsProcessorList,
        "mmtokenizer": _MMSentencePieceTokenizer(str(root / "mm_tokenizer_v0.2_hf/tokenizer.model")),
        "codectool": CodecManipulator("xcodec", 0, 1),
        "codectool_stage2": CodecManipulator("xcodec", 0, 8),
        "device": device, "model_stage2": model,
    }
    exec(compile(ast.Module(body=picked, type_ignores=[]), str(root / "infer.py"), "exec"), ns)

    # UPSTREAM DEFECT, patched at the call boundary only: a track shorter than one 300-frame chunk
    # has `num_batch == 0`, and `stage2_inference` still calls `stage2_generate(prompt[:, :0],
    # batch_size=0)`, whose `offset_tok_ids` takes `max()` of an empty array and raises — upstream
    # cannot upsample anything under 6 s. Zero full chunks evidently mean zero output (the tail
    # call right after handles the frames), so that one call returns an empty id row; every other
    # call reaches the verbatim function. The Rust port does the same (`stage2::chunk_plan`).
    generate = ns["stage2_generate"]

    def stage2_generate(model, prompt, batch_size=16):
        if prompt.shape[-1] == 0:
            return np.zeros((0,), dtype=np.int64)
        return generate(model, prompt, batch_size=batch_size)

    ns["stage2_generate"] = stage2_generate
    return ns


def run_stage2(ns, cb0_rows: list[list[int]], batch_size: int):
    """Run upstream `stage2_inference` over each codebook-0 row (as the stage-1 `.npy` files it
    reads) and return each fixed `[8, T]` grid, plus the count of codes `fix_output` repaired."""
    import numpy as np

    out = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "s2").mkdir()
        paths = []
        for i, row in enumerate(cb0_rows):
            p = tmp / f"case{i}.npy"
            np.save(p, np.asarray([row], dtype=np.int32))  # stage-1 writes [1, T] codebook-0 codes
            paths.append(str(p))
        results = ns["stage2_inference"](ns["model_stage2"], paths, str(tmp / "s2"), batch_size=batch_size)
        for path in results:
            grid = np.load(path)
            assert grid.shape[0] == 8, grid.shape
            out.append(grid)
    return out


def hex_row(row) -> str:
    return "".join(f"{int(c):03x}" for c in row)


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size} bytes)")


# --- mock -----------------------------------------------------------------------------------
def cmd_mock(_args) -> None:
    import numpy as np
    import torch

    root = upstream_dir()
    model = MockModel(torch, np)
    ns = load_upstream(root, model, torch.device("cpu"))

    # Count repairs by wrapping ids2npy: the grid before `fix_output` is its return value.
    raw = []
    ids2npy = ns["codectool_stage2"].ids2npy
    ns["codectool_stage2"].ids2npy = lambda ids: raw.append(ids2npy(ids)) or raw[-1]

    cases = []
    for name, frames, batch, seed in MOCK_CASES:
        cb0 = mock_cb0(frames, seed)
        raw.clear()
        (grid,) = run_stage2(ns, [cb0], batch)
        (pre,) = raw
        invalid = int(((pre < 0) | (pre > 1023)).sum())
        assert (grid[0] == np.asarray(cb0)).all()
        assert ((grid >= 0) & (grid <= 1023)).all(), f"{name}: fix_output left an invalid code"
        assert invalid > 0, f"{name}: the mock must exercise fix_output"
        cases.append({
            "name": name, "frames": frames, "batch_size": batch, "cb0_seed": seed,
            "repaired_codes": invalid,
            "codebooks_hex": [hex_row(r) for r in grid],
        })
        print(f"{name}: T={frames} batch={batch} repaired={invalid}")
    write(FIXTURES / "stage2_mock_reference.json", {
        "producer": "scripts/reference/yue_stage2_reference.py mock",
        "upstream": {"repo": "multimodal-art-projection/YuE", "commit": UPSTREAM_COMMIT,
                     "functions": list(UPSTREAM_DEFS), "sha256": UPSTREAM_SHA256},
        "mock": {"p": MOCK_P, "a": MOCK_A, "model_vocab": MODEL_VOCAB,
                 "cb0": "(t*37 + (t*t)%101 + cb0_seed) % 1024"},
        "encoding": "codebooks_hex: 8 rows, 3 hex digits per code",
        "cases": cases,
    })


# --- real -----------------------------------------------------------------------------------
def synthetic_clip(np, seconds: float, sr: int = 16_000):
    """A deterministic, arithmetic, music-shaped 16 kHz clip: a plucked-string arpeggio over a
    sustained bass with a vibrato lead — no third-party audio."""
    t = np.arange(int(seconds * sr)) / sr
    x = 0.25 * np.sin(2 * np.pi * 110.0 * t)
    notes = (261.63, 329.63, 392.0, 523.25)
    step = 0.25
    for i, f in enumerate(notes * math.ceil(seconds / (step * len(notes)))):
        start = i * step
        env = np.where(t >= start, np.exp(-6.0 * np.clip(t - start, 0, None)), 0.0)
        x += 0.2 * env * np.sin(2 * np.pi * f * (t - start))
    x += 0.12 * np.sin(2 * np.pi * 659.25 * t + 3.0 * np.sin(2 * np.pi * 5.5 * t))
    return (0.8 * x / np.max(np.abs(x))).astype(np.float32)


def encode_cb0(root: Path, np, torch) -> list[int]:
    """Upstream's ICL encode: `codec_model.encode(audio, target_bw=0.5)` → codebook 0."""
    xcodec = root / "xcodec_mini_infer"
    sys.path[:0] = [str(xcodec), str(xcodec / "descriptaudiocodec")]
    from models.soundstream_hubert_new import SoundStream
    from omegaconf import OmegaConf

    cwd = os.getcwd()
    os.chdir(root)  # SoundStream resolves ./xcodec_mini_infer/semantic_ckpts relative to cwd
    try:
        cfg = OmegaConf.load("./xcodec_mini_infer/final_ckpt/config.yaml")
        codec = SoundStream(**cfg.generator.config)
        state = torch.load("./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth", map_location="cpu",
                           weights_only=False)
        codec.load_state_dict(state["codec_model"])
        codec.eval()
        clip = torch.as_tensor(synthetic_clip(np, 1.0))[None, None, :]
        with torch.no_grad():
            raw = codec.encode(clip, target_bw=0.5)
        raw = raw.transpose(0, 1).cpu().numpy().astype(np.int16)  # infer.py's ICL path
        return [int(c) for c in raw[0][0]]
    finally:
        os.chdir(cwd)


# (name, first frame, frames): two single-chunk (ragged-tail) cases cut from the 50-frame encode.
REAL_CASES = (("encode_0_40", 0, 40), ("encode_17_33", 17, 16))


def cmd_real(args) -> None:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    root = upstream_dir()
    snap = os.environ.get("YUE_S2_BF16_SNAPSHOT")
    if not snap:
        sys.exit("set YUE_S2_BF16_SNAPSHOT to the dense stage-2 snapshot directory")
    torch.manual_seed(0)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    cb0_all = encode_cb0(root, np, torch)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.compute_dtype]
    # infer.py: from_pretrained(torch_dtype=bfloat16, attn_implementation="sdpa"); the checkpoint
    # is bf16, so float32 here is the same weights upcast (what candle's CPU lane computes in).
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=dtype, attn_implementation="sdpa")
    model.eval()
    ns = load_upstream(root, model, torch.device("cpu"))
    raw = []
    ids2npy = ns["codectool_stage2"].ids2npy
    ns["codectool_stage2"].ids2npy = lambda ids: raw.append(ids2npy(ids)) or raw[-1]
    rows = [cb0_all[a : a + n] for _, a, n in REAL_CASES]
    grids = run_stage2(ns, rows, batch_size=4)
    cases = []
    for (name, a, n), cb0, grid, pre in zip(REAL_CASES, rows, grids, raw):
        assert (grid[0] == np.asarray(cb0)).all()
        cases.append({
            "name": name, "frames": n, "cb0": cb0,
            "repaired_codes": int(((pre < 0) | (pre > 1023)).sum()),
            "codebooks": [[int(c) for c in r] for r in grid],
        })
        print(f"{name}: T={n} repaired={cases[-1]['repaired_codes']}")
    out = args.output or FIXTURES / f"stage2_real_reference{'' if args.compute_dtype == 'float32' else '_' + args.compute_dtype}.json"
    write(Path(out), {
        "producer": f"scripts/reference/yue_stage2_reference.py real --compute-dtype {args.compute_dtype}",
        "upstream": {"repo": "multimodal-art-projection/YuE", "commit": UPSTREAM_COMMIT,
                     "functions": list(UPSTREAM_DEFS), "sha256": UPSTREAM_SHA256},
        "checkpoint": {"repo": "m-a-p/YuE-s2-1B-general", "revision": STAGE2_REVISION,
                       "weights": "bf16", "compute_dtype": args.compute_dtype, "device": "cpu",
                       "attn_implementation": "sdpa"},
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
        "cb0_source": "xcodec encode (target_bw=0.5) of synthetic_clip(1.0 s) — codebook 0",
        "encoded_cb0": cb0_all,
        "cases": cases,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mock").set_defaults(fn=cmd_mock)
    real = sub.add_parser("real")
    real.add_argument("--compute-dtype", choices=("float32", "bfloat16"), default="float32")
    real.add_argument("--output", help="override the output path")
    real.set_defaults(fn=cmd_real)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
