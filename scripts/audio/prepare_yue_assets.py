#!/usr/bin/env python3
"""Prepare the YuE (epic sc-19373) weight rehosts locally: safetensors conversion + q8/q4 tiers.

sc-19374 / sc-19375. Produces, per SceneWorks rehost repo, a staging directory that is uploaded
verbatim (see the generated ``PUBLISH.sh``; this script never touches the Hub):

``xcodec`` — ``m-a-p/xcodec_mini_infer`` converted to safetensors-only:

* ``final_ckpt/ckpt_00360000.safetensors`` — the ``codec_model`` state dict (SoundStream encoder +
  HuBERT semantic branch + RVQ + decoder), float32 exactly as upstream. The training-only
  ``optimizer_*`` / ``lr_scheduler_*`` / ``mfd`` (discriminator) entries are dropped.
* ``decoders/decoder_131000.safetensors`` (vocal Vocos) and ``decoders/decoder_151000.safetensors``
  (instrumental Vocos), float32.
* ``semantic_ckpts/hf_1_325000/model.safetensors`` + its ``config.json`` /
  ``preprocessor_config.json`` (the HuBERT the upstream ``SoundStream.__init__`` instantiates before
  the codec state dict overwrites it — kept so the reference constructor still resolves).
* ``final_ckpt/config.yaml``, ``decoders/config.yaml``, ``mm_tokenizer_v0.2_hf/tokenizer.model``
  (+ a derived ``tokenizer.json``).

Every converted file is re-read and compared tensor-by-tensor (``torch.equal``, same key set,
dtype and shape) against the upstream ``.pth`` / ``.bin``, and each conversion is performed twice
and byte-compared so the recorded sha256 is reproducible (``save_file`` header order).

``lm`` — one of the seven YuE LMs (``m-a-p/YuE-s1-7B-anneal-{en,zh,jp-kr}-{cot,icl}``,
``m-a-p/YuE-s2-1B-general``), already downloaded at its pinned revision into ``<dest>/bf16``:

* ``bf16/`` — the upstream safetensors snapshot, verified safetensors-only (no pickle, no
  AppleDouble ``._*`` sidecars), plus a derived ``tokenizer.json``.
* ``q8/`` and ``q4/`` — candle-llm ``prepare_snapshot`` output (the ``prepare_snapshot`` example in
  ``crates/llm/candle-llm``), **pre-quantized**: every layer projection is stored as a GGML block
  tensor (``U8`` ``[rows, blocks_per_row, block_bytes]`` — Q8_0 in q8; Q4_K in q4, Q4_0 where the
  input dim is not 256-aligned) that the candle loader rebuilds directly, never re-quantizing;
  embeddings, LM head and norms stay bf16. ``config.json`` carries
  ``quantization: {bits, storage: "ggml"}``; ``tokenizer.model`` and ``generation_config.json``
  are copied from bf16. Each tier directory is self-contained, so a user downloads only the tier
  they pick.

The derived ``tokenizer.json`` (candle-llm's loader and preparer require one) is built from the
mm SentencePiece ``tokenizer.model`` as a byte-fallback BPE with the mm tokenizer's special tokens
added at their ids, and is checked id-for-id against the upstream ``_MMSentencePieceTokenizer``
over the upstream prompt examples plus a seeded fuzz corpus; any mismatch aborts.

Each repo also gets ``LICENSE`` + ``NOTICE`` (the upstream YuE Apache-2.0 files — Section 4(d)
requires the NOTICE be retained), ``README.md``, ``SOURCE_REVISION.json`` (upstream repo +
revision, YuE code commit, sha256 of every file) and ``sceneworks-tiers.json``.

Every heavy subprocess (the Rust preparer) runs under an RSS guard that kills it past
``--rss-limit-gb`` (default 48). The torch conversions run in-process; run the whole script under
an external guard as well.

Run with the YuE reference venv (torch, safetensors, sentencepiece, tokenizers, transformers):

    PY=/Users/michael/.cache/sceneworks-yue-ref/venv/bin/python
    cargo build --release -p candle-llm --example prepare_snapshot
    $PY scripts/audio/prepare_yue_assets.py xcodec --src <xcodec_mini_infer> --dest <assets>/xcodec-mini-infer \\
        --yue <YuE clone @ YuE-v1> --revision <hf sha>
    $PY scripts/audio/prepare_yue_assets.py lm --dest <assets>/yue-s2-1b-general-candle \\
        --yue <YuE clone> --source-repo m-a-p/YuE-s2-1B-general --revision <hf sha> \\
        --preparer target/release/examples/prepare_snapshot

``lm --retier`` regenerates q8/q4 (and the metadata) for a repo already staged at the same
upstream revision, without a fresh download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

RSS_LIMIT_GB_DEFAULT = 48.0

MM_TOKENIZER_REL = "inference/mm_tokenizer_v0.2_hf/tokenizer.model"

XCODEC_FILES = {
    # upstream relative path -> (converted relative path, state-dict key or None for a flat dict)
    "final_ckpt/ckpt_00360000.pth": ("final_ckpt/ckpt_00360000.safetensors", "codec_model"),
    "decoders/decoder_131000.pth": ("decoders/decoder_131000.safetensors", None),
    "decoders/decoder_151000.pth": ("decoders/decoder_151000.safetensors", None),
    "semantic_ckpts/hf_1_325000/pytorch_model.bin": (
        "semantic_ckpts/hf_1_325000/model.safetensors",
        None,
    ),
}
XCODEC_COPIES = [
    "final_ckpt/config.yaml",
    "decoders/config.yaml",
    "semantic_ckpts/hf_1_325000/config.json",
    "semantic_ckpts/hf_1_325000/preprocessor_config.json",
]

STAGE1_VOCAB = 83_968
STAGE2_VOCAB = 83_840


# ---------------------------------------------------------------------------------------------
# helpers


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tree_rss_kb(root_pid: int) -> int:
    """Summed RSS (KiB) of ``root_pid`` and all its descendants, via ``ps``."""
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True, encoding="utf-8").stdout
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        pid, ppid, kb = (int(p) for p in parts)
        rss[pid] = kb
        children.setdefault(ppid, []).append(pid)
    total, stack = 0, [root_pid]
    while stack:
        pid = stack.pop()
        total += rss.get(pid, 0)
        stack.extend(children.get(pid, []))
    return total


def run_guarded(cmd: list[str], limit_gb: float, env: dict | None = None) -> tuple[str, float]:
    """Run ``cmd`` and kill it if its process-tree RSS exceeds ``limit_gb``. Returns (stdout, peak GB)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8", env=env, start_new_session=True)
    peak = 0.0
    limit_kb = int(limit_gb * 1024 * 1024)
    while proc.poll() is None:
        kb = tree_rss_kb(proc.pid)
        peak = max(peak, kb / 1024 / 1024)
        if kb > limit_kb:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise SystemExit(f"RSS guard: {cmd[0]} exceeded {limit_gb} GB ({kb / 1024 / 1024:.1f} GB); killed")
        time.sleep(0.5)
    stdout = proc.stdout.read() if proc.stdout else ""
    if proc.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)} failed with exit code {proc.returncode}")
    return stdout, peak


def safetensors_header(path: Path) -> dict:
    with path.open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def dir_headers(d: Path) -> dict:
    merged: dict = {}
    for p in sorted(d.glob("*.safetensors")):
        merged.update(safetensors_header(p))
    return merged


def import_mm_tokenizer(yue: Path):
    sys.path.insert(0, str(yue / "inference"))
    from mmtokenizer import _MMSentencePieceTokenizer  # type: ignore

    return _MMSentencePieceTokenizer(str(yue / MM_TOKENIZER_REL))


# ---------------------------------------------------------------------------------------------
# tokenizer.json


def build_tokenizer_json(model_file: Path, yue: Path, out: Path) -> dict:
    """Derive a HF fast ``tokenizer.json`` from the mm SentencePiece model and verify id parity."""
    from sentencepiece import sentencepiece_model_pb2 as pb
    from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers
    from transformers.tokenization_utils_base import generate_merges

    proto = pb.ModelProto()
    proto.ParseFromString(model_file.read_bytes())
    if proto.trainer_spec.model_type != 2 or not proto.trainer_spec.byte_fallback:
        raise SystemExit("mm tokenizer.model is expected to be a byte-fallback BPE model")
    vocab = {p.piece: i for i, p in enumerate(proto.pieces)}
    scores = {p.piece: p.score for p in proto.pieces}
    tok = Tokenizer(
        models.BPE(
            vocab=vocab,
            merges=generate_merges(vocab, scores),
            unk_token="<unk>",
            fuse_unk=True,
            byte_fallback=True,
        )
    )
    # SentencePiece identity normalizer + add_dummy_prefix, applied per split segment, which is
    # exactly what `_MMSentencePieceTokenizer.tokenize` does (it encodes each run between specials).
    tok.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    tok.decoder = decoders.Sequence(
        [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(" ", 1, 0)]
    )
    mm = import_mm_tokenizer(yue)
    if (yue / MM_TOKENIZER_REL).read_bytes() != model_file.read_bytes():
        raise SystemExit(f"{model_file} differs from the YuE mm tokenizer model")
    specials = sorted(mm._special_tokens.items(), key=lambda kv: kv[1])
    tok.add_special_tokens([AddedToken(t, special=True, normalized=False) for t, _ in specials])
    for t, i in specials:
        if tok.token_to_id(t) != i:
            raise SystemExit(f"special {t}: tokenizer.json id {tok.token_to_id(t)} != mm id {i}")

    corpus = [
        "[verse]\nhello world",
        "Generate music from the given lyrics segment by segment.\n[Genre] inspiring female uplifting pop",
        "[start_of_segment][verse]\nline one\n[end_of_segment]",
        "<SOA><stage_1>x<EOA> <stage_2>  y",
        "  leading  spaces\tand tabs\n\nnewlines ",
        "中文歌词测试，我爱你。日本語の歌詞 한국어 가사",
        "emoji 🎵🎶 and bytes \x00\x7f",
        "",
    ]
    for p in sorted((yue / "prompt_egs").glob("*.txt")):
        corpus.append(p.read_text(encoding="utf-8"))
    lyrics = "".join(corpus)
    rng = random.Random(19374)
    specials_text = [t for t, _ in specials]
    for _ in range(3000):
        a = rng.randrange(len(lyrics))
        s = lyrics[a : a + rng.randrange(1, 200)]
        if rng.random() < 0.3:
            k = rng.randrange(len(s) + 1)
            s = s[:k] + rng.choice(specials_text) + s[k:]
        corpus.append(s)
    mismatches = 0
    for text in corpus:
        if mm.tokenize(text) != tok.encode(text, add_special_tokens=False).ids:
            mismatches += 1
    if mismatches:
        raise SystemExit(f"tokenizer.json parity: {mismatches}/{len(corpus)} mismatches vs mm tokenizer")
    tok.save(str(out))
    return {"parity_cases": len(corpus), "mismatches": 0, "vocab_size": tok.get_vocab_size()}


# ---------------------------------------------------------------------------------------------
# xcodec


def _state_dict(path: Path, key: str | None):
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj[key] if key else obj
    tensors = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    if len(tensors) != len(sd):
        raise SystemExit(f"{path}: non-tensor entries in the state dict")
    return tensors


def convert_state_dict(src: Path, key: str | None, dest: Path) -> dict:
    import torch
    from safetensors.torch import load_file, save_file

    sd = _state_dict(src, key)
    # `.contiguous().clone()`: never serialize a strided view or a shared storage (the classic
    # non-contiguous `.numpy()` corruption); clone also breaks storage sharing between keys.
    out = {k: v.detach().contiguous().clone() for k, v in sd.items()}
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dest), metadata={"format": "pt"})
    digest = sha256(dest)
    # Reproducibility: a second conversion must be byte-identical.
    again = dest.with_suffix(".again.safetensors")
    save_file(out, str(again), metadata={"format": "pt"})
    if sha256(again) != digest:
        raise SystemExit(f"{dest}: safetensors conversion is not byte-deterministic")
    again.unlink()
    # Round trip: every tensor equal (value, dtype, shape) to the upstream pickle.
    back = load_file(str(dest))
    if set(back) != set(sd):
        raise SystemExit(f"{dest}: key set differs from {src}")
    for k, v in sd.items():
        b = back[k]
        if b.dtype != v.dtype or b.shape != v.shape or not torch.equal(b, v):
            raise SystemExit(f"{dest}: tensor {k} differs from {src}")
    dtypes = sorted({str(v.dtype).replace("torch.", "") for v in sd.values()})
    return {
        "source": str(src.name),
        "source_key": key,
        "tensors": len(sd),
        "params": int(sum(v.numel() for v in sd.values())),
        "dtypes": dtypes,
        "roundtrip": "tensor-equal",
        "sha256": digest,
    }


def cmd_xcodec(a: argparse.Namespace) -> None:
    src, dest, yue = Path(a.src), Path(a.dest), Path(a.yue)
    dest.mkdir(parents=True, exist_ok=True)
    report: dict = {"conversions": {}}
    for rel, (out_rel, key) in XCODEC_FILES.items():
        report["conversions"][out_rel] = convert_state_dict(src / rel, key, dest / out_rel)
        print(f"converted {rel} -> {out_rel}: {report['conversions'][out_rel]['tensors']} tensors, tensor-equal")
    for rel in XCODEC_COPIES:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, dest / rel)
    tok_dir = dest / "mm_tokenizer_v0.2_hf"
    tok_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(yue / MM_TOKENIZER_REL, tok_dir / "tokenizer.model")
    report["tokenizer_json"] = build_tokenizer_json(tok_dir / "tokenizer.model", yue, tok_dir / "tokenizer.json")
    write_json(
        dest / "sceneworks-tiers.json",
        {
            "schema": 1,
            "tiered": False,
            "carve_out": (
                "epic sc-19373 R2 (approved): the xcodec codec and both Vocos decoders are NOT "
                "quantized with the LM tier. Every tier (bf16, q8, q4) resolves to these same "
                "files, stored at upstream precision."
            ),
            "tiers": {tier: {"dir": ".", "storage_dtype": "float32"} for tier in ("bf16", "q8", "q4")},
        },
    )
    finish_repo(dest, yue, a.source_repo, a.revision, report, xcodec_readme(a.source_repo, a.revision, report))


# ---------------------------------------------------------------------------------------------
# LMs


def cmd_lm(a: argparse.Namespace) -> None:
    dest, yue = Path(a.dest), Path(a.yue)
    bf16 = dest / "bf16"
    marker = bf16 / ".download-complete"
    if a.retier:
        staged = json.loads((dest / "SOURCE_REVISION.json").read_text(encoding="utf-8"))
        if (staged["upstream_repo"], staged["upstream_revision"]) != (a.source_repo, a.revision):
            raise SystemExit(f"{dest} is staged from a different upstream revision")
    elif not marker.is_file() or marker.read_text(encoding="utf-8").strip() != f"{a.source_repo}@{a.revision}":
        raise SystemExit(f"{bf16} is not a completed download of {a.source_repo}@{a.revision}")
    shutil.rmtree(bf16 / ".cache", ignore_errors=True)
    for p in bf16.rglob("*"):
        if p == marker:
            continue
        if p.name.startswith("._") or p.suffix in (".bin", ".pth", ".pt", ".ckpt", ".pkl"):
            raise SystemExit(f"{p}: not a safetensors-only snapshot")
    cfg = json.loads((bf16 / "config.json").read_text(encoding="utf-8"))
    if cfg.get("architectures") != ["LlamaForCausalLM"]:
        raise SystemExit(f"{bf16}: unexpected architectures {cfg.get('architectures')}")
    vocab = cfg["vocab_size"]
    if vocab not in (STAGE1_VOCAB, STAGE2_VOCAB):
        raise SystemExit(f"{bf16}: unexpected vocab_size {vocab}")
    headers = dir_headers(bf16)
    if not headers:
        raise SystemExit(f"{bf16}: no safetensors")
    report: dict = {
        "vocab_size": vocab,
        "bf16": {"tensors": len(headers), "dtypes": sorted({h["dtype"] for h in headers.values()})},
    }
    report["tokenizer_json"] = build_tokenizer_json(bf16 / "tokenizer.model", yue, bf16 / "tokenizer.json")

    env = dict(os.environ, CANDLE_LLM_DEVICE="cpu")
    for tier in ("q8", "q4"):
        out = dest / tier
        shutil.rmtree(out, ignore_errors=True)
        stdout, peak = run_guarded([a.preparer, str(bf16), str(out), tier], a.rss_limit_gb, env)
        prepared = json.loads(stdout.strip().splitlines()[-1])
        for name in ("tokenizer.model", "generation_config.json"):
            shutil.copy2(bf16 / name, out / name)
        qcfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
        bits = {"q8": 8, "q4": 4}[tier]
        if qcfg.get("quantization") != {"bits": bits, "storage": "ggml"}:
            raise SystemExit(f"{out}: quantization block {qcfg.get('quantization')} != bits {bits} ggml")
        ggml_types = verify_tier(out, headers, tier)
        tier_bytes = sum(p.stat().st_size for p in out.glob("*.safetensors"))
        report[tier] = {
            "tensors": prepared["num_tensors"],
            "quantization": qcfg["quantization"],
            "ggml": ggml_types,
            "safetensors_bytes": tier_bytes,
            "preparer_peak_rss_gb": round(peak, 1),
        }
        print(
            f"{dest.name}/{tier}: prepared ({prepared['num_tensors']} tensors, "
            f"{tier_bytes / 1e9:.2f} GB, {ggml_types}, peak RSS {peak:.1f} GB)"
        )

    if marker.is_file():
        marker.unlink()
    write_json(
        dest / "sceneworks-tiers.json",
        {
            "schema": 1,
            "tiered": True,
            "tiers": {
                "bf16": {"dir": "bf16", "format": "hf-safetensors", "storage_dtype": "bfloat16"},
                **{
                    tier: {
                        "dir": tier,
                        "format": "candle-llm-prepared",
                        "prequantized": True,
                        "quantization": report[tier]["quantization"],
                        "projection_storage": "ggml-blocks (U8 [rows, blocks_per_row, block_bytes])",
                        "ggml": report[tier]["ggml"],
                        "dense_tensors_dtype": "bfloat16 (embeddings, lm_head, norms)",
                    }
                    for tier in ("q8", "q4")
                },
            },
            "companion_unquantized": {
                "repo": "SceneWorks/xcodec-mini-infer",
                "note": (
                    "epic sc-19373 R2 carve-out (approved): the xcodec codec and Vocos decoders "
                    "are not tiered; every LM tier pairs with the same upstream-precision files"
                ),
            },
        },
    )
    finish_repo(dest, yue, a.source_repo, a.revision, report, lm_readme(a.source_repo, a.revision, report, cfg))


# GGML block byte size -> (type, weights per block), the candle-llm stored-block convention.
GGML_BLOCKS = {18: ("Q4_0", 32), 34: ("Q8_0", 32), 144: ("Q4_K", 256)}


def expected_ggml(tier: str, cols: int) -> str:
    if tier == "q8":
        return "Q8_0"
    return "Q4_K" if cols % 256 == 0 else "Q4_0"


def verify_tier(out: Path, bf16_headers: dict, tier: str) -> dict:
    """Check a prepared tier holds pre-quantized projections and dense everything else.

    Every ``*_proj.weight`` must be a ``U8`` GGML block tensor of the expected type whose logical
    shape is the bf16 weight's; every other tensor must match bf16 exactly (dtype + shape).
    Returns ``{ggml type: projection count}``.
    """
    qh = dir_headers(out)
    if set(qh) != set(bf16_headers):
        raise SystemExit(f"{out}: tensor names differ from bf16")
    counts: dict[str, int] = {}
    for k, h in bf16_headers.items():
        q = qh[k]
        if k.endswith("_proj.weight"):
            rows, cols = h["shape"]
            if q["dtype"] != "U8" or len(q["shape"]) != 3:
                raise SystemExit(f"{out}: {k} is not stored quantized ({q['dtype']} {q['shape']})")
            name, block = GGML_BLOCKS.get(q["shape"][2], (None, 0))
            if name != expected_ggml(tier, cols) or q["shape"][:2] != [rows, cols // block]:
                raise SystemExit(f"{out}: {k} stored as {q['shape']}, expected {expected_ggml(tier, cols)} of [{rows}, {cols}]")
            counts[name] = counts.get(name, 0) + 1
        elif q["shape"] != h["shape"] or q["dtype"] != h["dtype"]:
            raise SystemExit(f"{out}: {k} shape/dtype differs from bf16")
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------------------------
# per-repo metadata


def finish_repo(dest: Path, yue: Path, source_repo: str, revision: str, report: dict, readme: str) -> None:
    shutil.copy2(yue / "LICENSE", dest / "LICENSE")
    shutil.copy2(yue / "NOTICE", dest / "NOTICE")
    (dest / "README.md").write_text(readme, encoding="utf-8")
    yue_commit = subprocess.run(
        ["git", "-C", str(yue), "rev-parse", "HEAD"], capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout.strip()
    files = {}
    for p in sorted(dest.rglob("*")):
        if p.is_file() and p.name != "SOURCE_REVISION.json":
            files[str(p.relative_to(dest))] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    write_json(
        dest / "SOURCE_REVISION.json",
        {
            "upstream_repo": source_repo,
            "upstream_revision": revision,
            "upstream_code": "https://github.com/multimodal-art-projection/YuE",
            "upstream_code_commit": yue_commit,
            "license": "apache-2.0",
            "prepared_by": "inference scripts/audio/prepare_yue_assets.py (sc-19374 / sc-19375)",
            "report": report,
            "files": files,
        },
    )


FRONT_MATTER = """---
license: apache-2.0
library_name: safetensors
tags:
  - music
  - lyrics2song
  - yue
  - sceneworks
---
"""


def gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def fmt_types(types: dict) -> str:
    return " + ".join(f"{k} ×{v}" for k, v in types.items())


def lm_readme(repo: str, rev: str, report: dict, cfg: dict) -> str:
    stage = "stage-1 (7B lyrics → codebook-0 LM)" if report["vocab_size"] == STAGE1_VOCAB else "stage-2 (1B codebook upsampler)"
    return FRONT_MATTER + f"""
# {repo.split('/')[-1]} — SceneWorks rehost (bf16 / q8 / q4)

A redistribution mirror of [`{repo}`](https://huggingface.co/{repo}) at revision `{rev}`, the YuE
{stage} from HKUST / M-A-P, published by **SceneWorks** so the weights resolve by an immutable
commit SHA for SceneWorks Inference's candle YuE engine. It is **not** an official M-A-P
distribution.

## Layout

| Dir | Contents |
|---|---|
| `bf16/` | the upstream safetensors snapshot, unmodified, plus a derived `tokenizer.json` |
| `q8/` | pre-quantized Q8 tier ({gb(report['q8']['safetensors_bytes'])}): projections stored as GGML {fmt_types(report['q8']['ggml'])} blocks |
| `q4/` | pre-quantized Q4 tier ({gb(report['q4']['safetensors_bytes'])}): projections stored as GGML {fmt_types(report['q4']['ggml'])} blocks |

Each tier directory is self-contained (config, tokenizer, weights) — download only the tier you
use. The q8/q4 tiers are produced from `bf16/` by candle-llm's snapshot preparer
(`prepare_snapshot`): every attention/MLP projection is quantized once and stored as its raw GGML
blocks — a `U8` tensor `[rows, blocks_per_row, block_bytes]`, the block type given by
`block_bytes` (18 = Q4_0, 34 = Q8_0, 144 = Q4_K) — which the candle loader rebuilds directly
(no dequantize/re-quantize). `config.json` carries `quantization: {{bits, storage: "ggml"}}`.
Embeddings, the LM head and norms stay bf16 in every tier. The xcodec codec + Vocos decoders are **not** tiered
(approved carve-out); every tier pairs with
[`SceneWorks/xcodec-mini-infer`](https://huggingface.co/SceneWorks/xcodec-mini-infer).

`tokenizer.model` is the upstream mm SentencePiece model (the source of truth). `tokenizer.json`
is derived from it (byte-fallback BPE + the mm special tokens at their ids) and verified id-for-id
against the upstream `_MMSentencePieceTokenizer` over {report['tokenizer_json']['parity_cases']} cases
(0 mismatches).

Vocabulary width {cfg['vocab_size']}, context {cfg['max_position_embeddings']}.

## Provenance

`SOURCE_REVISION.json` records the upstream repo + revision, the YuE code commit, and the sha256
of every file. Prepared by `scripts/audio/prepare_yue_assets.py` in the SceneWorks Inference
repository.

## License

**Apache-2.0**, © 2025 Ruibin Yuan and core contributors from M-A-P and HKUST. See `LICENSE` and
`NOTICE` (retained per Section 4(d) of the Apache License). Upstream project:
https://github.com/multimodal-art-projection/YuE
"""


def xcodec_readme(repo: str, rev: str, report: dict) -> str:
    rows = "\n".join(
        f"| `{k}` | {v['tensors']} | {', '.join(v['dtypes'])} | `{v['sha256'][:16]}…` |" for k, v in report["conversions"].items()
    )
    return FRONT_MATTER + f"""
# xcodec-mini-infer — SceneWorks safetensors rehost

A safetensors-only redistribution mirror of [`{repo}`](https://huggingface.co/{repo}) at revision
`{rev}` — the xcodec codec (SoundStream/SEANet + RVQ + HuBERT semantic branch), the two Vocos
44.1 kHz upsamplers (vocal `decoder_131000`, instrumental `decoder_151000`) and the mm tokenizer
used by YuE — published by **SceneWorks** for SceneWorks Inference's candle YuE engine. It is
**not** an official M-A-P distribution.

## Contents

| File | Tensors | dtype | sha256 |
|---|---|---|---|
{rows}

Each converted file is tensor-equal (value, dtype, shape) to the upstream pickle. From
`ckpt_00360000.pth` only the `codec_model` state dict is kept; the training-only optimizer,
LR-scheduler and discriminator (`mfd`) entries are dropped. The YAML configs, the HuBERT
`config.json` / `preprocessor_config.json`, and `mm_tokenizer_v0.2_hf/tokenizer.model` are copied
verbatim; `mm_tokenizer_v0.2_hf/tokenizer.json` is derived (see the LM repos' READMEs).

No Python code from the upstream repo is redistributed.

## Tiers

Not tiered — approved carve-out (epic sc-19373 R2): every LM tier (bf16 / q8 / q4) uses these same
files at upstream precision. `sceneworks-tiers.json` records this.

## License

**Apache-2.0**, © 2025 Ruibin Yuan and core contributors from M-A-P and HKUST. See `LICENSE` and
`NOTICE` (retained per Section 4(d) of the Apache License).
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("xcodec")
    x.add_argument("--src", required=True, help="hf download of m-a-p/xcodec_mini_infer")
    x.add_argument("--dest", required=True)
    x.add_argument("--yue", required=True, help="YuE clone at the YuE-v1 branch")
    x.add_argument("--source-repo", default="m-a-p/xcodec_mini_infer")
    x.add_argument("--revision", required=True)
    x.set_defaults(func=cmd_xcodec)
    m = sub.add_parser("lm")
    m.add_argument("--dest", required=True, help="repo staging dir; upstream snapshot already in <dest>/bf16")
    m.add_argument("--yue", required=True)
    m.add_argument("--source-repo", required=True)
    m.add_argument("--revision", required=True)
    m.add_argument("--preparer", required=True, help="built candle-llm `prepare_snapshot` example binary")
    m.add_argument("--rss-limit-gb", type=float, default=RSS_LIMIT_GB_DEFAULT)
    m.add_argument("--retier", action="store_true", help="regenerate q8/q4 of an already staged repo")
    m.set_defaults(func=cmd_lm)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
