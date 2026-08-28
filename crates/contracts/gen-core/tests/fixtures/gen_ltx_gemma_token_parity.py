#!/usr/bin/env python3
"""Regenerate the LTX Gemma token-id parity goldens (sc-18762).

Runs the *reference* path — Lightricks/LTX-2 @ d151147788a9284cca791edc6ce898007e727fe6,
``ltx_core.text_encoders.gemma.gemma_assets.build_gemma_hf_tokenizer`` +
``ltx_core.text_encoders.gemma.tokenizer.LTXGemmaTokenizer.tokenize_with_weights`` — over the two
real Gemma tokenizers SceneWorks ships against, and writes the ids/masks the Rust
``gen_core::gemma_assets::LtxGemmaTokenizer`` must reproduce exactly.

Both generations go through the *same* upstream class on purpose: that is what proves the
exactly-one-BOS policy is generation-agnostic (Gemma 3's post_processor already emits ``<bos>``,
Gemma 4's is a pass-through and emits none).

Usage (needs `pip install transformers tokenizers`; no network, no weights are read):

    python3 gen_ltx_gemma_token_parity.py \
        --gemma4-te /path/to/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
        --gemma3-root /path/to/ltx-2.3/gemma \
        --out-dir .

The Gemma 4 assets are unpacked straight out of the single-file text encoder with a header parse
plus two seeks — no tensor payload and no HF cache are touched.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

# gemma_assets.py
GEMMA_CONFIG_METADATA_KEY = "gemma_config"
TOKENIZER_JSON_TENSOR_KEY = "tokenizer_json"
HF_ASSET_TENSOR_PREFIX = "hf_asset__"
TOKENIZER_MAX_LENGTH = 1024
_TOKENIZER_CONFIG_SKIP = frozenset(
    {
        "tokenizer_class",
        "auto_map",
        "model_max_length",
        "backend",
        "is_local",
        "local_files_only",
        "processor_class",
        "added_tokens_decoder",
    }
)

# The parity prompt set: empty, whitespace-only, plain ASCII, whitespace-padded (exercises the
# upstream ``text.strip()``), non-ASCII + CJK + emoji, a prompt whose literal text already opens
# with the ``<bos>`` added-token string (must NOT be double-BOSed), and one long enough to truncate.
PROMPTS = {
    "empty": "",
    "whitespace_only": "   \n\t  ",
    "ascii": "A cinematic shot of a red fox running through snow at dawn.",
    "leading_trailing_ws": "   a red fox in the snow   \n",
    "non_ascii": "Une caméra suit un renard roux — 日本語のテキスト, Ω, naïve café 🎬🎞️",
    "literal_bos_prefix": "<bos>a red fox in the snow",
    "very_long": (
        "a red fox runs across the frozen lake while the camera tracks left, "
        "snow drifting through the amber dawn light, " * 40
    ),
}
MAX_LENGTH = 64


def unpack_single_file(path: Path) -> tuple[dict, bytes, dict[str, bytes]]:
    """GemmaAssets.from_single_file, minus torch: header parse + targeted reads."""
    with path.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))
        base = 8 + header_len
        meta = header["__metadata__"]
        config = json.loads(meta[GEMMA_CONFIG_METADATA_KEY])

        def read(key: str) -> bytes:
            info = header[key]
            start, end = info["data_offsets"]
            handle.seek(base + start)
            payload = handle.read(end - start)
            if len(payload) != end - start:
                raise ValueError(f"short read of {key!r}")
            return payload

        tokenizer_json = read(TOKENIZER_JSON_TENSOR_KEY)
        sidecars = {
            key[len(HF_ASSET_TENSOR_PREFIX):]: read(key)
            for key in header
            if key.startswith(HF_ASSET_TENSOR_PREFIX)
        }
    return config, tokenizer_json, sidecars


def load_root(root: Path) -> tuple[dict, bytes, dict[str, bytes]]:
    """GemmaAssets.from_root."""
    config = json.loads((root / "config.json").read_bytes())
    tokenizer_json = (root / "tokenizer.json").read_bytes()
    sidecars = {
        path.name: path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix in (".json", ".jinja")
        and path.name not in ("config.json", "tokenizer.json")
    }
    return config, tokenizer_json, sidecars


def chat_template_from(sidecars: dict[str, bytes]) -> str | None:
    if (tpl := sidecars.get("chat_template.jinja")) is not None:
        return tpl.decode()
    raw = sidecars.get("chat_template.json")
    if raw is None:
        return None
    loaded = json.loads(raw)
    if isinstance(loaded, str):
        return loaded
    if isinstance(loaded, dict) and isinstance(loaded.get("chat_template"), str):
        return loaded["chat_template"]
    return None


def build_hf_tokenizer(tokenizer_json: bytes, sidecars: dict[str, bytes]) -> PreTrainedTokenizerFast:
    cfg = json.loads(sidecars["tokenizer_config.json"])
    kwargs = {k: v for k, v in cfg.items() if k not in _TOKENIZER_CONFIG_SKIP}
    if (tpl := chat_template_from(sidecars)) is not None:
        kwargs.setdefault("chat_template", tpl)
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_buffer(tokenizer_json),
        model_max_length=TOKENIZER_MAX_LENGTH,
        **kwargs,
    )


def ltx_gemma_tokenize(hf: PreTrainedTokenizerFast, text: str, max_length: int):
    """LTXGemmaTokenizer.tokenize_with_weights with padding_side=left."""
    hf.model_max_length = max_length
    hf.padding_side = "left"
    if hf.pad_token is None:
        hf.pad_token = hf.eos_token
    text = text.strip()
    bos_id = hf.bos_token_id
    if bos_id is None:
        raise ValueError("Tokenizer is missing bos_token_id; encode path requires a leading BOS.")
    encoded = hf(text, padding=False, truncation=True, max_length=max_length)
    input_ids = list(encoded["input_ids"])
    if not input_ids or input_ids[0] != bos_id:
        input_ids = [bos_id, *input_ids][:max_length]
    padded = hf.pad(
        {"input_ids": [input_ids]},
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
    )
    return list(padded["input_ids"][0]), list(padded["attention_mask"][0])


def emit(label: str, source: str, tokenizer_json: bytes, sidecars: dict[str, bytes]) -> dict:
    hf = build_hf_tokenizer(tokenizer_json, sidecars)
    cases = {}
    for name, text in PROMPTS.items():
        ids, mask = ltx_gemma_tokenize(hf, text, MAX_LENGTH)
        raw = list(hf(text.strip(), padding=False, truncation=False)["input_ids"])
        cases[name] = {
            "prompt": text,
            "ids": ids,
            "mask": mask,
            # The un-policied encode, so the Rust test can also show WHICH bug the policy fixes:
            # gemma4 emits no leading BOS here, gemma3 emits exactly one.
            "raw_first_id": raw[0] if raw else None,
            "raw_len": len(raw),
        }
    import tokenizers as _tokenizers
    import transformers as _transformers

    return {
        "_provenance": {
            "story": "sc-18762",
            "reference": "github.com/Lightricks/LTX-2 @ d151147788a9284cca791edc6ce898007e727fe6",
            "reference_path": "packages/ltx-core/src/ltx_core/text_encoders/gemma/"
            "{gemma_assets.py::build_gemma_hf_tokenizer, tokenizer.py::LTXGemmaTokenizer}",
            "source": source,
            "transformers": _transformers.__version__,
            "tokenizers": _tokenizers.__version__,
            "generator": "gen-core/tests/fixtures/gen_ltx_gemma_token_parity.py",
        },
        "label": label,
        "max_length": MAX_LENGTH,
        "bos_token_id": hf.bos_token_id,
        "pad_token_id": hf.pad_token_id,
        "eos_token_id": hf.eos_token_id,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemma4-te", type=Path, required=True)
    parser.add_argument("--gemma3-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    _, tokenizer_json, sidecars = unpack_single_file(args.gemma4_te)
    gemma4 = emit(
        "gemma4-12b-with-proj-ltx-2.5",
        "Lightricks/LTX-2.5 text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors "
        "(packed tokenizer_json + hf_asset__tokenizer_config.json)",
        tokenizer_json,
        sidecars,
    )
    (args.out_dir / "ltx25_gemma4_token_parity.json").write_text(
        json.dumps(gemma4, indent=1, ensure_ascii=False) + "\n"
    )

    _, tokenizer_json, sidecars = load_root(args.gemma3_root)
    gemma3 = emit(
        "gemma-3-12b-it (LTX-2.3)",
        "SceneWorks/ltx-2.3-mlx gemma/ directory root (tokenizer.json + tokenizer_config.json)",
        tokenizer_json,
        sidecars,
    )
    (args.out_dir / "ltx23_gemma3_token_parity.json").write_text(
        json.dumps(gemma3, indent=1, ensure_ascii=False) + "\n"
    )
    print("wrote ltx25_gemma4_token_parity.json + ltx23_gemma3_token_parity.json")


if __name__ == "__main__":
    main()
