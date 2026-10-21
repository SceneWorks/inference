"""Dump the Iris-3B prompt-template token ids from the pinned Qwen3-VL tokenizer (sc-25679).

Upstream (`iris3b/text/qwen3_vl.py`) tokenizes the chat-template PREFIX, the SUFFIX and each CAPTION
separately (`add_special_tokens=False`) and concatenates them, truncating only the caption to
`max_length - len(suffix)`. This records those three id lists for a small prompt battery so the
native tokenizer path is pinned to the frozen tokenizer without committing the 11 MB tokenizer.json.

Inputs: the pinned `Qwen/Qwen3-VL-4B-Instruct` snapshot in the HF cache (see `_iris_common.py`).
Run: `python -I tools/dump_iris_tokenizer.py` in the isolated reference venv.
Output: `mlx-gen-iris/tests/fixtures/iris_tokenizer_ids.json`.
"""

from __future__ import annotations

import hashlib

from _iris_common import FIXTURE_DIR, QWEN3_VL_REPO, QWEN3_VL_REVISION, write_json
from _paths import hf_hub_cache

# Copied verbatim from iris3b/text/qwen3_vl.py @ the pinned commit.
PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial "
    "relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

PROMPTS = [
    "a red fox sleeping in fresh snow, golden hour",
    "",
    "A hand-painted wooden sign in a flower shop window that says “Fresh Tulips Today”",
    "  leading and trailing spaces  ",
    "emoji \U0001f98a and CJK 狐狸 mixed",
    "<|im_end|> literal special token text",
]


def main() -> None:
    from transformers import AutoTokenizer

    snapshot = hf_hub_cache() / f"models--{QWEN3_VL_REPO.replace('/', '--')}" / "snapshots" / QWEN3_VL_REVISION
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    digest = hashlib.sha256((snapshot / "tokenizer.json").read_bytes()).hexdigest()
    out = {
        "qwen3_vl_revision": QWEN3_VL_REVISION,
        "tokenizer_json_sha256": digest,
        "pad_token_id": tok.pad_token_id,
        "prefix_ids": tok.encode(PROMPT_PREFIX, add_special_tokens=False),
        "suffix_ids": tok.encode(PROMPT_SUFFIX, add_special_tokens=False),
        "captions": [
            {"prompt": p, "ids": tok([p], add_special_tokens=False)["input_ids"][0]} for p in PROMPTS
        ],
    }
    write_json(FIXTURE_DIR / "iris_tokenizer_ids.json", out)


if __name__ == "__main__":
    main()
