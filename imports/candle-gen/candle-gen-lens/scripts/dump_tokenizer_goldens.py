#!/usr/bin/env python
"""Dump Lens tokenizer golden ids for the candle-gen-lens sc-5109 parity test.

Renders the Lens harmony chat prompt via the model's own `chat_template.jinja` (the authoritative
path — `apply_chat_template` over the [system, user, assistant-thinking] conversation, split at
`<|return|>`), tokenizes with `add_special_tokens=True`, and saves the ids per prompt so the Rust
`LensTokenizer` (which hand-renders the same harmony string) can be checked for byte-identical ids.

Run (from the worktree root):
  & "...\\lens-venv\\Scripts\\python.exe" candle-gen-lens\\scripts\\dump_tokenizer_goldens.py [out_dir]
Default out_dir: .scratch/lens-tok-goldens/
"""
import datetime
import glob
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

_CHAT_SYSTEM = (
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background."
)
_CHAT_ASSISTANT_THINKING = "Need to generate one image according to the description."

# Short / single-char / non-ASCII / long (the long one pushes well past 97 so txt_offset is testable).
PROMPTS = [
    "a red cube on a wooden table",
    "X",
    "猫が窓辺で眠っている",
    (
        "A photorealistic wide-angle photograph of a bustling Tokyo street at night in the rain, "
        "neon signs reflecting off the wet asphalt, dozens of pedestrians with transparent umbrellas, "
        "a yellow taxi at a crosswalk, steam rising from a ramen stall, cinematic shallow depth of field."
    ),
]


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".scratch/lens-tok-goldens")
    out_dir.mkdir(parents=True, exist_ok=True)
    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--microsoft--Lens" / "snapshots"
    tok_dir = sorted(glob.glob(str(hub / "*" / "tokenizer")))[-1]
    print(f"tokenizer: {tok_dir}")
    tok = AutoTokenizer.from_pretrained(tok_dir)
    date = datetime.date.today().isoformat()

    def render_ids(prompt: str):
        conversation = [
            {"role": "system", "content": _CHAT_SYSTEM, "thinking": None},
            {"role": "user", "content": prompt, "thinking": None},
            {"role": "assistant", "thinking": _CHAT_ASSISTANT_THINKING, "content": ""},
        ]
        text = tok.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        ).split("<|return|>")[0]
        return tok(text, add_special_tokens=True)["input_ids"]

    tensors = {"date_utf8": torch.tensor(list(date.encode()), dtype=torch.uint8)}
    meta = {"n_prompts": str(len(PROMPTS)), "date": date}
    for i, prompt in enumerate(PROMPTS):
        ids = render_ids(prompt)
        tensors[f"ids_{i}"] = torch.tensor(ids, dtype=torch.int64)
        meta[f"prompt_{i}"] = prompt
        print(f"prompt {i}: L={len(ids)}  first8={ids[:8]}")
    save_file(tensors, str(out_dir / "tokenizer_goldens.safetensors"), metadata=meta)
    print(f"wrote {out_dir / 'tokenizer_goldens.safetensors'}  (date={date})")


if __name__ == "__main__":
    main()
