#!/usr/bin/env python
"""Dump the end-to-end Lens gpt-oss encoder golden (mlx-gen sc-3171).

Runs the **authoritative** `LensGptOssEncoder` (the SceneWorks `_vendor/lens` subclass of
`transformers.GptOssForCausalLM`) encoder-only over a battery of prompts and records, per prompt:

  - `ids_{i}`         — the harmony-rendered `input_ids` `[1, L]` (so the Rust parity test feeds
                        byte-identical tokens, decoupling the encoder check from the tokenizer);
  - `cap_{i}_{j}`     — the captured hidden state at selected layer `j` (`[1, L, 2880]`), `j` in
                        `0..len(SELECTED)` (selection order [5, 11, 17, 23]).

The model is loaded with `Mxfp4Config(dequantize=True)` (experts → dense **bf16**, runnable on CPU)
and forced to the **eager** attention + experts paths — the exact dense math the Rust port
reproduces. The forward is the vendor `LensGptOssEncoder.forward` feature path verbatim
(`set_selected_layers([5,11,17,23])` → capture each selected layer's *output* → early-exit at 23).

Memory: the dequantized encoder is ~40 GB bf16; this process peaks ~45 GB then exits before the Rust
test loads its own copy (sequential, not concurrent). Needs ~64 GB+ free.

Run (from repo root):
  ~/Repos/mflux/.venv/bin/python tools/dump_lens_encoder_golden.py
Writes `tools/golden/lens_encoder_golden.safetensors` (gitignored real-weights golden).
"""

from __future__ import annotations

import datetime
import glob
import importlib.util
import os

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, Mxfp4Config
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

HOME = os.path.expanduser("~")
SNAP_GLOB = f"{HOME}/.cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots/*/text_encoder"
TOK_GLOB = f"{HOME}/.cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots/*/tokenizer"
VENDOR_TE = os.path.expanduser(
    "~/Repos/SceneWorks/apps/worker/scene_worker/_vendor/lens/text_encoder.py"
)
OUT = os.path.join(os.path.dirname(__file__), "golden", "lens_encoder_golden.safetensors")

SELECTED_LAYERS = [5, 11, 17, 23]

_CHAT_SYSTEM = (
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background."
)
_CHAT_ASSISTANT_THINKING = "Need to generate one image according to the description."

# A battery spanning short / degenerate / non-ASCII / long. The long prompt pushes the total token
# count past the 128 sliding-window so the even (sliding-attention) layers' window mask is exercised.
PROMPTS = [
    "a red cube on a wooden table",
    "X",
    "猫が窓辺で眠っている",
    (
        "A photorealistic wide-angle photograph of a bustling Tokyo street at night in the rain, "
        "neon signs in red blue and green reflecting off the wet asphalt, dozens of pedestrians "
        "holding transparent umbrellas, a yellow taxi waiting at a crosswalk, steam rising from a "
        "ramen stall on the left, tall glass skyscrapers fading into a foggy sky, cinematic shallow "
        "depth of field, golden bokeh, ultra detailed, 8k, captured on a 35mm lens at f1.4."
    ),
]


def load_vendor_encoder_cls():
    spec = importlib.util.spec_from_file_location("lens_text_encoder", VENDOR_TE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LensGptOssEncoder


@torch.no_grad()
def run_encoder(model, input_ids, attention_mask, selected):
    """The `LensGptOssEncoder.forward` feature path, verbatim, with this transformers version's mask
    kwargs (`input_embeds` / `cache_position` — the vendor file targets transformers 5.8 where the
    kwarg is `inputs_embeds`). Embed → per-layer sliding/full mask → run layers → capture each
    selected layer's *output* → early-exit after the max selected layer."""
    m = model.model
    inputs_embeds = m.embed_tokens(input_ids)
    seq_len = inputs_embeds.shape[1]
    cache_position = torch.arange(seq_len, device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0).expand_as(input_ids)

    mask_kwargs = {
        "config": m.config,
        "input_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }

    hidden_states = inputs_embeds
    position_embeddings = m.rotary_emb(hidden_states, position_ids)
    max_layer = max(selected)
    index_lookup = {idx: pos for pos, idx in enumerate(selected)}
    captured = [None] * len(selected)
    for i, decoder_layer in enumerate(m.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=mask_mapping[m.config.layer_types[i]],
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        if i in index_lookup:
            captured[index_lookup[i]] = hidden_states
        if i == max_layer:
            break
    return captured


def main() -> None:
    te_matches = sorted(glob.glob(SNAP_GLOB))
    tok_matches = sorted(glob.glob(TOK_GLOB))
    if not te_matches:
        raise SystemExit(f"no Lens-Turbo text_encoder snapshot at {SNAP_GLOB}")
    if not tok_matches:
        raise SystemExit(f"no Lens-Turbo tokenizer snapshot at {TOK_GLOB}")
    te_dir, tok_dir = te_matches[-1], tok_matches[-1]

    tok = AutoTokenizer.from_pretrained(tok_dir)

    def render_ids(prompt: str) -> torch.Tensor:
        conversation = [
            {"role": "system", "content": _CHAT_SYSTEM, "thinking": None},
            {"role": "user", "content": prompt, "thinking": None},
            {"role": "assistant", "thinking": _CHAT_ASSISTANT_THINKING, "content": ""},
        ]
        text = tok.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        ).split("<|return|>")[0]
        ids = tok(text, add_special_tokens=True)["input_ids"]
        return torch.tensor([ids], dtype=torch.long)

    config = AutoConfig.from_pretrained(te_dir)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"

    LensGptOssEncoder = load_vendor_encoder_cls()
    print("loading encoder (dequantize MXFP4 → bf16, CPU)…", flush=True)
    model = LensGptOssEncoder.from_pretrained(
        te_dir,
        config=config,
        quantization_config=Mxfp4Config(dequantize=True),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    ).eval()

    tensors: dict[str, torch.Tensor] = {}
    meta = {
        "n_prompts": str(len(PROMPTS)),
        "n_selected": str(len(SELECTED_LAYERS)),
        "selected_layers": ",".join(str(x) for x in SELECTED_LAYERS),
        "hidden_size": str(config.hidden_size),
        "current_date": datetime.date.today().isoformat(),
    }

    for i, prompt in enumerate(PROMPTS):
        input_ids = render_ids(prompt)
        attention_mask = torch.ones_like(input_ids)
        captured = run_encoder(model, input_ids, attention_mask, SELECTED_LAYERS)
        tensors[f"ids_{i}"] = input_ids.to(torch.int32).cpu()
        for j, h in enumerate(captured):
            # store f32 for a precise, dtype-stable comparison target
            tensors[f"cap_{i}_{j}"] = h.to(torch.float32).cpu()
        meta[f"prompt_{i}"] = prompt
        print(
            f"prompt {i}: L={input_ids.shape[1]} captured "
            f"{[tuple(h.shape) for h in captured]}",
            flush=True,
        )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(tensors, OUT, metadata=meta)
    print(f"wrote {OUT}  (current_date={meta['current_date']})")


if __name__ == "__main__":
    main()
