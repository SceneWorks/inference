"""sc-5985 — golden dump for the Ideogram 4 text-encoder parity test.

Runs the **transformers** Qwen3-VL text stack (an independent forward graph) on the converted
bf16 `text_encoder` weights, text-only, and saves the concatenated hidden states at the 13
Ideogram layers `(0,3,…,33,35)` → `[1, seq, 53248]`, plus the `input_ids` / `attention_mask`
the Rust parity test feeds verbatim. Loading the *converted bf16* (not the source fp8) isolates
the forward-graph check from the already-verified fp8 dequant (sc-5984).

Run:
  ~/mlx-flux-venv/bin/python tools/dump_ideogram4_te_golden.py \
      --converted ~/.cache/ideogram4-mlx-convert \
      --out tools/golden/ideogram4_te.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import torch

EXTRACTED_LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]
PROMPT = "A medium-shot photograph of a red fox sitting in a snowy forest at golden hour."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--converted", type=Path, default=Path.home() / ".cache/ideogram4-mlx-convert")
    ap.add_argument("--snapshot", type=Path, default=None, help="fp8 snapshot (for the tokenizer)")
    ap.add_argument("--out", type=Path, default=Path("tools/golden/ideogram4_te.safetensors"))
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    te_dir = args.converted / "text_encoder"
    if not (te_dir / "model.safetensors").exists():
        sys.exit(f"converted TE not found: {te_dir}")

    from transformers import AutoConfig, AutoTokenizer
    try:
        from transformers import Qwen3VLModel
    except Exception as e:  # pragma: no cover
        sys.exit(f"transformers has no Qwen3VLModel ({e}); upgrade transformers")

    tok_dir = (args.snapshot or args.converted) / "tokenizer"
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    enc = tok(args.prompt, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
    print(f"prompt: {args.prompt!r}  seq_len={input_ids.shape[1]}")

    cfg = AutoConfig.from_pretrained(str(te_dir))
    print(f"loading Qwen3VLModel ({cfg.model_type}) bf16 on CPU …")
    model = Qwen3VLModel.from_pretrained(
        str(te_dir), torch_dtype=torch.bfloat16, config=cfg
    ).eval()

    # Replicate the pipeline's `_get_qwen3_vl_embeddings` exactly: capture the RAW output of each
    # decoder layer (`captured[layer_idx] = decoder_layer(...)`), NOT `output_hidden_states` (whose
    # last entry is final-norm'd). Single unpadded prompt → positions 0..s, full attention.
    from transformers.masking_utils import create_causal_mask

    lm = model.language_model
    with torch.no_grad():
        inputs_embeds = lm.embed_tokens(input_ids)
        pos_2d = torch.arange(input_ids.shape[1])[None, :]  # [1, s]
        position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)  # [4, 1, s]
        text_position_ids = position_ids_4d[0]
        mrope_position_ids = position_ids_4d[1:]
        causal_mask = create_causal_mask(
            config=lm.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        position_embeddings = lm.rotary_emb(inputs_embeds, mrope_position_ids)
        tap = set(EXTRACTED_LAYERS)
        captured: dict[int, torch.Tensor] = {}
        hidden = inputs_embeds
        for li, layer in enumerate(lm.layers):
            out = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
            hidden = out[0] if isinstance(out, tuple) else out
            if li in tap:
                captured[li] = hidden
    selected = [captured[i].to(torch.float32) for i in EXTRACTED_LAYERS]
    # Interleave (matches `_encode_text`: stack → permute → reshape, feature = h*n + layer).
    stacked = torch.stack(selected, dim=0)  # [n, B, L, H]
    stacked = torch.permute(stacked, (1, 2, 3, 0))  # [B, L, H, n]
    golden = stacked.reshape(stacked.shape[0], stacked.shape[1], -1)  # [B, L, H*n]
    print(f"golden: {tuple(golden.shape)} (expect [..,..,53248])")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.out),
        {
            "input_ids": mx.array(input_ids.to(torch.int32).numpy()),
            "attention_mask": mx.array(attention_mask.to(torch.int32).numpy()),
            "golden": mx.array(golden.numpy()),
        },
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
