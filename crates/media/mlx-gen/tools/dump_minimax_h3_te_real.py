"""MiniMax-H3 text-encoder **real-weight** conditioning reference (sc-18741).

Runs the OFFICIAL conditioner path on the real 62 GB ``text_encoder/`` and writes the `context` the
DiT is conditioned on, so the Rust real-weight lane can assert numeric parity against it.

The conditioner call is `diffusers.modular_pipelines.minimax_h3.encoders`' own, reproduced line for
line:

    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]        # no chat template
    outputs   = text_encoder.model(input_ids=..., attention_mask=ones,
                                   mm_token_type_ids=..., use_cache=False,
                                   output_hidden_states=True)
    context   = outputs.hidden_states[50]                                        # untrimmed

Nothing is sliced. sc-17143 rendered ``chat_template.json``, prepended
``<|im_start|>user\\n``, appended ``<|im_end|>\\n<|im_start|>assistant\\n``, and dropped 3 leading
ids — so the DiT saw the prompt plus a 5-token generation cue (and, for a whitespace-leading prompt,
minus a real prompt token, because the tokenizer merges the template's trailing newline into it).

**Memory.** The conditioner is ~62 GB on MPS and the spike established it does NOT release inside a
process (`.to("cpu")`, `del`, `gc.collect()`, `torch.mps.empty_cache()` all fail), so this runs as
its own process and the Rust side runs as another. Nothing else large may be resident.

    MINIMAX_H3_SNAPSHOT=<snapshot-root> python3 tools/dump_minimax_h3_te_real.py \
        --out ~/minimax-h3-real-te-reference.safetensors
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from safetensors.numpy import save_file

from _paths import hf_hub_cache

# `components.text_encoder_layer` on `MiniMaxH3ModularPipeline`.
TEXT_ENCODER_LAYER = 50
PROMPT = "a red fox leaps over a mossy log at dawn"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output .safetensors path (keep it OUTSIDE the repo)")
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    import transformers
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    snapshot = Path(
        os.environ.get("MINIMAX_H3_SNAPSHOT")
        or next(iter(sorted((hf_hub_cache() / "models--MiniMaxAI--MiniMax-H3" / "snapshots").glob("*"))))
    )

    tokenizer = AutoTokenizer.from_pretrained(snapshot / "tokenizer")
    # THE reference call — no chat template, no special tokens.
    token_ids = tokenizer(args.prompt, add_special_tokens=False)["input_ids"]
    print(f"presentation: {len(token_ids)} ids {token_ids}")
    assert all(i < 151643 for i in token_ids), "the presentation must be plain text only"

    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        snapshot / "text_encoder", dtype=torch.bfloat16
    )
    text_encoder.eval()
    num_layers = text_encoder.config.text_config.num_hidden_layers
    assert num_layers > TEXT_ENCODER_LAYER, f"{num_layers} layers cannot serve hidden_states[{TEXT_ENCODER_LAYER}]"

    processor = Qwen3VLProcessor.from_pretrained(snapshot / "text_encoder")
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long)

    with torch.no_grad():
        outputs = text_encoder.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            output_hidden_states=True,
        )
    context = outputs.hidden_states[TEXT_ENCODER_LAYER].to(torch.float32)
    assert context.shape[1] == len(token_ids), "the reference context keeps one row per token"
    print(
        f"context {tuple(context.shape)}  std {context.std().item():.6f}  "
        f"max|x| {context.abs().max().item():.6f}"
    )

    tensors = {
        "in.input_ids": input_ids.to(torch.int32).numpy().copy(),
        "in.attention_mask": torch.ones_like(input_ids).to(torch.int32).numpy().copy(),
        "out.context": context.numpy().copy(),
    }
    meta = {
        "provenance": "official-conditioner",
        "reference": "diffusers.MiniMaxH3TextEncoderStep",
        "reference_version": f"transformers {transformers.__version__}",
        "snapshot": snapshot.name,
        "prompt": args.prompt,
        "text_encoder_layer": str(TEXT_ENCODER_LAYER),
        "applies_chat_template": "false",
        "dtype": "bfloat16",
        "story": "sc-18741",
    }
    out = Path(args.out).expanduser()
    save_file(tensors, str(out), metadata=meta)
    print(f"wrote {out} @ {snapshot.name}")


if __name__ == "__main__":
    main()
