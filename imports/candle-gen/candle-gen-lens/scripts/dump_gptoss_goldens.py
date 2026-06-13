#!/usr/bin/env python
"""Dump gpt-oss-20b encoder golden tensors for the candle-gen-lens sc-5108 parity test.

Runs the HF `microsoft/Lens` text_encoder (a `GptOssForCausalLM`) forward over a fixed token
sequence in the transformers-5.8 `lens-venv`, and saves the per-layer hidden states + final
last_hidden_state as a single .safetensors the Rust parity test compares against.

MXFP4 is forced to DEQUANTIZE to bf16 (`Mxfp4Config(dequantize=True)`) so the reference matches the
Rust port's bf16 bring-up (sc-5108) exactly — i.e. both dequantize the experts to bf16 and run bf16,
removing any native-MXFP4-kernel vs. CPU-dequant discrepancy from the comparison.

Usage (from the worktree root):
  & "C:\\Users\\Michael\\AppData\\Roaming\\SceneWorks\\python\\lens-venv\\Scripts\\python.exe" \\
      candle-gen-lens\\scripts\\dump_gptoss_goldens.py [out_dir]

Default out_dir: .scratch/gptoss-goldens/  (not committed — goldens are large + regenerable).
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, Mxfp4Config
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

# Fixed, deterministic token sequence. 160 tokens > sliding_window (128) so the sliding-attention
# layers are genuinely exercised (a shorter prompt makes the sliding and full masks identical). The
# ids are dumped into the goldens file so the Rust side reads the exact same input.
VOCAB = 201088
SEQ = 160
IDS = [(i * 7919 + 17) % VOCAB for i in range(SEQ)]

# HF hidden_states indices to dump. Index 0 = embeddings; index i (1..=num_layers) = output of the
# loop's i-th append. Includes Lens's capture points [5, 11, 17, 23] plus neighbours + the final.
DUMP_INDICES = [0, 1, 5, 11, 12, 17, 23, 24]


# Lens capture indices (the LensGptOssEncoder feature path): the *output* of these decoder layers.
SELECTED_LAYERS = [5, 11, 17, 23]


@torch.no_grad()
def run_capture(model, input_ids, selected):
    """The vendor `LensGptOssEncoder.forward` feature path: embed → per-layer sliding/full mask →
    run layers → capture each selected layer's *output* (raw residual stream, no final norm) →
    early-exit after the max selected layer. Reproduced on the stock GptOssModel internals."""
    m = model.model
    inputs_embeds = m.embed_tokens(input_ids)
    seq_len = inputs_embeds.shape[1]
    cache_position = torch.arange(seq_len, device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    mask_kwargs = {
        "config": m.config,
        "input_embeds": inputs_embeds,
        "attention_mask": torch.ones_like(input_ids),
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }
    hidden = inputs_embeds
    position_embeddings = m.rotary_emb(hidden, position_ids)
    captured = {}
    mx = max(selected)
    for i, layer in enumerate(m.layers):
        hidden = layer(
            hidden,
            attention_mask=mask_mapping[m.config.layer_types[i]],
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        if i in selected:
            captured[i] = hidden
        if i == mx:
            break
    return captured


def find_text_encoder() -> Path:
    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--microsoft--Lens" / "snapshots"
    cands = sorted(hub.glob("*/text_encoder"))
    if not cands:
        sys.exit(f"no microsoft/Lens text_encoder snapshot under {hub}")
    return cands[-1]


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".scratch/gptoss-goldens")
    out_dir.mkdir(parents=True, exist_ok=True)
    te = find_text_encoder()
    print(f"text_encoder: {te}")

    # Force MXFP4 -> bf16 dequant so the reference mirrors the Rust port's bf16 expert path.
    model = AutoModelForCausalLM.from_pretrained(
        te,
        torch_dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True),
        device_map="cuda",
    ).eval()

    # Confirm the on-disk MXFP4 expert tensor shapes (the Rust dequant assumes blocks [E, out, nb, 16],
    # scales [E, out, nb]). Print the raw state-dict shapes for layer 0 so the Rust layout can be
    # validated before the CUDA build.
    sd = model.state_dict()
    for name in [
        "model.layers.0.mlp.experts.gate_up_proj_blocks",
        "model.layers.0.mlp.experts.gate_up_proj_scales",
        "model.layers.0.mlp.experts.down_proj_blocks",
        "model.layers.0.mlp.experts.down_proj_scales",
        "model.layers.0.self_attn.sinks",
    ]:
        if name in sd:
            print(f"  shape {name}: {tuple(sd[name].shape)} dtype {sd[name].dtype}")

    input_ids = torch.tensor([IDS], dtype=torch.long, device="cuda")
    with torch.no_grad():
        out = model.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    hs = out.hidden_states
    print(f"num hidden_states: {len(hs)} (expect num_layers + 1 = 25)")
    tensors = {}
    for i in DUMP_INDICES:
        t = hs[i][0].to(torch.float32).contiguous().cpu()
        tensors[f"hidden_{i:02d}"] = t
        print(f"  hidden_{i:02d}: {tuple(t.shape)}")
    tensors["last_hidden_state"] = out.last_hidden_state[0].to(torch.float32).contiguous().cpu()
    tensors["input_ids"] = input_ids[0].to(torch.int64).cpu()

    # sc-5110: the raw layer-OUTPUT captures at [5,11,17,23] (the LensGptOssEncoder feature path).
    caps = run_capture(model, input_ids, SELECTED_LAYERS)
    for s in SELECTED_LAYERS:
        t = caps[s][0].to(torch.float32).contiguous().cpu()
        tensors[f"cap_{s:02d}"] = t
        print(f"  cap_{s:02d}: {tuple(t.shape)}")

    dst = out_dir / "gptoss_goldens.safetensors"
    save_file(tensors, str(dst))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
