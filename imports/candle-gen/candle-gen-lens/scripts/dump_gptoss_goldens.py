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

# Fixed, deterministic token sequence. 160 tokens > sliding_window (128) so the sliding-attention
# layers are genuinely exercised (a shorter prompt makes the sliding and full masks identical). The
# ids are dumped into the goldens file so the Rust side reads the exact same input.
VOCAB = 201088
SEQ = 160
IDS = [(i * 7919 + 17) % VOCAB for i in range(SEQ)]

# HF hidden_states indices to dump. Index 0 = embeddings; index i (1..=num_layers) = output of the
# loop's i-th append. Includes Lens's capture points [5, 11, 17, 23] plus neighbours + the final.
DUMP_INDICES = [0, 1, 5, 11, 12, 17, 23, 24]


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

    dst = out_dir / "gptoss_goldens.safetensors"
    save_file(tensors, str(dst))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
