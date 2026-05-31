"""Dump a Qwen-Image text-encoder parity golden for the Rust port (sc-2348, slice 2).

The text encoder is ~7B params (too big to dump as weights), and its on-disk HF layout
(`model.embed_tokens.weight`, `model.layers.{i}.…`, `model.norm.weight`) maps directly onto the Rust
module tree under the `"model"` prefix — so the Rust test loads the real snapshot weights itself.
This script dumps only fixed inputs + the fork's f32 outputs (encoder hidden states + the
drop-34 prompt embeds).

Run from the mflux fork venv:
    cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/dump_qwen_text_encoder_golden.py

Output (gitignored): tools/golden/qwen_text_encoder_golden.safetensors
"""

import os

import mlx.core as mx
from mlx.utils import tree_map

from mflux.models.common.config import ModelConfig
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

cfg = ModelConfig.qwen_image()
weights = WeightLoader.load(weight_definition=QwenWeightDefinition, model_path=cfg.model_name)
te = QwenTextEncoder()
WeightApplier.apply_and_quantize(
    weights=weights,
    quantize_arg=None,
    weight_definition=QwenWeightDefinition,
    models={"text_encoder": te},
)
# Cast to f32 so the golden matches the Rust f32 forward (the bf16 weights promote to f32 there).
# The 7B encoder in f32 is ~28 GB — run on a machine with enough RAM (the fork loads it anyway).
te.update(tree_map(lambda a: a.astype(mx.float32), te.parameters()))

mx.random.seed(0)
S = 40
input_ids = mx.random.randint(0, 152064, (1, S)).astype(mx.int32)
attention_mask = mx.ones((1, S), dtype=mx.int32)

hidden = te.encoder(input_ids, attention_mask)  # final-normed [1, S, 3584]
prompt_embeds, _ = QwenTextEncoder._process_text_embeddings_mlx(
    hidden_states=hidden, attention_mask=attention_mask, drop_idx=34, dtype=mx.float32
)
mx.eval(hidden, prompt_embeds)

out = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "hidden_states": hidden.astype(mx.float32),
    "prompt_embeds": prompt_embeds.astype(mx.float32),
}
golden_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(golden_dir, exist_ok=True)
path_out = os.path.join(golden_dir, "qwen_text_encoder_golden.safetensors")
mx.save_safetensors(path_out, out)
print(f"hidden={hidden.shape} prompt_embeds={prompt_embeds.shape}")
print(f"wrote {path_out}")
