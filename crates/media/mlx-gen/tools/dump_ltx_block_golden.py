"""LTX-2.3 DiT single-block golden — reference `BasicAVTransformerBlock` (video-only) I/O (sc-2679 S3a).

Builds the reference video transformer block with the 2.3 config (dim 4096, 32 heads × 128, gated),
loads **block 0**'s weights from the real `ltx_2_3_base_q8` `transformer.safetensors` —
**dequantizing** the Q8 attn/ff Linears (`mx.dequantize`, group 64 / 8-bit) to dense f32 — casts the
whole block to f32, and runs one forward over deterministic synthetic inputs. The Rust `VideoBlock`
(mlx-gen-ltx/tests/block_parity.rs) dequantizes the SAME Q8 weights and must reproduce the output.

f32 isolates the block **math** (gated attention, q/k-norm, SPLIT RoPE, 9-row adaLN, prompt-adaLN,
text cross-attention, gelu-tanh FF) from the Q8 quantized-matmul path (S3b). The RoPE cos/sin and the
adaLN-single timestep projections are fed as fixture inputs (their own gates are S0 / S3b), so this
isolates the block.

Run (mflux venv + mlx_video source):
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg \
      ~/Repos/mflux/.venv/bin/python tools/dump_ltx_block_golden.py
Output (committed): mlx-gen-ltx/tests/fixtures/ltx_block_golden.safetensors
"""

import glob
import os
import sys
from pathlib import Path

from _paths import fixture


def _find_mlx_video_src() -> str:
    if env := os.environ.get("MLX_VIDEO_SRC"):
        return str(Path(env).expanduser())
    for cand in sorted(glob.glob(str(Path.home() / ".cache/uv/archive-v0/*/mlx_video"))):
        return str(Path(cand).parent)
    raise SystemExit("Set MLX_VIDEO_SRC to the dir containing `mlx_video/`.")


sys.path.insert(0, _find_mlx_video_src())

# text_encoder.py (imported transitively) pulls mlx_vlm; stub it (we never touch Gemma here).
import types  # noqa: E402

for _name in ("mlx_vlm", "mlx_vlm.models", "mlx_vlm.models.gemma3"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
_lang = types.ModuleType("mlx_vlm.models.gemma3.language")
_lang.Gemma3Model = object
sys.modules["mlx_vlm.models.gemma3.language"] = _lang
_cfg = types.ModuleType("mlx_vlm.models.gemma3.config")
_cfg.TextConfig = object
sys.modules["mlx_vlm.models.gemma3.config"] = _cfg

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_map  # noqa: E402

from mlx_video.models.ltx.config import LTXRopeType, TransformerConfig  # noqa: E402
from mlx_video.models.ltx.transformer import (  # noqa: E402
    BasicAVTransformerBlock,
    TransformerArgs,
)

MODEL = Path.home() / "Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8"

DIM, HEADS, HEAD_DIM = 4096, 32, 128
S, CTX = 8, 16  # tokens, context length
GROUP, BITS = 64, 8

video_cfg = TransformerConfig(
    dim=DIM, heads=HEADS, d_head=HEAD_DIM, context_dim=DIM, apply_gated_attention=True
)
block = BasicAVTransformerBlock(
    idx=0, video=video_cfg, audio=None, rope_type=LTXRopeType.SPLIT, norm_eps=1e-6
)

# Load block 0, dequantizing the Q8 attn/ff Linears to dense f32.
raw = mx.load(str(MODEL / "transformer.safetensors"))
prefix = "transformer_blocks.0."
block_w = {}
for k in raw:
    if not k.startswith(prefix):
        continue
    sub = k[len(prefix):]
    if sub.endswith(".scales") or sub.endswith(".biases"):
        continue
    if sub.endswith(".weight") and (k[:-len(".weight")] + ".scales") in raw:
        base = k[:-len(".weight")]
        dense = mx.dequantize(
            raw[base + ".weight"], raw[base + ".scales"], raw[base + ".biases"],
            group_size=GROUP, bits=BITS,
        )
        block_w[sub] = dense.astype(mx.float32)
    else:
        block_w[sub] = raw[k].astype(mx.float32)

block.load_weights(list(block_w.items()), strict=False)
block.update(tree_map(lambda p: p.astype(mx.float32), block.parameters()))
mx.eval(block.parameters())

# Deterministic synthetic inputs (f32).
mx.random.seed(3)
x = mx.random.normal((1, S, DIM)).astype(mx.float32)
context = mx.random.normal((1, CTX, DIM)).astype(mx.float32)
timesteps = mx.random.normal((1, 1, 9 * DIM)).astype(mx.float32)
prompt_timestep = mx.random.normal((1, 1, 2 * DIM)).astype(mx.float32)
# Valid RoPE tables (cos²+sin²=1) of the per-head shape (1, heads, S, head_dim//2).
theta = mx.random.normal((1, HEADS, S, HEAD_DIM // 2)).astype(mx.float32)
cos, sin = mx.cos(theta), mx.sin(theta)

args = TransformerArgs(
    x=x,
    context=context,
    context_mask=None,
    timesteps=timesteps,
    embedded_timestep=mx.zeros((1, 1, DIM), dtype=mx.float32),
    positional_embeddings=(cos, sin),
    cross_positional_embeddings=None,
    cross_scale_shift_timestep=None,
    cross_gate_timestep=None,
    prompt_timestep=prompt_timestep,
    enabled=True,
)

video_out, _ = block(video=args, audio=None)
out = video_out.x
mx.eval(out)
print(f"block: x{x.shape} -> {out.shape}")

tensors = {
    "x": x,
    "context": context,
    "timesteps": timesteps,
    "prompt_timestep": prompt_timestep,
    "cos": cos,
    "sin": sin,
    "out": out.astype(mx.float32),
}
out_path = fixture("mlx-gen-ltx/tests/fixtures/ltx_block_golden.safetensors")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out_path, tensors, metadata={"S": str(S), "ctx": str(CTX), "dim": str(DIM)})
print(f"wrote {out_path}")
