"""Dump a tiny Z-Image text-encoder EncoderLayer (+ RoPE cos/sin) parity fixture from the fork.

Run from the fork:  cd ~/repos/mflux && uv run python tools/dump_z_image_text_encoder_layer.py

The layer exercises attention (GQA + q_norm/k_norm + HF half-split RoPE + causal SDPA) + SwiGLU
MLP + pre-norm residuals together. cos/sin double as the RoPE golden. Tiny random config.
"""

import mlx.core as mx
from mflux.models.z_image.model.z_image_text_encoder.encoder_layer import EncoderLayer
from mflux.models.z_image.model.z_image_text_encoder.rope import RotaryEmbedding
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder

from _paths import fixture

OUT = fixture("mlx-gen-z-image/tests/fixtures/text_encoder_layer.safetensors")

mx.random.seed(0)
H, NH, NKV, HD, INTER, SEQ = 64, 4, 2, 16, 128, 6
EPS, THETA = 1e-6, 1_000_000.0

layer = EncoderLayer(
    hidden_size=H,
    num_attention_heads=NH,
    num_key_value_heads=NKV,
    intermediate_size=INTER,
    head_dim=HD,
    rms_norm_eps=EPS,
)
rope = RotaryEmbedding(dim=HD, base=THETA)

x = mx.random.normal((1, SEQ, H)).astype(mx.float32)
pos = mx.broadcast_to(mx.arange(SEQ, dtype=mx.int32)[None, :], (1, SEQ))
cos, sin = rope(x, pos)
mask = TextEncoder._create_causal_mask(SEQ, mx.float32)
out = layer(x, mask, (cos, sin))
mx.eval(out)

tensors = {
    "in": x,
    "cos": cos,
    "sin": sin,
    "mask": mask,
    "out": out,
    "input_layernorm.weight": layer.input_layernorm.weight,
    "post_attention_layernorm.weight": layer.post_attention_layernorm.weight,
    "self_attn.q_proj.weight": layer.self_attn.q_proj.weight,
    "self_attn.k_proj.weight": layer.self_attn.k_proj.weight,
    "self_attn.v_proj.weight": layer.self_attn.v_proj.weight,
    "self_attn.o_proj.weight": layer.self_attn.o_proj.weight,
    "self_attn.q_norm.weight": layer.self_attn.q_norm.weight,
    "self_attn.k_norm.weight": layer.self_attn.k_norm.weight,
    "mlp.gate_proj.weight": layer.mlp.gate_proj.weight,
    "mlp.up_proj.weight": layer.mlp.up_proj.weight,
    "mlp.down_proj.weight": layer.mlp.down_proj.weight,
}
tensors = {k: v.astype(mx.float32) for k, v in tensors.items()}
meta = {"cfg": f"{H},{NH},{NKV},{HD},{INTER},{SEQ},{EPS!r},{THETA!r}"}

mx.save_safetensors(OUT, tensors, meta)
print(f"wrote {OUT}: {sorted(tensors)} meta={meta}")
