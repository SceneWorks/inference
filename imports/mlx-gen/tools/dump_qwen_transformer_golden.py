"""Dump Qwen-Image transformer parity goldens for the Rust port (sc-2348, slice 3).

The transformer is ~20B params (can't dump weights), so this validates the novel math two ways:
  1. **3D RoPE** (`QwenEmbedRopeMLX`) — no weights; compare img/txt cos/sin for a fixed grid.
  2. **One dual-stream block** at small dims (dim 256, 2 heads × 128) with **synthetic** weights —
     dump the block's f32 params (fork-internal keys) + fixed inputs + the fork's outputs.
The full 60-layer real-weight forward is validated end-to-end (slice 4) against the image golden.

Run from the mflux fork venv:
    cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/dump_qwen_transformer_golden.py
Output (gitignored): tools/golden/qwen_transformer_golden.safetensors
"""

import os

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from mflux.models.qwen.model.qwen_transformer.qwen_rope import QwenEmbedRopeMLX
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import QwenTransformerBlock

mx.random.seed(0)
rope = QwenEmbedRopeMLX(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)

# --- 1. RoPE golden: a non-square grid + text length that exercises the scale_rope centering. ---
ROPE_H, ROPE_W, ROPE_TXT = 64, 48, 20
(ric, ris), (rtc, rts) = rope(video_fhw=[(1, ROPE_H, ROPE_W)], txt_seq_lens=[ROPE_TXT])

# --- 2. Single dual-stream block at small dims with synthetic (random f32) weights. ---
DIM, HEADS, HD = 256, 2, 128
BH, BW, BTXT = 4, 4, 8  # img_seq = 16
blk = QwenTransformerBlock(dim=DIM, num_heads=HEADS, head_dim=HD)
blk.update(tree_map(lambda a: a.astype(mx.float32), blk.parameters()))

hidden = mx.random.normal((1, BH * BW, DIM)).astype(mx.float32)
enc = mx.random.normal((1, BTXT, DIM)).astype(mx.float32)
temb = mx.random.normal((1, DIM)).astype(mx.float32)
(bic, bis), (btc, bts) = rope(video_fhw=[(1, BH, BW)], txt_seq_lens=[BTXT])
enc_out, hid_out = blk(
    hidden_states=hidden,
    encoder_hidden_states=enc,
    encoder_hidden_states_mask=None,
    text_embeddings=temb,
    image_rotary_emb=((bic, bis), (btc, bts)),
    block_idx=0,
)
mx.eval(ric, ris, rtc, rts, enc_out, hid_out)

out = {k: v.astype(mx.float32) for k, v in tree_flatten(blk.parameters())}  # block params (no prefix)
out.update(
    {
        "rope_img_cos": ric.astype(mx.float32),
        "rope_img_sin": ris.astype(mx.float32),
        "rope_txt_cos": rtc.astype(mx.float32),
        "rope_txt_sin": rts.astype(mx.float32),
        "io_hidden": hidden,
        "io_enc": enc,
        "io_temb": temb,
        "io_img_cos": bic.astype(mx.float32),
        "io_img_sin": bis.astype(mx.float32),
        "io_txt_cos": btc.astype(mx.float32),
        "io_txt_sin": bts.astype(mx.float32),
        "io_hidden_out": hid_out.astype(mx.float32),
        "io_enc_out": enc_out.astype(mx.float32),
    }
)
golden_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(golden_dir, exist_ok=True)
path_out = os.path.join(golden_dir, "qwen_transformer_golden.safetensors")
mx.save_safetensors(path_out, out)
print(f"rope img={ric.shape} txt={rtc.shape}; block hid_out={hid_out.shape} enc_out={enc_out.shape}")
print(f"wrote {path_out} ({len(out)} tensors)")
