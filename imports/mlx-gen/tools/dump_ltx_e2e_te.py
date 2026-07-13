"""LTX-2.3 e2e golden — PHASE A: reference text encoder (sc-2679 S6).

Tokenizes a fixed prompt with the Gemma-3 `AutoTokenizer` (left-pad, `max_length`, `add_special_
tokens`) and runs the reference TE (Gemma backbone → per-token-RMS feature extractor → connector) →
`video_embeddings`. The TE is pure dense, so this runs in the **mflux venv (0.31.0)** (it has
`transformers`); the quantized pipeline (PHASE B) needs 0.31.2.

Writes the intermediate `tools/golden/ltx_e2e_te.safetensors` (input_ids + video_embeddings) consumed
by `dump_ltx_e2e_golden.py` (PHASE B).

Run:
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg \
      ~/Repos/mflux/.venv/bin/python tools/dump_ltx_e2e_te.py
"""

import sys
import types
from pathlib import Path

from _paths import fixture

ARC = str(Path.home() / ".cache/uv/archive-v0/DtG1XO51ABFxUGHg")  # mlx_video
VLM = str(Path.home() / ".cache/uv/archive-v0/69kyKiVsISWokLQN")  # mlx_vlm
LM = str(Path.home() / ".cache/uv/archive-v0/tKxBd9P9nMT7vnfO")  # mlx_lm
for p in (ARC, VLM, LM):
    sys.path.insert(0, p)

# __path__ stubs so importing mlx_vlm.models.gemma3.* skips the heavy mlx_vlm/__init__ chain.
_vlm = Path(VLM) / "mlx_vlm"
for name, d in [
    ("mlx_vlm", _vlm),
    ("mlx_vlm.models", _vlm / "models"),
    ("mlx_vlm.models.gemma3", _vlm / "models" / "gemma3"),
]:
    m = types.ModuleType(name)
    m.__path__ = [str(d)]
    sys.modules[name] = m

import glob  # noqa: E402

import mlx.core as mx  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from mlx_video.models.ltx.text_encoder import (  # noqa: E402
    Embeddings1DConnector,
    LanguageModel,
    norm_and_concat_per_token_rms,
    rescale_norm,
)

BASE = Path.home() / "Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8"
HIDDEN, OUT_DIM = 3840, 4096
DIM, HEADS, HEAD_DIM, LAYERS, REGISTERS, MAX_POS = 4096, 32, 128, 8, 128, [4096]
# Must be a multiple of the connector's 128 learnable registers (it tiles registers over seq_len).
MAX_LEN = 128
PROMPT = "A cat playing a grand piano on a city rooftop at sunset."


def gemma_path() -> str:
    base = Path.home() / ".cache/huggingface/hub/models--mlx-community--gemma-3-12b-it-bf16/snapshots"
    snaps = sorted(glob.glob(str(base / "*")))
    if not snaps:
        raise SystemExit("gemma-3-12b-it-bf16 snapshot not found in HF cache")
    return snaps[-1]


gp = gemma_path()
tok = AutoTokenizer.from_pretrained(gp, trust_remote_code=True)
tok.padding_side = "left"
enc = tok(PROMPT, return_tensors="np", max_length=MAX_LEN, truncation=True, padding="max_length")
input_ids = mx.array(enc["input_ids"])
attention_mask = mx.array(enc["attention_mask"])

lm = LanguageModel.from_pretrained(gp)
mx.eval(lm.parameters())
_, all_hidden = lm(
    inputs=input_ids, input_embeddings=None, attention_mask=attention_mask, output_hidden_states=True
)

normed = norm_and_concat_per_token_rms(all_hidden, attention_mask).astype(mx.bfloat16)
rescaled = rescale_norm(normed, OUT_DIM, HIDDEN)
raw = mx.load(str(BASE / "connector.safetensors"))
agg_w = raw["text_embedding_projection.video_aggregate_embed.weight"].astype(mx.bfloat16)
agg_b = raw["text_embedding_projection.video_aggregate_embed.bias"].astype(mx.bfloat16)
video_features = rescaled @ agg_w.T + agg_b

conn = Embeddings1DConnector(
    dim=DIM, num_heads=HEADS, head_dim=HEAD_DIM, num_layers=LAYERS,
    num_learnable_registers=REGISTERS, positional_embedding_max_pos=MAX_POS,
    apply_gated_attention=True,
)
prefix = "video_embeddings_connector."
mapped, registers = {}, None
for k, v in raw.items():
    if not k.startswith(prefix):
        continue
    sub = k[len(prefix):]
    v = v.astype(mx.bfloat16)
    if sub == "learnable_registers":
        registers = v
        continue
    sub = sub.replace(".ff.net.0.proj.", ".ff.proj_in.").replace(".ff.net.2.", ".ff.proj_out.")
    sub = sub.replace(".to_out.0.", ".to_out.")
    mapped[sub] = v
conn.load_weights(list(mapped.items()), strict=False)
if registers is not None:
    conn.learnable_registers = registers
mx.eval(conn.parameters())

additive = (attention_mask.astype(mx.bfloat16) - 1.0).reshape(attention_mask.shape[0], 1, 1, -1) * 1e9
video_embeddings, _ = conn(video_features, additive)
mx.eval(video_embeddings)

tensors = {
    "input_ids": input_ids.astype(mx.int32),
    "video_embeddings": video_embeddings.astype(mx.bfloat16),
}
out = fixture("tools/golden/ltx_e2e_te.safetensors")
Path(out).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out, tensors, metadata={"max_len": str(MAX_LEN), "prompt": PROMPT})
print(f"wrote {out}: input_ids {input_ids.shape}, video_embeddings {video_embeddings.shape}")
