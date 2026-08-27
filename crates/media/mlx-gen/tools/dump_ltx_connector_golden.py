"""LTX-2.3 connector golden — reference Embeddings1DConnector I/O (sc-2679 S1, re-derived sc-21663).

Weight-grounded but Gemma-free: loads `video_embeddings_connector.*` / `audio_embeddings_connector.*`
from the LTX-2.3 tier's `connector.safetensors`, builds the reference connector pair (dim 4096 =
32 heads x 128 video / 2048 = 32 x 64 audio, 8 layers, gated, 128 registers, max_pos [4096]), runs
an **f32** forward over a deterministic left-padded random feature input, and dumps
input / mask / output. The Rust `Connector` (mlx-gen-ltx/tests/connector_parity.rs and its candle
twin) loads the SAME connector.safetensors weights and must reproduce the outputs.

Oracle provenance (sc-21663). The semantic authority is Lightricks' canonical **ltx_core**
(Lightricks/LTX-2 @ v1.2.0, `_ltx25_diffvae_ref.REFERENCE_COMMIT` — the stack the checkpoints were
trained with). The mlx_video port this golden was originally dumped from disagrees with it in
three places, each patched below before the dump:

* the per-head attention gate is `sigmoid(logits)` where ltx_core uses `2 * sigmoid(logits)`
  (zero-init identity; mlx_video's own DiT attention applies the `2 *` and only its text-encoder
  connector forgot it),
* the FFN activation is exact erf-GELU where ltx_core's `GELUApprox` is tanh-approximate, and
* the RoPE table keeps float64 through cos/sin where ltx_core's `generate_freq_grid_np` rounds the
  log-spaced indices to float32 BEFORE the position multiply (its "double precision" covers only
  the exponentials; the ~1e-3-rad top-frequency difference matters — see the bar note below).

Why the execution vehicle is still MLX (mlx_video's module, patched) rather than torch: the
corrected `2·sigmoid` gating makes the 8-layer stack expansive (~2x/layer), so ANY per-op backend
difference is amplified ~50x by the final output. Measured on these weights+inputs, plain f32
GEMM accumulation-order differences between MLX-Metal and torch-CPU (~3e-4 peak-rel per 4096-wide
projection) reach ~1.4e-2 video / ~8e-2 audio at the output — a cross-backend floor far above the
5e-3 bar `connector_parity.rs` holds the port to. Dumping from patched-MLX keeps the comparison
same-framework (the old fixture's convention too), preserving the bar's power; the semantics were
cross-checked against stock torch ltx_core (`EmbeddingsProcessor.create_embeddings` at the pinned
commit), which reproduces this oracle exactly to that measured cross-backend floor and reproduces
the PREVIOUS (un-patched) fixture to 1.9e-4 when the three divergences are re-introduced — i.e.
the patches below are the complete semantic delta.

Run (mflux venv + mlx_video source):
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg \
    LTX_EROS_DIR=<LTX-2.3 tier dir with connector.safetensors + embedded_config.json> \
      ~/Repos/mflux/.venv/bin/python tools/dump_ltx_connector_golden.py
Output (committed): mlx-gen-ltx/tests/fixtures/ltx_connector_golden.safetensors
"""

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

from _ltx25_diffvae_ref import REFERENCE_COMMIT
from _paths import fixture, require_env


def _find_mlx_video_src() -> str:
    if env := os.environ.get("MLX_VIDEO_SRC"):
        return str(Path(env).expanduser())
    for cand in sorted(glob.glob(str(Path.home() / ".cache/uv/archive-v0/*/mlx_video"))):
        return str(Path(cand).parent)
    raise SystemExit("Set MLX_VIDEO_SRC to the dir containing `mlx_video/`.")


sys.path.insert(0, _find_mlx_video_src())

# text_encoder.py imports `mlx_vlm.models.gemma3.{language,config}` at module load (pulling in the
# whole mlx_lm/mlx_vlm tree). We only need `Embeddings1DConnector`, which never touches the Gemma
# class, so stub those names rather than installing the dependency tree.
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
import mlx.nn as nn  # noqa: E402

from mlx_video.models.ltx.text_encoder import (  # noqa: E402
    ConnectorAttention,
    ConnectorFeedForward,
    Embeddings1DConnector,
)

# --- ltx_core (training) semantics over the mlx_video port — see the module docstring. -----------


def _ff_call_tanh_gelu(self, x):
    x = nn.gelu_approx(self.proj_in(x))  # ltx_core GELUApprox: gelu(x, approximate="tanh")
    x = self.dropout(x)
    return self.proj_out(x)


def _attn_call_2sigmoid(self, x, attention_mask=None, pe=None):
    batch_size, seq_len, _ = x.shape
    q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
    q, k = self.q_norm(q), self.k_norm(k)
    q = mx.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
    k = mx.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
    v = mx.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
    if pe is not None:
        q = self._apply_split_rope(q, pe[0], pe[1])
        k = self._apply_split_rope(k, pe[0], pe[1])
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
    out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
    if self.to_gate_logits is not None:
        gates = 2.0 * nn.sigmoid(self.to_gate_logits(x))  # ltx_core: 2·sigmoid, zero-init identity
        gates = mx.expand_dims(gates, axis=-1)
        out = mx.reshape(out, (batch_size, seq_len, self.num_heads, self.head_dim))
        out = out * gates
        out = mx.reshape(out, (batch_size, seq_len, -1))
    return self.to_out(out)


def _rope_f32_quantized_indices(self, seq_len, dtype):
    """ltx_core `generate_freq_grid_np` + `generate_freqs`: f64 exponentials rounded to f32 BEFORE
    the (f32) position multiply; cos/sin on the f32 angles."""
    dim = self.num_heads * self.head_dim
    n = dim // 2  # n_elem = 2 * len(max_pos) = 2
    step = 1.0 / (n - 1)
    idx32 = (np.power(10000.0, np.arange(n, dtype=np.float64) * step) * (np.pi / 2)).astype(np.float32)
    t = np.arange(seq_len, dtype=np.float64)
    scaled32 = ((t / self.positional_embedding_max_pos[0]) * 2.0 - 1.0).astype(np.float32)
    ang = (scaled32[:, None] * idx32[None, :]).astype(np.float32)
    half = self.head_dim // 2
    cos = np.cos(ang).reshape(seq_len, self.num_heads, half).transpose(1, 0, 2)[np.newaxis]
    sin = np.sin(ang).reshape(seq_len, self.num_heads, half).transpose(1, 0, 2)[np.newaxis]
    return mx.array(cos.astype(np.float32)).astype(dtype), mx.array(sin.astype(np.float32)).astype(dtype)


ConnectorFeedForward.__call__ = _ff_call_tanh_gelu
ConnectorAttention.__call__ = _attn_call_2sigmoid
Embeddings1DConnector._precompute_freqs_cis = _rope_f32_quantized_indices
# -------------------------------------------------------------------------------------------------

MODEL_DIR = Path(require_env(
    "LTX_EROS_DIR",
    "the LTX-2.3 model/tier directory holding connector.safetensors and embedded_config.json "
    "(e.g. the SceneWorks/ltx-2.3-mlx bf16 tier)",
)).expanduser()

tcfg = json.loads((MODEL_DIR / "embedded_config.json").read_text())["transformer"]
DIM = int(tcfg["connector_num_attention_heads"]) * int(tcfg["connector_attention_head_dim"])
HEADS = int(tcfg["connector_num_attention_heads"])
HEAD_DIM = int(tcfg["connector_attention_head_dim"])
LAYERS = int(tcfg["connector_num_layers"])
REGISTERS = int(tcfg["connector_num_learnable_registers"])
MAX_POS = list(tcfg["connector_positional_embedding_max_pos"])
AUDIO_HEADS = int(tcfg["audio_connector_num_attention_heads"])
AUDIO_HEAD_DIM = int(tcfg["audio_connector_attention_head_dim"])
AUDIO_DIM = AUDIO_HEADS * AUDIO_HEAD_DIM
GATED = bool(tcfg["connector_apply_gated_attention"])

SEQ, NUM_VALID = 256, 40  # left-padded: 216 pad + 40 valid; SEQ % REGISTERS == 0.

mx.random.seed(0)

# Build the reference connector with the tier config + gated attention.
conn = Embeddings1DConnector(
    dim=DIM,
    num_heads=HEADS,
    head_dim=HEAD_DIM,
    num_layers=LAYERS,
    num_learnable_registers=REGISTERS,
    positional_embedding_max_pos=MAX_POS,
    apply_gated_attention=GATED,
)

# Load video connector weights from connector.safetensors with the reference key remapping.
raw = mx.load(str(MODEL_DIR / "connector.safetensors"))


def load_connector(module, prefix):
    mapped, registers = {}, None
    for k, v in raw.items():
        if not k.startswith(prefix):
            continue
        sub = k[len(prefix):]
        v = v.astype(mx.float32)  # f32 reference
        if sub == "learnable_registers":
            registers = v
            continue
        sub = sub.replace(".ff.net.0.proj.", ".ff.proj_in.")
        sub = sub.replace(".ff.net.2.", ".ff.proj_out.")
        sub = sub.replace(".to_out.0.", ".to_out.")
        mapped[sub] = v
    module.load_weights(list(mapped.items()), strict=False)
    if registers is not None:
        module.learnable_registers = registers
    mx.eval(module.parameters())


load_connector(conn, "video_embeddings_connector.")

# Deterministic left-padded input + additive mask.
features = mx.random.normal((1, SEQ, DIM)).astype(mx.float32)
mask01 = mx.concatenate(
    [mx.zeros((1, SEQ - NUM_VALID), dtype=mx.int32), mx.ones((1, NUM_VALID), dtype=mx.int32)],
    axis=1,
)
additive = (mask01.astype(mx.float32) - 1.0).reshape(1, 1, 1, SEQ) * 1e9

video_embeddings, _ = conn(features, additive)
mx.eval(video_embeddings)

# --- Audio connector (sc-2684): dim 2048 = 32 heads × 64 head_dim, same 8 layers / 128 regs. ---
audio_conn = Embeddings1DConnector(
    dim=AUDIO_DIM,
    num_heads=AUDIO_HEADS,
    head_dim=AUDIO_HEAD_DIM,
    num_layers=LAYERS,
    num_learnable_registers=REGISTERS,
    positional_embedding_max_pos=MAX_POS,
    apply_gated_attention=GATED,
)
load_connector(audio_conn, "audio_embeddings_connector.")

audio_features = mx.random.normal((1, SEQ, AUDIO_DIM)).astype(mx.float32)
audio_embeddings, _ = audio_conn(audio_features, additive)
mx.eval(audio_embeddings)

for name, t in (("video_embeddings", video_embeddings), ("audio_embeddings", audio_embeddings)):
    if not mx.isfinite(t).all().item():
        raise SystemExit(f"{name} contains a non-finite value")

tensors = {
    "features": features,
    "mask01": mask01.astype(mx.int32),
    "video_embeddings": video_embeddings.astype(mx.float32),
    "audio_features": audio_features,
    "audio_embeddings": audio_embeddings.astype(mx.float32),
}
meta = {
    "seq": str(SEQ),
    "num_valid": str(NUM_VALID),
    "dim": str(DIM),
    "audio_dim": str(AUDIO_DIM),
    "reference": (
        f"ltx_core semantics (Lightricks/LTX-2 @ {REFERENCE_COMMIT}, v1.2.0): 2*sigmoid gate, "
        "tanh-approx GELU, f32-quantized RoPE indices — executed via the patched mlx_video "
        "Embeddings1DConnector (same-framework oracle, sc-21663; see the dumper docstring)"
    ),
    "weights": "connector.safetensors, video_embeddings_connector.* + audio_embeddings_connector.*",
    "dtype": "f32",
}
out = fixture("mlx-gen-ltx/tests/fixtures/ltx_connector_golden.safetensors")
Path(out).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out, tensors, metadata=meta)
print(f"wrote {out}")
print(f"  features {features.shape}  video_embeddings {video_embeddings.shape}")
print(f"  audio_features {audio_features.shape}  audio_embeddings {audio_embeddings.shape}")
