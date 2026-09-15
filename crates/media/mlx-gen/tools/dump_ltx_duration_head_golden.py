"""LTX-2.5 `DurationHead` reference golden — sc-18774.

Dumps f32 reference I/O for the MLX/candle `DurationHead` ports, run against the REAL
`model_patches/ltx-2.5-duration-head-bf16.safetensors` (~4 MB — the cheapest new module in epic
18755).

Unlike the other `dump_ltx25_*_golden.py` scripts, this one does NOT need `LTX2_SRC` (a full
`Lightricks/LTX-2` checkout importable as `ltx_core`): `DurationHead`/`AttentionPooler` have no
dependency on the rest of that package beyond `torch.nn` and a `Disposable` lifecycle mixin that is
never touched by `forward()`. The module class below is copied verbatim from
`packages/ltx-core/src/ltx_core/duration_head/duration_head.py` at the pinned reference commit
(see REFERENCE_COMMIT), with the `Disposable` base dropped (forward-pass-irrelevant) and otherwise
byte-identical — diff it against the upstream file if the pin ever moves.

Three golden cases, since the head is explicitly modality-agnostic (upstream: "pass either or both
of {audio, video} connector outputs"): video-only, audio-only, and both together. Inputs are the
same deterministic, smooth `probe()` field used by `dump_ltx25_diffvae_golden.py` (see
`_ltx25_diffvae_ref.py`), reproduced here directly (no LTX2_SRC needed for it either) so a caption
connector's real activation statistics don't matter — any deterministic tensor is a valid probe for
the module's arithmetic.

Everything runs f32: the checkpoint ships bf16, both Rust ports upcast to f32 immediately (this
head is a "quality island" utility, not part of the hot denoise loop — see the ports' module docs),
so this is a correctness check, not a rounding check.

Run:
    LTX25_DURATION_HEAD_FILE=/Volumes/Models/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/<rev>/model_patches/ltx-2.5-duration-head-bf16.safetensors \\
      python3 tools/dump_ltx_duration_head_golden.py
Output (committed, copied to BOTH backends):
    mlx-gen-ltx/tests/fixtures/ltx_duration_head_golden.safetensors
    candle-gen-ltx/../candle-gen-ltx/tests/fixtures/ltx_duration_head_golden.safetensors
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from _paths import fixture

REFERENCE_COMMIT = "d151147788a9284cca791edc6ce898007e727fe6"
POOLER_HIDDEN_DIM = 256
NUM_QUERIES = 1
NUM_POOLER_HEADS = 4
MLP_HIDDEN = 256
VIDEO_CROSS_ATTENTION_DIM = 4096
AUDIO_CROSS_ATTENTION_DIM = 2048


# --- verbatim from packages/ltx-core/src/ltx_core/duration_head/duration_head.py @ REFERENCE_COMMIT,
# --- minus the `Disposable` mixin (irrelevant to forward()) ------------------------------------
class AttentionPooler(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_queries: int = 1, num_heads: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        pooled, _ = self.cross_attn(queries, tokens, tokens, need_weights=False)
        return pooled


class DurationHead(nn.Module):
    def __init__(
        self,
        video_cross_attention_dim: int = 4096,
        audio_cross_attention_dim: int = 2048,
        pooler_hidden_dim: int = 256,
        num_queries: int = 1,
        num_pooler_heads: int = 4,
        mlp_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.pooler_hidden_dim = pooler_hidden_dim

        self.video_input_proj = nn.Linear(video_cross_attention_dim, pooler_hidden_dim)
        self.video_modality_emb = nn.Parameter(torch.randn(pooler_hidden_dim) * 0.02)

        self.audio_input_proj = nn.Linear(audio_cross_attention_dim, pooler_hidden_dim)
        self.audio_modality_emb = nn.Parameter(torch.randn(pooler_hidden_dim) * 0.02)

        self.attention_pooler = AttentionPooler(
            hidden_dim=pooler_hidden_dim,
            num_queries=num_queries,
            num_heads=num_pooler_heads,
        )
        self.mlp_hidden = nn.Linear(pooler_hidden_dim * num_queries, mlp_hidden)
        self.mlp_out = nn.Linear(mlp_hidden, 1)

    def forward(
        self,
        video_tokens: torch.Tensor | None = None,
        audio_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if video_tokens is None and audio_tokens is None:
            raise ValueError("DurationHead.forward requires at least one of video_tokens / audio_tokens")

        token_groups: list[torch.Tensor] = []
        if video_tokens is not None:
            token_groups.append(self.video_input_proj(video_tokens) + self.video_modality_emb)
        if audio_tokens is not None:
            token_groups.append(self.audio_input_proj(audio_tokens) + self.audio_modality_emb)

        tokens = torch.cat(token_groups, dim=1)
        pooled = self.attention_pooler(tokens)
        pooled_flat = pooled.reshape(pooled.shape[0], -1)
        hidden = torch.nn.functional.gelu(self.mlp_hidden(pooled_flat), approximate="tanh")
        log_duration = self.mlp_out(hidden).squeeze(-1)
        return log_duration.exp()


# --- end verbatim reference ----------------------------------------------------------------------


def probe(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """Deterministic, band-limited probe tensor (same formula as `_ltx25_diffvae_ref.probe`)."""
    n = 1
    for dim in shape:
        n *= dim
    idx = torch.arange(n, dtype=torch.float64)
    values = (
        torch.sin(idx * 0.013_1 + seed * 1.7) * torch.cos(idx * 0.007_3 - seed * 0.31) * 0.9
        + 0.1 * torch.sin(idx * 0.000_37 + seed)
    )
    return values.reshape(shape).to(torch.float16).to(torch.float32)


def duration_head_file() -> Path:
    if path := os.environ.get("LTX25_DURATION_HEAD_FILE"):
        return Path(path)
    default = (
        Path.home()
        / "Library/Application Support/SceneWorks/data/models/mlx/ltx_2_5"
        / "model_patches/ltx-2.5-duration-head-bf16.safetensors"
    )
    return default


def load_reference_head(path: Path) -> DurationHead:
    """Build the reference `DurationHead` and load the real checkpoint's `duration_head.*`-prefixed
    weights (stripping the prefix — the checkpoint's module IS this head, at the top level)."""
    head = DurationHead(
        video_cross_attention_dim=VIDEO_CROSS_ATTENTION_DIM,
        audio_cross_attention_dim=AUDIO_CROSS_ATTENTION_DIM,
        pooler_hidden_dim=POOLER_HIDDEN_DIM,
        num_queries=NUM_QUERIES,
        num_pooler_heads=NUM_POOLER_HEADS,
        mlp_hidden=MLP_HIDDEN,
    )
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():  # noqa: SIM118
            if not key.startswith("duration_head."):
                continue
            stripped = key[len("duration_head.") :]
            state[stripped] = f.get_tensor(key).to(torch.float32)
    missing, unexpected = head.load_state_dict(state, strict=True)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    return head.eval()


def report(name: str, tensor: torch.Tensor) -> None:
    flat = tensor.detach().to(torch.float32).flatten()
    print(f"  {name:<20} {tuple(tensor.shape)!s:<16} value(s) {flat.tolist()}")


def main() -> None:
    path = duration_head_file()
    print(f"[ref] {path}")
    if not path.is_file():
        raise SystemExit(f"duration-head checkpoint not found: {path} (set LTX25_DURATION_HEAD_FILE)")
    head = load_reference_head(path)

    torch.manual_seed(0)
    video = probe((1, 6, VIDEO_CROSS_ATTENTION_DIM), seed=1)
    audio = probe((1, 4, AUDIO_CROSS_ATTENTION_DIM), seed=2)

    with torch.no_grad():
        seconds_video_only = head(video, None)
        seconds_audio_only = head(None, audio)
        seconds_both = head(video, audio)

    for name, t in [
        ("seconds_video_only", seconds_video_only),
        ("seconds_audio_only", seconds_audio_only),
        ("seconds_both", seconds_both),
    ]:
        report(name, t)
        if not torch.isfinite(t).all():
            raise SystemExit(f"{name} is not finite — refusing to commit a poisoned golden")

    tensors: dict[str, torch.Tensor] = {
        "video_tokens": video.contiguous(),
        "audio_tokens": audio.contiguous(),
        "seconds_video_only": seconds_video_only.contiguous(),
        "seconds_audio_only": seconds_audio_only.contiguous(),
        "seconds_both": seconds_both.contiguous(),
    }
    meta = {
        "story": "sc-18774",
        "reference": f"Lightricks/LTX-2 @ {REFERENCE_COMMIT} (v1.2.0)",
        "checkpoint": "model_patches/ltx-2.5-duration-head-bf16.safetensors",
        "dtype": "float32 (weights upcast from bf16)",
        "note": "DurationHead/AttentionPooler copied verbatim from the reference module (minus the "
        "forward-irrelevant Disposable mixin) rather than imported via LTX2_SRC, since this head "
        "has no other ltx_core dependency.",
    }

    out_mlx = fixture("mlx-gen-ltx/tests/fixtures/ltx_duration_head_golden.safetensors")
    save_file(tensors, out_mlx, metadata=meta)
    print(f"[out] {out_mlx}")

    out_candle = fixture(
        "../candle-gen/candle-gen-ltx/tests/fixtures/ltx_duration_head_golden.safetensors"
    )
    shutil.copyfile(out_mlx, out_candle)
    print(f"[out] {out_candle}")


if __name__ == "__main__":
    main()
