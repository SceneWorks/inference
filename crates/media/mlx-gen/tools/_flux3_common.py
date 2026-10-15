"""Shared setup for the FLUX 3 Image Klein fixture generators (`dump_flux3_*.py`, sc-26234).

Python is used ONLY here, offline, as the reference oracle: the frozen upstream modules are imported
from a source checkout and run on CPU. Nothing at runtime depends on Python.

Frozen upstream (recorded in `gen-core/src/flux3.rs` as constants — keep the two in lockstep):
  * code      GitHub `black-forest-labs/flux-3-image-klein` @ 45e57c9fb06f9751cdced29a65c5d2171003be05
  * package   HF `black-forest-labs/flux-3-image-klein` @ 8454017f9ebaada30c998fccd9f67a80c7580f68
              (DiT bf16 + `ae.safetensors` f32 + the bundled `text_encoder/`, byte-identical to
              `Qwen/Qwen3-VL-4B-Instruct` @ ebb281ec70b05090aa6165b016eac8ec08e71b17)

Inputs (no machine-specific defaults are baked in):
  FLUX3_SRC  path to the frozen source tree (a git checkout or the extracted GitHub tarball of the
             pinned commit). The sha256 of every upstream module the oracle imports is verified
             before use, so a drifted tree fails loudly instead of producing a silently different
             golden.

Run inside an isolated venv (torch CPU, transformers >= 4.57 with Qwen3-VL, safetensors, tokenizers)
with `python -I`, e.g.:
    FLUX3_SRC=/path/to/flux-3-image-klein python -I tools/dump_flux3_golden.py

The upstream `sampling.py` decorates its loops with `torch.autocast("cuda", bfloat16)`, which is a
no-op on CPU: the oracle therefore runs the DiT in **f32** (weights cast to f32) for the DiT, sampler
and end-to-end goldens, and the native f32 gates (`Precision::Fp32` on the MLX CPU stream) are held to
that. A separate bf16 DiT golden runs the same forward under `torch.autocast("cpu", bfloat16)` — the
CPU twin of the release's CUDA autocast policy — for the bf16 compute gate.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from _paths import fixture, require_env

FLUX3_COMMIT = "45e57c9fb06f9751cdced29a65c5d2171003be05"
FLUX3_PACKAGE_REVISION = "8454017f9ebaada30c998fccd9f67a80c7580f68"
QWEN3_VL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
QWEN3_VL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

FIXTURE_DIR = Path(fixture("mlx-gen-flux3/tests/fixtures"))
SNAPSHOT = FIXTURE_DIR / "tiny-snapshot"
ENCODER_DIR = SNAPSHOT / "text_encoder"

# sha256 of every upstream module the oracle executes, at FLUX3_COMMIT. A mismatch means the tree is
# not the frozen source.
UPSTREAM_SHA256 = {
    "src/flux3/__init__.py": "3b3abca628f914a41617041b8a5e550e71fde73d25dbde6e546ed49aa4ca74fc",
    "src/flux3/autoencoder.py": "960e2a98046bcb78f2d36fbbd4848db726f14810c7887485f7e2682346f95189",
    "src/flux3/dit.py": "d4c0d2aea5282ae63ac32a1f3029a738ddab951bb1033d6f3a2e9bd192c89934",
    "src/flux3/sampling.py": "22be0d823fb6a2c5bc993c77fe0836ca729219d34cc08e86d01a398e04bdcdf5",
    "src/flux3/text_encoder.py": "3bce784f43b8e8d9eee0dab75d928006398efd80e77f9f179bba6ddd28ffe74b",
    "src/flux3/pipeline.py": "33c323ebaeaf56790f3d0022a57827b5348446a8a4b372ab2bcc606d722e76a9",
    "src/flux3/checkpoint.py": "a49b33fc83c391fdac6628be39ca3748c88bfd8e63b731f997d2f161d5ef9be3",
}

# ---- miniature geometry (shared by every dump script so the e2e fixture composes) ---------------
# Autoencoder: the release is CH 128, CH_MULT (1, 2, 4, 4), 2 resnets, EMBED_DIM 32 (128 packed
# channels). The miniature keeps every structural switch (four levels, two resnets, mid attention,
# GroupNorm with 32 groups) and shrinks the widths: 32 channels is the smallest GroupNorm(32) width.
AE_CH = 32
AE_CH_MULT = (1, 2, 4, 4)
AE_NUM_RES_BLOCKS = 2
AE_EMBED_DIM = 4
AE_LATENT_CHANNELS = 4 * AE_EMBED_DIM  # packed token channels (128 in the release)
# Text encoder: a full-depth 36-layer miniature Qwen3-VL tower, 32 wide, so the release's layer
# selection [4, 8, ..., 32] is the law under test.
TEXT_DIM = 32
TEXT_LAYERS = 36
OUTPUT_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
PAD_MULTIPLE = 8  # 80 in the release; the window law is the same
# DiT: HIDDEN 3072 / HEADS 24 / DEPTH 5 / DEPTH_SINGLE 28 / AXES (32, 32, 32, 32) in the release.
DIT_HIDDEN = 64
DIT_HEADS = 2
DIT_DEPTH = 2
DIT_DEPTH_SINGLE = 3
DIT_AXES_DIM = (8, 8, 8, 8)  # head dim 32
DIT_MLP_RATIO = 3.0
DIT_THETA = 10000
DIT_IN_CHANNELS = {"image": AE_LATENT_CHANNELS, "image_cond": AE_LATENT_CHANNELS, "text": len(OUTPUT_LAYERS) * TEXT_DIM}
# Perturbation scale of the miniature towers' weights: small enough that the random blocks stay
# contractive, so bf16 activation rounding does not compound into O(1) drift with depth.
TOWER_SCALE = 0.1


def upstream_root() -> Path:
    root = Path(require_env("FLUX3_SRC", "the frozen black-forest-labs/flux-3-image-klein source at " + FLUX3_COMMIT))
    for rel, want in UPSTREAM_SHA256.items():
        got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"{rel}: sha256 {got} != pinned {want}")
    return root


def import_upstream() -> Path:
    """Verify the frozen tree and put `src/` on `sys.path` so `import flux3.*` resolves."""
    root = upstream_root()
    sys.path.insert(0, str(root / "src"))
    return root


def patch_autoencoder_constants(autoencoder_module) -> None:
    """Shrink upstream `autoencoder.py`'s module constants to the miniature geometry. The classes
    read them at construction, so a patched module builds the miniature graph through the real
    upstream code."""
    autoencoder_module.CH = AE_CH
    autoencoder_module.CH_MULT = AE_CH_MULT
    autoencoder_module.NUM_RES_BLOCKS = AE_NUM_RES_BLOCKS
    autoencoder_module.EMBED_DIM = AE_EMBED_DIM
    autoencoder_module.LATENT_CHANNELS = AE_LATENT_CHANNELS


def patch_dit_constants(dit_module) -> None:
    """Shrink upstream `dit.py`'s module constants to the miniature geometry."""
    dit_module.IN_CHANNELS = dict(DIT_IN_CHANNELS)
    dit_module.HIDDEN = DIT_HIDDEN
    dit_module.HEADS = DIT_HEADS
    dit_module.DEPTH = DIT_DEPTH
    dit_module.DEPTH_SINGLE = DIT_DEPTH_SINGLE
    dit_module.AXES_DIM = DIT_AXES_DIM
    dit_module.MLP_RATIO = DIT_MLP_RATIO
    dit_module.THETA = DIT_THETA


def tiny_autoencoder(autoencoder_module):
    """The seeded miniature autoencoder through upstream's `AutoEncoder` class (module constants
    patched), every parameter perturbed away from its init and the BatchNorm running statistics
    set to non-trivial values so the normalization is load-bearing. Deterministic: every dump
    script that needs the autoencoder builds the same weights."""
    import torch

    patch_autoencoder_constants(autoencoder_module)
    g = torch.Generator().manual_seed(2602)
    model = autoencoder_module.AutoEncoder().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or ".norm" in name and name.endswith(".weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.2 * torch.randn(p.shape, generator=g))
        model.bn.running_mean.copy_(0.5 * torch.randn(model.bn.running_mean.shape, generator=g))
        model.bn.running_var.copy_(0.5 + torch.rand(model.bn.running_var.shape, generator=g))
    return model


def write_autoencoder_fixture(model) -> Path:
    """`tiny-snapshot/ae.safetensors` in the released layout (upstream key names, f32)."""
    from safetensors.torch import save_file

    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT / "ae.safetensors"
    state = {k: v.detach().contiguous().to(__import__("torch").float32) for k, v in model.state_dict().items()}
    save_file(state, str(path), metadata={"format": "pt"})
    print(f"wrote {path} ({len(state)} tensors, {path.stat().st_size} bytes)")
    return path


def save_safetensors(path: Path, tensors: dict, metadata: dict | None = None) -> None:
    import torch
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {
        k: v.detach().contiguous().to(torch.float32) if v.is_floating_point() else v.detach().contiguous()
        for k, v in tensors.items()
    }
    save_file(flat, str(path), metadata={k: str(v) for k, v in (metadata or {}).items()})
    print(f"wrote {path} ({len(flat)} tensors, {path.stat().st_size} bytes)")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}")
