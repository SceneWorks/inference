"""Mage-Flow (microsoft/Mage, MIT) parity-reference harness — sc-14036 (epic 14034).

Emits the deterministic boundary goldens that gate the native Mage-Flow port's Rust parity
suites (`mlx-gen-mage` in P1-P3, `candle-gen-mage` in P4). This is the **parity oracle**: it
drives the *vendored, frozen* reference implementation at
`_vendor/mage_flow/` (see `_vendor/VENDORED.md`) at a fixed prompt / seed /
geometry and dumps per-boundary `.safetensors` tensors the Rust engines must reproduce.

Python is permitted in **test harnesses only** — never in the product path. This mirrors the
existing reference dumps in this directory (`dump_mochi_golden.py`, `dump_flux2_e2e_golden.py`)
and the real-weights golden convention documented in `tools/golden/README.md`.

## Goldens emitted (into the gitignored `tools/golden/`)

| stage       | file                                    | boundary                                              |
| ----------- | --------------------------------------- | ----------------------------------------------------- |
| `noise`     | `mage_flow_noise_golden.safetensors`    | Gaussian-Shading initial noise (NOT plain randn)       |
| `vae`       | `mage_flow_vae_golden.safetensors`      | Mage-VAE encode (posterior moments at t=0) + decode    |
| `te`        | `mage_flow_te_golden.safetensors`       | Qwen3-VL conditioning (gen drop 34 / edit drop 64)     |
| `dit_block` | `mage_flow_dit_block_golden.safetensors`| ONE NR-MMDiT dual-stream block forward                 |
| `dit`       | `mage_flow_dit_golden.safetensors`      | the full 12-block stack forward (velocity)             |
| `e2e`       | `mage_flow_e2e_golden.safetensors`+`.png`| full txt2img denoise -> final latent + decoded image   |
| `edit`      | `mage_flow_edit_golden.safetensors`+`.png`| edit sequence assembly + edited image                 |

`dit_block` and `dit` are captured **for free** during the `e2e` denoise via
`register_forward_hook(with_kwargs=True)` on `transformer.transformer_blocks[0]` and on the
whole `transformer`, firing at sampler step 0. That fixes both fixtures to the reference
forward without hand-replicating patchify / RoPE / time-embed wiring here.

## What the goldens pin (the corrections that matter)

* The initial latent is **Gaussian-Shading watermarked noise** (`mage_latent.encode_noise`,
  key 20260720), NOT `utils.get_noise`'s plain `randn` — the reference computes the randn and
  then throws it away (`pipeline.py:303-308`). `plain_randn` is dumped alongside so a port can
  prove it is NOT accidentally matching the discarded tensor.
* TE conditioning is the **final (36th) hidden state AFTER the final RMSNorm**, not the
  penultimate layer: `text_encoder.py:156` reads `outputs[0]` from a `Qwen3VLTextModel.forward`
  whose last statement before returning is `hidden_states = self.norm(hidden_states)`
  (`text_encoder.py:290`). `*_hidden_full` is the pre-drop tensor and `*_txt` the post-drop one,
  so a port can bisect "wrong layer" from "wrong drop_idx".
* There is **no latent scale/shift**: the VAE latent enters the DiT through a bare
  `img_in = Linear(128 -> 3072)` (`mage_flow.py:73,109`).

## Determinism

Prompt / negative / seed / geometry / steps / cfg are all fixed (env-overridable). Sampling is
deterministic: the flow-match Euler schedule has no per-step noise and the initial latent is
key+seed derived (CPU generator inside `encode_noise`). Tensors are stored upcast to float32 and
`.contiguous()` (a strided reference tensor is silently mis-stored by `safetensors` against its
declared C-contiguous shape — see the `sc-11985` note in `dump_mochi_golden.py`).

The reference itself runs **bfloat16** — `pipeline.load_from_repo` hard-casts transformer / text
encoder / VAE to bf16 and `mage_layers.get_timestep_embedding` deliberately rounds its frequency
table to bf16 (the model was trained with that rounding). That is not configurable upstream, so
this harness does not invent a dtype knob.

## Prerequisites & run

Needs the pinned reference env (see `_vendor/VENDORED.md`): python 3.11/3.12,
torch 2.13.0, torchvision 0.28.0, diffusers 0.38.0, transformers 5.5.0 (`<5.6`),
accelerate 1.13.0, safetensors 0.8.0, einops, pydantic, pillow, loguru. `flash-attn 2.8.3` is
CUDA-only and is **not** required here: both the DiT and the HF text encoder route through the
reference's own backend shim, and this harness selects `sdpa` off CUDA
(`_attn_backend.set_attn_backend` / `text_encoder._resolve_hf_attn_impl` both accept it), so the
oracle runs on Mac (MPS/CPU) as well as CUDA.

    MAGE_SNAPSHOT=/path/to/Mage-Flow MAGE_EDIT_SNAPSHOT=/path/to/Mage-Flow-Edit \
      python tools/dump_mage_flow_golden.py --stage all

`--stage {noise,vae,te,dit_block,dit,e2e,edit,all}` selects which goldens to write (default
`all`). `noise` needs no weights at all; `vae` needs only `vae/`; the rest load the full repo.

Env overrides: `MAGE_SNAPSHOT`, `MAGE_EDIT_SNAPSHOT`, `MAGE_PROMPT`, `MAGE_NEG`,
`MAGE_EDIT_INSTRUCTION`, `MAGE_EDIT_REF`, `MAGE_SEED`, `MAGE_H`, `MAGE_W`, `MAGE_STEPS`,
`MAGE_CFG`, `MAGE_EDIT_STEPS`, `MAGE_GS_KEY`, `MAGE_DEVICE` (`cuda`|`mps`|`cpu`, default auto),
`MAGE_ATTN` (`flash2`|`flash4`|`sdpa`, default `flash2` on CUDA else `sdpa`).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.numpy import save_file

from _paths import fixture, hf_hub_cache

# The frozen reference lives in the crate's `_vendor/` (the established home for vendored
# third-party reference source here). Put it on the path before importing it, so the harness
# runs as `python tools/dump_mage_flow_golden.py` with no PYTHONPATH.
VENDOR_ROOT = Path(__file__).resolve().parents[1] / "_vendor"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

# ---------------------------------------------------------------------------- config

SEED = int(os.environ.get("MAGE_SEED", "42"))
PROMPT = os.environ.get(
    "MAGE_PROMPT",
    "a calico kitten sitting on a wooden windowsill beside a blue ceramic mug",
)
NEGATIVE = os.environ.get("MAGE_NEG", " ")
EDIT_INSTRUCTION = os.environ.get(
    "MAGE_EDIT_INSTRUCTION", "Replace the background with a field of sunflowers"
)
# The reference's own edit example image, vendored with the package (MIT).
EDIT_REF = os.environ.get("MAGE_EDIT_REF", str(VENDOR_ROOT / "mage_flow" / "assets" / "dog.jpg"))

# Small by default: these are boundary oracles, not benchmarks. 256x256 -> a 16x16 latent
# (256 image tokens), which still exercises packing, msrope centering and the full block stack.
HEIGHT = int(os.environ.get("MAGE_H", "256"))
WIDTH = int(os.environ.get("MAGE_W", "256"))
STEPS = int(os.environ.get("MAGE_STEPS", "4"))
CFG = float(os.environ.get("MAGE_CFG", "5.0"))
EDIT_STEPS = int(os.environ.get("MAGE_EDIT_STEPS", "4"))
GS_KEY = int(os.environ.get("MAGE_GS_KEY", "20260720"))

VL_COND_LONG_EDGE = 384  # reference default (pipeline.generate_edits)


def _auto_device() -> str:
    if override := os.environ.get("MAGE_DEVICE"):
        return override
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _auto_device()
# flash-attn is a CUDA extension. The reference ships an `sdpa` fallback for both the DiT
# (`_attn_backend._resolve_sdpa`) and the HF text encoder (`_resolve_hf_attn_impl`), which is
# what makes this oracle runnable on a Mac. Same math, one SDPA dispatch per packed sequence.
ATTN = os.environ.get("MAGE_ATTN") or ("flash2" if DEVICE == "cuda" else "sdpa")


def _snapshot_dir(env: str, repo_id: str) -> str:
    """Resolve a Mage-Flow snapshot: `$env`, else the standard HF cache, else the repo id.

    A cache can hold SEVERAL revisions of the same repo (a model-card edit alone mints a new
    snapshot whose weight symlinks point at the identical blobs). Picking one by name sort is
    arbitrary and can land on a half-downloaded revision, so: prefer whatever `refs/main`
    points at, then fall back to the newest snapshot that actually carries a `model_index.json`.
    The chosen revision is recorded in every golden's metadata (`_str_meta`).
    """
    if value := os.environ.get(env):
        return str(Path(value).expanduser())
    root = hf_hub_cache() / f"models--{repo_id.replace('/', '--')}"
    snapshots = root / "snapshots"
    if snapshots.is_dir():
        head = root / "refs" / "main"
        if head.is_file():
            candidate = snapshots / head.read_text(encoding="utf-8").strip()
            if (candidate / "model_index.json").is_file():
                return str(candidate)
        complete = [p for p in snapshots.iterdir() if (p / "model_index.json").is_file()]
        if complete:
            return str(max(complete, key=lambda p: p.stat().st_mtime))
    return repo_id  # let huggingface_hub resolve/download it


def _revision_of(path: str) -> str:
    """The HF revision a resolved snapshot path represents (`unknown` outside the cache)."""
    name = Path(path).name
    return name if len(name) == 40 and all(c in "0123456789abcdef" for c in name) else "unknown"


# ------------------------------------------------------------------------- utilities


def _f32(t: torch.Tensor) -> np.ndarray:
    """Detached float32 C-contiguous numpy view of a real reference tensor.

    `.contiguous()` is REQUIRED before `.numpy()`: `safetensors.save_file` serializes the raw
    buffer against the declared C-contiguous shape, so a strided array is SILENTLY mis-stored
    (see `dump_mochi_golden.py`, sc-11985).

    Complex input is REJECTED rather than cast. `Tensor.to(torch.float32)` on a complex tensor
    silently keeps only the real part, and Mage's msrope table (`MageFlowEmbedRope`) is
    `complex64` — a golden dumped that way would hand a port half its RoPE and still "match".
    Complex tensors go through [`_split_complex`] instead.
    """
    if t.is_complex():
        raise TypeError("complex tensor reached _f32 — use _split_complex (real part only would be stored)")
    return t.detach().to("cpu", torch.float32).contiguous().numpy()


def _split_complex(t: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """`(real, imag)` float32 halves of a complex reference tensor (the RoPE frequency table)."""
    c = t.detach().to("cpu").resolve_conj().contiguous()
    return (
        torch.view_as_real(c)[..., 0].to(torch.float32).contiguous().numpy(),
        torch.view_as_real(c)[..., 1].to(torch.float32).contiguous().numpy(),
    )


def _i64(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu", torch.int64).contiguous().numpy()


def _shapes_arr(img_shapes) -> np.ndarray:
    """Flatten the reference's `img_shapes` (list-of-list-of `(frame, h, w)`) to `[n, 3]` int32.

    The frame axis IS the msrope frame index (edit: target 0, ref_j j), so this array is the
    machine-readable answer to "what coordinates does the packed sequence carry".
    """
    flat = []
    for entry in img_shapes:
        if isinstance(entry, (list, tuple)) and entry and isinstance(entry[0], (list, tuple)):
            flat.extend(tuple(int(v) for v in s) for s in entry)
        else:
            flat.append(tuple(int(v) for v in entry))
    return np.asarray(flat, dtype=np.int32)


def _meta() -> dict[str, np.ndarray]:
    return {
        "geometry": np.array([HEIGHT, WIDTH, STEPS, EDIT_STEPS], dtype=np.int32),
        "seed": np.array([SEED], dtype=np.int64),
        "cfg": np.array([CFG], dtype=np.float32),
        "gs_key": np.array([GS_KEY], dtype=np.int64),
        # (gen, edit) system-prompt token counts dropped from the TE hidden state.
        "drop_idx": np.array([34, 64], dtype=np.int32),
        "static_shift": np.array([6.0], dtype=np.float32),
    }


_REVISIONS: dict[str, str] = {}


def _str_meta() -> dict[str, str]:
    return {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE,
        "edit_instruction": EDIT_INSTRUCTION,
        "edit_ref": os.path.basename(EDIT_REF),
        "device": DEVICE,
        "attn": ATTN,
        "reference": "microsoft/Mage @ _vendor/mage_flow (see VENDORED.md)",
        **_REVISIONS,
    }


def _write(rel_name: str, tensors: dict[str, np.ndarray]) -> None:
    out = fixture(f"tools/golden/{rel_name}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out, metadata=_str_meta())
    print(f"wrote {out}")
    for k, v in tensors.items():
        print(f"    {k:28s} {tuple(v.shape)} {v.dtype}")


def _write_png(rel_name: str, image: Image.Image) -> None:
    out = fixture(f"tools/golden/{rel_name}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"wrote {out}")


def _pil_u8(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


# ------------------------------------------------------------------ reference loading


def _load_model(repo_dir: str):
    """Load the frozen reference with the harness's attention backend selected.

    `pipeline.load_from_repo` builds a `ModelConfig` with `attn_type` defaulted to `"flash2"`,
    which is CUDA-only. Rather than fork the vendored file, rebind the `ModelConfig` name the
    pipeline module resolves to a subclass carrying the requested default — `MageFlowModel`
    reads `config.attn_type` for BOTH the DiT shim (`set_attn_backend`) and the HF text encoder
    (`_resolve_hf_attn_impl`), so this one hook covers the whole graph and the vendored source
    stays byte-identical to upstream.
    """
    from mage_flow import pipeline as ref_pipeline
    from mage_flow.models.mage_flow import ModelConfig

    class _HarnessModelConfig(ModelConfig):
        attn_type: str = ATTN

    ref_pipeline.ModelConfig = _HarnessModelConfig
    print(f"loading Mage-Flow reference from {repo_dir} (device={DEVICE}, attn={ATTN}, bf16)")
    model = ref_pipeline.load_from_repo(repo_dir, DEVICE)
    ref_pipeline.ModelConfig = ModelConfig
    return model


def _assert_not_screened(model, prompt: str, ref_pils=None) -> None:
    """Run the reference's MANDATORY content gate and hard-fail if it blocks.

    `generate_images` / `generate_edits` are FAIL-CLOSED: any screening error (including a
    backend that cannot run `.generate()`) yields `violates=True` and a **blank white refusal
    image**. Screening here first turns that into a loud failure instead of a golden full of
    255s that every parity test would then happily "match".
    """
    verdict = (
        model.txt_enc.screen_edit(prompt, ref_pils)
        if ref_pils
        else model.txt_enc.screen_text(prompt)
    )
    if verdict.violates:
        raise RuntimeError(
            f"reference content gate BLOCKED the fixed prompt {prompt!r} "
            f"({verdict.categories}: {verdict.reason}). Refusing to emit a refusal-image golden."
        )
    print(f"content gate: pass ({verdict.reason or 'ok'})")


# ------------------------------------------------------------------------ stage: noise


def dump_noise() -> None:
    """Gaussian-Shading initial-noise golden — weight-free, the cheapest gate in the set.

    `pipeline.generate_images` computes `utils.get_noise` (plain seeded `randn`) and then
    **overwrites** it with `mage_latent.encode_noise(...)` before the denoise loop, so a port
    that seeds a plain `randn` is wrong from token 0. Both tensors are dumped: the port must
    match `gs_noise` and must NOT match `plain_randn`.
    """
    from mage_flow.models.modules.mage_latent import (
        DEFAULT_GS_PAYLOAD,
        _pad_and_pos,
        _payload_to_bits,
        decode_bits,
        encode_noise,
        resolve_gs_key,
    )
    from mage_flow.models.utils import get_noise

    key = resolve_gs_key(GS_KEY)
    ch, gh, gw = 128, HEIGHT // 16, WIDTH // 16

    gs = encode_noise((ch, gh, gw), key=key, seed=SEED, device="cpu", dtype=torch.float32)
    # What the pipeline ACTUALLY feeds the denoise loop: the same tensor at bf16
    # (`pipeline.py:307-308` passes `dtype=torch.bfloat16`). Dumped separately so the e2e
    # golden's `traj_step0` first half is comparable bit-for-bit — against `gs_noise` it would
    # differ by bf16 rounding alone and read as a spurious failure.
    gs_bf16 = encode_noise((ch, gh, gw), key=key, seed=SEED, device="cpu", dtype=torch.bfloat16)
    plain = get_noise(1, ch, HEIGHT, WIDTH, torch.device("cpu"), torch.float32, SEED)
    tiny = encode_noise((8, 2, 2), key=key, seed=SEED, device="cpu", dtype=torch.float32)

    stats = decode_bits(gs, key=key)
    n_tiny = 8 * 2 * 2
    pad_tiny, pos_tiny = _pad_and_pos(n_tiny, key)

    tensors = {
        "gs_noise": _f32(gs),
        "gs_noise_bf16": _f32(gs_bf16),
        "plain_randn": _f32(plain),
        "gs_noise_tiny": _f32(tiny),
        # Key-schedule internals for the tiny case, so a port can bisect the derivation
        # (payload bits -> per-entry XOR pad + message index -> inverse-normal-CDF) instead of
        # only seeing the final tensor.
        "msg_bits": _payload_to_bits(DEFAULT_GS_PAYLOAD).astype(np.int64),
        "pad_tiny": pad_tiny.astype(np.int64),
        "pos_tiny": pos_tiny.astype(np.int64),
        "detect_raw_acc": np.array([stats["raw_acc"]], dtype=np.float64),
        "detect_msg_acc": np.array([stats["msg_acc"]], dtype=np.float64),
        "detect_z_score": np.array([stats["z_score"]], dtype=np.float64),
        "latent_shape": np.array([1, ch, gh, gw], dtype=np.int32),
        "tiny_shape": np.array([1, 8, 2, 2], dtype=np.int32),
        **_meta(),
    }
    _write("mage_flow_noise_golden.safetensors", tensors)
    print(
        f"    watermark detect: raw_acc={stats['raw_acc']:.4f} msg_acc={stats['msg_acc']:.4f} "
        f"present={stats['present']}"
    )


# -------------------------------------------------------------------------- stage: vae


def _ref_pixels(path: str, height: int, width: int, device) -> tuple[torch.Tensor, np.ndarray]:
    """Reference-preprocessed pixels `[3, H, W]` in [-1, 1] plus the resized RGB8 bytes.

    Uses the reference's own `_preprocess_ref_image` (torchvision BICUBIC + 0.5/0.5 normalize)
    so the golden pins the exact preprocessing; the u8 array lets the Rust side feed
    byte-identical pixels without reimplementing torchvision's resize.
    """
    from mage_flow.pipeline import _preprocess_ref_image

    pil = Image.open(path).convert("RGB")
    pixels = _preprocess_ref_image(pil, height, width, device)
    u8 = ((pixels.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return pixels, u8.permute(1, 2, 0).contiguous().numpy()


def dump_vae(repo_dir: str) -> None:
    """Mage-VAE encode (posterior moments at t=0) + decode goldens.

    Encode is a SINGLE forward at t=0 with a zero `z_t`: `forward_pred(zeros, zeros, x)` packs
    `mean = out[:, :128]` and `logvar = out[:, 128:].clamp(-20, 10)` (`mage_vae.py:597-606`).
    `enc_latent` is the deterministic branch (`sample_posterior=False` -> the mean); the
    pipeline's edit path instead SAMPLES `mean + exp(0.5*logvar)*randn_like` off the global RNG
    (`ModelConfig.vae_sample_posterior` defaults True), which no port can reproduce bit-wise —
    hence the mean and the logvar are both dumped so a port can gate the moments and then apply
    its own RNG.
    """
    from mage_flow.models.modules.mage_vae import MageVAE

    ckpt = os.path.join(repo_dir, "vae", "diffusion_pytorch_model.safetensors")
    vae = MageVAE(ckpt_path=ckpt, sample_posterior=False).eval().to(DEVICE).to(torch.bfloat16)

    pixels, u8 = _ref_pixels(EDIT_REF, HEIGHT, WIDTH, DEVICE)
    x = pixels.unsqueeze(0).to(DEVICE, torch.bfloat16)

    with torch.no_grad():
        mean, logvar = vae._moments(x)
        latent = vae.encode(x)  # == mean (sample_posterior=False)
        decoded = vae.decode(latent)

        # A second decode from a seeded synthetic latent isolates the decoder from the encoder
        # (a port with a broken encoder would otherwise pass the decode gate on its own error).
        gen = torch.Generator(device="cpu").manual_seed(SEED + 1)
        synth = torch.randn(
            (1, MageVAE.latent_channels, HEIGHT // 16, WIDTH // 16),
            generator=gen,
            dtype=torch.float32,
        ).to(DEVICE, torch.bfloat16)
        decoded_synth = vae.decode(synth)

    tensors = {
        "image_u8": u8,
        "pixels": _f32(x),
        "enc_mean": _f32(mean),
        "enc_logvar": _f32(logvar),
        "enc_latent": _f32(latent),
        "dec_from_latent": _f32(decoded),
        "synth_latent": _f32(synth),
        "dec_from_synth": _f32(decoded_synth),
        **_meta(),
    }
    _write("mage_flow_vae_golden.safetensors", tensors)


# --------------------------------------------------------------------------- stage: te


class _HiddenCapture:
    """Records the Qwen3-VL encoder's `last_hidden_state` — the PRE-drop conditioning tensor.

    `TextEncoder.forward` only returns the post-`drop_idx` slice, so hooking `hf_module` is the
    only way to dump the full sequence. That is what lets a port distinguish "wrong hidden
    layer / missing final RMSNorm" from "wrong drop_idx".
    """

    def __init__(self) -> None:
        self.hidden: np.ndarray | None = None
        self.handle = None

    def hook(self, _module, _args, _kwargs, output):
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is not None:
            self.hidden = _f32(hidden.squeeze(0))

    def register(self, hf_module: torch.nn.Module) -> None:
        self.handle = hf_module.register_forward_hook(self.hook, with_kwargs=True)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def dump_te(model) -> dict[str, np.ndarray]:
    """Qwen3-VL conditioning golden for BOTH templates (gen drop 34 / edit drop 64)."""
    from mage_flow.pipeline import (
        _encode_edits_packed,
        _encode_texts_packed,
        _resize_long_edge,
        _template_info,
    )

    dev = torch.device(DEVICE)
    tensors: dict[str, np.ndarray] = {}

    # --- gen (text-only) -------------------------------------------------------------
    info = _template_info("mage-flow")
    template, drop_idx = info["template"], int(info["start_idx"])
    tokenizer = model.txt_enc.tokenizer
    ids = tokenizer(
        template.format(PROMPT),
        max_length=model.txt_enc.tokenizer_max_length + drop_idx,
        truncation=True,
        return_tensors="pt",
    ).input_ids.squeeze(0)

    cap = _HiddenCapture()
    cap.register(model.txt_enc.hf_module)
    try:
        txt, vec, lens = _encode_texts_packed(model, [PROMPT, NEGATIVE], template, drop_idx, dev)
    finally:
        cap.remove()

    pos_len, neg_len = int(lens[0]), int(lens[1])
    tensors.update(
        {
            "gen_input_ids": _i64(ids),
            "gen_hidden_full": cap.hidden,  # [pos_L + neg_L, 2560], post-final-RMSNorm, pre-drop
            "gen_txt": _f32(txt[:pos_len]),
            "gen_vec": _f32(vec[0:1]),
            "gen_txt_len": np.array([pos_len], dtype=np.int32),
            "neg_txt": _f32(txt[pos_len : pos_len + neg_len]),
            "neg_vec": _f32(vec[1:2]),
            "neg_txt_len": np.array([neg_len], dtype=np.int32),
            "gen_drop_idx": np.array([drop_idx], dtype=np.int32),
        }
    )

    # --- edit (multimodal: instruction + reference image through the VL vision tower) --
    einfo = _template_info("mage-flow-edit")
    etemplate, edrop = einfo["template"], int(einfo["start_idx"])
    ref_pil = Image.open(EDIT_REF).convert("RGB")
    vl_ref = _resize_long_edge(ref_pil, VL_COND_LONG_EDGE)
    processor = model.txt_enc.processor
    vl = processor(
        text=[etemplate.format(f"Image 1: <|vision_start|><|image_pad|><|vision_end|>{EDIT_INSTRUCTION}")],
        images=[vl_ref],
        padding=True,
        return_tensors="pt",
    )

    cap = _HiddenCapture()
    cap.register(model.txt_enc.hf_module)
    try:
        etxt, evec, elens = _encode_edits_packed(
            model, [[vl_ref]], [EDIT_INSTRUCTION], etemplate, edrop, dev
        )
    finally:
        cap.remove()

    tensors.update(
        {
            "edit_input_ids": _i64(vl["input_ids"].squeeze(0)),
            "edit_pixel_values": _f32(vl["pixel_values"]),
            "edit_image_grid_thw": _i64(vl["image_grid_thw"]),
            "edit_vl_ref_u8": _pil_u8(vl_ref),
            "edit_hidden_full": cap.hidden,
            "edit_txt": _f32(etxt),
            "edit_vec": _f32(evec),
            "edit_txt_len": np.array([int(elens[0])], dtype=np.int32),
            "edit_drop_idx": np.array([edrop], dtype=np.int32),
        }
    )

    _write("mage_flow_te_golden.safetensors", {**tensors, **_meta()})
    return tensors


# ------------------------------------------------------- stage: dit_block / dit / e2e


def _flatten(prefix: str, value: object, out: dict[str, np.ndarray]) -> None:
    """Store tensors under `prefix`, indexing into tuples/lists (`prefix.0`, ...).

    Non-tensor kwargs (`attention_kwargs=None`, `img_shapes=[...]`) are skipped, so a hook can
    hand this its whole `kwargs` dict and only the tensors survive. Dtype is preserved by class:
    complex tensors split into `{prefix}_re` / `{prefix}_im` (the msrope table), integer tensors
    (`*_cu_seqlens`) stay integral, everything else upcasts to f32.
    """
    if isinstance(value, torch.Tensor):
        if value.is_complex():
            real, imag = _split_complex(value)
            out[f"{prefix}_re"], out[f"{prefix}_im"] = real, imag
        elif not value.is_floating_point():
            out[prefix] = _i64(value)
        else:
            out[prefix] = _f32(value)
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            _flatten(f"{prefix}.{i}", item, out)


class _StepCapture:
    """Captures a module's first forward (step 0) plus the `img` latent at every step.

    One hook serves three goldens: the step-0 input/output bundle (`dit` or `dit_block`), the
    early-step latent trajectory (`traj_step{0,1}`, the tight integration gate used by the
    Kolors/SD3 suites here), and — for edit — the assembled `[noisy_target, ref...]` sequence
    and its `img_shapes` frame indices.
    """

    def __init__(self, name: str, track_latents: bool = False) -> None:
        self.name = name
        self.track_latents = track_latents
        self.captured: dict[str, np.ndarray] | None = None
        self.shapes: np.ndarray | None = None
        self.latents: list[np.ndarray] = []
        self.calls = 0
        self.handle = None

    def hook(self, _module, args, kwargs, output):
        if self.captured is None:
            rec: dict[str, np.ndarray] = {}
            for i, item in enumerate(args):
                _flatten(f"{self.name}_in.arg{i}", item, rec)
            for key, item in kwargs.items():
                _flatten(f"{self.name}_in.{key}", item, rec)
            _flatten(f"{self.name}_out", output, rec)
            self.captured = rec
            if "img_shapes" in kwargs and kwargs["img_shapes"] is not None:
                self.shapes = _shapes_arr(kwargs["img_shapes"])
        if self.track_latents and self.calls < 2:
            img = kwargs.get("img", args[0] if args else None)
            if isinstance(img, torch.Tensor):
                self.latents.append(_f32(img))
        self.calls += 1

    def register(self, module: torch.nn.Module) -> None:
        self.handle = module.register_forward_hook(self.hook, with_kwargs=True)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class _DecodeCapture:
    """Wraps `pipeline._decode_one` to record the final image tokens before VAE decode.

    `generate_images` returns PIL images only, so the final LATENT — the thing a sampler-parity
    test actually wants — is otherwise unobservable without reimplementing the denoise loop.
    """

    def __init__(self, module) -> None:
        self.module = module
        self.original = module._decode_one
        self.tokens: list[np.ndarray] = []

    def __enter__(self):
        def _wrapped(model, tokens, height, width, dev):
            self.tokens.append(_f32(tokens))
            return self.original(model, tokens, height, width, dev)

        self.module._decode_one = _wrapped
        return self

    def __exit__(self, *_exc):
        self.module._decode_one = self.original
        return False


def _schedule_tensors(steps: int) -> dict[str, np.ndarray]:
    """The flow-match Euler sigma/timestep schedule the reference builds for `steps`.

    `build_scheduler` feeds `sigmas = linspace(1, 1/N, N)` into a
    `FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=6.0,
    use_dynamic_shifting=False)`, which applies the static shift `6s/(1+5s)` and appends a
    terminal 0. **Turbo is the same formula at N=4** — there is no separate distilled timestep
    table (`pipeline.py:37-50`).
    """
    from mage_flow.pipeline import build_scheduler

    scheduler = build_scheduler(steps)
    return {
        f"sigmas_{steps}": scheduler.sigmas.detach().to("cpu", torch.float32).numpy(),
        f"timesteps_{steps}": scheduler.timesteps.detach().to("cpu", torch.float32).numpy(),
    }


def dump_e2e(model, want_block: bool, want_dit: bool) -> None:
    """Full txt2img denoise golden, with the DiT-stack and DiT-block goldens captured en route."""
    from mage_flow import pipeline as ref_pipeline

    _assert_not_screened(model, PROMPT)

    dit_cap = _StepCapture("dit", track_latents=True)
    dit_cap.register(model.transformer)
    block_cap = _StepCapture("block")
    if want_block:
        block_cap.register(model.transformer.transformer_blocks[0])

    try:
        with _DecodeCapture(ref_pipeline) as dec:
            images = ref_pipeline.generate_images(
                model,
                [PROMPT],
                neg_prompts=[NEGATIVE],
                seeds=[SEED],
                steps=STEPS,
                cfg=CFG,
                heights=[HEIGHT],
                widths=[WIDTH],
                device=DEVICE,
                gs_key=GS_KEY,
            )
    finally:
        dit_cap.remove()
        block_cap.remove()

    if not dec.tokens:
        raise RuntimeError("no image was decoded — the reference refused or produced nothing")
    image = images[0]
    final_tokens = dec.tokens[-1]
    gh, gw = HEIGHT // 16, WIDTH // 16
    final_latent = final_tokens.reshape(1, gh, gw, -1).transpose(0, 3, 1, 2)

    tensors = {
        "final_tokens": final_tokens,
        "final_latent": np.ascontiguousarray(final_latent),
        "image_u8": _pil_u8(image),
        # `traj_step{i}` is the latent as the transformer sees it at sampler step i. Under the
        # default `batch_cfg=True` the image stream is DUPLICATED (cond copy then uncond copy),
        # so each is `[1, 2*gh*gw, C]` and the first half is the real latent — `traj_step0`'s
        # first half must equal `gs_noise_bf16` from the `noise` golden (the bf16 tensor, not the
        # f32 one: the pipeline feeds the denoise loop at bf16).
        **{f"traj_step{i}": lat for i, lat in enumerate(dit_cap.latents)},
        **_schedule_tensors(STEPS),
        **_schedule_tensors(4),  # the Turbo schedule, for the few-step gate
        **_schedule_tensors(30),  # the Base schedule
        **_meta(),
    }
    if dit_cap.shapes is not None:
        tensors["img_shapes"] = dit_cap.shapes
    _write("mage_flow_e2e_golden.safetensors", tensors)
    _write_png("mage_flow_e2e_golden.png", image)

    if want_dit:
        if dit_cap.captured is None:
            raise RuntimeError("transformer hook never fired — cannot emit the DiT golden")
        extra = {"img_shapes": dit_cap.shapes} if dit_cap.shapes is not None else {}
        _write(
            "mage_flow_dit_golden.safetensors",
            {**dit_cap.captured, **extra, **_meta()},
        )
    if want_block:
        if block_cap.captured is None:
            raise RuntimeError("block[0] hook never fired — cannot emit the DiT-block golden")
        _write("mage_flow_dit_block_golden.safetensors", {**block_cap.captured, **_meta()})


# ------------------------------------------------------------------------- stage: edit


def dump_edit(model) -> None:
    """Instruction-edit golden: sequence assembly + the edited image.

    The step-0 `img` capture is the load-bearing part — it is the assembled image stream, which
    the reference builds as `[noisy_target, ref_1, ..., ref_N]` (target FIRST, refs CLEAN and
    re-concatenated every step) and steps ONLY over the target slice (`pipeline.py:552-565`).
    `img_shapes` carries the msrope frame index per segment: target 0, ref_j j.
    """
    from mage_flow import pipeline as ref_pipeline

    ref_pil = Image.open(EDIT_REF).convert("RGB")
    _assert_not_screened(model, EDIT_INSTRUCTION, [ref_pil])

    dit_cap = _StepCapture("dit", track_latents=True)
    dit_cap.register(model.transformer)
    try:
        with _DecodeCapture(ref_pipeline) as dec:
            images = ref_pipeline.generate_edits(
                model,
                [EDIT_INSTRUCTION],
                [ref_pil],
                neg_prompts=[NEGATIVE],
                seeds=[SEED],
                steps=EDIT_STEPS,
                cfg=CFG,
                heights=[HEIGHT],
                widths=[WIDTH],
                device=DEVICE,
                gs_key=GS_KEY,
                vl_cond_long_edge=VL_COND_LONG_EDGE,
            )
    finally:
        dit_cap.remove()

    if not dec.tokens:
        raise RuntimeError("no image was decoded — the reference refused or produced nothing")
    image = images[0]
    final_tokens = dec.tokens[-1]
    gh, gw = HEIGHT // 16, WIDTH // 16
    final_latent = final_tokens.reshape(1, gh, gw, -1).transpose(0, 3, 1, 2)

    tensors = {
        "ref_u8": _pil_u8(ref_pil.resize((WIDTH, HEIGHT), Image.BICUBIC)),
        "final_tokens": final_tokens,
        "final_latent": np.ascontiguousarray(final_latent),
        "image_u8": _pil_u8(image),
        # step-0 assembled image stream: [noisy_target(gh*gw), ref_1(gh*gw), ...] (x2 under
        # batch_cfg, cond half first).
        **{f"seq_step{i}": lat for i, lat in enumerate(dit_cap.latents)},
        "target_tokens": np.array([gh * gw], dtype=np.int32),
        **_schedule_tensors(EDIT_STEPS),
        **_meta(),
    }
    if dit_cap.shapes is not None:
        tensors["img_shapes"] = dit_cap.shapes
    _write("mage_flow_edit_golden.safetensors", tensors)
    _write_png("mage_flow_edit_golden.png", image)


# ------------------------------------------------------------------------------ main

_STAGES = ("noise", "vae", "te", "dit_block", "dit", "e2e", "edit")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=[*_STAGES, "all"], default="all")
    args = parser.parse_args()
    stages = set(_STAGES) if args.stage == "all" else {args.stage}

    gen_repo = _snapshot_dir("MAGE_SNAPSHOT", "microsoft/Mage-Flow")
    edit_repo = _snapshot_dir("MAGE_EDIT_SNAPSHOT", "microsoft/Mage-Flow-Edit")
    _REVISIONS["gen_revision"] = _revision_of(gen_repo)
    _REVISIONS["edit_revision"] = _revision_of(edit_repo)
    print(f"gen  repo: {gen_repo}\nedit repo: {edit_repo}")

    if "noise" in stages:
        dump_noise()

    if "vae" in stages:
        dump_vae(gen_repo)

    gen_stages = stages & {"te", "dit_block", "dit", "e2e"}
    if gen_stages:
        model = _load_model(gen_repo)
        if "te" in stages:
            dump_te(model)
        if gen_stages & {"dit_block", "dit", "e2e"}:
            dump_e2e(model, want_block="dit_block" in stages, want_dit="dit" in stages)
        del model

    if "edit" in stages:
        dump_edit(_load_model(edit_repo))

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
