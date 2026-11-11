"""MiniMax-H3-**Image** `edit` real-weight first-step reference (SW-7092).

A three-reference edit through the official diffusers components of ``MiniMaxAI/MiniMax-H3-Image``,
written so the real-weight gates on BOTH backends (``mlx-gen-minimax-h3``
``h3_image_real::h3_image_edit_*`` and ``candle-gen-minimax-h3``
``real_weights::image_edit_*``) can assert parity on the published tensors:

1. **The presentation** (``MiniMaxH3ImageEditTextEncoderStep``, encoders.py): for each reference a
   ``"Picture i: "`` label and a vision block of ``VISION_TOKENS[i]`` ``<|image_pad|>`` rows, then
   the prompt, tokenized verbatim (no chat template, no special tokens) with the snapshot's own
   ``tokenizer/`` — ids and per-row tags. The vision-token counts are fixed small numbers rather
   than a Qwen3-VL processor run: the count is the template's input, and the tower/grid arithmetic
   is the shared Qwen3-VL preprocessor's own parity.
2. **The reference latents** (``MiniMaxH3ImageEditVaeEncoderStep``): three deterministic RGB images
   (one wider than a 256 px tile, so the tiled encode and its latent-space blend run, and one with
   off-grid pixel and latent sizes, so both crops run) are cropped to whole 16 px cells,
   ImageNet-normalized and encoded by ``AutoencoderKLMiniMaxH3Image`` (tiling on, as shipped); the
   posterior sample is ``mean + std * eps`` with ``eps`` drawn here and recorded, then rounded to
   float16, cropped to whole patches and normalized. The images are used at their own size: the
   upstream resize to a ``[1024, 2048]`` short edge is the shared ``bound_short_edge`` + Lanczos
   step, pinned by its own unit tests, and a 1024 px reference would make the DiT forward below
   too long for a CPU reference.
3. **The DiT's first step** (``MiniMaxH3ImageTransformer2DModel``): ``[text | 3 references |
   generated 512x512 image]`` built by ``build_image_packed_sequence`` (before_denoise.py,
   reproduced line for line), the reference rows ``scheduler.scale_noise(latent, tensor(0.999),
   noise)`` and patchified in front of the generated rows (``MiniMaxH3ImageEditPrepareLatentsStep``),
   the per-row timestep plan with the reference rows held at ``0.999``
   (``MiniMaxH3ImageSetTimestepsStep``), evaluated at step 0 (head 0) and step 3 (head 3). The
   context is seeded noise in place of the Qwen3-VL encode, as in the ``text2image`` producer.

Compute precision and the asserted tensors
------------------------------------------

The DiT forwards are computed exactly as the ``text2image`` DiT producer
(``dump_minimax_h3_image_dit_real.py``) computes its asserted tensors: **float32 activations over
the bf16-stored weights** (``compute_dtype = float32``), through the same ``F32Compute`` hooks that
upcast one top-level module's bf16 parameters at a time. A bf16-compute reference rounds its own
residual stream after every one of the 50 blocks, which a comparison against a different bf16
implementation cannot tell apart from that implementation's error; the edit gates assert against
float32 activations for the same reason the ``text2image`` gates do.

* ``out.step0`` — step 0 on the packed rows (``in.rows``);
* ``out.step3`` — step 3 on the same rows, an input no sampler shows the model at ``t = 0.25``
  (printed by the gates, not asserted);
* ``diag.traj.rows3`` / ``diag.traj.out3`` — the released sampler walked from ``in.rows`` through
  steps 0, 1, 2 (``x + (sigma_i - sigma_{i+1}) * v`` in float32 on the generated rows; the reference
  rows ride through unchanged, ``MiniMaxH3ImageLoopSchedulerStep``) and the step-3 velocity there:
  the asserted step 3;
* ``diag.bf16.step0`` / ``diag.bf16.traj.out3`` — the same two asserted inputs at bf16 compute (the
  checkpoint as published), the reference's own rounding, printed and never asserted;
* ``diag.step{0,3}.*`` — the reference's intermediate stages, recorded by forward hooks as in the
  ``text2image`` producer: ``temb`` (``[2, time_embed_dim]``, one row per distinct timestep in
  ascending order — the step's timestep, then the reference rows' ``0.999``), ``norm_out.shift`` /
  ``norm_out.scale`` (same rows), ``block0.adaln`` (``[6, 2 * MODALITY_NUM, hidden_size]``),
  ``hidden.block{k}`` for every ``k`` in ``diag_layers`` and ``pre_head``;
* ``diag.sweep.{f32,bf16}.block{k}`` — the per-block sweep: the residual stream after every block
  ``k`` in ``sweep_layers`` (every 5th block of the first half, block ``num_layers / 2 - 1``, then
  every block of the second half) at step 0, from BOTH step-0 forwards: the float32-activation one
  (stored float32) and the bf16-compute one (stored bfloat16, its own dtype, so losslessly). The
  gates print, per block and per row class (text, reference, generated), their error against each
  and the reference's own bf16-vs-f32 drift, so the first block where a backend departs from
  diffusers' own bf16 rounding is named rather than inferred from three samples.

Hash-pinned: the output carries ``producer_sha256``, the SHA-256 of THIS file's bytes, and the Rust
gates compare it to the committed copy. The output is not committed. It is ~0.47 GB, almost all of
it the sweep: 31 blocks x ``[1, seq_len, hidden_size]`` (462 x 5376 here) at 9.9 MB float32 plus
5.0 MB bfloat16.

Run (CPU torch, in the reviewed Mage reference environment; ~70 GB of host RAM: the bf16 DiT plus
one module's float32 copy, plus the sweep's ~0.47 GB of retained block outputs):

    MINIMAX_H3_IMAGE_SNAPSHOT=<snapshot-root> python3 tools/dump_minimax_h3_image_edit_real.py \
        --out <dir>/edit-reference.safetensors

The reference classes are loaded exactly as the ``text2image`` DiT and VAE producers load them:
installed when the installed diffusers ships them, else the pinned upstream files refused unless
their SHA-256 matches.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import os
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

SEED = 7092
REFERENCE_COMMIT = "875373ac7f5856071e345a3e394c74efccc6b9e6"
DIT_MODULE = "diffusers.models.transformers.transformer_minimax_h3_image"
DIT_MODULE_SHA256 = "0928960f80447c5e4db43d61a361ccc4c785021207325aabe65416f8f6f94e62"
VAE_MODULE = "diffusers.models.autoencoders.autoencoder_kl_minimax_h3_image"
VAE_MODULE_SHA256 = "82da645ee07b0c0443aacd64f160c14c3fe928d61d64023e0102ecbe99557ce1"
UPSTREAM = f"https://raw.githubusercontent.com/huggingface/diffusers/{REFERENCE_COMMIT}/src/"
# pdd_config.json / MiniMaxH3ImageSetTimestepsStep.sample_sigmas
SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]
STEPS = (0, 3)
TEXT_TAG, IMAGE_TAG = 1, 0
ROPE_SPATIAL_SCALE = 32
REFERENCE_NOISE_AUG = 0.999
MAX_SEQUENCE_LENGTH = 4096
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
# (width, height) of the three references, in packed order.
REFERENCE_SIZES = ((160, 96), (330, 272), (128, 128))
VISION_TOKENS = (16, 12, 20)
PROMPT = "Put the striped object from <Picture 1> on the table in <Picture 2>, lit like <Picture 3>."
TARGET_LATENT_HW = (32, 32)


def load_pinned(module_name: str, relative: str, sha256: str, stub_tensor_parallel: bool):
    """Return ``(module, source)``: installed, or the pinned upstream file refused on a digest mismatch."""
    try:
        return importlib.import_module(module_name), "installed"
    except ModuleNotFoundError:
        pass
    with urllib.request.urlopen(UPSTREAM + relative, timeout=60) as response:
        body = response.read()
    digest = hashlib.sha256(body).hexdigest()
    if digest != sha256:
        raise SystemExit(f"{UPSTREAM + relative} has sha256 {digest}, expected {sha256}")
    path = Path(tempfile.mkdtemp()) / Path(relative).name
    path.write_bytes(body)
    import diffusers.hooks
    import diffusers.models.autoencoders  # noqa: F401 — parent packages of the relative imports
    import diffusers.models.transformers  # noqa: F401

    substituted = []
    if stub_tensor_parallel:
        try:
            importlib.import_module("diffusers.hooks.tensor_parallel")
        except ModuleNotFoundError:
            stub = types.ModuleType("diffusers.hooks.tensor_parallel")

            class PackedColwiseParallel:  # noqa: D401 — metadata placeholder, never invoked
                def __init__(self, *args, **kwargs):
                    self.args, self.kwargs = args, kwargs

            stub.PackedColwiseParallel = PackedColwiseParallel
            sys.modules["diffusers.hooks.tensor_parallel"] = stub
            diffusers.hooks.tensor_parallel = stub
            substituted.append("hooks.tensor_parallel.PackedColwiseParallel")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    source = f"diffusers@{REFERENCE_COMMIT}:{digest}"
    if substituted:
        source += " placeholder:" + ",".join(substituted)
    return module, source


def reference_image(index: int, width: int, height: int) -> np.ndarray:
    """A deterministic ``(height, width, 3)`` uint8 image: gradients plus a per-reference stripe."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    r = (x / max(width - 1, 1)) * 255.0
    g = (y / max(height - 1, 1)) * 255.0
    b = 127.5 + 127.5 * np.sin((x + 2.0 * y) / (6.0 + 3.0 * index))
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 255.0).astype(np.uint8)


def presentation(tokenizer):
    """``MiniMaxH3ImageEditTextEncoderStep``'s ids and tags for ``VISION_TOKENS`` (encoders.py)."""
    token_ids, token_tags = [], []
    for index, count in enumerate(VISION_TOKENS):
        label_ids = tokenizer(f"Picture {index + 1}: ", add_special_tokens=False)["input_ids"]
        label_ids = label_ids[:MAX_SEQUENCE_LENGTH]
        vision_ids = (
            [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * count
            + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
        )
        token_ids += label_ids + vision_ids
        token_tags += [TEXT_TAG] * len(label_ids) + [IMAGE_TAG] * len(vision_ids)
    prompt_ids = tokenizer(PROMPT, add_special_tokens=False)["input_ids"][:MAX_SEQUENCE_LENGTH]
    token_ids += prompt_ids
    token_tags += [TEXT_TAG] * len(prompt_ids)
    return token_ids, token_tags


def patchify(latents: torch.Tensor, patch) -> torch.Tensor:
    """``patchify_video_latents`` (before_denoise.py) for a ``(1, C, T, H, W)`` latent."""
    pt, ph, pw = patch
    b, c, t, h, w = latents.shape
    latents = latents.reshape(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, c * pt * ph * pw).contiguous()


def spatial_grid(dim: int, patch: int, sqrt_area: float) -> np.ndarray:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // patch, endpoint=False) * ROPE_SPATIAL_SCALE


def frame_grid(h: int, w: int, ph: int, pw: int) -> torch.Tensor:
    sqrt_area = np.sqrt(h * w)
    hg = torch.from_numpy(spatial_grid(h, ph, sqrt_area))
    wg = torch.from_numpy(spatial_grid(w, pw, sqrt_area))
    return torch.stack([g.reshape(-1) for g in torch.meshgrid(hg, wg, indexing="ij")], dim=-1)


def build_image_packed_sequence(text_token_tags, reference_grids, latent_h, latent_w, patch):
    """``build_image_packed_sequence`` (before_denoise.py), line for line."""
    _, ph, pw = patch
    num_text = text_token_tags.shape[0]
    num_reference_rows = sum((h // ph) * (w // pw) for h, w in reference_grids)
    seq = num_text + num_reference_rows + (latent_h // ph) * (latent_w // pw)
    position_ids = torch.zeros(seq, 3, dtype=torch.float64)
    position_ids[:num_text, 0] = torch.arange(num_text, dtype=torch.float64)
    cursor = num_text
    rotary_time = float(num_text)
    for h, w in reference_grids:
        rows = slice(cursor, cursor + (h // ph) * (w // pw))
        cursor = rows.stop
        position_ids[rows, 0] = rotary_time
        position_ids[rows, 1:] = frame_grid(h, w, ph, pw)
        rotary_time += 1.0
    position_ids[cursor:, 0] = rotary_time
    position_ids[cursor:, 1:] = frame_grid(latent_h, latent_w, ph, pw)
    text_indices = torch.arange(num_text)
    image_indices = torch.arange(num_text, seq)
    token_tags = torch.empty(seq, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[image_indices] = IMAGE_TAG
    return position_ids, token_tags, image_indices, text_indices, num_reference_rows


class F32Compute:
    """Run each hooked module with its bf16 parameters upcast to float32, one module at a time.

    The ``text2image`` DiT producer's own hooks: the pre-hook swaps every bf16 parameter's storage
    for a float32 copy (and widens a bf16 positional input) and the forward hook puts the bf16
    storage back, so the float32 copies live only while their module runs. ``enabled`` False is the
    checkpoint as published (bf16 compute).
    """

    def __init__(self, model, modules):
        self.enabled = False
        self.saved: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        covered = set()
        for module in modules:
            module.register_forward_pre_hook(self._up)
            module.register_forward_hook(self._down)
            covered.update(id(p) for p in module.parameters())
        missed = [n for n, p in model.named_parameters() if p.dtype != torch.float32 and id(p) not in covered]
        assert not missed, f"bf16 parameters outside every float32-compute module: {missed[:8]}"

    def _up(self, module, inputs):
        if not self.enabled:
            return None
        assert not self.saved, "float32-compute modules must not nest"
        for p in module.parameters():
            if p.dtype == torch.bfloat16:
                self.saved.append((p, p.data))
                p.data = p.data.to(torch.float32)
        return tuple(x.to(torch.float32) if torch.is_tensor(x) and x.dtype == torch.bfloat16 else x for x in inputs)

    def _down(self, _module, _inputs, _output):
        for p, data in self.saved:
            p.data = data
        self.saved.clear()


class StageTaps:
    """Forward hooks recording the reference's own intermediate tensors (see the module docs)."""

    def __init__(self, model, layers: tuple[int, ...]):
        self.layers = layers
        self.armed = False
        self.seen: dict[str, torch.Tensor] = {}
        model.time_embedder.register_forward_hook(self._put("temb"))
        model.norm_out.linear.register_forward_hook(self._put("norm_out.linear"))
        model.transformer_blocks[0].adaln_proj.register_forward_hook(self._put("block0.adaln"))
        model.norm_out.register_forward_hook(self._put("pre_head"))
        for k in layers:
            model.transformer_blocks[k].register_forward_hook(self._put(f"hidden.block{k}"))

    def _put(self, key: str):
        def hook(_module, _inputs, output):
            if self.armed:
                self.seen[key] = output

        return hook

    def take(self, step: int) -> dict[str, np.ndarray]:
        seen, self.seen = self.seen, {}
        f32 = lambda t: t.detach().to(torch.float32).numpy().copy()  # noqa: E731
        shift, scale = seen["norm_out.linear"].chunk(2, dim=-1)
        out = {
            f"diag.step{step}.temb": f32(seen["temb"]),
            f"diag.step{step}.norm_out.shift": f32(shift),
            f"diag.step{step}.norm_out.scale": f32(scale),
            f"diag.step{step}.block0.adaln": f32(torch.stack(seen["block0.adaln"])),
            f"diag.step{step}.pre_head": f32(seen["pre_head"]),
        }
        for k in self.layers:
            out[f"diag.step{step}.hidden.block{k}"] = f32(seen[f"hidden.block{k}"])
        return out


def sweep_layers(num_layers: int) -> tuple[int, ...]:
    """Every 5th block of the first half, the first half's last block, every block of the second."""
    half = num_layers // 2
    return tuple(sorted(set(range(0, half, 5)) | {half - 1} | set(range(half, num_layers))))


class BlockSweep:
    """Forward hooks keeping the residual stream after every ``layers`` block, in its own dtype.

    ``label`` names the forward being recorded (``None`` records nothing). An output is kept as the
    block returned it — float32 under ``F32Compute``, bfloat16 without it — so the bf16 sweep is
    stored losslessly. A block returns a fresh tensor (no later in-place op touches it), so keeping
    the reference costs one ``[1, seq_len, hidden_size]`` tensor per block and no copy.
    """

    def __init__(self, model, layers: tuple[int, ...]):
        self.layers = layers
        self.label: str | None = None
        self.seen: dict[str, torch.Tensor] = {}
        for k in layers:
            model.transformer_blocks[k].register_forward_hook(self._put(k))

    def _put(self, k: int):
        def hook(_module, _inputs, output):
            if self.label is not None:
                self.seen[f"diag.sweep.{self.label}.block{k}"] = output.detach().contiguous()

        return hook


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output .safetensors path (keep it OUTSIDE the repo)")
    args = ap.parse_args()

    import diffusers
    from transformers import Qwen2TokenizerFast

    dit_module, dit_source = load_pinned(
        DIT_MODULE, "diffusers/models/transformers/transformer_minimax_h3_image.py", DIT_MODULE_SHA256, True
    )
    vae_module, vae_source = load_pinned(
        VAE_MODULE, "diffusers/models/autoencoders/autoencoder_kl_minimax_h3_image.py", VAE_MODULE_SHA256, False
    )
    snapshot = Path(os.environ["MINIMAX_H3_IMAGE_SNAPSHOT"])
    producer_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    torch.set_num_threads(os.cpu_count() or 1)
    generator = torch.Generator().manual_seed(SEED)
    tensors = {}

    # --- 1. presentation -------------------------------------------------------------------------
    tokenizer = Qwen2TokenizerFast.from_pretrained(snapshot / "tokenizer")
    token_ids, token_tags = presentation(tokenizer)
    tensors["in.token_ids"] = np.asarray(token_ids, dtype=np.float32)
    tensors["in.token_tags"] = np.asarray(token_tags, dtype=np.float32)
    assert all(float(i) == i for i in token_ids), "ids must be exact in float32"

    # --- 2. reference latents --------------------------------------------------------------------
    vae = vae_module.AutoencoderKLMiniMaxH3Image.from_pretrained(snapshot / "vae_diffusers", torch_dtype=torch.float32)
    vae.eval()
    assert vae.use_tiling and (vae.tile_sample_min_height, vae.tile_sample_min_overlap_height) == (256, 64)
    ratio = vae.spatial_compression_ratio
    pixel_mean = torch.tensor(PIXEL_MEAN).view(1, -1, 1, 1)
    pixel_std = torch.tensor(PIXEL_STD).view(1, -1, 1, 1)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1)
    patch = (1, 2, 2)
    reference_latents = []
    for index, (width, height) in enumerate(REFERENCE_SIZES):
        image = reference_image(index, width, height)
        pixels = torch.from_numpy(image).permute(2, 0, 1)[None]
        pixels = pixels[..., : pixels.shape[-2] // ratio * ratio, : pixels.shape[-1] // ratio * ratio]
        pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
        with torch.no_grad():
            posterior = vae.encode(pixels, return_dict=False)[0]
        eps = torch.randn(posterior.mean.shape, generator=generator, dtype=torch.float32)
        # `DiagonalGaussianDistribution.sample`: mean + std * randn, with the draw recorded.
        sampled = (posterior.mean + posterior.std * eps).to(torch.float16).float()
        sampled = sampled[..., : sampled.shape[-2] // patch[1] * patch[1], : sampled.shape[-1] // patch[2] * patch[2]]
        latent = (sampled - latents_mean) / latents_std
        reference_latents.append(latent)
        tensors[f"ref{index}.image"] = image.astype(np.float32)
        tensors[f"ref{index}.mean"] = posterior.mean.numpy().copy()
        tensors[f"ref{index}.posterior_noise"] = eps.numpy().copy()
        tensors[f"ref{index}.latent"] = latent.numpy().copy()
        print(f"reference {index}: {width}x{height} px -> latent {tuple(latent.shape)}")
    del vae

    # --- 3. the DiT's first step -----------------------------------------------------------------
    model = dit_module.MiniMaxH3ImageTransformer2DModel.from_pretrained(
        snapshot / "transformer_diffusers", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    cfg = model.config
    assert tuple(cfg.patch_size) == patch, cfg.patch_size
    assert cfg.num_pdd_heads == len(SIGMAS) - 1, cfg.num_pdd_heads
    assert model.proj_in.weight.dtype == torch.float32, "proj_in must keep its published float32"
    assert model.pdd_proj_out[0].weight.dtype == torch.float32, "the heads must keep float32"
    context = torch.randn(1, len(token_ids), cfg.text_dim, generator=generator, dtype=torch.float32).to(torch.bfloat16)
    target = torch.randn(1, cfg.in_channels, 1, *TARGET_LATENT_HW, generator=generator, dtype=torch.float32)
    reference_noise = [torch.randn(lat.shape, generator=generator, dtype=torch.float32) for lat in reference_latents]
    condition_rows = []
    for latent, noise in zip(reference_latents, reference_noise):
        # `scheduler.scale_noise(latents, torch.tensor(0.999), noise)`: t*x + (1 - t)*n in float32.
        t = torch.tensor(REFERENCE_NOISE_AUG)
        noised = t * latent + (1.0 - t) * noise
        condition_rows.append(patchify(noised[:, :, None], patch))
    condition_rows = torch.cat(condition_rows)
    image_rows = torch.cat([condition_rows, patchify(target, patch)])[None]
    grids = [tuple(l.shape[-2:]) for l in reference_latents]
    position_ids, tags, image_indices, text_indices, num_reference_rows = build_image_packed_sequence(
        torch.tensor(token_tags), grids, *TARGET_LATENT_HW, patch
    )
    assert condition_rows.shape[0] == num_reference_rows
    timesteps = torch.tensor([1.0 - s for s in SIGMAS[:-1]], dtype=torch.float32)
    # The released sampler's grid: sigma recovered from each float32 timestep, terminal 0.
    sigmas = torch.cat([1.0 - timesteps, torch.zeros(1)])
    seq = position_ids.shape[0]
    diag_layers = (0, cfg.num_layers // 2 - 1, cfg.num_layers - 1)
    taps = StageTaps(model, diag_layers)
    sweep = BlockSweep(model, sweep_layers(cfg.num_layers))
    # Every top-level module holding bf16 parameters; `proj_in`, `time_embedder` and the heads are
    # float32 in the checkpoint already. The coverage assert inside refuses a missed one.
    f32 = F32Compute(
        model,
        [model.context_embedder, model.token_refiner, *model.transformer_blocks, model.norm_out],
    )

    def forward(rows: torch.Tensor, step: int) -> torch.Tensor:
        row_timesteps = torch.full((seq,), float(timesteps[step]), dtype=torch.float32)
        row_timesteps[image_indices[:num_reference_rows]] = REFERENCE_NOISE_AUG
        unique, inverse = torch.unique(row_timesteps, sorted=True, return_inverse=True)
        assert unique.numel() == 2 and unique[1].item() == torch.tensor(REFERENCE_NOISE_AUG).item()
        out = model(
            hidden_states=rows,
            encoder_hidden_states=context,
            timestep=unique,
            timestep_indices=inverse,
            token_tags=tags,
            position_ids=position_ids.to(torch.float32),
            image_indices=image_indices,
            text_indices=text_indices,
            pdd_step_index=step,
            return_dict=False,
        )[0]
        print(
            f"step {step} (t={timesteps[step].item():.6f}, head {step}, float32 compute {f32.enabled}): "
            f"{tuple(out.shape)} std {out.float().std().item():.6f} L2 {out.float().flatten().norm().item():.6f}"
        )
        return out.to(torch.float32)

    outputs = {}
    diag = {}
    with torch.no_grad():
        f32.enabled = True
        for step in STEPS:
            taps.armed = True
            sweep.label = "f32" if step == 0 else None
            outputs[step] = forward(image_rows, step)
            taps.armed = False
            sweep.label = None
            assert taps.seen["hidden.block0"].dtype == torch.float32, "the block stack must run in float32"
            diag.update(taps.take(step))
        # On-trajectory step 3: the sampler's own x_3, reached from the same rows. Only the generated
        # rows move; the reference rows ride through every step unchanged.
        rows = image_rows
        velocity = outputs[0]
        for step in range(1, 4):
            generated = rows[:, num_reference_rows:] + (sigmas[step - 1] - sigmas[step]) * velocity[:, num_reference_rows:]
            rows = torch.cat([image_rows[:, :num_reference_rows], generated], dim=1)
            velocity = forward(rows, step)
        diag["diag.traj.rows3"] = rows.numpy().copy()
        diag["diag.traj.out3"] = velocity.numpy().copy()
        # The reference's own rounding: the same two asserted inputs at bf16 compute.
        f32.enabled = False
        sweep.label = "bf16"
        diag["diag.bf16.step0"] = forward(image_rows, 0).numpy().copy()
        sweep.label = None
        diag["diag.bf16.traj.out3"] = forward(rows, 3).numpy().copy()

    tensors["in.context"] = context.to(torch.float32).numpy().copy()
    tensors["in.target_rows"] = patchify(target, patch)[None].numpy().copy()
    for index, noise in enumerate(reference_noise):
        tensors[f"ref{index}.aug_noise"] = noise.numpy().copy()
    tensors["in.rows"] = image_rows.numpy().copy()
    for step, out in outputs.items():
        tensors[f"out.step{step}"] = out.numpy().copy()
    tensors.update(diag)
    for key, value in sweep.seen.items():
        want = torch.bfloat16 if ".bf16." in key else torch.float32
        assert value.dtype == want, f"{key} is {value.dtype}, expected {want}"
    assert len(sweep.seen) == 2 * len(sweep.layers), sorted(sweep.seen)
    meta = {
        "provenance": "converted-checkpoint",
        "reference": "diffusers.MiniMaxH3ImageTransformer2DModel+AutoencoderKLMiniMaxH3Image",
        "reference_version": diffusers.__version__,
        "reference_module": f"{dit_source} {vae_source}",
        "torch": torch.__version__,
        "weight_dtype": "bfloat16 (proj_in, time_embedder, pdd_proj_out float32)",
        "compute_dtype": "float32",
        "compute_precision": "float32 activations over bf16-stored weights, upcast per top-level module on forward",
        "bf16_diagnostics": "diag.bf16.step0, diag.bf16.traj.out3",
        "snapshot": snapshot.name,
        "seed": str(SEED),
        "prompt": PROMPT,
        "vision_tokens": ",".join(str(n) for n in VISION_TOKENS),
        "num_references": str(len(REFERENCE_SIZES)),
        "num_reference_rows": str(num_reference_rows),
        "latent_hw": f"{TARGET_LATENT_HW[0]}x{TARGET_LATENT_HW[1]}",
        "steps": ",".join(str(s) for s in STEPS),
        "timesteps": ",".join(f"{timesteps[s].item():.9g}" for s in STEPS),
        "diag_layers": ",".join(str(k) for k in diag_layers),
        "sweep_layers": ",".join(str(k) for k in sweep.layers),
        "sweep_step": "0",
        "sweep_dtypes": "diag.sweep.f32.* float32, diag.sweep.bf16.* bfloat16",
        "producer": "crates/media/mlx-gen/tools/dump_minimax_h3_image_edit_real.py",
        "producer_sha256": producer_sha256,
        "story": "SW-7092",
    }
    out_path = Path(args.out).expanduser()
    # `safetensors.torch` rather than `.numpy`: numpy has no bfloat16 for the bf16 sweep.
    payload = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in tensors.items()}
    payload.update(sweep.seen)
    save_file(payload, str(out_path), metadata=meta)
    print(f"wrote {out_path} from {meta['reference']} {meta['reference_version']} @ {snapshot.name}")
    print(f"producer_sha256 {producer_sha256}")


if __name__ == "__main__":
    main()
