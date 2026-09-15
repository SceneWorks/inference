"""MiniMax-H3 audio-VAE **encode** parity fixtures (sc-17149, ref2va).

Runs the official ``diffusers.AutoencoderKLMiniMaxH3Audio`` — the class that loads the
**published** ``audio_vae/`` component — and dumps inputs, weights and outputs for the encode
half: the DAC convolutional trunk, the causal-attention ``pre_block`` projection, and the
``mean_proj`` / ``logs_proj`` posterior heads.

Why the reference is diffusers and not ``FL2VA/audio_vae``
----------------------------------------------------------

``layout.py`` Rule 3: *a fixture generated from reference modules cannot validate a
converted-checkpoint loader*. For the audio VAE the conversion is an identity
(``layout::AUDIO_VAE_IS_UNCONVERTED``), so both layouts agree — but the rule is about which
graph the golden came from, not only which names it used, and the encode half has a second
reason to prefer diffusers: **the snapshot's ``DacAudioVAE`` has no ``encode`` method at all.**
It ships ``preprocess`` and ``decode`` only (inference bundle). ``AutoencoderKLMiniMaxH3Audio``
is therefore the *only* executable reference for the encode path, and every fixture written
here records that provenance in its safetensors metadata.

The two graphs were cross-read before writing this: ``FL2VA/audio_vae/dac_audio_vae.py``'s
``Encoder``/``EncoderBlock``/``ResidualUnit``/``Snake1d`` and ``dac_attn_proj.py``'s
``AttnProjection``/``CausalAttention``/``GeGluMlp`` are reproduced by diffusers op for op,
including the head **mean-pool** (not concat) and the adaptive average pool that follows it.

Tiny-but-real geometry
----------------------

The shipped encoder is 2048-wide (``pre_block.attn.qkv.weight`` alone is [6144, 2048] = 50 MB),
so the committed fixture shrinks the width. Everything structural is preserved, and two knobs
are chosen so the fixture is *harder* than the shipped model rather than easier:

* ``encoder_rates = (2, 5)`` keeps **both** stride parities. ``Conv1d(k=2s, stride=s,
  padding=ceil(s/2))`` has a different length rule for odd ``s`` (``ceil(5/2) = 3``, not
  ``2.5``), and the shipped ``(2, 4, 4, 5, 5)`` chain only lands on ``L/800`` because of it. A
  fixture with all-even strides would not police that.
* ``num_attention_heads = 2`` with ``latent_dim = 96`` puts the attention head width at 48
  against a 32-channel output, so ``adaptive_avg_pool1d`` runs with **ragged, overlapping**
  windows (48 -> 32). The shipped model pools 256 -> 32, which is an exact 8:1 and would pass
  against a naive ``reshape(...).mean(-1)``. Both cases are additionally dumped standalone as
  ``out.pool.*`` so the port's pooling is pinned against torch directly.

``latent_dim = encoder_dim * 2**len(encoder_rates)`` (24 * 4 = 96) is preserved, because
``MiniMaxH3AudioVaeConfig::from_source_files`` *derives* it that way and rejects a checkpoint
whose ``metadata.json`` disagrees.

Requires torch + diffusers + safetensors, and the snapshot for the real-weight goldens. Run:

    MINIMAX_H3_SNAPSHOT=<snapshot-root> \
        ~/.minimax-h3-spike/venv/bin/python \
        mlx-gen-minimax-h3/tools/dump_minimax_h3_audio_vae_encode.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.numpy import save_file

# `_paths.py` lives in the shared `crates/media/mlx-gen/tools/` directory. This generator sits in
# the crate-local `tools/` dir instead (sc-17149 was implemented alongside concurrent edits to the
# shared one), so the shared helpers are imported by path rather than copied.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from _paths import fixture, hf_hub_cache  # noqa: E402

SEED = 17149

# ---------------------------------------------------------------------------------------------
# Tiny geometry — see the module docstring for why each value is what it is.
# ---------------------------------------------------------------------------------------------
ENCODER_DIM = 24
ENCODER_RATES = (2, 5)  # hop 10; keeps one ODD stride, whose padding rule differs.
LATENT_DIM = ENCODER_DIM * 2 ** len(ENCODER_RATES)  # 96 — the constructor's own derivation.
LATENT_CHANNELS = 32  # the REAL latent width, so the real latents_mean/std apply verbatim.
NUM_HEADS = 2  # head_dim 48 -> ragged 48->32 adaptive pool (the shipped model pools 256->32).
# Decoder knobs. `AutoencoderKLMiniMaxH3Audio` refuses to construct unless the decoder upsamples
# by exactly the encoder hop, so these mirror ENCODER_RATES. Nothing decoder-side is dumped.
DECODER_DIM = 8
DECODER_RATES = (5, 2)
DECODER_KERNEL_SIZES = (10, 4)

HOP = math.prod(ENCODER_RATES)
NUM_FRAMES = 32  # pre_block sequence length — long enough that the causal mask is observable.
SAMPLES = NUM_FRAMES * HOP

# The shipped model's own attention geometry, for the standalone pooling golden.
SHIPPED_HEAD_DIM = 2048 // 8

# Real-weight probe: 0.25 s of stereo at 32 kHz -> 10 latent frames per channel.
REAL_SAMPLES = 8000

# `weight_v` is drawn at this scale so its row norms are far from 1. A port that read `weight_v`
# raw, or multiplied by `weight_g` without dividing by the norm, would then be loudly wrong rather
# than plausibly wrong — `weight_norm_skipped_rel` in the metadata is the measured margin.
WEIGHT_V_SCALE = 3.0


def randomize_encode_half(model, generator) -> None:
    """Re-randomize every encode-half parameter, so no role is indistinguishable from another.

    The initialized values are degenerate in four separate ways, each of which would make a
    golden dumped as-constructed a false green:

    * ``Snake1d.alpha`` initializes to **ones**, so ``(alpha + 1e-9)^-1 sin(alpha x)^2`` is the
      same for every channel and a port that ignored ``alpha`` entirely would reproduce it.
    * ``LayerNorm`` initializes to ``weight=1, bias=0``, so ``norm1``/``norm2``/``norm3`` are
      interchangeable with each other and with no affine at all.
    * ``q_bias`` and ``v_bias`` initialize to **zeros**, so the fused-QKV bias assembly
      ``cat(q_bias, zero_k_bias, v_bias)`` is unobservable — swapping the two, or dropping the
      bias, would change nothing.
    * ``mean_proj`` and ``logs_proj`` are separate heads with identical shapes; drawn
      independently here so swapping them is detectable.

    ``zero_k_bias`` is deliberately left at zero: it is a *buffer* in the published checkpoint,
    and the reference concatenates it as-is.
    """
    def randn(shape):
        return torch.randn(tuple(shape), generator=generator)

    def rand(shape):
        return torch.rand(tuple(shape), generator=generator)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not is_encode_half(name):
                continue
            if name.endswith(".alpha"):
                # Snake1d. Bounded away from 0 (the `(alpha + 1e-9)^-1` pole) and spread per
                # channel so the activation is genuinely per-channel.
                param.copy_(rand(param.shape) * 0.8 + 0.6)
            elif name.endswith(".weight_g"):
                # A weight-normed conv's per-output-channel gain: `||w_row|| == g` exactly, so
                # `g ~ 1` keeps the trunk at unit gain through the stack.
                param.copy_(rand(param.shape) * 0.4 + 0.7)
            elif name.endswith(".weight_v"):
                # Direction only — the norm is divided out. Drawn LARGE on purpose (see
                # WEIGHT_V_SCALE).
                param.copy_(randn(param.shape) * WEIGHT_V_SCALE)
            elif ".norm" in name and name.endswith(".weight"):
                param.copy_(1.0 + randn(param.shape) * 0.2)  # LayerNorm gain
            elif name.endswith(".weight"):
                fan_in = int(np.prod(param.shape[1:])) if param.dim() > 1 else 1
                param.copy_(randn(param.shape) / max(fan_in, 1) ** 0.5)
            else:
                param.copy_(randn(param.shape) * 0.1)  # biases, incl. q_bias / v_bias


def is_encode_half(name: str) -> bool:
    return name.startswith(("encoder.", "pre_block.", "mean_proj.", "logs_proj."))


def encode_state(model) -> dict[str, np.ndarray]:
    """The encode-half tensors (parameters AND buffers) under the published names.

    ``AutoencoderKLMiniMaxH3Audio`` builds its convolutions with the legacy
    ``torch.nn.utils.weight_norm``, whose ``state_dict`` writes ``weight_g`` / ``weight_v``
    directly — the published spelling — so unlike the decode generator there is no name mapping
    to apply here. ``tests/audio_vae_encode_parity.rs`` asserts the dumped names are exactly the
    port's declared set, and the real-weight test asserts they are exactly the checkpoint's.
    """
    return {
        name: tensor.detach().to(torch.float32).numpy().copy()
        for name, tensor in model.state_dict().items()
        if is_encode_half(name)
    }


def np32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).numpy().copy()


def peak_rel(a: np.ndarray, b: np.ndarray) -> float:
    """``max|a-b| / peak|b|`` — the crate's own parity metric (`tests/common/mod.rs::rel`)."""
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def mutation_probe(model, sample: torch.Tensor, baseline: np.ndarray, mutate) -> float:
    """Apply ``mutate`` to a COPY of a tensor, re-encode, restore, and report the peak-relative move.

    These run on the torch side so the fixture can state — as measured evidence in its own
    metadata — that it is capable of catching the defect classes the port could plausibly ship.
    A committed golden that a wrong port would also reproduce is the sc-18740 failure mode.
    """
    saved: list[tuple[torch.nn.Parameter, torch.Tensor]] = []

    def stash(param: torch.nn.Parameter) -> None:
        saved.append((param, param.detach().clone()))

    with torch.no_grad():
        mutate(stash)
        moved = np32(model.encode(sample).latent_dist.mode())
        for param, original in saved:
            param.copy_(original)
    restored = np32(model.encode(sample).latent_dist.mode())
    assert np.array_equal(restored, baseline), "mutation_probe failed to restore the model"
    return peak_rel(moved, baseline)


def adaptive_avg_pool_reference(length: int, out: int) -> list[tuple[int, int]]:
    """PyTorch's adaptive-pool window boundaries, for the assertion below."""
    return [((i * length) // out, -(-((i + 1) * length) // out)) for i in range(out)]


def main() -> None:
    import diffusers
    from diffusers import AutoencoderKLMiniMaxH3Audio

    snapshot = Path(
        os.environ.get("MINIMAX_H3_SNAPSHOT")
        or next(
            iter(
                sorted(
                    (hf_hub_cache() / "models--MiniMaxAI--MiniMax-H3" / "snapshots").glob("*")
                )
            )
        )
    )

    generator = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    out: dict[str, np.ndarray] = {}
    meta: dict[str, str] = {}

    # -----------------------------------------------------------------------------------------
    # (a) adaptive_avg_pool1d, standalone, in BOTH regimes.
    #
    # `CausalAttention` mean-pools the heads away and then adaptively average-pools the remaining
    # head width down to `out_dim`. The shipped model's 256 -> 32 is an exact 8:1, which a naive
    # `reshape(..., out, L//out).mean(-1)` reproduces; the ragged case does not, because
    # PyTorch's windows then OVERLAP (`[floor(i*L/out), ceil((i+1)*L/out))`).
    # -----------------------------------------------------------------------------------------
    # `varsize` is not a geometry this model reaches; it is dumped because 48 -> 32 happens to
    # give every window the same WIDTH (2) even though the windows overlap, whereas 44 -> 32 has
    # widths of both 2 and 3. Together the three cases pin all of: the exact-divisor fast case,
    # overlapping windows, and windows of differing width.
    pool_cases = {
        "uniform": SHIPPED_HEAD_DIM,
        "ragged": LATENT_DIM // NUM_HEADS,
        "varsize": 44,
    }
    for tag, length in pool_cases.items():
        pool_x = torch.randn(2, 5, length, generator=generator)
        out[f"in.pool.{tag}"] = np32(pool_x)
        out[f"out.pool.{tag}"] = np32(F.adaptive_avg_pool1d(pool_x, LATENT_CHANNELS))
        out[f"const.pool.{tag}"] = np.asarray([length, LATENT_CHANNELS], dtype=np.int32)

    uniform = adaptive_avg_pool_reference(SHIPPED_HEAD_DIM, LATENT_CHANNELS)
    assert all(uniform[i][1] == uniform[i + 1][0] for i in range(len(uniform) - 1)), (
        "the shipped 256 -> 32 case must be a disjoint tiling"
    )
    ragged = adaptive_avg_pool_reference(LATENT_DIM // NUM_HEADS, LATENT_CHANNELS)
    assert any(ragged[i][1] > ragged[i + 1][0] for i in range(len(ragged) - 1)), (
        "the model's own pooling windows do not overlap; a reshape-and-mean port would pass"
    )
    varsize = adaptive_avg_pool_reference(44, LATENT_CHANNELS)
    assert len({end - start for start, end in varsize}) > 1, (
        "the varsize pooling case has uniform window widths after all"
    )

    # -----------------------------------------------------------------------------------------
    # (b) the tiny model.
    # -----------------------------------------------------------------------------------------
    model = AutoencoderKLMiniMaxH3Audio(
        encoder_dim=ENCODER_DIM,
        encoder_rates=ENCODER_RATES,
        latent_dim=LATENT_DIM,
        latent_channels=LATENT_CHANNELS,
        num_attention_heads=NUM_HEADS,
        decoder_dim=DECODER_DIM,
        decoder_rates=DECODER_RATES,
        decoder_kernel_sizes=DECODER_KERNEL_SIZES,
        sampling_rate=32000,
    )
    model.eval()
    randomize_encode_half(model, generator)

    assert model.hop_length == HOP, model.hop_length
    assert model.pre_block.attn.head_dim == LATENT_DIM // NUM_HEADS
    assert model.pre_block.attn.head_dim != LATENT_CHANNELS, (
        "head_dim == out_dim would make the adaptive pool an identity"
    )
    zero_k = model.pre_block.attn.zero_k_bias
    assert bool((zero_k == 0).all()), "zero_k_bias is a zero buffer in the published checkpoint"

    out.update(encode_state(model))

    # -----------------------------------------------------------------------------------------
    # (c) the encoder trunk on its own: [B, 1, samples] -> [B, latent_dim, frames].
    # -----------------------------------------------------------------------------------------
    waveform = torch.randn(2, 1, SAMPLES, generator=generator) * 0.3
    with torch.no_grad():
        trunk = model.encoder(waveform)
    out["in.encode.waveform"] = np32(waveform)
    out["out.trunk.hidden"] = np32(trunk)
    assert trunk.shape == (2, LATENT_DIM, NUM_FRAMES), trunk.shape

    # ...and the first three stages of it separately, so the whole-trunk residual can be
    # attributed. MLX evaluates f32 matmul in reduced precision on Metal, so a convolutional
    # stack accumulates a floor that has nothing to do with the port being right; walking
    # conv_in -> one residual unit -> one EncoderBlock -> the whole trunk shows that growth
    # explicitly instead of asking a reader to take a loose end-to-end bound on trust.
    with torch.no_grad():
        conv_in = model.encoder.block[0](waveform)
        out["out.stage.conv_in"] = np32(conv_in)
        out["out.stage.unit0"] = np32(model.encoder.block[1].block[0](conv_in))
        out["out.stage.block1"] = np32(model.encoder.block[1](conv_in))

    # -----------------------------------------------------------------------------------------
    # (d) `pre_block` on its own, at NLC — the causal attention, the head mean-pool, the adaptive
    #     pool, the GeGLU MLP and the two residual adds. Highest-risk piece of the encode half.
    # -----------------------------------------------------------------------------------------
    pre_x = torch.randn(2, NUM_FRAMES, LATENT_DIM, generator=generator)
    with torch.no_grad():
        out["in.pre_block.x"] = np32(pre_x)
        out["out.pre_block.y"] = np32(model.pre_block(pre_x))
        # The attention branch alone, so a wrong residual assembly cannot hide inside the sum.
        out["out.pre_block.attn"] = np32(model.pre_block.attn(model.pre_block.norm1(pre_x)))

    # -----------------------------------------------------------------------------------------
    # (e) the whole `encode`, including the right-pad to a multiple of the hop.
    # -----------------------------------------------------------------------------------------
    with torch.no_grad():
        posterior = model.encode(waveform).latent_dist
        out["out.encode.mean"] = np32(posterior.mean)
        out["out.encode.logs"] = np32(posterior.logs)
        out["out.encode.std"] = np32(posterior.std)
        assert posterior.mean.shape == (2, LATENT_CHANNELS, NUM_FRAMES), posterior.mean.shape

        # A length that is NOT a multiple of the hop: `encode` zero-pads on the RIGHT to
        # `ceil(S / hop) * hop`, so this must produce the SAME frame count as the padded input.
        ragged = waveform[..., : SAMPLES - HOP + 1].contiguous()
        out["in.encode_pad.waveform"] = np32(ragged)
        out["out.encode_pad.mean"] = np32(model.encode(ragged).latent_dist.mode())
        assert out["out.encode_pad.mean"].shape == (2, LATENT_CHANNELS, NUM_FRAMES)

    baseline = out["out.encode.mean"]

    # -----------------------------------------------------------------------------------------
    # (f) mutation probes, measured on the REFERENCE. Each is a defect this port could plausibly
    #     ship; the recorded number is how far the golden would move if it did.
    # -----------------------------------------------------------------------------------------
    attn = model.pre_block.attn
    mlp = model.pre_block.mlp

    def swap_posterior_heads(stash):
        stash(model.mean_proj.weight)
        stash(model.mean_proj.bias)
        stash(model.logs_proj.weight)
        stash(model.logs_proj.bias)
        mw, mb = model.mean_proj.weight.clone(), model.mean_proj.bias.clone()
        model.mean_proj.weight.copy_(model.logs_proj.weight)
        model.mean_proj.bias.copy_(model.logs_proj.bias)
        model.logs_proj.weight.copy_(mw)
        model.logs_proj.bias.copy_(mb)

    def interleave_qkv(stash):
        # The plausible wrong split: read the fused [3*D, D] projection as D per-head groups of
        # (q, k, v) rather than as three contiguous D-row thirds (layout.py Rule 2's hazard, in
        # the shape it would take here).
        stash(attn.qkv.weight)
        w = attn.qkv.weight.detach().clone()
        d = w.shape[1]
        regrouped = w.reshape(3, attn.num_heads, attn.head_dim, d).permute(1, 0, 2, 3)
        attn.qkv.weight.copy_(regrouped.reshape(3 * d, d))

    def swap_geglu_halves(stash):
        # `act(w0(x)) * w1(x)`: w0 is the GATE, w1 the VALUE. They are two separate tensors here
        # (so `layout::split_gate_value` does not apply) but they are shape-identical, which is
        # exactly the sc-18740 signature.
        stash(mlp.w0.weight)
        stash(mlp.w0.bias)
        stash(mlp.w1.weight)
        stash(mlp.w1.bias)
        w0w, w0b = mlp.w0.weight.clone(), mlp.w0.bias.clone()
        mlp.w0.weight.copy_(mlp.w1.weight)
        mlp.w0.bias.copy_(mlp.w1.bias)
        mlp.w1.weight.copy_(w0w)
        mlp.w1.bias.copy_(w0b)

    def skip_weight_norm(stash):
        # A port that used `weight_v` directly, without the `g / ||v||` rescale.
        conv = model.encoder.block[0]
        stash(conv.weight_g)
        conv.weight_g.copy_(torch.linalg.vector_norm(conv.weight_v, dim=(1, 2), keepdim=True))

    meta["mutation_swap_posterior_heads_rel"] = (
        f"{mutation_probe(model, waveform, baseline, swap_posterior_heads):.6e}"
    )
    meta["mutation_interleaved_qkv_rel"] = (
        f"{mutation_probe(model, waveform, baseline, interleave_qkv):.6e}"
    )
    meta["mutation_geglu_half_swap_rel"] = (
        f"{mutation_probe(model, waveform, baseline, swap_geglu_halves):.6e}"
    )
    meta["mutation_weight_norm_skipped_rel"] = (
        f"{mutation_probe(model, waveform, baseline, skip_weight_norm):.6e}"
    )

    # Causality, measured directly rather than by monkey-patching: perturb the LAST frame of the
    # `pre_block` input and confirm the earlier rows do not move.
    with torch.no_grad():
        bumped = pre_x.clone()
        # A random perturbation, NOT a constant offset: `norm1`/`norm3` are LayerNorms, which
        # mean-centre a constant shift straight back out and would make this probe inert.
        bumped[:, -1, :] += torch.randn(bumped.shape[0], LATENT_DIM, generator=generator) * 3.0
        moved = model.pre_block(bumped)
        head = peak_rel(np32(moved[:, :-1, :]), out["out.pre_block.y"][:, :-1, :])
        tail = peak_rel(np32(moved[:, -1:, :]), out["out.pre_block.y"][:, -1:, :])
    assert head < 1e-6, f"pre_block is not causal: earlier rows moved by {head:.3e}"
    assert tail > 1e-2, f"the perturbation did not reach the last row ({tail:.3e})"
    meta["causal_tail_only_rel"] = f"{tail:.6e}"

    # -----------------------------------------------------------------------------------------
    # (g) the REAL 577 MB audio VAE. Small enough to load; only the goldens are committed.
    # -----------------------------------------------------------------------------------------
    real = AutoencoderKLMiniMaxH3Audio.from_pretrained(
        snapshot / "audio_vae", torch_dtype=torch.float32
    )
    real.eval()
    assert real.hop_length == 800, real.hop_length
    assert real.pre_block.attn.head_dim == SHIPPED_HEAD_DIM, real.pre_block.attn.head_dim

    time = torch.arange(REAL_SAMPLES, dtype=torch.float64) / 32000.0
    # Two genuinely different channels, so a mono-collapsing port cannot pass. Tonal plus a
    # little noise: a pure-noise probe would sit almost entirely above the encoder's band.
    left = 0.4 * torch.sin(2 * math.pi * 220.0 * time) + 0.2 * torch.sin(
        2 * math.pi * 1310.0 * time
    )
    right = 0.35 * torch.sin(2 * math.pi * 330.0 * time + 0.7) + 0.15 * torch.sin(
        2 * math.pi * 2750.0 * time
    )
    noise = torch.randn(2, REAL_SAMPLES, generator=generator, dtype=torch.float64) * 0.02
    real_wave = (torch.stack([left, right]) + noise).unsqueeze(1).to(torch.float32)
    assert float(real_wave.abs().max()) < 1.0, "the real probe clips"

    with torch.no_grad():
        real_post = real.encode(real_wave).latent_dist
    out["real.in.waveform"] = np32(real_wave)
    out["real.out.mean"] = np32(real_post.mean)
    out["real.out.logs"] = np32(real_post.logs)
    assert real_post.mean.shape == (2, LATENT_CHANNELS, REAL_SAMPLES // 800)
    real_gap = peak_rel(out["real.out.mean"][0], out["real.out.mean"][1])
    assert real_gap > 1e-2, f"the two real channels encode near-identically ({real_gap:.3e})"

    out["const.geometry"] = np.asarray(
        [ENCODER_DIM, len(ENCODER_RATES), LATENT_DIM, LATENT_CHANNELS, NUM_HEADS, HOP],
        dtype=np.int32,
    )
    out["const.encoder_rates"] = np.asarray(ENCODER_RATES, dtype=np.int32)

    # -----------------------------------------------------------------------------------------
    # Non-degeneracy. A constant or all-zero golden is a false green.
    # -----------------------------------------------------------------------------------------
    for key, value in out.items():
        assert np.isfinite(value).all(), f"{key} has non-finite entries"
        if key.startswith(("out.", "real.out.")):
            assert float(value.std()) > 1e-4, f"{key} is ~constant (std {value.std()})"
    mean_vs_logs = peak_rel(out["out.encode.mean"], out["out.encode.logs"])
    assert mean_vs_logs > 1e-1, (
        f"mean and logs are near-identical ({mean_vs_logs:.3e}); the golden could not tell the "
        "two posterior heads apart"
    )
    trunk_rms = float(np.sqrt((out["out.trunk.hidden"] ** 2).mean()))
    assert 1e-2 < trunk_rms < 1e3, (
        f"the encoder trunk RMS is {trunk_rms:.3e}; Snake would be effectively linear (too small) "
        "or the golden would be dominated by round-off (too large)"
    )
    meta.update(
        {
            "provenance": "converted-checkpoint",
            "reference": "diffusers.AutoencoderKLMiniMaxH3Audio",
            "reference_version": diffusers.__version__,
            "snapshot": snapshot.name,
            "half": "encode",
            "story": "sc-17149",
            "seed": str(SEED),
            "trunk_rms": f"{trunk_rms:.6e}",
            "real_channel_gap_rel": f"{real_gap:.6e}",
        }
    )

    print(
        f"  encoder_dim={ENCODER_DIM} rates={ENCODER_RATES} latent_dim={LATENT_DIM} "
        f"latent_channels={LATENT_CHANNELS} heads={NUM_HEADS} head_dim={LATENT_DIM // NUM_HEADS} "
        f"hop={HOP}"
    )
    print(f"  trunk RMS {trunk_rms:.4f}; mean-vs-logs {mean_vs_logs:.3e}")
    for key in sorted(meta):
        print(f"  meta {key}: {meta[key]}")
    for key in sorted(out):
        if key.startswith(("in.", "out.", "const.", "real.")):
            print(f"  {key}: {list(out[key].shape)}")

    path = fixture("mlx-gen-minimax-h3/tests/fixtures/audio_vae_encode.safetensors")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, path, metadata=meta)
    print(f"wrote {path} ({len(out)} tensors) from {meta['reference']} {meta['reference_version']}")


if __name__ == "__main__":
    main()
