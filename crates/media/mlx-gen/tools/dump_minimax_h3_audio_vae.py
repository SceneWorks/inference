"""MiniMax-H3 audio-VAE decode parity fixtures (sc-17141).

Runs the **Apache-2.0 reference implementation shipped inside the MiniMax-H3 snapshot**
(``FL2VA/audio_vae/*.py`` — ``dac_audio_vae.py``'s ``DacAudioVAE``, ``dac_bigvgan.py``'s
``BigVGAN``/``AMPBlock1``, ``dac_activations.py``'s ``SnakeBeta`` and the
``dac_alias_free_*`` Kaiser-sinc resamplers) and dumps inputs, weights and outputs. The
Rust/MLX port is dimension-parametric, so it runs the same tiny config and must reproduce
these tensors.

This is a **DAC-lineage encoder + BigVGAN decoder**, not an LTX-style audio VAE. Only the
decode half is ported: ``DacAudioVAE.decode(z) = decoder(dec_in_proj(z))``.

Why tiny-but-real: the shipped decoder is 1024-wide with a 2048-channel ``conv_pre``
(605 MB) — too large to commit. Every *structural* knob is preserved, and the ones that
drive the arithmetic are the REAL ones:

    sample_rate 32000  ->  upsample_rates        [5,5,2,2,2,2,2]  (hop 800 => 40 Hz tokens)
                           upsample_kernel_sizes [9,9,4,4,4,4,4]
                           resblock_kernel_sizes [3,7,11]
                           resblock_dilations    [[1,3,5]] * 3
                           activation            "snakebeta"
                           snake_logscale        True
                           use_tanh_at_final     False   (=> clamp, not tanh)
                           use_bias_at_final     False   (=> conv_post has no bias)

Only the *width* is shrunk (``decoder_dim`` 1024 -> 128, ``latent_dim`` 2048 -> 64).
``vae_latent_channels`` stays 32 so the real 32-entry ``latents_mean``/``latents_std``
de-normalization is exercised verbatim, and all 7 upsample stages / 21 AMP blocks / 127
anti-aliased activations are present, so the fixture's decode really is 800x.

**The last five knobs above appear in NO config file and leave NO tensor behind.** They are
selected by the ``if sample_rate == 32000`` branch inside ``DacAudioVAE.__init__``. A port
that took BigVGAN's upstream defaults would get ``use_tanh_at_final=True`` (tanh instead of
clamp) and ``snake_logscale=False`` (raw alpha instead of ``exp(alpha)``) and would look
complete against the checkpoint while being silently wrong — the audio analogue of the
sc-17140 ``qk_norm_type`` finding.

Weights are emitted in the published checkpoint's naming. The reference builds its convs with
``torch.nn.utils.parametrizations.weight_norm``, whose ``state_dict`` writes
``parametrizations.weight.original0/original1``; the published checkpoint uses the older
``weight_g``/``weight_v`` spelling (torch's compat load-hook bridges the two). This script
maps back to ``weight_g``/``weight_v`` so the Rust loader consumes exactly the published
names, and ``tests/real_weights.rs`` proves the mapped name set matches the real checkpoint.

Requires torch + safetensors. Run:

    MINIMAX_H3_SNAPSHOT=<snapshot-root> python3 tools/dump_minimax_h3_audio_vae.py
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

from _paths import fixture, hf_hub_cache

SEED = 17141

# Tiny geometry. `sample_rate` MUST stay 32000 — it is what selects the production BigVGAN
# branch, so shrinking it would dump a golden for a model this port never runs.
SAMPLE_RATE = 32000
DECODER_DIM = 128  # 128 // 2**7 == 1: the smallest width that survives all 7 halvings.
# BigVGAN `num_mels` == `conv_pre` in-channels (2048 in the shipped model). Must stay a multiple
# of `vae_latent_channels`: the encode-half `AttnProjection` is built unconditionally and asserts
# the two divide, even though `decode` never touches it.
LATENT_DIM = 64
VAE_LATENT_CHANNELS = 32  # the REAL latent width, so latents_mean/std apply verbatim.
DECODER_RATES = [5, 5, 2, 2, 2, 2, 2]  # metadata.json kwargs.decoder_rates
ENCODER_RATES = [2, 2]  # encode half is not ported; kept tiny so construction is cheap.
ENCODER_DIM = 8

# Conv init gain. Near-unit, so the activations stay large enough through the 7-stage stack that
# `sin^2(alpha*x)` is genuinely nonlinear (measured: perturbing one `act.alpha` moves the decode by
# 26%), but not so large that the reduced-precision matmul noise MLX introduces per convolution
# (~1e-3 relative, and unavoidable on Metal) is amplified past a useful parity bound. Measured
# end-to-end residual at this setting: 4.4e-3, against a 2e-2 gate.
WEIGHT_SCALE = 0.9
# Where the decoded waveform should peak: loud enough that `clamp` and `tanh` visibly disagree,
# quiet enough that the clamp is not what the golden is measuring.
TARGET_PEAK = 0.5

# The production per-channel de-normalization statistics, verbatim from
# `FL2VA/audio_vae/config.json` (identical in the repackaged root `audio_vae/config.json`).
LATENTS_MEAN = [
    -0.020211687488382354, 0.3876466479950502, -0.04398279799186767, -0.28591514936373,
    0.08179686214561671, -0.35782641352446604, 0.040623809960919084, -0.01552534501956604,
    -0.223362481667332, 0.1821006842509091, 0.2941778783780663, -0.07901167601970885,
    -0.056815072777201, -0.3699028221860095, -0.31616315591624855, 0.5905951377425391,
    -0.052139568068853864, 0.013673160263486295, -0.03691647864630577, 0.09732660653298163,
    -0.3394662328788498, -0.30685677538541667, -0.24504598907458763, -0.034698524462007344,
    0.02868032184767538, -0.21217779266454084, -0.1678263169941987, 0.3221287889040614,
    -0.1223055851554907, 0.4356604928128464, -0.0502599202236253, 0.3979258376211797,
]
LATENTS_STD = [
    1.6895524230479284, 2.76263727217653, 1.7945344281264435, 1.6801681847309828,
    1.6390226546605453, 2.7788298348882177, 1.7659090095747236, 1.6199757612137327,
    2.6336525640336896, 1.8539356672817833, 2.5056497896915633, 1.811019237886178,
    1.9579657790720237, 1.6685498243529284, 1.4922469314453364, 3.298670198067373,
    1.9491804496832168, 1.8720003270431442, 1.8334080103291832, 1.6488070416529093,
    1.6176957696319716, 1.9131449234774398, 1.5695245398428617, 1.6943659940415912,
    1.8318420762504692, 1.5540637421583379, 1.9344930328968526, 1.599198216109855,
    1.718045989838149, 1.6307219190837705, 1.8661226051202384, 1.5613768203168363,
]


def load_reference_package(snapshot: Path):
    """Import the snapshot's ``FL2VA/audio_vae`` bundle as a package.

    The bundle has no ``__init__.py`` but uses relative imports, so synthesize a package
    whose ``__path__`` points at it rather than copying files around.
    """
    audio_vae = snapshot / "FL2VA" / "audio_vae"
    if not (audio_vae / "dac_audio_vae.py").is_file():
        raise SystemExit(f"reference bundle not found under {audio_vae}")
    pkg = types.ModuleType("mmh3_audio_vae")
    pkg.__path__ = [str(audio_vae)]
    sys.modules["mmh3_audio_vae"] = pkg
    from mmh3_audio_vae import dac_activations, dac_alias_free_act  # noqa: E402
    from mmh3_audio_vae import dac_alias_free_filter, dac_alias_free_resample  # noqa: E402
    from mmh3_audio_vae import dac_audio_vae, dac_bigvgan  # noqa: E402

    return types.SimpleNamespace(
        DacAudioVAE=dac_audio_vae.DacAudioVAE,
        AttrDict=dac_bigvgan.AttrDict,
        AMPBlock1=dac_bigvgan.AMPBlock1,
        SnakeBeta=dac_activations.SnakeBeta,
        Activation1d=dac_alias_free_act.Activation1d,
        UpSample1d=dac_alias_free_resample.UpSample1d,
        DownSample1d=dac_alias_free_resample.DownSample1d,
        kaiser_sinc_filter1d=dac_alias_free_filter.kaiser_sinc_filter1d,
    )


# `parametrizations.weight_norm` state_dict names -> the published checkpoint's names.
PARAMETRIZED_TO_PUBLISHED = {
    "parametrizations.weight.original0": "weight_g",
    "parametrizations.weight.original1": "weight_v",
}


def to_published_name(name: str) -> str:
    for src, dst in PARAMETRIZED_TO_PUBLISHED.items():
        if name.endswith("." + src):
            return name[: -len(src)] + dst
    return name


def decode_state_published(model) -> dict[str, np.ndarray]:
    """The decode-half tensors (params AND buffers) under the published names.

    `decode` reads exactly `dec_in_proj.*` and `decoder.*`; the encoder, `mean_proj`,
    `logs_proj` and `pre_block` belong to the encode half and are deliberately absent.
    """
    out: dict[str, np.ndarray] = {}
    for name, tensor in model.state_dict().items():
        if not (name.startswith("decoder.") or name.startswith("dec_in_proj.")):
            continue
        out[to_published_name(name)] = tensor.detach().to(torch.float32).numpy().copy()
    return out


def randomize(model, generator, scale: float):
    """Re-randomize the decode-half parameters.

    Three reasons, all false-green guards:

    * ``SnakeBeta`` initializes ``alpha``/``beta`` to ONES, and with ``alpha_logscale=True``
      the forward exponentiates them — so a golden dumped as-shipped-initialized would pass
      against a port that ignored ``exp`` for `beta` but not `alpha`, and could not
      distinguish alpha from beta at all. Randomizing them separates every role.
    * ``init_weights`` draws conv weights from ``N(0, 0.01)``, which after 7 upsample stages
      leaves the output near zero — a near-constant golden proves nothing.
    * The convs are drawn at **unit gain** (``1/sqrt(fan_in)``) rather than attenuated, so the
      activations stay O(1) through the stack. That matters more than it looks:
      ``sin²(αx)/β`` is quadratic in `x` near zero, so a decoder whose internals sit at ~1e-3
      is effectively LINEAR and its golden would be reproduced by a port with no periodic
      activation at all. ``main`` measures the pre-``conv_post`` RMS and refuses to write a
      fixture whose internals have collapsed.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not (name.startswith("decoder.") or name.startswith("dec_in_proj.")):
                continue
            if name.endswith(".act.alpha") or name.endswith(".act.beta"):
                # log-scale: keep exp(.) in a sane band so the periodic term stays observable.
                param.copy_(torch.randn(param.shape, generator=generator) * 0.4)
            elif name.endswith(".bias"):
                param.copy_(torch.randn(param.shape, generator=generator) * 0.05)
            else:
                fan_in = int(np.prod(param.shape[1:])) if param.dim() > 1 else 1
                param.copy_(
                    torch.randn(param.shape, generator=generator) / max(fan_in, 1) ** 0.5 * scale
                )


def calibrate_output_level(model, probe: torch.Tensor, target: float) -> float:
    """Scale ``conv_post`` so the decoded waveform peaks near ``target``.

    ``use_tanh_at_final = False`` means the decoder ends in ``clamp(-1, 1)``. Two opposite
    failure modes make a golden worthless:

    * saturated everywhere — any port that also saturates reproduces it;
    * a whisper at ~1e-3 — ``clamp`` and ``tanh`` agree to ~1e-6 there, so the golden cannot
      tell the shipped ``use_tanh_at_final=False`` from BigVGAN's ``True`` default.

    ``conv_post`` is the last (linear) op before the bound, and its weight-norm ``g`` scales it
    exactly, so one probe decode is enough to land the level. Returns the gain applied.
    """
    with torch.no_grad():
        peak = float(model.decode(probe).abs().max())
        if not (0.0 < peak < 1.0):
            raise SystemExit(
                f"probe decode peaked at {peak}; it is already clamped, so the level cannot be "
                "calibrated linearly — lower the randomization scale"
            )
        gain = target / peak
        g = model.decoder.conv_post.parametrizations.weight.original0
        g.copy_(g * gain)
        return gain


def np32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).numpy().copy()


def main() -> None:
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
    ref = load_reference_package(snapshot)

    generator = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)

    out: dict[str, np.ndarray] = {}

    # -----------------------------------------------------------------------------------
    # (a) kaiser_sinc_filter1d — the resampler's filter derivation, on its own.
    #
    # The published checkpoint STORES these filters as buffers, so the Rust loader reads
    # them; this pins the derivation independently so the port can also build them (and so a
    # checkpoint whose filters were computed differently is detectable). The four cases walk
    # the whole branch table: the shipped (ratio 2, even kernel) case, a wider ratio, the ODD
    # kernel branch (`time = arange(k) - half`, vs `arange(-h,h) + 0.5`), and a low-`A` case
    # that takes the `elif A >= 21` beta branch instead of `A > 50`.
    # -----------------------------------------------------------------------------------
    kaiser_cases = {
        # ratio 2 with the shipped kernel_size 12 — exactly what every Activation1d uses.
        "r2k12": (0.5 / 2, 0.6 / 2, 12),
        # ratio 4, the default `int(6*ratio//2)*2 = 24` kernel.
        "r4k24": (0.5 / 4, 0.6 / 4, 24),
        # odd kernel size -> the `time = arange(kernel_size) - half_size` branch.
        "odd11": (0.25, 0.3, 11),
        # small kernel -> A = 2.285*(2-1)*pi*1.2 + 7.95 = 16.56 < 21 -> beta = 0.
        "beta0": (0.25, 0.3, 4),
    }
    for tag, (cutoff, half_width, ksize) in kaiser_cases.items():
        f = ref.kaiser_sinc_filter1d(cutoff, half_width, ksize)
        out[f"out.kaiser.{tag}"] = np32(f)
        out[f"const.kaiser.{tag}"] = np.asarray(
            [cutoff, half_width, float(ksize)], dtype=np.float32
        )

    # -----------------------------------------------------------------------------------
    # (b) SnakeBeta alone, in BOTH log-scale modes.
    #
    # The shipped model uses `snake_logscale=True`, which appears in no config file. Dumping
    # the False variant too lets the Rust side assert the port does NOT reproduce it — proof
    # that the flag is genuinely applied rather than incidentally matching.
    # -----------------------------------------------------------------------------------
    snake_c, snake_t = 5, 17
    snake_x = torch.randn(2, snake_c, snake_t, generator=generator)
    snake_alpha = torch.randn(snake_c, generator=generator) * 0.7
    snake_beta = torch.randn(snake_c, generator=generator) * 0.7
    out["in.snake.x"] = np32(snake_x)
    out["in.snake.alpha"] = np32(snake_alpha)
    out["in.snake.beta"] = np32(snake_beta)
    for tag, logscale in (("log", True), ("linear", False)):
        act = ref.SnakeBeta(snake_c, alpha_logscale=logscale)
        with torch.no_grad():
            act.alpha.copy_(snake_alpha)
            act.beta.copy_(snake_beta)
            out[f"out.snake.{tag}"] = np32(act(snake_x))

    # -----------------------------------------------------------------------------------
    # (c) the alias-free resamplers, each on its own, then the composed Activation1d.
    # -----------------------------------------------------------------------------------
    rs_c, rs_t = 3, 23
    rs_x = torch.randn(2, rs_c, rs_t, generator=generator)
    out["in.resample.x"] = np32(rs_x)
    up = ref.UpSample1d(ratio=2, kernel_size=12)
    down = ref.DownSample1d(ratio=2, kernel_size=12)
    with torch.no_grad():
        out["out.resample.up"] = np32(up(rs_x))
        out["out.resample.down"] = np32(down(rs_x))
        # The round trip the anti-aliased activation actually performs.
        out["out.resample.up_down"] = np32(down(up(rs_x)))
    out["const.resample.up_filter"] = np32(up.filter)
    out["const.resample.down_filter"] = np32(down.lowpass.filter)

    act1d_c, act1d_t = 4, 19
    act1d_x = torch.randn(2, act1d_c, act1d_t, generator=generator)
    act1d_inner = ref.SnakeBeta(act1d_c, alpha_logscale=True)
    with torch.no_grad():
        act1d_inner.alpha.copy_(torch.randn(act1d_c, generator=generator) * 0.5)
        act1d_inner.beta.copy_(torch.randn(act1d_c, generator=generator) * 0.5)
    act1d = ref.Activation1d(activation=act1d_inner)
    with torch.no_grad():
        out["in.act1d.x"] = np32(act1d_x)
        out["in.act1d.alpha"] = np32(act1d_inner.alpha)
        out["in.act1d.beta"] = np32(act1d_inner.beta)
        out["out.act1d.y"] = np32(act1d(act1d_x))

    # -----------------------------------------------------------------------------------
    # (d) one AMPBlock1 — the residual/dilation/activation-pairing arithmetic in isolation.
    #     `acts1 = activations[::2]`, `acts2 = activations[1::2]`: a port that paired them
    #     0,1,2 / 3,4,5 instead would still load every tensor.
    # -----------------------------------------------------------------------------------
    amp_h = ref.AttrDict(snake_logscale=True)
    amp_c, amp_t = 6, 29
    amp = ref.AMPBlock1(amp_h, amp_c, kernel_size=7, dilation=(1, 3, 5), activation="snakebeta")
    with torch.no_grad():
        for name, param in amp.named_parameters():
            if name.endswith(".alpha") or name.endswith(".beta"):
                param.copy_(torch.randn(param.shape, generator=generator) * 0.4)
            elif name.endswith(".bias"):
                param.copy_(torch.randn(param.shape, generator=generator) * 0.05)
            else:
                fan_in = int(np.prod(param.shape[1:])) if param.dim() > 1 else 1
                param.copy_(
                    torch.randn(param.shape, generator=generator) / max(fan_in, 1) ** 0.5 * 0.8
                )
        amp_x = torch.randn(1, amp_c, amp_t, generator=generator)
        out["in.amp.x"] = np32(amp_x)
        out["out.amp.y"] = np32(amp(amp_x))
    for name, tensor in amp.state_dict().items():
        out["amp." + to_published_name(name)] = tensor.detach().to(torch.float32).numpy().copy()

    # -----------------------------------------------------------------------------------
    # (e) the whole model: BigVGAN forward and `DacAudioVAE.decode`.
    # -----------------------------------------------------------------------------------
    model = ref.DacAudioVAE(
        encoder_dim=ENCODER_DIM,
        encoder_rates=ENCODER_RATES,
        latent_dim=LATENT_DIM,
        decoder_dim=DECODER_DIM,
        decoder_rates=DECODER_RATES,
        sample_rate=SAMPLE_RATE,
        vae_latent_channels=VAE_LATENT_CHANNELS,
        attn_proj=True,
        decoder_type="bigvgan",
    )
    model.eval()
    randomize(model, generator, WEIGHT_SCALE)

    # The knobs that come from the `sample_rate` branch and appear in no config file.
    h = model.decoder.h
    assert h.upsample_rates == DECODER_RATES, h.upsample_rates
    assert h.upsample_kernel_sizes == [9, 9, 4, 4, 4, 4, 4], h.upsample_kernel_sizes
    assert h.resblock_kernel_sizes == [3, 7, 11]
    assert h.resblock_dilation_sizes == [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    assert h.snake_logscale is True
    assert h.use_tanh_at_final is False
    assert h.use_bias_at_final is False
    assert h.num_mels == LATENT_DIM
    assert model.decoder.conv_post.bias is None, "use_bias_at_final=False => no conv_post bias"
    hop = int(np.prod(DECODER_RATES))
    assert hop == 800 and SAMPLE_RATE // hop == 40, "40 Hz token rate"

    n_tokens = 4
    bigvgan_x = torch.randn(1, LATENT_DIM, n_tokens, generator=generator)
    z = torch.randn(1, VAE_LATENT_CHANNELS, n_tokens, generator=generator)

    # Land the waveform at a realistic level BEFORE dumping any weights (see the docstring).
    gain = calibrate_output_level(model, z, TARGET_PEAK)

    # How hard the periodic activations are actually driven. `sin²(αx)/β` is quadratic near
    # zero, so a decoder whose internals sit near zero is effectively linear and its golden
    # would pass against a port with no SnakeBeta at all.
    pre_post = {}
    handle = model.decoder.conv_post.register_forward_pre_hook(
        lambda _m, args: pre_post.update(rms=float(args[0].pow(2).mean().sqrt()))
    )
    with torch.no_grad():
        model.decode(z)
    handle.remove()
    assert pre_post["rms"] > 0.05, (
        f"the decoder's internal activations RMS to {pre_post['rms']:.3e}; SnakeBeta would be "
        "effectively linear and the golden could not police it"
    )

    out.update(decode_state_published(model))

    with torch.no_grad():
        out["in.bigvgan.x"] = np32(bigvgan_x)
        out["out.bigvgan.y"] = np32(model.decoder(bigvgan_x))

        out["in.decode.z"] = np32(z)
        decoded = model.decode(z)
        out["out.decode.audio"] = np32(decoded)
        assert decoded.shape == (1, 1, n_tokens * hop), decoded.shape

    # -----------------------------------------------------------------------------------
    # (f) stereo: two INDEPENDENT 32-channel latents decoded through the same mono decoder,
    #     plus the de-normalization the pipeline applies first, plus the interleaved PCM the
    #     `AudioTrack` carries. A mono-duplicated output would pass a lazy test, so the two
    #     channels are drawn independently and `main` asserts they actually differ.
    # -----------------------------------------------------------------------------------
    latents_mean = torch.tensor(LATENTS_MEAN, dtype=torch.float32).view(1, 1, -1, 1)
    latents_std = torch.tensor(LATENTS_STD, dtype=torch.float32).view(1, 1, -1, 1)
    out["const.latents_mean"] = np.asarray(LATENTS_MEAN, dtype=np.float32)
    out["const.latents_std"] = np.asarray(LATENTS_STD, dtype=np.float32)

    with torch.no_grad():
        # [B, output_channel, vae_latent_channels, T]
        stereo_z = torch.randn(1, 2, VAE_LATENT_CHANNELS, n_tokens, generator=generator)
        out["in.stereo.z"] = np32(stereo_z)
        denorm = stereo_z * latents_std + latents_mean
        out["out.stereo.denorm"] = np32(denorm)
        # Fold the stereo axis into the batch: the decoder is mono (conv_post -> 1 channel).
        folded = denorm.reshape(2, VAE_LATENT_CHANNELS, n_tokens)
        waves = model.decode(folded)  # [2, 1, L]
        assert waves.shape == (2, 1, n_tokens * hop), waves.shape
        stereo = waves.reshape(1, 2, n_tokens * hop)
        out["out.stereo.audio"] = np32(stereo)
        # Interleaved f32 PCM, exactly `AudioTrack::samples`: L0, R0, L1, R1, ...
        interleaved = stereo[0].transpose(0, 1).reshape(-1)
        out["out.stereo.interleaved"] = np32(interleaved)

    # -----------------------------------------------------------------------------------
    # Non-degeneracy. A constant, all-zero or fully-saturated golden is a false green.
    # -----------------------------------------------------------------------------------
    for key, value in out.items():
        assert np.isfinite(value).all(), f"{key} has non-finite entries"
        if key.startswith("out."):
            assert float(value.std()) > 1e-4, f"{key} is ~constant (std {value.std()})"

    audio = out["out.stereo.audio"]
    saturated = float(np.mean(np.abs(audio) >= 1.0 - 1e-6))
    assert saturated < 0.05, (
        f"{saturated:.1%} of the decoded audio is clamped at +-1; a saturated golden would be "
        "reproduced by any port that also saturates"
    )
    left, right = audio[0, 0], audio[0, 1]
    channel_gap = float(np.abs(left - right).max() / max(np.abs(right).max(), 1e-12))
    assert channel_gap > 1e-2, (
        f"the two stereo channels are near-identical (rel gap {channel_gap:.3e}); the golden "
        "would not catch a mono-duplicating port"
    )
    # The interleave really is L,R,L,R — not two concatenated blocks.
    inter = out["out.stereo.interleaved"]
    assert np.array_equal(inter[0::2], left) and np.array_equal(inter[1::2], right)

    peak = float(np.abs(audio).max())
    assert peak > 0.2, f"the golden peaks at only {peak:.3e}; clamp and tanh agree there"

    print(
        f"  decoder_dim={DECODER_DIM} latent_dim={LATENT_DIM} "
        f"vae_latent_channels={VAE_LATENT_CHANNELS} hop={hop} "
        f"stages={len(model.decoder.ups)} resblocks={len(model.decoder.resblocks)}"
    )
    print(
        f"  conv_post gain={gain:.4g} pre-conv_post RMS={pre_post['rms']:.3f} "
        f"peak={peak:.3f} saturated={saturated:.2%} stereo channel gap={channel_gap:.3f}"
    )
    for key in sorted(out):
        if key.startswith(("in.", "out.", "const.")):
            print(f"  {key}: {list(out[key].shape)}")

    path = fixture("mlx-gen-minimax-h3/tests/fixtures/audio_vae_decode.safetensors")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, path)
    print(f"wrote {path} ({len(out)} tensors)")


if __name__ == "__main__":
    main()
