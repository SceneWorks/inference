"""MiniMax-H3 joint audio+video denoise parity fixture (sc-17146).

Runs the **official diffusers** ``MiniMaxH3Scheduler`` — two instances, loaded from the *published*
``scheduler/`` and ``audio_scheduler/`` config folders — plus the reference's own
``MiniMaxH3PrepareLayoutStep.build_packed_sequence`` and ``build_row_timesteps``, and dumps a whole
**2-step** joint loop: the sigma grids, the per-row timestep plan, the packed layout, and the
latents after each of the two Euler updates.

Read ``mlx-gen-minimax-h3/src/layout.rs`` rule 3 before changing anything here
--------------------------------------------------------------------------------

A golden and a loader that share a layout prove only that they agree with each other. The two
schedulers here are **not** constructed with hand-typed shifts: they are
``from_pretrained(snapshot, subfolder=...)``, so the 12.0 / 3.0 pair is read from the same published
bytes production reads. If MiniMax ever reships those configs, this fixture changes and the Rust
test fails, rather than both sides agreeing with a constant someone typed twice.

Why a velocity table instead of a model
---------------------------------------

The DiT's 17 input/output projections are sc-17147's, so there is no runnable transformer at this
slice. Feeding the loop a **pre-drawn velocity per step** makes the golden a test of exactly what
sc-17146 owns — the two schedules, the reversed velocity sign, the two-source sigma, the row
bookkeeping and the conditioning-tail write — with no dependence on the unported half. The
velocities are seeded and dumped, so the Rust loop consumes the identical numbers.

Why the geometry is real rather than shrunk
-------------------------------------------

124 frames is the *shortest legal render* (``17n + 5``), giving 37 latent frames and 207 stereo
audio latents. That is still only 653 packed rows at a 4x6 latent, and it keeps three properties a
shrunken geometry would lose:

* 37 latent frames walks the ``_ROPE_FRAMES_PER_LATENT`` ``(1,4,4,4,4)`` cycle **past index 5**,
  which ``positions.rs`` flags as "past every tiny fixture";
* ``207 != 37 * k``, so the audio and video tracks have the real +8.33 ms length residual;
* both keyframe anchors are present, so the ``max(video_t, 0.999)`` conditioning class and the
  discontiguous ``video_indices`` are both exercised.

    MINIMAX_H3_SNAPSHOT=... ~/.minimax-h3-spike/venv/bin/python tools/dump_minimax_h3_av_denoise.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from _paths import fixture, hf_hub_cache

# --- the layout the goldens are computed over ---------------------------------------------------
NUM_FRAMES = 124  # the shortest legal render: 17n + 5
NUM_LATENT_FRAMES = 37  # 5n + 2
NUM_AUDIO_LATENTS = 207  # round(124 / 24 * 40)
LATENT_HEIGHT = 4
LATENT_WIDTH = 6
PATCH_SIZE = (1, 2, 2)
NUM_TEXT_TOKENS = 5
AUDIO_CHANNELS = 2
KEYFRAME_ANCHORS = ("first", "last")

# Modality tags: video 0, text 1, audio 2.
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2

# Video latent channels x prod(patch) and audio latent channels — the row widths.
VIDEO_FEATURES = 24 * PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2]
AUDIO_FEATURES = 32

# A 2-STEP schedule: the terminal sigma is inside the requested count, so 3 requested steps drive
# exactly 2 model evaluations.
NUM_INFERENCE_STEPS = 3
EXPECTED_EVALS = 2


def np32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).contiguous().numpy()


def snapshot_root() -> Path:
    """The published MiniMax-H3 snapshot — the scheduler configs come from it, not from literals."""
    if explicit := os.environ.get("MINIMAX_H3_SNAPSHOT"):
        root = Path(explicit)
    else:
        snaps = sorted(
            (hf_hub_cache() / "models--MiniMaxAI--MiniMax-H3" / "snapshots").glob("*")
        )
        if not snaps:
            raise SystemExit(
                "MINIMAX_H3_SNAPSHOT is required: point it at a MiniMaxAI/MiniMax-H3 snapshot root "
                "(the dir holding `scheduler/` and `audio_scheduler/`)."
            )
        root = snaps[-1]
    for sub in ("scheduler", "audio_scheduler"):
        if not (root / sub / "scheduler_config.json").is_file():
            raise SystemExit(f"{root} has no {sub}/scheduler_config.json")
    return root


def schedulers(root: Path):
    """The two published schedulers. `shift` is READ from the checkpoint, never typed here."""
    try:
        from diffusers import MiniMaxH3Scheduler
    except ImportError:  # pragma: no cover - older diffusers layout
        from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    video = MiniMaxH3Scheduler.from_pretrained(root, subfolder="scheduler")
    audio = MiniMaxH3Scheduler.from_pretrained(root, subfolder="audio_scheduler")
    assert video.config.shift == 12.0, video.config.shift
    assert audio.config.shift == 3.0, audio.config.shift
    return video, audio


def packed_layout():
    """The reference's own packed layout, so the row order is dumped rather than described."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
    )

    text_token_tags = torch.full((NUM_TEXT_TOKENS,), TEXT_TAG, dtype=torch.long)
    return MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=text_token_tags,
        num_latent_frames=NUM_LATENT_FRAMES,
        latent_height=LATENT_HEIGHT,
        latent_width=LATENT_WIDTH,
        num_audio_latents=NUM_AUDIO_LATENTS,
        patch_size=PATCH_SIZE,
        audio_channels=AUDIO_CHANNELS,
        audio_tag=AUDIO_TAG,
        video_tag=VIDEO_TAG,
        keyframe_anchors=KEYFRAME_ANCHORS,
    )


def run_loop(video_sched, audio_sched, latents, audio_latents, velocities, ncond_v, ncond_a):
    """The reference denoise loop, verbatim in structure (`denoise.py` MiniMaxH3LoopSchedulerStep).

    Returns the per-step latents after each Euler update.
    """
    latents = latents.clone()
    audio_latents = audio_latents.clone()
    history = []
    for i, t in enumerate(video_sched.timesteps):
        v_vel, a_vel = velocities[i]
        latents[ncond_v:] = video_sched.step(
            v_vel[ncond_v:].float(), t, latents[ncond_v:], return_dict=False
        )[0]
        audio_latents[ncond_a:] = audio_sched.step(
            a_vel[ncond_a:].float(),
            audio_sched.timesteps[i],
            audio_latents[ncond_a:],
            return_dict=False,
        )[0]
        history.append((latents.clone(), audio_latents.clone()))
    return history


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative max-abs-diff — the only metric this epic gates on."""
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def main() -> None:
    import diffusers

    root = snapshot_root()
    generator = torch.Generator().manual_seed(17146)
    out: dict[str, np.ndarray] = {}

    # ---- the packed layout, straight from the reference -----------------------------------------
    (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_condition_video_rows,
        num_condition_audio_rows,
    ) = packed_layout()
    seq_len = position_ids.shape[0]
    rows_per_frame = (LATENT_HEIGHT // PATCH_SIZE[1]) * (LATENT_WIDTH // PATCH_SIZE[2])
    assert num_condition_video_rows == len(KEYFRAME_ANCHORS) * rows_per_frame
    assert num_condition_audio_rows == 0, "fl2va carries no reference soundtrack"
    assert seq_len == NUM_TEXT_TOKENS + num_condition_video_rows + NUM_AUDIO_LATENTS * AUDIO_CHANNELS + NUM_LATENT_FRAMES * rows_per_frame

    out["layout.position_ids"] = np32(position_ids)
    out["layout.token_tags"] = token_tags.to(torch.float32).numpy()
    out["layout.video_indices"] = video_indices.to(torch.float32).numpy()
    out["layout.audio_indices"] = audio_indices.to(torch.float32).numpy()
    out["layout.text_indices"] = text_indices.to(torch.float32).numpy()

    # `video_indices` skips the audio block — assert it here so the fixture cannot be regenerated
    # into a shape where the Rust test's discontiguity check is vacuous.
    assert video_indices[num_condition_video_rows - 1] + 1 != video_indices[num_condition_video_rows]

    # ---- the two schedules, from the published configs -------------------------------------------
    video_sched, audio_sched = schedulers(root)
    video_sched.set_timesteps(NUM_INFERENCE_STEPS)
    audio_sched.set_timesteps(NUM_INFERENCE_STEPS)
    assert len(video_sched.timesteps) == EXPECTED_EVALS, len(video_sched.timesteps)
    assert len(audio_sched.timesteps) == EXPECTED_EVALS

    out["out.video_sigmas"] = np32(video_sched.sigmas)
    out["out.audio_sigmas"] = np32(audio_sched.sigmas)
    out["out.video_timesteps"] = np32(video_sched.timesteps)
    out["out.audio_timesteps"] = np32(audio_sched.timesteps)

    # ---- the per-row timestep plan ----------------------------------------------------------------
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3SetTimestepsStep,
    )

    keyframe_noise_aug = 0.999
    row_timestep_values = []
    for i in range(EXPECTED_EVALS):
        unique, inverse = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            video_indices,
            audio_indices,
            num_condition_video_rows,
            num_condition_audio_rows,
            int(text_indices.numel()),
            float(video_sched.timesteps[i]),
            float(audio_sched.timesteps[i]),
            max(float(video_sched.timesteps[i]), keyframe_noise_aug),
            1.0,
        )
        # Resolve to a per-row VALUE. The port dedups into one GLOBAL table in first-appearance
        # order while the reference builds a per-step table sorted ascending, so the index tensors
        # are not comparable — the resolved values are, and they are what the AdaLN row actually
        # depends on.
        out[f"out.row_timesteps.{i}"] = np32(unique[inverse])
        out[f"out.step_timesteps.{i}"] = np32(unique)
        row_timestep_values.append(unique[inverse])

    # ---- the joint loop ---------------------------------------------------------------------------
    num_video_rows = int(video_indices.numel())
    num_audio_rows = int(audio_indices.numel())
    latents = torch.randn(num_video_rows, VIDEO_FEATURES, generator=generator, dtype=torch.float32)
    audio_latents = torch.randn(
        num_audio_rows, AUDIO_FEATURES, generator=generator, dtype=torch.float32
    )
    velocities = [
        (
            torch.randn(num_video_rows, VIDEO_FEATURES, generator=generator, dtype=torch.float32),
            torch.randn(num_audio_rows, AUDIO_FEATURES, generator=generator, dtype=torch.float32),
        )
        for _ in range(EXPECTED_EVALS)
    ]
    out["in.video_latents"] = np32(latents)
    out["in.audio_latents"] = np32(audio_latents)
    for i, (v, a) in enumerate(velocities):
        out[f"in.video_velocity.{i}"] = np32(v)
        out[f"in.audio_velocity.{i}"] = np32(a)

    history = run_loop(
        video_sched,
        audio_sched,
        latents,
        audio_latents,
        velocities,
        num_condition_video_rows,
        num_condition_audio_rows,
    )
    for i, (v, a) in enumerate(history):
        out[f"out.video_latents.{i}"] = np32(v)
        out[f"out.audio_latents.{i}"] = np32(a)

    final_video, final_audio = history[-1]
    # The conditioning anchors must be bit-identical to the input: the loop writes only the tail.
    assert torch.equal(
        final_video[:num_condition_video_rows], latents[:num_condition_video_rows]
    ), "the keyframe anchors moved"
    assert not torch.equal(
        final_video[num_condition_video_rows:], latents[num_condition_video_rows:]
    )

    # ---- measured negative controls ----------------------------------------------------------------
    def rerun(video_shift: float, audio_shift: float):
        v, a = schedulers(root)
        v.set_shift(video_shift)
        a.set_shift(audio_shift)
        v.set_timesteps(NUM_INFERENCE_STEPS)
        a.set_timesteps(NUM_INFERENCE_STEPS)
        return run_loop(
            v, a, latents, audio_latents, velocities, num_condition_video_rows,
            num_condition_audio_rows,
        )[-1]

    swapped_v, swapped_a = rerun(3.0, 12.0)
    swap_rel = max(rel(swapped_v, final_video), rel(swapped_a, final_audio))

    both_v, both_a = rerun(12.0, 12.0)
    both_rel = rel(both_a, final_audio)

    def shift(sigma, s):
        return s * sigma / (1 + (s - 1) * sigma)

    base = torch.linspace(1.0, 0.0, NUM_INFERENCE_STEPS, dtype=torch.float32)
    once = torch.unique_consecutive(shift(base, 12.0))
    twice = torch.unique_consecutive(shift(shift(base, 12.0), 12.0))
    double_rel = rel(twice, once)

    # The reversed velocity sign: diffusers' `x0 = x_t - sigma*v` against MiniMax's `+`.
    sign_rel = rel(*(lambda t, x, v: (x - (1 - t) * v, x + (1 - t) * v))(
        float(video_sched.timesteps[0]), latents, velocities[0][0]
    ))

    # Pinning the text rows at a clean 1.0 — what sc-17145's docs and the sc-17242 spike both
    # describe — instead of letting them keep the video timestep. Measured as the fraction of rows
    # whose timestep would change.
    text_rows_changed = 0
    for i in range(EXPECTED_EVALS):
        want = row_timestep_values[i].clone()
        mutated = want.clone()
        mutated[text_indices] = 1.0
        text_rows_changed = max(text_rows_changed, int((mutated != want).sum()))
    assert text_rows_changed == NUM_TEXT_TOKENS, text_rows_changed

    for name, value in (
        ("sigma swap", swap_rel),
        ("one shift for both", both_rel),
        ("double-applied shift", double_rel),
        ("reversed velocity sign", sign_rel),
    ):
        assert value > 1e-2, f"the {name} control moves the output by only {value:.3e}"

    # ---- non-degeneracy ----------------------------------------------------------------------------
    for key, value in out.items():
        assert np.isfinite(value).all(), f"{key} has non-finite entries"
        if key.startswith("out.") and value.size > 1:
            assert float(value.std()) > 1e-4, f"{key} is ~constant ({value.std()})"

    for key in sorted(out):
        print(f"  {key}: {list(out[key].shape)}")
    print(f"  seq_len={seq_len}, video rows={num_video_rows} ({num_condition_video_rows} anchored), "
          f"audio rows={num_audio_rows}")
    print(f"  video sigmas {[round(float(x), 6) for x in video_sched.sigmas]}")
    print(f"  audio sigmas {[round(float(x), 6) for x in audio_sched.sigmas]}")
    print(f"  controls: swap {swap_rel:.6e}  both {both_rel:.6e}  double {double_rel:.6e}  "
          f"sign {sign_rel:.6e}")

    meta = {
        "provenance": "converted-checkpoint",
        "reference": "diffusers.MiniMaxH3Scheduler",
        "reference_version": diffusers.__version__,
        "layout_reference": "diffusers.MiniMaxH3PrepareLayoutStep.build_packed_sequence",
        "scheduler_source": "published scheduler/ + audio_scheduler/ configs",
        "video_sigma_shift": f"{float(video_sched.config.shift)}",
        "audio_sigma_shift": f"{float(audio_sched.config.shift)}",
        "keyframe_noise_aug": f"{keyframe_noise_aug}",
        "num_inference_steps": f"{NUM_INFERENCE_STEPS}",
        "num_evals": f"{EXPECTED_EVALS}",
        "num_frames": f"{NUM_FRAMES}",
        "text_rows_at_video_timestep": f"{NUM_TEXT_TOKENS}",
        "sigma_swap_rel": f"{swap_rel:.6e}",
        "one_shift_for_both_rel": f"{both_rel:.6e}",
        "double_shift_rel": f"{double_rel:.6e}",
        "velocity_sign_rel": f"{sign_rel:.6e}",
        "story": "sc-17146",
    }

    path = fixture("mlx-gen-minimax-h3/tests/fixtures/av_denoise.safetensors")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({k: torch.from_numpy(v) for k, v in out.items()}, path, metadata=meta)
    print(f"wrote {path} ({len(out)} tensors) from {meta['reference']} {meta['reference_version']}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
