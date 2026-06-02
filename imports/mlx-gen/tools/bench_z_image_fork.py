"""sc-2352: Z-Image-turbo timing baseline on the frozen Python mflux fork.

Run from the fork:  cd ~/repos/mflux && uv run python tools/bench_z_image_fork.py

Mirrors `mlx-gen-z-image/tests/bench_z_image.rs` so the numbers line up, and reports BOTH the
fork's eager path (apples-to-apples with the Rust port, which has no compile) and the fork's
production path, which wraps the per-step transformer in `mx.compile` (see
`src/mflux/models/z_image/variants/z_image.py` `_predict`). On M3+ the fork always compiles, so
the compiled number is what a fork user actually gets.

  * end-to-end wall-clock (encode + denoise + unpack + VAE decode), the latency a user feels;
  * pure DiT per-step time with `mx.eval` forced each step.

MLX is lazy, so every timed block ends in an `mx.eval` to force the compute it is measuring.
"""

import time

import mlx.core as mx
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import (
    FlowMatchEulerDiscreteScheduler as S,
)
from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.z_image.z_image_initializer import ZImageInitializer

SIZES = [(256, 256), (512, 512), (1024, 1024)]  # (W, H)
STEPS, RUNS, PROMPT = 4, 3, "a fox"


class Holder:
    pass


def build_sigmas(w, h, steps):
    seq_len = (h // 16) * (w // 16)
    mu = S._compute_empirical_mu(seq_len, steps)
    sigmas = S._time_shift_exponential_array(mu, 1.0, mx.linspace(1.0, 1.0 / steps, steps))
    return mx.concatenate([sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0)


def make_predict(model, compiled):
    """The transformer call as the fork wraps it (turbo: no CFG, so predict == transformer)."""

    def predict(latents, timestep, sigmas, cap):
        return model.transformer(timestep=timestep, x=latents, cap_feats=cap, sigmas=sigmas)

    return mx.compile(predict) if compiled else predict


def denoise(predict, cap, sigmas, w, h, seed, timed_steps=None):
    latents = ZImageLatentCreator.create_noise(seed, h, w)
    mx.eval(latents)
    for t in range(STEPS):
        t0 = time.perf_counter()
        ts = mx.array(1.0 - float(sigmas[t]), dtype=mx.float32)
        latents = latents + (sigmas[t + 1] - sigmas[t]) * predict(latents, ts, sigmas, cap)
        if timed_steps is not None:
            mx.eval(latents)
            timed_steps.append(time.perf_counter() - t0)
    return latents


def median(v):
    return sorted(v)[len(v) // 2]


def per_step(predict, cap, sigmas, w, h):
    denoise(predict, cap, sigmas, w, h, 7, timed_steps=[])  # warmup (also traces the compile)
    steps = []
    denoise(predict, cap, sigmas, w, h, 8, timed_steps=steps)
    return sum(steps[1:]) / (len(steps) - 1), steps


def e2e(model, predict, sigmas, w, h):
    def run(seed):
        t0 = time.perf_counter()
        cap = PromptEncoder.encode_prompt(PROMPT, tok, model.text_encoder)
        latents = denoise(predict, cap, sigmas, w, h, seed)
        unpacked = ZImageLatentCreator.unpack_latents(latents, h, w)
        decoded = model.vae.decode(unpacked)
        mx.eval(decoded)
        return time.perf_counter() - t0

    run(0)  # warmup
    totals = [run(r + 1) for r in range(RUNS)]
    return median(totals), totals


def main():
    global tok
    model = Holder()
    ZImageInitializer.init(model, model_config=ModelConfig.z_image_turbo(), quantize=None)
    tok = model.tokenizers["z_image"]

    print(f"\n# Z-Image-turbo (Python mflux fork) — {STEPS} steps, bf16, median of {RUNS} runs (after warmup)")
    for (w, h) in SIZES:
        sigmas = build_sigmas(w, h, STEPS)
        cap = PromptEncoder.encode_prompt(PROMPT, tok, model.text_encoder)

        eager = make_predict(model, compiled=False)
        comp = make_predict(model, compiled=True)

        eager_step, eager_steps = per_step(eager, cap, sigmas, w, h)
        comp_step, comp_steps = per_step(comp, cap, sigmas, w, h)
        eager_e2e, _ = e2e(model, eager, sigmas, w, h)
        comp_e2e, comp_runs = e2e(model, comp, sigmas, w, h)

        print(
            f"{w}x{h}:\n"
            f"    DiT s/step   eager {eager_step:.3f}  | compiled {comp_step:.3f}   "
            f"(eager per-step={[round(s, 3) for s in eager_steps]}, compiled={[round(s, 3) for s in comp_steps]})\n"
            f"    e2e s/image  eager {eager_e2e:.3f}  | compiled {comp_e2e:.3f}   "
            f"(compiled runs={[round(t, 3) for t in comp_runs]})"
        )


if __name__ == "__main__":
    main()
