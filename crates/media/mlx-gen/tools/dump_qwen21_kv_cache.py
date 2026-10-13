"""Write the Qwen-Image 2.1 **prefix KV cache** goldens (sc-25903).

Reloads the tiny snapshot `dump_qwen21_pipeline.py` wrote and drives the frozen upstream
`QwenImage21Pipeline` with `use_kv_cache=True` — the path the other generators deliberately turn
off — recording what the Rust port's cached path has to reproduce. Everything is captured from the
pipeline's own transformer calls (a forward hook), so the oracle is what upstream actually ran, not
a re-derivation of it:

  * per prompt branch (`pos`, and `neg` under true CFG): the `encoder_hidden_states` and `img_mask`
    the transformer was given, and the joint layout they imply (metadata `segments`);
  * the packed condition latents, in order, and the initial noise;
  * per model evaluation `n` and branch: the target latents and `timestep` it was called with, the
    `kv_cache_mode` (`extract` for the first, `cached` after), and the target velocity it returned;
  * per branch and block: the cached prefix `k` / `v` exactly as `QwenImage21KVLayerCache` holds
    them (`[1, prefix, heads, head_dim]`; `k` post-norm and post-RoPE, `v` the raw projection);
  * the final latents with the cache on (`latents_final`) and, for reference, with it off
    (`latents_final_prefill`) — upstream documents the two paths as equally valid but not
    bit-identical, and the metadata records how far apart they are here.

Cases (64x64 targets, three Euler steps so two evaluations decode from the cache):

  * `t2i`      — text only: the prefix is the prompt;
  * `t2i_cfg`  — text only with true CFG: two caches, prompts of different length;
  * `ref1`     — one reference;
  * `ref2`     — two references;
  * `ref2_cfg` — two references with true CFG;
  * `aspect3`  — three references of different, non-square aspect ratios (heterogeneous blocks).

Before writing anything it re-runs `dump_qwen21_edit.py`'s `ref1` case with the cache off and
checks the result against the committed `qwen21_edit.safetensors`, so a venv that does not
reproduce the frozen fixtures cannot silently produce this one.

Run with the pinned venv (CPU), after `dump_qwen21_pipeline.py`:
    python3.12 tools/dump_qwen21_kv_cache.py

The committed fixture was written from a fresh venv holding diffusers @ 8b3c707e (pip
`git+https://github.com/huggingface/diffusers@8b3c707ebd3ec4881f4190cf42931da07eaf3b65`),
transformers 5.17.0, torch 2.14.1 (+ torchvision, which `Qwen3VLVideoProcessor` imports), numpy,
pillow, safetensors and jinja2; the frozen-fixture check above passed at max|Δ| = 0.
Output: `mlx-gen-qwen-image-2-1/tests/fixtures/qwen21_kv_cache.safetensors`
"""

from __future__ import annotations

import json

import torch

from _qwen21_common import FIXTURE_DIR, Z_DIM, load_tiny_pipeline, save_safetensors
from dump_qwen21_edit import synthetic

# Every word below is in the tiny WordLevel vocabulary (see `_qwen21_common.VOCAB_WORDS`). The two
# prompts tokenize to different lengths, so the two CFG branches have different prefixes.
PROMPT = "a red fox in the forest"
NEGATIVE = "blurry low quality photo"
OUTPUT_RESOLUTION = 64
TARGET = (64, 64)
STEPS = 3
TRUE_CFG = 2.5
BRANCHES = ("pos", "neg")


def segments_of(img_mask: torch.Tensor, img_shapes) -> list:
    """The joint layout as `["text", n]` / `["image", h, w]` runs, target block last — the same
    walk `QwenImage21Transformer2DModel.build_token_metadata` does over the slot mask."""
    mask = img_mask[0].tolist()
    shapes = [(int(h), int(w)) for (_, h, w) in img_shapes[0]]
    out, cursor, text_run, next_image = [], 0, 0, 0
    while cursor < len(mask):
        if not mask[cursor]:
            text_run += 1
            cursor += 1
            continue
        if text_run:
            out.append(["text", text_run])
            text_run = 0
        h, w = shapes[next_image]
        slots = h * w // 4
        assert all(mask[cursor : cursor + slots]), "an image block's slots are contiguous"
        out.append(["image", h, w])
        cursor += slots
        next_image += 1
    assert next_image == len(shapes) and not text_run, "the target block ends the sequence"
    return out


def pipe_kwargs(images, width, height, negative, true_cfg) -> dict:
    kwargs = {
        "prompt": PROMPT,
        "width": width,
        "height": height,
        "num_inference_steps": STEPS,
        "output_type": "latent",
    }
    if images:
        kwargs["image"] = list(images)
        kwargs["output_resolution"] = OUTPUT_RESOLUTION
    if negative is not None:
        kwargs["negative_prompt"] = negative
        kwargs["true_cfg_scale"] = true_cfg
    return kwargs


def run_case(pipe, name, images, out, meta, negative=None, true_cfg=1.0, seed=7):
    width, height = TARGET
    target_tokens = (height // 16) * (width // 16)
    torch.manual_seed(seed)
    init = torch.randn(1, target_tokens, Z_DIM)
    kwargs = pipe_kwargs(images, width, height, negative, true_cfg)

    calls = []

    def hook(module, args, call_kwargs, output):
        assert not args, "the pipeline calls the transformer with keyword arguments only"
        calls.append(
            {
                "hidden_states": call_kwargs["hidden_states"].detach().clone(),
                "encoder_hidden_states": call_kwargs["encoder_hidden_states"].detach().clone(),
                "timestep": call_kwargs["timestep"].detach().clone(),
                "img_shapes": call_kwargs["img_shapes"],
                "img_mask": call_kwargs["img_mask"].detach().clone(),
                "kv_cache": call_kwargs["kv_cache"],
                "mode": call_kwargs["kv_cache_mode"],
                "velocity": output[0].detach().clone(),
            }
        )

    handle = pipe.transformer.register_forward_hook(hook, with_kwargs=True)
    try:
        final = pipe(latents=init.clone(), use_kv_cache=True, **kwargs).images
    finally:
        handle.remove()
    prefill = pipe(latents=init.clone(), use_kv_cache=False, **kwargs).images

    branches = 2 if negative is not None else 1
    assert len(calls) == STEPS * branches, (len(calls), STEPS, branches)
    modes = []
    for n in range(STEPS):
        step_calls = calls[n * branches : (n + 1) * branches]
        mode = step_calls[0]["mode"]
        assert mode == ("extract" if n == 0 else "cached"), (n, mode)
        modes.append(mode)
        target = step_calls[0]["hidden_states"][:, -target_tokens:]
        out[f"{name}/eval_{n}/latents"] = target
        out[f"{name}/eval_{n}/timestep"] = step_calls[0]["timestep"].to(torch.float32)
        for branch, call in zip(BRANCHES, step_calls):
            assert call["mode"] == mode
            assert torch.equal(call["hidden_states"][:, -target_tokens:], target)
            out[f"{name}/eval_{n}/{branch}/velocity"] = call["velocity"][:, -target_tokens:]

    # The condition latents, split back out of the first call's image stream, in order.
    first = calls[0]
    shapes = [(int(h), int(w)) for (_, h, w) in first["img_shapes"][0]]
    cursor = 0
    for i, (h, w) in enumerate(shapes[:-1]):
        out[f"{name}/ref_latents_{i}"] = first["hidden_states"][:, cursor : cursor + h * w]
        cursor += h * w
    assert cursor + target_tokens == first["hidden_states"].shape[1]

    branch_meta = {}
    for b, branch in enumerate(BRANCHES[:branches]):
        call = calls[b]
        segments = segments_of(call["img_mask"], call["img_shapes"])
        prefix = sum(s[1] if s[0] == "text" else s[1] * s[2] for s in segments[:-1])
        out[f"{name}/{branch}/encoder_hidden_states"] = call["encoder_hidden_states"]
        out[f"{name}/{branch}/img_mask"] = call["img_mask"].to(torch.int32)
        cache = call["kv_cache"]
        assert cache is not None and cache is calls[b + branches]["kv_cache"]
        for i, layer in enumerate(cache.layer_caches):
            k, v = layer.get()
            assert k.shape[1] == prefix and v.shape[1] == prefix, (k.shape, prefix)
            out[f"{name}/{branch}/k_{i}"] = k.detach().clone()
            out[f"{name}/{branch}/v_{i}"] = v.detach().clone()
        branch_meta[branch] = {"segments": segments, "prefix_tokens": prefix}
    if branches == 2:
        assert calls[0]["kv_cache"] is not calls[1]["kv_cache"], "one cache per branch"
        assert branch_meta["pos"]["prefix_tokens"] != branch_meta["neg"]["prefix_tokens"], (
            "the two prompts must give prefixes of different length"
        )

    out[f"{name}/latents_init"] = init.clone()
    out[f"{name}/latents_final"] = final
    out[f"{name}/latents_final_prefill"] = prefill
    gap = (final - prefill).abs().max().item()
    meta[name] = {
        "prompt": PROMPT,
        "negative_prompt": negative,
        "true_cfg_scale": true_cfg,
        "width": width,
        "height": height,
        "steps": STEPS,
        "references": len(images),
        "evaluations": STEPS,
        "modes": modes,
        "blocks": len(calls[0]["kv_cache"].layer_caches),
        "branches": branch_meta,
        "upstream_cached_vs_prefill_max_abs": gap,
    }
    print(
        f"{name}: {len(images)} refs, prefix "
        f"{[branch_meta[b]['prefix_tokens'] for b in branch_meta]}, "
        f"upstream cached-vs-prefill max|Δ| = {gap:.3e} (peak {prefill.abs().max():.3f})"
    )


def assert_reproduces_frozen_edit_fixture(pipe) -> None:
    """`dump_qwen21_edit.py`'s `ref1` (cache off, seed 7, two steps) against the committed file."""
    from safetensors.torch import load_file

    frozen = load_file(str(FIXTURE_DIR / "qwen21_edit.safetensors"))
    init = frozen["ref1/latents_init"]
    got = pipe(
        prompt=PROMPT,
        image=[synthetic(1, 48, 48)],
        width=TARGET[0],
        height=TARGET[1],
        num_inference_steps=2,
        latents=init.clone(),
        output_type="latent",
        output_resolution=OUTPUT_RESOLUTION,
        use_kv_cache=False,
    ).images
    drift = (got - frozen["ref1/latents_final"]).abs().max().item()
    print(f"frozen qwen21_edit ref1/latents_final reproduced to max|Δ| = {drift:.3e}")
    assert drift <= 1e-5, f"this venv does not reproduce the frozen edit fixture ({drift:.3e})"


def main() -> None:
    pipe = load_tiny_pipeline()
    pipe.set_progress_bar_config(disable=True)
    assert pipe.transformer.config.causal_condition
    assert_reproduces_frozen_edit_fixture(pipe)
    out, meta = {}, {}

    photo_a = synthetic(1, 48, 48)
    photo_b = synthetic(2, 80, 80)
    wide, tall, tie = synthetic(21, 96, 24), synthetic(22, 24, 96), synthetic(23, 50, 32)

    run_case(pipe, "t2i", [], out, meta)
    run_case(pipe, "t2i_cfg", [], out, meta, negative=NEGATIVE, true_cfg=TRUE_CFG)
    run_case(pipe, "ref1", [photo_a], out, meta)
    run_case(pipe, "ref2", [photo_a, photo_b], out, meta)
    run_case(pipe, "ref2_cfg", [photo_a, photo_b], out, meta, negative=NEGATIVE, true_cfg=TRUE_CFG)
    run_case(pipe, "aspect3", [wide, tall, tie], out, meta)

    # Metadata values are JSON so the Rust tests parse one stable format.
    save_safetensors(
        FIXTURE_DIR / "qwen21_kv_cache.safetensors",
        out,
        {k: json.dumps(v, sort_keys=True) for k, v in meta.items()},
    )


if __name__ == "__main__":
    main()
