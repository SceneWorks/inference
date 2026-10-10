"""Dump Qwen-Image 2.1 DiT goldens for the Rust port (sc-24108).

Runs the tiny snapshot's `QwenImage21Transformer2DModel` (the frozen upstream class) on fixed
inputs in the exact T2I layout the pipeline builds — text tokens first, one vision slot per 2x2
group of target latents appended — and records, per case, the inputs, the joint RoPE table the
model built (`pos_embed`), every block's output (forward hooks) and the final velocity. Cases:

  * `square`  — text 5, target 2x2 (one slot), timestep 0.7;
  * `wide`    — text 5, target 4x2 (h=4, w=2: two slots), timestep 0.3, exercising the centred
                height/width RoPE grid and the multi-slot expansion;
  * `square_t0` — `square` at timestep 0.0 (the row text tokens always modulate from).

No KV cache: this is the `QwenImage21AttnProcessor` prefill path the Rust port mirrors.

Run with the pinned venv: `python3.12 tools/dump_qwen21_transformer.py`
Output: `tests/fixtures/qwen21_transformer.safetensors`.
"""

from __future__ import annotations

import torch

from _qwen21_common import FIXTURE_DIR, TEXT_HIDDEN, Z_DIM, load_tiny_pipeline, save_safetensors

TEXT_LEN = 5


def run_case(model, name, height, width, timestep, out, meta, seed):
    g = torch.Generator("cpu").manual_seed(seed)
    tokens = height * width
    hidden = torch.randn((1, tokens, Z_DIM), generator=g)
    enc = torch.randn((1, TEXT_LEN, TEXT_HIDDEN), generator=g)
    img_mask = torch.zeros((1, TEXT_LEN + tokens // 4), dtype=torch.bool)
    img_mask[:, TEXT_LEN:] = True
    t = torch.tensor([timestep], dtype=torch.float32)

    blocks = {}
    trace = {}
    handles = []

    def tap(name):
        def hook(m, a, o):
            trace[name] = (o[0] if isinstance(o, tuple) else o).detach().clone()

        return hook

    for i, block in enumerate(model.transformer_blocks):
        handles.append(
            block.register_forward_hook(
                lambda m, a, o, i=i: blocks.__setitem__(i, o.detach().clone())
            )
        )
        handles.append(block.attn.register_forward_hook(tap(f"block_{i}_attn")))
        handles.append(block.img_mlp.register_forward_hook(tap(f"block_{i}_mlp")))
    # Every non-block stage, so a parity failure localises to one module.
    for stage, module in [
        ("img_in", model.img_in),
        ("txt_in", model.txt_in),
        ("temb", model.time_text_embed),
        ("modulation", model.modulation),
        ("norm_out", model.norm_out),
        ("proj_out", model.proj_out),
    ]:
        handles.append(module.register_forward_hook(tap(stage)))
    rope = {}
    handles.append(
        model.pos_embed.register_forward_hook(lambda m, a, o: rope.__setitem__("freqs", o.detach().clone()))
    )
    try:
        with torch.no_grad():
            velocity = model(
                hidden_states=hidden,
                encoder_hidden_states=enc,
                timestep=t,
                img_shapes=[[(1, height, width)]],
                img_mask=img_mask,
                return_dict=False,
            )[0]
    finally:
        for h in handles:
            h.remove()
    freqs = rope["freqs"]  # complex [S, head_dim/2]
    out[f"{name}/hidden_states"] = hidden
    out[f"{name}/encoder_hidden_states"] = enc
    out[f"{name}/timestep"] = t
    out[f"{name}/rope_cos"] = freqs.real.contiguous()
    out[f"{name}/rope_sin"] = freqs.imag.contiguous()
    for i, o in blocks.items():
        out[f"{name}/block_{i}_out"] = o[0]
    for stage, value in trace.items():
        out[f"{name}/trace/{stage}"] = value[0] if value.dim() == 3 else value
    # The model returns the whole joint sequence; the pipeline keeps the target's tail.
    out[f"{name}/velocity"] = velocity[0]
    meta[f"{name}/height"] = height
    meta[f"{name}/width"] = width
    meta[f"{name}/text_len"] = TEXT_LEN
    meta[f"{name}/timestep"] = timestep
    print(f"{name}: velocity {tuple(velocity.shape)} |v|max={velocity.abs().max():.4f}")


def main() -> None:
    pipe = load_tiny_pipeline()
    model = pipe.transformer.eval()
    out, meta = {}, {}
    run_case(model, "square", 2, 2, 0.7, out, meta, seed=11)
    run_case(model, "wide", 4, 2, 0.3, out, meta, seed=12)
    run_case(model, "square_t0", 2, 2, 0.0, out, meta, seed=11)
    save_safetensors(FIXTURE_DIR / "qwen21_transformer.safetensors", out, meta)


if __name__ == "__main__":
    main()
