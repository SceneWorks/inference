"""Dump Qwen-Image 2.1 text-conditioning goldens for the Rust port (sc-24108).

Runs the frozen pipeline's own `encode_prompt` on the tiny snapshot — raw T2I template string ->
processor tokenization -> Qwen3-VL text decoder with the final RMSNorm neutralised -> drop the
system-role prefix (`_drop_idx`) -> `prompt_embeds` — and records, per prompt: the template token
ids the processor produced, the drop count, and the resulting embeddings. The Rust text path
(`text_encoder.rs`) must reproduce all three.

Also records the raw last-layer hidden state (before the drop) so a template/tokenizer mismatch is
distinguishable from a decoder-math mismatch.

Run with the pinned venv: `python3.12 tools/dump_qwen21_text_encoder.py`
Output: `tests/fixtures/qwen21_text_encoder.safetensors`.
"""

from __future__ import annotations

import torch

from _qwen21_common import FIXTURE_DIR, SYS_PROMPT, load_tiny_pipeline, save_safetensors

PROMPTS = {
    "fox": "a red fox in the forest",
    "negative": "blurry low quality photo",
    # The pipeline maps an empty prompt to a single space ("Qwen has no bos token").
    "empty": "",
    "rgba": "This is an RGBA image with transparency . a cat sticker . The image has alpha channel and the background is transparent .",
}


def main() -> None:
    pipe = load_tiny_pipeline()
    out, meta = {}, {"sys_prompt": SYS_PROMPT, "drop_idx": pipe._drop_idx}
    template = pipe.prompt_template_t2i
    for name, prompt in PROMPTS.items():
        text = template.format(prompt if prompt else " ")
        ids = pipe.processor(text=[text], padding=True, padding_side="left", return_tensors="pt")
        raw = {}
        layers = {}
        handles = []
        lm = pipe.text_encoder.model.language_model

        def capture(module, args, output):
            raw["hidden"] = (output[0] if isinstance(output, tuple) else output).detach().clone()

        handles.append(lm.layers[-1].register_forward_hook(capture))
        handles.append(
            lm.embed_tokens.register_forward_hook(
                lambda m, a, o: layers.__setitem__("embed", o.detach().clone())
            )
        )
        for i, layer in enumerate(lm.layers):
            handles.append(
                layer.register_forward_hook(
                    lambda m, a, o, i=i: layers.__setitem__(
                        f"layer_{i}", (o[0] if isinstance(o, tuple) else o).detach().clone()
                    )
                )
            )
        try:
            with torch.no_grad():
                embeds, mask, _ = pipe.encode_prompt(prompt=prompt)
        finally:
            for h in handles:
                h.remove()
        for stage, value in layers.items():
            out[f"{name}/trace/{stage}"] = value[0]
        assert mask is None, "single unpadded prompt: no mask"
        out[f"{name}/input_ids"] = ids.input_ids[0].to(torch.int32)
        out[f"{name}/last_hidden"] = raw["hidden"][0]
        out[f"{name}/prompt_embeds"] = embeds[0]
        meta[f"{name}/prompt"] = prompt
        print(f"{name}: ids={ids.input_ids[0].tolist()} embeds={tuple(embeds.shape)}")
    save_safetensors(FIXTURE_DIR / "qwen21_text_encoder.safetensors", out, meta)


if __name__ == "__main__":
    main()
