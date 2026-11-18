"""Generate the independent Transformers oracle for the dispatch-only Loop release proof."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

FIXTURES = (
    ("text", "What is 13 times 7? Answer with just the number.", None),
    ("red", "What is the dominant color of this image? Answer with one word.", [205, 35, 35]),
    ("blue", "What is the dominant color of this image? Answer with one word.", [35, 70, 200]),
)
STEPS = 3


def run_case(model, inputs, torch, *, cached, tokens=None, expected_plan):
    input_ids = inputs["input_ids"]
    common = {key: value for key, value in inputs.items() if key != "input_ids"}
    attention = common.pop("attention_mask")
    token_types = common.pop("mm_token_type_ids", None)
    past = None
    scores = []
    chosen = []
    for step in range(STEPS):
        new_input = input_ids[:, -1:] if cached and step else input_ids
        media = {} if cached and step else common
        if token_types is not None:
            media = {
                **media,
                "mm_token_type_ids": token_types[:, -1:] if cached and step else token_types,
            }
        position_inputs = {"attention_mask": attention}
        if token_types is not None:
            position_inputs["mm_token_type_ids"] = token_types
        if "image_grid_thw" in common:
            position_inputs["image_grid_thw"] = common["image_grid_thw"]
        positions = model._prepare_position_ids_for_generation(input_ids, position_inputs)
        output = model(
            input_ids=new_input, attention_mask=attention, use_cache=cached,
            position_ids=positions[..., -new_input.shape[-1]:],
            past_key_values=past, logits_to_keep=1, **media,
        )
        if cached:
            past = output.past_key_values
            if tuple(past._loop_execution_plan) != expected_plan or len(past.layers) != len(expected_plan):
                raise RuntimeError("upstream did not allocate an independent cache per Loop visit")
            if past.get_seq_length() != input_ids.shape[-1]:
                raise RuntimeError("upstream Loop cache sequence length mismatch")
        row = output.logits[0, -1, :].float().cpu()
        if not torch.isfinite(row).all():
            raise RuntimeError("non-finite oracle logits")
        next_id = int(row.argmax()) if tokens is None else tokens[step]
        scores.append(row)
        chosen.append(next_id)
        input_ids = torch.cat((input_ids, input_ids.new_tensor([[next_id]])), dim=1)
        attention = torch.cat((attention, attention.new_ones((1, 1))), dim=1)
        if token_types is not None:
            token_types = torch.cat((token_types, token_types.new_zeros((1, 1))), dim=1)
    return scores, chosen


def main(snapshot: Path, inventory_sha: str, output: Path) -> None:
    import torch
    import transformers
    from PIL import Image, __version__ as pillow_version
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if transformers.__version__ != "5.18.0" or torch.__version__ != "2.11.0+cu130":
        raise RuntimeError("official Transformers/PyTorch reference toolchain mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("reference requires one admitted native BF16 CUDA device")
    # Pin the recurrent path to the reviewed wheel environment (without optional accelerator
    # kernels) and exclude TF32 rounding from prospective BF16 tolerance calibration.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="eager",
    ).eval()
    layers = model.model.language_model.layers
    plan = (*range(4), *range(4, 28), *range(4, 28), *range(28, 32))
    if len(layers) != 32 or any(
        param.dtype != torch.float32
        for layer in layers[4:28]
        for router in (layer.mhc_attn, layer.mhc_ffn)
        for param in router.parameters()
    ):
        raise RuntimeError("reference did not load expected physical layers and FP32 mHC")
    cases = []
    with torch.inference_mode():
        for name, prompt, color in FIXTURES:
            image = None if color is None else {"width": 256, "height": 256, "rgb": color}
            content = [{"type": "text", "text": prompt}]
            processor_kwargs = {}
            if image is not None:
                content.insert(0, {"type": "image"})
                processor_kwargs["images"] = [Image.new("RGB", (256, 256), tuple(color))]
            rendered = processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            inputs = processor(text=[rendered], return_tensors="pt", **processor_kwargs).to("cuda:0")
            model.set_attn_implementation("eager")
            baseline, chosen = run_case(model, inputs, torch, cached=True, expected_plan=plan)
            cold, _ = run_case(model, inputs, torch, cached=False, tokens=chosen, expected_plan=plan)
            model.set_attn_implementation("sdpa")
            if model.model.language_model.config._attn_implementation != "sdpa":
                raise RuntimeError("reference did not activate independent SDPA attention")
            alternate, _ = run_case(model, inputs, torch, cached=True, tokens=chosen, expected_plan=plan)
            steps = []
            for index, (logits, token_id) in enumerate(zip(baseline, chosen, strict=True)):
                deviations = [
                    ((other[index] - other[index][token_id]) - (logits - logits[token_id])).abs().max().item()
                    for other in (cold, alternate)
                ]
                # A centered pair has two independently rounded BF16 scores per implementation.
                # Below one, use the unit-scale spacing to allow for cancellation in centering.
                max_score = max(1.0, float(logits.abs().max()))
                bf16_spacing = math.ldexp(1.0, math.floor(math.log2(max_score)) - 7)
                tolerance = max(deviations) + 2 * bf16_spacing
                ids = list(dict.fromkeys([token_id, *logits.topk(32).indices.tolist(),
                                           *range(0, logits.numel(), 4093)]))
                steps.append({
                    "token_id": token_id, "token_ids": ids, "logits": logits[ids].tolist(),
                    "max_abs_error": tolerance, "source_cold_max_abs": deviations[0],
                    "source_sdpa_max_abs": deviations[1], "bf16_spacing": bf16_spacing,
                    "log_probability": float(logits.log_softmax(0)[token_id]),
                    "top_1_margin": float(logits.topk(2).values.diff().abs()[0]),
                })
            cases.append({
                "id": name, "prompt": prompt, "thinking": False, "image": image,
                "input_ids": inputs["input_ids"][0].tolist(), "steps": steps,
                "image_grid_thw": inputs["image_grid_thw"].tolist() if image else None,
                "thinking_prompt": processor.apply_chat_template(
                    [{"role": "user", "content": content}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=True,
                ),
            })
    payload = {
        "schema_version": 1, "hf_revision": snapshot.name,
        "source_inventory_sha256": inventory_sha,
        "comparison": "BF16 PyTorch CUDA eager cached, eager full replay, SDPA cached",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(), "pillow": pillow_version,
        "cases": cases,
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
