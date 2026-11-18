"""Generate the independent Transformers oracle for the dispatch-only Loop release proof."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

COLOR_PROMPT = "What is the dominant color of this image? Answer with one word."
RED = [205, 35, 35]
BLUE = [35, 70, 200]
FIXTURES = (
    ("text", "What is 13 times 7? Answer with just the number.", None, "91"),
    ("red", COLOR_PROMPT, {"width": 256, "height": 256, "rgb": RED}, "red"),
    ("blue", COLOR_PROMPT, {"width": 256, "height": 256, "rgb": BLUE}, "blue"),
    ("quadrants", "What color covers the upper left of the image? Answer with one word.",
     {"width": 353, "height": 249, "rgb": RED, "split_x": 209, "split_y": 155,
      "quadrants": [RED, BLUE, [36, 174, 47], [232, 190, 40]]}, "red"),
)
STEPS = 3


def fixture_image(image, pil):
    width, height = image["width"], image["height"]
    if "quadrants" not in image:
        return pil.new("RGB", (width, height), tuple(image["rgb"]))
    colors = [bytes(rgb) for rgb in image["quadrants"]]
    pixels = b"".join(colors[2 * (y >= image["split_y"]) + (x >= image["split_x"])]
                      for y in range(height) for x in range(width))
    return pil.frombytes("RGB", (width, height), pixels)


def processor_evidence(image, image_bytes, inputs, patch_size, temporal_size, merge_size):
    grid = inputs["image_grid_thw"].tolist()
    rgb = inputs["pixel_values"].float().cpu().contiguous()
    _, rows, columns = grid[0]
    height, width = rows * patch_size, columns * patch_size
    expected = [[1, 16, 22]] if "quadrants" in image else [[1, 16, 16]]
    if grid != expected or tuple(rgb.shape) != (rows * columns, 3 * temporal_size * patch_size**2):
        raise RuntimeError(f"unexpected SenseNova processor tensor geometry: {grid}, {rgb.shape}")
    points = [
        (1, 1), (patch_size + 1, 1), (1, patch_size + 1), (patch_size + 1, patch_size + 1),
        (2 * patch_size + 1, 1), (1, 2 * patch_size + 1), (width - 2, 1),
        (1, height - 2), (width - 2, height - 2),
    ]
    corners = {(1, 1): 0, (width - 2, 1): 1, (1, height - 2): 2, (width - 2, height - 2): 3}
    if "quadrants" in image:
        # Original input coordinate of the color break projected into the resize.
        edge_x = round(image["split_x"] * width / image["width"])
        edge_y = round(image["split_y"] * height / image["height"])
        points += [(edge_x + delta, height // 4) for delta in (-2, -1, 0, 1, 2)]
        points += [(width // 4, edge_y + delta) for delta in (-2, -1, 0, 1, 2)]
    samples = []
    for x, y in points:
        patch_row, local_y = divmod(y, patch_size)
        patch_column, local_x = divmod(x, patch_size)
        block_y, inner_y = divmod(patch_row, merge_size)
        block_x, inner_x = divmod(patch_column, merge_size)
        patch = ((block_y * (columns // merge_size) + block_x) * merge_size**2
                 + inner_y * merge_size + inner_x)
        for channel in range(3):
            for frame in range(temporal_size):
                offset = ((channel * temporal_size + frame) * patch_size + local_y) * patch_size + local_x
                value = float(rgb[patch, offset])
                if (x, y) in corners:
                    palette = image.get("quadrants", [image["rgb"]] * 4)
                    want = 2.0 * palette[corners[x, y]][channel] / 255.0 - 1.0
                    if not math.isclose(value, want, rel_tol=0, abs_tol=1e-6):
                        raise RuntimeError("reference patch order or normalized corner color differs")
                samples.append({"patch": patch, "offset": offset, "value": value})
    # Existing Qwen35 tests compare this native PIL-style resizer to torchvision BICUBIC.
    sample_abs_tolerance = 0.02 if "quadrants" in image else 1e-5
    return {
        "image_grid_thw": grid, "image_processor_samples": samples,
        "image_processor_sample_abs_tolerance": sample_abs_tolerance,
        "image_processor_tolerance_source": (
            "crates/llm/candle-llm/src/image.rs::tests::end_to_end_within_resampler_tolerance"
            if "quadrants" in image
            else "crates/llm/candle-llm/src/image.rs::tests::patchify_layout_matches_reference"
        ),
        "image_rgb_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "image_pixel_values_f32_sha256": hashlib.sha256(rgb.numpy().astype("<f4", copy=False).tobytes()).hexdigest(),
        "image_pixel_values_shape": list(rgb.shape), "image_resized_wh": [width, height],
    }


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
    import torchvision
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
    image_processor = processor.image_processor
    patch_size, temporal_size, merge_size = (image_processor.patch_size,
                                              image_processor.temporal_patch_size,
                                              image_processor.merge_size)
    if (patch_size, temporal_size, merge_size) != (16, 2, 2):
        raise RuntimeError("pinned SenseNova processor patch settings differ")
    processor_binding = {
        "processor_class": type(processor).__name__,
        "image_processor_class": type(image_processor).__name__,
        "size": dict(image_processor.size),
        "resample": int(image_processor.resample),
        "rescale_factor": image_processor.rescale_factor,
        "image_mean": image_processor.image_mean, "image_std": image_processor.image_std,
        "patch_size": patch_size, "temporal_patch_size": temporal_size, "merge_size": merge_size,
        "torchvision": torchvision.__version__,
    }
    loaded_parameter_dtypes = {key: str(parameter.dtype).removeprefix("torch.") for key, parameter in {
        "model.language_model.layers.0.linear_attn.A_log": layers[0].linear_attn.A_log,
        "model.language_model.layers.0.linear_attn.dt_bias": layers[0].linear_attn.dt_bias,
        "model.language_model.layers.4.mhc_attn.weight": layers[4].mhc_attn.weight,
        "model.language_model.layers.4.mhc_attn.bias": layers[4].mhc_attn.bias,
        "model.language_model.layers.4.mhc_ffn.weight": layers[4].mhc_ffn.weight,
        "model.language_model.layers.4.mhc_ffn.bias": layers[4].mhc_ffn.bias,
    }.items()}
    cases = []
    with torch.inference_mode():
        for name, prompt, image, expected_answer in FIXTURES:
            content = [{"type": "text", "text": prompt}]
            processor_kwargs = {}
            if image is not None:
                content.insert(0, {"type": "image"})
                pil_image = fixture_image(image, Image)
                processor_kwargs["images"] = [pil_image]
            rendered = processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            inputs = processor(text=[rendered], return_tensors="pt", **processor_kwargs)
            vision = {"image_grid_thw": None, "image_processor_samples": []}
            if image is not None:
                vision = processor_evidence(image, pil_image.tobytes(), inputs, patch_size, temporal_size, merge_size)
            inputs = inputs.to("cuda:0")
            model.set_attn_implementation("eager")
            baseline, chosen = run_case(model, inputs, torch, cached=True, expected_plan=plan)
            stop_ids = model.generation_config.eos_token_id
            if isinstance(stop_ids, int):
                stop_ids = [stop_ids]
            first_end = next((i for i, token in enumerate(chosen) if token in stop_ids), len(chosen))
            completion = processor.tokenizer.decode(chosen[:first_end], skip_special_tokens=True)
            if completion.strip().strip(".!").lower() != expected_answer:
                raise RuntimeError(f"upstream reference {name} did not produce {expected_answer!r}: {completion!r}")
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
                "expected_answer": expected_answer, "reference_completion": completion,
                **vision,
                "thinking_prompt": processor.apply_chat_template(
                    [{"role": "user", "content": content}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=True,
                ),
            })
    payload = {
        "schema_version": 1, "hf_revision": snapshot.name,
        "source_inventory_sha256": inventory_sha,
        "comparison": "BF16 PyTorch CUDA eager cached, eager full replay, SDPA cached",
        "processor_bindings": processor_binding, "loaded_parameter_dtypes": loaded_parameter_dtypes,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(), "pillow": pillow_version,
        "cases": cases,
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
