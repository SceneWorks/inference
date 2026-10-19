"""CPU-only frozen-teacher precision oracle on the COMMITTED SYNTHETIC Qwen fixture.

Uses the direct get_qwen3vl_prompt_embeds forward contract from diffusers7ecb,
including Module.to(selected dtype), explicit pad positions and selected-state cast.
No production Python, network/model downloads, GPU calls or real weights.
"""
from pathlib import Path
import importlib.metadata
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _flux3_common import FIXTURE_DIR, SNAPSHOT, OUTPUT_LAYERS
import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen3VLForConditionalGeneration

torch.set_num_threads(2)
base = load_file(str(FIXTURE_DIR / "flux3_text_golden.safetensors"))
windows = {}
for name in ["short", "exact", "long", "empty"]:
    windows[name] = {field: base[f"{name}/{field}"].clone() for field in ["input_ids", "mask", "positions"]}
windows["long_cap16"] = {field: values[:16].clone() for field, values in windows["long"].items()}
windows["rotary_probe"] = {"input_ids": windows["short"]["input_ids"][:4].clone(), "mask": torch.ones(4,dtype=torch.int32), "positions": torch.tensor([0,127,255,511],dtype=torch.int32)}
real = int(windows["empty"]["mask"].sum())
length = windows["short"]["input_ids"].numel()
empty_ids = torch.full((length,), int(base["pad_id"][0]), dtype=torch.int32)
empty_ids[:real] = windows["empty"]["input_ids"][:real]
empty_mask = torch.zeros(length, dtype=torch.int32)
empty_mask[:real] = 1
empty_positions = torch.arange(length, dtype=torch.int32).masked_fill(empty_mask == 0, 1)
windows["empty_at_short_window"] = {"input_ids": empty_ids, "mask": empty_mask, "positions": empty_positions}
out = {}
observed = {}
for label, dtype in [("f32", torch.float32), ("bf16", torch.bfloat16), ("f16", torch.float16)]:
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(SNAPSHOT / "text_encoder"), torch_dtype=dtype, local_files_only=True)
    # The pinned trainer explicitly calls .to(dtype), beyond from_pretrained's dtype.
    model.to(device="cpu", dtype=dtype).requires_grad_(False).eval()
    observed[label] = {"parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}), "rotary_buffers": {name: str(value.dtype) for name, value in model.named_buffers() if "inv_freq" in name}}
    for name, window in windows.items():
        with torch.no_grad():
            result = model(input_ids=window["input_ids"][None].long(), attention_mask=window["mask"][None].long(), position_ids=window["positions"][None].long(), output_hidden_states=True, use_cache=False)
            stacked = torch.stack([result.hidden_states[k] for k in OUTPUT_LAYERS], dim=1).to(dtype)
            b, n, s, d = stacked.shape
            tokens = stacked.permute(0, 2, 1, 3).reshape(b, s, n*d)[0]
        assert tokens.dtype == dtype
        out[f"{label}/{name}/tokens"] = tokens.float().contiguous()
        for field, values in window.items():
            out[f"{label}/{name}/{field}"] = values.contiguous().clone()
    for name, values in model.named_buffers():
        if "inv_freq" in name:
            out[f"{label}/rotary/{name}"] = values.float().contiguous().clone()
    del model

def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"

metadata = {
    "fixture": "SYNTHETIC miniature Qwen3-VL, NOT actual BFL/Qwen model weights",
    "training_reference": "7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941",
    "oracle_scope": "direct frozen teacher on fixed pre-tokenized windows; no full datasets/Trainer execution",
    "reference_environment": json.dumps({"python":sys.version, **{name:version(name) for name in ["torch","diffusers","transformers","datasets","Pillow","accelerate","safetensors"]}}, sort_keys=True),
    "observed_model_dtypes": json.dumps(observed, sort_keys=True),
    "precision_contract": "source selected weight dtype, then explicit Module.to(dtype), mask pad positions=1, selected hidden stack cast to dtype",
}
save_file(out, str(FIXTURE_DIR / "flux3_text_training_golden.safetensors"), metadata=metadata)
print("CPU synthetic frozen teacher: F32/BF16/F16, native windows + cap override + matched-window empty caption")
print(metadata["observed_model_dtypes"])
