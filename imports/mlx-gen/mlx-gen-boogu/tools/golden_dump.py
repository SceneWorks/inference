"""Dump golden tensors from the reference Boogu pipeline for MLX parity (E2/E3).

Hooks `processor.apply_chat_template` (captures the tokenized instruction) and
`transformer.forward` (captures the conditioning + latent I/O on the first call), runs a
tiny 1-step Base generation to trigger one forward, and writes everything to
reference/goldens/boogu_golden.safetensors (+ a meta.json).

Usage: PYTHONPATH=/tmp/boogu-ref python golden_dump.py
"""
import os, json, glob
os.environ.setdefault("device", "mps")
import torch
from safetensors.torch import save_file
from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline
BooguImagePipeline._validate_device_format = lambda self, *a, **k: None

SNAP = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Boogu--Boogu-Image-0.1-Base/snapshots/*/")))[0]
OUT = os.path.expanduser("~/Repos/mlx-gen-wt-boogu/reference/goldens")
os.makedirs(OUT, exist_ok=True)
PROMPT = "a red apple on a wooden table"  # short + deterministic

def cpu(t):
    return t.detach().to(torch.float32).cpu().contiguous()

pipe = BooguImagePipeline.from_pretrained(SNAP, torch_dtype=torch.bfloat16, trust_remote_code=True)
pipe.to("mps")

cap = {}      # captured tensors
meta = {}

# 1. Capture the tokenized instruction.
_orig_act = pipe.processor.apply_chat_template
def wrap_act(*a, **k):
    out = _orig_act(*a, **k)
    if "tok_input_ids" not in cap and hasattr(out, "get") and out.get("input_ids") is not None:
        cap["tok_input_ids"] = out["input_ids"].detach().cpu().to(torch.int32).contiguous()
        cap["tok_attention_mask"] = out["attention_mask"].detach().cpu().to(torch.int32).contiguous()
    return out
pipe.processor.apply_chat_template = wrap_act

# 2. Capture the DiT forward I/O on the first call, then abort the run.
class _Stop(Exception):
    pass

_orig_fwd = pipe.transformer.forward
def wrap_fwd(hidden_states, timestep, instruction_hidden_states, freqs_cis,
             instruction_attention_mask, ref_image_hidden_states=None, **kw):
    # instruction_hidden_states is the raw Qwen3-VL output (E2 golden); may be tensor or list.
    ih = instruction_hidden_states
    if isinstance(ih, (list, tuple)):
        ih = ih[-1]
    cap["instruction_hidden_states"] = cpu(ih)               # [B, L, 4096]  -> E2 golden
    cap["timestep"] = cpu(timestep.reshape(-1))
    hs0 = hidden_states[0] if isinstance(hidden_states, (list, tuple)) else hidden_states
    cap["dit_in_latent_chw"] = cpu(hs0)                      # [C,H,W] patchify input -> E3
    cap["instruction_attention_mask"] = instruction_attention_mask.detach().cpu().to(torch.int32).contiguous()
    out = _orig_fwd(hidden_states, timestep, instruction_hidden_states, freqs_cis,
                    instruction_attention_mask, ref_image_hidden_states=ref_image_hidden_states, **kw)
    o0 = out[0] if isinstance(out, (list, tuple)) else out
    cap["dit_out_velocity_chw"] = cpu(o0)                    # [C,H,W] velocity -> E3
    meta["dit_in_shape"] = list(hs0.shape)
    meta["instr_hidden_shape"] = list(ih.shape)
    raise _Stop()
pipe.transformer.forward = wrap_fwd

try:
    pipe(instruction=PROMPT, negative_instruction="", height=256, width=256,
         max_input_image_pixels=256 * 256, max_input_image_side_length=512,
         num_inference_steps=1, text_guidance_scale=4.0, device="mps",
         generator=torch.Generator("cpu").manual_seed(0))
except _Stop:
    print("[golden] captured first DiT forward; aborting run.")

save_file(cap, os.path.join(OUT, "boogu_golden.safetensors"))
meta["prompt"] = PROMPT
meta["keys"] = {k: list(v.shape) for k, v in cap.items()}
with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("[golden] saved:", json.dumps(meta["keys"], indent=2))
