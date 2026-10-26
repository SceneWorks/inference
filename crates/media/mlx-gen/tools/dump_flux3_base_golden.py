"""CPU-only base/CFG/additive-LoRA oracle against frozen BFL45e57c9 (sc-26239).
Reads existing tiny snapshot/text fixtures; never overwrites distilled fixtures.
FLUX3_SRC=<verified frozen tree> python -I tools/dump_flux3_base_golden.py
No real weights, network, accelerator initialization, or production Python path.
"""
from pathlib import Path
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _flux3_common import import_upstream, patch_dit_constants, FIXTURE_DIR, SNAPSHOT, FLUX3_COMMIT
import_upstream()
import torch
from safetensors.torch import load_file, save_file
import flux3.dit as dit_module
import flux3.pipeline as pipeline
import flux3.sampling as sampling

patch_dit_constants(dit_module)
torch.set_num_threads(4)
weights = {k: v.float() for k, v in load_file(str(SNAPSHOT / "model.safetensors")).items() if not k.startswith("ref_time_in.")}
model = dit_module.DiT(flow_map=False).eval()
model.load_state_dict(weights, strict=True)
base_snapshot = FIXTURE_DIR / "tiny-base-snapshot"
base_snapshot.mkdir(exist_ok=True)
save_file({k: v.to(torch.bfloat16).contiguous() for k,v in weights.items()}, str(base_snapshot / "model.safetensors"), metadata={"upstream_code_revision": FLUX3_COMMIT, "fixture": "tiny geometry, not real base weights"})
config = json.loads((SNAPSHOT / "config.json").read_text(encoding="utf8"))
config["model"] = "flux-3-image-klein-base-tiny-fixture"
config["dit"]["flow_map"] = False
config["sampler"]["steps"] = 50
config["sampler"]["guidance"] = 3.5
(base_snapshot / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf8")
text = load_file(str(FIXTURE_DIR / "flux3_text_golden.safetensors"))
inputs = load_file(str(FIXTURE_DIR / "flux3_dit_golden.safetensors"))
x, ids = inputs["single/x"], inputs["single/image_ids"]
pos = (text["short/tokens"], text["short/ids"])
neg = (text["empty/tokens"], text["empty/ids"])
# A second nonempty negative caption makes the unconditional row load-bearing.
negative_alt = (text["long/tokens"], text["long/ids"])
schedule = sampling.timesteps(4, 6.93)

def pack(negative):
    p = pipeline.pack([[negative, None, (x.clone(), ids)], [pos, None, (x.clone(), ids)]])
    p.data = {m: v.float() for m, v in p.data.items()}
    p.timesteps = {m: v.float() for m, v in p.timesteps.items()}
    return p

out = {"noise": x, "image_ids": ids, "schedule": schedule}
with torch.inference_mode():
    for guidance in [0.0, 1.0, 3.5]:
        out[f"cfg_{guidance}"] = sampling.denoise(model, pack(neg), schedule, guidance)
    out["negative_alt"] = sampling.denoise(model, pack(negative_alt), schedule, 3.5)
# Independently apply LoRA via native module hooks, not via any Rust-side installer.
generator = torch.Generator().manual_seed(26239)
adapter = {}
alpha, rank, scale = 6.0, 2, -0.25
handles = []
for name in ["content_mode_blocks.image.0.q_proj", "content_mode_blocks.image_cond.1.k_proj", "txt_mode_blocks.0.v_proj", "single_blocks.0.attn_out", "single_blocks.1.mlp_in", "single_blocks.2.mlp_out"]:
    layer = model.get_submodule(name)
    down = 0.15 * torch.randn((rank, layer.in_features), generator=generator)
    up = 0.15 * torch.randn((layer.out_features, rank), generator=generator)
    diffusers_name = name.replace("content_mode_blocks.", "mode_blocks.").replace("single_blocks.", "single_transformer_blocks.")
    for a, b in [(".q_proj", ".attn.to_q"), (".k_proj", ".attn.to_k"), (".v_proj", ".attn.to_v"), (".attn_out", ".attn.to_out.0"), (".mlp_in", ".ff.linear_in"), (".mlp_out", ".ff.linear_out")]:
        diffusers_name = diffusers_name.replace(a, b)
    adapter[f"transformer.{diffusers_name}.lora_A.weight"] = down
    adapter[f"transformer.{diffusers_name}.lora_B.weight"] = up
    def hook(_module, args, result, down=down, up=up):
        return result + (args[0].float() @ down.T @ up.T) * (scale * alpha / rank)
    handles.append(layer.register_forward_hook(hook))
with torch.inference_mode():
    out["adapter_on"] = sampling.denoise(model, pack(neg), schedule, 3.5)
for handle in handles:
    handle.remove()
assert not torch.allclose(out["adapter_on"], out["cfg_3.5"], atol=1e-5)
assert not torch.allclose(out["negative_alt"], out["cfg_3.5"], atol=1e-5)
metadata = {"upstream_code_revision": FLUX3_COMMIT, "base_model": "black-forest-labs/flux-3-image-klein-base", "base_revision": "1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4", "lora_alpha": str(alpha), "oracle": "frozen BFL DiT(flow_map=False), CPU FP32, additive hooks, no real weights", "scale": str(scale)}
save_file({k: v.contiguous().float() for k,v in out.items()}, str(FIXTURE_DIR / "flux3_base_golden.safetensors"), metadata=metadata)
save_file(adapter, str(FIXTURE_DIR / "flux3_base_lora.safetensors"), metadata=metadata)
print("base CFG0/1/3.5 + explicit negative + six-family additive LoRA fixtures written")
