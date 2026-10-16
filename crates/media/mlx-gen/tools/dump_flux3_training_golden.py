"""Synthetic CPU-only gradient/AdamW oracle for sc-26240. No production training route.

Run with -I, FLUX3_SRC pointing to the exact hash-verified frozen BFL tree,
and FLUX3_TRAINING_REF pointing to the two exact frozen diffusers scripts.
The forward is frozen BFL45e57c9; objective/discrete shift6.93/AdamW/default
plain-LoRA targets follow diffusers7ecb1e1b. These are NOT actual BFL weights.
"""
from pathlib import Path
import hashlib
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _flux3_common import import_upstream, patch_dit_constants, FIXTURE_DIR, SNAPSHOT, FLUX3_COMMIT
import_upstream()
import torch
from safetensors.torch import load_file, save_file
import flux3.dit as dit_module

TRAINING_REF = "7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941"
from _paths import require_env
training_source = Path(require_env("FLUX3_TRAINING_REF", "directory containing the two frozen diffusers7ecb1e1b training scripts"))
training_hashes = {
    "train_dreambooth_lora_flux3_klein.py": "89cc2ff1af1909c1652deefdee7a60e589c3d4e7169d22ab7e05e3ca0c49be72",
    "train_dreambooth_lora_flux3_klein_img2img.py": "c33420f2f487476474ae12e5490655874a6f5226712b2e74f64e19dce6a630c9",
}
for name, expected in training_hashes.items():
    actual = hashlib.sha256((training_source / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"frozen training source mismatch: {name}: {actual} != {expected}")
torch.set_num_threads(2)
patch_dit_constants(dit_module)
model = dit_module.DiT(flow_map=False).eval()
base = {k: v.float() for k, v in load_file(str(SNAPSHOT / "model.safetensors")).items() if not k.startswith("ref_time_in.")}
model.load_state_dict(base, strict=True)
model.requires_grad_(False)
text = load_file(str(FIXTURE_DIR / "flux3_text_golden.safetensors"))
inputs = load_file(str(FIXTURE_DIR / "flux3_dit_golden.safetensors"))
clean = inputs["single/x"].float() * 0.4
noise = torch.cos(torch.arange(clean.numel()).reshape(clean.shape).float() * 0.13)
# Pinned _img2img training keeps paired images in the SAME bucket/geometry and
# places the reference at L=-1, T=0 (also S3 inference's FIRST-reference id law).
# This replaces the former arbitrary T1/L0 probe; paired preprocessing is training-specific.
reference = clean.clone() * 0.7
reference_ids = inputs["single/image_ids"].clone()
reference_ids[:, 0] = 0
reference_ids[:, 3] = -1
rank, alpha = 3, 1.5
factors, handles, out = {}, [], {}
for name, layer in model.named_modules():
    if not isinstance(layer, torch.nn.Linear) or not name.endswith((".q_proj", ".k_proj", ".v_proj", ".attn_out", ".mlp_in", ".mlp_out")):
        continue
    down = torch.nn.Parameter(torch.sin(torch.arange(rank * layer.in_features).reshape(rank, layer.in_features).float() * 0.17) * (0.05 / rank))
    up = torch.nn.Parameter(torch.zeros(layer.out_features, rank))
    factors[name + ".lora_A.weight"] = down
    factors[name + ".lora_B.weight"] = up
    out["initial/" + name + ".lora_A.weight"] = down.detach().clone()
    out["initial/" + name + ".lora_B.weight"] = up.detach().clone()
    def hook(_module, args, result, down=down, up=up):
        return result + (args[0] @ down.T @ up.T) * (alpha / rank)
    handles.append(layer.register_forward_hook(hook))
optimizer = torch.optim.AdamW(list(factors.values()), lr=0.0003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
out.update({"clean": clean, "noise": noise, "reference": reference, "reference_ids": reference_ids, "text": text["short/tokens"].float(), "text_ids": text["short/ids"], "image_ids": inputs["single/image_ids"]})
for step, index in enumerate([137, 721]):
    # Explicit source discrete schedule: scheduler init descending1..1000, fixed shift6.93.
    s = torch.tensor((1000 - index) / 1000, dtype=torch.float32)
    sigma = 6.93 * s / (1 + (6.93 - 1) * s)
    noisy = (1 - sigma) * clean + sigma * noise
    data = {"image": noisy, "image_cond": reference, "text": out["text"]}
    ids = {"image": out["image_ids"], "image_cond": reference_ids, "text": out["text_ids"]}
    times = {m: torch.full((v.shape[0],), sigma.item() if m == "image" else 0.0) for m, v in data.items()}
    packed = dit_module.Packed(data, ids, times, {m: [(0, v.shape[0])] for m, v in data.items()})
    prediction = model(packed)
    loss = ((prediction.float() - (noise - clean).float()) ** 2).mean()
    out[f"step{step}/sigma"] = sigma.reshape(1)
    out[f"step{step}/prediction"] = prediction.detach().clone()
    out[f"step{step}/loss"] = loss.detach().reshape(1)
    loss.backward()
    for name, parameter in factors.items():
        assert parameter.grad is not None, name
        out[f"step{step}/gradient/{name}"] = parameter.grad.detach().clone()
    if step == 1:
        for name, parameter in factors.items():
            if name.endswith("lora_A.weight"):
                assert parameter.grad.abs().max() > 0, name
    optimizer.step()
    for name, parameter in factors.items():
        out[f"step{step}/factor/{name}"] = parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
for handle in handles:
    handle.remove()
assert all(not parameter.requires_grad for parameter in model.parameters())
save_file({k: v.detach().contiguous().float() for k, v in out.items()}, str(FIXTURE_DIR / "flux3_training_golden.safetensors"), metadata={"fixture": "synthetic tiny geometry, NOT actual BFL weights", "training_reference": TRAINING_REF, "training_img2img_source_sha256": training_hashes["train_dreambooth_lora_flux3_klein_img2img.py"], "conditioning_position": "T0/H0.5/W0.5/L-1, paired same-grid cached synthetic latents", "initialization": "controlled synthetic sinusoidal A, zero B; not a PEFT RNG-stream identity claim", "forward_reference": FLUX3_COMMIT, "optimizer": "torch AdamW CPU lr0.0003 beta0.9,0.999 eps1e-8 wd0.0001", "rank": str(rank), "alpha": str(alpha), "precision": "f32", "steps": "2"})
print(f"Synthetic CPU oracle: {len(factors)} factors, two nonzero-gradient AdamW steps; no real weights")
