#!/usr/bin/env python3
"""Bind provisioned installed assets for eight exact bounded native training cases.

This setup reads file metadata only. Production readers inside the ONE manual
invocation still authenticate/load actual inputs; paths are not runtime proof.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HUB = Path("E:/huggingface/hub")
def snapshot(repository: str, revision: str, suffix: str = "") -> Path:
    return HUB / ("models--" + repository.replace("/", "--")) / "snapshots" / revision / suffix

ANIMA = snapshot("circlestone-labs/Anima", "53eec3898af698b2cf2a11379021fc9c5465d228", "split_files")
MAGE = snapshot("SceneWorks/Mage-Flow-Base", "d642341926fcdb450c17c8fbda03759e8f731c9b", "bf16")
MAGE_SHARED = snapshot("SceneWorks/Mage-Flow-Components-mlx", "c936de2a107ee8d0869137e73943f6414f23adaa", "bf16")
MOSS = snapshot("OpenMOSS-Team/MOSS-Image-Base", "1ceede80badddd6a7812069619edaa5633587192")
MOSS_SHARED = snapshot("OpenMOSS-Team/MOSS-Image-SFT", "fc5de0d56fc910cbca6f6625a70f5f582b26deb1")
FLUX3 = snapshot("black-forest-labs/flux-3-image-klein-base", "1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4")
IRIS = snapshot("speridlabs/iris-3b", "7445443349bc9abe3c96f01ff793e2098ca012b3")
IRIS_TE = snapshot("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17")
KREA = snapshot("krea/Krea-2-Turbo", "1161245028ef398cd0a951101b2bbf486464f841")

# Exact /integration/ spelling is the existing proof identity protocol.
BESPOKE_TRAINING_PROOFS = {
    "candle-gen-anima/integration/trainer_cuda_smoke::anima_bounded_acceleration_acceptance": {"ANIMA_TRAIN_SNAPSHOT": ANIMA},
    "candle-gen-mage/integration/trainer_cuda_smoke::mage_bounded_acceleration_acceptance": {"MAGE_BASE_TRAIN_SNAPSHOT": MAGE, "MAGE_TRAIN_TEXT_ENCODER": MAGE_SHARED / "text_encoder", "MAGE_TRAIN_VAE": MAGE_SHARED / "vae"},
    "candle-gen-moss-image/integration/real_weights::moss_bounded_acceleration_acceptance": {"MOSS_BASE_SNAPSHOT": MOSS, "MOSS_SHARED_SNAPSHOT": MOSS_SHARED},
    "candle-gen-flux3/integration/real_weights::flux3_bounded_acceleration_acceptance": {"FLUX3_BASE_PACKAGE_DIR": FLUX3},
    "candle-gen-iris/integration/train_real_weights::iris_generation_bounded_acceleration_acceptance": {"IRIS_WEIGHTS_DIR": IRIS, "IRIS_TEXT_ENCODER_DIR": IRIS_TE},
    "candle-gen-iris/integration/depth_train_real_weights::iris_depth_bounded_acceleration_acceptance": {"IRIS_DEPTH_DIR": IRIS / "depth"},
    "candle-gen-iris/integration/restoration_train_real_weights::iris_restoration_bounded_acceleration_acceptance": {
        "IRIS_UPSCALER_DIR": IRIS / "upscaler",
        "IRIS_LPIPS_VGG16_DIR": snapshot("timm/vgg16.tv_in1k", "b8d8aa2dd860af9233c8c67385a8097fd6c35d3f"),
        "IRIS_LPIPS_LIN_DIR": snapshot("vivym/lpips", "270571f1fb2a2c4c5f920cec96cd87838c345982"),
        "IRIS_CRITIC_TRUNK_DIR": snapshot("laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup", "9f3e8ee3f383c672388d9178afe70af9e63ac9df")},
    "candle-gen-krea/integration/trainer_e2e::krea_control_bounded_acceleration_acceptance": {"KREA_CONTROL_BASE_DIR": KREA},
}

REQUIRED_FILES = {
    "ANIMA_TRAIN_SNAPSHOT": ("diffusion_models/anima-base-v1.0.safetensors", "text_encoders/qwen_3_06b_base.safetensors", "vae/qwen_image_vae.safetensors"),
    "MAGE_BASE_TRAIN_SNAPSHOT": ("transformer/config.json", "transformer/diffusion_pytorch_model.safetensors"),
    "MAGE_TRAIN_TEXT_ENCODER": ("config.json", "tokenizer.json"),
    "MAGE_TRAIN_VAE": ("config.json", "diffusion_pytorch_model.safetensors"),
    "MOSS_BASE_SNAPSHOT": ("transformer/config.json", "transformer/diffusion_pytorch_model.safetensors.index.json"),
    "MOSS_SHARED_SNAPSHOT": ("text_encoder/config.json", "tokenizer/tokenizer.json", "vae/config.json"),
    "FLUX3_BASE_PACKAGE_DIR": ("config.json", "model.safetensors", "ae.safetensors"),
    "IRIS_WEIGHTS_DIR": ("config.yaml", "model.safetensors"),
    "IRIS_TEXT_ENCODER_DIR": ("config.json", "tokenizer.json", "model.safetensors.index.json"),
    "IRIS_DEPTH_DIR": ("config.yaml", "model.safetensors", "empty_prompt.safetensors"),
    "IRIS_UPSCALER_DIR": ("config.yaml", "model.safetensors", "empty_prompt.safetensors"),
    "IRIS_LPIPS_VGG16_DIR": ("model.safetensors",), "IRIS_LPIPS_LIN_DIR": ("vgg_lpips_linear.safetensors",),
    "IRIS_CRITIC_TRUNK_DIR": ("open_clip_pytorch_model.bin",),
    "KREA_CONTROL_BASE_DIR": ("model_index.json", "transformer/config.json", "text_encoder/config.json", "tokenizer/tokenizer.json", "vae/config.json", "vae/diffusion_pytorch_model.safetensors"),
}

def exact_bindings(rows: list[dict], backend: str) -> dict[str, Path] | None:
    from scripts.ci.training_comparison import is_recipe
    if backend != "candle-cuda" or len(rows) != 1:
        return None
    row = rows[0]
    key = row["package"] + "/" + row["target"] + "/" + row["fqn"]
    selected = row.get("selected_configuration", {})
    if row.get("target_kind") != "test" or (selected.get("name") != "fixed_cuda" and not is_recipe(selected.get("name"))):
        return None
    return BESPOKE_TRAINING_PROOFS.get(key)

def validate_paths(bindings: dict[str, Path]) -> None:
    for name, root in bindings.items():
        if not root.is_dir(): raise ValueError(f"missing pinned installed directory {name}: {root}")
        for relative in REQUIRED_FILES[name]:
            path = root / relative
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing/empty pinned installed asset {name}: {path}")

def main() -> int:
    from scripts.ci.training_comparison import is_recipe
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    identities = plan.get("bindings", [])
    configs = plan.get("configurations", {})
    if (plan.get("backend") != "candle-cuda" or plan.get("selected") != 1
            or len(identities) != 1 or identities[0] not in BESPOKE_TRAINING_PROOFS
            or (configs.get(identities[0], {}).get("name") != "fixed_cuda" and not is_recipe(configs.get(identities[0], {}).get("name")))):
        raise ValueError("bespoke asset setup requires one exact reviewed fixed_cuda case")
    bindings = BESPOKE_TRAINING_PROOFS[identities[0]]
    validate_paths(bindings)
    # This directory is deliberately OUTSIDE artifact publication and the checkout.
    # Full Iris exports/recovery states and ControlNet weights are not LoRA files.
    run = os.environ["GITHUB_RUN_ID"] + "-" + os.environ["GITHUB_RUN_ATTEMPT"]
    if not re.fullmatch(r"[0-9]+-[0-9]+", run): raise ValueError("invalid runner run identity")
    runtime = Path("E:/sceneworks-runtime/model-proofs/sc-26850") / run / identities[0].split("::")[-1]
    runtime.mkdir(parents=True, exist_ok=False)
    (runtime / ".sc-26850-owned-runtime").write_text(identities[0] + "\n", encoding="utf-8")
    with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as out:
        for name, path in {**bindings, "BESPOKE_RUNTIME_DIR": runtime}.items(): out.write(f"{name}={path}\n")
    print(json.dumps({"case":identities[0],"configuration":"fixed_cuda","installed_paths":{k:str(v) for k,v in bindings.items()},"owned_runtime_directory":str(runtime),"claim":"metadata-only asset binding; actual production reader authentication belongs to the one manual invocation"}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
