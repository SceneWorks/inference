#!/usr/bin/env python3
"""Bind pinned assets for eight exact bounded native training cases.

Each input is a `release/real-weight-models.toml` row in the runner's Hugging Face hub cache; a
missing or incomplete one is first downloaded as that row's file slice, a complete one is reused
with no transfer. Production readers inside the ONE manual invocation still authenticate/load
actual inputs; paths are not runtime proof.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path, PureWindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.ci.proof_asset_snapshots import Input, manifest_bindings, manifest_keys, materialize  # noqa: E402

ANIMA = Input("anima-base-train-proof", "split_files")
MAGE = Input("mage-flow-base-bf16-train-proof", "bf16")
MAGE_SHARED = "mage-flow-components-bf16-train-proof"
MOSS = Input("moss-image-base-transformer-proof")
MOSS_SHARED = Input("moss-image-sft-shared-proof")
FLUX3 = Input("flux3-image-klein-base-proof")
IRIS = Input("iris-3b-generation-proof")
IRIS_TE = Input("qwen3-vl-4b-instruct-cuda")
KREA = Input("krea-2-turbo-control-train-proof")

# Exact /integration/ spelling is the existing proof identity protocol.
BESPOKE_TRAINING_PROOFS = {
    "candle-gen-anima/integration/trainer_cuda_smoke::anima_bounded_acceleration_acceptance": {"ANIMA_TRAIN_SNAPSHOT": ANIMA},
    "candle-gen-mage/integration/trainer_cuda_smoke::mage_bounded_acceleration_acceptance": {"MAGE_BASE_TRAIN_SNAPSHOT": MAGE, "MAGE_TRAIN_TEXT_ENCODER": Input(MAGE_SHARED, "bf16/text_encoder"), "MAGE_TRAIN_VAE": Input(MAGE_SHARED, "bf16/vae")},
    "candle-gen-moss-image/integration/real_weights::moss_bounded_acceleration_acceptance": {"MOSS_BASE_SNAPSHOT": MOSS, "MOSS_SHARED_SNAPSHOT": MOSS_SHARED},
    "candle-gen-flux3/integration/real_weights::flux3_bounded_acceleration_acceptance": {"FLUX3_BASE_PACKAGE_DIR": FLUX3},
    "candle-gen-iris/integration/train_real_weights::iris_generation_bounded_acceleration_acceptance": {"IRIS_WEIGHTS_DIR": IRIS, "IRIS_TEXT_ENCODER_DIR": IRIS_TE},
    "candle-gen-iris/integration/depth_train_real_weights::iris_depth_bounded_acceleration_acceptance": {"IRIS_DEPTH_DIR": Input("iris-3b-depth-proof", "depth")},
    "candle-gen-iris/integration/restoration_train_real_weights::iris_restoration_bounded_acceleration_acceptance": {
        "IRIS_UPSCALER_DIR": Input("iris-3b-upscaler-proof", "upscaler"),
        "IRIS_LPIPS_VGG16_DIR": Input("vgg16-tv-in1k-cuda"),
        "IRIS_LPIPS_LIN_DIR": Input("lpips-vgg-lin-cuda"),
        "IRIS_CRITIC_TRUNK_DIR": Input("openclip-convnext-xxlarge-cuda")},
    "candle-gen-krea/integration/trainer_e2e::krea_control_bounded_acceleration_acceptance": {"KREA_CONTROL_BASE_DIR": KREA},
}
MANIFEST_BINDINGS = manifest_bindings(BESPOKE_TRAINING_PROOFS)
MANIFEST_KEYS = manifest_keys(BESPOKE_TRAINING_PROOFS)

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

def exact_bindings(rows: list[dict], backend: str) -> dict[str, Input] | None:
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


def owned_runtime_root(identity: str, environ: dict | None = None) -> Path:
    env = os.environ if environ is None else environ
    run = env["GITHUB_RUN_ID"] + "-" + env["GITHUB_RUN_ATTEMPT"]
    if not re.fullmatch(r"[0-9]+-[0-9]+", run): raise ValueError("invalid runner run identity")
    work = Path(env["DEV_HOST_JOB_ROOT"])
    parent = Path(env["DEV_HOST_ROOT"]) / "t"
    if work.is_symlink() or work.resolve().parent != parent.resolve() or not re.fullmatch(r"[a-f0-9]{8}", work.name):
        raise ValueError("bespoke runtime must be in a dev-host t/ entry")
    owner = json.loads((work / ".actions-owner.json").read_text(encoding="utf-8"))
    for field, key in (("repository", "GITHUB_REPOSITORY"), ("run", "GITHUB_RUN_ID"), ("attempt", "GITHUB_RUN_ATTEMPT"), ("job", "GITHUB_JOB"), ("runner", "RUNNER_NAME")):
        if owner.get(field) != env[key]: raise ValueError("bespoke runtime requires this exact job's managed temp directory")
    if PureWindowsPath(env.get("HF_HOME", "")).as_posix().lower() != "e:/huggingface":
        raise ValueError("Windows proof model downloads must use E:/huggingface")
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if env.get(key) and PureWindowsPath(env[key]).as_posix().lower() != "e:/huggingface/hub":
            raise ValueError("Windows proof model downloads must use E:/huggingface/hub")
    return work / "model-proofs" / run / identity.split("::")[-1]

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
    # This directory is deliberately OUTSIDE artifact publication and the checkout.
    # Full Iris exports/recovery states and ControlNet weights are not LoRA files.
    runtime = owned_runtime_root(identities[0])
    # Downloads stay in E:\huggingface; only generated proof state belongs in job storage.
    bindings = materialize(BESPOKE_TRAINING_PROOFS[identities[0]])
    validate_paths(bindings)
    runtime.mkdir(parents=True, exist_ok=False)
    (runtime / ".sc-26850-owned-runtime").write_text(identities[0] + "\n", encoding="utf-8")
    with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as out:
        for name, path in {**bindings, "BESPOKE_RUNTIME_DIR": runtime}.items(): out.write(f"{name}={path}\n")
    print(json.dumps({"case":identities[0],"configuration":"fixed_cuda","installed_paths":{k:str(v) for k,v in bindings.items()},"owned_runtime_directory":str(runtime),"claim":"pinned runner-cache asset binding; actual production reader authentication belongs to the one manual invocation"}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
