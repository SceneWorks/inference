"""Bind pinned assets for exact bounded CUDA proofs, downloading a missing input into the runner's
Hugging Face hub cache first (no compute)."""
from __future__ import annotations
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.ci.proof_asset_snapshots import Input, manifest_keys, materialize  # noqa: E402

BINDINGS = {
    "candle-gen-lens/trainer_e2e::lens_bounded_acceleration_acceptance": {
        "LENS_BASE_SNAPSHOT": Input("lens-base-train-proof")},
    "candle-gen-z-image/trainer_e2e::z_image_bounded_acceleration_acceptance": {
        "Z_IMAGE_SNAPSHOT": Input("z-image-turbo-train-proof")},
    "candle-gen-wan/trainer_e2e::wan_t2v_14b_bounded_acceleration_acceptance": {
        "WAN_T2V_14B_SNAPSHOT": Input("wan2.2-t2v-a14b-train-proof")},
    "candle-gen-wan/training_modes_cuda::wan_ti2v_5b_bounded_acceleration_acceptance": {
        "WAN_TI2V_5B_TRAIN_SNAPSHOT": Input("wan2.2-ti2v-5b-train-proof")},
    "candle-gen-ltx/trainer_e2e::ltx23_bounded_acceleration_acceptance": {
        "LTX_TRAINING_TIER": Input("ltx-2.3-mlx-q4-train-proof", "q4"),
        "LTX_GEMMA_DIR": Input("ltx-2.3-mlx-q4-train-proof", "gemma")},
    "candle-gen-ltx/trainer_e2e::ltx25_bounded_acceleration_acceptance": {
        "LTX25_TRAINING_TIER": Input("ltx-2.5-mlx-dev-q4-train-proof", "dev/q4")},
    "candle-gen-sdxl/trainer_e2e::sdxl_bounded_acceleration_acceptance": {
        "SDXL_SNAPSHOT": Input("sdxl-base-1.0-fp16-train-proof"),
        "SDXL_TOKENIZER_CLIP_L_DIR": Input("clip-vit-large-patch14-tokenizer-proof"),
        "SDXL_TOKENIZER_CLIP_BIGG_DIR": Input("clip-vit-bigg-14-laion2b-tokenizer-proof"),
        "SDXL_VAE_FP16_FIX_DIR": Input("sdxl-vae-fp16-fix-train-proof")},
    "candle-gen-kolors/trainer_cuda_smoke::kolors_bounded_acceleration_acceptance": {
        "KOLORS_TRAIN_SNAPSHOT": Input("kolors-diffusers-fp16-train-proof")},
    "candle-gen-sd3/trainer_cuda_smoke::sd35_medium_bounded_acceleration_acceptance": {
        "SD35_MEDIUM_TRAIN_SNAPSHOT": Input("sd3.5-medium-train-proof"),
        # Generation's exact production artifact is distinct from the upstream training source.
        "SD35_MEDIUM_SAMPLE_SNAPSHOT": Input("sd3.5-medium-mlx-bf16-sample-proof", "bf16")},
    "candle-gen-qwen-image-2-1/train_peak_real_weights::qwen21_bounded_acceleration_acceptance": {
        "CANDLE_GEN_QWEN_IMAGE_2_1_SNAPSHOT": Input("qwen-image-2-1-train-proof")},
}

# Upstream ChatGLM3 ships only a slow tokenizer, and the Kolors readers load the derived fast
# `tokenizer/tokenizer.json` from the snapshot itself (candle-gen-kolors/src/tokenizer.rs:49-50). A
# snapshot without it gets the pinned derived file the worker overlays at install time.
OVERLAYS = {"KOLORS_TRAIN_SNAPSHOT": (Input("kolors-chatglm3-tokenizer-proof", "tokenizer.json"),
                                      "tokenizer/tokenizer.json")}
MANIFEST_KEYS = manifest_keys(BINDINGS, {name: {name: source} for name, (source, _) in OVERLAYS.items()})


def selected_binding(selection: str) -> dict[str, Input]:
    # Both exact spellings accepted by the existing executor; packages/all never
    # acquire this setup route, so they cannot launch a historical trainer matrix.
    selection = selection.replace("/integration/", "/", 1)
    if selection not in BINDINGS:
        raise ValueError("not an exact bounded cohort CUDA selection")
    return BINDINGS[selection]


def overlay_derived_files(paths: dict[str, Path]) -> None:
    """Place each pinned derived file a bound snapshot lacks; an existing one is kept."""
    for name, (source, relative) in OVERLAYS.items():
        if name not in paths:
            continue
        target = paths[name] / relative
        if target.is_file() and target.stat().st_size > 0:
            continue
        origin = materialize({name: source})[name]
        partial = target.with_name(target.name + ".partial")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, partial)
        os.replace(partial, target)
        print(f"{name}: placed {source.model} {source.path} at {target}", flush=True)


def main() -> None:
    request = json.loads(os.environ["MODEL_PROOF_REQUEST"])
    from scripts.ci.training_comparison import is_recipe
    if request.get("configuration") != "fixed_cuda" and not is_recipe(request.get("configuration")):
        raise ValueError("bounded cohort setup requires fixed_cuda")
    paths = materialize(selected_binding(request["selection"]))
    overlay_derived_files(paths)
    for name, path in paths.items():
        if not path.is_dir():
            raise ValueError(f"missing pinned installed asset {name}: {path}")
        files = list(path.rglob("*.safetensors")) if "TOKENIZER" not in name else list(path.rglob("*token*"))
        if not files or any(not file.is_file() or file.stat().st_size <= 0 for file in files):
            raise ValueError(f"missing or empty pinned component/tokenizer in {name}")
    with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as output:
        for name, path in paths.items():
            output.write(f"{name}={path.as_posix()}\n")
    print("Bound pinned runner-cache cohort paths; actual source readers must still admit content: " + ", ".join(paths))


if __name__ == "__main__":
    main()
