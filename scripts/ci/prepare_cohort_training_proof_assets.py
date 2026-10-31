"""Bind installed pinned assets for exact bounded CUDA proofs (no fetch/compute)."""
from __future__ import annotations
import json
import os
from pathlib import Path

CACHE = Path("E:/huggingface/hub")
def snapshot(repo: str, revision: str, subdir: str = "") -> Path:
    return CACHE / ("models--" + repo.replace("/", "--")) / "snapshots" / revision / subdir

BINDINGS = {
    "candle-gen-lens/trainer_e2e::lens_bounded_acceleration_acceptance": {
        "LENS_BASE_SNAPSHOT": snapshot("SceneWorks/Lens", "5c5521d4417a3cae55816929ece69319d1e7712a")},
    "candle-gen-z-image/trainer_e2e::z_image_bounded_acceleration_acceptance": {
        "Z_IMAGE_SNAPSHOT": snapshot("Tongyi-MAI/Z-Image-Turbo", "f332072aa78be7aecdf3ee76d5c247082da564a6")},
    "candle-gen-wan/trainer_e2e::wan_t2v_14b_bounded_acceleration_acceptance": {
        "WAN_T2V_14B_SNAPSHOT": snapshot("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "5be7df9619b54f4e2667b2755bc6a756675b5cd7")},
    "candle-gen-wan/training_modes_cuda::wan_ti2v_5b_bounded_acceleration_acceptance": {
        "WAN_TI2V_5B_TRAIN_SNAPSHOT": snapshot("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "b8fff7315c768468a5333511427288870b2e9635")},
    "candle-gen-ltx/trainer_e2e::ltx23_bounded_acceleration_acceptance": {
        "LTX_TRAINING_TIER": snapshot("SceneWorks/ltx-2.3-mlx", "254989c3ca7ee691187647f350b112c0c448789d", "q4"),
        "LTX_GEMMA_DIR": snapshot("SceneWorks/ltx-2.3-mlx", "254989c3ca7ee691187647f350b112c0c448789d", "gemma")},
    "candle-gen-ltx/trainer_e2e::ltx25_bounded_acceleration_acceptance": {
        "LTX25_TRAINING_TIER": snapshot("SceneWorks/ltx-2.5-mlx", "081658ce6886cacba20817ce0359bbefef706ff2", "dev/q4")},
    "candle-gen-sdxl/trainer_e2e::sdxl_bounded_acceleration_acceptance": {
        "SDXL_SNAPSHOT": snapshot("stabilityai/stable-diffusion-xl-base-1.0", "462165984030d82259a11f4367a4eed129e94a7b"),
        "SDXL_TOKENIZER_CLIP_L_DIR": snapshot("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41"),
        "SDXL_TOKENIZER_CLIP_BIGG_DIR": snapshot("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "743c27bd53dfe508a0ade0f50698f99b39d03bec"),
        "SDXL_VAE_FP16_FIX_DIR": snapshot("madebyollin/sdxl-vae-fp16-fix", "207b116dae70ace3637169f1ddd2434b91b3a8cd")},
    "candle-gen-kolors/trainer_cuda_smoke::kolors_bounded_acceleration_acceptance": {
        "KOLORS_TRAIN_SNAPSHOT": snapshot("Kwai-Kolors/Kolors-diffusers", "7e091c75199e910a26cd1b51ed52c28de5db3711")},
    "candle-gen-sd3/trainer_cuda_smoke::sd35_medium_bounded_acceleration_acceptance": {
        "SD35_MEDIUM_TRAIN_SNAPSHOT": snapshot("stabilityai/stable-diffusion-3.5-medium", "b940f670f0eda2d07fbb75229e779da1ad11eb80"),
        # Generation's exact production artifact is distinct from the upstream training source.
        "SD35_MEDIUM_SAMPLE_SNAPSHOT": snapshot("SceneWorks/sd3.5-medium-mlx", "5413e962bb326db248be2026a93b147c323392b6", "bf16")},
    "candle-gen-qwen-image-2-1/train_peak_real_weights::qwen21_bounded_acceleration_acceptance": {
        "CANDLE_GEN_QWEN_IMAGE_2_1_SNAPSHOT": snapshot("Qwen/Qwen-Image-2.1", "790c92633540aa0cb11d9abf19eb46d861714758")},
}


def selected_binding(selection: str) -> dict[str, Path]:
    # Both exact spellings accepted by the existing executor; packages/all never
    # acquire this setup route, so they cannot launch a historical trainer matrix.
    selection = selection.replace("/integration/", "/", 1)
    if selection not in BINDINGS:
        raise ValueError("not an exact bounded cohort CUDA selection")
    return BINDINGS[selection]


def main() -> None:
    request = json.loads(os.environ["MODEL_PROOF_REQUEST"])
    if request.get("configuration") != "fixed_cuda":
        raise ValueError("bounded cohort setup requires fixed_cuda")
    paths = selected_binding(request["selection"])
    for name, path in paths.items():
        if not path.is_dir():
            raise ValueError(f"missing pinned installed asset {name}: {path}")
        files = list(path.rglob("*.safetensors")) if "TOKENIZER" not in name else list(path.rglob("*token*"))
        if not files or any(not file.is_file() or file.stat().st_size <= 0 for file in files):
            raise ValueError(f"missing or empty pinned component/tokenizer in {name}")
    with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as output:
        for name, path in paths.items():
            output.write(f"{name}={path.as_posix()}\n")
    print("Bound existing pinned cohort paths; actual source readers must still admit content: " + ", ".join(paths))


if __name__ == "__main__":
    main()
