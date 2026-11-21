"""Check the registered MLX providers' concrete checkpoint and projection constructor seams.

The per-family entries identify production upload and binding sites. The registry surface tests
also compare every instantiated descriptor ID. Neither check loads a model or executes a kernel.
"""
from __future__ import annotations

import re
from pathlib import Path


# (file that loads the source, one or more projection constructor files).
MLX_MEDIA = {
    "anima": ("loader.rs", ("conditioner.rs", "transformer.rs")),
    "bernini": ("pipeline.rs", ("clip_diff.rs", "connector.rs", "qwen2_5_vl.rs")),
    "boogu": ("loader.rs", ("quant.rs",)),
    "chroma": ("loader.rs", ("quant.rs",)),
    "clip": ("lib.rs", ("lib.rs",)),
    "flux": ("loader.rs", ("ip_adapter.rs", "text_encoder.rs", "transformer.rs")),
    "flux2": ("loader.rs", ("text_encoder/mod.rs", "transformer.rs")),
    "flux3": ("model.rs", ("nn.rs",)),
    "ideogram": ("loader.rs", ("quant.rs",)),
    "iris": ("model.rs", ("adapters.rs", "nn.rs")),
    "joycaption": ("model.rs", ("model.rs",)),
    "kolors": ("block_stream.rs", ("chatglm3.rs",)),
    "krea": ("loader.rs", ("quant.rs",)),
    "krea_realtime": ("t2v.rs", ("load.rs",)),
    "lens": ("pipeline.rs", ("quant.rs", "reasoner.rs", "dit/mod.rs")),
    "ltx": ("block_stream.rs", ("gemma.rs", "transformer/linear.rs", "diff_vae.rs")),
    "mage": ("block_stream.rs", ("quant.rs", "text_encoder/load.rs")),
    "minimax_h3": ("block_stream.rs", ("quant.rs",)),
    "minimax_h3::h3_image": ("h3_image/dit.rs", ("h3_image/dit.rs", "quant.rs")),
    "mochi": ("text_encoder.rs", ("transformer.rs",)),
    "moss_image": ("loader.rs", ("transformer.rs", "text_encoder.rs")),
    "pulid": ("pulid_flux.rs", ("ca.rs", "eva_clip/attention.rs", "eva_clip/mlp.rs")),
    "qwen_image": ("loader.rs", ("text_encoder/attention.rs", "text_encoder/mlp.rs")),
    "qwen_image_2_1": ("loader.rs", ("transformer.rs",)),
    "sana": ("block_stream.rs", ("quant.rs",)),
    "scail2": ("generate.rs", ("clip.rs", "model.rs")),
    "sd3": ("loader.rs", ("transformer.rs",)),
    "sdxl": ("loader.rs", ("quant.rs", "ip_adapter.rs")),
    "seedvr2": ("pipeline.rs", ("dit.rs", "vae.rs")),
    "sensenova": ("loader.rs", ("quant.rs",)),
    "supra2": ("model.rs", ("dit.rs",)),
    "svd": ("model.rs", ("image_encoder.rs", "transformer.rs")),
    "wan": ("block_stream.rs", ("transformer.rs", "vace.rs")),
    "z_image": ("loader.rs", ("quant.rs",)),
}
# Model variants implemented in a registered crate share its source and constructor trace.
# Reviewing a new registration requires locating that variant's concrete consumer.
MLX_REGISTRATIONS = {
    "anima": "generator:model::AESTHETIC_REGISTRATION generator:model::BASE_REGISTRATION generator:model::TURBO_REGISTRATION trainer:training::AESTHETIC_TRAINER_REGISTRATION trainer:training::BASE_TRAINER_REGISTRATION trainer:training::TURBO_TRAINER_REGISTRATION",
    "bernini": "generator:bernini::FULL_REGISTRATION generator:pipeline::RENDERER_REGISTRATION",
    "boogu": "generator:model::BASE_REGISTRATION generator:model::EDIT_REGISTRATION generator:model::TURBO_REGISTRATION",
    "chroma": "generator:model::BASE_REGISTRATION generator:model::FLASH_REGISTRATION generator:model::HD_REGISTRATION",
    "clip": "image_embedder:IMAGE_REGISTRATION text_embedder:TEXT_REGISTRATION",
    "flux": "generator:model::DEV_REGISTRATION generator:model::SCHNELL_REGISTRATION generator:model_control::DEV_CONTROL_REGISTRATION",
    "flux2": "generator:model::DEV_EDIT_REGISTRATION generator:model::DEV_REGISTRATION generator:model::KLEIN_EDIT_REGISTRATION generator:model::KLEIN_KV_EDIT_REGISTRATION generator:model::KLEIN_REGISTRATION generator:model_control::DEV_CONTROL_REGISTRATION",
    "flux3": "generator:model::BASE_REGISTRATION generator:model::REGISTRATION trainer:training::trainer::TRAINER_REGISTRATION",
    "ideogram": "generator:model::QUALITY_REGISTRATION generator:model::TURBO_REGISTRATION",
    "iris": "generator:model::REGISTRATION trainer:train::REGISTRATION trainer:train::depth::REGISTRATION trainer:train::restore::REGISTRATION transform:restoration::REGISTRATION",
    "joycaption": "captioner:model::REGISTRATION",
    "kolors": "generator:crate::registry::REGISTRATION trainer:training::TRAINER_REGISTRATION",
    "krea": "generator:model::EDIT_REGISTRATION generator:model::RAW_REGISTRATION generator:model::TURBO_EDIT_REGISTRATION generator:model::TURBO_REGISTRATION generator:model_control::CONTROL_REGISTRATION trainer:training::TRAINER_REGISTRATION",
    "krea_realtime": "generator:pipeline::REGISTRATION",
    "lens": "generator:registry::BASE_REGISTRATION generator:registry::TURBO_REGISTRATION trainer:training::TRAINER_REGISTRATION",
    "ltx": "generator:model::REGISTRATION generator:model::REGISTRATION_25 trainer:training::TRAINER_REGISTRATION trainer:training::TRAINER_REGISTRATION_25",
    "mage": "generator:model::REGISTRATION generator:model::REGISTRATION_BASE generator:model::REGISTRATION_EDIT generator:model::REGISTRATION_EDIT_BASE generator:model::REGISTRATION_EDIT_TURBO generator:model::REGISTRATION_TURBO trainer:training::REGISTRATION",
    "minimax_h3": "generator:model::REGISTRATION",
    "minimax_h3::h3_image": "generator:model::REGISTRATION trainer:crate::training::TRAINER_REGISTRATION",
    "mochi": "generator:model::REGISTRATION",
    "moss_image": "generator:model::BASE_REGISTRATION generator:model::SFT_REGISTRATION trainer:training::BASE_REGISTRATION trainer:training::SFT_REGISTRATION",
    "pulid": "generator:pulid_flux::REGISTRATION",
    "qwen_image": "generator:model::REGISTRATION generator:model_control::REGISTRATION generator:model_edit::REGISTRATION",
    "qwen_image_2_1": "generator:model::REGISTRATION generator:model::TURBO_REGISTRATION trainer:training::TRAINER_REGISTRATION",
    "sana": "generator:model::BASE_REGISTRATION generator:model::SPRINT_REGISTRATION",
    "scail2": "generator:pipeline::REGISTRATION",
    "sd3": "generator:model::LARGE_REGISTRATION generator:model::MEDIUM_REGISTRATION generator:model::TURBO_REGISTRATION trainer:training::LARGE_TRAINER_REGISTRATION trainer:training::MEDIUM_TRAINER_REGISTRATION",
    "sdxl": "generator:model::REGISTRATION trainer:training::TRAINER_REGISTRATION",
    "seedvr2": "generator:registry::BASE_REGISTRATION generator:registry::SEVEN_B_REGISTRATION generator:registry::THREE_B_REGISTRATION",
    "sensenova": "generator:model::FAST_REGISTRATION generator:model::QUALITY_REGISTRATION",
    "supra2": "generator:model::REGISTRATION",
    "svd": "generator:model::REGISTRATION",
    "wan": "generator:model::I2V_14B_REGISTRATION generator:model::T2V_14B_REGISTRATION generator:model::TI2V_REGISTRATION generator:model_vace::VACE_FUN_REGISTRATION generator:model_vace::VACE_REGISTRATION trainer:training::I2V_14B_TRAINER_REGISTRATION trainer:training::T2V_14B_TRAINER_REGISTRATION trainer:training::TI2V_5B_TRAINER_REGISTRATION",
    "z_image": "generator:model::REGISTRATION generator:model_base::REGISTRATION generator:model_base_control::REGISTRATION generator:model_control::REGISTRATION trainer:training::REGISTRATION",
}


def _production(path: Path) -> str:
    if not path.is_file():
        return ""
    source = re.sub(r"//[^\n]*", "", path.read_text(encoding="utf-8"))
    # Ignore out-of-line test imports and entire top-level test-only modules; several providers
    # declare test modules before their production constructors.
    return re.sub(r"(?ms)^#\[cfg\(test\)\]\s*mod\s+\w+\s*(?:;|\{.*?^\})", "", source)


REGISTER = re.compile(
    r"\.register_(generator|trainer|transform|captioner|image_embedder|text_embedder)"
    r"\(\s*([\w:]+)"
)
OPEN = re.compile(
    r"Weights::from_(?:file(?:_with_fp8(?:_filter)?)?|(?:dir|file)_with_(?:projection_layout|layout)|dir|paths|source_map)\("
    r"|read_logical_weights\(|mlx_llm::text_registry\(|DitBlockStream::"
)
BIND = re.compile(
    r"mlx_gen::quant::(?:native_lin|lin(?:_with_bias)?)\("
    r"|(?:mlx_gen::)?NativeProjection::from_(?:key|weights)\("
    r"|mlx_llm::text_registry\(|\.into_language_weights\(|DitBlock::from_weights\("
)
TEXT = {
    "provider": ("provider.rs", ("Qwen35Model::build_lazy(", "CausalLm::build_lazy("),
                 ("models/llama.rs", "models/qwen35.rs")),
    "joycaption": ("joycaption.rs", ("CausalLm::from_weights(",), ("models/llama.rs",)),
    "starvector_1b": ("starvector_1b.rs", ("GptBigCode::from_weights(",), ("models/gpt_bigcode.rs",)),
    "starvector_8b": ("starvector_8b.rs", ("StarCoder2::from_weights(",), ("models/starcoder2.rs",)),
}


def check_mlx(root: Path) -> list[str]:
    media = root / "crates/media/mlx-gen"
    catalog = _production(media / "mlx-gen-catalog/src/lib.rs")
    entries = set(re.findall(r"mlx_gen_([a-z_0-9]+(?:::h3_image)?)::register_providers\(", catalog))
    issues = []
    if entries != MLX_MEDIA.keys():
        issues.append(f"MLX media registration lacks a current constructor trace: {sorted(entries ^ MLX_MEDIA.keys())}")
    for name in sorted(entries & MLX_MEDIA.keys()):
        family = name.split("::")[0].replace("_", "-")
        src = media / f"mlx-gen-{family}" / "src"
        reader, modules = MLX_MEDIA[name]
        if not OPEN.search(_production(src / reader)):
            issues.append(f"{name}: {reader} lost its source-aware checkpoint loader")
        for module in modules:
            if not BIND.search(_production(src / module)):
                issues.append(f"{name}: {module} lost its shared projection binding")
        lib = src / ("h3_image/mod.rs" if "::" in name else "lib.rs")
        actual = {f"{kind}:{registration}" for kind, registration in REGISTER.findall(_production(lib))}
        expected = set(MLX_REGISTRATIONS[name].split())
        if actual != expected:
            issues.append(f"{name}: review constructor routing for {sorted(actual ^ expected)}")
    llm = root / "crates/llm/mlx-llm/src"
    text = _production(llm / "lib.rs").split("pub fn register_text_providers", 1)[-1].split("\n}", 1)[0]
    registrations = set(re.findall(r"\.register\(([a-z_0-9]+)::REGISTRATION\)", text))
    if registrations != TEXT.keys():
        issues.append(f"MLX text registration lacks a constructor trace: {sorted(registrations ^ TEXT.keys())}")
    for name in sorted(registrations & TEXT.keys()):
        file, factories, models = TEXT[name]
        for factory in factories:
            if factory not in _production(llm / file):
                issues.append(f"{name}: {file} no longer loads {factory}")
        for path in models:
            if not re.search(r"Projection::(?:native|from_named)\(", _production(llm / path)):
                issues.append(f"{name}: {path} lost shared native projection construction")
    for path, needle in (
        (media / "src/weights.rs", "mlx_layers::source::prepare_sources("),
        (llm / "primitives/weights.rs", "mlx_layers::source::prepare_sources_with_experts("),
        (llm / "primitives/moe.rs", "obs.expert("),
        (llm / "models/qwen35.rs", "SwitchLinear::from_checkpoint("),
        (media / "mlx-gen-sdxl/src/ldm.rs", "insert_source_projection("),
        (media / "mlx-gen-z-image/src/comfyui.rs", "insert_split_projection("),
        (media / "mlx-gen-qwen-image-2-1/src/loader.rs", "visual.into_language_weights("),
        (media / "mlx-gen-flux3/src/model.rs", "weights.native_tensor_keys()"),
        (media / "mlx-gen-flux3/src/model.rs", "storage.validate_source_headers("),
        (media / "mlx-gen-wan/src/memory_strategy.rs", "plan_projection_sources("),
        (media / "mlx-gen-svd/src/memory_strategy.rs", "projected_float_component_bytes("),
        (media / "mlx-gen-minimax-h3/src/text_encoder/mod.rs", "Weights::from_selected_files("),
        (media / "mlx-gen-lens/src/text_encoder/gpt_oss.rs", "SwitchLinear::from_scalar_source("),
        (media / "mlx-gen-minimax-h3/src/dit/heads.rs", "source_projection("),
        (media / "src/quant.rs", "NativeProjection::from_weights("),
        (llm / "primitives/projection.rs", "NativeProjection::from_key("),
    ):
        if needle not in _production(path):
            issues.append(f"{path.relative_to(root)} lacks {needle}")
    return issues


def check_all(root: Path) -> list[str]:
    from check_native_candle_media import check_candle_media
    from native_projection_llm_audio import check_candle_llm_audio

    return check_mlx(root) + check_candle_media(root) + check_candle_llm_audio(root)


if __name__ == "__main__":
    import sys
    problems = check_all(Path(__file__).resolve().parents[1])
    for problem in problems:
        print(problem, file=sys.stderr)
    raise SystemExit(bool(problems))
