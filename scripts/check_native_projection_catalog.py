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
    "instantid": ("model.rs", ()),
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
    "anima": "route:model::route_aesthetic route:model::route_base route:model::route_turbo trainer:training::AESTHETIC_TRAINER_REGISTRATION trainer:training::BASE_TRAINER_REGISTRATION trainer:training::TURBO_TRAINER_REGISTRATION",
    "bernini": "route:bernini::route_full route:pipeline::route_renderer",
    "boogu": "route:model::route_base route:model::route_edit route:model::route_turbo",
    "chroma": "route:model::route_base route:model::route_flash route:model::route_hd",
    "clip": "image_embedder:IMAGE_REGISTRATION text_embedder:TEXT_REGISTRATION",
    "flux": "route:model::route_dev route:model::route_dev_ip_adapter route:model::route_schnell route:model::route_schnell_ip_adapter route:model_control::route_dev_control",
    "flux2": "route:model::route_dev route:model::route_dev_edit route:model::route_klein route:model::route_klein_edit route:model::route_klein_kv_edit route:model_control::route_dev_control",
    "flux3": "route:model::route route:model::route_base trainer:training::trainer::TRAINER_REGISTRATION",
    "ideogram": "route:model::route_quality route:model::route_turbo",
    "instantid": "route:registry::route_instantid",
    "iris": "route:model::route route:model::route_depth route:model::route_restore trainer:train::REGISTRATION trainer:train::depth::REGISTRATION trainer:train::restore::REGISTRATION transform:restoration::REGISTRATION",
    "joycaption": "captioner:model::REGISTRATION",
    "kolors": "route:crate::registry::route trainer:training::TRAINER_REGISTRATION",
    "krea": "route:model::route_edit route:model::route_raw route:model::route_turbo route:model::route_turbo_edit route:model_control::route_control trainer:training::TRAINER_REGISTRATION",
    "krea_realtime": "route:pipeline::route",
    "lens": "route:registry::route_base route:registry::route_turbo trainer:training::TRAINER_REGISTRATION",
    "ltx": "route:model::route route:model::route_25 trainer:training::TRAINER_REGISTRATION trainer:training::TRAINER_REGISTRATION_25",
    "mage": "route:model::route route:model::route_base route:model::route_edit route:model::route_edit_base route:model::route_edit_turbo route:model::route_turbo trainer:training::REGISTRATION",
    "minimax_h3": "route:model::route",
    "minimax_h3::h3_image": "route:model::route trainer:crate::training::TRAINER_REGISTRATION",
    "mochi": "route:model::route_mochi",
    "moss_image": "route:model::route_base route:model::route_sft trainer:training::BASE_REGISTRATION trainer:training::SFT_REGISTRATION",
    "pulid": "route:pulid_flux::route_flux1_dev_pulid route:pulid_flux::route_pulid_flux",
    "qwen_image": "route:model::route_qwen_image route:model_control::route_qwen_image_control route:model_edit::route_qwen_image_edit",
    "qwen_image_2_1": "route:model::route_base route:model::route_turbo trainer:training::TRAINER_REGISTRATION",
    "sana": "route:model::route_base route:model::route_sprint",
    "scail2": "route:pipeline::route_scail2",
    "sd3": "route:model::route_large route:model::route_medium route:model::route_turbo trainer:training::LARGE_TRAINER_REGISTRATION trainer:training::MEDIUM_TRAINER_REGISTRATION",
    "sdxl": "route:model::route_sdxl trainer:training::TRAINER_REGISTRATION",
    "seedvr2": "route:registry::route_3b route:registry::route_7b route:registry::route_base",
    "sensenova": "route:model::route_fast route:model::route_quality",
    "supra2": "route:model::route_supra2",
    "svd": "route:model::route_svd",
    "wan": "route:model::route_i2v_14b route:model::route_t2v_14b route:model::route_ti2v route:model_vace::route_vace route:model_vace::route_vace_fun trainer:training::I2V_14B_TRAINER_REGISTRATION trainer:training::T2V_14B_TRAINER_REGISTRATION trainer:training::TI2V_5B_TRAINER_REGISTRATION",
    "z_image": "route:model::route_turbo route:model_base::route_base route:model_base_control::route_base_control route:model_control::route_turbo_control trainer:training::REGISTRATION",
}


def _production(path: Path) -> str:
    if not path.is_file():
        return ""
    source = re.sub(r"//[^\n]*", "", path.read_text(encoding="utf-8"))
    # Ignore out-of-line test imports and entire top-level test-only modules; several providers
    # declare test modules before their production constructors.
    return re.sub(r"(?ms)^#\[cfg\(test\)\]\s*mod\s+\w+\s*(?:;|\{.*?^\})", "", source)


REGISTER = re.compile(
    r"\.register_(generator|route|trainer|transform|captioner|image_embedder|text_embedder)"
    r"\(\s*([\w:]+)"
)
OPEN = re.compile(
    r"Weights::from_(?:file(?:_with_fp8(?:_filter)?)?|(?:dir|file)_with_(?:projection_layout|layout)|dir|paths|source_map)\("
    r"|read_logical_weights\(|mlx_llm::text_registry\(|DitBlockStream::"
)
BIND = re.compile(
    r"mlx_gen::quant::(?:native_lin|lin(?:_with_bias)?)\("
    r"|(?:mlx_gen::)?NativeProjection::from_(?:key|weights)\("
    r"|\.source_projection\("
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
    layers = root / "crates/layers/mlx-layers/src"
    for path, module, symbol in (
        (media / "src/quant.rs", "quant", "native_lin"),
        (llm / "primitives/projection.rs", "llm_projection", "Projection"),
    ):
        export = rf"pub\s+use\s+mlx_layers::{module}::(?:\{{[^}}]*\b{symbol}\b[^}}]*\}}|{symbol}|\*)\s*;"
        if not re.search(export, _production(path)):
            issues.append(f"{path.relative_to(root)} lost the shared {module}::{symbol} export")
    for path, needle in (
        (media / "src/weights.rs", "mlx_layers::source::prepare_sources("),
        (llm / "primitives/weights.rs", "mlx_layers::source::prepare_sources_for_consumers("),
        (llm / "primitives/weights.rs", "crate::models::qwen35::stacked_expert_key,"),
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
        (layers / "quant.rs", "NativeProjection::from_weights("),
        (layers / "llm_projection.rs", "NativeProjection::from_key("),
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
