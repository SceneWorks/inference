"""Guard the production registration, file reader and operator paths of the Candle media catalog."""
from pathlib import Path
import re

from native_projection_llm_audio import _production


# Each pair identifies the actual source/component reader and a projection constructor on its
# normal model path. Extra entries cover distinct formats or source adapters in the same family.
ROUTES = {
    "anima": ("loader:mmap_var_builder(", "vae:remap_file_var_builder(", "nn:Linear::from_var_builder("),
    "bernini": ("components:component_vb(", "nn:Linear::from_var_builder(", "vision:native_linear_weight_flattened("),
    "boogu": ("pipeline:Weights::from_dir(", "loader:st.linear("),
    "chroma": ("pipeline:load_sorted_mmap(", "quant:Self::linear_detect_gs("),
    "clip": ("lib:mmap_var_builder(", "lib:w.linear("),
    "depth": ("common:candle_gen::Weights::from_files(", "common:w.0.native_linear("),
    "face": ("common:mmap_var_builder(", "common:Linear::from_var_builder(", "iresnet:w.linear("),
    "flux": ("flux1_load:mmap_var_builder(", "control:candle_nn::linear(", "ip_adapter:w.linear("),
    "flux2": ("lib:load_path_mmap(", "single_file:LogicalWeightReader", "single_file:.native_projection("),
    "flux3": ("dit:mmap_var_builder(", "nn:vb.native_linear_weight_with_shape(", "nn:publisher.linear(", "dit:validate_source_checkpoint("),
    "ideogram": ("pipeline:Weights::from_dir(", "loader:st.linear("),
    "instantid": ("model:Weights::from_file(", "model:Resampler::from_weights(", "model:load_sdxl_controlnet("),
    "iris": ("nn:mmap_var_builder(", "nn:src.native_linear("),
    "joycaption": ("lib:candle_llm::text_registry(", "lib:.load_for_model_with("),
    "kolors": ("pipeline:mmap_var_builder(", "chatglm3:QLinear::linear_detect_gs("),
    "krea": ("loader:LogicalWeightReader", "loader:.native_projection(", "control_provider:Weights::from_file("),
    "lens": ("text_encoder:mmap_var_builder(", "quant:AdaptLinear as QLinear"),
    "ltx": ("tier:mmap_var_builder_with_geometry(", "quant:.native_linear_weight("),
    "mage": ("transformer:Weights::from_dir(", "transformer:linear(weights, base, bias)"),
    "minimax_h3": ("dit/model:Weights::from_files_native_int8(", "quant:w.native_linear_shaped(", "dit/heads:w.native_linear_shaped("),
    "mochi": ("transformer:mmap_var_builder(", "nn:linear_b("),
    "moss_image": ("loader:from_authenticated_buffers(", "transformer:w.linear("),
    "pid": ("engine:Weights::from_files(", "nn:w.linear(", "gemma2:linear_from_weights("),
    "pulid": ("pulid_flux:Weights::from_file(", "idformer:w.linear(", "ca:w.linear("),
    "qwen_image": ("comfyui:remap_file_var_builder(", "quant:AdaptLinear as QLinear"),
    "qwen_image_2_1": ("loader:load_sorted_mmap(", "quant:AdaptLinear as QLinear"),
    "sam3": ("common:candle_gen::Weights::from_files(", "common:w.0.native_linear("),
    "sana": ("pipeline:Weights::from_files(", "transformer:linear_from_weights("),
    "scail2": ("pipeline:src.native_linear_weight(", "common:QLinear::linear_detect("),
    "sd3": ("pipeline:mmap_var_builder(", "conditioning:head.native_linear_weight(", "quant:candle_nn::linear("),
    "sdxl": ("ldm:remap_file_var_builder(", "conditioning:mmap_var_builder(", "ip_adapter:w.linear("),
    "seedvr2": ("weights:candle_gen::Weights::from_file(", "dit:w.native_projection(", "vae:w.native_projection("),
    "sensenova": ("lib:mmap_var_builder(", "quant:QLinear::from_dense("),
    "supra2": ("dit:candle_gen::Weights::from_file(", "dit:w.linear("),
    "svd": ("lib:mmap_var_builder(", "transformer:linear_no_bias("),
    "wan": ("text_encode:mmap_var_builder(", "quant:AdaptLinear as QLinear"),
    "z_image": (
        "comfyui:remap_file_var_builder(",
        "dit:Execution::Inference => AdaptLinear::linear_detect(",
        "text_encoder:Self::Dense => QLinear::linear_no_bias(",
        "text_encoder:Self::DetectPacked => QLinear::linear_detect(",
    ),
}

# These library APIs are composed by other registered routes or by the product worker itself.
DIRECT = {"depth", "face", "pid", "sam3"}


def check_candle_media(root: Path) -> list[str]:
    media = root / "crates/media/candle-gen"
    catalog = _production(media / "candle-gen-catalog/src/lib.rs")
    exports = set(re.findall(r"pub\s+use\s+candle_gen_(\w+)\s+as\s+\w+\s*;", catalog))
    registrations = set(re.findall(r"candle_gen_(\w+)::register_providers\(registry\)", catalog))
    issues = []
    for name, actual, expected in [
        ("exports", exports, set(ROUTES)),
        ("registered", registrations, set(ROUTES) - DIRECT),
    ]:
        if actual != expected:
            issues.append(f"Candle media {name} changed. Trace the constructor path for {sorted(actual ^ expected)}")
    for name, entries in ROUTES.items():
        for entry in entries:
            module, call = entry.split(":", 1)
            path = media / ("candle-gen-" + name.replace("_", "-")) / "src" / (module + ".rs")
            if not path.is_file() or call not in _production(path):
                issues.append(f"Candle media {name} no longer uses {module} / {call}")
    return issues
