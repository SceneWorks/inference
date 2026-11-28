"""Verify registered Candle text/audio providers still reach their checkpoint constructors."""

import re
from pathlib import Path

from check_model_inference_policy import rust_function_bodies, rust_production_mask


# Each witness names a production function and the next callable on the ordinary load or
# request-time construction path. Parameterized registrations also have exact-ID Rust tests.
AUDIO_ROUTES = {
    "acestep": [
        ("model", "pipeline", "from_snapshot"), ("pipeline", "from_snapshot", "mmap_vb"),
        ("pipeline", "mmap_vb", "mmap_var_builder"), ("tokenizer", "load_vb", "mmap_var_builder"),
        ("vae", "read_prefixed_tensors", "coefficients"),
    ],
    "chatterbox": [
        ("model", "ensure_t3", "mmap_var_builder"),
        *((part, "from_snapshot", "mmap_var_builder") for part in ("campplus", "flow", "hift", "s3tokenizer")),
        ("perth", "from_safetensors", "mmap_var_builder"),
    ],
    "chatterbox-ve": [("model", "embed", "encoder"), ("model", "encoder", "mmap_var_builder"),
                      ("encoder", "new", "lstm"), ("encoder", "new", "linear")],
    "clap": [("model", "build", "state_builder")],
    "kokoro": [("model", "pipeline", "from_snapshot"), ("pipeline", "from_snapshot", "normalized_state_builder")],
    "mmaudio": [
        ("generator", "pipeline", "from_components"), ("generator_44k", "pipeline", "from_components"),
        ("clip", "load_from_pth", "state_builder"), ("model", "load_from_pth", "state_builder"),
        ("mmdit", "load_config_from_pth", "state_builder"),
        *(("output", loader, "state_builder") for loader in (
            "load_vae_from_pth", "load_bigvgan_from_pth", "load_vae_44k_from_pth", "load_bigvgan_v2_from_pth")),
        ("blocks", "load", "native_linear_weight_flattened"),
    ],
    "moss-sfx": [("model", "pipeline", "from_snapshot"), ("pipeline", "from_snapshot", "mmap_var_builder"),
                 ("pipeline", "mmap_text_encoder_shards", "mmap_var_builder"), ("vae", "load_decode_tensors", "coefficients")],
    "moss-tts": [("model", "pipeline", "from_snapshot"), ("model", "from_snapshot", "mmap_var_builder"),
                 ("codec", "load", "normalized_state_builder"), ("backbone", "new", "native_tied_head")],
    "moss-tts-realtime": [("model", "pipeline", "from_snapshot"), ("model", "from_snapshot", "mmap_var_builder"),
                          ("codec", "load", "mmap_var_builder"), ("codec", "encoder_half", "mmap_var_builder")],
    "openvoice": [("model", "pipeline", "from_snapshot"), ("pipeline", "from_snapshot", "state_var_builder"),
                  ("weights", "state_var_builder", "normalized_state_builder")],
    "stable-audio-3": [("model", "pipeline", "from_layout_with_adapters"), ("weights", "mmap_backend", "AudioCheckpoint::open"),
                       ("weights", "packed_metal_backend", "AudioCheckpoint::open"),
                       ("adapters", "native_linear_weight", "LoraFactor::new")],
    "whisper": [("model", "transcribe", "build"), ("model", "build", "mmap_var_builder"),
                ("model", "build", "native_tied_head")],
    "yue": [("codec", "load", "AudioCheckpoint::open"), ("icl", "load", "AudioCheckpoint::open"),
            ("vocoder", "load_decoder", "AudioCheckpoint::open")],
    "yue2": [("provider", "load_generator", "load_with_stage_compute"), ("model", "open_tier", "mmap_var_builder")],
    # This direct API takes the VerifiedClosure carrying the exact parent and derivative hashes.
    "sheetsage2": [("provider", "load", "authorize_closure"), ("provider", "load", "Weights::load")],
}

TEXT_ROUTES = {
    "provider": [
        ("provider", "load_dir", "Weights::from_dir"), ("models/llama", "projection_from_weights", "checkpoint_projection"),
        ("models/qwen35", "keyed_projection", "checkpoint_projection"),
        ("models/qwen35", "stacked_expert_projections", "native_projection_slice"),
        ("models/block_draft", "proj", "native_linear"), ("models/gemma4_mm", "from_weights", "linear"),
        ("models/qwen35_vision", "from_weights_layout", "native_flattened_projection"),
    ],
    "llava": [("llava", "from_dir_with", "Weights::from_dir"), ("llava", "from_weights", "linear")],
    "starvector": [("starvector", "load", "Weights::from_dir"), ("models/starvector", "decoder_projection", "linear")],
    "starvector_8b": [("starvector_8b", "load", "Weights::from_dir"), ("models/starcoder2", "from_weights", "linear"),
                      ("models/siglip", "from_weights", "linear")],
}

# Most crates expose exactly one model::REGISTRATION. These are the catalog's variant factories,
# including families which register an array via fold and those re-exporting their builder.
VARIANT_REGISTRATIONS = {
    "acestep": {"model::route"},
    "chatterbox": {"model::route"},
    "kokoro": {"model::route"},
    "mmaudio": {"generator::route", "generator_44k::route"},
    "moss-sfx": {"model::route"},
    "moss-tts": {"route"},
    "moss-tts-realtime": {"route"},
    "stable-audio-3": {"model::" + symbol for symbol in (
        "route", "sfx_route", "medium_route", "music_base_route",
        "sfx_base_route", "medium_base_route")},
    "yue": {"model::route_" + language + "_" + mode for language in ("en", "zh", "jp_kr") for mode in ("cot", "icl")},
    "yue2": {"provider::route"},
}


def _audio_registrations(folder):
    code = "\n".join(_production(path) for path in sorted(folder.glob("*.rs")))
    body = next(_bodies(code, "register_providers"), "")
    entries = set(re.findall(r"\.register_\w+\(\s*([\w:]+)(?:\(\))?\s*\)", body))
    if "REGISTRATIONS" in body:
        array = re.search(r"const REGISTRATIONS\s*:\s*\[[^\]]+\]\s*=\s*\[([^\]]*)\]", code)
        if array:
            entries.discard("reg")
            entries.update(re.findall(r"[A-Z][A-Z0-9_]+", array[1]))
    return entries


# Raw mmap entry points are confined to the shared source seam and specifically typed readers.
# Hashing or writing a conversion tier never executes its stored codes as dense operands.
RAW_READS = re.compile(r"\b(?:VarBuilder::from_mmaped_safetensors|MmapedSafetensors::(?:multi|new)|safetensors::load|mmap_safetensors)\s*\(")
RAW_OWNERS = {
    ("audio/candle-audio/src/weights.rs", "mmap_safetensors"),
    ("audio/candle-audio/src/weights.rs", "open"),
    ("audio/candle-audio-yue2/src/snapshot.rs", "native_tensor_digests"),
    ("audio/candle-audio-yue2/src/tier.rs", "assemble"),
    ("audio/candle-audio-yue2/src/vae.rs", "load_files_with_dtype"),  # verified F32 convolution VAE
    ("audio/candle-audio-yue2/src/quality.rs", "load_reference"),  # offline reference data
    ("audio/candle-audio-sheetsage2/src/model/weights.rs", "load"),  # verified F32 only
    ("llm/candle-llm/src/primitives/weights.rs", "from_file_host_raw"),  # legacy GGML / ModelOpt conversion
    ("llm/candle-llm/src/primitives/weights.rs", "checkpoint_source"),
    ("llm/candle-llm/src/models/block_draft.rs", "open_on_device"),
}


def _production(path: Path) -> str:
    return rust_production_mask(path.read_text(encoding="utf-8"))


def _bodies(text: str, name: str):
    return (body for decl, body in rust_function_bodies(text) if decl == name)


def check_candle_llm_audio(root: Path) -> list[str]:
    errors = []
    crates = root / "crates"
    audio = crates / "audio"
    text = crates / "llm/candle-llm/src"
    catalog = _production(audio / "candle-audio-catalog/src/lib.rs")
    registration = next(_bodies(catalog, "register_providers"), "")
    discovered = {name.replace("_", "-") for name in re.findall(r"\bcandle_audio_(\w+)::register_providers\(", registration)}
    direct = re.search(r"pub const CRATE_API_PROVIDERS[^;]*;", catalog)
    discovered.update(name.replace("_", "-") for name in re.findall(r"candle_audio_(\w+)::", direct[0] if direct else ""))
    text_catalog = _production(text / "lib.rs")
    registered_text = {name for fn in ("register_text_providers", "register_cuda_text_providers")
                       for body in _bodies(text_catalog, fn)
                       for name in re.findall(r"\.register\(\s*(\w+)::REGISTRATION\s*\)", body)}
    def audit(found, witnesses, folder, label):
        for name in sorted(found ^ set(witnesses)):
            errors.append(f"{label} {name}: registry and audited checkpoint constructors disagree")
        for name in sorted(found & set(witnesses)):
            for module, fn, callee in witnesses[name]:
                path = folder(name) / (module + ".rs")
                target = r"\b" + re.escape(callee) + r"\s*(?:\(|[,)]|$)"
                if not path.is_file() or not any(re.search(target, body) for body in _bodies(_production(path), fn)):
                    errors.append(f"{label} {name}: {module}::{fn} has lost its {callee} checkpoint route")

    for name in sorted(discovered - {"sheetsage2"}):
        folder = audio / ("candle-audio-" + name) / "src"
        found = {entry.removeprefix("model::") if name not in VARIANT_REGISTRATIONS else entry
                 for entry in _audio_registrations(folder)}
        expected = VARIANT_REGISTRATIONS.get(name, {"REGISTRATION"})
        for entry in sorted(found ^ expected):
            errors.append(f"Candle audio {name}: registration {entry} needs a checkpoint constructor witness")
    for fn in ("register_text_providers", "register_cuda_text_providers"):
        for body in _bodies(text_catalog, fn):
            for expr in re.findall(r"\.register\(([^)]+)\)", body):
                if not re.fullmatch(r"\s*\w+::REGISTRATION\s*", expr):
                    errors.append(f"Candle text: unreviewed checkpoint registration {expr}")

    audit(discovered, AUDIO_ROUTES, lambda name: audio / ("candle-audio-" + name) / "src", "Candle audio")
    audit(registered_text, TEXT_ROUTES, lambda name: text, "Candle text")
    registry = _production(crates / "contracts/gen-core/src/registry/providers.rs")
    scopes = _production(crates / "contracts/gen-core/src/registry/media_receipts.rs")
    for fn, trait in (("audio_embedder", "AudioEmbedder"), ("voice_embedder", "VoiceEmbedder"),
                      ("transcriber", "Transcriber"), ("audio_transform", "AudioTransform")):
        if not re.search(r"dyn\s+" + trait + r"\s*,\s*super::media_receipts::load_" + fn, registry):
            errors.append(f"{trait} registry load no longer retains its checkpoint receipts")
        if not re.search(r"scope_model!\(\s*load_" + fn + r"\s*,\s*" + trait, scopes):
            errors.append(f"{trait} deferred work has no owning checkpoint scope")
    sources = [text, audio / "candle-audio/src", *(audio / ("candle-audio-" + name) / "src" for name in sorted(discovered))]
    for folder in sources:
        for path in sorted(folder.rglob("*.rs")):
            local = path.relative_to(folder)
            if local.name == "tests.rs" or {"tests", "bin"} & set(local.parts):
                continue
            code = _production(path)
            for match in RAW_READS.finditer(code):
                if re.search(r"\bfn\s*$", code[max(0,match.start()-8):match.start()]):
                    continue
                callers = re.findall(r"\bfn\s+(\w+)", code[:match.start()])
                owner = (path.relative_to(crates).as_posix(), callers[-1] if callers else "<module>")
                if owner not in RAW_OWNERS:
                    errors.append(f"{owner[0]}::{owner[1]} bypasses the shared checkpoint reader via {match[0]}")
    return errors
