"""Candle Krea Raw phase comparison at production widths with one image DiT block."""

import json
import math
from pathlib import Path
import re

from scripts.ci.flux1_pair import fixture
from scripts.ci.model_inference_proofs import unlinked_path

PROFILE = "krea-raw-native-phases"
PACKAGE = "candle-gen-krea"
BACKEND = "candle-cuda"
PROBE = Path(__file__).with_name("krea_phase_tier1_probe.rs.in")
KREA = f"crates/media/candle-gen/{PACKAGE}/src"
SAMPLER = "crates/media/candle-gen/candle-gen/src/sampler.rs"
COMPLETE = "KREA_NATIVE_PHASE_TIER1_COMPLETED"
SEED = 0x5EED
FIXTURE_FILES = {
    "transformer/config.json", "native.safetensors", "text.safetensors",
    "trunk.safetensors", "fold-only.safetensors",
}
CASE_ORDER = ("resident-selected", "resident-base", "staged-selected", "staged-base")
# Krea2Config::turbo() is the published Raw/Turbo architecture in both source arms.
# The Rust probe checks the materialized config against that constructor on the device run.
CONFIG = {
    "num_attention_heads": 48, "num_key_value_heads": 12, "attention_head_dim": 128,
    "intermediate_size": 16384, "num_layers": 1, "in_channels": 64, "norm_eps": 1e-5,
    "axes_dims_rope": [32, 48, 48], "rope_theta": 1000.0, "timestep_embed_dim": 256,
    "num_text_layers": 12, "num_layerwise_text_blocks": 2, "num_refiner_text_blocks": 2,
    "text_hidden_dim": 2560, "text_intermediate_size": 6912,
    "text_num_attention_heads": 20, "text_num_key_value_heads": 20,
}
HIDDEN = CONFIG["num_attention_heads"] * CONFIG["attention_head_dim"]
TARGETS = (("text", "txt_in.linear_2", HIDDEN), ("trunk", "final_layer.linear", CONFIG["in_channels"]))


def package(backend):
    if backend != BACKEND:
        raise ValueError("Krea native phase fixture requires Candle CUDA")
    return PACKAGE


def contract(backend):
    package(backend)
    return {
        "geometry": {"published_raw_revision": "4ad9f4b627a647fad78b3dfeebb09f2654aeb494",
                     "config": CONFIG, "published_block_depth": 28, "hidden": HIDDEN,
                     "image": [256, 256], "latent": [1, 16, 32, 32], "patch": 2,
                     "bf16_weight_payload_bytes": 2 * sum(map(math.prod, native_shapes().values()))},
        "settings": {"model": "krea_2_raw", "source": "pinned native single-file",
                     "compute_dtype": "BF16", "storage_dtype": "BF16", "device": "cuda",
                     "sampler": "dpmpp_2m_sde", "image_seed": SEED, "steps": [2, 3, 2],
                     "cfg": [0.0, 2.0, 0.0], "selected_slots": [[], [1, 0, 1], []],
                     "selected_scales": [[], [-0.5, 1.5, 0.25], []],
                     "targets": [target for _, target, _ in TARGETS],
                     "default_scales": [0.625, 0.875], "fixture_seed": fixture.SEED,
                     "residency": ["resident", "request-staged"]},
        "tolerance": {"class": "AP13(a)", "output": "bit-exact finite F32",
                      "provenance": "use the exact ENG-4a phase renderer and same fixture/settings"},
        "compared": ("registered Raw route and refusals; native DiT placement/conditioning/solver "
                     "through the production phase driver; deterministic caption inputs and identity decode"),
    }


def native_shapes():
    """The native checkpoint key layout with Raw widths and one image DiT block."""
    shapes = {}
    head_dim = CONFIG["attention_head_dim"]
    text_hidden = CONFIG["text_hidden_dim"]

    def linear(key, out_features, in_features, bias=False):
        shapes[f"{key}.weight"] = [out_features, in_features]
        if bias:
            shapes[f"{key}.bias"] = [out_features]

    def block(key, hidden, kv_heads, intermediate):
        for norm in ("prenorm", "postnorm"):
            shapes[f"{key}.{norm}.scale"] = [hidden]
        for norm in ("qnorm", "knorm"):
            shapes[f"{key}.attn.qknorm.{norm}.scale"] = [head_dim]
        kv_width = kv_heads * head_dim
        for projection, out_features in (("wq", hidden), ("wk", kv_width),
                                         ("wv", kv_width), ("gate", hidden), ("wo", hidden)):
            linear(f"{key}.attn.{projection}", out_features, hidden)
        linear(f"{key}.mlp.gate", intermediate, hidden)
        linear(f"{key}.mlp.up", intermediate, hidden)
        linear(f"{key}.mlp.down", hidden, intermediate)

    linear("first", HIDDEN, CONFIG["in_channels"], True)
    linear("tmlp.0", HIDDEN, CONFIG["timestep_embed_dim"], True)
    linear("tmlp.2", HIDDEN, HIDDEN, True)
    linear("tproj.1", 6 * HIDDEN, HIDDEN, True)
    shapes["txtmlp.0.scale"] = [text_hidden]
    linear("txtmlp.1", HIDDEN, text_hidden, True)
    linear("txtmlp.3", HIDDEN, HIDDEN, True)
    for stack, depth in (("layerwise_blocks", CONFIG["num_layerwise_text_blocks"]),
                         ("refiner_blocks", CONFIG["num_refiner_text_blocks"])):
        for index in range(depth):
            block(f"txtfusion.{stack}.{index}", text_hidden,
                  CONFIG["text_num_key_value_heads"], CONFIG["text_intermediate_size"])
    linear("txtfusion.projector", 1, CONFIG["num_text_layers"])
    block("blocks.0", HIDDEN, CONFIG["num_key_value_heads"], CONFIG["intermediate_size"])
    shapes["blocks.0.mod.lin"] = [6, HIDDEN]
    shapes["last.modulation.lin"] = [2, HIDDEN]
    shapes["last.norm.scale"] = [HIDDEN]
    linear("last.linear", CONFIG["in_channels"], HIDDEN, True)
    return {"model.diffusion_model." + key: shape for key, shape in sorted(shapes.items())}


def materialize(root):
    root.mkdir()
    (root / "transformer").mkdir()
    (root / "transformer/config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    fixture.write_component(root / "native.safetensors", native_shapes())
    for name, target, out_features in TARGETS:
        fixture.write_component(root / f"{name}.safetensors", {
            f"{target}.lora_A.weight": [4, HIDDEN], f"{target}.lora_B.weight": [out_features, 4],
        })
    fixture.write_component(root / "fold-only.safetensors", {
        "text_fusion.projector.diff": [1, CONFIG["num_text_layers"]],
    })
    return identify(root)


def identify(root):
    names = {path.relative_to(root).as_posix() for path in root.rglob("*") if not path.is_dir()}
    if names != FIXTURE_FILES:
        raise ValueError("unexpected Krea fixture files or absent pinned inputs")
    for name in FIXTURE_FILES:
        path = root / name
        unlinked_path(path, root)
        if not path.is_file() or path.lstat().st_nlink != 1:
            raise ValueError("Krea fixture must contain only regular files with one link")
    return {name: fixture.sha(root / name) for name in sorted(FIXTURE_FILES)}


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Krea phase overlay could not bind source seam {old}")
    return source.replace(old, new)


def overlay(read, arm, backend, binary):
    package(backend)
    if arm not in ("base", "head"):
        raise ValueError("unexpected Krea comparison arm")
    pipeline = f"{KREA}/pipeline.rs"
    source = read(pipeline).decode("utf-8")
    before, rest = source.split("fn render_multiphase_driver<T>(", 1)
    body, after = rest.split("pub(crate) fn render_multiphase(", 1)
    match = "Ok(v.to_dtype(DType::F32)?)"
    wrapper = after.split("\n}", 1)[0]
    if body.count(match) != 1 or wrapper.count("render_multiphase_driver(") != 1:
        raise ValueError("Krea overlay must bind the production phase renderer and its velocity")
    lib = f"{KREA}/lib.rs"
    lib_source = read(lib)
    if lib_source.count(b"pipeline::render_multiphase(") != 2:
        raise ValueError("the Candle Krea provider must route both placements through the observed renderer")
    body = body.replace(match, 'sga_krea_pair::tensor("velocity", &v)?;\n                        ' + match)
    source = (before + "fn render_multiphase_driver<T>(" + body
              + "pub(crate) fn render_multiphase(" + after
              + '\n#[path = "sga_krea_pair.rs"]\npub mod sga_krea_pair;\n')
    transformer = f"{KREA}/transformer/mod.rs"
    prep = "let mut ref_tokens = Vec::with_capacity(refs.len());"
    transformer_source = replace_once(read(transformer).decode("utf-8"), prep,
                                      'crate::pipeline::sga_krea_pair::tensor("prepare", &context)?;\n        ' + prep)
    rng = 'crate::execution::stream("curated_step_noise", sub);'
    sampler_source = replace_once(read(SAMPLER).decode("utf-8"), rng,
                                  rng + '\n        println!("KREA_NATIVE_PHASE_DRAW {seed} {step} {sub}");')
    arm_source = PROBE.read_text(encoding="utf-8")
    legacy, current = arm_source.split("// --- current arm ---\n")
    common, old = legacy.split("// --- baseline arm ---\n")
    return {
        pipeline: source.encode("utf-8"),
        lib: lib_source + b"\n#[doc(hidden)]\npub use pipeline::sga_krea_pair::compare_native_phases;\n",
        transformer: transformer_source.encode("utf-8"),
        SAMPLER: sampler_source.encode("utf-8"),
        f"{KREA}/sga_krea_pair.rs": (common + (old if arm == "base" else current)).encode("utf-8"),
        f"{KREA}/bin/{binary}.rs": b"fn main() { candle_gen_krea::compare_native_phases().unwrap(); }\n",
    }


def observe(stdout, result_path, backend):
    package(backend)
    if stdout.splitlines().count(COMPLETE) != 1:
        raise ValueError("Krea native phase comparison did not reach its terminal body")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (not isinstance(result, dict) or set(result) != {"device", "cases"}
            or not isinstance(result["device"], str)
            or not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", result["device"])):
        raise ValueError("Krea proof did not select a supervisor-provided CUDA UUID")
    cases = result["cases"]
    if (not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases)
            or [case.get("case") for case in cases] != list(CASE_ORDER)):
        raise ValueError("Krea fixture did not finish every selected placement and control")
    for case in cases:
        if (set(case) != {"case", "prepare", "velocity", "latent", "progress"}
                or len(case["prepare"]) != 4 or len(case["velocity"]) != 7
                or len(case["latent"]) != 1 or case["prepare"][0] != case["prepare"][3]
                or case["progress"] != [[1, 2], [2, 2], [1, 3], [2, 3], [3, 3], [1, 2], [2, 2], [0, 0]]
                or any(not re.fullmatch(r"[0-9a-f]{64}", value)
                       for field in ("prepare", "velocity", "latent") for value in case[field])):
            raise ValueError("incomplete Krea phase or incorrect base/CFG/decoder trace")
    for selected, bare in (cases[:2], cases[2:]):
        if (selected["prepare"][0] != bare["prepare"][0]
                or selected["prepare"][1] == bare["prepare"][1]
                or selected["velocity"][:2] != bare["velocity"][:2]
                or selected["latent"] == bare["latent"]):
            raise ValueError("Krea adaptation was absent or leaked into an initial base phase")
    if [{key: case[key] for key in case if key != "case"} for case in cases[:2]] != [
            {key: case[key] for key in case if key != "case"} for case in cases[2:]]:
        raise ValueError("resident and staged Krea paths disagree")
    draw_pattern = re.compile(r"^KREA_NATIVE_PHASE_DRAW (\d+) (\d+) (\d+)$", re.MULTILINE)
    draws = [tuple(map(int, match)) for match in draw_pattern.findall(stdout)]
    expected = [(SEED, index, (SEED + 0x9E3779B97F4A7C15 * (index + 1)) % 2**64)
                for index in (0, 1, 0, 1, 2, 0)] * len(cases)
    if draws != expected:
        raise ValueError("Krea stochastic draws did not restart at each declared phase")
    return {**result, "draws": draws}
