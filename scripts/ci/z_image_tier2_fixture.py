"""Native Candle Z-Image Tier 2 bindings for the shared terminal pair controller."""

import hashlib
import json
from pathlib import Path
import struct
import zlib

from scripts.ci import stage_pinned_hub_snapshots as hub
from scripts.ci.pinned_hub_source import file_inventory_seal, pinned_file_identity
from scripts.ci.sga_adapter_comparison import bounded_owned_file
from scripts.ci.sga_paired_performance import CEILING, INTERVAL_COVERAGE, PAIRS
from scripts.ci.veda_extract_frame_png import png
from scripts.ci.wan_adapter_tier1_fixture import unique_object
from scripts.ci.z_image_tier1_fixture import cuda_receipt, replace_once

PROFILE = "z-image"
CASE = {"turbo": "z_image_turbo", "base": "z_image"}
BUILD_FEATURES = "cuda"
SIZE = {"turbo": 256, "base": 512}
STEPS = {"turbo": 4, "base": 12}
PROMPT = {"turbo": "a fox",
          "base": "a photo of a rusty robot holding a lit candle, dramatic cinematic lighting"}
NEGATIVE = {"turbo": None, "base": "blurry, low quality, watermark"}
GUIDANCE = {"turbo": None, "base": 4.0}
SEED = 42
ARTIFACT_LIMITS = {"image.png": 512 * 512 * 3 + 8192}
MODELS = {
    "turbo": ("Tongyi-MAI/Z-Image-Turbo", "f332072aa78be7aecdf3ee76d5c247082da564a6",
              "deb57c388c0ba16250156fd5da4d9cf78bb28147641274bff645b21682bf1db6", 32_848_305_533),
    "base": ("Tongyi-MAI/Z-Image", "04cc4abb7c5069926f75c9bfde9ef43d49423021",
             "ee56d28be7a1a49fb80fbaace260bfa4a56f19b43d622425193263ca2554b3dc", 20_538_488_559),
}
COMPONENTS = ("scheduler", "text_encoder", "tokenizer", "transformer", "vae")
GIB = 2**30
MIB = 2**20
POOL_BYTES = 30 * GIB
CHILD_SECONDS = 300


def package(backend):
    if backend != "candle-cuda":
        raise ValueError("Z-Image terminal fixture requires native Candle CUDA")
    return "candle-gen-z-image"


def policy(variant, uuid):
    MODELS[variant]
    return {"schemaVersion": 1, "backend": "windows-cuda", "cudaDeviceUuid": uuid,
            "deadlineSeconds": CHILD_SECONDS, "pollMillis": 1000, "termGraceMillis": 5000,
            "hostFreeReserveBytes": 8 * GIB, "gpuFreeReserveBytes": 4 * GIB,
            "childFootprintCapBytes": 48 * GIB, "childGpuCapBytes": 34 * GIB,
            "stdoutCapBytes": MIB, "stderrCapBytes": MIB, "eventCapBytes": 4 * MIB}


def request(variant):
    return {"provider": CASE[variant], "width": SIZE[variant], "height": SIZE[variant],
            "count": 1, "steps": STEPS[variant], "seed": SEED, "prompt": PROMPT[variant],
            "negative_prompt": NEGATIVE[variant], "guidance": GUIDANCE[variant]}


def contract(variant):
    repo, revision, seal, weight_bytes = MODELS[variant]
    return {
        "model": {"repository": repo, "revision": revision, "files_sha256": seal,
                  "asset_bytes": weight_bytes, "license": "Apache-2.0"},
        "settings": {**request(variant), "sampler": "Euler", "schedule": "native static shift "
                     + ("3" if variant == "turbo" else "6"), "dtype": "BF16",
                     "transformer_storage_dtype": "F32" if variant == "turbo" else "BF16",
                     "placement": "resident", "decoder": "native VAE", "adapters": [],
                     "device_capability": "sm_120", "pool_cap_bytes": POOL_BYTES,
                     "cublas_workspace": ":4096:8"},
        "comparison": {"numerics": "AP13(a), identical native plan, sampler sigma bits, and nondegenerate RGB8",
                       "warmup": ["base", "head"], "cache": "fresh process with warmed OS file cache",
                       "pairs": PAIRS, "order": [["base", "head"], ["head", "base"]],
                       "phase_breakdown": "diagnostic wall time at provider load, synchronized denoise and decode boundaries",
                       "time_ceiling": CEILING, "peak_ceiling": CEILING,
                       "median_interval": "min/max exact sign interval",
                       "interval_coverage": INTERVAL_COVERAGE, "child_deadline_seconds": CHILD_SECONDS},
    }


def overlay(read, _arm, backend, binary):
    family = f"crates/media/candle-gen/{package(backend)}/src"
    shared = "crates/media/candle-gen/candle-gen/src/sampler.rs"
    source = read(shared)
    begin = b"    run_flow_sampler_with_phase_transition_spans(\n        sampler_name,\n"
    end = b"        predict,\n        None,\n    )\n}\n\n#[allow(clippy::too_many_arguments)]\npub fn run_flow_sampler_with_phase_transition_spans("
    source = replace_once(source, begin, b"""    let device = latents.device().clone();
    device.synchronize()?;
    let start = std::time::Instant::now();
    let out = run_flow_sampler_with_phase_transition_spans(
        sampler_name,
""")
    source = replace_once(source, end, b"""        predict,
        None,
    )?;
    out.device().synchronize()?;
    let ended = std::time::Instant::now();
    let sample = ZImageTerminalSample {
        started: start, ended, device, dtype: out.dtype(), shape: out.dims().to_vec(),
        sigmas: sigmas.iter().map(|v| v.to_bits()).collect(),
    };
    Z_IMAGE_TERMINAL_SAMPLE.with(|slot| assert!(slot.replace(Some(sample)).is_none()));
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn run_flow_sampler_with_phase_transition_spans(""")
    source += b"""
pub struct ZImageTerminalSample {
    pub started: std::time::Instant,
    pub ended: std::time::Instant,
    pub device: candle_core::Device,
    pub dtype: candle_core::DType,
    pub shape: Vec<usize>,
    pub sigmas: Vec<u32>,
}
thread_local! {
    static Z_IMAGE_TERMINAL_SAMPLE: std::cell::RefCell<Option<ZImageTerminalSample>> =
        const { std::cell::RefCell::new(None) };
}
pub fn take_z_image_terminal_sample() -> Option<ZImageTerminalSample> {
    Z_IMAGE_TERMINAL_SAMPLE.with(|slot| slot.take())
}
"""
    probe = Path(__file__).with_name("z_image_tier2_candle.rs.in").read_bytes()
    receipt = replace_once(cuda_receipt(), b"(0, 10 * 2_u64.pow(30))",
                           f"(0, {POOL_BYTES})".encode())
    probe = replace_once(probe, b"// __SGA_CUDA_RECEIPT__", receipt)
    return {shared: source, f"{family}/bin/{binary}.rs": probe}


def asset_inventory(variant, upstream):
    files = []
    for name, row in sorted(upstream.items()):
        if name != "model_index.json" and name.split("/")[0] not in COMPONENTS:
            continue
        files.append(pinned_file_identity(name, row))
    if file_inventory_seal(files) != MODELS[variant][2] or sum(row["bytes"] for row in files) != MODELS[variant][3]:
        raise ValueError("Z-Image snapshot component inventory differs from its approved revision")
    return files


def staged_assets(variant, model_hub, *, stage=True):
    repo, revision, seal, _ = MODELS[variant]
    destination = hub.snapshot_dir(model_hub, repo, revision)
    _, _, upstream = hub.pinned_manifest(repo, revision)
    files = asset_inventory(variant, upstream)
    names = {row["path"] for row in files}
    if stage:
        pin = {"repository": repo, "revision": revision, "subfolder": ""}
        verified = hub.stage({"checkpoint": pin}, model_hub,
                             manifest=lambda *_: (None, None, {name: upstream[name] for name in names}))
        observed = verified["checkpoint"]["files"]
    else:
        observed = hub.verify_pinned_hub_files(repo, revision, destination, tuple(sorted(names)))["files"]
    if observed != files:
        raise ValueError("authenticated Z-Image snapshot bytes differ from the approved inventory")
    # The native component builders enumerate the safetensors in these three local directories.
    for component in ("text_encoder", "transformer", "vae"):
        expected = {name for name in names if name.startswith(component + "/") and name.endswith(".safetensors")}
        actual = {f"{component}/{entry.name}" for entry in (destination / component).iterdir()
                  if entry.suffix == ".safetensors"}
        if actual != expected:
            raise ValueError(f"unbound Z-Image shard in {component}")
    # Hub response counters can change; authenticate and retain the stable selected file inventory.
    return destination, {"repository": repo, "revision": revision, "files_sha256": seal, "files": files}


def verify_plan(row, variant):
    expected = request(variant)
    if not isinstance(row, dict) or set(row) != expected.keys() | {"sigmas_bits"} or any(
            type(row[key]) is not type(value) or row[key] != value for key, value in expected.items()):
        raise ValueError("Z-Image planned built-in request settings differ")
    bits = row["sigmas_bits"]
    if not isinstance(bits, list) or len(bits) != STEPS[variant] + 1 or not all(
            type(value) is int and 0 <= value <= 0x3f800000 for value in bits):
        raise ValueError("Z-Image native schedule bits are invalid")
    sigmas = [struct.unpack("<f", struct.pack("<I", value))[0] for value in bits]
    if bits[-1] != 0 or any(left <= right for left, right in zip(sigmas, sigmas[1:])):
        raise ValueError("Z-Image native schedule must decrease to zero")
    return row


def preflight(stdout, variant):
    prefix = "SGA_TERMINAL_PREFLIGHT "
    rows = [line.removeprefix(prefix) for line in stdout.splitlines() if line.startswith(prefix)]
    if len(rows) != 1:
        raise ValueError("missing unique Z-Image model-free preflight")
    return verify_plan(json.loads(rows[0], object_pairs_hook=unique_object), variant)


def prepare_artifacts(output, work, variant):
    # Leave raw RGB in owned scratch. Archive its lossless PNG after observation validates it.
    size = SIZE[variant]
    expected = size * size * 3
    source = bounded_owned_file(output / "image.rgb", work, expected)
    with source.open("rb") as stream:
        pixels = stream.read(expected + 1)
    if len(pixels) != expected:
        raise ValueError("Z-Image did not retain a complete RGB8 frame")
    target = output / "image.png"
    with target.open("xb") as stream:
        stream.write(png(size, size, pixels))
    if read_artifact_pixels(target, size) != pixels:
        raise ValueError("Z-Image PNG changed the measured RGB8 frame")


def read_artifact_pixels(image, size):
    # The owned encoder writes only IHDR/IDAT/IEND with unfiltered RGB8 rows.
    limit = ARTIFACT_LIMITS["image.png"]
    with image.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Z-Image PNG exceeds its evidence byte limit")
    signature = b"\x89PNG\r\n\x1a\n"
    if not raw.startswith(signature):
        raise ValueError("Z-Image evidence lacks a PNG signature")
    offset = len(signature)
    chunks = []
    for expected in (b"IHDR", b"IDAT", b"IEND"):
        if offset + 12 > len(raw):
            raise ValueError("Z-Image PNG chunk is truncated")
        count, kind = struct.unpack_from(">I4s", raw, offset)
        payload_start, end = offset + 8, offset + 8 + count
        if kind != expected or end + 4 > len(raw):
            raise ValueError("Z-Image PNG chunk is invalid")
        payload = raw[payload_start:end]
        if zlib.crc32(kind + payload) != struct.unpack_from(">I", raw, end)[0]:
            raise ValueError("Z-Image PNG chunk checksum differs")
        chunks.append(payload)
        offset = end + 4
    header, compressed, trailer = chunks
    if offset != len(raw) or trailer or header != struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0):
        raise ValueError("Z-Image PNG geometry or chunk layout differs from the RGB8 frame")
    stride = 1 + 3 * size
    decoder = zlib.decompressobj()
    try:
        rows = decoder.decompress(compressed, size * stride + 1)
    except zlib.error as error:
        raise ValueError("Z-Image PNG image data is invalid") from error
    if (not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or len(rows) != size * stride
            or any(rows[::stride])):
        raise ValueError("Z-Image PNG does not contain the exact unfiltered RGB8 rows")
    return b"".join(rows[offset + 1:offset + stride] for offset in range(0, len(rows), stride))


def observe(stdout, result, _uuid, variant):
    if stdout.splitlines().count("SGA_Z_IMAGE_TIER2_BODY " + variant) != 1:
        raise ValueError("missing completed resident Z-Image generation")
    row = json.loads(result.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(row, dict) or set(row) != {"plan", "sample", "progress", "e2e_ns", "denoise_ns", "phase_ns"}:
        raise ValueError("invalid Z-Image terminal receipt")
    plan = verify_plan(row["plan"], variant)
    sample = {"dtype": "BF16", "device": "candle-cuda:0", "sigmas_bits": plan["sigmas_bits"],
              "latent_shape": [1, 16, 1, SIZE[variant] // 8, SIZE[variant] // 8]}
    if row["sample"] != sample or row["progress"] != list(range(1, STEPS[variant] + 1)):
        raise ValueError("Z-Image sampler, device, dtype, or progress differs from the plan")
    end_to_end, denoise = row["e2e_ns"], row["denoise_ns"]
    if (type(end_to_end) is not int or type(denoise) is not int
            or not STEPS[variant] <= denoise <= end_to_end <= CHILD_SECONDS * 10**9):
        raise ValueError("Z-Image native timing receipt is malformed or exceeds the child budget")
    phases = row["phase_ns"]
    if (not isinstance(phases, dict) or set(phases) != {"provider_load", "render_before_denoise",
                                                       "after_denoise_before_decode", "decode_through_host_rgb"}
            or any(type(value) is not int or value < 0 for value in phases.values())
            or sum(phases.values()) + denoise != end_to_end):
        raise ValueError("Z-Image render phases do not account for measured time")
    image_path = result.with_name(result.name.removesuffix("result.json") + "image.png")
    image = read_artifact_pixels(image_path, SIZE[variant])
    mean = sum(image) / len(image)
    variance = sum(value * value for value in image) / len(image) - mean ** 2
    if not 8 < mean < 248 or variance <= 64 or image == image[:3] * (len(image) // 3):
        raise ValueError("Z-Image emitted a saturated or nearly flat RGB8 frame")
    return {"identity": {"plan": plan, "sample": sample, "progress": row["progress"],
                         "rgb_sha256": hashlib.sha256(image).hexdigest()},
            "timing": {"e2e_ns": end_to_end, "denoise_per_step_ns": denoise // STEPS[variant]}}
