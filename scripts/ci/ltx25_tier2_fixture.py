"""Frozen native LTX 2.5 Candle cases for the ENG-4b real-weight comparison."""

import hashlib
import json
from pathlib import Path
import re
import struct

from scripts.ci import stage_pinned_hub_snapshots as hub
from scripts.ci.sga_paired_performance import CEILING, INTERVAL_COVERAGE, PAIRS
from scripts.ci.wan_adapter_tier1_fixture import fetch_pinned_header, unique_object

REPOSITORY = "SceneWorks/ltx-2.5-mlx"
REVISION = "081658ce6886cacba20817ce0359bbefef706ff2"
ADAPTER = "distilled_lora/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
ADAPTER_SHA256 = "86370bbf79a9eb4edaa158907e2b48a5188fe4c5dc8ce30c7eb8f2f131a9bbf5"
ADAPTER_HEADER = "715a1cdcba6078f4e7bdeac1c0820954f331fc2a381ec04bf41a586e7a74d42f"
ADAPTER_TARGETS = 1660
HEADER = {"dev": "7ae3673a1ec39ace021c3b816c07dc548e45966f74baca32f6dc4dfe8dcc1274",
          "distilled": "4e7dc8ce86a0367fade5f968eb3388fc5b276b5568f6bc4776d8d85ebfcdc47f"}
CASE = {"dev": "ltx25-packed-q4-blackwell-dev-v1", "distilled": "ltx25-packed-q4-blackwell-v1"}
BUILD_FEATURES = "cuda,terminal-quant-measurement"
HUB_ENV = "LTX25_PROOF_HUB_ROOT"
STAGE1 = {"dev": 30, "distilled": 8}
SCHEDULE = {"dev": "ebd978a8584cd008a7e91be4bd2069fac1335e88fc2ea723ea89ea00ce872d5c",
            "distilled": "ab94365300112897cef3abb66b7c08c851fd4aa345349d64fbcc177873fe1ce8",
            "refinement": "1703669e7d66775e40fb2520f307d0e48041de8910da1aee71171f5995698dd7"}
FILES = frozenset({"audio_vae.safetensors", "connector.safetensors", "diffusion_vae_encoder.safetensors",
                   "duration_head.safetensors", "embedded_config.json", "spatial_upsampler.safetensors",
                   "split_model.json", "temporal_upsampler.safetensors", "text_encoder.safetensors",
                   "transformer.safetensors", "vae_decoder.safetensors", "vae_diffusion_decoder.safetensors",
                   "vae_encoder.safetensors", "vocoder.safetensors"})
GIB = 2**30
POOL_BYTES = 80 * GIB
CHILD_SECONDS = 900
PAIR_ORDER = ("base", "head"), ("head", "base")


def contract(variant):
    return {"model": {"repository": REPOSITORY, "revision": REVISION, "bundle": f"{variant}/q4",
                      "transformer_header_sha256": HEADER[variant]},
            "settings": {"device_capability": "sm_120", "native_case": CASE[variant], "variant": variant,
                         "tier": "q4", "precision": "shipping BF16 DiT and native decoder dtypes",
                         "seed": 18777, "width": 512, "height": 512, "frames": 17, "fps": 24,
                         "stage_steps": [STAGE1[variant], 3], "stage1_sampler": "native-euler" if variant == "dev" else "rectified-flow",
                         "sigma_f32_le_sha256": [SCHEDULE[variant], SCHEDULE["refinement"]],
                         "decoder": "CausalDiffusionVAE selected by the native terminal bundle",
                         "adapter": ({"file": ADAPTER, "sha256": ADAPTER_SHA256, "header_sha256": ADAPTER_HEADER,
                                      "rank": 450, "alpha": 450, "targets": ADAPTER_TARGETS, "pass_scales": [0, 1]}
                                     if variant == "dev" else None),
                         "pool_cap_bytes": POOL_BYTES, "cublas_workspace": ":4096:8"},
            "comparison": {"numerics": "AP13(a), decoded RGB8 and finite stereo PCM f32 bits; reject degenerate outputs",
                           "warmup": ["base", "head"], "cache": "fresh process per sample, shared OS file cache after staged verification and unscored paired warmup",
                           "pairs": PAIRS, "order": [list(order) for order in PAIR_ORDER], "time_ceiling": CEILING,
                           "peak_ceiling": CEILING, "median_interval": "min/max exact sign interval",
                           "interval_coverage": INTERVAL_COVERAGE, "child_deadline_seconds": CHILD_SECONDS}}


def package(_backend):
    return "candle-gen-ltx"


def policy(_variant, uuid):
    from scripts.ci.sga_adapter_comparison import POLICY

    return dict(POLICY, backend="windows-cuda", cudaDeviceUuid=uuid, deadlineSeconds=CHILD_SECONDS,
                hostFreeReserveBytes=8 * GIB, childFootprintCapBytes=56 * GIB,
                gpuFreeReserveBytes=4 * GIB, childGpuCapBytes=88 * GIB)


def edit_once(source, marker, replacement):
    if source.count(marker) != 1:
        raise ValueError(f"expected one source-owned LTX Tier 2 boundary for {marker!r}")
    return source.replace(marker, replacement)


def overlay(read, _arm, backend, binary):
    if backend != "candle-cuda":
        raise ValueError("LTX Tier 2 fixture requires the native Candle CUDA provider")
    source = "crates/media/candle-gen/candle-gen-ltx/src"
    measurement = f"{source}/quant_measurement.rs"
    campaign = edit_once(read(measurement), b"\nfn terminal_generation_request(", b"\npub fn terminal_generation_request(")
    lib = f"{source}/lib.rs"
    data = read(lib)
    adapter = b"        adapters::install_ltx25_adapters(&mut avdit, adapters)?;"
    data = edit_once(data, adapter, b"        let sga_adapters = adapters::install_ltx25_adapters(&mut avdit, adapters)?;\n"
                     b"        println!(\"SGA_LTX_ADAPTER {} {}\", sga_adapters.applied, sga_adapters.skipped_keys);")
    # Mark the existing render's two compute spans. Other render routes are outside this fixture.
    for marker, event in (
        (b"        let (vlat, alat) = if self.transformer_variant.is_dev() {", b"        let sga_stage1 = sga_ltx25_timer_start(&self.device)?;\n"),
        (b"        // The denoised half-resolution latent enters the learned upsampler", b"        sga_ltx25_timer_end(&self.device, sga_stage1, 0, &stage1_plan.sigmas)?;\n\n"),
        (b"        let stage2 = if stage2_keyframes.is_empty() {", b"        let sga_stage2 = sga_ltx25_timer_start(&self.device)?;\n"),
        (b"        let (vlat, alat) = (stage2.video, stage2.audio);", b"        sga_ltx25_timer_end(&self.device, sga_stage2, 1, &STAGE2_SIGMAS)?;\n"),
    ):
        data = edit_once(data, marker, event + marker)
    data += b'''
fn sga_ltx25_timer_start(device: &Device) -> CResult<std::time::Instant> {
    device.synchronize()?;
    Ok(std::time::Instant::now())
}
pub fn sga_ltx25_schedule_hash(sigmas: &[f32]) -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    for sigma in sigmas {
        hash.update(sigma.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}
fn sga_ltx25_timer_end(device: &Device, start: std::time::Instant, phase: u32, sigmas: &[f32]) -> CResult<()> {
    device.synchronize()?;
    let elapsed = start.elapsed().as_nanos();
    println!("SGA_LTX_STAGE {phase} {} {elapsed} {}", sigmas.len() - 1, sga_ltx25_schedule_hash(sigmas));
    Ok(())
}
'''
    probe = Path(__file__).with_name("ltx25_tier2_candle.rs.in").read_bytes()
    return {lib: data, measurement: campaign, f"{source}/bin/{binary}.rs": probe}


def projection_shape(transformer, canonical):
    # The converted-tier VarBuilder in both sources applies these exact key rewrites.
    key = ((canonical + ".").replace(".to_out.0.", ".to_out.").replace(".net.0.proj.", ".proj_in.")
           .replace(".net.2.", ".proj_out.").replace(".linear_1.", ".linear1.").replace(".linear_2.", ".linear2."))
    weight = transformer.get(key + "weight", {})
    shape = weight.get("shape", [])
    if len(shape) != 2 or weight.get("dtype") not in ("BF16", "U32"):
        raise ValueError(f"missing or malformed LTX adapter target {canonical}")
    out_dim, in_dim = shape
    if weight["dtype"] == "U32":
        in_dim *= 8
        if transformer.get(key + "scales", {}).get("shape") != [out_dim, in_dim // 64]:
            raise ValueError(f"LTX packed target group changed for {canonical}")
    return out_dim, in_dim


def verify_headers(variant, transformer_raw, adapter_raw=None):
    if hashlib.sha256(transformer_raw).hexdigest() != HEADER[variant]:
        raise ValueError("LTX transformer geometry changed")
    transformer = json.loads(transformer_raw, object_pairs_hook=unique_object)
    meta = transformer.pop("__metadata__")
    if (meta.get("model_version"), meta.get("sceneworks_tier"), meta.get("variant")) != ("2.5.0", "q4", variant):
        raise ValueError("LTX transformer version, quantization, or route changed")
    if variant == "distilled":
        if adapter_raw is not None:
            raise ValueError("distilled LTX checkpoint already contains the refinement recipe")
        return
    if adapter_raw is None or hashlib.sha256(adapter_raw).hexdigest() != ADAPTER_HEADER:
        raise ValueError("missing exact LTX refinement adapter header")
    adapter = json.loads(adapter_raw, object_pairs_hook=unique_object)
    meta = adapter.pop("__metadata__")
    if (meta.get("lora_rank"), meta.get("lora_alpha")) != ("450", "450") or len(adapter) != 2 * ADAPTER_TARGETS:
        raise ValueError("LTX refinement adapter scaling or target count changed")
    for name, row in adapter.items():
        if not name.startswith("diffusion_model.") or not name.endswith((".lora_A.weight", ".lora_B.weight")):
            raise ValueError(f"unexpected LTX refinement adapter key {name}")
        canonical, role, _ = name.removeprefix("diffusion_model.").rsplit(".", 2)
        out_dim, in_dim = projection_shape(transformer, canonical)
        rank = min(450, out_dim, in_dim)
        expected = [rank, in_dim] if role == "lora_A" else [out_dim, rank]
        other_role = "lora_B" if role == "lora_A" else "lora_A"
        if row.get("dtype") != "BF16" or row.get("shape") != expected or f"diffusion_model.{canonical}.{other_role}.weight" not in adapter:
            raise ValueError(f"LTX refinement adapter factor does not match {canonical}")


def fetch_geometry(variant):
    transformer = fetch_pinned_header(REPOSITORY, REVISION, f"{variant}/q4/transformer.safetensors", HEADER[variant])
    adapter = fetch_pinned_header(REPOSITORY, REVISION, ADAPTER, ADAPTER_HEADER) if variant == "dev" else None
    verify_headers(variant, transformer, adapter)


def read_header(path):
    with path.open("rb") as source:
        length, = struct.unpack("<Q", source.read(8))
        if not 0 < length <= 2**20:
            raise ValueError("LTX local metadata is outside the header budget")
        return source.read(length)


def staged_assets(variant, model_hub, *, stage=True):
    bundle = f"{variant}/q4"
    groups = {"bundle": {"repository": REPOSITORY, "revision": REVISION, "subfolder": bundle}}
    if variant == "dev":
        groups["refinement"] = {"repository": REPOSITORY, "revision": REVISION, "subfolder": "distilled_lora"}
    snapshot = hub.snapshot_dir(model_hub, REPOSITORY, REVISION)
    _, _, metadata = hub.pinned_manifest(REPOSITORY, REVISION)
    expected = {f"{bundle}/{name}" for name in FILES}
    if set(hub.pinned_files(metadata, bundle)) != expected:
        raise ValueError("unexpected LTX bundle closure at the pinned revision")
    if variant == "dev":
        if set(hub.pinned_files(metadata, "distilled_lora")) != {ADAPTER}:
            raise ValueError("unexpected LTX refinement adapter closure")
        expected.add(ADAPTER)
    if stage:
        hub.stage(groups, model_hub, manifest=lambda *_: (None, None, metadata))
    verified = hub.verify_pinned_hub_files(REPOSITORY, REVISION, snapshot, tuple(sorted(expected)))
    verify_headers(variant, read_header(snapshot / bundle / "transformer.safetensors"),
                   read_header(snapshot / ADAPTER) if variant == "dev" else None)
    return snapshot, {"manifest_sha256": verified["manifest_sha256"], "files": verified["files"]}


def preflight(stdout, variant):
    if stdout.splitlines() != [f"SGA_LTX_NATIVE_CASE {CASE[variant]} {SCHEDULE[variant]} {SCHEDULE['refinement']}"]:
        raise ValueError("LTX native bundle preflight did not complete on the exact campaign case")


def observe(stdout, result, uuid, variant):
    lines = stdout.splitlines()
    if lines.count(f"SGA_LTX_REAL_COMPLETE {CASE[variant]}") != 1:
        raise ValueError("missing completed native LTX result")
    adapters = [line for line in lines if line.startswith("SGA_LTX_ADAPTER ")]
    if adapters != [f"SGA_LTX_ADAPTER {ADAPTER_TARGETS if variant == 'dev' else 0} 0"]:
        raise ValueError("native LTX reported unexpected installed adapter targets")
    measurements = [line for line in lines if line.startswith("SGA_LTX_STAGE ")]
    stages = [re.fullmatch(r"SGA_LTX_STAGE (0|1) ([1-9][0-9]?) ([1-9][0-9]{0,14}) ([0-9a-f]{64})", line)
              for line in measurements]
    steps = [STAGE1[variant], 3]
    if (len(stages) != 2 or not all(stages)
            or [(int(stage[1]), int(stage[2])) for stage in stages] != list(enumerate(steps))
            or [stage[4] for stage in stages] != [SCHEDULE[variant], SCHEDULE["refinement"]]):
        raise ValueError("missing synchronized LTX denoise stage timing or mismatched step counts")
    durations = [int(stage[3]) for stage in stages]
    row = json.loads(result.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    total = sum(steps)
    if (set(row) != {"case", "cuda_device_uuid", "progress", "e2e_ns", "video", "audio"}
            or row["case"] != CASE[variant] or row["cuda_device_uuid"] != uuid
            or row["progress"] != [[step, total] for step in range(1, total + 1)]
            or type(row["e2e_ns"]) is not int or not sum(durations) < row["e2e_ns"] <= CHILD_SECONDS * 10**9):
        raise ValueError("native LTX runtime, device, or output receipt differs from the declared route")
    video, audio = row["video"], row["audio"]
    if (set(video) != {"size", "frames", "fps", "rgb_sha256"} or video["size"] != [512, 512]
            or (video["frames"], video["fps"]) != (17, 24) or not re.fullmatch(r"[0-9a-f]{64}", video["rgb_sha256"])
            or set(audio) != {"channels", "sample_rate", "samples", "f32_sha256"} or audio["channels"] != 2
            or type(audio["sample_rate"]) is not int or audio["sample_rate"] <= 0
            or type(audio["samples"]) is not int or audio["samples"] <= 0 or audio["samples"] % 2
            or not re.fullmatch(r"[0-9a-f]{64}", audio["f32_sha256"])):
        raise ValueError("native LTX decoded image/audio digests or shape are malformed")
    return {"identity": {"video": video, "audio": audio, "steps": steps, "adapter_targets": ADAPTER_TARGETS if variant == "dev" else 0},
            "timing": {"e2e_ns": row["e2e_ns"], "denoise_ns_per_outer_step": sum(durations) / total,
                       "base_ns_per_outer_step": durations[0] / steps[0], "refinement_ns_per_outer_step": durations[1] / steps[1]}}
