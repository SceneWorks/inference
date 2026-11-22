"""Source-bound real-weight LTX 2.5 tests through the native MLX provider."""

import json
from pathlib import Path
import re

from scripts.ci.ltx25_tier2_fixture import (
    ADAPTER, ADAPTER_HEADER, ADAPTER_SHA256, ADAPTER_TARGETS, edit_once,
    model_asset_contract, staged_assets as staged_assets,
)
from scripts.ci.sga_paired_performance import CEILING, INTERVAL_COVERAGE, PAIRS
from scripts.ci.wan_adapter_tier1_fixture import unique_object

CASE = {"dev": "ltx25-packed-q4-mlx-dev-v1", "distilled": "ltx25-packed-q4-mlx-distilled-v1"}
STAGE1 = {"dev": 30, "distilled": 8}
GIB = 2**30
CHILD_SECONDS = 900
HOST_RESERVE_BYTES = 8 * GIB
CHILD_FOOTPRINT_CAP_BYTES = 80 * GIB


def contract(variant):
    return {
        "model": model_asset_contract(variant),
        "settings": {"device": {"kind": "mlx-metal", "index": 0}, "native_case": CASE[variant],
                     "variant": variant, "tier": "q4", "precision": "shipping BF16 compute",
                     "seed": 18777, "width": 512, "height": 512, "frames": 17, "fps": 24,
                     "stage_steps": [STAGE1[variant], 3], "sampler_request": None,
                     "stage1_sampler": "native-guided-euler" if variant == "dev" else "native-distilled-euler",
                     "stage2_sampler": "native-distilled-euler", "decode": "convolutional video VAE and audio vocoder",
                     "adapter": ({"file": ADAPTER, "sha256": ADAPTER_SHA256, "header_sha256": ADAPTER_HEADER,
                                  "rank": 450, "alpha": 450, "targets": ADAPTER_TARGETS,
                                  "default_scale": 0, "pass_scales": [0, 1]} if variant == "dev" else None)},
        "comparison": {"numerics": "AP13(a), decoded RGB8 and finite stereo PCM f32 bits; reject degenerate outputs",
                       "denoise_timing": "synchronized sampler spans after each stage's input latents materialize",
                       "warmup": ["base", "head"],
                       "cache": "fresh process per sample, shared OS file cache after verified staging and paired warmup",
                       "pairs": PAIRS, "order": [["base", "head"], ["head", "base"]],
                       "time_ceiling": CEILING, "peak_ceiling": CEILING,
                       "median_interval": "min/max exact sign interval", "interval_coverage": INTERVAL_COVERAGE,
                       "child_deadline_seconds": CHILD_SECONDS},
    }


def policy(_variant, runner):
    from scripts.ci.sga_adapter_comparison import POLICY

    if runner != "nax":
        raise ValueError("native LTX MLX comparison requires the provisioned secondary Mac route")
    return dict(POLICY, backend="darwin-mlx", deadlineSeconds=CHILD_SECONDS,
                hostFreeReserveBytes=HOST_RESERVE_BYTES, childFootprintCapBytes=CHILD_FOOTPRINT_CAP_BYTES)


def package(backend):
    if backend != "mlx":
        raise ValueError("native LTX MLX comparison requires MLX")
    return "mlx-gen-ltx"


def overlay(read, arm, backend, binary):
    source = f"crates/media/mlx-gen/{package(backend)}/src"
    pipeline = f"{source}/pipeline.rs"
    staging = f"{source}/model/staging.rs"
    first_pass = {"base": b"    dit.set_lora_pass(0);",
                  "head": b"    let stage_one_scales = ScaleContext::new().with_pass(0);"}[arm]
    second_pass = {"base": b"    dit.set_lora_pass(1);",
                   "head": b"    let stage_two_scales = ScaleContext::new().with_pass(1);"}[arm]
    start = b"pub fn generate_av_latents_for_variant("
    stop = b"\n}\n\n/// A stage-1 in-context conditioning clip"
    before, rest = read(pipeline).split(start, 1)
    body, after = rest.split(stop, 1)
    body = edit_once(body, first_pass,
                     b"    let sga_stage1 = sga_ltx25_timer_start(&vlat1, audio_s1_noise)?;\n" + first_pass)
    upsample = b"    let v = upsample_latents(&v, upsampler, latent_mean, latent_std)?;"
    body = edit_once(body, upsample,
                     b"    sga_ltx25_timer_end(sga_stage1, &v, &a, 0, stage1_sigmas.len() - 1)?;\n" + upsample)
    body = edit_once(body, second_pass,
                     b"    let sga_stage2 = sga_ltx25_timer_start(&vlat2, &a)?;\n" + second_pass)
    body = edit_once(body, b"    match (sampler, vstate2.as_ref()) {",
                     b"    let (v, a) = match (sampler, vstate2.as_ref()) {")
    if not body.endswith(b"\n    }"):
        raise ValueError("LTX second native denoise no longer returns the stage's latent pair")
    body += (b"?;\n    sga_ltx25_timer_end(sga_stage2, &v, &a, 1, STAGE2_SIGMAS.len() - 1)?;"
             b"\n    Ok((v, a))")
    pipeline_data = before + start + body + stop + after + b'''

fn sga_ltx25_timer_start(video: &Array, audio: &Array) -> Result<std::time::Instant> {
    mlx_rs::transforms::eval([video, audio])?;
    Ok(std::time::Instant::now())
}

fn sga_ltx25_timer_end(start: std::time::Instant, video: &Array, audio: &Array,
                       stage: u32, steps: usize) -> Result<()> {
    mlx_rs::transforms::eval([video, audio])?;
    println!("SGA_LTX_STAGE {stage} {steps} {}", start.elapsed().as_nanos());
    Ok(())
}
'''
    install = b'''                crate::adapters::apply_ltx25_adapters(
                    &mut transformer,
                    &self.adapters,
                    crate::pipeline::NUM_DENOISE_PASSES,
                )?;'''
    staging_data = edit_once(read(staging), install,
                             install.replace(b"crate::adapters::", b"let applied = crate::adapters::")
                             + b'\n                println!("SGA_LTX_ADAPTER {} {}", applied.applied, applied.skipped.len());')
    probe = Path(__file__).with_name("ltx25_tier2_mlx.rs.in").read_bytes()
    return {pipeline: pipeline_data, staging: staging_data, f"{source}/bin/{binary}.rs": probe}


def preflight(stdout, variant):
    expected = f"SGA_LTX_NATIVE_CASE {CASE[variant]}"
    if stdout.splitlines() != [expected]:
        raise ValueError("LTX MLX bundle preflight did not complete on the exact native case")
    return {"case": CASE[variant], "steps": [STAGE1[variant], 3]}


def observe(stdout, result, runner, variant):
    if runner != "nax":
        raise ValueError("unexpected native LTX MLX runner route")
    lines = stdout.splitlines()
    if lines.count(f"SGA_LTX_REAL_COMPLETE {CASE[variant]}") != 1:
        raise ValueError("missing completed native MLX LTX result")
    adapters = [line for line in lines if line.startswith("SGA_LTX_ADAPTER ")]
    if adapters != ([f"SGA_LTX_ADAPTER {ADAPTER_TARGETS} 0"] if variant == "dev" else []):
        raise ValueError("native LTX did not report the declared refinement adapter installation")
    measurements = [line for line in lines if line.startswith("SGA_LTX_STAGE ")]
    stages = [re.fullmatch(r"SGA_LTX_STAGE (0|1) ([1-9][0-9]?) ([1-9][0-9]{0,14})", line) for line in measurements]
    steps = [STAGE1[variant], 3]
    if (len(stages) != 2 or any(stage is None for stage in stages)
            or [(int(stage[1]), int(stage[2])) for stage in stages] != list(enumerate(steps))):
        raise ValueError("missing synchronized MLX denoise spans or mismatched step counts")
    durations = [int(stage[3]) for stage in stages]
    row = json.loads(result.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    total = sum(steps)
    if (set(row) != {"case", "device", "progress", "e2e_ns", "video", "audio"}
            or row["case"] != CASE[variant] or row["device"] != {"kind": "mlx-metal", "index": 0}
            or row["progress"] != [[step, total] for step in range(1, total + 1)]
            or type(row["e2e_ns"]) is not int or not sum(durations) < row["e2e_ns"] <= CHILD_SECONDS * 10**9):
        raise ValueError("native MLX LTX device, progress or runtime differs from its declared route")
    video, audio = row["video"], row["audio"]
    if (set(video) != {"size", "frames", "fps", "rgb_sha256"} or video["size"] != [512, 512]
            or (video["frames"], video["fps"]) != (17, 24) or not re.fullmatch(r"[0-9a-f]{64}", video["rgb_sha256"])
            or set(audio) != {"channels", "sample_rate", "samples", "f32_sha256"} or audio["channels"] != 2
            or type(audio["sample_rate"]) is not int or audio["sample_rate"] <= 0
            or type(audio["samples"]) is not int or audio["samples"] <= 0 or audio["samples"] % 2
            or not re.fullmatch(r"[0-9a-f]{64}", audio["f32_sha256"])):
        raise ValueError("native MLX LTX decoded image/audio receipt is malformed")
    return {"identity": {"video": video, "audio": audio, "steps": steps,
                         "adapter_targets": ADAPTER_TARGETS if variant == "dev" else 0},
            "timing": {"e2e_ns": row["e2e_ns"], "denoise_ns_per_outer_step": sum(durations) / total,
                       "base_ns_per_outer_step": durations[0] / steps[0], "refinement_ns_per_outer_step": durations[1] / steps[1]}}
