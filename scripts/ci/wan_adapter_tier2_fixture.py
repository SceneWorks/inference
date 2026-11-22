"""Wan product A14B T2V q4 with the real Lightning pair on each native device."""

import json
from pathlib import Path
import re
from statistics import fmean
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import sga_paired_performance as perf
from scripts.ci import stage_pinned_hub_snapshots as hub
from scripts.ci.pinned_hub_source import pinned_manifest, verify_pinned_hub_files
from scripts.ci.wan_adapter_tier1_fixture import PACKAGE, REPOSITORY, REVISION, unique_object

CASE = {"a14b-t2v-mlx": "wan-a14b-t2v-q4-lightning-mlx",
        "a14b-t2v-candle": "wan-a14b-t2v-q4-lightning-candle"}
VARIANT_BACKENDS = {"a14b-t2v-mlx": "mlx", "a14b-t2v-candle": "candle-cuda"}
HUB_ENV = "FLUX1_HUB"
GIB = 2**30
POOL_BYTES = 24 * GIB
CHILD_SECONDS = {"mlx": 300, "candle-cuda": 240}
MLX_NATIVE_DECODED_FRAMES = 12
STAGE_TIMEOUT_SECONDS = 2700
RECHECK_TIMEOUT_SECONDS = 600
LIGHTNING_DIR = "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1"
WEIGHT_PINS = {
    "mlx": {"repository": REPOSITORY, "revision": REVISION, "subfolder": "q4"},
    "candle-cuda": {"repository": "SceneWorks/wan2.2-t2v-a14b-candle",
                    "revision": "da1909b66b360e1ea8cdeb3e39e40dca172cfa32", "subfolder": "q4"},
}
LIGHTNING_PIN = {"repository": "lightx2v/Wan2.2-Lightning",
                 "revision": "18bccf8884ec0a078eed79785eb4ef13ea16ce1e", "subfolder": LIGHTNING_DIR}
FILES = {
    "mlx": {
        "q4/config.json": (1296, "d404dda725fffd33fd0b5fa0c84f49b764232225"),
        "q4/high_noise_model.safetensors": (8378982847, "f7d1a8a808d9f22eb156bc98ce8a90df78312980722ef023588a377e46a8ec2d"),
        "q4/low_noise_model.safetensors": (8378982717, "0152aeb12c6154dd666702f3b85015118ae10fa60ea0acc969eef0b244eed246"),
        "q4/t5_encoder.safetensors": (11361845591, "2d0aaee4de5ef756c32a7d3e0c1879a3427fa4f53a656bbc7c46198d705e2612"),
        "q4/tokenizer.json": (16837417, "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b"),
        "q4/vae.safetensors": (507591266, "92e17d4b51b712b9e0e5f6af9f4b07cdd28fbdfdf046db670740f6c098583626"),
    },
    "candle-cuda": {
        "q4/.gitattributes": (1860, "75b8d68701da756cee0d6c9fe3bc7b5ed74955a6"),
        "q4/README.md": (18747, "52bfe6e6d881824e9d4db88ac5cffe57dec4b0b9"),
        "q4/assets/comp_effic.png": (202156, "75ee012dcfb08365bec67a3ec7afc126fc2817f79b9f80e38711792d4770e32b"),
        "q4/assets/logo.png": (56322, "0c55854cbd9692975f217714ffd83fd4b37f5dca"),
        "q4/assets/moe_2.png": (527914, "4ea471ccb64349bd08bc9a78f336ae000e9ca3b40da9a652b8028b214a8c6093"),
        "q4/assets/moe_arch.png": (74900, "7822af1e65215ee2a9449c9b7616afd713f67a01"),
        "q4/assets/performance.png": (306535, "97ef99c13c8ae717a8a11c8d8ec927b69077c647cc6689755d08fc38e7fbb830"),
        "q4/assets/vae.png": (165486, "4aaea5e187f1c5908e15ade5bef24c9fb59882986bc3d2ad75f7fe820f3d772f"),
        "q4/model_index.json": (498, "f622781c91d39ea85a7a2c4e9f3e3652e6a467d8"),
        "q4/scheduler/scheduler_config.json": (820, "b5c4eaa781a10db1471bba005afeb90ce0391df5"),
        "q4/text_encoder/config.json": (855, "ab4a73bce055c6e32e66133032dcb3adfb26ee8d"),
        "q4/text_encoder/model-00001-of-00003.safetensors": (4935812536, "a8e861969c7433e707cc5a74065d795d36cca07ec96eb6763eb4083df7248f58"),
        "q4/text_encoder/model-00002-of-00003.safetensors": (4983103192, "d57d948ece4837d850b7a859a4415121d57cacf8b9ee1d4db200c67f592902d7"),
        "q4/text_encoder/model-00003-of-00003.safetensors": (1442935480, "0da9ee284e21d1406df708788db1d502d95d75f69faa25cd26151bf8829b7c5f"),
        "q4/text_encoder/model.safetensors.index.json": (22476, "f3d3d4da90eb33e14c92d88ea346370fa3c0b5b2"),
        "q4/tokenizer/special_tokens_map.json": (7079, "2ed25bf989a28d20b5d4b5822fbc24666d12a6f7"),
        "q4/tokenizer/spiece.model": (4548313, "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"),
        "q4/tokenizer/tokenizer.json": (16837459, "20a46ac256746594ed7e1e3ef733b83fbc5a6f0922aa7480eda961743de080ef"),
        "q4/tokenizer/tokenizer_config.json": (61758, "09d434f9457238f697f4c208aab47f58caa15bfe"),
        "q4/transformer/config.json": (495, "7f44f6167861ec54586f0b9fd7a336de2c4726af"),
        "q4/transformer/model.safetensors": (8948213392, "54aa3aae48726a383dd7b9fa4eb152fe1903d456f6f6505fa97424edf6aa511d"),
        "q4/transformer/quantize_config.json": (56, "3e239833da45bcb4c2ff32bb2c169512bf7a79aa"),
        "q4/transformer_2/config.json": (495, "7f44f6167861ec54586f0b9fd7a336de2c4726af"),
        "q4/transformer_2/model.safetensors": (8948213392, "37f27b82ba3b9293287ea876657993dfc4868cf4ac02a1da6528ce59c62c3200"),
        "q4/transformer_2/quantize_config.json": (56, "3e239833da45bcb4c2ff32bb2c169512bf7a79aa"),
        "q4/vae/config.json": (724, "c190b2929391232ec2f6e493dfe93542927e74ea"),
        "q4/vae/diffusion_pytorch_model.safetensors": (507591892, "d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793"),
    },
    "lightning": {
        f"{LIGHTNING_DIR}/high_noise_model.safetensors": (1226977424, "698321cb86bd30c4af06c9b84e656a1048c8cb54e06d50694536fb5de37fde41"),
        f"{LIGHTNING_DIR}/low_noise_model.safetensors": (1226977424, "ec95216e614b3c132c11bfb387b11feedf62163150ccc9068bca8a189771e75a"),
    },
}
PROBE = Path(__file__).with_name("wan_adapter_tier2_probe.rs.in")
CANDLE_PROBE = Path(__file__).with_name("wan_adapter_tier2_candle.rs.in")


def package(backend):
    return PACKAGE if backend == "mlx" else "candle-gen-wan"


def replace_once(raw, before, after):
    if raw.count(before) != 1:
        raise ValueError("Wan timing overlay no longer matches the actual outer step")
    return raw.replace(before, after)


MEMORY_PHASES = ("te-drained", "high-drained", "high-released", "low-drained", "vae-materialized")
POOL_FIELDS = ("poolUsedBytes", "poolUsedHighBytes", "poolReservedBytes", "poolReservedHighBytes")


def memory_overlay(raw):
    start = b"    fn render_sequential("
    before, rest = raw.split(start, 1)
    stop = b"\n    }\n}"
    body, after = rest.split(stop, 1)
    te = b"            self.device.synchronize()?;\n            (pos, neg)"
    body = replace_once(body, te, b"            self.device.synchronize()?;\n"
                        b'            sga_wan_pool("te-drained")?;\n            (pos, neg)')
    swap = b"        staged_expert_swap("
    body = replace_once(body, swap, b'        let mut sga_swap_phases = ["high-drained", "high-released", "low-drained"].into_iter();\n' + swap)
    body = replace_once(body, b"            || Ok(self.device.synchronize()?),", b'''            || {
                self.device.synchronize()?;
                let phase = sga_swap_phases.next().ok_or_else(|| {
                    CandleError::Msg("unexpected exact Wan expert sync".into())
                })?;
                sga_wan_pool(phase)
            },''')
    vae = b"        let images = frames_to_images(&decoded)?;"
    body = replace_once(body, vae, vae + b'\n        sga_wan_pool("vae-materialized")?;')
    return before + start + body + stop + after + b'''

fn sga_wan_pool(phase: &str) -> CResult<()> {
    let pool = candle_gen::candle_core::cuda_backend::proof_pool::snapshot(0)?
        .ok_or_else(|| CandleError::Msg("Wan diagnostic CUDA pool was not installed".into()))?;
    let at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| CandleError::Msg(error.to_string()))?
        .as_nanos();
    println!("SGA_WAN_MEMORY {phase} {at} {} {} {} {}", pool.used_bytes, pool.used_high_bytes,
             pool.reserved_bytes, pool.reserved_high_bytes);
    Ok(())
}
'''


def overlay(read, _arm, backend, binary, *, phase_memory=False):
    if backend == "candle-cuda":
        prefix = f"crates/media/candle-gen/{package(backend)}/src/"
        path = prefix + "wan14b.rs"
        raw = read(path)
        marker = b"            return Ok((dit, Some(report.applied)));"
        raw = replace_once(raw, marker, b"            println!(\"SGA_WAN_EXPERT_ADAPTERS {:?} {}\", expert, report.applied);\n" + marker)
        marker = b"            let t = sched.timestep(i);"
        raw = replace_once(raw, marker, marker + b"\n            let sga_step_start = std::time::Instant::now();")
        marker = b"            *latents = sched.step(&v, latents)?; // 16-channel latent (out_dim 16)"
        raw = replace_once(raw, marker, marker + b"\n            self.device.synchronize()?;\n"
                           b"            println!(\"SGA_WAN_OUTER_STEP {} {} {}\", i + 1, t.to_bits(), sga_step_start.elapsed().as_nanos());")
        if phase_memory:
            raw = memory_overlay(raw)
        return {path: raw, prefix + f"bin/{binary}.rs": CANDLE_PROBE.read_bytes()}
    if phase_memory:
        raise ValueError("Wan phase memory diagnostic requires the native CUDA fixture")
    prefix = f"crates/media/mlx-gen/{PACKAGE}/src/"
    path = prefix + "pipeline/moe.rs"
    raw = replace_once(read(path), b"        let pred = predict(e.transformer, latents, t, &cache, e.guidance, y)?;",
                       b"        let sga_step_start = std::time::Instant::now();\n"
                       b"        let pred = predict(e.transformer, latents, t, &cache, e.guidance, y)?;")
    raw = replace_once(raw, b"        on_step(i + 1);", b"        println!(\"SGA_WAN_OUTER_STEP {} {} {}\", "
                       b"i + 1, t.to_bits(), sga_step_start.elapsed().as_nanos());\n        on_step(i + 1);")
    return {path: raw, prefix + f"bin/{binary}.rs": PROBE.read_bytes()}


def pins(backend):
    return {"model": WEIGHT_PINS[backend], "lightning": LIGHTNING_PIN}


def fixture_identity(backend):
    return {f"{slot}/{path}": {"bytes": size, "digest": digest,
                               "kind": "lfs_sha256" if len(digest) == 64 else "git_blob_sha1"}
            for slot in pins(backend) for path, (size, digest) in FILES[backend if slot == "model" else slot].items()}


def contract(variant):
    backend = VARIANT_BACKENDS[variant]
    cuda = backend == "candle-cuda"
    size, requested_frames = (320, 17) if cuda else (256, 9)
    device = {"kind": "candle-cuda" if cuda else "mlx-metal", "index": 0}
    allocation = {"pool_cap_bytes": POOL_BYTES, "cublas_workspace": ":4096:8"} if cuda else {}
    return {
        "settings": {"route": "wan2_2_t2v_14b", "tier": "packed-q4", "precision": "BF16",
                     "residency": "sequential expert swap", "device": device, **allocation,
                     "sampler": "uni_pc" if cuda else "unipc", "scheduler_shift": 12.0,
                     "guidance": 1.0, "steps": 4, "boundary_timestep": 875.0, "seed": 42,
                     "width": size, "height": size, "frames": requested_frames, "fps": 16,
                     **({} if cuda else {"native_decoded_frames": MLX_NATIVE_DECODED_FRAMES}),
                     "lightning": ["high@1.0", "low@1.0"],
                     "prompt": "A red fox trots across a snowy meadow at sunrise, filmed at ground level.",
                     "negative_prompt": "native provider default"},
        "sources": pins(backend), "expected_asset_identity": fixture_identity(backend),
        "comparison": {"pairs": perf.PAIRS, "warmup": ["base", "head"],
                       "cache": "fresh child and allocator per render, shared OS cache unflushed",
                       "asset_stage_timeout_seconds": STAGE_TIMEOUT_SECONDS,
                       "asset_recheck_timeout_seconds": RECHECK_TIMEOUT_SECONDS,
                       "timing": ["provider load through decoded RGB", "mean synchronized outer step"],
                       "child_deadline_seconds": CHILD_SECONDS[backend], "ceiling": perf.CEILING,
                       "interval_coverage": perf.INTERVAL_COVERAGE,
                       "parity": "exact timesteps, expert boundary, adapter counts and decoded RGB8 digests",
                       "adapter_evidence": ("per-expert install counts from the disposable proof overlay; provider "
                                            "does not publish per-file adapter reports" if cuda else
                                            "per-file provider apply and skipped-target reports")},
    }


def policy(variant, binding):
    from scripts.ci.sga_adapter_comparison import POLICY

    backend = VARIANT_BACKENDS[variant]
    common = dict(POLICY, deadlineSeconds=CHILD_SECONDS[backend],
                  hostFreeReserveBytes=8 * GIB, childFootprintCapBytes=56 * GIB)
    if backend == "candle-cuda":
        return dict(common, backend="windows-cuda", cudaDeviceUuid=binding,
                    gpuFreeReserveBytes=2 * GIB, childGpuCapBytes=28 * GIB)
    return common


def asset_environment(variant, cache):
    pin = LIGHTNING_PIN
    snapshot = hub.snapshot_dir(cache, pin["repository"], pin["revision"])
    if VARIANT_BACKENDS[variant] == "mlx":
        return {"PAIR_WAN_LIGHTNING": str(snapshot)}
    return {f"PAIR_WAN_LIGHTNING_{name.upper()}": str(snapshot / LIGHTNING_DIR / f"{name}_noise_model.safetensors")
            for name in ("high", "low")}


def manifest(backend, repository, revision):
    slots = pins(backend)
    slot = next(slot for slot, pin in slots.items() if (pin["repository"], pin["revision"]) == (repository, revision))
    url, raw, inventory = pinned_manifest(repository, revision)
    expected = FILES[backend if slot == "model" else slot]
    if slot == "model" and set(hub.pinned_files(inventory, "q4")) != set(expected):
        raise ValueError("Wan model tier file inventory differs from the frozen production snapshot")
    filtered = {}
    for path, (size, digest) in expected.items():
        row = inventory[path]
        observed = (row.get("lfs") or {}).get("sha256") if len(digest) == 64 else row.get("blobId")
        if observed != digest or row.get("size") != size:
            raise ValueError(f"Wan Hub manifest differs from the frozen proof asset: {path}")
        filtered[path] = row
    return url, raw, filtered


def assets(mode, backend, cache):
    groups = pins(backend)
    if mode == "stage":
        rows = hub.stage(groups, cache, manifest=lambda repo, rev: manifest(backend, repo, rev))
    else:
        rows = {slot: verify_pinned_hub_files(
            pin["repository"], pin["revision"], hub.snapshot_dir(cache, pin["repository"], pin["revision"]),
            tuple(FILES[backend if slot == "model" else slot])) for slot, pin in groups.items()}
    items = {f"{slot}/{row['path']}": {key: row[key] for key in ("bytes", "digest", "kind")}
             for slot in groups for row in rows[slot]["files"]}
    if items != fixture_identity(backend):
        raise ValueError("Wan bound asset bytes differ from the frozen proof assets")
    return items


def staged_assets(variant, cache, *, stage=True):
    backend = VARIANT_BACKENDS[variant]
    pin = WEIGHT_PINS[backend]
    snapshot = hub.snapshot_dir(cache, pin["repository"], pin["revision"]) / pin["subfolder"]
    completed = subprocess.run([sys.executable, "-I", str(Path(__file__)), "stage" if stage else "identify",
                                backend, str(cache)], timeout=STAGE_TIMEOUT_SECONDS if stage else RECHECK_TIMEOUT_SECONDS,
                               encoding="utf-8", stdout=subprocess.PIPE, check=True)
    return snapshot, json.loads(completed.stdout, object_pairs_hook=unique_object)


def preflight(stdout, variant):
    if stdout.splitlines() != [f"SGA_WAN_PREFLIGHT {CASE[variant]}"]:
        raise ValueError("Wan preflight did not authenticate the native packed route and adapter pair")


def observe_memory(stdout, pool):
    lines = [line for line in stdout.splitlines() if line.startswith("SGA_WAN_MEMORY")]
    result, previous = {}, (0, 0, 0)
    if len(lines) != len(MEMORY_PHASES):
        raise ValueError("Wan diagnostic requires all five native memory boundaries")
    for phase, line in zip(MEMORY_PHASES, lines, strict=True):
        match = re.fullmatch(rf"SGA_WAN_MEMORY {phase} ([1-9][0-9]{{17,19}})((?: [1-9][0-9]{{0,11}}){{4}})", line)
        if match is None:
            raise ValueError("Wan diagnostic native memory boundary is missing, malformed or unordered")
        at_ns = int(match[1])
        used, used_high, reserved, reserved_high = map(int, match[2].split())
        if (not previous[0] <= at_ns or not used <= min(reserved, used_high)
                or not max(reserved, used_high) <= reserved_high <= min(POOL_BYTES, pool["poolReservedHighBytes"])
                or not previous[1] <= used_high <= pool["poolUsedHighBytes"] or not previous[2] <= reserved_high):
            raise ValueError("Wan diagnostic pool readings differ from the bounded child high-water counters")
        result[phase] = {"at_unix_ns": at_ns, **dict(zip(POOL_FIELDS, (used, used_high, reserved, reserved_high), strict=True))}
        previous = (at_ns, used_high, reserved_high)
    return result


def observe_windows_host(guard):
    row = guard.get("windows_host_peaks")
    kinds = ("working_set", "private_usage", "job_peak")
    samples = [sample for sample in guard["samples"] if sample.get("sample_kind") == "process"]
    if (not isinstance(row, dict) or set(row) != {"pid", "sample_count", *kinds}
            or row["pid"] != guard["pid"] or type(row["sample_count"]) is not int
            or not 0 < row["sample_count"] == len(samples)
            or any(not isinstance(item := row[kind], dict) or set(item) != {"bytes", "at_unix_ns"}
                   or type(item["bytes"]) is not int or item["bytes"] <= 0
                   or type(item["at_unix_ns"]) is not int or item["at_unix_ns"] <= 0 for kind in kinds)
            or max(row[kind]["bytes"] for kind in kinds) != guard["peak_host_bytes"]):
        raise ValueError("Wan diagnostic requires exact single-child Windows host footprint counters")
    return row


def observe(stdout, path, binding, variant):
    backend = VARIANT_BACKENDS[variant]
    cuda = backend == "candle-cuda"
    lines = stdout.splitlines()
    row = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    steps = [line for line in lines if line.startswith("SGA_WAN_OUTER_STEP")]
    if lines.count("SGA_WAN_REAL_COMPLETE " + CASE[variant]) != 1 or len(steps) != 4:
        raise ValueError("Wan child did not complete its four real outer steps")
    observed = []
    for index, line in enumerate(steps, 1):
        match = re.fullmatch(r"SGA_WAN_OUTER_STEP ([1-4]) ([0-9]{1,20}) ([1-9][0-9]{0,11})", line)
        if match is None or int(match[1]) != index or not 0 < int(match[2]) < 2**(64 if cuda else 32):
            raise ValueError("Wan outer-step observation is invalid")
        bits, nanos = int(match[2]), int(match[3])
        timestep, = struct.unpack("<d" if cuda else "<f", struct.pack("<Q" if cuda else "<I", bits))
        if not 0 < timestep <= 1000:
            raise ValueError("Wan effective timestep is out of bounds")
        observed.append((bits, nanos, "high" if timestep >= 875 else "low"))
    timesteps = [bits for bits, _, _ in observed]
    experts = [expert for _, _, expert in observed]
    progress = []
    previous = None
    for index, expert in enumerate(experts, 1):
        if expert != previous:
            progress.append("renderer")
            previous = expert
        progress.append(f"{index}/4")
    if cuda:
        progress.append("renderer")  # The Candle route announces loading its staged VAE.
    progress.append("decode")
    digests = row["frame_rgb8_sha256"]
    device = {"kind": "candle-cuda" if cuda else "mlx-metal", "index": 0}
    if (row["device"] != device or set(experts) != {"high", "low"}
            or timesteps != sorted(timesteps, reverse=True) or row["progress"] != progress
            or not isinstance(digests, list) or len(digests) != (17 if cuda else MLX_NATIVE_DECODED_FRAMES)
            or any(not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in digests)
            or len(set(digests)) < 2):
        raise ValueError("Wan output does not prove both expert branches and the frozen decoded clip")
    peaks = {}
    if cuda:
        reports = [line for line in lines if line.startswith("SGA_WAN_EXPERT_ADAPTERS")]
        counts = [re.fullmatch(r"SGA_WAN_EXPERT_ADAPTERS (High|Low) ([1-9][0-9]{0,5})", line) for line in reports]
        if (row.get("cuda_device_uuid") != binding or len(counts) != 2 or not all(counts)
                or [count[1] for count in counts] != ["High", "Low"]):
            raise ValueError("Candle did not install positive adapter targets for both expert stages on the selected device")
        split = experts.index("low")
        stages = [line for line in lines if line.startswith(("SGA_WAN_EXPERT_ADAPTERS", "SGA_WAN_OUTER_STEP"))]
        if stages != [reports[0], *steps[:split], reports[1], *steps[split:]]:
            raise ValueError("Candle adapter installs did not precede their expert's outer steps")
        adapters = [{"expert": count[1], "applied": int(count[2])} for count in counts]
        device = {**device, "uuid": binding}
    else:
        adapters = row["adapters"]
        if len(adapters) != 2:
            raise ValueError("MLX did not return both real adapter reports")
        for adapter, name in zip(adapters, ("High", "Low"), strict=True):
            if (adapter["file"] != name.lower() + "_noise_model.safetensors"
                    or adapter["expert"] != f"Some({name})" or adapter["scale"] != 1.0
                    or type(adapter["applied"]) is not int or adapter["applied"] <= 0
                    or not isinstance(adapter["skipped"], list)):
                raise ValueError("MLX adapter did not install on its authoritative expert")
        if type(row["mlx_peak_bytes"]) is not int or row["mlx_peak_bytes"] <= 0:
            raise ValueError("Wan MLX allocator did not report positive peak bytes")
        peaks["mlx_allocator_bytes"] = row["mlx_peak_bytes"]
    total = row["end_to_end_ns"]
    durations = [nanos for _, nanos, _ in observed]
    if type(total) is not int or not sum(durations) <= total <= CHILD_SECONDS[backend] * 10**9:
        raise ValueError("Wan request did not retain positive bounded render timing")
    return {"identity": {"device": device, "adapters": adapters, "progress": progress,
                         "frame_rgb8_sha256": digests, "steps": [[bits, expert] for bits, _, expert in observed]},
            "timing": {"e2e_ns": total, "outer_step_ns": fmean(durations)}, "peaks": peaks}


if __name__ == "__main__":
    # Keep progress on the workflow console and reserve stdout for the JSON receipt.
    mode, backend, cache = sys.argv[1:]
    if mode not in ("stage", "identify"):
        raise ValueError("invalid asset operation")
    from contextlib import redirect_stdout
    with redirect_stdout(sys.stderr):
        receipt = assets(mode, backend, Path(cache))
    print(json.dumps(receipt, sort_keys=True))
