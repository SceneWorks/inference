"""Immutable SCAIL-2 plural fit campaign and accepted native measurements."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from scripts.ci import video_tier1_evidence as evidence, wddm_process_memory as wddm
from scripts.media_campaign_supervisor import validate_admission

GIB = 1024**3
MIB = 1024**2
SELECTION = "scail2-plural-fit"
PROTOCOL = SELECTION + "-v1"
REV = "ce88cfdb1008f395e9c820e525e6db7b6695f7b3"
REPO = "SceneWorks/scail2-mlx"
INVENTORY = "cbe28c27f7a594cb3da8ec54d7f293963c36188a2d371a12be6c3e7fea42f59f"
GPU = "GPU-e4b79931-7be6-f216-460a-f5405cfafffe"
SLOT = "SCAIL_PLURAL_Q4"
FILES = {
    "clip.safetensors": (
        2528346712,
        "90ddb7ff2d56e640b8ef991f498ec1f597fd86794e5c4790eb91622c5f551a5d",
    ),
    "config.json": (332, "7ccd156e99a51a80f93143acfc53a1f1edd0ca0f0f095cbdf35cc60345729061"),
    "dit.safetensors": (
        9578472129,
        "1a46f805cf22dc74efeab0d4eae6a21ee48f4234281c4d09cf0755378661f614",
    ),
    "t5_encoder.safetensors": (
        11361845504,
        "70e44fce2fef5e7bf70a519f65a6dfa3a16a000f489c8c492c913eea77167f0d",
    ),
    "tokenizer.json": (
        16837417,
        "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b",
    ),
    "vae.safetensors": (
        507591212,
        "42159a8b571dbeb3ea40327b88a6161a5342c0511202af7c031360629757163d",
    ),
}
COUNTS = {"mlx": [6, 2], "candle-cuda": [6, 2, 3, 4, 5]}
FQN = "plural_fit::measure_real_weights"
CARRIERS = json.loads(Path(__file__).with_name("carriers.json").read_text())


def policy(backend):
    settings = dict(
        schemaVersion=1,
        deadlineSeconds=2400,
        pollMillis=250,
        termGraceMillis=2000,
        stdoutCapBytes=4 * MIB,
        stderrCapBytes=4 * MIB,
        eventCapBytes=16 * MIB,
    )
    if backend == "mlx":
        return dict(
            settings,
            backend="darwin-mlx",
            childFootprintCapBytes=96 * GIB,
            hostFreeReserveBytes=12 * GIB,
        )
    if backend == "candle-cuda":
        return dict(
            settings,
            backend="windows-cuda",
            childFootprintCapBytes=64 * GIB,
            hostFreeReserveBytes=8 * GIB,
            cudaDeviceUuid=GPU,
            childGpuCapBytes=84 * GIB,
            gpuFreeReserveBytes=6 * GIB,
        )
    raise ValueError("unregistered plural SCAIL backend")


def validate_stage(stage, root):
    if (
        (stage["repository"], stage["revision"], stage["subfolder"], stage["status"])
        != (REPO, REV, "q4", "hub_content_verified")
        or Path(stage["path"]).resolve() != root.resolve()
        or [f["path"] for f in stage["files"]] != ["q4/" + n for n in FILES]
    ):
        raise ValueError("staging did not verify precisely the canonical q4 tier")
    for item, (name, (size, digest)) in zip(stage["files"], FILES.items()):
        kind, wanted = (
            ("git_blob_sha1", "2f1647f3edc2ec8d1c9660f3910a91f45d06a675")
            if name == "config.json"
            else ("lfs_sha256", digest)
        )
        if (item["bytes"], item["kind"], item["digest"]) != (size, kind, wanted):
            raise ValueError("staged file identity differs from the canonical tier")


def validate_native(data, backend, count):
    output = data["output"]
    if (
        data["schema"] != 1
        or data["backend"] != backend
        or data["count"] != count
        or data["files"] != [{"name": n, "sha256": d} for n, (_, d) in FILES.items()]
        or data["prompt_tokens"] != [512, 512]
        or data["steps"] != [[n, 4] for n in range(1, 5)]
        or (output["width"], output["height"], output["fps"], output["frames"])
        != (832, 480, 16, 48 if backend == "mlx" else 45)
        or not re.fullmatch(r"[0-9a-f]{64}", output["sha256"])
        or not isinstance(data["axes"], dict)
        or not data["axes"]
        or len(data["phases"]) != 1
        or data["phases"][0]["references"] != count
        or data["source_bytes"] != (2 * count + 90) * 832 * 480 * 3
    ):
        raise ValueError("native plural render did not match its pinned inputs and operating point")
    axes = data["axes"]
    digest = hashlib.sha256(b"scail2-plural-weights-v1")
    for name in [
        "config.json",
        "tokenizer.json",
        "dit.safetensors",
        "t5_encoder.safetensors",
        "clip.safetensors",
        "vae.safetensors",
    ]:
        digest.update(bytes.fromhex(FILES[name][1]))
    hashes = data["character_hashes"]
    if (
        axes.get("artifact_sha256") != digest.hexdigest()
        or axes.get("artifact_tier") != "q4"
        or axes.get("effective_tier") != "q4"
        or (
            axes.get("backend"),
            axes.get("width"),
            axes.get("height"),
            axes.get("references"),
            axes.get("history_order"),
            axes.get("classifier_free_guidance"),
            axes.get("source_bytes"),
        )
        != ("mlx" if backend == "mlx" else "cuda", 832, 480, count, 4, True, data["source_bytes"])
        or hashes != CARRIERS["ordered_pairs"][:count]
        or data["driving_hashes"]
        != [CARRIERS["driving_rgb"], CARRIERS["driving_masks"][str(count)]]
        or any(not re.fullmatch(r"[a-f0-9]{64}", h) for pair in hashes for h in pair)
        or len({h for pair in hashes for h in pair}) != 2 * count
        or not 0 < data["render_ns"] <= data["process_work_ns"]
    ):
        raise ValueError("native axes, weights or ordered reference/mask buffers differ")
    if backend == "mlx" and (
        len(axes.get("windows", [])) != 1 or axes["windows"][0].get("mlx_unsplit_ffn") is not True
    ):
        raise ValueError("MLX packed Q4 FFN execution class is unproven")
    identity = data["execution"]
    expected = "mlx" if backend == "mlx" else "candle"
    if (
        identity["provider_id"] != "scail2_14b"
        or identity["backend"] != expected
        or "unipc" not in json.dumps(identity["sampling"]).lower()
        or ("scail2_decode" if backend == "mlx" else "z16_causal_streaming_no_spatial_tiles")
        not in json.dumps(identity["quality_policy"])
    ):
        raise ValueError("native execution identity lacks the consumed solver or codec policy")


def measured_row(data, guard, pool, adapter, hardware, backend, count, approved, revision):
    validate_native(data, backend, count)
    validate_admission(guard["admission"], policy_sha256=approved.sha256)
    if (
        guard["pid"] != data["pid"]
        or guard["returncode"] != 0
        or not 0 < guard["elapsed_seconds"] <= 2400
        or guard["fit_observation"]["host_baseline"] is None
        or not 0
        < guard["fit_observation"]["host_baseline"]
        <= guard["peak_host_bytes"]
        <= approved.child_footprint_cap_bytes
        or not 0 <= data["allocator_baseline"] <= data["allocator_peak"]
        or not data["allocator_peak"]
    ):
        raise ValueError("missing or unbounded process or allocator measurement")
    samples = guard["fit_observation"]["host_samples_monotonic_ns_bytes"]
    if not samples or samples[0][1] != guard["fit_observation"]["host_baseline"]:
        raise ValueError("missing owned host baseline or trace")
    host = guard["peak_host_bytes"]
    if backend == "mlx":
        if (
            not guard["lifetime_peak_host_bytes"]
            or data["allocator_peak"] > approved.child_footprint_cap_bytes
        ):
            raise ValueError("missing Darwin kernel lifetime footprint or allocator over cap")
        machine = {"platform": "macos", "device": hardware["chip"], "runtime": hardware["os_build"]}
        gross = (
            0,
            0,
            max(
                host,
                data["allocator_peak"] + data["source_bytes"],
                data["axes"]["floor_unified_bytes"],
            ),
        )
    else:
        if (
            pool["pid"] != data["pid"]
            or pool["cudaDeviceUuid"] != GPU
            or not 0
            < pool["poolUsedHighBytes"]
            <= pool["poolReservedHighBytes"]
            == data["allocator_peak"]
        ):
            raise ValueError("CUDA native pool differs from the owned process")
        cuda = evidence.cuda_process(guard, data["allocator_peak"], adapter)
        shared = guard["fit_observation"]["wddm_shared"]
        shared_peak = wddm.sampled_peak(shared, pid=data["pid"], adapter=adapter, allow_zero=True)
        dedicated = cuda["dedicated"]["sampled_peak_bytes"]
        if (
            dedicated is None
            or shared_peak is None
            or shared["created_filetime"]
            != guard["cuda_attribution"]["wddm_dedicated"]["created_filetime"]
            or hardware["uuid"] != GPU
            or int(hardware["index"]) != 1
        ):
            raise ValueError("missing owned WDDM dedicated or shared observations")
        raw = guard["fit_observation"]
        counters = raw["windows_peak"]
        if (
            counters["owned_process_count"] != 1
            or min(
                counters["peak_private_commit_bytes"],
                counters["peak_job_commit_bytes"],
                counters["peak_working_set_bytes"],
            )
            <= 0
            or not raw["card_samples_monotonic_ns_free_owned_or_fallback_bytes"]
            or raw["wddm_shared_counter"] != "GPU Process Memory/Shared Usage"
        ):
            raise ValueError("missing lifetime Windows or card observations")
        host = max(
            host,
            counters["peak_job_commit_bytes"],
            counters["peak_private_commit_bytes"],
            counters["peak_working_set_bytes"],
        )
        device = max(
            cuda["upper"], dedicated, data["allocator_peak"], data["axes"]["floor_device_bytes"]
        )
        if device > approved.child_gpu_cap_bytes:
            raise ValueError("observed CUDA peak exceeds the guard")
        machine = {
            "platform": "windows",
            "device": hardware["name"],
            "runtime": hardware["driver_version"],
        }
        # The guard's Windows Job lifetime peak covers private commit. WDDM shared memory may
        # overlap it; charge both for the conservative host envelope and retain each raw counter.
        gross = (max(host + shared_peak, data["axes"]["floor_host_bytes"]), device, 0)
    if max(gross[0], gross[2]) > approved.child_footprint_cap_bytes:
        raise ValueError("measured host/unified requirement exceeds approved cap")
    return dict(
        machine,
        inference_revision=revision,
        witness=dict(
            axes=data["axes"],
            text_tokens=min(data["prompt_tokens"]),
            gross_host_bytes=gross[0],
            gross_device_bytes=gross[1],
            gross_unified_bytes=gross[2],
            mlx_decode_budget_bytes=28 * GIB if backend == "mlx" else None,
        ),
    )


def digest_json(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
