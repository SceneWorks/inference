"""Stage and attest immutable inputs for the dedicated SenseNova Loop proof."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ci.model_inference_proofs import repository_identity
from scripts.ci.model_proof_resources import (REGISTERED_CUDA_CARDS, choose_card, claim_chosen_card,
                                              executed_card_policy, execution_claim, require_claimed_card,
                                              shared_owned_completion)
from scripts.ci.sensenova_loop_reference import validate_cuda_witness
from scripts.media_campaign_supervisor import (
    SupervisionError, canonical, load_policy, run_guarded, validate_admission,
)
from scripts.release.ensure_model_snapshot import download_snapshot, ensure_snapshot, runner_hub_cache
from scripts.release.verify_model_snapshot import load_model, verify_snapshot

MANIFEST = ROOT / "release/real-weight-models.toml"
SOURCE = ROOT / "scripts/ci/sensenova-loop-source.json"
MODEL_KEY = "sensenova-loop-9b"
GIB = 1024 ** 3


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def inventory(files: list[Path], root: Path) -> list[dict]:
    return [{"path": path.relative_to(root).as_posix(), "size": path.stat().st_size,
             "sha256": digest(path)} for path in sorted(set(files))]


def complete_cache_file(path: Path, size: int) -> bool:
    try:
        info = path.stat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_size == size


def check_source_disk(snapshot: Path, source: dict, receipt: Path) -> None:
    """Preflight the Hub reconstruction destination without reading model file contents."""
    blobs = snapshot.parent.parent / "blobs"
    disk_path = next(path for path in (blobs, *blobs.parents) if path.is_dir())
    free_bytes = shutil.disk_usage(disk_path).free
    # A finalized LFS blob may exist before its snapshot symlink is installed.
    # Partial files and Xet chunks do not prove that the payload can be reconstructed
    # in place. Charge their full immutable file size to the remaining download.
    missing = [item for item in source["files"]
               if not complete_cache_file(snapshot / item["path"], item["size"])
               and not complete_cache_file(blobs / item["sha256"], item["size"])]
    missing_bytes = sum(item["size"] for item in missing)
    write(receipt, {
        "repository": source["repository"], "revision": source["revision"],
        "snapshot": str(snapshot), "free_space_measured_at": str(disk_path),
        "free_bytes": free_bytes, "missing_payload_bytes": missing_bytes,
        "missing_files": [{"path": item["path"], "size": item["size"]} for item in missing],
        "payload_fits_sampled_free_space": free_bytes >= missing_bytes,
    })
    if free_bytes < missing_bytes:
        raise RuntimeError(
            f"insufficient space for pinned SenseNova source: {disk_path} has {free_bytes} bytes free "
            f"for {missing_bytes} bytes across {len(missing)} missing payloads"
        )


def source_snapshot(*, disk_receipt: Path | None = None, download=download_snapshot):
    source = json.loads(SOURCE.read_bytes())
    model = load_model(MANIFEST, MODEL_KEY)
    expected_files = [entry["path"] for entry in source["files"]]
    if (source["repository"], source["revision"], expected_files) != (
        model["repository"], model["revision"], model["expected_files"]
    ):
        raise ValueError("fixture manifest disagrees with pinned source bytes")
    # The runner's own hub cache (HF_HUB_CACHE, else $HF_HOME/hub, else the fleet root), never the
    # library's `~/.cache/huggingface/hub` fallback; in Actions a fetch may land nowhere else.
    hub = runner_hub_cache().resolve()
    snapshot = hub / "models--sensenova--SenseNova-Loop-9B" / "snapshots" / source["revision"]
    if disk_receipt is not None:
        check_source_disk(snapshot, source, disk_receipt)
    ensure_snapshot(model, snapshot, download,
                    download_root=hub if os.environ.get("GITHUB_ACTIONS") == "true" else None)
    verify_snapshot(model, snapshot)
    observed = inventory([snapshot / name for name in expected_files], snapshot)
    if observed != source["files"]:
        differing = [want["path"] for want, got in zip(source["files"], observed, strict=True) if want != got]
        raise ValueError(f"gated source does not match the release bytes: {differing}")
    return snapshot, source, digest(SOURCE)


def offline_environment() -> dict[str, str]:
    # Isolate reviewed upstream code from runner auth and the persistent Hub/module cache.
    env = {key: value for key, value in os.environ.items()
           if key.upper() not in {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN",
                                  "GH_TOKEN", "GITHUB_TOKEN"}}
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    return env


def registered_index(uuid):
    """The physical index the claim binds for a registered card; any other card is refused."""
    if uuid not in REGISTERED_CUDA_CARDS:
        raise ValueError(f"SenseNova CUDA policy names unregistered card {uuid}")
    return REGISTERED_CUDA_CARDS[uuid]


def guarded(argv, backend, policy, work, env, *, label, request_payload=None, selection=None, record=None,
            deadline_monotonic=None):
    """Run one stage under its claim. A card-free request (`selection`) whose card is taken before
    the claim switches to the other registered card when that fits now; `record` then holds the
    new `cuda_card_selection`. `argv` may be a function of the policy of the card in force."""
    cuda = backend == "candle-cuda"
    bound = {"policy": policy}

    def bind(chosen):
        path = work / f"{label}-policy.json"
        write(path, chosen)
        bound["policy"] = load_policy(path)
        if cuda:
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = bound["policy"].cuda_device_uuid

    if cuda:
        registered_index(policy.cuda_device_uuid)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = policy.cuda_device_uuid
    payload = json.loads(policy.canonical_bytes)
    primary = None
    release_error = None
    claim = {}
    try:
        # The claim binds the registered physical index of the card the policy names or selected.
        with claim_chosen_card(backend, request_payload or payload, payload, selection if cuda else None, work, env,
                               claim_policy=lambda chosen: bound["policy"], rebind=bind, record=record,
                               constraints=(lambda chosen: {"required_cuda_physical_index": registered_index(chosen["cudaDeviceUuid"])})
                               if cuda else {}, claim_with=execution_claim, require_owned_completion=cuda,
                               **({"original_deadline_monotonic": deadline_monotonic} if deadline_monotonic else {})) as claim:
            policy = bound["policy"]
            kwargs = {}
            if cuda:
                kwargs = {"retain_owned_completion": True,
                          "claim_deadline_monotonic": claim["claim_deadline_monotonic"]}
            if "SCENEWORKS_METAL_EXECUTION_LOCK_FD" in env:
                kwargs["pass_fds"] = (int(env["SCENEWORKS_METAL_EXECUTION_LOCK_FD"]),)
            try:
                # On WDDM an owned process can run genuine CUDA work without attributable
                # samples. The existing device free/reserve watchdog remains active. The
                # reference additionally requires its own CUDA allocator/identity receipt.
                result = run_guarded(
                    argv(policy) if callable(argv) else argv, cwd=ROOT, env=env, policy=policy, stdout_path=work / f"{label}.stdout.log",
                    stderr_path=work / f"{label}.stderr.log", event_path=work / f"{label}.events.jsonl",
                    require_gpu_sample=False, **kwargs,
                )
            except SupervisionError as error:
                primary = error
                completion = getattr(error, "owned_completion", None)
                if cuda and isinstance(completion, dict):
                    claim["owned_execution_completion"] = completion
            else:
                if cuda:
                    claim["owned_execution_completion"] = result.owned_completion
                if result.returncode != 0:
                    raise RuntimeError(f"{label} exited {result.returncode}; see bounded stage logs")
    except Exception as error:
        if primary is None:
            raise
        release_error = error
    if primary is not None:
        if release_error is not None:
            primary.add_note(f"Physical release also failed ({type(release_error).__name__}): {release_error}")
        try:
            write(work / f"{label}.failure.json", {
                "supervision_reason": primary.reason,
                "owned_completion": getattr(primary, "owned_completion", None),
                "release_failure": type(release_error).__name__ if release_error is not None else None,
                "lease_released": claim.get("lease_released"),
                "post_execution_verified": claim.get("post_execution_verified"),
                "watchdog_device_memory": primary.watchdog_device_memory,
                "watchdog_child_footprint": primary.watchdog_child_footprint,
            })
        except OSError as error:
            primary.add_note(f"Could not retain supervision diagnostics: {type(error).__name__}")
        raise primary
    return {"process": asdict(result), "device_claim": claim}


def reference_measurements(payload, policy_data, supervision):
    witness = payload["cuda_witness"]
    expected_uuid = policy_data["cudaDeviceUuid"]
    cap = policy_data["childGpuCapBytes"]
    validate_cuda_witness(witness, expected_uuid, cap)
    claim, process = supervision["device_claim"], supervision["process"]
    if (claim["device_uuid"] != expected_uuid or claim["physical_index"] != REGISTERED_CUDA_CARDS.get(expected_uuid)
            or claim["lease_released"] is not True or claim["post_execution_verified"] is not True
            or not shared_owned_completion(claim)
            or claim["owned_execution_completion"] != process["owned_completion"]
            or process["owned_completion"].get("root_returncode") != 0
            or process["owned_completion"].get("root_pid") != process["pid"]
            or process["returncode"] != 0):
        raise ValueError("reference lacks a successful exact registered-GPU claim and owned Windows Job exit")
    policy_sha = hashlib.sha256(canonical(policy_data)).hexdigest()
    admission = validate_admission(process["admission"], policy_sha256=policy_sha)
    reserve = policy_data["gpuFreeReserveBytes"]
    if (admission["backend"], admission["cudaDeviceUuid"], admission["childGpuCapBytes"],
            admission["gpuFreeReserveBytes"]) != ("windows-cuda", expected_uuid, cap, reserve):
        raise ValueError("external reference admission differs from the leased CUDA cap and free reserve")
    sampled_peak = process["peak_gpu_bytes"]
    if sampled_peak is not None and (type(sampled_peak) is not int or not 0 < sampled_peak <= cap):
        raise ValueError("external CUDA sampled peak is invalid")
    global_samples = [sample for sample in process["samples"]
                      if sample.get("phase") == "cuda-global-used-upper-bound-sample"]
    if any(sample["cudaDeviceUuid"] != expected_uuid or sample["gpuFreeBytes"] < reserve
           or not 0 <= sample["childGlobalUsedGrowthBytes"] <= cap for sample in global_samples):
        raise ValueError("WDDM global samples violate the leased CUDA cap or free reserve")
    return {
        "pytorch": witness, "admitted_child_gpu_cap_bytes": cap,
        "external_supervisor_peak_bytes": sampled_peak,
        "external_supervisor_sample": "unknown" if sampled_peak is None else "owned-process-attributed",
        "external_free_reserve_bytes": reserve,
        "wddm_global_sample_count": len(global_samples),
        "wddm_global_peak_used_bytes": max((sample["globalUsedUpperBoundBytes"] for sample in global_samples), default=None),
        "wddm_global_peak_growth_bytes": max((sample["childGlobalUsedGrowthBytes"] for sample in global_samples), default=None),
    }


def reference(args, source, source_sha, snapshot, work, report, request):
    # A reference policy that names no card runs on the registered card with the most free memory that
    # fits; selection and claim share one wait budget.
    deadline = time.monotonic() + request["sensenova_reference_policy"]["deadlineSeconds"]
    policy_data, selection = choose_card(request["sensenova_reference_policy"], work, deadline_monotonic=deadline)
    path = work / "reference-policy.json"
    write(path, policy_data)
    policy = load_policy(path)
    env = offline_environment()
    for key in ("HF_TOKEN_PATH", "HF_HOME", "HF_MODULES_CACHE", "HF_HUB_CACHE"):
        env.pop(key, None)
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_NO_CUDA_MEMORY_CACHING"):
        env.pop(key, None)
    env["PYTORCH_ALLOC_CONF"] = "backend:native"
    env["HF_HOME"] = str(work / "reference-hf")
    env["HF_MODULES_CACHE"] = str(work / "reference-modules")
    oracle = report / "oracle.json"
    reference_python = args.reference_python.resolve(strict=True)
    record = {} if selection is None else {"cuda_card_selection": selection}
    receipt = guarded(
        lambda held: [str(reference_python), "-I", str(ROOT / "scripts/ci/sensenova_loop_reference.py"),
                      str(snapshot), source_sha, str(oracle), held.cuda_device_uuid, str(held.child_gpu_cap_bytes)],
        "candle-cuda", policy, work, env, label="reference",
        request_payload=request["sensenova_reference_policy"], selection=selection, record=record,
        deadline_monotonic=deadline if selection is not None else None,
    )
    selection = record.get("cuda_card_selection")
    policy_data = executed_card_policy(request["sensenova_reference_policy"], selection)
    measurements = reference_measurements(json.loads(oracle.read_bytes()), policy_data, receipt)
    write(report / "reference.json", {
        "source_sha": args.expected_sha, "source_manifest_sha256": source_sha,
        "revision": source["revision"], "oracle_sha256": digest(oracle),
        "policy": request["sensenova_reference_policy"], "supervision": receipt, "cuda_measurements": measurements,
        **({"cuda_card_selection": selection} if selection is not None else {}),
    })


def native(args, source, source_sha, snapshot, work, report, request):
    oracle = args.oracle_dir / "oracle.json"
    reference_receipt = json.loads((args.oracle_dir / "reference.json").read_bytes())
    if (reference_receipt["source_sha"], reference_receipt["source_manifest_sha256"],
            reference_receipt["revision"], reference_receipt["oracle_sha256"]) != (
            args.expected_sha, source_sha, source["revision"], digest(oracle)):
        raise ValueError("oracle receipt does not bind exact engine source, HF bytes and oracle payload")
    if reference_receipt["policy"] != request["sensenova_reference_policy"]:
        raise ValueError("oracle used a different policy than the dispatch")
    payload = json.loads(oracle.read_bytes())
    if (payload["schema_version"], payload["hf_revision"], payload["source_inventory_sha256"]) != (
        1, source["revision"], source_sha
    ):
        raise ValueError("oracle payload identifies different model bytes")
    executed = executed_card_policy(reference_receipt["policy"], reference_receipt.get("cuda_card_selection"))
    if reference_measurements(payload, executed, reference_receipt["supervision"]) != reference_receipt["cuda_measurements"]:
        raise ValueError("reference CUDA measurements differ from the exact oracle and physical lease")
    if shutil.disk_usage(work).free < 36 * GIB:
        raise RuntimeError("need at least 36 GiB of free temporary space for native q8/q4 and build inputs")
    backend = args.backend
    package = "mlx-llm" if backend == "mlx" else "candle-llm"
    target = work / "target"
    env = offline_environment()
    env["CARGO_TARGET_DIR"] = str(target)
    subprocess.run(["cargo", "build", "--locked", "--release", "-p", package, "--example", "prepare_snapshot"],
                   cwd=ROOT, env=env, check=True, timeout=2700)
    exe = target / "release/examples" / ("prepare_snapshot.exe" if os.name == "nt" else "prepare_snapshot")
    native_deadline = time.monotonic() + request["policy"]["deadlineSeconds"]
    native_policy, native_selection = choose_card(request["policy"], work, deadline_monotonic=native_deadline)
    policy_path = work / "native-policy.json"
    write(policy_path, native_policy)
    policy = load_policy(policy_path)
    tiers = {}
    native_record = {} if native_selection is None else {"cuda_card_selection": native_selection}
    for tier in ("q8", "q4"):
        directory = work / tier
        # Only the first tier may switch a taken card; the second runs on the card the first used.
        supervision = guarded([str(exe), str(snapshot), str(directory), tier], backend, policy, work, env,
                              label=tier, request_payload=request["policy"],
                              selection=native_record.get("cuda_card_selection") if tier == "q8" else None,
                              record=native_record,
                              deadline_monotonic=native_deadline if tier == "q8" and native_selection is not None else None)
        if tier == "q8" and backend == "candle-cuda" and "cuda_card_selection" in native_record:
            native_policy = executed_card_policy(request["policy"], native_record["cuda_card_selection"])
            write(policy_path, native_policy)
            policy = load_policy(policy_path)
        lines = (work / f"{tier}.stdout.log").read_text(encoding="utf-8").splitlines()
        conversion = json.loads(lines[-1])
        config = json.loads((directory / "config.json").read_bytes())
        if (conversion.get("quantized") != tier.upper() or conversion.get("num_tensors", 0) <= 0
                or config.get("quantization", {}).get("bits") != int(tier[1:])):
            raise ValueError(f"converter did not materialize the requested {tier} tier")
        files = inventory([path for path in directory.rglob("*") if path.is_file()], directory)
        if not files:
            raise ValueError(f"converter wrote no {tier} files")
        tiers[tier] = {"conversion": conversion, "files": files, "supervision": supervision}
    write(report / "native-inputs.json", {
        "source": repository_identity(args.expected_sha), "backend": backend,
        "hf_revision": source["revision"], "source_manifest_sha256": source_sha,
        "oracle_sha256": digest(oracle), "tiers": tiers,
        **native_record,
    })
    variables = {
        "SENSENOVA_LOOP_BF16_SNAPSHOT": snapshot,
        "SENSENOVA_LOOP_Q8_SNAPSHOT": work / "q8",
        "SENSENOVA_LOOP_Q4_SNAPSHOT": work / "q4",
        "SENSENOVA_LOOP_ORACLE": oracle,
        "HF_HUB_OFFLINE": "1",
    }
    with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as stream:
        for key, value in variables.items():
            stream.write(f"{key}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mlx", "candle-cuda"), required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--reference-python", type=Path)
    mode.add_argument("--oracle-dir", type=Path)
    args = parser.parse_args()
    if os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch":
        raise ValueError("SenseNova proof runs only via explicit workflow_dispatch")
    repository_identity(args.expected_sha)
    for directory in (args.work, args.report):
        if not directory.resolve().is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
            raise ValueError("SenseNova proof outputs must be owned under RUNNER_TEMP")
        directory.mkdir()
    request = json.loads(os.environ["MODEL_PROOF_REQUEST"])
    if request["sensenova_source_sha"] != args.expected_sha:
        raise ValueError("SenseNova resource request identifies a different engine source")
    phase = reference if args.reference_python else native
    try:
        disk_receipt = args.report / "source-disk-preflight.json" if args.backend == "mlx" else None
        snapshot, source, source_sha = source_snapshot(disk_receipt=disk_receipt)
        phase(args, source, source_sha, snapshot, args.work, args.report, request)
    finally:
        # Upload only known bounded file types, never snapshots, environment or worker caches.
        for label in ("reference", "q8", "q4"):
            for suffix in ("stdout.log", "stderr.log", "events.jsonl", "failure.json"):
                filename = f"{label}.{suffix}"
                trace = args.work / filename
                if trace.is_file() and not trace.is_symlink():
                    with trace.open("rb") as stream:
                        (args.report / filename).write_bytes(stream.read(8 * 1024 * 1024))


if __name__ == "__main__":
    main()
