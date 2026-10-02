"""No accelerator execution: guard regressions for the bounded tile diagnostic."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

CI = Path(__file__).resolve().parents[1] / "ci"
WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/yue2-bf16-tile-diagnostic.yml"
sys.path.insert(0, str(CI))
spec = importlib.util.spec_from_file_location("yue2_bf16_tile_diagnostic", CI / "yue2_bf16_tile_diagnostic.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)
from yue2_decoder_trace_overlay import canonical_tree_order


def synthetic_trace_build(root: Path) -> argparse.Namespace:
    engine = root / "engine"
    overlay = root / "overlay"
    evidence = root / "evidence"
    template = root / "control/scripts/ci/yue2_bf16_tile_diagnostic"
    staged = evidence / "harness"
    for directory in (engine, overlay, evidence, template, staged):
        directory.mkdir(parents=True, exist_ok=True)
    packages = "".join(f'[[package]]\nname="test-dependency-{index}"\nversion="1.0.0"\n'
                       for index in range(205))
    (overlay / "Cargo.lock").write_text(packages, encoding="utf-8")
    for directory in (template, staged):
        (directory / "Cargo.lock.snapshot").write_text(packages, encoding="utf-8")
    (staged / "Cargo.lock").write_text(packages, encoding="utf-8")
    (template / "src").mkdir()
    (staged / "src").mkdir()
    (template / "src/main.rs").write_text("fn main() {}\n", encoding="utf-8")
    (staged / "src/main.rs").write_text("fn main() {}\n", encoding="utf-8")
    vae = overlay / "crates/audio/candle-audio-yue2/src/vae.rs"
    vae.parent.mkdir(parents=True)
    vae.write_text("// diagnostic overlay\n", encoding="utf-8")
    archive = root / "m3-tracked-source.tar.gz"
    patch_file = root / "decoder-trace-vae.patch"
    archive.write_bytes(b"tracked source")
    patch_file.write_bytes(b"declared VAE patch")
    control_sha = "a" * 40
    provenance = {"engine_sha": diag.ENGINE_SHA, "control_sha": control_sha,
                  "tracked_archive_sha256": diag.sha256(archive),
                  "overlay_patch_sha256": diag.sha256(patch_file),
                  "derivative_vae_sha256": diag.sha256(vae),
                  "derivative_tree_sha256": "synthetic-tree",
                  "cargo_lock_sha256": diag.sha256(overlay / "Cargo.lock")}
    provenance_path = root / "overlay-provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    binary = root / "diagnostic.exe"
    binary.write_bytes(b"reviewed executable")
    kernel = (overlay / "crates/media/candle-gen/vendor/candle-kernels").as_posix()
    kernel_id = f"path+file:///{kernel}#0.10.2"
    rows = [
        {"reason": "compiler-artifact", "target": {"name": "candle_core"},
         "features": ["cuda", "cudarc", "default"],
         "package_id": "git+https://github.com/huggingface/candle?rev=1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d#candle-core@0.10.2"},
        {"reason": "compiler-artifact", "target": {"name": "candle_kernels"},
         "package_id": kernel_id},
        {"reason": "compiler-artifact", "target": {"name": "yue2-bf16-tile-diagnostic"},
         "executable": str(binary)},
    ]
    build = evidence / "build.jsonl"
    build.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    identity = {"binary_sha256": diag.sha256(binary), "build_json_sha256": diag.sha256(build),
                "candle_core_features": ["cuda", "cudarc", "default"],
                "candle_core_package_id": rows[0]["package_id"],
                "vendored_kernel_package_id": kernel_id, "derivative_source": str(overlay),
                "native_candle_source": None}
    (evidence / "build-identity.json").write_text(json.dumps(identity), encoding="utf-8")
    def hashes(directory: Path) -> dict:
        return {str(path.relative_to(directory)): diag.sha256(path)
                for path in directory.rglob("*") if path.is_file()}
    harness = {"engine_sha": diag.ENGINE_SHA, "control_sha": control_sha,
               "m3_dependency_tuples": 205,
               "declared_core_source_exception": None, "native_candle_source": None,
               "overlay_sha256": diag.sha256(provenance_path),
               "standalone_lock_sha256": diag.sha256(template / "Cargo.lock.snapshot"),
               "template_files": hashes(template), "staged_files": hashes(staged)}
    (evidence / "harness-provenance.json").write_text(json.dumps(harness), encoding="utf-8")
    return argparse.Namespace(binary=binary, reference=root / "reference", evidence=evidence,
                              engine_sha=diag.ENGINE_SHA, control_sha=control_sha,
                              diagnostic="decoder_trace", overlay_root=overlay)


def synthetic_native_build(root: Path) -> argparse.Namespace:
    args = synthetic_trace_build(root)
    candle = root / "candle-overlay"
    (candle / "candle-core/src/cuda_backend").mkdir(parents=True)
    (candle / "candle-core/src/cuda_backend/mod.rs").write_text("// observed native GEMM column\n",
                                                                encoding="utf-8")
    (candle / "Cargo.toml").write_text("# pinned kernel path\n", encoding="utf-8")
    old = (args.overlay_root / "Cargo.lock").read_text(encoding="utf-8")
    old = old.replace('name="test-dependency-0"\nversion="1.0.0"',
                      'name="candle-core"\nversion="0.10.2"\nsource="git+https://github.com/huggingface/candle?rev=1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d#1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d"')
    (args.overlay_root / "Cargo.lock").write_text(old, encoding="utf-8")
    new = old.replace('source="git+https://github.com/huggingface/candle?rev=1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d#1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d"', '')
    template = root / "control/scripts/ci/yue2_bf16_tile_diagnostic"
    (template / "Cargo.lock.native-convt.snapshot").write_text(new, encoding="utf-8")
    (args.evidence / "harness/Cargo.lock").write_text(new, encoding="utf-8")
    provenance = root / "native-provenance.json"
    provenance.write_text('{}\n', encoding="utf-8")
    build = args.evidence / "build.jsonl"
    rows = [json.loads(line) for line in build.read_text(encoding="utf-8").splitlines()]
    rows[0]["package_id"] = f"path+file:///{candle / 'candle-core'}#0.10.2"
    build.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    identity_path = args.evidence / "build-identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity.update(build_json_sha256=diag.sha256(build),
                    candle_core_package_id=rows[0]["package_id"], native_candle_source=str(candle))
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    harness_path = args.evidence / "harness-provenance.json"
    harness = json.loads(harness_path.read_text(encoding="utf-8"))
    harness.update(m3_dependency_tuples=204,
                   declared_core_source_exception=["candle-core", None, "1.0.0", None],
                   native_candle_source=str(candle), overlay_sha256=diag.sha256(provenance),
                   standalone_lock_sha256=diag.sha256(template / "Cargo.lock.native-convt.snapshot"))
    harness["declared_core_source_exception"] = ["candle-core", None, "0.10.2", None]
    harness["standalone_lock_sha256"] = diag.sha256(template / "Cargo.lock.native-convt.snapshot")
    def hashes(directory: Path) -> dict:
        return {str(path.relative_to(directory)): diag.sha256(path)
                for path in directory.rglob("*") if path.is_file()}
    harness["template_files"] = hashes(template)
    harness["staged_files"] = hashes(args.evidence / "harness")
    harness_path.write_text(json.dumps(harness), encoding="utf-8")
    args.diagnostic = "native_convt_columns"
    args.native_candle_root = candle
    return args


def synthetic_math_report(root: Path, applicable: bool) -> tuple[Path, dict, dict]:
    data = root / "data"
    data.mkdir()
    fixture = root / "crates/audio/candle-audio-yue2/tests/fixtures/vae_real_reference.json"
    fixture.parent.mkdir(parents=True)
    standard = {"repo": "m-a-p/YuE2-Vae", "revision": "95535e72a97bc0f09b8ada125d26b4009428c0e8",
                "weights_sha256": diag.DECODER_SHA256["standard"],
                "config_sha256": "f0191bb9694009956de44e0c361a6f1334760be4c8f848e599bde242a54a0970"}
    meta = {"decoders": {"standard": standard}}
    fixture.write_text(json.dumps(meta), encoding="utf-8")

    def array(name: str, dtype: str, channels: int, frames: int, changed: bool = False) -> dict:
        values = [0.0] * (channels * frames)
        if changed:
            values[4] = 0.03125
        raw = struct.pack("<" + "f" * len(values), *values)
        path = data / f"{name}.f32le"
        path.write_bytes(raw)
        return {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                "layout": "bct_f32le", "shape": [1, channels, frames], "dtype": dtype}

    def arm(label: str, dtype: str, changed: bool = False) -> dict:
        stages = (("input", 64), ("preBias", 1), ("postBias", 1))
        full = {stage: array(f"{label}-full-{stage}", dtype, channels, 75)
                for stage, channels in stages}
        windows = []
        for index, start in enumerate(range(0, 75, 16)):
            end, left = min(start + 16, 75), max(0, start - 16)
            right = min(75, end + 16)
            captures = {stage: array(f"{label}-tile-{index}-{stage}", dtype, channels, right-left,
                                     changed and index == 0 and stage == "preBias")
                        for stage, channels in stages}
            comparisons = {}
            for stage, channels in stages:
                full_values = diag.first_conv_array(data, full[stage], dtype, [1, channels, 75])
                tile_values = diag.first_conv_array(data, captures[stage], dtype, [1, channels, right-left])
                comparisons[stage] = diag.first_conv_comparison(
                    full_values, tile_values, channels, 75, right-left, start, left, end-start)
            windows.append({"start": start, "end": end, "left": left, "right": right,
                            "coreLength": end-start, "captures": captures, "alignedCore": comparisons})
        return {"resident": {"sourceDtype": "F32", "foldDtype": "F32", "residentDtype": dtype,
                             "weightShape": [1,64,7], "biasShape": [1],
                             "weightF32LeSha256": "a"*64, "biasF32LeSha256": "b"*64},
                "full": full, "windows": windows}

    baseline, f32 = arm("bf16", "BF16", applicable), arm("f32", "F32")
    controlled = {"status": "not_applicable", "reason": "no_positive_pre_bias_residual",
                  "flagged": None, "crossArm": None, "modeEventsFile": "math-mode-events.jsonl"}
    events = [{"action": "before_default_arm", "status": "CUBLAS_STATUS_SUCCESS", "rawMode": 0}]
    if applicable:
        flagged = arm("bf16-disallow", "BF16")
        comparisons = {"full": {}, "windows": []}
        stages = (("input",64),("preBias",1),("postBias",1))
        for stage, channels in stages:
            a = diag.first_conv_array(data, baseline["full"][stage], "BF16", [1,channels,75])
            b = diag.first_conv_array(data, flagged["full"][stage], "BF16", [1,channels,75])
            comparisons["full"][stage] = diag.first_conv_all_comparison(a,b,channels,75,0)
        for original, changed in zip(baseline["windows"],flagged["windows"]):
            length = original["right"]-original["left"]
            row = {}
            for stage, channels in stages:
                a = diag.first_conv_array(data, original["captures"][stage], "BF16", [1,channels,length])
                b = diag.first_conv_array(data, changed["captures"][stage], "BF16", [1,channels,length])
                row[stage] = diag.first_conv_all_comparison(a,b,channels,length,original["left"])
            comparisons["windows"].append(row)
        controlled.update(status="collected", reason="positive_pre_bias_residual",
                          flagged=flagged, crossArm=comparisons)
        events += [{"action": action, "status": "CUBLAS_STATUS_SUCCESS", "rawMode": mode}
                   for action,mode in (("before_flagged_arm",0),("set_disallow",16),
                                       ("read_disallow",16),("restore_default",0),("read_restored",0))]
    event_path = data / controlled["modeEventsFile"]
    event_path.write_text("".join(json.dumps(row)+"\n" for row in events), encoding="utf-8")
    controlled["modeEventsSha256"] = hashlib.sha256(event_path.read_bytes()).hexdigest()
    report = {"schemaVersion": 3, "selector": "first_conv_math",
              "purpose": "controlled_diagnostic_only_no_gate_change", "engineSha": diag.ENGINE_SHA,
              "referenceSha256": diag.REFERENCE_SHA256,
              "referenceMetadataSha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
              "latentIdentity": {"sha256": diag.LATENT_SHA256, "shape": [75,64],
                                 "source": {"stage_identity": "precision_reference:long_latent"}},
              "backend": "cuda", "deviceOrdinal": 0, "decoderIdentity": standard,
              "frames": 75, "coreFrames": 16, "haloFrames": 16,
              "operator": {"name": "decoder.layers.0.Conv1d", "kernel": 7, "padding": 3,
                           "stride": 1, "dilation": 1, "groups": 1},
              "originalWaveformObservation": {"runId": "36884387320", "clampedMaxAbs": 0.03125,
                                              "originalBound": 1/64,
                                              "interpretation": "prior_failed_waveform_proof_not_a_first_conv_gate"},
              "runs": {"bf16": baseline, "f32": f32}, "controlled": controlled}
    (data / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return data, meta, report


class DiagnosticGuards(unittest.TestCase):
    def test_native_columns_compare_actual_blck_contributors_and_signed_zero(self):
        full = bytes(75 * 1024 * 12 * 2)
        tile = bytearray(32 * 1024 * 12 * 2)
        tile[3 * 2:3 * 2 + 2] = (0x3f80).to_bytes(2, "little")  # j=3, current row 0, tap 3.
        tile[4 * 2:4 * 2 + 2] = (0x8000).to_bytes(2, "little")  # signed-zero-only change.
        result = diag.native_contributor_stats(full, bytes(tile), "BF16")
        self.assertEqual(result["comparedContributors"], 347136)
        self.assertEqual((result["bitDifferences"], result["positiveDifferences"],
                          result["signedZeroOnly"], result["maxAbs"]), (2, 1, 1, 1.0))
        self.assertEqual(result["firstDifferent"], {
            "rawFrame": 3, "inputRow": 0, "kernelTap": 3, "channel": 0,
            "full": 0.0, "tile": 1.0, "absError": 1.0})
        with self.assertRaisesRegex(RuntimeError, "unexpected lengths"):
            diag.native_contributor_stats(full, bytes(tile[:-2]), "BF16")
        with self.assertRaisesRegex(RuntimeError, "first ConvT divergence"):
            diag.verify_native_column_data(Path("/nonexistent"), {
                "earliestBf16Stage": 1, "adaptive": {"status": "collected", "window": 0},
                "nativeColumns": {"status": "collected", "stage": 2, "window": 0}})

    def test_decoder_trace_tree_hash_uses_case_sensitive_posix_utf8_order(self):
        windows_root = PureWindowsPath("C:/source")
        windows_entries = [windows_root / name for name in
                           ("advisory-ignores.toml", "AGENTS.md", "Zeta")]
        self.assertEqual([path.name for path in canonical_tree_order(windows_root, windows_entries)],
                         ["AGENTS.md", "Zeta", "advisory-ignores.toml"])
        self.assertNotEqual([path.name for path in sorted(windows_entries)],
                            ["AGENTS.md", "Zeta", "advisory-ignores.toml"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("advisory-ignores.toml", "AGENTS.md", "Zeta")
            for name in names:
                (root / name).write_bytes(name.encode("utf-8"))
            expected = hashlib.sha256()
            for name in sorted(names, key=lambda item: item.encode("utf-8")):
                encoded = name.encode("utf-8")
                expected.update(len(encoded).to_bytes(4, "little"))
                expected.update(encoded)
                expected.update(hashlib.sha256(encoded).digest())
            self.assertEqual(diag.tree_digest(root), expected.hexdigest())

    def test_decoder_trace_build_and_harness_mutations_refuse_prelaunch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = synthetic_trace_build(root)
            old = Path.cwd()
            try:
                os.chdir(root / "engine")
                provenance_path = root / "overlay-provenance.json"
                diag.verify_trace_prelaunch_build(args, args.overlay_root, provenance_path)
                build = args.evidence / "build.jsonl"
                identity_path = args.evidence / "build-identity.json"
                harness_path = args.evidence / "harness-provenance.json"
                original_binary = args.binary.read_bytes()
                args.binary.write_bytes(b"changed executable")
                with self.assertRaisesRegex(RuntimeError, "executable or saved build"):
                    diag.verify_trace_prelaunch_build(args, args.overlay_root, provenance_path)
                args.binary.write_bytes(original_binary)
                original_build = build.read_bytes()
                build.write_bytes(original_build + b"{}\n")
                with self.assertRaisesRegex(RuntimeError, "executable or saved build"):
                    diag.verify_trace_prelaunch_build(args, args.overlay_root, provenance_path)
                build.write_bytes(original_build)
                identity = json.loads(identity_path.read_text(encoding="utf-8"))
                identity["derivative_source"] = str(root / "wrong-overlay")
                identity_path.write_text(json.dumps(identity), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "executable or saved build"):
                    diag.verify_trace_prelaunch_build(args, args.overlay_root, provenance_path)
                identity["derivative_source"] = str(args.overlay_root)
                identity_path.write_text(json.dumps(identity), encoding="utf-8")
                rows = [json.loads(line) for line in build.read_text(encoding="utf-8").splitlines()]
                rows[0]["features"].append("cudnn")
                build.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                identity["build_json_sha256"] = diag.sha256(build)
                identity_path.write_text(json.dumps(identity), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "build JSON changed CUDA features"):
                    diag.verify_trace_prelaunch_build(args, args.overlay_root, provenance_path)
                build.write_bytes(original_build)
                identity["build_json_sha256"] = diag.sha256(build)
                identity_path.write_text(json.dumps(identity), encoding="utf-8")
                rows = [json.loads(line) for line in original_build.decode("utf-8").splitlines()]
                rows[1]["package_id"] = "git+https://example.invalid/alternate-kernels"
                build.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                identity["build_json_sha256"] = diag.sha256(build)
                identity["vendored_kernel_package_id"] = rows[1]["package_id"]
                identity_path.write_text(json.dumps(identity), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "build JSON changed CUDA features"):
                    diag.verify_trace_prelaunch_build(args, args.overlay_root, provenance_path)
                build.write_bytes(original_build)
                identity["build_json_sha256"] = diag.sha256(build)
                identity["vendored_kernel_package_id"] = json.loads(
                    original_build.decode("utf-8").splitlines()[1])["package_id"]
                identity_path.write_text(json.dumps(identity), encoding="utf-8")
                harness = json.loads(harness_path.read_text(encoding="utf-8"))
                harness["overlay_sha256"] = "0" * 64
                harness_path.write_text(json.dumps(harness), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "harness provenance"):
                    diag.verify_trace_prelaunch_build(args, args.overlay_root, provenance_path)
                harness["overlay_sha256"] = diag.sha256(provenance_path)
                harness_path.write_text(json.dumps(harness), encoding="utf-8")
                (args.evidence / "harness/src/main.rs").write_text("fn changed() {}\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "harness source"):
                    diag.verify_trace_prelaunch_build(args, args.overlay_root, provenance_path)
            finally:
                os.chdir(old)

    def test_native_column_build_allows_only_declared_core_source_and_exact_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = synthetic_native_build(root)
            old = Path.cwd()
            try:
                os.chdir(root / "engine")
                provenance = root / "native-provenance.json"
                diag.verify_trace_prelaunch_build(args, args.overlay_root, provenance,
                                                  args.native_candle_root)
                build = args.evidence / "build.jsonl"
                identity_path = args.evidence / "build-identity.json"
                original = build.read_bytes()
                rows = [json.loads(line) for line in original.decode("utf-8").splitlines()]
                rows[0]["package_id"] = "git+https://github.com/huggingface/candle#other-core"
                build.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                identity = json.loads(identity_path.read_text(encoding="utf-8"))
                identity["build_json_sha256"] = diag.sha256(build)
                identity["candle_core_package_id"] = rows[0]["package_id"]
                identity_path.write_text(json.dumps(identity), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "derivative Candle backend"):
                    diag.verify_trace_prelaunch_build(args, args.overlay_root, provenance,
                                                      args.native_candle_root)
                build.write_bytes(original)
                identity["build_json_sha256"] = diag.sha256(build)
                identity["candle_core_package_id"] = json.loads(
                    original.decode("utf-8").splitlines()[0])["package_id"]
                identity_path.write_text(json.dumps(identity), encoding="utf-8")
                args.binary.write_bytes(b"replaced after build")
                with self.assertRaisesRegex(RuntimeError, "executable or saved build"):
                    diag.verify_trace_prelaunch_build(args, args.overlay_root, provenance,
                                                      args.native_candle_root)
            finally:
                os.chdir(old)

    def test_native_source_patches_and_workflow_preserve_only_bounded_selector(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("repository: huggingface/candle", workflow)
        self.assertIn("ref: 1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d", workflow)
        self.assertIn('if: inputs.diagnostic == \'native_convt_columns\'', workflow)
        self.assertIn("--features cuda,native_convt_columns", workflow)
        self.assertIn("--native-candle-root", workflow)
        self.assertIn("native-convt-candle-kernel-path.patch", workflow)
        self.assertIn("--locked --offline --release", workflow)
        source = (CI / "yue2_native_convt_overlay.py").read_text(encoding="utf-8")
        for required in ("CANDLE_SHA", "CANDLE_TREE", "CANDLE_BACKEND_SHA",
                         "apply_one_patch(candle_overlay, backend_patch, CANDLE_BACKEND)",
                         'apply_one_patch(candle_overlay, kernel_path_patch, Path("Cargo.toml"))',
                         "candle_derivative_tree_sha256"):
            self.assertIn(required, source)
        self.assertNotIn("cublasSetMathMode", source)

    def test_decoder_trace_binary_mismatch_reaches_neither_census_nor_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = synthetic_trace_build(root)
            args.binary.write_bytes(b"replaced executable")
            old = Path.cwd()
            try:
                os.chdir(root / "engine")
                with patch.dict(os.environ, RUNNER_NAME="cuda-windows-2", CUDA_VISIBLE_DEVICES="0"), \
                     patch.object(diag, "verify_revisions"), patch.object(diag, "verify_reference"), \
                     patch.object(diag, "tree_digest", return_value="synthetic-tree"), \
                     patch.object(diag.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="")), \
                     patch.object(diag, "cuda_census") as census, \
                     patch.object(diag.subprocess, "Popen") as launch:
                    with self.assertRaisesRegex(RuntimeError, "executable or saved build"):
                        diag.execute(args)
                    census.assert_not_called()
                    launch.assert_not_called()
            finally:
                os.chdir(old)

    def test_decoder_trace_resource_bound_comes_from_all_33_native_stage_shapes(self):
        expected = diag.decoder_trace_expected_bytes()
        self.assertEqual(expected["source_lengths"], [75, 32, 48, 48, 43, 27])
        self.assertEqual(expected["stage_count"], 33)
        self.assertEqual(expected["bf16_raw_bytes"], 817244160)
        self.assertEqual(expected["f32_raw_bytes"], 1634488320)
        self.assertEqual(expected["max_stage_f32_bytes"], 36847616)

    def test_decoder_trace_inventory_refuses_extra_missing_corrupt_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            raw = b"\x00\x80\xc0\x3f"
            (data / "native.bf16le").write_bytes(raw)
            report = {"capture": {"file": "native.bf16le", "sha256": hashlib.sha256(raw).hexdigest(),
                                  "bytes": len(raw)}}
            (data / "report.json").write_text(json.dumps(report), encoding="utf-8")
            diag.verify_decoder_trace_file_set(data, report, minimum_files=1)
            with self.assertRaisesRegex(RuntimeError, "coverage incomplete"):
                diag.verify_decoder_trace_file_set(data, report)
            (data / "extra.bin").write_bytes(b"extra")
            with self.assertRaisesRegex(RuntimeError, "unreferenced"):
                diag.verify_decoder_trace_file_set(data, report, minimum_files=1)
            (data / "extra.bin").unlink()
            (data / "native.bf16le").write_bytes(b"wrong")
            with self.assertRaisesRegex(RuntimeError, "hash/size changed"):
                diag.verify_decoder_trace_file_set(data, report, minimum_files=1)
            (data / "native.bf16le").unlink()
            (data / "native.bf16le").symlink_to(data / "report.json")
            with self.assertRaisesRegex(RuntimeError, "hash/size changed"):
                diag.verify_decoder_trace_file_set(data, report, minimum_files=1)
            (data / "native.bf16le").unlink()
            with self.assertRaisesRegex(RuntimeError, "hash/size changed"):
                diag.verify_decoder_trace_file_set(data, report, minimum_files=1)
    def test_explicit_selector_keeps_waveform_default_and_forwards_to_child(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("default: waveform", workflow)
        self.assertIn("options: [waveform, first_conv, first_conv_math, decoder_trace, native_convt_columns]", workflow)
        self.assertIn('run --diagnostic "$env:YUE2_DIAGNOSTIC_SELECTOR"', workflow)
        self.assertEqual(diag.DIAGNOSTICS, ("waveform", "first_conv", "first_conv_math",
                                            "decoder_trace", "native_convt_columns"))

    def test_first_conv_math_conditional_arms_and_restore_receipt(self):
        for applicable in (False, True):
            with self.subTest(applicable=applicable), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data, meta, report = synthetic_math_report(root, applicable)
                old = Path.cwd()
                try:
                    os.chdir(root)
                    diag.verify_first_conv_math_data(data, meta)
                    if applicable:
                        comparison = report["controlled"]["crossArm"]["windows"][0]["preBias"]
                        comparison["differentValues"] = 0
                        (data / "report.json").write_text(json.dumps(report), encoding="utf-8")
                        with self.assertRaisesRegex(RuntimeError, "flagged window tensor comparison"):
                            diag.verify_first_conv_math_data(data, meta)
                        comparison["differentValues"] = 1
                        (data / "report.json").write_text(json.dumps(report), encoding="utf-8")
                        events = data / "math-mode-events.jsonl"
                        rows = events.read_text(encoding="utf-8").splitlines()
                        rows[-1] = json.dumps({"action":"read_restored","status":"CUBLAS_STATUS_SUCCESS","rawMode":16})
                        events.write_text("\n".join(rows)+"\n", encoding="utf-8")
                        report["controlled"]["modeEventsSha256"] = hashlib.sha256(events.read_bytes()).hexdigest()
                        (data / "report.json").write_text(json.dumps(report), encoding="utf-8")
                        with self.assertRaisesRegex(RuntimeError, "sequence incomplete"):
                            diag.verify_first_conv_math_data(data, meta)
                    else:
                        report["controlled"]["status"] = "collected"
                        (data / "report.json").write_text(json.dumps(report), encoding="utf-8")
                        with self.assertRaisesRegex(RuntimeError, "inapplicable"):
                            diag.verify_first_conv_math_data(data, meta)
                finally:
                    os.chdir(old)

    def test_first_conv_math_requires_exact_data_files_in_both_outcomes(self):
        for applicable in (False, True):
            with self.subTest(applicable=applicable), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data, meta, report = synthetic_math_report(root, applicable)
                old = Path.cwd()
                try:
                    os.chdir(root)
                    diag.verify_first_conv_math_data(data, meta)

                    extra = data / "unreferenced-flagged.f32le"
                    extra.write_bytes(b"unreferenced")
                    with self.assertRaisesRegex(RuntimeError, "file set differs"):
                        diag.verify_first_conv_math_data(data, meta)
                    extra.unlink()

                    unexpected_dir = data / "unreferenced-directory"
                    unexpected_dir.mkdir()
                    with self.assertRaisesRegex(RuntimeError, "non-regular entry"):
                        diag.verify_first_conv_math_data(data, meta)
                    unexpected_dir.rmdir()

                    array = data / report["runs"]["bf16"]["full"]["input"]["file"]
                    original_array = array.read_bytes()
                    array.unlink()
                    with self.assertRaisesRegex(RuntimeError, "array escaped evidence"):
                        diag.verify_first_conv_math_data(data, meta)
                    array.write_bytes(original_array)

                    events = data / "math-mode-events.jsonl"
                    original_events = events.read_bytes()
                    events.unlink()
                    with self.assertRaisesRegex(RuntimeError, "event hash mismatch"):
                        diag.verify_first_conv_math_data(data, meta)
                    events.write_bytes(original_events)

                    if os.name != "nt":
                        link = data / "unreferenced-symlink"
                        link.symlink_to(array)
                        with self.assertRaisesRegex(RuntimeError, "non-regular entry"):
                            diag.verify_first_conv_math_data(data, meta)
                        link.unlink()

                    diag.verify_first_conv_math_data(data, meta)
                finally:
                    os.chdir(old)

    def test_math_gate_signed_zero_and_input_mismatch_do_not_enable_flag(self):
        windows = [{"alignedCore": {"input": {"differentValues": 0},
                                    "preBias": {"differentValues": 1, "maxAbs": 0.0}}}]
        self.assertEqual(diag.first_conv_math_gate(windows), "no_positive_pre_bias_residual")
        windows[0]["alignedCore"]["preBias"]["maxAbs"] = 0.03125
        self.assertEqual(diag.first_conv_math_gate(windows), "positive_pre_bias_residual")
        windows[0]["alignedCore"]["input"]["differentValues"] = 1
        self.assertEqual(diag.first_conv_math_gate(windows), "input_core_mismatch")

    def test_math_report_preserves_input_bit_mismatch_as_not_applicable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, meta, report = synthetic_math_report(root, False)
            window = report["runs"]["bf16"]["windows"][0]
            input_row = window["captures"]["input"]
            path = data / input_row["file"]
            raw = bytearray(path.read_bytes())
            raw[4*4:5*4] = struct.pack("<f", -0.0)
            path.write_bytes(raw)
            input_row["sha256"] = hashlib.sha256(raw).hexdigest()
            full = diag.first_conv_array(data, report["runs"]["bf16"]["full"]["input"],
                                         "BF16", [1,64,75])
            tile = diag.first_conv_array(data, input_row, "BF16", [1,64,32])
            window["alignedCore"]["input"] = diag.first_conv_comparison(full,tile,64,75,32,0,0,16)
            report["controlled"]["reason"] = "input_core_mismatch"
            (data / "report.json").write_text(json.dumps(report), encoding="utf-8")
            old = Path.cwd()
            try:
                os.chdir(root)
                diag.verify_first_conv_math_data(data, meta)
            finally:
                os.chdir(old)

    def test_first_conv_f32_json_scalars_compare_by_actual_tensor_bits(self):
        exact = struct.unpack("<f", struct.pack("<f", 1.2345679))[0]
        expected = {"maxAbs": exact, "differentValues": 1, "comparedValues": 16,
                    "firstDifferent": {"channel": 2, "globalLatentFrame": 17,
                                       "fullValue": exact, "tileValue": 0.0, "absError": exact}}
        rounded_json = json.loads(json.dumps(expected).replace(str(exact), "1.2345679"))
        self.assertNotEqual(expected, rounded_json)
        self.assertTrue(diag.same_first_conv_comparison(rounded_json, expected))
        rounded_json["firstDifferent"]["globalLatentFrame"] += 1
        self.assertFalse(diag.same_first_conv_comparison(rounded_json, expected))

    def test_first_conv_report_recomputes_raw_residuals_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            fixture = root / "crates/audio/candle-audio-yue2/tests/fixtures/vae_real_reference.json"
            fixture.parent.mkdir(parents=True)
            standard = {"repo": "m-a-p/YuE2-Vae",
                        "revision": "95535e72a97bc0f09b8ada125d26b4009428c0e8",
                        "weights_sha256": diag.DECODER_SHA256["standard"],
                        "config_sha256": "f0191bb9694009956de44e0c361a6f1334760be4c8f848e599bde242a54a0970"}
            meta = {"decoders": {"standard": standard}}
            fixture.write_text(json.dumps(meta), encoding="utf-8")

            def array(name, dtype, channels, frames):
                raw = struct.pack("<" + "f" * (channels * frames), *([0.0] * (channels * frames)))
                path = data / f"{name}.f32le"
                path.write_bytes(raw)
                return {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw), "layout": "bct_f32le",
                        "shape": [1, channels, frames], "dtype": dtype}

            runs = {}
            for label, dtype in (("bf16", "BF16"), ("f32", "F32")):
                full = {stage: array(f"{label}-full-{stage}", dtype, channels, 75)
                        for stage, channels in (("input", 64), ("preBias", 1), ("postBias", 1))}
                windows = []
                for index, start in enumerate(range(0, 75, 16)):
                    end = min(start + 16, 75)
                    left, right = max(0, start - 16), min(75, end + 16)
                    captures = {stage: array(f"{label}-tile-{index}-{stage}", dtype, channels, right - left)
                                for stage, channels in (("input", 64), ("preBias", 1), ("postBias", 1))}
                    comparisons = {stage: {"maxAbs": 0.0, "differentValues": 0, "firstDifferent": None,
                                           "comparedValues": channels * (end - start)}
                                   for stage, channels in (("input", 64), ("preBias", 1), ("postBias", 1))}
                    windows.append({"start": start, "end": end, "left": left, "right": right,
                                    "coreLength": end - start, "captures": captures,
                                    "alignedCore": comparisons})
                runs[label] = {"resident": {"sourceDtype": "F32", "foldDtype": "F32",
                                            "residentDtype": dtype, "weightShape": [1, 64, 7],
                                            "biasShape": [1], "weightF32LeSha256": "a" * 64,
                                            "biasF32LeSha256": "b" * 64},
                               "full": full, "windows": windows}
            report = {"schemaVersion": 2, "selector": "first_conv",
                      "purpose": "diagnostic_only_no_gate_change", "engineSha": diag.ENGINE_SHA,
                      "referenceSha256": diag.REFERENCE_SHA256,
                      "referenceMetadataSha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                      "latentIdentity": {"sha256": diag.LATENT_SHA256, "shape": [75, 64],
                                         "source": {"stage_identity": "precision_reference:long_latent"}},
                      "backend": "cuda", "deviceOrdinal": 0, "decoderIdentity": standard,
                      "frames": 75, "coreFrames": 16, "haloFrames": 16,
                      "operator": {"name": "decoder.layers.0.Conv1d", "kernel": 7,
                                   "padding": 3, "stride": 1, "dilation": 1, "groups": 1},
                      "originalWaveformObservation": {"runId": "36884387320", "clampedMaxAbs": 0.03125,
                                                      "originalBound": 1 / 64,
                                                      "interpretation": "prior_failed_waveform_proof_not_a_first_conv_gate"},
                      "runs": runs}
            report_path = data / "report.json"
            old = Path.cwd()
            try:
                os.chdir(root)
                report_path.write_text(json.dumps(report), encoding="utf-8")
                diag.verify_first_conv_data(data, meta)
                window = report["runs"]["bf16"]["windows"][0]
                tile = window["captures"]["preBias"]
                tile_path = data / tile["file"]
                raw = bytearray(tile_path.read_bytes())
                raw[4 * 4:5 * 4] = struct.pack("<f", 0.03125)
                tile_path.write_bytes(raw)
                tile["sha256"] = hashlib.sha256(raw).hexdigest()
                window["alignedCore"]["preBias"] = {
                    "maxAbs": 0.03125, "differentValues": 1, "comparedValues": 16,
                    "firstDifferent": {"channel": 0, "globalLatentFrame": 4,
                                       "fullValue": 0.0, "tileValue": 0.03125, "absError": 0.03125}}
                report_path.write_text(json.dumps(report), encoding="utf-8")
                diag.verify_first_conv_data(data, meta)
                window["alignedCore"]["preBias"]["differentValues"] = 0
                report_path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "residual does not match"):
                    diag.verify_first_conv_data(data, meta)
                window["alignedCore"]["preBias"]["differentValues"] = 1
                report_path.write_text(json.dumps(report), encoding="utf-8")
                (data / runs["bf16"]["full"]["preBias"]["file"]).write_bytes(b"corrupt")
                with self.assertRaisesRegex(RuntimeError, "size/hash mismatch"):
                    diag.verify_first_conv_data(data, meta)
            finally:
                os.chdir(old)

    @staticmethod
    def assert_locked_fetch_before_offline_build(workflow: str) -> None:
        step = workflow.split("      - name: Build the external M3 VAE diagnostic harness\n", 1)[1].split(
            "      - name: Run only the bounded same-input VAE diagnostic\n", 1)[0]
        lines = [line.strip() for line in step.splitlines()]
        prepare = next(i for i, line in enumerate(lines) if "prepare-harness" in line)
        fetch = next(i for i, line in enumerate(lines) if line.startswith("cargo fetch "))
        builds = [i for i, line in enumerate(lines) if line.startswith("cargo build ")]
        assert len(builds) == 3 and prepare < fetch < min(builds)
        assert lines[fetch] == (
            'cargo fetch --locked --manifest-path "%RUNNER_TEMP%\\yue2-bf16-tile-diagnostic\\harness\\Cargo.toml" '
            '--target x86_64-pc-windows-msvc > "%RUNNER_TEMP%\\yue2-bf16-tile-diagnostic\\fetch.log" 2>&1')
        assert lines[fetch + 1] == (
            'if errorlevel 1 (type "%RUNNER_TEMP%\\yue2-bf16-tile-diagnostic\\fetch.log"& exit /b 1)')
        prefix = ('cargo build --locked --offline --release --manifest-path '
                  '"%RUNNER_TEMP%\\yue2-bf16-tile-diagnostic\\harness\\Cargo.toml" --features ')
        suffix = (' --message-format=json > "%RUNNER_TEMP%\\yue2-bf16-tile-diagnostic\\build.jsonl" '
                  '2> "%RUNNER_TEMP%\\yue2-bf16-tile-diagnostic\\build.log"')
        assert {lines[i].removeprefix(prefix).removesuffix(suffix) for i in builds} == {
            "cuda", "cuda,decoder_trace", "cuda,native_convt_columns"}
        assert all(lines[i].startswith(prefix) and lines[i].endswith(suffix) for i in builds)
        assert 'path: ${{ runner.temp }}/yue2-bf16-tile-diagnostic' in workflow

    def test_locked_target_fetch_stages_before_unchanged_offline_build(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assert_locked_fetch_before_offline_build(workflow)
        mutations = {
            "missing fetch": ('          cargo fetch --locked', '          cargo missing --locked'),
            "unlocked fetch": ('cargo fetch --locked', 'cargo fetch'),
            "wrong target": ('--target x86_64-pc-windows-msvc', '--target x86_64-unknown-linux-gnu'),
            "ignored fetch error": ('if errorlevel 1 (type "%RUNNER_TEMP%\\yue2-bf16-tile-diagnostic\\fetch.log"& exit /b 1)', 'rem ignored fetch error'),
            "online build": ('cargo build --locked --offline', 'cargo build --locked'),
        }
        for name, (old, new) in mutations.items():
            with self.subTest(name=name):
                self.assertIn(old, workflow)
                with self.assertRaises((AssertionError, StopIteration)):
                    self.assert_locked_fetch_before_offline_build(workflow.replace(old, new, 1))
        fetch_line = next(line for line in workflow.splitlines() if "cargo fetch --locked" in line)
        build_line = next(line for line in workflow.splitlines() if "cargo build --locked --offline" in line)
        moved = workflow.replace(fetch_line + "\n", "", 1).replace(build_line, build_line + "\n" + fetch_line, 1)
        with self.assertRaises(AssertionError):
            self.assert_locked_fetch_before_offline_build(moved)

    def test_foreign_context_never_launches_a_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "diagnostic.exe"
            binary.write_bytes(b"not executed")
            args = argparse.Namespace(binary=binary, reference=root / "reference", evidence=root / "evidence",
                                      engine_sha=diag.ENGINE_SHA, control_sha="a" * 40)
            with patch.dict(os.environ, RUNNER_NAME="cuda-windows", CUDA_VISIBLE_DEVICES="0"), \
                 patch.object(diag, "verify_revisions"), patch.object(diag, "verify_reference"), \
                 patch.object(diag.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="")), \
                 patch.object(diag, "cuda_census", return_value=("foreign PID", ["foreign PID"])), \
                 patch.object(diag.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(RuntimeError, "foreign accelerator ownership"):
                    diag.execute(args)
                launch.assert_not_called()
            self.assertEqual((args.evidence / "census-before.txt").read_text(encoding="utf-8"), "foreign PID")

    def test_other_source_cannot_run_under_the_failed_source_identity(self):
        with patch.object(diag, "verify_revisions") as verify:
            with self.assertRaisesRegex(RuntimeError, "exact failed M3"):
                diag.execute(argparse.Namespace(engine_sha="b" * 40))
            verify.assert_not_called()

    def test_staging_cannot_dirty_stationary_engine_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "engine"
            engine.mkdir()
            template = root / "template"
            template.mkdir()
            args = argparse.Namespace(engine_sha=diag.ENGINE_SHA, control_sha="a" * 40,
                                      template=template, destination=engine / "harness")
            old = Path.cwd()
            try:
                os.chdir(engine)
                with patch.object(diag, "verify_revisions"):
                    with self.assertRaisesRegex(RuntimeError, "outside stationary checkouts"):
                        diag.prepare_harness(args)
                self.assertFalse(args.destination.exists())
            finally:
                os.chdir(old)

    def test_dependency_drift_refuses_before_building_an_alternate_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "engine"
            engine.mkdir()
            (engine / "Cargo.lock").write_text('[[package]]\nname="candle-core"\nversion="0.10.2"\n', encoding="utf-8")
            template = root / "template"
            template.mkdir()
            (template / "Cargo.lock.snapshot").write_text('[[package]]\nname="candle-core"\nversion="0.10.3"\n', encoding="utf-8")
            args = argparse.Namespace(engine_sha=diag.ENGINE_SHA, control_sha="a" * 40,
                                      template=template, destination=root / "outside" / "harness")
            old = Path.cwd()
            try:
                os.chdir(engine)
                with patch.object(diag, "verify_revisions"):
                    with self.assertRaisesRegex(RuntimeError, "dependencies differ"):
                        diag.prepare_harness(args)
                self.assertFalse(args.destination.exists())
            finally:
                os.chdir(old)

    def test_ambiguous_build_output_cannot_select_an_unrelated_binary(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "diagnostic.exe"
            binary.touch()
            row = {"reason": "compiler-artifact", "target": {"name": "yue2-bf16-tile-diagnostic"},
                   "executable": str(binary)}
            build = root / "build.jsonl"
            build.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "one diagnostic executable"):
                diag.resolve_binary(build, root / "binary.txt")
            self.assertFalse((root / "binary.txt").exists())

    def test_different_cuda_implementation_cannot_reach_execution(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "diagnostic.exe"
            binary.touch()
            target = {"reason": "compiler-artifact", "target": {"name": "yue2-bf16-tile-diagnostic"},
                      "executable": str(binary)}
            core = {"reason": "compiler-artifact", "target": {"name": "candle_core"},
                    "features": ["cuda", "cudarc", "default", "cudnn"]}
            kernel = {"reason": "compiler-artifact", "target": {"name": "candle_kernels"},
                      "package_id": "path+file:///D:/actions/engine/crates/media/candle-gen/vendor/candle-kernels#0.10.2"}
            build = root / "build.jsonl"
            build.write_text("".join(json.dumps(row) + "\n" for row in (target, core, kernel)), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Candle features differ"):
                diag.resolve_binary(build, root / "binary.txt")
            core["features"].remove("cudnn")
            kernel["package_id"] = "git+https://github.com/huggingface/candle#candle-kernels@0.10.2"
            build.write_text("".join(json.dumps(row) + "\n" for row in (target, core, kernel)), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "vendored CUDA kernels"):
                diag.resolve_binary(build, root / "binary.txt")
            self.assertFalse((root / "binary.txt").exists())


if __name__ == "__main__":
    unittest.main()
