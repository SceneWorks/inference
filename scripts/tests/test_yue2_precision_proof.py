"""CPU-only controls for the dispatch-only precision proof."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("yue2_precision_proof", ROOT / "scripts/ci/yue2_precision_proof.py")
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)
WORKFLOW = ROOT / ".github/workflows/yue2-precision-proof.yml"


class PrecisionControlTests(unittest.TestCase):
    def test_workflow_control_and_engine_revisions_are_independent_and_exact(self):
        engine, control = "a" * 40, "b" * 40
        result = type("Result", (), {"stdout": engine + "\n"})()
        with patch.dict("os.environ", {"GITHUB_SHA": control}), \
             patch.object(CONTROL.subprocess, "run", return_value=result):
            CONTROL.verify_revisions(engine, control)
            with self.assertRaisesRegex(RuntimeError, "control SHA differs"):
                CONTROL.verify_revisions(engine, "c" * 40)
            with self.assertRaisesRegex(RuntimeError, "engine checkout differs"):
                CONTROL.verify_revisions("c" * 40, control)
            with self.assertRaisesRegex(RuntimeError, "control SHA must be"):
                CONTROL.verify_revisions(engine, "short")

    def test_pmon_refuses_compute_and_mixed_compute_graphics(self):
        output = "# gpu pid type fb sm\n0 111 G 12 0\n0 222 C+G 0 0\n0 333 C 256 75\n"
        self.assertEqual(CONTROL.compute_capable_rows(output), ["0 222 C+G 0 0", "0 333 C 256 75"])
        self.assertEqual(CONTROL.compute_capable_rows("# gpu pid type fb sm\n0 111 G 12 0\n"), [])
        with self.assertRaisesRegex(RuntimeError, "typed process columns"):
            CONTROL.compute_capable_rows("0 333 C 256 75\n")
        for ambiguous in ("0 444 ? 0 0", "0 xyz C+G 0 0", "0 555", "0 - C 0 0", "1 444 C 0 0"):
            with self.subTest(ambiguous=ambiguous), self.assertRaises(RuntimeError):
                CONTROL.compute_capable_rows("# gpu pid type fb sm\n" + ambiguous + "\n")
        self.assertEqual(CONTROL.compute_capable_rows("# gpu pid type fb sm\n0 - - - -\n"), [])

    def test_cuda_pmon_fallback_refuses_any_query_process_and_fails_closed(self):
        failed = type("Result", (), {"returncode": 1, "stdout": "", "stderr": "pmon unsupported"})()
        apps = type("Result", (), {"returncode": 0, "stdout": "222, C:\\cuda-test.exe\n", "stderr": ""})()
        with patch.object(CONTROL.subprocess, "run", side_effect=[failed, apps]) as run:
            raw, busy = CONTROL.cuda_census()
        self.assertIn("pmon unsupported", raw)
        self.assertEqual(busy, ["222, C:\\cuda-test.exe"])
        self.assertIn("--query-compute-apps=pid,process_name", run.call_args_list[1].args[0])
        for bad in ("not a csv row", "222, ", "abc, C:\\cuda-test.exe"):
            with self.subTest(bad=bad), self.assertRaisesRegex(RuntimeError, "ambiguous"):
                CONTROL.query_compute_apps_rows(bad)
        query_failed = type("Result", (), {"returncode": 1, "stdout": "", "stderr": "unsupported"})()
        with patch.object(CONTROL.subprocess, "run", side_effect=[failed, query_failed]):
            with self.assertRaisesRegex(RuntimeError, "CUDA census unavailable"):
                CONTROL.cuda_census()

    def test_metal_census_refuses_foreign_workers_by_executable_only(self):
        rows = "\n".join((
            "101 /tmp/precision_real_weights-deadbeef",
            "102 /tmp/sequential_residency_real_weights-deadbeef",
            "103 /tmp/mlx-gen-qwen-image",
            "104 /tmp/memory-mlx-adapter",
            "105 /tmp/sceneworks-worker",
            "106 /opt/actions-runner/bin/Runner.Worker",
            "107 /bin/zsh",
            "108 /Applications/Safari.app/Contents/MacOS/Safari",
            "109 /tmp/mlx-gen",
            "110 /tmp/candle-gen",
        )) + "\n"
        result = type("Result", (), {"returncode": 0, "stdout": rows, "stderr": ""})()
        with patch.object(CONTROL.subprocess, "run", return_value=result) as run:
            raw, busy = CONTROL.metal_census()
        self.assertEqual(raw, rows)
        self.assertEqual([int(line.split()[0]) for line in busy], [101, 102, 103, 104, 105, 109, 110])
        self.assertEqual(run.call_args.args[0], ["/bin/ps", "-axo", "pid=,comm="])

    def test_one_exact_ignored_test_must_execute(self):
        good = "test explicit_stage_precision_real_weights ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out\n"
        self.assertTrue(CONTROL.one_test_executed(good))
        for bad in (good.replace("1 passed", "0 passed"), good.replace("explicit_stage_precision_real_weights", "wrong_test"), good.replace("0 ignored", "1 ignored")):
            self.assertFalse(CONTROL.one_test_executed(bad))

    def test_stage_markers_remain_machine_readable(self):
        line = 'test explicit_stage_precision_real_weights ... YUE2_PRECISION_STAGE {"stage":"Bf16:standard:encoder","event":"start","unixMs":100}'
        self.assertEqual(CONTROL.stage_markers(line)[0]["unixMs"], 100)
        with self.assertRaises(json.JSONDecodeError):
            CONTROL.stage_markers(line.replace('"unixMs":100', '"unixMs":'))
        markers = [{'stage':'Bf16:standard:encoder','event':'start','unixMs':100},
                   {'stage':'Bf16:standard:encoder','event':'end','unixMs':200}]
        samples = [{'started_utc_ns':110_000_000,'ended_utc_ns':120_000_000},
                   {'started_utc_ns':195_000_000,'ended_utc_ns':205_000_000}]
        self.assertEqual(CONTROL.stage_sample_coverage(markers,samples)['Bf16:standard:encoder'],
                         {'fully_contained_samples':1,'overlapping_samples':2})
        self.assertIn('Bf16:cross_policy_cached_decode:end', CONTROL.missing_stage_markers(markers))

    def test_reference_mutation_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "vae_real_reference.safetensors").write_bytes(b"wrong reference")
            (root / "reference-provenance.json").write_text(json.dumps({"engine_sha": "a" * 40, "sha256": CONTROL.REFERENCE_SHA256}), encoding="utf-8")
            args = type("Args", (), {"directory": root, "engine_sha": "a" * 40})()
            with self.assertRaisesRegex(RuntimeError, "digest differs"):
                CONTROL.verify_reference(args)
            args.engine_sha = "b" * 40
            with self.assertRaisesRegex(RuntimeError, "engine SHA differs"):
                CONTROL.verify_reference(args)

    def test_actual_rust_receipt_shape_and_cross_policy_mutations(self):
        def decoder(variant, dtype):
            return {"variant": variant, "weightsSha256": CONTROL.DECODER_SHA256[variant],
                    "parameterDtype": dtype, "activationDtype": dtype,
                    "decodeCases": [{"name": "long"}, {"name": "production"}],
                    "encoderMean": {"snrDb": 100}, "encoderScale": {"snrDb": 100}}
        cases = []
        for index, policy in enumerate(("Bf16", "Auto", "Fp32"), 1):
            vae = "bfloat16" if policy == "Bf16" else "float32"
            model = "float32" if policy == "Fp32" else "bfloat16"
            dtype = "BF16" if policy == "Bf16" else "F32"
            cases.append({"requestedPolicy": policy,
                          "effectiveDtypes": {"ar": model, "nar": model, "vaeDecoder": vae, "vaeEncoder": vae},
                          "generation": {"config": {"compute_policy": policy.lower(), "vae_dtype": vae},
                                         "runIdentity": str(index), "legacyCacheIdentity": str(index + 3)},
                          "decoders": [decoder("standard", dtype), decoder("legacy", dtype)]})
        receipt = {"schemaVersion": 1, "backend": "cuda", "referenceSha256": CONTROL.REFERENCE_SHA256,
                   "listeningDir": "/persistent/audio/run-1",
                   "cases": cases, "legacyToBf16CachedDecode": {
                       "sourceIdentity": "legacy", "targetIdentity": "new-waveform",
                       "sourceGeneration": {"identity": "legacy", "config": {"vae_dtype": "float32"}},
                       "targetConfig": {"compute_policy": "bf16", "vae_dtype": "bfloat16",
                                        "decoder_release": "legacy", "cached_decode": {"source_identity": "legacy"}}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            def check(value):
                path.write_text(json.dumps(value), encoding="utf-8")
                CONTROL.validate_receipt(path, "cuda", Path("/persistent/audio/run-1"))
            check(receipt)
            for mutation in (
                lambda x: x["cases"][0]["effectiveDtypes"].__setitem__("vaeDecoder", "float32"),
                lambda x: x["cases"][0]["decoders"][1].__setitem__("weightsSha256", "bad"),
                lambda x: x["legacyToBf16CachedDecode"]["targetConfig"].__setitem__("vae_dtype", "float32"),
                lambda x: x.__setitem__("listeningDir", "/tmp/deleted-audio"),
            ):
                changed = json.loads(json.dumps(receipt))
                mutation(changed)
                with self.assertRaises(RuntimeError):
                    check(changed)

    def test_workflow_is_dispatch_only_and_selects_one_new_test(self):
        source = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", source)
        self.assertNotIn("schedule:", source)
        self.assertIn("options: [fixture, cuda, metal, cuda-diagnostic]", source)
        self.assertIn("if: inputs.stage == 'fixture'", source)
        self.assertIn("if: inputs.stage == 'cuda'", source)
        self.assertIn("if: inputs.stage == 'metal'", source)
        self.assertIn("group: inference-real-weights-physical-host", source)
        self.assertIn('CUDA_VISIBLE_DEVICES: "0"', source)
        self.assertEqual(source.count("path: ${{ env.YUE2_PRECISION_WORK_DIR }}/**/*.wav"), 2)
        self.assertEqual(source.count("if: ${{ always() && env.YUE2_PRECISION_WORK_DIR != '' }}"), 2)
        self.assertNotIn("path: ${{ env.YUE2_PRECISION_WORK_DIR }}\n", source)
        self.assertIn("yue2-precision-listening-cuda-cc-by-nc-internal-", source)
        self.assertIn("yue2-precision-listening-metal-cc-by-nc-internal-", source)
        self.assertIn("test \"$RUNNER_NAME\" = nax-macos-2", source)
        self.assertIn("--test precision_real_weights", source)
        self.assertIn("expected_control_sha:", source)
        self.assertIn("ref: ${{ inputs.expected_engine_sha }}", source)
        self.assertLess(source.index("Select checked Git Bash before pinned Rust"),
                        source.index("uses: dtolnay/rust-toolchain@"))
        self.assertIn('if not exist "C:\\Program Files\\Git\\bin\\bash.exe" exit /b 1', source)
        self.assertIn('echo C:\\Program Files\\Git\\bin>>"%GITHUB_PATH%"', source)
        self.assertNotIn("tier_quality_against_the_f32_reference", source)
        self.assertNotIn("registered_loader_generates_a_song_with_every_artifact", source)
        self.assertNotIn("SIGKILL", source)

    def test_cuda_diagnostic_is_provenance_guarded_and_cannot_launch_proof(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        job = workflow.split("  cuda_diagnostic:\n", 1)[1].split("  reference:\n", 1)[0]
        probe = (ROOT / "scripts/ci/yue2_cuda_context_diagnostic.ps1").read_text(encoding="utf-8")
        self.assertIn("if: inputs.stage == 'cuda-diagnostic'", job)
        self.assertIn("group: inference-real-weights-physical-host", workflow)
        self.assertIn("$env:GITHUB_SHA -cne $env:EXPECTED_CONTROL_SHA", job)
        self.assertIn("(git -C ../engine rev-parse HEAD).Trim() -cne $env:EXPECTED_ENGINE_SHA", job)
        self.assertIn("diagnostic_pid must be a positive decimal PID", job)
        self.assertIn("if: always()", job)
        self.assertNotIn("cargo ", job)
        self.assertNotIn("download-artifact", job)
        self.assertNotIn("yue2_precision_proof.py run", job)
        for required in ("cuDeviceGetLuid", "cuDeviceGetPCIBusId", "Get-Counter",
                         "Get-AuthenticodeSignature", "process-before", "process-after",
                         "compute-apps-$gpu-$i", "pmon-$gpu-$i", "driverInitializationOnly"):
            self.assertIn(required, probe)
        for forbidden in ("extern int cuCtxCreate", "extern int cuDevicePrimaryCtxRetain",
                          "extern int cudaMalloc", "Start-Process", "Stop-Process"):
            self.assertNotIn(forbidden, probe)


if __name__ == "__main__":
    unittest.main()
