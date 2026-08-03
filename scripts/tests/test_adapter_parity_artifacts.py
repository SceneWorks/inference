import copy
import hashlib
import importlib.util
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS = ROOT / "crates/media/mlx-gen/tools"
sys.path.insert(0, str(TOOLS))
import _adapter_parity_provenance as PROVENANCE
import record_adapter_parity_transcript as RECORDER

SPEC = importlib.util.spec_from_file_location(
    "verify_adapter_parity_artifacts",
    TOOLS / "verify_adapter_parity_artifacts.py",
)
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


class AdapterParityArtifactProvenanceTest(unittest.TestCase):
    def write_safetensors(self, path, metadata):
        header = json.dumps(
            {
                "__metadata__": metadata,
                "fixture": {
                    "dtype": "U8",
                    "shape": [1],
                    "data_offsets": [0, 1],
                },
            },
            separators=(",", ":"),
        ).encode("utf-8")
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0")

    def frozen_repo(self, directory):
        root = pathlib.Path(directory)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(
            ["git", "-C", str(root), "remote", "add", "origin", "git@github.com:example/reference.git"],
            check=True,
        )
        source = root / "src/reference.py"
        source.parent.mkdir()
        source.write_text("VALUE = 1\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.name=Codex Test",
                "-c",
                "user.email=codex@example.invalid",
                "commit",
                "-q",
                "--no-gpg-sign",
                "-m",
                "fixture",
            ],
            check=True,
        )
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        return root, source, revision

    def valid_manifest(self):
        manifest = copy.deepcopy(VERIFY.load_manifest())
        manifest["results"]["hyper_flux_scale_zero"]["byte_differences"] = 0
        for name, result in manifest["results"]["fork_parity"].items():
            result["samples_gt8"] = 0
            result["base_floor"] = 1
            result["cap"] = 3
            result["rgb_samples"] = 200
            if name.startswith("z_image_"):
                result["residual_samples_gt8"] = 0
                result["zero_residual_samples_gt8"] = 2
                result["residual_cap"] = 1
            else:
                result["effect_gate"].update(
                    {
                        "status": "locked",
                        "effect_samples_gt8": 2,
                        "minimum_samples_gt8": 1,
                    }
                )
        evidence = manifest["results"]["evidence"]
        evidence["status"] = "verified"
        evidence.pop("pending_reason", None)
        evidence["transcript"]["bytes"] = 1
        evidence["transcript"]["sha256"] = "1" * 64
        evidence["receipt"]["bytes"] = 1
        evidence["receipt"]["sha256"] = "2" * 64
        for index, (name, record) in enumerate(manifest["artifacts"].items(), start=1):
            if record["bytes"] < 1:
                record["bytes"] = index
            if len(record["sha256"]) != 64:
                record["sha256"] = f"{index:064x}"
            evidence["artifact_sha256"][name] = record["sha256"]
        return manifest

    def test_tracked_manifest_and_script_hashes_are_locked(self):
        VERIFY.validate_manifest(self.valid_manifest(), TOOLS)

    def test_missing_artifact_is_rejected(self):
        manifest = self.valid_manifest()
        manifest["artifacts"].pop("qwen_lokr_golden")
        with self.assertRaisesRegex(VERIFY.InvalidManifest, "inventory"):
            VERIFY.validate_manifest(manifest, TOOLS)

    def test_wrong_reference_revision_is_rejected(self):
        manifest = self.valid_manifest()
        manifest["reference"]["revision"] = "0" * 40
        with self.assertRaisesRegex(VERIFY.InvalidManifest, "mflux revision"):
            VERIFY.validate_manifest(manifest, TOOLS)

    def test_nonzero_hyper_scale_zero_result_is_rejected(self):
        manifest = self.valid_manifest()
        manifest["results"]["hyper_flux_scale_zero"]["byte_differences"] = 1
        with self.assertRaisesRegex(VERIFY.InvalidManifest, "not bit-exact"):
            VERIFY.validate_manifest(manifest, TOOLS)

    def test_provider_specific_adapter_gates_reject_inert_mutations(self):
        manifest = self.valid_manifest()
        z_result = manifest["results"]["fork_parity"]["z_image_lokr"]
        z_result["zero_residual_samples_gt8"] = z_result["residual_cap"]
        with self.assertRaisesRegex(VERIFY.InvalidManifest, "residual mutation gate"):
            VERIFY.validate_manifest(manifest, TOOLS)

        manifest = self.valid_manifest()
        q_result = manifest["results"]["fork_parity"]["qwen_lokr"]
        q_result["effect_gate"]["effect_samples_gt8"] = 0
        with self.assertRaisesRegex(VERIFY.InvalidManifest, "invalid measured effect"):
            VERIFY.validate_manifest(manifest, TOOLS)

        manifest = self.valid_manifest()
        q_result = manifest["results"]["fork_parity"]["qwen_lora"]
        q_result["cap"] -= 1
        with self.assertRaisesRegex(VERIFY.InvalidManifest, "floor-relative formula"):
            VERIFY.validate_manifest(manifest, TOOLS)

    def test_pending_qwen_effect_gate_cannot_fabricate_measurement(self):
        manifest = copy.deepcopy(VERIFY.load_manifest())
        evidence = manifest["results"]["evidence"]
        evidence["status"] = "diagnostic_pending"
        evidence["pending_reason"] = "test fixture awaits effect diagnostic"
        for record in (evidence["transcript"], evidence["receipt"]):
            record["bytes"] = -1
            record["sha256"] = "PENDING"
        for result in manifest["results"]["fork_parity"].values():
            if not result.get("effect_gate"):
                continue
            gate = result["effect_gate"]
            gate["status"] = "diagnostic_pending"
            gate.pop("effect_samples_gt8")
            gate.pop("minimum_samples_gt8")
        VERIFY.validate_manifest(manifest, TOOLS)
        gate = manifest["results"]["fork_parity"]["qwen_lora"]["effect_gate"]
        gate["effect_samples_gt8"] = 1
        with self.assertRaisesRegex(VERIFY.InvalidManifest, "fabricates evidence"):
            VERIFY.validate_manifest(manifest, TOOLS)

    def test_changed_dump_script_is_rejected(self):
        manifest = self.valid_manifest()
        manifest["scripts"]["dump_qwen_adapter_golden.py"] = "0" * 64
        with self.assertRaisesRegex(VERIFY.InvalidManifest, "script hash mismatch"):
            VERIFY.validate_manifest(manifest, TOOLS)

    def test_changed_gitignored_artifact_is_rejected(self):
        manifest = self.valid_manifest()
        with tempfile.TemporaryDirectory() as directory:
            artifact = pathlib.Path(directory) / "artifact.safetensors"
            payload = b"reference"
            artifact.write_bytes(payload)
            for record in manifest["artifacts"].values():
                record["local_path"] = str(artifact)
                record["bytes"] = len(payload)
                record["sha256"] = hashlib.sha256(payload).hexdigest()
            VERIFY.verify_artifact_files(manifest, TOOLS)

            artifact.write_bytes(b"mutati0n!")
            with self.assertRaisesRegex(VERIFY.InvalidManifest, "sha256 mismatch"):
                VERIFY.verify_artifact_files(manifest, TOOLS)

    def test_frozen_reference_rejects_tracked_and_untracked_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source, revision = self.frozen_repo(directory)
            kwargs = {
                "expected_revision": revision,
                "expected_identity": "github.com/example/reference",
            }
            PROVENANCE.assert_frozen_repository(root, **kwargs)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "tracked/index state"):
                PROVENANCE.assert_frozen_repository(root, **kwargs)
            subprocess.run(["git", "-C", str(root), "restore", "src/reference.py"], check=True)
            (root / "src/untracked.py").write_text("VALUE = 3\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "untracked source"):
                PROVENANCE.assert_frozen_repository(root, **kwargs)

    def test_proof_source_digest_is_stable_when_untracked_file_is_staged(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, revision = self.frozen_repo(directory)
            proof = root / "src/proof.py"
            proof.write_text("PROOF = 1\n", encoding="utf-8")
            manifest = {
                "implementation_base": revision,
                "scripts": {},
            }
            kwargs = {
                "root": root,
                "source_files": ("src/proof.py",),
                "permitted_changes": {"src/proof.py"},
            }
            before = VERIFY.source_state(manifest, **kwargs)
            subprocess.run(["git", "-C", str(root), "add", "src/proof.py"], check=True)
            after = VERIFY.source_state(manifest, **kwargs)
            self.assertEqual(before, after)

            (root / "src/outside.rs").write_text("fn changed() {}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "outside the bound allowlist"):
                VERIFY.source_state(manifest, **kwargs)

    def test_residual_diagnostic_uses_exact_sanitized_runs(self):
        runs = RECORDER.residual_diagnostic_runs(self.valid_manifest())
        self.assertEqual(
            [run["name"] for run in runs],
            ["z_image_residual_diagnostic", "qwen_residual_diagnostic"],
        )
        expected_base = RECORDER.proof_environment()
        for run in runs:
            self.assertIn("--exact", run["argv"])
            self.assertIn("residual_mutation_diagnostic", run["argv"])
            self.assertEqual(
                {key: run["env"][key] for key in expected_base},
                expected_base,
            )
            self.assertNotIn("GH_TOKEN", run["env"])
        effect_runs = RECORDER.qwen_effect_diagnostic_runs(self.valid_manifest())
        self.assertEqual([run["name"] for run in effect_runs], ["qwen_effect_diagnostic"])
        self.assertIn("--exact", effect_runs[0]["argv"])
        self.assertIn("adapter_effect_diagnostic", effect_runs[0]["argv"])
        self.assertEqual(
            {key: effect_runs[0]["env"][key] for key in expected_base},
            expected_base,
        )

    def test_diagnostics_refuse_reserved_output_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            receipt = root / "adapter_parity_receipt.json"
            acceptance = root / "sc-15505-real-weight-transcript.json"
            diagnostic = root / "sc-15505-residual-diagnostic-transcript.json"
            effect = root / "sc-15505-qwen-effect-diagnostic-transcript.json"
            receipt.write_text("do not replace\n", encoding="utf-8")

            for flag, reserved in (
                ("--residual-diagnostic", receipt),
                ("--qwen-effect-diagnostic", acceptance),
            ):
                with (
                    mock.patch.object(RECORDER, "RECEIPT", receipt),
                    mock.patch.object(RECORDER, "ACCEPTANCE_TRANSCRIPT", acceptance),
                    mock.patch.object(RECORDER, "DIAGNOSTIC_TRANSCRIPT", diagnostic),
                    mock.patch.object(RECORDER, "QWEN_EFFECT_TRANSCRIPT", effect),
                    mock.patch.object(
                        sys,
                        "argv",
                        [
                            "record_adapter_parity_transcript.py",
                            flag,
                            "--output",
                            str(reserved),
                        ],
                    ),
                    self.assertRaises(SystemExit) as raised,
                ):
                    RECORDER.main()
                self.assertEqual(raised.exception.code, 2)
            self.assertEqual(receipt.read_text(encoding="utf-8"), "do not replace\n")
            self.assertFalse(acceptance.exists())
            self.assertFalse(diagnostic.exists())
            self.assertFalse(effect.exists())

    def test_residual_results_are_bound_to_their_run_and_exact_fields(self):
        z_results = {
            name: {
                "residual_samples_gt8": 10,
                "zero_residual_samples_gt8": 20,
                "rgb_samples": 100,
            }
            for name in ("z_image_lora", "z_image_lokr")
        }
        RECORDER.validate_residual_run_results("z_image_residual_diagnostic", z_results)

        wrong_fields = copy.deepcopy(z_results)
        wrong_fields["z_image_lora"]["base_floor"] = 1
        with self.assertRaisesRegex(ValueError, "field inventory mismatch"):
            RECORDER.validate_residual_run_results(
                "z_image_residual_diagnostic",
                wrong_fields,
            )

        cross_run = copy.deepcopy(z_results)
        cross_run["qwen_lora"] = cross_run.pop("z_image_lora")
        with self.assertRaisesRegex(ValueError, "result inventory mismatch"):
            RECORDER.validate_residual_run_results(
                "z_image_residual_diagnostic",
                cross_run,
            )

        inconsistent = {
            **z_results,
            "qwen_lora": {
                "residual_samples_gt8": 10,
                "zero_residual_samples_gt8": 20,
                "rgb_samples": 101,
            },
        }
        with self.assertRaisesRegex(ValueError, "one nonzero RGB sample count"):
            RECORDER.validate_shared_residual_sample_count(inconsistent)

    def test_residual_results_start_on_lines_after_cargo_test_prefix(self):
        output = (
            "test residual_mutation_diagnostic ... \n"
            "SC15505_RESULT z_image_lora residual_samples_gt8=10 "
            "zero_residual_samples_gt8=20 rgb_samples=100\n"
            "SC15505_RESULT z_image_lokr residual_samples_gt8=11 "
            "zero_residual_samples_gt8=21 rgb_samples=100\n"
            "ok\n"
        )
        parsed = RECORDER.parsed_results(output)
        RECORDER.validate_residual_run_results("z_image_residual_diagnostic", parsed)

    def test_qwen_effect_results_bind_structure_and_exact_fields(self):
        results = {
            "qwen_lora": {
                "effect_samples_gt8": 100,
                "scale_zero_byte_differences": 0,
                "applied": 24,
                "unmatched": 0,
                "rgb_samples": 1_000,
            },
            "qwen_lokr": {
                "effect_samples_gt8": 200,
                "scale_zero_byte_differences": 0,
                "applied": 21,
                "unmatched": 0,
                "rgb_samples": 1_000,
            },
        }
        RECORDER.validate_qwen_effect_results(results)

        wrong_fields = copy.deepcopy(results)
        wrong_fields["qwen_lora"]["residual_samples_gt8"] = 1
        with self.assertRaisesRegex(ValueError, "field inventory mismatch"):
            RECORDER.validate_qwen_effect_results(wrong_fields)

        wrong_applied = copy.deepcopy(results)
        wrong_applied["qwen_lokr"]["applied"] = 20
        with self.assertRaisesRegex(ValueError, "applied module count"):
            RECORDER.validate_qwen_effect_results(wrong_applied)

        unmatched = copy.deepcopy(results)
        unmatched["qwen_lora"]["unmatched"] = 1
        with self.assertRaisesRegex(ValueError, "unmatched adapter paths"):
            RECORDER.validate_qwen_effect_results(unmatched)

        dropped = copy.deepcopy(results)
        dropped["qwen_lora"]["effect_samples_gt8"] = 0
        with self.assertRaisesRegex(ValueError, "invalid RGB/effect sample count"):
            RECORDER.validate_qwen_effect_results(dropped)

    def test_base_adapter_metadata_binds_prompt_guidance_and_reference(self):
        z_base = {field: f"value:{field}" for field in VERIFY.Z_IMAGE_BEHAVIOR_FIELDS}
        z_adapter = dict(z_base)
        VERIFY.verify_matching_generation_metadata(
            z_base,
            z_adapter,
            VERIFY.Z_IMAGE_BEHAVIOR_FIELDS,
            "z_image_lora",
        )
        z_adapter["prompt"] = "different prompt"
        with self.assertRaisesRegex(VERIFY.InvalidManifest, "prompt"):
            VERIFY.verify_matching_generation_metadata(
                z_base,
                z_adapter,
                VERIFY.Z_IMAGE_BEHAVIOR_FIELDS,
                "z_image_lora",
            )

        qwen_base = {field: f"value:{field}" for field in VERIFY.QWEN_BEHAVIOR_FIELDS}
        qwen_adapter = dict(qwen_base)
        qwen_adapter["guidance"] = "different guidance"
        with self.assertRaisesRegex(VERIFY.InvalidManifest, "guidance"):
            VERIFY.verify_matching_generation_metadata(
                qwen_base,
                qwen_adapter,
                VERIFY.QWEN_BEHAVIOR_FIELDS,
                "qwen_lora",
            )

    def test_snapshot_claim_and_inventory_are_content_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            revision = "a" * 40
            snapshot = (
                pathlib.Path(directory)
                / "models--example--model"
                / "snapshots"
                / revision
                / "bf16"
            )
            snapshot.mkdir(parents=True)
            refs = snapshot.parents[2] / "refs"
            refs.mkdir()
            (refs / "main").write_text(revision, encoding="utf-8")
            model = snapshot / "model.safetensors"
            model.write_bytes(b"weights1")
            _, before = PROVENANCE.assert_hf_snapshot(
                snapshot,
                repository="example/model",
                revision=revision,
                subdirectory="bf16",
            )
            with self.assertRaisesRegex(RuntimeError, "claimed repository/revision"):
                PROVENANCE.assert_hf_snapshot(
                    snapshot,
                    repository="example/model",
                    revision="b" * 40,
                    subdirectory="bf16",
                )
            model.write_bytes(b"weights2")
            _, after = PROVENANCE.assert_hf_snapshot(
                snapshot,
                repository="example/model",
                revision=revision,
                subdirectory="bf16",
            )
            self.assertNotEqual(before, after)

    def test_snapshot_inventory_hashes_dereferenced_blob_content(self):
        with tempfile.TemporaryDirectory() as directory:
            revision = "c" * 40
            repository = pathlib.Path(directory) / "models--example--linked"
            snapshot = repository / "snapshots" / revision / "bf16"
            blob = repository / "blobs" / ("d" * 64)
            snapshot.mkdir(parents=True)
            blob.parent.mkdir()
            blob.write_bytes(b"weights1")
            (snapshot / "model.safetensors").symlink_to(
                pathlib.Path("../../../blobs") / blob.name
            )
            refs = repository / "refs"
            refs.mkdir()
            (refs / "main").write_text(revision, encoding="utf-8")
            _, before = PROVENANCE.assert_hf_snapshot(
                snapshot,
                repository="example/linked",
                revision=revision,
                subdirectory="bf16",
            )
            blob.write_bytes(b"weights2")
            _, after = PROVENANCE.assert_hf_snapshot(
                snapshot,
                repository="example/linked",
                revision=revision,
                subdirectory="bf16",
            )
            self.assertNotEqual(before, after)

    def test_generated_metadata_is_bound_to_manifest(self):
        manifest = self.valid_manifest()
        name = "z_image_lora_adapter"
        record = manifest["artifacts"][name]
        source = record["source"]
        metadata = {
            "artifact_role": "adapter",
            "adapter_kind": "lora",
            "reference_mflux_repository": manifest["reference"]["repository"],
            "reference_mflux_revision": manifest["reference"]["revision"],
            "reference_script_sha256": manifest["scripts"][source["script"]],
            "reference_provenance_sha256": manifest["scripts"][
                "_adapter_parity_provenance.py"
            ],
            "reference_model_repository": source["model_repository"],
            "reference_model_revision": source["model_revision"],
            "reference_model_ref": source["model_reference"],
            "reference_model_subdirectory": source["model_subdirectory"],
            "reference_model_path": source["model_path"],
            "reference_model_inventory_sha256": source["model_inventory_sha256"],
            "reference_runtime": json.dumps(manifest["reference"]["runtime"]),
        }
        with tempfile.TemporaryDirectory() as directory:
            fixture = pathlib.Path(directory) / "fixture.safetensors"
            self.write_safetensors(fixture, metadata)
            VERIFY.verify_generated_metadata(manifest, name, record, fixture)
            metadata["reference_model_revision"] = "0" * 40
            self.write_safetensors(fixture, metadata)
            with self.assertRaisesRegex(VERIFY.InvalidManifest, "model revision mismatch"):
                VERIFY.verify_generated_metadata(manifest, name, record, fixture)
            metadata["reference_model_revision"] = source["model_revision"]
            metadata["reference_model_path"] = str(
                pathlib.PurePosixPath(source["model_path"]).parent / ("0" * 40)
            )
            self.write_safetensors(fixture, metadata)
            with self.assertRaisesRegex(VERIFY.InvalidManifest, "model path mismatch"):
                VERIFY.verify_generated_metadata(manifest, name, record, fixture)

    def test_generated_model_path_is_compared_without_resolving_it_locally(self):
        # Parity goldens are dumped on one host and the manifest records that host's
        # absolute path (see crates/media/mlx-gen/tools/golden/README.md). Verification
        # asks whether a golden was dumped against the model directory the manifest
        # names, never whether that path exists here — so it must not re-root the
        # recorded path onto the verifying host. Exercise both a POSIX-shaped and a
        # Windows-shaped recording so the comparison stays an identity on either OS.
        for recorded in (
            "/Users/reference/.cache/huggingface/hub/models--example--model/snapshots/"
            + "a" * 40,
            "E:\\goldens\\hub\\models--example--model\\snapshots\\" + "a" * 40,
        ):
            with self.subTest(recorded=recorded):
                manifest = self.valid_manifest()
                name = "z_image_lora_adapter"
                record = manifest["artifacts"][name]
                source = record["source"]
                source["model_path"] = recorded
                metadata = {
                    "artifact_role": "adapter",
                    "adapter_kind": "lora",
                    "reference_mflux_repository": manifest["reference"]["repository"],
                    "reference_mflux_revision": manifest["reference"]["revision"],
                    "reference_script_sha256": manifest["scripts"][source["script"]],
                    "reference_provenance_sha256": manifest["scripts"][
                        "_adapter_parity_provenance.py"
                    ],
                    "reference_model_repository": source["model_repository"],
                    "reference_model_revision": source["model_revision"],
                    "reference_model_ref": source["model_reference"],
                    "reference_model_subdirectory": source["model_subdirectory"],
                    "reference_model_path": recorded,
                    "reference_model_inventory_sha256": source["model_inventory_sha256"],
                    "reference_runtime": json.dumps(manifest["reference"]["runtime"]),
                }
                with tempfile.TemporaryDirectory() as directory:
                    fixture = pathlib.Path(directory) / "fixture.safetensors"
                    self.write_safetensors(fixture, metadata)
                    VERIFY.verify_generated_metadata(manifest, name, record, fixture)

    def test_transcript_measurements_are_bound_to_artifact_hashes(self):
        manifest = self.valid_manifest()
        results = manifest["results"]
        specs = VERIFY.expected_runs(manifest)
        result_lines = {
            "hyper_flux_scale_zero": (
                "SC15505_RESULT hyper_flux_scale_zero "
                "byte_differences=0 rgb_samples=786432"
            ),
            "z_image_lora": (
                "SC15505_RESULT z_image_lora samples_gt8=0 base_floor=1 cap=3 "
                "residual_samples_gt8=0 zero_residual_samples_gt8=2 residual_cap=1 "
                "rgb_samples=200"
            ),
            "z_image_lokr": (
                "SC15505_RESULT z_image_lokr samples_gt8=0 base_floor=1 cap=3 "
                "residual_samples_gt8=0 zero_residual_samples_gt8=2 residual_cap=1 "
                "rgb_samples=200"
            ),
            "qwen_lora": (
                "SC15505_RESULT qwen_lora samples_gt8=0 base_floor=1 cap=3 "
                "effect_samples_gt8=2 minimum_samples_gt8=1 "
                "scale_zero_byte_differences=0 applied=24 unmatched=0 rgb_samples=200"
            ),
            "qwen_lokr": (
                "SC15505_RESULT qwen_lokr samples_gt8=0 base_floor=1 cap=3 "
                "effect_samples_gt8=2 minimum_samples_gt8=1 "
                "scale_zero_byte_differences=0 applied=21 unmatched=0 rgb_samples=200"
            ),
        }
        runs = [
            {
                **spec,
                "returncode": 0,
                "stdout": result_lines[spec["name"]] + "\ntest result: ok.\n",
                "stderr": "",
            }
            for spec in specs
        ]
        source = {
            "commit": manifest["implementation_base"],
            "base_commit": manifest["implementation_base"],
            "source_sha256": "d" * 64,
            "files": {"fixture": "e" * 64},
            "changed_paths": ["fixture"],
        }
        execution = {"fixture": "execution"}
        models = {"fixture": {"inventory_sha256": "f" * 64}}
        parsed = {
            name: VERIFY.parsed_results(line)[name]
            for name, line in result_lines.items()
        }
        transcript = {
            "schema": 2,
            "story": "sc-15505",
            "source": source,
            "source_after": source,
            "execution": execution,
            "execution_after": execution,
            "models_before": models,
            "models_after": models,
            "artifacts_before": results["evidence"]["artifact_sha256"],
            "artifacts_after": results["evidence"]["artifact_sha256"],
            "runs": runs,
            "results": parsed,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "transcript.json"

            def write_transcript():
                path.write_text(json.dumps(transcript), encoding="utf-8")
                record = results["evidence"]["transcript"]
                record["local_path"] = str(path)
                record["bytes"] = path.stat().st_size
                record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                receipt_path = pathlib.Path(directory) / "receipt.json"
                receipt_path.write_text(
                    json.dumps(VERIFY.receipt_for(transcript, path)),
                    encoding="utf-8",
                )
                receipt_record = results["evidence"]["receipt"]
                receipt_record["local_path"] = str(receipt_path)
                receipt_record["bytes"] = receipt_path.stat().st_size
                receipt_record["sha256"] = hashlib.sha256(
                    receipt_path.read_bytes()
                ).hexdigest()

            write_transcript()
            with (
                mock.patch.object(VERIFY, "model_inventories", return_value=models),
                mock.patch.object(VERIFY, "source_state", return_value=source),
                mock.patch.object(VERIFY, "execution_metadata", return_value=execution),
            ):
                VERIFY.verify_result_transcript(manifest, TOOLS)
                runs[1]["stdout"] = runs[1]["stdout"].replace(
                    "z_image_lora samples_gt8=0",
                    "z_image_lora samples_gt8=1",
                )
                transcript["results"]["z_image_lora"]["samples_gt8"] = 1
                write_transcript()
                with self.assertRaisesRegex(VERIFY.InvalidManifest, "measurements mismatch"):
                    VERIFY.verify_result_transcript(manifest, TOOLS)
                runs[1]["stdout"] = runs[1]["stdout"].replace(
                    "z_image_lora samples_gt8=1",
                    "z_image_lora samples_gt8=0",
                )
                runs[1]["stdout"] += result_lines["z_image_lora"] + "\n"
                transcript["results"]["z_image_lora"]["samples_gt8"] = 0
                write_transcript()
                with self.assertRaisesRegex(VERIFY.InvalidManifest, "duplicate result name"):
                    VERIFY.verify_result_transcript(manifest, TOOLS)

    def test_transcript_rejects_duplicate_result_lines_and_fields(self):
        line = (
            "SC15505_RESULT z_image_lora samples_gt8=0 no_adapter_samples_gt8=2 "
            "base_floor=0 cap=1 rgb_samples=786432\n"
        )
        with self.assertRaisesRegex(ValueError, "duplicate result name"):
            VERIFY.parsed_results(line + line)
        duplicate_field = (
            "SC15505_RESULT z_image_lora samples_gt8=0 samples_gt8=1 "
            "no_adapter_samples_gt8=2 base_floor=0 cap=1 rgb_samples=786432\n"
        )
        with self.assertRaisesRegex(ValueError, "duplicate result field"):
            VERIFY.parsed_results(duplicate_field)


if __name__ == "__main__":
    unittest.main()
