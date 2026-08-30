"""Static guards for the one-run SC-22261 StarVector admission workflow."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/real-weights.yml"
MODELS = ROOT / "release/real-weight-models.toml"
CORPUS = ROOT / "release/starvector-terminal-corpus-v1.json"
SCHEMA = ROOT / "release/starvector-terminal-receipt-v1.schema.json"
HARNESS = ROOT / "scripts/release/starvector_terminal_evidence.mjs"
PREFLIGHT_ASSEMBLER = ROOT / "scripts/release/starvector_terminal_preflight.mjs"


def terminal_workflow_errors(workflow: str) -> list[str]:
    """Return the terminal-lane omissions that must never silently join all/scheduled sweeps."""
    errors = []
    if "starvector-terminal" not in workflow.split("options:", 1)[1].split("sceneworks_revision:", 1)[0]:
        errors.append("dispatcher profile missing")
    start = workflow.find("  starvector-terminal-mlx:")
    end = workflow.find("\n  starvector-terminal-candle:", start)
    mlx = workflow[start:end]
    if "github.event_name == 'workflow_dispatch'" not in mlx or "inputs.profile == 'starvector-terminal'" not in mlx:
        errors.append("MLX terminal lane is not dispatch-only")
    if "runs-on: [self-hosted, macOS, ARM64, real-weights]" not in mlx:
        errors.append("MLX terminal lane does not target the provisioned real-weights host")
    for snapshot in (
        "/Users/Shared/SceneWorks/starvector-terminal/weights/models/starvector-1b",
        "/Users/Shared/SceneWorks/starvector-terminal/weights/models/starvector-8b",
    ):
        if snapshot not in mlx:
            errors.append(f"MLX terminal snapshot path missing {snapshot}")
    for name in (
        "starvector_1b::tests::real_weight_provider_satisfies_shared_starvector_conformance",
        "starvector_8b::tests::real_weight_provider_satisfies_shared_starvector_conformance",
        "verify_model_snapshot.py --model starvector-1b-im2svg",
        "verify_model_snapshot.py --model starvector-8b-im2svg",
    ):
        if name not in mlx:
            errors.append(f"MLX terminal command missing {name}")
    if mlx.count("--exact --ignored --nocapture") != 2:
        errors.append("MLX terminal command missing exact filters")
    for artifact in (
        "inventory/starvector-1b-inventory.json",
        "inventory/starvector-8b-inventory.json",
        "hooks/mlx-starvector-1b.log",
        "hooks/mlx-starvector-8b.log",
        "github.run_id",
        "github.run_attempt",
    ):
        if artifact not in mlx:
            errors.append(f"MLX terminal provenance missing {artifact}")
    candle_start = workflow.find("  starvector-terminal-candle:")
    candle_end = workflow.find("\n  mlx-llm:", candle_start)
    candle = workflow[candle_start:candle_end]
    if "needs: starvector-terminal-mlx" not in candle:
        errors.append("Candle lane no longer serializes after MLX")
    if "runs-on: [self-hosted, windows, cuda, real-weights]" not in candle:
        errors.append("Candle lane no longer uses authoritative CUDA")
    for snapshot in (
        r"D:\\sceneworks-terminal\\weights\\models\\starvector-1b",
        r"D:\\sceneworks-terminal\\weights\\models\\starvector-8b",
    ):
        if snapshot not in candle:
            errors.append(f"Candle terminal snapshot path missing {snapshot}")
    for name in (
        "starvector::tests::real_weight_provider_satisfies_shared_starvector_conformance",
        "starvector_8b::tests::real_weight_provider_satisfies_shared_starvector_conformance",
        "starvector-terminal-mlx-${{ github.sha }}",
    ):
        if name not in candle:
            errors.append(f"Candle terminal command missing {name}")
    if candle.count("--exact --ignored --nocapture") != 2:
        errors.append("Candle terminal command missing exact filters")
    for artifact in (
        "candle-cuda-starvector-1b.log",
        "candle-cuda-starvector-8b.log",
        "starvector_terminal_preflight.mjs assemble",
        '--head-sha "%GITHUB_SHA%"',
        '--workflow-run-id "%GITHUB_RUN_ID%"',
        '--workflow-run-attempt "%GITHUB_RUN_ATTEMPT%"',
        "starvector-terminal-preflight-${{ github.sha }}-${{ github.run_id }}-${{ github.run_attempt }}",
        "Upload complete terminal preflight provenance",
    ):
        if artifact not in candle:
            errors.append(f"Candle terminal provenance missing {artifact}")
    return errors


class StarVectorTerminalPolicyTests(unittest.TestCase):
    def test_exact_native_model_rows_are_wired_to_terminal_profile(self) -> None:
        models = {
            model["key"]: model
            for model in tomllib.loads(MODELS.read_text(encoding="utf-8"))["models"]
        }
        expected = {
            "starvector-1b-im2svg": (
                "starvector/starvector-1b-im2svg",
                "380ab95d25a8e9ab1dc825debe238b4953ae13b9",
                "STARVECTOR_1B_SNAPSHOT",
                12,
            ),
            "starvector-8b-im2svg": (
                "starvector/starvector-8b-im2svg",
                "518beea8dcb5f7a37c5911e92d1d62a76beee7f9",
                "STARVECTOR_8B_SNAPSHOT",
                13,
            ),
        }
        for key, (repository, revision, environment, file_count) in expected.items():
            with self.subTest(key=key):
                model = models[key]
                self.assertEqual(model["repository"], repository)
                self.assertEqual(model["revision"], revision)
                self.assertEqual(model["license"], "Apache-2.0")
                self.assertEqual(model["profiles"], ["starvector-terminal"])
                self.assertEqual(model["environment"], [environment])
                self.assertEqual(len(model["expected_files"]), file_count)

    def test_terminal_workflow_is_serial_dispatch_only_and_exact_name_selected(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertEqual(terminal_workflow_errors(workflow), [])
        self.assertNotIn("inputs.profile == 'starvector-terminal'", workflow[workflow.index("  mlx-llm:"):])
        self.assertNotIn("inputs.profile == 'starvector-terminal'", workflow[workflow.index("  candle-llm:"):])

    def test_terminal_workflow_policy_detects_dispatch_serial_and_exact_command_mutations(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        mlx_start = workflow.index("  starvector-terminal-mlx:")
        candle_start = workflow.index("  starvector-terminal-candle:")
        candle_end = workflow.index("\n  mlx-llm:", candle_start)

        def mutate(start: int, end: int, old: str) -> str:
            return workflow[:start] + workflow[start:end].replace(old, "MUTATED", 1) + workflow[end:]

        cases = (
            (mutate(candle_start, candle_end, "needs: starvector-terminal-mlx"), "serializes"),
            (mutate(mlx_start, candle_start, "--exact --ignored --nocapture"), "MLX terminal command missing exact filters"),
            (mutate(mlx_start, candle_start, "inputs.profile == 'starvector-terminal'"), "MLX terminal lane is not dispatch-only"),
            (mutate(candle_start, candle_end, ' --workflow-run-attempt "%GITHUB_RUN_ATTEMPT%"'), "workflow-run-attempt"),
        )
        for mutated, expected in cases:
            with self.subTest(expected=expected):
                self.assertTrue(any(expected in error for error in terminal_workflow_errors(mutated)))

    def test_preflight_assembler_binds_exact_relative_sources_and_is_deterministic(self) -> None:
        contents = {
            "inventory/starvector-1b-inventory.json": b"one inventory\n",
            "inventory/starvector-8b-inventory.json": b"eight inventory\n",
            "hooks/mlx-starvector-1b.log": b"mlx one\n",
            "hooks/mlx-starvector-8b.log": b"mlx eight\n",
            "hooks/candle-cuda-starvector-1b.log": b"candle one\n",
            "hooks/candle-cuda-starvector-8b.log": b"candle eight\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative, payload in contents.items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            command = [
                "node",
                str(PREFLIGHT_ASSEMBLER),
                "assemble",
                "--root",
                str(root),
                "--head-sha",
                "a" * 40,
                "--workflow-run-id",
                "12345",
                "--workflow-run-attempt",
                "2",
            ]
            subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
            output = root / "starvector-terminal-preflight.json"
            first = output.read_bytes()
            value = json.loads(first)
            self.assertEqual(value["workflow_run_id"], "12345")
            self.assertEqual(value["workflow_run_attempt"], 2)
            self.assertEqual(value["head_sha"], "a" * 40)
            records = value["inventory_artifacts"] + value["hook_logs"]
            self.assertEqual({record["path"] for record in records}, set(contents))
            for record in records:
                self.assertEqual(record["sha256"], hashlib.sha256(contents[record["path"]]).hexdigest())
            subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(output.read_bytes(), first)
            (root / "hooks/candle-cuda-starvector-8b.log").unlink()
            failed = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8")
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("required non-empty regular file is missing", failed.stderr)

    def test_schema_and_corpus_bind_all_required_counts_and_boundary(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
        self.assertEqual(schema["properties"]["runs"]["maxItems"], 4)
        self.assertNotIn("eight_b_uplift", schema["properties"])
        image_quality = schema["$defs"]["image_quality"]["properties"]
        self.assertEqual(image_quality["cases"]["minItems"], 120)
        self.assertEqual(image_quality["cases"]["maxItems"], 120)
        self.assertNotIn("median_ssim", image_quality)
        case = schema["$defs"]["image_case"]["properties"]
        self.assertEqual(case["case_index"]["maximum"], 119)
        self.assertEqual(case["ssim"]["type"], ["number", "null"])
        parity = schema["$defs"]["parity"]["properties"]
        self.assertEqual(parity["case_count"]["const"], 20)
        self.assertEqual(parity["cases"]["minItems"], 20)
        self.assertIn("hostile_sanitizer", schema["properties"])
        self.assertIn("prompt_composition", schema["properties"])
        self.assertIn("artifact_manifest", schema["properties"])
        self.assertEqual(schema["$defs"]["hostile"]["properties"]["cases"]["minItems"], 200)
        self.assertEqual(schema["$defs"]["prompt"]["properties"]["cases"]["minItems"], 60)
        corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
        self.assertEqual(corpus["upstream_image_quality_cases"]["required_count"], 120)
        self.assertEqual(corpus["deterministic_parity_cases"]["required_count_per_backend"], 20)
        self.assertEqual(corpus["sceneworks_owned_suites"]["hostile_sanitizer"]["required_count"], 200)
        self.assertEqual(corpus["sceneworks_owned_suites"]["prompt_composition"]["required_count"], 60)
        self.assertRegex(corpus["sceneworks_owned_suites"]["hostile_sanitizer"]["content_identity_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(corpus["sceneworks_owned_suites"]["prompt_composition"]["content_identity_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(corpus["upstream_image_quality_cases"]["sources"]), 4)

    def test_harness_rejects_a_corpus_count_mutation(self) -> None:
        corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
        corpus["upstream_image_quality_cases"]["sources"][0]["row_count"] = 29
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            path.write_text(json.dumps(corpus), encoding="utf-8")
            result = subprocess.run(
                ["node", str(HARNESS), "validate-plan", "--corpus", str(path)],
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("immutable source identity", result.stderr)


if __name__ == "__main__":
    unittest.main()
