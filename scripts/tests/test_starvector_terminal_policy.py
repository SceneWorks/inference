"""Static guards for the one-run SC-22261 StarVector admission workflow."""

from __future__ import annotations

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
    candle_start = workflow.find("  starvector-terminal-candle:")
    candle_end = workflow.find("\n  mlx-llm:", candle_start)
    candle = workflow[candle_start:candle_end]
    if "needs: starvector-terminal-mlx" not in candle:
        errors.append("Candle lane no longer serializes after MLX")
    if "runs-on: [self-hosted, windows, cuda, real-weights]" not in candle:
        errors.append("Candle lane no longer uses authoritative CUDA")
    for name in (
        "starvector::tests::real_weight_provider_satisfies_shared_starvector_conformance",
        "starvector_8b::tests::real_weight_provider_satisfies_shared_starvector_conformance",
        "starvector-terminal-mlx-${{ github.sha }}",
    ):
        if name not in candle:
            errors.append(f"Candle terminal command missing {name}")
    if candle.count("--exact --ignored --nocapture") != 2:
        errors.append("Candle terminal command missing exact filters")
    return errors


class StarVectorTerminalPolicyTests(unittest.TestCase):
    def test_exact_native_model_rows_are_wired_to_terminal_profile(self) -> None:
        models = {model["key"]: model for model in tomllib.loads(MODELS.read_text())["models"]}
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
        )
        for mutated, expected in cases:
            with self.subTest(expected=expected):
                self.assertTrue(any(expected in error for error in terminal_workflow_errors(mutated)))

    def test_schema_and_corpus_bind_all_required_counts_and_boundary(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
        self.assertEqual(schema["properties"]["runs"]["maxItems"], 4)
        self.assertEqual(schema["$defs"]["uplift"]["properties"]["bootstrap_confidence"]["const"], 0.95)
        self.assertEqual(schema["$defs"]["image_quality"]["properties"]["case_count"]["type"], "integer")
        corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
        self.assertEqual(corpus["upstream_image_quality_cases"]["required_count"], 120)
        self.assertEqual(corpus["deterministic_parity_cases"]["required_count_per_backend"], 20)
        self.assertEqual(corpus["sceneworks_owned_suites"]["hostile_sanitizer"]["required_count"], 200)
        self.assertEqual(corpus["sceneworks_owned_suites"]["prompt_composition"]["required_count"], 60)
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
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exact first thirty", result.stderr)


if __name__ == "__main__":
    unittest.main()
