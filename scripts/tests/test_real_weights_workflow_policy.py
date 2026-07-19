from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "real-weights.yml"
MANIFEST = ROOT / "release" / "real-weight-models.toml"


class RealWeightsWorkflowPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")
        cls.job = cls.text.split("  candle-residency-ab:\n", 1)[1]

    def test_residency_gate_is_explicitly_dispatched_only(self):
        self.assertIn("options: [all, llm, media, audio, residency-ab]", self.text)
        self.assertIn(
            "if: github.event_name == 'workflow_dispatch' && inputs.profile == 'residency-ab'",
            self.job,
        )
        condition = self.job.splitlines()[4]
        self.assertNotIn("schedule", condition)
        self.assertNotIn("profile == 'all'", condition)
        self.assertIn("runs-on: [self-hosted, windows, cuda, real-weights]", self.job)

    def test_immutable_snapshots_are_repository_variables(self):
        self.assertIn("QWEN_IMAGE_SNAPSHOT: ${{ vars.QWEN_IMAGE_SNAPSHOT }}", self.job)
        self.assertIn("FLUX_DEV_DIR: ${{ vars.FLUX_DEV_DIR }}", self.job)
        self.assertNotIn("secrets.", self.job)
        self.assertIn("ensure_model_snapshot.py --model qwen-image --snapshot \"%QWEN_IMAGE_SNAPSHOT%\"", self.job)
        self.assertIn("verify_model_snapshot.py --model flux-1-dev --snapshot \"%FLUX_DEV_DIR%\"", self.job)
        self.assertNotIn("ensure_model_snapshot.py --model flux-1-dev", self.job)

        models = {
            model["key"]: model
            for model in tomllib.loads(MANIFEST.read_text(encoding="utf-8"))["models"]
        }
        self.assertEqual(
            models["qwen-image"]["revision"],
            "75e0b4be04f60ec59a75f475837eced720f823b6",
        )
        self.assertEqual(
            models["flux-1-dev"]["revision"],
            "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
        )
        self.assertIn("tokenizer/tokenizer.json", models["qwen-image"]["expected_files"])
        self.assertIn("tokenizer_2/tokenizer.json", models["flux-1-dev"]["expected_files"])

    def test_each_model_runs_resident_and_sequential_in_separate_processes(self):
        qwen_command = "cargo test --locked -p candle-gen-qwen-image --features cuda qwen_image_probed_generate_for_offload_ab"
        flux_command = "cargo test --locked -p candle-gen-flux --features cuda flux_dev_probed_generate_for_offload_ab"
        self.assertEqual(self.job.count(qwen_command), 2)
        self.assertEqual(self.job.count(flux_command), 2)
        self.assertIn('set "QWEN_OFFLOAD_MODE=spec-sequential"', self.job)
        self.assertIn('set "FLUX_OFFLOAD_MODE=spec-sequential"', self.job)
        self.assertIn('set "CANDLE_GEN_OFFLOAD="', self.job)
        self.assertEqual(self.job.count("verify_residency_ab.py"), 2)
        self.assertEqual(self.job.count("--min-reduction-mib 512"), 2)

    def test_raw_outputs_are_compared_and_always_uploaded_with_logs(self):
        self.assertEqual(self.job.count("fc /b"), 2)
        for name in (
            "qwen-resident.rgb",
            "qwen-sequential.rgb",
            "flux-dev-resident.rgb",
            "flux-dev-sequential.rgb",
            "qwen-resident.log",
            "qwen-sequential.log",
            "flux-dev-resident.log",
            "flux-dev-sequential.log",
            "qwen-vram-compare.log",
            "flux-dev-vram-compare.log",
        ):
            self.assertIn(name, self.job)
        self.assertIn("if: always()", self.job)
        self.assertIn("actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02", self.job)


if __name__ == "__main__":
    unittest.main()
