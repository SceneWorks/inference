"""Metadata-only MLX bounded-training asset binding tests; no model inputs are loaded."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from scripts.ci import model_inference_proofs as proofs
from scripts.ci import prepare_mlx_training_proof_assets as assets

ROOT = Path(__file__).resolve().parents[2]


class MlxTrainingAssets(unittest.TestCase):
    def row(self, identity):
        package, target, fqn = identity.split("/", 2)
        return {"package": package, "target": target, "target_kind": "test", "fqn": fqn}

    def test_every_binding_is_one_exact_registered_mlx_row(self):
        rows = {proofs.identity(row): row for row in proofs.load_registry()}
        for identity, bindings in assets.MLX_TRAINING_PROOFS.items():
            self.assertIn(identity, rows)
            self.assertEqual(proofs.lane(rows[identity]), "mlx")
            self.assertTrue(rows[identity].get("output_artifacts"), identity)
            row = self.row(identity)
            self.assertEqual(assets.exact_bindings([row], "mlx", "nax"), bindings)
            self.assertIsNone(assets.exact_bindings([row], "mlx", None))
            self.assertIsNone(assets.exact_bindings([row], "mlx", "hosted"))
            for backend in ("candle-cpu", "candle-metal", "candle-cuda"):
                self.assertIsNone(assets.exact_bindings([row], backend, "nax"))
            self.assertIsNone(assets.exact_bindings([row, row], "mlx", "nax"))
            self.assertIsNone(assets.exact_bindings([{**row, "fqn": "other"}], "mlx", "nax"))
        self.assertIsNone(assets.exact_bindings([], "mlx", "nax"))

    def test_pins_are_exact_revisions_with_required_files(self):
        for bindings in assets.MLX_TRAINING_PROOFS.values():
            for name, pin in bindings.items():
                self.assertRegex(name, r"^[A-Z][A-Z0-9_]*$")
                self.assertRegex(pin.revision, r"^[0-9a-f]{40}$")
                self.assertTrue(pin.files, name)
                self.assertNotIn("..", Path(pin.subdir).parts)

    def test_roots_are_ordered_tilde_expanded_and_unique(self):
        home = Path("/Users/runner")
        roots = assets.candidate_roots({"MLX_GEN_MODELS_ROOT": "~/.cache/huggingface/hub"}, home)
        self.assertEqual([root for _, root in roots], [
            home / ".cache/huggingface/hub", assets.PROVISIONED_HUB, home / "sceneworks-rw-weights/hub"])
        roots = assets.candidate_roots({"MLX_GEN_MODELS_ROOT": str(assets.PROVISIONED_HUB)}, home)
        self.assertEqual(len(roots), 2)
        self.assertEqual(len(assets.candidate_roots({"MLX_GEN_MODELS_ROOT": "relative"}, home)), 2)

    def test_resolution_requires_the_complete_pin_and_reports_an_inventory(self):
        pin = assets.Pin("org/model", "a" * 40, "tier", files=("config.json",))
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            roots = [("first", first), ("second", second)]
            self.assertIsNone(assets.resolve(pin, roots))
            snapshot = second / pin.relative()
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_bytes(b"")
            self.assertIsNone(assets.resolve(pin, roots))
            report = assets.inventory([pin], roots, ("org/other",))
            self.assertEqual(report["org/model"]["roots"][f"second:{second}"], {"a" * 40: {"tier": ["config.json"]}})
            self.assertEqual(report["org/model"]["roots"][f"first:{first}"], "root absent")
            self.assertEqual(report["org/other"]["roots"][f"second:{second}"], "repository absent")
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(assets.resolve(pin, roots), snapshot)
            root_pin = assets.Pin("org/model", "a" * 40, "tier", files=("config.json",), binds_root=True)
            self.assertEqual(assets.resolve(root_pin, roots), second)

    def test_a_symlinked_root_is_never_a_binding_source(self):
        pin = assets.Pin("org/model", "b" * 40, files=("config.json",))
        with tempfile.TemporaryDirectory() as directory:
            real = Path(directory) / "real"
            (real / pin.relative()).mkdir(parents=True)
            (real / pin.relative() / "config.json").write_text("{}", encoding="utf-8")
            link = Path(directory) / "link"
            try:
                link.symlink_to(real, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks unavailable")
            self.assertTrue(assets.linked_root(link))
            self.assertIsNone(assets.resolve(pin, [("link", link)]))
            self.assertEqual(assets.resolve(pin, [("link", link), ("real", real)]), real / pin.relative())

    def test_main_binds_only_the_exact_planned_case_on_the_provisioned_mac(self):
        identity = next(iter(assets.MLX_TRAINING_PROOFS))
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "plan.json"
            plan.write_text(json.dumps({"backend": "mlx", "selected": 1, "bindings": [identity],
                                        "request": {"runner": "nax"}, "asset_setup": "mlx-training"}))
            github_env = Path(directory) / "env"
            env = {"RUNNER_OS": "macOS", "RUNNER_NAME": "nax-macos-1", "GITHUB_ENV": str(github_env)}
            with patch.dict("os.environ", env, clear=True), patch("sys.argv", ["x", "--plan", str(plan)]):
                with self.assertRaises(ValueError):
                    assets.main()

    def test_plan_routes_bound_rows_and_workflow_runs_the_setup_on_macos(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/model-inference-proofs.yml").read_text(encoding="utf-8"))
        steps = workflow["jobs"]["prove"]["steps"]
        step = next(step for step in steps if "prepare_mlx_training_proof_assets.py" in step.get("run", ""))
        self.assertIn("matrix.asset_setup == 'mlx-training'", step["if"])
        self.assertIn("runner.os == 'macOS'", step["if"])
        command = step["run"]
        self.assertLess(command.index("model_inference_proofs.py --plan"),
                        command.index("prepare_mlx_training_proof_assets.py"))
        self.assertIn("vars.MLX_GEN_MODELS_ROOT", step["env"]["MLX_GEN_MODELS_ROOT"])
        source = (ROOT / "scripts/ci/model_inference_proofs.py").read_text(encoding="utf-8")
        self.assertIn("asset_setup = 'mlx-training'", source)


if __name__ == "__main__":
    unittest.main()
