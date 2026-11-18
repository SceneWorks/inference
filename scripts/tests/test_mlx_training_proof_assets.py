"""Metadata-only MLX bounded-training asset binding tests; no model inputs are loaded."""
import json
import os
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
            self.assertEqual(assets.exact_bindings([row], "mlx", "nax-primary"), bindings)
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

    def test_roots_are_ordered_tilde_expanded_resolved_and_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve()
            roots = assets.candidate_roots({"MLX_GEN_MODELS_ROOT": "~/.cache/huggingface/hub"}, home)
            self.assertEqual([root for _, root in roots], [
                Path(os.path.realpath(home / ".cache/huggingface/hub")),
                Path(os.path.realpath(assets.PROVISIONED_HUB)),
                Path(os.path.realpath(home / "sceneworks-rw-weights/hub"))])
            self.assertTrue(roots[0][0].startswith("MLX_GEN_MODELS_ROOT="))
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
            self.assertEqual(report["org/model"]["roots"]["second"], {"a" * 40: {"tier": ["config.json"]}})
            self.assertEqual(report["org/model"]["roots"]["first"], "root absent")
            self.assertEqual(report["org/other"]["roots"]["second"], "repository absent")
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(assets.resolve(pin, roots), snapshot)
            root_pin = assets.Pin("org/model", "a" * 40, "tier", files=("config.json",), binds_root=True)
            self.assertEqual(assets.resolve(root_pin, roots), second)

    def test_a_symlinked_root_resolves_to_its_real_store(self):
        pin = assets.Pin("org/model", "b" * 40, files=("config.json",))
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve()
            real = home / "store"
            (real / pin.relative()).mkdir(parents=True)
            (real / pin.relative() / "config.json").write_text("{}", encoding="utf-8")
            link = home / ".cache" / "hub"
            link.parent.mkdir()
            try:
                link.symlink_to(real, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks unavailable")
            roots = assets.candidate_roots({"MLX_GEN_MODELS_ROOT": "~/.cache/hub"}, home)
            self.assertEqual(roots[0][1], real)
            self.assertIn(f"-> {real}", roots[0][0])
            self.assertEqual(assets.resolve(pin, roots), real / pin.relative())

    def test_download_missing_is_opt_in_and_fetches_only_absent_exact_revisions(self):
        present = assets.Pin("org/present", "c" * 40, files=("config.json",))
        absent = assets.Pin("org/absent", "d" * 40, "tier", files=("config.json",))
        twin = assets.Pin("org/absent", "d" * 40, "other", files=("config.json",))
        bindings = {"PRESENT": present, "ABSENT": absent, "TWIN": twin}
        with tempfile.TemporaryDirectory() as directory:
            primary = Path(directory).resolve() / "hub"
            (primary / present.relative()).mkdir(parents=True)
            (primary / present.relative() / "config.json").write_text("{}", encoding="utf-8")
            roots = [(f"MLX_GEN_MODELS_ROOT={primary}", primary)]
            calls = []

            def fake_download(pin, cache_dir):
                calls.append((pin.repository, pin.revision, cache_dir))
                for same in (absent, twin):
                    (cache_dir / same.relative()).mkdir(parents=True, exist_ok=True)
                    (cache_dir / same.relative() / "config.json").write_text("{}", encoding="utf-8")

            resolved, downloads = assets.bind(bindings, roots, False, download=fake_download)
            self.assertEqual(calls, [])
            self.assertEqual(downloads, [])
            self.assertIsNone(resolved["ABSENT"])
            resolved, downloads = assets.bind(bindings, roots, True, download=fake_download)
            self.assertEqual(calls, [("org/absent", "d" * 40, primary)])
            self.assertEqual(downloads, [{"repository": "org/absent", "revision": "d" * 40, "cache_dir": str(primary)}])
            self.assertEqual(resolved["ABSENT"], primary / absent.relative())
            self.assertEqual(resolved["TWIN"], primary / twin.relative())
            self.assertEqual(resolved["PRESENT"], primary / present.relative())

    def test_a_refused_download_is_recorded_and_the_pin_stays_missing(self):
        gated = assets.Pin("org/gated", "e" * 40, files=("config.json",))
        with tempfile.TemporaryDirectory() as directory:
            primary = Path(directory).resolve()
            roots = [(f"MLX_GEN_MODELS_ROOT={primary}", primary)]

            def refused(pin, cache_dir):
                raise RuntimeError("403 Client Error\nCannot access gated repo")

            resolved, downloads = assets.bind({"GATED": gated}, roots, True, download=refused)
            self.assertIsNone(resolved["GATED"])
            self.assertEqual(downloads[0]["error"], "RuntimeError: 403 Client Error")

    def test_request_validates_download_missing_and_the_primary_mac_route(self):
        policy = {"backend": "darwin-mlx"}
        value = proofs.request(json.dumps({"selection": "x", "policy": policy, "runner": "nax-primary",
                                           "download_missing": True}), "mlx")
        self.assertTrue(value["download_missing"])
        for bad in ({"download_missing": "yes"}, {"download_missing": 1}, {"runner": "nax-3"}):
            with self.assertRaises(ValueError):
                proofs.request(json.dumps({"selection": "x", "policy": policy, **bad}), "mlx")
        self.assertEqual(proofs.NAX_ROUTES["nax-primary"]["name"], "nax-macos")
        self.assertIn("rw-starvector", proofs.NAX_ROUTES["nax-primary"]["labels"])
        self.assertEqual(assets.NAX_RUNNERS, {k: v["name"] for k, v in proofs.NAX_ROUTES.items()})

    def test_main_binds_only_the_exact_planned_case_on_the_provisioned_mac(self):
        identity = next(iter(assets.MLX_TRAINING_PROOFS))
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "plan.json"
            plan.write_text(json.dumps({"backend": "mlx", "selected": 1, "bindings": [identity],
                                        "request": {"runner": "nax"}, "asset_setup": "mlx-training"}),
                            encoding="utf-8")
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
