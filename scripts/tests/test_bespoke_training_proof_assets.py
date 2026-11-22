"""Metadata-only exact source asset binding tests; no model inputs are loaded."""
from pathlib import Path
import json
import tempfile
import tomllib
import unittest
import yaml
from unittest.mock import patch
from scripts.ci import prepare_bespoke_training_proof_assets as assets

class BespokeTrainingAssets(unittest.TestCase):
    def test_owned_runtime_confines_state_and_refuses_other_jobs_or_download_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); work = root / 't' / '1234abcd'; work.mkdir(parents=True)
            env = dict(DEV_HOST_ROOT=str(root), DEV_HOST_JOB_ROOT=str(work), GITHUB_RUN_ID='123', GITHUB_RUN_ATTEMPT='2',
                       GITHUB_REPOSITORY='SceneWorks/inference', GITHUB_JOB='prove', RUNNER_NAME='runner', HF_HOME='E:\\huggingface')
            owner = dict(repository=env['GITHUB_REPOSITORY'], run='123', attempt='2', job='prove', runner='runner')
            (work / '.actions-owner.json').write_text(json.dumps(owner), encoding='utf-8')
            identity = next(iter(assets.BESPOKE_TRAINING_PROOFS))
            self.assertEqual(assets.owned_runtime_root(identity, env), work / 'model-proofs' / '123-2' / identity.split('::')[-1])
            for patch in ({'GITHUB_RUN_ATTEMPT':'3'}, {'GITHUB_JOB':'other'}, {'HF_HOME':'D:/models'}, {'HF_HUB_CACHE':'D:/models/hub'}, {'DEV_HOST_JOB_ROOT':str(root)}):
                with self.subTest(patch=patch), self.assertRaises(ValueError): assets.owned_runtime_root(identity, {**env, **patch})
    def row(self, identity):
        package, target, fqn = identity.split("/", 2)
        return {"package":package,"target":target,"target_kind":"test","fqn":fqn,
                "selected_configuration":{"name":"fixed_cuda"}}
    def test_only_eight_exact_fixed_cuda_bindings(self):
        self.assertEqual(len(assets.BESPOKE_TRAINING_PROOFS), 8)
        for identity, expected in assets.BESPOKE_TRAINING_PROOFS.items():
            row = self.row(identity)
            self.assertEqual(assets.exact_bindings([row], "candle-cuda"), expected)
            for backend in ("candle-cpu", "candle-metal", "mlx"):
                self.assertIsNone(assets.exact_bindings([row], backend))
            for change in ({"target":"all"},{"fqn":"all"},{"target_kind":"lib"},
                           {"selected_configuration":{"name":"source_selected_cuda"}},
                           {"selected_configuration":{}}):
                self.assertIsNone(assets.exact_bindings([{**row,**change}],"candle-cuda"))
            self.assertIsNone(assets.exact_bindings([row,row],"candle-cuda"))
        self.assertIsNone(assets.exact_bindings([],"candle-cuda"))
    def test_metadata_admission_requires_nonempty_named_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.dict(assets.REQUIRED_FILES,{"TEST_ASSET":("config.json",)},clear=True):
                with self.assertRaises(ValueError): assets.validate_paths({"TEST_ASSET":root})
                (root/"config.json").write_bytes(b"")
                with self.assertRaises(ValueError): assets.validate_paths({"TEST_ASSET":root})
                (root/"config.json").write_text("{}", encoding="utf-8")
                assets.validate_paths({"TEST_ASSET":root})
    def test_workflow_rebuilds_the_plan_on_the_actual_worker_before_asset_setup(self):
        root = Path(__file__).resolve().parents[2]
        workflow = yaml.safe_load((root / '.github/workflows/model-inference-proofs.yml').read_text(encoding='utf-8'))
        steps = next(job['steps'] for job in workflow['jobs'].values()
                     if any('prepare_bespoke_training_proof_assets.py' in step.get('run', '')
                            for step in job.get('steps', [])))
        step = next(step for step in steps if 'prepare_bespoke_training_proof_assets.py' in step.get('run', ''))
        setup = next(s for s in steps if s.get('with', {}).get('dev-host') == 'true')
        self.assertLess(steps.index(setup), steps.index(step))
        command = step['run']
        self.assertIn("matrix.asset_setup == 'bespoke-training'", step['if'])
        self.assertIn('--request-env MODEL_PROOF_REQUEST --expected-sha "%GITHUB_SHA%"', command)
        self.assertLess(command.index('model_inference_proofs.py --plan'), command.index('if errorlevel 1 exit'))
        self.assertLess(command.index('if errorlevel 1 exit'), command.index('prepare_bespoke_training_proof_assets.py'))
        # The hash-pinned huggingface_hub a missing input is downloaded with reaches the setup in-process.
        self.assertLess(command.index('set "PROOF_HF_HUB_PATH='), command.index('prepare_bespoke_training_proof_assets.py'))

    def test_base_sources_are_exact_installed_revisions(self):
        manifest = Path(__file__).resolve().parents[2] / "release/real-weight-models.toml"
        models = {row["key"]: row for row in tomllib.loads(manifest.read_text(encoding="utf-8"))["models"]}
        for bindings in assets.BESPOKE_TRAINING_PROOFS.values():
            for name,item in bindings.items():
                model = models[item.model]
                self.assertRegex(model["revision"], r"^[0-9a-f]{40}$")
                self.assertIn(name,assets.REQUIRED_FILES)
                # The metadata admission's files are inside the row's verified slice.
                prefix = item.path + "/" if item.path else ""
                for relative in assets.REQUIRED_FILES[name]:
                    self.assertIn(prefix + relative, model["expected_files"])

if __name__ == "__main__": unittest.main()
