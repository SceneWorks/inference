"""Metadata-only exact source asset binding tests; no model inputs are loaded."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.ci import prepare_bespoke_training_proof_assets as assets

class BespokeTrainingAssets(unittest.TestCase):
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
                (root/"config.json").write_text("{}")
                assets.validate_paths({"TEST_ASSET":root})
    def test_base_sources_are_exact_installed_revisions(self):
        for bindings in assets.BESPOKE_TRAINING_PROOFS.values():
            for name,path in bindings.items():
                self.assertIn("snapshots",path.parts)
                self.assertEqual(len(path.parts[path.parts.index("snapshots")+1]),40)
                self.assertIn(name,assets.REQUIRED_FILES)

if __name__ == "__main__": unittest.main()
