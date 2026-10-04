"""Reject unpinned training reuse and execute semantic fixtures on CPU only."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from scripts.ci.qwen21_diagnostic_adapter import (
    ADAPTER_SHA, CAPTION, DATASET, validate_provenance, validate_training,
)

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "crates/media/mlx-gen/mlx-gen-qwen-image-2-1/tests/support/edit_protocol.rs"
RUSTC = shutil.which("rustc")


class DiagnosticProvenanceTests(unittest.TestCase):
    def receipt(self):
        metadata = {"family": "qwen-image-2-1", "baseModel": "qwen_image_2_1",
                    "trainingMode": "edit", "networkType": "lokr", "rank": "16", "alpha": "16",
                    "license": "Qwen Research License Agreement (research/evaluation only)"}
        receipt = {"training": {"steps": 120, "stepsRun": 120, "losses": [0.1] * 120,
            "stepSamples": [{"step": i} for i in range(1, 121)],
            "trainingStageTraceComplete": True, "adapterSha256": ADAPTER_SHA, "adapterBytes": 6759417,
            "dataset": {"sha256": DATASET}, "metadata": metadata,
            "editProtocol": {"trainingReferenceCount": 1, "evaluationReferenceCount": 2,
                             "trainingCaption": CAPTION}}}
        return receipt, metadata

    def test_original_provenance_and_complete_receipt(self):
        manifest = json.loads((ROOT / "scripts/ci/qwen21_diagnostic_adapter.json").read_text(encoding="utf-8"))
        self.assertEqual(len(validate_provenance(manifest)), 3)
        validate_training(*self.receipt())

    def test_diagnostic_cannot_become_training_or_acceptance_and_keeps_all_failures(self):
        source = (SUPPORT.parent.parent / "lora_real_weights.rs").read_text(encoding="utf-8")
        body = source.split("fn diagnostic_reused_edit_adapter_semantics()", 1)[1].split("fn diagnostic_render_checks", 1)[0]
        self.assertNotIn("train(&", body)
        self.assertIn('"acceptanceEvidence": false', body)
        self.assertIn("if render_count != 19", body)
        self.assertIn('"failures": failures', body)
        self.assertRegex(body, r"images:\s*vec!\[\s*to_image\(edit_source\(99,\s*RENDER_EDGE\)\),\s*to_image\(key\.clone\(\)\)")
        self.assertLess(body.index('"diagnostic_semantics"'), body.index("failures.is_empty()"))

    def test_provenance_hash_and_acceptance_scope_mutants(self):
        manifest = json.loads((ROOT / "scripts/ci/qwen21_diagnostic_adapter.json").read_text(encoding="utf-8"))
        mutants = []
        for key, value in [("trainingSourceMain", "0" * 40), ("trainingRun", 1), ("steps", 2),
                           ("trainingCaption", "grey inversion"), ("datasetSha256", "unknown")]:
            bad = copy.deepcopy(manifest); bad["trainingProvenance"][key] = value; mutants.append(bad)
        for index in range(3):
            bad = copy.deepcopy(manifest); bad["adapters"][index]["sha256"] = ""; mutants.append(bad)
        bad = copy.deepcopy(manifest); bad["acceptanceEvidence"] = True; mutants.append(bad)
        bad = copy.deepcopy(manifest); del bad["trainingProvenance"]; mutants.append(bad)
        bad = copy.deepcopy(manifest); bad["adapters"].pop(); mutants.append(bad)
        for bad in mutants:
            with self.subTest(mutant=bad), self.assertRaises(ValueError):
                validate_provenance(bad)

    def test_incomplete_steps_wrong_family_license_and_nonfinite_loss(self):
        for key, value in [("stepsRun", 119), ("losses", [float("nan")] * 120),
                           ("stepSamples", [{"step": 1}] * 120), ("trainingStageTraceComplete", False),
                           ("adapterSha256", ""), ("adapterBytes", 1)]:
            receipt, metadata = self.receipt(); receipt["training"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_training(receipt, metadata)
        for key in ["family", "license", "networkType", "trainingMode"]:
            receipt, metadata = self.receipt(); metadata[key] = "wrong"
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_training(receipt, metadata)


@unittest.skipUnless(RUSTC, "standalone semantic CPU fixtures require rustc")
class SemanticCpuTests(unittest.TestCase):
    def compile_and_test(self, source):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / "protocol.rs"; binary = root / "protocol-test"
            path.write_text(source, encoding="utf-8")
            result = subprocess.run([RUSTC, "--edition=2021", "--test", str(path), "-o", str(binary)],
                capture_output=True, text=True, encoding="utf-8", check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            return subprocess.run([str(binary)], capture_output=True, text=True, encoding="utf-8", check=False)

    def test_actual_palette_caption_and_original_rgb_math(self):
        result = self.compile_and_test(SUPPORT.read_text(encoding="utf-8"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_actual_gray_key_missing_prefix_caption_and_wrong_transform_mutants(self):
        source = SUPPORT.read_text(encoding="utf-8")
        implementation, tests = source.split("#[cfg(test)]", 1)
        mutants = [
            ('levels[(index / 4) % 4],\n        levels[index % 4],', 'levels[index / 16],\n        levels[index / 16],'),
            ('zxq style, zxq edit, zxq invert:', 'zxq style:'),
            ('do not copy its layout";', 'copy its layout";'),
            ('((255 - value) / 64)', '((255 - value) / 85)'),
        ]
        for old, new in mutants:
            with self.subTest(mutation=old):
                self.assertIn(old, implementation)
                result = self.compile_and_test(implementation.replace(old, new) + "#[cfg(test)]" + tests)
                self.assertNotEqual(result.returncode, 0, "semantic implementation mutant escaped CPU test")

    def test_palette_bytes_match_frozen_protocol_without_a_png_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / "palette.rs"; binary = root / "palette"
            module_path = json.dumps(SUPPORT.as_posix())
            path.write_text(f'#[allow(dead_code)] #[path={module_path}] mod protocol;\n'
                'fn main() { use std::io::Write; let mut pixels=Vec::new();'
                'for y in 0..768 { for x in 0..768 { pixels.extend(protocol::palette_pixel(x,y,768)); }}'
                'std::io::stdout().write_all(&pixels).unwrap(); }\n', encoding="utf-8")
            subprocess.run([RUSTC, "--edition=2021", str(path), "-o", str(binary)], check=True, capture_output=True)
            result = subprocess.run([str(binary)], check=True, capture_output=True)
            self.assertEqual(hashlib.sha256(result.stdout).hexdigest(),
                             "5d94a5f50c63a47a102dc972dc24d06dd57b8636a946dc628c06b940162feb25")


if __name__ == "__main__":
    unittest.main()
