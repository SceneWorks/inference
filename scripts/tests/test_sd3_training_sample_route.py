"""Source/metadata regressions for SD3 sample admission; never execute tensors or models."""
from pathlib import Path
import json
import os
import re
import tempfile
import unittest
from unittest.mock import patch

from scripts.check_model_inference_policy import rust_mask
from scripts.ci import prepare_cohort_training_proof_assets as assets
from scripts.ci.model_proof_inputs import validate_contract

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "crates/media/candle-gen/candle-gen-sd3/tests/trainer_cuda_smoke.rs"
CASE = "trainer_cuda_smoke::sd35_medium_bounded_acceleration_acceptance"
SELECTION = "candle-gen-sd3/" + CASE


def function(source, name):
    masked = rust_mask(source)
    match = re.search(r"\bfn\s+" + re.escape(name) + r"\s*\(", masked)
    if match is None:
        raise ValueError("missing function " + name)
    start = masked.index("{", match.end())
    depth = 1
    end = start + 1
    while depth:
        depth += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    return source[start:end]


def sample_contract_errors(source):
    """Discriminate the failed unbound/upstream sample request without constructing a model."""
    errors = []
    helper = function(source, "sample_load_spec")
    for variant, label in (("medium", "Medium"), ("large", "Large")):
        if not re.search(
            rf'"sd3_5_{variant}"\s*=>\s*\(\s*'
            rf'candle_gen_sd3::memory_strategy::Sd35Route::{label},\s*'
            rf'"SD35_{variant.upper()}_SAMPLE_SNAPSHOT"', helper
        ):
            errors.append("crossed sample source/route " + variant)
    if ".with_resolved_route(route.provider_id())" not in helper:
        errors.append("missing exact resolved route")
    if "Sd35LoadReceipt::capture(route, &spec)" not in helper:
        errors.append("missing production artifact admission")
    if ".with_quant(" in helper or ".precision =" in helper:
        errors.append("sample must retain default BF16 physical tier selection")
    for name in ("run", "bounded_run"):
        body = function(source, name)
        preflight = body.find("let sample_spec = sample_load_spec(id);")
        train = body.find(".load_trainer(")
        if not 0 <= preflight < train:
            errors.append("sample admission must precede trainer load: " + name)
        if "sample_spec.with_adapters(" not in body:
            errors.append("adapter sample must use the admitted production spec: " + name)
    return errors


class Sd3TrainingSampleRoute(unittest.TestCase):
    def test_exact_production_sample_contract_and_preflight_order(self):
        self.assertEqual(sample_contract_errors(SOURCE.read_text(encoding="utf-8")), [])

    def test_regression_rejects_unbound_crossed_or_late_sample_admission(self):
        source = SOURCE.read_text(encoding="utf-8")
        mutations = [
            source.replace(".with_resolved_route(route.provider_id())", ""),
            source.replace("Sd35LoadReceipt::capture(route, &spec)", "removed_receipt_check()"),
            source.replace('"SD35_MEDIUM_SAMPLE_SNAPSHOT"', '"SD35_MEDIUM_TRAIN_SNAPSHOT"'),
            source.replace("let sample_spec = sample_load_spec(id);", "", 1),
            source.replace("sample_spec.with_adapters(", "LoadSpec::new(WeightsSource::Dir(snapshot)).with_adapters("),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutations.index(mutation)):
                self.assertNotEqual(sample_contract_errors(mutation), [])

    def test_registry_declares_distinct_training_and_generation_inputs(self):
        rows = json.loads((ROOT / "scripts/ci/model-inference-proofs.json").read_text(encoding="utf-8"))
        smoke = [row for row in rows if row["package"] == "candle-gen-sd3"
                 and row["fqn"].startswith("trainer_cuda_smoke::")]
        self.assertEqual(len(smoke), 5)
        for row in smoke:
            variant = "MEDIUM" if "medium" in row["fqn"] else "LARGE"
            self.assertIn(f"SD35_{variant}_TRAIN_SNAPSHOT", row["required_environment"])
            self.assertIn(f"SD35_{variant}_SAMPLE_SNAPSHOT", row["required_environment"])
            validate_contract(row, required=True)
        bounded = next(row for row in smoke if row["fqn"] == CASE)
        weights = next(item for item in bounded["input_contract"]["requirements"]
                       if item["role"] == "model-weights")
        self.assertEqual(weights["readers"], ["candle-gen::loader::mmap_var_builder"])
        self.assertEqual(weights["minimum_collections"], 2)
        self.assertIn("distinct pinned upstream", bounded["fixture_identity"])

    def test_dispatch_binds_exact_installed_turnkey_separately_from_training(self):
        binding = assets.selected_binding(SELECTION)
        self.assertEqual(binding, assets.selected_binding(SELECTION.replace("/trainer_", "/integration/trainer_")))
        self.assertEqual(set(binding), {"SD35_MEDIUM_TRAIN_SNAPSHOT", "SD35_MEDIUM_SAMPLE_SNAPSHOT"})
        self.assertEqual(binding["SD35_MEDIUM_TRAIN_SNAPSHOT"], assets.snapshot(
            "stabilityai/stable-diffusion-3.5-medium", "b940f670f0eda2d07fbb75229e779da1ad11eb80"))
        self.assertEqual(binding["SD35_MEDIUM_SAMPLE_SNAPSHOT"], assets.snapshot(
            "SceneWorks/sd3.5-medium-mlx", "5413e962bb326db248be2026a93b147c323392b6", "bf16"))
        for selection in ("candle-gen-sd3", "all", SELECTION.replace("medium", "large")):
            with self.assertRaises(ValueError):
                assets.selected_binding(selection)

    def test_missing_generation_asset_rejects_setup_before_exporting_training_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            sample = root / "sample"
            train.mkdir()
            (train / "metadata-fixture.safetensors").write_bytes(b"metadata fixture only")
            export = root / "github-env"
            binding = {"SD35_MEDIUM_TRAIN_SNAPSHOT": train, "SD35_MEDIUM_SAMPLE_SNAPSHOT": sample}
            environment = {"MODEL_PROOF_REQUEST": json.dumps({"selection": SELECTION, "configuration": "fixed_cuda"}),
                           "GITHUB_ENV": str(export)}
            with patch.dict(assets.BINDINGS, {SELECTION: binding}, clear=True), patch.dict(os.environ, environment):
                with self.assertRaisesRegex(ValueError, "SD35_MEDIUM_SAMPLE_SNAPSHOT"):
                    assets.main()
                self.assertFalse(export.exists())
                sample.mkdir()
                (sample / "metadata-fixture.safetensors").write_bytes(b"metadata fixture only")
                assets.main()
            self.assertEqual(export.read_text(encoding="utf-8"), "".join(
                f"{name}={path.as_posix()}\n" for name, path in binding.items()))


if __name__ == "__main__":
    unittest.main()
