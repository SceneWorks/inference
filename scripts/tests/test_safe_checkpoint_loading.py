"""CPU checks for the checkpoint formats accepted by conversion and vendored loaders."""

import ast
import importlib.util
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

try:
    import torch
    from safetensors.torch import load_file
except ImportError:
    torch = None


ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_marker(path):
    pathlib.Path(path).write_text("pickle executed", encoding="utf-8")


class _Marker:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return _write_marker, (self.path,)


@unittest.skipUnless(torch is not None, "torch and safetensors are required")
class SafeCheckpointLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.tensor = torch.tensor([[1.25, -2.5], [3.0, 4.75]], dtype=torch.float32)
        self.marker = self.root / "marker.txt"
        self.safe = self.root / "safe.pt"
        self.unsafe = self.root / "unsupported.pt"
        torch.save({"codec_model": {"encoder.weight": self.tensor}, "epoch": 3}, self.safe)
        torch.save({"codec_model": {"encoder.weight": self.tensor}, "extra": _Marker(self.marker)}, self.unsafe)

    def assert_tensor(self, value):
        self.assertEqual(tuple(value.shape), (2, 2))
        self.assertEqual(value.dtype, torch.float32)
        self.assertTrue(torch.equal(value, self.tensor))

    def test_pid_conversion_and_default_rejection(self):
        script = ROOT / "crates/media/mlx-gen/tools/convert_pid.py"
        good = self.root / "pid.pt"
        bad = self.root / "pid-unsupported.pt"
        output = self.root / "pid.safetensors"
        torch.save({"state_dict": {"net.patch_blocks.0.weight": self.tensor,
                                    "net_ema.patch_blocks.0.weight": self.tensor}}, good)
        subprocess.run([sys.executable, str(script), str(good), str(output)], check=True, capture_output=True)
        converted = load_file(str(output))
        self.assertEqual(set(converted), {"patch_blocks.0.weight"})
        self.assert_tensor(converted["patch_blocks.0.weight"])

        torch.save({"state_dict": {"net.patch_blocks.0.weight": self.tensor},
                    "extra": _Marker(self.marker)}, bad)
        denied = subprocess.run([sys.executable, str(script), str(bad), "--dry-run"], capture_output=True, text=True, encoding="utf-8")
        self.assertNotEqual(denied.returncode, 0)
        self.assertIn("safe tensor checkpoint load failed", denied.stderr)
        self.assertFalse(self.marker.exists())

    def test_yue_conversion_flat_and_wrapped_state_dicts(self):
        prepare = _load("prepare_yue_assets", "scripts/audio/prepare_yue_assets.py")
        output = self.root / "codec.safetensors"
        prepare.convert_state_dict(self.safe, "codec_model", output)
        self.assert_tensor(load_file(str(output))["encoder.weight"])
        flat = self.root / "decoder.pt"
        torch.save({"decoder.weight": self.tensor}, flat)
        prepare.convert_state_dict(flat, None, self.root / "decoder.safetensors")
        self.assert_tensor(load_file(str(self.root / "decoder.safetensors"))["decoder.weight"])
        with self.assertRaisesRegex(ValueError, "safe tensor checkpoint load failed"):
            prepare._state_dict(self.unsafe, "codec_model")
        self.assertFalse(self.marker.exists())

    def test_yue_reference_loaders(self):
        for name in ("yue_icl_reference", "yue_stage2_reference", "yue_xcodec_reference"):
            with self.subTest(name=name):
                module = _load(name, f"scripts/reference/{name}.py")
                self.assert_tensor(module.load_codec_state(self.safe)["encoder.weight"])
                with self.assertRaisesRegex(ValueError, "tensor-only PyTorch export"):
                    module.load_codec_state(self.unsafe)
                self.assertFalse(self.marker.exists())

    def test_vendored_bigvgan_and_mage_vae(self):
        paths = (
            "crates/audio/candle-audio-mmaudio/_vendor/mmaudio/ext/bigvgan/utils.py",
            "crates/audio/candle-audio-mmaudio/_vendor/mmaudio/ext/bigvgan_v2/utils.py",
        )
        for i, path in enumerate(paths):
            with self.subTest(path=path):
                module = _load(f"bigvgan_checkpoint_{i}", path)
                self.assert_tensor(module.load_checkpoint(self.safe, "cpu")["codec_model"]["encoder.weight"])
                with self.assertRaisesRegex(ValueError, "tensor-only weights"):
                    module.load_checkpoint(self.unsafe, "cpu")
                self.assertFalse(self.marker.exists())

        loguru = types.ModuleType("loguru")
        loguru.logger = types.SimpleNamespace(info=lambda *a: None, warning=lambda *a: None)
        with patch.dict(sys.modules, {"loguru": loguru}):
            module = _load("mage_vae_checkpoint", "crates/media/mlx-gen/_vendor/mage_flow/models/modules/mage_vae.py")
        torch.save({"state_dict": {"encoder.weight": self.tensor}}, self.root / "mage-vae.pt")
        self.assert_tensor(module._load_state_dict(str(self.root / "mage-vae.pt"))["encoder.weight"])
        with self.assertRaisesRegex(ValueError, "tensor-only weights"):
            module._load_state_dict(str(self.unsafe))
        self.assertFalse(self.marker.exists())

    def test_mage_model_weight_loader(self):
        messages = []
        loguru = types.ModuleType("loguru")
        loguru.logger = types.SimpleNamespace(info=lambda message: messages.append(message),
                                              warning=lambda *a: None)
        package = types.ModuleType("checkpoint_test_mage")
        package.__path__ = []
        models = types.ModuleType("checkpoint_test_mage.models")
        models.__path__ = []
        mage_flow = types.ModuleType("checkpoint_test_mage.models.mage_flow")
        mage_flow.MageFlow = object
        mage_flow.MageFlowParams = object
        checkpoint_paths = _load("checkpoint_test_mage.checkpoint_paths",
                                 "crates/media/mlx-gen/_vendor/mage_flow/checkpoint_paths.py")
        modules = {"loguru": loguru, package.__name__: package, models.__name__: models,
                   mage_flow.__name__: mage_flow, checkpoint_paths.__name__: checkpoint_paths}
        with patch.dict(sys.modules, modules):
            utils = _load("checkpoint_test_mage.models.utils",
                          "crates/media/mlx-gen/_vendor/mage_flow/models/utils.py")
        model = torch.nn.Linear(2, 2)
        torch.save({"weight": self.tensor, "bias": torch.tensor([0.5, -1.0])}, self.root / "mage-model.pt")
        self.assertTrue(utils.load_model_weight(model, str(self.root / "mage-model.pt")))
        self.assert_tensor(model.weight.detach())
        self.assertTrue(torch.equal(model.bias.detach(), torch.tensor([0.5, -1.0])))
        self.assertFalse(utils.load_model_weight(model, str(self.unsafe)))
        self.assertTrue(any("tensor-only weights" in message for message in messages))
        self.assertFalse(self.marker.exists())

    def test_every_target_load_uses_explicit_safe_policy(self):
        paths = (
            "crates/media/mlx-gen/tools/convert_pid.py",
            "scripts/audio/prepare_yue_assets.py",
            "scripts/reference/yue_icl_reference.py",
            "scripts/reference/yue_stage2_reference.py",
            "scripts/reference/yue_xcodec_reference.py",
            "crates/audio/candle-audio-mmaudio/_vendor/mmaudio/ext/bigvgan/utils.py",
            "crates/audio/candle-audio-mmaudio/_vendor/mmaudio/ext/bigvgan_v2/utils.py",
            "crates/audio/candle-audio-mmaudio/_vendor/mmaudio/ext/synchformer/motionformer.py",
            "crates/media/mlx-gen/_vendor/mage_flow/models/mage_flow.py",
            "crates/media/mlx-gen/_vendor/mage_flow/models/modules/mage_vae.py",
            "crates/media/mlx-gen/_vendor/mage_flow/models/utils.py",
        )
        for path in paths:
            with self.subTest(path=path):
                tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Attribute) and node.func.attr == "load"
                         and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"]
                self.assertEqual(len(calls), 1)
                self.assertTrue(any(keyword.arg == "weights_only" and isinstance(keyword.value, ast.Constant)
                                    and keyword.value.value is True for keyword in calls[0].keywords))


if __name__ == "__main__":
    unittest.main()
