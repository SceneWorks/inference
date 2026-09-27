"""Exercise MageFlow's shard containment with small CPU checkpoint files."""

from __future__ import annotations

import importlib.util
import importlib.metadata
import json
import os
import sys
import tempfile
import tomllib
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "crates/media/mlx-gen/_vendor"
SPEC = importlib.util.spec_from_file_location("mage_checkpoint_paths", VENDOR / "mage_flow/checkpoint_paths.py")
assert SPEC and SPEC.loader
paths = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paths)

try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch = None


class _MageCheckpointFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.index = self.model / "diffusion_pytorch_model.safetensors.index.json"

    def index_shard(self, name):
        self.index.write_text(json.dumps({"weight_map": {"weight": name}}), encoding="utf-8")


class MageCheckpointPathTests(_MageCheckpointFixture, unittest.TestCase):

    def test_rejects_traversal_absolute_symlink_and_nonregular_shards(self):
        outside = self.root / "outside.safetensors"
        outside.write_bytes(b"outside")
        (self.model / "elsewhere.safetensors").symlink_to(outside)
        (self.model / "directory.safetensors").mkdir()
        for name, message in (
            ("../outside.safetensors", "relative path"),
            (str(outside), "relative path"),
            (r"C:\outside.safetensors", "invalid checkpoint shard path"),
            ("elsewhere.safetensors", "escapes authorized roots"),
            ("directory.safetensors", "not a regular file"),
        ):
            with self.subTest(name=name):
                self.index_shard(name)
                with self.assertRaisesRegex(ValueError, message):
                    paths.validate_checkpoint_index(self.index)
                with self.assertRaises(ValueError):
                    paths.preflight_model_source(str(self.model))

    def test_accepts_regular_nested_and_internal_symlink_shards(self):
        nested = self.model / "shards"
        nested.mkdir()
        (nested / "weight.safetensors").write_bytes(b"weights")
        self.index_shard("shards/weight.safetensors")
        self.assertEqual(paths.validate_checkpoint_index(self.index), [nested.resolve() / "weight.safetensors"])
        (self.model / "linked.safetensors").symlink_to(nested / "weight.safetensors")
        self.index_shard("linked.safetensors")
        self.assertEqual(paths.validate_checkpoint_index(self.index), [nested.resolve() / "weight.safetensors"])

    def test_accepts_only_hub_snapshot_blob_symlinks(self):
        repo = self.root / "models--org--model"
        snapshot = repo / "snapshots" / ("a" * 40)
        component = snapshot / "text_encoder"
        component.mkdir(parents=True)
        blobs = repo / "blobs"
        blobs.mkdir()
        blob = blobs / ("b" * 64)
        blob.write_bytes(b"weights")
        (component / "shard.safetensors").symlink_to(os.path.relpath(blob, component))
        self.assertEqual(paths.resolve_checkpoint_shard(component, "shard.safetensors"),
                         blob.resolve())
        rogue = self.root / "rogue.safetensors"
        rogue.write_bytes(b"rogue")
        (component / "rogue.safetensors").symlink_to(rogue)
        with self.assertRaisesRegex(ValueError, "escapes authorized roots"):
            paths.resolve_checkpoint_shard(component, "rogue.safetensors")


@unittest.skipUnless(torch is not None, "torch and safetensors are required")
class MageCheckpointLoaderTests(_MageCheckpointFixture, unittest.TestCase):
    def test_mage_index_loader_preserves_tensor_values_and_preflights_all_shards(self):
        sys.path.insert(0, str(VENDOR))
        self.addCleanup(lambda: sys.path.remove(str(VENDOR)))
        from mage_flow.models.utils import load_hf_style_weight

        tensor = torch.tensor([[1.25, -2.5], [3.0, 4.75]], dtype=torch.float32)
        save_file({"weight": tensor}, str(self.model / "shard.safetensors"))
        self.index_shard("shard.safetensors")
        loaded = load_hf_style_weight(str(self.model), "cpu")
        self.assertEqual(set(loaded), {"weight"})
        self.assertEqual(tuple(loaded["weight"].shape), (2, 2))
        self.assertTrue(torch.equal(loaded["weight"], tensor))
        self.index_shard("../outside.safetensors")
        with self.assertRaisesRegex(ValueError, "relative path"):
            load_hf_style_weight(str(self.model), "cpu")
        self.index.write_text(json.dumps({"weight_map": {
            "first": "shard.safetensors", "second": "../outside.safetensors",
        }}), encoding="utf-8")
        with mock.patch("mage_flow.models.utils.load_file") as shard_loader:
            with self.assertRaisesRegex(ValueError, "relative path"):
                load_hf_style_weight(str(self.model), "cpu")
            shard_loader.assert_not_called()

        repo = self.root / "models--org--model"
        component = repo / "snapshots" / ("a" * 40) / "transformer"
        component.mkdir(parents=True)
        blob = repo / "blobs" / ("b" * 64)
        blob.parent.mkdir()
        save_file({"weight": tensor}, str(blob))
        (component / "linked.safetensors").symlink_to(os.path.relpath(blob, component))
        (component / self.index.name).write_text(
            json.dumps({"weight_map": {"weight": "linked.safetensors"}}), encoding="utf-8"
        )
        self.assertTrue(torch.equal(
            load_hf_style_weight(str(component), "cpu")["weight"], tensor
        ))

    def test_text_encoder_entrypoint_preflights_before_any_hf_loader(self):
        sys.path.insert(0, str(VENDOR))
        self.addCleanup(lambda: sys.path.remove(str(VENDOR)))
        from mage_flow.models.modules import text_encoder

        self.index_shard("../outside.safetensors")
        with mock.patch.object(text_encoder.AutoTokenizer, "from_pretrained") as tokenizer:
            with self.assertRaisesRegex(ValueError, "relative path"):
                text_encoder.TextEncoder(
                    model_name="test", version=str(self.model), tokenizer_max_length=4,
                    prompt_template=None, dit_structure={}, attn_type="sdpa",
                )
            tokenizer.assert_not_called()

        tensor = torch.tensor([1.0], dtype=torch.float32)
        save_file({"weight": tensor}, str(self.model / "shard.safetensors"))
        self.index_shard("shard.safetensors")
        with (
            mock.patch.object(text_encoder.AutoTokenizer, "from_pretrained",
                              return_value=types.SimpleNamespace()) as tokenizer,
            mock.patch.object(text_encoder.CustomQwen3VLForConditionalGeneration,
                              "from_pretrained", return_value=torch.nn.Linear(1, 1)) as model_loader,
            mock.patch.object(text_encoder.AutoProcessor, "from_pretrained",
                              return_value=object()) as processor,
        ):
            text_encoder.TextEncoder(
                model_name="test", version=str(self.model), tokenizer_max_length=4,
                prompt_template=None, dit_structure={}, attn_type="sdpa",
            )
            tokenizer.assert_called_once()
            model_loader.assert_called_once()
            processor.assert_called_once()

    def test_qwen_sharded_from_pretrained_loads_and_rejects_bad_index(self):
        try:
            version = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("Transformers is not installed")
        if version != "5.10.4":
            self.skipTest("requires the locked Mage reference environment")
        sys.path.insert(0, str(VENDOR))
        self.addCleanup(lambda: sys.path.remove(str(VENDOR)))
        from accelerate import big_modeling
        from accelerate.utils import modeling
        from mage_flow.models.modules.text_encoder import CustomQwen3VLForConditionalGeneration
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig,
        )

        text = Qwen3VLTextConfig(
            hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8, intermediate_size=32, vocab_size=32,
            rope_scaling={"rope_type": "default", "mrope_interleaved": True,
                          "mrope_section": [2, 1, 1]}, use_cache=False,
        )
        vision = Qwen3VLVisionConfig(
            depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
            patch_size=2, spatial_merge_size=2, temporal_patch_size=2,
            out_hidden_size=16, num_position_embeddings=16, deepstack_visual_indexes=[],
        )
        config = Qwen3VLConfig(
            text_config=text, vision_config=vision, image_token_id=31, video_token_id=30,
        )
        original = CustomQwen3VLForConditionalGeneration(config)
        original.save_pretrained(self.model, max_shard_size="10KB")
        index = self.model / "model.safetensors.index.json"
        self.assertTrue(index.is_file())
        with (
            mock.patch.object(modeling, "load_checkpoint_in_model",
                              side_effect=AssertionError("unexpected Accelerate loader")) as direct,
            mock.patch.object(big_modeling, "load_checkpoint_and_dispatch",
                              side_effect=AssertionError("unexpected Accelerate dispatch")) as dispatch,
        ):
            loaded = CustomQwen3VLForConditionalGeneration.from_pretrained(self.model)
            direct.assert_not_called()
            dispatch.assert_not_called()
        self.assertEqual(set(loaded.state_dict()), set(original.state_dict()))
        self.assertTrue(torch.equal(
            loaded.state_dict()["model.language_model.embed_tokens.weight"],
            original.state_dict()["model.language_model.embed_tokens.weight"],
        ))
        with mock.patch("huggingface_hub.snapshot_download", return_value=str(self.model)) as download:
            remote = CustomQwen3VLForConditionalGeneration.from_pretrained(
                "org/tiny-qwen", revision="a" * 40,
            )
            download.assert_called_once_with(repo_id="org/tiny-qwen", revision="a" * 40)
        self.assertEqual(set(remote.state_dict()), set(original.state_dict()))

        document = json.loads(index.read_text(encoding="utf-8"))
        first = next(iter(document["weight_map"]))
        document["weight_map"][first] = "../outside.safetensors"
        index.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "relative path"):
            CustomQwen3VLForConditionalGeneration.from_pretrained(self.model)


class MageDependencyTests(unittest.TestCase):
    def test_direct_pins_and_package_metadata_agree(self):
        vendor = VENDOR / "mage_flow"
        reference_spec = importlib.util.spec_from_file_location(
            "mage_reference_environment", ROOT / "scripts/release/mage_reference_environment.py"
        )
        reference = importlib.util.module_from_spec(reference_spec)
        reference_spec.loader.exec_module(reference)
        package = tomllib.loads((vendor / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = package["project"]["dependencies"]
        self.assertEqual(reference.REFERENCE_PACKAGES["transformers"], "5.10.4")
        self.assertEqual(reference.REFERENCE_PACKAGES["accelerate"], "1.13.0")
        for name, version in reference.REFERENCE_PACKAGES.items():
            pin = f"{name}=={version}"
            for manifest in ("requirements.txt", "requirements-oracles.in",
                             "requirements-oracles.txt"):
                manifest_pin = (
                    f"{name.replace('_', '-')}=={version}"
                    if manifest == "requirements-oracles.txt" else pin
                )
                self.assertIn(manifest_pin, (vendor / manifest).read_text(encoding="utf-8"))
            if name in ("transformers", "accelerate"):
                self.assertIn(pin, dependencies)

    def test_locked_transformers_rejects_chat_template_traversal(self):
        try:
            version = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("Transformers is not installed")
        if version != "5.10.4":
            self.skipTest("requires the locked Mage reference environment")
        from tokenizers import Tokenizer, models
        from transformers import PreTrainedTokenizerFast

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "tokenizer"
            output.mkdir()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
                unk_token="[UNK]",
            )
            tokenizer.chat_template = {"default": "{{ messages }}", "../escape": "{{ messages }}"}
            with self.assertRaisesRegex(ValueError, "Invalid chat template name"):
                tokenizer.save_pretrained(output)
            self.assertFalse((root / "escape.jinja").exists())


if __name__ == "__main__":
    unittest.main()
