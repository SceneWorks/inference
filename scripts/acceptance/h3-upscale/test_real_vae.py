"""Mutation checks for independent source/guide parity assertions (CPU only)."""
import unittest
import tempfile
import json
import contextlib
import io
from pathlib import Path
from safetensors.torch import save_file
import torch
from compare_real_vae import CAPTURES, COMFY_PIN, TOLERANCES, compare_captures, published_to_comfy, compare_saved_reference, sha256


class CaptureComparisonTests(unittest.TestCase):
    def setUp(self):
        self.expected = {name: torch.linspace(-2, 2, 96).reshape(1, 24, 1, 2, 2)
                         for name in CAPTURES}

    def test_exact_and_small_backend_errors_pass(self):
        actual = {k: v + .001 for k, v in self.expected.items()}
        self.assertTrue(all(v["pass"] for v in compare_captures(self.expected, actual).values()))

    def test_each_native_capture_mutation_fails_independently(self):
        for name in CAPTURES:
            with self.subTest(capture=name):
                actual = {k: v.clone() for k, v in self.expected.items()}
                actual[name].flatten()[7] += .5
                result = compare_captures(self.expected, actual)
                self.assertFalse(result[name]["pass"])
                self.assertTrue(all(v["pass"] for k, v in result.items() if k != name))

    def test_shared_native_error_cannot_become_its_own_reference(self):
        changed = {k: v + .5 for k, v in self.expected.items()}
        # The original blind spot: inheriting the native input on both sides
        # would accept the same error. Independent RGB-derived expectations do not.
        self.assertTrue(all(v["pass"] for v in compare_captures(changed, changed).values()))
        self.assertTrue(all(not v["pass"] for v in compare_captures(self.expected, changed).values()))

    def test_relative_limit_is_required_even_when_absolute_passes(self):
        expected = {k: v / 100 for k, v in self.expected.items()}
        actual = {k: v + .01 for k, v in expected.items()}
        self.assertTrue(all(not v["pass"] for v in compare_captures(expected, actual).values()))

    def test_shape_nonfinite_and_missing_captures_refuse(self):
        for mutation in ("shape", "nonfinite", "missing"):
            actual = {k: v.clone() for k, v in self.expected.items()}
            if mutation == "shape": actual["guide.normalized"] = actual["guide.normalized"].flatten()
            elif mutation == "nonfinite": actual["source.normalized"].flatten()[0] = float("nan")
            else: del actual["guide.normalized"]
            with self.assertRaises((ValueError, KeyError)):
                compare_captures(self.expected, actual)

    def test_saved_reference_binds_rgb_and_export_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); reference = root / "reference"; reference.mkdir()
            evidence = root / "native"; evidence.mkdir()
            for case in ("h3", "other-model", "live-action"):
                folder = evidence / case; folder.mkdir()
                rgb = folder / "source.rgb"; rgb.write_bytes(b"independent source RGB")
                exported = reference / (case + ".safetensors")
                save_file(self.expected, str(exported))
                save_file(self.expected, str(folder / "guided.intermediates.safetensors"))
                metadata = {"comfy_commit": COMFY_PIN, "tolerances": TOLERANCES,
                            "reference_sha256": sha256(exported), "rgb_sha256": sha256(rgb)}
                (reference / (case + ".json")).write_text(json.dumps(metadata))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(compare_saved_reference(reference, evidence, root / "good"))
            rgb = evidence / "h3/source.rgb"; rgb.write_bytes(b"different source RGB")
            with self.assertRaisesRegex(ValueError, "RGB inputs differ"):
                compare_saved_reference(reference, evidence, root / "bad-rgb")
            rgb.write_bytes(b"independent source RGB")
            exported = reference / "h3.safetensors"; exported.write_bytes(b"changed export")
            with self.assertRaisesRegex(ValueError, "reference tensors"):
                compare_saved_reference(reference, evidence, root / "bad-export")

    def test_published_projection_transform_interleaves_heads_and_swaps_gate(self):
        state = {"decoder.proj_in." + suffix: torch.ones(2) for suffix in ("weight", "bias")}
        state["encoder.down_blocks.2.resnets.0.conv_shortcut.weight"] = torch.tensor([7.])
        state["encoder.down_blocks.1.downsamplers.0.conv.weight"] = torch.tensor([8.])
        for block in range(36):
            prefix = f"decoder.transformer_blocks.{block}."
            for suffix in ("weight", "bias"):
                for index, part in enumerate(("q", "k", "v")):
                    state[prefix + f"attn.to_{part}.{suffix}"] = torch.arange(2048.) + index * 10000
                for name in ("attn.to_out.0", "ff.net.2"):
                    state[prefix + name + "." + suffix] = torch.tensor([9.])
                state[prefix + "ff.net.0.proj." + suffix] = torch.tensor([1., 2., 3., 4.])
        converted = published_to_comfy(state)
        for block in range(36):
            prefix = f"decoder.transformer_blocks.{block}."
            for suffix in ("weight", "bias"):
                fused = converted[prefix + "attn.to_qkv." + suffix].reshape(32, 3, 64)
                for index, part in enumerate(("q", "k", "v")):
                    self.assertTrue(torch.equal(fused[:, index].flatten(), state[prefix + f"attn.to_{part}.{suffix}"]))
                self.assertTrue(torch.equal(converted[prefix + "ff.w1." + suffix], torch.tensor([3., 4., 1., 2.])))
        self.assertEqual(float(converted["encoder.down.2.block.0.nin_shortcut.weight"]), 7.)
        self.assertEqual(float(converted["encoder.down.1.downsample.conv.weight"]), 8.)


if __name__ == "__main__": unittest.main()
