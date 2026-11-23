"""Header, routing and layout checks for the Mochi/SVD proof. No learned execution."""
import hashlib
import itertools
import json
from pathlib import Path
import struct
import unittest

from scripts.ci.ms_pair import fixture


def unpack(row):
    _, _, blocks = row
    return [v for v, in struct.iter_unpack("<f", b"".join(blocks))]


class VideoFixtureTests(unittest.TestCase):
    def test_source_depth_and_upstream_skip_widths(self):
        mochi = fixture._header("mochi")["geometry"]["transformer/model.safetensors"]
        self.assertIn("transformer_blocks.0.attn1.to_add_out.weight", mochi)
        self.assertNotIn("transformer_blocks.1.attn1.to_add_out.weight", mochi)
        self.assertIn("transformer_blocks.1.norm1_context.linear_1.weight", mochi)
        self.assertEqual(fixture._retained("transformer_blocks.47.attn1.to_k.weight", True),
                         "transformer_blocks.1.attn1.to_k.weight")
        for variant, dtype in (("mlx-svd", "F16"), ("cuda-svd", "F32")):
            rows, = fixture._header(variant)["geometry"].values()
            for stage, (out, first, last) in enumerate(((1280, 2560, 2560), (1280, 2560, 1920),
                                                       (640, 1920, 960), (320, 960, 640))):
                for layer, width in enumerate((first, last)):
                    key = f"up_blocks.{stage}.resnets.{layer}.spatial_res_block.conv1.weight"
                    self.assertEqual(rows[key], {"dtype": dtype, "shape": [out, width, 3, 3]})
        aux = json.loads((fixture.HERE / "aux_shapes.json").read_text(encoding="utf-8"))
        self.assertEqual(aux["image_encoder"]["vision_model.encoder.layers.31.mlp.fc1.weight"], [5120, 1280])
        self.assertEqual(aux["vae"]["quant_conv.weight"], [8, 8, 1, 1])

    def test_svd_inputs_retain_frame_channel_and_pixel_coordinates(self):
        candle, mlx = [fixture._inputs(k) for k in ("cuda-svd", "mlx-svd")]
        self.assertEqual(unpack(candle["image_embeds"]), unpack(mlx["image_embeds"]))
        for key in ("noise", "image_latents"):
            cf, cl = unpack(candle[key]), unpack(mlx[key])
            for f, c, h, w in itertools.product(range(3), range(4), range(8), range(16)):
                self.assertEqual(cf[((f*4+c)*8+h)*16+w], cl[((f*8+h)*16+w)*4+c])
            self.assertEqual(cf[:512] == cf[512:1024], key == "image_latents")
        self.assertNotEqual(unpack(candle["noise"]), unpack(candle["image_latents"]))

    def test_mochi_condition_rows_are_distinct_and_masked(self):
        inputs = fixture._inputs("mochi")
        self.assertEqual(unpack(inputs["mask"]), [1.] * 13 + [0.] * 243 + [1.] * 37 + [0.] * 219)
        rows = inputs["context"][2]
        first = next(rows)
        positive = next(itertools.islice(rows, 255, None))
        self.assertNotEqual(hashlib.sha256(first).digest(), hashlib.sha256(positive).digest())
        self.assertNotEqual(positive, bytes(len(positive)))

    def test_cases_bind_schedule_endpoints_and_actual_evaluation_events(self):
        for name, variant in fixture.CASES.items():
            spec = fixture.case_spec(name)
            settings = spec["settings"]
            self.assertEqual(spec["outer_steps"], 4)
            self.assertEqual(len(spec["trace"]), 14 if name == "mlx-svd-heun" else 7 if "heun" in name else 4)
            if variant == "mochi":
                self.assertEqual(settings["sigma_bits"][::4], [0, fixture._bits(1)])
            else:
                self.assertEqual(settings["sigma_bits"][::4], [fixture._bits(700), 0])
            if name == "mlx-svd-heun":
                self.assertEqual([e["sigma_bits"] for e in spec["trace"] if e["event"] == "evaluate"],
                                 [settings["sigma_bits"][i] for i in (0,1,1,2,2,3,3)])

    def test_base_and_candidate_share_the_same_registered_public_denoise_probe(self):
        root = Path(__file__).resolve().parents[2]
        for backend, model in itertools.product(("mlx", "candle-cuda"), ("mochi", "svd")):
            render = lambda arm: fixture.overlay_source("candidate", backend, model, arm,
                                                         lambda revision, path: (root / path).read_bytes())
            blob = render("base")
            self.assertEqual(blob, render("head"))
            self.assertEqual(blob.count(b'#[ignore = "model inference; explicit model-proof workflow"]'), 1)
            self.assertNotIn(b"@SVD_AUX_SHAPES@", blob)
            self.assertNotIn(b"unsafe", blob)
