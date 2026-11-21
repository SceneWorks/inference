import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

from scripts.ci.wan_pair import fixture


class WanFixtureTest(unittest.TestCase):
    def test_retains_one_production_width_block_and_matching_experts(self):
        for variant in set(fixture.CASES.values()):
            with self.subTest(variant=variant):
                entry = fixture._header(variant)
                self.assertRegex(entry["source"]["snapshot"], r"^[a-f0-9]{40}$")
                for digest in entry["source"]["header_sha256"].values():
                    self.assertRegex(digest, r"^[a-f0-9]{64}$")
                experts = list(entry["geometry"].values())
                self.assertEqual(len(experts), 2 if "a14b" in variant else 1)
                self.assertEqual(experts[0], experts[-1])
                for tensors in experts:
                    self.assertEqual(len(tensors), 42)
                    self.assertTrue(all(not k.startswith("blocks.") or k.startswith("blocks.0.")
                                        for k in tensors))
                    mlx = variant.startswith("mlx")
                    dtype = "BF16" if mlx else "F32"
                    self.assertEqual({row["dtype"] for row in tensors.values()}, {dtype})
                    dim, ffn = (5120, 13824) if "a14b" in variant else (3072, 14336)
                    channels = 36 if "a14b-i2v" in variant else 16 if "a14b" in variant else 48
                    patch = "patch_embedding_proj.weight" if mlx else "patch_embedding.weight"
                    shape = [dim, channels * 4] if mlx else [dim, channels, 1, 2, 2]
                    self.assertEqual(tensors[patch]["shape"], shape)
                    ffn_key = "blocks.0.ffn.fc1.weight" if mlx else "blocks.0.ffn.net.0.proj.weight"
                    self.assertEqual(tensors[ffn_key]["shape"], [ffn, dim])

    def test_corrector_first_crosses_each_expert_boundary(self):
        for case, boundary, expected in [
            ("mlx-a14b-t2v-resident-heun", 875, [999, 972, 972, 922, 922, 799, 799]),
            ("mlx-a14b-i2v-swapped-heun", 900, [999, 937, 937, 833, 833, 624, 624]),
        ]:
            with self.subTest(case=case):
                trace = fixture.case_spec(case)["trace"]
                bits = [e["sigma_bits"] for e in trace if e["event"] == "evaluate"]
                timesteps = [int(struct.unpack("<f", struct.pack("<I", x))[0] * 1000) for x in bits]
                self.assertEqual(timesteps, expected)
                first_low = next(i for i, t in enumerate(timesteps) if t < boundary)
                self.assertEqual(first_low % 2, 1)
                if "swapped" in case:
                    loads = [i for i, e in enumerate(trace) if e["event"] == "load_expert"]
                    self.assertEqual(len(loads), 2)
                    self.assertEqual(trace[loads[1] - 1]["evaluation"], first_low)

    def test_preserves_storage_conversion_and_writes_exact_offsets(self):
        shape = [7, 11]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "weights.safetensors"
            tensors = {}
            for dtype in ("F32", "BF16"):
                meta = {"dtype": dtype, "shape": shape}
                tensors[dtype] = (dtype, shape, fixture._weight_blocks("probe", meta))
            fixture._write_tensors(output, tensors)
            with output.open("rb") as source:
                size, = struct.unpack("<Q", source.read(8))
                header = json.loads(source.read(size))
                count = math.prod(shape)
                self.assertEqual(header["F32"]["data_offsets"], [0, 4 * count])
                self.assertEqual(header["BF16"]["data_offsets"], [4 * count, 6 * count])
                f32, bf16 = source.read(4 * count), source.read()
            self.assertEqual(len(bf16), count * 2)
            for i, (value,) in enumerate(struct.iter_unpack("<f", f32)):
                self.assertTrue(math.isfinite(value) and value != 0)
                self.assertEqual(f32[4 * i + 2:4 * i + 4], bf16[2 * i:2 * i + 2])
