"""No model compute: terminal pairing decisions and native receipt contracts."""

import hashlib
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zlib

from scripts.ci import ltx25_tier2_fixture as ltx
from scripts.ci import sga_terminal_comparison as comparison
from scripts.ci import wan_adapter_tier2_fixture as wan
from scripts.ci import z_image_tier2_fixture as z_image

UUID = "GPU-a123bcde-4321-abcd-abcd-0123456789ab"
REQUEST = {"selection": comparison.SELECTION, "profile": "ltx25", "variant": "dev",
           "cuda_device_uuid": UUID, "download_missing": True}


class TerminalTests(unittest.TestCase):
    def test_request_is_one_known_route_on_one_declared_device(self):
        self.assertIs(comparison.profile_for("candle-cuda", REQUEST), ltx)
        for request in (dict(REQUEST, variant="../nested"), dict(REQUEST, profile="../ltx25"),
                        dict(REQUEST, download_missing=1), dict(REQUEST, cuda_device_uuid="0"),
                        dict(REQUEST, unrelated="argv")):
            with self.assertRaises(ValueError):
                comparison.profile_for("candle-cuda", request)
        with self.assertRaises(ValueError):
            comparison.profile_for("mlx", REQUEST)
        request = {key: value for key, value in REQUEST.items() if key != "cuda_device_uuid"}
        with patch.dict(comparison.PROFILES, {("mlx", "ltx25"): "ltx25_tier2_fixture"}):
            self.assertIs(comparison.profile_for("mlx", dict(request, runner="nax")), ltx)
            with self.assertRaises(ValueError):
                comparison.profile_for("mlx", dict(request, runner="nax-primary"))
        mac_limits = comparison.sample_limits(ltx, dict(ltx.policy("dev", UUID), backend="darwin-mlx"))
        self.assertNotIn("cuda-pool.json", mac_limits)

    def test_pair_order_and_exact_decisions(self):
        base = {"identity": {"v": "same", "a": "same"}, "timing": {"e2e_ns": 1000}, "peaks": {"device": 1000}}
        samples = {name: base for name, _ in comparison.SAMPLES}
        self.assertEqual(comparison.summarize(samples)["status"], "pass")
        self.assertEqual(list(samples)[:6], ["warmup-base", "warmup-head", "pair-1-base", "pair-1-head", "pair-2-head", "pair-2-base"])
        samples["pair-4-head"] = dict(base, timing={"e2e_ns": 1021})
        self.assertEqual(comparison.summarize(samples)["status"], "inconclusive")
        samples["pair-4-head"] = dict(base, peaks={"device": 1021})
        self.assertEqual(comparison.summarize(samples)["status"], "fail")
        samples["pair-4-head"] = dict(base, identity={"v": "drift", "a": "same"})
        with self.assertRaisesRegex(ValueError, "output"):
            comparison.summarize(samples)
        del samples["pair-4-head"]
        with self.assertRaisesRegex(ValueError, "every frozen sample"):
            comparison.summarize(samples)

    def test_ltx_stages_adapter_route_and_decoded_content(self):
        for variant, stage1 in ltx.STAGE1.items():
            total = stage1 + 3
            case = ltx.CASE[variant]
            adapter_count = 1660 if variant == "dev" else 0
            row = {"case": case, "cuda_device_uuid": UUID, "e2e_ns": 1000,
                   "progress": [[index, total] for index in range(1, total + 1)],
                   "video": {"size": [512, 512], "frames": 17, "fps": 24, "rgb_sha256": "1" * 64},
                   "audio": {"channels": 2, "sample_rate": 24000, "samples": 48000, "f32_sha256": "2" * 64}}
            base, refine = ltx.SCHEDULE[variant], ltx.SCHEDULE["refinement"]
            log = (f"SGA_LTX_ADAPTER {adapter_count} 0\nSGA_LTX_STAGE 0 {stage1} 200 {base}\n"
                   f"SGA_LTX_STAGE 1 3 100 {refine}\nSGA_LTX_REAL_COMPLETE {case}\n")
            ltx.preflight(f"SGA_LTX_NATIVE_CASE {case} {base} {refine}\n", variant)
            with TemporaryDirectory() as directory:
                result = Path(directory) / "result.json"
                result.write_text(json.dumps(row), encoding="utf-8")
                observed = ltx.observe(log, result, UUID, variant)
                self.assertEqual(observed["identity"]["steps"], [stage1, 3])
                self.assertEqual(observed["timing"]["denoise_ns_per_outer_step"], 300 / total)
                for malformed in (log.replace(" 3 100", " 2 100"), log.replace(" 0\n", " 1\n", 1), log + log):
                    with self.assertRaises(ValueError):
                        ltx.observe(malformed, result, UUID, variant)
                result.write_text(json.dumps(dict(row, progress=[])), encoding="utf-8")
                with self.assertRaises(ValueError):
                    ltx.observe(log, result, UUID, variant)

    def test_wan_variant_and_packed_model_are_bound_to_backend(self):
        cuda = dict(REQUEST, profile="wan", variant="a14b-t2v-candle")
        mlx = {key: value for key, value in cuda.items() if key != "cuda_device_uuid"}
        mlx = dict(mlx, variant="a14b-t2v-mlx", runner="nax")
        self.assertIs(comparison.profile_for("candle-cuda", cuda), wan)
        self.assertIs(comparison.profile_for("mlx", mlx), wan)
        for backend, request, wrong in (("candle-cuda", cuda, mlx), ("mlx", mlx, cuda)):
            with self.assertRaises(ValueError):
                comparison.profile_for(backend, dict(request, variant=wrong["variant"]))
        self.assertEqual(wan.policy(cuda["variant"], UUID)["deadlineSeconds"], 240)
        self.assertNotEqual(wan.contract(cuda["variant"])["sources"]["model"],
                            wan.contract(mlx["variant"])["sources"]["model"])

    def test_wan_candle_receipt_requires_both_experts_to_install_adapters(self):
        variant = "a14b-t2v-candle"
        trace = [f"SGA_WAN_OUTER_STEP {i} {struct.unpack('<Q', struct.pack('<d', t))[0]} 1000"
                 for i, t in enumerate((1000.0, 980.0, 925.0, 807.0), 1)]
        row = {"device": {"kind": "candle-cuda", "index": 0}, "cuda_device_uuid": UUID,
               "end_to_end_ns": 12000, "frame_rgb8_sha256": [f"{i:064x}" for i in range(17)],
               "progress": ["renderer", "1/4", "2/4", "3/4", "renderer", "4/4", "renderer", "decode"]}
        with TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            result.write_text(json.dumps(row), encoding="utf-8")
            log = "\n".join(["SGA_WAN_EXPERT_ADAPTERS High 8", *trace[:3], "SGA_WAN_EXPERT_ADAPTERS Low 8",
                             trace[3], "SGA_WAN_REAL_COMPLETE " + wan.CASE[variant]])
            sample = wan.observe(log, result, UUID, variant)
            self.assertEqual(sample["identity"]["adapters"], [{"expert": "High", "applied": 8},
                                                              {"expert": "Low", "applied": 8}])
            self.assertEqual(sample["timing"]["outer_step_ns"], 1000)
            for wrong in (log.replace("Low 8", "Low 0"), log.replace("Low 8", "High 8"),
                          log.replace("SGA_WAN_EXPERT_ADAPTERS Low 8\n" + trace[3],
                                      trace[3] + "\nSGA_WAN_EXPERT_ADAPTERS Low 8"),
                          log.replace(trace[-1].split()[2], trace[0].split()[2])):
                with self.assertRaises(ValueError):
                    wan.observe(wrong, result, UUID, variant)

    def test_z_image_archives_only_verified_png_with_exact_rgb_pixels(self):
        with TemporaryDirectory() as directory:
            work = Path(directory) / "work"
            archive = Path(directory) / "evidence"
            archive.mkdir()
            for variant, size in z_image.SIZE.items():
                with self.subTest(variant=variant):
                    sample = work / variant
                    sample.mkdir(parents=True)
                    pixels = bytes(range(256)) * (size * size * 3 // 256)
                    (sample / "image.rgb").write_bytes(pixels)
                    z_image.prepare_artifacts(sample, work, variant)
                    steps = z_image.STEPS[variant]
                    sigmas = [0x3F800000 - index * 65536 for index in range(steps)] + [0]
                    row = {"plan": dict(z_image.request(variant), sigmas_bits=sigmas),
                           "sample": {"dtype": "BF16", "device": "candle-cuda:0", "sigmas_bits": sigmas,
                                      "latent_shape": [1, 16, 1, size // 8, size // 8]},
                           "progress": list(range(1, steps + 1)), "e2e_ns": 1000, "denoise_ns": 600,
                           "phase_ns": {"provider_load": 100, "render_before_denoise": 150,
                                        "after_denoise_before_decode": 10, "decode_through_host_rgb": 140}}
                    result = sample / "result.json"
                    result.write_text(json.dumps(row), encoding="utf-8")
                    log = "SGA_Z_IMAGE_TIER2_BODY " + variant
                    (sample / "stdout.log").write_text(log, encoding="utf-8")
                    limits = comparison.sample_limits(z_image, z_image.policy(variant, UUID))
                    rejected = dict(comparison.collect_sample(variant + "-failed", sample, limits,
                                                              archive, work, skip=z_image.ARTIFACT_LIMITS))
                    self.assertIn(variant + "-failed.stdout.log", rejected)
                    self.assertNotIn(variant + "-failed.image.png", rejected)
                    sealed = dict(comparison.collect_sample(variant, sample, limits, archive, work))
                    self.assertIn(variant + ".image.png", sealed)
                    self.assertNotIn(variant + ".image.rgb", sealed)
                    original = z_image.observe(log, result, UUID, variant)
                    replay = z_image.observe(log, archive / (variant + ".result.json"), UUID, variant)
                    self.assertEqual(original, replay)
                    row["phase_ns"]["provider_load"] += 1
                    result.write_text(json.dumps(row), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "render phases do not account"):
                        z_image.observe(log, result, UUID, variant)
                    self.assertEqual(replay["identity"]["rgb_sha256"], hashlib.sha256(pixels).hexdigest())

    def test_z_image_refuses_corrupt_or_oversized_png_data(self):
        with TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            good = z_image.png(1, 1, b"abc")
            image.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "checksum"):
                z_image.read_artifact_pixels(image, 1)
            for compressed in (zlib.compress(b"\0abcd"), zlib.compress(b"\0abc") + zlib.compress(b"extra")):
                kind = b"IDAT"
                idat = struct.pack(">I", len(compressed)) + kind + compressed
                idat += struct.pack(">I", zlib.crc32(kind + compressed))
                image.write_bytes(good[:33] + idat + good[-12:])
                with self.assertRaisesRegex(ValueError, "exact unfiltered RGB8"):
                    z_image.read_artifact_pixels(image, 1)

    def test_mlx_terminal_uses_configured_cache_without_implicit_fallback(self):
        with TemporaryDirectory() as directory, patch.object(comparison, "hub_root") as generic:
            cache = Path(directory) / "fleet"
            env = {"HF_HUB_CACHE": str(cache)}
            self.assertEqual(comparison.terminal_hub_root("mlx", ltx, env), cache.resolve())
            unavailable = Path(directory) / "ordinary-file"
            unavailable.touch()
            with self.assertRaises(FileExistsError):
                comparison.terminal_hub_root("mlx", ltx, {"HF_HUB_CACHE": str(unavailable)})
            generic.assert_not_called()


if __name__ == "__main__":
    unittest.main()
