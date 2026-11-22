"""No model compute: terminal pairing decisions and native receipt contracts."""

from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

from scripts.ci import ltx25_tier2_fixture as ltx
from scripts.ci import ltx25_tier2_mlx as ltx_mlx
from scripts.ci import sga_terminal_comparison as comparison
from scripts.ci import wan_adapter_tier2_fixture as wan
from scripts.ci import z_image_tier2_fixture as z_image
from scripts.ci.pinned_hub_source import file_inventory_seal

UUID = "GPU-a123bcde-4321-abcd-abcd-0123456789ab"
REQUEST = {"selection": comparison.SELECTION, "profile": "ltx25", "variant": "dev",
           "cuda_device_uuid": UUID, "download_missing": True}
DIAGNOSTIC = dict(REQUEST, cuda_device_uuid=comparison.WINDOWS_DEV_GPU, diagnostic="windows-dev-memory")


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

    def test_windows_diagnostic_selector_and_plan_pin_only_the_repaired_base(self):
        self.assertIs(comparison.profile_for("candle-cuda", DIAGNOSTIC), ltx)
        for request in (dict(DIAGNOSTIC, cuda_device_uuid=UUID), dict(DIAGNOSTIC, variant="distilled"),
                        dict(DIAGNOSTIC, profile="z-image"), dict(DIAGNOSTIC, diagnostic=True),
                        dict(DIAGNOSTIC, diagnostic="unknown"), dict(DIAGNOSTIC, download_missing=False)):
            with self.assertRaises(ValueError):
                comparison.profile_for("candle-cuda", request)
        mlx = {key: value for key, value in DIAGNOSTIC.items() if key != "cuda_device_uuid"}
        with self.assertRaises(ValueError):
            comparison.profile_for("mlx", dict(mlx, runner="nax"))

        controller = {"revision": "1" * 40, "tree": "2" * 40}
        repair = {"adapter.rs": b"repaired"}

        def make_overlay(read, arm, backend, binary, *, dev_loader, diagnostic):
            self.assertEqual((arm, backend, diagnostic), ("base", "candle-cuda", True))
            self.assertEqual(dev_loader, repair)
            return {"base.rs": read("base.rs")}

        def git_show(command, **_kwargs):
            self.assertEqual(command[:2], ["git", "show"])
            revision, path = command[2].split(":")
            expected = comparison.BASE if path == "base.rs" else comparison.WINDOWS_DEV_REPAIR
            self.assertEqual(revision, expected)
            return b"frozen base" if path == "base.rs" else repair[path]

        with (patch.object(comparison.proof, "repository_identity", return_value=controller),
              patch.object(comparison.subprocess, "run", return_value=SimpleNamespace(returncode=0)),
              patch.object(comparison.subprocess, "check_output", side_effect=git_show),
              patch.object(comparison.paired, "git", return_value="3" * 40),
              patch.object(ltx, "dev_loader_sources", side_effect=lambda read: {"adapter.rs": read("adapter.rs")}),
              patch.object(ltx, "overlay", side_effect=make_overlay)):
            contract = comparison.plan("candle-cuda", DIAGNOSTIC, controller["revision"])
        self.assertEqual(set(contract["overlay_sha256"]), {"base"})
        self.assertEqual(contract["expected_samples"], ["warmup-base"])
        self.assertEqual(contract["base"], {"revision": comparison.BASE, "tree": "3" * 40})
        self.assertEqual(contract["diagnostic_source"], controller)
        self.assertNotIn("head", contract)
        self.assertNotIn("comparison", contract)
        self.assertEqual(contract["baseline_loader_repair"]["source_revision"], comparison.WINDOWS_DEV_REPAIR)
        self.assertEqual(contract["baseline_loader_repair"]["shared_source_sha256"],
                         comparison.source_pair.overlay_identity(repair))
        self.assertEqual(contract["model"], ltx.contract("dev")["model"])
        self.assertEqual(contract["settings"], ltx.contract("dev")["settings"])
        self.assertEqual(contract["policy"], ltx.policy("dev", DIAGNOSTIC["cuda_device_uuid"]))

    def test_windows_diagnostic_runs_one_warmup_and_archives_bounded_failure_without_scoring(self):
        policy = ltx.policy("dev", DIAGNOSTIC["cuda_device_uuid"])
        base = {"revision": comparison.BASE, "tree": "3" * 40}
        build = dict(base, overlay_sha256={"lib.rs": "4" * 64}, binary_sha256="5" * 64)
        contract = {"backend": "candle-cuda", "request": DIAGNOSTIC, "base": base,
                    "diagnostic_source": {"revision": "1" * 40, "tree": "2" * 40},
                    "policy": policy, "expected_samples": ["warmup-base"],
                    "overlay_sha256": {"base": build["overlay_sha256"]},
                    "settings": ltx.contract("dev")["settings"]}
        for fails in (False, True):
            with self.subTest(fails=fails), TemporaryDirectory() as directory, ExitStack() as stack:
                tmp = Path(directory).resolve()
                args = SimpleNamespace(backend="candle-cuda", output=tmp / "artifact")
                env = {"RUNNER_TEMP": str(tmp), "GITHUB_RUN_ID": "1", "GITHUB_RUN_ATTEMPT": "1",
                       "MODEL_PROOF_RUNNER_ENVIRONMENT": "self-hosted"}
                stack.enter_context(patch.dict(comparison.os.environ, env))

                @contextmanager
                def claim(*_args, **_kwargs):
                    yield {"post_execution_verified": True}

                def invoke(_binary, _arm_root, output, *_args, **_kwargs):
                    output.mkdir()
                    (output / "stdout.log").write_text("SGA_LTX_MEMORY_PHASE adapter-start 1000\n", encoding="utf-8")
                    if fails:
                        raise ValueError("child-footprint: owned tree exceeded child cap")

                replacements = [
                    (comparison.proof, "require_host", None),
                    (comparison.source_pair, "attempt_work_root", tmp),
                    (comparison.paired, "git", comparison.BASE),
                    (comparison.paired, "hardware", {}),
                    (comparison, "require_sm120", "sm_120"),
                    (comparison, "terminal_hub_root", tmp),
                    (ltx, "staged_assets", (tmp / "snapshot", {"fixture": True})),
                    (comparison, "verify_assets", None),
                    (comparison, "terminal_overlay", (None, {})),
                    (comparison.source_pair, "build", (Path("base.exe"), build.copy())),
                    (comparison.subprocess, "run", SimpleNamespace(stdout="preflight")),
                    (ltx, "preflight", None),
                    (comparison, "sample_limits", {"stdout.log": 1024}),
                    (comparison, "read_sample", {}),
                ]
                mocks = {name: stack.enter_context(patch.object(owner, name, return_value=value))
                         for owner, name, value in replacements}
                stack.enter_context(patch.object(comparison, "execution_claim", claim))
                warmup = stack.enter_context(patch.object(comparison.paired, "invoke", side_effect=invoke))
                stack.enter_context(patch.object(comparison, "summarize", side_effect=AssertionError("must not score")))
                if fails:
                    with self.assertRaisesRegex(ValueError, "child-footprint"):
                        comparison.execute(args, contract, ltx)
                else:
                    comparison.execute(args, contract, ltx)
                mocks["build"].assert_called_once()
                self.assertEqual(mocks["build"].call_args.args[2], "base")
                mocks["run"].assert_called_once()
                warmup.assert_called_once()
                self.assertEqual(warmup.call_args.args[2].name, "warmup-base")
                receipt_path = args.output / "receipt.json"
                receipt = comparison.load_json(receipt_path)
                self.assertFalse(receipt["accepted"])
                self.assertEqual(receipt["diagnostic_completed"], not fails)
                self.assertEqual(set(receipt["files"]), {"warmup-base.stdout.log"})
                self.assertNotIn("comparison_result", receipt)
                comparison.verify(args, contract, ltx)
                for invalid in (dict(receipt, accepted=True), dict(receipt, comparison_result={}),
                                dict(receipt, arms={"head": build})):
                    receipt_path.write_text(json.dumps(invalid), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "unscored and baseline-only"):
                        comparison.verify(args, contract, ltx)
                wrong_base = dict(receipt, arms={"base": dict(build, revision="f" * 40)})
                receipt_path.write_text(json.dumps(wrong_base), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "exact compiled source"):
                    comparison.verify(args, contract, ltx)

    def test_accepted_uploads_replay_the_frozen_assets_but_failure_archives_may_omit_them(self):
        sample = {"identity": {"fixture": "same"}, "timing": {"e2e_ns": 1000}, "peaks": {"device": 1000}}
        decision = comparison.summarize({name: sample for name, _ in comparison.SAMPLES})
        toolchain = {"rustc_version": "rustc fixture", "cargo_version": "cargo fixture"}
        arm = {"overlay_sha256": "fixture", "binary_sha256": "a" * 64,
               "preflight_stdout": "fixture", "toolchain": toolchain}
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            args = SimpleNamespace(output=root)
            receipt_path = root / "receipt.json"
            env = {"RUNNER_TEMP": str(root), "GITHUB_RUN_ID": "1", "GITHUB_RUN_ATTEMPT": "1"}
            with (patch.dict(comparison.os.environ, env), patch.object(comparison, "sample_limits", return_value={}),
                  patch.object(comparison, "read_sample", return_value=sample)):
                for profile in (ltx, ltx_mlx, z_image):
                    for variant in profile.CASE:
                        with self.subTest(profile=profile.__name__, variant=variant):
                            contract = {**profile.contract(variant), "base": {}, "head": {}, "policy": {},
                                        "request": {"variant": variant},
                                        "overlay_sha256": {arm: "fixture" for arm in comparison.ARMS}}
                            files = [{"path": "model_index.json", "bytes": 10, "kind": "git_blob_sha1", "digest": "a" * 40},
                                     {"path": "transformer/model.safetensors", "bytes": 20,
                                      "kind": "lfs_sha256", "digest": "b" * 64}]
                            if variant == "dev":
                                files.insert(0, {"path": ltx.ADAPTER, "bytes": 30,
                                                 "kind": "lfs_sha256", "digest": ltx.ADAPTER_SHA256})
                            seal = file_inventory_seal(files)
                            model = contract["model"]
                            # Synthetic file seals exercise replay without asset downloads or model compute.
                            model.update(files_sha256=seal, asset_bytes=sum(row["bytes"] for row in files))
                            assets = {"repository": model["repository"], "revision": model["revision"],
                                      "files_sha256": seal, "files": files}
                            receipt = {**contract, "run_id": "1", "run_attempt": "1", "accepted": True,
                                       "assets": assets, "files": {}, "arms": {name: arm for name in comparison.ARMS},
                                       "comparison_result": decision, "physical_claim": {"post_execution_verified": True}}
                            with patch.object(profile, "preflight", return_value=None):
                                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                                comparison.verify(args, contract, profile)
                                forged_rows = deepcopy(files)
                                forged_rows[-1]["digest"] = "c" * 64
                                wrong = (None, dict(assets, repository="example/other"), dict(assets, revision="0" * 40),
                                         dict(assets, files=files[:-1]), dict(assets, files=files + files[-1:]),
                                         dict(assets, files=list(reversed(files))),
                                         dict(assets, files=[dict(files[0], bytes=0), *files[1:]]),
                                         dict(assets, files=forged_rows),
                                         dict(assets, files=forged_rows, files_sha256=file_inventory_seal(forged_rows)))
                                for forged in wrong:
                                    receipt_path.write_text(json.dumps(dict(receipt, assets=forged)), encoding="utf-8")
                                    with self.assertRaisesRegex(ValueError, "terminal asset"):
                                        comparison.verify(args, contract, profile)
                                del receipt["assets"]
                                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                                with self.assertRaisesRegex(ValueError, "terminal asset"):
                                    comparison.verify(args, contract, profile)
                                receipt_path.write_text(json.dumps(dict(receipt, accepted=False)), encoding="utf-8")
                                comparison.verify(args, contract, profile)

    def test_dev_adapter_sha_is_checked_independently(self):
        for profile in (ltx, ltx_mlx):
            contract = profile.contract("dev")
            model = contract["model"]
            files = [{"path": ltx.ADAPTER, "bytes": 42, "kind": "lfs_sha256", "digest": "a" * 64}]
            model.update(files_sha256=file_inventory_seal(files), asset_bytes=42)
            assets = {"repository": model["repository"], "revision": model["revision"],
                      "files_sha256": model["files_sha256"], "files": files}
            with self.assertRaisesRegex(ValueError, "frozen refinement adapter SHA-256"):
                comparison.verify_assets(contract, assets)

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

    def test_ltx_dev_baseline_applies_only_the_bound_loader_sources(self):
        source = ltx.SOURCE
        shared = {f"{source}/{filename}": f"repaired {filename}".encode() for filename in ltx.DEV_LOADER_FILES}
        originals = {f"{source}/{filename}": filename.encode() for filename in ltx.DEV_LOADER_BASE_SHA256}
        frozen_sha = {Path(path).name: hashlib.sha256(data).hexdigest() for path, data in originals.items()}
        lib = f"{source}/lib.rs"
        other_markers = b"\n".join((
            b"        let (vlat, alat) = if self.transformer_variant.is_dev() {",
            b"        // The denoised half-resolution latent enters the learned upsampler",
            b"        let stage2 = if stage2_keyframes.is_empty() {",
            b"        let (vlat, alat) = (stage2.video, stage2.audio);",
        ))
        metadata = {f"{source}/quant_measurement.rs": b"\nfn terminal_generation_request("}
        old_cancel = b"\n".join(original for original, _ in ltx.DEV_CANCELLATION_EDITS)
        new_cancel = b"\n".join(repaired for _, repaired in ltx.DEV_CANCELLATION_EDITS)
        base = {**originals, **metadata, lib: b"\n".join((ltx.ORIGINAL_INSTALL, other_markers, old_cancel, b"// frozen scales"))}
        head = {**shared, **metadata, lib: b"\n".join((ltx.REPAIRED_INSTALL, other_markers, new_cancel, b"// request scopes"))}
        with patch.dict(ltx.DEV_LOADER_BASE_SHA256, frozen_sha, clear=True):
            base_overlay = ltx.overlay(base.__getitem__, "base", "candle-cuda", "test", dev_loader=shared)
            head_overlay = ltx.overlay(head.__getitem__, "head", "candle-cuda", "test", dev_loader=shared)
            self.assertEqual({path: base_overlay[path] for path in shared}, shared)
            self.assertTrue(head_overlay.keys().isdisjoint(shared))
            for marker, instrumented in ((b"// frozen scales", base_overlay), (b"// request scopes", head_overlay)):
                self.assertIn(marker, instrumented[lib])
                self.assertEqual(instrumented[lib].count(b"SGA_LTX_ADAPTER"), 1)
                self.assertIn(b"let sga_adapters = if self.transformer_variant.is_dev()", instrumented[lib])
                self.assertIn(b"adapters::install_ltx25_dev_adapters(&mut avdit, adapters, cancel)?\n", instrumented[lib])
                self.assertIn(new_cancel, instrumented[lib])
            drift_path = f"{source}/adapters.rs"
            base[drift_path] = b"old source drift"
            with self.assertRaisesRegex(ValueError, "unexpected original LTX Dev loader"):
                ltx.overlay(base.__getitem__, "base", "candle-cuda", "test", dev_loader=shared)
            head[drift_path] = b"candidate drift"
            with self.assertRaisesRegex(ValueError, "candidate LTX Dev loader differs"):
                ltx.overlay(head.__getitem__, "head", "candle-cuda", "test", dev_loader=shared)

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
        self.assertEqual(wan.contract(cuda["variant"])["settings"]["frames"], 17)
        self.assertNotIn("native_decoded_frames", wan.contract(cuda["variant"])["settings"])
        self.assertEqual(wan.contract(mlx["variant"])["settings"]["frames"], 9)
        self.assertEqual(wan.contract(mlx["variant"])["settings"]["native_decoded_frames"], 12)
        for request in (cuda, mlx):
            contract = wan.contract(request["variant"])
            comparison.verify_assets(contract, contract["expected_asset_identity"])
            with self.assertRaisesRegex(ValueError, "frozen asset content"):
                comparison.verify_assets(contract, {})

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

    def test_wan_mlx_receipt_compares_all_native_decoded_frames(self):
        variant = "a14b-t2v-mlx"
        # The native MLX warmup completed this exact four-step schedule before the old output check.
        trace = [f"SGA_WAN_OUTER_STEP {i} {bits} 1000"
                 for i, bits in enumerate((1148829696, 1148387328, 1147568128, 1145552896), 1)]
        log = "\n".join([*trace, "SGA_WAN_REAL_COMPLETE " + wan.CASE[variant]])
        digests = [f"{i:064x}" for i in range(12)]
        row = {"device": {"kind": "mlx-metal", "index": 0}, "mlx_peak_bytes": 30000,
               "end_to_end_ns": 12000, "frame_rgb8_sha256": digests,
               "progress": ["renderer", "1/4", "2/4", "3/4", "renderer", "4/4", "decode"],
               "adapters": [{"file": expert.lower() + "_noise_model.safetensors",
                             "expert": f"Some({expert})", "scale": 1.0, "applied": 8, "skipped": []}
                            for expert in ("High", "Low")]}
        with TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            result.write_text(json.dumps(row), encoding="utf-8")
            full = wan.observe(log, result, "nax", variant)
            self.assertEqual(full["identity"]["frame_rgb8_sha256"], digests)

            for count in (9, 11, 13):
                altered = (digests + ["f" * 64])[:count]
                result.write_text(json.dumps(dict(row, frame_rgb8_sha256=altered)), encoding="utf-8")
                with self.subTest(count=count), self.assertRaisesRegex(ValueError, "frozen decoded clip"):
                    wan.observe(log, result, "nax", variant)

            altered = [*digests[:-1], "f" * 64]
            result.write_text(json.dumps(dict(row, frame_rgb8_sha256=altered)), encoding="utf-8")
            samples = {name: full for name, _ in comparison.SAMPLES}
            samples["pair-6-head"] = wan.observe(log, result, "nax", variant)
            with self.assertRaisesRegex(ValueError, "terminal output"):
                comparison.summarize(samples)

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
