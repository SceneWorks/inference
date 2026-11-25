"""Portable comparison-harness checks. No model, backend or Cargo executable is run."""
from contextlib import ExitStack, contextmanager, redirect_stderr
import copy
import hashlib
from io import StringIO
import json
import math
import os
import stat
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.ci.flux1_pair import evidence, fixture, overlay, performance
from scripts.ci import flux1_comparison as comparison
from scripts.ci.model_proof_resources import CudaClaimRejected
from scripts.media_campaign_supervisor import RunResult


class FluxPairTests(unittest.TestCase):
    def test_private_checkout_does_not_clone_missing_promisor_history(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            original, runner = root / "original", root / "runner"
            original.mkdir()
            comparison.git(original, "init", "--quiet")
            comparison.git(original, "config", "user.name", "Proof Test")
            comparison.git(original, "config", "user.email", "proof@example.invalid")
            comparison.git(original, "config", "uploadpack.allowFilter", "true")
            path = original / "source"
            path.write_text("old source\n", encoding="utf-8")
            comparison.git(original, "add", "source")
            comparison.git(original, "commit", "--quiet", "-m", "old source")
            missing = comparison.git(original, "rev-parse", "HEAD:source")
            path.write_text("current source\n", encoding="utf-8")
            comparison.git(original, "commit", "--quiet", "-am", "current source")
            subprocess.run(["git", "clone", "--quiet", "--filter=blob:none",
                            original.as_uri(), str(runner)], check=True, timeout=60)
            self.assertIn("?" + missing,
                          comparison.git(runner, "rev-list", "--objects", "--missing=print", "HEAD"))
            original.rename(root / "offline")
            expected = {"revision": comparison.git(runner, "rev-parse", "HEAD"),
                        "tree": comparison.git(runner, "rev-parse", "HEAD^{tree}")}
            work = root / "proof"
            work.mkdir()
            candidate = work / "head"
            with self.assertRaisesRegex(ValueError, "exact source checkout"):
                comparison.add_source_worktree(runner, candidate, "0" * 40)
            self.assertEqual(comparison.add_source_worktree(runner, candidate, expected["revision"]), expected)
            self.assertEqual((candidate / "source").read_text(encoding="utf-8"), "current source\n")
            (candidate / "source").write_text("private probe\n", encoding="utf-8")
            comparison.remove_source_worktrees(work, {"head": runner})
            self.assertFalse(candidate.exists())
            self.assertNotIn(str(candidate), comparison.git(runner, "worktree", "list", "--porcelain"))
            self.assertEqual((runner / "source").read_text(encoding="utf-8"), "current source\n")

    def test_source_widths_and_variant_difference(self):
        dev, schnell = (fixture.geometry(v) for v in evidence.VARIANTS)
        self.assertEqual({k: len(v) for k, v in dev.items()},
                         {"transformer": 66, "text_encoder": 20, "text_encoder_2": 12, "vae": 244})
        extra = set(dev["transformer"]) - set(schnell["transformer"])
        self.assertEqual(len(extra), 4)
        self.assertTrue(all("guidance_embedder" in k for k in extra))
        self.assertEqual(dev["text_encoder_2"]["encoder.block.0.layer.0.SelfAttention.q.weight"], [4096, 4096])
        self.assertEqual(dev["transformer"]["x_embedder.weight"], [3072, 64])

    def test_weight_bytes_are_reproducible_and_key_specific(self):
        with TemporaryDirectory() as root:
            files = [Path(root) / f"{i}.safetensors" for i in range(2)]
            for file in files:
                fixture.write_component(file, {"a": [3, 2], "b": [3, 2]})
            one = files[0].read_bytes()
            self.assertEqual(one, files[1].read_bytes())
            header_size, = struct.unpack("<Q", one[:8])
            data = one[8 + header_size:]
            self.assertEqual(len(data), 24)
            self.assertNotEqual(data[:12], data[12:])

    def test_planner_rejects_wrong_runner_or_unbounded_child(self):
        policy = json.loads((comparison.ROOT / "docs/evidence/sga-eng-1a/mlx-route-plan.json")
                            .read_text(encoding="utf-8"))["request"]["policy"]
        request = {"runner": "nax", "selection": "flux1-base-pair", "policy": policy}
        head = "f" * 40
        with patch.object(comparison.proof, "repository_identity", return_value={"revision": head, "tree": "a" * 40}):
            self.assertEqual(comparison.plan("mlx", request, head, "tier1")["runner"][-1], "rw-llm")
            with self.assertRaises(ValueError):
                comparison.plan("mlx", dict(request, runner="hosted"), head, "tier1")
            with self.assertRaises(ValueError):
                comparison.plan("mlx", dict(request, policy=dict(policy, deadlineSeconds=1201)), head, "tier1")
            with self.assertRaises(ValueError):
                comparison.plan("mlx", request, comparison.BASE, "tier1")


    def test_eng4b_q4_fixture_packs_real_width_matrices_and_verifies_headers(self):
        real = fixture.geometry("dev")
        for component, key, code_shape, scale_shape in (
            ("transformer", "transformer_blocks.0.attn.to_q", [3072, 384], [3072, 48]),
            ("text_encoder", "text_model.embeddings.token_embedding", [49408, 96], [49408, 12]),
            ("text_encoder_2", "shared", [32128, 512], [32128, 64]),
        ):
            entries = {name: (dtype, shape) for name, dtype, shape, _ in
                       fixture.tensor_layout(real[component], q4=True)}
            self.assertEqual(entries[key + ".weight"], ("U32", code_shape))
            self.assertEqual(entries[key + ".scales"], ("F32", scale_shape))
            self.assertEqual(entries[key + ".biases"], ("F32", scale_shape))
        # Exercise storage and reader integrity on tiny byte fixtures only. The real-width
        # geometry above is inspected without allocating or executing any model tensor.
        tiny = {component: {"matrix.weight": [2, 64], "norm.weight": [64]}
                for component in fixture.COMPONENTS}
        with TemporaryDirectory() as tmp, patch.object(fixture, "geometry", return_value=tiny):
            snapshot = Path(tmp)
            fixture.make_snapshot(snapshot, "dev", q4=True)
            fixture.validate_q4_snapshot(snapshot, "dev")
            arrays = snapshot / "text_encoder/model.safetensors"
            with arrays.open("ab") as output:
                output.write(b"x")
            with self.assertRaisesRegex(ValueError, "wrong synthetic tensor layout: text_encoder"):
                fixture.validate_q4_snapshot(snapshot, "dev")

    def test_eng4b_plan_binds_reviewed_sources_and_two_preflights(self):
        policy = json.loads((comparison.ROOT / "docs/evidence/sga-eng-1a/cuda-route-plan.json")
                            .read_text(encoding="utf-8"))["request"]["policy"]
        request = {"selection": "flux1-eng4b-pair", "policy": policy}
        head = "f" * 40
        with patch.object(comparison.proof, "repository_identity",
                          return_value={"revision": head, "tree": "a" * 40}):
            plan = comparison.plan("candle-cuda", request, head, fixture.ENG4B)
            self.assertEqual(plan["base"], "e86b8d5fc493c16b2d9d852e0c9ecbdefee58294")
            self.assertEqual(plan["base_tree"], "88e1369f11fbc1be2d2fb9b0063db8725463db92")
            self.assertEqual(plan["asset_setup"], "flux1-eng4b-comparison")
            self.assertEqual(comparison.phase_names(fixture.ENG4B),
                             ("preflight-base-dev", "preflight-head-dev", "base-dev", "head-dev"))
            self.assertEqual(plan["settings"]["depth"], {"double": 1, "single": 1, "clip": 1, "t5": 1})
            self.assertGreater(plan["cuda_claim_budget_seconds"], 4 * policy["deadlineSeconds"])
            legacy = dict(request, selection="flux1-base-pair")
            self.assertEqual(comparison.plan("candle-cuda", legacy, head, "tier1")["base"], comparison.BASE)
            with self.assertRaisesRegex(ValueError, "exact flux1-eng4b-pair"):
                comparison.plan("candle-cuda", legacy, head, fixture.ENG4B)
            with self.assertRaisesRegex(ValueError, "distinct head"):
                comparison.plan("candle-cuda", request, overlay.ENG4B_BASE, fixture.ENG4B)
            with self.assertRaisesRegex(ValueError, "Candle CUDA lane"):
                comparison.plan("mlx", request, head, fixture.ENG4B)

    def test_eng4b_artifact_binds_each_planned_arm(self):
        mu = math.exp(0.5)
        plan = {"provider": "flux1_dev", "backend": "candle", "inputs": {"requested_seed": 104729},
                "pass": {"sigmas": [mu / (mu + 3 / n - 1) for n in (3, 2, 1)] + [0.],
                         "phase": {"first_step": 0, "end_step": 3, "sampler": "euler",
                                   "guidance": 3.5, "stream": dict(width=256, height=256, patch_stride=16,
                                                                   tokens=256, token_width=64)}}}
        contract = {"tier": fixture.ENG4B, "base": overlay.ENG4B_BASE, "base_tree": overlay.ENG4B_BASE_TREE,
                    "head": "d" * 40, "head_tree": "e" * 40}
        receipt = {"preflight": {arm + "-dev": copy.deepcopy(plan) for arm in evidence.ARMS},
                   "fixtures": {"dev": {"tier": fixture.ENG4B, "variant": "dev", "depths": [1, 1],
                                        "fixture_storage": dict(Q4=fixture.Q4, dense_vae="BF16")}},
                   "arms": {arm: {"build": {"revision": contract[arm], "tree": contract[arm + "_tree"],
                                           "overlay_sha256": "a" * 64, "binary_sha256": "b" * 64},
                                  "renders": {"dev": {"planned_route": copy.deepcopy(plan)}}}
                            for arm in evidence.ARMS}}
        comparison.verify_eng4b_identity(receipt, contract, "candle-cuda")
        receipt["arms"]["base"]["renders"]["dev"]["planned_route"] = None
        with self.assertRaisesRegex(ValueError, "render plan differs"):
            comparison.verify_eng4b_identity(receipt, contract, "candle-cuda")

    def test_eng4b_real_plan_uses_published_inputs_and_frozen_serial_budget(self):
        policy = json.loads((comparison.ROOT / "docs/evidence/sga-eng-1a/cuda-route-plan.json")
                            .read_text(encoding="utf-8"))["request"]["policy"]
        head = "f" * 40
        request = {"selection": "flux1-eng4b-pair", "policy": policy}
        with patch.object(comparison.proof, "repository_identity",
                          return_value={"revision": head, "tree": "a" * 40}):
            with self.assertRaisesRegex(ValueError, "360 second"):
                comparison.plan("candle-cuda", request, head, fixture.ENG4B_REAL)
            policy["deadlineSeconds"] = 360
            contract = comparison.plan("candle-cuda", request, head, fixture.ENG4B_REAL)
            download = dict(request, download_missing=True)
            self.assertTrue(comparison.plan("candle-cuda", download, head, fixture.ENG4B_REAL)
                            ["request"]["download_missing"])
            for bad, tier in ((dict(request, download_missing=1), fixture.ENG4B_REAL),
                              (download, fixture.ENG4B)):
                with self.assertRaisesRegex(ValueError, "exact flux1-eng4b-pair"):
                    comparison.plan("candle-cuda", bad, head, tier)
        self.assertEqual((contract["base"], contract["head"]),
                         ("be45bdf746c5d4a12d4db76f9653b8686e7f33f8", head))
        self.assertEqual(contract["base_tree"], "4d6eb59e08edcd257fd907247acd5e8f0cc5c545")
        self.assertEqual(contract["asset_setup"], "flux1-eng4b-terminal-comparison")
        refs = contract["pinned_fixtures"]
        self.assertEqual(refs["dev"]["source_snapshot"]["snapshot"], "323fd12d79f78ad444e882e8d8e871914584f2b9")
        self.assertEqual(refs["schnell"]["source_snapshot"]["snapshot"], "bba3ae01dfd94089f173c05edd4e1a4c551f2599")
        self.assertTrue(all(row["depths"] == [19, 38] and len(row["files"]) > 4 for row in refs.values()))
        phases = comparison.phase_names(fixture.ENG4B_REAL)
        self.assertEqual(len(phases), 32)
        self.assertEqual(phases[:4], ("preflight-base-dev", "preflight-base-schnell",
                                      "preflight-head-dev", "preflight-head-schnell"))
        self.assertEqual(phases[8:12], ("measure-dev-1-base", "measure-dev-1-head",
                                       "measure-dev-2-head", "measure-dev-2-base"))
        self.assertEqual(phases[-1], "measure-schnell-6-base")
        self.assertLess(contract["cuda_claim_budget_seconds"], 200 * 60)

    def test_terminal_staging_binds_only_exact_fixture_paths_and_transfer_seals(self):
        provenance = json.loads((fixture.HERE / "headers.json").read_text(encoding="utf-8"))
        fixtures = {(row["repo"], row["snapshot"]): row["q4_files"]
                    for row in provenance.values()}
        inventories = {
            source: {"q4/" + name: dict(size=identity["bytes"], **(
                {"lfs": {"size": identity["bytes"], "sha256": identity["sha256"]}}
                if name.endswith(".safetensors") else {}))
                     for name, identity in files.items()}
            for source, files in fixtures.items()
        }
        for inventory in inventories.values():
            inventory["q4/unused.safetensors"] = {"size": 1}
        def upstream(repository, revision):
            return "manifest", b"document", inventories[repository, revision]

        def stage(pins, root, *, manifest, fetch):
            self.assertEqual(set(pins), {"dev", "schnell"})
            for pin in pins.values():
                repo, revision = pin["repository"], pin["revision"]
                approved = fixtures[repo, revision]
                self.assertEqual(pin["subfolder"], "q4")
                _, _, selected = manifest(repo, revision)
                self.assertEqual(set(selected), {"q4/" + name for name in approved})
                base = comparison.hub_snapshots.snapshot_dir(root, repo, revision)
                for path in selected:
                    fetch("reviewed immutable URL", base / path)

        with (TemporaryDirectory() as directory,
              patch.object(comparison.hub_snapshots, "pinned_manifest", side_effect=upstream),
              patch.object(comparison.hub_snapshots, "stage", side_effect=stage),
              patch.object(comparison, "fetch_pinned") as transfer):
            root = Path(directory).resolve()
            comparison.stage_terminal_snapshots(root, provenance)
            self.assertEqual(len(transfer.call_args_list), 18)
            seals = {(item.args[1], item.kwargs["expected_sha256"], item.kwargs["expected_bytes"])
                     for item in transfer.call_args_list}
            self.assertEqual(seals, {
                (comparison.hub_snapshots.snapshot_dir(root, repo, revision) / "q4" / name,
                 identity["sha256"], identity["bytes"])
                for (repo, revision), files in fixtures.items() for name, identity in files.items()
            })
            first = provenance["dev"]
            inventories[first["repo"], first["snapshot"]]["q4/transformer/model.safetensors"]["lfs"]["sha256"] = "0" * 64
            transfer.reset_mock()
            with self.assertRaisesRegex(ValueError, "exact fixture"):
                comparison.stage_terminal_snapshots(root, provenance)
            transfer.assert_not_called()

    def test_terminal_staging_refuses_cache_and_lock_paths_outside_the_hub(self):
        provenance = json.loads((fixture.HERE / "headers.json").read_text(encoding="utf-8"))
        with TemporaryDirectory() as directory, patch.object(comparison.hub_snapshots, "stage") as stage:
            root = Path(directory).resolve()
            hub, outside = root / "hub", root / "elsewhere"
            hub.mkdir()
            outside.mkdir()
            dev = provenance["dev"]
            tier = comparison.hub_snapshots.snapshot_dir(hub, dev["repo"], dev["snapshot"]) / "q4"
            tier.parent.mkdir(parents=True)
            for path, message in ((tier, "escapes configured model hub"),
                                  (hub / ".locks", "linked evidence ancestor")):
                try:
                    path.symlink_to(outside, target_is_directory=True)
                except (NotImplementedError, PermissionError):
                    self.skipTest("this host cannot create directory symlinks for the fixture")
                with self.assertRaisesRegex(ValueError, message):
                    comparison.stage_terminal_snapshots(hub, provenance)
                path.unlink()
            stage.assert_not_called()

    def test_terminal_snapshot_refuses_unlisted_weights_before_opening_any_file(self):
        with TemporaryDirectory() as directory, patch.object(fixture, "sha") as digest:
            root = Path(directory)
            pinned = fixture.real_snapshot_identity("dev", fixture.ENG4B_REAL)["files"]
            for relative in (*pinned, "transformer/extra.safetensors"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with self.assertRaisesRegex(ValueError, "pinned file names"):
                fixture.snapshot_identity(root, "dev", fixture.ENG4B_REAL)
            digest.assert_not_called()

    def test_paired_timing_distinguishes_pass_fail_and_inconclusive(self):
        rows = {phase: dict.fromkeys((*performance.TIMING, *performance.MEMORY), 100)
                for phase, _, _ in performance.cases(("dev",))}
        for phase, _, arm in performance.cases(("dev",)):
            if arm == "head":
                rows[phase] = dict.fromkeys(rows[phase], 102)
        self.assertEqual(performance.evaluate(rows, ("dev",))["status"], "pass")
        rows["measure-dev-2-head"][performance.TIMING[0]] = 103
        inconclusive = performance.evaluate(rows, ("dev",))
        self.assertEqual(inconclusive["status"], "inconclusive")
        self.assertEqual(inconclusive["variants"]["dev"][performance.TIMING[0]]["interval"], [1.02, 1.03])
        for phase, _, arm in performance.cases(("dev",)):
            if arm == "head":
                rows[phase][performance.TIMING[0]] = 104
        self.assertEqual(performance.evaluate(rows, ("dev",))["status"], "fail")
        rows["measure-dev-1-head"][performance.MEMORY[0]] = 103
        self.assertEqual(performance.evaluate(rows, ("dev",))["variants"]["dev"][performance.MEMORY[0]]["status"],
                         "fail")

    def test_measured_output_requires_a_warmed_completed_matching_body(self):
        payload = b"finished PNG fixture"
        preflight = {"provider": "flux1_dev"}
        reference = {"states": [{"dtype": "BF16"}], "steps": [[1, 3], [2, 3], [3, 3]],
                     "png_sha256": hashlib.sha256(payload).hexdigest()}
        policy = {"deadlineSeconds": 360, "childFootprintCapBytes": 1000,
                  "childGpuCapBytes": 1000, "cudaDeviceUuid": "test"}
        pool = dict(pid=34, cudaDeviceUuid="test", poolUsedBytes=250,
                    poolUsedHighBytes=300, poolReservedBytes=400, poolReservedHighBytes=500)
        files = {"stdout.log": b"FLUX1_PAIR_DENOISE 800 BF16 candle-cuda\r\n"
                               b"FLUX1_PAIR_DENOISE 600 BF16 candle-cuda\r\n"
                               b"FLUX1_PAIR_PHASES 150 10 240\r\n"
                               b"FLUX1_PAIR_TIMING 1000 700\r\nFLUX1_PAIR_COMPLETE flux1_dev\r\n",
                 "plan.json": json.dumps(preflight).encode(),
                 "progress.json": json.dumps(reference["steps"]).encode(),
                 "output.png": payload,
                 "supervisor.json": b'{"returncode": 0, "pid": 34}',
                 "cuda-pool.json": json.dumps(pool).encode()}
        expected = dict(zip((*performance.TIMING, *performance.MEMORY), (2e-7, 1e-6, 700, 300, 500)))
        expected["phase_nanoseconds"] = {
            "before_denoise": 150, "after_denoise_before_decode": 10, "decode_through_host_rgb": 240,
        }
        self.assertEqual(performance.read_sample(files.__getitem__, "dev", preflight, reference, policy), expected)
        for name, value, reason in (
            ("stdout.log", files["stdout.log"].split(b"\n", 1)[1], "warm-up"),
            ("stdout.log", files["stdout.log"].replace(b"150 10 240", b"150 10 239"), "render phases"),
            ("stdout.log", files["stdout.log"].replace(b"150 10 240", b"150 -1 251"), "render phases"),
            ("plan.json", b'{"provider": "other"}', "route"),
            ("output.png", b"mismatch", "RGB8"),
            ("cuda-pool.json", json.dumps(dict(pool, pid=99)).encode(), "allocation high-water"),
        ):
            observed = dict(files, **{name: value})
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, reason):
                performance.read_sample(observed.__getitem__, "dev", preflight, reference, policy)

    def test_guard_failure_preserves_reason_and_prints_only_available_stderr_tail(self):
        for reason, log in [("child-exit", b"not-in-tail" + b"x" * 4000 + b"failure-\xff"),
                            ("spawn-failure", None)]:
            with self.subTest(reason=reason), TemporaryDirectory() as tmp:
                failure = comparison.SupervisionError(reason, "original reason")

                def fail_guard(*args, **kwargs):
                    if log is not None:
                        kwargs["stderr_path"].write_bytes(log)
                    raise failure

                err = StringIO()
                with patch.object(comparison, "run_guarded", side_effect=fail_guard), redirect_stderr(err):
                    with self.assertRaises(comparison.SupervisionError) as raised:
                        comparison.invoke(Path("unused"), Path(tmp), Path(tmp) / "out", {}, None, {})
                self.assertIs(raised.exception, failure)
                if log is None:
                    self.assertEqual(err.getvalue(), "")
                else:
                    self.assertEqual(err.getvalue(), log[-4000:].decode("utf-8", errors="replace") + "\n")

    def test_partial_and_complete_comparisons_retain_logs_under_one_aggregate_claim(self):
        scenarios = [(tier, failure) for tier in ("tier1", *fixture.ENG4B_TIERS)
                     for failure in (None, "child", "cleanup")]
        scenarios.append((fixture.ENG4B_REAL, "noise"))
        for tier, failure in scenarios:
            fail = failure == "child"
            with self.subTest(tier=tier, failure=failure), TemporaryDirectory() as tmp, ExitStack() as mocks:
                root = Path(tmp).resolve()
                policy = json.loads((comparison.ROOT / "docs/evidence/sga-eng-1a/cuda-route-plan.json")
                                    .read_text(encoding="utf-8"))["request"]["policy"]
                policy["deadlineSeconds"] = 120
                contract = {"tier": tier, "request": {"policy": policy}}
                eng4b = tier in fixture.ENG4B_TIERS
                measured = tier == fixture.ENG4B_REAL
                if eng4b:
                    baseline, baseline_tree = overlay.eng4b_baseline(tier)
                    contract.update(base=baseline, base_tree=baseline_tree,
                                    head="a" * 40, head_tree="b" * 40)
                profiles = comparison.variants(tier)
                shapes = {v: {} for v in profiles}
                if tier == fixture.ENG4B:
                    shapes["dev"] = {"tier": tier, "variant": "dev", "depths": [1, 1],
                                     "fixture_storage": dict(Q4=fixture.Q4, dense_vae="BF16")}
                if measured:
                    shapes = {v: fixture.real_snapshot_identity(v, tier) for v in profiles}
                    contract["pinned_fixtures"] = shapes
                args = SimpleNamespace(backend="candle-cuda", output=root / "artifact")
                clock = [1000.]
                phase_order = []

                @contextmanager
                def claim(backend, claim_policy, claim_dir, env, *, original_deadline_monotonic=None):
                    self.assertEqual(backend, "candle-cuda")
                    if not fail:
                        self.assertGreater(claim_policy.deadline_seconds,
                                           len(comparison.phase_names(tier)) * policy["deadlineSeconds"])
                        self.assertEqual(original_deadline_monotonic, clock[0] + claim_policy.deadline_seconds)
                    self.assertEqual(claim_policy.cuda_device_uuid, policy["cudaDeviceUuid"])
                    yield {"post_execution_verified": True}
                    self.assertLess(clock[0] + 30, original_deadline_monotonic)

                def child(argv, *, env, policy, stdout_path, stderr_path, event_path, **kwargs):
                    # Emulate libtest/probe/supervisor artifacts without a subprocess.
                    self.assertEqual(policy.deadline_seconds, 120)
                    clock[0] += 40
                    out = Path(env["PAIR_OUT"])
                    phase_order.append(out.name)
                    model = env["PAIR_MODEL"]
                    dev = model == "flux1_dev"
                    mu = math.exp(0.5) if dev else 1.0
                    sigmas = [mu / (mu + 3 / n - 1) for n in (3, 2, 1)] + [0.]
                    plan = {"provider": model, "backend": "candle", "inputs": {"requested_seed": 104729},
                            "pass": {"sigmas": sigmas, "phase": {"first_step": 0, "end_step": 3,
                                "sampler": "euler", "guidance": 3.5 if dev else 0.,
                                "stream": dict(width=256, height=256, patch_stride=16, tokens=256, token_width=64)}}}
                    comparison.write(out / "plan.json", plan)
                    stderr_path.write_text("diagnostic " + env["HF_TOKEN"], encoding="utf-8")
                    event_path.write_text('{"sample":1}\n', encoding="utf-8")
                    if "--preflight" in argv:
                        stdout_path.write_text("FLUX1_PAIR_PREFLIGHT " + model + "\n", encoding="utf-8")
                    else:
                        lines = []
                        if "--measure" in argv:
                            duration = 1100 if failure == "noise" and out.name == "measure-dev-2-head" else 1000
                            lines.extend(("FLUX1_PAIR_DENOISE 900 BF16 candle-cuda",
                                          "FLUX1_PAIR_DENOISE 600 BF16 candle-cuda",
                                          f"FLUX1_PAIR_PHASES {duration - 600} 0 0",
                                          f"FLUX1_PAIR_TIMING {duration} 4096"))
                        else:
                            for i, sigma in enumerate(sigmas):
                                bits = struct.pack(">f", sigma).hex()
                                (out / f"state-{bits}.bin").write_bytes(struct.pack("<f", float(i)) * 16384)
                                lines.append(f"FLUX1_PAIR_STATE {bits} BF16 candle-cuda")
                        lines.append("FLUX1_PAIR_COMPLETE " + model)
                        stdout_path.write_text("\n".join(lines), encoding="utf-8")
                        comparison.write(out / "cuda-pool.json", {"pid": 1, "cudaDeviceUuid": policy.cuda_device_uuid,
                                                                  "poolUsedBytes": 1, "poolUsedHighBytes": 1,
                                                                  "poolReservedBytes": 1,
                                                                  "poolReservedHighBytes": 1})
                        if fail:
                            raise comparison.SupervisionError("child-exit", "probe failed")
                        comparison.write(out / "progress.json", [[1, 3], [2, 3], [3, 3]])
                        (out / "output.rgb").write_bytes(bytes(range(256)) * 768)
                    return RunResult(1, 0, 4096, 2048, 8192, 8192, 40, (), {})

                mocks.enter_context(patch.dict(os.environ, RUNNER_TEMP=str(root), GITHUB_RUN_ID="7",
                                               GITHUB_RUN_ATTEMPT="1", RUNNER_NAME="mock", HF_TOKEN="mock-secret"))
                def git_identity(source, operation, target):
                    self.assertEqual(operation, "rev-parse")
                    if target == "HEAD^{tree}":
                        return contract["base_tree"]
                    return contract.get("base", comparison.BASE)

                def build(work, sources, arm, backend, selected_tier, env):
                    self.assertEqual(selected_tier, tier)
                    identity = ({"revision": contract[arm], "tree": contract[arm + "_tree"],
                                 "overlay_sha256": "a" * 64, "binary_sha256": "b" * 64} if eng4b else {})
                    if measured:
                        identity.update(build_command=list(performance.BUILD), toolchain={"channel": "1.96.0"})
                    return root / "probe", identity

                mocks.enter_context(patch.object(comparison, "git", side_effect=git_identity))
                mocks.enter_context(patch.object(comparison, "build", side_effect=build))
                mocks.enter_context(patch.object(comparison, "hardware", return_value={}))
                mocks.enter_context(patch.object(comparison, "snapshots", return_value=(
                    {v: root for v in profiles}, shapes)))
                mocks.enter_context(patch.object(comparison.fixture, "snapshot_identity",
                                                 side_effect=lambda path, variant, tier: shapes[variant]))
                mocks.enter_context(patch.object(comparison.proof, "require_host"))
                mocks.enter_context(patch.object(comparison, "execution_claim", claim))
                mocks.enter_context(patch.object(comparison, "time", SimpleNamespace(monotonic=lambda: clock[0]), create=True))
                mocks.enter_context(patch.object(comparison, "run_guarded", side_effect=child))
                if fail:
                    with self.assertRaisesRegex(comparison.SupervisionError, "probe failed"), redirect_stderr(StringIO()):
                        comparison.execute(args, contract)
                elif failure == "cleanup":
                    with (patch.object(comparison, "remove_work", side_effect=PermissionError("pack in use")),
                          self.assertRaisesRegex(PermissionError, "pack in use")):
                        comparison.execute(args, contract)
                elif failure == "noise":
                    with self.assertRaisesRegex(ValueError, "performance is inconclusive"):
                        comparison.execute(args, contract)
                else:
                    comparison.execute(args, contract)
                    self.assertEqual(len(phase_order), len(comparison.phase_names(tier)))
                    if eng4b:
                        self.assertEqual(tuple(phase_order), comparison.phase_names(tier))
                    self.assertGreater(clock[0] - 1000., policy["deadlineSeconds"])
                comparison.verify(args, contract)
                receipt = json.loads((args.output / "receipt.json").read_text(encoding="utf-8"))
                self.assertEqual(receipt["accepted"], failure is None)
                self.assertEqual((root / "f1p-7-1").exists(), failure == "cleanup")
                if failure == "cleanup":
                    self.assertEqual(receipt["cleanup_failure"], "pack in use")
                if measured and not fail:
                    self.assertEqual(receipt["performance"]["status"],
                                     "inconclusive" if failure == "noise" else "pass")
                    self.assertEqual(len(receipt["performance_samples"]), 24)
                    self.assertEqual(tuple(phase_order), comparison.phase_names(tier))
                self.assertIn("logs", receipt)
                for phase in phase_order:
                    for suffix in ("stdout.log", "stderr.log", "guard.jsonl", "supervisor.json"):
                        self.assertIn(phase + "." + suffix, receipt["logs"])
                    if not phase.startswith("preflight-"):
                        self.assertIn(phase + ".cuda-pool.json", receipt["logs"])
                    self.assertEqual((args.output / (phase + ".stderr.log")).read_text(encoding="utf-8"),
                                     "diagnostic [REDACTED]")
                if fail:
                    result = json.loads((args.output / "base-dev.supervisor.json").read_text(encoding="utf-8"))
                    self.assertEqual(result["reason"], "child-exit")
                    self.assertIn("probe failed", receipt["failure"])
                (args.output / "base-dev.stdout.log").write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "invalid comparison log"):
                    comparison.verify(args, contract)

    def test_windows_cleanup_recovers_transient_lock_and_readonly_pack(self):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            work = root / "f1p-7-1"
            pack = work / "base/.git/objects/pack/pack-example.idx"
            pack.parent.mkdir(parents=True)
            pack.touch(mode=0o444)
            readonly = PermissionError("access denied")
            readonly.winerror = 5
            sharing = PermissionError("sharing violation")
            sharing.winerror = 32
            rmtree = comparison.shutil.rmtree

            def delete(path, *, onexc):
                if remove.call_count == 1:
                    raise sharing
                onexc(os.unlink, str(pack), readonly)
                self.assertFalse(pack.exists())
                rmtree(path)

            with (patch.object(comparison.sys, "platform", "win32"),
                  patch.object(comparison.shutil, "rmtree", side_effect=delete) as remove,
                  patch.object(comparison.time, "sleep") as sleep,
                  patch.object(Path, "chmod", autospec=True, side_effect=Path.chmod) as chmod):
                comparison.remove_work(work, root)
            self.assertEqual(remove.call_count, 2)
            sleep.assert_called_once_with(0.25)
            self.assertEqual(chmod.call_args.args[0], pack)
            self.assertTrue(chmod.call_args.args[1] & stat.S_IWRITE)
            self.assertFalse(work.exists())

    def test_windows_cleanup_never_changes_permissions_through_hardlinks(self):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            work = root / "f1p-7-1"
            work.mkdir()
            outside = root / "retained.idx"
            outside.write_bytes(b"outside")
            pack = work / "pack.idx"
            os.link(outside, pack)
            outside.chmod(stat.S_IREAD)
            error = PermissionError("shared pack is read-only")
            error.winerror = 5
            try:
                with (patch.object(comparison.sys, "platform", "win32"), patch.object(comparison.time, "sleep"),
                      patch.object(Path, "chmod") as chmod,
                      patch.object(comparison.shutil, "rmtree",
                                   side_effect=lambda path, onexc: onexc(os.unlink, str(pack), error))):
                    with self.assertRaises(PermissionError) as raised:
                        comparison.remove_work(work, root)
                self.assertIs(raised.exception, error)
                chmod.assert_not_called()
                self.assertEqual(outside.read_bytes(), b"outside")
            finally:
                outside.chmod(stat.S_IREAD | stat.S_IWRITE)

    def test_persistently_locked_cleanup_preserves_primary_cuda_refusal_and_receipt(self):
        with TemporaryDirectory() as temp, ExitStack() as mocks:
            root = Path(temp).resolve()
            args = SimpleNamespace(backend="candle-cuda", output=root / "artifact")
            policy = json.loads((comparison.ROOT / "docs/evidence/sga-eng-1a/cuda-route-plan.json")
                                .read_text(encoding="utf-8"))["request"]["policy"]
            contract = {"tier": "tier1", "request": {"policy": policy}}
            rejection = CudaClaimRejected({"matched_uuid_count": 1, "occupied": True})
            locked = PermissionError("locked clone pack idx")
            locked.winerror = 5
            mocks.enter_context(patch.dict(os.environ, RUNNER_TEMP=str(root), GITHUB_RUN_ID="7",
                                           GITHUB_RUN_ATTEMPT="1", RUNNER_NAME="mock"))
            mocks.enter_context(patch.object(comparison.proof, "require_host"))
            for name, value in (("git", comparison.BASE), ("hardware", {}), ("build", (root / "probe", {})),
                                ("snapshots", ({v: root for v in evidence.VARIANTS}, {}))):
                mocks.enter_context(patch.object(comparison, name, return_value=value))
            mocks.enter_context(patch.object(comparison, "execution_claim", side_effect=rejection))
            mocks.enter_context(patch.object(comparison.sys, "platform", "win32"))
            remove = mocks.enter_context(patch.object(comparison.shutil, "rmtree", side_effect=locked))
            sleep = mocks.enter_context(patch.object(comparison.time, "sleep"))
            guard = mocks.enter_context(patch.object(comparison, "run_guarded"))
            with self.assertRaises(CudaClaimRejected) as raised:
                comparison.execute(args, contract)
            self.assertIs(raised.exception, rejection)
            self.assertIn("locked clone pack idx", rejection.__notes__[0])
            guard.assert_not_called()
            self.assertEqual([c.args[0] for c in sleep.call_args_list], [0.25, 1.0])
            self.assertEqual([c.args[0] for c in remove.call_args_list], [root / "f1p-7-1"] * 3)
            receipt = json.loads((args.output / "receipt.json").read_text(encoding="utf-8"))
            self.assertFalse(receipt["accepted"])
            self.assertEqual(receipt["cuda_diagnostic"], rejection.diagnostic)
            self.assertEqual(receipt["failure"], str(rejection))
            self.assertEqual(receipt["cleanup_failure"], str(locked))
            comparison.verify(args, contract)

    def test_no_card_fitting_before_the_deadline_publishes_a_verifiable_refusal(self):
        from scripts.ci import model_proof_resources as resources
        with TemporaryDirectory() as temp, ExitStack() as mocks:
            root = Path(temp).resolve()
            args = SimpleNamespace(backend="candle-cuda", output=root / "artifact")
            policy = json.loads((comparison.ROOT / "docs/evidence/sga-eng-1a/cuda-route-plan.json")
                                .read_text(encoding="utf-8"))["request"]["policy"]
            policy = {key: value for key, value in policy.items() if key != "cudaDeviceUuid"}
            contract = {"tier": "tier1", "request": {"policy": policy}}
            refusal = {"rule": resources.CARD_SELECTION_RULE, "polls": 4, "admission_error": "none fits",
                       "cards": [{"uuid": uuid, "index": index, "refusal": "1 MiB free < 2 MiB required"}
                                 for uuid, index in resources.REGISTERED_CUDA_CARDS.items()]}
            rejection = CudaClaimRejected(refusal, "no registered CUDA card fits this proof; no model execution")
            mocks.enter_context(patch.dict(os.environ, RUNNER_TEMP=str(root), GITHUB_RUN_ID="7",
                                           GITHUB_RUN_ATTEMPT="1", RUNNER_NAME="mock"))
            mocks.enter_context(patch.object(comparison.proof, "require_host"))
            for name, value in (("git", comparison.BASE), ("hardware", {}), ("build", (root / "probe", {})),
                                ("snapshots", ({v: root for v in evidence.VARIANTS}, {}))):
                mocks.enter_context(patch.object(comparison, name, return_value=value))
            mocks.enter_context(patch.object(comparison, "choose_card", side_effect=rejection))
            claim = mocks.enter_context(patch.object(comparison, "execution_claim"))
            guard = mocks.enter_context(patch.object(comparison, "run_guarded"))
            with self.assertRaises(CudaClaimRejected):
                comparison.execute(args, contract)
            guard.assert_not_called()
            claim.assert_not_called()
            receipt = json.loads((args.output / "receipt.json").read_text(encoding="utf-8"))
            self.assertFalse(receipt["accepted"])
            self.assertEqual(receipt[resources.SELECTION_REFUSAL], refusal)
            comparison.verify(args, contract)

    def test_late_cuda_phase_refuses_before_spawn(self):
        with TemporaryDirectory() as tmp, patch.object(comparison, "run_guarded") as guard:
            with patch.object(comparison.time, "monotonic", return_value=200.0):
                with self.assertRaisesRegex(ValueError, "insufficient CUDA claim time"):
                    comparison.invoke(Path("unused"), Path(tmp), Path(tmp) / "out", {}, None, {}, last_child_start=200.)
            guard.assert_not_called()

    def test_compiled_plan_and_observed_trajectory_must_match(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "output.rgb").write_bytes(bytes(range(256)) * 768)
            (root / "progress.json").write_text("[[1,3],[2,3],[3,3]]", encoding="utf-8")
            sigmas = [1, 2 / 3, 1 / 3, 0]
            plan = {"provider": "flux1_schnell", "backend": "mlx", "inputs": {"requested_seed": 104729},
                    "pass": {"sigmas": sigmas, "phase": {
                        "first_step": 0, "end_step": 3, "sampler": "euler", "guidance": 0,
                        "stream": {"width": 256, "height": 256, "patch_stride": 16, "tokens": 256, "token_width": 64}}}}
            (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            stdout = "FLUX1_PAIR_COMPLETE flux1_schnell\n"
            for index, sigma in enumerate(sigmas):
                bits = struct.pack(">f", sigma).hex()
                stdout += f"FLUX1_PAIR_STATE {bits} Float32 mlx-metal\n"
                (root / f"state-{bits}.bin").write_bytes(struct.pack("<f", float(index)) * (256 * 64))
            head = evidence.read_arm(root, stdout, "head", "schnell", "mlx", root)
            base = evidence.read_arm(root, stdout, "base", "schnell", "mlx", root)
            evidence.compare(base, head)
            changed = copy.deepcopy(head)
            changed["states"][1]["sha256"] = hashlib.sha256(b"different").hexdigest()
            with self.assertRaisesRegex(ValueError, "states"):
                evidence.compare(base, changed)
            with self.assertRaisesRegex(ValueError, "device"):
                evidence.read_arm(root, stdout.replace("mlx-metal", "mlx-cpu"), "head", "schnell", "mlx", root)
            plan["pass"]["sigmas"][1] = 0.8
            with self.assertRaisesRegex(ValueError, "schedule"):
                evidence.validate_plan(plan, "schnell", "mlx")


if __name__ == "__main__":
    unittest.main()
