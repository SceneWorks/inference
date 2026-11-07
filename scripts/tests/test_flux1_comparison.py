"""Portable comparison-harness checks. No model, backend or Cargo executable is run."""
from contextlib import ExitStack, contextmanager, redirect_stderr
import copy
import hashlib
from io import StringIO
import json
import math
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.ci.flux1_pair import evidence, fixture
from scripts.ci import flux1_comparison as comparison
from scripts.media_campaign_supervisor import RunResult


class FluxPairTests(unittest.TestCase):
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
        for fail in (False, True):
            with self.subTest(fail=fail), TemporaryDirectory() as tmp, ExitStack() as mocks:
                root = Path(tmp).resolve()
                policy = json.loads((comparison.ROOT / "docs/evidence/sga-eng-1a/cuda-route-plan.json")
                                    .read_text(encoding="utf-8"))["request"]["policy"]
                policy["deadlineSeconds"] = 120
                contract = {"tier": "tier1", "request": {"policy": policy}}
                args = SimpleNamespace(backend="candle-cuda", output=root / "artifact")
                clock = [1000.]
                phase_order = []

                @contextmanager
                def claim(backend, claim_policy, claim_dir, env, *, original_deadline_monotonic=None):
                    self.assertEqual(backend, "candle-cuda")
                    if not fail:
                        self.assertGreater(claim_policy.deadline_seconds, 6 * policy["deadlineSeconds"])
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
                        for i, sigma in enumerate(sigmas):
                            bits = struct.pack(">f", sigma).hex()
                            (out / f"state-{bits}.bin").write_bytes(struct.pack("<f", float(i)) * 16384)
                            lines.append(f"FLUX1_PAIR_STATE {bits} BF16 candle-cuda")
                        lines.append("FLUX1_PAIR_COMPLETE " + model)
                        stdout_path.write_text("\n".join(lines), encoding="utf-8")
                        if fail:
                            raise comparison.SupervisionError("child-exit", "probe failed")
                        comparison.write(out / "progress.json", [[1, 3], [2, 3], [3, 3]])
                        (out / "output.rgb").write_bytes(bytes(range(256)) * 768)
                    return RunResult(1, 0, 4096, 2048, 8192, 8192, 40, (), {})

                mocks.enter_context(patch.dict(os.environ, RUNNER_TEMP=str(root), GITHUB_RUN_ID="7",
                                               GITHUB_RUN_ATTEMPT="1", RUNNER_NAME="mock", HF_TOKEN="mock-secret"))
                for name, value in (("git", comparison.BASE), ("hardware", {}), ("build", (root / "probe", {})),
                                    ("snapshots", ({v: root for v in evidence.VARIANTS},
                                                   {v: {} for v in evidence.VARIANTS}))):
                    mocks.enter_context(patch.object(comparison, name, return_value=value))
                mocks.enter_context(patch.object(comparison.fixture, "snapshot_identity", return_value={}))
                mocks.enter_context(patch.object(comparison.proof, "require_host"))
                mocks.enter_context(patch.object(comparison, "execution_claim", claim))
                mocks.enter_context(patch.object(comparison, "time", SimpleNamespace(monotonic=lambda: clock[0]), create=True))
                mocks.enter_context(patch.object(comparison, "run_guarded", side_effect=child))
                if fail:
                    with self.assertRaisesRegex(comparison.SupervisionError, "probe failed"), redirect_stderr(StringIO()):
                        comparison.execute(args, contract)
                else:
                    comparison.execute(args, contract)
                    self.assertEqual(len(phase_order), 6)
                    self.assertGreater(clock[0] - 1000., policy["deadlineSeconds"])
                comparison.verify(args, contract)
                receipt = json.loads((args.output / "receipt.json").read_text(encoding="utf-8"))
                self.assertEqual(receipt["accepted"], not fail)
                self.assertFalse((root / "f1p-7-1").exists())
                self.assertIn("logs", receipt)
                for phase in phase_order:
                    for suffix in ("stdout.log", "stderr.log", "guard.jsonl", "supervisor.json"):
                        self.assertIn(phase + "." + suffix, receipt["logs"])
                    self.assertEqual((args.output / (phase + ".stderr.log")).read_text(encoding="utf-8"),
                                     "diagnostic [REDACTED]")
                if fail:
                    result = json.loads((args.output / "base-dev.supervisor.json").read_text(encoding="utf-8"))
                    self.assertEqual(result["reason"], "child-exit")
                    self.assertIn("probe failed", receipt["failure"])
                (args.output / "base-dev.stdout.log").write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "invalid comparison log"):
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
