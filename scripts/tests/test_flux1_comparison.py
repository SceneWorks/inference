"""Portable comparison-harness checks. No model, backend or Cargo executable is run."""
from contextlib import redirect_stderr
import copy
import hashlib
from io import StringIO
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.ci.flux1_pair import evidence, fixture
from scripts.ci import flux1_comparison as comparison


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
