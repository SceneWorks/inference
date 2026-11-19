"""Metadata-only regressions for the dedicated SenseNova evidence protocol."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.ci.model_inference_proofs import NAX_ROUTES, SENSENOVA_PROOFS, request, require_host
from scripts.ci.sensenova_loop_assets import check_source_disk, source_snapshot

from scripts.ci.sensenova_loop_seal import (
    TIERS, verify_memory_endpoints, verify_oracle_marker, verify_semantic_margins,
)


class SenseNovaProofProtocolTests(unittest.TestCase):
    def test_oracle_hash_must_start_its_own_line_after_libtest_prefix(self):
        sha = "a" * 64
        marker = f"SENSENOVA_LOOP_ORACLE_SHA256 {sha}"
        prefix = "test sensenova_loop_real_weights::sensenova_loop_release_proof ... "
        verify_oracle_marker([prefix + "\n" + marker + "\nok"], sha)
        for invalid in (
            prefix + marker + "\nok",
            f"model said: {marker}\n",
            f"{marker} extra-text",
            f"{marker}\n{marker}",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                verify_oracle_marker([invalid], sha)

    def test_three_tiers_require_every_positive_margin(self):
        decisions = {"text": 2, "red": 1, "blue": 1, "quadrants": 1}
        step = {
            "token_ids": [1, 2], "logits": [5.0, 2.0],
            "semantic_score": {"positive_token_ids": [1], "negative_token_ids": [2], "source_positive_margin": 3.0},
        }
        oracle = {"cases": [{"id": name, "steps": [step] * count} for name, count in decisions.items()]}
        receipts = [
            "SENSENOVA_LOOP_SEMANTIC " + json.dumps({"tier": tier, "case_id": name, "step": index, "margin": 0.125})
            for tier in TIERS for name, count in decisions.items() for index in range(count)
        ]
        self.assertEqual(len(verify_semantic_margins(oracle, ["\n".join(receipts)])), 15)
        wrong_sign = receipts[-1].replace("0.125", "-0.125")
        prefixed = "model said: " + receipts[-1]
        for alternative in ([], [wrong_sign], [prefixed], [receipts[-1], receipts[-1]]):
            with self.subTest(alternative=alternative), self.assertRaises(ValueError):
                verify_semantic_margins(oracle, ["\n".join(receipts[:-1] + alternative)])
        step["semantic_score"]["source_positive_margin"] = 2.0
        with self.assertRaises(ValueError):
            verify_semantic_margins(oracle, ["\n".join(receipts)])


    def test_measured_endpoints_are_required_per_backend_and_anchor(self):
        mlx = {
            "product": {
                "tier": "BF16", "image": [1536, 1024], "prefill_tokens": 1560, "generated_tokens": 1,
                "load_metal_peak_bytes": 20, "request_metal_peak_bytes": 30, "loaded_process_bytes": 15,
            },
            "sampled_process_peak_bytes": 40,
        }
        cuda = {
            "tier": "BF16", "width": 1536, "height": 1024, "prompt_tokens": 1560, "generated_tokens": 1,
            "after_request": {"pool_reserved_peak_bytes": 30, "pool_used_peak_bytes": 20},
            "device": "candle-cuda", "memory_scope": "per-tier",
        }
        for backend, marker, bf16, q4 in (
            ("mlx", "SENSENOVA_LOOP_MEMORY", mlx, {**mlx, "product": {**mlx["product"], "tier": "Q4"}}),
            ("candle-cuda", "SENSENOVA_LOOP_MEMORY_ENDPOINT", cuda, {**cuda, "tier": "Q4"}),
        ):
            def log(*records):
                return ["\n".join(marker + " " + json.dumps(record) for record in records)]

            self.assertEqual(set(verify_memory_endpoints(backend, log(bf16, q4))), {"BF16", "Q4"})
            zero_peak = [log(bf16, q4)[0].replace(": 30", ": 0")]
            for invalid in (log(bf16), log(bf16, q4, q4), zero_peak):
                with self.subTest(backend=backend), self.assertRaises(ValueError):
                    verify_memory_endpoints(backend, invalid)


class SenseNovaProvisioningTests(unittest.TestCase):
    def test_mlx_requires_matching_explicit_provisioned_runner(self):
        value = {
            "selection": SENSENOVA_PROOFS["mlx"], "sensenova_source_sha": "a" * 40,
            "policy": {"backend": "darwin-mlx"},
            "sensenova_reference_policy": {"backend": "windows-cuda"},
        }
        for route, label in (("nax", "rw-llm"), ("nax-primary", "rw-starvector")):
            with self.subTest(route=route):
                parsed = request(json.dumps({**value, "runner": route}), "mlx")
                self.assertIn(label, NAX_ROUTES[parsed["runner"]]["labels"])
                env = {"GITHUB_EVENT_NAME": "workflow_dispatch", "RUNNER_OS": "macOS"}
                # The labels choose the Mac; either interchangeable Mac satisfies either route.
                for name in ("nax-macos", "nax-macos-2"):
                    require_host("mlx", {**env, "RUNNER_NAME": name}, route)
                with self.assertRaisesRegex(ValueError, "wrong proof host platform"):
                    require_host("mlx", {**env, "RUNNER_OS": "Linux"}, route)
        with self.assertRaisesRegex(ValueError, "explicit provisioned Mac route"):
            request(json.dumps(value), "mlx")
        with self.assertRaisesRegex(ValueError, "explicit provisioned Mac route"):
            request(json.dumps({**value, "runner": "hosted"}), "mlx")

    def test_missing_payload_space_counts_finalized_cache_entries_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            hub = Path(temporary)
            repo = hub / "models--sample--model"
            snapshot = repo / "snapshots" / "pinned"
            blobs = repo / "blobs"
            blobs.mkdir(parents=True)
            snapshot.mkdir(parents=True)
            (blobs / "existing").write_bytes(b"ready")
            (snapshot / "cached").symlink_to(blobs / "existing")
            (blobs / "finalized").write_bytes(b"ready")
            (blobs / "incomplete.incomplete").write_bytes(b"half")
            (snapshot / "short").write_bytes(b"too short")
            source = {
                "repository": "sample/model", "revision": "pinned", "files": [
                    {"path": "cached", "size": 5, "sha256": "existing"},
                    {"path": "pending-link", "size": 5, "sha256": "finalized"},
                    {"path": "pending-download", "size": 12, "sha256": "incomplete"},
                    {"path": "short", "size": 10, "sha256": "short"},
                ],
            }
            receipt = hub / "refused.json"
            with patch("scripts.ci.sensenova_loop_assets.shutil.disk_usage", return_value=SimpleNamespace(free=21)) as disk:
                with self.assertRaisesRegex(RuntimeError, "21 bytes free for 22 bytes"):
                    check_source_disk(snapshot, source, receipt)
                disk.assert_called_once_with(blobs)
            result = json.loads(receipt.read_bytes())
            self.assertEqual(result["missing_files"], [{"path": "pending-download", "size": 12}, {"path": "short", "size": 10}])
            self.assertFalse(result["payload_fits_sampled_free_space"])
            receipt = hub / "admitted.json"
            with patch("scripts.ci.sensenova_loop_assets.shutil.disk_usage", return_value=SimpleNamespace(free=22)):
                check_source_disk(snapshot, source, receipt)
            self.assertTrue(json.loads(receipt.read_bytes())["payload_fits_sampled_free_space"])

    def test_missing_source_is_fetched_through_the_runner_hub_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve() / "hf-home"
            calls = []

            def refuse_after_recording(**kwargs):
                calls.append(kwargs)
                raise RuntimeError("no network in this test")

            environment = {"HF_HOME": str(home), "GITHUB_ACTIONS": "true"}
            with patch.dict("os.environ", environment, clear=True):
                with self.assertRaisesRegex(RuntimeError, "cache-correct fetch"):
                    source_snapshot(download=refuse_after_recording)
        # Through $HF_HOME/hub, the runner's cache, as a cache fetch (no local_dir); never the
        # library's ~/.cache/huggingface/hub fallback.
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["cache_dir"], str(home / "hub"))
        self.assertEqual(calls[0]["repo_id"], "sensenova/SenseNova-Loop-9B")
        self.assertNotIn("local_dir", calls[0])


if __name__ == "__main__":
    unittest.main()
