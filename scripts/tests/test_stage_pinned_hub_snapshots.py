"""Registry-declared pinned Hub snapshots: slot contract, request binding, staging and the receipt
the proof re-checks. Mock Hub and fetch only; no network, model payload or GPU work."""

import errno
import hashlib
import io
import os
import json
from pathlib import Path
import tempfile
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from scripts.ci import model_inference_proofs as proofs
from scripts.ci import stage_pinned_hub_snapshots as staging
from scripts.ci.pinned_hub_source import verify_pinned_hub_files

ROOT = Path(__file__).resolve().parents[2]
REV = "76f3c445cd5347fdb85fc24066350724fe272017"
MLX = f"SceneWorks/qwen3.8-27b-abliterated-mlx@{REV}"
FULL = "SceneWorks/qwen3.8-27b-full-mlx@" + "a" * 40 + "/q8-full"


def row(*slots, required=("FULL", "PLAIN")):
    return {"package": "mlx-llm", "required_environment": list(required), "hub_snapshots": list(slots)}


SLOTS = ({"environment": "FULL", "request_bound": True},
         {"environment": "PLAIN", "choices": [MLX + "/q8", MLX + "/q4"]})


class SlotContract(unittest.TestCase):
    def test_each_slot_has_exactly_one_source_and_a_required_environment_name(self):
        proofs.validate_hub_snapshot_slots(row(*SLOTS))
        proofs.validate_hub_snapshot_slots(row({"environment": "FULL", "pin": MLX}, required=("FULL",)))
        for bad in ({"environment": "FULL"},
                    {"environment": "FULL", "pin": MLX, "request_bound": True},
                    {"environment": "OTHER", "pin": MLX},
                    {"environment": "FULL", "pin": "SceneWorks/x@main/q8"},
                    {"environment": "FULL", "pin": MLX + "/../q8"},
                    {"environment": "FULL", "choices": []},
                    {"environment": "FULL", "request_bound": False},
                    {"environment": "FULL", "pin": MLX, "files": ["config.json"]}):
            with self.subTest(slot=bad), self.assertRaises(ValueError):
                proofs.validate_hub_snapshot_slots(row(bad))
        with self.assertRaises(ValueError):
            proofs.validate_hub_snapshot_slots(row(SLOTS[0], SLOTS[0]))

    def test_pin_grammar_parses_repository_revision_and_subfolder(self):
        self.assertEqual(proofs.parse_hub_pin(MLX + "/q8"),
                         {"repository": "SceneWorks/qwen3.8-27b-abliterated-mlx", "revision": REV, "subfolder": "q8"})
        self.assertEqual(proofs.parse_hub_pin(MLX)["subfolder"], "")
        self.assertEqual(proofs.parse_hub_pin(MLX + "/bf16/vae")["subfolder"], "bf16/vae")
        for bad in (MLX[:-1], MLX + "/", "SceneWorks@" + REV, MLX + "//q8"):
            with self.subTest(pin=bad), self.assertRaises(ValueError):
                proofs.parse_hub_pin(bad)


class RequestBinding(unittest.TestCase):
    def bind(self, requested, *rows):
        return proofs.bind_hub_snapshots(rows or (row(*SLOTS),), {"hub_snapshots": requested} if requested else {})

    def test_request_bound_and_choice_slots_take_the_request_and_fixed_pins_do_not(self):
        bound = self.bind({"FULL": FULL, "PLAIN": MLX + "/q4"})
        self.assertEqual(bound["PLAIN"]["subfolder"], "q4")
        self.assertEqual(bound["FULL"]["repository"], "SceneWorks/qwen3.8-27b-full-mlx")
        fixed = row({"environment": "FULL", "pin": MLX + "/q4"}, required=("FULL",))
        self.assertEqual(self.bind({}, fixed)["FULL"]["subfolder"], "q4")
        with self.assertRaisesRegex(ValueError, "cannot rebind"):
            self.bind({"FULL": MLX + "/q8"}, fixed)

    def test_missing_unlisted_or_undeclared_bindings_are_refused(self):
        with self.assertRaisesRegex(ValueError, "must bind"):
            self.bind({"PLAIN": MLX + "/q8"})
        with self.assertRaisesRegex(ValueError, "registered pins"):
            self.bind({"FULL": FULL, "PLAIN": MLX + "/bf16"})
        with self.assertRaisesRegex(ValueError, "no selected case declares"):
            self.bind({"FULL": FULL, "PLAIN": MLX + "/q8", "EXTRA": FULL})
        with self.assertRaisesRegex(ValueError, "different snapshots"):
            self.bind({}, row({"environment": "FULL", "pin": MLX + "/q8"}, required=("FULL",)),
                      row({"environment": "FULL", "pin": MLX + "/q4"}, required=("FULL",)))

    def test_request_field_is_a_name_to_pin_map(self):
        policy = {"backend": "darwin-mlx"}
        for bad in ([], {}, {"FULL": 1}):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                proofs.request(json.dumps({"selection": "x", "policy": policy, "hub_snapshots": bad}), "mlx")
        proofs.request(json.dumps({"selection": "x", "policy": policy, "hub_snapshots": {"FULL": FULL}}), "mlx")


class RegisteredHubSnapshotCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {proofs.identity(item): item for item in json.loads(proofs.REGISTRY_PATH.read_text(encoding="utf-8"))}

    def test_every_registered_slot_set_satisfies_the_slot_contract(self):
        for binding, item in self.rows.items():
            if "hub_snapshots" in item:
                with self.subTest(binding=binding):
                    proofs.validate_hub_snapshot_slots(item)

    def test_dspine_smoke_binds_target_and_draft_by_fixed_pin(self):
        item = self.rows["mlx-llm/integration/block_draft::qwen38_27b_dspine_accepts_more_than_one_token_per_pass"]
        bound = proofs.bind_hub_snapshots((item,), {})
        self.assertEqual(bound["MLX_LLM_BLOCK_DRAFT_TARGET"],
                         {"repository": "SceneWorks/qwen3.8-27b-abliterated-mlx", "revision": REV, "subfolder": "q4"})
        self.assertEqual(bound["MLX_LLM_BLOCK_DRAFT_SOURCE"]["repository"], "SceneWorks/qwen3.8-27b-dspine-b16")
        self.assertEqual(bound["MLX_LLM_BLOCK_DRAFT_SOURCE"]["subfolder"], "")

    def test_the_nax_route_selects_only_the_rw_llm_mac(self):
        self.assertIn("rw-llm", proofs.NAX_ROUTES["nax"]["labels"])
        self.assertEqual(proofs.NAX_ROUTES["nax"]["name"], "nax-macos-2")


def sibling(name, data):
    return {"rfilename": name, "size": len(data),
            "lfs": {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}}


class Staging(unittest.TestCase):
    FILES = {"q8/config.json": b"{}", "q8/model.safetensors": b"weights", "q4/model.safetensors": b"other"}
    PIN = {"repository": "SceneWorks/qwen3.8-27b-abliterated-mlx", "revision": REV, "subfolder": "q8"}

    def manifest(self, repository, revision):
        return "url", b"raw", {name: sibling(name, data) for name, data in self.FILES.items()}

    def fetcher(self, calls, placed_by_another_job=False):
        def fetch(url, destination):
            self.assertFalse(destination.exists(), "an existing file is never replaced")
            calls.append(url)
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = self.FILES[url.split(f"/resolve/{REV}/", 1)[1]]
            destination.write_bytes(data)
            return len(data), hashlib.sha256(data).hexdigest(), None, placed_by_another_job
        return fetch

    def verify(self, repository, revision, snapshot, files):
        metadata = json.dumps({"id": repository, "sha": revision, "private": False, "gated": False,
                               "siblings": [sibling(name, data) for name, data in self.FILES.items()]}).encode()
        with patch("scripts.ci.pinned_hub_source.urllib.request.urlopen", return_value=io.BytesIO(metadata)):
            return verify_pinned_hub_files(repository, revision, snapshot, files)

    def stage(self, hub, calls, free=10**15):
        return staging.stage({"PLAIN": self.PIN}, hub, manifest=self.manifest, fetch=self.fetcher(calls),
                             verify=self.verify, free_bytes=lambda path: free)

    def test_stages_only_absent_subfolder_files_keeps_existing_and_verifies_all(self):
        with tempfile.TemporaryDirectory() as temporary:
            hub = Path(temporary)
            base = staging.snapshot_dir(hub, self.PIN["repository"], REV)
            (base / "q8").mkdir(parents=True)
            (base / "q8/config.json").write_bytes(b"{}")
            calls = []
            receipt = self.stage(hub, calls)["PLAIN"]
            self.assertEqual(calls, [f"https://huggingface.co/{self.PIN['repository']}/resolve/{REV}/q8/model.safetensors"])
            self.assertEqual(receipt["path"], str(base / "q8"))
            self.assertEqual(receipt["status"], "hub_content_verified")
            self.assertEqual(receipt["kept_existing"], ["q8/config.json"])
            self.assertFalse((base / "q4").exists())
            calls.clear()
            self.assertEqual(self.stage(hub, calls)["PLAIN"]["fetched"], [])
            self.assertEqual(calls, [])

    def test_existing_file_with_other_bytes_is_refused_not_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            hub = Path(temporary)
            base = staging.snapshot_dir(hub, self.PIN["repository"], REV)
            (base / "q8").mkdir(parents=True)
            (base / "q8/model.safetensors").write_bytes(b"WEIGHTS")
            with self.assertRaisesRegex(ValueError, "content mismatch"):
                self.stage(hub, [])
            self.assertEqual((base / "q8/model.safetensors").read_bytes(), b"WEIGHTS")

    def test_two_stagers_of_the_same_file_download_it_once(self):
        # Two runner listeners on one host stage the same snapshot; the second waits on the lock.
        with tempfile.TemporaryDirectory() as temporary:
            hub = Path(temporary)
            base = staging.snapshot_dir(hub, self.PIN["repository"], REV)
            (base / "q8").mkdir(parents=True)
            (base / "q8/config.json").write_bytes(b"{}")
            calls, results, errors = [], {}, []
            first_fetching, second_waiting = threading.Event(), threading.Event()
            fetch = self.fetcher(calls)

            def slow_fetch(url, destination):
                first_fetching.set()
                self.assertTrue(second_waiting.wait(10), "the second stager never waited on the lock")
                return fetch(url, destination)

            real_try_lock = staging._try_lock

            def try_lock(handle):
                locked = real_try_lock(handle)
                if not locked:
                    second_waiting.set()
                return locked

            def run(name, fetcher):
                try:
                    results[name] = staging.stage({"PLAIN": self.PIN}, hub, manifest=self.manifest, fetch=fetcher,
                                                  verify=self.verify, free_bytes=lambda path: 10**15)["PLAIN"]
                except Exception as error:  # surfaced below
                    errors.append(error)

            with patch.object(staging, "_try_lock", side_effect=try_lock), \
                    patch.object(staging, "LOCK_POLL_SECONDS", 0.01), patch("builtins.print"):
                first = threading.Thread(target=run, args=("first", slow_fetch))
                first.start()
                self.assertTrue(first_fetching.wait(10))
                second = threading.Thread(target=run, args=("second", fetch))
                second.start()
                first.join(20)
                second.join(20)
            self.assertEqual(errors, [])
            self.assertEqual(len(calls), 1)
            self.assertEqual(results["first"]["fetched"], ["q8/model.safetensors"])
            self.assertEqual(results["second"]["fetched"], [])
            self.assertIn("q8/model.safetensors", results["second"]["kept_existing"])
            for receipt in results.values():
                self.assertEqual(receipt["status"], "hub_content_verified")
            self.assertEqual([p.name for p in (base / "q8").iterdir() if p.name.endswith(".partial")], [])

    def test_staging_lock_lives_outside_the_snapshot_and_is_released_after_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            hub = Path(temporary)
            with staging.staging_lock(hub, self.PIN["repository"], REV, "q8/model.safetensors"):
                pass
            with staging.staging_lock(hub, self.PIN["repository"], REV, "q8/model.safetensors"):
                pass
            lock = (hub / ".locks/models--SceneWorks--qwen3.8-27b-abliterated-mlx/model-proof-staging"
                    / REV / "q8/model.safetensors.lock")
            self.assertTrue(lock.is_file())
            self.assertFalse(staging.snapshot_dir(hub, self.PIN["repository"], REV).exists())

    def test_a_file_another_job_placed_mid_download_is_kept_not_fetched(self):
        with tempfile.TemporaryDirectory() as temporary:
            hub = Path(temporary)
            receipt = staging.stage({"PLAIN": self.PIN}, hub, manifest=self.manifest,
                                    fetch=self.fetcher([], placed_by_another_job=True), verify=self.verify,
                                    free_bytes=lambda path: 10**15)["PLAIN"]
            self.assertEqual(receipt["fetched"], [])
            self.assertEqual(receipt["placed_by_another_job"], ["q8/config.json", "q8/model.safetensors"])
            self.assertEqual(receipt["kept_existing"], ["q8/config.json", "q8/model.safetensors"])
            self.assertEqual(receipt["status"], "hub_content_verified")

    def test_only_lock_contention_waits_and_other_lock_errors_are_raised(self):
        def failing(code):
            def flock(*_args):
                raise OSError(code, os.strerror(code))
            return flock
        with tempfile.TemporaryDirectory() as temporary, open(Path(temporary) / "x.lock", "a+b") as handle:
            for code in (errno.EWOULDBLOCK, errno.EAGAIN):
                with self.subTest(posix_busy=code), patch("fcntl.flock", side_effect=failing(code)):
                    self.assertFalse(staging._try_lock(handle))
            for code in (errno.ENOLCK, errno.ENOTSUP, errno.EBADF, errno.EINVAL):
                with self.subTest(posix_error=code), patch("fcntl.flock", side_effect=failing(code)), \
                        self.assertRaises(OSError):
                    staging._try_lock(handle)
            for code, busy in ((errno.EACCES, True), (errno.EDEADLK, True), (errno.ENOLCK, False),
                               (errno.EBADF, False)):
                fake = SimpleNamespace(LK_NBLCK=2, locking=failing(code))
                with self.subTest(msvcrt=code), patch.object(staging, "os", SimpleNamespace(name="nt")), \
                        patch.dict(sys.modules, {"msvcrt": fake}):
                    if busy:
                        self.assertFalse(staging._try_lock(handle))
                    else:
                        with self.assertRaises(OSError):
                            staging._try_lock(handle)

    def test_a_long_wait_on_the_staging_lock_is_logged_periodically(self):
        clock = {"now": 0.0}
        attempts = iter([False] * 7 + [True])
        def sleep(seconds):
            clock["now"] += 60
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(staging, "_try_lock", side_effect=lambda _handle: next(attempts)), \
                patch.object(staging.time, "monotonic", side_effect=lambda: clock["now"]), \
                patch.object(staging.time, "sleep", side_effect=sleep), patch("builtins.print") as printed:
            with staging.staging_lock(Path(temporary), self.PIN["repository"], REV, "q8/model.safetensors"):
                pass
        lines = [call.args[0] for call in printed.call_args_list]
        self.assertTrue(lines[0].startswith("waiting for another job staging"))
        self.assertEqual([line.split(" (")[1].split(" s)")[0] for line in lines[1:]], ["180", "360"])

    def test_disk_shortfall_is_refused_before_any_fetch(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls = []
            with self.assertRaisesRegex(ValueError, "GiB free"):
                self.stage(Path(temporary), calls, free=staging.DISK_RESERVE_BYTES)
            self.assertEqual(calls, [])

    def test_proof_requires_the_receipt_and_the_bound_paths(self):
        value = {"hub_snapshots": {"FULL": FULL, "PLAIN": MLX + "/q8"}}
        bound = proofs.bind_hub_snapshots((row(*SLOTS),), value)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            snapshots = {name: {**pin, "path": f"/hub/{name}", "status": "hub_content_verified"}
                         for name, pin in bound.items()}
            path.write_text(json.dumps({"snapshots": snapshots}), encoding="utf-8")
            env = {"MODEL_PROOF_HUB_SNAPSHOT_RECEIPT": str(path), "FULL": "/hub/FULL", "PLAIN": "/hub/PLAIN"}
            self.assertEqual(set(proofs.staged_hub_snapshots(row(*SLOTS), value, env)), {"FULL", "PLAIN"})
            for broken in ({**env, "PLAIN": "/elsewhere"}, {k: v for k, v in env.items() if k != "MODEL_PROOF_HUB_SNAPSHOT_RECEIPT"}):
                with self.subTest(env=broken), self.assertRaises(ValueError):
                    proofs.staged_hub_snapshots(row(*SLOTS), value, broken)
            with self.assertRaises(ValueError):
                proofs.staged_hub_snapshots(row(*SLOTS), {"hub_snapshots": {"FULL": FULL, "PLAIN": MLX + "/q4"}}, env)


class WorkflowStaging(unittest.TestCase):
    def test_staging_runs_before_the_proof_on_every_platform_from_the_validated_plan(self):
        steps = yaml.safe_load((ROOT / ".github/workflows/model-inference-proofs.yml").read_text(encoding="utf-8"))["jobs"]["prove"]["steps"]
        names = [step.get("name") for step in steps]
        for platform in ("Linux/macOS", "Windows"):
            index = names.index(f"Stage and bind declared pinned Hub snapshots ({platform})")
            step = steps[index]
            self.assertIn("matrix.asset_setup == 'hub-snapshots'", step["if"])
            self.assertIn("--plan", step["run"])
            self.assertIn("stage_pinned_hub_snapshots.py", step["run"])
            self.assertIn("MODEL_PROOF_HUB_ROOT", step["env"])
            self.assertLess(index, next(i for i, name in enumerate(names)
                                        if name and name.startswith("Run exact registered model tests")))


if __name__ == "__main__":
    unittest.main()
