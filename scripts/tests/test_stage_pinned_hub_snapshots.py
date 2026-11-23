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
from scripts.ci import flux1_conditional_asset_inventory as inventory
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

    def test_real_weight_tier_cases_bind_plain_tiers_by_pin_and_full_tiers_by_request(self):
        for binding, repository in (
                ("mlx-llm/integration/qwen38_quantized_tiers::qwen38_full_tier_replies_agree_with_the_plain_tier",
                 "SceneWorks/qwen3.8-27b-abliterated-mlx@76f3c445cd5347fdb85fc24066350724fe272017"),
                ("candle-llm/qwen38_quantized_tiers/qwen38_full_tier_replies_agree_with_the_plain_tier",
                 "SceneWorks/qwen3.8-27b-abliterated-candle@accaccc109912b9b194f7d5a47f2123fdc70672e")):
            slots = {slot["environment"]: slot for slot in self.rows[binding]["hub_snapshots"]}
            self.assertEqual(slots["QWEN38_FULL_TIER_SNAPSHOT"], {"environment": "QWEN38_FULL_TIER_SNAPSHOT", "request_bound": True})
            self.assertEqual(slots["QWEN38_PLAIN_TIER_SNAPSHOT"]["choices"], [repository + "/q8", repository + "/q4"])

    def test_tiny_candle_tier_case_declares_cpu_default_and_cuda_recipe(self):
        item = self.rows["candle-llm/candle_llm/provider::tests::qwen35_fully_quantized_tiers_decode_like_the_plain_tier"]
        self.assertEqual(proofs.lane(proofs.configured_case(item)), "candle-cpu")
        cuda = proofs.configured_case(item, "source-fixed-candle-cuda")
        self.assertEqual(proofs.lane(cuda), "candle-cuda")
        self.assertIn("cuda", cuda["cargo_features"])

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
        # Routed by label only: the Macs are interchangeable, so the route names no machine.
        self.assertIn("rw-llm", proofs.NAX_ROUTES["nax"]["labels"])
        self.assertNotIn("name", proofs.NAX_ROUTES["nax"])


def sibling(name, data):
    return {"rfilename": name, "size": len(data),
            "lfs": {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}}


class Staging(unittest.TestCase):
    FILES = {"q8/config.json": b"{}", "q8/model.safetensors": b"weights", "q4/model.safetensors": b"other"}
    PIN = {"repository": "SceneWorks/qwen3.8-27b-abliterated-mlx", "revision": REV, "subfolder": "q8"}

    def manifest(self, repository, revision):
        return "url", b"raw", {name: sibling(name, data) for name, data in self.FILES.items()}

    def fetcher(self, calls, placed_by_another_job=False):
        def fetch(url, destination, *, expected_bytes, expected_sha256, publish):
            self.assertFalse(destination.exists(), "an existing file is never replaced")
            calls.append(url)
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = self.FILES[url.split(f"/resolve/{REV}/", 1)[1]]
            self.assertEqual(expected_bytes, len(data))
            self.assertEqual(expected_sha256, hashlib.sha256(data).hexdigest())
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

    def stage_downloads(self, hub, bodies, *, manifest=None):
        def open_response(request, **_kwargs):
            data = bodies[request.full_url.split(f"/resolve/{REV}/", 1)[1]]
            reply = io.BytesIO(data)
            reply.headers = {"Content-Length": str(len(data))}
            return reply

        with patch("urllib.request.build_opener", return_value=SimpleNamespace(open=open_response)):
            return staging.stage({"PLAIN": self.PIN}, hub, manifest=manifest or self.manifest,
                                 verify=self.verify, free_bytes=lambda _path: 10**15)["PLAIN"]

    def test_stages_only_absent_subfolder_files_keeps_existing_and_verifies_all(self):
        with tempfile.TemporaryDirectory() as temporary:
            hub = Path(temporary).resolve()
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

    @unittest.skipIf(os.name == "nt", "Windows publishes with no-replace rename")
    def test_unsupported_darwin_hardlink_publishes_a_verified_hub_blob(self):
        with tempfile.TemporaryDirectory() as temporary:
            hub = Path(temporary).resolve()
            base = staging.snapshot_dir(hub, self.PIN["repository"], REV)
            destination = base / "q8/model.safetensors"
            destination.parent.mkdir(parents=True)
            (base / "q8/config.json").write_bytes(b"{}")
            blobs = base.parent.parent / "blobs"
            symlink = os.symlink

            def atomic(target, final):
                self.assertEqual(final, destination)
                self.assertFalse(Path(target).is_absolute())
                blob = (final.parent / target).resolve(strict=True)
                self.assertTrue(blob.is_relative_to(blobs))
                self.assertEqual(blob.read_bytes(), b"weights")
                self.assertFalse(final.exists())
                symlink(target, final)

            error = OSError(errno.ENOTSUP, "hard links unavailable")
            with (patch.object(sys, "platform", "darwin"), patch("os.link", side_effect=error),
                  patch.object(inventory, "_renamex_excl", return_value=False),
                  patch("os.symlink", side_effect=atomic) as commit, patch("os.replace") as replace):
                result = self.stage_downloads(hub, self.FILES)
            commit.assert_called_once()
            replace.assert_not_called()
            self.assertEqual(result["fetched"], ["q8/model.safetensors"])
            self.assertTrue(destination.is_symlink())
            self.assertEqual(destination.read_bytes(), b"weights")
            self.assertEqual(list(destination.parent.glob("*.partial")), [])

    @unittest.skipIf(os.name == "nt", "Windows publishes with no-replace rename")
    def test_unsupported_hardlink_preserves_and_reconciles_a_concurrent_file(self):
        for peer_data in (b"weights", b"WEIGHTS"):
            with self.subTest(peer_data=peer_data), tempfile.TemporaryDirectory() as temporary:
                hub = Path(temporary).resolve()
                base = staging.snapshot_dir(hub, self.PIN["repository"], REV)
                destination = base / "q8/model.safetensors"
                destination.parent.mkdir(parents=True)
                (base / "q8/config.json").write_bytes(b"{}")

                symlink = os.symlink

                def placed(source, final):
                    # Place the file after the final path guard, at the actual publication call.
                    final.write_bytes(peer_data)
                    symlink(source, final)

                error = OSError(errno.ENOTSUP, "hard links unavailable")
                with (patch.object(sys, "platform", "darwin"), patch("os.link", side_effect=error),
                      patch.object(inventory, "_renamex_excl", return_value=False),
                      patch("os.symlink", side_effect=placed), patch("os.replace") as replace):
                    if peer_data == b"weights":
                        result = self.stage_downloads(hub, self.FILES)
                        self.assertEqual(result["placed_by_another_job"], ["q8/model.safetensors"])
                        self.assertEqual(result["fetched"], [])
                    else:
                        with self.assertRaisesRegex(ValueError, "placed concurrently with other bytes"):
                            self.stage_downloads(hub, self.FILES)
                replace.assert_not_called()
                self.assertFalse(destination.is_symlink())
                self.assertEqual(destination.read_bytes(), peer_data)
                self.assertEqual(list(destination.parent.glob("*.partial")), [])
                self.assertEqual(list((base.parent.parent / "blobs").iterdir()), [])

    @unittest.skipIf(os.name == "nt", "Windows publishes with no-replace rename")
    def test_unavailable_publication_primitive_leaves_no_snapshot_leaf_or_owned_blob(self):
        for platform in ("darwin", "linux"):
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as temporary:
                hub = Path(temporary).resolve()
                base = staging.snapshot_dir(hub, self.PIN["repository"], REV)
                destination = base / "q8/model.safetensors"
                destination.parent.mkdir(parents=True)
                (base / "q8/config.json").write_bytes(b"{}")
                error = OSError(errno.ENOTSUP, "publication unavailable")

                with (patch.object(sys, "platform", platform), patch("os.link", side_effect=error),
                      patch.object(inventory, "_renamex_excl", return_value=False),
                      patch("os.symlink", side_effect=error) as link, patch("os.replace") as replace):
                    with self.assertRaises(OSError) as failure:
                        self.stage_downloads(hub, self.FILES)
                self.assertEqual(failure.exception.errno, errno.ENOTSUP)
                replace.assert_not_called()
                self.assertFalse(os.path.lexists(destination))
                self.assertEqual(list(destination.parent.glob("*.partial")), [])
                blobs = base.parent.parent / "blobs"
                if platform == "darwin":
                    link.assert_called_once()
                    self.assertEqual(list(blobs.iterdir()), [])
                else:
                    link.assert_not_called()
                    self.assertFalse(blobs.exists())

    @unittest.skipIf(os.name == "nt", "Windows publishes with no-replace rename")
    def test_darwin_fallback_rejects_a_redirected_blob_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            hub, outside = root / "hub", root / "outside"
            hub.mkdir()
            outside.mkdir()
            base = staging.snapshot_dir(hub, self.PIN["repository"], REV)
            destination = base / "q8/model.safetensors"
            destination.parent.mkdir(parents=True)
            (base / "q8/config.json").write_bytes(b"{}")
            (base.parent.parent / "blobs").symlink_to(outside, target_is_directory=True)

            error = OSError(errno.ENOTSUP, "hard links unavailable")
            with (patch.object(sys, "platform", "darwin"), patch("os.link", side_effect=error),
                  patch.object(inventory, "_renamex_excl", return_value=False),
                  self.assertRaisesRegex(ValueError, "linked evidence ancestor")):
                self.stage_downloads(hub, self.FILES)
            self.assertEqual(list(outside.iterdir()), [])
            self.assertFalse(os.path.lexists(destination))
            self.assertEqual(list(destination.parent.glob("*.partial")), [])

    @unittest.skipIf(os.name == "nt", "Windows publishes with no-replace rename")
    def test_darwin_fallback_cleans_only_unpublished_blob_on_interruption(self):
        for operation in ("rename", "symlink"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary:
                hub = Path(temporary).resolve()
                base = staging.snapshot_dir(hub, self.PIN["repository"], REV)
                destination = base / "q8/model.safetensors"
                destination.parent.mkdir(parents=True)
                (base / "q8/config.json").write_bytes(b"{}")
                system_call = getattr(os, operation)

                def interrupted(*args):
                    system_call(*args)
                    raise KeyboardInterrupt()

                error = OSError(errno.ENOTSUP, "hard links unavailable")
                with (patch.object(sys, "platform", "darwin"), patch("os.link", side_effect=error),
                      patch.object(inventory, "_renamex_excl", return_value=False),
                      patch.object(os, operation, side_effect=interrupted),
                      self.assertRaises(KeyboardInterrupt)):
                    self.stage_downloads(hub, self.FILES)
                self.assertEqual(list(destination.parent.glob("*.partial")), [])
                blobs = base.parent.parent / "blobs"
                if operation == "rename":
                    self.assertFalse(os.path.lexists(destination))
                    self.assertEqual(list(blobs.iterdir()), [])
                else:
                    self.assertTrue(destination.is_symlink())
                    self.assertEqual(destination.read_bytes(), b"weights")
                    self.assertEqual(list(blobs.iterdir()), [destination.resolve().parent])

    def test_refuses_wrong_manifest_digest_before_publishing_lfs_or_git_bytes(self):
        leaf = "q8/model.safetensors"
        for kind in ("lfs", "git"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                hub = Path(temporary).resolve()
                base = staging.snapshot_dir(hub, self.PIN["repository"], REV)
                destination = base / leaf
                destination.parent.mkdir(parents=True)
                (base / "q8/config.json").write_bytes(b"{}")

                def manifest(repository, revision):
                    url, raw, defaults = self.manifest(repository, revision)
                    if kind == "git":
                        good = self.FILES[leaf]
                        defaults[leaf] = {"size": len(good),
                                          "blobId": hashlib.sha1(b"blob 7\0" + good).hexdigest()}
                    return url, raw, defaults

                with self.assertRaisesRegex(ValueError, "SHA-(256|1)"):
                    self.stage_downloads(hub, {**self.FILES, leaf: b"WEIGHTS"}, manifest=manifest)
                self.assertFalse(destination.exists())
                self.assertEqual(list(destination.parent.glob("*.partial")), [])

    def test_refuses_redirected_snapshot_and_lock_paths_before_fetching(self):
        for location in ("snapshot-parent", "snapshot-leaf", "lock-parent", "lock-leaf"):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                hub, outside = root / "hub", root / "outside"
                hub.mkdir()
                outside.mkdir()
                target = outside / "target"
                target.write_bytes(b"weights")
                base = staging.snapshot_dir(hub, self.PIN["repository"], REV)
                paths = {
                    "snapshot-parent": base / "q8",
                    "snapshot-leaf": base / "q8/model.safetensors",
                    "lock-parent": hub / ".locks",
                    "lock-leaf": staging.staging_lock_path(hub, self.PIN["repository"], REV,
                                                          "q8/model.safetensors"),
                }
                link = paths[location]
                link.parent.mkdir(parents=True, exist_ok=True)
                directory = location.endswith("parent")
                try:
                    link.symlink_to(outside if directory else target, target_is_directory=directory)
                except (NotImplementedError, PermissionError):
                    self.skipTest("this host cannot create symlinks for the fixture")
                calls = []
                with self.assertRaisesRegex(ValueError, "linked evidence ancestor|escapes configured model hub"):
                    self.stage(hub, calls)
                self.assertEqual(calls, [])
                self.assertEqual(list(outside.iterdir()), [target])
                self.assertEqual(target.read_bytes(), b"weights")

    def test_keeps_an_existing_hub_blob_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            hub = Path(temporary).resolve()
            base = staging.snapshot_dir(hub, self.PIN["repository"], REV)
            blob = base.parent.parent / "blobs" / "existing"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"weights")
            existing = base / "q8/model.safetensors"
            existing.parent.mkdir(parents=True)
            try:
                existing.symlink_to(blob)
            except (NotImplementedError, PermissionError):
                self.skipTest("this host cannot create symlinks for the fixture")
            result = self.stage(hub, [])["PLAIN"]
            self.assertEqual(result["kept_existing"], ["q8/model.safetensors"])
            self.assertTrue(existing.is_symlink())

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

            def slow_fetch(url, destination, **options):
                first_fetching.set()
                self.assertTrue(second_waiting.wait(10), "the second stager never waited on the lock")
                return fetch(url, destination, **options)

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

    def test_shared_locker_validates_the_opened_lock_before_seeding_or_waiting(self):
        with tempfile.TemporaryDirectory() as temporary:
            hub = Path(temporary).resolve()
            file = "q8/model.safetensors"
            with (patch("os.path.samestat", return_value=False), patch.object(staging, "_try_lock") as attempt,
                  self.assertRaisesRegex(ValueError, "staging lock changed during open")):
                with staging.staging_lock(hub, self.PIN["repository"], REV, file):
                    self.fail("an unbound lock should never be acquired")
            attempt.assert_not_called()
            lock = staging.staging_lock_path(hub, self.PIN["repository"], REV, file)
            self.assertEqual(lock.read_bytes(), b"")

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
            self.assertEqual(set(proofs.staged_hub_snapshots(row(*SLOTS), bound, env)), {"FULL", "PLAIN"})
            for broken in ({**env, "PLAIN": "/elsewhere"}, {k: v for k, v in env.items() if k != "MODEL_PROOF_HUB_SNAPSHOT_RECEIPT"}):
                with self.subTest(env=broken), self.assertRaises(ValueError):
                    proofs.staged_hub_snapshots(row(*SLOTS), bound, broken)
            other = proofs.bind_hub_snapshots((row(*SLOTS),), {"hub_snapshots": {"FULL": FULL, "PLAIN": MLX + "/q4"}})
            with self.assertRaises(ValueError):
                proofs.staged_hub_snapshots(row(*SLOTS), other, env)
            with self.assertRaisesRegex(ValueError, "not bound by the selection"):
                proofs.staged_hub_snapshots(row(*SLOTS), {"FULL": bound["FULL"]}, env)

    def test_a_mixed_selection_checks_each_case_against_its_own_slots(self):
        # A package-wide mlx-llm selection: the full-tier case's slots are request-bound, the
        # block-draft case pins its own. The request binds the full-tier slots, which the block-draft
        # case does not declare; the selection as a whole does.
        rows = {proofs.identity(item): item for item in json.loads(proofs.REGISTRY_PATH.read_text(encoding="utf-8"))}
        tier = rows["mlx-llm/integration/qwen38_quantized_tiers::qwen38_full_tier_replies_agree_with_the_plain_tier"]
        draft = rows["mlx-llm/integration/block_draft::qwen38_27b_dspine_accepts_more_than_one_token_per_pass"]
        plain = next(slot for slot in tier["hub_snapshots"] if slot["environment"] == "QWEN38_PLAIN_TIER_SNAPSHOT")
        value = {"hub_snapshots": {"QWEN38_FULL_TIER_SNAPSHOT": FULL, "QWEN38_PLAIN_TIER_SNAPSHOT": plain["choices"][0]}}
        with self.assertRaisesRegex(ValueError, "no selected case declares"):
            proofs.bind_hub_snapshots((draft,), value)
        bound = proofs.bind_hub_snapshots((tier, draft), value)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            snapshots = {name: {**pin, "path": f"/hub/{name}", "status": "hub_content_verified"}
                         for name, pin in bound.items()}
            path.write_text(json.dumps({"snapshots": snapshots}), encoding="utf-8")
            env = {"MODEL_PROOF_HUB_SNAPSHOT_RECEIPT": str(path), **{name: f"/hub/{name}" for name in bound}}
            for item in (tier, draft):
                own = {slot["environment"] for slot in item["hub_snapshots"]}
                with self.subTest(case=proofs.identity(item)):
                    self.assertEqual(set(proofs.staged_hub_snapshots(item, bound, env)), own)
            self.assertNotIn("QWEN38_FULL_TIER_SNAPSHOT", proofs.staged_hub_snapshots(draft, bound, env))


class WorkflowStaging(unittest.TestCase):
    def test_unmounted_macos_volume_hub_root_falls_back_consistently_with_campaign_planner(self):
        from scripts.release import speculative_bench_campaign as campaign

        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            absent = "/Volumes/SceneWorksAbsentModelsVolumeForTest/huggingface/hub"
            env = {"MODEL_PROOF_HUB_ROOT": absent, "HOME": str(home)}
            expected = home / "Library/Caches/SceneWorks/model-proof/huggingface/hub"
            self.assertEqual(staging.resolve_hub_root(env, platform="darwin"), expected)
            self.assertEqual(staging.resolve_hub_root(env, platform="linux"), Path(absent))
            with patch.object(staging.sys, "platform", "darwin"):
                resolved_path, pin = campaign.hub_pin(
                    f"hub:SceneWorks/qwen3.8-27b-abliterated-mlx@{REV}/q4", env, "row r"
                )
            self.assertEqual(pin["subfolder"], "q4")
            self.assertEqual(
                resolved_path,
                str(
                    Path(os.path.realpath(expected))
                    / "models--SceneWorks--qwen3.8-27b-abliterated-mlx"
                    / "snapshots"
                    / REV
                    / "q4"
                ),
            )
            local_root = str(home / "custom-hub")
            self.assertEqual(
                staging.resolve_hub_root({"MODEL_PROOF_HUB_ROOT": local_root, "HOME": str(home)}, platform="darwin"),
                Path(local_root),
            )

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
