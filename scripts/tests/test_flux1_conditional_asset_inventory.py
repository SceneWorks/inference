"""Weights-free checks for the manual conditional asset inventory protocol."""
import hashlib
import io
from contextlib import redirect_stderr
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.ci import flux1_conditional_asset_inventory as assets
from scripts.ci.flux1_pair import conditional

OTHER_REV = "c" * 40


def put(root, relative, content=b"fixed"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def snapshot(hub, role, revision=None):
    repo, _, pinned = assets.PRODUCT_SOURCES[role]
    return assets.cache_dir(hub, repo) / "snapshots" / (revision or pinned)


def install(hub, role, content=None, revision=None):
    """Place the product file for `role` at its pinned (or the given) snapshot revision."""
    name = assets.PRODUCT_SOURCES[role][1]
    return put(hub, (snapshot(hub, role, revision) / name).relative_to(hub),
               content if content is not None else role.encode())


def install_gemma(hub, revision=None, extra=()):
    directory = snapshot(hub, "pid_gemma", revision)
    for name in ("config.json", "tokenizer.json", "LICENSE", *extra):
        put(hub, (directory / name).relative_to(hub), name.encode())
    shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    for name in shards:
        put(hub, (directory / name).relative_to(hub), name.encode())
    put(hub, (directory / "model.safetensors.index.json").relative_to(hub),
        json.dumps({"weight_map": {"a": shards[0], "b": shards[1]}}).encode())
    return directory


def install_all(hub):
    for role in assets.PRODUCT_SOURCES:
        if role != "pid_gemma":
            install(hub, role)
    return install_gemma(hub)


def no_listing():
    """Fail on any directory listing in this process (the child only stats and reads)."""
    def refuse(*args, **kwargs):
        raise AssertionError(f"directory listed: {args!r}")
    return (patch.object(assets.os, "scandir", side_effect=refuse),
            patch.object(assets.os, "listdir", side_effect=refuse),
            patch.object(assets.os, "walk", side_effect=refuse),
            patch.object(Path, "iterdir", refuse),
            patch.object(Path, "glob", refuse),
            patch.object(Path, "rglob", refuse))


class ConditionalAssetInventory(unittest.TestCase):
    def inventory(self, hub, runner="nax", **kwargs):
        guards = no_listing()
        for guard in guards:
            guard.start()
        try:
            return assets.inventory(hub, runner, **kwargs)
        finally:
            for guard in guards:
                guard.stop()

    def test_every_case_role_has_one_product_source(self):
        roles = {role for _, _, required in conditional.CASES.values() for role in required}
        self.assertEqual(set(assets.PRODUCT_SOURCES), roles)
        for repo, _, revision in assets.PRODUCT_SOURCES.values():
            self.assertNotIn(repo, {"SceneWorks/flux1-dev-mlx", "SceneWorks/flux1-schnell-mlx",
                                    "black-forest-labs/FLUX.1-dev"})
            self.assertRegex(revision, r"^[0-9a-f]{40}$")
        self.assertEqual(assets.PRODUCT_SOURCES["pid_decoder"][:2],
                         ("SceneWorks/pid-flux", "pid_flux_2kto4k.safetensors"))

    def test_pinned_product_files_are_selected_without_listing_any_directory(self):
        with TemporaryDirectory() as tmp:
            hub = Path(tmp).resolve() / "hub"
            hub.mkdir()
            control = install(hub, "control", b"control")
            gemma = install_all(hub)
            log = StringIO()
            with redirect_stderr(log):
                result = self.inventory(hub)
            self.assertEqual(result["protocol"], "flux1-conditional-asset-inventory-v2")
            for key in ("missing_roles", "off_pin_roles", "unreachable_roles", "incomplete_roles"):
                self.assertEqual(result[key], [], key)
            self.assertTrue(all(row["ready"] for row in result["cases"].values()))
            row = result["roles"]["control"]["candidates"][0]
            self.assertEqual(row["path"], str(control.resolve()))
            self.assertEqual(row["sha256"], hashlib.sha256(b"control").hexdigest())
            self.assertEqual(row["revision"], assets.PRODUCT_SOURCES["control"][2])
            self.assertEqual(row["revision_provenance"], "product-manifest-pin")
            files = result["roles"]["pid_gemma"]["candidates"][0]["files"]
            self.assertEqual(set(files), {"config.json", "tokenizer.json", "LICENSE",
                                          "model.safetensors.index.json",
                                          "model-00001-of-00002.safetensors",
                                          "model-00002-of-00002.safetensors"})
            self.assertEqual(files["config.json"], hashlib.sha256(b"config.json").hexdigest())
            self.assertEqual(result["roles"]["pid_gemma"]["candidates"][0]["path"], str(gemma))
            self.assertIn("role=control repo=Shakker-Labs", log.getvalue())
            self.assertNotIn("pulid-real", self.inventory(hub, "cuda-windows")["cases"])

    def test_other_repositories_with_the_same_file_names_are_ignored(self):
        with TemporaryDirectory() as tmp:
            hub = Path(tmp).resolve() / "hub"
            hub.mkdir()
            control = install(hub, "control")
            decoder = install(hub, "pid_decoder", b"2kto4k")
            # The Windows inventory found these as rival `control` / `pid_decoder` candidates.
            for repo, name in (("InstantX--InstantID", "ControlNetModel/diffusion_pytorch_model.safetensors"),
                               ("SceneWorks--qwen-image-2512-fun-controlnet-union", "diffusion_pytorch_model.safetensors"),
                               ("nvidia--PiD", "flux_2k.safetensors"),
                               ("SceneWorks--flux1-dev-mlx", "q4/transformer/diffusion_pytorch_model.safetensors")):
                put(hub, f"models--{repo}/snapshots/{'a' * 40}/{name}")
            put(hub, (snapshot(hub, "pid_decoder") / "pid_flux_2k.safetensors").relative_to(hub))
            result = self.inventory(hub, "cuda-windows", metadata_only=True)
            self.assertEqual(result["roles"]["control"]["status"], "unique")
            self.assertEqual([row["path"] for row in result["roles"]["control"]["candidates"]],
                             [str(control.resolve())])
            self.assertEqual(result["roles"]["pid_decoder"]["status"], "unique")
            self.assertEqual([row["path"] for row in result["roles"]["pid_decoder"]["candidates"]],
                             [str(decoder.resolve())])
            self.assertNotIn("ambiguous_roles", result)

    def test_refs_main_off_pin_and_missing_are_distinct_and_never_ready(self):
        with TemporaryDirectory() as tmp:
            hub = Path(tmp).resolve() / "hub"
            hub.mkdir()
            install(hub, "control", revision=OTHER_REV)
            repo = assets.PRODUCT_SOURCES["control"][0]
            put(hub, (assets.cache_dir(hub, repo) / "refs/main").relative_to(hub),
                (OTHER_REV + "\n").encode())
            result = self.inventory(hub)
            control = result["roles"]["control"]
            self.assertEqual(control["status"], "off_pin")
            self.assertEqual(control["candidates"][0]["revision"], OTHER_REV)
            self.assertEqual(control["candidates"][0]["revision_provenance"], "huggingface-refs-main")
            self.assertIn("control", result["off_pin_roles"])
            self.assertFalse(result["cases"]["control-pose"]["ready"])
            self.assertEqual(result["roles"]["pulid"]["status"], "missing")
            self.assertIn("pulid", result["missing_roles"])
            # A repository without a pin follows refs/main.
            pinless = dict(assets.PRODUCT_SOURCES, control=(repo, "diffusion_pytorch_model.safetensors", None))
            with patch.object(assets, "PRODUCT_SOURCES", pinless):
                self.assertEqual(self.inventory(hub)["roles"]["control"]["status"], "unique")

    def test_probe_timeout_is_unreachable_not_absent(self):
        with TemporaryDirectory() as tmp:
            hub = Path(tmp).resolve() / "hub"
            hub.mkdir()
            with self.assertRaises(assets.Unreachable):
                assets.probe(hub, timeout=0.0001)
            install_all(hub)
            actual = assets.probe
            blocked = assets.cache_dir(hub, assets.PRODUCT_SOURCES["pulid"][0])

            def hanging(path, *args, **kwargs):
                if Path(path).is_relative_to(blocked):
                    raise assets.Unreachable(f"stat {path} did not return within 10 s")
                return actual(path, *args, **kwargs)

            with patch.object(assets, "probe", side_effect=hanging):
                result = self.inventory(hub)
            self.assertEqual(result["roles"]["pulid"]["status"], "unreachable")
            self.assertEqual(result["unreachable_roles"], ["pulid"])
            self.assertNotIn("pulid", result["missing_roles"])
            self.assertFalse(result["cases"]["pulid-fake"]["ready"])
            self.assertTrue(result["cases"]["control-pose"]["ready"])

    def test_incomplete_gemma_and_invalid_index(self):
        with TemporaryDirectory() as tmp:
            hub = Path(tmp).resolve() / "hub"
            hub.mkdir()
            directory = snapshot(hub, "pid_gemma")
            put(hub, (directory / "config.json").relative_to(hub))
            result = self.inventory(hub)
            self.assertEqual(result["roles"]["pid_gemma"]["status"], "incomplete")
            self.assertFalse(result["cases"]["pulid-pid"]["ready"])
            put(hub, (directory / "model.safetensors.index.json").relative_to(hub),
                json.dumps({"weight_map": {"a": "../escape.safetensors"}}).encode())
            with self.assertRaisesRegex(ValueError, "shard index is invalid"):
                self.inventory(hub)

    def test_symlink_escape_and_non_file_fail_fast(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            hub = root / "hub"
            hub.mkdir()
            external = put(root, "external.safetensors")
            path = snapshot(hub, "control") / assets.PRODUCT_SOURCES["control"][1]
            path.parent.mkdir(parents=True)
            path.symlink_to(external)
            with self.assertRaisesRegex(ValueError, "escapes configured model hub"):
                self.inventory(hub)
            path.unlink()
            path.mkdir()
            with self.assertRaisesRegex(ValueError, "not a file"):
                self.inventory(hub)

    def test_metadata_mode_reads_no_weight_bytes_and_budgets_hold(self):
        with TemporaryDirectory() as tmp:
            hub = Path(tmp).resolve() / "hub"
            hub.mkdir()
            install_all(hub)
            with patch.object(assets, "digest", side_effect=AssertionError("metadata read weight bytes")):
                metadata = self.inventory(hub, metadata_only=True)
            self.assertFalse(metadata["hashes_complete"])
            self.assertNotIn("sha256", metadata["roles"]["control"]["candidates"][0])
            self.assertIn("file_bytes", metadata["roles"]["pid_gemma"]["candidates"][0])
            self.assertFalse(metadata["cases"]["control-pose"]["ready"])
            with patch.object(assets, "MAX_HASH_BYTES", 2):
                with self.assertRaisesRegex(ValueError, "budget exceeded"):
                    self.inventory(hub)
            with patch.object(assets, "MAX_METADATA_SECONDS", -1):
                with self.assertRaisesRegex(ValueError, "budget exceeded"):
                    self.inventory(hub, metadata_only=True)

    def test_pinned_path_is_the_exact_hub_snapshot_file(self):
        hub = Path("/hub")
        repo, name, revision = assets.PRODUCT_SOURCES["control"]
        self.assertEqual(assets.pinned_path(hub, repo, revision, name),
                         hub / "models--Shakker-Labs--FLUX.1-dev-ControlNet-Union-Pro-2.0"
                         / "snapshots" / revision / name)
        self.assertEqual(assets.HF_RESOLVE.format(repo=repo, revision=revision, name=name),
                         "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"
                         f"/resolve/{revision}/diffusion_pytorch_model.safetensors")
        for bad in ("../escape", "/abs", "a\\b", ""):
            with self.assertRaisesRegex(ValueError, "unsafe"):
                assets.pinned_path(hub, repo, revision, bad)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            assets.pinned_path(hub, repo, "main", name)

    def test_fetch_streams_atomically_and_never_overwrites(self):
        class Response:
            def __init__(self, body, length=None):
                self.body, self.headers = io.BytesIO(body), {"Content-Length": str(length or len(body))}
            def read(self, size):
                return self.body.read(size)
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return False

        class Opener:
            def __init__(self, body, etag=None, length=None):
                self.body, self.linked_etag, self.length, self.requests = body, etag, length, []
            def open(self, request, timeout):
                self.requests.append(request)
                return Response(self.body, self.length)

        with TemporaryDirectory() as tmp:
            hub = Path(tmp).resolve() / "hub"
            target = hub / "models--a--b/snapshots" / ("a" * 40) / "w.safetensors"
            opener = Opener(b"weights", etag=hashlib.sha256(b"weights").hexdigest())
            size, sha, etag, placed = assets.fetch_pinned("https://example/w", target, token="t", opener=opener)
            self.assertEqual((size, sha), (7, hashlib.sha256(b"weights").hexdigest()))
            self.assertFalse(placed)
            self.assertEqual(etag, sha)
            self.assertEqual(target.read_bytes(), b"weights")
            self.assertFalse(target.is_symlink())
            self.assertEqual(opener.requests[0].unredirected_hdrs["Authorization"], "Bearer t")
            self.assertEqual([p.name for p in target.parent.iterdir()], ["w.safetensors"])
            with self.assertRaises(FileExistsError):
                assets.fetch_pinned("https://example/w", target, opener=Opener(b"other"))
            self.assertEqual(target.read_bytes(), b"weights")
            other = target.with_name("x.safetensors")
            for opener, message in ((Opener(b"bad", etag="0" * 64), "LFS SHA-256"),
                                    (Opener(b"short", length=9), "short download")):
                with self.assertRaisesRegex(ValueError, message):
                    assets.fetch_pinned("https://example/x", other, opener=opener)
                self.assertFalse(other.exists())
            with self.assertRaisesRegex(ValueError, "sealed SHA-256"):
                assets.fetch_pinned("https://example/x", other, opener=Opener(b"bad", etag=hashlib.sha256(b"bad").hexdigest()),
                                    expected_sha256="c" * 64)
            self.assertFalse(other.exists())
            self.assertEqual([p.name for p in target.parent.iterdir()], ["w.safetensors"])

    def test_fetch_accepts_a_concurrently_placed_identical_file_and_refuses_other_bytes(self):
        # Run 44431653018: another job on the same host placed the destination mid-download.
        class Racing:
            """Places `placed` at the destination once this job's download has been fully read."""
            def __init__(self, destination, body, placed):
                self.destination, self.body, self.placed = destination, io.BytesIO(body), placed
                self.headers = {"Content-Length": str(len(body))}
            def read(self, size):
                chunk = self.body.read(size)
                if not chunk and not self.destination.exists():
                    self.destination.write_bytes(self.placed)
                return chunk
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return False

        class Opener:
            def __init__(self, destination, placed):
                self.destination, self.placed, self.linked_etag = destination, placed, None
            def open(self, request, timeout):
                return Racing(self.destination, b"weights", self.placed)

        with TemporaryDirectory() as tmp:
            target = Path(tmp).resolve() / "snapshots" / ("a" * 40) / "w.safetensors"
            target.parent.mkdir(parents=True)
            size, sha, _, placed = assets.fetch_pinned("https://example/w", target, opener=Opener(target, b"weights"))
            self.assertEqual((size, sha), (7, hashlib.sha256(b"weights").hexdigest()))
            self.assertTrue(placed, "an identical file another job placed is reported, not claimed as fetched")
            self.assertEqual(target.read_bytes(), b"weights")
            self.assertEqual([p.name for p in target.parent.iterdir()], ["w.safetensors"])
            other = target.with_name("x.safetensors")
            for placed in (b"WEIGHTS", b"short"):
                other.unlink(missing_ok=True)
                with self.subTest(placed=placed), self.assertRaisesRegex(ValueError, "placed concurrently"):
                    assets.fetch_pinned("https://example/x", other, opener=Opener(other, placed))
                self.assertEqual(other.read_bytes(), placed)
            self.assertEqual(sorted(p.name for p in target.parent.iterdir()), ["w.safetensors", "x.safetensors"])

    def test_stage_records_a_file_another_job_placed_as_kept_not_fetched(self):
        with TemporaryDirectory() as tmp:
            hub = Path(tmp).resolve() / "hub"
            hub.mkdir()
            install_all(hub)
            install(hub, "pulid").unlink()

            def fetch(url, destination, token=None):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"fetched")
                return 7, hashlib.sha256(b"fetched").hexdigest(), None, True

            receipts = assets.stage_pinned_assets(hub, "nax", fetch=fetch)
            self.assertEqual([(r["role"], r["action"]) for r in receipts],
                             [("pulid", "kept-existing-placed-by-another-job")])

    def test_stage_fetches_only_missing_pinned_files_and_records_receipts(self):
        with TemporaryDirectory() as tmp:
            hub = Path(tmp).resolve() / "hub"
            hub.mkdir()
            install_all(hub)
            pulid = install(hub, "pulid")
            pulid.unlink()  # missing role
            control_repo = assets.PRODUCT_SOURCES["control"][0]
            calls = []

            def fetch(url, destination, token=None):
                calls.append((url, destination, token))
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"fetched")
                return 7, hashlib.sha256(b"fetched").hexdigest(), None, False

            receipts = assets.stage_pinned_assets(hub, "nax", token="t", fetch=fetch)
            self.assertEqual([(r["role"], r["action"]) for r in receipts], [("pulid", "fetched")])
            repo, name, revision = assets.PRODUCT_SOURCES["pulid"]
            self.assertEqual(calls, [(f"https://huggingface.co/{repo}/resolve/{revision}/{name}",
                                      pulid, "t")])
            self.assertEqual(receipts[0]["sha256"], hashlib.sha256(b"fetched").hexdigest())
            self.assertEqual(receipts[0]["bytes"], 7)
            self.assertFalse(receipts[0]["hub_lfs_sha256_verified"])
            self.assertEqual(self.inventory(hub)["roles"]["pulid"]["status"], "unique")
            # An off-pin role fetches the pinned revision; refs/main and the off-pin file stay.
            control = install(hub, "control")
            control.unlink()
            off_pin = install(hub, "control", b"old", revision=OTHER_REV)
            refs = put(hub, (assets.cache_dir(hub, control_repo) / "refs/main").relative_to(hub),
                       OTHER_REV.encode())
            calls.clear()
            receipts = assets.stage_pinned_assets(hub, "nax", fetch=fetch)
            self.assertEqual([r["role"] for r in receipts], ["control"])
            self.assertEqual(calls[0][1], control)
            self.assertEqual(off_pin.read_bytes(), b"old")
            self.assertEqual(refs.read_bytes(), OTHER_REV.encode())
            # Nothing missing: nothing fetched.
            calls.clear()
            self.assertEqual(assets.stage_pinned_assets(hub, "nax", fetch=fetch), [])
            self.assertEqual(calls, [])

    def test_stage_gemma_fetches_the_index_then_its_shards_and_keeps_existing_files(self):
        with TemporaryDirectory() as tmp:
            hub = Path(tmp).resolve() / "hub"
            hub.mkdir()
            install_all(hub)
            directory = snapshot(hub, "pid_gemma")
            for name in ("model.safetensors.index.json", "model-00002-of-00002.safetensors"):
                (directory / name).unlink()
            shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")

            def fetch(url, destination, token=None):
                body = (json.dumps({"weight_map": {"a": shards[0], "b": shards[1]}}).encode()
                        if destination.name.endswith("index.json") else b"shard")
                destination.write_bytes(body)
                return len(body), hashlib.sha256(body).hexdigest(), None, False

            receipts = assets.stage_pinned_assets(hub, "nax", fetch=fetch)
            actions = {r["file"]: r["action"] for r in receipts}
            self.assertEqual(actions["model.safetensors.index.json"], "fetched")
            self.assertEqual(actions[shards[1]], "fetched")
            self.assertEqual(actions[shards[0]], "kept-existing")
            self.assertEqual(actions["config.json"], "kept-existing")
            self.assertEqual(self.inventory(hub)["roles"]["pid_gemma"]["status"], "unique")

    def test_exact_runner_cli_and_request_shape(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            hub = root / "hub"
            hub.mkdir()
            install(hub, "control", b"control")
            output = root / "inventory.json"
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=assets.ROOT, text=True,
                                          encoding="utf-8").strip()
            command = [sys.executable, "-I", str(Path(assets.__file__)), "--backend", "mlx",
                       "--request-env", "MODEL_PROOF_REQUEST", "--expected-sha", sha, "--output", str(output)]
            env = dict(os.environ, RUNNER_NAME="nax-macos-2", FLUX1_HUB=str(hub), FLUX1_PAIR_TIER="none",
                       MODEL_PROOF_REQUEST=json.dumps({"selection": "flux1-asset-inventory",
                                                       "runner": "nax"}))
            plan = subprocess.run(command[:-2] + ["--plan"], env=env, capture_output=True, text=True,
                                  encoding="utf-8", check=True)
            self.assertEqual(json.loads(plan.stdout)["selected_runner"], "nax")
            self.assertEqual(json.loads(plan.stdout)["runner"], ["self-hosted", "macOS", "ARM64", "rw-llm"])
            # The pre-route request spelling still selects the same labels, not a machine.
            legacy = assets.plan_request("mlx", {"selection": "flux1-asset-inventory",
                                                 "runner": "nax-macos-2"}, sha, "none")
            self.assertEqual((legacy["selected_runner"], legacy["runner"]),
                             ("nax", ["self-hosted", "macOS", "ARM64", "rw-llm"]))
            metadata_request = {"selection": "flux1-asset-inventory", "runner": "nax",
                                "mode": "metadata"}
            self.assertEqual(assets.plan_request("mlx", metadata_request, sha, "none")["mode"], "metadata")
            with self.assertRaisesRegex(ValueError, "exact selection"):
                assets.plan_request("mlx", {**metadata_request, "roots": ["flux1"]}, sha, "none")
            subprocess.run(command, env=env, check=True, capture_output=True)
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["runner"], "nax")
            self.assertEqual(written["runner_name"], "nax-macos-2")
            self.assertEqual(written["roles"]["control"]["status"], "unique")
            # Either interchangeable Mac may run the macOS route; its name is only recorded.
            env["RUNNER_NAME"] = "nax-macos"
            subprocess.run(command, env=env, check=True, capture_output=True)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["runner_name"], "nax-macos")
            with self.assertRaisesRegex(ValueError, "backend and selected runner differ"):
                assets.plan_request("candle-cuda", {"selection": "flux1-asset-inventory",
                                                     "runner": "nax"}, sha, "none")
            with self.assertRaisesRegex(ValueError, "flux1_pair=none"):
                assets.plan_request("mlx", {"selection": "flux1-asset-inventory",
                                            "runner": "nax"}, sha, "conditional-tier1")
            windows = assets.plan_request("candle-cuda", {"selection": "flux1-asset-inventory",
                                                         "runner": "cuda-windows-2"}, sha, "none")
            self.assertIn("cuda-windows-2", windows["runner"])
            self.assertIn("real-weights", windows["runner"])

    def test_sealed_case_inventory_stages_only_selected_missing_roles_and_refuses_drift(self):
        cases = ["control-pose", "ip-dev", "pulid-fake"]
        sealed = assets.sealed_case_assets(cases, "cuda-windows-2")
        self.assertEqual(set(sealed), {"control", "ip_adapter", "image_encoder", "pulid", "eva",
                                       "arcface", "scrfd", "bisenet"})
        request = {"selection": "flux1-asset-inventory", "runner": "cuda-windows-2", "mode": "hash",
                   "cases": cases}
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=assets.ROOT, text=True,
                                      encoding="utf-8").strip()
        plan = assets.plan_request("candle-cuda", request, sha, "none")
        self.assertEqual(plan["selected_cases"], cases)
        self.assertEqual(plan["runner"], assets.LANES["cuda-windows-2"])
        for invalid in ([], ["control-pose", "control-pose"], ["pulid-pid"]):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "sealed|distinct"):
                assets.plan_request("candle-cuda", dict(request, cases=invalid), sha, "none")
        with self.assertRaisesRegex(ValueError, "exact Windows"):
            assets.plan_request("mlx", dict(request, runner="nax"), sha, "none")

        tiny = {role: dict(pin, sha256=hashlib.sha256(role.encode()).hexdigest())
                for role, pin in sealed.items()}
        real_stage = assets.stage_pinned_assets
        with TemporaryDirectory() as tmp:
            hub = Path(tmp).resolve() / "hub"
            hub.mkdir()
            install(hub, "scrfd")
            output = hub.parent / "receipt.json"
            by_destination = {assets.pinned_path(hub, *assets.stage_files(role, hub)[0]): role for role in tiny}
            fetched = []

            def fetch(url, path, token=None, *, expected_sha256):
                role = by_destination[path]
                fetched.append(role)
                self.assertEqual(expected_sha256, tiny[role]["sha256"])
                body = io.BytesIO(role.encode())
                body.headers = {"Content-Length": str(len(role))}
                return assets.fetch_pinned(url, path, token=token,
                                           opener=SimpleNamespace(open=lambda *_args, **_kwargs: body),
                                           expected_sha256=expected_sha256)

            argv = [str(Path(assets.__file__)), "--backend", "candle-cuda", "--request-env", "INVENTORY_REQUEST",
                    "--expected-sha", sha, "--stage-pinned-assets", "--output", str(output)]
            env = dict(RUNNER_NAME="cuda-windows-2", FLUX1_HUB=str(hub), FLUX1_PAIR_TIER="none",
                       INVENTORY_REQUEST=json.dumps(request))
            with (patch.object(assets, "sealed_case_assets", return_value=tiny),
                  patch.object(assets, "stage_pinned_assets",
                               side_effect=lambda *args, **kwargs: real_stage(*args, fetch=fetch, **kwargs)),
                  patch.object(assets.faulthandler, "dump_traceback_later"),
                  patch.object(assets.faulthandler, "cancel_dump_traceback_later"),
                  patch.object(assets.signal, "signal"),
                  patch.object(sys, "argv", argv), patch.dict(os.environ, env), redirect_stderr(StringIO())):
                assets.main()
                receipt = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(receipt["source_sha"], sha)
                self.assertEqual(receipt["runner_name"], "cuda-windows-2")
                self.assertEqual(set(receipt["roles"]), set(tiny))
                self.assertTrue(all(row["ready"] for row in receipt["cases"].values()))
                self.assertEqual(sorted(fetched), sorted(set(tiny) - {"scrfd"}))
                self.assertFalse(snapshot(hub, "pid_gemma").exists())
                self.assertFalse(snapshot(hub, "pid_decoder").exists())

                installed = install(hub, "control", b"off-seal")
                fetched.clear()
                with self.assertRaises(SystemExit) as exit_info:
                    assets.main()
                self.assertEqual(exit_info.exception.code, 2)
                self.assertEqual(installed.read_bytes(), b"off-seal")
                self.assertEqual(fetched, [])
                receipt = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(receipt["digest_mismatch_roles"], ["control"])
                self.assertFalse(receipt["cases"]["control-pose"]["ready"])
                self.assertTrue(receipt["cases"]["ip-dev"]["ready"])

    def test_inventoried_gemma_stages_by_exact_name_without_listing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            hub = root / "hub"
            hub.mkdir()
            install_gemma(hub)
            row = self.inventory(hub)["roles"]["pid_gemma"]["candidates"][0]
            gemma = {"pid_gemma": {"path": row["path"], "files": row["files"],
                                   "revision": row["revision"]}}
            work = root / "proof"
            work.mkdir()
            guards = no_listing()
            with patch.dict(os.environ, FLUX1_HUB=str(hub)):
                for guard in guards:
                    guard.start()
                try:
                    _, staged = conditional.prepare_assets(work, assets.ROOT, gemma)
                finally:
                    for guard in guards:
                        guard.stop()
            self.assertEqual((Path(staged["pid_gemma"]["staged"]) / "config.json").read_bytes(),
                             b"config.json")
            self.assertEqual(set(staged["pid_gemma"]["files"]), set(row["files"]))
            # A timed-out stat is refused as unreachable, never treated as absent or present.
            def hanging(path, *args, **kwargs):
                raise conditional.Unreachable(f"stat {path} did not return within 10 s")
            for attempt, failure in (("unreachable", hanging), ("missing", None)):
                retry = root / attempt
                retry.mkdir()
                with self.subTest(attempt=attempt), patch.dict(os.environ, FLUX1_HUB=str(hub)):
                    if failure:
                        with patch.object(conditional, "probe", side_effect=failure):
                            with self.assertRaisesRegex(ValueError, "unreachable"):
                                conditional.prepare_assets(retry, assets.ROOT, gemma)
                    else:
                        (Path(row["path"]) / "LICENSE").unlink()
                        with self.assertRaisesRegex(ValueError, "missing: PiD Gemma LICENSE"):
                            conditional.prepare_assets(retry, assets.ROOT, gemma)


if __name__ == "__main__":
    unittest.main()
