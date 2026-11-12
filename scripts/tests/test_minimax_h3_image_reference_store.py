"""The MiniMax-H3-Image reference job uses a runner-local, branch-independent store."""

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.ci import minimax_h3_image_reference_store as store

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "real-weights.yml"
JOB = "minimax-h3-image-reference"
LAUNCHER = "python3.12 -I scripts/ci/inference_metal_launcher.py --wait-seconds 7200 -- /bin/bash -e {0}"
KEY = "h3img-ref-v1-" + "b" * 64


def key(char: str) -> str:
    return "h3img-ref-v1-" + char * 64


class ReferenceJobPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"][JOB]["steps"]

    def test_job_does_not_use_actions_cache(self) -> None:
        # actions/cache is branch-scoped: a new story branch never restores a sibling's entry.
        for step in self.steps:
            self.assertFalse(str(step.get("uses", "")).startswith("actions/cache"), step)

    def test_producer_steps_stay_under_the_metal_launcher(self) -> None:
        producers = [step for step in self.steps if "produce" in step.get("name", "").lower()]
        self.assertEqual(len(producers), 2)
        for step in producers:
            self.assertEqual(step["shell"], LAUNCHER)
            self.assertEqual(step["if"], "steps.store.outputs.hit != 'true'")

    def test_store_restores_before_and_publishes_after_production(self) -> None:
        names = [step.get("name", "") for step in self.steps]
        restore = names.index("Restore the reference from the runner store")
        publish = names.index("Publish the reference to the runner store")
        produced = [i for i, name in enumerate(names) if "produce" in name.lower()]
        self.assertLess(restore, min(produced))
        self.assertLess(max(produced), names.index("Report the reference"))
        self.assertLess(names.index("Report the reference"), publish)
        self.assertTrue(any(str(step.get("uses", "")).startswith("actions/upload-artifact@") for step in self.steps))


class ReferenceStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = self.tmp / "store"
        self.work = self.tmp / "work"
        self.work.mkdir()
        (self.work / "reference.safetensors").write_bytes(b"vae")
        (self.work / "dit-reference.safetensors").write_bytes(b"dit")

    def quiet(self, function, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*args)

    def test_publish_then_restore_round_trips_and_verifies(self) -> None:
        entry = self.quiet(store.publish, self.root, KEY, self.work)
        manifest = json.loads((entry / store.MANIFEST).read_text())
        self.assertEqual(sorted(manifest["files"]), ["dit-reference.safetensors", "reference.safetensors"])
        self.assertEqual([p.name for p in self.root.iterdir()], [KEY])  # no partial left behind
        dest = self.tmp / "dest"
        self.assertTrue(self.quiet(store.restore, self.root, KEY, dest))
        self.assertEqual((dest / "dit-reference.safetensors").read_bytes(), b"dit")
        self.assertFalse((dest / store.MANIFEST).exists())

    def test_missing_entry_is_a_miss_and_clears_dest(self) -> None:
        dest = self.tmp / "dest"
        dest.mkdir()
        (dest / "stale").write_bytes(b"x")
        self.assertFalse(self.quiet(store.restore, self.root, KEY, dest))
        self.assertFalse(dest.exists())

    def test_corrupt_entry_is_a_miss_and_deletes_only_that_key(self) -> None:
        entry = self.quiet(store.publish, self.root, KEY, self.work)
        other = self.quiet(store.publish, self.root, key("c"), self.work)
        (entry / "reference.safetensors").write_bytes(b"tampered")
        dest = self.tmp / "dest"
        self.assertFalse(self.quiet(store.restore, self.root, KEY, dest))
        self.assertFalse(entry.exists())
        self.assertFalse(dest.exists())
        self.assertTrue(other.is_dir())

    def test_extra_or_missing_files_fail_verification(self) -> None:
        entry = self.quiet(store.publish, self.root, KEY, self.work)
        (entry / "unexpected").write_bytes(b"x")
        self.assertFalse(self.quiet(store.restore, self.root, KEY, self.tmp / "dest"))
        self.assertFalse(entry.exists())

    def test_rejects_keys_outside_the_naming_scheme(self) -> None:
        for bad in ("../h3img-ref-v1-" + "a" * 64, "other", "h3img-ref-v1-zz"):
            with self.assertRaises(ValueError):
                store.restore(self.root, bad, self.tmp / "dest")

    def test_prune_keeps_newest_three_and_only_touches_h3img_ref_names(self) -> None:
        self.root.mkdir()
        for index, char in enumerate("abcde"):
            path = self.root / key(char)
            path.mkdir()
            os.utime(path, (1000 + index, 1000 + index))
        unrelated = [self.root / "keep-me", self.root / "h3img-other", self.root / "hub"]
        for path in unrelated:
            path.mkdir()
            os.utime(path, (1, 1))
        (self.root / "loose-file").write_bytes(b"x")
        old_partial = self.root / f"{key('f')}.partial-1-0000abcd"
        fresh_partial = self.root / f"{key('f')}.partial-2-0000abcd"
        old_partial.mkdir()
        fresh_partial.mkdir()
        os.utime(old_partial, (1, 1))
        now = 1 + store.PARTIAL_MAX_AGE_SECONDS + 10
        os.utime(fresh_partial, (now, now))
        removed = self.quiet(store.prune, self.root, 3, now)
        self.assertEqual(sorted(p.name for p in removed), sorted([key("a"), key("b"), old_partial.name]))
        remaining = sorted(p.name for p in self.root.iterdir())
        self.assertEqual(remaining, sorted([key("c"), key("d"), key("e"), fresh_partial.name,
                                            "keep-me", "h3img-other", "hub", "loose-file"]))
        for path in removed:
            self.assertTrue(path.name.startswith("h3img-ref-v"))


if __name__ == "__main__":
    unittest.main()
