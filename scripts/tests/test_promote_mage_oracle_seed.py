from __future__ import annotations

import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "release"
    / "promote_mage_oracle_seed.py"
)


def load_script():
    spec = importlib.util.spec_from_file_location("promote_mage_oracle_seed", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PromoteMageOracleSeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_script()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.seed = self.root / "operator-seed"
        self.source = self.root / "verified-copy"
        self.seed.mkdir()
        for name in self.module.FILES:
            (self.seed / name).write_bytes(f"legacy:{name}".encode())
        shutil.copytree(self.seed, self.source)
        for name in self.module.TARGETS:
            (self.source / name).write_bytes(f"strict:{name}".encode())
        self.receipt = self.root / "receipt.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def old_hashes(self) -> dict[str, str]:
        return {
            name: self.module.sha256(self.seed / name)
            for name in self.module.TARGETS
        }

    def promote(self, *, allow_already_current: bool = False):
        return self.module.promote(
            self.source,
            self.seed,
            self.old_hashes(),
            runner_name="nax-macos-test",
            slot="single",
            revision="a" * 40,
            receipt_path=self.receipt,
            allow_already_current=allow_already_current,
        )

    def test_promotes_only_two_manifests_and_writes_exact_receipt(self) -> None:
        before = {name: (self.seed / name).read_bytes() for name in self.module.FILES}
        receipt = self.promote()

        for name in self.module.UNCHANGED:
            self.assertEqual((self.seed / name).read_bytes(), before[name])
            self.assertEqual((self.seed / name).read_bytes(), (self.source / name).read_bytes())
        for name in self.module.TARGETS:
            self.assertEqual((self.seed / name).read_bytes(), (self.source / name).read_bytes())
        self.assertEqual(receipt, json.loads(self.receipt.read_text(encoding="utf-8")))
        self.assertEqual(receipt["runnerName"], "nax-macos-test")
        self.assertEqual(receipt["slot"], "single")
        self.assertEqual(receipt["revision"], "a" * 40)
        self.assertEqual(receipt["unchangedFileCount"], 18)
        self.assertIn("mage_flow_e2e_golden.png", self.module.UNCHANGED)
        self.assertIn("mage_flow_edit_golden.png", self.module.UNCHANGED)
        self.assertEqual([record["name"] for record in receipt["targets"]], list(self.module.TARGETS))
        transaction, _lock = self.module._managed_paths(self.seed.resolve())
        self.assertFalse(transaction.exists())

    def test_refuses_a_noop_or_changed_seed_target(self) -> None:
        for name in self.module.TARGETS:
            (self.source / name).write_bytes((self.seed / name).read_bytes())
        with self.assertRaisesRegex(self.module.PromotionError, "forbidden no-op"):
            self.promote()

        (self.source / self.module.TARGETS[0]).write_bytes(b"strict-inner")
        (self.source / self.module.TARGETS[1]).write_bytes(b"strict-outer")
        expected = self.old_hashes()
        (self.seed / self.module.TARGETS[1]).write_bytes(b"concurrent-drift")
        with self.assertRaisesRegex(self.module.PromotionError, "changed since import"):
            self.module.promote(
                self.source,
                self.seed,
                expected,
                runner_name="runner",
                slot="secondary",
                revision="b" * 40,
                receipt_path=self.receipt,
            )

    def test_retry_accepts_only_an_exact_already_current_seed(self) -> None:
        imported_legacy_hashes = self.old_hashes()
        for name in self.module.TARGETS:
            (self.seed / name).write_bytes((self.source / name).read_bytes())
        receipt = self.module.promote(
            self.source,
            self.seed,
            imported_legacy_hashes,
            runner_name="nax-macos-test",
            slot="single",
            revision="a" * 40,
            receipt_path=self.receipt,
            allow_already_current=True,
        )
        self.assertEqual(receipt["status"], "already-current")
        self.assertEqual(receipt, json.loads(self.receipt.read_text(encoding="utf-8")))
        for record in receipt["targets"]:
            self.assertEqual(record["old"], record["new"])

        (self.seed / self.module.TARGETS[0]).write_bytes(b"legacy-inner")
        self.receipt.unlink()
        with self.assertRaisesRegex(self.module.PromotionError, "mixed legacy/current"):
            self.promote(allow_already_current=True)

    def test_refuses_non_target_drift_and_root_alias(self) -> None:
        (self.seed / self.module.UNCHANGED[0]).write_bytes(b"drift")
        with self.assertRaisesRegex(self.module.PromotionError, "non-target Mage oracle"):
            self.promote()
        with self.assertRaisesRegex(self.module.PromotionError, "must not alias"):
            self.module.promote(
                self.seed,
                self.seed,
                self.old_hashes(),
                runner_name="runner",
                slot="single",
                revision="c" * 40,
                receipt_path=self.receipt,
            )

    def test_receipt_cannot_overwrite_source_seed_or_managed_state(self) -> None:
        transaction, lock = self.module._managed_paths(self.seed.resolve())
        forbidden = (
            self.seed / self.module.UNCHANGED[0],
            self.source / self.module.UNCHANGED[0],
            transaction / "receipt.json",
            lock,
        )
        before = {name: (self.seed / name).read_bytes() for name in self.module.FILES}
        for receipt in forbidden:
            with self.subTest(receipt=receipt):
                with self.assertRaisesRegex(
                    self.module.PromotionError,
                    "must not already exist|overlaps managed|promotion receipt parent",
                ):
                    self.module.promote(
                        self.source,
                        self.seed,
                        self.old_hashes(),
                        runner_name="runner",
                        slot="single",
                        revision="c" * 40,
                        receipt_path=receipt,
                    )
                self.assertEqual(
                    {name: (self.seed / name).read_bytes() for name in self.module.FILES},
                    before,
                )

    def test_inventory_rejects_missing_extra_symlink_and_hardlink(self) -> None:
        missing = self.module.UNCHANGED[0]
        (self.source / missing).unlink()
        with self.assertRaisesRegex(self.module.PromotionError, "inventory is not exact"):
            self.promote()
        (self.source / missing).write_bytes(f"legacy:{missing}".encode())

        extra = self.source / "unexpected"
        extra.write_bytes(b"x")
        with self.assertRaisesRegex(self.module.PromotionError, "inventory is not exact"):
            self.promote()
        extra.unlink()

        target = self.source / missing
        target.unlink()
        target.symlink_to(self.seed / missing)
        with self.assertRaisesRegex(self.module.PromotionError, "regular, non-symlink"):
            self.promote()
        target.unlink()
        external = self.root / "external"
        external.write_bytes(f"legacy:{missing}".encode())
        os.link(external, target)
        with self.assertRaisesRegex(self.module.PromotionError, "exactly one hard link"):
            self.promote()

    def test_reclaims_only_an_exact_claimed_pre_ready_transaction(self) -> None:
        transaction, _lock = self.module._managed_paths(self.seed.resolve())
        transaction.mkdir()
        (transaction / self.module.CLAIM).write_bytes(
            self.module._json_bytes(self.module._claim_document(self.seed.resolve()))
        )
        partial = transaction / self.module._blob_name(
            "old", self.module.TARGETS[0]
        )
        partial.write_bytes(b"partial")
        original = {name: (self.seed / name).read_bytes() for name in self.module.FILES}

        self.assertTrue(self.module.recover_only(self.seed))
        self.assertFalse(transaction.exists())
        self.assertEqual(
            {name: (self.seed / name).read_bytes() for name in self.module.FILES},
            original,
        )

    def test_reclaims_interrupted_atomic_claim_and_ready_temps(self) -> None:
        transaction, _lock = self.module._managed_paths(self.seed.resolve())
        transaction.mkdir()
        (transaction / f".{self.module.CLAIM}.tmp").write_bytes(b"partial")
        self.assertTrue(self.module.recover_only(self.seed))
        self.assertFalse(transaction.exists())

        transaction.mkdir()
        (transaction / self.module.CLAIM).write_bytes(
            self.module._json_bytes(self.module._claim_document(self.seed.resolve()))
        )
        (transaction / f".{self.module.READY}.tmp").write_bytes(b"partial")
        self.assertTrue(self.module.recover_only(self.seed))
        self.assertFalse(transaction.exists())

    def test_refuses_unclaimed_or_unsafe_transaction(self) -> None:
        transaction, _lock = self.module._managed_paths(self.seed.resolve())
        transaction.mkdir()
        (transaction / self.module._blob_name("old", self.module.TARGETS[0])).write_bytes(
            b"unclaimed"
        )
        with self.assertRaisesRegex(self.module.PromotionError, "unclaimed"):
            self.module.recover_only(self.seed)

        shutil.rmtree(transaction)
        external = self.root / "external-transaction"
        external.mkdir()
        transaction.symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(self.module.PromotionError, "unsafe"):
            self.module.recover_only(self.seed)

    def test_crash_after_inner_replace_recovers_forward(self) -> None:
        original_replace = self.module._replace_target
        calls = 0

        def crash_after_inner(seed, transaction, record):
            nonlocal calls
            calls += 1
            original_replace(seed, transaction, record)
            if calls == 1:
                raise self.module.PromotionError("simulated hard interruption")

        with mock.patch.object(self.module, "_replace_target", crash_after_inner):
            with self.assertRaisesRegex(self.module.PromotionError, "hard interruption"):
                self.promote()

        self.assertEqual(
            (self.seed / self.module.TARGETS[0]).read_bytes(),
            (self.source / self.module.TARGETS[0]).read_bytes(),
        )
        self.assertNotEqual(
            (self.seed / self.module.TARGETS[1]).read_bytes(),
            (self.source / self.module.TARGETS[1]).read_bytes(),
        )
        recovery_receipt = self.root / "recovery-receipt.json"
        self.assertTrue(
            self.module.recover_only(
                self.seed,
                receipt_path=recovery_receipt,
                runner_name="recovery-runner",
                slot="primary",
                revision="d" * 40,
            )
        )
        for name in self.module.TARGETS:
            self.assertEqual((self.seed / name).read_bytes(), (self.source / name).read_bytes())
        receipt = json.loads(recovery_receipt.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "recovered")
        self.assertEqual(
            receipt["recoveredBy"],
            {
                "runnerName": "recovery-runner",
                "slot": "primary",
                "revision": "d" * 40,
            },
        )
        transaction, _lock = self.module._managed_paths(self.seed.resolve())
        self.assertFalse(transaction.exists())

    def test_recovery_refuses_target_drift_outside_old_new_set(self) -> None:
        original_replace = self.module._replace_target

        def stop_before_replace(_seed, _transaction, _record):
            raise self.module.PromotionError("stop")

        with mock.patch.object(self.module, "_replace_target", stop_before_replace):
            with self.assertRaisesRegex(self.module.PromotionError, "stop"):
                self.promote()
        (self.seed / self.module.TARGETS[0]).write_bytes(b"third-party-change")
        with self.assertRaisesRegex(self.module.PromotionError, "outside the transaction"):
            self.module.recover_only(self.seed)
        # The transaction remains for operator inspection rather than masking the drift.
        transaction, _lock = self.module._managed_paths(self.seed.resolve())
        self.assertTrue(transaction.is_dir())

    def test_retry_after_ready_cleanup_crash_certifies_already_current_seed(self) -> None:
        original_remove = self.module._remove_transaction

        def crash_after_ready(transaction):
            (transaction / self.module.READY).unlink()
            self.module._fsync_directory(transaction)
            raise self.module.PromotionError("simulated cleanup interruption")

        with mock.patch.object(self.module, "_remove_transaction", crash_after_ready):
            with self.assertRaisesRegex(self.module.PromotionError, "cleanup interruption"):
                self.promote()
        self.receipt.unlink()

        transaction, _lock = self.module._managed_paths(self.seed.resolve())
        self.assertTrue(transaction.is_dir())
        self.assertTrue(self.module.recover_only(self.seed))
        self.assertFalse(transaction.exists())
        retry = self.promote(allow_already_current=True)
        self.assertEqual(retry["status"], "already-current")
        for name in self.module.TARGETS:
            self.assertEqual((self.seed / name).read_bytes(), (self.source / name).read_bytes())

    def test_receipt_verifier_requires_one_exact_run_certifier(self) -> None:
        receipt_dir = self.root / "receipts"
        receipt_dir.mkdir()
        receipt = self.promote()
        shutil.copy2(
            self.receipt,
            receipt_dir / "mage-seed-promotion-single.json",
        )
        self.module.verify_receipt(receipt_dir, "a" * 40)

        receipt["slot"] = "primary"
        (receipt_dir / "mage-seed-promotion-single.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        with self.assertRaisesRegex(self.module.PromotionError, "exact run"):
            self.module.verify_receipt(receipt_dir, "a" * 40)

    def test_receipt_verifier_accepts_recovery_identity_and_rejects_inventory_drift(self) -> None:
        receipt_dir = self.root / "receipts"
        receipt_dir.mkdir()
        receipt = self.promote()
        receipt["status"] = "recovered"
        receipt["recoveredBy"] = {
            "runnerName": "recovery-runner",
            "slot": "single",
            "revision": "a" * 40,
        }
        (receipt_dir / "mage-seed-promotion-single.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        self.module.verify_receipt(receipt_dir, "a" * 40)

        (receipt_dir / "unexpected.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(self.module.PromotionError, "inventory is not exact"):
            self.module.verify_receipt(receipt_dir, "a" * 40)


if __name__ == "__main__":
    unittest.main()
