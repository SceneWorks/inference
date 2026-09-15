import unittest
from copy import deepcopy
from datetime import date
from pathlib import Path

from scripts.ci.check_advisory_policy import PolicyError, validate_documents, validate_policy


ROOT = Path(__file__).resolve().parents[2]
TODAY = date(2026, 8, 27)
ADVISORY = "RUSTSEC-2024-0436"


def documents() -> tuple[dict, dict]:
    deny = {"advisories": {"ignore": [ADVISORY]}}
    metadata = {
        "version": 1,
        "ignore": [
            {
                "advisory": ADVISORY,
                "reason": "A compatible upstream replacement is not available yet.",
                "reachability": "Compile-time proc-macro only; not linked into the shipped runtime.",
                "owner": "@michaeltrefry",
                "expires": date(2026, 11, 30),
            }
        ],
    }
    return deny, metadata


class AdvisoryPolicyTests(unittest.TestCase):
    def test_committed_policy_is_current_and_exact(self) -> None:
        entries = validate_policy(
            ROOT / "deny.toml",
            ROOT / "advisory-ignores.toml",
            today=TODAY,
        )
        self.assertEqual([entry.advisory for entry in entries], [ADVISORY])

    def test_requires_reason_reachability_owner_and_expiry(self) -> None:
        for missing in ("reason", "reachability", "owner", "expires"):
            with self.subTest(missing=missing):
                deny, metadata = documents()
                del metadata["ignore"][0][missing]
                with self.assertRaisesRegex(PolicyError, f"missing fields: {missing}"):
                    validate_documents(deny, metadata, today=TODAY)

        for field in ("reason", "reachability"):
            with self.subTest(placeholder=field):
                deny, metadata = documents()
                metadata["ignore"][0][field] = "todo"
                with self.assertRaisesRegex(PolicyError, "at least 20 characters"):
                    validate_documents(deny, metadata, today=TODAY)

        deny, metadata = documents()
        metadata["ignore"][0]["owner"] = "someone"
        with self.assertRaisesRegex(PolicyError, "accountable @user"):
            validate_documents(deny, metadata, today=TODAY)

    def test_rejects_expired_or_non_date_expiry(self) -> None:
        for expiry, message in (
            (TODAY, "must be after"),
            (date(2026, 8, 26), "must be after"),
            ("2026-11-30", "unquoted TOML local date"),
        ):
            with self.subTest(expiry=expiry):
                deny, metadata = documents()
                metadata["ignore"][0]["expires"] = expiry
                with self.assertRaisesRegex(PolicyError, message):
                    validate_documents(deny, metadata, today=TODAY)

    def test_deny_and_metadata_sets_must_match_exactly(self) -> None:
        deny, metadata = documents()
        deny["advisories"]["ignore"].append("RUSTSEC-2026-9999")
        with self.assertRaisesRegex(PolicyError, "missing metadata"):
            validate_documents(deny, metadata, today=TODAY)

        deny, metadata = documents()
        deny["advisories"]["ignore"] = []
        with self.assertRaisesRegex(PolicyError, "metadata without a deny.toml ignore"):
            validate_documents(deny, metadata, today=TODAY)

    def test_rejects_duplicates_and_unknown_fields(self) -> None:
        deny, metadata = documents()
        deny["advisories"]["ignore"].append(ADVISORY)
        with self.assertRaisesRegex(PolicyError, "duplicate advisory"):
            validate_documents(deny, metadata, today=TODAY)

        deny, metadata = documents()
        metadata["ignore"].append(deepcopy(metadata["ignore"][0]))
        with self.assertRaisesRegex(PolicyError, "duplicate advisory"):
            validate_documents(deny, metadata, today=TODAY)

        deny, metadata = documents()
        metadata["ignore"][0]["ticket"] = "SC-11295"
        with self.assertRaisesRegex(PolicyError, "unknown fields: ticket"):
            validate_documents(deny, metadata, today=TODAY)


if __name__ == "__main__":
    unittest.main()
