"""Metadata-only regressions for the dedicated SenseNova evidence protocol."""

import json
import unittest

from scripts.ci.sensenova_loop_seal import TIERS, verify_oracle_marker, verify_semantic_margins


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


if __name__ == "__main__":
    unittest.main()
