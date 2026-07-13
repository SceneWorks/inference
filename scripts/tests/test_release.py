import unittest

from scripts.release.build_release import resolve_lock_dependency, spdx_id, validate_tag


class ReleaseTests(unittest.TestCase):
    def test_accepts_final_and_candidate_tags(self) -> None:
        self.assertIsNotNone(validate_tag("runtime-2026.07.0"))
        self.assertIsNotNone(validate_tag("runtime-2026.07.0-rc.1"))

    def test_rejects_ambiguous_or_invalid_tags(self) -> None:
        for tag in ("v1.0.0", "runtime-2026.13.0", "runtime-26.07.0", "runtime-2026.7.0"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                validate_tag(tag)

    def test_spdx_identity_includes_version_and_source(self) -> None:
        first = spdx_id("same-name", "1.0.0", "registry+one")
        self.assertEqual(first, spdx_id("same-name", "1.0.0", "registry+one"))
        self.assertNotEqual(first, spdx_id("same-name", "2.0.0", "registry+one"))
        self.assertNotEqual(first, spdx_id("same-name", "1.0.0", "registry+two"))

    def test_lock_dependency_disambiguates_duplicate_versions(self) -> None:
        packages = {
            "same": [
                {"name": "same", "version": "1.0.0"},
                {"name": "same", "version": "2.0.0"},
            ]
        }
        self.assertEqual(
            resolve_lock_dependency("same 2.0.0", packages)["version"], "2.0.0"
        )
        with self.assertRaises(RuntimeError):
            resolve_lock_dependency("same", packages)


if __name__ == "__main__":
    unittest.main()
