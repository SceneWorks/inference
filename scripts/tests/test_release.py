import copy
import unittest

from scripts.release.build_release import (
    resolve_lock_dependency,
    spdx_id,
    validate_model_weight_licenses,
    validate_tag,
)
from scripts.release.verify_release import verify_workspace_manifest


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

    def test_release_workspace_requires_named_runtime_bundles(self) -> None:
        packages = [
            {"name": name}
            for name in ("runtime-catalog", "runtime-macos", "runtime-cuda", "runtime-cpu")
        ]
        verify_workspace_manifest(
            {"workspace": {"package_count": len(packages), "packages": packages}}
        )

        with self.assertRaises(RuntimeError):
            verify_workspace_manifest(
                {"workspace": {"package_count": len(packages) + 1, "packages": packages}}
            )

    def _schema3_manifest(self) -> dict:
        """A minimal but shaped-correctly schema-3 manifest (sc-16663).

        Two providers sharing one component row is the case schema 2 could not express: an artifact
        loaded by more than one provider is ONE row, keyed by artifact rather than by provider.
        """
        return {
            "schema_version": 3,
            "kind": "model-weight-licenses",
            "families": [
                {
                    "id": "apache-2-0",
                    "spdx_id": "Apache-2.0",
                    "name": "Apache License 2.0",
                    "text_url": "https://www.apache.org/licenses/LICENSE-2.0.txt",
                    "terms": [{"term": "attribution_required"}],
                }
            ],
            "components": [
                {
                    "component": "kokoro_82m",
                    "source_url": "https://huggingface.co/hexgrad/Kokoro-82M",
                    "gated": False,
                    "declared": "apache-2.0",
                    "family": "apache-2-0",
                    "attribution": "Kokoro-82M © hexgrad",
                    "retrieved": "2026-08-02",
                }
            ],
            "providers": [
                {
                    "provider_id": "kokoro_82m",
                    "components": ["kokoro_82m"],
                    "terms": [{"term": "attribution_required"}],
                }
            ],
        }

    def test_model_licenses_accepts_a_complete_schema3_manifest(self) -> None:
        providers = validate_model_weight_licenses(self._schema3_manifest())
        self.assertEqual(providers[0]["provider_id"], "kokoro_82m")

    def test_model_licenses_rejects_wrong_kind_or_empty(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_model_weight_licenses({"kind": "something-else", "providers": []})
        for section in ("families", "components", "providers"):
            manifest = self._schema3_manifest()
            manifest[section] = []
            with self.assertRaises(RuntimeError):
                validate_model_weight_licenses(manifest)

    def test_model_licenses_rejects_schema_2(self) -> None:
        """The retired schema must not validate: its rows carry no family and no provenance."""
        manifest = self._schema3_manifest()
        manifest["schema_version"] = 2
        with self.assertRaises(RuntimeError):
            validate_model_weight_licenses(manifest)

    def test_model_licenses_rejects_commercial_use(self) -> None:
        """`commercial_use` stored a legal conclusion and is gone; a row carrying it is rejected
        rather than ignored, so a stale emitter cannot slip one back into a release bundle."""
        for section, key in (("components", "component"), ("providers", "provider_id")):
            manifest = self._schema3_manifest()
            manifest[section][0]["commercial_use"] = False
            with self.assertRaises(RuntimeError) as caught:
                validate_model_weight_licenses(manifest)
            self.assertIn("commercial_use", str(caught.exception))
            self.assertIn(manifest[section][0][key], str(caught.exception))

    def test_model_licenses_rejects_missing_field(self) -> None:
        for section, field in (
            ("families", "text_url"),
            ("components", "source_url"),
            ("components", "declared"),
            ("components", "retrieved"),
        ):
            manifest = self._schema3_manifest()
            del manifest[section][0][field]
            with self.assertRaises(RuntimeError):
                validate_model_weight_licenses(manifest)

    def test_model_licenses_rejects_blank_field(self) -> None:
        manifest = self._schema3_manifest()
        manifest["components"][0]["declared"] = "   "
        with self.assertRaises(RuntimeError):
            validate_model_weight_licenses(manifest)

    def test_model_licenses_rejects_duplicate_keys(self) -> None:
        for section in ("families", "components", "providers"):
            manifest = self._schema3_manifest()
            manifest[section].append(copy.deepcopy(manifest[section][0]))
            with self.assertRaises(RuntimeError):
                validate_model_weight_licenses(manifest)

    def test_model_licenses_accepts_one_component_shared_by_two_providers(self) -> None:
        # `chatterbox_tts` and `chatterbox_ve` load the same artifact. Schema 2 duplicated the row
        # per provider; schema 3 points both at one (sc-16663).
        manifest = self._schema3_manifest()
        manifest["components"][0]["component"] = "chatterbox"
        manifest["providers"] = [
            {
                "provider_id": "chatterbox_tts",
                "components": ["chatterbox"],
                "terms": [{"term": "attribution_required"}],
            },
            {
                "provider_id": "chatterbox_ve",
                "components": ["chatterbox"],
                "terms": [{"term": "attribution_required"}],
            },
        ]
        providers = validate_model_weight_licenses(manifest)
        self.assertEqual(len(providers), 2)

    def test_model_licenses_requires_gated_boolean_and_component_mapping(self) -> None:
        manifest = self._schema3_manifest()
        manifest["components"][0]["gated"] = "auto"
        with self.assertRaises(RuntimeError):
            validate_model_weight_licenses(manifest)

        manifest = self._schema3_manifest()
        manifest["providers"][0]["components"] = []
        with self.assertRaises(RuntimeError):
            validate_model_weight_licenses(manifest)

        manifest = self._schema3_manifest()
        del manifest["providers"][0]["terms"]
        with self.assertRaises(RuntimeError):
            validate_model_weight_licenses(manifest)


if __name__ == "__main__":
    unittest.main()
