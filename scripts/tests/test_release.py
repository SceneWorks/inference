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

    def _permissive_manifest(self) -> dict:
        return {
            "schema_version": 1,
            "kind": "model-weight-licenses",
            "providers": [
                {
                    "provider_id": "kokoro_82m",
                    "spdx_id": "Apache-2.0",
                    "license_name": "Apache License 2.0",
                    "source_url": "https://huggingface.co/hexgrad/Kokoro-82M",
                    "commercial_use": True,
                    "attribution": "Kokoro-82M © hexgrad",
                    "restriction": None,
                }
            ],
        }

    def test_model_licenses_accepts_a_complete_permissive_manifest(self) -> None:
        providers = validate_model_weight_licenses(self._permissive_manifest())
        self.assertEqual(providers[0]["provider_id"], "kokoro_82m")

    def test_model_licenses_rejects_wrong_kind_or_empty(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_model_weight_licenses({"kind": "something-else", "providers": []})
        with self.assertRaises(RuntimeError):
            validate_model_weight_licenses({"kind": "model-weight-licenses", "providers": []})

    def test_model_licenses_rejects_missing_field(self) -> None:
        manifest = self._permissive_manifest()
        del manifest["providers"][0]["source_url"]
        with self.assertRaises(RuntimeError):
            validate_model_weight_licenses(manifest)

    def test_model_licenses_rejects_duplicate_provider_id(self) -> None:
        manifest = self._permissive_manifest()
        manifest["providers"].append(copy.deepcopy(manifest["providers"][0]))
        with self.assertRaises(RuntimeError):
            validate_model_weight_licenses(manifest)

    def test_model_licenses_accepts_multi_component_provider(self) -> None:
        # A multi-checkpoint provider contributes a composite row (component null) plus per-checkpoint
        # attribution rows sharing the provider id, keyed by (provider_id, component) (sc-13493).
        manifest = self._permissive_manifest()
        manifest["providers"] = [
            {
                "provider_id": "mmaudio_small_16k",
                "component": None,
                "spdx_id": "LicenseRef-MMAudio-small-16k-composite",
                "license_name": "MMAudio small_16k composite",
                "source_url": "https://huggingface.co/hkchengrex/MMAudio",
                "commercial_use": False,
                "attribution": "MMAudio assembles five checkpoints",
                "restriction": "Research / non-commercial only.",
            },
            {
                "provider_id": "mmaudio_small_16k",
                "component": "synchformer_vfeat",
                "spdx_id": "MIT",
                "license_name": "MIT License",
                "source_url": "https://github.com/v-iashin/Synchformer",
                "commercial_use": True,
                "attribution": "Synchformer © 2024 Vladimir Iashin — MIT",
                "restriction": None,
            },
        ]
        providers = validate_model_weight_licenses(manifest)
        self.assertEqual(len(providers), 2)

    def test_model_licenses_rejects_duplicate_provider_component_pair(self) -> None:
        manifest = self._permissive_manifest()
        row = copy.deepcopy(manifest["providers"][0])
        row["component"] = "dup"
        manifest["providers"].append(copy.deepcopy(row))
        manifest["providers"].append(copy.deepcopy(row))
        with self.assertRaises(RuntimeError):
            validate_model_weight_licenses(manifest)

    def test_model_licenses_requires_restriction_for_non_commercial(self) -> None:
        manifest = self._permissive_manifest()
        manifest["providers"][0]["commercial_use"] = False
        manifest["providers"][0]["restriction"] = None
        with self.assertRaises(RuntimeError):
            validate_model_weight_licenses(manifest)
        # A non-commercial entry WITH a restriction note is accepted.
        manifest["providers"][0]["spdx_id"] = "CC-BY-NC-4.0"
        manifest["providers"][0]["restriction"] = "Non-commercial use only (CC-BY-NC-4.0)."
        validate_model_weight_licenses(manifest)


if __name__ == "__main__":
    unittest.main()
