import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.release.build_release import (
    _provider_term_union,
    load_model_weight_licenses,
    merge_model_weight_licenses,
    render_model_licenses,
    resolve_lock_dependency,
    spdx_id,
    validate_model_weight_licenses,
    validate_tag,
)
from scripts.release.verify_release import verify_workspace_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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

    # -- sc-16664: family resolution -------------------------------------------------------------

    def test_model_licenses_rejects_unresolved_family_reference(self) -> None:
        """A component naming a family the document does not carry.

        The reference is the whole route from an artifact to the licence text a consumer shows, so
        an id that resolves to nothing is a row whose terms silently vanish from every union that
        includes it — `provider_terms` contributes nothing for an unresolved family rather than
        failing, by design, so this is the layer that has to catch it.
        """
        manifest = self._schema3_manifest()
        manifest["components"][0]["family"] = "apache-2-0-typo"
        with self.assertRaises(RuntimeError) as caught:
            validate_model_weight_licenses(manifest)
        message = str(caught.exception)
        self.assertIn("apache-2-0-typo", message)
        self.assertIn("kokoro_82m", message)

    def test_model_licenses_requires_attribution_when_the_family_demands_it(self) -> None:
        """`AttributionRequired` on the family implies a non-blank attribution on the component."""
        for attribution in (None, "", "   "):
            manifest = self._schema3_manifest()
            manifest["components"][0]["attribution"] = attribution
            with self.subTest(attribution=attribution):
                with self.assertRaises(RuntimeError) as caught:
                    validate_model_weight_licenses(manifest)
                self.assertIn("attribution", str(caught.exception))

    def test_model_licenses_allows_absent_attribution_without_the_term(self) -> None:
        """A family that does not require attribution leaves the field genuinely optional."""
        manifest = self._schema3_manifest()
        manifest["families"][0]["terms"] = [{"term": "notice_file_required"}]
        manifest["components"][0]["attribution"] = None
        manifest["providers"][0]["terms"] = [{"term": "notice_file_required"}]
        self.assertEqual(len(validate_model_weight_licenses(manifest)), 1)

    def test_model_licenses_rejects_blank_attribution_even_without_the_term(self) -> None:
        """`Some("")` is neither an attribution nor the recorded absence, exactly as in Rust."""
        manifest = self._schema3_manifest()
        manifest["families"][0]["terms"] = [{"term": "notice_file_required"}]
        manifest["providers"][0]["terms"] = [{"term": "notice_file_required"}]
        manifest["components"][0]["attribution"] = "   "
        with self.assertRaises(RuntimeError):
            validate_model_weight_licenses(manifest)

    # -- sc-16664: the derived provider view -----------------------------------------------------

    def test_model_licenses_rejects_a_stale_derived_provider_view(self) -> None:
        """The emitted union is recomputed and compared, so a hand-edited section cannot ship.

        The derived view is the field a consumer actually joins over. Nothing else in the document
        would notice if it stopped agreeing with the components it claims to summarize.
        """
        manifest = self._schema3_manifest()
        manifest["providers"][0]["terms"] = [{"term": "notice_file_required"}]
        with self.assertRaises(RuntimeError) as caught:
            validate_model_weight_licenses(manifest)
        message = str(caught.exception)
        self.assertIn("kokoro_82m", message)
        self.assertIn("attribution_required", message)
        self.assertIn("notice_file_required", message)

    def test_model_licenses_rejects_a_derived_view_missing_gated_access(self) -> None:
        """`gated` lives on the component, not the family, and must still reach the union."""
        manifest = self._schema3_manifest()
        manifest["components"][0]["gated"] = True
        with self.assertRaises(RuntimeError) as caught:
            validate_model_weight_licenses(manifest)
        self.assertIn("gated_access", str(caught.exception))

        manifest["providers"][0]["terms"] = [
            {"term": "attribution_required"},
            {"term": "gated_access"},
        ]
        self.assertEqual(len(validate_model_weight_licenses(manifest)), 1)

    def test_model_licenses_rejects_a_reordered_derived_view(self) -> None:
        """Order is part of the contract: sorted by `tag()`, not by variant declaration order."""
        manifest = self._schema3_manifest()
        manifest["families"][0]["terms"] = [
            {"term": "attribution_required"},
            {"term": "notice_file_required"},
        ]
        manifest["providers"][0]["terms"] = [
            {"term": "notice_file_required"},
            {"term": "attribution_required"},
        ]
        with self.assertRaises(RuntimeError):
            validate_model_weight_licenses(manifest)

    # -- sc-16664: calendar dates ----------------------------------------------------------------

    def test_model_licenses_rejects_impossible_calendar_dates(self) -> None:
        """Month lengths and the Gregorian leap rule, mirroring `license::is_iso_date`.

        `datetime.date.fromisoformat("2026-02-31")` *raises*, so a validator that leaned on it would
        answer a malformed row with a stack trace instead of a message naming the row.
        """
        for retrieved in (
            "2026-02-31",
            "2026-04-31",
            "2026-02-29",
            "1900-02-29",
            "0000-01-01",
            "2026-13-01",
            "2026-01-00",
            "2026-1-01",
            "2026/01/01",
            "20260101",
            "2026-01-01T00:00:00Z",
        ):
            manifest = self._schema3_manifest()
            manifest["components"][0]["retrieved"] = retrieved
            with self.subTest(retrieved=retrieved):
                with self.assertRaises(RuntimeError) as caught:
                    validate_model_weight_licenses(manifest)
                self.assertIn("retrieved", str(caught.exception))

    def test_model_licenses_accepts_real_calendar_dates(self) -> None:
        for retrieved in ("2024-02-29", "2000-02-29", "2026-12-31", "2026-01-01"):
            manifest = self._schema3_manifest()
            manifest["components"][0]["retrieved"] = retrieved
            with self.subTest(retrieved=retrieved):
                self.assertEqual(len(validate_model_weight_licenses(manifest)), 1)

    # -- sc-16664: blank identity fields ---------------------------------------------------------

    def test_model_licenses_rejects_whitespace_only_identity_fields(self) -> None:
        """Whitespace-only is rejected wherever Rust rejects it, so the two agree row for row."""
        cases = (
            ("families", 0, "id"),
            ("families", 0, "spdx_id"),
            ("families", 0, "name"),
            ("families", 0, "text_url"),
            ("components", 0, "component"),
            ("components", 0, "source_url"),
            ("components", 0, "declared"),
            ("components", 0, "family"),
            ("providers", 0, "provider_id"),
        )
        for section, index, field in cases:
            manifest = self._schema3_manifest()
            manifest[section][index][field] = "   "
            with self.subTest(section=section, field=field):
                with self.assertRaises(RuntimeError):
                    validate_model_weight_licenses(manifest)

    def test_model_licenses_rejects_a_blank_component_key_in_a_provider(self) -> None:
        manifest = self._schema3_manifest()
        manifest["providers"][0]["components"] = ["  "]
        with self.assertRaises(RuntimeError):
            validate_model_weight_licenses(manifest)

    # -- sc-16664: the rest of the Rust conformance surface --------------------------------------

    def test_model_licenses_rejects_gated_access_declared_by_a_family(self) -> None:
        """Gating is a per-checkpoint distribution fact; a family declaring it is malformed."""
        manifest = self._schema3_manifest()
        manifest["families"][0]["terms"].append({"term": "gated_access"})
        with self.assertRaises(RuntimeError) as caught:
            validate_model_weight_licenses(manifest)
        self.assertIn("gated_access", str(caught.exception))

    def test_model_licenses_rejects_a_flow_down_term_naming_another_family(self) -> None:
        manifest = self._schema3_manifest()
        manifest["families"][0]["terms"] = [
            {"term": "downstream_license_copy", "family": "some-other-family"}
        ]
        manifest["components"][0]["attribution"] = None
        with self.assertRaises(RuntimeError) as caught:
            validate_model_weight_licenses(manifest)
        self.assertIn("some-other-family", str(caught.exception))

    def test_model_licenses_rejects_a_blank_optional_address(self) -> None:
        """`None` is "the text names no address"; `Some("")` is a malformed row, not a second
        spelling of it."""
        for term, field in (
            ("acceptable_use_policy", "url"),
            ("registration_required", "contact"),
        ):
            manifest = self._schema3_manifest()
            manifest["families"][0]["terms"] = [{"term": term, field: "  "}]
            manifest["components"][0]["attribution"] = None
            manifest["providers"][0]["terms"] = [{"term": term, field: "  "}]
            with self.subTest(term=term):
                with self.assertRaises(RuntimeError):
                    validate_model_weight_licenses(manifest)

    def test_model_licenses_accepts_a_null_optional_address(self) -> None:
        manifest = self._schema3_manifest()
        manifest["families"][0]["terms"] = [{"term": "acceptable_use_policy", "url": None}]
        manifest["components"][0]["attribution"] = None
        manifest["providers"][0]["terms"] = [{"term": "acceptable_use_policy", "url": None}]
        self.assertEqual(len(validate_model_weight_licenses(manifest)), 1)

    def test_model_licenses_rejects_an_unknown_term_tag(self) -> None:
        """A tag Python does not model would sort as a bare variant and could dedup against a
        different term. The Rust `to_json` match is exhaustive for the same reason."""
        manifest = self._schema3_manifest()
        manifest["families"][0]["terms"] = [{"term": "invented_term"}]
        manifest["components"][0]["attribution"] = None
        manifest["providers"][0]["terms"] = [{"term": "invented_term"}]
        with self.assertRaises(RuntimeError) as caught:
            validate_model_weight_licenses(manifest)
        self.assertIn("invented_term", str(caught.exception))

    def test_model_licenses_rejects_unknown_component_and_duplicate_reference(self) -> None:
        manifest = self._schema3_manifest()
        manifest["providers"][0]["components"] = ["kokoro_82m", "not_a_component"]
        with self.assertRaises(RuntimeError) as caught:
            validate_model_weight_licenses(manifest)
        self.assertIn("not_a_component", str(caught.exception))

        manifest = self._schema3_manifest()
        manifest["providers"][0]["components"] = ["kokoro_82m", "kokoro_82m"]
        with self.assertRaises(RuntimeError):
            validate_model_weight_licenses(manifest)

    # -- sc-16664: the real committed manifest ---------------------------------------------------

    def test_committed_manifest_passes_the_full_validator(self) -> None:
        """The audio catalog's own output, emitted by `component_licenses_manifest_json`.

        This is the load-bearing check that the Python recomputation agrees with Rust
        `provider_terms()`: 18 provider unions covering every term shape in the catalog — including
        `gated_access` raised from six component rows, eleven distinct `downstream_license_copy`
        families and three `revenue_ceiling` boundaries — are recomputed here and compared against
        what Rust emitted. A single disagreement in ordering, dedup or payload handling fails it.
        """
        document = json.loads(
            (REPOSITORY_ROOT / "release/model-weight-licenses.json").read_text(encoding="utf-8")
        )
        providers = validate_model_weight_licenses(document)
        self.assertEqual(len(providers), len(document["providers"]))
        self.assertTrue(any(provider["terms"] for provider in providers))

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


class ProviderTermUnionTests(unittest.TestCase):
    """The recomputation must match `license::provider_terms` in the cases the committed table does
    not happen to exercise — otherwise the derived-view check passes today and starts producing
    false failures the moment a media catalog lands a row that does exercise them."""

    def _union(self, families: list, components: list, keys: list) -> list:
        return _provider_term_union(
            {"provider_id": "p", "components": keys, "terms": []},
            {row["component"]: row for row in components},
            {family["id"]: family for family in families},
        )

    def _component(self, key: str, family: str, gated: bool = False) -> dict:
        return {
            "component": key,
            "source_url": "https://example.invalid/card",
            "gated": gated,
            "declared": "example",
            "family": family,
            "attribution": None,
            "retrieved": "2026-08-02",
        }

    def test_ceilings_at_one_amount_with_different_boundaries_stay_distinct(self) -> None:
        """"more than $1,000,000" and "at least $1,000,000" differ at exactly that amount, so they
        are two disclosures. A key that ignored the boundary would silently drop one — and the
        committed audio table happens not to put both readings in one union, so only this case
        catches it."""
        families = [
            {
                "id": "stability",
                "spdx_id": "LicenseRef-A",
                "name": "A",
                "text_url": "https://example.invalid/a",
                "terms": [
                    {"term": "revenue_ceiling", "amount_usd": 1000000, "boundary": "exclusive"}
                ],
            },
            {
                "id": "ltx",
                "spdx_id": "LicenseRef-B",
                "name": "B",
                "text_url": "https://example.invalid/b",
                "terms": [
                    {"term": "revenue_ceiling", "amount_usd": 1000000, "boundary": "inclusive"}
                ],
            },
        ]
        components = [self._component("a", "stability"), self._component("b", "ltx")]
        expected = [
            {"term": "revenue_ceiling", "amount_usd": 1000000, "boundary": "exclusive"},
            {"term": "revenue_ceiling", "amount_usd": 1000000, "boundary": "inclusive"},
        ]
        # Both component orders, because a key that dropped the boundary would still keep both
        # elements (they differ by value, so the dedup spares them) and fail only by *ordering* them
        # however the components happened to be listed. Rust iterates the declared component order
        # while the manifest emits the sorted key list, so an incomplete key is exactly the defect
        # that makes those two disagree.
        for keys in (["a", "b"], ["b", "a"]):
            with self.subTest(keys=keys):
                self.assertEqual(self._union(families, components, keys), expected)

    def test_flow_downs_from_two_families_stay_two_duties(self) -> None:
        """A distributor hands over two documents, not one."""
        families = [
            {
                "id": f"family-{index}",
                "spdx_id": f"LicenseRef-{index}",
                "name": str(index),
                "text_url": "https://example.invalid/x",
                "terms": [{"term": "downstream_license_copy", "family": f"family-{index}"}],
            }
            for index in (1, 2)
        ]
        components = [self._component("a", "family-1"), self._component("b", "family-2")]
        self.assertEqual(len(self._union(families, components, ["a", "b"])), 2)

    def test_identical_terms_from_two_components_collapse(self) -> None:
        families = [
            {
                "id": "apache-2-0",
                "spdx_id": "Apache-2.0",
                "name": "Apache License 2.0",
                "text_url": "https://example.invalid/apache",
                "terms": [{"term": "notice_file_required"}],
            }
        ]
        components = [self._component("a", "apache-2-0"), self._component("b", "apache-2-0")]
        self.assertEqual(
            self._union(families, components, ["a", "b"]), [{"term": "notice_file_required"}]
        )

    def test_gated_access_survives_a_component_whose_family_does_not_resolve(self) -> None:
        """Gating is a property of the checkpoint, so losing it to an unrelated table defect would
        under-report a fact the consumer needs."""
        components = [self._component("a", "missing-family", gated=True)]
        self.assertEqual(self._union([], components, ["a"]), [{"term": "gated_access"}])

    def test_an_unresolved_component_key_contributes_nothing(self) -> None:
        self.assertEqual(self._union([], [], ["absent"]), [])

    def test_order_follows_the_tag_not_the_declaration_order(self) -> None:
        """`gated_access` is declared last in the Rust enum and sorts first by tag; a union ordered
        by variant position would emit the reverse."""
        families = [
            {
                "id": "mixed",
                "spdx_id": "LicenseRef-Mixed",
                "name": "Mixed",
                "text_url": "https://example.invalid/m",
                "terms": [
                    {"term": "non_commercial_weights"},
                    {"term": "attribution_required"},
                    {"term": "acceptable_use_policy", "url": None},
                ],
            }
        ]
        components = [self._component("a", "mixed", gated=True)]
        self.assertEqual(
            [term["term"] for term in self._union(families, components, ["a"])],
            [
                "acceptable_use_policy",
                "attribution_required",
                "gated_access",
                "non_commercial_weights",
            ],
        )

    def test_a_null_address_sorts_before_a_named_one_and_does_not_collapse(self) -> None:
        families = [
            {
                "id": f"family-{index}",
                "spdx_id": f"LicenseRef-{index}",
                "name": str(index),
                "text_url": "https://example.invalid/x",
                "terms": [{"term": "acceptable_use_policy", "url": url}],
            }
            for index, url in ((1, None), (2, "https://example.invalid/policy"))
        ]
        components = [self._component("a", "family-1"), self._component("b", "family-2")]
        self.assertEqual(
            self._union(families, components, ["a", "b"]),
            [
                {"term": "acceptable_use_policy", "url": None},
                {"term": "acceptable_use_policy", "url": "https://example.invalid/policy"},
            ],
        )


class ModelLicenseMergeTests(unittest.TestCase):
    """The bundle aggregates one manifest per catalog — audio, MLX media, Candle media (sc-16664).

    A component row is a property of the *checkpoint*, and MLX and Candle load the same checkpoints,
    so a row appearing in two catalogs must agree. Disagreement is a real defect (one catalog read
    the licence on a different date, or transcribed it differently) and fails the build.
    """

    def _catalog(self, *, family: str = "apache-2-0", provider: str = "kokoro_82m") -> dict:
        return {
            "schema_version": 3,
            "kind": "model-weight-licenses",
            "families": [
                {
                    "id": family,
                    "spdx_id": "Apache-2.0",
                    "name": "Apache License 2.0",
                    "text_url": "https://www.apache.org/licenses/LICENSE-2.0.txt",
                    "terms": [{"term": "notice_file_required"}],
                }
            ],
            "components": [
                {
                    "component": "kokoro_82m",
                    "source_url": "https://huggingface.co/hexgrad/Kokoro-82M",
                    "gated": False,
                    "declared": "apache-2.0",
                    "family": family,
                    "attribution": None,
                    "retrieved": "2026-08-02",
                }
            ],
            "providers": [
                {
                    "provider_id": provider,
                    "components": ["kokoro_82m"],
                    "terms": [{"term": "notice_file_required"}],
                }
            ],
        }

    def test_merge_collapses_rows_two_catalogs_agree_on(self) -> None:
        merged = merge_model_weight_licenses(
            [("release/audio.json", self._catalog()), ("release/mlx.json", self._catalog())]
        )
        self.assertEqual(len(merged["components"]), 1)
        self.assertEqual(len(merged["families"]), 1)
        self.assertEqual(len(merged["providers"]), 1)
        validate_model_weight_licenses(merged)

    def test_merge_unions_disjoint_catalogs_and_sorts_every_section(self) -> None:
        audio = self._catalog()
        mlx = self._catalog()
        mlx["components"][0]["component"] = "aaa_flux_vae"
        mlx["providers"][0] = {
            "provider_id": "aaa_flux",
            "components": ["aaa_flux_vae"],
            "terms": [{"term": "notice_file_required"}],
        }
        merged = merge_model_weight_licenses([("release/z.json", audio), ("release/a.json", mlx)])
        self.assertEqual(
            [row["component"] for row in merged["components"]], ["aaa_flux_vae", "kokoro_82m"]
        )
        self.assertEqual(
            [row["provider_id"] for row in merged["providers"]], ["aaa_flux", "kokoro_82m"]
        )
        validate_model_weight_licenses(merged)

    def test_merge_reports_both_rows_when_a_component_disagrees(self) -> None:
        """The message has to be a diff a human can act on, not just "mismatch": both source rows,
        both source files, and the fields that differ."""
        audio = self._catalog()
        mlx = self._catalog()
        mlx["components"][0]["retrieved"] = "2026-07-01"
        mlx["components"][0]["declared"] = "apache-2.0-only"
        with self.assertRaises(RuntimeError) as caught:
            merge_model_weight_licenses([("release/audio.json", audio), ("release/mlx.json", mlx)])
        message = str(caught.exception)
        self.assertIn("kokoro_82m", message)
        self.assertIn("release/audio.json", message)
        self.assertIn("release/mlx.json", message)
        # Both rows, in full.
        self.assertIn("2026-08-02", message)
        self.assertIn("2026-07-01", message)
        self.assertIn("apache-2.0-only", message)
        # And a pointer at what to look at.
        self.assertIn("declared", message)
        self.assertIn("retrieved", message)

    def test_merge_reports_both_rows_when_a_family_disagrees(self) -> None:
        audio = self._catalog()
        mlx = self._catalog()
        mlx["families"][0]["text_url"] = "https://example.invalid/other-copy"
        with self.assertRaises(RuntimeError) as caught:
            merge_model_weight_licenses([("release/audio.json", audio), ("release/mlx.json", mlx)])
        message = str(caught.exception)
        self.assertIn("apache-2-0", message)
        self.assertIn("https://www.apache.org/licenses/LICENSE-2.0.txt", message)
        self.assertIn("https://example.invalid/other-copy", message)

    def test_merge_reports_a_provider_id_collision_only_visible_after_merging(self) -> None:
        """Two catalogs registering one id against different components. Neither manifest is
        malformed on its own — this is exactly the collision a per-catalog run cannot see."""
        audio = self._catalog()
        mlx = self._catalog()
        mlx["components"][0]["component"] = "kokoro_82m_v2"
        mlx["providers"][0]["components"] = ["kokoro_82m_v2"]
        validate_model_weight_licenses(audio)
        validate_model_weight_licenses(mlx)
        with self.assertRaises(RuntimeError) as caught:
            merge_model_weight_licenses([("release/audio.json", audio), ("release/mlx.json", mlx)])
        message = str(caught.exception)
        self.assertIn("kokoro_82m", message)
        self.assertIn("kokoro_82m_v2", message)

    def test_conformance_runs_over_the_merged_table(self) -> None:
        """A component whose family is carried only by the *other* catalog resolves after the merge.

        Proof the validator sees the merged table rather than each manifest in turn: this document
        pair fails a per-catalog run and passes a merged one.
        """
        audio = self._catalog()
        mlx = self._catalog()
        mlx["families"] = []
        mlx["components"][0]["component"] = "kokoro_82m_second_shard"
        mlx["providers"][0] = {
            "provider_id": "kokoro_82m_shards",
            "components": ["kokoro_82m_second_shard"],
            "terms": [{"term": "notice_file_required"}],
        }
        with self.assertRaises(RuntimeError):
            validate_model_weight_licenses(mlx)
        merged = merge_model_weight_licenses(
            [("release/audio.json", audio), ("release/mlx.json", mlx)]
        )
        self.assertEqual(len(validate_model_weight_licenses(merged)), 2)

    def test_merge_names_the_file_a_broken_source_came_from(self) -> None:
        for broken in ({"kind": "something-else"}, {"kind": "model-weight-licenses"}):
            with self.subTest(broken=broken):
                with self.assertRaises(RuntimeError) as caught:
                    merge_model_weight_licenses(
                        [("release/audio.json", self._catalog()), ("release/broken.json", broken)]
                    )
                self.assertIn("release/broken.json", str(caught.exception))

    def test_merge_requires_at_least_one_source(self) -> None:
        with self.assertRaises(RuntimeError):
            merge_model_weight_licenses([])


class ModelLicenseDiscoveryTests(unittest.TestCase):
    """Only the audio manifest exists today; the MLX and Candle media manifests arrive with
    sc-16665/16666/16667. Their absence must not fail a release now, and their arrival must need no
    edit here — discovery is by shape, so a manifest that lands is merged automatically."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="sc16664-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        (self.root / "release").mkdir()
        self.audio = json.loads(
            (REPOSITORY_ROOT / "release/model-weight-licenses.json").read_text(encoding="utf-8")
        )
        self._write("model-weight-licenses.json", self.audio)

    def _write(self, name: str, document: dict) -> None:
        (self.root / "release" / name).write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def test_absent_media_manifests_do_not_fail_the_build(self) -> None:
        document, sources = load_model_weight_licenses(self.root)
        self.assertEqual(sources, ["release/model-weight-licenses.json"])
        self.assertEqual(document["components"], self.audio["components"])

    def test_a_landing_media_manifest_is_picked_up_with_no_code_edit(self) -> None:
        # A media catalog that reuses the shared component rows and registers its own provider id —
        # the shape sc-16665/16666/16667 land. Its derived union has to be right too.
        media = copy.deepcopy(self.audio)
        borrowed = copy.deepcopy(self.audio["providers"][0])
        borrowed["provider_id"] = "flux1_dev"
        media["providers"] = [borrowed]
        self._write("model-weight-licenses-mlx-media.json", media)
        document, sources = load_model_weight_licenses(self.root)
        self.assertEqual(
            sources,
            [
                "release/model-weight-licenses-mlx-media.json",
                "release/model-weight-licenses.json",
            ],
        )
        self.assertIn("flux1_dev", [row["provider_id"] for row in document["providers"]])
        self.assertEqual(len(document["components"]), len(self.audio["components"]))

    def test_a_missing_audio_manifest_still_fails(self) -> None:
        (self.root / "release" / "model-weight-licenses.json").unlink()
        with self.assertRaises(RuntimeError) as caught:
            load_model_weight_licenses(self.root)
        self.assertIn("release/model-weight-licenses.json", str(caught.exception))

    def test_a_disagreeing_media_manifest_fails_the_build(self) -> None:
        media = copy.deepcopy(self.audio)
        media["components"][0]["retrieved"] = "2020-01-01"
        self._write("model-weight-licenses-candle-media.json", media)
        with self.assertRaises(RuntimeError) as caught:
            load_model_weight_licenses(self.root)
        self.assertIn("2020-01-01", str(caught.exception))

    def test_the_merged_document_renders_as_the_rust_emitter_would(self) -> None:
        """One source in, the committed bytes back out: today's release artifact is unchanged."""
        document, _ = load_model_weight_licenses(self.root)
        self.assertEqual(
            render_model_licenses(document),
            (REPOSITORY_ROOT / "release/model-weight-licenses.json").read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
