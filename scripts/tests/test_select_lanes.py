import unittest

from scripts.ci.select_lanes import LANES, select_lanes


class SelectLanesTests(unittest.TestCase):
    def test_contract_change_fans_out_to_both_backends(self) -> None:
        lanes = select_lanes(["crates/contracts/core-llm/src/lib.rs"])
        for lane in (
            "workspace",
            "contracts",
            "candle_cpu",
            "macos_metal",
            "windows_cuda",
            "real_weights",
        ):
            self.assertTrue(lanes[lane], lane)
        self.assertFalse(lanes["docs"])

    def test_gen_core_testkit_uses_the_contract_lane_set(self) -> None:
        lanes = select_lanes(["crates/contracts/gen-core-testkit/src/lib.rs"])
        selected = {
            lane
            for lane, enabled in lanes.items()
            if enabled
        }
        self.assertEqual(
            selected,
            {
                "workspace",
                "contracts",
                "candle_cpu",
                "macos_metal",
                "windows_cuda",
                "real_weights",
            },
        )

    def test_mlx_provider_change_stays_on_macos(self) -> None:
        lanes = select_lanes(["crates/media/mlx-gen/mlx-gen-flux/src/lib.rs"])
        self.assertTrue(lanes["workspace"])
        self.assertTrue(lanes["macos_metal"])
        self.assertTrue(lanes["real_weights"])
        self.assertFalse(lanes["candle_cpu"])
        self.assertFalse(lanes["windows_cuda"])

    def test_candle_change_includes_cpu_metal_and_cuda(self) -> None:
        lanes = select_lanes(["crates/media/candle-gen/candle-gen-flux/src/lib.rs"])
        self.assertTrue(lanes["candle_cpu"])
        self.assertTrue(lanes["macos_metal"])
        self.assertTrue(lanes["windows_cuda"])
        self.assertFalse(lanes["contracts"])

    def test_audio_family_is_candle_classified(self) -> None:
        # The Candle audio lane (sc-12835) runs on every platform: CPU/CUDA natively and macOS
        # through the mlx bundle's audio section — never fail-safe-to-all as an unknown path.
        for path in (
            "crates/audio/candle-audio/src/lib.rs",
            "crates/audio/candle-audio-catalog/src/lib.rs",
            "crates/audio/candle-audio-stable-audio-3/src/t5gemma.rs",
        ):
            with self.subTest(path=path):
                lanes = select_lanes([path])
                selected = {lane for lane, enabled in lanes.items() if enabled}
                self.assertEqual(
                    selected,
                    {
                        "workspace",
                        "candle_cpu",
                        "macos_metal",
                        "windows_cuda",
                        "real_weights",
                    },
                )

    def test_shared_runtime_catalog_fans_out_to_every_platform(self) -> None:
        lanes = select_lanes(["crates/bundles/runtime-catalog/src/lib.rs"])
        for lane in (
            "candle_cpu",
            "macos_metal",
            "windows_cuda",
            "real_weights",
            "release",
        ):
            self.assertTrue(lanes[lane], lane)

    def test_named_runtime_bundle_selects_only_its_platform(self) -> None:
        cases = {
            "runtime-macos": "macos_metal",
            "runtime-cpu": "candle_cpu",
            "runtime-cuda": "windows_cuda",
        }
        for bundle, expected in cases.items():
            with self.subTest(bundle=bundle):
                lanes = select_lanes([f"crates/bundles/{bundle}/src/lib.rs"])
                self.assertTrue(lanes[expected])
                self.assertTrue(lanes["real_weights"])
                self.assertTrue(lanes["release"])

    def test_docs_only_does_not_build_backends(self) -> None:
        lanes = select_lanes(["docs/migration/PHASE_2_CHECKPOINT.md"])
        self.assertTrue(lanes["workspace"])
        self.assertTrue(lanes["docs"])
        self.assertFalse(lanes["macos_metal"])
        self.assertFalse(lanes["candle_cpu"])

    def test_sa3_reference_fixture_selects_every_consumer_platform(self) -> None:
        lanes = select_lanes(
            ["docs/migration/sa3-sampler-reference/guidance.safetensors"]
        )
        selected = {lane for lane, enabled in lanes.items() if enabled}
        self.assertEqual(
            selected,
            {
                "workspace",
                "docs",
                "candle_cpu",
                "macos_metal",
                "windows_cuda",
            },
        )

    def test_sa3_migration_prose_remains_docs_only(self) -> None:
        lanes = select_lanes(["docs/migration/SC_14540_CHUNKED_SAME.md"])
        selected = {lane for lane, enabled in lanes.items() if enabled}
        self.assertEqual(selected, {"workspace", "docs"})

    def test_sa3_fixture_matching_is_normalized_and_boundary_safe(self) -> None:
        normalized = select_lanes(
            [r".\docs\migration\sa3-reference\manifest.json"]
        )
        normalized_selected = {
            lane for lane, enabled in normalized.items() if enabled
        }
        self.assertEqual(
            normalized_selected,
            {
                "workspace",
                "docs",
                "candle_cpu",
                "macos_metal",
                "windows_cuda",
            },
        )

        for path in (
            "docs/migration/sa3-reference-notes.md",
            "docs/migration/sa3-reference-copy/manifest.json",
            "docs/migration/sa3ish-reference/manifest.json",
            "docs/migration/archive/sa3-reference/manifest.json",
            "docs/migration/sa3-reference/../SC_14534_SA3_REFERENCE_PARITY.md",
        ):
            with self.subTest(path=path):
                selected = {
                    lane
                    for lane, enabled in select_lanes([path]).items()
                    if enabled
                }
                self.assertEqual(selected, {"workspace", "docs"})

    def test_root_doc_and_meta_files_are_docs_only(self) -> None:
        for path in (
            ".github/CODEOWNERS",
            ".gitignore",
            "AGENTS.md",
            "CLAUDE.md",
            "SECURITY.md",
        ):
            with self.subTest(path=path):
                lanes = select_lanes([path])
                selected = {lane for lane, enabled in lanes.items() if enabled}
                self.assertEqual(selected, {"workspace", "docs"})

    def test_root_manifest_and_unknown_paths_fail_safe(self) -> None:
        for path in ("Cargo.toml", "new-build-system/config.json"):
            with self.subTest(path=path):
                lanes = select_lanes([path])
                self.assertTrue(all(lanes[lane] for lane in LANES))

    def test_dependency_policy_files_select_release_without_a_path_gated_audit(self) -> None:
        self.assertNotIn("supply_chain", LANES)
        for path in ("LICENSE", "deny.toml", "advisory-ignores.toml"):
            with self.subTest(path=path):
                selected = {
                    lane
                    for lane, enabled in select_lanes([path]).items()
                    if enabled
                }
                self.assertEqual(selected, {"workspace", "release"})

    def test_empty_or_forced_input_selects_everything(self) -> None:
        self.assertTrue(all(select_lanes([]).values()))
        self.assertTrue(all(select_lanes(["README.md"], force_all=True).values()))


if __name__ == "__main__":
    unittest.main()
