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
                # The contract crates compile into the iOS binary too, so a change here must
                # rebuild that target.
                "ios_build",
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

    def test_mlx_change_also_rebuilds_the_ios_target(self) -> None:
        # The mlx crates are the iOS lane's build surface as well as the macOS one: both triples
        # compile the same sources against the same pinned mlx-sys, so an mlx change that only
        # rebuilt macOS could land an iOS break unseen.
        for path in (
            "crates/llm/mlx-llm/src/lib.rs",
            "crates/media/mlx-gen/mlx-gen-flux/src/lib.rs",
        ):
            self.assertTrue(select_lanes([path])["ios_build"], path)

    def test_candle_only_change_skips_the_ios_lane(self) -> None:
        # The iOS bundle is mlx; nothing candle-only reaches it. This is the negative half of the
        # test above -- without it, "ios_build" could quietly become always-on and still pass.
        for path in (
            "crates/llm/candle-llm/src/lib.rs",
            "crates/media/candle-gen/candle-gen-flux/src/lib.rs",
            "crates/audio/candle-audio-kokoro/src/lib.rs",
        ):
            self.assertFalse(select_lanes([path])["ios_build"], path)

    def test_runtime_ios_bundle_stays_on_the_ios_lane(self) -> None:
        # The iOS bundle is LLM-only and MLX-backed: it must not wake the candle or cuda lanes,
        # and it must not wake macos_metal either -- the two bundles share an engine but are
        # separate composition roots.
        lanes = select_lanes(["crates/bundles/runtime-ios/src/lib.rs"])
        self.assertTrue(lanes["ios_build"])
        self.assertTrue(lanes["release"])
        self.assertFalse(lanes["macos_metal"])
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

    def test_empty_or_forced_input_selects_everything(self) -> None:
        self.assertTrue(all(select_lanes([]).values()))
        self.assertTrue(all(select_lanes(["README.md"], force_all=True).values()))


if __name__ == "__main__":
    unittest.main()
