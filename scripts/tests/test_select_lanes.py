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
