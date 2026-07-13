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

    def test_docs_only_does_not_build_backends(self) -> None:
        lanes = select_lanes(["docs/migration/PHASE_2_CHECKPOINT.md"])
        self.assertTrue(lanes["workspace"])
        self.assertTrue(lanes["docs"])
        self.assertFalse(lanes["macos_metal"])
        self.assertFalse(lanes["candle_cpu"])

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
