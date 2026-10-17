import re
import shutil
import subprocess
import unittest
from pathlib import Path

from scripts.ci.select_lanes import LANES, only_lanes, select_lanes

REPO_ROOT = Path(__file__).resolve().parents[2]


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
        self.assertFalse(lanes["ios_build"])
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

    def test_shared_quant_kernels_are_candle_classified(self) -> None:
        # candle-quant-kernels (sc-24135) serves both candle-gen and candle-llm: every Candle lane,
        # and never the fail-safe-to-all an unclassified top-level path would take.
        lanes = select_lanes(["crates/kernels/candle-quant-kernels/src/nvfp4.rs"])
        selected = {lane for lane, enabled in lanes.items() if enabled}
        self.assertEqual(
            selected,
            {"workspace", "candle_cpu", "macos_metal", "windows_cuda", "real_weights"},
        )

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
            "runtime-ios": "macos_metal",
            "runtime-cpu": "candle_cpu",
            "runtime-cuda": "windows_cuda",
        }
        for bundle, expected in cases.items():
            with self.subTest(bundle=bundle):
                lanes = select_lanes([f"crates/bundles/{bundle}/src/lib.rs"])
                self.assertTrue(lanes[expected])
                self.assertTrue(lanes["real_weights"])
                self.assertTrue(lanes["release"])
                for other in {"macos_metal", "candle_cpu", "windows_cuda"} - {expected}:
                    self.assertFalse(lanes[other], other)

    def test_ios_bound_paths_select_the_ios_build_lane(self) -> None:
        for path in (
            "crates/bundles/runtime-ios/src/lib.rs",
            "crates/media/mlx-gen/mlx-gen-ios-catalog/src/lib.rs",
            "crates/media/mlx-gen/mlx-gen-sana/src/lib.rs",
            "crates/media/mlx-gen/src/lib.rs",
            "crates/media/mlx-gen/Cargo.toml",
            "crates/llm/mlx-llm/src/lib.rs",
            "crates/contracts/gen-core/src/lib.rs",
            "Cargo.lock",
            ".cargo/config.toml",
            ".github/workflows/ci.yml",
            "scripts/ci/assert_metallib_triple.sh",
        ):
            with self.subTest(path=path):
                self.assertTrue(select_lanes([path])["ios_build"])

    def test_non_ios_paths_do_not_select_the_ios_build_lane(self) -> None:
        for path in (
            "crates/media/mlx-gen/mlx-gen-flux/src/lib.rs",
            "crates/media/mlx-gen/mlx-gen-catalog/src/lib.rs",
            "crates/media/mlx-gen/mlx-gen-sanaish/src/lib.rs",
            "crates/bundles/runtime-macos/src/lib.rs",
            "crates/media/candle-gen/candle-gen-flux/src/lib.rs",
            "crates/contracts/gen-core-testkit/src/lib.rs",
            "docs/migration/PHASE_2_CHECKPOINT.md",
        ):
            with self.subTest(path=path):
                self.assertFalse(select_lanes([path])["ios_build"])

    def test_every_crate_on_the_ios_graph_selects_the_ios_build_lane(self) -> None:
        """The static iOS path list must cover runtime-ios's live dependency graph.

        `cargo tree -p runtime-ios` resolves runtime-ios's OWN features (unlike `cargo metadata`,
        which unifies features across the whole workspace), so this is the graph the iOS lane
        actually builds. The path list may over-select; it must not miss a crate.
        """
        if shutil.which("cargo") is None:
            self.skipTest("cargo is not installed")
        result = subprocess.run(
            [
                "cargo",
                "tree",
                "--locked",
                "-p",
                "runtime-ios",
                "--target",
                "aarch64-apple-ios",
                "-e",
                "normal,build",
                "--prefix",
                "none",
                "--format",
                "{p}",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            encoding="utf-8",
        )
        root = REPO_ROOT.resolve()
        crate_dirs = set()
        for match in re.finditer(r"\((/[^)]+)\)", result.stdout):
            path = Path(match.group(1)).resolve()
            if path.is_relative_to(root):
                crate_dirs.add(path.relative_to(root).as_posix())
        self.assertIn("crates/bundles/runtime-ios", crate_dirs)
        self.assertIn("crates/media/mlx-gen/mlx-gen-ios-catalog", crate_dirs)
        for crate_dir in sorted(crate_dirs):
            with self.subTest(crate=crate_dir):
                self.assertTrue(select_lanes([f"{crate_dir}/src/lib.rs"])["ios_build"])

    def test_ios_host_selects_the_harness_lanes_only(self) -> None:
        # The device harness is cross-built and iOS-Clippied on `ios_build`; its weights-free tests
        # and host Clippy run on `macos_metal`. No shipped crate depends on `ios-host/`, so it is
        # classified rather than taking the fail-safe-to-all an unknown top-level path would.
        for path in (
            "ios-host/smoke/src/lib.rs",
            "ios-host/smoke/src/campaign.rs",
            "ios-host/smoke/src/campaign_media.rs",
            "ios-host/smoke/src/campaign_admission_tests.rs",
            "ios-host/smoke/Cargo.lock",
            "ios-host/smoke/examples/campaign_catalog.rs",
            "ios-host/project.yml",
        ):
            with self.subTest(path=path):
                selected = {lane for lane, enabled in select_lanes([path]).items() if enabled}
                self.assertEqual(selected, {"workspace", "ios_build", "macos_metal"})
        # Like every manifest, the harness manifest also selects `release`; nothing else.
        selected = {
            lane
            for lane, enabled in select_lanes(["ios-host/smoke/Cargo.toml"]).items()
            if enabled
        }
        self.assertEqual(selected, {"workspace", "ios_build", "macos_metal", "release"})

    def test_ios_device_scripts_select_the_ios_build_lane(self) -> None:
        for path in ("scripts/ios/run_smoke.sh", "scripts/ios/campaign.py"):
            with self.subTest(path=path):
                lanes = select_lanes([path])
                self.assertTrue(lanes["ios_build"])
                self.assertTrue(lanes["macos_metal"])

    def test_every_crate_on_the_harness_graph_selects_both_harness_lanes(self) -> None:
        """Every crate the device harness compiles must select both lanes that build it.

        The harness (`ios-host/smoke`) is its own cargo workspace with a wider graph than
        runtime-ios (core-llm-testkit, and z-image behind `zimage`), so a crate on it but not on
        runtime-ios's graph would otherwise reach neither the harness cross-build nor its tests.
        """
        if shutil.which("cargo") is None:
            self.skipTest("cargo is not installed")
        result = subprocess.run(
            [
                "cargo",
                "tree",
                "--locked",
                "--manifest-path",
                "ios-host/smoke/Cargo.toml",
                "--features",
                "media,zimage",
                "--target",
                "aarch64-apple-ios",
                "-e",
                "normal,build",
                "--prefix",
                "none",
                "--format",
                "{p}",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            encoding="utf-8",
        )
        root = REPO_ROOT.resolve()
        crate_dirs = set()
        for match in re.finditer(r"\((/[^)]+)\)", result.stdout):
            path = Path(match.group(1)).resolve()
            if path.is_relative_to(root):
                crate_dirs.add(path.relative_to(root).as_posix())
        self.assertIn("ios-host/smoke", crate_dirs)
        self.assertIn("crates/contracts/core-llm/core-llm-testkit", crate_dirs)
        self.assertIn("crates/media/mlx-gen/mlx-gen-z-image", crate_dirs)
        for crate_dir in sorted(crate_dirs):
            with self.subTest(crate=crate_dir):
                lanes = select_lanes([f"{crate_dir}/src/lib.rs"])
                self.assertTrue(lanes["ios_build"])
                self.assertTrue(lanes["macos_metal"])

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

    def test_only_selects_exactly_the_named_lanes(self) -> None:
        lanes = only_lanes(["windows_cuda"])
        self.assertEqual({lane for lane, enabled in lanes.items() if enabled}, {"windows_cuda"})
        self.assertFalse(lanes["macos_metal"])
        self.assertEqual(set(lanes), set(LANES))
        for bad in ([], ["windows-cuda"], ["windows_cuda", "nope"]):
            with self.subTest(selection=bad), self.assertRaises(ValueError):
                only_lanes(bad)


if __name__ == "__main__":
    unittest.main()
