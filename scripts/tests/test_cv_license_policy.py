"""Keep extracted SceneWorks code licensed and outside default bundle graphs."""
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CV = ROOT / "crates/media/gen-cv"


def manifest(path):
    return tomllib.loads(path.read_text(encoding="utf-8"))


def enabled_features(features, roots):
    pending = list(roots)
    enabled = set()
    while pending:
        feature = pending.pop()
        if feature not in enabled:
            enabled.add(feature)
            pending.extend(features.get(feature, []))
    return enabled


class CvLicensePolicyTests(unittest.TestCase):
    def test_extracted_source_retains_agpl_and_attribution(self):
        self.assertEqual(manifest(CV / "Cargo.toml")["package"]["license"], "AGPL-3.0-or-later")
        self.assertIn("GNU AFFERO GENERAL PUBLIC LICENSE", (CV / "LICENSE").read_text(encoding="utf-8"))
        notice = (CV / "NOTICE").read_text(encoding="utf-8")
        self.assertIn("Michael Trefry and the SceneWorks contributors", notice)
        self.assertIn("a1d95e2c43eef4a34a3fe5bf420bb2cc2abe181f", notice)
        self.assertIn("AGPL-3.0-or-later", notice)

    def test_agpl_exception_is_scoped_to_extracted_crate(self):
        policy = manifest(ROOT / "deny.toml")["licenses"]
        self.assertNotIn("AGPL-3.0-or-later", policy["allow"])
        exceptions = [item for item in policy["exceptions"] if "AGPL-3.0-or-later" in item["allow"]]
        self.assertEqual(exceptions, [{"name": "sceneworks-gen-cv", "allow": ["AGPL-3.0-or-later"]}])

    def test_default_media_and_other_facets_do_not_enable_cv(self):
        for name in ("runtime-macos", "runtime-cuda"):
            with self.subTest(bundle=name):
                data = manifest(ROOT / "crates/bundles" / name / "Cargo.toml")
                self.assertTrue(data["dependencies"]["sceneworks-gen-cv"]["optional"])
                features = data["features"]
                for feature in features.keys() - {"cv"}:
                    self.assertNotIn("dep:sceneworks-gen-cv", enabled_features(features, [feature]), feature)
                self.assertIn("dep:sceneworks-gen-cv", enabled_features(features, ["cv"]))

    def test_mac_opt_in_preserves_mlx_engine(self):
        data = manifest(ROOT / "crates/bundles/runtime-macos/Cargo.toml")
        self.assertIn("mlx", data["dependencies"]["sceneworks-gen-cv"]["features"])


if __name__ == "__main__":
    unittest.main()
