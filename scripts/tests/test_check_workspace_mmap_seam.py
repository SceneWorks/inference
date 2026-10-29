"""Guardrail tests for the candle-gen raw-mmap seam gate (sc-27279).

The gate fails on any production ``MmapedSafetensors::new/multi`` / ``from_mmaped_safetensors`` site
in candle-gen outside the audited loader seam. Both directions are covered, plus the scoping rules:
test code (``#[cfg(test)]`` items and ``tests/`` targets), comments, and the seam itself are out of
scope.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_gate_module():
    spec = importlib.util.spec_from_file_location(
        "check_workspace", ROOT / "scripts" / "check-workspace.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRODUCTION_SITE = """
pub fn open(path: &Path) -> Result<MmapedSafetensors> {
    unsafe { MmapedSafetensors::new(path) }
}
"""

TEST_AND_COMMENT_SITES = """
// A comment mentioning MmapedSafetensors::new(path) is not a site.
pub fn fine() {}

#[cfg(test)]
mod tests {
    #[test]
    fn maps() {
        let _ = unsafe { MmapedSafetensors::multi(&[path]) };
    }
}
"""


class CandleGenRawMmapRatchetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = load_gate_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_counts_production_sites_only(self) -> None:
        self.write("crates/media/candle-gen/candle-gen-x/src/lib.rs", PRODUCTION_SITE)
        self.write("crates/media/candle-gen/candle-gen-x/src/other.rs", TEST_AND_COMMENT_SITES)
        self.write("crates/media/candle-gen/candle-gen-x/tests/t.rs", PRODUCTION_SITE)
        self.write("crates/media/candle-gen/candle-gen/src/loader.rs", PRODUCTION_SITE)
        sites = self.gate.candle_gen_raw_mmap_sites(self.root)
        self.assertEqual(sites, ["crates/media/candle-gen/candle-gen-x/src/lib.rs:3"])

    def test_fails_on_a_production_site_and_passes_without_one(self) -> None:
        self.write("crates/media/candle-gen/candle-gen-x/src/other.rs", TEST_AND_COMMENT_SITES)
        self.write("crates/media/candle-gen/candle-gen/src/loader.rs", PRODUCTION_SITE)
        self.gate.check_candle_gen_raw_mmap_seam(self.root)
        self.write("crates/media/candle-gen/candle-gen-x/src/lib.rs", PRODUCTION_SITE)
        with self.assertRaises(AssertionError):
            self.gate.check_candle_gen_raw_mmap_seam(self.root)

    def test_real_tree_has_no_site_outside_the_seam(self) -> None:
        self.assertEqual(self.gate.candle_gen_raw_mmap_sites(ROOT), [])

if __name__ == "__main__":
    unittest.main()
