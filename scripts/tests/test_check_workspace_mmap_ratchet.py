"""Guardrail tests for the candle-gen raw-mmap ratchet (sc-27279).

The gate counts production ``MmapedSafetensors::new/multi`` / ``from_mmaped_safetensors`` sites in
candle-gen outside the audited loader seam and fails when the count rises above the frozen baseline.
Both directions are covered, plus the scoping rules that keep the real tree's count meaningful:
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

    def test_fails_above_baseline_and_passes_at_or_below(self) -> None:
        self.write("crates/media/candle-gen/candle-gen-x/src/lib.rs", PRODUCTION_SITE)
        original = self.gate.CANDLE_GEN_RAW_MMAP_BASELINE
        try:
            self.gate.CANDLE_GEN_RAW_MMAP_BASELINE = 1
            self.gate.check_candle_gen_raw_mmap_ratchet(self.root)
            self.gate.CANDLE_GEN_RAW_MMAP_BASELINE = 0
            with self.assertRaises(AssertionError):
                self.gate.check_candle_gen_raw_mmap_ratchet(self.root)
        finally:
            self.gate.CANDLE_GEN_RAW_MMAP_BASELINE = original

    def test_real_tree_is_within_baseline(self) -> None:
        sites = self.gate.candle_gen_raw_mmap_sites(ROOT)
        self.assertLessEqual(len(sites), self.gate.CANDLE_GEN_RAW_MMAP_BASELINE)


if __name__ == "__main__":
    unittest.main()
