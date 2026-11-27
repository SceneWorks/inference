"""Self-tests for `scripts/check_mlx_rope_modulate_routes.py` (SW-6140).

The gate is a pure source scan, so it is tested the way a scan must be: each case mutates ONE
thing in a synthetic provider crate and asserts the verdict flips, and the real repository is
asserted clean so the committed route table is exercised by ordinary CI.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "check_mlx_rope_modulate_routes.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_mlx_rope_modulate_routes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gate = _load()

PRIVATE_ROPE = """
fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let halves = split(x, 2, 3)?;
    let rot = concatenate_axis(&[&halves[1].negative()?, &halves[0]], 3)?;
    Ok(add(&multiply(x, cos)?, &multiply(&rot, sin)?)?)
}
"""

SHARED_ROPE = """
fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    // a comment that mentions multiply(x, cos) must not count
    mlx_gen::qkv::apply_rope_eager(
        x,
        mlx_gen::qkv::RopeTables::new(cos, sin),
        mlx_gen::qkv::RopeStyle::RotateHalf,
        mlx_gen::qkv::RotationAxes::TokenMajor,
        None,
        mlx_gen::qkv::RopeDtype::Promoted,
    )
}
"""

PRIVATE_MODULATE = """
fn modulate(x: &Array, scale: &Array, shift: &Array) -> Result<Array> {
    Ok(x.multiply(&scale.add(&one)?)?.add(shift)?)
}
"""

ROPE_IMPL = """
pub struct Rope { cos: Array, sin: Array }
impl Rope {
    pub fn apply(&self, x: &Array) -> Result<Array> {
        Ok(add(&multiply(x, &self.cos)?, &multiply(x, &self.sin)?)?)
    }
}
"""

TEST_ONLY = """
fn helper() {}
#[cfg(test)]
mod tests {
    fn modulate(x: &Array, s: &Array) -> Array { multiply(x, s).unwrap() }
}
"""

ROW = {"rope": "shared: qkv::apply_rope", "modulate": "shared: nn::modulate_eager"}


class RouteGate(unittest.TestCase):
    def tree(self, files: dict[str, str], crates: dict | None = None, exceptions=None) -> Path:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        base = root / "crates" / "media" / "mlx-gen"
        base.mkdir(parents=True)
        for rel, text in files.items():
            path = base / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        table = {
            "version": 1,
            "crates": {"mlx-gen-demo": ROW} if crates is None else crates,
            "exceptions": exceptions or [],
        }
        (base / "rope_modulate_routes.json").write_text(json.dumps(table), encoding="utf-8")
        return root

    def test_repository_routes_are_clean(self):
        self.assertEqual(gate.check(), [])

    def test_private_rotation_is_rejected_and_shared_delegation_accepted(self):
        bad = gate.check(self.tree({"mlx-gen-demo/src/attention.rs": PRIVATE_ROPE}))
        self.assertEqual(len(bad), 1, bad)
        self.assertIn("attention.rs::apply_rope: private rotation", bad[0])
        self.assertEqual(gate.check(self.tree({"mlx-gen-demo/src/attention.rs": SHARED_ROPE})), [])

    def test_method_form_modulate_is_rejected(self):
        bad = gate.check(self.tree({"mlx-gen-demo/src/block.rs": PRIVATE_MODULATE}))
        self.assertEqual(len(bad), 1, bad)
        self.assertIn("block.rs::modulate", bad[0])

    def test_apply_on_a_rope_type_is_in_scope(self):
        bad = gate.check(self.tree({"mlx-gen-demo/src/nn.rs": ROPE_IMPL}))
        self.assertEqual(len(bad), 1, bad)
        self.assertIn("nn.rs::apply", bad[0])

    def test_test_modules_and_test_files_are_not_scanned(self):
        files = {
            "mlx-gen-demo/src/lib.rs": TEST_ONLY,
            "mlx-gen-demo/src/block/tests.rs": PRIVATE_MODULATE,
        }
        self.assertEqual(gate.check(self.tree(files, crates={})), [])

    def test_exception_must_carry_a_reason_and_must_not_go_stale(self):
        files = {"mlx-gen-demo/src/block.rs": PRIVATE_MODULATE}
        row = {"file": "crates/media/mlx-gen/mlx-gen-demo/src/block.rs", "fn": "modulate",
               "route": "kept", "reason": "a gate multiply, not a modulation affine"}
        self.assertEqual(gate.check(self.tree(files, exceptions=[row])), [])
        terse = gate.check(self.tree(files, exceptions=[{**row, "reason": "why"}]))
        self.assertTrue(any("needs a reason" in e for e in terse), terse)
        shared = {"mlx-gen-demo/src/block.rs": SHARED_ROPE.replace("apply_rope(x", "modulate(x", 1)}
        stale = gate.check(self.tree(shared, exceptions=[row]))
        self.assertTrue(any("stale exception (no private arithmetic" in e for e in stale), stale)
        gone = gate.check(self.tree({"mlx-gen-demo/src/lib.rs": "fn other() {}"}, exceptions=[row]))
        self.assertTrue(any("stale exception (fn not found)" in e for e in gone), gone)

    def test_every_crate_with_rope_functions_needs_a_route_row(self):
        bad = gate.check(self.tree({"mlx-gen-demo/src/attention.rs": SHARED_ROPE}, crates={}))
        self.assertEqual(bad, ["mlx-gen-demo: has RoPE/modulate functions but no route-table row"])
        ghost = gate.check(self.tree({}, crates={"mlx-gen-ghost": ROW}))
        self.assertTrue(any("does not exist: mlx-gen-ghost" in e for e in ghost), ghost)

    def test_comment_and_string_arithmetic_is_ignored(self):
        text = 'fn rope_label() -> &\'static str { "multiply(add(x))" }\n'
        self.assertEqual(gate.check(self.tree({"mlx-gen-demo/src/lib.rs": text})), [])


if __name__ == "__main__":
    unittest.main()
