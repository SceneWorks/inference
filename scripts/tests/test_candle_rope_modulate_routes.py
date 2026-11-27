"""Self-tests for `scripts/check_candle_rope_modulate_routes.py` (SW-6129).

The gate is a pure source scan, so it is tested the way a scan must be: each case mutates ONE
thing in a synthetic provider crate and asserts the verdict flips, and the real repository is
asserted clean so the committed route table is exercised by ordinary CI. (The candle twin of
`test_mlx_rope_modulate_routes.py`, SW-6140.)
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "check_candle_rope_modulate_routes.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_candle_rope_modulate_routes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gate = _load()

PRIVATE_ROPE = """
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(3)? / 2;
    let rot = Tensor::cat(&[&x.narrow(3, half, half)?.neg()?, &x.narrow(3, 0, half)?], 3)?;
    x.broadcast_mul(cos)? + rot.broadcast_mul(sin)?
}
"""

SHARED_ROPE = """
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // a comment that mentions x.broadcast_mul(cos) must not count
    candle_gen::rope::apply_rope(x, cos, sin, &SPEC)
}
"""

PRIVATE_MODULATE = """
fn modulate(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}
"""

OPERATOR_MODULATE = """
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    Ok(((x * (scale + 1.0)?)? + shift)?)
}
"""

INLINE_MODULATE = """
impl Block {
    fn forward(&self, x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
        let n = layer_norm(x)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(shift)?;
        self.attn.forward(&n)
    }
}
"""

SCALE_ONLY = """
impl Block {
    fn forward(&self, x: &Tensor, scale: &Tensor) -> Result<Tensor> {
        rmsnorm(x)?.broadcast_mul(&(scale + 1.0)?)
    }
}
"""

ROPE_IMPL = """
pub struct RopeTable { cos: Tensor, sin: Tensor }
impl RopeTable {
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&self.cos)?.add(&x.broadcast_mul(&self.sin)?)
    }
}
"""

TEST_ONLY = """
fn helper() {}
#[cfg(test)]
mod tests {
    fn modulate(x: &Tensor, s: &Tensor) -> Tensor { x.broadcast_mul(s).unwrap() }
}
"""

ROW = {"rope": "shared: rope::apply_rope", "modulate": "shared: modulate::modulate"}


class RouteGate(unittest.TestCase):
    def tree(self, files: dict[str, str], crates: dict | None = None, exceptions=None) -> Path:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        base = root / "crates" / "media" / "candle-gen"
        base.mkdir(parents=True)
        for rel, text in files.items():
            path = base / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        table = {
            "version": 1,
            "crates": {"candle-gen-demo": ROW} if crates is None else crates,
            "exceptions": exceptions or [],
        }
        (base / "rope_modulate_routes.json").write_text(json.dumps(table), encoding="utf-8")
        return root

    def test_repository_routes_are_clean(self):
        self.assertEqual(gate.check(), [])

    def test_private_rotation_is_rejected_and_shared_delegation_accepted(self):
        bad = gate.check(self.tree({"candle-gen-demo/src/attention.rs": PRIVATE_ROPE}))
        self.assertEqual(len(bad), 1, bad)
        self.assertIn("attention.rs::apply_rope: private rotation", bad[0])
        self.assertEqual(gate.check(self.tree({"candle-gen-demo/src/attention.rs": SHARED_ROPE})), [])

    def test_method_and_operator_form_modulate_are_rejected(self):
        for body in (PRIVATE_MODULATE, OPERATOR_MODULATE):
            bad = gate.check(self.tree({"candle-gen-demo/src/block.rs": body}))
            self.assertTrue(bad and all("block.rs::" in e for e in bad), bad)
            self.assertTrue(any("block.rs::modulate" in e for e in bad), bad)

    def test_inline_modulate_is_rejected_but_scale_only_is_not(self):
        bad = gate.check(self.tree({"candle-gen-demo/src/block.rs": INLINE_MODULATE}))
        self.assertEqual(len(bad), 1, bad)
        self.assertIn("block.rs::<inline modulate>", bad[0])
        self.assertEqual(gate.check(self.tree({"candle-gen-demo/src/block.rs": SCALE_ONLY})), [])

    def test_apply_on_a_rope_type_is_in_scope(self):
        bad = gate.check(self.tree({"candle-gen-demo/src/nn.rs": ROPE_IMPL}))
        self.assertEqual(len(bad), 1, bad)
        self.assertIn("nn.rs::apply", bad[0])

    def test_test_modules_and_test_files_are_not_scanned(self):
        files = {
            "candle-gen-demo/src/lib.rs": TEST_ONLY,
            "candle-gen-demo/src/block/tests.rs": PRIVATE_MODULATE + INLINE_MODULATE,
        }
        self.assertEqual(gate.check(self.tree(files, crates={})), [])

    def test_exception_must_carry_a_reason_and_must_not_go_stale(self):
        files = {"candle-gen-demo/src/block.rs": PRIVATE_MODULATE}
        row = {"file": "crates/media/candle-gen/candle-gen-demo/src/block.rs", "fn": "modulate",
               "route": "trainer", "reason": "a trainer twin that keeps its composable ops (E9)"}
        inline = {**row, "fn": gate.INLINE_MODULATE_FN}
        self.assertEqual(gate.check(self.tree(files, exceptions=[row, inline])), [])
        terse = gate.check(self.tree(files, exceptions=[{**row, "reason": "why"}, inline]))
        self.assertTrue(any("needs a reason" in e for e in terse), terse)
        bad_route = gate.check(self.tree(files, exceptions=[{**row, "route": "later"}, inline]))
        self.assertTrue(any("route must be one of" in e for e in bad_route), bad_route)
        shared = {"candle-gen-demo/src/block.rs": SHARED_ROPE.replace("apply_rope(x", "modulate(x", 1)}
        stale = gate.check(self.tree(shared, exceptions=[row]))
        self.assertTrue(any("stale exception (no private arithmetic" in e for e in stale), stale)
        gone = gate.check(self.tree({"candle-gen-demo/src/lib.rs": "fn other() {}"}, exceptions=[row]))
        self.assertTrue(any("stale exception (fn not found)" in e for e in gone), gone)

    def test_every_crate_with_rope_functions_needs_a_route_row(self):
        bad = gate.check(self.tree({"candle-gen-demo/src/attention.rs": SHARED_ROPE}, crates={}))
        self.assertEqual(bad, ["candle-gen-demo: has RoPE/modulate functions but no route-table row"])
        ghost = gate.check(self.tree({}, crates={"candle-gen-ghost": ROW}))
        self.assertTrue(any("does not exist: candle-gen-ghost" in e for e in ghost), ghost)

    def test_comment_and_string_arithmetic_is_ignored(self):
        text = 'fn rope_label() -> &\'static str { "x.broadcast_mul(add(x))" }\n'
        self.assertEqual(gate.check(self.tree({"candle-gen-demo/src/lib.rs": text})), [])


if __name__ == "__main__":
    unittest.main()
