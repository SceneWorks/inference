"""Guardrail tests for the provider memory-strategy skeleton ratchet (sc-27307).

A provider skeleton (`safety_check`, `registered_*`, `architecture_facts`,
`production_calibration_fingerprint`) counts as bespoke unless its body delegates to a
`ProviderMemoryProfile` const. The ratchet fails on a new bespoke copy and on a stale allowlist
entry, and ignores test code, comments, string literals and body-less declarations.
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


BESPOKE = """
pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || Ok(());
    standard_memory_strategy_safety_check(contract, context, None, Some(&route_gate))
}
"""

DELEGATING = """
const MEMORY_PROFILE: ProviderMemoryProfile = ProviderMemoryProfile::STANDARD;

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    tier: MemoryNumericTier,
) -> MemorySafetyDecision {
    gen_core::standard_safety_check(&MEMORY_PROFILE, contract, context, tier)
}

fn architecture_facts_from_snapshot(_: &str, _: &Path, _: &LoadSpec) -> MemoryArchitectureFacts {
    MemoryArchitectureFacts::default()
}
"""

OUT_OF_SCOPE = """
// fn safety_check(contract: &MemoryProviderContract) -> MemorySafetyDecision { bespoke }
const HELP: &str = "fn registered_begin_request(spec) { bespoke }";

trait Hooks {
    fn safety_check(&self, context: &MemoryRunContext) -> MemorySafetyDecision;
}

#[cfg(test)]
mod tests {
    fn registered_valid_fixture() -> Vec<()> {
        Vec::new()
    }
}
"""


class MemoryProfileRatchetTests(unittest.TestCase):
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

    def errors(self, allowlist: dict[tuple[str, str], str]) -> list[str]:
        original = self.gate.BESPOKE_MEMORY_SKELETONS
        self.gate.BESPOKE_MEMORY_SKELETONS = allowlist
        try:
            return self.gate.memory_profile_ratchet_errors(self.root)
        finally:
            self.gate.BESPOKE_MEMORY_SKELETONS = original

    def test_a_new_bespoke_skeleton_fails(self) -> None:
        self.write("crates/media/mlx-gen/mlx-gen-new/src/memory_strategy.rs", BESPOKE)
        errors = self.errors({})
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("mlx-gen-new::safety_check is a new bespoke", errors[0])
        self.assertIn("memory_strategy.rs:2", errors[0])

    def test_an_allowlisted_bespoke_skeleton_passes(self) -> None:
        self.write("crates/media/mlx-gen/mlx-gen-new/src/memory_strategy.rs", BESPOKE)
        self.assertEqual(self.errors({("mlx-gen-new", "safety_check"): "seal-bound"}), [])

    def test_a_stale_allowlist_entry_fails_so_the_ratchet_tightens(self) -> None:
        self.write("crates/media/candle-gen/candle-gen-new/src/memory_strategy.rs", DELEGATING)
        errors = self.errors({("candle-gen-new", "safety_check"): "seal-bound"})
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("candle-gen-new::safety_check no longer defines a bespoke skeleton", errors[0])

    def test_a_profile_delegation_and_a_named_derivation_hook_are_not_bespoke(self) -> None:
        self.write("crates/media/candle-gen/candle-gen-new/src/memory_strategy.rs", DELEGATING)
        self.assertEqual(self.gate.bespoke_memory_skeletons(self.root), {})

    def test_comments_literals_declarations_and_test_code_are_out_of_scope(self) -> None:
        self.write("crates/media/mlx-gen/mlx-gen-new/src/memory_strategy.rs", OUT_OF_SCOPE)
        self.write(
            "crates/media/mlx-gen/mlx-gen-new/src/memory_strategy/tests.rs",
            BESPOKE,
        )
        self.write("crates/media/mlx-gen/mlx-gen-new/tests/integration.rs", BESPOKE)
        self.assertEqual(self.gate.bespoke_memory_skeletons(self.root), {})

    def test_skeletons_outside_memory_strategy_rs_are_counted(self) -> None:
        self.write("crates/media/mlx-gen/mlx-gen-new/src/block_memory_strategy.rs", BESPOKE)
        self.write("crates/media/candle-gen/candle-gen-other/src/lib.rs", BESPOKE)
        self.assertEqual(
            sorted(self.gate.bespoke_memory_skeletons(self.root)),
            [("candle-gen-other", "safety_check"), ("mlx-gen-new", "safety_check")],
        )

    def test_the_live_tree_matches_the_allowlist(self) -> None:
        self.assertEqual(self.gate.memory_profile_ratchet_errors(ROOT), [])

    def test_every_allowlist_entry_names_its_genuine_difference(self) -> None:
        for (crate, skeleton), reason in self.gate.BESPOKE_MEMORY_SKELETONS.items():
            self.assertIn(skeleton, self.gate.MEMORY_PROFILE_SKELETONS, crate)
            self.assertGreater(len(reason.split()), 4, f"{crate}::{skeleton}")


if __name__ == "__main__":
    unittest.main()
