import importlib.util
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


class WithinWorkspaceTests(unittest.TestCase):
    """`check_filesystem` must ignore build output, the git store, and nested tooling worktrees."""

    def setUp(self) -> None:
        self.gate = load_gate_module()
        # Pin ROOT to a fixed checkout so `_within_workspace` is exercised deterministically,
        # independent of where the test happens to run from.
        self.root = Path("/repo")
        self.gate.ROOT = self.root

    def within(self, relative: str) -> bool:
        return self.gate._within_workspace(self.root / relative)

    def test_root_artifacts_belong_to_the_workspace(self) -> None:
        self.assertTrue(self.within("Cargo.lock"))
        self.assertTrue(self.within("crates/llm/mlx-llm/Cargo.toml"))

    def test_build_output_and_git_store_are_ignored(self) -> None:
        self.assertFalse(self.within("target/debug/Cargo.lock"))
        self.assertFalse(self.within(".git/modules/x/Cargo.lock"))

    def test_agent_tooling_worktrees_are_ignored(self) -> None:
        self.assertFalse(self.within(".claude/worktrees/some-session/Cargo.lock"))
        self.assertFalse(self.within(".codex/worktrees/some-session/Cargo.toml"))

    def test_running_from_inside_a_worktree_still_counts_its_own_lockfile(self) -> None:
        # Regression: the filter is on the ROOT-relative path, so a worktree ROOT whose absolute
        # path contains ".claude" does not exclude the worktree's own root artifacts.
        self.gate.ROOT = Path("/repo/.claude/worktrees/pin-bump")
        self.assertTrue(self.gate._within_workspace(self.gate.ROOT / "Cargo.lock"))
        self.assertFalse(
            self.gate._within_workspace(self.gate.ROOT / "target" / "debug" / "Cargo.lock")
        )

    def test_ignored_parts_cover_the_documented_set(self) -> None:
        self.assertEqual(self.gate.IGNORED_TREE_PARTS, {".git", "target", ".claude", ".codex"})


if __name__ == "__main__":
    unittest.main()
