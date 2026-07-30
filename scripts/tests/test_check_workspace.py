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


class PidDecodeRouteAdoptionTests(unittest.TestCase):
    """sc-15775: a PiD-eligible provider that adopts bounded decode must declare its per-route
    decode domain through the shared `mlx_gen_pid::DecodeRoutes`.

    Native VAE tile edges and PiD tile edges are disjoint by construction, so a provider that emits
    its native ladder into a `use_pid` request is refused at generate time rather than re-planned.
    The shared type makes that declaration unrepresentable — this gate is the tripwire for a provider
    that never reaches for it, so the obligation cannot be silently skipped.
    """

    def setUp(self) -> None:
        self.gate = load_gate_module()

    def metadata(self, *, name: str, depends_on_pid: bool) -> dict:
        member = f"path+file:///repo/crates/media/mlx-gen/{name}#0.0.0"
        return {
            "workspace_members": [member],
            "packages": [
                {
                    "id": member,
                    "name": name,
                    "manifest_path": str(self.crate / "Cargo.toml"),
                    "dependencies": (
                        [{"name": self.gate.PID_SEAM_CRATE}] if depends_on_pid else []
                    ),
                }
            ],
        }

    def write_provider(self, body: str) -> None:
        self.crate = self.root / "crates" / "media" / "mlx-gen" / "provider"
        (self.crate / "src").mkdir(parents=True, exist_ok=True)
        (self.crate / "src" / "lib.rs").write_text(body, encoding="utf-8")

    def run_gate(self, *, name: str = "provider", depends_on_pid: bool = True) -> None:
        self.gate.check_pid_decode_route_adoption(
            self.metadata(name=name, depends_on_pid=depends_on_pid), self.root
        )

    def setUpRoot(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    # A provider that reaches the seam: PiD-eligible, registers a contract, publishes rung-2 edges.
    ADOPTER_WITHOUT_ROUTES = """
        pub fn registry() { register_memory_strategy(MEMORY_REGISTRATION); }
        pub fn ranges() { let _ = decode_tile_edges: vec![768, 640, 512]; }
    """

    def test_an_adopter_that_never_declares_its_routes_fails(self) -> None:
        self.setUpRoot()
        self.write_provider(self.ADOPTER_WITHOUT_ROUTES)
        with self.assertRaises(AssertionError) as caught:
            self.run_gate()
        message = str(caught.exception)
        self.assertIn("provider", message)
        self.assertIn("sc-15775", message)
        self.assertIn("never declares its per-route decode domain", message)

    def test_declaring_the_routes_through_the_shared_type_satisfies_the_gate(self) -> None:
        self.setUpRoot()
        for marker in self.gate.PID_DECODE_ROUTE_MARKERS:
            with self.subTest(marker=marker):
                self.write_provider(
                    self.ADOPTER_WITHOUT_ROUTES + f"\n fn d() -> {marker} {{ todo!() }}\n"
                )
                self.run_gate()

    def test_a_provider_that_cannot_reach_the_seam_is_not_asked_to_declare_routes(self) -> None:
        self.setUpRoot()
        # No PiD dependency: not PiD-eligible, so its native ladder can never reach the PiD seam.
        self.write_provider(self.ADOPTER_WITHOUT_ROUTES)
        self.run_gate(depends_on_pid=False)
        # PiD-eligible but no memory-strategy contract registered: nothing can select rung 2 for it.
        self.write_provider("pub fn ranges() { let _ = decode_tile_edges: vec![512]; }")
        self.run_gate()
        # PiD-eligible and registered, but publishes no rung-2 decode candidates.
        self.write_provider("pub fn registry() { register_memory_strategy(REG); }")
        self.run_gate()

    def test_the_seam_crate_itself_is_exempt(self) -> None:
        self.setUpRoot()
        self.write_provider(self.ADOPTER_WITHOUT_ROUTES)
        self.run_gate(name=self.gate.PID_SEAM_CRATE, depends_on_pid=True)


if __name__ == "__main__":
    unittest.main()
