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


class RustCommentStrippingTests(unittest.TestCase):
    """The gate matches on comment-stripped source, so the stripper is load-bearing: if it ate a
    string literal the checks would go blind, and if it missed a comment they would be satisfiable by
    one — which is the exact defect sc-15775's review found."""

    def setUp(self) -> None:
        self.gate = load_gate_module()

    def strip(self, source: str) -> str:
        return self.gate.strip_rust_comments(source)

    def test_line_and_block_comments_are_removed(self) -> None:
        self.assertNotIn("DecodeRoutes", self.strip("// use DecodeRoutes one day\n"))
        self.assertNotIn("DecodeRoutes", self.strip("/* DecodeRoutes::new */"))
        self.assertNotIn("DecodeRoutes", self.strip("/// Declares [`DecodeRoutes`].\n"))
        self.assertNotIn("DecodeRoutes", self.strip("//! Module docs for DecodeRoutes.\n"))

    def test_block_comments_nest_as_they_do_in_rust(self) -> None:
        stripped = self.strip("/* outer /* inner DecodeRoutes */ still comment */ real()")
        self.assertNotIn("DecodeRoutes", stripped)
        self.assertIn("real()", stripped)

    def test_code_after_a_comment_survives(self) -> None:
        stripped = self.strip("// nope\nDecodeRoutes::new(id, edges, 64)")
        self.assertIn("DecodeRoutes::new(", stripped)

    def test_comment_markers_inside_string_literals_are_not_comments(self) -> None:
        # A string containing `//` must not blind the rest of the line.
        stripped = self.strip('let s = "http://x // y"; DecodeRoutes::new(a, b, c)')
        self.assertIn("DecodeRoutes::new(", stripped)
        stripped = self.strip('let s = "/* not a comment"; DecodeRoutes::new(a, b, c)')
        self.assertIn("DecodeRoutes::new(", stripped)

    def test_raw_strings_and_escapes_do_not_derail_the_scanner(self) -> None:
        for source in (
            'let s = r"a // b"; DecodeRoutes::new(x, y, z)',
            'let s = r#"a "quoted" // b"#; DecodeRoutes::new(x, y, z)',
            'let s = br#"bytes // b"#; DecodeRoutes::new(x, y, z)',
            r'let s = "esc \" still string // b"; DecodeRoutes::new(x, y, z)',
        ):
            with self.subTest(source=source):
                self.assertIn("DecodeRoutes::new(", self.strip(source))

    def test_lifetimes_are_not_mistaken_for_char_literals(self) -> None:
        source = "fn f<'a>(x: &'a str) -> &'a str { x } // gone\nDecodeRoutes::new(a, b, c)"
        stripped = self.strip(source)
        self.assertIn("DecodeRoutes::new(", stripped)
        self.assertNotIn("gone", stripped)

    def test_char_literals_containing_comment_markers_are_preserved(self) -> None:
        stripped = self.strip("let c = '/'; let d = '\\n'; DecodeRoutes::new(a, b, c)")
        self.assertIn("DecodeRoutes::new(", stripped)

    def test_line_positions_are_preserved(self) -> None:
        source = "a();\n// comment\nb();\n"
        self.assertEqual(len(self.strip(source).splitlines()), len(source.splitlines()))

    def test_optional_literal_blanking_preserves_real_calls_and_line_positions(self) -> None:
        source = (
            'const A: &str = "DecodeRoutes::new(x, y, z)";\n'
            'const B: &str = r#"routes.validate(true, e, o)"#;\n'
            "DecodeRoutes::new(a, b, c)\n"
        )
        stripped = self.gate.strip_rust_comments(source, strip_literals=True)
        self.assertNotIn("routes.validate", stripped)
        self.assertEqual(stripped.count("DecodeRoutes::new"), 1)
        self.assertEqual(len(stripped.splitlines()), len(source.splitlines()))


class PidDecodeRouteAdoptionTests(unittest.TestCase):
    """sc-15775: a PiD-eligible provider that adopts bounded decode must construct its native ladder
    through the checked `mlx_gen_pid::DecodeRoutes::new` and gate admission on the result's `validate`.

    Native VAE tile edges and PiD tile edges are disjoint by construction, so a provider that emits
    its native ladder into a `use_pid` request is refused at generate time rather than re-planned.
    `DecodeRoutes::new` refuses to construct an overlapping declaration at all; this gate is the
    tripwire for a provider that never reaches for it, so the obligation cannot be silently skipped.

    These tests assert the gate demands evidence of a **call**. The first revision matched the bare
    substrings "DecodeRoutes" / "decode_routes" and was defeated three ways by the adversarial review;
    each of those defeats is pinned below as a negative case.
    """

    def setUp(self) -> None:
        self.gate = load_gate_module()
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.crate = self.root / "crates" / "media" / "mlx-gen" / "provider"

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
        (self.crate / "src").mkdir(parents=True, exist_ok=True)
        (self.crate / "src" / "lib.rs").write_text(body, encoding="utf-8")

    def run_gate(self, *, name: str = "provider", depends_on_pid: bool = True) -> None:
        self.gate.check_pid_decode_route_adoption(
            self.metadata(name=name, depends_on_pid=depends_on_pid), self.root
        )

    def assert_gate_fails(self, body: str, *, because: str) -> None:
        self.write_provider(body)
        with self.assertRaises(AssertionError) as caught:
            self.run_gate()
        message = str(caught.exception)
        self.assertIn("provider", message)
        self.assertIn("sc-15775", message)
        self.assertIn(because, message)

    # A provider that reaches the seam: PiD-eligible, registers a contract, implements rung 2.
    ADOPTER_WITHOUT_ROUTES = """
        pub fn registry() { register_memory_strategy(MEMORY_REGISTRATION); }
        pub fn ranges() -> MemoryParameterRanges {
            MemoryParameterRanges { decode_tile_edges: vec![768, 640, 512], ..Default::default() }
        }
    """

    # The declaration half: a real call to the checked constructor.
    CONSTRUCTS = """
        fn decode_routes(id: &str) -> CoreResult<DecodeRoutes> {
            mlx_gen_pid::DecodeRoutes::new(id, EDGES.iter().copied(), OVERLAP).map_err(oops)
        }
    """

    # The admission half: a real call to `validate` on the declared routes.
    ADMITS = """
        fn admit(&self, id: &str, use_pid: bool, e: Option<u32>, o: Option<u32>) -> CoreResult<()> {
            decode_routes(id)?.validate(use_pid, e, o).map_err(CoreError::Unsupported)
        }
    """

    def test_an_adopter_that_never_declares_its_routes_fails(self) -> None:
        self.assert_gate_fails(
            self.ADOPTER_WITHOUT_ROUTES, because="never calls the checked constructor"
        )

    def test_a_full_adopter_satisfies_the_gate(self) -> None:
        """Both halves present, so the gate is satisfiable at all — otherwise every negative below
        would be vacuous."""
        self.write_provider(self.ADOPTER_WITHOUT_ROUTES + self.CONSTRUCTS + self.ADMITS)
        self.run_gate()

    def test_the_panicking_test_suite_spelling_also_counts_as_construction(self) -> None:
        body = (
            self.ADOPTER_WITHOUT_ROUTES
            + "\n fn t() { let routes = mlx_gen_pid::assert_decode_routes(ID, EDGES, 64); }\n"
            + self.ADMITS
        )
        self.write_provider(body)
        self.run_gate()

    def test_construction_without_admission_fails(self) -> None:
        """Declaring the routes and never gating on them leaves the hazard wide open."""
        self.assert_gate_fails(
            self.ADOPTER_WITHOUT_ROUTES + self.CONSTRUCTS,
            because="never calls `validate` on a declared route set",
        )

    def test_admission_without_construction_fails(self) -> None:
        self.assert_gate_fails(
            self.ADOPTER_WITHOUT_ROUTES + self.ADMITS,
            because="never calls the checked constructor",
        )

    def test_a_mention_in_a_comment_does_not_satisfy_the_gate(self) -> None:
        """Review defeat (a). A gate whose purpose is to replace doc-comment enforcement must not be
        satisfiable by a doc comment."""
        for comment in (
            "// TODO(sc-15775): should use DecodeRoutes one day.",
            "/// Eventually declares its routes via [`DecodeRoutes::new`].",
            "/* decode_routes() would go here; also assert_decode_routes and .validate( */",
            "//! This crate will call DecodeRoutes::new and .validate( someday.",
        ):
            with self.subTest(comment=comment):
                self.assert_gate_fails(
                    f"{self.ADOPTER_WITHOUT_ROUTES}\n{comment}\n",
                    because="never calls the checked constructor",
                )

    def test_an_unused_import_does_not_satisfy_the_gate(self) -> None:
        """Review defeat (b)."""
        for import_line in (
            "#[allow(unused)]\nuse mlx_gen_pid::decode_routes;",
            "#[allow(unused_imports)]\nuse mlx_gen_pid::DecodeRoutes;",
            "use mlx_gen_pid::{decode_routes, DecodeRoutes};",
        ):
            with self.subTest(import_line=import_line):
                self.assert_gate_fails(
                    f"{self.ADOPTER_WITHOUT_ROUTES}\n{import_line}\n",
                    because="never calls the checked constructor",
                )

    def test_markers_inside_string_literals_do_not_satisfy_the_gate(self) -> None:
        body = self.ADOPTER_WITHOUT_ROUTES + """
            const FAKE: &str = r#"
                DecodeRoutes::new(id, EDGES, OVERLAP);
                decode_routes(id)?.validate(use_pid, edge, overlap);
            "#;
        """
        self.assert_gate_fails(body, because="never calls the checked constructor")

    def test_markers_confined_to_cfg_test_items_do_not_satisfy_the_gate(self) -> None:
        fixtures = {
            "module": """
                #[cfg(test)]
                mod tests {
                    fn only_in_tests() {
                        let routes = DecodeRoutes::new(ID, EDGES, OVERLAP).unwrap();
                        routes.validate(true, Some(2048), Some(256)).unwrap();
                    }
                }
            """,
            "function": """
                #[cfg(test)]
                /// Test-only evidence; this semicolon must not terminate the cfg item span.
                fn fake_adoption() {
                    let routes = DecodeRoutes::new(ID, EDGES, OVERLAP).unwrap();
                    routes.validate(true, Some(2048), Some(256)).unwrap();
                }
            """,
            "composite all": """
                #[cfg(all(feature = "fixture", test))]
                fn feature_test_adoption() {
                    let routes = DecodeRoutes::new(ID, EDGES, OVERLAP).unwrap();
                    routes.validate(true, Some(2048), Some(256)).unwrap();
                }
            """,
            "const generic in function header": """
                struct Foo<const N: usize>;
                #[cfg(test)]
                fn const_generic_test_adoption() -> Foo<{ 1 }> {
                    let routes = DecodeRoutes::new(ID, EDGES, OVERLAP).unwrap();
                    routes.validate(true, Some(2048), Some(256)).unwrap();
                    loop {}
                }
            """,
            "const initializer branches": """
                #[cfg(test)]
                const FAKE: () = if true {
                } else {
                    let routes = DecodeRoutes::new(ID, EDGES, OVERLAP).unwrap();
                    routes.validate(true, Some(2048), Some(256)).unwrap();
                };
            """,
        }
        for label, fixture in fixtures.items():
            with self.subTest(label=label):
                self.assert_gate_fails(
                    self.ADOPTER_WITHOUT_ROUTES + fixture,
                    because="never calls the checked constructor",
                )

    def test_inner_cfg_test_file_cannot_supply_evidence(self) -> None:
        body = "#![cfg(test)]\n" + self.ADOPTER_WITHOUT_ROUTES + self.CONSTRUCTS + self.ADMITS
        self.assert_gate_fails(body, because="never calls the checked constructor")

    def test_cfg_field_parser_overblanking_cannot_disarm_production_triggers(self) -> None:
        body = """
            struct Fields {
                #[cfg(test)]
                fixture_only: u8,
            }
        """ + self.ADOPTER_WITHOUT_ROUTES
        self.assert_gate_fails(body, because="never calls the checked constructor")

    def test_cfg_any_test_item_is_retained_when_a_production_feature_can_enable_it(self) -> None:
        body = self.ADOPTER_WITHOUT_ROUTES + """
            #[cfg(any(test, feature = "shipping"))]
            fn production_adoption() {
                let routes = DecodeRoutes::new(ID, EDGES, OVERLAP).unwrap();
                routes.validate(true, Some(2048), Some(256)).unwrap();
            }
        """
        self.write_provider(body)
        self.run_gate()

    def test_fake_cfg_attributes_in_comments_and_strings_cannot_disarm_the_gate(self) -> None:
        fixtures = {
            "line comment": "// #[cfg(test)] fn fake() {",
            "block comment": "/* #[cfg(test)] fn fake() { */",
            "string": 'const FAKE_CFG: &str = "#[cfg(test)] fn fake() {";',
            "raw string": 'const RAW_FAKE_CFG: &str = r#"#[cfg(test)] fn fake() {"#;',
        }
        for label, fake_attribute in fixtures.items():
            with self.subTest(label=label):
                self.write_provider(
                    fake_attribute
                    + "\n"
                    + self.ADOPTER_WITHOUT_ROUTES
                    + self.CONSTRUCTS
                    + self.ADMITS
                )
                self.run_gate()

    def test_ranges_built_through_a_shared_helper_still_trigger_the_gate(self) -> None:
        """Review defeat (c): with only `decode_tile_edges` in the trigger set, an adopter whose
        `MemoryParameterRanges` came from a shared helper never armed the gate at all. The executable
        half of rung 2 — its own `configure_decode` hook — is what it cannot delegate."""
        body = """
            pub fn registry() { register_memory_strategy(MEMORY_REGISTRATION); }
            pub fn ranges() -> shared::Ranges { shared::bounded_decode_ranges(&[768, 640], 64) }
            impl MemoryRequestScope for Scope {
                fn configure_decode(&mut self, edge: u32, overlap: u32, _g: MemoryGeometry)
                    -> CoreResult<()> { self.edge = edge; Ok(()) }
            }
        """
        self.assert_gate_fails(body, because="never calls the checked constructor")

    def test_a_provider_that_cannot_reach_the_seam_is_not_asked_to_declare_routes(self) -> None:
        cases = {
            # No PiD dependency: not PiD-eligible, so its native ladder can never reach the PiD seam.
            "not pid-eligible": (self.ADOPTER_WITHOUT_ROUTES, False),
            # PiD-eligible but no contract registered: nothing can select rung 2 for it.
            "no memory-strategy contract": (
                "pub fn ranges() -> MemoryParameterRanges { Default::default() }",
                True,
            ),
            # PiD-eligible and registered, but shows no sign of implementing rung 2 at all.
            "no rung-2 adoption": (
                "pub fn registry() { register_memory_strategy(REG); }",
                True,
            ),
        }
        for label, (body, depends_on_pid) in cases.items():
            with self.subTest(case=label):
                self.write_provider(body)
                self.run_gate(depends_on_pid=depends_on_pid)

    def test_required_configure_decode_hook_that_only_rejects_is_not_rung_two_adoption(self) -> None:
        """A rung-4-only adopter must implement the trait method, but cannot emit a tile."""
        self.write_provider(
            "pub fn registry() { register_memory_strategy(REG); }\n"
            "impl MemoryRequestScope for Scope {\n"
            "  fn configure_decode(&mut self, _e: u32, _o: u32, _g: MemoryGeometry) "
            "-> CoreResult<()> { Err(CoreError::Unsupported(\"not implemented\".into())) }\n"
            "}\n"
        )
        self.run_gate()

    def test_rejection_followed_by_reachable_code_is_still_rung_two_adoption(self) -> None:
        self.assert_gate_fails(
            "pub fn registry() { register_memory_strategy(REG); }\n"
            "impl MemoryRequestScope for Scope {\n"
            "  fn configure_decode(&mut self, _e: u32, _o: u32, _g: MemoryGeometry) "
            "-> CoreResult<()> { Err(CoreError::Unsupported(\"no\".into())); bounded_decode_call() }\n"
            "}\n",
            because="never calls the checked constructor",
        )

    def test_a_commented_out_registration_does_not_arm_the_gate(self) -> None:
        """Comments are stripped before the TRIGGER too, not only before the evidence."""
        self.write_provider(
            "// pub fn registry() { register_memory_strategy(MEMORY_REGISTRATION); }\n"
            "pub fn ranges() -> MemoryParameterRanges { Default::default() }\n"
        )
        self.run_gate()

    def test_the_seam_crate_itself_is_exempt(self) -> None:
        self.write_provider(self.ADOPTER_WITHOUT_ROUTES)
        self.run_gate(name=self.gate.PID_SEAM_CRATE, depends_on_pid=True)

    def test_every_rung_two_trigger_spelling_arms_the_gate(self) -> None:
        """The trigger set is deliberately wide (fail-closed). Pin each spelling, so narrowing it
        back to `decode_tile_edges` alone — which is what review defeat (c) exploited — is a test
        failure rather than a silent regression."""
        for marker in self.gate.PID_RUNG_TWO_MARKERS:
            with self.subTest(marker=marker):
                self.assert_gate_fails(
                    "pub fn registry() { register_memory_strategy(REG); }\n"
                    f"pub fn r() {{ let _ = {marker}; }}\n",
                    because="never calls the checked constructor",
                )


if __name__ == "__main__":
    unittest.main()
