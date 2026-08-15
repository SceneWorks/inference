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

    def write_provider(self, body: str, extra: dict[str, str] | None = None) -> None:
        (self.crate / "src").mkdir(parents=True, exist_ok=True)
        (self.crate / "src" / "lib.rs").write_text(body, encoding="utf-8")
        for name, text in (extra or {}).items():
            (self.crate / "src" / name).write_text(text, encoding="utf-8")

    def run_gate(self, *, name: str = "provider", depends_on_pid: bool = True) -> None:
        self.gate.check_pid_decode_route_adoption(
            self.metadata(name=name, depends_on_pid=depends_on_pid), self.root
        )

    def assert_gate_passes(self, body: str) -> None:
        self.write_provider(body)
        self.run_gate()

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
            "qualified const function": """
                #[cfg(test)]
                const unsafe fn qualified_const_test_adoption() {
                    let routes = DecodeRoutes::new(ID, EDGES, OVERLAP).unwrap();
                    routes.validate(true, Some(2048), Some(256)).unwrap();
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
            "const initializer bare comparison": """
                #[cfg(test)]
                const FAKE: () = if 1 < 2 {
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
        failure rather than a silent regression.

        SC-15525 refined *what the markers are evaluated against*, not the set: a provider that
        publishes a decode DOMAIN still trips on any marker, while one that publishes none cannot
        emit a native tile into the PiD seam and is exempt (see the gate's own comment). Both halves
        are pinned here, so the exemption cannot widen into "any crate that mentions rung 2".
        """
        for marker in self.gate.PID_RUNG_TWO_MARKERS:
            with self.subTest(marker=marker, domain="published"):
                # A published domain is the hazard: every marker must still trip alongside it.
                self.assert_gate_fails(
                    "pub fn registry() { register_memory_strategy(REG); }\n"
                    f"pub fn r() {{ let _ = {marker}; }}\n"
                    "pub fn d() { let _ = MemoryParameterRanges { decode_tile_edges: v, "
                    "decode_overlaps: w }; }\n",
                    because="never calls the checked constructor",
                )

    def test_the_domain_markers_alone_still_trip_the_gate(self) -> None:
        """The two markers that ARE the hazard need no corroboration: publishing a native ladder is
        exactly what reaches `GenerationMemory::decode_tile_edge`."""
        for marker in ("decode_tile_edges", "decode_overlaps"):
            with self.subTest(marker=marker):
                self.assert_gate_fails(
                    "pub fn registry() { register_memory_strategy(REG); }\n"
                    f"pub fn r() {{ let _ = MemoryParameterRanges {{ {marker}: v }}; }}\n",
                    because="never calls the checked constructor",
                )

    # ── SC-15525's rung-2-Missing exemption, and the three shapes it must NOT cover ──────────────
    #
    # The first revision keyed the exemption on the ABSENCE of the two domain literals and pinned only
    # the `fn configure_decode` hook shape. Adversarial review defeated both halves: no MLX provider
    # writes that method (shape C below), and a provider can declare rung 2 *Implemented* while
    # keeping every domain literal in another crate (shape B). Each defeat is a named test here.

    # The real production shape: `BoundedDecode` declared Missing through a named `const`, plus the
    # refusing decode closure every MLX family hands the shared request scope.
    REFUSES_RUNG_TWO = (
        "pub fn registry() { register_memory_strategy(REG); }\n"
        "pub const DECODE_SUPPORT: MemoryStrategySupport = MemoryStrategySupport::Missing;\n"
        "pub fn c() { match s { MemoryStrategy::BoundedDecode => DECODE_SUPPORT,\n"
        "    _ => MemoryParameterRanges::default() }; }\n"
        "fn begin(id: &'static str) -> Result<()> {\n"
        "    let cfg = MlxRequestScopeConfig::new(id, g, m, use_pid, blocks,\n"
        "        move |_use_pid, edge, overlap| Err(refuse_decode(id, Some(edge), Some(overlap))),\n"
        "    )?;\n"
        "    Ok(())\n"
        "}\n"
    )

    def test_a_provider_that_declares_rung_two_missing_is_exempt(self) -> None:
        """SC-15525: declaring `BoundedDecode` **Missing** — and refusing at both decode seams — is
        not adoption. Such a provider publishes no native ladder, so there is nothing for the PiD
        seam to receive. This is the shape `mlx-gen-sdxl` actually ships."""
        self.assert_gate_passes(self.REFUSES_RUNG_TWO)

    def test_the_exemption_needs_the_missing_declaration_not_just_a_bare_mention(self) -> None:
        """The exemption is keyed on the POSITIVE claim. Naming `BoundedDecode` without a support
        expression this reader can resolve to `Missing` proves nothing, so it must not buy an exit —
        absence of a published domain is corroboration, never the key."""
        self.assert_gate_fails(
            "pub fn registry() { register_memory_strategy(REG); }\n"
            "pub fn c() { let _ = MemoryStrategy::BoundedDecode; }\n",
            because="never calls the checked constructor",
        )

    def test_the_exemption_does_not_cover_a_declared_implemented_rung_two(self) -> None:
        """**Review defeat (B).** A provider whose `MemoryParameterRanges` are built by a helper in
        another crate writes neither domain literal in its own source — while declaring rung 2
        **Implemented**. The first revision exempted exactly that. Both the inline and the
        via-`const` spellings of `Implemented` must arm the gate."""
        for support in (
            "MemoryStrategySupport::Implemented",
            "DECODE_SUPPORT",
        ):
            with self.subTest(support=support):
                self.assert_gate_fails(
                    "pub fn registry() { register_memory_strategy(REG); }\n"
                    "pub const DECODE_SUPPORT: MemoryStrategySupport = "
                    "MemoryStrategySupport::Implemented;\n"
                    f"pub fn c() {{ match s {{ MemoryStrategy::BoundedDecode => {support},\n"
                    "    _ => MemoryParameterRanges::default() }; }\n"
                    "pub fn r() -> MemoryParameterRanges { shared::decode_ranges(EDGES) }\n",
                    because="never calls the checked constructor",
                )

    def test_the_exemption_does_not_cover_a_live_mlx_decode_closure(self) -> None:
        """**Review defeat (C) — the serious one.** No MLX provider writes `fn configure_decode`;
        that method lives once on `MlxRequestScopeCore`, and each family hands the constructor a
        closure instead. A closure that can SUCCEED admits a native geometry into the PiD seam, so
        the exemption must not cover it however rung 2 is declared."""
        for body in ("Ok(())", "self.plan(edge, overlap)", "routes.pick(edge)"):
            with self.subTest(body=body):
                self.assert_gate_fails(
                    self.REFUSES_RUNG_TWO.replace(
                        "Err(refuse_decode(id, Some(edge), Some(overlap)))", body
                    ),
                    because="never calls the checked constructor",
                )

    def test_the_exemption_does_not_cover_a_post_construction_validator_swap(self) -> None:
        """**Review defeat (D).** `MlxRequestScopeConfig.decode_validator` is a `pub` field, so a
        rejecting `::new` proves nothing if the next line overwrites it. `mlx-gen-sdxl` already
        mutates the config two lines after `::new`, so this is one line from shipping code."""
        self.assert_gate_fails(
            self.REFUSES_RUNG_TWO.replace(
                "    Ok(())\n",
                "    config.decode_validator = Box::new(|_, _, _| Ok(()));\n    Ok(())\n",
            ),
            because="never calls the checked constructor",
        )

    def test_the_exemption_does_not_cover_a_struct_literal_config(self) -> None:
        """**Review defeat (E) — the vacuous one.** Every field of the config is `pub`, so a struct
        literal never touches `::new`. The first revision then found zero call sites, ran an empty
        loop, and returned `True` — a silent pass, which is exactly what the function's own docstring
        says it never does."""
        self.assert_gate_fails(
            "pub fn registry() { register_memory_strategy(REG); }\n"
            "pub const DECODE_SUPPORT: MemoryStrategySupport = MemoryStrategySupport::Missing;\n"
            "pub fn c() { match s { MemoryStrategy::BoundedDecode => DECODE_SUPPORT,\n"
            "    _ => MemoryParameterRanges::default() }; }\n"
            "fn begin(id: &'static str) -> Result<()> {\n"
            "    let cfg = MlxRequestScopeConfig {\n"
            "        provider_id: id,\n"
            "        decode_validator: Box::new(|_, _, _| Ok(())),\n"
            "        ..Default::default()\n"
            "    };\n"
            "    Ok(())\n"
            "}\n",
            because="never calls the checked constructor",
        )

    def test_naming_the_scope_config_without_a_resolvable_construction_is_not_exempt(self) -> None:
        """The inversion root cause A asked for: a crate that NAMES the scope config but whose
        construction this reader cannot resolve must ARM the gate, not disarm it. The first revision
        returned `True` from an empty loop — a silent pass its own docstring forbade.

        A crate that never names the type at all is a different case and stays exempt: it installs no
        validator, so there is nothing to be wrong about. That is the rung-4-only adopter, pinned by
        `test_required_configure_decode_hook_that_only_rejects_is_not_rung_two_adoption`.
        """
        self.assert_gate_fails(
            "pub fn registry() { register_memory_strategy(REG); }\n"
            "pub const DECODE_SUPPORT: MemoryStrategySupport = MemoryStrategySupport::Missing;\n"
            "pub fn c() { match s { MemoryStrategy::BoundedDecode => DECODE_SUPPORT,\n"
            "    _ => MemoryParameterRanges::default() }; }\n"
            "fn begin(cfg: MlxRequestScopeConfig) -> Result<()> { Ok(build(cfg)) }\n",
            because="never calls the checked constructor",
        )

    # ── SC-15525 probe, root causes A-E. Each shape below compiles and is rustfmt-stable. ────────

    def test_the_exemption_does_not_cover_an_aliased_constructor(self) -> None:
        """**Root cause A.** A literal-substring reader is one rename away from vacuous: the
        constructor is resolved by TYPE, so `use … as X`, `type X = …` and the qualified
        `<path::Type>::new` form all have to land on the same check."""
        for preamble, call in (
            (
                "use mlx_gen::request_scope::MlxRequestScopeConfig as ScopeConfig;\n",
                "ScopeConfig::new(id, g, m, p, b, move |_u, e, o| self.plan(e, o))",
            ),
            (
                "type Scope = MlxRequestScopeConfig;\n",
                "Scope::new(id, g, m, p, b, move |_u, e, o| self.plan(e, o))",
            ),
            (
                "",
                "<mlx_gen::request_scope::MlxRequestScopeConfig>::new(id, g, m, p, b, "
                "move |_u, e, o| self.plan(e, o))",
            ),
        ):
            with self.subTest(call=call[:40]):
                self.assert_gate_fails(
                    preamble
                    + "pub fn registry() { register_memory_strategy(REG); }\n"
                    "pub const DECODE_SUPPORT: MemoryStrategySupport = MemoryStrategySupport::Missing;\n"
                    "pub fn c() { match s { MemoryStrategy::BoundedDecode => DECODE_SUPPORT,\n"
                    "    _ => MemoryParameterRanges::default() }; }\n"
                    f"fn begin() -> Result<()> {{ let cfg = {call}?; Ok(()) }}\n",
                    because="never calls the checked constructor",
                )

    def test_a_second_accepting_construction_site_defeats_a_rejecting_one(self) -> None:
        """**Root cause A.** One well-behaved `::new` does not license another that is not."""
        self.assert_gate_fails(
            self.REFUSES_RUNG_TWO
            + "fn other(id: &'static str) -> Result<()> {\n"
            "    let c2 = <MlxRequestScopeConfig>::new(id, g, m, p, b, |_u, e, o| Ok(()))?;\n"
            "    Ok(())\n"
            "}\n",
            because="never calls the checked constructor",
        )

    def test_the_exemption_does_not_cover_a_conditional_inside_the_err_argument(self) -> None:
        """**Root cause B.** `Err(` + balanced `)` + end-of-body is a SHAPE. These arguments can
        early-return `Ok`, and a macro hides the whole question from both this reader and rustfmt."""
        for body in (
            "Err(match plan(edge, overlap) { Ok(()) => return Ok(()), Err(e) => e })",
            "Err(if self.tiles_ok(edge) { return Ok(()); } else { nope() })",
            "Err(or_accept!(plan(edge, overlap)))",
            "Err(plan(edge, overlap)?)",
        ):
            with self.subTest(body=body[:40]):
                self.assert_gate_fails(
                    self.REFUSES_RUNG_TWO.replace(
                        "Err(refuse_decode(id, Some(edge), Some(overlap)))", body
                    ),
                    because="never calls the checked constructor",
                )

    def test_whitespace_does_not_skip_the_configure_decode_check(self) -> None:
        """**Root cause C.** The short-circuit matched the literal `"fn configure_decode"` while the
        helper matched `\\bfn\\s+configure_decode\\b`. One extra space slipped between them and
        defeated this file's own `…_live_configure_decode_hook` test."""
        for spelling in ("fn  configure_decode", "fn\n    configure_decode"):
            with self.subTest(spelling=spelling.replace("\n", "\\n")):
                self.assert_gate_fails(
                    self.REFUSES_RUNG_TWO
                    + f"{spelling}(&mut self, e: u32, o: u32) -> Result<()> {{ self.plan(e, o) }}\n",
                    because="never calls the checked constructor",
                )

    def test_a_dead_const_in_another_file_cannot_satisfy_a_live_implemented_arm(self) -> None:
        """**Root cause D.** The const hop used a crate-wide search, so a deprecated same-named const
        anywhere satisfied a live arm. The hop is now scoped to the arm's own file."""
        self.write_provider(
            "pub fn registry() { register_memory_strategy(REG); }\n"
            "pub fn c() { match s { MemoryStrategy::BoundedDecode => DECODE_SUPPORT,\n"
            "    _ => MemoryParameterRanges::default() }; }\n",
            extra={
                "deprecated_v1.rs": "pub const DECODE_SUPPORT: MemoryStrategySupport = "
                "MemoryStrategySupport::Missing;\n"
            },
        )
        with self.assertRaises(AssertionError):
            self.run_gate()

    def test_an_or_pattern_or_guard_arm_declaring_implemented_is_not_exempt(self) -> None:
        """**Root cause E, in its full shape.** The old arm regex could not cross an or-pattern
        (``BoundedDecode | BoundedAttention =>``) or a guard containing ``>=`` (its ``[^=]*?`` stops
        at the ``=``), so a live ``Implemented`` declaration was *invisible* to it.

        Invisibility alone is harmless — zero arms fails closed. The defeat needs the second half,
        and it is the half a narrower test misses: an ordinary ``Missing`` arm elsewhere in the crate,
        which the old regex *did* see and which then satisfied its every-arm loop on its own. So the
        shape here is the real one — a hidden ``Implemented`` **paired with** a visible ``Missing`` —
        and it is exempt under the pre-fix gate.
        """
        for arm in (
            "MemoryStrategy::BoundedDecode | MemoryStrategy::BoundedAttention => "
            "MemoryStrategySupport::Implemented,",
            "MemoryStrategy::BoundedDecode if self.tiles >= 2 => MemoryStrategySupport::Implemented,",
        ):
            with self.subTest(arm=arm[:50]):
                self.assert_gate_fails(
                    self.REFUSES_RUNG_TWO
                    + f"pub fn live() {{ match s {{ {arm}\n"
                    "    _ => MemoryParameterRanges::default() }; }\n",
                    because="never calls the checked constructor",
                )

    def test_a_dead_sibling_missing_arm_does_not_license_a_live_one(self) -> None:
        """**Root cause E.** Exactly one `BoundedDecode` support arm may exist; a second is how a
        dead declaration was made to stand in for a live one."""
        self.assert_gate_fails(
            self.REFUSES_RUNG_TWO
            + "pub fn legacy() { match s { MemoryStrategy::BoundedDecode => DECODE_SUPPORT, "
            "_ => x }; }\n",
            because="never calls the checked constructor",
        )

    def test_the_exemption_does_not_cover_a_validator_installed_without_assignment(self) -> None:
        """**Round-3 residual on root cause A.** The post-construction guard matched
        `.decode_validator =`, which is one *spelling* of installing a validator, not the act. Each
        shape below puts an accepting validator into the config with no `=` next to the field."""
        for install in (
            "std::mem::replace(&mut cfg.decode_validator, "
            "Box::new(|_u, _e, _o| Ok(())));",
            "swap(&mut cfg.decode_validator, &mut accepting);",
            "install_validator(&mut cfg.decode_validator);",
        ):
            with self.subTest(install=install[:40]):
                self.assert_gate_fails(
                    self.REFUSES_RUNG_TWO + f"fn late(cfg: &mut Scope) {{ {install} }}\n",
                    because="never calls the checked constructor",
                )

    def test_a_second_same_named_support_const_in_the_file_is_not_exempt(self) -> None:
        """**Round-3 residual on root cause D.** Scoping the const hop to the arm's own file was
        necessary and not sufficient: it still asked "does a `= Missing` declaration exist?" rather
        than "is the one this arm binds Missing?". An inner module carrying a dead `Missing` beside a
        live `Implemented` answered the first question yes. This reader resolves no module paths, so
        two declarations of the name must fail closed rather than guess."""
        self.assert_gate_fails(
            "pub fn registry() { register_memory_strategy(REG); }\n"
            "pub const DECODE_SUPPORT: MemoryStrategySupport = MemoryStrategySupport::Implemented;\n"
            "mod deprecated_v1 {\n"
            "    pub const DECODE_SUPPORT: MemoryStrategySupport = MemoryStrategySupport::Missing;\n"
            "}\n"
            "pub fn c() { match s { MemoryStrategy::BoundedDecode => DECODE_SUPPORT,\n"
            "    _ => MemoryParameterRanges::default() }; }\n"
            "fn begin(id: &'static str) -> Result<()> {\n"
            "    let cfg = MlxRequestScopeConfig::new(id, g, m, use_pid, blocks,\n"
            "        move |_use_pid, edge, overlap| Err(refuse_decode(id, Some(edge), Some(overlap))),\n"
            "    )?;\n"
            "    Ok(())\n"
            "}\n",
            because="never calls the checked constructor",
        )

    def test_a_bitwise_or_argument_does_not_false_red_a_rejecting_call_site(self) -> None:
        """The one FALSE RED the probe found: a top-level `|` in an earlier argument used to swallow
        the closure's own delimiters, route-checking a call site that genuinely rejects."""
        self.assert_gate_passes(
            self.REFUSES_RUNG_TWO.replace(
                "MlxRequestScopeConfig::new(id, g, m, use_pid, blocks,",
                "MlxRequestScopeConfig::new(id, FLAG_A | FLAG_B, m, use_pid, blocks,",
            )
        )

    def test_the_exemption_does_not_cover_a_named_decode_validator(self) -> None:
        """A validator passed by name is not a shape this reader can prove refuses, so it fails
        closed rather than being taken on trust."""
        self.assert_gate_fails(
            self.REFUSES_RUNG_TWO.replace(
                "move |_use_pid, edge, overlap| "
                "Err(refuse_decode(id, Some(edge), Some(overlap)))",
                "self.decode_validator",
            ),
            because="never calls the checked constructor",
        )

    def test_the_exemption_does_not_cover_a_live_configure_decode_hook(self) -> None:
        """The trait-method half of the same rule. It is the shape a NON-MLX adopter would write, so
        it stays pinned even though no MLX family uses it."""
        self.assert_gate_fails(
            self.REFUSES_RUNG_TWO
            + "fn configure_decode(&mut self, e: u32, o: u32) -> Result<()> { self.plan(e, o) }\n",
            because="never calls the checked constructor",
        )

    def test_a_cfg_test_missing_declaration_cannot_buy_the_exemption(self) -> None:
        """The support declaration is read off the cfg(test)-blanked stream, so a test fixture that
        writes `BoundedDecode => Missing` cannot exempt a production contract that never does."""
        self.assert_gate_fails(
            "pub fn registry() { register_memory_strategy(REG); }\n"
            "pub fn r() -> MemoryParameterRanges { shared::decode_ranges(EDGES) }\n"
            "#[cfg(test)]\n"
            "mod tests {\n"
            "    pub const DECODE_SUPPORT: MemoryStrategySupport = MemoryStrategySupport::Missing;\n"
            "    fn c() { match s { MemoryStrategy::BoundedDecode => DECODE_SUPPORT,\n"
            "        _ => MemoryParameterRanges::default() }; }\n"
            "}\n",
            because="never calls the checked constructor",
        )



class SnapshotPathDerivationTests(unittest.TestCase):
    """`check_snapshot_path_derivation` must separate a `$HOME` *derivation* from a `$HOME`
    *fallback*, because a naive grep for `env::var("HOME")` cannot: 102 files matched that grep when
    the gate was written and only 15 carried the defect."""

    def setUp(self) -> None:
        self.gate = load_gate_module()

    def check(self, source: str):
        """Run the gate over a temp tree containing exactly `source`, returning the failure or None."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "crate" / "tests").mkdir(parents=True)
            (root / "crate" / "tests" / "real_weights.rs").write_text(source, encoding="utf-8")
            try:
                self.gate.check_snapshot_path_derivation(root)
            except AssertionError as error:
                return str(error)
            return None

    def test_a_derived_snapshot_path_with_no_override_fails(self) -> None:
        failure = self.check(
            'fn snapshot() -> PathBuf {\n'
            '    PathBuf::from(std::env::var("HOME").unwrap())\n'
            '        .join(".cache/mlx-gen-models/bernini_full_mlx_bf16")\n'
            '}\n'
        )
        self.assertIsNotNone(failure)
        self.assertIn("real_weights.rs:1", failure)

    def test_an_override_with_a_home_fallback_passes(self) -> None:
        self.assertIsNone(
            self.check(
                'fn converted_root() -> PathBuf {\n'
                '    std::env::var("MLX_GEN_CONVERTED_ROOT")\n'
                '        .map(PathBuf::from)\n'
                '        .unwrap_or_else(|_| {\n'
                '            PathBuf::from(std::env::var("HOME").expect("HOME"))\n'
                '                .join(".cache/mlx-gen-models")\n'
                '        })\n'
                '}\n'
            )
        )

    def test_a_bare_home_accessor_is_not_the_defect(self) -> None:
        """The `mlx-gen-scail2` / `mlx-gen-krea-realtime` shape: the helper joins nothing, and its
        callers wrap it in an override. Flagging it would make the gate wrong on 4 live files."""
        self.assertIsNone(
            self.check(
                'fn home() -> PathBuf {\n'
                '    PathBuf::from(std::env::var("HOME").unwrap())\n'
                '}\n'
                'fn snapshot_dir() -> PathBuf {\n'
                '    std::env::var("SCAIL2_SNAPSHOT_DIR")\n'
                '        .map(PathBuf::from)\n'
                '        .unwrap_or_else(|_| home().join(".cache/scail2-mlx-convert"))\n'
                '}\n'
            )
        )

    def test_a_tilde_expander_is_not_the_defect(self) -> None:
        """The `mlx-gen-wan` shape: `$HOME` expands a `~/` prefix on a value that already came from
        an override, so the override is upstream of the read rather than absent."""
        self.assertIsNone(
            self.check(
                'fn env_path(var: &str) -> Option<PathBuf> {\n'
                '    std::env::var_os(var).map(|s| {\n'
                '        let s = s.to_string_lossy();\n'
                '        if let Some(rest) = s.strip_prefix("~/") {\n'
                '            if let Some(home) = std::env::var_os("HOME") {\n'
                '                return PathBuf::from(format!("{}/{rest}", home.to_string_lossy()));\n'
                '            }\n'
                '        }\n'
                '        PathBuf::from(s.to_string())\n'
                '    })\n'
                '}\n'
            )
        )

    def test_an_override_named_by_a_computed_string_still_counts(self) -> None:
        """The `mlx-gen-mochi` shape: `env::var(format!("MOCHI_Q{bits}_DIR"))`. The override is real
        even though no literal variable name appears, which is why the gate tests for *any* non-HOME
        env read rather than for a name."""
        self.assertIsNone(
            self.check(
                'fn tier_dir(bits: u32) -> PathBuf {\n'
                '    if let Ok(d) = std::env::var(format!("MOCHI_Q{bits}_DIR")) {\n'
                '        return PathBuf::from(d);\n'
                '    }\n'
                '    PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/mochi-tiers")\n'
                '}\n'
            )
        )

    def test_a_commented_out_override_does_not_disarm_the_gate(self) -> None:
        self.assertIsNotNone(
            self.check(
                'fn snapshot() -> PathBuf {\n'
                '    // std::env::var("SOME_OVERRIDE")\n'
                '    PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/models/x")\n'
                '}\n'
            )
        )

class CrossBackendGeometryTests(unittest.TestCase):
    """`check_cross_backend_geometry` must catch each clause of its contract on its own.

    Every test below mutates **one** thing and asserts **which** message comes back. Mutating several
    at once would only prove the set fires, not that each member does, and a bare "it failed" assert
    goes inert the moment the tree would have failed for an unrelated reason.

    The pair here is synthetic on purpose: it exercises the mechanism (cast stripping, `usize`/`i32`
    spellings, same-crate identifier folding, arrays, strings) without pinning these tests to the
    real crates' current contents. `CrossBackendGeometryLiveTests` covers the real tree.
    """

    A = "crates/media/candle-gen/candle-gen-demo"
    B = "crates/media/mlx-gen/mlx-gen-demo"

    LIB = (
        'pub const MODEL_ID: &str = "demo";\n'
        "pub const SIZE_MULTIPLE: u32 = VAE_RATIO as u32 * 2;\n"
    )
    CONFIG_A = (
        "pub const VAE_RATIO: usize = 16;\n"
        "pub const FACTORS: [usize; 3] = [2, 2, 4];\n"
        "pub const SAMPLE_RATE: usize = 32_000;\n"
        "pub const NORM_EPS: f64 = 1e-5;\n"
    )
    CONFIG_B = (
        "pub const VAE_RATIO: i32 = 16;\n"
        "pub const FACTORS: [i32; 3] = [2, 2, 4];\n"
        "pub const SAMPLE_RATE: i32 = 32000;\n"
        "pub const NORM_EPS: f32 = 0.00001;\n"
    )
    FIXTURES = "pub const SHARED_FIXTURE_DIM: usize = 8;\n"
    FIXTURES_B = "pub const SHARED_FIXTURE_DIM: i32 = 8;\n"

    def setUp(self) -> None:
        self.gate = load_gate_module()
        self.gate.CROSS_BACKEND_GEOMETRY_EXEMPT_FAMILIES = {}
        self.gate.CROSS_BACKEND_GEOMETRY_EXEMPTIONS = {}
        self.gate.CROSS_BACKEND_GEOMETRY_REFERENCE = {
            "demo": {
                "SIZE_MULTIPLE": (32.0,),
                "VAE_RATIO": (16.0,),
                "FACTORS": (2.0, 2.0, 4.0),
            }
        }

    def check(self, **overrides):
        """Build the pair, apply the overrides, and return the failure text or None.

        Keys are ``"<crate>/<path>"``; a value of None deletes the file, and ``drop_crate="b"``
        drops the whole crate from the workspace the way deleting it would.
        """
        files = {
            "a/src/lib.rs": self.LIB,
            "a/src/config.rs": self.CONFIG_A,
            "a/tests/common/mod.rs": self.FIXTURES,
            "b/src/lib.rs": self.LIB,
            "b/src/config.rs": self.CONFIG_B,
            "b/tests/common/mod.rs": self.FIXTURES_B,
        }
        drop_crate = overrides.pop("drop_crate", None)
        files.update(overrides)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packages = []
            for key, source in files.items():
                side, relative = key.split("/", 1)
                if drop_crate == side:
                    continue
                crate = self.A if side == "a" else self.B
                if source is None:
                    continue
                path = root / crate / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")
            for side, crate in (("a", self.A), ("b", self.B)):
                if drop_crate == side:
                    continue
                name = "candle-gen-demo" if side == "a" else "mlx-gen-demo"
                packages.append(
                    {"name": name, "manifest_path": str(root / crate / "Cargo.toml")}
                )
            try:
                self.gate.check_cross_backend_geometry({"packages": packages}, root)
            except AssertionError as error:
                return str(error)
            return None

    # --- the baseline ----------------------------------------------------------------------------

    def test_the_unmutated_pair_passes(self) -> None:
        """Without this every "it failed" below would prove nothing."""
        self.assertIsNone(self.check())

    def test_type_and_digit_separator_spellings_are_not_divergence(self) -> None:
        """`usize` vs `i32` and `32_000` vs `32000` are the same number; a gate that red-flagged them
        would be turned off within a week."""
        self.assertIsNone(self.check())

    def test_scientific_notation_folds_to_a_number(self) -> None:
        """`1e-5` and `0.00001` are the same eps, and the baseline pair spells it both ways.

        The subtlety this pins: an identifier pattern that does not exclude the `e` of a float
        literal reads `1e-5` as the identifier `e`, fails to resolve it, and quietly demotes the
        constant to text-only comparison — losing both the numeric compare and any reference pin,
        while still reporting green."""
        self.assertIsNone(self.check())
        self.assertEqual(self.gate._const_numbers("1e-5", {}), (1e-5,))
        self.assertEqual(self.gate._const_numbers("0.00001", {}), (1e-5,))

    # --- clause: the two crates agree ------------------------------------------------------------

    def test_a_diverging_eps_is_still_caught(self) -> None:
        failure = self.check(**{"b/src/config.rs": self.CONFIG_B.replace("0.00001", "0.000001")})
        self.assertIsNotNone(failure)
        self.assertIn("`NORM_EPS` diverges", failure)

    def test_a_diverging_scalar_is_caught(self) -> None:
        """The sc-19419 defect itself: 16 on one backend, 32 on the other."""
        failure = self.check(**{"a/src/lib.rs": self.LIB.replace(" * 2;", ";")})
        self.assertIsNotNone(failure)
        self.assertIn("`SIZE_MULTIPLE` diverges", failure)

    def test_an_identifier_folds_against_its_own_crates_declaration(self) -> None:
        """`SIZE_MULTIPLE = VAE_RATIO * 2` must fold against the `VAE_RATIO` its *own* backend
        declares, so a crate that quietly redefines the base still reports what it compiles to."""
        failure = self.check(
            **{"b/src/config.rs": self.CONFIG_B.replace("VAE_RATIO: i32 = 16", "VAE_RATIO: i32 = 8")}
        )
        self.assertIsNotNone(failure)
        self.assertIn("`SIZE_MULTIPLE` diverges", failure)
        self.assertIn("`VAE_RATIO`", failure)

    def test_a_single_array_element_is_caught(self) -> None:
        """Elementwise relative max-abs-diff, not a norm or a checksum over the vector — those went
        blind to exactly this in this family."""
        failure = self.check(**{"a/src/config.rs": self.CONFIG_A.replace("[2, 2, 4]", "[2, 3, 4]")})
        self.assertIsNotNone(failure)
        self.assertIn("`FACTORS`", failure)

    def test_a_changed_array_length_is_caught(self) -> None:
        failure = self.check(
            **{"b/src/config.rs": self.CONFIG_B.replace("[i32; 3] = [2, 2, 4]", "[i32; 2] = [2, 2]")}
        )
        self.assertIsNotNone(failure)
        self.assertIn("`FACTORS`", failure)

    def test_a_diverging_non_numeric_constant_is_caught(self) -> None:
        """Strings, bools and enum paths fall back to normalized-text equality rather than being
        skipped — a skipped constant is an uncovered one."""
        failure = self.check(**{"b/src/lib.rs": self.LIB.replace('"demo"', '"other"')})
        self.assertIsNotNone(failure)
        self.assertIn("`MODEL_ID` diverges", failure)

    def test_a_constant_only_one_backend_declares_is_not_a_divergence(self) -> None:
        """The two backends legitimately declare different constants — mlx-gen carries Metal memory
        registrations candle has no analogue for. Only the shared names are a checkable claim, and a
        gate that red-flagged the rest would be a permanent red rather than a signal."""
        self.assertIsNone(
            self.check(**{"b/src/config.rs": self.CONFIG_B + "pub const ONLY_HERE: i32 = 1;\n"})
        )

    def test_a_constant_moved_to_another_file_is_still_compared(self) -> None:
        """The reason the hand-maintained file list is gone: under it, moving a declaration out of
        the listed set removed it from the comparison. Every `.rs` under `src/` is read now, so the
        divergence follows the constant wherever it goes."""
        failure = self.check(
            **{
                "b/src/config.rs": self.CONFIG_B.replace("VAE_RATIO: i32 = 16;\n", ""),
                "b/src/nested/deep.rs": "pub const VAE_RATIO: i32 = 8;\n",
            }
        )
        self.assertIsNotNone(failure)
        self.assertIn("`VAE_RATIO`", failure)

    def test_a_name_declared_twice_compares_the_set_of_values(self) -> None:
        """`MODEL_ID` is declared once per variant module in several real families. Two backends
        agree when they declare the same values under a name, whichever module each puts them in —
        and disagree when the sets differ, which is the z-image/qwen-image shape."""
        both = 'pub const MODEL_ID: &str = "demo_turbo";\n'
        self.assertIsNone(
            self.check(**{"a/src/extra.rs": both, "b/src/extra.rs": both})
        )
        failure = self.check(**{"b/src/extra.rs": both})
        self.assertIsNotNone(failure)
        self.assertIn("`MODEL_ID` diverges", failure)

    def test_a_module_path_qualifier_folds_to_the_same_crate_declaration(self) -> None:
        """`config::VAE_RATIO` and a bare `VAE_RATIO` name the same constant; the path is how it is
        reached, not what it is, and the two backends organize their modules differently."""
        self.assertIsNone(
            self.check(
                **{"b/src/lib.rs": self.LIB.replace("VAE_RATIO as u32", "config::VAE_RATIO as u32")}
            )
        )

    def test_the_backend_shim_prefix_folds(self) -> None:
        """`candle_gen::gen_core::X` and `mlx_gen::gen_core::X` are the same item named through each
        backend's own shim, so the shim segment carries no value."""
        self.assertIsNone(
            self.check(
                **{
                    "a/src/extra.rs": "pub const REG: T = candle_gen::gen_core::T { id: 1 };\n",
                    "b/src/extra.rs": "pub const REG: T = mlx_gen::gen_core::T { id: 1 };\n",
                }
            )
        )

    def test_a_digit_separator_inside_an_identifier_is_not_stripped(self) -> None:
        """The regression this replaced a lookbehind-only rule for: `SD3_5_LARGE_ID` became
        `SD35_LARGE_ID`, resolved against nothing, and reported `sd3`'s `MODEL_ID` as a divergence
        it did not have. The same rule collapsed the string `"ltx_2_3"` to `"ltx_23"`, which would
        have compared *equal* to a genuinely different `"ltx_23"`."""
        self.assertEqual(self.gate._normalize_const_value("SD3_5_LARGE_ID"), "SD3_5_LARGE_ID")
        self.assertEqual(self.gate._normalize_const_value('"ltx_2_3"'), '"ltx_2_3"')
        self.assertEqual(self.gate._normalize_const_value("32_000"), "32000")
        self.assertEqual(self.gate._normalize_const_value("0.858_090_34"), "0.85809034")
        self.assertIsNone(
            self.check(
                **{
                    "a/src/extra.rs": 'pub const NAME: &str = "demo_2_3";\n',
                    "b/src/extra.rs": (
                        'pub const DEMO_2_3_ID: &str = "demo_2_3";\n'
                        "pub const NAME: &str = crate::config::DEMO_2_3_ID;\n"
                    ),
                }
            )
        )

    def test_an_index_into_an_array_constant_folds(self) -> None:
        """`LEGAL_FRAME_COUNTS[0] as f64 / FPS` is a real declaration in both MiniMax-H3 crates. A
        gate that cannot fold it compares two durations as text and passes on any pair of
        spellings."""
        self.assertIsNone(
            self.check(
                **{
                    "a/src/extra.rs": "pub const LAST: usize = FACTORS[FACTORS.len() - 1];\n",
                    "b/src/extra.rs": "pub const LAST: i32 = 4;\n",
                }
            )
        )
        failure = self.check(
            **{
                "a/src/extra.rs": "pub const LAST: usize = FACTORS[FACTORS.len() - 1];\n",
                "b/src/extra.rs": "pub const LAST: i32 = 2;\n",
            }
        )
        self.assertIsNotNone(failure)
        self.assertIn("`LAST` diverges", failure)

    # --- clause: both agree with the reference ---------------------------------------------------

    def test_both_backends_agreeing_on_the_wrong_value_is_still_caught(self) -> None:
        """The clause that makes this gate more than a consistency check. "Fixing" a divergence by
        copying the wrong number across must not buy a green — that is how the original defect would
        have been resolved by anyone reading only the red."""
        wrong = self.LIB.replace(" * 2;", ";")
        failure = self.check(**{"a/src/lib.rs": wrong, "b/src/lib.rs": wrong})
        self.assertIsNotNone(failure)
        self.assertIn("released checkpoint read through diffusers says (32.0,)", failure)

    def test_a_reference_constant_that_is_not_declared_at_all_is_caught(self) -> None:
        """Deleting a pinned constant from both sides must not read as agreement."""
        stripped = self.CONFIG_A.replace("pub const FACTORS: [usize; 3] = [2, 2, 4];\n", "")
        failure = self.check(
            **{
                "a/src/config.rs": stripped,
                "b/src/config.rs": self.CONFIG_B.replace(
                    "pub const FACTORS: [i32; 3] = [2, 2, 4];\n", ""
                ),
            }
        )
        self.assertIsNotNone(failure)
        self.assertIn("is pinned against the diffusers reference but is not declared", failure)

    def test_a_reference_constant_declared_twice_with_different_values_is_caught(self) -> None:
        """Two values under one name leaves the gate unable to say which one the reference pins —
        which is a failure, not a coin flip."""
        failure = self.check(
            **{"a/src/extra.rs": "pub const VAE_RATIO: usize = 8;\n"}
        )
        self.assertIsNotNone(failure)
        self.assertIn("different values, so the gate cannot tell which one", failure)

    def test_a_reference_constant_that_will_not_resolve_to_numbers_is_caught(self) -> None:
        """"Cannot verify" is a violation wherever a number was required — never a pass."""
        failure = self.check(
            **{
                "a/src/config.rs": self.CONFIG_A.replace(
                    "VAE_RATIO: usize = 16", 'VAE_RATIO: &str = "sixteen"'
                ),
                "b/src/config.rs": self.CONFIG_B.replace(
                    "VAE_RATIO: i32 = 16", 'VAE_RATIO: &str = "sixteen"'
                ),
            }
        )
        self.assertIsNotNone(failure)
        self.assertIn("does not resolve to numbers", failure)

    # --- clause: exemptions cannot outlive their subject -----------------------------------------

    def test_an_exemption_suppresses_a_divergence_it_names(self) -> None:
        self.gate.CROSS_BACKEND_GEOMETRY_EXEMPTIONS = {("demo", "MODEL_ID"): "per-variant id"}
        self.assertIsNone(self.check(**{"b/src/lib.rs": self.LIB.replace('"demo"', '"other"')}))

    def test_an_exemption_for_a_constant_that_now_agrees_is_caught(self) -> None:
        """A stale exemption is a claim about the tree that the tree no longer supports. Left alone
        it becomes a hole nobody remembers opening."""
        self.gate.CROSS_BACKEND_GEOMETRY_EXEMPTIONS = {("demo", "MODEL_ID"): "per-variant id"}
        failure = self.check()
        self.assertIsNotNone(failure)
        self.assertIn("the two backends now agree about it", failure)

    def test_an_exemption_for_a_constant_no_longer_declared_on_both_sides_is_caught(self) -> None:
        self.gate.CROSS_BACKEND_GEOMETRY_EXEMPTIONS = {("demo", "GONE"): "was different once"}
        failure = self.check()
        self.assertIsNotNone(failure)
        self.assertIn("is no longer declared on both sides", failure)

    def test_an_exemption_naming_a_family_that_does_not_exist_is_caught(self) -> None:
        self.gate.CROSS_BACKEND_GEOMETRY_EXEMPTIONS = {("ghost", "X"): "gone"}
        failure = self.check()
        self.assertIsNotNone(failure)
        self.assertIn("is not a dual-backend family any more", failure)

    def test_a_reference_block_naming_a_family_that_does_not_exist_is_caught(self) -> None:
        self.gate.CROSS_BACKEND_GEOMETRY_REFERENCE["ghost"] = {"X": (1.0,)}
        failure = self.check()
        self.assertIsNotNone(failure)
        self.assertIn("drop the reference block", failure)

    def test_a_family_exemption_naming_a_family_that_does_not_exist_is_caught(self) -> None:
        self.gate.CROSS_BACKEND_GEOMETRY_EXEMPT_FAMILIES = {"ghost": "gone"}
        failure = self.check()
        self.assertIsNotNone(failure)
        self.assertIn("drop the exemption", failure)

    # --- clause: the fixture geometry agrees -----------------------------------------------------

    def test_a_diverging_fixture_constant_is_caught(self) -> None:
        """The second half of sc-19496: both crates commit byte-identical fixture bytes and load
        them through their own hand-typed geometry, so a drift here leaves both lanes internally
        consistent and both parity suites green on different shapes."""
        failure = self.check(**{"b/tests/common/mod.rs": self.FIXTURES_B.replace("= 8", "= 9")})
        self.assertIsNotNone(failure)
        self.assertIn("fixture geometry `SHARED_FIXTURE_DIM` diverges", failure)

    def test_a_fixture_constant_added_to_only_the_candle_side_is_caught(self) -> None:
        """Drift one step earlier than a value difference. Paired with the test below so that *both*
        directions of the name-set comparison are covered — one test can only exercise one of them,
        and a single-direction suite leaves the other clause free to be deleted."""
        failure = self.check(
            **{"a/tests/common/mod.rs": self.FIXTURES + "pub const SHARED_FIXTURE_X: usize = 1;\n"}
        )
        self.assertIsNotNone(failure)
        self.assertIn("`SHARED_FIXTURE_X` is declared in", failure)
        self.assertIn("candle-gen-demo/tests but not in", failure)

    def test_a_fixture_constant_added_to_only_the_mlx_side_is_caught(self) -> None:
        failure = self.check(
            **{"b/tests/common/mod.rs": self.FIXTURES_B + "pub const SHARED_FIXTURE_X: i32 = 1;\n"}
        )
        self.assertIsNotNone(failure)
        self.assertIn("`SHARED_FIXTURE_X` is declared in", failure)
        self.assertIn("mlx-gen-demo/tests but not in", failure)

    def test_a_reference_pinned_family_with_no_fixture_constants_is_caught(self) -> None:
        """Lifting the numbers back into function bodies would put them out of the gate's reach
        again while every other clause stayed green."""
        failure = self.check(**{"a/tests/common/mod.rs": "// nothing shared here\n"})
        self.assertIsNotNone(failure)
        self.assertIn("declares no `SHARED_FIXTURE_*` constants under tests/", failure)

    def test_the_fixture_clause_reads_the_mlx_side_too(self) -> None:
        failure = self.check(**{"b/tests/common/mod.rs": "// nothing shared here\n"})
        self.assertIsNotNone(failure)
        self.assertIn("declares no `SHARED_FIXTURE_*` constants under tests/", failure)

    # --- clause: coverage is the workspace, not a list --------------------------------------------

    def test_an_exempt_family_is_not_compared(self) -> None:
        self.gate.CROSS_BACKEND_GEOMETRY_EXEMPT_FAMILIES = {"demo": "synthetic"}
        self.assertIsNone(self.check(**{"b/src/lib.rs": self.LIB.replace('"demo"', '"other"')}))

    def test_a_workspace_with_no_dual_backend_pair_fails(self) -> None:
        """A gate that finds nothing to compare must be loud, not green: that is the exact failure
        mode a curated pair table had, one family at a time."""
        failure = self.check(drop_crate="b")
        self.assertIsNotNone(failure)
        self.assertIn("reported no candle-gen-X/mlx-gen-X pair at all", failure)

    def test_a_crate_without_a_lib_rs_fails(self) -> None:
        failure = self.check(**{"b/src/lib.rs": None})
        self.assertIsNotNone(failure)
        self.assertIn("src/lib.rs is missing", failure)

    def test_a_declaration_inside_a_block_comment_is_not_a_declaration(self) -> None:
        """The load-bearing test for comment stripping.

        A `///` line cannot reach column 0, so a doc comment could never have satisfied the parser
        anyway and a test built on one proves nothing about the stripper. A *block* comment can hold
        a line starting at column 0, so this is the shape that actually distinguishes a gate reading
        stripped source from one reading raw text: without stripping, `GHOST` parses as a real
        declaration on one backend only."""
        self.assertIsNone(
            self.check(
                **{"a/src/config.rs": "/*\npub const GHOST: usize = 1;\n*/\n" + self.CONFIG_A}
            )
        )

    def test_prose_claiming_the_right_value_does_not_excuse_the_wrong_one(self) -> None:
        """The exact shape of the sc-19419 defect: a comment asserting agreement that the code does
        not have. The comment is ignored and the declaration is what is judged."""
        failure = self.check(
            **{
                "a/src/config.rs": "/// Matches the sibling backend: 16.\n"
                + self.CONFIG_A.replace("VAE_RATIO: usize = 16", "VAE_RATIO: usize = 8")
            }
        )
        self.assertIsNotNone(failure)
        self.assertIn("`VAE_RATIO`", failure)


class CrossBackendGeometryLiveTests(unittest.TestCase):
    """The synthetic pair proves the mechanism; this proves it is pointed at the real crates."""

    def setUp(self) -> None:
        self.gate = load_gate_module()
        self.metadata = self.gate.cargo_metadata(True)

    def test_the_shipped_workspace_has_no_cross_backend_geometry_drift(self) -> None:
        self.gate.check_cross_backend_geometry(self.metadata, ROOT)

    def test_the_reference_pins_the_value_settled_against_diffusers(self) -> None:
        """`SIZE_MULTIPLE` is 32 — `vae_spatial_compression_ratio * patch_size[2]` = 16 * 2 — and the
        16 the candle crate shipped until sc-19419 was the VAE-only alignment. Pinned here as well as
        in the gate so that relaxing the gate's table is itself a red."""
        reference = self.gate.CROSS_BACKEND_GEOMETRY_REFERENCE["minimax-h3"]
        self.assertEqual(reference["SIZE_MULTIPLE"], (32.0,))
        self.assertEqual(reference["VAE_RATIO"], (16.0,))
        self.assertEqual(reference["VAE_RATIO_T"], (4.0,))
        self.assertEqual(reference["FRAMES_PER_CHUNK"], (17.0,))
        self.assertEqual(reference["LATENTS_PER_CHUNK"], (5.0,))
        self.assertEqual(len(reference["LEGAL_FRAME_COUNTS"]), 14)
        self.assertEqual(reference["LEGAL_FRAME_COUNTS"][0], 124.0)
        self.assertEqual(reference["LEGAL_FRAME_COUNTS"][-1], 345.0)

    def test_every_dual_backend_family_in_the_workspace_is_reached(self) -> None:
        """The sc-19496 clause, asserted with no maintained number: the families the gate compares
        are exactly the `candle-gen-X`/`mlx-gen-X` pairs `cargo metadata` reports, minus whatever
        carries a written exemption. A family added to the workspace is compared without anyone
        remembering to list it."""
        families = {family for family, _, _ in self.gate._dual_backend_families(self.metadata)}
        names = {package["name"] for package in self.metadata["packages"]}
        expected = {
            name[len("candle-gen-") :]
            for name in names
            if name.startswith("candle-gen-")
            and f"mlx-gen-{name[len('candle-gen-') :]}" in names
        }
        self.assertEqual(families, expected)
        self.assertIn("minimax-h3", families)
        self.assertEqual(self.gate.CROSS_BACKEND_GEOMETRY_EXEMPT_FAMILIES, {})

    def test_every_exemption_names_a_family_that_exists(self) -> None:
        """`check_cross_backend_geometry` enforces this too; asserted separately so that deleting
        the enforcement does not go unnoticed."""
        families = {family for family, _, _ in self.gate._dual_backend_families(self.metadata)}
        for family, constant in self.gate.CROSS_BACKEND_GEOMETRY_EXEMPTIONS:
            self.assertIn(family, families, f"{family}/{constant}")
        for family in self.gate.CROSS_BACKEND_GEOMETRY_REFERENCE:
            self.assertIn(family, families)

    def test_the_minimax_h3_fixture_geometry_is_declared_on_both_sides(self) -> None:
        """The hand-maintained fixture configs sc-19496 was filed for. Asserted structurally — equal
        name sets and equal values, with no count kept here — because a maintained number is the
        thing that goes stale."""
        families = {
            family: (candle, mlx)
            for family, candle, mlx in self.gate._dual_backend_families(self.metadata)
        }
        candle, mlx = families["minimax-h3"]
        prefix = self.gate.CROSS_BACKEND_FIXTURE_PREFIX
        left = self.gate._crate_pub_consts(candle, "tests", prefix=prefix)
        right = self.gate._crate_pub_consts(mlx, "tests", prefix=prefix)
        self.assertTrue(left)
        self.assertEqual(set(left), set(right))
        for constant in left:
            self.assertEqual(
                self.gate._canonical_const_values(left[constant], left),
                self.gate._canonical_const_values(right[constant], right),
                constant,
            )


if __name__ == "__main__":
    unittest.main()
