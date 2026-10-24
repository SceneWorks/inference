"""CPU-only selected-capture command/discovery/execution contract tests (no Rust/GPU)."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
import yaml
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("ios_selected_captures", ROOT / "scripts/ci/ios_selected_captures.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
POLICY_SPEC = importlib.util.spec_from_file_location("capture_source_policy", ROOT / "scripts/check_model_inference_policy.py")
source_policy = importlib.util.module_from_spec(POLICY_SPEC)
POLICY_SPEC.loader.exec_module(source_policy)


class SelectedCaptures(unittest.TestCase):
    def setUp(self):
        self.row = runner.REGISTRY[0]
        self.env = dict(GITHUB_EVENT_NAME="workflow_dispatch", RUNNER_NAME="nax-macos-2",
                        RUNNER_OS="macOS", RUNNER_ARCH="ARM64")

    def discovery(self):
        return f"{self.row['fqn']}: test\n\n1 test, 0 benchmarks\n"

    def execution(self):
        return ("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 99 filtered out;\n"
                f"IOS_CAPTURE_BODY_COMPLETED {self.row['fqn']}\n"
                "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 99 filtered out;\n")

    def test_every_family_is_selectable(self):
        for family in runner.FAMILIES:
            rows = runner.select(family)
            self.assertTrue(rows)
            self.assertTrue(all(row['package'] == runner.FAMILY_PACKAGES[family] for row in rows))

    def test_every_exact_binding_is_selectable(self):
        for row in runner.REGISTRY:
            self.assertEqual(runner.select(runner.identity(row)), (row,))

    def test_all_has_unique_exact_bindings(self):
        rows = runner.select("all")
        self.assertEqual(len(rows), len({runner.identity(row) for row in rows}))

    def test_arbitrary_shell_and_patterns_are_rejected(self):
        for value in ("", "*", "iris; echo bad", "--ignored", "../../Cargo.toml", "$(whoami)"):
            with self.assertRaises(ValueError):
                runner.select(value)

    def test_only_dispatch_on_exact_second_host_is_allowed(self):
        runner.require_dispatch(self.env)
        for key in self.env:
            env = dict(self.env, **{key: "other"})
            with self.assertRaises(ValueError):
                runner.require_dispatch(env)

    def test_missing_environment_refuses(self):
        with self.assertRaises(ValueError):
            runner.require_dispatch({})

    def test_commands_are_fixed_lib_exact_ignored(self):
        for row in runner.REGISTRY:
            command = runner.command(row, False)
            self.assertEqual(command[:5], ["cargo", "test", "--locked", "--no-default-features", "-p"])
            self.assertEqual(command[5:9], [row['package'], '--lib', row['fqn'], '--'])
            self.assertIn('--exact', command)
            self.assertIn('--ignored', command)
            self.assertNotIn('shell', command)

    def test_unregistered_commands_refuse(self):
        with self.assertRaises(ValueError):
            runner.command(dict(self.row, fqn="arbitrary"), False)

    def test_exact_discovery_passes(self):
        runner.verify_discovery(self.row, self.discovery())

    def test_zero_discovery_refuses(self):
        with self.assertRaises(ValueError):
            runner.verify_discovery(self.row, "0 tests, 0 benchmarks\n")

    def test_wrong_discovery_refuses(self):
        with self.assertRaises(ValueError):
            runner.verify_discovery(self.row, self.discovery().replace(self.row['fqn'], "wrong"))

    def test_multiple_discovery_refuses(self):
        with self.assertRaises(ValueError):
            runner.verify_discovery(self.row, self.discovery() + "extra: test\n")

    def test_missing_discovery_count_refuses(self):
        with self.assertRaises(ValueError):
            runner.verify_discovery(self.row, self.discovery().split('\n')[0])

    def test_both_completed_bodies_pass(self):
        runner.verify_execution(self.row, self.execution())

    def test_silent_ignored_child_refuses(self):
        with self.assertRaises(ValueError):
            runner.verify_execution(self.row, self.execution().replace('1 passed', '0 passed', 1))

    def test_parent_only_refuses(self):
        with self.assertRaises(ValueError):
            runner.verify_execution(self.row, self.execution().splitlines()[0])

    def test_missing_body_evidence_refuses(self):
        with self.assertRaises(ValueError):
            runner.verify_execution(self.row, self.execution().replace('IOS_CAPTURE_BODY_COMPLETED', 'OTHER'))

    def test_duplicate_body_evidence_refuses(self):
        with self.assertRaises(ValueError):
            runner.verify_execution(self.row, self.execution() + f"IOS_CAPTURE_BODY_COMPLETED {self.row['fqn']}\n")

    def test_failed_summary_refuses(self):
        with self.assertRaises(ValueError):
            runner.verify_execution(self.row, self.execution().replace('0 failed', '1 failed', 1))

    def test_failed_exit_is_preserved_and_receipt_records_no_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            env = dict(self.env, RUNNER_TEMP=temp)
            with patch.dict(runner.os.environ, env, clear=True), patch.object(runner.subprocess, 'run', return_value=subprocess.CompletedProcess([], 23, 'intentional failure\n')):
                self.assertEqual(runner.run(runner.identity(self.row)), 23)
            receipt = json.loads((Path(temp) / 'ios-selected-captures/receipt.json').read_text(encoding='utf-8'))
            self.assertEqual((receipt['selected'], receipt['completed']), (1, 0))

    def test_actual_entrypoint_rejects_unknown_selection_without_cargo(self):
        result = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/ci/ios_selected_captures.py'), '--select', 'bad;command'], capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 2)

    def test_actual_entrypoint_lists_exact_registry_without_execution(self):
        result = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/ci/ios_selected_captures.py'), '--select', runner.identity(self.row), '--list'], capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [self.row])

    def test_child_helpers_include_ignored_and_body_protocol(self):
        for rel in sorted({row['source'] for row in runner.REGISTRY}):
            text = (ROOT / rel).read_text(encoding='utf-8')
            if 'fn in_cpu_child' in text or 'fn cpu_child' in text:
                self.assertIn('--include-ignored', text, rel)
                self.assertIn('IOS_CAPTURE_IN_CHILD', text, rel)
        for row in runner.REGISTRY:
            text = (ROOT / row['source']).read_text(encoding='utf-8')
            self.assertIn('IOS_CAPTURE_BODY_COMPLETED ' + row['fqn'], text)
            test = text[:text.index('fn ' + row['fqn'].split('::')[-1] + '()')]
            self.assertIn('#[ignore', test[test.rfind('#[test]'):])

    def test_ordinary_cpu_children_do_not_dispatch_ignored_captures(self):
        cases = [
            ('crates/media/mlx-gen/src/ios_quantum.rs',
             'cpu_quantum_materializes_refuses_and_restores_cache',
             'ios_quantum::cpu_quantum_materializes_refuses_and_restores_cache'),
            ('crates/media/mlx-gen/src/sampler.rs',
             'curated_solver_step_guards_preserve_parity_and_refuse_before_evaluation',
             'sampler::tests::curated_solver_step_guards_preserve_parity_and_refuse_before_evaluation'),
        ]
        for path, function, target in cases:
            source = (ROOT / path).read_text(encoding='utf-8')
            start = source.index('fn ' + function)
            dispatch = source[start:start + 800]
            self.assertIn(target, dispatch, path)
            self.assertNotIn('--include-ignored', dispatch, path)

    def test_capture_markers_and_registry_are_bidirectionally_complete(self):
        markers = set()
        for path in (ROOT / 'crates/media/mlx-gen').rglob('*.rs'):
            markers.update(re.findall(r'IOS_CAPTURE_BODY_COMPLETED ([\w:]+)', path.read_text(encoding='utf-8')))
        self.assertEqual(markers, {row['fqn'] for row in runner.REGISTRY})

    def test_non_inference_guards_remain_ordinary(self):
        rows = [("mlx-gen-iris/src/quantum_proofs.rs", "proof_policy_cannot_be_disabled_by_print_only"),
                ("mlx-gen-iris/src/anchor_tests.rs", "missing_phase_is_not_a_zero_or_a_skipped_assertion"),
                ("mlx-gen-sdxl/src/quantum_tests.rs", "measured_lower_bound_is_hard_and_overpricing_is_report_only"),
                ("mlx-gen-sdxl/src/quantum_tests.rs", "ios_boundary_preserves_actual_operand_dtype_and_values"),
                ("mlx-gen-sdxl/src/text_encoder.rs", "final_pool_prices_live_eos_and_native_fallback_scalar_operands"),
                ("mlx-gen-sdxl/src/ios_pricing.rs", "ancestral_body_matching_preserves_precision_and_cfg_coordinates"),
                ("mlx-gen-z-image/src/quantum_tests.rs", "streamed_read_capacity_distinguishes_dtype_and_each_page_boundary"),
                ("mlx-gen-lens/src/registry.rs", "sparse_footprint_growth_rewrites_the_exact_payload_extent"),
                ("mlx-gen-anima/src/quantum_tests.rs", "sdpa_scores_follow_the_metal_kernel_support")]
        for rel, name in rows:
            source = (ROOT / 'crates/media/mlx-gen' / rel).read_text(encoding='utf-8')
            self.assertIn('fn ' + name + '()', source, rel)
            matches = [item for item in source_policy.functions(source) if item['name'] == name]
            self.assertEqual(len(matches), 1, rel)
            self.assertFalse(matches[0]['ignored'], rel)

    def test_full_request_rgb_witness_is_preserved_as_a_manual_model_proof(self):
        name = "optimized_bf16_img2img_preserves_full_request_rgb"
        path = "crates/media/mlx-gen/mlx-gen-sdxl/src/quantum_tests_additivity.rs"
        functions = source_policy.functions((ROOT / path).read_text(encoding="utf-8"))
        test = next(item for item in functions if item["name"] == name)
        self.assertTrue(test["ignored"])
        self.assertTrue(test["canonical"])
        rows = json.loads((ROOT / "scripts/ci/model-inference-proofs.json").read_text(encoding="utf-8"))
        self.assertTrue(any(row["package"] == "mlx-gen-sdxl" and row["source"] == path
                            and row["fqn"].endswith("::" + name) for row in rows))

    def test_profile_specific_selection_preserves_the_legacy_compile_binding(self):
        workflow = yaml.safe_load((ROOT / '.github/workflows/real-weights.yml').read_text(encoding='utf-8'))
        inputs = workflow[True]['workflow_dispatch']['inputs']
        self.assertNotIn('ios_capture_selection', inputs)
        # Frozen protocol default, not a corpus assertion: capture's profile-specific key
        # must not enter the compile profile's default payload or strict unknown-key contract.
        default = inputs['ios_pricing_compile_binding']['default']
        self.assertEqual(default, '{"source_sha":"","manifest_sha256":"","host":"second-mac"}')
        self.assertNotIn('selection', json.loads(default))
        capture = workflow['jobs']['ios-selected-captures']
        self.assertEqual(capture['env']['IOS_CAPTURE_BINDING'], '${{ inputs.ios_pricing_compile_binding }}')
        self.assertIn('--binding "$IOS_CAPTURE_BINDING"', capture['steps'][2]['run'])
        self.assertNotIn('fromJSON(', json.dumps(capture))
        compile_env = workflow['jobs']['ios-pricing-compile-only']['env']
        for key, value in {'SOURCE_SHA': 'source_sha', 'SOURCE_MANIFEST_SHA256': 'manifest_sha256',
                           'COMPILE_HOST': 'host'}.items():
            self.assertEqual(compile_env[key], '${{ fromJSON(inputs.ios_pricing_compile_binding).' + value + ' }}')
        self.assertEqual(compile_env['COMPILE_PHASE'],
                         "${{ fromJSON(inputs.ios_pricing_compile_binding).phase || 'compile-only' }}")
        self.assertEqual(runner.selection_from_binding(default), 'iris')

    def test_json_selection_preserves_all_families_and_every_exact_registered_binding(self):
        for choice in ('all', *runner.FAMILIES, *(runner.identity(row) for row in runner.REGISTRY)):
            with self.subTest(selection=choice):
                supplied = json.dumps({'selection': choice})
                projected = runner.selection_from_binding(supplied)
                self.assertEqual(runner.select(projected), runner.select(choice))

    def test_blank_binding_and_absent_key_default_but_invalid_present_selection_refuses(self):
        for supplied in ('', '  ', '{}', '{"source_sha":"","manifest_sha256":"","host":"second-mac"}'):
            self.assertEqual(runner.selection_from_binding(supplied), 'iris')
        for supplied in ('not-json', '[]', 'null', '{"selection":""}', '{"selection":null}',
                         '{"selection":false}', '{"selection":[]}', '{"selection":"unknown"}',
                         '{"selection":"sdxl; echo bad"}'):
            with self.subTest(binding=supplied), self.assertRaises(ValueError):
                runner.selection_from_binding(supplied)

    def test_actual_binding_entrypoint_lists_defaults_and_explicit_selections_without_execution(self):
        for supplied, selection in (('', 'iris'), ('{}', 'iris'), ('{"selection":"sdxl"}', 'sdxl'),
                                    ('{"selection":"all"}', 'all')):
            result = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/ci/ios_selected_captures.py'),
                                     '--binding', supplied, '--list'], capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), list(runner.select(selection)))
        for supplied in ('{"selection":""}', '{"selection":"unknown"}'):
            result = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/ci/ios_selected_captures.py'),
                                     '--binding', supplied, '--list'], capture_output=True, text=True, encoding='utf-8')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('capture prerequisite/validation failure', result.stderr)

    def test_workflow_is_dispatch_only_and_host_enforced(self):
        text = (ROOT / '.github/workflows/real-weights.yml').read_text(encoding='utf-8')
        job = text.split('  ios-selected-captures:\n')[1].split('  ios-pricing-compile-only:\n')[0]
        self.assertIn("github.event_name == 'workflow_dispatch'", job)
        self.assertIn('runs-on: [self-hosted, macOS, ARM64, rw-llm]', job)
        self.assertIn('[[ "$RUNNER_NAME" == nax-macos-2 ]]', job)
        self.assertIn('--binding "$IOS_CAPTURE_BINDING"', job)
        self.assertNotIn('campaign.py', job)


class ActualLibtestProtocol(unittest.TestCase):
    """Real, tiny CPU rustc/libtest processes, not MLX/native accounting validation."""
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.row = runner.REGISTRY[0]
        module, name = cls.row['fqn'].split('::')
        for mutation in (False, True):
            source = cls.root / f'capture-{mutation}.rs'
            include = '' if mutation else ', "--include-ignored"'
            source.write_text('mod ' + module + ' { #[test] #[ignore] fn ' + name + '() {\n'
                'if std::env::var_os("BODY_CHILD").is_none() {\n'
                'let status = std::process::Command::new(std::env::current_exe().unwrap())\n'
                '.args(["--exact", "' + cls.row['fqn'] + '", "--nocapture"' + include + '])\n'
                '.env("BODY_CHILD", "1").status().unwrap(); assert!(status.success()); return; }\n'
                'assert_eq!(2 + 2, 4); eprintln!("\\nIOS_CAPTURE_BODY_COMPLETED ' + cls.row['fqn'] + '");\n} }\n',
                encoding='utf-8')
            result = subprocess.run(['nice', '-n', '10', str(Path.home() / '.cargo/bin/rustc'),
                                     '--crate-name', 'capture_protocol', '--test', str(source),
                                     '-C', 'debuginfo=0', '-o', str(cls.root / f'capture-{mutation}')],
                                    capture_output=True, text=True, encoding='utf-8')
            if result.returncode:
                raise RuntimeError('CPU Rust protocol fixture did not compile: ' + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def invoke(self, mutation, discover=False):
        result = subprocess.run([str(self.root / f'capture-{mutation}'), '--exact', self.row['fqn'],
                                 '--ignored', '--list' if discover else '--nocapture'],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def test_actual_ignored_discovery(self):
        runner.verify_discovery(self.row, self.invoke(False, True))

    def test_actual_parent_and_body_completion(self):
        runner.verify_execution(self.row, self.invoke(False))

    def test_compiling_missing_include_ignored_mutation_is_rejected(self):
        # This mutant is deliberately a successful parent over a silently skipped ignored child.
        # The selected runner must still reject it based on actual completed counts/body evidence.
        with self.assertRaises(ValueError):
            runner.verify_execution(self.row, self.invoke(True))


if __name__ == '__main__':
    unittest.main()
