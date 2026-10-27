"""Ordinary CI checks only: source/schema/protocol/filesystem mocks, never NN execution."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


policy = module('model_policy_test', ROOT / 'scripts/check_model_inference_policy.py')
proof = module('model_proof_test', ROOT / 'scripts/ci/model_inference_proofs.py')


def row(**change):
    value = {'package': 'mlx-gen-test', 'target': 'lib', 'fqn': 'tests::model_case',
             'source': 'case.rs', 'execution_role': 'direct', 'backend': 'cpu',
             'fixture_identity': 'source-defined synthetic fixture (protocol mock only)',
             'configuration_source': 'case.rs::tests::model_case',
             'output_contract': 'existing test assertions and result output'}
    value.update(change)
    return value


def output(value, count=1, devices=('mlx-cpu',), parent=False):
    text = 'test result: ok. 1 passed; 0 failed; 0 ignored;\n' * count
    text += 'MODEL_INFERENCE_BODY_COMPLETED ' + value['fqn'] + '\n'
    for device in devices:
        text += 'MODEL_INFERENCE_DEVICE ' + value['fqn'] + ' ' + device + '\n'
    if parent:
        text += 'MODEL_INFERENCE_PARENT_COMPLETED ' + value['fqn'] + '\n'
    return text


class RustPolicyParsing(unittest.TestCase):
    def test_canonical_prefix_keeps_prior_prerequisite_reason(self):
        source = '#[test]\n#[ignore = "' + policy.CANONICAL + '; requires a fixture"]\nfn model_case() { }'
        items = policy.functions(source)
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0]['ignored'])
        self.assertTrue(items[0]['canonical'])

    def test_no_helper_or_quoted_function_is_a_test(self):
        source = '''
// #[test] fn comment_fake() {}
const SCRIPT: &str = r###"#[test] fn raw_fake() {}"###;
#[test] fn actual() { fn private_helper() {} let _ = 'x'; }
fn lifetime<'a>(x: &'a str) -> &'a str { x }
'''
        self.assertEqual([item['name'] for item in policy.functions(source)], ['actual'])

    def test_multiline_attributes_and_inline_modules(self):
        source = 'mod tests { #[test]\n#[ignore =\n"' + policy.CANONICAL + '"]\nfn actual() {} }'
        item = policy.functions(source)[0]
        self.assertEqual(item['inline_modules'], ('tests',))
        self.assertTrue(item['canonical'])

    def test_registry_case_cannot_become_ordinary_or_unreachable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'case.rs'
            text = '#[test]\n#[ignore = "' + policy.CANONICAL + '"]\nfn model_case() { println!("MODEL_INFERENCE_BODY_COMPLETED tests::model_case"); }'
            path.write_text(text, encoding='utf-8')
            self.assertEqual(policy.registry_errors(root, [row()]), [])
            path.write_text(text.replace('#[ignore = "' + policy.CANONICAL + '"]', ''), encoding='utf-8')
            self.assertTrue(policy.registry_errors(root, [row()]))
            path.write_text(text, encoding='utf-8')
            self.assertTrue(policy.registry_errors(root, [row(fqn='tests::missing')]))

    def test_new_tagged_case_requires_registration_not_a_pinned_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'crates/example/src/lib.rs'
            path.parent.mkdir(parents=True)
            path.write_text('#[test]\n#[ignore = "' + policy.CANONICAL + '"]\nfn new_case() {}', encoding='utf-8')
            self.assertTrue(policy.registry_errors(root, []))


class WorkflowPolicy(unittest.TestCase):
    def test_ordinary_ci_cannot_override_model_ignores(self):
        for flag in ('--ignored', '--include-ignored', '--run-ignored all'):
            workflow = {'on': {'pull_request': None}, 'jobs': {'test': {'steps': [{'run': 'cargo test -- ' + flag}]}}}
            self.assertTrue(policy.workflow_errors(workflow, 'ci.yml'))
        ordinary = {'on': {'push': None, 'pull_request': None}, 'jobs': {'test': {'steps': [{'run': 'cargo test --locked --lib --tests'}]}}}
        self.assertEqual(policy.workflow_errors(ordinary, 'ci.yml'), [])

    def test_model_workflow_is_manual_only_and_does_not_replace_pending_dispatches(self):
        # The execution route, not its workflow filename, establishes the policy.
        workflow = {'on': {'workflow_dispatch': None}, 'jobs': {'proof': {'steps': [
            {'run': 'python3 scripts/ci/model_inference_proofs.py --backend mlx'}]}}}
        self.assertEqual(policy.workflow_errors(workflow, 'arbitrary-name.yml'), [])
        workflow['on']['push'] = None
        self.assertTrue(policy.workflow_errors(workflow, 'arbitrary-name.yml'))
        del workflow['on']['push']
        workflow['concurrency'] = {'group': 'backend', 'cancel-in-progress': False}
        self.assertTrue(policy.workflow_errors(workflow, 'arbitrary-name.yml'))

    def test_normal_ci_cannot_call_model_proof_runner_or_workflow(self):
        for job in ({'steps': [{'run': 'python3 scripts/ci/model_inference_proofs.py --run'}]},
                    {'uses': './.github/workflows/model-inference-proofs.yml'}):
            self.assertTrue(policy.workflow_errors({'on': ['push'], 'jobs': {'bad': job}}, 'ci.yml'))


class WorkflowExecutionGraph(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def script(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
        return path

    def workflow(self, command, event='push', guard=None):
        job = {'steps': [{'run': command}]}
        if guard:
            job['if'] = guard
        return {'on': {event: None, 'workflow_dispatch': None}, 'jobs': {'execute': job}}

    def errors(self, workflow):
        return policy.workflow_errors(workflow, 'renamable.yml', self.root)

    def test_schedule_model_workflow_rejected(self):
        workflow = self.workflow('cargo test -p fixture -- --ignored', 'schedule')
        self.assertTrue(any('automatic schedule' in error and 'model' in error for error in self.errors(workflow)))

    def test_all_non_dispatch_events_are_automatic(self):
        for event in ('release', 'repository_dispatch', 'workflow_call', 'workflow_run', 'merge_group', 'pull_request_target'):
            with self.subTest(event=event):
                self.assertTrue(self.errors(self.workflow('cargo test -- --ignored', event)))

    def test_push_model_script_edge_rejected_with_source_chain(self):
        self.script('scripts/inner.sh', 'cargo test -p fixture -- --ignored\n')
        self.script('scripts/outer.sh', 'bash scripts/inner.sh\n')
        errors = self.errors(self.workflow('bash scripts/outer.sh'))
        self.assertTrue(any('model' in error and 'scripts/outer.sh' in error and 'scripts/inner.sh' in error for error in errors))

    def test_inline_shell_comment_cannot_hide_the_next_source_edge(self):
        self.script('scripts/model.py', 'import mlx.core as mx\nmx.eval(mx.array([1]))\n')
        for shell in ('bash', 'pwsh'):
            with self.subTest(shell=shell):
                extension = 'sh' if shell == 'bash' else 'ps1'
                path = self.script('scripts/driver.' + extension,
                                   'echo metadata # harmless comment\npython3 scripts/model.py\n')
                errors = self.errors(self.workflow(f'{shell} {path}'))
                self.assertTrue(any('scripts/model.py' in error and 'learned-framework' in error
                                    for error in errors), errors)

    def test_comment_boundary_keeps_metadata_tool_substitutions_ordinary(self):
        self.script('scripts/metadata.sh', 'echo check # metadata only\nvalues=$(strings fixture.metallib)\necho "$values"\n')
        self.assertEqual(self.errors(self.workflow('bash scripts/metadata.sh')), [])

    def test_shell_comments_preserve_quoted_escaped_and_word_internal_hashes(self):
        self.script('scripts/model.py', 'import mlx.core as mx\n')
        for text in ('echo "#literal" foo#bar \\#escaped # python3 scripts/model.py\n',
                     "echo '#literal' # cargo test -- --ignored\necho metadata\n"):
            with self.subTest(text=text):
                self.assertEqual(self.errors(self.workflow(text)), [])

    def test_python_process_edge_reaches_model(self):
        self.script('scripts/driver.py', 'import subprocess\ncommand = ["cargo", "test", "--", "--ignored"]\nsubprocess.run(command)\n')
        self.assertTrue(any('model' in error for error in self.errors(self.workflow('python3 scripts/driver.py'))))

    def test_cmd_and_powershell_script_edges_reach_model(self):
        for extension in ('cmd', 'ps1'):
            with self.subTest(extension=extension):
                self.script('scripts/driver.' + extension, 'cargo test -- --ignored\n')
                self.assertTrue(any('model' in error for error in self.errors(self.workflow('scripts/driver.' + extension))))

    def test_unresolved_dynamic_execution_fails_closed(self):
        self.assertTrue(any('unknown' in error for error in self.errors(self.workflow('"$DYNAMIC_RUNNER" --run'))))

    def test_nested_unresolved_dynamic_execution_fails_closed(self):
        self.script('scripts/opaque.sh', '"$DYNAMIC_RUNNER" --run\n')
        self.assertTrue(any('unknown' in error for error in self.errors(self.workflow('bash scripts/opaque.sh'))))

    def test_dynamic_and_model_paths_pass_inside_dispatch_gate(self):
        for command in ('"$DYNAMIC_RUNNER" --run', 'cargo test -- --ignored'):
            with self.subTest(command=command):
                self.assertEqual(self.errors(self.workflow(command, guard="github.event_name == 'workflow_dispatch'")), [])

    def test_dispatch_substring_in_or_is_not_a_dominating_gate(self):
        self.assertTrue(self.errors(self.workflow('cargo test -- --ignored', guard="github.event_name == 'workflow_dispatch' || always()")))

    def test_step_gate_is_a_dominating_gate(self):
        workflow = self.workflow('cargo test -- --ignored')
        workflow['jobs']['execute']['steps'][0]['if'] = "github.event_name == 'workflow_dispatch'"
        self.assertEqual(self.errors(workflow), [])

    def test_dependent_job_requires_success_or_its_own_gate(self):
        workflow = {'on': ['push', 'workflow_dispatch'], 'jobs': {
            'select': {'if': "github.event_name == 'workflow_dispatch'", 'steps': [{'run': 'echo selected'}]},
            'execute': {'needs': 'select', 'steps': [{'run': 'cargo test -- --ignored'}]}}}
        self.assertEqual(self.errors(workflow), [])
        workflow['jobs']['execute']['if'] = 'always()'
        self.assertTrue(self.errors(workflow))
        workflow['jobs']['execute']['if'] = "always() && needs.select.result == 'success'"
        self.assertEqual(self.errors(workflow), [])

    def test_unknown_output_comparison_is_not_a_gate(self):
        self.assertTrue(self.errors(self.workflow('cargo test -- --ignored', guard="needs.config.outputs.mode == 'w1'")))

    def test_metadata_build_lint_listing_and_ordinary_tests_pass(self):
        for command in ('cargo metadata --no-deps', 'cargo build --all-targets', 'cargo clippy --all-targets',
                        'cargo test --locked --lib', 'cargo test -- --list --ignored', 'cargo test --no-run -- --ignored'):
            with self.subTest(command=command):
                self.assertEqual(self.errors(self.workflow(command)), [])
        self.script('scripts/metadata.py', 'import subprocess\nsubprocess.run(["cargo", "metadata", "--no-deps"])\n')
        self.assertEqual(self.errors(self.workflow('python3 scripts/metadata.py')), [])

    def test_metadata_script_can_resolve_shell_function(self):
        self.script('scripts/meta.sh', 'metadata() { cargo metadata --no-deps; }\nmetadata\n')
        self.assertEqual(self.errors(self.workflow('bash scripts/meta.sh')), [])

    def test_dispatch_only_unknown_driver_is_not_automatic(self):
        workflow = self.workflow('"$DYNAMIC_RUNNER" --run')
        workflow['on'] = {'workflow_dispatch': None}
        self.assertEqual(self.errors(workflow), [])

    def test_bare_and_unconfirmed_all_dispatch_fail_fast(self):
        self.assertIsNotNone(policy.selection_error(''))
        self.assertIsNotNone(policy.selection_error('all'))
        self.assertIsNone(policy.selection_error('all', True))
        self.assertIsNone(policy.selection_error('llm'))
        self.assertIsNone(policy.selection_error('media'))

    def test_profile_default_and_confirmation_are_enforced_without_filename(self):
        workflow = {'on': {'workflow_dispatch': {'inputs': {
            'profile': {'required': True, 'default': '', 'options': ['', 'all', 'llm']},
            'confirm_all': {'type': 'boolean', 'default': False}}}}, 'jobs': {
            'execute': {'if': "inputs.profile == 'llm' || (inputs.profile == 'all' && inputs.confirm_all)",
                        'steps': [{'run': 'cargo test -- --ignored'}]}}}
        self.assertEqual(self.errors(workflow), [])
        workflow['on']['workflow_dispatch']['inputs']['profile']['default'] = 'all'
        self.assertTrue(self.errors(workflow))
        workflow['on']['workflow_dispatch']['inputs']['profile']['default'] = ''
        workflow['jobs']['execute']['if'] = "inputs.profile == 'all' || inputs.profile == 'llm'"
        self.assertTrue(self.errors(workflow))

    def test_current_workflow_controls_preserve_profiles_and_non_model_jobs(self):
        import yaml
        real = yaml.safe_load((ROOT / '.github/workflows/real-weights.yml').read_text(encoding="utf-8"))
        events = real.get('on', real.get(True))
        self.assertEqual(set(events), {'workflow_dispatch'})
        selection = events['workflow_dispatch']['inputs']
        self.assertEqual(selection['profile']['default'], '')
        self.assertNotIn('confirm_all', selection)
        self.assertEqual(len(selection), 25)
        self.assertIn('all', selection['profile']['options'])
        self.assertIn('validate-selection', real['jobs'])
        kv = yaml.safe_load((ROOT / '.github/workflows/kv-poc-campaign.yml').read_text(encoding="utf-8"))
        self.assertIn('push', kv.get('on', kv.get(True)))
        possible = policy.automatic_jobs(kv, 'push')
        for name in ('a1', 'a3', 'a2', 'b', 'nf', 'c', 'd', 'd-control'):
            self.assertFalse(possible[name], name)
        for name in ('config', 'probe', 'build', 'w2-assets', 'w2-build', 'cleanup'):
            self.assertTrue(possible[name], name)

    def test_dynamic_python_named_variable_is_not_interpreter_proof(self):
        self.assertTrue(self.errors(self.workflow('"$DYNAMIC_PYTHON" -m unittest')))

    def test_unresolved_runner_cannot_claim_a_listing_protocol(self):
        self.assertTrue(self.errors(self.workflow('"$DYNAMIC_RUNNER" --list --ignored')))

    def test_sudo_and_timeout_cannot_hide_model_script_edges(self):
        self.script('scripts/model.sh', 'cargo test -- --ignored\n')
        for command in ('sudo bash scripts/model.sh', 'timeout 10 bash scripts/model.sh'):
            with self.subTest(command=command):
                self.assertTrue(any('model' in error for error in self.errors(self.workflow(command))))

    def test_source_proven_tool_lookup_is_not_a_function_name_exemption(self):
        template = ("resolve_tool() {\n"
                    "  if command -v lms >/dev/null 2>&1; then command -v lms\n"
                    '  elif [ -x "$HOME/.lmstudio/bin/lms" ]; then echo "FALLBACK"\n'
                    "  fi\n}\n"
                    'tool="$(resolve_tool)"\n"$tool" ps --json\n')
        self.script('scripts/tool.sh', template.replace('FALLBACK', '$HOME/.lmstudio/bin/lms'))
        self.assertEqual(self.errors(self.workflow('bash scripts/tool.sh')), [])
        self.script('scripts/tool.sh', template.replace('FALLBACK', '$DYNAMIC_RUNNER'))
        self.assertTrue(self.errors(self.workflow('bash scripts/tool.sh')))
        self.script('scripts/tool.sh', template.replace('$HOME/.lmstudio/bin/lms', '$DYNAMIC_RUNNER').replace('FALLBACK', '$DYNAMIC_RUNNER'))
        self.assertTrue(self.errors(self.workflow('bash scripts/tool.sh')))

    def test_python_cli_self_test_does_not_traverse_unused_model_branch(self):
        self.script('scripts/cli.py', "import argparse, subprocess\n"
                    "def produce():\n    subprocess.run(['cargo', 'test', '--', '--ignored'])\n"
                    "def main():\n    p = argparse.ArgumentParser()\n"
                    "    p.add_argument('--self-test', action='store_true')\n"
                    "    args = p.parse_args()\n    if args.self_test:\n        return 0\n    produce()\n"
                    "if __name__ == '__main__':\n    main()\n")
        self.assertEqual(self.errors(self.workflow('python3 scripts/cli.py --self-test')), [])
        self.assertTrue(any('model' in error for error in self.errors(self.workflow('python3 scripts/cli.py'))))

    def test_python_metadata_argv_in_distinct_functions_resolves_locally(self):
        self.script('scripts/meta.py', "import subprocess\n"
                    "def one():\n    command = ['cargo', 'metadata', '--no-deps']\n    subprocess.run(command)\n"
                    "def two():\n    command = ['cargo', 'tree']\n    subprocess.run(command)\n"
                    "one()\ntwo()\n")
        self.assertEqual(self.errors(self.workflow('python3 scripts/meta.py')), [])

    def test_python_argv_mutation_cannot_hide_model_execution(self):
        for mutation in ("command.append('--ignored')", "command.extend(['--include-ignored'])"):
            self.script('scripts/mutating.py', "import subprocess\ncommand = ['cargo', 'test', '--']\n" + mutation + "\nsubprocess.run(command)\n")
            self.assertTrue(any('model' in error for error in self.errors(self.workflow('python3 scripts/mutating.py'))))
        self.script('scripts/mutating.py', "import subprocess\ncommand = ['cargo', 'test', '--no-run', '--', '--ignored']\ncommand.remove('--no-run')\nsubprocess.run(command)\n")
        self.assertTrue(self.errors(self.workflow('python3 scripts/mutating.py')))

    def test_computed_python_process_and_local_alias_fail_closed(self):
        self.script('scripts/computed.py', "import subprocess\ngetattr(subprocess, 'run')(['cargo', 'test', '--', '--ignored'])\n")
        self.assertTrue(self.errors(self.workflow('python3 scripts/computed.py')))
        self.script('scripts/computed.py', "import subprocess\ndef model():\n    subprocess.run(['cargo', 'test', '--', '--ignored'])\nrun = model\nrun()\n")
        self.assertTrue(any('model' in error for error in self.errors(self.workflow('python3 scripts/computed.py'))))
        self.script('scripts/computed.py', "import os\nos.execv('/usr/bin/python3', ['python3', 'model.py'])\n")
        self.assertTrue(self.errors(self.workflow('python3 scripts/computed.py')))

    def test_local_python_import_edges_are_resolved_without_executing_them(self):
        self.script('scripts/helper.py', "import subprocess\ndef run():\n    subprocess.run(['cargo', 'test', '--', '--ignored'])\n")
        self.script('scripts/wrapper.py', "from helper import run\nrun()\n")
        self.assertTrue(any('model' in error and 'scripts/helper.py' in error for error in self.errors(self.workflow('python3 scripts/wrapper.py'))))
        self.script('scripts/helper.py', "import subprocess\ndef run():\n    subprocess.run(['cargo', 'metadata'])\n")
        self.assertEqual(self.errors(self.workflow('python3 scripts/wrapper.py')), [])

    def test_dynamic_proof_interpreter_cannot_hide_shared_pending_group(self):
        workflow = {'on': {'workflow_dispatch': None}, 'concurrency': {'group': 'shared'}, 'jobs': {
            'proof': {'steps': [{'run': '"$PYTHON" scripts/ci/model_inference_proofs.py --backend mlx'}]}}}
        self.assertTrue(self.errors(workflow))

    def test_selection_cli_returns_clear_fail_fast_diagnostics(self):
        import contextlib
        import io
        for profile, confirmed, expected in (('', 'false', 1), ('all', 'false', 1), ('all', 'true', 0), ('llm', 'false', 0)):
            stream = io.StringIO()
            with patch('sys.argv', ['policy', '--validate-selection']), patch.dict(os.environ, {'MODEL_PROFILE': profile, 'MODEL_CONFIRM_ALL': confirmed}), contextlib.redirect_stderr(stream):
                self.assertEqual(policy.main(), expected)
            if expected:
                self.assertIn('::error::', stream.getvalue())

    def test_python_class_constructor_cannot_hide_native_model_execution(self):
        self.script('scripts/helper.py', "import subprocess\nclass Launcher:\n    def __init__(self):\n        subprocess.run(['cargo', 'test', '--', '--ignored'])\n")
        self.script('scripts/wrapper.py', "from helper import Launcher\nLauncher()\n")
        self.assertTrue(any('model' in error for error in self.errors(self.workflow('python3 scripts/wrapper.py'))))
        self.script('scripts/helper.py', "import subprocess\nclass Launcher:\n    def run(self):\n        subprocess.run(['cargo', 'test', '--', '--ignored'])\n")
        self.script('scripts/wrapper.py', "from helper import Launcher\nLauncher().run()\n")
        self.assertTrue(self.errors(self.workflow('python3 scripts/wrapper.py')))

    def test_python_process_alias_cannot_hide_model_argv(self):
        self.script('scripts/alias.py', "import subprocess as child\nlaunch = child.run\nlaunch(['cargo', 'test', '--', '--ignored'])\n")
        self.assertTrue(any('model' in error for error in self.errors(self.workflow('python3 scripts/alias.py'))))

    def test_reusable_model_workflow_inherits_the_caller_dispatch_gate(self):
        self.script('.github/workflows/child.yml', "on: workflow_call\njobs:\n  model:\n    steps:\n      - run: cargo test -- --ignored\n")
        workflow = {'on': ['push', 'workflow_dispatch'], 'jobs': {'caller': {
            'if': "github.event_name == 'workflow_dispatch'", 'uses': './.github/workflows/child.yml'}}}
        self.assertEqual(self.errors(workflow), [])
        del workflow['jobs']['caller']['if']
        self.assertTrue(self.errors(workflow))

    def test_unresolved_remote_reusable_workflow_requires_dispatch_gate(self):
        workflow = {'on': ['push', 'workflow_dispatch'], 'jobs': {'remote': {'uses': 'owner/repo/.github/workflows/proof.yml@frozen'}}}
        self.assertTrue(self.errors(workflow))
        workflow['jobs']['remote']['if'] = "github.event_name == 'workflow_dispatch'"
        self.assertEqual(self.errors(workflow), [])

    def test_current_profile_gates_are_checked_generically(self):
        import yaml
        workflow = yaml.safe_load((ROOT / '.github/workflows/real-weights.yml').read_text(encoding="utf-8"))
        self.assertEqual(policy.workflow_errors(workflow, 'renamed.yml', ROOT), [])


    def test_selection_barrier_dominates_existing_predicates_without_rewriting_them(self):
        workflow = {'on': {'workflow_dispatch': {'inputs': {
            'profile': {'required': True, 'default': '', 'options': ['', 'all', 'llm']},
            'binding': {'type': 'string', 'default': '{}'}}}}, 'jobs': {
            'validate': {'steps': [{'env': {'MODEL_PROFILE': '${{ inputs.profile }}', 'MODEL_SELECTION_BINDING': "${{ inputs.profile == 'all' && inputs.binding || '' }}"},
                                   'run': 'python3 scripts/check_model_inference_policy.py --validate-selection'}]},
            'model': {'needs': 'validate', 'if': "inputs.profile == 'all' || inputs.profile == 'llm'",
                      'steps': [{'run': 'cargo test -- --ignored'}]},
            'dependent': {'needs': 'model', 'if': "inputs.profile == 'all'", 'steps': [{'run': 'cargo test -- --ignored'}]}}}
        self.assertEqual(self.errors(workflow), [])
        workflow['jobs']['model']['if'] = "always() && (inputs.profile == 'all' || inputs.profile == 'llm')"
        self.assertTrue(self.errors(workflow), 'status-function bypass must not launder a failed validator')
        workflow['jobs']['model']['if'] = "inputs.profile == 'all' || inputs.profile == 'llm'"
        del workflow['jobs']['model']['needs']
        self.assertTrue(self.errors(workflow), 'an independent model root is not protected')

    def test_failed_selection_may_still_run_non_model_receipts(self):
        workflow = {'on': {'workflow_dispatch': {'inputs': {
            'profile': {'required': True, 'default': '', 'options': ['', 'all']}, 'binding': {'type': 'string', 'default': '{}'}}}}, 'jobs': {
            'validate': {'steps': [{'env': {'MODEL_PROFILE': '${{ inputs.profile }}', 'MODEL_SELECTION_BINDING': "${{ inputs.profile == 'all' && inputs.binding || '' }}"},
                                   'run': 'python3 scripts/check_model_inference_policy.py --validate-selection'}]},
            'receipt': {'needs': 'validate', 'if': "always() && inputs.profile == 'all'", 'steps': [{'run': 'echo receipt'}]}}}
        self.assertEqual(self.errors(workflow), [])
        workflow['jobs']['validate']['steps'][0]['continue-on-error'] = True
        self.assertTrue(self.errors(workflow), 'a swallowed validator failure is not a barrier')
        workflow['jobs']['validate']['steps'][0]['continue-on-error'] = False
        workflow['on']['workflow_dispatch']['inputs']['binding']['default'] = '{"confirm_all":true}'
        self.assertTrue(self.errors(workflow), 'a default cannot pre-confirm the aggregate')

    def test_all_confirmation_is_an_exact_true_only_object(self):
        for binding in ('', '{}', '{"confirm_all":false}', '{"confirm_all":1}', '{"confirm_all":"true"}',
                        '{"confirm_all":true,"extra":1}', '{"confirm_all":false,"confirm_all":true}', '[]', 'null'):
            with self.subTest(binding=binding):
                self.assertIsNotNone(policy.selection_binding_error('all', binding))
        self.assertIsNone(policy.selection_binding_error('all', '{"confirm_all":true}'))
        self.assertIsNone(policy.selection_binding_error('llm', 'not JSON'))
        self.assertIsNone(policy.selection_binding_error('ios-selected-captures', 'unchanged legacy selector'))
        self.assertIsNotNone(policy.selection_binding_error('', '{"confirm_all":true}'))

    def test_selection_cli_does_not_read_the_ios_carrier_on_other_profiles(self):
        import contextlib
        import io
        class Environment(dict):
            def get(self, key, default=None):
                if key == 'MODEL_SELECTION_BINDING':
                    raise AssertionError('non-all validation read the iOS carrier')
                return super().get(key, default)
        environment = Environment(MODEL_PROFILE='ios-selected-captures', MODEL_SELECTION_BINDING='legacy')
        with patch('sys.argv', ['policy', '--validate-selection']), patch.object(os, 'environ', environment), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(policy.main(), 0)

    def test_current_root_jobs_depend_on_validation_and_ios_binding_is_not_redefined(self):
        import yaml
        workflow = yaml.safe_load((ROOT / '.github/workflows/real-weights.yml').read_text(encoding="utf-8"))
        inputs = workflow.get('on', workflow.get(True))['workflow_dispatch']['inputs']
        self.assertEqual(len(inputs), 25)
        self.assertEqual(inputs['ios_pricing_compile_binding']['default'], '{"source_sha":"","manifest_sha256":"","host":"second-mac"}')
        validator = workflow['jobs']['validate-selection']['steps'][-1]
        self.assertEqual(validator['env']['MODEL_SELECTION_BINDING'], "${{ inputs.profile == 'all' && inputs.ios_pricing_compile_binding || '' }}")
        self.assertIn('validate-selection', policy.selection_barriers(workflow, inputs))
        for name, job in workflow['jobs'].items():
            if name != 'validate-selection':
                self.assertTrue(job.get('needs'), name)


class PrebuiltAssignmentParser(unittest.TestCase):
    def parser(self, output):
        import subprocess
        source = (ROOT / '.github/kv-poc/build.sh').read_text(encoding="utf-8")
        function = 'parse_prebuilt_env() {' + source.split('parse_prebuilt_env() {', 1)[1].split('\n}\n', 1)[0] + '\n}\n'
        # Execute ONLY the isolated assignment parser; never source/run the build workflow.
        program = function + '\nparse_prebuilt_env "$TEST_OUTPUT" || exit $?\nprintf "%s\\n%s" "$PMETAL_MLX_PREBUILT_DIR" "$PMETAL_METALLIB_PATH"\n'
        return subprocess.run(['bash', '-c', program], env={'PATH': os.environ['PATH'], 'TEST_OUTPUT': output},
                              text=True, capture_output=True, timeout=5, encoding="utf-8")

    def test_complete_contract_and_raw_spaces_are_preserved(self):
        directory = '/tmp/cache with spaces/Release'
        result = self.parser('PMETAL_MLX_PREBUILT_DIR=' + directory + '\nPMETAL_METALLIB_PATH=' + directory + '/mlx.metallib')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, directory + '\n' + directory + '/mlx.metallib')
        source = (ROOT / 'scripts/fetch-prebuilt-mlx.sh').read_text(encoding="utf-8")
        import re
        self.assertEqual(set(re.findall(r'echo "([A-Z_]+)=', source)), {'PMETAL_MLX_PREBUILT_DIR', 'PMETAL_METALLIB_PATH'})

    def test_command_substitution_is_rejected_without_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / 'must-not-exist'
            result = self.parser('PMETAL_MLX_PREBUILT_DIR=$(touch ' + str(marker) + ')\nPMETAL_METALLIB_PATH=/tmp/mlx.metallib')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('invalid prebuilt output', result.stderr)
            self.assertFalse(marker.exists())

    def test_contract_drift_and_shell_metacharacters_fail_loudly(self):
        normal = 'PMETAL_MLX_PREBUILT_DIR=/tmp/cache\nPMETAL_METALLIB_PATH=/tmp/cache/mlx.metallib'
        for output in (normal + '\nPATH=/tmp/other', normal + '\nPMETAL_MLX_PREBUILT_DIR=/duplicate',
                       'PMETAL_MLX_PREBUILT_DIR=/tmp/cache', normal.replace('/tmp/cache', '/tmp/x;echo unsafe'),
                       normal.replace('/tmp/cache', '/tmp/`echo unsafe`')):
            with self.subTest(output=output):
                result = self.parser(output)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('invalid prebuilt output', result.stderr)


class ExactProofProtocol(unittest.TestCase):
    def test_iris_cuda_physical_index_is_recipe_bound(self):
        rows = proof.load_registry()
        resident = next(item for item in rows if item['package'] == 'candle-gen-iris'
                        and item['fqn'] == 'quant_real_weights::real_tiers_shrink_the_resident_weights')
        render = next(item for item in rows if item['package'] == 'candle-gen-iris'
                      and item['fqn'] == 'quant_real_weights::real_quantized_renders_track_the_bf16_render_at_512')
        for item in (resident, render):
            cuda = proof.configured_case(item, 'source_selected_cuda')
            self.assertEqual(cuda['expected_devices'], ['candle-cuda'])
            self.assertEqual(cuda['required_cuda_physical_index'], 1)
            with self.assertRaisesRegex(ValueError, 'effective configuration differs'):
                proof.apply_configuration({**cuda, 'required_cuda_physical_index': 0}, {})
            # CI_IMPACT: registry-only recipe binding; no CUDA or model work.
            gpu0 = proof.configured_case(item, 'source_selected_cuda_gpu0')
            self.assertEqual(gpu0['expected_devices'], ['candle-cuda'])
            self.assertEqual(gpu0['cargo_features'], cuda['cargo_features'])
            self.assertEqual(gpu0['required_cuda_physical_index'], 0)
            self.assertEqual(gpu0['input_contract'], cuda['input_contract'])
            self.assertEqual(gpu0['output_contract'], cuda['output_contract'])
            proof.apply_configuration(gpu0, {})
            with self.assertRaisesRegex(ValueError, 'effective configuration differs'):
                proof.apply_configuration({**gpu0, 'required_cuda_physical_index': 1}, {})
        cpu = proof.configured_case(render, 'source_default_cpu')
        self.assertEqual(cpu['expected_devices'], ['candle-cpu'])
        self.assertNotIn('required_cuda_physical_index', cpu)
        invalid = copy.deepcopy(render)
        invalid['configurations']['source_default_cpu']['required_cuda_physical_index'] = 1
        with self.assertRaisesRegex(ValueError, 'requires a CUDA source recipe'):
            proof.validate_configurations(invalid)

    def test_direct_and_cpu_roles_are_exact_not_any_nonzero_summary(self):
        proof.verify_execution(row(), output(row()))
        child = row(execution_role='cpu_child')
        proof.verify_execution(child, output(child, count=2))
        with self.assertRaises(ValueError):
            proof.verify_execution(child, output(child))
        with self.assertRaises(ValueError):
            proof.verify_execution(row(), output(row(), count=2))

    def test_nested_ranks_require_waiting_root_and_every_result_success(self):
        case = row(execution_role='nested_rank_cpu', expected_successful_summaries=4)
        text = output(case, count=4, parent=True)
        proof.verify_execution(case, text)
        for invalid in (text.replace('MODEL_INFERENCE_PARENT_COMPLETED', 'MISSING_PARENT'),
                        text + 'test result: FAILED. 0 passed; 1 failed; 0 ignored;\n',
                        text.replace('1 passed; 0 failed; 0 ignored;', '0 passed; 0 failed; 1 ignored;', 1)):
            with self.assertRaises(ValueError):
                proof.verify_execution(case, invalid)

    def test_skip_missing_duplicate_body_never_proves_execution(self):
        case = row()
        valid = output(case)
        for text in ('test result: ok. 0 passed; 0 failed; 1 ignored;\n',
                     valid.replace('MODEL_INFERENCE_BODY_COMPLETED', 'MISSING_BODY'),
                     valid + 'MODEL_INFERENCE_BODY_COMPLETED ' + case['fqn'] + '\n'):
            with self.assertRaises(ValueError):
                proof.verify_execution(case, text)

    def test_actual_device_is_in_scope_set_not_host_or_default_context(self):
        case = row(expected_devices=['mlx-cpu', 'mlx-metal'])
        proof.verify_execution(case, output(case, devices=('mlx-cpu', 'mlx-metal')))
        with self.assertRaises(ValueError):
            proof.verify_execution(case, output(case, devices=('mlx-cpu',)))
        context_only = output(row(), devices=()) + 'MODEL_INFERENCE_DEVICE_CONTEXT tests::model_case default=mlx-metal\n'
        with self.assertRaises(ValueError):
            proof.verify_execution(row(), context_only)
        agnostic = row(backend='mlx')
        observed = proof.verify_execution(agnostic, output(agnostic, devices=('mlx-cpu', 'mlx-metal')))
        self.assertEqual(observed['actual_devices'], ['mlx-cpu', 'mlx-metal'])

    def test_discovery_is_one_exact_ignored_case(self):
        proof.verify_discovery(row(), 'tests::model_case: test\n\n1 test, 0 benchmarks\n')
        for invalid in ('0 tests, 0 benchmarks\n', 'wrong: test\n1 test, 0 benchmarks\n'):
            with self.assertRaises(ValueError):
                proof.verify_discovery(row(), invalid)

    def test_fixed_commands_preserve_declared_target_features_and_defaults(self):
        case = row(target='integration', target_kind='test', cargo_features=['metal'])
        args = proof.command(case, False)
        self.assertIn('--test', args)
        self.assertIn('integration', args)
        self.assertIn('--features', args)
        self.assertNotIn('--no-default-features', args)
        self.assertEqual(args[-4:], ['--exact', '--ignored', '--nocapture', '--test-threads=1'])
        self.assertIn('--no-default-features', proof.command(row(no_default_features=True), True))

    def test_selector_and_request_are_not_commands_or_hidden_resources(self):
        cases = (row(),)
        self.assertEqual(proof.select(cases, 'mlx-gen-test/tests::model_case', 'mlx'), cases)
        for selection in ('', '*', '$(echo bad)', 'mlx-gen-test; command'):
            with self.assertRaises(ValueError):
                proof.select(cases, selection, 'mlx')
        value = proof.request('{"selection":"all","policy":{"backend":"darwin-mlx"}}', 'mlx')
        self.assertEqual(value['selection'], 'all')
        for raw in ('{}', '{"selection":"all"}', '{"selection":"all","policy":{"backend":"windows-cuda"}}'):
            with self.assertRaises(ValueError):
                proof.request(raw, 'mlx')


class OwnedEvidence(unittest.TestCase):
    def test_output_is_owned_and_fresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            self.assertEqual(proof.owned_output(base / 'new', base), (base / 'new').resolve())
            with self.assertRaises(ValueError):
                proof.owned_output(base / 'new', base)
            with self.assertRaises(ValueError):
                proof.owned_output(base.parent / 'outside-evidence', base)

    def test_no_checkpoint_symlink_hardlink_oversize_or_external_upload(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            owned = base / 'owned'
            published = base / 'published'
            owned.mkdir()
            published.mkdir()
            regular = owned / 'image.png'
            regular.write_bytes(b'output')
            proof.publish_file(regular, published / 'image.png', owned, 100, destination_root=published)
            with self.assertRaises(ValueError):
                proof.publish_file(regular, published / 'large.png', owned, 1, destination_root=published)
            checkpoint = owned / 'model.safetensors'
            checkpoint.write_bytes(b'not-uploadable')
            with self.assertRaises(ValueError):
                proof.publish_file(checkpoint, published / 'model.safetensors', owned, 100, destination_root=published)
            external = base / 'reference.png'
            external.write_bytes(b'private-input')
            link = owned / 'link.png'
            link.symlink_to(external)
            with self.assertRaises(ValueError):
                proof.publish_file(link, published / 'link.png', owned, 100, destination_root=published)
            hard = owned / 'hard.png'
            os.link(external, hard)
            with self.assertRaises(ValueError):
                proof.publish_file(hard, published / 'hard.png', owned, 100, destination_root=published)

    def test_logs_redact_credentials_even_when_short(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'stdout.log'
            destination = root / 'published.log'
            source.write_text('credential=abc', encoding='utf-8')
            with patch.dict(os.environ, {'HF_TOKEN': 'abc'}):
                proof.publish_file(source, destination, root, 100, log=True)
            self.assertNotIn('abc', destination.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
