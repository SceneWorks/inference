"""Source/fixture-only graph tests. Every Cargo invocation is mocked; no NN execution.

Retains the predecessor's base/recipe/lane/manifest/CI cases, repaired to use
actual Cargo ownership and exact source-declared recipes rather than guessed
CPU compatibility of empty Cargo feature definitions.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import re
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import compile_model_proof_targets as compile_proof

FEATURE = 'model-inference-proof'


def command(lane='cpu', *packages):
    args = ['cargo', 'clippy', '--locked', '--keep-going']
    for package in packages or ('candle-gen*', 'candle-llm', 'runtime-cpu'):
        args += ['-p', package]
    args += ['--all-targets']
    if lane != 'cpu':
        args += ['--features', lane]
    return args + ['--', '-D', 'warnings']


def feature_set(args):
    boundary = args.index('--') if '--' in args else len(args)
    values = set()
    for i, arg in enumerate(args[:boundary]):
        if arg == '--features':
            values.update(args[i + 1].replace(',', ' ').split())
    return values


def row(package='candle-llm', features=None, lane='cpu'):
    return {'package': package, 'source': f'crates/{package}/src/lib.rs',
            'backend': 'candle-' + lane, 'expected_devices': ['candle-' + lane],
            'cargo_features': [FEATURE] if features is None else features}


def configured(package='candle-llm', recipes=None, kind='feature-only', default=None):
    value = row(package, [])
    value.update(backend=None, expected_devices=[], configuration_kind=kind,
                 configuration_controls=[] if kind == 'feature-only' else ['TEST_DEVICE'],
                 default_configuration=default, configurations={})
    for name, (features, lane, execution_lane) in (recipes or {}).items():
        recipe = {'source_identity': value['source'] + '::tests::proof/' + name,
                  'backend': 'candle-' + lane, 'expected_devices': ['candle-' + lane],
                  'cargo_features': features, 'scope': 'complete source body',
                  'required_work': ['complete source body'],
                  'environment': {} if kind == 'feature-only' else {'TEST_DEVICE': None}}
        if execution_lane:
            recipe['execution_lane'] = 'candle-' + execution_lane
        value['configurations'][name] = recipe
    if default:
        recipe = value['configurations'][default]
        for key in ('backend', 'expected_devices', 'cargo_features'):
            value[key] = recipe[key]
    return value


class Fixture:
    """Private source + metadata fixture with real Cargo manifests, never Cargo execution."""
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.compile-proof-fixture-', dir=ROOT)
        self.root = Path(self.temp.name)
        self.packages = {}
        for name in ('candle-gen', 'candle-gen-example', 'candle-llm', 'not-selected'):
            self.add(name, {FEATURE: [], 'cuda': [], 'metal': [], 'sc9581-ab': [],
                            'test-kv-arm': [], 'flash-attn': ['cuda'],
                            'real-weight-diagnostics': [], 'testkit': []})
        self.add('runtime-cpu', {})
        self.add('runtime-cuda', {'cuda': [], 'flash-attn': ['candle-llm/flash-attn']})

    def add(self, name, features, manifest=None):
        manifest = self.root / (manifest or f'crates/{name}/Cargo.toml')
        manifest.parent.mkdir(parents=True, exist_ok=True)
        (manifest.parent / 'src').mkdir(exist_ok=True)
        (manifest.parent / 'src/lib.rs').write_text('// source fixture\n', encoding="utf-8")
        manifest.write_text('[package]\nname = ' + json.dumps(name)
                            + '\nversion = "0.0.0"\n[features]\n'
                            + ''.join(key + ' = ' + json.dumps(value) + '\n'
                                      for key, value in features.items()), encoding="utf-8")
        self.packages[name] = {'id': name, 'name': name, 'features': features,
                               'manifest_path': str(manifest)}
        return manifest

    def metadata(self):
        return {'packages': list(self.packages.values()), 'workspace_members': list(self.packages)}

    def close(self):
        self.temp.cleanup()


class CompileProofTargets(unittest.TestCase):
    def setUp(self):
        self.fixture = Fixture()
        self.addCleanup(self.fixture.close)

    def graphs(self, rows, source=None, lane='cpu', metadata=None):
        return compile_proof.proof_commands(source or command(lane),
                                            metadata or self.fixture.metadata(), tuple(rows),
                                            host_lane=lane, root=self.fixture.root)

    def extras(self, graphs):
        base = feature_set(graphs[0])
        return [feature_set(graph) - base for graph in graphs[1:]]

    def test_qualified_features_follow_metadata_and_exact_existing_package_selection(self):
        original = command('metal')
        actual = compile_proof.proof_command(original, self.fixture.metadata())
        boundary = original.index('--')
        self.assertEqual(actual[:boundary], original[:boundary])
        self.assertEqual(actual[boundary + 2:], original[boundary:])
        self.assertEqual(feature_set(actual) - {'metal'},
                         {p + '/' + FEATURE for p in ('candle-gen', 'candle-gen-example', 'candle-llm')})
        self.assertNotIn('not-selected/' + FEATURE, feature_set(actual))
        self.fixture.packages['candle-gen-example']['features'].pop(FEATURE)
        actual = compile_proof.proof_command(original, self.fixture.metadata())
        self.assertNotIn('candle-gen-example/' + FEATURE, feature_set(actual))

    def test_base_pass_does_not_depend_on_registered_rows_or_runnable_defaults(self):
        original = command()
        self.assertEqual(self.graphs([]), [compile_proof.proof_command(original, self.fixture.metadata())])
        null = configured(recipes={'gpu': ([FEATURE, 'cuda'], 'cuda', None)})
        self.assertEqual(self.graphs([null]), self.graphs([]))

    def test_check_and_package_selector_forms_preserve_cpu_or_cuda_graph(self):
        for selector in (['--package=candle-gen*'], ['-pcandle-gen*'], ['--package', 'candle-gen*']):
            source = ['cargo', 'check', '--all-targets', *selector, '--features', 'cuda']
            graph = self.graphs([], source, 'cuda')[0]
            self.assertEqual(graph[:len(source)], source)
            self.assertNotIn('metal', feature_set(graph))

    def test_unsupported_execution_or_unbounded_graph_is_rejected(self):
        for source in (['cargo', 'test', '--all-targets', '-p', 'candle-llm'],
                       ['cargo', 'run', '-p', 'candle-llm'],
                       ['cargo', 'clippy', '-p', 'candle-llm'],
                       ['cargo', 'check', '--all-targets', '--workspace'],
                       ['cargo', 'check', '--all-targets', '--all-features', '-p', 'candle-llm'],
                       ['cargo', 'check', '--all-targets', '-p'],
                       ['cargo', 'check', '--all-targets', '-p', 'candle-llm', '--features']):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.graphs([], source)

    def test_missing_selected_package_or_hooks_cannot_report_coverage(self):
        for package in ('missing*', 'runtime-cpu'):
            with self.assertRaises(ValueError):
                self.graphs([], command('cpu', package))

    def test_nested_execution_lane_recipe_only_and_separate_from_actual_device(self):
        value = configured(recipes={'cpu-on-cuda': ([FEATURE, 'cuda', 'testkit'], 'cpu', 'cuda')})
        cuda = self.graphs([value], command('cuda'), 'cuda')
        self.assertEqual(self.extras(cuda), [{'candle-llm/cuda', 'candle-llm/testkit'}])
        self.assertEqual(len(self.graphs([value])), 1)
        value['execution_lane'] = 'candle-cuda'
        with self.assertRaisesRegex(ValueError, 'recipe-only'):
            self.graphs([value])

    def test_null_source_default_does_not_invent_a_device_or_drop_nested_graphs(self):
        value = configured(recipes={'cpu': ([FEATURE, 'test-kv-arm'], 'cpu', None)})
        self.assertEqual(self.extras(self.graphs([value])), [{'candle-llm/test-kv-arm'}])
        self.assertIsNone(value['backend'])

    def test_root_and_all_nested_configurations_are_traversed(self):
        value = configured(recipes={'base': ([FEATURE], 'cpu', None),
                                     'kv': ([FEATURE, 'test-kv-arm'], 'cpu', None),
                                     'ab': ([FEATURE, 'sc9581-ab'], 'cpu', None)}, default='base')
        self.assertEqual(self.extras(self.graphs([value])),
                         [{'candle-llm/sc9581-ab'}, {'candle-llm/test-kv-arm'}])
        root = row('candle-gen', [FEATURE, 'real-weight-diagnostics'])
        self.assertEqual(len(self.graphs([root, value])), 4)

    def test_source_controls_recipes_receive_the_same_nested_coverage(self):
        value = configured(kind='source-controls',
                           recipes={'cpu': ([FEATURE, 'test-kv-arm'], 'cpu', None),
                                    'cuda': ([FEATURE, 'cuda', 'flash-attn'], 'cuda', None)})
        self.assertEqual(self.extras(self.graphs([value])), [{'candle-llm/test-kv-arm'}])
        self.assertIn('candle-llm/flash-attn', feature_set(self.graphs([value], command('cuda'), 'cuda')[-1]))

    def test_every_recipe_feature_mechanism_dedupes_per_set_not_per_case(self):
        rows = [row('candle-gen-example', [FEATURE, 'sc9581-ab']),
                row('candle-llm', [FEATURE, 'test-kv-arm']),
                row('candle-gen', [FEATURE, 'real-weight-diagnostics'])]
        graphs = self.graphs(rows * 7)
        self.assertEqual(len(graphs), 4)
        self.assertCountEqual(self.extras(graphs), [{'candle-gen-example/sc9581-ab'},
                                                  {'candle-llm/test-kv-arm'},
                                                  {'candle-gen/real-weight-diagnostics'}])
        cuda_rows = [row('candle-llm', [FEATURE, 'cuda', 'flash-attn'], 'cuda'),
                     row('candle-gen', [FEATURE, 'cuda', 'testkit'], 'cuda')]
        cuda = self.graphs(cuda_rows * 3, command('cuda'), 'cuda')
        self.assertEqual(len(cuda), 3)
        self.assertNotIn('candle-llm/flash-attn', feature_set(cuda[0]))
        self.assertFalse(any({'candle-gen/testkit', 'candle-llm/flash-attn'} <= feature_set(g) for g in cuda))

    def test_equal_mechanisms_share_a_graph_without_unioning_different_recipes(self):
        rows = [row('candle-gen', [FEATURE, 'sc9581-ab']),
                row('candle-gen-example', [FEATURE, 'sc9581-ab']),
                row('candle-gen', [FEATURE, 'test-kv-arm'])]
        self.assertEqual(len(self.graphs(rows)), 3)
        self.assertIn({'candle-gen/sc9581-ab', 'candle-gen-example/sc9581-ab'}, self.extras(self.graphs(rows)))

    def test_metal_and_cuda_graphs_never_mix(self):
        rows = [row('candle-gen', [FEATURE, 'cuda', 'testkit'], 'cuda'),
                row('candle-gen-example', [FEATURE, 'metal', 'real-weight-diagnostics'], 'metal')]
        for lane, forbidden in (('cuda', 'metal'), ('metal', 'cuda')):
            for graph in self.graphs(rows, command(lane), lane):
                self.assertFalse(any(f.endswith('/' + forbidden) or f == forbidden for f in feature_set(graph)))
        with self.assertRaisesRegex(ValueError, 'contradict'):
            self.graphs([row(features=[FEATURE, 'cuda', 'metal'], lane='cuda')], command('cuda'), 'cuda')

    def test_source_host_conflict_is_diagnosed_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, 'contradict'):
            self.graphs([row(features=[FEATURE, 'flash-attn'])])
        value = configured(recipes={'bad': ([FEATURE, 'cuda'], 'cpu', None)})
        with self.assertRaisesRegex(ValueError, 'execution lane'):
            self.graphs([value])

    def test_empty_feature_definition_is_not_permission_to_enable_it(self):
        features = dict(self.fixture.packages['candle-llm']['features'], **{'unused-empty': []})
        self.fixture.add('candle-llm', features)
        self.assertFalse(any('unused-empty' in f for g in self.graphs([row()]) for f in feature_set(g)))
        value = row(features=[FEATURE, 'sc9581-ab'], lane='metal')
        self.assertEqual(len(self.graphs([value])), 1)
        self.assertEqual(len(self.graphs([value], command('metal'), 'metal')), 2)

    def test_unrequested_runtime_package_is_never_injected(self):
        rows = [row(features=[FEATURE, 'cuda', 'flash-attn'], lane='cuda')]
        for lane in ('cpu', 'cuda'):
            graphs = self.graphs(rows, command(lane, 'candle-llm'), lane)
            self.assertFalse(any(f.startswith('runtime-cuda/') for g in graphs for f in feature_set(g)))
        self.assertEqual(len(self.graphs(rows)), 1)

    def test_configuration_manifest_is_the_actual_selected_source_manifest(self):
        value = row(features=[FEATURE, 'test-kv-arm'])
        # Shared-source consumer: source is NOT under the selected consumer manifest.
        value['source'] = 'crates/candle-gen/src/lib.rs'
        value['configuration_manifest'] = 'crates/candle-llm/Cargo.toml'
        self.assertEqual(len(self.graphs([value])), 2)

    def test_manifest_path_traversal_json_missing_and_wrong_package_fail_closed(self):
        for manifest in ('../Cargo.toml', str(self.fixture.root / 'Cargo.toml'),
                         'config.json', 'missing/Cargo.toml', 'crates/candle-gen/Cargo.toml'):
            value = row()
            value['configuration_manifest'] = manifest
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                self.graphs([value])

    def test_manifest_same_name_but_wrong_metadata_owner_fails_closed(self):
        other = self.fixture.root / 'duplicate/Cargo.toml'
        other.parent.mkdir()
        other.write_text(Path(self.fixture.packages['candle-llm']['manifest_path']).read_text(encoding="utf-8"), encoding="utf-8")
        value = row()
        value['configuration_manifest'] = 'duplicate/Cargo.toml'
        with self.assertRaisesRegex(ValueError, 'ownership'):
            self.graphs([value])

    def test_source_traversal_and_missing_source_fail_closed(self):
        for source in ('../source.rs', '/source.rs', 'missing.rs'):
            value = row()
            value['source'] = source
            with self.assertRaisesRegex(ValueError, 'source'):
                self.graphs([value])

    def test_manifest_feature_missing_from_source_manifest_fails_closed(self):
        for value in (row(features=[FEATURE, 'unknown']),
                      configured(recipes={'bad': ([FEATURE, 'unknown'], 'cpu', None)})):
            with self.assertRaisesRegex(ValueError, 'absent'):
                self.graphs([value])

    def test_metadata_source_declaration_mismatch_fails_closed(self):
        metadata = copy.deepcopy(self.fixture.metadata())
        next(p for p in metadata['packages'] if p['name'] == 'candle-llm')['features']['not-in-source'] = []
        with self.assertRaisesRegex(ValueError, 'differ'):
            self.graphs([row()], metadata=metadata)

    def test_filter_selected_packages_before_validating_unrelated_separate_workspace(self):
        foreign = {'package': 'ios-smoke', 'source': '../not-in-this-workspace',
                   'configuration_manifest': 'sdk/Cargo.toml', 'cargo_features': ['unknown']}
        self.assertEqual(self.graphs([foreign, row()]), self.graphs([row()]))
        self.fixture.add('ios-smoke', {FEATURE: []}, 'sdk/Cargo.toml')
        foreign.update(source='sdk/src/lib.rs', configuration_manifest='sdk/Cargo.toml',
                       cargo_features=[FEATURE], backend='mlx')
        graphs = self.graphs([foreign], command('metal', 'ios-smoke'), 'metal')
        self.assertEqual(len(graphs), 1)
        self.assertIn('ios-smoke/' + FEATURE, feature_set(graphs[0]))

    def test_selected_separate_workspace_manifest_is_forwarded_to_actual_metadata(self):
        self.fixture.add('ios-smoke', {FEATURE: []}, 'sdk/Cargo.toml')
        for flag in (['--manifest-path', 'sdk/Cargo.toml'], ['--manifest-path=sdk/Cargo.toml']):
            source = command('metal', 'ios-smoke')
            source[2:2] = flag
            metadata_command = compile_proof._metadata_command(source, self.fixture.root)
            self.assertEqual(metadata_command[-2:], ['--manifest-path', 'sdk/Cargo.toml'])
            self.assertIn('--no-deps', metadata_command)
        source = command()
        source[2:2] = ['--manifest-path', '../Cargo.toml']
        with self.assertRaisesRegex(ValueError, 'actual Cargo.toml'):
            compile_proof._metadata_command(source, self.fixture.root)

    def test_manifest_symlink_cannot_escape_or_substitute_metadata_owner(self):
        target = self.fixture.root / 'crates/candle-llm/Cargo.toml'
        link = self.fixture.root / 'linked/Cargo.toml'
        link.parent.mkdir()
        link.symlink_to(target)
        value = row()
        value['configuration_manifest'] = 'linked/Cargo.toml'
        self.assertEqual(len(self.graphs([value])), 1)
        # A same-checkout alias is the same actual owner; a different owner is not.
        link.unlink()
        link.symlink_to(self.fixture.root / 'crates/candle-gen/Cargo.toml')
        with self.assertRaisesRegex(ValueError, 'declared selected Cargo package'):
            self.graphs([value])

    def test_original_caller_flags_all_targets_and_base_hooks_are_preserved_in_every_graph(self):
        source = command('cpu')
        source[2:2] = ['--features', 'candle-gen/testkit']
        value = row(features=[FEATURE, 'test-kv-arm'])
        graphs = self.graphs([value], source)
        boundary = source.index('--')
        for graph in graphs:
            self.assertEqual(graph[:boundary], source[:boundary])
            self.assertEqual(graph[boundary + 2:], source[boundary:])
            self.assertIn('candle-gen/testkit', feature_set(graph))
            self.assertTrue({'candle-gen/' + FEATURE, 'candle-gen-example/' + FEATURE,
                             'candle-llm/' + FEATURE} <= feature_set(graph))
            self.assertNotIn('--tests', graph)
            self.assertNotIn('--workspace', graph)
            self.assertNotIn('--all-features', graph)

    def test_no_default_features_recipe_is_explicit_not_silently_built_with_defaults(self):
        value = row(features=[FEATURE, 'test-kv-arm'])
        value['no_default_features'] = True
        with self.assertRaisesRegex(ValueError, 'requires --no-default-features'):
            self.graphs([value])
        source = command()
        source.insert(2, '--no-default-features')
        self.assertEqual(len(self.graphs([value], source)), 2)

    def test_unknown_feature_is_not_hidden_by_other_host_filter(self):
        with self.assertRaisesRegex(ValueError, 'absent'):
            self.graphs([row(features=[FEATURE, 'cuda', 'unknown'], lane='cuda')])

    def test_caller_feature_forms_and_default_forwarding_detect_backend_conflicts(self):
        for flag in (['--features', 'candle-llm/cuda metal'], ['--features=cuda,metal'], ['-Fcuda,metal']):
            source = ['cargo', 'check', '-p', 'candle-llm', '--all-targets', *flag]
            with self.assertRaisesRegex(ValueError, 'lane'):
                self.graphs([], source, 'cuda')
        self.fixture.packages['candle-llm']['features']['default'] = ['metal']
        with self.assertRaisesRegex(ValueError, 'lane'):
            self.graphs([], command('cuda'), 'cuda')

    def test_qualified_backend_already_in_caller_dedupes_base_recipe(self):
        source = command('cuda')
        value = row(features=[FEATURE, 'cuda'], lane='cuda')
        self.assertEqual(len(self.graphs([value], source, 'cuda')), 1)

    def test_unknown_lane_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unknown compile-host'):
            self.graphs([], lane='gpu')

    def test_missing_or_bad_registry_fails_closed_before_metadata(self):
        for content in (None, '[]', '{}', 'invalid JSON'):
            registry = self.fixture.root / 'registry.json'
            if content is not None:
                registry.write_text(content, encoding="utf-8")
            with patch.object(compile_proof.subprocess, 'run') as runner, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    compile_proof.main(['--registry', str(registry), '--', *command()])
                runner.assert_not_called()

    def test_main_runs_metadata_then_each_graph_compiler_with_failure_preserved(self):
        registry = self.fixture.root / 'registry.json'
        registry.write_text(json.dumps([row(features=[FEATURE, 'test-kv-arm'])]), encoding="utf-8")
        calls = []
        def execute(argv, **kwargs):
            calls.append(argv)
            self.assertEqual(kwargs['cwd'], self.fixture.root)
            if 'metadata' in argv:
                self.assertIn('--no-deps', argv)
                return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(self.fixture.metadata()))
            return subprocess.CompletedProcess(argv, 42 if len(calls) == 2 else 0)
        with patch.object(compile_proof, 'ROOT', self.fixture.root), \
                patch.object(compile_proof.subprocess, 'run', side_effect=execute), contextlib.redirect_stdout(io.StringIO()):
            result = compile_proof.main(['--registry', str(registry), '--lane', 'cpu', '--', *command()])
        self.assertEqual(result, 42)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(c[:2] == ['cargo', 'clippy'] and '--all-targets' in c for c in calls[1:]))

    def test_execution_command_is_rejected_before_cargo_metadata(self):
        with patch.object(compile_proof.subprocess, 'run') as execute, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                compile_proof.main(['--', 'cargo', 'test', '--all-targets'])
            execute.assert_not_called()

    def test_cpu_metal_cuda_keep_default_lint_and_add_compile_only_hooks(self):
        workflow = (ROOT / '.github/workflows/ci.yml').read_text(encoding="utf-8")
        # Inspect exact source blocks: no YAML dependency or workflow execution.
        def steps_for(job):
            block = re.search(r'^  ' + re.escape(job) + r':\n(.*?)(?=^  [a-zA-Z][\w-]*:|\Z)',
                              workflow, re.M | re.S).group(1)
            result = []
            for match in re.finditer(r'^      - name: (.*?)\n(.*?)(?=^      - |\Z)', block, re.M | re.S):
                body = match.group(2)
                run = body.split('        run:', 1)[-1]
                condition = body.split('        if:', 1)[-1].split('        run:', 1)[0]
                result.append({'name': match.group(1), 'run': run, 'if': condition})
            return result
        for job_name, backend in (('candle-cpu-lint', 'CPU'), ('macos-metal', 'Metal'),
                                  ('windows-cuda-check', 'CUDA')):
            steps = steps_for(job_name)
            default_name = {'CPU': 'Clippy Candle packages', 'Metal': 'Clippy Candle Metal packages',
                            'CUDA': 'Clippy Candle CUDA packages (no GPU required)'}[backend]
            default = next(s for s in steps if s.get('name') == default_name)
            proof = next(s for s in steps if s.get('name') ==
                         f'Compile and lint Candle {backend} model-proof hooks (no model execution)')
            self.assertLess(steps.index(default), steps.index(proof))
            self.assertNotIn('compile_model_proof_targets.py', default['run'])
            self.assertIn(f'compile_model_proof_targets.py --lane {backend.lower()} -- cargo clippy --locked', proof['run'])
            self.assertIn('--all-targets', proof['run'])
            self.assertIn('-- -D warnings', proof['run'])
            self.assertNotIn('--all-features', proof['run'])
            if backend == 'Metal':
                self.assertEqual(proof['if'], default['if'])
                self.assertIn('run_ci_collection.py --', proof['run'])
            if backend == 'CUDA':
                self.assertIn('"%REVIEWED_PYTHON%" -I', proof['run'])
                self.assertIn('call "%VCVARS%"', proof['run'])
        # MLX has no proof feature in current metadata, but retains all-targets lint.
        mlx = next(s for s in steps_for('macos-metal') if s.get('name') == 'Clippy MLX packages')
        self.assertIn('--all-targets', mlx['run'])
        self.assertIn("-p 'mlx-gen-*'", mlx['run'])


if __name__ == '__main__':
    unittest.main()
