"""Source-only routing contracts; never dispatch, probe a device or execute a model."""
from pathlib import Path
import os
import copy
import posixpath
import shlex
import re
import subprocess
import tempfile
import unittest
import yaml
from scripts.ci.real_weights_workflow import inline_text as real_weights_inline_text

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / '.github/workflows'


def document(name):
    source = (WORKFLOWS / name).read_text(encoding='utf-8')
    if name == 'real-weights.yml':
        source = real_weights_inline_text(source, ROOT)
    value = yaml.safe_load(source)
    if name == 'model-inference-proofs.yml':
        for step in value['jobs']['prove']['steps']:
            command = step.get('run', '')
            if command.startswith(('. ./scripts/ci/real-weights/model-proof-execute/',
                                   'source scripts/ci/real-weights/model-proof-execute/')):
                path = command.split(' ', 1)[1].removeprefix('./')
                step['run'] = (ROOT / path).read_text(encoding='utf-8')
    return value


def yue_metal_launcher_errors(job, expected_cwd, wait_seconds, names):
    """Resolve the controller shell exactly as Actions does from each step's cwd."""
    errors = []
    seen = set()
    checkouts = {step.get('with', {}).get('path'): step.get('with', {})
                 for step in job['steps'] if 'checkout@' in step.get('uses', '')}
    if not {'control', expected_cwd}.issubset(checkouts):
        errors.append('missing controller/engine/app checkout paths')
    if checkouts.get('engine', {}).get('ref') != '${{ inputs.expected_engine_sha }}':
        errors.append('engine checkout must retain its exact frozen source ref')
    if expected_cwd == 'app' and checkouts.get('app', {}).get('ref') != '${{ inputs.expected_app_sha }}':
        errors.append('app checkout must retain its exact frozen source ref')
    for step in job['steps']:
        if step.get('name') not in names:
            continue
        seen.add(step['name'])
        cwd = step.get('working-directory', job.get('defaults', {}).get('run', {}).get('working-directory', '.'))
        command = shlex.split(step.get('shell', ''))
        expected = ['python3.12', '-I', '../control/scripts/ci/inference_metal_launcher.py',
                    '--wait-seconds', str(wait_seconds), '--', '/bin/bash',
                    '--noprofile', '--norc', '-eo', 'pipefail', '{0}']
        if command != expected or '${{' in step.get('shell', ''):
            errors.append('launcher, admission budget or isolated Bash flags changed')
        if cwd != expected_cwd or len(command) < 3 or posixpath.normpath(posixpath.join(cwd, command[2])) != 'control/scripts/ci/inference_metal_launcher.py':
            errors.append('launcher does not resolve from the frozen working directory')
    if seen != names:
        errors.append('missing selected Metal proof step')
    return errors


class ModelWorkflowRoutes(unittest.TestCase):
    def test_new_endpoint_is_manual_only_and_existing_endpoint_bridges_exact_feature_ref(self):
        proof = document('model-inference-proofs.yml')
        events = proof.get('on', proof.get(True))
        self.assertEqual(set(events), {'workflow_dispatch'})
        self.assertEqual(set(events['workflow_dispatch']['inputs']), {'backend', 'request'})
        self.assertNotIn('concurrency', proof)
        bridge = document('real-weights.yml')
        inputs = bridge.get('on', bridge.get(True))['workflow_dispatch']['inputs']
        self.assertLessEqual(len(inputs), 25)
        self.assertIn('model-inference-proofs', inputs['profile']['options'])
        plan = bridge['jobs']['model-proof-plan']
        self.assertEqual(plan['if'], "github.event_name == 'workflow_dispatch' && inputs.profile == 'model-inference-proofs'")
        command = next(step for step in plan['steps'] if step.get('id') == 'plan')
        self.assertEqual(command['env']['MODEL_PROOF_BINDING'], '${{ inputs.ios_pricing_compile_binding }}')
        self.assertIn('--binding-env MODEL_PROOF_BINDING', command['run'])
        self.assertNotIn('fromJSON(inputs.ios_pricing_compile_binding)', command['run'])
        self.assertIn('github.run_id', bridge['concurrency']['group'])
        self.assertIn('github.run_attempt', bridge['concurrency']['group'])

    def test_owned_paths_and_runner_context_are_step_not_job_environment(self):
        for filename, name in (('model-inference-proofs.yml', 'prove'), ('real-weights.yml', 'model-proof-execute')):
            with self.subTest(workflow=filename):
                job = document(filename)['jobs'][name]
                self.assertNotIn('MODEL_PROOF_EVIDENCE', job['env'])
                self.assertNotIn('MODEL_PROOF_RUNNER_ENVIRONMENT', job['env'])
                setup = next(step for step in job['steps'] if step.get('name') == 'Set owned proof paths and actual runner environment')
                self.assertIn('${{ runner.temp }}', setup['env']['OWNED_PROOF_EVIDENCE'])
                self.assertEqual(setup['env']['ACTUAL_RUNNER_ENVIRONMENT'], '${{ runner.environment }}')
                validator = next(step for step in job['steps'] if step.get('id') == 'evidence')
                upload = job['steps'][-1]
                self.assertIn('--verify-artifacts', validator['run'])
                self.assertIn("steps.evidence.outcome == 'success'", upload['if'])

    def test_proof_endpoints_select_rustup_before_reporting_or_building(self):
        selected = []
        bridge_raw = yaml.safe_load((WORKFLOWS / 'real-weights.yml').read_text(encoding='utf-8'))
        dedicated_raw = yaml.safe_load((WORKFLOWS / 'model-inference-proofs.yml').read_text(encoding='utf-8'))
        for name in ('Select real pinned Rust toolchain on Windows',
                     'Report actual host and pinned toolchain (no model execution)'):
            bridge_step = next(step for step in bridge_raw['jobs']['model-proof-execute']['steps']
                               if step.get('name') == name)
            dedicated_step = next(step for step in dedicated_raw['jobs']['prove']['steps']
                                  if step.get('name') == name)
            self.assertEqual(bridge_step['run'], dedicated_step['run'])
        for filename, name in (('model-inference-proofs.yml', 'prove'), ('real-weights.yml', 'model-proof-execute')):
            steps = document(filename)['jobs'][name]['steps']
            posix = next(step for step in steps if step.get('name') == 'Select Rustup proxies on Linux and macOS')
            windows = next(step for step in steps if step.get('name') == 'Select real pinned Rust toolchain on Windows')
            report = next(step for step in steps if step.get('name') == 'Report actual host and pinned toolchain (no model execution)')
            self.assertEqual(posix['if'], "runner.os != 'Windows'")
            self.assertEqual(windows['if'], "runner.os == 'Windows'")
            self.assertEqual(windows['shell'], 'powershell')
            self.assertIn('rust-toolchain.toml', windows['run'])
            self.assertIn("@('cargo', 'rustc')", windows['run'])
            self.assertIn('MODEL_PROOF_RUST_BIN=', windows['run'])
            self.assertIn('$env:GITHUB_PATH', windows['run'])
            self.assertIn('$proofRustupHome', windows['run'])
            self.assertIsNone(re.search(r'(?im)^\s*\$(?:home|env:home)\s*=', windows['run']))
            self.assertNotIn('CARGO_HOME', windows['run'])
            self.assertLess(steps.index(posix), steps.index(report))
            self.assertLess(steps.index(windows), steps.index(report))
            self.assertIn('cargo --version --verbose', report['run'])
            self.assertIn('rustc -vV', report['run'])
            self.assertIn('command -v cargo', report['run'])
            self.assertIn('command -v rustc', report['run'])
            selected.append((posix, windows))
        self.assertEqual(selected[0], selected[1])

    def test_bridge_preserves_iris_asset_setup_from_validated_plan(self):
        bridge = document('real-weights.yml')['jobs']
        dedicated = document('model-inference-proofs.yml')['jobs']['prove']
        plan = next(step for step in bridge['model-proof-plan']['steps'] if step.get('id') == 'plan')
        self.assertIn('"asset_setup":p["asset_setup"]', plan['run'])
        steps = bridge['model-proof-execute']['steps']
        for platform, backend in (('macOS', 'mlx'), ('Windows', 'candle-cuda')):
            step = next(step for step in steps if step.get('name') ==
                        f'Verify pinned Iris quant snapshots on {"the secondary Mac" if platform == "macOS" else "Windows"}')
            self.assertIn("matrix.asset_setup == 'iris-quant'", step['if'])
            self.assertIn(f'--backend {backend}', step['run'])
            self.assertEqual(step, next(item for item in dedicated['steps'] if item.get('name') == step['name']))
            self.assertLess(steps.index(step), next(i for i, item in enumerate(steps)
                              if item.get('name', '').startswith('Run exact registered model tests')))

    @unittest.skipUnless(os.name == 'posix', 'POSIX shell mock; Windows selector is source-checked')
    def test_rustup_selection_precedes_shadow_tools_and_rejects_missing_proxy(self):
        steps = document('model-inference-proofs.yml')['jobs']['prove']['steps']
        command = next(step['run'] for step in steps if step.get('name') == 'Select Rustup proxies on Linux and macOS')
        for custom_home, missing_proxy in ((False, False), (True, False), (True, True)):
            with self.subTest(custom_home=custom_home, missing_proxy=missing_proxy), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                home = root / 'home'
                cargo_home = root / 'custom cargo' if custom_home else home / '.cargo'
                rustup_bin = cargo_home / 'bin'
                shadow = root / 'shadow'
                rustup_bin.mkdir(parents=True)
                shadow.mkdir()
                for tool in ('cargo', 'rustc', 'rustup'):
                    (shadow / tool).write_text('#!/bin/sh\nprintf shadow\\n\n', encoding='utf-8')
                    (shadow / tool).chmod(0o755)
                    if missing_proxy and tool == 'rustc':
                        continue
                    (rustup_bin / tool).write_text('#!/bin/sh\nprintf rustup\\n\n', encoding='utf-8')
                    (rustup_bin / tool).chmod(0o755)
                destination = root / 'github-path'
                env = dict(os.environ, HOME=str(home), PATH=str(shadow) + os.pathsep + os.environ['PATH'], GITHUB_PATH=str(destination))
                env.pop('CARGO_HOME', None)
                if custom_home:
                    env['CARGO_HOME'] = str(cargo_home)
                result = subprocess.run(['/bin/bash', '-eo', 'pipefail', '-c', command], env=env, capture_output=True, text=True, encoding="utf-8")
                if missing_proxy:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(destination.exists())
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(destination.read_text(encoding='utf-8'), str(rustup_bin) + '\n')
                    env['PATH'] = str(rustup_bin) + os.pathsep + env['PATH']
                    resolved = subprocess.run(['/bin/bash', '-c', 'command -v cargo; command -v rustc; command -v rustup'], env=env, capture_output=True, text=True, check=True, encoding="utf-8")
                    self.assertEqual(resolved.stdout.splitlines(), [str(rustup_bin / tool) for tool in ('cargo', 'rustc', 'rustup')])

    def test_kv_gpu_boundaries_retain_frozen_targets_and_include_controller_lease(self):
        workflow = document('kv-poc-campaign.yml')
        seen = set()
        for name, job in workflow['jobs'].items():
            for step in job.get('steps', []):
                command = step.get('run', '')
                if ('.github/kv-poc/phase.sh run ' not in command
                        and command.strip() not in ('bash .github/kv-poc/probe.sh', 'bash .github/kv-poc/build.sh models',
                                                    'bash .github/kv-poc/build.sh w2-models', 'bash .github/kv-poc/build.sh w2-inputs')):
                    continue
                seen.add(name)
                self.assertIn('python3.12 -I', step['shell'])
                self.assertIn('scripts/ci/inference_metal_launcher.py', step['shell'])
                self.assertIn('--wait-seconds', step['shell'])
                self.assertIn('{0}', step['shell'])
                checkout = next(item for item in job['steps'] if 'checkout@' in item.get('uses', ''))
                self.assertEqual(set(checkout['with']['sparse-checkout'].splitlines()), {'.github', 'scripts/ci'})
        self.assertTrue(seen)

    def test_yue_precision_uses_controller_helper_without_changing_frozen_engine_workdir(self):
        cases = (('yue2-precision-proof.yml', 'engine', 14400,
                  {'Run exactly one Metal precision test with external sampling'}),
                 ('yue2-app-precision-profile.yml', 'app', 28800,
                  {'Install pinned BF16 model and both decoders through the app, without acceptance cases',
                   'Capture exactly seven Metal app precision profiles'}))
        for filename, cwd, budget, names in cases:
            job = document(filename)['jobs']['metal']
            self.assertEqual(yue_metal_launcher_errors(job, cwd, budget, names), [])
            for field, value in (('shell', 'bash'),
                                 ('shell', 'python3.12 -I "${{ github.workspace }}/control/scripts/ci/inference_metal_launcher.py" --wait-seconds 14400 -- /bin/bash -e {0}'),
                                 ('working-directory', 'control')):
                mutated = copy.deepcopy(job)
                step = next(step for step in mutated['steps'] if step.get('name') in names)
                step[field] = value
                self.assertTrue(yue_metal_launcher_errors(mutated, cwd, budget, names))
            for before, after in (('../control/', './control/'), ('-I ', ''),
                                  (str(budget), '1'), ('--noprofile ', '')):
                mutated = copy.deepcopy(job)
                step = next(step for step in mutated['steps'] if step.get('name') in names)
                step['shell'] = step['shell'].replace(before, after)
                self.assertTrue(yue_metal_launcher_errors(mutated, cwd, budget, names))
        ci = document('ci.yml')['jobs']['macos-nax']
        self.assertIn('inference_metal_launcher.py', ci['defaults']['run']['shell'])
        self.assertTrue(ci['defaults']['run']['shell'].endswith('/bin/bash -e {0}'))


class RawCudaProofAggregation(unittest.TestCase):
    PROOFS = {
        'raw_streaming': 'training_attention::tests::streaming_forward_backward_matches_composable',
        'raw_reference': 'reference_training_attention::tests::dense_reference_forward_backward_matches_composable',
        'raw_rms': 'train::norm::ownership_tests::shared_rms_contracts_match_composable',
        'raw_rope': 'train::krea_ops::tests::rms_rope_forward_backward_matches_composable',
        'raw_graph': 'train::norm::ownership_tests::graph_retaining_norm_rope_attention_fallbacks',
        'raw_half': 'train::projection::tests::half_precision_keeps_output_scale_rounding',
    }

    def assert_contract(self, job):
        steps = {step.get('id'): step for step in job['steps'] if step.get('id')}
        gate = steps['raw_proofs_gate']
        self.assertEqual(gate['if'], 'always()')
        self.assertFalse(gate.get('continue-on-error', False))
        self.assertEqual(gate['shell'], 'bash')
        self.assertEqual(set(gate['env'].values()),
                         {f'${{{{ steps.{key}.outcome }}}}' for key in self.PROOFS})
        for key, selector in self.PROOFS.items():
            step = steps[key]
            self.assertIs(step['continue-on-error'], True)
            self.assertNotIn('if', step)  # failure is normalized only until the final outcome gate
            self.assertIn(selector + ' -- --exact --ignored --nocapture', step['run'])
            self.assertIn('test result: ok. 1 passed; 0 failed; 0 ignored;', step['run'])
            self.assertLess(job['steps'].index(step), job['steps'].index(gate))
        self.assertEqual(steps['raw_graph']['env']['CANDLE_GRAD_DO_NOT_DETACH'], '1')
        self.assertNotIn('CANDLE_GRAD_DO_NOT_DETACH', job.get('env', {}))
        self.assertFalse(job.get('continue-on-error', False))

    def test_six_independent_exact_proofs_have_a_non_suppressed_outcome_gate(self):
        workflow = document('ci.yml')
        job = workflow['jobs']['windows-cuda']
        self.assert_contract(job)
        for field, value in [('if', 'success()'), ('continue-on-error', True)]:
            mutated = copy.deepcopy(job)
            gate = next(step for step in mutated['steps'] if step.get('id') == 'raw_proofs_gate')
            gate[field] = value
            with self.assertRaises(AssertionError):
                self.assert_contract(mutated)
        mutated = copy.deepcopy(job)
        gate = next(step for step in mutated['steps'] if step.get('id') == 'raw_proofs_gate')
        gate['env'] = {key: value.replace('.outcome', '.conclusion') for key, value in gate['env'].items()}
        with self.assertRaises(AssertionError):
            self.assert_contract(mutated)
        self.assertIn('windows-cuda', workflow['jobs']['gate']['needs'])
        self.assertIn('always()', workflow['jobs']['gate']['if'])

    def test_raw_only_dispatch_is_explicit_validated_and_preserves_default_full_suite(self):
        from scripts.tests.test_ci_workflow_policy import evaluate_policy, posix_shell
        workflow = document('ci.yml')
        flag = workflow[True]['workflow_dispatch']['inputs']['raw_training_math_only']
        self.assertEqual(flag['type'], 'boolean')
        self.assertIs(flag['default'], False)
        job = workflow['jobs']['windows-cuda']
        broad = {'Build Candle CUDA packages', 'Test Candle CUDA packages',
                 'Test LLM-only CUDA bundle', 'Test LLM+audio CUDA bundle'}
        skipped = {step['name'] for step in job['steps']
                   if step.get('if') == '${{ !inputs.raw_training_math_only }}'}
        self.assertEqual(skipped, broad)
        for raw_only in [False, True]:
            for step in job['steps']:
                if step.get('name') in broad:
                    expression = step['if'].removeprefix('${{').removesuffix('}}').strip()
                    self.assertEqual(evaluate_policy(expression, lanes_include_cuda=True,
                                                     event_name='workflow_dispatch', head_repository='',
                                                     dispatch_lanes='windows-cuda', raw_training_math_only=raw_only),
                                     not raw_only)
        self.assert_contract(job)  # all six exact proofs still run in either mode
        self.assertNotIn('raw_training_math_only', str(workflow['jobs']['windows-cuda-check']))
        validation = next(step for step in workflow['jobs']['changes']['steps']
                          if step.get('name') == 'Validate manual raw-only selection')
        self.assertNotIn('if', validation)
        self.assertFalse(validation.get('continue-on-error', False))
        shell = posix_shell()
        if shell is None:
            self.skipTest('POSIX shell unavailable')
        for lane in ['all', 'windows-cuda', 'windows-gpu-diagnostic']:
            for raw_only in [False, True]:
                result = subprocess.run([shell, '-e', '-c', validation['run']],
                                        env={**os.environ, 'EVENT_NAME': 'workflow_dispatch',
                                             'DISPATCH_LANES': lane, 'RAW_ONLY': str(raw_only).lower()},
                                        capture_output=True, text=True, encoding='utf-8', timeout=10)
                self.assertEqual(result.returncode == 0, not raw_only or lane == 'windows-cuda')
                # Invalid diagnostic+raw-only still creates the validation job;
                # the normal diagnostic does not start the CI lane fan-out.
                changes = evaluate_policy(workflow['jobs']['changes']['if'], lanes_include_cuda=False,
                                          event_name='workflow_dispatch', head_repository='',
                                          dispatch_lanes=lane, raw_training_math_only=raw_only)
                self.assertEqual(changes, lane != 'windows-gpu-diagnostic' or raw_only)
        self.assertIn('inputs.raw_training_math_only', workflow['jobs']['gate']['if'])
        self.assertIn('!inputs.raw_training_math_only', workflow['jobs']['windows-gpu-diagnostic']['if'])

    def test_real_aggregate_script_rejects_each_failed_skipped_or_missing_proof(self):
        # Execute only the tiny gate over mocked status strings, never a cargo,
        # tensor, GPU, or model command. Use the existing cross-platform shell finder.
        from scripts.tests.test_ci_workflow_policy import posix_shell
        shell = posix_shell()
        if shell is None:
            self.skipTest('POSIX shell unavailable')
        workflow = document('ci.yml')
        gate = next(step for step in workflow['jobs']['windows-cuda']['steps']
                    if step.get('id') == 'raw_proofs_gate')
        names = list(gate['env'])
        def run(outcomes):
            return subprocess.run([shell, '-e', '-o', 'pipefail', '-c', gate['run']],
                                  env={**os.environ, **outcomes}, capture_output=True,
                                  text=True, encoding='utf-8', timeout=10)
        success = dict.fromkeys(names, 'success')
        self.assertEqual(run(success).returncode, 0)
        for name in names:
            for outcome in ['failure', 'cancelled', 'skipped', '']:
                with self.subTest(name=name, outcome=outcome):
                    result = run({**success, name: outcome})
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn(f'::error::{name}', result.stdout)
        # An unsuppressed failure in windows-cuda must also fail the top CI gate.
        ci_gate = workflow['jobs']['gate']['steps'][0]
        result = subprocess.run([shell, '-e', '-o', 'pipefail', '-c', ci_gate['run']],
                                env={**os.environ, 'RESULTS': 'success failure skipped'},
                                capture_output=True, text=True, encoding='utf-8', timeout=10)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
