"""Source-only routing contracts; never dispatch, probe a device or execute a model."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / '.github/workflows'


def document(name):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding='utf-8'))


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
        for filename, name in (('model-inference-proofs.yml', 'prove'), ('real-weights.yml', 'model-proof-execute')):
            steps = document(filename)['jobs'][name]['steps']
            posix = next(step for step in steps if step.get('name') == 'Select Rustup proxies on Linux and macOS')
            windows = next(step for step in steps if step.get('name') == 'Select Rustup proxies on Windows')
            report = next(step for step in steps if step.get('name') == 'Report actual host and pinned toolchain (no model execution)')
            self.assertEqual(posix['if'], "runner.os != 'Windows'")
            self.assertEqual(windows['if'], "runner.os == 'Windows'")
            self.assertEqual(windows['shell'], 'cmd')
            self.assertIn('(cargo rustc rustup)', windows['run'])
            self.assertIn('if not exist', windows['run'])
            self.assertIn('%GITHUB_PATH%', windows['run'])
            self.assertLess(steps.index(posix), steps.index(report))
            self.assertLess(steps.index(windows), steps.index(report))
            self.assertIn('cargo --version', report['run'])
            self.assertIn('rustc --version', report['run'])
            selected.append((posix, windows))
        self.assertEqual(selected[0], selected[1])

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
        cases = (('yue2-precision-proof.yml', {'Run exactly one Metal precision test with external sampling'}),
                 ('yue2-app-precision-profile.yml', {'Install pinned BF16 model and both decoders through the app, without acceptance cases',
                                                   'Capture exactly seven Metal app precision profiles'}))
        for filename, names in cases:
            job = document(filename)['jobs']['metal']
            for step in job['steps']:
                if step.get('name') not in names:
                    continue
                self.assertIn('${{ github.workspace }}/control/scripts/ci/inference_metal_launcher.py', step['shell'])
                self.assertIn('python3.12 -I', step['shell'])
                self.assertIn('--noprofile --norc -eo pipefail {0}', step['shell'])
            self.assertTrue(any(step.get('with', {}).get('ref') == '${{ inputs.expected_engine_sha }}'
                                for step in job['steps']))
        ci = document('ci.yml')['jobs']['macos-nax']
        self.assertIn('inference_metal_launcher.py', ci['defaults']['run']['shell'])
        self.assertTrue(ci['defaults']['run']['shell'].endswith('/bin/bash -e {0}'))


if __name__ == '__main__':
    unittest.main()
