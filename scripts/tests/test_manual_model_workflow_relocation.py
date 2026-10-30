"""Source/YAML-only regression coverage for manual model workflow relocation.

No Cargo, model, kernel, GPU, or platform process is executed by these tests.
"""
import copy
import hashlib
import json
import os
import shutil
import re
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml

from scripts.check_model_inference_policy import can_run
from scripts.ci import moss_edit_cuda
from scripts.ci.real_weights_workflow import inline_text, script_references

ROOT = Path(__file__).resolve().parents[2]
MOVED = ('synthetic-w4a8-mlx', 'synthetic-w4a8-cuda', 'moss-edit-cuda', 'moss-edit-metal')
PROFILES = (*MOVED, 'synthetic-w4a8-both')
# Frozen command/step identities from the exact pre-relocation source, including
# assertion counts, selectors, device restrictions, artifacts and source controls.
STEP_DIGESTS = {
    'synthetic-w4a8-mlx': '78bb9c33e074b7b428719315b20f1a79da2e897953020825230a95c5d3bf32c2',
    'synthetic-w4a8-cuda': '5fa5828e13bcb4a9f8aca59d37bd21130e9d082dc4d2b3bb550c651db5f8c23e',
    'moss-edit-cuda': 'ea2605f4670833ace4a09b29b5a1ec5b137a9235f9295cc277bd19f18d995390',
    'moss-edit-metal': '35530430c9273d807490666d45f79e31d9f9a4159ad834526d8eecb8b642777c',
    'h3-pruned-hosted-metal': '3cb82643a6543b19344d94a73c0211948f11b0c463a733ce3e8064dfa6fc77a8',
}


# Original STEP_DIGESTS above remain the relocation receipt. The only later step
# deltas add source-traced fixture module forwards and their exact names/counts.
# Cross-backend Replay is an explicit non-NN skip; all original learned cases remain.
REPAIRED_H3_STEP_DIGESTS = {
    'synthetic-w4a8-mlx': '0a1d2ece9873b905af658a0fbaf89fb5a03a307f757b6dd8194b5c982ff417ab',
    'synthetic-w4a8-cuda': '36713e1d72834a98a25671e02e949f893862d0722e8c58ea5e582a488ed1c105',
}

# Main merge d68b36d93 adds exactly {uses: ./.github/actions/cleanup-runner-storage}
# after checkout in these two jobs. Every other complete step value equals parent
# 8889135c4 (the H3 receipt above and original MOSS receipt); model commands,
# assertions, source/device binding and selectors are unchanged. Retain both old
# receipts and accept only the full reviewed new lists, never strip cleanup steps.
REVIEWED_STORAGE_STEP_DIGESTS = {
    'synthetic-w4a8-cuda': 'e75d9a616cdc6eeb4c291863f790dea2ec800b0a1c37490627bc52a3418f7bcc',
    'moss-edit-cuda': '3cd52c621b7251bdbefc183df2cd535875d402b32a51afb38cab0627e1835c49',
}
CURRENT_STEP_DIGESTS = {**STEP_DIGESTS, **REPAIRED_H3_STEP_DIGESTS, **REVIEWED_STORAGE_STEP_DIGESTS}

# sc-27365 repins only the receipt-upload action of these jobs from the drifted v7.0.1 SHA to
# the v4 SHA every other workflow step uses. The receipts above stay authoritative: the test maps
# exactly that `uses:` value back and requires the complete step list to match unchanged.
UPLOAD_ARTIFACT_V4 = 'actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02'
UPLOAD_ARTIFACT_RECEIPT = 'actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a'
REPINNED_UPLOAD_ARTIFACT = ('moss-edit-cuda', 'moss-edit-metal')


def receipt_upload_pin(steps):
    """Copy of `steps` with the sc-27365 v4 upload pin mapped back to the receipt's SHA."""
    steps = copy.deepcopy(steps)
    for step in steps:
        if step.get('uses') == UPLOAD_ARTIFACT_V4:
            step['uses'] = UPLOAD_ARTIFACT_RECEIPT
    return steps


def receipt_steps(name, steps):
    return receipt_upload_pin(steps) if name in REPINNED_UPLOAD_ARTIFACT else steps

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class ManualModelWorkflowRelocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ordinary = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text(encoding='utf-8'))
        cls.manual_text = inline_text()
        cls.manual = yaml.safe_load(cls.manual_text)

    def test_no_learned_dispatch_route_or_input_remains_in_ordinary_ci(self):
        for job in (*MOVED, 'h3-pruned-hosted-metal'):
            self.assertNotIn(job, self.ordinary['jobs'])
        events = self.ordinary.get('on', self.ordinary.get(True))
        inputs = events['workflow_dispatch']['inputs']
        self.assertEqual(set(inputs), {'lanes'})
        self.assertEqual(inputs['lanes']['options'], ['all', 'windows-cuda', 'windows-gpu-diagnostic'])
        for job in self.ordinary['jobs'].values():
            for step in job.get('steps', []):
                command = step.get('run', '')
                self.assertNotIn('scripts.ci.moss_edit_cuda', command)
                self.assertNotIn('scripts/ci/moss_edit_metal.py', command)
                self.assertNotIn('pruned_adaln::native_pruned_adaln_matches_pinned_reference', command)
        self.assertEqual(set(self.ordinary['jobs']['gate']['needs']), set(self.ordinary['jobs']) - {'gate'})

    def test_every_explicit_profile_preserves_its_exact_route_set(self):
        expected = {
            'synthetic-w4a8-mlx': {'synthetic-w4a8-mlx', 'h3-pruned-hosted-metal'},
            'synthetic-w4a8-cuda': {'synthetic-w4a8-cuda'},
            'synthetic-w4a8-both': {'synthetic-w4a8-mlx', 'synthetic-w4a8-cuda', 'h3-pruned-hosted-metal'},
            'moss-edit-cuda': {'moss-edit-cuda'},
            'moss-edit-metal': {'moss-edit-metal'},
        }
        jobs = {name: self.manual['jobs'][name] for name in (*MOVED, 'h3-pruned-hosted-metal')}
        for event in ('workflow_dispatch', 'push', 'pull_request', 'merge_group', 'schedule', 'workflow_call'):
            for profile in (*PROFILES, '', 'all', 'media', 'synthetic-w4a8-mlx-extra'):
                context = {'github.event_name': {event}, 'inputs.profile': {profile}}
                actual = {name for name, job in jobs.items() if can_run(job['if'], context)}
                self.assertEqual(actual, expected.get(profile, set()) if event == 'workflow_dispatch' else set(), (event, profile))

    def test_relocated_jobs_keep_the_original_complete_steps(self):
        for name in MOVED:
            steps = copy.deepcopy(self.manual['jobs'][name]['steps'])
            if name == 'moss-edit-metal':
                # Same reviewed interpreter convention as every macOS real-weight job.
                for step in steps:
                    if 'run' in step: step['run'] = step['run'].replace('python3.12 ', 'python3 ')
            self.assertEqual(digest(receipt_steps(name, steps)), CURRENT_STEP_DIGESTS[name], name)
        hosted = self.manual['jobs']['h3-pruned-hosted-metal']
        step = next(step for step in hosted['steps'] if step.get('name', '').startswith('Check H3 precise'))
        self.assertEqual(digest(step), STEP_DIGESTS['h3-pruned-hosted-metal'])
        self.assertEqual(hosted['runs-on'], 'macos-15')
        self.assertEqual(hosted['env']['MACOSX_DEPLOYMENT_TARGET'], '15.0')
        self.assertEqual(self.manual['jobs']['synthetic-w4a8-mlx']['runs-on'], ['self-hosted', 'macOS', 'ARM64', 'rw-llm'])
        self.assertEqual(self.manual['jobs']['synthetic-w4a8-mlx']['env']['MACOSX_DEPLOYMENT_TARGET'], '26.2')

    def test_model_jobs_keep_the_dispatched_source_without_a_default_main_checkout(self):
        def assert_dispatch_source(jobs):
            for name in (*MOVED, 'h3-pruned-hosted-metal'):
                steps = jobs[name]['steps']
                checkouts = [step for step in steps if step.get('uses', '').startswith('actions/checkout@')]
                self.assertEqual(len(checkouts), 1, name)
                reference = checkouts[0].get('with', {}).get('ref')
                self.assertIn(reference, (None, '', '${{ github.sha }}', '${{ github.ref }}'), name)
                for step in steps:
                    self.assertNotRegex(step.get('run', ''), r'\bgit\s+(?:checkout|switch)\s+(?:--\s+)?[\"\']?(?:refs/heads/)?main\b', name)
        assert_dispatch_source(self.manual['jobs'])
        for name in (*MOVED, 'h3-pruned-hosted-metal'):
            mutant = copy.deepcopy(self.manual['jobs'])
            checkout = next(step for step in mutant[name]['steps'] if step.get('uses', '').startswith('actions/checkout@'))
            checkout.setdefault('with', {})['ref'] = 'main'
            with self.subTest(job=name), self.assertRaises(AssertionError):
                assert_dispatch_source(mutant)
        mutant = copy.deepcopy(self.manual['jobs'])
        mutant['synthetic-w4a8-mlx']['steps'].append({'run': 'git checkout main'})
        with self.assertRaises(AssertionError):
            assert_dispatch_source(mutant)

    def test_all_new_manual_roots_use_the_existing_intentional_selection_barrier(self):
        for name in (*MOVED, 'h3-pruned-hosted-metal', 'candle-supra2', 'mlx-supra2', 'mlx-flux3-klein', 'candle-flux3-klein'):
            self.assertEqual(self.manual['jobs'][name]['needs'], 'validate-selection', name)
        events = self.manual.get('on', self.manual.get(True))
        inputs = events['workflow_dispatch']['inputs']
        self.assertLessEqual(len(inputs), 25)
        self.assertEqual(inputs['profile']['default'], '')
        self.assertTrue(inputs['profile']['required'])
        self.assertTrue(set(PROFILES) <= set(inputs['profile']['options']))

    def test_explicit_runs_keep_distinct_queue_entries_without_cancellation(self):
        concurrency = self.manual['concurrency']
        self.assertIs(concurrency['cancel-in-progress'], False)
        expression = concurrency['group']
        for profile in PROFILES: self.assertIn("inputs.profile == '" + profile + "'", expression)
        self.assertIn("format('inference-model-proof-{0}-{1}', github.run_id, github.run_attempt)", expression)

    def test_new_script_indirections_have_exact_inline_round_trip_and_safe_bash_syntax(self):
        text = (ROOT / '.github/workflows/real-weights.yml').read_text(encoding='utf-8')
        for name in script_references(text):
            path = ROOT / name
            self.assertTrue(path.is_file(), name)
            body = path.read_text(encoding='utf-8')
            self.assertTrue(body.endswith('\n') and not body.endswith('\n\n'), name)
            if name.split('/')[-2] in (*MOVED, 'h3-pruned-hosted-metal') and path.suffix == '.sh':
                git_bash = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git/bin/bash.exe'
                bash = str(git_bash) if os.name == 'nt' and git_bash.is_file() else shutil.which('bash')
                self.assertIsNotNone(bash, 'Bash is required to validate relocated workflow scripts')
                subprocess.run([bash, '-n', str(path)], check=True, capture_output=True)
        self.assertLess((ROOT / '.github/workflows/real-weights.yml').stat().st_size, 450000)

    def test_manual_cuda_membership_and_override_include_each_ignored_case(self):
        for _, _, _, expected, expected_ignored in moss_edit_cuda.ARMS:
            self.assertEqual(expected_ignored, expected)
        source = (ROOT / 'scripts/ci/moss_edit_cuda.py').read_text(encoding='utf-8')
        self.assertIn("execution_flags = ['--ignored']", source)
        self.assertNotIn("execution_flags = ['--include-ignored']", source)


    def test_cuda_controller_executes_only_the_exact_ignored_members_after_listing(self):
        for missing_ignored in (False, True):
            with self.subTest(missing_ignored=missing_ignored), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                target = out / 'target'
                target.mkdir()
                binaries = {}
                for package in (moss_edit_cuda.MOSS, moss_edit_cuda.FLUX):
                    binary = target / (package + '.exe')
                    binary.write_bytes(b'mocked test artifact; never executed')
                    binaries[package] = binary
                numerical = []
                def runner(args, env, dest, label, seconds, **kwargs):
                    if label == 'compile':
                        self.assertIn('--no-run', args)
                        rows = [dict(reason='compiler-artifact', executable=str(binary), target=dict(name=package.replace('-', '_'), kind=['lib']), profile=dict(test=True)) for package, binary in binaries.items()]
                        return 0, '\n'.join(json.dumps(row) for row in rows)
                    arm = next(arm for arm in moss_edit_cuda.ARMS if label.startswith(arm[0] + '-'))
                    expected = arm[3]
                    if label.endswith(('-list', '-ignored')):
                        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '-1')
                        names = set() if missing_ignored and label == 'edit-ignored' else expected
                        return 0, '\n'.join(name + ': test' for name in sorted(names))
                    if label.endswith('-device'):
                        return 0, '1, ' + moss_edit_cuda.GPU_UUID + ', ' + moss_edit_cuda.GPU_PCI
                    self.assertTrue(label.endswith('-tests'))
                    self.assertIn('--ignored', args)
                    self.assertNotIn('--include-ignored', args)
                    self.assertEqual(env['CUDA_VISIBLE_DEVICES'], moss_edit_cuda.GPU_UUID)
                    self.assertEqual(env['SCENEWORKS_GPU_ID'], '1')
                    numerical.append(label)
                    return 0, f'test result: ok. {len(expected)} passed; 0 failed; 0 ignored;'
                env = {'CARGO_TARGET_DIR': str(target)}
                if missing_ignored:
                    with self.assertRaises(ValueError):
                        moss_edit_cuda.execute(env, out, runner=runner, source_check=lambda: {'source': 'fixed'})
                    self.assertEqual(numerical, [])
                else:
                    results = moss_edit_cuda.execute(env, out, runner=runner, source_check=lambda: {'source': 'fixed'})
                    self.assertEqual(numerical, ['edit-tests', 'layout-tests', 'encoder-tests'])
                    self.assertEqual(sum(len(row['expectedTests']) for row in results), 4)

if __name__ == '__main__':
    unittest.main()
