"""No model/tensor work: discriminate the fixed Krea recovery evidence protocol."""
import copy
import functools
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import krea_recovery as recovery
from scripts.ci import model_inference_proofs as proof


@functools.lru_cache(maxsize=1)
def source_row():
    return next(row for row in proof.load_registry() if recovery.applies(row))


def fixture(seconds=6., dispatches=168, loss_change=0.):
    row = proof.configured_case(source_row(), recovery.CONFIGURATION)
    old = recovery.baseline()
    losses = [float(step['loss']) + loss_change for step in old['reference_segments'][1]['steps']]
    reference = sum(float(s['seconds']) for s in old['reference_segments'][0]['steps'][2:]) / 4
    shared = sum(float(s['seconds']) for s in old['reference_segments'][1]['steps'][2:]) / 4
    uncertainty = old['timing_mean_rounding_uncertainty_seconds']
    parity = all(abs(float(s['loss']) - loss) <= .01 * max(abs(float(s['loss'])), 1)
                 for s, loss in zip(old['reference_segments'][1]['steps'], losses))
    go = dispatches > 0 and parity and seconds < shared - uncertainty
    ambiguous25 = abs(seconds - .75 * reference) <= .75 * uncertainty
    report = {'schema_version': 1, 'new_optimizer_updates': 6, 'new_segment': 2,
        'steady_updates': 4, 'prior_conservative_update_charge': 14, 'hard_update_cap': 20,
        'remaining_updates_after_success': 0, 'baseline': old,
        'reference_logged_mean_seconds': reference, 'shared_logged_mean_seconds': shared,
        'candidate_mean_seconds': seconds, 'candidate_steady_seconds': [seconds] * 4,
        'candidate_losses': losses, 'candidate_dispatches': dispatches, 'loss_parity': parity,
        'candidate_go': go, 'go_rounding_boundary_ambiguous': abs(seconds - shared) <= uncertainty,
        'target_met': go and not ambiguous25 and seconds <= .75 * (reference - uncertainty),
        'target_rounding_boundary_ambiguous': ambiguous25,
        'candidate_reduction_vs_original_logged_mean': 1 - seconds / reference,
        'saved_adapter_role': recovery.ADAPTER_ROLE, 'shared_adapter_available': False}
    identity = {'source': '1' * 40, 'run': '41199999999', 'attempt': '1'}
    lines = [f'test result: ok. 1 passed; 0 failed; 0 ignored;',
             'MODEL_INFERENCE_BODY_COMPLETED ' + recovery.FQN,
             'MODEL_INFERENCE_DEVICE ' + recovery.FQN + ' candle-cuda']
    for i in range(1, 7):
        for phase in ('update-start', 'update-committed'):
            lines.append(f'KREA_PROFILE_BOUNDARY phase={phase} physical_update={i} segment=2 logical_step={i}')
        lines.append(f'KREA_PROFILE_STEP physical_update={i}/6 segment=2 logical_step={i} warmup={str(i <= 2).lower()} seconds={seconds:.6f} loss={losses[i-1]} attention_candidate={dispatches // 6} attention_reference={28 if not dispatches else 0}')
        lines.append(f'KREA_PROFILE_GRADIENTS physical_update={i}/6 finite=224 total=224')
    lines.extend(['KREA_PROFILE_ATTRIBUTION segment=2 synchronized_inclusive_spans={synthetic scalar fixture only}',
                  'KREA_PROFILE_OPTIMIZER segment=2 synchronized_seconds=0.01',
                  'KREA_PROFILE_PREPARATION seconds=0.1',
                  'KREA_PROFILE_RECOVERY_ADAPTER changed=true',
                  'KREA_PROFILE_RECOVERY_RESULT ' + json.dumps(report),
                  'KREA_PROFILE_RECOVERY_EXECUTION ' + json.dumps(identity)])
    return row, lines, report, identity


def log(lines):
    return '\n'.join(lines) + '\n'


class KreaRecoveryProtocol(unittest.TestCase):
    def test_fixed_binding_default18_and_explicit6_are_distinct_and_inherited_flag_is_unset(self):
        source = source_row()
        default = proof.configured_case(source)
        self.assertEqual(default['selected_configuration']['name'], 'fixed_cuda')
        env = {recovery.CONTROL: '1'}
        proof.apply_configuration(default, env)
        self.assertNotIn(recovery.CONTROL, env)
        explicit = proof.configured_case(source, recovery.CONFIGURATION)
        proof.apply_configuration(explicit, env)
        self.assertEqual(env[recovery.CONTROL], '1')
        for mutate in (lambda r: r['configurations'][recovery.CONFIGURATION]['environment'].update({recovery.CONTROL: '0'}),
                       lambda r: r['configurations'][recovery.CONFIGURATION].pop('source_guard'),
                       lambda r: r['configurations'][recovery.CONFIGURATION].update(source_guard='18-updates')):
            invalid = copy.deepcopy(source); mutate(invalid)
            with self.assertRaises(ValueError):
                proof.configured_case(invalid, recovery.CONFIGURATION)

    def test_source_loop_caps_before_work_and_replays_same_logical_schedule(self):
        # Model-free source reachability guard: removing/moving this admission guard
        # must not turn a metadata-only budget check into six followed by more work.
        source = (ROOT / 'crates/media/candle-gen/candle-gen/src/train/flow_match.rs').read_text()
        guard = source.index('|| cfg.steps != profile_update_limit')
        self.assertLess(source.index('let profile_update_limit = if recovery_profile { 6 } else { 18 };'), guard)
        self.assertLess(guard, source.index('model.cache(req, device, on_progress)?'))
        self.assertIn('for step in start_step.saturating_add(1)..=cfg.steps', source)
        self.assertIn('(step - 1) % 6 + 1', source)
        self.assertIn('step_sample(aux_driver.as_mut(), logical_step, &schedule)', source)
        self.assertIn('if segment == 1 || recovery_profile', source)
        self.assertIn('profile_losses[..2].iter().any(|steps| !steps.is_empty())', source)
        profile = (ROOT / 'crates/media/candle-gen/candle-gen/src/train/profile.rs').read_text()
        scope = profile[profile.index('pub fn with_bounded_krea_recovery'):profile.index('pub(crate) fn finish_krea_recovery')]
        self.assertIn('RECOVERY.set(true)', scope)
        self.assertNotIn('SEGMENT.set(2)', scope)  # cache/build use original segment0 math

    def test_source_bound_original_scalar_identity_and_means(self):
        old = recovery.baseline()
        self.assertEqual(old['source'], '69c25a1d87f761464cb200802dfbb04125e9f54c')
        self.assertEqual((old['run'], old['attempt'], old['artifact_id']), (41127490702, 1, 13013732335))
        self.assertEqual(old['body_stderr_sha256'], '5bb020a304f514c890c3139efdf456f6955d1f26b6a54346052172d8f6b8e0e8')
        self.assertFalse(old['accepted_model_run'])
        self.assertEqual(old['timing_mean_rounding_uncertainty_seconds'], .0000005)
        self.assertEqual([len(s['steps']) for s in old['reference_segments']], [6, 6])

    def test_complete_scalar_fixture_can_prove_go_and25_but_not_runtime(self):
        row, lines, report, identity = fixture()
        result = proof.verify_execution(row, log(lines))
        self.assertEqual(result['krea_recovery'], report)
        self.assertTrue(report['candidate_go']); self.assertTrue(report['target_met'])
        self.assertEqual(result['krea_candidate_execution'], identity)

    def test_no_go_keeps_actual_candidate_and_cannot_claim_shared_or25(self):
        for options in ({'seconds': 9.5}, {'dispatches': 0}, {'loss_change': .02}):
            row, lines, report, _ = fixture(**options)
            self.assertFalse(report['candidate_go']); self.assertFalse(report['target_met'])
            self.assertEqual(proof.verify_execution(row, log(lines))['krea_recovery']['saved_adapter_role'], recovery.ADAPTER_ROLE)
            for key, value in (('candidate_go', True), ('target_met', True), ('shared_adapter_available', True),
                               ('saved_adapter_role', 'selected-shared-reference')):
                changed = copy.deepcopy(report); changed[key] = value
                mutated = [line if not line.startswith('KREA_PROFILE_RECOVERY_RESULT ') else 'KREA_PROFILE_RECOVERY_RESULT ' + json.dumps(changed) for line in lines]
                with self.subTest(options=options, key=key), self.assertRaises(ValueError):
                    proof.verify_execution(row, log(mutated))

    def test_original_logged_rounding_boundaries_stay_ambiguous_not_go_or25(self):
        old = recovery.baseline()
        shared = sum(float(s['seconds']) for s in old['reference_segments'][1]['steps'][2:]) / 4
        reference = sum(float(s['seconds']) for s in old['reference_segments'][0]['steps'][2:]) / 4
        for seconds in (shared - .0000002, shared + .0000002):
            row, lines, report, _ = fixture(seconds=seconds)
            self.assertTrue(report['go_rounding_boundary_ambiguous']); self.assertFalse(report['candidate_go'])
            proof.verify_execution(row, log(lines))
        for seconds in (.75 * reference - .0000001, .75 * reference + .0000001):
            row, lines, report, _ = fixture(seconds=seconds)
            self.assertTrue(report['target_rounding_boundary_ambiguous']); self.assertFalse(report['target_met'])
            proof.verify_execution(row, log(lines))

    def test_missing_extra_replayed_or18_labeled_updates_cannot_be_accepted(self):
        row, lines, _, _ = fixture()
        mutations = [lambda xs: xs + ['KREA_PROFILE_BOUNDARY phase=update-start physical_update=7 segment=2 logical_step=7'],
                     lambda xs: [x for x in xs if 'phase=update-committed physical_update=6 ' not in x],
                     lambda xs: [x.replace('segment=2', 'segment=1') for x in xs],
                     lambda xs: [x.replace('/6 ', '/18 ') for x in xs],
                     lambda xs: [x.replace('finite=224', 'finite=223') for x in xs],
                     lambda xs: [x for x in xs if not x.startswith('KREA_PROFILE_RECOVERY_ADAPTER')],
                     lambda xs: xs + ['KREA_PROFILE_RESULT updates=18'],
                     lambda xs: [x for x in xs if not x.startswith('KREA_PROFILE_ATTRIBUTION')],
                     lambda xs: [x for x in xs if not x.startswith('KREA_PROFILE_RECOVERY_RESULT')]]
        for mutate in mutations:
            with self.assertRaises(ValueError):
                proof.verify_execution(row, log(mutate(lines)))
        with self.assertRaisesRegex(ValueError, 'masquerade'):
            proof.verify_execution(proof.configured_case(source_row()), log(lines))

    def test_original_evidence_loss_timing_budget_and_new_identity_tampering_rejected(self):
        row, lines, report, _ = fixture()
        for mutate in (lambda r: r.update(new_optimizer_updates=18),
                       lambda r: r['baseline'].update(source='2' * 40),
                       lambda r: r['baseline'].update(run=41199999999),
                       lambda r: r['baseline'].update(body_stderr_sha256='0' * 64),
                       lambda r: r['candidate_losses'].__setitem__(5, .1),
                       lambda r: r['candidate_steady_seconds'].__setitem__(0, 1.),
                       lambda r: r.update(candidate_dispatches=0),
                       lambda r: r.update(candidate_mean_seconds=0.),
                       lambda r: r.update(hard_update_cap=21)):
            changed = copy.deepcopy(report); mutate(changed)
            mutated = [x if not x.startswith('KREA_PROFILE_RECOVERY_RESULT ') else 'KREA_PROFILE_RECOVERY_RESULT ' + json.dumps(changed) for x in lines]
            with self.assertRaises(ValueError):
                proof.verify_execution(row, log(mutated))

    def test_metrics_retain_candidate_source_and_actual_candidate_adapter_even_no_go(self):
        row, lines, report, identity = fixture(seconds=9.5)
        result = proof.verify_execution(row, log(lines))
        metrics = {'completed_optimizer_updates': 6, 'recovery': report, 'candidate_execution': identity,
                   'losses_by_physical_update': report['candidate_losses'], 'saved_adapter_role': recovery.ADAPTER_ROLE,
                   'graph_retaining_backward': False, 'selected_adapter_final_loss': report['candidate_losses'][-1],
                   'configuration': {'rank':16, 'alpha':16., 'learning_rate':.001, 'resolution':1024,
                   'train_dtype':'bf16', 'gradient_accumulation':1, 'gradient_checkpointing':False,
                   'seed':7, 'sample_every':500, 'save_every':0}}
        recovery.verify_metrics(row, metrics, result, identity['source'])
        for mutate in (lambda m: m.update(completed_optimizer_updates=18),
                       lambda m: m.update(saved_adapter_role='reference'),
                       lambda m: m['configuration'].update(sample_every=0),
                       lambda m: m['candidate_execution'].update(source='2' * 40)):
            invalid = copy.deepcopy(metrics); mutate(invalid)
            with self.assertRaises(ValueError):
                recovery.verify_metrics(row, invalid, result, identity['source'])


if __name__ == '__main__':
    unittest.main()
