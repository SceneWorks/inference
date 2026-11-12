"""The real-weights log check applies the proof runner's own execution contract (no model runs)."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock

from scripts.ci import model_inference_proofs as proofs
from scripts.ci import verify_model_inference_log as check

ROOT = Path(__file__).resolve().parents[2]
WRAPPERS = ROOT / 'scripts/ci/real-weights'
MLX, CANDLE = 'mlx-gen-minimax-h3', 'candle-gen-minimax-h3'
# Every registered gate the mlx-minimax-h3-image and candle-minimax-h3-image jobs run, by wrapper.
GATES = {
    'mlx-minimax-h3-image/run-the-image-vae-gates.sh': (MLX, 'mlx', [
        'real_weights::image_vae_decode_matches_the_official_diffusers_vae',
        'real_weights::image_vae_tiled_decode_lowers_the_16mp_peak']),
    'mlx-minimax-h3-image/run-the-dit-parity-and-the-2048-render.sh': (MLX, 'mlx', [
        *('h3_image_real::' + name for name in (
            'h3_image_dit_names_match_the_published_index', 'h3_image_dit_first_step_matches_diffusers',
            'h3_image_t2i_renders_2048_bf16', 'h3_image_edit_presentation_matches_the_processor_template',
            'h3_image_edit_reference_latents_match_diffusers', 'h3_image_edit_first_step_matches_diffusers',
            'h3_image_edit_renders_three_references_bf16', 'h3_image_t2i_renders_2048_q8',
            'h3_image_t2i_renders_2048_q4', 'h3_image_streamed_dit_matches_resident_q4',
            'h3_image_t2i_renders_16mp_streamed_q4')),
        'h3_image_bounded_attention_real::h3_image_rung3_lowers_the_denoise_peak_at_image_geometry',
        *('h3_image_lora_real::' + name for name in (
            'h3_image_lora_layouts_fold_identically_on_the_real_dit',
            'h3_image_streamed_blocks_carry_the_lora_like_resident_ones',
            'h3_image_adaln_and_output_head_loras_reach_the_denoiser'))]),
    'mlx-minimax-h3-image/run-the-sampler-a-b.sh': (MLX, 'mlx', [
        'h3_image_sampler_real::h3_image_sampler_ab_renders_2048_bf16']),
    'candle-minimax-h3-image/run-the-image-vae-gates.cmd': (CANDLE, 'candle-cuda', [
        'real_weights::image_vae_decode_matches_the_official_diffusers_vae',
        'real_weights::image_vae_tiled_decode_lowers_the_16mp_peak']),
    'candle-minimax-h3-image/run-the-image-dit-and-text-to-image-gates.cmd': (CANDLE, 'candle-cuda', [
        *('real_weights::' + name for name in (
            'image_dit_first_step_matches_the_official_diffusers_transformer',
            'image_text_to_image_renders_2048_on_cuda', 'image_edit_presentation_matches_the_processor_template',
            'image_edit_reference_latents_match_the_official_diffusers_vae',
            'image_edit_first_step_matches_the_official_diffusers_transformer',
            'image_edit_renders_three_references_on_cuda', 'image_t2i_renders_16mp_streamed_on_cuda',
            'image_t2i_renders_2048_q8_on_cuda', 'image_t2i_renders_2048_q4_on_cuda')),
        *('h3_image_lora_real::' + name for name in (
            'h3_image_lora_layouts_fold_identically_on_the_real_dit_cuda',
            'h3_image_adaln_and_output_head_loras_reach_the_denoiser_cuda',
            'h3_image_streamed_blocks_carry_the_lora_like_resident_ones_cuda'))]),
    'candle-minimax-h3-image/run-the-sampler-a-b.cmd': (CANDLE, 'candle-cuda', [
        'h3_image_sampler_real::image_sampler_ab_renders_2048_on_cuda']),
}
# Index and tokenizer checks: the body runs no learned tensor, so there is no device to observe.
NO_LEARNED_WORK = {
    'h3_image_real::h3_image_dit_names_match_the_published_index',
    'h3_image_real::h3_image_edit_presentation_matches_the_processor_template',
    'real_weights::image_edit_presentation_matches_the_processor_template',
}


def log(fqn: str, device: str | None, *, body: int = 1,
        summary: str = 'ok. 1 passed; 0 failed; 0 ignored;') -> str:
    lines = ['running 1 test'] + ([f'MODEL_INFERENCE_DEVICE {fqn} {device}'] if device else [])
    lines += [f'MODEL_INFERENCE_BODY_COMPLETED {fqn}'] * body
    return '\n'.join(lines + [f'test result: {summary} 0 measured; 3 filtered out']) + '\n'


class GateEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = proofs.load_registry()

    def test_every_gate_is_registered_for_its_backend_with_the_wrapper_features(self) -> None:
        for wrapper, (package, backend, fqns) in GATES.items():
            text = (WRAPPERS / wrapper).read_text(encoding='utf-8')
            built = {feature for match in re.findall(r'--features ([\w,-]+)', text) for feature in match.split(',')}
            self.assertIn('MODEL_INFERENCE_CASE_FQN', text, wrapper)
            self.assertRegex(text, r'verify_model_inference_log\.py --package ' + re.escape(package), wrapper)
            for fqn in fqns:
                with self.subTest(gate=fqn, wrapper=wrapper):
                    name = fqn.split('::')[1]
                    # The tier renders are named once with the tier as a parameter.
                    forms = {name, re.sub(r'_q[48](?=_on_cuda$)', '_%~1', name), re.sub(r'_q[48]$', '_$tier', name)}
                    self.assertTrue(any(form in text for form in forms), forms)
                    row = check.registered_case(self.rows, package, fqn, None, backend)
                    self.assertEqual(row['execution_role'], 'direct')
                    self.assertEqual(set(proofs.features(row)), built)

    def test_device_set_and_body_are_required(self) -> None:
        for wrapper, (package, backend, fqns) in GATES.items():
            device = 'mlx-metal' if backend == 'mlx' else backend
            for fqn in fqns:
                if fqn in NO_LEARNED_WORK:
                    continue
                row = check.registered_case(self.rows, package, fqn, None, backend)
                with self.subTest(gate=fqn):
                    self.assertEqual(proofs.verify_execution(row, log(fqn, device))['actual_devices'], [device])
                    wrong = 'candle-cuda' if backend == 'mlx' else 'candle-cpu'
                    for bad in (log(fqn, wrong), log(fqn, None), log(fqn, device, body=0),
                                log(fqn, device, body=2),
                                log(fqn, device, summary='ok. 0 passed; 0 failed; 1 ignored;')):
                        with self.assertRaises(ValueError):
                            proofs.verify_execution(row, bad)

    def test_only_the_index_and_tokenizer_gates_skip_the_device_set(self) -> None:
        for wrapper, (package, backend, fqns) in GATES.items():
            text = (WRAPPERS / wrapper).read_text(encoding='utf-8')
            exempt = [fqn for fqn in fqns if fqn in NO_LEARNED_WORK]
            self.assertEqual(text.count('--no-learned-work'), 1 if exempt else 0, wrapper)
            for fqn in exempt:
                row = check.registered_case(self.rows, package, fqn, None, backend)
                with self.subTest(gate=fqn):
                    check.verify_without_learned_work(row, log(fqn, None))
                    for bad in (log(fqn, None, body=0), log(fqn, None, body=2),
                                log(fqn, None, summary='ok. 0 passed; 0 failed; 1 ignored;'),
                                log(fqn, None) + f'MODEL_INFERENCE_PREREQUISITE_MISSING {fqn}\n'):
                        with self.assertRaises(ValueError):
                            check.verify_without_learned_work(row, bad)

    def test_backend_selection_refuses_a_backend_the_case_does_not_declare(self) -> None:
        with self.assertRaises(ValueError):
            check.registered_case(self.rows, CANDLE, 'real_weights::image_vae_tiled_decode_lowers_the_16mp_peak',
                                  None, 'candle-metal')
        with self.assertRaises(ValueError):
            check.registered_case(self.rows, MLX, 'real_weights::image_vae_tiled_decode_lowers_the_16mp_peak',
                                  None, 'candle-cuda')

    def test_cli_rejects_a_log_without_device_evidence(self) -> None:
        fqn = 'h3_image_sampler_real::h3_image_sampler_ab_renders_2048_bf16'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gate.log'
            path.write_text(log(fqn, None), encoding='utf-8')
            argv = ['verify', '--package', MLX, '--backend', 'mlx', '--fqn', fqn, '--log', str(path)]
            with mock.patch.object(sys, 'argv', argv), redirect_stderr(io.StringIO()):
                self.assertEqual(check.main(), 1)


if __name__ == '__main__':
    unittest.main()
