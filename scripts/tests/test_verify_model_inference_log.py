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
MLX = ('mlx-gen-minimax-h3', 'h3_image_sampler_real::h3_image_sampler_ab_renders_2048_bf16', None)
CANDLE = ('candle-gen-minimax-h3', 'h3_image_sampler_real::image_sampler_ab_renders_2048_on_cuda', 'fixed_cuda')
WRAPPERS = {
    MLX: ROOT / 'scripts/ci/real-weights/mlx-minimax-h3-image/run-the-sampler-a-b.sh',
    CANDLE: ROOT / 'scripts/ci/real-weights/candle-minimax-h3-image/run-the-sampler-a-b.cmd',
}


def log(fqn: str, device: str, *, body: int = 1, summary: str = 'ok. 1 passed; 0 failed; 0 ignored;') -> str:
    lines = ['running 1 test', f'MODEL_INFERENCE_DEVICE {fqn} {device}']
    lines += [f'MODEL_INFERENCE_BODY_COMPLETED {fqn}'] * body
    return '\n'.join(lines + [f'test result: {summary} 0 measured; 3 filtered out']) + '\n'


class SamplerAbEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = proofs.load_registry()

    def verify(self, case: tuple, text: str) -> dict:
        package, fqn, configuration = case
        return proofs.verify_execution(check.registered_case(self.rows, package, fqn, configuration), text)

    def test_registered_device_set_and_body_are_required(self) -> None:
        for case, device, wrong in ((MLX, 'mlx-metal', 'candle-cuda'), (CANDLE, 'candle-cuda', 'candle-cpu')):
            fqn = case[1]
            with self.subTest(case=fqn):
                self.assertEqual(self.verify(case, log(fqn, device))['actual_devices'], [device])
                for bad in (log(fqn, wrong), log(fqn, device, body=0), log(fqn, device, body=2),
                            log(fqn, device, summary='ok. 0 passed; 0 failed; 1 ignored;'),
                            log(fqn, device).replace('MODEL_INFERENCE_DEVICE', 'NO_DEVICE')):
                    with self.assertRaises(ValueError):
                        self.verify(case, bad)

    def test_wrappers_set_the_fqn_build_the_registered_features_and_run_the_check(self) -> None:
        for case, path in WRAPPERS.items():
            package, fqn, configuration = case
            row = check.registered_case(self.rows, package, fqn, configuration)
            text = path.read_text(encoding='utf-8')
            with self.subTest(wrapper=path.name):
                self.assertIn('MODEL_INFERENCE_CASE_FQN', text)
                self.assertIn(fqn, text)
                built = set(re.search(r'--features ([\w,-]+)', text)[1].split(','))
                self.assertEqual(built, set(proofs.features(row)))
                self.assertRegex(text, r'verify_model_inference_log\.py --package ' + re.escape(package))
                if configuration:
                    self.assertIn('--configuration ' + configuration, text)

    def test_cli_rejects_a_log_without_device_evidence(self) -> None:
        package, fqn, _ = MLX
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gate.log'
            path.write_text(log(fqn, 'mlx-metal').replace('MODEL_INFERENCE_DEVICE', 'X'), encoding='utf-8')
            argv = ['verify', '--package', package, '--fqn', fqn, '--log', str(path)]
            with mock.patch.object(sys, 'argv', argv), redirect_stderr(io.StringIO()):
                self.assertEqual(check.main(), 1)


if __name__ == '__main__':
    unittest.main()
