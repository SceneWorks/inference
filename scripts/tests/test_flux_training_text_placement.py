"""Source-only native input placement regressions; no tensor/device/model work."""
from pathlib import Path
import re
import unittest

from scripts.check_model_inference_policy import matching_brace, rust_mask


def raw_training_branch(source):
    mask = rust_mask(source)
    start = mask.index('fn forward_impl(')
    gate = mask.index('_ if training =>', start)
    opening = mask.index('{', gate)
    return source[opening:matching_brace(mask, opening)]


def has_native_placement(source):
    try:
        branch = rust_mask(raw_training_branch(source))
    except ValueError:
        return False
    normalized = re.sub(r'\s+', '', branch)
    return ('letx=to(&x,self.txt_in.dtype())?.to_device(&self.device)?;' in normalized
            and normalized.index('to_device(&self.device)') < normalized.index('self.txt_in.forward(&x)?')
            and all(x not in normalized for x in ('cache_enabled', 'projection_enabled', 'Workflow::')))


class FluxTrainingTextPlacement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.source = (cls.root / 'crates/media/candle-gen/candle-gen-flux3/src/dit.rs').read_text(encoding='utf-8')

    def test_raw_training_placement_is_before_unpacked_projection_for_both_workflows(self):
        self.assertTrue(has_native_placement(self.source))
        prepared = self.source.index('_ if training && self.training_text_prepared =>')
        raw = self.source.index('_ if training =>', prepared)
        inference = self.source.index('_ => self.txt_in.forward(&x)?', raw)
        self.assertLess(prepared, raw)
        self.assertLess(raw, inference)
        self.assertIn('x.to_device(&self.device)?', self.source[prepared:raw])
        runner = (self.root / 'crates/media/candle-gen/candle-gen-flux3/src/training/runner.rs').read_text(encoding='utf-8')
        self.assertIn('CaptionChoice::Original => &example.text', runner)
        self.assertIn('CaptionChoice::EmptyAtWindow', runner)
        self.assertIn('text,\n                    text_ids:', runner)
        # The raw training arm has no Captioned/PairedEdit or projection-pack
        # gate: both modes, original/empty text and E3 on/off reach placement.
        self.assertNotIn('Workflow::', raw_training_branch(self.source))

    def test_old_missing_transfer_wrong_device_dtype_and_dead_branch_are_rejected(self):
        for changed in (
            self.source.replace('to(&x, self.txt_in.dtype())?.to_device(&self.device)?', 'to(&x, self.txt_in.dtype())?'),
            self.source.replace('to(&x, self.txt_in.dtype())?.to_device(&self.device)?', 'to(&x, self.txt_in.dtype())?.to_device(&Device::Cpu)?'),
            self.source.replace('to(&x, self.txt_in.dtype())?.to_device(&self.device)?', 'x.to_device(&self.device)?'),
            self.source.replace('_ if training =>', '_ if false =>'),
        ):
            with self.subTest(mutation=changed != self.source):
                self.assertNotEqual(changed, self.source)
                self.assertFalse(has_native_placement(changed))


if __name__ == '__main__':
    unittest.main()
