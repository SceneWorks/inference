"""Source-only regression for the native MOSS norm's comparison witness."""
from pathlib import Path
import re
import unittest
from scripts.check_model_inference_policy import matching_brace, rust_mask


def legacy_norm_branch(source):
    start = source.index('impl Norm {')
    mask = rust_mask(source)
    gate = mask.index('if candle_gen::train::policy::legacy_e1()', start)
    opening = mask.index('{', gate)
    return source[opening:matching_brace(mask, opening)]


def has_actual_legacy_witness(source):
    branch = legacy_norm_branch(source)
    mask = rust_mask(branch)
    gate = mask.find('if candle_gen::train::comparison::active_stage().is_some()')
    if gate < 0:
        return False
    opening = mask.index('{', gate)
    end = matching_brace(mask, opening)
    calls = list(re.finditer(r'cohort_reuse\("comparison:e1-norm:legacy"\)', branch))
    return len(calls) == 1 and opening < calls[0].start() < end and mask[calls[0].start():].startswith('cohort_reuse(')


class MossLegacyNormWitness(unittest.TestCase):
    def test_actual_norm_branch_records_generic_e1_witness(self):
        path = Path(__file__).resolve().parents[2] / 'crates/media/candle-gen/candle-gen-moss-image/src/transformer.rs'
        source = path.read_text(encoding='utf-8')
        self.assertTrue(has_actual_legacy_witness(source))
        branch = legacy_norm_branch(source)
        self.assertIn('moss:comparison-legacy-rms', branch)
        for changed in (
            source.replace('"comparison:e1-norm:legacy"', '"comparison:e1-norm:current"'),
            source.replace('cohort_reuse("comparison:e1-norm:legacy")', 'cohort_reuse("unrelated")'),
            source.replace('if candle_gen::train::comparison::active_stage().is_some()', 'if false'),
        ):
            with self.subTest(mutation=changed != source):
                self.assertFalse(has_actual_legacy_witness(changed))


if __name__ == '__main__':
    unittest.main()
