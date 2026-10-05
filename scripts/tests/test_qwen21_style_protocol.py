"""Keep the original style donor's direction request separate from edit transfers."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "crates/media/mlx-gen/mlx-gen-qwen-image-2-1/tests/lora_real_weights.rs"


def body(source, name):
    return source.split(f"fn {name}(", 1)[1].split("\n}\n", 1)[0]


def protocol_errors(source):
    errors = []
    request = body(source, "original_style_request")
    if request.strip() != ") -> GenerationRequest {\n    GenerationRequest {\n        prompt: style_protocol::ORIGINAL_STYLE_PROMPT.to_owned(),\n        ..t2i_request()\n    }":
        errors.append("original request changes more than its frozen prompt")
    matrix = body(source, "imported_adapters_move_t2i_and_two_reference_edit_every_tier")
    for fragment in ['verify_original_style_donor(entry, &file)',
                     '(mode == "t2i").then(||', '"{tier}_t2i_original_style_base"',
                     'if style_protocol::uses_original_style_request(name, mode)',
                     '(image, facts, &style_request)', '(&base, &base_facts, &request)',
                     'paired_request,', '"base": paired_base_facts',
                     '"request": direction_request_facts(paired_request)',
                     'mean_abs_diff(&adapted, paired_base)',
                     'palette_distance(paired_base) - palette_distance(&adapted)',
                     'images.push((label, paired_base.clone(), adapted))',
                     '"additionalOriginalStyleBaseRenders": 3',
                     'case["adapter"] == "mlx_t2i_1000_steps" && case["mode"] == "t2i"']:
        if fragment not in matrix:
            errors.append(fragment)
    diagnostic = body(source, "diagnostic_reused_t2i_style_direction")
    for fragment in ['let request = original_style_request()', 'for (tier, quant) in tiers()',
                     'verify_original_style_donor(entry, &file)', 'donors.len(),',
                     'directory.is_absolute() && directory.is_dir()',
                     'Path::new(basename).components().count() == 1',
                     '"acceptanceEvidence": false', '"retrain": false', '"renderCount": 6',
                     '"sourceCandidate": source', '"trainingProvenance": provenance',
                     '"request": direction_request_facts(&request)',
                     'adapter(&file, 1.0, AdapterKind::Lora)',
                     'let guard = Footprint::start(&out)',
                     'assert_sane(label, base)', 'assert_sane(label, adapted)',
                     'assert_overlay_not_underpredicted(label, &case["base"], &case["adapted"])',
                     'case["meanAbsDiff"].as_f64().unwrap() >= ADAPTER_MOVES_FLOOR',
                     'case["paletteDistanceGain"].as_f64().unwrap() >= PALETTE_DISTANCE_GAIN_FLOOR']:
        if fragment not in diagnostic:
            errors.append(fragment)
    if 'train(' in diagnostic or 'TrainingRequest' in diagnostic:
        errors.append("diagnostic must never train")
    if diagnostic.index('"style-direction"') > diagnostic.index('assert_sane(label, base)'):
        errors.append("write all evidence before assertions")
    for fragment in ['assert_eq!(entry["sha256"], style_protocol::DONOR_SHA256)',
                     'assert_eq!(entry["size"], style_protocol::DONOR_BYTES)',
                     'assert_eq!(sha256_file(file), style_protocol::DONOR_SHA256)',
                     'style_protocol::palette_distance(&img.pixels)',
                     'const PALETTE_DISTANCE_GAIN_FLOOR: f64 = 1.0;',
                     'const ADAPTER_MOVES_FLOOR: f64 = 2.0;',
                     'const EDIT_GAIN_FLOOR: f64 = 1.0;',
                     'const SEED: u64 = 24163;', 'const RENDER_STEPS: u32 = 8;']:
        if fragment not in source:
            errors.append(fragment)
    return errors


class StyleProtocolTests(unittest.TestCase):
    def test_original_donor_pairs_and_diagnostic_keep_guards(self):
        self.assertEqual(protocol_errors(SOURCE.read_text(encoding="utf8")), [])

    def test_wrong_pair_provenance_and_weakened_oracles_are_rejected(self):
        source = SOURCE.read_text(encoding="utf8")
        mutations = [
            ('(image, facts, &style_request)', '(image, facts, &request)'),
            ('"base": paired_base_facts', '"base": base_facts'),
            ('mean_abs_diff(&adapted, paired_base)', 'mean_abs_diff(&adapted, &base)'),
            ('palette_distance(paired_base) - palette_distance(&adapted)', 'palette_distance(&adapted) - palette_distance(paired_base)'),
            ('(mode == "t2i").then(||', '(mode == "two_reference_edit").then(||'),
            ('let request = original_style_request()', 'let request = t2i_request()'),
            ('"acceptanceEvidence": false', '"acceptanceEvidence": true'),
            ('verify_original_style_donor(entry, &file)', '// skip donor verification'),
            ('assert_eq!(sha256_file(file), style_protocol::DONOR_SHA256)', 'assert_eq!(entry["sha256"], style_protocol::DONOR_SHA256)'),
            ('const PALETTE_DISTANCE_GAIN_FLOOR: f64 = 1.0;', 'const PALETTE_DISTANCE_GAIN_FLOOR: f64 = 0.0;'),
            ('case["paletteDistanceGain"].as_f64().unwrap() >= PALETTE_DISTANCE_GAIN_FLOOR', 'case["meanAbsDiff"].as_f64().unwrap() >= ADAPTER_MOVES_FLOOR'),
            ('assert_sane(label, adapted)', '// no sanity check'),
            ('let guard = Footprint::start(&out)', 'let guard = unguarded()'),
        ]
        for old, new in mutations:
            with self.subTest(mutation=old):
                self.assertIn(old, source)
                self.assertTrue(protocol_errors(source.replace(old, new)))


if __name__ == "__main__":
    unittest.main()
