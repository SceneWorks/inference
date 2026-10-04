"""Representative training must disclose its scope and retain two-reference evaluation."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "crates/media/mlx-gen/mlx-gen-qwen-image-2-1/tests/lora_real_weights.rs"


def protocol_errors(source):
    errors = []
    body = source.split("fn edit_lokr_trains_and_moves_two_reference_edits_every_tier()", 1)[-1].split("fn stacked_adapters", 1)[0]
    for fragment in ['TRAIN_EDIT_INSTRUCTION.into()', 'vec![src_path]',
                     '"trainingReferenceCount": 1, "evaluationReferenceCount": 2',
                     '"trainingCaption": TRAIN_EDIT_INSTRUCTION, "evaluationCaption": EDIT_INSTRUCTION',
                     '"dataset": dataset_receipt(&req.items)', '"evaluationKeySha256": sha256_file(&key_path)',
                     'write_json(&out, "edit-training-protocol", &protocol)',
                     'trained.facts["editProtocol"] = protocol', 'prompt: EDIT_INSTRUCTION.to_owned()',
                     'images: vec![to_image(eval_src), to_image(key)]',
                     'training_steps("QWEN_IMAGE_2_1_LORA_EDIT_STEPS", 120)']:
        if fragment not in body:
            errors.append(fragment)
    if body.index('write_json(&out, "edit-training-protocol", &protocol)') > body.index('train(&req'):
        errors.append("protocol must precede admission")
    captions = dict(re.findall(r'const (\w*EDIT_INSTRUCTION): &str =\s*"([^"]+)";', source))
    if captions.get("TRAIN_EDIT_INSTRUCTION") != captions.get("EDIT_INSTRUCTION", "").replace(" shown in image 2", ""):
        errors.append("training caption names a nonexistent second reference or changes RGB semantics")
    for fragment in ['const EDIT_GAIN_FLOOR: f64 = 1.0;', 'const ADAPTER_MOVES_FLOOR: f64 = 2.0;',
                     '"adapterSha256": sha256_file(canonical)',
                     '"orderedReferences": item.reference_image_paths.iter()',
                     'Sha256::digest(serde_json::to_vec(&rows).unwrap())']:
        if fragment not in source:
            errors.append(fragment)
    return errors


class EditProtocolTests(unittest.TestCase):
    def test_disclosed_scope_preserves_evaluation(self):
        self.assertEqual(protocol_errors(SOURCE.read_text(encoding="utf-8")), [])

    def test_misleading_or_weakened_protocol_mutants_are_rejected(self):
        source = SOURCE.read_text(encoding="utf-8")
        mutations = [
            ('vec![src_path]', 'vec![src_path, key_path.clone()]'),
            ('TRAIN_EDIT_INSTRUCTION.into()', 'EDIT_INSTRUCTION.into()'),
            ('"trainingReferenceCount": 1', '"trainingReferenceCount": 2'),
            ('images: vec![to_image(eval_src), to_image(key)]', 'images: vec![to_image(eval_src)]'),
            ('const EDIT_GAIN_FLOOR: f64 = 1.0;', 'const EDIT_GAIN_FLOOR: f64 = 0.1;'),
            ('"adapterSha256": sha256_file(canonical)', '"adapterSha256": "unknown"'),
            ('"orderedReferences": item.reference_image_paths.iter()', '"references": item.reference_image_paths.iter()'),
            ('Sha256::digest(serde_json::to_vec(&rows).unwrap())', 'Sha256::digest(b"dataset")'),
            ('"evaluationKeySha256": sha256_file(&key_path)', '"evaluationKeySha256": "unknown"'),
        ]
        for old, new in mutations:
            with self.subTest(mutation=old):
                self.assertIn(old, source)
                self.assertTrue(protocol_errors(source.replace(old, new)))


if __name__ == "__main__":
    unittest.main()
