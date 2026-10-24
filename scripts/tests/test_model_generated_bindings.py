"""Exact audited Rust macro bindings; source text only, no Rust/model execution."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import check_model_inference_policy as policy

H3 = 'crates/media/candle-gen/candle-gen-minimax-h3/tests/turbo_lora.rs'
SD3 = 'crates/media/candle-gen/candle-gen-sd3/tests/trainer_cuda_smoke.rs'


def alpha_source(names=('nn_a', 'nn_b')):
    arms = '\n'.join('"' + name + '" => println!("\\nMODEL_INFERENCE_BODY_COMPLETED turbo_lora::' + name + '"),' for name in names)
    invocations = '\n'.join(name + ', metadata_' + name + ' => AlphaSpelling::InBand;' for name in names)
    return '''macro_rules! per_alpha_spelling {
    ($($name:ident, $chain:ident => $spelling:expr;)+) => {
        const GENERATED_SPELLINGS: &[AlphaSpelling] = &[$($spelling),+];
        $(
            #[test]
            #[ignore = "model inference; explicit model workflow only"]
            fn $name() {
                assert_twin_equivalence($spelling);
                #[cfg(feature = "model-inference-proof")]
                match stringify!($name) {
''' + arms + '''
                    _ => panic!("undeclared generated model-proof case"),
                };
            }
            #[test]
            fn $chain() { assert_alpha_resolves_before_the_split($spelling); }
        )+
    };
}
per_alpha_spelling! {
''' + invocations + '\n}\n'


def smoke_source(names=('nn_a', 'nn_b')):
    arms = '\n'.join('"' + name + '" => println!("\\nMODEL_INFERENCE_BODY_COMPLETED trainer_cuda_smoke::' + name + '"),' for name in names)
    invocations = '\n'.join('smoke!(' + name + ', "fixture_model", "FIXTURE_ENV", NetworkType::Lora);' for name in names)
    return '''macro_rules! smoke {
    ($name:ident, $id:literal, $env:literal, $kind:expr) => {
        #[test]
        #[ignore = "needs real SD3.5 weights and explicitly scheduled CUDA"]
        fn $name() {
            run($id, $env, $kind);
            #[cfg(feature = "model-inference-proof")]
            match stringify!($name) {
''' + arms + '''
                _ => panic!("undeclared generated model-proof case"),
            };
        }
    };
}
''' + invocations + '\n'


def rows(source, namespace, names=('nn_a', 'nn_b')):
    return [{'package': 'candle-gen-fixture', 'target': 'integration', 'fqn': namespace + '::' + name,
             'source': source, 'execution_role': 'direct', 'backend': 'candle-cpu'} for name in names]


class GeneratedBindings(unittest.TestCase):
    def assert_guard(self, relative, source, bindings, valid):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / relative; path.parent.mkdir(parents=True); path.write_text(source, encoding="utf-8")
            errors = policy.registry_errors(root, bindings)
            self.assertEqual(not errors, valid, '\n'.join(errors))

    def test_exact_alpha_macro_expands_nn_and_retains_ordinary_metadata_without_population_gate(self):
        for names in (('nn_a', 'nn_b'), ('nn_a', 'nn_b', 'nn_c')):
            text = alpha_source(names)
            generated = [item for item in policy.functions(text, source=H3) if item.get('generated_by')]
            self.assertEqual({item['name'] for item in generated if item['ignored']}, set(names))
            self.assertEqual({item['name'] for item in generated if not item['ignored']}, {'metadata_' + name for name in names})
            self.assert_guard(H3, text, rows(H3, 'turbo_lora', names), True)

    def test_exact_smoke_macro_expands_only_declared_actual_invocations(self):
        for names in (('nn_a',), ('nn_a', 'nn_b', 'nn_c')):
            text = smoke_source(names)
            generated = [item for item in policy.functions(text, source=SD3) if item.get('generated_by')]
            self.assertEqual({item['generated_fqn'] for item in generated}, {'trainer_cuda_smoke::' + name for name in names})
            self.assert_guard(SD3, text, rows(SD3, 'trainer_cuda_smoke', names), True)

    def test_alpha_invocation_name_ignore_body_and_definition_mutations_reject(self):
        text = alpha_source()
        variants = [text.replace('nn_a, metadata_nn_a =>', 'other_name, metadata_nn_a =>'),
                    text.replace('MODEL_INFERENCE_BODY_COMPLETED turbo_lora::nn_a', 'MODEL_INFERENCE_BODY_COMPLETED wrong::nn_a'),
                    text.replace('#[ignore = "model inference; explicit model workflow only"]', ''),
                    text.replace('assert_twin_equivalence($spelling);', ''),
                    text.replace('model-inference-proof', 'not-the-proof-feature'),
                    text.replace('$name:ident', '$name:expr'),
                    text.replace('const GENERATED_SPELLINGS: &[AlphaSpelling] = &[$($spelling),+];', ''),
                    text.replace('assert_alpha_resolves_before_the_split($spelling)', 'assert_twin_equivalence($spelling)'),
                    text.replace('per_alpha_spelling! {\nnn_a,', 'per_alpha_spelling! {\nnot valid arguments;\nnn_a,')]
        for variant in variants:
            with self.subTest(source=variant[:90]):
                self.assert_guard(H3, variant, rows(H3, 'turbo_lora'), False)

    def test_smoke_invocation_name_ignore_body_and_definition_mutations_reject(self):
        text = smoke_source()
        variants = [text.replace('smoke!(nn_a,', 'smoke!(wrong_name,'),
                    text.replace('MODEL_INFERENCE_BODY_COMPLETED trainer_cuda_smoke::nn_a', 'MODEL_INFERENCE_BODY_COMPLETED other::nn_a'),
                    text.replace('#[ignore = "needs real SD3.5 weights and explicitly scheduled CUDA"]', ''),
                    text.replace('run($id, $env, $kind);', ''),
                    text.replace('$id:literal', '$id:expr'),
                    text.replace('"FIXTURE_ENV", NetworkType::Lora', 'unknown_argument, NetworkType::Lora'),
                    text.replace('macro_rules! smoke', 'macro_rules! different'),
                    text.replace('_ => panic!("undeclared generated model-proof case"),', '_ => (),')]
        for variant in variants:
            with self.subTest(source=variant[:90]):
                self.assert_guard(SD3, variant, rows(SD3, 'trainer_cuda_smoke'), False)

    def test_unknown_macro_source_or_row_alias_cannot_get_function_exemption(self):
        source = 'crates/unknown/tests/macro.rs'
        self.assert_guard(source, smoke_source(), rows(source, 'trainer_cuda_smoke'), False)
        bindings = rows(SD3, 'trainer_cuda_smoke')
        bindings[0]['source_function'] = 'nn_b'
        self.assert_guard(SD3, smoke_source(), bindings, False)
        bindings = rows(SD3, 'trainer_cuda_smoke')
        bindings[0]['body_completion_marker'] = 'MODEL_INFERENCE_BODY_COMPLETED trainer_cuda_smoke::nn_b'
        self.assert_guard(SD3, smoke_source(), bindings, False)
        self.assert_guard(H3, alpha_source(), rows(H3, 'turbo_lora') + rows(H3, 'turbo_lora', ('metadata_nn_a',)), False)

    def test_actual_frozen_source_forms_are_recognized_without_executing_them(self):
        for source in (H3, SD3):
            text = (ROOT / source).read_text(encoding="utf-8")
            generated = [item for item in policy.functions(text, source=source) if item.get('generated_by')]
            self.assertTrue(generated)
            self.assertTrue(all(item['generated_fqn'].endswith('::' + item['name']) for item in generated if item['ignored']))


if __name__ == '__main__':
    unittest.main()
