"""Bind the terminal stack hook to its CPU-tested held-out edit protocol."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "crates/media/mlx-gen/mlx-gen-qwen-image-2-1/tests/lora_real_weights.rs"


def protocol_errors(source):
    body = source.split("fn stacked_adapters_apply_with_independent_weights()", 1)[1].split("// ── 4.", 1)[0]
    errors = []
    compact = "".join(body.split())
    fragments = [
        'stack_protocol::request(to_image(edit_source(99,RENDER_EDGE)),to_image(edit_key(TRAIN_EDGE)),)',
        'collect_references(&request)', 'load_vision_config(&snapshot())',
        'reference::reference_fit((image.width,image.height),index,&vision,)',
        'stack_protocol::reference_receipt(&references,&fitted_sizes)',
        'assert_eq!(request.memory_reference_count(),2)',
        'fitted_sizes.iter().all(|size|*size==(1024,1024))',
        'request_transient_budget_bytes(request.width,request.height,',
        'request.memory_reference_count(),false,tile_edge,)',
        'letstacks:[(&str,Vec<AdapterSpec>);6]',
        '"renderCount":6', '"stackMovementFloor":STACK_MOVES_FLOOR',
        'f["route"]=json!("two_reference_edit")',
        'f["referenceCount"]=json!(request.memory_reference_count())',
        'f["inputs"]=inputs.clone()', 'f["transientBudgetBytes"]=json!(transient)',
        'f["allocatorMemoryLimitBytes"]=json!(allocator_resident.saturating_add(transient))',
        '&json!({"protocol":protocol,"renders":facts,"comparisons":comparisons})',
        'comparisons["t2i_1_edit_0==t2i_1"].as_bool().unwrap()',
        'comparisons["t2i_0_edit_1==edit_1"].as_bool().unwrap()',
        'moved>=STACK_MOVES_FLOOR', 'assert_sane(label,img)',
    ]
    for fragment in fragments:
        if fragment not in compact:
            errors.append(fragment)
    if 'letrequest=t2i_request()' in compact:
        errors.append("stack fell back to T2I")
    if body.index('write_json(&out, "two-reference-stack-protocol"') > body.index('for (label, stack) in stacks'):
        errors.append("protocol must precede weight loads and renders")
    for fragment in ['const STACK_MOVES_FLOOR: f64 = 0.5;', 'const NON_DEGENERATE_STD: f64 = 8.0;',
                     'static_rows <= 0.25', 'const EDIT_GAIN_FLOOR: f64 = 1.0;',
                     'to_image(edit_key(RENDER_EDGE))']:
        if fragment not in source:
            errors.append(fragment)
    return errors


class StackProtocolTests(unittest.TestCase):
    def test_stack_uses_fixed_ordered_inputs_and_request_specific_forecast(self):
        self.assertEqual(protocol_errors(SOURCE.read_text(encoding="utf8")), [])

    def test_old_route_wrong_key_underpriced_forecast_and_false_receipts_fail(self):
        source = SOURCE.read_text(encoding="utf8")
        mutations = [
            ('stack_protocol::request(', 'wrong_request('),
            ('to_image(edit_key(TRAIN_EDGE)),', 'to_image(edit_key(RENDER_EDGE)),'),
            ('request.memory_reference_count(),\n            false,', '0,\n            false,'),
            ('f["route"] = json!("two_reference_edit")', 'f["route"] = json!("t2i")'),
            ('f["inputs"] = inputs.clone()', 'f["inputs"] = json!({})'),
            ('"renderCount":6', '"renderCount":8'),
            ('const STACK_MOVES_FLOOR: f64 = 0.5;', 'const STACK_MOVES_FLOOR: f64 = 0.1;'),
            ('moved >= STACK_MOVES_FLOOR', 'moved >= 0.0'),
        ]
        for old, new in mutations:
            with self.subTest(mutation=old):
                self.assertIn(old, source)
                self.assertTrue(protocol_errors(source.replace(old, new)))


if __name__ == "__main__":
    unittest.main()
