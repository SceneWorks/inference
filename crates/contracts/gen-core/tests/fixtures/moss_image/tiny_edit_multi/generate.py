#!/usr/bin/env python3
"""Add independent 2/3-reference cases; never write historical fixtures or native data."""
import hashlib
import json
import inspect
import sys
from pathlib import Path

out = Path(__file__).resolve().parent
upstream = out.parent
program_path = upstream / 'tiny_edit_v2/generate.py'
program = program_path.read_text()
assert hashlib.sha256(program.encode()).hexdigest() == 'd464fcac7bf3b10989ec43f71f77a6f38340655ad59f618adc4a2745c0ed61f7'
parent = upstream / 'tiny_edit_v2/fixture.json'
frozen_hashes = {
    'tiny_edit/fixture.json': '4a326adbd707bf11ce45f0e0d62d315690e295e067bac12f3c486e932f51475f',
    'tiny_edit_v1_time_probe/fixture.json': 'c2af600c1c966ced4f8485081fc469d4b206a91205db49a7f5400e66204c5985',
    'tiny_edit_v2/fixture.json': '398b6b2260c15b670655700ed552d4ee935d493af5c9f2a72c85570cd7d121f1',
    'tiny_edit_v4_sampler_analysis/fp32_sampler.json': '905bef23f86f6bf08e116c716a03051ebbfa4f886d7b3dd9786e68e0d064cf56',
    'tiny_edit_v4_sampler_analysis/analysis.json': '1e89a8c037569b988a89703c86894c6907da8c9097c92176d94b461663dcf6da',
}
def verify_frozen():
    for relative, digest in frozen_hashes.items():
        assert hashlib.sha256((upstream / relative).read_bytes()).hexdigest() == digest, relative
verify_frozen()
parent_bytes = parent.read_bytes()
assert hashlib.sha256(parent_bytes).hexdigest() == '398b6b2260c15b670655700ed552d4ee935d493af5c9f2a72c85570cd7d121f1'
# Only the capture case selection/output path changes. Frozen model/LoRA/source
# extraction, weight construction, source attention and scheduler math stay exact.
for old, new in [("OUT = HERE / 'tiny_edit_v2'", "OUT = HERE / 'tiny_edit_multi'"),
                 ('for count in [1,4]:', 'for count in [2,3]:'),
                 ('if count==4:', 'if count>=2:')]:
    assert program.count(old) == 1
    program = program.replace(old, new)
original = json.loads(parent_bytes)
# Strict represented-weight sampler additions use the unchanged source graph.
source_program = upstream / 'tiny_edit_v4_sampler_analysis/generate.py'
code = source_program.read_text()
assert hashlib.sha256(code.encode()).hexdigest() == 'b1079205b9d4676ef9d7bcd8c0fb931a3d4e9b43c603cb26467563156a6045e3'
ns = {'__file__': str(source_program), '__name__': '_sampler_definitions'}
exec(compile(code.split('\nresult=dict(schema=1')[0], str(source_program), 'exec'), ns)
torch, copy = ns['torch'], ns['copy']
source, tensor, run, refs = ns['source'], ns['tensor'], ns['run'], ns['refs']
# Reproduce original BF16 source trajectories exactly before accepting this environment.
with torch.no_grad():
    for case in original['cases']:
        replay = run(ns['editor'], refs[:case['reference_count']], case['sampling'])
        assert all(value == 0 for step in replay['steps'] for value in step['source_replay_errors'].values())
        assert replay['final'] == case['sampling']['final']
model = ns['model'].float().eval()
feditor = source['MossEditModel'](model).eval()
strict_original = json.loads((upstream / 'tiny_edit_v4_sampler_analysis/fp32_sampler.json').read_bytes())
with torch.no_grad():
    for case in strict_original['cases']:
        replay = run(feditor, refs[:case['reference_count']])
        assert replay == case['correct'], 'original strict F32 source trajectory drift'
verify_frozen()
result = dict(schema=1, device='cpu', native_data=False, real_weights=False,
              parent_fixture_sha256=hashlib.sha256(parent_bytes).hexdigest(),
              source_sha256=original['source_sha256'], cases=[])
with torch.no_grad():
    for count in [2, 3]:
        rr = refs[:count]
        good = run(feditor, rr)
        controls = {}
        wrong = source['MossEditModel'](copy.deepcopy(model)).eval()
        bad = run(wrong, rr, mutation='reference_uses_body_time')
        controls['reference_uses_body_time'] = bad
        wrong = source['MossEditModel'](copy.deepcopy(model)).eval()
        body = wrong._body_block
        wrong._body_block = lambda layer, x, rope, mod, reference_kv=None, text_length=0: body(layer, x, rope, mod, None, text_length)
        controls['body_cannot_read_references'] = run(wrong, rr)
        controls['reversed_reference_order'] = run(feditor, list(reversed(rr)))
        result['cases'].append(dict(reference_count=count, correct=good, controls=controls))
namespace = {'__file__': str(out / 'generate.py'), '__name__': '_source_multi'}
exec(compile(program, str(program_path), 'exec'), namespace)
assert parent.read_bytes() == parent_bytes
fixture = json.loads((out / 'fixture.json').read_bytes())
original = json.loads(parent_bytes)
assert fixture['weights'] == original['weights']
assert fixture['adapter'] == original['adapter']
assert fixture['inputs'] == original['inputs']
(out / 'fp32_sampler.json').write_text(json.dumps(result, separators=(',', ':')) + '\n')
assert parent.read_bytes() == parent_bytes
verify_frozen()
receipt = dict(python_executable=sys.executable, python_version=sys.version,
               torch_version=torch.__version__, diffusers_version=ns['diffusers'].__version__,
               scheduler_sha256=hashlib.sha256(Path(inspect.getfile(ns['FlowMatchEulerDiscreteScheduler'])).read_bytes()).hexdigest(),
               frozen_artifacts_sha256=frozen_hashes,
               original_bf16_and_f32_replay='exact 1/4 reference trajectories',
               generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               parent_unchanged=True, native_data=False, real_weights=False,
               fixture_sha256=hashlib.sha256((out / 'fixture.json').read_bytes()).hexdigest(),
               fp32_sampler_sha256=hashlib.sha256((out / 'fp32_sampler.json').read_bytes()).hexdigest(),
               source_sha256=original['source_sha256'])
(out / 'multi_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
