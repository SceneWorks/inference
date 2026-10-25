#!/usr/bin/env python3
"""Frozen source full sampler with exactly BF16-represented model values cast to FP32."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
nsx={'__file__':str(P/'generate.py'),'__name__':'_sampler_definitions'}
code=(P/'generate.py').read_text()
assert hashlib.sha256(code.encode()).hexdigest()=='b1079205b9d4676ef9d7bcd8c0fb931a3d4e9b43c603cb26467563156a6045e3'
exec(compile(code.split('\nresult=dict(schema=1')[0],str(P/'generate.py'),'exec'),nsx)
globals().update({k:v for k,v in nsx.items() if k not in ('__file__','__name__','P')})
fmodel=copy.deepcopy(model).float().eval()
assert all(torch.equal(v.float(),fmodel.state_dict()[k]) for k,v in model.state_dict().items())
feditor=source['MossEditModel'](fmodel).eval()
result=dict(schema=1,device='cpu',native_data=False,real_weights=False,parent_fixture_sha256=hashlib.sha256(parent_bytes).hexdigest(),source_sha256=original['source_sha256'],policy='Every BF16 base/adapter model value cast exactly to FP32, unchanged source graph and scheduler, FP32 inputs already represented in parent. No native data.',cases=[])
with torch.no_grad():
 for count in [1,4]:
  rr=refs[:count];good=run(feditor,rr);controls={}
  wrong=source['MossEditModel'](copy.deepcopy(fmodel)).eval()
  bad=run(wrong,rr,mutation='reference_uses_body_time');controls['reference_uses_body_time']=dict(max_abs_difference=dist(as_t(good['final']),as_t(bad['final'])),**bad)
  wrong=source['MossEditModel'](copy.deepcopy(fmodel)).eval();ob=wrong._body_block
  wrong._body_block=lambda layer,body,rope,mod,reference_kv=None,text_length=0:ob(layer,body,rope,mod,None,text_length)
  bad=run(wrong,rr);controls['body_cannot_read_references']=dict(max_abs_difference=dist(as_t(good['final']),as_t(bad['final'])),**bad)
  if count==4:
   bad=run(feditor,list(reversed(rr)));controls['reversed_reference_order']=dict(max_abs_difference=dist(as_t(good['final']),as_t(bad['final'])),**bad)
  result['cases'].append(dict(reference_count=count,correct=good,controls=controls))
path=P/'fp32_sampler.json';path.write_text(json.dumps(result,separators=(',',':'))+'\n')
receipt=dict(status='PASS_SOURCE_FP32_REPRESENTED_BF16_FULL_SAMPLER',fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_sha256=result['source_sha256'],model_values_exactly_parent_bf16=True,parent_unchanged=PARENT.read_bytes()==parent_bytes,cases=[dict(reference_count=c['reference_count'],controls={k:v['max_abs_difference'] for k,v in c['controls'].items()},manual_euler_max_abs=max(s['manual_euler_max_abs'] for s in c['correct']['steps'])) for c in result['cases']])
(P/'fp32_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
