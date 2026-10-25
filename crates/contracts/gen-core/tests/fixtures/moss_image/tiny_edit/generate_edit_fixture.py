#!/usr/bin/env python3
"""Frozen upstream CPU synthetic oracle; source files stay external to inference."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
import ast, copy, hashlib, json, math, signal, time, types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

signal.alarm(120)
started = time.monotonic()
import torch
from torch import nn
from torch.nn import functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

HERE = Path(__file__).resolve().parent
OUT = HERE / 'tiny_edit'
OUT.mkdir(exist_ok=True)
CONTRACT = 'ref-prefix-only-t0__target-text-original-moss-v1'
sources = {
    'author': (HERE/'official-sft/pipeline_moss_image.py', '1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31'),
    'edit': (HERE/'ComfyUI-Moss-Image-Edit/moss_runtime/model.py', 'f0f5d3011c1f689547d290231a8a50a1c4d16f58c561556b53db60a0edbe9885'),
    'lora': (HERE/'ComfyUI-Moss-Image-Edit/moss_runtime/weights.py', '692f9df723323a95cdb24bbf612c7b6d2a98709cefcbee479e2d7dc57aab0705'),
}
module = types.ModuleType('_moss_frozen_edit_fixture')
sys.modules[module.__name__] = module
ns = module.__dict__
ns.update(dict(torch=torch,nn=nn,F=F,ModelMixin=ModelMixin,ConfigMixin=ConfigMixin,
    register_to_config=register_to_config,Transformer2DModelOutput=Transformer2DModelOutput,
    FlowMatchEulerDiscreteScheduler=FlowMatchEulerDiscreteScheduler,
    dataclass=dataclass,field=field,Any=Any,Sequence=Sequence,EDIT_CONTRACT=CONTRACT,math=math))
def extract(name, names):
    path, digest = sources[name]
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest, (name,'source drift')
    tree = ast.parse(data, filename=str(path))
    nodes = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)

extract('author',['_RMSNorm','_situ','_validate_skip_layers','_rotate','_attention','_TextBlock','_TextProjection','_Block','_RoPE','MOSSImageTransformer','MOSSImageScheduler'])
extract('edit',['_tensor_stamp','MossEditCache','MossEditModel'])
extract('lora',['_validate_options','LoRALinear'])
torch.set_default_device('cpu')
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
cfg = dict(dim=16,ffn_dim=24,num_heads=2,num_layers=2,text_dim=12,freq_dim=8,latent_channels=4,rope_base=10000.0)
model = ns['MOSSImageTransformer'](**cfg).to(device='cpu',dtype=torch.float32).eval()
with torch.no_grad():
    for ordinal,(name,p) in enumerate(model.named_parameters()):
        if name.endswith(('norm.weight','norm1.weight','norm2.weight')):
            values=[1.0+0.03*math.sin((i+1)*(ordinal+1)*0.071) for i in range(p.numel())]
        elif name.endswith('.bias'):
            values=[0.07*math.cos((i+1)*(ordinal+1)*0.091) for i in range(p.numel())]
        else:
            div=math.sqrt(p.shape[-1] if p.ndim>1 else p.numel())
            values=[0.65*math.sin((i+1)*0.117+(ordinal+1)*0.319)/div for i in range(p.numel())]
        p.copy_(torch.tensor(values,device='cpu',dtype=torch.float32).reshape(p.shape))
model.to(dtype=torch.bfloat16)
def tensor(t):
    assert t.device.type=='cpu' and torch.isfinite(t).all()
    return dict(shape=list(t.shape),dtype=str(t.dtype).removeprefix('torch.'),values=t.detach().float().flatten().tolist())
weights = {k:tensor(v) for k,v in model.state_dict().items()}
adapters={}
rank=2
with torch.no_grad():
    for layer in range(cfg['num_layers']):
        for ordinal,name in enumerate(['q','k','v','o','ffn_up_gate','ffn_down']):
            prefix=f'layers.{layer}.proj.{name}'
            base=model.get_submodule(prefix)
            wrapped=ns['LoRALinear'](base,rank,rank,0.0,torch.bfloat16)
            for key,p in [('lora_A',wrapped.lora_A),('lora_B',wrapped.lora_B)]:
                phase=(layer+1)*0.31+(ordinal+1)*0.23+(0.17 if key=='lora_B' else 0)
                values=[0.45*math.sin((i+1)*0.191+phase) for i in range(p.numel())]
                p.copy_(torch.tensor(values,dtype=torch.float32,device='cpu').reshape(p.shape))
                adapters[f'{prefix}.{key}']=tensor(p)
            parent,_,leaf=prefix.rpartition('.')
            setattr(model.get_submodule(parent),leaf,wrapped)
model.requires_grad_(False).eval()
editor=ns['MossEditModel'](model).eval()

def wave(shape, phase, scale=0.6):
    count=math.prod(shape)
    return torch.tensor([scale*math.sin((i+1)*0.19+phase) for i in range(count)],device='cpu',dtype=torch.float32).reshape(shape)
hidden=wave((1,4,2,3),0.0)
refs=[wave((1,4,h,w),0.37*(i+1)) for i,(h,w) in enumerate([(2,2),(1,3),(3,1),(1,2)])]
positive=wave((1,4,12),0.43,0.8)
negative=wave((1,3,12),1.13,0.8)
pos_mask=torch.tensor([[1,0,1,1]],device='cpu',dtype=torch.bool)
neg_mask=torch.tensor([[1,1,0]],device='cpu',dtype=torch.bool)
t=0.73
def predict(ed, x, time, rr, ctx=positive, mask=pos_mask, cache=None):
    return ed(x,time,ctx,reference_latents=rr,encoder_attention_mask=mask,cache=cache)
def maxerr(a,b): return (a.float()-b.float()).abs().max().item()
fixture=dict(schema=1,device='cpu',real_weights=False,config=cfg,rank=rank,alpha=rank,
    source_sha256={k:v[1] for k,v in sources.items()},weights=weights,adapter=adapters,
    inputs=dict(hidden_states=tensor(hidden),references=[tensor(r) for r in refs],positive=tensor(positive),negative=tensor(negative),positive_mask=tensor(pos_mask),negative_mask=tensor(neg_mask),normalized_timestep=t),cases=[])

with torch.no_grad():
    for count in [1,4]:
        rr=refs[:count]
        cache=editor.prepare_cache(rr)
        uncached=predict(editor,hidden,t,rr)
        cached=predict(editor,hidden,t,rr,cache=cache)
        assert maxerr(uncached,cached)==0.0
        neg=predict(editor,hidden,t,rr,negative,neg_mask,cache)
        tokens,rope=editor._reference_inputs(tuple(rr))
        mod0,head0=editor._time(0.0,hidden.device)
        modt,headt=editor._time(t,hidden.device)
        # Wrong controls alter only the named contract. The correct expected outputs
        # above execute unchanged frozen upstream AST definitions.
        wrong_time=ns['MossEditModel'](copy.deepcopy(model)).eval()
        original_time=wrong_time._time
        wrong_time._time=lambda x,d: original_time(t if float(torch.as_tensor(x))==0.0 else x,d)
        bad_time=predict(wrong_time,hidden,t,rr)
        no_ref=ns['MossEditModel'](copy.deepcopy(model)).eval()
        original_body=no_ref._body_block
        no_ref._body_block=lambda layer,body,rope,mod,reference_kv=None,text_length=0: original_body(layer,body,rope,mod,None,text_length)
        bad_body=predict(no_ref,hidden,t,rr)
        case=dict(reference_count=count,cached=tensor(cached),uncached=tensor(uncached),negative_cached=tensor(neg),
            reference_tokens=tensor(tokens),reference_rope=tensor(rope),reference_modulation=tensor(mod0),reference_head=tensor(head0),body_modulation=tensor(modt),body_head=tensor(headt),
            reference_kv=[dict(key=tensor(k),value=tensor(v)) for k,v in cache.keys_values],
            controls=dict(reference_uses_body_time=dict(output=tensor(bad_time),max_abs_difference=maxerr(cached,bad_time)),body_cannot_read_references=dict(output=tensor(bad_body),max_abs_difference=maxerr(cached,bad_body))))
        if count==4:
            swapped=predict(editor,hidden,t,list(reversed(rr)))
            case['controls']['reversed_reference_order']=dict(output=tensor(swapped),max_abs_difference=maxerr(cached,swapped))
        # Real author scheduler; explicit initial noise avoids cross-runtime PRNG assumptions.
        scheduler=ns['MOSSImageScheduler'](shift=5.0,training_shift=1.0)
        scheduler.set_timesteps(2,device='cpu')
        x=hidden.clone()
        steps=[]
        for timecode in scheduler.timesteps:
            norm=timecode.float().reshape(1)/1000
            c=predict(editor,x,norm,rr,cache=cache).float()
            u=predict(editor,x,norm,rr,negative,neg_mask,cache).float()
            velocity=u+4.0*(c-u)
            nextx=scheduler.step(velocity,timecode,x,return_dict=False)[0].float()
            steps.append(dict(timestep=float(timecode),normalized_timestep=float(norm.item()),input=tensor(x),conditional=tensor(c),unconditional=tensor(u),guided=tensor(velocity),output=tensor(nextx)))
            x=nextx
        case['sampling']=dict(cfg=4.0,inference_shift=5.0,training_shift=1.0,sigmas=tensor(scheduler.sigmas),steps=steps,final=tensor(x))
        fixture['cases'].append(case)

    # Projection rounding probe distinguishes BF16 edit from FP32 DMD-like residual.
    projection=model.layers[0].proj['q']
    probe=wave((1,3,16),0.17,0.7)
    expected=projection(probe.to(torch.bfloat16))
    original=projection.base(probe.to(torch.bfloat16))
    fp32_residual=F.linear(F.linear(probe.to(torch.bfloat16).float(),projection.lora_A.float()),projection.lora_B.float())
    wrong_fp32=original.float()+fp32_residual
    merged=F.linear(probe.to(torch.bfloat16),(projection.base.weight.float()+projection.lora_B.float()@projection.lora_A.float()).bfloat16())
    fixture['projection_probe']=dict(input=tensor(probe.to(torch.bfloat16)),expected=tensor(expected),base=tensor(original),wrong_fp32_residual=tensor(wrong_fp32),wrong_merged_bf16=tensor(merged),fp32_max_abs_difference=maxerr(expected,wrong_fp32),merged_max_abs_difference=maxerr(expected,merged))

fixture['elapsed_seconds']=time.monotonic()-started
path=OUT/'fixture.json'
path.write_text(json.dumps(fixture,separators=(',',':'))+'\n')
receipt=dict(status='PASS_CPU_SYNTHETIC_FROZEN_EDIT',fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_sha256=fixture['source_sha256'],torch_version=torch.__version__,device='cpu',real_weights=False,config=cfg,rank=rank,cases=[dict(reference_count=c['reference_count'],cached_uncached_max_abs=0.0,controls={k:v['max_abs_difference'] for k,v in c['controls'].items()}) for c in fixture['cases']],projection_controls={k:v for k,v in fixture['projection_probe'].items() if k.endswith('difference')},elapsed_seconds=fixture['elapsed_seconds'],bytes=path.stat().st_size)
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
signal.alarm(0)
