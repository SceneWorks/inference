#!/usr/bin/env python3
"""Synthetic CPU DMD oracle from exact frozen reference wrapper/recurrence AST.

Tiny dimensions/rank/codes/Q/replacements are synthetic test data, not author model hparams.
Numerical definitions are unmodified author transformer and frozen evaluated reference wrapper.
Run python3 -B with PYTHONDONTWRITEBYTECODE=1. No real weights or accelerator access.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
import argparse, ast, copy, hashlib, json, math, signal, time
from pathlib import Path
signal.alarm(120)
started=time.monotonic()
import torch
from torch import nn
from torch.nn import functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--author-source',type=Path,default=HERE/'frozen_author_pipeline.py')
parser.add_argument('--reference-source',type=Path,required=True,help='Exact external SceneWorks evaluation reference.py; AGPL source is not bundled here')
parser.add_argument('--base-fixture',type=Path,default=HERE/'fixture.json')
parser.add_argument('--output-dir',type=Path,default=HERE)
args=parser.parse_args()
OUT=args.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
source=args.author_source.read_bytes();ref=args.reference_source.read_bytes()
source_hash=hashlib.sha256(source).hexdigest();ref_hash=hashlib.sha256(ref).hexdigest()
assert ref_hash=='6f202ab34de607fb7a1272cf50fc08e89f9a0ea21529b712c86f214dc467741a'
assert source_hash=='1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31'
base=json.loads(args.base_fixture.read_bytes());assert base['source_sha256']==source_hash
names=['_RMSNorm','_situ','_validate_skip_layers','_rotate','_attention','_TextBlock','_TextProjection','_Block','_RoPE','MOSSImageTransformer']
tree=ast.parse(source)
selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
assert [n.name for n in selected]==names
ns=dict(torch=torch,nn=nn,F=F,ModelMixin=ModelMixin,ConfigMixin=ConfigMixin,register_to_config=register_to_config,Transformer2DModelOutput=Transformer2DModelOutput)
exec(compile(ast.Module(body=selected,type_ignores=[]),str(args.author_source),'exec'),ns)
rt=ast.parse(ref)
loop=next(n for n in rt.body if isinstance(n,ast.FunctionDef) and n.name=='dmd_loop')
install=next(n for n in rt.body if isinstance(n,ast.FunctionDef) and n.name=='install_adapter')
wrapper=next(n for n in install.body if isinstance(n,ast.ClassDef) and n.name=='Unmerged')
exec(compile(ast.Module(body=[loop,wrapper],type_ignores=[]),str(args.reference_source),'exec'),ns)
torch.set_default_device('cpu');torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

def restore(v):
 return torch.tensor(v['values'],device='cpu',dtype=torch.float32).reshape(v['shape'])
def tensor(t):
 assert t.device.type=='cpu'
 return dict(shape=list(t.shape),dtype=str(t.dtype).replace('torch.',''),values=t.detach().float().flatten().tolist())
model=ns['MOSSImageTransformer'](**base['config']).to(device='cpu',dtype=torch.float32).eval()
model.load_state_dict({k:restore(v) for k,v in base['weights'].items()},strict=True)
model.to(dtype=torch.bfloat16)
replacement_keys=['time_embedding.0.weight','time_embedding.0.bias','time_embedding.2.weight','time_embedding.2.bias','shared_block_modulation.bias','video_head_modulation.bias']
replacements={}
for ordinal,key in enumerate(replacement_keys):
 original=restore(base['weights'][key]);flat=original.flatten()
 delta=torch.tensor([0.08*math.sin((i+1)*0.239+(ordinal+1)*0.57) for i in range(flat.numel())],device='cpu',dtype=torch.float32)
 replacements[key]=(flat+delta).reshape(original.shape)
# Same exact FP32 island installation order as install_adapter, with synthetic replacement values.
for name in ('time_embedding','shared_block_modulation','video_head_modulation'):
 getattr(model,name).float()
params=dict(model.named_parameters())
with torch.no_grad():
 for key,value in replacements.items():params[key].copy_(value)
rank=3;alpha=3.0
adapters={}
for index,layer in enumerate(model.layers):
 for ordinal,name in enumerate(('q','k','v','o','ffn_up_gate','ffn_down')):
  target=layer.proj[name]
  down=torch.tensor([0.31*math.sin((i+1)*0.173+(index+1)*0.43+ordinal*0.19) for i in range(rank*target.in_features)],device='cpu',dtype=torch.float32).reshape(rank,target.in_features)
  up=torch.tensor([0.29*math.cos((i+1)*0.131+(index+1)*0.37+ordinal*0.11) for i in range(target.out_features*rank)],device='cpu',dtype=torch.float32).reshape(target.out_features,rank)
  prefix=f'layers.{index}.proj.{name}'
  adapters[prefix]=dict(down=tensor(down),up=tensor(up),alpha=alpha)
  layer.proj[name]=ns['Unmerged'](target,down,up,alpha)

class IncorrectCastDown(ns['Unmerged']):
 def forward(self,x):return super().forward(x).to(self.backbone.weight.dtype)

wrong_cast=copy.deepcopy(model)
wrong_merge=copy.deepcopy(model)
for candidate,kind in ((wrong_cast,'cast_down'),(wrong_merge,'merged_bf16')):
 for layer in candidate.layers:
  for name in ('q','k','v','o','ffn_up_gate','ffn_down'):
   current=layer.proj[name]
   if kind=='cast_down':layer.proj[name]=IncorrectCastDown(current.backbone,current.down,current.up,alpha)
   else:
    merged=nn.Linear(current.backbone.in_features,current.backbone.out_features,bias=False,device='cpu',dtype=torch.bfloat16)
    with torch.no_grad():merged.weight.copy_((current.backbone.weight.float()+current.scale*(current.up@current.down)).to(torch.bfloat16))
    layer.proj[name]=merged
noise=restore(base['inputs']['hidden_states'])[:1].contiguous()
context=restore(base['inputs']['encoder_hidden_states'])[:1].contiguous()
mask=restore(base['inputs']['encoder_attention_mask'])[:1].bool().contiguous()
q=torch.tensor([[0.37,0.21,-0.04,0.13],[-0.19,0.42,0.09,-0.08],[0.03,0.17,0.31,-0.14],[0.11,-0.05,0.23,0.29]],device='cpu',dtype=torch.float32)
codes=torch.tensor([831.25,251.5,12.75],device='cpu',dtype=torch.float32)
runs={}
for name,candidate in (('correct_widened',model),('incorrect_cast_down',wrong_cast),('incorrect_merged_bf16',wrong_merge)):
 trace=[];islands={};handles=[]
 for island in ('time_embedding','shared_block_modulation','video_head_modulation'):
  def hook(module,args,result,name=island):
   assert result.dtype==torch.float32
   islands.setdefault(name,[]).append(tensor(result))
  handles.append(candidate.get_submodule(island).register_forward_hook(hook))
 def observer(index,x,v,x0):trace.append(dict(index=index,input=tensor(x),velocity=tensor(v),x0=tensor(x0)))
 with torch.inference_mode():
  final=ns['dmd_loop'](noise,lambda x:torch.einsum('oc,bchw->bohw',q,x),codes,lambda x,code:candidate(x,code,context,mask).sample.float(),steps=3,observer=observer)
 for handle in handles:handle.remove()
 runs[name]=dict(steps=trace,final=tensor(final),fp32_islands=islands)
# A focused projection fixture isolates residual widening even if downstream BF16 masks a delta.
projection=model.layers[0].proj['o']
probe=torch.tensor([0.7*math.sin((i+1)*0.31) for i in range(2*16)],device='cpu',dtype=torch.float32).reshape(1,2,16)
with torch.inference_mode():
 leaf_correct=projection(probe);leaf_bad=wrong_cast.layers[0].proj['o'](probe);leaf_merged=wrong_merge.layers[0].proj['o'](probe.to(torch.bfloat16))

def maximum_difference(a,b):return max(abs(x-y) for x,y in zip(a['values'],b['values']))
differences={name:dict(final_max_abs=maximum_difference(runs['correct_widened']['final'],run['final']),step_velocity_max_abs=[maximum_difference(a['velocity'],b['velocity']) for a,b in zip(runs['correct_widened']['steps'],run['steps'])]) for name,run in runs.items() if name!='correct_widened'}
assert all(d['final_max_abs']>0 for d in differences.values()),differences
assert leaf_correct.dtype==torch.float32 and leaf_bad.dtype==torch.bfloat16
fixture=dict(schema=1,device='cpu',torch_version=torch.__version__,source_sha256=source_hash,reference_sha256=ref_hash,base_fixture_sha256=hashlib.sha256(args.base_fixture.read_bytes()).hexdigest(),config=base['config'],synthetic_hyperparameters=True,rank=rank,alpha=alpha,base_weights=base['weights'],replacement_tensors={k:tensor(v) for k,v in replacements.items()},adapters=adapters,noise=tensor(noise),conditioning=tensor(context),mask=tensor(mask),mix_matrix=tensor(q),step_codes=tensor(codes),runs=runs,projection_probe=dict(module='layers.0.proj.o',input=tensor(probe),correct=tensor(leaf_correct),incorrect_cast_down=tensor(leaf_bad),incorrect_merged_bf16=tensor(leaf_merged)),incorrect_path_differences=differences,extraction='Unmodified frozen dmd_loop and install_adapter nested Unmerged class AST; exact author transformer AST. Tiny rank/dimensions/replacement contents/codes/Q are synthetic. Whole install_adapter validates production rank128/42blocks so tiny installation manually repeats its FP32 island/install order.',elapsed_seconds=time.monotonic()-started)
path=OUT/'dmd_fixture.json';path.write_text(json.dumps(fixture,separators=(',',':'))+'\n')
receipt=dict(status='PASS_CPU_SYNTHETIC_DMD_WIDENING_AND_RECURRENCE',fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_sha256=source_hash,reference_sha256=ref_hash,bytes=path.stat().st_size,elapsed_seconds=fixture['elapsed_seconds'],torch_version=torch.__version__,device='cpu',real_weights=False,image_generation=False,steps=3,rank=rank,incorrect_path_differences=differences,projection_output_dtypes=dict(correct=str(leaf_correct.dtype),incorrect_cast_down=str(leaf_bad.dtype)),cli_source_arguments=['--author-source','--reference-source','--base-fixture','--output-dir'])
(OUT/'dmd_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2));signal.alarm(0)
