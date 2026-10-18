#!/usr/bin/env python3
"""Bounded CPU author-class oracle; no installed packages, real weights, or accelerator access.

The exact frozen class/function AST definitions run unchanged. Import-only dependencies for
pipeline loading are omitted because local transformers is unavailable; actual installed
Torch and Diffusers ModelMixin/ConfigMixin/register_to_config/output types are retained.
Run with python3 -B and PYTHONDONTWRITEBYTECODE=1.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
import argparse, ast, copy, hashlib, json, math, signal, time
from pathlib import Path

signal.alarm(120)
started = time.monotonic()
import torch
from torch import nn
from torch.nn import functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--author-source', type=Path, default=HERE/'frozen_author_pipeline.py')
parser.add_argument('--output-dir', type=Path, default=HERE)
args = parser.parse_args()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = args.author_source.resolve()
EXPECTED = '1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31'
source = SOURCE.read_bytes()
assert hashlib.sha256(source).hexdigest() == EXPECTED
names = ['_RMSNorm','_situ','_validate_skip_layers','_rotate','_attention','_TextBlock','_TextProjection','_Block','_RoPE','MOSSImageTransformer']
tree = ast.parse(source, filename=str(SOURCE))
selected = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
assert [n.name for n in selected] == names
namespace = dict(torch=torch,nn=nn,F=F,ModelMixin=ModelMixin,ConfigMixin=ConfigMixin,register_to_config=register_to_config,Transformer2DModelOutput=Transformer2DModelOutput)
exec(compile(ast.Module(body=selected,type_ignores=[]),str(SOURCE),'exec'),namespace)
torch.set_default_device('cpu')
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
cfg = dict(dim=16,ffn_dim=24,num_heads=2,num_layers=3,text_dim=12,freq_dim=8,latent_channels=4,rope_base=10000.0)
model = namespace['MOSSImageTransformer'](**cfg).to(device='cpu',dtype=torch.float32).eval()
with torch.no_grad():
 for ordinal,(name,p) in enumerate(model.named_parameters()):
  count=p.numel()
  if name.endswith('norm.weight') or name.endswith('norm1.weight') or name.endswith('norm2.weight'):
   values=[1.0+0.03*math.sin((i+1)*(ordinal+1)*0.071) for i in range(count)]
  elif name.endswith('.bias'):
   values=[0.07*math.cos((i+1)*(ordinal+1)*0.091) for i in range(count)]
  else:
   divisor=math.sqrt(p.shape[-1] if p.ndim>1 else count)
   values=[0.65*math.sin((i+1)*0.117+(ordinal+1)*0.319)/divisor for i in range(count)]
  p.copy_(torch.tensor(values,device='cpu',dtype=torch.float32).reshape(p.shape))

def tensor(t):
 assert t.device.type=='cpu'
 return dict(shape=list(t.shape),dtype=str(t.dtype).replace('torch.',''),values=t.detach().float().flatten().tolist())

hidden=torch.tensor([0.6*math.sin((i+1)*0.19) for i in range(2*4*2*3)],device='cpu',dtype=torch.float32).reshape(2,4,2,3)
context=torch.tensor([0.8*math.cos((i+1)*0.137) for i in range(2*4*12)],device='cpu',dtype=torch.float32).reshape(2,4,12)
mask=torch.tensor([[1,1,1,0],[1,0,1,1]],device='cpu',dtype=torch.bool)
times=torch.tensor([731.25,271.5],device='cpu',dtype=torch.float32)
fixture = dict(schema=1,source_sha256=EXPECTED,source_revision='fc5de0d56fc910cbca6f6625a70f5f582b26deb1',torch_version=torch.__version__,device='cpu',config=cfg,extraction='Exact unmodified frozen class/function AST definitions; real Diffusers mixins; unused loading imports omitted because transformers is unavailable.',weights={k:tensor(v) for k,v in model.state_dict().items()},inputs=dict(hidden_states=tensor(hidden),encoder_hidden_states=tensor(context),encoder_attention_mask=tensor(mask),timestep=tensor(times)),cases=[])
for dtype in (torch.float32,torch.bfloat16):
 try:
  oracle=copy.deepcopy(model).to(device='cpu',dtype=dtype).eval()
  assert all(p.device.type=='cpu' for p in oracle.parameters())
  for skipped in ([],[1]):
   captures={}
   handles=[]
   for name in ('text_embedder','text_embedder.blocks.0','text_embedder.blocks.1','layers.0.q_norm','layers.0.k_norm','layers.0','layers.1','layers.2','final_norm_video','final_linear_video'):
    def hook(module,args,result,name=name):captures.setdefault(name,[]).append(tensor(result))
    handles.append(oracle.get_submodule(name).register_forward_hook(hook))
   with torch.inference_mode():
    expected=oracle(hidden,times,context,mask,skip_layers=skipped).sample
   for h in handles:h.remove()
   assert torch.isfinite(expected).all()
   fixture['cases'].append(dict(dtype=str(dtype).replace('torch.',''),skip_layers=skipped,velocity=tensor(expected),intermediates=captures))
 except (RuntimeError,TypeError) as e:
  if dtype is torch.float32:raise
  fixture['bf16_unavailable']=repr(e)
# Direct author leaf-op values detect norm precision, hidden/gate ordering and four-axis rotation.
a=torch.tensor([[0.3,-0.7,1.2,-1.4,0.1,0.8,-0.4,0.9]],device='cpu',dtype=torch.float32)
b=torch.tensor([[-1.1,0.2,0.5,-0.8,1.4,-0.2,0.6,0.3]],device='cpu',dtype=torch.float32)
rope=namespace['_RoPE'](8,10000.0).to(device='cpu')(2,3,3,torch.device('cpu'))
fixture['leaf_ops']=dict(situ_hidden=tensor(a),situ_gate=tensor(b),situ_expected=tensor(namespace['_situ'](a,b)),rope_2x3_text3=tensor(rope),rms_input=tensor(a),rms_weight=tensor(model.layers[0].q_norm.weight),rms_expected=tensor(model.layers[0].q_norm(a)))
fixture['elapsed_seconds']=time.monotonic()-started
path=OUT/'fixture.json'
path.write_text(json.dumps(fixture,separators=(',',':'))+'\n')
(OUT/'frozen_author_pipeline.py').write_bytes(source)
receipt=dict(status='PASS_CPU_SYNTHETIC_AUTHOR_GRAPH',fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_sha256=EXPECTED,torch_version=torch.__version__,config=cfg,cases=[dict(dtype=c['dtype'],skip_layers=c['skip_layers']) for c in fixture['cases']],bytes=path.stat().st_size,elapsed_seconds=fixture['elapsed_seconds'],device='cpu',real_weights=False,image_generation=False,ast_definitions=names)
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
signal.alarm(0)
