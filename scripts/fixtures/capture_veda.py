#!/usr/bin/env python3
"""Capture the frozen upstream CPU oracle; never derive expected masks from Rust.
Run in a CPU-only environment with torch==2.14.1 and safetensors==0.7.0.
The predictor header is the retained E7 artifact header; no real weight download.
"""
import argparse, hashlib, importlib.util, json, pathlib, sys, types, random
import torch, safetensors
assert torch.__version__ == '2.14.1', torch.__version__
p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--header',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
torch.set_num_threads(1)
root=a.source/'veda_comfy/core'
pkg=types.ModuleType('frozen');pkg.__path__=[str(root)];sys.modules['frozen']=pkg
mods={}
for name in ['tiling','plans','selection','predictor']:
 spec=importlib.util.spec_from_file_location('frozen.'+name,root/(name+'.py'));m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);mods[name]=m
tiling,plans,selection,predictor=(mods[n] for n in ['tiling','plans','selection','predictor'])
plan_json=json.loads(json.loads(a.header.read_text())['__metadata__']['plans'])
table=plans.PlanTable([plans.TilePlan.from_json(p) for p in plan_json.values()])
choices=[]
for grid in [p.grid for p in table.plans.values()]+[(t,w,h) for t,h,w in [p.grid for p in table.plans.values()]]+[(42,24,42),(62,19,31),(37,32,18),(37,18,32),(72,24,42),(37,25,43),(7,10,18)]:
 c=table.select(grid);choices.append(dict(grid=list(grid),name=c.plan.name,plan_grid=list(c.plan.grid),shapes=[str(s) for s in c.plan.shapes],exact=c.exact))
layouts=[]
for spans,seq in [([(3,[3,5,19],[2,4,16])],291),([(2,[1,5,19],[2,4,16]),(101,[3,5,19],[2,4,16])],393),([(4,[2,4,4],[2,4,16]),(40,[1,3,17],[1,8,16]),(100,[3,5,19],[2,4,16])],389),([(0,[1,1,10],[1,1,128])],10)]:
 ss=[tiling.TiledSpan(start,tuple(grid),tiling.TileShape(*shape)) for start,grid,shape in spans];l=tiling.build_tile_layout(ss,seq)
 layouts.append(dict(spans=[dict(start=s,grid=g,shape=h) for s,g,h in spans],seq=seq,perm=l.perm.tolist(),valid=l.valid_count.tolist(),reference=l.n_ref_tiles,ref_tokens=l.ref_tokens,target_tokens=l.target_tokens))
# Exercise top-k tie cutoff, heap branch, partial/global rows and two independent blocks.
cases=[]
for n,ref,valid in [(8,0,[128]*7+[4,3,0]),(10,3,[64,80,42]+[128]*6+[6,2,0]),(10,3,[64,80,0]+[128]*6+[6,2,0]),(129,0,[128]*129+[3,0])]:
 # Synthetic empty key tiles cannot arise from ordinary positive grids. Frozen
 # scatter_ forces even their diagonal; mask it out under E2's explicit empty-key rule.
 l=types.SimpleNamespace(n_video_tiles=n,n_ref_tiles=ref,n_tiles=len(valid),target_tokens=sum(valid[ref:n]),ref_tokens=sum(valid[:ref]),kv_ok=torch.tensor([v>0 for v in valid]))
 for kind in ['ties','near_ties','ranked']:
  base=torch.arange(n*n,dtype=torch.int64).reshape(n,n)
  scores=(torch.zeros(n,n) if kind=='ties' else ((base*17+5)%31).float() if kind=='ranked' else (((base*17+5)%3).float()*2**-22+1))
  scores=torch.stack([scores,scores.flip(-1)])
  for generated,reference in [(('ratio',0.1),('ratio',1.0)),(('tiles',2.3),('tiles',1.5)),(('ratio',1.0),('tiles',2.0))]:
   gb=selection.Budget(**{generated[0]:generated[1]});rb=selection.Budget(**{reference[0]:reference[1]})
   index,keep=selection.select(scores,l,selection.column_blocks(l,gb,rb));mask=selection.block_mask(index,keep,l)
   upstream_rows=[[torch.nonzero(row).view(-1).tolist() for row in head] for head in mask]
   mask &= l.kv_ok[None,None,:]
   rows=[[torch.nonzero(row).view(-1).tolist() for row in head] for head in mask]
   cases.append(dict(name=f'{n}-{kind}-{generated}-{reference}',video=n,reference=ref,valid=valid,ref_tokens=l.ref_tokens,target_tokens=l.target_tokens,heads=2,scores=scores.flatten().tolist(),generated=list(generated),reference_budget=list(reference),upstream_rows=[r for h in upstream_rows for r in h],rows=[r for h in rows for r in h]))
# Actual gather, BF16 inputs, masked TripPool and FP8*scale->BF16 projections.
scorers=[]
for layout in layouts[:3]:
 ss=[tiling.TiledSpan(s['start'],tuple(s['grid']),tiling.TileShape(*s['shape'])) for s in layout['spans']];l=tiling.build_tile_layout(ss,layout['seq']);heads,dim=2,4
 raw=torch.arange(layout['seq']*heads*dim,dtype=torch.int64)
 q=((raw*13%37)-18).float().reshape(layout['seq'],heads,dim).div(16).to(torch.bfloat16);k=((raw*7%41)-20).float().reshape_as(q).div(16).to(torch.bfloat16)
 qt=tiling.gather_tiles(q,l,torch.arange(heads));kt=tiling.gather_tiles(k,l,torch.arange(heads))
 fp8_bits=torch.tensor(([0x18,0x9b,0x20,0x28,0x1a,0x98,0x30,0xa0]*12),dtype=torch.uint8).view(heads,3*dim,dim)
 scale=torch.tensor([0.03003,0.07111],dtype=torch.float32)
 proj=(fp8_bits.view(torch.float8_e4m3fn).float()*scale[:,None,None]).to(torch.bfloat16)
 fq=predictor.pool_video_tiles(qt,l);fk=predictor.pool_video_tiles(kt,l);logits=predictor.tile_logits(fq,fk,proj,proj.flip(1))
 index,keep=selection.select(logits,l,selection.column_blocks(l,selection.Budget(tiles=2.3),selection.Budget(tiles=1.5)))
 mask=selection.block_mask(index,keep,l)
 scorers.append(dict(layout=layout,heads=heads,dim=dim,q=qt.permute(1,0,2).flatten().float().tolist(),k=kt.permute(1,0,2).flatten().float().tolist(),fp8=fp8_bits.flatten().tolist(),scales=scale.tolist(),proj=proj.view(torch.int16).flatten().tolist(),fq=fq.flatten().tolist(),fk=fk.flatten().tolist(),logits=logits.flatten().tolist(),rows=[torch.nonzero(row).view(-1).tolist() for h in mask for row in h]))
random.seed(25770)
topk_probes=[]
for n in [3,4,7,8,10,16,24,32,63,64,65,129,256,512]:
 for count in [1,2,3,5,8,16,31,65]:
  if count>n:continue
  for rep in range(10):
   values=[float(random.randrange(4)) for _ in range(n)];own=random.randrange(n);scores=torch.tensor(values);scores[own]=float('inf')
   topk_probes.append(dict(scores=values,own=own,k=count,prefix=sorted(torch.topk(scores,count).indices[:count-1].tolist())))
result=dict(provenance=dict(revision='60bfae688897dc41bc8685ca9f3130d7ed8f3b0c',torch=torch.__version__,safetensors=safetensors.__version__,device='cpu',machine='arm64',input_dtype='bfloat16',projection_storage='float8_e4m3fn scaled FP32 then bfloat16',scorer_dtype='float32',score_fixture_dtype='float32',tie_authority='PyTorch CPU libc++ nth_element/partial_sort',source_sha256={str(p.relative_to(a.source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')},header_sha256=hashlib.sha256(a.header.read_bytes()).hexdigest(),empty_key_rule='Empty diagonals excluded per E2; source scatter_ would re-enable an artificial empty diagonal (positive grid layouts contain no empty tiles)'),choices=choices,layouts=layouts,selections=cases,scorers=scorers,topk_probes=topk_probes,rounding_probes=[dict(input=v,cost=int(round(v,3)*1000+0.5)) for v in [0.0005,0.0015,0.0025,1.2345,1.2355,0.1235,0.9995]])
a.out.write_text(json.dumps(result,separators=(',',':'))+'\n')
print(json.dumps(dict(torch=torch.__version__,plans=len(plan_json),choices=len(choices),layouts=len(layouts),selections=len(cases),scorers=len(scorers),topk_probes=len(topk_probes),fixture_sha256=hashlib.sha256(a.out.read_bytes()).hexdigest())))
