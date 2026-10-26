#!/usr/bin/env python3
"""CPU BF16 fused-linear vs split-bias probe for versioned frozen-source fixtures."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
sys.dont_write_bytecode=True
import hashlib,json
from pathlib import Path
import torch
torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
def tensor(a):
    return torch.tensor(a['values'],dtype={'bfloat16':torch.bfloat16,'float32':torch.float32}[a['dtype']],device='cpu').reshape(a['shape'])
def array(t):
    return {'shape':list(t.shape),'dtype':str(t.dtype).removeprefix('torch.'),'values':t.float().flatten().tolist()}
for version in ['tiny_edit_v1_time_probe','tiny_edit_v2']:
    path=HERE/version/'fixture.json'
    data=json.loads(path.read_text())
    rows=[]
    with torch.no_grad():
        for probe in data['time_probes']:
            modules=[]
            for prefix,key in [('time_embedding.0','time1'),('time_embedding.2','time2'),('shared_block_modulation','modulation_flat'),('video_head_modulation','head_flat')]:
                x=tensor(probe[key+'_input']);weight=tensor(data['weights'][prefix+'.weight'])
                bias=data['weights'].get(prefix+'.bias');bias=None if bias is None else tensor(bias)
                fused=torch.nn.functional.linear(x,weight,bias)
                assert torch.equal(fused,tensor(probe[key]))
                mat=x@weight.T
                split=mat if bias is None else mat+bias
                modules.append({'module':prefix,'source_output':probe[key],'matmul_only':array(mat),'split_bias_output':array(split),'split_bias_max_abs_difference':float((fused.float()-split.float()).abs().max())})
            rows.append({'normalized_timestep':probe['normalized_timestep'],'modules':modules})
    diagnostic={'schema':1,'device':'cpu','source_fixture_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'note':'Expected source outputs are actual unchanged-source module hooks. Split matmul-plus-bias is a deliberately different CPU arithmetic probe, not the expected graph.','probes':rows}
    (HERE/version/'bias-rounding-diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
