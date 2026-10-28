#!/usr/bin/env python3
"""Exact frozen Comfy reference preprocessing on CPU with tiny synthetic pixels."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
sys.dont_write_bytecode=True
import ast,hashlib,json,math,signal
from pathlib import Path
signal.alarm(60)
import torch
from PIL import Image, __version__ as pillow_version
HERE=Path(__file__).resolve().parent
torch.set_num_threads(1)
ns=dict(torch=torch,Image=Image,math=math)
sources={}
for name,relative in [('image_to_tensor','ComfyUI-Moss-Image-Edit/moss_runtime/processing.py'),('_reference_tensor','ComfyUI-Moss-Image-Edit/nodes.py')]:
    p=HERE/relative
    source=p.read_bytes()
    sources[relative]=hashlib.sha256(source).hexdigest()
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
cases=[]
for width,height,max_edge in [(19,37,32),(65,29,32),(8,40,48),(23,17,48),(19,37,47),(48,16,48)]:
    pixels=[(x*37+y*73+c*53+((x*y+c*7)%13)*19)%256 for y in range(height) for x in range(width) for c in range(3)]
    input_tensor=torch.tensor(pixels,device='cpu',dtype=torch.float32).reshape(1,height,width,3)/255.0
    with torch.no_grad(): normalized=ns['_reference_tensor'](input_tensor,max_edge)
    rgb=((normalized[0].permute(1,2,0)+1)*127.5).round().to(torch.uint8).flatten().tolist()
    out_h,out_w=normalized.shape[-2:]
    cases.append(dict(width=width,height=height,max_edge=max_edge,input_rgb=pixels,output_width=out_w,output_height=out_h,expected_rgb=rgb,expected_normalized_nchw=dict(shape=list(normalized.shape),values=normalized.flatten().tolist()),input_sha256=hashlib.sha256(bytes(pixels)).hexdigest(),output_sha256=hashlib.sha256(bytes(rgb)).hexdigest()))
data=dict(schema=1,source_sha256=sources,pillow_version=pillow_version,torch_version=torch.__version__,device='cpu',input_layout='uint8 RGB row-major; frozen node receives FP32 bytes/255 NHWC batch1',output_layout='uint8 RGB row-major; normalized float NCHW also retained',cases=cases)
out=HERE/'tiny_edit/resize_fixture.json'
out.write_text(json.dumps(data,separators=(',',':'))+'\n')
receipt=dict(status='PASS_CPU_FROZEN_PILLOW_PREPROCESS',fixture_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pillow_version=pillow_version,source_sha256=sources,cases=[{k:c[k] for k in ['width','height','max_edge','output_width','output_height','output_sha256']} for c in cases],bytes=out.stat().st_size)
(HERE/'tiny_edit/resize_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
signal.alarm(0)
