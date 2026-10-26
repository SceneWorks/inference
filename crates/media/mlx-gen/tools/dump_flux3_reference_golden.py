"""CPU fixture oracle: execute only frozen reference-preparation function definitions."""
import ast
import hashlib
import json
import math
import pathlib
import sys
import numpy as np
from PIL import Image, ImageOps

source, target = map(pathlib.Path, sys.argv[1:])
source_sha256 = '33c323ebaeaf56790f3d0022a57827b5348446a8a4b372ab2bcc606d722e76a9'
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha256, 'not frozen BFL45e57 pipeline.py'
target.mkdir(parents=True, exist_ok=True)
names = {'_ceil16', 'reference_canvas', 'prepare_reference', 'prepare_multi_reference'}
tree = ast.parse(source.read_text())
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(nodes) == len(names)
ns = dict(math=math, np=np, Image=Image, ImageOps=ImageOps, MATCH_MAX_PIXELS=4*1024*1024,
          MAX_SIDE=4096, MIN_SIDE=64, DOWNSAMPLING_RATIO=16, MULTI_REFERENCE_MAX_PIXELS=1024*1024)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
rows = []
for name, width, height, multi in [('reflect',83,67,False),('symmetric',3,1,False),('single-resize',1,5000,False),('multi-resize',2051,1027,True)]:
    y,x=np.indices((height,width),dtype=np.uint32)
    pixels=np.stack(((x*17+y*7)%256,(x*3+y*19)%256,(x*29+y*11)%256),axis=-1).astype(np.uint8)
    original=Image.fromarray(pixels)
    if multi:
        output=ns['prepare_multi_reference'](original)
        crop=None
    else:
        output,crop=ns['prepare_reference'](original)
    file=name+'.rgb'
    (target/file).write_bytes(np.asarray(output).tobytes())
    rows.append(dict(name=name,input=[width,height],multi=multi,output=list(output.size),crop=crop,file=file))
# Actual EXIF-bearing PNG: both forks start from the SAME oriented original, before AE drops alpha.
y,x=np.indices((67,83),dtype=np.uint32)
rgba=np.stack(((x*17+y*7)%256,(x*3+y*19)%256,(x*29+y*11)%256,np.array([0,128,255],dtype=np.uint8)[(x+y)%3]),axis=-1).astype(np.uint8)
exif=Image.Exif(); exif[274]=6
Image.fromarray(rgba).save(target/'exif6.png',exif=exif)
with Image.open(target/'exif6.png') as opened:
    oriented=ImageOps.exif_transpose(opened)
    rgb=oriented.convert('RGB')
    white=Image.alpha_composite(Image.new('RGBA',oriented.size,'white'),oriented.convert('RGBA')).convert('RGB')
    prepared,crop=ns['prepare_reference'](opened)
    (target/'exif6.rgb').write_bytes(np.asarray(rgb).tobytes())
    (target/'exif6.pup.rgb').write_bytes(np.asarray(white).tobytes())
    (target/'exif6.prepared.rgb').write_bytes(np.asarray(prepared).tobytes())
    multi=ns['prepare_multi_reference'](opened)
    (target/'exif6.multi.rgb').write_bytes(np.asarray(multi).tobytes())
    exif_case=dict(file='exif6.png',stored=[83,67],orientation=6,oriented=list(oriented.size),canvas=list(prepared.size),crop=crop)
(target/'reference-preprocess.json').write_text(json.dumps(dict(upstream='45e57c9fb06f9751cdced29a65c5d2171003be05',source_sha256=source_sha256,dtype='RGB uint8',pillow=Image.__version__,numpy=np.__version__,exif=exif_case,functions=sorted(names),pattern=['(x*17+y*7)%256','(x*3+y*19)%256','(x*29+y*11)%256'],cases=rows),indent=2)+'\n')
print([(r['name'],r['output'],r['crop']) for r in rows])
