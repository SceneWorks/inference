"""CPU-only encoded image-input oracle: selected, SHA-verified frozen BFL function bodies.

Arguments: frozen src/flux3/pipeline.py and an output fixture directory. No model imports/GPU.
Encoded-mode fixtures execute the actual frozen PUP policy as well as the AE preparation.
"""
import ast
import copy
import hashlib
import io
import json
import math
import pathlib
import struct
import sys
import zlib

import numpy as np
from PIL import Image, ImageOps

source, target = map(pathlib.Path, sys.argv[1:])
source_sha256 = "33c323ebaeaf56790f3d0022a57827b5348446a8a4b372ab2bcc606d722e76a9"
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha256, "not frozen BFL45e57 pipeline.py"
target.mkdir(parents=True, exist_ok=True)
names = {"_ceil16", "reference_canvas", "prepare_reference", "prepare_multi_reference"}
tree = ast.parse(source.read_text(encoding="utf8"))
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
assert len(nodes) == len(names)
ns = dict(math=math, np=np, Image=Image, ImageOps=ImageOps, MATCH_MAX_PIXELS=4*1024*1024,
          MAX_SIDE=4096, MIN_SIDE=64, DOWNSAMPLING_RATIO=16, MULTI_REFERENCE_MAX_PIXELS=1024*1024)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)
# Execute ONLY frozen PUP preprocessing constants/functions, never its imports/server/env writes.
pup_source=source.with_name("pup.py")
pup_sha256="e3561ae3cbff1a080ad8854c0b89bd0687c1ea9d84c74f5203b44844cbd4684d"
assert hashlib.sha256(pup_source.read_bytes()).hexdigest()==pup_sha256, "not frozen BFL45e57 pup.py"
pup_tree=ast.parse(pup_source.read_text(encoding="utf8"))
constant_names={"MAX_REFERENCES","REFERENCE_MULTIPLE","MAX_ASPECT_RATIO","MAX_PIXELS_BY_REFS","JPEG_QUALITY"}
pup_nodes=[n for n in pup_tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constant_names for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name in {"reference_size","policy_reference"})]
policy=next(n for n in pup_nodes if isinstance(n,ast.FunctionDef) and n.name=="policy_reference")
# The pre-JPEG RGB fork is the literal source body up to its reference_size call, not a new policy.
rgb_policy=copy.deepcopy(policy)
rgb_policy.name="rgb_policy"
cut=next(i for i,n in enumerate(rgb_policy.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="size" for t in n.targets))
rgb_policy.body=rgb_policy.body[:cut]+[ast.Return(value=ast.Name(id="image",ctx=ast.Load()))]
pup_ns=dict(Image=Image,ImageOps=ImageOps,math=math,io=io)
module=ast.fix_missing_locations(ast.Module(body=pup_nodes+[rgb_policy],type_ignores=[]))
exec(compile(module,str(pup_source),"exec"),pup_ns)
alpha_condition=next(n.test for n in policy.body if isinstance(n,ast.If))
alpha_expression=compile(ast.Expression(body=alpha_condition),str(pup_source),"eval")


def chunk(kind, payload):
    return struct.pack(">I",len(payload))+kind+payload+struct.pack(">I",zlib.crc32(kind+payload)&0xffffffff)


def png16(name, pixels, colour):
    """Independent minimal lossless PNG encoding; Pillow decides how the encoded file opens."""
    height,width=pixels.shape[:2]
    raw=b"".join(b"\0"+row.astype(">u2").tobytes() for row in pixels)
    header=struct.pack(">IIBBBBB",width,height,16,colour,0,0,0)
    (target/(name+".png")).write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",header)+chunk(b"IDAT",zlib.compress(raw))+chunk(b"IEND",b""))


luma=np.array([0,17,63,127,128,191,223,254,255,12,35,200],dtype=np.uint8).reshape(3,4)
alpha=np.resize(np.array([0,128,255],dtype=np.uint8),luma.shape)
Image.fromarray(np.stack((luma,alpha),axis=-1)).save(target/"mode-la8.png")
indices=(np.arange(12,dtype=np.uint8)%4).reshape(3,4)
palette=Image.fromarray(indices).convert("P")
palette.putpalette([230,12,7,9,210,44,3,18,230,40,70,110]+[0]*756)
palette.save(target/"mode-palette-alpha.png",transparency=bytes([0,128,255,63]))
palette.save(target/"mode-palette-opaque.png")
rgb8=np.stack((luma,np.roll(luma,1),np.roll(luma,2)),axis=-1)
Image.fromarray(rgb8).save(target/"mode-rgb-trns.png",transparency=tuple(int(c) for c in rgb8[0,0]))
Image.fromarray(luma).save(target/"mode-luma-trns.png",transparency=0)
luma16=np.array([0,1,127,254,255,256,257,4096,16384,32767,32768,65535],dtype=np.uint16).reshape(3,4)
rgb16=np.stack((luma16,np.roll(luma16,1),np.roll(luma16,2)),axis=-1)
alpha16=np.resize(np.array([0,32768,65535],dtype=np.uint16),luma.shape)
png16("mode-luma16",luma16,0)
png16("mode-rgb16",rgb16,2)
png16("mode-la16",np.stack((luma16,alpha16),axis=-1),4)
png16("mode-rgba16",np.concatenate((rgb16,alpha16[...,None]),axis=-1),6)
format_names=["mode-la8","mode-palette-alpha","mode-palette-opaque","mode-rgb-trns","mode-luma-trns","mode-luma16","mode-rgb16","mode-la16","mode-rgba16"]
formats=[]
for name in format_names:
    file=target/(name+".png")
    header=file.read_bytes()
    with Image.open(file) as opened:
        oriented=ImageOps.exif_transpose(opened)
        carrier="rgba" if eval(alpha_expression,pup_ns,{"image":oriented}) else "rgb"
        ae_rgb=oriented.convert("RGB")
        pup_rgb=pup_ns["rgb_policy"](opened,1)
        prepared,_=ns["prepare_reference"](opened)
        for suffix,image in [("ae",ae_rgb),("pup",pup_rgb),("prepared",prepared)]:
            (target/(name+"."+suffix+".rgb")).write_bytes(np.asarray(image).tobytes())
        if carrier=="rgba":
            (target/(name+".decoded.rgba")).write_bytes(np.asarray(oriented.convert("RGBA")).tobytes())
        row=dict(name=name,file=file.name,source_mode=opened.mode,bit_depth=header[24],png_colour_type=header[25],carrier=carrier,size=list(oriented.size),canvas=list(prepared.size))
        for count in [1,10]:
            full=pup_ns["policy_reference"](opened,count)
            (target/(name+f".policy{count}.rgb")).write_bytes(np.asarray(full).tobytes())
            row[f"policy{count}_size"]=list(full.size)
        formats.append(row)

metadata=dict(upstream="45e57c9fb06f9751cdced29a65c5d2171003be05",source_sha256=source_sha256,pup_source_sha256=pup_sha256,dtype="RGB uint8",pillow=Image.__version__,numpy=np.__version__,encoded_formats=formats)
(target/"image-input.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf8",newline="\n")
print("Encoded source modes:",[(r["name"],r["source_mode"],r["carrier"]) for r in formats])
