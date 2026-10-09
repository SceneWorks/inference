"""One captured-input linear diagnostic. Preparation never constructs MLX tensors."""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, struct
from pathlib import Path

PREFIX = "transformer_blocks.0.attn.to_k"
BASE = "004cee41f380ff263796ad90277e5f6224d73ae3"
REV = "1691de01c24a070131e0a28bf4c065fd027f4fe9"
KEYS = [PREFIX + s for s in (".weight", ".scales", ".biases")]
F32 = 2.0**-24
F64 = 2.0**-53
MIN_NORMAL = 2.0**-126

def sha(b): return hashlib.sha256(b).hexdigest()
def dump(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2); f.write("\n")
def read_json(path): return json.loads(path.read_bytes())
def header(path):
    with path.open("rb") as f:
        n=struct.unpack("<Q", f.read(8))[0]
        if n > 16 << 20: raise ValueError("oversized safetensors header")
        return json.loads(f.read(n)), 8+n
def finite(b):
    return all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", b))

def prepare(fixture, snapshot, out):
    """Read only exact selected payload ranges; do not materialize whole shard files."""
    if out.exists(): raise ValueError("exclusive preparation output required")
    binding=read_json(fixture/"bindings.json")
    if binding["sourceCurrent"]!=BASE or binding["module"]!=PREFIX: raise ValueError("input authority")
    if binding["originalActivationDtype"]!="Float32": raise ValueError("original dtype")
    if snapshot.name!=REV: raise ValueError("pinned snapshot basename")
    component=snapshot/"q4"/"transformer"
    if not component.is_dir(): raise ValueError("pinned Q4 component absent")
    selected={}; provenance=[]; index_facts=[]
    indexes=sorted(component.glob("*.safetensors.index.json"))
    sources={}
    if indexes:
        for idx in indexes:
            b=idx.read_bytes(); j=json.loads(b)
            index_facts.append({"path":str(idx),"bytes":len(b),"sha256":sha(b)})
            for key in KEYS:
                if key in j["weight_map"]:
                    name=j["weight_map"][key]
                    if Path(name).name!=name: raise ValueError("non-local shard name")
                    if key in sources and sources[key]!=name: raise ValueError("ambiguous index")
                    sources[key]=name
    else:
        for p in sorted(component.glob("*.safetensors")):
            h,_=header(p)
            for key in KEYS:
                if key in h:
                    if key in sources: raise ValueError("duplicate selected tensor")
                    sources[key]=p.name
    if set(sources)!=set(KEYS): raise ValueError("selected tensor closure absent")
    out.mkdir()
    files=[]
    for entry in binding["files"]:
        if Path(entry["file"]).name!=entry["file"]:raise ValueError("non-local fixture filename")
        p=fixture/entry["file"]; b=p.read_bytes()
        if len(b)!=entry["bytes"] or sha(b)!=entry["sha256"] or not finite(b): raise ValueError("fixture bytes")
        (out/entry["file"]).write_bytes(b); files.append(entry)
    mini={}; payloads=[]; cursor=0
    for key in KEYS:
        path=component/sources[key]; h,start=header(path); t=h[key]
        begin,end=t["data_offsets"]; n=end-begin
        expected_shape=[4096,512] if key.endswith(".weight") else [4096,64]
        allowed={"U32"} if key.endswith(".weight") else {"F32","F16","BF16"}
        if t["shape"]!=expected_shape or t["dtype"] not in allowed: raise ValueError("selected header shape/dtype")
        width=4 if t["dtype"] in {"U32","F32"} else 2
        if n!=math.prod(t["shape"])*width or n>8<<20: raise ValueError("selected payload range")
        with path.open("rb") as f:
            f.seek(start+begin); b=f.read(n)
        if len(b)!=n: raise ValueError("truncated selected payload")
        raw_name=key.rsplit(".",1)[1]+".raw"
        (out/raw_name).write_bytes(b)
        e={"file":raw_name,"bytes":n,"sha256":sha(b),"dtype":t["dtype"],"shape":t["shape"]}
        files.append(e); selected[key]=e
        provenance.append({"key":key,"shard":str(path),"headerSha256":sha(json.dumps(h,sort_keys=True,separators=(",",":")).encode()),
            "shardBytes":path.stat().st_size,"headerPayloadOffset":start,"dataOffsets":[begin,end],"tensorPayload":e,
            "publisherPayloadHashClaimed":False})
        mini[key]={"dtype":t["dtype"],"shape":t["shape"],"data_offsets":[cursor,cursor+n]}; cursor+=n; payloads.append(b)
    hb=json.dumps(mini,separators=(",",":")).encode(); hb+=b" "*((-len(hb))%8)
    # Small exact three-tensor container; this is explicitly not the original shard.
    with (out/"selected.safetensors").open("xb") as f:
        f.write(struct.pack("<Q",len(hb)));f.write(hb)
        for b in payloads:f.write(b)
    b=(out/"selected.safetensors").read_bytes()
    files.append({"file":"selected.safetensors","bytes":len(b),"sha256":sha(b)})
    entries={e["role"]:e for e in binding["files"]}
    original=entries["input"]; captured=(out/original["file"]).read_bytes()
    # M65 is the smallest nonbatched M making floor512/(ceil(M/32)*128)=1.
    replay=b"".join(captured[(i%3)*16384:(i%3+1)*16384] for i in range(65))
    replay_entry={"file":"replay-x.f32","shape":[1,65,4096],"dtype":"F32","bytes":len(replay),"sha256":sha(replay)}
    (out/replay_entry["file"]).write_bytes(replay); files.append(replay_entry)
    cfg=component/"config.json"; cb=cfg.read_bytes(); config=json.loads(cb)
    q=config.get("quantization",{})
    if q.get("bits")!=4 or q.get("group_size")!=64: raise ValueError("Q4 group config")
    result={"schema":"sc24163-single-linear-prepared-v1","kind":"SINGLE_LINEAR_DIAGNOSTIC_ONLY","acceptance":False,
        "module":PREFIX,"sourceBase":BASE,"bits":4,"groupSize":64,"originalActivationDtype":"Float32",
        "replayActivationDtype":"Float32","fixtureBindingsSha256":sha((fixture/"bindings.json").read_bytes()),
        "sourcePriorCapture":binding["sourcePriorCapture"],"inputReceipt":binding["inputReceipt"],"failedDonorSha256":binding["failedDonorSha256"],"snapshotRevision":REV,"snapshotRoot":str(snapshot),
        "indexFacts":index_facts,"configFact":{"path":str(cfg),"sha256":sha(cb)},"selected":selected,
        "provenance":provenance,"files":files,"capturedInput":original,"originalActivationShape":[1,10583,4096],"constructedReplay":"65 rows, row i copies captured row i%3 bit-exactly; not original full activation","input":replay_entry,"w1":entries["w1"],"w2a":entries["w2a"],"w2b":entries["w2b"]}
    dump(out/"prepared.json",result); print(sha((out/"prepared.json").read_bytes()))

def verify_build(target, out):
    wanted={"utils.h":"mlx/utils.h","matmul.cpp":"mlx/backend/metal/matmul.cpp",
        "quantized.cpp":"mlx/backend/metal/quantized.cpp","quantized.h":"mlx/backend/metal/kernels/quantized.h",
        "device.cpp":"mlx/backend/metal/device.cpp", "mma.h":"mlx/backend/metal/kernels/steel/gemm/mma.h"}
    facts={}; texts={}
    for key,suffix in wanted.items():
        candidates=[p for p in target.rglob(key) if p.as_posix().endswith(suffix) and "mlx-src" in p.as_posix()]
        if not candidates: raise ValueError("compiled MLX source absent: "+key)
        hashes={sha(p.read_bytes()) for p in candidates}
        if len(hashes)!=1: raise ValueError("ambiguous compiled MLX sources: "+key)
        p=candidates[0]; b=p.read_bytes(); texts[key]=b.decode()
        facts[key]={"path":str(p),"bytes":len(b),"sha256":sha(b),"equivalentCopies":len(candidates)}
    if not re.search(r'static\s+bool\s+enable_tf32_\s*=\s*get_var\("MLX_ENABLE_TF32",\s*1\)',texts["utils.h"]):
        raise ValueError("static precision knob source")
    condition=r'env::enable_tf32\(\)\s*\|\|\s*(?:a|x)\.dtype\(\)\s*!=\s*float32'
    for key in ("matmul.cpp","quantized.cpp"):
        if not re.search(condition,texts[key]): raise ValueError("strict float32 dispatch condition: "+key)
    if "scale" not in texts["quantized.h"] or "dequantize" not in texts["quantized.h"]: raise ValueError("affine kernel source")
    q=texts["quantized.cpp"]
    for marker in ["int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4", "if (M >= vector_limit)",
            "int bm = 32, bn = 32", "int split_k = std::max(1, 512 / current_tgs)", "if (split_k <= 1)",
            "bool non_batched = w.ndim() == 2 && x.flags().row_contiguous"]:
        if marker not in q: raise ValueError("replay dispatch predicate drift: "+marker)
    start=q.index("inline int get_qmv_batch_limit("); end=q.index("inline int add_strides_and_shapes",start)
    limits=[int(x) for x in re.findall(r"return\s+(\d+)\s*;",q[start:end])]
    if not limits or max(limits)>32: raise ValueError("QMV batch threshold exceeds replay")
    if "typename AccumType = float" not in texts["mma.h"] or "BlockMMA<T, T, BM, BN, BK" not in texts["quantized.h"]:
        raise ValueError("strict represented Float32 operands/accumulator source")
    if "bool is_nax_available()" not in texts["device.cpp"]: raise ValueError("NAX capability source")
    result={"schema":"sc24163-single-linear-compiled-source-proof-v1","strictFloat32DispatchSourceVerified":True,
        "basis":"actual staged MLX source conditions, Float32 inputs, fresh process MLX_ENABLE_TF32=0",
        "files":facts,"replayM":65,"originalM":10583,"qmmSplitKPartitions":1,"minimumReplayMWithoutSplitK":65,"hardwareNaxObservationRequired":True,"strictOperandAndAccumulator":"Float32 BlockMMA default AccumType=float","qmvBatchLimits":limits,"acceptance":False}
    dump(out,result); print(sha(out.read_bytes()))

def gamma(n,u):
    if n*u>=1: raise ValueError("unbounded reduction")
    return n*u/(1-n*u)

def bf16(values):
    import numpy as np
    a=np.asarray(values,dtype=np.float32).copy(); u=a.view(np.uint32)
    u[:]=((u.astype(np.uint64)+0x7fff+((u>>16)&1))&0xffff0000).astype(np.uint32)
    return a.astype(np.float64)

def packed_reference(x,codes,scales,biases):
    import numpy as np
    k=x.shape[1]; group=k//scales.shape[1]
    s=np.repeat(scales,group,axis=1); b=np.repeat(biases,group,axis=1)
    coefficients=codes*s+b; d=np.abs(codes*s)+np.abs(b)
    reference=x@coefficients.T
    bound=(gamma(2*k+8,F32)+gamma(2*k+8,F64))*(np.abs(x)@d.T)
    bound+=(2*k+8)*MIN_NORMAL*(1+np.abs(x).sum(axis=1,keepdims=True))
    return reference,bound

def structured_reference(x,a,b):
    import numpy as np
    rows=x.shape[0]; xr=x.reshape(rows,64,64); t=a@xr; y=t@b.T
    abs_first=np.abs(a)@np.abs(xr)
    e1=gamma(128,F32)*abs_first+128*MIN_NORMAL
    e2=gamma(128,F32)*(np.abs(t)@np.abs(b.T))+(1+gamma(128,F32))*(e1@np.abs(b.T))
    e2+=(gamma(128,F64)+gamma(128,F64)**2)*(abs_first@np.abs(b.T))
    e2+=128*MIN_NORMAL*(1+np.abs(b.T).sum(axis=0))
    return y.reshape(rows,4096),e2.reshape(rows,4096)

def fits(actual, reference, bound):
    import numpy as np
    if actual.shape!=reference.shape or bound.shape!=reference.shape or not all(np.isfinite(v).all() for v in (actual,reference,bound)) or (bound<0).any(): return False
    return bool((np.abs(actual-reference)<=bound).all())

def load_f32(root,e):
    import numpy as np
    p=root/e["file"]; b=p.read_bytes()
    if sha(b)!=e["sha256"] or len(b)!=e["bytes"]: raise ValueError("output payload hash")
    a=np.frombuffer(b,dtype="<f4").astype(np.float64).reshape(e["shape"])
    if not np.isfinite(a).all(): raise ValueError("nonfinite output")
    return a

def decode_float(root,e):
    import numpy as np
    b=(root/e["file"]).read_bytes()
    if sha(b)!=e["sha256"] or len(b)!=e["bytes"]: raise ValueError("packed value hash")
    if e["dtype"]=="BF16":
        a=(np.frombuffer(b,dtype="<u2").astype(np.uint32)<<16).view(np.float32)
    else:a=np.frombuffer(b,dtype="<f4" if e["dtype"]=="F32" else "<f2")
    return a.astype(np.float64).reshape(e["shape"])

def mutants(x,codes,s,b,a,c):
    import numpy as np
    pr,pb=packed_reference(x,codes,s,b); sr,sb=structured_reference(x,a,c)
    # All comparisons are on the actual captured X. These synthetic outputs are declared mutants.
    wrong_nibbles=codes.reshape(codes.shape[0],-1,8)[:,:,::-1].reshape(codes.shape)
    mutants={"packedTranspose":(x@((codes*np.repeat(s,64,axis=1)+np.repeat(b,64,axis=1)).T).T,pr,pb),
        "nibbleOrder":(packed_reference(x,wrong_nibbles,s,b)[0],pr,pb),
        "omittedBias":(packed_reference(x,codes,s,np.zeros_like(b))[0],pr,pb),
        "structuredTranspose":(structured_reference(x,a.T,c)[0],sr,sb),
        "wrongScale":(2*sr,sr,sb),"droppedResidual":(np.zeros_like(sr),sr,sb)}
    result={name:not fits(*v) for name,v in mutants.items()}
    if not all(result.values()):raise ValueError("nondiscriminating mutants: "+str(result))
    return result

def analyze(prepared_root, capture_root, out):
    import numpy as np
    p=read_json(prepared_root/"prepared.json")
    receipts={m:read_json(capture_root/m/"receipt.json") for m in ("default","strict")}
    for mode,r in receipts.items():
        if r["hardwareNaxAvailable"] is not True: raise ValueError("default NAX unavailable")
        if r["mode"]!=mode or r["preparedSha256"]!=sha((prepared_root/"prepared.json").read_bytes()): raise ValueError("receipt identity")
        if (r["ditForwards"],r["trainingSteps"],r["renderCount"])!=(0,0,0): raise ValueError("count drift")
        if r["gemmQmmEvaluations"]!=(4 if mode=="default" else 5): raise ValueError("operation drift")
        if r["runner"]!="nax-macos-2" or r["runAttempt"]!="1":raise ValueError("physical owner")
        if r["originalActivationDtype"]!=r["replayActivationDtype"] or r["originalActivationDtype"]!="Float32":raise ValueError("typed path")
        if r["restoredCacheLimitBytes"]!=r["beforeCacheLimitBytes"] or r["restoredMemoryLimitBytes"]!=r["beforeMemoryLimitBytes"]:raise ValueError("restore")
        if r["physicalPeakBytes"]>4<<30 or not r["watchdogJoined"] or not r["retiredBeforeGuardDrop"]:raise ValueError("lifecycle")
    x=load_f32(prepared_root,p["input"]).reshape(-1,4096)
    a=load_f32(prepared_root,p["w1"]); wa=load_f32(prepared_root,p["w2a"]); wb=load_f32(prepared_root,p["w2b"])
    packed=p["selected"][KEYS[0]]; raw=(prepared_root/packed["file"]).read_bytes()
    if sha(raw)!=packed["sha256"]:raise ValueError("packed words")
    words=np.frombuffer(raw,dtype="<u4").reshape(4096,512)
    codes=((words[:,:,None] >> (4*np.arange(8,dtype=np.uint32)))&15).reshape(4096,4096).astype(np.float64)
    s=decode_float(prepared_root,p["selected"][KEYS[1]]); b=decode_float(prepared_root,p["selected"][KEYS[2]])
    if any(receipts["default"][k]!=receipts["strict"][k] for k in ("source","runId","runAttempt","compiledSourceProofSha256")):raise ValueError("process binding drift")
    for file in capture_root.rglob("*abort*"):raise ValueError("typed abort present: "+str(file))
    default_outputs={k:load_f32(capture_root/"default",v) for k,v in receipts["default"]["outputs"].items()}
    r=receipts["strict"]; rr=capture_root/"strict"; outputs={k:load_f32(rr,v) for k,v in r["outputs"].items() if k!="delta"}
    raww2=outputs["rawW2"]; w2ref=wa@wb
    w2bound=(gamma(32,F32)+gamma(32,F64))*(np.abs(wa)@np.abs(wb))+32*MIN_NORMAL
    checks={"rawFactorProduct":fits(raww2,w2ref,w2bound),
        "storedW1":bool(np.array_equal(outputs["storedW1"],bf16(a))),"storedW2":bool(np.array_equal(outputs["storedW2"],bf16(raww2))),
        "loadedScales":bool(np.array_equal(outputs["loadedScales"],s)),"loadedBiases":bool(np.array_equal(outputs["loadedBiases"],b))}
    pr,pb=packed_reference(x,codes,s,b); sr,sb=structured_reference(x,outputs["storedW1"],outputs["storedW2"])
    checks["packedBase"]=fits(outputs["base"].reshape(-1,4096),pr,pb)
    checks["structured"]=fits(outputs["structured"].reshape(-1,4096),sr,sb)
    e=r["outputs"]["delta"]; path=rr/e["file"]
    with path.open("rb") as f:
        digest=hashlib.sha256()
        for block in iter(lambda:f.read(8<<20),b""):digest.update(block)
    if digest.hexdigest()!=e["sha256"] or path.stat().st_size!=67108864:raise ValueError("delta bytes")
    delta=np.memmap(path,dtype="<f4",mode="r",shape=(4096,4096))
    # First rounding is the production f32 Kronecker product; power-of-two alpha/rank1 is exact.
    expected=bf16((np.kron(a,raww2)).astype(np.float32)).reshape(4096,4096)
    checks["directDeltaReconstruction"]=bool(np.array_equal(delta,expected))
    dr=x@delta.astype(np.float64).T
    db=(gamma(8192,F32)+gamma(8192,F64))*(np.abs(x)@np.abs(delta.astype(np.float64)).T)+8192*MIN_NORMAL
    checks["directResidual"]=fits(outputs["direct"].reshape(-1,4096),dr,db)
    mutation=mutants(x,codes,s,b,outputs["storedW1"],outputs["storedW2"])
    errors={"defaultStrictPackedMaxAbs":float(np.abs(default_outputs["base"]-outputs["base"]).max()),"defaultStrictStructuredMaxAbs":float(np.abs(default_outputs["structured"]-outputs["structured"]).max()),"packedMaxAbs":float(np.abs(outputs["base"].reshape(-1,4096)-pr).max()),
        "structuredMaxAbs":float(np.abs(outputs["structured"].reshape(-1,4096)-sr).max()),"directMaxAbs":float(np.abs(outputs["direct"].reshape(-1,4096)-dr).max())}
    result={"schema":"sc24163-single-linear-cpu-analysis-v1","acceptance":False,"donorAccepted":False,"qualityAccepted":False,
        "strictSelectedTypedPathWithinBounds":all(checks.values()),"checks":checks,"mutants":mutation,"errors":errors,
        "preparedSha256":sha((prepared_root/"prepared.json").read_bytes()),"nativeSource":r["source"],"runId":r["runId"],"compiledSourceProofSha256":r["compiledSourceProofSha256"],
        "scope":"one Float32 to_k, M65 cycling three actual captured rows; no actual full-shape/BF16/image conclusion",
        "defaultPrecisionBoundClaimed":False,"productionFixAutomaticallyJustified":False,
        "stoppingCondition":"A pass ends this discriminator; do not repeat120 training or expand modules on this result."}
    dump(out,result); print(json.dumps(result,indent=2))

def selftest(fixture):
    import numpy as np
    j=read_json(fixture/"bindings.json"); entries={e["role"]:e for e in j["files"]}
    x=load_f32(fixture,entries["input"]).reshape(-1,4096)
    a=bf16(load_f32(fixture,entries["w1"])); wa=load_f32(fixture,entries["w2a"]); wb=load_f32(fixture,entries["w2b"])
    b=bf16((wa@wb).astype(np.float32))
    # Quantized fixtures are synthetic until actual pinned packed operands are read on Mac2.
    codes=np.tile(np.arange(4096,dtype=float)%16,(16,1)); scales=np.full((16,64),.03125); biases=np.full((16,64),-.21875)
    # Avoid the transpose shape fixture depending on a square full model for ordinary preflight.
    pr,pb=packed_reference(x,codes,scales,biases)
    assert fits(pr,pr,pb)
    qr,_=packed_reference(x,codes.reshape(16,-1,8)[:,:,::-1].reshape(16,4096),scales,biases)
    assert not fits(qr,pr,pb),"nibble mutant"
    qr,_=packed_reference(x,codes,scales,np.zeros_like(biases));assert not fits(qr,pr,pb),"bias mutant"
    sr,sb=structured_reference(x,a,b); assert fits(sr,sr,sb)
    assert not fits(structured_reference(x,a.T,b)[0],sr,sb),"transpose mutant"
    assert not fits(2*sr,sr,sb),"scale mutant"
    assert not fits(np.zeros_like(sr),sr,sb),"drop mutant"
    assert not fits(sr[:,::-1],sr,sb),"row-major permutation mutant"
    assert not fits(np.full(sr.shape,np.nan),sr,sb),"nonfinite mutant"
    assert not fits(sr[:,:-1],sr,sb),"shape mutant"
    print(json.dumps({"actualActivationUsed":True,"actualDonorFactorsUsed":True,"syntheticPackedCodesDisclosed":True,
        "mutantsRejected":8,"nativeExecuted":False,"scope":"local comparator semantic preflight"}))

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    s=sub.add_parser("prepare");s.add_argument("--fixture",type=Path,required=True);s.add_argument("--snapshot",type=Path,required=True);s.add_argument("--out",type=Path,required=True)
    s=sub.add_parser("verify-build");s.add_argument("--target",type=Path,required=True);s.add_argument("--out",type=Path,required=True)
    s=sub.add_parser("analyze");s.add_argument("--prepared",type=Path,required=True);s.add_argument("--captures",type=Path,required=True);s.add_argument("--out",type=Path,required=True)
    s=sub.add_parser("selftest");s.add_argument("--fixture",type=Path,required=True)
    a=p.parse_args()
    if a.cmd=="prepare":prepare(a.fixture,a.snapshot,a.out)
    elif a.cmd=="verify-build":verify_build(a.target,a.out)
    elif a.cmd=="analyze":analyze(a.prepared,a.captures,a.out)
    else:selftest(a.fixture)
if __name__=="__main__":main()
