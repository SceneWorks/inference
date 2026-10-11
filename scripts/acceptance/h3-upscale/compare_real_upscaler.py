"""Independent reference comparison of native captured real-weight upscaler latents.

This is optional prototype/reference tooling; it does not execute product work.
Compare max error normalized by reference maximum as well as max absolute error.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import load_file


def main():
    parser=argparse.ArgumentParser()
    for name in ("upstream","weights","intermediates","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--source-reference",type=Path,help="independent fixed-RGB VAE export as learned-network input")
    args=parser.parse_args()
    torch.set_num_threads(16)
    source=args.upstream.read_text(encoding="utf-8")
    names={"_gn","_Res","_Temporal","LatentUpscaler"}
    tree=ast.parse(source)
    body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    scope={"torch":torch,"nn":torch.nn,"F":torch.nn.functional,"TAG":"real reference"}
    exec(compile(ast.Module(body=body,type_ignores=[]),str(args.upstream),"exec"),scope)
    # Import the actual upstream normalization constants, not the Rust values.
    constants=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {"LATENT_MEAN","LATENT_STD"} for t in n.targets)]
    exec(compile(ast.Module(body=constants,type_ignores=[]),str(args.upstream),"exec"),scope)
    sd=load_file(str(args.weights),device="cpu")
    if any(k.startswith("upscaler.") for k in sd):
        sd={k[len("upscaler."):]:v for k,v in sd.items() if k.startswith("upscaler.")}
    net=scope["LatentUpscaler"](sd).eval().half().to(args.device)
    net.load_state_dict(sd,strict=True)
    captured=load_file(str(args.intermediates),device="cpu")
    independent=load_file(str(args.source_reference),device="cpu") if args.source_reference else None
    source_normalized=(independent if independent is not None else captured)["source.normalized"].half().to(args.device)
    mean=torch.tensor(scope["LATENT_MEAN"],dtype=torch.float16,device=args.device).reshape(1,24,1,1,1)
    std=torch.tensor(scope["LATENT_STD"],dtype=torch.float16,device=args.device).reshape(1,24,1,1,1)
    with torch.inference_mode():
        expected=(net((source_normalized-mean)/std,(12,36,64),2.)*std+mean).float().cpu()
    from safetensors.torch import save_file
    save_file({"upscale.normalized":expected},str(args.out.with_suffix(".safetensors")))
    actual=captured["upscale.normalized"].float()
    diff=(actual-expected).abs()
    absolute=float(diff.max()); relative=absolute/max(float(expected.abs().max()),1e-12)
    # FP16 tap-by-tap conv3d accumulates differently from cuDNN's fused conv3d.
    # Declared deviation, bounded by both absolute and relative error.
    result={"upstream_commit":"36cb612ec3df30094eeb4fec66f1528925ebba73","torch":torch.__version__,"reference_device":args.device,
            "max_abs":absolute,"max_relative":relative,"rms":float(diff.square().mean().sqrt()),
            "tolerances":{"max_abs":.08,"max_relative":.015},
            "pass":absolute<=.08 and relative<=.015,
            "reference_dtype":"fp16",
            "upstream_source_sha256":hashlib.sha256(args.upstream.read_bytes()).hexdigest(),
            "weights_sha256":hashlib.sha256(args.weights.read_bytes()).hexdigest(),
            "native_intermediates_sha256":hashlib.sha256(args.intermediates.read_bytes()).hexdigest(),
            "expected_sha256":hashlib.sha256(args.out.with_suffix(".safetensors").read_bytes()).hexdigest(),
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "named_deviation":"Candle FP16 Conv3d temporal-tap accumulation and intermediate separable resize rounding versus upstream fused PyTorch Conv3d/trilinear; checked on actual input/weights"}
    if args.source_reference:
        result["independent_source_reference_sha256"]=hashlib.sha256(args.source_reference.read_bytes()).hexdigest()
        result["reference_input"]="independent pinned real VAE export from fixed RGB; native latents are comparison-only"
    args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    if not result["pass"]:
        raise SystemExit("real upscaler numerical comparison failed")


if __name__=="__main__":
    main()
