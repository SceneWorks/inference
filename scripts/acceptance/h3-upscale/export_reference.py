"""Export weights-free goldens from the actual pinned upstream class (CPU only).

No implementation-under-test is imported. Upstream source must be supplied from
the immutable commit; its SHA256 is recorded beside the fixture.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comfy-model", type=Path, required=True)
    parser.add_argument("--guide-source", type=Path, required=True)
    parser.add_argument("--comfy-sampling", type=Path, required=True)
    parser.add_argument("--comfy-samplers", type=Path, required=True)
    parser.add_argument("--comfy-vae", type=Path, required=True)
    parser.add_argument("--runtime-noise", type=Path)
    args = parser.parse_args()
    source = args.upstream.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {"_gn", "_Res", "_Temporal", "LatentUpscaler"}
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in selected} != names:
        raise ValueError("pinned upstream network definitions missing")
    scope = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional, "TAG": "reference fixture"}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(args.upstream), "exec"), scope)
    # Nontrivial fixed graph: residual then temporal in each side, 32 GN groups,
    # 24 input channels, time>1, anisotropic spatial input, nonzero biases.
    state = {}
    def add(name, shape, gain):
        count = 1
        for extent in shape:
            count *= extent
        state[name] = (torch.sin(torch.arange(count, dtype=torch.float32) * .37 + len(name)) * gain).reshape(shape)
    def affine(name, channels):
        add(name + ".weight", (channels,), .07)
        state[name + ".weight"] += 1
        add(name + ".bias", (channels,), .025)
    def conv(name, output, input_, kernel):
        add(name + ".weight", (output, input_, *kernel), .018)
        add(name + ".bias", (output,), .025)
    conv("conv_in", 32, 24, (3,3,3)); conv("conv_out",24,32,(3,3,3))
    for name, shape in [("embed.0",(16,1)),("embed.2",(16,16))]:
        add(name+".weight",shape,.07); add(name+".bias",(shape[0],),.025)
    affine("norm_out",32)
    for side in ("in_blocks","out_blocks"):
        stem = side + ".0"
        affine(stem+".in_layers.0",32); conv(stem+".in_layers.2",32,32,(3,3,3))
        add(stem+".emb_layers.1.weight",(64,16),.04); add(stem+".emb_layers.1.bias",(64,),.025)
        affine(stem+".out_norm",32); conv(stem+".out_layers.2",32,32,(3,3,3))
        stem=side+".1"; affine(stem+".norm",32)
        conv(stem+".dwconv",32,1,(3,1,1)); conv(stem+".pwconv",32,32,(1,1,1))
    net = scope["LatentUpscaler"](state).eval()
    net.load_state_dict(state, strict=True)
    x = (torch.cos(torch.arange(24*3*2*3, dtype=torch.float32)*.13)*.4).reshape(1,24,3,2,3)
    with torch.inference_mode():
        output=net(x,(3,4,6),2.)
        resized=torch.nn.functional.interpolate(x,size=(3,4,6),mode="trilinear",align_corners=False)
    tensors={"weight."+k:v.contiguous() for k,v in state.items()}
    tensors.update({"input":x,"output":output,"resized":resized})
    # Pinned Comfy H3 VAE encode returns (posterior.mean - mean) / std.
    # The upsampler then applies its own identical statistics transform before
    # the network, and reverses only that transform. The output is normalized
    # VAE/DiT state, not posterior raw state. This composition catches the
    # superficially plausible raw->network->VAE-normalize boundary error.
    constants=[node for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id in {"LATENT_MEAN","LATENT_STD"} for target in node.targets)]
    exec(compile(ast.Module(body=constants,type_ignores=[]),str(args.upstream),"exec"),scope)
    mean=torch.tensor(scope["LATENT_MEAN"]).reshape(1,24,1,1,1)
    std=torch.tensor(scope["LATENT_STD"]).reshape(1,24,1,1,1)
    from types import SimpleNamespace
    vae_source=args.comfy_vae.read_text(encoding="utf-8")
    vae_tree=ast.parse(vae_source)
    vae_constants={}
    assignments=[node for node in vae_tree.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id in {"LATENTS_MEAN","LATENTS_STD"} for target in node.targets)]
    exec(compile(ast.Module(body=assignments,type_ignores=[]),str(args.comfy_vae),"exec"),vae_constants)
    vae_class=next(node for node in vae_tree.body if isinstance(node,ast.ClassDef) and node.name=="MiniMaxH3VideoVAE")
    encode=next(node for node in vae_class.body if isinstance(node,ast.FunctionDef) and node.name=="encode")
    exec(compile(ast.Module(body=[encode],type_ignores=[]),str(args.comfy_vae),"exec"),scope)
    moments=torch.cat([x,torch.zeros_like(x)],dim=1)
    posterior=SimpleNamespace(latents_mean=torch.tensor(vae_constants["LATENTS_MEAN"]),latents_std=torch.tensor(vae_constants["LATENTS_STD"]),encode_temporal=lambda pixels,device:moments)
    normalized=scope["encode"](posterior,torch.zeros(1,3,3,2,3))
    with torch.inference_mode():
        composed=net((normalized-mean)/std,(3,4,6),2.)*std+mean
    tensors.update({"boundary.raw":x.clone(),"boundary.normalized":normalized,"boundary.upscale":composed})
    from types import SimpleNamespace
    sampling_source=args.comfy_sampling.read_text(encoding="utf-8")
    selected=[node for node in ast.parse(sampling_source).body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in {"time_snr_shift","ModelSamplingDiscreteFlow","ModelSamplingAV"}]
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(args.comfy_sampling),"exec"),scope)
    sampler_source=args.comfy_samplers.read_text(encoding="utf-8")
    selected=[node for node in ast.parse(sampler_source).body if isinstance(node,ast.FunctionDef) and node.name=="simple_scheduler"]
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(args.comfy_samplers),"exec"),scope)
    schedule=scope["ModelSamplingAV"](SimpleNamespace(sampling_settings={"shift":12.,"audio_shift":3.}))
    fractions=[.1,.2,.05,.0011,1.]
    denoise=torch.tensor(fractions)
    sigma=torch.stack([scope["simple_scheduler"](schedule,int(1./d))[-2] for d in fractions])
    tensors.update({"schedule.denoise":denoise,"schedule.sigma":sigma})
    model_source=args.comfy_model.read_text(encoding="utf-8")
    model_names={"_axis_from_sqrt_area","_frame_grid","_video_t_spans","_video_t_grid","_video_grid","_audio_grid","_ref_t_span","PackedLayout"}
    definitions=[node for node in ast.parse(model_source).body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in model_names]
    layout_scope={"torch":torch,"math":math,"FRAME_PER_TOKEN":(1,4,4,4,4),"FRAME_RESCALE":5./3.}
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(args.comfy_model),"exec"),layout_scope)
    guide_source=args.guide_source.read_text(encoding="utf-8")
    definitions=[node for node in ast.parse(guide_source).body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in {"_ShapeOnly","_shrink"}]
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(args.guide_source),"exec"),layout_scope)
    latent=torch.zeros(1,24,12,18,32)
    stand_in=layout_scope["_ShapeOnly"](12)
    layout=layout_scope["PackedLayout"](3,12,36,64,65,keyframes=[{"latent":stand_in,"resolved_frame_index":0}])
    layout_scope["_shrink"](layout,36,64,[{"latent":latent}],[2])
    tensors.update({"layout.positions":layout.position_ids,"layout.video_indices":layout.img_pos,
                    "layout.audio_indices":layout.audio_pos,"layout.img_update":layout.img_update.to(torch.int64)})
    rgb=(torch.sin(torch.arange(3*3*4*6,dtype=torch.float32)*.7)*.5+.5).reshape(1,3,3,4,6)
    resized_rgb=torch.nn.functional.interpolate(rgb.flatten(0,1).unsqueeze(1),size=(3,8,12),mode="trilinear",align_corners=False)
    # Source guide path is separable 2D antialiased bicubic at each time.
    planes=rgb.permute(0,2,1,3,4).reshape(3,3,4,6)
    planes=torch.nn.functional.interpolate(planes,size=(8,12),mode="bicubic",align_corners=False,antialias=True).clamp(0,1)
    guide_pixels=torch.nn.functional.interpolate(planes,size=(4,6),mode="bicubic",align_corners=False,antialias=True).clamp(0,1).reshape(1,3,3,4,6).permute(0,2,1,3,4).contiguous()
    tensors.update({"guide.rgb":rgb,"guide.output":guide_pixels})
    if args.runtime_noise:
        noise={"video":torch.randn((1,24,12,36,64),generator=torch.Generator().manual_seed(444)),
               "guide":torch.randn((1,12*9*16,96),generator=torch.Generator().manual_seed(444))}
        save_file(noise,str(args.runtime_noise))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    save_file(tensors,str(args.out))
    args.out.with_suffix(".json").write_text(json.dumps({
        "upstream_commit":"36cb612ec3df30094eeb4fec66f1528925ebba73",
        "upstream_file_sha256":hashlib.sha256(source.encode()).hexdigest(),
        "comfy_commit":"7a5dad695fe1cae25efcb2550530fb20ef68da3d",
        "comfy_model_sha256":hashlib.sha256(model_source.encode()).hexdigest(),
        "comfy_sampling_sha256":hashlib.sha256(sampling_source.encode()).hexdigest(),
        "comfy_samplers_sha256":hashlib.sha256(sampler_source.encode()).hexdigest(),
        "comfy_vae_sha256":hashlib.sha256(vae_source.encode()).hexdigest(),
        "guide_source_sha256":hashlib.sha256(guide_source.encode()).hexdigest(),
        "torch":torch.__version__,"device":"cpu","dtype":"float32",
        "max_absolute_tolerance":0.00003,"max_relative_tolerance":0.0001,
        "fixture_sha256":hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "role":"weights-free algorithm reference; not quality or real-weight runtime evidence"
    },indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
