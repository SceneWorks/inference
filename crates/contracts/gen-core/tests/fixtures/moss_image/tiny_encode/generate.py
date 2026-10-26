#!/usr/bin/env python3
"""Independent frozen Diffusers/Comfy CPU encoding oracle; synthetic tiny VAE."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import sys
sys.dont_write_bytecode=True
import ast,hashlib,json,math,signal,tempfile,time,zipfile
from pathlib import Path
signal.alarm(120)
started=time.monotonic()
HERE=Path(__file__).resolve().parent
UPSTREAM=HERE.parent
WHEEL_SHA='18e53f9e539096320470f62c6360a6fd5727ff28cffda566265316e13fcdb612'
ENCODE_SHA='43e5fd31a40500cb720b573b547e39e9b0a298837b2ac3bd21f7a7d8bd8c521f'
CONFIG_SHA='cc77f3f0ce0357039c71018dc19edaa4fcceb0c2f7488fdc2b9c0580d6baafa1'
wheel=HERE/'diffusers-0.38.0-py3-none-any.whl'
assert hashlib.sha256(wheel.read_bytes()).hexdigest()==WHEEL_SHA
source=UPSTREAM/'ComfyUI-Moss-Image-Edit/moss_runtime/processing.py'
source_bytes=source.read_bytes()
assert hashlib.sha256(source_bytes).hexdigest()==ENCODE_SHA
official=UPSTREAM/'official-sft/vae/config.json'
official_bytes=official.read_bytes()
assert hashlib.sha256(official_bytes).hexdigest()==CONFIG_SHA
with tempfile.TemporaryDirectory(prefix='diffusers-source-',dir=HERE) as unpacked:
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith('diffusers/'):
                archive.extract(name,unpacked)
        license_name=next(n for n in archive.namelist() if n.endswith('/LICENSE'))
        license_bytes=archive.read(license_name)
        (HERE/'LICENSE.diffusers').write_bytes(license_bytes)
    sys.path.insert(0,unpacked)
    import torch,diffusers
    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
    assert diffusers.__version__=='0.38.0'
    torch.set_default_device('cpu')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(26753)
    config={k:v for k,v in json.loads(official_bytes).items() if not k.startswith('_')}
    # Explicitly agreed native-injectable synthetic geometry. All other official settings retained.
    config['block_out_channels']=[32,32,32,32]
    config['decoder_block_out_channels']=[32,32,32,32]
    model=AutoencoderKLFlux2(**config).to(device='cpu',dtype=torch.float32).eval()
    ns=dict(math=math)
    tree=ast.parse(source_bytes,filename=str(source))
    names=['_vae_placement','_bn_stats','encode_image']
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert [n.name for n in nodes]==names
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    recipes=[]
    probes={}
    with torch.no_grad():
        for name,t in model.state_dict().items():
            if not (name.startswith(('encoder.','quant_conv.','decoder.','post_quant_conv.')) or name.startswith('bn.running_')):
                continue
            shape=list(t.shape)
            seed=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')
            index=torch.arange(t.numel(),device='cpu',dtype=torch.int64)
            bits=(index+1)^seed
            bits=(bits^(bits<<13))&0xffffffff
            bits=bits^(bits>>17)
            bits=(bits^(bits<<5))&0xffffffff
            signed=(bits&65535)-32768
            kind='weight'
            if name=='bn.running_mean':
                values=((index%17)-8).float()/64
                kind='bn_mean'
            elif name=='bn.running_var':
                values=1+((index*7)%31).float()/32
                kind='bn_var'
            elif name in ('post_quant_conv.weight','quant_conv.weight'):
                channels=shape[1]
                values=signed.float()/8388608+(index//channels==index%channels).float()
                kind='pointwise_identity'
            elif len(shape)==1 and name.endswith('.weight'):
                values=1+signed.float()/262144
                kind='norm'
            elif name.endswith('.bias'):
                values=signed.float()/1048576
                kind='bias'
            else:
                values=signed.float()/8388608
            t.copy_(values.reshape(shape))
            recipes.append(dict(name=name,shape=shape,seed=seed,kind=kind))
            probes[name]=values[:4].tolist()
        assert all(p.device.type=='cpu' for p in model.parameters())
        pixels=torch.tensor([((i*37+i//3*11)%251-125)/128 for i in range(3*16*32)],device='cpu',dtype=torch.float32).reshape(1,3,16,32)
        posterior=model.encode(pixels).latent_dist
        mean=posterior.mode()
        expected=ns['encode_image'](model,pixels,sample_posterior=False)
        sample_wrong=ns['encode_image'](model,pixels,sample_posterior=True,generator=torch.Generator('cpu').manual_seed(26753))
        b,c,h,w=mean.shape
        raw_packed=mean.reshape(b,c,h//2,2,w//2,2).permute(0,1,3,5,2,4).reshape(b,c*4,h//2,w//2).float()
        patch_major=mean.reshape(b,c,h//2,2,w//2,2).permute(0,3,5,1,2,4).reshape(b,c*4,h//2,w//2).float()
        bn_mean=model.bn.running_mean.reshape(1,-1,1,1)
        bn_std=(model.bn.running_var+model.config.batch_norm_eps).sqrt().reshape(1,-1,1,1)
        wrong_packing=(patch_major-bn_mean)/bn_std
        missing_bn=raw_packed
        assert torch.equal(expected,(raw_packed-bn_mean)/bn_std)
        mutants=dict(posterior_sample=sample_wrong,patch_major_packing=wrong_packing,missing_bn=missing_bn)
        deltas={k:float((expected-v).abs().max()) for k,v in mutants.items()}
        assert min(deltas.values())>0.001,deltas
        def array(t):
            assert t.device.type=='cpu' and torch.isfinite(t).all()
            return dict(shape=list(t.shape),dtype=str(t.dtype).removeprefix('torch.'),values=t.detach().flatten().tolist())
        source_hashes={}
        for relative in ['models/autoencoders/autoencoder_kl_flux2.py','models/autoencoders/vae.py','models/unets/unet_2d_blocks.py','models/resnet.py','models/attention_processor.py']:
            source_hashes[relative]=hashlib.sha256((Path(unpacked)/'diffusers'/relative).read_bytes()).hexdigest()
        fixture=dict(schema=1,device='cpu',real_weights=False,native_execution=False,config=config,
            source_wheel_sha256=WHEEL_SHA,encoder_helper_sha256=ENCODE_SHA,official_config_sha256=CONFIG_SHA,source_sha256=source_hashes,
            torch_version=torch.__version__,diffusers_version=diffusers.__version__,
            formula='sha256-name-seeded xorshift32 exact-power-of-two v1; pointwise_identity uses actual input channels; see generate.py',
            weights=recipes,weight_prefix_probes=probes,pixels_nchw=array(pixels),pixels_nhwc=array(pixels.permute(0,2,3,1)),
            mean_nchw=array(mean),mean_nhwc=array(mean.permute(0,2,3,1)),posterior_logvar_nchw=array(posterior.logvar),
            raw_packed_nchw=array(raw_packed),normalized_packed_nchw=array(expected),bn_mean=array(bn_mean),bn_std=array(bn_std),
            mutants={k:array(v) for k,v in mutants.items()},mutation_max_abs=deltas,absolute_tolerance=0.0001)
        path=HERE/'fixture.json'
        path.write_text(json.dumps(fixture,separators=(',',':'))+'\n')
        receipt=dict(status='PASS_CPU_FROZEN_DIFFUSERS_ENCODE',fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_wheel_sha256=WHEEL_SHA,encoder_helper_sha256=ENCODE_SHA,official_config_sha256=CONFIG_SHA,source_sha256=source_hashes,license_sha256=hashlib.sha256(license_bytes).hexdigest(),device='cpu',threads=1,real_weights=False,native_execution=False,config=config,torch_version=torch.__version__,diffusers_version=diffusers.__version__,weight_tensors=len(recipes),weight_elements=sum(math.prod(r['shape']) for r in recipes),mutation_max_abs=deltas,absolute_tolerance=0.0001,seconds=time.monotonic()-started,bytes=path.stat().st_size)
        (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt,indent=2))
signal.alarm(0)
