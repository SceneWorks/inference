#!/usr/bin/env python3
"""Regenerate the actual-width MOSS VAE oracle on CPU, without installing packages or loading weights.

Use python -B, --diffusers-wheel (the frozen 0.38.0 wheel), and --author-pipeline
(the existing frozen Apache MOSS pipeline). Outputs contain recipes, not tensor payloads.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json, signal, tempfile, time, zipfile
from pathlib import Path
signal.alarm(120)
started = time.monotonic()
WHEEL_SHA = '18e53f9e539096320470f62c6360a6fd5727ff28cffda566265316e13fcdb612'
PIPELINE_SHA = '1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--diffusers-wheel', type=Path, required=True)
parser.add_argument('--author-pipeline', type=Path, required=True)
parser.add_argument('--vae-config', type=Path, default=Path(__file__).resolve().parent/'config.json')
parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
assert hashlib.sha256(args.diffusers_wheel.read_bytes()).hexdigest() == WHEEL_SHA
assert hashlib.sha256(args.author_pipeline.read_bytes()).hexdigest() == PIPELINE_SHA
args.output_dir.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='moss-small-vae-author-') as unpacked:
    with zipfile.ZipFile(args.diffusers_wheel) as archive:
        for entry in archive.namelist():
            if entry.startswith('diffusers/'):
                archive.extract(entry, unpacked)
        license_name = next(n for n in archive.namelist() if n.endswith('/LICENSE'))
        license_bytes = archive.read(license_name)
        (args.output_dir / 'LICENSE.diffusers').write_bytes(license_bytes)
    sys.path.insert(0, unpacked)
    import torch, diffusers
    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
    assert diffusers.__version__ == '0.38.0'
    torch.set_default_device('cpu')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(26325)
    config_bytes = args.vae_config.read_bytes()
    assert hashlib.sha256(config_bytes).hexdigest() == 'cc77f3f0ce0357039c71018dc19edaa4fcceb0c2f7488fdc2b9c0580d6baafa1'
    config = {k:v for k,v in json.loads(config_bytes).items() if not k.startswith('_')}
    model = AutoencoderKLFlux2(**config).to(device='cpu', dtype=torch.float32).eval()
    recipes = []
    probes = {}
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if not (name.startswith('decoder.') or name.startswith('post_quant_conv.') or name.startswith('bn.running_')):
                continue
            shape = list(tensor.shape)
            seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
            index = torch.arange(tensor.numel(), device='cpu', dtype=torch.int64)
            bits = (index + 1) ^ seed
            bits = (bits ^ (bits << 13)) & 0xffffffff
            bits = bits ^ (bits >> 17)
            bits = (bits ^ (bits << 5)) & 0xffffffff
            signed = (bits & 65535) - 32768
            kind = 'weight'
            if name == 'bn.running_mean':
                values = ((index % 17) - 8).float() / 64
                kind = 'bn_mean'
            elif name == 'bn.running_var':
                values = 1 + ((index * 7) % 31).float() / 32
                kind = 'bn_var'
            elif name == 'post_quant_conv.weight':
                values = signed.float() / 8388608 + (index // 32 == index % 32).float()
                kind = 'post_quant_identity'
            elif len(shape) == 1 and name.endswith('.weight'):
                values = 1 + signed.float() / 262144
                kind = 'norm'
            elif name.endswith('.bias'):
                values = signed.float() / 1048576
                kind = 'bias'
            else:
                values = signed.float() / 8388608
            tensor.copy_(values.reshape(shape))
            recipes.append(dict(name=name, shape=shape, seed=seed, kind=kind))
            probes[name] = values[:4].tolist()
        assert all(p.device.type == 'cpu' for p in model.parameters())
        # Source-faithful MOSS pipeline BN de-normalization and channel-major unpatchify.
        packed = torch.tensor([((i * 37) % 251 - 125) / 128 for i in range(128)], dtype=torch.float32, device='cpu').reshape(1,128,1,1)
        def decode(packed, *, wrong_order=False, omit_bn=False):
            z = packed
            if not omit_bn:
                z = z * (model.bn.running_var + model.config.batch_norm_eps).sqrt().view(1,-1,1,1) + model.bn.running_mean.view(1,-1,1,1)
            b,c,h,w = z.shape
            if wrong_order:
                z = z.reshape(b,2,2,c//4,h,w).permute(0,3,4,1,5,2).reshape(b,c//4,2*h,2*w)
            else:
                z = z.reshape(b,c//4,2,2,h,w).permute(0,1,4,2,5,3).reshape(b,c//4,2*h,2*w)
            return model.decode(z, return_dict=False)[0]
        expected = decode(packed)
        wrong_order = decode(packed, wrong_order=True)
        wrong_bn = decode(packed, omit_bn=True)
        for pixels in (expected,wrong_order,wrong_bn):
            assert pixels.device.type == 'cpu' and torch.isfinite(pixels).all()
        deltas = dict(wrong_patch_order=float((expected-wrong_order).abs().max()), missing_bn=float((expected-wrong_bn).abs().max()))
        print(json.dumps({"mutation_max_abs": deltas}), file=sys.stderr)
        assert min(deltas.values()) > 0.002
        def array(t):
            return dict(shape=list(t.shape),values=t.flatten().tolist())
        sources = {}
        for relative in ['models/autoencoders/autoencoder_kl_flux2.py','models/autoencoders/vae.py','models/unets/unet_2d_blocks.py','models/resnet.py','models/attention_processor.py']:
            sources[relative] = hashlib.sha256((Path(unpacked)/'diffusers'/relative).read_bytes()).hexdigest()
        fixture = dict(schema=1,torch_version=torch.__version__,diffusers_version=diffusers.__version__,device='cpu',
            source_wheel_sha256=WHEEL_SHA,author_pipeline_sha256=PIPELINE_SHA,config_sha256=hashlib.sha256(config_bytes).hexdigest(),source_sha256=sources,
            config=config,formula='sha256-name-seeded xorshift32 v1; exact power-of-two FP32 recipes; see generate.py',
            weights=recipes,weight_prefix_probes=probes,packed_nchw=array(packed),pixels_nchw=array(expected),
            mutants=dict(wrong_patch_order=array(wrong_order),missing_bn=array(wrong_bn)),mutation_max_abs=deltas,
            absolute_tolerance=0.0001,real_weights=False)
        (args.output_dir/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
        receipt=dict(status='PASS_CPU_FROZEN_AUTHOR_SMALL_VAE',seconds=time.monotonic()-started,torch_version=torch.__version__,
            diffusers_version=diffusers.__version__,device='cpu',threads=2,decoder_widths=config['decoder_block_out_channels'],
            weight_tensors=len(recipes),weight_elements=sum(t.numel() for n,t in model.state_dict().items() if n in probes),
            fixture_bytes=(args.output_dir/'fixture.json').stat().st_size,fixture_sha256=hashlib.sha256((args.output_dir/'fixture.json').read_bytes()).hexdigest(),
            mutation_max_abs=deltas,source_wheel_sha256=WHEEL_SHA,license_sha256=hashlib.sha256(license_bytes).hexdigest(),native_execution=False)
        print(json.dumps(receipt,indent=2))
