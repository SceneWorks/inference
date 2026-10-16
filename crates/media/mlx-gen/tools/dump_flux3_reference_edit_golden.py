"""CPU tiny ordered-reference editing oracle, frozen BFL45e57c9 (sc-26236).
Reads existing snapshots; writes only flux3_reference_edit_golden.safetensors.
FLUX3_SRC=<frozen checkout> python -I tools/dump_flux3_reference_edit_golden.py
No real weights, network, accelerator initialization, or production Python path.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from _flux3_common import import_upstream, patch_dit_constants, tiny_autoencoder, FIXTURE_DIR, SNAPSHOT, FLUX3_COMMIT
import_upstream()
import torch
import numpy as np
from PIL import Image
from safetensors.torch import load_file,save_file
import flux3.dit as dit_module
import flux3.autoencoder as ae_module
import flux3.pipeline as pipeline
import flux3.sampling as sampling

torch.set_num_threads(2)
patch_dit_constants(dit_module)
ae=tiny_autoencoder(ae_module)
flow=dit_module.DiT(flow_map=True).eval()
flow.load_state_dict({k:v.float() for k,v in load_file(str(SNAPSHOT/'model.safetensors')).items()},strict=True)
base=dit_module.DiT(flow_map=False).eval()
base.load_state_dict({k:v.float() for k,v in load_file(str(FIXTURE_DIR/'tiny-base-snapshot/model.safetensors')).items()},strict=True)
text=load_file(str(FIXTURE_DIR/'flux3_text_golden.safetensors'))
inputs=load_file(str(FIXTURE_DIR/'flux3_dit_golden.safetensors'))
x,ids=inputs['single/x'],inputs['single/image_ids']
pos=(text['short/tokens'],text['short/ids'])
neg=(text['empty/tokens'],text['empty/ids'])
schedule=sampling.timesteps(4,6.93)
out={'noise':x,'image_ids':ids,'schedule':schedule}
originals=[]
for i in range(10):
    w,h=(83,67) if i%2==0 else (67,99)
    y,xx=np.indices((h,w),dtype=np.uint32)
    rgb=np.stack(((xx*17+y*7+i*23)%256,(xx*3+y*19+i*41)%256,(xx*29+y*11+i*13)%256),axis=-1).astype(np.uint8)
    originals.append(Image.fromarray(rgb))
    out[f'input/{i}']=torch.from_numpy(rgb.copy())
    alpha=np.array([0,128,255],dtype=np.uint8)[(xx+y)%3]
    rgba=np.concatenate((rgb,alpha[...,None]),axis=-1)
    rgba_image=Image.fromarray(rgba)
    out[f'rgba/{i}']=torch.from_numpy(rgba.copy())
    white=Image.alpha_composite(Image.new('RGBA',rgba_image.size,'white'),rgba_image).convert('RGB')
    out[f'pup_rgb/{i}']=torch.from_numpy(np.asarray(white).copy())
    # Frozen AE policy intentionally ignores alpha; PUP's fork does NOT.
    for prepare in (lambda im:pipeline.prepare_reference(im)[0],pipeline.prepare_multi_reference):
        assert np.array_equal(np.asarray(prepare(rgba_image)),np.asarray(prepare(originals[-1])))

def pack(cond,base_mode):
    samples=[[neg,cond,(x.clone(),ids)],[pos,cond,(x.clone(),ids)]] if base_mode else [[pos,cond,(x.clone(),ids)]]
    packed=pipeline.pack(samples)
    packed.data={m:v.float() for m,v in packed.data.items()}
    packed.timesteps={m:v.float() for m,v in packed.timesteps.items()}
    return packed

with torch.inference_mode():
    for name,images in [('one',originals[:1]),('two',originals[:2]),('ten',originals),('reversed',list(reversed(originals)))]:
        prepared=[pipeline.prepare_multi_reference(im) for im in images] if len(images)>1 else [pipeline.prepare_reference(images[0])[0]]
        cond=pipeline.Pipeline.encode_references(SimpleNamespace(autoencoder=ae,device='cpu'),prepared)
        out[name+'/reference_tokens']=cond[0].float()
        out[name+'/reference_ids']=cond[1].float()
        tokens=sampling.denoise_flow_map(flow,pack(cond,False),schedule)
        out[name+'/flow_tokens']=tokens
        decoded=ae.decode(tokens,ids/torch.tensor([1.,.5,.5,1.]))
        out[name+'/decoded']=decoded
        out[name+'/rgb8']=torch.from_numpy(np.asarray(pipeline.to_pil(decoded)).copy())
        if name in ('two','ten'):
            for guidance in (0.,1.,3.5):
                out[name+f'/cfg_{guidance}']=sampling.denoise(base,pack(cond,True),schedule,guidance)
assert not torch.allclose(out['ten/flow_tokens'],out['reversed/flow_tokens'],atol=1e-6)
metadata={'upstream_code_revision':FLUX3_COMMIT,'oracle':'frozen Pipeline.encode_references/pack + sampling.denoise[_flow_map], CPU FP32 over BF16-rounded reference tokens; tiny existing weights','references':'ordered distinct non-square RGB; one/two/ten/reversed; base always negative/positive'}
save_file({k:v.contiguous() for k,v in out.items()},str(FIXTURE_DIR/'flux3_reference_edit_golden.safetensors'),metadata=metadata)
print('tiny source AE conditioning + 1/2/10/reversed flow edits + base0/1/3.5 reference fixtures written')
