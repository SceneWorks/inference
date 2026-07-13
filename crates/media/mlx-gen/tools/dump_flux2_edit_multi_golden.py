"""Real-weights end-to-end golden for FLUX.2-klein **multi-image** EDIT (`MultiReference`, sc-2645),
for the #[ignore]d Rust parity test. Runs the fork's edit pipeline with TWO distinct reference
images, capturing the concatenated 2-ref `image_latents`, the step-0 velocity, the final latents,
the decoded image, and the two resized reference pixels.

Two distinct refs (`flux2_klein_edit.jpg` + `flux2_klein.jpg`) are each resized to 256² (LANCZOS)
and persisted as lossless PNGs so the fork and the Rust test consume byte-identical u8 pixels (the
in-pipeline LANCZOS resize is then a no-op).

Dense (default): `ModelConfig.precision = float32`, dumps `image_latents` / `v0` / `latents` /
`decoded` → `flux2_edit_multi.safetensors`. Quantized (`BITS=8`): default bf16 precision +
`quantize=BITS`, dumps only `decoded` (+ refs) → `flux2_edit_multi_q{bits}.safetensors` for the
coherence-floor render gate.

Gitignored output. Run from the mflux fork venv:
    cd ~/repos/mflux && .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_edit_multi_golden.py
    cd ~/repos/mflux && BITS=8 .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_edit_multi_golden.py
"""

import os
import tempfile

import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.models.common.config.model_config import ModelConfig

BITS = int(os.environ["BITS"]) if os.environ.get("BITS") else None
if BITS is None:
    ModelConfig.precision = mx.float32  # the dense Rust pipeline runs f32 activations

from mflux.models.common.config import ModelConfig as MC  # noqa: E402
from mflux.models.common.config.config import Config  # noqa: E402
from mflux.models.flux2.variants import Flux2KleinEdit  # noqa: E402
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers  # noqa: E402

from _paths import fixture  # noqa: E402

PROMPT = "blend the two scenes into a single dreamlike landscape"
SEED, STEPS, SIZE, GUIDANCE = 0, 4, 256, 1.0
ASSETS = [
    "/Users/michael/repos/mflux/src/mflux/assets/flux2_klein_edit.jpg",
    "/Users/michael/repos/mflux/src/mflux/assets/flux2_klein.jpg",
]

# Resize each ref to 256² (LANCZOS) → lossless PNG so the fork and Rust see byte-identical pixels.
refs_u8 = []
tmp_paths = []
for asset in ASSETS:
    im = PIL.Image.open(asset).convert("RGB").resize((SIZE, SIZE), PIL.Image.LANCZOS)
    refs_u8.append(np.array(im, dtype=np.uint8))  # [256,256,3]
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    im.save(tmp.name)
    tmp_paths.append(tmp.name)

print(f"FLUX.2-klein-9b multi-edit golden: bits={BITS}, precision={MC.precision}, refs={len(ASSETS)}")
model = Flux2KleinEdit(quantize=BITS, model_config=MC.flux2_klein_9b())
config = Config(
    model_config=model.model_config,
    num_inference_steps=STEPS,
    height=SIZE,
    width=SIZE,
    guidance=GUIDANCE,
    image_path=tmp_paths[0],
    scheduler="flow_match_euler_discrete",
)

prompt_embeds, text_ids, neg_embeds, neg_ids = model._encode_prompt_pair(
    prompt=PROMPT, negative_prompt=" ", guidance=GUIDANCE
)
latents, latent_ids, lat_h, lat_w = _Flux2KleinEditHelpers.prepare_generation_latents(
    seed=SEED, height=SIZE, width=SIZE
)
# N-reference conditioning: each ref VAE-encoded → patchify → BN-normalize → pack, t=10+10·i ids,
# concatenated on the sequence axis.
image_latents, image_latent_ids = _Flux2KleinEditHelpers.prepare_reference_image_conditioning(
    vae=model.vae, tiling_config=model.tiling_config, image_paths=tmp_paths, height=SIZE, width=SIZE, batch_size=1
)

predict = model._predict(model.transformer)
v0 = None
for t in range(config.init_time_step, config.num_inference_steps):
    noise = predict(
        latents=latents,
        image_latents=image_latents,
        latent_ids=latent_ids,
        image_latent_ids=image_latent_ids,
        prompt_embeds=prompt_embeds,
        text_ids=text_ids,
        negative_prompt_embeds=neg_embeds,
        negative_text_ids=neg_ids,
        guidance=GUIDANCE,
        timestep=config.scheduler.timesteps[t],
    )
    if v0 is None:
        v0 = noise
    latents = config.scheduler.step(noise=noise, timestep=t, latents=latents, sigmas=config.scheduler.sigmas)
    mx.eval(latents)

packed = latents.reshape(latents.shape[0], lat_h, lat_w, latents.shape[-1]).transpose(0, 3, 1, 2)
decoded = model.vae.decode_packed_latents(packed)
mx.eval(decoded)

out = {
    "ref0_u8": mx.array(refs_u8[0].astype(np.int32)),  # [256,256,3]
    "ref1_u8": mx.array(refs_u8[1].astype(np.int32)),  # [256,256,3]
    "decoded": decoded.astype(mx.float32),  # NCHW [1,3,256,256]
}
if BITS is None:
    out["image_latents"] = image_latents.astype(mx.float32)  # [1, 2·seq_ref, 128]
    out["v0"] = v0.astype(mx.float32)  # [1, seq_tgt, 128]
    out["latents"] = latents.astype(mx.float32)

suffix = f"_q{BITS}" if BITS is not None else ""
path = fixture(f"tools/golden/flux2_edit_multi{suffix}.safetensors")
if BITS is not None:
    mx.save_safetensors(path, out, metadata={"bits": str(BITS)})
else:
    mx.save_safetensors(path, out)
print(f"wrote {path}")
print(f"  image_latents {tuple(image_latents.shape)}  decoded {tuple(decoded.shape)}")
