# Independent reference VAE encoding oracle

The prior retained `mlx-gen-moss-image/tests/fixtures/small_vae` fixture covers decoder, post_quant and BN **de-normalization** only. Its generator deliberately skips encoder/quant_conv weights and never calls encode. It therefore does not cover posterior mode, forward packing or BN normalization.

This fixture fills that numerical gap with a CPU-only synthetic model, using a tiny shape agreed with the native implementer before generation: encoder and decoder widths `[32,32,32,32]`, four stages, two encoder resnets per stage, GroupNorm32, latent_channels32, quant_conv and post_quant_conv enabled. Input RGB16x32 yields mean2x4 and packed1x2. All remaining options come from the exact pinned official MOSS SFT VAE config. Production MOSS widths remain encoder `[128,256,512,512]` and decoder `[96,192,384,384]`; the fixture is a constructor/injection numerical test, not full production-shape weight or device proof.

`generate.py` verifies the exact Diffusers0.38.0 wheel SHA256 `18e53f9e539096320470f62c6360a6fd5727ff28cffda566265316e13fcdb612`, extracts it temporarily under this evidence directory without installation, and executes its unchanged `AutoencoderKLFlux2`. It imports the exact unchanged `_vae_placement`, `_bn_stats`, `encode_image` AST definitions from externally frozen Comfy reference `moss_runtime/processing.py`, SHA256 `43e5fd31a40500cb720b573b547e39e9b0a298837b2ac3bd21f7a7d8bd8c521f`. Correct normalized packed expected output comes directly from `encode_image(...,sample_posterior=False)`; the generator's separate explicit packing is only an asserted intermediate and mutation source. No native helper or native output constructs expected values.

The generator runs Torch2.9.1 on CPU, deterministic algorithms, one thread, no bytecode, no installed package changes, no real weights, no image generation and no accelerator initialization; timeout120 seconds. One execution took3.705 seconds.

- `fixture.json`:132703 bytes; SHA256 `e08be4bb50782160a8e43cd74f2376277bc7a46ae4eae1182230e5375a92bd0c`.
- `receipt.json` binds the exact generator, output, external helper, official config and wheel/module source hashes, settings and mutation deltas.
- `LICENSE.diffusers`: exact wheel Apache2.0 license. The linked Comfy source is GPLv3; it is kept external and not transplanted into inference. Fixtures contain synthetic recipes and expected outputs; no trained weights. This says nothing about the independently distributed edit adapter's unspecified weight license.

Tensor objects have `shape`, `dtype`, flattened row-major `values`. Inputs are retained in NCHW and NHWC. `mean_nhwc` is `[1,2,4,32]`; `normalized_packed_nchw` is `[1,128,1,2]`. Also retains raw mean NCHW, posterior logvar, raw packed values, BN mean/std, and mutation outputs.

`weights` has242 shape/seed/kind recipes totaling546147 synthetic elements. `weight_prefix_probes` contains the first four exact FP32 values per tensor to validate cross-language reconstruction before comparing inference. Recipes retain original Diffusers tensor layout/names; native MLX convolution loader should perform its existing weight-layout transformation.

Recipe arithmetic (uint32 wrapping, each flattened index i): `bits=(i+1)^seed`; `bits=(bits^(bits<<13))&0xffffffff`; `bits=bits^(bits>>17)`; `bits=(bits^(bits<<5))&0xffffffff`; `signed=(bits&65535)-32768`. Seeds are low4 SHA256 bytes of exact tensor name, little endian.

| kind | exact FP32 value |
| --- | --- |
| weight | signed/8388608 |
| bias | signed/1048576 |
| norm | 1+signed/262144 |
| bn_mean | ((i%17)-8)/64 |
| bn_var | 1+((i*7)%31)/32 |
| pointwise_identity | signed/8388608 + indicator((i/input_channels)==(i%input_channels)) |

`pointwise_identity` uses the tensor's actual input-channel dimension:64 for quant_conv,32 for post_quant_conv. Normalization is `(packed-mean)/sqrt(var+1e-4)`, channel-major `c,dy,dx` packing, no posterior sampling. Retained wrong outputs differ from expected by max abs2.8314033 (posterior sample),0.1621527 (patch-major packing),0.1249954 (missing BN), all above declared1e-4 absolute bound. Native tests should assert finite full shape/length and complete elementwise errors, and confirm each negative control fails that bound. No native parity pass has been claimed.

Regenerate from this directory's existing frozen surrounding references:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python3 -B generate.py
```

Source authority: [official SFT VAE config](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/fc5de0d56fc910cbca6f6625a70f5f582b26deb1/vae/config.json), [Comfy encode helper](https://github.com/Chenkin-x/ComfyUI-Moss-Image-Edit/blob/f70d118a814e4a7e9b834597408af2f470eddd6c/moss_runtime/processing.py), [frozen Diffusers wheel](https://files.pythonhosted.org/packages/42/c0/3237566ea6e3a542f3c0669a253d62fe75f27b84b3d7bd4fb3b5ee89d73c/diffusers-0.38.0-py3-none-any.whl).
