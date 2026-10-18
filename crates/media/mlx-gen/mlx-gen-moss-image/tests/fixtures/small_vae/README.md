# Actual-width MOSS small VAE author oracle

This fixture executes the frozen reference's **Diffusers 0.38.0** `AutoencoderKLFlux2`
on Torch 2.9.1 CPU, with the actual MOSS decoder widths `[96,192,384,384]`, standard
encoder config `[128,256,512,512]`, latent32/packed128, three decoder resnets per stage,
FP32, BN epsilon1e-4 and channel-major 2×2 unpatching. Only synthetic decoder,
post-quant and BN weights are used. No checkpoint tensors, image generation pipeline,
GPU, MLX or native output participates in constructing the reference.

`fixture.json` is 120,382 bytes; SHA256
`89572df808130ece07ce344982aed1c2f3462ebbd64cd5138b49d23852310a0c`.
It contains 142 name/shape/seed recipes (27,947,491 FP32 elements, reconstructed at
runtime), 128 input values, 768 raw expected pixels, and two deliberately wrong author
outputs. The generator implements an exact integer xorshift32/power-of-two recipe;
the native test reconstructs those weights without committing a 112MB payload.
Per-tensor four-element author probes check the cross-language reconstruction on CPU.

The author CPU controls change channel-major patch order to patch-major order and
omit BN de-normalization. Their maximum absolute pixel errors are respectively
0.1167287826538086 and 0.01709931343793869, both well above the 1e-4 absolute comparison
bound. Shape equality, finite values and complete pixel length are checked before
comparison. The native test also refuses loading these weights as a standard large
FLUX.2 decoder. No native numerical pass is claimed until the ignored test runs in
an explicitly admitted Metal window.

Portable regeneration (no package/environment installation; Python dependencies must
already be available):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B tests/fixtures/small_vae/generate.py \
  --diffusers-wheel /explicit/diffusers-0.38.0-py3-none-any.whl \
  --author-pipeline /explicit/frozen_author_pipeline.py
```

Run from this crate, or pass an explicit `--output-dir`. The 120-second-bounded
script verifies both source hashes, extracts the exact wheel to a temporary directory,
imports the unchanged author class plus its wheel-local implementation dependencies,
sets CPU explicitly with two Torch threads, and produces the fixture/license. It
performs no download and no real-weight loading. Python bytecode writes are disabled.

Source authority:

- Frozen reference dependency receipt pins the 0.38.0 wheel at
  `https://files.pythonhosted.org/packages/42/c0/3237566ea6e3a542f3c0669a253d62fe75f27b84b3d7bd4fb3b5ee89d73c/diffusers-0.38.0-py3-none-any.whl`,
  SHA256 `18e53f9e539096320470f62c6360a6fd5727ff28cffda566265316e13fcdb612`.
  The full-wheel hash covers every imported source file; individual key source
  hashes are also retained in the fixture. Copyright Hugging Face, Apache-2.0;
  exact wheel license is included as `LICENSE.diffusers`.
- Frozen Apache MOSS pipeline from SFT revision
  `fc5de0d56fc910cbca6f6625a70f5f582b26deb1`, SHA256
  `1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31`,
  supplies the BN de-normalization/unpatching arithmetic before its raw `.decode`
  result. Existing shared fixture includes that source and its Apache license.
- The retained pixels precede author output clamp/postprocessing, matching the
  native `decode_packed_latents` output boundary. This tests numerical decoding,
  not comparative model image quality or performance.

After hardware admission, the exact ignored integration test is
`small_vae::actual_moss_small_decoder_matches_frozen_author_pixels_on_metal` with
`MOSS_NATIVE_TEST_ADMITTED=1`. Every other test in this module is tensor-free.

`config.json` is the exact frozen SFT VAE configuration; SHA256
`cc77f3f0ce0357039c71018dc19edaa4fcceb0c2f7488fdc2b9c0580d6baafa1`.
The generator verifies this input before author construction.
