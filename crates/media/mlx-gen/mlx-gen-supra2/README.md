# Native Supra2-IMG on MLX (sc-24213)

Engine id **`supra2_img`** (family `supra2`, backend `mlx`), registered in `mlx-gen-catalog`.
256×256 text-to-image from `SupraLabs/Supra2-IMG` @ `10dec6e4…`: a 104.1M-parameter
rectified-flow DiT, `google/flan-t5-base` @ `7bcac572…` (128-token context, attention mask)
and `stabilityai/sd-vae-ft-mse` @ `31f26fde…` (latent / 0.18215). Everything runs in f32,
resident; quantization, adapters, PiD, staging and deferred loading are refused.

## Install-time conversion (backend-neutral)

`gen_core_checkpoint::supra2::convert_pinned_checkpoint(model_final_ema.pt, out_dir)` verifies the pinned
checkpoint's SHA-256, validates the exact 237-tensor state dict (upstream's `ema`-else-`model`
selection, `patch`/`ctx_len` config, finite values) and the stored unconditional embedding, and
writes the bundle:

```
<bundle>/
  denoiser.safetensors      F32, upstream state_dict keys verbatim (byte-identical values)
  conditioning.safetensors  uncond_text F32 [128, 768] + uncond_mask F32 [128]   (only if stored)
  supra2_bundle.json        schema, source identity, geometry, file sizes + SHA-256, component pins
```

`FLAN_T5_BASE.verify(dir)` / `SD_VAE.verify(dir)` check the component snapshots at install
time (sizes + SHA-256). The same CLI: `cargo run --release -p sceneworks-gen-core-checkpoint
--example supra2_convert -- <pt> <out> [<t5 dir> <vae dir>]`. Nothing is published or downloaded.

## Load contract

```rust
LoadSpec::new(WeightsSource::Dir(bundle_dir))
    .with_component("text_encoder", WeightsSource::Dir(flan_t5_base_snapshot))
    .with_component("vae", WeightsSource::Dir(sd_vae_ft_mse_snapshot))
```

Load re-verifies the bundle manifest digests and refuses any bundle not converted from the pinned
checkpoint with the published geometry and component pins, or whose converted denoiser and stored
unconditional conditioning are not the pinned conversion (`read_pinned_bundle`), then checks the
pinned component file sizes and their `config.json` geometry. Footprint/admission queries read the
same manifest with a sizes-only check (no re-hash). Requests: prompt, `seed` (default 0; image *i* uses `seed + i` through the
backend-neutral `initial_noise`), `guidance` (default 3.0; CFG only when `> 1`), `steps` (default
50, uniform Euler flow), `count` (1–64 individual images). Negative prompts, prompt batches,
conditioning and other sizes are refused. Prompts longer than 128 tokens are truncated exactly as
the HF tokenizer does (`</s>` kept); `generate_with_report` then carries a `prompt_truncated`
warning (`prompt_tokens=… ctx_len=…`), and the descriptor advertises the limit as
`Capabilities::max_prompt_tokens` (with the 256×256 geometry as `min_size`/`max_size`).

## Tests

* `tiny_dit_parity` (default, MLX CPU device): random-weight upstream `SupraDiT` fixture, ≤ 1e-5 rel.
* `tiny_generator` (default, MLX CPU device): the `Generator` surface on a tiny random-weight
  pipeline — the truncation warning and cancellation before and during the denoise.
* `real_weights::*` (`#[ignore]`d; env vars in the file header): Flan-T5, stored conditioning, DiT,
  VAE and four whole-pipeline trajectories vs the committed torch CPU golden shared with the
  Candle twin (`tools/dump_supra2_reference.py`), plus the registry-path render and cancellation.
  Dispatched on Apple Silicon by the `mlx-supra2` job of `real-weights.yml` (`supra2` profile). `SUPRA2_TEST_DEVICE=cpu` runs
  the exact-f32 structural gate; the default Metal run carries Metal's reduced-precision f32 GEMM
  floor in its declared bounds.
