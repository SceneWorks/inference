//! Lens VAE decode (sc-3169) — a thin shim over the **already-ported** Flux.2 `AutoencoderKLFlux2`
//! ([`mlx_gen_flux2::Flux2Vae`]). The Lens latent space is the Flux.2 one (32-ch latent, 2×2 patchify
//! into the 128-ch transformer space, BatchNorm-stats normalization), so the entire `LensPipeline._decode`
//! reduces to: reshape the DiT output into the packed grid and call the shared decode.
//!
//! ## Why the reshape is the whole shim
//! The reference `_decode` does `rearrange(b (h w) (c p1 p2) -> b c (h p1) (w p2))` then
//! `_patchify_latents` (re-pack 2×2) → bn de-normalize → `_unpatchify_latents` → `vae.decode`. The
//! rearrange-then-patchify pair is an **identity** that collapses to a plain reshape from
//! `[B, h·w, 128]` to the packed grid `[B, h, w, 128]` (the DiT's 128 channels already carry the
//! `c·4 + p1·2 + p2` packing, which is exactly [`Flux2Vae::decode_packed_latents`]'s expected channel
//! order). De-normalize + unpatchify + decode are then the shared Flux.2 path verbatim
//! (`x·std + mean` ≡ the reference's `x/scale − shift` with `scale = 1/std`, `shift = −mean`).

use mlx_rs::Array;

use mlx_gen::Result;
use mlx_gen_flux2::Flux2Vae;

/// Decode the Lens DiT output into an image. `dit_out`: `[B, h·w, 128]` (the transformer's packed
/// patch-space velocity at the final step); `(latent_h, latent_w)` is the latent grid
/// (`= height/16, width/16`). Returns `[B, H, W, 3]` (NHWC) in ~`[−1, 1]`, where `H = latent_h·16`,
/// `W = latent_w·16`.
pub fn decode(vae: &Flux2Vae, dit_out: &Array, latent_h: usize, latent_w: usize) -> Result<Array> {
    let b = dit_out.shape()[0];
    let c = dit_out.shape()[2]; // 128 packed channels
    let packed = dit_out.reshape(&[b, latent_h as i32, latent_w as i32, c])?;
    vae.decode_packed_latents(&packed)
}
