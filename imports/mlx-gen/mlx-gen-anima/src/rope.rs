//! Cosmos 3-axis rotary position embedding for the DiT self-attention (`CosmosRotaryPosEmbed`,
//! images ⇒ `fps=None`). Builds the `(cos, sin)` tables for a `(pe_t, pe_h, pe_w)` post-patch latent
//! grid over `head_dim` channels, split `dim_t | dim_h | dim_w` and duplicated (`[t,h,w]·2`) so the
//! half-split rotate (`apply_text_rope`) rotates matching frequency pairs.
//!
//! **Per-axis OOD reject (VERIFIED trap, same class as the PiD sincos-pos-embed bug).** The reference
//! indexes `seq = arange(max(max_size))` per spatial axis; a request whose post-patch extent exceeds
//! `max_size` (`(128,120,120)` = `max_size/patch`, ~1920 px/side spatially) would index positions the
//! model never trained on (and truncate the `seq` slice → shape corruption). We **error** here rather
//! than silently emit garbage (or clamp/wrap the index) — `1536²` (post-patch 96) fits, `>1920 px/side`
//! is rejected. (Unreachable via the normal path — `RES_MAX=1536` caps the post-patch extent at 96 < 120
//! — but kept as defense against a future max-size change.)

use mlx_rs::Array;

use mlx_gen::{Error, Result};

use crate::config::DitConfig;

/// Precomputed RoPE tables for a specific latent grid: `cos`/`sin` are `[1, seq_len, head_dim]`,
/// ready for `mlx_gen::nn::apply_text_rope` (broadcast over heads).
pub struct CosmosRope {
    pub cos: Array,
    pub sin: Array,
}

/// Build the Cosmos image RoPE for a `(pe_t, pe_h, pe_w)` post-patch grid. `pe_*` are the latent
/// extents AFTER patchify (`num_frames/p_t`, `H_latent/p_h`, `W_latent/p_w`). Errors (never indexes
/// OOB) if any axis exceeds its `max_size`.
pub fn cosmos_image_rope(
    cfg: &DitConfig,
    pe_t: usize,
    pe_h: usize,
    pe_w: usize,
) -> Result<CosmosRope> {
    let (max_t, max_h, max_w) = cfg.max_size;
    if pe_t > max_t || pe_h > max_h || pe_w > max_w {
        return Err(Error::Msg(format!(
            "anima: latent grid (t={pe_t}, h={pe_h}, w={pe_w}) exceeds Cosmos RoPE max_size \
             (t={max_t}, h={max_h}, w={max_w}); reduce the requested size (post-patch h/w must be \
             <= {max_h}, ~{}px/side)",
            max_h * cfg.patch_size.1 * crate::config::VAE_COMPRESSION as usize
        )));
    }

    let head_dim = cfg.attention_head_dim;
    // dim_h = dim_w = head_dim//6*2; dim_t takes the remainder. Half-dims sum to head_dim/2.
    let dim_h = head_dim / 6 * 2;
    let dim_w = head_dim / 6 * 2;
    let dim_t = head_dim - dim_h - dim_w;

    // NTK RoPE scaling per axis: theta_axis = 10000 * scale ** (dim/(dim-2)).
    let base = 10000.0f64;
    let ntk = |scale: f32, dim: usize| base * (scale as f64).powf(dim as f64 / (dim as f64 - 2.0));
    let t_theta = ntk(cfg.rope_scale.0, dim_t);
    let h_theta = ntk(cfg.rope_scale.1, dim_h);
    let w_theta = ntk(cfg.rope_scale.2, dim_w);

    // Per-axis frequency vectors: freq[k] = 1 / theta ** (2k/dim), k in 0..dim/2.
    let freqs = |theta: f64, dim: usize| -> Vec<f64> {
        (0..dim / 2)
            .map(|k| 1.0 / theta.powf(2.0 * k as f64 / dim as f64))
            .collect()
    };
    let t_freqs = freqs(t_theta, dim_t); // len dim_t/2
    let h_freqs = freqs(h_theta, dim_h); // len dim_h/2
    let w_freqs = freqs(w_theta, dim_w); // len dim_w/2

    let seq_len = pe_t * pe_h * pe_w;
    let mut cos = vec![0f32; seq_len * head_dim];
    let mut sin = vec![0f32; seq_len * head_dim];

    // Flatten order matches patchify (t slowest, w fastest → index = (t*pe_h + h)*pe_w + w).
    for t in 0..pe_t {
        for h in 0..pe_h {
            for w in 0..pe_w {
                let p = (t * pe_h + h) * pe_w + w;
                // The 128-vector = [emb_t, emb_h, emb_w, emb_t, emb_h, emb_w].
                let row = &mut cos[p * head_dim..(p + 1) * head_dim];
                let srow = &mut sin[p * head_dim..(p + 1) * head_dim];
                let mut fill = |off: &mut usize, angles: &[f64]| {
                    for &a in angles {
                        row[*off] = a.cos() as f32;
                        srow[*off] = a.sin() as f32;
                        *off += 1;
                    }
                };
                let emb_t: Vec<f64> = t_freqs.iter().map(|&f| t as f64 * f).collect();
                let emb_h: Vec<f64> = h_freqs.iter().map(|&f| h as f64 * f).collect();
                let emb_w: Vec<f64> = w_freqs.iter().map(|&f| w as f64 * f).collect();
                let mut off = 0usize;
                fill(&mut off, &emb_t);
                fill(&mut off, &emb_h);
                fill(&mut off, &emb_w);
                fill(&mut off, &emb_t);
                fill(&mut off, &emb_h);
                fill(&mut off, &emb_w);
            }
        }
    }

    Ok(CosmosRope {
        cos: Array::from_slice(&cos, &[1, seq_len as i32, head_dim as i32]),
        sin: Array::from_slice(&sin, &[1, seq_len as i32, head_dim as i32]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_shape_for_1024() {
        // 1024² ⇒ latent 128×128 ⇒ post-patch 64×64 ⇒ seq 4096.
        let cfg = DitConfig::anima();
        let r = cosmos_image_rope(&cfg, 1, 64, 64).unwrap();
        assert_eq!(r.cos.shape(), &[1, 4096, 128]);
        assert_eq!(r.sin.shape(), &[1, 4096, 128]);
    }

    #[test]
    fn rope_first_position_is_ones_and_zeros() {
        // Position (0,0,0): all angles are 0 ⇒ cos=1, sin=0 across the head dim.
        let cfg = DitConfig::anima();
        let r = cosmos_image_rope(&cfg, 1, 2, 2).unwrap();
        let cos = r.cos.as_slice::<f32>();
        let sin = r.sin.as_slice::<f32>();
        for d in 0..128 {
            assert!((cos[d] - 1.0).abs() < 1e-6, "cos[0,{d}]");
            assert!(sin[d].abs() < 1e-6, "sin[0,{d}]");
        }
    }

    #[test]
    fn rope_rejects_out_of_range_per_axis() {
        let cfg = DitConfig::anima(); // max_size (128, 120, 120)
                                      // 121 latent positions/spatial axis = > ~1920px/side. Must reject, never index OOB.
        assert!(cosmos_image_rope(&cfg, 1, 121, 96).is_err(), "h axis OOD");
        assert!(cosmos_image_rope(&cfg, 1, 96, 121).is_err(), "w axis OOD");
        // 1536² ⇒ post-patch 96 ⇒ fits.
        assert!(cosmos_image_rope(&cfg, 1, 96, 96).is_ok());
        // 1920² ⇒ post-patch 120 ⇒ the boundary fits.
        assert!(cosmos_image_rope(&cfg, 1, 120, 120).is_ok());
    }

    #[test]
    fn rope_split_dims_sum_to_head_dim() {
        let head_dim = 128usize;
        let dim_h = head_dim / 6 * 2;
        let dim_w = head_dim / 6 * 2;
        let dim_t = head_dim - dim_h - dim_w;
        assert_eq!(dim_h, 42);
        assert_eq!(dim_w, 42);
        assert_eq!(dim_t, 44);
        // half-dims (the concatenated freq block) sum to head_dim/2 = 64, duplicated to 128.
        assert_eq!(dim_t / 2 + dim_h / 2 + dim_w / 2, 64);
    }
}
