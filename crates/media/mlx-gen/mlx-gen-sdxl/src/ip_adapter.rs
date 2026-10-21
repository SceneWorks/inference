//! IP-Adapter (image-prompt conditioning) for SDXL — spike (sc-3056).
//!
//! Two pieces live here:
//! 1. [`Resampler`] — the IP-Adapter "plus" image projection (`image_proj.*` in
//!    `ip-adapter-plus_sdxl_vit-h.safetensors`). A perceiver/Resampler that maps the ViT-H
//!    penultimate hidden states `[B, 257, 1280]` → **16 image tokens × 2048** (the UNet
//!    cross-attention width). This is the *original Tencent* `Resampler`/`PerceiverAttention`
//!    layout (fused `to_kv`, `norm1`/`norm2`, bias-free projections), NOT the diffusers refactor.
//! 2. (next) the **decoupled cross-attention** primitive (`to_k_ip`/`to_v_ip` + `ip_adapter_scale`)
//!    injected into the SDXL UNet cross-attn — built generic so the token source is pluggable
//!    (ViT-H→Resampler here; an ArcFace Resampler for InstantID, sc-3113).

pub(crate) mod pricing;

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, broadcast_to, concatenate_axis, split};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::image::resize_bicubic_u8;
use mlx_gen::media::Image;
use mlx_gen::nn::gelu_exact;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::vision_encoder::ClipVisionEncoder;

/// CLIP ViT image preprocessing for IP-Adapter (`CLIPImageProcessor`): resize the shortest side to
/// 224 (bicubic), center-crop 224×224, rescale `[0,255]→[0,1]`, normalize by the CLIP mean/std.
/// Returns NHWC `[1, 224, 224, 3]` f32. (ViT-L-336 towers — e.g. Kolors IP-Adapter — use
/// [`preprocess_clip_image_sized`] with 336.)
pub fn preprocess_clip_image(image: &Image) -> Result<Array> {
    preprocess_clip_image_sized(image, 224)
}

/// [`preprocess_clip_image`] parametrized by the CLIP crop size (224 for ViT-H/ViT-L-224, **336** for
/// the Kolors IP-Adapter's ViT-L/14-336 tower). Same `CLIPImageProcessor` recipe: shortest-side
/// bicubic resize to `size`, center-crop `size`×`size`, `[0,255]→[0,1]`, CLIP mean/std normalize.
#[allow(clippy::excessive_precision)] // canonical CLIP mean/std (f32 rounds the last digit)
pub fn preprocess_clip_image_sized(image: &Image, size: usize) -> Result<Array> {
    const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
    const STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];
    let geometry = preprocess_geometry(image.width, image.height, image.pixels.len() as u64, size)?;
    let (iw, ih) = (geometry.source_width, geometry.source_height);
    let (rw, rh) = (geometry.resized_width, geometry.resized_height);
    let resized = resize_bicubic_u8(&image.pixels, ih, iw, rh, rw)?; // HWC f32 [0,255]
                                                                     // Center-crop size×size + normalize.
    let top = (rh - size) / 2;
    let left = (rw - size) / 2;
    let mut out = vec![0f32; size * size * 3];
    for y in 0..size {
        for x in 0..size {
            for c in 0..3 {
                let v = resized[((top + y) * rw + (left + x)) * 3 + c] / 255.0;
                out[(y * size + x) * 3 + c] = (v - MEAN[c]) / STD[c];
            }
        }
    }
    Ok(Array::from_slice(&out, &[1, size as i32, size as i32, 3]))
}

/// The IP-Adapter image-token source: the CLIP ViT-H encoder + the Resampler. Produces the 16 image
/// tokens consumed by the UNet's decoupled cross-attention. (Generic seam: InstantID swaps this for
/// an ArcFace Resampler, sc-3113 — the UNet injection primitive is identical.)
pub struct IpImageEncoder {
    encoder: ClipVisionEncoder,
    resampler: Resampler,
    /// The CLIP crop size the encoder was trained at (224 for ViT-H/ViT-L-224, 336 for the Kolors
    /// ViT-L/14-336). Drives [`preprocess_clip_image_sized`].
    image_size: usize,
}

impl IpImageEncoder {
    /// ViT-H / ViT-L-224 tower (224px CLIP preprocess).
    pub fn new(encoder: ClipVisionEncoder, resampler: Resampler) -> Self {
        Self::with_image_size(encoder, resampler, 224)
    }

    /// Like [`new`](Self::new) but with an explicit CLIP crop size (336 for the Kolors ViT-L/14-336).
    pub fn with_image_size(
        encoder: ClipVisionEncoder,
        resampler: Resampler,
        image_size: usize,
    ) -> Self {
        Self {
            encoder,
            resampler,
            image_size,
        }
    }

    /// Reference image → `[1, num_queries, output_dim]` IP tokens (16×2048 for plus-vit-h), at the
    /// resampler's weight dtype. CLIP preprocess → ViT penultimate → Resampler.
    pub fn tokens(&self, image: &Image) -> Result<Array> {
        let dtype = self.resampler.dtype();
        let pixels = preprocess_clip_image_sized(image, self.image_size)?.as_dtype(dtype)?;
        let penultimate = self.encoder.penultimate(&pixels)?;
        self.resampler.forward(&penultimate)
    }

    /// Header-only pricing of [`Self::tokens`]: every quantum the image encode checks (the vision
    /// tower, then the Resampler), in order, with the guards' own formulas. Returns the shape-only
    /// `[1, num_queries, output_dim]` tokens.
    #[cfg(test)]
    pub(crate) fn price_tokens(&self, out: &mut crate::quantum::Quanta) -> Result<Array> {
        let size = self.image_size as i32;
        let pixels = crate::quantum::shaped(&[1, size, size, 3], self.resampler.dtype())?;
        let penultimate = self.encoder.price_penultimate(&pixels, out)?;
        self.resampler.price_forward(&penultimate, out)
    }

    /// A zeros token tensor matching [`tokens`](Self::tokens)'s shape/dtype — the CFG uncond row.
    pub fn zeros_like_tokens(&self, dtype: Dtype) -> Result<Array> {
        let n = self.resampler.num_queries;
        let d = self.resampler.output_dim();
        Ok(mlx_rs::ops::zeros::<f32>(&[1, n, d])?.as_dtype(dtype)?)
    }
}

/// The CFG pair of image-prompt `tokens` `[1, T, D]`: the tokens over a zeros uncond row of the
/// same shape and dtype (what [`IpImageEncoder::zeros_like_tokens`] builds), `[2, T, D]`.
pub(crate) fn cfg_pair(tokens: &Array) -> Result<Array> {
    let zeros = mlx_rs::ops::zeros::<f32>(tokens.shape())?.as_dtype(tokens.dtype())?;
    Ok(mlx_rs::ops::concatenate_axis(&[tokens, &zeros], 0)?)
}

/// Load the decoupled cross-attention **K/V projection pairs** from an IP-Adapter checkpoint
/// (`ip_adapter.{n}.to_k_ip/to_v_ip.weight`, bias-free `[hidden, cross_attention_dim]`), returned in
/// the diffusers `ip_adapter.{1,3,…}` **numeric order** — which is the UNet cross-attention walk
/// order ([`crate::unet::UNet2DConditionModel::install_ip_adapter`]). 70 pairs for SDXL.
pub(crate) fn ip_kv_indices<'a>(keys: impl Iterator<Item = &'a str>) -> Result<Vec<u32>> {
    let mut indices: Vec<u32> = keys
        .filter_map(|key| {
            key.strip_prefix("ip_adapter.")
                .and_then(|r| r.strip_suffix(".to_k_ip.weight"))
                .and_then(|n| n.parse::<u32>().ok())
        })
        .collect();
    indices.sort_unstable();
    if indices.is_empty() {
        return Err(Error::Msg(
            "ip_adapter: no ip_adapter.{n}.to_k_ip.weight keys found".into(),
        ));
    }
    Ok(indices)
}

pub fn load_ip_kv_pairs(w: &Weights) -> Result<Vec<(Array, Array)>> {
    ip_kv_indices(w.keys())?
        .into_iter()
        .map(|n| {
            let k = w
                .require(&format!("ip_adapter.{n}.to_k_ip.weight"))?
                .clone();
            let v = w
                .require(&format!("ip_adapter.{n}.to_v_ip.weight"))?
                .clone();
            Ok((k, v))
        })
        .collect()
}

const LN_EPS: f32 = 1e-5;

/// IP-Adapter "plus" Resampler config. Defaults are `ip-adapter-plus_sdxl_vit-h`.
#[derive(Clone, Debug)]
pub struct ResamplerConfig {
    /// Working width (`dim`); also the latent/query width. 1280 for plus-vit-h.
    pub dim: i32,
    /// Number of perceiver blocks (`depth`). 4 for plus-vit-h.
    pub depth: i32,
    /// Attention heads. 20 for plus-vit-h (head_dim 64).
    pub heads: i32,
    pub dim_head: i32,
    /// Output query tokens (`num_queries`). 16 for plus-vit-h.
    pub num_queries: i32,
    /// Input feature width (ViT-H hidden) feeding `proj_in`. 1280.
    pub embed_dim: i32,
    /// Output token width (= UNet cross_attention_dim). 2048 for SDXL.
    pub output_dim: i32,
}

impl ResamplerConfig {
    pub fn plus_sdxl_vit_h() -> Self {
        Self {
            dim: 1280,
            depth: 4,
            heads: 20,
            dim_head: 64,
            num_queries: 16,
            embed_dim: 1280,
            output_dim: 2048,
        }
    }

    /// Kolors IP-Adapter-Plus Resampler (`image_proj.*` of `Kwai-Kolors/Kolors-IP-Adapter-Plus`'s
    /// `ip_adapter_plus_general.safetensors`; sc-3098). Same Tencent layout as
    /// [`plus_sdxl_vit_h`](Self::plus_sdxl_vit_h), but the latent/working width is **2048** (not 1280)
    /// and the input features are the **ViT-L/14-336** penultimate (1024-d, vs ViT-H 1280). Pinned by
    /// the on-disk weight shapes: `latents [1,16,2048]`, `proj_in [2048,1024]`, `proj_out [2048,2048]`,
    /// `to_q [768,2048]` (inner = `heads·dim_head` = 768); `dim_head=64` (universal for IP-Adapter
    /// Resamplers) ⇒ `heads=12`. depth 4, 16 queries, output 2048 (= the U-Net cross-attention dim).
    pub fn kolors_plus() -> Self {
        Self {
            dim: 2048,
            depth: 4,
            heads: 12,
            dim_head: 64,
            num_queries: 16,
            embed_dim: 1024,
            output_dim: 2048,
        }
    }

    /// InstantID's face Resampler (`image_proj.*` of `InstantX/InstantID` `ip-adapter.bin`; sc-3110).
    /// The vendored InstantID `Resampler` (`_vendor/instantid/ip_adapter/resampler.py`) is the *same*
    /// Tencent layout as [`plus_sdxl_vit_h`](Self::plus_sdxl_vit_h); the only delta is the input
    /// feature width — a single 512-d antelopev2 ArcFace embedding (fed `[B, 1, 512]`) instead of the
    /// 257-token ViT-H penultimate. InstantID instantiates it with `apply_pos_emb=False` and
    /// `num_latents_mean_pooled=0`, so the position-embedding and mean-pooled-latent branches are
    /// absent — exactly the [`Resampler`] this module already implements.
    pub fn instantid_face() -> Self {
        Self {
            dim: 1280,
            depth: 4,
            heads: 20,
            dim_head: 64,
            num_queries: 16,
            embed_dim: 512,
            output_dim: 2048,
        }
    }
}

/// PerceiverAttention block (`layers.{i}.0`): cross-attention from the learned `latents` (queries)
/// to `cat([image_features, latents])` (keys/values), with a fused `to_kv` projection.
struct PerceiverAttention {
    norm1_w: Array, // LN on the image features (x)
    norm1_b: Array,
    norm2_w: Array, // LN on the latents
    norm2_b: Array,
    to_q: AdaptableLinear,   // bias-free
    to_kv: AdaptableLinear,  // bias-free, fused [2*inner, dim]
    to_out: AdaptableLinear, // bias-free
    heads: i32,
    dim_head: i32,
    scale: f32,
}

impl PerceiverAttention {
    fn from_weights(w: &Weights, prefix: &str, cfg: &ResamplerConfig) -> Result<Self> {
        let g = |name: &str| w.require(&format!("{prefix}.{name}")).cloned();
        let lin = |name: &str| -> Result<AdaptableLinear> {
            Ok(AdaptableLinear::dense(
                w.require(&format!("{prefix}.{name}"))?.clone(),
                None,
            ))
        };
        Ok(Self {
            norm1_w: g("norm1.weight")?,
            norm1_b: g("norm1.bias")?,
            norm2_w: g("norm2.weight")?,
            norm2_b: g("norm2.bias")?,
            to_q: lin("to_q.weight")?,
            to_kv: lin("to_kv.weight")?,
            to_out: lin("to_out.weight")?,
            heads: cfg.heads,
            dim_head: cfg.dim_head,
            scale: (cfg.dim_head as f32).powf(-0.5),
        })
    }

    /// `x`: image features `[B, Nx, dim]`; `latents`: `[B, Nq, dim]`. Returns `[B, Nq, dim]` (the
    /// to_out projection; the Resampler adds the residual outside).
    fn forward(&self, x: &Array, latents: &Array) -> Result<Array> {
        let x = layer_norm(x, Some(&self.norm1_w), Some(&self.norm1_b), LN_EPS)?;
        let latents = layer_norm(latents, Some(&self.norm2_w), Some(&self.norm2_b), LN_EPS)?;
        let b = latents.shape()[0];
        let nq = latents.shape()[1];

        let q = self.to_q.forward(&latents)?;
        let kv_input = concatenate_axis(&[&x, &latents], 1)?; // [B, Nx+Nq, dim]
        let kv = self.to_kv.forward(&kv_input)?; // [B, S, 2*inner]
        let parts = split(&kv, 2, -1)?;
        let (k, v) = (&parts[0], &parts[1]);

        let to_heads = |a: &Array| -> Result<Array> {
            let n = a.shape()[1];
            Ok(a.reshape(&[b, n, self.heads, self.dim_head])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = to_heads(&q)?;
        let k = to_heads(k)?;
        let v = to_heads(v)?;
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?;
        let o = o
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, nq, self.heads * self.dim_head])?;
        self.to_out.forward(&o)
    }
}

/// FeedForward block (`layers.{i}.1`): LayerNorm → Linear(dim, 4·dim) → GELU → Linear(4·dim, dim),
/// the two Linears bias-free. The Resampler adds the residual outside.
struct ResamplerFeedForward {
    ln_w: Array,
    ln_b: Array,
    fc1: AdaptableLinear,
    fc2: AdaptableLinear,
}

impl ResamplerFeedForward {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            ln_w: w.require(&format!("{prefix}.0.weight"))?.clone(),
            ln_b: w.require(&format!("{prefix}.0.bias"))?.clone(),
            fc1: AdaptableLinear::dense(w.require(&format!("{prefix}.1.weight"))?.clone(), None),
            fc2: AdaptableLinear::dense(w.require(&format!("{prefix}.3.weight"))?.clone(), None),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let y = layer_norm(x, Some(&self.ln_w), Some(&self.ln_b), LN_EPS)?;
        let y = self.fc1.forward(&y)?;
        let y = gelu_exact(&y)?;
        self.fc2.forward(&y)
    }
}

/// The IP-Adapter "plus" image projection (`image_proj.*`): ViT-H penultimate hidden states →
/// `[B, num_queries, output_dim]` image tokens.
pub struct Resampler {
    /// `[1, num_queries, dim]` learned query latents.
    latents: Array,
    proj_in: AdaptableLinear,  // embed_dim → dim (+bias)
    proj_out: AdaptableLinear, // dim → output_dim (+bias)
    norm_out_w: Array,
    norm_out_b: Array,
    layers: Vec<(PerceiverAttention, ResamplerFeedForward)>,
    dim: i32,
    num_queries: i32,
}

impl Resampler {
    /// The compute dtype (the learned latents' dtype).
    pub fn dtype(&self) -> Dtype {
        self.latents.dtype()
    }

    /// The output token width (= UNet `cross_attention_dim`, 2048 for SDXL).
    pub fn output_dim(&self) -> i32 {
        self.norm_out_w.shape()[0]
    }

    /// Load from the `image_proj` namespace of an IP-Adapter-plus checkpoint.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &ResamplerConfig) -> Result<Self> {
        let latents = w.require(&format!("{prefix}.latents"))?.clone();
        let proj_in = AdaptableLinear::dense(
            w.require(&format!("{prefix}.proj_in.weight"))?.clone(),
            Some(w.require(&format!("{prefix}.proj_in.bias"))?.clone()),
        );
        let proj_out = AdaptableLinear::dense(
            w.require(&format!("{prefix}.proj_out.weight"))?.clone(),
            Some(w.require(&format!("{prefix}.proj_out.bias"))?.clone()),
        );
        let layers = (0..cfg.depth)
            .map(|i| -> Result<_> {
                let attn =
                    PerceiverAttention::from_weights(w, &format!("{prefix}.layers.{i}.0"), cfg)?;
                let ff = ResamplerFeedForward::from_weights(w, &format!("{prefix}.layers.{i}.1"))?;
                Ok((attn, ff))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            latents,
            proj_in,
            proj_out,
            norm_out_w: w.require(&format!("{prefix}.norm_out.weight"))?.clone(),
            norm_out_b: w.require(&format!("{prefix}.norm_out.bias"))?.clone(),
            layers,
            dim: cfg.dim,
            num_queries: cfg.num_queries,
        })
    }

    /// `image_features`: ViT-H penultimate `[B, Nx, embed_dim]` → image tokens
    /// `[B, num_queries, output_dim]`.
    pub fn forward(&self, image_features: &Array) -> Result<Array> {
        let b = image_features.shape()[0];
        pricing::validate_latents(self.latents.shape(), self.num_queries, self.dim)?;
        // sc-26270: proj_in, each perceiver layer, and proj_out + norm are guarded iOS quanta
        // (pass-throughs off device), each counted from its own ops: proj_in's output (+ its
        // bias add); each layer per `layer_scratch`; proj_out's output, its bias add and the
        // LayerNorm output.
        let _quantum_scope = mlx_gen::ios_quantum::Scope::try_new()?;
        let nx = image_features.shape()[1];
        let act = crate::quantum::element_bytes(&self.latents);

        let mut latents = broadcast_to(&self.latents, &[b, self.num_queries, self.dim])?;
        let proj_in_scratch =
            pricing::resampler_in_scratch(b, nx, act, crate::quantum::linear_out(&self.proj_in));
        let x =
            mlx_gen::ios_quantum::run(crate::quantum::RESAMPLER_PROJ_IN, proj_in_scratch, || {
                self.proj_in.forward(image_features)
            })?;
        for (attn, ff) in &self.layers {
            latents = mlx_gen::ios_quantum::run(
                crate::quantum::RESAMPLER_LAYER,
                self.layer_scratch(attn, ff, b, nx),
                || Self::layer_body(attn, ff, &x, &latents),
            )?;
        }
        let proj_out_scratch = self.proj_out_scratch(b, latents.dtype());
        mlx_gen::ios_quantum::run(crate::quantum::RESAMPLER_PROJ_OUT, proj_out_scratch, || {
            let out = self.proj_out.forward(&latents)?;
            Ok(layer_norm(
                &out,
                Some(&self.norm_out_w),
                Some(&self.norm_out_b),
                LN_EPS,
            )?)
        })
    }

    /// Header-only pricing of [`Self::forward`] over the shape-only `image_features`: every
    /// quantum it checks, in order, with the guards' own formulas. Returns the shape-only tokens.
    #[cfg(test)]
    pub(crate) fn price_forward(
        &self,
        image_features: &Array,
        out: &mut crate::quantum::Quanta,
    ) -> Result<Array> {
        let b = image_features.shape()[0];
        let nx = image_features.shape()[1];
        let act = crate::quantum::element_bytes(&self.latents);
        let rows = |n: i32, width: u64| {
            mlx_gen::ios_quantum::tensor_bytes(act, &[b as u64, n as u64, width])
        };
        out.push((
            crate::quantum::RESAMPLER_PROJ_IN,
            2 * rows(nx, crate::quantum::linear_out(&self.proj_in)),
        ));
        for (attn, ff) in &self.layers {
            out.push((
                crate::quantum::RESAMPLER_LAYER,
                self.layer_scratch(attn, ff, b, nx),
            ));
        }
        out.push((
            crate::quantum::RESAMPLER_PROJ_OUT,
            self.proj_out_scratch(b, self.latents.dtype()),
        ));
        crate::quantum::shaped(&[b, self.num_queries, self.output_dim()], self.dtype())
    }

    fn proj_out_scratch(&self, b: i32, input: Dtype) -> u64 {
        pricing::resampler_out_scratch(
            b,
            self.num_queries,
            self.dim,
            input,
            &crate::quantum::LinearWork::of(&self.proj_out),
            &crate::quantum::activation_facts(&self.norm_out_w),
            &crate::quantum::activation_facts(&self.norm_out_b),
        )
    }

    /// One perceiver layer: attention over `[x, latents]` + residual, then the FFN + residual.
    fn layer_body(
        attn: &PerceiverAttention,
        ff: &ResamplerFeedForward,
        x: &Array,
        latents: &Array,
    ) -> Result<Array> {
        let latents = add(&attn.forward(x, latents)?, latents)?;
        Ok(add(&ff.forward(&latents)?, &latents)?)
    }

    /// One layer's guarded quantum body and price (sc-26270 proof tests), over projected image
    /// features `x` `[B, nx, dim]` and the `[B, Nq, dim]` latents.
    #[cfg(test)]
    pub(crate) fn layer_quantum_for_test<'a>(
        &'a self,
        index: usize,
        x: &Array,
        latents: &Array,
    ) -> (u64, impl Fn() -> Result<Array> + 'a) {
        let (attn, ff) = &self.layers[index];
        let s = x.shape();
        let scratch = self.layer_scratch(attn, ff, s[0], s[1]);
        let (x, latents) = (x.clone(), latents.clone());
        (scratch, move || Self::layer_body(attn, ff, &x, &latents))
    }

    /// iOS quantum scratch (sc-26270) of one perceiver layer over `nx` image tokens plus the
    /// `Nq` queries, from the loaded widths (`to_q`/`to_kv` rows, the FFN's `fc1` rows), counted as
    /// new buffers under `quantum::conv_sum`'s retention model: key side (`S = nx + Nq` rows) — the
    /// LayerNormed image tokens, the `[x, latents]` concatenation, the `to_kv` output (two inner
    /// widths) and the head-split reshape copies of its two strided halves (+ the unfused
    /// fallback's row-contiguous k/v copies); query side (`Nq` rows) — both LayerNorm outputs, q,
    /// the attention output and its reshape copy, the projection, the FFN's LayerNorm and `fc2`
    /// (+ the fallback's scaled q), `fc1` and its GELU; plus the scores on the running backend.
    /// Each query-side LayerNorm is `quantum::layer_norm_buffers` of its input.
    fn layer_scratch(
        &self,
        attn: &PerceiverAttention,
        ff: &ResamplerFeedForward,
        b: i32,
        nx: i32,
    ) -> u64 {
        pricing::resampler_layer_scratch(
            crate::quantum::element_bytes(&self.latents),
            self.num_queries,
            self.dim,
            crate::quantum::linear_out(&attn.to_q),
            crate::quantum::linear_out(&ff.fc1),
            attn.heads,
            attn.dim_head,
            b,
            nx,
        )
    }
}

/// Actual shortest-side geometry and HOST requested/logical extents. This is not MLX allocation
/// identity, RSS/physical demand or an allocator-capacity certificate.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ClipPreprocessGeometry {
    pub source_width: usize,
    pub source_height: usize,
    pub source_buffer_bytes: u64,
    pub resized_width: usize,
    pub resized_height: usize,
    pub crop_size: usize,
}
pub(crate) fn preprocess_geometry(
    width: u32,
    height: u32,
    buffer_bytes: u64,
    size: usize,
) -> Result<ClipPreprocessGeometry> {
    let (iw, ih) = (width as usize, height as usize);
    if buffer_bytes
        != mlx_gen::gen_core::imageops::checked_image_buffer_len(iw, ih, 3).unwrap_or(usize::MAX)
            as u64
    {
        return Err(Error::Msg(format!(
            "ip-adapter image buffer {} != {iw}x{ih}x3",
            buffer_bytes
        )));
    }
    if iw == 0 || ih == 0 {
        return Err(Error::Msg(format!(
            "ip-adapter reference image has a zero dimension ({iw}x{ih})"
        )));
    }
    // SAME native calculation/rounding/max, not output geometry or a square/crop substitution.
    let scale = size as f64 / iw.min(ih) as f64;
    let rw = ((iw as f64 * scale).round() as usize).max(size);
    let rh = ((ih as f64 * scale).round() as usize).max(size);
    Ok(ClipPreprocessGeometry {
        source_width: iw,
        source_height: ih,
        source_buffer_bytes: buffer_bytes,
        resized_width: rw,
        resized_height: rh,
        crop_size: size,
    })
}
/// Four actual host allocation stages in resize_u8 followed by CLIP normalization. Fixed-size
/// metadata only; no per-output coefficient plan or pixel buffer is built for pricing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClipHostStage {
    HorizontalCoefficients,
    VerticalCoefficients,
    ResizeOutput,
    Normalize,
}

/// Requested/logical buffer lengths, NOT Vec capacity, allocator extent, native data_size or
/// process growth. The borrowed source span and native crop output are separate existing owner
/// terms at the served IP window; no physical transfer certificate is exposed by this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ClipHostRequestedWork {
    stages: [(ClipHostStage, u64); 4],
}
impl ClipHostRequestedWork {
    pub(crate) fn peak_requested(&self) -> u64 {
        self.stages
            .iter()
            .map(|(_, bytes)| *bytes)
            .max()
            .unwrap_or(0)
    }
}

/// Window-length upper from the actual precompute_coeffs producer, for an axis that passed the
/// producer's zero checks. ClipPreprocessGeometry's source edges originate as u32, so they are
/// exactly representable in f64. If in/out>=1, floating division cannot cross ceil(in/out): its
/// relative rounding times out is less than in*2^-52 <1. For upsampling filterscale is exactly1.
/// Thus support <=2*ceil(in/out). The two rounded floor endpoints add at most two integer taps;
/// clipping bounds the result by in. This is a discrete endpoint bound, never a fitted offset.
/// F64 coefficient generation and I64 quantization have separate row-record size terms.
fn clip_axis_coeff_requested(input: usize, output: usize) -> Result<(u64, u64)> {
    let overflow = || Error::Msg("sdxl IP host preprocessing extent overflow".into());
    if input == 0 || output == 0 {
        return Err(Error::Msg("sdxl IP host coefficient axis is zero".into()));
    }
    let input = input as u64;
    let output = output as u64;
    let scale_ceiling = input.div_ceil(output).max(1);
    let taps = scale_ceiling
        .checked_mul(4)
        .and_then(|n| n.checked_add(2))
        .ok_or_else(overflow)?
        .min(input);
    let coefficients = output.checked_mul(taps).ok_or_else(overflow)?;
    let floating = coefficients
        .checked_mul(std::mem::size_of::<f64>() as u64)
        .and_then(|n| {
            output
                .checked_mul(std::mem::size_of::<(usize, Vec<f64>)>() as u64)
                .and_then(|rows| n.checked_add(rows))
        })
        .ok_or_else(overflow)?;
    let fixed = coefficients
        .checked_mul(std::mem::size_of::<i64>() as u64)
        .and_then(|n| {
            output
                .checked_mul(std::mem::size_of::<(usize, Vec<i64>)>() as u64)
                .and_then(|rows| n.checked_add(rows))
        })
        .ok_or_else(overflow)?;
    Ok((floating, fixed))
}

impl ClipPreprocessGeometry {
    pub(crate) fn host_requested_work(&self) -> Result<ClipHostRequestedWork> {
        if self.resized_height == 0 || self.resized_width == 0 {
            return Err(Error::Msg(format!(
                "resize_u8: zero dimension — {}×{} → {}×{} (all edges must be > 0)",
                self.source_width, self.source_height, self.resized_width, self.resized_height
            )));
        }
        let overflow = || Error::Msg("sdxl IP host preprocessing extent overflow".into());
        let product = |values: &[u64]| {
            values
                .iter()
                .try_fold(1u64, |n, &v| n.checked_mul(v))
                .ok_or_else(overflow)
        };
        let sum = |values: &[u64]| {
            values
                .iter()
                .try_fold(0u64, |n, &v| n.checked_add(v))
                .ok_or_else(overflow)
        };
        let (hf, hi) = clip_axis_coeff_requested(self.source_width, self.resized_width)?;
        let (vf, vi) = clip_axis_coeff_requested(self.source_height, self.resized_height)?;
        let horizontal = product(&[self.source_height as u64, self.resized_width as u64, 3, 4])?;
        let resized = product(&[self.resized_height as u64, self.resized_width as u64, 3, 4])?;
        let crop = product(&[self.crop_size as u64, self.crop_size as u64, 3, 4])?;
        Ok(ClipHostRequestedWork {
            stages: [
                // quantize_coeffs borrows the temporary precompute F64 table: BOTH coexist.
                (ClipHostStage::HorizontalCoefficients, sum(&[hf, hi])?),
                // Horizontal fixed coefficients and image stay live during vertical setup.
                (
                    ClipHostStage::VerticalCoefficients,
                    sum(&[hi, horizontal, vf, vi])?,
                ),
                (
                    ClipHostStage::ResizeOutput,
                    sum(&[hi, horizontal, vi, resized])?,
                ),
                // resize_u8's tables/horizontal leave their Rust scopes; resized and crop coexist.
                // This is Rust requested-object lifetime only, not allocator/RSS retirement credit.
                (ClipHostStage::Normalize, sum(&[resized, crop])?),
            ],
        })
    }

    /// Constant-time requested/logical host upper at the existing served IP consumer. There is
    /// no output-sized schedule allocation, tensor factory, device/stream call or physical law.
    pub(crate) fn host_requested_upper(&self) -> Result<u64> {
        Ok(self.host_requested_work()?.peak_requested())
    }
}

#[cfg(test)]
mod clip_host_requested_tests {
    use super::*;

    #[test]
    fn clip_host_work_stages_cover_both_coefficient_types_and_images_host() {
        let geometry = preprocess_geometry(8, 4, 96, 4).unwrap();
        assert_eq!((geometry.resized_width, geometry.resized_height), (8, 4));
        let rf = std::mem::size_of::<(usize, Vec<f64>)>() as u64;
        let ri = std::mem::size_of::<(usize, Vec<i64>)>() as u64;
        // Independent literals: h8outputs*6taps*8=384; v4*4*8=128;
        // horizontal4*8*3*4=384, resized384, crop4*4*3*4=192.
        let work = geometry.host_requested_work().unwrap();
        assert_eq!(
            work.stages,
            [
                (ClipHostStage::HorizontalCoefficients, 768 + 8 * rf + 8 * ri),
                (ClipHostStage::VerticalCoefficients, 1024 + 4 * rf + 12 * ri),
                (ClipHostStage::ResizeOutput, 1280 + 12 * ri),
                (ClipHostStage::Normalize, 576),
            ]
        );
        assert_eq!(
            geometry.host_requested_upper().unwrap(),
            (768 + 8 * rf + 8 * ri)
                .max(1024 + 4 * rf + 12 * ri)
                .max(1280 + 12 * ri)
                .max(576)
        );
        assert_eq!(
            geometry.source_buffer_bytes, 96,
            "actual borrowed source span is not fabricated crop bytes"
        );
    }

    #[test]
    fn clip_host_work_orientation_changes_live_setup_without_aspect_family_split_host() {
        let landscape = preprocess_geometry(8, 4, 96, 4)
            .unwrap()
            .host_requested_work()
            .unwrap();
        let portrait = preprocess_geometry(4, 8, 96, 4)
            .unwrap()
            .host_requested_work()
            .unwrap();
        let rf = std::mem::size_of::<(usize, Vec<f64>)>() as u64;
        let ri = std::mem::size_of::<(usize, Vec<i64>)>() as u64;
        assert_eq!(
            portrait.stages,
            [
                (ClipHostStage::HorizontalCoefficients, 256 + 4 * rf + 4 * ri),
                (ClipHostStage::VerticalCoefficients, 1280 + 8 * rf + 12 * ri),
                (ClipHostStage::ResizeOutput, 1280 + 12 * ri),
                (ClipHostStage::Normalize, 576),
            ]
        );
        assert!(
            portrait.peak_requested() > landscape.peak_requested(),
            "same source span/crop/resize area is not the same ordered coefficient work"
        );
        // No per-aspect measurement/anchor family or hardware probe is used or introduced.
    }

    #[test]
    fn clip_axis_requested_integer_bounds_and_validation_order_host() {
        let rf = std::mem::size_of::<(usize, Vec<f64>)>() as u64;
        let ri = std::mem::size_of::<(usize, Vec<i64>)>() as u64;
        for (input, output, taps) in [
            (8, 8, 6),
            (4, 4, 4),
            (16, 8, 10),
            (20, 18, 10),
            (3, 16, 3),
            (65535, 1, 65535),
        ] {
            let (floating, fixed) = clip_axis_coeff_requested(input, output).unwrap();
            assert_eq!(floating, output as u64 * (taps * 8 + rf));
            assert_eq!(fixed, output as u64 * (taps * 8 + ri));
        }
        assert!(preprocess_geometry(0, 8, 1, 4)
            .unwrap_err()
            .to_string()
            .contains("image buffer"));
        assert!(preprocess_geometry(0, 8, 0, 4)
            .unwrap_err()
            .to_string()
            .contains("zero dimension"));
        assert!(preprocess_geometry(8, 4, 96, 0)
            .unwrap()
            .host_requested_work()
            .unwrap_err()
            .to_string()
            .contains("resize_u8: zero dimension"));
    }
}
