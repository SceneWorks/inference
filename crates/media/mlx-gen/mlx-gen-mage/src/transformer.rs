//! The `MageFlow` NR-MMDiT denoiser — **owned by sc-14040**.
//!
//! Port of `_vendor/mage_flow/models/mage_flow.py:58-153`. Assembly, in order:
//!
//! ```text
//! ms_pe   = MageFlowEmbedRope(img_shapes)              // crate::rope_embedder
//! img     = Linear(in_channels → hidden_size)(img)     // img_in — NO latent scale/shift
//! txt     = RMSNorm(context_in_dim, eps=1e-6)(txt)     // txt_norm
//! temb    = MageFlowTimestepProjEmbeddings(sigma)      // crate::timestep_embedder
//! txt     = Linear(context_in_dim → hidden_size)(txt)  // txt_in
//! temb    = temb + 0                                   // pooled text vec is zeroed (:116)
//! for block in transformer_blocks: (txt, img) = block(...)   // crate::transformer_block
//! img     = AdaLayerNormContinuous(img, temb); img = proj_out(img)   // crate::final_layer
//! ```
//!
//! Shapes come from [`crate::config::MageFlowConfig`], which reads **only** the nine
//! `transformer/config.json` fields the reference reads and pins the rest as constants — see that
//! module's "config-strip trap" note before adding a field here.
//!
//! The output is the flow-matching **velocity**; the sampler ([`crate::pipeline`], sc-14041)
//! applies `x += (σ_next − σ_cur) · v`.
//!
//! ## Precision
//!
//! The reference casts the whole transformer to bf16 after load (`pipeline.py:753`) and runs the
//! denoise loop with **no autocast** — so bf16 weights *and* bf16 activations are the trained
//! configuration, and [`MageTransformer::load`] keeps the checkpoint dtype rather than upcasting.
//! The three places the reference deliberately leaves bf16 are reproduced where they live: the
//! msrope rotation runs in f32 and casts back (`mage_layers.py:18-21`), the sinusoidal timestep
//! table is f32 with a **bf16-rounded** frequency vector (`:42-46`), and the norms accumulate in
//! f32 internally. [`MageTransformer::cast_weights`] exists for the parity suite, which uses an
//! f32 run as a control on how much of the residual gap is bf16 rounding.
//!
//! ## Weight loading
//!
//! There is no shared `loader.rs` (see the decision note in `lib.rs`): loading belongs in **this
//! file**, beside the module it constructs, so the concurrent text-encoder and VAE ports never
//! touch a file this story owns. The transformer is a single
//! `transformer/diffusion_pytorch_model.safetensors`, bf16, loaded non-strict (`pipeline.py:748`)
//! — [`MageTransformer::from_weights`] therefore ignores unexpected keys but still **requires**
//! every key it consumes, so a partially-remapped checkpoint fails loudly instead of silently
//! running with randomly-initialised layers the way `strict=False` does in torch.

use std::path::Path;

use mlx_rs::fast::rms_norm;
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::attention::DualStream;
use crate::config::{MageFlowConfig, NORM_EPS};
use crate::feed_forward::FfnActivation;
use crate::final_layer::MageFinalLayer;
use crate::rope_embedder::{MsRope, PackContext, PackLayout};
use crate::timestep_embedder::MageTimestepEmbedder;
use crate::transformer_block::MageTransformerBlock;

/// The safetensors file the reference loads the DiT from (`pipeline.py:748`).
pub const TRANSFORMER_WEIGHTS_FILE: &str = "diffusion_pytorch_model.safetensors";

/// The config the reference reads its nine consumed fields out of (`pipeline.py:724`).
pub const TRANSFORMER_CONFIG_FILE: &str = "config.json";

/// A PyTorch `nn.Linear` — a `[out, in]` weight plus a bias.
///
/// Every Linear in this DiT carries a bias, so the bias is not optional: `qkv_bias: true`, the
/// diffusers defaults `added_proj_bias`/`out_bias`/`sample_proj_bias`/`bias` are all `True`, and
/// `proj_out` passes `bias=True` explicitly (`mage_flow.py:91`). A checkpoint missing one is a
/// load error, not a silently-zeroed bias.
///
/// This lives in `transformer.rs` rather than a shared `loader.rs` because this file owns weight
/// loading for the DiT (see the module note above and the `lib.rs` decision record).
#[derive(Clone)]
pub struct Linear {
    inner: mlx_gen::adapters::AdaptableLinear,
    in_features: i32,
    out_features: i32,
    dtype: Dtype,
}

impl std::fmt::Debug for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Linear")
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("dtype", &self.dtype)
            .field("quantized", &self.inner.is_quantized())
            .finish()
    }
}

impl Linear {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let weight = w.require(&format!("{prefix}.weight"))?.clone();
        let bias = w.require(&format!("{prefix}.bias"))?.clone();
        let shape = weight.shape();
        Ok(Self {
            in_features: shape[1],
            out_features: shape[0],
            dtype: weight.dtype(),
            inner: mlx_gen::adapters::AdaptableLinear::dense(weight, Some(bias)),
        })
    }

    pub fn out_features(&self) -> i32 {
        self.out_features
    }

    pub fn in_features(&self) -> i32 {
        self.in_features
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        self.inner.cast_weights(dtype)?;
        self.dtype = dtype;
        Ok(())
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.inner.quantize(bits, None)
    }

    pub(crate) fn is_quantized(&self) -> bool {
        self.inner.is_quantized()
    }

    /// `y = x · Wᵀ + b`, as the fused `addmm` MLX's own `nn.Linear` uses — a separate
    /// `matmul` + `add` double-rounds the bias in bf16, which compounds over 12 blocks.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.inner.forward(x)
    }
}

/// The 4B NR-MMDiT: `img_in` / `txt_norm` / `txt_in` / timestep embedder → `depth` dual-stream
/// blocks → `norm_out` / `proj_out`.
#[derive(Clone)]
pub struct MageTransformer {
    cfg: MageFlowConfig,
    rope: MsRope,
    img_in: Linear,
    txt_norm: Array,
    txt_in: Linear,
    time_embed: MageTimestepEmbedder,
    blocks: Vec<MageTransformerBlock>,
    final_layer: MageFinalLayer,
}

impl MageTransformer {
    /// Quantize every DiT Linear to MLX group-wise Q4/Q8; norms and RoPE stay dense.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.img_in.quantize(bits)?;
        self.txt_in.quantize(bits)?;
        self.time_embed.quantize(bits)?;
        for block in &mut self.blocks {
            block.quantize(bits)?;
        }
        self.final_layer.quantize(bits)?;
        let got = self.quantized_linear_count();
        let expected = 2 + 2 + self.blocks.len() * 14 + 2;
        if got != expected {
            return Err(Error::Msg(format!(
                "mage_flow: DiT quantization packed {got}/{expected} required Linear projections"
            )));
        }
        Ok(())
    }

    pub fn quantized_linear_count(&self) -> usize {
        usize::from(self.img_in.is_quantized())
            + usize::from(self.txt_in.is_quantized())
            + self.time_embed.quantized_linear_count()
            + self
                .blocks
                .iter()
                .map(MageTransformerBlock::quantized_linear_count)
                .sum::<usize>()
            + self.final_layer.quantized_linear_count()
    }

    /// Load from a caller-provisioned `transformer/` directory: `config.json` for the nine consumed
    /// fields, `diffusion_pytorch_model.safetensors` for the weights.
    ///
    /// The path is passed in, never derived — this repository's models arrive as
    /// caller-provisioned local paths (the epic-13657 boundary enforced by `check-workspace.py`).
    pub fn load(transformer_dir: impl AsRef<Path>) -> Result<Self> {
        let dir = transformer_dir.as_ref();
        let json = std::fs::read_to_string(dir.join(TRANSFORMER_CONFIG_FILE))?;
        let cfg = MageFlowConfig::from_transformer_config_json(&json)?;
        let mut weights = Weights::from_file(dir.join(TRANSFORMER_WEIGHTS_FILE))?;
        Self::from_weights_draining(&mut weights, cfg)
    }

    /// Build from an already-loaded checkpoint. Keys follow the published diffusers layout
    /// (`img_in`, `txt_norm`, `txt_in`, `time_text_embed.timestep_embedder.linear_{1,2}`,
    /// `transformer_blocks.{i}.*`, `norm_out.linear`, `proj_out`) — no remapping is needed.
    pub fn from_weights(w: &Weights, cfg: MageFlowConfig) -> Result<Self> {
        cfg.validate()?;
        let rope = MsRope::from_config(&cfg)?;
        let img_in = Linear::from_weights(w, "img_in")?;
        let txt_in = Linear::from_weights(w, "txt_in")?;
        let time_embed = MageTimestepEmbedder::from_weights(w, "time_text_embed")?;
        let blocks = (0..cfg.depth)
            .map(|i| {
                MageTransformerBlock::from_weights(w, &format!("transformer_blocks.{i}"), &cfg)
            })
            .collect::<Result<Vec<_>>>()?;
        let model = Self {
            rope,
            img_in,
            txt_norm: w.require("txt_norm.weight")?.clone(),
            txt_in,
            time_embed,
            blocks,
            final_layer: MageFinalLayer::from_weights(w, "norm_out", "proj_out")?,
            cfg,
        };
        model.check_geometry()?;
        Ok(model)
    }

    /// Production loader that releases each source block from the checkpoint map as soon as the
    /// corresponding module owns it. `Array::clone` is a shallow MLX handle; removing the map's
    /// handle therefore avoids retaining a second full-checkpoint reference while later Q4/Q8
    /// packing materializes replacement buffers.
    pub fn from_weights_draining(w: &mut Weights, cfg: MageFlowConfig) -> Result<Self> {
        cfg.validate()?;
        let rope = MsRope::from_config(&cfg)?;
        let img_in = Linear::from_weights(w, "img_in")?;
        w.remove_prefix("img_in.");
        let txt_norm = w.require("txt_norm.weight")?.clone();
        w.remove("txt_norm.weight");
        let txt_in = Linear::from_weights(w, "txt_in")?;
        w.remove_prefix("txt_in.");
        let time_embed = MageTimestepEmbedder::from_weights(w, "time_text_embed")?;
        w.remove_prefix("time_text_embed.");
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let prefix = format!("transformer_blocks.{i}");
            blocks.push(MageTransformerBlock::from_weights(w, &prefix, &cfg)?);
            w.remove_prefix(&format!("{prefix}."));
        }
        let final_layer = MageFinalLayer::from_weights(w, "norm_out", "proj_out")?;
        w.remove_prefix("norm_out.");
        w.remove_prefix("proj_out.");
        let model = Self {
            cfg,
            rope,
            img_in,
            txt_norm,
            txt_in,
            time_embed,
            blocks,
            final_layer,
        };
        model.check_geometry()?;
        if !w.is_empty() {
            return Err(Error::Msg(format!(
                "mage_flow: transformer drain left {} source tensors resident",
                w.len()
            )));
        }
        Ok(model)
    }

    /// The loaded tensors must actually have the geometry the config claims. `strict=False` in the
    /// reference means a renamed or reshaped tensor is silently skipped; here the shapes are
    /// cross-checked so a mismatched checkpoint is a load error rather than a wrong image.
    fn check_geometry(&self) -> Result<()> {
        let expect = |what: &str, got: (i32, i32), want: (i32, i32)| -> Result<()> {
            if got != want {
                return Err(Error::Msg(format!(
                    "mage_flow: {what} is {got:?} but the config implies {want:?}"
                )));
            }
            Ok(())
        };
        let dim = self.cfg.hidden_size;
        expect(
            "img_in",
            (self.img_in.out_features(), self.img_in.in_features()),
            (dim, self.cfg.in_channels),
        )?;
        expect(
            "txt_in",
            (self.txt_in.out_features(), self.txt_in.in_features()),
            (dim, self.cfg.context_in_dim),
        )?;
        if self.txt_norm.shape() != [self.cfg.context_in_dim] {
            return Err(Error::Msg(format!(
                "mage_flow: txt_norm is {:?} but context_in_dim is {}",
                self.txt_norm.shape(),
                self.cfg.context_in_dim
            )));
        }
        expect(
            "time_text_embed",
            (self.time_embed.out_dim(), dim),
            (dim, dim),
        )?;
        let out_channels = self.cfg.patch_size * self.cfg.patch_size * self.cfg.out_channels;
        expect(
            "proj_out",
            (self.final_layer.out_channels(), dim),
            (out_channels, dim),
        )?;
        Ok(())
    }

    pub fn config(&self) -> &MageFlowConfig {
        &self.cfg
    }

    pub fn rope(&self) -> &MsRope {
        &self.rope
    }

    pub fn blocks(&self) -> &[MageTransformerBlock] {
        &self.blocks
    }

    pub fn timestep_embedder(&self) -> &MageTimestepEmbedder {
        &self.time_embed
    }

    pub fn final_layer(&self) -> &MageFinalLayer {
        &self.final_layer
    }

    /// The dtype the weights are held at — bf16 straight off the published checkpoint.
    pub fn dtype(&self) -> Dtype {
        self.img_in.dtype()
    }

    /// Swap every block's FFN activation — **a parity-suite divergence knob**; see
    /// [`crate::feed_forward::MageFeedForward::set_activation`]. Production never calls it.
    pub fn set_ffn_activation(&mut self, activation: FfnActivation) {
        for block in &mut self.blocks {
            block.set_ffn_activation(activation);
        }
    }

    /// Cast every weight (see the module's precision note). Production keeps the checkpoint's bf16.
    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        self.img_in.cast_weights(dtype)?;
        self.txt_in.cast_weights(dtype)?;
        if self.txt_norm.dtype() != dtype {
            self.txt_norm = self.txt_norm.as_dtype(dtype)?;
        }
        self.time_embed.cast_weights(dtype)?;
        for block in &mut self.blocks {
            block.cast_weights(dtype)?;
        }
        self.final_layer.cast_weights(dtype)
    }

    /// Build the per-forward pack context (msrope table + segment indices) for a layout.
    pub fn pack_context(&self, layout: PackLayout) -> Result<PackContext> {
        PackContext::new(layout, &self.rope)
    }

    /// One denoiser evaluation.
    ///
    /// - `img`: `[1, img_tokens, in_channels]` — raw VAE latents, **no scale or shift**
    ///   ([`crate::config::LATENT_SCALE_SHIFT`] is `None`).
    /// - `txt`: `[1, txt_tokens, context_in_dim]` — the final post-RMSNorm Qwen3-VL hidden states
    ///   with the template prefix already dropped.
    /// - `sigma`: `[segments]` — the scheduler sigma in `[0, 1]`, one entry per packed segment.
    /// - returns `[1, img_tokens, out_channels]`: the flow-matching velocity.
    pub fn forward(
        &self,
        img: &Array,
        txt: &Array,
        sigma: &Array,
        ctx: &PackContext,
    ) -> Result<Array> {
        let (mut stream, temb) = self.embed(img, txt, sigma, ctx)?;
        for block in &self.blocks {
            stream = block.forward(&stream, &temb, ctx)?;
            // Bound the lazy graph: without this MLX holds all 12 blocks' intermediates until the
            // first read, which is the peak-memory shape a 2048² pack cannot afford.
            mlx_rs::transforms::eval([&stream.img, &stream.txt])?;
        }
        // Only the image stream reaches the head; the text stream is dropped (`mage_flow.py:146`).
        self.final_layer.forward(&stream.img, &temb, ctx)
    }

    /// The pre-block half of [`MageTransformer::forward`]: `img_in`, `txt_norm → txt_in`, and the
    /// timestep conditioning (`mage_flow.py:107-118`), returning the two streams a block consumes
    /// plus `temb`.
    ///
    /// Split out because it is exactly what the one-block golden captures (`block_in.*`), so the
    /// parity suite can gate the input projections and the timestep embedder on their own instead
    /// of only seeing them through 12 blocks of compounding.
    pub fn embed(
        &self,
        img: &Array,
        txt: &Array,
        sigma: &Array,
        ctx: &PackContext,
    ) -> Result<(DualStream, Array)> {
        let tokens = ctx.layout().img_tokens();
        let txt_tokens = ctx.layout().txt_tokens();
        if img.shape() != [1, tokens, self.cfg.in_channels] {
            return Err(Error::Msg(format!(
                "mage_flow: img must be [1, {tokens}, {}], got {:?}",
                self.cfg.in_channels,
                img.shape()
            )));
        }
        if txt.shape() != [1, txt_tokens, self.cfg.context_in_dim] {
            return Err(Error::Msg(format!(
                "mage_flow: txt must be [1, {txt_tokens}, {}], got {:?}",
                self.cfg.context_in_dim,
                txt.shape()
            )));
        }
        if sigma.ndim() != 1 || sigma.shape()[0] != ctx.segments() as i32 {
            return Err(Error::Msg(format!(
                "mage_flow: sigma must be [{}] (one per packed segment), got {:?}",
                ctx.segments(),
                sigma.shape()
            )));
        }

        let dtype = self.dtype();
        let img = self.img_in.forward(&img.as_dtype(dtype)?)?;
        let txt = rms_norm(&txt.as_dtype(dtype)?, &self.txt_norm, NORM_EPS)?;
        // `timesteps.to(img.dtype)` (`mage_flow.py:112`) — sigma rounds to the model dtype BEFORE
        // the sinusoid, which is what the embedder's f32 arithmetic then consumes.
        let temb = self.time_embed.forward(&sigma.as_dtype(dtype)?)?;
        let txt = self.txt_in.forward(&txt)?;
        // `temb = temb + txt_vec` where `txt_vec` is an all-zero `[B, dim]` of the same dtype
        // (`mage_flow.py:116-118`) — an exact no-op, so the pooled text vector is not ported at all.
        Ok((DualStream { txt, img }, temb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_requires_its_bias() {
        let mut w = Weights::empty();
        w.insert("p.weight", Array::from_slice(&[1.0f32, 2.0], &[1, 2]));
        assert!(Linear::from_weights(&w, "p").is_err());
        w.insert("p.bias", Array::from_slice(&[0.5f32], &[1]));
        let lin = Linear::from_weights(&w, "p").unwrap();
        assert_eq!(lin.in_features(), 2);
        assert_eq!(lin.out_features(), 1);
        let x = Array::from_slice(&[1.0f32, 1.0], &[1, 2]);
        assert_eq!(lin.forward(&x).unwrap().as_slice::<f32>(), &[3.5]);
    }

    #[test]
    fn missing_transformer_keys_are_a_load_error_not_a_silent_skip() {
        // torch's `strict=False` would leave these randomly initialised; the port refuses.
        let err =
            match MageTransformer::from_weights(&Weights::empty(), MageFlowConfig::mage_flow()) {
                Ok(_) => panic!("empty weights unexpectedly loaded"),
                Err(err) => err,
            };
        assert!(
            matches!(err, Error::MissingTensor(ref key) if key.starts_with("img_in")),
            "expected a missing-tensor error naming img_in, got {err:?}"
        );
    }
}
