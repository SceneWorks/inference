//! Cosmos-Predict2 DiT — the Anima image transformer, transcribed from diffusers
//! `transformer_cosmos.py::CosmosTransformer3DModel` (+ the `Cosmos-2.0-Diffusion-2B-Text2Image`
//! config). Weight keys are the **original Cosmos** names (`{prefix}.blocks.N.*`,
//! `{prefix}.x_embedder.proj.1`, `{prefix}.t_embedder.1.*`, `{prefix}.final_layer.*`) — the single-file
//! bf16 checkpoint loads unchanged, no diffusers rename applied. `prefix` is detected per file (`net`
//! for the base cut, `model.diffusion_model` for turbo/aesthetic; see [`crate::loader`]).
//!
//! Ported pieces: `CosmosPatchEmbed`, `CosmosTimestepEmbedding`/`CosmosEmbedding`,
//! `CosmosAdaLayerNorm(Zero)`, `CosmosAttention` (q/k RMSNorm + half-split RoPE on self-attn),
//! `CosmosTransformerBlock`, final layer. **Skipped** (config-off for Anima): learnable pos-embed
//! (`extra_pos_embed_type=null`), cross-attn projection, ControlNet hooks, img-context.

use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::fast::{layer_norm, rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, multiply, split, zeros_dtype};
use mlx_rs::transforms::checkpoint;
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{prefixed_paths, AdaptableHost, AdaptableLinear, Adapter};
use mlx_gen::nn::{gelu_exact, modulate, silu, timestep_sincos};
use mlx_gen::qkv::{self, AttnPrepSpec, QkNormSpec, QkvSource, RopeSpec, RopeStyle, RopeTables};
use mlx_gen::train::lora::LoraParams;
use mlx_gen::weights::{join, Weights};
use mlx_gen::Result;

use crate::config::DitConfig;
use crate::rope::cosmos_image_rope;

/// q/k RMSNorm eps in diffusers `Attention` (`qk_norm="rms_norm"`, default `eps=1e-5`).
const ATTN_QK_NORM_EPS: f32 = 1e-5;
/// LayerNorm / time-embed-norm eps (`elementwise_affine=false, eps=1e-6`).
const NORM_EPS: f32 = 1e-6;
/// Sinusoidal timestep-embedding `max_period` (`Timesteps` default).
const TIME_MAX_PERIOD: f64 = 10000.0;

/// Dense linear (no bias) — adapter-ready (`AdaptableLinear`); auto-detects a packed Q4/Q8 base.
fn lin(w: &Weights, name: &str) -> Result<AdaptableLinear> {
    mlx_gen::quant::lin(w, name, false, mlx_gen::quant::DEFAULT_GROUP_SIZE)
}

/// `x[:, :end]` along axis 1.
fn head_cols(x: &Array, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (0..end).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[end]), 1)?)
}

/// `CosmosEmbedding`: sinusoidal time_proj → `CosmosTimestepEmbedding` (`temb`, 3·hidden) +
/// `RMSNorm` (`embedded_timestep`, hidden).
struct TimeEmbed {
    linear_1: AdaptableLinear,
    linear_2: AdaptableLinear,
    norm: Array,
    hidden: usize,
}

impl TimeEmbed {
    fn from_weights(w: &Weights, prefix: &str, cfg: &DitConfig) -> Result<Self> {
        Ok(Self {
            linear_1: lin(w, &join(prefix, "t_embedder.1.linear_1"))?,
            linear_2: lin(w, &join(prefix, "t_embedder.1.linear_2"))?,
            norm: w.require(&join(prefix, "t_embedding_norm.weight"))?.clone(),
            hidden: cfg.hidden_size(),
        })
    }

    /// `sigma`: `[B]`. Returns `(temb [B, 3·hidden], embedded [B, hidden])` in `dtype`.
    fn forward(&self, sigma: &Array, dtype: Dtype) -> Result<(Array, Array)> {
        let proj = timestep_sincos(sigma, self.hidden, TIME_MAX_PERIOD, 0.0)?.as_dtype(dtype)?;
        let temb = self
            .linear_2
            .forward(&silu(&self.linear_1.forward(&proj)?)?)?;
        let embedded = rms_norm(&proj, &self.norm, NORM_EPS)?;
        Ok((temb, embedded))
    }
}

/// `CosmosAdaLayerNormZero` (norm1/2/3): LayerNorm(no affine) then `(1+scale)·norm + shift`, plus a
/// `gate`. `linear_2` emits `3·hidden` (shift|scale|gate), added to `temb`.
#[derive(Clone)]
struct AdaLayerNormZero {
    linear_1: AdaptableLinear,
    linear_2: AdaptableLinear,
}

impl AdaLayerNormZero {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        // e.g. prefix = "net.blocks.0.adaln_modulation_self_attn"
        Ok(Self {
            linear_1: lin(w, &join(prefix, "1"))?,
            linear_2: lin(w, &join(prefix, "2"))?,
        })
    }

    /// Returns `(modulated_norm [B,S,H], gate [B,1,H])`.
    fn forward(&self, hidden: &Array, embedded: &Array, temb: &Array) -> Result<(Array, Array)> {
        let e = self
            .linear_2
            .forward(&self.linear_1.forward(&silu(embedded)?)?)?;
        let e = add(&e, temb)?; // [B, 3H]
        let parts = split(&e, 3, 1)?; // shift, scale, gate
        let shift = parts[0].expand_dims(1)?;
        let scale = parts[1].expand_dims(1)?;
        let gate = parts[2].expand_dims(1)?;
        let normed = layer_norm(hidden, None, None, NORM_EPS)?;
        Ok((modulate(&normed, &scale, &shift, true)?, gate))
    }
}

/// `CosmosAdaLayerNorm` (final `norm_out`): LayerNorm(no affine) then `(1+scale)·norm + shift`.
/// `linear_2` emits `2·hidden` (shift|scale), added to `temb[..., :2·hidden]`.
#[derive(Clone)]
struct AdaLayerNorm {
    linear_1: AdaptableLinear,
    linear_2: AdaptableLinear,
    hidden: i32,
}

impl AdaLayerNorm {
    fn from_weights(w: &Weights, prefix: &str, cfg: &DitConfig) -> Result<Self> {
        Ok(Self {
            linear_1: lin(w, &join(prefix, "1"))?,
            linear_2: lin(w, &join(prefix, "2"))?,
            hidden: cfg.hidden_size() as i32,
        })
    }

    fn forward(&self, hidden: &Array, embedded: &Array, temb: &Array) -> Result<Array> {
        let e = self
            .linear_2
            .forward(&self.linear_1.forward(&silu(embedded)?)?)?;
        let e = add(&e, &head_cols(temb, 2 * self.hidden)?)?; // + temb[:, :2H]
        let parts = split(&e, 2, 1)?; // shift, scale
        let shift = parts[0].expand_dims(1)?;
        let scale = parts[1].expand_dims(1)?;
        let normed = layer_norm(hidden, None, None, NORM_EPS)?;
        modulate(&normed, &scale, &shift, true)
    }
}

/// `CosmosAttention` — self (attn1: q/k/v from hidden, RoPE) or cross (attn2: q from hidden, k/v from
/// text, no RoPE). Per-head q/k RMSNorm; heads == kv_heads (no GQA repeat for Anima).
#[derive(Clone)]
struct Attention {
    to_q: AdaptableLinear,
    to_k: AdaptableLinear,
    to_v: AdaptableLinear,
    to_out: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    heads: i32,
    head_dim: i32,
    scale: f32,
    /// sc-10576 — run the SDPA segment inside an `mlx::checkpoint` so its backward recomputes the
    /// attention instead of retaining the `[heads, sq, sk]` probability matrix. For the 1536² DiT
    /// (`sq ≈ 9216` image tokens) that retained self-attention array is the dominant training
    /// working-set term; MLX has no fused SDPA backward (the grad decomposes to naive attention), so
    /// wrapping it in `checkpoint` is numerically identical (same math, recomputed). Inference never
    /// sets it (default off, zero cost); the trainer enables it when whole-block checkpointing is OFF
    /// (LoKr / the dense path), and turns it off when whole-block checkpointing already covers the
    /// block (nesting would recompute attention twice for no memory win).
    ckpt_sdpa: bool,
}

impl Attention {
    fn from_weights(w: &Weights, prefix: &str, cfg: &DitConfig) -> Result<Self> {
        let hd = cfg.attention_head_dim as i32;
        Ok(Self {
            to_q: lin(w, &join(prefix, "q_proj"))?,
            to_k: lin(w, &join(prefix, "k_proj"))?,
            to_v: lin(w, &join(prefix, "v_proj"))?,
            to_out: lin(w, &join(prefix, "output_proj"))?,
            norm_q: w.require(&join(prefix, "q_norm.weight"))?.clone(),
            norm_k: w.require(&join(prefix, "k_norm.weight"))?.clone(),
            heads: cfg.num_attention_heads as i32,
            head_dim: hd,
            scale: (hd as f32).powf(-0.5),
            ckpt_sdpa: false,
        })
    }

    /// Toggle SDPA-segment gradient checkpointing (sc-10576). Training-only knob — see `ckpt_sdpa`.
    fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.ckpt_sdpa = on;
    }

    /// `hidden`: `[B, Sq, H]`. `encoder`: `Some([B, Sk, Ctx])` for cross-attn (else self-attn on
    /// `hidden`). `rope`: `Some((cos,sin))` applies half-split RoPE (self-attn only).
    ///
    /// `plan` is ladder rung 3 (SC-15524). [`AttentionPlan::UNBOUNDED`] is exactly the historical
    /// single fused call; a bounded plan splits the QUERY rows into chunks that each attend over the
    /// COMPLETE k/v and concatenates the outputs back along the query axis. That is bit-exact — it
    /// changes when score rows are materialized, never what they contain — so precision, seed,
    /// schedule and conditioning are untouched. It applies to BOTH attentions: self-attn (`Sq = Sk =`
    /// the image token count, the quadratic one) and cross-attn into the conditioner's 512 tokens.
    fn forward(
        &self,
        hidden: &Array,
        encoder: Option<&Array>,
        rope: Option<(&Array, &Array)>,
        plan: mlx_gen::attention::AttentionPlan<'_>,
    ) -> Result<Array> {
        let kv_src = encoder.unwrap_or(hidden);

        // SC-18319 — the shared prologue. The DiT attention is **knob 3's reference case**: the
        // SAME code path runs self-attention with RoPE and cross-attention with none, so "RoPE
        // optional" is a per-call knob rather than a per-family one. `RopeStyle::None` skips the
        // rotation outright (it is not an identity table, which would still cost two multiplies and
        // an add over the whole q/k stream in all 28 blocks). The rest: separate q/k/v (knob 9),
        // per-head RMSNorm over `head_dim` after the head split (knob 1), half-split `rotate_half`
        // rotation over full-width `[1, S, head_dim]` tables (knob 2 — `nn::apply_text_rope`
        // verbatim), token-major axes, one shared q/k table (knob 6 off — the conditioner is where
        // the tables differ).
        let spec = AttnPrepSpec::new(self.heads, self.head_dim)
            .with_qk_norm(QkNormSpec::per_head(
                &self.norm_q,
                &self.norm_k,
                ATTN_QK_NORM_EPS,
            ))
            .with_rope(match rope {
                Some((cos, sin)) => RopeSpec {
                    style: RopeStyle::RotateHalf,
                    q: Some(RopeTables::new(cos, sin)),
                    k: Some(RopeTables::new(cos, sin)),
                    ..RopeSpec::default()
                },
                None => RopeSpec::default(),
            });
        let heads = qkv::prepare(
            QkvSource::Separate {
                q: &self.to_q.forward(hidden)?,
                k: &self.to_k.forward(kv_src)?,
                v: &self.to_v.forward(kv_src)?,
            },
            &spec,
        )?;
        let (q, k, v) = (heads.q, heads.k, heads.v);
        let o = if self.ckpt_sdpa {
            // sc-10576: checkpoint just the SDPA. q/k/v are the threaded inputs (grads to the QKV
            // projections — and their LoRA — flow back through them); only the f32 scale is captured.
            // The backward recomputes THIS layer's decomposed attention alone, so the seq² probability
            // matrix is a per-layer transient, never retained across the 28 blocks.
            let scale = self.scale;
            let mut seg = checkpoint(move |inp: &[Array]| -> MlxResult<Vec<Array>> {
                Ok(vec![scaled_dot_product_attention(
                    &inp[0], &inp[1], &inp[2], scale, None, None,
                )?])
            });
            seg(&[q, k, v])?.into_iter().next().ok_or_else(|| {
                mlx_gen::Error::Msg("anima: checkpoint SDPA produced no output".into())
            })?
        } else {
            mlx_gen::attention::sdpa_budgeted_bhsd(&q, &k, &v, self.scale, None, plan)?
        };
        self.to_out.forward(&qkv::merge_heads(&o)?)
    }
}

/// `FeedForward(mult=4, activation="gelu")` — `net.2(gelu_exact(net.0.proj(x)))`, no bias.
#[derive(Clone)]
struct FeedForward {
    proj_in: AdaptableLinear,  // mlp.layer1 -> ff.net.0.proj
    proj_out: AdaptableLinear, // mlp.layer2 -> ff.net.2
}

impl FeedForward {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            proj_in: lin(w, &join(prefix, "layer1"))?,
            proj_out: lin(w, &join(prefix, "layer2"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.proj_out
            .forward(&gelu_exact(&self.proj_in.forward(x)?)?)
    }
}

/// `CosmosTransformerBlock`: gated self-attn → gated cross-attn → gated FF.
///
/// `pub(crate)` so ladder rung 4 ([`crate::block_stream`]) can rebuild one from the checkpoint with
/// the SAME constructor the resident stack used. A second, stream-local block constructor would be a
/// second way to build a block, which is the divergence hazard the shared ladder exists to remove.
#[derive(Clone)]
pub(crate) struct Block {
    norm1: AdaLayerNormZero,
    attn1: Attention,
    norm2: AdaLayerNormZero,
    attn2: Attention,
    norm3: AdaLayerNormZero,
    ff: FeedForward,
}

impl Block {
    pub(crate) fn from_weights(w: &Weights, prefix: &str, cfg: &DitConfig) -> Result<Self> {
        Ok(Self {
            norm1: AdaLayerNormZero::from_weights(w, &join(prefix, "adaln_modulation_self_attn"))?,
            attn1: Attention::from_weights(w, &join(prefix, "self_attn"), cfg)?,
            norm2: AdaLayerNormZero::from_weights(w, &join(prefix, "adaln_modulation_cross_attn"))?,
            attn2: Attention::from_weights(w, &join(prefix, "cross_attn"), cfg)?,
            norm3: AdaLayerNormZero::from_weights(w, &join(prefix, "adaln_modulation_mlp"))?,
            ff: FeedForward::from_weights(w, &join(prefix, "mlp"))?,
        })
    }

    /// Toggle SDPA-segment gradient checkpointing on both attentions (self + cross) — sc-10576.
    fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.attn1.set_sdpa_checkpoint(on);
        self.attn2.set_sdpa_checkpoint(on);
    }

    pub(crate) fn forward(
        &self,
        hidden: &Array,
        encoder: &Array,
        embedded: &Array,
        temb: &Array,
        rope: (&Array, &Array),
        plan: mlx_gen::attention::AttentionPlan<'_>,
    ) -> Result<Array> {
        // 1. self attention (RoPE)
        let (normed, gate) = self.norm1.forward(hidden, embedded, temb)?;
        let attn = self.attn1.forward(&normed, None, Some(rope), plan)?;
        let hidden = add(hidden, &multiply(&gate, &attn)?)?;
        // 2. cross attention (no RoPE). No attention mask over the conditioner's 512-token output: the
        // diffusers reference leaves the zero-padded positions UNMASKED too — `AnimaTextConditioner`
        // right-pads with zeros and returns a bare tensor (condition_embedder_anima.py:346, no mask),
        // and Cosmos cross-attn runs SDPA with `attn_mask=None` (transformer_cosmos.py:204). Padded keys
        // are zero vectors, not −inf, so they share logit 0; matching the reference means no mask here.
        // Do NOT "fix" this into a mask — that would introduce a conditioning divergence, not remove one.
        let (normed, gate) = self.norm2.forward(&hidden, embedded, temb)?;
        let attn = self.attn2.forward(&normed, Some(encoder), None, plan)?;
        let hidden = add(&hidden, &multiply(&gate, &attn)?)?;
        // 3. feed forward
        let (normed, gate) = self.norm3.forward(&hidden, embedded, temb)?;
        let ff = self.ff.forward(&normed)?;
        Ok(add(&hidden, &multiply(&gate, &ff)?)?)
    }
}

/// The full Cosmos-Predict2 DiT.
pub struct CosmosDiT {
    patch_embed: AdaptableLinear, // x_embedder.proj.1
    time_embed: TimeEmbed,
    blocks: Vec<Block>,
    /// Ladder rung 4 (SC-15524): how to rebuild the 28 `blocks` from the checkpoint one window at a
    /// time. `None` when the load did not select [`mlx_gen::LoadShape::DeferredMaterialization`], so
    /// a rung-4 request on an eagerly materialized generator is a typed rejection rather than a
    /// silent resident execution. See [`crate::block_stream`].
    block_stream: Option<crate::block_stream::AnimaBlockStream>,
    norm_out: AdaLayerNorm,
    proj_out: AdaptableLinear, // final_layer.linear
    cfg: DitConfig,
}

impl CosmosDiT {
    /// `prefix` is the checkpoint's DiT root (`"net"`); keys are the original Cosmos names.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: DitConfig) -> Result<Self> {
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::from_weights(
                w,
                &join(prefix, &format!("blocks.{i}")),
                &cfg,
            )?);
        }
        Ok(Self {
            patch_embed: lin(w, &join(prefix, "x_embedder.proj.1"))?,
            time_embed: TimeEmbed::from_weights(w, prefix, &cfg)?,
            blocks,
            block_stream: None,
            norm_out: AdaLayerNorm::from_weights(
                w,
                &join(prefix, "final_layer.adaln_modulation"),
                &cfg,
            )?,
            proj_out: lin(w, &join(prefix, "final_layer.linear"))?,
            cfg,
        })
    }

    pub fn config(&self) -> &DitConfig {
        &self.cfg
    }

    /// Arm ladder rung 4 (SC-15524) by recording where `blocks` can be re-read from.
    ///
    /// `source` must be the exact variant checkpoint the resident stack was built from and `prefix`
    /// the detected DiT root, so a streamed block reads the identical keys. Called by
    /// [`crate::loader::load_heavy_phase_streamable`]; deliberately not called for an eager load,
    /// whose blocks are already committed and would pay a second copy.
    pub(crate) fn with_block_stream(
        mut self,
        source: mlx_gen::WeightsSource,
        prefix: &str,
    ) -> Self {
        self.block_stream = Some(crate::block_stream::AnimaBlockStream::new(
            source, prefix, self.cfg,
        ));
        self
    }

    /// Whether rung 4 can execute on this DiT (a re-openable checkpoint was recorded).
    pub fn can_stream_blocks(&self) -> bool {
        self.block_stream.is_some()
    }

    /// Capture the adapters installed on `blocks` so a streamed block reproduces them. A no-op when
    /// rung 4 is not armed. Called by the heavy loader *after*
    /// [`crate::adapters::apply_anima_adapters`], so the resident and streamed paths cannot disagree
    /// about which adapters landed where.
    pub(crate) fn capture_block_adapters(&mut self) {
        if let Some(stream) = self.block_stream.as_mut() {
            stream.capture_adapters(&mut self.blocks);
        }
    }

    /// Turn a request's window size into the plan the windowed forward runs under, or `None` for the
    /// historical resident stack. Errors when a window is asked for but this DiT has no re-openable
    /// checkpoint, so a rung-4 request can never silently execute the resident path.
    pub(crate) fn block_window<'a>(
        &self,
        window_size: Option<usize>,
        cancel: &'a mlx_gen::CancelFlag,
    ) -> Result<Option<crate::block_stream::BlockWindow<'a>>> {
        let Some(size) = window_size else {
            return Ok(None);
        };
        if self.block_stream.is_none() {
            return Err(mlx_gen::Error::Unsupported(
                "anima: bounded transformer residency needs a deferred-materialization load with a \
                 re-openable checkpoint; this generator materialized its blocks eagerly"
                    .to_owned(),
            ));
        }
        Ok(Some(crate::block_stream::BlockWindow {
            plan: mlx_gen::block_residency::BlockPlan::new(self.blocks.len(), size)?,
            cancel,
        }))
    }

    /// Disarm rung 4 after an in-place mutation a re-materialized block could not reproduce.
    ///
    /// The stream's contract is that a materialized block is byte-identical to its resident twin,
    /// kept by replaying the captured adapters. The training-only SDPA-checkpoint flag is per-block
    /// state with NO on-disk representation, so a streamed forward after it would silently run a
    /// different backward. The rung is turned off instead: `block_window` then returns a typed error
    /// for a window request. Inference never calls the mutator, so this is inert on every render.
    fn disarm_block_stream(&mut self) {
        self.block_stream = None;
    }

    /// Patchify a `[B, C, 1, Hl, Wl]` latent (`C=17` after mask concat) to `[B, seq, C·ph·pw]`.
    fn patchify(&self, x: &Array) -> Result<Array> {
        let (pt, ph, pw) = self.cfg.patch_size;
        let sh = x.shape();
        let (b, c, t, hl, wl) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let (pt, ph, pw) = (pt as i32, ph as i32, pw as i32);
        // reshape (B, C, T/pt, pt, Hl/ph, ph, Wl/pw, pw)
        let x = x.reshape(&[b, c, t / pt, pt, hl / ph, ph, wl / pw, pw])?;
        // permute (0,2,4,6,1,3,5,7) -> (B, T/pt, Hl/ph, Wl/pw, C, pt, ph, pw)
        let x = x.transpose_axes(&[0, 2, 4, 6, 1, 3, 5, 7])?;
        // flatten patch dims -> (B, seq, C*pt*ph*pw)
        let seq = (t / pt) * (hl / ph) * (wl / pw);
        Ok(x.reshape(&[b, seq, c * pt * ph * pw])?)
    }

    /// Inverse of [`patchify`] on the `proj_out` output `[B, seq, ph·pw·pt·out_ch]` → `[B, out_ch, 1, Hl, Wl]`.
    fn unpatchify(&self, x: &Array, pe_t: i32, pe_h: i32, pe_w: i32) -> Result<Array> {
        let (pt, ph, pw) = self.cfg.patch_size;
        let (pt, ph, pw) = (pt as i32, ph as i32, pw as i32);
        let oc = self.cfg.out_channels as i32;
        let b = x.shape()[0];
        // [B, seq, ph*pw*pt*oc] -> [B, pe_t, pe_h, pe_w, ph, pw, pt, oc]
        let x = x.reshape(&[b, pe_t, pe_h, pe_w, ph, pw, pt, oc])?;
        // permute (0,7,1,6,2,4,3,5) -> [B, oc, pe_t, pt, pe_h, ph, pe_w, pw]
        let x = x.transpose_axes(&[0, 7, 1, 6, 2, 4, 3, 5])?;
        // collapse the patch pairs -> [B, oc, pe_t*pt, pe_h*ph, pe_w*pw]
        Ok(x.reshape(&[b, oc, pe_t * pt, pe_h * ph, pe_w * pw])?)
    }

    /// Shared prep + transformer-block loop. Runs `num_blocks` blocks (`None` = all) over the patchified
    /// hidden state and returns `(hidden [B, seq, hidden], (pe_t, pe_h, pe_w), embedded, temb)`. The
    /// per-op math is identical to a full [`forward`]; only the number of blocks run is parameterized —
    /// so [`forward`] and the stage-4 golden hook [`forward_hidden`] share ONE copy of the block math.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn run_blocks(
        &self,
        latents: &Array,
        sigma: &Array,
        encoder: &Array,
        dtype: Dtype,
        num_blocks: Option<usize>,
        plan: mlx_gen::attention::AttentionPlan<'_>,
        window: Option<crate::block_stream::BlockWindow<'_>>,
    ) -> Result<(Array, (i32, i32, i32), Array, Array)> {
        let latents = latents.as_dtype(dtype)?;
        let sh = latents.shape();
        let (b, _c, t, hl, wl) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let (pt, ph, pw) = self.cfg.patch_size;
        let (pe_t, pe_h, pe_w) = (t / pt as i32, hl / ph as i32, wl / pw as i32);

        // 1. concat the (all-zeros, full-res-resized-to-latent) padding mask channel -> [B,17,1,Hl,Wl].
        let hidden = if self.cfg.concat_padding_mask {
            let pad = zeros_dtype(&[b, 1, t, hl, wl], dtype)?;
            concatenate_axis(&[&latents, &pad], 1)?
        } else {
            latents
        };

        // 2. RoPE for this latent grid (per-axis OOD-guarded).
        let rope = cosmos_image_rope(&self.cfg, pe_t as usize, pe_h as usize, pe_w as usize)?;

        // 3. patchify + patch-embed -> [B, seq, hidden].
        let hidden = self.patch_embed.forward(&self.patchify(&hidden)?)?;

        // 4. time embedding.
        let (temb, embedded) = self.time_embed.forward(sigma, dtype)?;

        // 5. transformer blocks (all, or the first `num_blocks` for the stage-4 localization golden).
        let take = num_blocks.unwrap_or(self.blocks.len());
        let hidden = match window {
            // Rung 4: rebuild the stack from the checkpoint one window at a time. The lifecycle —
            // one fresh lazy view per window, materialize before release, `clear_cache` after, and a
            // cancellation check at every window boundary — belongs entirely to
            // `mlx_gen::block_residency::run_windowed`; nothing here re-implements it.
            Some(window) => {
                if take != self.blocks.len() {
                    return Err(mlx_gen::Error::Unsupported(
                        "anima: the partial-stack golden hook cannot be windowed".to_owned(),
                    ));
                }
                self.run_windowed_blocks(hidden, encoder, &embedded, &temb, &rope, plan, window)?
            }
            None => {
                let mut hidden = hidden;
                for block in self.blocks.iter().take(take) {
                    hidden = block.forward(
                        &hidden,
                        encoder,
                        &embedded,
                        &temb,
                        (&rope.cos, &rope.sin),
                        plan,
                    )?;
                }
                hidden
            }
        };
        Ok((hidden, (pe_t, pe_h, pe_w), embedded, temb))
    }

    /// Rung 4's block loop: walk the 28 blocks in windows, materializing each window's weights from
    /// the checkpoint and releasing them before the next window opens.
    ///
    /// The `materialize` callback is the MLX-specific trap SC-15750 measured and pinned: MLX is
    /// lazy, so the carried hidden state is an unevaluated graph node still referencing the window's
    /// weights, and dropping before forcing evaluation frees NOTHING (correct output either way —
    /// silent, not loud). Forcing the carried activation is what makes the window's release real.
    #[allow(clippy::too_many_arguments)]
    fn run_windowed_blocks(
        &self,
        hidden: Array,
        encoder: &Array,
        embedded: &Array,
        temb: &Array,
        rope: &crate::rope::CosmosRope,
        plan: mlx_gen::attention::AttentionPlan<'_>,
        window: crate::block_stream::BlockWindow<'_>,
    ) -> Result<Array> {
        let stream = self.block_stream.as_ref().ok_or_else(|| {
            mlx_gen::Error::Unsupported(
                "anima: a transformer window was planned without a re-openable checkpoint"
                    .to_owned(),
            )
        })?;
        if window.plan.n_blocks() != self.blocks.len() || stream.n_blocks() != self.blocks.len() {
            return Err(mlx_gen::Error::Msg(format!(
                "anima: block-window plan covers {} blocks and the stream {}, but the stack has {}",
                window.plan.n_blocks(),
                stream.n_blocks(),
                self.blocks.len()
            )));
        }
        mlx_gen::block_residency::run_windowed(
            &window.plan,
            window.cancel,
            hidden,
            || stream.open(),
            |mut hidden, view, range| {
                for index in range {
                    let block = stream.materialize(view, index)?;
                    hidden = block.forward(
                        &hidden,
                        encoder,
                        embedded,
                        temb,
                        (&rope.cos, &rope.sin),
                        plan,
                    )?;
                }
                Ok(hidden)
            },
            |hidden| Ok(mlx_rs::transforms::eval([hidden])?),
        )
    }

    /// Denoise forward. `latents`: `[B, 16, 1, Hl, Wl]` (any dtype — cast to `dtype`). `sigma`: `[B]`.
    /// `encoder`: `[B, 512, text_embed_dim]`. Returns the velocity `[B, 16, 1, Hl, Wl]` in `dtype`.
    pub fn forward(
        &self,
        latents: &Array,
        sigma: &Array,
        encoder: &Array,
        dtype: Dtype,
    ) -> Result<Array> {
        self.forward_bounded(
            latents,
            sigma,
            encoder,
            dtype,
            mlx_gen::attention::AttentionPlan::UNBOUNDED,
            None,
        )
    }

    /// [`forward`](Self::forward) under an explicit memory plan — ladder rungs 3 and 4 (SC-15524).
    ///
    /// `plan` bounds the attention scratch; `window` rebuilds the block stack from the checkpoint one
    /// window at a time instead of reading the resident `blocks`. Both are execution-shape choices
    /// only: the same arithmetic, in the same order, over the same weights. `UNBOUNDED` + `None` is
    /// exactly [`forward`](Self::forward).
    pub(crate) fn forward_bounded(
        &self,
        latents: &Array,
        sigma: &Array,
        encoder: &Array,
        dtype: Dtype,
        plan: mlx_gen::attention::AttentionPlan<'_>,
        window: Option<crate::block_stream::BlockWindow<'_>>,
    ) -> Result<Array> {
        let (hidden, (pe_t, pe_h, pe_w), embedded, temb) =
            self.run_blocks(latents, sigma, encoder, dtype, None, plan, window)?;
        // 6. output norm + projection + unpatchify.
        let hidden = self.norm_out.forward(&hidden, &embedded, &temb)?;
        let hidden = self.proj_out.forward(&hidden)?;
        self.unpatchify(&hidden, pe_t, pe_h, pe_w)
    }

    /// Toggle SDPA-segment gradient checkpointing (sc-10576) on every DiT block's self- + cross-attn.
    /// Training-only: the trainer turns it ON when whole-block checkpointing is OFF (the dense / LoKr
    /// path) and OFF when `forward_with_main_checkpointed` is
    /// used (the block recompute already covers attention). Numerically identical to the retained
    /// backward; inference never calls it.
    pub fn set_sdpa_checkpoint(&mut self, on: bool) {
        for b in &mut self.blocks {
            b.set_sdpa_checkpoint(on);
        }
        // The flag has no on-disk representation, so a re-materialized block could not reproduce it.
        // Disarm rung 4 rather than let a streamed forward run a silently different backward.
        self.disarm_block_stream();
    }

    /// Training forward with **per-block gradient checkpointing** (sc-10576). Numerically identical to
    /// [`forward`](Self::forward), but each of the 28 DiT blocks runs inside an `mlx::transforms::checkpoint`
    /// segment whose EXPLICIT inputs are `[hidden, encoder, a_0, b_0, …]` — the block hidden state, the
    /// conditioner output, and that block's trainable LoRA factors — so the reverse pass RECOMPUTES the
    /// block instead of retaining its (seq²) activations, while gradients still flow to the factors AND
    /// to `encoder`.
    ///
    /// **`encoder` is threaded, never captured (the sc-10522 correctness trap).** Anima cross-attends
    /// the conditioner output in EVERY block; if `encoder` were captured as a closure constant the
    /// backward would produce no cotangent for it, the 60 `llm_adapter` conditioner adapters would
    /// receive ZERO gradient (silently inert while target-count and DiT-loss checks still pass), and the
    /// conditioner would never train. Threading it as an explicit `checkpoint` input makes its gradient
    /// accumulate across the 28 segments and flow back through the (in-graph) conditioner forward.
    /// `embedded`/`temb`/`rope` come from the untrained time-embed and are captured (constants).
    ///
    /// The pre-block patch/time embed and the post-block `norm_out`/`proj_out`/unpatchify run normally
    /// (their LoRA, if any, is installed on `self` by the caller and trains through ordinary autograd);
    /// the block stack is where the activation memory concentrates, so that is what is checkpointed.
    /// `params` is the live trainable factor map; `block_local_targets[i]` lists the adapter-routable
    /// LOCAL paths (e.g. `"self_attn.q_proj"`, `"adaln_modulation_mlp.2"`) trained on block `i`, in the
    /// order their `lora_a`/`lora_b` are threaded. Blocks with no trained targets still run checkpointed
    /// (hidden+encoder inputs only), so the whole stack is uniformly recompute-on-backward.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_with_main_checkpointed(
        &self,
        latents: &Array,
        sigma: &Array,
        encoder: &Array,
        dtype: Dtype,
        params: &LoraParams,
        block_local_targets: &[Vec<String>],
        alpha: f32,
    ) -> Result<Array> {
        self.checkpointed_impl(
            latents,
            sigma,
            encoder,
            dtype,
            params,
            block_local_targets,
            alpha,
            true,
        )
    }

    /// The shared body behind `forward_with_main_checkpointed`.
    /// `thread_encoder` gates the sc-10522 trap: `true` threads `encoder` as an explicit checkpoint
    /// input (the production path); `false` CAPTURES it as a constant (the deliberately-wrong path the
    /// grad-parity mutation test drives to prove the conditioner grads collapse to zero when captured).
    #[allow(clippy::too_many_arguments)]
    fn checkpointed_impl(
        &self,
        latents: &Array,
        sigma: &Array,
        encoder: &Array,
        dtype: Dtype,
        params: &LoraParams,
        block_local_targets: &[Vec<String>],
        alpha: f32,
        thread_encoder: bool,
    ) -> Result<Array> {
        // --- prep: identical to `run_blocks` steps 1-4 (mask concat, RoPE, patch-embed, time-embed) ---
        let latents = latents.as_dtype(dtype)?;
        let sh = latents.shape();
        let (b, _c, t, hl, wl) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let (pt, ph, pw) = self.cfg.patch_size;
        let (pe_t, pe_h, pe_w) = (t / pt as i32, hl / ph as i32, wl / pw as i32);

        let hidden = if self.cfg.concat_padding_mask {
            let pad = zeros_dtype(&[b, 1, t, hl, wl], dtype)?;
            concatenate_axis(&[&latents, &pad], 1)?
        } else {
            latents
        };
        let rope = cosmos_image_rope(&self.cfg, pe_t as usize, pe_h as usize, pe_w as usize)?;
        let mut hidden = self.patch_embed.forward(&self.patchify(&hidden)?)?;
        let (temb, embedded) = self.time_embed.forward(sigma, dtype)?;

        // --- checkpointed block loop ---
        for (i, block) in self.blocks.iter().enumerate() {
            // Cheap clone (Arrays are refcounted): the closure must OWN its state — the backward
            // recompute runs after this frame is gone, so a borrow of `self` would dangle.
            let mut blk = block.clone();
            let locals = block_local_targets.get(i).cloned().unwrap_or_default();
            // Captured constants (from the untrained time-embed + RoPE): identical to the dense path.
            let emb_c = embedded.clone();
            let temb_c = temb.clone();
            let cos_c = rope.cos.clone();
            let sin_c = rope.sin.clone();
            let alpha_c = alpha;

            // Threaded inputs: [hidden, (encoder), a_0, b_0, a_1, b_1, …] (raw `[r,in]`/`[out,r]`).
            let mut inputs: Vec<Array> = Vec::with_capacity(2 + 2 * locals.len());
            inputs.push(hidden.clone());
            if thread_encoder {
                inputs.push(encoder.clone());
            }
            let enc_captured = (!thread_encoder).then(|| encoder.clone());
            let factor_base = if thread_encoder { 2 } else { 1 };
            for local in &locals {
                let ak = format!("blocks.{i}.{local}.lora_a");
                let bk = format!("blocks.{i}.{local}.lora_b");
                inputs.push(
                    params
                        .get(ak.as_str())
                        .ok_or_else(|| mlx_gen::Error::Msg(format!("LoRA param missing: {ak}")))?
                        .clone(),
                );
                inputs.push(
                    params
                        .get(bk.as_str())
                        .ok_or_else(|| mlx_gen::Error::Msg(format!("LoRA param missing: {bk}")))?
                        .clone(),
                );
            }

            let locals_c = locals.clone();
            let mut seg = checkpoint(move |inp: &[Array]| -> MlxResult<Vec<Array>> {
                // Reinstall the explicit-input factors with the SAME `(transpose, alpha/rank fold,
                // scale=1)` `install_training_lora` applies, so the checkpointed block forward is
                // numerically identical to the installed-adapter path and grads route back to `inp`.
                // Dtype-following on the hidden state (bf16 training): the f32 factors join the bf16
                // stream or every adapted Linear would re-promote the block to f32.
                let dt = inp[0].dtype();
                for (j, local) in locals_c.iter().enumerate() {
                    let a = inp[factor_base + 2 * j].t().as_dtype(dt)?; // [r,in] -> [in,r]
                    let rank = a.shape()[1] as f32;
                    let bmat = inp[factor_base + 2 * j + 1]
                        .t() // [out,r] -> [r,out]
                        .multiply(Array::from_slice(&[alpha_c / rank], &[1]))?
                        .as_dtype(dt)?;
                    let segs: Vec<&str> = local.split('.').collect();
                    blk.adaptable_mut(&segs)
                        .ok_or_else(|| {
                            Exception::custom(format!("checkpoint LoRA target not found: {local}"))
                        })?
                        .set_training_adapters(vec![Adapter::Lora {
                            a,
                            b: bmat,
                            scale: 1.0,
                        }]);
                }
                let enc_ref: &Array = if thread_encoder {
                    &inp[1]
                } else {
                    enc_captured
                        .as_ref()
                        .expect("enc_captured is Some on the capture path")
                };
                let out = blk
                    .forward(
                        &inp[0],
                        enc_ref,
                        &emb_c,
                        &temb_c,
                        (&cos_c, &sin_c),
                        // Training never selects a bounded rung: `eval` is invalid inside an autograd
                        // trace, and `AttentionBudget::CONSTRAINED` sets `eval_per_chunk`.
                        mlx_gen::attention::AttentionPlan::UNBOUNDED,
                    )
                    .map_err(|e| Exception::custom(e.to_string()))?;
                Ok(vec![out])
            });
            hidden = seg(&inputs)?.into_iter().next().ok_or_else(|| {
                mlx_gen::Error::Msg("anima: checkpoint block produced no output".into())
            })?;
        }

        // --- output tail: identical to `forward` (norm_out → proj_out → unpatchify) ---
        let hidden = self.norm_out.forward(&hidden, &embedded, &temb)?;
        let hidden = self.proj_out.forward(&hidden)?;
        self.unpatchify(&hidden, pe_t, pe_h, pe_w)
    }

    /// Test-only sibling of `forward_with_main_checkpointed`
    /// that deliberately CAPTURES `encoder` instead of threading it — the sc-10522 inert-conditioner
    /// bug. Used only by the grad-parity mutation test to prove the conditioner factors receive zero
    /// gradient on the wrong path (so the correct path's non-zero conditioner grads are meaningful).
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_with_main_checkpointed_encoder_captured(
        &self,
        latents: &Array,
        sigma: &Array,
        encoder: &Array,
        dtype: Dtype,
        params: &LoraParams,
        block_local_targets: &[Vec<String>],
        alpha: f32,
    ) -> Result<Array> {
        self.checkpointed_impl(
            latents,
            sigma,
            encoder,
            dtype,
            params,
            block_local_targets,
            alpha,
            false,
        )
    }

    /// Test hook (sc-10524 stage-4 golden): the raw hidden state after `num_blocks` transformer blocks
    /// (**pre** `norm_out` / `proj_out` / unpatchify), shape `[B, seq, hidden]`. `None` runs all blocks.
    /// Lets the parity golden localize a drift to a single block vs. the full 28-block stack.
    #[doc(hidden)]
    pub fn forward_hidden(
        &self,
        latents: &Array,
        sigma: &Array,
        encoder: &Array,
        dtype: Dtype,
        num_blocks: Option<usize>,
    ) -> Result<Array> {
        Ok(self
            .run_blocks(
                latents,
                sigma,
                encoder,
                dtype,
                num_blocks,
                mlx_gen::attention::AttentionPlan::UNBOUNDED,
                None,
            )?
            .0)
    }
}

// ---- LoRA/LoKr adapter surface (sc-10521) --------------------------------------------------------
//
// The official Anima LoRAs address DiT modules by their **original Cosmos** names (the same names the
// checkpoint uses — `self_attn`/`cross_attn`, `q_proj`/`k_proj`/`v_proj`/`output_proj`,
// `mlp.layer1`/`mlp.layer2`, and the three `adaln_modulation_{self_attn,cross_attn,mlp}.{1,2}` down/up
// pairs), under the ComfyUI `diffusion_model.` prefix (stripped by the loader). Every projection is an
// `AdaptableLinear`, so the host is just a dotted-path → `&mut AdaptableLinear` router. The three
// adaLN-modulation `.1`/`.2` pairs ARE LoRA targets (adaln_lora_dim 256) — easy to miss (16 of a
// block's targets, 6 of which are adaLN), so they are routed here explicitly.

impl AdaptableHost for Attention {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["q_proj"] => Some(&mut self.to_q),
            ["k_proj"] => Some(&mut self.to_k),
            ["v_proj"] => Some(&mut self.to_v),
            ["output_proj"] => Some(&mut self.to_out),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["q_proj", "k_proj", "v_proj", "output_proj"]
            .into_iter()
            .map(String::from)
            .collect()
    }
}

impl AdaptableHost for FeedForward {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["layer1"] => Some(&mut self.proj_in),
            ["layer2"] => Some(&mut self.proj_out),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["layer1", "layer2"].into_iter().map(String::from).collect()
    }
}

/// The adaLN-modulation `SiLU → linear_1 → linear_2` down/up pair is addressed by the trained files as
/// `‹adaln_modulation_*›.1` / `.2` — both LoRA targets (the `.2` up-projection's out_features is
/// `adaln_lora_dim`·… , confirming these pairs really are adapted). Shared by [`AdaLayerNormZero`]
/// (blocks) and [`AdaLayerNorm`] (final layer) via their identical `linear_1`/`linear_2` fields.
impl AdaptableHost for AdaLayerNormZero {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["1"] => Some(&mut self.linear_1),
            ["2"] => Some(&mut self.linear_2),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["1", "2"].into_iter().map(String::from).collect()
    }
}

impl AdaptableHost for AdaLayerNorm {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["1"] => Some(&mut self.linear_1),
            ["2"] => Some(&mut self.linear_2),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["1", "2"].into_iter().map(String::from).collect()
    }
}

impl AdaptableHost for Block {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["self_attn", rest @ ..] => self.attn1.adaptable_mut(rest),
            ["cross_attn", rest @ ..] => self.attn2.adaptable_mut(rest),
            ["mlp", rest @ ..] => self.ff.adaptable_mut(rest),
            ["adaln_modulation_self_attn", rest @ ..] => self.norm1.adaptable_mut(rest),
            ["adaln_modulation_cross_attn", rest @ ..] => self.norm2.adaptable_mut(rest),
            ["adaln_modulation_mlp", rest @ ..] => self.norm3.adaptable_mut(rest),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = Vec::new();
        out.extend(prefixed_paths("self_attn", &self.attn1));
        out.extend(prefixed_paths("cross_attn", &self.attn2));
        out.extend(prefixed_paths("mlp", &self.ff));
        out.extend(prefixed_paths("adaln_modulation_self_attn", &self.norm1));
        out.extend(prefixed_paths("adaln_modulation_cross_attn", &self.norm2));
        out.extend(prefixed_paths("adaln_modulation_mlp", &self.norm3));
        out
    }
}

/// The Cosmos DiT adapter host: `blocks.N.*` route into the per-block leaves (the 448 = 28×16 target
/// surface the official LoRAs address); the DiT globals (`x_embedder.proj.1`, `t_embedder.1.linear_*`,
/// `final_layer.*`) are routable too, though the shipped Anima LoRAs never target them. Only the
/// block targets are enumerated for the kohya table (matching Z-Image's convention — globals stay
/// reachable via the dotted diffusers/peft form, which is all the real files use).
impl AdaptableHost for CosmosDiT {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["blocks", n, rest @ ..] => self
                .blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            ["x_embedder", "proj", "1"] => Some(&mut self.patch_embed),
            ["t_embedder", "1", "linear_1"] => Some(&mut self.time_embed.linear_1),
            ["t_embedder", "1", "linear_2"] => Some(&mut self.time_embed.linear_2),
            ["final_layer", "adaln_modulation", rest @ ..] => self.norm_out.adaptable_mut(rest),
            ["final_layer", "linear"] => Some(&mut self.proj_out),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (i, b) in self.blocks.iter().enumerate() {
            out.extend(prefixed_paths(&format!("blocks.{i}"), b));
        }
        out
    }
}

// -------------------------------------------------------------------------------------------------
// Test-only structural constructor
// -------------------------------------------------------------------------------------------------

/// Build the full DiT module tree with **placeholder** (1×1) weights so the target-enumeration guard
/// (sc-10522) can exercise the real [`AdaptableHost::adaptable_paths`] surface without the licensed
/// checkpoint or any Metal compute — enumeration only walks the module tree; no placeholder tensor is
/// ever evaluated.
#[cfg(test)]
mod structural {
    use super::*;

    fn ph_lin() -> AdaptableLinear {
        AdaptableLinear::dense(Array::from_slice(&[0.0f32], &[1, 1]), None)
    }
    fn ph_norm() -> Array {
        Array::from_slice(&[1.0f32], &[1])
    }

    impl CosmosDiT {
        /// A weight-free DiT with `cfg.num_layers` structurally-complete blocks (28 for Anima),
        /// for the path-enumeration guard. Placeholder tensors only; nothing is evaluated.
        pub(crate) fn structural(cfg: DitConfig) -> Self {
            let heads = cfg.num_attention_heads as i32;
            let head_dim = cfg.attention_head_dim as i32;
            let attn = || Attention {
                to_q: ph_lin(),
                to_k: ph_lin(),
                to_v: ph_lin(),
                to_out: ph_lin(),
                norm_q: ph_norm(),
                norm_k: ph_norm(),
                heads,
                head_dim,
                scale: 1.0,
                ckpt_sdpa: false,
            };
            let adaln0 = || AdaLayerNormZero {
                linear_1: ph_lin(),
                linear_2: ph_lin(),
            };
            let blocks = (0..cfg.num_layers)
                .map(|_| Block {
                    norm1: adaln0(),
                    attn1: attn(),
                    norm2: adaln0(),
                    attn2: attn(),
                    norm3: adaln0(),
                    ff: FeedForward {
                        proj_in: ph_lin(),
                        proj_out: ph_lin(),
                    },
                })
                .collect();
            Self {
                patch_embed: ph_lin(),
                time_embed: TimeEmbed {
                    linear_1: ph_lin(),
                    linear_2: ph_lin(),
                    norm: ph_norm(),
                    hidden: cfg.hidden_size(),
                },
                blocks,
                block_stream: None,
                norm_out: AdaLayerNorm {
                    linear_1: ph_lin(),
                    linear_2: ph_lin(),
                    hidden: cfg.hidden_size() as i32,
                },
                proj_out: ph_lin(),
                cfg,
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Test-only SYNTHETIC constructor — real (small) random weights, evaluable on Metal (sc-10576)
// -------------------------------------------------------------------------------------------------

/// Build a dimension-parametric DiT with real, small random weights so the grad-parity test
/// (sc-10576) can run a checkpointed vs. dense forward+backward on Metal without the licensed 2B
/// checkpoint. Every module shape is derived from `cfg`, so a tiny `cfg` (few heads, 2 blocks) yields
/// a cheap-but-structurally-faithful model whose block loop, cross-attention into `encoder`, and full
/// adapter surface (16 targets/block) match the real DiT exactly.
#[cfg(test)]
pub(crate) mod synthetic {
    use super::*;
    use mlx_rs::random;

    /// A deterministic per-tensor RNG: each `lin`/`norm` call advances the seed so every weight is
    /// distinct but reproducible. Weights are scaled down (0.05) so a 2-block forward stays finite.
    pub(crate) struct Rng(pub u64);

    impl Rng {
        fn lin(&mut self, out: i32, in_f: i32) -> AdaptableLinear {
            self.0 = self.0.wrapping_add(1);
            let key = random::key(self.0).unwrap();
            let w = random::normal::<f32>(&[out, in_f], None, None, Some(&key)).unwrap();
            let w = multiply(&w, Array::from_slice(&[0.05f32], &[1])).unwrap();
            AdaptableLinear::dense(w, None)
        }

        /// RMSNorm scale ≈ 1 (a fresh net's norm weights), so `rms_norm` doesn't zero the stream.
        fn norm(&mut self, d: i32) -> Array {
            Array::ones::<f32>(&[d]).unwrap()
        }

        fn attn(&mut self, heads: i32, head_dim: i32, q_in: i32, kv_in: i32) -> Attention {
            let inner = heads * head_dim;
            Attention {
                to_q: self.lin(inner, q_in),
                to_k: self.lin(inner, kv_in),
                to_v: self.lin(inner, kv_in),
                to_out: self.lin(q_in, inner),
                norm_q: self.norm(head_dim),
                norm_k: self.norm(head_dim),
                heads,
                head_dim,
                scale: (head_dim as f32).powf(-0.5),
                ckpt_sdpa: false,
            }
        }

        fn adaln0(&mut self, h: i32, adaln: i32) -> AdaLayerNormZero {
            AdaLayerNormZero {
                linear_1: self.lin(adaln, h),     // h -> adaln_inner
                linear_2: self.lin(3 * h, adaln), // adaln_inner -> 3·hidden (shift|scale|gate)
            }
        }
    }

    impl CosmosDiT {
        /// A synthetic DiT with real random weights (see module docs). `cfg` sets every dim; use a
        /// tiny `cfg` for a Metal-cheap grad-parity model.
        pub(crate) fn synthetic(cfg: DitConfig, seed: u64) -> Self {
            let h = cfg.hidden_size() as i32;
            let heads = cfg.num_attention_heads as i32;
            let head_dim = cfg.attention_head_dim as i32;
            let (pt, ph, pw) = cfg.patch_size;
            let (pt, ph, pw) = (pt as i32, ph as i32, pw as i32);
            let patch_in = cfg.patch_in_channels() as i32 * pt * ph * pw;
            let patch_out = pt * ph * pw * cfg.out_channels as i32;
            let text = cfg.text_embed_dim as i32;
            let adaln = cfg.adaln_lora_dim as i32;
            let ff = (cfg.mlp_ratio * h as f32).round() as i32;

            let mut r = Rng(seed);
            let blocks = (0..cfg.num_layers)
                .map(|_| Block {
                    norm1: r.adaln0(h, adaln),
                    attn1: r.attn(heads, head_dim, h, h), // self-attn: kv from hidden
                    norm2: r.adaln0(h, adaln),
                    attn2: r.attn(heads, head_dim, h, text), // cross-attn: kv from encoder (text_dim)
                    norm3: r.adaln0(h, adaln),
                    ff: FeedForward {
                        proj_in: r.lin(ff, h),
                        proj_out: r.lin(h, ff),
                    },
                })
                .collect();
            Self {
                patch_embed: r.lin(h, patch_in),
                time_embed: TimeEmbed {
                    linear_1: r.lin(h, h),     // sincos[hidden] -> hidden
                    linear_2: r.lin(3 * h, h), // -> temb[3·hidden]
                    norm: r.norm(h),
                    hidden: cfg.hidden_size(),
                },
                blocks,
                block_stream: None,
                norm_out: AdaLayerNorm {
                    linear_1: r.lin(adaln, h),
                    linear_2: r.lin(2 * h, adaln), // shift|scale (2·hidden)
                    hidden: h,
                },
                proj_out: r.lin(patch_out, h),
                cfg,
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Ladder rung 3 — bounded attention, weight-free (SC-15524)
// -------------------------------------------------------------------------------------------------

/// The MLX twin of `mlx_gen_z_image`'s `tests/attention_budget_dit.rs`, on the synthetic DiT so it
/// runs in ordinary CI rather than only on a Mac holding the licensed 4.18 GB checkpoint.
///
/// ## Why a tripped cancel flag is the assertion
///
/// An equivalence check alone — "chunked output == unbounded output" — is a **false green**: it
/// passes trivially if the plan never reaches an attention, or if the budget never chunks. Deleting
/// the whole rung leaves it green, and that is exactly the hole this module closes.
///
/// So each equivalence check is paired with a plan whose cancel flag is **already tripped**. Between
/// chunks, [`mlx_gen::attention::sdpa_budgeted_bhsd`] consults that flag and returns
/// [`mlx_gen::Error::Canceled`]; the unbounded fast path has no chunk boundary and never looks at
/// it. A forward that returns `Ok` under a tripped-cancel *chunking* plan therefore proves one of:
/// the plan was dropped somewhere in the call chain, or the budget did not chunk.
///
/// ## Why two budgets rather than one
///
/// Every Anima block runs **two** attentions — self-attn over the image tokens and cross-attn into
/// the conditioner output — and `Block::forward` hands the same plan to both. One tripped-cancel
/// case cannot distinguish "the plan reached both" from "the plan reached whichever one runs first".
/// The budget is a score-element cap, and the two attentions have different key lengths, so a budget
/// can be sized to chunk exactly one of them: with 2 heads, `Sq = 16` image tokens and an encoder of
/// `Sk` tokens, self-attn chunks iff `budget < 2·16·16 = 512` and cross-attn iff `budget < 2·16·Sk`.
/// A short encoder (`Sk = 4`) isolates the self-attn; a long one (`Sk = 64`) isolates the cross.
#[cfg(test)]
mod bounded_attention_tests {
    use super::*;
    use mlx_gen::attention::{AttentionBudget, AttentionPlan};
    use mlx_gen::{CancelFlag, Error};
    use mlx_rs::random;

    /// 2 heads × 8 = hidden 16, 2 blocks. `max_size` is post-`patch_size` (1,2,2), so an 8×8 latent
    /// grid patchifies to 4×4 = **16 image tokens** — the `Sq` both attentions see.
    fn cfg() -> DitConfig {
        DitConfig {
            in_channels: 4,
            out_channels: 4,
            num_attention_heads: 2,
            attention_head_dim: 8,
            num_layers: 2,
            mlp_ratio: 2.0,
            text_embed_dim: 16,
            adaln_lora_dim: 8,
            max_size: (4, 16, 16),
            patch_size: (1, 2, 2),
            rope_scale: (1.0, 4.0, 4.0),
            concat_padding_mask: true,
        }
    }

    const IMAGE_TOKENS: i32 = 16;
    const HEADS: u64 = 2;

    /// `[1, C, 1, 8, 8]` latents, a `[1]` sigma and a `[1, encoder_tokens, 16]` conditioner output.
    fn inputs(cfg: &DitConfig, encoder_tokens: i32) -> (Array, Array, Array) {
        let key = random::key(7).unwrap();
        let latents = random::normal::<f32>(
            &[1, cfg.in_channels as i32, 1, 8, 8][..],
            None,
            None,
            Some(&key),
        )
        .unwrap();
        let sigma = Array::from_slice(&[0.7f32], &[1]);
        let key = random::key(9).unwrap();
        let encoder = random::normal::<f32>(
            &[1, encoder_tokens, cfg.text_embed_dim as i32][..],
            None,
            None,
            Some(&key),
        )
        .unwrap();
        (latents, sigma, encoder)
    }

    /// The largest budget that still chunks a `[1, HEADS, IMAGE_TOKENS, D] × [.., sk, D]` call is
    /// `HEADS · sk · (IMAGE_TOKENS - 1)`; one element above `HEADS · sk · IMAGE_TOKENS` never does.
    /// Derived rather than hardcoded so a config change cannot silently turn a chunking case into a
    /// single-call one and leave the tripped-cancel assertion unreachable.
    fn chunks_at(sk: i32) -> u64 {
        HEADS * sk as u64 * (IMAGE_TOKENS as u64 - 1)
    }
    fn single_call_at(sk: i32) -> u64 {
        HEADS * sk as u64 * IMAGE_TOKENS as u64
    }

    fn peak_relative_delta(a: &Array, b: &Array) -> f32 {
        let n: i32 = b.shape().iter().product();
        let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
        let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
        let (xs, ys) = (a.as_slice::<f32>(), b.as_slice::<f32>());
        let peak = ys.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
        xs.iter()
            .zip(ys)
            .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
            / peak
    }

    /// Chunking changes the query GEMM's `M` dimension, so the two forwards are tolerance-equivalent
    /// rather than bit-identical (the same 1e-3 bound the sibling families use over a whole forward).
    const TOLERANCE: f32 = 1e-3;

    /// The plan reaches BOTH DiT attentions, and it really chunks at each of them.
    #[test]
    fn the_attention_plan_reaches_both_dit_attentions_and_chunks() {
        let cfg = cfg();
        let model = CosmosDiT::synthetic(cfg, 11);
        let cancel = CancelFlag::default();
        cancel.cancel();

        for (label, encoder_tokens, budget) in [
            // A short encoder: cross-attn already fits, so ONLY the self-attn can chunk. An abort
            // therefore proves the plan reached the self-attention.
            ("self-attn", 4, chunks_at(IMAGE_TOKENS)),
            // A long encoder: the budget is above the self-attn's whole score matrix, so only the
            // cross-attn chunks. An abort proves the plan reached the cross-attention.
            ("cross-attn", 64, single_call_at(IMAGE_TOKENS)),
        ] {
            let (latents, sigma, encoder) = inputs(&cfg, encoder_tokens);
            // The isolation this case claims, asserted rather than assumed: exactly one of the two
            // attentions is over its budget at this (budget, encoder) pair.
            let self_chunks = budget < single_call_at(IMAGE_TOKENS);
            let cross_chunks = budget < single_call_at(encoder_tokens);
            assert!(
                self_chunks ^ cross_chunks,
                "{label}: the budget must isolate ONE attention (self={self_chunks}, \
                 cross={cross_chunks}), else an abort cannot attribute the plan to it"
            );

            let plan = AttentionPlan::budgeted(AttentionBudget::from_score_elements(budget, true))
                .with_cancel(&cancel);
            match model.forward_bounded(&latents, &sigma, &encoder, Dtype::Float32, plan, None) {
                Err(Error::Canceled) => {}
                other => panic!(
                    "{label}: a tripped cancel under a plan that chunks the {label} must abort the \
                     forward between chunks — got {:?}. Either the plan is not threaded to that \
                     attention, or the budget did not chunk there, which would make rung 3 \
                     unfalsifiable on this route.",
                    other.map(|v| v.shape().to_vec())
                ),
            }

            // ...and the unbounded fast path has no chunk boundary, so the same tripped flag cannot
            // affect it. This is what makes the abort above attributable to CHUNKING.
            let plan = AttentionPlan::UNBOUNDED.with_cancel(&cancel);
            assert!(
                model
                    .forward_bounded(&latents, &sigma, &encoder, Dtype::Float32, plan, None)
                    .is_ok(),
                "{label}: a tripped cancel must not affect the unbounded (default) forward"
            );
        }
    }

    /// Bounded attention is a memory-shape choice, not a numeric one: a plan that chunks BOTH
    /// attentions must reproduce the unbounded velocity, and the default `forward` must be
    /// **exactly** the explicitly-unbounded one.
    #[test]
    fn the_chunked_forward_matches_the_unbounded_forward() {
        let cfg = cfg();
        let model = CosmosDiT::synthetic(cfg, 11);
        let encoder_tokens = 16;
        let (latents, sigma, encoder) = inputs(&cfg, encoder_tokens);

        // Under the smaller of the two ceilings, both attentions chunk.
        let budget = chunks_at(IMAGE_TOKENS).min(chunks_at(encoder_tokens));
        assert!(budget < single_call_at(IMAGE_TOKENS) && budget < single_call_at(encoder_tokens));
        let plan = AttentionPlan::budgeted(AttentionBudget::from_score_elements(budget, true));

        let unbounded = model
            .forward(&latents, &sigma, &encoder, Dtype::Float32)
            .unwrap();
        let chunked = model
            .forward_bounded(&latents, &sigma, &encoder, Dtype::Float32, plan, None)
            .unwrap();
        assert_eq!(chunked.shape(), unbounded.shape(), "output shape");
        let delta = peak_relative_delta(&chunked, &unbounded);
        assert!(
            delta < TOLERANCE,
            "bounded attention changed the DiT velocity: peak-relative delta {delta:e}"
        );

        // The historical path is untouched: `forward` and an explicitly UNBOUNDED `forward_bounded`
        // agree EXACTLY, not within a tolerance.
        let explicit = model
            .forward_bounded(
                &latents,
                &sigma,
                &encoder,
                Dtype::Float32,
                AttentionPlan::UNBOUNDED,
                None,
            )
            .unwrap();
        assert_eq!(peak_relative_delta(&unbounded, &explicit), 0.0);
    }
}
