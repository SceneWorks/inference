//! SDXL dual-CLIP conditioning (sc-5491, epic 5480) — the candle twin of
//! `mlx-gen-sdxl::pipeline::encode_conditioning`. Plain SDXL and the edit/control conditioners
//! share the window encoding and projection used for SDXL micro-conditioning:
//!
//! - **cross-attention conditioning** `[B, n·77, 2048]` = `cat(penultimate(CLIP-L)[768],
//!   penultimate(CLIP-bigG)[1280])` — the *second-to-last* encoder-layer hidden state (`hidden_states[-2]`,
//!   pre-final-layer-norm) of each encoder, via candle's
//!   [`ClipTextTransformer::forward_until_encoder_layer`] at `until_layer = -2`;
//! - **pooled text-embeds** `[B, 1280]` = `text_projection(finalₙₒᵣₘ(CLIP-bigG)[eos])` — the bigG
//!   final-layer-norm hidden at the EOS position (the arg-max token, EOS being the highest id) projected
//!   through `text_projection` (the `CLIPTextModelWithProjection` head candle's `ClipTextTransformer`
//!   omits, loaded here from the same `text_encoder_2` checkpoint).
//!
//! **sc-20528:** a prompt past CLIP's architectural 77-token context is split into windows by the
//! crate-private `long_prompt` chunker and the windows' hidden states are concatenated on the sequence axis
//! (`n` = 1 for every prompt that fits, so those encodings are unchanged). The pooled embed stays
//! single-window — it is taken from window 0's EOS.
//!
//! **CFG batch order is uncond-first** (`[negative, prompt]`) to match the candle txt2img +
//! [`crate::denoise`] convention (`eps_uncond + cfg·(eps_cond − eps_uncond)`, chunk 0 = uncond) — NOT
//! the mlx crate's positive-first order. The InstantID glue batches the face tokens uncond-first too.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor, D};
use tokenizers::Tokenizer;

use candle_gen::gen_core::WeightsSource;
use candle_gen::quant::QLinear;
use candle_gen::{CandleError, Result};

// The vendored, packed-detecting CLIP tower + its config (sc-9527): a pure superset of the stock
// `stable_diffusion::clip` — a dense snapshot loads exactly as before, a packed MLX tier loads its
// dual CLIP straight from the packed parts.
use crate::clip::{self, ClipTextTransformer};
// sc-20528: the shared CLIP long-prompt chunker (A1111/compel windows), so this conditioner and the
// txt2img pipeline handle an over-long prompt identically instead of both hard-erroring.
use crate::long_prompt::ChunkPlan;
use crate::pipeline::{resolve_tokenizer_file, snapshot_file, Clip};

/// The EOS position of a CLIP token row = the arg-max token id (EOS = `<|endoftext|>` = 49407 is the
/// highest id, and SDXL pads with `"!"` so there is exactly one EOS). diffusers pools the bigG hidden
/// state here for the `text_embeds`. Returns 0 for an empty row (degenerate; never happens for a real
/// 77-token row that always carries BOS+EOS).
fn eos_position(ids: &[u32]) -> usize {
    ids.iter()
        .enumerate()
        .max_by_key(|(_, &v)| v)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Pool the bigG final-layer-norm hidden `final_g` `[B, 77, 1280]` at each row's EOS position and run it
/// through `text_projection` → pooled text-embeds `[B, 1280]`. `token_rows[b]` is row `b`'s padded ids
/// (for the EOS lookup). Split out so the gather/projection is testable with synthetic tensors.
fn pool_eos(
    final_g: &Tensor,
    token_rows: &[Vec<u32>],
    text_projection: &QLinear,
) -> Result<Tensor> {
    let mut rows = Vec::with_capacity(token_rows.len());
    for (b, ids) in token_rows.iter().enumerate() {
        rows.push(final_g.i((b, eos_position(ids)))?); // [1280]
    }
    let eos_hidden = Tensor::stack(&rows, 0)?; // [B, 1280]
    Ok(text_projection.forward(&eos_hidden)?)
}

/// Encode one tower for an already aligned batch of text rows. Only bigG owns the pooled head.
/// Keeping this per tower lets the staged fused path release CLIP-L before loading bigG.
fn encode_clip_windows(
    tower: &ClipTextTransformer,
    projection: Option<&QLinear>,
    tokenizer: &Tokenizer,
    plan: &ChunkPlan,
    texts: &[&str],
    chunks: usize,
    device: &Device,
) -> Result<(Tensor, Option<Tensor>)> {
    let mut hidden_rows = Vec::with_capacity(texts.len());
    let mut pooled_rows = Vec::with_capacity(texts.len());
    for text in texts {
        let rows = plan.rows_aligned(tokenizer, text, chunks)?;
        let mut hidden = Vec::with_capacity(rows.len());
        // Use batch 1 for each window. CFG rows are stacked only after the causal CLIP forward.
        for (index, row) in rows.iter().enumerate() {
            let ids = Tensor::new(row.as_slice(), device)?.reshape((1, plan.window()))?;
            let (final_hidden, penultimate) =
                tower.forward_until_encoder_layer(&ids, usize::MAX, -2)?;
            hidden.push(penultimate);
            if index == 0 {
                if let Some(projection) = projection {
                    pooled_rows.push(pool_eos(
                        &final_hidden,
                        std::slice::from_ref(row),
                        projection,
                    )?);
                }
            }
        }
        hidden_rows.push(if hidden.len() == 1 {
            hidden.remove(0)
        } else {
            Tensor::cat(&hidden, 1)?
        });
    }
    let pooled = projection
        .map(|_| Tensor::cat(&pooled_rows, 0))
        .transpose()?;
    Ok((Tensor::cat(&hidden_rows, 0)?, pooled))
}

/// One SDXL CLIP tower and its pooled head, when the tower is bigG.
pub(crate) struct ClipTower {
    tower: ClipTextTransformer,
    projection: Option<QLinear>,
    pub(crate) representation: &'static str,
}

impl ClipTower {
    pub(crate) fn load(
        vb: candle_nn::VarBuilder<'_>,
        which: Clip,
        group_size: usize,
        representation: &'static str,
    ) -> Result<Self> {
        let cfg = which.vendored_config();
        let tower = ClipTextTransformer::new_gs(vb.clone(), &cfg, group_size)?;
        let projection = if matches!(which, Clip::BigG) {
            Some(clip::text_projection(&vb, cfg.projection_dim, group_size)?)
        } else {
            None
        };
        Ok(Self {
            tower,
            projection,
            representation,
        })
    }

    pub(crate) fn load_snapshot(
        root: &Path,
        which: Clip,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let (_tok_repo, dense_sub) = which.sources();
        let (file, group_size, representation) =
            match crate::pipeline::detect_packed_clip(root, &which)? {
                Some((file, group_size)) => (file, group_size, "packed_or_imported"),
                None => (
                    snapshot_file(root, dense_sub)?,
                    candle_gen::quant::MLX_GROUP_SIZE,
                    "dense",
                ),
            };
        let vb = candle_gen::mmap_var_builder(&[file], dtype, device)?;
        Self::load(vb, which, group_size, representation)
    }

    pub(crate) fn quantize_onto(
        &mut self,
        quant: candle_gen::gen_core::Quant,
        device: &Device,
    ) -> Result<()> {
        self.tower.quantize_onto(quant, device)?;
        if let Some(projection) = &mut self.projection {
            projection.quantize_onto(quant, device)?;
        }
        Ok(())
    }

    pub(crate) fn encode(
        &self,
        tokenizer: &Tokenizer,
        plan: &ChunkPlan,
        texts: &[&str],
        chunks: usize,
        device: &Device,
    ) -> Result<(Tensor, Option<Tensor>)> {
        encode_clip_windows(
            &self.tower,
            self.projection.as_ref(),
            tokenizer,
            plan,
            texts,
            chunks,
            device,
        )
    }
}

pub(crate) fn combine_clip_embeddings(
    l: (Tensor, Option<Tensor>),
    g: (Tensor, Option<Tensor>),
) -> Result<(Tensor, Tensor)> {
    let pooled =
        g.1.ok_or_else(|| CandleError::Msg("SDXL bigG pooled embedding missing".into()))?;
    Ok((Tensor::cat(&[l.0, g.0], D::Minus1)?, pooled))
}

/// A loaded SDXL dual-CLIP conditioner with two tokenizers. Built once per model.
pub struct SdxlConditioner {
    tok_l: Tokenizer,
    tok_g: Tokenizer,
    clip_l: ClipTower,
    clip_g: ClipTower,
    cfg_l: clip::Config,
    cfg_g: clip::Config,
    device: Device,
}

impl SdxlConditioner {
    /// Load both CLIP encoders from the SDXL snapshot (`text_encoder/` = CLIP-L, `text_encoder_2/` =
    /// bigG), the bigG `text_projection`, and the two model-agnostic tokenizers from the caller-staged
    /// `tokenizer_clip_l` / `tokenizer_clip_bigg` components (epic 13657, sc-13663 — passed in, never
    /// self-fetched). `dtype` is the compute dtype (f16 for production). Both encoders and the
    /// projection head packed-detect through the vendored tower (sc-9527), so a packed MLX tier loads
    /// its dual CLIP straight from the packed parts; a dense snapshot is byte-identical.
    pub fn load(
        root: &Path,
        device: &Device,
        dtype: DType,
        tokenizer_clip_l: &WeightsSource,
        tokenizer_clip_bigg: &WeightsSource,
    ) -> Result<Self> {
        let cfg_l = clip::Config::sdxl();
        let cfg_g = clip::Config::sdxl2();

        let tok_l_file = resolve_tokenizer_file(tokenizer_clip_l);
        let tok_g_file = resolve_tokenizer_file(tokenizer_clip_bigg);
        let tok_l = candle_gen::gen_core::__model_input_reader!(
Tokenizer::from_file(&tok_l_file),
"crates/media/candle-gen/candle-gen-sdxl/src/conditioning.rs::load::Tokenizer::from_file@7400",
Tokenizer,
&tok_l_file,
|__model_input_path| Tokenizer::from_file(__model_input_path)
)
        .map_err(|e| {
            CandleError::Msg(format!(
                "sdxl: load CLIP-L tokenizer from {}: {e}",
                tok_l_file.display()
            ))
        })?;
        let tok_g = candle_gen::gen_core::__model_input_reader!(
Tokenizer::from_file(&tok_g_file),
"crates/media/candle-gen/candle-gen-sdxl/src/conditioning.rs::load::Tokenizer::from_file@7631",
Tokenizer,
&tok_g_file,
|__model_input_path| Tokenizer::from_file(__model_input_path)
)
        .map_err(|e| {
            CandleError::Msg(format!(
                "sdxl: load CLIP-bigG tokenizer from {}: {e}",
                tok_g_file.display()
            ))
        })?;

        let clip_l = ClipTower::load_snapshot(root, Clip::L, device, dtype)?;
        let clip_g = ClipTower::load_snapshot(root, Clip::BigG, device, dtype)?;

        Ok(Self {
            tok_l,
            tok_g,
            clip_l,
            clip_g,
            cfg_l,
            cfg_g,
            device: device.clone(),
        })
    }

    /// The two encoders' [`ChunkPlan`]s (`pad_with` + `max_position_embeddings`), built per encode
    /// call — no weights, no tensors.
    fn chunk_plans(&self) -> Result<(ChunkPlan, ChunkPlan)> {
        Ok((
            ChunkPlan::new(
                &self.tok_l,
                self.cfg_l.pad_with.as_deref(),
                self.cfg_l.max_position_embeddings,
            )?,
            ChunkPlan::new(
                &self.tok_g,
                self.cfg_g.pad_with.as_deref(),
                self.cfg_g.max_position_embeddings,
            )?,
        ))
    }

    /// Encode `prompt` (+ `negative` under CFG) into the SDXL conditioning `[B, n·77, 2048]` and pooled
    /// text-embeds `[B, 1280]`. With CFG the batch is **uncond-first** (`[negative, prompt]`), matching
    /// [`crate::denoise`]. Without CFG (`cfg_on = false`) a single cond row.
    ///
    /// `n` is the number of CLIP windows the longest text in the request needs (sc-20528): `1` for
    /// every prompt that fits CLIP's 77-token context — the byte-identical pre-sc-20528 encoding —
    /// and `ceil(content / 75)` for a longer one, whose windows are concatenated on the sequence axis.
    /// The pooled text-embeds always come from the **first** window, which is the only one whose EOS
    /// is diffusers' pooled position.
    pub fn encode(&self, prompt: &str, negative: &str, cfg_on: bool) -> Result<(Tensor, Tensor)> {
        // Token rows, uncond-first under CFG.
        let texts: Vec<&str> = if cfg_on {
            vec![negative, prompt]
        } else {
            vec![prompt]
        };

        // One window count for BOTH encoders and BOTH CFG rows: the encoders are concatenated on the
        // feature axis and the rows stacked on the batch axis, so every encoding must share a sequence
        // length. A short negative is topped up with empty `BOS EOS pad…` windows.
        let (plan_l, plan_g) = self.chunk_plans()?;
        let chunks = crate::long_prompt::common_chunks(
            &[(&plan_l, &self.tok_l), (&plan_g, &self.tok_g)],
            &texts,
        )?;

        let l = self
            .clip_l
            .encode(&self.tok_l, &plan_l, &texts, chunks, &self.device)?;
        let g = self
            .clip_g
            .encode(&self.tok_g, &plan_g, &texts, chunks, &self.device)?;
        combine_clip_embeddings(l, g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// `eos_position` = the arg-max id (EOS is the highest CLIP id). Finds the EOS even with padding
    /// after it, and the BOS (id 49406) before it.
    #[test]
    fn eos_position_is_argmax() {
        // [BOS=49406, tok, tok, EOS=49407, pad=256, pad=256] → EOS at index 3.
        let row = [49406u32, 320, 540, 49407, 256, 256];
        assert_eq!(eos_position(&row), 3);
        // A row whose EOS is last.
        assert_eq!(eos_position(&[49406, 1, 2, 3, 49407]), 4);
    }

    /// `pool_eos` gathers the per-row EOS hidden and projects it — shape `[B, out]`, and it actually
    /// selects the EOS row (not row 0). Synthetic `final_g` with a distinctive EOS-position vector.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn pool_eos_gathers_eos_and_projects() {
        let dev = Device::Cpu;
        // B=1, S=4, H=3. EOS at index 2 (token ids put the max there).
        let token_rows = vec![vec![49406u32, 10, 49407, 256]];
        // final_g: rows 0..4 are [r,r,r]; the EOS row (index 2) is [2,2,2].
        let data: Vec<f32> = (0..4).flat_map(|r| vec![r as f32; 3]).collect();
        let final_g = Tensor::from_vec(data, (1, 4, 3), &dev).unwrap();
        // Identity projection (3→3) so the output is exactly the gathered EOS hidden — wrapped as a
        // dense `QLinear` (the sc-9527 packed-detecting projection type; here the dense arm).
        let eye =
            Tensor::from_vec(vec![1f32, 0., 0., 0., 1., 0., 0., 0., 1.], (3, 3), &dev).unwrap();
        let proj = QLinear::from_dense(candle_gen::quant::DenseLinear::Linear(
            candle_nn::Linear::new(eye, None),
        ));
        let pooled = pool_eos(&final_g, &token_rows, &proj).unwrap();
        assert_eq!(pooled.dims(), &[1, 3]);
        let v = pooled.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![2.0, 2.0, 2.0]); // the EOS-position (index 2) hidden, projected by identity

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED conditioning::tests::pool_eos_gathers_eos_and_projects");
    }
}
