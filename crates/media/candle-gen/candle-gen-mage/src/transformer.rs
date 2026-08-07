//! Mage dual-stream NR-MMDiT.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::gen_core::{PrecisionFloorComponent, Quant};
use candle_gen::quant::{PackedWeightSidecars, QLinear};
use candle_gen_boogu::loader::Weights;

use crate::config::{HEAD_DIM, NORM_EPS};
use crate::quant::{linear, move_onto, quantize_onto, tensor_onto};
use crate::rope::{self, PackLayout, RopeTable};

fn streamed_linear(
    weights: &Weights,
    base: &str,
    bias: bool,
    sidecars: Option<&PackedWeightSidecars>,
) -> Result<QLinear> {
    let Some(sidecars) = sidecars.filter(|_| weights.contains(&format!("{base}.scales"))) else {
        return linear(weights, base, bias);
    };
    if !sidecars.contains(base) {
        candle_core::bail!(
            "mage streamed packed projection `{base}` has no prepared device-format sidecar"
        );
    }
    let dense_bias = if bias {
        Some(weights.get(&format!("{base}.bias"))?)
    } else {
        None
    };
    Ok(QLinear::from_qtensor_dequant(
        Arc::new(sidecars.load(base, weights.device())?),
        dense_bias,
    ))
}

fn rms(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let norm = x
        .to_dtype(DType::F32)?
        .sqr()?
        .mean_keepdim(D::Minus1)?
        .affine(1., NORM_EPS)?
        .sqrt()?;
    x.to_dtype(DType::F32)?
        .broadcast_div(&norm)?
        .broadcast_mul(&weight.to_dtype(DType::F32)?)?
        .to_dtype(x.dtype())
}

fn layer_norm(x: &Tensor) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let centered = xf.broadcast_sub(&mean)?;
    let denom = centered
        .sqr()?
        .mean_keepdim(D::Minus1)?
        .affine(1., NORM_EPS)?
        .sqrt()?;
    centered.broadcast_div(&denom)?.to_dtype(x.dtype())
}

fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&(scale + 1.)?)?.broadcast_add(shift)
}

struct TimestepEmbedder {
    l1: QLinear,
    l2: QLinear,
}

impl TimestepEmbedder {
    fn load(w: &Weights) -> Result<Self> {
        Ok(Self {
            l1: linear(w, "time_text_embed.timestep_embedder.linear_1", true)?,
            l2: linear(w, "time_text_embed.timestep_embedder.linear_2", true)?,
        })
    }

    fn forward(&self, sigma: &Tensor, dtype: DType) -> Result<Tensor> {
        let half = 128usize;
        let mut freqs: Vec<f32> = (0..half)
            .map(|i| (-10_000f32.ln() * i as f32 / half as f32).exp())
            .collect();
        if dtype == DType::BF16 {
            freqs = Tensor::from_vec(freqs, half, sigma.device())?
                .to_dtype(DType::BF16)?
                .to_dtype(DType::F32)?
                .to_vec1()?;
        }
        let f = Tensor::from_vec(freqs, (1, half), sigma.device())?;
        let t = sigma.to_dtype(DType::F32)?.unsqueeze(1)?;
        let a = t.broadcast_mul(&f)?.affine(1000., 0.)?;
        let emb = Tensor::cat(&[a.cos()?, a.sin()?], 1)?.to_dtype(dtype)?;
        self.l2.forward(&self.l1.forward(&emb)?.silu()?)
    }

    fn place(&mut self, quant: Option<Quant>, device: &Device) -> Result<()> {
        place_linear(&mut self.l1, quant, device)?;
        place_linear(&mut self.l2, quant, device)
    }

    fn quantized_count(&self) -> usize {
        usize::from(self.l1.is_quantized()) + usize::from(self.l2.is_quantized())
    }
}

struct FeedForward {
    proj: QLinear,
    out: QLinear,
}

impl FeedForward {
    fn load_with_sidecars(
        w: &Weights,
        prefix: &str,
        sidecars: Option<&PackedWeightSidecars>,
    ) -> Result<Self> {
        Ok(Self {
            proj: streamed_linear(w, &format!("{prefix}.net.0.proj"), true, sidecars)?,
            out: streamed_linear(w, &format!("{prefix}.net.2"), true, sidecars)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Candle's `gelu` is the tanh approximation used by diffusers' gelu-approximate.
        self.out.forward(&self.proj.forward(x)?.gelu()?)
    }

    fn place(&mut self, quant: Option<Quant>, device: &Device) -> Result<()> {
        place_linear(&mut self.proj, quant, device)?;
        place_linear(&mut self.out, quant, device)
    }

    fn quantized_count(&self) -> usize {
        usize::from(self.proj.is_quantized()) + usize::from(self.out.is_quantized())
    }
}

struct JointAttention {
    to_q: QLinear,
    to_k: QLinear,
    to_v: QLinear,
    to_out: QLinear,
    add_q: QLinear,
    add_k: QLinear,
    add_v: QLinear,
    add_out: QLinear,
    norm_q: Tensor,
    norm_k: Tensor,
    norm_add_q: Tensor,
    norm_add_k: Tensor,
    heads: usize,
}

impl JointAttention {
    fn load_with_sidecars(
        w: &Weights,
        prefix: &str,
        heads: usize,
        sidecars: Option<&PackedWeightSidecars>,
    ) -> Result<Self> {
        let l = |name: &str| streamed_linear(w, &format!("{prefix}.{name}"), true, sidecars);
        Ok(Self {
            to_q: l("to_q")?,
            to_k: l("to_k")?,
            to_v: l("to_v")?,
            to_out: l("to_out.0")?,
            add_q: l("add_q_proj")?,
            add_k: l("add_k_proj")?,
            add_v: l("add_v_proj")?,
            add_out: l("to_add_out")?,
            norm_q: w.get(&format!("{prefix}.norm_q.weight"))?,
            norm_k: w.get(&format!("{prefix}.norm_k.weight"))?,
            norm_add_q: w.get(&format!("{prefix}.norm_added_q.weight"))?,
            norm_add_k: w.get(&format!("{prefix}.norm_added_k.weight"))?,
            heads,
        })
    }

    fn forward_with_memory(
        &self,
        image: &Tensor,
        text: &Tensor,
        table: &RopeTable,
        layout: &PackLayout,
        attention_budget: usize,
        cancel: &candle_gen::gen_core::CancelFlag,
    ) -> Result<(Tensor, Tensor)> {
        let (_, ni, _) = image.dims3()?;
        let (_, nt, _) = text.dims3()?;
        let shape_i = (ni, self.heads, HEAD_DIM);
        let shape_t = (nt, self.heads, HEAD_DIM);
        let iq = rope::apply(
            &rms(&self.to_q.forward(image)?.reshape(shape_i)?, &self.norm_q)?,
            table,
        )?;
        let ik = rope::apply(
            &rms(&self.to_k.forward(image)?.reshape(shape_i)?, &self.norm_k)?,
            table,
        )?;
        let iv = self.to_v.forward(image)?.reshape(shape_i)?;
        let tq = rms(
            &self.add_q.forward(text)?.reshape(shape_t)?,
            &self.norm_add_q,
        )?;
        let tk = rms(
            &self.add_k.forward(text)?.reshape(shape_t)?,
            &self.norm_add_k,
        )?;
        let tv = self.add_v.forward(text)?.reshape(shape_t)?;

        let image_cu = layout.image_cu();
        let text_cu = layout.text_cu();
        let mut image_parts = Vec::with_capacity(layout.segments());
        let mut text_parts = Vec::with_capacity(layout.segments());
        for s in 0..layout.segments() {
            if cancel.is_cancelled() {
                candle_core::bail!("mage canceled");
            }
            let il = image_cu[s + 1] - image_cu[s];
            let tl = text_cu[s + 1] - text_cu[s];
            let joint = |t: &Tensor, i: &Tensor| -> Result<Tensor> {
                Tensor::cat(
                    &[t.narrow(0, text_cu[s], tl)?, i.narrow(0, image_cu[s], il)?],
                    0,
                )?
                .transpose(0, 1)?
                .unsqueeze(0)
            };
            let q = joint(&tq, &iq)?;
            let k = joint(&tk, &ik)?;
            let v = joint(&tv, &iv)?;
            let plan = candle_gen::gen_core::attention_budget::AttentionPlan::budgeted(
                candle_gen::gen_core::attention_budget::AttentionBudget::from_score_elements(
                    attention_budget as u64,
                    false,
                ),
            )
            .with_cancel(cancel);
            let o = candle_gen::sdpa_planned_bhsd(
                &q,
                &k,
                &v,
                (HEAD_DIM as f64).powf(-0.5),
                None,
                candle_nn::ops::softmax_last_dim,
                plan,
            )
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?
            .squeeze(0)?
            .transpose(0, 1)?
            .reshape((tl + il, self.heads * HEAD_DIM))?;
            text_parts.push(o.narrow(0, 0, tl)?);
            image_parts.push(o.narrow(0, tl, il)?);
        }
        let image = Tensor::cat(&image_parts.iter().collect::<Vec<_>>(), 0)?.unsqueeze(0)?;
        let text = Tensor::cat(&text_parts.iter().collect::<Vec<_>>(), 0)?.unsqueeze(0)?;
        Ok((self.to_out.forward(&image)?, self.add_out.forward(&text)?))
    }

    fn place(&mut self, quant: Option<Quant>, device: &Device) -> Result<()> {
        for linear in [
            &mut self.to_q,
            &mut self.to_k,
            &mut self.to_v,
            &mut self.to_out,
            &mut self.add_q,
            &mut self.add_k,
            &mut self.add_v,
            &mut self.add_out,
        ] {
            place_linear(linear, quant, device)?;
        }
        for tensor in [
            &mut self.norm_q,
            &mut self.norm_k,
            &mut self.norm_add_q,
            &mut self.norm_add_k,
        ] {
            tensor_onto(tensor, device)?;
        }
        Ok(())
    }

    fn quantized_count(&self) -> usize {
        [
            &self.to_q,
            &self.to_k,
            &self.to_v,
            &self.to_out,
            &self.add_q,
            &self.add_k,
            &self.add_v,
            &self.add_out,
        ]
        .into_iter()
        .filter(|linear| linear.is_quantized())
        .count()
    }
}

struct Block {
    image_mod: QLinear,
    text_mod: QLinear,
    attention: JointAttention,
    image_ff: FeedForward,
    text_ff: FeedForward,
}

enum MageBlocks {
    Resident(Vec<Block>),
    Streamed {
        dir: PathBuf,
        depth: usize,
        heads: usize,
        sidecars: Option<Arc<PackedWeightSidecars>>,
        target_device: Device,
    },
}

#[cfg(feature = "testkit")]
pub mod block_window_probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static MATERIALIZED_WINDOWS: AtomicUsize = AtomicUsize::new(0);

    pub fn reset() {
        MATERIALIZED_WINDOWS.store(0, Ordering::Relaxed);
    }

    pub fn materialized_windows() -> usize {
        MATERIALIZED_WINDOWS.load(Ordering::Relaxed)
    }

    pub(super) fn record() {
        MATERIALIZED_WINDOWS.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(feature = "testkit")]
fn record_materialized_window() {
    block_window_probe::record();
}

#[cfg(not(feature = "testkit"))]
fn record_materialized_window() {}

struct Mods {
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
}

impl Block {
    fn load(w: &Weights, prefix: &str, heads: usize) -> Result<Self> {
        Self::load_with_sidecars(w, prefix, heads, None)
    }

    fn load_with_sidecars(
        w: &Weights,
        prefix: &str,
        heads: usize,
        sidecars: Option<&PackedWeightSidecars>,
    ) -> Result<Self> {
        Ok(Self {
            image_mod: streamed_linear(w, &format!("{prefix}.img_mod.1"), true, sidecars)?,
            text_mod: streamed_linear(w, &format!("{prefix}.txt_mod.1"), true, sidecars)?,
            attention: JointAttention::load_with_sidecars(
                w,
                &format!("{prefix}.attn"),
                heads,
                sidecars,
            )?,
            image_ff: FeedForward::load_with_sidecars(w, &format!("{prefix}.img_mlp"), sidecars)?,
            text_ff: FeedForward::load_with_sidecars(w, &format!("{prefix}.txt_mlp"), sidecars)?,
        })
    }

    fn mods(linear: &QLinear, temb: &Tensor, ids: &Tensor, tokens: usize) -> Result<(Mods, Mods)> {
        let p = linear.forward(&temb.silu()?)?;
        let dim = p.dim(1)? / 6;
        let one = |offset: usize| -> Result<Mods> {
            let expand = |part: usize| -> Result<Tensor> {
                p.narrow(1, offset + part * dim, dim)?
                    .contiguous()?
                    .index_select(ids, 0)?
                    .reshape((1, tokens, dim))
            };
            Ok(Mods {
                shift: expand(0)?,
                scale: expand(1)?,
                gate: expand(2)?,
            })
        };
        Ok((one(0)?, one(3 * dim)?))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_memory(
        &self,
        image: Tensor,
        text: Tensor,
        temb: &Tensor,
        table: &RopeTable,
        layout: &PackLayout,
        attention_budget: usize,
        cancel: &candle_gen::gen_core::CancelFlag,
    ) -> Result<(Tensor, Tensor)> {
        let image_ids = layout.image_segment_ids(image.device())?;
        let text_ids: Vec<u32> = layout
            .text_lens()
            .iter()
            .enumerate()
            .flat_map(|(i, n)| std::iter::repeat_n(i as u32, *n))
            .collect();
        let text_ids = Tensor::from_vec(text_ids, layout.text_tokens(), text.device())?;
        let (im1, im2) = Self::mods(&self.image_mod, temb, &image_ids, layout.image_tokens())?;
        let (tx1, tx2) = Self::mods(&self.text_mod, temb, &text_ids, layout.text_tokens())?;
        let imn = modulate(&layer_norm(&image)?, &im1.shift, &im1.scale)?;
        let txn = modulate(&layer_norm(&text)?, &tx1.shift, &tx1.scale)?;
        let (ia, ta) = self.attention.forward_with_memory(
            &imn,
            &txn,
            table,
            layout,
            attention_budget,
            cancel,
        )?;
        let image = (&image + ia.broadcast_mul(&im1.gate)?)?;
        let text = (&text + ta.broadcast_mul(&tx1.gate)?)?;
        let iff =
            self.image_ff
                .forward(&modulate(&layer_norm(&image)?, &im2.shift, &im2.scale)?)?;
        let tff = self
            .text_ff
            .forward(&modulate(&layer_norm(&text)?, &tx2.shift, &tx2.scale)?)?;
        Ok((
            (&image + iff.broadcast_mul(&im2.gate)?)?,
            (&text + tff.broadcast_mul(&tx2.gate)?)?,
        ))
    }

    fn place(&mut self, quant: Option<Quant>, device: &Device) -> Result<()> {
        place_linear(&mut self.image_mod, quant, device)?;
        place_linear(&mut self.text_mod, quant, device)?;
        self.attention.place(quant, device)?;
        self.image_ff.place(quant, device)?;
        self.text_ff.place(quant, device)
    }

    fn quantized_count(&self) -> usize {
        usize::from(self.image_mod.is_quantized())
            + usize::from(self.text_mod.is_quantized())
            + self.attention.quantized_count()
            + self.image_ff.quantized_count()
            + self.text_ff.quantized_count()
    }
}

pub struct MageTransformer {
    image_in: QLinear,
    text_norm: Tensor,
    text_in: QLinear,
    timestep: TimestepEmbedder,
    blocks: MageBlocks,
    final_mod: QLinear,
    output: QLinear,
    dtype: DType,
}

impl MageTransformer {
    pub fn load(dir: &Path, cfg: &crate::config::MageConfig, device: &Device) -> Result<Self> {
        Self::load_with_quant(dir, cfg, None, device)
    }

    pub fn load_with_quant(
        dir: &Path,
        cfg: &crate::config::MageConfig,
        quant: Option<Quant>,
        device: &Device,
    ) -> Result<Self> {
        let staging = if quant.is_some() {
            Device::Cpu
        } else {
            device.clone()
        };
        let mut weights = Weights::from_dir(dir, &staging, DType::BF16)?;
        // Dense snapshots stage on CPU and fold each projection directly onto the target. Physical
        // q4/q8 tiers are already packed: reopen their mmap on the target so `from_packed_gs` builds
        // each resident GGUF tensor there rather than leaving an idempotent packed projection on CPU.
        if quant.is_some() && weights.packed().is_some() {
            weights = Weights::from_dir(dir, device, DType::BF16)?;
        }
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(Block::load(
                &weights,
                &format!("transformer_blocks.{i}"),
                cfg.num_heads,
            )?);
        }
        let mut transformer = Self {
            image_in: linear(&weights, "img_in", true)?,
            text_norm: weights.get("txt_norm.weight")?,
            text_in: linear(&weights, "txt_in", true)?,
            timestep: TimestepEmbedder::load(&weights)?,
            blocks: MageBlocks::Resident(blocks),
            final_mod: linear(&weights, "norm_out.linear", true)?,
            output: linear(&weights, "proj_out", true)?,
            dtype: DType::BF16,
        };
        if quant.is_some() {
            transformer.place(quant, device)?;
            let count = transformer.quantized_linear_count();
            if count != 174 {
                candle_core::bail!(
                    "mage transformer quantized {count}/174 required live projections"
                );
            }
        }
        Ok(transformer)
    }

    /// Load the non-block shell once and retain only the host-backed transformer directory for the
    /// shared block-window driver. Each window reopens a fresh mmap view and materializes exactly its
    /// block range on the target device.
    pub fn load_block_streamed(
        dir: &Path,
        cfg: &crate::config::MageConfig,
        quant: Option<Quant>,
        target_device: &Device,
        cancel: &candle_gen::gen_core::CancelFlag,
    ) -> Result<Self> {
        let mut weights = Weights::from_dir(dir, &Device::Cpu, DType::BF16)?;
        let sidecars = if let Some(packed) = weights.packed() {
            let files = candle_gen::sorted_safetensors(dir, "mage")
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            let (_, sidecars) = PackedWeightSidecars::open_and_prepare_prefix_cancelable(
                &files,
                dir,
                packed,
                target_device,
                cancel,
                "transformer_blocks.",
            )?;
            Some(Arc::new(sidecars))
        } else {
            None
        };
        if quant.is_some() && sidecars.is_none() {
            candle_core::bail!(
                "mage dense q4/q8 transformer snapshots do not provide prepared device-format block sidecars"
            );
        }
        if quant.is_some() && sidecars.is_some() {
            weights = Weights::from_dir(dir, target_device, DType::BF16)?;
        }
        let mut transformer = Self {
            image_in: linear(&weights, "img_in", true)?,
            text_norm: weights.get("txt_norm.weight")?,
            text_in: linear(&weights, "txt_in", true)?,
            timestep: TimestepEmbedder::load(&weights)?,
            blocks: MageBlocks::Streamed {
                dir: dir.to_path_buf(),
                depth: cfg.depth,
                heads: cfg.num_heads,
                sidecars,
                target_device: target_device.clone(),
            },
            final_mod: linear(&weights, "norm_out.linear", true)?,
            output: linear(&weights, "proj_out", true)?,
            dtype: DType::BF16,
        };
        transformer.place_non_blocks(quant, target_device)?;
        Ok(transformer)
    }

    fn place(&mut self, quant: Option<Quant>, device: &Device) -> Result<()> {
        self.place_non_blocks(quant, device)?;
        let MageBlocks::Resident(blocks) = &mut self.blocks else {
            return Ok(());
        };
        for block in blocks {
            block.place(quant, device)?;
        }
        Ok(())
    }

    fn place_non_blocks(&mut self, quant: Option<Quant>, device: &Device) -> Result<()> {
        place_linear(&mut self.image_in, quant, device)?;
        tensor_onto(&mut self.text_norm, device)?;
        place_linear(&mut self.text_in, quant, device)?;
        self.timestep.place(quant, device)?;
        let final_mod_quant = quant.map(|selected| {
            crate::quant::component_quant(PrecisionFloorComponent::TransformerHead, selected)
        });
        place_linear(&mut self.final_mod, final_mod_quant, device)?;
        place_linear(&mut self.output, quant, device)
    }

    pub fn quantized_linear_count(&self) -> usize {
        usize::from(self.image_in.is_quantized())
            + usize::from(self.text_in.is_quantized())
            + self.timestep.quantized_count()
            + self
                .blocks
                .resident()
                .map(|blocks| blocks.iter().map(Block::quantized_count).sum::<usize>())
                .unwrap_or(0)
            + usize::from(self.final_mod.is_quantized())
            + usize::from(self.output.is_quantized())
    }

    /// Inputs are packed `[1, image_tokens, 128]`, `[1, text_tokens, 2560]`, and one sigma per
    /// attention segment. Output is packed flow velocity `[1, image_tokens, 128]`.
    pub fn forward(
        &self,
        image: &Tensor,
        text: &Tensor,
        sigma: &Tensor,
        layout: &PackLayout,
    ) -> Result<Tensor> {
        self.forward_with_memory(
            image,
            text,
            sigma,
            layout,
            candle_gen::ATTN_SCORES_BUDGET,
            usize::MAX,
            &candle_gen::gen_core::CancelFlag::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_memory(
        &self,
        image: &Tensor,
        text: &Tensor,
        sigma: &Tensor,
        layout: &PackLayout,
        attention_budget: usize,
        transformer_window: usize,
        cancel: &candle_gen::gen_core::CancelFlag,
    ) -> Result<Tensor> {
        let table = RopeTable::build(layout, self.dtype, image.device())?;
        let image = self.image_in.forward(&image.to_dtype(self.dtype)?)?;
        let text = self
            .text_in
            .forward(&rms(&text.to_dtype(self.dtype)?, &self.text_norm)?)?;
        let temb = self
            .timestep
            .forward(&sigma.to_dtype(self.dtype)?, self.dtype)?;
        let (next_image, next_text) = match &self.blocks {
            MageBlocks::Resident(blocks) => {
                let mut image = image;
                let mut text = text;
                for block in blocks {
                    if cancel.is_cancelled() {
                        candle_core::bail!("mage canceled");
                    }
                    (image, text) = block.forward_with_memory(
                        image,
                        text,
                        &temb,
                        &table,
                        layout,
                        attention_budget,
                        cancel,
                    )?;
                }
                (image, text)
            }
            MageBlocks::Streamed {
                dir,
                depth,
                heads,
                sidecars,
                target_device,
            } => {
                let plan =
                    candle_gen::block_window::BlockPlan::new(*depth, transformer_window.max(1))
                        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
                candle_gen::block_window::run_windowed(
                    target_device,
                    &plan,
                    cancel,
                    (image, text),
                    || Weights::from_dir(dir, target_device, DType::BF16).map_err(Into::into),
                    |(mut image, mut text), weights, range| {
                        record_materialized_window();
                        let blocks = range
                            .map(|index| {
                                Block::load_with_sidecars(
                                    weights,
                                    &format!("transformer_blocks.{index}"),
                                    *heads,
                                    sidecars.as_deref(),
                                )
                            })
                            .collect::<Result<Vec<_>>>()?;
                        for block in &blocks {
                            if cancel.is_cancelled() {
                                return Err(candle_gen::CandleError::Canceled);
                            }
                            (image, text) = block
                                .forward_with_memory(
                                    image,
                                    text,
                                    &temb,
                                    &table,
                                    layout,
                                    attention_budget,
                                    cancel,
                                )
                                .map_err(candle_gen::CandleError::from)?;
                        }
                        Ok((image, text))
                    },
                )
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?
            }
        };
        let image = next_image;
        let _text = next_text;
        let params = self.final_mod.forward(&temb.silu()?)?;
        let dim = params.dim(1)? / 2;
        let ids = layout.image_segment_ids(image.device())?;
        // Output head is scale,shift—the opposite of block shift,scale,gate.
        let scale = params
            .narrow(1, 0, dim)?
            .contiguous()?
            .index_select(&ids, 0)?
            .reshape((1, layout.image_tokens(), dim))?;
        let shift = params
            .narrow(1, dim, dim)?
            .contiguous()?
            .index_select(&ids, 0)?
            .reshape((1, layout.image_tokens(), dim))?;
        self.output
            .forward(&modulate(&layer_norm(&image)?, &shift, &scale)?)
    }
}

impl MageBlocks {
    fn resident(&self) -> Option<&[Block]> {
        match self {
            Self::Resident(blocks) => Some(blocks),
            Self::Streamed { .. } => None,
        }
    }
}

fn place_linear(linear: &mut QLinear, quant: Option<Quant>, device: &Device) -> Result<()> {
    match quant {
        Some(quant) => quantize_onto(linear, quant, device),
        None => move_onto(linear, device),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn final_head_is_scale_then_shift() {
        let d = Device::Cpu;
        let x = Tensor::new(&[[[0f32, 4.]]], &d).unwrap();
        let scale = Tensor::ones((1, 1, 2), DType::F32, &d).unwrap();
        let shift = Tensor::new(&[[[10f32, 20.]]], &d).unwrap();
        let got = modulate(&layer_norm(&x).unwrap(), &shift, &scale)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        assert!((got[0][0][0] - 8.).abs() < 1e-4);
        assert!((got[0][0][1] - 22.).abs() < 1e-4);
    }

    #[test]
    fn quantized_projection_count_tracks_the_full_architecture() {
        assert_eq!(6 + crate::config::DEPTH * 14, 174);
    }

    #[test]
    fn packed_streamed_projection_loads_prepared_device_format_sidecar() -> Result<()> {
        let device = Device::Cpu;
        let base = "transformer_blocks.0.attn.to_q";
        let dense = Tensor::randn(0f32, 1f32, (64usize, 128usize), &device)?;
        let (weight, scales, biases) = candle_gen::quant::pack_mlx_affine(&dense, 4, 64)?;
        let mut tensors = HashMap::new();
        tensors.insert(format!("{base}.weight"), weight);
        tensors.insert(format!("{base}.scales"), scales);
        tensors.insert(format!("{base}.biases"), biases);
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        candle_core::safetensors::save(&tensors, dir.join("model.safetensors"))?;
        std::fs::write(
            dir.join("config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )?;

        let weights = Weights::from_dir(&dir, &device, DType::BF16)?;
        let files = candle_gen::sorted_safetensors(&dir, "mage-test")
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        let (_source, sidecars) = PackedWeightSidecars::open_and_prepare_prefix_cancelable(
            &files,
            &dir,
            weights.packed().expect("packed fixture"),
            &device,
            &candle_gen::gen_core::CancelFlag::default(),
            "transformer_blocks.",
        )?;
        assert_eq!(sidecars.created_count(), 1);
        assert!(sidecars.contains(base));
        let direct = linear(&weights, base, false)?;
        let streamed = streamed_linear(&weights, base, false, Some(&sidecars))?;
        let input = Tensor::randn(0f32, 1f32, (3usize, 128usize), &device)?;
        assert_eq!(
            direct.forward(&input)?.to_vec2::<f32>()?,
            streamed.forward(&input)?.to_vec2::<f32>()?,
            "device-format sidecar changed the packed projection"
        );

        drop(streamed);
        drop(direct);
        drop(sidecars);
        drop(weights);
        Ok(())
    }
}
