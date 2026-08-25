//! FLUX.2 Klein **universal single-file import** (epic 11037, sc-21485): the `bfl` dialect's
//! canonical [`LogicalKeyMapping`] plus the planned weight source the transformer builds from.
//!
//! A community FLUX.2-klein transformer ships as ONE flat `.safetensors` in the original
//! (ComfyUI/BFL) key convention — `img_in`, `double_blocks.N.img_attn.qkv`, `final_layer.
//! adaLN_modulation.1` — with, on the NVFP4 export (`wikeeyang/Flux2-Klein-9B-True-V2`'s
//! `…-nvfp4mixed.safetensors`), the attention/MLP projections stored as `.comfy_quant`-described
//! NVFP4 (U8 nibbles + `weight_scale` E4M3 block scales + `weight_scale_2` per-tensor F32) and the
//! embedders / modulations / norms / final layer preserved dense BF16 by the producer.
//!
//! **No provider-local format parser** (epic E1): this module declares *keys and layout* only.
//! [`Flux2BflToDiffusersMapping`] renames the 1:1 tensors and declares the two fused layouts as
//! plan-time transforms (sc-21547) — the fused `qkv` `[3·d, d]` row-slices into `to_q`/`to_k`/
//! `to_v` (and `add_{q,k,v}_proj` for the txt stream), and the `final_layer.adaLN_modulation.1`
//! **half swap** (BFL packs `(shift, scale)`; diffusers/this crate read `(scale, shift)` —
//! load-bearing: a swapped half in the wrong slot modulates every output patch, mlx sc-2220).
//! The shared reader ([`candle_gen::logical_weights`]) then produces Dense or `PackedNvfp4`
//! logical projections; which rows are NVFP4, where their scales live, and whether a row
//! materializes packed or dense is decided entirely by the checkpoint's own descriptors, the codec
//! registry, and the residency policy — never by a provider descriptor, scale, or U8-shape
//! classifier.
//!
//! A **dev** BFL file cannot load through this mapping: dev's `guidance_in.*` and `.scale`
//! per-head-norm spellings map to `None`, and the plan compiler refuses any unmapped key by name
//! (epic E2 — exact identity, fail closed) rather than skipping it.

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::{Linear, RmsNorm};
use candle_gen::gen_core::checkpoint_codec::{
    LogicalKeyMapping, LogicalTransformDeclaration, LogicalTransformOutput,
};
use candle_gen::gen_core::checkpoint_facts::CheckpointWeightFacts;
use candle_gen::logical_weights::{LogicalTensor, LogicalWeightReader};
use candle_gen::quant::{ActPrecision, Nvfp4Context, Nvfp4Linear};

use crate::config::Flux2Config;
use crate::quant::QLinear;

/// Top-level (non-block) 1:1 renames: BFL physical → diffusers logical. The klein community
/// convention — no `guidance_in.*` (klein is CFG-free-distilled; a file that carries one refuses
/// as unmapped, which is how a dev checkpoint stays out of the klein provider).
const TOP_RENAMES: &[(&str, &str)] = &[
    ("img_in.weight", "x_embedder.weight"),
    ("txt_in.weight", "context_embedder.weight"),
    (
        "time_in.in_layer.weight",
        "time_guidance_embed.timestep_embedder.linear_1.weight",
    ),
    (
        "time_in.out_layer.weight",
        "time_guidance_embed.timestep_embedder.linear_2.weight",
    ),
    (
        "double_stream_modulation_img.lin.weight",
        "double_stream_modulation_img.linear.weight",
    ),
    (
        "double_stream_modulation_txt.lin.weight",
        "double_stream_modulation_txt.linear.weight",
    ),
    (
        "single_stream_modulation.lin.weight",
        "single_stream_modulation.linear.weight",
    ),
    ("final_layer.linear.weight", "proj_out.weight"),
];

/// The fused-AdaLN physical key and its diffusers logical key. Declared as a **half-swap
/// transform**, never a rename: BFL stores `(shift, scale)`, the diffusers `norm_out.linear` reads
/// `(scale, shift)`.
const ADALN_SOURCE: &str = "final_layer.adaLN_modulation.1.weight";
const ADALN_TARGET: &str = "norm_out.linear.weight";

/// Per-double-block 1:1 renames (physical suffix → diffusers suffix under
/// `transformer_blocks.{i}.`), excluding the fused `qkv` tensors declared as row-slice transforms.
/// Klein community per-head-norm spelling is `.weight` (the BFL-official dev export spells those
/// `.scale` and is NOT this dialect).
const DOUBLE_RENAMES: &[(&str, &str)] = &[
    ("img_attn.norm.query_norm.weight", "attn.norm_q.weight"),
    ("img_attn.norm.key_norm.weight", "attn.norm_k.weight"),
    ("img_attn.proj.weight", "attn.to_out.0.weight"),
    ("img_mlp.0.weight", "ff.linear_in.weight"),
    ("img_mlp.2.weight", "ff.linear_out.weight"),
    (
        "txt_attn.norm.query_norm.weight",
        "attn.norm_added_q.weight",
    ),
    ("txt_attn.norm.key_norm.weight", "attn.norm_added_k.weight"),
    ("txt_attn.proj.weight", "attn.to_add_out.weight"),
    ("txt_mlp.0.weight", "ff_context.linear_in.weight"),
    ("txt_mlp.2.weight", "ff_context.linear_out.weight"),
];

/// Fused-qkv physical suffix → the three diffusers row-slice target suffixes, per stream.
const DOUBLE_QKV: &[(&str, [&str; 3])] = &[
    (
        "img_attn.qkv.weight",
        ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"],
    ),
    (
        "txt_attn.qkv.weight",
        [
            "attn.add_q_proj.weight",
            "attn.add_k_proj.weight",
            "attn.add_v_proj.weight",
        ],
    ),
];

/// Per-single-block 1:1 renames (diffusers keeps the fused single block, so `linear1` is a rename
/// rather than a split).
const SINGLE_RENAMES: &[(&str, &str)] = &[
    ("linear1.weight", "attn.to_qkv_mlp_proj.weight"),
    ("linear2.weight", "attn.to_out.weight"),
    ("norm.query_norm.weight", "attn.norm_q.weight"),
    ("norm.key_norm.weight", "attn.norm_k.weight"),
];

/// Split `bare` as `{table_prefix}.{index}.{suffix}`, returning
/// `(index, suffix)`; `None` when it is not under `table_prefix` or has no well-formed index.
fn block_suffix<'k>(bare: &'k str, table_prefix: &str) -> Option<(usize, &'k str)> {
    let rest = bare.strip_prefix(table_prefix)?.strip_prefix('.')?;
    let dot = rest.find('.')?;
    let index: usize = rest[..dot].parse().ok()?;
    Some((index, &rest[dot + 1..]))
}

/// The `bfl` dialect's canonical key mapping: klein-community BFL keys → the diffusers logical
/// schema [`crate::transformer::Flux2Transformer`] reads, with the fused-QKV row slices and the
/// AdaLN half swap declared as plan-time transforms (sc-21547) and the architecture's true
/// geometry declared from the variant config.
///
/// # Bare keys only, deliberately (sc-21485 review, minor 5)
///
/// This mapping reads the **bare** BFL namespace, the shape the klein community single file
/// ships, and it is the only namespace the `bfl` dialect's registry signature
/// (`flux2-klein-bfl-v1`) claims. An earlier revision carried a detected `prefix` so a
/// `model.diffusion_model.`-nested export could route here too; that branch was unreachable in
/// production and is gone:
///
/// * a prefixed key set already satisfies `flux2-comfyui-v1`
///   (`model.diffusion_model.double_blocks.0.img_attn.qkv.weight`), so routing sends it to the
///   `comfyui` dialect and the 32B **dev** provider before this mapping is ever constructed;
/// * registering a prefixed `bfl` signature would make BOTH adapters claim the same file, so
///   making the branch reachable is not a local change to this type — it needs the `comfyui`
///   signature tightened first.
///
/// Carrying the parameter anyway implied a routing capability that does not exist. Today a
/// prefixed klein file routes to dev and refuses there; if that ever needs fixing, the fix belongs
/// in `FLUX2_CHECKPOINT_ADAPTER`'s signatures, and this mapping grows the namespace back with a
/// routing test alongside it. A prefixed key reaching *here* is unmapped at plan time — the
/// fail-closed answer.
#[derive(Clone, Copy, Debug)]
pub struct Flux2BflToDiffusersMapping<'a> {
    cfg: &'a Flux2Config,
}

impl<'a> Flux2BflToDiffusersMapping<'a> {
    /// The id the `FLUX2_CHECKPOINT_ADAPTER` registry row declares for the `bfl` dialect.
    pub const MAPPING_ID: &'static str = "flux2-bfl-to-diffusers-v1";

    pub const fn new(cfg: &'a Flux2Config) -> Self {
        Self { cfg }
    }

    fn inner(&self) -> usize {
        self.cfg.inner_dim()
    }

    fn ff_hidden(&self) -> usize {
        (self.cfg.mlp_ratio * self.inner() as f32) as usize
    }

    /// The architecture's expected `[out, in]` for a diffusers logical key, from the variant
    /// config. `None` for keys whose geometry the mapping does not pin (per-head norms are rank-1
    /// dense and carry their own shape).
    fn diffusers_logical_shape(&self, logical_key: &str) -> Option<Vec<usize>> {
        let inner = self.inner();
        let ff_hidden = self.ff_hidden();
        let mlp_hidden = self.cfg.single_mlp_hidden();
        match logical_key {
            "x_embedder.weight" => Some(vec![inner, self.cfg.in_channels]),
            "context_embedder.weight" => Some(vec![inner, self.cfg.joint_attention_dim]),
            "time_guidance_embed.timestep_embedder.linear_1.weight" => {
                Some(vec![inner, self.cfg.timestep_channels])
            }
            "time_guidance_embed.timestep_embedder.linear_2.weight" => Some(vec![inner, inner]),
            "double_stream_modulation_img.linear.weight"
            | "double_stream_modulation_txt.linear.weight" => Some(vec![6 * inner, inner]),
            "single_stream_modulation.linear.weight" => Some(vec![3 * inner, inner]),
            "proj_out.weight" => Some(vec![self.cfg.out_channels, inner]),
            _ => {
                if let Some((_, suffix)) = block_suffix(logical_key, "transformer_blocks") {
                    return match suffix {
                        "attn.to_out.0.weight" | "attn.to_add_out.weight" => {
                            Some(vec![inner, inner])
                        }
                        "ff.linear_in.weight" | "ff_context.linear_in.weight" => {
                            Some(vec![2 * ff_hidden, inner])
                        }
                        "ff.linear_out.weight" | "ff_context.linear_out.weight" => {
                            Some(vec![inner, ff_hidden])
                        }
                        _ => None,
                    };
                }
                if let Some((_, suffix)) = block_suffix(logical_key, "single_transformer_blocks") {
                    return match suffix {
                        "attn.to_qkv_mlp_proj.weight" => {
                            Some(vec![3 * inner + 2 * mlp_hidden, inner])
                        }
                        "attn.to_out.weight" => Some(vec![inner, inner + mlp_hidden]),
                        _ => None,
                    };
                }
                None
            }
        }
    }
}

impl LogicalKeyMapping for Flux2BflToDiffusersMapping<'_> {
    fn mapping_id(&self) -> &'static str {
        Self::MAPPING_ID
    }

    fn logical_key(&self, physical_key: &str) -> Option<String> {
        let bare = physical_key;
        for (src, dst) in TOP_RENAMES {
            if bare == *src {
                return Some((*dst).to_owned());
            }
        }
        if let Some((index, suffix)) = block_suffix(bare, "double_blocks") {
            for (src, dst) in DOUBLE_RENAMES {
                if suffix == *src {
                    return Some(format!("transformer_blocks.{index}.{dst}"));
                }
            }
            return None;
        }
        if let Some((index, suffix)) = block_suffix(bare, "single_blocks") {
            for (src, dst) in SINGLE_RENAMES {
                if suffix == *src {
                    return Some(format!("single_transformer_blocks.{index}.{dst}"));
                }
            }
            return None;
        }
        None
    }

    fn logical_shape(&self, logical_key: &str) -> Option<Vec<usize>> {
        self.diffusers_logical_shape(logical_key)
    }

    fn logical_transform(&self, physical_key: &str) -> Option<LogicalTransformDeclaration> {
        let bare = physical_key;
        let inner = self.inner();
        if bare == ADALN_SOURCE {
            // BFL packs `(shift, scale)`; diffusers reads `(scale, shift)`. This tensor is dense
            // BF16 on every known klein export, and the plan compiler refuses a half swap on a
            // packed residency by name rather than permuting packed rows.
            return Some(
                LogicalTransformDeclaration::new(vec![LogicalTransformOutput::half_swap(
                    ADALN_TARGET,
                )])
                .with_source_logical_shape(vec![2 * inner, inner]),
            );
        }
        let (index, suffix) = block_suffix(bare, "double_blocks")?;
        for (src, targets) in DOUBLE_QKV {
            if suffix == *src {
                let outputs = targets
                    .iter()
                    .enumerate()
                    .map(|(slot, target)| {
                        LogicalTransformOutput::row_slice(
                            format!("transformer_blocks.{index}.{target}"),
                            slot * inner,
                            inner,
                        )
                    })
                    .collect();
                return Some(
                    LogicalTransformDeclaration::new(outputs)
                        .with_source_logical_shape(vec![3 * inner, inner]),
                );
            }
        }
        None
    }
}

/// The planned single-file weight source the FLUX.2 transformer builds from (sc-21485): the shared
/// [`LogicalWeightReader`] over the compiled plan, the compute device/dtype, and the ONE shared
/// per-device [`Nvfp4Context`] every packed projection stages through (sc-12274 — never a private
/// 32 MiB cuBLASLt workspace per projection).
///
/// The provider owns **construction only**: a `Dense` logical tensor becomes a dense
/// [`QLinear`], a `PackedNvfp4` one becomes an [`Nvfp4Linear`] wrapped as an adapter-capable
/// [`QLinear`] base ([`QLinear::from_nvfp4`], sc-21483). Which of the two arrives is the plan's
/// decision, made by the codec registry + residency policy — the provider never re-classifies.
pub(crate) struct PlannedDitWeights {
    reader: LogicalWeightReader,
    device: Device,
    dtype: DType,
    ctx: Nvfp4Context,
}

impl PlannedDitWeights {
    pub(crate) fn new(
        reader: LogicalWeightReader,
        device: Device,
        dtype: DType,
        ctx: Nvfp4Context,
    ) -> Self {
        Self {
            reader,
            device,
            dtype,
            ctx,
        }
    }

    /// Whether the plan holds `logical_key` (the `contains_tensor` analog).
    pub(crate) fn contains(&self, logical_key: &str) -> bool {
        self.reader.planned(logical_key).is_some()
    }

    /// Materialize one planned logical tensor as a **dense** tensor at the compute dtype on the
    /// compute device (norm weights and other non-projection leaves). A packed plan entry here is
    /// a defect: no packed row is a norm/leaf in this architecture.
    pub(crate) fn dense(&self, logical_key: &str) -> Result<Tensor> {
        match self
            .reader
            .read(logical_key)
            .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))?
        {
            LogicalTensor::Dense(tensor) => tensor.to_dtype(self.dtype)?.to_device(&self.device),
            LogicalTensor::PackedNvfp4 { .. } | LogicalTensor::PackedFp8E4M3 { .. } => {
                Err(candle_gen::candle_core::Error::Msg(format!(
                    "flux2 single-file: `{logical_key}` was planned packed but is read as a dense \
                     leaf; the mapping and the model disagree about this tensor's role"
                )))
            }
        }
    }

    /// Build one projection from its planned logical `{base}.weight` (+ optional `{base}.bias`):
    /// dense rows become a dense [`QLinear`] at the compute dtype; `PackedNvfp4` rows become an
    /// [`Nvfp4Linear`] built against the shared context, with the mixed-precision default
    /// activation policy keyed on the diffusers dotted name.
    pub(crate) fn qlinear(
        &self,
        base: &str,
        in_dim: usize,
        out_dim: usize,
        bias: bool,
    ) -> Result<QLinear> {
        let weight_key = format!("{base}.weight");
        let bias_key = format!("{base}.bias");
        let dense_bias = if bias {
            Some(self.dense(&bias_key)?)
        } else {
            None
        };
        match self
            .reader
            .read(&weight_key)
            .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))?
        {
            LogicalTensor::Dense(weight) => {
                let weight = weight.to_dtype(self.dtype)?.to_device(&self.device)?;
                let (rows, cols) = weight.dims2()?;
                if (rows, cols) != (out_dim, in_dim) {
                    return Err(candle_gen::candle_core::Error::Msg(format!(
                        "flux2 single-file: `{weight_key}` decoded to [{rows}, {cols}], expected \
                         [{out_dim}, {in_dim}]"
                    )));
                }
                Ok(QLinear::from_dense(
                    Linear::new(weight, dense_bias),
                    in_dim,
                    out_dim,
                ))
            }
            LogicalTensor::PackedNvfp4 { tensor, .. } => {
                let (rows, cols) = (tensor.rows, tensor.cols);
                if (rows, cols) != (out_dim, in_dim) {
                    return Err(candle_gen::candle_core::Error::Msg(format!(
                        "flux2 single-file: packed `{weight_key}` is [{rows}, {cols}], expected \
                         [{out_dim}, {in_dim}]"
                    )));
                }
                let lin = Nvfp4Linear::from_packed_in(
                    *tensor,
                    dense_bias,
                    &self.device,
                    ActPrecision::for_outlier_layer(base),
                    &self.ctx,
                )?;
                Ok(QLinear::from_nvfp4(lin))
            }
            LogicalTensor::PackedFp8E4M3 { .. } => {
                Err(candle_gen::candle_core::Error::Msg(format!(
                    "flux2 single-file: `{weight_key}` was planned as packed fp8, which this \
                     loader does not construct; replan with a dense residency policy"
                )))
            }
        }
    }

    /// Materialize a rank-1 norm weight and wrap it as an [`RmsNorm`] at `eps`.
    pub(crate) fn rms_norm(&self, logical_key: &str, eps: f64) -> Result<RmsNorm> {
        Ok(RmsNorm::new(self.dense(logical_key)?, eps))
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    /// The three correlated source/capability/receipt facts for everything materialized so far
    /// (sc-21484) — the shared reader's surface, exposed unmodified.
    pub(crate) fn checkpoint_weight_facts(&self) -> candle_gen::Result<CheckpointWeightFacts> {
        self.reader.checkpoint_weight_facts()
    }
}

/// The semantic identity of a compiled plan, for the linked-versus-managed equality the story
/// pins: the mapping id plus, per `(codec, residency)` row, the logical keys it covers. Two copies
/// of one artifact must compile to exactly this same value regardless of which path they load
/// from.
pub fn plan_semantic_summary(
    plan: &candle_gen::gen_core::checkpoint_codec::LogicalWeightPlan,
) -> (String, Vec<(String, String, Vec<String>)>) {
    let mut by_row: HashMap<(String, String), Vec<String>> = HashMap::new();
    for tensor in &plan.tensors {
        by_row
            .entry((
                tensor.codec_id.to_owned(),
                format!("{:?}", tensor.residency.mode),
            ))
            .or_default()
            .push(tensor.logical_key.clone());
    }
    let mut rows: Vec<(String, String, Vec<String>)> = by_row
        .into_iter()
        .map(|((codec, mode), mut keys)| {
            keys.sort();
            (codec, mode, keys)
        })
        .collect();
    rows.sort();
    (plan.mapping_id.to_owned(), rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::Path;

    use candle_gen::candle_core::safetensors::MmapedSafetensors;
    use candle_gen::candle_nn::VarBuilder;
    use candle_gen::gen_core::checkpoint_codec::{
        CheckpointCodecRegistration, CodecResidencyPolicy, ResidencyMode, TensorCodecSpec,
        NVFP4_CODEC,
    };
    use candle_gen::logical_weights::{plan_logical_weights, CandleCodecResidency};
    use candle_gen::quant::Nvfp4Tensor;

    use crate::transformer::Flux2Transformer;

    /// A klein-shaped architecture at fixture width: 1 double + 1 single block, `inner = 16`.
    /// Only the MMDiT fields matter here; the TE fields are copied from klein and never read.
    fn tiny_cfg() -> Flux2Config {
        let mut cfg = crate::config::Flux2Variant::Klein9b.config();
        cfg.num_double_layers = 1;
        cfg.num_single_layers = 1;
        cfg.num_heads = 2;
        cfg.head_dim = 8;
        cfg.in_channels = 8;
        cfg.out_channels = 8;
        cfg.joint_attention_dim = 12;
        cfg.timestep_channels = 8;
        cfg.axes_dim = [2, 2, 2, 2];
        cfg
    }

    /// [`tiny_cfg`] at an NVFP4-legal width: `inner = 128` (rows of the fused qkv = 384, so each
    /// 128-row slice is exactly one scale-factor atom tile), `in_features` 32-aligned.
    fn nvfp4_cfg() -> Flux2Config {
        let mut cfg = tiny_cfg();
        cfg.num_heads = 16;
        cfg.head_dim = 8;
        cfg
    }

    fn mapping_for(cfg: &Flux2Config) -> Flux2BflToDiffusersMapping<'_> {
        Flux2BflToDiffusersMapping::new(cfg)
    }

    #[test]
    fn mapping_maps_the_klein_key_surface_and_refuses_foreign_keys() {
        let cfg = tiny_cfg();
        let mapping = mapping_for(&cfg);
        assert_eq!(mapping.mapping_id(), Flux2BflToDiffusersMapping::MAPPING_ID);
        assert_eq!(
            mapping.logical_key("img_in.weight").as_deref(),
            Some("x_embedder.weight")
        );
        assert_eq!(
            mapping
                .logical_key("double_blocks.3.img_attn.proj.weight")
                .as_deref(),
            Some("transformer_blocks.3.attn.to_out.0.weight")
        );
        assert_eq!(
            mapping
                .logical_key("single_blocks.11.linear1.weight")
                .as_deref(),
            Some("single_transformer_blocks.11.attn.to_qkv_mlp_proj.weight")
        );
        // Foreign keys refuse: dev's guidance embedder and the BFL-official `.scale` per-head-norm
        // spelling are NOT this dialect, so a dev checkpoint cannot slip through the klein route.
        assert_eq!(mapping.logical_key("guidance_in.in_layer.weight"), None);
        // sc-21485 review (minor 5): this dialect is the BARE namespace, the only one
        // `flux2-klein-bfl-v1` claims. A `model.diffusion_model.`-nested key is routed to the
        // `comfyui` dialect (the dev provider) long before this mapping is built, so there is no
        // prefix branch here — and a prefixed key that somehow reaches it is unmapped, which the
        // plan compiler turns into a named refusal rather than a silent skip.
        assert_eq!(
            mapping.logical_key("model.diffusion_model.img_in.weight"),
            None
        );
        assert_eq!(
            mapping.logical_transform("model.diffusion_model.double_blocks.0.img_attn.qkv.weight"),
            None
        );
        assert_eq!(
            mapping.logical_key("double_blocks.0.img_attn.norm.query_norm.scale"),
            None
        );
        // The fused tensors are transforms, not renames.
        assert_eq!(
            mapping.logical_key("double_blocks.0.img_attn.qkv.weight"),
            None
        );
        assert!(mapping
            .logical_transform("double_blocks.0.img_attn.qkv.weight")
            .is_some());
        assert!(mapping
            .logical_transform("final_layer.adaLN_modulation.1.weight")
            .is_some());
    }

    #[test]
    fn fused_qkv_and_adaln_are_declared_transforms() {
        let cfg = tiny_cfg();
        let inner = cfg.inner_dim();
        let mapping = mapping_for(&cfg);
        let qkv = mapping
            .logical_transform("double_blocks.0.txt_attn.qkv.weight")
            .expect("fused qkv declares a transform");
        assert_eq!(
            qkv.source_logical_shape.as_deref(),
            Some(&[3 * inner, inner][..])
        );
        let expected = [
            ("transformer_blocks.0.attn.add_q_proj.weight", 0),
            ("transformer_blocks.0.attn.add_k_proj.weight", inner),
            ("transformer_blocks.0.attn.add_v_proj.weight", 2 * inner),
        ];
        assert_eq!(qkv.outputs.len(), 3);
        for (output, (key, start)) in qkv.outputs.iter().zip(expected) {
            assert_eq!(output.logical_key, key);
            let rows = output.rows.expect("a row slice");
            assert_eq!((rows.start, rows.len), (start, inner));
            assert!(!output.half_swap);
        }
        // The three slices exactly partition the fused axis — the compiler enforces this too
        // (SliceOverlap / SliceGap), but the declaration itself must already be a partition.
        let covered: usize = qkv.outputs.iter().map(|o| o.rows.unwrap().len).sum();
        assert_eq!(covered, 3 * inner);

        let adaln = mapping
            .logical_transform("final_layer.adaLN_modulation.1.weight")
            .expect("adaLN declares a transform");
        assert_eq!(adaln.outputs.len(), 1);
        assert_eq!(adaln.outputs[0].logical_key, ADALN_TARGET);
        assert!(
            adaln.outputs[0].half_swap,
            "the shift/scale swap is the point"
        );
        assert_eq!(adaln.outputs[0].rows, None);
        assert_eq!(
            adaln.source_logical_shape.as_deref(),
            Some(&[2 * inner, inner][..])
        );
    }

    // ---- fixture writers -----------------------------------------------------------------------

    /// Deterministic pseudo-random f32s (LCG) so fixtures are stable without a rand dependency.
    fn fill(seed: u32, n: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(12345);
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 9) as f32 / (1u32 << 23) as f32) - 1.0
            })
            .collect()
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| {
                let bits = v.to_bits();
                let rounded = (bits.wrapping_add(0x8000 + ((bits >> 16) & 1))) >> 16;
                (rounded as u16).to_le_bytes()
            })
            .collect()
    }

    /// The complete tiny klein BFL key surface for `cfg`, as `(name, shape)`.
    fn tiny_bfl_surface(cfg: &Flux2Config) -> Vec<(String, Vec<usize>)> {
        let inner = cfg.inner_dim();
        let ff = (cfg.mlp_ratio * inner as f32) as usize;
        let mlp = cfg.single_mlp_hidden();
        let mut keys: Vec<(String, Vec<usize>)> = vec![
            ("img_in.weight".into(), vec![inner, cfg.in_channels]),
            ("txt_in.weight".into(), vec![inner, cfg.joint_attention_dim]),
            (
                "time_in.in_layer.weight".into(),
                vec![inner, cfg.timestep_channels],
            ),
            ("time_in.out_layer.weight".into(), vec![inner, inner]),
            (
                "double_stream_modulation_img.lin.weight".into(),
                vec![6 * inner, inner],
            ),
            (
                "double_stream_modulation_txt.lin.weight".into(),
                vec![6 * inner, inner],
            ),
            (
                "single_stream_modulation.lin.weight".into(),
                vec![3 * inner, inner],
            ),
            (
                "final_layer.linear.weight".into(),
                vec![cfg.out_channels, inner],
            ),
            (ADALN_SOURCE.into(), vec![2 * inner, inner]),
        ];
        for i in 0..cfg.num_double_layers {
            let p = format!("double_blocks.{i}");
            keys.push((format!("{p}.img_attn.qkv.weight"), vec![3 * inner, inner]));
            keys.push((format!("{p}.txt_attn.qkv.weight"), vec![3 * inner, inner]));
            keys.push((format!("{p}.img_attn.proj.weight"), vec![inner, inner]));
            keys.push((format!("{p}.txt_attn.proj.weight"), vec![inner, inner]));
            keys.push((format!("{p}.img_mlp.0.weight"), vec![2 * ff, inner]));
            keys.push((format!("{p}.img_mlp.2.weight"), vec![inner, ff]));
            keys.push((format!("{p}.txt_mlp.0.weight"), vec![2 * ff, inner]));
            keys.push((format!("{p}.txt_mlp.2.weight"), vec![inner, ff]));
            for norm in [
                "img_attn.norm.query_norm.weight",
                "img_attn.norm.key_norm.weight",
                "txt_attn.norm.query_norm.weight",
                "txt_attn.norm.key_norm.weight",
            ] {
                keys.push((format!("{p}.{norm}"), vec![cfg.head_dim]));
            }
        }
        for i in 0..cfg.num_single_layers {
            let p = format!("single_blocks.{i}");
            keys.push((
                format!("{p}.linear1.weight"),
                vec![3 * inner + 2 * mlp, inner],
            ));
            keys.push((format!("{p}.linear2.weight"), vec![inner, inner + mlp]));
            keys.push((format!("{p}.norm.query_norm.weight"), vec![cfg.head_dim]));
            keys.push((format!("{p}.norm.key_norm.weight"), vec![cfg.head_dim]));
        }
        keys
    }

    /// Write a complete tiny dense-bf16 klein BFL single file for `cfg`.
    fn write_dense_bfl_file(path: &Path, cfg: &Flux2Config) {
        let surface = tiny_bfl_surface(cfg);
        let payloads: Vec<(String, Vec<usize>, Vec<u8>)> = surface
            .into_iter()
            .enumerate()
            .map(|(index, (name, shape))| {
                let n: usize = shape.iter().product();
                let bytes = bf16_bytes(&fill(index as u32 + 1, n));
                (name, shape, bytes)
            })
            .collect();
        let tensors: BTreeMap<&str, ::safetensors::tensor::TensorView<'_>> = payloads
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.as_str(),
                    ::safetensors::tensor::TensorView::new(
                        ::safetensors::Dtype::BF16,
                        shape.clone(),
                        bytes,
                    )
                    .unwrap(),
                )
            })
            .collect();
        ::safetensors::serialize_to_file(tensors, None, path).unwrap();
    }

    // ---- dense round-trip: fused-QKV split + AdaLN half swap at VALUE level --------------------

    #[test]
    fn dense_plan_splits_qkv_and_swaps_adaln_halves_by_value() {
        let cfg = tiny_cfg();
        let inner = cfg.inner_dim();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("klein-tiny.safetensors");
        write_dense_bfl_file(&path, &cfg);

        let mapping = mapping_for(&cfg);
        let plan = plan_logical_weights(&path, &mapping, &CandleCodecResidency::DENSE)
            .expect("the tiny klein surface plans");
        let reader = LogicalWeightReader::open(&path, plan, &Device::Cpu).expect("reader opens");

        // SAFETY: read-only mmap of the fixture this test just wrote.
        let st = unsafe { MmapedSafetensors::new(&path) }.unwrap();
        let cpu = Device::Cpu;
        let fused = st
            .load("double_blocks.0.img_attn.qkv.weight", &cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        for (slot, key) in [
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_k.weight",
            "transformer_blocks.0.attn.to_v.weight",
        ]
        .iter()
        .enumerate()
        {
            let LogicalTensor::Dense(got) = reader.read(key).unwrap() else {
                panic!("dense residency must materialize dense");
            };
            let got = got.to_dtype(DType::F32).unwrap();
            let want = fused.narrow(0, slot * inner, inner).unwrap();
            let delta = (got - want)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert_eq!(delta, 0.0, "{key}: fused-QKV slice must be byte-exact");
        }

        // AdaLN at VALUE level: BFL stores (shift, scale); the logical tensor must hold
        // (scale, shift) — i.e. the halves EXCHANGED, not merely reshaped. A regression that
        // returns the unswapped rows lands the scale in the shift slot (the magenta-grid /
        // periodic-weave class) and this assertion goes red.
        let LogicalTensor::Dense(swapped) = reader.read(ADALN_TARGET).unwrap() else {
            panic!("adaLN plans dense");
        };
        let swapped = swapped.to_dtype(DType::F32).unwrap();
        let stored = st
            .load(ADALN_SOURCE, &cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let want = Tensor::cat(
            &[
                stored.narrow(0, inner, inner).unwrap(),
                stored.narrow(0, 0, inner).unwrap(),
            ],
            0,
        )
        .unwrap();
        let delta = (swapped.clone() - want)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(delta, 0.0, "adaLN halves must be exchanged exactly");
        // And the unswapped identity would NOT satisfy the check (the fixture halves differ).
        let unswapped_delta = (swapped - stored)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            unswapped_delta > 0.0,
            "fixture must distinguish swapped from unswapped"
        );
    }

    // ---- the planned path reproduces the trusted loader-native converter -----------------------

    /// End-to-end conformance: the same tiny dense BFL file built once through the **plan-driven**
    /// path (`Flux2Transformer::new_planned`) and once through the trusted loader-native converter
    /// (`convert::build_target_state_dict` → VarBuilder), then run through one identical forward.
    /// Outputs must agree exactly — the declarative transform plan is the same semantic remap.
    #[test]
    fn planned_transformer_matches_the_loader_native_converter() {
        let cfg = tiny_cfg();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("klein-tiny.safetensors");
        write_dense_bfl_file(&path, &cfg);
        let cpu = Device::Cpu;

        // Path A: plan-driven.
        let mapping = mapping_for(&cfg);
        let plan = plan_logical_weights(&path, &mapping, &CandleCodecResidency::DENSE).unwrap();
        let reader = LogicalWeightReader::open(&path, plan, &cpu).unwrap();
        let src = PlannedDitWeights::new(reader, cpu.clone(), DType::F32, Nvfp4Context::none());
        let planned = Flux2Transformer::new_planned(&cfg, &src).expect("planned transformer");

        // Path B: the trusted converter (klein community tables + chunk3 + swap_halves).
        // SAFETY: read-only mmap of the fixture this test just wrote.
        let st = unsafe { MmapedSafetensors::new(&path) }.unwrap();
        let map = crate::convert::build_target_state_dict(&st).unwrap();
        let map: HashMap<_, _> = map
            .into_iter()
            .map(|(name, tensor)| (name, tensor.to_dtype(DType::F32).unwrap()))
            .collect();
        let vb = VarBuilder::from_tensors(map, DType::F32, &cpu);
        let converted = Flux2Transformer::new(&cfg, vb).expect("converted transformer");

        let img = Tensor::from_vec(
            fill(901, 4 * cfg.in_channels),
            (1, 4, cfg.in_channels),
            &cpu,
        )
        .unwrap();
        let txt = Tensor::from_vec(
            fill(902, 3 * cfg.joint_attention_dim),
            (1, 3, cfg.joint_attention_dim),
            &cpu,
        )
        .unwrap();
        let img_ids: Vec<[i64; 4]> = (0..4).map(|i| [0, i / 2, i % 2, 0]).collect();
        let txt_ids: Vec<[i64; 4]> = (0..3).map(|_| [0, 0, 0, 0]).collect();
        let a = planned
            .forward(&img, &txt, &img_ids, &txt_ids, 500.0, None)
            .unwrap();
        let b = converted
            .forward(&img, &txt, &img_ids, &txt_ids, 500.0, None)
            .unwrap();
        let delta = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(
            delta, 0.0,
            "the plan-driven load must equal the loader-native converter exactly"
        );
    }

    // ---- NVFP4: packed split agrees with the dense decode; refusals stay closed ----------------

    /// Residency policy forcing NVFP4 rows packed on any host — the CPU-lane stand-in for a
    /// `sm_120` device, exactly the krea test pattern.
    struct ForcePackedNvfp4;

    impl CodecResidencyPolicy for ForcePackedNvfp4 {
        fn residency(
            &self,
            codec: &CheckpointCodecRegistration,
            spec: &TensorCodecSpec,
            stored_shape: &[usize],
        ) -> ResidencyMode {
            if codec.codec_id == NVFP4_CODEC.codec_id {
                return ResidencyMode::Packed;
            }
            CandleCodecResidency::DENSE.residency(codec, spec, stored_shape)
        }
    }

    /// Write a single-tensor NVFP4 klein fixture: the fused img qkv (`[3·inner, inner]` logical,
    /// varied nibbles + block scales) with its Kitchen companions and the file-wide
    /// `_quantization_metadata` declaration.
    fn write_nvfp4_qkv_file(path: &Path, cfg: &Flux2Config) {
        let inner = cfg.inner_dim();
        let (rows, cols) = (3 * inner, inner);
        let packed: Vec<u8> = (0..rows * cols / 2)
            .map(|i| ((i * 37 + i / 11) % 251) as u8)
            .collect();
        let mut scales = vec![0u8; Nvfp4Tensor::scale_tensor_len(rows, cols)];
        let sf_rows = rows.next_multiple_of(128);
        for r in 0..rows {
            for blk in 0..cols / 16 {
                // Valid UE4M3 magnitudes (no NaN / sign bit): exponents around 1.0.
                scales[Nvfp4Tensor::scale_offset_for(r, blk, sf_rows)] =
                    0x30 + ((r + blk) % 16) as u8;
            }
        }
        let global = 1.5f32.to_le_bytes();
        let scale_shape = candle_gen::gen_core::nvfp4_scale_shape([rows, cols]).to_vec();
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "double_blocks.0.img_attn.qkv.weight",
            ::safetensors::tensor::TensorView::new(
                ::safetensors::Dtype::U8,
                vec![rows, cols / 2],
                &packed,
            )
            .unwrap(),
        );
        tensors.insert(
            "double_blocks.0.img_attn.qkv.weight_scale",
            ::safetensors::tensor::TensorView::new(
                ::safetensors::Dtype::F8_E4M3,
                scale_shape,
                &scales,
            )
            .unwrap(),
        );
        tensors.insert(
            "double_blocks.0.img_attn.qkv.weight_scale_2",
            ::safetensors::tensor::TensorView::new(::safetensors::Dtype::F32, vec![], &global)
                .unwrap(),
        );
        let metadata = std::collections::HashMap::from([(
            "_quantization_metadata".to_string(),
            r#"{"format_version": "1.0", "layers": {"double_blocks.0.img_attn.qkv": {"format": "nvfp4"}}}"#
                .to_string(),
        )]);
        ::safetensors::serialize_to_file(tensors, Some(metadata), path).unwrap();
    }

    #[test]
    fn packed_nvfp4_qkv_split_agrees_with_the_dense_decode() {
        let cfg = nvfp4_cfg();
        let inner = cfg.inner_dim();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("klein-nvfp4-qkv.safetensors");
        write_nvfp4_qkv_file(&path, &cfg);
        let cpu = Device::Cpu;
        let mapping = mapping_for(&cfg);

        // Dense decode of the same file — the reference the packed split must agree with.
        let dense_plan =
            plan_logical_weights(&path, &mapping, &CandleCodecResidency::DENSE).unwrap();
        let dense_reader = LogicalWeightReader::open(&path, dense_plan, &cpu).unwrap();

        let packed_plan = plan_logical_weights(&path, &mapping, &ForcePackedNvfp4).unwrap();
        let packed_reader = LogicalWeightReader::open(&path, packed_plan, &cpu).unwrap();

        for key in [
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_k.weight",
            "transformer_blocks.0.attn.to_v.weight",
        ] {
            let LogicalTensor::Dense(reference) = dense_reader.read(key).unwrap() else {
                panic!("dense policy materializes dense");
            };
            let reference = reference.to_dtype(DType::F32).unwrap();
            let LogicalTensor::PackedNvfp4 { tensor, .. } = packed_reader.read(key).unwrap() else {
                panic!("forced-packed policy must materialize PackedNvfp4 for {key}");
            };
            assert_eq!((tensor.rows, tensor.cols), (inner, inner));
            let dequant = tensor.dequantize().unwrap();
            let delta = (dequant - reference)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert_eq!(
                delta, 0.0,
                "{key}: the packed row slice must dequantize to the dense decode's slice"
            );
        }
    }

    /// A mapping wrapper that mis-declares transforms — the epic-style mutation witnesses: an
    /// overlapping fused partition and a half swap on a packed row must both go red at PLAN time.
    struct Mutated<'a> {
        base: Flux2BflToDiffusersMapping<'a>,
        overlap: bool,
        packed_half_swap: bool,
    }

    impl LogicalKeyMapping for Mutated<'_> {
        fn mapping_id(&self) -> &'static str {
            self.base.mapping_id()
        }
        fn logical_key(&self, physical_key: &str) -> Option<String> {
            self.base.logical_key(physical_key)
        }
        fn logical_shape(&self, logical_key: &str) -> Option<Vec<usize>> {
            self.base.logical_shape(logical_key)
        }
        fn logical_transform(&self, physical_key: &str) -> Option<LogicalTransformDeclaration> {
            let mut declaration = self.base.logical_transform(physical_key)?;
            if self.overlap && physical_key.ends_with("img_attn.qkv.weight") {
                // Shift the second slice back one row: rows overlap and the partition breaks.
                if let Some(rows) = declaration.outputs[1].rows.as_mut() {
                    rows.start -= 1;
                }
            }
            if self.packed_half_swap && physical_key.ends_with("img_attn.qkv.weight") {
                declaration.outputs[0].half_swap = true;
            }
            Some(declaration)
        }
    }

    #[test]
    fn overlapping_qkv_partition_refuses_at_plan_time() {
        let cfg = nvfp4_cfg();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("klein-nvfp4-qkv.safetensors");
        write_nvfp4_qkv_file(&path, &cfg);
        let mapping = Mutated {
            base: mapping_for(&cfg),
            overlap: true,
            packed_half_swap: false,
        };
        let error = plan_logical_weights(&path, &mapping, &CandleCodecResidency::DENSE)
            .expect_err("an overlapping partition must refuse")
            .to_string();
        assert!(error.contains("overlap"), "got: {error}");
    }

    #[test]
    fn half_swap_on_a_packed_row_refuses_at_plan_time() {
        let cfg = nvfp4_cfg();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("klein-nvfp4-qkv.safetensors");
        write_nvfp4_qkv_file(&path, &cfg);
        let mapping = Mutated {
            base: mapping_for(&cfg),
            overlap: false,
            packed_half_swap: true,
        };
        let error = plan_logical_weights(&path, &mapping, &ForcePackedNvfp4)
            .expect_err("a half swap on a packed residency must refuse")
            .to_string();
        assert!(
            error.contains("half swap") || error.contains("half-swap"),
            "got: {error}"
        );
        // The same declaration under a dense residency is legal — the refusal is really keyed on
        // the residency, not on the key.
        plan_logical_weights(&path, &mapping, &CandleCodecResidency::DENSE)
            .expect("a dense residency accepts the (mutated) half swap");
    }

    #[test]
    fn a_dev_shaped_file_refuses_through_the_klein_mapping() {
        let cfg = tiny_cfg();
        let inner = cfg.inner_dim();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dev-guidance.safetensors");
        let bytes = bf16_bytes(&fill(7, inner * cfg.timestep_channels));
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "guidance_in.in_layer.weight",
            ::safetensors::tensor::TensorView::new(
                ::safetensors::Dtype::BF16,
                vec![inner, cfg.timestep_channels],
                &bytes,
            )
            .unwrap(),
        );
        ::safetensors::serialize_to_file(tensors, None, &path).unwrap();
        let error = plan_logical_weights(&path, &mapping_for(&cfg), &CandleCodecResidency::DENSE)
            .expect_err("a dev-only key must refuse, never skip")
            .to_string();
        assert!(
            error.contains("guidance_in.in_layer.weight"),
            "got: {error}"
        );
    }

    #[test]
    fn plan_semantic_summary_is_stable_across_copies() {
        let cfg = tiny_cfg();
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("managed.safetensors");
        write_dense_bfl_file(&a, &cfg);
        let b = dir.path().join("linked.safetensors");
        std::fs::copy(&a, &b).unwrap();
        let mapping = mapping_for(&cfg);
        let plan_a = plan_logical_weights(&a, &mapping, &CandleCodecResidency::DENSE).unwrap();
        let plan_b = plan_logical_weights(&b, &mapping, &CandleCodecResidency::DENSE).unwrap();
        assert_eq!(
            plan_semantic_summary(&plan_a),
            plan_semantic_summary(&plan_b)
        );
    }
}
