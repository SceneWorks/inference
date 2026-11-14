//! The H3-Image **training memory estimate** on MLX (SW-7097), computed the way inference admission
//! computes a render's ([`crate::h3_image::admission`]): the same header-read weight facts
//! ([`ImageMemoryFacts::resolve`]), the same per-phase pricing on unified memory (device + host),
//! the same rule that the peak is the **largest phase** (each phase releases its component before
//! the next maps), and the same [`MemoryBudget::fits`] test.
//!
//! | phase | priced as |
//! | --- | --- |
//! | conditioning | inference admission's own conditioning phase for each item at rung 4 (the text encoder streamed one layer at a time; an edit's vision tower and reference encode), plus the cache built so far |
//! | latents | the VAE encoder over the largest target, plus the cache |
//! | train | [`gen_core_train::h3_image::train_step_bytes`] (one streamed block, every block input, one block's backward) plus the MLX command-buffer pipelining and eval slack, plus the cache |
//! | preview | inference admission's whole-render peak at rung 4 for the preview request, plus the training state that stays resident through it |
//!
//! Every number is DERIVED (structural counts over the staged tier's headers and the request's
//! shapes), never measured; the family's measured anchors are taken once at the end of the feature.

use mlx_gen::gen_core::{
    Error, LoadSpec, MemoryBudget, MemoryNumericTier, MemoryPhase, MemoryProviderContract,
    MemorySelection, MemoryStrategy, Result,
};

use gen_core_train::h3_image::{
    train_step_bytes, DitGeometry, DitWeightBytes, TrainStepBytes, TrainStepShape, TRAINER_ID,
};

use crate::h3_image::admission::{
    estimate, ladder, modulation_table_bytes, sequence_rows, ImageMemoryFacts, ImageRequestShape,
};

/// MLX keeps up to this many command buffers in flight (`transforms.cpp`), each retaining its
/// ops' inputs until completion; the one being encoded is one more. The Qwen-Image 2.1 trainer's
/// `training_memory` derives the same envelope.
const MAX_ACTIVE_TASKS: u64 = 10;
/// An op's input and output are both retained by its buffer.
const BUFFER_INPUT_OUTPUTS: u64 = 2;
/// `device.cpp` commits a buffer once it crosses 50 MiB; the allocation crossing it is 51 MiB.
const METAL_BUFFER_BYTES: u64 = 51 << 20;
/// Headroom for MLX's own evaluation scratch around one forced graph.
const EVAL_SLACK_BYTES: u64 = 1 << 30;

/// Whole heads one training attention chunk covers over `seq_len` rows: inference rung 3's own
/// plan ([`crate::h3_image::memory::image_attention_budget`] at the declared
/// [`crate::h3_image::memory::IMAGE_ATTENTION_CHUNK_SIZE`], floored at one whole head).
pub fn training_attention_heads_per_chunk(seq_len: u64) -> u64 {
    let heads = DitGeometry::SHIPPED.heads;
    crate::h3_image::memory::image_attention_heads_per_chunk(
        training_attention_budget(seq_len),
        heads,
        seq_len,
    )
    .clamp(1, heads)
}

/// The training attention budget over `seq_len` rows (see [`training_attention_heads_per_chunk`]).
pub fn training_attention_budget(seq_len: u64) -> mlx_gen::attention::AttentionBudget {
    crate::h3_image::memory::image_attention_budget(
        crate::h3_image::memory::IMAGE_ATTENTION_CHUNK_SIZE,
        seq_len,
    )
}

/// One dataset item as the estimate sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainingItemShape {
    /// The target's training canvas `(width, height)` at the run's largest edge.
    pub canvas: (u32, u32),
    /// Caption tokens after the 4096-token truncation.
    pub caption_tokens: u32,
    /// Each reference's original `(width, height)`; empty for a captioned item.
    pub references: Vec<(u32, u32)>,
}

/// The run-level settings the estimate reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainingRunShape {
    pub items: Vec<TrainingItemShape>,
    /// Bucket edges: every item is cached once per edge.
    pub edges: u32,
    /// Activation width (2 bf16, 4 f32).
    pub width: u64,
    pub trainable_elements: u64,
    pub optimizer_state_per_element: u64,
    pub lokr_delta_block_max: u64,
    pub lokr_delta_total: u64,
    pub checkpointed: bool,
    /// Training attention chunked on whole heads, each chunk recomputed in the backward (E20) —
    /// the trainer always runs this; `false` prices the unchunked call.
    pub chunked_attention: bool,
    /// The preview request (`None` when previews are off): item 0's canvas and references.
    pub preview: Option<TrainingItemShape>,
}

/// The four phases' unified-memory demand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingEstimate {
    pub conditioning: u64,
    pub latents: u64,
    pub train: u64,
    pub preview: u64,
    /// The train phase's own breakdown.
    pub step: TrainStepBytes,
}

impl TrainingEstimate {
    /// The run peak: the largest phase.
    pub fn peak(&self) -> u64 {
        self.conditioning
            .max(self.latents)
            .max(self.train)
            .max(self.preview)
    }

    /// The phase that sets [`Self::peak`].
    pub fn binding_phase(&self) -> &'static str {
        let peak = self.peak();
        if peak == self.train {
            "the training step"
        } else if peak == self.conditioning {
            "caption encoding"
        } else if peak == self.latents {
            "latent encoding"
        } else {
            "the preview render"
        }
    }
}

/// The rung-4 selection (both transformers streamed one block at a time) the trainer runs every
/// inference phase at — the cheapest rung on the ladder.
pub fn streamed_selection(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
) -> Result<MemorySelection> {
    ladder(contract, tier)
        .into_iter()
        .find(|s| s.strategy == MemoryStrategy::BoundedTransformerResidency)
        .ok_or_else(|| {
            Error::Unsupported(format!(
                "{TRAINER_ID} trainer: the load declares no transformer-streaming rung; training \
                 streams the frozen DiT and text encoder and needs a \
                 LoadShape::DeferredMaterialization load"
            ))
        })
}

fn request_shape(item: &TrainingItemShape) -> ImageRequestShape {
    ImageRequestShape {
        width: item.canvas.0,
        height: item.canvas.1,
        count: 1,
        prompt_tokens: Some(item.caption_tokens),
        references: item.references.clone(),
    }
}

fn phase(estimate: &crate::h3_image::admission::ImageMemoryEstimate, which: MemoryPhase) -> u64 {
    estimate
        .phases
        .iter()
        .find(|p| p.phase == which)
        .map_or(0, |p| p.total_bytes())
}

/// `(text, reference, image)` rows of one item's training sequence: the caption and reference rows
/// exactly as admission counts them, and the target's rows on its own training canvas (a target
/// is encoded at the training canvas, never at the generator's larger minimum-area canvas).
fn train_rows(item: &TrainingItemShape) -> Result<(u64, u64, u64)> {
    let (text, reference, _) = sequence_rows(&request_shape(item))?;
    let cell = u64::from(gen_core_train::h3_image::TRAINING_CANVAS_STRIDE);
    let image = (u64::from(item.canvas.0) / cell) * (u64::from(item.canvas.1) / cell);
    Ok((text, reference, image))
}

/// The cached dataset: each item's context (float32 rows of the text encoder's width) and its
/// reference latents once, its target latent once per edge.
fn cache_bytes(facts_text_width: u64, run: &TrainingRunShape) -> Result<u64> {
    let mut total = 0u64;
    for item in &run.items {
        let (text, reference, image) = train_rows(item)?;
        // A packed row is a 2x2 patch of 24-channel latent cells: 96 float32 values.
        let per_row = DitGeometry::SHIPPED.patch_dim * 4;
        total = total
            .saturating_add(text * facts_text_width * 4)
            .saturating_add(reference * per_row)
            .saturating_add(u64::from(run.edges.max(1)) * image * per_row);
    }
    Ok(total)
}

/// The estimate of `run` against `facts` on `contract` — see the module docs.
///
/// `facts` prices the generator's own phases (conditioning, previews) at the load's precision;
/// `train_facts` prices the training DiT's weights at the run's compute dtype — a float32 run casts
/// every dense weight the DiT loads to float32 (packed q4/q8 bases stay packed), so its facts are
/// the same headers resolved at [`mlx_gen::Precision::Fp32`].
pub fn estimate_training(
    contract: &MemoryProviderContract,
    facts: &ImageMemoryFacts,
    train_facts: &ImageMemoryFacts,
    tier: MemoryNumericTier,
    run: &TrainingRunShape,
) -> Result<TrainingEstimate> {
    let selection = streamed_selection(contract, tier)?;
    let text_width = crate::text_encoder::MiniMaxH3TeConfig::qwen3_vl_32b().hidden_size as u64;
    let cache = cache_bytes(text_width, run)?;
    let geometry = DitGeometry::SHIPPED;
    let edit = run.items.iter().any(|i| !i.references.is_empty());
    let a = run.width;
    let weights = DitWeightBytes {
        trunk: train_facts.dit_trunk_bytes,
        block_body: train_facts.dit_body_bytes.clone(),
        block_adaln: train_facts.dit_adaln_bytes.clone(),
        modulation_tables: modulation_table_bytes(edit, a),
    };
    let encoder = crate::config::MiniMaxH3VaeConfig::image();
    let widest = encoder
        .block_out_channels
        .iter()
        .copied()
        .max()
        .unwrap_or(0) as u64;
    let mut conditioning = 0u64;
    let mut latents = 0u64;
    let mut train = 0u64;
    let mut step = TrainStepBytes::default();
    for item in &run.items {
        let shape = request_shape(item);
        let inference = estimate(contract, facts, &shape, &selection)?;
        conditioning = conditioning.max(phase(&inference, MemoryPhase::Conditioning));
        // The target encode: the encoder, its widest full-resolution activation (three live at
        // once, float32) and the target's pixels.
        let pixels = u64::from(item.canvas.0) * u64::from(item.canvas.1);
        latents = latents.max(facts.encoder_bytes + pixels * widest * 4 * 3 + pixels * 3 * 4);
        let (text_rows, reference_rows, image_rows) = train_rows(item)?;
        let s = TrainStepShape {
            text_rows,
            reference_rows,
            image_rows,
            width: a,
            trainable_elements: run.trainable_elements,
            optimizer_state_per_element: run.optimizer_state_per_element,
            lokr_delta_block_max: run.lokr_delta_block_max,
            lokr_delta_total: run.lokr_delta_total,
            checkpointed: run.checkpointed,
            attention_heads_per_chunk: if run.chunked_attention {
                training_attention_heads_per_chunk(text_rows + reference_rows + image_rows)
            } else {
                geometry.heads
            },
        };
        let bytes = train_step_bytes(&geometry, &weights, &s);
        let seq = s.sequence();
        // Uncompleted buffers pinning non-score operands: the widest is the SwiGLU's float32
        // `[S, 2·ffn]` widen. Score-sized arrays a buffer pins are already among the attention
        // call's own (`SCORE_ARRAYS_PER_ATTENTION_VJP` / `PINNED_COMMAND_BUFFERS`).
        let pipelined = BUFFER_INPUT_OUTPUTS
            * (MAX_ACTIVE_TASKS + 1)
            * (seq * geometry.ffn * 2 * 4 + METAL_BUFFER_BYTES);
        let total = bytes.total() + pipelined + EVAL_SLACK_BYTES;
        if total > train {
            train = total;
            step = bytes;
        }
    }
    let conditioning = conditioning.saturating_add(cache);
    let latents = latents.saturating_add(cache);
    let train = train.saturating_add(cache);
    // Previews run the generator while the training DiT stays loaded and the adapter state live:
    // its trunk and tables when streamed (no block is held between steps), or the whole resident
    // DiT — every block body, and every installed LoKr delta — when not.
    let retained = if run.checkpointed {
        weights.trunk + weights.modulation_tables
    } else {
        weights
            .block_body
            .iter()
            .fold(weights.trunk + weights.modulation_tables, |t, b| {
                t.saturating_add(*b)
            })
            .saturating_add(2 * run.lokr_delta_total)
    };
    let preview = match &run.preview {
        Some(item) => estimate(contract, facts, &request_shape(item), &selection)?
            .peak_bytes()
            .saturating_add(cache)
            .saturating_add(step.adapter_state)
            .saturating_add(retained),
        None => 0,
    };
    Ok(TrainingEstimate {
        conditioning,
        latents,
        train,
        preview,
        step,
    })
}

/// This device's training budget, as admission builds one: MLX's memory limit with the engine's
/// [`mlx_gen::memory::SAFE_FRAC`] reserved as headroom.
pub fn device_budget() -> MemoryBudget {
    budget_of(mlx_rs::memory::get_memory_limit() as u64)
}

/// A budget of `total_bytes` with the engine's headroom reserved.
pub fn budget_of(total_bytes: u64) -> MemoryBudget {
    MemoryBudget {
        total_bytes,
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: (total_bytes as f64 * (1.0 - mlx_gen::memory::SAFE_FRAC)) as u64,
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// Refuse `estimate` when its peak does not fit `budget`, naming the binding phase and the levers.
pub fn admit_training(
    estimate: &TrainingEstimate,
    budget: MemoryBudget,
    run: &TrainingRunShape,
    edge: u32,
) -> Result<()> {
    if budget.fits(estimate.peak()) {
        return Ok(());
    }
    let mut advice = Vec::new();
    if !run.checkpointed {
        advice.push(
            "enable Gradient Checkpointing (streams the frozen DiT one block at a time and \
             recomputes each block in the backward)"
                .to_string(),
        );
    }
    advice.push(format!("lower the training resolution (now {edge} px)"));
    if run.items.iter().any(|i| !i.references.is_empty()) {
        advice.push("use fewer or smaller reference images".to_string());
    }
    if run.preview.is_some() {
        advice.push("turn off preview samples".to_string());
    }
    advice.push("stage the q4 or q8 DiT tier".to_string());
    advice.push("lower the rank".to_string());
    Err(Error::Unsupported(format!(
        "{TRAINER_ID} trainer: this run's derived peak memory is ~{:.1} GiB in {}, which exceeds \
         this device's ~{:.1} GiB effective budget (captions ~{:.1} / latents ~{:.1} / train \
         ~{:.1} / preview ~{:.1} GiB). Refusing before any weight is read. To fit: {}.",
        gib(estimate.peak()),
        estimate.binding_phase(),
        gib(budget.effective_bytes()),
        gib(estimate.conditioning),
        gib(estimate.latents),
        gib(estimate.train),
        gib(estimate.preview),
        advice.join("; ")
    )))
}

/// The weights-free contract and tier a trainer spec prices at (used by tests and by the trainer).
pub fn contract_and_tier(spec: &LoadSpec) -> Result<(MemoryProviderContract, MemoryNumericTier)> {
    Ok((
        crate::h3_image::memory::contract_for(spec)?,
        mlx_gen::request_scope::loaded_tier(spec),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{LoadShape, Quant, WeightsSource};

    const GB: u64 = 1_000_000_000;

    fn contract() -> MemoryProviderContract {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-minimax-h3-image".into()))
            .with_load_shape(LoadShape::DeferredMaterialization);
        crate::h3_image::memory::weights_free_contract(&spec).unwrap()
    }

    fn tier(quant: Option<Quant>) -> MemoryNumericTier {
        MemoryNumericTier {
            precision: mlx_gen::Precision::Bf16,
            quant,
            component_precision_floors: &[],
        }
    }

    fn item(references: Vec<(u32, u32)>) -> TrainingItemShape {
        TrainingItemShape {
            canvas: (2048, 2048),
            caption_tokens: 120,
            references,
        }
    }

    fn run(checkpointed: bool) -> TrainingRunShape {
        TrainingRunShape {
            items: vec![item(vec![]); 8],
            edges: 1,
            width: 2,
            // Rank-16 LoRA on every block leaf.
            trainable_elements: 50
                * 16
                * (3 * (5376 + 7168) + (7168 + 5376) + (5376 + 2 * 14336) + (14336 + 5376)),
            optimizer_state_per_element: 2,
            lokr_delta_block_max: 0,
            lokr_delta_total: 0,
            checkpointed,
            chunked_attention: true,
            preview: None,
        }
    }

    /// The conditioning phase IS inference admission's, at rung 4 (plus the cache): the same
    /// facts, the same selection, the same per-phase sum.
    #[test]
    fn conditioning_is_inference_admissions_own_phase() {
        let c = contract();
        let facts = ImageMemoryFacts::derived(Some(4));
        let t = tier(Some(Quant::Q4));
        let r = run(true);
        let est = estimate_training(&c, &facts, &facts, t, &r).unwrap();
        let sel = streamed_selection(&c, t).unwrap();
        let inference = estimate(&c, &facts, &request_shape(&r.items[0]), &sel).unwrap();
        let cache = cache_bytes(5120, &r).unwrap();
        assert_eq!(
            est.conditioning,
            phase(&inference, MemoryPhase::Conditioning) + cache
        );
        assert_eq!(
            est.peak(),
            [est.conditioning, est.latents, est.train, est.preview]
                .into_iter()
                .max()
                .unwrap()
        );
    }

    /// The default run — q4 tier, 2048², checkpointed (streamed) — fits a 128 GB machine by
    /// design and its train phase holds one block of DiT weights, not fifty; the same run with the
    /// whole DiT resident and every block's activations retained does not fit.
    #[test]
    fn the_default_2048_run_is_admitted_streamed_and_refused_resident() {
        let c = contract();
        let facts = ImageMemoryFacts::derived(Some(4));
        let t = tier(Some(Quant::Q4));
        let streamed = estimate_training(&c, &facts, &facts, t, &run(true)).unwrap();
        assert!(
            streamed.step.weights < facts.dit_bytes() / 4,
            "{} vs {}",
            streamed.step.weights,
            facts.dit_bytes()
        );
        admit_training(&streamed, budget_of(128 * GB), &run(true), 2048).unwrap();
        let resident = estimate_training(&c, &facts, &facts, t, &run(false)).unwrap();
        assert!(resident.train > streamed.train);
        let err = admit_training(&resident, budget_of(128 * GB), &run(false), 2048)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Gradient Checkpointing"), "{err}");
        // Even streamed, a small device refuses with the binding phase named.
        let err = admit_training(&streamed, budget_of(16 * GB), &run(true), 2048)
            .unwrap_err()
            .to_string();
        assert!(err.contains("exceeds"), "{err}");
    }

    /// An edit's references lengthen the sequence, so the train phase grows with them.
    #[test]
    fn references_raise_the_train_phase() {
        let c = contract();
        let facts = ImageMemoryFacts::derived(Some(4));
        let t = tier(Some(Quant::Q4));
        let t2i = estimate_training(&c, &facts, &facts, t, &run(true)).unwrap();
        let mut edit = run(true);
        edit.items = vec![item(vec![(1024, 1024); 2]); 8];
        let edit = estimate_training(&c, &facts, &facts, t, &edit).unwrap();
        assert!(edit.train > t2i.train);
        assert!(edit.conditioning > t2i.conditioning);
    }

    /// E20: the 2048² edit with two 1024² references (about 8.2 k rows) is admitted on a 128 GB
    /// machine with training attention chunked by design. Unchunked, the call's six full
    /// `[56, S, S]` score arrays (~50 GB) make the train phase the binding phase: it still fits
    /// 128 GB after the double-count correction, and is refused at 80 GB where the chunked run fits.
    #[test]
    fn the_2048_two_reference_edit_fits_128_gb_with_chunked_attention() {
        let c = contract();
        let facts = ImageMemoryFacts::derived(Some(4));
        let t = tier(Some(Quant::Q4));
        let mut edit = run(true);
        edit.items = vec![item(vec![(1024, 1024); 2])];
        let (text, reference, image) = train_rows(&edit.items[0]).unwrap();
        let seq = text + reference + image;
        assert!(seq > 8_000, "{text} + {reference} + {image}");
        assert!(training_attention_heads_per_chunk(seq) < 56);
        let chunked = estimate_training(&c, &facts, &facts, t, &edit).unwrap();
        admit_training(&chunked, budget_of(128 * GB), &edit, 2048).unwrap();
        let unchunked_run = TrainingRunShape {
            chunked_attention: false,
            ..edit.clone()
        };
        let unchunked = estimate_training(&c, &facts, &facts, t, &unchunked_run).unwrap();
        let full_scores = 6 * 56 * seq * seq * 2;
        assert!(unchunked.step.block_backward >= full_scores);
        // Chunked: one head's 6 + 11 arrays in place of 56 heads' 6.
        assert_eq!(
            unchunked.step.block_backward - chunked.step.block_backward,
            full_scores - 17 * seq * seq * 2
        );
        assert!(unchunked.train > chunked.train + full_scores / 2);
        admit_training(&chunked, budget_of(80 * GB), &edit, 2048).unwrap();
        assert!(admit_training(&unchunked, budget_of(80 * GB), &unchunked_run, 2048).is_err());
    }

    /// Previews render while the training DiT stays loaded: a resident (non-checkpointed) run
    /// keeps every block body, so its preview phase is priced above the streamed run's by them.
    #[test]
    fn a_resident_runs_preview_keeps_the_whole_training_dit() {
        let c = contract();
        let facts = ImageMemoryFacts::derived(Some(4));
        let t = tier(Some(Quant::Q4));
        let preview = |checkpointed: bool| TrainingRunShape {
            preview: Some(item(vec![])),
            ..run(checkpointed)
        };
        let streamed = estimate_training(&c, &facts, &facts, t, &preview(true)).unwrap();
        let resident = estimate_training(&c, &facts, &facts, t, &preview(false)).unwrap();
        let bodies: u64 = facts.dit_body_bytes.iter().sum();
        assert!(resident.preview >= streamed.preview + bodies);
    }

    /// The training DiT's weights come from `train_facts` (the run's compute dtype), the
    /// generator's phases from `facts`: wider training weights raise the train phase only.
    #[test]
    fn training_weights_are_priced_at_the_compute_dtype() {
        let c = contract();
        let facts = ImageMemoryFacts::derived(Some(4));
        let mut wide = facts.clone();
        wide.dit_body_bytes.iter_mut().for_each(|b| *b *= 2);
        let t = tier(Some(Quant::Q4));
        let narrow = estimate_training(&c, &facts, &facts, t, &run(true)).unwrap();
        let widened = estimate_training(&c, &facts, &wide, t, &run(true)).unwrap();
        assert!(widened.train > narrow.train);
        assert_eq!(widened.conditioning, narrow.conditioning);
    }

    #[test]
    fn a_resident_load_declares_no_streaming_rung() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-minimax-h3-image".into()));
        let c = crate::h3_image::memory::weights_free_contract(&spec).unwrap();
        assert!(streamed_selection(&c, tier(Some(Quant::Q4))).is_err());
    }
}
