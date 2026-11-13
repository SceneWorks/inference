//! The H3-Image **training memory estimate** on candle/CUDA (SW-7098), computed the way inference
//! admission computes a render's ([`crate::image_memory_strategy::admission`]): the same
//! header-read weight facts ([`ImageMemoryFacts::resolve`]), the same rung-4 selection the trainer
//! runs every inference phase at, the same separate GPU and host columns per phase, and the same
//! rule that the peak is the **largest phase** (each phase releases its component and synchronizes
//! before the next maps). The GPU column is compared with the device's training budget, as
//! admission compares a render's.
//!
//! | phase | priced as |
//! | --- | --- |
//! | conditioning | inference admission's own vision-tower / conditioning / reference-encode phases for each item at rung 4, plus the cache built so far |
//! | latents | the VAE encoder over the largest target, plus the cache |
//! | train | [`gen_core_train::h3_image::train_step_bytes`] (one streamed block, every block input, one block's backward) plus candle's float32 attention scores and allocator slack, plus the cache; host: the runtime-format DiT stack the streamed route keeps |
//! | preview | inference admission's whole-render peak at rung 4 for the preview request, plus the training state that stays resident through it |
//!
//! Every number is DERIVED (structural counts over the staged tier's headers and the request's
//! shapes), never measured; the family's measured anchors are taken once at the end of the feature.

use candle_gen::gen_core::h3_image as contract;
use candle_gen::gen_core::{
    Error, LoadSpec, MemoryBudget, MemoryNumericTier, MemoryProviderContract, MemorySelection,
    MemoryStrategy, Result,
};

use gen_core_train::h3_image::{
    train_step_bytes, DitGeometry, DitWeightBytes, TrainStepBytes, TrainStepShape, TRAINER_ID,
};

use crate::image_memory_strategy::admission::{
    estimate, ladder, ImageMemoryEstimate, ImageMemoryFacts, ImagePhase, ImageRequestShape,
};

/// Float32 score-shaped tensors candle's differentiated materialized attention holds per block
/// beyond the activation-width ones [`train_step_bytes`] counts: the scores, the max-shifted
/// scores, their exponent and the float32 probabilities (`crate::nn::sdpa_materialized` keeps the
/// score and softmax island in float32, and the composable softmax retains each for its backward).
pub const F32_SCORE_BUFFERS: u64 = 4;
/// Headroom for the CUDA caching allocator's fragmentation around one block's backward.
const ALLOCATOR_SLACK_BYTES: u64 = 1 << 30;

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
    /// The preview request (`None` when previews are off): item 0's canvas and references.
    pub preview: Option<TrainingItemShape>,
}

/// One phase's demand in the two pools, which are physically separate on CUDA.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseBytes {
    pub device: u64,
    pub host: u64,
}

/// The four phases' demand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingEstimate {
    pub conditioning: PhaseBytes,
    pub latents: PhaseBytes,
    pub train: PhaseBytes,
    pub preview: PhaseBytes,
    /// The train phase's own breakdown.
    pub step: TrainStepBytes,
}

impl TrainingEstimate {
    fn phases(&self) -> [(&'static str, PhaseBytes); 4] {
        [
            ("caption encoding", self.conditioning),
            ("latent encoding", self.latents),
            ("the training step", self.train),
            ("the preview render", self.preview),
        ]
    }

    /// The run's GPU peak: the largest phase.
    pub fn peak_device(&self) -> u64 {
        self.phases()
            .iter()
            .map(|(_, p)| p.device)
            .max()
            .unwrap_or(0)
    }

    /// The run's host peak: the largest phase.
    pub fn peak_host(&self) -> u64 {
        self.phases().iter().map(|(_, p)| p.host).max().unwrap_or(0)
    }

    /// The phase that sets [`Self::peak_device`].
    pub fn binding_phase(&self) -> &'static str {
        let peak = self.peak_device();
        self.phases()
            .into_iter()
            .find(|(_, p)| p.device == peak)
            .map_or("the training step", |(name, _)| name)
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
                 LoadShape::DeferredMaterialization load of a streamable source"
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

fn phase(estimate: &ImageMemoryEstimate, which: &[ImagePhase]) -> PhaseBytes {
    estimate
        .phases
        .iter()
        .filter(|p| which.contains(&p.phase))
        .fold(PhaseBytes::default(), |acc, p| PhaseBytes {
            device: acc.device.max(p.device_bytes),
            host: acc.host.max(p.host_bytes),
        })
}

/// `(text, reference, image)` rows of one item's training sequence: the caption and reference rows
/// exactly as admission counts them, and the target's rows on its own training canvas (a target
/// is encoded at the training canvas, never at the generator's larger minimum-area canvas).
fn train_rows(item: &TrainingItemShape) -> (u64, u64, u64) {
    let rows = crate::image_memory_strategy::admission::ImageRequestRows::of(&request_shape(item));
    let cell = u64::from(gen_core_train::h3_image::TRAINING_CANVAS_STRIDE);
    let image = (u64::from(item.canvas.0) / cell) * (u64::from(item.canvas.1) / cell);
    (rows.text_rows, rows.reference_rows.iter().sum(), image)
}

/// The cached dataset on the device: each item's refined text rows (activation width) and its
/// reference latents once, its target latent once per edge (float32 packed rows).
fn cache_bytes(hidden: u64, run: &TrainingRunShape) -> u64 {
    let per_row = DitGeometry::SHIPPED.patch_dim * 4;
    run.items.iter().fold(0u64, |total, item| {
        let (text, reference, image) = train_rows(item);
        total
            .saturating_add(text * hidden * run.width)
            .saturating_add(reference * per_row)
            .saturating_add(u64::from(run.edges.max(1)) * image * per_row)
    })
}

/// The DiT weight bytes the train phase holds, from admission's header-read facts: the trunk, each
/// block's body (its stack bytes less its `adaln_proj`, evicted after the precompute), each block's
/// `adaln_proj`, and the modulation tables over the run's distinct timesteps.
fn dit_weights(facts: &ImageMemoryFacts, edit: bool) -> DitWeightBytes {
    let stream = &facts.stream;
    let blocks = stream.stacks.first().cloned().unwrap_or_default();
    let adaln = facts.task.evicted_bytes / blocks.len().max(1) as u64;
    DitWeightBytes {
        trunk: stream.resident_bytes,
        block_body: blocks.iter().map(|b| b.saturating_sub(adaln)).collect(),
        block_adaln: vec![adaln; blocks.len()],
        modulation_tables: facts.task.table_bytes_per_timestep
            * (u64::from(contract::FIXED_STEPS) + u64::from(edit)),
    }
}

/// The estimate of `run` against `facts` on `contract` — see the module docs.
pub fn estimate_training(
    contract: &MemoryProviderContract,
    facts: &ImageMemoryFacts,
    tier: MemoryNumericTier,
    run: &TrainingRunShape,
) -> Result<TrainingEstimate> {
    let selection = streamed_selection(contract, tier)?;
    let geometry = DitGeometry::SHIPPED;
    let cache = cache_bytes(geometry.hidden, run);
    let edit = run.items.iter().any(|i| !i.references.is_empty());
    let weights = dit_weights(facts, edit);
    let widest = crate::config::MiniMaxH3VaeConfig::image()
        .block_out_channels
        .iter()
        .copied()
        .max()
        .unwrap_or(0) as u64;
    let host_dit = if run.checkpointed {
        facts.dit_prepare_host.max(facts.dit_retained_host)
    } else {
        facts.dit_prepare_host
    };
    let mut conditioning = PhaseBytes::default();
    let mut latents = PhaseBytes::default();
    let mut train = PhaseBytes::default();
    let mut step = TrainStepBytes::default();
    for item in &run.items {
        let inference = estimate(contract, facts, &request_shape(item), &selection)?;
        let c = phase(
            &inference,
            &[
                ImagePhase::VisionTower,
                ImagePhase::Conditioning,
                ImagePhase::ReferenceEncode,
            ],
        );
        conditioning.device = conditioning.device.max(c.device);
        conditioning.host = conditioning.host.max(c.host);
        // The target encode: the encoder, its widest full-resolution activation (three live at
        // once, float32) and the target's pixels.
        let pixels = u64::from(item.canvas.0) * u64::from(item.canvas.1);
        latents.device = latents
            .device
            .max(facts.encoder_bytes + pixels * widest * 4 * 3 + pixels * 3 * 4);
        latents.host = latents.host.max(pixels * 3 * (1 + 4));
        let (text_rows, reference_rows, image_rows) = train_rows(item);
        let s = TrainStepShape {
            text_rows,
            reference_rows,
            image_rows,
            width: run.width,
            trainable_elements: run.trainable_elements,
            optimizer_state_per_element: run.optimizer_state_per_element,
            lokr_delta_block_max: run.lokr_delta_block_max,
            lokr_delta_total: run.lokr_delta_total,
            checkpointed: run.checkpointed,
        };
        let bytes = train_step_bytes(&geometry, &weights, &s);
        let seq = s.sequence();
        let f32_scores = geometry.heads * seq * seq * 4 * F32_SCORE_BUFFERS;
        let live_blocks = if run.checkpointed { 1 } else { geometry.layers };
        let total = bytes.total() + live_blocks * f32_scores + ALLOCATOR_SLACK_BYTES;
        if total > train.device {
            train.device = total;
            step = bytes;
        }
    }
    let with_cache = |p: PhaseBytes| PhaseBytes {
        device: p.device.saturating_add(cache),
        host: p.host,
    };
    let conditioning = with_cache(conditioning);
    let latents = with_cache(latents);
    let train = PhaseBytes {
        device: train.device.saturating_add(cache),
        host: host_dit,
    };
    // Previews run the generator with the training DiT's trunk and tables still on the device, the
    // adapter state live, and (streamed) its host stack still held.
    let preview = match &run.preview {
        Some(item) => {
            let render = estimate(contract, facts, &request_shape(item), &selection)?;
            PhaseBytes {
                device: render
                    .peak_device_bytes()
                    .saturating_add(cache)
                    .saturating_add(step.adapter_state)
                    .saturating_add(weights.trunk + weights.modulation_tables),
                host: render
                    .peak_host_bytes()
                    .saturating_add(if run.checkpointed {
                        facts.dit_retained_host
                    } else {
                        0
                    }),
            }
        }
        None => PhaseBytes::default(),
    };
    Ok(TrainingEstimate {
        conditioning,
        latents,
        train,
        preview,
        step,
    })
}

/// A training budget of `free_bytes` of device memory (no further headroom: the caller's figure
/// already carries the engine's training safe fraction).
pub fn budget_of(free_bytes: u64) -> MemoryBudget {
    MemoryBudget {
        total_bytes: free_bytes,
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    }
}

fn gb(bytes: u64) -> f64 {
    crate::image_memory_strategy::admission::decimal_gb(bytes)
}

/// Refuse `estimate` when its GPU peak does not fit `budget`, naming the binding phase and the
/// levers.
pub fn admit_training(
    estimate: &TrainingEstimate,
    budget: MemoryBudget,
    run: &TrainingRunShape,
    edge: u32,
) -> Result<()> {
    if budget.fits(estimate.peak_device()) {
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
    let p = |b: PhaseBytes| format!("{:.1}/{:.1}", gb(b.device), gb(b.host));
    Err(Error::Unsupported(format!(
        "{TRAINER_ID} trainer: this run's derived GPU peak is ~{:.1} GB in {}, which exceeds this \
         device's ~{:.1} GB training budget (GPU/host GB: captions {} / latents {} / train {} / \
         preview {}). Refusing before any weight is read. To fit: {}.",
        gb(estimate.peak_device()),
        estimate.binding_phase(),
        gb(budget.effective_bytes()),
        p(estimate.conditioning),
        p(estimate.latents),
        p(estimate.train),
        p(estimate.preview),
        advice.join("; ")
    )))
}

/// The contract and tier a trainer spec prices at.
pub fn contract_and_tier(spec: &LoadSpec) -> Result<(MemoryProviderContract, MemoryNumericTier)> {
    Ok((
        crate::image_memory_strategy::contract_for(spec)?,
        MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{LoadShape, Quant, WeightsSource};

    const GB: u64 = 1_000_000_000;

    fn contract() -> MemoryProviderContract {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-minimax-h3-image".into()))
            .with_load_shape(LoadShape::DeferredMaterialization);
        crate::image_memory_strategy::weights_free_contract(&spec).unwrap()
    }

    fn tier(quant: Option<Quant>) -> MemoryNumericTier {
        MemoryNumericTier {
            precision: candle_gen::gen_core::Precision::Bf16,
            quant,
            component_precision_floors: &[],
        }
    }

    fn facts() -> ImageMemoryFacts {
        crate::image_memory_strategy::admission::shipped_scale_facts(0)
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
            preview: None,
        }
    }

    /// The conditioning phase IS inference admission's, at rung 4 (plus the cache): the same
    /// facts, the same selection, the same per-phase maximum; the peak is the largest phase.
    #[test]
    fn conditioning_is_inference_admissions_own_phase() {
        let c = contract();
        let f = facts();
        let t = tier(None);
        let r = run(true);
        let est = estimate_training(&c, &f, t, &r).unwrap();
        let sel = streamed_selection(&c, t).unwrap();
        let inference = estimate(&c, &f, &request_shape(&r.items[0]), &sel).unwrap();
        let want = phase(&inference, &[ImagePhase::Conditioning]);
        assert_eq!(est.conditioning.device, want.device + cache_bytes(5376, &r));
        assert_eq!(est.conditioning.host, want.host);
        assert_eq!(
            est.peak_device(),
            [
                est.conditioning.device,
                est.latents.device,
                est.train.device,
                est.preview.device
            ]
            .into_iter()
            .max()
            .unwrap()
        );
    }

    /// The default run — 2048², checkpointed (streamed) — fits an 80 GB card by design, and its
    /// train phase holds one block of DiT weights on the device, not fifty, with the stack on the
    /// host; the same run with the whole DiT resident and every block's activations retained does
    /// not fit.
    #[test]
    fn the_default_2048_run_is_admitted_streamed_and_refused_resident() {
        let c = contract();
        let f = facts();
        let t = tier(None);
        let streamed = estimate_training(&c, &f, t, &run(true)).unwrap();
        let dit: u64 = f.stream.resident_bytes + f.stream.stacks[0].iter().sum::<u64>();
        assert!(
            streamed.step.weights < dit / 10,
            "{} vs {dit}",
            streamed.step.weights
        );
        assert!(streamed.train.host >= f.dit_retained_host);
        admit_training(&streamed, budget_of(80 * GB), &run(true), 2048).unwrap();
        let resident = estimate_training(&c, &f, t, &run(false)).unwrap();
        assert!(resident.train.device > streamed.train.device);
        let err = admit_training(&resident, budget_of(80 * GB), &run(false), 2048)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Gradient Checkpointing"), "{err}");
        // Even streamed, a small card refuses with the binding phase named.
        let err = admit_training(&streamed, budget_of(8 * GB), &run(true), 2048)
            .unwrap_err()
            .to_string();
        assert!(err.contains("exceeds"), "{err}");
    }

    /// An edit's references lengthen the sequence, so the train phase grows with them, and so does
    /// the conditioning phase (the vision tower and the reference encode).
    #[test]
    fn references_raise_the_train_phase() {
        let c = contract();
        let f = facts();
        let t = tier(None);
        let t2i = estimate_training(&c, &f, t, &run(true)).unwrap();
        let mut edit = run(true);
        edit.items = vec![item(vec![(1024, 1024); 2]); 8];
        let edit = estimate_training(&c, &f, t, &edit).unwrap();
        assert!(edit.train.device > t2i.train.device);
        assert!(edit.conditioning.device > t2i.conditioning.device);
    }

    /// A preview adds the generator's whole render on top of the training state.
    #[test]
    fn a_preview_prices_the_render_over_the_training_state() {
        let c = contract();
        let f = facts();
        let t = tier(None);
        let mut with = run(true);
        with.preview = Some(item(vec![]));
        let est = estimate_training(&c, &f, t, &with).unwrap();
        let sel = streamed_selection(&c, t).unwrap();
        let render = estimate(&c, &f, &request_shape(&with.items[0]), &sel).unwrap();
        assert!(est.preview.device > render.peak_device_bytes());
        assert!(est.preview.host >= render.peak_host_bytes() + f.dit_retained_host);
    }

    #[test]
    fn a_resident_load_declares_no_streaming_rung() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-minimax-h3-image".into()));
        let c = crate::image_memory_strategy::weights_free_contract(&spec).unwrap();
        assert!(streamed_selection(&c, tier(None)).is_err());
    }
}
