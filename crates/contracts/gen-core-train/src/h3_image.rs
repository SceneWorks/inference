//! **MiniMax-H3-Image LoRA/LoKr training**, the backend-neutral half (SW-7097): what the MLX
//! trainer (`mlx_gen_minimax_h3::training`) and the Candle one (SW-7098) must agree on so an
//! adapter trained on either loads, trains and resumes identically on both.
//!
//! Tensor-free, like the rest of this crate. It owns:
//!
//! * the **trainer identity** — [`TRAINER_ID`], [`trainer_descriptor`], the adapter provenance
//!   ([`ADAPTER_PROVENANCE`], [`EDIT_ADAPTER_MARKER`]) and the run defaults
//!   ([`default_training_config`]: 2048², gradient checkpointing on);
//! * the **trainable surface** — [`resolve_target_paths`]: the six leaves of every DiT block by
//!   default, suffix-matched like PEFT's `target_modules`, with a typed refusal for the modules a
//!   distilled H3-Image run computes once per request and therefore cannot train;
//! * the **flow-matching training contract** — [`training_step`] and [`training_mix`], the noise
//!   level, head and regression target one micro-step trains on;
//! * the **training canvas** — [`training_canvas`];
//! * the **training-step memory envelope** — [`train_step_bytes`], the structural count both
//!   backends price their train phase with.
//!
//! # The flow-matching contract
//!
//! H3-Image is distilled to a fixed six-step sampler with one output head per step
//! (`pdd_proj_out.{i}`). Its velocity points **toward the data**: the released step is
//! `x_next = x + (σ_i − σ_{i+1})·v` and the LCM sampler's data estimate is `x0 = x + σ·v`. So at
//! noise level `σ` the interpolant is `x_σ = (1 − σ)·x0 + σ·ε` and the regression target is
//! `v = x0 − ε` (`dx/dσ = ε − x0`, stepped with a negative `dσ`). A micro-step trains at one of
//! the six **published** noise levels — [`gen_core::h3_image::sampler_sigmas`], the grid the
//! released sampler steps on — through that step's own head and its own AdaLN timestep
//! `t = 1 − σ`, exactly the forward the render runs at that step. The configured
//! `timestep_type` / `timestep_bias` distribution is honored by drawing a continuous noise level
//! from it and training at the nearest published level ([`training_step`]).

use gen_core::h3_image::{sampler_sigmas, FIXED_STEPS, MAX_REFERENCES, SIZE_STRIDE};
use gen_core::license::components::MINIMAX_H3_IMAGE;

use gen_core::generator::Modality;

use super::*;

/// Registry id of the H3-Image trainer: the generator id of the base model it adapts, so a trained
/// adapter's `baseModel` names the route that loads it.
pub const TRAINER_ID: &str = "minimax_h3_image";

/// The descriptor family (the generator's own).
pub const FAMILY: &str = "minimax_h3_image";

/// The default square training edge: H3-Image's native 4 MP generation canvas
/// ([`gen_core::h3_image::MIN_GENERATION_AREA_DEFAULT`] is `2048²`).
pub const DEFAULT_TRAINING_RESOLUTION: u32 = 2048;

/// Every training canvas edge is a multiple of this: one VAE cell (16 px) times the DiT's 2×2
/// patch — the generator's own [`SIZE_STRIDE`].
pub const TRAINING_CANVAS_STRIDE: u32 = SIZE_STRIDE;

/// The leaves of every DiT block a default run trains — the block's four attention projections and
/// its SwiGLU pair, in the published diffusers spelling the H3-Image LoRA loader addresses.
pub const BLOCK_TARGETS: [&str; 6] = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
];

/// The adapter-addressable modules (SW-7096) a training run cannot train, and why. Each runs once
/// per request, before the per-step forward, from inputs every step shares — the six timesteps or
/// the caption — so its output is a constant of the step's graph and no gradient reaches it.
pub const UNTRAINABLE_MODULES: [(&str, &str); 7] = [
    (
        "adaln_proj",
        "each block's AdaLN projection is precomputed once over the six timesteps and evicted",
    ),
    (
        "time_embedder",
        "the timestep MLP feeds only that precompute",
    ),
    (
        "norm_out",
        "the final AdaLN projection is precomputed with the blocks'",
    ),
    (
        "token_refiner",
        "the caption rows are refined once per item",
    ),
    (
        "context_embedder",
        "the caption rows are embedded once per item",
    ),
    (
        "proj_in",
        "the trainer differentiates the block stack only; the image input projection stays frozen",
    ),
    (
        "pdd_proj_out",
        "the trainer differentiates the block stack only; the six output heads stay frozen",
    ),
];

/// Provenance and licence stamped into every saved adapter's `__metadata__` (final and
/// intermediate, LoRA and LoKr), next to the shared `networkType` / `rank` / `alpha` reload keys.
/// `family` / `baseModel` are the pair SceneWorks' importer reads first; `license` /
/// `modelspec.license` / `licenseNotice` record the MiniMax-H3-Image Community License the base
/// weights are under, which a fine-tune of them inherits.
pub const ADAPTER_PROVENANCE: [(&str, &str); 6] = [
    ("family", FAMILY),
    ("baseModel", TRAINER_ID),
    ("ss_base_model_version", TRAINER_ID),
    ("license", MINIMAX_H3_IMAGE.declared),
    ("modelspec.license", MINIMAX_H3_IMAGE.declared),
    ("licenseNotice", LICENSE_NOTICE),
];

/// The attribution the base weights' licence requires, repeated in every adapter.
const LICENSE_NOTICE: &str = match MINIMAX_H3_IMAGE.attribution {
    Some(text) => text,
    None => "",
};

/// The extra `__metadata__` entry an adapter trained on an edit-pair dataset carries, so the
/// product can tell an edit adapter from a text-to-image one without reading tensors. The file is
/// the same PEFT/LyCORIS layout either way.
pub const EDIT_ADAPTER_MARKER: (&str, &str) = ("trainingMode", "edit");

/// The trainer's identity and capability surface on `backend` (`"mlx"` / `"candle"`). One
/// definition, so the two backends cannot advertise different trainers.
pub fn trainer_descriptor(backend: &'static str) -> TrainerDescriptor {
    TrainerDescriptor {
        id: TRAINER_ID,
        family: FAMILY,
        backend,
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
        supports_control: false,
        supports_full_finetune: false,
        // An edit pair carries the generator's own reference cap.
        max_reference_images: MAX_REFERENCES as u32,
        // Weight and gradient noise act at the shared adapter optimizer update; resolution buckets
        // cache every item once per edge. Every other technique is refused by the shared floor.
        techniques: TrainingTechniques {
            resolution_buckets: true,
            ..TrainingTechniques::ADAPTER_NOISE
        },
    }
}

/// The run defaults a client starts from: the shared defaults at [`DEFAULT_TRAINING_RESOLUTION`]
/// with gradient checkpointing on (the configuration the memory estimate admits by design — a
/// non-checkpointed run holds the whole DiT resident). Every field stays configurable.
pub fn default_training_config() -> TrainingConfig {
    TrainingConfig {
        resolution: DEFAULT_TRAINING_RESOLUTION,
        gradient_checkpointing: true,
        ..TrainingConfig::default()
    }
}

/// The H3-Image request floor every backend applies before any weight is read: no control images,
/// no reference/control workflow selected through `model_options`, and every training edge large
/// enough to hold one canvas stride. The shared control / full-fine-tune / technique / edit floors
/// run beside it ([`validate_floors`]).
pub fn validate_request(req: &TrainingRequest) -> gen_core::Result<()> {
    let prefix = format!("{TRAINER_ID} trainer");
    if let Some(idx) = req
        .items
        .iter()
        .position(|item| item.control_image_path.is_some())
    {
        return Err(gen_core::Error::Msg(format!(
            "{prefix}: item {idx} carries a control image; control images are not part of \
             MiniMax-H3-Image LoRA/LoKr training"
        )));
    }
    refuse_reference_control_model_options(&prefix, req)?;
    for bucket in req.config.training_buckets() {
        if bucket.resolution < TRAINING_CANVAS_STRIDE {
            return Err(gen_core::Error::Msg(format!(
                "{prefix}: training resolution {} is below one {TRAINING_CANVAS_STRIDE} px canvas \
                 stride",
                bucket.resolution
            )));
        }
    }
    Ok(())
}

/// Every shared floor plus [`validate_request`], in the order a trainer's `validate` runs them.
pub fn validate_floors(desc: &TrainerDescriptor, req: &TrainingRequest) -> gen_core::Result<()> {
    validate_control_request(desc, req)?;
    validate_full_finetune_request(desc, req)?;
    validate_training_techniques(desc, req)?;
    validate_edit_request(desc, req)?;
    validate_request(req)
}

/// Every module path an adapter can address on an H3-Image DiT of `num_layers` blocks,
/// `num_refiner_layers` refiner blocks and `num_heads` output heads — the SW-7096 loader's surface
/// (`image_adapter_target_paths`), spelled without a backend.
pub fn adapter_surface(
    num_layers: usize,
    num_refiner_layers: usize,
    num_heads: usize,
) -> Vec<String> {
    let mut v = Vec::new();
    for i in 0..num_layers {
        v.extend(
            BLOCK_TARGETS
                .iter()
                .map(|leaf| format!("transformer_blocks.{i}.{leaf}")),
        );
    }
    for i in 0..num_refiner_layers {
        v.extend(
            BLOCK_TARGETS
                .iter()
                .map(|leaf| format!("token_refiner.refiner_blocks.{i}.{leaf}")),
        );
    }
    v.extend((0..num_layers).map(|i| format!("transformer_blocks.{i}.adaln_proj.linear")));
    v.extend(
        [
            "proj_in",
            "context_embedder",
            "time_embedder.linear_1",
            "time_embedder.linear_2",
            "norm_out.linear",
        ]
        .map(String::from),
    );
    v.extend((0..num_heads).map(|i| format!("pdd_proj_out.{i}")));
    v
}

/// Why `path` cannot be trained, or `None` for a trainable block leaf.
fn untrainable_reason(path: &str) -> Option<&'static str> {
    if let Some(rest) = path.strip_prefix("transformer_blocks.") {
        let leaf = rest.split_once('.').map_or("", |(_, leaf)| leaf);
        if BLOCK_TARGETS.contains(&leaf) {
            return None;
        }
    }
    UNTRAINABLE_MODULES
        .iter()
        .find(|(stem, _)| path.starts_with(stem) || path.contains(&format!(".{stem}")))
        .map(|(_, why)| *why)
        .or(Some("it is not a DiT block leaf"))
}

/// Resolve `cfg.lora_target_modules` to full dotted paths on `surface` ([`adapter_surface`]), in
/// block order. Empty: every block's [`BLOCK_TARGETS`]. Otherwise each entry suffix-matches
/// anywhere, PEFT's `target_modules` rule, over the trainable block leaves; an entry that matches
/// only modules the run cannot train ([`UNTRAINABLE_MODULES`]) is a typed refusal naming them and
/// the reason, and a list that matches nothing is refused too — never a silently smaller adapter.
pub fn resolve_target_paths(
    surface: &[String],
    cfg: &TrainingConfig,
) -> gen_core::Result<Vec<String>> {
    let suffixes: Vec<&str> = if cfg.lora_target_modules.is_empty() {
        BLOCK_TARGETS.to_vec()
    } else {
        cfg.lora_target_modules.iter().map(String::as_str).collect()
    };
    let mut paths = Vec::new();
    let mut refused = Vec::new();
    for suffix in &suffixes {
        let matched: Vec<&String> = surface
            .iter()
            .filter(|p| p.as_str() == *suffix || p.ends_with(&format!(".{suffix}")))
            .collect();
        let trainable: Vec<&String> = matched
            .iter()
            .copied()
            .filter(|p| untrainable_reason(p).is_none())
            .collect();
        if trainable.is_empty() {
            refused.extend(
                matched
                    .iter()
                    .filter_map(|p| untrainable_reason(p).map(|why| format!("{p} ({why})"))),
            );
        }
        paths.extend(trainable.into_iter().cloned());
    }
    // Block order, one entry per module, whatever order the suffixes came in.
    let paths: Vec<String> = surface
        .iter()
        .filter(|p| paths.contains(p))
        .cloned()
        .collect();
    if !refused.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{TRAINER_ID} trainer: lora_target_modules {:?} select module(s) a MiniMax-H3-Image run \
             cannot train: {}; target the DiT block leaves ({}) instead",
            cfg.lora_target_modules,
            refused.join("; "),
            BLOCK_TARGETS.join(", ")
        )));
    }
    if paths.is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{TRAINER_ID} trainer: lora_target_modules {:?} matched no trainable module (default: \
             every block's {})",
            cfg.lora_target_modules,
            BLOCK_TARGETS.join("/")
        )));
    }
    Ok(paths)
}

/// The DiT block a block-leaf path belongs to (`transformer_blocks.{i}.…` → `i`).
pub fn target_block(path: &str) -> Option<usize> {
    path.strip_prefix("transformer_blocks.")?
        .split_once('.')?
        .0
        .parse()
        .ok()
}

/// The distilled step one micro-step trains: its index (which output head and which AdaLN
/// timestep), its noise level `σ` and its timestep `t = 1 − σ`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainingStep {
    pub index: usize,
    pub sigma: f32,
    pub timestep: f32,
}

/// The distilled step nearest the drawn noise level `t ∈ [0, 1]` — the configured timestep
/// distribution, quantized onto the six published levels the heads were distilled at. Ties go to
/// the noisier step. Pure, so a resumed run draws the same steps as an uninterrupted one.
pub fn training_step(t: f32) -> TrainingStep {
    let sigmas = sampler_sigmas();
    let steps = FIXED_STEPS as usize;
    let t = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let index = (0..steps)
        .min_by(|&a, &b| {
            (sigmas[a] - t)
                .abs()
                .total_cmp(&(sigmas[b] - t).abs())
                .then(a.cmp(&b))
        })
        .unwrap_or(0);
    let sigma = sigmas[index];
    TrainingStep {
        index,
        sigma,
        timestep: 1.0 - sigma,
    }
}

/// `(data, noise)` coefficients of the training interpolant at `sigma`:
/// `x_σ = data·x0 + noise·ε` with `data = 1 − σ`, `noise = σ`. The regression target is
/// `x0 − ε`, the velocity the released sampler steps with.
pub fn training_mix(sigma: f32) -> (f32, f32) {
    (1.0 - sigma, sigma)
}

/// The square-or-aspect training canvas of one target at `edge`: a captioned item trains on its
/// centre square at `edge²`; an edit target keeps its aspect at about `edge²` pixels, so it stays
/// aligned with its references. Both sides floor to [`TRAINING_CANVAS_STRIDE`] (at least one
/// stride).
pub fn training_canvas(edit: bool, (width, height): (u32, u32), edge: u32) -> (u32, u32) {
    let snap = |v: f64| {
        let s = f64::from(TRAINING_CANVAS_STRIDE);
        ((v / s).floor() * s).max(s) as u32
    };
    if !edit || width == 0 || height == 0 {
        let e = snap(f64::from(edge));
        return (e, e);
    }
    let area = f64::from(edge) * f64::from(edge);
    let aspect = f64::from(width) / f64::from(height);
    let h = (area / aspect).sqrt();
    (snap(h * aspect), snap(h))
}

// ── the training-step memory envelope ───────────────────────────────────────────────────────────

/// Residual-width (`[S, inner]`) tensors one block's backward holds for its **attention** half: the
/// normed input and the modulated `h`, q/k/v, q/k after the per-head RMS norm and after the rotary
/// (whose float32 planes count double at bf16 width), the attention output and `to_out`. A
/// structural count off the block forward, not a measurement — the same convention as the
/// Qwen-Image 2.1 trainer's.
pub const ATTENTION_SAVED_HIDDEN: u64 = 14;
/// `[S, hidden]` tensors the feed-forward half holds regardless of its width: the residual after
/// attention, the modulated `h`, and the output.
pub const FFN_SAVED_HIDDEN: u64 = 3;
/// `[S, ffn]` tensors the SwiGLU holds: the `2·ffn` projection, `silu(gate)` and the product.
pub const FFN_SAVED_WIDE: u64 = 4;
/// `[S, hidden]` modulation rows a block gathers from the AdaLN table (`shift`/`scale`/`gate`, for
/// attention and for the MLP).
pub const MODULATION_ROWS: u64 = 6;
/// Score-sized (`[B, heads, Sq, Sk]`) arrays one attention call's backward ever creates on MLX
/// Metal — derived from MLX v0.32.0, the version the pinned mlx-rs links:
///
/// * training runs the unfused SDPA for the forward and the backward:
///   `backend/metal/scaled_dot_product_attention.cpp:601-605` (`is_training` ⇒ fallback) and
///   `:792-794` (`ScaledDotProductAttentionVJP::use_fallback` is always true), so the VJP is the
///   autodiff of `fast.cpp:717-787`'s fallback;
/// * that forward creates two: `fast.cpp:733` `scores = matmul(q·scale, kᵀ)` (`q·scale` at
///   `:724` is q-sized) and `fast.cpp:776` `softmax(scores)` — this DiT passes no mask, no sinks and
///   no GQA repeat (`mlx-gen-minimax-h3/src/dit/layers.rs:358` calls `sdpa_bounded_bhsd(.., None,
///   ..)` with equal q/kv heads), and it has no dropout;
/// * the backward creates four: `Matmul::vjp` of `softmax · v` (`primitives.cpp:3010`,
///   `dP = g · vᵀ`), and `Softmax::vjp` (`primitives.cpp:5355-5358`) `s·dP`, `s·Σ(s·dP)` and their
///   difference — the row sum is `[.., Sq, 1]`; the `q·kᵀ` grads (`primitives.cpp:3010/3013`) read
///   that difference through a transpose view and are q/k-sized.
///
/// At the softmax VJP five of the six are live (`s`, `dP`, `s·dP`, `s·Σ`, the difference); the
/// sixth, the pre-softmax scores, can still be pinned by an uncompleted command buffer. An array
/// a buffer pins is one of these six, so six bound the score bytes one call holds — the earlier
/// "five live plus eleven buffers each pinning a whole score call" pricing counted the same arrays
/// twice. Correction, not loosening: every array is still charged once, at full size.
pub const SCORE_ARRAYS_PER_ATTENTION_VJP: u64 = 6;
/// Command buffers MLX keeps in flight (`transforms.cpp`'s ten) plus the one being encoded: under
/// chunked attention, the most earlier chunks' score arrays an uncompleted buffer can still pin.
pub const PINNED_COMMAND_BUFFERS: u64 = 11;
/// f32 buffers every trainable element carries whatever the optimizer: the factor, its gradient
/// and the gradient-accumulation buffer.
pub const TRAINABLE_BASE_BUFFERS: u64 = 3;

/// The H3-Image DiT geometry the envelope counts in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DitGeometry {
    pub layers: u64,
    pub hidden: u64,
    pub heads: u64,
    pub head_dim: u64,
    pub ffn: u64,
    /// Channels of one packed image row (`in_channels · prod(patch)`, 96 shipped).
    pub patch_dim: u64,
}

impl DitGeometry {
    /// The shipped `transformer_diffusers/` geometry.
    pub const SHIPPED: Self = Self {
        layers: 50,
        hidden: 5376,
        heads: 56,
        head_dim: 128,
        ffn: 14336,
        patch_dim: 96,
    };

    fn inner(&self) -> u64 {
        self.heads * self.head_dim
    }
}

/// The DiT weight bytes the train phase holds, read off the staged tier by the backend.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DitWeightBytes {
    /// I/O projections, the six heads and the token refiner — resident throughout.
    pub trunk: u64,
    /// Each block's body.
    pub block_body: Vec<u64>,
    /// Each block's AdaLN projection — read once by the precompute.
    pub block_adaln: Vec<u64>,
    /// The precomputed modulation tables (every block, every distinct timestep).
    pub modulation_tables: u64,
}

/// One micro-step's shape and run settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainStepShape {
    /// Packed caption rows.
    pub text_rows: u64,
    /// An edit's reference rows.
    pub reference_rows: u64,
    /// Generated (target) rows.
    pub image_rows: u64,
    /// Activation width (2 bf16, 4 f32).
    pub width: u64,
    /// Trainable adapter elements.
    pub trainable_elements: u64,
    /// f32 optimizer-state elements per trainable element.
    pub optimizer_state_per_element: u64,
    /// LoKr dense-delta elements: the largest block's (streamed) and all blocks' (resident).
    pub lokr_delta_block_max: u64,
    pub lokr_delta_total: u64,
    /// `true`: the blocks stream one at a time and the backward recomputes each from its saved
    /// input; `false`: the whole DiT is resident and every block's activations are retained.
    pub checkpointed: bool,
    /// Heads one attention chunk covers (`heads` when attention is not chunked). Chunked training
    /// attention recomputes each chunk in the backward, so no call ever holds more than one chunk's
    /// scores (E20).
    pub attention_heads_per_chunk: u64,
}

impl TrainStepShape {
    pub fn sequence(&self) -> u64 {
        self.text_rows + self.reference_rows + self.image_rows
    }
}

/// The train phase's terms, each a structural upper bound (derived, unmeasured).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrainStepBytes {
    /// DiT weights live at the peak: the trunk, the modulation tables and one block (streamed) or
    /// every block (resident) — or the precompute's one AdaLN projection, when that is larger.
    pub weights: u64,
    /// Factors, gradients, accumulation and optimizer state, f32.
    pub adapter_state: u64,
    /// The forward's retained activations: every block input (streamed) or every block's saved
    /// tensors and softmax output (resident).
    pub retained: u64,
    /// One block's backward working set: its saved tensors, modulation rows and the attention
    /// call's score arrays ([`SCORE_ARRAYS_PER_ATTENTION_VJP`]).
    pub block_backward: u64,
    /// The output head's tail: `norm_out`, head and loss over the image rows, f32.
    pub tail: u64,
    /// LoKr dense deltas live at once (bf16).
    pub lokr_deltas: u64,
    /// The AdaLN precompute's peak, before the first step: the trunk, the tables and one AdaLN
    /// projection (streamed) or the whole DiT (resident). Not co-resident with the step's terms.
    pub precompute: u64,
}

impl TrainStepBytes {
    /// The phase peak: the step's terms together, or the precompute when that is larger.
    pub fn total(&self) -> u64 {
        let step = [
            self.weights,
            self.adapter_state,
            self.retained,
            self.block_backward,
            self.tail,
            self.lokr_deltas,
        ]
        .into_iter()
        .fold(0u64, u64::saturating_add);
        step.max(self.precompute)
    }
}

/// The train phase of one micro-step on `geometry` with `weights` — the term every backend adds its
/// own pipelining and eval slack to. See the constants for what each count holds.
pub fn train_step_bytes(
    geometry: &DitGeometry,
    weights: &DitWeightBytes,
    shape: &TrainStepShape,
) -> TrainStepBytes {
    let s = shape.sequence();
    let a = shape.width;
    let max = |v: &[u64]| v.iter().copied().max().unwrap_or(0);
    let sum = |v: &[u64]| v.iter().copied().fold(0u64, u64::saturating_add);
    let row = s * geometry.hidden * a;
    let block_saved = (ATTENTION_SAVED_HIDDEN * s * geometry.inner()
        + FFN_SAVED_HIDDEN * s * geometry.hidden
        + FFN_SAVED_WIDE * s * geometry.ffn
        + MODULATION_ROWS * s * geometry.hidden)
        * a;
    let chunk_heads = shape
        .attention_heads_per_chunk
        .clamp(1, geometry.heads.max(1));
    let chunked = chunk_heads < geometry.heads;
    let chunk_scores = chunk_heads * s * s * a;
    // Unchunked: the six arrays of the one call, at full size, and the forward's saved softmax per
    // retained block. Chunked: the backward's six for the chunk being differentiated plus at most
    // the eleven a pending buffer can still pin from earlier chunks; a chunk saves only its q/k/v.
    let (score_bytes, saved_scores) = if chunked {
        (
            (SCORE_ARRAYS_PER_ATTENTION_VJP + PINNED_COMMAND_BUFFERS) * chunk_scores,
            0,
        )
    } else {
        (SCORE_ARRAYS_PER_ATTENTION_VJP * chunk_scores, chunk_scores)
    };
    let (weights_bytes, precompute) = if shape.checkpointed {
        (
            weights.trunk + weights.modulation_tables + max(&weights.block_body),
            weights.trunk + weights.modulation_tables + max(&weights.block_adaln),
        )
    } else {
        // Resident: the precompute runs with every block whole, then evicts the projections.
        let step = weights.trunk + weights.modulation_tables + sum(&weights.block_body);
        (step, step + sum(&weights.block_adaln))
    };
    let retained = if shape.checkpointed {
        (geometry.layers + 1) * row
    } else {
        geometry.layers * (block_saved + saved_scores)
    };
    let image = shape.reference_rows + shape.image_rows;
    TrainStepBytes {
        weights: weights_bytes,
        adapter_state: shape.trainable_elements
            * (TRAINABLE_BASE_BUFFERS + shape.optimizer_state_per_element)
            * 4,
        retained,
        block_backward: block_saved + score_bytes,
        tail: 3 * row + image * geometry.patch_dim * 4 * 4,
        lokr_deltas: 2 * if shape.checkpointed {
            2 * shape.lokr_delta_block_max
        } else {
            shape.lokr_delta_total
        },
        precompute,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(targets: &[&str]) -> TrainingConfig {
        TrainingConfig {
            lora_target_modules: targets.iter().map(|s| s.to_string()).collect(),
            ..TrainingConfig::default()
        }
    }

    fn surface() -> Vec<String> {
        adapter_surface(50, 2, 6)
    }

    #[test]
    fn the_default_targets_are_every_block_leaf_in_block_order() {
        let paths = resolve_target_paths(&surface(), &cfg(&[])).unwrap();
        assert_eq!(paths.len(), 50 * 6);
        assert_eq!(paths[0], "transformer_blocks.0.attn.to_q");
        assert_eq!(paths[299], "transformer_blocks.49.ff.net.2");
        // Block order: every path's block is non-decreasing.
        let blocks: Vec<usize> = paths.iter().map(|p| target_block(p).unwrap()).collect();
        assert!(blocks.windows(2).all(|w| w[0] <= w[1]));
        assert!(!paths.iter().any(|p| p.starts_with("token_refiner")));
    }

    #[test]
    fn an_explicit_suffix_matches_block_leaves_only_where_trainable() {
        let q = resolve_target_paths(&surface(), &cfg(&["to_q"])).unwrap();
        assert_eq!(q.len(), 50, "the refiner's to_q is not trained");
        // A suffix that lands on the refiner is refused, naming why.
        let err = resolve_target_paths(&surface(), &cfg(&["refiner_blocks.0.attn.to_q"]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("refined once per item"), "{err}");
        for (target, why) in [
            ("adaln_proj.linear", "precomputed once"),
            ("time_embedder.linear_1", "timestep MLP"),
            ("norm_out.linear", "final AdaLN"),
            ("context_embedder", "embedded once"),
            ("proj_in", "image input projection"),
            ("pdd_proj_out.3", "output heads"),
        ] {
            let err = resolve_target_paths(&surface(), &cfg(&[target]))
                .unwrap_err()
                .to_string();
            assert!(err.contains(why), "{target}: {err}");
        }
        assert!(resolve_target_paths(&surface(), &cfg(&["nonexistent"])).is_err());
    }

    /// The contract the trainer regresses: the interpolant at the step's sigma, the target
    /// `x0 − ε`, and the released sampler's two forms recover the data / the next interpolant.
    #[test]
    fn the_training_target_is_the_velocity_the_released_sampler_steps_with() {
        let sigmas = sampler_sigmas();
        let (x0, eps) = (0.37f32, -1.21f32);
        for i in 0..FIXED_STEPS as usize {
            let step = training_step(sigmas[i]);
            assert_eq!(step.index, i);
            assert_eq!(step.sigma, sigmas[i]);
            assert_eq!(step.timestep, 1.0 - sigmas[i]);
            let (d, n) = training_mix(step.sigma);
            let x = d * x0 + n * eps;
            let v = x0 - eps;
            // LCM's data estimate: x0 = x + σ·v.
            assert!((x + step.sigma * v - x0).abs() < 1e-6);
            // Released Euler: x + (σ_i − σ_{i+1})·v is the interpolant at σ_{i+1}.
            let next = x + (sigmas[i] - sigmas[i + 1]) * v;
            let (dn, nn) = training_mix(sigmas[i + 1]);
            assert!((next - (dn * x0 + nn * eps)).abs() < 1e-6);
        }
    }

    #[test]
    fn a_drawn_level_trains_at_the_nearest_published_step() {
        assert_eq!(training_step(1.0).index, 0);
        assert_eq!(training_step(0.0).index, 5);
        assert_eq!(training_step(0.76).index, 3);
        assert_eq!(training_step(0.4).index, 4);
        assert_eq!(training_step(f32::NAN).index, 0);
        // The midpoint between two levels goes to the noisier one.
        let s = sampler_sigmas();
        assert_eq!(training_step((s[3] + s[4]) / 2.0).index, 3);
    }

    #[test]
    fn the_training_canvas_is_square_for_captions_and_keeps_an_edits_aspect() {
        assert_eq!(training_canvas(false, (3000, 2000), 2048), (2048, 2048));
        assert_eq!(training_canvas(false, (3000, 2000), 1000), (992, 992));
        let (w, h) = training_canvas(true, (3000, 2000), 2048);
        assert_eq!((w % 32, h % 32), (0, 0));
        assert!((f64::from(w) / f64::from(h) - 1.5).abs() < 0.05);
        assert!(u64::from(w) * u64::from(h) <= 2048 * 2048);
        let (w, h) = training_canvas(true, (10, 4000), 16);
        assert_eq!((w, h % 32), (32, 0));
    }

    #[test]
    fn the_descriptor_and_defaults_match_the_generator() {
        let d = trainer_descriptor("mlx");
        assert_eq!((d.id, d.family, d.backend), (TRAINER_ID, FAMILY, "mlx"));
        assert_eq!(d.max_reference_images, 16);
        assert!(d.techniques.resolution_buckets && d.techniques.weight_noise);
        assert!(!d.techniques.subject_mask_loss && !d.supports_control);
        let c = default_training_config();
        assert_eq!(c.resolution, 2048);
        assert!(c.gradient_checkpointing);
        assert!(ADAPTER_PROVENANCE
            .iter()
            .any(|(k, v)| *k == "license" && v.contains("minimax-h3-image")));
        assert!(!LICENSE_NOTICE.is_empty());
    }

    #[test]
    fn the_request_floor_refuses_control_and_sub_stride_edges() {
        let mut req = TrainingRequest {
            items: vec![TrainingItem::captioned("/a.png".into(), "a cat".into())],
            config: default_training_config(),
            output_dir: "unused".into(),
            file_name: "ok.safetensors".into(),
            trigger_words: vec![],
            cancel: gen_core::CancelFlag::new(),
        };
        assert!(validate_request(&req).is_ok());
        req.config.resolution = 16;
        assert!(validate_request(&req).is_err());
        req.config.resolution = 2048;
        req.items[0].control_image_path = Some("/c.png".into());
        assert!(validate_request(&req).is_err());
    }

    fn weights() -> DitWeightBytes {
        DitWeightBytes {
            trunk: 2_000_000_000,
            block_body: vec![760_000_000; 50],
            block_adaln: vec![520_000_000; 50],
            modulation_tables: 70_000_000,
        }
    }

    fn shape(checkpointed: bool) -> TrainStepShape {
        TrainStepShape {
            text_rows: 256,
            reference_rows: 0,
            image_rows: 4096,
            width: 2,
            trainable_elements: 50 * 16 * 200_000,
            optimizer_state_per_element: 2,
            lokr_delta_block_max: 0,
            lokr_delta_total: 0,
            checkpointed,
            attention_heads_per_chunk: 56,
        }
    }

    /// The streamed run holds one block of weights and only the block inputs; the resident one
    /// holds the whole DiT and every block's saved tensors, so it is strictly larger.
    #[test]
    fn checkpointed_streaming_bounds_weights_to_one_block_and_activations_to_block_inputs() {
        let g = DitGeometry::SHIPPED;
        let w = weights();
        let streamed = train_step_bytes(&g, &w, &shape(true));
        let s = 4352u64;
        assert_eq!(streamed.weights, 2_000_000_000 + 70_000_000 + 760_000_000);
        assert_eq!(streamed.retained, 51 * s * 5376 * 2);
        assert_eq!(
            streamed.block_backward,
            (14 * s * 7168 + 3 * s * 5376 + 4 * s * 14336 + 6 * s * 5376) * 2 + 6 * 56 * s * s * 2
        );
        // Chunked to one head: 6 + 11 one-head score arrays, and no block saves its scores.
        let one = TrainStepShape {
            attention_heads_per_chunk: 1,
            ..shape(true)
        };
        let chunked = train_step_bytes(&g, &w, &one);
        assert_eq!(
            chunked.block_backward,
            (14 * s * 7168 + 3 * s * 5376 + 4 * s * 14336 + 6 * s * 5376) * 2 + 17 * s * s * 2
        );
        assert!(chunked.total() < streamed.total());
        let resident = train_step_bytes(&g, &w, &shape(false));
        assert!(resident.weights > 50 * 760_000_000);
        assert!(resident.retained > 20 * streamed.retained);
        assert!(resident.total() > streamed.total());
        let wide = train_step_bytes(
            &g,
            &w,
            &TrainStepShape {
                width: 4,
                ..shape(true)
            },
        );
        assert!(wide.total() > streamed.total());
    }
}
