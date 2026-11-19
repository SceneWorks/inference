//! **MiniMax-H3-Image LoRA/LoKr training**, the backend-neutral half (SW-7097): what the MLX
//! trainer (`mlx_gen_minimax_h3::training`) and the Candle one (SW-7098) must agree on so an
//! adapter trained on either loads, trains and resumes identically on both.
//!
//! Tensor-free, like the rest of this crate. It owns:
//!
//! * the **trainer identity** — [`TRAINER_ID`], [`trainer_descriptor`], the adapter provenance
//!   ([`ADAPTER_PROVENANCE`], [`EDIT_ADAPTER_MARKER`]) and the run defaults
//!   ([`default_training_config`]: 2048², gradient checkpointing on);
//! * the **trainable surface** — [`resolve_target_paths`]: the six leaves of every DiT block and
//!   of every token-refiner block by default (ai-toolkit's transformer-only surface for this
//!   architecture), suffix-matched like PEFT's `target_modules`, with a typed refusal for the
//!   modules a distilled H3-Image run computes once per request or keeps frozen;
//! * the **flow-matching training contract** — [`training_step`] and [`training_mix`], the noise
//!   level, head and regression target one micro-step trains on;
//! * the **training canvas** — [`training_canvas`]: every target keeps its aspect at about the
//!   bucket edge's area, never upscaled past its own;
//! * the **training-step memory envelope** — [`train_step_bytes`], the structural count both
//!   backends price their train phase with, and [`TrainingAdapterBytes`], the frozen de-distill
//!   training adapter's share of it.
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
//! from it and training the head whose equal-width slice of `t` the draw falls in
//! ([`training_step`]): head `i` owns `t ∈ ((5 − i)/6, (6 − i)/6]`, so a uniform draw trains every
//! head exactly as often as the six-step render runs it, and a biased draw still favours the noisy
//! (`t → 1`, head 0) or clean (`t → 0`, head 5) end — but never starves a head the way snapping to
//! the *nearest* published σ did (σ is 1.0, 0.9675, 0.8875, … so a sigmoid draw reached heads 0
//! and 1 on well under 1 % of the steps, while the shared blocks the LoRA changes still feed them
//! at every render).

use gen_core::h3_adapter_memory::H3AdapterMemoryProfile;
use gen_core::h3_image::{sampler_sigmas, FIXED_STEPS, MAX_REFERENCES, SIZE_STRIDE};
use gen_core::license::components::MINIMAX_H3_IMAGE;

use gen_core::generator::Modality;

use super::*;

/// Registry id of the H3-Image trainer: the generator id of the base model it adapts, so a trained
/// adapter's `baseModel` names the route that loads it.
pub const TRAINER_ID: &str = "minimax_h3_image";

/// The descriptor family (the generator's own).
pub const FAMILY: &str = "minimax_h3_image";

/// The default training edge (a canvas of about `edge²` pixels): H3-Image's native 4 MP
/// generation canvas ([`gen_core::h3_image::MIN_GENERATION_AREA_DEFAULT`] is `2048²`).
pub const DEFAULT_TRAINING_RESOLUTION: u32 = 2048;

/// Every training canvas edge is a multiple of this: one VAE cell (16 px) times the DiT's 2×2
/// patch — the generator's own [`SIZE_STRIDE`].
pub const TRAINING_CANVAS_STRIDE: u32 = SIZE_STRIDE;

/// The leaves of every DiT block — and of every token-refiner block — a default run trains: the
/// block's four attention projections and its SwiGLU pair, in the published diffusers spelling the
/// H3-Image LoRA loader addresses.
///
/// The default surface is ai-toolkit's for `minimax_h3_image` (`transformer_only` over the
/// transformer's `blocks`, which matches both `blocks.{i}` and `token_refiner.blocks.{i}`, with
/// `adaln_proj` ignored): every linear inside those blocks, and nothing outside them. The
/// refiner's leaves carry the caption rows, so training them lets the adapter bind its trigger
/// text, as an ai-toolkit run of the same architecture does.
pub const BLOCK_TARGETS: [&str; 6] = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
];

/// The adapter-addressable modules (SW-7096) a training run cannot train, and why. The first three
/// run once per request, before the per-step forward, from the six timesteps every step shares, so
/// their outputs are constants of the step's graph and no gradient reaches them. The rest sit
/// outside the transformer and refiner blocks, which is the surface ai-toolkit trains for this
/// architecture too.
pub const UNTRAINABLE_MODULES: [(&str, &str); 6] = [
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
        // cache every item once per edge; the training adapter (a frozen de-distill LoRA such as
        // `ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter`) is installed on the training
        // DiT only — previews render through a fresh generator without it, and the saved adapter
        // holds the trainable factors alone (see [`TrainingAdapterBytes`]). Every other technique
        // is refused by the shared floor.
        techniques: TrainingTechniques {
            resolution_buckets: true,
            training_adapter: true,
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

/// The leaf of a DiT-block or refiner-block path (`…blocks.{i}.<leaf>` → `<leaf>`), when `path`
/// is one of [`BLOCK_TARGETS`] under either container.
fn trainable_block_leaf(path: &str) -> Option<&str> {
    let rest = path
        .strip_prefix("transformer_blocks.")
        .or_else(|| path.strip_prefix(REFINER_BLOCKS_PREFIX))?;
    let (index, leaf) = rest.split_once('.')?;
    index.parse::<usize>().ok()?;
    BLOCK_TARGETS.contains(&leaf).then_some(leaf)
}

/// Why `path` cannot be trained, or `None` for a trainable block or refiner-block leaf.
fn untrainable_reason(path: &str) -> Option<&'static str> {
    if trainable_block_leaf(path).is_some() {
        return None;
    }
    UNTRAINABLE_MODULES
        .iter()
        .find(|(stem, _)| path.starts_with(stem) || path.contains(&format!(".{stem}")))
        .map(|(_, why)| *why)
        .or(Some("it is not a DiT block leaf"))
}

/// Resolve `cfg.lora_target_modules` to full dotted paths on `surface` ([`adapter_surface`]), in
/// surface order (the DiT blocks, then the refiner blocks). Empty: [`BLOCK_TARGETS`] of every DiT
/// block and every refiner block. Otherwise each entry suffix-matches
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
             cannot train: {}; target the DiT or token-refiner block leaves ({}) instead",
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

/// The DiT block a block-leaf path belongs to (`transformer_blocks.{i}.…` → `i`); `None` for a
/// refiner-block or trunk path.
pub fn target_block(path: &str) -> Option<usize> {
    path.strip_prefix("transformer_blocks.")?
        .split_once('.')?
        .0
        .parse()
        .ok()
}

/// The container prefix of a token-refiner block path.
pub const REFINER_BLOCKS_PREFIX: &str = "token_refiner.refiner_blocks.";

/// The token-refiner block a refiner-leaf path belongs to
/// (`token_refiner.refiner_blocks.{i}.…` → `i`).
pub fn refiner_block(path: &str) -> Option<usize> {
    path.strip_prefix(REFINER_BLOCKS_PREFIX)?
        .split_once('.')?
        .0
        .parse()
        .ok()
}

/// Whether any resolved target is a token-refiner leaf — the run then differentiates the caption's
/// refiner forward every step instead of refining each caption once.
pub fn trains_refiner(paths: &[String]) -> bool {
    paths.iter().any(|p| refiner_block(p).is_some())
}

/// The frozen **training adapter**'s footprint on the training DiT ([`TrainingConfig::training_adapter`]):
/// H3-Image is a six-step distilled model, and plain flow-matching fine-tuning un-distills it (a
/// long LoRA samples blurry, with ghosted limbs). A de-distill LoRA applied to the base at strength
/// 1.0 for every training forward/backward — and absent from previews and the saved adapter — lets
/// the user's adapter learn the concept instead of undoing the distillation (the Z-Image
/// precedent, sc-25213).
///
/// Both backends install the file through their own **inference** H3-Image adapter install (the
/// same key mapping a render uses: kohya, ComfyUI fused `qkv_proj`, diffusers/PEFT, LoKr) as a
/// frozen forward-time residual over the untouched — possibly packed — base. These are the bytes
/// that install holds, reduced from the inference adapter memory profile
/// ([`H3AdapterMemoryProfile`], header-only) to what a block-streaming trainer needs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrainingAdapterBytes {
    /// Every installed factor plus its prepared caches.
    pub resident: u64,
    /// The share on modules outside the 50 blocks (refiner, I/O projections, heads) — resident
    /// with the trunk whatever the block residency.
    pub trunk: u64,
    /// The largest single block's share — what one streamed block window adds.
    pub block_max: u64,
    /// Bytes per sequence row the residuals' forward keeps alive (their intermediates and outputs,
    /// which autograd retains for the backward): the largest block's, and every block's.
    pub rows_block_max: u64,
    pub rows_total: u64,
}

impl TrainingAdapterBytes {
    /// Reduce an inference adapter memory profile of the training adapter alone.
    pub fn from_profile(profile: &H3AdapterMemoryProfile) -> Self {
        let mut blocks = std::collections::BTreeMap::<usize, (u64, u64)>::new();
        let mut trunk = 0u64;
        let mut rows_total = 0u64;
        for p in &profile.projections {
            rows_total = rows_total.saturating_add(p.forward_bytes_per_row);
            match target_block(&p.target.path) {
                Some(block) => {
                    let entry = blocks.entry(block).or_default();
                    entry.0 = entry.0.saturating_add(p.resident_bytes);
                    entry.1 = entry.1.saturating_add(p.forward_bytes_per_row);
                }
                None => trunk = trunk.saturating_add(p.resident_bytes),
            }
        }
        let projected: u64 = profile
            .projections
            .iter()
            .fold(0u64, |t, p| t.saturating_add(p.resident_bytes));
        Self {
            // The profile's own total is authoritative; never price below the per-projection sum.
            resident: profile.resident_bytes.max(projected),
            trunk,
            block_max: blocks.values().map(|b| b.0).max().unwrap_or(0),
            rows_block_max: blocks.values().map(|b| b.1).max().unwrap_or(0),
            rows_total,
        }
    }

    /// The bytes held on the training device: the trunk share plus one block window (`streamed`),
    /// or everything (resident blocks).
    pub fn device(&self, streamed: bool) -> u64 {
        if streamed {
            self.trunk.saturating_add(self.block_max)
        } else {
            self.resident
        }
    }

    /// The bytes a streamed run keeps on the host beside its block stack (the blocks' share).
    pub fn host(&self, streamed: bool) -> u64 {
        if streamed {
            self.resident.saturating_sub(self.trunk)
        } else {
            0
        }
    }

    /// Bytes per sequence row the forward retains: one live block (checkpointed) or every block.
    pub fn rows(&self, checkpointed: bool) -> u64 {
        if checkpointed {
            self.rows_block_max
        } else {
            self.rows_total
        }
    }
}

/// The distilled step one micro-step trains: its index (which output head and which AdaLN
/// timestep), its noise level `σ` and its timestep `t = 1 − σ`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainingStep {
    pub index: usize,
    pub sigma: f32,
    pub timestep: f32,
}

/// The distilled step a drawn noise level `t ∈ [0, 1]` trains: the configured timestep
/// distribution, mapped onto the six heads by **equal quantile slices of `t`** —
/// `index = min(⌊(1 − t)·6⌋, 5)`, so `t → 1` (noisiest) trains head 0 and `t → 0` head 5. A
/// uniform draw trains each head exactly 1/6 of the time (the render runs each once); a biased draw
/// shifts that mass toward one end without ever starving a head. (Snapping to the *nearest*
/// published σ — the grid is 1.0, 0.9675, 0.8875, 0.75, … — left heads 0 and 1 almost untrained
/// under a sigmoid draw, while the adapted shared blocks still feed them at every render.) A
/// non-finite draw trains head 0. Pure, so a resumed run draws the same steps as an uninterrupted
/// one.
pub fn training_step(t: f32) -> TrainingStep {
    let steps = FIXED_STEPS as usize;
    let t = if t.is_finite() {
        f64::from(t.clamp(0.0, 1.0))
    } else {
        1.0
    };
    let index = (((1.0 - t) * steps as f64).floor() as usize).min(steps - 1);
    distilled_step(index)
}

/// Distilled step `index` (`0..6`, clamped): its head, its published noise level `σ` and its
/// AdaLN timestep `t = 1 − σ`.
pub fn distilled_step(index: usize) -> TrainingStep {
    let sigmas = sampler_sigmas();
    let index = index.min(FIXED_STEPS as usize - 1);
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

/// The training canvas `(width, height)` of one target whose source is `width × height`, at bucket
/// edge `edge` — one rule for captioned items and edit targets alike: the target keeps its own
/// aspect at about `edge²` pixels, and is never upscaled past its source (a source smaller than
/// `edge²` trains at its own area). Both sides floor to [`TRAINING_CANVAS_STRIDE`] (at least one
/// stride), so the canvas never exceeds the source's area nor `edge²`.
///
/// The whole image is trained, resized (never cropped) onto the canvas — the aspect drift the
/// stride floor introduces is under one stride per side. An edit's references are conditioned at
/// their own sizes by the render's conditioning phase, independently of the target's canvas, and
/// the target keeps its aspect either way, so capping its area keeps references and target
/// aligned exactly as before. A degenerate (zero) side falls back to the square `edge` canvas.
pub fn training_canvas((width, height): (u32, u32), edge: u32) -> (u32, u32) {
    // The shared aspect-preserving rule every trainer uses (SW-7172), at H3's stride. H3 keeps the
    // unbounded-aspect form: its references condition at their own sizes, so no 4:1 source crop.
    crate::canvas::aspect_canvas((width, height), edge, TRAINING_CANVAS_STRIDE)
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
    /// `true`: the backward recomputes each block from its saved input, so only the block inputs
    /// are retained; `false`: the whole DiT is resident and every block's activations are retained.
    pub checkpointed: bool,
    /// Under `checkpointed`: `true` keeps every block body resident on the device (no per-step
    /// weight transfer), `false` streams the blocks through a one-block device window. The AdaLN
    /// precompute runs streamed either way (one `adaln_proj` at a time), and the bodies move to the
    /// device after it. Ignored when `checkpointed` is `false` (that run is always resident).
    pub resident_blocks: bool,
    /// Heads one attention chunk covers (`heads` when attention is not chunked). Chunked training
    /// attention recomputes each chunk in the backward, so no call ever holds more than one chunk's
    /// scores (E20).
    pub attention_heads_per_chunk: u64,
    /// Token-refiner blocks the step differentiates ([`trains_refiner`]): their forward over the
    /// caption rows is retained whole for the backward (it is not checkpointed). `0` when no
    /// refiner leaf is a target and the caption rows are a constant of the step.
    pub refiner_layers: u64,
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
    /// every block body (resident, checkpointed or not).
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
        // The precompute streams one `adaln_proj` at a time in both checkpointed modes; resident
        // bodies move to the device only after it, so they are never co-resident with it.
        let bodies = if shape.resident_blocks {
            sum(&weights.block_body)
        } else {
            max(&weights.block_body)
        };
        (
            weights.trunk + weights.modulation_tables + bodies,
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
    // A trained refiner's forward over the caption rows, retained for the backward: each refiner
    // block's saved tensors at the text rows (the refiner blocks have the DiT blocks' widths) and
    // its attention's saved scores, live across the whole step.
    let t = shape.text_rows;
    let refiner_block_saved = (ATTENTION_SAVED_HIDDEN * t * geometry.inner()
        + FFN_SAVED_HIDDEN * t * geometry.hidden
        + FFN_SAVED_WIDE * t * geometry.ffn)
        * a
        + SCORE_ARRAYS_PER_ATTENTION_VJP * geometry.heads * t * t * a;
    let retained = retained.saturating_add(shape.refiner_layers * refiner_block_saved);
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

    /// ai-toolkit's surface for this architecture: every leaf of the 50 DiT blocks, then every leaf
    /// of the 2 refiner blocks — nothing outside a block. Mutation: drop the refiner prefix from
    /// [`trainable_block_leaf`] ⇒ the default is 300 paths and the count assert fails.
    #[test]
    fn the_default_targets_are_every_dit_and_refiner_block_leaf() {
        let paths = resolve_target_paths(&surface(), &cfg(&[])).unwrap();
        assert_eq!(paths.len(), (50 + 2) * 6);
        assert_eq!(paths[0], "transformer_blocks.0.attn.to_q");
        assert_eq!(paths[299], "transformer_blocks.49.ff.net.2");
        assert_eq!(paths[300], "token_refiner.refiner_blocks.0.attn.to_q");
        assert_eq!(paths[311], "token_refiner.refiner_blocks.1.ff.net.2");
        // Surface order: the DiT blocks non-decreasing, then the refiner blocks non-decreasing.
        let blocks: Vec<usize> = paths[..300]
            .iter()
            .map(|p| target_block(p).unwrap())
            .collect();
        assert!(blocks.windows(2).all(|w| w[0] <= w[1]));
        let refiner: Vec<usize> = paths[300..]
            .iter()
            .map(|p| refiner_block(p).unwrap())
            .collect();
        assert!(refiner.windows(2).all(|w| w[0] <= w[1]));
        assert!(trains_refiner(&paths));
        // Nothing outside a block.
        assert!(paths
            .iter()
            .all(|p| target_block(p).is_some() || refiner_block(p).is_some()));
        assert!(!paths.iter().any(|p| p.contains("adaln_proj")));
    }

    #[test]
    fn an_explicit_suffix_matches_block_leaves_only_where_trainable() {
        let q = resolve_target_paths(&surface(), &cfg(&["to_q"])).unwrap();
        assert_eq!(q.len(), 50 + 2, "the DiT's and the refiner's to_q");
        assert!(trains_refiner(&q));
        // A suffix that lands on the refiner alone trains it alone.
        let r = resolve_target_paths(&surface(), &cfg(&["refiner_blocks.0.attn.to_q"])).unwrap();
        assert_eq!(
            r,
            vec!["token_refiner.refiner_blocks.0.attn.to_q".to_owned()]
        );
        // A suffix only the DiT blocks carry leaves the refiner constant.
        let d = resolve_target_paths(&surface(), &cfg(&["transformer_blocks.3.ff.net.2"])).unwrap();
        assert_eq!(d.len(), 1);
        assert!(!trains_refiner(&d));
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
            let step = distilled_step(i);
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

    /// Equal quantile slices of `t`, noisiest first: head `i` owns `((5 − i)/6, (6 − i)/6]`.
    #[test]
    fn a_drawn_level_trains_the_head_whose_slice_of_t_it_falls_in() {
        assert_eq!(training_step(1.0).index, 0);
        assert_eq!(training_step(0.999).index, 0);
        assert_eq!(training_step(0.0).index, 5);
        assert_eq!(training_step(0.001).index, 5);
        assert_eq!(training_step(0.5).index, 3);
        assert_eq!(training_step(f32::NAN).index, 0);
        assert_eq!(training_step(f32::INFINITY).index, 0);
        assert_eq!(training_step(-0.5).index, 5);
        assert_eq!(training_step(1.5).index, 0);
        for k in 1..FIXED_STEPS as usize {
            // Boundary `t = (6 − k)/6`: just above trains head k − 1, just below head k.
            let b = (FIXED_STEPS as usize - k) as f32 / FIXED_STEPS as f32;
            assert_eq!(training_step(b + 1e-4).index, k - 1, "above boundary {k}");
            assert_eq!(training_step(b - 1e-4).index, k, "below boundary {k}");
        }
        // The step carries the head's own published σ and timestep.
        let st = training_step(0.95);
        assert_eq!(st, distilled_step(0));
        assert_eq!(st.sigma, sampler_sigmas()[0]);
        assert_eq!(distilled_step(99), distilled_step(5));
    }

    /// A uniform draw trains every head exactly as often as the six-step render runs it.
    #[test]
    fn a_uniform_draw_trains_every_head_equally() {
        let n = 6000u32;
        let mut counts = [0u32; FIXED_STEPS as usize];
        for i in 0..n {
            // Cell midpoints of a uniform grid over (0, 1).
            counts[training_step((i as f32 + 0.5) / n as f32).index] += 1;
        }
        assert_eq!(counts, [n / 6; FIXED_STEPS as usize]);
    }

    #[test]
    fn the_training_canvas_keeps_the_aspect_at_the_edge_area_and_never_upscales() {
        // Larger than edge²: scaled down at its aspect, stride-floored, area within edge².
        let (w, h) = training_canvas((3000, 2000), 2048);
        assert_eq!((w % 32, h % 32), (0, 0));
        assert!((f64::from(w) / f64::from(h) - 1.5).abs() < 0.05);
        assert!(u64::from(w) * u64::from(h) <= 2048 * 2048);
        // A square source at edge²: the square canvas; a larger square scales down.
        assert_eq!(training_canvas((2048, 2048), 2048), (2048, 2048));
        assert_eq!(training_canvas((4096, 4096), 1000), (992, 992));
        // A 1365×2048 portrait at 2048: never upscaled, kept whole at its own area (width floored
        // to the stride) — not centre-cropped to a square and upscaled to 2048².
        assert_eq!(training_canvas((1365, 2048), 2048), (1344, 2048));
        // The same portrait at a smaller edge: scaled down, height stays the long side.
        let (w, h) = training_canvas((1365, 2048), 1024);
        assert!(h > w && u64::from(w) * u64::from(h) <= 1024 * 1024);
        assert!((f64::from(w) / f64::from(h) - 1365.0 / 2048.0).abs() < 0.03);
        // Landscape keeps width as the long side: the pair is (width, height).
        assert_eq!(training_canvas((2048, 1365), 2048), (2048, 1344));
        // A tiny source keeps one stride; a degenerate one falls back to the square edge.
        assert_eq!(training_canvas((10, 4000), 16), (32, 320));
        assert_eq!(training_canvas((10, 20), 2048), (32, 32));
        assert_eq!(training_canvas((0, 20), 1000), (992, 992));
        // Never larger than the source's area (once it holds a stride on each side).
        for (sw, sh) in [
            (1365, 2048),
            (640, 480),
            (5000, 300),
            (1024, 1024),
            (3000, 2000),
        ] {
            let (w, h) = training_canvas((sw, sh), 2048);
            assert!(u64::from(w) * u64::from(h) <= u64::from(sw) * u64::from(sh));
        }
    }

    #[test]
    fn the_descriptor_and_defaults_match_the_generator() {
        let d = trainer_descriptor("mlx");
        assert_eq!((d.id, d.family, d.backend), (TRAINER_ID, FAMILY, "mlx"));
        assert_eq!(d.max_reference_images, 16);
        assert!(d.techniques.resolution_buckets && d.techniques.weight_noise);
        assert!(!d.techniques.subject_mask_loss && !d.supports_control);
        assert!(d.techniques.training_adapter);
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
            resident_blocks: false,
            attention_heads_per_chunk: 56,
            refiner_layers: 0,
        }
    }

    /// A trained refiner adds its retained forward over the caption rows — each block's saved
    /// tensors at `T` rows plus its scores — to every residency, and nothing else.
    #[test]
    fn a_trained_refiner_retains_its_forward_over_the_caption_rows() {
        let g = DitGeometry::SHIPPED;
        let w = weights();
        let t = 256u64;
        let per_block = (14 * t * 7168 + 3 * t * 5376 + 4 * t * 14336) * 2 + 6 * 56 * t * t * 2;
        for checkpointed in [true, false] {
            let frozen = train_step_bytes(&g, &w, &shape(checkpointed));
            let trained = train_step_bytes(
                &g,
                &w,
                &TrainStepShape {
                    refiner_layers: 2,
                    ..shape(checkpointed)
                },
            );
            assert_eq!(trained.retained - frozen.retained, 2 * per_block);
            assert_eq!(trained.total() - frozen.total(), 2 * per_block);
            assert_eq!(trained.block_backward, frozen.block_backward);
            assert_eq!(trained.weights, frozen.weights);
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

    /// Resident + checkpointed: every block body on the device, but only the block inputs retained
    /// and one block's backward live — the streamed run's activations with the resident run's
    /// weights, and strictly between the two.
    #[test]
    fn resident_checkpointed_holds_every_body_and_only_the_block_inputs() {
        let g = DitGeometry::SHIPPED;
        let w = weights();
        let streamed = train_step_bytes(&g, &w, &shape(true));
        let resident_ckpt = train_step_bytes(
            &g,
            &w,
            &TrainStepShape {
                resident_blocks: true,
                ..shape(true)
            },
        );
        let resident = train_step_bytes(&g, &w, &shape(false));
        assert_eq!(
            resident_ckpt.weights,
            2_000_000_000 + 70_000_000 + 50 * 760_000_000
        );
        assert_eq!(resident_ckpt.retained, streamed.retained);
        assert_eq!(resident_ckpt.block_backward, streamed.block_backward);
        // The precompute still streams its AdaLN projections one at a time.
        assert_eq!(resident_ckpt.precompute, streamed.precompute);
        assert_eq!(resident_ckpt.total() - streamed.total(), 49 * 760_000_000);
        assert!(resident_ckpt.total() < resident.total());
        // Ignored when not checkpointed.
        let ignored = train_step_bytes(
            &g,
            &w,
            &TrainStepShape {
                resident_blocks: true,
                ..shape(false)
            },
        );
        assert_eq!(ignored, resident);
    }

    /// The training adapter's footprint splits the inference profile into the trunk share (always
    /// resident), the largest block's share (one streamed window) and the forward's per-row bytes.
    /// Mutations: sum the rows by max ⇒ the totals assert fails; count the refiner as a block ⇒ the
    /// trunk assert fails; take the smallest block ⇒ the `block_max` assert fails.
    #[test]
    fn the_training_adapter_footprint_splits_trunk_and_blocks() {
        use gen_core::h3_adapter_memory::{H3AdapterProjectionMemory, H3AdapterTarget};
        use gen_core::h3_memory::RowDomain;
        let projection = |path: &str, resident: u64, per_row: u64| H3AdapterProjectionMemory {
            target: H3AdapterTarget {
                path: path.to_owned(),
                input: 8,
                output: 8,
                packed: false,
                activation_width: 2,
                activation_dtype: gen_core::weightsmeta::Dtype::BF16,
                rows: RowDomain::Packed,
            },
            installed_bytes: resident,
            resident_bytes: resident,
            forward_fixed_bytes: 0,
            forward_bytes_per_row: per_row,
        };
        let profile = H3AdapterMemoryProfile {
            resident_bytes: 10 + 20 + 30 + 100,
            install_peak_bytes: 500,
            projections: vec![
                projection("transformer_blocks.0.attn.to_q", 10, 2),
                projection("transformer_blocks.0.ff.net.2", 20, 3),
                projection("transformer_blocks.1.attn.to_k", 30, 4),
                projection("token_refiner.refiner_blocks.0.attn.to_q", 100, 1),
            ],
        };
        let bytes = TrainingAdapterBytes::from_profile(&profile);
        assert_eq!(
            bytes,
            TrainingAdapterBytes {
                resident: 160,
                trunk: 100,
                block_max: 30,
                rows_block_max: 5,
                rows_total: 10,
            }
        );
        assert_eq!(bytes.device(true), 130);
        assert_eq!(bytes.device(false), 160);
        assert_eq!(bytes.host(true), 60);
        assert_eq!(bytes.host(false), 0);
        assert_eq!(bytes.rows(true), 5);
        assert_eq!(bytes.rows(false), 10);
        assert_eq!(
            TrainingAdapterBytes::from_profile(&H3AdapterMemoryProfile::default()),
            TrainingAdapterBytes::default()
        );
    }
}
