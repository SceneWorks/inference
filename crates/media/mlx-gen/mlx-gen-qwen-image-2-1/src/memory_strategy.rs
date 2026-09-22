//! Qwen-Image 2.1 MLX request-memory contract and the **derived** memory model behind it
//! (sc-24112).
//!
//! Three rungs are implemented and declared; two are classified rather than implied by the shared
//! ladder order:
//!
//! | rung | strategy | this provider |
//! |---|---|---|
//! | 1 | [`MemoryStrategy::StagedResidency`] | **implemented** — `Residency::from_policy` loads the Qwen3 tower, encodes, drops it, then loads the DiT + VAE |
//! | 2 | [`MemoryStrategy::BoundedDecode`] | **implemented** — [`crate::pipeline::decode_tiling`] → `QwenImage21Vae::decode_rgba_tiled` |
//! | 3 | [`MemoryStrategy::BoundedAttention`] | not applicable — the DiT's block-causal joint SDPA is a per-segment kernel with no chunked variant on this route |
//! | 4 | [`MemoryStrategy::BoundedTransformerResidency`] | not applicable — this crate has no block-streaming loader |
//!
//! # Everything numeric here is DERIVED, never measured
//!
//! [`derived`] publishes a closed-form estimate of resident weights and of the warm activation
//! transient for every tier at every preset. It is built from **parameter counts × dtype width** and
//! from a **structural count of the live tensors** in this crate's own ported forward — nothing in
//! this file is an on-device observation, and this provider therefore registers **no**
//! [`mlx_gen::gen_core::ActivationMemoryRegistration`]: that carrier's contract is explicit that
//! "providers publish only real on-device measurements … no registration means unmeasured", and
//! filing a derivation there would launder an estimate into evidence. The epic's terminal
//! measurement story owns the real anchor; until then `activation_memory_bytes_1024` answers `None`
//! for `qwen_image_2_1` and callers fall back to the asset-facts + generic-headroom estimate path,
//! where the estimate safety margin is applied.
//!
//! None of these numbers are copied from the 2512 `qwen_image` route. That route's DiT is ~3× this
//! one and its text tower is a different architecture; its measured anchor is not evidence here.

use std::path::Path;

use mlx_gen::asset_facts::{projected_safetensors_bytes, ResidentProjection};
use mlx_gen::gen_core::{
    self, Error as CoreError, MemoryAssetFacts, MemoryBackendRealization, MemoryBehaviorFixture,
    MemoryBehaviorRoute, MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategySupport, ResidentRequestMemory, Result as CoreResult,
};
use mlx_gen::{LoadSpec, WeightsSource};

use crate::config::{MAX_REFERENCE_IMAGES, PRESETS};
use crate::model::MODEL_ID;
use crate::pipeline::{DECODE_OVERLAP, DECODE_TILE_EDGE};
use crate::quant::{Tier, GROUP_SIZE};

/// Content fingerprint of this contract. Load shape is a separate typed axis on
/// [`MemoryCalibrationIdentity`], so it is deliberately absent from the string.
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "qwen-image-2-1-mlx-derived-2026-09-22-v1";

/// The decode tile edges this route publishes. **Derived, not a measured ladder**: 512 is the
/// crate's shipped default ([`DECODE_TILE_EDGE`]) and the neighbours bracket it on the same 64-px
/// overlap. A measured ladder (the 2512 route has one) is the terminal story's to establish; until
/// then the published domain is deliberately narrow so a caller cannot select a geometry no one has
/// ever exercised.
pub const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512, 384, 256];

// ================================================================================================
// Derived model.
// ================================================================================================

/// The closed-form memory model for `qwen_image_2_1` — **derived estimates, never measurements**.
///
/// Every constant below is either a parameter count taken from the *tensor headers* of the frozen
/// snapshot ([`crate::UPSTREAM_HF_REVISION`] — counting parameters, not observing memory) or a
/// structural count of live tensors read off this crate's own forward. They are named individually
/// so the terminal measurement story can replace them one at a time instead of re-deriving the
/// whole table.
pub mod derived {
    use super::{Tier, GROUP_SIZE};
    use crate::config::SizePreset;

    /// Marker every published derived number carries into a report or a story description.
    pub const PROVENANCE: &str =
        "derived (parameter counts x dtype + structural activation model); \
                                  NOT measured on device";

    // ── Parameter counts, from the frozen snapshot's safetensors headers ────────────────────────

    /// Parameters in the DiT's 232 group-quantizable 2-D `Linear` weights.
    pub const DIT_LINEAR_PARAMS: u64 = 7_115_112_448;
    /// Bytes the DiT's 65 one-dimensional RMSNorm scales occupy at bf16 (never quantized).
    pub const DIT_DENSE_BYTES: u64 = 24_576;
    /// Parameters in the language tower's 252 group-quantizable decoder `Linear` weights.
    pub const LM_LINEAR_PARAMS: u64 = 6_945_767_424;
    /// Parameters in the language tower's token embedding — dense at every tier.
    pub const LM_EMBEDDING_PARAMS: u64 = 622_329_856;
    /// Bytes the language tower's 145 one-dimensional norm scales occupy at bf16.
    pub const LM_DENSE_NORM_BYTES: u64 = 616_448;
    /// Resident bytes of the RGBA autoencoder. The released file ships **f32** and MLX loads at the
    /// on-disk dtype, so this is the stored size and it does not vary with the tier.
    pub const VAE_BYTES: u64 = 1_350_961_616;

    /// Width of one bf16 element.
    pub const BF16_WIDTH: u64 = 2;

    /// Resident bytes of a group-quantized weight block: `params · bits / 8` codes plus one bf16
    /// scale and one bf16 bias per group of [`GROUP_SIZE`] (`params / group · 4`). This is exactly
    /// the arithmetic `mlx_gen::asset_facts::ResidentProjection::GroupQuantized` applies, so the
    /// derived table and the on-disk pricing cannot drift.
    pub const fn packed_bytes(params: u64, bits: u32) -> u64 {
        params * bits as u64 / 8 + (params / GROUP_SIZE as u64) * 4
    }

    const fn component_bytes(params: u64, bits: Option<i32>) -> u64 {
        match bits {
            None => params * BF16_WIDTH,
            Some(bits) => packed_bytes(params, bits as u32),
        }
    }

    /// Derived resident weight bytes of one tier, decomposed the way the memory contract's asset
    /// facts are.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ResidentWeights {
        /// The Qwen3 language tower actually loaded (`model.language_model.*`). The checkpoint's
        /// `lm_head` and `model.visual.*` ride along on disk but are never materialized here.
        pub conditioning: u64,
        /// The single-stream DiT.
        pub transformer: u64,
        /// The RGBA autoencoder.
        pub decoder: u64,
    }

    impl ResidentWeights {
        /// Everything resident at once — the `Resident` offload policy.
        pub const fn resident_total(self) -> u64 {
            self.conditioning + self.transformer + self.decoder
        }

        /// The `Sequential` floor: the text encoder is dropped before the DiT and VAE are loaded,
        /// so the peak is `max(tower, DiT + VAE)` — **never the sum**.
        pub const fn sequential_floor(self) -> u64 {
            let heavy = self.transformer + self.decoder;
            if self.conditioning > heavy {
                self.conditioning
            } else {
                heavy
            }
        }
    }

    /// Derived resident weights of `tier`.
    pub const fn resident_weights(tier: Tier) -> ResidentWeights {
        ResidentWeights {
            conditioning: component_bytes(LM_LINEAR_PARAMS, tier.text_encoder_bits())
                + LM_EMBEDDING_PARAMS * BF16_WIDTH
                + LM_DENSE_NORM_BYTES,
            transformer: component_bytes(DIT_LINEAR_PARAMS, tier.transformer_bits())
                + DIT_DENSE_BYTES,
            decoder: VAE_BYTES,
        }
    }

    // ── Activation transient ────────────────────────────────────────────────────────────────────

    /// Pixels one latent token covers on each axis: the VAE's 16× spatial scale (the DiT's latents
    /// are unpatched, `patch_size == 1`).
    pub const PIXELS_PER_TOKEN: u64 = 16;
    /// The DiT's joint hidden width (`num_attention_heads · attention_head_dim`).
    pub const DIT_INNER: u64 = 4096;
    /// `mlp_ratio`: the gated FFN's intermediate width is `DIT_INNER · DIT_MLP_RATIO`.
    pub const DIT_MLP_RATIO: u64 = 3;

    /// Live `[S, inner]`-shaped tensors at the DiT's per-block high-water mark, counted off
    /// [`crate::transformer`]'s forward. The FFN is the wider of the two peaks:
    ///
    /// * the residual `x`, the normalised+modulated `h` → 2
    /// * `gate_layer(h)` and `proj(h)`, each `inner · mlp_ratio` wide → `2 · DIT_MLP_RATIO` = 6
    /// * their product, reused in place, and `out(...)` → 1
    /// * the four per-token modulation rows (`scale1`, `gate1`, `scale2`, `gate2`) held across the
    ///   whole block → 4
    ///
    /// (The attention peak — `x`, `h`, q/k/v, the two RoPE half-planes and the SDPA output — counts
    /// 8, so the FFN's 13 dominates.) **A structural count, not a measurement**: it ignores
    /// allocator slack, the MLX graph's transient retention and any kernel workspace, all of which a
    /// real sweep would fold in.
    pub const DIT_LIVE_HIDDEN_TENSORS: u64 = 13;

    /// Bytes per RoPE table entry pair (`cos` + `sin`, both f32, `head_dim / 2` wide):
    /// `2 · 64 · 4`.
    pub const DIT_ROPE_BYTES_PER_TOKEN: u64 = 512;

    /// Live full-resolution feature maps at the VAE decoder's tail, each
    /// `full_res_channels · H · W` f32: the residual input, the residual branch's output, and the
    /// `norm_out`/`conv_out` input. The decoder is f32 throughout.
    pub const VAE_LIVE_FULL_RES_MAPS: u64 = 3;
    /// Channels the decoder's last stage runs at full output resolution — the same
    /// `full_res_channels` `gen_core::tiling::VaeTiling::QWEN_IMAGE_2_1` declares.
    pub const VAE_FULL_RES_CHANNELS: u64 = 144;
    /// Width of one decoder element (the released VAE ships and loads f32).
    pub const VAE_ELEMENT_WIDTH: u64 = 4;

    /// Conditioning tokens the published table assumes — a full 256-token prompt through the T2I
    /// template. The request-time estimate takes the real count.
    pub const TABLE_CONDITIONING_TOKENS: u64 = 256;

    /// Latent tokens **one reference image** contributes to the joint sequence.
    ///
    /// A reference is **not** admitted at the request's own geometry: `reference::prepare` fits
    /// every condition image to [`crate::config::OUTPUT_RESOLUTION`] (upstream's
    /// `output_resolution` default) before it reaches the vision tower and the VAE, so its latent
    /// grid is `(1024/16)² = 4096` tokens whatever the target size. Pricing references at the
    /// target's own token count — which an earlier revision of this module did — overstates the
    /// worst-case joint sequence by more than 3x at the largest preset, which is a consumer
    /// refusing requests that would in fact have fitted.
    pub const REFERENCE_FIT_TOKENS: u64 = (crate::config::OUTPUT_RESOLUTION as u64
        / PIXELS_PER_TOKEN)
        * (crate::config::OUTPUT_RESOLUTION as u64 / PIXELS_PER_TOKEN);

    /// Latent tokens one image of `width × height` contributes to the joint sequence.
    pub const fn image_tokens(width: u32, height: u32) -> u64 {
        (width as u64 / PIXELS_PER_TOKEN) * (height as u64 / PIXELS_PER_TOKEN)
    }

    /// The joint sequence length the DiT attends over: the conditioning tokens, the target image,
    /// and [`REFERENCE_FIT_TOKENS`] per reference image.
    ///
    /// **Reference images are extra image tokens, nothing else** — they enter as additional
    /// `Segment::Image` blocks of the joint layout. They are priced at the **fitted** 1024² grid,
    /// not at the target's geometry: see [`REFERENCE_FIT_TOKENS`]. This is the quantity a consumer
    /// must gate on, and it grows with the reference count independently of the target size.
    pub const fn joint_tokens(
        width: u32,
        height: u32,
        conditioning_tokens: u64,
        reference_count: u32,
    ) -> u64 {
        conditioning_tokens
            + image_tokens(width, height)
            + reference_count as u64 * REFERENCE_FIT_TOKENS
    }

    /// Derived DiT activation transient for a joint sequence of `tokens`, at bf16.
    ///
    /// Linear in `tokens`, hence linear in image **area** — which is the shape the MLX image lane
    /// has consistently shown above 1024². The `true_cfg` negative branch runs sequentially over
    /// the same buffers, so it does not double this; it doubles only the resident conditioning,
    /// which [`activation_bytes`] accounts for separately.
    pub const fn dit_activation_bytes(tokens: u64) -> u64 {
        tokens * DIT_INNER * DIT_LIVE_HIDDEN_TENSORS * BF16_WIDTH
            + tokens * DIT_ROPE_BYTES_PER_TOKEN
    }

    /// Derived VAE decode transient for one untiled `width × height` decode.
    pub const fn vae_decode_activation_bytes(width: u32, height: u32) -> u64 {
        VAE_LIVE_FULL_RES_MAPS
            * VAE_FULL_RES_CHANNELS
            * width as u64
            * height as u64
            * VAE_ELEMENT_WIDTH
    }

    /// Derived VAE decode transient when [`super::MemoryStrategy::BoundedDecode`] is engaged: the
    /// full-resolution tail runs one `tile_edge²` tile at a time, plus the blended output canvas
    /// (RGBA f32 at full resolution) the tiles are written into.
    pub const fn tiled_vae_decode_activation_bytes(
        width: u32,
        height: u32,
        tile_edge: u32,
        overlap: u32,
    ) -> u64 {
        let edge = tile_edge as u64 + overlap as u64;
        VAE_LIVE_FULL_RES_MAPS * VAE_FULL_RES_CHANNELS * edge * edge * VAE_ELEMENT_WIDTH
            + 4 * width as u64 * height as u64 * VAE_ELEMENT_WIDTH
    }

    /// The warm activation high-water mark of one request: the larger of the denoise and decode
    /// peaks, plus the conditioning that stays resident across both.
    ///
    /// `use_negative` doubles the retained conditioning (the positive and negative embeddings are
    /// both alive for the whole denoise).
    pub const fn activation_bytes(
        width: u32,
        height: u32,
        conditioning_tokens: u64,
        reference_count: u32,
        use_negative: bool,
        tile_edge: Option<u32>,
    ) -> u64 {
        let tokens = joint_tokens(width, height, conditioning_tokens, reference_count);
        let denoise = dit_activation_bytes(tokens);
        let decode = match tile_edge {
            Some(edge) => tiled_vae_decode_activation_bytes(width, height, edge, 64),
            None => vae_decode_activation_bytes(width, height),
        };
        let branches = if use_negative { 2 } else { 1 };
        let retained = branches * conditioning_tokens * DIT_INNER * BF16_WIDTH;
        retained + if denoise > decode { denoise } else { decode }
    }

    /// One published row of the derived table: a tier at a preset.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct TableRow {
        pub tier: Tier,
        pub preset: SizePreset,
        /// Weights held at once under the `Resident` policy.
        pub resident_weight_bytes: u64,
        /// `max(tower, DiT + VAE)` under the `Sequential` policy.
        pub sequential_weight_floor_bytes: u64,
        /// Warm transient at [`TABLE_CONDITIONING_TOKENS`], no references, no negative branch,
        /// untiled decode.
        pub activation_bytes: u64,
        /// The same with bounded decode engaged at the shipped 512/64 geometry.
        pub tiled_activation_bytes: u64,
    }

    impl TableRow {
        /// Derived request peak under `Sequential` + bounded decode — the configuration a memory-
        /// constrained host runs.
        pub const fn bounded_peak_bytes(self) -> u64 {
            self.sequential_weight_floor_bytes + self.tiled_activation_bytes
        }

        /// Derived request peak under `Resident` + untiled decode.
        pub const fn resident_peak_bytes(self) -> u64 {
            self.resident_weight_bytes + self.activation_bytes
        }
    }

    /// The full derived table: every tier at every upstream preset, densest tier first.
    pub fn table() -> Vec<TableRow> {
        let mut rows = Vec::with_capacity(Tier::ALL.len() * crate::config::PRESETS.len());
        for tier in Tier::ALL {
            let weights = resident_weights(tier);
            for preset in crate::config::PRESETS {
                rows.push(TableRow {
                    tier,
                    preset,
                    resident_weight_bytes: weights.resident_total(),
                    sequential_weight_floor_bytes: weights.sequential_floor(),
                    activation_bytes: activation_bytes(
                        preset.width,
                        preset.height,
                        TABLE_CONDITIONING_TOKENS,
                        0,
                        false,
                        None,
                    ),
                    tiled_activation_bytes: activation_bytes(
                        preset.width,
                        preset.height,
                        TABLE_CONDITIONING_TOKENS,
                        0,
                        false,
                        Some(super::DECODE_TILE_EDGE),
                    ),
                });
            }
        }
        rows
    }
}

// ================================================================================================
// Admission geometry — what a consumer must gate on.
// ================================================================================================

/// The geometry envelope `qwen_image_2_1` reports to its consumer contract so SceneWorks can admit
/// or refuse a request **without silently shrinking it** (sc-24112).
///
/// A caller that only knows `max_size` cannot bound this route: the joint sequence the DiT attends
/// over grows with the reference count as well as the target area, and the DiT transient is linear
/// in that sequence. These fields make both axes explicit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionGeometry {
    /// Longest side any preset uses (`Capabilities::max_size`).
    pub max_side: u32,
    /// Largest preset **area** in pixels — the real driver, and **not** `max_side²` nor the square
    /// preset: 2752×1536 is the *widest* (4.23 Mpx) and 2048×2048 the *default* (4.19 Mpx), but the
    /// 4:3 2400×1792 preset is the largest by area (4.30 Mpx). A consumer that budgeted from
    /// `max_size` or from the default preset would under-budget the real worst case.
    pub max_preset_area: u64,
    /// Latent tokens the largest-area preset contributes.
    pub max_target_image_tokens: u64,
    /// Most reference images the joint layout accepts ([`MAX_REFERENCE_IMAGES`]).
    pub max_reference_images: u32,
    /// Latent tokens **one** reference adds — [`derived::REFERENCE_FIT_TOKENS`], the fitted 1024²
    /// grid every condition image is resized to, **not** the target's own token count. References
    /// are extra image-token blocks; there is no separate reference budget.
    pub tokens_per_max_reference: u64,
    /// The worst-case joint sequence: the largest-area target plus
    /// [`Self::max_reference_images`] fitted references, plus the conditioning tokens the published
    /// table assumes.
    pub max_joint_tokens: u64,
    /// Pixels one latent token covers on each axis.
    pub pixels_per_token: u64,
    /// Images one request may ask for (`Capabilities::max_count`); they render sequentially, so
    /// this multiplies time rather than peak.
    pub max_batch: u32,
}

/// The route's admission envelope, derived from the frozen presets and the joint-layout limit.
pub fn admission_geometry() -> AdmissionGeometry {
    let max_side = PRESETS
        .iter()
        .map(|p| p.width.max(p.height))
        .max()
        .unwrap_or(0);
    let widest = PRESETS
        .iter()
        .copied()
        .max_by_key(|p| p.width as u64 * p.height as u64)
        .expect("the preset table is non-empty");
    let target_tokens = derived::image_tokens(widest.width, widest.height);
    AdmissionGeometry {
        max_side,
        max_preset_area: widest.width as u64 * widest.height as u64,
        max_target_image_tokens: target_tokens,
        max_reference_images: MAX_REFERENCE_IMAGES as u32,
        tokens_per_max_reference: derived::REFERENCE_FIT_TOKENS,
        max_joint_tokens: derived::joint_tokens(
            widest.width,
            widest.height,
            derived::TABLE_CONDITIONING_TOKENS,
            MAX_REFERENCE_IMAGES as u32,
        ),
        pixels_per_token: derived::PIXELS_PER_TOKEN,
        max_batch: crate::model::descriptor().capabilities.max_count,
    }
}

// ================================================================================================
// Contract.
// ================================================================================================

/// Architecture axes, read from this crate's frozen config types rather than re-parsed here.
fn architecture_facts() -> mlx_gen::gen_core::MemoryArchitectureFacts {
    use mlx_gen::architecture_facts as af;
    let dit = crate::config::TransformerConfig::production();
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: af::axis(dit.num_attention_heads),
        head_dim: af::axis(dit.attention_head_dim),
        transformer_blocks: af::axis(dit.num_layers),
        // `patch_size == 1`: the 2.1 latents are unpatched, so a latent token IS a latent cell.
        patch_size: af::axis(1),
        latent_channels: af::axis(dit.out_channels),
        vae_spatial_scale: af::axis(derived::PIXELS_PER_TOKEN as usize),
        // Structurally absent: a still-image autoencoder with no temporal axis in this route.
        vae_temporal_scale: None,
        activation_dtype_width: Some(mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH),
    }
}

/// Bytes the resolved tier's components occupy once loaded, read from the snapshot's own tensor
/// headers.
///
/// Public so the ignored real-weights test can hold [`derived::resident_weights`]'s hand-entered
/// parameter counts to what the frozen snapshot actually contains — header reads only, no tensor
/// data.
///
/// Each component is priced at what **its** loader materializes:
///
/// * a **packed** tier is already in its resident form on disk, so it prices
///   [`ResidentProjection::Stored`]; a **dense** snapshot with a Q4/Q8 request is quantized at load,
///   so it prices [`ResidentProjection::GroupQuantized`] at the tier's per-component bit-width
///   (which is *not* uniform — the language tower has a declared Q4 floor);
/// * the language tower prices the loaded `model.language_model.*` prefix only. The checkpoint's
///   untied `lm_head` and the whole `model.visual.*` tower are on disk but materialized by nothing
///   on this route, so they are [`ResidentProjection::Omit`] — charging them would bill ~2.4 GB of
///   weights no render touches;
/// * the VAE prices `Stored`: it ships f32 and MLX loads at the on-disk dtype.
pub fn asset_facts(spec: &LoadSpec, root: &Path) -> CoreResult<MemoryAssetFacts> {
    let tier = resolved_tier(spec, root)?;
    let load_time_quant = crate::quant::installed_tier(root)? == Tier::Bf16;
    let projection = |bits: Option<i32>| -> ResidentProjection {
        match (load_time_quant, bits) {
            (true, Some(bits)) => ResidentProjection::GroupQuantized {
                bits,
                group_size: GROUP_SIZE as usize,
            },
            _ => ResidentProjection::Stored,
        }
    };
    let transformer = projected_safetensors_bytes(root.join("transformer"), |_| {
        projection(tier.transformer_bits())
    })?;
    let te_projection = projection(tier.text_encoder_bits());
    let conditioning = projected_safetensors_bytes(root.join("text_encoder"), |header| {
        if header.name.starts_with(crate::loader::TEXT_ENCODER_PREFIX) {
            te_projection
        } else {
            ResidentProjection::Omit
        }
    })?;
    let decoder = projected_safetensors_bytes(root.join("vae"), |_| ResidentProjection::Stored)?;
    Ok(MemoryAssetFacts {
        base_bytes: conditioning
            .saturating_add(transformer)
            .saturating_add(decoder),
        conditioning_bytes: conditioning,
        transformer_bytes: transformer,
        decoder_bytes: decoder,
        overlay_bytes: 0,
    })
}

/// The tier this load resolves to: the installed tier of a packed snapshot, or the requested tier
/// of a dense one. Refuses a request the snapshot cannot serve, by the same rule
/// [`crate::quant::needs_load_time_quant`] applies at load.
fn resolved_tier(spec: &LoadSpec, root: &Path) -> CoreResult<Tier> {
    crate::quant::needs_load_time_quant(root, spec.quantize)?;
    match crate::quant::installed_tier(root)? {
        Tier::Bf16 => Tier::from_selected(spec.quantize).ok_or_else(|| {
            CoreError::Unsupported(format!(
                "{MODEL_ID}: {:?} is not an MLX affine tier (Q4/Q8)",
                spec.quantize
            ))
        }),
        installed => Ok(installed),
    }
}

/// The numeric tier this generator actually runs.
///
/// `component_precision_floors` is **empty and stays empty**: a tier is a whole-pipeline contract
/// here, so there is no component resident above the selected width for a receipt to record.
fn loaded_tier(spec: &LoadSpec) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    }
}

/// The executable contract for a real load — [`weights_free_memory_strategy_contract`] plus the
/// exact on-disk component inventory.
pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let mut contract = weights_free_memory_strategy_contract(provider_id, spec)?;
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(CoreError::Msg(format!(
            "{MODEL_ID}: memory facts require a snapshot directory"
        )));
    };
    contract.asset_facts = asset_facts(spec, root)?;
    Ok(contract)
}

/// Declaration-equivalent contract with no asset facts — the registry conformance surface. It
/// carries the same shape, calibration identity and strategy classification as the loaded contract
/// so nothing can mistake an unpriced surface for evidence.
pub fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    if provider_id != MODEL_ID {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: not a Qwen-Image 2.1 MLX memory provider"
        )));
    }
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::MlxMetal {
            // Unified memory: the staged phases release into the wired-residency budget, weights are
            // mmap-backed and lazy per tensor, and MLX's lazy graph needs an explicit `eval` before a
            // phase drop frees anything — which `QwenImage21::generate_impl` does before the
            // Sequential text-encoder drop.
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.architecture_facts = architecture_facts();
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        MEMORY_CALIBRATION_FINGERPRINT,
        spec.load_shape,
    ));
    contract.phase_facts = Some(mlx_gen::gen_core::MemoryPhaseFacts::staged(
        mlx_gen::gen_core::StagedWeightSchedule::TwoStage,
    ));
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        decode_tiling: true,
        attention_chunking: false,
        transformer_window_materialization: false,
    };
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::DecodeTileArea,
        ],
    };
    // The load-time default tiles nothing, so an explicit `Resident` selection has no default to
    // override.
    contract.resident_request_memory = ResidentRequestMemory::PreserveLoadDefaults;
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident
            | MemoryStrategy::StagedResidency
            | MemoryStrategy::BoundedDecode => MemoryStrategySupport::Implemented,
            MemoryStrategy::BoundedAttention => MemoryStrategySupport::StructurallyNotApplicable {
                reason: "qwen_image_2_1's joint attention is block-causal over per-segment spans; \
                         the chunked-score kernel has no block-causal variant on this route"
                    .to_owned(),
            },
            MemoryStrategy::BoundedTransformerResidency => {
                MemoryStrategySupport::StructurallyNotApplicable {
                    reason: "qwen_image_2_1 has no block-streaming loader; every DiT block is \
                             materialized by `load_transformer`"
                        .to_owned(),
                }
            }
        };
        capability.parameters = match capability.strategy {
            MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                decode_tile_edges: DECODE_TILE_EDGES.to_vec(),
                decode_overlaps: vec![DECODE_OVERLAP],
                ..Default::default()
            },
            _ => MemoryParameterRanges::default(),
        };
    }
    Ok(contract)
}

fn validate_decode(edge: Option<u32>, overlap: Option<u32>) -> CoreResult<()> {
    let edge = edge.ok_or_else(|| {
        CoreError::Unsupported(format!(
            "{MODEL_ID}: bounded decode requires an explicit tile edge"
        ))
    })?;
    let overlap = overlap.ok_or_else(|| {
        CoreError::Unsupported(format!(
            "{MODEL_ID}: bounded decode requires an explicit overlap"
        ))
    })?;
    if !DECODE_TILE_EDGES.contains(&edge) || overlap != DECODE_OVERLAP {
        return Err(CoreError::Unsupported(format!(
            "{MODEL_ID}: decode geometry {edge}/{overlap} is outside the published domain \
             {DECODE_TILE_EDGES:?} at overlap {DECODE_OVERLAP}"
        )));
    }
    Ok(())
}

/// The provider safety check: the shared handshake, then this route's own gate. It can reject,
/// never swap in a different strategy or tier.
pub fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if !matches!(context.mode, MemoryMode::TextToImage) {
            return Err(CoreError::Unsupported(format!(
                "{MODEL_ID}: this story's route is text-to-image; got {:?}",
                context.mode
            )));
        }
        // References are PRICED, not refused: sc-24110 advertises Reference/MultiReference on this
        // route, and each condition image is an extra joint-sequence block worth
        // `derived::REFERENCE_FIT_TOKENS`. The only reference gate here is the count the joint
        // layout can express; the per-image validation (geometry, grid agreement, the 1..=10 bound
        // as a request error) is `crate::reference`'s, and it must stay there so a bad reference is
        // a request refusal rather than a memory refusal.
        if context.geometry.reference_count as usize > MAX_REFERENCE_IMAGES {
            return Err(CoreError::Unsupported(format!(
                "{MODEL_ID}: {} reference images exceed the {MAX_REFERENCE_IMAGES} the joint layout \
                 can express",
                context.geometry.reference_count
            )));
        }
        if context.use_pid {
            return Err(CoreError::Unsupported(format!(
                "{MODEL_ID}: there is no PiD decoder on this route"
            )));
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            validate_decode(
                context.selection.parameters.decode_tile_edge,
                context.selection.parameters.decode_overlap,
            )?;
        }
        Ok(())
    };
    mlx_gen::gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(loaded_tier(spec)),
        Some(&route_gate),
    )
}

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || !matches!(
            contract.capability(strategy).map(|c| &c.support),
            Some(MemoryStrategySupport::Implemented)
        )
    {
        return Ok(Vec::new());
    }
    let context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        loaded_tier(spec),
        MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: true,
            overlay: None,
        },
    )?;
    Ok(vec![MemoryBehaviorFixture::new(context)])
}

pub(crate) fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(spec, contract, context) {
        return Err(CoreError::Unsupported(reason));
    }
    let config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        MODEL_ID,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        crate::config::TransformerConfig::production().num_layers,
        move |use_pid, edge, overlap| {
            if use_pid {
                return Err(CoreError::Unsupported(format!(
                    "{MODEL_ID}: there is no PiD decoder on this route"
                )));
            }
            validate_decode(Some(edge), Some(overlap))
        },
    )?;
    Ok(Some(Box::new(
        mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(
            config,
            mlx_gen::request_scope::MlxScopeCleanup::Device,
        ),
    )))
}

/// The shared-ladder registration pair this provider publishes.
pub const MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: MODEL_ID,
    contract: |spec| memory_strategy_contract(MODEL_ID, spec),
    safety_check,
};

/// Behaviour registration: the provider-owned valid fixtures and the executable admission seam.
pub const MEMORY_BEHAVIOR_REGISTRATION: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: registered_valid_fixture,
        begin_request: registered_begin_request,
    };

#[cfg(test)]
mod tests {
    use super::derived::*;
    use super::*;
    use mlx_gen::Quant;

    const GIB: f64 = (1_u64 << 30) as f64;

    #[test]
    fn derived_weight_totals_reconstruct_the_frozen_bf16_snapshot() {
        let bf16 = resident_weights(Tier::Bf16);
        // The pinned `*.safetensors.index.json` totals for the two sharded components, and the
        // stored size of the single-file VAE. If a future pin changes these, the derivation is the
        // thing that must be re-derived — not silently rescaled.
        assert_eq!(bf16.transformer, 14_230_249_472);
        assert_eq!(bf16.conditioning, 15_136_811_008);
        assert_eq!(bf16.decoder, 1_350_961_616);
        assert_eq!(bf16.resident_total(), 30_718_022_096);
    }

    #[test]
    fn packed_bytes_matches_the_shared_resident_projection_arithmetic() {
        // `params · bits / 8` codes plus one bf16 scale + one bf16 bias per group of 64.
        for params in [64_u64, 4096 * 4096, DIT_LINEAR_PARAMS] {
            for bits in [4_u32, 8] {
                assert_eq!(
                    packed_bytes(params, bits),
                    params * bits as u64 / 8 + params / 16,
                    "params {params} at Q{bits}"
                );
            }
        }
    }

    /// The tier ladder is strictly monotone in every component that packs, the VAE never moves, and
    /// the Q4 tier's text encoder equals the Q8 tier's — the declared floor, visible in the numbers.
    #[test]
    fn the_derived_tier_ladder_is_strictly_monotone_in_every_packable_component() {
        let bf16 = resident_weights(Tier::Bf16);
        let q8 = resident_weights(Tier::Q8);
        let q4 = resident_weights(Tier::Q4);

        assert!(q8.transformer < bf16.transformer);
        assert!(q4.transformer < q8.transformer);
        assert!(q8.conditioning < bf16.conditioning);
        assert!(
            q4.conditioning < q8.conditioning,
            "a q4 tier runs a q4 tower: {} must be below {}",
            q4.conditioning,
            q8.conditioning
        );
        assert_eq!(bf16.decoder, q4.decoder, "the all-conv VAE never packs");
        assert!(q4.resident_total() < q8.resident_total());
        assert!(q8.resident_total() < bf16.resident_total());

        // The numbers this PR publishes, to 0.01 GiB. They are DERIVED, and pinning them here is
        // what makes a silent change to the derivation visible in review.
        for (tier, resident, floor) in [
            (Tier::Bf16, 28.61, 14.51),
            (Tier::Q8, 16.33, 8.30),
            (Tier::Q4, 9.78, 4.99),
        ] {
            let w = resident_weights(tier);
            assert!(
                (w.resident_total() as f64 / GIB - resident).abs() < 0.01,
                "{tier:?} resident {:.3} GiB != {resident}",
                w.resident_total() as f64 / GIB
            );
            assert!(
                (w.sequential_floor() as f64 / GIB - floor).abs() < 0.01,
                "{tier:?} sequential floor {:.3} GiB != {floor}",
                w.sequential_floor() as f64 / GIB
            );
        }
    }

    /// Staging makes the floor `max`, never the sum — and at Q4 the *text encoder* is the binding
    /// term, which is the finding that justified packing the tower at all.
    #[test]
    fn the_sequential_floor_is_a_max_and_the_heavy_pair_binds_at_every_tier() {
        for tier in Tier::ALL {
            let w = resident_weights(tier);
            assert!(w.sequential_floor() < w.resident_total());
            assert_eq!(
                w.sequential_floor(),
                w.conditioning.max(w.transformer + w.decoder)
            );
        }
        // The DiT + VAE pair binds the staged floor at EVERY tier, because both sides of the max
        // shrink together — which is the point of a whole-pipeline tier. (Under the withdrawn Q8
        // text-encoder floor the tower stayed at 8.03 GiB and became the binding term at q4, so the
        // q4 floor barely moved off q8's: 8.03 instead of 4.99. That is what the floor cost.)
        for tier in Tier::ALL {
            let w = resident_weights(tier);
            assert!(
                w.transformer + w.decoder > w.conditioning,
                "{tier:?}: the heavy pair binds the staged floor ({} vs tower {})",
                w.transformer + w.decoder,
                w.conditioning
            );
            assert_eq!(w.sequential_floor(), w.transformer + w.decoder);
        }
    }

    #[test]
    fn activation_is_linear_in_area_and_covers_every_preset() {
        // The DiT term is exactly linear in token count, hence in area.
        let a = dit_activation_bytes(image_tokens(1024, 1024));
        let b = dit_activation_bytes(image_tokens(2048, 1024));
        let c = dit_activation_bytes(image_tokens(2048, 2048));
        assert_eq!(b, 2 * a);
        assert_eq!(c, 4 * a);

        let rows = table();
        assert_eq!(rows.len(), Tier::ALL.len() * PRESETS.len());
        for preset in PRESETS {
            let tiers: Vec<_> = rows.iter().filter(|r| r.preset == preset).collect();
            assert_eq!(tiers.len(), Tier::ALL.len(), "{} coverage", preset.ratio);
            for row in tiers {
                assert!(row.activation_bytes > 0);
                assert!(
                    row.tiled_activation_bytes < row.activation_bytes,
                    "{} {:?}: bounded decode must lower the transient",
                    preset.ratio,
                    row.tier
                );
                assert!(row.bounded_peak_bytes() < row.resident_peak_bytes());
            }
        }
        // Activation is tier-independent by construction (activations stay bf16 whatever the weight
        // tier), so only the weight terms separate the tiers.
        let square: Vec<_> = rows
            .iter()
            .filter(|r| r.preset.width == 2048 && r.preset.height == 2048)
            .collect();
        assert!(square
            .windows(2)
            .all(|w| w[0].activation_bytes == w[1].activation_bytes));
    }

    #[test]
    fn references_enter_as_extra_image_tokens_and_the_envelope_reports_them() {
        let g = admission_geometry();
        assert_eq!(g.max_side, 2752);
        // The 4:3 2400x1792 preset, NOT the 2752-wide one and NOT the 2048 square default.
        assert_eq!(g.max_preset_area, 2400 * 1792);
        assert!(g.max_preset_area > 2752 * 1536);
        assert!(g.max_preset_area > 2048 * 2048);
        assert_eq!(g.max_target_image_tokens, 150 * 112);
        assert_eq!(g.max_reference_images, 10);
        // `reference::prepare` FITS every condition image to `OUTPUT_RESOLUTION` (1024) before the
        // vision tower and the VAE see it, so one reference is 64x64 latent tokens whatever the
        // target size — NOT the target's 150x112.
        //
        // *Mutation that reds this:* `tokens_per_max_reference: target_tokens` (what this module
        // published before the fix pass), which overstates the envelope by 3.4x.
        assert_eq!(g.tokens_per_max_reference, REFERENCE_FIT_TOKENS);
        assert_eq!(g.tokens_per_max_reference, 64 * 64);
        assert_eq!(g.pixels_per_token, 16);
        assert_eq!(g.max_batch, 8);
        assert_eq!(
            g.max_joint_tokens,
            TABLE_CONDITIONING_TOKENS + 150 * 112 + 10 * 64 * 64,
            "the largest-area target plus ten FITTED references"
        );
        assert_eq!(g.max_joint_tokens, 58_016, "256 + 16_800 + 10 x 4_096");
        // A reference adds a fixed block, so the transient grows linearly in the count at a rate
        // the target size does not change — the quantity a consumer must gate on.
        let bare = dit_activation_bytes(joint_tokens(2048, 2048, 0, 0));
        let one = dit_activation_bytes(joint_tokens(2048, 2048, 0, 1));
        let ten = dit_activation_bytes(joint_tokens(2048, 2048, 0, 10));
        assert_eq!(one - bare, dit_activation_bytes(REFERENCE_FIT_TOKENS));
        assert_eq!(ten - bare, 10 * (one - bare));
        assert!(
            ten < 11 * bare,
            "fitted references are cheaper than target-sized ones"
        );
    }

    #[test]
    fn the_contract_declares_the_three_implemented_rungs_and_classifies_the_rest() {
        let spec = LoadSpec::new(WeightsSource::Dir("/qwen-image-2-1".into()));
        let contract = weights_free_memory_strategy_contract(MODEL_ID, &spec).unwrap();
        for strategy in [
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented,
                "{strategy:?}"
            );
        }
        for strategy in [
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert!(
                matches!(
                    contract.capability(strategy).unwrap().support,
                    MemoryStrategySupport::StructurallyNotApplicable { .. }
                ),
                "{strategy:?} must be classified, not implied by the ladder order"
            );
        }
        assert_eq!(contract.asset_facts, MemoryAssetFacts::default());
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            MEMORY_CALIBRATION_FINGERPRINT
        );
        gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        assert!(weights_free_memory_strategy_contract("qwen_image", &spec).is_err());
    }

    /// No tier promotes any component, so no tier's receipt carries a floor.
    ///
    /// *Mutation that reds this:* returning a non-empty `component_precision_floors` from
    /// `loaded_tier` for any selection.
    #[test]
    fn no_tier_records_a_component_precision_floor() {
        let dir = WeightsSource::Dir("/qwen-image-2-1".into());
        for quant in [None, Some(Quant::Q8), Some(Quant::Q4)] {
            let mut spec = LoadSpec::new(dir.clone());
            if let Some(quant) = quant {
                spec = spec.with_quant(quant);
            }
            assert!(
                loaded_tier(&spec).component_precision_floors.is_empty(),
                "{quant:?}: a whole-pipeline tier promotes nothing"
            );
        }
    }

    /// References are **priced**, not refused: sc-24110 advertises them on this route, so a memory
    /// refusal would make every reference render impossible. The count the joint layout cannot
    /// express is still refused, and PiD and a crossed tier still are.
    ///
    /// *Mutation that reds this:* restoring the `reference_count != 0` refusal in `safety_check`.
    #[test]
    fn references_are_priced_and_only_pid_tier_and_over_count_are_refused() {
        let spec = LoadSpec::new(WeightsSource::Dir("/qwen-image-2-1".into()));
        let contract = weights_free_memory_strategy_contract(MODEL_ID, &spec).unwrap();
        let fixture = registered_valid_fixture(&spec, &contract, MemoryStrategy::StagedResidency)
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(
            safety_check(&spec, &contract, &fixture.context),
            MemorySafetyDecision::Accept
        );

        for count in [1_u32, MAX_REFERENCE_IMAGES as u32] {
            let mut referenced = fixture.context.clone();
            referenced.geometry.reference_count = count;
            referenced.has_reference = true;
            assert_eq!(
                safety_check(&spec, &contract, &referenced),
                MemorySafetyDecision::Accept,
                "{count} references must be admitted at the derived peak, not refused"
            );
        }
        let mut too_many = fixture.context.clone();
        too_many.geometry.reference_count = MAX_REFERENCE_IMAGES as u32 + 1;
        too_many.has_reference = true;
        let over = safety_check(&spec, &contract, &too_many);
        assert!(
            matches!(over, MemorySafetyDecision::Reject { .. }),
            "{over:?}"
        );

        let mut pid = fixture.context.clone();
        pid.use_pid = true;
        assert!(matches!(
            safety_check(&spec, &contract, &pid),
            MemorySafetyDecision::Reject { .. }
        ));

        // A Q4 admission against a bf16-loaded generator must not pass.
        let mut crossed = fixture.context;
        crossed.selection.tier.quant = Some(Quant::Q4);
        assert!(matches!(
            safety_check(&spec, &contract, &crossed),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn bounded_decode_publishes_a_domain_and_refuses_everything_outside_it() {
        assert!(DECODE_TILE_EDGES.contains(&DECODE_TILE_EDGE));
        for edge in DECODE_TILE_EDGES {
            assert!(validate_decode(Some(*edge), Some(DECODE_OVERLAP)).is_ok());
        }
        assert!(validate_decode(Some(1024), Some(DECODE_OVERLAP)).is_err());
        assert!(validate_decode(Some(DECODE_TILE_EDGE), Some(96)).is_err());
        assert!(validate_decode(None, Some(DECODE_OVERLAP)).is_err());
        assert!(validate_decode(Some(DECODE_TILE_EDGE), None).is_err());
    }

    /// This route publishes no activation anchor. The registration carrier is measurement-only by
    /// contract, and everything in [`derived`] is an estimate.
    #[test]
    fn no_activation_anchor_is_registered_for_a_derived_model() {
        let registry = crate::provider_registry().unwrap();
        assert_eq!(
            registry.activation_memory_bytes_1024(MODEL_ID).unwrap(),
            None,
            "a derived estimate must not be filed as a measured anchor"
        );
        assert!(PROVENANCE.contains("NOT measured"));
    }
}
