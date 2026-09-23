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
//!
//! # The decode transient is priced for MLX's *pipelined* evaluation (sc-24114)
//!
//! The first revision of this model counted the decoder's structurally live full-resolution maps
//! (three) and priced the untiled 2048² decode at 6.75 GiB. The memory campaign then hard-stopped
//! every 2048² cell above 100 GB, and one traced run showed why: MLX commits every full-res decoder
//! op as its own Metal command buffer and keeps ten in flight ahead of the GPU, each pinning its
//! input and output map, so the free-running untiled decode holds **25** maps — 56.6 GiB measured
//! against 56.25 derived ([`derived::VAE_PIPELINED_DECODE_MAPS`]) — and the allocator then pools
//! every freed map up to the device's working-set line (64.7 GiB cached, 109.9 GB footprint).
//! The DiT step carries the same in-flight term on top of its structural count
//! ([`derived::DIT_LIVE_HIDDEN_TENSORS`] + [`derived::MLX_MAX_ACTIVE_TASKS`]). Three things
//! changed, and every published number follows from them:
//!
//! * the **default decode is bounded** (512/64) wherever the untiled decode's pipelined transient
//!   exceeds the bounded transient that replaces it — every area above 512², so the fitted 1024²
//!   grid and every upstream preset — through [`default_decode_is_bounded`], which
//!   [`crate::pipeline::decode_tiling`] applies whenever nothing forces tiling (tiling is a
//!   preferred memory optimization; bitwise parity with candle is not a requirement). The bounded
//!   decode's own peak at a preset is the head's global attention
//!   ([`derived::VAE_HEAD_ATTENTION_MATRICES`]), which no tiling touches;
//! * every request installs [`AllocatorBounds`]: a **cache limit** of the request's transient
//!   budget, so freed buffers above it return to the OS and the process footprint tracks active
//!   memory, and an MLX **memory limit** of resident + budget as *uncredited* defence in depth —
//!   it makes the eval loop drain in-flight work when active memory passes the budget, but a
//!   retired command buffer's arrays are released one autorelease-pool drain later, so it only
//!   trims the pipelined set (17 maps measured at 1024² against 25 free-running), and the table
//!   does not count on it;
//! * the table's `activation_bytes` is the free-running untiled transient, `tiled_activation_bytes`
//!   the bounded one, and `default_activation_bytes` the one the default path runs — the number a
//!   consumer's floor must be built from.
//!
//! The 25-map and 6-matrix figures are the two derived numbers that have been held against an
//! observation; that makes them checked derivations, not measurements, and neither is filed as
//! evidence. The reference route adds a third term: every condition image is VAE-**encoded** at
//! the fitted 1024² grid inside the heavy phase, and the encoder's 96-channel full-resolution
//! stage under the same 25-map pipeline ([`derived::vae_encode_activation_bytes`]) is 9.38 GiB —
//! above the bounded decode's 6.06, so it is the request peak of a few-reference request at a
//! preset (the ten-reference denoise, 10.58 GiB, overtakes it). Not modelled: the text-conditioning phase itself (~1.7 GiB in the traced T2I runs,
//! whatever the target size), which is below the decode at every size from 1024² up.
//!
//! With every area above 512² decoding bounded, the default-path activation is monotonic in area,
//! and the largest-**area** preset (2400×1792, not the 2048² default) carries the largest
//! default-path peak: [`derived::default_path_peak_max_bytes`] is the number a consumer floor is
//! built from. The admissible domain reaches 2752² off every preset, where the head's attention
//! alone is 19.56 GiB — [`derived::admissible_square_peak_bytes`] states that number.

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

/// The decode tile edges this route publishes — **derived defaults, pending the epic-end
/// campaign**, not a measured ladder. 512 is the crate's shipped default ([`DECODE_TILE_EDGE`]) and
/// the neighbours bracket it on the same 64-px overlap; 768 and 640 are published on the strength
/// of the 2512 route's measured ladder (same 16x autoencoder family, same overlap), not on a run of
/// this VAE. The terminal story's measurement campaign is what turns these into a measured ladder;
/// until then the domain is deliberately narrow so a caller cannot select a geometry no sibling has
/// ever exercised, and nothing here claims a measurement.
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
    //
    // Verified against the frozen snapshot's headers (`UPSTREAM_HF_REVISION`
    // 790c92633540aa0cb11d9abf19eb46d861714758) on 2026-09-22 by
    // `tiers::derived_parameter_counts_match_the_frozen_snapshot` (header reads only): the
    // loaded `model.language_model.*` tower prices 15_136_811_008 B, `transformer/`
    // 14_230_249_472 B and `vae/` 1_350_961_616 B at bf16/bf16/f32 — exactly what
    // `resident_weights(Tier::Bf16)` derives from the four counts below. That test stays
    // `#[ignore]`d only because it needs the 31 GB snapshot on disk; re-run it on any pin bump.

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
    /// 8, so the FFN's 13 dominates.) **A structural count, not a measurement**, and only the
    /// synchronous floor: [`dit_activation_bytes`] adds one `[S, inner]` output per in-flight
    /// command buffer ([`MLX_MAX_ACTIVE_TASKS`] plus the one being scheduled) for MLX's pipelined
    /// evaluation — 24 tensors in all, 3.05 GiB at 2048² against the 2.77–3.02 GiB three traced
    /// runs measured for the denoise.
    pub const DIT_LIVE_HIDDEN_TENSORS: u64 = 13;

    /// Bytes per RoPE table entry pair (`cos` + `sin`, both f32, `head_dim / 2` wide):
    /// `2 · 64 · 4`.
    pub const DIT_ROPE_BYTES_PER_TOKEN: u64 = 512;

    /// **Structurally** live full-resolution feature maps at the VAE decoder's tail, each
    /// `full_res_channels · H · W` f32: the residual input, the residual branch's output, and the
    /// `norm_out`/`conv_out` input. The decoder is f32 throughout. This is what a *synchronous*
    /// executor holds; on MLX it is only the floor of [`VAE_PIPELINED_DECODE_MAPS`] — see there.
    pub const VAE_LIVE_FULL_RES_MAPS: u64 = 3;
    /// Channels the decoder's last stage runs at full output resolution — the same
    /// `full_res_channels` `gen_core::tiling::VaeTiling::QWEN_IMAGE_2_1` declares.
    pub const VAE_FULL_RES_CHANNELS: u64 = 144;
    /// Width of one decoder element (the released VAE ships and loads f32).
    pub const VAE_ELEMENT_WIDTH: u64 = 4;
    /// Channels the **encoder**'s first stage runs at full input resolution (`base_dim` = 96 in
    /// `vae/config.json`; the decoder's 144 is `decoder_base_dim`). The reference route encodes
    /// every condition image here, at the fitted 1024² grid.
    pub const VAE_ENCODE_FULL_RES_CHANNELS: u64 = 96;

    // ── MLX's pipelined evaluation (sc-24114) ───────────────────────────────────────────────────
    //
    // MLX does not execute a graph synchronously. `eval` walks the tape and commits one Metal
    // command buffer per op whose outputs exceed `max_mb_per_buffer` (40–50 MB) — every full-res
    // decoder op at any shipped preset — and keeps committing ahead of the GPU until
    // `MAX_ACTIVE_TASKS` (10, `mlx/transforms.cpp`) command buffers are in flight OR active memory
    // exceeds the allocator's *memory limit*. An in-flight op pins its input and its output until
    // its command buffer retires, so the decode's live set is NOT the structural 3 maps: it is the
    // structural set plus two maps for each of the ten in-flight ops and the one being scheduled.
    //
    // Measured on 2026-09-23 (2048², bf16, one untiled decode, `MLX_MAX_OPS_PER_BUFFER` unset):
    // active memory rose from 29.66 GiB to a 86.29 GiB peak — a 56.63 GiB transient against the
    // 56.25 GiB `VAE_PIPELINED_DECODE_MAPS` derives (0.7% apart) — while the "3 live maps" model
    // said 6.75 GiB. The allocator then kept 64.68 GiB of that in its cache (nothing bounded it
    // below the ~0.95 × recommendedMaxWorkingSetSize gc line), for a 109.85 GB process footprint.
    //
    // The memory limit is NOT credited against this. `mlx/transforms.cpp` waits only while command
    // buffers are in flight, and a retired buffer's arrays are released when its completion
    // closure is destroyed — one autorelease-pool drain later — so once the structural set plus
    // the lagging releases sit above the limit the loop proceeds one op at a time without ever
    // getting back under it. Measured the same day at 1024² untiled with the limit at resident
    // + 4.9 GiB: a 9.67 GiB transient (17 maps) against 14.06 GiB (25 maps) free-running.

    /// Command buffers MLX keeps in flight before its eval loop waits for the GPU
    /// (`MAX_ACTIVE_TASKS` in `mlx/transforms.cpp`; there is no runtime knob).
    pub const MLX_MAX_ACTIVE_TASKS: u64 = 10;
    /// Full-resolution maps one in-flight decoder op pins: its input and its output.
    pub const MLX_MAPS_PER_INFLIGHT_OP: u64 = 2;
    /// Full-res maps the untiled decode allocates under MLX's pipelined evaluation: the structural
    /// set plus an input/output pair for the ten in-flight ops and the one being scheduled,
    /// `3 + 2 · (10 + 1)` = 25. This is the number the table prices the untiled decode at and the
    /// one the campaign observed.
    pub const VAE_PIPELINED_DECODE_MAPS: u64 =
        VAE_LIVE_FULL_RES_MAPS + MLX_MAPS_PER_INFLIGHT_OP * (MLX_MAX_ACTIVE_TASKS + 1);
    /// Slack above the derived transient in the request-scoped allocator bounds: kernel workspace
    /// (Winograd transforms at the 288-channel half-resolution stage, SDPA scratch) that no map
    /// count carries. 1 GiB.
    pub const MLX_EVAL_SLACK_BYTES: u64 = 1 << 30;

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
    /// has consistently shown above 1024². The structural [`DIT_LIVE_HIDDEN_TENSORS`] plus one
    /// `[S, inner]` output per in-flight command buffer ([`MLX_MAX_ACTIVE_TASKS`]): every
    /// per-block op at a preset writes more than `max_mb_per_buffer`, so each is its own command
    /// buffer and the pipeline keeps ten outputs allocated ahead of the GPU, plus the one it is
    /// scheduling. The `true_cfg`
    /// negative branch runs sequentially over the same buffers, so it does not double this; it
    /// doubles only the resident conditioning, which [`activation_bytes`] accounts for separately.
    pub const fn dit_activation_bytes(tokens: u64) -> u64 {
        tokens * DIT_INNER * (DIT_LIVE_HIDDEN_TENSORS + MLX_MAX_ACTIVE_TASKS + 1) * BF16_WIDTH
            + tokens * DIT_ROPE_BYTES_PER_TOKEN
    }

    /// Bytes of one full-resolution decoder map at `width × height`.
    pub const fn full_res_map_bytes(width: u32, height: u32) -> u64 {
        VAE_FULL_RES_CHANNELS * width as u64 * height as u64 * VAE_ELEMENT_WIDTH
    }

    /// `[h·w, h·w]` f32 matrices live in the decode **head**'s global attention: the mid-block's
    /// single-head self-attention spans every latent position with `head_dim` = 1152, far past
    /// the fused SDPA kernel's head width, so MLX materialises the score matrix — `q·kᵀ`, its
    /// scaling and the softmax output — and, because every one of them is its own command buffer
    /// whose arrays are released only when its completion closure is destroyed at the next pool
    /// drain, the same three again. Six matrices of `image_tokens²` elements; at 2048² each is
    /// 1 GiB. The head runs once, untiled, **before**
    /// the tail, so it competes with the tail for the decode peak rather than adding to it.
    ///
    /// Measured 2026-09-23 on the bounded 2048² path: a 5.55 GiB decode transient against the
    /// 6.06 GiB this term derives (the tile maps are 0.15 GiB each and cannot reach it); an
    /// earlier run with a 1.4 GiB tighter memory limit showed 3.96 GiB, the limit engaged.
    pub const VAE_HEAD_ATTENTION_MATRICES: u64 = 6;

    /// The decode head's attention transient at `width × height`:
    /// [`VAE_HEAD_ATTENTION_MATRICES`] `[tokens, tokens]` f32 matrices.
    pub const fn vae_head_attention_bytes(width: u32, height: u32) -> u64 {
        let tokens = image_tokens(width, height);
        VAE_HEAD_ATTENTION_MATRICES * tokens * tokens * VAE_ELEMENT_WIDTH
    }

    const fn max(a: u64, b: u64) -> u64 {
        if a > b {
            a
        } else {
            b
        }
    }

    /// Derived VAE decode transient for one untiled `width × height` decode: the larger of the
    /// head's attention matrices and the tail's [`VAE_PIPELINED_DECODE_MAPS`] full-res maps (the
    /// tail, at every size from 512² up).
    pub const fn vae_decode_activation_bytes(width: u32, height: u32) -> u64 {
        max(
            vae_head_attention_bytes(width, height),
            VAE_PIPELINED_DECODE_MAPS * full_res_map_bytes(width, height),
        )
    }

    /// Derived VAE decode transient when the decode is bounded: the head runs once at full latent
    /// extent (its attention matrices are the peak at every preset), then the full-resolution tail
    /// runs one `tile_edge²` tile at a time behind a per-tile `eval` (so the pipelined set is a
    /// tile's, [`VAE_PIPELINED_DECODE_MAPS`] maps of `tile_edge²`); plus the blended RGBA f32
    /// output canvas the tiles are folded into, which is live through both. `overlap` shapes the
    /// tile grid, not the tile the tail decodes — `gen_core::tiling::split_spatial` cuts
    /// `tile_edge`-wide tiles at a `tile_edge − overlap` stride — so it does not enter the bytes.
    ///
    /// **Known err-low where the tile term binds.** Measured 2026-09-23 at 1024² through the
    /// bounded default (nine 512/64 tiles): a 4.98 GiB decode transient against the 3.53 GiB this
    /// derives — the up-block 3 → 4 transition inside a tile (a 576-channel nearest-upsampled map,
    /// the 288-channel conv and shortcut outputs) is wider than the 144-channel steady state the
    /// map count prices. It never reaches the head term at a preset (5.55 measured against 6.06
    /// derived at 2048², the tail hidden under it), so no floor moves; the request budget's 1 GiB
    /// slack and the memory limit absorbed it (37.2 GB footprint peak at 1024²).
    pub const fn tiled_vae_decode_activation_bytes(
        width: u32,
        height: u32,
        tile_edge: u32,
        _overlap: u32,
    ) -> u64 {
        max(
            vae_head_attention_bytes(width, height),
            VAE_PIPELINED_DECODE_MAPS * full_res_map_bytes(tile_edge, tile_edge),
        ) + 4 * width as u64 * height as u64 * VAE_ELEMENT_WIDTH
    }

    /// Derived VAE **encode** transient for one `width × height` condition image — the reference
    /// route's `vae.encode_mode` in the heavy phase: [`VAE_PIPELINED_DECODE_MAPS`] maps of the
    /// encoder's [`VAE_ENCODE_FULL_RES_CHANNELS`]-channel full-resolution stage, f32. The pipeline
    /// depth, not the reference count, bounds the live set: every reference's encode is lazy and
    /// materializes inside the first denoise step's eval, ten command buffers at a time whatever
    /// the count. The encoder's mid-block attention runs at the 64² latent (6 × 4096² × 4 B =
    /// 384 MiB) and never reaches this term.
    ///
    /// **Derived, err-high.** The one trace (2026-09-23, two 1024²-fitted references, 2048²
    /// target, two steps, `QWEN_IMAGE_2_1_EDIT_CASES=2:2048x2048:2`) put the whole heavy-phase
    /// transient — the lazy reference encodes materializing inside the denoise evals — at
    /// 5.64 GiB (peak_active 35.41 over 29.77 GiB resident), against the 9.38 GiB this term
    /// derives: the encoder's elementwise ops donate their input buffers where the decoder's
    /// residual structure cannot, so fewer than 25 maps are ever allocated. The 25-map figure is
    /// kept as the stated upper bound rather than fitted to one observation.
    pub const fn vae_encode_activation_bytes(width: u32, height: u32) -> u64 {
        VAE_PIPELINED_DECODE_MAPS
            * VAE_ENCODE_FULL_RES_CHANNELS
            * width as u64
            * height as u64
            * VAE_ELEMENT_WIDTH
    }

    /// The warm activation high-water mark of one request: the largest of the denoise, decode and
    /// — with references — the fitted-1024² reference encode peaks, plus the conditioning that
    /// stays resident across all of them.
    ///
    /// `use_negative` doubles the retained conditioning (the positive and negative embeddings are
    /// both alive for the whole denoise). `tile_edge` is the bounded-decode tile, `None` for the
    /// untiled decode; [`super::default_decode_is_bounded`] is the policy that picks between them
    /// when a request does not force tiling.
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
        let encode = if reference_count > 0 {
            vae_encode_activation_bytes(
                crate::config::OUTPUT_RESOLUTION,
                crate::config::OUTPUT_RESOLUTION,
            )
        } else {
            0
        };
        let branches = if use_negative { 2 } else { 1 };
        let retained = branches * conditioning_tokens * DIT_INNER * BF16_WIDTH;
        retained + max(max(denoise, decode), encode)
    }

    /// The transient one request is **budgeted** at — [`activation_bytes`] for the decode the
    /// request will actually run (forced tile, else the default policy) plus
    /// [`MLX_EVAL_SLACK_BYTES`]. This is the number [`super::AllocatorBounds`] hands MLX as the
    /// request's cache limit and, on top of the resident set, as its memory limit; the
    /// runtime is held to it rather than merely estimated by it.
    pub fn request_transient_budget_bytes(
        width: u32,
        height: u32,
        conditioning_tokens: u64,
        reference_count: u32,
        use_negative: bool,
        forced_tile_edge: Option<u32>,
    ) -> u64 {
        let tile_edge = forced_tile_edge.or_else(|| {
            super::default_decode_is_bounded(width, height).then_some(super::DECODE_TILE_EDGE)
        });
        activation_bytes(
            width,
            height,
            conditioning_tokens,
            reference_count,
            use_negative,
            tile_edge,
        ) + MLX_EVAL_SLACK_BYTES
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
        /// untiled decode ([`VAE_PIPELINED_DECODE_MAPS`] maps).
        pub activation_bytes: u64,
        /// The same with bounded decode engaged at the shipped 512/64 geometry.
        pub tiled_activation_bytes: u64,
        /// Whether the **default** decode at this preset is the bounded one
        /// ([`super::default_decode_is_bounded`]).
        pub default_decode_bounded: bool,
    }

    impl TableRow {
        /// The transient the default path runs at this preset: `tiled_activation_bytes` where the
        /// default decode is bounded, else `activation_bytes`.
        pub const fn default_activation_bytes(self) -> u64 {
            if self.default_decode_bounded {
                self.tiled_activation_bytes
            } else {
                self.activation_bytes
            }
        }

        /// Derived request peak under `Sequential` + bounded decode — the configuration a memory-
        /// constrained host runs.
        pub const fn bounded_peak_bytes(self) -> u64 {
            self.sequential_weight_floor_bytes + self.tiled_activation_bytes
        }

        /// Derived request peak under `Resident` + **untiled** decode — what a caller that forces
        /// the single pass above the threshold pays. Not the default path at any preset.
        pub const fn resident_peak_bytes(self) -> u64 {
            self.resident_weight_bytes + self.activation_bytes
        }

        /// Derived request peak under `Resident` on the **default** path — the number a
        /// consumer's floor should be built from.
        pub const fn default_peak_bytes(self) -> u64 {
            self.resident_weight_bytes + self.default_activation_bytes()
        }
    }

    /// The default-path activation at `width × height` under the table's assumptions
    /// ([`TABLE_CONDITIONING_TOKENS`], no references, no negative branch): the decode the policy
    /// picks ([`super::default_decode_is_bounded`]) against the denoise.
    pub fn default_activation_bytes_at(width: u32, height: u32) -> u64 {
        activation_bytes(
            width,
            height,
            TABLE_CONDITIONING_TOKENS,
            0,
            false,
            super::default_decode_is_bounded(width, height).then_some(super::DECODE_TILE_EDGE),
        )
    }

    /// The preset whose default-path peak is the largest: the largest-**area** preset, 2400×1792,
    /// because with every preset decoding bounded the peak is the decode head's attention
    /// (quadratic in the token count, 6.31 GiB there against 6.0 at the 2048² default) and the
    /// canvas; the denoise (linear in area, 3.16 GiB) is below it at every preset.
    pub fn default_path_peak_max_preset() -> SizePreset {
        crate::config::PRESETS
            .into_iter()
            .max_by_key(|preset| default_activation_bytes_at(preset.width, preset.height))
            .expect("the preset table is non-empty")
    }

    /// The largest default-path request peak of `tier` over the **preset table** — the number a
    /// consumer floor is built from: `resident + default_activation_bytes` at
    /// [`default_path_peak_max_preset`]. Since every area above 512² decodes bounded, the
    /// default-path activation is monotonic in area (head attention + canvas + denoise all grow
    /// with it), so the largest-area preset binds and no sub-preset geometry can exceed it; the
    /// single-pass region (≤ 512², 3.52 GiB) is far below.
    ///
    /// The *admissible* domain is wider than the presets: `Capabilities::max_size` admits any
    /// side up to 2752 on the 32-px grid, and at 2752² (7.57 Mpx, no preset) the head's
    /// attention alone is 19.56 GiB — see [`admissible_square_peak_bytes`]. The floor rule is
    /// stated on the presets, as the manifest's "at the 2048-square default preset" rule always
    /// was; a caller that admits off-preset geometry must price it from
    /// [`default_activation_bytes_at`].
    pub fn default_path_peak_max_bytes(tier: Tier) -> u64 {
        let preset = default_path_peak_max_preset();
        resident_weights(tier).resident_total()
            + default_activation_bytes_at(preset.width, preset.height)
    }

    /// The default-path peak of `tier` at the largest admissible square, `max_side²` — off every
    /// preset, published so the gap to the preset-based floor is a stated number, not a surprise.
    pub fn admissible_square_peak_bytes(tier: Tier) -> u64 {
        let side = super::admission_geometry().max_side;
        resident_weights(tier).resident_total() + default_activation_bytes_at(side, side)
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
                    default_decode_bounded: super::default_decode_is_bounded(
                        preset.width,
                        preset.height,
                    ),
                });
            }
        }
        rows
    }
}

// ================================================================================================
// Default decode policy and the request-scoped allocator bounds (sc-24114).
// ================================================================================================

/// The provider's automatic decode policy: whether a `width × height` request that does not force
/// tiling (`GenerationMemory::tile_vae_decode`) decodes bounded at the shipped 512/64 geometry.
///
/// **Tiling is a memory optimization and is preferred wherever it saves memory** (product decision
/// on sc-24114: bitwise parity with the candle lane is not required, the renders must look the
/// same, which the tiling parity tests establish at ≤ 2/255 mean). So: `true` wherever the
/// untiled decode's pipelined transient ([`derived::vae_decode_activation_bytes`], 25 full-res
/// maps) exceeds the bounded transient that replaces it
/// ([`derived::tiled_vae_decode_activation_bytes`] at [`DECODE_TILE_EDGE`]/[`DECODE_OVERLAP`]:
/// 25 maps of one 512² tile, or the head's attention matrices where those are larger, plus the
/// RGBA canvas). The two are equal at exactly 512² and the untiled cost grows 14.4 kB/px against
/// the bounded cost's 16 B/px, so every area above 512² — the fitted 1024² grid (14.06 GiB
/// untiled vs 3.52 bounded) and every upstream preset (56.25 vs 6.06 at 2048²) — decodes
/// bounded; at and below 512² the decode is the exact single pass, which is also where the tiler
/// itself would run one tile. Tier-independent by construction: the seam is a property of the
/// geometry, and a q4 render must not diverge from a bf16 one at the same size.
///
/// The shared `GenerationMemory` field only *forces* tiling; `false` (and an absent block) means
/// "the provider's own threshold applies", so there is no request-level way to ask for an untiled
/// decode above the threshold. That is deliberate: the untiled transient at a preset is a
/// multiple of every tier's resident set.
pub const fn default_decode_is_bounded(width: u32, height: u32) -> bool {
    derived::vae_decode_activation_bytes(width, height)
        > derived::tiled_vae_decode_activation_bytes(
            width,
            height,
            DECODE_TILE_EDGE,
            DECODE_OVERLAP,
        )
}

/// Request-scoped MLX allocator bounds (sc-24114). Two knobs, both restored on drop:
///
/// * **cache limit** (`mlx_rs::memory::set_cache_limit`): the allocator returns freed buffers
///   above it to the OS immediately instead of pooling them up to ~0.95 × the device's
///   recommended working set. Set to the request's transient budget, the process footprint the OS
///   (and the campaign's ceiling) sees tracks `active + budget` instead of growing to the pool
///   line — the untiled 2048² decode left 64.68 GiB pooled with the limit unset. This is the knob
///   the footprint numbers rest on.
/// * **memory limit** (`mlx_rs::memory::set_memory_limit`): MLX's eval loop drains in-flight
///   command buffers while active memory is above it. Set to the resident set plus the budget it
///   is **defence in depth that the derived table does not credit**: a retired command buffer's
///   arrays are released one autorelease-pool drain after it completes, so the loop cannot get
///   back under the limit once the lagging releases exceed the budget, and it then proceeds one op
///   at a time (17 maps measured at 1024² untiled against 25 free-running). It cannot fail a
///   request: with nothing in flight it never waits.
///
/// Neither changes a single computed value: the memory limit only reorders when work is committed
/// and the cache limit only decides where a dead buffer goes.
pub struct AllocatorBounds {
    previous_memory_limit: usize,
    previous_cache_limit: usize,
}

impl AllocatorBounds {
    /// Bound the allocator for one request: memory limit `resident + transient_budget`, cache limit
    /// `transient_budget`. `resident_bytes` is the larger of what is active now (a warm resident
    /// pair) and the tier's derived resident set (a staged request enters with nothing loaded).
    ///
    /// Both limits are process-global, and a harness may already have lowered one — the memory
    /// cap `mlx_gen::memory::apply_memory_cap_env` installs, a caller's own cache bound — so each
    /// is installed as `min(previous, request)`: a request can only ever tighten what it found.
    pub fn enter(resident_bytes: u64, transient_budget_bytes: u64) -> Self {
        let resident = resident_bytes.max(mlx_rs::memory::get_active_memory() as u64);
        let memory_limit =
            usize::try_from(resident.saturating_add(transient_budget_bytes)).unwrap_or(usize::MAX);
        let cache_limit = usize::try_from(transient_budget_bytes).unwrap_or(usize::MAX);
        let (previous_memory_limit, previous_cache_limit) = Self::current();
        mlx_rs::memory::set_memory_limit(memory_limit.min(previous_memory_limit));
        mlx_rs::memory::set_cache_limit(cache_limit.min(previous_cache_limit));
        Self {
            previous_memory_limit,
            previous_cache_limit,
        }
    }

    /// The limits currently installed, `(memory, cache)`, for tests and traces.
    pub fn current() -> (usize, usize) {
        let cache = mlx_rs::memory::set_cache_limit(0);
        mlx_rs::memory::set_cache_limit(cache);
        (mlx_rs::memory::get_memory_limit(), cache)
    }
}

impl Drop for AllocatorBounds {
    fn drop(&mut self) {
        mlx_rs::memory::set_memory_limit(self.previous_memory_limit);
        mlx_rs::memory::set_cache_limit(self.previous_cache_limit);
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
///   so it prices [`ResidentProjection::GroupQuantized`] at the tier's bit-width — one width for
///   every packable component, because a tier is a whole-pipeline contract (the earlier Q8 tower
///   floor on the q4 tier was withdrawn; `Tier::text_encoder_bits == Tier::transformer_bits`);
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
            // Unreachable in practice — `needs_load_time_quant` refused a non-affine request one
            // line up — but the refusal wording stays the shared one.
            CoreError::Unsupported(spec.quantize.map_or_else(
                || format!("{MODEL_ID}: a dense snapshot resolves to the bf16 tier"),
                crate::quant::non_affine_quant_refusal,
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
    // The load-time default is the provider's own decode threshold (`default_decode_is_bounded`:
    // bounded 512/64 wherever that saves memory, i.e. above 512²; the single pass below) plus the
    // request-scoped allocator bounds; a `Resident` selection keeps exactly that, so there is no
    // load-time staging or windowing for an explicit selection to override.
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

    /// The tier ladder is strictly monotone in every component that packs — the tower included,
    /// since a q4 tier runs a q4 tower — and the VAE never moves.
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

    /// The decode and denoise models against the on-device observations this module was corrected
    /// from (sc-24114, bf16, 2026-09-23): a free-running untiled 2048² decode took active memory
    /// from 29.66 GiB to 86.29 GiB — a 56.63 GiB transient — the bounded 2048² decode 5.55 GiB,
    /// and the two-step 2048² denoise 2.77–3.02 GiB. The earlier "3 live maps" model priced the
    /// untiled decode at 6.75 GiB; the pipelined model prices it within 1% of the observation,
    /// the bounded default at its head's 6 GiB of attention matrices, and the denoise at 3.05 GiB.
    ///
    /// *Mutation that reds this:* `VAE_PIPELINED_DECODE_MAPS = VAE_LIVE_FULL_RES_MAPS` (the old
    /// synchronous model), dropping `MLX_MAX_ACTIVE_TASKS` from `dit_activation_bytes`, or pricing
    /// the tile at `(edge + overlap)²`.
    #[test]
    fn the_decode_and_denoise_models_reproduce_the_measured_transients() {
        const MEASURED_UNTILED_2048_GIB: f64 = 56.63;
        const MEASURED_BOUNDED_2048_GIB: f64 = 5.55;
        const MEASURED_DENOISE_2048_GIB: [f64; 2] = [2.77, 3.02];
        let map = full_res_map_bytes(2048, 2048) as f64 / GIB;
        assert!(
            (map - 2.25).abs() < 0.001,
            "one 2048² map is 2.25 GiB, got {map}"
        );
        assert_eq!(VAE_PIPELINED_DECODE_MAPS, 25);

        let untiled = vae_decode_activation_bytes(2048, 2048) as f64 / GIB;
        assert!((untiled - 56.25).abs() < 0.01, "{untiled}");
        assert!(
            (untiled - MEASURED_UNTILED_2048_GIB).abs() / MEASURED_UNTILED_2048_GIB < 0.02,
            "free-running untiled decode: derived {untiled:.2} GiB vs measured \
             {MEASURED_UNTILED_2048_GIB} GiB"
        );
        let bounded =
            tiled_vae_decode_activation_bytes(2048, 2048, super::DECODE_TILE_EDGE, 64) as f64 / GIB;
        assert!((bounded - 6.06).abs() < 0.01, "{bounded}");
        let head = vae_head_attention_bytes(2048, 2048) as f64 / GIB;
        assert!((head - 6.0).abs() < 0.001, "{head}");
        assert!(
            bounded >= MEASURED_BOUNDED_2048_GIB
                && (bounded - MEASURED_BOUNDED_2048_GIB) / MEASURED_BOUNDED_2048_GIB < 0.10,
            "bounded decode: derived {bounded:.2} GiB vs measured {MEASURED_BOUNDED_2048_GIB} GiB"
        );

        let denoise = dit_activation_bytes(joint_tokens(2048, 2048, TABLE_CONDITIONING_TOKENS, 0))
            as f64
            / GIB;
        assert!((denoise - 3.05).abs() < 0.01, "{denoise}");
        assert!(
            denoise > MEASURED_DENOISE_2048_GIB[0] * 0.95
                && denoise < MEASURED_DENOISE_2048_GIB[1] * 1.05,
            "DiT step: derived {denoise:.2} GiB vs measured {MEASURED_DENOISE_2048_GIB:?}"
        );
        // At the presets the bounded decode (its head) is above the DiT step, so the default
        // transient IS the bounded decode plus the retained conditioning.
        let row = table()
            .into_iter()
            .find(|r| r.tier == Tier::Bf16 && r.preset.width == 2048 && r.preset.height == 2048)
            .unwrap();
        assert!(row.default_decode_bounded);
        assert!((row.tiled_activation_bytes as f64 / GIB - 6.06).abs() < 0.01);
        assert!((row.activation_bytes as f64 / GIB - 56.25).abs() < 0.01);
        assert!(
            (row.default_peak_bytes() as f64 / GIB - 34.67).abs() < 0.01,
            "bf16 default peak at 2048²: {:.3} GiB",
            row.default_peak_bytes() as f64 / GIB
        );
        assert!((row.resident_peak_bytes() as f64 / GIB - 84.86).abs() < 0.01);
        // The rule the consumer floors are built from: the default-path peak per tier at the
        // default preset, to 0.01 GiB.
        for (tier, peak) in [(Tier::Bf16, 34.67), (Tier::Q8, 22.40), (Tier::Q4, 15.85)] {
            let row = table()
                .into_iter()
                .find(|r| r.tier == tier && r.preset.width == 2048 && r.preset.height == 2048)
                .unwrap();
            assert!(
                (row.default_peak_bytes() as f64 / GIB - peak).abs() < 0.01,
                "{tier:?} default peak {:.3} GiB != {peak}",
                row.default_peak_bytes() as f64 / GIB
            );
        }
    }

    /// The default decode policy: bounded wherever the untiled transient exceeds the bounded one
    /// it replaces — every area above 512², so the fitted 1024² grid and every upstream preset —
    /// and the exact single pass at or below 512²; the budget the allocator bounds are set from
    /// follows the same choice.
    ///
    /// *Mutation that reds this:* `>=` in `default_decode_is_bounded` (512² would tile as one
    /// tile), a threshold on the resident set (which left 1024² single-pass), or a threshold at the
    /// default preset's own area.
    #[test]
    fn the_default_decode_is_bounded_wherever_it_saves_memory() {
        // At exactly 512² the tile IS the image: untiled 3.52 GiB vs bounded 3.52 + canvas.
        let untiled_512 = vae_decode_activation_bytes(512, 512);
        let bounded_512 = tiled_vae_decode_activation_bytes(512, 512, DECODE_TILE_EDGE, 64);
        assert_eq!(
            bounded_512 - untiled_512,
            4 * 512 * 512 * 4,
            "the canvas only"
        );
        assert!((untiled_512 as f64 / GIB - 3.52).abs() < 0.01);
        assert!(!default_decode_is_bounded(512, 512));
        assert!(!default_decode_is_bounded(256, 1024));
        // One grid step more and the single pass costs more than the tiles.
        assert!(default_decode_is_bounded(544, 512));
        assert!(default_decode_is_bounded(512, 544));
        // The fitted 1024² grid: 14.06 GiB untiled against 3.52 + 0.02 (canvas) bounded.
        assert!((vae_decode_activation_bytes(1024, 1024) as f64 / GIB - 14.06).abs() < 0.01);
        assert!(
            (tiled_vae_decode_activation_bytes(1024, 1024, DECODE_TILE_EDGE, 64) as f64 / GIB
                - 3.53)
                .abs()
                < 0.01
        );
        assert!(default_decode_is_bounded(1024, 1024));
        assert!(default_decode_is_bounded(1056, 1024));
        for preset in PRESETS {
            assert!(
                default_decode_is_bounded(preset.width, preset.height),
                "{} decodes bounded by default",
                preset.ratio
            );
        }
        assert!(table().iter().all(|row| row.default_decode_bounded));

        // The budget is the default path's transient plus the slack; a forced tile overrides it.
        let bounded = request_transient_budget_bytes(2048, 2048, 256, 0, false, None);
        assert_eq!(
            bounded,
            activation_bytes(2048, 2048, 256, 0, false, Some(super::DECODE_TILE_EDGE))
                + MLX_EVAL_SLACK_BYTES
        );
        let untiled = request_transient_budget_bytes(1024, 1024, 256, 0, false, None);
        assert_eq!(
            untiled,
            activation_bytes(1024, 1024, 256, 0, false, Some(super::DECODE_TILE_EDGE))
                + MLX_EVAL_SLACK_BYTES,
            "1024² decodes bounded by default"
        );
        assert_eq!(
            request_transient_budget_bytes(512, 512, 256, 0, false, None),
            activation_bytes(512, 512, 256, 0, false, None) + MLX_EVAL_SLACK_BYTES,
            "512² is the single pass"
        );
        let forced = request_transient_budget_bytes(1024, 1024, 256, 0, false, Some(256));
        assert_eq!(
            forced,
            activation_bytes(1024, 1024, 256, 0, false, Some(256)) + MLX_EVAL_SLACK_BYTES
        );
        assert!(forced < untiled);
        // Ten fitted references raise the denoise term, never the decode term.
        assert!(
            request_transient_budget_bytes(2048, 2048, 256, 10, false, None) > bounded,
            "references are joint tokens and price into the budget"
        );
    }

    /// A `Resident` selection through the registered scope leaves the request with no memory
    /// block, so the provider's own threshold decides: bounded at the default preset, untiled at
    /// a small size. A `BoundedDecode` selection forces the tile it admitted at any size.
    ///
    /// *Mutation that reds this:* `ResidentRequestMemory::ExplicitResident` together with a
    /// `decode_tiling` that treats `tile_vae_decode: false` as "untiled".
    #[test]
    fn a_resident_selection_defers_to_the_default_decode_policy() {
        use gen_core::{GenerationRequest, MemoryRunOutcome};

        let spec = LoadSpec::new(WeightsSource::Dir("/qwen-image-2-1".into()));
        let contract = weights_free_memory_strategy_contract(MODEL_ID, &spec).unwrap();
        let scoped = |strategy: MemoryStrategy, edge: u32, tile: Option<u32>| {
            let mut context = mlx_gen::gen_core::standard_memory_behavior_context(
                &contract,
                strategy,
                loaded_tier(&spec),
                MemoryBehaviorRoute {
                    mode: MemoryMode::TextToImage,
                    reference_count: 0,
                    use_pid: false,
                    has_phases: true,
                    overlay: None,
                },
            )
            .unwrap();
            context.geometry.width = edge;
            context.geometry.height = edge;
            context.geometry.batch = 1;
            context.selection.parameters.decode_tile_edge = tile;
            context.selection.parameters.decode_overlap = tile.map(|_| DECODE_OVERLAP);
            let mut request = GenerationRequest {
                prompt: "a red fox".into(),
                width: edge,
                height: edge,
                ..Default::default()
            };
            match registered_begin_request(&spec, &contract, &context).unwrap() {
                Some(mut scope) => {
                    scope.configure_request(&mut request).unwrap();
                    scope.finish(MemoryRunOutcome::Complete).unwrap();
                }
                None => assert_eq!(strategy, MemoryStrategy::Resident),
            }
            (request.memory, crate::pipeline::decode_tiling(&request))
        };

        let (memory, tiling) = scoped(MemoryStrategy::Resident, 2048, None);
        assert_eq!(
            memory, None,
            "a resident selection preserves the load defaults"
        );
        let spatial = tiling
            .expect("the default preset decodes bounded")
            .spatial
            .unwrap();
        assert_eq!((spatial.tile_px, spatial.overlap_px), (512, 64));

        let (memory, tiling) = scoped(MemoryStrategy::Resident, 512, None);
        assert_eq!(memory, None);
        assert!(
            tiling.is_none(),
            "below the threshold the decode is untiled"
        );

        let (memory, tiling) = scoped(MemoryStrategy::BoundedDecode, 512, Some(256));
        assert!(memory.unwrap().tile_vae_decode);
        let spatial = tiling
            .expect("a bounded selection forces tiling")
            .spatial
            .unwrap();
        assert_eq!((spatial.tile_px, spatial.overlap_px), (256, 64));
    }

    /// The allocator bounds install `resident + budget` / `budget`, never above a limit a harness
    /// already lowered, and restore the previous limits on drop, so a request never leaks its
    /// bounds into the next one.
    ///
    /// *Mutation that reds this:* installing the request's limits unconditionally (the first
    /// revision), which raised a harness's `MLX_GEN_MEMORY_CAP_GIB` cap for the request.
    #[test]
    fn allocator_bounds_are_installed_for_the_request_and_restored_after_it() {
        let (memory_before, cache_before) = AllocatorBounds::current();
        {
            let _bounds = AllocatorBounds::enter(28 << 30, 3 << 30);
            let (memory, cache) = AllocatorBounds::current();
            assert_eq!(cache, 3 << 30);
            // `resident` is the larger of the argument and what is active now.
            assert!(memory >= (28 << 30) + (3 << 30), "{memory}");
            assert!(memory <= (28 << 30) + (3 << 30) + mlx_rs::memory::get_active_memory());
        }
        assert_eq!(AllocatorBounds::current(), (memory_before, cache_before));

        // A harness that lowered either limit keeps it: the request only ever tightens.
        mlx_rs::memory::set_memory_limit(20 << 30);
        mlx_rs::memory::set_cache_limit(1 << 30);
        {
            let _bounds = AllocatorBounds::enter(28 << 30, 3 << 30);
            assert_eq!(AllocatorBounds::current(), (20 << 30, 1 << 30));
        }
        assert_eq!(AllocatorBounds::current(), (20 << 30, 1 << 30));
        mlx_rs::memory::set_memory_limit(memory_before);
        mlx_rs::memory::set_cache_limit(cache_before);
        assert_eq!(AllocatorBounds::current(), (memory_before, cache_before));
    }

    /// The consumer-floor number: with every area above 512² decoding bounded, the default-path
    /// activation is monotonic in area and the largest-**area** preset (2400×1792 — not the 2048²
    /// default, not the 2752-wide one) binds, through the decode head's attention (6.31 GiB) plus
    /// the canvas. The accessor equals that preset's default peak, is at least every preset's,
    /// and is above the whole single-pass region; the off-preset 2752² admissible square is
    /// stated separately.
    ///
    /// *Mutation that reds this:* building the accessor from the 2048² default preset, or from
    /// the untiled decode.
    #[test]
    fn the_default_path_peak_maximum_binds_at_the_largest_area_preset() {
        // 2400×1792 and its 3:4 twin tie by construction (same area, same token count).
        let binding = default_path_peak_max_preset();
        assert_eq!(binding.width * binding.height, 2400 * 1792);
        let activation = default_activation_bytes_at(2400, 1792) as f64 / GIB;
        assert!((activation - 6.37).abs() < 0.01, "{activation}");
        assert!((vae_head_attention_bytes(2400, 1792) as f64 / GIB - 6.31).abs() < 0.01);
        for (tier, expected) in [(Tier::Bf16, 34.99), (Tier::Q8, 22.71), (Tier::Q4, 16.15)] {
            let max = default_path_peak_max_bytes(tier);
            assert!(
                (max as f64 / GIB - expected).abs() < 0.01,
                "{tier:?}: {:.3} GiB != {expected}",
                max as f64 / GIB
            );
            for row in table().into_iter().filter(|r| r.tier == tier) {
                assert!(
                    max >= row.default_peak_bytes(),
                    "{tier:?} {}: floor source below a preset's default peak",
                    row.preset.ratio
                );
                assert!(
                    row.preset.width * row.preset.height < 2400 * 1792
                        || max == row.default_peak_bytes()
                );
            }
            // The single-pass region tops out at 512²: 3.52 GiB, below the presets' 6.37.
            assert!(
                max > resident_weights(tier).resident_total()
                    + default_activation_bytes_at(512, 512)
            );
        }
        // The floors a consumer derives with its `ceil(peak GiB x 1.25)` rule.
        let floors: Vec<u64> = [Tier::Bf16, Tier::Q8, Tier::Q4]
            .into_iter()
            .map(|tier| (default_path_peak_max_bytes(tier) as f64 / GIB * 1.25).ceil() as u64)
            .collect();
        assert_eq!(floors, [44, 29, 21]);
        // The admissible 2752² square is off every preset and 13 GiB above: a stated number.
        let square = admissible_square_peak_bytes(Tier::Bf16) as f64 / GIB;
        assert!((square - 48.28).abs() < 0.01, "{square}");
        assert!((vae_head_attention_bytes(2752, 2752) as f64 / GIB - 19.56).abs() < 0.01);
    }

    /// The reference route's encode term: 25 pipelined maps of the encoder's 96-channel
    /// full-resolution stage at the fitted 1024² grid, 9.38 GiB — above the presets' bounded
    /// decode, so it is the request peak of a few-reference request at a preset (the denoise
    /// overtakes it near ten), and below the 1024² untiled decode, so it never moves the floor.
    ///
    /// *Mutation that reds this:* dropping the encode term from `activation_bytes`, or pricing it
    /// at the decoder's 144 channels.
    #[test]
    fn a_reference_request_is_priced_at_the_pipelined_encode_of_the_fitted_grid() {
        let encode = vae_encode_activation_bytes(1024, 1024) as f64 / GIB;
        assert!((encode - 9.375).abs() < 0.001, "{encode}");
        let plain = activation_bytes(2048, 2048, 256, 0, false, Some(DECODE_TILE_EDGE));
        let one = activation_bytes(2048, 2048, 256, 1, false, Some(DECODE_TILE_EDGE));
        let ten = activation_bytes(2048, 2048, 256, 10, false, Some(DECODE_TILE_EDGE));
        assert!((plain as f64 / GIB - 6.06).abs() < 0.01);
        assert_eq!(
            one,
            256 * DIT_INNER * BF16_WIDTH + vae_encode_activation_bytes(1024, 1024),
            "one reference: the encode binds"
        );
        // The encode term itself does not grow with the count — the pipeline depth bounds it —
        // but ten fitted references make the joint sequence 57.6k tokens, where the denoise
        // (10.58 GiB) overtakes the encode.
        assert_eq!(
            ten,
            256 * DIT_INNER * BF16_WIDTH + dit_activation_bytes(joint_tokens(2048, 2048, 256, 10)),
            "ten references: the denoise binds"
        );
        assert!(ten > one);
        assert!(
            activation_bytes(1024, 1024, 256, 10, false, None)
                == 256 * DIT_INNER * BF16_WIDTH + vae_decode_activation_bytes(1024, 1024),
            "a forced single pass at 1024² still binds over ten references"
        );
        assert!(request_transient_budget_bytes(2048, 2048, 256, 1, false, None) > plain);
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

    /// A **transparent** reference must survive the admission -> execution round trip and be
    /// priced like an opaque one (sc-24111).
    ///
    /// Two halves, both load-bearing:
    ///
    /// * `geometry_from_request` (admission) and `memory_reference_count` (execution) must both
    ///   score the request at ONE reference, so the scope's geometry gate lets it through. When
    ///   `image_reference_count` scored `ReferenceRgba` at zero, admission at `reference_count: 1`
    ///   was followed by a hard refusal here — "references=0 does not fit admitted ...
    ///   references=1" — and an admission taken FROM the request priced the edit with zero
    ///   reference tokens.
    /// * the joint-token pricing must add exactly one `REFERENCE_FIT_TOKENS` block, the same block
    ///   an RGB reference adds: a transparent reference is fitted to `OUTPUT_RESOLUTION` like any
    ///   other.
    ///
    /// *Mutation that reds this:* drop `Conditioning::ReferenceRgba` from the `=> 1` arm of
    /// `gen_core::GenerationRequest::image_reference_count` (i.e. restore the `_ => 0` wildcard).
    #[test]
    fn a_transparent_reference_passes_the_geometry_gate_and_is_priced_like_an_opaque_one() {
        use gen_core::{Conditioning, GenerationRequest, Image, RgbaImage};

        let rgb_reference = || Conditioning::Reference {
            image: Image {
                width: 64,
                height: 64,
                pixels: vec![0; 64 * 64 * 3],
            },
            strength: None,
        };
        let rgba_reference = || Conditioning::ReferenceRgba {
            image: RgbaImage {
                width: 64,
                height: 64,
                pixels: vec![0; 64 * 64 * 4],
            },
            strength: None,
        };

        let request_with = |conditioning: Vec<Conditioning>| GenerationRequest {
            prompt: "a red fox".into(),
            width: 2048,
            height: 2048,
            conditioning,
            ..Default::default()
        };

        // 1. Admission and execution agree, and agree with the RGB carrier.
        let transparent = request_with(vec![rgba_reference()]);
        let opaque = request_with(vec![rgb_reference()]);
        assert_eq!(
            gen_core::wan_i2v_memory::geometry_from_request(&transparent).reference_count,
            1,
            "admission must see one reference"
        );
        assert_eq!(
            transparent.memory_reference_count(),
            1,
            "execution must see the same one"
        );
        assert_eq!(
            gen_core::wan_i2v_memory::geometry_from_request(&transparent),
            gen_core::wan_i2v_memory::geometry_from_request(&opaque),
            "a transparent reference admits identically to an opaque one"
        );

        // 2. The scope's geometry gate accepts it, admitted from the request itself.
        let spec = LoadSpec::new(WeightsSource::Dir("/qwen-image-2-1".into()));
        let contract = weights_free_memory_strategy_contract(MODEL_ID, &spec).unwrap();
        let fixture = registered_valid_fixture(&spec, &contract, MemoryStrategy::StagedResidency)
            .unwrap()
            .pop()
            .unwrap();
        let mut context = fixture.context.clone();
        let admitted = gen_core::wan_i2v_memory::geometry_from_request(&transparent);
        context.geometry.width = admitted.width;
        context.geometry.height = admitted.height;
        context.geometry.batch = admitted.batch;
        // `geometry_from_request` reports `frames: 0` for a still-image request while the scope
        // compares against its own `default_frames`; that axis is not what this test is about, so
        // the fixture's value stands and the reference axis is the only one under test.
        context.geometry.reference_count = admitted.reference_count;
        context.has_reference = true;
        assert_eq!(
            safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Accept,
            "a one-reference transparent edit must be admitted"
        );

        let mut scope = registered_begin_request(&spec, &contract, &context)
            .unwrap()
            .expect("this provider publishes a request scope");
        let mut executing = transparent.clone();
        scope.configure_request(&mut executing).expect(
            "the admitted transparent-reference request must pass the scope's geometry gate",
        );
        scope.finish(gen_core::MemoryRunOutcome::Complete).unwrap();

        // 3. ...and it is priced with exactly one fitted reference block.
        let bare = joint_tokens(2048, 2048, 0, 0);
        let one = joint_tokens(2048, 2048, 0, transparent.memory_reference_count());
        assert_eq!(
            one - bare,
            REFERENCE_FIT_TOKENS,
            "a transparent reference costs one fitted block, like every other reference"
        );
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
