//! Z-Image MLX adoption of the shared image-memory contract (SC-15449), including ladder rung 3
//! (bounded attention, SC-15615).
//!
//! All four registered Z-Image variants — `z_image_turbo`, `z_image`, `z_image_turbo_control`,
//! `z_image_control` — share one DiT, one staged [`Residency`](mlx_gen::Residency) seam, one tiled VAE
//! decode, and now one bounded-attention primitive, so they share one contract builder. Only the
//! provider id and the advertised request surface differ.
//!
//! ## Declared rungs
//!
//! | Rung | Support | Executable seam |
//! |---|---|---|
//! | 0 Resident | Implemented | `Residency::resident` — encoder + DiT + VAE held warm |
//! | 1 Staged residency | Implemented (load-time, see below) | `Residency::run_staged` (sc-10839 / sc-13571): encode → drop encoder → denoise → **drop DiT** → decode |
//! | 2 Bounded decode | Implemented | `Vae::decode_tiled` at 512 px / 64 px overlap ([`crate::pipeline::decode_tiling`]) |
//! | 3 Bounded attention | Implemented | [`mlx_gen::attention::sdpa_budgeted_bhsd`] threaded through every DiT attention (SC-15615) |
//! | 4 Bounded transformer residency | **Missing** | No per-block wired-residency window exists on the MLX Z-Image DiT |
//!
//! Rung 4 is declared `Missing`, not `StructurallyNotApplicable`: the architecture *does* have 30
//! independent trunk blocks a window could materialize one at a time, so the rung is applicable and
//! simply unimplemented. Declaring it `StructurallyNotApplicable` would be a false Full.
//!
//! **Rung 1 has no request-scoped lever.** Staging is gated on [`mlx_gen::Residency::is_sequential`],
//! which comes from the *load-time* [`OffloadPolicy`](mlx_gen::OffloadPolicy), so selecting
//! `StagedResidency` on a generator that was loaded `Resident` yields resident behaviour — the
//! selection is honoured only if the consumer also loaded the provider `Sequential`.
//! [`z_image_generation_memory`] maps rung 1 to an all-false [`GenerationMemory`] for exactly that
//! reason: there is nothing per-request to turn on. Krea's CUDA adoption has the same shape, so this
//! is a shared-contract gap (a load-time-vs-request-time seam) rather than a Z-Image one; it is
//! recorded here so no calibration reads a rung-1 cell as request-selectable.
//!
//! ## What rung 3 is worth here, and why it is not CUDA's rung 3 (SC-15615)
//!
//! The Candle/CUDA twin's rung 3 is what bought its 8 GB fit — staged Q4 denoise 8.394 → 5.709 GB
//! (−32%, SC-15256) — because candle's `attention_basic` **materializes** the `[B,H,Sq,Sk]` score
//! tensor and chunking bounds it. MLX's fused `scaled_dot_product_attention` never materializes that
//! tensor (pinned by `mlx_gen::attention`'s `fused_sdpa_does_not_materialize_the_scores`), so the
//! same knob buys something different and much smaller here.
//!
//! Measured on Apple M5 Max, real `z_image_turbo` **q4** weights, 1024², 4 steps, count 1
//! (`tests/bounded_attention_real_weights.rs`). **Two harnesses, deliberately:** the first four rows
//! drive `denoise_with_progress` on a directly-loaded DiT with synthetic caption conditioning — no
//! text encoder, no VAE, no `Residency::run_staged` — because that is the only way to swap the
//! attention plan under otherwise identical conditions and run the never-chunks control. The last two
//! rows are the real staged `generate`. The two baselines therefore differ (4.7746 vs 4.898 GiB — the
//! staged run additionally holds the ~157 MB VAE through denoise), and so do the savings (−0.080 vs
//! −0.245 GiB): freeing a transient earlier is worth more when more is co-resident. Read the first
//! group as *attribution* and the second as *the number that ships*.
//!
//! | Arm | Denoise peak | vs unbounded |
//! |---|---:|---:|
//! | DiT denoise loop, unbounded | 4.7746 GiB | — |
//! | DiT denoise loop, 64 Mi chunk, **lazy** | 4.7708 GiB | −0.08% (noise) |
//! | DiT denoise loop, 64 Mi chunk + per-chunk eval | 4.6944 GiB | **−1.7%** |
//! | DiT denoise loop, never-chunks budget + eval flag (control) | 4.7747 GiB | +0.00% |
//! | Full staged generate, no budget | 4.898 GiB | — |
//! | Full staged generate, 64 Mi budget | 4.653 GiB | **−5.0%** |
//!
//! So the rung is real, measured, and quality-preserving — the image is **bit-identical** (max channel
//! delta 0/255) and the raw velocity checksum matches to the last digit — but the mechanism is the
//! lazy-graph cut the per-chunk `eval` forces, not a bounded score matrix. The control arm proves it:
//! the eval flag on a budget that never chunks is inert.
//!
//! **Consequence for the 8 GB question (GitHub #1932).** Rung 3 does not close a gap here the way it
//! did on CUDA, because on this lane there was no 8 GB gap at the denoise phase to begin with: the
//! staged q4 1024² peak is 4.898 GiB without it. Rung 3 takes that to 4.653 GiB. What decides an 8 GB
//! *Mac* is therefore the reserve policy and the admission arithmetic (SC-15611 / SC-15614), not this
//! rung — and the measurement above was taken on a 128 GB machine, so it is a peak measurement, not a
//! demonstration that an 8 GB Mac completes the render.
//!
//! ## Route coverage
//!
//! The attention budget itself reaches **every** advertised denoise route — plain t2i, base CFG (both
//! the cond and the uncond forward), turbo control and base control (including the ControlNet
//! branch's own blocks), and the PiD route, whose denoise is the ordinary DiT.
//!
//! **But rungs 2 and above are refused on the PiD route** (see `safety_check`), and because the
//! ladder is cumulative that takes rung 3 with it: a `use_pid` request cannot select bounded
//! attention through the contract, only through the raw `GenerationMemory::chunk_attention` request
//! knob. The reason is rung 2, not rung 3 — PiD replaces the native VAE decode with a
//! super-resolving student that plans its own tile edge/overlap from its own budget
//! (`mlx_gen_pid::budget`) and never reads this contract's `decode_tile_edge`/`decode_overlap`, so
//! admitting a rung-2-or-deeper selection would execute a different strategy than the selector chose.
//! Refusing is the contract-correct outcome; making rung 3 independently selectable on PiD needs
//! PiD's planner reconciled with the shared parameters, which is SC-15510's PiD/alternate-decode
//! coverage.
//!
//! ## Ownership
//!
//! This file declares *structure and parameter domains only*. Measured coefficients, envelopes, and
//! per-tier peaks live in SceneWorks generated evidence keyed by
//! [`IMAGE_MEMORY_CALIBRATION_FINGERPRINT`]; the worker owns live-budget accounting and least-cost
//! selection. The scope below is defense in depth: it can reject a selection, never substitute one.

use mlx_gen::gen_core::{
    Error as CoreError, GenerationMemory, GenerationRequest, ImageMemoryAssetFacts,
    ImageMemoryBackendRealization, ImageMemoryCalibrationIdentity, ImageMemoryFormulaKind,
    ImageMemoryFormulaVariable, ImageMemoryGeometry, ImageMemoryLifecycleCapabilities,
    ImageMemoryParameterRanges, ImageMemoryPhase, ImageMemoryProviderContract,
    ImageMemoryRequestScope, ImageMemoryRunContext, ImageMemoryRunOutcome,
    ImageMemoryRuntimeSemantics, ImageMemorySafetyDecision, ImageMemorySelection,
    ImageMemoryStrategy, ImageMemoryStrategyCapability, ImageMemoryStrategyParameters,
    ImageMemoryStrategySupport, LoadSpec, PerComponentBytes, Result as CoreResult,
};

/// The decode tile edge / overlap the Z-Image MLX bounded decode is fixed at — the 512 px parity
/// sweet spot for this GroupNorm VAE (sc-13571, [`crate::pipeline::decode_tiling`]).
pub const DECODE_TILE_EDGE: u32 = 512;
/// The decode overlap paired with [`DECODE_TILE_EDGE`].
pub const DECODE_OVERLAP: u32 = 64;

/// The one bounded-attention parameter this provider accepts: the shared
/// [`mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET`] (64 Mi score elements per attention call),
/// the exact knob the Candle/CUDA Z-Image rung 3 measured in SC-15256 — reused verbatim so a
/// cross-backend comparison of the same rung is meaningful. It is the only candidate advertised in
/// `attention_chunk_sizes`, and the request scope re-validates it.
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;

/// Calibration content fingerprint. It must change whenever quantization floors, tensor layout, or
/// execution structure change in a way that invalidates measurements taken against this provider.
///
/// `-v1` is the first Z-Image MLX declaration. The name states the rungs it actually carries — staged
/// residency, tiled decode, bounded attention — so no evidence taken against it can be read as
/// covering the still-unimplemented bounded-transformer-residency rung.
pub const IMAGE_MEMORY_CALIBRATION_FINGERPRINT: &str =
    "z-image-mlx-staged-tiled-decode-bounded-attention-v1";

/// Build the Z-Image MLX provider contract for `provider_id`.
///
/// `spec` supplies the load-exact asset facts: the component `.safetensors` sums under the resolved
/// snapshot root, which is what the MLX loader actually materializes (the tier subdirectory is already
/// the spec root for a pre-quantized turnkey). A single-file (ComfyUI) source has no component tree,
/// so its asset facts stay zero rather than reporting a fabricated split.
pub fn image_memory_contract(provider_id: &str, spec: &LoadSpec) -> ImageMemoryProviderContract {
    ImageMemoryProviderContract {
        provider_id: provider_id.to_owned(),
        backend: ImageMemoryBackendRealization::MlxMetal {
            // Unified memory: the wired-residency budget is what the staged phases release, weights
            // are mmap-backed, and MLX's lazy graph needs explicit `eval` before a phase drop frees
            // anything (`Residency::run_staged` owns that discipline). No host↔device transfer.
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
        strategies: ImageMemoryStrategy::ALL
            .into_iter()
            .map(|strategy| ImageMemoryStrategyCapability {
                strategy,
                support: match strategy {
                    ImageMemoryStrategy::BoundedTransformerResidency => {
                        ImageMemoryStrategySupport::Missing
                    }
                    _ => ImageMemoryStrategySupport::Implemented,
                },
                parameters: match strategy {
                    ImageMemoryStrategy::BoundedDecode => ImageMemoryParameterRanges {
                        decode_tile_edges: vec![DECODE_TILE_EDGE],
                        decode_overlaps: vec![DECODE_OVERLAP],
                        ..Default::default()
                    },
                    ImageMemoryStrategy::BoundedAttention => ImageMemoryParameterRanges {
                        attention_chunk_sizes: vec![ATTENTION_CHUNK_SIZE],
                        ..Default::default()
                    },
                    _ => ImageMemoryParameterRanges::default(),
                },
            })
            .collect(),
        lifecycle: ImageMemoryLifecycleCapabilities {
            phases: vec![
                ImageMemoryPhase::Conditioning,
                ImageMemoryPhase::Denoise,
                ImageMemoryPhase::Decode,
            ],
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: false,
        },
        formula: ImageMemoryFormulaKind::PhaseEnvelope {
            phases: vec![
                ImageMemoryPhase::Conditioning,
                ImageMemoryPhase::Denoise,
                ImageMemoryPhase::Decode,
            ],
            variables: vec![
                ImageMemoryFormulaVariable::AssetBytes,
                ImageMemoryFormulaVariable::PixelCount,
                ImageMemoryFormulaVariable::BatchCount,
                ImageMemoryFormulaVariable::ConditioningTokenCount,
                ImageMemoryFormulaVariable::DecodeTileArea,
                ImageMemoryFormulaVariable::AttentionChunkSize,
            ],
        },
        calibration: Some(ImageMemoryCalibrationIdentity::new(
            IMAGE_MEMORY_CALIBRATION_FINGERPRINT,
        )),
        asset_facts: asset_facts(spec),
        runtime: ImageMemoryRuntimeSemantics::default(),
    }
}

/// Component `.safetensors` sums for the spec's snapshot root. A [`WeightsSource::File`] source has
/// no component tree, so every field stays `0` (the truthful "unknown", not a guess).
fn asset_facts(spec: &LoadSpec) -> ImageMemoryAssetFacts {
    let Ok(components) =
        PerComponentBytes::from_spec_subdirs(spec, &["text_encoder"], &["transformer"], &["vae"])
    else {
        return ImageMemoryAssetFacts::default();
    };
    ImageMemoryAssetFacts {
        base_bytes: components
            .text_encoder
            .saturating_add(components.dit)
            .saturating_add(components.vae),
        conditioning_bytes: components.text_encoder,
        transformer_bytes: components.dit,
        decoder_bytes: components.vae,
        // A PiD overlay is a separate student checkpoint outside the snapshot component tree, and it
        // is only resident when a request sets `use_pid`. Reporting the base snapshot's bytes here
        // would be wrong, and guessing the overlay's would be worse, so it stays 0 until per-model
        // calibration measures it.
        overlay_bytes: 0,
    }
}

/// The shared ladder → this provider's existing per-request execution controls.
///
/// The ladder is **cumulative**: every rung above `StagedResidency` also carries the levers below it,
/// so a rung-3 selection tiles the decode as well. `Resident` returns `None`, which is the historical
/// fast path (`GenerationRequest::memory` untouched).
pub(crate) fn z_image_generation_memory(
    selection: &ImageMemorySelection,
) -> Option<GenerationMemory> {
    match selection.strategy {
        ImageMemoryStrategy::Resident => None,
        ImageMemoryStrategy::StagedResidency => Some(GenerationMemory::default()),
        ImageMemoryStrategy::BoundedDecode => Some(GenerationMemory {
            tile_vae_decode: true,
            ..Default::default()
        }),
        ImageMemoryStrategy::BoundedAttention => Some(GenerationMemory {
            tile_vae_decode: true,
            chunk_attention: true,
            ..Default::default()
        }),
        // Rung 4 is Missing, so `validate_selection` rejects it before this maps anything; the arm
        // exists so implementing the rung later cannot silently fall through to a weaker lever set.
        ImageMemoryStrategy::BoundedTransformerResidency => Some(GenerationMemory {
            tile_vae_decode: true,
            chunk_attention: true,
            stream_transformer_blocks: true,
            ..Default::default()
        }),
    }
}

/// Request-scoped lifecycle state for one admitted Z-Image generation.
///
/// Holds no MLX arrays: its whole job is to translate the shared selection into
/// [`GenerationRequest::memory`], reject parameters this provider does not implement, and guarantee
/// the terminal synchronize-and-release on success, cancellation, **and** error.
pub(crate) struct ZImageImageMemoryScope {
    pub(crate) provider_id: &'static str,
    pub(crate) geometry: ImageMemoryGeometry,
    pub(crate) memory: Option<GenerationMemory>,
    pub(crate) finished: bool,
}

impl ZImageImageMemoryScope {
    fn ensure_active(&self) -> CoreResult<()> {
        if self.finished {
            Err(CoreError::Msg(format!(
                "{}: image-memory request scope is already finished",
                self.provider_id
            )))
        } else {
            Ok(())
        }
    }

    /// Terminal barrier + cache eviction, idempotent.
    ///
    /// MLX is lazy and its allocator retains freed buffers in a cache, so "the request is over" is
    /// only true after (a) a synchronous barrier on the default stream — which is what
    /// [`mlx_rs::Array::eval`] is — and (b) an explicit [`clear_cache`](mlx_rs::memory::clear_cache).
    /// Without both, a canceled or errored request can leave partially-resident buffers that poison
    /// the next request's budget. This runs on every exit path, including [`Drop`].
    fn synchronize_and_release(&mut self) -> CoreResult<()> {
        let barrier = mlx_rs::Array::from(0.0_f32);
        barrier.eval().map_err(mlx_gen::Error::from)?;
        drop(barrier);
        mlx_rs::memory::clear_cache();
        self.finished = true;
        Ok(())
    }
}

impl ImageMemoryRequestScope for ZImageImageMemoryScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> CoreResult<()> {
        self.ensure_active()?;
        if request.width != self.geometry.width
            || request.height != self.geometry.height
            || request.count == 0
            || request.count > self.geometry.batch
        {
            return Err(CoreError::Unsupported(format!(
                "{}: request geometry {}x{} count {} does not match admitted {}x{} count {}",
                self.provider_id,
                request.width,
                request.height,
                request.count,
                self.geometry.width,
                self.geometry.height,
                self.geometry.batch
            )));
        }
        // The shared selection is authoritative and request-scoped: overwrite (never merge) whatever
        // a reused warm request carried, so a deeper prior rung cannot leak into this run.
        request.memory = self.memory;
        Ok(())
    }

    fn enter_phase(&mut self, _phase: ImageMemoryPhase) -> CoreResult<()> {
        // The phase boundaries themselves are owned by `Residency::run_staged`, which already
        // evaluates and drops between phases; the scope only has to stay live across them.
        self.ensure_active()
    }

    fn leave_phase(&mut self, _phase: ImageMemoryPhase) -> CoreResult<()> {
        self.ensure_active()
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        _geometry: ImageMemoryGeometry,
    ) -> CoreResult<()> {
        self.ensure_active()?;
        if tile_edge == DECODE_TILE_EDGE && overlap == DECODE_OVERLAP {
            Ok(())
        } else {
            Err(CoreError::Unsupported(format!(
                "{}: decode tiling is fixed at {DECODE_TILE_EDGE}/{DECODE_OVERLAP}, got \
                 {tile_edge}/{overlap}",
                self.provider_id
            )))
        }
    }

    fn configure_attention(&mut self, chunk_size: u32) -> CoreResult<()> {
        self.ensure_active()?;
        if chunk_size == ATTENTION_CHUNK_SIZE {
            Ok(())
        } else {
            Err(CoreError::Unsupported(format!(
                "{}: attention chunk size is fixed at {ATTENTION_CHUNK_SIZE} score elements, got \
                 {chunk_size}",
                self.provider_id
            )))
        }
    }

    fn materialize_transformer_window(
        &mut self,
        _first_block: u32,
        _block_count: u32,
    ) -> CoreResult<()> {
        self.ensure_active()?;
        Err(CoreError::Unsupported(format!(
            "{}: bounded transformer residency is not implemented on the MLX Z-Image DiT",
            self.provider_id
        )))
    }

    fn finish(&mut self, _outcome: ImageMemoryRunOutcome) -> CoreResult<()> {
        // Deliberately outcome-independent: cancellation and error need the barrier + eviction at
        // least as much as success does.
        self.ensure_active()?;
        self.synchronize_and_release()
    }
}

impl Drop for ZImageImageMemoryScope {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.synchronize_and_release();
        }
    }
}

/// The provider safety check every Z-Image variant shares: the calibration handshake, then the shared
/// contract's own selection validation, then the budget. Defense in depth only — it can reject, it can
/// never swap in a different strategy or numeric tier.
pub(crate) fn safety_check(
    contract: &ImageMemoryProviderContract,
    context: &ImageMemoryRunContext,
) -> ImageMemorySafetyDecision {
    let Some(calibration) = contract.calibration.as_ref() else {
        return ImageMemorySafetyDecision::Reject {
            reason: format!("{}: no calibration identity declared", contract.provider_id),
        };
    };
    if context.calibration_abi != calibration.abi
        || context.calibration_fingerprint != calibration.fingerprint
    {
        return ImageMemorySafetyDecision::Reject {
            reason: format!(
                "{}: calibration handshake mismatch (admitted abi {} fingerprint {:?}, provider abi \
                 {} fingerprint {:?})",
                contract.provider_id,
                context.calibration_abi,
                context.calibration_fingerprint,
                calibration.abi,
                calibration.fingerprint
            ),
        };
    }
    if let Err(error) = contract.validate_selection(&context.selection) {
        return ImageMemorySafetyDecision::Reject {
            reason: error.to_string(),
        };
    }
    // Defense in depth for the PiD overlay route. A PiD request replaces the native VAE decode with
    // the super-resolving student, which plans its OWN tile edge/overlap from its own budget
    // (`mlx_gen_pid::budget`) and ignores this contract's decode parameters entirely
    // ([`crate::pipeline::decode_batch`] takes the PiD arm before the tiling arm). Admitting a
    // bounded-decode selection here would therefore execute a different strategy than the one the
    // shared selector chose — precisely what the contract forbids. Rungs 0-1 are unaffected: staged
    // residency behaves identically with the overlay, and rung 3 rides the DiT denoise, which PiD
    // does not touch. Reconciling PiD's planner with the shared parameters is SC-15510's
    // "PiD/alternate-decode" coverage.
    if context.use_pid && context.selection.strategy >= ImageMemoryStrategy::BoundedDecode {
        return ImageMemorySafetyDecision::Reject {
            reason: format!(
                "{}: {:?} is not admissible with the PiD decode overlay — the PiD decoder plans its \
                 own tiling and would not honour decode_tile_edge/decode_overlap (SC-15510 owns \
                 reconciling them); select StagedResidency or run without PiD",
                contract.provider_id, context.selection.strategy
            ),
        };
    }
    if !context.budget.fits(context.predicted_peak_bytes) {
        return ImageMemorySafetyDecision::Reject {
            reason: format!(
                "{}: predicted peak {} exceeds effective budget {}",
                contract.provider_id,
                context.predicted_peak_bytes,
                context.budget.effective_bytes()
            ),
        };
    }
    ImageMemorySafetyDecision::Accept
}

/// Open a request scope after `safety_check` accepted `context`.
pub(crate) fn begin_request(
    provider_id: &'static str,
    contract: &ImageMemoryProviderContract,
    context: &ImageMemoryRunContext,
) -> CoreResult<Option<Box<dyn ImageMemoryRequestScope + 'static>>> {
    if let ImageMemorySafetyDecision::Reject { reason } = safety_check(contract, context) {
        return Err(CoreError::Unsupported(reason));
    }
    Ok(Some(Box::new(ZImageImageMemoryScope {
        provider_id,
        geometry: context.geometry,
        memory: z_image_generation_memory(&context.selection),
        finished: false,
    })))
}

/// The strategy parameters this provider accepts, for a caller that wants the whole domain in one
/// value (the conformance tests and the SceneWorks evidence writer both key off this).
pub fn declared_parameters() -> ImageMemoryStrategyParameters {
    ImageMemoryStrategyParameters {
        decode_tile_edge: Some(DECODE_TILE_EDGE),
        decode_overlap: Some(DECODE_OVERLAP),
        attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
        transformer_window_size: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::attention::AttentionBudget;
    use mlx_gen::gen_core::WeightsSource;
    use mlx_gen::gen_core::{
        ImageMemoryBudget, ImageMemoryCacheState, ImageMemoryMode, ImageMemoryNumericTier,
        Precision, Quant, IMAGE_MEMORY_CALIBRATION_ABI,
    };

    fn spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent-z-image-snapshot".into()))
    }

    fn contract() -> ImageMemoryProviderContract {
        image_memory_contract(crate::model::MODEL_ID, &spec())
    }

    fn selection(strategy: ImageMemoryStrategy) -> ImageMemorySelection {
        ImageMemorySelection {
            strategy,
            // The shared contract requires the selection to carry exactly the parameters the rungs
            // up to and including the selected one own — no more, no less.
            parameters: match strategy {
                ImageMemoryStrategy::Resident | ImageMemoryStrategy::StagedResidency => {
                    ImageMemoryStrategyParameters::default()
                }
                ImageMemoryStrategy::BoundedDecode => ImageMemoryStrategyParameters {
                    decode_tile_edge: Some(DECODE_TILE_EDGE),
                    decode_overlap: Some(DECODE_OVERLAP),
                    ..Default::default()
                },
                ImageMemoryStrategy::BoundedAttention => ImageMemoryStrategyParameters {
                    decode_tile_edge: Some(DECODE_TILE_EDGE),
                    decode_overlap: Some(DECODE_OVERLAP),
                    attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                    transformer_window_size: None,
                },
                ImageMemoryStrategy::BoundedTransformerResidency => ImageMemoryStrategyParameters {
                    decode_tile_edge: Some(DECODE_TILE_EDGE),
                    decode_overlap: Some(DECODE_OVERLAP),
                    attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                    transformer_window_size: Some(1),
                },
            },
            tier: ImageMemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
            },
        }
    }

    fn context(strategy: ImageMemoryStrategy) -> ImageMemoryRunContext {
        ImageMemoryRunContext {
            selection: selection(strategy),
            calibration_abi: IMAGE_MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: IMAGE_MEMORY_CALIBRATION_FINGERPRINT.to_owned(),
            mode: ImageMemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: true,
            geometry: ImageMemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
            },
            overlay: None,
            budget: ImageMemoryBudget {
                total_bytes: 8 * 1000 * 1000 * 1000,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 2 * 1000 * 1000 * 1000,
            },
            predicted_peak_bytes: 4 * 1000 * 1000 * 1000,
            cache_state: ImageMemoryCacheState::Cold,
            evidence_revision: "test".to_owned(),
        }
    }

    #[test]
    fn contract_is_internally_conformant() {
        let contract = contract();
        assert_eq!(contract.conformance_errors(), Vec::<String>::new());
        gen_core_testkit::check_image_memory_contract(&contract).unwrap();
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            "z-image-mlx-staged-tiled-decode-bounded-attention-v1"
        );
    }

    #[test]
    fn rungs_zero_through_three_are_implemented_and_selectable() {
        let contract = contract();
        for strategy in [
            ImageMemoryStrategy::Resident,
            ImageMemoryStrategy::StagedResidency,
            ImageMemoryStrategy::BoundedDecode,
            ImageMemoryStrategy::BoundedAttention,
        ] {
            assert!(
                matches!(
                    contract.capability(strategy).map(|c| &c.support),
                    Some(ImageMemoryStrategySupport::Implemented)
                ),
                "{strategy:?} must be Implemented"
            );
            contract.validate_selection(&selection(strategy)).unwrap();
        }
    }

    /// SC-15615: rung 3 is selectable and its **exact chunk parameter is recorded** — the same 64 Mi
    /// score budget the Candle twin measured, so the two backends' rung 3 is the same knob. Nothing
    /// else is accepted, at either the contract or the scope layer.
    #[test]
    fn bounded_attention_records_exactly_one_chunk_parameter() {
        let contract = contract();
        let capability = contract
            .capability(ImageMemoryStrategy::BoundedAttention)
            .unwrap();
        assert!(matches!(
            capability.support,
            ImageMemoryStrategySupport::Implemented
        ));
        assert_eq!(
            capability.parameters.attention_chunk_sizes,
            vec![ATTENTION_CHUNK_SIZE]
        );
        assert_eq!(ATTENTION_CHUNK_SIZE, 64 * 1024 * 1024);
        assert!(contract.lifecycle.attention_chunking);

        // A foreign chunk size is rejected by the static validator...
        let mut sel = selection(ImageMemoryStrategy::BoundedAttention);
        sel.parameters.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE / 2);
        let err = contract.validate_selection(&sel).unwrap_err().to_string();
        assert!(err.contains("attention"), "{err}");

        // ...and a selection that omits the parameter entirely is rejected too, so the rung can
        // never run at an unrecorded chunk size.
        let mut sel = selection(ImageMemoryStrategy::BoundedAttention);
        sel.parameters.attention_chunk_size = None;
        assert!(contract.validate_selection(&sel).is_err());
    }

    #[test]
    fn bounded_transformer_residency_is_missing_not_structurally_na() {
        let contract = contract();
        assert!(matches!(
            contract
                .capability(ImageMemoryStrategy::BoundedTransformerResidency)
                .map(|c| &c.support),
            Some(ImageMemoryStrategySupport::Missing)
        ));
        assert!(!contract.lifecycle.transformer_window_materialization);
        let err = contract
            .validate_selection(&selection(ImageMemoryStrategy::BoundedTransformerResidency))
            .unwrap_err()
            .to_string();
        assert!(err.contains("Missing"), "{err}");
    }

    /// Rung 4 becomes reachable the moment it is implemented, because every rung below it is
    /// Implemented — i.e. this declaration leaves no cumulative prerequisite hole for SC-15510's
    /// follow-on to trip over.
    #[test]
    fn rung_four_becomes_selectable_once_implemented() {
        let mut contract = contract();
        for capability in &mut contract.strategies {
            if capability.strategy == ImageMemoryStrategy::BoundedTransformerResidency {
                capability.support = ImageMemoryStrategySupport::Implemented;
                capability.parameters.transformer_window_sizes = vec![1];
            }
        }
        contract.lifecycle.transformer_window_materialization = true;
        assert_eq!(contract.conformance_errors(), Vec::<String>::new());
        contract
            .validate_selection(&selection(ImageMemoryStrategy::BoundedTransformerResidency))
            .unwrap();
    }

    #[test]
    fn the_ladder_maps_to_cumulative_request_controls() {
        assert_eq!(
            z_image_generation_memory(&selection(ImageMemoryStrategy::Resident)),
            None
        );
        assert_eq!(
            z_image_generation_memory(&selection(ImageMemoryStrategy::StagedResidency)),
            Some(GenerationMemory::default())
        );
        assert_eq!(
            z_image_generation_memory(&selection(ImageMemoryStrategy::BoundedDecode)),
            Some(GenerationMemory {
                tile_vae_decode: true,
                ..Default::default()
            })
        );
        // Not selectable today, but the mapping is still the cumulative one.
        assert_eq!(
            z_image_generation_memory(&selection(ImageMemoryStrategy::BoundedAttention)),
            Some(GenerationMemory {
                tile_vae_decode: true,
                chunk_attention: true,
                ..Default::default()
            })
        );
    }

    /// The `chunk_attention` request knob is the executable half of rung 3: the shared selection maps
    /// onto it, and the calibration A/B drives it directly.
    #[test]
    fn the_request_level_chunk_attention_knob_is_still_honored() {
        let plain = GenerationRequest {
            prompt: "a fox".to_owned(),
            ..Default::default()
        };
        assert_eq!(
            crate::pipeline::attention_budget(&plain),
            AttentionBudget::UNBOUNDED
        );

        let bounded = GenerationRequest {
            prompt: "a fox".to_owned(),
            memory: Some(GenerationMemory {
                chunk_attention: true,
                ..Default::default()
            }),
            ..plain.clone()
        };
        assert_eq!(
            crate::pipeline::attention_budget(&bounded),
            AttentionBudget::CONSTRAINED
        );
        assert_eq!(
            AttentionBudget::CONSTRAINED.max_score_elements(),
            u64::from(ATTENTION_CHUNK_SIZE)
        );

        // Staged-only selection must NOT engage the budget (the rung boundary is real).
        let staged = GenerationRequest {
            memory: Some(GenerationMemory::default()),
            ..plain
        };
        assert_eq!(
            crate::pipeline::attention_budget(&staged),
            AttentionBudget::UNBOUNDED
        );
    }

    /// The PiD overlay replaces the decode with a student that plans its own tiling, so a
    /// bounded-decode (or deeper) selection must be refused rather than silently executing different
    /// parameters than the selector chose.
    #[test]
    fn the_pid_route_refuses_bounded_decode_but_keeps_the_cheaper_rungs() {
        let contract = contract();
        for strategy in [
            ImageMemoryStrategy::BoundedDecode,
            ImageMemoryStrategy::BoundedAttention,
        ] {
            let mut ctx = context(strategy);
            ctx.use_pid = true;
            match safety_check(&contract, &ctx) {
                ImageMemorySafetyDecision::Reject { reason } => {
                    assert!(reason.contains("PiD decode overlay"), "{reason}")
                }
                other => panic!("{strategy:?} with PiD must be rejected, got {other:?}"),
            }
            assert!(begin_request(crate::model::MODEL_ID, &contract, &ctx).is_err());
        }
        // Resident and staged residency stay available with the overlay — the DiT/encoder staging is
        // unaffected by which decoder runs.
        for strategy in [
            ImageMemoryStrategy::Resident,
            ImageMemoryStrategy::StagedResidency,
        ] {
            let mut ctx = context(strategy);
            ctx.use_pid = true;
            assert!(
                matches!(
                    safety_check(&contract, &ctx),
                    ImageMemorySafetyDecision::Accept
                ),
                "{strategy:?} must stay admissible with PiD"
            );
        }
        // And without PiD, bounded decode is admissible as usual — the guard is not a blanket refusal.
        assert!(matches!(
            safety_check(&contract, &context(ImageMemoryStrategy::BoundedDecode)),
            ImageMemorySafetyDecision::Accept
        ));
    }

    #[test]
    fn a_stale_calibration_fingerprint_never_admits_an_optimized_fit() {
        let contract = contract();
        let mut ctx = context(ImageMemoryStrategy::BoundedDecode);
        ctx.calibration_fingerprint = "z-image-mlx-something-older".to_owned();
        match safety_check(&contract, &ctx) {
            ImageMemorySafetyDecision::Reject { reason } => {
                assert!(
                    reason.contains("calibration handshake mismatch"),
                    "{reason}"
                )
            }
            other => panic!("stale fingerprint must be rejected, got {other:?}"),
        }
        assert!(begin_request(crate::model::MODEL_ID, &contract, &ctx).is_err());

        // A mismatched ABI is equally fatal.
        let mut ctx = context(ImageMemoryStrategy::BoundedDecode);
        ctx.calibration_abi = IMAGE_MEMORY_CALIBRATION_ABI + 1;
        assert!(matches!(
            safety_check(&contract, &ctx),
            ImageMemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn an_over_budget_prediction_is_rejected_before_any_work() {
        let contract = contract();
        let mut ctx = context(ImageMemoryStrategy::BoundedDecode);
        ctx.predicted_peak_bytes = ctx.budget.effective_bytes() + 1;
        match safety_check(&contract, &ctx) {
            ImageMemorySafetyDecision::Reject { reason } => {
                assert!(reason.contains("exceeds effective budget"), "{reason}")
            }
            other => panic!("over-budget must be rejected, got {other:?}"),
        }
        // Exact-boundary fits are accepted (the shared contract's documented rule).
        let mut ctx = context(ImageMemoryStrategy::BoundedDecode);
        ctx.predicted_peak_bytes = ctx.budget.effective_bytes();
        assert!(matches!(
            safety_check(&contract, &ctx),
            ImageMemorySafetyDecision::Accept
        ));
    }

    #[test]
    fn the_scope_overwrites_warm_request_state_and_finishes_once() {
        let contract = contract();
        let ctx = context(ImageMemoryStrategy::BoundedDecode);
        let mut scope = begin_request(crate::model::MODEL_ID, &contract, &ctx)
            .unwrap()
            .expect("an accepted context opens a scope");

        // A warm request carrying a DEEPER prior rung must be overwritten, not merged.
        let mut request = GenerationRequest {
            prompt: "a fox".to_owned(),
            width: 1024,
            height: 1024,
            count: 1,
            memory: Some(GenerationMemory {
                chunk_attention: true,
                stream_transformer_blocks: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        assert_eq!(
            request.memory,
            Some(GenerationMemory {
                tile_vae_decode: true,
                ..Default::default()
            })
        );
        assert_eq!(
            crate::pipeline::attention_budget(&request),
            AttentionBudget::UNBOUNDED,
            "a warm request's stale chunk_attention must not survive re-selection"
        );

        // Geometry drift from the admitted envelope is rejected.
        request.width = 1280;
        assert!(scope.configure_request(&mut request).is_err());
        request.width = 1024;
        // As is a count above the admitted batch, and a zero count.
        request.count = 2;
        assert!(scope.configure_request(&mut request).is_err());
        request.count = 0;
        assert!(scope.configure_request(&mut request).is_err());
        request.count = 1;

        scope.finish(ImageMemoryRunOutcome::Complete).unwrap();
        assert!(scope.finish(ImageMemoryRunOutcome::Complete).is_err());
        assert!(scope.configure_request(&mut request).is_err());
    }

    #[test]
    fn the_scope_accepts_only_its_declared_parameters() {
        let contract = contract();
        let ctx = context(ImageMemoryStrategy::BoundedDecode);
        let mut scope = begin_request(crate::model::MODEL_ID, &contract, &ctx)
            .unwrap()
            .unwrap();
        scope
            .configure_decode(DECODE_TILE_EDGE, DECODE_OVERLAP, ctx.geometry)
            .unwrap();
        assert!(scope
            .configure_decode(256, DECODE_OVERLAP, ctx.geometry)
            .is_err());
        assert!(scope
            .configure_decode(DECODE_TILE_EDGE, 128, ctx.geometry)
            .is_err());
        // The attention hook exists (the A/B drives it) and pins the one recorded parameter.
        scope.configure_attention(ATTENTION_CHUNK_SIZE).unwrap();
        assert!(scope.configure_attention(ATTENTION_CHUNK_SIZE / 2).is_err());
        // The transformer window is not implemented at any size.
        assert!(scope.materialize_transformer_window(0, 1).is_err());
        scope.finish(ImageMemoryRunOutcome::Complete).unwrap();
    }

    #[test]
    fn a_canceled_or_errored_run_still_releases() {
        let contract = contract();
        for outcome in [
            ImageMemoryRunOutcome::Canceled,
            ImageMemoryRunOutcome::Error {
                message: "boom".to_owned(),
            },
        ] {
            let ctx = context(ImageMemoryStrategy::BoundedDecode);
            let mut scope = begin_request(crate::model::MODEL_ID, &contract, &ctx)
                .unwrap()
                .unwrap();
            scope.finish(outcome).unwrap();
        }
        // A scope dropped without `finish` (a panic / early-return path) must still release without
        // panicking or double-releasing.
        let ctx = context(ImageMemoryStrategy::BoundedDecode);
        drop(begin_request(crate::model::MODEL_ID, &contract, &ctx).unwrap());
    }

    #[test]
    fn every_registered_variant_declares_the_same_conformant_contract() {
        for id in [
            crate::model::MODEL_ID,
            crate::model_base::MODEL_ID,
            crate::model_control::MODEL_ID,
            crate::model_base_control::MODEL_ID,
        ] {
            let contract = image_memory_contract(id, &spec());
            assert_eq!(contract.provider_id, id);
            assert_eq!(contract.conformance_errors(), Vec::<String>::new(), "{id}");
            gen_core_testkit::check_image_memory_contract(&contract).unwrap();
        }
    }

    #[test]
    fn declared_parameters_match_the_contract_ranges() {
        let contract = contract();
        let params = declared_parameters();
        assert_eq!(
            contract
                .capability(ImageMemoryStrategy::BoundedDecode)
                .unwrap()
                .parameters
                .decode_tile_edges,
            vec![params.decode_tile_edge.unwrap()]
        );
        assert_eq!(params.decode_overlap, Some(DECODE_OVERLAP));
        assert_eq!(
            contract
                .capability(ImageMemoryStrategy::BoundedAttention)
                .unwrap()
                .parameters
                .attention_chunk_sizes,
            vec![params.attention_chunk_size.unwrap()]
        );
        assert_eq!(params.transformer_window_size, None);
    }

    /// A single-file (ComfyUI) source has no component tree, so the asset facts must be the truthful
    /// zero rather than a fabricated split.
    #[test]
    fn a_single_file_source_reports_no_asset_facts() {
        let spec = LoadSpec::new(WeightsSource::File(
            "/nonexistent/z-image.safetensors".into(),
        ));
        let contract = image_memory_contract(crate::model::MODEL_ID, &spec);
        assert_eq!(contract.asset_facts, ImageMemoryAssetFacts::default());
        assert_eq!(contract.conformance_errors(), Vec::<String>::new());
    }
}
