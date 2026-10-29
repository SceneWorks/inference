//! Per-component byte accounting from the selected artifacts: DiT allocation, source-pruned
//! AdaLN sizing and the materialized precision widths.

use super::*;

/// The stage-ordered lifecycle every MiniMax-H3 render runs.
pub(super) fn phases() -> Vec<MemoryPhase> {
    vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ]
}

/// On-disk bytes of the four components one render can touch.
///
/// Absent components contribute zero ([`safetensors_path_bytes`]), so a weights-free spec yields a
/// zero footprint rather than an error — the same behavior `PerComponentBytes` has. The DiT is
/// resolved the way `crate::model` resolves it, so a **tiered** install (a staged `transformer`
/// component) is charged at the staged tier's bytes rather than the flat snapshot's.
pub(super) struct ComponentBytes {
    pub(super) text_encoder: u64,
    pub(super) dit: u64,
    /// The AdaLN sub-stack's residency **on the tier that was actually resolved** — see
    /// [`resolved_adaln_bytes`]. Carried alongside `dit` rather than recomputed in
    /// [`adaln_component`] so that the weights-free contract, which resolves nothing, states its
    /// architecture fact in one place instead of re-deriving it from a zero footprint.
    pub(super) adaln: u64,
    /// Source-pruned operands/cache remain live; a mixed task route cannot claim dense eviction.
    pub(super) pruning: Option<SourcePruneSizing>,
    pub(super) video_vae: u64,
    pub(super) audio_vae: u64,
    /// Load-exact bytes of the configured LoRA stack, held **resident for the whole render**
    /// (sc-18650).
    ///
    /// [`crate::adapters`]'s declared strategy is a *forward-time residual* over
    /// `mlx_gen::adapters::AdaptableLinear` — "never a merged weight", on **every** quant tier,
    /// because a packed leaf cannot absorb a bf16 delta without dequantizing and quant tier is a
    /// creative choice in this product. The factors are therefore still resident at the last denoise
    /// step, which makes [`AdapterResidencyMode::Additive`] the honest mode: `Folded` would declare
    /// zero and under-declare the floor, exactly the failure the text-encoder note below records.
    pub(super) overlay: u64,
    pub(super) native_quant: Option<mlx_gen::gen_core::h3_memory::NativeAllocationFacts>,
}

/// Bytes a `spec.precision`-following component occupies once loaded (SC-22667).
///
/// `LinearNoBias::from_weights` and the DiT block builder call `cast_weights(dtype)` on every
/// **dense** base, and `model.rs` selects that dtype as `Float32` under `Precision::Fp32` and
/// `Bfloat16` otherwise. A bf16-stored dense checkpoint loaded at Fp32 therefore materializes twice
/// its on-disk size, which the plain `safetensors_path_bytes` sum could not see. A packed q4/q8
/// tier is unaffected: `cast_weights` is deliberately not applied to a packed base, and the shared
/// projection detects an existing affine pack by its `.scales` companion and leaves it at stored
/// width.
pub(super) fn materialized_component_bytes(
    path: &std::path::Path,
    precision: mlx_gen::Precision,
) -> u64 {
    if precision == mlx_gen::Precision::Fp32 {
        materialized_f32_bytes(path)
    } else {
        safetensors_path_bytes(path)
    }
}

/// Bytes a component that is materialized f32 regardless of `spec.precision` occupies.
///
/// Falls back to the on-disk sum when the source cannot be read as safetensors. This runs at
/// contract time, ahead of any component load, so an absent or not-yet-staged component must keep
/// its previous accounting rather than turn the contract into a refusal.
pub(super) fn materialized_f32_bytes(path: &std::path::Path) -> u64 {
    let stored = safetensors_path_bytes(path);
    mlx_gen::asset_facts::projected_safetensors_bytes(path, |_| {
        mlx_gen::asset_facts::ResidentProjection::Float32
    })
    .unwrap_or(stored)
}

pub(super) fn native_backend() -> mlx_gen::gen_core::h3_memory::NativeBackend {
    mlx_gen::gen_core::h3_memory::NativeBackend::Mlx {
        tile_columns: mlx_gen::w4a8_int8::TILE_COLS as u64,
    }
}

pub(super) fn dit_allocation(
    path: &std::path::Path,
    precision: mlx_gen::Precision,
    pruned: bool,
) -> mlx_gen::gen_core::Result<gen_core_h3::h3_memory::NativeComponentAllocation> {
    let width = if precision == mlx_gen::Precision::Fp32 {
        4
    } else {
        2
    };
    let mut allocation =
        gen_core_h3::h3_memory::component_allocation(path, native_backend(), width, |h| {
            if pruned
                && (h.name == TABLE_KEY
                    || h.name.contains(".adaln_proj.linear.")
                    || h.name.starts_with("norm_out.linear."))
            {
                None
            } else if h.name.starts_with("transformer_blocks.")
                || h.name.starts_with("token_refiner.")
            {
                Some(width)
            } else {
                None
            }
        })?;
    // Unresolved inventory probes name no tensor the DiT consumes. Retain their established
    // file-size bound; real dense/native sources use the materialized header facts above.
    if allocation.projections.is_empty() && safetensors_path_bytes(path) > 0 {
        let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(path)?;
        if !headers
            .iter()
            .any(|h| h.name.ends_with(".weight") || h.name == TABLE_KEY)
        {
            allocation.resident_bytes = materialized_component_bytes(path, precision);
        }
    }
    Ok(allocation)
}

/// Source headers and the loader's config, without mapping any tensor payload. An absent
/// component retains the existing unresolved declaration; unreadable materialized headers refuse.
pub(super) fn source_prune_sizing(
    dir: &std::path::Path,
) -> mlx_gen::gen_core::Result<Option<SourcePruneSizing>> {
    if safetensors_path_bytes(dir) == 0 {
        return Ok(None);
    }
    let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(dir)?;
    if !headers.iter().any(|h| {
        h.name == TABLE_KEY
            || h.name.ends_with(".adaln_proj.linear.weight")
            || h.name == "norm_out.linear.weight"
    }) {
        return Ok(None);
    }
    let config_path = dir.join("config.json");
    let text = std::fs::read_to_string(&config_path)
        .map_err(|error| CoreError::Unsupported(format!("{}: {error}", config_path.display())))?;
    let config = crate::dit::MiniMaxH3DitConfig::from_diffusers_json(&text)
        .map_err(|error| CoreError::Unsupported(error.to_string()))?;
    let headers = headers
        .into_iter()
        .map(|h| (h.name, (h.dtype, h.shape)))
        .collect::<std::collections::BTreeMap<_, _>>();
    let Some(topology) = PrunedAdaLn::inspect(
        config.hidden_size as usize,
        config.num_layers as usize,
        config.time_embed_dim as usize,
        |name| {
            // Match native curve inspection: packed code columns are not reduced coordinates.
            if let Some(base) = name.strip_suffix(".weight") {
                if headers.contains_key(&format!("{base}.scales"))
                    || headers.contains_key(&format!("{base}.comfy_quant"))
                {
                    return None;
                }
            }
            headers
                .get(name)
                .filter(|(dtype, _)| {
                    matches!(
                        dtype,
                        mlx_gen::gen_core::weightsmeta::Dtype::F16
                            | mlx_gen::gen_core::weightsmeta::Dtype::BF16
                            | mlx_gen::gen_core::weightsmeta::Dtype::F32
                    )
                })
                .cloned()
        },
    )
    .map_err(CoreError::Unsupported)?
    else {
        return Ok(None);
    };
    let rows = SourcePruneSizing::conditioned_timestep_bound(crate::model::MAX_STEPS as usize)
        .map_err(CoreError::Unsupported)?;
    SourcePruneSizing::for_topology(&topology, rows)
        .map(Some)
        .map_err(CoreError::Unsupported)
}

impl ComponentBytes {
    pub(super) fn resolve(spec: &LoadSpec) -> mlx_gen::gen_core::Result<Self> {
        let root = match &spec.weights {
            WeightsSource::Dir(root) => root.clone(),
            WeightsSource::File(path) => path.parent().unwrap_or(path).to_path_buf(),
        };
        let dit = match spec.components.get(DIT_COMPONENT) {
            Some(WeightsSource::Dir(staged)) => staged.clone(),
            _ => root.join(DIT_COMPONENT),
        };
        use mlx_gen::gen_core::h3_memory::{NativeAllocationFacts, NativeTaskAllocation};
        let reference = dit.with_file_name(crate::model::REFERENCE_DIT_PARTITION);
        let backend = native_backend();
        let mut tasks = Vec::new();
        let mut overlay = 0u64;
        let mut dit_bytes = 0;
        let mut pruning = None;
        let mut pruned_resident = 0;
        let mut max_adaln = 0;
        let mut resolved = false;
        // Each render loads ONE task. Correct source-pruned operands before taking the maximum.
        for directory in [&dit, &reference] {
            let sizing = source_prune_sizing(directory)?;
            let allocation = dit_allocation(directory, spec.precision, sizing.is_some())?;
            let bytes = if let Some(sizing) = sizing {
                let live = sizing.resident_bytes().map_err(CoreError::Unsupported)?;
                if live > pruned_resident {
                    pruning = Some(sizing);
                    pruned_resident = live;
                }
                // Preserve source-width pruning operands; S5 alone replaces this subset with
                // the actual F32 compute operands and retained schedule tables.
                sizing
                    .materialized_bytes(allocation.resident_bytes, false)
                    .map_err(CoreError::Unsupported)?
            } else {
                allocation.resident_bytes
            };
            max_adaln = max_adaln.max(allocation.adaln_bytes);
            dit_bytes = dit_bytes.max(bytes);
            resolved |= allocation.resolved;
            let patch_size = if !allocation.resolved {
                [1, 2, 2]
            } else {
                let text = std::fs::read_to_string(directory.join("config.json"))?;
                crate::dit::MiniMaxH3DitConfig::from_diffusers_json(&text)
                    .map_err(|e| CoreError::Unsupported(e.to_string()))?
                    .patch_size
                    .map(|n| n as u64)
            };
            let window = if allocation.resolved {
                Some(
                    mlx_gen::gen_core::h3_memory::window::WindowAllocation::from_resident_tensors(
                        &allocation.phase_resident_by_name,
                        &allocation.projections,
                        bytes.saturating_sub(allocation.resident_bytes),
                        sizing.is_some(),
                    )?
                    .with_source_casts(&allocation.source_cast_by_name)?,
                )
            } else {
                None
            };
            let adapter = if allocation.resolved && !spec.adapters.is_empty() {
                let profile = adapters::profile(
                    directory,
                    &spec.adapters,
                    &allocation.projections,
                    spec.precision,
                )?;
                overlay = overlay.max(profile.resident_bytes);
                Some(profile)
            } else {
                None
            };
            tasks.push(NativeTaskAllocation {
                adapter,
                resident_bytes: bytes,
                constructor_bytes: 0,
                patch_size,
                reference_task: directory == &reference,
                window,
                evicted_bytes: if sizing.is_some() {
                    0
                } else {
                    allocation.adaln_bytes
                },
                table_bytes_per_timestep: if sizing.is_some() {
                    0
                } else {
                    allocation.table_bytes_per_timestep
                },
                projections: allocation.projections,
            });
        }
        let has_native = resolved;
        let mut native_quant = has_native.then_some(NativeAllocationFacts {
            conditioning_stream: None,
            conditioning: None,
            conditioning_resident_bytes: 0,
            conditioning_load_bytes: 0,
            audio_decode_load_bytes: 0,
            video_decode_resident_bytes: 0,
            video_decode_load_bytes: 0,
            audio_decode_resident_bytes: 0,
            default_evaluations: crate::model::DEFAULT_STEPS as usize,
            backend,
            tasks,
        });
        if let Some(facts) = &mut native_quant {
            conditioning::complete(&root, spec.precision, facts)?;
        }
        // **The condition encoder honors its own staged override too** (sc-19120 / sc-20267). It did
        // NOT before, and the omission was `DIT_COMPONENT`'s own bug one component over: sc-19120
        // made the text encoder per-tier and staged **independently** of the DiT, so a split `q4`
        // install stages [`TEXT_ENCODER_COMPONENT`] outside the snapshot and may carry no
        // `root/text_encoder` at all. That measured 0 and published an 18.72 GB **under**-declaration
        // of the largest single component this family loads — and an under-declared floor admits a
        // render that then OOMs, which is the whole failure the ladder exists to prevent.
        //
        // `safetensors_path_bytes` returns 0 for an absent directory, so the flat-snapshot path is
        // byte-identical. Found while porting this contract to the candle lane (sc-20267), where the
        // same line had the same defect; fixed in both rather than recorded in one.
        let text_encoder = match spec.components.get(TEXT_ENCODER_COMPONENT) {
            Some(WeightsSource::Dir(staged)) => staged.clone(),
            _ => root.join(TEXT_ENCODER_COMPONENT),
        };
        let mut conditioning_bytes = safetensors_path_bytes(&text_encoder);
        if let Some(native) = &mut native_quant {
            native.conditioning_stream = conditioning::text_encoder_stream(&text_encoder)?;
            if let Some(stream) = &native.conditioning_stream {
                conditioning_bytes = stream.peak_bytes(None);
            }
        }
        // Fails closed rather than guessing: an additive stack whose file cannot be sized would
        // otherwise be charged 0, and a zero overlay on a render that holds 300 MB of factors is an
        // under-declaration in the OOM direction. The shared helper is the same one ltx uses.
        // Only unresolved inventory fixtures retain the compatibility file bound. Real task
        // partitions use source-selected installed originals/prepared caches above.
        if !resolved && !spec.adapters.is_empty() {
            overlay = mlx_gen::gen_core::adapter_stack_resident_bytes(
                &spec.adapters,
                AdapterResidencyMode::Additive,
            )
            .ok_or_else(|| CoreError::Unsupported("H3 adapter source is unavailable".into()))?;
        }
        Ok(Self {
            // The Qwen3 condition encoder is built at `Dtype::Bfloat16` on every path
            // (`text_encoder/encoder.rs`), so it does not follow `spec.precision`.
            text_encoder: conditioning_bytes,
            dit: dit_bytes,
            adaln: if pruning.is_some() {
                pruned_resident
            } else {
                if has_native {
                    max_adaln
                } else {
                    resolved_adaln_bytes(&dit, dit_bytes)
                }
            },
            pruning,
            video_vae: native_quant
                .as_ref()
                .filter(|n| n.conditioning.is_some())
                .map_or_else(
                    || materialized_component_bytes(&root.join("vae"), spec.precision),
                    |n| n.video_decode_resident_bytes,
                ),
            // The audio VAE is a BigVGAN stack built at `Dtype::Float32` unconditionally, on both
            // the encode and the decode side (`model.rs`) — never at `self.dtype`.
            audio_vae: native_quant
                .as_ref()
                .filter(|n| n.conditioning.is_some())
                .map_or_else(
                    || materialized_f32_bytes(&root.join("audio_vae")),
                    |n| n.audio_decode_resident_bytes,
                ),
            overlay,
            native_quant,
        })
    }

    /// The declaration-only footprint: no filesystem, no resolved tier, and therefore the
    /// architecture's own bf16 figure for the sub-stack.
    pub(super) fn weights_free() -> Self {
        Self {
            text_encoder: 0,
            dit: 0,
            adaln: ADALN_EVICTED_BYTES,
            pruning: None,
            video_vae: 0,
            audio_vae: 0,
            overlay: 0,
            native_quant: None,
        }
    }

    /// The two decoders are one contract field; H3 is the first family with two of them.
    pub(super) fn decoder(&self) -> u64 {
        self.video_vae.saturating_add(self.audio_vae)
    }

    pub(super) fn base(&self) -> u64 {
        self.text_encoder
            .saturating_add(self.dit)
            .saturating_add(self.decoder())
    }
}

/// Device bytes the 50-block `adaln_proj` stack holds on a tier packed at `bits`.
///
/// The same accounting `crate::quant::nbytes` performs on a *loaded* projection, done from the
/// shipped configuration instead of from a device tensor, because a contract is resolved before
/// anything is loaded. Per block:
///
/// * a packed tier holds the triple — `out · in · bits / 8` code bytes, plus a bf16 `scales` **and**
///   `biases` entry per [`crate::quant::GROUP_SIZE`] input group, plus the dense `bias` row;
/// * `bits >= 16` is the unpacked bf16 `weight` + `bias`, i.e. [`ADALN_EVICTED_BYTES`].
///
/// `crate::convert` packs the projections at the same width and group size as the rest of the DiT's
/// linears, so this is a *tier* fact rather than a per-artifact measurement.
pub(super) fn adaln_stack_bytes(bits: i32) -> u64 {
    let config = crate::dit::MiniMaxH3DitConfig::default();
    let out = config.adaln_out_features() as u64;
    let inp = config.time_embed_dim as u64;
    let blocks = DIT_BLOCKS as u64;
    if bits >= 16 {
        return blocks * (out * inp + out) * 2;
    }
    let groups = inp / crate::quant::GROUP_SIZE as u64;
    let codes = out * inp * bits.max(0) as u64 / 8;
    // `scales` and `biases` are one bf16 element per output row per input group.
    let group_metadata = out * groups * 2 * 2;
    let dense_bias = out * 2;
    blocks * (codes + group_metadata + dense_bias)
}

/// The AdaLN sub-stack's resident bytes **on the tier staged at `dit_dir`**, whose footprint is
/// `dit_bytes`.
///
/// [`ADALN_EVICTED_BYTES`] is a bf16 figure. `ComponentBytes::resolve` already honours a staged tier
/// for `transformer_bytes`, so declaring the flat 26.02 GB against an 18.78 GB q4 DiT declares a
/// sub-stack larger than the stack containing it — which `conformance_errors` refuses and
/// `Registry::memory_strategy_contract` turns into a hard error, i.e. a q4 render that cannot
/// resolve a contract at all. **A quant tier is a whole-pipeline contract; this is the segment that
/// was still reading bf16.**
///
/// Two independent legs, and the **smaller** wins:
///
/// * the **marker** leg reads the staged tier's own `config.json` `quantization.bits` — the exact
///   authority `crate::model::reconcile_tier` treats as decisive about which tier is staged — and
///   re-derives the stack at that width through [`adaln_stack_bytes`]. Exact at every shipped tier.
/// * the **footprint** leg scales the bf16 stack by the resolved DiT's share of the bf16 DiT. Never
///   exact — the f32 I/O heads, `context_embedder` and norms `crate::convert` leaves dense in every
///   tier do not shrink, so the whole-DiT ratio sits ~0.7 % above the projections' own at q4 — but
///   it reads nothing except a number `build_contract` already holds.
///
/// Taking the minimum is what makes this **safe** rather than merely accurate, and each leg closes
/// the other's failure:
///
/// * marker alone reports 26.02 GB for a packed tier whose marker is missing or unreadable, which is
///   BLOCKER 1 again;
/// * footprint alone over-declares the eviction, and an over-declared saving is the OOM direction —
///   the exact asymmetry [`ADALN_MODULATION_TABLE_MAX_BYTES`] is chosen on.
///
/// Containment holds unconditionally either way: [`ADALN_EVICTED_BYTES`] is 39.3 % of
/// [`DIT_BF16_BYTES`], so the footprint leg is always strictly below `dit_bytes` and the minimum is
/// therefore always below it too.
///
/// **The floor.** Below a resolved DiT of ~9.86 GB the scaled stack falls under the retained
/// [`ADALN_MODULATION_TABLE_MAX_BYTES`] table and the declared eviction would genuinely exclude
/// nothing, which conformance refuses. No shipped tier is close: `q4` is 18.78 GB, 1.9x above it —
/// `the_shipped_tiers_sit_clear_of_the_floor_where_the_eviction_stops_excluding_anything` pins that.
pub(super) fn resolved_adaln_bytes(dit_dir: &std::path::Path, dit_bytes: u64) -> u64 {
    if dit_bytes == 0 {
        // Nothing was resolved, so there is no tier to scale to and the declaration falls back to
        // the architecture fact. `conformance_errors` skips sub-stack containment against zero asset
        // facts for exactly this case.
        return ADALN_EVICTED_BYTES;
    }
    let marked = mlx_gen::quant::packed_quant_bits_at(dit_dir)
        .ok()
        .flatten()
        .map_or(ADALN_EVICTED_BYTES, adaln_stack_bytes);
    // u128 because `ADALN_EVICTED_BYTES · DIT_BF16_BYTES` is ~1.7e21 and overflows u64.
    let scaled = (u128::from(ADALN_EVICTED_BYTES) * u128::from(dit_bytes)
        / u128::from(DIT_BF16_BYTES)) as u64;
    marked.min(scaled)
}
