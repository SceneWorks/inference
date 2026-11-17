//! The header-derived **warm** estimate of one H3 video request (see `crate::model`'s module docs).
//!
//! The staged render maps and releases each component in its own phase, so its device peak is the
//! **maximum** of its phases. A resident request may instead keep every component it loads on the
//! device between requests: the layer-50 conditioner with the vision tower, the evicted DiT of the
//! selected partition with its whole-schedule modulation table and any adapter stack, the video
//! VAE (both halves when the request encodes keyframes or references, else the decode half), the
//! reference-soundtrack encoder, and the audio VAE decoder. Its peak is then every warm component
//! at once plus the largest working set any one phase adds on top of its own component.
//!
//! Every figure comes from the load's [`NativeAllocationFacts`] (safetensors headers, configs and
//! file sizes) and the request's own geometry — the same inventory the worker's admission prices
//! the staged phases from. An estimate, not a measurement.

use std::path::Path;

use candle_gen::gen_core::h3_memory::{NativeAllocationFacts, NativeRows, NativeTaskSelection};
use candle_gen::gen_core::veda::workspace::{Request, CUDA_DENSE_SCORE_ELEMENTS};
use candle_gen::gen_core::{Error as CoreError, GenerationRequest, MemoryProviderContract};
use gen_core_h3::h3_memory::NativeAllocationFactsExt;

use crate::image_memory_strategy::admission::{
    block_working_bytes, minimax_vision_dims, VISION_PATCH,
};
use crate::MODEL_ID;

/// The device bytes a resident H3 video generator keeps warm between requests, per component.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VideoWarmWeights {
    /// The layer-50 conditioner (50 layers at their runtime format, rotary tables) and the f32
    /// vision tower beside it.
    pub conditioner: u64,
    /// The selected partition's DiT after the AdaLN precompute-and-evict, its retained
    /// modulation table over this request's schedule, and the configured adapter stack.
    pub dit: u64,
    /// The video VAE's decode half.
    pub video_vae_decode: u64,
    /// The video VAE's two halves plus the reference-soundtrack encoder, which an encoding request
    /// (fl2va / ref2va) keeps in the decode half's place.
    pub encoders: u64,
    /// The audio VAE's decode half (the vocoder).
    pub audio_decoder: u64,
    /// Whether the request encodes keyframes or references, so [`Self::encoders`] is held.
    pub encodes: bool,
}

impl VideoWarmWeights {
    /// Every component this request keeps warm, together.
    pub fn total(&self) -> u64 {
        self.conditioner
            .saturating_add(self.dit)
            .saturating_add(if self.encodes {
                self.encoders
            } else {
                self.video_vae_decode
            })
            .saturating_add(self.audio_decoder)
    }
}

/// One request's staged and warm device peaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoWarmEstimate {
    pub weights: VideoWarmWeights,
    /// The staged render's peak: the largest single phase.
    pub staged_peak: u64,
    /// The warm render's peak: every warm component plus the largest phase working set.
    pub warm_peak: u64,
    /// The DiT partition this request denoises from.
    pub task: NativeTaskSelection,
}

fn add_all(values: impl IntoIterator<Item = u64>) -> candle_gen::gen_core::Result<u64> {
    candle_gen::gen_core::checked_byte_total(MODEL_ID, "warm estimate", values)
}

fn mul(a: u64, b: u64) -> candle_gen::gen_core::Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| CoreError::Unsupported(format!("{MODEL_ID}: warm estimate size overflow")))
}

/// The warm and staged peaks of `req` on `contract`'s source, at `evaluations` model evaluations.
/// `te_dir` is the resolved text-encoder directory (its config sizes the conditioner's working
/// set) and `dit_hidden` the selected partition's hidden width. `Err` when the source carries no
/// native allocation inventory, which keeps nothing warm.
pub fn video_warm_estimate(
    contract: &MemoryProviderContract,
    te_dir: &Path,
    dit_hidden: u64,
    req: &GenerationRequest,
    evaluations: usize,
) -> candle_gen::gen_core::Result<VideoWarmEstimate> {
    let native = contract
        .phase_facts
        .as_ref()
        .and_then(|facts| facts.native_quant.as_ref())
        .ok_or_else(|| {
            CoreError::Unsupported(format!(
                "{MODEL_ID}: the source has no native allocation inventory to price a warm set"
            ))
        })?;
    let request: Request<'_> = req.into();
    let rows = native.request_rows(request, evaluations)?;
    let patch = native.selected_patch(request)?;
    let conditioning = native.conditioning.as_ref().ok_or_else(|| {
        CoreError::Unsupported(format!(
            "{MODEL_ID}: the source has no VAE allocation facts to price a warm set"
        ))
    })?;
    let workspace =
        gen_core_h3::h3_conditioning_memory::workspace_with_patch(&request, conditioning, patch)?;
    let encodes = !req.conditioning.is_empty();
    let weights = warm_weights(contract, native, rows, encodes)?;

    // --- conditioning: the vision tower (fl2va / ref2va), then the layer-50 tap ----------------
    let te_cfg = crate::text_encoder::MiniMaxH3TeConfig::from_component_dir(te_dir)
        .map_err(|e| CoreError::Unsupported(e.to_string()))?;
    let (_, _, te_decode) = super::runtime_payloads(te_dir, true)?;
    let f32_bytes = 4;
    let te_work = add_all([
        block_working_bytes(
            rows.conditioning,
            te_cfg.hidden_size as u64,
            te_cfg.num_heads as u64,
            te_cfg.num_kv_heads as u64,
            te_cfg.head_dim as u64,
            te_cfg.intermediate_size as u64,
            f32_bytes,
            CUDA_DENSE_SCORE_ELEMENTS,
        )?,
        te_decode,
        // The f32 layer-50 context the later phases carry.
        mul(
            mul(rows.conditioning, te_cfg.hidden_size as u64)?,
            f32_bytes,
        )?,
    ])?;
    let vision_work = if encodes {
        // One source at a time, bounded by the canvas budget every keyframe and reference is
        // fitted inside: the tower's live block plus the f32 patch pixels.
        let patches = u64::from(crate::pipeline::CANVAS_MAX_PIXELS / (VISION_PATCH * VISION_PATCH));
        let (hidden, heads, head_dim, mlp) = minimax_vision_dims();
        add_all([
            block_working_bytes(
                patches,
                hidden,
                heads,
                heads,
                head_dim,
                mlp,
                f32_bytes,
                CUDA_DENSE_SCORE_ELEMENTS,
            )?,
            mul(
                patches,
                u64::from(VISION_PATCH * VISION_PATCH) * 3 * f32_bytes * 2,
            )?,
        ])?
    } else {
        0
    };
    let conditioning_work = te_work.max(vision_work);

    // --- reference / keyframe encode -------------------------------------------------------------
    let encode_work = if encodes {
        workspace.conditioning_bytes
    } else {
        0
    };

    // --- denoise ----------------------------------------------------------------------------------
    let mut dit_peak = native.resident_peak_bytes_windowed(rows, None)?;
    if let Some(bound) = super::veda_workspace_bound_for_contract(request, contract)? {
        dit_peak = dit_peak.max(native.denoise_peak_bytes_with_veda(rows, None, &bound)?);
    }
    // The packed rows' carried hidden state and velocity, f32, over the whole sequence.
    let sampler = mul(mul(rows.packed, dit_hidden)?, f32_bytes * 2)?;
    let overlay = contract.asset_facts.overlay_bytes;
    let denoise = add_all([dit_peak, overlay, sampler])?;

    let phases = PhaseBytes {
        conditioning_work,
        encode: encodes.then_some((native.conditioning_resident_bytes, encode_work)),
        denoise,
        video_decode: (
            native.video_decode_resident_bytes,
            workspace.video_decode_bytes,
        ),
        audio_decode: (
            native.audio_decode_resident_bytes,
            workspace.audio_decode_bytes,
        ),
    };
    let (staged_peak, warm_peak) = peaks(&weights, &phases)?;
    Ok(VideoWarmEstimate {
        weights,
        staged_peak,
        warm_peak,
        task: rows.task,
    })
}

/// Each phase's device bytes, as the staged render allocates them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PhaseBytes {
    /// The vision tower's or the conditioner's working set, whichever is larger, above their
    /// weights ([`VideoWarmWeights::conditioner`]).
    conditioning_work: u64,
    /// `(weights, working set)` of the keyframe / reference encode, when the request encodes.
    encode: Option<(u64, u64)>,
    /// The whole denoise phase: the DiT's load, precompute and denoise peak with the adapter
    /// stack and the sampler state.
    denoise: u64,
    /// `(decode-half weights, working set)` of the video decode.
    video_decode: (u64, u64),
    /// `(weights, working set)` of the audio decode.
    audio_decode: (u64, u64),
}

/// `(staged, warm)` device peaks. Staged: the largest phase, each holding only its own
/// component. Warm: every warm component at once, plus the largest working set any phase adds
/// above its own warm component — for the denoise phase, everything above the warm DiT, which
/// includes the full-stack load and precompute before the eviction.
fn peaks(
    weights: &VideoWarmWeights,
    phases: &PhaseBytes,
) -> candle_gen::gen_core::Result<(u64, u64)> {
    let (encode_weights, encode_work) = phases.encode.unwrap_or((0, 0));
    let staged = [
        add_all([weights.conditioner, phases.conditioning_work])?,
        add_all([encode_weights, encode_work])?,
        phases.denoise,
        add_all([phases.video_decode.0, phases.video_decode.1])?,
        add_all([phases.audio_decode.0, phases.audio_decode.1])?,
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    let working = [
        phases.conditioning_work,
        encode_work,
        phases.denoise.saturating_sub(weights.dit),
        phases.video_decode.1,
        phases.audio_decode.1,
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    Ok((staged, add_all([weights.total(), working])?))
}

/// See [`VideoWarmWeights`].
fn warm_weights(
    contract: &MemoryProviderContract,
    native: &NativeAllocationFacts,
    rows: NativeRows,
    encodes: bool,
) -> candle_gen::gen_core::Result<VideoWarmWeights> {
    let conditioner = native
        .conditioning_stream
        .as_ref()
        .map_or(contract.asset_facts.conditioning_bytes, |stream| {
            stream.peak_bytes(None)
        });
    let dit = native
        .tasks
        .iter()
        .filter(|task| match rows.task {
            NativeTaskSelection::Base => !task.reference_task,
            NativeTaskSelection::Reference => task.reference_task,
            NativeTaskSelection::All => true,
        })
        .map(|task| {
            add_all([
                task.resident_bytes.saturating_sub(task.evicted_bytes),
                mul(task.table_bytes_per_timestep, rows.schedule)?,
            ])
        })
        .try_fold(0u64, |max, bytes| bytes.map(|bytes| max.max(bytes)))?;
    Ok(VideoWarmWeights {
        conditioner,
        dit: add_all([dit, contract.asset_facts.overlay_bytes])?,
        video_vae_decode: native.video_decode_resident_bytes,
        encoders: native.conditioning_resident_bytes,
        audio_decoder: native.audio_decode_resident_bytes,
        encodes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    fn t2va() -> (VideoWarmWeights, PhaseBytes) {
        let weights = VideoWarmWeights {
            conditioner: 30 * GB,
            dit: 25 * GB,
            video_vae_decode: 5 * GB,
            encoders: 6 * GB,
            audio_decoder: GB,
            encodes: false,
        };
        let phases = PhaseBytes {
            conditioning_work: 2 * GB,
            encode: None,
            denoise: 40 * GB,
            video_decode: (5 * GB, 8 * GB),
            audio_decode: (GB, GB / 2),
        };
        (weights, phases)
    }

    /// The staged peak is the largest single phase; the warm peak is every warm component plus
    /// the largest working set any phase adds above its own component.
    #[test]
    fn the_warm_peak_holds_every_component_and_the_largest_working_set() {
        let (weights, phases) = t2va();
        let (staged, warm) = peaks(&weights, &phases).unwrap();
        assert_eq!(staged, 40 * GB, "the denoise phase binds the staged render");
        assert_eq!(
            weights.total(),
            61 * GB,
            "the decode half, not the encoders"
        );
        // The denoise phase adds 15 GB above the warm DiT (its load, precompute and sampler).
        assert_eq!(warm, 61 * GB + 15 * GB);

        // An encoding request holds both VAE halves and the soundtrack encoder instead of the
        // decode half, and its encode working set can bind.
        let weights = VideoWarmWeights {
            encodes: true,
            ..weights
        };
        let phases = PhaseBytes {
            encode: Some((6 * GB, 20 * GB)),
            ..phases
        };
        let (staged, warm) = peaks(&weights, &phases).unwrap();
        assert_eq!(staged, 40 * GB);
        assert_eq!(weights.total(), 62 * GB);
        assert_eq!(warm, 62 * GB + 20 * GB, "the encode working set binds");
    }

    /// A source with no native allocation inventory prices no warm set, so it stages.
    #[test]
    fn a_source_without_native_facts_keeps_nothing_warm() {
        let contract = super::super::weights_free_contract(&candle_gen::gen_core::LoadSpec::new(
            candle_gen::gen_core::WeightsSource::Dir("/nonexistent".into()),
        ))
        .unwrap();
        let req = GenerationRequest {
            width: 1344,
            height: 768,
            frames: Some(124),
            ..Default::default()
        };
        assert!(video_warm_estimate(&contract, Path::new("/nonexistent"), 5120, &req, 50).is_err());
    }

    /// **Header-derived per-tier peaks on a real snapshot** — reads safetensors headers and
    /// configs only; no tensor is loaded and no model runs. Ignored by default: set
    /// `MINIMAX_H3_SNAPSHOT` to the upstream `MiniMaxAI/MiniMax-H3` snapshot and
    /// `MINIMAX_H3_TIER_SNAPSHOT` to the `SceneWorks/minimax-h3-mlx` snapshot (with `bf16/`, `q8/`,
    /// `q4/` subtrees) and run with `--ignored --nocapture`.
    #[test]
    #[ignore = "reads a real H3 snapshot's headers; set MINIMAX_H3_SNAPSHOT and MINIMAX_H3_TIER_SNAPSHOT"]
    fn header_derived_per_tier_peaks() {
        use candle_gen::gen_core::{AudioTrack, Conditioning, Image, LoadSpec, WeightsSource};
        let root = std::path::PathBuf::from(std::env::var("MINIMAX_H3_SNAPSHOT").unwrap());
        let tiers = std::path::PathBuf::from(std::env::var("MINIMAX_H3_TIER_SNAPSHOT").unwrap());
        let image = |w: u32, h: u32| Image {
            width: w,
            height: h,
            pixels: vec![0; (w * h * 3) as usize],
        };
        let audio = AudioTrack {
            samples: vec![0.0; 32_000 * 6 * 2],
            sample_rate: crate::audio_config::AUDIO_SAMPLE_RATE,
            channels: 2,
            stems: Vec::new(),
        };
        let modes: Vec<(&str, Vec<Conditioning>)> = vec![
            ("t2va", Vec::new()),
            (
                "fl2va (first+last keyframe)",
                vec![
                    Conditioning::Keyframe {
                        image: image(1344, 768),
                        frame_idx: 0,
                        strength: 1.0,
                    },
                    Conditioning::Keyframe {
                        image: image(1344, 768),
                        frame_idx: -1,
                        strength: 1.0,
                    },
                ],
            ),
            (
                "ref2va (2 images + audio ref)",
                vec![
                    Conditioning::Reference {
                        image: image(1344, 768),
                        strength: None,
                    },
                    Conditioning::Reference {
                        image: image(768, 1344),
                        strength: None,
                    },
                    Conditioning::ReferenceAudio {
                        audio: audio.clone(),
                        strength: None,
                    },
                ],
            ),
        ];
        for tier in ["bf16", "q8", "q4"] {
            let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
            spec.components.insert(
                crate::tier::DIT_COMPONENT.into(),
                WeightsSource::Dir(tiers.join(tier).join("transformer")),
            );
            let te_dir = if tier == "bf16" {
                root.join("text_encoder")
            } else {
                tiers.join(tier).join("text_encoder")
            };
            spec.components.insert(
                crate::tier::TEXT_ENCODER_COMPONENT.into(),
                WeightsSource::Dir(te_dir.clone()),
            );
            let contract = super::super::contract_for(&spec).unwrap();
            let cfg = crate::dit::MiniMaxH3DitConfig::from_diffusers_json(
                &std::fs::read_to_string(tiers.join(tier).join("transformer/config.json")).unwrap(),
            )
            .unwrap();
            for frames in [124u32, 345] {
                for (mode, conditioning) in &modes {
                    let req = GenerationRequest {
                        width: 1344,
                        height: 768,
                        frames: Some(frames),
                        conditioning: conditioning.clone(),
                        ..Default::default()
                    };
                    let e =
                        video_warm_estimate(&contract, &te_dir, cfg.hidden_size as u64, &req, 50)
                            .unwrap();
                    let w = e.weights;
                    let gb = |b: u64| b as f64 / 1e9;
                    println!(
                        "{tier} {frames}f {mode}: staged {:.1} GB, warm {:.1} GB \
                         (te+vision {:.1}, dit {:.1}, vae {:.1}, audio dec {:.1})",
                        gb(e.staged_peak),
                        gb(e.warm_peak),
                        gb(w.conditioner),
                        gb(w.dit),
                        gb(if w.encodes {
                            w.encoders
                        } else {
                            w.video_vae_decode
                        }),
                        gb(w.audio_decoder),
                    );
                }
            }
        }
    }
}
