//! The [`gen_core::Generator`] adapter for the six YuE stage-1 variants: descriptors, the
//! `GenerationRequest` → [`YueRequest`] mapping, the `LoadSpec` gate, the registrations, and the
//! model-weight licence rows.

use std::path::PathBuf;

use candle_audio::gen_core::{
    self, AudioStem, AudioTrack, Capabilities, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, LoadPhase, LoadSpec, Modality, ModelDescriptor, Progress, Quant,
    WeightsSource,
};

use crate::config::{
    requested_tier, Assets, Guidance, IclReference, IclTracks, Language, Mode, Variant, YueRequest,
};
use crate::engine::{Stage, YueEngine, YueEvent};
use crate::stages::StageSet;
use crate::vocoder::Track;

/// Model family of every variant.
pub const FAMILY: &str = "yue";
/// Output sample rate (the Vocos rate).
pub const SAMPLE_RATE: u32 = crate::vocoder::SAMPLE_RATE;
/// Output channel count (mono mix and stems).
pub const CHANNELS: u16 = 1;

/// The [`LoadSpec::components`] id of the stage-2 snapshot (`m-a-p/YuE-s2-1B-general`).
pub const STAGE2_COMPONENT_ID: &str = "stage2";
/// The [`LoadSpec::components`] id of the `m-a-p/xcodec_mini_infer` snapshot (codec, semantic
/// branch, both Vocos decoders).
pub const XCODEC_COMPONENT_ID: &str = "xcodec";
/// Every component a YuE load requires, beyond the stage-1 snapshot in `LoadSpec::weights`.
pub const REQUIRED_COMPONENTS: &[&str] = &[STAGE2_COMPONENT_ID, XCODEC_COMPONENT_ID];

/// Provenance pins (the checkpoints the licence rows were read from; never fetched here).
pub const STAGE2_HUB_REPO: &str = "m-a-p/YuE-s2-1B-general";
/// See [`STAGE2_HUB_REPO`].
pub const STAGE2_HUB_REVISION: &str = "9dfa90b7013f6b5e7eb5eb2991620dca33058a0e";
/// See [`STAGE2_HUB_REPO`].
pub const XCODEC_HUB_REPO: &str = "m-a-p/xcodec_mini_infer";
/// See [`STAGE2_HUB_REPO`].
pub const XCODEC_HUB_REVISION: &str = "fe781a67815ab47b4a3a5fce1e8d0a692da7e4e5";

/// The pinned stage-1 revision for `variant` (provenance only).
pub fn stage1_hub_revision(variant: Variant) -> &'static str {
    match (variant.language, variant.mode) {
        (Language::En, Mode::Cot) => "454c20e1748888800f8e4b3da45125f55482d967",
        (Language::En, Mode::Icl) => "024ea105533fdd99f8a67ee75abce61c7b813938",
        (Language::Zh, Mode::Cot) => "46b16f41821b3cab5af17146e703a61c3db1af66",
        (Language::Zh, Mode::Icl) => "e631706ec784e48ba646f0fdfc3d4d7fc1fa6d72",
        (Language::JpKr, Mode::Cot) => "9dd546b94b2316c0b994176c98c831d74f66a425",
        (Language::JpKr, Mode::Icl) => "2e34fc94fa01e02b1d3d6f687ac9ac88ebaa9a74",
    }
}

/// The weights-free descriptor for `variant`.
///
/// - **Mode**: ICL variants advertise [`ConditioningKind::ReferenceAudio`] (the reference clip);
///   CoT variants advertise no conditioning, so the shared floor refuses a reference sent to them.
///   Only ICL advertises `supports_reference_region` (the reference window).
/// - **Song controls**: `supports_segmented_lyrics` (segment count + per-segment token budget),
///   `supports_repetition_penalty` and `supports_output_limiter` (clamp vs. rescale) on every
///   variant.
/// - **Tier**: `supported_quants` = q4 + q8 (bf16 is the unquantized load) for both LMs.
/// - **Guidance**: `supports_guidance` — CFG on/off (and its scale) is a request knob.
pub fn descriptor_for(variant: Variant) -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: REQUIRED_COMPONENTS,
        id: variant.id(),
        family: FAMILY,
        backend: "candle",
        modality: Modality::Audio,
        capabilities: Capabilities {
            max_count: 1,
            supports_guidance: true,
            conditioning: match variant.mode {
                Mode::Cot => Vec::new(),
                Mode::Icl => vec![ConditioningKind::ReferenceAudio],
            },
            supported_quants: &[Quant::Q4, Quant::Q8],
            audio_sample_rates: vec![SAMPLE_RATE],
            audio_languages: variant.language.codes().to_vec(),
            supports_segmented_lyrics: true,
            supports_repetition_penalty: true,
            supports_reference_region: variant.mode == Mode::Icl,
            supports_output_limiter: true,
            ..Default::default()
        },
    }
}

/// Refuse an audio field this model does not read, rather than drop it silently.
fn refuse(id: &str, field: &str, why: &str) -> gen_core::Error {
    gen_core::Error::Unsupported(format!("{id}: `{field}` is not a YuE control — {why}"))
}

/// Map a [`GenerationRequest`] onto the engine's [`YueRequest`].
///
/// | request field | YuE control |
/// |---|---|
/// | `prompt` | genre / style tags |
/// | `audio.lyrics` | structured lyrics (required) |
/// | `audio.segments` | segment count (default 2; capped at the lyric sections) |
/// | `audio.max_new_tokens_per_segment` | stage-1 token budget per segment (default 3000) |
/// | `audio.repetition_penalty` | stage-1 repetition penalty (default 1.1) |
/// | `seed` | sampler seed (default 42) |
/// | `guidance` | `None` ⇒ the 1.5 / 1.2 schedule; `0 ..= 1` ⇒ explicitly off; `> 1` ⇒ that scale for every segment; negative / non-finite ⇒ refused |
/// | `ReferenceAudio` conditioning | ICL reference — dual-track when the clip carries `vocals` + `instrumental` stems |
/// | `audio.reference_region` | ICL window (default 0–30 s; open end ⇒ clip end) |
/// | `audio.output_limiter` | `Clamp` (±0.99, default) or `Rescale` (× min(0.99 / peak, 1)) — upstream `save_audio` / `--rescale` |
pub fn map_request(variant: Variant, req: &GenerationRequest) -> gen_core::Result<YueRequest> {
    let id = variant.id();
    let audio = req.audio.clone().unwrap_or_default();
    if req.steps.is_some() {
        return Err(refuse(
            id,
            "steps",
            "YuE is autoregressive; bound it with audio.max_new_tokens_per_segment",
        ));
    }
    if audio.target_duration.is_some() {
        return Err(refuse(
            id,
            "audio.target_duration",
            "song length follows the lyrics and audio.segments",
        ));
    }
    if audio.bpm.is_some() || audio.musical_key.is_some() {
        return Err(refuse(
            id,
            "audio.bpm / audio.musical_key",
            "put tempo and key in the genre tags (prompt)",
        ));
    }
    if audio.voice.is_some() {
        return Err(refuse(
            id,
            "audio.voice",
            "describe the voice in the genre tags",
        ));
    }
    let mut out = YueRequest::new(req.prompt.clone(), audio.lyrics.clone().unwrap_or_default());
    if let Some(n) = audio.segments {
        out.segments = n;
    }
    if let Some(n) = audio.max_new_tokens_per_segment {
        out.decode.max_new_tokens = n;
        out.decode.min_new_tokens = out.decode.min_new_tokens.min(n);
    }
    if let Some(p) = audio.repetition_penalty {
        out.decode.repetition_penalty = p;
    }
    if let Some(l) = audio.output_limiter {
        out.limiter = l;
    }
    if let Some(seed) = req.seed {
        out.seed = seed;
    }
    if let Some(g) = req.guidance {
        if !g.is_finite() || g < 0.0 {
            return Err(gen_core::Error::Msg(format!(
                "{id}: guidance must be a finite, non-negative scale (0..=1 turns CFG off), got {g}"
            )));
        }
        out.decode.guidance = if g <= 1.0 {
            Guidance::Off
        } else {
            Guidance::On { first: g, rest: g }
        };
    }

    let mut references = req.conditioning.iter().filter_map(|c| match c {
        Conditioning::ReferenceAudio { audio, strength } => Some((audio, strength)),
        _ => None,
    });
    if let Some((track, strength)) = references.next() {
        if references.next().is_some() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: pass one ReferenceAudio (carry a dual-track reference as `vocals` + \
                 `instrumental` stems on it)"
            )));
        }
        if strength.is_some() {
            return Err(refuse(
                id,
                "ReferenceAudio.strength",
                "the ICL prompt has no strength",
            ));
        }
        out.icl = Some(icl_reference(id, track, audio.reference_region)?);
    } else if audio.reference_region.is_some() {
        return Err(gen_core::Error::Msg(format!(
            "{id}: audio.reference_region was set without a ReferenceAudio clip"
        )));
    }
    Ok(out)
}

fn icl_reference(
    id: &str,
    track: &AudioTrack,
    region: Option<gen_core::TimeRegion>,
) -> gen_core::Result<IclReference> {
    let stem = |name: &str| track.stems.iter().find(|s| s.name == name);
    let as_track = |samples: &[f32]| AudioTrack {
        samples: samples.to_vec(),
        sample_rate: track.sample_rate,
        channels: track.channels,
        stems: Vec::new(),
    };
    let tracks = match (stem("vocals"), stem("instrumental")) {
        (Some(v), Some(i)) => IclTracks::Dual {
            vocals: as_track(&v.samples),
            instrumental: as_track(&i.samples),
        },
        (None, None) => IclTracks::Single(as_track(&track.samples)),
        _ => {
            return Err(gen_core::Error::Msg(format!(
                "{id}: a dual-track reference needs BOTH `vocals` and `instrumental` stems"
            )))
        }
    };
    let (default_start, default_end) = IclReference::DEFAULT_WINDOW;
    let (start_secs, end_secs) = match region {
        None => (default_start, default_end),
        Some(r) => {
            let frames = track.samples.len() / usize::from(track.channels.max(1));
            let clip_secs = frames as f32 / track.sample_rate.max(1) as f32;
            (r.start_secs, r.end_secs.unwrap_or(clip_secs))
        }
    };
    Ok(IclReference {
        tracks,
        start_secs,
        end_secs,
    })
}

/// A loaded (lazy) YuE generator for one variant.
#[derive(Debug)]
pub struct YueGenerator {
    descriptor: ModelDescriptor,
    engine: YueEngine,
}

impl YueGenerator {
    /// The engine behind this generator.
    pub fn engine(&self) -> &YueEngine {
        &self.engine
    }
}

impl Generator for YueGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        let id = self.descriptor.id;
        self.descriptor
            .capabilities
            .validate_request_audio(id, req)?;
        map_request(self.engine.variant(), req)?.validate(self.engine.variant())
    }

    /// Progress: `Progress::Step` once per finished lyric segment and once per stage-2 track
    /// (`total = segments + 2`), `Progress::Loading(Renderer)` as each LM loads, and one
    /// `Progress::Decoding` before the codec + vocoder pass.
    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let yue = map_request(self.engine.variant(), req)?;
        let mut done = 0u32;
        let mut total = 0u32;
        let mut on_event = |event: YueEvent| match event {
            YueEvent::StageLoaded(Stage::Stage1 | Stage::Stage2) => {
                on_progress(Progress::Loading(LoadPhase::Renderer))
            }
            YueEvent::SegmentStarted { total: n, .. } => total = n as u32 + 2,
            YueEvent::SegmentFinished { .. } | YueEvent::TrackUpsampled(_) => {
                done += 1;
                on_progress(Progress::Step {
                    current: done,
                    total,
                });
            }
            YueEvent::Decoding => on_progress(Progress::Decoding),
            YueEvent::StageLoaded(_) | YueEvent::StageReleased(_) => {}
        };
        let out = self.engine.render(&yue, &req.cancel, &mut on_event)?;
        Ok(GenerationOutput::Audio(AudioTrack {
            samples: out.mix,
            sample_rate: out.sample_rate,
            channels: CHANNELS,
            stems: vec![
                AudioStem {
                    name: Track::Vocals.stem_name().into(),
                    samples: out.vocals,
                },
                AudioStem {
                    name: Track::Instrumental.stem_name().into(),
                    samples: out.instrumental,
                },
            ],
        }))
    }
}

/// Validate a [`LoadSpec`] for `variant` and resolve the staged [`Assets`] and asserted tier.
fn resolve(
    variant: Variant,
    spec: &LoadSpec,
) -> gen_core::Result<(Assets, Option<crate::config::Tier>)> {
    let id = variant.id();
    let stage1 = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{id} expects the stage-1 snapshot directory ({}), not a single file",
                variant.stage1_repo()
            )))
        }
    };
    let tier = requested_tier(id, spec.quantize)?;
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id} does not support LoRA/LoKr adapters"
        )));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id} does not support control/IP-adapter overlays"
        )));
    }
    gen_core::reject_unknown_components(spec, REQUIRED_COMPONENTS, id)?;
    let dir = |component: &str, label: &str| -> gen_core::Result<PathBuf> {
        match gen_core::require_component(spec, component, id, label)? {
            WeightsSource::Dir(p) => Ok(p.clone()),
            WeightsSource::File(p) => Err(gen_core::Error::Msg(format!(
                "{id}: component `{component}` must be a snapshot directory, got file {}",
                p.display()
            ))),
        }
    };
    let assets = Assets {
        stage1,
        stage2: dir(STAGE2_COMPONENT_ID, "YuE stage-2 (s2-1B) snapshot")?,
        xcodec: dir(XCODEC_COMPONENT_ID, "xcodec_mini_infer snapshot")?,
    };
    Ok((assets, tier))
}

/// Construct the (lazy) generator for `variant` over an explicit [`StageSet`] — the public engine
/// entry the end-to-end seam test drives with [`StageSet::stubs`]. Nothing is read from disk here;
/// each stage loads when the render reaches it.
pub fn load_with_stages(
    variant: Variant,
    spec: &LoadSpec,
    stages: StageSet,
) -> gen_core::Result<YueGenerator> {
    let (assets, tier) = resolve(variant, spec)?;
    Ok(YueGenerator {
        descriptor: descriptor_for(variant),
        engine: YueEngine::new(variant, assets, tier, stages),
    })
}

/// Construct the production generator for `variant`.
pub fn load_variant(variant: Variant, spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(load_with_stages(
        variant,
        spec,
        StageSet::production(),
    )?))
}

macro_rules! variant_entry_points {
    ($($desc:ident, $load:ident, $reg:ident => $lang:ident, $mode:ident;)*) => {$(
        #[doc = concat!("Descriptor of the `", stringify!($lang), "`/`", stringify!($mode), "` variant.")]
        pub fn $desc() -> ModelDescriptor {
            descriptor_for(Variant::new(Language::$lang, Mode::$mode))
        }
        #[doc = concat!("Load the `", stringify!($lang), "`/`", stringify!($mode), "` variant.")]
        pub fn $load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
            load_variant(Variant::new(Language::$lang, Mode::$mode), spec)
        }
        candle_audio::register_generators! {
            pub const $reg = $desc => $load
        }
    )*};
}

variant_entry_points! {
    descriptor_en_cot, load_en_cot, REGISTRATION_EN_COT => En, Cot;
    descriptor_en_icl, load_en_icl, REGISTRATION_EN_ICL => En, Icl;
    descriptor_zh_cot, load_zh_cot, REGISTRATION_ZH_COT => Zh, Cot;
    descriptor_zh_icl, load_zh_icl, REGISTRATION_ZH_ICL => Zh, Icl;
    descriptor_jp_kr_cot, load_jp_kr_cot, REGISTRATION_JP_KR_COT => JpKr, Cot;
    descriptor_jp_kr_icl, load_jp_kr_icl, REGISTRATION_JP_KR_ICL => JpKr, Icl;
}

/// Every variant's registration, in catalog order ([`Variant::ALL`]).
pub const REGISTRATIONS: [gen_core::registry::ModelRegistration; 6] = [
    REGISTRATION_EN_COT,
    REGISTRATION_EN_ICL,
    REGISTRATION_ZH_COT,
    REGISTRATION_ZH_ICL,
    REGISTRATION_JP_KR_COT,
    REGISTRATION_JP_KR_ICL,
];

// ---------------------------------------------------------------------------------------------
// Model-weight licences (schema 3). Disclosure only: each row records what the upstream model card
// declares (all eight read `license:apache-2.0`, ungated, on `retrieved`).
// ---------------------------------------------------------------------------------------------

const ATTRIBUTION: &str =
    "YuE © Multimodal Art Projection (M-A-P) and HKUST — licensed under Apache-2.0";
const RETRIEVED: &str = "2026-09-24";

macro_rules! apache_row {
    ($name:ident, $key:literal, $repo:literal) => {
        #[doc = concat!("Licence row for `", $repo, "`.")]
        pub const $name: gen_core::ComponentLicense = gen_core::ComponentLicense {
            component: $key,
            source_url: concat!("https://huggingface.co/", $repo),
            gated: false,
            declared: "apache-2.0",
            family: "apache-2-0",
            attribution: Some(ATTRIBUTION),
            retrieved: RETRIEVED,
        };
    };
}

apache_row!(
    LICENSE_S1_EN_COT,
    "yue_s1_7b_anneal_en_cot",
    "m-a-p/YuE-s1-7B-anneal-en-cot"
);
apache_row!(
    LICENSE_S1_EN_ICL,
    "yue_s1_7b_anneal_en_icl",
    "m-a-p/YuE-s1-7B-anneal-en-icl"
);
apache_row!(
    LICENSE_S1_ZH_COT,
    "yue_s1_7b_anneal_zh_cot",
    "m-a-p/YuE-s1-7B-anneal-zh-cot"
);
apache_row!(
    LICENSE_S1_ZH_ICL,
    "yue_s1_7b_anneal_zh_icl",
    "m-a-p/YuE-s1-7B-anneal-zh-icl"
);
apache_row!(
    LICENSE_S1_JP_KR_COT,
    "yue_s1_7b_anneal_jp_kr_cot",
    "m-a-p/YuE-s1-7B-anneal-jp-kr-cot"
);
apache_row!(
    LICENSE_S1_JP_KR_ICL,
    "yue_s1_7b_anneal_jp_kr_icl",
    "m-a-p/YuE-s1-7B-anneal-jp-kr-icl"
);
apache_row!(LICENSE_S2, "yue_s2_1b_general", "m-a-p/YuE-s2-1B-general");
apache_row!(
    LICENSE_XCODEC,
    "xcodec_mini_infer",
    "m-a-p/xcodec_mini_infer"
);

/// Every artifact the YuE providers load — one row each.
pub const COMPONENT_LICENSES: &[gen_core::ComponentLicense] = &[
    LICENSE_S1_EN_COT,
    LICENSE_S1_EN_ICL,
    LICENSE_S1_ZH_COT,
    LICENSE_S1_ZH_ICL,
    LICENSE_S1_JP_KR_COT,
    LICENSE_S1_JP_KR_ICL,
    LICENSE_S2,
    LICENSE_XCODEC,
];

/// Provider → component mapping: each variant loads its own stage-1 checkpoint plus the shared
/// stage-2 and xcodec snapshots.
pub const PROVIDER_COMPONENTS: &[gen_core::ProviderComponents] = &[
    gen_core::ProviderComponents {
        provider_id: "yue_en_cot",
        components: &[
            "yue_s1_7b_anneal_en_cot",
            "yue_s2_1b_general",
            "xcodec_mini_infer",
        ],
    },
    gen_core::ProviderComponents {
        provider_id: "yue_en_icl",
        components: &[
            "yue_s1_7b_anneal_en_icl",
            "yue_s2_1b_general",
            "xcodec_mini_infer",
        ],
    },
    gen_core::ProviderComponents {
        provider_id: "yue_zh_cot",
        components: &[
            "yue_s1_7b_anneal_zh_cot",
            "yue_s2_1b_general",
            "xcodec_mini_infer",
        ],
    },
    gen_core::ProviderComponents {
        provider_id: "yue_zh_icl",
        components: &[
            "yue_s1_7b_anneal_zh_icl",
            "yue_s2_1b_general",
            "xcodec_mini_infer",
        ],
    },
    gen_core::ProviderComponents {
        provider_id: "yue_jp_kr_cot",
        components: &[
            "yue_s1_7b_anneal_jp_kr_cot",
            "yue_s2_1b_general",
            "xcodec_mini_infer",
        ],
    },
    gen_core::ProviderComponents {
        provider_id: "yue_jp_kr_icl",
        components: &[
            "yue_s1_7b_anneal_jp_kr_icl",
            "yue_s2_1b_general",
            "xcodec_mini_infer",
        ],
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::gen_core::{AudioParams, OutputLimiter, TimeRegion};

    fn cot() -> Variant {
        Variant::new(Language::En, Mode::Cot)
    }

    fn icl() -> Variant {
        Variant::new(Language::En, Mode::Icl)
    }

    fn song(audio: AudioParams) -> GenerationRequest {
        GenerationRequest {
            prompt: "uplifting pop".into(),
            audio: Some(AudioParams {
                lyrics: Some("[verse]\nhello".into()),
                ..audio
            }),
            ..Default::default()
        }
    }

    fn clip(stems: Vec<AudioStem>) -> AudioTrack {
        AudioTrack {
            samples: vec![0.1; 16_000 * 40],
            sample_rate: 16_000,
            channels: 1,
            stems,
        }
    }

    #[test]
    fn descriptors_advertise_mode_tier_and_language() {
        let d = descriptor_for(icl());
        assert_eq!(d.id, "yue_en_icl");
        assert_eq!(
            d.capabilities.conditioning,
            [ConditioningKind::ReferenceAudio]
        );
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert_eq!(d.capabilities.audio_sample_rates, [44_100]);
        assert_eq!(d.required_components, REQUIRED_COMPONENTS);
        let d = descriptor_for(Variant::new(Language::JpKr, Mode::Cot));
        assert!(d.capabilities.conditioning.is_empty());
        assert_eq!(d.capabilities.audio_languages, ["ja", "ko"]);
    }

    #[test]
    fn every_r5_knob_maps_and_defaults_stay_defaults() {
        let r = map_request(cot(), &song(AudioParams::default())).unwrap();
        assert_eq!(
            r.limiter,
            OutputLimiter::Clamp,
            "upstream's default limiter"
        );
        assert_eq!(r, {
            let mut want = YueRequest::new("uplifting pop", "[verse]\nhello");
            want.icl = None;
            want
        });

        let mut req = song(AudioParams {
            segments: Some(5),
            max_new_tokens_per_segment: Some(50),
            repetition_penalty: Some(1.3),
            output_limiter: Some(OutputLimiter::Rescale),
            ..Default::default()
        });
        req.seed = Some(7);
        req.guidance = Some(0.0);
        let r = map_request(cot(), &req).unwrap();
        assert_eq!(r.segments, 5);
        assert_eq!(r.decode.max_new_tokens, 50);
        assert_eq!(
            r.decode.min_new_tokens, 50,
            "the floor never exceeds the budget"
        );
        assert_eq!(r.decode.repetition_penalty, 1.3);
        assert_eq!(r.limiter, OutputLimiter::Rescale);
        assert_eq!(r.seed, 7);
        assert_eq!(r.decode.guidance, Guidance::Off);
        req.guidance = Some(1.0);
        let r = map_request(cot(), &req).unwrap();
        assert_eq!(r.decode.guidance, Guidance::Off, "1.0 is the off boundary");
        for bad in [-0.5, f32::NAN, f32::INFINITY] {
            req.guidance = Some(bad);
            assert!(
                matches!(map_request(cot(), &req), Err(gen_core::Error::Msg(_))),
                "guidance {bad} must be refused"
            );
        }
        req.guidance = Some(2.0);
        let r = map_request(cot(), &req).unwrap();
        assert_eq!(
            r.decode.guidance,
            Guidance::On {
                first: 2.0,
                rest: 2.0
            }
        );
    }

    /// The ICL window as a plain `(start, end)` pair of seconds.
    fn window(r: &IclReference) -> (f32, f32) {
        (r.start_secs, r.end_secs)
    }

    #[test]
    fn icl_reference_maps_single_dual_and_window() {
        let mut req = song(AudioParams::default());
        req.conditioning = vec![Conditioning::ReferenceAudio {
            audio: clip(Vec::new()),
            strength: None,
        }];
        let r = map_request(icl(), &req).unwrap();
        let reference = r.icl.unwrap();
        assert!(matches!(reference.tracks, IclTracks::Single(_)));
        assert_eq!(window(&reference), (0.0, 30.0));

        let stem = |name: &str| AudioStem {
            name: name.into(),
            samples: vec![0.2; 16_000 * 40],
        };
        req.conditioning = vec![Conditioning::ReferenceAudio {
            audio: clip(vec![stem("vocals"), stem("instrumental")]),
            strength: None,
        }];
        req.audio.as_mut().unwrap().reference_region = Some(TimeRegion {
            start_secs: 5.0,
            end_secs: None,
        });
        let reference = map_request(icl(), &req).unwrap().icl.unwrap();
        assert!(matches!(reference.tracks, IclTracks::Dual { .. }));
        assert_eq!(window(&reference), (5.0, 40.0));

        req.conditioning = vec![Conditioning::ReferenceAudio {
            audio: clip(vec![stem("vocals")]),
            strength: None,
        }];
        assert!(
            map_request(icl(), &req).is_err(),
            "half a dual-track reference"
        );
    }

    #[test]
    fn unread_fields_are_refused_not_dropped() {
        for audio in [
            AudioParams {
                target_duration: Some(30.0),
                ..Default::default()
            },
            AudioParams {
                bpm: Some(120.0),
                ..Default::default()
            },
            AudioParams {
                voice: Some("x".into()),
                ..Default::default()
            },
        ] {
            assert!(matches!(
                map_request(cot(), &song(audio)),
                Err(gen_core::Error::Unsupported(_))
            ));
        }
        let mut req = song(AudioParams::default());
        req.steps = Some(4);
        assert!(matches!(
            map_request(cot(), &req),
            Err(gen_core::Error::Unsupported(_))
        ));
    }

    #[test]
    fn load_gate_requires_both_components_and_a_published_tier() {
        let root = tempfile::tempdir().unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().join("s1")))
            .with_component(
                STAGE2_COMPONENT_ID,
                WeightsSource::Dir(root.path().join("s2")),
            )
            .with_component(
                XCODEC_COMPONENT_ID,
                WeightsSource::Dir(root.path().join("xc")),
            );
        assert!(load_en_cot(&spec).is_ok());
        gen_core_testkit::check_component_load_gate(load_en_cot, &spec, REQUIRED_COMPONENTS)
            .expect("missing / unknown components must fail at load");
        let mut q = spec.clone();
        q.quantize = Some(Quant::Nvfp4);
        assert!(matches!(
            load_en_cot(&q),
            Err(gen_core::Error::Unsupported(_))
        ));
        q.quantize = Some(Quant::Q4);
        let g = load_with_stages(cot(), &q, StageSet::stubs()).unwrap();
        assert_eq!(g.engine().tier(), Some(crate::config::Tier::Q4));
        let file = LoadSpec::new(WeightsSource::File(root.path().join("x.safetensors")));
        assert!(load_en_cot(&file).is_err());
    }

    #[test]
    fn component_rows_are_well_formed_apache() {
        for row in COMPONENT_LICENSES {
            assert!(
                row.is_well_formed(gen_core::LICENSE_FAMILIES),
                "{}",
                row.component
            );
        }
        assert_eq!(PROVIDER_COMPONENTS.len(), Variant::ALL.len());
        for (p, v) in PROVIDER_COMPONENTS.iter().zip(Variant::ALL) {
            assert_eq!(p.provider_id, v.id());
        }
    }
}
