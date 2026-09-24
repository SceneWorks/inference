//! The end-to-end seam test: one public engine call (`Generator::generate` on a generator built by
//! [`crate::load_with_stages`]) runs every stage and returns the mix plus both stems. It drives the
//! weights-free [`StageSet::stubs`] wiring, so it stays green as each stage story swaps its
//! production loader — later stories keep this file passing.
//!
//! Lives in the lib (not `tests/`) so the CPU lane's `--lib` run executes it.

use std::sync::{Arc, Mutex};

use candle_audio::candle_core::Tensor;
use candle_audio::gen_core::{
    AudioParams, AudioStem, AudioTrack, CancelFlag, Conditioning, Error, GenerationOutput,
    GenerationRequest, Generator, LoadSpec, Progress, WeightsSource,
};

use crate::codec::CodecDecoder;
use crate::config::{Assets, IclReference, Language, Mode, Variant};
use crate::icl::{IclEncoder, IclPromptCodes};
use crate::model::{load_with_stages, STAGE2_COMPONENT_ID, XCODEC_COMPONENT_ID};
use crate::stage1::{SegmentStart, Stage1Model, Stage1Step, STUB_FRAMES_PER_SEGMENT};
use crate::stage2::Stage2Model;
use crate::stages::StageSet;
use crate::tokens::CodecFrames;
use crate::vocoder::{Track, Vocoder, SAMPLES_PER_FRAME};

type Log = Arc<Mutex<Vec<String>>>;

fn push(log: &Log, entry: &str) {
    log.lock().unwrap().push(entry.to_string());
}

/// A stage handle that records its own drop.
struct Logged<T: ?Sized> {
    name: &'static str,
    log: Log,
    inner: Box<T>,
}

impl<T: ?Sized> Drop for Logged<T> {
    fn drop(&mut self) {
        push(&self.log, &format!("{}:drop", self.name));
    }
}

impl IclEncoder for Logged<dyn IclEncoder> {
    fn encode(
        &self,
        reference: &IclReference,
        cancel: &CancelFlag,
    ) -> crate::gen_core::Result<IclPromptCodes> {
        self.inner.encode(reference, cancel)
    }
}

/// Stage 1 wrapper: counts `step` calls and trips `trip` on the `trip_at`-th one.
struct Stage1Probe {
    logged: Logged<dyn Stage1Model>,
    steps: Arc<Mutex<usize>>,
    trip: Option<(usize, CancelFlag)>,
}

impl Stage1Model for Stage1Probe {
    fn begin_render(&mut self, seed: u64) -> crate::gen_core::Result<()> {
        self.logged.inner.begin_render(seed)
    }
    fn begin_segment(&mut self, segment: &SegmentStart<'_>) -> crate::gen_core::Result<()> {
        self.logged.inner.begin_segment(segment)
    }
    fn step(&mut self) -> crate::gen_core::Result<Stage1Step> {
        let mut steps = self.steps.lock().unwrap();
        *steps += 1;
        if let Some((at, flag)) = &self.trip {
            if *steps == *at {
                flag.cancel();
            }
        }
        self.logged.inner.step()
    }
    fn end_segment(&mut self) -> crate::gen_core::Result<()> {
        self.logged.inner.end_segment()
    }
}

impl Stage2Model for Logged<dyn Stage2Model> {
    fn upsample(
        &mut self,
        cb0: &[u32],
        cancel: &CancelFlag,
    ) -> crate::gen_core::Result<CodecFrames> {
        self.inner.upsample(cb0, cancel)
    }
}

impl CodecDecoder for Logged<dyn CodecDecoder> {
    fn decode(
        &self,
        frames: &CodecFrames,
        cancel: &CancelFlag,
    ) -> crate::gen_core::Result<crate::codec::DecodedTrack> {
        self.inner.decode(frames, cancel)
    }
}

impl Vocoder for Logged<dyn Vocoder> {
    fn decode(
        &self,
        track: Track,
        embedding: &Tensor,
        cancel: &CancelFlag,
    ) -> crate::gen_core::Result<Vec<f32>> {
        self.inner.decode(track, embedding, cancel)
    }
}

/// The stub stage set with every weight-bearing stage wrapped to log its load and drop, plus a
/// stage-1 step counter and an optional cancel trip at a given step.
fn instrumented(
    log: &Log,
    steps: &Arc<Mutex<usize>>,
    trip: Option<(usize, CancelFlag)>,
) -> StageSet {
    let base = StageSet::stubs();
    let mut set = base.clone();
    let (l, b) = (log.clone(), base.icl_encoder.clone());
    set.icl_encoder = Arc::new(move |a: &Assets| {
        push(&l, "icl:load");
        let inner = b(a)?;
        Ok(Box::new(Logged {
            name: "icl",
            log: l.clone(),
            inner,
        }) as Box<dyn IclEncoder>)
    });
    let (l, b, s) = (log.clone(), base.stage1.clone(), steps.clone());
    set.stage1 = Arc::new(move |a: &Assets, t| {
        push(&l, "stage1:load");
        let inner = b(a, t)?;
        Ok(Box::new(Stage1Probe {
            logged: Logged {
                name: "stage1",
                log: l.clone(),
                inner,
            },
            steps: s.clone(),
            trip: trip.clone(),
        }) as Box<dyn Stage1Model>)
    });
    let (l, b) = (log.clone(), base.stage2.clone());
    set.stage2 = Arc::new(move |a: &Assets, t| {
        push(&l, "stage2:load");
        let inner = b(a, t)?;
        Ok(Box::new(Logged {
            name: "stage2",
            log: l.clone(),
            inner,
        }) as Box<dyn Stage2Model>)
    });
    let (l, b) = (log.clone(), base.codec.clone());
    set.codec = Arc::new(move |a: &Assets| {
        push(&l, "codec:load");
        let inner = b(a)?;
        Ok(Box::new(Logged {
            name: "codec",
            log: l.clone(),
            inner,
        }) as Box<dyn CodecDecoder>)
    });
    let (l, b) = (log.clone(), base.vocoder.clone());
    set.vocoder = Arc::new(move |a: &Assets| {
        push(&l, "vocoder:load");
        let inner = b(a)?;
        Ok(Box::new(Logged {
            name: "vocoder",
            log: l.clone(),
            inner,
        }) as Box<dyn Vocoder>)
    });
    set
}

fn spec() -> LoadSpec {
    // Paths only — the stub stages never open them.
    LoadSpec::new(WeightsSource::Dir("/staged/yue-s1".into()))
        .with_component(
            STAGE2_COMPONENT_ID,
            WeightsSource::Dir("/staged/yue-s2".into()),
        )
        .with_component(
            XCODEC_COMPONENT_ID,
            WeightsSource::Dir("/staged/xcodec".into()),
        )
}

const LYRICS: &str = "[verse]\nwalking down the empty street\n[chorus]\nsing it loud\n[outro]\nbye";

fn song(segments: Option<u32>) -> GenerationRequest {
    GenerationRequest {
        prompt: "uplifting pop female vocal".into(),
        seed: Some(11),
        audio: Some(AudioParams {
            lyrics: Some(LYRICS.into()),
            segments,
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn engine(
    variant: Variant,
    log: &Log,
    steps: &Arc<Mutex<usize>>,
    trip: Option<(usize, CancelFlag)>,
) -> crate::model::YueGenerator {
    load_with_stages(variant, &spec(), instrumented(log, steps, trip)).unwrap()
}

fn cot() -> Variant {
    Variant::new(Language::En, Mode::Cot)
}

fn audio(out: GenerationOutput) -> AudioTrack {
    match out {
        GenerationOutput::Audio(t) => t,
        _ => panic!("expected audio"),
    }
}

fn position(log: &[String], entry: &str) -> usize {
    log.iter()
        .position(|e| e == entry)
        .unwrap_or_else(|| panic!("{entry} missing from {log:?}"))
}

#[test]
fn one_engine_call_returns_a_mix_and_both_stems_with_stage1_released_before_the_codec_loads() {
    let (log, steps) = (Log::default(), Arc::new(Mutex::new(0)));
    let g = engine(cot(), &log, &steps, None);
    let mut events = Vec::new();
    let track = audio(g.generate(&song(None), &mut |p| events.push(p)).unwrap());

    // Mix + the two named stems, all at the Vocos rate, mono, equal length.
    assert_eq!(track.sample_rate, 44_100);
    assert_eq!(track.channels, 1);
    let names: Vec<&str> = track.stems.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["vocals", "instrumental"]);
    // Default 2 segments × STUB_FRAMES_PER_SEGMENT frames × 882 samples per frame.
    let want = 2 * STUB_FRAMES_PER_SEGMENT * SAMPLES_PER_FRAME;
    assert_eq!(track.samples.len(), want);
    assert!(track.stems.iter().all(|s| s.samples.len() == want));
    assert!(track.samples.iter().any(|&x| x != 0.0));
    assert_ne!(track.stems[0].samples, track.stems[1].samples);

    // Staged residency: every stage is released before the next loads — in particular stage 1's
    // weights are gone before the codec loads.
    let log = log.lock().unwrap().clone();
    assert_eq!(
        log,
        [
            "stage1:load",
            "stage1:drop",
            "stage2:load",
            "stage2:drop",
            "codec:load",
            "codec:drop",
            "vocoder:load",
            "vocoder:drop"
        ]
    );
    assert!(position(&log, "stage1:drop") < position(&log, "codec:load"));

    // Segment-level progress: one Step per segment + one per stage-2 track, Decoding once.
    let steps: Vec<(u32, u32)> = events
        .iter()
        .filter_map(|p| match p {
            Progress::Step { current, total } => Some((*current, *total)),
            _ => None,
        })
        .collect();
    assert_eq!(steps, [(1, 4), (2, 4), (3, 4), (4, 4)]);
    assert_eq!(
        events
            .iter()
            .filter(|p| matches!(p, Progress::Decoding))
            .count(),
        1
    );
}

#[test]
fn engine_events_report_every_segment_with_its_label() {
    let (log, steps) = (Log::default(), Arc::new(Mutex::new(0)));
    let g = engine(cot(), &log, &steps, None);
    let req = crate::model::map_request(cot(), &song(Some(3))).unwrap();
    let mut events = Vec::new();
    g.engine()
        .render(&req, &CancelFlag::new(), &mut |e| events.push(e))
        .unwrap();
    let started: Vec<(usize, usize, String)> = events
        .iter()
        .filter_map(|e| match e {
            crate::YueEvent::SegmentStarted {
                index,
                total,
                label,
            } => Some((*index, *total, label.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(
        started,
        [
            (0, 3, "verse".to_string()),
            (1, 3, "chorus".to_string()),
            (2, 3, "outro".to_string())
        ]
    );
    let finished = events
        .iter()
        .filter(|e| {
            matches!(e, crate::YueEvent::SegmentFinished { tokens, .. }
                if *tokens == 2 * STUB_FRAMES_PER_SEGMENT)
        })
        .count();
    assert_eq!(finished, 3);
}

#[test]
fn cancel_during_stage1_returns_within_one_decode_step() {
    let (log, steps) = (Log::default(), Arc::new(Mutex::new(0)));
    let cancel = CancelFlag::new();
    // Trip the flag from inside the 5th decode step (mid segment 0).
    let g = engine(cot(), &log, &steps, Some((5, cancel.clone())));
    let mut req = song(None);
    req.cancel = cancel;
    let err = g.generate(&req, &mut |_| {}).unwrap_err();
    assert!(matches!(err, Error::Canceled), "{err:?}");
    assert_eq!(
        *steps.lock().unwrap(),
        5,
        "no decode step may run after the one that observed the cancel"
    );
    // Cancelled mid-stage-1: stage 1 is released and nothing after it ever loads.
    let log = log.lock().unwrap().clone();
    assert_eq!(log, ["stage1:load", "stage1:drop"]);
}

#[test]
fn per_segment_budget_and_segment_cap_are_honored() {
    let (log, steps) = (Log::default(), Arc::new(Mutex::new(0)));
    let g = engine(cot(), &log, &steps, None);
    let mut req = song(Some(9));
    req.audio.as_mut().unwrap().max_new_tokens_per_segment = Some(4);
    let mut totals = Vec::new();
    let track = audio(
        g.generate(&req, &mut |p| {
            if let Progress::Step { total, .. } = p {
                totals.push(total)
            }
        })
        .unwrap(),
    );
    // Nine requested, three lyric sections: three segments (+ two stage-2 tracks).
    assert!(totals.iter().all(|&t| t == 5), "{totals:?}");
    // Four tokens per segment = two frames per segment.
    assert_eq!(track.samples.len(), 3 * 2 * SAMPLES_PER_FRAME);
    assert_eq!(*steps.lock().unwrap(), 3 * 4);
}

#[test]
fn icl_render_releases_the_reference_encoder_before_stage1_loads() {
    let variant = Variant::new(Language::Zh, Mode::Icl);
    let (log, steps) = (Log::default(), Arc::new(Mutex::new(0)));
    let g = engine(variant, &log, &steps, None);
    let stem = |name: &str, v: f32| AudioStem {
        name: name.into(),
        samples: vec![v; 16_000 * 2],
    };
    let mut req = song(None);
    req.conditioning = vec![Conditioning::ReferenceAudio {
        audio: AudioTrack {
            samples: vec![0.3; 16_000 * 2],
            sample_rate: 16_000,
            channels: 1,
            stems: vec![stem("vocals", 0.1), stem("instrumental", 0.2)],
        },
        strength: None,
    }];
    let track = audio(g.generate(&req, &mut |_| {}).unwrap());
    assert_eq!(track.stems.len(), 2);
    let log = log.lock().unwrap().clone();
    assert!(
        position(&log, "icl:drop") < position(&log, "stage1:load"),
        "{log:?}"
    );

    // The same reference is refused by a CoT variant (the shared floor, typed).
    let cot_g = engine(cot(), &log_default(), &steps, None);
    assert!(matches!(
        cot_g.generate(&req, &mut |_| {}),
        Err(Error::Unsupported(_))
    ));
}

fn log_default() -> Log {
    Log::default()
}

#[test]
fn the_generator_meets_the_shared_progress_cancel_and_seed_contracts() {
    let (log, steps) = (Log::default(), Arc::new(Mutex::new(0)));
    let g = engine(cot(), &log, &steps, None);
    let req = song(None);
    gen_core_testkit::check_progress_with(&g, &req, Some(4)).unwrap();
    gen_core_testkit::check_progress_contract_with(&g, &req).unwrap();
    // The cancellation check trips the request's flag, so it gets its own request.
    gen_core_testkit::check_cancellation_with(&g, &song(None)).unwrap();
    gen_core_testkit::check_precancellation_with(&g, &req).unwrap();
    assert!(!req.cancel.is_cancelled());

    let a = audio(g.generate(&req, &mut |_| {}).unwrap());
    let b = audio(g.generate(&req, &mut |_| {}).unwrap());
    assert_eq!(a, b, "same request + seed is byte-identical");
    let mut other = req.clone();
    other.seed = Some(12);
    let c = audio(g.generate(&other, &mut |_| {}).unwrap());
    assert_ne!(a.samples, c.samples, "the seed reaches stage 1");
}

fn refused<T>(r: crate::gen_core::Result<T>) -> bool {
    matches!(r, Err(Error::Unsupported(_)))
}

#[test]
fn production_wiring_refuses_instead_of_rendering_placeholder_audio() {
    // The registered entry point loads lazily, then the render refuses at its first stage.
    let g = crate::model::load_en_cot(&spec()).unwrap();
    assert!(refused(g.generate(&song(None), &mut |_| {})));

    // Every production stage refuses on its own, so no later stage can fall back to its stub.
    let s = StageSet::production();
    let assets = Assets {
        stage1: "/staged/yue-s1".into(),
        stage2: "/staged/yue-s2".into(),
        xcodec: "/staged/xcodec".into(),
    };
    assert!(refused((s.tokenizer)(&assets)), "tokenizer");
    assert!(refused((s.icl_encoder)(&assets)), "icl encoder");
    assert!(refused((s.stage1)(&assets, None)), "stage 1");
    // Stage 2 is implemented (sc-19381): with nothing staged it fails to load, never a stub.
    assert!((s.stage2)(&assets, None).is_err(), "stage 2");
    assert!(refused((s.codec)(&assets)), "codec");
    assert!(refused((s.vocoder)(&assets)), "vocoder");
    assert!(refused((s.splice)(&[0.0], &[0.0])), "splice");
}
