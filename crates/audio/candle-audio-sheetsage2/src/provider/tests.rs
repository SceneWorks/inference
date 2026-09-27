use super::*;
use crate::model::tests::tiny_reference;

pub(crate) fn tiny_files() -> ClosureFiles {
    let reference = tiny_reference();
    let config = reference["sheetsage2_config"].clone();
    let tensors = |bytes: &[u8]| {
        candle_audio::candle_core::safetensors::load_buffer(bytes, &Device::Cpu).unwrap()
    };
    ClosureFiles {
        mert_config: config["backbone_config"].clone(),
        sheetsage2_config: config,
        head: Weights::from_map(
            "tiny head",
            tensors(include_bytes!(
                "../../testdata/tiny/sheetsage2_head.safetensors"
            )),
        ),
        parent: Weights::from_map(
            "tiny parent",
            tensors(include_bytes!(
                "../../testdata/tiny/mert_parent.safetensors"
            )),
        ),
    }
}

/// Window settings that fit the tiny model's 1 s window (the defaults assume 300 s).
pub(crate) fn tiny_settings() -> TranscriptionSettings {
    TranscriptionSettings {
        overlap_seconds: 0.5,
        lookahead_seconds: 0.25,
        ..TranscriptionSettings::default()
    }
}

pub(crate) fn tiny_identity() -> ClosureIdentity {
    ClosureIdentity {
        sheetsage2: ["tiny".into(), "fixture".into(), "-".into(), "-".into()],
        mert: ["tiny".into(), "fixture".into(), "-".into(), "-".into()],
        ported_code_revision: crate::PORTED_CODE_REVISION.into(),
        tokenizer_fingerprint: String::new(),
        device: "cpu".into(),
    }
}

/// Unload is observable: the model is dropped (no reference survives), the process-wide live-model
/// count returns to where it was, and the receipt reports what was released.
///
/// Mutation that must fail: keep a clone of the `Arc` inside the transcriber (e.g. a cache), or
/// skip the `LIVE_MODELS` decrement in `Drop`.
#[test]
fn unload_releases_the_model_observably() {
    let _serial = crate::test_lock();
    let before = live_models();
    let transcriber = Transcriber::from_files(tiny_files(), tiny_identity(), &Device::Cpu).unwrap();
    assert_eq!(live_models(), before + 1);
    assert!(transcriber.model().parameter_bytes() > 0);
    let receipt = transcriber.unload();
    assert!(receipt.released);
    assert!(receipt.parameter_bytes > 0);
    assert_eq!(receipt.live_models_after, before);
    assert_eq!(live_models(), before);
}

/// End to end on the tiny real-architecture model, in CI: the provider window-plans the source,
/// encodes, decodes greedily under the grammar with the same window, stop time and token limit as
/// upstream's reference run, and builds a complete [`Transcription`] (stitching, every export, both
/// scores, octave evidence, review) from upstream's exact tokens.
///
/// Mutation that must fail: a token limit one below the decoder context (`max_len - 1`).
#[test]
fn transcribe_reproduces_the_reference_and_builds_a_transcription() {
    let _serial = crate::test_lock();
    let transcriber = Transcriber::from_files(tiny_files(), tiny_identity(), &Device::Cpu).unwrap();
    let reference = crate::model::tests::tiny_reference();
    let tensors = candle_audio::candle_core::safetensors::load_buffer(
        include_bytes!("../../testdata/tiny/reference.safetensors"),
        &Device::Cpu,
    )
    .unwrap();
    let waveform = tensors["input.waveform"].to_vec1::<f32>().unwrap();
    let mut progress = Vec::new();
    let t = transcriber
        .transcribe(
            &SourceAudio::mono(waveform.clone()),
            &tiny_settings(),
            |p| {
                progress.push(p);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(
        progress[0],
        Progress::Encoding {
            window: 1,
            windows: 1
        }
    );
    assert_eq!(progress.last(), Some(&Progress::Notation));
    let expected: Vec<u32> = reference["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(t.stitched.records.len(), 1);
    assert_eq!(t.stitched.records[0].tokens, expected);
    assert_eq!(t.source.samples, waveform.len());
    assert_eq!(t.source.sample_rate, 1600);
    assert_eq!(t.audio(), waveform.as_slice());
    assert!(t.stitched.decoded.events.len() > 3);
    assert!(t.full.text("events.json").is_some());
    assert!(t.full.midi("transcription.mid").is_some());
    assert_eq!(
        t.closure.tokenizer_fingerprint,
        reference["tokenizer"]["fingerprint"]
    );
    transcriber.unload();
}

/// A generated sequence that does not decode is an error, never an empty transcription: the
/// upstream rule "an event must start with a sub-beat shift" is enforced when a window is accepted.
#[test]
fn undecodable_tokens_surface_as_an_error() {
    use crate::pipeline::Stitcher;
    let tokenizer = crate::tokenizer::Tokenizer::new(1.0, 100, None).unwrap();
    let mut stitcher = Stitcher::new(
        &tokenizer,
        &crate::tokenizer::FULL_TASK_PROMPTS,
        0.8,
        0.5,
        0.25,
        48,
    )
    .unwrap();
    let (mut tokens, prefix_len) = stitcher.prefix(0).unwrap();
    tokens.push(tokenizer.eighth_position.start);
    tokens.push(crate::tokenizer::EOS);
    let err = stitcher.accept(0, tokens, prefix_len).unwrap_err();
    assert!(err.to_string().contains("has no subbeat shift"), "{err}");
}

/// Cancellation through the progress callback stops the transcription with the caller's error.
#[test]
fn progress_can_cancel() {
    let _serial = crate::test_lock();
    let transcriber = Transcriber::from_files(tiny_files(), tiny_identity(), &Device::Cpu).unwrap();
    let err = transcriber
        .transcribe(
            &SourceAudio::mono(vec![0.1; 1200]),
            &tiny_settings(),
            |_| Err(Error::Request("cancelled".into())),
        )
        .unwrap_err();
    assert_eq!(err.to_string(), "request: cancelled");
    transcriber.unload();
}

#[test]
fn source_audio_is_validated_and_identified() {
    let short = SourceAudio::mono(vec![0.0; 1000]);
    assert!(short.prepare(None, 24_000, 1025).is_err());
    let mut nan = vec![0.0; 2000];
    nan[7] = f32::NAN;
    assert!(SourceAudio::mono(nan).prepare(None, 24_000, 1025).is_err());
    let (samples, identity) = SourceAudio::mono(vec![0.25; 48_000])
        .with_name("clip")
        .prepare(Some(1.5), 24_000, 1025)
        .unwrap();
    assert_eq!(samples.len(), 36_000);
    assert_eq!(identity.samples, 36_000);
    assert!(identity.conversion.contains("cropped to 1.5 s"));
    let mut hasher = Sha256::new();
    for s in &samples {
        hasher.update(s.to_le_bytes());
    }
    let expected: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(identity.sha256, expected);
    // Resampled sources must match the model rate.
    let stereo = SourceAudio::interleaved(&vec![0.1; 44_100 * 2], 44_100, 2).unwrap();
    assert_eq!(stereo.samples().len(), 24_000);
    assert!(stereo.prepare(None, 1_600, 65).is_err());
}

/// The SheetSage2 config must pin the inventory's MERT parent.
#[test]
fn parent_pin_is_cross_checked() {
    let reference = tiny_reference();
    let mut config = reference["sheetsage2_config"].clone();
    check_parent_pin(&config).unwrap();
    config["base_model_sha256"] = serde_json::json!("0".repeat(64));
    assert!(check_parent_pin(&config).is_err());
}

/// The published component mapping names exactly the cover closure, every key resolves to a
/// licence row, and the derived provider terms are noncommercial.
///
/// Mutation that must fail: drop MERT from `PROVIDER_COMPONENTS`, or publish an Apache row.
#[test]
fn published_components_are_the_cover_closure_with_noncommercial_terms() {
    use candle_audio::gen_core::{provider_terms, LicenseTerm, LICENSE_FAMILIES};
    let keys: Vec<&str> = Closure::Cover
        .components()
        .iter()
        .map(|id| id.component().key)
        .collect();
    assert_eq!(PROVIDER_COMPONENTS[0].components, keys.as_slice());
    for key in &keys {
        assert!(
            COMPONENT_LICENSES.iter().any(|row| row.component == *key),
            "{key}"
        );
    }
    let terms = provider_terms(
        &PROVIDER_COMPONENTS[0],
        COMPONENT_LICENSES,
        LICENSE_FAMILIES,
    );
    assert!(
        terms.contains(&LicenseTerm::NonCommercialWeights),
        "{terms:?}"
    );
}
