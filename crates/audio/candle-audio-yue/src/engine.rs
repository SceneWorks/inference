//! The YuE engine: one call runs tokenizer → (ICL encoder) → stage 1 → stage 2 → xcodec → Vocos →
//! low-band splice through the [`StageSet`] seams and returns the mix plus the vocal and
//! instrumental stems.
//!
//! ## Staged residency
//!
//! Each stage is loaded when its turn comes and **released before the next stage loads**, so the
//! resident floor is the largest single stage, not the sum: the 7B stage-1 LM is dropped before the
//! stage-2 LM loads, and both are gone before the codec and the vocoders load. The ICL encoder
//! (xcodec encoder + HuBERT) is released before stage 1 loads.
//!
//! ## Progress and cancellation
//!
//! The engine owns the stage-1 decode loop and checks the render's [`CancelFlag`] before every
//! [`Stage1Model::step`](crate::stage1::Stage1Model::step), so a cancel during stage 1 returns
//! [`gen_core::Error::Canceled`] within one decode step whatever the stage-1 implementation. The
//! flag is also checked before every stage load and handed to every later stage. Progress is
//! reported per lyric segment ([`YueEvent`]).

use candle_audio::gen_core::{self, CancelFlag};

use crate::config::{Assets, Tier, Variant, YueRequest};
use crate::stage1::{SegmentStart, Stage1Step};
use crate::stages::StageSet;
use crate::tokenizer::PromptInput;
use crate::tokens::{split_raw_output, TrackCodes, CODEBOOK_SIZE, EOA};
use crate::vocoder::Track;

/// A pipeline stage, as named in [`YueEvent`]s.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Prompt builder.
    Tokenizer,
    /// ICL reference encoder.
    IclEncoder,
    /// Stage-1 LM.
    Stage1,
    /// Stage-2 LM.
    Stage2,
    /// xcodec decoder.
    Codec,
    /// Vocos upsamplers.
    Vocoder,
}

/// What the engine reports while it renders.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum YueEvent {
    /// A stage's weights were loaded.
    StageLoaded(Stage),
    /// A stage's weights were released.
    StageReleased(Stage),
    /// Stage 1 started lyric segment `index` of `total`.
    SegmentStarted {
        /// 0-based segment index.
        index: usize,
        /// Segments this render runs.
        total: usize,
        /// The section label (`verse`, `chorus`, …).
        label: String,
    },
    /// Stage 1 finished lyric segment `index` of `total`.
    SegmentFinished {
        /// 0-based segment index.
        index: usize,
        /// Segments this render runs.
        total: usize,
        /// Tokens the segment produced.
        tokens: usize,
    },
    /// Stage 2 finished upsampling one track.
    TrackUpsampled(Track),
    /// The codec + vocoder decode began.
    Decoding,
}

/// A finished render: mono audio at [`sample_rate`](Self::sample_rate), all three the same length.
#[derive(Clone, Debug, PartialEq)]
pub struct YueOutput {
    /// The final mix (after the low-band splice).
    pub mix: Vec<f32>,
    /// The vocal stem.
    pub vocals: Vec<f32>,
    /// The instrumental stem.
    pub instrumental: Vec<f32>,
    /// Output sample rate ([`crate::vocoder::SAMPLE_RATE`]).
    pub sample_rate: u32,
}

/// One loaded (lazy) YuE engine: a variant, its staged assets, the asserted tier, and its stages.
#[derive(Clone, Debug)]
pub struct YueEngine {
    variant: Variant,
    assets: Assets,
    tier: Option<Tier>,
    stages: StageSet,
}

/// Emit `StageReleased` when a stage handle goes out of scope, after the stage itself drops.
fn release<T: ?Sized>(stage: Stage, handle: Box<T>, on_event: &mut dyn FnMut(YueEvent)) {
    drop(handle);
    on_event(YueEvent::StageReleased(stage));
}

fn check_cancel(cancel: &CancelFlag) -> gen_core::Result<()> {
    if cancel.is_cancelled() {
        Err(gen_core::Error::Canceled)
    } else {
        Ok(())
    }
}

impl YueEngine {
    /// Construct an engine. Nothing is loaded until [`render`](Self::render).
    pub fn new(variant: Variant, assets: Assets, tier: Option<Tier>, stages: StageSet) -> Self {
        Self {
            variant,
            assets,
            tier,
            stages,
        }
    }

    /// The stage-1 variant.
    pub fn variant(&self) -> Variant {
        self.variant
    }

    /// The asserted LM tier (`None` = detect from the staged snapshots).
    pub fn tier(&self) -> Option<Tier> {
        self.tier
    }

    /// The staged assets.
    pub fn assets(&self) -> &Assets {
        &self.assets
    }

    /// Render one song.
    pub fn render(
        &self,
        req: &YueRequest,
        cancel: &CancelFlag,
        on_event: &mut dyn FnMut(YueEvent),
    ) -> gen_core::Result<YueOutput> {
        let id = self.variant.id();
        req.validate(self.variant)?;
        check_cancel(cancel)?;
        let s = &self.stages;

        // ICL reference → codec token block (encoder released before anything else loads).
        let icl_codes = match &req.icl {
            Some(reference) => {
                let encoder = (s.icl_encoder)(&self.assets)?;
                on_event(YueEvent::StageLoaded(Stage::IclEncoder));
                let codes = encoder.encode(reference, cancel);
                release(Stage::IclEncoder, encoder, on_event);
                Some(codes?)
            }
            None => None,
        };

        // Prompt.
        check_cancel(cancel)?;
        let tokenizer = (s.tokenizer)(&self.assets)?;
        on_event(YueEvent::StageLoaded(Stage::Tokenizer));
        let prompt = tokenizer.build(&PromptInput {
            genres: &req.genres,
            lyrics: &req.lyrics,
            icl: icl_codes.as_ref(),
        });
        release(Stage::Tokenizer, tokenizer, on_event);
        let prompt = prompt?;
        if prompt.segments.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: the lyrics produced no segments"
            )));
        }
        let total = prompt.segments.len().min(req.segments as usize);

        // Stage 1: segment-by-segment codebook-0 decode, cancel checked before every step.
        check_cancel(cancel)?;
        let mut stage1 = (s.stage1)(&self.assets, self.tier)?;
        on_event(YueEvent::StageLoaded(Stage::Stage1));
        let segments = self.run_stage1(stage1.as_mut(), req, &prompt, total, cancel, on_event);
        release(Stage::Stage1, stage1, on_event);
        let tracks = stage1_tracks(&prompt, &segments?)
            .map_err(|e| gen_core::Error::Msg(format!("{id}: {e}")))?;
        if tracks.vocals.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: stage 1 produced no audio frames"
            )));
        }
        // The reference's stage 2 asserts every code is inside codebook 0 (`offset_tok_ids`); a
        // token the allow-list admits beyond it cannot be upsampled.
        if let Some(code) = tracks
            .vocals
            .iter()
            .chain(&tracks.instrumental)
            .find(|&&c| c >= CODEBOOK_SIZE)
        {
            return Err(gen_core::Error::Msg(format!(
                "{id}: stage 1 emitted a token outside codebook 0 (code {code}), which stage 2 \
                 cannot upsample"
            )));
        }

        // Stage 2: upsample each track to the full codebook grid.
        check_cancel(cancel)?;
        let mut stage2 = (s.stage2)(&self.assets, self.tier)?;
        on_event(YueEvent::StageLoaded(Stage::Stage2));
        let grids = (|| {
            let mut out = Vec::with_capacity(2);
            for (track, cb0) in [
                (Track::Vocals, &tracks.vocals),
                (Track::Instrumental, &tracks.instrumental),
            ] {
                check_cancel(cancel)?;
                let grid = stage2.upsample(cb0, cancel)?;
                grid.check_against(cb0)
                    .map_err(|e| gen_core::Error::Msg(format!("{id}: stage 2 {track:?}: {e}")))?;
                on_event(YueEvent::TrackUpsampled(track));
                out.push(grid);
            }
            Ok::<_, gen_core::Error>(out)
        })();
        release(Stage::Stage2, stage2, on_event);
        let grids = grids?;

        // Codec decode: 16 kHz waveform + Vocos embedding per track.
        check_cancel(cancel)?;
        on_event(YueEvent::Decoding);
        let codec = (s.codec)(&self.assets)?;
        on_event(YueEvent::StageLoaded(Stage::Codec));
        let decoded: gen_core::Result<Vec<_>> =
            grids.iter().map(|g| codec.decode(g, cancel)).collect();
        release(Stage::Codec, codec, on_event);
        let decoded = decoded?;

        // Vocos: 44.1 kHz stems.
        check_cancel(cancel)?;
        let vocoder = (s.vocoder)(&self.assets)?;
        on_event(YueEvent::StageLoaded(Stage::Vocoder));
        let stems: gen_core::Result<Vec<Vec<f32>>> = [Track::Vocals, Track::Instrumental]
            .iter()
            .zip(&decoded)
            .map(|(&track, d)| vocoder.decode(track, &d.embedding, cancel))
            .collect();
        release(Stage::Vocoder, vocoder, on_event);
        let mut stems = stems?;
        check_cancel(cancel)?;

        // Mixes and the low-band splice.
        let mix_codec_rate = sum_tracks(&decoded[0].wave, &decoded[1].wave);
        let mut instrumental = stems.pop().unwrap_or_default();
        let mut vocals = stems.pop().unwrap_or_default();
        let mix_vocoder_rate = sum_tracks(&vocals, &instrumental);
        let mut mix = (s.splice)(&mix_codec_rate, &mix_vocoder_rate)?;
        let len = mix.len().min(vocals.len()).min(instrumental.len());
        if len == 0 {
            return Err(gen_core::Error::Msg(format!(
                "{id}: the vocoder produced no audio"
            )));
        }
        mix.truncate(len);
        vocals.truncate(len);
        instrumental.truncate(len);
        Ok(YueOutput {
            mix,
            vocals,
            instrumental,
            sample_rate: crate::vocoder::SAMPLE_RATE,
        })
    }

    /// The stage-1 decode loop: per segment, prefill its block, then step until `<EOA>` or the
    /// per-segment budget, checking `cancel` before every step.
    fn run_stage1(
        &self,
        stage1: &mut dyn crate::stage1::Stage1Model,
        req: &YueRequest,
        prompt: &crate::tokenizer::Stage1Prompt,
        total: usize,
        cancel: &CancelFlag,
        on_event: &mut dyn FnMut(YueEvent),
    ) -> gen_core::Result<Vec<Vec<u32>>> {
        stage1.begin_render(req.seed)?;
        let mut segments = Vec::with_capacity(total);
        for (index, block) in prompt.segments.iter().take(total).enumerate() {
            check_cancel(cancel)?;
            on_event(YueEvent::SegmentStarted {
                index,
                total,
                label: block.label.clone(),
            });
            stage1.begin_segment(&SegmentStart {
                index,
                prompt: &block.ids,
                guidance_scale: req.decode.guidance.scale_for(index),
                decode: &req.decode,
            })?;
            let mut tokens = Vec::new();
            while tokens.len() < req.decode.max_new_tokens as usize {
                check_cancel(cancel)?;
                match stage1.step()? {
                    Stage1Step::Token(t) => tokens.push(t),
                    Stage1Step::EndOfAudio => break,
                }
            }
            stage1.end_segment()?;
            on_event(YueEvent::SegmentFinished {
                index,
                total,
                tokens: tokens.len(),
            });
            segments.push(tokens);
        }
        Ok(segments)
    }
}

/// Rebuild the render's stage-1 sequence — each run segment's prompt block, its generated tokens
/// and the `<EOA>` that closed it (sampled or forced) — and split it with the reference `save`
/// semantics. Complete `<SOA>…<EOA>` pairs inside the prompt blocks (the ICL reference block) are
/// prompt audio, skipped as the reference skips its first pair for an audio-prompted render.
pub(crate) fn stage1_tracks(
    prompt: &crate::tokenizer::Stage1Prompt,
    generated: &[Vec<u32>],
) -> Result<TrackCodes, String> {
    let mut raw = Vec::new();
    let mut prompt_pairs = 0;
    for (block, tokens) in prompt.segments.iter().zip(generated) {
        prompt_pairs += block.ids.iter().filter(|&&t| t == EOA).count();
        raw.extend_from_slice(&block.ids);
        raw.extend_from_slice(tokens);
        raw.push(EOA);
    }
    split_raw_output(&raw, prompt_pairs)
}

/// Sample-wise sum of two mono tracks, over the shorter length.
fn sum_tracks(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}
