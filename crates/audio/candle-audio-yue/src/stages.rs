//! The engine's stage wiring: one loader per seam. [`StageSet::production`] is what the registered
//! providers load through; [`StageSet::stubs`] is the weights-free set the end-to-end seam test
//! drives. Each field is independently replaceable, so a test can wrap one stage (to observe its
//! load/drop order or trip a cancel mid-decode) and keep the rest.

use std::sync::Arc;

use candle_audio::gen_core;

use crate::codec::CodecDecoder;
use crate::config::{Assets, Tier};
use crate::icl::IclEncoder;
use crate::splice::SpliceFn;
use crate::stage1::Stage1Model;
use crate::stage2::Stage2Model;
use crate::tokenizer::PromptTokenizer;
use crate::vocoder::Vocoder;

/// A tokenizer loader.
pub type TokenizerLoader =
    Arc<dyn Fn(&Assets) -> gen_core::Result<Box<dyn PromptTokenizer>> + Send + Sync>;
/// An ICL encoder loader.
pub type IclEncoderLoader =
    Arc<dyn Fn(&Assets) -> gen_core::Result<Box<dyn IclEncoder>> + Send + Sync>;
/// A stage-1 loader (`None` tier = detect from the staged snapshot).
pub type Stage1Loader =
    Arc<dyn Fn(&Assets, Option<Tier>) -> gen_core::Result<Box<dyn Stage1Model>> + Send + Sync>;
/// A stage-2 loader (`None` tier = detect from the staged snapshot).
pub type Stage2Loader =
    Arc<dyn Fn(&Assets, Option<Tier>) -> gen_core::Result<Box<dyn Stage2Model>> + Send + Sync>;
/// A codec-decoder loader.
pub type CodecLoader =
    Arc<dyn Fn(&Assets) -> gen_core::Result<Box<dyn CodecDecoder>> + Send + Sync>;
/// A vocoder loader.
pub type VocoderLoader = Arc<dyn Fn(&Assets) -> gen_core::Result<Box<dyn Vocoder>> + Send + Sync>;

/// The pipeline stages. The engine loads each one when its turn comes and releases it before the
/// next loads, so at most one stage's weights are resident at a time.
#[derive(Clone)]
pub struct StageSet {
    /// Prompt builder (sc-19376).
    pub tokenizer: TokenizerLoader,
    /// ICL reference encoder (sc-19379) — loaded for ICL renders only.
    pub icl_encoder: IclEncoderLoader,
    /// Stage-1 7B LM (sc-19380).
    pub stage1: Stage1Loader,
    /// Stage-2 1B LM (sc-19381).
    pub stage2: Stage2Loader,
    /// xcodec decoder (sc-19377).
    pub codec: CodecLoader,
    /// Vocos upsamplers (sc-19378).
    pub vocoder: VocoderLoader,
    /// Low-band splice (sc-19378).
    pub splice: SpliceFn,
}

impl StageSet {
    /// The production wiring — each module's `load` (and [`crate::splice::splice`]).
    pub fn production() -> Self {
        Self {
            tokenizer: Arc::new(crate::tokenizer::load),
            icl_encoder: Arc::new(crate::icl::load),
            stage1: Arc::new(crate::stage1::load),
            stage2: Arc::new(crate::stage2::load),
            codec: Arc::new(crate::codec::load),
            vocoder: Arc::new(crate::vocoder::load),
            splice: crate::splice::splice,
        }
    }

    /// The weights-free stub wiring — each module's `load_stub` (and
    /// [`crate::splice::splice_stub`]).
    pub fn stubs() -> Self {
        Self {
            tokenizer: Arc::new(crate::tokenizer::load_stub),
            icl_encoder: Arc::new(crate::icl::load_stub),
            stage1: Arc::new(crate::stage1::load_stub),
            stage2: Arc::new(crate::stage2::load_stub),
            codec: Arc::new(crate::codec::load_stub),
            vocoder: Arc::new(crate::vocoder::load_stub),
            splice: crate::splice::splice_stub,
        }
    }
}

impl std::fmt::Debug for StageSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StageSet").finish_non_exhaustive()
    }
}
