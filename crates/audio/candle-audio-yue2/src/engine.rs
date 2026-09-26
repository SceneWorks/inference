//! The native YuE2 engine (sc-22994): one loaded model closure that runs plan → semantic →
//! acoustic synthesis → decode, with every stage also invocable on its own.
//!
//! Ported from the pinned upstream `src/yue2/pipeline.py` (`YuE2Pipeline.plan`,
//! `generate_semantic`, `synthesize`, `decode`, `effective_config`, `__call__`), Apache-2.0, commit
//! [`YUE2_SOURCE_COMMIT`](crate::inventory::YUE2_SOURCE_COMMIT). Nothing here starts a Python
//! process; the model, tokenizer and decoders are the native ports of the sibling modules.
//!
//! # Loading
//!
//! [`Yue2Engine::load`] authorizes the closure for
//! [`IntendedUse::NoncommercialExperimentation`] (the only use the CC BY-NC 4.0 weights permit
//! — [`crate::license`]), then resolves and verifies the YuE2-3B snapshot and `qwen.tiktoken`
//! **immediately before loading them** (the crate's load-boundary rule). The decoders are resolved,
//! verified and loaded the first time a decode asks for them. Everything is read from local
//! snapshots only: a missing snapshot is an explicit
//! [`AssetError::CacheMiss`](crate::snapshot::AssetError::CacheMiss), never a download.
//!
//! # Stages
//!
//! | stage | entry point | upstream |
//! |---|---|---|
//! | plan | [`Yue2Engine::plan`] | `pipe.plan` |
//! | semantic | [`Yue2Engine::generate_semantic`] | `pipe.generate_semantic` |
//! | acoustic | [`Yue2Engine::synthesize`] | `pipe.synthesize` |
//! | decode | [`Yue2Engine::decode`] | `pipe.decode` |
//! | all four | [`Yue2Engine::generate`] | `pipe(...)` |
//!
//! Each stage takes [`EngineHooks`]: a cancellation flag polled at every bounded boundary of the
//! stage (between prefill chunks and before every token, before each velocity evaluation, before
//! every decode tile) and an [`EngineObserver`] that sees stage starts/finishes, output tokens and
//! step progress. A cancelled stage returns [`gen_core::Error::Canceled`] and no partial output.
//!
//! The artifact-backed runs (transactional output, resume, cached decoding) are in
//! [`crate::run`]; the self-contained closure export in [`crate::closure`].
//!
//! # Randomness (epic E9)
//!
//! The ABC and semantic stages each restart the request seed's SplitMix64 stream
//! ([`stage_rng`]) and the acoustic stage draws the song's noise once from the same seed
//! ([`SongNoise::seeded`]). A seed reproduces a native run on the same backend and dtype; it is not
//! PyTorch's stream and not a cross-platform bit-exact guarantee.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use candle_audio::candle_core::{DType, Device};
use candle_audio::gen_core;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::decode::{decode_latents, DecodeMode, DecodeOptions, DecodedAudio};
use crate::generate::{
    generate_semantic as semantic_decode, plan_score, stage_rng, Cot, DecodeObserver, Decoded,
    Hooks, ScorePlan, SemanticInput,
};
use crate::inventory::{Closure, Component, ComponentId, VaeVariant};
use crate::latent::AcousticLatents;
use crate::license::{self, Authorization, IntendedUse};
use crate::nar::{
    synthesize as nar_synthesize, NarOptions, QueryTile, SongNoise, Synthesis, SynthesisHooks,
    SynthesisObserver, SynthesisRequest, Yue2Nar,
};
use crate::plan::{PlanStep, SymbolicPlan};
use crate::protocol::{
    check_generation_budget, CotMode, GenerationConfig, Sampling, SongRequest, CONTEXT,
    PROTOCOL_VERSION,
};
use crate::sampling::{self, Phase};
use crate::snapshot::{self, SnapshotDirs};
use crate::tokenizer::Yue2TextTokenizer;
use crate::vae::{variant_name, VaeParts, Yue2Vae};

/// The provider's engine identity, written into every artifact record (distinct from YuE1's
/// `yue_*` engines, epic E1).
pub const ENGINE_ID: &str = "yue2";

/// Schema of the identities this module derives (plan / semantic / synthesis / decode stages and
/// the run identity).
pub const IDENTITY_SCHEMA: &str = "yue2-engine-v1";

/// A pipeline stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Stage {
    /// Symbolic planning (the ABC score), or taking an external / restored score.
    Plan,
    /// Semantic codec tokens.
    Semantic,
    /// Acoustic flow matching (latents).
    Synthesis,
    /// VAE decoding (48 kHz stereo).
    Decode,
}

impl Stage {
    /// Every stage, in pipeline order.
    pub const ALL: [Stage; 4] = [
        Stage::Plan,
        Stage::Semantic,
        Stage::Synthesis,
        Stage::Decode,
    ];

    /// The stage's artifact-record name.
    pub fn name(self) -> &'static str {
        match self {
            Stage::Plan => "plan",
            Stage::Semantic => "semantic",
            Stage::Synthesis => "synthesis",
            Stage::Decode => "decode",
        }
    }
}

/// What happened to a stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageEvent {
    /// The stage started computing.
    Started,
    /// The stage finished computing.
    Finished,
    /// The stage was not computed: a verified, identity-matching artifact was reused.
    Reused,
}

/// Observes an engine run. Every method has a no-op default.
pub trait EngineObserver {
    /// A stage started, finished or was reused.
    fn on_stage(&mut self, stage: Stage, event: StageEvent) {
        let _ = (stage, event);
    }

    /// An output token of the plan or semantic stage (the end id included), as sampled.
    fn on_token(&mut self, stage: Stage, token: u32) {
        let _ = (stage, token);
    }

    /// Acoustic synthesis progress: `completed` of `total` midpoint steps over all chunks.
    fn on_synthesis_progress(&mut self, completed: usize, total: usize) {
        let _ = (completed, total);
    }

    /// Decode progress: `completed` of `total` tiles.
    fn on_decode_progress(&mut self, completed: usize, total: usize) {
        let _ = (completed, total);
    }
}

/// The no-op observer.
impl EngineObserver for () {}

/// Cancellation and observation for an engine call.
pub struct EngineHooks<'a> {
    /// Polled at every bounded boundary of every stage; `true` cancels.
    pub cancelled: &'a dyn Fn() -> bool,
    /// Receives stage events, tokens and progress.
    pub observer: &'a mut dyn EngineObserver,
}

impl EngineHooks<'_> {
    /// `Err(Canceled)` when the cancellation flag is set.
    pub fn check_cancel(&self) -> gen_core::Result<()> {
        if (self.cancelled)() {
            Err(gen_core::Error::Canceled)
        } else {
            Ok(())
        }
    }
}

/// Memory controls of the engine. Neither changes a result (they are recorded in the effective
/// configuration, not in any stage identity).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngineOptions {
    /// Acoustic-stage query tiling and AR offload.
    pub nar: NarOptions,
    /// Decode tiling.
    pub decode: DecodeOptions,
}

impl Default for EngineOptions {
    /// Upstream's off-CUDA acoustic defaults and the production decode tiling.
    fn default() -> Self {
        Self {
            nar: NarOptions::default(),
            decode: DecodeOptions::production(),
        }
    }
}

/// Per-request settings that are not part of the [`SongRequest`] itself (upstream's
/// `abc_sampling` / `semantic_sampling` / `generation_config` and the pipeline's decoder).
#[derive(Clone, Debug, PartialEq)]
pub struct SongSettings {
    /// Both phases' sampling and the midpoint ODE step count.
    pub generation: GenerationConfig,
    /// The decoder the audio is rendered with.
    pub decoder: VaeVariant,
}

impl Default for SongSettings {
    /// The released generation configuration and the standard (listening) decoder.
    fn default() -> Self {
        Self {
            generation: GenerationConfig::default(),
            decoder: VaeVariant::Standard,
        }
    }
}

/// The semantic stage's result (upstream `SemanticResult`).
#[derive(Clone, Debug, PartialEq)]
pub struct SemanticResult {
    /// The plan the tokens were generated from.
    pub plan: SymbolicPlan,
    /// Codec indices `0 … CODEC_SIZE − 1` (vocabulary id − `CODEC_OFFSET`), the end id excluded.
    pub codes: Vec<u32>,
    /// The phase hit its token budget before `MUSIC_END`.
    pub truncated: bool,
    /// Upstream's timing record for the phase.
    pub timing: Map<String, Value>,
}

/// One whole song (upstream `SongResult`).
#[derive(Clone, Debug)]
pub struct SongResult {
    /// The semantic result, which carries the plan.
    pub semantic: SemanticResult,
    /// The acoustic latents.
    pub latents: AcousticLatents,
    /// The decoded 48 kHz stereo audio with decoder and latent identity.
    pub audio: DecodedAudio,
    /// The effective configuration ([`Yue2Engine::effective_config`]).
    pub config: Value,
    /// The run identity ([`Yue2Engine::run_identity`]).
    pub identity: String,
    /// Per-stage timings (upstream's `timing` keys plus `reused` stages).
    pub timing: Map<String, Value>,
}

impl SongResult {
    /// Upstream's `truncated` record: `{"abc": …, "semantic": …}`.
    pub fn truncated(&self) -> Value {
        json!({"abc": self.semantic.plan.truncated(), "semantic": self.semantic.truncated})
    }
}

/// Where decoders come from.
enum VaeSource {
    /// Resolve and verify from local snapshots immediately before loading.
    Snapshots(SnapshotDirs),
    /// The synthetic fixture decoders (unit tests).
    #[cfg(test)]
    Fixture,
}

/// A loaded YuE2 model closure: the MoT (AR + NAR paths), the tokenizer and the decoders.
pub struct Yue2Engine {
    tokenizer: Yue2TextTokenizer,
    nar: Mutex<Yue2Nar>,
    vae_source: VaeSource,
    vaes: Mutex<BTreeMap<&'static str, Arc<Yue2Vae>>>,
    mot_identity: Value,
    tokenizer_identity: Value,
    generation: GenerationConfig,
    options: EngineOptions,
    authorization: Authorization,
    load_timing: Map<String, Value>,
}

impl std::fmt::Debug for Yue2Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Yue2Engine")
            .field("mot", &self.mot_identity)
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

/// The identity record of a pinned component: key, repository, revision and every pinned file's
/// SHA-256 (the bytes verification checked).
pub fn component_identity(component: &Component) -> Value {
    let files: Map<String, Value> = component
        .files
        .iter()
        .map(|f| {
            (
                f.path.to_string(),
                json!({"sha256": f.sha256, "bytes": f.bytes}),
            )
        })
        .collect();
    json!({
        "component": component.key,
        "repo": component.repo.id,
        "revision": component.repo.revision,
        "files": files,
    })
}

/// Canonical JSON (object keys sorted at every level, no whitespace): the byte form every identity
/// in this module hashes, independent of `serde_json`'s map-order feature.
pub fn canonical_json(value: &Value) -> String {
    fn sorted(v: &Value) -> Value {
        match v {
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                let mut out = Map::new();
                for k in keys {
                    out.insert(k.clone(), sorted(&map[k]));
                }
                Value::Object(out)
            }
            Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
            other => other.clone(),
        }
    }
    // With `preserve_order` a `Map` keeps insertion order (sorted above); without it, it is a
    // `BTreeMap` (sorted by construction). Either way the bytes are the sorted form.
    sorted(value).to_string()
}

/// SHA-256 (lower-case hex) of [`canonical_json`]`(value)`.
pub fn identity_of(value: &Value) -> String {
    hex(&Sha256::digest(canonical_json(value).as_bytes()))
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The token sampler's controls for a validated protocol [`Sampling`].
pub fn sampler(s: &Sampling) -> sampling::Sampling {
    let count = |v: u64| usize::try_from(v).unwrap_or(usize::MAX);
    sampling::Sampling {
        temperature: s.temperature(),
        top_p: s.top_p(),
        top_k: count(s.top_k()),
        repetition_penalty: s.repetition_penalty(),
        penalty_window: count(s.penalty_window()),
        min_tokens: count(s.min_tokens()),
        max_tokens: count(s.max_tokens()),
    }
}

fn cot_of(mode: CotMode) -> Cot {
    match mode {
        CotMode::Off => Cot::Off,
        CotMode::Melody => Cot::Melody,
        CotMode::Full => Cot::Full,
    }
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float32",
        DType::BF16 => "bfloat16",
        DType::F16 => "float16",
        _ => "other",
    }
}

fn device_name(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "metal",
    }
}

/// Upstream's per-decode timing record from a native decode.
fn decode_timing(decoded: &Decoded) -> Map<String, Value> {
    let mut t = Map::new();
    t.insert("seconds".into(), json!(decoded.seconds));
    t.insert("prefill_seconds".into(), json!(decoded.prefill_seconds));
    t.insert("output_tokens".into(), json!(decoded.output_tokens));
    t.insert("content_tokens".into(), json!(decoded.tokens.len()));
    let tps = if decoded.seconds > 0.0 {
        decoded.output_tokens as f64 / decoded.seconds
    } else {
        0.0
    };
    t.insert("output_tps".into(), json!(tps));
    t.insert("prefix_tokens".into(), json!(decoded.prefix_tokens));
    t.insert("cfg_branches".into(), json!(decoded.cfg_branches));
    t.insert("execution".into(), json!("eager"));
    t.insert("attention".into(), json!("sdpa"));
    t
}

/// Forwards a decode's tokens to the engine observer as one stage's tokens.
struct TokenForward<'a> {
    stage: Stage,
    observer: &'a mut dyn EngineObserver,
}

impl DecodeObserver for TokenForward<'_> {
    fn on_token(&mut self, _phase: Phase, token: u32) {
        self.observer.on_token(self.stage, token);
    }
}

/// Forwards synthesis progress to the engine observer.
struct ProgressForward<'a> {
    observer: &'a mut dyn EngineObserver,
}

impl SynthesisObserver for ProgressForward<'_> {
    fn on_progress(&mut self, completed: usize, total: usize) {
        self.observer.on_synthesis_progress(completed, total);
    }
}

pub(crate) fn msg(what: impl std::fmt::Display) -> gen_core::Error {
    gen_core::Error::Msg(format!("YuE2: {what}"))
}

impl Yue2Engine {
    /// Load the engine from local snapshots (see the [module docs](self)): authorize, then resolve,
    /// verify and load the YuE2-3B MoT (both paths and the NAR heads) and `qwen.tiktoken`. `dtype`
    /// is the MoT compute dtype (`DType::F32` on the CPU, which has no BF16 matmul; BF16 is the
    /// released accelerator dtype). The decoders stay on disk until a decode needs them.
    pub fn load(
        dirs: &SnapshotDirs,
        dtype: DType,
        device: &Device,
        generation: GenerationConfig,
        options: EngineOptions,
    ) -> gen_core::Result<Self> {
        let authorization = license::authorize(
            &[ComponentId::Lm, ComponentId::QwenTiktoken],
            IntendedUse::NoncommercialExperimentation,
        )
        .map_err(|e| gen_core::Error::Unsupported(e.to_string()))?;
        let start = Instant::now();
        let tokenizer = Yue2TextTokenizer::load(dirs).map_err(|e| match e {
            crate::tokenizer::TokenizerError::Asset(a) => gen_core::Error::from(a),
            other => msg(other),
        })?;
        let nar = Yue2Nar::load(dirs, dtype, device)?;
        let mut load_timing = Map::new();
        load_timing.insert(
            "resolve_verify_and_load_seconds".into(),
            json!(start.elapsed().as_secs_f64()),
        );
        Ok(Self {
            tokenizer,
            nar: Mutex::new(nar),
            vae_source: VaeSource::Snapshots(dirs.clone()),
            vaes: Mutex::new(BTreeMap::new()),
            mot_identity: component_identity(ComponentId::Lm.component()),
            tokenizer_identity: component_identity(ComponentId::QwenTiktoken.component()),
            generation,
            options,
            authorization,
            load_timing,
        })
    }

    /// An engine over the synthetic test models (tiny MoT with NAR heads, synthetic tokenizer
    /// ranks, the fixture VAEs).
    #[cfg(test)]
    pub(crate) fn synthetic(options: EngineOptions) -> Self {
        Self {
            tokenizer: {
                let bytes = std::fs::read(crate::test_fixtures::dir().join("synthetic.tiktoken"))
                    .expect("synthetic.tiktoken");
                Yue2TextTokenizer::padded_for_tests(&bytes).expect("synthetic table parses")
            },
            nar: Mutex::new(crate::nar::synthetic::model(1.0)),
            vae_source: VaeSource::Fixture,
            vaes: Mutex::new(BTreeMap::new()),
            mot_identity: json!({"component": "synthetic_mot", "weights_sha256": "synthetic"}),
            tokenizer_identity: json!({"component": "synthetic_tiktoken"}),
            generation: GenerationConfig::default(),
            options,
            authorization: license::authorize(
                &[ComponentId::Lm, ComponentId::QwenTiktoken],
                IntendedUse::NoncommercialExperimentation,
            )
            .expect("noncommercial experimentation is permitted"),
            load_timing: Map::new(),
        }
    }

    /// The tokenizer (to restore saved plans with [`SymbolicPlan::restore`]).
    pub fn tokenizer(&self) -> &Yue2TextTokenizer {
        &self.tokenizer
    }

    /// The engine's default generation configuration (upstream `pipe.generation_config`).
    pub fn generation_config(&self) -> &GenerationConfig {
        &self.generation
    }

    /// The memory controls.
    pub fn options(&self) -> &EngineOptions {
        &self.options
    }

    /// The MoT compute dtype.
    pub fn dtype(&self) -> DType {
        self.lock_nar()
            .map(|n| n.lm().dtype())
            .unwrap_or(DType::F32)
    }

    /// The device the MoT lives on.
    pub fn device(&self) -> Device {
        self.lock_nar()
            .map(|n| n.lm().device().clone())
            .unwrap_or(Device::Cpu)
    }

    /// Default [`SongSettings`] of this engine: its generation configuration and the standard
    /// decoder.
    pub fn default_settings(&self) -> SongSettings {
        SongSettings {
            generation: self.generation.clone(),
            decoder: VaeVariant::Standard,
        }
    }

    /// Lock the model, restoring AR weights a panicking synthesis left offloaded.
    fn lock_nar(&self) -> gen_core::Result<MutexGuard<'_, Yue2Nar>> {
        match self.nar.lock() {
            Ok(guard) => Ok(guard),
            Err(poisoned) => {
                let mut guard = poisoned.into_inner();
                guard.restore_ar()?;
                Ok(guard)
            }
        }
    }

    /// SHA-256 of the MoT weights the stages run (part of every stage identity).
    pub fn weights_sha256(&self) -> gen_core::Result<String> {
        Ok(self.lock_nar()?.weights_sha256().to_string())
    }

    /// The source / model identities recorded in artifacts (upstream `pipe.weights`, plus the
    /// tokenizer, the compute dtype and the device).
    pub fn model_identity(&self) -> gen_core::Result<Value> {
        let nar = self.lock_nar()?;
        Ok(json!({
            "engine": ENGINE_ID,
            "mot": self.mot_identity,
            "mot_native_weights_sha256": nar.weights_sha256(),
            "tokenizer": self.tokenizer_identity,
            "model_dtype": dtype_name(nar.lm().dtype()),
            "source": {
                "repository": crate::inventory::YUE2_SOURCE_REPO,
                "commit": crate::inventory::YUE2_SOURCE_COMMIT,
            },
        }))
    }

    /// The licence record for a generation that decodes with `decoder`: the use authorized, each
    /// component's recorded basis, and the attributions that must accompany the output.
    pub fn license_record(&self, decoder: VaeVariant) -> gen_core::Result<Value> {
        let auth = license::authorize_closure(
            Closure::Generation { vae: decoder },
            IntendedUse::NoncommercialExperimentation,
        )
        .map_err(|e| gen_core::Error::Unsupported(e.to_string()))?;
        let grants: Vec<Value> = auth
            .grants()
            .iter()
            .map(|(id, basis)| {
                json!({
                    "component": id.component().key,
                    "license": license::policy(*id).license.declared,
                    "family": basis.family,
                    "clause": basis.clause,
                })
            })
            .collect();
        Ok(json!({
            "intended_use": "noncommercial_experimentation",
            "grants": grants,
            "attributions": auth.attributions(),
            "note": "YuE2 weights are CC BY-NC 4.0: noncommercial experimentation only; commercial \
                     use and redistribution are not authorized.",
        }))
    }

    /// The load-time authorization (MoT + tokenizer).
    pub fn authorization(&self) -> &Authorization {
        &self.authorization
    }

    /// Resolve (immediately before loading), verify and load decoder `variant`, or reuse the one
    /// already loaded.
    pub fn vae(&self, variant: VaeVariant) -> gen_core::Result<Arc<Yue2Vae>> {
        let key = variant_name(variant);
        let mut vaes = self.vaes.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(vae) = vaes.get(key) {
            return Ok(Arc::clone(vae));
        }
        license::authorize_closure(
            Closure::Generation { vae: variant },
            IntendedUse::NoncommercialExperimentation,
        )
        .map_err(|e| gen_core::Error::Unsupported(e.to_string()))?;
        let vae = match &self.vae_source {
            VaeSource::Snapshots(dirs) => {
                let id = match variant {
                    VaeVariant::Standard => ComponentId::VaeStandard,
                    VaeVariant::Legacy => ComponentId::VaeLegacy,
                };
                let verified = snapshot::resolve_component(id, dirs)?;
                let device = self.device();
                Yue2Vae::load(&verified, VaeParts::DecoderOnly, &device).map_err(vae_error)?
            }
            #[cfg(test)]
            VaeSource::Fixture => crate::vae::tests::tiny(variant, VaeParts::DecoderOnly),
        };
        let vae = Arc::new(vae);
        vaes.insert(key, Arc::clone(&vae));
        Ok(vae)
    }

    /// Upstream `effective_config`: the generation configuration with its overrides, guidance,
    /// runtime, dtype, device, memory and decoder settings actually used.
    pub fn effective_config(&self, request: &SongRequest, settings: &SongSettings) -> Value {
        let generation = settings.generation.to_json();
        let defaults = GenerationConfig::default().to_json();
        let mut overrides = Map::new();
        if let (Some(g), Some(d)) = (generation.as_object(), defaults.as_object()) {
            for (k, v) in g {
                if d.get(k) != Some(v) {
                    overrides.insert(k.clone(), v.clone());
                }
            }
        }
        if request.guidance() != request.cot().default_guidance() {
            overrides.insert("cfg_scale".into(), json!(request.guidance()));
        }
        let (core, halo, decode) = match self.options.decode.mode {
            DecodeMode::Tiled { core_frames } => (
                json!(core_frames),
                json!(self.options.decode.halo_frames),
                "halo_crop",
            ),
            DecodeMode::Full => (Value::Null, Value::Null, "full"),
        };
        let query_tile = match self.options.nar.query_tile {
            QueryTile::Upstream => json!("upstream"),
            QueryTile::Whole => json!("whole"),
            QueryTile::Rows(n) => json!({"rows": n}),
            QueryTile::ScoreBytes(b) => json!({"score_bytes": b}),
        };
        let (dtype, device) = match self.lock_nar() {
            Ok(n) => (dtype_name(n.lm().dtype()), device_name(n.lm().device())),
            Err(_) => ("unknown", "unknown"),
        };
        json!({
            "engine": ENGINE_ID,
            "protocol": PROTOCOL_VERSION,
            "generation": generation,
            "overrides": overrides,
            "cot": request.cot().as_str(),
            "cfg_scale": request.guidance(),
            "cfg_negative": if request.cot() == CotMode::Off {
                "instruction_only"
            } else {
                "same_instruction_and_exact_abc"
            },
            "backend": "candle",
            "quantization": "none",
            "model_dtype": dtype,
            "vae_dtype": "float32",
            "vae_decode": decode,
            "vae_core_frames": core,
            "vae_halo_frames": halo,
            "device": device,
            "offload_ar": self.options.nar.offload_ar,
            "query_tile": query_tile,
            "decoder_release": variant_name(settings.decoder),
            "token_rng": "splitmix64 per stage from the request seed",
            "noise": "splitmix64 box-muller from the request seed",
            "runtime": {
                "crate": env!("CARGO_PKG_NAME"),
                "version": env!("CARGO_PKG_VERSION"),
                "upstream_commit": crate::inventory::YUE2_SOURCE_COMMIT,
            },
            "validation_status": "unvalidated",
        })
    }

    /// The run identity (upstream `identity({"request", "config", "weights"})`): a SHA-256 over
    /// the request, the effective configuration and the model identities. A resumed run must match
    /// it exactly.
    pub fn run_identity(
        &self,
        request: &SongRequest,
        settings: &SongSettings,
    ) -> gen_core::Result<String> {
        Ok(identity_of(&json!({
            "schema": IDENTITY_SCHEMA,
            "request": request.to_json(),
            "config": self.effective_config(request, settings),
            "weights": self.model_identity()?,
        })))
    }

    /// Identity of the plan stage for `request` sampled with `abc`: everything a planned score is
    /// a function of (request, ABC sampling, MoT weights, tokenizer, dtype).
    pub fn plan_stage_identity(
        &self,
        request: &SongRequest,
        abc: &Sampling,
    ) -> gen_core::Result<String> {
        Ok(identity_of(&json!([
            IDENTITY_SCHEMA,
            "plan",
            PROTOCOL_VERSION,
            request.to_json(),
            abc.to_json(),
            self.weights_sha256()?,
            self.tokenizer_identity,
            dtype_name(self.dtype()),
        ])))
    }

    /// Identity of the semantic stage: the exact plan, the semantic sampling, the MoT weights and
    /// the dtype.
    pub fn semantic_stage_identity(
        &self,
        plan: &SymbolicPlan,
        semantic: &Sampling,
    ) -> gen_core::Result<String> {
        Ok(identity_of(&json!([
            IDENTITY_SCHEMA,
            "semantic",
            PROTOCOL_VERSION,
            plan.identity().to_string(),
            semantic.to_json(),
            self.weights_sha256()?,
            dtype_name(self.dtype()),
        ])))
    }

    /// Identity of the acoustic stage: the acoustic stage identity of [`crate::nar`] (weights,
    /// prefix, codes, noise, steps, context, method) plus the dtype.
    pub fn synthesis_stage_identity(
        &self,
        semantic: &SemanticResult,
        generation: &GenerationConfig,
    ) -> gen_core::Result<String> {
        let noise = SongNoise::seeded(semantic.plan.request().seed(), semantic.codes.len());
        let request = SynthesisRequest {
            prefix: semantic.plan.prefix(),
            codes: &semantic.codes,
            noise: &noise,
            steps: step_count(generation),
            context: CONTEXT,
        };
        Ok(identity_of(&json!([
            IDENTITY_SCHEMA,
            "synthesis",
            request.stage_identity(&self.weights_sha256()?),
            dtype_name(self.dtype()),
        ])))
    }

    /// Plan `request` (upstream `pipe.plan`): `cot = off` and an external score need no sampling;
    /// otherwise the ABC is sampled with `abc` from the request seed, after the context-budget
    /// check. The plan carries the exact sampled token ids.
    pub fn plan(
        &self,
        request: &SongRequest,
        abc: &Sampling,
        hooks: &mut EngineHooks<'_>,
    ) -> gen_core::Result<SymbolicPlan> {
        hooks.check_cancel()?;
        hooks.observer.on_stage(Stage::Plan, StageEvent::Started);
        let plan = match SymbolicPlan::prepare(request.clone(), &self.tokenizer).map_err(msg)? {
            PlanStep::Ready(plan) => plan,
            PlanStep::GenerateAbc(planning) => {
                check_generation_budget(planning.prefix().len(), None, abc, 1.0).map_err(msg)?;
                let nar = self.lock_nar()?;
                let mut rng = stage_rng(request.seed());
                let mut forward = TokenForward {
                    stage: Stage::Plan,
                    observer: &mut *hooks.observer,
                };
                let (score, decoded) = plan_score(
                    nar.lm(),
                    cot_of(request.cot()),
                    planning.prefix(),
                    &sampler(abc),
                    &mut rng,
                    Hooks {
                        cancelled: hooks.cancelled,
                        observer: &mut forward,
                    },
                )?;
                drop(nar);
                planning
                    .finish(
                        &self.tokenizer,
                        score.abc_ids().to_vec(),
                        decode_timing(&decoded),
                        decoded.truncated,
                    )
                    .map_err(msg)?
            }
        };
        hooks.observer.on_stage(Stage::Plan, StageEvent::Finished);
        Ok(plan)
    }

    /// Generate the semantic codec tokens of `plan` (upstream `pipe.generate_semantic`). The plan is
    /// re-verified against its request first (its prefix must be the one the request and exact ABC
    /// ids produce), then guidance keeps the exact score in its negative branch (or only the
    /// instruction for `cot = off`).
    pub fn generate_semantic(
        &self,
        plan: &SymbolicPlan,
        semantic: &Sampling,
        hooks: &mut EngineHooks<'_>,
    ) -> gen_core::Result<SemanticResult> {
        hooks.check_cancel()?;
        hooks
            .observer
            .on_stage(Stage::Semantic, StageEvent::Started);
        let conditioning = plan
            .semantic_conditioning(&self.tokenizer, semantic)
            .map_err(msg)?;
        let score = ScorePlan::of_plan(plan)?;
        let nar = self.lock_nar()?;
        let mut rng = stage_rng(plan.request().seed());
        let mut forward = TokenForward {
            stage: Stage::Semantic,
            observer: &mut *hooks.observer,
        };
        let tokens = semantic_decode(
            nar.lm(),
            &SemanticInput {
                plan: &score,
                prefix: &conditioning.positive,
                negative: conditioning.negative.as_deref(),
                cfg_scale: conditioning.cfg_scale,
            },
            &sampler(semantic),
            &mut rng,
            Hooks {
                cancelled: hooks.cancelled,
                observer: &mut forward,
            },
        )?;
        drop(nar);
        hooks
            .observer
            .on_stage(Stage::Semantic, StageEvent::Finished);
        Ok(SemanticResult {
            plan: plan.clone(),
            truncated: tokens.decoded.truncated,
            timing: decode_timing(&tokens.decoded),
            codes: tokens.codes,
        })
    }

    /// Synthesize the acoustic latents of `semantic` (upstream `pipe.synthesize`): the song's noise
    /// drawn once from the request seed, `generation.ode_steps` midpoint steps over the protocol's
    /// original context chunks.
    pub fn synthesize(
        &self,
        semantic: &SemanticResult,
        generation: &GenerationConfig,
        hooks: &mut EngineHooks<'_>,
    ) -> gen_core::Result<Synthesis> {
        hooks.check_cancel()?;
        let expected = crate::protocol::token_prefixes(
            semantic.plan.request(),
            &self.tokenizer,
            Some(semantic.plan.abc_ids()),
        )
        .map_err(msg)?;
        if expected != semantic.plan.prefix() {
            return Err(msg(
                "the semantic result does not retain the request's exact prefix",
            ));
        }
        hooks
            .observer
            .on_stage(Stage::Synthesis, StageEvent::Started);
        let noise = SongNoise::seeded(semantic.plan.request().seed(), semantic.codes.len());
        let request = SynthesisRequest {
            prefix: semantic.plan.prefix(),
            codes: &semantic.codes,
            noise: &noise,
            steps: step_count(generation),
            context: CONTEXT,
        };
        let mut nar = self.lock_nar()?;
        let mut forward = ProgressForward {
            observer: &mut *hooks.observer,
        };
        let synthesis = nar_synthesize(
            &mut nar,
            &request,
            &self.options.nar,
            SynthesisHooks {
                cancelled: hooks.cancelled,
                observer: &mut forward,
            },
        )?;
        drop(nar);
        hooks
            .observer
            .on_stage(Stage::Synthesis, StageEvent::Finished);
        Ok(synthesis)
    }

    /// Decode verified latents with decoder `variant` (upstream `pipe.decode`): the latents are
    /// re-verified against their identity at the decode boundary; the output is clamped 48 kHz
    /// stereo carrying the decoder and latent identities.
    pub fn decode(
        &self,
        latents: &AcousticLatents,
        variant: VaeVariant,
        hooks: &mut EngineHooks<'_>,
    ) -> gen_core::Result<DecodedAudio> {
        hooks.check_cancel()?;
        let vae = self.vae(variant)?;
        hooks.check_cancel()?;
        hooks.observer.on_stage(Stage::Decode, StageEvent::Started);
        let observer = &mut *hooks.observer;
        let audio = decode_latents(
            &vae,
            latents,
            &self.options.decode,
            hooks.cancelled,
            &mut |completed, total| observer.on_decode_progress(completed, total),
        )
        .map_err(vae_error)?;
        hooks.observer.on_stage(Stage::Decode, StageEvent::Finished);
        Ok(audio)
    }

    /// One whole song in memory (upstream `pipe(...)`): plan → semantic → synthesis → decode. The
    /// artifact-backed form is [`Yue2Engine::generate_to`](crate::run).
    pub fn generate(
        &self,
        request: &SongRequest,
        settings: &SongSettings,
        hooks: &mut EngineHooks<'_>,
    ) -> gen_core::Result<SongResult> {
        let start = Instant::now();
        let plan = self.plan(request, settings.generation.abc(), hooks)?;
        self.generate_from_plan_timed(plan, settings, hooks, start)
    }

    /// Semantic → synthesis → decode from an exact plan (a fresh one, or one restored with
    /// [`SymbolicPlan::restore`]); an unchanged restored plan keeps its exact token ids.
    pub fn generate_from_plan(
        &self,
        plan: SymbolicPlan,
        settings: &SongSettings,
        hooks: &mut EngineHooks<'_>,
    ) -> gen_core::Result<SongResult> {
        self.generate_from_plan_timed(plan, settings, hooks, Instant::now())
    }

    fn generate_from_plan_timed(
        &self,
        plan: SymbolicPlan,
        settings: &SongSettings,
        hooks: &mut EngineHooks<'_>,
        start: Instant,
    ) -> gen_core::Result<SongResult> {
        let semantic = self.generate_semantic(&plan, settings.generation.semantic(), hooks)?;
        let synthesis = self.synthesize(&semantic, &settings.generation, hooks)?;
        hooks.check_cancel()?;
        let vae_start = Instant::now();
        let audio = self.decode(&synthesis.latents, settings.decoder, hooks)?;
        let mut timing = Map::new();
        timing.insert("abc".into(), Value::Object(plan.timing().clone()));
        timing.insert("semantic".into(), Value::Object(semantic.timing.clone()));
        timing.insert("nar_seconds".into(), json!(synthesis.seconds));
        timing.insert(
            "vae_seconds".into(),
            json!(vae_start.elapsed().as_secs_f64()),
        );
        timing.insert("load".into(), Value::Object(self.load_timing.clone()));
        timing.insert("e2e_seconds".into(), json!(start.elapsed().as_secs_f64()));
        let request = plan.request().clone();
        Ok(SongResult {
            config: self.effective_config(&request, settings),
            identity: self.run_identity(&request, settings)?,
            semantic,
            latents: synthesis.latents,
            audio,
            timing,
        })
    }

    /// The load timings.
    pub fn load_timing(&self) -> &Map<String, Value> {
        &self.load_timing
    }
}

fn step_count(generation: &GenerationConfig) -> usize {
    usize::try_from(generation.ode_steps()).unwrap_or(usize::MAX)
}

/// A decoder error as an engine error; a cancelled decode stays typed.
pub(crate) fn vae_error(e: crate::vae::VaeError) -> gen_core::Error {
    match e {
        crate::vae::VaeError::Cancelled { .. } => gen_core::Error::Canceled,
        crate::vae::VaeError::Asset(a) => gen_core::Error::from(a),
        other => msg(other),
    }
}
