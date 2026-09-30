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
//! # Precision (sc-22995)
//!
//! [`Yue2Engine::load_with_precision`] takes a [`ModelPrecision`]: the weight [`Tier`] the
//! `m-a-p/YuE2-3B` directory must hold (the released `bf16` checkpoint, or a derived `q8` / `q4`
//! tier snapshot, [`crate::tier`]; `None` loads whichever is staged) and the AR mode
//! ([`ArPrecision`]; the experimental FP8 mode of [`crate::fp8`], prepared before every AR stage
//! and restored to the exact BF16 originals before the acoustic stage). Both are recorded in the
//! effective configuration and in every stage identity they can change, so a stage computed at one
//! precision is never reused by a run at another.
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

use crate::decode::{decode_latents, DecodeMode, DecodeOptions, DecodedAudio};
use crate::fp8::{self, ArPrecision};
use crate::generate::{
    generate_semantic as semantic_decode, plan_score, stage_rng, timing_of, DecodeObserver, Hooks,
};
use crate::inventory::{Closure, Component, ComponentId, VaeVariant};
use crate::latent::AcousticLatents;
use crate::license::{self, Authorization, IntendedUse};
use crate::nar::{
    synthesize as nar_synthesize, NarOptions, QueryTile, SongNoise, Synthesis, SynthesisHooks,
    SynthesisObserver, SynthesisRequest, Yue2Nar,
};
use crate::plan::{PlanStep, SymbolicPlan};
use crate::precision::{Residency, Tier};
use crate::protocol::{
    check_generation_budget, CotMode, GenerationConfig, Sampling, SongRequest, CONTEXT,
    PROTOCOL_VERSION,
};
use crate::sampling::Phase;
use crate::snapshot::{self, SnapshotDirs};
use crate::tokenizer::Yue2TextTokenizer;
use crate::vae::{variant_name, VaeParts, Yue2Vae};

/// The provider's engine identity, written into every artifact record (distinct from YuE1's
/// `yue_*` engines, epic E1).
pub const ENGINE_ID: &str = "yue2";

/// Schema of the identities this module derives (plan / semantic / synthesis / decode stages and
/// the run identity).
pub const IDENTITY_SCHEMA: &str = "yue2-engine-v1";

/// The native runtime's build identity: a SHA-256 over this crate's `src/` tree (see `build.rs`).
/// Bound into the effective configuration and every stage identity, so a code change never reuses
/// a run or checkpoint produced by other code (upstream binds `runtime_sha256` the same way).
pub const SOURCE_DIGEST: &str = env!("YUE2_SOURCE_DIGEST");

/// Everything outside a stage's own inputs that its result depends on: the MoT weights, the
/// tokenizer, the compute dtype, the device (backend) and the runtime build. The stage identities
/// are pure functions of these keys and the stage inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct IdentityKeys {
    /// SHA-256 of the MoT weights file loaded.
    pub weights_sha256: String,
    /// The tokenizer's identity record.
    pub tokenizer: Value,
    /// The MoT compute dtype.
    pub dtype: DType,
    /// The device (backend) the stages ran on: `cpu`, `metal` or `cuda`.
    pub device: &'static str,
    /// The runtime build identity ([`SOURCE_DIGEST`]).
    pub runtime: &'static str,
    /// The loaded weight tier (`bf16` / `q8` / `q4`, sc-22995).
    pub tier: &'static str,
    /// The AR stages' mode (`none` / `fp8`, upstream's `quantization`, sc-22995).
    pub ar: &'static str,
    /// The decoder execution dtype; legacy FP32 keeps its historical identity shape.
    pub vae_dtype: DType,
    /// Explicit stage policy; Legacy keeps the existing stage-key shape.
    pub compute_policy: gen_core::Yue2ComputePolicy,
}

impl IdentityKeys {
    fn bind_policy(&self, mut input: Value) -> Value {
        if self.compute_policy != gen_core::Yue2ComputePolicy::Legacy {
            input
                .as_array_mut()
                .expect("stage identity is an array")
                .push(json!({
                    "compute_policy": policy_name(self.compute_policy)
                }));
        }
        input
    }

    /// The plan stage for `request` sampled with `abc`.
    pub fn plan(&self, request: &SongRequest, abc: &Sampling) -> String {
        identity_of(&self.bind_policy(json!([
            IDENTITY_SCHEMA,
            "plan",
            PROTOCOL_VERSION,
            request.to_json(),
            abc.to_json(),
            self.weights_sha256,
            self.tokenizer,
            dtype_name(self.dtype),
            self.device,
            self.runtime,
            {"tier": self.tier, "ar": self.ar},
        ])))
    }

    /// The semantic stage for `plan` sampled with `semantic`.
    pub fn semantic(&self, plan: &SymbolicPlan, semantic: &Sampling) -> String {
        identity_of(&self.bind_policy(json!([
            IDENTITY_SCHEMA,
            "semantic",
            PROTOCOL_VERSION,
            plan.identity().to_string(),
            semantic.to_json(),
            self.weights_sha256,
            self.tokenizer,
            dtype_name(self.dtype),
            self.device,
            self.runtime,
            {"tier": self.tier, "ar": self.ar},
        ])))
    }

    /// The acoustic stage identity of [`crate::nar`] (weights, dtype, prefix, codes, noise, steps,
    /// context, method) — the `stage_identity` its latents carry as their source.
    pub fn nar(&self, semantic: &SemanticResult, generation: &GenerationConfig) -> String {
        let noise = SongNoise::seeded(semantic.plan.request().seed(), semantic.codes.len());
        SynthesisRequest {
            prefix: semantic.plan.prefix(),
            codes: &semantic.codes,
            noise: &noise,
            steps: step_count(generation),
            context: CONTEXT,
        }
        .stage_identity(&self.weights_sha256, self.dtype)
    }

    /// A cached decode of the run `source` (its run identity) whose latents are `latent`, with
    /// the decoder identified by `vae`, by the MoT identified by `weights`. The decode tiling is not
    /// bound: it changes no sample (a memory control, [`MEMORY_CONFIG_KEYS`]), so a cached decode
    /// under other tiling is the same decode.
    pub fn cached_decode(
        &self,
        source: &str,
        latent: &Value,
        vae: &Value,
        weights: &Value,
    ) -> String {
        let mut input = json!({
            "schema": IDENTITY_SCHEMA,
            "kind": "cached_decode",
            "source": source,
            "latent": latent,
            "vae": vae,
            "weights": weights,
            "device": self.device,
            "runtime": self.runtime,
        });
        if self.vae_dtype != DType::F32 {
            input["vae_dtype"] = json!(dtype_name(self.vae_dtype));
        }
        if self.compute_policy != gen_core::Yue2ComputePolicy::Legacy {
            input["compute_policy"] = json!(policy_name(self.compute_policy));
        }
        identity_of(&input)
    }

    /// The synthesis stage: [`Self::nar`] plus the device, runtime and tier. Not the AR mode: the
    /// acoustic stage always runs the tier's own weights (the FP8 AR mode is restored to BF16
    /// before it, [`crate::fp8`]), so its latents are the same in both modes.
    pub fn synthesis(&self, semantic: &SemanticResult, generation: &GenerationConfig) -> String {
        identity_of(&self.bind_policy(json!([
            IDENTITY_SCHEMA,
            "synthesis",
            self.nar(semantic, generation),
            dtype_name(self.dtype),
            self.device,
            self.runtime,
            {"tier": self.tier},
        ])))
    }
}

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

/// The effective-configuration keys that record memory controls ([`EngineOptions`]). None
/// changes a result, so no run, stage or cached-decode identity binds them
/// ([`Yue2Engine::identity_config`]); `config.json` records them.
pub const MEMORY_CONFIG_KEYS: [&str; 5] = [
    "offload_ar",
    "query_tile",
    "vae_decode",
    "vae_core_frames",
    "vae_halo_frames",
];

/// Memory controls of the engine. Neither changes a result (they are recorded in the effective
/// configuration, not in any identity — see [`MEMORY_CONFIG_KEYS`]).
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
/// `abc_sampling` / `semantic_sampling` / `generation_config`, the pipeline's decoder and its
/// memory controls).
#[derive(Clone, Debug, PartialEq)]
pub struct SongSettings {
    /// Both phases' sampling and the midpoint ODE step count.
    pub generation: GenerationConfig,
    /// The decoder the audio is rendered with.
    pub decoder: VaeVariant,
    /// This request's memory controls (sc-22988); `None` ⇒ the engine's own
    /// ([`Yue2Engine::options`]). They change no result: the effective configuration records
    /// them, no stage identity binds them.
    pub options: Option<EngineOptions>,
}

impl Default for SongSettings {
    /// The released generation configuration, the standard (listening) decoder and the engine's
    /// memory controls.
    fn default() -> Self {
        Self {
            generation: GenerationConfig::default(),
            decoder: VaeVariant::Standard,
            options: None,
        }
    }
}

/// The precision a [`Yue2Engine`] loads the MoT at (see the [module docs](self#precision-sc-22995)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModelPrecision {
    /// The tier the YuE2-3B directory must hold; `None` loads the staged one.
    pub tier: Option<Tier>,
    /// How the AR stages multiply the AR projections.
    pub ar: ArPrecision,
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
    /// The fixture decoders, with availability checked against provisioned snapshot directories
    /// (unit tests of the early decoder check).
    #[cfg(test)]
    FixtureChecked(SnapshotDirs),
}

fn vae_component(variant: VaeVariant) -> ComponentId {
    match variant {
        VaeVariant::Standard => ComponentId::VaeStandard,
        VaeVariant::Legacy => ComponentId::VaeLegacy,
    }
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
    /// The MoT's compute dtype, device and weights digest, fixed at load (offload moves AR
    /// weights to host memory and back but changes none of these).
    dtype: DType,
    vae_dtype: DType,
    compute_policy: gen_core::Yue2ComputePolicy,
    device: Device,
    weights_sha256: String,
    /// The loaded weight tier and the AR mode (sc-22995).
    tier: Tier,
    ar: ArPrecision,
    /// Attention heads and position capacity of the MoT: the bound a per-request attention chunk
    /// is checked against before any compute (sc-22988).
    attention_heads: usize,
    max_positions: usize,
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
    crate::durable::sha256_hex(canonical_json(value).as_bytes())
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float32",
        DType::BF16 => "bfloat16",
        DType::F16 => "float16",
        _ => "other",
    }
}

fn policy_name(policy: gen_core::Yue2ComputePolicy) -> &'static str {
    match policy {
        gen_core::Yue2ComputePolicy::Legacy => "legacy",
        gen_core::Yue2ComputePolicy::Auto => "auto",
        gen_core::Yue2ComputePolicy::Bf16 => "bf16",
        gen_core::Yue2ComputePolicy::Fp32 => "fp32",
    }
}

fn check_stage_compute(
    policy: gen_core::Yue2ComputePolicy,
    model: DType,
    vae: DType,
    device: &Device,
    ar: ArPrecision,
) -> gen_core::Result<()> {
    use gen_core::Yue2ComputePolicy as Policy;
    let valid = match policy {
        Policy::Legacy => matches!(model, DType::F32 | DType::BF16) && vae == DType::F32,
        Policy::Auto => {
            model
                == if device.is_cpu() {
                    DType::F32
                } else {
                    DType::BF16
                }
                && vae == DType::F32
        }
        Policy::Bf16 => !device.is_cpu() && model == DType::BF16 && vae == DType::BF16,
        Policy::Fp32 => model == DType::F32 && vae == DType::F32,
    };
    if !valid {
        return Err(gen_core::Error::Unsupported(format!(
            "yue2: {policy:?} requires matching supported MoT/VAE stage dtypes on {device:?}; got {model:?}/{vae:?}"
        )));
    }
    if ar == ArPrecision::Fp8 && !matches!(policy, Policy::Auto | Policy::Legacy) {
        return Err(gen_core::Error::Unsupported(
            "yue2: experimental FP8 AR requires explicit Auto stage mixing (or a legacy load)"
                .into(),
        ));
    }
    Ok(())
}

fn device_name(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "metal",
    }
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
        Self::load_with_precision(
            dirs,
            dtype,
            device,
            ModelPrecision::default(),
            generation,
            options,
        )
    }

    /// [`Yue2Engine::load`] at an explicit [`ModelPrecision`]: an asserted tier the directory does
    /// not hold is refused, and the FP8 AR mode is refused before any weight is read unless the
    /// device is CUDA computing in BF16 (then prepared right after loading, which also checks the
    /// compute capability and the tier) — never served at another precision.
    pub fn load_with_precision(
        dirs: &SnapshotDirs,
        dtype: DType,
        device: &Device,
        precision: ModelPrecision,
        generation: GenerationConfig,
        options: EngineOptions,
    ) -> gen_core::Result<Self> {
        Self::load_with_stage_compute(
            dirs,
            dtype,
            DType::F32,
            gen_core::Yue2ComputePolicy::Legacy,
            device,
            precision,
            generation,
            options,
        )
    }

    /// Load at an explicit YuE2 stage compute policy. A VAE dtype mismatch is refused before
    /// snapshot verification, so a strict BF16/F32 request never silently runs a mixed decoder.
    #[allow(clippy::too_many_arguments)]
    pub fn load_with_stage_compute(
        dirs: &SnapshotDirs,
        dtype: DType,
        vae_dtype: DType,
        compute_policy: gen_core::Yue2ComputePolicy,
        device: &Device,
        precision: ModelPrecision,
        generation: GenerationConfig,
        options: EngineOptions,
    ) -> gen_core::Result<Self> {
        check_stage_compute(compute_policy, dtype, vae_dtype, device, precision.ar)?;
        if precision.ar == ArPrecision::Fp8 {
            fp8::check_fp8_request(precision.tier.unwrap_or(Tier::Bf16), dtype, device)?;
        }
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
        let mut nar = Yue2Nar::load_tier(dirs, precision.tier, dtype, device)?;
        if precision.ar == ArPrecision::Fp8 {
            fp8::prepare_fp8_ar(nar.lm_mut())?;
        }
        let (dtype, device, weights_sha256, tier) = (
            nar.lm().dtype(),
            nar.lm().device().clone(),
            nar.weights_sha256().to_string(),
            nar.lm().tier(),
        );
        let (attention_heads, max_positions) = (
            nar.lm().config().num_attention_heads,
            nar.lm().config().max_position_embeddings,
        );
        let mut mot_identity = component_identity(ComponentId::Lm.component());
        if tier != Tier::Bf16 {
            mot_identity["tier"] = json!({
                "tier": tier.name(),
                "conversion": crate::tier::CONVERSION_ID,
                "weights_sha256": weights_sha256,
            });
        }
        let mut load_timing = Map::new();
        load_timing.insert(
            "resolve_verify_and_load_seconds".into(),
            json!(start.elapsed().as_secs_f64()),
        );
        Ok(Self {
            tokenizer,
            nar: Mutex::new(nar),
            dtype,
            vae_dtype,
            compute_policy,
            device,
            weights_sha256,
            tier,
            ar: precision.ar,
            attention_heads,
            max_positions,
            vae_source: VaeSource::Snapshots(dirs.clone()),
            vaes: Mutex::new(BTreeMap::new()),
            mot_identity,
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
        Self::synthetic_with(
            crate::nar::synthetic::model(1.0),
            ArPrecision::Native,
            options,
        )
    }

    /// The synthetic engine behind the registered loader's gate: the same precision checks
    /// [`Yue2Engine::load_with_precision`] applies before reading weights (an FP8 AR request off
    /// CUDA/BF16, and an asserted tier that is not the staged one — the synthetic model holds the
    /// released `bf16` weights — are refused). `weights` names the staged directory in the refusal.
    #[cfg(test)]
    pub(crate) fn synthetic_at(
        precision: ModelPrecision,
        options: EngineOptions,
        weights: &std::path::Path,
    ) -> gen_core::Result<Self> {
        let engine = Self::synthetic(options);
        if precision.ar == ArPrecision::Fp8 {
            fp8::check_fp8_request(
                precision.tier.unwrap_or(Tier::Bf16),
                engine.dtype,
                &engine.device,
            )?;
        }
        crate::model::check_tier(precision.tier, engine.tier, weights)?;
        Ok(engine)
    }

    /// The synthetic engine on `device` computing in `dtype`, running the AR stages in `ar` (the
    /// FP8 mode is prepared as [`Yue2Engine::load_with_precision`] does).
    #[cfg(all(test, feature = "cuda"))]
    pub(crate) fn synthetic_on(device: &Device, dtype: DType, ar: ArPrecision) -> Self {
        let mut nar = crate::nar::synthetic::model_on_dtype(1.0, device, dtype);
        if ar == ArPrecision::Fp8 {
            fp8::prepare_fp8_ar(nar.lm_mut()).expect("FP8 AR on the synthetic model");
        }
        Self::synthetic_with(nar, ar, EngineOptions::default())
    }

    #[cfg(test)]
    fn synthetic_with(nar: Yue2Nar, ar: ArPrecision, options: EngineOptions) -> Self {
        Self {
            tokenizer: {
                let bytes = std::fs::read(crate::test_fixtures::dir().join("synthetic.tiktoken"))
                    .expect("synthetic.tiktoken");
                Yue2TextTokenizer::padded_for_tests(&bytes).expect("synthetic table parses")
            },
            dtype: nar.lm().dtype(),
            vae_dtype: DType::F32,
            compute_policy: gen_core::Yue2ComputePolicy::Legacy,
            device: nar.lm().device().clone(),
            tier: nar.lm().tier(),
            attention_heads: nar.lm().config().num_attention_heads,
            max_positions: nar.lm().config().max_position_embeddings,
            nar: Mutex::new(nar),
            weights_sha256: "synthetic".into(),
            ar,
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

    /// The synthetic engine, but with decoder availability checked against `dirs` (the fixture
    /// decoders still render).
    #[cfg(test)]
    pub(crate) fn with_checked_decoders(mut self, dirs: SnapshotDirs) -> Self {
        self.vae_source = VaeSource::FixtureChecked(dirs);
        self
    }

    #[cfg(test)]
    pub(crate) fn with_stage_compute(
        mut self,
        policy: gen_core::Yue2ComputePolicy,
        vae_dtype: DType,
    ) -> Self {
        self.compute_policy = policy;
        self.vae_dtype = vae_dtype;
        self
    }

    /// The tokenizer (to restore saved plans with [`SymbolicPlan::restore`]).
    pub fn tokenizer(&self) -> &Yue2TextTokenizer {
        &self.tokenizer
    }

    /// The engine's default generation configuration (upstream `pipe.generation_config`).
    pub fn generation_config(&self) -> &GenerationConfig {
        &self.generation
    }

    /// The engine's memory controls (what a request without its own uses).
    pub fn options(&self) -> &EngineOptions {
        &self.options
    }

    /// The memory controls `settings` runs with: its own, or the engine's.
    pub fn options_for(&self, settings: &SongSettings) -> EngineOptions {
        settings.options.unwrap_or(self.options)
    }

    /// The MoT's attention heads and position capacity: a per-request attention chunk
    /// ([`QueryTile::ScoreElements`]) of at least `heads × positions` elements holds a query row
    /// for every chunk this model can attend over.
    pub fn attention_bounds(&self) -> (usize, usize) {
        (self.attention_heads, self.max_positions)
    }

    /// The MoT compute dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The device the MoT lives on (the decoders load onto it too).
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The loaded weight tier.
    pub fn tier(&self) -> Tier {
        self.tier
    }

    /// The AR mode.
    pub fn ar_precision(&self) -> ArPrecision {
        self.ar
    }

    /// Measured resident weight bytes of the loaded MoT and NAR heads (the decoders are loaded on
    /// demand and are FP32 at every tier). In the FP8 AR mode the host-resident BF16 originals are
    /// [`Residency::host_bytes`].
    pub fn weight_residency(&self) -> gen_core::Result<Residency> {
        Ok(self.lock_nar()?.weight_residency())
    }

    /// The FP8 AR mode's status (upstream `quantization_status`).
    pub fn fp8_status(&self) -> gen_core::Result<fp8::Fp8Status> {
        Ok(fp8::status(self.lock_nar()?.lm()))
    }

    /// Put the exact BF16 AR originals back if the FP8 AR mode is active (upstream `restore_ar`):
    /// what anything that uses the model outside the AR stages must call first. The acoustic stage
    /// does it itself; the next AR stage prepares FP8 again.
    pub fn restore_ar_bf16(&self) -> gen_core::Result<()> {
        fp8::restore_ar_bf16(self.lock_nar()?.lm_mut())
    }

    /// Lock the model for an AR stage, preparing the FP8 AR mode first when the engine runs it.
    pub(crate) fn lock_nar_for_ar(&self) -> gen_core::Result<MutexGuard<'_, Yue2Nar>> {
        let mut nar = self.lock_nar()?;
        if self.ar == ArPrecision::Fp8 {
            fp8::prepare_fp8_ar(nar.lm_mut())?;
        }
        Ok(nar)
    }

    /// Default [`SongSettings`] of this engine: its generation configuration and the standard
    /// decoder.
    pub fn default_settings(&self) -> SongSettings {
        SongSettings {
            generation: self.generation.clone(),
            decoder: VaeVariant::Standard,
            options: None,
        }
    }

    /// Lock the model, restoring AR weights a panicking synthesis left offloaded.
    fn lock_nar(&self) -> gen_core::Result<MutexGuard<'_, Yue2Nar>> {
        match self.nar.lock() {
            Ok(guard) => Ok(guard),
            Err(poisoned) => {
                let mut guard = poisoned.into_inner();
                guard.restore_ar()?;
                // A stage that panicked may have left the FP8 AR mode half-applied: restore the
                // BF16 originals; the next AR stage prepares FP8 again.
                fp8::restore_ar_bf16(guard.lm_mut())?;
                Ok(guard)
            }
        }
    }

    /// SHA-256 of the MoT weights the stages run (part of every stage identity).
    pub fn weights_sha256(&self) -> gen_core::Result<String> {
        Ok(self.weights_sha256.clone())
    }

    /// The source / model identities recorded in artifacts (upstream `pipe.weights`, plus the
    /// tokenizer, the compute dtype and the device).
    pub fn model_identity(&self) -> gen_core::Result<Value> {
        Ok(json!({
            "engine": ENGINE_ID,
            "mot": self.mot_identity,
            "mot_native_weights_sha256": self.weights_sha256,
            "tokenizer": self.tokenizer_identity,
            "model_dtype": dtype_name(self.dtype),
            "weight_tier": self.tier.name(),
            "quantization": self.ar.name(),
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
                let verified = snapshot::resolve_component(vae_component(variant), dirs)?;
                let device = &self.device;
                Yue2Vae::load_with_dtype(&verified, VaeParts::DecoderOnly, device, self.vae_dtype)
                    .map_err(vae_error)?
            }
            #[cfg(test)]
            VaeSource::Fixture | VaeSource::FixtureChecked(_) => {
                crate::vae::tests::tiny_with_dtype(variant, VaeParts::DecoderOnly, self.vae_dtype)
            }
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
        let options = self.options_for(settings);
        let (core, halo, decode) = match options.decode.mode {
            DecodeMode::Tiled { core_frames } => (
                json!(core_frames),
                json!(options.decode.halo_frames),
                "halo_crop",
            ),
            DecodeMode::Full => (Value::Null, Value::Null, "full"),
        };
        let query_tile = match options.nar.query_tile {
            QueryTile::Upstream => json!("upstream"),
            QueryTile::Whole => json!("whole"),
            QueryTile::Rows(n) => json!({"rows": n}),
            QueryTile::ScoreBytes(b) => json!({"score_bytes": b}),
            QueryTile::ScoreElements(n) => json!({"score_elements": n}),
        };
        let (dtype, device) = (dtype_name(self.dtype), device_name(&self.device));
        let mut config = json!({
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
            "quantization": self.ar.name(),
            "weight_tier": self.tier.name(),
            "precision_policy": "sc-22995 V2 (candle_audio_yue2::precision)",
            "model_dtype": dtype,
            "vae_dtype": dtype_name(self.vae_dtype),
            "vae_decode": decode,
            "vae_core_frames": core,
            "vae_halo_frames": halo,
            "device": device,
            "offload_ar": options.nar.offload_ar,
            "query_tile": query_tile,
            "decoder_release": variant_name(settings.decoder),
            "token_rng": "splitmix64 per stage from the request seed",
            "noise": "splitmix64 box-muller from the request seed",
            "runtime": {
                "crate": env!("CARGO_PKG_NAME"),
                "version": env!("CARGO_PKG_VERSION"),
                "upstream_commit": crate::inventory::YUE2_SOURCE_COMMIT,
                "source_sha256": SOURCE_DIGEST,
            },
            "validation_status": "unvalidated",
        });
        if self.compute_policy != gen_core::Yue2ComputePolicy::Legacy {
            config["compute_policy"] = json!(policy_name(self.compute_policy));
            config["effective_stage_dtypes"] = json!({
                "ar": dtype,
                "nar": dtype,
                "vae_decoder": dtype_name(self.vae_dtype),
                "vae_encoder": dtype_name(self.vae_dtype),
            });
            config["fp32_numerical_internals"] = json!({
                "rmsnorm_reduction": true,
                "nar_sinusoid_construction": true,
                "logits_and_sampling": true,
                "vae_checkpoint_preparation": true,
                "vae_posterior_softplus": true,
                "durable_latents_and_wav": true,
                "ggml_quantized_matmul_operand_and_result": self.tier != Tier::Bf16,
            });
        }
        config
    }

    /// The effective configuration without its memory controls ([`MEMORY_CONFIG_KEYS`]): what a
    /// run's result depends on, and so what its identity binds. [`Yue2Engine::effective_config`]
    /// (the full record, `config.json`) keeps them.
    pub fn identity_config(&self, request: &SongRequest, settings: &SongSettings) -> Value {
        let mut config = self.effective_config(request, settings);
        if let Some(map) = config.as_object_mut() {
            for key in MEMORY_CONFIG_KEYS {
                map.remove(key);
            }
        }
        config
    }

    /// The run identity (upstream `identity({"request", "config", "weights"})`): a SHA-256 over
    /// the request, the effective configuration without its memory controls
    /// ([`Yue2Engine::identity_config`]) and the model identities. A resumed run must match it
    /// exactly; a run resumed under other memory controls (offload, attention chunking, decode
    /// tiling — none changes a result) is the same run.
    pub fn run_identity(
        &self,
        request: &SongRequest,
        settings: &SongSettings,
    ) -> gen_core::Result<String> {
        Ok(identity_of(&json!({
            "schema": IDENTITY_SCHEMA,
            "request": request.to_json(),
            "config": self.identity_config(request, settings),
            "weights": self.model_identity()?,
            "vae": self.vae_identity(settings.decoder)?,
        })))
    }

    /// Identity of the plan stage for `request` sampled with `abc`: everything a planned score is
    /// a function of (request, ABC sampling, MoT weights, tokenizer, dtype).
    pub fn plan_stage_identity(
        &self,
        request: &SongRequest,
        abc: &Sampling,
    ) -> gen_core::Result<String> {
        Ok(self.identity_keys().plan(request, abc))
    }

    /// Identity of the semantic stage: the exact plan, the semantic sampling, the MoT weights and
    /// the dtype.
    pub fn semantic_stage_identity(
        &self,
        plan: &SymbolicPlan,
        semantic: &Sampling,
    ) -> gen_core::Result<String> {
        Ok(self.identity_keys().semantic(plan, semantic))
    }

    /// Identity of the acoustic stage: the acoustic stage identity of [`crate::nar`] (weights,
    /// prefix, codes, noise, steps, context, method) plus the dtype.
    pub fn synthesis_stage_identity(
        &self,
        semantic: &SemanticResult,
        generation: &GenerationConfig,
    ) -> gen_core::Result<String> {
        Ok(self.identity_keys().synthesis(semantic, generation))
    }

    /// The keys every stage identity of this engine binds.
    pub fn identity_keys(&self) -> IdentityKeys {
        IdentityKeys {
            weights_sha256: self.weights_sha256.clone(),
            tokenizer: self.tokenizer_identity.clone(),
            dtype: self.dtype,
            device: device_name(&self.device),
            runtime: SOURCE_DIGEST,
            tier: self.tier.name(),
            ar: self.ar.name(),
            vae_dtype: self.vae_dtype,
            compute_policy: self.compute_policy,
        }
    }

    /// The pinned identity of decoder `variant` (its component record — every file's SHA-256),
    /// without loading it.
    pub fn vae_identity(&self, variant: VaeVariant) -> gen_core::Result<Value> {
        match &self.vae_source {
            VaeSource::Snapshots(_) => Ok(component_identity(vae_component(variant).component())),
            #[cfg(test)]
            VaeSource::Fixture | VaeSource::FixtureChecked(_) => {
                Ok(self.vae(variant)?.identity().to_json())
            }
        }
    }

    /// Refuse early — before any model compute — when decoder `variant`'s snapshot is not
    /// provisioned. An existence check only; the decoder is fully verified when it is loaded.
    pub fn check_decoder_available(&self, variant: VaeVariant) -> gen_core::Result<()> {
        let key = variant_name(variant);
        if self
            .vaes
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .contains_key(key)
        {
            return Ok(());
        }
        // Outside unit tests the source is always `Snapshots` (the fixture arms are test-only).
        #[allow(clippy::infallible_destructuring_match)]
        let dirs = match &self.vae_source {
            VaeSource::Snapshots(dirs) => dirs,
            #[cfg(test)]
            VaeSource::FixtureChecked(dirs) => dirs,
            #[cfg(test)]
            VaeSource::Fixture => return Ok(()),
        };
        dirs.snapshot_dir(&vae_component(variant).component().repo)?;
        Ok(())
    }

    /// Plan `request` (upstream `pipe.plan`): `cot = off` and an external score need no sampling;
    /// otherwise the ABC is sampled with `generation`'s ABC controls from the request seed, after
    /// the context-budget check. The plan carries the exact sampled token ids.
    pub fn plan(
        &self,
        request: &SongRequest,
        generation: &GenerationConfig,
        hooks: &mut EngineHooks<'_>,
    ) -> gen_core::Result<SymbolicPlan> {
        hooks.check_cancel()?;
        hooks.observer.on_stage(Stage::Plan, StageEvent::Started);
        let plan = match SymbolicPlan::prepare(request.clone(), &self.tokenizer).map_err(msg)? {
            PlanStep::Ready(plan) => plan,
            PlanStep::GenerateAbc(planning) => {
                check_generation_budget(planning.prefix().len(), None, generation.abc(), 1.0)
                    .map_err(msg)?;
                let nar = self.lock_nar_for_ar()?;
                let mut rng = stage_rng(request.seed());
                let mut forward = TokenForward {
                    stage: Stage::Plan,
                    observer: &mut *hooks.observer,
                };
                let (plan, _) = plan_score(
                    nar.lm(),
                    planning,
                    &self.tokenizer,
                    generation,
                    &mut rng,
                    Hooks {
                        cancelled: hooks.cancelled,
                        observer: &mut forward,
                    },
                )?;
                plan
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
        generation: &GenerationConfig,
        hooks: &mut EngineHooks<'_>,
    ) -> gen_core::Result<SemanticResult> {
        hooks.check_cancel()?;
        hooks
            .observer
            .on_stage(Stage::Semantic, StageEvent::Started);
        let conditioning = plan
            .semantic_conditioning(&self.tokenizer, generation.semantic())
            .map_err(msg)?;
        let nar = self.lock_nar_for_ar()?;
        let mut rng = stage_rng(plan.request().seed());
        let mut forward = TokenForward {
            stage: Stage::Semantic,
            observer: &mut *hooks.observer,
        };
        let tokens = semantic_decode(
            nar.lm(),
            plan,
            &conditioning,
            generation,
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
            timing: timing_of(&tokens.decoded),
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
        self.synthesize_with(semantic, generation, &self.options.nar, hooks)
    }

    /// [`Yue2Engine::synthesize`] with explicit acoustic memory controls (a request's own).
    pub fn synthesize_with(
        &self,
        semantic: &SemanticResult,
        generation: &GenerationConfig,
        nar_options: &NarOptions,
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
        // The acoustic stage prefills the song prefix with the AR path: always the exact BF16
        // originals, never the FP8 mode (upstream `synthesize` → `restore_ar`).
        fp8::restore_ar_bf16(nar.lm_mut())?;
        let mut forward = ProgressForward {
            observer: &mut *hooks.observer,
        };
        let synthesis = nar_synthesize(
            &mut nar,
            &request,
            nar_options,
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
        self.decode_with(latents, variant, &self.options.decode, hooks)
    }

    /// [`Yue2Engine::decode`] with explicit decode tiling (a request's own).
    pub fn decode_with(
        &self,
        latents: &AcousticLatents,
        variant: VaeVariant,
        decode: &DecodeOptions,
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
            decode,
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
        self.check_decoder_available(settings.decoder)?;
        let plan = self.plan(request, &settings.generation, hooks)?;
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
        self.check_decoder_available(settings.decoder)?;
        self.generate_from_plan_timed(plan, settings, hooks, Instant::now())
    }

    fn generate_from_plan_timed(
        &self,
        plan: SymbolicPlan,
        settings: &SongSettings,
        hooks: &mut EngineHooks<'_>,
        start: Instant,
    ) -> gen_core::Result<SongResult> {
        let options = self.options_for(settings);
        let semantic = self.generate_semantic(&plan, &settings.generation, hooks)?;
        let synthesis =
            self.synthesize_with(&semantic, &settings.generation, &options.nar, hooks)?;
        hooks.check_cancel()?;
        let vae_start = Instant::now();
        let audio =
            self.decode_with(&synthesis.latents, settings.decoder, &options.decode, hooks)?;
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

#[cfg(test)]
mod identity_tests {
    use super::*;
    use crate::protocol::SongRequestSpec;

    fn request() -> SongRequest {
        SongRequest::new(SongRequestSpec::new(
            "warm piano pop",
            "[Verse]\nla la la\n",
        ))
        .unwrap()
    }

    #[test]
    fn fp8_is_only_compatible_with_auto_or_historical_legacy_policy() {
        for policy in [
            gen_core::Yue2ComputePolicy::Bf16,
            gen_core::Yue2ComputePolicy::Fp32,
        ] {
            let dtype = if policy == gen_core::Yue2ComputePolicy::Bf16 {
                DType::BF16
            } else {
                DType::F32
            };
            let err = check_stage_compute(policy, dtype, dtype, &Device::Cpu, ArPrecision::Fp8)
                .unwrap_err();
            assert!(matches!(err, gen_core::Error::Unsupported(_)));
        }
        check_stage_compute(
            gen_core::Yue2ComputePolicy::Auto,
            DType::F32,
            DType::F32,
            &Device::Cpu,
            ArPrecision::Fp8,
        )
        .unwrap();
        check_stage_compute(
            gen_core::Yue2ComputePolicy::Legacy,
            DType::F32,
            DType::F32,
            &Device::Cpu,
            ArPrecision::Fp8,
        )
        .unwrap();
    }

    /// The run identity binds the result-changing configuration — tier, dtype, device, AR mode,
    /// generation settings, decoder — and none of the memory controls
    /// ([`MEMORY_CONFIG_KEYS`]), which `config.json` still records.
    ///
    /// Mutations that must fail: stop removing the memory keys; remove `model_dtype` /
    /// `weight_tier` / `device` from the projection as well.
    #[test]
    fn the_run_identity_binds_results_not_memory_controls() {
        let mut engine = Yue2Engine::synthetic(EngineOptions::default());
        let request = request();
        let plain = engine.default_settings();
        let bounded = SongSettings {
            options: Some(EngineOptions {
                nar: NarOptions {
                    query_tile: QueryTile::ScoreElements(1 << 30),
                    offload_ar: true,
                },
                decode: DecodeOptions::tiled(3).unwrap(),
            }),
            ..plain.clone()
        };
        let full = engine.effective_config(&request, &bounded);
        let projected = engine.identity_config(&request, &bounded);
        for key in MEMORY_CONFIG_KEYS {
            assert!(full.get(key).is_some(), "config.json records {key}");
            assert!(projected.get(key).is_none(), "the identity drops {key}");
        }
        for key in [
            "model_dtype",
            "weight_tier",
            "device",
            "quantization",
            "generation",
        ] {
            assert_eq!(projected[key], full[key], "the identity keeps {key}");
        }
        let id = engine.run_identity(&request, &plain).unwrap();
        assert_eq!(engine.run_identity(&request, &bounded).unwrap(), id);
        let legacy = SongSettings {
            decoder: VaeVariant::Legacy,
            ..plain.clone()
        };
        assert_ne!(
            engine.run_identity(&request, &legacy).unwrap(),
            id,
            "decoder"
        );
        let other_steps = SongSettings {
            generation: GenerationConfig::new(
                *plain.generation.abc(),
                *plain.generation.semantic(),
                plain.generation.ode_steps() as i64 + 1,
            )
            .unwrap(),
            ..plain.clone()
        };
        assert_ne!(
            engine.run_identity(&request, &other_steps).unwrap(),
            id,
            "steps"
        );
        let original = (engine.tier, engine.dtype, engine.ar);
        engine.tier = Tier::Q8;
        assert_ne!(engine.run_identity(&request, &plain).unwrap(), id, "tier");
        engine.tier = original.0;
        engine.dtype = if original.1 == DType::F32 {
            DType::BF16
        } else {
            DType::F32
        };
        assert_ne!(engine.run_identity(&request, &plain).unwrap(), id, "dtype");
        engine.dtype = original.1;
        engine.ar = ArPrecision::Fp8;
        assert_ne!(
            engine.run_identity(&request, &plain).unwrap(),
            id,
            "AR mode"
        );
        engine.ar = original.2;
        assert_eq!(engine.run_identity(&request, &plain).unwrap(), id);
        // The device is bound by name; the stage identities bind it too (see
        // `run::tests::every_stage_identity_binds_every_input_it_depends_on`).
        assert_eq!(projected["device"], "cpu");
    }

    #[test]
    fn explicit_policy_and_vae_dtype_bind_run_identity_without_changing_legacy() {
        let base = Yue2Engine::synthetic(EngineOptions::default());
        let settings = base.default_settings();
        let request = request();
        let legacy_id = base.run_identity(&request, &settings).unwrap();
        let legacy_config = base.effective_config(&request, &settings);
        assert!(legacy_config.get("compute_policy").is_none());
        assert_eq!(legacy_config["vae_dtype"], "float32");

        let auto = Yue2Engine::synthetic(EngineOptions::default())
            .with_stage_compute(gen_core::Yue2ComputePolicy::Auto, DType::F32);
        assert_ne!(auto.run_identity(&request, &settings).unwrap(), legacy_id);
        assert_eq!(
            auto.effective_config(&request, &settings)["compute_policy"],
            "auto"
        );

        let strict = Yue2Engine::synthetic(EngineOptions::default())
            .with_stage_compute(gen_core::Yue2ComputePolicy::Bf16, DType::BF16);
        assert_ne!(strict.run_identity(&request, &settings).unwrap(), legacy_id);
        assert_ne!(
            strict.run_identity(&request, &settings).unwrap(),
            auto.run_identity(&request, &settings).unwrap()
        );
        assert_eq!(
            strict.effective_config(&request, &settings)["vae_dtype"],
            "bfloat16"
        );

        let fp32 = base.with_stage_compute(gen_core::Yue2ComputePolicy::Fp32, DType::F32);
        assert_ne!(fp32.run_identity(&request, &settings).unwrap(), legacy_id);
    }
}
