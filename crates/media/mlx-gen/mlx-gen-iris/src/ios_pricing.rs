//! Header-only iOS counted pricing of the three Iris routes (epic 16774): `iris_3b` (generation,
//! including prompt batches), `iris_3b_restore` and `iris_3b_depth`.
//!
//! [`ios_pricing`] prices one request per phase — the component bytes resident under the request's
//! regime (eager or streamed load, staged or warm, the rung-4 window and scope, the rung-3 pixel-row
//! bound) plus the LARGEST counted quantum the phase checks — through the very formulas the runtime
//! guards check ([`crate::dit`], [`crate::text_encoder`], [`crate::stream`], [`crate::solver`],
//! [`crate::model`], [`crate::depth`], [`crate::restoration`]), each fed the widths the loaded
//! module would have, read here from the safetensors headers and the configs. Adapter inventory
//! shares the strict loader's format checks and bounded packed-shape scalar reader; no model or
//! adapter MLX tensors are allocated. Tokenization uses the actual tokenizer when needed.
//! [`ios_quantum_phase`] attributes every quantum label the crate checks to its phase.
//!
//! Exact source windows pair each solver evaluation's rows/history/serial carry with its forward.
//! The compatibility phase summary is a conservative envelope: maximum retained ownership plus
//! the largest guard scratch (and that scratch's binding), which need not come from one window.
//! Certificates use the exact windows, never a subtraction of independent summary maxima. `stream_tests::the_pricing_states_the_largest_check_of_
//! every_phase` checks the guard formulas real runs observe — every quantum
//! label priced at its largest observed check, no label priced that the run never checks —
//! for generation (eager rung 0 / rung 3, deferred rung 4 under each scope, text and prompt batch,
//! dense, quantized at load and packed), restoration and depth (eager, deferred, packed).
//!
//! Generation requires `generation: Some(actual_request)`: forward rows are the actual prompt
//! batch (doubled only at guided evaluation times), while `count` is serial with separately retained
//! solved outputs. `geometry.batch` remains the total output count for admission consistency, not
//! simultaneous activation rows. `token_window` (when non-zero) is the longest encoded window.

use std::path::Path;

use gen_core_iris::depth::DEPTH_EXTRA_INPUT_CHANNELS;
use gen_core_iris::downstream::TaskExport;
use gen_core_iris::restoration::RestorationResources;
use gen_core_iris::{
    DpmStep, GenerationParams, GenerationResources, IrisConfig, IrisTask, ModelConfig as IrisModel,
    Prediction, PROMPT_PREFIX,
};
use mlx_gen::gen_core::weightsmeta::{Dtype as Stored, SafetensorsTensorHeader as Header};
use mlx_gen::gen_core::{GenerationMemory, LoadShape, MemoryMode, OffloadPolicy, Precision};
use mlx_gen::ios_pricing::{IosPhase, IosPricing, IosPricingRequest, IosWindowPrice, IosWorkBasis};
use mlx_gen::ios_quantum::{sampler_step_bytes, tensor_bytes};
use mlx_gen::{Error, GenerationRequest, Quant, Result};

use crate::dit::{
    quanta as dit_q, EmbedGeometry, IrisDiT, PixelBlockGeometry, Stack, StackInventory,
    TextBlockGeometry, TrunkGeometry,
};
use crate::nn::{BackboneQuant, ProjectionLayout, QUANT_GROUP_SIZE};
use crate::retention::allocated;
use crate::stream::{largest_window, BlockRead, CastPolicy};
use crate::text_encoder::{quanta as te_q, LANGUAGE_TOWER_PREFIX};

mod bounder;
mod native_ledger;
pub use bounder::{
    ios_bounder, ios_bounder_with_replay, ios_bounder_with_replay_analysis, IrisBoundAnalysis,
    IrisReplay,
};

/// The request's finite source/request-derived HOST (Rust heap) growth upper outside the MLX
/// counter, including the caller-stated callback term. Refuses with the exact named gap while
/// any term is unproved (today: generation's tokenizer/regex upper; depth/restoration's
/// complete execution-time metadata coverage); never a partial sum.
pub fn host_growth_upper(
    req: &IosPricingRequest<'_>,
    callback: Option<mlx_gen::ios_pricing::IosCallbackHost>,
) -> Result<u64> {
    native_ledger::host_growth_upper(req, callback)
}

/// The OS thread-stack bytes the request's code explicitly requests (restoration's colour-fix
/// workers), a separate physical quantity never included in heap or native totals.
pub fn os_stack_request_bytes(req: &IosPricingRequest<'_>) -> Result<u64> {
    if req.provider == crate::restoration::MODEL_ID {
        native_ledger::restoration_worker_os_stack_request_bytes(req)
    } else {
        Ok(0)
    }
}

/// Canonical Iris sampling recipe. Normalization is exclusively the production
/// `GenerationParams::resolve` + `plan`; this is not the generic diffusion solver factory.
#[derive(Clone, Debug)]
pub struct GenerationRecipe {
    pub params: GenerationParams,
    pub plan: Vec<DpmStep>,
    /// Independent serial solves; each solve has `params.prompts.len()` rows.
    pub count: u32,
    /// Faithful replay against the same checkpoint config, with seed and sampling defaults fixed.
    pub replay: GenerationRequest,
}

/// One actual model evaluation and the solver/history state alive at that same epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationEpoch {
    pub serial: u32,
    pub index: usize,
    pub step: DpmStep,
    pub guided: bool,
    pub forward_rows: u64,
    /// The current implementation retains previous x0 even for order-1/Euler.
    pub history_rows: u64,
    /// Earlier clamped results held until the common RGB conversion.
    pub completed_rows: u64,
}

/// Validated canonical wire identity; no caller-provided class or floating-point equality keys.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SamplingToken(String);

/// Unresolved raw sampling controls for weights-free listing. None stays None: this is NOT a
/// normalized recipe, memory class, or assertion that a template is legal for an unseen config.
/// It contains no carrier text/media/seed/count/geometry and is safe to enumerate without cloning
/// those payloads. `clear_negative_prompt` is the no-CFG representative's explicit control action.
#[derive(
    Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(deny_unknown_fields)]
pub struct SamplingTemplate {
    steps: Option<u32>,
    sampler: Option<String>,
    guidance_bits: Option<u32>,
    scheduler_shift_bits: Option<u32>,
    interval_bits: Option<[u32; 2]>,
    clear_negative_prompt: bool,
}

impl SamplingTemplate {
    pub fn from_request(request: &GenerationRequest) -> Self {
        Self {
            steps: request.steps,
            sampler: request.sampler.clone(),
            guidance_bits: request.guidance.map(f32::to_bits),
            scheduler_shift_bits: request.scheduler_shift.map(f32::to_bits),
            interval_bits: request
                .cfg_interval
                .map(|(lo, hi)| [lo.to_bits(), hi.to_bits()]),
            clear_negative_prompt: false,
        }
    }

    pub fn to_token(&self) -> Result<String> {
        Ok(format!(
            "iris-sampling-template-v1:{}",
            serde_json::to_string(self)
                .map_err(|e| Error::Msg(format!("Iris raw sampling template: {e}")))?
        ))
    }

    pub fn from_token(token: &str) -> Result<Self> {
        let json = token
            .strip_prefix("iris-sampling-template-v1:")
            .ok_or_else(|| {
                Error::Unsupported("invalid Iris raw sampling template prefix".into())
            })?;
        let template: Self = serde_json::from_str(json)
            .map_err(|e| Error::Msg(format!("Iris raw sampling template: {e}")))?;
        if template.to_token()? != token {
            return Err(Error::Unsupported(
                "Iris raw sampling template is not canonical".into(),
            ));
        }
        Ok(template)
    }

    pub fn to_value(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self)
            .map_err(|e| Error::Msg(format!("Iris raw sampling template: {e}")))
    }

    fn apply_raw(&self, request: &mut GenerationRequest) {
        request.steps = self.steps;
        request.sampler = self.sampler.clone();
        request.guidance = self.guidance_bits.map(f32::from_bits);
        request.scheduler_shift = self.scheduler_shift_bits.map(f32::from_bits);
        request.cfg_interval = self
            .interval_bits
            .map(|[lo, hi]| (f32::from_bits(lo), f32::from_bits(hi)));
        if self.clear_negative_prompt {
            request.negative_prompt = None;
        }
    }

    /// Mutate only captured controls, then validate through the actual config/served factory.
    /// No carrier payload is cloned. An Err leaves the raw controls applied, but authorizes no work.
    /// Raw defaults resolve HERE, never to a fabricated weights-free class.
    pub fn apply_to(
        &self,
        config: &IrisConfig,
        carrier: &mut GenerationRequest,
    ) -> Result<SamplingToken> {
        self.apply_raw(carrier);
        canonical_token(config, carrier)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SamplingWire {
    order: u8,
    steps: u32,
    guidance_bits: u32,
    interval_bits: [u32; 2],
    flow_shift_bits: u64,
    prediction: String,
    num_train_timesteps: u64,
    prompt_rows: u32,
    serial_count: u32,
}

/// Computed graph semantics, not continuous scalar values or an admission permission. Epoch runs
/// are compressed only when their actual (guided, history, second-update, terminal) facts match.
/// Prompt rows/count remain distinct until the complete additivity proof permits sharing.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SamplingMemoryClass {
    pub order: u8,
    pub clean_prediction: bool,
    pub prompt_rows: u32,
    pub serial_count: u32,
    pub epoch_runs: Vec<[bool; 4]>,
}

impl SamplingWire {
    fn of(config: &IrisConfig, params: &GenerationParams, count: u32) -> Self {
        Self {
            order: params.order as u8,
            steps: params.steps as u32,
            guidance_bits: params.cfg_scale.to_bits(),
            interval_bits: [
                (params.cfg_interval.0 as f32).to_bits(),
                (params.cfg_interval.1 as f32).to_bits(),
            ],
            flow_shift_bits: params.shift.to_bits(),
            prediction: match params.prediction {
                Prediction::Velocity => "v",
                Prediction::Clean => "x",
            }
            .into(),
            num_train_timesteps: config.flow.num_train_timesteps as u64,
            prompt_rows: params.prompts.len() as u32,
            serial_count: count,
        }
    }

    fn metadata_config(&self, actual: &IrisConfig) -> Result<IrisConfig> {
        let mut config = actual.clone();
        config.flow.shift = f64::from_bits(self.flow_shift_bits);
        config.flow.prediction = self.prediction.clone();
        config.flow.num_train_timesteps =
            usize::try_from(self.num_train_timesteps).map_err(|_| {
                Error::Msg("Iris recipe timestep count does not fit this target".into())
            })?;
        config.validate_supported().map_err(Error::from)?;
        Ok(config)
    }

    /// Validate normalized metadata directly through the actual pure plan/gate methods: no fake
    /// architecture config, prompts or invented checkpoint defaults are needed to parse a family.
    fn checked_plan(&self) -> Result<gen_core_iris::DpmSolverPlan> {
        let images = self
            .prompt_rows
            .checked_mul(self.serial_count)
            .ok_or_else(|| Error::Msg("Iris recipe image count overflow".into()))?;
        let guidance = f32::from_bits(self.guidance_bits);
        let interval = (
            f32::from_bits(self.interval_bits[0]) as f64,
            f32::from_bits(self.interval_bits[1]) as f64,
        );
        let shift = f64::from_bits(self.flow_shift_bits);
        if self.prompt_rows == 0
            || self.serial_count == 0
            || images > crate::model::MAX_COUNT
            || !guidance.is_finite()
            || !shift.is_finite()
            || shift <= 0.0
            || !(0.0..=1.0).contains(&interval.0)
            || !(0.0..=1.0).contains(&interval.1)
            || interval.0 >= interval.1
            || self.num_train_timesteps == 0
            || usize::try_from(self.num_train_timesteps).is_err()
            || Prediction::from_name(&self.prediction).is_none()
        {
            return Err(Error::Unsupported(
                "invalid normalized Iris sampling metadata".into(),
            ));
        }
        if guidance == 1.0 && self.interval_bits != [0.0f32.to_bits(), 1.0f32.to_bits()] {
            return Err(Error::Unsupported(
                "unguided Iris recipe cannot carry an ineffective interval".into(),
            ));
        }
        let plan =
            gen_core_iris::DpmSolverPlan::new(self.steps as usize, self.order as usize, shift)
                .map_err(Error::from)?;
        if guidance != 1.0 && plan.cfg_span(guidance, interval).is_none() {
            return Err(Error::Unsupported(
                "Iris recipe interval contains no actual guided evaluation".into(),
            ));
        }
        Ok(plan)
    }

    fn apply(
        &self,
        actual: &IrisConfig,
        carrier: &GenerationRequest,
    ) -> Result<(IrisConfig, GenerationRequest)> {
        let config = self.metadata_config(actual)?;
        if carrier.prompt_batch.len().max(1) != self.prompt_rows as usize {
            return Err(Error::Unsupported(
                "Iris recipe prompt rows disagree with the actual mode carrier".into(),
            ));
        }
        let mut request = carrier.clone();
        request.sampler = Some(
            match self.order {
                1 => "euler",
                2 => "dpmpp_2m",
                _ => {
                    return Err(Error::Unsupported(
                        "Iris recipe order is not an actual served solver".into(),
                    ))
                }
            }
            .into(),
        );
        request.steps = Some(self.steps);
        request.guidance = Some(f32::from_bits(self.guidance_bits));
        request.cfg_interval = if request.guidance == Some(1.0) {
            None
        } else {
            Some((
                f32::from_bits(self.interval_bits[0]),
                f32::from_bits(self.interval_bits[1]),
            ))
        };
        // Replay through sampling-only metadata, NOT a lossy f64 -> request-f32 conversion.
        request.scheduler_shift = None;
        request.count = self.serial_count;
        let params = resolved_generation_controls(&config, &request, request.seed.unwrap_or(0))?;
        if Self::of(&config, &params, request.count) != *self {
            return Err(Error::Unsupported(
                "Iris recipe is not canonical under the actual normalizer/plan".into(),
            ));
        }
        Ok((config, request))
    }
}

/// Read the actual provisioned Iris metadata config, without tensor loads or guessed defaults.
pub fn generation_metadata_config(spec: &mlx_gen::LoadSpec) -> Result<IrisConfig> {
    let resources =
        GenerationResources::from_spec(spec, crate::model::MODEL_ID).map_err(Error::from)?;
    IrisConfig::from_dir(&resources.backbone_dir).map_err(Error::from)
}

/// Exact resolved sampling identity. Prompts/media/seed/step arrays are deliberately not wire data.
pub fn canonical_token(config: &IrisConfig, request: &GenerationRequest) -> Result<SamplingToken> {
    let params = resolved_generation_controls(config, request, request.seed.unwrap_or(0))?;
    let wire = SamplingWire::of(config, &params, request.count);
    let encoded = format!(
        "iris-sampling-v1:{}",
        serde_json::to_string(&wire).map_err(|e| Error::Msg(e.to_string()))?
    );
    SamplingToken::from_token(&encoded)
}

impl SamplingToken {
    pub fn from_token(token: &str) -> Result<Self> {
        let payload = token
            .strip_prefix("iris-sampling-v1:")
            .ok_or_else(|| Error::Unsupported("not an Iris sampling-v1 token".into()))?;
        let wire: SamplingWire =
            serde_json::from_str(payload).map_err(|e| Error::Msg(e.to_string()))?;
        wire.checked_plan()?;
        let canonical = format!(
            "iris-sampling-v1:{}",
            serde_json::to_string(&wire).map_err(|e| Error::Msg(e.to_string()))?
        );
        if canonical != token {
            return Err(Error::Unsupported(
                "Iris recipe token is not canonical".into(),
            ));
        }
        Ok(Self(canonical))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Exact normalized wire metadata, including lossless f64/f32 bit representations.
    pub fn to_value(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self.wire()?).map_err(|e| Error::Msg(e.to_string()))
    }

    pub fn steps(&self) -> Result<u32> {
        Ok(self.wire()?.steps)
    }

    fn wire(&self) -> Result<SamplingWire> {
        serde_json::from_str(
            self.0
                .strip_prefix("iris-sampling-v1:")
                .ok_or_else(|| Error::Unsupported("invalid Iris recipe prefix".into()))?,
        )
        .map_err(|e| Error::Msg(e.to_string()))
    }

    pub fn metadata_config(&self, actual: &IrisConfig) -> Result<IrisConfig> {
        self.wire()?.metadata_config(actual)
    }

    pub fn apply_to(
        &self,
        actual_config: &IrisConfig,
        carrier: &GenerationRequest,
    ) -> Result<(IrisConfig, GenerationRequest)> {
        self.wire()?.apply(actual_config, carrier)
    }

    /// Replay against the ACTUAL runtime config, without overriding artifact metadata or cloning
    /// carrier payloads. Work cardinality must already match; only sampling controls are changed.
    pub fn apply_runtime_to(
        &self,
        actual_config: &IrisConfig,
        carrier: &mut GenerationRequest,
    ) -> Result<()> {
        let wire = self.wire()?;
        wire.checked_plan()?;
        let prediction = match actual_config.prediction().map_err(Error::from)? {
            Prediction::Velocity => "v",
            Prediction::Clean => "x",
        };
        if prediction != wire.prediction
            || actual_config.flow.num_train_timesteps as u64 != wire.num_train_timesteps
            || carrier.prompt_batch.len().max(1) != wire.prompt_rows as usize
            || carrier.count != wire.serial_count
        {
            return Err(Error::Unsupported(
                "Iris runtime recipe disagrees with actual artifact/work basis".into(),
            ));
        }
        let shift = f64::from_bits(wire.flow_shift_bits);
        let scheduler_shift = if actual_config.flow.shift.to_bits() == wire.flow_shift_bits {
            None
        } else {
            let request_shift = shift as f32;
            if (request_shift as f64).to_bits() != wire.flow_shift_bits {
                return Err(Error::Unsupported("Iris runtime recipe shift is neither the actual f64 default nor an exact f32 override".into()));
            }
            Some(request_shift)
        };
        let sampler = match wire.order {
            1 => "euler",
            2 => "dpmpp_2m",
            _ => {
                return Err(Error::Unsupported(
                    "Iris runtime recipe order is not a served solver".into(),
                ))
            }
        };
        let previous = SamplingTemplate::from_request(carrier);
        carrier.sampler = Some(sampler.into());
        carrier.steps = Some(wire.steps);
        carrier.guidance = Some(f32::from_bits(wire.guidance_bits));
        carrier.cfg_interval = if carrier.guidance == Some(1.0) {
            None
        } else {
            Some((
                f32::from_bits(wire.interval_bits[0]),
                f32::from_bits(wire.interval_bits[1]),
            ))
        };
        carrier.scheduler_shift = scheduler_shift;
        let result = canonical_token(actual_config, carrier).and_then(|actual| {
            if actual == *self {
                Ok(())
            } else {
                Err(Error::Unsupported(
                    "Iris runtime replay changed canonical sampling identity".into(),
                ))
            }
        });
        if result.is_err() {
            previous.apply_raw(carrier);
        }
        result
    }

    pub fn memory_class(&self) -> Result<SamplingMemoryClass> {
        let wire = self.wire()?;
        let plan = wire.checked_plan()?;
        let guidance = f32::from_bits(wire.guidance_bits);
        let interval = (
            f32::from_bits(wire.interval_bits[0]) as f64,
            f32::from_bits(wire.interval_bits[1]) as f64,
        );
        let epoch_runs = epoch_class_runs(&plan, guidance, interval);
        Ok(SamplingMemoryClass {
            order: wire.order,
            clean_prediction: wire.prediction == "x",
            prompt_rows: wire.prompt_rows,
            serial_count: wire.serial_count,
            epoch_runs,
        })
    }
}

fn epoch_class_runs(
    plan: &gen_core_iris::DpmSolverPlan,
    guidance: f32,
    interval: (f64, f64),
) -> Vec<[bool; 4]> {
    let mut runs = Vec::new();
    for index in plan.epoch_indices(guidance, interval) {
        let step = plan.step(index).expect("bounded canonical epoch");
        let facts = [
            gen_core_iris::cfg_active(guidance, step.s, interval),
            index != 0,
            matches!(step.update, gen_core_iris::DpmUpdate::Second { .. }),
            index + 1 == plan.len(),
        ];
        if runs.last() != Some(&facts) {
            runs.push(facts);
        }
    }
    runs
}

/// Finite graph-mechanism representatives, not a measurement/step grid. First + terminal and at
/// most two monotone CFG transitions require at most three steady slots (five evaluations total).
/// Deduplication keeps the shortest actual producer-derived representative of each epoch-run class.
/// N=2 is never claimed to observe an interior Second update.
/// Configured mechanism representatives without a carrier payload clone. Build thresholds from
/// the ACTUAL default-shift indexed plan as well as the explicit unit-shift override. Prefer the
/// shortest canonical N per source-derived mechanism class, then actual default shift on ties.
/// This preference is NOT a physical-equivalence certificate: pricing retains each request's
/// actual solver/plan-vector payload, and certified transfer still requires complete window proof.
/// Full actual-default provenance remains available via canonical_token and the raw carrier
/// template, independently of this capture preference. Original controls/negative ownership are
/// restored on success AND failure; the full default token must not force a long capture run.
pub fn sampling_representatives_in_place(
    config: &IrisConfig,
    carrier: &mut GenerationRequest,
) -> Result<Vec<SamplingToken>> {
    let actual = canonical_token(config, carrier)?;
    let actual_wire = actual.wire()?;
    let default_shift = config.flow.shift.to_bits();
    let rank = (
        actual_wire.steps,
        actual_wire.flow_shift_bits != default_shift,
    );
    let mut by_class = std::collections::BTreeMap::from([(actual.memory_class()?, (rank, actual))]);
    let mut templates = sampling_grammar_templates(config.flow.shift, None)?;
    templates.extend(sampling_grammar_templates(1.0, Some(1.0))?);
    let controls = SamplingTemplate::from_request(carrier);
    let mut negative = carrier.negative_prompt.take();
    for template in templates {
        if !template.clear_negative_prompt {
            carrier.negative_prompt = negative.take();
        }
        let candidate = template.apply_to(config, carrier).and_then(|token| {
            let class = token.memory_class()?;
            let wire = token.wire()?;
            Ok((
                token,
                class,
                (wire.steps, wire.flow_shift_bits != default_shift),
            ))
        });
        if !template.clear_negative_prompt {
            negative = carrier.negative_prompt.take();
        }
        match candidate {
            Ok((token, class, rank)) => match by_class.entry(class) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((rank, token));
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    if rank < entry.get().0 {
                        entry.insert((rank, token));
                    }
                }
            },
            Err(error) => {
                controls.apply_raw(carrier);
                carrier.negative_prompt = negative;
                return Err(error);
            }
        }
    }
    controls.apply_raw(carrier);
    carrier.negative_prompt = negative;
    Ok(by_class.into_values().map(|(_, token)| token).collect())
}

/// Weights-free raw controls only: unresolved default flow shift is preserved until the actual
/// provisioned config is read. Enumerating this grammar never clones carrier text/media/seed.
pub fn sampling_representative_templates(carrier: &GenerationRequest) -> Vec<SamplingTemplate> {
    let mut representatives =
        sampling_grammar_templates(1.0, Some(1.0)).expect("constant valid Iris sampling grammar");
    // Unresolved actual defaults remain provenance, not a fabricated weights-free class.
    representatives.push(SamplingTemplate::from_request(carrier));
    representatives
}

fn sampling_grammar_templates(
    shift: f64,
    shift_override: Option<f32>,
) -> Result<Vec<SamplingTemplate>> {
    let mut by_class = std::collections::BTreeMap::new();
    for sampler in gen_core_iris::SAMPLERS {
        let order = match sampler {
            "euler" => 1,
            "dpmpp_2m" => 2,
            _ => unreachable!("served Iris solver needs a canonical order"),
        };
        // This finite grammar bound is independent of any requested step count or hardware cell.
        for steps in 1..=5 {
            let plan =
                gen_core_iris::DpmSolverPlan::new(steps, order, shift).map_err(Error::from)?;
            let mut controls = vec![(1.0, None)];
            for start in 0..steps {
                for end in start + 1..=steps {
                    let hi = if start == 0 {
                        1.0
                    } else {
                        ((plan.step(start - 1).unwrap().s + plan.step(start).unwrap().s) * 0.5)
                            as f32
                    };
                    let lo = if end == steps {
                        0.0
                    } else {
                        ((plan.step(end - 1).unwrap().s + plan.step(end).unwrap().s) * 0.5) as f32
                    };
                    // Request endpoints are f32 while the actual indexed plan is f64. Collapsed
                    // or empty intervals are NOT served controls and receive no class/coverage
                    // claim. Classify each legal candidate by its ACTUAL span, never the desired
                    // start/end indices before endpoint rounding.
                    if (0.0..=1.0).contains(&lo)
                        && (0.0..=1.0).contains(&hi)
                        && lo < hi
                        && plan.cfg_span(3.0, (lo as f64, hi as f64)).is_some()
                    {
                        controls.push((3.0, Some((lo, hi))));
                    }
                }
            }
            for (guidance, interval) in controls {
                let bounds = interval.map_or((0.0, 1.0), |(lo, hi)| (lo as f64, hi as f64));
                let class = (order, epoch_class_runs(&plan, guidance, bounds));
                let template = SamplingTemplate {
                    sampler: Some(sampler.into()),
                    steps: Some(steps as u32),
                    guidance_bits: Some(guidance.to_bits()),
                    scheduler_shift_bits: shift_override.map(f32::to_bits),
                    interval_bits: interval.map(|(lo, hi)| [lo.to_bits(), hi.to_bits()]),
                    clear_negative_prompt: guidance == 1.0,
                };
                by_class.entry(class).or_insert(template);
            }
        }
    }
    Ok(by_class.into_values().collect())
}

fn resolved_generation_controls(
    config: &IrisConfig,
    request: &GenerationRequest,
    default_seed: u64,
) -> Result<GenerationParams> {
    crate::model::descriptor()
        .capabilities
        .validate_request(crate::model::MODEL_ID, request)
        .map_err(Error::from)?;
    gen_core_iris::validate_generation_request(crate::model::MODEL_ID, request, config)
        .map_err(Error::from)?;
    crate::model::refuse_unhonored_memory(request)?;
    GenerationParams::resolve(request, default_seed, config).map_err(Error::from)
}

/// Executable post-admission recipe. Token/class/price paths use bounded metadata instead.
pub fn resolved_generation_recipe(
    config: &IrisConfig,
    request: &GenerationRequest,
    default_seed: u64,
) -> Result<GenerationRecipe> {
    let params = resolved_generation_controls(config, request, default_seed)?;
    let plan = params.plan().map_err(Error::from)?;
    let mut replay = request.clone();
    replay.seed = Some(params.seed);
    replay.steps = Some(params.steps as u32);
    replay.guidance = Some(params.cfg_scale);
    replay.cfg_interval = params
        .uses_cfg()
        .then_some((params.cfg_interval.0 as f32, params.cfg_interval.1 as f32));
    replay.sampler = Some(
        if params.order == 1 {
            "euler"
        } else {
            "dpmpp_2m"
        }
        .into(),
    );
    // Preserve an absent scheduler_shift: config.flow.shift is f64 and need not round-trip via
    // the request's f32 override. Replay uses that same checkpoint, never a rounded substitute.
    Ok(GenerationRecipe {
        count: request.count,
        params,
        plan,
        replay,
    })
}

impl GenerationRecipe {
    pub fn epochs(&self) -> impl Iterator<Item = GenerationEpoch> + '_ {
        let rows = self.params.prompts.len() as u64;
        (0..self.count).flat_map(move |serial| {
            self.plan
                .iter()
                .copied()
                .enumerate()
                .map(move |(index, step)| {
                    let guided = self.params.cfg_at(&step);
                    GenerationEpoch {
                        serial,
                        index,
                        step,
                        guided,
                        forward_rows: rows * if guided { 2 } else { 1 },
                        history_rows: if index == 0 { 0 } else { rows },
                        completed_rows: rows * u64::from(serial),
                    }
                })
        })
    }

    /// Provenance is exact resolved controls, not a continuous-float memory-family grid.
    pub fn provenance(&self) -> serde_json::Value {
        serde_json::json!({
            "sampler": if self.params.order == 1 { "euler" } else { "dpmpp_2m" },
            "order": self.params.order, "steps": self.plan.len(),
            "guidance": self.params.cfg_scale, "cfg_interval": self.params.cfg_interval,
            "flow_shift": self.params.shift,
            "prediction": match self.params.prediction { Prediction::Velocity => "v", Prediction::Clean => "x" },
            "prompt_rows": self.params.prompts.len(), "serial_count": self.count,
            "width": self.params.width, "height": self.params.height,
            "seed": self.params.seed, "mode": "text_to_image",
            "prompt_batch": self.replay.prompt_batch, "prompt": self.replay.prompt,
            "negative_prompt": self.params.negative_prompt,
        })
    }
}

/// Whether an `iris_3b` prompt batch cannot be derived by scaling the text anchor.
///
/// Prompt rows and serial count retain different owners, so no batch-1 scale or additive law is
/// claimed. A batch is certified only by [`ios_bounder`], which prices the ACTUAL rows/count with
/// the cumulative native ledger rather than scaling or reusing a batch-1 certificate.
pub const IOS_NON_ADDITIVE_BATCH: bool = true;

/// The phase every quantum label of this crate runs in.
pub fn ios_quantum_phase(label: &str) -> Option<IosPhase> {
    use crate::depth::quanta as depth_q;
    use crate::restoration::quanta as restore_q;
    let phase = match label {
        te_q::LOAD
        | crate::stream::EXPORT_TARGET_QUANTUM
        | crate::nn::PENDING_CONSTRUCTION_QUANTUM
        | crate::nn::ADAPTED_BASE_EVAL_QUANTUM
        | crate::stream::LOAD_TARGET_QUANTUM
        | crate::model::BACKBONE_LOAD_QUANTUM
        | crate::adapters::ADAPTER_LOAD_QUANTUM
        | mlx_gen::adapters::loader::ADAPTER_LOAD_QUANTUM
        | crate::adapters::ADAPTER_HOST_QUANTUM
        | crate::adapters::ADAPTER_WIDTH_QUANTUM
        | crate::adapters::ADAPTER_MATERIALIZE_QUANTUM
        | depth_q::LOAD
        | restore_q::LOAD => IosPhase::Load,
        te_q::TOWER
        | te_q::EMBEDDING
        | te_q::LAYER
        | te_q::DECODER_LAYER
        | te_q::CONDITIONING
        | te_q::LAYER_READ
        | te_q::STREAMED_LAYER => IosPhase::Conditioning,
        depth_q::INPUT
        | depth_q::HOST_PREP
        | depth_q::RGB
        | restore_q::PLAN
        | restore_q::HOST_PLANES => IosPhase::Encode,
        dit_q::EMBED
        | dit_q::TEXT_LAYER_BLOCK
        | dit_q::TEXT_POOL
        | dit_q::TEXT_MASK
        | dit_q::TEXT_REFINER_BLOCK
        | dit_q::TEXT_OUTPUT
        | dit_q::TRUNK_POSITIONS
        | dit_q::TRUNK_BLOCK
        | dit_q::PIXEL_EMBED
        | dit_q::PIXEL_POSITIONS
        | dit_q::PIXEL_BLOCK
        | dit_q::OUTPUT
        | dit_q::BLOCK_READ
        | dit_q::OUTPUT_ROWS
        | crate::solver::PLAN_QUANTUM
        | crate::solver::STEP_QUANTUM
        | crate::solver::FINALIZE_QUANTUM
        | restore_q::TILE => IosPhase::Denoise,
        crate::model::IMAGE_QUANTUM
        | depth_q::REDUCER
        | depth_q::HOST_MAP
        | depth_q::HOST_POST
        | restore_q::POST_TILE => IosPhase::Decode,
        _ => return None,
    };
    Some(phase)
}

/// Every quantum label the crate checks (the coverage set of [`ios_quantum_phase`]).
fn quantum_labels_iter() -> impl Iterator<Item = &'static str> {
    [
        crate::stream::EXPORT_TARGET_QUANTUM,
        crate::nn::PENDING_CONSTRUCTION_QUANTUM,
        crate::nn::ADAPTED_BASE_EVAL_QUANTUM,
        crate::stream::LOAD_TARGET_QUANTUM,
        crate::model::BACKBONE_LOAD_QUANTUM,
        crate::model::IMAGE_QUANTUM,
        crate::adapters::ADAPTER_LOAD_QUANTUM,
        mlx_gen::adapters::loader::ADAPTER_LOAD_QUANTUM,
        crate::adapters::ADAPTER_HOST_QUANTUM,
        crate::adapters::ADAPTER_WIDTH_QUANTUM,
        crate::adapters::ADAPTER_MATERIALIZE_QUANTUM,
        crate::solver::PLAN_QUANTUM,
        crate::solver::STEP_QUANTUM,
        crate::solver::FINALIZE_QUANTUM,
    ]
    .into_iter()
    .chain(te_q::ALL.iter().copied())
    .chain(dit_q::ALL.iter().copied())
    .chain(crate::depth::quanta::ALL.iter().copied())
    .chain(crate::restoration::quanta::ALL.iter().copied())
}

pub fn quantum_labels() -> Vec<&'static str> {
    let mut labels: Vec<_> = quantum_labels_iter().collect();
    labels.sort_unstable();
    labels.dedup();
    labels
}

/// Price one Iris request: see the module docs.
pub fn ios_pricing(req: &IosPricingRequest<'_>) -> Result<IosPricing> {
    Ok(pricing_facts(req)?.pricing)
}

pub fn pricing_facts(req: &IosPricingRequest<'_>) -> Result<PricingFacts> {
    Ok(price(req, Pricing::default())?.into_facts())
}

pub fn pricing_facts_with_token(
    req: &IosPricingRequest<'_>,
    token: &SamplingToken,
) -> Result<PricingFacts> {
    req.validate_work_context()?;
    if req.provider != crate::model::MODEL_ID {
        return Err(Error::Unsupported(
            "Iris sampling token is only for iris_3b generation".into(),
        ));
    }
    expect_mode(req.mode, &MemoryMode::TextToImage)?;
    Ok(generation(req, Pricing::default(), Some(token))?.into_facts())
}

/// Reprice an anchor with its OWN resolved sampling metadata, keeping actual artifact architecture.
/// Production default pricing never changes its checkpoint config; only this metadata replay does.
pub fn ios_pricing_with_token(
    req: &IosPricingRequest<'_>,
    token: &SamplingToken,
) -> Result<IosPricing> {
    Ok(pricing_facts_with_token(req, token)?.pricing)
}

/// Explicit anchor-metadata replay: no fake RGB request is constructed. Actual work uses
/// `pricing_facts` with its real matching Some context; replay only reconstructs counted profiles.
pub fn pricing_facts_with_restoration_token(
    req: &IosPricingRequest<'_>,
    token: &RestorationToken,
) -> Result<PricingFacts> {
    req.validate_work_context()?;
    if req.provider != crate::restoration::MODEL_ID {
        return Err(Error::Unsupported("not an Iris restoration replay".into()));
    }
    expect_mode(
        req.mode,
        &crate::memory_strategy::memory_mode(IrisTask::Restoration),
    )?;
    let resources = RestorationResources::from_spec(req.spec, req.provider).map_err(Error::from)?;
    Ok(restoration(req, Pricing::default(), &resources, Some(token))?.into_facts())
}

pub fn pricing_facts_with_depth_token(
    req: &IosPricingRequest<'_>,
    token: &DepthToken,
) -> Result<PricingFacts> {
    req.validate_work_context()?;
    if req.provider != crate::depth::DEPTH_MODEL_ID {
        return Err(Error::Unsupported("not an Iris depth replay".into()));
    }
    expect_mode(
        req.mode,
        &crate::memory_strategy::memory_mode(IrisTask::Depth),
    )?;
    let export =
        TaskExport::from_spec(req.spec, IrisTask::Depth, req.provider).map_err(Error::from)?;
    Ok(depth(req, Pricing::default(), &export, Some(token))?.into_facts())
}

/// How [`price`] treats the artifacts and the levers (the prediction's knobs; the public pricing
/// is [`Pricing::default`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Pricing {
    /// Price a dense artifact asked for at `q8`/`q4` as the packed export of that tier
    /// (`crate::packed`) — what an iOS load reads — instead of the dense file.
    pub assume_packed: bool,
    /// Rung 3 runs the PiT stage per batch row. `false` prices the batched stage under rung 3 —
    /// the route before that lever existed (the prediction's "before" column).
    pub pixel_rows: bool,
}

impl Default for Pricing {
    fn default() -> Self {
        Self {
            assume_packed: false,
            pixel_rows: true,
        }
    }
}

pub(crate) fn price(req: &IosPricingRequest<'_>, opts: Pricing) -> Result<Priced> {
    req.validate_work_context()?;
    let spec = req.spec;
    match req.provider {
        crate::model::MODEL_ID => {
            expect_mode(req.mode, &MemoryMode::TextToImage)?;
            generation(req, opts, None)
        }
        crate::restoration::MODEL_ID => {
            expect_mode(
                req.mode,
                &crate::memory_strategy::memory_mode(IrisTask::Restoration),
            )?;
            let resources = RestorationResources::from_spec(spec, crate::restoration::MODEL_ID)
                .map_err(Error::from)?;
            restoration(req, opts, &resources, None)
        }
        crate::depth::DEPTH_MODEL_ID => {
            expect_mode(
                req.mode,
                &crate::memory_strategy::memory_mode(IrisTask::Depth),
            )?;
            let export = TaskExport::from_spec(spec, IrisTask::Depth, crate::depth::DEPTH_MODEL_ID)
                .map_err(Error::from)?;
            depth(req, opts, &export, None)
        }
        other => Err(Error::Unsupported(format!(
            "{other}: not an Iris-3B MLX memory provider"
        ))),
    }
}

fn expect_mode(got: &MemoryMode, want: &MemoryMode) -> Result<()> {
    if got != want {
        return Err(Error::Unsupported(format!(
            "iris pricing: the route's request surface is {:?}, got {:?}",
            want.as_key(),
            got.as_key()
        )));
    }
    Ok(())
}

fn compute_bytes(precision: Precision) -> u64 {
    match precision {
        Precision::Bf16 => 2,
        Precision::Fp32 => 4,
    }
}

fn compute_dtype(precision: Precision) -> mlx_rs::Dtype {
    match precision {
        Precision::Bf16 => mlx_rs::Dtype::Bfloat16,
        Precision::Fp32 => mlx_rs::Dtype::Float32,
    }
}

fn elements(h: &Header) -> u64 {
    h.shape.iter().map(|&d| d as u64).product()
}

fn header(name: String, dtype: Stored, shape: Vec<usize>, width: u64) -> Header {
    let data_bytes = shape.iter().map(|&d| d as u64).product::<u64>() * width;
    Header {
        name,
        dtype,
        shape,
        data_bytes,
    }
}

/// The headers of a backbone checkpoint as the packed export of `quant` stores them
/// (`crate::packed`): every projection [`ProjectionLayout::of`] packs as its u32 codes over the
/// padded input, bf16 scales and biases, and an `in_features` marker when padded; every other
/// matrix at bf16; vectors and the text position table as released.
fn emulate_packed_backbone(headers: &[Header], prefix: &str, quant: Quant) -> Result<Vec<Header>> {
    let bq = BackboneQuant::for_load(Some(quant))?;
    let mut out = Vec::with_capacity(headers.len());
    for h in headers {
        let Some(name) = h.name.strip_prefix(prefix) else {
            // The converter copies out-of-backbone tensors verbatim (depth's reducer).
            out.push(h.clone());
            continue;
        };
        match h.shape.as_slice() {
            [rows, input] if name.ends_with(".weight") => match ProjectionLayout::of(*input, bq) {
                ProjectionLayout::Quantized { bits, padded_in } => {
                    let base = &h.name[..h.name.len() - ".weight".len()];
                    let words = padded_in * bits as usize / 32;
                    out.push(header(h.name.clone(), Stored::U32, vec![*rows, words], 4));
                    for part in ["scales", "biases"] {
                        out.push(header(
                            format!("{base}.{part}"),
                            Stored::BF16,
                            vec![*rows, padded_in / QUANT_GROUP_SIZE],
                            2,
                        ));
                    }
                    if padded_in != *input {
                        out.push(header(
                            format!("{base}{}", crate::packed::IN_FEATURES_SUFFIX),
                            Stored::I32,
                            Vec::new(),
                            4,
                        ));
                    }
                }
                ProjectionLayout::Dense => {
                    out.push(header(h.name.clone(), Stored::BF16, h.shape.clone(), 2))
                }
            },
            shape if shape.len() >= 2 && !name.ends_with("y_pos_embedding") => {
                out.push(header(h.name.clone(), Stored::BF16, h.shape.clone(), 2))
            }
            _ => out.push(h.clone()),
        }
    }
    Ok(out)
}

/// The headers of the Qwen3-VL tower as its packed export stores them: every attention / MLP
/// projection packed at the tower's group, the embedding and every other float tensor at bf16.
fn emulate_packed_tower(
    headers: &[Header],
    cfg: &mlx_llm::ModelConfig,
    quant: Quant,
) -> Result<Vec<Header>> {
    let spec = crate::text_encoder::text_encoder_quant_spec(cfg, Some(quant))?
        .ok_or_else(|| Error::Msg("iris pricing: no tower quantization".into()))?;
    let embedding = format!("{LANGUAGE_TOWER_PREFIX}embed_tokens.weight");
    let mut out = Vec::with_capacity(headers.len());
    for h in headers {
        match h.shape.as_slice() {
            [rows, input] if h.name.ends_with(".weight") && h.name != embedding => {
                let base = &h.name[..h.name.len() - ".weight".len()];
                let group = spec.group_size as usize;
                out.push(header(
                    h.name.clone(),
                    Stored::U32,
                    vec![*rows, input * spec.bits as usize / 32],
                    4,
                ));
                for part in ["scales", "biases"] {
                    out.push(header(
                        format!("{base}.{part}"),
                        Stored::BF16,
                        vec![*rows, input / group],
                        2,
                    ));
                }
            }
            _ if matches!(h.dtype, Stored::F32 | Stored::F16) => {
                out.push(header(h.name.clone(), Stored::BF16, h.shape.clone(), 2))
            }
            _ => out.push(h.clone()),
        }
    }
    Ok(out)
}

/// The empty-prompt states as the encoder-free loaders hold them: the `embeddings` tensor decoded
/// to f32, one allocation (the mask is read to the host).
fn held_states(path: &Path) -> Result<u64> {
    let headers = crate::stream::headers(path)?;
    let embeddings = headers
        .iter()
        .find(|h| h.name == "embeddings")
        .ok_or_else(|| Error::MissingTensor(format!("{}: embeddings", path.display())))?;
    crate::stream::empty_prompt_upload_bytes(embeddings.element_count().map_err(Error::from)? as u64)
}

/// Model/stream loaders materialize EACH retained safetensors tensor separately. The file's
/// aggregate payload length is not its allocator charge: each request above PAGE_BYTES rounds
/// independently (retention::allocated / the native macOS allocator). Apply that same source law
/// before summing, without saturation. The keep predicate is the actual eager/skeleton selection;
/// untouched streamed block payloads must not enter the skeleton's load floor.
fn materialized_load_bytes(headers: &[Header], keep: impl Fn(&str) -> bool) -> Result<u64> {
    crate::stream::materialized_load_bytes(headers, keep)
}

/// `(image_and_text_share_parent, text_retains_joined_parent)` after `completed` trunk blocks.
/// Mirrors `load_block`'s text_out predicate and `single`'s final image-only return. A dropped
/// final text update can retain the PENULTIMATE joined parent via y while x owns a new buffer.
fn trunk_parent_layout(model: &IrisModel, completed: usize) -> Result<(bool, bool)> {
    if completed > model.depth {
        return Err(Error::Unsupported(
            "Iris completed trunk index outside actual stack".into(),
        ));
    }
    let joined = completed > model.dual_depth
        && (completed < model.depth || model.final_block_text == "keep");
    let previous_joined = completed == model.depth
        && model.final_block_text != "keep"
        && completed
            .checked_sub(1)
            .is_some_and(|previous| previous > model.dual_depth);
    Ok((joined, joined || previous_joined))
}

/// A bounded group of distinct native backing requests at one source entry. `count` is never
/// a view count: row views retain their original full parent once. These source requests are
/// payload accounting, not fresh-allocation or physical-footprint evidence; reuse remains
/// unproved until an actual matching allocator witness establishes it.
struct NativeHeldRequest {
    owner: &'static str,
    bytes: u64,
    count: u64,
    birth: &'static str,
    block: Option<usize>,
    first_row: u64,
    liveness: &'static str,
    reuse: &'static str,
}

impl NativeHeldRequest {
    fn priced(&self, page_bytes: u64) -> Result<u64> {
        mlx_gen::ios_quantum::fresh_metal_allocation_bytes(self.bytes, page_bytes)
            .and_then(|one| one.checked_mul(self.count))
            .ok_or_else(|| Error::Unsupported(format!(
                "Iris backing arithmetic overflow: {} (birth {}, block {:?}, first row {}, live {}, reuse {})",
                self.owner, self.birth, self.block, self.first_row, self.liveness, self.reuse
            )))
    }
}

/// One backbone checkpoint's facts, from its headers (emulated as packed where the artifact will
/// be read packed): the module widths every runtime scratch formula reads, and the per-stack
/// window reads.
struct Backbone {
    cfg: IrisModel,
    layers: u64,
    cw: u64,
    input_channels: u64,
    prefix: &'static str,
    /// The headers the loaded module is built from (packed where the load is packed).
    headers: std::collections::HashMap<String, Header>,
    inventory: StackInventory,
    adapter_facts: crate::adapters::ProjectionAdapterFacts,
}

impl Backbone {
    fn new(
        config: &IrisConfig,
        headers: &[Header],
        prefix: &'static str,
        precision: Precision,
        input_channels: u64,
    ) -> Result<Self> {
        let inventory = StackInventory::from_headers(
            headers,
            prefix,
            &config.model,
            CastPolicy::Backbone {
                compute: compute_dtype(precision),
            },
        )?;
        Ok(Self {
            cfg: config.model.clone(),
            layers: config.text_encoder.hidden_layers.len() as u64,
            cw: compute_bytes(precision),
            input_channels,
            prefix,
            headers: headers
                .iter()
                .map(|h| {
                    let name = h.name.strip_prefix(prefix).unwrap_or(&h.name).to_owned();
                    (name, h.clone())
                })
                .collect(),
            inventory,
            adapter_facts: Default::default(),
        })
    }

    fn get(&self, name: &str) -> Result<&Header> {
        self.headers
            .get(name)
            .ok_or_else(|| Error::MissingTensor(format!("{}{name}", self.prefix)))
    }

    /// Output rows of projection `p`.
    fn rows(&self, p: &str) -> Result<u64> {
        Ok(self.get(&format!("{p}.weight"))?.shape[0] as u64)
    }

    /// Logical input width of projection `p` (a packed one's padded input; no Iris projection
    /// but the SwiGLU down projections is padded, and those are priced through [`Self::pad`]).
    fn input(&self, p: &str) -> Result<u64> {
        match self.headers.get(&format!("{p}.scales")) {
            Some(scales) => Ok(scales.shape[1] as u64 * QUANT_GROUP_SIZE as u64),
            None => Ok(self.get(&format!("{p}.weight"))?.shape[1] as u64),
        }
    }

    /// Zero columns a packed projection `p` with logical input `input` is padded by.
    fn pad(&self, p: &str, input: u64) -> Result<u64> {
        Ok(match self.headers.get(&format!("{p}.scales")) {
            Some(scales) => {
                (scales.shape[1] as u64 * QUANT_GROUP_SIZE as u64).saturating_sub(input)
            }
            None => 0,
        })
    }

    fn norm(&self, n: &str) -> Result<u64> {
        Ok(self.get(&format!("{n}.weight"))?.shape[0] as u64)
    }

    fn trunk(&self, i: usize, g: TrunkGeometry) -> Result<u64> {
        let p = |name: &str| format!("blocks.{i}.{name}");
        let text_out = self.cfg.final_block_text == "keep" || i + 1 < self.cfg.depth;
        let unfused = crate::retention::unfused();
        if i < self.cfg.dual_depth {
            let ff_x = self.rows(&p("mlp_x.w1"))?;
            let pad_x = self.pad(&p("mlp_x.w2"), ff_x)?;
            let (ff, pad) = if text_out {
                let ff_y = self.rows(&p("mlp_y.w1"))?;
                (ff_x.max(ff_y), pad_x.max(self.pad(&p("mlp_y.w2"), ff_y)?))
            } else {
                (ff_x, pad_x)
            };
            let g = TrunkGeometry {
                dim: self.norm(&p("norm_x1"))?,
                ff,
                text: text_out,
                down_pad: pad,
                ..g
            };
            Ok(IrisDiT::dual_block_bytes_with_adapters(
                &g,
                unfused,
                &self.adapter_facts,
                &format!("blocks.{i}"),
            ))
        } else {
            let ff = self.rows(&p("mlp.w1"))?;
            let g = TrunkGeometry {
                dim: self.norm(&p("norm1"))?,
                ff,
                text: text_out,
                down_pad: self.pad(&p("mlp.w2"), ff)?,
                ..g
            };
            Ok(IrisDiT::single_block_bytes_with_adapters(
                &g,
                unfused,
                &self.adapter_facts,
                &format!("blocks.{i}"),
            ))
        }
    }

    /// Every quantum one forward over `[b, C, h, w]` checks under `plan`, by label; and its
    /// largest live carry between quanta.
    fn forward(
        &self,
        b: u64,
        h: u64,
        w: u64,
        plan: mlx_gen::AttentionPlan<'_>,
        by_row: bool,
        out: &mut Vec<(&'static str, u64)>,
    ) -> Result<u64> {
        let m = &self.cfg;
        let cw = self.cw;
        let p = m.patch_size as u64;
        if p == 0 || h == 0 || w == 0 || !h.is_multiple_of(p) || !w.is_multiple_of(p) {
            return Err(Error::Unsupported(
                "Iris invocation is not on its actual patch grid".into(),
            ));
        }
        let n = (h / p)
            .checked_mul(w / p)
            .ok_or_else(|| Error::Unsupported("Iris patch count overflow".into()))?;
        let t = m.text_len as u64;
        let hidden = m.hidden_size as u64;
        let heads = m.num_heads as u64;
        let patch_batch = b
            .checked_mul(n)
            .ok_or_else(|| Error::Unsupported("Iris patch batch overflow".into()))?;
        if [
            b,
            h,
            w,
            n,
            t,
            hidden,
            heads,
            patch_batch,
            n.saturating_add(t),
        ]
        .iter()
        .any(|&extent| extent > i32::MAX as u64)
        {
            return Err(Error::Unsupported(
                "Iris tensor extent does not fit the backend i32 shape ABI".into(),
            ));
        }
        for dims in [
            [b, h, w, m.pixel.hidden_size as u64, 4],
            [b, t, self.layers, m.text_dim as u64, 4],
        ] {
            dims.into_iter()
                .try_fold(1u64, |count, dim| count.checked_mul(dim))
                .ok_or_else(|| Error::Unsupported("Iris invocation payload overflow".into()))?;
        }
        let kv_heads = m.num_kv_heads.unwrap_or(m.num_heads) as u64;
        let unfused = crate::retention::unfused();
        let rows = |plan, batch: u64, heads: u64, tokens: u64| {
            IrisDiT::query_rows(plan, batch as i32, heads as i32, tokens as i32)
        };
        let cores = self.rows("modulation_cores.adaln_img")?
            + if m.dual_depth > 0 {
                self.rows("modulation_cores.adaln_txt")?
            } else {
                0
            };
        out.push((
            dit_q::EMBED,
            IrisDiT::embed_bytes_with_adapters(
                EmbedGeometry {
                    compute_bytes: cw,
                    batch: b,
                    patches: n,
                    patch_in: self.input("s_embedder.proj")?,
                    hidden: self.rows("s_embedder.proj")?,
                    sinusoid: self.input("t_embedder.mlp.0")?,
                    t_hidden: self.rows("t_embedder.mlp.0")?,
                    t_out: self.rows("t_embedder.mlp.2")?,
                    cores,
                },
                &self.adapter_facts,
            )
            .saturating_add(sampler_step_bytes(b * self.input_channels * h * w)),
        ));
        for i in 0..Stack::Layer.depth(m) {
            let s = |name: &str| format!("y_embedder.layer_blocks.{i}.{name}");
            let heads = m.text_lap_num_heads as u64;
            let ff = self.rows(&s("mlp.0"))?;
            out.push((
                dit_q::TEXT_LAYER_BLOCK,
                IrisDiT::text_block_bytes_with_adapters(
                    TextBlockGeometry {
                        compute_bytes: cw,
                        batch: b * t,
                        tokens: self.layers,
                        dim: self.norm(&s("norm1"))?,
                        heads,
                        ff,
                        query_rows: rows(plan, b * t, heads, self.layers),
                        eval_per_chunk: plan.budget.eval_per_chunk(),
                        refiner: false,
                        down_pad: self.pad(&s("mlp.2"), ff)?,
                        stream_bytes: 4,
                    },
                    unfused,
                    &self.adapter_facts,
                    &format!("y_embedder.layer_blocks.{i}"),
                ),
            ));
        }
        out.push((
            dit_q::TEXT_POOL,
            IrisDiT::text_pool_bytes_with_adapters(
                cw,
                b,
                t,
                self.layers,
                m.text_dim as u64,
                self.rows("y_embedder.refiner.proj")?,
                &self.adapter_facts,
            ),
        ));
        out.push((dit_q::TEXT_MASK, IrisDiT::text_mask_bytes(b, t)));
        for i in 0..Stack::Refiner.depth(m) {
            let s = |name: &str| format!("y_embedder.refiner.blocks.{i}.{name}");
            let ff = self.rows(&s("mlp.w1"))?;
            out.push((
                dit_q::TEXT_REFINER_BLOCK,
                IrisDiT::text_block_bytes_with_adapters(
                    TextBlockGeometry {
                        compute_bytes: cw,
                        batch: b,
                        tokens: t,
                        dim: self.norm(&s("norm1"))?,
                        heads,
                        ff,
                        query_rows: rows(plan, b, heads, t),
                        eval_per_chunk: plan.budget.eval_per_chunk(),
                        refiner: true,
                        down_pad: self.pad(&s("mlp.w2"), ff)?,
                        stream_bytes: cw,
                    },
                    unfused,
                    &self.adapter_facts,
                    &format!("y_embedder.refiner.blocks.{i}"),
                ),
            ));
        }
        out.push((
            dit_q::TEXT_OUTPUT,
            IrisDiT::text_output_bytes(cw, b, t, self.norm("y_embedder.refiner.norm")?, unfused),
        ));
        let trunk = TrunkGeometry {
            compute_bytes: cw,
            batch: b,
            img_tokens: n,
            txt_tokens: t,
            dim: hidden,
            heads,
            kv_heads,
            ff: 0,
            query_rows: rows(plan, b, heads, n + t),
            eval_per_chunk: plan.budget.eval_per_chunk(),
            split_qkv: kv_heads != heads,
            text: true,
            down_pad: 0,
            img_stream_bytes: 4,
        };
        out.push((
            dit_q::TRUNK_POSITIONS,
            IrisDiT::trunk_entry_bytes(b, n, t, hidden / heads.max(1), hidden),
        ));
        for i in 0..m.depth {
            let g = TrunkGeometry {
                img_stream_bytes: if i == 0 { cw } else { 4 },
                ..trunk
            };
            out.push((dit_q::TRUNK_BLOCK, self.trunk(i, g)?));
        }
        out.push((
            dit_q::PIXEL_EMBED,
            IrisDiT::pixel_embed_bytes_with_adapters(
                cw,
                b,
                h,
                w,
                p,
                self.input("modulation_cores.adaln_img")?,
                self.rows("pixel_embedder.proj")?,
                self.input("pixel_embedder.proj")?,
                &self.adapter_facts,
            ),
        ));
        out.push((
            dit_q::PIXEL_POSITIONS,
            mlx_gen::ios_quantum::rope_table_bytes(
                n,
                (m.pixel.attn_hidden_size / m.pixel.num_heads.max(1)) as u64,
            ),
        ));
        let row_b = if by_row { 1 } else { b };
        for i in 0..m.pixel.depth {
            let s = |name: &str| format!("pixel_blocks.{i}.{name}");
            let heads = m.pixel.num_heads as u64;
            out.push((
                dit_q::PIXEL_BLOCK,
                IrisDiT::pixel_block_bytes_with_adapters(
                    PixelBlockGeometry {
                        compute_bytes: cw,
                        batch: row_b,
                        patches: n,
                        patch_pixels: p * p,
                        pixel_width: self.norm(&s("norm1"))?,
                        compact_width: self.rows(&s("compress"))?,
                        modulation_width: self.rows(&s("adaln"))?,
                        condition_width: self.input(&s("adaln"))?,
                        fc1_width: self.rows(&s("mlp.fc1"))?,
                        heads,
                        query_rows: rows(plan, row_b, heads, n),
                        eval_per_chunk: plan.budget.eval_per_chunk(),
                    },
                    unfused,
                    &self.adapter_facts,
                    &format!("pixel_blocks.{i}"),
                ),
            ));
        }
        out.push((
            dit_q::OUTPUT,
            IrisDiT::output_bytes_with_adapters(
                cw,
                row_b,
                n,
                p * p,
                self.input("final_layer.linear")?,
                self.rows("final_layer.linear")?,
                unfused,
                &self.adapter_facts,
            ),
        ));
        if by_row {
            out.push((
                dit_q::OUTPUT_ROWS,
                IrisDiT::output_rows_bytes(b as i32, m.in_channels as i32, h as i32, w as i32),
            ));
        }
        // The largest carry between quanta. Held through the whole forward (`forward_impl`'s
        // locals): the timestep embedding and both modulation cores, and the trunk's image and text
        // RoPE tables (cos and sin, `[positions, head_dim / 2]` f32). The trunk adds its image and
        // text streams. The PiT stage adds the trunk's text stream (kept to the end), the pixel
        // RoPE tables, the per-patch conditioning and the pixel stream — run per row, the first
        // block's batched input stays allocated until its last row is replaced, beside the `b − 1`
        // rows already produced — and the finished rows.
        let rope = |positions: u64, head_dim: u64| 2 * allocated(4 * positions * (head_dim / 2));
        let head_dim = hidden / heads.max(1);
        let t_out = self.rows("t_embedder.mlp.2")?;
        let held = allocated(4 * b * t_out)
            + allocated(4 * b * self.rows("modulation_cores.adaln_img")?)
            + if m.dual_depth > 0 {
                allocated(4 * b * self.rows("modulation_cores.adaln_txt")?)
            } else {
                0
            }
            + rope(n, head_dim)
            + rope(t, head_dim);
        let text_stream = allocated(4 * b * t * hidden);
        let trunk_carry = held + allocated(4 * b * n * hidden) + text_stream;
        let pixel_width = m.pixel.hidden_size as u64;
        let pixel_head = (m.pixel.attn_hidden_size / m.pixel.num_heads.max(1)) as u64;
        let pixel_row = allocated(cw * n * p * p * pixel_width);
        let pixel_carry = held
            + text_stream
            + rope(n, pixel_head)
            + allocated(cw * b * n * p * p * pixel_width)
            + if by_row { (b - 1) * pixel_row } else { 0 }
            + allocated(4 * b * n * hidden)
            + allocated(4 * b * m.in_channels as u64 * h * w);
        Ok(trunk_carry.max(pixel_carry))
    }

    fn weight_window_at(&self, label: &str, index: usize, window: Option<usize>) -> Result<u64> {
        let stack = match label {
            dit_q::TEXT_LAYER_BLOCK => Stack::Layer,
            dit_q::TEXT_REFINER_BLOCK => Stack::Refiner,
            dit_q::TRUNK_BLOCK => Stack::Trunk,
            dit_q::PIXEL_BLOCK => Stack::Pixel,
            _ => return Ok(0),
        };
        let reads = self.inventory.reads(stack);
        if reads.is_empty() || index >= reads.len() {
            return Err(Error::Unsupported(
                "Iris source block index outside actual stack".into(),
            ));
        }
        let count = window.unwrap_or(reads.len()).clamp(1, reads.len());
        let first = index / count * count;
        Ok(crate::stream::window_held_bytes(
            &reads[first..(first + count).min(reads.len())],
        ))
    }

    /// Named native owners at the actual operator entry, excluding caller solver/conditioning
    /// owners, cached position table and this operation's scratch. No independent max carry is
    /// re-used as a coincident lower bound. `row` is the actual per-row PiT/output iteration.
    #[allow(clippy::too_many_arguments)]
    fn native_held_requests_at(
        &self,
        b: u64,
        h: u64,
        w: u64,
        by_row: bool,
        label: &str,
        index: usize,
        row: u64,
    ) -> Result<Vec<NativeHeldRequest>> {
        let m = &self.cfg;
        let p = m.patch_size as u64;
        if p == 0 || m.num_heads == 0 || m.pixel.num_heads == 0 || b == 0 || row >= b {
            return Err(Error::Unsupported(
                "invalid Iris source backing geometry/row".into(),
            ));
        }
        let bytes = |dimensions: &[u64]| {
            dimensions
                .iter()
                .try_fold(1u64, |size, d| size.checked_mul(*d))
                .ok_or_else(|| Error::Unsupported("Iris backing request product overflow".into()))
        };
        let n = bytes(&[h / p, w / p])?;
        let t = m.text_len as u64;
        let hidden = m.hidden_size as u64;
        let cw = self.cw;
        if label == dit_q::EMBED {
            return Ok(Vec::new());
        }
        let mut owners = Vec::new();
        // At most a fixed number of groups. Multiplicity describes DISTINCT native requests,
        // never row views; reuse/donation has not been certified by these source records.
        let mut add = |owner, size, count, birth, block, first_row, liveness| {
            if count != 0 {
                owners.push(NativeHeldRequest {
                    owner,
                    bytes: size,
                    count,
                    birth,
                    block,
                    first_row,
                    liveness,
                    reuse: "unproved; no fresh-allocation certificate",
                });
            }
        };
        add(
            "t_emb",
            bytes(&[cw, b, self.rows("t_embedder.mlp.2")?])?,
            1,
            dit_q::EMBED,
            None,
            0,
            "forward lexical t_emb",
        );
        add(
            "core_img",
            bytes(&[cw, b, self.rows("modulation_cores.adaln_img")?])?,
            1,
            dit_q::EMBED,
            None,
            0,
            "forward lexical core_img",
        );
        if m.dual_depth > 0 {
            add(
                "core_txt",
                bytes(&[cw, b, self.rows("modulation_cores.adaln_txt")?])?,
                1,
                dit_q::EMBED,
                None,
                0,
                "forward lexical core_txt",
            );
        }
        let image_initial = bytes(&[cw, b, n, hidden])?;
        let image_f32 = bytes(&[4, b, n, hidden])?;
        let layer_y = bytes(&[4, b, t, self.layers, m.text_dim as u64])?;
        let text_cw = bytes(&[cw, b, t, hidden])?;
        let text_f32 = bytes(&[4, b, t, hidden])?;
        let mask = bytes(&[4, b, t, t])?;
        let head = hidden / m.num_heads as u64;
        let rope_image = bytes(&[4, n, head / 2])?;
        let rope_text = bytes(&[4, t, head / 2])?;
        let rope_pixel = bytes(&[
            4,
            n,
            (m.pixel.attn_hidden_size / m.pixel.num_heads) as u64 / 2,
        ])?;
        let conditioning = bytes(&[if m.depth == 0 { cw } else { 4 }, b, n, hidden])?;
        let pixels_full = bytes(&[cw, b, h, w, m.pixel.hidden_size as u64])?;
        let pixels_row = bytes(&[cw, h, w, m.pixel.hidden_size as u64])?;
        let rgb_row = bytes(&[4, m.in_channels as u64, h, w])?;
        let joined = bytes(&[
            4,
            b,
            n.checked_add(t)
                .ok_or_else(|| Error::Unsupported("Iris joined token count overflow".into()))?,
            hidden,
        ])?;
        let layer_depth = Stack::Layer.depth(m);
        let add_layer = |add: &mut dyn FnMut(
            &'static str,
            u64,
            u64,
            &'static str,
            Option<usize>,
            u64,
            &'static str,
        )| {
            if layer_depth != 0 {
                add(
                    "layer_y",
                    layer_y,
                    1,
                    dit_q::TEXT_LAYER_BLOCK,
                    Some(layer_depth - 1),
                    0,
                    "text_adapter_exec shadowed layer-stack output",
                );
            }
            // With zero layer blocks the initial reshape aliases caller-owned text.states.
        };
        match label {
            dit_q::TEXT_LAYER_BLOCK => {
                add(
                    "image_initial",
                    image_initial,
                    1,
                    dit_q::EMBED,
                    None,
                    0,
                    "forward s before trunk",
                );
                if index != 0 {
                    add(
                        "layer_y",
                        layer_y,
                        1,
                        dit_q::TEXT_LAYER_BLOCK,
                        Some(index - 1),
                        0,
                        "current layer-stack carry",
                    );
                }
            }
            dit_q::TEXT_POOL
            | dit_q::TEXT_MASK
            | dit_q::TEXT_REFINER_BLOCK
            | dit_q::TEXT_OUTPUT => {
                add(
                    "image_initial",
                    image_initial,
                    1,
                    dit_q::EMBED,
                    None,
                    0,
                    "forward s before trunk",
                );
                add_layer(&mut add);
                if label != dit_q::TEXT_POOL {
                    let birth = if label == dit_q::TEXT_REFINER_BLOCK && index != 0
                        || label == dit_q::TEXT_OUTPUT && Stack::Refiner.depth(m) != 0
                    {
                        dit_q::TEXT_REFINER_BLOCK
                    } else {
                        dit_q::TEXT_POOL
                    };
                    let block = if birth == dit_q::TEXT_REFINER_BLOCK {
                        Some(if label == dit_q::TEXT_OUTPUT {
                            Stack::Refiner.depth(m) - 1
                        } else {
                            index - 1
                        })
                    } else {
                        None
                    };
                    add(
                        "refiner_y",
                        text_cw,
                        1,
                        birth,
                        block,
                        0,
                        "current refiner input/output binding",
                    );
                }
                if matches!(label, dit_q::TEXT_REFINER_BLOCK | dit_q::TEXT_OUTPUT) {
                    add(
                        "key_mask",
                        mask,
                        1,
                        dit_q::TEXT_MASK,
                        None,
                        0,
                        "text_adapter_exec lexical key_mask",
                    );
                }
            }
            dit_q::TRUNK_POSITIONS => {
                add(
                    "image_initial",
                    image_initial,
                    1,
                    dit_q::EMBED,
                    None,
                    0,
                    "forward s before trunk",
                );
                add(
                    "text_base",
                    text_f32,
                    1,
                    dit_q::TEXT_OUTPUT,
                    None,
                    0,
                    "forward lexical text_base",
                );
            }
            dit_q::TRUNK_BLOCK
            | dit_q::PIXEL_EMBED
            | dit_q::PIXEL_POSITIONS
            | dit_q::PIXEL_BLOCK
            | dit_q::OUTPUT
            | dit_q::OUTPUT_ROWS => {
                add(
                    "image_rope_cos_sin",
                    rope_image,
                    2,
                    dit_q::TRUNK_POSITIONS,
                    None,
                    0,
                    "forward ctx owns two distinct arrays",
                );
                add(
                    "text_rope_cos_sin",
                    rope_text,
                    2,
                    dit_q::TRUNK_POSITIONS,
                    None,
                    0,
                    "forward ctx owns two distinct arrays",
                );
                add(
                    "text_base",
                    text_f32,
                    1,
                    dit_q::TEXT_OUTPUT,
                    None,
                    0,
                    "forward lexical text_base",
                );
                let completed = if label == dit_q::TRUNK_BLOCK {
                    index
                } else {
                    m.depth
                };
                let (shared, text_parent_joined) = trunk_parent_layout(m, completed)?;
                let last = completed.checked_sub(1);
                let text_birth_block = if text_parent_joined && !shared {
                    completed.checked_sub(2)
                } else {
                    last
                };
                add(
                    "trunk_text_parent",
                    if text_parent_joined { joined } else { text_f32 },
                    1,
                    if text_birth_block.is_some() {
                        dit_q::TRUNK_BLOCK
                    } else {
                        dit_q::TRUNK_POSITIONS
                    },
                    text_birth_block,
                    0,
                    "trunk y / final _y retains full parent, including joined image view",
                );
                if matches!(label, dit_q::TRUNK_BLOCK | dit_q::PIXEL_EMBED) && !shared {
                    add(
                        "trunk_image_parent",
                        if completed == 0 {
                            image_initial
                        } else {
                            image_f32
                        },
                        1,
                        if completed == 0 {
                            dit_q::EMBED
                        } else {
                            dit_q::TRUNK_BLOCK
                        },
                        last,
                        0,
                        "current s; explicitly dropped after pixel embedding",
                    );
                }
                if !matches!(label, dit_q::TRUNK_BLOCK | dit_q::PIXEL_EMBED) {
                    add(
                        "s_cond",
                        conditioning,
                        1,
                        dit_q::PIXEL_EMBED,
                        None,
                        0,
                        "conds views retain full s_cond parent",
                    );
                    if label != dit_q::PIXEL_POSITIONS {
                        add(
                            "pixel_rope_cos_sin",
                            rope_pixel,
                            2,
                            dit_q::PIXEL_POSITIONS,
                            None,
                            0,
                            "forward pctx owns two distinct arrays",
                        );
                    }
                    if by_row {
                        add(
                            "pixels_full_parent",
                            pixels_full,
                            1,
                            dit_q::PIXEL_EMBED,
                            None,
                            0,
                            "lexical pixels retained while split views / replaced rows run",
                        );
                        let separate = if label == dit_q::PIXEL_BLOCK {
                            if index == 0 {
                                row
                            } else {
                                b
                            }
                        } else if label == dit_q::OUTPUT_ROWS
                            || label == dit_q::PIXEL_POSITIONS
                            || m.pixel.depth == 0
                        {
                            0
                        } else {
                            b
                        };
                        // On a later block, processed rows were born in this block and pending
                        // rows in the previous one. They are different buffers, not parent views.
                        let current_rows = if label == dit_q::PIXEL_BLOCK {
                            row.min(separate)
                        } else {
                            0
                        };
                        add(
                            "pixel_rows_current",
                            pixels_row,
                            current_rows,
                            dit_q::PIXEL_BLOCK,
                            Some(index),
                            0,
                            "processed row outputs retained in parts",
                        );
                        add(
                            "pixel_rows_previous",
                            pixels_row,
                            separate - current_rows,
                            dit_q::PIXEL_BLOCK,
                            if label == dit_q::PIXEL_BLOCK {
                                index.checked_sub(1)
                            } else {
                                m.pixel.depth.checked_sub(1)
                            },
                            current_rows,
                            "pending/completed distinct row outputs retained in parts",
                        );
                        let outputs = if label == dit_q::OUTPUT {
                            row
                        } else if label == dit_q::OUTPUT_ROWS {
                            b
                        } else {
                            0
                        };
                        add("rgb_rows", rgb_row, outputs, dit_q::OUTPUT, None, 0,
                            "outs owns distinct completed RGB arrays; parts dropped before OUTPUT_ROWS");
                    } else {
                        let completed_pixel = if label == dit_q::PIXEL_POSITIONS {
                            0
                        } else if label == dit_q::PIXEL_BLOCK {
                            index
                        } else {
                            m.pixel.depth
                        };
                        add(
                            "pixels_current_parent",
                            pixels_full,
                            1,
                            if completed_pixel == 0 {
                                dit_q::PIXEL_EMBED
                            } else {
                                dit_q::PIXEL_BLOCK
                            },
                            completed_pixel.checked_sub(1),
                            0,
                            "one current full-batch parts backing; no extra original after move",
                        );
                    }
                }
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "unknown Iris ownership point {other}"
                )))
            }
        }
        Ok(owners)
    }

    #[allow(clippy::too_many_arguments)]
    fn held_at(
        &self,
        b: u64,
        h: u64,
        w: u64,
        by_row: bool,
        label: &str,
        index: usize,
        row: u64,
    ) -> Result<u64> {
        self.native_held_requests_at(b, h, w, by_row, label, index, row)?
            .iter()
            .try_fold(0u64, |sum, group| {
                sum.checked_add(group.priced(crate::retention::PAGE_BYTES)?)
                    .ok_or_else(|| Error::Unsupported("Iris held backing sum overflow".into()))
            })
    }

    /// The pixel embedder's `[h, w, pixel]` f32 position table: built by the first forward at a
    /// size and cached by the module (`IrisDiT::pixel_pos`), so resident from then on.
    fn pixel_table(&self, h: u64, w: u64) -> Result<u64> {
        Ok(allocated(4 * h * w * self.rows("pixel_embedder.proj")?))
    }
}

/// A backbone checkpoint priced for a load: its facts, its resident bytes and its load floor.
struct PricedBackbone {
    facts: Backbone,
    resident: u64,
    floor: u64,
    load_targets: u64,
    constructor_outputs: u64,
    evaluation_outputs: u64,
    width_outputs: u64,
    read_sources_upper: u64,
}

/// Price the backbone `path` (export key prefix `prefix`) for a load at `precision` / `requested`,
/// eager or `streamed`.
#[allow(clippy::too_many_arguments)]
fn priced_backbone(
    config: &IrisConfig,
    path: &Path,
    prefix: &'static str,
    precision: Precision,
    requested: Option<Quant>,
    streamed: bool,
    input_channels: u64,
    opts: Pricing,
) -> Result<PricedBackbone> {
    let real = crate::stream::headers(path)?;
    let artifact = crate::packed::artifact_tier(path)?;
    crate::packed::check_requested("the backbone", artifact, requested)?;
    priced_backbone_from_headers(
        BackbonePricingInput {
            config,
            prefix,
            precision,
            requested,
            streamed,
            input_channels,
            opts,
        },
        &real,
        artifact,
    )
}

struct BackbonePricingInput<'a> {
    config: &'a IrisConfig,
    prefix: &'static str,
    precision: Precision,
    requested: Option<Quant>,
    streamed: bool,
    input_channels: u64,
    opts: Pricing,
}

fn priced_backbone_from_headers(
    input: BackbonePricingInput<'_>,
    real: &[Header],
    artifact: Option<Quant>,
) -> Result<PricedBackbone> {
    let BackbonePricingInput {
        config,
        prefix,
        precision,
        requested,
        streamed,
        input_channels,
        opts,
    } = input;
    let requantized = artifact.is_none() && requested.is_some();
    if requantized && streamed && !opts.assume_packed {
        crate::stream::refuse_stream_requantization(
            crate::model::MODEL_ID,
            "the backbone",
            artifact,
            requested,
        )?;
    }
    let built = match (requantized, requested) {
        (true, Some(quant)) => emulate_packed_backbone(real, prefix, quant)?,
        _ => real.to_vec(),
    };
    let skeleton = |name: &str| {
        name.strip_prefix(prefix)
            .is_none_or(|n| Stack::of(n).is_none())
    };
    let keep = |name: &str| !streamed || skeleton(name);
    // The load floor is the file the load reads: the packed export when one is assumed.
    let floor_headers = if opts.assume_packed { &built[..] } else { real };
    let floor = materialized_load_bytes(floor_headers, keep)?;
    let load_view = floor_headers
        .iter()
        .map(|h| (h.name.strip_prefix(prefix).unwrap_or(&h.name), h))
        .collect();
    let load_constructors: std::collections::HashMap<_, _> = floor_headers
        .iter()
        .map(|h| {
            (
                h.name.strip_prefix(prefix).unwrap_or(&h.name).to_owned(),
                h.clone(),
            )
        })
        .collect();
    let constructor_outputs = if prefix == "pixel." {
        depth_constructor_output_bytes(&load_constructors, precision, streamed)?
    } else {
        backbone_constructor_output_bytes(&load_constructors, precision, |name| {
            !streamed || Stack::of(name).is_none()
        })?
    };
    let read_sources_upper =
        crate::stream::backbone_read_bytes(&load_view, compute_dtype(precision), |name| {
            !streamed || Stack::of(name).is_none()
        })?;
    let load_quant = if requantized && !opts.assume_packed {
        BackboneQuant::for_load(requested)?
    } else {
        None
    };
    let quantized = crate::stream::backbone_quantization_bytes(
        &load_view,
        compute_dtype(precision),
        load_quant,
        |name| !streamed || Stack::of(name).is_none(),
    )?;
    let eval_quantized = crate::stream::backbone_quantization_evaluation_bytes(
        &load_view,
        compute_dtype(precision),
        load_quant,
        |name| !streamed || Stack::of(name).is_none(),
    )?;
    let width_outputs = crate::stream::installed_width_output_bytes(
        &load_view,
        compute_dtype(precision),
        |name| !streamed || Stack::of(name).is_none(),
    )?;
    let evaluation_outputs = constructor_outputs
        .checked_add(eval_quantized)
        .ok_or_else(|| Error::Unsupported("Iris pending evaluation byte overflow".into()))?;
    let constructor_outputs = constructor_outputs.checked_add(quantized).ok_or_else(|| {
        Error::Unsupported("Iris pending quantized constructor byte overflow".into())
    })?;
    let load_targets = read_sources_upper
        .checked_add(constructor_outputs)
        .ok_or_else(|| Error::Unsupported("Iris joint load target byte overflow".into()))?;
    let quant = if requantized {
        BackboneQuant::for_load(requested)?
    } else {
        None
    };
    let facts = Backbone::new(config, &built, prefix, precision, input_channels)?;
    let resident =
        crate::memory_strategy::backbone_bytes_from_headers(real, precision, quant, keep)
            .map_err(Error::from)?;
    Ok(PricedBackbone {
        facts,
        resident,
        floor,
        load_targets,
        constructor_outputs,
        evaluation_outputs,
        width_outputs,
        read_sources_upper,
    })
}

/// The Qwen3-VL tower priced for a load.
struct PricedTower {
    cfg: mlx_llm::ModelConfig,
    resident: u64,
    floor: u64,
    load_owned_upper: u64,
    reads: Vec<BlockRead>,
}

fn priced_tower(
    dir: &Path,
    requested: Option<Quant>,
    streamed: bool,
    opts: Pricing,
) -> Result<PricedTower> {
    let cfg = mlx_llm::ModelConfig::from_dir(dir)
        .map_err(|e| Error::Msg(format!("iris pricing: the text encoder config: {e}")))?;
    let untied = !cfg.tie_word_embeddings;
    let files = crate::text_encoder::weight_shards(dir, untied)?;
    let mut real = Vec::new();
    for file in &files {
        real.extend(
            crate::stream::headers(file)?
                .into_iter()
                .filter(|h| crate::text_encoder::is_text_tower_tensor(&h.name, untied)),
        );
    }
    let packed = cfg.quantization.is_some();
    let artifact = if packed {
        crate::packed::text_encoder_tier(dir)?
    } else {
        None
    };
    crate::packed::check_requested("the text encoder", artifact, requested)?;
    let requantized = !packed && requested.is_some();
    if requantized && streamed && !opts.assume_packed {
        crate::stream::refuse_stream_requantization(
            crate::model::MODEL_ID,
            "the text encoder",
            None,
            requested,
        )?;
    }
    let built = match (requantized, requested) {
        (true, Some(quant)) => emulate_packed_tower(&real, &cfg, quant)?,
        _ => real.clone(),
    };
    let embed = format!("{LANGUAGE_TOWER_PREFIX}embed_tokens.weight");
    let norm = format!("{LANGUAGE_TOWER_PREFIX}norm.weight");
    let skeleton = |name: &str| name == embed || name == norm;
    let floor_headers = if opts.assume_packed { &built } else { &real };
    let resident = if streamed {
        built
            .iter()
            .filter(|h| skeleton(&h.name))
            .map(|h| allocated(elements(h) * 2))
            .sum()
    } else {
        crate::memory_strategy::text_encoder_bytes(dir, requested).map_err(Error::from)?
    };
    // Both tower forms own build_rope()'s f32 frequency row; it is not a checkpoint tensor
    // (and the shared decoder's param_groups does not enumerate it).
    let resident = resident
        .checked_add(tower_rope_bytes(&cfg)?)
        .ok_or_else(|| Error::Unsupported("Iris tower resident byte arithmetic overflow".into()))?;
    // Return the checked logical floor and its same-owner target/rounding upper together;
    // a saturating source sum must never be treated as an exact subtraction identity.
    let (floor, load_owned_upper) = tower_load_output_bytes(
        floor_headers,
        |name| !streamed || skeleton(name),
        tower_rope_bytes(&cfg)?,
    )?;
    let load_quant = if requantized && !opts.assume_packed {
        crate::text_encoder::text_encoder_quant_spec(&cfg, requested)?
    } else {
        None
    };
    let load_owned_upper = load_owned_upper
        .checked_add(crate::stream::tower_quantization_bytes(
            floor_headers,
            |name| !streamed || skeleton(name),
            load_quant,
        )?)
        .ok_or_else(|| {
            Error::Unsupported("Iris tower pending quantization byte overflow".into())
        })?;
    let mut grouped: Vec<Vec<&Header>> = vec![Vec::new(); cfg.num_layers];
    for h in &built {
        if let Some(i) = crate::text_encoder::tower_layer_of(&h.name) {
            if let Some(slot) = grouped.get_mut(i) {
                slot.push(h);
            }
        }
    }
    let reads = grouped
        .iter()
        .map(|layer| crate::stream::block_read(layer, CastPolicy::Tower))
        .collect();
    Ok(PricedTower {
        cfg,
        resident,
        floor,
        load_owned_upper,
        reads,
    })
}

/// Qwen3-VL's selected load leaves: fresh BF16 parameter targets, its model-owned rotary row,
/// and per-source allocator rounding beyond the logical-byte raw guard. Packed codes/scales/
/// biases are used as stored. This does not claim to price dense-requantization intermediates;
/// their separate cumulative native graph remains in native_ledger.
fn tower_load_output_bytes(
    headers: &[Header],
    keep: impl Fn(&str) -> bool,
    rotary_row: u64,
) -> Result<(u64, u64)> {
    crate::stream::tower_load_output_bytes(headers, keep, rotary_row)
}

pub use native_ledger::{
    host_analysis_without_facts, IrisHostAnalysis, IrisHostGap, IrisMetadataFacts,
    KnownHostFragments,
};

/// Guarded source facts are prepared once for one actual request OR its exact recorded replay.
/// This private carrier cannot be constructed by a catalog caller to bless unrelated bytes.
struct GuardedSources<'a> {
    task: GuardedTask,
    bb: PricedBackbone,
    raw_headers: std::collections::HashMap<String, Header>,
    empty_headers: Vec<Header>,
    metadata: IrisMetadataFacts,
    replay: Option<IrisReplay<'a>>,
    caller_bytes: u64,
}

enum GuardedTask {
    Restoration {
        resources: RestorationResources,
        plan: gen_core_iris::restoration::RestorationPlanMetadata,
    },
    Depth {
        export: TaskExport,
        plan: gen_core_iris::depth::DepthInputPlan,
    },
}

impl GuardedSources<'_> {
    fn check_restoration_token(
        &self,
        req: &IosPricingRequest<'_>,
        token: &RestorationToken,
    ) -> Result<()> {
        if req.provider != crate::restoration::MODEL_ID
            || !matches!(self.replay, Some(IrisReplay::Restoration(expected)) if expected == token)
        {
            return Err(Error::Unsupported(
                "Iris restoration diagnostic token/source mismatch".into(),
            ));
        }
        Ok(())
    }

    fn export(&self) -> &TaskExport {
        match &self.task {
            GuardedTask::Restoration { resources, .. } => &resources.export,
            GuardedTask::Depth { export, .. } => export,
        }
    }
}

struct GuardedPricingFacts<'a> {
    facts: PricingFacts,
    sources: GuardedSources<'a>,
}

fn guarded_source_read<T>(
    inventory: &mlx_gen::gen_core::SourceInventory,
    req: &IosPricingRequest<'_>,
    capacity: &mut dyn FnMut(u64) -> Result<()>,
    read: impl FnOnce(&mut dyn FnMut(u64) -> Result<()>) -> Result<T>,
) -> Result<T> {
    source_read_with_capacity(inventory, req.spec, capacity, read, |error| {
        native_ledger::source_guard_refusal(req, error).unwrap_or_else(std::convert::identity)
    })
}

fn source_read_with_capacity<T>(
    inventory: &mlx_gen::gen_core::SourceInventory,
    spec: &mlx_gen::LoadSpec,
    capacity: &mut dyn FnMut(u64) -> Result<()>,
    read: impl FnOnce(&mut dyn FnMut(u64) -> Result<()>) -> Result<T>,
    source_error: impl FnOnce(Error) -> Error,
) -> Result<T> {
    // Preserve the original provider callback error over the neutral core error boundary.
    // Once denied, no second callback invocation or subsequent guarded allocation is allowed.
    let mut original_error = None;
    let result = {
        let mut denied = false;
        let mut checked = |bytes| {
            if denied {
                return Err(mlx_gen::gen_core::Error::Unsupported(
                    "Iris source metadata capacity already refused".into(),
                ));
            }
            match capacity(bytes) {
                Ok(()) => Ok(()),
                Err(error) => {
                    denied = true;
                    original_error = Some(error);
                    Err(mlx_gen::gen_core::Error::Unsupported(
                        "Iris source metadata capacity refused".into(),
                    ))
                }
            }
        };
        (|| {
            inventory
                .verify_for_with_capacity(spec, &mut checked)
                .map_err(Error::from)?;
            inventory.read_unchanged_with_capacity(&mut checked, |next| {
                let mut provider_next = |bytes| next(bytes).map_err(Error::from);
                read(&mut provider_next)
            })
        })()
    };
    match original_error {
        Some(error) => Err(error),
        None => result.map_err(source_error),
    }
}

fn observed_metadata_read<T>(
    metadata: &mut IrisMetadataFacts,
    capacity: &mut dyn FnMut(u64) -> Result<()>,
    read: impl FnOnce(
        &mut mlx_gen::gen_core::weightsmeta::SourceMetadataObserver<'_>,
    ) -> mlx_gen::gen_core::Result<T>,
) -> Result<T> {
    let mut original_error = None;
    let mut observe = |event: mlx_gen::gen_core::weightsmeta::SourceMetadataRead<'_>| match metadata
        .observe(event, capacity)
    {
        Ok(()) => Ok(()),
        Err(error) => {
            original_error = Some(error);
            Err(mlx_gen::gen_core::Error::Unsupported(
                "Iris source metadata observer refused".into(),
            ))
        }
    };
    let result = read(&mut observe);
    match original_error {
        Some(error) => Err(error),
        None => result.map_err(Error::from),
    }
}

pub fn check_replay_source_capacity(
    token: &str,
    capacity: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<()> {
    native_ledger::IrisMetadataFacts::default().check_replay_token(token, capacity)
}

/// Preserve each existing token constructor's prefix-error order. A prefix-invalid token has
/// no parser work: its original constructor rejects before serde. This is NOT a certificate or
/// permission to parse arbitrary JSON. Generation is intentionally refused at aa011 first.
pub fn check_replay_source_capacity_for_provider(
    provider: &str,
    token: &str,
    capacity: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<()> {
    let prefix = replay_prefix(provider)?;
    if !token.starts_with(prefix) {
        return Ok(());
    }
    if provider == crate::model::MODEL_ID {
        return native_ledger::require_guarded_generation_gap();
    }
    check_replay_source_capacity(token, capacity)
}

fn replay_prefix(provider: &str) -> Result<&'static str> {
    Ok(match provider {
        crate::model::MODEL_ID => "iris-sampling-v1:",
        crate::restoration::MODEL_ID => "iris-restoration-v1:",
        crate::depth::DEPTH_MODEL_ID => "iris-depth-v1:",
        _ => {
            return Err(Error::Unsupported(
                "not an Iris guarded replay route".into(),
            ))
        }
    })
}

/// Provenance of a caller's explicitly empirical preparation envelope. This does not assert
/// a source-derived allocation bound or approve the engineering policy that consumes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourcePreparationEstimateProvenance {
    /// Measured whole-process growth above the actual BeforeRequestConstruction baseline,
    /// plus the engineering policy's margin. The capture must begin before preparation in a
    /// fresh process; an AfterModelLoad baseline or an unbound lifetime peak is insufficient.
    /// The caller verifies checkpoint/source, device, OS and execution identity before reuse.
    BeforeRequestConstructionPeakWithMargin,
}

/// An engineering estimate checked before compatibility metadata/tokenizer parsing.
/// This is deliberately distinct from the private complete-HOST source proof inputs.
/// Constructing it does not authorize a policy, certify its bytes, or alter default refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EstimatedSourcePreparation {
    bytes: u64,
    provenance: SourcePreparationEstimateProvenance,
}

impl EstimatedSourcePreparation {
    pub fn new(bytes: u64, provenance: SourcePreparationEstimateProvenance) -> Result<Self> {
        if bytes == 0 {
            return Err(Error::Unsupported(
                "Iris estimated source preparation needs a nonzero measured envelope".into(),
            ));
        }
        Ok(Self { bytes, provenance })
    }

    pub fn bytes(self) -> u64 {
        self.bytes
    }

    pub fn provenance(self) -> SourcePreparationEstimateProvenance {
        self.provenance
    }

    fn check(self, capacity: &mut dyn FnMut(u64) -> Result<()>) -> Result<()> {
        capacity(self.bytes)
    }
}

/// Empirical-policy counterpart of the source-only replay preflight. A wrong prefix still
/// reaches the token constructor's original error without parsing. A valid prefix checks the
/// measured preparation envelope before the existing JSON-layout check. Generation's default
/// source-certified route continues to refuse at aa011.
pub fn check_replay_with_estimated_preparation(
    provider: &str,
    token: &str,
    estimate: EstimatedSourcePreparation,
    capacity: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<()> {
    if !token.starts_with(replay_prefix(provider)?) {
        return Ok(());
    }
    estimate.check(capacity)?;
    check_replay_source_capacity(token, capacity)
}

/// Price under an explicitly selected empirical preparation policy. The caller must verify
/// the estimate's capture identity and policy approval. Check its whole preparation envelope
/// before inventory verification and again immediately before compatibility parsing; preserve
/// the retained inventory's before/after identity checks and capacity-check error precedence.
///
/// Compatibility readers are not per-allocation instrumented. This path checks an estimated
/// aggregate envelope, not a mathematical parser/tokenizer bound. Its returned HOST analysis
/// remains incomplete, and it never constructs a MetadataHostBound or TokenizerHostBound.
pub fn pricing_facts_with_estimated_preparation(
    req: &IosPricingRequest<'_>,
    replay: Option<IrisReplay<'_>>,
    estimate: EstimatedSourcePreparation,
    capacity: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<PricingFacts> {
    req.validate_work_context()?;
    let mut facts =
        with_estimated_source_preparation(req.spec, estimate, capacity, || match replay {
            None => pricing_facts(req),
            Some(IrisReplay::Generation(token)) => pricing_facts_with_token(req, token),
            Some(IrisReplay::Restoration(token)) => {
                pricing_facts_with_restoration_token(req, token)
            }
            Some(IrisReplay::Depth(token)) => pricing_facts_with_depth_token(req, token),
        })?;
    facts.host_analysis = Some(host_analysis_without_facts(req, None)?);
    Ok(facts)
}

/// Apply the same empirical preparation envelope and retained-source checks to an existing
/// metadata/key producer (for example token-window or canonical sampling-key preparation).
/// The closure must perform only source metadata work covered by the caller's measured envelope;
/// this is not a model-execution allowance or a source proof. The caller owns policy/identity
/// qualification exactly as for [`pricing_facts_with_estimated_preparation`].
fn with_estimated_source_preparation<T>(
    spec: &mlx_gen::LoadSpec,
    estimate: EstimatedSourcePreparation,
    capacity: &mut dyn FnMut(u64) -> Result<()>,
    read: impl FnOnce() -> Result<T>,
) -> Result<T> {
    estimate.check(capacity)?;
    let inventory = spec.source_inventory().ok_or_else(|| {
        Error::Unsupported("Iris estimated metadata requires retained SourceInventory".into())
    })?;
    source_read_with_capacity(
        inventory,
        spec,
        capacity,
        |next| {
            estimate.check(next)?;
            read()
        },
        std::convert::identity,
    )
}

/// Guarded source pricing for SDK/catalog callers that possess the actual metadata-capacity
/// callback. Both ordinary and token-replay routes use the same single-pass held-facts path.
pub fn pricing_facts_with_source_capacity(
    req: &IosPricingRequest<'_>,
    replay: Option<IrisReplay<'_>>,
    capacity: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<PricingFacts> {
    Ok(guarded_pricing_facts(req, replay, capacity)?.facts)
}

fn guarded_pricing_facts<'a>(
    req: &IosPricingRequest<'_>,
    replay: Option<IrisReplay<'a>>,
    capacity: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<GuardedPricingFacts<'a>> {
    req.validate_work_context()?;
    match (req.provider, replay) {
        (crate::model::MODEL_ID, None | Some(IrisReplay::Generation(_))) => {
            expect_mode(req.mode, &MemoryMode::TextToImage)?
        }
        (crate::restoration::MODEL_ID, None | Some(IrisReplay::Restoration(_))) => expect_mode(
            req.mode,
            &crate::memory_strategy::memory_mode(IrisTask::Restoration),
        )?,
        (crate::depth::DEPTH_MODEL_ID, None | Some(IrisReplay::Depth(_))) => expect_mode(
            req.mode,
            &crate::memory_strategy::memory_mode(IrisTask::Depth),
        )?,
        (provider, Some(_)) => {
            return Err(Error::Unsupported(format!(
                "{provider}: Iris replay token belongs to another task"
            )))
        }
        _ => return Err(Error::Unsupported("not an Iris guarded host route".into())),
    }
    // Generation must fail BEFORE any config/tokenizer/header work; do not reorder aa011.
    native_ledger::require_guarded_host_route(req)?;
    native_ledger::check_encoder_free_native_axes(req)?;
    let inventory = req.spec.source_inventory().ok_or_else(|| {
        Error::Unsupported("Iris guarded metadata requires retained SourceInventory".into())
    })?;
    let mut metadata = IrisMetadataFacts::default();
    let task = match req.provider {
        crate::restoration::MODEL_ID => IrisTask::Restoration,
        crate::depth::DEPTH_MODEL_ID => IrisTask::Depth,
        _ => return Err(Error::Unsupported("not an Iris guarded host route".into())),
    };
    let export = {
        guarded_source_read(inventory, req, capacity, |next| {
            observed_metadata_read(&mut metadata, next, |observe| {
                TaskExport::from_spec_observed(req.spec, task, req.provider, observe)
            })
        })?
    };
    // Existing readers retain raw JSON. Observe it in place; never reread just for diagnostics.
    let read = |path: &Path,
                metadata: &mut IrisMetadataFacts,
                capacity: &mut dyn FnMut(u64) -> Result<()>| {
        guarded_source_read(inventory, req, capacity, |next| {
            observed_metadata_read(metadata, next, |observe| {
                mlx_gen::gen_core::weightsmeta::safetensors_file_tensor_locations_observed(
                    path, observe,
                )
            })
        })
    };
    let weights = read(&export.weights_path(), &mut metadata, capacity)?;
    let real: Vec<_> = weights
        .tensors
        .into_iter()
        .map(|location| location.header)
        .collect();
    let artifact =
        crate::packed::artifact_tier_from_headers(&export.weights_path(), &real, || {
            metadata.observe(
                mlx_gen::gen_core::weightsmeta::SourceMetadataRead::Json {
                    file_bytes: weights.file_len,
                    bytes: &weights.header_json,
                },
                capacity,
            )?;
            mlx_gen::gen_core::weightsmeta::safetensors_metadata_from_json(
                &export.weights_path(),
                &weights.header_json,
            )
            .map_err(Error::from)
        })?;
    crate::memory_strategy::refuse_unloadable_quant(req.provider, req.spec).map_err(Error::from)?;
    crate::packed::refuse_dense_requested_from_tier(
        req.provider,
        &export.weights_path(),
        req.spec.quantize,
        artifact,
    )?;
    if task == IrisTask::Restoration && artifact.is_some() && req.spec.precision != Precision::Bf16
    {
        return Err(Error::Unsupported(format!(
            "iris: {} is a packed {artifact:?} artifact, which runs under the bf16 compute policy",
            export.weights_path().display()
        )));
    }
    metadata.capture_backbone(weights.file_len, &weights.header_json, &real)?;
    let streamed = req.spec.load_shape == LoadShape::DeferredMaterialization;
    let prefix = if task == IrisTask::Depth {
        "pixel."
    } else {
        ""
    };
    let input_channels = (export.config.model.in_channels as u64)
        .checked_add(if task == IrisTask::Depth {
            DEPTH_EXTRA_INPUT_CHANNELS as u64
        } else {
            0
        })
        .ok_or_else(|| Error::Unsupported("Iris guarded channel extent overflow".into()))?;
    let bb = priced_backbone_from_headers(
        BackbonePricingInput {
            config: &export.config,
            prefix,
            precision: req.spec.precision,
            requested: req.spec.quantize,
            streamed,
            input_channels,
            opts: Pricing::default(),
        },
        &real,
        artifact,
    )?;
    let empty = read(&export.empty_prompt_path(), &mut metadata, capacity)?;
    let empty_headers: Vec<_> = empty
        .tensors
        .into_iter()
        .map(|location| location.header)
        .collect();
    metadata.capture_empty(empty.file_len, &empty.header_json, &empty_headers)?;
    let (task, caller_bytes) = match (task, replay) {
        (IrisTask::Restoration, None | Some(IrisReplay::Restoration(_))) => {
            let token = match replay {
                Some(IrisReplay::Restoration(token)) => Some(token),
                _ => None,
            };
            if let Some(token) = token {
                metadata.check_replay_token(token.as_str(), capacity)?;
            }
            let resources =
                RestorationResources::from_export(export, req.provider).map_err(Error::from)?;
            let plan = restoration_priced_plan(req, &resources, token)?;
            let caller = if let Some(token) = token {
                token.wire()?.source_capacity_bytes
            } else {
                req.transform
                    .ok_or_else(|| Error::Unsupported("missing restoration host source".into()))?
                    .image
                    .pixels
                    .capacity() as u64
            };
            (GuardedTask::Restoration { resources, plan }, caller)
        }
        (IrisTask::Depth, None | Some(IrisReplay::Depth(_))) => {
            let (plan, caller) = if let Some(IrisReplay::Depth(token)) = replay {
                metadata.check_replay_token(token.as_str(), capacity)?;
                let wire = token.wire()?;
                if wire.patch != export.config.model.patch_size as u32 {
                    return Err(Error::Unsupported(
                        "Iris depth replay architecture patch mismatch".into(),
                    ));
                }
                (wire.plan()?, wire.source_capacity_bytes)
            } else {
                let actual = req
                    .depth
                    .ok_or_else(|| Error::Unsupported("missing actual depth context".into()))?;
                gen_core_iris::depth::validate_depth_request(actual).map_err(Error::from)?;
                (
                    gen_core_iris::depth::plan_depth_input(
                        actual.image.width,
                        actual.image.height,
                        actual.resolution,
                        export.config.model.patch_size,
                    )
                    .map_err(Error::from)?,
                    actual.image.pixels.capacity() as u64,
                )
            };
            (GuardedTask::Depth { export, plan }, caller)
        }
        _ => {
            return Err(Error::Unsupported(
                "Iris replay token belongs to another task".into(),
            ))
        }
    };
    let raw_headers = real.into_iter().map(|h| (h.name.clone(), h)).collect();
    let sources = GuardedSources {
        task,
        bb,
        raw_headers,
        empty_headers,
        metadata,
        replay,
        caller_bytes,
    };
    let mut priced = match &sources.task {
        GuardedTask::Restoration { resources, .. } => restoration_with_sources(
            req,
            Pricing::default(),
            resources,
            match replay {
                Some(IrisReplay::Restoration(token)) => Some(token),
                _ => None,
            },
            Some(&sources),
        )?,
        GuardedTask::Depth { export, .. } => depth_with_sources(
            req,
            Pricing::default(),
            export,
            match replay {
                Some(IrisReplay::Depth(token)) => Some(token),
                _ => None,
            },
            Some(&sources),
        )?,
    };
    priced.metadata = Some(sources.metadata);
    let mut facts = priced.into_facts();
    facts.host_analysis = Some(native_ledger::host_analysis_from_sources(
        req,
        &sources,
        facts.metadata.as_ref().ok_or_else(|| {
            Error::Unsupported(
                "Iris guarded source pricing lost its captured metadata facts".into(),
            )
        })?,
    )?);
    Ok(GuardedPricingFacts { facts, sources })
}

/// A request's pricing and every quantum it was folded from (`(phase, label, scratch)`).
#[derive(Default)]
pub(crate) struct Priced {
    pub pricing: IosPricing,
    pub quanta: Vec<(IosPhase, &'static str, u64)>,
    phase_areas: std::collections::BTreeMap<IosPhase, u64>,
    host_analysis: Option<IrisHostAnalysis>,
    metadata: Option<IrisMetadataFacts>,
}

/// Basis coordinate of the NEW internal tensor/host allocation helpers. This is a producer-stage
/// requirement, NOT an actual baseline snapshot; callers attach their real SDK/capture fact.
pub const INTERNAL_GROWTH_COORDINATE: mlx_gen::ios_pricing::IosMemoryCoordinate =
    mlx_gen::ios_pricing::IosMemoryCoordinate::BeforeModelLoad;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputStorageFacts {
    pub width: u32,
    pub height: u32,
    pub logical_bytes: u64,
    pub capacity_bytes: u64,
}

/// Actual provided RGB storage; never an image header fabricated into a zero-filled work item.
/// At BeforeModelLoad it is a baseline owner; BeforeRequestConstruction requires its real producer.
pub fn input_storage_facts(req: &IosPricingRequest<'_>) -> Result<Option<InputStorageFacts>> {
    req.validate_work_context()?;
    let image = req
        .transform
        .map(|work| &work.image)
        .or_else(|| req.depth.map(|work| &work.image));
    Ok(image.map(|image| InputStorageFacts {
        width: image.width,
        height: image.height,
        logical_bytes: image.pixels.len() as u64,
        capacity_bytes: image.pixels.capacity() as u64,
    }))
}

/// Actual per-phase binding area, paired with that point's price; not the legacy aggregate/source
/// geometry. Host buffers and serial outputs are explicit resident terms, not fake forward rows.
#[derive(Clone, Debug)]
pub struct PricingFacts {
    pub pricing: IosPricing,
    /// Diagnostic area of the compatibility summary's largest scratch check, not the area of
    /// its independent resident maximum. Exact source-window geometry lives in `pricing.windows`.
    pub phase_areas: std::collections::BTreeMap<IosPhase, u64>,
    pub host_analysis: Option<IrisHostAnalysis>,
    pub metadata: Option<IrisMetadataFacts>,
}

impl Priced {
    fn into_facts(self) -> PricingFacts {
        PricingFacts {
            pricing: self.pricing,
            phase_areas: self.phase_areas,
            host_analysis: self.host_analysis,
            metadata: self.metadata,
        }
    }
}

/// Record complete coincident points without treating any logical upper as a physical lower.
/// The argument list mirrors the source ledger: structural components, ambient owners, guard,
/// actual operands and semantic operation class remain separate until the source bounder runs.
#[allow(clippy::too_many_arguments)]
fn take_source_at(
    pricing: &mut Priced,
    phase: IosPhase,
    structural: u64,
    held: u64,
    quanta: &[(&'static str, u64)],
    basis: IosWorkBasis,
    class: &str,
    owner: &str,
    area: u64,
) {
    assert!(area > 0, "actual source window area must be positive");
    for (operator, &(label, scratch)) in quanta.iter().enumerate() {
        assert_eq!(
            ios_quantum_phase(label),
            Some(phase),
            "source window has wrong phase: {label}"
        );
        let total = structural.saturating_add(held);
        let previous = pricing.pricing.phase(phase).cloned();
        let largest_check = previous
            .as_ref()
            .is_none_or(|old| scratch > old.scratch_bytes);
        pricing.pricing.take_window(IosWindowPrice {
            phase,
            class: Some(format!("{class}/{label}/operator{operator}")),
            label: label.into(),
            resident_bytes: structural,
            held_bytes: held,
            scratch_bytes: scratch,
            basis: Some(basis.clone()),
            owner_domain: owner.into(),
        });
        // The public phase compatibility view promises the phase's largest guard and enough
        // retained owners for every check. These independent maxima are a conservative envelope,
        // NOT a coincident source point. Exact windows above remain the certificate authority;
        // never subtract this summary or treat it as a native/physical lower bound.
        let summary = pricing
            .pricing
            .phases
            .iter_mut()
            .find(|p| p.phase == phase)
            .expect("the source window created its phase");
        summary.resident_bytes = total.max(previous.as_ref().map_or(0, |old| old.resident_bytes));
        if largest_check {
            summary.scratch_bytes = scratch;
            summary.binding = label.into();
            pricing.phase_areas.insert(phase, area);
        } else if let Some(previous) = previous {
            summary.scratch_bytes = previous.scratch_bytes;
            summary.binding = previous.binding;
        }
        pricing.quanta.push((phase, label, scratch));
    }
}

fn plan_of(execution: &GenerationMemory) -> mlx_gen::AttentionPlan<'static> {
    if execution.chunk_attention {
        mlx_gen::AttentionPlan::budgeted(mlx_gen::AttentionBudget::from_score_elements(
            u64::from(
                execution
                    .attention_chunk_size
                    .unwrap_or(crate::memory_strategy::ATTENTION_CHUNK_SIZE),
            ),
            true,
        ))
    } else {
        mlx_gen::AttentionPlan::UNBOUNDED
    }
}

/// Qwen3-VL's build_rope owns one f32 row over the resolved rotary dimension, in resident AND
/// streamed form. Per-pass cos/sin tables are separate walk producers, not model parameters.
fn tower_rope_bytes(cfg: &mlx_llm::ModelConfig) -> Result<u64> {
    crate::stream::tower_rope_bytes(cfg)
}

/// One resident decoder walk's native-owned IDs, rotary tables, hidden states and K/V capacity. Unlike the streamed
/// picker, CausalLm::hidden_states returns ALL layers and owns a cache for every decoder layer.
/// Round each independently allocated state/K/V operand before multiplying by its source count.
/// This is a resident upper, not a simultaneous lower or a cumulative/process certificate.
fn resident_text_walk_bytes(cfg: &mlx_llm::ModelConfig, tokens: u64) -> Result<u64> {
    let overflow = || Error::Unsupported("Iris resident text walk byte arithmetic overflow".into());
    let product = |factors: &[u64]| {
        factors.iter().try_fold(1u64, |n, factor| {
            n.checked_mul(*factor).ok_or_else(overflow)
        })
    };
    let allocation = |bytes| {
        mlx_gen::ios_quantum::fresh_metal_allocation_bytes(bytes, crate::retention::PAGE_BYTES)
            .ok_or_else(overflow)
    };
    let layers = u64::try_from(cfg.num_layers).map_err(|_| overflow())?;
    let hidden = allocation(product(&[2, tokens, cfg.hidden_size.max(0) as u64])?)?;
    let block = mlx_llm::primitives::kv_cache::KV_BLOCK_TOKENS as u64;
    let cache = allocation(product(&[
        2,
        tokens.div_ceil(block),
        block,
        cfg.num_kv_heads.max(0) as u64,
        cfg.head_dim.max(0) as u64,
    ])?)?;
    let states = layers
        .checked_add(1)
        .and_then(|n| n.checked_mul(hidden))
        .ok_or_else(overflow)?;
    let caches = layers
        .checked_mul(2)
        .and_then(|n| n.checked_mul(cache))
        .ok_or_else(overflow)?;
    // resident_states keeps its uploaded IDs; lazy decoder graphs read the one shared per-pass
    // cosine/sine pair. The model-owned frequency row is already in tower.resident.
    let ids = allocation(product(&[4, tokens])?)?;
    let rotary = u64::try_from(cfg.rotary_dim()).map_err(|_| overflow())?;
    let table = allocation(product(&[2, tokens, rotary])?)?;
    let rope = table.checked_mul(2).ok_or_else(overflow)?;
    states
        .checked_add(caches)
        .and_then(|n| n.checked_add(ids))
        .and_then(|n| n.checked_add(rope))
        .ok_or_else(overflow)
}

/// The encoded window priced: the request's own when it names one, else the full window (the
/// chat-template prefix and a caption at the budget, its suffix inside the budget).
fn token_window(req: &IosPricingRequest<'_>, te_dir: &Path, max_length: u64) -> Result<u64> {
    if req.token_window > 0 {
        return Ok(u64::from(req.token_window));
    }
    let tokenizer = gen_core_tokenizer::TextTokenizer::from_file(
        te_dir.join("tokenizer.json"),
        gen_core_tokenizer::TokenizerConfig {
            max_length: usize::MAX,
            pad_token_id: 0,
            chat_template: gen_core_tokenizer::ChatTemplate::None,
            pad_to_max_length: false,
        },
    )
    .map_err(Error::from)?;
    let prefix = tokenizer
        .encode_ids(PROMPT_PREFIX, false)
        .map_err(Error::from)?
        .len() as u64;
    Ok(prefix + max_length)
}

/// Cold/first, first steady, and final positions of the repeated serial/row payload graph.
/// In these loops intermediate payloads differ only by nonnegative completed-output counts;
/// solver, CFG, block, cache birth and final-clamp graph transitions are enumerated separately.
/// This bounded payload envelope does NOT establish physical witness coverage for omitted work.
fn payload_transition_indices(count: u64) -> impl Iterator<Item = u64> {
    [
        count.checked_sub(1).map(|_| 0),
        (count > 1).then_some(1),
        count.checked_sub(1).filter(|last| *last > 1),
    ]
    .into_iter()
    .flatten()
}

/// The executed solve's graph-shaping controls: prompt rows, CFG gate, solver plan, serial count.
/// Built from the resolved actual request, or exactly from a recorded sampling token's normalized
/// wire (the same values `SamplingToken::apply_to` writes into a request before resolution).
struct GenerationEnvelope {
    rows: u64,
    cfg_scale: f32,
    cfg_interval: (f64, f64),
    plan: gen_core_iris::DpmSolverPlan,
    count: u32,
}

impl GenerationEnvelope {
    fn of_params(params: &GenerationParams, count: u32) -> Result<Self> {
        Ok(Self {
            rows: params.prompts.len() as u64,
            cfg_scale: params.cfg_scale,
            cfg_interval: params.cfg_interval,
            plan: params.plan_metadata().map_err(Error::from)?,
            count,
        })
    }

    fn of_wire(wire: &SamplingWire) -> Result<Self> {
        Ok(Self {
            rows: u64::from(wire.prompt_rows),
            cfg_scale: f32::from_bits(wire.guidance_bits),
            cfg_interval: (
                f32::from_bits(wire.interval_bits[0]) as f64,
                f32::from_bits(wire.interval_bits[1]) as f64,
            ),
            plan: wire.checked_plan()?,
            count: wire.serial_count,
        })
    }

    fn uses_cfg(&self) -> bool {
        self.cfg_scale != 1.0
    }

    fn cfg_at(&self, step: &DpmStep) -> bool {
        gen_core_iris::cfg_active(self.cfg_scale, step.s, self.cfg_interval)
    }

    fn epochs(&self) -> impl Iterator<Item = GenerationEpoch> + '_ {
        let indices = self.plan.epoch_indices(self.cfg_scale, self.cfg_interval);
        let rows = self.rows;
        payload_transition_indices(u64::from(self.count))
            .map(|serial| serial as u32)
            .flat_map(move |serial| {
                indices.clone().into_iter().map(move |index| {
                    let step = self.plan.step(index).expect("bounded epoch index");
                    let guided = self.cfg_at(&step);
                    GenerationEpoch {
                        serial,
                        index,
                        step,
                        guided,
                        forward_rows: rows * if guided { 2 } else { 1 },
                        history_rows: if index == 0 { 0 } else { rows },
                        completed_rows: rows * u64::from(serial),
                    }
                })
            })
    }
}

/// One generation side's executed controls, resolved without inventing a request.
struct GenerationWork {
    config: IrisConfig,
    envelope: GenerationEnvelope,
    rungs: crate::model::RequestRungs,
    prediction: Prediction,
    /// The ACTUAL prompts and CFG negative, `None` for a payload-free token replay: its caption
    /// tokens are not recorded, so callers bound each window by `assemble_window`'s full cap.
    prompts: Option<(Vec<String>, String)>,
    /// The actual request's preview sink state, `None` (unknown) for a token replay.
    preview_active: Option<bool>,
}

/// Resolve one generation side: an actual `GenerationRequest` (optionally replayed through its
/// recorded token), or, for a payload-free catalog anchor, the recorded token alone, with the
/// work cardinality, geometry and admitted `GenerationMemory` taken from the pricing request.
/// Neither present refuses exactly; no request is constructed.
fn generation_work(
    req: &IosPricingRequest<'_>,
    token: Option<&SamplingToken>,
    resources: &GenerationResources,
    streamed: bool,
) -> Result<GenerationWork> {
    let config = IrisConfig::from_dir(&resources.backbone_dir).map_err(Error::from)?;
    let work = match (req.generation, token) {
        (Some(supplied), token) => {
            let (config, replay) = match token {
                Some(token) => token.apply_to(&config, supplied)?,
                None => (config, supplied.clone()),
            };
            let actual = &replay;
            let params = resolved_generation_controls(&config, actual, actual.seed.unwrap_or(0))?;
            if actual.width != req.geometry.width
                || actual.height != req.geometry.height
                || actual.memory.unwrap_or_default() != *req.execution
            {
                return Err(Error::Unsupported(
                    "Iris pricing request context disagrees with geometry/execution".into(),
                ));
            }
            GenerationWork {
                envelope: GenerationEnvelope::of_params(&params, actual.count)?,
                rungs: crate::model::request_rungs(actual, streamed)?,
                prediction: params.prediction,
                prompts: Some((params.prompts.clone(), params.negative_prompt.clone())),
                preview_active: Some(actual.preview.is_active()),
                config,
            }
        }
        (None, Some(token)) => {
            let wire = token.wire()?;
            let config = wire.metadata_config(&config)?;
            let memory = Some(*req.execution);
            crate::model::refuse_unhonored_memory_of(memory)?;
            GenerationWork {
                envelope: GenerationEnvelope::of_wire(&wire)?,
                rungs: crate::model::memory_rungs(memory, streamed)?,
                prediction: config.prediction().map_err(Error::from)?,
                prompts: None,
                preview_active: None,
                config,
            }
        }
        (None, None) => {
            return Err(Error::Unsupported(
                "iris_3b pricing requires the actual generation request or its recorded sampling \
                 token (sampling, CFG and serial count)"
                    .into(),
            ))
        }
    };
    let total_images = work
        .envelope
        .rows
        .checked_mul(u64::from(work.envelope.count))
        .ok_or_else(|| Error::Msg("Iris total image count overflow".into()))?;
    if total_images != u64::from(req.geometry.batch) {
        return Err(Error::Unsupported(
            "Iris pricing request context disagrees with geometry/execution".into(),
        ));
    }
    Ok(work)
}

/// Owners already present before adapter reconstruction, and a conservative after-build upper.
/// Sources and constructor scalars remain counted at both boundaries: finishing reconstruction
/// does not grant retirement credit. Only allocation sites marked as new retained outputs join
/// the late upper; retained LoRA views alias sources and must not be counted a second time.
fn adapter_pending_held_bytes(
    host_bytes: u64,
    files: &[crate::adapters::AdapterFileReadFacts],
    loads: &std::collections::BTreeMap<String, crate::adapters::ProjectionAdapterLoadFacts>,
) -> Result<(u64, u64)> {
    let overflow = || Error::Unsupported("Iris pending adapter owner byte overflow".into());
    let sources = files.iter().try_fold(host_bytes, |sum, file| {
        sum.checked_add(file.source_bytes)
            .and_then(|bytes| bytes.checked_add(file.graph_constructor_bytes))
            .ok_or_else(overflow)
    })?;
    let late = loads.values().try_fold(sources, |sum, load| {
        sum.checked_add(load.new_retained_bytes()?)
            .ok_or_else(overflow)
    })?;
    Ok((sources, late))
}

fn generation(
    req: &IosPricingRequest<'_>,
    opts: Pricing,
    token: Option<&SamplingToken>,
) -> Result<Priced> {
    if req.transform.is_some() {
        return Err(Error::Unsupported(
            "Iris generation cannot carry transform context".into(),
        ));
    }
    let spec = req.spec;
    crate::memory_strategy::refuse_streamed_adapters(crate::model::MODEL_ID, spec)
        .map_err(Error::from)?;
    crate::memory_strategy::refuse_unloadable_quant(crate::model::MODEL_ID, spec)
        .map_err(Error::from)?;
    let resources =
        GenerationResources::from_spec(spec, crate::model::MODEL_ID).map_err(Error::from)?;
    let streamed = spec.load_shape == LoadShape::DeferredMaterialization;
    let GenerationWork {
        config,
        envelope: recipe,
        rungs,
        ..
    } = generation_work(req, token, &resources, streamed)?;
    let prompts = recipe.rows;
    let staged = req.execution.stage_residency || spec.offload_policy == OffloadPolicy::Sequential;
    let plan = plan_of(req.execution);
    let mut bb = priced_backbone(
        &config,
        &resources
            .backbone_dir
            .join(gen_core_iris::BACKBONE_WEIGHTS_FILE),
        "",
        spec.precision,
        spec.quantize,
        streamed,
        config.model.in_channels as u64,
        opts,
    )?;
    let base_bb_resident = bb.resident;
    let (adapter_inventory, packed_adapter_hosts, width_read_bytes) = if spec.adapters.is_empty() {
        (crate::adapters::AdapterHeaderInventory::default(), 0, 0)
    } else {
        let host = crate::adapters::projection_host_facts_from_headers(
            &resources
                .backbone_dir
                .join(gen_core_iris::BACKBONE_WEIGHTS_FILE),
        )?;
        (
            crate::adapters::header_inventory(
                &spec.adapters,
                &host.shapes,
                compute_dtype(spec.precision),
                IrisTask::Generation,
                crate::model::MODEL_ID,
            )?,
            host.packed_projections,
            host.width_read_bytes,
        )
    };
    let adapter_loads = adapter_inventory.projection_loads()?;
    let adapter_files = adapter_inventory.file_read_facts()?;
    let adapter_host_bytes = crate::adapters::host_constructor_bytes(packed_adapter_hosts)?;
    let (adapter_build_held, adapter_evaluation_held) =
        adapter_pending_held_bytes(adapter_host_bytes, &adapter_files, &adapter_loads)?;
    let adapter_resident = adapter_inventory
        .projections
        .values()
        .flatten()
        .flat_map(mlx_gen::adapters::facts::AdapterFacts::tensors)
        .try_fold(0u64, |sum, tensor| {
            let request = tensor.shape.iter().try_fold(
                mlx_gen::adapters::facts::dtype_width(tensor.dtype),
                |size, &dim| {
                    let dim = u64::try_from(dim).map_err(|_| {
                        Error::Unsupported("negative Iris adapter retained extent".into())
                    })?;
                    size.checked_mul(dim).ok_or_else(|| {
                        Error::Unsupported("Iris adapter retained request overflow".into())
                    })
                },
            )?;
            let bytes = mlx_gen::ios_quantum::fresh_metal_allocation_bytes(
                request,
                crate::retention::PAGE_BYTES,
            )
            .ok_or_else(|| {
                Error::Unsupported("Iris adapter retained allocation overflow".into())
            })?;
            sum.checked_add(bytes)
                .ok_or_else(|| Error::Unsupported("Iris adapter retained sum overflow".into()))
        })?;
    bb.resident = bb
        .resident
        .checked_add(adapter_resident)
        .ok_or_else(|| Error::Unsupported("Iris backbone/adapter resident sum overflow".into()))?;
    bb.facts.adapter_facts = adapter_inventory.projections.clone();
    let te = priced_tower(&resources.text_encoder_dir, spec.quantize, streamed, opts)?;
    let te_cfg = &te.cfg;
    let max_length = config.text_encoder.max_length as u64;
    let layers = config.text_encoder.hidden_layers.len() as u64;
    let dim = config.text_encoder.dim as u64;
    let deepest = *config.text_encoder.hidden_layers.last().unwrap_or(&0);
    let n = token_window(req, &resources.text_encoder_dir, max_length)?;
    let cond_elements = max_length * layers * dim;
    let cond = allocated(tensor_bytes(4, &[cond_elements]));
    let uncond = u64::from(recipe.uses_cfg());
    let (h, w) = (
        u64::from(req.geometry.height),
        u64::from(req.geometry.width),
    );
    let channels = config.model.in_channels as u64;
    let warm_te = if staged { 0 } else { te.resident };
    // Outside the denoise a streamed backbone holds no window: only its skeleton.
    let warm_bb = if staged { 0 } else { bb.resident };
    let mut pricing = Priced::default();
    pricing.pricing.phase_not_run(
        IosPhase::Encode,
        "Iris text generation has no init-image encoder",
    );

    // Load: the tower, then the backbone (staged: once the conditioning exists; warm: beside the
    // tower), and every adapter file.
    let load_conditioning = if staged { (prompts + uncond) * cond } else { 0 };
    let load_tower = if staged { 0 } else { te.resident };
    take_source_at(
        &mut pricing,
        IosPhase::Load,
        0,
        0,
        &[
            (te_q::LOAD, te.floor),
            (
                crate::stream::LOAD_TARGET_QUANTUM,
                te.floor.checked_add(te.load_owned_upper).ok_or_else(|| {
                    Error::Unsupported("Iris tower joint load byte overflow".into())
                })?,
            ),
        ],
        IosWorkBasis::default(),
        "iris.generation.load/text",
        "model::load_model original raw floor then actual joint load-target guard",
        1,
    );
    take_source_at(
        &mut pricing,
        IosPhase::Load,
        load_tower,
        load_conditioning,
        &[
            (crate::model::BACKBONE_LOAD_QUANTUM, bb.floor),
            (crate::stream::LOAD_TARGET_QUANTUM, bb.load_targets),
        ],
        IosWorkBasis::default(),
        "iris.generation.load/backbone",
        "model::load_model original raw floor then actual joint load-target guard",
        1,
    );
    if !spec.adapters.is_empty() {
        let add = |a: u64, b: u64| {
            a.checked_add(b).ok_or_else(|| {
                Error::Unsupported("Iris adapter load ownership sum overflow".into())
            })
        };
        let host = adapter_host_bytes;
        // Conservative load payload upper: source base inventory and its eventual cast parameters
        // can coexist during build. This is not a coincident physical lower or an anchor credit.
        let base_upper = add(add(load_tower, base_bb_resident)?, bb.floor)?;
        if host == 0 {
            let first = adapter_files
                .first()
                .ok_or_else(|| Error::Unsupported("iris: no adapter file read facts".into()))?;
            take_source_at(
                &mut pricing,
                IosPhase::Load,
                base_upper,
                load_conditioning,
                &[(
                    crate::adapters::ADAPTER_LOAD_QUANTUM,
                    first.file_floor_bytes,
                )],
                IosWorkBasis::default(),
                "iris.generation.load/first_dense_adapter_floor",
                "adapters::install file floor before zero-native host construction",
                1,
            );
        }
        if width_read_bytes != 0 {
            take_source_at(
                &mut pricing,
                IosPhase::Load,
                base_upper,
                load_conditioning,
                &[(crate::adapters::ADAPTER_WIDTH_QUANTUM, width_read_bytes)],
                IosWorkBasis::default(),
                "iris.generation.load/adapter_width_reads",
                "adapters::logical_projection_shapes before original try_item source/cast eval",
                1,
            );
        }
        take_source_at(
            &mut pricing,
            IosPhase::Load,
            base_upper,
            load_conditioning,
            &[(crate::adapters::ADAPTER_HOST_QUANTUM, host)],
            IosWorkBasis::default(),
            "iris.generation.load/adapter_host",
            "adapters::install before placeholders; zero-native dense stage is still observed",
            1,
        );
        let mut previous_file_payload = 0u64;
        for read in &adapter_files {
            take_source_at(
                &mut pricing,
                IosPhase::Load,
                base_upper,
                add(add(load_conditioning, host)?, previous_file_payload)?,
                &[
                    (
                        crate::adapters::ADAPTER_LOAD_QUANTUM,
                        read.read_guard_bytes()?,
                    ),
                    // apply_adapters_strict invokes the shared loader's own pre-file floor.
                    // It opens the same file before any current-file graph constructor runs.
                    (
                        mlx_gen::adapters::loader::ADAPTER_LOAD_QUANTUM,
                        read.file_floor_bytes,
                    ),
                ],
                IosWorkBasis::default(),
                "iris.generation.load/adapter_file",
                "adapters::install ordered strict file read/graph construction",
                1,
            );
            previous_file_payload = add(
                previous_file_payload,
                add(read.source_bytes, read.graph_constructor_bytes)?,
            )?;
        }
        for (path, load) in &adapter_loads {
            // Loader construction order is not BTreeMap order. Keep EVERY OTHER projection's
            // distinct newly produced parameter as an upper, without subtracting a max or counting
            // this projection's final output twice beside its reconstruction scratch. File aliases
            // remain counted in source inventory, not as another new retained allocation.
            let other_new = adapter_loads
                .iter()
                .filter(|(other, _)| *other != path)
                .try_fold(0u64, |sum, (_, other)| {
                    add(sum, other.new_retained_bytes()?)
                })?;
            let held = add(
                add(add(load_conditioning, host)?, previous_file_payload)?,
                other_new,
            )?;
            take_source_at(&mut pricing, IosPhase::Load, base_upper, held,
                &[(crate::adapters::ADAPTER_MATERIALIZE_QUANTUM, load.materialize_scratch_bytes()?)],
                IosWorkBasis::default(), "iris.generation.load/adapter_materialize",
                "nn::Loader::linear reconstruction upper; other outputs retained conservatively, not a simultaneous lower", 1);
        }
    }

    let dense_load_quant = !streamed
        && spec.quantize.is_some()
        && crate::packed::artifact_tier(
            &resources
                .backbone_dir
                .join(gen_core_iris::BACKBONE_WEIGHTS_FILE),
        )?
        .is_none();
    if !spec.adapters.is_empty() || dense_load_quant {
        let residuals = adapter_loads.values().try_fold(0u64, |sum, load| {
            sum.checked_add(load.materialize_scratch_bytes()?)
                .ok_or_else(|| Error::Unsupported("Iris pending residual byte overflow".into()))
        })?;
        let stage_plan = crate::stream::PendingBackbone {
            constructors: bb.constructor_outputs,
            evaluation: bb.evaluation_outputs,
            width_outputs: bb.width_outputs,
        };
        let pending = stage_plan
            .before_build_targets(!spec.adapters.is_empty())?
            .checked_add(residuals)
            .ok_or_else(|| Error::Unsupported("Iris pending construction byte overflow".into()))?;
        let model_held = load_conditioning
            .checked_add(bb.read_sources_upper)
            .ok_or_else(|| Error::Unsupported("Iris adapted base held byte overflow".into()))?;
        // File factors, graph scalars and packed-host placeholders already exist at this
        // boundary. A future retained delta is not a substitute for those input owners.
        let held = model_held.checked_add(adapter_build_held).ok_or_else(|| {
            Error::Unsupported("Iris pending adapter source held byte overflow".into())
        })?;
        // Completed operations may still have live buffers. Keep their source upper outside
        // future-allocation scratch; this is not retirement or common-owner subtraction.
        let pre_completed = if spec.adapters.is_empty() {
            0
        } else {
            bb.width_outputs
        };
        let pre_held = held
            .checked_add(pre_completed)
            .ok_or_else(|| Error::Unsupported("Iris completed width held byte overflow".into()))?;
        take_source_at(
            &mut pricing,
            IosPhase::Load,
            load_tower,
            pre_held,
            &[(crate::nn::PENDING_CONSTRUCTION_QUANTUM, pending)],
            IosWorkBasis::default(),
            "iris.generation.load/pending_construction",
            "active source/read inputs and completed installed-width targets; pending build",
            1,
        );
        // Keep the same source owners conservatively after evaluation, and add only distinct
        // new retained outputs. File aliases are already in adapter_evaluation_held's sources.
        let completed = stage_plan.completed_before_evaluation()?;
        let late_held = model_held
            .checked_add(adapter_evaluation_held)
            .and_then(|bytes| bytes.checked_add(completed))
            .ok_or_else(|| Error::Unsupported("Iris completed build held byte overflow".into()))?;
        take_source_at(&mut pricing, IosPhase::Load, load_tower, late_held,
            &[(crate::nn::ADAPTED_BASE_EVAL_QUANTUM, stage_plan.after_build_targets()?)],
            IosWorkBasis::default(), "iris.generation.load/pending_evaluation",
            "completed getters/build buffers retained conservatively; only pending evaluation scratch", 1);
    }

    // Conditioning: every prompt and the CFG null in turn; the conditionings already returned
    // stay resident.
    let held_conds = (prompts + uncond - 1) * cond;
    let n_i = n as i32;
    let real_max = (0..=max_length as i32)
        .map(|real| crate::text_encoder::conditioning_bytes(layers, max_length, dim, real))
        .max()
        .unwrap_or(0);
    let mut text = vec![(te_q::CONDITIONING, real_max)];
    let walk;
    let mut text_weight_window = 0;
    if streamed {
        let run = deepest.min(te_cfg.num_layers);
        let window = rungs.text_window.unwrap_or(run.max(1));
        text.push((
            te_q::EMBEDDING,
            crate::text_encoder::streamed_embedding_bytes(te_cfg, n),
        ));
        text.push((te_q::LAYER_READ, largest_window(&te.reads[..run], window).0));
        text.push((
            te_q::STREAMED_LAYER,
            crate::text_encoder::streamed_layer_bytes(te_cfg, n),
        ));
        // The walk's carry and picked states, and the RoPE tables.
        text_weight_window = largest_window(&te.reads[..run], window).1;
        walk = (layers + 1) * allocated(2 * n * te_cfg.hidden_size.max(0) as u64)
            + 2 * allocated(2 * n * te_cfg.head_dim.max(0) as u64)
            + text_weight_window;
    } else {
        // resident_states constructs the complete decoder graph/cache BEFORE its layer checks
        // and retains every returned hidden-state handle through conditioning. Those sources
        // are resident at checks, not merely part of layer_scratch's whole-walk guard. Keep a
        // source-shaped upper here even for an unpicked trailing layer; no retirement credit.
        walk = resident_text_walk_bytes(te_cfg, n)?;
        let scratch = |k: usize| {
            let (layer, held) = crate::text_encoder::decoder_layer_parts(te_cfg, n_i, k);
            layer.saturating_add(held)
        };
        text.push((te_q::TOWER, scratch(deepest)));
        text.push((te_q::EMBEDDING, scratch(0)));
        text.push((te_q::LAYER, (1..=deepest).map(scratch).max().unwrap_or(0)));
        text.push((
            te_q::DECODER_LAYER,
            (1..=te_cfg.num_layers).map(scratch).max().unwrap_or(0),
        ));
    }
    take_source_at(
        &mut pricing,
        IosPhase::Conditioning,
        te.resident + warm_bb + text_weight_window,
        held_conds
            + walk
                .checked_sub(text_weight_window)
                .expect("window weight is part of the walk ledger"),
        &text,
        IosWorkBasis {
            forward_rows: 1,
            token_count: n,
            ..Default::default()
        },
        "iris.generation.conditioning",
        "text_encoder::encode_in source walk and prior conditionings",
        n,
    );

    // All conditionings remain owned by the outer request. Within EACH serial solve, concatenate
    // with one input aliases its original buffer; a multi-prompt batch and the CFG batch allocate.
    let held_batches = if prompts > 1 {
        allocated(4 * prompts * cond_elements)
    } else {
        0
    } + if uncond != 0 {
        allocated(4 * 2 * prompts * cond_elements)
    } else {
        0
    };
    let state_elements = prompts * channels * h * w;
    let state = allocated(4 * state_elements);
    let table = bb.facts.pixel_table(h, w)?;
    let conditioning = (prompts + uncond) * cond;
    let plan_host = recipe.plan.host_bytes().map_err(Error::from)?;
    let solve_held = conditioning + held_batches + plan_host;

    // Pointwise epochs: pair this forward's actual rows/binding with the history and completed
    // outputs alive at THAT evaluation, then maximize. Never subtract independent phase maxima.
    for epoch in recipe.epochs() {
        let b = epoch.forward_rows;
        let mut forward = Vec::new();
        let by_row = opts.pixel_rows && rungs.pixel_rows && b > 1;
        // Preserve raw guard formulas, but never reuse their independent maximum carry as a
        // coincident source owner. Every actual operation gets its own lexical-owner ledger.
        let _raw_max_carry = bb.facts.forward(b, h, w, plan, by_row, &mut forward)?;
        // The original noise stays owned by model::generate_impl. First x aliases it; later x and
        // previous x0 are separate f32 states. Euler also retains previous x0 in the real sampler.
        let current_history = state + if epoch.index == 0 { 0 } else { 2 * state };
        let completed = u64::from(epoch.serial) * state;
        let cfg_input = if epoch.guided {
            allocated(4 * b * channels * h * w)
        } else {
            0
        };
        let model_time = allocated(4 * b);
        for (operator, &(label, scratch)) in forward.iter().enumerate() {
            let index = forward[..operator]
                .iter()
                .filter(|point| point.0 == label)
                .count();
            let by_one_row = by_row && matches!(label, dit_q::PIXEL_BLOCK | dit_q::OUTPUT);
            let area = match label {
                dit_q::TEXT_LAYER_BLOCK
                | dit_q::TEXT_POOL
                | dit_q::TEXT_MASK
                | dit_q::TEXT_REFINER_BLOCK
                | dit_q::TEXT_OUTPUT => b * config.model.text_len as u64,
                dit_q::PIXEL_POSITIONS => {
                    (h / config.model.patch_size as u64) * (w / config.model.patch_size as u64)
                }
                _ if by_one_row => h * w,
                _ => b * h * w,
            };
            let window = if streamed {
                bb.facts.weight_window_at(label, index, rungs.dit_window)?
            } else {
                0
            };
            let table_owned = epoch.serial != 0
                || epoch.index != 0
                || matches!(
                    label,
                    dit_q::PIXEL_POSITIONS
                        | dit_q::PIXEL_BLOCK
                        | dit_q::OUTPUT
                        | dit_q::OUTPUT_ROWS
                );
            for row in payload_transition_indices(if by_one_row { b } else { 1 }) {
                let owned = bb.facts.held_at(b, h, w, by_row, label, index, row)?;
                take_source_at(&mut pricing, IosPhase::Denoise, bb.resident + warm_te + window,
                    solve_held + completed + current_history + cfg_input + model_time + owned + if table_owned { table } else { 0 },
                    &[(label, scratch)], IosWorkBasis {
                        forward_rows: if by_one_row { 1 } else { b }, token_count: config.model.text_len as u64,
                        model_width: w as u32, model_height: h as u32, source_plane_bytes: state,
                        processed_plane_bytes: allocated(4 * b * channels * h * w), output_plane_bytes: completed,
                        patch_width: config.model.patch_size as u32, patch_height: config.model.patch_size as u32,
                    }, &format!("iris.gen.forward/op{operator}/guided{}/history{}/first{}/terminal{}/serial{}/row-first{}",
                        epoch.guided, epoch.index != 0, epoch.index == 0, epoch.index + 1 == recipe.plan.len(), epoch.serial != 0, row == 0),
                    "pipeline::denoise_in -> solver::sample -> dit::forward_impl exact lexical owners", area);
            }
        }
        if streamed {
            for (stack, next_label) in [
                (Stack::Layer, dit_q::TEXT_LAYER_BLOCK),
                (Stack::Refiner, dit_q::TEXT_REFINER_BLOCK),
                (Stack::Trunk, dit_q::TRUNK_BLOCK),
                (Stack::Pixel, dit_q::PIXEL_BLOCK),
            ] {
                let reads = bb.facts.inventory.reads(stack);
                if reads.is_empty() {
                    continue;
                } // actual run_stack returns without any read
                let count = rungs
                    .dit_window
                    .unwrap_or(reads.len())
                    .clamp(1, reads.len());
                for first in (0..reads.len()).step_by(count) {
                    let end = (first + count).min(reads.len());
                    let owned = bb.facts.held_at(b, h, w, by_row, next_label, first, 0)?;
                    let cache = if epoch.serial != 0 || epoch.index != 0 || stack == Stack::Pixel {
                        table
                    } else {
                        0
                    };
                    take_source_at(&mut pricing, IosPhase::Denoise, bb.resident + warm_te,
                        solve_held + cache + completed + current_history + cfg_input + model_time + owned,
                        &[(dit_q::BLOCK_READ, crate::stream::window_read_bytes(&reads[first..end]))],
                        IosWorkBasis { forward_rows: b, token_count: config.model.text_len as u64,
                            model_width: w as u32, model_height: h as u32, source_plane_bytes: state,
                            output_plane_bytes: completed, patch_width: config.model.patch_size as u32,
                            patch_height: config.model.patch_size as u32, ..Default::default() },
                        &format!("iris.gen.read/{stack:?}/block{first}/guided{}/history{}/terminal{}/serial{}",
                            epoch.guided, epoch.index != 0, epoch.index + 1 == recipe.plan.len(), epoch.serial != 0),
                        "dit::run_stack window entry before new block ownership", 1);
                }
            }
        }
        // Model predictions are already evaluated when the forward returns. CFG's split views
        // share ONE 2P prediction allocation; the lazy combine materializes in the step quantum.
        let predictions = allocated(4 * b * channels * h * w);
        take_source_at(
            &mut pricing,
            IosPhase::Denoise,
            bb.resident + warm_te,
            solve_held + table + completed + current_history + predictions,
            &[(
                crate::solver::STEP_QUANTUM,
                crate::solver::step_scratch(state_elements, &epoch.step.update),
            )],
            IosWorkBasis {
                forward_rows: prompts,
                model_width: w as u32,
                model_height: h as u32,
                source_plane_bytes: state,
                processed_plane_bytes: predictions,
                output_plane_bytes: completed,
                ..Default::default()
            },
            &format!(
                "iris.gen.solver/update{}/guided{}/history{}/first{}/terminal{}/serial{}",
                if matches!(epoch.step.update, gen_core_iris::DpmUpdate::Second { .. }) {
                    2
                } else {
                    1
                },
                epoch.guided,
                epoch.index != 0,
                epoch.index == 0,
                epoch.index + 1 == recipe.plan.len(),
                epoch.serial != 0
            ),
            "solver::sample + step_update and lazy cfg_combine",
            prompts * h * w,
        );
    }
    for serial in payload_transition_indices(u64::from(recipe.count)) {
        take_source_at(
            &mut pricing,
            IosPhase::Denoise,
            bb.resident + warm_te,
            conditioning + serial * state + state + if serial == 0 { 0 } else { table },
            &[(crate::solver::PLAN_QUANTUM, plan_host)],
            IosWorkBasis {
                forward_rows: prompts,
                model_width: w as u32,
                model_height: h as u32,
                source_plane_bytes: state,
                output_plane_bytes: serial * state,
                ..Default::default()
            },
            &format!("iris.gen.plan/serial{}", serial != 0),
            "pipeline::denoise_in plan/sigma Vec before allocation",
            1,
        );
        // sample has returned, dropping previous x0 BEFORE the final clamp. The original noise,
        // current solution and solve-local conditioning batches remain alive until clamp finishes.
        take_source_at(
            &mut pricing,
            IosPhase::Denoise,
            bb.resident + warm_te,
            solve_held + table + serial * state + 2 * state,
            &[(
                crate::solver::FINALIZE_QUANTUM,
                crate::solver::final_clamp_scratch(state_elements),
            )],
            IosWorkBasis {
                forward_rows: prompts,
                model_width: w as u32,
                model_height: h as u32,
                source_plane_bytes: state,
                processed_plane_bytes: state,
                output_plane_bytes: serial * state,
                ..Default::default()
            },
            &format!("iris.gen.final_clamp/serial{}", serial != 0),
            "pipeline::denoise_in evaluated final clamp after sample history drop",
            prompts * h * w,
        );
        // Decode runs only after ALL serial solves. Their f32 outputs stay resident, while the
        // solve-local batches and original noises have dropped. Earlier RGB8 results are host bytes.
        let prior_rgb = serial * prompts * 3 * h * w;
        take_source_at(
            &mut pricing,
            IosPhase::Decode,
            bb.resident + warm_te,
            conditioning + u64::from(recipe.count) * state + table + prior_rgb,
            &[(
                crate::model::IMAGE_QUANTUM,
                crate::model::image_scratch(prompts * 3 * h * w, prompts),
            )],
            IosWorkBasis {
                forward_rows: prompts,
                model_width: w as u32,
                model_height: h as u32,
                source_plane_bytes: u64::from(recipe.count) * state,
                output_plane_bytes: prior_rgb,
                ..Default::default()
            },
            &format!("iris.gen.rgb/serial{}", serial != 0),
            "model::generate_impl retains every solved batch through serial RGB conversion",
            prompts * h * w,
        );
    }
    Ok(pricing)
}

/// The encoder-free routes' backbone forward and stream (their deferred load streams at the
/// published window; no request selects anything else).
fn export_window(streamed: bool) -> Option<usize> {
    streamed.then_some(crate::stream::TRANSFORMER_WINDOW_SIZE as usize)
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RestorationToken(String);

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RestorationWire {
    source_width: u32,
    source_height: u32,
    source_capacity_bytes: u64,
    scale_bits: u64,
    input_sizing: u8,
    color_fix: bool,
    tile: u32,
    patch: u32,
    sigma_bits: u64,
    num_train_timesteps: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RestorationMemoryClass {
    pub budget_applied: bool,
    pub enlarged: bool,
    pub single_pass: bool,
    pub color_fix: bool,
}

impl RestorationWire {
    fn plan(&self) -> Result<gen_core_iris::restoration::RestorationPlanMetadata> {
        let sigma = f64::from_bits(self.sigma_bits);
        let scale = f64::from_bits(self.scale_bits);
        let image_bytes = u64::from(self.source_width)
            .checked_mul(u64::from(self.source_height))
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| Error::Unsupported("Iris restoration source byte overflow".into()))?;
        if self.source_capacity_bytes < image_bytes
            || self.num_train_timesteps == 0
            || !sigma.is_finite()
            || sigma <= 0.0
            || sigma > 1.0
            || !scale.is_finite()
            || scale <= 0.0
            || !(scale as f32).is_finite()
        {
            return Err(Error::Unsupported(
                "invalid normalized Iris restoration metadata".into(),
            ));
        }
        let roundtrip = gen_core_iris::restoration::request_scale(scale as f32);
        if roundtrip.to_bits() != self.scale_bits {
            return Err(Error::Unsupported(
                "Iris restoration scale is not a canonical request scale".into(),
            ));
        }
        let input_sizing = match self.input_sizing {
            0 => mlx_gen::InputSizing::Original,
            1 => mlx_gen::InputSizing::Budgeted,
            _ => return Err(Error::Unsupported("invalid Iris input sizing".into())),
        };
        let plan = gen_core_iris::restoration::plan_metadata(
            gen_core_iris::restoration::Dims::new(self.source_width, self.source_height),
            &gen_core_iris::restoration::RestorationOptions {
                scale,
                input_sizing,
                color_fix: self.color_fix,
            },
            gen_core_iris::restoration::TileGeometry {
                tile: self.tile,
                patch: self.patch,
            },
            crate::restoration::MODEL_ID,
        )
        .map_err(Error::from)?;
        crate::restoration::host_plane_bytes_metadata(&plan)?;
        Ok(plan)
    }
}

pub fn canonical_restoration_token(
    spec: &mlx_gen::LoadSpec,
    actual: &mlx_gen::TransformRequest,
) -> Result<RestorationToken> {
    let resources =
        RestorationResources::from_spec(spec, crate::restoration::MODEL_ID).map_err(Error::from)?;
    let plan = restoration_input_plan(spec, actual)?;
    let wire = RestorationWire {
        source_width: actual.image.width,
        source_height: actual.image.height,
        source_capacity_bytes: actual.image.pixels.capacity() as u64,
        scale_bits: plan.scale.to_bits(),
        input_sizing: if plan.input_sizing == mlx_gen::InputSizing::Original {
            0
        } else {
            1
        },
        color_fix: plan.color_fix,
        tile: plan.tile,
        patch: plan.patch,
        sigma_bits: resources.settings.sigma.to_bits(),
        num_train_timesteps: resources.export.config.flow.num_train_timesteps as u64,
    };
    RestorationToken::from_token(&format!(
        "iris-restoration-v1:{}",
        serde_json::to_string(&wire).map_err(|e| Error::Msg(e.to_string()))?
    ))
}

/// Finite actual-source control/topology representatives. No source image is cloned or invented;
/// every refusal is retained explicitly, and valid plans dedup by semantic class, not exact size.
pub fn restoration_representatives(
    spec: &mlx_gen::LoadSpec,
    carrier: &mlx_gen::TransformRequest,
) -> Result<Vec<Result<RestorationToken>>> {
    let actual = canonical_restoration_token(spec, carrier)?;
    let base = actual.wire()?;
    let mut classes = std::collections::BTreeSet::from([actual.memory_class()?]);
    let mut representatives = vec![Ok(actual)];
    for input_sizing in [0u8, 1] {
        let source = gen_core_iris::restoration::Dims::new(base.source_width, base.source_height);
        let input = if input_sizing == 0 {
            source
        } else {
            gen_core_iris::restoration::fit_budget_dims(
                source,
                gen_core_iris::restoration::INPUT_BUDGET,
            )
        };
        let short = input.width.min(input.height) as f64;
        let small = ((base.tile / 2).max(1) as f64 / short) as f32;
        let large = ((u64::from(base.tile) + u64::from(base.patch)) as f64 / short) as f32;
        for color_fix in [false, true] {
            for scale in [
                f64::from_bits(base.scale_bits) as f32,
                small.next_up(),
                large.next_up(),
            ] {
                let mut wire = base.clone();
                wire.input_sizing = input_sizing;
                wire.color_fix = color_fix;
                wire.scale_bits = gen_core_iris::restoration::request_scale(scale).to_bits();
                let token = serde_json::to_string(&wire)
                    .map_err(|e| Error::Msg(e.to_string()))
                    .and_then(|wire| {
                        RestorationToken::from_token(&format!("iris-restoration-v1:{wire}"))
                    });
                match token {
                    Ok(token) => {
                        if classes.insert(token.memory_class()?) {
                            representatives.push(Ok(token));
                        }
                    }
                    Err(error) => representatives.push(Err(error)),
                }
            }
        }
    }
    Ok(representatives)
}

impl RestorationToken {
    fn wire(&self) -> Result<RestorationWire> {
        serde_json::from_str(
            self.0
                .strip_prefix("iris-restoration-v1:")
                .ok_or_else(|| Error::Unsupported("invalid Iris restoration prefix".into()))?,
        )
        .map_err(|e| Error::Msg(e.to_string()))
    }
    pub fn from_token(token: &str) -> Result<Self> {
        let wire: RestorationWire = serde_json::from_str(
            token
                .strip_prefix("iris-restoration-v1:")
                .ok_or_else(|| Error::Unsupported("not an Iris restoration token".into()))?,
        )
        .map_err(|e| Error::Msg(e.to_string()))?;
        wire.plan()?;
        let canonical = format!(
            "iris-restoration-v1:{}",
            serde_json::to_string(&wire).map_err(|e| Error::Msg(e.to_string()))?
        );
        if canonical != token {
            return Err(Error::Unsupported(
                "noncanonical Iris restoration token".into(),
            ));
        }
        Ok(Self(canonical))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
    pub fn to_value(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self.wire()?).map_err(|e| Error::Msg(e.to_string()))
    }
    pub fn plan(&self) -> Result<gen_core_iris::restoration::RestorationPlanMetadata> {
        self.wire()?.plan()
    }
    /// Replay actual task controls in place; source RGB ownership/capacity and artifact metadata
    /// must match. No image, request or config clone and no fabricated source are involved.
    pub fn apply_runtime_to(
        &self,
        spec: &mlx_gen::LoadSpec,
        carrier: &mut mlx_gen::TransformRequest,
    ) -> Result<()> {
        let wire = self.wire()?;
        wire.plan()?;
        let resources = RestorationResources::from_spec(spec, crate::restoration::MODEL_ID)
            .map_err(Error::from)?;
        let geometry = resources.settings.geometry(&resources.export.config);
        if (
            carrier.image.width,
            carrier.image.height,
            carrier.image.pixels.capacity() as u64,
        ) != (
            wire.source_width,
            wire.source_height,
            wire.source_capacity_bytes,
        ) || geometry.tile != wire.tile
            || geometry.patch != wire.patch
            || resources.settings.sigma.to_bits() != wire.sigma_bits
            || resources.export.config.flow.num_train_timesteps as u64 != wire.num_train_timesteps
        {
            return Err(Error::Unsupported(
                "Iris restoration runtime recipe disagrees with actual source/artifact".into(),
            ));
        }
        let input_sizing = match wire.input_sizing {
            0 => mlx_gen::InputSizing::Original,
            1 => mlx_gen::InputSizing::Budgeted,
            _ => {
                return Err(Error::Unsupported(
                    "invalid Iris restoration runtime input sizing".into(),
                ))
            }
        };
        let previous = (carrier.target, carrier.input_sizing, carrier.color_fix);
        carrier.target = mlx_gen::TargetSize::Scale(f64::from_bits(wire.scale_bits) as f32);
        carrier.input_sizing = input_sizing;
        carrier.color_fix = Some(wire.color_fix);
        let result = canonical_restoration_token(spec, carrier).and_then(|actual| {
            if actual == *self {
                Ok(())
            } else {
                Err(Error::Unsupported(
                    "Iris restoration runtime replay changed canonical identity".into(),
                ))
            }
        });
        if result.is_err() {
            (carrier.target, carrier.input_sizing, carrier.color_fix) = previous;
        }
        result
    }
    pub fn memory_class(&self) -> Result<RestorationMemoryClass> {
        let plan = self.plan()?;
        Ok(RestorationMemoryClass {
            budget_applied: plan.budget_applied,
            enlarged: plan.enlarged,
            single_pass: plan.single_pass(),
            color_fix: plan.color_fix,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DepthToken(String);

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DepthWire {
    source_width: u32,
    source_height: u32,
    source_capacity_bytes: u64,
    max_side: u32,
    patch: u32,
    num_train_timesteps: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DepthMemoryClass {
    pub resized: bool,
}

impl DepthWire {
    fn plan(&self) -> Result<gen_core_iris::depth::DepthInputPlan> {
        let bytes = u64::from(self.source_width)
            .checked_mul(u64::from(self.source_height))
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| Error::Unsupported("Iris depth source byte overflow".into()))?;
        if self.source_capacity_bytes < bytes || self.num_train_timesteps == 0 {
            return Err(Error::Unsupported(
                "invalid Iris depth source metadata".into(),
            ));
        }
        let resolution = if self.max_side == 0 {
            gen_core_iris::depth::DepthResolution::Native
        } else {
            gen_core_iris::depth::DepthResolution::Capped(self.max_side)
        };
        let plan = gen_core_iris::depth::plan_depth_input(
            self.source_width,
            self.source_height,
            resolution,
            self.patch as usize,
        )
        .map_err(Error::from)?;
        crate::depth::depth_host_bytes(&plan)?;
        Ok(plan)
    }
}

pub fn canonical_depth_token(
    spec: &mlx_gen::LoadSpec,
    actual: &gen_core_iris::depth::DepthRequest,
) -> Result<DepthToken> {
    let export = TaskExport::from_spec(spec, IrisTask::Depth, crate::depth::DEPTH_MODEL_ID)
        .map_err(Error::from)?;
    depth_input_plan(spec, actual)?;
    let wire = DepthWire {
        source_width: actual.image.width,
        source_height: actual.image.height,
        source_capacity_bytes: actual.image.pixels.capacity() as u64,
        max_side: actual.resolution.max_side(),
        patch: export.config.model.patch_size as u32,
        num_train_timesteps: export.config.flow.num_train_timesteps as u64,
    };
    DepthToken::from_token(&format!(
        "iris-depth-v1:{}",
        serde_json::to_string(&wire).map_err(|e| Error::Msg(e.to_string()))?
    ))
}

/// Native/current/source-max/patch-cap representatives use the actual source and actual export
/// patch. Unreachable resize classes are not manufactured with fake media.
pub fn depth_representatives(
    spec: &mlx_gen::LoadSpec,
    carrier: &gen_core_iris::depth::DepthRequest,
) -> Result<Vec<Result<DepthToken>>> {
    let actual = canonical_depth_token(spec, carrier)?;
    let base = actual.wire()?;
    let mut classes = std::collections::BTreeSet::from([actual.memory_class()?]);
    let mut representatives = vec![Ok(actual)];
    for max_side in [0, base.source_width.max(base.source_height), base.patch] {
        let mut wire = base.clone();
        wire.max_side = max_side;
        let token = serde_json::to_string(&wire)
            .map_err(|e| Error::Msg(e.to_string()))
            .and_then(|wire| DepthToken::from_token(&format!("iris-depth-v1:{wire}")));
        match token {
            Ok(token) => {
                if classes.insert(token.memory_class()?) {
                    representatives.push(Ok(token));
                }
            }
            Err(error) => representatives.push(Err(error)),
        }
    }
    Ok(representatives)
}

impl DepthToken {
    fn wire(&self) -> Result<DepthWire> {
        serde_json::from_str(
            self.0
                .strip_prefix("iris-depth-v1:")
                .ok_or_else(|| Error::Unsupported("invalid Iris depth prefix".into()))?,
        )
        .map_err(|e| Error::Msg(e.to_string()))
    }
    pub fn from_token(token: &str) -> Result<Self> {
        let wire: DepthWire = serde_json::from_str(
            token
                .strip_prefix("iris-depth-v1:")
                .ok_or_else(|| Error::Unsupported("not an Iris depth token".into()))?,
        )
        .map_err(|e| Error::Msg(e.to_string()))?;
        wire.plan()?;
        let canonical = format!(
            "iris-depth-v1:{}",
            serde_json::to_string(&wire).map_err(|e| Error::Msg(e.to_string()))?
        );
        if canonical != token {
            return Err(Error::Unsupported("noncanonical Iris depth token".into()));
        }
        Ok(Self(canonical))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
    pub fn to_value(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self.wire()?).map_err(|e| Error::Msg(e.to_string()))
    }
    pub fn plan(&self) -> Result<gen_core_iris::depth::DepthInputPlan> {
        self.wire()?.plan()
    }
    /// Apply max-side/native controls to the real source request without copying its RGB payload.
    /// Actual source/capacity and export patch/timestep metadata must match the captured token.
    pub fn apply_runtime_to(
        &self,
        spec: &mlx_gen::LoadSpec,
        carrier: &mut gen_core_iris::depth::DepthRequest,
    ) -> Result<()> {
        let wire = self.wire()?;
        wire.plan()?;
        let export = TaskExport::from_spec(spec, IrisTask::Depth, crate::depth::DEPTH_MODEL_ID)
            .map_err(Error::from)?;
        if (
            carrier.image.width,
            carrier.image.height,
            carrier.image.pixels.capacity() as u64,
        ) != (
            wire.source_width,
            wire.source_height,
            wire.source_capacity_bytes,
        ) || export.config.model.patch_size as u32 != wire.patch
            || export.config.flow.num_train_timesteps as u64 != wire.num_train_timesteps
        {
            return Err(Error::Unsupported(
                "Iris depth runtime recipe disagrees with actual source/artifact".into(),
            ));
        }
        let previous = carrier.resolution;
        carrier.resolution = if wire.max_side == 0 {
            gen_core_iris::depth::DepthResolution::Native
        } else {
            gen_core_iris::depth::DepthResolution::Capped(wire.max_side)
        };
        let result = canonical_depth_token(spec, carrier).and_then(|actual| {
            if actual == *self {
                Ok(())
            } else {
                Err(Error::Unsupported(
                    "Iris depth runtime replay changed canonical identity".into(),
                ))
            }
        });
        if result.is_err() {
            carrier.resolution = previous;
        }
        result
    }
    pub fn memory_class(&self) -> Result<DepthMemoryClass> {
        Ok(DepthMemoryClass {
            resized: self.plan()?.resized(),
        })
    }
}

/// Provider-owned actual scalar/indexed plans for metadata/capture/anchor basis. No image/tensor
/// allocation, tile-axis collection or copied global row rule is involved.
pub fn restoration_input_plan(
    spec: &mlx_gen::LoadSpec,
    actual: &mlx_gen::TransformRequest,
) -> Result<gen_core_iris::restoration::RestorationPlanMetadata> {
    let resources =
        RestorationResources::from_spec(spec, crate::restoration::MODEL_ID).map_err(Error::from)?;
    gen_core_iris::restoration::plan_request_metadata(
        actual,
        resources.settings.geometry(&resources.export.config),
        crate::restoration::MODEL_ID,
    )
    .map_err(Error::from)
}

pub fn depth_input_plan(
    spec: &mlx_gen::LoadSpec,
    actual: &gen_core_iris::depth::DepthRequest,
) -> Result<gen_core_iris::depth::DepthInputPlan> {
    let export = TaskExport::from_spec(spec, IrisTask::Depth, crate::depth::DEPTH_MODEL_ID)
        .map_err(Error::from)?;
    gen_core_iris::depth::validate_depth_request(actual).map_err(Error::from)?;
    gen_core_iris::depth::plan_depth_input(
        actual.image.width,
        actual.image.height,
        actual.resolution,
        export.config.model.patch_size,
    )
    .map_err(Error::from)
}

/// Price the actual original-source/scale/sizing/color-fix plan, not a fabricated processing source.
fn restoration_priced_plan(
    req: &IosPricingRequest<'_>,
    resources: &RestorationResources,
    token: Option<&RestorationToken>,
) -> Result<gen_core_iris::restoration::RestorationPlanMetadata> {
    if req.generation.is_some() || req.depth.is_some() {
        return Err(Error::Unsupported(
            "Iris restoration cannot carry generation/depth context".into(),
        ));
    }
    if let Some(token) = token {
        let plan = token.plan()?;
        if plan.patch != resources.export.config.model.patch_size as u32
            || plan.source.width != req.geometry.width
            || plan.source.height != req.geometry.height
            || req.geometry.batch != 1
            || req.geometry.frames != 1
            || req.geometry.reference_count != 0
        {
            return Err(Error::Unsupported(
                "Iris restoration replay disagrees with actual architecture/source key".into(),
            ));
        }
        return Ok(plan);
    }
    let actual = req.transform.ok_or_else(|| Error::Unsupported(
        "Iris restoration pricing requires the actual TransformRequest source/scale/sizing/color-fix context".into()
    ))?;
    let p = gen_core_iris::restoration::plan_request_metadata(
        actual,
        resources.settings.geometry(&resources.export.config),
        crate::restoration::MODEL_ID,
    )
    .map_err(Error::from)?;
    if actual.image.width != req.geometry.width
        || actual.image.height != req.geometry.height
        || req.geometry.batch != 1
        || req.geometry.frames != 1
        || req.geometry.reference_count != 0
    {
        return Err(Error::Unsupported(
            "Iris restoration context disagrees with source geometry".into(),
        ));
    }
    Ok(p)
}

/// The host-plane bytes a restoration request's Decode resident holds (the planes still allocated
/// when the last tile returns): host memory, which a device capture's footprint sees and MLX's
/// active memory does not.
#[cfg(test)]
pub(crate) fn restoration_decode_host_bytes(req: &IosPricingRequest<'_>) -> Result<u64> {
    let resources = RestorationResources::from_spec(req.spec, crate::restoration::MODEL_ID)
        .map_err(Error::from)?;
    let p = restoration_priced_plan(req, &resources, None)?;
    Ok(crate::restoration::host_plane_bytes_metadata(&p)?.live_at_last_tile)
}

fn restoration(
    req: &IosPricingRequest<'_>,
    opts: Pricing,
    resources: &RestorationResources,
    token: Option<&RestorationToken>,
) -> Result<Priced> {
    restoration_with_sources(req, opts, resources, token, None)
}

fn restoration_with_sources(
    req: &IosPricingRequest<'_>,
    opts: Pricing,
    resources: &RestorationResources,
    token: Option<&RestorationToken>,
    sources: Option<&GuardedSources<'_>>,
) -> Result<Priced> {
    let spec = req.spec;
    crate::memory_strategy::refuse_unloadable_quant(crate::restoration::MODEL_ID, spec)
        .map_err(Error::from)?;
    if !opts.assume_packed && sources.is_none() {
        crate::packed::refuse_dense_requested_as_packed(
            crate::restoration::MODEL_ID,
            &resources.export.weights_path(),
            spec.quantize,
        )?;
    }
    let streamed = spec.load_shape == LoadShape::DeferredMaterialization;
    let config = &resources.export.config;
    let owned_bb;
    let bb = if let Some(sources) = sources {
        &sources.bb
    } else {
        owned_bb = priced_backbone(
            config,
            &resources.export.weights_path(),
            "",
            spec.precision,
            spec.quantize,
            streamed,
            config.model.in_channels as u64,
            opts,
        )?;
        &owned_bb
    };
    let (states, prompt_floor) = if let Some(sources) = sources {
        (
            crate::stream::empty_prompt_upload_bytes(sources.metadata.empty_embedding_elements)?,
            materialized_load_bytes(&sources.empty_headers, |_| true)?,
        )
    } else {
        let states = held_states(&resources.export.empty_prompt_path())?;
        let prompt_floor = materialized_load_bytes(
            &crate::stream::headers(&resources.export.empty_prompt_path())?,
            |_| true,
        )?;
        (states, prompt_floor)
    };
    let owned_plan;
    let p = if let Some(GuardedSources {
        task: GuardedTask::Restoration { plan, .. },
        ..
    }) = sources
    {
        plan
    } else {
        owned_plan = restoration_priced_plan(req, resources, token)?;
        &owned_plan
    };
    let (th, tw) = if p.single_pass() {
        (p.padded.height as u64, p.padded.width as u64)
    } else {
        (p.tile as u64, p.tile as u64)
    };
    let host = crate::restoration::host_plane_bytes_metadata(p)?;
    let axes = p.axis_bytes().map_err(Error::from)?;
    let caller_rgb = if let Some(sources) = sources {
        sources.caller_bytes
    } else {
        match token {
            Some(token) => token.wire()?.source_capacity_bytes,
            None => req
                .transform
                .ok_or_else(|| Error::Unsupported("missing actual restoration context".into()))?
                .image
                .pixels
                .capacity() as u64,
        }
    };
    let readback = tensor_bytes(4, &[3, th, tw]);
    let tile_host_ambient = host
        .live_at_last_tile
        .checked_sub(readback)
        .expect("actual last-tile owner ledger contains its velocity readback");
    let window = export_window(streamed);
    // A streamed backbone holds a window only inside its forward (the tile phase).
    let rb = bb.resident + states;
    let mut pricing = Priced::default();
    pricing.pricing.phase_not_run(
        IosPhase::Conditioning,
        "Iris restoration loads shipped empty-prompt states, no text tower",
    );
    let basis = IosWorkBasis {
        forward_rows: 1,
        token_count: config.model.text_len as u64,
        model_width: tw as u32,
        model_height: th as u32,
        source_plane_bytes: caller_rgb,
        processed_plane_bytes: tensor_bytes(4, &[3, p.padded.height as u64, p.padded.width as u64]),
        output_plane_bytes: tensor_bytes(1, &[3, p.output.height as u64, p.output.width as u64]),
        patch_width: p.patch,
        patch_height: p.patch,
    };
    take_source_at(
        &mut pricing,
        IosPhase::Load,
        0,
        caller_rgb,
        &[
            (crate::restoration::quanta::LOAD, bb.floor + prompt_floor),
            (
                crate::stream::LOAD_TARGET_QUANTUM,
                bb.load_targets.checked_add(states).ok_or_else(|| {
                    Error::Unsupported("Iris restoration joint load byte overflow".into())
                })?,
            ),
        ],
        basis.clone(),
        "iris.restoration.load",
        "restoration::load raw HOST-inclusive floor then native targets plus explicit F32 upload",
        1,
    );
    take_source_at(
        &mut pricing,
        IosPhase::Load,
        0,
        caller_rgb + states + 4 * config.model.text_len as u64,
        &[
            (crate::model::BACKBONE_LOAD_QUANTUM, bb.floor),
            (crate::stream::LOAD_TARGET_QUANTUM, bb.load_targets),
        ],
        basis.clone(),
        "iris.restoration.backbone_load",
        "restoration::load decoded prompt outside the backbone joint guard",
        1,
    );
    take_source_at(
        &mut pricing,
        IosPhase::Load,
        bb.resident,
        caller_rgb,
        &[(crate::stream::EXPORT_TARGET_QUANTUM, states)],
        basis.clone(),
        "iris.restoration.load/upload",
        "evaluated backbone retained before fresh F32 prompt upload",
        1,
    );
    if axes != 0 {
        take_source_at(
            &mut pricing,
            IosPhase::Encode,
            rb,
            caller_rgb,
            &[(crate::restoration::quanta::PLAN, axes)],
            basis.clone(),
            "iris.restoration.axes",
            "restoration::plan indexed metadata before axis Vec collection",
            1,
        );
    }
    take_source_at(
        &mut pricing,
        IosPhase::Encode,
        rb,
        caller_rgb + axes,
        &[(crate::restoration::quanta::HOST_PLANES, host.peak())],
        basis.clone(),
        "iris.restoration.host_entry",
        "restore driver before clone/preprocess/canvas allocations",
        u64::from(p.processing.width) * u64::from(p.processing.height),
    );
    let tile = crate::restoration::tile_scratch(3 * th * tw);
    let mut forward = Vec::new();
    let _raw_max_carry = bb.facts.forward(
        1,
        th,
        tw,
        mlx_gen::AttentionPlan::UNBOUNDED,
        false,
        &mut forward,
    )?;
    let table = bb.facts.pixel_table(th, tw)?;
    let tiles = p.forward_count().map_err(Error::from)?;
    let mut representatives = vec![0, tiles - 1];
    if tiles > 2 {
        representatives.push(1);
    }
    representatives.sort_unstable();
    representatives.dedup();
    for tile_index in representatives {
        let cached_before = tile_index != 0;
        take_source_at(
            &mut pricing,
            IosPhase::Denoise,
            rb,
            caller_rgb + axes + tile_host_ambient + if cached_before { table } else { 0 },
            &[(crate::restoration::quanta::TILE, tile)],
            basis.clone(),
            &format!(
                "iris.restoration.tile/first{}/last{}",
                tile_index == 0,
                tile_index + 1 == tiles
            ),
            "run_restoration crop before velocity upload/forward/readback",
            th * tw,
        );
        for (operator, &(label, scratch)) in forward.iter().enumerate() {
            let index = forward[..operator]
                .iter()
                .filter(|point| point.0 == label)
                .count();
            let weight_window = if streamed {
                bb.facts.weight_window_at(label, index, window)?
            } else {
                0
            };
            let owned = bb.facts.held_at(1, th, tw, false, label, index, 0)?;
            let cached = cached_before
                || matches!(
                    label,
                    dit_q::PIXEL_POSITIONS
                        | dit_q::PIXEL_BLOCK
                        | dit_q::OUTPUT
                        | dit_q::OUTPUT_ROWS
                );
            let area = if matches!(
                label,
                dit_q::TEXT_LAYER_BLOCK
                    | dit_q::TEXT_POOL
                    | dit_q::TEXT_MASK
                    | dit_q::TEXT_REFINER_BLOCK
                    | dit_q::TEXT_OUTPUT
            ) {
                config.model.text_len as u64
            } else {
                th * tw
            };
            take_source_at(
                &mut pricing,
                IosPhase::Denoise,
                rb + weight_window,
                caller_rgb
                    + axes
                    + tile_host_ambient
                    + allocated(readback)
                    + 4
                    + owned
                    + if cached { table } else { 0 },
                &[(label, scratch)],
                basis.clone(),
                &format!(
                    "iris.restoration.forward/op{operator}/first{}/last{}",
                    tile_index == 0,
                    tile_index + 1 == tiles
                ),
                "restoration::velocity_body -> dit::forward_impl coincident owners",
                area,
            );
        }
        if streamed {
            for (stack, next_label) in [
                (Stack::Layer, dit_q::TEXT_LAYER_BLOCK),
                (Stack::Refiner, dit_q::TEXT_REFINER_BLOCK),
                (Stack::Trunk, dit_q::TRUNK_BLOCK),
                (Stack::Pixel, dit_q::PIXEL_BLOCK),
            ] {
                let reads = bb.facts.inventory.reads(stack);
                if reads.is_empty() {
                    continue;
                }
                let count = window.unwrap_or(reads.len()).clamp(1, reads.len());
                for first in (0..reads.len()).step_by(count) {
                    let end = (first + count).min(reads.len());
                    let owned = bb.facts.held_at(1, th, tw, false, next_label, first, 0)?;
                    let cached = cached_before || stack == Stack::Pixel;
                    take_source_at(
                        &mut pricing,
                        IosPhase::Denoise,
                        rb,
                        caller_rgb
                            + axes
                            + tile_host_ambient
                            + allocated(readback)
                            + 4
                            + owned
                            + if cached { table } else { 0 },
                        &[(
                            dit_q::BLOCK_READ,
                            crate::stream::window_read_bytes(&reads[first..end]),
                        )],
                        basis.clone(),
                        &format!(
                            "iris.restoration.read/{stack:?}/block{first}/first{}/last{}",
                            tile_index == 0,
                            tile_index + 1 == tiles
                        ),
                        "restoration backbone window entry before block read",
                        1,
                    );
                }
            }
        }
    }
    take_source_at(
        &mut pricing,
        IosPhase::Decode,
        rb,
        caller_rgb + axes + host.live_at_last_tile + table,
        &[(crate::restoration::quanta::POST_TILE, host.post_tile_new())],
        basis,
        "iris.restoration.post_tile",
        "actual last velocity returned; Gaussian/crop/readback still owned",
        u64::from(p.output.width) * u64::from(p.output.height),
    );
    Ok(pricing)
}

fn depth(
    req: &IosPricingRequest<'_>,
    opts: Pricing,
    export: &TaskExport,
    token: Option<&DepthToken>,
) -> Result<Priced> {
    depth_with_sources(req, opts, export, token, None)
}

/// Fresh targets created inside backbone construction: projection-bias compute casts, f32
/// norm/modulation/table casts, logical-width item targets, and depth-reducer compute casts.
/// Sum as a conservative load-interval upper, not a claim all targets coexist or an entry lower.
/// A raw file floor does not include a newly created target buffer.
/// Keys are Backbone's already-normalized logical names; no source reopen or string allocation.
fn backbone_constructor_output_bytes(
    headers: &std::collections::HashMap<String, Header>,
    precision: Precision,
    keep: impl Fn(&str) -> bool,
) -> Result<u64> {
    let view = headers.iter().map(|(name, h)| (name.as_str(), h)).collect();
    crate::stream::constructor_output_bytes(&view, compute_dtype(precision), keep)
}

// Depth uses the identical law, including its out-of-backbone reducer, with the actual
// eager/deferred selection. Keep this route adapter for the existing depth callers/tests.
fn depth_constructor_output_bytes(
    headers: &std::collections::HashMap<String, Header>,
    precision: Precision,
    streamed: bool,
) -> Result<u64> {
    backbone_constructor_output_bytes(headers, precision, |name| {
        !streamed || Stack::of(name).is_none()
    })
}

fn depth_with_sources(
    req: &IosPricingRequest<'_>,
    opts: Pricing,
    export: &TaskExport,
    token: Option<&DepthToken>,
    sources: Option<&GuardedSources<'_>>,
) -> Result<Priced> {
    let spec = req.spec;
    crate::memory_strategy::refuse_unloadable_quant(crate::depth::DEPTH_MODEL_ID, spec)
        .map_err(Error::from)?;
    if !opts.assume_packed && sources.is_none() {
        crate::packed::refuse_dense_requested_as_packed(
            crate::depth::DEPTH_MODEL_ID,
            &export.weights_path(),
            spec.quantize,
        )?;
    }
    let streamed = spec.load_shape == LoadShape::DeferredMaterialization;
    let config = &export.config;
    let input_channels = (config.model.in_channels + DEPTH_EXTRA_INPUT_CHANNELS) as u64;
    let owned_bb;
    let bb = if let Some(sources) = sources {
        &sources.bb
    } else {
        owned_bb = priced_backbone(
            config,
            &export.weights_path(),
            "pixel.",
            spec.precision,
            spec.quantize,
            streamed,
            input_channels,
            opts,
        )?;
        &owned_bb
    };
    let (states, prompt_floor) = if let Some(sources) = sources {
        (
            crate::stream::empty_prompt_upload_bytes(sources.metadata.empty_embedding_elements)?,
            materialized_load_bytes(&sources.empty_headers, |_| true)?,
        )
    } else {
        let states = held_states(&export.empty_prompt_path())?;
        let prompt_floor = materialized_load_bytes(
            &crate::stream::headers(&export.empty_prompt_path())?,
            |_| true,
        )?;
        (states, prompt_floor)
    };
    let window = export_window(streamed);
    // A streamed backbone holds a window only inside its forward.
    let db = bb.resident + states;
    if req.generation.is_some()
        || req.transform.is_some()
        || req.geometry.batch != 1
        || req.geometry.frames != 1
        || req.geometry.reference_count != 0
    {
        return Err(Error::Unsupported(
            "Iris depth has an incompatible work context".into(),
        ));
    }
    let input_plan = if let Some(GuardedSources {
        task: GuardedTask::Depth { plan, .. },
        ..
    }) = sources
    {
        *plan
    } else if let Some(token) = token {
        let wire = token.wire()?;
        if wire.patch != config.model.patch_size as u32 {
            return Err(Error::Unsupported(
                "Iris depth replay architecture patch mismatch".into(),
            ));
        }
        wire.plan()?
    } else {
        let actual = req.depth.ok_or_else(|| {
            Error::Unsupported(
                "Iris depth pricing requires actual DepthRequest source/resolution".into(),
            )
        })?;
        if sources.is_some() {
            gen_core_iris::depth::validate_depth_request(actual).map_err(Error::from)?;
            gen_core_iris::depth::plan_depth_input(
                actual.image.width,
                actual.image.height,
                actual.resolution,
                config.model.patch_size,
            )
            .map_err(Error::from)?
        } else {
            depth_input_plan(spec, actual)?
        }
    };
    if input_plan.source_width != req.geometry.width
        || input_plan.source_height != req.geometry.height
    {
        return Err(Error::Unsupported(
            "Iris depth context disagrees with source geometry".into(),
        ));
    }
    let (h, w) = (
        u64::from(input_plan.model_height),
        u64::from(input_plan.model_width),
    );
    let caller_rgb = if let Some(sources) = sources {
        sources.caller_bytes
    } else {
        match token {
            Some(token) => token.wire()?.source_capacity_bytes,
            None => req
                .depth
                .ok_or_else(|| Error::Unsupported("missing actual depth context".into()))?
                .image
                .pixels
                .capacity() as u64,
        }
    };
    let mut pricing = Priced::default();
    pricing.pricing.phase_not_run(
        IosPhase::Conditioning,
        "Iris depth uses shipped empty prompt, no text tower",
    );
    let host = crate::depth::depth_host_bytes(&input_plan)?;
    let basis = IosWorkBasis {
        forward_rows: 1,
        token_count: config.model.text_len as u64,
        model_width: w as u32,
        model_height: h as u32,
        source_plane_bytes: caller_rgb,
        processed_plane_bytes: host.input_live,
        output_plane_bytes: tensor_bytes(
            4,
            &[
                u64::from(input_plan.source_width),
                u64::from(input_plan.source_height),
            ],
        ),
        patch_width: config.model.patch_size as u32,
        patch_height: config.model.patch_size as u32,
    };
    take_source_at(
        &mut pricing,
        IosPhase::Load,
        0,
        caller_rgb,
        &[
            (crate::depth::quanta::LOAD, bb.floor + prompt_floor),
            (
                crate::stream::LOAD_TARGET_QUANTUM,
                bb.load_targets.checked_add(states).ok_or_else(|| {
                    Error::Unsupported("Iris depth joint load byte overflow".into())
                })?,
            ),
        ],
        basis.clone(),
        "iris.depth.load",
        "depth::load_depth_export original raw floor then actual joint targets",
        1,
    );
    let stage_plan = crate::stream::PendingBackbone {
        constructors: bb.constructor_outputs,
        evaluation: bb.evaluation_outputs,
        width_outputs: bb.width_outputs,
    };
    let late_targets = stage_plan
        .after_build_targets()?
        .checked_add(states)
        .ok_or_else(|| Error::Unsupported("Iris depth final targets byte overflow".into()))?;
    let completed = stage_plan.completed_before_evaluation()?;
    let late_sources = caller_rgb
        .checked_add(bb.read_sources_upper)
        .and_then(|bytes| bytes.checked_add(completed))
        .ok_or_else(|| Error::Unsupported("Iris depth final source byte overflow".into()))?;
    take_source_at(
        &mut pricing,
        IosPhase::Load,
        0,
        late_sources,
        &[(crate::stream::EXPORT_TARGET_QUANTUM, late_targets)],
        basis.clone(),
        "iris.depth.load/final_targets",
        "read inputs and completed getters retained; only pending targets plus F32 upload",
        1,
    );
    let input = crate::depth::depth_input_scratch(1, h as i32, w as i32);
    take_source_at(
        &mut pricing,
        IosPhase::Encode,
        db,
        caller_rgb,
        &[(crate::depth::quanta::HOST_PREP, host.preprocessing_peak)],
        basis.clone(),
        "iris.depth.host_preprocessing",
        "estimate_with before prepare_depth_input",
        h * w,
    );
    take_source_at(
        &mut pricing,
        IosPhase::Encode,
        db,
        caller_rgb + host.input_live,
        &[(crate::depth::quanta::RGB, host.rgb_copy)],
        basis.clone(),
        "iris.depth.rgb_copy",
        "depth::predict before Array::from_slice",
        h * w,
    );
    take_source_at(
        &mut pricing,
        IosPhase::Encode,
        db,
        caller_rgb + host.input_live + host.rgb_copy,
        &[(crate::depth::quanta::INPUT, input)],
        basis.clone(),
        "iris.depth.input",
        "depth_input_body RGB concat plus zero channel",
        h * w,
    );
    let held_input = allocated(4 * input_channels * h * w) + 4;
    let mut forward = Vec::new();
    let _raw_max_carry = bb.facts.forward(
        1,
        h,
        w,
        mlx_gen::AttentionPlan::UNBOUNDED,
        false,
        &mut forward,
    )?;
    let table = bb.facts.pixel_table(h, w)?;
    for (operator, &(label, scratch)) in forward.iter().enumerate() {
        let index = forward[..operator]
            .iter()
            .filter(|point| point.0 == label)
            .count();
        let weight_window = if streamed {
            bb.facts.weight_window_at(label, index, window)?
        } else {
            0
        };
        let owned = bb.facts.held_at(1, h, w, false, label, index, 0)?;
        let cached = matches!(
            label,
            dit_q::PIXEL_POSITIONS | dit_q::PIXEL_BLOCK | dit_q::OUTPUT | dit_q::OUTPUT_ROWS
        );
        let area = if matches!(
            label,
            dit_q::TEXT_LAYER_BLOCK
                | dit_q::TEXT_POOL
                | dit_q::TEXT_MASK
                | dit_q::TEXT_REFINER_BLOCK
                | dit_q::TEXT_OUTPUT
        ) {
            config.model.text_len as u64
        } else {
            h * w
        };
        take_source_at(
            &mut pricing,
            IosPhase::Denoise,
            db + weight_window,
            caller_rgb
                + host.input_live
                + host.rgb_copy
                + held_input
                + owned
                + if cached { table } else { 0 },
            &[(label, scratch)],
            basis.clone(),
            &format!("iris.depth.forward/op{operator}"),
            "depth_forward_body -> dit::forward_impl actual native owners",
            area,
        );
    }
    if streamed {
        for (stack, next_label) in [
            (Stack::Layer, dit_q::TEXT_LAYER_BLOCK),
            (Stack::Refiner, dit_q::TEXT_REFINER_BLOCK),
            (Stack::Trunk, dit_q::TRUNK_BLOCK),
            (Stack::Pixel, dit_q::PIXEL_BLOCK),
        ] {
            let reads = bb.facts.inventory.reads(stack);
            if reads.is_empty() {
                continue;
            }
            let count = window.unwrap_or(reads.len()).clamp(1, reads.len());
            for first in (0..reads.len()).step_by(count) {
                let end = (first + count).min(reads.len());
                let owned = bb.facts.held_at(1, h, w, false, next_label, first, 0)?;
                take_source_at(
                    &mut pricing,
                    IosPhase::Denoise,
                    db,
                    caller_rgb
                        + host.input_live
                        + host.rgb_copy
                        + held_input
                        + owned
                        + if stack == Stack::Pixel { table } else { 0 },
                    &[(
                        dit_q::BLOCK_READ,
                        crate::stream::window_read_bytes(&reads[first..end]),
                    )],
                    basis.clone(),
                    &format!("iris.depth.read/{stack:?}/block{first}"),
                    "depth streamed block window before read",
                    1,
                );
            }
        }
    }
    let channels = config.model.in_channels as u64;
    // The forward's input (`[1, C + extra, h, w]` f32) and model time stay live to its end.
    take_source_at(
        &mut pricing,
        IosPhase::Decode,
        db,
        caller_rgb
            + host.input_live
            + host.rgb_copy
            + held_input
            + table
            + allocated(4 * channels * h * w),
        &[(
            crate::depth::quanta::REDUCER,
            crate::depth::depth_reducer_bytes(
                channels * h * w,
                channels,
                compute_bytes(spec.precision),
            ),
        )],
        basis.clone(),
        "iris.depth.reducer",
        "depth_forward_body retains input/time/backbone output through reducer",
        h * w,
    );
    take_source_at(
        &mut pricing,
        IosPhase::Decode,
        db,
        caller_rgb + host.input_live + host.rgb_copy + table + allocated(4 * h * w),
        &[(crate::depth::quanta::HOST_MAP, host.map_copy)],
        basis.clone(),
        "iris.depth.host_map",
        "depth::predict native map and RGB before host to_vec",
        h * w,
    );
    if input_plan.resized() {
        take_source_at(
            &mut pricing,
            IosPhase::Decode,
            db,
            caller_rgb + host.input_live + host.map_copy + table,
            &[(crate::depth::quanta::HOST_POST, host.postprocess_new)],
            basis,
            "iris.depth.source_resize",
            "estimate_with depth_to_source input/prediction/source-output axes",
            u64::from(input_plan.source_width) * u64::from(input_plan.source_height),
        );
    }
    Ok(pricing)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_token_is_pure_lossless_and_replays_its_own_metadata() {
        let mut config = IrisConfig::from_dir(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-snapshot/iris"),
        )
        .unwrap();
        config.flow.shift = 4.123456789012345;
        config.flow.prediction = "x".into();
        config.flow.num_train_timesteps = 777;
        assert_ne!(config.flow.shift, config.flow.shift as f32 as f64);
        let mut request = GenerationRequest {
            prompt: "not in the wire".into(),
            width: 32,
            height: 16,
            seed: Some(42),
            count: 2,
            steps: Some(4),
            ..Default::default()
        };
        let original = resolved_generation_recipe(&config, &request, 0).unwrap();
        let token = canonical_token(&config, &request).unwrap();
        assert_eq!(SamplingToken::from_token(token.as_str()).unwrap(), token);
        assert_eq!(token.steps().unwrap(), 4);
        assert_eq!(
            token.to_value().unwrap()["flow_shift_bits"]
                .as_u64()
                .unwrap(),
            config.flow.shift.to_bits()
        );
        assert!(!token.as_str().contains("not in the wire"));
        assert!(!token
            .to_value()
            .unwrap()
            .as_object()
            .unwrap()
            .contains_key("seed"));
        let mut current = config.clone();
        current.flow.shift = 2.0;
        current.flow.prediction = "v".into();
        current.flow.num_train_timesteps = 1000;
        let mut carrier = request.clone();
        carrier.seed = Some(7);
        carrier.steps = Some(1);
        carrier.guidance = Some(1.0);
        let (metadata, replay) = token.apply_to(&current, &carrier).unwrap();
        assert_eq!(
            metadata.model, current.model,
            "actual architecture must not be fabricated"
        );
        assert_eq!(metadata.flow.shift.to_bits(), config.flow.shift.to_bits());
        assert_eq!(metadata.flow.prediction, "x");
        assert_eq!(metadata.flow.num_train_timesteps, 777);
        assert_eq!(
            replay.seed,
            Some(7),
            "recipe must preserve carrier seed/media"
        );
        assert_eq!(
            replay.scheduler_shift, None,
            "never round the stored f64 shift to request f32"
        );
        let resolved = resolved_generation_recipe(&metadata, &replay, 0).unwrap();
        assert_eq!(resolved.plan, original.plan);
        assert_eq!(resolved.params.prediction, original.params.prediction);
        assert_eq!(resolved.params.cfg_scale, original.params.cfg_scale);
        let mut forged = token.to_value().unwrap();
        forged["order"] = serde_json::json!(3);
        assert!(SamplingToken::from_token(&format!("iris-sampling-v1:{forged}")).is_err());
        let mut forged = token.to_value().unwrap();
        forged["claimed_class"] = serde_json::json!("euler-no-history");
        assert!(SamplingToken::from_token(&format!("iris-sampling-v1:{forged}")).is_err());
        let mut forged = token.to_value().unwrap();
        forged["flow_shift_bits"] = serde_json::json!(f64::NAN.to_bits());
        assert!(SamplingToken::from_token(&format!("iris-sampling-v1:{forged}")).is_err());
        let short = canonical_token(
            &config,
            &GenerationRequest {
                steps: Some(2),
                ..request.clone()
            },
        )
        .unwrap();
        assert!(!short
            .memory_class()
            .unwrap()
            .epoch_runs
            .iter()
            .any(|epoch| epoch[2]));
        let warm = canonical_token(
            &config,
            &GenerationRequest {
                steps: Some(3),
                ..request.clone()
            },
        )
        .unwrap();
        assert!(warm
            .memory_class()
            .unwrap()
            .epoch_runs
            .iter()
            .any(|epoch| epoch[2]));
        let representatives = sampling_representatives_in_place(&config, &mut request).unwrap();
        for order in [1, 2] {
            let classes: Vec<_> = representatives
                .iter()
                .map(|token| token.memory_class().unwrap())
                .filter(|class| class.order == order)
                .collect();
            assert!(classes
                .iter()
                .any(|class| class.epoch_runs.iter().all(|epoch| !epoch[0])));
            assert!(classes
                .iter()
                .any(|class| class.epoch_runs.iter().all(|epoch| epoch[0])));
            assert!(classes
                .iter()
                .any(|class| class.epoch_runs.iter().any(|epoch| epoch[0])
                    && class.epoch_runs.iter().any(|epoch| !epoch[0])));
        }
    }

    #[test]
    fn sampling_runtime_replay_uses_actual_config_and_lossless_request_controls() {
        let mut config = IrisConfig::from_dir(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-snapshot/iris"),
        )
        .unwrap();
        config.flow.shift = 1.000_000_000_1;
        let mut carrier = GenerationRequest {
            prompt: "runtime carrier".into(),
            width: 32,
            height: 16,
            steps: Some(3),
            sampler: Some("euler".into()),
            guidance: Some(1.0),
            count: 2,
            seed: Some(7),
            ..Default::default()
        };
        let token = canonical_token(&config, &carrier).unwrap();
        let payload = carrier.prompt.as_ptr();
        carrier.steps = Some(9);
        token.apply_runtime_to(&config, &mut carrier).unwrap();
        assert_eq!(canonical_token(&config, &carrier).unwrap(), token);
        assert_eq!(carrier.prompt.as_ptr(), payload);
        assert_eq!(
            (carrier.seed, carrier.count, carrier.scheduler_shift),
            (Some(7), 2, None)
        );
        let saved = SamplingTemplate::from_request(&carrier);
        let mut incompatible = config.clone();
        incompatible.flow.shift = 1.000_000_000_2;
        assert!(token.apply_runtime_to(&incompatible, &mut carrier).is_err());
        assert_eq!(SamplingTemplate::from_request(&carrier), saved);
        incompatible = config.clone();
        incompatible.flow.prediction = if config.flow.prediction == "v" {
            "x"
        } else {
            "v"
        }
        .into();
        assert!(token.apply_runtime_to(&incompatible, &mut carrier).is_err());
        incompatible = config.clone();
        incompatible.flow.num_train_timesteps += 1;
        assert!(token.apply_runtime_to(&incompatible, &mut carrier).is_err());
        carrier.scheduler_shift = Some(2.5);
        let overridden = canonical_token(&config, &carrier).unwrap();
        carrier.scheduler_shift = None;
        overridden.apply_runtime_to(&config, &mut carrier).unwrap();
        assert_eq!(carrier.scheduler_shift, Some(2.5));
        assert_eq!(canonical_token(&config, &carrier).unwrap(), overridden);
    }

    #[test]
    fn task_runtime_replay_applies_controls_without_replacing_actual_rgb() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let restore_spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(
            fixtures.join("tiny-snapshot/upscaler"),
        ));
        let depth_spec =
            mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(fixtures.join("tiny-depth")));
        let mut restore = mlx_gen::TransformRequest {
            image: mlx_gen::Image {
                width: 24,
                height: 16,
                pixels: vec![127; 24 * 16 * 3],
            },
            target: mlx_gen::TargetSize::Scale(1.25),
            input_sizing: mlx_gen::InputSizing::Original,
            color_fix: Some(false),
            ..Default::default()
        };
        let token = canonical_restoration_token(&restore_spec, &restore).unwrap();
        let rgb = restore.image.pixels.as_ptr();
        restore.target = mlx_gen::TargetSize::Scale(2.0);
        restore.input_sizing = mlx_gen::InputSizing::Budgeted;
        restore.color_fix = Some(true);
        token.apply_runtime_to(&restore_spec, &mut restore).unwrap();
        assert_eq!(
            canonical_restoration_token(&restore_spec, &restore).unwrap(),
            token
        );
        assert_eq!(restore.image.pixels.as_ptr(), rgb);
        let mut incompatible = token.wire().unwrap();
        incompatible.sigma_bits = (f64::from_bits(incompatible.sigma_bits) * 0.5).to_bits();
        let incompatible = RestorationToken::from_token(&format!(
            "iris-restoration-v1:{}",
            serde_json::to_string(&incompatible).unwrap()
        ))
        .unwrap();
        assert!(incompatible
            .apply_runtime_to(&restore_spec, &mut restore)
            .is_err());
        let mut depth = gen_core_iris::depth::DepthRequest {
            image: restore.image,
            resolution: gen_core_iris::depth::DepthResolution::Native,
            ..Default::default()
        };
        let token = canonical_depth_token(&depth_spec, &depth).unwrap();
        depth.resolution = gen_core_iris::depth::DepthResolution::Capped(8);
        token.apply_runtime_to(&depth_spec, &mut depth).unwrap();
        assert_eq!(canonical_depth_token(&depth_spec, &depth).unwrap(), token);
        assert_eq!(depth.image.pixels.as_ptr(), rgb);
        let mut incompatible = token.wire().unwrap();
        incompatible.num_train_timesteps += 1;
        let incompatible = DepthToken::from_token(&format!(
            "iris-depth-v1:{}",
            serde_json::to_string(&incompatible).unwrap()
        ))
        .unwrap();
        assert!(incompatible
            .apply_runtime_to(&depth_spec, &mut depth)
            .is_err());
    }

    #[test]
    fn raw_sampling_templates_preserve_unresolved_defaults_and_apply_only_controls_in_place() {
        let mut config = IrisConfig::from_dir(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-snapshot/iris"),
        )
        .unwrap();
        config.flow.shift = 1.000_000_000_1;
        let mut carrier = GenerationRequest {
            prompt: "carrier text is not template payload".into(),
            negative_prompt: Some("actual negative carrier".into()),
            width: 32,
            height: 16,
            count: 2,
            seed: Some(7),
            ..Default::default()
        };
        let raw = SamplingTemplate::from_request(&carrier);
        let wire = raw.to_token().unwrap();
        assert_eq!(SamplingTemplate::from_token(&wire).unwrap(), raw);
        assert!(!wire.contains(&carrier.prompt));
        let value = raw.to_value().unwrap();
        assert!(value["scheduler_shift_bits"].is_null());
        assert!(value["steps"].is_null());
        let templates = sampling_representative_templates(&carrier);
        assert_eq!(
            templates.last().unwrap(),
            &raw,
            "actual unresolved default carrier must survive listing"
        );
        let prompt_backing = carrier.prompt.as_ptr();
        let token = raw.apply_to(&config, &mut carrier).unwrap();
        assert_eq!(
            carrier.prompt.as_ptr(),
            prompt_backing,
            "applying controls replaced/cloned payload"
        );
        assert_eq!(
            (carrier.width, carrier.height, carrier.count, carrier.seed),
            (32, 16, 2, Some(7))
        );
        assert_eq!(
            carrier.scheduler_shift, None,
            "raw default must not become a lossy f32 shift"
        );
        assert_eq!(
            token.to_value().unwrap()["flow_shift_bits"],
            serde_json::json!(config.flow.shift.to_bits())
        );
        let negative_backing = carrier.negative_prompt.as_ref().unwrap().as_ptr();
        let representatives = sampling_representatives_in_place(&config, &mut carrier).unwrap();
        let actual_class = token.memory_class().unwrap();
        let representative = representatives
            .iter()
            .find(|candidate| candidate.memory_class().unwrap() == actual_class)
            .expect("actual default mechanism class disappeared");
        assert!(
            representative.steps().unwrap() < token.steps().unwrap(),
            "long actual default replaced a valid shorter capture representative"
        );
        assert_eq!(
            representative.to_value().unwrap()["flow_shift_bits"],
            serde_json::json!(config.flow.shift.to_bits()),
            "actual-default shift must win equal-length mechanism representatives"
        );
        assert_eq!(
            canonical_token(&config, &carrier).unwrap(),
            token,
            "full actual default provenance must still resolve losslessly"
        );
        assert_eq!(carrier.prompt.as_ptr(), prompt_backing);
        assert_eq!(
            carrier.negative_prompt.as_ref().unwrap().as_ptr(),
            negative_backing,
            "configured representatives replaced/dropped negative-prompt ownership"
        );
        assert_eq!(SamplingTemplate::from_request(&carrier), raw);
        let mut forged = raw.to_value().unwrap();
        forged["claimed_class"] = serde_json::json!("default");
        assert!(
            SamplingTemplate::from_token(&format!("iris-sampling-template-v1:{forged}")).is_err()
        );
    }

    #[test]
    fn trunk_parent_layout_tracks_dropped_final_text_and_previous_joined_parent() {
        let mut model = IrisConfig::from_dir(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-snapshot/iris"),
        )
        .unwrap()
        .model;
        model.dual_depth = 1;
        model.depth = 2;
        model.final_block_text = "drop".into();
        assert_eq!(trunk_parent_layout(&model, 0).unwrap(), (false, false));
        assert_eq!(trunk_parent_layout(&model, 1).unwrap(), (false, false));
        // The only single block drops text: image is new, text is the previous dual output.
        assert_eq!(trunk_parent_layout(&model, 2).unwrap(), (false, false));
        model.depth = 3;
        assert_eq!(trunk_parent_layout(&model, 2).unwrap(), (true, true));
        // Final image-only update leaves y retaining the whole penultimate joined parent.
        assert_eq!(trunk_parent_layout(&model, 3).unwrap(), (false, true));
        model.final_block_text = "keep".into();
        assert_eq!(trunk_parent_layout(&model, 3).unwrap(), (true, true));
        assert!(trunk_parent_layout(&model, 4).is_err());
    }

    #[test]
    fn repeated_payload_metadata_keeps_cold_steady_final_without_count_sized_vectors() {
        assert_eq!(
            payload_transition_indices(0).collect::<Vec<_>>(),
            Vec::<u64>::new()
        );
        assert_eq!(payload_transition_indices(1).collect::<Vec<_>>(), [0]);
        assert_eq!(payload_transition_indices(2).collect::<Vec<_>>(), [0, 1]);
        assert_eq!(payload_transition_indices(3).collect::<Vec<_>>(), [0, 1, 2]);
        assert_eq!(
            payload_transition_indices(u64::from(u32::MAX)).collect::<Vec<_>>(),
            [0, 1, u64::from(u32::MAX) - 1]
        );
    }

    #[test]
    fn grouped_backing_rounds_distinct_requests_before_multiplicity() {
        let group = NativeHeldRequest {
            owner: "distinct rows",
            bytes: 16_385,
            count: 2,
            birth: dit_q::PIXEL_BLOCK,
            block: Some(0),
            first_row: 0,
            liveness: "completed row outputs in parts",
            reuse: "unproved",
        };
        assert_eq!(group.priced(16_384).unwrap(), 65_536);
        assert_ne!(
            group.priced(16_384).unwrap(),
            mlx_gen::ios_quantum::fresh_metal_allocation_bytes(group.bytes * group.count, 16_384)
                .unwrap()
        );
        let overflow = NativeHeldRequest {
            count: u64::MAX,
            ..group
        };
        assert!(overflow.priced(16_384).is_err());
    }

    #[test]
    fn sampling_recipe_matches_iris_orders_guidance_history_and_serial_epochs() {
        let config = IrisConfig::from_dir(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-snapshot/iris"),
        )
        .unwrap();
        assert_eq!(
            crate::model::descriptor().capabilities.samplers,
            gen_core_iris::SAMPLERS
        );
        for sampler in gen_core_iris::SAMPLERS {
            for steps in [1, 2, 4] {
                for rows in [1usize, 2] {
                    for count in [1u32, 2] {
                        for guidance in [1.0f32, 3.0] {
                            let request = GenerationRequest {
                                prompt: if rows == 1 {
                                    "Iris recipe".into()
                                } else {
                                    String::new()
                                },
                                prompt_batch: if rows == 1 {
                                    Vec::new()
                                } else {
                                    vec!["a".into(), "b".into()]
                                },
                                width: 32,
                                height: 16,
                                count,
                                steps: Some(steps),
                                sampler: Some(sampler.into()),
                                guidance: Some(guidance),
                                ..Default::default()
                            };
                            let recipe = resolved_generation_recipe(&config, &request, 42).unwrap();
                            assert_eq!(recipe.params.order, if sampler == "euler" { 1 } else { 2 });
                            assert_eq!(recipe.plan.len(), steps as usize);
                            assert_eq!(recipe.plan.last().unwrap().t, 0.0);
                            assert!(matches!(
                                recipe.plan.last().unwrap().update,
                                gen_core_iris::DpmUpdate::First { cx: 0.0, c0: -1.0 }
                            ));
                            let epochs: Vec<_> = recipe.epochs().collect();
                            assert_eq!(epochs.len(), steps as usize * count as usize);
                            for epoch in epochs {
                                assert_eq!(epoch.guided, guidance != 1.0);
                                assert_eq!(
                                    epoch.forward_rows,
                                    rows as u64 * if guidance == 1.0 { 1 } else { 2 }
                                );
                                assert_eq!(
                                    epoch.history_rows,
                                    if epoch.index == 0 { 0 } else { rows as u64 },
                                    "Euler must retain actual previous x0 too"
                                );
                                assert_eq!(
                                    epoch.completed_rows,
                                    u64::from(epoch.serial) * rows as u64
                                );
                                assert_eq!(
                                    matches!(
                                        epoch.step.update,
                                        gen_core_iris::DpmUpdate::Second { .. }
                                    ),
                                    sampler == "dpmpp_2m"
                                        && epoch.index > 0
                                        && epoch.index + 1 < steps as usize
                                );
                            }
                            let replay =
                                resolved_generation_recipe(&config, &recipe.replay, 7).unwrap();
                            assert_eq!(replay.params, recipe.params);
                            assert_eq!(replay.plan, recipe.plan);
                            assert_eq!(replay.count, recipe.count);
                        }
                    }
                }
            }
        }
        let mixed = GenerationRequest {
            prompt: "mixed CFG".into(),
            width: 32,
            height: 16,
            steps: Some(4),
            guidance: Some(3.0),
            cfg_interval: Some((0.25, 0.75)),
            scheduler_shift: Some(1.0),
            ..Default::default()
        };
        let recipe = resolved_generation_recipe(&config, &mixed, 42).unwrap();
        let epochs: Vec<_> = recipe.epochs().collect();
        assert!(epochs.iter().any(|e| e.guided) && epochs.iter().any(|e| !e.guided));
        for epoch in epochs {
            assert_eq!(epoch.guided, 0.25 < epoch.step.s && epoch.step.s < 0.75);
        }
        let empty_interval = GenerationRequest {
            steps: Some(1),
            ..mixed
        };
        assert!(
            resolved_generation_recipe(&config, &empty_interval, 42).is_err(),
            "an interval with no actual guided evaluation remains refused"
        );
    }

    #[test]
    fn phase_compatibility_envelope_keeps_independent_winners_and_exact_windows() {
        for reverse in [false, true] {
            let mut priced = Priced::default();
            let mut points = [
                (900, 100, dit_q::TRUNK_BLOCK, 100, 4),
                (200, 200, dit_q::PIXEL_BLOCK, 800, 9),
            ];
            if reverse {
                points.reverse();
            }
            for (structural, held, label, scratch, area) in points {
                take_source_at(
                    &mut priced,
                    IosPhase::Denoise,
                    structural,
                    held,
                    &[(label, scratch)],
                    IosWorkBasis::default(),
                    "test/source",
                    "test/owner",
                    area,
                );
            }
            let phase = priced.pricing.phase(IosPhase::Denoise).unwrap();
            assert_eq!((phase.resident_bytes, phase.scratch_bytes), (1000, 800));
            assert_eq!(phase.binding, dit_q::PIXEL_BLOCK);
            assert_eq!(priced.phase_areas[&IosPhase::Denoise], 9);
            let mut windows: Vec<_> = priced
                .pricing
                .windows
                .iter()
                .map(|w| (w.resident_bytes, w.held_bytes, w.scratch_bytes))
                .collect();
            windows.sort_unstable();
            assert_eq!(windows, [(200, 200, 800), (900, 100, 100)]);
            assert_eq!(
                priced
                    .pricing
                    .windows
                    .iter()
                    .map(IosWindowPrice::logical_peak_bytes)
                    .max(),
                Some(1200)
            );
            assert_eq!(
                phase.peak_bytes(),
                1800,
                "the summary is an envelope, not a source window"
            );
        }
    }

    #[test]
    fn actual_zero_native_stage_keeps_coverage_without_inventing_scratch() {
        let mut priced = Priced::default();
        take_source_at(
            &mut priced,
            IosPhase::Load,
            12,
            3,
            &[(crate::adapters::ADAPTER_HOST_QUANTUM, 0)],
            IosWorkBasis::default(),
            "test/actual_zero_stage",
            "test/retained_owners",
            1,
        );
        assert_eq!(
            priced.quanta,
            [(IosPhase::Load, crate::adapters::ADAPTER_HOST_QUANTUM, 0)]
        );
        assert_eq!(priced.pricing.windows.len(), 1);
        assert_eq!(priced.pricing.windows[0].scratch_bytes, 0);
        assert_eq!(priced.pricing.windows[0].logical_peak_bytes(), 15);
        let phase = priced.pricing.phase(IosPhase::Load).unwrap();
        assert_eq!(phase.scratch_bytes, 0);
        assert_eq!(phase.resident_bytes, 15);
    }

    /// Every quantum label the crate checks is attributed to a phase.
    ///
    /// *Mutation that reds this:* dropping any arm of `ios_quantum_phase`.
    #[test]
    fn every_quantum_label_has_a_phase() {
        let labels = quantum_labels();
        assert!(labels.len() >= 30, "{labels:?}");
        for label in labels {
            assert!(ios_quantum_phase(label).is_some(), "{label} has no phase");
        }
        assert!(quantum_labels().contains(&mlx_gen::adapters::loader::ADAPTER_LOAD_QUANTUM));
        assert_eq!(
            ios_quantum_phase(mlx_gen::adapters::loader::ADAPTER_LOAD_QUANTUM),
            Some(IosPhase::Load)
        );
        assert_eq!(ios_quantum_phase("not an Iris quantum"), None);
    }
}

#[cfg(test)]
mod guarded_source_tests {
    use super::*;
    use mlx_gen::gen_core::{
        MemoryGeometry, MemoryNumericTier, MemorySelection, MemoryStrategy,
        MemoryStrategyParameters,
    };

    fn preparation_estimate() -> EstimatedSourcePreparation {
        // An explicitly synthetic envelope exercises capacity forwarding, not phone evidence.
        EstimatedSourcePreparation::new(
            123_456_789,
            SourcePreparationEstimateProvenance::BeforeRequestConstructionPeakWithMargin,
        )
        .unwrap()
    }

    #[test]
    fn estimated_replay_checks_capacity_before_parsing_without_changing_default() {
        assert!(EstimatedSourcePreparation::new(
            0,
            SourcePreparationEstimateProvenance::BeforeRequestConstructionPeakWithMargin,
        )
        .is_err());
        let estimate = preparation_estimate();
        assert_eq!(estimate.bytes(), 123_456_789);
        assert_eq!(
            estimate.provenance(),
            SourcePreparationEstimateProvenance::BeforeRequestConstructionPeakWithMargin
        );
        let mut calls = 0;
        let denied = check_replay_with_estimated_preparation(
            crate::model::MODEL_ID,
            "iris-sampling-v1:{invalid",
            estimate,
            &mut |bytes| {
                calls += 1;
                assert_eq!(bytes, estimate.bytes());
                Err(Error::Canceled)
            },
        );
        assert!(matches!(denied, Err(Error::Canceled)));
        assert_eq!(calls, 1);
        check_replay_with_estimated_preparation(
            crate::model::MODEL_ID,
            "iris-depth-v1:{}",
            estimate,
            &mut |_| panic!("wrong prefix must not parse or check an envelope"),
        )
        .unwrap();
        check_replay_with_estimated_preparation(
            crate::model::MODEL_ID,
            "iris-sampling-v1:{}",
            estimate,
            &mut |_| Ok(()),
        )
        .unwrap();
        assert!(check_replay_source_capacity_for_provider(
            crate::model::MODEL_ID,
            "iris-sampling-v1:{}",
            &mut |_| panic!("default generation source gap must still refuse first"),
        )
        .is_err());
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn estimated_preparation_preserves_second_check_and_after_read_identity() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metadata.json");
        std::fs::write(&path, b"{}").unwrap();
        let mut spec =
            mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(directory.path().to_path_buf()));
        let inventory = mlx_gen::gen_core::SourceInventory::capture(&spec).unwrap();
        spec.bind_source_inventory(inventory).unwrap();
        let estimate = preparation_estimate();
        let mut envelope_checks = 0;
        let denied = with_estimated_source_preparation(
            &spec,
            estimate,
            &mut |bytes| {
                if bytes == estimate.bytes() {
                    envelope_checks += 1;
                    if envelope_checks == 2 {
                        return Err(Error::Canceled);
                    }
                }
                Ok(())
            },
            || -> Result<()> { panic!("second capacity denial must precede parsing") },
        );
        assert!(matches!(denied, Err(Error::Canceled)));
        assert_eq!(envelope_checks, 2);
        let changed = with_estimated_source_preparation(&spec, estimate, &mut |_| Ok(()), || {
            std::fs::write(&path, b"{\"changed\":true}")?;
            Ok(())
        });
        assert!(
            changed.is_err(),
            "post-read identity failure must reject the result"
        );
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn estimated_generation_pricing_keeps_inventory_and_incomplete_host() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-snapshot");
        let mut spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(fixtures.join("iris")));
        spec.components.insert(
            "text_encoder".into(),
            mlx_gen::WeightsSource::Dir(fixtures.join("text_encoder")),
        );
        let work = GenerationRequest {
            prompt: "host metadata".into(),
            width: 16,
            height: 16,
            steps: Some(1),
            guidance: Some(1.0),
            ..Default::default()
        };
        let memory = GenerationMemory::default();
        let selected = selection();
        let request = |spec| IosPricingRequest {
            provider: crate::model::MODEL_ID,
            spec,
            mode: &MemoryMode::TextToImage,
            selection: &selected,
            execution: &memory,
            geometry: MemoryGeometry {
                width: 16,
                height: 16,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            generation: Some(&work),
            transform: None,
            depth: None,
            token_window: 0,
        };
        let estimate = preparation_estimate();
        assert!(matches!(
            pricing_facts_with_estimated_preparation(&request(&spec), None, estimate, &mut |_| {
                Err(Error::Canceled)
            }),
            Err(Error::Canceled)
        ));
        let missing =
            pricing_facts_with_estimated_preparation(&request(&spec), None, estimate, &mut |_| {
                Ok(())
            })
            .err()
            .unwrap();
        assert!(missing.to_string().contains("retained SourceInventory"));
        let mut bound_spec = spec.clone();
        let inventory = mlx_gen::gen_core::SourceInventory::capture(&bound_spec).unwrap();
        bound_spec.bind_source_inventory(inventory).unwrap();
        let req = request(&bound_spec);
        let mut estimate_checks = 0;
        let facts = pricing_facts_with_estimated_preparation(&req, None, estimate, &mut |bytes| {
            estimate_checks += usize::from(bytes == estimate.bytes());
            Ok(())
        })
        .unwrap();
        assert_eq!(estimate_checks, 2);
        let host = facts.host_analysis.unwrap();
        assert!(host.missing.contains(&Some(IrisHostGap::TokenizerRegex)));
        assert!(host.missing.contains(&Some(IrisHostGap::CapturedMetadata)));
        assert!(pricing_facts_with_source_capacity(&req, None, &mut |_| Ok(())).is_err());
        // A changed configured source cannot borrow the bound inventory.
        let mut changed = bound_spec.clone();
        changed.components.insert(
            "text_encoder".into(),
            mlx_gen::WeightsSource::Dir(fixtures.join("not-the-bound-encoder")),
        );
        assert!(pricing_facts_with_estimated_preparation(
            &request(&changed),
            None,
            estimate,
            &mut |_| Ok(()),
        )
        .is_err());
    }

    /// HOST-only fixture construction, NOT a served proof issuer. The test deliberately compares
    /// already-held source arithmetic against legacy loading arithmetic; guarded identity and
    /// capacity safety are tested separately at their original read boundaries.
    fn held_restoration<'a>(
        req: &IosPricingRequest<'_>,
        token: Option<&'a RestorationToken>,
    ) -> GuardedSources<'a> {
        let resources = RestorationResources::from_spec(req.spec, req.provider).unwrap();
        let plan = restoration_priced_plan(req, &resources, token).unwrap();
        let weights = mlx_gen::gen_core::weightsmeta::safetensors_file_tensor_locations(
            resources.export.weights_path(),
        )
        .unwrap();
        let empty = mlx_gen::gen_core::weightsmeta::safetensors_file_tensor_locations(
            resources.export.empty_prompt_path(),
        )
        .unwrap();
        let real: Vec<_> = weights
            .tensors
            .into_iter()
            .map(|location| location.header)
            .collect();
        let empty_headers: Vec<_> = empty
            .tensors
            .into_iter()
            .map(|location| location.header)
            .collect();
        let mut metadata = IrisMetadataFacts::default();
        metadata
            .capture_backbone(weights.file_len, &weights.header_json, &real)
            .unwrap();
        metadata
            .capture_empty(empty.file_len, &empty.header_json, &empty_headers)
            .unwrap();
        let bb = priced_backbone(
            &resources.export.config,
            &resources.export.weights_path(),
            "",
            req.spec.precision,
            req.spec.quantize,
            req.spec.load_shape == LoadShape::DeferredMaterialization,
            resources.export.config.model.in_channels as u64,
            Pricing::default(),
        )
        .unwrap();
        let caller_bytes = token
            .map(|token| token.wire().unwrap().source_capacity_bytes)
            .unwrap_or_else(|| req.transform.unwrap().image.pixels.capacity() as u64);
        GuardedSources {
            task: GuardedTask::Restoration { resources, plan },
            bb,
            raw_headers: real.into_iter().map(|h| (h.name.clone(), h)).collect(),
            empty_headers,
            metadata,
            replay: token.map(IrisReplay::Restoration),
            caller_bytes,
        }
    }

    fn selection() -> MemorySelection {
        MemorySelection {
            strategy: MemoryStrategy::Resident,
            parameters: MemoryStrategyParameters::default(),
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
        }
    }

    #[test]
    fn replay_preflight_preserves_prefix_rejection_and_generation_gap_priority() {
        let mut calls = 0;
        let mut capacity = |bytes| {
            assert!(bytes > 0);
            calls += 1;
            Ok(())
        };
        let wrong_family = "iris-depth-v1:{}";
        check_replay_source_capacity_for_provider(
            crate::restoration::MODEL_ID,
            wrong_family,
            &mut capacity,
        )
        .unwrap();
        let original = RestorationToken::from_token(wrong_family).unwrap_err();
        assert!(matches!(original, Error::Unsupported(ref reason)
            if reason == "not an Iris restoration token"));
        let generation = check_replay_source_capacity_for_provider(
            crate::model::MODEL_ID,
            "iris-sampling-v1:{}",
            &mut capacity,
        )
        .unwrap_err();
        assert!(generation
            .to_string()
            .contains("tokenizer/regex cold-construction"));
        assert!(generation.to_string().contains("(aa011)"));
        // Neither a known-invalid prefix nor the intentionally refused generation route is parsed.
        assert_eq!(calls, 0);
        check_replay_source_capacity_for_provider(
            crate::restoration::MODEL_ID,
            "iris-restoration-v1:{}",
            &mut |bytes| {
                assert!(bytes > 0);
                Ok(())
            },
        )
        .unwrap();
        assert!(RestorationToken::from_token("iris-restoration-v1:{}").is_err());
    }

    #[test]
    fn held_restoration_request_and_replay_preserve_native_and_host_recipes() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(
            fixtures.join("tiny-snapshot/upscaler"),
        ));
        let work = mlx_gen::TransformRequest {
            image: mlx_gen::Image {
                width: 16,
                height: 16,
                pixels: vec![128; 3 * 16 * 16],
            },
            target: mlx_gen::TargetSize::Scale(1.0),
            color_fix: Some(false),
            input_sizing: mlx_gen::InputSizing::Original,
            ..Default::default()
        };
        let mode = crate::memory_strategy::memory_mode(IrisTask::Restoration);
        let memory = GenerationMemory::default();
        let selected = selection();
        let req = IosPricingRequest {
            provider: crate::restoration::MODEL_ID,
            spec: &spec,
            mode: &mode,
            selection: &selected,
            execution: &memory,
            geometry: MemoryGeometry {
                width: 16,
                height: 16,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            generation: None,
            transform: Some(&work),
            depth: None,
            token_window: 0,
        };
        let actual = held_restoration(&req, None);
        let held = match &actual.task {
            GuardedTask::Restoration { resources, .. } => {
                restoration_with_sources(&req, Pricing::default(), resources, None, Some(&actual))
                    .unwrap()
            }
            _ => unreachable!(),
        };
        let legacy = ios_pricing(&req).unwrap();
        assert_eq!(held.pricing.phases, legacy.phases);
        assert_eq!(held.pricing.windows, legacy.windows);
        let old_native = native_ledger::cumulative_native_upper_any(&req, 16384).unwrap();
        let cpu = native_ledger::cumulative_native_upper_from_sources(
            &req,
            &actual,
            16384,
            native_ledger::NativeBackend::Cpu,
        )
        .unwrap();
        let metal = native_ledger::cumulative_native_upper_from_sources(
            &req,
            &actual,
            16384,
            native_ledger::NativeBackend::Metal,
        )
        .unwrap();
        assert_eq!(cpu.max(metal), old_native);
        assert_eq!(
            native_ledger::host_payload_growth_upper(&req, Some(&actual)).unwrap(),
            native_ledger::restoration_host_upper(&req).unwrap()
        );
        let token = canonical_restoration_token(&spec, &work).unwrap();
        let replay_req = IosPricingRequest {
            transform: None,
            ..req
        };
        let replay = held_restoration(&replay_req, Some(&token));
        assert_eq!(
            native_ledger::cumulative_native_upper_with_restoration_token(
                &replay_req,
                &token,
                16384,
                native_ledger::NativeBackend::Cpu,
                &replay
            )
            .unwrap(),
            cpu
        );
        assert_eq!(
            native_ledger::restoration_host_upper_with_token(&replay_req, &token, &replay).unwrap(),
            native_ledger::restoration_host_upper(&req).unwrap()
        );
    }

    #[test]
    fn no_facts_host_refusal_never_opens_the_supplied_source() {
        let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(
            Path::new("/this-Iris-source-must-not-be-read").to_owned(),
        ));
        let work = mlx_gen::TransformRequest {
            image: mlx_gen::Image {
                width: 16,
                height: 16,
                pixels: vec![0; 768],
            },
            target: mlx_gen::TargetSize::Scale(1.0),
            ..Default::default()
        };
        let mode = crate::memory_strategy::memory_mode(IrisTask::Restoration);
        let memory = GenerationMemory::default();
        let selected = selection();
        let req = IosPricingRequest {
            provider: crate::restoration::MODEL_ID,
            spec: &spec,
            mode: &mode,
            selection: &selected,
            execution: &memory,
            geometry: MemoryGeometry {
                width: 16,
                height: 16,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            generation: None,
            transform: Some(&work),
            depth: None,
            token_window: 0,
        };
        let analysis = host_analysis_without_facts(
            &req,
            Some(mlx_gen::ios_pricing::IosCallbackHost::NonAllocating),
        )
        .unwrap();
        assert_eq!(analysis.known, KnownHostFragments::default());
        let reason = analysis.refusal().to_string();
        assert!(reason.contains("complete execution-time parser/key/container allocation coverage"));
        assert!(reason.contains("SourceMetadataCapacity"));
        assert!(!reason.contains("this-Iris-source"));
        assert!(reason.len() < 2048);
    }
}

#[cfg(test)]
mod materialized_load_floor_tests {
    use super::*;

    // Raw logical fixture sum contrasts with the production per-buffer allocation law.
    fn stored(headers: &[Header], keep: impl Fn(&str) -> bool) -> u64 {
        headers
            .iter()
            .filter(|h| keep(&h.name))
            .fold(0u64, |a, h| a.saturating_add(h.data_bytes))
    }

    fn core_codes(name: &str, bits: usize) -> Header {
        // Exact production packing: 384 output rows, 64 inputs, u32 code words.
        let words = 64 * bits / 32;
        Header {
            name: name.into(),
            dtype: Stored::U32,
            shape: vec![384, words],
            data_bytes: (384 * words * 4) as u64,
        }
    }

    #[test]
    fn q8_core_code_operands_round_separately_before_load_sum() {
        let headers = [
            core_codes("modulation_cores.adaln_img.weight", 8),
            core_codes("modulation_cores.adaln_txt.weight", 8),
        ];
        let page = crate::retention::PAGE_BYTES;
        assert_eq!(headers[0].data_bytes, page + page / 2);
        assert_eq!(stored(&headers, |_| true), 3 * page);
        assert_eq!(
            materialized_load_bytes(&headers, |_| true).unwrap(),
            4 * page
        );
    }

    #[test]
    fn q4_codes_stay_exact_and_streamed_stack_headers_are_not_loaded() {
        let headers = [
            core_codes("modulation_cores.adaln_img.weight", 4),
            core_codes("modulation_cores.adaln_txt.weight", 4),
            core_codes("blocks.0.attn.qkv.weight", 8),
        ];
        let skeleton = |name: &str| Stack::of(name).is_none();
        assert_eq!(
            materialized_load_bytes(&headers, skeleton).unwrap(),
            stored(&headers, skeleton)
        );
        assert!(
            materialized_load_bytes(&headers, |_| true).unwrap()
                > materialized_load_bytes(&headers, skeleton).unwrap()
        );
    }

    #[test]
    fn materialized_load_rejects_rounding_and_sum_overflow() {
        let mut header = core_codes("modulation_cores.adaln_img.weight", 8);
        header.data_bytes = u64::MAX;
        assert!(materialized_load_bytes(std::slice::from_ref(&header), |_| true).is_err());
        header.data_bytes = u64::MAX - (crate::retention::PAGE_BYTES - 1);
        let headers = [header, core_codes("modulation_cores.adaln_txt.weight", 8)];
        assert!(materialized_load_bytes(&headers, |_| true).is_err());
    }
}

#[cfg(test)]
mod resident_text_walk_tests {
    use super::*;

    #[test]
    fn tower_rope_is_a_model_owned_row_not_a_file_projection() {
        let dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-snapshot/text_encoder");
        let cfg = mlx_llm::ModelConfig::from_dir(&dir).unwrap();
        assert_eq!(
            tower_rope_bytes(&cfg).unwrap(),
            allocated(4 * cfg.rotary_dim() as u64)
        );
        assert!(tower_rope_bytes(&cfg).unwrap() > 0);
    }

    #[test]
    fn resident_walk_prices_every_hidden_state_and_each_layer_kv_pair() {
        let dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-snapshot/text_encoder");
        let cfg = mlx_llm::ModelConfig::from_dir(&dir).unwrap();
        let tokens = 41;
        let state = allocated(2 * tokens * cfg.hidden_size as u64);
        let block = mlx_llm::primitives::kv_cache::KV_BLOCK_TOKENS as u64;
        let one_cache = allocated(
            2 * tokens.div_ceil(block) * block * cfg.num_kv_heads as u64 * cfg.head_dim as u64,
        );
        assert_eq!(
            resident_text_walk_bytes(&cfg, tokens).unwrap(),
            (cfg.num_layers as u64 + 1) * state
                + 2 * cfg.num_layers as u64 * one_cache
                + allocated(4 * tokens)
                + 2 * allocated(2 * tokens * cfg.rotary_dim() as u64)
        );
        assert!(resident_text_walk_bytes(&cfg, tokens).unwrap() > state);
    }

    #[test]
    fn resident_walk_rounds_source_operands_and_rejects_overflow() {
        let dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-snapshot/text_encoder");
        let cfg = mlx_llm::ModelConfig::from_dir(&dir).unwrap();
        let tokens = 257;
        let state = allocated(2 * tokens * cfg.hidden_size as u64);
        assert!(state > 2 * tokens * cfg.hidden_size as u64);
        let walk = resident_text_walk_bytes(&cfg, tokens).unwrap();
        assert!(walk >= (cfg.num_layers as u64 + 1) * state);
        assert!(resident_text_walk_bytes(&cfg, u64::MAX).is_err());
    }
}

#[cfg(test)]
mod depth_constructor_output_tests {
    use super::*;

    fn h(name: &str, dtype: Stored, shape: &[usize], bytes: u64) -> Header {
        Header {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            data_bytes: bytes,
        }
    }

    #[test]
    fn constructor_target_is_compared_with_post_read_dtype() {
        // A broadcastable matrix-shaped modulation bias is narrowed by the read policy, then
        // l.f32 restores it to f32. Its stored f32 dtype does NOT make the constructor an alias.
        let header = h("blocks.0.adaln.bias", Stored::F32, &[1, 384], 1536);
        let headers = [(header.name.clone(), header)].into_iter().collect();
        assert_eq!(
            depth_constructor_output_bytes(&headers, Precision::Bf16, false).unwrap(),
            1536
        );
    }

    #[test]
    fn constructor_outputs_include_bias_norm_and_reducer_casts_not_file_aliases() {
        let headers = [
            h("linear.weight", Stored::U32, &[384, 8], 12288),
            h("linear.bias", Stored::F32, &[384], 1536),
            h("norm.weight", Stored::BF16, &[64], 128),
            h("depth_reducer.weight", Stored::F32, &[1, 3, 1, 1], 12),
            h("depth_reducer.bias", Stored::F32, &[1], 4),
        ]
        .into_iter()
        .map(|h| (h.name.clone(), h))
        .collect();
        assert_eq!(
            depth_constructor_output_bytes(&headers, Precision::Bf16, false).unwrap(),
            // No .scales companion: the U32 matrix is dense and the constructor casts it.
            384 * 8 * 2 + 384 * 2 + 64 * 4 + 3 * 2 + 2
        );
    }

    #[test]
    fn streamed_constructor_upper_excludes_unloaded_stack_parameters() {
        let headers = [
            h("blocks.0.proj.weight", Stored::BF16, &[64, 64], 8192),
            h("blocks.0.proj.bias", Stored::F32, &[64], 256),
            h("depth_reducer.bias", Stored::F32, &[1], 4),
        ]
        .into_iter()
        .map(|h| (h.name.clone(), h))
        .collect();
        assert_eq!(
            depth_constructor_output_bytes(&headers, Precision::Bf16, true).unwrap(),
            2
        );
        assert_eq!(
            depth_constructor_output_bytes(&headers, Precision::Bf16, false).unwrap(),
            130
        );
    }

    #[test]
    fn depth_packing_forecast_keeps_outside_prefix_reducer_verbatim() {
        let source = [
            h("depth_reducer.weight", Stored::F32, &[1, 3, 1, 1], 12),
            h("pixel.proj.weight", Stored::F32, &[64, 64], 16384),
        ];
        let packed = emulate_packed_backbone(&source, "pixel.", Quant::Q4).unwrap();
        let reducer = packed
            .iter()
            .find(|h| h.name == "depth_reducer.weight")
            .unwrap();
        assert_eq!(reducer.dtype, Stored::F32);
        assert_eq!(reducer.shape, [1, 3, 1, 1]);
        assert_eq!(reducer.data_bytes, 12);
        assert!(packed.iter().any(|h| h.name == "pixel.proj.scales"));
    }
}

#[cfg(test)]
mod shared_load_target_tests {
    use super::*;

    fn h(name: &str, dtype: Stored, shape: &[usize], bytes: u64) -> Header {
        Header {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            data_bytes: bytes,
        }
    }

    #[test]
    fn shared_backbone_targets_honor_packed_bias_post_read_norm_and_keep() {
        let headers = [
            h("p.weight", Stored::U32, &[384, 8], 12288),
            h("p.scales", Stored::BF16, &[384, 1], 768),
            h("p.bias", Stored::F32, &[384], 1536),
            h("norm.weight", Stored::BF16, &[64], 128),
            h("blocks.0.adaln.bias", Stored::F32, &[1, 384], 1536),
        ]
        .into_iter()
        .map(|header| (header.name.clone(), header))
        .collect();
        assert_eq!(
            backbone_constructor_output_bytes(&headers, Precision::Bf16, |name| Stack::of(name)
                .is_none())
            .unwrap(),
            384 * 2 + 64 * 4
        );
        assert_eq!(
            backbone_constructor_output_bytes(&headers, Precision::Bf16, |_| true).unwrap(),
            384 * 2 + 64 * 4 + 384 * 4
        );
        // Packed bias's constructor target is BF16, independent of a bare-linear target.
        assert_eq!(
            backbone_constructor_output_bytes(&headers, Precision::Fp32, |name| name == "p.bias")
                .unwrap(),
            384 * 2
        );
    }

    #[test]
    fn dense_fp32_identity_targets_do_not_charge_raw_sources_again() {
        let headers = [
            h("p.weight", Stored::F32, &[64, 64], 16384),
            h("p.bias", Stored::F32, &[64], 256),
            h("norm.weight", Stored::F32, &[64], 256),
            h("y_pos_embedding", Stored::F32, &[1, 10, 64], 2560),
        ]
        .into_iter()
        .map(|header| (header.name.clone(), header))
        .collect();
        assert_eq!(
            backbone_constructor_output_bytes(&headers, Precision::Fp32, |_| true).unwrap(),
            0
        );
        assert_eq!(
            backbone_constructor_output_bytes(&headers, Precision::Bf16, |_| true).unwrap(),
            128
        );
    }

    #[test]
    fn scalar_and_matrix_width_targets_follow_actual_post_read_dtype() {
        let headers = [
            h("a.in_features", Stored::I32, &[1], 4),
            h("b.in_features", Stored::I64, &[], 8),
            h("c.in_features", Stored::I32, &[1, 1], 4),
        ]
        .into_iter()
        .map(|header| (header.name.clone(), header))
        .collect();
        assert_eq!(
            backbone_constructor_output_bytes(&headers, Precision::Bf16, |_| true).unwrap(),
            8
        );
        assert_eq!(
            backbone_constructor_output_bytes(&headers, Precision::Fp32, |_| true).unwrap(),
            8
        );
    }

    #[test]
    fn optional_projection_bias_has_no_target_when_absent() {
        let headers = [h("p.weight", Stored::BF16, &[64, 64], 8192)]
            .into_iter()
            .map(|header| (header.name.clone(), header))
            .collect();
        assert_eq!(
            backbone_constructor_output_bytes(&headers, Precision::Bf16, |_| true).unwrap(),
            0
        );
    }

    #[test]
    fn tower_targets_price_dense_casts_and_model_rotary_row() {
        let headers = [
            h("embed.weight", Stored::F32, &[25, 32], 3200),
            h("norm.weight", Stored::F32, &[32], 128),
        ];
        assert_eq!(
            tower_load_output_bytes(&headers, |_| true, 64).unwrap().1,
            1600 + 64 + 64
        );
        let same_dtype = [
            h("embed.weight", Stored::BF16, &[25, 32], 1600),
            h("norm.weight", Stored::BF16, &[32], 64),
        ];
        assert_eq!(
            tower_load_output_bytes(&same_dtype, |_| true, 64)
                .unwrap()
                .1,
            64
        );
        assert_eq!(
            tower_load_output_bytes(&headers, |_| false, 64).unwrap().1,
            64
        );
    }

    #[test]
    fn tower_rounding_is_per_source_and_preserved_pack_parts_are_not_cast() {
        let headers = [
            h(
                "model.language_model.layers.0.self_attn.q_proj.weight",
                Stored::F32,
                &[384, 16],
                24576,
            ),
            h(
                "model.language_model.layers.1.self_attn.k_proj.weight",
                Stored::U32,
                &[64, 16],
                4096,
            ),
            h(
                "model.language_model.layers.1.self_attn.k_proj.scales",
                Stored::F32,
                &[64, 1],
                256,
            ),
            h(
                "model.language_model.layers.1.self_attn.k_proj.biases",
                Stored::BF16,
                &[64, 1],
                128,
            ),
        ];
        // Codes stay stored; the F32 affine scale gets a distinct BF16 target, while the
        // already-BF16 affine bias aliases. Round each owning request before adding.
        let expected = (allocated(24576) - 24576) + allocated(12288) + allocated(128) + 64;
        assert_eq!(
            tower_load_output_bytes(&headers, |_| true, 64).unwrap().1,
            expected
        );
    }

    #[test]
    fn same_width_different_dtype_is_a_fresh_target_not_an_alias() {
        let headers = [h("norm.weight", Stored::F16, &[32], 64)];
        assert_eq!(
            tower_load_output_bytes(&headers, |_| true, 0).unwrap().1,
            64
        );
    }

    #[test]
    fn shared_target_accumulators_reject_overflow() {
        let header = h(
            "blocks.0.adaln.bias",
            Stored::BF16,
            &[usize::MAX, usize::MAX],
            0,
        );
        let headers = [(header.name.clone(), header)].into_iter().collect();
        assert!(backbone_constructor_output_bytes(&headers, Precision::Bf16, |_| true).is_err());
        let headers = [h("norm.weight", Stored::BF16, &[32], u64::MAX)];
        assert!(tower_load_output_bytes(&headers, |_| true, 0).is_err());
    }
}

#[cfg(test)]
mod pending_adapter_owner_tests {
    use super::*;
    use crate::adapters::{
        AdapterFileReadFacts, AdapterLoadAllocation, ProjectionAdapterLoadFacts,
    };
    use mlx_gen::adapters::facts::{AdapterFacts, TensorFacts};

    fn file(source_bytes: u64) -> AdapterFileReadFacts {
        AdapterFileReadFacts {
            path: "metadata-only-unused.safetensors".into(),
            source_bytes,
            file_floor_bytes: source_bytes,
            alpha_read_scratch_bytes: 0,
            graph_constructor_bytes: 8,
        }
    }

    #[test]
    fn pending_adapter_inputs_are_not_replaced_by_future_delta_owners() {
        let load = ProjectionAdapterLoadFacts {
            retained: vec![
                AdapterFacts::Materialized {
                    delta: TensorFacts::materialized_matrix(8, 8, mlx_rs::Dtype::Bfloat16),
                    disabled: false,
                },
                AdapterFacts::Lora {
                    a: TensorFacts::materialized_matrix(8, 4, mlx_rs::Dtype::Bfloat16),
                    b: TensorFacts::materialized_matrix(4, 8, mlx_rs::Dtype::Bfloat16),
                    disabled: false,
                },
            ],
            allocations: vec![
                AdapterLoadAllocation {
                    operation: "new retained delta",
                    shape: vec![8, 8],
                    dtype: mlx_rs::Dtype::Bfloat16,
                    minimum_bytes: 0,
                    retained_output: true,
                },
                AdapterLoadAllocation {
                    operation: "temporary reconstruction",
                    shape: vec![4, 64],
                    dtype: mlx_rs::Dtype::Bfloat16,
                    minimum_bytes: 0,
                    retained_output: false,
                },
            ],
            ..Default::default()
        };
        let loads = [("projection".into(), load)].into_iter().collect();
        let (before, after) = adapter_pending_held_bytes(4, &[file(4096)], &loads).unwrap();
        // Before build: all raw factors, existing graph scalars and the packed host scalar.
        // After build: retain those owners without retirement credit; only the new output joins.
        assert_eq!(before, 4 + 4096 + 8);
        assert_eq!(after, before + 8 * 8 * 2);
        assert!(
            before > 8 * 8 * 2,
            "a small future delta is not the raw factor inventory"
        );
        assert_eq!(
            after - before,
            loads["projection"].new_retained_bytes().unwrap()
        );
        assert!(
            loads["projection"].retained_bytes().unwrap() > after - before,
            "retained LoRA source aliases are not independent new output owners"
        );
    }

    #[test]
    fn pending_adapter_owner_sums_are_checked_and_empty_sources_stay_empty() {
        let loads = std::collections::BTreeMap::new();
        assert_eq!(adapter_pending_held_bytes(0, &[], &loads).unwrap(), (0, 0));
        assert!(adapter_pending_held_bytes(4, &[file(u64::MAX)], &loads).is_err());
    }
}
