//! Cold, gross physical fit for ordered multi-character SCAIL animation.
//!
//! A witness comes from a completed full-provider run on the worker's hardware/runtime
//! class. The caller authenticates that provenance. The engine owns the immutable artifact,
//! geometry, native algorithm and liveness checks. A sample from another numerical
//! realization, or an incomparable workspace, supplies no price.

use crate::{
    Error, GenerationRequest, LoadSpec, MemoryBudgetDomain, PinnedWeightsFile, RenderLoadBinding,
    Result, VideoRenderPlan, WeightsSource,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::Cell;

const MODEL: &str = "scail2_14b";
const MEMBERS: [&str; 6] = [
    "config.json",
    "tokenizer.json",
    "dit.safetensors",
    "t5_encoder.safetensors",
    "clip.safetensors",
    "vae.safetensors",
];

fn unsupported(reason: &str) -> Error {
    Error::Unsupported(format!("scail2 plural fit: {reason}"))
}
fn add(x: u64, y: u64) -> Result<u64> {
    x.checked_add(y)
        .ok_or_else(|| unsupported("physical byte count overflow"))
}
fn mul(x: u64, y: u64) -> Result<u64> {
    x.checked_mul(y)
        .ok_or_else(|| unsupported("physical byte count overflow"))
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Scail2FitBackend {
    Mlx,
    Cuda,
}

/// Physical dimensions compared componentwise with one measured run. No source values,
/// prompt text, FPS, seed or shift participate in cross-request capacity reuse.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Scail2FitAxes {
    pub backend: Scail2FitBackend,
    pub artifact_sha256: String,
    pub load_policy: String,
    pub execution_policy: Vec<String>,
    pub width: u32,
    pub height: u32,
    pub references: u32,
    pub windows: Vec<Scail2FitWindow>,
    pub source_bytes: u64,
    pub source_resize_bytes: u64,
    pub schedule_bytes: u64,
    pub history_order: u32,
    pub classifier_free_guidance: bool,
    pub floor_host_bytes: u64,
    pub floor_device_bytes: u64,
    pub floor_unified_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Scail2FitWindow {
    pub latent_frames: u32,
    pub history_frames: usize,
    pub decoded_frames: u32,
    pub tokens: u64,
    pub self_attention_query_rows: u64,
    pub self_attention_score_elements: u64,
}

/// One completed native run. Gross peaks include original RGB carriers and cold model
/// materialization. Discrete host/device peaks must originate in the same run.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scail2FitWitness {
    pub axes: Scail2FitAxes,
    pub gross_host_bytes: u64,
    pub gross_device_bytes: u64,
    pub gross_unified_bytes: u64,
    pub mlx_decode_budget_bytes: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct Scail2FitProfile {
    axes: Scail2FitAxes,
    load: RenderLoadBinding,
    pins: Vec<PinnedWeightsFile>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scail2FitPrice {
    pub host_bytes: u64,
    pub device_bytes: u64,
    pub unified_bytes: u64,
    mlx_decode_budget_bytes: Option<u64>,
}

thread_local! {
    static DECODE_CEILING: Cell<Option<u64>> = const {Cell::new(None)};
}

/// Read by the registered MLX provider on the generation thread. Nested runs restore the
/// previous ceiling, including when cancellation or an unwinding caller exits the run.
pub fn mlx_decode_budget_bytes() -> Option<u64> {
    DECODE_CEILING.get()
}

impl Scail2FitPrice {
    pub fn with_execution<T>(&self, run: impl FnOnce() -> T) -> T {
        struct Reset(Option<u64>);
        impl Drop for Reset {
            fn drop(&mut self) {
                DECODE_CEILING.set(self.0);
            }
        }
        let prior = DECODE_CEILING.get();
        let ceiling = match (prior, self.mlx_decode_budget_bytes) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let _reset = Reset(prior);
        DECODE_CEILING.set(ceiling);
        run()
    }
}

fn execution_policy(backend: Scail2FitBackend) -> Vec<String> {
    // Native scheduler aliases on Candle share UniPC. The policy below describes only
    // actual native memory switches. Other environment/device/build identity belongs to
    // the worker's authenticated hardware/runtime row.
    match backend {
        Scail2FitBackend::Mlx => [
            "SCAIL2_COMPUTE_BF16",
            "MLX_GEN_WAN_FFN_SEQ_CHUNK",
            "MLX_GEN_WAN_ATTN_QUERY_CHUNK",
            "MLX_GEN_WAN_EVAL_PER_BLOCK",
        ]
        .iter()
        .map(|name| format!("{name}={}", std::env::var(name).unwrap_or_default()))
        .collect(),
        Scail2FitBackend::Cuda => Vec::new(),
    }
}
fn resize_bytes(image: &crate::Image, width: u32, height: u32) -> Result<u64> {
    if (image.width, image.height) == (width, height) {
        return Ok(0);
    }
    // Both resize implementations use CPU f32 input, a contiguous CHW copy, a horizontal
    // pass [source height, target width], an RGB output and a copy to a backend tensor.
    // MLX also keeps the initializing pixel Vec and original Array in image_to_chw.
    // Count the complete larger path even when the smaller CPU-only path aliases a view.
    let source = mul(u64::from(image.width), u64::from(image.height))?;
    let output = mul(u64::from(width), u64::from(height))?;
    let horizontal = mul(u64::from(image.height), u64::from(width))?;
    // Bilinear uses two taps and bicubic four. Each axis owns a Vec per pixel and four
    // (index, f64) pairs; allowance uses local native layouts.
    let axes = u64::from(width) + u64::from(height);
    let taps = mul(
        axes,
        (std::mem::size_of::<Vec<(usize, f64)>>() + 4 * std::mem::size_of::<(usize, f64)>()) as u64,
    )?;
    add(
        add(mul(source, 4 * 3 * 4)?, mul(output, 2 * 3 * 4)?)?,
        add(mul(horizontal, 4)?, taps)?,
    )
}

fn source_pins(spec: &LoadSpec) -> Result<Vec<PinnedWeightsFile>> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(unsupported("requires a complete snapshot"));
    };
    MEMBERS
        .iter()
        .map(|name| root.join(name))
        .chain(spec.adapters.iter().map(|a| a.path.clone()))
        .map(|p| {
            let abs = std::path::absolute(&p)?;
            let retained = spec
                .source_inventory()
                .map(|inv| inv.member_pin(&abs))
                .transpose()?
                .flatten()
                .or_else(|| spec.prepared_file_pins().get(&abs));
            match retained {
                Some(pin) => {
                    pin.ensure_unchanged()?;
                    Ok(pin.clone())
                }
                None => PinnedWeightsFile::pin(&abs),
            }
        })
        .collect()
}

impl Scail2FitProfile {
    /// Seal directory members before the worker computes its existing cache key. This
    /// retains every prior source and leaves single-character and replacement loads alone.
    pub fn prepare_load(spec: &mut LoadSpec, req: &GenerationRequest) -> Result<()> {
        let pairs = super::Scail2CharacterReferences::from_request(MODEL, req)?;
        if pairs.additional.is_empty() {
            return Ok(());
        }
        let pins = source_pins(spec)?;
        spec.include_directory_member_pins(&pins)
    }

    pub fn axes(&self) -> &Scail2FitAxes {
        &self.axes
    }
    pub fn from_plan(
        spec: &LoadSpec,
        plan: &VideoRenderPlan,
        req: &GenerationRequest,
    ) -> Result<Option<Self>> {
        if plan.provider_id() != MODEL {
            return Ok(None);
        }
        let pairs = super::Scail2CharacterReferences::from_request(MODEL, req)?;
        if pairs.additional.is_empty() {
            return Ok(None);
        }
        plan.check_fit_load(spec)?;
        plan.check_inputs(req)?;
        if req.video_mode.as_deref() != Some("animation") || req.memory.is_some() {
            return Err(unsupported("requires the native unscoped animation path"));
        }
        let g = plan.geometry();
        if g.reference_count != pairs.reference_count()
            || !(2..=6).contains(&g.reference_count)
            || g.batch != 1
        {
            return Err(unsupported("character count differs from physical layout"));
        }
        let backend = match plan.layout().domain {
            MemoryBudgetDomain::UnifiedMemory => Scail2FitBackend::Mlx,
            MemoryBudgetDomain::DeviceVram => Scail2FitBackend::Cuda,
            MemoryBudgetDomain::HostRam => {
                return Err(unsupported(
                    "no accelerator realization for this host-only plan",
                ))
            }
        };
        let pins = source_pins(spec)?;
        let mut hash = Sha256::new();
        hash.update(b"scail2-plural-weights-v1");
        for pin in &pins {
            hash.update(pin.content_sha256());
        }
        for adapter in &spec.adapters {
            // A zero scale can disable a residual; nonzero finite scale values do not alter
            // its activation shapes. Rank, target set, folds and packing are content-bound.
            hash.update(format!("{:?}:{}", adapter.kind, adapter.scale == 0.0).as_bytes());
        }
        let config: serde_json::Value = pins[0].read_unchanged(|p| -> Result<_> {
            serde_json::from_reader(std::fs::File::open(p)?)
                .map_err(|_| unsupported("invalid snapshot config"))
        })?;
        let heads = config
            .get("num_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(40);
        if heads == 0 {
            return Err(unsupported("empty attention head count"));
        }
        let grid = mul(u64::from(g.width / 16), u64::from(g.height / 16))?;
        let mut windows = Vec::new();
        for phase in &plan.layout().phases {
            let f = u64::from(phase.latent_frames);
            let tokens = add(
                mul(grid, u64::from(g.reference_count) + f)?,
                mul(grid / 4, f)?,
            )?;
            let (rows, scores) = if backend == Scail2FitBackend::Cuda {
                let stride = mul(heads, tokens)?;
                let rows = crate::attention_budget::AttentionBudget::from_score_elements(
                    crate::attention_budget::VIDEO_ATTENTION_SCORE_ELEMENTS,
                    false,
                )
                .query_block_rows(stride, tokens);
                (rows, mul(stride, rows)?)
            } else {
                (0, 0)
            };
            windows.push(Scail2FitWindow {
                latent_frames: phase.latent_frames,
                history_frames: phase.history_frames,
                decoded_frames: phase.decoded_frames,
                tokens,
                self_attention_query_rows: rows,
                self_attention_score_elements: scores,
            });
        }
        let mut resize = 0;
        let clip = req
            .control_clip()
            .ok_or_else(|| unsupported("driving clip missing"))?;
        for img in clip.frames.iter().chain(clip.mask).chain(
            std::iter::once(&pairs.primary)
                .chain(&pairs.additional)
                .flat_map(|p| [p.image, p.mask]),
        ) {
            resize = resize.max(resize_bytes(img, g.width, g.height)?);
        }
        let steps = req.steps.unwrap_or(40);
        let mut policy = execution_policy(backend);
        let sampler = req.sampler.as_deref().unwrap_or("unipc");
        // CUDA maps both advertised names to UniPC. MLX DPM++ has its own recurrence.
        policy.push(
            if backend == Scail2FitBackend::Mlx && sampler.eq_ignore_ascii_case("dpm++") {
                "dpm++"
            } else {
                "unipc"
            }
            .into(),
        );
        // The calibrated scheduler has bounded order-2 model history after step four.
        // Host schedule arrays are separate (native f64/f32 sigma/timestep buffers, phase
        // receipt and construction temporaries); allow eight f64 nodes per step and endpoint.
        let axes = Scail2FitAxes {
            backend,
            artifact_sha256: format!("{:x}", hash.finalize()),
            load_policy: format!(
                "{:?}/{:?}/{:?}/{:?}",
                spec.quantize, spec.precision, spec.offload_policy, spec.load_shape
            ),
            execution_policy: policy,
            width: g.width,
            height: g.height,
            references: g.reference_count,
            windows,
            source_bytes: plan.source_bytes(if backend == Scail2FitBackend::Mlx {
                MemoryBudgetDomain::UnifiedMemory
            } else {
                MemoryBudgetDomain::HostRam
            }),
            source_resize_bytes: resize,
            schedule_bytes: mul(u64::from(steps) + 1, 8 * 8)?,
            history_order: steps.min(4),
            classifier_free_guidance: req.guidance.unwrap_or(5.0) > 1.0,
            floor_host_bytes: plan.minimum_live_bytes(MemoryBudgetDomain::HostRam)?,
            floor_device_bytes: plan.minimum_live_bytes(MemoryBudgetDomain::DeviceVram)?,
            floor_unified_bytes: plan.minimum_live_bytes(MemoryBudgetDomain::UnifiedMemory)?,
        };
        Ok(Some(Self {
            axes,
            load: RenderLoadBinding::from_video_spec(spec),
            pins,
        }))
    }
    pub fn validate_load(&self, spec: &LoadSpec) -> Result<()> {
        if self.load != RenderLoadBinding::from_video_spec(spec) {
            return Err(unsupported("prepared source or placement changed"));
        }
        for pin in &self.pins {
            pin.ensure_unchanged()?;
        }
        let current = execution_policy(self.axes.backend);
        if !self.axes.execution_policy.starts_with(&current) {
            return Err(unsupported("native memory policy changed"));
        }
        Ok(())
    }
    pub fn price(&self, rows: &[Scail2FitWitness]) -> Result<Scail2FitPrice> {
        for row in rows {
            if let Some(price) = self.axes.price_from(row)? {
                return Ok(price);
            }
        }
        Err(unsupported(
            "no completed physical fit witness dominates this request",
        ))
    }
}

impl Scail2FitAxes {
    fn price_from(&self, row: &Scail2FitWitness) -> Result<Option<Scail2FitPrice>> {
        let a = &row.axes;
        if self.backend != a.backend
            || self.artifact_sha256 != a.artifact_sha256
            || self.load_policy != a.load_policy
            || self.execution_policy != a.execution_policy
            || self.width > a.width
            || self.height > a.height
            || self.references > a.references
            || self.history_order > a.history_order
            || (self.classifier_free_guidance && !a.classifier_free_guidance)
            || self.windows.len() > a.windows.len()
            || self.windows.iter().zip(&a.windows).any(|(x, y)| {
                x.latent_frames > y.latent_frames
                    || x.decoded_frames > y.decoded_frames
                    || x.history_frames > y.history_frames
                    || x.tokens > y.tokens
                    || x.self_attention_query_rows > y.self_attention_query_rows
                    || x.self_attention_score_elements > y.self_attention_score_elements
            })
        {
            return Ok(None);
        }
        // Grow known host-side inputs without copying the measured render's weights or
        // learned workspace into that domain. Substituting an unrelated tensor-byte lower
        // bound for a larger learned workspace would be unsound.
        let source_delta = self.source_bytes.saturating_sub(a.source_bytes);
        let extra = add(
            source_delta,
            add(
                self.source_resize_bytes
                    .saturating_sub(a.source_resize_bytes),
                self.schedule_bytes.saturating_sub(a.schedule_bytes),
            )?,
        )?;
        if row.gross_host_bytes < a.floor_host_bytes
            || row.gross_device_bytes < a.floor_device_bytes
            || row.gross_unified_bytes < a.floor_unified_bytes
        {
            return Ok(None);
        }
        let mut price = Scail2FitPrice {
            host_bytes: row.gross_host_bytes,
            device_bytes: row.gross_device_bytes,
            unified_bytes: row.gross_unified_bytes,
            mlx_decode_budget_bytes: row.mlx_decode_budget_bytes,
        };
        match self.backend {
            Scail2FitBackend::Cuda
                if price.unified_bytes == 0
                    && price.host_bytes > 0
                    && price.device_bytes > 0
                    && price.mlx_decode_budget_bytes.is_none() =>
            {
                price.host_bytes = add(price.host_bytes, extra)?;
            }
            Scail2FitBackend::Mlx
                if price.host_bytes == 0
                    && price.device_bytes == 0
                    && price.unified_bytes > 0
                    && price
                        .mlx_decode_budget_bytes
                        .is_some_and(|v| v > 0 && v <= price.unified_bytes) =>
            {
                price.unified_bytes = add(price.unified_bytes, extra)?;
            }
            _ => return Ok(None),
        }
        if price.host_bytes < self.floor_host_bytes
            || price.device_bytes < self.floor_device_bytes
            || price.unified_bytes < self.floor_unified_bytes
        {
            return Ok(None);
        }
        Ok(Some(price))
    }
}
