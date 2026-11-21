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
    pub artifact_tier: String,
    pub effective_tier: String,
    pub load_policy: String,
    pub execution_policy: Vec<String>,
    pub mlx_dpmpp: bool,
    pub width: u32,
    pub height: u32,
    pub references: u32,
    pub windows: Vec<Scail2FitWindow>,
    pub source_bytes: u64,
    pub source_shape_sha256: String,
    pub driving_stack_bytes: u64,
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
    /// Both full and remainder FFN calls use the same non-split matrix kernels in this
    /// class. Outside it a row can still cover its directly observed native shapes.
    pub mlx_unsplit_ffn: bool,
    pub mlx_ffn_chunks: u64,
    pub mlx_ffn_tail: u64,
    pub self_attention_query_rows: u64,
    pub self_attention_score_elements: u64,
}

/// One completed native run. Gross peaks include original RGB carriers and cold model
/// materialization. Discrete host/device peaks must originate in the same run.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scail2FitWitness {
    pub axes: Scail2FitAxes,
    /// Observed UMT5 length of the shortest encoded branch. The proof saturates both
    /// prompts to the native 512-token limit so arbitrary user text can reuse the row.
    pub text_tokens: u32,
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
    // Source switches in the pinned MLX Metal scheduler/matmul implementation and
    // Candle QMatMul constructor can change allocator or weight storage for equal shapes.
    let keys: &[&str] = match backend {
        Scail2FitBackend::Mlx => &[
            "SCAIL2_COMPUTE_BF16",
            "MLX_GEN_WAN_FFN_SEQ_CHUNK",
            "MLX_GEN_WAN_ATTN_QUERY_CHUNK",
            "MLX_GEN_WAN_EVAL_PER_BLOCK",
            "MLX_BFS_MAX_WIDTH",
            "MLX_MAX_OPS_PER_BUFFER",
            "MLX_MAX_MB_PER_BUFFER",
            "MLX_METAL_FAST_SYNCH",
            "MLX_ENABLE_TF32",
            "MLX_METAL_GPU_ARCH",
        ],
        Scail2FitBackend::Cuda => &[
            "CANDLE_DEQUANTIZE_ALL",
            "CANDLE_DEQUANTIZE_ALL_F16",
            "CUBLAS_WORKSPACE_CONFIG",
        ],
    };
    keys.iter()
        .map(|name| format!("{name}={}", std::env::var(name).unwrap_or_default()))
        .collect()
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
        super::Scail2CharacterReferences::catalog_carrier(&req.conditioning, false)?;
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
        let (pairs, clip) =
            super::Scail2CharacterReferences::catalog_carrier(&req.conditioning, false)?;
        let g = plan.geometry();
        if g.reference_count != pairs.reference_count()
            || !(2..=6).contains(&g.reference_count)
            || g.batch != 1
        {
            return Err(unsupported("character count differs from physical layout"));
        }
        let backend = match (plan.backend_id(), plan.layout().domain) {
            ("mlx", MemoryBudgetDomain::UnifiedMemory) => Scail2FitBackend::Mlx,
            ("candle", MemoryBudgetDomain::DeviceVram) => Scail2FitBackend::Cuda,
            _ => {
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
        // The pinned MLX Metal affine matmul's split count is max(1,512 / output_tiles)
        // with 32x32 tiles. All varying packed projections have output width >= dim here.
        // The ordinary full SDPA path requires a supported 128-wide head and >8 query rows.
        let dim = config.get("dim").and_then(|v| v.as_u64()).unwrap_or(5120);
        let ffn = config
            .get("ffn_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(13824);
        let packed_bits = config
            .pointer("/quantization/bits")
            .and_then(|v| v.as_u64());
        let stored = match packed_bits {
            Some(4) => Some(crate::Quant::Q4),
            Some(8) => Some(crate::Quant::Q8),
            None if !config.get("quantization").is_some_and(|v| v.is_object()) => None,
            _ => return Err(unsupported("unrecognized packed tier declaration")),
        };
        if backend == Scail2FitBackend::Cuda && stored != spec.quantize
            || stored.is_some() && spec.quantize.is_some() && stored != spec.quantize
        {
            return Err(unsupported("requested and stored packed tier disagree"));
        }
        let artifact_tier = crate::Quant::tier_label(stored).to_owned();
        let effective_tier = crate::Quant::affine_tier_label(spec.quantize.or(stored))
            .ok_or_else(|| unsupported("unsupported numerical realization"))?
            .to_owned();
        let default_mlx_policy = [
            "SCAIL2_COMPUTE_BF16",
            "MLX_GEN_WAN_ATTN_QUERY_CHUNK",
            "MLX_GEN_WAN_EVAL_PER_BLOCK",
            "MLX_GEN_WAN_FFN_SEQ_CHUNK",
        ]
        .iter()
        .all(|key| std::env::var_os(key).is_none());
        let standard_packed = backend == Scail2FitBackend::Mlx
            && default_mlx_policy
            && matches!(packed_bits, Some(4 | 8))
            && spec.quantize.is_none()
            && spec.adapters.is_empty()
            && heads == 40
            && dim == 5120
            && ffn >= dim;
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
            // map_seq_chunks makes one complete native call for each 8192 rows and a tail.
            let chunks = tokens.div_ceil(8192);
            let tail = tokens - (chunks - 1) * 8192;
            let smallest = tail.min(grid);
            let unsplit = standard_packed
                && smallest >= 32
                && mul(smallest.div_ceil(32), dim.div_ceil(32))? >= 512;
            windows.push(Scail2FitWindow {
                mlx_unsplit_ffn: unsplit,
                mlx_ffn_chunks: chunks,
                mlx_ffn_tail: tail,
                latent_frames: phase.latent_frames,
                history_frames: phase.history_frames,
                decoded_frames: phase.decoded_frames,
                tokens,
                self_attention_query_rows: rows,
                self_attention_score_elements: scores,
            });
        }
        let mut resize = 0;
        let mut source_shapes = Sha256::new();
        for img in clip.frames.iter().chain(clip.mask).chain(
            std::iter::once(&pairs.primary)
                .chain(&pairs.additional)
                .flat_map(|p| [p.image, p.mask]),
        ) {
            source_shapes.update(img.width.to_le_bytes());
            source_shapes.update(img.height.to_le_bytes());
            resize = resize.max(resize_bytes(img, g.width, g.height)?);
        }
        let steps = req.steps.unwrap_or(40);
        let policy = execution_policy(backend);
        // The published Candle dpm++ alias uses native UniPC; MLX uses its own 2M solver.
        let mlx_dpmpp = backend == Scail2FitBackend::Mlx && req.sampler.as_deref() == Some("dpm++");
        // The calibrated scheduler has bounded order-2 model history after step four.
        // Host schedule arrays are separate (native f64/f32 sigma/timestep buffers, phase
        // receipt and construction temporaries); allow eight f64 nodes per step and endpoint.
        let axes = Scail2FitAxes {
            backend,
            artifact_tier,
            effective_tier,
            artifact_sha256: format!("{:x}", hash.finalize()),
            load_policy: format!(
                "{:?}/{:?}/{:?}/{:?}",
                spec.quantize, spec.precision, spec.offload_policy, spec.load_shape
            ),
            execution_policy: policy,
            mlx_dpmpp,
            width: g.width,
            height: g.height,
            references: g.reference_count,
            windows,
            source_bytes: plan.source_bytes(if backend == Scail2FitBackend::Mlx {
                MemoryBudgetDomain::UnifiedMemory
            } else {
                MemoryBudgetDomain::HostRam
            }),
            driving_stack_bytes: mul(
                mul(u64::from(g.width), u64::from(g.height))?,
                mul(clip.frames.len() as u64, 2 * 3 * 4)?,
            )?,
            source_shape_sha256: format!("{:x}", source_shapes.finalize()),
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
        if self.axes.execution_policy != current {
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

impl Scail2FitWindow {
    fn covered_by(&self, measured: &Self, backend: Scail2FitBackend) -> bool {
        // Eligibility for another reference count requires the same FFN partition
        // family with a dominated last partition. Each matrix call is unsplit and
        // the full Metal SDPA kernel has no key-split partial-result allocation.
        let mlx_covered = backend != Scail2FitBackend::Mlx
            || self.tokens == measured.tokens
            || self.mlx_unsplit_ffn
                && measured.mlx_unsplit_ffn
                && self.mlx_ffn_chunks == measured.mlx_ffn_chunks
                && self.mlx_ffn_tail <= measured.mlx_ffn_tail;
        self.latent_frames == measured.latent_frames
            && self.decoded_frames == measured.decoded_frames
            && self.history_frames <= measured.history_frames
            && self.tokens <= measured.tokens
            && self.self_attention_query_rows <= measured.self_attention_query_rows
            && self.self_attention_score_elements <= measured.self_attention_score_elements
            && mlx_covered
    }
}

impl Scail2FitAxes {
    fn price_from(&self, row: &Scail2FitWitness) -> Result<Option<Scail2FitPrice>> {
        let a = &row.axes;
        // Spatial convolution and native decoder tile dispatch both depend on the
        // two axes. A gross measurement alone does not isolate their scratch across lattices.
        let covered_windows = self.windows.len() <= a.windows.len()
            && self
                .windows
                .iter()
                .zip(&a.windows)
                .all(|(window, measured)| window.covered_by(measured, self.backend));
        if row.text_tokens != 512
            || self.backend != a.backend
            || self.artifact_sha256 != a.artifact_sha256
            || self.artifact_tier != a.artifact_tier
            || self.effective_tier != a.effective_tier
            || (!self.mlx_dpmpp && a.mlx_dpmpp)
            || self.load_policy != a.load_policy
            || self.execution_policy != a.execution_policy
            || self.width != a.width
            || self.height != a.height
            || self.references > a.references
            || self.history_order > a.history_order
            || (self.classifier_free_guidance && !a.classifier_free_guidance)
            || !covered_windows
        {
            return Ok(None);
        }
        // Grow known host-side inputs without copying the measured render's weights or
        // learned workspace into that domain. Substituting an unrelated tensor-byte lower
        // bound for a larger learned workspace would be unsound.
        let source_delta = self.source_bytes.saturating_sub(a.source_bytes);
        let stack_delta = self
            .driving_stack_bytes
            .saturating_sub(a.driving_stack_bytes);
        // Extra authored tail frames also coexist as individual resized tensors while
        // concatenating the two persistent RGB stacks. Two full stack sizes cover both.
        let extra = add(
            add(source_delta, mul(stack_delta, 2)?)?,
            add(
                if self.source_shape_sha256 == a.source_shape_sha256 {
                    0
                } else {
                    self.source_resize_bytes
                },
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
                if self.mlx_dpmpp && !a.mlx_dpmpp {
                    // Price every additional array in the native second-order 2M step even
                    // if MLX fuses or aliases one: previous x0, two x0 operations, two
                    // first-difference operations, three scaled terms and two subtractions.
                    // Each of its five scalar multiplies also constructs an f32 scalar.
                    let area = mul(u64::from(self.width / 8), u64::from(self.height / 8))?;
                    let frames = self
                        .windows
                        .iter()
                        .map(|w| u64::from(w.latent_frames))
                        .max()
                        .unwrap_or(0);
                    let latent = mul(mul(area, frames)?, 16 * 4)?;
                    price.unified_bytes = add(price.unified_bytes, add(mul(latent, 10)?, 5 * 4)?)?;
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::video::VideoDenoisePhase;
    use crate::{
        Conditioning, Image, MemoryBudget, MemoryGeometry, Quant, VideoPlanLayout,
        VideoPlanRegistration,
    };
    use tempfile::TempDir;

    fn layout(
        req: &GenerationRequest,
        domain: MemoryBudgetDomain,
    ) -> Result<Option<VideoPlanLayout>> {
        let count =
            super::super::Scail2CharacterReferences::from_request(MODEL, req)?.reference_count();
        let g = MemoryGeometry {
            width: req.width,
            height: req.height,
            reference_count: count,
            batch: 1,
            frames: 45,
        };
        let phase = VideoDenoisePhase::fixed(g, 12, (), vec![1., 0.], 1);
        Ok(Some(VideoPlanLayout::single(
            g,
            &phase,
            12 * 16 * 4 * 6240,
            45,
            4,
            vec![],
            domain,
        )))
    }
    fn spec(backend: Scail2FitBackend) -> (TempDir, LoadSpec) {
        let dir = tempfile::tempdir().unwrap();
        for name in MEMBERS {
            std::fs::write(
                dir.path().join(name),
                if name == "config.json" {
                    br#"{"quantization":{"bits":4,"group_size":64}}"#
                } else {
                    &b"tiny"[..]
                },
            )
            .unwrap();
        }
        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.path().into()));
        if backend == Scail2FitBackend::Cuda {
            spec.quantize = Some(Quant::Q4);
        }
        (dir, spec)
    }
    fn request(refs: u32) -> GenerationRequest {
        let image = |color: u8| Image {
            width: 1,
            height: 1,
            pixels: vec![color; 3],
        };
        let mut conditioning = Vec::new();
        for r in 0..refs {
            conditioning.extend([
                Conditioning::Reference {
                    image: image(r as u8),
                    strength: None,
                },
                Conditioning::Mask { image: image(255) },
            ]);
        }
        conditioning.push(Conditioning::ControlClip {
            frames: vec![image(20); 45],
            mask: vec![image(255); 45],
            start_frame: 0,
            masking_strength: 1.,
            mode: Default::default(),
        });
        GenerationRequest {
            width: 832,
            height: 480,
            steps: Some(4),
            video_mode: Some("animation".into()),
            conditioning,
            ..Default::default()
        }
    }
    fn plan(
        backend: Scail2FitBackend,
        spec: &LoadSpec,
        req: &GenerationRequest,
    ) -> VideoRenderPlan {
        let (backend, reg) = match backend {
            Scail2FitBackend::Mlx => (
                "mlx",
                VideoPlanRegistration {
                    provider_id: MODEL,
                    compile: |_, r| layout(r, MemoryBudgetDomain::UnifiedMemory),
                },
            ),
            Scail2FitBackend::Cuda => (
                "candle",
                VideoPlanRegistration {
                    provider_id: MODEL,
                    compile: |_, r| layout(r, MemoryBudgetDomain::DeviceVram),
                },
            ),
        };
        VideoRenderPlan::compile(&reg, backend, spec, req)
            .unwrap()
            .unwrap()
    }
    fn profile(b: Scail2FitBackend, spec: &LoadSpec, req: &GenerationRequest) -> Scail2FitProfile {
        Scail2FitProfile::from_plan(spec, &plan(b, spec, req), req)
            .unwrap()
            .unwrap()
    }
    fn witness(p: &Scail2FitProfile) -> Scail2FitWitness {
        let mlx = p.axes().backend == Scail2FitBackend::Mlx;
        Scail2FitWitness {
            axes: p.axes().clone(),
            text_tokens: 512,
            gross_unified_bytes: if mlx { 9 << 30 } else { 0 },
            gross_host_bytes: if mlx { 0 } else { 5 << 30 },
            gross_device_bytes: if mlx { 0 } else { 7 << 30 },
            mlx_decode_budget_bytes: mlx.then_some(3 << 30),
        }
    }
    #[test]
    fn scail2_fit_computes_native_workspace_for_every_count_and_refuses_crossed_cuda_rows() {
        for b in [Scail2FitBackend::Cuda, Scail2FitBackend::Mlx] {
            let (_dir, spec) = spec(b);
            let largest = profile(b, &spec, &request(6));
            for (i, count) in (2..=6).enumerate() {
                let p = profile(b, &spec, &request(count));
                let w = &p.axes.windows[0];
                assert_eq!(w.tokens, [26520, 28080, 29640, 31200, 32760][i]);
                assert_eq!(p.axes.artifact_tier, "q4");
                assert_eq!(p.axes.effective_tier, "q4");
                if b == Scail2FitBackend::Cuda {
                    assert_eq!(w.self_attention_query_rows, [942, 890, 843, 801, 763][i]);
                    assert_eq!(
                        w.self_attention_score_elements,
                        [999273600, 999648000, 999460800, 999648000, 999835200][i]
                    );
                }
                let exact = p.price(&[witness(&p)]).unwrap();
                assert_eq!(
                    p.price(&[witness(&largest)]).is_ok(),
                    b == Scail2FitBackend::Mlx && w.mlx_unsplit_ffn || count == 6
                );
                let domain = if b == Scail2FitBackend::Cuda {
                    exact.device_bytes
                } else {
                    exact.unified_bytes
                };
                let budget = MemoryBudget {
                    total_bytes: domain,
                    committed_bytes: 0,
                    reclaimable_bytes: 0,
                    reserved_headroom_bytes: 0,
                };
                assert!(budget.fits(domain));
                assert!(!MemoryBudget {
                    total_bytes: domain - 1,
                    ..budget
                }
                .fits(domain));
            }
        }
    }
    #[test]
    fn scail2_fit_binds_order_artifact_and_cache_placement_without_loading_tensors() {
        let (dir, mut sp) = spec(Scail2FitBackend::Cuda);
        sp.prepare_file_sources().unwrap(); // already-finalized directory-only cache source
        let mut bad = request(2);
        bad.conditioning.swap(0, 1);
        assert!(Scail2FitProfile::prepare_load(&mut sp, &bad).is_err());
        let mut duplicate = request(2);
        duplicate
            .conditioning
            .insert(0, duplicate.conditioning.last().unwrap().clone());
        assert!(Scail2FitProfile::prepare_load(&mut sp, &duplicate).is_err());
        let mut replace = request(2);
        replace.video_mode = Some("replacement".into());
        assert!(Scail2FitProfile::prepare_load(&mut sp, &replace).is_err());
        assert!(Scail2FitProfile::prepare_load(&mut sp, &request(7)).is_err());
        Scail2FitProfile::prepare_load(&mut sp, &request(1)).unwrap();
        assert_eq!(sp.prepared_file_pins().iter().count(), 0);
        let req = request(2);
        Scail2FitProfile::prepare_load(&mut sp, &req).unwrap();
        assert_eq!(sp.prepared_file_pins().iter().count(), 6);
        let planned = plan(Scail2FitBackend::Cuda, &sp, &req);
        let p = Scail2FitProfile::from_plan(&sp, &planned, &req)
            .unwrap()
            .unwrap();
        p.validate_load(&sp).unwrap();
        let reference_hashes = planned.reference_rgb_digests();
        assert_eq!(
            reference_hashes,
            [
                <[u8; 32]>::from(Sha256::digest([0, 0, 0])),
                Sha256::digest([1, 1, 1]).into()
            ]
        );
        let mut crossed = req.clone();
        crossed.conditioning.swap(0, 2);
        assert!(Scail2FitProfile::from_plan(&sp, &planned, &crossed).is_err());
        let mut placement = sp.clone();
        placement.quantize = Some(Quant::Q8);
        assert!(p.validate_load(&placement).is_err());
        std::fs::write(dir.path().join("vae.safetensors"), b"changed size").unwrap();
        assert!(p.validate_load(&sp).is_err());
    }
    #[test]
    fn scail2_fit_preserves_native_schedule_source_pixels_and_decoder_cap() {
        let (_dir, spec) = spec(Scail2FitBackend::Mlx);
        let req = request(6);
        let p = profile(Scail2FitBackend::Mlx, &spec, &req);
        let row = witness(&p);
        let mut different = req.clone();
        different.prompt = "unrelated prompt text".into();
        different.seed = Some(6789);
        different.fps = Some(12);
        different.scheduler_shift = Some(3.);
        different.steps = Some(40);
        different.sampler = Some("dpm++".into());
        if let Conditioning::Mask { image } = &mut different.conditioning[1] {
            image.pixels[0] ^= 255;
        }
        let active = profile(Scail2FitBackend::Mlx, &spec, &different)
            .price(std::slice::from_ref(&row))
            .unwrap();
        assert_eq!(
            active.unified_bytes,
            row.gross_unified_bytes + 36 * 64 + 12 * 6240 * 16 * 4 * 10 + 5 * 4
        );
        assert_eq!(mlx_decode_budget_bytes(), None);
        active.with_execution(|| {
            assert_eq!(mlx_decode_budget_bytes(), Some(3 << 30));
            Scail2FitPrice {
                mlx_decode_budget_bytes: Some(1 << 30),
                ..active
            }
            .with_execution(|| assert_eq!(mlx_decode_budget_bytes(), Some(1 << 30)));
            assert_eq!(mlx_decode_budget_bytes(), Some(3 << 30));
        });
        assert_eq!(mlx_decode_budget_bytes(), None);
        let mut bad = witness(&p);
        bad.text_tokens = 511;
        assert!(p.price(&[bad]).is_err());
        let mut over = witness(&p);
        over.gross_unified_bytes = u64::MAX;
        assert!(profile(Scail2FitBackend::Mlx, &spec, &different)
            .price(&[over])
            .is_err());
        let mut clipped = witness(&p);
        clipped.mlx_decode_budget_bytes = None;
        assert!(p.price(&[clipped]).is_err());
    }
}
