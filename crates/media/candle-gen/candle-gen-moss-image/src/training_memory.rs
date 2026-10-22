//! Header/shape-only upper bounds for native MOSS adapter training.
//!
//! Integration: build `FootprintFacts::from_sources` and `EstimateRequest::from_request`, then
//! `estimate(...).check_all(budget)` BEFORE Qwen loading, VAE construction, or backbone load.
//! Keep the estimate and recheck `Cast`, `Cache`, and `Backward` before those allocations (and
//! `Preview` before preview sampling). The trainer owns its existing optional budget override;
//! there is deliberately no new field on the generic training request. These are structural
//! operator bounds, not measured calibration, an allocator prediction, or a runtime RSS limit.
use crate::loader::Sources;
use candle_gen::gen_core::{
    self,
    moss_image::{MossImageAssetRole, MossImageConfig},
    moss_image_training::target_paths,
    weightsmeta::{safetensors_path_tensor_headers, SafetensorsTensorHeader},
};
use candle_gen::gen_core::{NetworkType, TrainingConfig, TrainingRequest};
use candle_llm::models::qwen35::Qwen35Config;
use std::fmt;

type Result<T> = std::result::Result<T, AdmissionError>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AdmissionError {
    Invalid(String),
    Overflow(&'static str),
    Source(String),
    HostBudgetExceeded {
        required: u64,
        available: u64,
    },
    BudgetExceeded {
        phase: Phase,
        required: u64,
        budget: u64,
    },
}
impl fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid MOSS training memory facts: {s}"),
            Self::Overflow(s) => write!(f, "MOSS training memory arithmetic overflow: {s}"),
            Self::Source(s) => write!(f, "MOSS training memory metadata: {s}"),
            Self::HostBudgetExceeded{required,available}=>write!(f,"MOSS encoded/decoder/serialization host bound {required} bytes exceeds {available} available HOST RAM bytes (separate from CUDA VRAM)"),
            Self::BudgetExceeded {
                phase,
                required,
                budget,
            } => {
                write!(f,
                "MOSS training {phase:?} structural working-set bound {required} bytes exceeds \
                 the {budget}-byte device/trainer budget; reduce resolution, caption length, \
                 dataset/buckets or rank, use bf16 compute, or enable gradient checkpointing \
                 (checkpointing does not remove the single-block FP32 attention workspace)")
            }
        }
    }
}
impl std::error::Error for AdmissionError {}
fn source(e: impl fmt::Display) -> AdmissionError {
    AdmissionError::Source(e.to_string())
}
fn invalid(s: impl Into<String>) -> AdmissionError {
    AdmissionError::Invalid(s.into())
}
fn sum(xs: impl IntoIterator<Item = u64>, what: &'static str) -> Result<u64> {
    xs.into_iter().try_fold(0u64, |a, b| {
        a.checked_add(b).ok_or(AdmissionError::Overflow(what))
    })
}
fn mul(xs: &[u64], what: &'static str) -> Result<u64> {
    xs.iter().try_fold(1u64, |a, b| {
        a.checked_mul(*b).ok_or(AdmissionError::Overflow(what))
    })
}
fn dim(n: usize) -> Result<u64> {
    u64::try_from(n).map_err(|_| invalid("dimension exceeds u64"))
}
fn positive(n: i32) -> Result<u64> {
    if n <= 0 {
        return Err(invalid("nonpositive encoder dimension"));
    }
    u64::try_from(n).map_err(|_| invalid("unrepresentable encoder dimension"))
}

/// Pure override resolution: a zero override is a real zero cap, not a request to disable guards.
pub(crate) fn resolve_budget(override_bytes: Option<u64>, device_bytes: u64) -> u64 {
    override_bytes.unwrap_or(device_bytes)
}
/// Resolve an actual allocator/device budget. CPU numerical oracles must explicitly provide
/// an override; unknown CUDA memory never disables preallocation admission.
pub(crate) fn device_budget_bytes(device: &candle_gen::candle_core::Device) -> Result<u64> {
    if !device.is_cuda() {
        return Err(invalid(
            "native MOSS training requires CUDA; CPU oracles require an explicit trainer budget",
        ));
    }
    #[cfg(feature = "cuda")]
    {
        // Query the caller's actual context, not nvidia-smi's physical ordinal or a separately
        // created logical device 0. Cudarc binds this exact context before cuMemGetInfo.
        let stream = device.as_cuda_device().map_err(source)?.cuda_stream();
        let (free,_total)=stream.context().mem_get_info().map_err(|e|invalid(format!("cannot query the admitted CUDA context: {e}; provide an explicit native trainer budget")))?;
        if free == 0 {
            return Err(invalid("admitted CUDA device has no free memory"));
        }
        Ok((free as f64 * candle_gen::train::flow_match::TRAIN_SAFE_FRAC) as u64)
    }
    #[cfg(not(feature = "cuda"))]
    Err(invalid(
        "CUDA training is not compiled; use the native CUDA runtime",
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    Load,
    Cast,
    Cache,
    Backward,
    Preview,
    Checkpoint,
}
#[derive(Clone, Debug)]
#[allow(dead_code)] // Component diagnostics and peak are optional to the trainer, exercised purely.
pub(crate) struct MemoryEstimate {
    pub load: u64,
    pub cast: u64,
    pub cache: u64,
    pub backward: u64,
    pub preview: u64,
    pub checkpoint: u64,
    pub context_token_bound: u64,
    pub host_peak_bytes: u64,
    /// Auditable components, useful for deterministic guard tests and refusal diagnostics.
    pub cache_bytes: u64,
    pub trainable_elements: u64,
    pub adapter_work_bytes: u64,
    pub attention_backward_bytes: u64,
    pub frozen_input_grad_bytes: u64,
}
impl MemoryEstimate {
    pub fn bytes(&self, phase: Phase) -> u64 {
        match phase {
            Phase::Load => self.load,
            Phase::Cast => self.cast,
            Phase::Cache => self.cache,
            Phase::Backward => self.backward,
            Phase::Preview => self.preview,
            Phase::Checkpoint => self.checkpoint,
        }
    }
    pub fn check(&self, phase: Phase, budget: u64) -> Result<()> {
        let required = self.bytes(phase);
        if required > budget {
            Err(AdmissionError::BudgetExceeded {
                phase,
                required,
                budget,
            })
        } else {
            Ok(())
        }
    }
    pub fn check_host(&self, available: u64, cpu_reference: bool) -> Result<()> {
        let required = if cpu_reference {
            sum(
                [self.host_peak_bytes, self.peak()],
                "CPU tensor plus host staging",
            )?
        } else {
            self.host_peak_bytes
        };
        if required > available {
            Err(AdmissionError::HostBudgetExceeded {
                required,
                available,
            })
        } else {
            Ok(())
        }
    }
    pub fn check_all(&self, budget: u64) -> Result<()> {
        for phase in [
            Phase::Load,
            Phase::Cast,
            Phase::Cache,
            Phase::Backward,
            Phase::Preview,
            Phase::Checkpoint,
        ] {
            self.check(phase, budget)?;
        }
        Ok(())
    }
    #[allow(dead_code)] // Optional aggregate diagnostic; phase guards do not depend on it.
    pub fn peak(&self) -> u64 {
        self.load
            .max(self.cast)
            .max(self.cache)
            .max(self.backward)
            .max(self.preview)
            .max(self.checkpoint)
    }
}

/// Config and ordinary-asset header facts are injectable, so miniature shapes exercise the same
/// production arithmetic. No production corpus count or machine-dependent byte value is pinned.
#[derive(Clone, Debug)]
pub(crate) struct FootprintFacts {
    pub config: MossImageConfig,
    pub encoder_config: Qwen35Config,
    pub encoder_bytes: u64,
    pub vae_bytes: u64,
    pub backbone_elements: u64,
    pub backbone_source_bytes: u64,
    pub largest_source_shard: u64,
    pub host_metadata_bytes: u64,
    pub vae_headers: Vec<SafetensorsTensorHeader>,
}
impl FootprintFacts {
    /// Inspect ONLY ordinary assets. `Asset::headers` is intentionally NOT used: its cache miss
    /// authenticates an owned payload. These readers inspect the pinned file's header, not tensors;
    /// the authenticated pins are checked before/after, including around every individual read.
    pub fn from_sources(s: &Sources) -> Result<Self> {
        s.ensure_unchanged().map_err(source)?;
        let encoder_config = s.text_config.clone();
        let mut shards = Vec::new();
        let mut largest_file = 0u64;
        for a in &s.ordinary {
            if !a.relative.ends_with(".safetensors") {
                continue;
            }
            a.pin.verify_unchanged().map_err(source)?;
            let headers = safetensors_path_tensor_headers(a.pin.loader_path()).map_err(source)?;
            // Actual owned-shard staging includes the safetensors JSON header and alignment,
            // not just tensor data_bytes. stat does not read payloads.
            largest_file = largest_file.max(
                std::fs::metadata(a.pin.loader_path())
                    .map_err(source)?
                    .len(),
            );
            a.pin.verify_unchanged().map_err(source)?;
            let kind = if a.role == MossImageAssetRole::Transformer {
                AssetKind::Backbone
            } else if a.relative.starts_with("text_encoder/") {
                AssetKind::Encoder
            } else if a.relative == "vae/diffusion_pytorch_model.safetensors" {
                AssetKind::Vae
            } else {
                return Err(invalid(format!(
                    "unexpected ordinary tensor asset {}",
                    a.relative
                )));
            };
            shards.push((kind, headers));
        }
        s.ensure_unchanged().map_err(source)?;
        let mut facts = Self::from_headers(s.config.clone(), encoder_config, &shards)?;
        facts.largest_source_shard = facts.largest_source_shard.max(largest_file);
        // Tokenizer/config parsing and lookup-table metadata are host allocations, not CUDA VRAM.
        for asset in &s.ordinary {
            if !asset.relative.ends_with(".safetensors") {
                let bytes = std::fs::metadata(asset.pin.loader_path())
                    .map_err(source)?
                    .len();
                facts.host_metadata_bytes = sum(
                    [
                        facts.host_metadata_bytes,
                        mul(&[bytes, 16], "tokenizer/config lookup metadata")?,
                    ],
                    "host metadata",
                )?;
            }
        }
        Ok(facts)
    }
    pub fn from_headers(
        config: MossImageConfig,
        encoder_config: Qwen35Config,
        shards: &[(AssetKind, Vec<SafetensorsTensorHeader>)],
    ) -> Result<Self> {
        let mut facts = Self {
            config,
            encoder_config,
            encoder_bytes: 0,
            vae_bytes: 0,
            backbone_elements: 0,
            backbone_source_bytes: 0,
            largest_source_shard: 0,
            host_metadata_bytes: 0,
            vae_headers: Vec::new(),
        };
        for (kind, headers) in shards {
            // The loader reads one owned source shard before copying selected arrays. All source
            // tensors, even unselected vision/LM-head tensors, contribute to that host staging.
            let shard = sum(headers.iter().map(|h| h.data_bytes), "source shard")?;
            facts.largest_source_shard = facts.largest_source_shard.max(shard);
            for h in headers {
                facts.host_metadata_bytes = sum(
                    [facts.host_metadata_bytes, 2048, h.name.len() as u64],
                    "owned header/tensor metadata",
                )?;
                let elements = h.element_count().map_err(source)?;
                if elements == 0 {
                    return Err(invalid(format!("empty tensor {}", h.name)));
                }
                match kind {
                    AssetKind::Encoder if h.name.starts_with("language_model.") => {
                        facts.encoder_bytes =
                            sum([facts.encoder_bytes, h.data_bytes], "language tower")?;
                    }
                    AssetKind::Vae if vae_selected(&h.name) => {
                        // Both encoder AND small decoder are always constructed; normalization
                        // statistics and convolution/linear weights all become FP32, whatever stored.
                        facts.vae_bytes =
                            sum([facts.vae_bytes, mul(&[elements, 4], "VAE fp32")?], "VAE")?;
                        facts.vae_headers.push(h.clone());
                    }
                    AssetKind::Backbone => {
                        facts.backbone_elements =
                            sum([facts.backbone_elements, elements], "backbone elements")?;
                        facts.backbone_source_bytes = sum(
                            [facts.backbone_source_bytes, h.data_bytes],
                            "backbone source",
                        )?;
                    }
                    _ => {}
                }
            }
        }
        if facts.encoder_bytes == 0
            || facts.vae_bytes == 0
            || facts.backbone_elements == 0
            || !facts
                .vae_headers
                .iter()
                .any(|h| h.name.starts_with("encoder."))
            || !facts
                .vae_headers
                .iter()
                .any(|h| h.name.starts_with("decoder."))
            || !facts
                .vae_headers
                .iter()
                .any(|h| h.name == "bn.running_mean")
            || !facts.vae_headers.iter().any(|h| h.name == "bn.running_var")
        {
            return Err(invalid("ordinary language tower, full VAE with FP32 statistics and backbone headers are required"));
        }
        Ok(facts)
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum AssetKind {
    Encoder,
    Vae,
    Backbone,
}
fn vae_selected(name: &str) -> bool {
    ["encoder.", "decoder.", "quant_conv.", "post_quant_conv."]
        .iter()
        .any(|p| name.starts_with(p))
        || name == "bn.running_mean"
        || name == "bn.running_var"
}

#[derive(Clone, Debug)]
pub(crate) struct EstimateRequest {
    pub config: TrainingConfig,
    pub items: u64,
    pub edges: Vec<u64>,
    /// Conservative ragged-caption bound, not padding or a measured mean. Production is 4096.
    pub context_tokens: u64,
    pub preview_contexts: u64,
    pub preview_edge: u64,
    pub decode_bytes: u64,
    pub dataset_metadata_bytes: u64,
    pub cache_token_count: Option<u64>,
    pub evaluation_canvas: Option<(u64, u64)>,
    pub evaluation_context_tokens: u64,
    pub audit_workspace_bytes: u64,
}
impl EstimateRequest {
    /// Decode/crop source pixels can exceed the target bucket; inspect only image headers here.
    #[cfg(test)]
    pub fn with_image_headers(self, req: &TrainingRequest) -> Result<Self> {
        let plan =
            gen_core::moss_image_training::recovery::DataPlan::from_request(req).map_err(source)?;
        self.with_plan(req, &plan)
    }
    pub fn with_plan(
        mut self,
        req: &TrainingRequest,
        plan: &gen_core::moss_image_training::recovery::DataPlan,
    ) -> Result<Self> {
        self.decode_bytes = self
            .decode_bytes
            .max(plan.decode_buffer_bound().map_err(source)?);
        self.dataset_metadata_bytes =
            gen_core::moss_image_training::recovery::DataPlan::metadata_bound(req)
                .map_err(source)?;
        self.cache_token_count = Some(plan.cached_token_count().map_err(source)?);
        Ok(self)
    }
    pub fn from_request(req: &TrainingRequest) -> Result<Self> {
        let c = &req.config;
        let previews = if c.sample_every > 0 {
            c.sample_prompts.len().min(4)
        } else {
            0
        };
        Ok(Self {
            config: c.clone(),
            items: dim(req.items.len())?,
            edges: candle_gen::train::dataset::bucket_edges(c)
                .into_iter()
                .map(u64::from)
                .collect(),
            context_tokens: 4096,
            // Each prompt plus the single shared negative; none when previews are disabled.
            preview_contexts: if previews > 0 { dim(previews)? + 1 } else { 0 },
            preview_edge: u64::from(candle_gen::train::dataset::bucket_resolution(c.resolution)),
            decode_bytes: 0,
            dataset_metadata_bytes: 0,
            cache_token_count: None,
            evaluation_canvas: None,
            evaluation_context_tokens: 0,
            audit_workspace_bytes: 0,
        })
    }
}

#[cfg(test)]
pub(crate) fn image_decode_bytes(path: &std::path::Path) -> Result<u64> {
    let (width, height) = image::image_dimensions(path).map_err(source)?;
    if width == 0 || height == 0 {
        return Err(invalid("source image has an empty geometry"));
    }
    // A decoded RGBA32F plane and codec workspace, plus RGB8 conversion and crop copies.
    // Price compressed source bytes as well; no image payload is decoded for this guard.
    sum(
        [
            mul(
                &[u64::from(width), u64::from(height), 4 * 4 * 2 + 3 * 2],
                "source decode/crop buffers",
            )?,
            std::fs::metadata(path).map_err(source)?.len(),
        ],
        "source image decode",
    )
}

/// Shape-derived trainable leaves and structured residual workspace; NEVER a dense [out,in]
/// adapter delta. The initial first LoKr factor is full; resume/import validation can reconstruct
/// a decomposed small factor, so price both small products transiently, never their Kronecker product.
fn adapter_shape(out: u64, input: u64, c: &TrainingConfig, tokens: u64) -> Result<(u64, u64)> {
    let r = u64::from(c.rank);
    match c.network_type {
        NetworkType::Lora => Ok((
            mul(&[r, sum([out, input], "LoRA dims")?], "LoRA factors")?,
            mul(
                &[tokens, sum([r, out], "LoRA forward")?, 4],
                "LoRA intermediate",
            )?,
        )),
        NetworkType::Lokr => {
            let checked =
                |n| i32::try_from(n).map_err(|_| invalid("adapter dimension exceeds i32"));
            let (oa, ob) =
                candle_gen::train::lora::factorization(checked(out)? as usize, c.decompose_factor);
            let (ia, ib) = candle_gen::train::lora::factorization(
                checked(input)? as usize,
                c.decompose_factor,
            );
            let (oa, ob, ia, ib) = (dim(oa)?, dim(ob)?, dim(ia)?, dim(ib)?);
            let first = mul(&[oa, ia], "LoKr first factor")?;
            let second = if mul(&[r, 2], "LoKr threshold")? < ob.max(ib) {
                mul(
                    &[r, sum([ob, ib], "LoKr rank dims")?],
                    "LoKr decomposed factors",
                )?
            } else {
                mul(&[ob, ib], "LoKr second factor")?
            };
            let low = mul(&[r, 2], "LoKr threshold")? < ob.max(ib);
            let cb = mul(&[ia, ob], "LoKr second-first contraction")?;
            let da = mul(&[ib, oa], "LoKr first-second contraction")?;
            let rank_work = if low {
                mul(
                    &[if cb <= da { ia } else { oa }, r],
                    "LoKr low-rank contraction",
                )?
            } else {
                0
            };
            let per_token = sum(
                [
                    input,
                    cb.min(da),
                    rank_work,
                    mul(&[4, out], "LoKr result/scale/add planes")?,
                ],
                "LoKr structured buffers",
            )?;
            let products = sum(
                [first, mul(&[ob, ib], "LoKr small reconstruction")?],
                "LoKr small products",
            )?;
            Ok((
                sum([first, second], "LoKr trainable factors")?,
                mul(
                    &[
                        sum(
                            [
                                mul(&[tokens, per_token, 2], "LoKr token workspace and VJP")?,
                                products,
                            ],
                            "LoKr workspace",
                        )?,
                        4,
                    ],
                    "LoKr fp32 workspace",
                )?,
            ))
        }
    }
}
fn optimizer_states(name: &str) -> Result<u64> {
    let n: String = name
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| *c != '-' && *c != '_')
        .collect();
    match n.as_str() {
        "adam" | "adamw" | "adam8bit" | "adamw8bit" => Ok(2),
        "rose" | "roseopt" => Ok(0),
        "prodigy" | "prodigyopt" => Ok(4),
        _ => Err(invalid(format!("unsupported optimizer {name}"))),
    }
}
fn compute_width(c: &TrainingConfig) -> Result<u64> {
    match c.train_dtype.to_ascii_lowercase().as_str() {
        "bf16" | "bfloat16" => Ok(2),
        "f32" | "float32" | "fp32" => Ok(4),
        _ => Err(invalid("compute dtype must be bf16 or f32")),
    }
}
fn image_tokens(edge: u64) -> Result<u64> {
    if edge == 0 || !edge.is_multiple_of(16) {
        return Err(invalid("bucket edge must be positive and divisible by 16"));
    }
    mul(&[edge / 16, edge / 16], "image tokens")
}

/// Whole-layer encoder live tensors plus fresh per-caption KV/recurrent state. The layer hook
/// evaluates each hidden state, so do not multiply its temporary activation stream by all layers.
pub(crate) fn encoder_work(c: &Qwen35Config, seq: u64) -> Result<u64> {
    let (hidden, wide, heads, kv, head) = (
        positive(c.hidden_size)?,
        positive(c.intermediate_size)?,
        positive(c.num_heads)?,
        positive(c.num_kv_heads)?,
        positive(c.head_dim)?,
    );
    for d in [
        c.linear_num_key_heads,
        c.linear_num_value_heads,
        c.linear_key_head_dim,
        c.linear_value_head_dim,
        c.linear_conv_kernel_dim,
    ] {
        positive(d)?;
    }
    if c.num_layers == 0 || c.full_attention_interval == 0 || c.moe.is_some() {
        return Err(invalid("expected nonempty dense headless Qwen3.5 encoder"));
    }
    // Candle's headless path uses a fresh reference cache: growing KV for full attention,
    // one FP32 SSM state and one convolution tail for each GDN layer, with no rollback ring.
    let full = dim(c.num_layers / c.full_attention_interval)?;
    let linear = dim(c.num_layers)? - full;
    let kv_state = mul(&[full, 2, seq, kv, head, 4], "encoder KV state")?;
    let ssm = mul(
        &[
            positive(c.linear_num_value_heads)?,
            positive(c.linear_value_head_dim)?,
            positive(c.linear_key_head_dim)?,
            4,
        ],
        "encoder SSM",
    )?;
    let conv_channels = sum(
        [
            mul(
                &[
                    2,
                    positive(c.linear_num_key_heads)?,
                    positive(c.linear_key_head_dim)?,
                ],
                "encoder conv keys",
            )?,
            mul(
                &[
                    positive(c.linear_num_value_heads)?,
                    positive(c.linear_value_head_dim)?,
                ],
                "encoder conv values",
            )?,
        ],
        "encoder conv channels",
    )?;
    let conv = mul(
        &[positive(c.linear_conv_kernel_dim)? - 1, conv_channels, 4],
        "encoder conv tail",
    )?;
    let state = sum(
        [
            kv_state,
            mul(
                &[linear, sum([ssm, conv], "encoder recurrent state")?],
                "all recurrent layers",
            )?,
        ],
        "encoder state",
    )?;
    let linear_qkv = sum(
        [
            mul(
                &[
                    2,
                    positive(c.linear_num_key_heads)?,
                    positive(c.linear_key_head_dim)?,
                ],
                "linear encoder keys",
            )?,
            mul(
                &[
                    positive(c.linear_num_value_heads)?,
                    positive(c.linear_value_head_dim)?,
                ],
                "linear encoder values",
            )?,
        ],
        "linear encoder QKV",
    )?;
    // Residual/norm/output/gating (8 hidden widths); SwiGLU up/gate/silu/product (4 FFN
    // widths); Q+gate and K/V; and linear QKV + convolution output + gated output.
    let live = mul(
        &[
            seq,
            sum(
                [
                    mul(&[8, hidden], "encoder residuals")?,
                    mul(&[4, wide], "encoder SwiGLU")?,
                    mul(&[2, heads, head], "encoder gated Q")?,
                    mul(&[2, kv, head], "encoder KV live")?,
                    mul(&[3, linear_qkv], "encoder linear attention live")?,
                ],
                "encoder layer live",
            )?,
            4,
        ],
        "encoder FP32 live",
    )?;
    // A conservative unfused attention/chunk bound: one scores tensor and its normalized
    // weights. Linear attention's blocks cannot exceed the entire sequence. No calibration.
    let scores = mul(
        &[
            2,
            heads.max(positive(c.linear_num_value_heads)?),
            seq,
            seq,
            4,
        ],
        "encoder attention bound",
    )?;
    sum([state, live, scores], "encoder work")
}

/// VAE operator outputs at their actual pyramid levels, derived from its own header shapes.
/// FP32 conv output+bias, norm input/square/normalize/affine, and linear output+bias are counted.
/// Sum all outputs (a conservative lazy-graph envelope), plus the largest convolution lowering
/// workspace, not a blanket multiplier on model size. No backward graph exists for cached VAE.
fn vae_work(headers: &[SafetensorsTensorHeader], edge: u64, decoder: bool) -> Result<u64> {
    vae_work_canvas(headers, edge, edge, decoder)
}
fn vae_work_canvas(
    headers: &[SafetensorsTensorHeader],
    width: u64,
    height: u64,
    decoder: bool,
) -> Result<u64> {
    let side = if decoder { "decoder." } else { "encoder." };
    let mut outputs = Vec::new();
    let mut conv_workspace = 0u64;
    for h in headers
        .iter()
        .filter(|h| h.name.starts_with(side) && h.name.ends_with(".weight"))
    {
        let mut divisor = if decoder { 8 } else { 1 };
        let block_prefix = if decoder {
            "decoder.up_blocks."
        } else {
            "encoder.down_blocks."
        };
        if let Some(rest) = h.name.strip_prefix(block_prefix) {
            let index: u32 = rest
                .split('.')
                .next()
                .ok_or_else(|| invalid("VAE block key"))?
                .parse()
                .map_err(|_| invalid("VAE block index"))?;
            if index > 3 {
                return Err(invalid("VAE pyramid has more than four stages"));
            }
            divisor = if decoder {
                1u64 << (3 - index)
            } else {
                1u64 << index
            };
            if decoder && h.name.contains(".upsamplers.") {
                divisor /= 2;
            }
        } else if h.name.contains(".mid_block.")
            || (!decoder && h.name.contains(".conv_out."))
            || (!decoder && h.name.contains(".conv_norm_out."))
        {
            divisor = 8;
        } else if decoder && (h.name.contains(".conv_out.") || h.name.contains(".conv_norm_out.")) {
            divisor = 1;
        }
        if divisor == 0 || !width.is_multiple_of(divisor) || !height.is_multiple_of(divisor) {
            return Err(invalid("invalid VAE pyramid edge"));
        }
        let area = mul(&[width / divisor, height / divisor], "VAE operator area")?;
        let channels = dim(*h
            .shape
            .first()
            .ok_or_else(|| invalid("scalar VAE operator weight"))?)?;
        let buffers = match h.shape.len() {
            1 => 4,
            2 => 2,
            4 => {
                // Bound im2col/GEMM staging by input channels times the kernel footprint.
                conv_workspace = conv_workspace.max(mul(
                    &[
                        area,
                        dim(h.shape[1])?,
                        dim(h.shape[2])?,
                        dim(h.shape[3])?,
                        4,
                    ],
                    "VAE convolution workspace",
                )?);
                2
            }
            _ => return Err(invalid(format!("unsupported VAE weight shape {}", h.name))),
        };
        outputs.push(mul(&[area, channels, buffers, 4], "VAE output buffers")?);
    }
    let mid_area = mul(&[width / 8, height / 8], "VAE mid attention sequence")?;
    let score_bytes = mul(&[mid_area, mid_area, 4], "VAE mid scores")?;
    let score_bytes = score_bytes.min(
        (candle_gen::ATTN_SCORES_BUDGET as u64).max(mul(&[mid_area, 4], "one VAE query row")?),
    );
    outputs.push(mul(&[score_bytes, 3], "VAE score/softmax temporaries")?);
    // Preprocessing RGB/normalization and raw moments/packed-latent normalization temporaries.
    let pixels = mul(
        &[width, height, 3, 4, if decoder { 3 } else { 1 }],
        "VAE device pixel buffers",
    )?;
    let packed_channels = headers
        .iter()
        .find(|h| h.name == "bn.running_mean")
        .ok_or_else(|| invalid("missing VAE normalization shape"))?
        .element_count()
        .map_err(source)?;
    let moments = mul(
        &[
            mul(&[width / 16, height / 16], "VAE packed grid")?,
            packed_channels,
            4,
            4,
        ],
        "VAE packing/BN buffers",
    )?;
    sum(
        [
            sum(outputs, "VAE operator outputs")?,
            conv_workspace,
            pixels,
            moments,
        ],
        "VAE work",
    )
}

pub(crate) fn estimate(f: &FootprintFacts, r: &EstimateRequest) -> Result<MemoryEstimate> {
    let c = &r.config;
    let g = &f.config;
    let (d, wide, heads, layers, text, channels) = (
        dim(g.dim)?,
        dim(g.ffn_dim)?,
        dim(g.num_heads)?,
        dim(g.num_layers)?,
        dim(g.text_dim)?,
        dim(g.latent_channels)?,
    );
    if [
        d,
        wide,
        heads,
        layers,
        text,
        channels,
        r.items,
        r.context_tokens,
    ]
    .contains(&0)
        || d % heads != 0
        || (r.edges.is_empty() && c.aspect_ratio_buckets.is_empty())
        || c.rank == 0
        || c.rank > i32::MAX as u32
        || c.batch_size == 0
        || c.gradient_accumulation == 0
        || r.context_tokens > 4096
    {
        return Err(invalid("positive representable geometry, rank, dataset/batch/accumulation and <=4096 context required"));
    }
    if f.encoder_bytes == 0 || f.vae_bytes == 0 || f.backbone_elements == 0 {
        return Err(invalid("missing component header facts"));
    }
    let width = compute_width(c)?;
    let mut canvases: Vec<(u64, u64)> = if c.aspect_ratio_buckets.is_empty() {
        r.edges.iter().map(|&edge| (edge, edge)).collect()
    } else {
        c.aspect_ratio_buckets
            .iter()
            .map(|row| (u64::from(row.width), u64::from(row.height)))
            .collect()
    };
    if let Some(canvas) = r.evaluation_canvas {
        canvases.push(canvas);
    }
    let token_areas = canvases
        .iter()
        .map(|&(width, height)| {
            if width == 0 || height == 0 || !width.is_multiple_of(16) || !height.is_multiple_of(16)
            {
                return Err(invalid("invalid training canvas"));
            }
            mul(&[width / 16, height / 16], "training canvas tokens")
        })
        .collect::<Result<Vec<_>>>()?;
    let largest_tokens = token_areas
        .iter()
        .copied()
        .max()
        .ok_or_else(|| invalid("no training canvas"))?;
    let total_cache_tokens = match r.cache_token_count {
        Some(tokens) => tokens,
        None => mul(
            &[r.items, sum(token_areas, "cache canvas tokens")?],
            "cache tokens",
        )?,
    };
    let cached_latents = if c.cache_latents {
        mul(&[total_cache_tokens, channels, 4], "latent cache")?
    } else {
        0
    };
    let mask_cache = if c.subject_mask_loss.is_some() {
        cached_latents
    } else {
        0
    };
    let context_count = if c.cache_text_embeddings { r.items } else { 1 };
    let contexts = mul(
        &[
            sum([context_count, r.preview_contexts], "context count")?,
            r.context_tokens,
            text,
            4,
        ],
        "ragged context cache",
    )?;
    let evaluation_tokens = if let Some((w, h)) = r.evaluation_canvas {
        mul(&[w / 16, h / 16], "heldout cached tokens")?
    } else {
        0
    };
    let evaluation_cache = sum(
        [
            mul(&[evaluation_tokens, channels, 4, 2], "heldout latent/mask")?,
            mul(&[r.evaluation_context_tokens, text, 4], "heldout context")?,
        ],
        "heldout cached entry",
    )?;
    let cache_bytes = sum(
        [cached_latents, mask_cache, contexts, evaluation_cache],
        "persistent caches",
    )?;
    let encoding_work = canvases
        .iter()
        .map(|&(w, h)| vae_work_canvas(&f.vae_headers, w, h, false))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .ok_or_else(|| invalid("no VAE canvas"))?;
    let backbone_compute = mul(&[f.backbone_elements, width], "compute backbone")?;
    let backbone_bf16 = mul(&[f.backbone_elements, 2], "BF16 source backbone")?;
    let source_backbone = f.backbone_source_bytes.max(backbone_bf16);
    let frozen = sum(
        [
            if !c.cache_latents || r.preview_contexts > 0 {
                f.vae_bytes
            } else {
                0
            },
            if !c.cache_text_embeddings {
                f.encoder_bytes
            } else {
                0
            },
            backbone_compute,
        ],
        "frozen training components",
    )?;
    let loaded = sum(
        [f.encoder_bytes, f.vae_bytes, source_backbone],
        "loaded components",
    )?;
    let load = sum([loaded], "loaded device components")?;
    // Existing projection custom/base handles may coexist with cast outputs; preserve the BF16
    // source and full cast destination until the parent has evaluated/replaced those handles.
    let cast = sum([loaded, backbone_compute], "device cast peak")?;
    let cache = sum(
        [
            f.encoder_bytes,
            f.vae_bytes,
            backbone_compute,
            cache_bytes,
            encoder_work(&f.encoder_config, r.context_tokens)?.max(encoding_work),
        ],
        "cache phase",
    )?;
    let on_demand = sum(
        [
            if !c.cache_text_embeddings {
                encoder_work(&f.encoder_config, r.context_tokens)?
            } else {
                0
            },
            if !c.cache_latents {
                sum(
                    [
                        encoding_work,
                        mul(&[largest_tokens, channels, 4, 2], "active latent and mask")?,
                    ],
                    "on-demand VAE",
                )?
            } else {
                0
            },
        ],
        "on-demand conditioning",
    )?;

    let tokens = sum([largest_tokens, r.context_tokens], "joint sequence")?;
    let preview_tokens = if r.preview_contexts > 0 {
        sum(
            [image_tokens(r.preview_edge)?, r.context_tokens],
            "preview joint sequence",
        )?
    } else {
        0
    };
    let mut trainable = Vec::new();
    let mut adapter_work = std::collections::BTreeMap::<usize, u64>::new();
    let mut preview_adapter_work = std::collections::BTreeMap::<usize, u64>::new();
    for path in target_paths(g.num_layers, &c.lora_target_modules).map_err(source)? {
        let (out, input) = if path.ends_with(".ffn_up_gate") {
            (mul(&[2, wide], "fused FFN width")?, d)
        } else if path.ends_with(".ffn_down") {
            (d, wide)
        } else {
            (d, d)
        };
        let (p, work) = adapter_shape(out, input, c, tokens)?;
        trainable.push(p);
        let layer = path
            .split('.')
            .nth(1)
            .ok_or_else(|| invalid("missing target layer"))?
            .parse::<usize>()
            .map_err(|_| invalid("invalid target layer"))?;
        let old = adapter_work.entry(layer).or_default();
        *old = sum([*old, work], "block adapter work")?;
        if preview_tokens > 0 {
            let old = preview_adapter_work.entry(layer).or_default();
            *old = sum(
                [*old, adapter_shape(out, input, c, preview_tokens)?.1],
                "preview block adapter work",
            )?;
        }
    }
    let trainable_elements = sum(trainable, "all trainable factors")?;
    let adapter_work_bytes = if c.gradient_checkpointing {
        adapter_work
            .values()
            .copied()
            .max()
            .ok_or_else(|| invalid("no selected targets"))?
    } else {
        sum(adapter_work.into_values(), "all structured residual work")?
    };
    // FP32 master; current/token-weighted gradients; old/new micro and accumulation sums;
    // averaged/clipped gradients; updated factors; old/new optimizer states and three update
    // arithmetic buffers. Summing these named lifetimes is an upper bound, not all overlap.
    let factor_buffers = sum(
        [
            1,
            2,
            if c.batch_size > 1 { 2 } else { 1 },
            if c.gradient_accumulation > 1 { 2 } else { 1 },
            2,
            1,
            mul(
                &[2, optimizer_states(&c.optimizer)?],
                "optimizer old/new states",
            )?,
            3,
            if c.weight_noise_sigma > 0.0 { 3 } else { 0 },
            if c.gradient_noise_eta > 0.0 { 3 } else { 0 },
            if c.resume { 1 } else { 0 },
        ],
        "factor buffer count",
    )?;
    let factors = mul(
        &[trainable_elements, 4, factor_buffers],
        "masters/gradients/optimizer/noise",
    )?;
    // Training residuals and SDPA are FP32 even with BF16 frozen-base compute. Do not price
    // these as BF16. Explicit attention + its softmax VJP: scores, p, dp, p*dp, ds (five S²
    // matrices). Checkpointing replays one block; dense retains every block's graph.
    let retained = if c.gradient_checkpointing { 1 } else { layers };
    let attention_backward_bytes = mul(
        &[5, 4, heads, tokens, tokens, retained],
        "FP32 attention backward",
    )?;
    // Per-block forward: two RMS norms (input/square/normalized/affine); QKV; Q/K norms;
    // two RoPEs (six hidden-sized intermediate planes each); attention result/O; residuals
    // and modulation. SiTU's hidden branch has three wide ops, gate has five, plus fused
    // up/gate (two widths) and product (one). Reserve equal-shaped VJP outputs as well.
    let hidden_ops = sum([2 * 4, 3, 2 * 4, 2 * 6, 2, 2, 4], "block hidden operations")?;
    let forward = mul(
        &[
            tokens,
            sum(
                [
                    mul(&[hidden_ops, d], "block hidden planes")?,
                    mul(&[11, wide], "SiTU wide planes")?,
                ],
                "block forward planes",
            )?,
            4,
            2,
            retained,
        ],
        "forward plus VJP activations",
    )?;
    // Frozen base linears still propagate input gradients; no base-weight gradients are
    // allocated. Q/K/V/O/up consume five hidden inputs; down consumes one FFN input.
    let input_planes = sum(
        [mul(&[5, d], "frozen hidden inputs")?, wide],
        "frozen linear inputs",
    )?;
    let frozen_input_grad_bytes = mul(
        &[
            tokens,
            input_planes,
            sum([4, width], "input grad plus compute cast")?,
            retained,
        ],
        "frozen input gradients",
    )?;
    let boundaries = if c.gradient_checkpointing {
        mul(
            &[sum([layers, 1], "checkpoint boundaries")?, tokens, d, 4],
            "checkpoint boundary stream",
        )?
    } else {
        0
    };
    // Two frozen text-refiner blocks execute before the joint trainable stack. Bound their
    // live norm/QKV/SiTU outputs and scores independently of the main-block checkpoint policy.
    let text_refiner = sum(
        [
            mul(
                &[
                    2,
                    r.context_tokens,
                    sum(
                        [
                            mul(&[hidden_ops, d], "text refiner hidden")?,
                            mul(&[11, wide], "text refiner SiTU")?,
                        ],
                        "text refiner planes",
                    )?,
                    4,
                ],
                "text refiner live",
            )?,
            mul(
                &[2, heads, r.context_tokens, r.context_tokens, 4],
                "text refiner scores",
            )?,
        ],
        "text refiner work",
    )?;
    // Noise, flow interpolation terms, target, prediction, difference, loss/mask and their VJPs.
    let flow = mul(
        &[largest_tokens, channels, 4, 12],
        "noise/flow/loss buffers",
    )?;
    // Global embedding/concatenation and final-head norm/modulation/input-gradient buffers:
    // four joint-stream hidden planes, two latent-input planes and time/shared/head modulation.
    let globals = mul(
        &[
            sum(
                [
                    mul(&[4, tokens, d], "global stream planes")?,
                    mul(&[2, largest_tokens, channels], "global latent planes")?,
                    dim(g.freq_dim)?,
                    mul(&[12, d], "time and modulation planes")?,
                ],
                "global forward/VJP elements",
            )?,
            4,
            2,
        ],
        "global forward/VJP bytes",
    )?;
    let backward = sum(
        [
            frozen,
            cache_bytes,
            factors,
            on_demand,
            adapter_work_bytes,
            attention_backward_bytes,
            forward,
            frozen_input_grad_bytes,
            boundaries,
            text_refiner,
            flow,
            globals,
        ],
        "backward phase",
    )?;
    let preview = if r.preview_contexts == 0 {
        0
    } else {
        // Positive/negative sampling is sequential. Current adapter/optimizer/cache remain live;
        // one full FP32 forward envelope also bounds the production per-head f32 inference path.
        sum(
            [
                frozen,
                cache_bytes,
                factors,
                mul(
                    &[
                        preview_tokens,
                        sum(
                            [
                                mul(&[hidden_ops, d], "preview hidden")?,
                                mul(&[11, wide], "preview SiTU")?,
                            ],
                            "preview planes",
                        )?,
                        4,
                    ],
                    "preview forward",
                )?,
                mul(
                    &[2, heads, preview_tokens, preview_tokens, 4],
                    "preview scores",
                )?,
                vae_work(&f.vae_headers, r.preview_edge, true)?,
                preview_adapter_work
                    .values()
                    .copied()
                    .max()
                    .ok_or_else(|| invalid("no preview adapter target"))?,
            ],
            "preview phase",
        )?
    };
    // Complete raw-factor/gradient/optimizer serialization plus restore staging. This is
    // structural residency accounting, not an empirical peak or a new measurement gate.
    let checkpoint = {
        sum(
            [
                backward,
                mul(
                    &[
                        trainable_elements,
                        4,
                        sum(
                            [
                                4,
                                mul(
                                    &[2, optimizer_states(&c.optimizer)?],
                                    "checkpoint optimizer copies",
                                )?,
                            ],
                            "checkpoint tensor copies",
                        )?,
                    ],
                    "checkpoint serialization staging",
                )?,
            ],
            "checkpoint phase",
        )?
    };
    let target_pixels = canvases
        .iter()
        .map(|&(w, h)| mul(&[w, h], "host training canvas"))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .ok_or_else(|| invalid("no host canvas"))?;
    let host_preprocess = sum(
        [
            r.decode_bytes,
            mul(
                &[target_pixels, 48],
                "host Lanczos/CHW normalization staging",
            )?,
        ],
        "host input work",
    )?;
    let host_serialization = mul(
        &[
            trainable_elements,
            4,
            sum(
                [
                    8,
                    mul(
                        &[4, optimizer_states(&c.optimizer)?],
                        "host optimizer serialization copies",
                    )?,
                ],
                "host checkpoint copies",
            )?,
        ],
        "host serialization/restore staging",
    )?;
    let host_peak_bytes = sum(
        [
            f.host_metadata_bytes,
            r.dataset_metadata_bytes,
            r.audit_workspace_bytes,
            f.largest_source_shard
                .max(host_preprocess)
                .max(host_serialization),
        ],
        "host phase peak",
    )?;
    Ok(MemoryEstimate {
        load: sum(
            [
                load,
                if r.audit_workspace_bytes > 0 {
                    16 * 1024 * 1024
                } else {
                    0
                },
            ],
            "load audit staging",
        )?,
        cast: sum(
            [
                cast,
                if r.audit_workspace_bytes > 0 {
                    16 * 1024 * 1024
                } else {
                    0
                },
            ],
            "cast audit staging",
        )?,
        cache: sum(
            [
                cache,
                if r.audit_workspace_bytes > 0 {
                    16 * 1024 * 1024
                } else {
                    0
                },
            ],
            "cache audit staging",
        )?,
        backward: sum(
            [
                backward,
                if r.audit_workspace_bytes > 0 {
                    16 * 1024 * 1024
                } else {
                    0
                },
            ],
            "backward audit staging",
        )?,
        preview: if preview > 0 {
            sum(
                [
                    preview,
                    if r.audit_workspace_bytes > 0 {
                        16 * 1024 * 1024
                    } else {
                        0
                    },
                ],
                "preview audit staging",
            )?
        } else {
            0
        },
        checkpoint: sum(
            [
                checkpoint,
                if r.audit_workspace_bytes > 0 {
                    16 * 1024 * 1024
                } else {
                    0
                },
            ],
            "checkpoint audit staging",
        )?,
        context_token_bound: r.context_tokens,
        host_peak_bytes,
        cache_bytes,
        trainable_elements,
        adapter_work_bytes,
        attention_backward_bytes,
        frozen_input_grad_bytes,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use candle_gen::gen_core::{weightsmeta::Dtype, ResolutionBucket};

    fn header(name: &str, shape: &[usize], width: u64) -> SafetensorsTensorHeader {
        let n = shape.iter().map(|n| *n as u64).product::<u64>();
        SafetensorsTensorHeader {
            name: name.into(),
            dtype: Dtype::BF16,
            shape: shape.to_vec(),
            data_bytes: n * width,
        }
    }
    pub(crate) fn tiny() -> (FootprintFacts, EstimateRequest) {
        let encoder = Qwen35Config::from_json(&serde_json::json!({
            "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 4,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
            "vocab_size": 32, "linear_num_value_heads": 2, "linear_num_key_heads": 1,
            "linear_key_head_dim": 2, "linear_value_head_dim": 4, "linear_conv_kernel_dim": 2,
            "full_attention_interval": 4
        }))
        .unwrap();
        let f = FootprintFacts::from_headers(
            MossImageConfig {
                dim: 8,
                ffn_dim: 16,
                num_heads: 2,
                num_layers: 3,
                text_dim: 8,
                freq_dim: 4,
                latent_channels: 128,
            },
            encoder,
            &[
                (
                    AssetKind::Encoder,
                    vec![
                        header("language_model.embed_tokens.weight", &[32, 8], 2),
                        header("language_model.layers.0.mlp.down_proj.weight", &[8, 16], 2),
                        header("language_model.layers.3.norm.weight", &[8], 2),
                    ],
                ),
                (
                    AssetKind::Backbone,
                    vec![header("layers.0.proj.q.weight", &[8, 8], 2)],
                ),
                (
                    AssetKind::Vae,
                    vec![
                        header("encoder.conv_in.weight", &[4, 3, 3, 3], 2),
                        header("decoder.conv_in.weight", &[4, 32, 3, 3], 2),
                        header("bn.running_mean", &[128], 4),
                        header("bn.running_var", &[128], 4),
                    ],
                ),
            ],
        )
        .unwrap();
        let r = EstimateRequest {
            config: TrainingConfig {
                rank: 1,
                resolution: 32,
                train_dtype: "bf16".into(),
                gradient_checkpointing: true,
                ..Default::default()
            },
            items: 2,
            edges: vec![32],
            context_tokens: 4,
            preview_contexts: 2,
            preview_edge: 32,
            decode_bytes: 0,
            dataset_metadata_bytes: 0,
            cache_token_count: None,
            evaluation_canvas: None,
            evaluation_context_tokens: 0,
            audit_workspace_bytes: 0,
        };
        (f, r)
    }
    #[test]
    fn every_phase_rejects_one_byte_below_and_accepts_boundary() {
        // Mutation TM01: replace `required > budget` by false (or >=), or omit a phase in check_all.
        let (f, r) = tiny();
        let e = estimate(&f, &r).unwrap();
        for phase in [
            Phase::Load,
            Phase::Cast,
            Phase::Cache,
            Phase::Backward,
            Phase::Preview,
            Phase::Checkpoint,
        ] {
            let n = e.bytes(phase);
            assert!(
                matches!(e.check(phase, n - 1), Err(AdmissionError::BudgetExceeded { phase: p, required, budget }) if p == phase && required == n && budget == n - 1)
            );
            assert!(e.check(phase, n).is_ok());
            let mut only = e.clone();
            only.load = 0;
            only.cast = 0;
            only.cache = 0;
            only.backward = 0;
            only.preview = 0;
            match phase {
                Phase::Load => only.load = n,
                Phase::Cast => only.cast = n,
                Phase::Cache => only.cache = n,
                Phase::Backward => only.backward = n,
                Phase::Preview => only.preview = n,
            }
            assert!(only.check_all(n - 1).is_err());
        }
        assert!(e.check_all(e.peak()).is_ok());
    }
    #[test]
    fn cpu_has_no_fabricated_cuda_memory_budget() {
        assert!(device_budget_bytes(&candle_gen::candle_core::Device::Cpu).is_err());
    }
    #[test]
    fn override_resolution_is_pure_and_zero_is_not_a_bypass() {
        // Mutation TM02: ignore Some(0), ignore a nonzero override, or ignore the device fallback.
        assert_eq!(resolve_budget(Some(0), 123), 0);
        assert_eq!(resolve_budget(Some(17), 123), 17);
        assert_eq!(resolve_budget(None, 123), 123);
        let (f, r) = tiny();
        assert!(estimate(&f, &r)
            .unwrap()
            .check_all(resolve_budget(Some(0), u64::MAX))
            .is_err());
    }
    #[test]
    fn source_image_headers_price_decode_before_target_bucket_resize() {
        let (f, mut r) = tiny();
        let baseline = estimate(&f, &r).unwrap();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("source.png");
        image::RgbImage::from_pixel(32, 64, image::Rgb([1, 2, 3]))
            .save(&path)
            .unwrap();
        let request = TrainingRequest {
            items: vec![candle_gen::gen_core::TrainingItem::captioned(path.clone(), "x".into()); 2],
            config: r.config.clone(),
            output_dir: temp.path().into(),
            file_name: "x.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: Default::default(),
        };
        let source_bytes = image_decode_bytes(&path).unwrap();
        r = r.with_image_headers(&request).unwrap();
        let pinned = gen_core::moss_image_training::recovery::DataPlan::from_request_budget(
            &request,
            2 * 1024 * 1024 * 1024,
        )
        .unwrap();
        assert_eq!(r.decode_bytes, pinned.decode_buffer_bound().unwrap());
        assert!(r.decode_bytes>=source_bytes,"encoded/decoded buffers must include the captured source and library decoder allowance");
        let actual = estimate(&f, &r).unwrap();
        assert_eq!(
            actual.cache, baseline.cache,
            "pure host image decode must not raise CUDA VRAM demand"
        );
        assert!(actual.host_peak_bytes > baseline.host_peak_bytes);
        assert!(actual
            .check_host(actual.host_peak_bytes - 1, false)
            .is_err());
        actual.check_host(actual.host_peak_bytes, false).unwrap();
        std::fs::write(&path, b"not an image").unwrap();
        assert!(image_decode_bytes(&path).is_err());
        assert!(r.with_image_headers(&request).is_err());
    }
    #[test]
    fn headers_price_full_tower_full_vae_stats_and_largest_shard() {
        // Mutation TM03: price encoder as selected layers only; omit decoder or FP32 BN stats;
        // use on-disk VAE widths instead of 4; exclude unselected payload from source staging.
        let (f, _) = tiny();
        assert_eq!(f.encoder_bytes, (32 * 8 + 8 * 16 + 8) * 2);
        assert_eq!(f.vae_bytes, (4 * 3 * 3 * 3 + 4 * 32 * 3 * 3 + 128 * 2) * 4);
        let mut larger = f.vae_headers.clone();
        larger.push(header("unused.payload", &[4096], 2));
        let h = FootprintFacts::from_headers(
            f.config.clone(),
            f.encoder_config.clone(),
            &[
                (
                    AssetKind::Encoder,
                    vec![header("language_model.embed_tokens.weight", &[32, 8], 2)],
                ),
                (AssetKind::Backbone, vec![header("q.weight", &[8, 8], 2)]),
                (AssetKind::Vae, larger),
            ],
        )
        .unwrap();
        assert_eq!(h.largest_source_shard, f.largest_source_shard + 4096 * 2);
        assert_eq!(h.vae_bytes, f.vae_bytes);
    }
    #[test]
    fn cache_sums_bucket_areas_but_contexts_share_across_buckets() {
        // Mutation TM04: use max area for cache, forget items, or duplicate context per bucket.
        let (f, mut r) = tiny();
        let a = estimate(&f, &r).unwrap();
        r.edges.push(64);
        let b = estimate(&f, &r).unwrap();
        assert_eq!(
            b.cache_bytes - a.cache_bytes,
            r.items * (64 / 16u64).pow(2) * 128 * 4
        );
        r.items += 1;
        let c = estimate(&f, &r).unwrap();
        assert_eq!(c.cache - b.cache, c.cache_bytes - b.cache_bytes);
        assert_eq!(c.backward - b.backward, c.cache_bytes - b.cache_bytes);
        assert_eq!(
            c.cache_bytes - b.cache_bytes,
            ((32 / 16u64).pow(2) + (64 / 16u64).pow(2)) * 128 * 4 + r.context_tokens * 8 * 4
        );
    }
    #[test]
    fn ragged_contexts_and_preview_resolution_are_not_ignored() {
        // Mutation TM05: use fixed/mean context length; drop preview conditioning or preview edge.
        let (f, mut r) = tiny();
        let a = estimate(&f, &r).unwrap();
        r.context_tokens *= 2;
        let b = estimate(&f, &r).unwrap();
        assert!(b.cache_bytes > a.cache_bytes && b.cache > a.cache && b.backward > a.backward);
        r.preview_contexts += 1;
        let c = estimate(&f, &r).unwrap();
        assert_eq!(c.preview - b.preview, c.cache_bytes - b.cache_bytes);
        assert_eq!(c.cache_bytes - b.cache_bytes, r.context_tokens * 8 * 4);
        r.preview_edge *= 2;
        let d = estimate(&f, &r).unwrap();
        let decoder_growth = vae_work(&f.vae_headers, r.preview_edge, true).unwrap()
            - vae_work(&f.vae_headers, r.preview_edge / 2, true).unwrap();
        assert!(
            d.preview - c.preview > decoder_growth,
            "preview sequence/attention/adapter work ignored the larger preview geometry"
        );
    }
    #[test]
    fn rank_suffix_selection_dtype_and_optimizer_are_structural() {
        // Mutation TM06: constant factor count, ignore suffixes, price F32 casts as BF16,
        // or treat optimizer state as absent. The exact expected count derives target geometry.
        let (f, mut r) = tiny();
        let a = estimate(&f, &r).unwrap();
        let per_layer = 4 * (8 + 8) + (2 * 16 + 8) + (8 + 16);
        assert_eq!(a.trainable_elements, 3 * per_layer);
        r.config.rank = 2;
        let b = estimate(&f, &r).unwrap();
        assert_eq!(b.trainable_elements, 2 * a.trainable_elements);
        r.config.lora_target_modules = vec!["q".into()];
        let q = estimate(&f, &r).unwrap();
        assert_eq!(q.trainable_elements, 3 * 2 * (8 + 8));
        r.config.train_dtype = "f32".into();
        let fp32 = estimate(&f, &r).unwrap();
        assert!(fp32.cast > q.cast && fp32.backward > q.backward);
        r.config.optimizer = "rose".into();
        let rose = estimate(&f, &r).unwrap();
        assert!(rose.backward < fp32.backward);
        r.config.lora_target_modules = vec!["not_a_target".into()];
        assert!(estimate(&f, &r).is_err());
    }
    #[test]
    fn quadratic_backward_remains_fp32_with_checkpoint_and_frozen_input_grads() {
        // Mutation TM07: replace FP32 width by compute width; retain zero/too many checkpoint
        // blocks; omit frozen input-grad buffers; price dense as one block.
        let (f, mut r) = tiny();
        let checkpoint = estimate(&f, &r).unwrap();
        let seq = (32 / 16u64).pow(2) + r.context_tokens;
        assert_eq!(checkpoint.attention_backward_bytes, 5 * 4 * 2 * seq * seq);
        assert_eq!(
            checkpoint.frozen_input_grad_bytes,
            seq * (5 * 8 + 16) * (4 + 2)
        );
        r.config.gradient_checkpointing = false;
        let dense = estimate(&f, &r).unwrap();
        assert_eq!(
            dense.attention_backward_bytes,
            checkpoint.attention_backward_bytes * 3
        );
        assert!(dense.backward > checkpoint.backward);
        assert_eq!(
            dense.adapter_work_bytes,
            checkpoint.adapter_work_bytes * f.config.num_layers as u64
        );
    }
    #[test]
    fn structured_lokr_uses_factors_and_never_a_dense_backbone_delta() {
        // Mutation TM08: charge out*in for LoKr trainable factors, omit decomposed rank factors,
        // remove structured intermediate/small reconstruction, or restore full out*in delta work.
        let (_, mut r) = tiny();
        r.config.network_type = NetworkType::Lokr;
        let (p, work) = adapter_shape(64, 64, &r.config, 3).unwrap();
        // factorization(64,-1)=(8,8), first8*8 + rank1*(8+8); structured buffers
        // 3*(8*8 + 1*8 + 8*1 + 64), plus only the two 8*8 small products.
        assert_eq!(p, 8 * 8 + (8 + 8));
        assert_eq!(work, (3 * (64 + 8 * 8 + 8 + 4 * 64) * 2 + 2 * 8 * 8) * 4);
        r.config.rank = 8;
        assert_eq!(adapter_shape(64, 64, &r.config, 3).unwrap().0, 2 * 8 * 8);
    }
    #[test]
    fn cache_cast_and_source_staging_are_not_backward_only_guards() {
        // Mutation TM09: omit encoder, source shard, source/destination cast coexistence, or cache.
        let (f, r) = tiny();
        let e = estimate(&f, &r).unwrap();
        assert_eq!(
            e.load,
            f.encoder_bytes + f.vae_bytes + f.backbone_source_bytes.max(f.backbone_elements * 2)
        );
        assert_eq!(e.cast, e.load + f.backbone_elements * 2);
        assert!(e.host_peak_bytes >= f.largest_source_shard + f.host_metadata_bytes);
        let mut host_growth = f.clone();
        host_growth.largest_source_shard += e.host_peak_bytes;
        let host_growth = estimate(&host_growth, &r).unwrap();
        assert_eq!(host_growth.load, e.load);
        assert!(host_growth.host_peak_bytes > e.host_peak_bytes);

        assert!(e.cache >= f.encoder_bytes + f.vae_bytes + f.backbone_elements * 2 + e.cache_bytes);
        let mut doubled = f.clone();
        doubled.encoder_bytes *= 2;
        let twice = estimate(&doubled, &r).unwrap();
        assert_eq!(twice.cache - e.cache, f.encoder_bytes);
        assert_eq!(
            twice.backward, e.backward,
            "encoder must be freed before backward"
        );
    }
    #[test]
    fn optimizer_noise_resume_and_accumulation_buffers_are_priced() {
        // Mutation TM12: individually remove the weight-noise, gradient-noise, resume, batch,
        // accumulation or optimizer-state term from factor_buffers; the same exact delta fails.
        let (f, mut r) = tiny();
        let a = estimate(&f, &r).unwrap();
        r.config.weight_noise_sigma = 0.01;
        r.config.gradient_noise_eta = 0.01;
        r.config.gradient_accumulation = 2;
        r.config.batch_size = 2;
        r.config.resume = true;
        r.config.optimizer = "prodigy".into();
        let b = estimate(&f, &r).unwrap();
        assert_eq!(
            b.backward - a.backward,
            a.trainable_elements * 4 * (3 + 3 + 1 + 1 + 1 + 4)
        );
    }
    #[test]
    fn overflow_and_invalid_metadata_fail_closed() {
        // Mutation TM10: replace checked arithmetic with saturation/wrapping/default success;
        // accept empty component metadata, nonpositive dimensions, unrepresentable rank or dtype.
        let (mut f, mut r) = tiny();
        r.items = u64::MAX;
        assert!(matches!(estimate(&f, &r), Err(AdmissionError::Overflow(_))));
        r.items = 2;
        f.backbone_elements = u64::MAX;
        assert!(matches!(estimate(&f, &r), Err(AdmissionError::Overflow(_))));
        let (mut f, mut r) = tiny();
        f.encoder_bytes = 0;
        assert!(estimate(&f, &r).is_err());
        f.encoder_bytes = 1;
        r.config.rank = u32::MAX;
        assert!(estimate(&f, &r).is_err());
        r.config.rank = 1;
        r.config.train_dtype = "unknown".into();
        assert!(estimate(&f, &r).is_err());
        r.config.train_dtype = "bf16".into();
        f.encoder_config.hidden_size = -1;
        assert!(estimate(&f, &r).is_err());
    }
    #[test]
    fn request_mapping_uses_all_buckets_and_conservative_context_bound() {
        // Mutation TM11: map only primary resolution; omit items/previews; use measured captions.
        let req = TrainingRequest {
            config: TrainingConfig {
                resolution: 96,
                resolution_buckets: vec![
                    ResolutionBucket {
                        resolution: 32,
                        repeats: 2,
                    },
                    ResolutionBucket {
                        resolution: 64,
                        repeats: 1,
                    },
                ],
                sample_every: 1,
                sample_prompts: vec!["x".into(); 8],
                ..Default::default()
            },
            items: vec![candle_gen::gen_core::TrainingItem::default(); 2],
            output_dir: Default::default(),
            file_name: "x".into(),
            trigger_words: Default::default(),
            cancel: Default::default(),
        };
        let r = EstimateRequest::from_request(&req).unwrap();
        assert_eq!(
            (
                r.items,
                r.edges,
                r.context_tokens,
                r.preview_contexts,
                r.preview_edge
            ),
            (2, vec![32, 64], 4096, 5, 96)
        );
    }
}
