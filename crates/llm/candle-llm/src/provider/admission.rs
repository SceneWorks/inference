//! Load-memory admission and pricing for [`LlamaProvider`](super::LlamaProvider) (E6, sc-24140):
//! the working set a load holds per allocation domain, the quantized / cast copies it builds
//! beside its source, the safetensors / GGUF header scans that price them without reading tensor
//! data, and the budgets they are admitted against. Split out of `provider.rs` as pure code motion
//! (sc-27302); the provider's tests still exercise it through the parent module.

use super::*;

/// One safetensors file's JSON header (every tensor's `dtype`, `shape` and offsets, plus any
/// `__metadata__`): the 8-byte length prefix and the header only, never tensor data.
pub(super) fn safetensors_header(path: &Path) -> Option<serde_json::Map<String, Value>> {
    read_safetensors_header(path).ok()
}

/// A real safetensors header is kilobytes to a few megabytes; never allocate for a corrupt length.
pub(crate) const MAX_SAFETENSORS_HEADER_BYTES: u64 = 256 << 20;

/// The fallible form of [`safetensors_header`] shared with the snapshot preparer (sc-27298): the
/// declared header length is bounded by [`MAX_SAFETENSORS_HEADER_BYTES`] *before* allocating, so a
/// corrupt or hostile length prefix is a typed error rather than an allocation abort.
pub(crate) fn read_safetensors_header(
    path: &Path,
) -> crate::error::Result<serde_json::Map<String, Value>> {
    use std::io::Read;
    let malformed = |why: String| {
        crate::error::Error::Msg(format!("safetensors header {}: {why}", path.display()))
    };
    let mut file = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len)
        .map_err(|e| malformed(format!("length prefix: {e}")))?;
    let len = u64::from_le_bytes(len);
    if len > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(malformed(format!(
            "declared length {len} exceeds the {MAX_SAFETENSORS_HEADER_BYTES}-byte cap"
        )));
    }
    // Nor allocate more than the file backs: a tiny file declaring a cap-sized header is short.
    let file_len = file.metadata()?.len();
    if len > file_len.saturating_sub(8) {
        return Err(malformed(format!(
            "declared length {len} exceeds the {file_len}-byte file"
        )));
    }
    let mut header = vec![0u8; len as usize];
    file.read_exact(&mut header)
        .map_err(|e| malformed(format!("{len}-byte header: {e}")))?;
    match serde_json::from_slice(&header).map_err(|e| malformed(e.to_string()))? {
        Value::Object(map) => Ok(map),
        _ => Err(malformed("not a JSON object".into())),
    }
}

/// Conservative load bounds for the two independent allocation domains.
///
/// Candle reads one safetensors shard into a host `Vec`, copies each tensor directly to CUDA at its
/// stored dtype, and drops the shard buffer before reading the next shard. The dense Qwen3.8
/// checkpoint is BF16 and `Tensor::to_dtype(BF16)` shares storage, so CUDA construction does not
/// retain a second expanded language-weight copy. The largest shard is therefore the host staging
/// bound; 25 percent device headroom covers constructor casts such as the BF16-to-F32 vision tower
/// while source tensors are still alive. CPU dense conversion retains source tensors plus
/// constructed F32 tensors. Packed CPU paths keep their existing two-copy upper bound, and external
/// projectors retain the audited four-copy conversion allowance.
///
/// The device holds the load's [`LoadWorkingSet`]: its resident source plus the 25 percent
/// headroom over it, and the copies the loader builds beside that source. A quantizing load
/// quantizes on the device while the source tensors are still resident (`Weights::from_dir` holds
/// every source tensor until the decoder is built), so the device additionally holds the growing
/// quantized copy of the projections the loader quantizes ([`snapshot_quantized_copy_bytes`]).
/// The GGML copies were unpriced until sc-24140's terminal measurement showed a Qwen3-8B Q8 load
/// peaking above the dense bound. The 25 percent headroom still covers the transient f32 copy of
/// the largest tensor the quantizer reads, and the CUDA pool's fragmentation around it.
///
/// A llama-family GGUF ([`gguf_working_set`]) holds its dense f32 map as the source and, on a
/// host device, prices the host domain by the same working set: there the f32 map *is* host
/// memory.
pub(super) fn load_memory_requirements(
    payload: u64,
    staging: u64,
    projector: u64,
    cuda: bool,
    working: LoadWorkingSet,
) -> Option<(u64, Option<u64>)> {
    let projector_bound = projector.checked_mul(4)?;
    let working_bytes = working
        .source
        .checked_add(working.source / 4)?
        .checked_add(working.cast)?
        .checked_add(working.copy)?;
    let host = match (working.host, cuda) {
        // Plus the stored GGML blocks the weight reader keeps on the host until the decoder is
        // built (sc-19375).
        (HostRule::Dense, true) => staging.checked_add(working.held)?,
        (HostRule::Dense, false) => payload.checked_mul(3)?,
        (HostRule::Packed, _) | (HostRule::Gguf, true) => payload.checked_mul(2)?,
        (HostRule::Gguf, false) => working_bytes,
    }
    .checked_add(projector_bound)?;
    let device = if cuda {
        Some(working_bytes.checked_add(projector_bound)?)
    } else {
        None
    };
    Some((host, device))
}

/// What [`LlamaProvider::load`] must find free before it reads a weight (E6), per allocation
/// domain — the figure the load admits against ([`LlamaProvider::load_memory_estimate`]), with
/// the parts of the device bound broken out. Only the estimate builds one (`#[non_exhaustive]`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadMemoryEstimate {
    /// The checkpoint payload on disk: every `*.safetensors` shard of the snapshot, or the file.
    pub payload_bytes: u64,
    /// Host bytes the load needs: one shard of staging on a CUDA load of a dense snapshot; on a
    /// host device a llama-family GGUF's whole working set (the f32 map it dequantizes into plus
    /// its quantized copy).
    pub host_required_bytes: u64,
    /// Device bytes a CUDA load needs (`None` on a host device): `source_bytes`, 25 percent
    /// headroom over it, `cast_copy_bytes`, `quantized_copy_bytes`, any external projector's
    /// bound and `graph_param_cache_bytes`.
    pub device_required_bytes: Option<u64>,
    /// The cross-turn prefix cache's budget the load asks for (sc-24437):
    /// [`LoadSpec::prefix_cache_bytes`], else the backend's defaults-table budget
    /// ([`core_llm::DecodeDefaults::prefix_cache_bytes`]). Memory
    /// the loaded model may hold **beside** the required bytes, in the domain the cache lives in
    /// (the device's on CUDA, the host's otherwise). It is not part of either required figure:
    /// `load` clamps it to the headroom its admission leaves, so it never refuses a load by
    /// itself, and reports the settled budget in [`core_llm::LoadReport::prefix_cache_bytes`].
    pub prefix_cache_bytes: u64,
    /// The source the load holds resident on the device while it builds the decoder: the
    /// payload of a safetensors or Prism checkpoint, the dense f32 map a llama-family GGUF is
    /// dequantized into (`0` off CUDA).
    pub source_bytes: u64,
    /// The compute-dtype (bf16) casts the load builds beside that source for the tensors it
    /// keeps dense: a llama-family GGUF's, cast from its f32 map, and a safetensors snapshot's
    /// tensors stored in a float dtype other than bf16 (f16, f32) (`0` for a bf16 snapshot, a
    /// Prism checkpoint, and off CUDA).
    pub cast_copy_bytes: u64,
    /// The quantized projection copy the load builds beside the source: exactly the projections
    /// the loader stores in the requested (or a snapshot's persisted) format, at the bytes the
    /// quantizer allocates for them (`0` for a dense or Prism load, and off CUDA).
    pub quantized_copy_bytes: u64,
    /// What the CUDA-graph runner's captures leave resident in the vendored candle's parameter
    /// cache for the rest of the process
    /// ([`graph_param_cache_load_bytes`](crate::decode::graph_param_cache_load_bytes), sc-24441):
    /// priced once per load because the cache outlives every request and the model itself.
    /// `0` off CUDA, with the graph runner off, and for a resident draft (never wrapped).
    pub graph_param_cache_bytes: u64,
}

/// How a load prices its host domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HostRule {
    /// A dense safetensors snapshot: one shard of staging on CUDA; the source plus the constructed
    /// f32 tensors (three payloads) on a host device.
    Dense,
    /// A packed Prism checkpoint: two payloads.
    Packed,
    /// A llama-family GGUF: tensor-at-a-time staging within two payloads on CUDA; the working set
    /// itself on a host device.
    Gguf,
}

/// What a load holds on the device it loads onto while it builds the decoder (E6, sc-24140).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LoadWorkingSet {
    /// The source tensors held resident until the decoder is built.
    pub(super) source: u64,
    /// Compute-dtype casts of the tensors kept dense, built beside the source.
    pub(super) cast: u64,
    /// The quantized projection copy, built beside the source.
    pub(super) copy: u64,
    /// Payload bytes held on the **host** rather than the device until the decoder is built: a
    /// prepared tier's stored GGML blocks, which the weight reader keeps host-side so each
    /// `QTensor` is built once on the device (sc-19375). Not part of `source`.
    pub(super) held: u64,
    /// How the host domain is priced.
    pub(super) host: HostRule,
}

impl LoadWorkingSet {
    /// A safetensors or Prism load: the payload itself is the resident source, plus the bf16
    /// `cast` of every tensor stored in another float dtype ([`snapshot_cast_copy_bytes`]) and
    /// the quantized `copy`.
    pub(super) fn resident_payload(payload: u64, cast: u64, copy: u64, packed: bool) -> Self {
        Self {
            source: payload,
            cast,
            copy,
            held: 0,
            host: if packed {
                HostRule::Packed
            } else {
                HostRule::Dense
            },
        }
    }

    /// Move `bytes` of the resident source to the host ([`LoadWorkingSet::held`]).
    pub(super) fn holding_on_host(self, bytes: u64) -> Self {
        Self {
            source: self.source.saturating_sub(bytes),
            held: bytes,
            ..self
        }
    }
}

/// The payload bytes of a snapshot's stored GGML block tensors
/// ([`ggml_block_storage`](crate::primitives::quant::ggml_block_storage)) — what
/// [`Weights::from_dir`] reads onto the host instead of the device (sc-19375). `None` on
/// overflow.
pub(super) fn stored_block_bytes(tensors: &SnapshotTensors) -> Option<u64> {
    tensors
        .values()
        .filter(|t| {
            crate::primitives::quant::ggml_block_storage(t.dtype == "U8", &t.shape).is_some()
        })
        .try_fold(0u64, |total, t| {
            let bytes = t
                .shape
                .iter()
                .try_fold(1u64, |n, &d| n.checked_mul(d as u64))?;
            total.checked_add(bytes)
        })
}

/// The projection format a load stores its quantized copy in, as admission prices it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CopyFormat {
    /// GGML Q4_K / Q8_0 (an explicit Q4 / Q8 request, or a snapshot's persisted block).
    Ggml(QuantSpec),
    /// NVFP4.
    Nvfp4,
}

impl CopyFormat {
    /// The format an explicit `spec.quantize` asks for — the specs `load` builds from it.
    pub(super) fn requested(quantize: Option<Quantize>) -> Option<Self> {
        Some(match quantize? {
            Quantize::Q4 => Self::Ggml(QuantSpec::q4()),
            Quantize::Q8 => Self::Ggml(QuantSpec::q8()),
            Quantize::Nvfp4 => Self::Nvfp4,
        })
    }

    /// Whether the loader keeps a `[rows, cols]` projection dense under this format: an NVFP4
    /// shape the FP4 GEMM cannot serve (the test `load_eligible` applies).
    pub(super) fn keeps_dense(self, rows: usize, cols: usize) -> bool {
        self == Self::Nvfp4 && nvfp4_device_bytes(rows, cols).is_none()
    }

    /// Device bytes of a dense `[rows, cols]` projection stored in this format: its GGML blocks,
    /// or its NVFP4 nibbles and scales — nothing for an NVFP4 shape `load_eligible` keeps dense
    /// (it shares the source's storage). `None` on overflow.
    pub(super) fn projection_bytes(self, rows: usize, cols: usize) -> Option<u64> {
        match self {
            Self::Ggml(quant) => {
                ggml_device_bytes((rows as u64).checked_mul(cols as u64)?, quant.dtype)
            }
            Self::Nvfp4 => Some(nvfp4_device_bytes(rows, cols).unwrap_or(0)),
        }
    }
}

/// The decoder a snapshot load builds, as admission prices its quantized copy (sc-24140).
pub(super) enum PricedDecoder {
    /// A Prism/Bonsai snapshot: packed blocks the loader wraps, never re-quantized.
    Prism,
    /// A llama-family [`CausalLm`].
    Causal(ModelConfig),
    /// The qwen3_5 hybrid ([`Qwen35Model`] and its [`Qwen35Mtp`]), and the stored-block format a
    /// prepared tier persists ([`crate::prepare::persisted_ggml_blocks`]).
    Qwen35(Qwen35Config, Option<QuantSpec>),
    /// No readable config, or one the loader refuses before it builds a projection.
    Refused,
}

impl PricedDecoder {
    /// An upper bound of the device tables the indexed MoE dispatch builds at a CUDA load
    /// (sc-24440): for every MoE layer (the MTP head's included), each of the three routed-expert
    /// projections holds [`MAX_TABLE_BYTES_PER_EXPERT`](candle_quant_kernels::MAX_TABLE_BYTES_PER_EXPERT)
    /// per expert at most. `0` for a dense model.
    pub(super) fn moe_indexed_table_bytes(&self) -> u64 {
        let per_layer = |experts: u64| {
            experts.saturating_mul(3 * candle_quant_kernels::MAX_TABLE_BYTES_PER_EXPERT as u64)
        };
        match self {
            PricedDecoder::Causal(cfg) => cfg.moe.map_or(0, |m| {
                let layers = cfg.num_layers.saturating_sub(m.first_k_dense_replace) as u64;
                layers.saturating_mul(per_layer(m.num_experts as u64))
            }),
            PricedDecoder::Qwen35(cfg, _) => cfg.moe.as_ref().map_or(0, |m| {
                let layers = cfg.num_layers.saturating_add(cfg.mtp_num_hidden_layers) as u64;
                layers.saturating_mul(per_layer(m.num_experts.max(0) as u64))
            }),
            PricedDecoder::Prism | PricedDecoder::Refused => 0,
        }
    }

    /// Dispatch `config` exactly as `load_dir` does.
    pub(super) fn from_config(config: Option<&Value>) -> Self {
        let Some(config) = config else {
            return Self::Refused;
        };
        if config.get("model_type").and_then(Value::as_str) == Some("prism_hadamard_qwen35") {
            return Self::Prism;
        }
        match Architecture::from_config(config) {
            Ok(Architecture::Qwen35) => Qwen35Config::from_json(config)
                .map_or(Self::Refused, |cfg| {
                    Self::Qwen35(cfg, crate::prepare::persisted_ggml_blocks(config))
                }),
            Ok(_) => ModelConfig::from_json(config).map_or(Self::Refused, Self::Causal),
            Err(_) => Self::Refused,
        }
    }

    pub(super) fn is_packed(&self) -> bool {
        matches!(self, Self::Prism)
    }

    /// The format the loader stores the projections in: the request; else, for a `CausalLm`,
    /// the snapshot's persisted `quantization` block (a prepared Q4 / Q8 snapshot, which
    /// `load_dir` re-quantizes on load or reads as stored blocks). The qwen3_5 loader honours
    /// only a stored-block (`storage: "ggml"`) block, so a bare persisted block there is not
    /// priced: that load keeps its projections dense.
    pub(super) fn format(&self, requested: Option<CopyFormat>) -> Option<CopyFormat> {
        match self {
            Self::Causal(cfg) => requested.or(cfg.quantization.map(CopyFormat::Ggml)),
            Self::Qwen35(_, persisted) => requested.or(persisted.map(CopyFormat::Ggml)),
            Self::Prism | Self::Refused => None,
        }
    }
}

/// A checkpoint tensor as its safetensors header describes it (never its data).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct TensorHeader {
    /// The safetensors dtype tag (`BF16`, `U32`, …).
    pub(super) dtype: String,
    /// The shape, outermost first.
    pub(super) shape: Vec<usize>,
}

impl TensorHeader {
    pub(super) fn dims2(&self) -> Option<(usize, usize)> {
        match self.shape[..] {
            [rows, cols] => Some((rows, cols)),
            _ => None,
        }
    }
}

/// Every tensor header of a snapshot, keyed by name.
pub(super) type SnapshotTensors = std::collections::HashMap<String, TensorHeader>;

/// The tensor headers of the shards `Weights::from_dir` reads — every `*.safetensors` file in
/// `dir`, in name order, a later shard's tensor replacing an earlier one's — from each shard's
/// JSON header only. An unreadable directory or header is [`CoreError::Load`] (the load would
/// fail reading it too).
pub(super) fn snapshot_tensor_headers(dir: &Path) -> CoreResult<SnapshotTensors> {
    let io_error =
        |error: std::io::Error| CoreError::Load(format!("checkpoint admission: {error}"));
    let mut shards: Vec<_> = std::fs::read_dir(dir)
        .map_err(io_error)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    let mut tensors = SnapshotTensors::new();
    for shard in shards {
        let malformed = || {
            CoreError::Load(format!(
                "checkpoint admission: unreadable safetensors header in {}",
                shard.display()
            ))
        };
        for (name, entry) in safetensors_header(&shard).ok_or_else(malformed)? {
            if name == "__metadata__" {
                continue;
            }
            let dtype = entry.get("dtype").and_then(Value::as_str);
            let shape = entry
                .get("shape")
                .and_then(Value::as_array)
                .and_then(|dims| {
                    dims.iter()
                        .map(|d| d.as_u64().and_then(|d| usize::try_from(d).ok()))
                        .collect::<Option<Vec<usize>>>()
                });
            let (Some(dtype), Some(shape)) = (dtype, shape) else {
                return Err(malformed());
            };
            tensors.insert(
                name,
                TensorHeader {
                    dtype: dtype.to_string(),
                    shape,
                },
            );
        }
    }
    Ok(tensors)
}

/// The quantized projection copy a snapshot load builds on the device beside its resident source
/// tensors (E6, sc-24140): exactly the tensors the loader stores in `format`, chosen by the
/// loader's own rule — for a `CausalLm` the layer projections
/// ([`quantizes_layer_tensor`](crate::models::llama::quantizes_layer_tensor) under
/// [`decoder_root`](crate::models::llama::decoder_root)) and, under NVFP4, the head; for the
/// qwen3_5 hybrid [`quantizes_tensor`](crate::models::qwen35::quantizes_tensor) over the decoder
/// and a configured MTP, and the head under every format — each at the bytes the quantizer
/// allocates for its shape ([`CopyFormat::projection_bytes`]). A tied head is quantized from a
/// copy of the embedding, so the embedding's shape prices it. Embeddings, norms, a GGML-dense
/// head and any vision tower are never priced: the loader never copies them.
///
/// A GGML format over an MLX-affine triple (a `.scales` sidecar) is a Q8_0 repack, priced only
/// where the loader performs one ([`affine_q8_repack_bytes`]). Returns the copy's bytes and the
/// float tensors the loader consumes into it whole — never kept dense, so never cast beside
/// their source ([`snapshot_cast_copy_bytes`]). `None` on overflow.
pub(super) fn snapshot_quantized_copy_bytes<'a>(
    decoder: &PricedDecoder,
    format: CopyFormat,
    tensors: &'a SnapshotTensors,
) -> Option<(u64, std::collections::HashSet<&'a str>)> {
    use crate::models::{llama, qwen35, split_layer_key};
    let mut total = 0u64;
    let mut consumed = std::collections::HashSet::new();
    let mut add = |bytes: u64| -> Option<()> {
        total = total.checked_add(bytes)?;
        Some(())
    };
    match decoder {
        PricedDecoder::Causal(cfg) => {
            let root = llama::decoder_root(cfg, "", |key| tensors.contains_key(key));
            for (key, tensor) in tensors {
                // The loader's own selection — the rule the preparer stores blocks by, too.
                if !llama::loads_quantized(cfg, &root, key) {
                    continue;
                }
                let Some((_, layer, suffix)) = split_layer_key(key) else {
                    continue;
                };
                let fused = matches!(
                    suffix,
                    "self_attn.qkv_proj.weight" | "mlp.gate_up_proj.weight"
                );
                // A stored GGML block tensor (a prepared Q4 / Q8 snapshot, sc-19375) is rebuilt
                // as `QTensor`s on the device — a fused one part by part — while its source
                // bytes stay on the host ([`stored_block_bytes`]).
                let stored = crate::primitives::quant::ggml_block_storage(
                    tensor.dtype == "U8",
                    &tensor.shape,
                );
                let stem = key.strip_suffix(".weight").unwrap_or(key);
                if let (None, Some(scales)) = (stored, tensors.get(&format!("{stem}.scales"))) {
                    // A fused triple is refused by the loader; an unfused one is repacked.
                    if !fused {
                        let biases = tensors.get(&format!("{stem}.biases"));
                        add(affine_q8_repack_bytes(format, tensor, scales, biases)?)?;
                    }
                    continue;
                }
                let Some((rows, cols)) = stored
                    .map(|(_, rows, cols)| (rows, cols))
                    .or_else(|| tensor.dims2())
                else {
                    continue;
                };
                // The loader splits a fused tensor into its q / k / v or gate / up rows and
                // quantizes each part on its own.
                let parts = match suffix {
                    "self_attn.qkv_proj.weight" => {
                        let la = cfg.layer_attention(layer);
                        let head_dim = la.head_dim as usize;
                        let kvd = la.num_kv_heads as usize * head_dim;
                        vec![cfg.num_heads as usize * head_dim, kvd, kvd]
                    }
                    "mlp.gate_up_proj.weight" => {
                        let inter = match cfg.gemma4.as_deref() {
                            Some(g) => g.layer_intermediate_size(layer, cfg.intermediate_size),
                            None => cfg.intermediate_size,
                        } as usize;
                        vec![inter, inter]
                    }
                    _ => vec![rows],
                };
                if let Some((dtype, _, _)) = stored {
                    for part in parts {
                        add(ggml_device_bytes(
                            (part as u64).checked_mul(cols as u64)?,
                            dtype,
                        )?)?;
                    }
                    continue;
                }
                // An NVFP4 part the FP4 GEMM cannot serve stays dense (as its source's cast).
                if !parts.iter().any(|&part| format.keeps_dense(part, cols)) {
                    consumed.insert(key.as_str());
                }
                for part in parts {
                    add(format.projection_bytes(part, cols)?)?;
                }
            }
            // `CausalLm` quantizes its head only under NVFP4; a tied head from the embedding.
            if format == CopyFormat::Nvfp4 {
                let head = if cfg.tie_word_embeddings {
                    format!("{root}.embed_tokens.weight")
                } else {
                    "lm_head.weight".to_string()
                };
                if let Some((rows, cols)) = tensors.get(&head).and_then(TensorHeader::dims2) {
                    // A tied head is a copy; the embedding itself stays dense.
                    if !cfg.tie_word_embeddings && !format.keeps_dense(rows, cols) {
                        consumed.insert("lm_head.weight");
                    }
                    add(format.projection_bytes(rows, cols)?)?;
                }
            }
        }
        PricedDecoder::Qwen35(cfg, _) => {
            let Ok(prefix) = qwen35_dense_prefix(|key| tensors.contains_key(key)) else {
                return Some((0, consumed));
            };
            for (key, tensor) in tensors {
                // The loader's own selection — the rule the preparer stores blocks by, too.
                if !qwen35::loads_quantized(cfg, prefix, key) {
                    continue;
                }
                consumed.insert(key.as_str());
                // The stacked MoE experts are quantized one expert slice at a time: a
                // `[mi, hidden]` gate and up from `gate_up_proj`, a `[hidden, mi]` down. A
                // per-expert snapshot's `experts.{e}.*_proj.weight` (sc-24438) is an ordinary
                // 2-D projection.
                let stacked = key
                    .rsplit_once(".mlp.experts.")
                    .map(|(_, name)| name)
                    .filter(|name| matches!(*name, "gate_up_proj" | "down_proj"));
                // A prepared tier's stored GGML blocks (sc-19375) are rebuilt slice by slice
                // exactly as stored, at the blocks each slice holds.
                if let Some((dtype, rows, cols)) = crate::primitives::quant::ggml_block_storage(
                    tensor.dtype == "U8",
                    &tensor.shape,
                ) {
                    let slices = match (stacked, cfg.moe) {
                        (Some("gate_up_proj"), Some(moe)) => 2 * moe.num_experts as usize,
                        (Some("down_proj"), Some(moe)) => moe.num_experts as usize,
                        (Some(_), _) => continue,
                        (None, _) => 1,
                    };
                    let slice = ggml_device_bytes(
                        (rows.checked_div(slices)? as u64).checked_mul(cols as u64)?,
                        dtype,
                    )?;
                    for _ in 0..slices {
                        add(slice)?;
                    }
                    continue;
                }
                match (stacked, cfg.moe, &tensor.shape[..]) {
                    (Some("gate_up_proj"), Some(moe), &[_, _, hidden]) => {
                        let slice =
                            format.projection_bytes(moe.moe_intermediate_size as usize, hidden)?;
                        for _ in 0..2 * moe.num_experts {
                            add(slice)?;
                        }
                    }
                    (Some("down_proj"), Some(moe), &[_, rows, cols]) => {
                        let slice = format.projection_bytes(rows, cols)?;
                        for _ in 0..moe.num_experts {
                            add(slice)?;
                        }
                    }
                    (Some(_), _, _) => {}
                    (None, _, _) => {
                        if let Some((rows, cols)) = tensor.dims2() {
                            add(format.projection_bytes(rows, cols)?)?;
                        }
                    }
                }
            }
            // The qwen3_5 head is quantized under every format; a tied one from the embedding.
            let head = if cfg.tie_word_embeddings {
                format!("{prefix}.embed_tokens.weight")
            } else {
                "lm_head.weight".to_string()
            };
            if let Some((rows, cols)) = tensors.get(&head).and_then(TensorHeader::dims2) {
                if !cfg.tie_word_embeddings {
                    consumed.insert("lm_head.weight");
                }
                add(format.projection_bytes(rows, cols)?)?;
            }
        }
        PricedDecoder::Prism | PricedDecoder::Refused => {}
    }
    Some((total, consumed))
}

/// The compute-dtype casts a CUDA load of a safetensors snapshot builds beside its resident
/// source (E6, sc-24140 review). The decoder takes every tensor it keeps dense through
/// `to_dtype(bf16)`, which shares a bf16 tensor's storage but copies any other float: an f16 or
/// f32 snapshot holds a second, 2-byte copy of each such tensor beside its source until the
/// decoder is built. So every float tensor not stored in bf16 is charged 2 bytes per element,
/// except those the loader consumes whole into its quantized copy (`consumed` — cast only
/// transiently, one projection at a time, inside the headroom) and an MLX-affine triple's
/// `.scales` / `.biases`, which the loader converts on the host. A tensor a multimodal tower casts
/// to f32 instead is charged its bf16 size; the rest of that cast stays in the headroom, as it
/// does for a bf16 snapshot. `None` on overflow.
pub(super) fn snapshot_cast_copy_bytes(
    tensors: &SnapshotTensors,
    consumed: &std::collections::HashSet<&str>,
) -> Option<u64> {
    let mut total = 0u64;
    for (key, tensor) in tensors {
        // `F16`, `F32`, `F64`, `F8_*`: every safetensors float tag but `BF16`.
        if !tensor.dtype.starts_with('F') || consumed.contains(key.as_str()) {
            continue;
        }
        let sidecar = [".scales", ".biases"].iter().any(|sidecar| {
            key.strip_suffix(sidecar)
                .is_some_and(|stem| tensors.contains_key(&format!("{stem}.weight")))
        });
        if sidecar {
            continue;
        }
        let elems = tensor
            .shape
            .iter()
            .try_fold(1u64, |n, &d| n.checked_mul(d as u64))?;
        total = total.checked_add(elems.checked_mul(2)?)?;
    }
    Some(total)
}

/// The Q8_0 copy the loader repacks an MLX-affine triple into (`Projection::load_mlx_affine_q8`),
/// priced only where it performs the repack: a GGML format of 8 bits over an 8-bit triple — the
/// geometry [`mlx_affine_q8_in_dim`] accepts for the format's group size. Any other triple (a
/// 4-bit pack, a 4-bit format, NVFP4, or no format at all) is the loader's typed refusal, which
/// builds no copy: `Some(0)`. `None` on overflow.
pub(super) fn affine_q8_repack_bytes(
    format: CopyFormat,
    weight: &TensorHeader,
    scales: &TensorHeader,
    biases: Option<&TensorHeader>,
) -> Option<u64> {
    let CopyFormat::Ggml(quant) = format else {
        return Some(0);
    };
    let geometry = (
        weight.dims2(),
        scales.dims2(),
        biases.and_then(TensorHeader::dims2),
    );
    let (Some(codes), Some(scales), Some(biases)) = geometry else {
        return Some(0);
    };
    let in_dim = (quant.bits() == 8)
        .then(|| {
            crate::primitives::quant::mlx_affine_q8_in_dim(
                weight.dtype == "U32",
                codes,
                scales,
                biases,
                quant.group_size(),
            )
        })
        .flatten();
    match in_dim {
        Some(in_dim) => ggml_device_bytes(
            (codes.0 as u64).checked_mul(in_dim as u64)?,
            candle_core::quantized::GgmlDType::Q8_0,
        ),
        None => Some(0),
    }
}

/// What a llama-family GGUF load holds on its device while it builds the decoder (E6, sc-24140).
///
/// [`GgufCheckpoint::open`] reads each tensor the loader maps ([`crate::gguf::remap_key`]) and
/// dequantizes it to a dense **f32** tensor on the loading device, keeping the whole map until the
/// provider is built — so the source is 4 bytes per element, several times the block-quantized
/// file. `CausalLm::from_weights_with` then builds the decoder beside that map: every tensor it
/// keeps dense is cast to the compute dtype (bf16 on CUDA — a second copy, 2 bytes per element;
/// on a host device the f32 compute dtype shares the source), and under a Q4 / Q8 request the
/// layer projections ([`quantizes_layer_tensor`](crate::models::llama::quantizes_layer_tensor))
/// are quantized to GGML ([`ggml_device_bytes`]). Read from the GGUF header with candle's own
/// reader, never tensor data. A Prism GGUF (packed blocks the loader wraps) is priced by
/// [`LoadWorkingSet::resident_payload`] instead.
pub(super) fn gguf_working_set(
    path: &Path,
    requested: Option<CopyFormat>,
    cuda: bool,
) -> CoreResult<LoadWorkingSet> {
    use crate::models::{llama, split_layer_key};
    let unreadable = |error: String| {
        CoreError::Load(format!(
            "checkpoint admission: gguf {}: {error}",
            path.display()
        ))
    };
    let mut file = std::fs::File::open(path).map_err(|e| unreadable(e.to_string()))?;
    let content = candle_core::quantized::gguf_file::Content::read_with_path(&mut file, path)
        .map_err(|e| unreadable(e.to_string()))?;
    // A GGUF load re-quantizes only to GGML; NVFP4 refuses a GGUF before admission.
    let quant = match requested {
        Some(CopyFormat::Ggml(quant)) => Some(quant),
        Some(CopyFormat::Nvfp4) | None => None,
    };
    let (mut source, mut cast, mut copy) = (0u64, 0u64, 0u64);
    let mut price = |key: &str, elems: u64| -> Option<()> {
        source = source.checked_add(elems.checked_mul(4)?)?;
        let projection = split_layer_key(key)
            .is_some_and(|(_, _, suffix)| llama::quantizes_layer_tensor(suffix));
        match quant.filter(|_| projection) {
            Some(quant) => copy = copy.checked_add(ggml_device_bytes(elems, quant.dtype)?)?,
            None if cuda => cast = cast.checked_add(elems.checked_mul(2)?)?,
            None => {}
        }
        Some(())
    };
    for (name, info) in &content.tensor_infos {
        if let Some(key) = crate::gguf::remap_key(name) {
            price(&key, info.shape.elem_count() as u64)
                .ok_or_else(|| CoreError::Load("load memory estimate overflow".into()))?;
        }
    }
    Ok(LoadWorkingSet {
        source,
        cast,
        copy,
        held: 0,
        host: HostRule::Gguf,
    })
}

#[cfg(test)]
thread_local! {
    /// A test's operational load budget on this thread ([`with_load_budget`]); the process-wide
    /// [`core_llm::AVAILABLE_MEMORY_OVERRIDE`] would leak into every concurrently loading test.
    static LOAD_BUDGET: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// Run `f` with load admission's operational budget at `budget` bytes on this thread.
#[cfg(test)]
pub(super) fn with_load_budget<R>(budget: u64, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<u64>);
    impl Drop for Restore {
        fn drop(&mut self) {
            LOAD_BUDGET.with(|b| b.set(self.0));
        }
    }
    let _restore = Restore(LOAD_BUDGET.with(|b| b.replace(Some(budget))));
    f()
}

/// The operational budget admission caps measured availability with
/// ([`core_llm::operational_memory_override`]) — the host domain's on a host device, the
/// device domain's on CUDA.
pub(super) fn load_memory_budget() -> CoreResult<Option<u64>> {
    #[cfg(test)]
    if let Some(budget) = LOAD_BUDGET.with(std::cell::Cell::get) {
        return Ok(Some(budget));
    }
    core_llm::operational_memory_override()
}

/// Resident bytes a companion MTP head adds to a load (epic sc-24432 E7, story sc-24444), from
/// its safetensors headers only: the payload (held on the load device while the predictor is
/// built) plus what the build allocates beside it — the Q8_0 repack of every MLX affine triple
/// ([`Projection::load_mlx_affine`](crate::primitives::projection::Projection::load_mlx_affine)),
/// a compute-dtype copy (priced at f32) of every other matrix, and the `1 + w` result of every
/// norm vector (f32). The head's per-request attention cache is priced by request admission on
/// the `mtp` route.
pub(super) fn companion_head_bytes(dir: &Path) -> CoreResult<u64> {
    let payload = core_llm::companion_head_payload_bytes(dir)?;
    let overflow = || CoreError::Load("companion MTP head admission overflow".into());
    let tensors = snapshot_tensor_headers(dir)?;
    let group_size = read_json(dir, "config.json").and_then(|config| {
        let quant = config
            .get("text_config")
            .and_then(|t| t.get("quantization"))
            .or_else(|| config.get("quantization"))?;
        quant.get("group_size")?.as_u64()
    });
    let mut built = 0u64;
    for (key, tensor) in &tensors {
        let stem = key.strip_suffix(".weight");
        let sidecar = [".scales", ".biases"].iter().any(|sidecar| {
            key.strip_suffix(sidecar)
                .is_some_and(|stem| tensors.contains_key(&format!("{stem}.weight")))
        });
        if sidecar {
            continue;
        }
        let scales = stem.and_then(|stem| tensors.get(&format!("{stem}.scales")));
        let bytes = match (tensor.dims2(), scales.and_then(TensorHeader::dims2)) {
            // An affine triple: its repacked `[rows, groups · group_size]`, or — with no group
            // size declared — the widest a U32 word packs (eight 4-bit codes).
            (Some((rows, words)), Some((_, groups))) if tensor.dtype == "U32" => {
                let cols = match group_size {
                    Some(group) => (groups as u64).checked_mul(group),
                    None => (words as u64).checked_mul(8),
                }
                .ok_or_else(overflow)?;
                crate::primitives::projection::ggml_device_bytes(
                    (rows as u64).checked_mul(cols).ok_or_else(overflow)?,
                    candle_core::quantized::GgmlDType::Q8_0,
                )
            }
            _ => tensor
                .shape
                .iter()
                .try_fold(4u64, |n, &d| n.checked_mul(d as u64)),
        }
        .ok_or_else(overflow)?;
        built = built.checked_add(bytes).ok_or_else(overflow)?;
    }
    payload.checked_add(built).ok_or_else(overflow)
}

/// Admit a companion MTP head's resident bytes on top of what the load already admitted — the
/// target and any admitted draft (E7) — in the host domain `(admitted, available)` and, on CUDA,
/// the device domain too ([`core_llm::admit_companion_head`], the rule MLX admits by), returning
/// the head and its bytes. `Err` is the named load fallback (E2): the head is unreadable, or the
/// admitted bytes plus the head's exceed a budget.
pub(super) fn admit_companion_head(
    head: &Path,
    host: (u64, u64),
    device: Option<(u64, u64)>,
) -> Result<(&Path, u64), String> {
    core_llm::admit_companion_head(
        head,
        companion_head_bytes(head),
        std::iter::once(("host", host.0, host.1))
            .chain(device.map(|(admitted, available)| ("device", admitted, available))),
    )
}

/// The process-wide override caps the execution-memory domain. On a discrete CUDA device, host
/// source staging has its own measured capacity and must not be capped by a VRAM snapshot.
pub(super) fn host_load_budget(cuda: bool, budget: Option<u64>) -> Option<u64> {
    (!cuda).then_some(budget).flatten()
}

/// Combine CUDA driver-free bytes with idle bytes retained by cudarc's asynchronous allocator.
///
/// `cuMemGetInfo` excludes memory reserved by the current CUDA memory pool, even when part of that
/// reservation is unused and immediately reusable by this process. Admission must include that
/// idle reservation without counting live pool allocations a second time. Invalid or internally
/// inconsistent snapshots fail closed.
#[cfg(any(feature = "cuda", test))]
pub(super) fn cuda_usable_memory_bytes(
    driver_free: u64,
    device_total: u64,
    pool_usage: Option<(u64, u64)>,
) -> Option<u64> {
    if driver_free > device_total {
        return None;
    }
    let Some((reserved, used)) = pool_usage else {
        return Some(driver_free);
    };
    if used > reserved || reserved > device_total {
        return None;
    }
    let free_plus_reserved = driver_free.checked_add(reserved)?;
    if free_plus_reserved > device_total {
        return None;
    }
    free_plus_reserved.checked_sub(used)
}

#[cfg(feature = "cuda")]
pub(super) fn cuda_pool_usage(
    context: &candle_core::cuda_backend::cudarc::driver::CudaContext,
) -> Option<(u64, u64)> {
    use candle_core::cuda_backend::cudarc::driver::{result, sys};

    context.bind_to_thread().ok()?;
    // SAFETY: the live context is bound to this thread, `cu_device` remains owned by it,
    // `get_mem_pool` returns that device's live current pool, and both output pointers have the u64
    // type required by these two CUDA attributes.
    unsafe {
        let pool = result::device::get_mem_pool(context.cu_device()).ok()?;
        let mut reserved = 0_u64;
        result::mem_pool::get_attribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            (&mut reserved as *mut u64).cast(),
        )
        .ok()?;
        let mut used = 0_u64;
        result::mem_pool::get_attribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            (&mut used as *mut u64).cast(),
        )
        .ok()?;
        Some((reserved, used))
    }
}

pub(super) fn request_available_memory(device: &Device) -> CoreResult<u64> {
    let capacity = if device.is_cuda() {
        #[cfg(feature = "cuda")]
        {
            // Candle frees asynchronous allocations on this device's stream. Complete those frees
            // before sampling the pool so admission observes the reusable post-load/request state.
            device.synchronize().map_err(|e| to_core(e.into()))?;
            device.as_cuda_device().ok().and_then(|d| {
                let stream = d.cuda_stream();
                let context = stream.context();
                let (free, total) = context.mem_get_info().ok()?;
                let pool_usage = if context.has_async_alloc() {
                    Some(cuda_pool_usage(context)?)
                } else {
                    None
                };
                cuda_usable_memory_bytes(free as u64, total as u64, pool_usage)
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    } else {
        core_llm::available_host_memory_bytes()
    };
    core_llm::effective_memory_budget(capacity, load_memory_budget()?)
}
