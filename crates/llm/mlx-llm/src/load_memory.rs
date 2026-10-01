//! Header-only upper bounds for the actual MLX checkpoint load paths.
use crate::primitives::projection::QuantSpec;
use core_llm::{Error, LoadSpec, Result};
use serde_json::Value;
use std::{fs::File, io::Read, path::Path};

fn overflow() -> Error {
    Error::Load("MLX load memory estimate overflow".into())
}

/// Price source buffers, conversion outputs and staging without evaluating any MLX array.
///
/// Safetensors, by which guarded real-weight probe backs the load's cell
/// ([`LoadCell`]: `model_type` × conversion × stored dtype; sc-24446):
///
/// * a cell probed under the provider's materialize-at-load order ([`MATERIALIZED_VERIFIED`]):
///   the group-by-group peak, [`materialized_bound`];
/// * a cell probed under the earlier verify-everything-then-derive order ([`EARLIER_VERIFIED`]):
///   the stored payload once plus every derived allocation ([`derived_bytes`]). The
///   materialize-at-load order holds no more than that beyond its pacing slack (each group's
///   sources are a slice of the payload, its outputs and transients a slice of the derived set);
/// * anything else: the conservative two-copy bound, raised to the derived total where that is
///   larger, so an unprobed load is never charged less than its derived allocations.
///
/// Every safetensors bound adds the load's host heap ([`host_bytes`]).
pub(crate) fn required_bytes(spec: &LoadSpec) -> Result<u64> {
    let path = Path::new(&spec.source);
    if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
        let language = gguf_bytes(path, false)?;
        let projector = spec
            .projector_source
            .as_ref()
            .map(|p| gguf_bytes(Path::new(p), true))
            .transpose()?
            .unwrap_or(0);
        return language.checked_add(projector).ok_or_else(overflow);
    }
    let quant = spec.quantize.map(crate::provider::quant_spec).transpose()?;
    SnapshotFacts::read(path)?.required(quant)
}

/// Everything [`required_bytes`] reads from a safetensors snapshot — header-only: its
/// `config.json`, every shard's header, the shards' total size and the `tokenizer.json` size. The
/// tests rebuild it from a committed manifest of a pinned snapshot (no weights), so the recorded
/// probes are recomputed from code on every run.
#[derive(Clone, Debug)]
pub(crate) struct SnapshotFacts {
    pub(crate) config: Value,
    pub(crate) headers: Vec<Value>,
    /// Total `*.safetensors` bytes ([`core_llm::checkpoint_payload_bytes`]).
    pub(crate) payload: u64,
    /// `tokenizer.json` bytes (0 when absent).
    pub(crate) tokenizer_bytes: u64,
}

impl SnapshotFacts {
    /// Read the facts of the snapshot directory `dir`.
    pub(crate) fn read(dir: &Path) -> Result<Self> {
        let config: Value = serde_json::from_reader(
            File::open(dir.join("config.json")).map_err(|e| Error::Load(e.to_string()))?,
        )
        .map_err(|e| Error::Load(e.to_string()))?;
        let tokenizer_bytes = match std::fs::metadata(dir.join("tokenizer.json")) {
            Ok(meta) => meta.len(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
            Err(e) => return Err(Error::Load(e.to_string())),
        };
        Ok(Self {
            config,
            headers: safetensors_headers(dir)?,
            payload: core_llm::checkpoint_payload_bytes(dir)?,
            tokenizer_bytes,
        })
    }

    /// The bound a load at `quant` is charged ([`required_bytes`]).
    pub(crate) fn required(&self, quant: Option<QuantSpec>) -> Result<u64> {
        let cell = LoadCell::of(&self.config, &self.headers, quant);
        let arrays = if cell.is_in(MATERIALIZED_VERIFIED) {
            materialized_bound(&self.headers, quant)?
        } else {
            let derived = self.earlier_arrays(quant)?;
            if cell.is_in(EARLIER_VERIFIED) {
                derived
            } else {
                derived.max(self.payload.checked_mul(2).ok_or_else(overflow)?)
            }
        };
        arrays.checked_add(self.host()?).ok_or_else(overflow)
    }

    /// The earlier order's peak: the payload once plus every derived allocation.
    pub(crate) fn earlier_arrays(&self, quant: Option<QuantSpec>) -> Result<u64> {
        self.payload
            .checked_add(derived_bytes(&self.headers, quant)?)
            .ok_or_else(overflow)
    }

    /// The load's host heap ([`host_bytes`]).
    pub(crate) fn host(&self) -> Result<u64> {
        host_bytes(self.tokenizer_bytes)
    }
}

/// How a checkpoint's weights reach the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Conversion {
    /// Dense floating weights, used as stored (cast to BF16 where wider).
    Dense,
    /// Dense weights quantized to Q4 at load.
    LoadQ4,
    /// Dense weights quantized to Q8 at load.
    LoadQ8,
    /// A snapshot that stores MLX affine-quantized projections.
    Stored,
}

/// The cell a load's bound is verified per: the snapshot's top-level `model_type` (exactly — a
/// family the loader dispatches alike, such as Mistral or dense Qwen2 through the Llama decoder,
/// is its own cell), the conversion, and the one floating dtype every decoder matrix is stored
/// in (`"mixed"` when they differ, which no table lists).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cell {
    pub(crate) model_type: &'static str,
    pub(crate) conversion: Conversion,
    pub(crate) dtype: &'static str,
}

/// A load's cell, as read from its snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LoadCell {
    pub(crate) model_type: String,
    pub(crate) conversion: Option<Conversion>,
    pub(crate) dtype: String,
}

impl LoadCell {
    pub(crate) fn of(config: &Value, headers: &[Value], quant: Option<QuantSpec>) -> Self {
        let model_type = config
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let names = || {
            headers
                .iter()
                .filter_map(Value::as_object)
                .flat_map(|h| h.iter())
        };
        let conversion = match quant.map(|q| q.bits) {
            Some(4) => Some(Conversion::LoadQ4),
            Some(8) => Some(Conversion::LoadQ8),
            Some(_) => None,
            None if names().any(|(name, _)| name.ends_with(".scales")) => Some(Conversion::Stored),
            None => Some(Conversion::Dense),
        };
        let mut dtypes: Vec<&str> = names()
            .filter(|(name, info)| {
                !media_tensor(name)
                    && !name.ends_with(".scales")
                    && !name.ends_with(".biases")
                    && info["shape"].as_array().is_some_and(|s| s.len() >= 2)
            })
            .filter_map(|(_, info)| info["dtype"].as_str())
            .filter(|d| matches!(*d, "F16" | "BF16" | "F32" | "F64"))
            .collect();
        dtypes.sort_unstable();
        dtypes.dedup();
        let dtype = match dtypes.as_slice() {
            [one] => (*one).to_string(),
            _ => "mixed".to_string(),
        };
        Self {
            model_type,
            conversion,
            dtype,
        }
    }

    pub(crate) fn is_in(&self, table: &[Cell]) -> bool {
        table.iter().any(|c| {
            Some(c.conversion) == self.conversion
                && c.model_type == self.model_type
                && c.dtype == self.dtype
        })
    }
}

/// Cells whose earlier-order bound ([`SnapshotFacts::earlier_arrays`]) a guarded real-weight
/// probe covers with margin — each backed by a `MEASURED` row the tests recompute from the
/// snapshot's committed manifest. Only these: Gemma 4 unified BF16 measured above its bound under
/// the uniform method, and every load-time Q4/Q8 probe was sampled every 100 ms across a
/// `clear_cache`, which can hide up to the load's derived bytes, so none clears the margin
/// floor; they keep the two-copy floor until the exact-peak re-probes
/// (`docs/reference/qwen38/native-memory-admission.md`).
pub(crate) const EARLIER_VERIFIED: &[Cell] = &[
    Cell {
        model_type: "llama",
        conversion: Conversion::Dense,
        dtype: "BF16",
    },
    Cell {
        model_type: "qwen3",
        conversion: Conversion::Dense,
        dtype: "BF16",
    },
    Cell {
        model_type: "gemma2",
        conversion: Conversion::Dense,
        dtype: "BF16",
    },
];

/// Cells whose [`materialized_bound`] an exact-peak (kernel `phys_footprint` maximum) probe of
/// the provider's materialize-at-load order covers with margin. Empty until those probes run.
pub(crate) const MATERIALIZED_VERIFIED: &[Cell] = &[];

/// Host heap per `tokenizer.json` byte: the parsed vocabulary, merges and added-token tables.
/// The sc-24446 probes measured 12–17× (Llama 3.2, Qwen3, Gemma 2); 24× keeps headroom.
const TOKENIZER_HEAP_PER_BYTE: u64 = 24;

/// Host heap a load adds independent of the tokenizer: the chat template, the lazy graph's node
/// allocations, and the Metal pipeline states its load-boundary checksum and first forward
/// compile (measured ≤ 30 MiB beside the tokenizer).
const LOAD_HOST_BYTES: u64 = 64 * 1024 * 1024;

/// The `phys_footprint` the Metal driver takes back when a process that has idled evaluates
/// again (sc-24446): it returns its command-queue and pipeline resources asynchronously (measured
/// 0.1–1.25 s after the last work) and the next evaluation re-acquires them — 112 MiB for one tiny
/// op and 129 MiB for the whole load of a 100 KB snapshot (exact kernel peak), 137–170 MiB for a
/// fixture's one-token request, on an idle Apple M5 Max. Charged to every load and every request
/// with ~1.5x headroom over the largest.
pub(crate) const MLX_DRIVER_WAKE_BYTES: u64 = 256 * 1024 * 1024;

/// Host (non-MLX) memory a safetensors load adds beside its arrays (sc-24446): the tokenizer
/// it parses (`tokenizer_bytes` of `tokenizer.json`), a fixed allowance for everything else
/// the load builds on the heap, and the driver's wake ([`MLX_DRIVER_WAKE_BYTES`]). The Metal
/// device and MLX's kernel library are process-wide one-time costs, not a load's.
fn host_bytes(tokenizer_bytes: u64) -> Result<u64> {
    tokenizer_bytes
        .checked_mul(TOKENIZER_HEAP_PER_BYTE)
        .and_then(|v| v.checked_add(LOAD_HOST_BYTES))
        .and_then(|v| v.checked_add(MLX_DRIVER_WAKE_BYTES))
        .ok_or_else(overflow)
}

/// Every `*.safetensors` header in `dir`, read header-only.
pub(crate) fn safetensors_headers(dir: &Path) -> Result<Vec<Value>> {
    let mut headers = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| Error::Load(e.to_string()))? {
        let path = entry.map_err(|e| Error::Load(e.to_string()))?.path();
        if path.extension().and_then(|s| s.to_str()) != Some("safetensors") {
            continue;
        }
        let mut file = File::open(&path).map_err(|e| Error::Load(e.to_string()))?;
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)
            .map_err(|e| Error::Load(e.to_string()))?;
        let len = u64::from_le_bytes(prefix);
        if len > 64 * 1024 * 1024 {
            return Err(Error::Load(
                "safetensors admission header exceeds 64 MiB".into(),
            ));
        }
        let mut header = vec![0u8; len as usize];
        file.read_exact(&mut header)
            .map_err(|e| Error::Load(e.to_string()))?;
        headers.push(serde_json::from_slice(&header).map_err(|e| Error::Load(e.to_string()))?);
    }
    Ok(headers)
}

/// The conversion intermediates every `*.safetensors` header in `dir` implies with no load-time
/// quantization ([`derived_bytes`]), read header-only.
fn safetensors_dir_extra(dir: &Path) -> Result<u64> {
    derived_bytes(&safetensors_headers(dir)?, None)
}

/// Resident bytes a companion MTP head adds to a load (epic sc-24432 E7, story sc-24444): its
/// safetensors payload (packed words, affine scales/biases and norm vectors stay resident in the
/// predictor) plus the conversion intermediates the header implies — the BF16 cast of any wider
/// float and the `1 + w` norm results. Header-only; never constructs a tensor. The head's
/// per-request attention cache is priced by request admission on the `mtp` route.
pub(crate) fn companion_head_bytes(dir: &Path) -> Result<u64> {
    core_llm::companion_head_payload_bytes(dir)?
        .checked_add(safetensors_dir_extra(dir)?)
        .ok_or_else(overflow)
}

/// MLX rounds every Metal buffer above one page up to a whole 16 KiB page; each allocation below
/// is charged this rounding once.
const ALLOCATION_ROUNDING: u64 = 16 * 1024;

/// Bytes the MLX safetensors loaders allocate **beyond** the stored payload, from the headers
/// alone (sc-24446) — the peak of the earlier verify-everything-then-derive order, which the
/// provider's materialize-at-load order ([`materialized_bound`]) never exceeds. Every term is
/// additive: a derived array either stays resident in the model
/// or, once consumed, returns its buffer to MLX's freed-buffer cache, which is reused only for an
/// allocation of (page-rounded) equal size and released only by `clear_cache` (the first decode
/// step) or allocator pressure — so a consumed transient still occupies memory through the load
/// and the first forward. The payload itself is read into owned Metal buffers (MLX `Load` is a
/// `pread` into `allocator::malloc`, not a mapping) and every accessed tensor is resident after
/// the constructors' `verify_accessed_gpu_view`, before any derived array is evaluated.
fn derived_bytes(headers: &[Value], quant: Option<QuantSpec>) -> Result<u64> {
    headers.iter().try_fold(0u64, |total, header| {
        total
            .checked_add(header_derived_bytes(header, quant)?)
            .ok_or_else(overflow)
    })
}

/// `n·bits/8` packed `uint32` words plus BF16 scales and biases per `group_size` elements: the
/// three outputs of MLX `quantize` on a BF16 input.
fn quantized_bytes(elements: u64, quant: QuantSpec) -> Result<u64> {
    let bits = u64::try_from(quant.bits).map_err(|_| overflow())?;
    let group = u64::try_from(quant.group_size).map_err(|_| overflow())?;
    let packed = elements
        .checked_mul(bits)
        .map(|v| v / 8)
        .ok_or_else(overflow)?;
    let affine = (elements / group).checked_mul(4).ok_or_else(overflow)?;
    packed.checked_add(affine).ok_or_else(overflow)
}

/// The expert index of a per-expert tensor (`…experts.{e}.…`), if this is one.
fn expert_index(name: &str) -> Option<u64> {
    let (_, rest) = name.split_once("experts.")?;
    rest.split('.').next()?.parse().ok()
}

fn header_derived_bytes(header: &Value, quant: Option<QuantSpec>) -> Result<u64> {
    header_tensors(header)?.iter().try_fold(0u64, |total, t| {
        total
            .checked_add(tensor_derived(t, quant)?)
            .ok_or_else(overflow)
    })
}

/// The arrays the loaders derive from one stored tensor, each charged a page of rounding — the
/// additive terms of [`derived_bytes`].
fn tensor_derived(t: &HeaderTensor<'_>, quant: Option<QuantSpec>) -> Result<u64> {
    let mut extra = 0u64;
    let mut add = |bytes: u64| -> Result<()> {
        extra = extra
            .checked_add(bytes)
            .and_then(|v| v.checked_add(ALLOCATION_ROUNDING))
            .ok_or_else(overflow)?;
        Ok(())
    };
    let (name, dtype, shape, elements, stored) = (t.name, t.dtype, &t.shape, t.elements, t.stored);
    let bf16 = elements.checked_mul(2).ok_or_else(overflow)?;
    let float = t.float();
    let vision = name.starts_with("model.visual.") || name.starts_with("vision_tower.");
    // The stored buffer itself (payload) is page-rounded too.
    add(0)?;
    // The load-boundary checksum widens a byte view to u32 when the size is not word-aligned.
    if stored % 4 != 0 {
        add(stored.checked_mul(4).ok_or_else(overflow)?)?;
    }
    // Language tensors are cast to BF16. MLX astype returns the original Array for an
    // unchanged dtype; vision clones retain their original dtype and U32 packed words remain
    // compact.
    if !vision && float && dtype != "BF16" {
        add(bf16)?;
    }
    // Vector norms (`1 + w`), exp(A_log) and sign transforms can retain both source and
    // result. Four bytes per element covers two additional BF16 intermediates.
    if !vision && shape.len() <= 1 {
        add(elements.checked_mul(4).ok_or_else(overflow)?)?;
    }
    // A channels-last patch kernel may require a contiguous transpose before reshape.
    if vision && name.ends_with("patch_embed.proj.weight") {
        add(elements.checked_mul(4).ok_or_else(overflow)?)?;
    }
    // Gathered splits are copies: Gemma 4's `[n, 2, hidden]` position table is split into its
    // row and column tables. (Qwen3.6's fused expert `gate_up_proj` is no longer split: the
    // MoE block runs it as one fused bank, sc-24446.)
    if name.ends_with(".pos_embedding") && shape.len() == 3 {
        add(bf16)?;
    }
    // Load-time affine quantization: MLX `quantize` writes new packed words, scales and
    // biases (one GPU kernel, no float intermediates) for every projection the decoders
    // quantize. Embeddings and the LM head stay dense; every other floating matrix whose
    // input width divides the group is charged (a dense router or 1-row gate is
    // over-charged, never missed).
    let quantized = match quant {
        Some(q)
            if !vision
                && float
                && shape.len() >= 2
                && !name.contains("embed_tokens")
                && !name.ends_with("lm_head.weight")
                && !name.ends_with(".scales")
                && !name.ends_with(".biases")
                && u64::try_from(q.group_size)
                    .ok()
                    .filter(|&g| g > 0)
                    .is_some_and(|g| shape.last().is_some_and(|last| last % g == 0)) =>
        {
            Some(quantized_bytes(elements, q)?)
        }
        _ => None,
    };
    if let Some(bytes) = quantized {
        add(bytes)?;
    }
    // Per-expert tensors are stacked into one `[experts, …]` array per projection
    // (`SwitchLinear::stack`, sc-24440): a copy of each expert's loaded form — quantized,
    // BF16, or the stored packed part.
    if expert_index(name).is_some() {
        add(match quantized {
            Some(bytes) => bytes,
            None if float && !name.ends_with(".scales") && !name.ends_with(".biases") => bf16,
            None => stored,
        })?;
    }
    Ok(extra)
}

/// One tensor of a safetensors header.
struct HeaderTensor<'a> {
    name: &'a str,
    dtype: &'a str,
    shape: Vec<u64>,
    elements: u64,
    /// Stored bytes.
    stored: u64,
}

impl HeaderTensor<'_> {
    fn float(&self) -> bool {
        matches!(self.dtype, "F16" | "BF16" | "F32" | "F64")
    }

    /// A stored affine-quantized part (`.scales` / `.biases`), read as stored.
    fn quant_part(&self) -> bool {
        self.name.ends_with(".scales") || self.name.ends_with(".biases")
    }
}

/// Every tensor of one safetensors header (the `__metadata__` entry skipped).
fn header_tensors(header: &Value) -> Result<Vec<HeaderTensor<'_>>> {
    let mut out = Vec::new();
    for (name, info) in header
        .as_object()
        .ok_or_else(|| Error::Load("safetensors header is not an object".into()))?
    {
        if name == "__metadata__" {
            continue;
        }
        let shape = info["shape"]
            .as_array()
            .ok_or_else(|| Error::Load("safetensors tensor lacks shape".into()))?
            .iter()
            .map(|v| v.as_u64().ok_or_else(overflow))
            .collect::<Result<Vec<u64>>>()?;
        let elements = shape
            .iter()
            .try_fold(1u64, |n, &v| n.checked_mul(v).ok_or_else(overflow))?;
        let dtype = info["dtype"]
            .as_str()
            .ok_or_else(|| Error::Load("safetensors tensor lacks dtype".into()))?;
        let width = match dtype {
            "F64" | "I64" | "U64" => 8,
            "F32" | "I32" | "U32" => 4,
            "F16" | "BF16" | "I16" | "U16" => 2,
            _ => 1,
        };
        out.push(HeaderTensor {
            name,
            dtype,
            shape,
            elements,
            stored: elements.checked_mul(width).ok_or_else(overflow)?,
        });
    }
    Ok(out)
}

/// A tensor no decoder loader reads: a vision tower, an image / audio projector or embedder. The
/// provider loads those through their own constructors, which keep every source resident beside
/// any derived array — so they are priced the old, additive way.
fn media_tensor(name: &str) -> bool {
    ["visual", "vision", "audio", "multi_modal_projector"]
        .iter()
        .any(|m| name.contains(m))
}

/// Dense matrices the decoders read with load-time quantization requested but never quantize:
/// the MoE router and shared-expert gate, and Qwen3.5's per-head decay / delta-strength inputs.
/// (Embeddings and the LM head are excluded by name below; conv kernels by their width.)
const NEVER_QUANTIZED: [&str; 4] = [
    ".mlp.gate.weight",
    "shared_expert_gate.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
];

/// The packed words + scales + biases a decoder tensor quantizes to at load, if it does: every
/// floating language matrix whose input width divides the group, except embeddings, the LM head,
/// stored quantized parts and [`NEVER_QUANTIZED`].
fn quantized_at_load(t: &HeaderTensor<'_>, quant: Option<QuantSpec>) -> Result<Option<u64>> {
    let Some(q) = quant else { return Ok(None) };
    let group = u64::try_from(q.group_size).map_err(|_| overflow())?;
    let quantizes = t.float()
        && t.shape.len() >= 2
        && !t.quant_part()
        && !t.name.contains("embed_tokens")
        && !t.name.ends_with("lm_head.weight")
        && !NEVER_QUANTIZED.iter().any(|s| t.name.ends_with(s))
        && group > 0
        && t.shape.last().is_some_and(|last| last % group == 0);
    quantizes
        .then(|| quantized_bytes(t.elements, q))
        .transpose()
}

/// The materialization group a decoder tensor belongs to: its layer (`…layers.{i}.`, decoder and
/// MTP layers alike), or `""` for everything outside a layer stack (embeddings, final norm, LM
/// head, the MTP head's fusion / norms). [`crate::primitives::Weights::materialize_groups`]
/// never evaluates two layers' sources together, and evaluates the non-layer arrays in batches
/// that each fall inside the `""` group.
fn materialization_group(name: &str) -> &str {
    let mut search = 0;
    while let Some(at) = name[search..].find("layers.") {
        let start = search + at + "layers.".len();
        let digits = name[start..].bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && name[start + digits..].starts_with('.') {
            return &name[..start + digits + 1];
        }
        search = start;
    }
    ""
}

/// Arrays the safetensors load holds at its peak under the provider's materialize-at-load order
/// (sc-24446): `resident + max(window)`.
///
/// [`Weights::materialize_groups`](crate::primitives::Weights::materialize_groups) reads one
/// group's sources (a decoder layer, or the arrays outside the stack), verifies them, evaluates the
/// group's conversions, drops the consumed sources and clears MLX's buffer cache before the next
/// group. While group `g` converts, memory holds the arrays already built, `g`'s sources and
/// transients and `g`'s outputs — at most every resident array plus `g`'s window:
///
/// * **resident** — each decoder tensor in its loaded form: quantized (`n·bits/8` words plus
///   BF16 scales and biases) when quantized at load, a BF16 cast of a wider float, a stacked copy
///   of a per-expert tensor, `max(stored, BF16)` for a vector (`1 + w` norms), else the stored
///   buffer itself; media tensors (vision / audio, loaded by their own constructors and never
///   released) at the stored payload plus every derived array ([`header_derived_bytes`]).
/// * **window** of a group — its consumed sources (every stored buffer replaced by a different
///   array), and its transients: a BF16 cast ahead of a quantize, a per-expert array ahead of its
///   stack, the norm's intermediates, the load-boundary checksum's word widening.
///
/// Plus what the pacing lets stay outstanding (`Weights::materialize_groups`): the Metal driver
/// returns a released buffer asynchronously, so one group's consumed sources (the largest such
/// group's) may still count while the next group converts, and the pacing slack.
///
/// Each buffer is charged one 16 KiB page of rounding (three for a quantized triple).
fn materialized_bound(headers: &[Value], quant: Option<QuantSpec>) -> Result<u64> {
    let r = ALLOCATION_ROUNDING;
    let sum = |terms: &[u64]| -> Result<u64> {
        terms
            .iter()
            .try_fold(0u64, |a, &b| a.checked_add(b).ok_or_else(overflow))
    };
    let mut resident = 0u64;
    let mut windows: std::collections::HashMap<&str, u64> = std::collections::HashMap::new();
    let mut sources: std::collections::HashMap<&str, u64> = std::collections::HashMap::new();
    for header in headers {
        for t in header_tensors(header)? {
            if media_tensor(t.name) {
                let derived = tensor_derived(&t, None)?;
                resident = sum(&[resident, t.stored, r, derived])?;
                continue;
            }
            let bf16 = t.elements.checked_mul(2).ok_or_else(overflow)?;
            let cast = t.float() && t.dtype != "BF16" && !t.quant_part();
            let per_expert = expert_index(t.name).is_some();
            let vector = t.shape.len() <= 1;
            let quantized = quantized_at_load(&t, quant)?;
            let (kept, consumed) = match quantized {
                Some(q) => (sum(&[q, 3 * r])?, true),
                None if per_expert => (sum(&[if cast { bf16 } else { t.stored }, r])?, true),
                None if vector => (sum(&[t.stored.max(bf16), r])?, true),
                None if cast => (sum(&[bf16, r])?, true),
                None => (sum(&[t.stored, r])?, false),
            };
            let mut window = Vec::new();
            if consumed {
                window.extend([t.stored, r]);
            }
            if quantized.is_some() && cast {
                window.extend([bf16, r]);
            }
            if per_expert {
                // Each expert's loaded form exists before the bank is stacked from it.
                match quantized {
                    Some(q) => window.extend([q, 3 * r]),
                    None if cast => window.extend([bf16, r]),
                    None => {}
                }
            }
            if vector {
                window.extend([t.elements.checked_mul(4).ok_or_else(overflow)?, r]);
            }
            if t.stored % 4 != 0 {
                window.extend([t.stored.checked_mul(4).ok_or_else(overflow)?, r]);
            }
            resident = sum(&[resident, kept])?;
            let group = windows.entry(materialization_group(t.name)).or_default();
            *group = sum(&[*group, sum(&window)?])?;
            if consumed {
                let released = sources.entry(materialization_group(t.name)).or_default();
                *released = sum(&[*released, t.stored, r])?;
            }
        }
    }
    let window = windows.values().copied().max().unwrap_or(0);
    // Pacing lets one released group's sources still be on their way back from the driver.
    let outstanding = sources.values().copied().max().unwrap_or(0);
    sum(&[
        resident,
        window,
        outstanding,
        crate::primitives::weights::MATERIALIZE_PACE_SLACK_BYTES,
    ])
}

fn gguf_bytes(path: &Path, projector: bool) -> Result<u64> {
    let file = crate::gguf::GgufFile::open(path).map_err(|e| Error::Load(e.to_string()))?;
    let mut retained = 0u64;
    let mut staging = 0u64;
    for tensor in &file.tensors {
        let elements = tensor
            .shape
            .iter()
            .try_fold(1u64, |n, &v| n.checked_mul(v as u64).ok_or_else(overflow))?;
        let (keep, temporary) = gguf_tensor_bytes(
            elements,
            tensor.shape.last().copied().unwrap_or(0) as u64,
            tensor.ggml_type,
            projector,
        )?;
        retained = retained.checked_add(keep).ok_or_else(overflow)?;
        staging = staging.max(temporary);
    }
    // GGUF is memory mapped. Include every source byte even though pages are reclaimable, all
    // retained outputs, and the largest tensor's simultaneous conversion/reorder host buffers.
    core_llm::checkpoint_payload_bytes(path)?
        .checked_add(retained)
        .and_then(|v| v.checked_add(staging))
        .ok_or_else(overflow)
}

fn gguf_tensor_bytes(n: u64, width: u64, kind: u32, projector: bool) -> Result<(u64, u64)> {
    if !projector && matches!(kind, 142 | 143) {
        // U32 affine words: n/4 bytes. Both F32 and F16 scale+bias arrays may coexist until
        // lazy conversion completes: n/16+n/32. Explicit signs use four bytes per column.
        let signs = width.checked_mul(4).ok_or_else(overflow)?;
        let kept = n
            .checked_mul(11)
            .and_then(|v| v.checked_div(32))
            .and_then(|v| v.checked_add(signs))
            .ok_or_else(overflow)?;
        let temporary = n
            .checked_mul(5)
            .and_then(|v| v.checked_div(16))
            .and_then(|v| v.checked_add(signs))
            .ok_or_else(overflow)?;
        Ok((kept, temporary))
    } else {
        // Dense language: F32 source + BF16 cast + norm result. Projector remains F32.
        // Reordering dense language rows also holds an old and new host Vec<f32>.
        let multiplier = if projector { 4 } else { 8 };
        let bytes = n.checked_mul(multiplier).ok_or_else(overflow)?;
        Ok((bytes, bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// MLX's Metal page (`vm_page_size` on Apple silicon): every buffer above it is rounded up.
    const R: u64 = 16 * 1024;
    /// The fixed host allowance of a load with no `tokenizer.json`: the heap and the driver's
    /// wake.
    const H: u64 = 64 * 1024 * 1024 + MLX_DRIVER_WAKE_BYTES;

    /// A load's host heap: 24 bytes per `tokenizer.json` byte plus the fixed allowance.
    #[test]
    fn a_load_is_charged_its_tokenizer_heap_and_a_fixed_host_allowance() {
        let (dir, payload) = snapshot(
            json!({"model_type": "qwen3"}),
            &[("model.norm.weight", "BF16", &[64])],
        );
        let bare = required(&dir, None);
        assert_eq!(bare, payload + 2 * R + 64 * 4 + H);
        std::fs::write(dir.path().join("tokenizer.json"), vec![b' '; 1000]).unwrap();
        assert_eq!(required(&dir, None), bare + 24 * 1000);
    }

    /// Every stored tensor and every derived array is charged one whole 16 KiB page of rounding.
    #[test]
    fn each_buffer_is_charged_a_metal_page_of_rounding() {
        let one = json!({"m.weight": {"dtype": "BF16", "shape": [64, 64]}});
        assert_eq!(derived_bytes(&[one], None).unwrap(), R);
    }

    /// A one-file snapshot with zeroed payloads for `tensors` (name, dtype, shape) under
    /// `config`; returns the directory and its payload bytes.
    fn snapshot(config: Value, tensors: &[(&str, &str, &[u64])]) -> (tempfile::TempDir, u64) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
        let mut header = serde_json::Map::new();
        let mut offset = 0u64;
        for (name, dtype, shape) in tensors {
            let width = match *dtype {
                "F32" | "U32" => 4,
                "F64" => 8,
                _ => 2,
            };
            let end = offset + width * shape.iter().product::<u64>();
            header.insert(
                name.to_string(),
                json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}),
            );
            offset = end;
        }
        let header = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(&header);
        bytes.resize(bytes.len() + offset as usize, 0);
        std::fs::write(dir.path().join("model.safetensors"), &bytes).unwrap();
        (dir, bytes.len() as u64)
    }

    fn required(dir: &tempfile::TempDir, quantize: Option<core_llm::Quantize>) -> u64 {
        let mut spec = LoadSpec::dense(dir.path().display().to_string());
        spec.quantize = quantize;
        required_bytes(&spec).unwrap()
    }

    type Tensors = Vec<(String, &'static str, Vec<u64>)>;

    /// A dense decoder slice: embedding, one projection, one norm, untied head.
    fn dense(prefix: &str) -> Tensors {
        vec![
            (
                format!("{prefix}.embed_tokens.weight"),
                "BF16",
                vec![16, 64],
            ),
            (
                format!("{prefix}.layers.0.mlp.up_proj.weight"),
                "BF16",
                vec![1024, 1024],
            ),
            (format!("{prefix}.norm.weight"), "BF16", vec![64]),
            ("lm_head.weight".into(), "BF16", vec![16, 64]),
        ]
    }

    fn refs(t: &Tensors) -> Vec<(&str, &str, &[u64])> {
        t.iter()
            .map(|(n, d, s)| (n.as_str(), *d, s.as_slice()))
            .collect()
    }

    /// The probed cells, unquantized: the payload once plus each buffer's page rounding and the
    /// norm's two BF16 intermediates — no second copy of any BF16 matrix. Every other cell keeps
    /// the two-copy floor, including a family the loader dispatches alike (Mistral, dense Qwen2 —
    /// the Llama decoder), a different stored dtype of a probed family (F16, F32), a config
    /// without `model_type`, and the families whose probes do not clear the margin (Qwen3.5/3.8,
    /// Qwen3-VL, Prism, Gemma 4 unified).
    ///
    /// MUTATION: drop the dtype or the `model_type` from `LoadCell::is_in`'s match and this goes
    /// RED.
    #[test]
    fn only_probed_cells_charge_the_payload_once_plus_derived_arrays() {
        let once = |payload: u64| payload + 4 * R + (64 * 4 + R) + H;
        for model_type in ["llama", "qwen3", "gemma2"] {
            let (dir, payload) =
                snapshot(json!({ "model_type": model_type }), &refs(&dense("model")));
            assert_eq!(required(&dir, None), once(payload), "{model_type}");
        }
        for (config, prefix) in [
            (json!({"model_type": "mistral"}), "model"),
            (json!({"model_type": "qwen2"}), "model"),
            (json!({}), "model"),
            (json!({"model_type": "qwen3_5"}), "model.language_model"),
            (json!({"model_type": "qwen3_5_text"}), "model"),
            (json!({"model_type": "qwen3_vl"}), "model.language_model"),
            (json!({"model_type": "prism_hadamard_qwen35"}), "model"),
            (
                json!({"model_type": "gemma4_unified"}),
                "model.language_model",
            ),
        ] {
            let (dir, payload) = snapshot(config.clone(), &refs(&dense(prefix)));
            assert_eq!(required(&dir, None), 2 * payload + H, "{config}");
        }
        for dtype in ["F16", "F32"] {
            let tensors: Tensors = dense("model")
                .into_iter()
                .map(|(n, _, s)| (n, dtype, s))
                .collect();
            let (dir, payload) = snapshot(json!({"model_type": "llama"}), &refs(&tensors));
            assert!(required(&dir, None) >= 2 * payload + H, "llama {dtype}");
        }
    }

    /// Load-time Q4/Q8 adds the packed words (`n·bits/8`) and BF16 scales + biases
    /// (`4·n/group`) of every quantized projection — never of the embedding or the LM head —
    /// on top of the BF16 payload that stays resident until the quantized arrays are evaluated
    /// (the earlier order's bound). No load-time cell is probed with margin, so each is charged
    /// at least two copies.
    #[test]
    fn quantize_at_load_charges_packed_words_scales_and_biases_per_bits() {
        let up = 1024 * 1024;
        for (quantize, quantized) in [
            (core_llm::Quantize::Q4, up / 2 + up / 64 * 4),
            (core_llm::Quantize::Q8, up + up / 64 * 4),
        ] {
            for model_type in ["qwen3_5", "qwen3", "llama", "gemma2", "gemma4_unified"] {
                let tensors = dense("model");
                let (dir, payload) = snapshot(json!({ "model_type": model_type }), &refs(&tensors));
                let facts = SnapshotFacts::read(dir.path()).unwrap();
                let q = crate::provider::quant_spec(quantize).unwrap();
                let earlier = payload + 4 * R + (64 * 4 + R) + (quantized + R);
                assert_eq!(
                    facts.earlier_arrays(Some(q)).unwrap(),
                    earlier,
                    "{model_type}"
                );
                assert_eq!(
                    required(&dir, Some(quantize)),
                    earlier.max(2 * payload) + H,
                    "{model_type} {quantize:?}"
                );
            }
        }
        // A matrix whose input width does not divide the group is never quantized.
        let (dir, payload) = snapshot(
            json!({"model_type": "qwen3"}),
            &[("model.layers.0.mlp.up_proj.weight", "BF16", &[128, 48])],
        );
        let facts = SnapshotFacts::read(dir.path()).unwrap();
        assert_eq!(
            facts.earlier_arrays(Some(QuantSpec::q4())).unwrap(),
            payload + R
        );
        // NVFP4 is refused by name, never priced as another format.
        let mut spec = LoadSpec::dense(dir.path().display().to_string());
        spec.quantize = Some(core_llm::Quantize::Nvfp4);
        assert!(required_bytes(&spec).is_err());
    }

    /// Unverified (architecture × conversion) keeps the two-copy bound: MoE checkpoints,
    /// stored-quantized snapshots outside Qwen3.5, Qwen3-VL load-time quantization, and the
    /// architectures with no measured peak.
    #[test]
    fn unverified_loads_keep_the_two_copy_bound() {
        let cases: [(Value, Tensors, Option<core_llm::Quantize>); 9] = [
            (json!({"model_type": "phi3"}), dense("model"), None),
            (json!({"model_type": "glm4"}), dense("model"), None),
            (json!({"model_type": "deepseek_v2"}), dense("model"), None),
            (json!({"model_type": "gemma4"}), dense("model"), None),
            (json!({"model_type": "other"}), dense("model"), None),
            (
                json!({"model_type": "qwen3_vl"}),
                dense("model.language_model"),
                Some(core_llm::Quantize::Q4),
            ),
            (
                json!({"model_type": "qwen3_5_moe"}),
                vec![(
                    "model.language_model.layers.0.mlp.experts.down_proj".into(),
                    "BF16",
                    vec![2, 1024, 1024],
                )],
                None,
            ),
            (
                json!({"model_type": "qwen2_moe"}),
                vec![(
                    "model.layers.0.mlp.experts.0.down_proj.weight".into(),
                    "BF16",
                    vec![1024, 1024],
                )],
                None,
            ),
            (
                json!({"model_type": "llama"}),
                vec![
                    (
                        "model.layers.0.mlp.up_proj.weight".into(),
                        "U32",
                        vec![1024, 128],
                    ),
                    (
                        "model.layers.0.mlp.up_proj.scales".into(),
                        "BF16",
                        vec![1024, 16],
                    ),
                    (
                        "model.layers.0.mlp.up_proj.biases".into(),
                        "BF16",
                        vec![1024, 16],
                    ),
                ],
                None,
            ),
        ];
        for (config, tensors, quantize) in cases {
            let (dir, payload) = snapshot(config.clone(), &refs(&tensors));
            let derived = payload
                + derived_bytes(
                    &safetensors_headers(dir.path()).unwrap(),
                    quantize.map(|q| crate::provider::quant_spec(q).unwrap()),
                )
                .unwrap();
            let charged = required(&dir, quantize) - H;
            assert!(charged >= 2 * payload, "{config}: {charged} < two copies");
            assert_eq!(charged, derived.max(2 * payload), "{config}");
        }
    }

    /// An unverified load is charged its derived allocations when they exceed two copies: a
    /// per-expert bank quantized to Q8 at load holds its BF16 payload, each expert's quantized
    /// triple and the stacked bank at once.
    #[test]
    fn a_derived_bound_above_two_copies_is_charged_in_full() {
        let (dir, payload) = snapshot(
            json!({"model_type": "qwen2_moe"}),
            &[(
                "model.layers.0.mlp.experts.0.up_proj.weight",
                "BF16",
                &[512, 64],
            )],
        );
        let n = 512 * 64;
        let q = n + n / 64 * 4;
        let derived = payload + R + (q + R) + (q + R);
        assert!(derived > 2 * payload);
        assert_eq!(required(&dir, Some(core_llm::Quantize::Q8)), derived + H);
    }

    /// Per-expert banks are stacked into one array per projection (`SwitchLinear::stack`): a
    /// copy of each expert's loaded form — BF16 when dense, the quantized triple (on top of the
    /// per-expert triple it was stacked from) at load-time Q4.
    #[test]
    fn per_expert_stacking_is_a_copy_of_the_loaded_form() {
        let tensors: [(&str, &str, &[u64]); 1] = [(
            "model.layers.0.mlp.experts.3.up_proj.weight",
            "BF16",
            &[128, 64],
        )];
        let headers = [json!({
            tensors[0].0: {"dtype": "BF16", "shape": [128, 64], "data_offsets": [0, 16384]}
        })];
        let n = 128 * 64;
        assert_eq!(derived_bytes(&headers, None).unwrap(), R + (2 * n + R));
        let q = n / 2 + n / 64 * 4;
        assert_eq!(
            derived_bytes(&headers, Some(QuantSpec::q4())).unwrap(),
            R + 2 * (q + R)
        );
        assert_eq!(expert_index(tensors[0].0), Some(3));
        assert_eq!(
            expert_index("model.layers.0.mlp.shared_experts.up_proj.weight"),
            None
        );
        assert_eq!(
            expert_index("model.layers.0.mlp.experts.gate_up_proj"),
            None
        );
    }

    /// Casts, vector intermediates, word-unaligned checksum views, split copies and vision
    /// transposes are each charged; same-dtype BF16 matrices are not.
    #[test]
    fn conversions_are_charged_and_shared_weights_are_not() {
        let h = json!({
            "model.language_model.layers.0.weight": {"dtype": "BF16", "shape": [1024, 1024]},
            "model.language_model.norm.weight": {"dtype": "BF16", "shape": [1024]},
            "mtp.weight": {"dtype": "F32", "shape": [16, 16]},
            "model.layers.0.layer_scalar": {"dtype": "BF16", "shape": [1]},
            "model.visual.proj.weight": {"dtype": "F16", "shape": [32, 32]},
            "model.visual.patch_embed.proj.weight": {"dtype": "BF16", "shape": [8, 2, 2]},
            "model.vision_embedder.pos_embedding": {"dtype": "BF16", "shape": [4, 2, 8]},
        });
        let expected = 7 * R
            + (1024 * 4 + R)
            + (16 * 16 * 2 + R)
            + (2 * 4 + R)
            + (4 + R)
            + (32 * 4 + R)
            + (64 * 2 + R);
        assert_eq!(derived_bytes(&[h], None).unwrap(), expected);
        assert!(derived_bytes(&[json!({"x":{"dtype":"F32","shape":[u64::MAX,2]}})], None).is_err());
        assert!(quantized_bytes(u64::MAX, QuantSpec::q8()).is_err());
    }

    /// A materialization group is one layer (decoder or MTP), or `""` outside every stack.
    #[test]
    fn tensors_group_by_their_layer() {
        for (name, group) in [
            (
                "model.language_model.layers.12.mlp.experts.gate_up_proj",
                "model.language_model.layers.12.",
            ),
            ("model.layers.3.self_attn.q_proj.weight", "model.layers.3."),
            ("mtp.layers.0.mlp.gate.weight", "mtp.layers.0."),
            ("model.language_model.embed_tokens.weight", ""),
            ("lm_head.weight", ""),
            ("mtp.fc.weight", ""),
            ("model.num_layers.weight", ""),
            ("model.layers.x.weight", ""),
        ] {
            assert_eq!(materialization_group(name), group, "{name}");
        }
    }

    /// The materialize-at-load bound, term by term, on a fused MoE layer and the arrays outside
    /// the stack at load-time Q4: every resident array in its loaded form plus the largest
    /// group's consumed sources and transients — never the whole payload.
    #[test]
    fn the_materialized_bound_is_every_resident_array_plus_the_largest_group_window() {
        let h = json!({
            "model.language_model.embed_tokens.weight": {"dtype": "BF16", "shape": [16, 64]},
            "model.language_model.norm.weight": {"dtype": "BF16", "shape": [64]},
            "lm_head.weight": {"dtype": "F32", "shape": [16, 64]},
            "model.language_model.layers.0.mlp.experts.gate_up_proj":
                {"dtype": "BF16", "shape": [4, 128, 64]},
            "model.language_model.layers.0.mlp.gate.weight": {"dtype": "BF16", "shape": [4, 64]},
            "model.language_model.layers.1.self_attn.q_proj.weight":
                {"dtype": "F32", "shape": [64, 64]},
        });
        let q4 = |n: u64| n / 2 + n / 64 * 4;
        let bank = 4 * 128 * 64;
        // Resident: the BF16 embedding as stored; the norm vector max(stored, BF16); the F32 head
        // cast to BF16; the fused bank and q_proj quantized (three buffers each); the router
        // as stored.
        let resident = (16 * 64 * 2 + R)
            + (64 * 2 + R)
            + (16 * 64 * 2 + R)
            + (q4(bank) + 3 * R)
            + (4 * 64 * 2 + R)
            + (q4(64 * 64) + 3 * R);
        // Windows: outside the stack the norm's source and intermediates and the head's F32
        // source; layer 0 the fused bank's BF16 source; layer 1 q_proj's F32 source and its
        // BF16 cast ahead of the quantize.
        let top = (64 * 2 + R) + (64 * 4 + R) + (16 * 64 * 4 + R);
        let layer0 = bank * 2 + R;
        let layer1 = (64 * 64 * 4 + R) + (64 * 64 * 2 + R);
        assert!(layer0 > top && layer0 > layer1);
        // Outstanding under pacing: the largest group's consumed sources (layer 0's bank; the
        // router is kept as stored), plus the pacing slack.
        let outstanding = bank * 2 + R;
        assert_eq!(
            materialized_bound(&[h], Some(QuantSpec::q4())).unwrap(),
            resident
                + layer0
                + outstanding
                + crate::primitives::weights::MATERIALIZE_PACE_SLACK_BYTES
        );
    }

    /// Per-expert tensors are stacked from their loaded form: the stacked copy stays, the source
    /// and (when quantized at load) each expert's quantized triple are the group's window.
    #[test]
    fn per_expert_banks_are_resident_once_stacked() {
        let h = json!({
            "model.layers.0.mlp.experts.0.up_proj.weight": {"dtype": "BF16", "shape": [128, 64]},
            "model.layers.0.mlp.experts.1.up_proj.weight": {"dtype": "BF16", "shape": [128, 64]},
        });
        let n = 128 * 64;
        let outstanding =
            2 * (2 * n + R) + crate::primitives::weights::MATERIALIZE_PACE_SLACK_BYTES;
        assert_eq!(
            materialized_bound(std::slice::from_ref(&h), None).unwrap(),
            2 * (2 * n + R) + 2 * (2 * n + R) + outstanding
        );
        let q = n / 2 + n / 64 * 4;
        assert_eq!(
            materialized_bound(&[h], Some(QuantSpec::q4())).unwrap(),
            2 * (q + 3 * R) + 2 * ((2 * n + R) + (q + 3 * R)) + outstanding
        );
    }

    /// Media tensors (vision towers, projectors, audio) are loaded by their own constructors,
    /// which keep every source beside its derived arrays: resident at payload plus every derived
    /// term, never quantized, never in a window.
    #[test]
    fn media_tensors_stay_resident_with_their_derived_arrays() {
        let h = json!({
            "model.visual.blocks.0.attn.qkv.weight": {"dtype": "F32", "shape": [64, 64]},
            "model.visual.patch_embed.proj.weight": {"dtype": "BF16", "shape": [8, 2, 64]},
        });
        let t = |name, dtype, shape: &[u64]| HeaderTensor {
            name,
            dtype,
            elements: shape.iter().product(),
            stored: shape.iter().product::<u64>() * if dtype == "F32" { 4 } else { 2 },
            shape: shape.to_vec(),
        };
        let qkv = t("model.visual.blocks.0.attn.qkv.weight", "F32", &[64, 64]);
        let patch = t("model.visual.patch_embed.proj.weight", "BF16", &[8, 2, 64]);
        let expected = (qkv.stored + R + tensor_derived(&qkv, None).unwrap())
            + (patch.stored + R + tensor_derived(&patch, None).unwrap());
        for quant in [None, Some(QuantSpec::q4())] {
            assert_eq!(
                materialized_bound(std::slice::from_ref(&h), quant).unwrap(),
                expected + crate::primitives::weights::MATERIALIZE_PACE_SLACK_BYTES
            );
        }
    }

    /// A wide Qwen3.6-MoE fixture (every projection's input a whole number of Q4/Q8 groups,
    /// stored F32 so every tensor is converted), fused or per-expert, written to disk.
    fn wide_qwen35_moe_snapshot(fused: bool) -> (tempfile::TempDir, crate::models::Qwen35Config) {
        use crate::models::qwen35::tests::{cfg_json_moe_mtp, seeded_moe_mtp_tensors};
        let mut config = cfg_json_moe_mtp();
        config["model_type"] = json!("qwen3_5_moe");
        let tc = config["text_config"].as_object_mut().unwrap();
        for (key, value) in [
            ("hidden_size", 128),
            ("intermediate_size", 128),
            ("head_dim", 32),
            ("linear_key_head_dim", 32),
            ("linear_value_head_dim", 32),
            ("moe_intermediate_size", 128),
            ("shared_expert_intermediate_size", 128),
        ] {
            tc.insert(key.into(), json!(value));
        }
        let cfg = crate::models::Qwen35Config::from_json(&config).unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
        let tensors = seeded_moe_mtp_tensors(&cfg, fused);
        mlx_rs::Array::save_safetensors(
            tensors.iter().map(|(k, v)| (k.as_str(), v)),
            None,
            dir.path().join("model.safetensors"),
        )
        .unwrap();
        (dir, cfg)
    }

    /// sc-24446: the bound covers what the provider's load order actually allocates. A wide
    /// Qwen3.6-MoE fixture (fused and per-expert, with its MTP head) is loaded exactly as the
    /// provider loads it — lazily, then [`Weights::materialize_groups`] — dense and at Q8 / Q4.
    /// Measured two ways: MLX's peak active memory stays within [`materialized_bound`] less the
    /// pacing slack (which only the footprint's host and driver noise may use), and the exact
    /// settled `phys_footprint` peak growth stays within the bound plus the load's fixed host
    /// allowance. The bound, less its slack, never exceeds the payload-plus-every-derived-array
    /// bound of the earlier order.
    ///
    /// MUTATION: skip releasing each group's sources (`Weights::materialize_groups`) and the
    /// active peak exceeds the bound from the first (fused, unquantized F32) case on.
    ///
    /// [`Weights::materialize_groups`]: crate::primitives::Weights::materialize_groups
    #[test]
    fn the_materialized_bound_covers_the_provider_load_order_on_a_fixture() {
        use crate::models::Qwen35Model;
        use crate::primitives::Weights;
        use crate::test_fixture::footprint;
        let slack = crate::primitives::weights::MATERIALIZE_PACE_SLACK_BYTES;
        for fused in [true, false] {
            for quant in [None, Some(QuantSpec::q8()), Some(QuantSpec::q4())] {
                let (dir, cfg) = wide_qwen35_moe_snapshot(fused);
                let headers = safetensors_headers(dir.path()).unwrap();
                let bound = materialized_bound(&headers, quant).unwrap();
                let payload = core_llm::checkpoint_payload_bytes(dir.path()).unwrap();
                let earlier = payload + derived_bytes(&headers, quant).unwrap();
                assert!(
                    bound - slack <= earlier,
                    "{fused} {quant:?}: {bound} > {earlier}"
                );

                mlx_rs::memory::clear_cache();
                footprint::settle();
                let base_footprint = footprint::current();
                footprint::reset_peak();
                let base = mlx_rs::memory::get_active_memory();
                mlx_rs::memory::reset_peak_memory();
                let mut w = Weights::from_dir(dir.path()).unwrap();
                let model =
                    Qwen35Model::build_lazy(&w, "model.language_model", cfg, quant).unwrap();
                let report = w.materialize_groups(&model.param_groups()).unwrap();
                let active = mlx_rs::memory::get_peak_memory().saturating_sub(base) as u64;
                let footprint = footprint::peak_since_reset().saturating_sub(base_footprint);
                assert_eq!(report.leftover, 0);
                assert!(
                    active <= bound - slack,
                    "fused={fused} {quant:?}: active peak {active} exceeds the bound {}",
                    bound - slack
                );
                assert!(
                    footprint <= bound + host_bytes(0).unwrap(),
                    "fused={fused} {quant:?}: footprint peak {footprint} exceeds {}",
                    bound + host_bytes(0).unwrap()
                );
                if quant.is_some() {
                    // Quantizing at load releases every converted source group by group.
                    assert!(bound - slack < earlier, "fused={fused} {quant:?}");
                }
                drop(model);
            }
        }
    }

    /// Where the committed snapshot manifests live: header-only facts of each pinned snapshot
    /// a `MEASURED` probe loaded — its `config.json`, every tensor's name, dtype and shape, the
    /// shards' total bytes and the `tokenizer.json` size. No weights.
    fn manifest_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/load_admission")
    }

    fn manifest_value(facts: &SnapshotFacts, source: &str) -> Value {
        let mut tensors = serde_json::Map::new();
        for header in &facts.headers {
            for (name, info) in header.as_object().unwrap() {
                if name == "__metadata__" {
                    continue;
                }
                tensors.insert(
                    name.clone(),
                    json!({"dtype": info["dtype"], "shape": info["shape"]}),
                );
            }
        }
        json!({
            "source": source,
            "config": facts.config,
            "payload": facts.payload,
            "tokenizer_bytes": facts.tokenizer_bytes,
            "tensors": tensors,
        })
    }

    /// A committed manifest as the facts [`required_bytes`] reads.
    pub(super) fn manifest_facts(name: &str) -> SnapshotFacts {
        let path = manifest_dir().join(format!("{name}.json"));
        let v: Value = serde_json::from_reader(
            File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
        )
        .unwrap();
        SnapshotFacts {
            config: v["config"].clone(),
            headers: vec![v["tensors"].clone()],
            payload: v["payload"].as_u64().unwrap(),
            tokenizer_bytes: v["tokenizer_bytes"].as_u64().unwrap(),
        }
    }

    /// The pinned snapshots, by manifest name: (name, path under `LOAD_PROBE_SNAPSHOT_ROOT`, or
    /// absolute). (A prepared per-expert Qwen3.6 snapshot's manifest is 9.4 MB — 92K tensor
    /// entries — and is not committed; its cell stays unverified.)
    const PINNED: &[(&str, &str)] = &[
        ("gemma-2-2b-it", "models--SceneWorks--gemma-2-2b-it/snapshots/684c553b5b41a1c835989d89f62f585e6269a7de"),
        ("llama-3.2-1b-instruct-bf16", "models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/863c846a9ac6fad4e49e1743d52984dff262e953"),
        ("qwen3-1.7b-bf16", "models--mlx-community--Qwen3-1.7B-bf16/snapshots/9cd6692855d3e06772228e9a962b2606359b2d24"),
        ("qwen3-8b", "models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"),
        ("ltx-2.5-enhancer", "models--SceneWorks--ltx-2.5-mlx/snapshots/791ef61731ad067bd13ebff8cc0f07532476d9ef/enhancer"),
        ("qwen3.8-27b", "models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
        ("qwen3.6-35b-a3b", "models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"),
    ];

    fn pinned_path(hub: &std::path::Path, rel: &str) -> std::path::PathBuf {
        if rel.starts_with('/') {
            std::path::PathBuf::from(rel)
        } else {
            hub.join(rel)
        }
    }

    /// (Re)write the committed manifests from the local pinned snapshots — header-only.
    #[test]
    #[ignore = "writes testdata from the pinned snapshots under LOAD_PROBE_SNAPSHOT_ROOT"]
    fn write_pinned_snapshot_manifests() {
        let hub = std::path::PathBuf::from(
            std::env::var("LOAD_PROBE_SNAPSHOT_ROOT").expect("set LOAD_PROBE_SNAPSHOT_ROOT"),
        );
        std::fs::create_dir_all(manifest_dir()).unwrap();
        for (name, rel) in PINNED {
            let facts = SnapshotFacts::read(&pinned_path(&hub, rel)).unwrap();
            let value = manifest_value(&facts, rel);
            std::fs::write(
                manifest_dir().join(format!("{name}.json")),
                serde_json::to_string(&value).unwrap() + "\n",
            )
            .unwrap();
        }
    }

    /// Header-only audit: each committed manifest still matches its local pinned snapshot.
    #[test]
    #[ignore = "requires the pinned snapshots under LOAD_PROBE_SNAPSHOT_ROOT"]
    fn committed_manifests_match_the_pinned_snapshots() {
        let hub = std::path::PathBuf::from(
            std::env::var("LOAD_PROBE_SNAPSHOT_ROOT").expect("set LOAD_PROBE_SNAPSHOT_ROOT"),
        );
        for (name, rel) in PINNED {
            let local = manifest_value(&SnapshotFacts::read(&pinned_path(&hub, rel)).unwrap(), rel);
            let committed: Value = serde_json::from_reader(
                File::open(manifest_dir().join(format!("{name}.json"))).unwrap(),
            )
            .unwrap();
            assert_eq!(local, committed, "{name}");
        }
    }

    /// Guarded real-weight probes (sc-24446), 2026-10-01, Apple M5 Max (Mac17,6), 128 GiB unified
    /// memory, macOS 26: one load through `LlamaProvider::load` plus one-token requests
    /// (`tests/load_admission_probe.rs`), each figure footprint growth over a baseline taken after
    /// Metal and MLX's kernel library initialized.
    ///
    /// **One method for every cell** ([`Measured::peak`]): the load's share of the peak is the
    /// larger of the footprint growth through the end of the load and the first one-token
    /// request's peak growth less that request's own working set (a second identical request on
    /// the materialized model — what request admission prices). These rows were sampled every
    /// 100 ms and ran the earlier verify-everything-then-derive order; the working set is MLX's
    /// active peak (the probe did not record its footprint), which can only overstate the load's
    /// share. Each row's estimate is recomputed from code, from the committed manifest of its
    /// pinned snapshot ([`manifest_facts`]). Recorded evidence, not a machine golden: the tests
    /// assert relations only.
    pub(super) const MEASURED: &[Measured] = &[
        Measured {
            config: "Gemma 2 2B-it BF16",
            manifest: "gemma-2-2b-it",
            quantize: None,
            order: Order::Earlier,
            sampling: Sampling::Every100Ms,
            after_load: Some(5_518_511_596),
            first_request: 9_052_000_000,
            request_working_set: 3_528_000_000,
        },
        Measured {
            config: "Gemma 2 2B-it load-time Q4",
            manifest: "gemma-2-2b-it",
            quantize: Some(core_llm::Quantize::Q4),
            order: Order::Earlier,
            sampling: Sampling::Every100Ms,
            after_load: None,
            first_request: 9_013_000_000,
            request_working_set: 2_426_000_000,
        },
        Measured {
            config: "Llama 3.2 1B Instruct BF16",
            manifest: "llama-3.2-1b-instruct-bf16",
            quantize: None,
            order: Order::Earlier,
            sampling: Sampling::Every100Ms,
            after_load: Some(2_622_474_088),
            first_request: 2_680_000_000,
            request_working_set: 35_000_000,
        },
        Measured {
            config: "Llama 3.2 1B Instruct load-time Q4",
            manifest: "llama-3.2-1b-instruct-bf16",
            quantize: Some(core_llm::Quantize::Q4),
            order: Order::Earlier,
            sampling: Sampling::Every100Ms,
            after_load: None,
            first_request: 3_065_000_000,
            request_working_set: 41_000_000,
        },
        Measured {
            config: "Llama 3.2 1B Instruct load-time Q8",
            manifest: "llama-3.2-1b-instruct-bf16",
            quantize: Some(core_llm::Quantize::Q8),
            order: Order::Earlier,
            sampling: Sampling::Every100Ms,
            after_load: None,
            first_request: 3_723_000_000,
            request_working_set: 38_000_000,
        },
        Measured {
            config: "Qwen3-1.7B BF16",
            manifest: "qwen3-1.7b-bf16",
            quantize: None,
            order: Order::Earlier,
            sampling: Sampling::Every100Ms,
            after_load: Some(3_573_647_904),
            first_request: 3_658_000_000,
            request_working_set: 46_000_000,
        },
        Measured {
            config: "Qwen3-1.7B load-time Q4",
            manifest: "qwen3-1.7b-bf16",
            quantize: Some(core_llm::Quantize::Q4),
            order: Order::Earlier,
            sampling: Sampling::Every100Ms,
            after_load: None,
            first_request: 4_456_000_000,
            request_working_set: 38_000_000,
        },
        Measured {
            config: "Qwen3-8B load-time Q4",
            manifest: "qwen3-8b",
            quantize: Some(core_llm::Quantize::Q4),
            order: Order::Earlier,
            sampling: Sampling::Every100Ms,
            after_load: None,
            first_request: 19_962_000_000,
            request_working_set: 87_000_000,
        },
        Measured {
            config: "Qwen3-8B load-time Q8",
            manifest: "qwen3-8b",
            quantize: Some(core_llm::Quantize::Q8),
            order: Order::Earlier,
            sampling: Sampling::Every100Ms,
            after_load: None,
            first_request: 23_988_000_000,
            request_working_set: 82_000_000,
        },
        Measured {
            config: "Qwen3-8B BF16",
            manifest: "qwen3-8b",
            quantize: None,
            order: Order::Earlier,
            sampling: Sampling::Every100Ms,
            after_load: Some(16_513_117_776),
            first_request: 16_610_000_000,
            request_working_set: 53_000_000,
        },
        Measured {
            config: "Gemma 4 unified (LTX-2.5 enhancer) BF16",
            manifest: "ltx-2.5-enhancer",
            quantize: None,
            order: Order::Earlier,
            sampling: Sampling::Every100Ms,
            after_load: Some(24_304_141_352),
            first_request: 30_482_000_000,
            request_working_set: 5_416_000_000,
        },
        Measured {
            config: "Gemma 4 unified (LTX-2.5 enhancer) load-time Q4",
            manifest: "ltx-2.5-enhancer",
            quantize: Some(core_llm::Quantize::Q4),
            order: Order::Earlier,
            sampling: Sampling::Every100Ms,
            after_load: None,
            first_request: 34_391_000_000,
            request_working_set: 4_466_000_000,
        },
        Measured {
            config: "Qwen3.8-27B load-time Q4",
            manifest: "qwen3.8-27b",
            quantize: Some(core_llm::Quantize::Q4),
            order: Order::Earlier,
            sampling: Sampling::Every100Ms,
            after_load: None,
            first_request: 69_713_000_000,
            request_working_set: 236_000_000,
        },
        Measured {
            config: "Qwen3.8-27B load-time Q8",
            manifest: "qwen3.8-27b",
            quantize: Some(core_llm::Quantize::Q8),
            order: Order::Earlier,
            sampling: Sampling::Every100Ms,
            after_load: None,
            first_request: 80_870_000_000,
            request_working_set: 230_000_000,
        },
    ];

    /// Which load order a probe ran.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum Order {
        /// Every accessed source verified up front, derived arrays built by the first forward.
        Earlier,
        /// The provider's group-by-group materialize-at-load order.
        Materialized,
    }

    /// How a probe read its peak.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum Sampling {
        /// `phys_footprint` sampled every 100 ms (and once after each step): a peak that a
        /// `clear_cache` cuts short between two samples is missed — at most what was allocated
        /// since the last sample, bounded by the arrays the first request derived.
        Every100Ms,
        /// The kernel's exact maximum (`ri_interval_max_phys_footprint` /
        /// `ri_lifetime_max_phys_footprint`): nothing missed. The re-probes record it.
        #[allow(dead_code)]
        Exact,
    }

    /// One recorded probe.
    pub(super) struct Measured {
        /// What was loaded and how.
        pub(super) config: &'static str,
        /// The committed manifest of the pinned snapshot (`crates/llm/testdata/load_admission`).
        pub(super) manifest: &'static str,
        pub(super) quantize: Option<core_llm::Quantize>,
        pub(super) order: Order,
        pub(super) sampling: Sampling,
        /// Footprint growth through the end of the load, when recorded.
        pub(super) after_load: Option<u64>,
        /// The first one-token request's peak footprint growth over the pre-load baseline.
        pub(super) first_request: u64,
        /// The request's own working set (a second identical request).
        pub(super) request_working_set: u64,
    }

    impl Measured {
        /// The load's share of the peak — the one method every cell is measured by.
        pub(super) fn peak(&self) -> u64 {
            self.after_load
                .unwrap_or(0)
                .max(self.first_request.saturating_sub(self.request_working_set))
        }

        fn quant(&self) -> Option<QuantSpec> {
            self.quantize
                .map(|q| crate::provider::quant_spec(q).unwrap())
        }

        /// The bound this probe's order is charged, recomputed from the manifest.
        pub(super) fn estimate(&self, facts: &SnapshotFacts) -> u64 {
            let arrays = match self.order {
                Order::Earlier => facts.earlier_arrays(self.quant()).unwrap(),
                Order::Materialized => materialized_bound(&facts.headers, self.quant()).unwrap(),
            };
            arrays + facts.host().unwrap()
        }

        /// The margin the estimate must keep above [`Measured::peak`]: 0.5 % for run-to-run
        /// variation (the tightest exact relation recorded so far), plus — for a sampled probe —
        /// everything the sampler can have missed: the arrays the first request derived
        /// ([`derived_bytes`]), which a `clear_cache` between two 100 ms samples can hide. (The
        /// 0.5 s memory guard was observed under-reading one 81 GB load by 5 GiB.)
        pub(super) fn margin(&self, facts: &SnapshotFacts) -> u64 {
            let variation = self.peak() / 200;
            match self.sampling {
                Sampling::Exact => variation,
                Sampling::Every100Ms => {
                    variation + derived_bytes(&facts.headers, self.quant()).unwrap()
                }
            }
        }

        /// The cell this probe measured.
        pub(super) fn cell(&self, facts: &SnapshotFacts) -> LoadCell {
            LoadCell::of(&facts.config, &facts.headers, self.quant())
        }

        pub(super) fn covered(&self, facts: &SnapshotFacts) -> bool {
            self.estimate(facts) >= self.peak() + self.margin(facts)
        }
    }

    /// Print every recorded probe recomputed from code (for the doc's table).
    #[test]
    #[ignore = "prints the recorded probes' recomputed relations"]
    fn print_recorded_probes() {
        for m in MEASURED {
            let facts = manifest_facts(m.manifest);
            println!(
                "PROBE {} | cell {:?} | estimate {} | peak {} | margin {} | covered {}",
                m.config,
                m.cell(&facts),
                m.estimate(&facts),
                m.peak(),
                m.margin(&facts),
                m.covered(&facts)
            );
        }
    }

    /// sc-24446: every recorded probe is recomputed from code — its estimate from the committed
    /// manifest of its pinned snapshot, its peak by the one method — and a cell is verified
    /// (`EARLIER_VERIFIED` / `MATERIALIZED_VERIFIED`) **only** if a probe of that order measured
    /// it, and every such probe is covered with the margin. A probe that is not covered keeps its
    /// cell on the two-copy floor.
    ///
    /// MUTATION: add a cell to either table without a covering probe (e.g. Gemma 4 unified BF16,
    /// whose uniform peak 25.066 GB exceeds its 25.065 GB estimate, or Llama load-time Q4), or drop
    /// the sampler term from [`Measured::margin`] (which would let the sampled load-time probes
    /// count as covered), and this goes RED.
    #[test]
    fn verified_cells_are_exactly_the_probed_and_covered_ones() {
        for (table, order) in [
            (EARLIER_VERIFIED, Order::Earlier),
            (MATERIALIZED_VERIFIED, Order::Materialized),
        ] {
            for cell in table {
                let probes: Vec<&Measured> = MEASURED
                    .iter()
                    .filter(|m| m.order == order)
                    .filter(|m| m.cell(&manifest_facts(m.manifest)).is_in(&[*cell]))
                    .collect();
                assert!(
                    !probes.is_empty(),
                    "{cell:?}: verified with no probe behind it"
                );
            }
        }
        for m in MEASURED {
            let facts = manifest_facts(m.manifest);
            let table = match m.order {
                Order::Earlier => EARLIER_VERIFIED,
                Order::Materialized => MATERIALIZED_VERIFIED,
            };
            if m.cell(&facts).is_in(table) {
                assert!(
                    m.covered(&facts),
                    "{}: estimate {} does not cover peak {} + margin {}",
                    m.config,
                    m.estimate(&facts),
                    m.peak(),
                    m.margin(&facts)
                );
            }
        }
        // The Gemma 4 unified BF16 probe, by the uniform method, sits above its estimate.
        let gemma4 = MEASURED
            .iter()
            .find(|m| m.config == "Gemma 4 unified (LTX-2.5 enhancer) BF16")
            .unwrap();
        let facts = manifest_facts(gemma4.manifest);
        assert!(gemma4.peak() > gemma4.estimate(&facts));
        assert!(!gemma4.cell(&facts).is_in(EARLIER_VERIFIED));
        // A sampled load-time probe cannot clear its margin: the first request's derived arrays,
        // which a `clear_cache` between samples can hide, are as large as the margin it needs.
        for m in MEASURED.iter().filter(|m| m.quantize.is_some()) {
            let facts = manifest_facts(m.manifest);
            assert_eq!(m.sampling, Sampling::Every100Ms);
            assert!(
                !m.covered(&facts),
                "{}: a sampled load-time probe counted",
                m.config
            );
        }
    }

    /// Header-only operational audit: never constructs a tensor or starts a model.
    #[test]
    #[ignore = "requires MLX_LOAD_ADMISSION_CASES containing pinned local paths and expected bounds"]
    fn pinned_header_only_load_admission_bounds() {
        let path = std::env::var("MLX_LOAD_ADMISSION_CASES").expect("set MLX_LOAD_ADMISSION_CASES");
        let cases: Value = serde_json::from_reader(File::open(path).unwrap()).unwrap();
        let mut mismatches = Vec::new();
        for case in cases.as_array().unwrap() {
            let mut spec = LoadSpec::dense(case["source"].as_str().unwrap());
            if let Some(projector) = case["projector_source"].as_str() {
                spec = spec.with_projector(projector);
            }
            spec.quantize = match case["quantize"].as_str() {
                None => None,
                Some("q4") => Some(core_llm::Quantize::Q4),
                Some("q8") => Some(core_llm::Quantize::Q8),
                Some(other) => panic!("unknown quantize {other}"),
            };
            let actual = required_bytes(&spec).unwrap();
            println!(
                "{}",
                json!({"source":spec.source,"projector_source":spec.projector_source,
                    "quantize":case["quantize"],"required_bytes":actual})
            );
            if case["expected_bytes"].as_u64() != Some(actual) {
                mismatches.push(spec.source);
            }
        }
        assert!(mismatches.is_empty(), "{mismatches:?}");
    }

    #[test]
    fn gguf_counts_source_conversion_outputs_and_largest_staging() {
        assert_eq!(
            gguf_tensor_bytes(1024, 128, 142, false).unwrap(),
            (864, 832)
        );
        assert_eq!(
            gguf_tensor_bytes(1024, 128, 143, false).unwrap(),
            (864, 832)
        );
        assert_eq!(
            gguf_tensor_bytes(1024, 128, 1, false).unwrap(),
            (8192, 8192)
        );
        assert_eq!(gguf_tensor_bytes(1024, 128, 8, true).unwrap(), (4096, 4096));
        assert!(gguf_tensor_bytes(u64::MAX, 128, 143, false).is_err());
    }
}
