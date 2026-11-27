//! Family-generic offline builder for **Candle-native GGML tiers** (SW-7285).
//!
//! A published MLX-affine tier stores each quantized projection as a triple — `{base}.weight` (U32
//! codes) + `{base}.scales` + `{base}.biases`. A Windows/Linux load has to repack that into GGML
//! blocks on first use. This module does the repack **offline** and persists the result, so the
//! loader reads one stored block tensor per projection and the install is a straight mmap.
//!
//! Output layout (the loader and SceneWorks depend on it, keep it exactly):
//!
//! * each triple becomes ONE `{base}.weight` **stored GGML block tensor**, `U8 [out, in / 32,
//!   block_bytes]` (`Q4_1` 20 B, `Q8_0` 34 B); `.scales` / `.biases` are dropped;
//! * every other tensor is copied byte-for-byte at its stored dtype and shape;
//! * shard file names are preserved; `*.index.json` is rewritten (`weight_map` from what was written,
//!   `metadata.total_size` restated);
//! * `config.json`'s `quantization` block becomes `{"bits": N, "storage": "ggml"}`;
//! * every other sidecar is copied verbatim;
//! * the component is written to `<dst>.partial` and renamed into place at the end; an existing
//!   `dst` is a hard error.
//!
//! # Source modes ([`SourceMode`])
//!
//! * [`SourceMode::FromMlx`] — quantize from the MLX triple itself. **Q4 is lossless** (`Q4_1`
//!   carries the MLX per-group `scale`/`bias` as its `d`/`m`, up to the f16 cast); the group size is
//!   read from the component's `config.json` (`quantization.group_size`, default 64) and may be any
//!   multiple of 32 — each MLX group splits into `group / 32` consecutive `Q4_1` blocks sharing the
//!   scale/bias. Q8 through this mode is the legacy dequant-then-requant (lossy).
//! * [`SourceMode::FromBf16`] — **Q8_0 straight from the dense source.** The MLX q8 component is
//!   only a *structural template*: a tensor is quantized iff it has a `.scales` sibling there; its
//!   dense `{base}.weight` is read from the bf16/f16/f32 source directory (any sharding — tensors
//!   are looked up by name across all its shards) and quantized with candle's own `Q8_0` quantizer.
//!   The template can be **header-only** (no q8 weights downloaded): `config.json`, optionally the
//!   index, and one `<shard>.safetensors.header.json` dump per shard ([`dump_headers`]). Then the
//!   non-quantized tensors also come from the dense source, cast to the dtype the template header
//!   declares (every cast is reported in [`ComponentReport::casts`]). With a full-tier template
//!   (real shards) non-quantized tensors are copied from the template verbatim.
//!
//! Processing streams tensor by tensor: a source shard is read one tensor at a time by seeking, so
//! peak memory is one output shard plus one tensor (plus its f32 working copy for Q8).
//!
//! # Layouts beyond the diffusers component directory (SW-7285 C2)
//!
//! * **Stacked MoE experts** — a rank-3 triple (`[E, out, in_packed]` codes, `[E, out, in / g]`
//!   scales/biases; Lens's gpt-oss text encoder) becomes ONE rank-4 stored tensor `[E, out, in / 32,
//!   block_bytes]` (what `candle-gen-lens`'s `stored_experts` slices per expert). From-bf16 reads a
//!   float `[E, out, in]` source, or — the Lens bf16 tier's actual form — the **MXFP4** pair
//!   `{base}_blocks` / `{base}_scales`, dequantized exactly and quantized to `Q8_0` per expert
//!   ([`ComponentReport::mxfp4_sources`]).
//! * **Other quantization markers** — `config.json` `quantization.bits` (the MLX convention), else
//!   `quantize_config.json` (Wan: top-level `bits`; LTX-2.3: nested), rewritten by
//!   [`ggml_quantize_config`]; else (ideogram / SenseNova, which write no marker) the width is
//!   inferred from the packed shapes at the MLX default group 64 and must agree across the
//!   component, and `config.json` gains `quantization = {"bits": N, "storage": "ggml"}`.
//! * **Split tiers** ([`convert_split_tier`]) — the LTX `split_model.json` flat directories: same
//!   file names, packed files converted with their `__metadata__` kept, the rest copied, and the
//!   manifest restated ([`ggml_split_manifest`], marker `"quantization_storage": "ggml"`).
//!
//! The diffusers-component output above is unchanged byte for byte (pinned by a digest test).

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use candle_layers::llm_projection::to_ggml_block_tensor;
use serde_json::Value;

use crate::quant::{mlx_packed_bits_gs, repack_packed_weight_uncached, MLX_GROUP_SIZE};

/// `quantization.storage` value that marks a stored-GGML tier (the `candle-llm` sc-19375 marker).
pub const GGML_STORAGE: &str = "ggml";

/// Elements per `Q4_1` / `Q8_0` block.
const GGML_BLOCK: usize = 32;

/// Typed failure of a tier build.
#[derive(Debug, thiserror::Error)]
pub enum GgmlTierError {
    #[error("ggml tier: {what}: {source}")]
    Io {
        what: String,
        #[source]
        source: std::io::Error,
    },
    #[error("ggml tier: {} already exists - refusing to overwrite a tier", .0.display())]
    DestinationExists(PathBuf),
    #[error("ggml tier: no .safetensors in {}", .0.display())]
    NoShards(PathBuf),
    #[error("ggml tier: {} has no config.json", .0.display())]
    MissingConfig(PathBuf),
    #[error("ggml tier: config.json: {0}")]
    Config(String),
    #[error(
        "ggml tier: MLX group_size {group_size} is not a multiple of the GGML block ({GGML_BLOCK})"
    )]
    GroupSize { group_size: usize },
    #[error("ggml tier: {path}: {detail}")]
    BadSafetensors { path: String, detail: String },
    #[error("ggml tier: {tensor}: {detail}")]
    OrphanTriple { tensor: String, detail: String },
    #[error("ggml tier: `{tensor}` is {dtype}, expected the U32 code stream of an MLX pack")]
    NotPackedCodes { tensor: String, dtype: String },
    #[error(
        "ggml tier: `{tensor}` is a rank-{rank} packed tensor; only rank-2 projections and rank-3 \
         stacked experts convert"
    )]
    UnsupportedRank { tensor: String, rank: usize },
    #[error("ggml tier: `{tensor}` shapes disagree: {detail}")]
    TripleShape { tensor: String, detail: String },
    #[error(
        "ggml tier: {} declares no quantization marker and its packed triples do not agree on one \
         width at the MLX default group {MLX_GROUP_SIZE}: `{tensor}` packs {bits} bits, `{first}` \
         packs {first_bits}", .dir.display()
    )]
    InferredBits {
        dir: PathBuf,
        tensor: String,
        bits: usize,
        first: String,
        first_bits: usize,
    },
    #[error("ggml tier: split tier {}: {detail}", .dir.display())]
    Split { dir: PathBuf, detail: String },
    #[error(
        "ggml tier: `{tensor}` packs {inferred}-bit codes at group {group_size}, but the \
         component config declares {expected} bits (wrong group_size?)"
    )]
    BitsMismatch {
        tensor: String,
        inferred: usize,
        expected: usize,
        group_size: usize,
    },
    #[error("ggml tier: from-bf16 needs a q8 template (bits 8); config declares bits {0}")]
    TemplateNotQ8(u64),
    #[error("ggml tier: quantized tensor `{tensor}` is missing from the dense source (looked up `{looked_up}`)")]
    MissingDenseSource { tensor: String, looked_up: String },
    #[error(
        "ggml tier: dense source `{tensor}` has shape {found:?}, the MLX template needs {expected:?}"
    )]
    DenseShapeMismatch {
        tensor: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    #[error("ggml tier: dense source `{tensor}` is {dtype}, expected bf16/f16/f32")]
    DenseDtype { tensor: String, dtype: String },
    #[error(
        "ggml tier: `{tensor}` has in_features {in_features}, not a multiple of the GGML block ({GGML_BLOCK})"
    )]
    InFeaturesNotBlockMultiple { tensor: String, in_features: usize },
    #[error("ggml tier: tensor `{0}` appears in more than one dense-source shard")]
    DuplicateDenseTensor(String),
    #[error("ggml tier: {0}")]
    Spec(String),
    #[error("ggml tier: {0}")]
    Candle(String),
    #[error("ggml tier: progress callback: {0}")]
    Callback(String),
}

impl From<candle_core::Error> for GgmlTierError {
    fn from(e: candle_core::Error) -> Self {
        Self::Candle(e.to_string())
    }
}

impl From<GgmlTierError> for crate::CandleError {
    fn from(e: GgmlTierError) -> Self {
        crate::CandleError::Msg(e.to_string())
    }
}

type Result<T> = std::result::Result<T, GgmlTierError>;

fn io<T>(r: std::io::Result<T>, what: impl std::fmt::Display) -> Result<T> {
    r.map_err(|source| GgmlTierError::Io {
        what: what.to_string(),
        source,
    })
}

/// What one component conversion did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ComponentReport {
    /// Every dtype cast a from-bf16 header-template build applied to a non-quantized tensor:
    /// `(tensor, source dtype, template dtype)`.
    pub casts: Vec<(String, String, String)>,
    /// Shards written.
    pub shards: usize,
    /// Projections replaced by stored GGML blocks.
    pub converted: usize,
    /// Tensors copied unchanged.
    pub passthrough: usize,
    /// Total tensor payload bytes written (the restated `metadata.total_size`).
    pub tensor_bytes: u64,
    /// Source tensor payload bytes read (template shards, plus dense-source tensors in from-bf16).
    pub bytes_in: u64,
    /// From-bf16 projections whose dense source was an **MXFP4** fused-expert pair
    /// (`{base}_blocks` + `{base}_scales`, the gpt-oss original the Lens bf16 tier ships) rather than
    /// a float `{base}.weight`: dequantized exactly, then `Q8_0`-quantized.
    pub mxfp4_sources: usize,
    /// Files copied byte-for-byte (a split tier's components that hold no packed triple).
    pub copied_files: usize,
}

/// An explicit tensor-name prefix rewrite between the MLX template and the dense source. Never
/// guessed: a template name that starts with `template` is looked up in the source as `source` +
/// the remainder; other names are looked up unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixMap {
    pub template: String,
    pub source: String,
}

impl PrefixMap {
    /// Parse `TEMPLATE_PREFIX=SOURCE_PREFIX` (either side may be empty).
    pub fn parse(s: &str) -> Result<Self> {
        let (template, source) = s.split_once('=').ok_or_else(|| {
            GgmlTierError::Spec(format!(
                "prefix map `{s}` must be TEMPLATE_PREFIX=SOURCE_PREFIX"
            ))
        })?;
        Ok(Self {
            template: template.to_string(),
            source: source.to_string(),
        })
    }

    fn apply(maps: &[PrefixMap], name: &str) -> String {
        for m in maps {
            if let Some(rest) = name.strip_prefix(&m.template) {
                return format!("{}{rest}", m.source);
            }
        }
        name.to_string()
    }
}

/// The dense source of a from-bf16 build.
#[derive(Clone, Debug)]
pub struct Bf16Source {
    /// Directory holding the dense `.safetensors` shards (any sharding).
    pub dir: PathBuf,
    /// Explicit template-name -> source-name prefix rewrites.
    pub prefix_maps: Vec<PrefixMap>,
}

/// Where a tier's quantized values come from.
#[derive(Clone, Debug)]
pub enum SourceMode {
    /// From the MLX triple (Q4 lossless; Q8 via dequant + requant).
    FromMlx,
    /// `Q8_0` from the dense source, MLX q8 component as the structural template.
    FromBf16(Bf16Source),
}

// ---------------------------------------------------------------------------------------------
// config.json
// ---------------------------------------------------------------------------------------------

/// First `quantization` object that carries `bits`, searched recursively: `(bits, group_size)`.
fn quant_params(value: &Value) -> Option<(u64, Option<u64>)> {
    match value {
        Value::Object(map) => {
            if let Some(Value::Object(q)) = map.get("quantization") {
                if let Some(bits) = q.get("bits").and_then(Value::as_u64) {
                    return Some((bits, q.get("group_size").and_then(Value::as_u64)));
                }
            }
            map.iter()
                .filter(|(k, _)| k.as_str() != "quantization")
                .find_map(|(_, v)| quant_params(v))
        }
        Value::Array(items) => items.iter().find_map(quant_params),
        _ => None,
    }
}

/// Rewrite every `quantization` object that carries `bits` into the GGML marker, recursively.
fn mark_ggml(value: &mut Value) -> usize {
    let mut n = 0;
    match value {
        Value::Object(map) => {
            if let Some(Value::Object(q)) = map.get_mut("quantization") {
                if let Some(bits) = q.get("bits").cloned() {
                    q.clear();
                    q.insert("bits".into(), bits);
                    q.insert("storage".into(), Value::from(GGML_STORAGE));
                    n += 1;
                }
            }
            for (key, child) in map.iter_mut() {
                if key != "quantization" {
                    n += mark_ggml(child);
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(|v| n += mark_ggml(v)),
        _ => {}
    }
    n
}

/// The rewritten `config.json` text for a GGML tier. Refuses a config with no packed marker: a
/// dense component has nothing to convert and must not be relabelled.
pub fn ggml_config(text: &str) -> Result<String> {
    let mut config: Value =
        serde_json::from_str(text).map_err(|e| GgmlTierError::Config(e.to_string()))?;
    if mark_ggml(&mut config) == 0 {
        return Err(GgmlTierError::Config(
            "carries no `quantization.bits` block - only a packed MLX tier converts".into(),
        ));
    }
    serde_json::to_string_pretty(&config).map_err(|e| GgmlTierError::Config(e.to_string()))
}

/// The rewritten `quantize_config.json` text for a GGML tier — the marker file the Wan candle tiers
/// (`{"bits": N, "quantization": {"group_size": 64}}`, top-level `bits`) and the LTX-2.3 split tiers
/// (`{"quantization": {"bits": N, "group_size": 64}}`) carry instead of a `config.json` block.
///
/// A top-level `bits` is kept (the Wan memory strategy reads it) and `quantization` becomes
/// `{"bits": N, "storage": "ggml"}`; the nested form is rewritten exactly as [`ggml_config`] does.
/// The group size is dropped either way: stored blocks carry no group, and every reader of these
/// files defaults an absent group to the MLX 64.
pub fn ggml_quantize_config(text: &str) -> Result<String> {
    let mut config: Value =
        serde_json::from_str(text).map_err(|e| GgmlTierError::Config(e.to_string()))?;
    let Some(bits) = config.get("bits").and_then(Value::as_u64) else {
        return ggml_config(text);
    };
    let map = config
        .as_object_mut()
        .ok_or_else(|| GgmlTierError::Config("quantize_config.json is not an object".into()))?;
    map.insert(
        "quantization".into(),
        serde_json::json!({ "bits": bits, "storage": GGML_STORAGE }),
    );
    serde_json::to_string_pretty(&config).map_err(|e| GgmlTierError::Config(e.to_string()))
}

/// `(bits, group_size)` a component config declares. `group_size` defaults to 64 (MLX's default).
fn config_quant(text: &str) -> Result<(u64, usize)> {
    let config: Value =
        serde_json::from_str(text).map_err(|e| GgmlTierError::Config(e.to_string()))?;
    let (bits, group) = quant_params(&config).ok_or_else(|| {
        GgmlTierError::Config(
            "carries no `quantization.bits` block - only a packed MLX tier converts".into(),
        )
    })?;
    Ok((bits, group.map_or(MLX_GROUP_SIZE, |g| g as usize)))
}

/// `(bits, group_size)` a `quantize_config.json` declares: top-level `bits` (Wan) or a nested
/// `quantization.bits` (LTX-2.3); the group from `quantization.group_size`, default 64.
fn quantize_config_quant(text: &str) -> Result<(u64, usize)> {
    let config: Value =
        serde_json::from_str(text).map_err(|e| GgmlTierError::Config(e.to_string()))?;
    if let Some(bits) = config.get("bits").and_then(Value::as_u64) {
        let group = config
            .get("quantization")
            .and_then(|q| q.get("group_size"))
            .and_then(Value::as_u64)
            .map_or(MLX_GROUP_SIZE, |g| g as usize);
        return Ok((bits, group));
    }
    config_quant(text)
}

/// Which file declares a component's quantization, and so which file carries the GGML marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Marker {
    /// `config.json` carries a `quantization` block with `bits` (the MLX tier convention).
    Config,
    /// `quantize_config.json` carries it (Wan candle tiers, LTX-2.3 split tiers).
    QuantizeConfig,
    /// No file declares it (the ideogram / SenseNova MLX converters write no block): the width is
    /// inferred from the packed shapes at the MLX default group 64 — the group those families'
    /// loaders hard-code — and must agree across every triple. `config.json` gains the GGML marker.
    Inferred,
}

const QUANTIZE_CONFIG: &str = "quantize_config.json";

/// Resolve a component's `(bits, group_size, marker)` (see [`Marker`]). `Inferred` returns bits 0;
/// the caller infers them from the template shards.
fn component_quant(src: &Path) -> Result<(u64, usize, Marker)> {
    let config_path = src.join("config.json");
    let qc_path = src.join(QUANTIZE_CONFIG);
    let config_text = if config_path.is_file() {
        Some(io(
            std::fs::read_to_string(&config_path),
            config_path.display(),
        )?)
    } else {
        None
    };
    if let Some(text) = &config_text {
        let value: Value =
            serde_json::from_str(text).map_err(|e| GgmlTierError::Config(e.to_string()))?;
        if quant_params(&value).is_some() {
            let (bits, group) = config_quant(text)?;
            return Ok((bits, group, Marker::Config));
        }
    }
    if qc_path.is_file() {
        let text = io(std::fs::read_to_string(&qc_path), qc_path.display())?;
        let (bits, group) = quantize_config_quant(&text)?;
        return Ok((bits, group, Marker::QuantizeConfig));
    }
    if config_text.is_some() {
        return Ok((0, MLX_GROUP_SIZE, Marker::Inferred));
    }
    Err(GgmlTierError::MissingConfig(src.to_path_buf()))
}

/// `config.json` text with the GGML marker inserted at the top level (an [`Marker::Inferred`]
/// component, whose config declared no quantization at all).
fn insert_ggml_marker(text: &str, bits: u64) -> Result<String> {
    let mut config: Value =
        serde_json::from_str(text).map_err(|e| GgmlTierError::Config(e.to_string()))?;
    config
        .as_object_mut()
        .ok_or_else(|| GgmlTierError::Config("config.json is not an object".into()))?
        .insert(
            "quantization".into(),
            serde_json::json!({ "bits": bits, "storage": GGML_STORAGE }),
        );
    serde_json::to_string_pretty(&config).map_err(|e| GgmlTierError::Config(e.to_string()))
}

// ---------------------------------------------------------------------------------------------
// Streaming safetensors reader (header + seek; never maps or loads a whole shard)
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    start: u64,
    end: u64,
}

impl Entry {
    fn len(&self) -> u64 {
        self.end - self.start
    }
}

/// Read just the safetensors header JSON bytes of `path` (the first `8 + N` bytes of the file).
fn read_header_bytes(path: &Path) -> Result<(Vec<u8>, u64)> {
    let mut f = io(std::fs::File::open(path), path.display())?;
    let file_len = io(f.metadata(), path.display())?.len();
    let mut len = [0u8; 8];
    io(f.read_exact(&mut len), path.display())?;
    let header_len = u64::from_le_bytes(len);
    if header_len == 0 || header_len > 256 * 1024 * 1024 || 8 + header_len > file_len {
        return Err(GgmlTierError::BadSafetensors {
            path: path.display().to_string(),
            detail: format!("implausible header length {header_len}"),
        });
    }
    let mut header = vec![0u8; header_len as usize];
    io(f.read_exact(&mut header), path.display())?;
    Ok((header, file_len))
}

/// One safetensors file: its parsed header. Tensor bytes are read on demand. A header-only
/// template shard (`path == None`) carries structure but no data.
#[derive(Debug)]
struct ShardFile {
    path: Option<PathBuf>,
    data_base: u64,
    entries: BTreeMap<String, Entry>,
    /// The header's `__metadata__`. A split tier keeps it on every component (the LTX-2.5
    /// `sceneworks_tier` / `model_version` / config stamps).
    metadata: Metadata,
}

/// A safetensors header's `__metadata__`, as far as this builder cares.
///
/// Only a split-tier conversion writes metadata back, so only it can be harmed by an unreadable
/// one; a component conversion ignores the field entirely (the slice-B behaviour), whatever it
/// holds.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
enum Metadata {
    /// No `__metadata__` key, or `"__metadata__": null` (MLX writes the latter, e.g. scail2's
    /// `dit.safetensors`): there is none, and none is written.
    #[default]
    Absent,
    /// A string -> string map (the safetensors spec form).
    Map(HashMap<String, String>),
    /// Anything else (a non-object, or a non-string value). Refused only where it would be
    /// written.
    Unreadable(String),
}

impl Metadata {
    fn map(&self) -> Option<&HashMap<String, String>> {
        match self {
            Metadata::Map(m) => Some(m),
            _ => None,
        }
    }
}

fn parse_metadata(v: &Value) -> Metadata {
    match v {
        Value::Null => Metadata::Absent,
        Value::Object(map) => {
            let mut out = HashMap::with_capacity(map.len());
            for (k, v) in map {
                match v.as_str() {
                    Some(s) => {
                        out.insert(k.clone(), s.to_string());
                    }
                    None => {
                        return Metadata::Unreadable(format!("`__metadata__.{k}` is not a string"))
                    }
                }
            }
            Metadata::Map(out)
        }
        other => Metadata::Unreadable(format!("`__metadata__` is {other}, not an object")),
    }
}

impl ShardFile {
    fn open(path: &Path) -> Result<Self> {
        let (header, file_len) = read_header_bytes(path)?;
        Self::parse(Some(path), &header, 8 + header.len() as u64, Some(file_len))
    }

    /// A header-only template shard from a dumped header JSON file.
    fn open_header_dump(path: &Path) -> Result<Self> {
        let header = io(std::fs::read(path), path.display())?;
        Self::parse(None, &header, 0, None).map_err(|e| match e {
            GgmlTierError::BadSafetensors { detail, .. } => GgmlTierError::BadSafetensors {
                path: path.display().to_string(),
                detail,
            },
            other => other,
        })
    }

    fn parse(
        path: Option<&Path>,
        header: &[u8],
        data_base: u64,
        file_len: Option<u64>,
    ) -> Result<Self> {
        let label = path.map_or_else(|| "<header dump>".to_string(), |p| p.display().to_string());
        let bad = |detail: String| GgmlTierError::BadSafetensors {
            path: label.clone(),
            detail,
        };
        let parsed: Value = serde_json::from_slice(header).map_err(|e| bad(e.to_string()))?;
        let obj = parsed
            .as_object()
            .ok_or_else(|| bad("header is not a JSON object".into()))?;
        let mut entries = BTreeMap::new();
        let mut metadata = Metadata::Absent;
        for (name, v) in obj {
            if name == "__metadata__" {
                metadata = parse_metadata(v);
                continue;
            }
            let dtype = v["dtype"]
                .as_str()
                .ok_or_else(|| bad(format!("`{name}` has no dtype")))?
                .to_string();
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .ok_or_else(|| bad(format!("`{name}` has no shape")))?
                .iter()
                .map(|d| d.as_u64().map(|d| d as usize))
                .collect::<Option<_>>()
                .ok_or_else(|| bad(format!("`{name}` has a bad shape")))?;
            let offs = v["data_offsets"]
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or_else(|| bad(format!("`{name}` has no data_offsets")))?;
            let (start, end) = (
                offs[0].as_u64().unwrap_or(u64::MAX),
                offs[1].as_u64().unwrap_or(0),
            );
            if end < start || file_len.is_some_and(|len| data_base + end > len) {
                return Err(bad(format!("`{name}` data offsets are outside the file")));
            }
            entries.insert(
                name.clone(),
                Entry {
                    dtype,
                    shape,
                    start,
                    end,
                },
            );
        }
        Ok(Self {
            path: path.map(Path::to_path_buf),
            data_base,
            entries,
            metadata,
        })
    }

    fn read_raw(&self, name: &str) -> Result<Vec<u8>> {
        let e = &self.entries[name];
        let path = self
            .path
            .as_ref()
            .ok_or_else(|| GgmlTierError::BadSafetensors {
                path: name.to_string(),
                detail: "header-only template shard carries no tensor data".into(),
            })?;
        let mut f = io(std::fs::File::open(path), path.display())?;
        io(
            f.seek(SeekFrom::Start(self.data_base + e.start)),
            path.display(),
        )?;
        let mut buf = vec![0u8; e.len() as usize];
        io(f.read_exact(&mut buf), path.display())?;
        Ok(buf)
    }

    fn read_tensor(&self, name: &str) -> Result<Tensor> {
        let e = &self.entries[name];
        let dtype = match e.dtype.as_str() {
            "U32" => DType::U32,
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            other => {
                return Err(GgmlTierError::DenseDtype {
                    tensor: name.to_string(),
                    dtype: other.to_string(),
                })
            }
        };
        let raw = self.read_raw(name)?;
        Ok(Tensor::from_raw_buffer(
            &raw,
            dtype,
            &e.shape,
            &Device::Cpu,
        )?)
    }
}

/// All shards of a dense-source directory, with a tensor-name index across them.
///
/// A name is looked up in the **same-named** shard first (a flat per-file layout such as scail2's
/// `dit.safetensors` / `clip.safetensors`, or an LTX split tier, holds the counterpart of template
/// file `F` in source file `F`, and different files may reuse a tensor name); otherwise the name
/// must be unique across the set (a re-sharded diffusers source). `same_file_only` (split tiers)
/// forbids the cross-file fallback entirely.
struct ShardSet {
    shards: Vec<(std::ffi::OsString, ShardFile)>,
    index: HashMap<String, Vec<usize>>,
    same_file_only: bool,
}

impl ShardSet {
    fn open(dir: &Path, same_file_only: bool) -> Result<Self> {
        let mut shards = Vec::new();
        let mut index: HashMap<String, Vec<usize>> = HashMap::new();
        for path in safetensors_files(dir)? {
            let shard = ShardFile::open(&path)?;
            for name in shard.entries.keys() {
                index.entry(name.clone()).or_default().push(shards.len());
            }
            let file = path.file_name().expect("shard file name").to_owned();
            shards.push((file, shard));
        }
        if shards.is_empty() {
            return Err(GgmlTierError::NoShards(dir.to_path_buf()));
        }
        Ok(Self {
            shards,
            index,
            same_file_only,
        })
    }

    /// The dense counterpart of `name` for template shard `file`.
    fn find(&self, name: &str, file: &std::ffi::OsStr) -> Result<Option<(&ShardFile, &Entry)>> {
        let Some(hits) = self.index.get(name) else {
            return Ok(None);
        };
        let same = hits.iter().find(|&&i| self.shards[i].0 == file);
        let pick = match (same, hits.as_slice()) {
            (Some(&i), _) => i,
            _ if self.same_file_only => return Ok(None),
            (None, [i]) => *i,
            (None, _) => return Err(GgmlTierError::DuplicateDenseTensor(name.to_string())),
        };
        let shard = &self.shards[pick].1;
        Ok(Some((shard, &shard.entries[name])))
    }
}

/// Suffix of a dumped safetensors header (`<shard>.safetensors` + this).
pub const HEADER_DUMP_SUFFIX: &str = ".header.json";

fn is_header_dump(p: &Path) -> bool {
    p.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.ends_with(".safetensors.header.json"))
}

fn is_shard(p: &Path) -> bool {
    p.extension().and_then(|s| s.to_str()) == Some("safetensors")
}

fn component_files(src: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = io(std::fs::read_dir(src), src.display())?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .filter(|p| {
            !p.file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n.starts_with('.'))
        })
        .collect();
    files.sort();
    Ok(files)
}

fn safetensors_files(dir: &Path) -> Result<Vec<PathBuf>> {
    Ok(component_files(dir)?
        .into_iter()
        .filter(|p| is_shard(p))
        .collect())
}

/// Build a header-only template directory `dst` from the component directory `src` (a local copy of
/// an MLX q8 component): copies `config.json`, the index and every other non-shard file, and writes
/// `<shard>.safetensors.header.json` holding only each shard's safetensors header (the first
/// `8 + N` bytes of the shard are read, nothing else). Returns the number of shards dumped.
pub fn dump_headers(src: &Path, dst: &Path) -> Result<usize> {
    io(std::fs::create_dir_all(dst), dst.display())?;
    let mut shards = 0;
    for file in component_files(src)? {
        let name = file.file_name().expect("file name").to_owned();
        if is_shard(&file) {
            let (header, _) = read_header_bytes(&file)?;
            write_header_dump(dst, &name.to_string_lossy(), &header)?;
            shards += 1;
        } else if !is_header_dump(&file) {
            io(std::fs::copy(&file, dst.join(&name)), file.display())?;
        }
    }
    Ok(shards)
}

/// Write one shard's raw safetensors header JSON as `<dst>/<shard_name>.header.json` (the format
/// [`convert_component`] reads for a header-only template). The bytes are validated as a header.
pub fn write_header_dump(dst: &Path, shard_name: &str, header: &[u8]) -> Result<()> {
    ShardFile::parse(None, header, 0, None)?;
    let target = dst.join(format!("{shard_name}{HEADER_DUMP_SUFFIX}"));
    io(std::fs::write(&target, header), target.display())
}

// ---------------------------------------------------------------------------------------------
// Quantization primitives
// ---------------------------------------------------------------------------------------------

/// Stored GGML block bytes plus their `[rows, cols / 32, block_bytes]` shape (rank-3), or
/// `[experts, rows, cols / 32, block_bytes]` for stacked experts (rank-4).
type Blocks = (Vec<u8>, Vec<usize>);

fn blocks_of(qt: &QTensor, tensor: &str) -> Result<Blocks> {
    let t =
        to_ggml_block_tensor(qt).map_err(|e| GgmlTierError::Candle(format!("{tensor}: {e}")))?;
    let shape = t.dims().to_vec();
    Ok((t.flatten_all()?.to_vec1::<u8>()?, shape))
}

/// The leading (expert) dim of a packed `[out, cols]` / `[experts, out, cols]` tensor and its
/// `(rows, cols)`. Rank 3 is the stacked-MoE layout (gpt-oss fused experts, Lens's text encoder).
fn split_lead(tensor: &str, dims: &[usize]) -> Result<(Option<usize>, usize, usize)> {
    match *dims {
        [rows, cols] => Ok((None, rows, cols)),
        [experts, rows, cols] => Ok((Some(experts), rows, cols)),
        _ => Err(GgmlTierError::UnsupportedRank {
            tensor: tensor.to_string(),
            rank: dims.len(),
        }),
    }
}

/// Convert one MLX triple into stored GGML blocks. `cfg_bits` (from the component config), when
/// given, must agree with the width the packed shapes imply at `group_size`.
///
/// A rank-3 triple (`[E, out, in_packed]` codes + `[E, out, in / group]` scales/biases, the stacked
/// MoE experts) becomes ONE rank-4 stored tensor `[E, out, in / 32, block_bytes]`: GGML quantizes
/// each block on its own, so the stacked blocks are byte-for-byte the per-expert repacks stacked —
/// what the Lens text encoder's `stored_experts` slices per expert.
fn blocks_from_mlx(
    base: &str,
    wq: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: usize,
    cfg_bits: Option<u64>,
) -> Result<Blocks> {
    if !group_size.is_multiple_of(GGML_BLOCK) {
        return Err(GgmlTierError::GroupSize { group_size });
    }
    let weight = format!("{base}.weight");
    if wq.dtype() != DType::U32 {
        return Err(GgmlTierError::NotPackedCodes {
            tensor: weight,
            dtype: format!("{:?}", wq.dtype()),
        });
    }
    let (lead, rows, wq_cols) = split_lead(&weight, wq.dims())?;
    let s_dims = scales.dims();
    let lead_ok = s_dims.len() == wq.rank()
        && s_dims[..s_dims.len() - 1] == wq.dims()[..wq.rank() - 1]
        && biases.dims() == s_dims;
    if !lead_ok {
        return Err(GgmlTierError::TripleShape {
            tensor: base.to_string(),
            detail: format!(
                "weight {:?}, scales {s_dims:?}, biases {:?}",
                wq.dims(),
                biases.dims()
            ),
        });
    }
    let s_cols = s_dims[s_dims.len() - 1];
    if s_cols == 0 {
        return Err(GgmlTierError::OrphanTriple {
            tensor: base.to_string(),
            detail: "scales have no groups".into(),
        });
    }
    let inferred = mlx_packed_bits_gs(wq_cols, s_cols, group_size);
    if let Some(expected) = cfg_bits {
        if inferred as u64 != expected {
            return Err(GgmlTierError::BitsMismatch {
                tensor: base.to_string(),
                inferred,
                expected: expected as usize,
                group_size,
            });
        }
    }
    let flat_rows = lead.unwrap_or(1) * rows;
    // The repack receives scales/biases at f32; widening bf16/f16 is exact and the block fields are
    // f16 either way. A stacked tensor repacks as `[E * out, ..]` rows (row-independent blocks).
    let (wq, scales, biases) = if lead.is_some() {
        (
            wq.reshape((flat_rows, wq_cols))?,
            scales.reshape((flat_rows, s_cols))?,
            biases.reshape((flat_rows, s_cols))?,
        )
    } else {
        (wq.clone(), scales.clone(), biases.clone())
    };
    let scales = scales.to_dtype(DType::F32)?;
    let biases = biases.to_dtype(DType::F32)?;
    let qt = repack_packed_weight_uncached(&wq, &scales, &biases, group_size, &Device::Cpu)?;
    let (bytes, mut shape) = blocks_of(&qt, base)?;
    if let Some(experts) = lead {
        shape = vec![experts, rows, shape[1], shape[2]];
    }
    Ok((bytes, shape))
}

/// Quantize a dense `[out, in]` (or stacked `[E, out, in]`) weight straight to `Q8_0` blocks.
/// `expected` is the shape the MLX template's triple implies. A stacked weight is quantized one
/// expert at a time (bounded f32 working copy) and stored `[E, out, in / 32, 34]`.
fn blocks_q8_from_dense(base: &str, dense: &Tensor, expected: &[usize]) -> Result<Blocks> {
    let tensor = format!("{base}.weight");
    if !matches!(dense.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        return Err(GgmlTierError::DenseDtype {
            tensor,
            dtype: format!("{:?}", dense.dtype()),
        });
    }
    if dense.dims() != expected {
        return Err(GgmlTierError::DenseShapeMismatch {
            tensor,
            expected: expected.to_vec(),
            found: dense.dims().to_vec(),
        });
    }
    let (lead, rows, in_features) = split_lead(&tensor, expected)?;
    if !in_features.is_multiple_of(GGML_BLOCK) {
        return Err(GgmlTierError::InFeaturesNotBlockMultiple {
            tensor,
            in_features,
        });
    }
    let Some(experts) = lead else {
        let f32w = dense.to_dtype(DType::F32)?.contiguous()?;
        let qt = QTensor::quantize(&f32w, GgmlDType::Q8_0)?;
        return blocks_of(&qt, base);
    };
    let mut bytes = Vec::new();
    for x in 0..experts {
        let f32w = dense.get(x)?.to_dtype(DType::F32)?.contiguous()?;
        let (b, _) = blocks_of(&QTensor::quantize(&f32w, GgmlDType::Q8_0)?, base)?;
        bytes.extend_from_slice(&b);
    }
    let block_bytes = GgmlDType::Q8_0.type_size();
    Ok((
        bytes,
        vec![experts, rows, in_features / GGML_BLOCK, block_bytes],
    ))
}

/// FP4 (e2m1) code -> value, codes 0..7 positive, 8..15 negative (the OCP MXFP4 grid; the same
/// table the Lens text encoder's `dequant_mxfp4` decodes with).
const FP4_E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// MXFP4 bytes per 32-value block (two e2m1 codes per byte).
const MXFP4_BLOCK_BYTES: usize = 16;

/// `Q8_0` blocks from an **MXFP4** dense source — `{base}_blocks` (`U8 [.., out, in / 32, 16]`,
/// low nibble first) + `{base}_scales` (`U8 [.., out, in / 32]`, e8m0: `2^(s - 127)`), the
/// gpt-oss fused-expert form the Lens bf16 tier ships for its text-encoder experts. Each matrix
/// (each expert of a stacked tensor) is dequantized exactly to f32 and quantized with candle's own
/// `Q8_0`; `expected` is the `[out, in]` / `[E, out, in]` the template triple implies.
fn blocks_q8_from_mxfp4(
    base: &str,
    blocks: (&Entry, Vec<u8>),
    scales: (&Entry, Vec<u8>),
    expected: &[usize],
) -> Result<Blocks> {
    let tensor = format!("{base}_blocks");
    let (lead, rows, in_features) = split_lead(&tensor, expected)?;
    if !in_features.is_multiple_of(GGML_BLOCK) {
        return Err(GgmlTierError::InFeaturesNotBlockMultiple {
            tensor,
            in_features,
        });
    }
    let nb = in_features / GGML_BLOCK;
    let outer = &expected[..expected.len() - 1];
    let want_blocks: Vec<usize> = outer
        .iter()
        .copied()
        .chain([nb, MXFP4_BLOCK_BYTES])
        .collect();
    let want_scales: Vec<usize> = outer.iter().copied().chain([nb]).collect();
    for (name, entry, want) in [
        (&tensor, blocks.0, &want_blocks),
        (&format!("{base}_scales"), scales.0, &want_scales),
    ] {
        if entry.dtype != "U8" || entry.shape != *want {
            return Err(GgmlTierError::DenseShapeMismatch {
                tensor: format!("{name} ({})", entry.dtype),
                expected: want.clone(),
                found: entry.shape.clone(),
            });
        }
    }
    let (blk, scl) = (blocks.1, scales.1);
    let matrices = lead.unwrap_or(1);
    let mut bytes = Vec::new();
    let mut grid = vec![0f32; rows * in_features];
    for m in 0..matrices {
        for row in 0..rows {
            for b in 0..nb {
                let at = (m * rows + row) * nb + b;
                let scale = 2f32.powi(scl[at] as i32 - 127);
                let src = &blk[at * MXFP4_BLOCK_BYTES..(at + 1) * MXFP4_BLOCK_BYTES];
                let dst = &mut grid[row * in_features + b * GGML_BLOCK..][..GGML_BLOCK];
                for (j, byte) in src.iter().enumerate() {
                    dst[2 * j] = FP4_E2M1[(byte & 0x0f) as usize] * scale;
                    dst[2 * j + 1] = FP4_E2M1[(byte >> 4) as usize] * scale;
                }
            }
        }
        let t = Tensor::from_slice(&grid, (rows, in_features), &Device::Cpu)?;
        let (b, _) = blocks_of(&QTensor::quantize(&t, GgmlDType::Q8_0)?, base)?;
        bytes.extend_from_slice(&b);
    }
    let block_bytes = GgmlDType::Q8_0.type_size();
    let shape = match lead {
        Some(e) => vec![e, rows, nb, block_bytes],
        None => vec![rows, nb, block_bytes],
    };
    Ok((bytes, shape))
}

/// Stored GGML blocks of one MLX affine triple, as a `U8` tensor: `[out, in / 32, block_bytes]`
/// (`Q4_1` from q4, lossless; `Q8_0` from q8 via dequant + requant), or `[E, out, in / 32,
/// block_bytes]` for a rank-3 stacked-expert triple. The bit width is inferred from the shapes at
/// `group_size`. The one MLX-triple -> stored-block conversion: [`convert_component`] and the
/// family tier builders (e.g. `candle-gen-wan`'s) both call it.
pub fn mlx_triple_to_ggml_blocks(
    base: &str,
    wq: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: usize,
) -> Result<Tensor> {
    let (bytes, shape) = blocks_from_mlx(base, wq, scales, biases, group_size, None)?;
    Ok(Tensor::from_vec(bytes, shape, &Device::Cpu)?)
}

/// `Q8_0` stored blocks of a dense bf16/f16/f32 weight `[out, in]` (or stacked `[E, out, in]`) as a
/// `U8` tensor `[out, in / 32, 34]` (`[E, out, in / 32, 34]`), quantized with candle's own `Q8_0` —
/// the one dense -> `Q8_0` conversion the from-bf16 builds and the family tier builders share.
pub fn dense_to_q8_0_blocks(base: &str, dense: &Tensor) -> Result<Tensor> {
    let (bytes, shape) = blocks_q8_from_dense(base, dense, dense.dims())?;
    Ok(Tensor::from_vec(bytes, shape, &Device::Cpu)?)
}

// ---------------------------------------------------------------------------------------------
// In-memory conversion (kept for the per-crate fixtures and tests)
// ---------------------------------------------------------------------------------------------

/// Convert one shard's tensors held in memory: each MLX triple becomes stored GGML blocks (at
/// `group_size`), everything else passes through. `report` is updated in place.
pub fn convert_tensors(
    tensors: HashMap<String, Tensor>,
    group_size: usize,
    report: &mut ComponentReport,
) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::with_capacity(tensors.len());
    for (name, t) in &tensors {
        if let Some(base) = name
            .strip_suffix(".scales")
            .or_else(|| name.strip_suffix(".biases"))
        {
            if !tensors.contains_key(&format!("{base}.weight")) {
                return Err(GgmlTierError::OrphanTriple {
                    tensor: name.clone(),
                    detail: format!("has no `{base}.weight` in its shard"),
                });
            }
            continue;
        }
        let packed = name.strip_suffix(".weight").and_then(|base| {
            let scales = tensors.get(&format!("{base}.scales"))?;
            Some((base, scales))
        });
        let Some((base, scales)) = packed else {
            out.insert(name.clone(), t.clone());
            report.passthrough += 1;
            continue;
        };
        let biases =
            tensors
                .get(&format!("{base}.biases"))
                .ok_or_else(|| GgmlTierError::OrphanTriple {
                    tensor: format!("{base}.scales"),
                    detail: "has no `.biases`".into(),
                })?;
        let (bytes, shape) = blocks_from_mlx(base, t, scales, biases, group_size, None)?;
        out.insert(name.clone(), Tensor::from_vec(bytes, shape, &Device::Cpu)?);
        report.converted += 1;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Streaming component conversion
// ---------------------------------------------------------------------------------------------

struct OutTensor {
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl safetensors::View for OutTensor {
    fn dtype(&self) -> safetensors::Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Borrowed(&self.data)
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn st_dtype(s: &str, tensor: &str) -> Result<safetensors::Dtype> {
    serde_json::from_value(Value::String(s.to_string())).map_err(|_| {
        GgmlTierError::BadSafetensors {
            path: tensor.to_string(),
            detail: format!("unknown dtype `{s}`"),
        }
    })
}

struct Ctx<'a> {
    mode: &'a SourceMode,
    dense: Option<ShardSet>,
    group_size: usize,
    cfg_bits: u64,
    /// Write each template shard's `__metadata__` into its output shard (split tiers: the
    /// LTX-2.5 per-component stamps). Component tiers keep the slice-B output (no metadata).
    keep_metadata: bool,
}

/// Little-endian bytes of a float tensor at `dtype` (F32/F16/BF16).
fn float_le_bytes(t: &Tensor, dtype: DType) -> Result<Vec<u8>> {
    let flat = t.to_dtype(dtype)?.flatten_all()?;
    Ok(match dtype {
        DType::F32 => flat
            .to_vec1::<f32>()?
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
        DType::F16 => flat
            .to_vec1::<half::f16>()?
            .into_iter()
            .flat_map(half::f16::to_le_bytes)
            .collect(),
        DType::BF16 => flat
            .to_vec1::<half::bf16>()?
            .into_iter()
            .flat_map(half::bf16::to_le_bytes)
            .collect(),
        other => {
            return Err(GgmlTierError::DenseDtype {
                tensor: String::new(),
                dtype: format!("{other:?}"),
            })
        }
    })
}

/// A non-quantized tensor of a from-bf16 header template, taken from the dense source: shape must
/// match the template, a dtype difference is cast to the template's dtype and reported.
fn dense_passthrough(
    name: &str,
    entry: &Entry,
    file: &std::ffi::OsStr,
    src: &Bf16Source,
    dense: &ShardSet,
    report: &mut ComponentReport,
) -> Result<OutTensor> {
    let looked_up = PrefixMap::apply(&src.prefix_maps, name);
    let (dshard, dentry) =
        dense
            .find(&looked_up, file)?
            .ok_or_else(|| GgmlTierError::MissingDenseSource {
                tensor: name.to_string(),
                looked_up: looked_up.clone(),
            })?;
    if dentry.shape != entry.shape {
        return Err(GgmlTierError::DenseShapeMismatch {
            tensor: name.to_string(),
            expected: entry.shape.clone(),
            found: dentry.shape.clone(),
        });
    }
    report.bytes_in += dentry.len();
    let dtype = st_dtype(&entry.dtype, name)?;
    if dentry.dtype == entry.dtype {
        return Ok(OutTensor {
            dtype,
            shape: entry.shape.clone(),
            data: dshard.read_raw(&looked_up)?,
        });
    }
    let target = match entry.dtype.as_str() {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        _ => {
            return Err(GgmlTierError::DenseDtype {
                tensor: name.to_string(),
                dtype: format!("{} (template wants {})", dentry.dtype, entry.dtype),
            })
        }
    };
    let t = dshard.read_tensor(&looked_up)?;
    let data = float_le_bytes(&t, target)?;
    report
        .casts
        .push((name.to_string(), dentry.dtype.clone(), entry.dtype.clone()));
    Ok(OutTensor {
        dtype,
        shape: entry.shape.clone(),
        data,
    })
}

/// The `Q8_0` blocks of template triple `base` (`entry` = its `.weight`) from the dense source:
/// its float `{base}.weight`, or — when the source carries none — an MXFP4 `{base}_blocks` +
/// `{base}_scales` pair (the Lens / gpt-oss fused experts).
fn q8_from_dense_source(
    base: &str,
    name: &str,
    entry: &Entry,
    s_shape: &[usize],
    file: &std::ffi::OsStr,
    ctx: &Ctx,
    report: &mut ComponentReport,
) -> Result<Blocks> {
    let SourceMode::FromBf16(src) = ctx.mode else {
        unreachable!("q8_from_dense_source runs only in from-bf16 mode");
    };
    // Structure only: the template's shapes, never its data.
    if entry.dtype != "U32" {
        return Err(GgmlTierError::NotPackedCodes {
            tensor: name.to_string(),
            dtype: entry.dtype.clone(),
        });
    }
    let (_, _, wq_cols) = split_lead(name, &entry.shape)?;
    let shapes_agree = s_shape.len() == entry.shape.len()
        && s_shape[..s_shape.len() - 1] == entry.shape[..entry.shape.len() - 1]
        && s_shape[s_shape.len() - 1] > 0;
    if !shapes_agree {
        return Err(GgmlTierError::OrphanTriple {
            tensor: format!("{base}.scales"),
            detail: format!("scales shape {s_shape:?} does not match `{name}`"),
        });
    }
    let s_cols = s_shape[s_shape.len() - 1];
    let inferred = mlx_packed_bits_gs(wq_cols, s_cols, ctx.group_size);
    if inferred != 8 {
        return Err(GgmlTierError::BitsMismatch {
            tensor: base.to_string(),
            inferred,
            expected: 8,
            group_size: ctx.group_size,
        });
    }
    let mut expected = entry.shape.clone();
    *expected.last_mut().expect("rank checked") = s_cols * ctx.group_size;
    let mapped = PrefixMap::apply(&src.prefix_maps, base);
    let looked_up = format!("{mapped}.weight");
    let dense_set = ctx.dense.as_ref().expect("dense source opened");
    if let Some((dshard, dentry)) = dense_set.find(&looked_up, file)? {
        let dense = dshard.read_tensor(&looked_up)?;
        report.bytes_in += dentry.len();
        return blocks_q8_from_dense(base, &dense, &expected);
    }
    let (b_name, s_name) = (format!("{mapped}_blocks"), format!("{mapped}_scales"));
    match (
        dense_set.find(&b_name, file)?,
        dense_set.find(&s_name, file)?,
    ) {
        (Some((bshard, bentry)), Some((sshard, sentry))) => {
            report.bytes_in += bentry.len() + sentry.len();
            report.mxfp4_sources += 1;
            blocks_q8_from_mxfp4(
                base,
                (bentry, bshard.read_raw(&b_name)?),
                (sentry, sshard.read_raw(&s_name)?),
                &expected,
            )
        }
        _ => Err(GgmlTierError::MissingDenseSource {
            tensor: name.to_string(),
            looked_up,
        }),
    }
}

fn convert_shard(
    shard: &ShardFile,
    file: &std::ffi::OsStr,
    ctx: &Ctx,
    report: &mut ComponentReport,
) -> Result<BTreeMap<String, OutTensor>> {
    let mut out = BTreeMap::new();
    for (name, entry) in &shard.entries {
        if let Some(base) = name
            .strip_suffix(".scales")
            .or_else(|| name.strip_suffix(".biases"))
        {
            if !shard.entries.contains_key(&format!("{base}.weight")) {
                return Err(GgmlTierError::OrphanTriple {
                    tensor: name.clone(),
                    detail: format!("has no `{base}.weight` in its shard"),
                });
            }
            continue;
        }
        let base = name
            .strip_suffix(".weight")
            .filter(|b| shard.entries.contains_key(&format!("{b}.scales")));
        let Some(base) = base else {
            let t = match (ctx.mode, shard.path.is_some()) {
                (SourceMode::FromBf16(src), false) => dense_passthrough(
                    name,
                    entry,
                    file,
                    src,
                    ctx.dense.as_ref().expect("dense source opened"),
                    report,
                )?,
                _ => {
                    report.bytes_in += entry.len();
                    OutTensor {
                        dtype: st_dtype(&entry.dtype, name)?,
                        shape: entry.shape.clone(),
                        data: shard.read_raw(name)?,
                    }
                }
            };
            out.insert(name.clone(), t);
            report.passthrough += 1;
            continue;
        };
        let (s_name, b_name) = (format!("{base}.scales"), format!("{base}.biases"));
        if !shard.entries.contains_key(&b_name) {
            return Err(GgmlTierError::OrphanTriple {
                tensor: s_name,
                detail: "has no `.biases`".into(),
            });
        }
        let blocks = match ctx.mode {
            SourceMode::FromMlx => {
                let wq = shard.read_tensor(name)?;
                let scales = shard.read_tensor(&s_name)?;
                let biases = shard.read_tensor(&b_name)?;
                report.bytes_in +=
                    entry.len() + shard.entries[&s_name].len() + shard.entries[&b_name].len();
                blocks_from_mlx(
                    base,
                    &wq,
                    &scales,
                    &biases,
                    ctx.group_size,
                    Some(ctx.cfg_bits),
                )?
            }
            SourceMode::FromBf16(_) => q8_from_dense_source(
                base,
                name,
                entry,
                &shard.entries[&s_name].shape,
                file,
                ctx,
                report,
            )?,
        };
        out.insert(
            name.clone(),
            OutTensor {
                dtype: safetensors::Dtype::U8,
                shape: blocks.1,
                data: blocks.0,
            },
        );
        report.converted += 1;
    }
    Ok(out)
}

fn partial_path(dst: &Path) -> PathBuf {
    let mut name = dst.file_name().unwrap_or_default().to_os_string();
    name.push(".partial");
    dst.with_file_name(name)
}

/// The template shards of `files`: real `.safetensors`, or — for from-bf16 only, when there are
/// none — header-only `<shard>.safetensors.header.json` dumps (see [`dump_headers`]).
fn template_shards(
    files: &[PathBuf],
    mode: &SourceMode,
) -> Result<Vec<(std::ffi::OsString, ShardFile)>> {
    let mut template = Vec::new();
    for p in files.iter().filter(|p| is_shard(p)) {
        let name = p.file_name().expect("shard file name").to_owned();
        template.push((name, ShardFile::open(p)?));
    }
    if template.is_empty() && matches!(mode, SourceMode::FromBf16(_)) {
        for p in files.iter().filter(|p| is_header_dump(p)) {
            let name = p.file_name().expect("header dump name").to_string_lossy();
            let shard_name = name
                .strip_suffix(HEADER_DUMP_SUFFIX)
                .expect("suffix checked");
            template.push((shard_name.into(), ShardFile::open_header_dump(p)?));
        }
    }
    Ok(template)
}

/// The width every packed triple of `template` implies at `group_size`, for a component that
/// declares none ([`Marker::Inferred`]). All triples must agree, and on 4 or 8.
fn infer_bits(
    src: &Path,
    template: &[(std::ffi::OsString, ShardFile)],
    group_size: usize,
) -> Result<u64> {
    let mut first: Option<(String, usize)> = None;
    for (_, shard) in template {
        for (name, entry) in &shard.entries {
            let Some(base) = name.strip_suffix(".scales") else {
                continue;
            };
            let Some(w) = shard.entries.get(&format!("{base}.weight")) else {
                continue; // reported as an orphan by the conversion itself
            };
            let (Some(&wq_cols), Some(&s_cols)) = (w.shape.last(), entry.shape.last()) else {
                continue;
            };
            if s_cols == 0 {
                continue;
            }
            let bits = mlx_packed_bits_gs(wq_cols, s_cols, group_size);
            match &first {
                None => first = Some((base.to_string(), bits)),
                Some((f, fb)) if *fb != bits => {
                    return Err(GgmlTierError::InferredBits {
                        dir: src.to_path_buf(),
                        tensor: base.to_string(),
                        bits,
                        first: f.clone(),
                        first_bits: *fb,
                    })
                }
                Some(_) => {}
            }
        }
    }
    match first {
        Some((_, bits @ (4 | 8))) => Ok(bits as u64),
        Some((tensor, bits)) => Err(GgmlTierError::InferredBits {
            dir: src.to_path_buf(),
            tensor: tensor.clone(),
            bits,
            first: tensor,
            first_bits: bits,
        }),
        None => Err(GgmlTierError::Config(format!(
            "{} declares no quantization and holds no packed triple - only a packed MLX tier \
             converts",
            src.display()
        ))),
    }
}

/// Convert the MLX-tier component directory `src` into a GGML-tier component at `dst`.
///
/// `dst` must not exist. `progress` is called after each shard with that shard's output path and
/// the running report. See the module docs for the exact transformation.
pub fn convert_component(
    src: &Path,
    dst: &Path,
    mode: &SourceMode,
    progress: &mut dyn FnMut(&Path, &ComponentReport) -> Result<()>,
) -> Result<ComponentReport> {
    if dst.exists() {
        return Err(GgmlTierError::DestinationExists(dst.to_path_buf()));
    }
    let (declared_bits, group_size, marker) = component_quant(src)?;
    let files = component_files(src)?;
    // Full-tier template: real shards. From-bf16 may instead use a header-only template, one
    // `<shard>.safetensors.header.json` dump per shard (see `dump_headers`).
    let template = template_shards(&files, mode)?;
    if template.is_empty() {
        return Err(GgmlTierError::NoShards(src.to_path_buf()));
    }
    let cfg_bits = match marker {
        Marker::Inferred => infer_bits(src, &template, group_size)?,
        _ => declared_bits,
    };
    if matches!(mode, SourceMode::FromBf16(_)) && cfg_bits != 8 {
        return Err(GgmlTierError::TemplateNotQ8(cfg_bits));
    }
    if matches!(mode, SourceMode::FromMlx) && !group_size.is_multiple_of(GGML_BLOCK) {
        return Err(GgmlTierError::GroupSize { group_size });
    }
    let ctx = Ctx {
        mode,
        dense: match mode {
            SourceMode::FromBf16(s) => Some(ShardSet::open(&s.dir, false)?),
            SourceMode::FromMlx => None,
        },
        group_size,
        cfg_bits,
        keep_metadata: false,
    };

    let partial = partial_path(dst);
    if partial.exists() {
        io(std::fs::remove_dir_all(&partial), partial.display())?;
    }
    io(std::fs::create_dir_all(&partial), partial.display())?;

    let mut report = ComponentReport::default();
    let mut weight_map = serde_json::Map::new();
    for (name, shard) in &template {
        let name = name.as_os_str();
        let (target, keys) = write_shard(shard, name, &partial, &ctx, &mut report)?;
        for key in keys {
            weight_map.insert(key, Value::from(name.to_string_lossy().into_owned()));
        }
        progress(&target, &report)?;
    }

    for file in files.iter().filter(|p| !is_shard(p) && !is_header_dump(p)) {
        let name = file.file_name().expect("sidecar file name");
        let target = partial.join(name);
        let text_name = name.to_string_lossy();
        if text_name == "config.json" && marker != Marker::QuantizeConfig {
            let text = io(std::fs::read_to_string(file), file.display())?;
            let rewritten = match marker {
                Marker::Config => ggml_config(&text)?,
                _ => insert_ggml_marker(&text, cfg_bits)?,
            };
            io(std::fs::write(&target, rewritten), target.display())?;
        } else if text_name == QUANTIZE_CONFIG && marker == Marker::QuantizeConfig {
            let text = io(std::fs::read_to_string(file), file.display())?;
            io(
                std::fs::write(&target, ggml_quantize_config(&text)?),
                target.display(),
            )?;
        } else if text_name.ends_with(".index.json") {
            let text = io(std::fs::read_to_string(file), file.display())?;
            let mut index: Value = serde_json::from_str(&text)
                .map_err(|e| GgmlTierError::Config(format!("{text_name}: {e}")))?;
            let mut keys: Vec<_> = weight_map.keys().cloned().collect();
            keys.sort();
            let sorted: serde_json::Map<String, Value> = keys
                .into_iter()
                .map(|k| {
                    let v = weight_map[&k].clone();
                    (k, v)
                })
                .collect();
            index["weight_map"] = Value::Object(sorted);
            index["metadata"] = serde_json::json!({ "total_size": report.tensor_bytes });
            let text = serde_json::to_string_pretty(&index)
                .map_err(|e| GgmlTierError::Config(format!("{text_name}: {e}")))?;
            io(std::fs::write(&target, text), target.display())?;
        } else {
            io(std::fs::copy(file, &target), file.display())?;
        }
    }
    io(std::fs::rename(&partial, dst), dst.display())?;
    Ok(report)
}

/// Convert one template shard and write it to `<dir>/<name>`; returns the path and the names of
/// the tensors written.
fn write_shard(
    shard: &ShardFile,
    name: &std::ffi::OsStr,
    dir: &Path,
    ctx: &Ctx,
    report: &mut ComponentReport,
) -> Result<(PathBuf, Vec<String>)> {
    let out = convert_shard(shard, name, ctx, report)?;
    let keys: Vec<String> = out.keys().cloned().collect();
    for t in out.values() {
        report.tensor_bytes += t.data.len() as u64;
    }
    let target = dir.join(name);
    let metadata = if ctx.keep_metadata {
        if let Metadata::Unreadable(detail) = &shard.metadata {
            return Err(GgmlTierError::BadSafetensors {
                path: target.display().to_string(),
                detail: format!("template {detail}; a split tier keeps each file's metadata"),
            });
        }
        shard.metadata.map().cloned()
    } else {
        None
    };
    safetensors::serialize_to_file(out, metadata, &target)
        .map_err(|e| GgmlTierError::Candle(format!("{}: {e}", target.display())))?;
    report.shards += 1;
    Ok((target, keys))
}

// ---------------------------------------------------------------------------------------------
// Split tiers (LTX-2.3 / LTX-2.5 `split_model.json` flat directories)
// ---------------------------------------------------------------------------------------------

/// The split-tier manifest file (`split_model.json`, `format: "split"`).
pub const SPLIT_MANIFEST: &str = "split_model.json";

/// The top-level `split_model.json` key a GGML split tier carries: `"quantization_storage":
/// "ggml"` — every packed Linear of the tier is a stored GGML block tensor, no MLX triple remains.
pub const SPLIT_STORAGE_KEY: &str = "quantization_storage";

/// What one file of a split tier became.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SplitFileStats {
    /// Tensors in the written file.
    pub tensors: usize,
    /// Projections stored as GGML blocks in it.
    pub converted: usize,
    /// The written file's size in bytes.
    pub bytes: u64,
}

fn split_err(dir: &Path, detail: impl Into<String>) -> GgmlTierError {
    GgmlTierError::Split {
        dir: dir.to_path_buf(),
        detail: detail.into(),
    }
}

/// The rewritten `split_model.json` for a GGML split tier.
///
/// * `quantization_storage: "ggml"` is added; `quantized`, `tier`, `quantization_bits` and
///   `quantization_group_size` are kept (the tier stamps, the memory strategy and the 2.5
///   validator's group-64 pin all still read them);
/// * every `component_detail` row is restated for the file it names: `tensors` (the `.scales` /
///   `.biases` legs are gone) and `bytes` from `files`; `quantized_linears` must equal the
///   projections stored as blocks in that file (a disagreement is refused, it is the
///   whole-pipeline tier contract); `total_bytes` is restated as the rows' sum;
/// * a manifest without rows (the LTX-2.3 form) only gains the marker.
pub fn ggml_split_manifest(
    dir: &Path,
    text: &str,
    files: &BTreeMap<String, SplitFileStats>,
) -> Result<String> {
    let mut manifest: Value =
        serde_json::from_str(text).map_err(|e| split_err(dir, format!("{SPLIT_MANIFEST}: {e}")))?;
    if manifest.get("format").and_then(Value::as_str) != Some("split") {
        return Err(split_err(
            dir,
            format!("{SPLIT_MANIFEST} is not `format: \"split\"`"),
        ));
    }
    if let Some(rows) = manifest
        .get_mut("component_detail")
        .and_then(Value::as_array_mut)
    {
        let mut total = 0u64;
        for row in rows.iter_mut() {
            let file = row
                .get("file")
                .and_then(Value::as_str)
                .ok_or_else(|| split_err(dir, "a component_detail row has no `file`"))?
                .to_string();
            let stats = files.get(&file).ok_or_else(|| {
                split_err(dir, format!("manifest row names `{file}`, not written"))
            })?;
            if let Some(declared) = row.get("quantized_linears").and_then(Value::as_u64) {
                if declared != stats.converted as u64 {
                    return Err(split_err(
                        dir,
                        format!(
                            "`{file}` declares {declared} quantized_linears but holds {} packed \
                             triples",
                            stats.converted
                        ),
                    ));
                }
            }
            let row = row
                .as_object_mut()
                .ok_or_else(|| split_err(dir, "a component_detail row is not an object"))?;
            if row.contains_key("tensors") {
                row.insert("tensors".into(), Value::from(stats.tensors as u64));
            }
            if row.contains_key("bytes") {
                row.insert("bytes".into(), Value::from(stats.bytes));
            }
            total += stats.bytes;
        }
        if manifest.get("total_bytes").is_some() {
            manifest["total_bytes"] = Value::from(total);
        }
    }
    manifest
        .as_object_mut()
        .ok_or_else(|| split_err(dir, format!("{SPLIT_MANIFEST} is not an object")))?
        .insert(SPLIT_STORAGE_KEY.into(), Value::from(GGML_STORAGE));
    serde_json::to_string_pretty(&manifest)
        .map_err(|e| split_err(dir, format!("{SPLIT_MANIFEST}: {e}")))
}

/// Convert a **split tier** directory `src` — a flat directory of per-component `.safetensors`
/// files described by `split_model.json` (`format: "split"`: the `SceneWorks/ltx-2.3-mlx` and
/// `ltx-2.5-mlx` `q4` / `q8` tiers) — into a GGML split tier at `dst`.
///
/// * file names are kept; a file holding packed triples is converted (each triple -> one stored
///   GGML block tensor, every other tensor kept) and **keeps its `__metadata__`** (the
///   `sceneworks_tier` / `model_version` / config stamps the LTX-2.5 validator checks); a file with
///   no packed triple is copied byte-for-byte;
/// * the geometry comes from the manifest (`quantization_bits`, `quantization_group_size`), which
///   must declare `quantized: true`;
/// * `split_model.json` is rewritten by [`ggml_split_manifest`]; an LTX-2.3 `quantize_config.json`
///   by [`ggml_quantize_config`]; every other sidecar (`embedded_config.json`, `config.json`) is
///   copied verbatim;
/// * from-bf16: `Bf16Source::dir` is the **bf16 split tier directory itself** (same file names);
///   each tensor is read from the same-named dense file only (component files reuse tensor names),
///   and a header-only template (see [`dump_headers`]) rebuilds even the dense-only files from it,
///   under the template's own `__metadata__`.
pub fn convert_split_tier(
    src: &Path,
    dst: &Path,
    mode: &SourceMode,
    progress: &mut dyn FnMut(&Path, &ComponentReport) -> Result<()>,
) -> Result<ComponentReport> {
    if dst.exists() {
        return Err(GgmlTierError::DestinationExists(dst.to_path_buf()));
    }
    let manifest_path = src.join(SPLIT_MANIFEST);
    if !manifest_path.is_file() {
        return Err(split_err(src, format!("no {SPLIT_MANIFEST}")));
    }
    let manifest_text = io(
        std::fs::read_to_string(&manifest_path),
        manifest_path.display(),
    )?;
    let manifest: Value = serde_json::from_str(&manifest_text)
        .map_err(|e| split_err(src, format!("{SPLIT_MANIFEST}: {e}")))?;
    if manifest.get("format").and_then(Value::as_str) != Some("split") {
        return Err(split_err(
            src,
            format!("{SPLIT_MANIFEST} is not `format: \"split\"`"),
        ));
    }
    if manifest.get(SPLIT_STORAGE_KEY).is_some() {
        return Err(split_err(src, "is already a GGML split tier"));
    }
    if manifest.get("quantized").and_then(Value::as_bool) != Some(true) {
        return Err(split_err(
            src,
            "declares `quantized: false` - a dense tier has nothing to convert",
        ));
    }
    let bits = manifest
        .get("quantization_bits")
        .and_then(Value::as_u64)
        .ok_or_else(|| split_err(src, "has no `quantization_bits`"))?;
    let group_size = manifest
        .get("quantization_group_size")
        .and_then(Value::as_u64)
        .map_or(MLX_GROUP_SIZE, |g| g as usize);
    let qc_path = src.join(QUANTIZE_CONFIG);
    if qc_path.is_file() {
        let text = io(std::fs::read_to_string(&qc_path), qc_path.display())?;
        let (qc_bits, qc_group) = quantize_config_quant(&text)?;
        if (qc_bits, qc_group) != (bits, group_size) {
            return Err(split_err(
                src,
                format!(
                    "{QUANTIZE_CONFIG} declares {qc_bits} bits / group {qc_group}, the manifest \
                     {bits} / {group_size}"
                ),
            ));
        }
    }
    if matches!(mode, SourceMode::FromBf16(_)) && bits != 8 {
        return Err(GgmlTierError::TemplateNotQ8(bits));
    }
    if !group_size.is_multiple_of(GGML_BLOCK) {
        return Err(GgmlTierError::GroupSize { group_size });
    }
    let files = component_files(src)?;
    let template = template_shards(&files, mode)?;
    if template.is_empty() {
        return Err(GgmlTierError::NoShards(src.to_path_buf()));
    }
    let ctx = Ctx {
        mode,
        dense: match mode {
            SourceMode::FromBf16(s) => Some(ShardSet::open(&s.dir, true)?),
            SourceMode::FromMlx => None,
        },
        group_size,
        cfg_bits: bits,
        keep_metadata: true,
    };

    let partial = partial_path(dst);
    if partial.exists() {
        io(std::fs::remove_dir_all(&partial), partial.display())?;
    }
    io(std::fs::create_dir_all(&partial), partial.display())?;

    let mut report = ComponentReport::default();
    let mut stats = BTreeMap::new();
    for (name, shard) in &template {
        let packed = shard.entries.keys().any(|k| k.ends_with(".scales"));
        let target = partial.join(name);
        let (tensors, converted) = match (&shard.path, packed) {
            (Some(path), false) => {
                // Nothing to convert: the file is copied byte-for-byte.
                io(std::fs::copy(path, &target), path.display())?;
                let n = shard.entries.len();
                report.passthrough += n;
                report.copied_files += 1;
                let len: u64 = shard.entries.values().map(Entry::len).sum();
                report.bytes_in += len;
                report.tensor_bytes += len;
                (n, 0)
            }
            _ => {
                let before = report.converted;
                let (_, keys) = write_shard(shard, name, &partial, &ctx, &mut report)?;
                (keys.len(), report.converted - before)
            }
        };
        let bytes = io(std::fs::metadata(&target), target.display())?.len();
        stats.insert(
            name.to_string_lossy().into_owned(),
            SplitFileStats {
                tensors,
                converted,
                bytes,
            },
        );
        progress(&target, &report)?;
    }

    for file in files.iter().filter(|p| !is_shard(p) && !is_header_dump(p)) {
        let name = file.file_name().expect("sidecar file name");
        let target = partial.join(name);
        let text_name = name.to_string_lossy();
        if text_name == SPLIT_MANIFEST {
            let text = ggml_split_manifest(src, &manifest_text, &stats)?;
            io(std::fs::write(&target, text), target.display())?;
        } else if text_name == QUANTIZE_CONFIG {
            let text = io(std::fs::read_to_string(file), file.display())?;
            io(
                std::fs::write(&target, ggml_quantize_config(&text)?),
                target.display(),
            )?;
        } else {
            io(std::fs::copy(file, &target), file.display())?;
        }
    }
    io(std::fs::rename(&partial, dst), dst.display())?;
    Ok(report)
}

// ---------------------------------------------------------------------------------------------
// CLI planning
// ---------------------------------------------------------------------------------------------

/// A tier directory (relative path segment(s), e.g. `q4` or `distilled/q8`) and how it is built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TierSpec {
    pub tier: String,
    /// `Some(dir)` => from-bf16 against `<dir>/<component>`; `None` => from-mlx.
    pub bf16_root: Option<PathBuf>,
}

impl TierSpec {
    /// Parse `TIER=from-mlx` or `TIER=from-bf16@<bf16 root>`.
    pub fn parse(s: &str) -> Result<Self> {
        let (tier, mode) = s.split_once('=').ok_or_else(|| {
            GgmlTierError::Spec(format!(
                "tier `{s}` must be TIER=from-mlx or TIER=from-bf16@BF16_ROOT"
            ))
        })?;
        let tier = tier.trim();
        check_rel(tier)?;
        let bf16_root = match mode.trim() {
            "from-mlx" => None,
            m => match m.strip_prefix("from-bf16@") {
                Some(root) if !root.is_empty() => Some(PathBuf::from(root)),
                _ => {
                    return Err(GgmlTierError::Spec(format!(
                        "tier `{s}`: mode must be `from-mlx` or `from-bf16@BF16_ROOT`"
                    )))
                }
            },
        };
        Ok(Self {
            tier: tier.to_string(),
            bf16_root,
        })
    }
}

fn check_rel(p: &str) -> Result<()> {
    let bad = p.is_empty()
        || Path::new(p).is_absolute()
        || p.split(['/', '\\']).any(|s| s == ".." || s.is_empty());
    if bad {
        return Err(GgmlTierError::Spec(format!(
            "`{p}` must be a non-empty relative path without `..`"
        )));
    }
    Ok(())
}

/// One component (or split-tier) build.
#[derive(Clone, Debug)]
pub struct Job {
    pub label: String,
    pub src: PathBuf,
    pub dst: PathBuf,
    pub mode: SourceMode,
    /// A whole split tier ([`convert_split_tier`]) rather than one component directory.
    pub split: bool,
}

impl Job {
    /// Run this job ([`convert_component`] or [`convert_split_tier`]).
    pub fn run(
        &self,
        progress: &mut dyn FnMut(&Path, &ComponentReport) -> Result<()>,
    ) -> Result<ComponentReport> {
        if self.split {
            convert_split_tier(&self.src, &self.dst, &self.mode, progress)
        } else {
            convert_component(&self.src, &self.dst, &self.mode, progress)
        }
    }
}

/// Expand `tiers x components` into jobs: `<src_root>/<tier>/<component>` ->
/// `<out_root>/<tier>/<component>`; a from-bf16 tier reads dense tensors from
/// `<bf16 root>/<component>`.
///
/// The component `.` is the tier directory itself (`<src_root>/<tier>` -> `<out_root>/<tier>`, dense
/// tensors from `<bf16 root>`), for flat tiers whose root holds the config and shards.
///
/// With **no components**, each tier is a **split tier** ([`convert_split_tier`]):
/// `<src_root>/<tier>` -> `<out_root>/<tier>`, and a from-bf16 tier's `<bf16 root>` is the bf16
/// split tier directory itself (e.g. `.../ltx-2.5-mlx/distilled/bf16`).
pub fn plan_jobs(
    src_root: &Path,
    out_root: &Path,
    tiers: &[TierSpec],
    components: &[String],
    prefix_maps: &[PrefixMap],
) -> Result<Vec<Job>> {
    let mut jobs = Vec::new();
    for c in components {
        check_rel(c)?;
    }
    if components.is_empty() {
        for t in tiers {
            jobs.push(Job {
                label: format!("{} (split)", t.tier),
                src: src_root.join(&t.tier),
                dst: out_root.join(&t.tier),
                mode: match &t.bf16_root {
                    None => SourceMode::FromMlx,
                    Some(root) => SourceMode::FromBf16(Bf16Source {
                        dir: root.clone(),
                        prefix_maps: prefix_maps.to_vec(),
                    }),
                },
                split: true,
            });
        }
        return Ok(jobs);
    }
    for t in tiers {
        for c in components {
            // `.` names the tier directory itself as the component (a flat tier whose root holds
            // the config and shards: scail2-mlx, sensenova-u1-8b-mlx).
            let (rel, dense) = if c == "." {
                (PathBuf::from(&t.tier), PathBuf::new())
            } else {
                (Path::new(&t.tier).join(c), PathBuf::from(c))
            };
            jobs.push(Job {
                label: format!("{}/{c}", t.tier),
                src: src_root.join(&rel),
                dst: out_root.join(&rel),
                mode: match &t.bf16_root {
                    None => SourceMode::FromMlx,
                    Some(root) => SourceMode::FromBf16(Bf16Source {
                        dir: if c == "." {
                            root.clone()
                        } else {
                            root.join(&dense)
                        },
                        prefix_maps: prefix_maps.to_vec(),
                    }),
                },
                split: false,
            });
        }
    }
    Ok(jobs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_layers::llm_projection::from_ggml_block_tensor;
    use half::f16;

    fn tmp() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    /// MLX q4 pack at `group`: f16-exact scales/biases, position-dependent codes; also returns the
    /// exact affine grid `scale * q + bias` it represents.
    fn q4_pack(out: usize, inp: usize, group: usize, phase: usize) -> [Tensor; 3] {
        let dev = Device::Cpu;
        let codes: Vec<u32> = (0..out * inp)
            .map(|i| (((i + phase) * 7 + i / 13) % 16) as u32)
            .collect();
        let groups = out * (inp / group);
        let scales: Vec<f32> = (0..groups)
            .map(|k| 0.0625 * (((k + phase) % 7) as f32 + 1.0))
            .collect();
        let biases: Vec<f32> = (0..groups)
            .map(|k| -0.5 - 0.125 * ((k + phase) % 5) as f32)
            .collect();
        let words: Vec<u32> = codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | (q << (4 * i)))
            })
            .collect();
        [
            Tensor::from_vec(words, (out, inp / 8), &dev).unwrap(),
            Tensor::from_vec(scales, (out, inp / group), &dev).unwrap(),
            Tensor::from_vec(biases, (out, inp / group), &dev).unwrap(),
        ]
    }

    /// Arbitrary (not self-consistent) MLX q8 triple: only the shapes matter for a template.
    fn q8_template(out: usize, inp: usize, group: usize) -> [Tensor; 3] {
        let dev = Device::Cpu;
        let words: Vec<u32> = (0..out * inp / 4)
            .map(|i| (i as u32).wrapping_mul(2654435761))
            .collect();
        [
            Tensor::from_vec(words, (out, inp / 4), &dev).unwrap(),
            Tensor::ones((out, inp / group), DType::BF16, &dev).unwrap(),
            Tensor::zeros((out, inp / group), DType::BF16, &dev).unwrap(),
        ]
    }

    fn put_triple(map: &mut HashMap<String, Tensor>, base: &str, t: [Tensor; 3]) {
        let [w, s, b] = t;
        map.insert(format!("{base}.weight"), w);
        map.insert(format!("{base}.scales"), s);
        map.insert(format!("{base}.biases"), b);
    }

    fn save(map: &HashMap<String, Tensor>, path: PathBuf) {
        candle_core::safetensors::save(map, path).unwrap();
    }

    fn write_component(dir: &Path, bits: u64, group: usize, shards: &[HashMap<String, Tensor>]) {
        std::fs::create_dir_all(dir).unwrap();
        let mut wm = serde_json::Map::new();
        for (i, shard) in shards.iter().enumerate() {
            let name = format!("model-{:05}-of-{:05}.safetensors", i + 1, shards.len());
            save(shard, dir.join(&name));
            for k in shard.keys() {
                wm.insert(k.clone(), Value::from(name.clone()));
            }
        }
        std::fs::write(
            dir.join("config.json"),
            format!(
                r#"{{"hidden_size": 8, "quantization": {{"bits": {bits}, "group_size": {group}}}}}"#
            ),
        )
        .unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::json!({"metadata": {"total_size": 1}, "weight_map": wm}).to_string(),
        )
        .unwrap();
        std::fs::write(dir.join("tokenizer.json"), "{\"t\":1}").unwrap();
    }

    fn norm() -> Tensor {
        Tensor::new(&[1f32, 2., 3., 4.], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    }

    fn bytes(t: &Tensor) -> Vec<u8> {
        t.flatten_all().unwrap().to_vec1::<u8>().unwrap()
    }

    fn dense_bf16(o: usize, i: usize, seed: f32) -> Tensor {
        let v: Vec<f32> = (0..o * i)
            .map(|k| ((k as f32 * seed).sin()) * 0.37 + 0.01)
            .collect();
        Tensor::from_vec(v, (o, i), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    }

    fn noop(_: &Path, _: &ComponentReport) -> Result<()> {
        Ok(())
    }

    fn load(path: PathBuf) -> HashMap<String, Tensor> {
        candle_core::safetensors::load(path, &Device::Cpu).unwrap()
    }

    #[test]
    fn config_marker_keeps_width_drops_group_and_refuses_dense() {
        let out =
            ggml_config(r#"{"h": 8, "quantization": {"bits": 4, "group_size": 64}}"#).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(
            v["quantization"],
            serde_json::json!({"bits": 4, "storage": "ggml"})
        );
        assert_eq!(v["h"], 8);
        assert!(matches!(
            ggml_config(r#"{"h": 8}"#),
            Err(GgmlTierError::Config(_))
        ));
    }

    #[test]
    fn q4_group_sizes_split_into_lossless_q4_1_blocks() {
        for group in [32usize, 64, 128] {
            let root = tmp();
            let (src, dst) = (root.path().join("src"), root.path().join("dst"));
            let (out, inp) = (6usize, 256usize);
            let [wq, s, b] = q4_pack(out, inp, group, 3);
            let mut shard = HashMap::new();
            put_triple(&mut shard, "p", [wq.clone(), s.clone(), b.clone()]);
            shard.insert("n.weight".into(), norm());
            write_component(&src, 4, group, &[shard]);

            let report = convert_component(&src, &dst, &SourceMode::FromMlx, &mut noop).unwrap();
            assert_eq!(
                (report.shards, report.converted, report.passthrough),
                (1, 1, 1)
            );

            let got = load(dst.join("model-00001-of-00001.safetensors"));
            assert!(!got.contains_key("p.scales") && !got.contains_key("p.biases"));
            let stored = &got["p.weight"];
            assert_eq!(stored.dims(), [out, inp / 32, 20], "group {group}");
            // Round-trip equality against the MLX affine grid through f16-cast scale/bias.
            let dq = from_ggml_block_tensor(stored, &Device::Cpu)
                .unwrap()
                .dequantize(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let words = wq.flatten_all().unwrap().to_vec1::<u32>().unwrap();
            let sc = s.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let bi = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let gpr = inp / group;
            for i in 0..out * inp {
                let q = (words[i / 8] >> (4 * (i % 8))) & 0xF;
                let g = (i / inp) * gpr + (i % inp) / group;
                let want = f16::from_f32(sc[g]).to_f32() * q as f32 + f16::from_f32(bi[g]).to_f32();
                assert_eq!(dq[i], want, "group {group} elem {i}");
            }
            // And byte-equal to the load-time repack at that group size.
            let expected = repack_packed_weight_uncached(&wq, &s, &b, group, &Device::Cpu).unwrap();
            assert_eq!(
                stored.flatten_all().unwrap().to_vec1::<u8>().unwrap(),
                expected.data().unwrap().into_owned()
            );
        }
    }

    #[test]
    fn layout_index_config_sidecars_and_partial_rename() {
        let root = tmp();
        let (src, dst) = (root.path().join("src"), root.path().join("nested/dst"));
        let mut a = HashMap::new();
        put_triple(&mut a, "a", q4_pack(4, 64, 64, 0));
        a.insert("n.weight".into(), norm());
        let mut b = HashMap::new();
        put_triple(&mut b, "b", q4_pack(4, 64, 64, 1));
        write_component(&src, 4, 64, &[a, b]);
        // A stale partial from an interrupted run is replaced.
        let stale = root.path().join("nested/dst.partial");
        std::fs::create_dir_all(&stale).unwrap();
        std::fs::write(stale.join("junk"), "x").unwrap();

        let mut seen = Vec::new();
        let report = convert_component(&src, &dst, &SourceMode::FromMlx, &mut |p, r| {
            seen.push((p.to_path_buf(), r.shards));
            Ok(())
        })
        .unwrap();
        assert_eq!(report.shards, 2);
        assert!(seen
            .iter()
            .all(|(p, _)| p.to_string_lossy().contains(".partial")));
        assert!(!stale.exists(), "partial renamed into place");
        assert!(dst.join("model-00002-of-00002.safetensors").is_file());
        assert_eq!(
            std::fs::read_to_string(dst.join("tokenizer.json")).unwrap(),
            "{\"t\":1}"
        );
        let cfg: Value =
            serde_json::from_str(&std::fs::read_to_string(dst.join("config.json")).unwrap())
                .unwrap();
        assert_eq!(
            cfg["quantization"],
            serde_json::json!({"bits": 4, "storage": "ggml"})
        );
        let idx: Value = serde_json::from_str(
            &std::fs::read_to_string(dst.join("model.safetensors.index.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(idx["metadata"]["total_size"], report.tensor_bytes);
        let wm = idx["weight_map"].as_object().unwrap();
        assert_eq!(wm["a.weight"], "model-00001-of-00002.safetensors");
        assert_eq!(wm["b.weight"], "model-00002-of-00002.safetensors");
        assert!(wm
            .keys()
            .all(|k| !k.ends_with(".scales") && !k.ends_with(".biases")));
        // passthrough bytes are verbatim
        let a = load(dst.join("model-00001-of-00002.safetensors"));
        assert_eq!(a["n.weight"].dtype(), DType::BF16);
        assert_eq!(
            a["n.weight"]
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            [1., 2., 3., 4.]
        );

        // Existing dst is a hard error; a failed run leaves no dst.
        assert!(matches!(
            convert_component(&src, &dst, &SourceMode::FromMlx, &mut noop),
            Err(GgmlTierError::DestinationExists(_))
        ));
    }

    #[test]
    fn q8_from_bf16_matches_candle_q8_0_bit_for_bit_across_resharding() {
        let root = tmp();
        let (src, dense, dst) = (
            root.path().join("q8"),
            root.path().join("bf16"),
            root.path().join("out"),
        );
        let (o1, i1, o2, i2, g) = (5usize, 96usize, 3usize, 64usize, 32usize);
        let mut s1 = HashMap::new();
        put_triple(&mut s1, "x.l1", q8_template(o1, i1, g));
        s1.insert("x.norm.weight".into(), norm());
        let mut s2 = HashMap::new();
        put_triple(&mut s2, "y.l2", q8_template(o2, i2, g));
        write_component(&src, 8, g, &[s1, s2]);

        let mk = |o: usize, i: usize, seed: f32| {
            let v: Vec<f32> = (0..o * i)
                .map(|k| ((k as f32 * seed).sin()) * 0.37 + 0.01)
                .collect();
            Tensor::from_vec(v, (o, i), &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let (w1, w2) = (mk(o1, i1, 0.7), mk(o2, i2, 1.3));
        // The dense source is sharded differently (opposite order, 3 files, extra tensors).
        std::fs::create_dir_all(&dense).unwrap();
        save(
            &HashMap::from([("y.l2.weight".to_string(), w2.clone())]),
            dense.join("a.safetensors"),
        );
        save(
            &HashMap::from([("x.l1.weight".to_string(), w1.clone())]),
            dense.join("b.safetensors"),
        );
        save(
            &HashMap::from([("unused.weight".to_string(), norm())]),
            dense.join("c.safetensors"),
        );

        let mode = SourceMode::FromBf16(Bf16Source {
            dir: dense,
            prefix_maps: vec![],
        });
        let report = convert_component(&src, &dst, &mode, &mut noop).unwrap();
        assert_eq!(
            (report.shards, report.converted, report.passthrough),
            (2, 2, 1)
        );
        assert!(report.bytes_in > 0 && report.tensor_bytes > 0);

        for (file, name, w) in [
            ("model-00001-of-00002.safetensors", "x.l1.weight", &w1),
            ("model-00002-of-00002.safetensors", "y.l2.weight", &w2),
        ] {
            let got = &load(dst.join(file))[name];
            let (o, i) = w.dims2().unwrap();
            assert_eq!(got.dims(), [o, i / 32, 34]);
            let want =
                QTensor::quantize(&w.to_dtype(DType::F32).unwrap(), GgmlDType::Q8_0).unwrap();
            assert_eq!(
                got.flatten_all().unwrap().to_vec1::<u8>().unwrap(),
                want.data().unwrap().into_owned(),
                "{name}"
            );
        }
        let m1 = load(dst.join("model-00001-of-00002.safetensors"));
        assert!(m1.contains_key("x.norm.weight") && !m1.contains_key("x.l1.scales"));
    }

    #[test]
    fn q8_from_bf16_prefix_map_is_explicit_only() {
        let root = tmp();
        let (src, dense) = (root.path().join("q8"), root.path().join("bf16"));
        let mut s = HashMap::new();
        put_triple(&mut s, "m.l", q8_template(2, 32, 32));
        write_component(&src, 8, 32, &[s]);
        std::fs::create_dir_all(&dense).unwrap();
        let w = Tensor::ones((2, 32), DType::F32, &Device::Cpu).unwrap();
        save(
            &HashMap::from([("model.l.weight".to_string(), w)]),
            dense.join("w.safetensors"),
        );

        let plain = SourceMode::FromBf16(Bf16Source {
            dir: dense.clone(),
            prefix_maps: vec![],
        });
        let err = convert_component(&src, &root.path().join("o1"), &plain, &mut noop).unwrap_err();
        assert!(
            matches!(&err, GgmlTierError::MissingDenseSource { looked_up, .. } if looked_up == "m.l.weight")
        );
        assert!(
            !root.path().join("o1").exists(),
            "failed run leaves no tier"
        );

        let mapped = SourceMode::FromBf16(Bf16Source {
            dir: dense,
            prefix_maps: vec![PrefixMap::parse("m.=model.").unwrap()],
        });
        convert_component(&src, &root.path().join("o2"), &mapped, &mut noop).unwrap();
    }

    fn header_fixture(root: &Path) -> (PathBuf, PathBuf, Tensor, Tensor) {
        let (src, dense) = (root.join("q8"), root.join("bf16"));
        let g = 32usize;
        let mut s1 = HashMap::new();
        put_triple(&mut s1, "x.l1", q8_template(5, 96, g));
        s1.insert("x.norm.weight".into(), norm());
        let mut s2 = HashMap::new();
        put_triple(&mut s2, "y.l2", q8_template(3, 64, g));
        write_component(&src, 8, g, &[s1, s2]);
        let (w1, w2) = (dense_bf16(5, 96, 0.7), dense_bf16(3, 64, 1.3));
        std::fs::create_dir_all(&dense).unwrap();
        save(
            &HashMap::from([("y.l2.weight".to_string(), w2.clone())]),
            dense.join("a.safetensors"),
        );
        save(
            &HashMap::from([
                ("x.l1.weight".to_string(), w1.clone()),
                (
                    "x.norm.weight".to_string(),
                    norm().to_dtype(DType::F32).unwrap(),
                ),
            ]),
            dense.join("b.safetensors"),
        );
        (src, dense, w1, w2)
    }

    #[test]
    fn q8_from_bf16_header_only_template_matches_candle_q8_0_and_reports_casts() {
        let root = tmp();
        let (src, dense, w1, w2) = header_fixture(root.path());
        let template = root.path().join("tpl");
        assert_eq!(dump_headers(&src, &template).unwrap(), 2);
        assert!(!template.join("model-00001-of-00002.safetensors").exists());
        let dst = root.path().join("out");
        let mode = SourceMode::FromBf16(Bf16Source {
            dir: dense,
            prefix_maps: vec![],
        });
        let report = convert_component(&template, &dst, &mode, &mut noop).unwrap();
        assert_eq!(
            (report.shards, report.converted, report.passthrough),
            (2, 2, 1)
        );
        assert_eq!(
            report.casts,
            [("x.norm.weight".to_string(), "F32".into(), "BF16".into())]
        );
        for (file, name, w) in [
            ("model-00001-of-00002.safetensors", "x.l1.weight", &w1),
            ("model-00002-of-00002.safetensors", "y.l2.weight", &w2),
        ] {
            let got = &load(dst.join(file))[name];
            let (o, i) = w.dims2().unwrap();
            assert_eq!(got.dims(), [o, i / 32, 34]);
            let want =
                QTensor::quantize(&w.to_dtype(DType::F32).unwrap(), GgmlDType::Q8_0).unwrap();
            assert_eq!(bytes(got), want.data().unwrap().into_owned(), "{name}");
        }
        let m1 = load(dst.join("model-00001-of-00002.safetensors"));
        assert_eq!(m1["x.norm.weight"].dtype(), DType::BF16);
        assert!(dst.join("tokenizer.json").is_file());
        assert!(dst.join("model.safetensors.index.json").is_file());
        assert!(!dst
            .join("model-00001-of-00002.safetensors.header.json")
            .exists());
        // A header-only template cannot drive from-mlx.
        assert!(matches!(
            convert_component(
                &template,
                &root.path().join("o2"),
                &SourceMode::FromMlx,
                &mut noop
            ),
            Err(GgmlTierError::NoShards(_))
        ));
    }

    #[test]
    fn header_template_refuses_misshaped_and_missing_non_quantized_tensors() {
        let root = tmp();
        let (src, dense, w1, _) = header_fixture(root.path());
        let template = root.path().join("tpl");
        dump_headers(&src, &template).unwrap();
        let mode = SourceMode::FromBf16(Bf16Source {
            dir: dense.clone(),
            prefix_maps: vec![],
        });
        save(
            &HashMap::from([
                ("x.l1.weight".to_string(), w1.clone()),
                (
                    "x.norm.weight".to_string(),
                    Tensor::ones(7, DType::BF16, &Device::Cpu).unwrap(),
                ),
            ]),
            dense.join("b.safetensors"),
        );
        assert!(matches!(
            convert_component(&template, &root.path().join("o1"), &mode, &mut noop),
            Err(GgmlTierError::DenseShapeMismatch { .. })
        ));
        save(
            &HashMap::from([("x.l1.weight".to_string(), w1)]),
            dense.join("b.safetensors"),
        );
        assert!(matches!(
            convert_component(&template, &root.path().join("o2"), &mode, &mut noop),
            Err(GgmlTierError::MissingDenseSource { .. })
        ));
        assert!(
            !root.path().join("o2").exists(),
            "failed run leaves no tier"
        );
    }

    #[test]
    fn refusal_paths_are_typed() {
        let root = tmp();
        // q4 with a group that is not a multiple of 32.
        let src = root.path().join("g48");
        let mut s = HashMap::new();
        put_triple(&mut s, "p", q4_pack(2, 96, 32, 0));
        write_component(&src, 4, 48, &[s]);
        assert!(matches!(
            convert_component(
                &src,
                &root.path().join("o"),
                &SourceMode::FromMlx,
                &mut noop
            ),
            Err(GgmlTierError::GroupSize { group_size: 48 })
        ));

        // from-bf16 against a q4 template.
        let q4 = root.path().join("q4");
        let mut s = HashMap::new();
        put_triple(&mut s, "p", q4_pack(2, 64, 64, 0));
        write_component(&q4, 4, 64, &[s]);
        let dense = root.path().join("d");
        std::fs::create_dir_all(&dense).unwrap();
        save(
            &HashMap::from([(
                "p.weight".to_string(),
                Tensor::ones((2, 64), DType::F32, &Device::Cpu).unwrap(),
            )]),
            dense.join("w.safetensors"),
        );
        let mode = SourceMode::FromBf16(Bf16Source {
            dir: dense.clone(),
            prefix_maps: vec![],
        });
        assert!(matches!(
            convert_component(&q4, &root.path().join("o"), &mode, &mut noop),
            Err(GgmlTierError::TemplateNotQ8(4))
        ));

        // q8 template, dense shape mismatch / in_features % 32 != 0.
        let q8 = root.path().join("q8");
        let mut s = HashMap::new();
        put_triple(&mut s, "p", q8_template(2, 64, 32));
        write_component(&q8, 8, 32, &[s]);
        let bad = root.path().join("bad");
        std::fs::create_dir_all(&bad).unwrap();
        save(
            &HashMap::from([(
                "p.weight".to_string(),
                Tensor::ones((2, 32), DType::F32, &Device::Cpu).unwrap(),
            )]),
            bad.join("w.safetensors"),
        );
        let mode = SourceMode::FromBf16(Bf16Source {
            dir: bad,
            prefix_maps: vec![],
        });
        assert!(matches!(
            convert_component(&q8, &root.path().join("o"), &mode, &mut noop),
            Err(GgmlTierError::DenseShapeMismatch { .. })
        ));
        let q8b = root.path().join("q8b");
        let mut s = HashMap::new();
        put_triple(&mut s, "p", q8_template(2, 48, 16));
        write_component(&q8b, 8, 16, &[s]);
        let d48 = root.path().join("d48");
        std::fs::create_dir_all(&d48).unwrap();
        save(
            &HashMap::from([(
                "p.weight".to_string(),
                Tensor::ones((2, 48), DType::F32, &Device::Cpu).unwrap(),
            )]),
            d48.join("w.safetensors"),
        );
        let mode = SourceMode::FromBf16(Bf16Source {
            dir: d48,
            prefix_maps: vec![],
        });
        assert!(matches!(
            convert_component(&q8b, &root.path().join("o"), &mode, &mut noop),
            Err(GgmlTierError::InFeaturesNotBlockMultiple {
                in_features: 48,
                ..
            })
        ));

        // Wrong group size read from config (g64 declared, pack is g32) is caught, not mis-converted.
        let wrong = root.path().join("wrong");
        let mut s = HashMap::new();
        put_triple(&mut s, "p", q4_pack(2, 128, 32, 0));
        write_component(&wrong, 4, 64, &[s]);
        assert!(convert_component(
            &wrong,
            &root.path().join("o"),
            &SourceMode::FromMlx,
            &mut noop
        )
        .is_err());
    }

    /// sha256 over every file of `dir` (sorted relative path, length, bytes).
    fn dir_digest(dir: &Path) -> String {
        use sha2::{Digest, Sha256};
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        files.sort();
        let mut h = Sha256::new();
        for f in files {
            let bytes = std::fs::read(&f).unwrap();
            h.update(f.file_name().unwrap().to_string_lossy().as_bytes());
            h.update((bytes.len() as u64).to_le_bytes());
            h.update(&bytes);
        }
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The diffusers-component layout (Qwen-Image / FLUX.2 dev / Qwen-Image-Edit, already published)
    /// must stay byte-identical: these digests were captured from the slice-B builder (77b4140039)
    /// before the SW-7285 C2 extensions, over a fixture covering q4 from-mlx at two group sizes and
    /// two shards, q8 from-bf16 with a full template, and q8 from-bf16 with a header-only template
    /// (with a cast).
    #[test]
    fn diffusers_layout_output_is_byte_identical_to_the_slice_b_builder() {
        let root = tmp();
        let r = root.path();
        // q4 from-mlx, group 64, two shards, nested config, sidecars.
        let q4 = r.join("q4");
        let mut a = HashMap::new();
        put_triple(&mut a, "blocks.0.attn.to_q", q4_pack(8, 128, 64, 1));
        a.insert("blocks.0.norm.weight".into(), norm());
        a.insert(
            "blocks.0.attn.to_q.bias".into(),
            Tensor::new(&[0.5f32; 8], &Device::Cpu).unwrap(),
        );
        let mut b = HashMap::new();
        put_triple(&mut b, "proj_out", q4_pack(4, 64, 64, 5));
        write_component(&q4, 4, 64, &[a, b]);
        convert_component(&q4, &r.join("o_q4"), &SourceMode::FromMlx, &mut noop).unwrap();
        // q4 from-mlx, group 32.
        let q4g32 = r.join("q4g32");
        let mut c = HashMap::new();
        put_triple(&mut c, "x", q4_pack(3, 96, 32, 2));
        write_component(&q4g32, 4, 32, &[c]);
        convert_component(&q4g32, &r.join("o_q4g32"), &SourceMode::FromMlx, &mut noop).unwrap();
        // q8 from-bf16, full template and header-only template.
        let (src, dense, _, _) = header_fixture(r);
        let mode = SourceMode::FromBf16(Bf16Source {
            dir: dense,
            prefix_maps: vec![],
        });
        convert_component(&src, &r.join("o_q8"), &mode, &mut noop).unwrap();
        let tpl = r.join("tpl");
        dump_headers(&src, &tpl).unwrap();
        convert_component(&tpl, &r.join("o_q8h"), &mode, &mut noop).unwrap();

        let got: Vec<(&str, String)> = ["o_q4", "o_q4g32", "o_q8", "o_q8h"]
            .into_iter()
            .map(|d| (d, dir_digest(&r.join(d))))
            .collect();
        let want = [
            (
                "o_q4",
                "53418ada2e4774c2b6c826f9c1af2c3e60efe26389ddfdeaed7ac120f991ba59",
            ),
            (
                "o_q4g32",
                "238f9a837779c9f5ec288aa105e9264bad9429ce845613b2acef5130f3a39a0b",
            ),
            (
                "o_q8",
                "f588fc6e12f3a8ae1ade370e10168715db87eb294e6cf5d816666970d638e61a",
            ),
            (
                "o_q8h",
                "f588fc6e12f3a8ae1ade370e10168715db87eb294e6cf5d816666970d638e61a",
            ),
        ];
        let drift: Vec<String> = got
            .iter()
            .zip(want)
            .filter(|((_, g), (_, w))| g != w)
            .map(|((d, g), _)| format!("{d}: {g}"))
            .collect();
        assert!(
            drift.is_empty(),
            "output drifted from the slice-B builder: {drift:#?}"
        );
    }

    // -----------------------------------------------------------------------------------------
    // SW-7285 C2: stacked experts, markers, split tiers
    // -----------------------------------------------------------------------------------------

    fn stack(ts: Vec<Tensor>) -> Tensor {
        Tensor::stack(&ts, 0).unwrap()
    }

    /// A rank-3 MLX q4 pack: `experts` distinct [`q4_pack`]s stacked on a leading axis.
    fn q4_pack3(experts: usize, out: usize, inp: usize, group: usize) -> [Tensor; 3] {
        let packs: Vec<[Tensor; 3]> = (0..experts)
            .map(|x| q4_pack(out, inp, group, 3 * x + 1))
            .collect();
        let pick = |i: usize| stack(packs.iter().map(|p| p[i].clone()).collect());
        [pick(0), pick(1), pick(2)]
    }

    fn q8_template3(experts: usize, out: usize, inp: usize, group: usize) -> [Tensor; 3] {
        let packs: Vec<[Tensor; 3]> = (0..experts).map(|_| q8_template(out, inp, group)).collect();
        let pick = |i: usize| stack(packs.iter().map(|p| p[i].clone()).collect());
        [pick(0), pick(1), pick(2)]
    }

    fn expert_bias(experts: usize, out: usize) -> Tensor {
        Tensor::ones((experts, out), DType::F32, &Device::Cpu).unwrap()
    }

    #[test]
    fn rank3_experts_q4_1_from_mlx_equal_the_per_expert_repack() {
        let root = tmp();
        let (src, dst) = (root.path().join("te"), root.path().join("out"));
        let (e, out, inp, g) = (3usize, 6usize, 128usize, 64usize);
        let gate_up = q4_pack3(e, 2 * out, inp, g);
        let down = q4_pack3(e, out, inp, g);
        let mut shard = HashMap::new();
        put_triple(&mut shard, "l.experts.gate_up_proj", gate_up.clone());
        put_triple(&mut shard, "l.experts.down_proj", down.clone());
        shard.insert(
            "l.experts.gate_up_proj_bias".into(),
            expert_bias(e, 2 * out),
        );
        // A rank-2 projection beside them still converts as before.
        put_triple(&mut shard, "l.attn.q", q4_pack(out, inp, g, 9));
        write_component(&src, 4, g, &[shard]);

        let report = convert_component(&src, &dst, &SourceMode::FromMlx, &mut noop).unwrap();
        assert_eq!((report.converted, report.passthrough), (3, 1));
        let got = load(dst.join("model-00001-of-00001.safetensors"));
        for (base, [wq, s, b], rows) in [
            ("l.experts.gate_up_proj", gate_up, 2 * out),
            ("l.experts.down_proj", down, out),
        ] {
            let stored = &got[&format!("{base}.weight")];
            assert_eq!(stored.dtype(), DType::U8);
            assert_eq!(stored.dims(), [e, rows, inp / 32, 20], "{base}");
            // The loader's header rule classifies it as stacked Q4_1 blocks of every expert.
            assert_eq!(
                candle_layers::llm_projection::ggml_block_storage(true, stored.dims()),
                Some((GgmlDType::Q4_1, e * rows, inp))
            );
            assert!(!got.contains_key(&format!("{base}.scales")));
            for x in 0..e {
                let want = repack_packed_weight_uncached(
                    &wq.get(x).unwrap(),
                    &s.get(x).unwrap(),
                    &b.get(x).unwrap(),
                    g,
                    &Device::Cpu,
                )
                .unwrap();
                assert_eq!(
                    bytes(&stored.get(x).unwrap()),
                    want.data().unwrap().into_owned(),
                    "{base} expert {x}"
                );
            }
        }
        assert_eq!(got["l.attn.q.weight"].dims(), [out, inp / 32, 20]);
        assert_eq!(got["l.experts.gate_up_proj_bias"].dims(), [e, 2 * out]);

        // The public primitive agrees with the component conversion.
        let [wq, s, b] = q4_pack3(e, out, inp, g);
        let blocks = mlx_triple_to_ggml_blocks("x", &wq, &s, &b, g).unwrap();
        assert_eq!(blocks.dims(), [e, out, inp / 32, 20]);
        // A scales leading axis that disagrees with the codes is refused, not reshaped.
        let bad = mlx_triple_to_ggml_blocks("x", &wq, &s.narrow(0, 0, 2).unwrap(), &b, g);
        assert!(matches!(bad, Err(GgmlTierError::TripleShape { .. })));
    }

    #[test]
    fn rank3_experts_q8_0_from_bf16_equal_candle_q8_0_per_expert() {
        let root = tmp();
        let (src, dense) = (root.path().join("q8"), root.path().join("bf16"));
        let (e, out, inp, g) = (2usize, 5usize, 96usize, 32usize);
        let mut shard = HashMap::new();
        put_triple(
            &mut shard,
            "m.experts.down_proj",
            q8_template3(e, out, inp, g),
        );
        shard.insert(
            "m.experts.down_proj_bias".into(),
            expert_bias(e, out).to_dtype(DType::BF16).unwrap(),
        );
        write_component(&src, 8, g, &[shard]);
        let w = stack(
            (0..e)
                .map(|x| dense_bf16(out, inp, 0.3 + x as f32))
                .collect(),
        );
        std::fs::create_dir_all(&dense).unwrap();
        save(
            &HashMap::from([
                ("m.experts.down_proj.weight".to_string(), w.clone()),
                ("m.experts.down_proj_bias".to_string(), expert_bias(e, out)),
            ]),
            dense.join("model.safetensors"),
        );
        let mode = SourceMode::FromBf16(Bf16Source {
            dir: dense,
            prefix_maps: vec![],
        });
        // Full template and header-only template (which casts the F32 expert bias to BF16).
        let tpl = root.path().join("tpl");
        dump_headers(&src, &tpl).unwrap();
        for (template, out_dir, casts) in [(&src, "o1", 0usize), (&tpl, "o2", 1)] {
            let dst = root.path().join(out_dir);
            let report = convert_component(template, &dst, &mode, &mut noop).unwrap();
            assert_eq!(report.converted, 1);
            assert_eq!(report.casts.len(), casts);
            let got = load(dst.join("model-00001-of-00001.safetensors"));
            let stored = &got["m.experts.down_proj.weight"];
            assert_eq!(stored.dims(), [e, out, inp / 32, 34]);
            for x in 0..e {
                let want = QTensor::quantize(
                    &w.get(x).unwrap().to_dtype(DType::F32).unwrap(),
                    GgmlDType::Q8_0,
                )
                .unwrap();
                assert_eq!(
                    bytes(&stored.get(x).unwrap()),
                    want.data().unwrap().into_owned(),
                    "expert {x}"
                );
            }
            assert_eq!(got["m.experts.down_proj_bias"].dtype(), DType::BF16);
        }
        // The public primitive is the same quantizer.
        let blocks = dense_to_q8_0_blocks("m", &w).unwrap();
        let got = load(root.path().join("o1/model-00001-of-00001.safetensors"));
        assert_eq!(bytes(&blocks), bytes(&got["m.experts.down_proj.weight"]));
    }

    /// An MXFP4 fused-expert pair `{base}_blocks` / `{base}_scales` (the gpt-oss form of the Lens
    /// bf16 text encoder): low nibble first, e8m0 scales.
    fn mxfp4_pair(e: usize, out: usize, inp: usize, seed: usize) -> (Tensor, Tensor, Vec<f32>) {
        let nb = inp / 32;
        let blocks: Vec<u8> = (0..e * out * nb * 16)
            .map(|i| ((i * 37 + seed * 11) % 256) as u8)
            .collect();
        let scales: Vec<u8> = (0..e * out * nb)
            .map(|i| 120 + ((i + seed) % 9) as u8)
            .collect();
        // Independent reference decode: sign bit 3, magnitude grid, 2^(s - 127).
        let mag = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let code = |c: u8| {
            let v = mag[(c & 7) as usize];
            if c & 8 != 0 {
                -v
            } else {
                v
            }
        };
        let mut grid = Vec::with_capacity(e * out * inp);
        for blk in 0..e * out * nb {
            let s = (2f64).powi(scales[blk] as i32 - 127) as f32;
            for byte in &blocks[blk * 16..(blk + 1) * 16] {
                grid.push(code(byte & 0x0f) * s);
                grid.push(code(byte >> 4) * s);
            }
        }
        let dev = Device::Cpu;
        (
            Tensor::from_vec(blocks, (e, out, nb, 16), &dev).unwrap(),
            Tensor::from_vec(scales, (e, out, nb), &dev).unwrap(),
            grid,
        )
    }

    #[test]
    fn rank3_experts_q8_0_from_an_mxfp4_source_dequantize_exactly() {
        let root = tmp();
        let (src, dense) = (root.path().join("q8"), root.path().join("bf16"));
        let (e, out, inp, g) = (2usize, 4usize, 128usize, 64usize);
        let mut shard = HashMap::new();
        put_triple(
            &mut shard,
            "x.experts.gate_up_proj",
            q8_template3(e, 2 * out, inp, g),
        );
        put_triple(
            &mut shard,
            "x.experts.down_proj",
            q8_template3(e, out, inp, g),
        );
        write_component(&src, 8, g, &[shard]);
        let (gb, gs, ggrid) = mxfp4_pair(e, 2 * out, inp, 1);
        let (db, ds, dgrid) = mxfp4_pair(e, out, inp, 2);
        std::fs::create_dir_all(&dense).unwrap();
        save(
            &HashMap::from([
                ("x.experts.gate_up_proj_blocks".to_string(), gb),
                ("x.experts.gate_up_proj_scales".to_string(), gs),
                ("x.experts.down_proj_blocks".to_string(), db),
                ("x.experts.down_proj_scales".to_string(), ds),
            ]),
            dense.join("model-00001-of-00001.safetensors"),
        );
        let dst = root.path().join("out");
        let mode = SourceMode::FromBf16(Bf16Source {
            dir: dense,
            prefix_maps: vec![],
        });
        let report = convert_component(&src, &dst, &mode, &mut noop).unwrap();
        assert_eq!((report.converted, report.mxfp4_sources), (2, 2));
        let got = load(dst.join("model-00001-of-00001.safetensors"));
        for (base, rows, grid) in [
            ("x.experts.gate_up_proj", 2 * out, ggrid),
            ("x.experts.down_proj", out, dgrid),
        ] {
            let stored = &got[&format!("{base}.weight")];
            assert_eq!(stored.dims(), [e, rows, inp / 32, 34]);
            for x in 0..e {
                let slice = grid[x * rows * inp..(x + 1) * rows * inp].to_vec();
                let t = Tensor::from_vec(slice, (rows, inp), &Device::Cpu).unwrap();
                let want = QTensor::quantize(&t, GgmlDType::Q8_0).unwrap();
                assert_eq!(
                    bytes(&stored.get(x).unwrap()),
                    want.data().unwrap().into_owned(),
                    "{base} expert {x}"
                );
            }
        }
    }

    #[test]
    fn quantize_config_marker_keeps_top_level_bits_and_nested_form() {
        let wan = ggml_quantize_config(
            "{\n  \"bits\": 8,\n  \"quantization\": { \"group_size\": 64 }\n}\n",
        )
        .unwrap();
        let v: Value = serde_json::from_str(&wan).unwrap();
        assert_eq!(
            v,
            serde_json::json!({"bits": 8, "quantization": {"bits": 8, "storage": "ggml"}})
        );
        let ltx23 =
            ggml_quantize_config(r#"{"quantization": {"bits": 4, "group_size": 64}}"#).unwrap();
        let v: Value = serde_json::from_str(&ltx23).unwrap();
        assert_eq!(
            v,
            serde_json::json!({"quantization": {"bits": 4, "storage": "ggml"}})
        );
        assert!(ggml_quantize_config(r#"{"group_size": 64}"#).is_err());
    }

    #[test]
    fn unmarked_components_infer_one_width_and_gain_the_marker() {
        let root = tmp();
        let src = root.path().join("tr");
        let mut s = HashMap::new();
        put_triple(&mut s, "a", q4_pack(4, 128, 64, 0));
        put_triple(&mut s, "b", q4_pack(2, 64, 64, 1));
        write_component(&src, 4, 64, &[s]);
        // The ideogram / SenseNova converters write a config with no quantization block.
        std::fs::write(src.join("config.json"), r#"{"num_layers": 2}"#).unwrap();
        let dst = root.path().join("out");
        let report = convert_component(&src, &dst, &SourceMode::FromMlx, &mut noop).unwrap();
        assert_eq!(report.converted, 2);
        let cfg: Value =
            serde_json::from_str(&std::fs::read_to_string(dst.join("config.json")).unwrap())
                .unwrap();
        assert_eq!(
            cfg,
            serde_json::json!({"num_layers": 2, "quantization": {"bits": 4, "storage": "ggml"}})
        );
        // Triples that disagree on the width are refused rather than guessed.
        let mixed = root.path().join("mixed");
        let mut s = HashMap::new();
        put_triple(&mut s, "a", q4_pack(4, 128, 64, 0));
        put_triple(&mut s, "b", q8_template(2, 128, 64));
        write_component(&mixed, 4, 64, &[s]);
        std::fs::write(mixed.join("config.json"), r#"{"num_layers": 2}"#).unwrap();
        assert!(matches!(
            convert_component(
                &mixed,
                &root.path().join("o2"),
                &SourceMode::FromMlx,
                &mut noop
            ),
            Err(GgmlTierError::InferredBits { .. })
        ));
        // A dense component with no marker is still not relabelled.
        let dense = root.path().join("dense");
        write_component(&dense, 4, 64, &[HashMap::from([("n".to_string(), norm())])]);
        std::fs::write(dense.join("config.json"), r#"{"num_layers": 2}"#).unwrap();
        assert!(matches!(
            convert_component(
                &dense,
                &root.path().join("o3"),
                &SourceMode::FromMlx,
                &mut noop
            ),
            Err(GgmlTierError::Config(_))
        ));
    }

    fn save_meta(map: &HashMap<String, Tensor>, meta: &[(&str, &str)], path: PathBuf) {
        let meta: HashMap<String, String> = meta
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let mut items: Vec<(String, Tensor)> =
            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        items.sort_by(|a, b| a.0.cmp(&b.0));
        safetensors::serialize_to_file(items, Some(meta), &path).unwrap();
    }

    fn conv(seed: f32) -> Tensor {
        Tensor::from_vec(
            (0..24).map(|i| i as f32 * seed).collect::<Vec<f32>>(),
            (2, 3, 2, 2),
            &Device::Cpu,
        )
        .unwrap()
    }

    /// An LTX-2.5-shaped split tier: `transformer.safetensors` packed (2 triples + a norm),
    /// `vae_encoder` / `vae_decoder` dense and sharing the tensor name `conv_in.weight`, every file
    /// stamped `sceneworks_tier`, plus the manifest and `embedded_config.json`.
    fn split_fixture(dir: &Path, tier: &str, bits: u64, packed: [HashMap<String, Tensor>; 1]) {
        std::fs::create_dir_all(dir).unwrap();
        let stamp = [
            ("sceneworks_tier", tier),
            ("model_version", "2.5.0"),
            ("config", "{\"transformer\": {}}"),
        ];
        let [tr] = packed;
        let tr_tensors = tr.len();
        save_meta(&tr, &stamp, dir.join("transformer.safetensors"));
        for (file, seed) in [("vae_encoder", 0.5f32), ("vae_decoder", 0.25)] {
            save_meta(
                &HashMap::from([("conv_in.weight".to_string(), conv(seed))]),
                &stamp[..2],
                dir.join(format!("{file}.safetensors")),
            );
        }
        let len = |f: &str| std::fs::metadata(dir.join(f)).unwrap().len();
        let dense_row = |name: &str| {
            serde_json::json!({
                "name": name, "file": format!("{name}.safetensors"), "tensors": 1,
                "quantized_linears": 0, "dense_float_tensors": 1, "passthrough_tensors": 0,
                "bytes": len(&format!("{name}.safetensors")),
                "dense_reason": "no-linear-weights", "dense_reason_detail": "convolutions only"
            })
        };
        let manifest = serde_json::json!({
            "format": "split", "model_version": "2.5.0", "variant": "distilled", "tier": tier,
            "quantized": true, "quantization_bits": bits, "quantization_group_size": 64,
            "components": ["transformer", "vae_encoder", "vae_decoder"],
            "component_detail": [
                {"name": "transformer", "file": "transformer.safetensors", "tensors": tr_tensors,
                 "quantized_linears": 2, "dense_float_tensors": 1, "passthrough_tensors": 0,
                 "bytes": len("transformer.safetensors")},
                dense_row("vae_encoder"),
                dense_row("vae_decoder"),
            ],
            "total_bytes": 1
        });
        std::fs::write(
            dir.join(SPLIT_MANIFEST),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.join("embedded_config.json"), "{\"vae\": {}}").unwrap();
    }

    fn read_meta(path: &Path) -> Option<HashMap<String, String>> {
        ShardFile::open(path).unwrap().metadata.map().cloned()
    }

    /// Rewrite `path`'s header so `__metadata__` is `value` (raw JSON), keeping the tensors.
    fn set_raw_metadata(path: &Path, value: &str) {
        let (header, _) = read_header_bytes(path).unwrap();
        let bytes = std::fs::read(path).unwrap();
        let data = bytes[8 + header.len()..].to_vec();
        let mut h: Value = serde_json::from_slice(&header).unwrap();
        h.as_object_mut()
            .unwrap()
            .insert("__metadata__".into(), serde_json::from_str(value).unwrap());
        let mut text = serde_json::to_vec(&h).unwrap();
        while !(8 + text.len()).is_multiple_of(8) {
            text.push(b' ');
        }
        let mut out = (text.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&text);
        out.extend_from_slice(&data);
        std::fs::write(path, out).unwrap();
    }

    /// MLX writes `"__metadata__": null` (the real scail2 `dit.safetensors`): that is "no
    /// metadata" for a component build, a split build and a header dump. Only an unreadable
    /// metadata is refused, and only by a split build (the one place metadata is written back).
    #[test]
    fn null_metadata_is_absent_and_only_a_split_build_refuses_unreadable_metadata() {
        assert_eq!(parse_metadata(&Value::Null), Metadata::Absent);
        assert!(matches!(
            parse_metadata(&serde_json::json!({"a": 1})),
            Metadata::Unreadable(_)
        ));
        let root = tmp();
        // Component build over a null-metadata shard.
        let src = root.path().join("q4");
        let mut s = HashMap::new();
        put_triple(&mut s, "blocks.0.q", q4_pack(4, 64, 64, 0));
        s.insert("n.weight".into(), norm());
        write_component(&src, 4, 64, &[s]);
        let shard = src.join("model-00001-of-00001.safetensors");
        set_raw_metadata(&shard, "null");
        assert_eq!(ShardFile::open(&shard).unwrap().metadata, Metadata::Absent);
        let dst = root.path().join("out");
        let report = convert_component(&src, &dst, &SourceMode::FromMlx, &mut noop).unwrap();
        assert_eq!(report.converted, 1);
        assert!(read_meta(&dst.join("model-00001-of-00001.safetensors")).is_none());
        // Header dumps (dump-headers / fetch-headers) accept it too.
        let tpl = root.path().join("tpl");
        assert_eq!(dump_headers(&src, &tpl).unwrap(), 1);
        let dump = tpl.join("model-00001-of-00001.safetensors.header.json");
        assert_eq!(
            ShardFile::open_header_dump(&dump).unwrap().metadata,
            Metadata::Absent
        );
        assert!(write_header_dump(&tpl, "x.safetensors", &std::fs::read(&dump).unwrap()).is_ok());
        // A component build ignores even unreadable metadata (the slice-B behaviour)...
        set_raw_metadata(&shard, r#"{"n": 3}"#);
        convert_component(
            &src,
            &root.path().join("o2"),
            &SourceMode::FromMlx,
            &mut noop,
        )
        .unwrap();

        // ...and a split build with a null-metadata packed file writes none.
        let split = root.path().join("split");
        let mut tr = HashMap::new();
        put_triple(&mut tr, "a", q4_pack(8, 128, 64, 1));
        put_triple(&mut tr, "b", q4_pack(4, 64, 64, 2));
        tr.insert("n.weight".into(), norm());
        split_fixture(&split, "q4", 4, [tr]);
        set_raw_metadata(&split.join("transformer.safetensors"), "null");
        let sdst = root.path().join("split_out");
        convert_split_tier(&split, &sdst, &SourceMode::FromMlx, &mut noop).unwrap();
        assert!(read_meta(&sdst.join("transformer.safetensors")).is_none());
        // An unreadable one is refused there, never silently dropped.
        set_raw_metadata(
            &split.join("transformer.safetensors"),
            r#"{"sceneworks_tier": 4}"#,
        );
        assert!(matches!(
            convert_split_tier(
                &split,
                &root.path().join("split_bad"),
                &SourceMode::FromMlx,
                &mut noop
            ),
            Err(GgmlTierError::BadSafetensors { .. })
        ));
    }

    fn manifest_of(dir: &Path) -> Value {
        serde_json::from_str(&std::fs::read_to_string(dir.join(SPLIT_MANIFEST)).unwrap()).unwrap()
    }

    #[test]
    fn split_tier_q4_round_trip_converts_stamps_and_restates_the_manifest() {
        let root = tmp();
        let (src, dst) = (root.path().join("q4"), root.path().join("out/q4"));
        let (p1, p2) = (q4_pack(8, 128, 64, 1), q4_pack(4, 64, 64, 2));
        let mut tr = HashMap::new();
        put_triple(&mut tr, "blocks.0.attn.to_q", p1.clone());
        put_triple(&mut tr, "blocks.0.ff.proj_in", p2.clone());
        tr.insert("blocks.0.norm.weight".into(), norm());
        split_fixture(&src, "q4", 4, [tr]);

        let report = convert_split_tier(&src, &dst, &SourceMode::FromMlx, &mut noop).unwrap();
        assert_eq!((report.converted, report.copied_files), (2, 2));

        // Packed file: blocks equal the repack, triples gone, `__metadata__` kept verbatim.
        let tr_out = dst.join("transformer.safetensors");
        let got = load(tr_out.clone());
        assert_eq!(got.len(), 3);
        for (base, [wq, s, b]) in [("blocks.0.attn.to_q", p1), ("blocks.0.ff.proj_in", p2)] {
            let want = repack_packed_weight_uncached(&wq, &s, &b, 64, &Device::Cpu).unwrap();
            assert_eq!(
                bytes(&got[&format!("{base}.weight")]),
                want.data().unwrap().into_owned()
            );
        }
        assert_eq!(
            read_meta(&tr_out),
            read_meta(&src.join("transformer.safetensors"))
        );
        assert_eq!(read_meta(&tr_out).unwrap()["sceneworks_tier"], "q4");
        // Dense files are byte-for-byte copies; sidecars verbatim.
        for f in [
            "vae_encoder.safetensors",
            "vae_decoder.safetensors",
            "embedded_config.json",
        ] {
            assert_eq!(
                std::fs::read(dst.join(f)).unwrap(),
                std::fs::read(src.join(f)).unwrap(),
                "{f}"
            );
        }
        // Manifest: marker added, rows restated, everything else kept.
        let m = manifest_of(&dst);
        assert_eq!(m[SPLIT_STORAGE_KEY], "ggml");
        for key in [
            "tier",
            "quantized",
            "quantization_bits",
            "quantization_group_size",
        ] {
            assert_eq!(m[key], manifest_of(&src)[key], "{key}");
        }
        let rows = m["component_detail"].as_array().unwrap();
        let mut total = 0;
        for row in rows {
            let file = row["file"].as_str().unwrap();
            let len = std::fs::metadata(dst.join(file)).unwrap().len();
            assert_eq!(row["bytes"], len, "{file}");
            total += len;
        }
        assert_eq!(rows[0]["tensors"], 3);
        assert_eq!(rows[0]["quantized_linears"], 2);
        assert_eq!(rows[1]["dense_reason"], "no-linear-weights");
        assert_eq!(m["total_bytes"], total);
        // No partial left behind; converting a GGML tier again is refused.
        assert!(!root.path().join("out/q4.partial").exists());
        assert!(matches!(
            convert_split_tier(
                &dst,
                &root.path().join("again"),
                &SourceMode::FromMlx,
                &mut noop
            ),
            Err(GgmlTierError::Split { .. })
        ));
    }

    #[test]
    fn split_tier_refuses_a_manifest_whose_packed_count_disagrees() {
        let root = tmp();
        let src = root.path().join("q4");
        let mut tr = HashMap::new();
        put_triple(&mut tr, "a", q4_pack(4, 64, 64, 0));
        tr.insert("n.weight".into(), norm());
        // The manifest declares 2 quantized_linears; the file holds 1.
        split_fixture(&src, "q4", 4, [tr]);
        let dst = root.path().join("out");
        assert!(matches!(
            convert_split_tier(&src, &dst, &SourceMode::FromMlx, &mut noop),
            Err(GgmlTierError::Split { .. })
        ));
        assert!(!dst.exists());
        // A manifest whose width the tier cannot convert from bf16.
        assert!(matches!(
            convert_split_tier(
                &src,
                &root.path().join("o2"),
                &SourceMode::FromBf16(Bf16Source {
                    dir: src.clone(),
                    prefix_maps: vec![]
                }),
                &mut noop
            ),
            Err(GgmlTierError::TemplateNotQ8(4))
        ));
    }

    #[test]
    fn split_tier_q8_from_bf16_header_template_reads_same_named_files_and_keeps_template_stamps() {
        let root = tmp();
        let (q8, bf16) = (root.path().join("q8"), root.path().join("bf16"));
        let mut tr = HashMap::new();
        put_triple(&mut tr, "blocks.0.attn.to_q", q8_template(8, 128, 64));
        put_triple(&mut tr, "blocks.0.ff.proj_in", q8_template(4, 64, 64));
        tr.insert("blocks.0.norm.weight".into(), norm());
        split_fixture(&q8, "q8", 8, [tr]);
        // The bf16 split tier: same file names, dense weights, its own stamps.
        let (w1, w2) = (dense_bf16(8, 128, 0.4), dense_bf16(4, 64, 0.9));
        let mut dense_tr = HashMap::new();
        dense_tr.insert("blocks.0.attn.to_q.weight".into(), w1.clone());
        dense_tr.insert("blocks.0.ff.proj_in.weight".into(), w2.clone());
        dense_tr.insert("blocks.0.norm.weight".into(), norm());
        split_fixture(&bf16, "bf16", 8, [dense_tr]);
        let tpl = root.path().join("tpl");
        dump_headers(&q8, &tpl).unwrap();
        assert!(!tpl.join("transformer.safetensors").exists());

        let dst = root.path().join("out");
        let mode = SourceMode::FromBf16(Bf16Source {
            dir: bf16,
            prefix_maps: vec![],
        });
        let report = convert_split_tier(&tpl, &dst, &mode, &mut noop).unwrap();
        assert_eq!(report.converted, 2);
        let got = load(dst.join("transformer.safetensors"));
        for (name, w) in [
            ("blocks.0.attn.to_q.weight", &w1),
            ("blocks.0.ff.proj_in.weight", &w2),
        ] {
            let want =
                QTensor::quantize(&w.to_dtype(DType::F32).unwrap(), GgmlDType::Q8_0).unwrap();
            assert_eq!(
                bytes(&got[name]),
                want.data().unwrap().into_owned(),
                "{name}"
            );
        }
        // Every file carries the q8 template's stamps, not the bf16 source's.
        for f in [
            "transformer.safetensors",
            "vae_encoder.safetensors",
            "vae_decoder.safetensors",
        ] {
            assert_eq!(
                read_meta(&dst.join(f)).unwrap()["sceneworks_tier"],
                "q8",
                "{f}"
            );
        }
        // `conv_in.weight` exists in both VAE files: each came from its own same-named file.
        for (f, seed) in [("vae_encoder", 0.5f32), ("vae_decoder", 0.25)] {
            let t = &load(dst.join(format!("{f}.safetensors")))["conv_in.weight"];
            assert_eq!(
                t.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                conv(seed).flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                "{f}"
            );
        }
        let m = manifest_of(&dst);
        assert_eq!(m[SPLIT_STORAGE_KEY], "ggml");
        assert_eq!(m["tier"], "q8");
        assert_eq!(m["component_detail"][0]["tensors"], 3);
    }

    #[test]
    fn split_jobs_are_planned_when_no_component_is_named() {
        let jobs = plan_jobs(
            Path::new("s"),
            Path::new("o"),
            &[
                TierSpec::parse("distilled/q4=from-mlx").unwrap(),
                TierSpec::parse("distilled/q8=from-bf16@b/distilled/bf16").unwrap(),
            ],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(jobs.len(), 2);
        assert!(jobs.iter().all(|j| j.split));
        assert_eq!(jobs[0].src, Path::new("s").join("distilled/q4"));
        assert_eq!(jobs[1].dst, Path::new("o").join("distilled/q8"));
        // `.` = the tier directory itself (flat scail2 / SenseNova tiers).
        let flat = plan_jobs(
            Path::new("s"),
            Path::new("o"),
            &[TierSpec::parse("q8=from-bf16@b/bf16").unwrap()],
            &[".".into()],
            &[],
        )
        .unwrap();
        assert_eq!(flat[0].src, Path::new("s").join("q8"));
        assert_eq!(flat[0].dst, Path::new("o").join("q8"));
        assert!(!flat[0].split);
        match &flat[0].mode {
            SourceMode::FromBf16(b) => assert_eq!(b.dir, PathBuf::from("b/bf16")),
            SourceMode::FromMlx => panic!("from-bf16 expected"),
        }
        match &jobs[1].mode {
            SourceMode::FromBf16(b) => assert_eq!(b.dir, PathBuf::from("b/distilled/bf16")),
            SourceMode::FromMlx => panic!("from-bf16 expected"),
        }
    }

    #[test]
    fn spec_parsing_and_job_planning() {
        assert_eq!(
            TierSpec::parse("q4=from-mlx").unwrap(),
            TierSpec {
                tier: "q4".into(),
                bf16_root: None
            }
        );
        let t = TierSpec::parse("distilled/q8=from-bf16@D:\\w\\bf16").unwrap();
        assert_eq!(t.bf16_root, Some(PathBuf::from("D:\\w\\bf16")));
        assert!(TierSpec::parse("q8=from-bf16").is_err());
        assert!(TierSpec::parse("../x=from-mlx").is_err());
        let jobs = plan_jobs(
            Path::new("s"),
            Path::new("o"),
            &[TierSpec::parse("distilled/q4=from-mlx").unwrap()],
            &["transformer".into()],
            &[],
        )
        .unwrap();
        assert_eq!(
            jobs[0].src,
            Path::new("s").join("distilled/q4").join("transformer")
        );
        assert_eq!(
            jobs[0].dst,
            Path::new("o").join("distilled/q4").join("transformer")
        );
        assert!(plan_jobs(Path::new("s"), Path::new("o"), &[], &["/abs".into()], &[]).is_err());
    }
}
