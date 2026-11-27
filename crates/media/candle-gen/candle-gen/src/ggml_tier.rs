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
        "ggml tier: `{tensor}` is a rank-{rank} packed tensor; only rank-2 projections convert"
    )]
    UnsupportedRank { tensor: String, rank: usize },
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
        for (name, v) in obj {
            if name == "__metadata__" {
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

/// All shards of a directory, with a tensor-name index across them.
struct ShardSet {
    shards: Vec<ShardFile>,
    index: HashMap<String, usize>,
}

impl ShardSet {
    fn open(dir: &Path) -> Result<Self> {
        let mut shards = Vec::new();
        let mut index = HashMap::new();
        for path in safetensors_files(dir)? {
            let shard = ShardFile::open(&path)?;
            for name in shard.entries.keys() {
                if index.insert(name.clone(), shards.len()).is_some() {
                    return Err(GgmlTierError::DuplicateDenseTensor(name.clone()));
                }
            }
            shards.push(shard);
        }
        if shards.is_empty() {
            return Err(GgmlTierError::NoShards(dir.to_path_buf()));
        }
        Ok(Self { shards, index })
    }

    fn find(&self, name: &str) -> Option<(&ShardFile, &Entry)> {
        let shard = &self.shards[*self.index.get(name)?];
        Some((shard, &shard.entries[name]))
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

/// Stored GGML block bytes plus their `[rows, cols / 32, block_bytes]` shape.
type Blocks = (Vec<u8>, Vec<usize>);

fn blocks_of(qt: &QTensor, tensor: &str) -> Result<Blocks> {
    let t =
        to_ggml_block_tensor(qt).map_err(|e| GgmlTierError::Candle(format!("{tensor}: {e}")))?;
    let shape = t.dims().to_vec();
    Ok((t.flatten_all()?.to_vec1::<u8>()?, shape))
}

/// Convert one MLX triple into stored GGML blocks. `cfg_bits` (from the component config), when
/// given, must agree with the width the packed shapes imply at `group_size`.
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
    if wq.dtype() != DType::U32 {
        return Err(GgmlTierError::NotPackedCodes {
            tensor: format!("{base}.weight"),
            dtype: format!("{:?}", wq.dtype()),
        });
    }
    if wq.rank() != 2 {
        return Err(GgmlTierError::UnsupportedRank {
            tensor: format!("{base}.weight"),
            rank: wq.rank(),
        });
    }
    let (wq_cols, s_cols) = (wq.dims2()?.1, scales.dims2()?.1);
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
    // The repack receives scales/biases at f32; widening bf16/f16 is exact and the block fields are
    // f16 either way.
    let scales = scales.to_dtype(DType::F32)?;
    let biases = biases.to_dtype(DType::F32)?;
    let qt = repack_packed_weight_uncached(wq, &scales, &biases, group_size, &Device::Cpu)?;
    blocks_of(&qt, base)
}

/// Quantize a dense `[out, in]` weight straight to `Q8_0` blocks. `expected` is the `[out, in]` the
/// MLX template's triple implies.
fn blocks_q8_from_dense(base: &str, dense: &Tensor, expected: (usize, usize)) -> Result<Blocks> {
    let tensor = format!("{base}.weight");
    if !matches!(dense.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        return Err(GgmlTierError::DenseDtype {
            tensor,
            dtype: format!("{:?}", dense.dtype()),
        });
    }
    if dense.dims() != [expected.0, expected.1] {
        return Err(GgmlTierError::DenseShapeMismatch {
            tensor,
            expected: vec![expected.0, expected.1],
            found: dense.dims().to_vec(),
        });
    }
    if !expected.1.is_multiple_of(GGML_BLOCK) {
        return Err(GgmlTierError::InFeaturesNotBlockMultiple {
            tensor,
            in_features: expected.1,
        });
    }
    let f32w = dense.to_dtype(DType::F32)?.contiguous()?;
    let qt = QTensor::quantize(&f32w, GgmlDType::Q8_0)?;
    blocks_of(&qt, base)
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
    src: &Bf16Source,
    dense: &ShardSet,
    report: &mut ComponentReport,
) -> Result<OutTensor> {
    let looked_up = PrefixMap::apply(&src.prefix_maps, name);
    let (dshard, dentry) =
        dense
            .find(&looked_up)
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

fn convert_shard(
    shard: &ShardFile,
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
            SourceMode::FromBf16(src) => {
                // Structure only: the template's shapes, never its data.
                if entry.dtype != "U32" {
                    return Err(GgmlTierError::NotPackedCodes {
                        tensor: name.clone(),
                        dtype: entry.dtype.clone(),
                    });
                }
                if entry.shape.len() != 2 {
                    return Err(GgmlTierError::UnsupportedRank {
                        tensor: name.clone(),
                        rank: entry.shape.len(),
                    });
                }
                let s_shape = &shard.entries[&s_name].shape;
                if s_shape.len() != 2 || s_shape[0] != entry.shape[0] || s_shape[1] == 0 {
                    return Err(GgmlTierError::OrphanTriple {
                        tensor: s_name,
                        detail: format!("scales shape {s_shape:?} does not match `{name}`"),
                    });
                }
                let (out_dim, wq_cols, s_cols) = (entry.shape[0], entry.shape[1], s_shape[1]);
                let inferred = mlx_packed_bits_gs(wq_cols, s_cols, ctx.group_size);
                if inferred != 8 {
                    return Err(GgmlTierError::BitsMismatch {
                        tensor: base.to_string(),
                        inferred,
                        expected: 8,
                        group_size: ctx.group_size,
                    });
                }
                let looked_up = format!("{}.weight", PrefixMap::apply(&src.prefix_maps, base));
                let dense_set = ctx.dense.as_ref().expect("dense source opened");
                let (dshard, dentry) = dense_set.find(&looked_up).ok_or_else(|| {
                    GgmlTierError::MissingDenseSource {
                        tensor: name.clone(),
                        looked_up: looked_up.clone(),
                    }
                })?;
                let dense = dshard.read_tensor(&looked_up)?;
                report.bytes_in += dentry.len();
                blocks_q8_from_dense(base, &dense, (out_dim, s_cols * ctx.group_size))?
            }
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
    let config_path = src.join("config.json");
    if !config_path.is_file() {
        return Err(GgmlTierError::MissingConfig(src.to_path_buf()));
    }
    let config_text = io(std::fs::read_to_string(&config_path), config_path.display())?;
    let (cfg_bits, group_size) = config_quant(&config_text)?;
    if matches!(mode, SourceMode::FromBf16(_)) && cfg_bits != 8 {
        return Err(GgmlTierError::TemplateNotQ8(cfg_bits));
    }
    if matches!(mode, SourceMode::FromMlx) && !group_size.is_multiple_of(GGML_BLOCK) {
        return Err(GgmlTierError::GroupSize { group_size });
    }
    let files = component_files(src)?;
    // Full-tier template: real shards. From-bf16 may instead use a header-only template, one
    // `<shard>.safetensors.header.json` dump per shard (see `dump_headers`).
    let mut template: Vec<(std::ffi::OsString, ShardFile)> = Vec::new();
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
    if template.is_empty() {
        return Err(GgmlTierError::NoShards(src.to_path_buf()));
    }
    let ctx = Ctx {
        mode,
        dense: match mode {
            SourceMode::FromBf16(s) => Some(ShardSet::open(&s.dir)?),
            SourceMode::FromMlx => None,
        },
        group_size,
        cfg_bits,
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
        let out = convert_shard(shard, &ctx, &mut report)?;
        for (key, t) in &out {
            report.tensor_bytes += t.data.len() as u64;
            weight_map.insert(
                key.clone(),
                Value::from(name.to_string_lossy().into_owned()),
            );
        }
        let target = partial.join(name);
        safetensors::serialize_to_file(out, None, &target)
            .map_err(|e| GgmlTierError::Candle(format!("{}: {e}", target.display())))?;
        report.shards += 1;
        progress(&target, &report)?;
    }

    for file in files.iter().filter(|p| !is_shard(p) && !is_header_dump(p)) {
        let name = file.file_name().expect("sidecar file name");
        let target = partial.join(name);
        let text_name = name.to_string_lossy();
        if text_name == "config.json" {
            io(
                std::fs::write(&target, ggml_config(&config_text)?),
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

/// One component build.
#[derive(Clone, Debug)]
pub struct Job {
    pub label: String,
    pub src: PathBuf,
    pub dst: PathBuf,
    pub mode: SourceMode,
}

/// Expand `tiers x components` into jobs: `<src_root>/<tier>/<component>` ->
/// `<out_root>/<tier>/<component>`; a from-bf16 tier reads dense tensors from
/// `<bf16 root>/<component>`.
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
    for t in tiers {
        for c in components {
            let rel = Path::new(&t.tier).join(c);
            jobs.push(Job {
                label: format!("{}/{c}", t.tier),
                src: src_root.join(&rel),
                dst: out_root.join(&rel),
                mode: match &t.bf16_root {
                    None => SourceMode::FromMlx,
                    Some(root) => SourceMode::FromBf16(Bf16Source {
                        dir: root.join(c),
                        prefix_maps: prefix_maps.to_vec(),
                    }),
                },
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
