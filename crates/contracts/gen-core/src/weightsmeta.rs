//! Backend-neutral **weights-metadata** layer (sc-3722): a safetensors header/byte-view reader plus
//! the LoRA / LoKr / LoHa / kohya **string + metadata parsing** that decides *what* an adapter file
//! is and *where* each factor binds — all with zero tensor deps.
//!
//! Two halves:
//! 1. [`CheckpointMeta`] — opens one `.safetensors` file (or a sharded dir) via the neutral
//!    `safetensors` crate and exposes keys, dtypes, shapes, the `__metadata__` map, and raw byte
//!    views, **without materializing tensors**. Candle reads torch safetensors straight through this;
//!    mlx-gen keeps its mlx-rs full-checkpoint loader and uses this for adapter/metadata inspection.
//! 2. The format predicates / factor-suffix tables / rank-alpha parsing / key-alias resolution that
//!    were inline in mlx-gen's `adapters/loader.rs`. The *factor-reconstruction math* (`kron`,
//!    `matmul`) stays in mlx-gen; only the string/metadata logic lives here so a candle adapter
//!    loader reuses it verbatim.
//!
//! Reference: PEFT (`networkType=lokr`, `rank`/`alpha` metadata, `‹path›.lokr_*` factors), LyCORIS
//! third-party LoKr/LoHa (`lokr_*`/`hada_*` factors, optional per-module `.alpha`), and kohya
//! (`lora_unet_<flattened path>.lora_down/up.weight` + `.alpha`).

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};

pub use safetensors::Dtype;
use safetensors::SafeTensors;

use crate::{Error, Result};

/// True when `path`'s file name begins with `.` — a hidden entry that is never a weight shard.
///
/// The case that motivates this: macOS writes an **AppleDouble sidecar** (`._<name>`) alongside a
/// file whenever it must persist extended attributes on a volume with no native xattr support
/// (exFAT/FAT external drives, SMB/NFS shares, cloud-sync folders), and those sidecars also survive
/// a Finder copy or a zip round-trip. `._model.safetensors` has extension `safetensors`, so an
/// extension-only filter admits it; it sorts *before* `model.safetensors`, so it is the first file a
/// sharded loader opens. Its 4-byte magic (`00 05 16 07`) then decodes as a ~2.2 TB safetensors
/// header length and the load dies with `[load_safetensors] Invalid json header length`
/// (SceneWorks#1333). No legitimate shard name starts with `.`, so skipping hidden entries is exact.
pub fn is_hidden_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.'))
}

// =================================================================================================
// CheckpointMeta — neutral safetensors header / byte-view reader.
// =================================================================================================

/// One tensor's neutral description: dtype, shape, and a borrowed view of its raw little-endian bytes
/// (row-major, exactly as stored). The backend lifts these bytes into its own array type.
#[derive(Clone, Copy)]
pub struct TensorView<'a> {
    pub dtype: Dtype,
    pub shape: &'a [usize],
    pub data: &'a [u8],
}

struct TensorLoc {
    shard: usize,
    dtype: Dtype,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// A safetensors checkpoint's **metadata** — keys, dtypes, shapes, byte ranges, and the file's
/// `__metadata__` map — without allocating any tensor. Backed by the owned file buffer(s), so byte
/// views borrow from `self`.
pub struct CheckpointMeta {
    buffers: Vec<Vec<u8>>,
    index: BTreeMap<String, TensorLoc>,
    file_metadata: BTreeMap<String, String>,
}

impl CheckpointMeta {
    /// Open one `.safetensors` file, reading its header (and the whole file buffer) but not parsing
    /// tensors into a tensor library.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let mut me = Self {
            buffers: Vec::new(),
            index: BTreeMap::new(),
            file_metadata: BTreeMap::new(),
        };
        me.add_file(path.as_ref())?;
        Ok(me)
    }

    /// Open and merge every `.safetensors` file under `dir` (sharded checkpoints). Keys are unioned;
    /// on a duplicate key the later file (sorted by path) wins — the same merge semantics as
    /// mlx-gen's `Weights::from_dir`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let mut files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .filter(|p| !is_hidden_file(p))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(Error::Msg(format!(
                "no .safetensors files in {}",
                dir.display()
            )));
        }
        let mut me = Self {
            buffers: Vec::new(),
            index: BTreeMap::new(),
            file_metadata: BTreeMap::new(),
        };
        for f in files {
            me.add_file(&f)?;
        }
        Ok(me)
    }

    fn add_file(&mut self, path: &Path) -> Result<()> {
        // Reads the WHOLE file into `self.buffers` and keeps it — `TensorLoc` indexes into the buffer
        // for lazy random-access tensor reads. This is intended for adapter-sized checkpoints; calling
        // `from_dir` on a multi-GB *base-model* dir would hold every file resident (host RAM). An mmap
        // backing (page on demand) is the efficiency follow-up if base-model use is ever needed (F-038).
        let buf = std::fs::read(path)?;
        // `read_metadata` returns (header_json_len, Metadata); the data region begins at 8 + n and
        // each tensor's data_offsets are relative to it.
        let (n, meta) = SafeTensors::read_metadata(&buf)
            .map_err(|e| Error::Msg(format!("safetensors header in {}: {e}", path.display())))?;
        let data_base = 8 + n;
        let shard = self.buffers.len();
        for (key, info) in meta.tensors() {
            self.index.insert(
                key,
                TensorLoc {
                    shard,
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    start: data_base + info.data_offsets.0,
                    end: data_base + info.data_offsets.1,
                },
            );
        }
        if let Some(kv) = meta.metadata() {
            for (k, v) in kv {
                self.file_metadata.insert(k.clone(), v.clone());
            }
        }
        self.buffers.push(buf);
        Ok(())
    }

    /// Tensor keys, sorted.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }

    /// `true` if `key` is present.
    pub fn contains(&self, key: &str) -> bool {
        self.index.contains_key(key)
    }

    /// A `__metadata__` value (e.g. `networkType`, `rank`, `alpha`), if present.
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.file_metadata.get(key).map(String::as_str)
    }

    /// A tensor's dtype/shape/raw byte view, or `None` if the key is absent.
    pub fn tensor(&self, key: &str) -> Option<TensorView<'_>> {
        self.index.get(key).map(|loc| TensorView {
            dtype: loc.dtype,
            shape: &loc.shape,
            data: &self.buffers[loc.shard][loc.start..loc.end],
        })
    }
}

/// One tensor's header-only footprint. Unlike [`TensorView`], this never retains or reads the data
/// region, so it is safe for multi-gigabyte provider checkpoints used by pre-load admission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafetensorsTensorHeader {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data_bytes: u64,
}

impl SafetensorsTensorHeader {
    /// Whether Candle's shared `Weights` loader casts this source dtype to its requested float
    /// compute dtype. Integer and boolean tensors remain stored-width.
    ///
    /// `F8_E4M3` is deliberately **excluded**: `candle_gen::weights::coerce_float` only casts
    /// `F16 | BF16 | F32 | F64`, so an fp8 payload stays at its stored width and needs an explicit,
    /// route-owned dequantization contract (paired `weight_scale`/`scales` companions) before it can
    /// be priced. Admission guards therefore fail closed on fp8 through this predicate; a route that
    /// genuinely accepts fp8 declares it explicitly, the way
    /// [`crate::encoder_contract`]'s ComfyUI fp8 policy does.
    pub fn is_float(&self) -> bool {
        matches!(
            self.dtype,
            Dtype::F16 | Dtype::BF16 | Dtype::F32 | Dtype::F64
        )
    }

    /// Checked element count described by the tensor shape.
    pub fn element_count(&self) -> Result<u64> {
        self.shape.iter().try_fold(1_u64, |count, dimension| {
            let dimension = u64::try_from(*dimension).map_err(|_| {
                Error::Msg(format!(
                    "tensor {:?} has an unrepresentable dimension",
                    self.name
                ))
            })?;
            count
                .checked_mul(dimension)
                .ok_or_else(|| Error::Msg(format!("tensor {:?} element count overflow", self.name)))
        })
    }

    /// Checked bytes after a loader materializes each logical element at `width` bytes.
    pub fn materialized_bytes(&self, width: u64) -> Result<u64> {
        self.element_count()?.checked_mul(width).ok_or_else(|| {
            Error::Msg(format!(
                "tensor {:?} {width}-byte materialization size overflow",
                self.name
            ))
        })
    }
}

/// Read tensor shapes/dtypes/byte ranges from one safetensors file or one recursively sharded
/// directory without reading tensor data. Directory duplicate-key semantics match
/// [`CheckpointMeta::from_dir`]: files are sorted and the later shard wins. File symlinks are
/// followed (as required by the Hugging Face cache), directory symlinks are skipped, and malformed
/// or unreadable trees fail closed instead of silently producing partial facts.
pub fn safetensors_path_tensor_headers(
    path: impl AsRef<Path>,
) -> Result<Vec<SafetensorsTensorHeader>> {
    const MAX_HEADER_SIZE: u64 = 100_000_000;

    fn read_file(path: &Path) -> Result<Vec<SafetensorsTensorHeader>> {
        let mut file = std::fs::File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut prefix = [0_u8; 8];
        file.read_exact(&mut prefix)?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len > MAX_HEADER_SIZE {
            return Err(Error::Msg(format!(
                "safetensors header in {} exceeds the {MAX_HEADER_SIZE}-byte maximum",
                path.display()
            )));
        }
        let data_start = 8_u64.checked_add(header_len).ok_or_else(|| {
            Error::Msg(format!(
                "safetensors header too large in {}",
                path.display()
            ))
        })?;
        if data_start > file_len {
            return Err(Error::Msg(format!(
                "safetensors header in {} extends past the file",
                path.display()
            )));
        }
        let header_len = usize::try_from(header_len).map_err(|_| {
            Error::Msg(format!(
                "safetensors header too large in {}",
                path.display()
            ))
        })?;
        let mut header = vec![0_u8; header_len];
        file.read_exact(&mut header)?;
        let json: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header)
            .map_err(|error| {
                Error::Msg(format!("safetensors header in {}: {error}", path.display()))
            })?;
        let available = file_len - data_start;
        let mut tensors = json
            .into_iter()
            .filter(|(name, _)| name != "__metadata__")
            .map(|(name, value)| {
                let info: safetensors::tensor::TensorInfo =
                    serde_json::from_value(value).map_err(|error| {
                        Error::Msg(format!(
                            "safetensors tensor {name:?} in {}: {error}",
                            path.display()
                        ))
                    })?;
                let (start, end) = info.data_offsets;
                let end_u64 = u64::try_from(end).map_err(|_| {
                    Error::Msg(format!(
                        "safetensors tensor {name:?} in {} has an unrepresentable end offset",
                        path.display()
                    ))
                })?;
                if start > end || end_u64 > available {
                    return Err(Error::Msg(format!(
                        "safetensors tensor {name:?} in {} has invalid offsets [{start}, {end})",
                        path.display()
                    )));
                }
                let data_bytes = u64::try_from(end - start).map_err(|_| {
                    Error::Msg(format!(
                        "safetensors tensor {name:?} in {} has an unrepresentable byte length",
                        path.display()
                    ))
                })?;
                let element_count = info.shape.iter().try_fold(1_u64, |count, dimension| {
                    let dimension = u64::try_from(*dimension).map_err(|_| {
                        Error::Msg(format!(
                            "safetensors tensor {name:?} in {} has an unrepresentable dimension",
                            path.display()
                        ))
                    })?;
                    count.checked_mul(dimension).ok_or_else(|| {
                        Error::Msg(format!(
                            "safetensors tensor {name:?} in {} has a shape product overflow",
                            path.display()
                        ))
                    })
                })?;
                let expected_bytes = element_count
                    .checked_mul(info.dtype.size() as u64)
                    .ok_or_else(|| {
                        Error::Msg(format!(
                            "safetensors tensor {name:?} in {} has a byte-size overflow",
                            path.display()
                        ))
                    })?;
                if data_bytes != expected_bytes {
                    return Err(Error::Msg(format!(
                        "safetensors tensor {name:?} in {} declares {data_bytes} payload bytes but {:?} {:?} requires {expected_bytes}",
                        path.display(), info.dtype, info.shape
                    )));
                }
                Ok((
                    start,
                    end,
                    SafetensorsTensorHeader {
                        name,
                        dtype: info.dtype,
                        shape: info.shape,
                        data_bytes,
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        tensors.sort_by(|left, right| {
            (left.0, left.1, left.2.name.as_str()).cmp(&(right.0, right.1, right.2.name.as_str()))
        });
        let mut expected_start = 0_usize;
        for (start, end, header) in &tensors {
            if *start != expected_start {
                return Err(Error::Msg(format!(
                    "safetensors tensor {:?} in {} starts at {start}, expected contiguous offset {expected_start}",
                    header.name,
                    path.display()
                )));
            }
            expected_start = *end;
        }
        if u64::try_from(expected_start).ok() != Some(available) {
            return Err(Error::Msg(format!(
                "safetensors tensor payload in {} covers {expected_start} bytes but file contains {available}",
                path.display()
            )));
        }
        Ok(tensors.into_iter().map(|(_, _, header)| header).collect())
    }

    fn collect_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                collect_files(&path, files)?;
                continue;
            }
            let metadata = std::fs::metadata(&path)?;
            if metadata.is_dir() {
                continue;
            }
            if path.extension().and_then(|value| value.to_str()) == Some("safetensors")
                && !is_hidden_file(&path)
            {
                files.push(path);
            }
        }
        Ok(())
    }

    let path = path.as_ref();
    if path.is_file() {
        return read_file(path);
    }
    let mut files = Vec::new();
    collect_files(path, &mut files)?;
    files.sort();
    if files.is_empty() {
        return Err(Error::Msg(format!(
            "no .safetensors files in {}",
            path.display()
        )));
    }
    let mut tensors = BTreeMap::new();
    for file in files {
        for tensor in read_file(&file)? {
            tensors.insert(tensor.name.clone(), tensor);
        }
    }
    Ok(tensors.into_values().collect())
}

/// Sum the on-disk bytes of every `.safetensors` weight file under `dir` (recursively), **without
/// materializing any tensor** — the tensor-free size primitive behind the provider footprint seam
/// (sc-10894, [`crate::registry::PerComponentBytes`]).
///
/// Chosen over reading each file through [`CheckpointMeta`] on purpose: `CheckpointMeta` buffers the
/// WHOLE file in host RAM (the F-038 caveat above), so measuring a multi-GB base component that way
/// would read tens of GB into memory inside a *pre-load* gate. A file-length sum is header-plus-tensor
/// bytes (the few-KB header overhead is negligible for a memory budget), touches no tensor data, and
/// makes **zero** MLX allocation.
///
/// File symlinks are followed (the HF cache stores each shard as a symlink into `blobs/`), while
/// directory symlinks are skipped to prevent a malformed snapshot from creating a recursive cycle.
/// AppleDouble `._*` sidecars and other hidden entries are skipped ([`is_hidden_file`]) — a
/// `._model.safetensors` masquerades as a shard and would double-count. Returns `0` when `dir` is
/// missing or holds no weights, so an absent component contributes nothing. This is the same
/// accounting the worker's whole-model sum uses, so a component's bytes and the whole-model total
/// stay directly comparable (`rest = total − text_encoder`).
pub fn safetensors_dir_bytes(dir: impl AsRef<Path>) -> u64 {
    fn walk(dir: &Path, total: &mut u64) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            // Recurse only into real directories. `metadata()` still follows file symlinks so HF blob
            // links contribute their target length, but a directory link cannot form a cycle here.
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                walk(&path, total);
                continue;
            }
            let Ok(meta) = std::fs::metadata(&path) else {
                continue;
            };
            if meta.is_dir() {
                continue;
            }
            let is_safetensors = path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e == "safetensors");
            if is_safetensors && !is_hidden_file(&path) {
                *total += meta.len();
            }
        }
    }
    let mut total = 0;
    walk(dir.as_ref(), &mut total);
    total
}

/// On-disk `.safetensors` bytes at `path`, whether it is a **directory** (the recursive
/// [`safetensors_dir_bytes`] sum) or a **single file** (its length, if a non-hidden `.safetensors`).
/// The dispatcher behind [`crate::registry::PerComponentBytes::from_root_subdirs`], so a provider can
/// name either a component *subdir* (`"text_encoder"`, `"transformer"`, `"vae"`) or a flat component
/// *file* (`"t5_encoder.safetensors"`, `"low_noise_model.safetensors"` — the bernini / anima layouts,
/// which put components in individual root-level files rather than diffusers subdirs). `0` when `path`
/// is missing or is not a countable weight.
pub fn safetensors_path_bytes(path: impl AsRef<Path>) -> u64 {
    let path = path.as_ref();
    let Ok(meta) = std::fs::metadata(path) else {
        return 0;
    };
    if meta.is_dir() {
        return safetensors_dir_bytes(path);
    }
    let is_safetensors = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e == "safetensors");
    if is_safetensors && !is_hidden_file(path) {
        meta.len()
    } else {
        0
    }
}

// =================================================================================================
// LoRA / LoKr / LoHa / kohya format parsing (string + metadata only).
// =================================================================================================

/// PEFT LoKr per-module factor suffixes; each factor is full (`lokr_w1`/`lokr_w2`) or low-rank
/// (`_a`/`_b`). `.lokr_w1_a`/`_b` precede the bare `.lokr_w1` so exact-suffix matching never mis-binds.
pub const LOKR_SUFFIXES: [&str; 6] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
];

/// Third-party LyCORIS LoKr factor suffixes — the PEFT set plus `lokr_t2` (the tucker/CP factor).
pub const LOKR_TP_SUFFIXES: [&str; 7] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
    ".lokr_t2",
];

/// Third-party LyCORIS LoHa factor suffixes — two low-rank Hadamard pairs + optional tucker `t1`/`t2`.
pub const LOHA_TP_SUFFIXES: [&str; 6] = [
    ".hada_w1_a",
    ".hada_w1_b",
    ".hada_w2_a",
    ".hada_w2_b",
    ".hada_t1",
    ".hada_t2",
];

/// The kohya flattened-path namespace prefix (`lora_unet_<dotted-path-with-dots→underscores>`).
pub const KOHYA_PREFIX: &str = "lora_unet_";

/// The exact kohya network module used by the MiniMax-H3 trainer. This format is intentionally
/// narrower than generic [`KOHYA_PREFIX`] handling: its four fused/raw leaves require numerical
/// conversion before they can bind to the diffusers-shaped runtime DiT.
pub const MINIMAX_H3_TRAINER_NETWORK_MODULE: &str = "networks.lora_minimax_h3";

/// The metadata flag which makes a 50-block, trunk-only H3 adapter intentional rather than partial.
pub const MINIMAX_H3_TOKEN_REFINER_METADATA_KEY: &str = "ss_h3_lora_token_refiner";

/// One of the four raw MiniMax-H3 trainer targets in every transformer block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MiniMaxH3TrainerLeaf {
    AttnQkvProj,
    AttnOutProj,
    MlpFc1,
    MlpFc2,
}

impl MiniMaxH3TrainerLeaf {
    /// The dotted raw-module spelling consumed by the H3 ComfyUI conversion.
    pub const fn dotted(self) -> &'static str {
        match self {
            Self::AttnQkvProj => "attn.qkv_proj",
            Self::AttnOutProj => "attn.out_proj",
            Self::MlpFc1 => "mlp.fc1",
            Self::MlpFc2 => "mlp.fc2",
        }
    }

    const fn flattened(self) -> &'static str {
        match self {
            Self::AttnQkvProj => "attn_qkv_proj",
            Self::AttnOutProj => "attn_out_proj",
            Self::MlpFc1 => "mlp_fc1",
            Self::MlpFc2 => "mlp_fc2",
        }
    }

    const fn geometry(self) -> (usize, usize) {
        match self {
            Self::AttnQkvProj => (5_376, 21_504),
            Self::AttnOutProj => (7_168, 5_376),
            Self::MlpFc1 => (5_376, 28_672),
            Self::MlpFc2 => (14_336, 5_376),
        }
    }
}

/// Which tensor in a raw MiniMax-H3 trainer target a key names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MiniMaxH3TrainerRole {
    Down,
    Up,
    Alpha,
}

impl MiniMaxH3TrainerRole {
    /// The equivalent suffix in the dotted ComfyUI-compatible intermediate key space.
    pub const fn dotted_suffix(self) -> &'static str {
        match self {
            Self::Down => ".lora_down.weight",
            Self::Up => ".lora_up.weight",
            Self::Alpha => ".alpha",
        }
    }
}

/// Parsed identity of one tensor in the exact MiniMax-H3 trainer namespace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MiniMaxH3TrainerKey {
    pub block: usize,
    pub leaf: MiniMaxH3TrainerLeaf,
    pub role: MiniMaxH3TrainerRole,
}

/// Structural receipt for one validated MiniMax-H3 trainer adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MiniMaxH3TrainerLayout {
    /// Source modules, before the fused QKV target expands to three runtime linears.
    pub source_targets: usize,
    /// Distinct factor ranks present in the file, sorted for stable receipts.
    pub ranks: Vec<usize>,
    /// This namespace is trunk-only by construction; the explicit metadata flag makes that intent
    /// machine-readable and prevents an accidentally partial export from looking valid.
    pub trunk_only: bool,
}

/// Parse one exact H3 trainer tensor name:
/// `lora_unet_blocks_{0..49}_{attn_qkv_proj,attn_out_proj,mlp_fc1,mlp_fc2}` followed by
/// `.lora_down.weight`, `.lora_up.weight`, or `.alpha`.
pub fn parse_minimax_h3_trainer_key(key: &str) -> Option<MiniMaxH3TrainerKey> {
    let rest = key.strip_prefix("lora_unet_blocks_")?;
    let digits_len = rest.bytes().take_while(u8::is_ascii_digit).count();
    if digits_len == 0 || rest.as_bytes().get(digits_len) != Some(&b'_') {
        return None;
    }
    let block = rest[..digits_len].parse::<usize>().ok()?;
    let target_and_role = &rest[digits_len + 1..];
    let (role, target) = [
        (MiniMaxH3TrainerRole::Down, ".lora_down.weight"),
        (MiniMaxH3TrainerRole::Up, ".lora_up.weight"),
        (MiniMaxH3TrainerRole::Alpha, ".alpha"),
    ]
    .into_iter()
    .find_map(|(role, suffix)| {
        target_and_role
            .strip_suffix(suffix)
            .map(|target| (role, target))
    })?;
    let leaf = [
        MiniMaxH3TrainerLeaf::AttnQkvProj,
        MiniMaxH3TrainerLeaf::AttnOutProj,
        MiniMaxH3TrainerLeaf::MlpFc1,
        MiniMaxH3TrainerLeaf::MlpFc2,
    ]
    .into_iter()
    .find(|leaf| target == leaf.flattened())?;
    (block < 50).then_some(MiniMaxH3TrainerKey { block, leaf, role })
}

/// Whether the keys or metadata claim the exact MiniMax-H3 trainer namespace. Callers use this
/// before generic kohya handling so the fused QKV and FC1 half-order transforms cannot be skipped.
pub fn is_minimax_h3_trainer_key_space<'a>(
    keys: impl IntoIterator<Item = &'a str>,
    network_module: Option<&str>,
) -> bool {
    network_module.is_some_and(|module| module.trim() == MINIMAX_H3_TRAINER_NETWORK_MODULE)
        || keys
            .into_iter()
            .any(|key| key.starts_with("lora_unet_blocks_"))
}

/// Validate the complete 50-block × four-leaf H3 trainer surface without materializing tensors.
///
/// `entries` supplies `(key, shape)` pairs from either backend or a safetensors header. Every key is
/// required to be in the exact namespace, every source target must carry down/up/alpha, factor
/// geometry and rank must compose, and the intentionally absent token refiner must be declared with
/// `ss_h3_lora_token_refiner=False`. This is deliberately strict only for this trainer namespace;
/// existing diffusers/PEFT, ComfyUI, and LoKr paths never call it.
pub fn validate_minimax_h3_trainer_layout(
    entries: impl IntoIterator<Item = (String, Vec<usize>)>,
    network_module: Option<&str>,
    token_refiner: Option<&str>,
) -> std::result::Result<MiniMaxH3TrainerLayout, String> {
    match network_module.map(str::trim) {
        Some(MINIMAX_H3_TRAINER_NETWORK_MODULE) => {}
        Some(other) => {
            return Err(format!(
                "unsupported MiniMax-H3 trainer namespace {other:?}; expected __metadata__[\"ss_network_module\"] = {MINIMAX_H3_TRAINER_NETWORK_MODULE:?}"
            ));
        }
        None => {
            return Err(format!(
                "MiniMax-H3 trainer keys require __metadata__[\"ss_network_module\"] = {MINIMAX_H3_TRAINER_NETWORK_MODULE:?}"
            ));
        }
    }
    if !token_refiner.is_some_and(|value| value.trim().eq_ignore_ascii_case("false")) {
        return Err(format!(
            "the 50-block MiniMax-H3 trainer adapter omits token-refiner targets, so __metadata__[\"{MINIMAX_H3_TOKEN_REFINER_METADATA_KEY}\"] must be False"
        ));
    }

    let mut groups: BTreeMap<
        (usize, MiniMaxH3TrainerLeaf),
        BTreeMap<MiniMaxH3TrainerRole, Vec<usize>>,
    > = BTreeMap::new();
    for (key, shape) in entries {
        let parsed = parse_minimax_h3_trainer_key(&key).ok_or_else(|| {
            format!(
                "unsupported or malformed MiniMax-H3 trainer tensor {key:?}; expected lora_unet_blocks_{{0..49}}_{{attn_qkv_proj,attn_out_proj,mlp_fc1,mlp_fc2}}.{{lora_down.weight,lora_up.weight,alpha}}"
            )
        })?;
        if groups
            .entry((parsed.block, parsed.leaf))
            .or_default()
            .insert(parsed.role, shape)
            .is_some()
        {
            return Err(format!(
                "duplicate MiniMax-H3 trainer tensor role in {key:?}"
            ));
        }
    }

    let mut ranks = BTreeSet::new();
    for block in 0..50 {
        for leaf in [
            MiniMaxH3TrainerLeaf::AttnQkvProj,
            MiniMaxH3TrainerLeaf::AttnOutProj,
            MiniMaxH3TrainerLeaf::MlpFc1,
            MiniMaxH3TrainerLeaf::MlpFc2,
        ] {
            let target = format!("lora_unet_blocks_{block}_{}", leaf.flattened());
            let roles = groups.get(&(block, leaf)).ok_or_else(|| {
                format!("partial MiniMax-H3 trainer adapter: missing target {target}")
            })?;
            let down = roles.get(&MiniMaxH3TrainerRole::Down).ok_or_else(|| {
                format!("partial MiniMax-H3 trainer adapter: {target} is missing lora_down.weight")
            })?;
            let up = roles.get(&MiniMaxH3TrainerRole::Up).ok_or_else(|| {
                format!("partial MiniMax-H3 trainer adapter: {target} is missing lora_up.weight")
            })?;
            let alpha = roles.get(&MiniMaxH3TrainerRole::Alpha).ok_or_else(|| {
                format!("partial MiniMax-H3 trainer adapter: {target} is missing alpha")
            })?;
            let (in_dim, out_dim) = leaf.geometry();
            if down.len() != 2 || down[0] == 0 || down[1] != in_dim {
                return Err(format!(
                    "malformed MiniMax-H3 trainer tensor {target}.lora_down.weight: expected [rank, {in_dim}], got {down:?}"
                ));
            }
            let rank = down[0];
            if up.as_slice() != [out_dim, rank] {
                return Err(format!(
                    "malformed MiniMax-H3 trainer tensor {target}.lora_up.weight: expected [{out_dim}, {rank}], got {up:?}"
                ));
            }
            if !(alpha.is_empty() || alpha.as_slice() == [1]) {
                return Err(format!(
                    "malformed MiniMax-H3 trainer tensor {target}.alpha: expected scalar [] or [1], got {alpha:?}"
                ));
            }
            ranks.insert(rank);
        }
    }
    if groups.len() != 200 {
        return Err(format!(
            "MiniMax-H3 trainer adapter must contain exactly 200 source targets (50 blocks × four leaves), found {}",
            groups.len()
        ));
    }
    Ok(MiniMaxH3TrainerLayout {
        source_targets: groups.len(),
        ranks: ranks.into_iter().collect(),
        trunk_only: true,
    })
}

/// Common LoRA namespace prefixes a PEFT/diffusers file may carry on its keys (LoKr keys are bare).
pub const COMMON_LORA_PREFIXES: [&str; 2] = ["transformer.", "diffusion_model."];

/// `true` if the file's `networkType` metadata marks it a (PEFT) LoKr adapter.
pub fn is_lokr_network_type(network_type: Option<&str>) -> bool {
    network_type
        .map(|s| s.trim().eq_ignore_ascii_case("lokr"))
        .unwrap_or(false)
}

/// `true` if any key is a LoKr factor (`*.lokr_w…`), regardless of `networkType` metadata — how a
/// **third-party** LyCORIS LoKr is recognized (those files ship the factors but not the PEFT stamp).
pub fn keys_contain_lokr<'a>(mut keys: impl Iterator<Item = &'a str>) -> bool {
    keys.any(|k| k.contains(".lokr_w"))
}

/// `true` if any key is a LoHa factor (`*.hada_w…`). Mutually exclusive with [`keys_contain_lokr`].
pub fn keys_contain_loha<'a>(mut keys: impl Iterator<Item = &'a str>) -> bool {
    keys.any(|k| k.contains(".hada_w"))
}

/// `true` if any key carries the kohya `lora_unet_` prefix (the only convention that flattens the
/// module path; PEFT/diffusers keep dots, LoKr is bare).
pub fn keys_are_kohya<'a>(mut keys: impl Iterator<Item = &'a str>) -> bool {
    keys.any(|k| k.starts_with(KOHYA_PREFIX))
}

/// The [`COMMON_LORA_PREFIXES`] namespace present in `keys`, if any.
pub fn detect_lora_prefix<'a>(keys: impl IntoIterator<Item = &'a str>) -> Option<&'static str> {
    let keys: Vec<&str> = keys.into_iter().collect();
    COMMON_LORA_PREFIXES
        .into_iter()
        .find(|&p| keys.iter().any(|k| k.starts_with(p)))
}

/// Parse the PEFT `(rank, alpha)` from safetensors metadata. `rank` defaults to `1.0`; `alpha`
/// defaults to `rank` (scale 1.0), matching PEFT.
pub fn parse_rank_alpha(rank: Option<&str>, alpha: Option<&str>) -> (f32, f32) {
    // Treat a parsed rank <= 0 the same as absent (→ 1.0): a zero rank would make the downstream
    // `alpha/rank` scale non-finite and NaN-poison the adapter merge (sc-5252/F-002).
    let rank = rank
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|&r| r > 0.0)
        .unwrap_or(1.0);
    let alpha = alpha.and_then(|s| s.parse::<f32>().ok()).unwrap_or(rank);
    (rank, alpha)
}

/// The safetensors `__metadata__` key under which PEFT / diffusers `save_lora_adapter` store the
/// LoRA config blob (sc-5513). Callers pass `meta(LORA_ADAPTER_METADATA_KEY)` to [`LoraAdapterMeta`].
pub const LORA_ADAPTER_METADATA_KEY: &str = "lora_adapter_metadata";

/// Parsed view of the PEFT / diffusers `lora_adapter_metadata` config blob (sc-5513 — the MLX sibling
/// of candle's sc-5374 `LoraAdapterMeta`).
///
/// `peft.save_pretrained()` and diffusers `save_lora_adapter` do **not** write a per-target `.alpha`
/// tensor — the kohya / SceneWorks-trainer convention every inference adapter loader reads first. They
/// store the LoRA scaling inside the safetensors header `__metadata__["lora_adapter_metadata"]`: a JSON
/// blob carrying `lora_alpha`, `r`, and the optional per-module `alpha_pattern` / `rank_pattern`
/// overrides. With no `.alpha` tensor the loaders would otherwise fall back to `alpha = rank` (scale
/// 1.0) and apply such a file at the WRONG strength whenever `lora_alpha ≠ r` (the common `alpha = 2r`
/// / `r/2` / fixed-16 cases). Parsing this blob lets the merge recover the true `(alpha/rank)` scaling.
///
/// Scope is the **LoRA** path: LoKr carries its `rank`/`alpha` as top-level `__metadata__` strings the
/// LoKr loaders already read via [`parse_rank_alpha`], so it is unaffected. The MLX/SceneWorks trainer's
/// own PEFT output writes a per-target `.alpha` tensor (and top-level `rank`/`alpha`, not this blob), so
/// its round-trip is also unaffected — this is purely external/community adapter coverage (same flavor
/// as sc-3671 / the candle sc-5225 / sc-5374 lineage).
#[derive(Debug, Default, Clone)]
pub struct LoraAdapterMeta {
    lora_alpha: Option<f32>,
    r: Option<f32>,
    alpha_pattern: BTreeMap<String, f32>,
    rank_pattern: BTreeMap<String, f32>,
}

impl LoraAdapterMeta {
    /// Parse the `lora_adapter_metadata` JSON `blob` (the value of [`LORA_ADAPTER_METADATA_KEY`] in a
    /// safetensors `__metadata__` map). Returns `None` when the blob is absent (`None` — kohya /
    /// trainer files, which carry a per-target `.alpha` tensor instead) or unparseable — treated as
    /// absent so a malformed blob can never poison the merge (the caller keeps today's `alpha = rank`
    /// default).
    ///
    /// A non-positive `r` / `rank_pattern` value is dropped (treated as absent): it is the scaling
    /// denominator, and a `≤ 0` rank would make `alpha/rank` non-finite and NaN-poison the residual —
    /// same guard as [`parse_rank_alpha`] (sc-5252/F-002). A `0` *alpha* is kept (a legitimate scale-0,
    /// disabled-adapter value).
    pub fn from_metadata(blob: Option<&str>) -> Option<Self> {
        let v: serde_json::Value = serde_json::from_str(blob?).ok()?;
        let num = |x: &serde_json::Value| x.as_f64().map(|f| f as f32);
        let pos = |x: &serde_json::Value| x.as_f64().map(|f| f as f32).filter(|&f| f > 0.0);
        // A `{module: number}` JSON object → `BTreeMap`, applying `keep` per value and skipping any
        // non-numeric / filtered-out entry.
        let pattern = |x: Option<&serde_json::Value>,
                       keep: &dyn Fn(&serde_json::Value) -> Option<f32>|
         -> BTreeMap<String, f32> {
            x.and_then(|v| v.as_object())
                .map(|o| {
                    o.iter()
                        .filter_map(|(k, val)| keep(val).map(|f| (k.clone(), f)))
                        .collect()
                })
                .unwrap_or_default()
        };
        Some(Self {
            lora_alpha: v.get("lora_alpha").and_then(num),
            r: v.get("r").and_then(pos),
            alpha_pattern: pattern(v.get("alpha_pattern"), &num),
            rank_pattern: pattern(v.get("rank_pattern"), &pos),
        })
    }

    /// PEFT per-module override resolution. A single `target_name_key` is the first pattern key (across
    /// `rank_pattern` ∪ `alpha_pattern`) that equals `module_path` or that `module_path` ends with as
    /// `.{key}` — PEFT's `re.match(r".*\.{key}$")` in `LoraModel._create_and_replace`. Deterministic on
    /// overlap (`BTreeMap` key order). `None` ⇒ no per-module override, use the globals.
    fn target_name_key(&self, module_path: &str) -> Option<&str> {
        self.rank_pattern
            .keys()
            .chain(self.alpha_pattern.keys())
            .map(String::as_str)
            .find(|&k| module_path == k || module_path.ends_with(&format!(".{k}")))
    }

    /// The effective `(alpha, rank)` for `module_path`: `alpha_pattern[key] → lora_alpha` and
    /// `rank_pattern[key] → r`, each `None` when unset. The caller uses `alpha` (falling back to the
    /// factor rank) as the numerator; `rank` is honored as the scaling denominator (a well-formed PEFT
    /// file stores `A` as `[r, in]`, so the factor's leading dim already equals this — the metadata
    /// value is used for faithfulness and as the source of truth PEFT itself scales by). Any returned
    /// `rank` is `> 0` ([`Self::from_metadata`] drops non-positive ranks).
    pub fn effective(&self, module_path: &str) -> (Option<f32>, Option<f32>) {
        let key = self.target_name_key(module_path);
        let alpha = key
            .and_then(|k| self.alpha_pattern.get(k))
            .copied()
            .or(self.lora_alpha);
        let rank = key
            .and_then(|k| self.rank_pattern.get(k))
            .copied()
            .or(self.r);
        (alpha, rank)
    }
}

/// Split a factor key into `(module_path, factor_name)` using `suffixes` (exact-suffix match, in
/// order — list `_a`/`_b` before the bare factor). `factor_name` has the leading `.` dropped (e.g.
/// `blk.0.lokr_w1_a` → `("blk.0", "lokr_w1_a")`). `None` if no suffix matches.
pub fn split_factor_key<'a>(key: &'a str, suffixes: &[&str]) -> Option<(&'a str, &'a str)> {
    for suffix in suffixes {
        if let Some(path) = key.strip_suffix(suffix) {
            // Slice the factor name out of `key` (drop the leading '.') so both halves borrow `key`.
            return Some((path, &key[path.len() + 1..]));
        }
    }
    None
}

/// Resolve a third-party flattened module key to a host dotted path. The key is `<PREFIX>_<stem>`
/// where `stem` is the diffusers path with dots flattened to underscores and `PREFIX` varies by
/// trainer (`lora_unet`, `lycoris`, …). Matched prefix-agnostically: `stem` (a `flattened → dotted`
/// table entry) must equal `raw` or be an `_`-delimited suffix of it; the longest such stem wins.
/// An **original-SD / A1111** key (`…_input_blocks_*` / `middle_block` / `output_blocks`) is retried
/// after translating the block marker onward to diffusers naming (sc-6051).
pub fn resolve_lokr_path<'a>(raw: &str, table: &'a BTreeMap<String, String>) -> Option<&'a str> {
    if let Some(dotted) = match_stem_suffix(raw, table) {
        return Some(dotted);
    }
    let translated = translate_original_sd_marker(raw)?;
    match_stem_suffix(&translated, table)
}

/// The longest `_`-delimited `table` stem that is a suffix of (or equal to) `raw`, mapped to its
/// dotted path. The matching half of [`resolve_lokr_path`], split out so the original-SD retry can
/// reuse it on a translated key.
fn match_stem_suffix<'a>(raw: &str, table: &'a BTreeMap<String, String>) -> Option<&'a str> {
    let mut best: Option<(&str, usize)> = None;
    for (stem, dotted) in table {
        let is_match = raw == stem
            || (raw.len() > stem.len()
                && raw.ends_with(stem.as_str())
                && raw.as_bytes()[raw.len() - stem.len() - 1] == b'_');
        let longer = match best {
            None => true,
            Some((_, l)) => stem.len() > l,
        };
        if is_match && longer {
            best = Some((dotted.as_str(), stem.len()));
        }
    }
    best.map(|(d, _)| d)
}

/// Build the kohya `flattened-stem → dotted-path` lookup from a host's routable target paths. The
/// stem is the dotted path with `.`→`_` (the kohya flattening), WITHOUT the `lora_unet_` prefix.
pub fn kohya_table(paths: &[String]) -> BTreeMap<String, String> {
    paths
        .iter()
        .map(|p| (p.replace('.', "_"), p.clone()))
        .collect()
}

/// SDXL **original-SD / A1111 → diffusers** UNet block-prefix map (flattened, `.`→`_`). kohya LoRAs
/// trained against the LDM/original-SD SDXL UNet name modules `lora_unet_input_blocks_*` /
/// `middle_block_*` / `output_blocks_*`, while diffusers (`pipe.save_lora_weights()`) names the *same*
/// modules `down_blocks_*` / `mid_block_*` / `up_blocks_*`. It is one network: the attention sub-path
/// (`transformer_blocks_*`, `attn1/2_to_*`, `proj_in/out`, `ff_net_*`) is byte-identical, so only the
/// block prefix (plus the resnet/sampler conv leaves — see [`SDXL_LEAF_MAP`]) differs. Entries follow
/// the stock SDXL structure: down = (DownBlock2D, CrossAttnDownBlock2D×2), mid = resnet/attn/resnet,
/// up = (CrossAttnUpBlock2D×2, UpBlock2D), 2 layers/block. The subset of diffusers'
/// `convert_ldm_unet_checkpoint` map an SDXL UNet actually exercises.
const SDXL_ORIGINAL_TO_DIFFUSERS_BLOCK: [(&str, &str); 35] = [
    // input_blocks (down path); input_blocks.0.0 is conv_in.
    ("input_blocks_0_0", "conv_in"),
    ("input_blocks_1_0", "down_blocks_0_resnets_0"),
    ("input_blocks_2_0", "down_blocks_0_resnets_1"),
    ("input_blocks_3_0", "down_blocks_0_downsamplers_0"),
    ("input_blocks_4_0", "down_blocks_1_resnets_0"),
    ("input_blocks_4_1", "down_blocks_1_attentions_0"),
    ("input_blocks_5_0", "down_blocks_1_resnets_1"),
    ("input_blocks_5_1", "down_blocks_1_attentions_1"),
    ("input_blocks_6_0", "down_blocks_1_downsamplers_0"),
    ("input_blocks_7_0", "down_blocks_2_resnets_0"),
    ("input_blocks_7_1", "down_blocks_2_attentions_0"),
    ("input_blocks_8_0", "down_blocks_2_resnets_1"),
    ("input_blocks_8_1", "down_blocks_2_attentions_1"),
    // middle_block (mid path).
    ("middle_block_0", "mid_block_resnets_0"),
    ("middle_block_1", "mid_block_attentions_0"),
    ("middle_block_2", "mid_block_resnets_1"),
    // output_blocks (up path); the upsample conv is the last sub-module of each attention up block.
    ("output_blocks_0_0", "up_blocks_0_resnets_0"),
    ("output_blocks_0_1", "up_blocks_0_attentions_0"),
    ("output_blocks_1_0", "up_blocks_0_resnets_1"),
    ("output_blocks_1_1", "up_blocks_0_attentions_1"),
    ("output_blocks_2_0", "up_blocks_0_resnets_2"),
    ("output_blocks_2_1", "up_blocks_0_attentions_2"),
    ("output_blocks_2_2", "up_blocks_0_upsamplers_0"),
    ("output_blocks_3_0", "up_blocks_1_resnets_0"),
    ("output_blocks_3_1", "up_blocks_1_attentions_0"),
    ("output_blocks_4_0", "up_blocks_1_resnets_1"),
    ("output_blocks_4_1", "up_blocks_1_attentions_1"),
    ("output_blocks_5_0", "up_blocks_1_resnets_2"),
    ("output_blocks_5_1", "up_blocks_1_attentions_2"),
    ("output_blocks_5_2", "up_blocks_1_upsamplers_0"),
    ("output_blocks_6_0", "up_blocks_2_resnets_0"),
    ("output_blocks_7_0", "up_blocks_2_resnets_1"),
    ("output_blocks_8_0", "up_blocks_2_resnets_2"),
    // final out.0/out.2 (group-norm / conv_out).
    ("out_0", "conv_norm_out"),
    ("out_2", "conv_out"),
];

/// SDXL resnet / sampler leaf-name remap (original-SD → diffusers), applied *after* the block prefix
/// (see [`SDXL_ORIGINAL_TO_DIFFUSERS_BLOCK`]). A ResBlock's `in_layers.2` / `emb_layers.1` /
/// `out_layers.3` / `skip_connection` are diffusers' `conv1` / `time_emb_proj` / `conv2` /
/// `conv_shortcut`; a Downsample's `op` conv is diffusers' `conv`. The norm leaves (`in_layers.0` /
/// `out_layers.0`) are not LoRA targets but remap for completeness. None of these substrings occur in
/// the attention sub-path, so a plain substring replace on the post-prefix stem is unambiguous.
const SDXL_LEAF_MAP: [(&str, &str); 7] = [
    ("_in_layers_0", "_norm1"),
    ("_in_layers_2", "_conv1"),
    ("_emb_layers_1", "_time_emb_proj"),
    ("_out_layers_0", "_norm2"),
    ("_out_layers_3", "_conv2"),
    ("_skip_connection", "_conv_shortcut"),
    ("_downsamplers_0_op", "_downsamplers_0_conv"),
];

/// Translate a flattened **original-SD / A1111** SDXL UNet stem (the kohya `lora_unet_` prefix already
/// stripped) to its **diffusers-block** equivalent, or `None` if `stem` is not original-SD naming
/// (already diffusers, a text-encoder key, or unknown). Block prefix via
/// `SDXL_ORIGINAL_TO_DIFFUSERS_BLOCK` (longest match — `input_blocks_4_1` over any shorter prefix),
/// then the resnet/sampler leaf remap (`SDXL_LEAF_MAP`). The attention sub-path is carried through
/// unchanged (identical in both layouts), so the result resolves against the same diffusers
/// `flattened → dotted` table the diffusers-named LoRAs use (sc-6051).
pub fn original_sd_to_diffusers_stem(stem: &str) -> Option<String> {
    // The map keys are disjoint `_`-delimited block paths, so at most one matches at the boundary;
    // `max_by_key` is defensive (longest wins) — e.g. `out_2` never matches an `output_blocks_*` stem.
    let (orig, diff) = SDXL_ORIGINAL_TO_DIFFUSERS_BLOCK
        .iter()
        .filter(|(orig, _)| {
            stem == *orig
                || stem
                    .strip_prefix(orig)
                    .is_some_and(|rest| rest.starts_with('_'))
        })
        .max_by_key(|(orig, _)| orig.len())?;
    let mut out = format!("{diff}{}", &stem[orig.len()..]);
    for (from, to) in SDXL_LEAF_MAP {
        if out.contains(from) {
            out = out.replace(from, to);
        }
    }
    Some(out)
}

/// Resolve a flattened kohya stem (`lora_unet_` prefix already stripped) to a host dotted path: a
/// direct `table` hit, else translate an **original-SD / A1111** stem to diffusers naming
/// ([`original_sd_to_diffusers_stem`]) and retry. Returns the owned dotted path. This is what lets a
/// civitai/A1111 SDXL LoRA (`lora_unet_input_blocks_*`) merge onto the same UNet as a diffusers-named
/// one (sc-6051).
pub fn resolve_kohya_stem(stem: &str, table: &BTreeMap<String, String>) -> Option<String> {
    if let Some(dotted) = table.get(stem) {
        return Some(dotted.clone());
    }
    let diffusers = original_sd_to_diffusers_stem(stem)?;
    table.get(&diffusers).cloned()
}

/// If `raw` (a third-party flattened key with an arbitrary trainer prefix) contains an original-SD
/// SDXL block marker, return `raw` with the marker-onward substring translated to diffusers naming.
fn translate_original_sd_marker(raw: &str) -> Option<String> {
    for marker in ["input_blocks_", "middle_block_", "output_blocks_"] {
        if let Some(idx) = raw.find(marker) {
            let translated = original_sd_to_diffusers_stem(&raw[idx..])?;
            return Some(format!("{}{translated}", &raw[..idx]));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView as StTensorView;

    /// A fixture root that removes itself on `Drop` (sc-17755).
    ///
    /// These tests used to build `temp_dir().join(format!("{prefix}{pid}"))` by hand. That is
    /// unique per *process*, not per call, and the trailing `remove_dir_all(...).ok()` lines were
    /// skipped entirely by a panicking test — so every failing run, and every run whose PID was new,
    /// left its tree behind (104 of them under `%TEMP%` on the CUDA box). `TempDir` removes the tree
    /// on `Drop`, including during unwind. Bind it for the whole test: `fixture_dir(..).path()`
    /// unbound is dropped at the end of the enclosing statement.
    fn fixture_dir(prefix: &str) -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix(prefix)
            .tempdir()
            .expect("fixture temp dir")
    }

    /// Guards the sc-17755 fix: the fixture root, and anything written into it, leave with the
    /// guard. Swap `fixture_dir` back for a bare `create_dir_all` on a `temp_dir()` join and this
    /// goes RED — which is the whole reason the leak went unnoticed for so long.
    #[test]
    fn fixture_dir_is_removed_on_drop() {
        let guard = fixture_dir("gencore_drop_guard_");
        let root = guard.path().to_path_buf();
        let file = root.join("w.safetensors");
        std::fs::write(&file, b"bytes").unwrap();
        assert!(file.is_file());
        drop(guard);
        assert!(!file.exists(), "fixture file survived: {}", file.display());
        assert!(!root.exists(), "fixture root survived: {}", root.display());
    }

    /// `is_float` answers exactly one question: does `candle_gen::weights::coerce_float` cast this
    /// source dtype to the requested compute dtype? Every admission guard prices a "float" tensor at
    /// the compute width and refuses everything else, so widening this set silently reprices — and
    /// silently *admits* — checkpoints those guards were written to reject.
    ///
    /// `F8_E4M3` was widened in once (f9dfb4785) with no consumer and reverted here: fp8 stays at
    /// stored width through `coerce_float`, so admitting it here would let an fp8 checkpoint through
    /// a guard that then mis-prices it. Routes that really accept fp8 opt in explicitly (see
    /// `encoder_contract`'s ComfyUI fp8 policy).
    ///
    /// The expectation is derived from an exhaustive dtype sweep rather than pinned as a count, so a
    /// new safetensors dtype is classified deliberately instead of quietly inheriting `false`.
    #[test]
    fn is_float_admits_only_candle_cast_float_dtypes() {
        const ALL_DTYPES: &[Dtype] = &[
            Dtype::BOOL,
            Dtype::U8,
            Dtype::I8,
            Dtype::F8_E5M2,
            Dtype::F8_E4M3,
            Dtype::I16,
            Dtype::U16,
            Dtype::F16,
            Dtype::BF16,
            Dtype::I32,
            Dtype::U32,
            Dtype::F32,
            Dtype::F64,
            Dtype::I64,
            Dtype::U64,
        ];

        let admitted = ALL_DTYPES
            .iter()
            .copied()
            .filter(|dtype| {
                SafetensorsTensorHeader {
                    name: "w".to_owned(),
                    dtype: *dtype,
                    shape: vec![1],
                    data_bytes: dtype.size() as u64,
                }
                .is_float()
            })
            .collect::<Vec<_>>();

        assert_eq!(
            admitted,
            vec![Dtype::F16, Dtype::BF16, Dtype::F32, Dtype::F64],
            "is_float must admit exactly the dtypes coerce_float casts"
        );
    }

    #[test]
    fn lokr_network_type_predicate() {
        assert!(is_lokr_network_type(Some("lokr")));
        assert!(is_lokr_network_type(Some("  LoKr ")));
        assert!(!is_lokr_network_type(Some("loha")));
        assert!(!is_lokr_network_type(None));
    }

    #[test]
    fn rank_alpha_defaults() {
        assert_eq!(parse_rank_alpha(Some("16"), Some("8")), (16.0, 8.0));
        // alpha defaults to rank (scale 1.0).
        assert_eq!(parse_rank_alpha(Some("16"), None), (16.0, 16.0));
        // rank defaults to 1.0.
        assert_eq!(parse_rank_alpha(None, None), (1.0, 1.0));
        // A parsed rank <= 0 is treated as absent (→ 1.0) so the downstream alpha/rank scale stays
        // finite rather than NaN-poisoning the merge (sc-5252/F-002). alpha then defaults to 1.0.
        assert_eq!(parse_rank_alpha(Some("0"), None), (1.0, 1.0));
        assert_eq!(parse_rank_alpha(Some("0"), Some("8")), (1.0, 8.0));
        assert_eq!(parse_rank_alpha(Some("-4"), None), (1.0, 1.0));
    }

    /// sc-5513: a diffusers / PEFT `lora_adapter_metadata` blob with `lora_alpha ≠ r` parses to the
    /// global pair, so the loaders can recover the true `(alpha/rank)` scaling (the story example:
    /// `lora_alpha = 16`, `r = 8` ⇒ scale 2.0). The pair applies to any module with no override.
    #[test]
    fn lora_adapter_meta_parses_global_alpha_rank() {
        let blob = r#"{"lora_alpha": 16, "r": 8, "target_modules": ["to_q", "to_v"]}"#;
        let cfg = LoraAdapterMeta::from_metadata(Some(blob)).expect("blob must parse");
        assert_eq!(
            cfg.effective("transformer_blocks.0.attn.to_q"),
            (Some(16.0), Some(8.0))
        );
    }

    /// sc-5513: PEFT `alpha_pattern` / `rank_pattern` override the globals for a module whose dotted
    /// path ends with the pattern key (`re.match(r".*\.{key}$")`); a non-matching module keeps the
    /// globals. Mirrors PEFT's single-`target_name_key` resolution.
    #[test]
    fn lora_adapter_meta_honors_per_module_patterns() {
        let blob = r#"{"lora_alpha": 8, "r": 8,
                       "alpha_pattern": {"to_q": 32},
                       "rank_pattern": {"to_q": 16}}"#;
        let cfg = LoraAdapterMeta::from_metadata(Some(blob)).unwrap();
        // `…attn.to_q` ends with `.to_q` → the override pair.
        assert_eq!(
            cfg.effective("transformer_blocks.0.attn.to_q"),
            (Some(32.0), Some(16.0))
        );
        // `…attn.to_k` matches no pattern → the globals.
        assert_eq!(
            cfg.effective("transformer_blocks.0.attn.to_k"),
            (Some(8.0), Some(8.0))
        );
    }

    /// sc-5513: an absent blob (`None` — kohya / trainer files) and a malformed blob both yield `None`,
    /// so the caller falls back to today's per-target-`.alpha`-or-`rank` behavior rather than erroring.
    #[test]
    fn lora_adapter_meta_absent_or_malformed_is_none() {
        assert!(LoraAdapterMeta::from_metadata(None).is_none());
        assert!(LoraAdapterMeta::from_metadata(Some("{not valid json")).is_none());
    }

    /// sc-5513: a non-positive `r` / `rank_pattern` value is dropped (treated as absent) so the scaling
    /// denominator stays `> 0` and `alpha/rank` can never NaN-poison the merge (sc-5252/F-002); the
    /// caller then falls back to the factor's leading dim. A `0` *alpha* is a legitimate scale-0 value
    /// and is kept.
    #[test]
    fn lora_adapter_meta_drops_nonpositive_rank() {
        let cfg = LoraAdapterMeta::from_metadata(Some(
            r#"{"lora_alpha": 16, "r": 0, "rank_pattern": {"to_q": -4}}"#,
        ))
        .unwrap();
        // r = 0 dropped → rank None (caller uses the factor leading dim); alpha still recovered.
        assert_eq!(cfg.effective("blocks.0.to_k"), (Some(16.0), None));
        // rank_pattern[to_q] = -4 dropped → falls back to the (absent) global r → None.
        assert_eq!(cfg.effective("blocks.0.to_q"), (Some(16.0), None));
        // A scale-0 (disabled) adapter alpha is preserved.
        let z = LoraAdapterMeta::from_metadata(Some(r#"{"lora_alpha": 0, "r": 8}"#)).unwrap();
        assert_eq!(z.effective("blocks.0.to_q"), (Some(0.0), Some(8.0)));
    }

    #[test]
    fn key_predicates() {
        let lokr = ["blk.0.lokr_w1_a", "blk.0.lokr_w1_b"];
        let loha = ["blk.0.hada_w1_a"];
        let kohya = ["lora_unet_down_blocks_0.lora_down.weight"];
        assert!(keys_contain_lokr(lokr.iter().copied()));
        assert!(!keys_contain_loha(lokr.iter().copied()));
        assert!(keys_contain_loha(loha.iter().copied()));
        assert!(keys_are_kohya(kohya.iter().copied()));
        assert_eq!(
            detect_lora_prefix(["transformer.blk.0.attn"].into_iter()),
            Some("transformer.")
        );
        assert_eq!(detect_lora_prefix(["bare.key"].into_iter()), None);
    }

    fn minimax_h3_trainer_entries() -> Vec<(String, Vec<usize>)> {
        let mut entries = Vec::new();
        for block in 0..50 {
            for (leaf, input, output) in [
                ("attn_qkv_proj", 5_376, 21_504),
                ("attn_out_proj", 7_168, 5_376),
                ("mlp_fc1", 5_376, 28_672),
                ("mlp_fc2", 14_336, 5_376),
            ] {
                let stem = format!("lora_unet_blocks_{block}_{leaf}");
                entries.push((format!("{stem}.lora_down.weight"), vec![16, input]));
                entries.push((format!("{stem}.lora_up.weight"), vec![output, 16]));
                entries.push((format!("{stem}.alpha"), Vec::new()));
            }
        }
        entries
    }

    #[test]
    fn minimax_h3_trainer_parser_is_exact_and_layout_is_complete() {
        assert_eq!(
            parse_minimax_h3_trainer_key("lora_unet_blocks_49_attn_qkv_proj.lora_down.weight"),
            Some(MiniMaxH3TrainerKey {
                block: 49,
                leaf: MiniMaxH3TrainerLeaf::AttnQkvProj,
                role: MiniMaxH3TrainerRole::Down,
            })
        );
        for malformed in [
            "lora_unet_blocks_50_attn_qkv_proj.lora_down.weight",
            "lora_unet_blocks_0_attn_q_proj.lora_down.weight",
            "lora_unet_blocks_0_mlp_fc1.lora_mid.weight",
            "lora_unet_blocks_x_mlp_fc2.alpha",
        ] {
            assert_eq!(
                parse_minimax_h3_trainer_key(malformed),
                None,
                "{malformed} must not be widened into the exact trainer namespace"
            );
        }

        let layout = validate_minimax_h3_trainer_layout(
            minimax_h3_trainer_entries(),
            Some(MINIMAX_H3_TRAINER_NETWORK_MODULE),
            Some("False"),
        )
        .expect("the exact 50-block trunk-only layout");
        assert_eq!(layout.source_targets, 200);
        assert_eq!(layout.ranks, vec![16]);
        assert!(layout.trunk_only);
    }

    #[test]
    fn minimax_h3_trainer_layout_rejects_independent_partial_shape_and_namespace_mutations() {
        let base = minimax_h3_trainer_entries();

        let mut partial = base.clone();
        partial.retain(|(key, _)| key != "lora_unet_blocks_7_mlp_fc2.alpha");
        let error = validate_minimax_h3_trainer_layout(
            partial,
            Some(MINIMAX_H3_TRAINER_NETWORK_MODULE),
            Some("False"),
        )
        .unwrap_err();
        assert!(
            error.contains("blocks_7_mlp_fc2") && error.contains("missing alpha"),
            "{error}"
        );

        let mut wrong_shape = base.clone();
        wrong_shape
            .iter_mut()
            .find(|(key, _)| key == "lora_unet_blocks_3_attn_qkv_proj.lora_up.weight")
            .unwrap()
            .1 = vec![21_503, 16];
        let error = validate_minimax_h3_trainer_layout(
            wrong_shape,
            Some(MINIMAX_H3_TRAINER_NETWORK_MODULE),
            Some("False"),
        )
        .unwrap_err();
        assert!(
            error.contains("[21504, 16]") && error.contains("[21503, 16]"),
            "{error}"
        );

        for (network, token_refiner, expected) in [
            (Some("networks.lora"), Some("False"), "unsupported"),
            (
                Some(MINIMAX_H3_TRAINER_NETWORK_MODULE),
                None,
                "must be False",
            ),
            (
                Some(MINIMAX_H3_TRAINER_NETWORK_MODULE),
                Some("True"),
                "must be False",
            ),
        ] {
            let error = validate_minimax_h3_trainer_layout(base.clone(), network, token_refiner)
                .unwrap_err();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn factor_key_split() {
        // `_a`/`_b` precede the bare factor → never mis-binds.
        assert_eq!(
            split_factor_key("a.b.lokr_w1_a", &LOKR_SUFFIXES),
            Some(("a.b", "lokr_w1_a"))
        );
        assert_eq!(
            split_factor_key("a.b.lokr_w2", &LOKR_SUFFIXES),
            Some(("a.b", "lokr_w2"))
        );
        assert_eq!(split_factor_key("a.b.weight", &LOKR_SUFFIXES), None);
    }

    #[test]
    fn lokr_path_resolution_longest_stem_wins() {
        let mut table = BTreeMap::new();
        table.insert("blocks_0_attn".to_string(), "blocks.0.attn".to_string());
        table.insert("attn".to_string(), "attn".to_string());
        // `<PREFIX>_blocks_0_attn` matches the longer stem, not the short `attn` suffix.
        assert_eq!(
            resolve_lokr_path("lora_unet_blocks_0_attn", &table),
            Some("blocks.0.attn")
        );
        assert_eq!(resolve_lokr_path("lycoris_attn", &table), Some("attn"));
        assert_eq!(resolve_lokr_path("lora_unet_unknown", &table), None);
    }

    /// sc-6051: an original-SD / A1111 SDXL stem translates to its diffusers-block equivalent. The
    /// attention sub-path rides through unchanged; only the block prefix changes.
    #[test]
    fn original_sd_stem_translates_attention_blocks() {
        // input_blocks.4.1 → down_blocks.1.attentions.0 (the canonical InstantID/civitai case).
        assert_eq!(
            original_sd_to_diffusers_stem("input_blocks_4_1_transformer_blocks_0_attn1_to_q")
                .as_deref(),
            Some("down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q")
        );
        // middle_block.1 → mid_block.attentions.0; output_blocks.5.1 → up_blocks.1.attentions.2.
        assert_eq!(
            original_sd_to_diffusers_stem("middle_block_1_proj_in").as_deref(),
            Some("mid_block_attentions_0_proj_in")
        );
        assert_eq!(
            original_sd_to_diffusers_stem("output_blocks_5_1_transformer_blocks_0_ff_net_0_proj")
                .as_deref(),
            Some("up_blocks_1_attentions_2_transformer_blocks_0_ff_net_0_proj")
        );
        // to_out.0 flattening (`to_out_0`) survives the prefix swap intact.
        assert_eq!(
            original_sd_to_diffusers_stem("output_blocks_2_1_transformer_blocks_0_attn2_to_out_0")
                .as_deref(),
            Some("up_blocks_0_attentions_2_transformer_blocks_0_attn2_to_out_0")
        );
        // An already-diffusers stem (or a text-encoder / unknown key) is not translated.
        assert!(original_sd_to_diffusers_stem(
            "down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q"
        )
        .is_none());
        assert!(
            original_sd_to_diffusers_stem("text_model_encoder_layers_0_self_attn_q_proj").is_none()
        );
    }

    /// sc-6051: the resnet / sampler leaf remap (the convs an A1111 LoRA may also train), applied
    /// after the block prefix.
    #[test]
    fn original_sd_stem_translates_resnet_and_sampler_leaves() {
        // ResBlock conv1/time_emb_proj/conv2/conv_shortcut.
        assert_eq!(
            original_sd_to_diffusers_stem("input_blocks_4_0_in_layers_2").as_deref(),
            Some("down_blocks_1_resnets_0_conv1")
        );
        assert_eq!(
            original_sd_to_diffusers_stem("input_blocks_4_0_emb_layers_1").as_deref(),
            Some("down_blocks_1_resnets_0_time_emb_proj")
        );
        assert_eq!(
            original_sd_to_diffusers_stem("output_blocks_0_0_out_layers_3").as_deref(),
            Some("up_blocks_0_resnets_0_conv2")
        );
        assert_eq!(
            original_sd_to_diffusers_stem("input_blocks_4_0_skip_connection").as_deref(),
            Some("down_blocks_1_resnets_0_conv_shortcut")
        );
        // Downsample `op` conv → diffusers `conv`; Upsample conv already matches.
        assert_eq!(
            original_sd_to_diffusers_stem("input_blocks_3_0_op").as_deref(),
            Some("down_blocks_0_downsamplers_0_conv")
        );
        assert_eq!(
            original_sd_to_diffusers_stem("output_blocks_2_2_conv").as_deref(),
            Some("up_blocks_0_upsamplers_0_conv")
        );
        // Top-level conv_in / conv_out.
        assert_eq!(
            original_sd_to_diffusers_stem("input_blocks_0_0").as_deref(),
            Some("conv_in")
        );
        assert_eq!(
            original_sd_to_diffusers_stem("out_2").as_deref(),
            Some("conv_out")
        );
    }

    /// sc-6051: `resolve_kohya_stem` hits the table directly for a diffusers stem, and via translation
    /// for an original-SD one — both landing on the same dotted path. `resolve_lokr_path` likewise
    /// resolves an original-SD-named third-party key.
    #[test]
    fn resolve_kohya_and_lokr_accept_original_sd_naming() {
        let dotted = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q";
        let table = kohya_table(&[dotted.to_string()]);
        // diffusers stem → direct hit.
        assert_eq!(
            resolve_kohya_stem(
                "down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q",
                &table
            )
            .as_deref(),
            Some(dotted)
        );
        // original-SD stem → translated hit on the SAME dotted path.
        assert_eq!(
            resolve_kohya_stem("input_blocks_4_1_transformer_blocks_0_attn1_to_q", &table)
                .as_deref(),
            Some(dotted)
        );
        assert!(resolve_kohya_stem("lora_te1_unknown", &table).is_none());
        // third-party LoKr/LoHa path: a prefixed original-SD key resolves through resolve_lokr_path.
        assert_eq!(
            resolve_lokr_path(
                "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q",
                &table
            ),
            Some(dotted)
        );
    }

    #[test]
    fn checkpoint_meta_reads_keys_dtype_shape_and_bytes() {
        // Serialize a tiny safetensors file, reopen it through CheckpointMeta, and assert the header
        // view + byte slice round-trip without a tensor library.
        let data: Vec<u8> = (0u8..16).collect(); // 4×i32 = 16 bytes
        let tv = StTensorView::new(Dtype::I32, vec![2, 2], &data).unwrap();
        let bytes = safetensors::serialize([("blk.weight", tv)], &None).unwrap();

        let tmp = fixture_dir("gencore_meta_");
        let dir = tmp.path();
        let path = dir.join("w.safetensors");
        std::fs::write(&path, &bytes).unwrap();

        let meta = CheckpointMeta::from_file(&path).unwrap();
        assert_eq!(meta.keys().collect::<Vec<_>>(), vec!["blk.weight"]);
        let t = meta.tensor("blk.weight").unwrap();
        assert_eq!(t.dtype, Dtype::I32);
        assert_eq!(t.shape, &[2, 2]);
        assert_eq!(t.data, &data[..]);
        assert!(meta.tensor("missing").is_none());
    }

    #[test]
    fn header_only_reader_rejects_oversized_sparse_header_before_allocation() {
        const OVERSIZED_HEADER: u64 = 100_000_001;
        let tmp = fixture_dir("gencore_oversized_header_");
        let path = tmp.path().join("w.safetensors");
        std::fs::write(&path, OVERSIZED_HEADER.to_le_bytes()).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(8 + OVERSIZED_HEADER)
            .unwrap();

        let error = safetensors_path_tensor_headers(&path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("100000000-byte maximum"), "{error}");
    }

    #[test]
    fn header_only_reader_rejects_non_contiguous_or_unowned_payload_bytes() {
        let tmp = fixture_dir("gencore_invalid_tensor_topology_");
        let root = tmp.path();
        let write_raw = |name: &str, raw_header: &str, payload_bytes: usize| {
            let path = root.join(name);
            let mut header = raw_header.as_bytes().to_vec();
            while !header.len().is_multiple_of(8) {
                header.push(b' ');
            }
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header);
            bytes.extend(vec![0_u8; payload_bytes]);
            std::fs::write(&path, bytes).unwrap();
            path
        };

        let overlap = write_raw(
            "overlap.safetensors",
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}"#,
            6,
        );
        let gap = write_raw(
            "gap.safetensors",
            r#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},"b":{"dtype":"F16","shape":[1],"data_offsets":[4,6]}}"#,
            6,
        );
        let trailing = write_raw(
            "trailing.safetensors",
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            8,
        );

        for path in [overlap, gap, trailing] {
            assert!(
                safetensors_path_tensor_headers(&path).is_err(),
                "{} must fail closed",
                path.display()
            );
        }
    }

    #[test]
    fn header_only_reader_rejects_shape_dtype_payload_mismatch() {
        let tmp = fixture_dir("gencore_header_shape_bytes_");
        let path = tmp.path().join("wrong-bytes.safetensors");
        let header = br#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0_u8; 4]);
        std::fs::write(&path, bytes).unwrap();

        let error = safetensors_path_tensor_headers(&path)
            .expect_err("shape/dtype byte mismatch must fail closed")
            .to_string();
        assert!(error.contains("requires 8"), "{error}");
    }

    #[test]
    fn is_hidden_file_flags_dotfiles_and_appledouble_sidecars() {
        assert!(is_hidden_file(Path::new("._model.safetensors")));
        assert!(is_hidden_file(Path::new("/a/b/._model.safetensors")));
        assert!(is_hidden_file(Path::new("/a/b/.DS_Store")));
        assert!(is_hidden_file(Path::new(".gitattributes")));
        // Real shards — including the sharded form — must not be flagged.
        assert!(!is_hidden_file(Path::new("model.safetensors")));
        assert!(!is_hidden_file(Path::new("/a/b/model.safetensors")));
        assert!(!is_hidden_file(Path::new(
            "diffusion_pytorch_model-00001-of-00002.safetensors"
        )));
        // A leading dot on a *directory* component is not a hidden file name.
        assert!(!is_hidden_file(Path::new("/a/.cache/model.safetensors")));
    }

    /// SceneWorks#1333: an AppleDouble sidecar (`._model.safetensors`) carries the `.safetensors`
    /// extension and sorts *before* the real shard, so an extension-only filter opened it first and
    /// died on its magic bytes. `from_dir` must skip it and load the real shard.
    #[test]
    fn from_dir_skips_appledouble_sidecar() {
        let data: Vec<u8> = (0u8..16).collect();
        let tv = StTensorView::new(Dtype::I32, vec![2, 2], &data).unwrap();
        let bytes = safetensors::serialize([("blk.weight", tv)], &None).unwrap();

        let tmp = fixture_dir("gencore_appledouble_");
        let dir = tmp.path();
        std::fs::write(dir.join("model.safetensors"), &bytes).unwrap();
        // A real AppleDouble header: magic 0x00051607, version 0x00020000. Its first 8 bytes decode
        // as a ~2.2 TB safetensors header length.
        std::fs::write(
            dir.join("._model.safetensors"),
            [0x00, 0x05, 0x16, 0x07, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00],
        )
        .unwrap();
        // Sanity: the sidecar really does sort first, so this test would fail without the skip.
        let mut names: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        names.sort();
        assert_eq!(names[0], std::ffi::OsStr::new("._model.safetensors"));

        let meta = CheckpointMeta::from_dir(dir).expect("sidecar must be skipped, not loaded");
        assert_eq!(meta.keys().collect::<Vec<_>>(), vec!["blk.weight"]);
    }

    /// sc-10894: `safetensors_dir_bytes` recurses, counts only `.safetensors`, and skips AppleDouble
    /// `._*` sidecars + non-weight files — the same accounting the worker's whole-model sum uses so a
    /// component's bytes stay comparable to the total.
    #[test]
    fn safetensors_dir_bytes_recurses_and_skips_sidecars_and_nonweights() {
        let tmp = fixture_dir("gencore_dirbytes_");
        let root = tmp.path();
        let te = root.join("text_encoder");
        let dit = root.join("transformer");
        std::fs::create_dir_all(&te).unwrap();
        std::fs::create_dir_all(&dit).unwrap();
        std::fs::write(te.join("model.safetensors"), vec![0u8; 1000]).unwrap();
        std::fs::write(dit.join("diffusion.safetensors"), vec![0u8; 2000]).unwrap();
        // AppleDouble sidecar + a non-weight file must NOT count.
        std::fs::write(te.join("._model.safetensors"), vec![0u8; 500]).unwrap();
        std::fs::write(dit.join("config.json"), vec![0u8; 700]).unwrap();

        assert_eq!(safetensors_dir_bytes(root), 3000);
        assert_eq!(safetensors_dir_bytes(root.join("transformer")), 2000);
        // Missing dir ⇒ 0 (no signal).
        assert_eq!(safetensors_dir_bytes(root.join("nope")), 0);
    }

    #[cfg(unix)]
    #[test]
    fn safetensors_dir_bytes_skips_directory_symlink_cycles_but_follows_file_symlinks() {
        use std::os::unix::fs::symlink;

        let tmp = fixture_dir("gencore_dirbytes_cycle_");
        let root = tmp.path();
        let blobs = root.join("blobs");
        let model = root.join("model");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(blobs.join("weight"), vec![0u8; 123]).unwrap();
        symlink(blobs.join("weight"), model.join("model.safetensors")).unwrap();
        symlink(root, model.join("cycle")).unwrap();

        assert_eq!(safetensors_dir_bytes(root), 123);
    }

    /// sc-10894: `safetensors_path_bytes` dispatches on kind — the recursive sum for a DIR, the file
    /// length for a single non-hidden `.safetensors` FILE (the bernini/anima flat-component layout), and
    /// `0` for a non-weight file or a `._*` sidecar file.
    #[test]
    fn safetensors_path_bytes_handles_file_and_dir() {
        let tmp = fixture_dir("gencore_pathbytes_");
        let root = tmp.path();
        let sub = root.join("vae");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("model.safetensors"), vec![0u8; 800]).unwrap();
        std::fs::write(root.join("t5_encoder.safetensors"), vec![0u8; 2000]).unwrap();
        std::fs::write(root.join("._t5_encoder.safetensors"), vec![0u8; 300]).unwrap();
        std::fs::write(root.join("config.json"), vec![0u8; 400]).unwrap();

        // A flat component file → its length.
        assert_eq!(
            safetensors_path_bytes(root.join("t5_encoder.safetensors")),
            2000
        );
        // A subdir → the recursive sum.
        assert_eq!(safetensors_path_bytes(root.join("vae")), 800);
        // A sidecar file / non-weight file / missing path → 0.
        assert_eq!(
            safetensors_path_bytes(root.join("._t5_encoder.safetensors")),
            0
        );
        assert_eq!(safetensors_path_bytes(root.join("config.json")), 0);
        assert_eq!(safetensors_path_bytes(root.join("missing.safetensors")), 0);
    }

    /// A dir holding *only* a sidecar has no shards — the error must say so rather than surfacing a
    /// corrupt-header failure from the sidecar.
    #[test]
    fn from_dir_with_only_a_sidecar_reports_no_shards() {
        let tmp = fixture_dir("gencore_only_sidecar_");
        let dir = tmp.path();
        std::fs::write(dir.join("._model.safetensors"), [0x00, 0x05, 0x16, 0x07]).unwrap();

        let err = match CheckpointMeta::from_dir(dir) {
            Ok(_) => panic!("a dir holding only a sidecar must not load"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("no .safetensors files"), "unexpected: {err}");
    }
}
