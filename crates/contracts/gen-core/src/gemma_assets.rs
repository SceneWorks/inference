//! **Self-contained Gemma text-encoder assets** (sc-18762) — the LTX-2.5 text encoder ships as one
//! `.safetensors` file that carries its own HuggingFace assets, so loading it needs **zero** other
//! files (no `google/gemma-3-12b-it` co-requisite snapshot the way LTX-2.3 did).
//!
//! Measured on `Lightricks/LTX-2.5` `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`
//! (snapshot `6c7e5e57…`, 686 tensors):
//!
//! | where | what |
//! |---|---|
//! | `__metadata__.gemma_config` | the full HF config JSON (no `config.json` file) |
//! | tensor `tokenizer_json` | `tokenizer.json` (32 169 626 bytes, `U8`) |
//! | tensor `hf_asset__tokenizer_config.json` | sidecar (2 749 bytes, `U8`) |
//! | tensor `hf_asset__processor_config.json` | sidecar (1 382 bytes, `U8`) |
//! | tensor `hf_asset__chat_template.jinja` | sidecar (17 466 bytes, `U8`) |
//! | tensor `hf_asset__generation_config.json` | sidecar (255 bytes, `U8`) |
//!
//! Reference: `packages/ltx-core/src/ltx_core/text_encoders/gemma/gemma_assets.py` @
//! `d151147788a9284cca791edc6ce898007e727fe6` (`GemmaAssets`, `_tensor_to_bytes`,
//! `_flatten_gemma4_unified_keys_for_comfy`).
//!
//! Two deliberate hardening choices over a literal port:
//!
//! 1. **Seek + read, never buffer the file.** [`GemmaAssets::from_single_file`] parses the
//!    safetensors header and then reads *only* the asset tensors' byte ranges. The 2.5 TE file is
//!    26.3 GB; [`crate::weightsmeta::CheckpointMeta`] would pull all of it into host RAM.
//! 2. **Every read is length-checked and fails loudly.** A missing metadata key, a missing asset
//!    tensor, a float-dtype asset, a byte range past EOF, or a short read is an `Err` — never a
//!    silent empty/partial asset. The companion hazard on the weight side is the same one upstream
//!    documented (`docs/ltx-2.5-gemma4-missing-tensors.md`): a wrong key remap left **11** tower
//!    tensors at random init under Comfy's `load_sd(..., strict=False)`. [`GemmaTeKeyMap`] is the
//!    strict counterpart — it canonicalizes both on-disk layouts, rejects collisions, and every
//!    lookup that misses is an [`Error::MissingTensor`].

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use safetensors::Dtype;

use crate::tokenizer::{ensure_single_leading_bos, TokenizerOutput};
use crate::{Error, Result};

/// `__metadata__` key holding the JSON-encoded HuggingFace Gemma config.
pub const GEMMA_CONFIG_METADATA_KEY: &str = "gemma_config";
/// Tensor key holding the packed `tokenizer.json` bytes.
pub const TOKENIZER_JSON_TENSOR_KEY: &str = "tokenizer_json";
/// Tensor-key prefix for a packed HF sidecar: `hf_asset__<filename>`.
pub const HF_ASSET_TENSOR_PREFIX: &str = "hf_asset__";

/// Sidecars the HF builders cannot work without — absence is a load error, not a warning.
pub const REQUIRED_SIDECARS: [&str; 2] = ["tokenizer_config.json", "processor_config.json"];

/// Sidecars an older / ComfyUI-produced pack may have stored as `__metadata__` **strings** rather
/// than `hf_asset__` tensors. Tensors win; metadata only fills a gap.
const METADATA_FALLBACK_SIDECARS: [&str; 4] = [
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "generation_config.json",
];

/// Sidecar file patterns collected off an HF directory root (upstream `_ASSET_GLOBS`).
const SIDECAR_EXTENSIONS: [&str; 2] = ["json", "jinja"];
/// Directory-root files that are *not* sidecars (upstream `_ASSET_GLOB_EXCLUDES`).
const SIDECAR_EXCLUDES: [&str; 2] = ["config.json", "tokenizer.json"];

/// Same bound `crate::weightsmeta::safetensors_path_tensor_headers` uses: a header larger than this
/// is a corrupt/mis-detected file (e.g. an AppleDouble `._` sidecar), not a real checkpoint.
const MAX_HEADER_SIZE: u64 = 100_000_000;

// =================================================================================================
// GemmaAssets
// =================================================================================================

/// The in-memory Gemma HF assets: the config JSON, `tokenizer.json`, and the sidecar files —
/// sourced either from a packed single-file text encoder or from a legacy HF directory root.
#[derive(Debug, Clone)]
pub struct GemmaAssets {
    source: String,
    config_json: String,
    tokenizer_json: Vec<u8>,
    sidecars: BTreeMap<String, Vec<u8>>,
}

impl GemmaAssets {
    /// Load from either a packed `.safetensors` text encoder or a legacy HF directory root
    /// (upstream `GemmaAssets.load`).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            return Self::from_root(path);
        }
        if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            return Self::from_single_file(path);
        }
        Err(Error::Msg(format!(
            "gemma assets: {} is neither a directory nor a .safetensors file",
            path.display()
        )))
    }

    /// Load every asset out of one packed `.safetensors` text encoder, reading **only** the asset
    /// tensors' byte ranges (never the multi-GB weight payload).
    pub fn from_single_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = std::fs::File::open(path)?;
        let file_len = file.metadata()?.len();
        let header = SafetensorsHeader::read(&mut file, file_len, path)?;

        let config_json = header
            .metadata
            .get(GEMMA_CONFIG_METADATA_KEY)
            .cloned()
            .ok_or_else(|| {
                Error::Msg(format!(
                    "gemma assets: {} has no __metadata__ key {GEMMA_CONFIG_METADATA_KEY:?} \
                     (JSON-encoded HuggingFace config) — this is not a self-contained Gemma text \
                     encoder",
                    path.display()
                ))
            })?;

        let tokenizer_json = header.read_asset(&mut file, TOKENIZER_JSON_TENSOR_KEY, path)?;

        let mut sidecars = BTreeMap::new();
        for key in header.asset_keys() {
            let Some(name) = key.strip_prefix(HF_ASSET_TENSOR_PREFIX) else {
                continue;
            };
            let name = name.to_string();
            let bytes = header.read_asset(&mut file, &key, path)?;
            sidecars.insert(name, bytes);
        }
        // Older / ComfyUI packs may carry the small JSON sidecars as metadata strings instead.
        for name in METADATA_FALLBACK_SIDECARS {
            if sidecars.contains_key(name) {
                continue;
            }
            if let Some(value) = header.metadata.get(name) {
                sidecars.insert(name.to_string(), value.as_bytes().to_vec());
            }
        }

        let assets = Self {
            source: path.display().to_string(),
            config_json,
            tokenizer_json,
            sidecars,
        };
        assets.require_sidecars()?;
        Ok(assets)
    }

    /// Load from a legacy HF Gemma directory root (`config.json` + `tokenizer.json` + sidecars),
    /// searched recursively so a nested `_readout_proj` layout resolves. First hit by sorted path
    /// wins, matching upstream `find_matching_file` / `rglob` + `setdefault`.
    pub fn from_root(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let mut files = Vec::new();
        collect_files(root, &mut files)?;
        files.sort();

        let find = |name: &str| -> Result<&PathBuf> {
            files
                .iter()
                .find(|p| p.file_name().and_then(|n| n.to_str()) == Some(name))
                .ok_or_else(|| {
                    Error::Msg(format!(
                        "gemma assets: no {name} under {} — a Gemma directory root must carry it",
                        root.display()
                    ))
                })
        };
        let config_json = std::fs::read_to_string(find("config.json")?)?;
        let tokenizer_json = std::fs::read(find("tokenizer.json")?)?;

        let mut sidecars: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for path in &files {
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if SIDECAR_EXCLUDES.contains(&name) {
                continue;
            }
            let is_sidecar = path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|ext| SIDECAR_EXTENSIONS.contains(&ext));
            if !is_sidecar || sidecars.contains_key(name) {
                continue;
            }
            sidecars.insert(name.to_string(), std::fs::read(path)?);
        }

        let assets = Self {
            source: root.display().to_string(),
            config_json,
            tokenizer_json,
            sidecars,
        };
        assets.require_sidecars()?;
        Ok(assets)
    }

    /// Where these assets came from (a file or directory path) — carried into every error message.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The raw HuggingFace config JSON text (`__metadata__.gemma_config`, or `config.json`).
    pub fn config_json(&self) -> &str {
        &self.config_json
    }

    /// The config's `model_type` — the discriminator upstream's `build_gemma_hf_config` keys on
    /// (`gemma4_unified` for the LTX-2.5 text encoder).
    pub fn config_model_type(&self) -> Result<String> {
        let value: serde_json::Value = serde_json::from_str(&self.config_json).map_err(|e| {
            Error::Msg(format!(
                "gemma assets: {} has an unparseable {GEMMA_CONFIG_METADATA_KEY}: {e}",
                self.source
            ))
        })?;
        value
            .get("model_type")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "gemma assets: {} config is missing model_type",
                    self.source
                ))
            })
    }

    /// The packed `tokenizer.json` bytes.
    pub fn tokenizer_json(&self) -> &[u8] {
        &self.tokenizer_json
    }

    /// Sidecar file names present, sorted.
    pub fn sidecar_names(&self) -> impl Iterator<Item = &str> {
        self.sidecars.keys().map(String::as_str)
    }

    /// One sidecar's bytes, or a loud error naming the source and the missing file.
    pub fn sidecar(&self, name: &str) -> Result<&[u8]> {
        self.sidecars.get(name).map(Vec::as_slice).ok_or_else(|| {
            Error::Msg(format!(
                "gemma assets from {} are missing sidecar {name:?} (embed it as \
                 {HF_ASSET_TENSOR_PREFIX}{name} or as a __metadata__ string)",
                self.source
            ))
        })
    }

    /// One sidecar decoded as UTF-8 text.
    pub fn sidecar_str(&self, name: &str) -> Result<&str> {
        std::str::from_utf8(self.sidecar(name)?).map_err(|e| {
            Error::Msg(format!(
                "gemma assets from {}: sidecar {name:?} is not UTF-8: {e}",
                self.source
            ))
        })
    }

    fn require_sidecars(&self) -> Result<()> {
        let missing: Vec<&str> = REQUIRED_SIDECARS
            .into_iter()
            .filter(|name| !self.sidecars.contains_key(*name))
            .collect();
        if !missing.is_empty() {
            return Err(Error::Msg(format!(
                "gemma assets from {} are missing required sidecar(s): {}. Embed each as \
                 {HF_ASSET_TENSOR_PREFIX}<name> (or as a __metadata__ string for small JSON)",
                self.source,
                missing.join(", ")
            )));
        }
        Ok(())
    }
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if crate::weightsmeta::is_hidden_file(&path) {
            continue;
        }
        // `file_type` does not follow symlinks: directory symlinks are skipped (no cycles), file
        // symlinks are kept (the HF cache stores every snapshot entry as one).
        if entry.file_type()?.is_dir() {
            collect_files(&path, out)?;
        } else {
            out.push(path);
        }
    }
    Ok(())
}

// =================================================================================================
// Safetensors header + targeted asset reads
// =================================================================================================

struct AssetLoc {
    dtype: Dtype,
    shape: Vec<usize>,
    start: u64,
    end: u64,
}

struct SafetensorsHeader {
    data_base: u64,
    file_len: u64,
    metadata: BTreeMap<String, String>,
    assets: BTreeMap<String, AssetLoc>,
}

impl SafetensorsHeader {
    /// Parse the 8-byte length prefix + JSON header, keeping only `__metadata__` and the **asset**
    /// tensor locations. Weight tensors are deliberately not indexed: a 686-entry index costs
    /// nothing to skip and this type must never tempt a caller into a whole-file read.
    fn read(file: &mut std::fs::File, file_len: u64, path: &Path) -> Result<Self> {
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix).map_err(|e| {
            Error::Msg(format!(
                "gemma assets: {} is too short for a safetensors header length: {e}",
                path.display()
            ))
        })?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len > MAX_HEADER_SIZE {
            return Err(Error::Msg(format!(
                "gemma assets: safetensors header in {} exceeds the {MAX_HEADER_SIZE}-byte maximum",
                path.display()
            )));
        }
        let data_base = 8u64
            .checked_add(header_len)
            .filter(|b| *b <= file_len)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "gemma assets: safetensors header in {} extends past the end of the file",
                    path.display()
                ))
            })?;
        let mut header = vec![0u8; header_len as usize];
        file.read_exact(&mut header)?;
        let json: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header)
            .map_err(|e| {
                Error::Msg(format!(
                    "gemma assets: safetensors header in {}: {e}",
                    path.display()
                ))
            })?;

        let mut metadata = BTreeMap::new();
        let mut assets = BTreeMap::new();
        for (key, value) in json {
            if key == "__metadata__" {
                let map = value.as_object().ok_or_else(|| {
                    Error::Msg(format!(
                        "gemma assets: __metadata__ in {} is not an object",
                        path.display()
                    ))
                })?;
                for (k, v) in map {
                    let text = v.as_str().ok_or_else(|| {
                        Error::Msg(format!(
                            "gemma assets: __metadata__.{k} in {} is not a string",
                            path.display()
                        ))
                    })?;
                    metadata.insert(k.clone(), text.to_string());
                }
                continue;
            }
            if !is_gemma_asset_key(&key) {
                continue;
            }
            let info: safetensors::tensor::TensorInfo =
                serde_json::from_value(value).map_err(|e| {
                    Error::Msg(format!(
                        "gemma assets: tensor {key:?} in {}: {e}",
                        path.display()
                    ))
                })?;
            let (start, end) = info.data_offsets;
            assets.insert(
                key,
                AssetLoc {
                    dtype: info.dtype,
                    shape: info.shape,
                    start: start as u64,
                    end: end as u64,
                },
            );
        }
        Ok(Self {
            data_base,
            file_len,
            metadata,
            assets,
        })
    }

    fn asset_keys(&self) -> Vec<String> {
        self.assets.keys().cloned().collect()
    }

    /// Read one asset tensor and decode it to bytes.
    ///
    /// Upstream's `_tensor_to_bytes` does `arr.astype(np.uint8).tobytes()` when the stored dtype is
    /// not `uint8` — ComfyUI packs have been seen storing `tokenizer_json` as `int8`. `astype`
    /// on an integer array keeps each element's **low byte**, so this reproduces it exactly for any
    /// integer width without assuming `U8`. Float dtypes are refused: `astype(np.uint8)` on a float
    /// array truncates toward zero and would silently corrupt a tokenizer, so a float asset is a
    /// packing bug we surface rather than launder.
    fn read_asset(&self, file: &mut std::fs::File, key: &str, path: &Path) -> Result<Vec<u8>> {
        let loc = self.assets.get(key).ok_or_else(|| {
            Error::MissingTensor(format!(
                "{key} in {} — the packed Gemma asset is absent",
                path.display()
            ))
        })?;
        let width = integer_dtype_width(loc.dtype).ok_or_else(|| {
            Error::Msg(format!(
                "gemma assets: tensor {key:?} in {} has dtype {:?}; packed assets must be an \
                 integer dtype (upstream `_tensor_to_bytes` casts integers to uint8)",
                path.display(),
                loc.dtype
            ))
        })?;
        let elements = loc.shape.iter().try_fold(1u64, |acc, dim| {
            acc.checked_mul(*dim as u64).ok_or_else(|| {
                Error::Msg(format!(
                    "gemma assets: tensor {key:?} in {} has a shape product overflow",
                    path.display()
                ))
            })
        })?;
        let declared = loc.end.checked_sub(loc.start).ok_or_else(|| {
            Error::Msg(format!(
                "gemma assets: tensor {key:?} in {} has inverted data_offsets [{}, {})",
                path.display(),
                loc.start,
                loc.end
            ))
        })?;
        let expected = elements.checked_mul(width as u64).ok_or_else(|| {
            Error::Msg(format!(
                "gemma assets: tensor {key:?} in {} has a byte-size overflow",
                path.display()
            ))
        })?;
        if declared != expected {
            return Err(Error::Msg(format!(
                "gemma assets: tensor {key:?} in {} declares {declared} payload bytes but {:?} \
                 {:?} requires {expected}",
                path.display(),
                loc.dtype,
                loc.shape
            )));
        }
        let abs_start = self.data_base.checked_add(loc.start);
        let abs_end = self.data_base.checked_add(loc.end);
        match (abs_start, abs_end) {
            (Some(s), Some(e)) if e <= self.file_len && s <= e => {}
            _ => {
                return Err(Error::Msg(format!(
                    "gemma assets: tensor {key:?} in {} spans data bytes [{}, {}) but the file \
                     holds only {} bytes after its {}-byte header — the packed asset is truncated",
                    path.display(),
                    loc.start,
                    loc.end,
                    self.file_len.saturating_sub(self.data_base),
                    self.data_base
                )));
            }
        }

        let len = usize::try_from(declared).map_err(|_| {
            Error::Msg(format!(
                "gemma assets: tensor {key:?} in {} is larger than this platform can address",
                path.display()
            ))
        })?;
        file.seek(SeekFrom::Start(abs_start.expect("checked above")))?;
        let mut raw = vec![0u8; len];
        file.read_exact(&mut raw).map_err(|e| {
            Error::Msg(format!(
                "gemma assets: short read of tensor {key:?} in {} ({len} bytes expected): {e}",
                path.display()
            ))
        })?;
        if width == 1 {
            return Ok(raw);
        }
        // Little-endian low byte of each element == numpy `astype(np.uint8)` on an integer array.
        Ok(raw.into_iter().step_by(width).collect())
    }
}

/// Byte width of an **integer** safetensors dtype, or `None` for anything else.
fn integer_dtype_width(dtype: Dtype) -> Option<usize> {
    match dtype {
        Dtype::BOOL | Dtype::U8 | Dtype::I8 => Some(1),
        Dtype::U16 | Dtype::I16 => Some(2),
        Dtype::U32 | Dtype::I32 => Some(4),
        Dtype::U64 | Dtype::I64 => Some(8),
        _ => None,
    }
}

/// `true` for the tensor keys that carry packed HF assets rather than model weights. A weight
/// loader must skip these; [`GemmaTeKeyMap`] does.
pub fn is_gemma_asset_key(key: &str) -> bool {
    key == TOKENIZER_JSON_TENSOR_KEY || key.starts_with(HF_ASSET_TENSOR_PREFIX)
}

// =================================================================================================
// Key layout: ComfyUI-flattened ⇄ legacy HF towers
// =================================================================================================

/// Rewrite one HF `gemma4_unified` key to the ComfyUI-flattened LTXAV text-encoder layout — the
/// layout the shipped 2.5 text encoder is already in. Keys already flattened pass through
/// unchanged, so this is idempotent and safe to run over either source.
///
/// Port of upstream `_flatten_gemma4_unified_keys_for_comfy`:
///
/// | HF | flattened |
/// |---|---|
/// | `model.language_model.*` | `model.*` |
/// | `model.vision_embedder.*` | `vision_model.*` |
/// | `model.embed_vision.*` | `multi_modal_projector.*` |
/// | `model.embed_audio.*` | `audio_projector.*` |
///
/// Skipping this remap is what left 11 tower tensors at random init under a non-strict load.
pub fn flatten_gemma4_unified_key(key: &str) -> Cow<'_, str> {
    const RULES: [(&str, &str); 4] = [
        ("model.language_model.", "model."),
        ("model.vision_embedder.", "vision_model."),
        ("model.embed_vision.", "multi_modal_projector."),
        ("model.embed_audio.", "audio_projector."),
    ];
    for (from, to) in RULES {
        if let Some(rest) = key.strip_prefix(from) {
            return Cow::Owned(format!("{to}{rest}"));
        }
    }
    Cow::Borrowed(key)
}

/// A checkpoint's text-encoder tensor keys canonicalized to the flattened layout, with the original
/// on-disk key retained for each.
///
/// Built with [`GemmaTeKeyMap::from_keys`], which accepts **either** on-disk layout and refuses a
/// remap that would collide two source keys onto one canonical name. Every lookup that misses is an
/// [`Error::MissingTensor`], so a mis-mapped tower surfaces as a load failure instead of silently
/// leaving tensors at random init.
#[derive(Debug, Clone, Default)]
pub struct GemmaTeKeyMap {
    by_canonical: BTreeMap<String, String>,
}

impl GemmaTeKeyMap {
    /// Canonicalize `keys`, skipping the packed asset tensors ([`is_gemma_asset_key`]).
    pub fn from_keys<'a>(keys: impl IntoIterator<Item = &'a str>) -> Result<Self> {
        let mut by_canonical: BTreeMap<String, String> = BTreeMap::new();
        for key in keys {
            if is_gemma_asset_key(key) {
                continue;
            }
            let canonical = flatten_gemma4_unified_key(key).into_owned();
            if let Some(previous) = by_canonical.insert(canonical.clone(), key.to_string()) {
                if previous != key {
                    return Err(Error::Msg(format!(
                        "gemma text encoder: keys {previous:?} and {key:?} both map to \
                         {canonical:?} — the checkpoint mixes the ComfyUI-flattened and legacy HF \
                         layouts for the same tensor"
                    )));
                }
            }
        }
        Ok(Self { by_canonical })
    }

    /// The on-disk key for a canonical name, or [`Error::MissingTensor`].
    pub fn source_for(&self, canonical: &str) -> Result<&str> {
        self.by_canonical
            .get(canonical)
            .map(String::as_str)
            .ok_or_else(|| {
                Error::MissingTensor(format!(
                    "{canonical} (gemma text encoder; neither the flattened nor the legacy HF \
                     spelling is present)"
                ))
            })
    }

    /// Whether a canonical key is present.
    pub fn contains(&self, canonical: &str) -> bool {
        self.by_canonical.contains_key(canonical)
    }

    /// Canonical keys, sorted.
    pub fn canonical_keys(&self) -> impl Iterator<Item = &str> {
        self.by_canonical.keys().map(String::as_str)
    }

    /// Number of canonicalized tensors.
    pub fn len(&self) -> usize {
        self.by_canonical.len()
    }

    /// Whether the map holds no tensors.
    pub fn is_empty(&self) -> bool {
        self.by_canonical.is_empty()
    }

    /// Fail if any of `canonical` is absent, naming **all** of them at once. This is the guard
    /// against the upstream random-init failure mode: a loader states the full key set it needs and
    /// finds out before any weight is bound.
    pub fn require_all<'a>(&self, canonical: impl IntoIterator<Item = &'a str>) -> Result<()> {
        let missing: Vec<&str> = canonical
            .into_iter()
            .filter(|key| !self.by_canonical.contains_key(*key))
            .collect();
        if !missing.is_empty() {
            return Err(Error::MissingTensor(format!(
                "gemma text encoder is missing {} required tensor(s): {}",
                missing.len(),
                missing.join(", ")
            )));
        }
        Ok(())
    }

    /// Count of canonical keys under `prefix` — how a caller checks a whole tower landed
    /// (`vision_model.`, `multi_modal_projector.`, `audio_projector.`).
    pub fn count_with_prefix(&self, prefix: &str) -> usize {
        self.by_canonical
            .keys()
            .filter(|key| key.starts_with(prefix))
            .count()
    }
}

// =================================================================================================
// Tokenizer
// =================================================================================================

/// The LTX Gemma prompt tokenizer: a fast tokenizer built from the packed `tokenizer.json`, with
/// the upstream `LTXGemmaTokenizer` policy applied on top.
///
/// Port of `ltx_core.text_encoders.gemma.tokenizer.LTXGemmaTokenizer.tokenize_with_weights` @
/// `d1511477`, which is one policy covering both Gemma generations:
///
/// 1. `text.strip()`
/// 2. encode with special tokens, right-truncate to `max_length`
/// 3. **prepend `<bos>` only if the sequence does not already start with it**, then re-truncate
/// 4. **left-pad** to `max_length` with `<pad>` (mask `0` on the pad run)
///
/// Step 3 is the whole point: Gemma 3's `tokenizer.json` carries a `TemplateProcessing`
/// post-processor that already emits `<bos>`, while Gemma 4's is a pass-through (measured on the
/// packed 2.5 tokenizer) and emits none. An unconditional prepend double-BOSes Gemma 3; no prepend
/// leaves Gemma 4 with none. The conditional gets **exactly one** in both cases.
///
/// Left-padding is not cosmetic: it places the real tokens at the high RoPE positions
/// `[max_length − L, max_length)`, which is what the Gemma forward was validated against.
pub struct LtxGemmaTokenizer {
    inner: tokenizers::Tokenizer,
    bos_id: i32,
    pad_id: i32,
    source: String,
}

impl std::fmt::Debug for LtxGemmaTokenizer {
    /// Elides the vocabulary — the packed Gemma tokenizer is 32 MB.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LtxGemmaTokenizer")
            .field("source", &self.source)
            .field("bos_id", &self.bos_id)
            .field("pad_id", &self.pad_id)
            .finish_non_exhaustive()
    }
}

impl LtxGemmaTokenizer {
    /// Build from loaded [`GemmaAssets`] — the packed `tokenizer.json` plus the `tokenizer_config.json`
    /// sidecar that names `<bos>`/`<pad>`. No filesystem access beyond what produced `assets`.
    pub fn from_assets(assets: &GemmaAssets) -> Result<Self> {
        Self::from_parts(
            assets.tokenizer_json(),
            assets.sidecar_str("tokenizer_config.json")?,
            assets.source(),
        )
    }

    /// Build from raw `tokenizer.json` bytes + `tokenizer_config.json` text. `source` only labels
    /// error messages.
    pub fn from_parts(tokenizer_json: &[u8], tokenizer_config: &str, source: &str) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_bytes(tokenizer_json)
            .map_err(|e| Error::Msg(format!("gemma tokenizer from {source}: {e}")))?;
        let config: serde_json::Value = serde_json::from_str(tokenizer_config).map_err(|e| {
            Error::Msg(format!(
                "gemma tokenizer from {source}: tokenizer_config.json: {e}"
            ))
        })?;
        let bos_token = special_token(&config, "bos_token").ok_or_else(|| {
            Error::Msg(format!(
                "gemma tokenizer from {source}: tokenizer_config.json declares no bos_token; the \
                 LTX encode path requires a leading BOS"
            ))
        })?;
        let bos_id = token_id(&inner, bos_token, "bos_token", source)?;
        // Upstream falls back to the EOS token when no pad token is declared.
        let pad_token = special_token(&config, "pad_token")
            .or_else(|| special_token(&config, "eos_token"))
            .ok_or_else(|| {
                Error::Msg(format!(
                    "gemma tokenizer from {source}: tokenizer_config.json declares neither \
                     pad_token nor eos_token"
                ))
            })?;
        let pad_id = token_id(&inner, pad_token, "pad_token", source)?;
        Ok(Self {
            inner,
            bos_id,
            pad_id,
            source: source.to_string(),
        })
    }

    /// The resolved `<bos>` id.
    pub fn bos_id(&self) -> i32 {
        self.bos_id
    }

    /// The resolved `<pad>` id.
    pub fn pad_id(&self) -> i32 {
        self.pad_id
    }

    /// Tokenize one prompt to `(1, max_length)` left-padded ids + mask. See the type docs for the
    /// exact policy. An empty (or whitespace-only) prompt yields `[<pad>… , <bos>]` — never an
    /// empty row — matching upstream, which pads the BOS-only sequence to `max_length`.
    pub fn encode(&self, prompt: &str, max_length: usize) -> Result<TokenizerOutput> {
        if max_length == 0 {
            return Err(Error::Msg(format!(
                "gemma tokenizer from {}: max_length must be greater than zero",
                self.source
            )));
        }
        let text = prompt.trim();
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| Error::Msg(format!("gemma tokenizer from {}: {e}", self.source)))?;
        let mut ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
        if ids.len() > max_length {
            ids.truncate(max_length);
        }
        ensure_single_leading_bos(&mut ids, self.bos_id, max_length);

        let valid = ids.len();
        let pad = max_length - valid;
        let mut padded = vec![self.pad_id; pad];
        padded.extend_from_slice(&ids);
        let mut mask = vec![0i32; pad];
        mask.resize(max_length, 1);
        Ok(TokenizerOutput { ids: padded, mask })
    }

    /// Detokenize ids back to text (`skip_special_tokens` drops BOS/pad/turn markers).
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| Error::Msg(format!("gemma tokenizer from {}: {e}", self.source)))
    }
}

/// Read a special-token declaration from `tokenizer_config.json`. HF writes these either as a bare
/// string or as an `AddedToken` object with a `content` field.
fn special_token<'a>(config: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    match config.get(key)? {
        serde_json::Value::String(s) => Some(s.as_str()),
        serde_json::Value::Object(map) => map.get("content")?.as_str(),
        _ => None,
    }
}

fn token_id(
    tokenizer: &tokenizers::Tokenizer,
    token: &str,
    label: &str,
    source: &str,
) -> Result<i32> {
    tokenizer
        .token_to_id(token)
        .map(|id| id as i32)
        .ok_or_else(|| {
            Error::Msg(format!(
                "gemma tokenizer from {source}: {label} {token:?} is not in the tokenizer vocabulary"
            ))
        })
}
