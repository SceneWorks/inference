//! The **stored GGML block** header rule (SW-7285): the one, tensor-free definition of how a
//! Candle-native q4 / q8 tier persists a quantized projection, shared by every contract that sizes
//! or admits weights from safetensors headers and by the Candle loaders
//! (`candle_layers::llm_projection::ggml_block_storage` maps this rule onto `GgmlDType`).
//!
//! A Candle-native tier stores each quantized matrix as ONE `{base}.weight` tensor, dtype `U8`,
//! shape `[rows, blocks_per_row, block_bytes]` (a stacked weight prepends its leading axes), with
//! no `.scales` / `.biases` siblings. The GGML block type is the one whose persisted block size is
//! `block_bytes`. The component `config.json` declares the tier with
//! `quantization = {"bits": N, "storage": "ggml"}` ([`GGML_STORAGE`]) and no `group_size`: a GGML
//! block's group is fixed by its type. A tensor is classified from its dtype and shape alone, so a
//! mislabelled config can neither hide nor invent a stored block.
//!
//! A q4 tier may keep some projections at 8 bits (`Q8_0`: the builder's `--keep-q8` recipe
//! override and the MLX converters' precision floors). The marker then lists them as
//! `"overrides": [{"pattern": "<regex over the base name>", "bits": 8, "count": N}]`
//! ([`StoredGgmlOverride`]). [`StoredGgmlWidths`] is the one parser and admission rule for that
//! declaration: a block at a width other than `bits` is admitted only when a declared override of
//! exactly that width names it, and every override's `count` must match the component.

/// The `quantization.storage` value a Candle-native tier's component `config.json` carries.
pub const GGML_STORAGE: &str = "ggml";

/// The GGML block types a stored block tensor may carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StoredGgmlBlock {
    /// 32 weights, f16 scale, 4-bit codes: 18 B.
    Q4_0,
    /// 32 weights, f16 scale + f16 min, 4-bit codes: 20 B. The media lane's 4-bit container (an MLX
    /// affine Q4 group-64 pack repacks into it losslessly).
    Q4_1,
    /// 32 weights, f16 scale, 8-bit codes: 34 B.
    Q8_0,
    /// 256-weight K-quant super-block: 144 B.
    Q4K,
}

/// Every [`StoredGgmlBlock`], in the order the classifier tries them.
pub const STORED_GGML_BLOCKS: [StoredGgmlBlock; 4] = [
    StoredGgmlBlock::Q4_0,
    StoredGgmlBlock::Q4_1,
    StoredGgmlBlock::Q8_0,
    StoredGgmlBlock::Q4K,
];

impl StoredGgmlBlock {
    /// Weights per block.
    pub const fn block_size(self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q8_0 => 32,
            Self::Q4K => 256,
        }
    }

    /// Persisted bytes per block (GGML's `type_size`).
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q8_0 => 34,
            Self::Q4K => 144,
        }
    }

    /// The `quantization.bits` tier this block type stores.
    pub const fn bits(self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q4K => 4,
            Self::Q8_0 => 8,
        }
    }
}

/// A classified stored block tensor: its block type and logical `[rows, cols]` (a stacked tensor's
/// `rows` count every leading slice).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoredGgmlGeometry {
    pub block: StoredGgmlBlock,
    pub rows: usize,
    pub cols: usize,
}

/// Classify a tensor header as a stored GGML block tensor: `U8`, rank >= 3, last axis one of the
/// persisted block sizes. `None` for anything else (dense, MLX `U32` codes, a checkpoint codec).
/// Overflowing geometry is `None` too.
pub fn stored_ggml_geometry(is_u8: bool, shape: &[usize]) -> Option<StoredGgmlGeometry> {
    let [lead @ .., rows, blocks, block_bytes] = shape else {
        return None;
    };
    if !is_u8 {
        return None;
    }
    let block = STORED_GGML_BLOCKS
        .into_iter()
        .find(|block| block.block_bytes() == *block_bytes)?;
    let rows = lead.iter().try_fold(*rows, |n, &d| n.checked_mul(d))?;
    Some(StoredGgmlGeometry {
        block,
        rows,
        cols: blocks.checked_mul(block.block_size())?,
    })
}

/// [`stored_ggml_geometry`] restricted to a **rank-3** (single matrix) tensor — the form of every
/// non-stacked projection and embedding.
pub fn stored_ggml_matrix_geometry(is_u8: bool, shape: &[usize]) -> Option<StoredGgmlGeometry> {
    if shape.len() != 3 {
        return None;
    }
    stored_ggml_geometry(is_u8, shape)
}

/// [`stored_ggml_geometry`] of a [`crate::weightsmeta::SafetensorsTensorHeader`].
pub fn stored_ggml_header_geometry(
    header: &crate::weightsmeta::SafetensorsTensorHeader,
) -> Option<StoredGgmlGeometry> {
    stored_ggml_geometry(header.dtype == crate::weightsmeta::Dtype::U8, &header.shape)
}

/// Whether a component `config.json` value declares a Candle-native GGML tier
/// (`quantization.storage == "ggml"`).
pub fn config_declares_ggml_storage(cfg: &serde_json::Value) -> bool {
    cfg.get("quantization")
        .and_then(|q| q.get("storage"))
        .and_then(serde_json::Value::as_str)
        == Some(GGML_STORAGE)
}

/// The `quantization` key a stored-GGML marker lists its **width overrides** under.
pub const OVERRIDES_KEY: &str = "overrides";

/// The block widths a `tier_bits` stored-GGML tier admits, the tier's own width first: a q4 tier
/// may keep projections at 8 bits (the builder's `--keep-q8` recipe override and the MLX
/// converters' 8-bit precision floors), a q8 tier is uniform. Any other tier admits nothing.
pub const fn admitted_widths(tier_bits: usize) -> &'static [usize] {
    match tier_bits {
        4 => &[4, 8],
        8 => &[8],
        _ => &[],
    }
}

/// The widths a `tier_bits` tier's `overrides` may declare: [`admitted_widths`] without the tier's
/// own width (an override at the tier width would declare nothing).
pub fn override_widths(tier_bits: usize) -> &'static [usize] {
    admitted_widths(tier_bits)
        .split_first()
        .map_or(&[], |(_, rest)| rest)
}

/// A malformed, unsupported or contradicted stored-GGML width declaration: the one typed refusal
/// every reader of the marker's `overrides` shares.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StoredGgmlWidthError {
    /// `quantization.overrides` is not a list of `{"pattern", "bits", "count"}` records.
    #[error("stored GGML marker `quantization.overrides`: {0}")]
    Malformed(String),
    /// An override declares a width the tier does not admit ([`admitted_widths`]).
    #[error(
        "stored GGML marker override `{pattern}` declares Q{bits}, which a Q{tier} tier does not \
         admit (admitted override widths: {admitted:?})"
    )]
    UnsupportedWidth {
        pattern: String,
        bits: usize,
        tier: usize,
        admitted: Vec<usize>,
    },
    /// A block wider (or narrower) than the tier that no override declares.
    #[error(
        "stored GGML {tensor} stores {block:?} blocks (Q{block_bits}) under a Q{tier} marker, and no \
         `quantization.overrides` pattern declares it"
    )]
    Undeclared {
        tensor: String,
        block: StoredGgmlBlock,
        block_bits: usize,
        tier: usize,
    },
    /// A tensor an override names stores a width other than the override's.
    #[error(
        "stored GGML {tensor} stores {block:?} blocks (Q{block_bits}), but override `{pattern}` \
         declares Q{bits}"
    )]
    OverrideWidth {
        tensor: String,
        block: StoredGgmlBlock,
        block_bits: usize,
        pattern: String,
        bits: usize,
    },
    /// An override's recorded `count` disagrees with the stored blocks it matches.
    #[error(
        "stored GGML marker override `{pattern}` records count {declared}, but {found} stored \
         block tensor(s) match it"
    )]
    CountMismatch {
        pattern: String,
        declared: usize,
        found: usize,
    },
}

/// One `quantization.overrides` record of a stored-GGML marker: every stored block projection whose
/// **base name** (the tensor name without `.weight`) matches `pattern` (unanchored
/// [`regex::Regex::is_match`]) is stored at `bits`, and `count` projections match. The builder
/// (`candle_gen::ggml_tier`) writes these records with [`Self::to_json`]; every reader parses them
/// with [`StoredGgmlWidths::from_marker`].
#[derive(Clone, Debug)]
pub struct StoredGgmlOverride {
    pattern: regex::Regex,
    bits: usize,
    count: usize,
}

impl PartialEq for StoredGgmlOverride {
    fn eq(&self, other: &Self) -> bool {
        self.pattern.as_str() == other.pattern.as_str()
            && self.bits == other.bits
            && self.count == other.count
    }
}

impl Eq for StoredGgmlOverride {}

impl StoredGgmlOverride {
    /// A record over an already-compiled pattern.
    pub fn new(pattern: regex::Regex, bits: usize, count: usize) -> Self {
        Self {
            pattern,
            bits,
            count,
        }
    }

    /// A record whose pattern is compiled from `pattern` (a bad regex is
    /// [`StoredGgmlWidthError::Malformed`]).
    pub fn parse(pattern: &str, bits: usize, count: usize) -> Result<Self, StoredGgmlWidthError> {
        let pattern = regex::Regex::new(pattern).map_err(|error| {
            StoredGgmlWidthError::Malformed(format!("pattern `{pattern}`: {error}"))
        })?;
        Ok(Self::new(pattern, bits, count))
    }

    pub fn pattern(&self) -> &str {
        self.pattern.as_str()
    }

    pub fn bits(&self) -> usize {
        self.bits
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// Whether this record names the projection `base`.
    pub fn matches(&self, base: &str) -> bool {
        self.pattern.is_match(base)
    }

    /// The marker record: `{"pattern": ..., "bits": N, "count": N}`.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "pattern": self.pattern.as_str(),
            "bits": self.bits,
            "count": self.count,
        })
    }

    fn from_json(record: &serde_json::Value) -> Result<Self, StoredGgmlWidthError> {
        let malformed = StoredGgmlWidthError::Malformed;
        let record = record
            .as_object()
            .ok_or_else(|| malformed(format!("record {record} is not an object")))?;
        if let Some(key) = record
            .keys()
            .find(|key| !matches!(key.as_str(), "pattern" | "bits" | "count"))
        {
            return Err(malformed(format!(
                "record carries unknown key `{key}`; expected pattern, bits, count"
            )));
        }
        let pattern = record
            .get("pattern")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| malformed("record has no string `pattern`".into()))?;
        let integer = |key: &str| {
            record
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| malformed(format!("record `{pattern}` has no integer `{key}`")))
        };
        let (bits, count) = (integer("bits")?, integer("count")?);
        if count == 0 {
            return Err(malformed(format!(
                "record `{pattern}` matches no projection (count 0)"
            )));
        }
        Self::parse(pattern, bits, count)
    }
}

/// The per-tensor width contract a stored-GGML marker declares: the tier width
/// (`quantization.bits`) plus its [`StoredGgmlOverride`]s. A stored block at the tier width that
/// no override names is admitted; a block at another width is admitted only when every override
/// naming its base declares exactly that width (and at least one does), and each override's
/// `count` must equal the stored blocks it matches across the whole component
/// ([`Self::admit_surface`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredGgmlWidths {
    bits: usize,
    overrides: Vec<StoredGgmlOverride>,
}

impl StoredGgmlWidths {
    /// A uniform tier: every stored block at `bits`.
    pub fn uniform(bits: usize) -> Self {
        Self {
            bits,
            overrides: Vec::new(),
        }
    }

    /// A tier with `overrides`, each of whose widths `bits` must admit
    /// ([`StoredGgmlWidthError::UnsupportedWidth`]).
    pub fn new(
        bits: usize,
        overrides: Vec<StoredGgmlOverride>,
    ) -> Result<Self, StoredGgmlWidthError> {
        let admitted = override_widths(bits);
        if let Some(bad) = overrides.iter().find(|o| !admitted.contains(&o.bits)) {
            return Err(StoredGgmlWidthError::UnsupportedWidth {
                pattern: bad.pattern().to_owned(),
                bits: bad.bits,
                tier: bits,
                admitted: admitted.to_vec(),
            });
        }
        Ok(Self { bits, overrides })
    }

    /// Parse a stored-GGML `quantization` object's `overrides` (absent = uniform) for a tier the
    /// caller has already read `bits` from. The one reader of the records the builder writes.
    pub fn from_marker(
        bits: usize,
        quantization: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Self, StoredGgmlWidthError> {
        let overrides = match quantization.get(OVERRIDES_KEY) {
            None => Vec::new(),
            Some(serde_json::Value::Array(records)) => records
                .iter()
                .map(StoredGgmlOverride::from_json)
                .collect::<Result<_, _>>()?,
            Some(other) => {
                return Err(StoredGgmlWidthError::Malformed(format!(
                    "{other} is not a list"
                )))
            }
        };
        Self::new(bits, overrides)
    }

    /// [`Self::from_marker`] of a component `config.json` value: `None` unless it declares a
    /// stored-GGML tier ([`config_declares_ggml_storage`]) with an integer `bits`.
    pub fn from_config(cfg: &serde_json::Value) -> Result<Option<Self>, StoredGgmlWidthError> {
        if !config_declares_ggml_storage(cfg) {
            return Ok(None);
        }
        let quantization = cfg
            .get("quantization")
            .and_then(serde_json::Value::as_object)
            .expect("config_declares_ggml_storage read an object");
        let bits = quantization
            .get("bits")
            .and_then(serde_json::Value::as_u64)
            .and_then(|bits| usize::try_from(bits).ok())
            .ok_or_else(|| {
                StoredGgmlWidthError::Malformed("the marker has no integer `bits`".into())
            })?;
        Self::from_marker(bits, quantization).map(Some)
    }

    /// The tier width (`quantization.bits`).
    pub fn bits(&self) -> usize {
        self.bits
    }

    pub fn overrides(&self) -> &[StoredGgmlOverride] {
        &self.overrides
    }

    /// Admit one stored block projection `base` of block type `block` against this declaration.
    pub fn admit(&self, base: &str, block: StoredGgmlBlock) -> Result<(), StoredGgmlWidthError> {
        let block_bits = block.bits();
        let mut named = false;
        for o in self.overrides.iter().filter(|o| o.matches(base)) {
            if o.bits != block_bits {
                return Err(StoredGgmlWidthError::OverrideWidth {
                    tensor: base.to_owned(),
                    block,
                    block_bits,
                    pattern: o.pattern().to_owned(),
                    bits: o.bits,
                });
            }
            named = true;
        }
        if named || block_bits == self.bits {
            return Ok(());
        }
        Err(StoredGgmlWidthError::Undeclared {
            tensor: base.to_owned(),
            block,
            block_bits,
            tier: self.bits,
        })
    }

    /// Admit every stored block projection of a **whole component** (`(base, block)` pairs) with
    /// [`Self::admit`], then require each override's `count` to equal the projections it matches.
    pub fn admit_surface<'a>(
        &self,
        stored: impl IntoIterator<Item = (&'a str, StoredGgmlBlock)>,
    ) -> Result<(), StoredGgmlWidthError> {
        let mut found = vec![0usize; self.overrides.len()];
        for (base, block) in stored {
            self.admit(base, block)?;
            for (n, o) in found.iter_mut().zip(&self.overrides) {
                *n += usize::from(o.matches(base));
            }
        }
        match self
            .overrides
            .iter()
            .zip(found)
            .find(|(o, found)| o.count != *found)
        {
            Some((o, found)) => Err(StoredGgmlWidthError::CountMismatch {
                pattern: o.pattern().to_owned(),
                declared: o.count,
                found,
            }),
            None => Ok(()),
        }
    }

    /// [`Self::admit_surface`] over safetensors headers: every `{base}.weight` the shared header
    /// rule ([`stored_ggml_header_geometry`]) classifies as a stored block.
    pub fn admit_headers<'a>(
        &self,
        headers: impl IntoIterator<Item = &'a crate::weightsmeta::SafetensorsTensorHeader>,
    ) -> Result<(), StoredGgmlWidthError> {
        self.admit_surface(headers.into_iter().filter_map(|header| {
            let base = header.name.strip_suffix(".weight")?;
            stored_ggml_header_geometry(header).map(|g| (base, g.block))
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_only_u8_block_tensors() {
        let g = |shape: &[usize]| stored_ggml_geometry(true, shape);
        assert_eq!(
            g(&[32, 4, 20]),
            Some(StoredGgmlGeometry {
                block: StoredGgmlBlock::Q4_1,
                rows: 32,
                cols: 128
            })
        );
        assert_eq!(g(&[8, 2, 34]).unwrap().block, StoredGgmlBlock::Q8_0);
        assert_eq!(g(&[8, 2, 18]).unwrap().block, StoredGgmlBlock::Q4_0);
        assert_eq!(
            g(&[2, 1, 144]),
            Some(StoredGgmlGeometry {
                block: StoredGgmlBlock::Q4K,
                rows: 2,
                cols: 256
            })
        );
        // Stacked experts fold the leading axis into rows.
        assert_eq!(g(&[4, 8, 2, 18]).unwrap().rows, 32);
        assert_eq!(stored_ggml_matrix_geometry(true, &[4, 8, 2, 18]), None);
        // Float of the same shape, a rank-2 code matrix, an unknown block size, and overflow.
        assert_eq!(stored_ggml_geometry(false, &[32, 4, 20]), None);
        assert_eq!(g(&[32, 80]), None);
        assert_eq!(g(&[32, 4, 21]), None);
        assert_eq!(g(&[usize::MAX, 2, 4, 20]), None);
        assert_eq!(g(&[1, usize::MAX, 20]), None);
    }

    #[test]
    fn block_tables_are_consistent() {
        for block in STORED_GGML_BLOCKS {
            assert!(matches!(block.bits(), 4 | 8));
            // bits per weight plus at least one f16 scale per block.
            assert!(block.block_bytes() * 8 >= block.block_size() * block.bits() + 16);
        }
    }

    #[test]
    fn config_marker_is_recognised() {
        let ggml = serde_json::json!({"quantization": {"bits": 4, "storage": "ggml"}});
        let mlx = serde_json::json!({"quantization": {"bits": 4, "group_size": 64}});
        assert!(config_declares_ggml_storage(&ggml));
        assert!(!config_declares_ggml_storage(&mlx));
        assert!(!config_declares_ggml_storage(&serde_json::json!({})));
    }

    fn widths(cfg: serde_json::Value) -> Result<StoredGgmlWidths, StoredGgmlWidthError> {
        StoredGgmlWidths::from_config(&serde_json::json!({ "quantization": cfg }))
            .map(|w| w.expect("a ggml marker"))
    }

    #[test]
    fn override_records_round_trip_through_the_one_parser() {
        let record = StoredGgmlOverride::parse(r"^blocks\.0\.q$", 8, 1).unwrap();
        let parsed = widths(serde_json::json!({
            "bits": 4, "storage": "ggml", "overrides": [record.to_json()]
        }))
        .unwrap();
        assert_eq!(parsed.bits(), 4);
        assert_eq!(parsed.overrides(), std::slice::from_ref(&record));
        assert_eq!(
            widths(serde_json::json!({"bits": 8, "storage": "ggml"})).unwrap(),
            StoredGgmlWidths::uniform(8)
        );
        assert_eq!(
            StoredGgmlWidths::from_config(&serde_json::json!({"quantization": {"bits": 4}}))
                .unwrap(),
            None,
            "an MLX affine marker is not a stored-GGML declaration"
        );
        assert_eq!(override_widths(4), &[8]);
        assert!(override_widths(8).is_empty() && override_widths(3).is_empty());
    }

    #[test]
    fn malformed_and_unsupported_overrides_are_typed_refusals() {
        let bad = |overrides: serde_json::Value| {
            widths(serde_json::json!({"bits": 4, "storage": "ggml", "overrides": overrides}))
                .unwrap_err()
        };
        for overrides in [
            serde_json::json!({"pattern": "q", "bits": 8, "count": 1}),
            serde_json::json!([{"pattern": "q", "bits": 8}]),
            serde_json::json!([{"pattern": "q", "bits": 8, "count": 0}]),
            serde_json::json!([{"pattern": "(", "bits": 8, "count": 1}]),
            serde_json::json!([{"pattern": "q", "bits": 8, "count": 1, "group_size": 64}]),
        ] {
            assert!(
                matches!(bad(overrides.clone()), StoredGgmlWidthError::Malformed(_)),
                "{overrides}"
            );
        }
        for bits in [4, 2, 16] {
            assert!(matches!(
                bad(serde_json::json!([{"pattern": "q", "bits": bits, "count": 1}])),
                StoredGgmlWidthError::UnsupportedWidth { .. }
            ));
        }
        // A q8 tier admits no override at all.
        assert!(matches!(
            widths(
                serde_json::json!({"bits": 8, "storage": "ggml", "overrides": [
                    {"pattern": "q", "bits": 8, "count": 1}
                ]})
            )
            .unwrap_err(),
            StoredGgmlWidthError::UnsupportedWidth { tier: 8, .. }
        ));
    }

    #[test]
    fn admission_requires_a_declared_override_and_its_exact_count() {
        use StoredGgmlBlock::{Q4_1, Q8_0};
        let w = StoredGgmlWidths::new(4, vec![StoredGgmlOverride::parse(r"\.q$", 8, 2).unwrap()])
            .unwrap();
        w.admit_surface([("a.q", Q8_0), ("b.q", Q8_0), ("a.k", Q4_1)])
            .unwrap();
        // An undeclared wider block.
        assert!(matches!(
            w.admit("a.k", Q8_0).unwrap_err(),
            StoredGgmlWidthError::Undeclared {
                tier: 4,
                block_bits: 8,
                ..
            }
        ));
        // A projection the override names at the tier width contradicts it.
        assert!(matches!(
            w.admit("a.q", Q4_1).unwrap_err(),
            StoredGgmlWidthError::OverrideWidth {
                bits: 8,
                block_bits: 4,
                ..
            }
        ));
        // The recorded count must match the component exactly.
        assert_eq!(
            w.admit_surface([("a.q", Q8_0)]).unwrap_err(),
            StoredGgmlWidthError::CountMismatch {
                pattern: r"\.q$".into(),
                declared: 2,
                found: 1
            }
        );
        // A uniform tier refuses any other width.
        assert!(StoredGgmlWidths::uniform(4).admit("a.q", Q8_0).is_err());
        StoredGgmlWidths::uniform(8).admit("a.q", Q8_0).unwrap();
    }
}
