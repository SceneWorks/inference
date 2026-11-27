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
}
