//! Allocation-free descriptions of the payloads actually retained by an installed residual.
//!
//! These are tensor metadata, not physical-memory measurements. A transpose/reshape is a view of
//! the listed payload, not another owned parameter. Reconstruction inputs are intentionally absent:
//! after `Adapter::materialize` only the resulting delta (or small structured factors) is retained.

use mlx_rs::{Array, Dtype};

use super::Adapter;

/// One retained payload, in its forward-time shape and dtype.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorFacts {
    pub shape: Vec<i32>,
    pub dtype: Dtype,
    /// Element strides for an unnormalized file/view payload. Empty on an evaluated matrix
    /// payload means the source-proven row-or-column-contiguous layout class: transpose views,
    /// lazy evaluation and Tucker contraction ordering can select either orientation. Both are
    /// read directly by matmul and reference one full payload; neither a transpose nor this
    /// normalized layout class implies another parameter allocation.
    pub strides: Vec<usize>,
}

impl TensorFacts {
    pub fn of(array: &Array) -> Self {
        Self {
            shape: array.shape().to_vec(),
            dtype: array.dtype(),
            strides: array.strides().to_vec(),
        }
    }

    /// One full materialized matrix payload, in either matmul-native orientation. This is an
    /// allocation/layout class, not a fabricated exact stride or physical-backing certificate.
    pub fn materialized_matrix(rows: i32, cols: i32, dtype: Dtype) -> Self {
        Self {
            shape: vec![rows, cols],
            dtype,
            strides: Vec::new(),
        }
    }

    fn materialized_of(array: &Array) -> Self {
        let mut facts = Self::of(array);
        if facts.shape.len() == 2 {
            let (rows, cols) = (
                facts.shape[0].max(0) as usize,
                facts.shape[1].max(0) as usize,
            );
            if facts.strides == [cols, 1] || facts.strides == [1, rows] {
                facts.strides.clear();
            }
        }
        facts
    }

    pub fn elements(&self) -> u64 {
        self.shape
            .iter()
            .map(|&n| n.max(0) as u64)
            .fold(1, u64::saturating_mul)
    }

    pub fn bytes(&self) -> u64 {
        self.elements().saturating_mul(dtype_width(self.dtype))
    }

    /// A contiguous matrix, or its transpose, as produced by the strict file loader.
    pub fn matrix(rows: i32, cols: i32, dtype: Dtype, transposed: bool) -> Self {
        Self {
            shape: vec![rows, cols],
            dtype,
            strides: if transposed {
                vec![1, rows.max(0) as usize]
            } else {
                vec![cols.max(0) as usize, 1]
            },
        }
    }
}

/// An ordered residual in `AdaptableLinear::apply_adapters` (inference, not training).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdapterFacts {
    /// The loader transposes file A `[rank,in]` and B `[out,rank]`; an alpha fold may promote B.
    Lora {
        a: TensorFacts,
        b: TensorFacts,
        disabled: bool,
    },
    /// Dense-host LoKr AND LoHa both retain their reconstructed `[out,in]` delta here.
    Materialized { delta: TensorFacts, disabled: bool },
    /// Packed-host structured LoKr. Iris's dense install host does not select this variant, but
    /// describing the actual enum exhaustively avoids mistaking small factors for a full delta.
    Structured {
        w1: TensorFacts,
        w2: TensorFacts,
        disabled: bool,
    },
}

impl AdapterFacts {
    pub fn of(adapter: &Adapter) -> Self {
        let disabled = adapter.is_disabled();
        match adapter {
            Adapter::Lora { a, b, .. } => Self::Lora {
                a: TensorFacts::materialized_of(a),
                b: TensorFacts::materialized_of(b),
                disabled,
            },
            Adapter::Lokr { delta, .. } => Self::Materialized {
                delta: TensorFacts::materialized_of(delta),
                disabled,
            },
            Adapter::LokrStructured { factors } => Self::Structured {
                w1: TensorFacts::of(&factors.w1),
                w2: TensorFacts::of(&factors.w2),
                disabled,
            },
        }
    }

    /// The payloads live even when scale zero skips their forward. The two LoRA factors are
    /// separate file tensors; their transposed views must each be counted once, not twice.
    pub fn tensors(&self) -> Vec<&TensorFacts> {
        match self {
            Self::Lora { a, b, .. } => vec![a, b],
            Self::Materialized { delta, .. } => vec![delta],
            Self::Structured { w1, w2, .. } => vec![w1, w2],
        }
    }

    pub fn disabled(&self) -> bool {
        match self {
            Self::Lora { disabled, .. }
            | Self::Materialized { disabled, .. }
            | Self::Structured { disabled, .. } => *disabled,
        }
    }
}

/// Stable semantic dtype tags, not Debug formatting or native ABI discriminants. These describe
/// the ACTUAL loaded payload (e.g. a native F8 file read as U8 is U8 here), not its file spelling.
pub fn dtype_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::Bool => "BOOL",
        Dtype::Uint8 => "U8",
        Dtype::Int8 => "I8",
        Dtype::Uint16 => "U16",
        Dtype::Int16 => "I16",
        Dtype::Uint32 => "U32",
        Dtype::Int32 => "I32",
        Dtype::Uint64 => "U64",
        Dtype::Int64 => "I64",
        Dtype::Float16 => "F16",
        Dtype::Bfloat16 => "BF16",
        Dtype::Float32 => "F32",
        Dtype::Float64 => "F64",
        Dtype::Complex64 => "C64",
    }
}

/// Dtype produced by the ordinary native safetensors entry used by these iOS adapter loaders.
/// The existing neutral pre-load header gate still rejects C64 before this conversion. This is
/// not the separate opt-in fp8 decoder, and does not add a native dtype or a new accepted format.
pub fn native_header_dtype(dtype: crate::gen_core::weightsmeta::Dtype) -> crate::Result<Dtype> {
    use crate::gen_core::weightsmeta::Dtype as S;
    Ok(match dtype {
        S::BOOL => Dtype::Bool,
        S::U8 | S::F8_E4M3 | S::F8_E8M0 => Dtype::Uint8,
        S::I8 => Dtype::Int8,
        S::I16 => Dtype::Int16,
        S::U16 => Dtype::Uint16,
        S::I32 => Dtype::Int32,
        S::U32 => Dtype::Uint32,
        S::I64 => Dtype::Int64,
        S::U64 => Dtype::Uint64,
        S::F16 => Dtype::Float16,
        S::BF16 => Dtype::Bfloat16,
        S::F32 => Dtype::Float32,
        S::F64 | S::F8_E5M2 => {
            return Err(crate::Error::Unsupported(format!(
                "native MLX safetensors does not load {}",
                dtype.as_str()
            )))
        }
    })
}

/// Header counterpart of the ordinary packed-detect Linear constructor, followed by the actual
/// provider's OPTIONAL load-time affine quantization. Convert-at-install providers pass None.
/// Uses the executable packed validator; no Array, factor value or model is constructed.
pub fn linear_header_facts(
    headers: &std::collections::BTreeMap<
        String,
        crate::gen_core::weightsmeta::SafetensorsTensorHeader,
    >,
    base: &str,
    group_size: i32,
    load_quantize_bits: Option<i32>,
) -> crate::Result<super::LinearFacts> {
    let weight = headers.get(&format!("{base}.weight")).ok_or_else(|| {
        crate::Error::Msg(format!("missing adapter base projection {base}.weight"))
    })?;
    let shape = |header: &crate::gen_core::weightsmeta::SafetensorsTensorHeader| {
        header
            .shape
            .iter()
            .map(|&dim| {
                i32::try_from(dim).map_err(|_| {
                    crate::Error::Unsupported(
                        "projection dimension exceeds native i32 shape".into(),
                    )
                })
            })
            .collect::<crate::Result<Vec<_>>>()
    };
    let ws = shape(weight)?;
    let dtype = native_header_dtype(weight.dtype)?;
    if ws.len() != 2 {
        return Err(crate::Error::Unsupported(format!(
            "adapter target {base} is not a Linear projection"
        )));
    }
    let (input, packed) = if let Some(scales) = headers.get(&format!("{base}.scales")) {
        let ss = shape(scales)?;
        crate::quant::packed_bits_from_metadata(
            &ws,
            dtype,
            &ss,
            native_header_dtype(scales.dtype)?,
            group_size,
        )?;
        if !headers.contains_key(&format!("{base}.biases")) {
            return Err(crate::Error::Msg(format!(
                "missing packed projection {base}.biases"
            )));
        }
        (
            ss[1]
                .checked_mul(group_size)
                .ok_or_else(|| crate::Error::Msg("packed logical width overflow".into()))?,
            true,
        )
    } else if let Some(bits) = load_quantize_bits {
        let packed_columns = ws[1]
            .checked_mul(bits)
            .and_then(|width| width.checked_div(32))
            .ok_or_else(|| crate::Error::Msg("load-time packed width overflow".into()))?;
        if group_size <= 0 || ws[1] % group_size != 0 {
            return Err(crate::Error::Unsupported(
                "load-time quantization input is not group-aligned".into(),
            ));
        }
        crate::quant::packed_bits_from_metadata(
            &[ws[0], packed_columns],
            Dtype::Uint32,
            &[ws[0], ws[1] / group_size],
            Dtype::Bfloat16,
            group_size,
        )?;
        (ws[1], true)
    } else {
        (ws[1], false)
    };
    Ok(super::LinearFacts {
        base_shape: vec![ws[0], input],
        is_quantized: packed,
        has_live_adapters: false,
        adapter_count: 0,
        bias_shape: None,
    })
}

/// MLX's element widths, including accepted mixed-dtype file factors.
pub fn dtype_width(dtype: Dtype) -> u64 {
    match dtype {
        Dtype::Bool | Dtype::Uint8 | Dtype::Int8 => 1,
        Dtype::Uint16 | Dtype::Int16 | Dtype::Float16 | Dtype::Bfloat16 => 2,
        Dtype::Uint32 | Dtype::Int32 | Dtype::Float32 => 4,
        Dtype::Uint64 | Dtype::Int64 | Dtype::Float64 | Dtype::Complex64 => 8,
    }
}
