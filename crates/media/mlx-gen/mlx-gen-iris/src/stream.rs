//! Rung 4 — **bounded transformer residency** for the Iris routes (epic 16774): the window
//! materialization both streamed components share, and the header-only price of one window read.
//!
//! A streamed component holds only its skeleton (embeddings, modulation cores, norms, heads)
//! resident; each block stack — the backbone's trunk, PiT, text-adapter layerwise and refiner
//! blocks, the Qwen3-VL tower's decoder layers — is read from the checkpoint ONE WINDOW at a time,
//! inside the window's own priced read quantum, run, and released before the next window is read.
//! The schedule is the shared [`mlx_gen::block_residency::run_windowed`] driver (evaluate the carry
//! before the window drops; drain the allocator after it).
//!
//! **The read is the eager loader's, per window.** Every tensor of the window is taken OUT of a
//! freshly opened lazy view, its file bytes read on MLX's CPU stream first (a GPU cast over an unread
//! load makes the command buffer wait on the disk — sc-24245), cast exactly as the eager loader casts
//! it (`CastPolicy`), and evaluated before the next tensor is read, so a window's read peak is the
//! window as held plus ONE source tensor beside its cast — never the whole window at both widths.
//! The block is then built from those materialized tensors by the SAME constructors the resident
//! load uses, so a streamed block is the resident block, value for value.
//!
//! **Only stored tiers stream.** A streamed load reads a tier exactly as the artifact stores it: the
//! dense release at the dense tier, or a packed export (`crate::packed`) at its own tier. A dense
//! artifact asked for at `q8`/`q4` would re-quantize every window on every forward — a load-time
//! requantization no capture measured, which publishes no calibration identity on any route
//! ([`crate::memory_strategy`]) — so the streamed loaders refuse it by name
//! (`refuse_stream_requantization`).
//!
//! `window_read_bytes` prices a window from the checkpoint's HEADERS with the rule
//! `materialize` executes; the runtime guard and the iOS pricing ([`crate::ios_pricing`]) both
//! call it, and `crate::quantum_proofs` proves it against MLX's measured peak of a real read.

use std::path::Path;

use mlx_gen::gen_core::weightsmeta::{Dtype as Stored, SafetensorsTensorHeader};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Quant, Result};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

/// Window sizes (blocks per window) rung 4 publishes. One: the window holds a single block, the
/// tightest bound the mechanism has, and the only value a selection may name.
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1];

/// The window rung 4 runs at when a selection names none.
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;

/// How a materialization casts one tensor — the loader whose cast it replays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CastPolicy {
    /// The backbone loaders ([`crate::model::load_backbone_quantized`],
    /// [`crate::depth::load_depth_export`]) at ordinary floating storage: floating matrices
    /// (rank ≥ 2) but the text position table and packed parts to compute; vectors as stored.
    /// The existing window policy leaves numeric/F64 matrices stored, unlike root eager reads;
    /// constructor accounting must therefore use this policy's actual post-read dtype.
    Backbone { compute: Dtype },
    /// The Qwen3-VL tower (`mlx_llm`'s decoder build): every float tensor but a packed projection's
    /// stored parts to bf16.
    Tower,
}

impl CastPolicy {
    fn target(self, name: &str, ndim: usize, packed_part: bool, float: bool) -> Option<Dtype> {
        if packed_part || !float {
            return None;
        }
        match self {
            CastPolicy::Backbone { compute } => {
                (ndim >= 2 && !is_position_table(name)).then_some(compute)
            }
            CastPolicy::Tower => Some(Dtype::Bfloat16),
        }
    }
}

/// The backbone's text position table, which every loader keeps as stored.
fn is_position_table(name: &str) -> bool {
    name == "y_pos_embedding" || name.ends_with(".y_pos_embedding")
}

/// Whether a stored tensor is part of a packed projection (its u32 codes, stored scales or biases)
/// — the rule of [`crate::model::is_packed_part`], from a header.
pub(crate) fn is_packed_part_header(header: &SafetensorsTensorHeader) -> bool {
    header.dtype == Stored::U32
        || header.name.ends_with(".scales")
        || header.name.ends_with(".biases")
}

fn stored_width(dtype: Stored) -> Option<u64> {
    match dtype {
        Stored::F32 => Some(4),
        Stored::BF16 | Stored::F16 => Some(2),
        _ => None,
    }
}

fn is_float(dtype: Stored) -> bool {
    stored_width(dtype).is_some()
}

/// The price of reading one tensor: what the window holds of it after the read, and the source
/// bytes that are live beside that only while the tensor itself is read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TensorRead {
    /// Bytes the materialized window holds (the cast copy, or the stored bytes when uncast).
    pub held: u64,
    /// The source bytes alive beside the cast while this one tensor is read (`0` when uncast: the
    /// source IS what is held).
    pub transient: u64,
}

/// [`TensorRead`] of `header` under `policy`: the cast is a new buffer only when its target dtype
/// differs from the stored one (MLX's `astype` to the stored dtype returns the array itself).
pub(crate) fn tensor_read(header: &SafetensorsTensorHeader, policy: CastPolicy) -> TensorRead {
    let packed = is_packed_part_header(header);
    let target = policy.target(
        &header.name,
        header.shape.len(),
        packed,
        is_float(header.dtype),
    );
    let width = |dtype: Dtype| -> u64 { crate::nn::dtype_bytes(dtype) };
    match (target, stored_width(header.dtype)) {
        (Some(target), Some(stored))
            if width(target) != stored || mismatched(header.dtype, target) =>
        {
            TensorRead {
                held: target_bytes(header, target).unwrap_or(u64::MAX),
                transient: source_allocation(header.data_bytes).unwrap_or(u64::MAX),
            }
        }
        _ => TensorRead {
            held: crate::retention::allocated(header.data_bytes),
            transient: 0,
        },
    }
}

/// A same-width cast that still changes the dtype (f16 → bf16) is a new buffer too.
fn mismatched(stored: Stored, target: Dtype) -> bool {
    matches!(
        (stored, target),
        (Stored::F16, Dtype::Bfloat16) | (Stored::BF16, Dtype::Float16)
    )
}

/// Bias targets after the ACTUAL window read: the shared constructor law distinguishes linear
/// biases (compute/BF16) from norm/modulation biases (F32), including same-width dtype changes.
fn bias_cast(
    header: &SafetensorsTensorHeader,
    headers: &[&SafetensorsTensorHeader],
    policy: CastPolicy,
) -> u64 {
    let CastPolicy::Backbone { compute } = policy else {
        return 0;
    };
    if !header.name.ends_with(".bias") {
        return 0;
    }
    let view = headers.iter().map(|h| (h.name.as_str(), *h)).collect();
    streamed_constructor_cast_bytes(header, &view, compute, policy)
}

/// The price of one block's read: what the materialized block holds, and the largest source
/// tensor alive beside its cast while it is read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockRead {
    /// Bytes the built block holds (its materialized tensors plus every constructor/getter
    /// target, conservatively retaining completed targets rather than assuming retirement).
    pub held: u64,
    /// The largest single [`TensorRead::transient`] of the block.
    pub transient: u64,
}

/// [`BlockRead`] of the tensors `headers` names, under `policy`.
pub(crate) fn block_read(headers: &[&SafetensorsTensorHeader], policy: CastPolicy) -> BlockRead {
    let view = headers.iter().map(|h| (h.name.as_str(), *h)).collect();
    let mut read = BlockRead::default();
    for header in headers {
        let t = tensor_read(header, policy);
        let constructor = match policy {
            CastPolicy::Backbone { .. } if header.name.ends_with(".bias") => {
                bias_cast(header, headers, policy)
            }
            CastPolicy::Backbone { compute } => {
                streamed_constructor_cast_bytes(header, &view, compute, policy)
            }
            CastPolicy::Tower => tower_constructor_cast_bytes(header, &view),
        };
        read.held = read.held.saturating_add(t.held).saturating_add(constructor);
        read.transient = read.transient.max(t.transient);
    }
    read
}

/// The read quantum's price for a window of `blocks`: every block as held, plus the largest
/// source tensor beside its cast. The runtime guard checks it before the window is read, and
/// [`crate::ios_pricing`] prices it, through this one function.
pub(crate) fn window_read_bytes(blocks: &[BlockRead]) -> u64 {
    let held = blocks.iter().fold(0u64, |a, b| a.saturating_add(b.held));
    let transient = blocks.iter().map(|b| b.transient).max().unwrap_or(0);
    held.saturating_add(transient)
}

/// What a window of `blocks` holds while it runs (its read's transient released).
pub(crate) fn window_held_bytes(blocks: &[BlockRead]) -> u64 {
    blocks.iter().fold(0u64, |a, b| a.saturating_add(b.held))
}

/// The largest window of `window` consecutive blocks of `blocks`.
pub(crate) fn largest_window(blocks: &[BlockRead], window: usize) -> (u64, u64) {
    let window = window.max(1);
    let mut read = 0u64;
    let mut held = 0u64;
    let mut first = 0;
    while first < blocks.len() {
        let end = (first + window).min(blocks.len());
        read = read.max(window_read_bytes(&blocks[first..end]));
        held = held.max(window_held_bytes(&blocks[first..end]));
        first = end;
    }
    (read, held)
}

/// Read the tensors `keys` names (`(file key, built key)`) out of the lazy `view`, one at a time:
/// each one's file bytes on MLX's CPU stream first, then its cast under `policy`, evaluated before
/// the next is read. The returned map holds exactly the window's materialized tensors under their
/// built keys; the view no longer holds them.
pub(crate) fn materialize(
    view: &mut Weights,
    keys: &[(String, String)],
    policy: CastPolicy,
) -> Result<Weights> {
    let mut out = Weights::empty();
    for (file_key, built_key) in keys {
        let array = view.remove(file_key).ok_or_else(|| {
            Error::Msg(format!(
                "iris stream: the checkpoint no longer holds `{file_key}` (it changed after load)"
            ))
        })?;
        mlx_rs::transforms::eval_pending_loads([&array])?;
        let packed = array.dtype() == Dtype::Uint32
            || built_key.ends_with(".scales")
            || built_key.ends_with(".biases");
        let float = matches!(
            array.dtype(),
            Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
        );
        let array = match policy.target(built_key, array.ndim(), packed, float) {
            Some(target) if target != array.dtype() => array.as_dtype(target)?,
            _ => array,
        };
        eval([&array])?;
        out.insert(built_key.clone(), array);
    }
    Ok(out)
}

/// The tensor headers of `path` as stored.
pub(crate) fn headers(path: &Path) -> Result<Vec<SafetensorsTensorHeader>> {
    mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(path)
        .map_err(|e| Error::Msg(format!("iris: reading {} headers: {e}", path.display())))
}

/// Native allocation floor of the tensors this Iris load actually materializes. This is NOT
/// the shared raw artifact floor: every kept tensor is a separate allocator request.
pub(crate) fn materialized_load_bytes(
    headers: &[SafetensorsTensorHeader],
    keep: impl Fn(&str) -> bool,
) -> Result<u64> {
    headers
        .iter()
        .filter(|h| keep(&h.name))
        .try_fold(0u64, |total, h| {
            mlx_gen::ios_quantum::fresh_metal_allocation_bytes(
                h.data_bytes,
                crate::retention::PAGE_BYTES,
            )
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or_else(|| {
                Error::Unsupported("Iris materialized load byte arithmetic overflow".into())
            })
        })
}

/// The real Iris pre-read guard and the source pricer share the SAME kept tensors and allocation
/// rule. Keep the generic raw before_load API unchanged (other providers depend on its contract).
pub(crate) fn before_materialized_load(
    label: &str,
    components: &[std::path::PathBuf],
    keep: impl Fn(&str) -> bool,
) -> Result<Option<LoadSources>> {
    if !mlx_gen::ios_quantum::enabled() {
        return Ok(None);
    }
    let mut sources = Vec::with_capacity(components.len());
    let mut total = 0u64;
    for path in components {
        let selected: Vec<_> = headers(path)?
            .into_iter()
            .filter(|h| keep(&h.name))
            .collect();
        total = total
            .checked_add(materialized_load_bytes(&selected, |_| true)?)
            .ok_or_else(load_overflow)?;
        sources.push(selected);
    }
    mlx_gen::ios_quantum::before(label, total)?;
    Ok(Some(sources))
}

/// A real pre-materialization stage, after the unchanged raw floor and before any native read,
/// compute cast, scalar item, constructor or internal parameter evaluation.
pub(crate) const LOAD_TARGET_QUANTUM: &str = "Iris load targets";

/// The actual late export upload/evaluation stage, after retained model growth.
pub(crate) const EXPORT_TARGET_QUANTUM: &str = "Iris export targets";

pub(crate) fn before_export_targets(constructors: u64, elements: usize) -> Result<()> {
    if !mlx_gen::ios_quantum::enabled() {
        return Ok(());
    }
    let elements = u64::try_from(elements).map_err(|_| load_overflow())?;
    let required = constructors
        .checked_add(empty_prompt_upload_bytes(elements)?)
        .ok_or_else(load_overflow)?;
    mlx_gen::ios_quantum::before(EXPORT_TARGET_QUANTUM, required)
}

/// Selected headers from the ORIGINAL raw-floor read. The first file is the backbone, remaining
/// files are shipped empty-prompt sources. No tensor data is read and no second parse is needed.
pub(crate) type LoadSources = Vec<Vec<SafetensorsTensorHeader>>;

fn load_overflow() -> Error {
    Error::Unsupported("Iris joint load byte arithmetic overflow".into())
}
fn source_allocation(bytes: u64) -> Result<u64> {
    mlx_gen::ios_quantum::fresh_metal_allocation_bytes(bytes, crate::retention::PAGE_BYTES)
        .ok_or_else(load_overflow)
}
fn target_bytes(header: &SafetensorsTensorHeader, target: Dtype) -> Result<u64> {
    let bytes = header
        .shape
        .iter()
        .try_fold(mlx_llm::primitives::nn::dtype_bytes(target), |n, dim| {
            n.checked_mul(*dim as u64).ok_or_else(load_overflow)
        })?;
    source_allocation(bytes)
}
fn same_dtype(stored: Stored, target: Dtype) -> bool {
    matches!(
        (stored, target),
        (Stored::BOOL, Dtype::Bool)
            | (Stored::U8, Dtype::Uint8)
            | (Stored::I8, Dtype::Int8)
            | (Stored::U16, Dtype::Uint16)
            | (Stored::I16, Dtype::Int16)
            | (Stored::U32, Dtype::Uint32)
            | (Stored::I32, Dtype::Int32)
            | (Stored::U64, Dtype::Uint64)
            | (Stored::I64, Dtype::Int64)
            | (Stored::F16, Dtype::Float16)
            | (Stored::BF16, Dtype::Bfloat16)
            | (Stored::F32, Dtype::Float32)
            | (Stored::F64, Dtype::Float64)
    )
}
fn read_cast(name: &str, header: &SafetensorsTensorHeader) -> bool {
    name != "depth_reducer.weight"
        && name != "depth_reducer.bias"
        && header.shape.len() >= 2
        && name != "y_pos_embedding"
        && !is_packed_part_header(header)
}

/// The constructor's target dtype, independent of how its source was read. Root loads and
/// deferred windows share these constructor sites but deliberately retain their own read policies.
fn backbone_constructor_target(
    name: &str,
    header: &SafetensorsTensorHeader,
    headers: &std::collections::HashMap<&str, &SafetensorsTensorHeader>,
    compute: Dtype,
) -> Option<Dtype> {
    let linear_base = name.strip_suffix(".bias").filter(|base| {
        headers.iter().any(|(key, sibling)| {
            sibling.shape.len() == 2 && key.strip_suffix(".weight") == Some(*base)
        })
    });
    if name.ends_with(crate::packed::IN_FEATURES_SUFFIX) {
        Some(Dtype::Int32)
    } else if name == "depth_reducer.weight" || name == "depth_reducer.bias" {
        Some(compute)
    } else if let Some(base) = linear_base {
        let packed = headers
            .keys()
            .any(|key| key.strip_suffix(".scales") == Some(base));
        Some(if packed { Dtype::Bfloat16 } else { compute })
    } else if name == "y_pos_embedding"
        || header.shape.len() == 1 && name.ends_with(".weight")
        || name.ends_with(".bias")
    {
        Some(Dtype::Float32)
    } else if header.shape.len() == 2 && name.ends_with(".weight") {
        let base = name.strip_suffix(".weight").expect("checked suffix");
        (!headers
            .keys()
            .any(|key| key.strip_suffix(".scales") == Some(base)))
        .then_some(compute)
    } else {
        None
    }
}

/// A constructor cast aliases only when its ACTUAL post-read dtype is already the target.
fn constructor_cast_bytes(
    header: &SafetensorsTensorHeader,
    target: Option<Dtype>,
    post_read: Option<Dtype>,
) -> Result<u64> {
    let Some(target) = target else {
        return Ok(0);
    };
    let aliases = post_read.map_or_else(|| same_dtype(header.dtype, target), |read| read == target);
    if aliases {
        Ok(0)
    } else {
        target_bytes(header, target)
    }
}

/// Window inventory keeps its existing saturating upper contract. On arithmetic overflow the
/// upper is unpayable, never a smaller allowance; root load/pricing helpers return the error.
fn streamed_constructor_cast_bytes(
    header: &SafetensorsTensorHeader,
    headers: &std::collections::HashMap<&str, &SafetensorsTensorHeader>,
    compute: Dtype,
    policy: CastPolicy,
) -> u64 {
    let post_read = policy.target(
        &header.name,
        header.shape.len(),
        is_packed_part_header(header),
        is_float(header.dtype),
    );
    constructor_cast_bytes(
        header,
        backbone_constructor_target(&header.name, header, headers, compute),
        post_read,
    )
    .unwrap_or(u64::MAX)
}

/// The seven decoder projection roles share load quantization and stored-affine normalization.
fn tower_layer_projection(base: &str) -> bool {
    let Some(layer) = crate::text_encoder::tower_layer_of(base) else {
        return false;
    };
    let stem = format!(
        "{}layers.{layer}.",
        crate::text_encoder::LANGUAGE_TOWER_PREFIX
    );
    let Some(role) = base.strip_prefix(&stem) else {
        return false;
    };
    matches!(
        role,
        "self_attn.q_proj"
            | "self_attn.k_proj"
            | "self_attn.v_proj"
            | "self_attn.o_proj"
            | "mlp.gate_proj"
            | "mlp.up_proj"
            | "mlp.down_proj"
    )
}

/// Only owners actually consumed by Qwen3-VL constructors normalize stored packed affines.
fn tower_packed_owner(base: &str) -> bool {
    base == "lm_head"
        || base.strip_prefix(crate::text_encoder::LANGUAGE_TOWER_PREFIX) == Some("embed_tokens")
        || tower_layer_projection(base)
}

fn tower_affine_target(
    header: &SafetensorsTensorHeader,
    headers: &std::collections::HashMap<&str, &SafetensorsTensorHeader>,
) -> Option<Dtype> {
    let base = header
        .name
        .strip_suffix(".scales")
        .or_else(|| header.name.strip_suffix(".biases"))?;
    (tower_packed_owner(base)
        && headers
            .get(format!("{base}.weight").as_str())
            .is_some_and(|h| h.shape.len() == 2)
        && headers.contains_key(format!("{base}.scales").as_str()))
    .then_some(Dtype::Bfloat16)
}

fn tower_constructor_cast_bytes(
    header: &SafetensorsTensorHeader,
    headers: &std::collections::HashMap<&str, &SafetensorsTensorHeader>,
) -> u64 {
    // materialize preserves packed parts; from_quantized/from_stored normalize their affine
    // parameters to BF16. Dense leaves have no additional post-materialize cast in StreamedLayer.
    constructor_cast_bytes(header, tower_affine_target(header, headers), None).unwrap_or(u64::MAX)
}

/// The SAME constructor target law used by root runtime and pricing. Keys are logical (depth
/// prefix removed), borrowed from already-held metadata; no source reopen or tensor read.
pub(crate) fn constructor_output_bytes(
    headers: &std::collections::HashMap<&str, &SafetensorsTensorHeader>,
    compute: Dtype,
    keep: impl Fn(&str) -> bool,
) -> Result<u64> {
    headers
        .iter()
        .filter(|(name, _)| keep(name))
        .try_fold(0u64, |sum, (name, header)| {
            let post_read = read_cast(name, header).then_some(compute);
            sum.checked_add(constructor_cast_bytes(
                header,
                backbone_constructor_target(name, header, headers, compute),
                post_read,
            )?)
            .ok_or_else(load_overflow)
        })
}

/// Pending source + every fresh model-compute read cast + constructor/item target. This is an
/// allocation upper for this real stage, not an entry-held lower or a net-resident subtraction.
pub(crate) fn backbone_read_bytes(
    headers: &std::collections::HashMap<&str, &SafetensorsTensorHeader>,
    compute: Dtype,
    keep: impl Fn(&str) -> bool,
) -> Result<u64> {
    headers
        .iter()
        .filter(|(name, _)| keep(name))
        .try_fold(0u64, |sum, (name, h)| {
            let sum = sum
                .checked_add(source_allocation(h.data_bytes)?)
                .ok_or_else(load_overflow)?;
            if read_cast(name, h) && !same_dtype(h.dtype, compute) {
                sum.checked_add(target_bytes(h, compute)?)
                    .ok_or_else(load_overflow)
            } else {
                Ok(sum)
            }
        })
}

pub(crate) fn backbone_load_target_bytes(
    headers: &std::collections::HashMap<&str, &SafetensorsTensorHeader>,
    compute: Dtype,
    keep: impl Fn(&str) -> bool,
) -> Result<u64> {
    backbone_read_bytes(headers, compute, &keep)?
        .checked_add(constructor_output_bytes(headers, compute, keep)?)
        .ok_or_else(load_overflow)
}

/// Return checked logical floor and additional A(raw)-raw + BF16 targets + model rotary row for
/// the IDENTICAL selected owners. Scalar widths/dtype identities are not inferred from width.
pub(crate) fn tower_load_output_bytes(
    headers: &[SafetensorsTensorHeader],
    keep: impl Fn(&str) -> bool,
    rotary_row: u64,
) -> Result<(u64, u64)> {
    let view = headers.iter().map(|h| (h.name.as_str(), h)).collect();
    headers
        .iter()
        .filter(|h| keep(&h.name))
        .try_fold((0u64, rotary_row), |(logical, upper), h| {
            let logical = logical
                .checked_add(h.data_bytes)
                .ok_or_else(load_overflow)?;
            let rounding = source_allocation(h.data_bytes)?
                .checked_sub(h.data_bytes)
                .ok_or_else(load_overflow)?;
            let upper = upper.checked_add(rounding).ok_or_else(load_overflow)?;
            let packed_code = h.shape.len() == 2
                && h.name.strip_suffix(".weight").is_some_and(|base| {
                    headers
                        .iter()
                        .any(|s| s.name.strip_suffix(".scales") == Some(base))
                });
            let target = if h.name.ends_with(".scales") || h.name.ends_with(".biases") {
                tower_affine_target(h, &view)
            } else if packed_code {
                None
            } else {
                Some(Dtype::Bfloat16)
            };
            upper
                .checked_add(constructor_cast_bytes(h, target, None)?)
                .map(|upper| (logical, upper))
                .ok_or_else(load_overflow)
        })
}

/// The downstream loader validates F32 [1,T,L,D], then creates one native from_slice target.
/// The file's raw HOST payload and its BOOL mask are not native source buffers for this upload.
pub(crate) fn empty_prompt_upload_bytes(elements: u64) -> Result<u64> {
    source_allocation(elements.checked_mul(4).ok_or_else(load_overflow)?)
}

/// Untrusted header widths reach these budget helpers before native i32 shape validation.
/// ProjectionLayout::of rounds up with unchecked arithmetic, so prove the alignment fits first;
/// overflow is a genuine arithmetic error here, not a cap or policy. Valid widths are unaffected.
fn checked_group_alignment(input: usize) -> Result<()> {
    let group = crate::nn::QUANT_GROUP_SIZE;
    input
        .div_ceil(group)
        .checked_mul(group)
        .map(|_| ())
        .ok_or_else(load_overflow)
}

/// Every stored-grid allocation is emitted by the same NN operator graph consumed by the
/// native cumulative ledger. Padding and compute→BF16 conversions precede that graph.
pub(crate) fn backbone_quantization_bytes(
    headers: &std::collections::HashMap<&str, &SafetensorsTensorHeader>,
    compute: Dtype,
    quant: Option<crate::nn::BackboneQuant>,
    keep: impl Fn(&str) -> bool,
) -> Result<u64> {
    let Some(quant) = quant else {
        return Ok(0);
    };
    headers
        .iter()
        .filter(|(name, _)| keep(name))
        .try_fold(0u64, |sum, (name, h)| {
            if h.shape.len() != 2 || !name.ends_with(".weight") {
                return Ok(sum);
            }
            let base = name.strip_suffix(".weight").expect("checked");
            if headers
                .keys()
                .any(|key| key.strip_suffix(".scales") == Some(base))
            {
                return Ok(sum);
            }
            checked_group_alignment(h.shape[1])?;
            let crate::nn::ProjectionLayout::Quantized { bits, padded_in } =
                crate::nn::ProjectionLayout::of(h.shape[1], Some(quant))
            else {
                return Ok(sum);
            };
            let (rows, input, padded) = (h.shape[0] as u64, h.shape[1] as u64, padded_in as u64);
            let mut extra = 0u64;
            let mut add = |bytes| {
                extra = extra
                    .checked_add(source_allocation(bytes)?)
                    .ok_or_else(load_overflow)?;
                Ok::<(), Error>(())
            };
            let native_width = mlx_llm::primitives::nn::dtype_bytes(compute);
            if padded != input {
                add(4)?; // default Int32 pad zero
                if compute != Dtype::Int32 {
                    add(native_width)?;
                }
                add(rows
                    .checked_mul(padded)
                    .and_then(|n| n.checked_mul(native_width))
                    .ok_or_else(load_overflow)?)?;
            }
            if compute != Dtype::Bfloat16 {
                add(rows
                    .checked_mul(padded)
                    .and_then(|n| n.checked_mul(2))
                    .ok_or_else(load_overflow)?)?;
                if headers
                    .keys()
                    .any(|key| key.strip_suffix(".bias") == Some(base))
                {
                    add(rows.checked_mul(2).ok_or_else(load_overflow)?)?;
                }
            }
            let quantized = crate::nn::stored_grid_allocation_bytes(
                rows,
                padded,
                crate::nn::QUANT_GROUP_SIZE as u64,
                bits as u64,
            )?;
            sum.checked_add(extra)
                .and_then(|n| n.checked_add(quantized))
                .ok_or_else(load_overflow)
        })
}

pub(crate) fn backbone_quantization_evaluation_bytes(
    headers: &std::collections::HashMap<&str, &SafetensorsTensorHeader>,
    compute: Dtype,
    quant: Option<crate::nn::BackboneQuant>,
    keep: impl Fn(&str) -> bool,
) -> Result<u64> {
    let complete = backbone_quantization_bytes(headers, compute, quant, &keep)?;
    let Some(quant) = quant else {
        return Ok(0);
    };
    let build =
        headers
            .iter()
            .filter(|(name, _)| keep(name))
            .try_fold(0u64, |sum, (name, h)| {
                if h.shape.len() != 2 || !name.ends_with(".weight") {
                    return Ok(sum);
                }
                let base = name.strip_suffix(".weight").expect("checked");
                if headers
                    .keys()
                    .any(|key| key.strip_suffix(".scales") == Some(base))
                {
                    return Ok(sum);
                }
                checked_group_alignment(h.shape[1])?;
                let crate::nn::ProjectionLayout::Quantized { bits, padded_in } =
                    crate::nn::ProjectionLayout::of(h.shape[1], Some(quant))
                else {
                    return Ok(sum);
                };
                let (build, _) = crate::nn::stored_grid_allocation_parts(
                    h.shape[0] as u64,
                    padded_in as u64,
                    crate::nn::QUANT_GROUP_SIZE as u64,
                    bits as u64,
                )?;
                // mlx-rs pad(value=None) eagerly constructs Array::from_int(0) during BUILD; its
                // compute-dtype cast and padded result are lazy and stay in the evaluation stage.
                let pad_zero = if padded_in != h.shape[1] {
                    source_allocation(4)?
                } else {
                    0
                };
                sum.checked_add(build)
                    .and_then(|n| n.checked_add(pad_zero))
                    .ok_or_else(load_overflow)
            })?;
    complete.checked_sub(build).ok_or_else(load_overflow)
}

pub(crate) fn tower_quantization_bytes(
    headers: &[SafetensorsTensorHeader],
    keep: impl Fn(&str) -> bool,
    quant: Option<mlx_llm::primitives::projection::QuantSpec>,
) -> Result<u64> {
    let Some(quant) = quant else {
        return Ok(0);
    };
    headers
        .iter()
        .filter(|h| keep(&h.name))
        .try_fold(0u64, |sum, h| {
            // Only Qwen3-VL LayerPlan decoder projections are load-quantized. Embedding,
            // untied lm_head, norms and any malformed rank-two tensor stay dense BF16.
            let projection = h
                .name
                .strip_suffix(".weight")
                .is_some_and(tower_layer_projection);
            if h.shape.len() != 2 || !projection {
                return Ok(sum);
            }
            let base = h.name.strip_suffix(".weight").expect("checked");
            if headers
                .iter()
                .any(|s| s.name.strip_suffix(".scales") == Some(base))
            {
                return Ok(sum);
            }
            let group = u64::try_from(quant.group_size).map_err(|_| load_overflow())?;
            let bits = u64::try_from(quant.bits).map_err(|_| load_overflow())?;
            sum.checked_add(crate::nn::stored_grid_allocation_bytes(
                h.shape[0] as u64,
                h.shape[1] as u64,
                group,
                bits,
            )?)
            .ok_or_else(load_overflow)
        })
}

/// Pure plan for the selected source closure. The first file is model parameters, subsequent
/// sources have an explicit fresh native F32 empty-prompt upload, counted exactly once.
pub(crate) fn backbone_target_plan(
    sources: &LoadSources,
    prefix: &str,
    compute: Dtype,
    quant: Option<crate::nn::BackboneQuant>,
) -> Result<(u64, u64)> {
    let (mut total, constructors) = if let Some(model) = sources.first() {
        let view = model
            .iter()
            .map(|h| (h.name.strip_prefix(prefix).unwrap_or(&h.name), h))
            .collect();
        let quantized = backbone_quantization_bytes(&view, compute, quant, |_| true)?;
        (
            backbone_load_target_bytes(&view, compute, |_| true)?
                .checked_add(quantized)
                .ok_or_else(load_overflow)?,
            constructor_output_bytes(&view, compute, |_| true)?
                .checked_add(quantized)
                .ok_or_else(load_overflow)?,
        )
    } else {
        (0, 0)
    };
    for prompt in sources.iter().skip(1) {
        if let Some(embedding) = prompt.iter().find(|h| h.name == "embeddings") {
            let elements = embedding.shape.iter().try_fold(1u64, |n, dim| {
                n.checked_mul(*dim as u64).ok_or_else(load_overflow)
            })?;
            total = total
                .checked_add(empty_prompt_upload_bytes(elements)?)
                .ok_or_else(load_overflow)?;
        }
    }
    Ok((total, constructors))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PendingBackbone {
    pub constructors: u64,
    pub evaluation: u64,
    /// Getter targets for the EXACT packed projection/marker set inspected by install.
    /// Completion is a schedule fact, never evidence that a target buffer was retired.
    pub width_outputs: u64,
}

impl PendingBackbone {
    pub(crate) fn before_build_targets(self, installed_width_pass: bool) -> Result<u64> {
        self.constructors
            .checked_sub(if installed_width_pass {
                self.width_outputs
            } else {
                0
            })
            .ok_or_else(load_overflow)
    }
    pub(crate) fn after_build_targets(self) -> Result<u64> {
        self.evaluation
            .checked_sub(self.width_outputs)
            .ok_or_else(load_overflow)
    }
    pub(crate) fn completed_before_evaluation(self) -> Result<u64> {
        self.constructors
            .checked_sub(self.evaluation)
            .and_then(|build| build.checked_add(self.width_outputs))
            .ok_or_else(load_overflow)
    }
}

/// install's successful width pass visits EVERY rank-two weight with rank-two scales and its
/// exact marker key, not merely projections with an attached residual. Reuse that source set;
/// unrelated suffix keys cannot grant completion credit. All facts are already captured here.
pub(crate) fn installed_width_output_bytes(
    headers: &std::collections::HashMap<&str, &SafetensorsTensorHeader>,
    compute: Dtype,
    keep: impl Fn(&str) -> bool,
) -> Result<u64> {
    constructor_output_bytes(headers, compute, |name| {
        if !keep(name) {
            return false;
        }
        let Some(base) = name.strip_suffix(crate::packed::IN_FEATURES_SUFFIX) else {
            return false;
        };
        headers.iter().any(|(key, h)| {
            keep(key) && key.strip_suffix(".weight") == Some(base) && h.shape.len() == 2
        }) && headers.iter().any(|(key, h)| {
            keep(key) && key.strip_suffix(".scales") == Some(base) && h.shape.len() == 2
        })
    })
}

/// Pure source plan (pricing/tests may call it directly): joint early cost plus the pending
/// constructor and post-build evaluation components. Performs no guard check and no I/O.
pub(crate) fn backbone_pending_plan(
    sources: &LoadSources,
    prefix: &str,
    compute: Dtype,
    quant: Option<crate::nn::BackboneQuant>,
) -> Result<(u64, PendingBackbone)> {
    let (total, constructors) = backbone_target_plan(sources, prefix, compute, quant)?;
    let (evaluation, width_outputs) = if let Some(model) = sources.first() {
        let view = model
            .iter()
            .map(|h| (h.name.strip_prefix(prefix).unwrap_or(&h.name), h))
            .collect();
        let width_outputs = installed_width_output_bytes(&view, compute, |_| true)?;
        let complete_quant = backbone_quantization_bytes(&view, compute, quant, |_| true)?;
        let eval_quant = backbone_quantization_evaluation_bytes(&view, compute, quant, |_| true)?;
        let evaluation = constructors
            .checked_sub(complete_quant)
            .and_then(|n| n.checked_add(eval_quant))
            .ok_or_else(load_overflow)?;
        (evaluation, width_outputs)
    } else {
        (0, 0)
    };
    Ok((
        total,
        PendingBackbone {
            constructors,
            evaluation,
            width_outputs,
        },
    ))
}

pub(crate) fn before_backbone_targets(
    sources: Option<LoadSources>,
    prefix: &str,
    compute: Dtype,
    quant: Option<crate::nn::BackboneQuant>,
) -> Result<PendingBackbone> {
    // Disabled: no budget-only planning or overflow arithmetic, even if facts were supplied.
    if !mlx_gen::ios_quantum::enabled() {
        return Ok(PendingBackbone::default());
    }
    let Some(sources) = sources else {
        return Ok(PendingBackbone::default());
    };
    let (total, pending) = backbone_pending_plan(&sources, prefix, compute, quant)?;
    mlx_gen::ios_quantum::before(LOAD_TARGET_QUANTUM, total)?;
    Ok(pending)
}

pub(crate) fn tower_rope_bytes(cfg: &mlx_llm::ModelConfig) -> Result<u64> {
    let dim = u64::try_from(cfg.rotary_dim()).map_err(|_| load_overflow())?;
    source_allocation(dim.checked_mul(4).ok_or_else(load_overflow)?)
}

/// Tower's original raw logical floor is unchanged; the NEW joint check is before the lazy
/// sources, BF16 casts, rotary from_slice and CausalLm's internal verify/eval can allocate.
pub(crate) fn before_tower_load(
    label: &str,
    components: &[std::path::PathBuf],
    keep: impl Fn(&str) -> bool,
    cfg: &mlx_llm::ModelConfig,
    requested: Option<Quant>,
) -> Result<()> {
    if !mlx_gen::ios_quantum::enabled() {
        return Ok(());
    }
    let mut selected = Vec::new();
    for path in components {
        let file = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(path)
            .map_err(|e| Error::Msg(e.to_string()))?;
        selected.extend(file.into_iter().filter(|h| keep(&h.name)));
    }
    let logical = selected.iter().try_fold(0u64, |sum, h| {
        sum.checked_add(h.data_bytes).ok_or_else(load_overflow)
    })?;
    mlx_gen::ios_quantum::before(label, logical)?;
    let (same_logical, additional) =
        tower_load_output_bytes(&selected, |_| true, tower_rope_bytes(cfg)?)?;
    debug_assert_eq!(logical, same_logical);
    let quant = if cfg.quantization.is_none() {
        crate::text_encoder::text_encoder_quant_spec(cfg, requested)?
    } else {
        None
    };
    let quantized = tower_quantization_bytes(&selected, |_| true, quant)?;
    mlx_gen::ios_quantum::before(
        LOAD_TARGET_QUANTUM,
        logical
            .checked_add(additional)
            .and_then(|n| n.checked_add(quantized))
            .ok_or_else(load_overflow)?,
    )
}

/// Refuse a streamed load of a DENSE artifact at a packed tier (see the module docs): the
/// artifact's tier `artifact` must be the requested `requested`, or there must be no request.
pub(crate) fn refuse_stream_requantization(
    id: &str,
    what: &str,
    artifact: Option<Quant>,
    requested: Option<Quant>,
) -> Result<()> {
    if artifact.is_none() && requested.is_some() {
        return Err(Error::Unsupported(format!(
            "{id}: a deferred-materialization (streamed) load reads {what} as stored, and it is \
             the dense artifact while LoadSpec::quantize asks for {requested:?}; streaming would \
             re-quantize every window on every forward — use the packed {requested:?} export \
             (`crate::packed`), or an eager load"
        )));
    }
    Ok(())
}

/// Evaluate `arrays` (a built block's parameters) in groups, so a window's derived arrays are
/// materialized at the read boundary and never inside a forward's command stream.
pub(crate) fn eval_all(arrays: &[&Array]) -> Result<()> {
    for group in arrays.chunks(64) {
        eval(group.iter().copied())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(name: &str, dtype: Stored, shape: &[usize]) -> SafetensorsTensorHeader {
        let width = match dtype {
            Stored::F32 | Stored::U32 | Stored::I32 => 4,
            _ => 2,
        };
        SafetensorsTensorHeader {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            data_bytes: shape.iter().product::<usize>() as u64 * width,
        }
    }

    /// The read rule is the eager loader's cast rule: an f32 matrix read at bf16 holds its bf16
    /// copy and is transiently beside its f32 source; a vector and the text position table stay
    /// as stored; a packed projection's parts are held as stored; a linear's f32 bias is cast by
    /// the constructor, a modulation bias (no sibling projection) is not.
    ///
    /// *Mutations that red this:* casting vectors in the backbone policy; dropping the transient;
    /// pricing a packed part's cast; dropping the linear-bias cast.
    #[test]
    fn a_window_read_prices_the_eager_loaders_casts() {
        let bf16 = CastPolicy::Backbone {
            compute: Dtype::Bfloat16,
        };
        let w = header("blocks.0.attn_proj.weight", Stored::F32, &[4096, 4096]);
        let b = header("blocks.0.attn_proj.bias", Stored::F32, &[4096]);
        let m = header("blocks.0.adaln.bias", Stored::F32, &[4096]);
        let pos = header("y_pos_embedding", Stored::F32, &[300, 4096]);
        let all = [&w, &b, &m, &pos];
        assert_eq!(
            tensor_read(&w, bf16),
            TensorRead {
                held: 4096 * 4096 * 2,
                transient: 4096 * 4096 * 4
            }
        );
        assert_eq!(tensor_read(&b, bf16).transient, 0);
        assert_eq!(tensor_read(&pos, bf16).held, pos.data_bytes);
        assert_eq!(bias_cast(&b, &all, bf16), 4096 * 2);
        assert_eq!(bias_cast(&m, &all, bf16), 0);
        let read = block_read(&all, bf16);
        assert_eq!(
            read.held,
            4096 * 4096 * 2 + 4096 * 4 + 4096 * 2 + 4096 * 4 + pos.data_bytes
        );
        assert_eq!(read.transient, 4096 * 4096 * 4);
        assert_eq!(
            window_read_bytes(&[read, read]),
            2 * read.held + read.transient
        );
        let f32 = CastPolicy::Backbone {
            compute: Dtype::Float32,
        };
        assert_eq!(tensor_read(&w, f32).transient, 0);
        let codes = header("blocks.0.attn_proj.weight", Stored::U32, &[4096, 512]);
        let scales = header("blocks.0.attn_proj.scales", Stored::BF16, &[4096, 64]);
        assert_eq!(tensor_read(&codes, bf16).held, codes.data_bytes);
        assert_eq!(tensor_read(&scales, bf16).transient, 0);
        let tower_norm = header("model.language_model.norm.weight", Stored::F32, &[64]);
        assert_eq!(tensor_read(&tower_norm, CastPolicy::Tower).held, 128);
        assert_eq!(tensor_read(&tower_norm, bf16).held, 256);
    }

    #[test]
    fn the_largest_window_walks_ragged_windows() {
        let b = |held, transient| BlockRead { held, transient };
        let blocks = [b(10, 1), b(20, 5), b(30, 2)];
        assert_eq!(largest_window(&blocks, 1), (32, 30));
        assert_eq!(largest_window(&blocks, 2), (35, 30));
        assert_eq!(largest_window(&blocks, 3), (65, 60));
    }
}

#[cfg(test)]
mod load_target_plan_tests {
    use super::*;
    fn h(name: &str, dtype: Stored, shape: &[usize], data_bytes: u64) -> SafetensorsTensorHeader {
        SafetensorsTensorHeader {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            data_bytes,
        }
    }
    #[test]
    fn actual_float16_compute_is_not_bfloat16_alias() {
        let headers = [
            h("p.weight", Stored::F16, &[64, 64], 8192),
            h("p.bias", Stored::F16, &[64], 128),
        ];
        let view = headers.iter().map(|h| (h.name.as_str(), h)).collect();
        let raw = 8192 + 128;
        assert_eq!(
            backbone_load_target_bytes(&view, Dtype::Float16, |_| true).unwrap(),
            raw
        );
        assert_eq!(
            backbone_load_target_bytes(&view, Dtype::Bfloat16, |_| true).unwrap(),
            2 * raw
        );
    }
    #[test]
    fn prompt_upload_proxy_is_not_a_backbone_read_compute_cast() {
        let sources = vec![
            vec![h("pixel.p.weight", Stored::BF16, &[64, 64], 8192)],
            vec![
                h("embeddings", Stored::F32, &[1, 10, 12, 32], 15360),
                h("mask", Stored::BOOL, &[1, 10], 10),
            ],
        ];
        let (joint, constructors) =
            backbone_target_plan(&sources, "pixel.", Dtype::Bfloat16, None).unwrap();
        assert_eq!(constructors, 0);
        assert_eq!(joint, 8192 + 15360);
    }
    #[test]
    fn matrix_numeric_storage_without_scales_gets_its_constructor_target() {
        let headers = [h("p.weight", Stored::U32, &[64, 64], 16384)];
        let view = headers.iter().map(|h| (h.name.as_str(), h)).collect();
        assert_eq!(
            constructor_output_bytes(&view, Dtype::Bfloat16, |_| true).unwrap(),
            8192
        );
    }
}

#[cfg(test)]
mod pending_quantization_tests {
    use super::*;
    fn h(name: &str, shape: &[usize]) -> SafetensorsTensorHeader {
        SafetensorsTensorHeader {
            name: name.into(),
            dtype: Stored::F32,
            shape: shape.to_vec(),
            data_bytes: shape.iter().product::<usize>() as u64 * 4,
        }
    }
    #[test]
    fn stored_grid_build_and_eval_partition_the_one_source_graph() {
        let (build, evaluate) = crate::nn::stored_grid_allocation_parts(8, 64, 64, 4).unwrap();
        assert_eq!(build, 4 + 4 + 4 * 8);
        assert_eq!(
            crate::nn::stored_grid_allocation_bytes(8, 64, 64, 4).unwrap(),
            build + evaluate
        );
        let mut native = 0;
        crate::nn::visit_stored_grid_allocations(8, 64, 64, 4, false, |_| {
            native += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(native, 18);
    }
    #[test]
    fn dense_quantization_counts_padding_dtype_and_pending_eval_without_build_constants() {
        let headers = [h("p.weight", &[8, 65]), h("p.bias", &[8])];
        let view = headers.iter().map(|h| (h.name.as_str(), h)).collect();
        let quant = crate::nn::BackboneQuant::for_load(Some(Quant::Q4)).unwrap();
        let padded_complete =
            backbone_quantization_bytes(&view, Dtype::Float16, quant, |_| true).unwrap();
        let padded_eval =
            backbone_quantization_evaluation_bytes(&view, Dtype::Float16, quant, |_| true).unwrap();
        let (build, _) = crate::nn::stored_grid_allocation_parts(8, 128, 64, 4).unwrap();
        // Padded projection: the eager Int32 pad zero (4 B) is a BUILD constructor too.
        assert_eq!(padded_complete - padded_eval, build + 4);
        assert_eq!(
            padded_complete - crate::nn::stored_grid_allocation_bytes(8, 128, 64, 4).unwrap(),
            4 + 2 + 8 * 128 * 2 + 8 * 128 * 2 + 8 * 2
        );
        let aligned = [h("q.weight", &[8, 64])];
        let aligned = aligned.iter().map(|h| (h.name.as_str(), h)).collect();
        let complete =
            backbone_quantization_bytes(&aligned, Dtype::Float16, quant, |_| true).unwrap();
        let eval =
            backbone_quantization_evaluation_bytes(&aligned, Dtype::Float16, quant, |_| true)
                .unwrap();
        let (build, _) = crate::nn::stored_grid_allocation_parts(8, 64, 64, 4).unwrap();
        assert_eq!(complete - eval, build, "no pad constructor without padding");
        assert_eq!(
            complete - crate::nn::stored_grid_allocation_bytes(8, 64, 64, 4).unwrap(),
            8 * 64 * 2
        );
    }
}

#[cfg(test)]
mod disabled_guard_tests {
    use super::*;
    #[test]
    fn disabled_new_guards_do_not_read_missing_sources_or_compute_overflowing_budgets() {
        assert!(!mlx_gen::ios_quantum::enabled());
        let missing = std::path::PathBuf::from("/definitely/missing/iris/model.safetensors");
        assert_eq!(
            before_materialized_load("raw", std::slice::from_ref(&missing), |_| true).unwrap(),
            None
        );
        assert_eq!(
            before_backbone_targets(None, "", Dtype::Bfloat16, None).unwrap(),
            PendingBackbone::default()
        );
        // Supplied facts whose budget arithmetic would overflow are never evaluated off path.
        let overflowing = SafetensorsTensorHeader {
            name: "p.weight".into(),
            dtype: Stored::F32,
            shape: vec![usize::MAX, usize::MAX],
            data_bytes: u64::MAX,
        };
        assert!(
            backbone_pending_plan(&vec![vec![overflowing.clone()]], "", Dtype::Bfloat16, None)
                .is_err()
        );
        assert_eq!(
            before_backbone_targets(Some(vec![vec![overflowing]]), "", Dtype::Bfloat16, None)
                .unwrap(),
            PendingBackbone::default()
        );
        before_export_targets(u64::MAX, usize::MAX).unwrap();
        let mut called = false;
        crate::nn::before_pending_construction(|| {
            called = true;
            Err(load_overflow())
        })
        .unwrap();
        assert!(!called);
        // The original raw API is untouched; this provider's tower helper alone is skipped.
        let cfg = mlx_llm::ModelConfig::from_dir(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/tiny-snapshot/text_encoder"),
        )
        .unwrap();
        before_tower_load("raw", &[missing], |_| true, &cfg, Some(Quant::Q4)).unwrap();
    }
}

#[cfg(test)]
mod tower_quantization_role_tests {
    use super::*;
    fn h(name: &str, shape: &[usize]) -> SafetensorsTensorHeader {
        SafetensorsTensorHeader {
            name: name.into(),
            dtype: Stored::F32,
            shape: shape.to_vec(),
            data_bytes: shape.iter().product::<usize>() as u64 * 4,
        }
    }
    #[test]
    fn only_decoder_layer_projections_receive_load_quantization_graphs() {
        let p = crate::text_encoder::LANGUAGE_TOWER_PREFIX;
        let projection = format!("{p}layers.0.self_attn.q_proj.weight");
        let headers = [
            h(&projection, &[64, 64]),
            h("lm_head.weight", &[64, 64]),
            h(&format!("{p}embed_tokens.weight"), &[64, 64]),
            h(&format!("{p}layers.0.input_layernorm.weight"), &[64, 64]),
        ];
        let quant = mlx_llm::primitives::projection::QuantSpec {
            group_size: 64,
            bits: 4,
            grid: mlx_llm::primitives::projection::QuantGrid::Stored,
        };
        assert_eq!(
            tower_quantization_bytes(&headers, |_| true, Some(quant)).unwrap(),
            crate::nn::stored_grid_allocation_bytes(64, 64, 64, 4).unwrap()
        );
    }
}

#[cfg(test)]
mod quantization_alignment_overflow_tests {
    use super::*;
    #[test]
    fn untrusted_width_alignment_overflow_is_an_error_not_a_panic() {
        let header = SafetensorsTensorHeader {
            name: "p.weight".into(),
            dtype: Stored::F32,
            shape: vec![0, usize::MAX],
            data_bytes: 0,
        };
        let view = [(header.name.as_str(), &header)].into_iter().collect();
        let quant = crate::nn::BackboneQuant::for_load(Some(Quant::Q4)).unwrap();
        assert!(backbone_quantization_bytes(&view, Dtype::Bfloat16, quant, |_| true).is_err());
        assert!(
            backbone_quantization_evaluation_bytes(&view, Dtype::Bfloat16, quant, |_| true)
                .is_err()
        );
        assert!(
            backbone_pending_plan(&vec![vec![header.clone()]], "", Dtype::Bfloat16, quant).is_err()
        );
        checked_group_alignment(65).unwrap();
    }
}

#[cfg(test)]
mod completed_width_plan_tests {
    use super::*;
    #[test]
    fn late_pending_excludes_only_already_executed_width_getter_targets() {
        for (dtype, shape, bytes, width_target) in [
            (Stored::I32, vec![1], 4, 0),
            (Stored::I64, vec![1], 8, 4),
            (Stored::F32, vec![1], 4, 4),
            (Stored::I32, vec![1, 1], 4, 4),
        ] {
            let mut source = vec![
                SafetensorsTensorHeader {
                    name: "p.weight".into(),
                    dtype: Stored::U32,
                    shape: vec![4, 8],
                    data_bytes: 128,
                },
                SafetensorsTensorHeader {
                    name: "p.scales".into(),
                    dtype: Stored::BF16,
                    shape: vec![4, 1],
                    data_bytes: 8,
                },
                SafetensorsTensorHeader {
                    name: "p.biases".into(),
                    dtype: Stored::BF16,
                    shape: vec![4, 1],
                    data_bytes: 8,
                },
                SafetensorsTensorHeader {
                    name: "p.in_features".into(),
                    dtype,
                    shape,
                    data_bytes: bytes,
                },
            ];
            // This unrelated key is not a packed projection getter and grants NO completion.
            source.push(SafetensorsTensorHeader {
                name: "unrelated.in_features".into(),
                dtype: Stored::I64,
                shape: vec![1],
                data_bytes: 8,
            });
            let sources = vec![source];
            let (joint, plan) = backbone_pending_plan(&sources, "", Dtype::Bfloat16, None).unwrap();
            let view = sources[0].iter().map(|h| (h.name.as_str(), h)).collect();
            assert_eq!(plan.width_outputs, width_target);
            assert_eq!(
                joint,
                backbone_load_target_bytes(&view, Dtype::Bfloat16, |_| true).unwrap()
            );
            assert_eq!(plan.before_build_targets(false).unwrap(), plan.constructors);
            assert_eq!(
                plan.before_build_targets(true).unwrap() + width_target,
                plan.constructors
            );
            assert_eq!(
                plan.after_build_targets().unwrap() + width_target,
                plan.evaluation
            );
            assert_eq!(plan.completed_before_evaluation().unwrap(), width_target);
        }
        let mixed = PendingBackbone {
            constructors: 100,
            evaluation: 70,
            width_outputs: 4,
        };
        assert_eq!(mixed.before_build_targets(true).unwrap() + 4, 100);
        assert_eq!(mixed.after_build_targets().unwrap(), 66);
        assert_eq!(mixed.completed_before_evaluation().unwrap(), 34);
        assert_eq!(
            mixed.after_build_targets().unwrap() + mixed.completed_before_evaluation().unwrap(),
            100
        );
        let bad = PendingBackbone {
            constructors: 0,
            evaluation: 0,
            width_outputs: 4,
        };
        assert!(bad.before_build_targets(true).is_err());
        assert!(bad.after_build_targets().is_err());
    }
}

#[cfg(test)]
mod selected_constructor_plan_tests {
    use super::*;

    fn h(name: &str, dtype: Stored, shape: &[usize], bytes: u64) -> SafetensorsTensorHeader {
        SafetensorsTensorHeader {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            data_bytes: bytes,
        }
    }

    #[test]
    fn window_constructor_targets_use_actual_read_policy_not_root_matrix_casts() {
        let policy = CastPolicy::Backbone {
            compute: Dtype::Bfloat16,
        };
        for (dtype, shape, bytes, expected) in [
            (Stored::I32, vec![1], 4, 0),
            (Stored::I64, vec![1], 8, 4),
            (Stored::I32, vec![1, 1], 4, 0),
            (Stored::F32, vec![1, 1], 4, 4),
        ] {
            let header = h("p.in_features", dtype, &shape, bytes);
            let view = [(header.name.as_str(), &header)].into_iter().collect();
            let read = tensor_read(&header, policy);
            let block = block_read(&[&header], policy);
            assert_eq!(block.held, read.held + expected);
            assert_eq!(block.transient, read.transient);
            if dtype == Stored::I32 && shape.len() == 2 {
                assert_eq!(
                    constructor_output_bytes(&view, Dtype::Bfloat16, |_| true).unwrap(),
                    4
                );
                assert_eq!(
                    policy.target(&header.name, 2, false, false),
                    None,
                    "integer window reads remain unchanged, unlike eager root reads"
                );
            }
        }
        let norm = h("norm.weight", Stored::BF16, &[64], 128);
        assert_eq!(block_read(&[&norm], policy).held, 128 + 256);
        let codes = h("p.weight", Stored::U32, &[64, 8], 2048);
        let scales = h("p.scales", Stored::BF16, &[64, 1], 128);
        let bias = h("p.bias", Stored::F16, &[64], 128);
        assert_eq!(
            block_read(&[&codes, &scales, &bias], policy).held,
            2048 + 128 + 128 + 128,
            "F16 packed bias to BF16 is a fresh same-width target"
        );
        let overflow = h("norm.weight", Stored::BF16, &[usize::MAX], 0);
        assert_eq!(
            block_read(&[&overflow], policy).held,
            u64::MAX,
            "an overflowing constructor upper cannot become an affordable window"
        );
        let overflow_read = h("p.weight", Stored::F32, &[usize::MAX, usize::MAX], 0);
        assert_eq!(
            block_read(&[&overflow_read], policy).held,
            u64::MAX,
            "read casts retain the same fail-closed overflow contract"
        );
    }

    #[test]
    fn packed_tower_affines_cast_for_real_layer_embedding_and_head_owners() {
        for base in [
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.self_attn.k_proj",
            "model.language_model.layers.0.self_attn.v_proj",
            "model.language_model.layers.0.self_attn.o_proj",
            "model.language_model.layers.0.mlp.gate_proj",
            "model.language_model.layers.0.mlp.up_proj",
            "model.language_model.layers.0.mlp.down_proj",
            "model.language_model.embed_tokens",
            "lm_head",
        ] {
            for dtype in [Stored::F16, Stored::F32, Stored::BF16] {
                let width = if dtype == Stored::F32 { 4 } else { 2 };
                let headers = [
                    h(&format!("{base}.weight"), Stored::U32, &[64, 8], 2048),
                    h(&format!("{base}.scales"), dtype, &[64, 1], 64 * width),
                    h(&format!("{base}.biases"), dtype, &[64, 1], 64 * width),
                ];
                let fresh = if dtype == Stored::BF16 { 0 } else { 2 * 128 };
                let (raw, extra) = tower_load_output_bytes(&headers, |_| true, 0).unwrap();
                assert_eq!(raw, 2048 + 2 * 64 * width);
                assert_eq!(
                    extra, fresh,
                    "{base} {dtype:?}: codes stay raw; each affine normalizes"
                );
                let refs: Vec<_> = headers.iter().collect();
                assert_eq!(block_read(&refs, CastPolicy::Tower).held, raw + fresh);
                assert_eq!(block_read(&refs, CastPolicy::Tower).transient, 0);
            }
        }
        assert!(!tower_layer_projection(
            "model.language_model.layers.0.unused.self_attn.q_proj"
        ));
        let unrelated = [
            h(
                "model.language_model.layers.0.unused.weight",
                Stored::U32,
                &[64, 8],
                2048,
            ),
            h(
                "model.language_model.layers.0.unused.scales",
                Stored::F32,
                &[64, 1],
                256,
            ),
            h(
                "model.language_model.layers.0.unused.biases",
                Stored::F32,
                &[64, 1],
                256,
            ),
        ];
        assert_eq!(
            tower_load_output_bytes(&unrelated, |_| true, 0).unwrap().1,
            0,
            "suffixes alone do not invent an unexecuted packed affine constructor"
        );
    }

    #[test]
    fn width_completion_uses_only_selected_projection_owners() {
        let headers = [
            h("blocks.0.attn_proj.weight", Stored::U32, &[64, 8], 2048),
            h("blocks.0.attn_proj.scales", Stored::BF16, &[64, 1], 128),
            h("blocks.0.attn_proj.in_features", Stored::I64, &[1], 8),
            h("norm.weight", Stored::F32, &[64], 256),
        ];
        let view = headers.iter().map(|h| (h.name.as_str(), h)).collect();
        assert_eq!(
            installed_width_output_bytes(&view, Dtype::Bfloat16, |_| true).unwrap(),
            4
        );
        let skeleton = |name: &str| crate::dit::Stack::of(name).is_none();
        assert_eq!(
            installed_width_output_bytes(&view, Dtype::Bfloat16, skeleton).unwrap(),
            0
        );
        let selected: Vec<_> = headers.into_iter().filter(|h| skeleton(&h.name)).collect();
        let (_, pending) =
            backbone_pending_plan(&vec![selected], "", Dtype::Bfloat16, None).unwrap();
        assert_eq!(pending.width_outputs, 0);
        assert_eq!(
            pending.after_build_targets().unwrap(),
            0,
            "unloaded stack getters cannot be subtracted from a skeleton"
        );
    }
}
