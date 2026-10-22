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
    /// [`crate::depth::load_depth_export`]): every matrix (rank ≥ 2) but the text position table
    /// and a packed projection's stored parts to the compute dtype; vectors as stored.
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
            let elements = header.shape.iter().map(|&d| d as u64).product::<u64>();
            TensorRead {
                held: crate::retention::allocated(elements * width(target)),
                transient: crate::retention::allocated(header.data_bytes),
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

/// One linear bias the block constructor casts AFTER the read (the backbone's `Loader::linear`
/// reads a vector bias at the compute dtype; a packed projection's bias at bf16): bytes of the
/// new buffer, `0` when the stored dtype already is the target. Only the backbone has biases on
/// its projections; the Qwen3-VL tower has none.
fn bias_cast(
    header: &SafetensorsTensorHeader,
    headers: &[&SafetensorsTensorHeader],
    policy: CastPolicy,
) -> u64 {
    let CastPolicy::Backbone { compute } = policy else {
        return 0;
    };
    let Some(base) = header.name.strip_suffix(".bias") else {
        return 0;
    };
    let weight = format!("{base}.weight");
    let Some(w) = headers.iter().find(|h| h.name == weight) else {
        return 0;
    };
    if w.shape.len() != 2 {
        return 0;
    }
    let packed = headers.iter().any(|h| h.name == format!("{base}.scales"));
    let target = if packed { Dtype::Bfloat16 } else { compute };
    match stored_width(header.dtype) {
        Some(stored) if stored != crate::nn::dtype_bytes(target) => {
            let elements = header.shape.iter().map(|&d| d as u64).product::<u64>();
            crate::retention::allocated(elements * crate::nn::dtype_bytes(target))
        }
        _ => 0,
    }
}

/// The price of one block's read: what the materialized block holds, and the largest source
/// tensor alive beside its cast while it is read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockRead {
    /// Bytes the built block holds (its materialized tensors plus the bias casts its constructor
    /// makes).
    pub held: u64,
    /// The largest single [`TensorRead::transient`] of the block.
    pub transient: u64,
}

/// [`BlockRead`] of the tensors `headers` names, under `policy`.
pub(crate) fn block_read(headers: &[&SafetensorsTensorHeader], policy: CastPolicy) -> BlockRead {
    let mut read = BlockRead::default();
    for header in headers {
        let t = tensor_read(header, policy);
        read.held = read
            .held
            .saturating_add(t.held)
            .saturating_add(bias_cast(header, headers, policy));
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
) -> Result<()> {
    if !mlx_gen::ios_quantum::enabled() {
        return Ok(());
    }
    let total = components.iter().try_fold(0u64, |total, path| {
        let bytes = materialized_load_bytes(&headers(path)?, &keep)?;
        total.checked_add(bytes).ok_or_else(|| {
            Error::Unsupported("Iris component load byte arithmetic overflow".into())
        })
    })?;
    mlx_gen::ios_quantum::before(label, total)
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
