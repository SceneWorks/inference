//! SDXL adapter application — LoRA (sc-2639, a faithful Rust port of the vendored SceneWorks
//! `lora.py` merge for the mlx-examples SDXL U-Net) and LoKr (sc-2640).
//!
//! **LoKr** (`<module>.lokr_w1/w2` (+ low-rank `_a`/`_b`), `networkType=lokr` + `alpha`/`rank` meta):
//! the vendored SDXL path *rejects* LoKr, so there is no fork to match — Rust is strictly more
//! capable. The delta is reconstructed with the validated LyCORIS formula (`reconstruct_lokr_delta`,
//! in f32) and merged (`W += δ·scale`) at the U-Net's load dtype, chaos-safe like LoRA. Keys
//! resolve through the same kohya table (`lora_unet_<flat>.lokr_*`) or bare/PEFT dotted paths.
//!
//! **LoRA** — two on-disk formats, both **merged into the dense U-Net weights at load** (`W += δ`,
//! NOT a forward-time residual): the delta is reconstructed in f32 and merged at the U-Net's load
//! dtype — **fp16 in production** (`model::load`), f32 in the stage gates. SDXL's ancestral sampler
//! is chaos-sensitive, and a residual's `W·x + δ·x` differs from the merged `(W+δ)·x` by ~1 ULP,
//! which cascades to a visible whole-image divergence. Merging reproduces the vendored merged-weight
//! forward bit-for-bit.
//!
//! - **kohya** (`lora_unet_<diffusers path, `.`→`_`>.lora_down/up.weight` + optional `.alpha`) — what
//!   `pipe.save_lora_weights()` and most HF community SDXL LoRAs (incl. LCM-LoRA) ship. The
//!   `_`-flattening is ambiguous (diffusers names like `down_blocks`/`transformer_blocks` already
//!   contain `_`), so the flattened stem is resolved against a table built by flattening every
//!   routable module path — the Rust equivalent of the vendored `unet.named_modules()` walk.
//! - **PEFT** (`base_model.model.unet.<dotted path>.lora_A/B.default.weight` + optional `.alpha`) —
//!   what `peft.save_pretrained()` / SceneWorks' `_SdxlLoraBackend` emit. The dotted path resolves
//!   directly. (kohya `lora_down`/`lora_up` == PEFT `lora_A`/`lora_B`.)
//!
//! Two coverage modes (see [`LoraCoverage`]):
//! - [`LoraCoverage::Complete`] (sc-2671 + sc-2919) — **the `model::load` default** (Michael's
//!   correctness-over-parity call, 2026-06-03): applies SDXL LoRAs in **full**, matching diffusers.
//!   On top of the vendored-reachable set it routes `mid_block.attentions.0` (attention + proj), the
//!   GEGLU feed-forward (`ff.net.0.proj` row-split into the value/gate halves `linear1`/`linear2`,
//!   `ff.net.2` → `linear3`) of every cross-attention transformer, **and the conv-layer LoRAs**
//!   (sc-2919) — resnet `conv1`/`conv2`/`conv_shortcut`, the down/up-samplers, `conv_in`/`conv_out`.
//!   Conv deltas are fused with [`conv_lora_delta`] and folded into the conv weights; everything else
//!   uses the same proven-bit-exact Linear merge (plus the FF row-split, a bit-exact gather of `B@A`).
//! - [`LoraCoverage::Vendored`] matches the vendored reachable surface **exactly** (515 modules on
//!   LCM-LoRA): down/up attention (`to_q/k/v`, `to_out.0`), `proj_in`/`proj_out`, resnet
//!   `time_emb_proj`. No `mid_block` (the vendored mlx-examples UNet names it `mid_blocks.1.…` so
//!   diffusers keys miss it), no ff/GEGLU, **no conv**, no text-encoder. `model::load` selects this
//!   only when `SDXL_LORA_VENDORED` is set — the escape hatch for byte-parity with the retired Python
//!   (Linear-only) path.
//!
//! Either way, out-of-surface keys (and, under vendored coverage, conv-shaped keys) are counted as
//! skipped and surfaced in the returned [`SdxlLoraReport`] — never silently dropped. **LoKr stays at
//! the vendored-equivalent (Linear) surface regardless of coverage** (sc-2671/sc-2919 are LoRA-only;
//! sc-2640 covered LoKr at that surface).

use std::collections::BTreeMap;

use mlx_rs::ops::{matmul, multiply};
use mlx_rs::{Array, Dtype};

#[cfg(test)]
use mlx_gen::adapters::loader::{is_loha_keys, is_lokr, is_lokr_keys};
use mlx_gen::adapters::loader::{parse_loha_thirdparty, parse_lokr_thirdparty, resolve_lokr_path};
use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::adapters::{
    build_lokr_factors, conv_lora_delta, reconstruct_lokr_delta, AdaptableHost, Adapter,
};
use mlx_gen::array::scalar;
use mlx_gen::gen_core::weightsmeta::{
    resolve_kohya_stem, LoraAdapterMeta, LOHA_TP_SUFFIXES, LOKR_TP_SUFFIXES,
    LORA_ADAPTER_METADATA_KEY,
};
use mlx_gen::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::unet::UNet2DConditionModel;

const KOHYA_PREFIX: &str = "lora_unet_";
const PEFT_PREFIX: &str = "base_model.model.unet.";

/// The SDXL install dispatch, shared by the payload loader and header-only semantic exporter.
/// In particular an unstamped Kohya LoRA never overrides a declared `Lokr` kind. Third-party
/// LyCORIS detection precedes the declared kind, exactly as the executable install path does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InstallKind {
    Lora,
    Lokr,
    ThirdPartyLokr,
    ThirdPartyLoha,
}

pub(crate) fn install_kind<'a>(
    kind: AdapterKind,
    network_type: Option<&str>,
    keys: impl IntoIterator<Item = &'a str>,
) -> Result<InstallKind> {
    use mlx_gen::gen_core::weightsmeta::{
        is_lokr_network_type, keys_contain_loha, keys_contain_lokr,
    };
    let keys: Vec<&str> = keys.into_iter().collect();
    let stamped_lokr = is_lokr_network_type(network_type);
    if !stamped_lokr && keys_contain_lokr(keys.iter().copied()) {
        return Ok(InstallKind::ThirdPartyLokr);
    }
    if keys_contain_loha(keys.iter().copied()) {
        return Ok(InstallKind::ThirdPartyLoha);
    }
    match kind {
        AdapterKind::Lokr => Ok(InstallKind::Lokr),
        AdapterKind::Lora if stamped_lokr => Err(Error::Msg(
            "sdxl: adapter declared Lora but its metadata says networkType=lokr".into(),
        )),
        AdapterKind::Lora => Ok(InstallKind::Lora),
    }
}

/// LoKr per-module factor suffixes (each factor is full `lokr_w1`/`lokr_w2` or a low-rank
/// `_a`/`_b` product). Exact-suffix matched; longest-first so `.lokr_w1_a` wins over `.lokr_w1`.
const LOKR_SUFFIXES: [&str; 6] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
];

#[derive(Clone, Copy)]
enum Role {
    Down,
    Up,
    Alpha,
}

#[derive(Default)]
struct LoraTriple {
    down: Option<Array>, // A: [rank, in]
    up: Option<Array>,   // B: [out, rank]
    alpha: Option<f32>,
}

/// Outcome of applying the SDXL adapter specs: how many module weights were merged, and how many
/// keys fell outside the routable surface (mid_block / ff / conv / text-encoder — surfaced, not
/// silently dropped). `merged` counts *base Linears updated*, so a GEGLU `ff.net.0.proj` LoRA (when
/// complete coverage row-splits it into `linear1`+`linear2`) contributes 2.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct SdxlLoraReport {
    pub merged: usize,
    pub skipped_keys: usize,
}

/// How much of the U-Net LoRA surface to reach.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LoraCoverage {
    /// Faithful to the vendored `lora.py` (515 modules on LCM-LoRA): no `mid_block`, no GEGLU FF.
    /// Byte-parity with the retired Python SDXL path — the `model::load` escape hatch
    /// (`SDXL_LORA_VENDORED`), no longer the default.
    Vendored,
    /// Strictly more correct than the vendored path (sc-2671 + sc-2919): also routes
    /// `mid_block.attentions.0`, the GEGLU feed-forward of every cross-attention transformer, and the
    /// **conv-layer LoRAs** (resnet convs, samplers, `conv_in`/`conv_out`). **The `model::load`
    /// default** — applies SDXL LoRAs in full (matching diffusers).
    Complete,
}

/// The actual load policy, also frozen into configured-overlay semantic facts.
pub(crate) fn configured_lora_coverage() -> LoraCoverage {
    if std::env::var_os("SDXL_LORA_VENDORED").is_some() {
        LoraCoverage::Vendored
    } else {
        LoraCoverage::Complete
    }
}

/// A diffusers module path the vendored `lora.py` cannot reach — `mid_block.*` (named `mid_blocks.*`
/// in mlx-examples) or a GEGLU FF leaf (`*.ff.net.*`). Gated out under [`LoraCoverage::Vendored`].
fn is_complete_only(path: &str) -> bool {
    path.starts_with("mid_block") || path.contains(".ff.net.")
}

/// Build the kohya `flattened→dotted` lookup table from a routable-path list.
fn build_table(paths: Vec<String>) -> BTreeMap<String, String> {
    paths
        .into_iter()
        .map(|p| (p.replace('.', "_"), p))
        .collect()
}

/// The actual installed target geometry, before any adapter is applied. Module paths are stable
/// semantic identifiers, never filesystem paths. A source affine pack and a later load-time pack
/// are deliberately distinct because only the former installs forward-time residuals.
#[derive(Clone, Debug)]
struct HeaderTarget {
    shape: Vec<i32>,
    linear: bool,
    source_packed_bits: Option<i32>,
    weight_dtype: Dtype,
    scales_dtype: Option<Dtype>,
    affine_biases_dtype: Option<Dtype>,
    /// Program read state, not a backing/release certificate: raw accessed Dir leaves are read;
    /// fused sources, deferred bodies and constructor-created view/take outputs can still be lazy.
    unread: bool,
    packing: serde_json::Value,
}

fn header_targets(
    headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
    cfg: &crate::config::UNetConfig,
    spec: &mlx_gen::gen_core::LoadSpec,
) -> Result<(
    BTreeMap<String, HeaderTarget>,
    BTreeMap<String, String>,
    BTreeMap<String, String>,
)> {
    let (vendored, complete, convs) = crate::unet::adapter_target_paths_from_headers(cfg, headers);
    let vendored_table = build_table(vendored);
    let mut complete_paths = complete.clone();
    complete_paths.extend(convs.iter().cloned());
    let complete_table = build_table(complete_paths);
    let mut targets = BTreeMap::new();
    let fused = matches!(&spec.weights, mlx_gen::WeightsSource::File(_));
    let streamed = crate::memory_strategy::streamable(spec);
    for path in complete {
        let projection = crate::quant::linear_from_headers(headers, &path)?;
        if path.ends_with(".ff.net.0.proj") {
            // This is the base-model constructor's EVEN split, not the adapter UP floor split.
            crate::quant::geglu_half_ranges(projection.shape[0])?;
        }
        for (target, rows) in lora_output_rows(&path, projection.shape[0]) {
            let mut shape = projection.shape.to_vec();
            if let Some((lo, hi)) = rows {
                shape[0] = hi - lo;
            }
            let unread = fused || rows.is_some()
                || (streamed && crate::loader::is_streamed_block_tensor(&target));
            targets.insert(target, HeaderTarget {
                shape, linear: true, source_packed_bits: projection.source_packed_bits,
                unread,
                weight_dtype: projection.weight_dtype, scales_dtype: projection.scales_dtype,
                affine_biases_dtype: projection.affine_biases_dtype,
                packing: serde_json::json!({"codes":projection.packed_codes_shape,"scales":projection.packed_scales_shape,
                    "biases":projection.packed_biases_shape,"rows":rows,"group_size":crate::quant::GROUP_SIZE}),
            });
        }
    }
    for path in convs {
        let name = format!("{path}.weight");
        let header = headers
            .iter()
            .find(|h| h.name == name)
            .ok_or_else(|| Error::MissingTensor(name))?;
        let mut shape: Vec<i32> = header
            .shape
            .iter()
            .map(|dim| {
                i32::try_from(*dim)
                    .map_err(|_| Error::Msg("sdxl conv dimension exceeds native i32".into()))
            })
            .collect::<Result<_>>()?;
        let shortcut = path.ends_with(".conv_shortcut");
        if shortcut {
            // ResnetBlock2D::from_weights reshapes this weight to its first two axes.
            if shape.len() < 2
                || shape[2..]
                    .iter()
                    .try_fold(1i64, |n, d| n.checked_mul(i64::from(*d)))
                    != Some(1)
            {
                return Err(Error::Msg(
                    "sdxl conv_shortcut cannot reshape to its Linear axes".into(),
                ));
            }
            shape.truncate(2);
        } else if shape.len() != 4 {
            return Err(Error::Msg(
                "sdxl NCHW convolution weight must have four axes".into(),
            ));
        }
        targets.insert(
            path,
            HeaderTarget {
                shape,
                linear: shortcut,
                source_packed_bits: None,
                weight_dtype: crate::quant::loaded_header_dtype(headers, header)?,
                scales_dtype: None,
                affine_biases_dtype: None,
                unread: true, // Constructor transpose/shortcut reshape is still a lazy graph.
                packing: serde_json::Value::Null,
            },
        );
    }
    Ok((targets, vendored_table, complete_table))
}

/// Rows `[lo, hi)` of a 2-D array. Bit-exact for slicing a `B@A` LoRA delta: matmul output rows are
/// independent, so `rows(B@A)` equals `(rows(B))@A` to the last bit.
fn rows(a: &Array, lo: i32, hi: i32) -> Result<Array> {
    let idx = Array::from_slice(&(lo..hi).collect::<Vec<i32>>(), &[hi - lo]);
    Ok(a.take_axis(&idx, 0)?)
}

// -------------------------------------------------------------------------------------------------
// sc-26270: every adapter file read and every per-target install is a guarded iOS quantum, priced
// from the code below with `quantum::adapter_install_bytes` (the target-sized buffers each install
// creates, at their widths). The guards are pass-throughs off device: every install builds and
// merges exactly the graph it always did.
// -------------------------------------------------------------------------------------------------

/// Bytes per element of `dtype`.
fn width(dtype: Dtype) -> u64 {
    crate::quantum::dtype_bytes(dtype)
}

/// Bytes of the f32 copy `a.as_dtype(Float32)` makes (none when `a` already is f32).
fn f32_copy(a: &Array) -> u64 {
    if a.dtype() == Dtype::Float32 {
        0
    } else {
        crate::quantum::elements(a).saturating_mul(4)
    }
}

/// The stored bytes of `base` when it has not been read yet: a lazily loaded base (the fused-LDM
/// route merges before the U-Net is materialized) is read inside the merge.
fn unread(base: &Array) -> u64 {
    if mlx_gen::array::is_materialized(base) {
        0
    } else {
        crate::quantum::bytes(base)
    }
}

/// [`lora_delta`] / [`conv_lora_delta`]'s target-sized buffers for `src`-typed factors: the f32
/// `up @ down` product and, below f32, its round trip through `src` (the scale multiply donates).
fn lora_chain(src: Dtype) -> Vec<u64> {
    if src == Dtype::Float32 {
        vec![4]
    } else {
        vec![4, width(src), 4]
    }
}

/// A Kronecker (LoKr) reconstruction's target-sized buffers for `factor`-typed factors, then the
/// caller's strength multiply: the `kron` product, its f32 scale multiply and the strength multiply
/// — neither multiply can donate, as each one's operand is a reshape VIEW of the previous buffer
/// (measured).
fn kron_chain(factor: Dtype) -> Vec<u64> {
    vec![width(factor), 4, 4]
}

/// A Hadamard (LoHa) reconstruction's, then the caller's strength multiply: the two low-rank
/// products (their elementwise product donates one), below f32 the promoting scale multiply (at
/// f32 it donates), and the strength multiply over the reshaped view.
fn hadamard_chain(factor: Dtype) -> Vec<u64> {
    let mut chain = vec![width(factor), width(factor)];
    if factor != Dtype::Float32 {
        chain.push(4);
    }
    chain.push(4);
    chain
}

/// [`AdaptableLinear::merge_dense_delta`] of an f32 delta into a `base`-typed weight: the delta's
/// cast (which the sum then donates) below f32; at f32 the sum donates the delta itself.
fn linear_merge(base: Dtype) -> Option<u64> {
    (base != Dtype::Float32).then(|| width(base))
}

/// The promoted dtype of the third-party factors stored under `raw` + one of `suffixes` in `w`,
/// and the bytes of their factor-sized buffers (every factor's f32 copy or low-rank product,
/// bounded by its own size times the rank-product fan-out).
fn factor_stats(w: &Weights, raw: &str, suffixes: &[&str]) -> (Dtype, u64) {
    let mut dtype: Option<Dtype> = None;
    let mut small = 0u64;
    for suffix in suffixes {
        if let Some(t) = w.get(&format!("{raw}{suffix}")) {
            dtype = Some(dtype.map_or(t.dtype(), |d| Dtype::from_promoting_types(d, t.dtype())));
            small = small.saturating_add(crate::quantum::elements(t).saturating_mul(4));
        }
    }
    (dtype.unwrap_or(Dtype::Float32), small)
}

/// The products of the low-rank factor pairs `(a, b)` (`a @ b`) present in `pairs`, at `factor`'s
/// width — the small Kronecker factors a LoKr rebuilds before its `kron`.
fn product_bytes(pairs: &[(Option<&Array>, Option<&Array>)], factor: Dtype) -> u64 {
    pairs
        .iter()
        .filter_map(|(a, b)| {
            Some((a.as_ref()?.shape()[0] as u64) * (b.as_ref()?.shape()[1] as u64))
        })
        .fold(0u64, |n, e| {
            n.saturating_add(e.saturating_mul(width(factor)))
        })
}

// Shared checked tensor/residual metadata; the local work below adds file scope, reconstruction
// and guard budgets, NOT another tensor/adapter DTO. TensorFacts are never backing identities.
use mlx_gen::adapters::facts::{AdapterFacts, TensorFacts};

/// The selected storage operation, after resolving the actual target. This is deliberately not
/// the file's declared AdapterKind: a packed LoRA retains factors, whereas a dense LoRA merges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AdapterInstallStorage {
    DenseLinear,
    DenseConv,
    PackedLora,
    PackedDelta,
    StructuredLokr,
}

#[derive(Clone, Debug)]
pub(crate) struct AdapterInstallWork {
    /// Actual resolved module path (GEGLU already split to linear1/linear2). Empty only for direct
    /// private single-projection fixtures, never inferred from the adapter's declared kind.
    pub target: String,
    pub storage: AdapterInstallStorage,
    /// The shared source-derived reconstruction/merge budget passed to the actual guard.
    pub scratch: u64,
    /// Logical index/source payload already created before the target guard. Keeping it separate
    /// preserves the existing graph order instead of pretending the guard preceded its creation.
    pub pre_guard: u64,
    /// Source/operand/reconstruction declarations; native observations may be lazy views/graphs.
    /// Header counterparts carry geometry/dtype only, never fabricated coefficient values.
    pub parents: Vec<TensorFacts>,
    /// Actual producer: outputs handed to finish; header plan: source-derived output declarations.
    /// Finish is not a last-owner/backing-release proof.
    pub installed: Vec<TensorFacts>,
    /// Only the real forward-disable predicate. Alpha may fold into a payload without selecting
    /// this branch; in particular TP deltas carry scale=1 even when their baked values are zero.
    pub disabled: bool,
}

/// Operator payload upper bound for Adapter::residual followed by the inference host's
/// narrowing/add, over the checked shared residual facts. No placeholder Array, alpha payload
/// read, donation, unique backing or cancellation credit is inferred from shapes. Base-output
/// storage is counted by the enclosing Linear quantum, not subtracted here.
#[cfg(test)]
pub(crate) fn adapter_forward_scratch(
    residual: &AdapterFacts,
    input: &[i32],
    dtype: Dtype,
    output: Dtype,
) -> u64 {
    adapter_forward_scratch_at_output(residual, input, dtype, output, 0)
}

/// `add_elements` is the selected host output AFTER this residual's broadcast add, not the
/// residual's own extent. Narrowing still touches only the residual. A singleton residual can
/// require a full host-sized add result, even though neither factor changes the base geometry.
pub(crate) fn adapter_forward_scratch_at_output(
    residual: &AdapterFacts,
    input: &[i32],
    dtype: Dtype,
    output: Dtype,
    add_elements: u64,
) -> u64 {
    let Some((_, leading)) = input.split_last() else {
        return 0;
    };
    let rows = leading
        .iter()
        .fold(1u64, |n, d| n.saturating_mul((*d).max(0) as u64));
    let cast = |p: &TensorFacts, to: Dtype| {
        if p.dtype == to {
            0
        } else {
            p.shape
                .iter()
                .fold(width(to), |n, d| n.saturating_mul((*d).max(0) as u64))
        }
    };
    let activation = |to: Dtype| {
        if dtype == to {
            0
        } else {
            input
                .iter()
                .fold(width(to), |n, d| n.saturating_mul((*d).max(0) as u64))
        }
    };
    let (disabled, elements, result_dtype, mut bytes, scaled) = match residual {
        AdapterFacts::Lora { a, b, disabled } => {
            let first = Dtype::from_promoting_types(dtype, a.dtype);
            let second = Dtype::from_promoting_types(first, b.dtype);
            let inner = rows.saturating_mul(a.shape.last().copied().unwrap_or(0).max(0) as u64);
            let elements = rows.saturating_mul(b.shape.last().copied().unwrap_or(0).max(0) as u64);
            let mut bytes = activation(first)
                .saturating_add(cast(a, first))
                .saturating_add(inner.saturating_mul(width(first)))
                .saturating_add(cast(b, second));
            if first != second {
                bytes = bytes.saturating_add(inner.saturating_mul(width(second)));
            }
            (*disabled, elements, second, bytes, true)
        }
        AdapterFacts::Materialized { delta, disabled } => (
            *disabled,
            rows.saturating_mul(delta.shape.first().copied().unwrap_or(0).max(0) as u64),
            dtype,
            cast(delta, dtype),
            true,
        ),
        AdapterFacts::Structured { w1, w2, disabled } => {
            let a = w1.shape.first().copied().unwrap_or(0).max(0) as u64;
            let b = w2.shape.first().copied().unwrap_or(0).max(0) as u64;
            let d = w2.shape.get(1).copied().unwrap_or(0).max(0) as u64;
            (
                *disabled,
                rows.saturating_mul(a).saturating_mul(b),
                dtype,
                cast(w1, dtype)
                    .saturating_add(cast(w2, dtype))
                    .saturating_add(
                        rows.saturating_mul(a)
                            .saturating_mul(d)
                            .saturating_mul(width(dtype)),
                    ),
                false,
            )
        }
    };
    if disabled {
        return 0;
    }
    bytes = bytes.saturating_add(elements.saturating_mul(width(result_dtype)));
    if scaled {
        // Dtype-matched scalar and the multiply's declared output; do not assume donation.
        bytes = bytes
            .saturating_add(width(result_dtype))
            .saturating_add(elements.saturating_mul(width(result_dtype)));
    }
    if result_dtype != output {
        bytes = bytes.saturating_add(elements.saturating_mul(width(output)));
    }
    bytes.saturating_add(elements.max(add_elements).saturating_mul(width(output)))
}

/// Logical counterpart of the actual residual operators. This reads no payload and is separate
/// from dtype narrowing: AdaptableLinear narrows the residual, but its ordered add still broadcasts.
pub(crate) fn adapter_residual_shape(residual: &AdapterFacts, input: &[i32]) -> Result<Vec<i32>> {
    match residual {
        AdapterFacts::Lora { a, b, .. } => matmul_shape(&matmul_shape(input, &a.shape)?, &b.shape),
        AdapterFacts::Materialized { delta, .. } => {
            let mut transposed = delta.shape.clone();
            transposed.reverse();
            matmul_shape(input, &transposed)
        }
        AdapterFacts::Structured { w1, w2, .. } => {
            let bad_shape = || Error::Msg("sdxl structured residual axes disagree".into());
            if w1.shape.len() != 2 || w2.shape.len() != 2 { return Err(bad_shape()); }
            let input_width = w1.shape[1].checked_mul(w2.shape[1]).ok_or_else(bad_shape)?;
            if input.last() != Some(&input_width) { return Err(bad_shape()); }
            let mut shape = input[..input.len() - 1].to_vec();
            shape.push(w1.shape[0].checked_mul(w2.shape[0]).ok_or_else(bad_shape)?);
            Ok(shape)
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum AdapterWork {
    /// Every file key, including skipped/text-encoder keys and disabled adapter payloads.
    FileMaterialized(Vec<(String, TensorFacts)>),
    Selected(InstallKind),
    /// Parser-local cast/readback workspace. Its copied scalar result is not a retained Array.
    AlphaRead { key: String, scratch: u64 },
    /// Graph operands created before a target guard, retained as logical parents, not backing IDs.
    Preparation(Vec<TensorFacts>),
    /// The guard was reached; reconstruction/validation may still refuse. No installed result.
    GuardAttempt(AdapterInstallWork),
    Install(AdapterInstallWork),
    /// Only the local Weights map's scope ended. Captured clones, views, lazy graphs and block
    /// stream sidecars can keep its payload alive; this event grants NO physical release credit.
    FileScopeEnd,
}

#[cfg(test)]
thread_local! {
    static ADAPTER_WORK: std::cell::RefCell<Option<Vec<AdapterWork>>> = const {
        std::cell::RefCell::new(None)
    };
}

#[cfg(test)]
fn record_adapter_work(make: impl FnOnce() -> AdapterWork) {
    ADAPTER_WORK.with(|slot| {
        if let Some(work) = slot.borrow_mut().as_mut() {
            work.push(make());
        }
    });
}

/// Observe the real producer without retaining any tensor/capture clone, changing evaluation, or
/// replacing alpha/scale values. Restores an enclosing recorder on return, error and unwind.
/// This is a bounded producer seam, NOT a header-only substitute for running the adapter loader.
#[cfg(test)]
pub(crate) fn with_adapter_work<T>(body: impl FnOnce() -> T) -> (T, Vec<AdapterWork>) {
    struct Restore(Option<Vec<AdapterWork>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            ADAPTER_WORK.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(ADAPTER_WORK.with(|slot| slot.replace(Some(Vec::new()))));
    let value = body();
    let work = ADAPTER_WORK.with(|slot| slot.borrow_mut().take().unwrap_or_default());
    (value, work)
}

impl AdapterInstallWork {
    fn from_facts(storage: AdapterInstallStorage, scratch: u64, parents: Vec<TensorFacts>) -> Self {
        Self {
            target: String::new(),
            storage,
            scratch,
            pre_guard: 0,
            parents,
            installed: Vec::new(),
            disabled: false,
        }
    }

    fn at(mut self, target: &str) -> Self {
        self.target = target.to_owned();
        self
    }

    /// Shared by the actual guarded producer and its header-selected plan. `up` is the actual
    /// selected operand. Row indices already exist at this point in the current runtime order;
    /// their retained graph input is separate from the take output evaluated INSIDE the guard.
    fn lora(
        base: Option<TensorFacts>, down: TensorFacts, up: TensorFacts,
        unread_base: u64, selected_rows: i32, disabled: bool,
    ) -> Self {
        let copy = |a: &TensorFacts| if a.dtype == Dtype::Float32 { 0 } else {
            a.elements().saturating_mul(width(Dtype::Float32))
        };
        let row_take = if selected_rows == 0 { 0 } else { up.bytes() };
        let mut parents = vec![down.clone(), up.clone()];
        let mut buffers = Vec::new();
        if selected_rows > 0 { buffers.push(up.clone()); }
        let (storage, previous_budget) = match base {
            Some(base) => {
                let mut chain = lora_chain(up.dtype);
                chain.extend(linear_merge(base.dtype));
                let previous = crate::quantum::adapter_install_bytes(
                    base.elements(), &chain,
                    copy(&up).saturating_add(copy(&down)).saturating_add(row_take), unread_base,
                );
                for input in [&down, &up] {
                    if input.dtype != Dtype::Float32 {
                        buffers.push(output_fact(input.shape.clone(), Dtype::Float32));
                    }
                }
                // Size the actual delta, not the pre-merge base: broadcast can widen a dense base.
                // Malformed shapes still reach the ORIGINAL native validation after the guard.
                if let Ok(delta) = matmul_shape(&up.shape, &down.shape) {
                    buffers.push(output_fact(delta.clone(), Dtype::Float32));
                    if up.dtype != Dtype::Float32 {
                        buffers.push(output_fact(delta.clone(), up.dtype));
                        buffers.push(output_fact(delta.clone(), Dtype::Float32));
                    }
                    buffers.push(output_fact(Vec::new(), Dtype::Float32));
                    buffers.push(output_fact(delta.clone(), Dtype::Float32));
                    if base.dtype != Dtype::Float32 {
                        buffers.push(output_fact(delta.clone(), base.dtype));
                    }
                    if let Ok(merged) = broadcast_shape(&base.shape, &delta) {
                        buffers.push(output_fact(merged, base.dtype));
                    }
                }
                parents.insert(0, base);
                (AdapterInstallStorage::DenseLinear, previous)
            }
            None => {
                buffers.push(output_fact(Vec::new(), Dtype::Float32));
                if up.dtype != Dtype::Float32 { buffers.push(output_fact(Vec::new(), up.dtype)); }
                buffers.push(output_fact(up.shape.clone(), up.dtype));
                (AdapterInstallStorage::PackedLora,
                    crate::quantum::adapter_install_bytes(0, &[], up.bytes().saturating_add(row_take), 0))
            }
        };
        let actual_sequence = buffers.iter().fold(unread_base, |n, a| n.saturating_add(
            mlx_gen::ios_quantum::allocated_bytes(a.bytes())));
        // Last declared buffer is the installed result; earlier ones are potential lazy parents.
        parents.extend(buffers.iter().take(buffers.len().saturating_sub(1)).cloned());
        let mut work = Self::from_facts(storage, previous_budget.max(actual_sequence), parents);
        work.disabled = storage == AdapterInstallStorage::PackedLora && disabled;
        work.pre_guard = mlx_gen::ios_quantum::allocated_bytes(
            (selected_rows.max(0) as u64).saturating_mul(width(Dtype::Int32)));
        if selected_rows > 0 {
            work.parents.push(TensorFacts {
                shape: vec![selected_rows],
                dtype: Dtype::Int32, strides: vec![1],
            });
        }
        work
    }

    fn delta(
        base: Option<TensorFacts>, delta: TensorFacts, chain: &[u64], small: u64, unread_base: u64,
        pre_guard: u64,
    ) -> Self {
        let mut widths = chain.to_vec();
        let mut parents = vec![delta.clone()];
        let merged_upper = base.as_ref().and_then(|base|
            broadcast_shape(&base.shape, &delta.shape).ok().map(|shape| output_fact(shape, base.dtype).bytes()))
            .map_or(unread_base, |bytes| unread_base.saturating_add(mlx_gen::ios_quantum::allocated_bytes(bytes)));
        let storage = if let Some(base) = base {
            // Preserve the shortcut/view merge's existing non-donation budget.
            widths.push(width(base.dtype));
            parents.insert(0, base);
            AdapterInstallStorage::DenseLinear
        } else { AdapterInstallStorage::PackedDelta };
        let mut work = Self::from_facts(storage,
            crate::quantum::adapter_install_bytes(delta.elements(), &widths, small, unread_base)
                .max(merged_upper), parents);
        work.pre_guard = pre_guard;
        work
    }

    fn conv(base: TensorFacts, delta: TensorFacts, source: Dtype, small: u64, unread_base: u64) -> Self {
        let mut chain = lora_chain(source);
        chain.extend(linear_merge(base.dtype));
        chain.push(width(base.dtype));
        let legacy = crate::quantum::adapter_install_bytes(delta.elements(), &chain, small, unread_base);
        let mut merged_upper = unread_base;
        if delta.shape.len() == 4 {
            let ohwi = vec![delta.shape[0], delta.shape[2], delta.shape[3], delta.shape[1]];
            if let Ok(shape) = broadcast_shape(&base.shape, &ohwi) {
                // A small broadcast delta can produce a FULL base-sized sum. Its cast remains
                // delta-sized; the actual merged result must never be priced at delta.elements().
                merged_upper = merged_upper.saturating_add(mlx_gen::ios_quantum::allocated_bytes(
                    output_fact(shape, base.dtype).bytes()));
                if delta.dtype != base.dtype {
                    merged_upper = merged_upper.saturating_add(mlx_gen::ios_quantum::allocated_bytes(
                        output_fact(ohwi, base.dtype).bytes()));
                }
            }
        }
        let mut work = Self::from_facts(AdapterInstallStorage::DenseConv,
            legacy.max(merged_upper), vec![base, delta]);
        // conv_lora_delta builds its strength scalar before routing to this target guard.
        work.pre_guard = mlx_gen::ios_quantum::allocated_bytes(width(Dtype::Float32));
        work
    }

    fn dense_lokr(base: TensorFacts, factors: Vec<TensorFacts>, factor: Dtype, products: u64, unread_base: u64) -> Self {
        let mut chain = kron_chain(factor);
        chain.extend(linear_merge(base.dtype));
        let scratch = crate::quantum::adapter_install_bytes(base.elements(), &chain, products, unread_base);
        let mut parents = vec![base];
        parents.extend(factors);
        Self::from_facts(AdapterInstallStorage::DenseLinear, scratch, parents)
    }

    /// Actual structured build sequence: optional small products, Float32 scalar multiply on w2,
    /// then the two output casts. Full Float32 w2 STILL creates a factor-sized multiply output,
    /// including strength zero. No donation, view-release or backing credit is assumed.
    fn structured(factors: &BTreeMap<String, TensorFacts>, legacy_small: u64, disabled: bool) -> Self {
        let mut buffers = Vec::new();
        let mut select = |full: &str, a: &str, b: &str| -> Option<TensorFacts> {
            if let Some(factor) = factors.get(full) { return Some(factor.clone()); }
            let (a, b) = (factors.get(a)?, factors.get(b)?);
            if a.shape.len() != 2 || b.shape.len() != 2 { return None; }
            let product = TensorFacts::materialized_matrix(a.shape[0], b.shape[1],
                Dtype::from_promoting_types(a.dtype, b.dtype));
            buffers.push(product.clone());
            Some(product)
        };
        let one = select("lokr_w1", "lokr_w1_a", "lokr_w1_b");
        let two = select("lokr_w2", "lokr_w2_a", "lokr_w2_b");
        // Missing/malformed factors still reach the ORIGINAL native validation after the guard;
        // metadata accounting must not change its error priority or invent a fallback install.
        if let Some(two) = two {
            let scaled = TensorFacts {
                shape: two.shape.clone(),
                dtype: Dtype::from_promoting_types(two.dtype, Dtype::Float32), strides: Vec::new(),
            };
            buffers.push(scaled.clone());
            if scaled.dtype != Dtype::Float32 {
                buffers.push(TensorFacts { dtype: Dtype::Float32, ..scaled });
            }
        }
        if let Some(one) = one {
            if one.dtype != Dtype::Float32 {
                buffers.push(TensorFacts { dtype: Dtype::Float32, ..one });
            }
        }
        buffers.push(TensorFacts { shape: Vec::new(), dtype: Dtype::Float32, strides: Vec::new() });
        let actual_sequence = buffers.iter().fold(0u64, |n, a| n.saturating_add(
            mlx_gen::ios_quantum::allocated_bytes(a.bytes())));
        let checked_previous = crate::quantum::adapter_install_bytes(0, &[], legacy_small, 0);
        let mut work = Self::from_facts(AdapterInstallStorage::StructuredLokr,
            checked_previous.max(actual_sequence), factors.values().cloned().collect());
        work.disabled = disabled;
        work
    }

    pub(crate) fn residual(&self) -> Option<AdapterFacts> {
        match self.storage {
            AdapterInstallStorage::PackedLora => Some(AdapterFacts::Lora {
                a: self.installed.first()?.clone(),
                b: self.installed.get(1)?.clone(),
                disabled: self.disabled,
            }),
            AdapterInstallStorage::PackedDelta => Some(AdapterFacts::Materialized {
                delta: self.installed.first()?.clone(),
                disabled: self.disabled,
            }),
            AdapterInstallStorage::StructuredLokr => Some(AdapterFacts::Structured {
                w1: self.installed.first()?.clone(),
                w2: self.installed.get(1)?.clone(),
                disabled: self.disabled,
            }),
            AdapterInstallStorage::DenseLinear | AdapterInstallStorage::DenseConv => None,
        }
    }
}

/// One adapter target's actual install. The private work record and the numeric ledger share the
/// guard budget; all original graph producers and finish targets remain in their original order.
fn guarded_install(
    work: AdapterInstallWork,
    install: impl FnOnce() -> Result<Vec<Array>>,
) -> Result<()> {
    let priced = crate::ios_pricing::adapter_install_quantum(&work, 0);
    mlx_gen::ios_quantum::before(priced.label, priced.scratch)?;
    let installed = install()?;
    let refs: Vec<&Array> = installed.iter().collect();
    mlx_gen::ios_quantum::finish(crate::quantum::ADAPTER_MERGE, &refs)?;
    #[cfg(test)]
    {
        let mut observed = work;
        observed.installed = installed.iter().map(TensorFacts::of).collect();
        record_adapter_work(|| AdapterWork::Install(observed));
    }
    Ok(())
}

/// The dense base weight of `lin` (the merge target).
fn dense_base(lin: &AdaptableLinear) -> Result<Array> {
    lin.dense_weight()
        .map(|(w, _)| w.clone())
        .ok_or_else(|| Error::Msg("sdxl adapter: expected a dense base".into()))
}

pub(crate) fn install_delta_at(
    lin: &mut AdaptableLinear, delta: Array, chain: &[u64], small: u64, target: &str, pre_guard: u64,
) -> Result<()> {
    if lin.is_quantized() {
        return guarded_install(
            AdapterInstallWork::delta(None, TensorFacts::of(&delta), chain, small, 0, pre_guard).at(target),
            || {
                lin.push(Adapter::Lokr {
                    delta: delta.clone(),
                    scale: 1.0,
                });
                Ok(vec![delta])
            },
        );
    }
    let base = dense_base(lin)?;
    guarded_install(
        AdapterInstallWork::delta(Some(TensorFacts::of(&base)), TensorFacts::of(&delta),
            chain, small, unread(&base), pre_guard).at(target),
        || {
            lin.merge_dense_delta(&delta)?;
            Ok(vec![dense_base(lin)?])
        },
    )
}

fn install_lora_at(
    lin: &mut AdaptableLinear, down: &Array, up: &Array, s: LoraScale,
    selected_rows: i32, target: &str,
) -> Result<()> {
    if lin.is_quantized() {
        let work = AdapterInstallWork::lora(None, TensorFacts::of(down), TensorFacts::of(up),
            0, selected_rows, s.scale == 0.0).at(target);
        return guarded_install(work, || {
            // `A = downᵀ`, `B = upᵀ` with `alpha/rank` folded into B (Wan's `install_one_lora`);
            // user `scale` stays the residual scale. Fold into an owned `b` (a dtype-matched
            // scalar keeps the factor dtype, matching the fork's weak-float multiply).
            let a = down.t();
            let mut b = up.t();
            b = multiply(&b, &scalar(s.alpha / s.rank).as_dtype(up.dtype())?)?;
            lin.push(Adapter::Lora {
                a: a.clone(),
                b: b.clone(),
                scale: s.scale,
            });
            Ok(vec![a, b])
        });
    }
    let base = dense_base(lin)?;
    let work = AdapterInstallWork::lora(Some(TensorFacts::of(&base)), TensorFacts::of(down),
        TensorFacts::of(up), unread(&base), selected_rows, false).at(target);
    guarded_install(
        work,
        || {
            let delta = lora_delta(down, up, s.alpha, s.rank, s.scale)?;
            lin.merge_dense_delta(&delta)?;
            Ok(vec![dense_base(lin)?])
        },
    )
}

pub(crate) fn install_conv_delta_at(
    conv: &mut mlx_gen::adapters::AdaptableConv2d, delta_nchw: &Array,
    src: Dtype, small: u64, target: &str,
) -> Result<()> {
    let work = AdapterInstallWork::conv(TensorFacts::of(conv.weight()), TensorFacts::of(delta_nchw),
        src, small, unread(conv.weight())).at(target);
    guarded_install(work, || {
        conv.merge_conv_delta(delta_nchw)?;
        Ok(vec![conv.weight().clone()])
    })
}

/// Apply a reconstructed `[out,in]` `delta` at the Linear at `dotted`, counting it (or surfacing a
/// miss as skipped). The path is the internal module path (FF already translated to `ff.linearN`).
///
/// On a **dense** base the delta is MERGED (`W += δ`) — the bit-exact, sampler-chaos-safe path SDXL
/// has always used. On a **packed** (Q4/Q8) base the merge is impossible (`merge_dense_delta` rejects
/// a quantized weight), so the delta rides as a forward-time [`Adapter::Lokr`] residual `x·δᵀ`
/// (sc-10734). This generic path is used for `conv_shortcut` (a 1×1 conv stored — and quantized — as
/// a Linear) and as the safety net; the main attention/FF LoRA targets take the memory-cheap
/// factor-based [`apply_lora_into`] instead of materializing `[out,in]` deltas.
fn merge_into(
    unet: &mut UNet2DConditionModel,
    dotted: &str,
    delta: &Array,
    (chain, small): (&[u64], u64),
    report: &mut SdxlLoraReport,
) -> Result<()> {
    let parts: Vec<&str> = dotted.split('.').collect();
    match unet.adaptable_mut(&parts) {
        Some(lin) => {
            // A packed base carries the delta as an `Adapter::Lokr` residual `scale·x·δᵀ`; with the
            // `[out,in]` δ and scale 1.0 this is exactly the merged `x·(W+δ)ᵀ − x·Wᵀ`, on a base that
            // cannot be merged into. A dense base merges it (`install_delta_at`).
            install_delta_at(lin, delta.clone(), chain, small, dotted,
                mlx_gen::ios_quantum::allocated_bytes(width(Dtype::Float32)))?;
            report.merged += 1;
        }
        None => report.skipped_keys += 1,
    }
    Ok(())
}

/// Apply a LoRA `(down, up)` factor pair at the Linear `dotted`, choosing by the base's quantization
/// (sc-10734). On a **dense** base: reconstruct the delta and MERGE it (`W += δ`) — the bit-exact,
/// chaos-safe path (a residual's `W·x + δ·x` differs from the merged `(W+δ)·x` by ~1 ULP, which SDXL's
/// ancestral sampler amplifies visibly). On a **packed** (Q4/Q8) base the merge is impossible, so push
/// a forward-time [`Adapter::Lora`] residual `scale·(x·A)·B` with `A = downᵀ`, `B = upᵀ` and the
/// `alpha/rank` factor folded into `B` (mirroring Wan's `install_one_lora`) — memory-cheap: the small
/// factors are stored, never the `[out,in]` delta, so a LoRA on q4/q8 stays at plain-LoRA memory.
/// A LoRA target's scaling: the `alpha`/`rank` fold and the user `scale`. Bundled so the routing
/// helpers stay under clippy's argument bound and thread one value.
#[derive(Clone, Copy)]
pub(crate) struct LoraScale {
    pub(crate) alpha: f32,
    pub(crate) rank: f32,
    pub(crate) scale: f32,
}

fn apply_lora_into(
    unet: &mut UNet2DConditionModel,
    dotted: &str,
    down: &Array,
    up: &Array,
    selected_rows: i32,
    s: LoraScale,
    report: &mut SdxlLoraReport,
) -> Result<()> {
    let parts: Vec<&str> = dotted.split('.').collect();
    match unet.adaptable_mut(&parts) {
        Some(lin) => {
            install_lora_at(lin, down, up, s, selected_rows, dotted)?;
            report.merged += 1;
        }
        None => report.skipped_keys += 1,
    }
    Ok(())
}

/// Route a LoRA `(down, up)` for diffusers `path` into the U-Net, translating the GEGLU FF:
/// `ff.net.0.proj` (a fused `[2·hidden, D]` proj) row-splits into the value half `ff.linear1` (rows
/// `[0:hidden]`) and the gate half `ff.linear2` (`[hidden:2·hidden]`); `ff.net.2` maps to `ff.linear3`.
/// Every other path (attention / proj / time_emb / mid_block) routes 1:1.
///
/// The FF split acts on the **UP** factor's output rows (`rows(up)`), not the reconstructed delta —
/// `rows(up@down) == rows(up)@down` to the last bit (see [`rows`]), so the dense merge stays
/// byte-identical to the old delta-row-split while the packed path never forms the delta.
fn apply_lora_routed(
    unet: &mut UNet2DConditionModel,
    path: &str,
    down: &Array,
    up: &Array,
    s: LoraScale,
    report: &mut SdxlLoraReport,
) -> Result<()> {
    for (target, selected_rows) in lora_output_rows(path, up.shape()[0]) {
        let selected = selected_rows.map(|(lo, hi)| rows(up, lo, hi)).transpose()?;
        apply_lora_into(
            unet,
            &target,
            down,
            selected.as_ref().unwrap_or(up),
            selected_rows.map_or(0, |(lo, hi)| hi - lo),
            s,
            report,
        )?;
    }
    Ok(())
}

/// Pure runtime/header route producer. Preserve the existing floor split for odd fused row
/// counts; validation/shape compatibility remains with the actual target install, not a guess here.
pub(crate) fn lora_output_rows(path: &str, output_rows: i32) -> Vec<(String, Option<(i32, i32)>)> {
    if let Some(prefix) = path.strip_suffix(".ff.net.0.proj") {
        let half = output_rows / 2;
        vec![
            (format!("{prefix}.ff.linear1"), Some((0, half))),
            (format!("{prefix}.ff.linear2"), Some((half, output_rows))),
        ]
    } else if let Some(prefix) = path.strip_suffix(".ff.net.2") {
        vec![(format!("{prefix}.ff.linear3"), None)]
    } else {
        vec![(path.into(), None)]
    }
}

/// Route a fused conv-layer LoRA `delta` (trained-file NCHW `[out, in, kH, kW]`) into the U-Net
/// (sc-2919). Most conv stems (`conv_in`/`conv_out`, resnet `conv1`/`conv2`, the down/up-sampler
/// `conv`) resolve to an [`mlx_gen::adapters::AdaptableConv2d`] and merge in NHWC. The lone exception
/// is `conv_shortcut`: a 1×1 conv stored as a Linear (`[out,in]`), so its `[out,in,1,1]` delta is
/// reshaped to `[out,in]` and merged via the dense-Linear path. A path that resolves to neither is
/// surfaced as skipped.
fn merge_conv_routed(
    unet: &mut UNet2DConditionModel,
    path: &str,
    delta_nchw: &Array,
    (src, small): (Dtype, u64),
    report: &mut SdxlLoraReport,
) -> Result<()> {
    let parts: Vec<&str> = path.split('.').collect();
    if let Some(conv) = unet.adaptable_conv_mut(&parts) {
        install_conv_delta_at(conv, delta_nchw, src, small, path)?;
        report.merged += 1;
        return Ok(());
    }
    // conv_shortcut (1×1) is a Linear: fold the [out,in,1,1] delta to [out,in]. It is a quantizable
    // Linear, so route through `merge_into` — which merges on a dense base but rides the delta as a
    // forward-time residual on a packed q4/q8 base (sc-10734), rather than a raw merge_dense_delta
    // that would reject the quantized weight. `merge_into` counts the merge or the skip itself.
    let sh = delta_nchw.shape();
    if sh[2] == 1 && sh[3] == 1 {
        return merge_into(
            unet,
            &parts.join("."),
            &delta_nchw.reshape(&[sh[0], sh[1]])?,
            (&lora_chain(src), small),
            report,
        );
    }
    report.skipped_keys += 1;
    Ok(())
}

/// Map one safetensors key to `(diffusers_dotted_path, role)`, or `None` if it targets a module
/// outside the routable surface (mirrors the vendored `_classify_key` returning `(None, None)`).
fn classify_key(key: &str, kohya_to_dotted: &BTreeMap<String, String>) -> Option<(String, Role)> {
    if let Some(rem) = key.strip_prefix(PEFT_PREFIX) {
        // PEFT: the dotted diffusers path resolves directly. Accept the peft `.default.weight`
        // infix and the bare `.weight` form.
        for (suf, role) in [
            (".lora_A.default.weight", Role::Down),
            (".lora_B.default.weight", Role::Up),
            (".lora_A.weight", Role::Down),
            (".lora_B.weight", Role::Up),
            (".alpha", Role::Alpha),
        ] {
            if let Some(path) = rem.strip_suffix(suf) {
                return Some((path.to_string(), role));
            }
        }
        return None;
    }
    if let Some(rem) = key.strip_prefix(KOHYA_PREFIX) {
        // kohya: resolve the flattened stem against the routable-path table — directly for a
        // diffusers-named stem, or via an original-SD/A1111 → diffusers translation (sc-6051).
        for (suf, role) in [
            (".lora_down.weight", Role::Down),
            (".lora_up.weight", Role::Up),
            (".alpha", Role::Alpha),
        ] {
            if let Some(stem) = rem.strip_suffix(suf) {
                return resolve_kohya_stem(stem, kohya_to_dotted).map(|d| (d, role));
            }
        }
        return None;
    }
    // `lora_te1_`/`lora_te2_`/… text-encoder keys land here — deliberately skipped (UNet-only).
    None
}

/// `δ = (B @ A) · (alpha/rank) · scale`, reproducing the vendored `lora.py` merge bit-for-bit.
///
/// The vendored computes `(b@a)` in the LoRA tensors' on-disk dtype (f16 for community/LCM LoRAs),
/// then `.astype(weight.dtype)` (f32) and `* effective_scale`. On the pmetal NAX build a 16-bit
/// `b@a` (K=rank≤512) would hit the dense GEMM bug, so we run the matmul in **f32** (correct) and
/// round the result back through the source dtype — MLX's f16 matmul equals `round_f16` of the
/// f32-accumulated product, so this is bit-identical to the reference without touching the bug.
pub fn lora_delta(down: &Array, up: &Array, alpha: f32, rank: f32, scale: f32) -> Result<Array> {
    if rank == 0.0 {
        // A zero rank (an empty/malformed down|up factor) makes `alpha/rank` non-finite, and the
        // resulting NaN delta would be merged into the U-Net weights, silently poisoning all
        // inference. Reject the adapter instead (sc-5252/F-002).
        return Err(Error::Msg(
            "lora_delta: zero rank — malformed adapter (empty down/up factor)".into(),
        ));
    }
    let src = up.dtype(); // f16 for kohya/community LoRAs; f32 makes the round-trip a no-op.
    let ba = matmul(
        &up.as_dtype(Dtype::Float32)?,
        &down.as_dtype(Dtype::Float32)?,
    )?;
    let ba = ba.as_dtype(src)?.as_dtype(Dtype::Float32)?;
    // effective_scale in f64 then f32, matching the reference's Python-float arithmetic.
    let eff = ((alpha as f64 / rank as f64) * scale as f64) as f32;
    Ok(multiply(&ba, scalar(eff))?)
}

fn read_scalar(a: &Array) -> Result<f32> {
    Ok(a.as_dtype(Dtype::Float32)?.reshape(&[1])?.as_slice::<f32>()[0])
}

/// Merge one LoRA file into `unet` at `scale`, classifying every key (both formats) and folding the
/// complete `(down, up)` pairs into their target weights. Half-pairs and out-of-surface / conv-shaped
/// keys are counted as skipped.
fn merge_one(
    unet: &mut UNet2DConditionModel,
    w: &Weights,
    scale: f32,
    kohya_to_dotted: &BTreeMap<String, String>,
    coverage: LoraCoverage,
    report: &mut SdxlLoraReport,
) -> Result<()> {
    merge_one_in_key_order(unet, w, scale, kohya_to_dotted, coverage, report,
        w.keys().map(str::to_string).collect::<Vec<_>>())
}

// The production wrapper preserves its original unsorted HashMap traversal and Vec allocation.
// Only the CPU fixture supplies another complete permutation, through the same parser/install body.
fn merge_one_in_key_order(
    unet: &mut UNet2DConditionModel,
    w: &Weights,
    scale: f32,
    kohya_to_dotted: &BTreeMap<String, String>,
    coverage: LoraCoverage,
    report: &mut SdxlLoraReport,
    keys: impl IntoIterator<Item = String>,
) -> Result<()> {
    let mut triples: BTreeMap<String, LoraTriple> = BTreeMap::new();
    for key in keys {
        match classify_key(&key, kohya_to_dotted) {
            Some((path, Role::Down)) => {
                triples.entry(path).or_default().down = Some(w.require(&key)?.clone())
            }
            Some((path, Role::Up)) => {
                triples.entry(path).or_default().up = Some(w.require(&key)?.clone())
            }
            Some((path, Role::Alpha)) => {
                triples.entry(path).or_default().alpha = Some(read_scalar(w.require(&key)?)?)
            }
            None => report.skipped_keys += 1,
        }
    }

    // PEFT/diffusers `save_lora_adapter` files carry no per-target `.alpha` tensor — `lora_alpha`/`r`
    // (+ per-module overrides) live in the `lora_adapter_metadata` header blob (sc-5513). `None` for
    // kohya files (those ship a `.alpha` tensor), in which case the per-target `.alpha` or the factor
    // rank is used exactly as before.
    let cfg = LoraAdapterMeta::from_metadata(w.metadata(LORA_ADAPTER_METADATA_KEY));
    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            // Half-pair (a down/up whose partner targeted a non-routable module) — skip.
            report.skipped_keys += 1;
            continue;
        };
        // Effective scaling: per-target `.alpha` tensor → `alpha_pattern`/`lora_alpha` blob → factor
        // rank (today's last-resort default). The denominator is the blob `r`/`rank_pattern` when given
        // (always `> 0`), else the stored `down` leading dim (which equals it for a well-formed file).
        let (cfg_alpha, cfg_rank) = cfg.as_ref().map_or((None, None), |c| c.effective(&path));
        // Conv-shaped (4-D) LoRAs (`down [rank,in,kH,kW]`, `up [out,rank,1,1]`): merge under
        // Complete coverage (sc-2919), where they fold into the conv weights (resnet conv1/conv2,
        // conv_shortcut, the down/up-samplers, conv_in/conv_out). Under the Linear-only vendored
        // coverage they stay dropped — preserving byte-parity with the retired Python `lora.py` path
        // (the vendored merge is Linear-only). This single gate covers both on-disk formats: a kohya
        // conv key only resolves to a dotted path when the complete table carries the conv stems, and
        // a PEFT conv key (dotted) is gated here.
        if down.ndim() == 4 || up.ndim() == 4 {
            if coverage != LoraCoverage::Complete {
                report.skipped_keys += 2;
                continue;
            }
            let rank = cfg_rank.unwrap_or(down.shape()[0] as f32);
            let alpha = t.alpha.or(cfg_alpha).unwrap_or(rank);
            let delta = conv_lora_delta(&down, &up, alpha, rank, scale)?;
            let small = f32_copy(&up).saturating_add(f32_copy(&down));
            merge_conv_routed(unet, &path, &delta, (up.dtype(), small), report)?;
            continue;
        }
        if down.ndim() != 2 || up.ndim() != 2 {
            // Neither 2-D (Linear) nor 4-D (conv) — an unexpected shape; surface it.
            report.skipped_keys += 2;
            continue;
        }
        // Under vendored coverage, keep mid_block/ff out. kohya keys for those never reach here (the
        // table excludes them), but a PEFT key carries its dotted path directly — gate it here so the
        // faithful 515-module merge is byte-identical regardless of on-disk format.
        if coverage == LoraCoverage::Vendored && is_complete_only(&path) {
            report.skipped_keys += 1;
            continue;
        }
        let rank = cfg_rank.unwrap_or(down.shape()[0] as f32);
        let alpha = t.alpha.or(cfg_alpha).unwrap_or(rank);
        // Routes 1:1 for attention/proj/time_emb/mid_block; row-splits the GEGLU `ff.net.0.proj`.
        // A PEFT path naming a genuinely non-routable module surfaces as skipped. Threads the factors
        // (not a pre-formed delta) so a packed base takes the memory-cheap forward-time residual
        // (sc-10734); a dense base merges exactly as before.
        apply_lora_routed(
            unet,
            &path,
            &down,
            &up,
            LoraScale { alpha, rank, scale },
            report,
        )?;
    }
    Ok(())
}

/// Map a LoKr key to `(diffusers_dotted_path, factor_name)`, or `None` if out of surface. kohya
/// `lora_unet_<flat>.lokr_*` resolves the flattened stem via the table; bare/PEFT `<dotted>.lokr_*`
/// (with an optional `base_model.model.unet.` prefix) resolves the dotted path directly.
fn classify_lokr_key(
    key: &str,
    kohya_to_dotted: &BTreeMap<String, String>,
) -> Option<(String, &'static str)> {
    for suf in LOKR_SUFFIXES {
        if let Some(stem) = key.strip_suffix(suf) {
            let factor = &suf[1..]; // drop the leading '.'
            return if let Some(flat) = stem.strip_prefix(KOHYA_PREFIX) {
                resolve_kohya_stem(flat, kohya_to_dotted).map(|d| (d, factor))
            } else {
                Some((
                    stem.strip_prefix(PEFT_PREFIX).unwrap_or(stem).to_string(),
                    factor,
                ))
            };
        }
    }
    None
}

/// The actionable error for a LoKr variant that is genuinely NOT deferrable via the vec-trick on a
/// packed base — only the conv-only tucker/CP `w2` form, which the all-Linear SDXL LoKr surface never
/// produces. Point the user at the bf16 tier (where it merges into the dense weight) rather than a
/// silent OOM/wrong result. sc-10734 / epic 10043 (mirrors the Wan message).
fn non_deferrable_lokr_error(path: &str) -> mlx_gen::Error {
    format!(
        "LoKr at '{path}' uses a tucker/CP (conv) factorization that cannot be applied on a quantized \
         (packed Q4/Q8) SDXL tier without materializing the full delta — use the bf16 tier (where it \
         merges into the dense weight). This form does not occur for the SDXL U-Net's Linear LoKr \
         targets (sc-10734)."
    )
    .into()
}

/// Merge one LoKr file into `unet` at `scale` (sc-2640). The vendored SDXL path *rejects* LoKr, so
/// there is no fork to match — we reconstruct the delta with the validated LyCORIS formula
/// (`reconstruct_lokr_delta`, f32 for the f32-everywhere SDXL path) and **merge** it (`W += δ·scale`),
/// chaos-safe like the LoRA path. `alpha`/`rank` come from the file metadata (alpha defaults to rank).
fn merge_one_lokr(
    unet: &mut UNet2DConditionModel,
    w: &Weights,
    scale: f32,
    kohya_to_dotted: &BTreeMap<String, String>,
    report: &mut SdxlLoraReport,
) -> Result<()> {
    let rank = w
        .metadata("rank")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);
    let alpha = w
        .metadata("alpha")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(rank);

    let mut grouped: BTreeMap<String, BTreeMap<&'static str, Array>> = BTreeMap::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        match classify_lokr_key(&key, kohya_to_dotted) {
            Some((path, factor)) => {
                grouped
                    .entry(path)
                    .or_default()
                    .insert(factor, w.require(&key)?.clone());
            }
            None => report.skipped_keys += 1,
        }
    }

    for (path, f) in grouped {
        // LoKr stays at the vendored-equivalent surface regardless of coverage (sc-2671 extends only
        // LoRA). mid_block is now routable, so gate it here to keep sc-2640's behaviour unchanged; FF
        // LoKr keys resolve to no internal Linear anyway (`ff.net.*` has no `adaptable_mut` arm).
        if is_complete_only(&path) {
            report.skipped_keys += 1;
            continue;
        }
        let parts: Vec<&str> = path.split('.').collect();
        // The factors' promoted dtype and the small Kronecker factors they rebuild (sc-26270).
        let factor = f
            .values()
            .map(Array::dtype)
            .reduce(Dtype::from_promoting_types)
            .unwrap_or(Dtype::Float32);
        let products = product_bytes(
            &[
                (f.get("lokr_w1_a"), f.get("lokr_w1_b")),
                (f.get("lokr_w2_a"), f.get("lokr_w2_b")),
            ],
            factor,
        );
        let factor_bytes = f.values().fold(0u64, |n, a| n.saturating_add(f32_copy(a)));
        match unet.adaptable_mut(&parts) {
            Some(lin) => {
                let base_shape = lin.base_shape();
                if lin.is_quantized() {
                    // Packed base (sc-10734 / epic 10043): the deferred, structured Kronecker residual
                    // — `vec(w1·reshape(x)·w2ᵀ)` — never materializes the `[out,in]` delta, so a LoKr
                    // works on q4/q8 at plain-LoRA memory. The FULL `(alpha/rank)·scale` bakes into the
                    // structured `w2` (a structured residual carries no separate scale). Priced: the
                    // rebuilt factors, the scaled `w2` and the f32 casts (all factor-sized).
                    let factor_work = f.iter().map(|(key, array)|
                        ((*key).to_owned(), TensorFacts::of(array))).collect();
                    let work = AdapterInstallWork::structured(&factor_work,
                        products.saturating_mul(2).saturating_add(factor_bytes),
                        (alpha / rank) * scale == 0.0).at(&path);
                    guarded_install(work, || {
                        match build_lokr_factors(
                            (alpha / rank) * scale,
                            &base_shape,
                            f.get("lokr_w1"),
                            f.get("lokr_w1_a"),
                            f.get("lokr_w1_b"),
                            f.get("lokr_w2"),
                            None, // SDXL LoKr targets are all Linear — never a tucker/CP w2
                            f.get("lokr_w2_a"),
                            f.get("lokr_w2_b"),
                            Dtype::Float32,
                        )? {
                            Some(factors) => {
                                let installed = vec![factors.w1.clone(), factors.w2.clone()];
                                lin.push(Adapter::LokrStructured { factors });
                                Ok(installed)
                            }
                            None => Err(non_deferrable_lokr_error(&path)),
                        }
                    })?;
                } else {
                    let base = dense_base(lin)?;
                    let work = AdapterInstallWork::dense_lokr(TensorFacts::of(&base),
                        f.values().map(TensorFacts::of).collect(), factor, products, unread(&base)).at(&path);
                    guarded_install(work, || {
                        let delta = reconstruct_lokr_delta(
                            alpha,
                            rank,
                            &base_shape,
                            f.get("lokr_w1"),
                            f.get("lokr_w1_a"),
                            f.get("lokr_w1_b"),
                            f.get("lokr_w2"),
                            f.get("lokr_w2_a"),
                            f.get("lokr_w2_b"),
                            Dtype::Float32,
                        )?;
                        // The alpha/rank factor is baked into `delta`; apply the user scale on top.
                        let delta = multiply(&delta, scalar(scale))?;
                        lin.merge_dense_delta(&delta)?;
                        Ok(vec![dense_base(lin)?])
                    })?;
                }
                report.merged += 1;
            }
            None => report.skipped_keys += 1,
        }
    }
    Ok(())
}

/// Merge one third-party-reconstructed `[out,in]` delta into `unet` at the dotted module `path`
/// (sc-3671). Shared by the third-party LoKr + LoHa merge paths: resolve → gate complete-only/conv →
/// `W += δ·scale`. An unresolved or non-Linear target is surfaced as skipped, never fatal.
fn merge_thirdparty_delta(
    unet: &mut UNet2DConditionModel,
    path: Option<&str>,
    delta_at: impl FnOnce(&[i32]) -> Result<Array>,
    (chain, small): (&[u64], u64),
    scale: f32,
    report: &mut SdxlLoraReport,
) -> Result<()> {
    let Some(path) = path else {
        report.skipped_keys += 1;
        return Ok(());
    };
    // LoKr/LoHa stay at the vendored-equivalent Linear surface (like `merge_one_lokr`): mid_block / FF
    // and conv targets resolve to no internal Linear, so gate them as skipped rather than mis-merge.
    if is_complete_only(path) {
        report.skipped_keys += 1;
        return Ok(());
    }
    let parts: Vec<&str> = path.split('.').collect();
    match unet.adaptable_mut(&parts) {
        Some(lin) => {
            let base_shape = lin.base_shape();
            // Built lazily (no buffer exists until the guarded install materializes it).
            let delta = delta_at(&base_shape)?;
            let delta = multiply(&delta, scalar(scale))?;
            // Packed base (sc-10734): third-party LoKr/LoHa have no single structured form here
            // (a LoHa in particular is not Kronecker-deferrable), so the reconstructed delta rides
            // as a forward-time `Adapter::Lokr` residual (`x·δᵀ`, scale already folded in) and
            // stays resident; a dense base merges it (`install_delta_at`).
            install_delta_at(lin, delta, chain, small, path,
                mlx_gen::ios_quantum::allocated_bytes(width(Dtype::Float32)).saturating_mul(2))?;
            report.merged += 1;
        }
        None => report.skipped_keys += 1,
    }
    Ok(())
}

/// Merge a third-party LyCORIS **LoKr** file (kohya/lycoris keys, per-module `.alpha`, no
/// `networkType` stamp) into `unet` at `scale` (sc-3671). Reuses the core per-module reconstruction
/// (`ThirdPartyLokr::delta`, f32 for the f32-everywhere SDXL path) and the same `W += δ·scale` merge
/// as the peft `merge_one_lokr`. Linear-only (matches the peft LoKr coverage).
fn merge_one_lokr_thirdparty(
    unet: &mut UNet2DConditionModel,
    w: &Weights,
    scale: f32,
    table: &BTreeMap<String, String>,
    report: &mut SdxlLoraReport,
) -> Result<()> {
    for (raw, g) in &parse_lokr_thirdparty(w)? {
        let (factor, small) = factor_stats(w, raw, &LOKR_TP_SUFFIXES);
        let products = product_bytes(
            &[
                (
                    w.get(&format!("{raw}.lokr_w1_a")),
                    w.get(&format!("{raw}.lokr_w1_b")),
                ),
                (
                    w.get(&format!("{raw}.lokr_w2_a")),
                    w.get(&format!("{raw}.lokr_w2_b")),
                ),
            ],
            factor,
        );
        merge_thirdparty_delta(
            unet,
            resolve_lokr_path(raw, table),
            |bs| g.delta(bs, Dtype::Float32),
            (&kron_chain(factor), small.saturating_add(products)),
            scale,
            report,
        )?;
    }
    Ok(())
}

/// Merge a third-party LyCORIS **LoHa** file into `unet` at `scale` (sc-3671). As
/// [`merge_one_lokr_thirdparty`] but the per-module delta is the Hadamard reconstruction
/// (`ThirdPartyLoha::delta`).
fn merge_one_loha_thirdparty(
    unet: &mut UNet2DConditionModel,
    w: &Weights,
    scale: f32,
    table: &BTreeMap<String, String>,
    report: &mut SdxlLoraReport,
) -> Result<()> {
    for (raw, g) in &parse_loha_thirdparty(w)? {
        let (factor, small) = factor_stats(w, raw, &LOHA_TP_SUFFIXES);
        merge_thirdparty_delta(
            unet,
            resolve_lokr_path(raw, table),
            |bs| g.delta(bs, Dtype::Float32),
            (&hadamard_chain(factor), small),
            scale,
            report,
        )?;
    }
    Ok(())
}

/// Merge every adapter spec in `specs` into `unet` — LoRA (sc-2639) and LoKr (sc-2640) — at the
/// **vendored-faithful** coverage (515 modules on LCM-LoRA; byte-parity with the retired Python
/// path). The vendored SDXL path supports LoRA only (it *rejects* LoKr); Rust is strictly more
/// capable. NOTE: `model::load` now defaults to [`LoraCoverage::Complete`] (sc-2671) — this faithful
/// entry point is reached only via the `SDXL_LORA_VENDORED` escape hatch. See
/// [`apply_sdxl_adapters_with`]. Errors if a non-empty spec list merges nothing (a real
/// format/prefix misconfiguration — e.g. an original-SD `lora_unet_input_blocks_*` file).
pub fn apply_sdxl_adapters(
    unet: &mut UNet2DConditionModel,
    specs: &[AdapterSpec],
) -> Result<SdxlLoraReport> {
    apply_sdxl_adapters_with(unet, specs, LoraCoverage::Vendored)
}

/// As [`apply_sdxl_adapters`], but with an explicit [`LoraCoverage`]. [`LoraCoverage::Complete`]
/// (sc-2671) reaches `mid_block` + the GEGLU FF for LoRA — strictly more correct than the vendored
/// path, at the cost of byte-parity with it. The SDXL [`crate::model::load`] uses `Complete` by
/// default and falls back to `Vendored` only when `SDXL_LORA_VENDORED` is set.
pub fn apply_sdxl_adapters_with(
    unet: &mut UNet2DConditionModel,
    specs: &[AdapterSpec],
    coverage: LoraCoverage,
) -> Result<SdxlLoraReport> {
    #[cfg(test)]
    { apply_sdxl_adapters_with_impl(unet, specs, coverage, None) }
    #[cfg(not(test))]
    { apply_sdxl_adapters_with_impl(unet, specs, coverage) }
}

/// Supply complete file-key permutations only to the existing CPU-test producer. Validation is
/// deliberately outside the merge body; omitted alpha/unresolved/skipped keys cannot disappear.
#[cfg(test)]
pub(crate) fn apply_sdxl_adapters_in_key_order_for_test(
    unet: &mut UNet2DConditionModel,
    specs: &[AdapterSpec],
    coverage: LoraCoverage,
    orders: &[Vec<String>],
) -> Result<SdxlLoraReport> {
    assert_eq!(orders.len(), specs.len(), "one complete order per configured adapter file");
    apply_sdxl_adapters_with_impl(unet, specs, coverage, Some(orders))
}

fn apply_sdxl_adapters_with_impl(
    unet: &mut UNet2DConditionModel,
    specs: &[AdapterSpec],
    coverage: LoraCoverage,
    #[cfg(test)] orders: Option<&[Vec<String>]>,
) -> Result<SdxlLoraReport> {
    if specs.is_empty() {
        return Ok(SdxlLoraReport::default());
    }
    // LoKr is always merged against the vendored-equivalent surface; LoRA uses the coverage table.
    // The complete LoRA table also carries the conv-layer stems (sc-2919) so kohya conv keys
    // (`lora_unet_..._conv1`, `..._downsamplers_0_conv`, `conv_in`, …) resolve to a dotted path and
    // reach the conv merge; the vendored table omits them (Linear-only byte-parity).
    let vendored_table = build_table(unet.lora_target_paths());
    let complete_table = (coverage == LoraCoverage::Complete).then(|| {
        let mut paths = unet.lora_target_paths_complete();
        paths.extend(unet.conv_target_paths());
        build_table(paths)
    });
    let lora_table = complete_table.as_ref().unwrap_or(&vendored_table);

    let mut report = SdxlLoraReport::default();
    #[cfg(test)]
    let mut file_index = 0usize;
    for spec in specs {
        // sc-26270: the adapter file's header floor is checked before it is read (no-op off device).
        if mlx_gen::ios_quantum::enabled() {
            let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&spec.path)
                .map_err(|e| Error::Msg(e.to_string()))?;
            mlx_gen::ios_quantum::before(crate::quantum::ADAPTER_LOAD, adapter_file_load_bytes(&headers)?)?;
        }
        let w = Weights::from_file(&spec.path)?;
        // Adapter tensors are lazy too. Materialize their source payload before the prepared-file
        // guard around the provider phase performs its post-read identity check.
        w.materialize()?;
        #[cfg(test)]
        let order = {
            let order = orders.map(|orders| &orders[file_index]);
            file_index += 1;
            if let Some(order) = order {
                let loaded: std::collections::BTreeSet<_> = w.keys().collect();
                let supplied: std::collections::BTreeSet<_> = order.iter().map(String::as_str).collect();
                assert_eq!(order.len(), loaded.len(), "complete file order must contain every loaded key exactly once");
                assert_eq!(supplied.len(), order.len(), "duplicate file-order key");
                assert_eq!(supplied, loaded, "file order omitted/replaced a loaded alpha, unresolved or skipped key");
            }
            order
        };
        #[cfg(test)]
        record_adapter_work(|| {
            AdapterWork::FileMaterialized(
                w.keys()
                    .map(|key| {
                        (
                            key.to_owned(),
                            TensorFacts::of(w.get(key).expect("existing adapter key")),
                        )
                    })
                    .collect(),
            )
        });
        // Third-party LyCORIS (sc-3671): `lokr_*` / `hada_*` keys without a `networkType=lokr` stamp,
        // so the caller can't label the kind — detect + route by keys before the kind match. (A peft
        // LoKr has the stamp and goes through the `Lokr` arm; `is_lokr_keys` excludes it via `is_lokr`.)
        let selected = install_kind(spec.kind, w.metadata("networkType"), w.keys())?;
        #[cfg(test)]
        record_adapter_work(|| AdapterWork::Selected(selected));
        match selected {
            InstallKind::ThirdPartyLokr => {
                merge_one_lokr_thirdparty(unet, &w, spec.scale, &vendored_table, &mut report)?;
            }
            InstallKind::ThirdPartyLoha => {
                merge_one_loha_thirdparty(unet, &w, spec.scale, &vendored_table, &mut report)?;
            }
            InstallKind::Lokr => {
                merge_one_lokr(unet, &w, spec.scale, &vendored_table, &mut report)?
            }
            InstallKind::Lora => {
                #[cfg(test)]
                if let Some(order) = order {
                    merge_one_in_key_order(unet, &w, spec.scale, lora_table, coverage, &mut report,
                        order.iter().cloned())?;
                } else {
                    merge_one(unet, &w, spec.scale, lora_table, coverage, &mut report)?;
                }
                #[cfg(not(test))]
                merge_one(unet, &w, spec.scale, lora_table, coverage, &mut report)?;
            }
        }
        #[cfg(test)]
        record_adapter_work(|| AdapterWork::FileScopeEnd);
    }

    if report.merged == 0 {
        return Err(Error::Msg(format!(
            "sdxl: no adapter target modules matched across {} file(s) — check the format \
             (expected kohya `lora_unet_` with diffusers `down_blocks_*` or original-SD \
             `input_blocks_*` block naming, PEFT `base_model.model.unet.`, or LoKr \
             `<module>.lokr_w1/w2` + networkType=lokr; conv-layer LoRAs merge only under Complete \
             coverage — i.e. not when `SDXL_LORA_VENDORED` is set)",
            specs.len()
        )));
    }
    Ok(report)
}

fn alpha_cast_bytes(name: &str, elements: u64, is_f32: bool) -> u64 {
    if is_f32 || !name.ends_with(".alpha") { return 0; }
    // scalar_alpha casts the FULL array before taking first(); do not silently narrow accepted
    // third-party alpha shapes or pretend this existing payload read is an <=8-byte control read.
    mlx_gen::ios_quantum::allocated_bytes(elements.saturating_mul(width(Dtype::Float32)))
}

pub(crate) fn adapter_file_load_bytes(
    headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
) -> Result<u64> {
    let mut raw = 0u64;
    let mut scalar = 0u64;
    for header in headers {
        raw = raw.checked_add(header.data_bytes)
            .ok_or_else(|| Error::Unsupported("sdxl adapter file accounting overflow".into()))?;
        scalar = scalar.max(alpha_cast_bytes(&header.name,
            header.element_count().map_err(|e| Error::Msg(e.to_string()))?,
            header.dtype == mlx_gen::gen_core::weightsmeta::Dtype::F32));
    }
    raw.checked_add(scalar)
        .ok_or_else(|| Error::Unsupported("sdxl adapter alpha accounting overflow".into()))
}

pub(crate) fn adapter_file_payload_budget(payloads: &[(String, TensorFacts)]) -> (u64, u64) {
    let raw = payloads.iter().fold(0u64, |n, (_, a)| n.saturating_add(a.bytes()));
    let scalar = payloads.iter().map(|(name, a)| alpha_cast_bytes(name, a.elements(),
        a.dtype == Dtype::Float32)).max().unwrap_or(0);
    (raw, raw.saturating_add(scalar))
}

fn header_tensor_fact(h: &mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader) -> Result<TensorFacts> {
    let shape = header_shape(h)?;
    let mut strides = vec![0; shape.len()];
    let mut stride = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride = stride.checked_mul(shape[i].max(0) as usize)
            .ok_or_else(|| Error::Msg("sdxl adapter native stride overflow".into()))?;
    }
    Ok(TensorFacts { shape, dtype: mlx_gen::adapters::facts::native_header_dtype(h.dtype)?, strides })
}

fn output_fact(shape: Vec<i32>, dtype: Dtype) -> TensorFacts {
    // A logical output declaration, not a fabricated final stride/backing identity.
    TensorFacts { shape, dtype, strides: Vec::new() }
}

fn header_factor_work(factors: &HeaderFactors<'_>) -> Result<BTreeMap<String, TensorFacts>> {
    factors.iter().filter(|(role, _)| role.as_str() != "alpha")
        .map(|(role, header)| Ok((role.clone(), header_tensor_fact(header)?))).collect()
}

fn factor_work_products(factors: &BTreeMap<String, TensorFacts>, dtype: Dtype) -> u64 {
    [("lokr_w1_a", "lokr_w1_b"), ("lokr_w2_a", "lokr_w2_b")].iter()
        .filter_map(|(a, b)| Some((factors.get(*a)?, factors.get(*b)?)))
        .fold(0u64, |n, (a, b)| n.saturating_add(
            (a.shape.first().copied().unwrap_or(0).max(0) as u64)
                .saturating_mul(b.shape.get(1).copied().unwrap_or(0).max(0) as u64)
                .saturating_mul(width(dtype))))
}

fn target_weight_fact(target: &HeaderTarget) -> TensorFacts {
    let shape = if target.linear { target.shape.clone() } else {
        vec![target.shape[0], target.shape[2], target.shape[3], target.shape[1]]
    };
    output_fact(shape, target.weight_dtype)
}

fn header_shape(h: &mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader) -> Result<Vec<i32>> {
    h.shape
        .iter()
        .map(|d| {
            i32::try_from(*d)
                .map_err(|_| Error::Msg("sdxl adapter dimension exceeds native i32".into()))
        })
        .collect()
}

fn shape_elements(shape: &[i32]) -> Result<u64> {
    shape.iter().try_fold(1u64, |n, d| {
        u64::try_from(*d)
            .ok()
            .and_then(|d| n.checked_mul(d))
            .ok_or_else(|| Error::Msg("sdxl adapter shape element count overflow".into()))
    })
}

// Shape interpretation of the actual matmul/add/einsum reconstruction below. These functions
// describe logical installed layouts only; they never certify physical backing or donation.
pub(crate) fn broadcast_shape(a: &[i32], b: &[i32]) -> Result<Vec<i32>> {
    let mut result = Vec::with_capacity(a.len().max(b.len()));
    for i in 0..a.len().max(b.len()) {
        let x = a.len().checked_sub(i + 1).map_or(1, |j| a[j]);
        let y = b.len().checked_sub(i + 1).map_or(1, |j| b[j]);
        result.push(if x == y {
            x
        } else if x == 1 {
            y
        } else if y == 1 {
            x
        } else {
            return Err(Error::Msg("sdxl adapter operands do not broadcast".into()));
        });
    }
    result.reverse();
    Ok(result)
}

pub(crate) fn matmul_shape(a: &[i32], b: &[i32]) -> Result<Vec<i32>> {
    if a.is_empty() || b.is_empty() {
        return Err(Error::Msg("sdxl adapter matmul operand is scalar".into()));
    }
    let bv = b.len() == 1;
    let av = a.len() == 1;
    let inner = if bv { b[0] } else { b[b.len() - 2] };
    if a[a.len() - 1] != inner {
        return Err(Error::Msg(
            "sdxl adapter matmul inner dimensions differ".into(),
        ));
    }
    let mut result = broadcast_shape(
        &a[..a.len().saturating_sub(2)],
        &b[..b.len().saturating_sub(2)],
    )?;
    if !av {
        result.push(a[a.len() - 2]);
    }
    if !bv {
        result.push(b[b.len() - 1]);
    }
    Ok(result)
}

fn tucker_shape(t: &[i32], a: &[i32], b: &[i32]) -> Result<Vec<i32>> {
    let contracts = |x: i32, y: i32| x == y || (x == 1 && y > 0) || (y == 1 && x > 0);
    // Native einsum broadcasts singleton contracting axes before batch_tensordot. A zero axis
    // cannot broadcast to one at that later operator, while equal zero axes remain native-valid.
    if t.len() != 4 || a.len() != 2 || b.len() != 2
        || !contracts(t[0], a[0]) || !contracts(t[1], b[0]) {
        return Err(Error::Msg(
            "sdxl adapter tucker operands do not satisfy ijhw,ip,jr->prhw".into(),
        ));
    }
    Ok(vec![a[1], b[1], t[2], t[3]])
}

type HeaderFactors<'a> =
    BTreeMap<String, &'a mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader>;

fn required_factor<'a>(
    factors: &HeaderFactors<'a>,
    role: &str,
) -> Result<&'a mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader> {
    factors
        .get(role)
        .copied()
        .ok_or_else(|| Error::Msg(format!("sdxl adapter missing {role} factor")))
}

fn factor_product_shape(
    factors: &HeaderFactors<'_>,
    full: &str,
    a: &str,
    b: &str,
) -> Result<Vec<i32>> {
    if let Some(h) = factors.get(full) {
        return header_shape(h);
    }
    matmul_shape(
        &header_shape(required_factor(factors, a)?)?,
        &header_shape(required_factor(factors, b)?)?,
    )
}

/// Header counterpart of one actual selected factor rebuild. Only completed logical outputs are
/// appended: a second-factor refusal keeps the first product but never invents the second result.
fn prepared_factor(
    factors: &HeaderFactors<'_>, full: &str, a: &str, b: &str, tucker: Option<&str>,
    work: &mut Vec<AdapterWork>,
) -> Result<TensorFacts> {
    if let Some(header) = factors.get(full) { return header_tensor_fact(header); }
    let a = header_tensor_fact(required_factor(factors, a)?)?;
    let b = header_tensor_fact(required_factor(factors, b)?)?;
    let result = if let Some(t) = tucker.and_then(|key| factors.get(key)) {
        let t = header_tensor_fact(t)?;
        output_fact(tucker_shape(&t.shape, &a.shape, &b.shape)?,
            Dtype::from_promoting_types(t.dtype, Dtype::from_promoting_types(a.dtype, b.dtype)))
    } else {
        output_fact(matmul_shape(&a.shape, &b.shape)?, Dtype::from_promoting_types(a.dtype, b.dtype))
    };
    work.push(AdapterWork::Preparation(vec![result.clone()]));
    Ok(result)
}

fn prepared_kron(
    factors: &HeaderFactors<'_>, work: &mut Vec<AdapterWork>,
) -> Result<(Vec<i32>, Vec<i32>)> {
    let one = prepared_factor(factors, "lokr_w1", "lokr_w1_a", "lokr_w1_b", None, work)?;
    let two = prepared_factor(factors, "lokr_w2", "lokr_w2_a", "lokr_w2_b", Some("lokr_t2"), work)?;
    let mut left = one.shape.clone();
    left.resize(left.len().max(two.shape.len()), 1);
    let rank = left.len().max(two.shape.len());
    let mut output = vec![1; rank];
    for (shape, _) in [(&left, one.dtype), (&two.shape, two.dtype)] {
        for (out, dim) in output[rank - shape.len()..].iter_mut().zip(shape) {
            *out = i32::checked_mul(*out, *dim).ok_or_else(|| Error::Msg("sdxl kron dimension overflow".into()))?;
        }
    }
    let dtype = Dtype::from_promoting_types(one.dtype, two.dtype);
    work.push(AdapterWork::Preparation(vec![output_fact(output.clone(), dtype)]));
    // Scalar(scale) is created only AFTER kron succeeds, before the final reshape.
    work.push(AdapterWork::Preparation(vec![output_fact(Vec::new(), Dtype::Float32),
        output_fact(output, Dtype::from_promoting_types(dtype, Dtype::Float32))]));
    Ok((one.shape, two.shape))
}

fn kronecker_shapes(factors: &HeaderFactors<'_>, thirdparty: bool) -> Result<(Vec<i32>, Vec<i32>)> {
    let one = factor_product_shape(factors, "lokr_w1", "lokr_w1_a", "lokr_w1_b")?;
    let two = if let Some(h) = factors.get("lokr_w2") {
        header_shape(h)?
    } else if thirdparty && factors.contains_key("lokr_t2") {
        tucker_shape(
            &header_shape(required_factor(factors, "lokr_t2")?)?,
            &header_shape(required_factor(factors, "lokr_w2_a")?)?,
            &header_shape(required_factor(factors, "lokr_w2_b")?)?,
        )?
    } else {
        factor_product_shape(factors, "lokr_w2", "lokr_w2_a", "lokr_w2_b")?
    };
    Ok((one, two))
}

/// SDXL-owned normalization of the actual configured adapter install. Header sources are mapped
/// through the runtime's topology, dispatch, key classifiers, GEGLU routes and structured selector.
/// This is a semantic graph description, not a memory certificate. Tensor-alpha values are never
/// read: unlike generic residual installers, SDXL's TP delta branch has scale=1 after reconstruction.
pub(crate) fn configured_adapter_semantic_facts(
    spec: &mlx_gen::gen_core::LoadSpec,
    unet_headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
) -> Result<serde_json::Value> {
    configured_adapter_semantic_facts_with(
        spec,
        unet_headers,
        &crate::config::UNetConfig::sdxl_base(),
    )
}

/// The same production normalization over an explicit constructor config (tiny CPU parity fixtures).
pub(crate) fn configured_adapter_semantic_facts_with(
    spec: &mlx_gen::gen_core::LoadSpec,
    unet_headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
    config: &crate::config::UNetConfig,
) -> Result<serde_json::Value> {
    configured_adapter_work_with(spec, unet_headers, config).map(|(facts, _)| facts)
}

/// Captured source metadata only. Raw file key order is NOT the native HashMap iteration order.
/// Role choices below intentionally form a conservative superset of possible last-winner worlds;
/// no independent role maxima, dtype joins, alias caps, or eager Cartesian world list are used.
struct HeaderAdapterFile {
    headers: Vec<mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader>,
    metadata: BTreeMap<String, String>,
    kind: std::result::Result<InstallKind, String>,
    capture_error: Option<String>,
}

enum HeaderAdapterStep {
    File(usize),
    Parse(usize, Vec<usize>),
    Group { file: usize, raw: String, roles: Vec<(String, Vec<usize>)> },
    End,
}

struct HeaderWorldFrame {
    step: usize,
    choices: Vec<usize>,
    applied: bool,
    work_len: usize,
    semantic_len: usize,
    matched: usize,
    undo: Vec<(String, HeaderTarget)>,
}

/// One whole selected-file outcome, or the genuine prefix that reached a source refusal. The
/// prefix has no invented successful install/forward. This is logical source-upper authority only.
/// `facts` renders one active world; pricing consumes `work` before taking an envelope.
pub(crate) struct AdapterOutcome<'a> {
    pub work: &'a [AdapterWork],
    pub refusal: Option<String>,
    pub facts: serde_json::Value,
}

fn header_classification(
    name: &str, kind: InstallKind, lora: &BTreeMap<String, String>,
    vendored: &BTreeMap<String, String>,
) -> Option<(String, String)> {
    use mlx_gen::gen_core::weightsmeta::split_factor_key;
    match kind {
        InstallKind::Lora => classify_key(name, lora).map(|(path, role)| (path,
            match role { Role::Down => "down", Role::Up => "up", Role::Alpha => "alpha" }.into())),
        InstallKind::Lokr => classify_lokr_key(name, vendored).map(|(p, r)| (p, r.into())),
        InstallKind::ThirdPartyLokr | InstallKind::ThirdPartyLoha => name.strip_suffix(".alpha")
            .map(|p| (p.to_owned(), "alpha".into()))
            .or_else(|| split_factor_key(name, if kind == InstallKind::ThirdPartyLokr {
                &LOKR_TP_SUFFIXES
            } else { &LOHA_TP_SUFFIXES }).map(|(p, r)| (p.to_owned(), r.to_owned()))),
    }
}

/// Iterative DFS with one active prefix and per-group changed-target undo. Files are captured once;
/// all classified alpha operands (including overwritten and unresolved TP keys) are parsed before
/// any BTree install. Native key traversal, payload parsing, merge math and guard positions are not
/// modified. Each role choice remains correlated across every later file, base pack and forward.
pub(crate) fn visit_configured_adapter_outcomes(
    spec: &mlx_gen::gen_core::LoadSpec,
    unet_headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
    config: &crate::config::UNetConfig,
    mut visit: impl FnMut(AdapterOutcome<'_>) -> Result<()>,
) -> Result<()> {
    use mlx_gen::gen_core::weightsmeta::{safetensors_file_metadata, safetensors_path_tensor_headers};
    use serde_json::json;
    let (mut targets, vendored, complete) = header_targets(unet_headers, config, spec)?;
    let coverage = configured_lora_coverage();
    let lora = if coverage == LoraCoverage::Complete { &complete } else { &vendored };
    let mut files = Vec::new();
    let mut steps = Vec::new();
    for (file, adapter) in spec.adapters.iter().enumerate() {
        let captured = (|| -> Result<_> {
            let headers = safetensors_path_tensor_headers(&adapter.path).map_err(|e| Error::Msg(e.to_string()))?;
            let metadata = safetensors_file_metadata(&adapter.path).map_err(|e| Error::Msg(e.to_string()))?;
            Ok((headers, metadata))
        })();
        let entry = match captured {
            Ok((headers, metadata)) => {
                let kind = install_kind(adapter.kind, metadata.get("networkType").map(String::as_str),
                    headers.iter().map(|h| h.name.as_str())).map_err(|e| e.to_string());
                HeaderAdapterFile { headers, metadata, kind, capture_error: None }
            }
            Err(error) => HeaderAdapterFile { headers: Vec::new(), metadata: BTreeMap::new(),
                kind: Err(error.to_string()), capture_error: Some(error.to_string()) },
        };
        steps.push(HeaderAdapterStep::File(file));
        let invalid_alpha = entry.headers.iter().enumerate().filter_map(|(index, header)| {
            let kind = *entry.kind.as_ref().ok()?;
            let (_, role) = header_classification(&header.name, kind, lora, &vendored)?;
            if role != "alpha" { return None; }
            let count = header.element_count().ok()?;
            (count == 0 || (kind == InstallKind::Lora && count != 1)).then_some(index)
        }).collect();
        steps.push(HeaderAdapterStep::Parse(file, invalid_alpha));
        if let Ok(kind) = &entry.kind {
            let mut groups: BTreeMap<String, BTreeMap<String, Vec<usize>>> = BTreeMap::new();
            for (index, header) in entry.headers.iter().enumerate() {
                if let Some((raw, role)) = header_classification(&header.name, *kind, lora, &vendored) {
                    groups.entry(raw).or_default().entry(role).or_default().push(index);
                }
            }
            // TP groups remain raw-key ordered SEQUENTIAL installs, even when two raw groups resolve
            // to the same module. Only actual classified-role collisions are winner alternatives.
            for (raw, roles) in groups {
                steps.push(HeaderAdapterStep::Group { file, raw, roles: roles.into_iter().collect() });
            }
        }
        steps.push(HeaderAdapterStep::End);
        files.push(entry);
    }
    let mut work = Vec::new();
    let mut semantic: Vec<(usize, serde_json::Value)> = Vec::new();
    let mut matched = 0usize;
    let render = |targets: &BTreeMap<String, HeaderTarget>, semantic: &[(usize, serde_json::Value)]| {
        let target_facts: BTreeMap<_, _> = targets.iter().map(|(name, t)| (name, json!({
            "shape":t.shape,"kind":if t.linear{"linear"}else{"conv"},"source_packed_bits":t.source_packed_bits,
            "weight_dtype":format!("{:?}",t.weight_dtype),"scales_dtype":t.scales_dtype.map(|d|format!("{d:?}")),
            "affine_biases_dtype":t.affine_biases_dtype.map(|d|format!("{d:?}")),"packing":t.packing,
        }))).collect();
        let adapters: Vec<_> = files.iter().enumerate().map(|(file, entry)| {
            let mut read: Vec<_> = entry.headers.iter().map(|h| json!({"shape":h.shape,
                "dtype":crate::quant::header_dtype(h.dtype).map(|d|format!("{d:?}"))
                    .unwrap_or_else(|e|e.to_string())})).collect();
            read.sort_by_key(serde_json::Value::to_string);
            let recipe: Vec<_> = entry.headers.iter().filter_map(|header| {
                let kind = *entry.kind.as_ref().ok()?;
                let (raw, role) = header_classification(&header.name, kind, lora, &vendored)?;
                Some(json!({"raw_group":raw,"role":role,"source_key":header.name,
                    "shape":header.shape,"dtype":format!("{:?}",header.dtype)}))
            }).collect();
            json!({"kind":entry.kind.as_ref().ok().map(|k|format!("{k:?}")),"read_layouts":read,
                "winner_recipe":recipe,"metadata":entry.metadata,
                "installed":semantic.iter().filter(|(i,_)| *i==file).map(|(_,v)|v).collect::<Vec<_>>()})
        }).collect();
        json!({"coverage":format!("{coverage:?}"),"targets":target_facts,"adapters":adapters,
            "outcome_authority":"conservative_raw_order_superset"})
    };
    if steps.is_empty() {
        return visit(AdapterOutcome { work: &work, refusal: None, facts: render(&targets, &semantic) });
    }
    let frame = |step, work_len, semantic_len, matched| HeaderWorldFrame {
        step, work_len, semantic_len, matched, applied: false, undo: Vec::new(),
        choices: match &steps[step] {
            HeaderAdapterStep::Group { roles, .. } => vec![0; roles.len()],
            HeaderAdapterStep::Parse(_, invalid) if !invalid.is_empty() => vec![0],
            _ => Vec::new(),
        },
    };
    let mut stack = vec![frame(0, 0, 0, 0)];
    while let Some(active) = stack.last_mut() {
        if active.applied {
            work.truncate(active.work_len);
            semantic.truncate(active.semantic_len);
            matched = active.matched;
            for (name, target) in active.undo.drain(..) { targets.insert(name, target); }
            let mut next = false;
            if let HeaderAdapterStep::Group { roles, .. } = &steps[active.step] {
                for index in (0..roles.len()).rev() {
                    active.choices[index] += 1;
                    if active.choices[index] < roles[index].1.len() { next = true; break; }
                    active.choices[index] = 0;
                }
            }
            if let HeaderAdapterStep::Parse(_, invalid) = &steps[active.step] {
                if !invalid.is_empty() {
                    active.choices[0] += 1;
                    next = active.choices[0] < invalid.len();
                }
            }
            if !next { stack.pop(); continue; }
        }
        active.applied = true;
        let result = (|| -> Result<()> {
            match &steps[active.step] {
                HeaderAdapterStep::File(file) => {
                    let entry = &files[*file];
                    if let Some(error) = &entry.capture_error { return Err(Error::Msg(error.clone())); }
                    work.push(AdapterWork::FileMaterialized(entry.headers.iter().map(|h|
                        Ok((h.name.clone(), header_tensor_fact(h)?))).collect::<Result<_>>()?));
                    let kind = entry.kind.as_ref().map_err(|e| Error::Msg(e.clone()))?;
                    work.push(AdapterWork::Selected(*kind));
                }
                HeaderAdapterStep::Parse(file, invalid) => {
                    let entry = &files[*file];
                    let kind = *entry.kind.as_ref().map_err(|e| Error::Msg(e.clone()))?;
                    for header in &entry.headers {
                        let Some((_, role)) = header_classification(&header.name, kind, lora, &vendored) else { continue; };
                        if role != "alpha" { continue; }
                        let fact = header_tensor_fact(header)?;
                        let count = fact.elements();
                        let valid = count != 0 && (kind != InstallKind::Lora || count == 1);
                        if !valid { continue; }
                        // Scalar extraction copies f32; its cast is NOT a retained sharedData root.
                        let scratch = if fact.dtype == Dtype::Float32 { 0 } else {
                            mlx_gen::ios_quantum::allocated_bytes(count.saturating_mul(width(Dtype::Float32)))
                        };
                        work.push(AdapterWork::AlphaRead { key: header.name.clone(), scratch });
                    }
                    if !invalid.is_empty() {
                        let header = &entry.headers[invalid[active.choices[0]]];
                        // LoRA constructs its FULL f32 cast before reshape([1]) can refuse; TP's
                        // size-zero predicate refuses before any cast. Enumerate each possible first
                        // invalid operand rather than borrowing one invalid key's smaller workspace.
                        if kind == InstallKind::Lora {
                            let fact = header_tensor_fact(header)?;
                            let scratch = if fact.dtype == Dtype::Float32 { 0 } else {
                                mlx_gen::ios_quantum::allocated_bytes(fact.elements().saturating_mul(4))
                            };
                            work.push(AdapterWork::AlphaRead { key: header.name.clone(), scratch });
                        }
                        return Err(Error::Msg(format!("sdxl adapter alpha shape violates actual scalar conversion: {}", header.name)));
                    }
                }
                HeaderAdapterStep::Group { file, raw, roles } => {
                    let entry = &files[*file];
                    let kind = *entry.kind.as_ref().map_err(|e| Error::Msg(e.clone()))?;
                    let factors: HeaderFactors<'_> = roles.iter().zip(&active.choices)
                        .map(|((role, candidates), choice)| (role.clone(), &entry.headers[candidates[*choice]])).collect();
                    let resolved = if matches!(kind, InstallKind::ThirdPartyLokr | InstallKind::ThirdPartyLoha) {
                        resolve_lokr_path(raw, &vendored).map(str::to_owned)
                    } else { Some(raw.clone()) };
                    if let Some(path) = resolved {
                        let rows = factors.get("up").map(|h| header_shape(h)).transpose()?
                            .and_then(|shape| shape.first().copied()).unwrap_or(0);
                        let routes = if kind == InstallKind::Lora { lora_output_rows(&path, rows) }
                            else { vec![(path, None)] };
                        for (name, _) in routes {
                            if let Some(target) = targets.get(&name) { active.undo.push((name, target.clone())); }
                        }
                    }
                    let mut installed = Vec::new();
                    let result = selected_header_group(raw, factors, kind, &entry.metadata,
                        &spec.adapters[*file], coverage, &vendored, &mut targets, &mut work,
                        &mut installed, &mut matched);
                    semantic.extend(installed.into_iter().map(|v| (*file, v)));
                    result?;
                }
                HeaderAdapterStep::End => work.push(AdapterWork::FileScopeEnd),
            }
            Ok(())
        })();
        let next = active.step + 1;
        if let Err(error) = result {
            visit(AdapterOutcome { work: &work, refusal: Some(error.to_string()), facts: render(&targets, &semantic) })?;
        } else if next == steps.len() {
            visit(AdapterOutcome { work: &work,
                refusal: (matched == 0).then(|| "sdxl: no adapter target modules matched".into()),
                facts: render(&targets, &semantic) })?;
        } else {
            stack.push(frame(next, work.len(), semantic.len(), matched));
        }
    }
    Ok(())
}

/// Legacy single-world witness access. Ambiguous files expose a compact source recipe, never an
/// eager JSON world list; production pricing uses the visitor and cannot choose this first world.
pub(crate) fn configured_adapter_work_with(
    spec: &mlx_gen::gen_core::LoadSpec,
    unet_headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
    config: &crate::config::UNetConfig,
) -> Result<(serde_json::Value, Vec<AdapterWork>)> {
    let mut first = None;
    let mut refusal = None;
    let mut count = 0u64;
    visit_configured_adapter_outcomes(spec, unet_headers, config, |outcome| {
        count = count.saturating_add(1);
        if let Some(error) = outcome.refusal { refusal.get_or_insert(error); }
        else if first.is_none() { first = Some((outcome.facts, outcome.work.to_vec())); }
        Ok(())
    })?;
    let Some((mut facts, work)) = first else { return Err(Error::Msg(refusal.unwrap_or_else(||
        "sdxl adapter outcome visitor returned no world".into()))); };
    if count > 1 {
        facts["outcome_space"] = serde_json::json!({"kind":"lazy_classified_role_winners",
            "authority":"conservative_superset","selected_outcome_count":count,
            "files":spec.adapters.iter().map(|a| &a.path).collect::<Vec<_>>()});
    }
    Ok((facts, work))
}

/// One selected BTree group, sharing the existing install numeric constructors. Mutation is
/// limited to the routed targets; the visitor journals those targets before entering this function.
fn selected_header_group(
    raw: &str, factors: HeaderFactors<'_>, kind: InstallKind,
    metadata: &BTreeMap<String, String>, adapter: &AdapterSpec, coverage: LoraCoverage,
    vendored: &BTreeMap<String, String>, targets: &mut BTreeMap<String, HeaderTarget>,
    work: &mut Vec<AdapterWork>, installed: &mut Vec<serde_json::Value>, matched: &mut usize,
) -> Result<()> {
    use serde_json::json;
    let cfg = LoraAdapterMeta::from_metadata(metadata.get(LORA_ADAPTER_METADATA_KEY).map(String::as_str));
        for (raw, factors) in std::iter::once((raw.to_owned(), factors)) {
            let path = if matches!(
                kind,
                InstallKind::ThirdPartyLokr | InstallKind::ThirdPartyLoha
            ) {
                let Some(path) = resolve_lokr_path(&raw, &vendored) else {
                    continue;
                };
                path.to_owned()
            } else {
                raw
            };
            if kind == InstallKind::Lora {
                let (Some(down), Some(up)) = (factors.get("down"), factors.get("up")) else {
                    continue;
                };
                let ds = header_shape(down)?;
                let us = header_shape(up)?;
                let (cfg_alpha, cfg_rank) =
                    cfg.as_ref().map_or((None, None), |c| c.effective(&path));
                let _ = cfg_alpha; // Value-only scale operand: it selects no SDXL install/forward graph.
                if ds.len() == 4 || us.len() == 4 {
                    if coverage != LoraCoverage::Complete {
                        continue;
                    }
                    if ds.len() != 4
                        || us.len() != 4
                        || us[1] != ds[0]
                        || shape_elements(&us[2..])? != 1
                    {
                        return Err(Error::Msg(
                            "sdxl conv LoRA cannot reshape/multiply its actual factors".into(),
                        ));
                    }
                    let delta = vec![us[0], ds[1], ds[2], ds[3]];
                    // conv_lora_delta executes before routing, including non-routable paths.
                    let down_fact = header_tensor_fact(down)?;
                    let up_fact = header_tensor_fact(up)?;
                    for fact in [&up_fact, &down_fact] {
                        if fact.dtype != Dtype::Float32 {
                            work.push(AdapterWork::Preparation(vec![output_fact(fact.shape.clone(), Dtype::Float32)]));
                        }
                    }
                    let product = vec![us[0], ds[1].checked_mul(ds[2]).and_then(|d| d.checked_mul(ds[3]))
                        .ok_or_else(|| Error::Msg("sdxl conv LoRA reshape dimension overflow".into()))?];
                    work.push(AdapterWork::Preparation(vec![output_fact(product.clone(), Dtype::Float32)]));
                    if up_fact.dtype != Dtype::Float32 {
                        work.push(AdapterWork::Preparation(vec![output_fact(product.clone(), up_fact.dtype),
                            output_fact(product.clone(), Dtype::Float32)]));
                    }
                    work.push(AdapterWork::Preparation(vec![output_fact(Vec::new(), Dtype::Float32),
                        output_fact(product, Dtype::Float32)]));
                    let Some(target) = targets.get_mut(&path) else {
                        continue;
                    };
                    if target.linear && (delta[2] != 1 || delta[3] != 1) {
                        continue;
                    }
                    let output = if target.linear { delta[..2].to_vec() } else { delta.clone() };
                    let base = target_weight_fact(target);
                    let unread_base = if target.unread { base.bytes() } else { 0 };
                    let down_work = header_tensor_fact(down)?;
                    let up_work = header_tensor_fact(up)?;
                    let copy = |a: &TensorFacts| if a.dtype == Dtype::Float32 { 0 } else {
                        a.elements().saturating_mul(width(Dtype::Float32))
                    };
                    let small = copy(&down_work).saturating_add(copy(&up_work));
                    let packed = target.source_packed_bits.is_some();
                    let point = if target.linear {
                        AdapterInstallWork::delta(if packed { None } else { Some(base) },
                            output_fact(output.clone(), Dtype::Float32), &lora_chain(up_work.dtype),
                            small, if packed { 0 } else { unread_base },
                            mlx_gen::ios_quantum::allocated_bytes(width(Dtype::Float32)))
                    } else {
                        AdapterInstallWork::conv(base, output_fact(delta, Dtype::Float32),
                            up_work.dtype, small, unread_base)
                    }.at(&path);
                    work.push(AdapterWork::GuardAttempt(point));
                    if !packed {
                        target.shape = broadcast_shape(&target.shape, &output)?;
                        target.unread = false;
                    }
                    if let Some(AdapterWork::GuardAttempt(point)) = work.last_mut() {
                        point.installed = vec![if packed { output_fact(output, Dtype::Float32) }
                            else { target_weight_fact(target) }];
                    }
                    if let Some(AdapterWork::GuardAttempt(point)) = work.last() {
                        work.push(AdapterWork::Install(point.clone()));
                    }
                    *matched += 1;
                    installed.push(json!({"target":path,"kind":"conv_lora","reconstruction":"f32_product_source_roundtrip",
                        "target_shape":target.shape,"installed":if target.source_packed_bits.is_some(){"materialized_delta_residual"}else{"merged_dense"},
                        "installed_dtype":if target.source_packed_bits.is_some(){"Float32".to_owned()}else{format!("{:?}",target.weight_dtype)},
                        "factor_layouts":factor_semantic_facts(&factors)?,"configured_zero_branch":false}));
                    continue;
                }
                if ds.len() != 2 || us.len() != 2 {
                    continue;
                }
                if coverage == LoraCoverage::Vendored && is_complete_only(&path) {
                    continue;
                }
                let rank = cfg_rank.unwrap_or(ds[0] as f32);
                for (route, rows) in lora_output_rows(&path, us[0]) {
                    if let Some((lo, hi)) = rows {
                        work.push(AdapterWork::Preparation(vec![
                            output_fact(vec![hi - lo], Dtype::Int32),
                            output_fact(vec![hi - lo, us[1]], crate::quant::header_dtype(up.dtype)?),
                        ]));
                    }
                    let Some(target) = targets.get_mut(&route).filter(|t| t.linear) else {
                        continue;
                    };
                    let output_rows = rows.map_or(us[0], |(lo, hi)| hi - lo);
                    let packed = target.source_packed_bits.is_some();
                    let down_work = header_tensor_fact(down)?;
                    let up_work = output_fact(vec![output_rows, us[1]],
                        mlx_gen::adapters::facts::native_header_dtype(up.dtype)?);
                    let base = target_weight_fact(target);
                    let unread_base = if target.unread && !packed { base.bytes() } else { 0 };
                    let point = AdapterInstallWork::lora(if packed { None } else { Some(base) },
                        down_work.clone(), up_work.clone(), unread_base,
                        rows.map_or(0, |(lo, hi)| hi - lo), packed && adapter.scale == 0.0).at(&route);
                    work.push(AdapterWork::GuardAttempt(point));
                    if !packed {
                        if rank == 0.0 {
                            return Err(Error::Msg(
                                "lora_delta: zero rank — malformed adapter".into(),
                            ));
                        }
                        let output = matmul_shape(&[output_rows, us[1]], &ds)?;
                        target.shape = broadcast_shape(&target.shape, &output)?;
                        target.unread = false;
                    }
                    if let Some(AdapterWork::GuardAttempt(point)) = work.last_mut() {
                        point.installed = if packed {
                            vec![output_fact(vec![ds[1], ds[0]], down_work.dtype),
                                output_fact(vec![us[1], output_rows], up_work.dtype)]
                        } else { vec![target_weight_fact(target)] };
                    }
                    if let Some(AdapterWork::GuardAttempt(point)) = work.last() {
                        work.push(AdapterWork::Install(point.clone()));
                    }
                    *matched += 1;
                    installed.push(json!({"target":route,"kind":"lora","target_shape":target.shape,
                        "source_packed_bits":target.source_packed_bits,"installed":if packed {"factor_residual"}else{"merged_dense"},
                        "reconstruction":if packed {"transpose_A_fold_alpha_into_B"}else{"f32_product_source_roundtrip"},
                        "down_shape":ds,"up_shape":[output_rows,us[1]],"up_rows":rows,
                        "down_dtype":format!("{:?}",crate::quant::header_dtype(down.dtype)?),
                        "up_dtype":format!("{:?}",crate::quant::header_dtype(up.dtype)?),
                        "configured_zero_branch":packed && adapter.scale==0.0}));
                }
                continue;
            }
            if is_complete_only(&path) {
                continue;
            }
            let Some(target) = targets.get_mut(&path).filter(|t| t.linear) else {
                continue;
            };
            let factor_work = header_factor_work(&factors)?;
            let factor = factor_work.values().map(|a| a.dtype)
                .reduce(Dtype::from_promoting_types).unwrap_or(Dtype::Float32);
            let products = factor_work_products(&factor_work, factor);
            let base = target_weight_fact(target);
            let packed = target.source_packed_bits.is_some();
            let unread_base = if target.unread && !packed { base.bytes() } else { 0 };
            let mut native_peft_point = None;
            if kind == InstallKind::Lokr {
                let rank = metadata.get("rank").and_then(|s| s.parse::<f32>().ok()).unwrap_or(1.0);
                let alpha = metadata.get("alpha").and_then(|s| s.parse::<f32>().ok()).unwrap_or(rank);
                let casts = factor_work.values().filter(|a| a.dtype != Dtype::Float32)
                    .fold(0u64, |n, a| n.saturating_add(a.elements().saturating_mul(4)));
                let point = if packed {
                    AdapterInstallWork::structured(&factor_work,
                        products.saturating_mul(2).saturating_add(casts), (alpha / rank) * adapter.scale == 0.0)
                } else {
                    AdapterInstallWork::dense_lokr(base.clone(), factor_work.values().cloned().collect(),
                        factor, products, unread_base)
                }.at(&path);
                // PEFT reconstruction/validation executes INSIDE the existing target guard.
                native_peft_point = Some(work.len());
                work.push(AdapterWork::GuardAttempt(point));
                // Match the native constructor INSIDE the reached guard: scale validation
                // precedes factor reconstruction/layout. Packed validates only the FULL scale;
                // dense PEFT independently requires finite positive rank and finite alpha.
                if packed {
                    let scale = (alpha / rank) * adapter.scale;
                    if !scale.is_finite() {
                        return Err(Error::Msg("LoKr: non-finite structured scale".into()));
                    }
                } else {
                    mlx_gen::adapters::validate_lokr_metadata_scale(alpha, rank)?;
                }
            }
            let mut disabled = false;
            let (reconstruction, storage, shapes) = if kind == InstallKind::Lokr
                || kind == InstallKind::ThirdPartyLokr
            {
                let thirdparty = kind == InstallKind::ThirdPartyLokr;
                if thirdparty {
                    let w1a = factors
                        .get("lokr_w1_a")
                        .map(|h| header_shape(h))
                        .transpose()?;
                    let t2 = factors
                        .get("lokr_t2")
                        .map(|h| header_shape(h))
                        .transpose()?;
                    let w2a = factors
                        .get("lokr_w2_a")
                        .map(|h| header_shape(h))
                        .transpose()?;
                    mlx_gen::adapters::loader::thirdparty_lokr_rank(
                        w1a.as_deref(),
                        t2.as_deref(),
                        w2a.as_deref(),
                    )?;
                }
                let (one, two) = if thirdparty { prepared_kron(&factors, work)? }
                    else { kronecker_shapes(&factors, false)? };
                let packed = target.source_packed_bits.is_some();
                let storage = if packed && !thirdparty {
                    let rank = metadata
                        .get("rank")
                        .and_then(|s| s.parse::<f32>().ok())
                        .unwrap_or(1.0);
                    let alpha = metadata
                        .get("alpha")
                        .and_then(|s| s.parse::<f32>().ok())
                        .unwrap_or(rank);
                    let scale = (alpha / rank) * adapter.scale;
                    mlx_gen::adapters::lokr_structured_layout(&target.shape, &one, &two, false)
                        .ok_or_else(|| non_deferrable_lokr_error(&path))?;
                    disabled = scale == 0.0;
                    "structured_residual"
                } else {
                    if shape_elements(&one)?.checked_mul(shape_elements(&two)?)
                        != Some(shape_elements(&target.shape)?)
                    {
                        return Err(Error::Msg(
                            "sdxl LoKr delta cannot reshape to target".into(),
                        ));
                    }
                    if packed {
                        "materialized_delta_residual"
                    } else {
                        "merged_dense"
                    }
                };
                ("kronecker", storage, vec![one, two])
            } else {
                for role in ["hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"] {
                    required_factor(&factors, role)?;
                }
                let w1b = header_shape(required_factor(&factors, "hada_w1_b")?)?;
                mlx_gen::adapters::loader::thirdparty_loha_rank(Some(&w1b))?;
                let tucker = factors.contains_key("hada_t1") && factors.contains_key("hada_t2");
                let one = prepared_factor(&factors, "hada_w1", "hada_w1_a", "hada_w1_b",
                    tucker.then_some("hada_t1"), work)?;
                let two = prepared_factor(&factors, "hada_w2", "hada_w2_a", "hada_w2_b",
                    tucker.then_some("hada_t2"), work)?;
                let combined = broadcast_shape(&one.shape, &two.shape)?;
                let product_dtype = Dtype::from_promoting_types(one.dtype, two.dtype);
                work.push(AdapterWork::Preparation(vec![output_fact(combined.clone(), product_dtype)]));
                work.push(AdapterWork::Preparation(vec![output_fact(Vec::new(), Dtype::Float32),
                    output_fact(combined.clone(), Dtype::from_promoting_types(product_dtype, Dtype::Float32))]));
                let (one, two) = (one.shape, two.shape);
                if shape_elements(&combined)? != shape_elements(&target.shape)?
                {
                    return Err(Error::Msg(
                        "sdxl LoHa delta cannot reshape to target".into(),
                    ));
                }
                (
                    "hadamard",
                    if target.source_packed_bits.is_some() {
                        "materialized_delta_residual"
                    } else {
                        "merged_dense"
                    },
                    vec![one, two],
                )
            };
            if matches!(kind, InstallKind::ThirdPartyLokr | InstallKind::ThirdPartyLoha) {
                // User strength scalar/multiply follows successful delta reshape and output cast.
                work.push(AdapterWork::Preparation(vec![output_fact(Vec::new(), Dtype::Float32),
                    output_fact(target.shape.clone(), Dtype::Float32)]));
            }
            if let Some(index) = native_peft_point {
                if let AdapterWork::GuardAttempt(point) = &mut work[index] {
                    point.installed = if packed {
                        vec![output_fact(shapes[0].clone(), Dtype::Float32),
                            output_fact(shapes[1].clone(), Dtype::Float32)]
                    } else { vec![target_weight_fact(target)] };
                }
            } else {
                let small = factor_work.values().fold(0u64, |n, a|
                    n.saturating_add(a.elements().saturating_mul(4)));
                let chain = if kind == InstallKind::ThirdPartyLokr { kron_chain(factor) }
                    else { hadamard_chain(factor) };
                let small = if kind == InstallKind::ThirdPartyLokr { small.saturating_add(products) }
                    else { small };
                // Third-party reconstruction/shape refusal occurs BEFORE this guarded install.
                let mut point = AdapterInstallWork::delta(if packed { None } else { Some(base) },
                    output_fact(target.shape.clone(), Dtype::Float32), &chain, small, unread_base,
                    mlx_gen::ios_quantum::allocated_bytes(width(Dtype::Float32)).saturating_mul(2)).at(&path);
                point.parents.extend(factor_work.values().cloned());
                point.installed = vec![if packed { output_fact(target.shape.clone(), Dtype::Float32) }
                    else { target_weight_fact(target) }];
                work.push(AdapterWork::GuardAttempt(point));
            }
            if let Some(AdapterWork::GuardAttempt(point)) = work.last() {
                work.push(AdapterWork::Install(point.clone()));
            }
            if !packed { target.unread = false; }
            *matched += 1;
            installed.push(json!({"target":path,"kind":format!("{kind:?}"),"target_shape":target.shape,
                "source_packed_bits":target.source_packed_bits,"installed":storage,"reconstruction":reconstruction,
                "reconstructed_factor_shapes":shapes,"factor_layouts":factor_semantic_facts(&factors)?,
                "configured_zero_branch":disabled}));
        }
    Ok(())
}

fn factor_semantic_facts(factors: &HeaderFactors<'_>) -> Result<serde_json::Value> {
    let mut result = serde_json::Map::new();
    for (role, header) in factors {
        if role == "alpha" {
            continue;
        } // A value operand, never a graph selector in SDXL's TP merge.
        result.insert(role.clone(), serde_json::json!({
            "shape": header.shape, "dtype": format!("{:?}", crate::quant::header_dtype(header.dtype)?),
        }));
    }
    Ok(serde_json::Value::Object(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::all_close;

    #[test]
    fn semantic_dispatch_is_actual_sdxl_precedence() {
        let kohya = ["lora_unet_down_blocks_1.lora_down.weight"];
        assert_eq!(
            install_kind(AdapterKind::Lokr, None, kohya).unwrap(),
            InstallKind::Lokr
        );
        assert_eq!(
            install_kind(AdapterKind::Lora, None, kohya).unwrap(),
            InstallKind::Lora
        );
        let mixed = ["target.lokr_w1", "target.hada_w1_a"];
        assert_eq!(
            install_kind(AdapterKind::Lora, None, mixed).unwrap(),
            InstallKind::ThirdPartyLokr
        );
        assert_eq!(
            install_kind(AdapterKind::Lokr, Some("lokr"), mixed).unwrap(),
            InstallKind::ThirdPartyLoha
        );
        assert!(install_kind(AdapterKind::Lora, Some("lokr"), kohya).is_err());
        assert_eq!(
            install_kind(AdapterKind::Lokr, Some("lokr"), ["target.lokr_w1"]).unwrap(),
            InstallKind::Lokr
        );
    }

    #[test]
    fn semantic_geglu_rows_share_actual_route_including_odd_rows() {
        assert_eq!(
            lora_output_rows("x.ff.net.0.proj", 5),
            vec![
                ("x.ff.linear1".into(), Some((0, 2))),
                ("x.ff.linear2".into(), Some((2, 5))),
            ]
        );
        assert_eq!(
            lora_output_rows("x.ff.net.2", 7),
            vec![("x.ff.linear3".into(), None)]
        );
        assert_eq!(
            lora_output_rows("x.attn1.to_q", 7),
            vec![("x.attn1.to_q".into(), None)]
        );
    }

    fn table() -> BTreeMap<String, String> {
        // A tiny routable surface: one attention leaf + a proj.
        [
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q",
            "up_blocks.0.attentions.0.proj_in",
        ]
        .into_iter()
        .map(|p| (p.replace('.', "_"), p.to_string()))
        .collect()
    }

    /// sc-3671: the SDXL crate reconstructs third-party LoKr/LoHa deltas (via the shared core pub
    /// helpers) bit-for-bit against the lycoris reference fixtures (`<repo>/tests/fixtures`, generated
    /// by `tools/sc364{2,3}_*reference.py` through `~/mlx-flux-venv`), and detects + resolves the
    /// trainer-flattened keys. The merge onto the UNet is the existing `merge_dense_delta` path.
    #[test]
    fn thirdparty_lycoris_reconstructs_against_reference_f32() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        // (fixture dir, file stem, is_loha)
        let cases: [(&str, &str, bool); 4] = [
            ("sc3642_lokr", "linear_w1full_w2lr", false),
            ("sc3642_lokr", "linear_bothlr", false),
            ("sc3642_lokr", "linear_bothfull", false),
            ("sc3643_loha", "linear", true),
        ];
        for (dir, stem, is_loha) in cases {
            let base = root.join("tests/fixtures").join(dir);
            let w = Weights::from_file(base.join(format!("{stem}.safetensors"))).unwrap();
            let exp =
                Weights::from_file(base.join(format!("{stem}.expected.safetensors"))).unwrap();
            if is_loha {
                assert!(is_loha_keys(&w), "{stem}: not detected as LoHa");
            } else {
                assert!(
                    is_lokr_keys(&w) && !is_lokr(&w),
                    "{stem}: not detected as 3rd-party LoKr"
                );
            }
            let table: BTreeMap<String, String> = exp
                .keys()
                .map(|d| (d.replace('.', "_"), d.to_string()))
                .collect();
            // Each fixture targets a single module ("proj").
            let want = exp.require("proj").unwrap();
            let got = if is_loha {
                let g = parse_loha_thirdparty(&w).unwrap();
                let (raw, m) = g.iter().next().unwrap();
                assert_eq!(resolve_lokr_path(raw, &table), Some("proj"));
                m.delta(want.shape(), Dtype::Float32).unwrap()
            } else {
                let g = parse_lokr_thirdparty(&w).unwrap();
                let (raw, m) = g.iter().next().unwrap();
                assert_eq!(resolve_lokr_path(raw, &table), Some("proj"));
                m.delta(want.shape(), Dtype::Float32).unwrap()
            };
            assert!(
                all_close(&got, want, 1e-4, 1e-5, false)
                    .unwrap()
                    .item::<bool>(),
                "{stem}: SDXL third-party reconstruction diverged from lycoris reference"
            );
        }
    }

    #[test]
    fn classify_kohya_resolves_flattened_stem_incl_to_out_0() {
        let t = table();
        // A kohya `to_q` key resolves through the flattened-stem table to its dotted path.
        let (path, role) = classify_key(
            "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight",
            &t,
        )
        .expect("kohya to_q should resolve");
        assert_eq!(
            path,
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"
        );
        assert!(matches!(role, Role::Down));
        // up + alpha roles classify too.
        assert!(matches!(
            classify_key(
                "lora_unet_up_blocks_0_attentions_0_proj_in.lora_up.weight",
                &t
            )
            .unwrap()
            .1,
            Role::Up
        ));
        assert!(matches!(
            classify_key("lora_unet_up_blocks_0_attentions_0_proj_in.alpha", &t)
                .unwrap()
                .1,
            Role::Alpha
        ));
    }

    /// sc-6051: an original-SD / A1111 kohya key (`lora_unet_input_blocks_4_1_…`) classifies onto the
    /// SAME diffusers dotted path as its `down_blocks` twin, so civitai SDXL LoRAs merge. The
    /// `table()` helper holds the diffusers paths; the translation runs inside `classify_key`.
    #[test]
    fn classify_kohya_translates_original_sd_naming() {
        let t = table();
        let (path, role) = classify_key(
            "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.lora_down.weight",
            &t,
        )
        .expect("original-SD input_blocks key should translate + resolve");
        assert_eq!(
            path,
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"
        );
        assert!(matches!(role, Role::Down));
        // output_blocks.0.* → up_blocks.0.* (proj_in lives on the attention module at index .1).
        assert_eq!(
            classify_key("lora_unet_output_blocks_0_1_proj_in.lora_up.weight", &t)
                .unwrap()
                .0,
            "up_blocks.0.attentions.0.proj_in"
        );
    }

    #[test]
    fn classify_skips_out_of_surface_and_text_encoder_keys() {
        let t = table();
        // mid_block / ff / conv stems aren't in the table → None (skipped, surfaced upstream).
        assert!(classify_key(
            "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight",
            &t
        )
        .is_none());
        assert!(classify_key(
            "lora_unet_down_blocks_1_resnets_0_conv1.lora_down.weight",
            &t
        )
        .is_none());
        // text-encoder LoRA keys are never UNet targets.
        assert!(classify_key(
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight",
            &t
        )
        .is_none());
    }

    #[test]
    fn classify_lokr_resolves_factors_via_table_and_bare() {
        let t = table();
        // kohya LoKr factor resolves the flattened stem via the table; longest suffix wins
        // (`.lokr_w1_a` over `.lokr_w1`).
        let (path, factor) =
            classify_lokr_key("lora_unet_up_blocks_0_attentions_0_proj_in.lokr_w1_a", &t)
                .expect("kohya lokr_w1_a should resolve");
        assert_eq!(path, "up_blocks.0.attentions.0.proj_in");
        assert_eq!(factor, "lokr_w1_a");
        assert_eq!(
            classify_lokr_key("lora_unet_up_blocks_0_attentions_0_proj_in.lokr_w2", &t)
                .unwrap()
                .1,
            "lokr_w2"
        );
        // bare / PEFT dotted paths resolve directly (no table); off-surface kohya stems are None.
        assert_eq!(
            classify_lokr_key("base_model.model.unet.foo.bar.lokr_w1", &t).unwrap(),
            ("foo.bar".to_string(), "lokr_w1")
        );
        assert!(
            classify_lokr_key("lora_unet_mid_block_attentions_0_proj_in.lokr_w1", &t).is_none()
        );
    }

    #[test]
    fn is_complete_only_flags_mid_block_and_ff_only() {
        // mid_block (any leaf) and GEGLU FF leaves are complete-only.
        for p in [
            "mid_block.attentions.0.transformer_blocks.0.attn1.to_q",
            "mid_block.attentions.0.proj_in",
            "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj",
            "up_blocks.0.attentions.0.transformer_blocks.0.ff.net.2",
        ] {
            assert!(is_complete_only(p), "{p} should be complete-only");
        }
        // The vendored-faithful surface is NOT complete-only.
        for p in [
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q",
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0",
            "up_blocks.0.attentions.0.proj_in",
            "down_blocks.0.resnets.0.time_emb_proj",
        ] {
            assert!(
                !is_complete_only(p),
                "{p} must stay in the faithful surface"
            );
        }
    }

    #[test]
    fn classify_complete_table_resolves_mid_block_and_ff_stems() {
        // A complete-style table additionally carries mid_block + ff diffusers paths.
        let t: BTreeMap<String, String> = [
            "mid_block.attentions.0.transformer_blocks.0.attn1.to_q",
            "mid_block.attentions.0.proj_in",
            "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj",
            "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.2",
        ]
        .into_iter()
        .map(|p| (p.replace('.', "_"), p.to_string()))
        .collect();
        let (mid, _) = classify_key(
            "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight",
            &t,
        )
        .expect("kohya mid_block attn should resolve under the complete table");
        assert_eq!(
            mid,
            "mid_block.attentions.0.transformer_blocks.0.attn1.to_q"
        );
        let (ff, _) = classify_key(
            "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_up.weight",
            &t,
        )
        .expect("kohya GEGLU ff.net.0.proj should resolve under the complete table");
        assert_eq!(
            ff,
            "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj"
        );
        // Those same stems are absent from the faithful table → None (proves the gate works by
        // table construction for kohya keys; PEFT keys are gated by `is_complete_only`).
        assert!(classify_key(
            "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight",
            &table()
        )
        .is_none());
    }

    #[test]
    fn classify_peft_resolves_dotted_path_with_default_infix() {
        let t = table();
        // PEFT keys carry the dotted diffusers path directly (with the peft `.default.` infix).
        let (path, role) = classify_key(
            "base_model.model.unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
            &t,
        )
        .unwrap();
        assert_eq!(
            path,
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"
        );
        assert!(matches!(role, Role::Down));
        // The bare `.weight` form (no `.default`) is also accepted.
        assert!(matches!(
            classify_key("base_model.model.unet.foo.bar.lora_B.weight", &t)
                .unwrap()
                .1,
            Role::Up
        ));
    }
}
