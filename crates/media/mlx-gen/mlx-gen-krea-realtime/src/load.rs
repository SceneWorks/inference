//! Krea Realtime 14B transformer **load path** (sc-8435 S2, non-gated).
//!
//! Krea Realtime 14B is Wan 2.1 T2V 14B weight-for-weight, so once
//! [`sanitize_krea_realtime_transformer`] has collapsed either on-disk layout to the internal Wan DiT
//! key names it loads straight into the reused [`mlx_gen_wan::WanTransformer`] via its
//! [`from_weights`](mlx_gen_wan::WanTransformer::from_weights). This module wraps that with an
//! **explicit completeness + shape check** ([`verify_transformer_tensors`]) so a truncated shard, a
//! stray extra tensor, or a wrong-shape weight fails loudly here — with a diff — instead of surfacing
//! as an opaque `require` error deep inside `from_weights` (or, worse, a silent mis-load).
//!
//! The expected internal tensor set + shapes are derived **purely from the config**
//! ([`expected_transformer_tensors`]), so the check is exact for any Wan geometry and needs no real
//! checkpoint. This is the non-gated S2 surface: `tests/` validate it against the S1 inventory with
//! synthesized fixtures; real-weight byte parity is the gated remainder on sc-8435.
//!
//! ## Quant tiers (sc-15203, S19)
//!
//! Krea Realtime ships three tiers — **bf16** (~28 GB), **Q8** (~14 GB) and **Q4** (~7 GB) — and the
//! quantized ones ship **pre-quantized (packed) on disk**, not quantized on the fly: the Wan
//! `_quantize_predicate` Linears (per-block self/cross-attention `q/k/v/o` + `ffn.fc1`/`fc2`) carry the
//! u32-code `{base}.weight` + `{base}.scales` + `{base}.biases` triple, and the reused
//! [`mlx_gen_wan::WanTransformer::from_weights`] builds them packed directly (its `load_linear` keys
//! off `.scales` presence, gated by [`WanModelConfig::quantization`]). The quant surface here is:
//!
//!   * [`expected_transformer_tensors`] emits the **packed** inventory for the predicate Linears when
//!     the config declares a tier, so [`verify_transformer_tensors`] stays exact on a packed snapshot
//!     instead of rejecting `.scales`/`.biases` as "extra" and the u32 codes as "wrong shape";
//!   * [`resolve_snapshot_quant`] / [`probe_packed_quant`] derive the tier **from the packed shapes**
//!     (`scales` is `[out, in/group_size]`, the u32 `weight` is `[out, in·bits/32]`, and `in` is known
//!     from the config geometry) — so both `bits` **and** `group_size` are recovered exactly and a
//!     snapshot with no `config.json` still loads at the right tier, while a `config.json` that
//!     *disagrees* with its own weights is a loud error rather than a silent mis-load (the sc-15154
//!     trap: an *assumed* group size turns a perfectly good artifact into an "illegal width");
//!   * [`resolve_load_time_quant`] reconciles a caller's [`LoadSpec::quantize`](mlx_gen::LoadSpec)
//!     against the tier actually on disk — a deliberately loud "stored wins".

use std::collections::{HashMap, HashSet};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Quant, Result};
use mlx_gen_wan::config::{WanModelConfig, WanQuant};
use mlx_gen_wan::WanTransformer;
use mlx_rs::Array;

use crate::config::KreaRealtimeConfig;
use crate::convert::sanitize_krea_realtime_transformer;

/// The reused Wan `_quantize_predicate` surface: the per-block Linears that ship **packed** on a
/// quantized tier (mirrors `mlx_gen_wan::convert`'s `WAN_QUANT_SUFFIXES` and the `Block::quantize`
/// surface). Everything else — the patch/text/time embeddings, `time_projection`, the modulation
/// tables, the qk/`norm3` norms and the output head — stays dense on every tier, exactly as the
/// reference predicate specifies.
///
/// This is the **source of truth** for the predicate surface, not a parallel description of it:
/// [`expected_transformer_tensors`] emits each block's Linears by iterating this list (so an entry
/// added/removed here moves the verified inventory), and `tests/quant_tiers.rs` cross-checks the list
/// against what the reused Wan packer actually packs.
pub const PACKED_LINEARS_PER_BLOCK: &[&str] = &[
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.o",
    "ffn.fc1",
    "ffn.fc2",
];

/// The Linear whose packed shapes [`resolve_snapshot_quant`] reads back to recover the tier. Present in
/// every Wan geometry with at least one block, always inside the quantize predicate, and always
/// `[dim, dim]` dense — so its `in` dimension is known from the config without trusting the file.
const QUANT_PROBE_LINEAR: &str = "blocks.0.self_attn.q";

/// The group sizes MLX's affine quantization actually implements (`mlx::core::quantize` accepts 32, 64
/// or 128). A `group_size` inferred from a snapshot's packed shapes is gated on this set for the same
/// reason the inferred `bits` is gated on `{4, 8}`: a `scales` tensor with, say, `dim` columns infers
/// `group_size = 1`, divides every predicate width happily, and would then fail opaquely deep inside
/// `from_quantized_parts`.
const SUPPORTED_GROUP_SIZES: [i32; 3] = [32, 64, 128];

/// One expected **internal** (post-[`sanitize_krea_realtime_transformer`]) transformer tensor: the key
/// [`mlx_gen_wan::WanTransformer::from_weights`] reads and the config-derived shape it must carry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    /// The internal tensor key (e.g. `blocks.0.self_attn.q.weight`, `patch_embedding_proj.weight`).
    pub name: String,
    /// The exact shape the tensor must have, derived from the [`WanModelConfig`].
    pub shape: Vec<i32>,
}

impl TensorSpec {
    fn new(name: impl Into<String>, shape: &[i32]) -> Self {
        Self {
            name: name.into(),
            shape: shape.to_vec(),
        }
    }
}

/// Push the tensors one biased `[out, in]` Linear contributes: the dense `{name}.weight` plus
/// `{name}.bias`, or — when `packed` is `Some` (this Linear is inside the quantize predicate on a
/// pre-quantized tier) — the MLX affine-quantized triple `{name}.weight` (u32 codes, at
/// `[out, in·bits/32]`), `{name}.scales` and `{name}.biases` (both `[out, in/group_size]`), plus the
/// still-dense `{name}.bias`. The dense bias is unaffected by quantization (MLX packs the weight
/// only), which is why a packed Linear contributes three weight tensors where a dense one contributes
/// one.
fn push_linear(
    specs: &mut Vec<TensorSpec>,
    name: String,
    out: i32,
    in_dim: i32,
    packed: Option<WanQuant>,
) {
    match packed {
        Some(q) => {
            specs.push(TensorSpec::new(
                format!("{name}.weight"),
                &[out, in_dim * q.bits / 32],
            ));
            let groups = in_dim / q.group_size;
            specs.push(TensorSpec::new(format!("{name}.scales"), &[out, groups]));
            specs.push(TensorSpec::new(format!("{name}.biases"), &[out, groups]));
        }
        None => specs.push(TensorSpec::new(format!("{name}.weight"), &[out, in_dim])),
    }
    specs.push(TensorSpec::new(format!("{name}.bias"), &[out]));
}

/// The `[out, in]` shape one [`PACKED_LINEARS_PER_BLOCK`] entry carries under this geometry. The eight
/// attention projections (`{self,cross}_attn.{q,k,v,o}`) are square `[dim, dim]`; only the FFN pair is
/// asymmetric — `fc1` widens `dim → ffn_dim` and `fc2` narrows back, so `fc2` packs over
/// `in = ffn_dim`, which is what makes an out-vs-in transposition in the packed inventory detectable.
/// (`packed_linear_list_is_the_attention_and_ffn_surface` pins the list this matches against, so a new
/// entry cannot silently fall through to the square case.)
fn predicate_linear_shape(suffix: &str, dim: i32, ffn: i32) -> (i32, i32) {
    match suffix {
        "ffn.fc1" => (ffn, dim),
        "ffn.fc2" => (dim, ffn),
        _ => (dim, dim),
    }
}

/// Every internal transformer tensor Krea Realtime's reused Wan DiT expects, with its config-derived
/// shape. This is the **post-sanitize** layout (`patch_embedding_proj`, `text_embedding_{0,1}`,
/// `time_projection`, `ffn.fc1`/`fc2`, `head.head`) — exactly what
/// [`mlx_gen_wan::WanTransformer::from_weights`] reads — not the native on-disk names.
///
/// Shapes for the shipped `wan21_t2v_14b` geometry (`dim=5120`, `ffn=13824`, `text_dim=4096`,
/// `freq_dim=256`, `in/out=16`, `patch=(1,2,2)`, `40` layers): `self_attn.q [5120,5120]`,
/// `ffn.fc1 [13824,5120]`, `patch_embedding_proj [5120,64]`, `head.head [64,5120]`,
/// `time_projection [30720,5120]`, `text_embedding_0 [5120,4096]`.
///
/// **Quant tiers (sc-15203).** When `cfg.quantization` declares a pre-quantized tier, the
/// [`PACKED_LINEARS_PER_BLOCK`] surface switches to the packed triple (u32 codes + `.scales` +
/// `.biases`) at that `bits`/`group_size` — so a Q4/Q8 snapshot verifies exactly. The whole-model
/// embeddings / `time_projection` / head stay dense on every tier (they are outside the reference
/// `_quantize_predicate`, and `WanTransformer::from_weights` loads them with `quant = None`
/// unconditionally), so this table and the loader cannot drift.
pub fn expected_transformer_tensors(cfg: &WanModelConfig) -> Vec<TensorSpec> {
    let dim = cfg.dim as i32;
    let ffn = cfg.ffn_dim as i32;
    let text_dim = cfg.text_dim as i32;
    let freq = cfg.freq_dim as i32;
    let (pt, ph, pw) = cfg.patch_size;
    // The patch-embed conv `[dim, in, pt, ph, pw]` flattens to a Linear `[dim, in·∏patch]`; the head
    // projects `dim → out·∏patch`.
    let patch_cols = (cfg.in_dim * pt * ph * pw) as i32;
    let head_out = (cfg.out_dim * pt * ph * pw) as i32;

    let mut specs = Vec::with_capacity(15 + cfg.num_layers * 27);

    // Patch embedding (conv→Linear reshape).
    specs.push(TensorSpec::new(
        "patch_embedding_proj.weight",
        &[dim, patch_cols],
    ));
    specs.push(TensorSpec::new("patch_embedding_proj.bias", &[dim]));

    // Text embedding Sequential: Linear(text_dim→dim), GELU, Linear(dim→dim).
    specs.push(TensorSpec::new("text_embedding_0.weight", &[dim, text_dim]));
    specs.push(TensorSpec::new("text_embedding_0.bias", &[dim]));
    specs.push(TensorSpec::new("text_embedding_1.weight", &[dim, dim]));
    specs.push(TensorSpec::new("text_embedding_1.bias", &[dim]));

    // Time embedding Sequential: Linear(freq_dim→dim), SiLU, Linear(dim→dim).
    specs.push(TensorSpec::new("time_embedding_0.weight", &[dim, freq]));
    specs.push(TensorSpec::new("time_embedding_0.bias", &[dim]));
    specs.push(TensorSpec::new("time_embedding_1.weight", &[dim, dim]));
    specs.push(TensorSpec::new("time_embedding_1.bias", &[dim]));

    // Time projection: Linear(dim→6·dim) (the six modulation vectors).
    specs.push(TensorSpec::new("time_projection.weight", &[6 * dim, dim]));
    specs.push(TensorSpec::new("time_projection.bias", &[6 * dim]));

    // Output head: modulated LayerNorm table + projection dim→out·∏patch.
    specs.push(TensorSpec::new("head.modulation", &[1, 2, dim]));
    specs.push(TensorSpec::new("head.head.weight", &[head_out, dim]));
    specs.push(TensorSpec::new("head.head.bias", &[head_out]));

    // The per-block attention + FFN Linears are the reference `_quantize_predicate` surface: packed on
    // a pre-quantized tier, dense on bf16. Emitted by iterating `PACKED_LINEARS_PER_BLOCK`, so that
    // constant *is* the predicate surface this inventory verifies rather than a parallel description of
    // it (`self_attn/cross_attn.{q,k,v,o}` + `ffn.fc1`/`fc2`; the FFN is Linear(dim→ffn), GELU,
    // Linear(ffn→dim)).
    let q = cfg.quantization;
    for i in 0..cfg.num_layers {
        let p = format!("blocks.{i}");
        // Per-block 6-vector modulation table.
        specs.push(TensorSpec::new(format!("{p}.modulation"), &[1, 6, dim]));
        for lin in PACKED_LINEARS_PER_BLOCK {
            let (out, in_dim) = predicate_linear_shape(lin, dim, ffn);
            push_linear(&mut specs, format!("{p}.{lin}"), out, in_dim, q);
        }
        // The full-dim qk-RMSNorm weights and the cross-attention pre-norm (affine LayerNorm) are
        // outside the quantize predicate and stay dense on every tier.
        for attn in ["self_attn", "cross_attn"] {
            specs.push(TensorSpec::new(format!("{p}.{attn}.norm_q.weight"), &[dim]));
            specs.push(TensorSpec::new(format!("{p}.{attn}.norm_k.weight"), &[dim]));
        }
        specs.push(TensorSpec::new(format!("{p}.norm3.weight"), &[dim]));
        specs.push(TensorSpec::new(format!("{p}.norm3.bias"), &[dim]));
    }

    specs
}

/// Cap a long list in an error message so a fully-missing map does not print thousands of lines.
fn preview(items: &[String]) -> String {
    const MAX: usize = 12;
    if items.len() <= MAX {
        items.join(", ")
    } else {
        format!(
            "{} … (+{} more)",
            items[..MAX].join(", "),
            items.len() - MAX
        )
    }
}

/// Assert `map` (an already-[`sanitize_krea_realtime_transformer`]d internal weight map) contains
/// **exactly** the tensors [`expected_transformer_tensors`] derives from `cfg`, each at its exact
/// shape — no missing, no extra, no wrong shape. On any discrepancy returns a single [`Error::Msg`]
/// summarizing the (capped) missing / extra / mis-shaped keys. Shape checks read only tensor metadata,
/// so this never forces MLX to materialize the (lazy) buffers.
pub fn verify_transformer_tensors(
    map: &HashMap<String, Array>,
    cfg: &WanModelConfig,
) -> Result<()> {
    let expected = expected_transformer_tensors(cfg);
    let expected_names: HashSet<&str> = expected.iter().map(|s| s.name.as_str()).collect();

    let mut missing = Vec::new();
    let mut mis_shape = Vec::new();
    for spec in &expected {
        match map.get(&spec.name) {
            None => missing.push(spec.name.clone()),
            Some(tensor) => {
                if tensor.shape() != spec.shape.as_slice() {
                    mis_shape.push(format!(
                        "{} (want {:?}, got {:?})",
                        spec.name,
                        spec.shape,
                        tensor.shape()
                    ));
                }
            }
        }
    }

    let mut extra: Vec<String> = map
        .keys()
        .filter(|k| !expected_names.contains(k.as_str()))
        .cloned()
        .collect();

    if missing.is_empty() && extra.is_empty() && mis_shape.is_empty() {
        return Ok(());
    }

    missing.sort();
    extra.sort();
    mis_shape.sort();
    Err(Error::Msg(format!(
        "krea-realtime: transformer tensor set does not match the wan21_t2v_14b inventory \
         (expected {} tensors): {} missing [{}], {} extra [{}], {} wrong-shape [{}]",
        expected.len(),
        missing.len(),
        preview(&missing),
        extra.len(),
        preview(&extra),
        mis_shape.len(),
        preview(&mis_shape),
    )))
}

/// Recover the pre-quantized tier a Krea Realtime transformer weight **map** ships at, or `None` for a
/// dense bf16 snapshot. The `HashMap` entry point, used after [`sanitize_krea_realtime_transformer`];
/// [`probe_packed_quant`] is the [`Weights`]-flavoured sibling.
///
/// The tier is derived from the packed **shapes**, not trusted from a manifest: MLX affine quantization
/// stores `scales` as `[out, in/group_size]` and the u32 codes as `[out, in·bits/32]`, and the probe
/// Linear's `in` is `cfg.dim` by construction — so `group_size = in/scales.cols` and
/// `bits = weight.cols·32/in` are both **exact**. That closes the sc-15154 trap where *assuming* a group
/// size makes a good artifact report an illegal bit-width.
///
/// [`WanModelConfig::quantization`] (the snapshot's `config.json` block, when it has one) is then
/// cross-checked against what the weights actually are; a disagreement in either direction is a hard
/// error, never a silent mis-load:
///
///   * manifest declares a tier but no Linear is packed ⇒ the weights are dense (or the wrong file),
///   * manifest declares `bits`/`group_size` different from the packed shapes ⇒ one of them is stale.
pub fn resolve_snapshot_quant(
    map: &HashMap<String, Array>,
    cfg: &WanModelConfig,
) -> Result<Option<WanQuant>> {
    packed_quant_from(
        map.keys().any(|k| k.ends_with(".scales")),
        |k| map.get(k),
        cfg,
    )
}

// The shared core of `resolve_snapshot_quant` / `probe_packed_quant`, over a key→tensor lookup: derive
// the tier from the packed shapes and cross-check it against any `config.json` manifest. See
// `resolve_snapshot_quant`'s docs for the full contract.
fn packed_quant_from<'a>(
    any_packed: bool,
    get: impl Fn(&str) -> Option<&'a Array>,
    cfg: &WanModelConfig,
) -> Result<Option<WanQuant>> {
    let declared = cfg.quantization;
    if !any_packed {
        if let Some(q) = declared {
            return Err(Error::Msg(format!(
                "krea-realtime: config.json declares a pre-quantized Q{} tier (group {}), but the \
                 transformer carries no packed weights (`.scales`) — the manifest and the weights \
                 disagree. Point at the matching packed snapshot, or drop the `quantization` block \
                 from a dense bf16 snapshot's config.json",
                q.bits, q.group_size
            )));
        }
        return Ok(None);
    }

    let scales = get(&format!("{QUANT_PROBE_LINEAR}.scales")).ok_or_else(|| {
        Error::Msg(format!(
            "krea-realtime: the transformer carries packed weights but `{QUANT_PROBE_LINEAR}.scales` \
             is missing — a partially-packed snapshot cannot be loaded (every quantize-predicate \
             Linear must be packed at one tier)"
        ))
    })?;
    let wq = get(&format!("{QUANT_PROBE_LINEAR}.weight")).ok_or_else(|| {
        Error::Msg(format!(
            "krea-realtime: `{QUANT_PROBE_LINEAR}.weight` is missing from a packed transformer"
        ))
    })?;

    // `blocks.0.self_attn.q` is `[dim, dim]` dense, so `in` is known from the geometry.
    let in_dim = cfg.dim as i32;
    let (s_shape, w_shape) = (scales.shape(), wq.shape());
    if s_shape.len() != 2 || w_shape.len() != 2 {
        return Err(Error::Msg(format!(
            "krea-realtime: packed `{QUANT_PROBE_LINEAR}` must be 2-D, got weight {w_shape:?} / \
             scales {s_shape:?}"
        )));
    }
    if s_shape[1] <= 0 || in_dim % s_shape[1] != 0 {
        return Err(Error::Msg(format!(
            "krea-realtime: packed `{QUANT_PROBE_LINEAR}.scales` has {} group column(s), which does \
             not divide the config's input width {in_dim} — the snapshot geometry does not match \
             this config",
            s_shape[1]
        )));
    }
    let group_size = in_dim / s_shape[1];
    if !SUPPORTED_GROUP_SIZES.contains(&group_size) {
        return Err(Error::Msg(format!(
            "krea-realtime: inferred packed group_size {group_size} ∉ {SUPPORTED_GROUP_SIZES:?} (the \
             group sizes MLX affine quantization implements) — from `{QUANT_PROBE_LINEAR}.scales` \
             cols {} over the config's input width {in_dim}; the snapshot is corrupt or was packed \
             for a different geometry",
            s_shape[1]
        )));
    }
    if w_shape[1] <= 0 || (w_shape[1] * 32) % in_dim != 0 {
        return Err(Error::Msg(format!(
            "krea-realtime: packed `{QUANT_PROBE_LINEAR}.weight` has {} u32 column(s), which is not a \
             whole bit-width over the config's input width {in_dim}",
            w_shape[1]
        )));
    }
    let bits = w_shape[1] * 32 / in_dim;
    if !matches!(bits, 4 | 8) {
        return Err(Error::Msg(format!(
            "krea-realtime: inferred packed bit-width {bits} ∉ {{4, 8}} (weight cols {}, scales cols \
             {} ⇒ group_size {group_size}, input width {in_dim}) — the snapshot is corrupt or was \
             packed for a different geometry",
            w_shape[1], s_shape[1]
        )));
    }
    // Every packed Linear in the predicate must be group-aligned at this width, or `from_weights`
    // would build a mis-shaped base further down.
    for width in [in_dim, cfg.ffn_dim as i32] {
        if width % group_size != 0 {
            return Err(Error::Msg(format!(
                "krea-realtime: inferred quantization group_size {group_size} does not divide the \
                 input width {width} of every quantize-predicate Linear (dim {in_dim}, ffn_dim {})",
                cfg.ffn_dim
            )));
        }
    }

    let found = WanQuant { bits, group_size };
    if let Some(q) = declared {
        if q != found {
            return Err(Error::Msg(format!(
                "krea-realtime: config.json declares Q{} at group {}, but the packed weights are Q{} \
                 at group {} — the manifest is stale relative to its own tensors; re-convert the \
                 snapshot or fix the `quantization` block",
                q.bits, q.group_size, found.bits, found.group_size
            )));
        }
    }
    Ok(Some(found))
}

/// Recover the pre-quantized tier a Krea Realtime transformer [`Weights`] handle ships at, or `None` for
/// a dense bf16 snapshot — the same contract as [`resolve_snapshot_quant`], over a `Weights` handle.
///
/// Reads **shape metadata only** (MLX safetensors loads are lazy, so nothing is materialized), which
/// makes it a free probe usable *before* the DiT is built — what the pipeline needs, since the UMT5 Q8
/// floor (sc-12831) depends on the DiT tier but the text encoder is staged first.
pub fn probe_packed_quant(w: &Weights, cfg: &WanModelConfig) -> Result<Option<WanQuant>> {
    packed_quant_from(w.keys().any(|k| k.ends_with(".scales")), |k| w.get(k), cfg)
}

/// Reconcile a caller's requested [`LoadSpec::quantize`](mlx_gen::LoadSpec) against the tier the
/// snapshot **actually** ships at (`packed`, from [`probe_packed_quant`]), returning the load-time
/// quantization to apply after the DiT is built — `None` when nothing further is needed.
///
/// Mirrors the sibling Wan providers' rule (`mlx_gen_wan::model`), with one hardening: it reconciles
/// against the tier read back from the *weights*, not from a `config.json` that may be absent or stale.
///
///   * **Pre-quantized snapshot** ⇒ `from_weights` already built it packed, so no load-time requant
///     (`None`). A request at a *different* width is a hard error — "stored wins", loudly, because
///     `AdaptableLinear::quantize` no-ops over packed weights and would otherwise silently serve the
///     stored tier while the caller believed it got the requested one.
///   * **Dense bf16 snapshot** ⇒ honor the request (quantize in memory after load).
///   * [`Quant::Nvfp4`] is a candle-only tier with no MLX affine equivalent; routing it through
///     `quantize(bits)` on its `bits() == 4` alone would silently serve Q4, so it is rejected as a
///     typed [`Error::Unsupported`] (the advertised `supported_quants` is `[Q4, Q8]`).
pub fn resolve_load_time_quant(
    model_id: &str,
    packed: Option<WanQuant>,
    requested: Option<Quant>,
) -> Result<Option<Quant>> {
    if requested == Some(Quant::Nvfp4) {
        return Err(Error::Unsupported(format!(
            "{model_id}: the NVFP4 tier is candle/CUDA-only and has no MLX affine equivalent — this \
             MLX engine offers Q4 / Q8 / bf16"
        )));
    }
    match (packed, requested) {
        (Some(stored), Some(req)) if stored.bits != req.bits() => Err(Error::Msg(format!(
            "{model_id}: this snapshot is pre-quantized Q{} (packed on disk), but a Q{} load was \
             requested — quantize is a no-op over packed weights, so the request would silently \
             serve Q{}. Point at the Q{} snapshot (or a dense bf16 one) instead",
            stored.bits,
            req.bits(),
            stored.bits,
            req.bits()
        ))),
        // Pre-quantized: `from_weights` already built it packed; no load-time requant.
        (Some(_), _) => Ok(None),
        // Dense bf16: honor the request.
        (None, req) => Ok(req),
    }
}

/// Load a native Krea Realtime 14B transformer weight map (either on-disk layout — single-file
/// `model.`-prefixed or sharded `transformer/` bare) into the reused [`mlx_gen_wan::WanTransformer`].
///
/// The pipeline is: [`sanitize_krea_realtime_transformer`] (normalize the layout, map onto the
/// internal Wan DiT names, cast the float tensors F16 → bf16) → [`resolve_snapshot_quant`] (recover the
/// on-disk tier from the packed shapes) → [`verify_transformer_tensors`] (assert the full,
/// exactly-shaped inventory for *that* tier is present) → [`mlx_gen_wan::WanTransformer::from_weights`],
/// which builds the quantize-predicate Linears packed when the resolved tier says so. The TE / VAE /
/// tokenizer are stock Wan and provisioned separately (Krea Realtime ships transformer-only), so this
/// loads the DiT only.
pub fn load_krea_realtime_transformer(
    raw: HashMap<String, Array>,
    cfg: &KreaRealtimeConfig,
) -> Result<WanTransformer> {
    Ok(load_krea_realtime_transformer_with_quant(raw, cfg)?.0)
}

/// [`load_krea_realtime_transformer`] plus the pre-quantized tier the snapshot turned out to ship at
/// (`None` = dense bf16) — the value the caller feeds [`resolve_load_time_quant`] to decide whether a
/// requested `LoadSpec::quantize` still has anything to do (sc-15203, S19).
///
/// The resolved tier is written onto a *copy* of the config before the DiT is built, so a packed
/// snapshot whose `config.json` is missing (or which has no `config.json` at all — Krea Realtime ships
/// transformer-only and the crate falls back to the shipped preset) still loads packed rather than
/// failing verification with thousands of "extra `.scales`" lines.
pub fn load_krea_realtime_transformer_with_quant(
    raw: HashMap<String, Array>,
    cfg: &KreaRealtimeConfig,
) -> Result<(WanTransformer, Option<WanQuant>)> {
    let sanitized = sanitize_krea_realtime_transformer(raw)?;
    let quant = resolve_snapshot_quant(&sanitized, &cfg.wan)?;
    let mut wan = cfg.wan.clone();
    wan.quantization = quant;
    verify_transformer_tensors(&sanitized, &wan)?;
    let weights = Weights::from_map(sanitized);
    Ok((WanTransformer::from_weights(&weights, &wan)?, quant))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_has_1095_tensors_at_audit_shapes() {
        let cfg = WanModelConfig::wan21_t2v_14b();
        let specs = expected_transformer_tensors(&cfg);
        // S1 audit: 1095 parameter tensors (the `freqs` RoPE buffer is dropped on sanitize).
        assert_eq!(specs.len(), 1095, "Wan2.1-14B transformer parameter count");

        let by_name: HashMap<&str, &[i32]> = specs
            .iter()
            .map(|s| (s.name.as_str(), s.shape.as_slice()))
            .collect();
        // The representative shapes from the S1 audit.
        assert_eq!(by_name["blocks.0.self_attn.q.weight"], &[5120, 5120]);
        assert_eq!(by_name["blocks.0.ffn.fc1.weight"], &[13824, 5120]);
        assert_eq!(by_name["patch_embedding_proj.weight"], &[5120, 64]);
        assert_eq!(by_name["head.head.weight"], &[64, 5120]);
        assert_eq!(by_name["time_projection.weight"], &[30720, 5120]);
        assert_eq!(by_name["text_embedding_0.weight"], &[5120, 4096]);
        assert_eq!(by_name["blocks.39.cross_attn.o.weight"], &[5120, 5120]);
    }

    #[test]
    fn inventory_scales_with_layer_count() {
        // Discriminating: the per-layer block (27 tensors) must actually be emitted per layer.
        let mut two = WanModelConfig::wan21_t2v_14b();
        two.num_layers = 2;
        let mut three = WanModelConfig::wan21_t2v_14b();
        three.num_layers = 3;
        let d =
            expected_transformer_tensors(&three).len() - expected_transformer_tensors(&two).len();
        assert_eq!(d, 27, "each transformer block contributes 27 tensors");
    }

    // ── Quant tiers (sc-15203, S19) ─────────────────────────────────────────────────────────────

    fn q4_cfg() -> WanModelConfig {
        let mut c = WanModelConfig::wan21_t2v_14b();
        c.quantization = Some(WanQuant {
            bits: 4,
            group_size: 64,
        });
        c
    }

    /// The packed inventory is the MLX affine layout at the declared width — hand-derived numeric
    /// literals, not the code's own formula re-run. `self_attn.q` is `[5120, 5120]` dense; at Q4/group-64
    /// the u32 codes are `[5120, 5120·4/32 = 640]` and both `scales`/`biases` are `[5120, 5120/64 = 80]`.
    /// The FFN's asymmetric widths (`fc1 [13824, 5120]`, `fc2 [5120, 13824]`) discriminate an
    /// out-vs-in transposition: `fc2` packs over `in = 13824` ⇒ `[5120, 1728]` codes / `[5120, 216]` groups.
    #[test]
    fn packed_inventory_uses_mlx_affine_shapes_at_the_declared_width() {
        let specs = expected_transformer_tensors(&q4_cfg());
        let by_name: HashMap<&str, &[i32]> = specs
            .iter()
            .map(|s| (s.name.as_str(), s.shape.as_slice()))
            .collect();

        assert_eq!(by_name["blocks.0.self_attn.q.weight"], &[5120, 640]);
        assert_eq!(by_name["blocks.0.self_attn.q.scales"], &[5120, 80]);
        assert_eq!(by_name["blocks.0.self_attn.q.biases"], &[5120, 80]);
        // The Linear's own dense bias is NOT quantized (MLX packs the weight only).
        assert_eq!(by_name["blocks.0.self_attn.q.bias"], &[5120]);

        assert_eq!(by_name["blocks.39.ffn.fc1.weight"], &[13824, 640]);
        assert_eq!(by_name["blocks.39.ffn.fc1.scales"], &[13824, 80]);
        assert_eq!(by_name["blocks.39.ffn.fc2.weight"], &[5120, 1728]);
        assert_eq!(by_name["blocks.39.ffn.fc2.scales"], &[5120, 216]);

        // Q8 is twice the code columns of Q4 at the same group size (the width genuinely propagates).
        let mut q8 = q4_cfg();
        q8.quantization = Some(WanQuant {
            bits: 8,
            group_size: 64,
        });
        let q8_specs = expected_transformer_tensors(&q8);
        let q8_by: HashMap<&str, &[i32]> = q8_specs
            .iter()
            .map(|s| (s.name.as_str(), s.shape.as_slice()))
            .collect();
        assert_eq!(q8_by["blocks.0.self_attn.q.weight"], &[5120, 1280]);
        // …and the group columns are unchanged by the bit width, only by the group size.
        assert_eq!(q8_by["blocks.0.self_attn.q.scales"], &[5120, 80]);
    }

    /// Only the reference `_quantize_predicate` surface packs: the 10 per-block attention/FFN Linears
    /// gain two tensors each (`.scales` + `.biases`) and nothing else moves. 1095 + 40·10·2 = 1895.
    /// The whole-model embeddings / `time_projection` / head stay **dense** on a packed tier — the
    /// discriminating half, since `WanTransformer::from_weights` loads them with `quant = None`
    /// unconditionally, so packing them here would make every Q4 snapshot fail verification.
    #[test]
    fn packed_inventory_covers_only_the_quantize_predicate() {
        let dense = expected_transformer_tensors(&WanModelConfig::wan21_t2v_14b());
        let packed = expected_transformer_tensors(&q4_cfg());
        assert_eq!(dense.len(), 1095);
        assert_eq!(packed.len(), 1895);
        assert_eq!(packed.len() - dense.len(), 40 * 10 * 2);

        let names: HashSet<&str> = packed.iter().map(|s| s.name.as_str()).collect();
        for lin in PACKED_LINEARS_PER_BLOCK {
            assert!(
                names.contains(format!("blocks.7.{lin}.scales").as_str()),
                "`blocks.7.{lin}` must be packed on a quantized tier"
            );
        }
        for dense_only in [
            "patch_embedding_proj",
            "text_embedding_0",
            "text_embedding_1",
            "time_embedding_0",
            "time_embedding_1",
            "time_projection",
            "head.head",
        ] {
            assert!(
                !names.contains(format!("{dense_only}.scales").as_str()),
                "`{dense_only}` is outside the quantize predicate and must stay dense"
            );
        }
        // The qk-RMSNorm / norm3 gains are not Linears and never pack.
        assert!(!names.contains("blocks.0.self_attn.norm_q.scales"));
        assert!(!names.contains("blocks.0.norm3.scales"));
    }

    /// A tiny synthetic packed probe pair: `scales [out, in/gs]` + u32 `weight [out, in·bits/32]`.
    fn packed_probe(map: &mut HashMap<String, Array>, out: i32, in_dim: i32, bits: i32, gs: i32) {
        let codes = in_dim * bits / 32;
        map.insert(
            format!("{QUANT_PROBE_LINEAR}.weight"),
            Array::zeros::<u32>(&[out, codes]).unwrap(),
        );
        map.insert(
            format!("{QUANT_PROBE_LINEAR}.scales"),
            Array::zeros::<f32>(&[out, in_dim / gs]).unwrap(),
        );
    }

    fn tiny_wan(dim: usize, ffn: usize) -> WanModelConfig {
        let mut c = WanModelConfig::wan21_t2v_14b();
        c.dim = dim;
        c.ffn_dim = ffn;
        c
    }

    /// The tier is recovered from the packed **shapes**, recovering `bits` AND `group_size` — the
    /// discriminating property vs. assuming a group size (sc-15154). Two different group sizes at the
    /// same bit width, and two different bit widths at the same group size, all resolve exactly.
    #[test]
    fn snapshot_quant_is_derived_from_the_packed_shapes() {
        let cfg = tiny_wan(256, 512);
        for (bits, gs) in [(4, 64), (8, 64), (4, 32), (8, 128)] {
            let mut map = HashMap::new();
            packed_probe(&mut map, 256, 256, bits, gs);
            assert_eq!(
                resolve_snapshot_quant(&map, &cfg).unwrap(),
                Some(WanQuant {
                    bits,
                    group_size: gs
                }),
                "Q{bits} at group {gs} must round-trip out of the packed shapes"
            );
        }
        // A dense map (no `.scales` anywhere) is the bf16 tier.
        assert_eq!(resolve_snapshot_quant(&HashMap::new(), &cfg).unwrap(), None);
    }

    /// A `config.json` that disagrees with its own tensors is a hard error in BOTH directions — never a
    /// silent mis-load. (Wan's loader keys packed-vs-dense off `.scales` presence per Linear, so a stale
    /// manifest would otherwise load a Q8 file at Q4's group scales, or a dense file "as packed".)
    #[test]
    fn manifest_disagreeing_with_the_weights_is_a_hard_error() {
        let mut cfg = tiny_wan(256, 512);

        // (a) manifest declares a tier, weights are dense.
        cfg.quantization = Some(WanQuant {
            bits: 4,
            group_size: 64,
        });
        let err = resolve_snapshot_quant(&HashMap::new(), &cfg)
            .expect_err("dense weights under a quantized manifest must fail");
        assert!(err.to_string().contains("no packed weights"), "got: {err}");

        // (b) manifest declares Q4, weights are packed Q8.
        let mut map = HashMap::new();
        packed_probe(&mut map, 256, 256, 8, 64);
        let err = resolve_snapshot_quant(&map, &cfg)
            .expect_err("a stale manifest width must fail, not be silently overridden");
        let msg = err.to_string();
        assert!(
            msg.contains("declares Q4") && msg.contains("are Q8"),
            "got: {msg}"
        );

        // (c) manifest declares Q8/group-32, weights are packed Q8/group-64 — the group size is checked
        // too, not just the bit width.
        cfg.quantization = Some(WanQuant {
            bits: 8,
            group_size: 32,
        });
        let err = resolve_snapshot_quant(&map, &cfg)
            .expect_err("a stale group size must fail as loudly as a stale width");
        assert!(err.to_string().contains("group"), "got: {err}");

        // (d) matching manifest passes.
        cfg.quantization = Some(WanQuant {
            bits: 8,
            group_size: 64,
        });
        assert_eq!(
            resolve_snapshot_quant(&map, &cfg).unwrap(),
            Some(WanQuant {
                bits: 8,
                group_size: 64
            })
        );
    }

    /// The inferred `group_size` is gated on MLX's implemented set, exactly as the sibling `bits` is
    /// gated on `{4, 8}` — otherwise a `scales` with `dim` columns infers `group_size = 1`, divides
    /// every predicate width, passes every other guard, and then fails opaquely inside MLX at
    /// `from_quantized_parts`. Discriminating: `group_size = 1` and `group_size = 16` are BOTH exact
    /// divisors of this geometry's `dim`/`ffn_dim`, so only the explicit set membership rejects them,
    /// while the neighbouring legal sizes still resolve.
    #[test]
    fn group_size_must_be_one_mlx_implements() {
        let cfg = tiny_wan(256, 512);
        // 1 (one scale per input element), 16 and 256 all divide dim 256 AND ffn_dim 512 exactly, so
        // every *other* guard passes them; only the set membership rejects them.
        for bad_gs in [1, 16, 256] {
            let mut map = HashMap::new();
            packed_probe(&mut map, 256, 256, 4, bad_gs);
            let err = resolve_snapshot_quant(&map, &cfg).expect_err(&format!(
                "group_size {bad_gs} is not an MLX group size and must be rejected"
            ));
            let msg = err.to_string();
            assert!(
                msg.contains(&format!("group_size {bad_gs}")) && msg.contains("corrupt"),
                "got: {msg}"
            );
        }
        // …and the three MLX-implemented sizes still resolve (the guard is a bound, not a blanket).
        for gs in [32, 64, 128] {
            let mut map = HashMap::new();
            packed_probe(&mut map, 256, 256, 8, gs);
            assert_eq!(
                resolve_snapshot_quant(&map, &cfg).unwrap(),
                Some(WanQuant {
                    bits: 8,
                    group_size: gs
                })
            );
        }
    }

    /// [`predicate_linear_shape`]'s square-`[dim, dim]` fallback is only safe because the list it
    /// matches against is exactly the eight attention projections plus the two asymmetric FFN Linears.
    /// Pinning that here means adding an entry with a different geometry to [`PACKED_LINEARS_PER_BLOCK`]
    /// (say a fused `qkv`) goes red rather than silently emitting a square shape for it.
    #[test]
    fn packed_linear_list_is_the_attention_and_ffn_surface() {
        let mut expected: Vec<String> = Vec::new();
        for attn in ["self_attn", "cross_attn"] {
            for proj in ["q", "k", "v", "o"] {
                expected.push(format!("{attn}.{proj}"));
            }
        }
        expected.push("ffn.fc1".into());
        expected.push("ffn.fc2".into());
        let mut got: Vec<String> = PACKED_LINEARS_PER_BLOCK
            .iter()
            .map(|s| s.to_string())
            .collect();
        got.sort();
        expected.sort();
        assert_eq!(got, expected);
        // The FFN pair is the asymmetric one, and `fc2` packs over `ffn_dim` (an out↔in transposition
        // here would make every packed FFN inventory wrong).
        assert_eq!(predicate_linear_shape("ffn.fc1", 64, 128), (128, 64));
        assert_eq!(predicate_linear_shape("ffn.fc2", 64, 128), (64, 128));
        assert_eq!(predicate_linear_shape("cross_attn.v", 64, 128), (64, 64));
    }

    /// A packed snapshot whose group size does not divide *every* predicate Linear's input width is
    /// rejected up front (the FFN packs over `ffn_dim`, not `dim`), instead of building a mis-shaped
    /// base deep inside `from_weights`.
    #[test]
    fn group_size_must_divide_every_predicate_input_width() {
        // dim 256 (divisible by 64), ffn_dim 100 (not) — the probe alone would happily say Q4/group-64.
        let cfg = tiny_wan(256, 100);
        let mut map = HashMap::new();
        packed_probe(&mut map, 256, 256, 4, 64);
        let err = resolve_snapshot_quant(&map, &cfg)
            .expect_err("a group size that misses ffn_dim must be rejected");
        assert!(err.to_string().contains("ffn_dim"), "got: {err}");
    }

    /// Load-time quant reconciles against the tier the weights ACTUALLY carry — "stored wins", loudly.
    /// The discriminating case is (packed Q8, requested Q4): `AdaptableLinear::quantize` no-ops over
    /// packed weights, so without this error the caller would silently be served Q8.
    #[test]
    fn load_time_quant_reconciles_against_the_stored_tier() {
        let q8 = Some(WanQuant {
            bits: 8,
            group_size: 64,
        });
        // Dense snapshot: the request is honored verbatim.
        assert_eq!(
            resolve_load_time_quant("krea_realtime_14b", None, Some(Quant::Q4)).unwrap(),
            Some(Quant::Q4)
        );
        assert_eq!(
            resolve_load_time_quant("krea_realtime_14b", None, None).unwrap(),
            None
        );
        // Packed snapshot at the SAME width: nothing further to do (already built packed).
        assert_eq!(
            resolve_load_time_quant("krea_realtime_14b", q8, Some(Quant::Q8)).unwrap(),
            None
        );
        // Packed snapshot with no request: likewise a no-op.
        assert_eq!(
            resolve_load_time_quant("krea_realtime_14b", q8, None).unwrap(),
            None
        );
        // Packed snapshot at a DIFFERENT width: hard error rather than a silent downgrade.
        let err = resolve_load_time_quant("krea_realtime_14b", q8, Some(Quant::Q4))
            .expect_err("Q4 requested over a packed Q8 snapshot must fail");
        assert!(err.to_string().contains("silently serve Q8"), "got: {err}");
        // NVFP4 is candle-only: rejected as a typed capability gap, never routed through `quantize(4)`.
        let err = resolve_load_time_quant("krea_realtime_14b", None, Some(Quant::Nvfp4))
            .expect_err("NVFP4 has no MLX affine equivalent");
        assert!(matches!(err, Error::Unsupported(_)), "got: {err:?}");
    }
}
