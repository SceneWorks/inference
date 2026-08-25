//! Community **single-file** Krea 2 DiT → in-memory diffusers-key remap (epic 14015, sc-14017 S0b).
//!
//! A ComfyUI-exported single-file Krea 2 checkpoint (e.g. `kreamania_variant5.safetensors`, a dense
//! bf16 merge) stores the DiT under **native-mmdit** tensor names beneath the
//! `model.diffusion_model.` prefix — `blocks.N.attn.{wq,wk,wv,wo,gate}`, `blocks.N.mlp.{gate,up,down}`,
//! `blocks.N.{prenorm,postnorm}.scale`, `blocks.N.attn.qknorm.{qnorm,knorm}.scale`, `blocks.N.mod.lin`,
//! the `txtfusion.{layerwise_blocks,refiner_blocks}.*` text-fusion stacks + `txtfusion.projector`,
//! `txtmlp.*`, `tmlp.*`, `tproj.*`, `first.*`, and `last.{linear,modulation,norm}.*`.
//!
//! The MLX [`crate::transformer::Krea2Transformer`] loads the **diffusers** key schema
//! (`transformer_blocks.N.attn.to_q`, `img_in`, `txt_in`, `time_embed`, `time_mod_proj`, `text_fusion`,
//! `final_layer.*`) — identity-keyed against the published `krea/Krea-2-Turbo` snapshot. So this module
//! renames every native key to its diffusers counterpart in memory, producing a [`Weights`] the existing
//! [`Krea2Transformer::from_weights`](crate::transformer::Krea2Transformer::from_weights) drops straight
//! into. Tensor **values and dtypes pass through untouched** — the community merge stores the norm/
//! modulation scales as bf16 (the published turnkey stores them f32; both upcast to f32 in the norm/
//! modulation forward), so a verbatim load is the faithful one.
//!
//! # Remap source
//!
//! The native↔diffusers correspondence is the **inverse** of candle's authoritative
//! `convrot_diffusers_to_native` in
//! `crates/media/candle-gen/candle-gen-krea/src/loader.rs` (sc-9300) — the map validated exhaustively
//! against the real native-mmdit header. It is replicated here (rather than shared) because that
//! function lives in the candle backend tree, is bolted to the INT8-ConvRot loader (int8 codes +
//! regular-Hadamard rotation), and runs diffusers→native; this MLX path wants the **pure key mapping
//! only** (no int8, no rotation) in the native→diffusers direction. Keep the two in lockstep: an edit to
//! the candle correspondence must be mirrored here.
//!
//! # Fail-closed
//!
//! [`remap_native_dit_to_diffusers`] fails closed (typed [`Error`], never a silent skip) on **any**
//! on-disk key it cannot map and on **any** two keys that would collide onto one diffusers name. The
//! complementary "every module weight the transformer needs is present" coverage + shape check is
//! [`crate::convert::validate_transformer`], run by the single-file loader after this remap.
//!
//! # Shape normalization
//!
//! The remap is a pure **key** rename. One diffusers-vs-native **shape** difference is normalized
//! separately by [`normalize_modulation_tables`] (a lossless row-major reshape of the per-block
//! `scale_shift_table` from the single file's flat `[6·hidden]` to the diffusers `[6, hidden]`), which
//! the single-file loader runs between the remap and `validate_transformer`.

use crate::config::Krea2Config;
use mlx_gen::gen_core::LogicalKeyMapping;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

/// Factors in the final continuous-AdaLN layer's `scale_shift_table`.
///
/// Architectural, not a published-model dimension: the reference `LastLayer`'s `SimpleModulation`
/// emits exactly `(scale, shift)` — two streams — where a single-stream block's
/// `DoubleSharedModulation` emits [`Krea2Config::MOD_FACTORS`] (6: pre/post × scale/shift/gate).
/// Both counts are fixed by the module code, so neither is a config field.
const FINAL_MOD_FACTORS: usize = 2;

/// Whether a [`KreaNativeToDiffusersMapping`] can declare the architecture's true logical shapes,
/// and — when it cannot — that this is a deliberate state rather than an accident (sc-20644).
///
/// MXFP8 storage is 32-padded on both axes and the file does not record the true shape, so the plan
/// compiler asks the adapter. A declared shape lets it unpad exactly; no declaration leaves it at
/// the **stored** padded shape, which the DiT's own [`crate::convert::validate_transformer`] then
/// refuses. Only a `Krea2Config` knows the architecture's dimensions, so a mapping with no config in
/// scope cannot answer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DeclaredLogicalShapes<'cfg> {
    /// **No config is in scope at this call site** — a header-only key-mapping check, or a plan
    /// compiled before the base tier (which owns `transformer/config.json`) has been resolved.
    ///
    /// `logical_shape` returns `None`, which is exactly the pre-sc-20644 behaviour: an MXFP8 plan
    /// unpads to the stored padded shape and `validate_transformer` is the backstop that refuses
    /// it, naming the tensor and the mismatch. This variant exists so "no config" is a named,
    /// documented state and never a guessed default architecture — and never a config-read error
    /// degraded into one. A call site that HAS a config must pass it; a call site whose config read
    /// fails must propagate that error rather than fall back here.
    NotInScope,
    /// The architecture config of the base tier this single file is being loaded against. Every
    /// declared shape is derived from it by [`diffusers_logical_shape`].
    FromConfig(&'cfg Krea2Config),
}

impl<'cfg> DeclaredLogicalShapes<'cfg> {
    /// [`FromConfig`](Self::FromConfig) when the base tier supplied an architecture config,
    /// [`NotInScope`](Self::NotInScope) when it carries none at all.
    ///
    /// The `None` here means **absent**, a checked condition the caller established by looking —
    /// never a config read that failed. A failed read is an error the caller must propagate; if it
    /// arrived here as `None` the plan would quietly fall back to padded shapes.
    pub const fn from_base(cfg: Option<&'cfg Krea2Config>) -> Self {
        match cfg {
            Some(cfg) => Self::FromConfig(cfg),
            None => Self::NotInScope,
        }
    }
}

/// The Krea 2 adapter's canonical key-mapping authority for the `krea-native` dialect (sc-20634):
/// the [`LogicalKeyMapping`] the mapped logical-weight reader consults, backed by
/// [`native_dit_key_to_diffusers`]. Its id is the one the portable
/// [`KREA_2_CHECKPOINT_ADAPTER`](mlx_gen::gen_core::KREA_2_CHECKPOINT_ADAPTER) row registers for
/// that dialect; the crate's conformance test proves the two agree. The id is a property of the
/// *key* correspondence, which carries no config, so it is identical in both
/// [`DeclaredLogicalShapes`] states.
///
/// # `logical_shape`: what lets a real MXFP8 Krea DiT load (sc-20644)
///
/// Constructed with [`for_config`](Self::for_config), the mapping declares every diffusers logical
/// key's true **unpadded** shape, derived from the `Krea2Config` by [`diffusers_logical_shape`], so
/// an MXFP8 layer's 32-padded storage unpads to the architecture's real geometry and the DiT loads.
/// Constructed with [`without_config`](Self::without_config) it declares nothing and the previous
/// fail-closed behaviour stands unchanged — see [`DeclaredLogicalShapes::NotInScope`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KreaNativeToDiffusersMapping<'cfg> {
    shapes: DeclaredLogicalShapes<'cfg>,
}

impl<'cfg> KreaNativeToDiffusersMapping<'cfg> {
    pub const MAPPING_ID: &'static str = "krea-native-to-diffusers-v1";

    /// The key mapping with the architecture config in scope: `logical_shape` declares the true
    /// unpadded shape of every key the [`Krea2Transformer`](crate::transformer::Krea2Transformer)
    /// loads.
    pub const fn for_config(cfg: &'cfg Krea2Config) -> Self {
        Self {
            shapes: DeclaredLogicalShapes::FromConfig(cfg),
        }
    }

    /// The key mapping with **no** config in scope — `logical_shape` declares nothing. See
    /// [`DeclaredLogicalShapes::NotInScope`] for when that is the correct choice.
    pub const fn without_config() -> Self {
        Self {
            shapes: DeclaredLogicalShapes::NotInScope,
        }
    }

    /// Build from an already-decided [`DeclaredLogicalShapes`] — the form the loader threads from
    /// its callers, so the "config or not" decision is made once at the entry point.
    pub const fn new(shapes: DeclaredLogicalShapes<'cfg>) -> Self {
        Self { shapes }
    }
}

impl LogicalKeyMapping for KreaNativeToDiffusersMapping<'_> {
    fn mapping_id(&self) -> &'static str {
        Self::MAPPING_ID
    }

    fn logical_key(&self, physical_key: &str) -> Option<String> {
        native_dit_key_to_diffusers(physical_key)
    }

    fn logical_shape(&self, logical_key: &str) -> Option<Vec<usize>> {
        match self.shapes {
            DeclaredLogicalShapes::NotInScope => None,
            DeclaredLogicalShapes::FromConfig(cfg) => diffusers_logical_shape(cfg, logical_key),
        }
    }
}

/// The Krea 2 **`diffusers`-dialect** mapping (sc-20651): the registered authority for a Krea 2
/// checkpoint that already ships canonical diffusers keys.
///
/// # Why this exists instead of `IdentityKeyMapping`
///
/// The registry used to name `identity-v1` here, and
/// [`IdentityKeyMapping`](mlx_gen::gen_core::IdentityKeyMapping)'s own doc comment forbids exactly
/// that use: it accepts **every** on-disk key as a logical weight, so an fp8 checkpoint carrying a
/// scale companion under an unrecognised suffix has that companion accepted as an ordinary weight
/// and its layer planned as undescribed fp8 at
/// [`ScalarScaleSource::Unit`](mlx_gen::gen_core::ScalarScaleSource) — decoding at unit scale,
/// silently wrong rather than refused.
///
/// This mapping is identity **only** over the keys the Krea 2 architecture actually contains, which
/// [`diffusers_logical_shape`] already enumerates exhaustively from the config, and returns `None`
/// for everything else. `None` is a typed
/// [`LogicalWeightPlanError::UnmappedKey`](mlx_gen::gen_core::LogicalWeightPlanError) naming the
/// offending tensor, so a foreign key — an unrecognised scale suffix included — refuses the import
/// instead of decoding wrong. The config is mandatory for the same reason: the key surface is
/// derived from it, and a mapping with no config could only be permissive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KreaDiffusersKeyMapping<'cfg> {
    cfg: &'cfg Krea2Config,
}

impl<'cfg> KreaDiffusersKeyMapping<'cfg> {
    /// Deliberately **not** spelled `MAPPING_ID`. The cross-backend workspace gate compares
    /// published consts of a `candle-gen-X`/`mlx-gen-X` pair by NAME, and `MAPPING_ID` already
    /// names the `krea-native` dialect's id on both backends. Two different dialects under one
    /// const name would either red that gate or, worse, be silenced by an exemption that hides a
    /// real future divergence — so the diffusers dialect's id carries the dialect in its name.
    pub const DIFFUSERS_MAPPING_ID: &'static str = "krea-2-diffusers-v1";

    pub const fn new(cfg: &'cfg Krea2Config) -> Self {
        Self { cfg }
    }
}

impl LogicalKeyMapping for KreaDiffusersKeyMapping<'_> {
    fn mapping_id(&self) -> &'static str {
        Self::DIFFUSERS_MAPPING_ID
    }

    fn logical_key(&self, physical_key: &str) -> Option<String> {
        // Recognised ⇔ the architecture declares a shape for it. One surface, one source of truth:
        // a key this returns `Some` for is a key the DiT really loads.
        diffusers_logical_shape(self.cfg, physical_key).map(|_| physical_key.to_owned())
    }

    fn logical_shape(&self, logical_key: &str) -> Option<Vec<usize>> {
        diffusers_logical_shape(self.cfg, logical_key)
    }
}

/// The architecture's true (unpadded) shape for one **diffusers** logical key, derived entirely from
/// `cfg` — the shape the [`Krea2Transformer`](crate::transformer::Krea2Transformer) module tree
/// constructs for that key and [`crate::convert::validate_transformer`] enforces.
///
/// Returns `None` for any key the architecture does not contain — including a recognized leaf on a
/// block index beyond the config's stack depth. `None` means "not declared", which leaves an MXFP8
/// layer at its stored padded shape and the DiT's own validation as the backstop; nothing is guessed.
///
/// Linear weights are `[out, in]`. The derivation, all from `cfg`:
/// * image stream width `hidden_size`, GQA projections [`Krea2Config::q_dim`] / [`Krea2Config::kv_dim`],
///   FFN inner `intermediate_size`;
/// * text-fusion stream width `text_hidden_dim`, its projections
///   `text_num_{attention,kv}_heads · attention_head_dim`, FFN inner `text_intermediate_size`;
/// * per-head QK RMSNorm scales are `[attention_head_dim]` (shared head dim across both streams);
/// * modulation tables are `[MOD_FACTORS, hidden]` per block and `[FINAL_MOD_FACTORS, hidden]` at
///   the output layer.
pub fn diffusers_logical_shape(cfg: &Krea2Config, logical_key: &str) -> Option<Vec<usize>> {
    let hidden = cfg.hidden_size;
    let text = cfg.text_hidden_dim;

    // Top-level (non-block) tensors.
    let top: Option<Vec<usize>> = match logical_key {
        "img_in.weight" => Some(vec![hidden, cfg.in_channels]),
        "img_in.bias" => Some(vec![hidden]),
        "txt_in.norm.weight" => Some(vec![text]),
        "txt_in.linear_1.weight" => Some(vec![hidden, text]),
        "txt_in.linear_2.weight" => Some(vec![hidden, hidden]),
        "txt_in.linear_1.bias" | "txt_in.linear_2.bias" => Some(vec![hidden]),
        "time_embed.linear_1.weight" => Some(vec![hidden, cfg.timestep_embed_dim]),
        "time_embed.linear_2.weight" => Some(vec![hidden, hidden]),
        "time_embed.linear_1.bias" | "time_embed.linear_2.bias" => Some(vec![hidden]),
        "time_mod_proj.weight" => Some(vec![cfg.time_mod_dim(), hidden]),
        "time_mod_proj.bias" => Some(vec![cfg.time_mod_dim()]),
        // The layer aggregator collapses the `num_text_layers` selected TE layers to one.
        "text_fusion.projector.weight" => Some(vec![1, cfg.num_text_layers]),
        "final_layer.linear.weight" => Some(vec![cfg.in_channels, hidden]),
        "final_layer.linear.bias" => Some(vec![cfg.in_channels]),
        "final_layer.norm.weight" => Some(vec![hidden]),
        "final_layer.scale_shift_table" => Some(vec![FINAL_MOD_FACTORS, hidden]),
        _ => None,
    };
    if top.is_some() {
        return top;
    }

    // `transformer_blocks.N.<leaf>` — the single-stream (GQA) stack.
    if let Some(rest) = logical_key.strip_prefix("transformer_blocks.") {
        let (index, leaf) = split_index(rest)?;
        if index >= cfg.num_layers {
            return None;
        }
        return block_leaf_shape(
            leaf,
            hidden,
            cfg.q_dim(),
            cfg.kv_dim(),
            cfg.intermediate_size,
            cfg.attention_head_dim,
            true,
        );
    }

    // `text_fusion.{layerwise,refiner}_blocks.N.<leaf>` — the full-attention text stacks.
    if let Some(rest) = logical_key.strip_prefix("text_fusion.") {
        for (kind, depth) in [
            ("layerwise_blocks.", cfg.num_layerwise_text_blocks),
            ("refiner_blocks.", cfg.num_refiner_text_blocks),
        ] {
            let Some(after) = rest.strip_prefix(kind) else {
                continue;
            };
            let (index, leaf) = split_index(after)?;
            if index >= depth {
                return None;
            }
            return block_leaf_shape(
                leaf,
                text,
                cfg.text_num_attention_heads * cfg.attention_head_dim,
                cfg.text_num_kv_heads * cfg.attention_head_dim,
                cfg.text_intermediate_size,
                cfg.attention_head_dim,
                // Text-fusion blocks are un-modulated: no per-block `scale_shift_table`.
                false,
            );
        }
    }

    None
}

/// Split `"<digits>.<leaf>"` into the parsed index and the leaf; `None` for anything else.
fn split_index(rest: &str) -> Option<(usize, &str)> {
    let (index, leaf) = rest.split_once('.')?;
    if index.is_empty() || !index.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some((index.parse().ok()?, leaf))
}

/// One block leaf's shape, parameterized by the block's stream width and projection widths so the
/// single-stream (GQA, image width) and text-fusion (full attention, text width) stacks share the
/// one derivation.
fn block_leaf_shape(
    leaf: &str,
    width: usize,
    q_dim: usize,
    kv_dim: usize,
    ffn: usize,
    head_dim: usize,
    modulated: bool,
) -> Option<Vec<usize>> {
    Some(match leaf {
        // Per-head QK RMSNorm scale — one weight per head dim, not per stream width.
        "attn.norm_q.weight" | "attn.norm_k.weight" => vec![head_dim],
        "attn.to_q.weight" | "attn.to_gate.weight" => vec![q_dim, width],
        "attn.to_k.weight" | "attn.to_v.weight" => vec![kv_dim, width],
        "attn.to_out.0.weight" => vec![width, q_dim],
        "ff.gate.weight" | "ff.up.weight" => vec![ffn, width],
        "ff.down.weight" => vec![width, ffn],
        "norm1.weight" | "norm2.weight" => vec![width],
        "scale_shift_table" if modulated => vec![Krea2Config::MOD_FACTORS, width],
        _ => return None,
    })
}

/// Translate a **native-mmdit** single-file DiT tensor key to the **diffusers** key the MLX
/// [`Krea2Transformer`](crate::transformer::Krea2Transformer) module tree loads. Returns `None` for any
/// key that is not a recognized Krea DiT tensor (including a key missing the `model.diffusion_model.`
/// prefix) — the caller collects those and errors, so an unexpected/foreign tensor never slips through
/// silently.
///
/// This is the exact inverse of candle's `convrot_diffusers_to_native`
/// (`candle-gen-krea/src/loader.rs`), minus the int8/rotation coupling — see the module docs. Shapes
/// line up 1:1 (the only reshapes — `time_mod_proj` / `scale_shift_table` — are done by the DiT), so this
/// is a pure rename.
pub fn native_dit_key_to_diffusers(key: &str) -> Option<String> {
    // The ComfyUI single file namespaces the whole DiT under `model.diffusion_model.`; a DiT tensor
    // without it is unrecognized (returns `None` → fail-closed at the call site).
    let key = key.strip_prefix("model.diffusion_model.")?;

    // Top-level (non-block) tensors.
    let top = match key {
        "first.weight" => Some("img_in.weight"),
        "first.bias" => Some("img_in.bias"),
        "txtmlp.0.scale" => Some("txt_in.norm.weight"),
        "txtmlp.1.weight" => Some("txt_in.linear_1.weight"),
        "txtmlp.1.bias" => Some("txt_in.linear_1.bias"),
        "txtmlp.3.weight" => Some("txt_in.linear_2.weight"),
        "txtmlp.3.bias" => Some("txt_in.linear_2.bias"),
        "tmlp.0.weight" => Some("time_embed.linear_1.weight"),
        "tmlp.0.bias" => Some("time_embed.linear_1.bias"),
        "tmlp.2.weight" => Some("time_embed.linear_2.weight"),
        "tmlp.2.bias" => Some("time_embed.linear_2.bias"),
        "tproj.1.weight" => Some("time_mod_proj.weight"),
        "tproj.1.bias" => Some("time_mod_proj.bias"),
        "txtfusion.projector.weight" => Some("text_fusion.projector.weight"),
        "last.linear.weight" => Some("final_layer.linear.weight"),
        "last.linear.bias" => Some("final_layer.linear.bias"),
        "last.norm.scale" => Some("final_layer.norm.weight"),
        "last.modulation.lin" => Some("final_layer.scale_shift_table"),
        _ => None,
    };
    if let Some(t) = top {
        return Some(t.to_string());
    }

    // Per-block leaf remap (shared by the single-stream `blocks` and the two text-fusion stacks).
    let leaf = |rest: &str| -> Option<&'static str> {
        Some(match rest {
            "attn.qknorm.qnorm.scale" => "attn.norm_q.weight",
            "attn.qknorm.knorm.scale" => "attn.norm_k.weight",
            "attn.wq.weight" => "attn.to_q.weight",
            "attn.wk.weight" => "attn.to_k.weight",
            "attn.wv.weight" => "attn.to_v.weight",
            "attn.wo.weight" => "attn.to_out.0.weight",
            "attn.gate.weight" => "attn.to_gate.weight",
            "mlp.gate.weight" => "ff.gate.weight",
            "mlp.up.weight" => "ff.up.weight",
            "mlp.down.weight" => "ff.down.weight",
            "prenorm.scale" => "norm1.weight",
            "postnorm.scale" => "norm2.weight",
            "mod.lin" => "scale_shift_table",
            _ => return None,
        })
    };

    // `blocks.N.<leaf>` → `transformer_blocks.N.<diffusers-leaf>`.
    if let Some(rest) = key.strip_prefix("blocks.") {
        if let Some((idx, tail)) = rest.split_once('.') {
            if !idx.is_empty() && idx.chars().all(|c| c.is_ascii_digit()) {
                return leaf(tail).map(|dl| format!("transformer_blocks.{idx}.{dl}"));
            }
        }
    }

    // `txtfusion.{layerwise,refiner}_blocks.N.<leaf>` → `text_fusion.{...}.N.<diffusers-leaf>`.
    if let Some(rest) = key.strip_prefix("txtfusion.") {
        for kind in ["layerwise_blocks.", "refiner_blocks."] {
            if let Some(after) = rest.strip_prefix(kind) {
                if let Some((idx, tail)) = after.split_once('.') {
                    if !idx.is_empty() && idx.chars().all(|c| c.is_ascii_digit()) {
                        return leaf(tail).map(|dl| format!("text_fusion.{kind}{idx}.{dl}"));
                    }
                }
            }
        }
    }

    None
}

/// Rename every tensor in a native-mmdit single-file DiT weight set to its diffusers key, moving the
/// tensors into a fresh [`Weights`] the [`Krea2Transformer`](crate::transformer::Krea2Transformer) loads
/// directly. Values and dtypes are preserved verbatim.
///
/// Fails closed (typed [`Error::Msg`]) — never a silent skip — when:
/// * an on-disk key maps to `None` ([`native_dit_key_to_diffusers`]) — a foreign/unexpected tensor, or a
///   key missing the `model.diffusion_model.` prefix; or
/// * two distinct native keys collide onto the same diffusers key (a non-injective mapping).
///
/// Presence of every diffusers key the DiT needs (and the representative shape checks) is the separate
/// [`crate::convert::validate_transformer`] pass the loader runs on the returned set — an unmapped key
/// cannot be caught there because it never enters the returned map, which is why the unmapped case is
/// enforced here.
pub fn remap_native_dit_to_diffusers(mut native: Weights) -> Result<Weights> {
    let keys: Vec<String> = native.keys().map(str::to_string).collect();

    let mut out = Weights::empty();
    // diffusers key → the native key that produced it, for a precise collision diagnostic.
    let mut source: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut unmapped: Vec<String> = Vec::new();
    let mut collisions: Vec<String> = Vec::new();

    for native_key in keys {
        let Some(diffusers_key) = native_dit_key_to_diffusers(&native_key) else {
            unmapped.push(native_key);
            continue;
        };
        // `keys` came from `native.keys()`, so the remove is infallible.
        let tensor = native
            .remove(&native_key)
            .ok_or_else(|| Error::MissingTensor(native_key.clone()))?;
        if let Some(prev) = source.insert(diffusers_key.clone(), native_key.clone()) {
            collisions.push(format!("{prev} + {native_key} → {diffusers_key}"));
            continue;
        }
        out.insert(diffusers_key, tensor);
    }

    if !unmapped.is_empty() {
        unmapped.sort();
        return Err(Error::Msg(format!(
            "krea single-file remap: {} on-disk DiT key(s) have no diffusers mapping (unrecognized \
             checkpoint, wrong family, or a key outside the `model.diffusion_model.` DiT namespace): \
             [{}]",
            unmapped.len(),
            preview(&unmapped),
        )));
    }
    if !collisions.is_empty() {
        collisions.sort();
        return Err(Error::Msg(format!(
            "krea single-file remap: {} diffusers key collision(s) — the native→diffusers map is not \
             injective over this checkpoint: [{}]",
            collisions.len(),
            preview(&collisions),
        )));
    }
    Ok(out)
}

/// Normalize the modulation ("`scale_shift_table`") tables to the diffusers 2-D `[factors, hidden]`
/// shape (`6` factors per single-stream block, `2` for the final continuous-AdaLN layer).
///
/// The remap ([`remap_native_dit_to_diffusers`]) is a pure key rename — values/dtypes/shapes pass
/// through. But a ComfyUI single file may store the **per-block** modulation table FLAT
/// (`[factors·hidden]`, e.g. variant5's `blocks.N.mod.lin` is `[36864]`) where the published diffusers
/// `transformer/` snapshot stores it 2-D `[6, hidden]`. The flat form is row-major-identical to the 2-D
/// form (the DiT reshapes it to `[1, 1, 6·hidden]` either way, so the forward is unaffected), but the
/// shape check in [`crate::convert::validate_transformer`] expects the 2-D diffusers shape. Reshaping
/// flat→2-D (a lossless row-major view) makes the remapped set shape-identical to a snapshot load.
/// Already-2-D tables (variant5's `last.modulation.lin` is `[2, hidden]`) pass through untouched.
///
/// Errors (never silently reshapes to a wrong grid) if a flat table's element count is not divisible by
/// its factor — a truncated/foreign tensor.
pub fn normalize_modulation_tables(w: &mut Weights) -> Result<()> {
    let keys: Vec<String> = w
        .keys()
        .filter(|k| k.ends_with(".scale_shift_table"))
        .map(str::to_string)
        .collect();
    for key in keys {
        // The final continuous-AdaLN table is 2-factor (scale/shift); every single-stream block table is
        // 6-factor (pre/post × scale/shift/gate). Matches `Krea2Config::MOD_FACTORS` (6) and the DiT's
        // `final_layer` reshape to `[1, 2, hidden]`.
        let factors: i32 = if key == "final_layer.scale_shift_table" {
            FINAL_MOD_FACTORS as i32
        } else {
            Krea2Config::MOD_FACTORS as i32
        };
        // `remove` → own the tensor so the re-insert doesn't fight the read borrow.
        let tensor = w
            .remove(&key)
            .ok_or_else(|| Error::MissingTensor(key.clone()))?;
        if tensor.shape().len() == 2 {
            w.insert(key, tensor); // already `[factors, hidden]` — snapshot-shaped.
            continue;
        }
        let numel: i32 = tensor.shape().iter().product();
        if numel % factors != 0 {
            return Err(Error::Msg(format!(
                "krea single-file remap: `{key}` has {numel} elements, not divisible by {factors} \
                 modulation factors — cannot reshape to the diffusers [{factors}, hidden] table"
            )));
        }
        let reshaped = tensor.reshape(&[factors, numel / factors])?;
        w.insert(key, reshaped);
    }
    Ok(())
}

/// First few items of a diagnostic list (the full set can be hundreds of keys).
fn preview(items: &[String]) -> String {
    const HEAD: usize = 8;
    let shown = items
        .iter()
        .take(HEAD)
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(", ");
    if items.len() > HEAD {
        format!("{shown}, …")
    } else {
        shown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::expected_transformer_keys;
    use mlx_rs::Array;
    use std::collections::BTreeSet;

    /// A deliberately *non*-published architecture: every derived width is a different number, so a
    /// shape rule that reached for the wrong config field (or hardcoded a Turbo dimension) produces
    /// a visibly wrong answer instead of accidentally matching. hidden 128 (8×16), GQA 8Q/2KV,
    /// text 64 (4×16) with 3 text KV heads, and distinct FFN / in-channel / timestep widths.
    fn asymmetric_config() -> Krea2Config {
        let cfg = Krea2Config {
            in_channels: 8,
            patch_size: 2,
            hidden_size: 128,
            num_attention_heads: 8,
            num_kv_heads: 2,
            attention_head_dim: 16,
            num_layers: 3,
            intermediate_size: 192,
            norm_eps: 1e-5,
            axes_dims_rope: [4, 6, 6],
            rope_theta: 1000.0,
            timestep_embed_dim: 24,
            num_text_layers: 5,
            num_layerwise_text_blocks: 2,
            num_refiner_text_blocks: 1,
            text_hidden_dim: 64,
            text_intermediate_size: 80,
            text_num_attention_heads: 4,
            text_num_kv_heads: 3,
        };
        cfg.validate().expect("the test architecture is coherent");
        cfg
    }

    /// The `diffusers`-dialect mapping is identity over the **real** Krea 2 diffusers key surface —
    /// every key `expected_transformer_keys` says the DiT loads — and REFUSES anything else,
    /// including the undescribed-fp8 scale companions that made `identity-v1` unsafe here.
    ///
    /// The refusal corpus is deliberately the real hazard, not a toy string: `.scale_weight` is the
    /// legacy ComfyUI marker convention, `.scale_input`/`.weight_scale_2` are companions this route
    /// does not claim on an undescribed layer, and each is a suffix an fp8 export really ships. With
    /// `identity-v1` every one of them was accepted as an ordinary weight, and its layer then
    /// planned as undescribed fp8 at unit scale.
    #[test]
    fn diffusers_dialect_mapping_accepts_the_architecture_and_refuses_everything_else() {
        let cfg = asymmetric_config();
        let mapping = KreaDiffusersKeyMapping::new(&cfg);
        assert_eq!(mapping.mapping_id(), "krea-2-diffusers-v1");

        let expected = expected_transformer_keys(&cfg);
        assert!(
            expected.len() > 50,
            "the corpus must be the real architecture, got {} keys",
            expected.len()
        );
        for key in &expected {
            assert_eq!(
                mapping.logical_key(key).as_deref(),
                Some(key.as_str()),
                "the diffusers dialect must accept its own canonical key {key:?} unchanged"
            );
            assert!(
                mapping.logical_shape(key).is_some(),
                "declared surface must declare a shape for {key:?}"
            );
        }

        // The scale-companion suffixes an fp8 export ships, hung off a REAL layer of this
        // architecture, plus a foreign tensor and an out-of-range block index.
        let real_layer = "transformer_blocks.0.attn.to_q";
        for suffix in [
            ".scale_weight",
            ".scale_input",
            ".weight_scale",
            ".weight_scale_2",
            ".input_scale",
            ".comfy_quant",
        ] {
            let key = format!("{real_layer}{suffix}");
            assert_eq!(
                mapping.logical_key(&key),
                None,
                "{key:?} is not a Krea 2 logical weight and must refuse, not decode at unit scale"
            );
        }
        assert_eq!(mapping.logical_key("foreign.weight"), None);
        assert_eq!(
            mapping.logical_key(&format!(
                "transformer_blocks.{}.attn.to_q.weight",
                cfg.num_layers
            )),
            None,
            "a block past the configured stack depth is not part of this architecture"
        );
    }

    /// The real native-mmdit key set captured from `kreamania_variant5.safetensors` (430 tensors) — the
    /// committed fixture (the 26 GB weights file itself is not committed). Comment/blank lines dropped.
    fn variant5_native_keys() -> Vec<String> {
        let raw = include_str!("../tests/fixtures/variant5_native_keys.txt");
        raw.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(str::to_string)
            .collect()
    }

    /// A 1-element bf16 placeholder — the remap moves tensor handles by key; values/shape are irrelevant
    /// to key coverage, so a scalar stands in for each of the 430 real tensors.
    fn stub() -> Array {
        Array::from_slice(&[0.0f32], &[1])
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .unwrap()
    }

    /// The fixture is the real header: 430 native keys, all under the DiT prefix.
    #[test]
    fn fixture_is_the_real_variant5_header() {
        let keys = variant5_native_keys();
        assert_eq!(keys.len(), 430, "variant5 ships 430 DiT tensors");
        assert!(
            keys.iter().all(|k| k.starts_with("model.diffusion_model.")),
            "every variant5 DiT tensor is under the `model.diffusion_model.` prefix"
        );
    }

    /// **Every variant5 key template maps to a valid module-tree (diffusers) key, and the covered set is
    /// EXACTLY the set the transformer requires** — driven by the real header + the loader's own
    /// `expected_transformer_keys`. Set equality proves both directions at once: full coverage (no
    /// missing module weight) and no stray mapping (no diffusers key the module tree does not consume).
    #[test]
    fn remap_covers_every_variant5_key_and_matches_expected_module_keys() {
        let mapped: BTreeSet<String> = variant5_native_keys()
            .iter()
            .map(|k| {
                native_dit_key_to_diffusers(k)
                    .unwrap_or_else(|| panic!("variant5 key has no diffusers mapping: {k}"))
            })
            .collect();

        let expected: BTreeSet<String> = expected_transformer_keys(&Krea2Config::turbo())
            .into_iter()
            .collect();

        let missing: Vec<&String> = expected.difference(&mapped).collect();
        let extra: Vec<&String> = mapped.difference(&expected).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "remap ≠ module keys: missing {missing:?}, extra {extra:?}"
        );
    }

    /// **The mapping is a bijection over the covered set (no collisions):** 430 distinct native keys map
    /// to 430 distinct diffusers keys.
    #[test]
    fn remap_is_injective_over_variant5() {
        let native = variant5_native_keys();
        let mapped: BTreeSet<String> = native
            .iter()
            .filter_map(|k| native_dit_key_to_diffusers(k))
            .collect();
        assert_eq!(
            mapped.len(),
            native.len(),
            "collision: {} native keys collapsed to {} diffusers keys",
            native.len(),
            mapped.len()
        );
    }

    /// **`remap_native_dit_to_diffusers` renames the whole real header into a loadable diffusers set** —
    /// the end-to-end remap over a `Weights` built from every real key (stub tensors), asserting the
    /// output key set equals the module tree's expected set.
    #[test]
    fn remap_weights_end_to_end_over_real_header() {
        let mut w = Weights::empty();
        for k in variant5_native_keys() {
            w.insert(k, stub());
        }
        let out = remap_native_dit_to_diffusers(w).expect("real header remaps cleanly");

        let got: BTreeSet<String> = out.keys().map(str::to_string).collect();
        let expected: BTreeSet<String> = expected_transformer_keys(&Krea2Config::turbo())
            .into_iter()
            .collect();
        assert_eq!(
            got, expected,
            "remapped key set must equal the module tree's keys"
        );
    }

    /// **Fail-closed on an unmapped on-disk key.** A foreign/unexpected tensor (here a key without the
    /// `model.diffusion_model.` prefix) yields a typed error naming it — never a silent skip.
    #[test]
    fn unmapped_key_fails_closed() {
        let mut w = Weights::empty();
        // A valid key so the map is non-empty, plus one that cannot map.
        w.insert("model.diffusion_model.first.weight", stub());
        w.insert("unexpected.foreign.tensor", stub());

        // `Weights` is not `Debug`, so match rather than `expect_err`.
        let err = match remap_native_dit_to_diffusers(w) {
            Ok(_) => panic!("an unmapped key must fail closed"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("no diffusers mapping"),
            "unexpected error: {err}"
        );
        assert!(
            err.contains("unexpected.foreign.tensor"),
            "error must name the key: {err}"
        );
    }

    /// **Fail-closed on a native key inside the DiT namespace that is not a recognized leaf.** A
    /// truncated/garbage tensor under the prefix (`…blocks.0.attn.bogus`) still errors.
    #[test]
    fn unrecognized_leaf_under_prefix_fails_closed() {
        let mut w = Weights::empty();
        w.insert("model.diffusion_model.first.weight", stub());
        w.insert("model.diffusion_model.blocks.0.attn.bogus", stub());
        let err = match remap_native_dit_to_diffusers(w) {
            Ok(_) => panic!("an unrecognized leaf must fail closed"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("no diffusers mapping"),
            "unexpected error: {err}"
        );
    }

    /// **Fail-closed detects a missing module weight (via `validate_transformer` over the remap).**
    /// Dropping one native key (`first.bias`) yields a remapped set the transformer coverage check
    /// rejects — the complementary half of the fail-closed contract (unmapped is caught in the remap;
    /// missing is caught downstream). Proven here by wiring the two together as the loader does.
    #[test]
    fn missing_required_key_is_caught_by_validation() {
        let mut w = Weights::empty();
        for k in variant5_native_keys() {
            if k == "model.diffusion_model.first.bias" {
                continue; // simulate a truncated download missing img_in.bias
            }
            w.insert(k, stub());
        }
        let remapped =
            remap_native_dit_to_diffusers(w).expect("remap of a present subset is clean");
        // The coverage half is `validate_transformer`; shapes are stubbed, so assert the coverage
        // message specifically (it runs before the shape checks).
        let err = crate::convert::validate_transformer(&remapped, &Krea2Config::turbo())
            .expect_err("a missing module weight must fail closed")
            .to_string();
        assert!(
            err.contains("img_in.bias"),
            "error must name the missing key: {err}"
        );
    }

    /// The individual top-level and per-block correspondences, spot-checked against candle's
    /// `convrot_diffusers_to_native` inverse (the load-bearing renames the epic calls out).
    #[test]
    fn spot_check_representative_renames() {
        let cases = [
            ("model.diffusion_model.first.weight", "img_in.weight"),
            (
                "model.diffusion_model.last.modulation.lin",
                "final_layer.scale_shift_table",
            ),
            (
                "model.diffusion_model.last.norm.scale",
                "final_layer.norm.weight",
            ),
            ("model.diffusion_model.txtmlp.0.scale", "txt_in.norm.weight"),
            (
                "model.diffusion_model.txtfusion.projector.weight",
                "text_fusion.projector.weight",
            ),
            (
                "model.diffusion_model.tproj.1.weight",
                "time_mod_proj.weight",
            ),
            (
                "model.diffusion_model.blocks.7.attn.wq.weight",
                "transformer_blocks.7.attn.to_q.weight",
            ),
            (
                "model.diffusion_model.blocks.7.attn.qknorm.qnorm.scale",
                "transformer_blocks.7.attn.norm_q.weight",
            ),
            (
                "model.diffusion_model.blocks.7.attn.wo.weight",
                "transformer_blocks.7.attn.to_out.0.weight",
            ),
            (
                "model.diffusion_model.blocks.7.mod.lin",
                "transformer_blocks.7.scale_shift_table",
            ),
            (
                "model.diffusion_model.txtfusion.refiner_blocks.1.mlp.down.weight",
                "text_fusion.refiner_blocks.1.ff.down.weight",
            ),
        ];
        for (native, diffusers) in cases {
            assert_eq!(
                native_dit_key_to_diffusers(native).as_deref(),
                Some(diffusers),
                "wrong remap for {native}"
            );
        }
    }

    /// **`normalize_modulation_tables` reshapes a flat per-block table to the diffusers 2-D shape and
    /// leaves an already-2-D final table alone.** A flat `[6·h]` block table (variant5's `mod.lin` form)
    /// becomes `[6, h]`; the `[2, h]` final table (variant5's `modulation.lin` form) is unchanged; the
    /// reshape is a lossless row-major view so the flattened values are preserved.
    #[test]
    fn normalize_reshapes_flat_block_table_and_keeps_2d_final() {
        let hidden = 4i32;
        // Flat block table `[6·hidden]` = [24], row-major values 0..24.
        let flat: Vec<f32> = (0..(6 * hidden)).map(|i| i as f32).collect();
        let final_2d: Vec<f32> = (0..(2 * hidden)).map(|i| (100 + i) as f32).collect();

        let mut w = Weights::empty();
        w.insert(
            "transformer_blocks.0.scale_shift_table",
            Array::from_slice(&flat, &[6 * hidden]),
        );
        w.insert(
            "final_layer.scale_shift_table",
            Array::from_slice(&final_2d, &[2, hidden]),
        );

        normalize_modulation_tables(&mut w).expect("normalization is clean");

        let block = w.require("transformer_blocks.0.scale_shift_table").unwrap();
        assert_eq!(
            block.shape(),
            &[6, hidden],
            "flat block table reshaped to [6, hidden]"
        );
        // Row-major values preserved by the reshape (contiguous view → physical buffer in order).
        assert_eq!(block.as_slice::<f32>(), flat.as_slice());

        let fin = w.require("final_layer.scale_shift_table").unwrap();
        assert_eq!(
            fin.shape(),
            &[2, hidden],
            "already-2-D final table unchanged"
        );
    }

    /// **`normalize_modulation_tables` fails closed on a flat table whose element count is not divisible
    /// by its factor** — a truncated/foreign tensor, not silently reshaped to a wrong grid.
    #[test]
    fn normalize_fails_closed_on_indivisible_flat_table() {
        let mut w = Weights::empty();
        // 25 is not divisible by the 6 block modulation factors.
        w.insert(
            "transformer_blocks.0.scale_shift_table",
            Array::from_slice(&[0.0f32; 25], &[25]),
        );
        let err = normalize_modulation_tables(&mut w)
            .expect_err("indivisible flat table must fail closed")
            .to_string();
        assert!(err.contains("not divisible"), "unexpected error: {err}");
    }

    // ── sc-20644: the declared logical shapes ──────────────────────────────────────────────────

    /// **With no config in scope the mapping declares nothing** — the explicit
    /// [`DeclaredLogicalShapes::NotInScope`] state, which is exactly the pre-sc-20644 behaviour
    /// (an MXFP8 plan stays at the stored padded shape and `validate_transformer` refuses it).
    /// The key correspondence and the registered mapping id are unaffected by the state.
    #[test]
    fn without_a_config_no_logical_shape_is_declared() {
        let mapping = KreaNativeToDiffusersMapping::without_config();
        for key in expected_transformer_keys(&asymmetric_config()) {
            assert_eq!(
                mapping.logical_shape(&key),
                None,
                "`{key}` must not be declared with no config in scope"
            );
        }
        assert_eq!(
            mapping.mapping_id(),
            KreaNativeToDiffusersMapping::MAPPING_ID
        );
        let cfg = asymmetric_config();
        assert_eq!(
            mapping.logical_key("model.diffusion_model.blocks.0.attn.wq.weight"),
            KreaNativeToDiffusersMapping::for_config(&cfg)
                .logical_key("model.diffusion_model.blocks.0.attn.wq.weight"),
            "the key correspondence carries no config and cannot depend on the state"
        );
    }

    /// **With a config in scope EVERY logical key the transformer loads has a declared shape** —
    /// driven by the loader's own `expected_transformer_keys`, so a key the module tree consumes
    /// can never be left undeclared (which would silently keep it at its padded storage).
    #[test]
    fn every_expected_transformer_key_has_a_declared_shape() {
        let cfg = asymmetric_config();
        let mapping = KreaNativeToDiffusersMapping::for_config(&cfg);
        for key in expected_transformer_keys(&cfg) {
            assert!(
                mapping.logical_shape(&key).is_some(),
                "`{key}` is loaded by the module tree but has no declared logical shape"
            );
        }
    }

    /// **The declared shapes ARE the architecture the DiT validates against.** A weight set built
    /// purely from `logical_shape` declarations passes `validate_transformer` — the DiT's own
    /// coverage + shape check — so a declaration can never be a geometry the loader then refuses.
    #[test]
    fn declared_shapes_satisfy_the_dit_architecture_validation() {
        let cfg = asymmetric_config();
        let mapping = KreaNativeToDiffusersMapping::for_config(&cfg);
        let mut w = Weights::empty();
        for key in expected_transformer_keys(&cfg) {
            let shape = mapping
                .logical_shape(&key)
                .unwrap_or_else(|| panic!("no declared shape for {key}"));
            let dims: Vec<i32> = shape.iter().map(|d| *d as i32).collect();
            w.insert(key, mlx_rs::ops::zeros::<f32>(&dims).unwrap());
        }
        crate::convert::validate_transformer(&w, &cfg)
            .expect("a DiT built from the declared shapes is the architecture the DiT expects");
    }

    /// **The shapes are DERIVED from the config, not hardcoded.** Two different architectures give
    /// different declarations for the same key, and each matches that architecture's own widths.
    #[test]
    fn declared_shapes_track_the_config_rather_than_a_published_model() {
        let tiny = asymmetric_config();
        let turbo = Krea2Config::turbo();
        let a = KreaNativeToDiffusersMapping::for_config(&tiny);
        let b = KreaNativeToDiffusersMapping::for_config(&turbo);

        // Single-stream GQA: Q spans all heads, K/V only the KV heads.
        assert_eq!(
            a.logical_shape("transformer_blocks.0.attn.to_q.weight"),
            Some(vec![tiny.q_dim(), tiny.hidden_size])
        );
        assert_eq!(
            a.logical_shape("transformer_blocks.0.attn.to_k.weight"),
            Some(vec![tiny.kv_dim(), tiny.hidden_size])
        );
        assert_eq!(
            b.logical_shape("transformer_blocks.0.attn.to_q.weight"),
            Some(vec![turbo.q_dim(), turbo.hidden_size])
        );
        assert_ne!(
            a.logical_shape("transformer_blocks.0.attn.to_q.weight"),
            b.logical_shape("transformer_blocks.0.attn.to_q.weight"),
            "two architectures must not declare the same shape"
        );
        // The shared modulation projection and the per-block table both carry MOD_FACTORS.
        assert_eq!(
            a.logical_shape("time_mod_proj.weight"),
            Some(vec![
                Krea2Config::MOD_FACTORS * tiny.hidden_size,
                tiny.hidden_size
            ])
        );
        assert_eq!(
            a.logical_shape("transformer_blocks.0.scale_shift_table"),
            Some(vec![Krea2Config::MOD_FACTORS, tiny.hidden_size])
        );
        // The output layer's SimpleModulation is 2-factor, not 6.
        assert_eq!(
            a.logical_shape("final_layer.scale_shift_table"),
            Some(vec![FINAL_MOD_FACTORS, tiny.hidden_size])
        );
        // Text-fusion blocks run at the TEXT width with their own KV head count, and are
        // un-modulated (no per-block table).
        assert_eq!(
            a.logical_shape("text_fusion.refiner_blocks.0.attn.to_k.weight"),
            Some(vec![
                tiny.text_num_kv_heads * tiny.attention_head_dim,
                tiny.text_hidden_dim
            ])
        );
        assert_eq!(
            a.logical_shape("text_fusion.refiner_blocks.0.ff.down.weight"),
            Some(vec![tiny.text_hidden_dim, tiny.text_intermediate_size])
        );
        assert_eq!(
            a.logical_shape("text_fusion.refiner_blocks.0.scale_shift_table"),
            None,
            "text-fusion blocks have no modulation table"
        );
        // Per-head QK norms are head-dim wide in BOTH streams, never the stream width.
        assert_eq!(
            a.logical_shape("transformer_blocks.0.attn.norm_q.weight"),
            Some(vec![tiny.attention_head_dim])
        );
        assert_eq!(
            a.logical_shape("text_fusion.layerwise_blocks.0.attn.norm_k.weight"),
            Some(vec![tiny.attention_head_dim])
        );
    }

    /// **Nothing outside the architecture is declared** — a foreign key, an unrecognized leaf, and a
    /// block index past the config's stack depth all return `None` rather than a guessed shape.
    #[test]
    fn keys_outside_the_architecture_are_not_declared() {
        let cfg = asymmetric_config();
        let mapping = KreaNativeToDiffusersMapping::for_config(&cfg);
        for key in [
            "foreign.weight",
            "transformer_blocks.0.attn.bogus.weight",
            "transformer_blocks.x.attn.to_q.weight",
            "text_fusion.unknown_blocks.0.norm1.weight",
        ] {
            assert_eq!(mapping.logical_shape(key), None, "`{key}` must not declare");
        }
        // One past each stack's depth.
        for key in [
            format!("transformer_blocks.{}.attn.to_q.weight", cfg.num_layers),
            format!(
                "text_fusion.layerwise_blocks.{}.norm1.weight",
                cfg.num_layerwise_text_blocks
            ),
            format!(
                "text_fusion.refiner_blocks.{}.norm1.weight",
                cfg.num_refiner_text_blocks
            ),
        ] {
            assert_eq!(
                mapping.logical_shape(&key),
                None,
                "`{key}` is past the configured stack depth"
            );
        }
    }
}
