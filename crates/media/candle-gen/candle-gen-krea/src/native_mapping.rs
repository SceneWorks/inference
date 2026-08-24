//! The Candle Krea 2 adapter's **canonical key-mapping authority** for the `krea-native` dialect
//! (epic 20398, sc-20651) — the Candle twin of `mlx_gen_krea::native_remap`.
//!
//! # Why this exists
//!
//! `candle-gen-krea` imported ComfyUI Kitchen NVFP4 checkpoints through a bespoke predicate — "this
//! `.weight` resolves to a `U8` tensor, therefore it is NVFP4" — with the companion keys spelled by
//! string concatenation at the read site and no `.comfy_quant` / `_quantization_metadata` descriptor
//! consulted anywhere. That is a second, private implementation of the thing
//! `gen_core::checkpoint_codec` exists to be: it cannot tell NVFP4 from any other `U8` payload, it
//! cannot see `full_precision_matrix_mult`, it prices nothing, and it diverges from MLX on the same
//! file. [`KreaNativeToDiffusersMapping`] is what lets the Candle loader plan through
//! [`candle_gen::logical_weights::plan_logical_weights`] instead, so the *descriptor* decides the
//! codec and the plan decides residency — the same authority MLX already reads.
//!
//! # The mapping
//!
//! A ComfyUI-exported single-file Krea 2 DiT stores the **native-mmdit** tensor names
//! (`blocks.N.attn.wq`, `prenorm.scale`, `tproj`, `first`, `last`, `txtmlp`, `txtfusion.*`),
//! optionally namespaced under `model.diffusion_model.`; the Candle
//! [`Krea2Transformer`](crate::transformer::Krea2Transformer) reads the **diffusers** schema. This
//! module is the native → diffusers direction, i.e. the exact inverse of
//! [`convrot_diffusers_to_native`](crate::loader::convrot_diffusers_to_native) — which is the map
//! validated exhaustively against the real 878-tensor ConvRot header. The inverse property is not
//! asserted by eye: [`tests::mapping_is_the_exact_inverse_of_the_authoritative_forward_map`] drives
//! every diffusers key the architecture contains through the forward map and back.
//!
//! `mapping_id` is [`KreaNativeToDiffusersMapping::MAPPING_ID`], the id the portable
//! `KREA_2_CHECKPOINT_ADAPTER` registry row declares for the `krea-native` dialect. One dialect has
//! one canonical mapping; the two engines each implement it, and each crate's conformance test
//! proves its row is backed by a real impl.
//!
//! # Declared logical shapes
//!
//! NVFP4 storage is 16-padded on both axes and the file does not record the layer's true geometry,
//! so a plan compiled from a mapping that declares no shape can only carry the **padded** grid
//! forward — and materializing that grid turns padding into weights. `gen_core` refuses such a
//! materialization by name, so the Krea import declares its shapes from the architecture config,
//! exactly as MLX does. [`DeclaredLogicalShapes`] makes "no config in scope" a named state rather
//! than an accident.

use candle_gen::gen_core::LogicalKeyMapping;

use crate::config::Krea2Config;
use crate::loader::convrot_diffusers_to_native;

/// Factors in the final continuous-AdaLN layer's `scale_shift_table`.
///
/// Architectural, not a published-model dimension: the reference `LastLayer`'s `SimpleModulation`
/// emits exactly `(scale, shift)` — two streams — where a single-stream block's
/// `DoubleSharedModulation` emits [`Krea2Config::MOD_FACTORS`] (6: pre/post × scale/shift/gate).
/// Both counts are fixed by the module code, so neither is a config field.
const FINAL_MOD_FACTORS: usize = 2;

/// Whether a [`KreaNativeToDiffusersMapping`] can declare the architecture's true logical shapes,
/// and — when it cannot — that this is a deliberate state rather than an accident.
///
/// Only a [`Krea2Config`] knows the architecture's dimensions, so a mapping with no config in scope
/// cannot answer. That is a real state (a header-only key check, or a plan compiled before the base
/// tier that owns `transformer/config.json` has been resolved) and it is named here so it can never
/// be a *guessed* default architecture, and never a config-read error degraded into one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DeclaredLogicalShapes<'cfg> {
    /// **No config is in scope at this call site.** [`LogicalKeyMapping::logical_shape`] returns
    /// `None` for every key. A dense / scalar-fp8 / int8 checkpoint still plans and reads (their
    /// stored grid *is* the logical one); a **block-padded** MXFP8 or NVFP4 layer plans (for
    /// pricing) but refuses to materialize, by name, in `gen_core`.
    NotInScope,
    /// The architecture config of the base tier this single file is loaded against. Every declared
    /// shape is derived from it by [`diffusers_logical_shape`].
    FromConfig(&'cfg Krea2Config),
}

impl<'cfg> DeclaredLogicalShapes<'cfg> {
    /// [`FromConfig`](Self::FromConfig) when the caller resolved an architecture config,
    /// [`NotInScope`](Self::NotInScope) when it has none at all.
    ///
    /// The `None` here means **absent**, a condition the caller established by looking — never a
    /// config read that failed. A failed read is an error the caller must propagate; arriving here
    /// as `None` would quietly drop the import back to padded shapes.
    pub const fn from_base(cfg: Option<&'cfg Krea2Config>) -> Self {
        match cfg {
            Some(cfg) => Self::FromConfig(cfg),
            None => Self::NotInScope,
        }
    }
}

/// The `krea-native` dialect's [`LogicalKeyMapping`]: on-disk native-mmdit key → canonical
/// diffusers key, plus the architecture's declared logical shapes when a config is in scope.
///
/// `prefix` is the namespace detected on the file being planned (`""` for the bare-keyed ComfyUI
/// INT8-ConvRot export, `"model.diffusion_model."` for the community single files). It is a
/// property of the *file*, not of the dialect, which is why it is carried on the value rather than
/// hardcoded — and why `logical_key` refuses a key that does not carry it, instead of accepting
/// both spellings and letting two prefixes collide onto one logical key.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KreaNativeToDiffusersMapping<'a> {
    prefix: &'a str,
    shapes: DeclaredLogicalShapes<'a>,
}

impl<'a> KreaNativeToDiffusersMapping<'a> {
    /// The id the `KREA_2_CHECKPOINT_ADAPTER` registry row declares for the `krea-native` dialect.
    /// Deliberately identical to the MLX implementation's: one dialect, one canonical mapping.
    pub const MAPPING_ID: &'static str = "krea-native-to-diffusers-v1";

    /// The namespace prefix used by files whose DiT is nested under a wrapper module.
    pub const DIFFUSION_MODEL_PREFIX: &'static str = "model.diffusion_model.";

    /// Build for one file's detected namespace prefix and declared-shape state.
    pub const fn new(prefix: &'a str, shapes: DeclaredLogicalShapes<'a>) -> Self {
        Self { prefix, shapes }
    }

    /// The mapping for a file under `prefix` with the architecture config in scope.
    pub const fn for_config(prefix: &'a str, cfg: &'a Krea2Config) -> Self {
        Self::new(prefix, DeclaredLogicalShapes::FromConfig(cfg))
    }

    /// The mapping for a file under `prefix` with **no** config in scope — see
    /// [`DeclaredLogicalShapes::NotInScope`] for what that costs.
    pub const fn without_config(prefix: &'a str) -> Self {
        Self::new(prefix, DeclaredLogicalShapes::NotInScope)
    }

    /// Whether this mapping declares logical shapes at all. The loader reads it to refuse a
    /// block-padded import up front — with a message about the *architecture config*, which is what
    /// the caller can actually fix — rather than letting the generic per-tensor refusal fire 224
    /// times at materialization.
    pub const fn declares_logical_shapes(&self) -> bool {
        matches!(self.shapes, DeclaredLogicalShapes::FromConfig(_))
    }
}

impl LogicalKeyMapping for KreaNativeToDiffusersMapping<'_> {
    fn mapping_id(&self) -> &'static str {
        Self::MAPPING_ID
    }

    fn logical_key(&self, physical_key: &str) -> Option<String> {
        let bare = physical_key.strip_prefix(self.prefix)?;
        native_dit_key_to_diffusers(bare)
    }

    fn logical_shape(&self, logical_key: &str) -> Option<Vec<usize>> {
        match self.shapes {
            DeclaredLogicalShapes::NotInScope => None,
            DeclaredLogicalShapes::FromConfig(cfg) => diffusers_logical_shape(cfg, logical_key),
        }
    }
}

/// Translate one **bare** (prefix already stripped) native-mmdit DiT key to its diffusers key.
///
/// `None` for anything that is not a recognized Krea DiT tensor — the plan compiler turns that into
/// a typed `UnmappedKey` refusal naming the tensor, so a foreign/renamed tensor is never skipped.
///
/// This is the exact inverse of [`convrot_diffusers_to_native`], and is *derived* from it rather
/// than transcribed: the correspondence has exactly one authority in this crate.
pub fn native_dit_key_to_diffusers(bare_key: &str) -> Option<String> {
    // Top-level (non-block) tensors: a small closed set, inverted by search over the forward map's
    // own domain so the two can never drift apart.
    for diffusers in TOP_LEVEL_DIFFUSERS_KEYS {
        if convrot_diffusers_to_native(diffusers).as_deref() == Some(bare_key) {
            return Some((*diffusers).to_string());
        }
    }

    // `blocks.N.<native-leaf>` → `transformer_blocks.N.<diffusers-leaf>`.
    if let Some(rest) = bare_key.strip_prefix("blocks.") {
        let (index, native_leaf) = split_index(rest)?;
        return invert_leaf(native_leaf).map(|leaf| format!("transformer_blocks.{index}.{leaf}"));
    }

    // `txtfusion.{layerwise,refiner}_blocks.N.<native-leaf>` → `text_fusion.{…}.N.<leaf>`.
    if let Some(rest) = bare_key.strip_prefix("txtfusion.") {
        for kind in ["layerwise_blocks.", "refiner_blocks."] {
            let Some(after) = rest.strip_prefix(kind) else {
                continue;
            };
            let (index, native_leaf) = split_index(after)?;
            return invert_leaf(native_leaf)
                .map(|leaf| format!("text_fusion.{kind}{index}.{leaf}"));
        }
    }

    None
}

/// Every non-block diffusers key the architecture contains — the domain
/// [`native_dit_key_to_diffusers`] inverts the forward map over. Block leaves are handled
/// separately because they are index-parameterized.
const TOP_LEVEL_DIFFUSERS_KEYS: &[&str] = &[
    "img_in.weight",
    "img_in.bias",
    "txt_in.norm.weight",
    "txt_in.linear_1.weight",
    "txt_in.linear_1.bias",
    "txt_in.linear_2.weight",
    "txt_in.linear_2.bias",
    "time_embed.linear_1.weight",
    "time_embed.linear_1.bias",
    "time_embed.linear_2.weight",
    "time_embed.linear_2.bias",
    "time_mod_proj.weight",
    "time_mod_proj.bias",
    "text_fusion.projector.weight",
    "final_layer.linear.weight",
    "final_layer.linear.bias",
    "final_layer.norm.weight",
    "final_layer.scale_shift_table",
];

/// Every per-block diffusers leaf the architecture contains — the domain the block arms invert the
/// forward map over.
const BLOCK_DIFFUSERS_LEAVES: &[&str] = &[
    "attn.norm_q.weight",
    "attn.norm_k.weight",
    "attn.to_q.weight",
    "attn.to_k.weight",
    "attn.to_v.weight",
    "attn.to_out.0.weight",
    "attn.to_gate.weight",
    "ff.gate.weight",
    "ff.up.weight",
    "ff.down.weight",
    "norm1.weight",
    "norm2.weight",
    "scale_shift_table",
];

/// Invert the per-block leaf half of [`convrot_diffusers_to_native`]. The forward map spells block
/// leaves only inside a `transformer_blocks.N.` key, so probe it at a fixed index and strip that
/// index back off — the leaf correspondence is index-independent by construction.
fn invert_leaf(native_leaf: &str) -> Option<&'static str> {
    const PROBE: &str = "transformer_blocks.0.";
    for leaf in BLOCK_DIFFUSERS_LEAVES {
        let native = convrot_diffusers_to_native(&format!("{PROBE}{leaf}"))?;
        if native.strip_prefix("blocks.0.") == Some(native_leaf) {
            return Some(leaf);
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

/// The architecture's true (unpadded) shape for one **diffusers** logical key, derived entirely
/// from `cfg` — the shape the [`Krea2Transformer`](crate::transformer::Krea2Transformer) module
/// tree constructs for that key and [`crate::convert::validate_native_transformer`] enforces.
///
/// `None` for any key the architecture does not contain, including a recognized leaf on a block
/// index beyond the config's stack depth. Nothing is guessed.
///
/// Linear weights are `[out, in]`. All widths come from `cfg`: image stream `hidden_size` with GQA
/// [`Krea2Config::q_dim`] / [`Krea2Config::kv_dim`] and FFN `intermediate_size`; text-fusion stream
/// `text_hidden_dim` with `text_num_{attention,kv}_heads · attention_head_dim` and
/// `text_intermediate_size`; per-head QK RMSNorm scales `[attention_head_dim]`; modulation tables
/// `[MOD_FACTORS, hidden]` per block and `[FINAL_MOD_FACTORS, hidden]` at the output layer.
pub fn diffusers_logical_shape(cfg: &Krea2Config, logical_key: &str) -> Option<Vec<usize>> {
    let hidden = cfg.hidden_size;
    let text = cfg.text_hidden_dim;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::expected_transformer_keys;
    use std::collections::BTreeSet;

    /// A deliberately *non*-published architecture: every derived width is a different number, so a
    /// shape rule that reached for the wrong config field (or hardcoded a Turbo dimension) produces
    /// a visibly wrong answer instead of accidentally matching.
    fn asymmetric_config() -> Krea2Config {
        Krea2Config {
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
        }
    }

    /// **The mapping is the exact inverse of the crate's authoritative forward map.** Driven by
    /// `expected_transformer_keys` — every diffusers key the module tree loads — so a leaf added to
    /// the architecture without a mapping entry fails here rather than at a customer's import.
    #[test]
    fn mapping_is_the_exact_inverse_of_the_authoritative_forward_map() {
        let cfg = Krea2Config::turbo();
        let mapping = KreaNativeToDiffusersMapping::without_config(
            KreaNativeToDiffusersMapping::DIFFUSION_MODEL_PREFIX,
        );
        let mut round_tripped = 0usize;
        for diffusers in expected_transformer_keys(&cfg) {
            let native = convrot_diffusers_to_native(&diffusers)
                .unwrap_or_else(|| panic!("no native key for {diffusers}"));
            let physical = format!(
                "{}{native}",
                KreaNativeToDiffusersMapping::DIFFUSION_MODEL_PREFIX
            );
            assert_eq!(
                mapping.logical_key(&physical).as_deref(),
                Some(diffusers.as_str()),
                "native `{physical}` must map back to `{diffusers}`"
            );
            round_tripped += 1;
        }
        assert_eq!(
            round_tripped,
            expected_transformer_keys(&cfg).len(),
            "every architecture key must round-trip"
        );
    }

    /// **The inverse is injective over the real architecture** — no two native keys collapse onto
    /// one diffusers key, which is what would let two tensors silently overwrite each other.
    #[test]
    fn mapping_is_injective_over_the_real_architecture() {
        let cfg = Krea2Config::turbo();
        let mapping = KreaNativeToDiffusersMapping::without_config("");
        let natives: Vec<String> = expected_transformer_keys(&cfg)
            .iter()
            .map(|k| convrot_diffusers_to_native(k).expect("forward map covers the architecture"))
            .collect();
        let logical: BTreeSet<String> = natives
            .iter()
            .filter_map(|native| mapping.logical_key(native))
            .collect();
        assert_eq!(
            logical.len(),
            natives.len(),
            "collision: {} native keys collapsed to {} logical keys",
            natives.len(),
            logical.len()
        );
    }

    /// **A key outside the dialect is unmapped, not guessed** — including a key that is missing the
    /// file's detected namespace prefix, which is exactly how a wrong-prefix plan would otherwise
    /// half-succeed.
    #[test]
    fn foreign_and_wrong_prefix_keys_are_unmapped() {
        let mapping = KreaNativeToDiffusersMapping::without_config(
            KreaNativeToDiffusersMapping::DIFFUSION_MODEL_PREFIX,
        );
        for key in [
            "model.diffusion_model.blocks.0.attn.bogus",
            "model.diffusion_model.blocks.x.attn.wq.weight",
            "some.foreign.tensor",
            // Correct native key, but without the prefix this file was detected to use.
            "blocks.0.attn.wq.weight",
        ] {
            assert_eq!(mapping.logical_key(key), None, "`{key}` must not map");
        }
        // The same bare key DOES map under a mapping built for the bare-key layout.
        assert_eq!(
            KreaNativeToDiffusersMapping::without_config("")
                .logical_key("blocks.0.attn.wq.weight")
                .as_deref(),
            Some("transformer_blocks.0.attn.to_q.weight")
        );
    }

    /// **With a config in scope every key the module tree loads has a declared shape**, and with no
    /// config none does. The first half is what lets a padded NVFP4/MXFP8 import materialize at all.
    #[test]
    fn declared_shapes_cover_the_architecture_only_with_a_config() {
        let cfg = asymmetric_config();
        let declared = KreaNativeToDiffusersMapping::for_config("", &cfg);
        let undeclared = KreaNativeToDiffusersMapping::without_config("");
        assert!(declared.declares_logical_shapes());
        assert!(!undeclared.declares_logical_shapes());
        for key in expected_transformer_keys(&cfg) {
            assert!(
                declared.logical_shape(&key).is_some(),
                "`{key}` is loaded by the module tree but has no declared logical shape"
            );
            assert_eq!(
                undeclared.logical_shape(&key),
                None,
                "`{key}` must not be declared with no config in scope"
            );
        }
    }

    /// **The shapes are DERIVED from the config, not hardcoded** — two architectures declare
    /// different geometry for the same key, and each half of the GQA/text-stream split is checked
    /// against that config's own widths.
    #[test]
    fn declared_shapes_track_the_config_rather_than_a_published_model() {
        let tiny = asymmetric_config();
        let turbo = Krea2Config::turbo();
        let a = KreaNativeToDiffusersMapping::for_config("", &tiny);
        let b = KreaNativeToDiffusersMapping::for_config("", &turbo);

        assert_eq!(
            a.logical_shape("transformer_blocks.0.attn.to_q.weight"),
            Some(vec![tiny.q_dim(), tiny.hidden_size])
        );
        assert_eq!(
            a.logical_shape("transformer_blocks.0.attn.to_k.weight"),
            Some(vec![tiny.kv_dim(), tiny.hidden_size])
        );
        assert_ne!(
            a.logical_shape("transformer_blocks.0.attn.to_q.weight"),
            b.logical_shape("transformer_blocks.0.attn.to_q.weight"),
            "two architectures must not declare the same shape"
        );
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
        assert_eq!(
            a.logical_shape("final_layer.scale_shift_table"),
            Some(vec![FINAL_MOD_FACTORS, tiny.hidden_size])
        );
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
        assert_eq!(
            a.logical_shape("transformer_blocks.0.attn.norm_q.weight"),
            Some(vec![tiny.attention_head_dim])
        );
    }

    /// **Nothing outside the architecture is declared** — a foreign key, an unrecognized leaf, and
    /// a block index past the config's stack depth all return `None` rather than a guessed shape.
    #[test]
    fn keys_outside_the_architecture_are_not_declared() {
        let cfg = asymmetric_config();
        let mapping = KreaNativeToDiffusersMapping::for_config("", &cfg);
        for key in [
            "foreign.weight",
            "transformer_blocks.0.attn.bogus.weight",
            "transformer_blocks.x.attn.to_q.weight",
            "text_fusion.unknown_blocks.0.norm1.weight",
        ] {
            assert_eq!(mapping.logical_shape(key), None, "`{key}` must not declare");
        }
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

    /// **The registered dialect id is the one both engines implement.** A rename on either side is
    /// a registry lie; this pins the Candle half against the literal the `KREA_2_CHECKPOINT_ADAPTER`
    /// row declares.
    #[test]
    fn mapping_id_is_the_registered_krea_native_dialect_id() {
        let mapping = KreaNativeToDiffusersMapping::without_config("");
        assert_eq!(mapping.mapping_id(), "krea-native-to-diffusers-v1");
        assert_eq!(
            mapping.mapping_id(),
            KreaNativeToDiffusersMapping::MAPPING_ID
        );
    }
}
