//! **The Klein role table** (sc-11045 fix round, epic E5): every projection site of the FLUX.2
//! Klein MMDiT the single-file NVFP4 import can serve, named **structurally** — the flux2 twin of
//! `candle-gen-krea`'s `KreaSite`/`LayerRole`/`ExecutionRole` pattern (sc-12121), adapted to the
//! MMDiT joint-attention topology.
//!
//! # Why a role table and not `ActPrecision::for_outlier_layer`
//!
//! The shared substring heuristic was tuned on SANA/diffusers spellings and mis-fires on klein's
//! diffusers key surface — exactly the defect class sc-12121 removed from Krea and the sc-11045
//! feature-end review found re-introduced here:
//!
//! * the **last** DiT block is unguarded (`last_block`/`final_block` never match
//!   `single_transformer_blocks.{N-1}.`), while `blocks.0.` fires only on the two index-0 tables;
//! * the **post-nonlinearity** class is unguarded: `ff.linear_out` reads the gated
//!   `act(in₁)·in₂` product and `attn.to_out.0`/`attn.to_out` read the attention output — the
//!   sc-12110 class that measured Dense in 28/28 Krea blocks — yet no anchor names them;
//! * the **context-reading** class is unguarded: klein has no `attn2`/`caption_projection` — its
//!   caption surface is `context_embedder` plus the whole txt stream of the double blocks
//!   (`add_{q,k,v}_proj`, `to_add_out`, `ff_context.*`).
//!
//! [`KleinSite::classify`] is the single parse from a dotted key to a site and is **total by
//! refusal**: an unrecognized key is [`KleinDenseReason::Unclassified`] — dense BF16, never the
//! packed lane by default (the sc-12140 rule).
//!
//! # Provenance of the class assignments
//!
//! Klein has no per-layer activation measurement yet, so every dense class here is **structural**,
//! carried over from the classes Krea *measured* (sc-12110) where the same structure exists:
//! post-nonlinearity inputs (gated-FF outputs, attention outputs, SiLU-conditioned modulation
//! reads), text-encoder-derived context reads, the trunk head, and the edge blocks. The edge set
//! is a superset of what the old heuristic guarded (it adds the last single block, the review's
//! named miss). A projection is only [`KleinExecutionRole::PackedW4A4`] when nothing structural is
//! known against it — normalized block-input reads in the interior compute bulk.

use candle_gen::quant::ActPrecision;

use crate::config::Flux2Config;

/// One of the twelve GEMM leaves of a klein **double** (joint img+txt) block, as spelled in the
/// diffusers logical schema under `transformer_blocks.{i}.`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoubleLeaf {
    /// `attn.to_q` — img-stream query, reads the normalized img block input.
    ToQ,
    /// `attn.to_k` — img-stream key.
    ToK,
    /// `attn.to_v` — img-stream value.
    ToV,
    /// `attn.add_q_proj` — txt-stream query, reads the normalized **text** stream.
    AddQ,
    /// `attn.add_k_proj` — txt-stream key.
    AddK,
    /// `attn.add_v_proj` — txt-stream value.
    AddV,
    /// `attn.to_out.0` — img-stream output projection, reads the joint-attention output.
    ToOut,
    /// `attn.to_add_out` — txt-stream output projection, reads the joint-attention output.
    ToAddOut,
    /// `ff.linear_in` — img-stream gated-FF ingest, reads the normalized img block input.
    FfIn,
    /// `ff.linear_out` — img-stream gated-FF output, reads the `act(in₁)·in₂` gated product.
    FfOut,
    /// `ff_context.linear_in` — txt-stream gated-FF ingest.
    FfCtxIn,
    /// `ff_context.linear_out` — txt-stream gated-FF output.
    FfCtxOut,
}

impl DoubleLeaf {
    /// Every leaf, in the order the block loads them — the coverage test's enumeration source.
    pub const ALL: [Self; 12] = [
        Self::ToQ,
        Self::ToK,
        Self::ToV,
        Self::AddQ,
        Self::AddK,
        Self::AddV,
        Self::ToOut,
        Self::ToAddOut,
        Self::FfIn,
        Self::FfOut,
        Self::FfCtxIn,
        Self::FfCtxOut,
    ];

    /// The block-relative dotted suffix this leaf is spelled as.
    pub fn leaf_key(self) -> &'static str {
        match self {
            Self::ToQ => "attn.to_q",
            Self::ToK => "attn.to_k",
            Self::ToV => "attn.to_v",
            Self::AddQ => "attn.add_q_proj",
            Self::AddK => "attn.add_k_proj",
            Self::AddV => "attn.add_v_proj",
            Self::ToOut => "attn.to_out.0",
            Self::ToAddOut => "attn.to_add_out",
            Self::FfIn => "ff.linear_in",
            Self::FfOut => "ff.linear_out",
            Self::FfCtxIn => "ff_context.linear_in",
            Self::FfCtxOut => "ff_context.linear_out",
        }
    }

    fn from_leaf_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|leaf| leaf.leaf_key() == key)
    }

    /// True iff this leaf reads the **text stream** — klein's caption-derived context surface,
    /// the analogue of the class Krea measured Dense at crush up to 40145× (sc-12110).
    ///
    /// Exhaustive on purpose: a newly added leaf must answer this question to compile, never
    /// default into the packed lane.
    pub fn reads_text_stream(self) -> bool {
        match self {
            Self::AddQ
            | Self::AddK
            | Self::AddV
            | Self::ToAddOut
            | Self::FfCtxIn
            | Self::FfCtxOut => true,
            Self::ToQ | Self::ToK | Self::ToV | Self::ToOut | Self::FfIn | Self::FfOut => false,
        }
    }

    /// True iff this leaf's **input** is a post-nonlinearity intermediate rather than a normalized
    /// block input — sc-12110's central partition finding, applied to klein's structure: the gated
    /// FF outputs read `act(in₁)·in₂` and both attention output projections read the attention
    /// output. Exhaustive for the same reason as [`Self::reads_text_stream`].
    pub fn reads_post_nonlinearity(self) -> bool {
        match self {
            Self::ToOut | Self::ToAddOut | Self::FfOut | Self::FfCtxOut => true,
            Self::ToQ
            | Self::ToK
            | Self::ToV
            | Self::AddQ
            | Self::AddK
            | Self::AddV
            | Self::FfIn
            | Self::FfCtxIn => false,
        }
    }
}

/// One of the two GEMM leaves of a klein **single** (fused) block, under
/// `single_transformer_blocks.{i}.`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SingleLeaf {
    /// `attn.to_qkv_mlp_proj` — the fused QKV+MLP ingest, reads the normalized block input.
    QkvMlp,
    /// `attn.to_out` — the fused output projection, reads `concat(attn_out, act(mlp))` — a
    /// post-nonlinearity intermediate on its MLP half.
    Out,
}

impl SingleLeaf {
    /// Every leaf, in load order.
    pub const ALL: [Self; 2] = [Self::QkvMlp, Self::Out];

    /// The block-relative dotted suffix this leaf is spelled as.
    pub fn leaf_key(self) -> &'static str {
        match self {
            Self::QkvMlp => "attn.to_qkv_mlp_proj",
            Self::Out => "attn.to_out",
        }
    }

    fn from_leaf_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|leaf| leaf.leaf_key() == key)
    }

    /// See [`DoubleLeaf::reads_post_nonlinearity`]; exhaustive for the same reason.
    pub fn reads_post_nonlinearity(self) -> bool {
        match self {
            Self::Out => true,
            Self::QkvMlp => false,
        }
    }
}

/// Every projection site of the klein trunk the single-file import serves through
/// `PlannedDitWeights::qlinear` — the sites mirror the loader's call graph
/// (`Flux2Transformer::new_planned`), which is what lets the coverage test enumerate the whole
/// surface from this enum and assert it round-trips.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KleinSite {
    /// `x_embedder` — the image-token ingest (Krea's `img_in`, measured perfectly Benign there).
    ImageIngest,
    /// `context_embedder` — projects the **raw Qwen3 TE hidden states** into DiT width: the
    /// massive-activation source itself, and the review's named unguarded key.
    ContextIngest,
    /// `time_guidance_embed.timestep_embedder.linear_1` — reads the bounded sinusoidal embedding.
    TimeEmbedIn,
    /// `time_guidance_embed.timestep_embedder.linear_2` — reads `silu(linear_1(...))`.
    TimeEmbedOut,
    /// `double_stream_modulation_{img,txt}.linear` / `single_stream_modulation.linear` — the AdaLN
    /// modulation tables, reading the SiLU-conditioned embedding.
    Modulation,
    /// `norm_out.linear` — the output AdaLN, reading the SiLU-conditioned embedding.
    NormOut,
    /// `proj_out` — the trunk head (Krea/SANA's measured-Dense head class, crush 909×/438×).
    TrunkHead,
    /// `transformer_blocks.{index}.{leaf}` — a double (joint) block projection.
    Double { index: usize, leaf: DoubleLeaf },
    /// `single_transformer_blocks.{index}.{leaf}` — a single (fused) block projection.
    Single { index: usize, leaf: SingleLeaf },
}

impl KleinSite {
    /// The site `name` (a diffusers dotted key **without** the `.weight` suffix) denotes, or
    /// `None` when the role table does not name it. Structural throughout: fixed keys match whole,
    /// block keys are parsed (prefix, integer index, one of the leaf suffixes) — never sniffed by
    /// substring.
    pub fn classify(name: &str) -> Option<Self> {
        match name {
            "x_embedder" => return Some(Self::ImageIngest),
            "context_embedder" => return Some(Self::ContextIngest),
            "time_guidance_embed.timestep_embedder.linear_1" => return Some(Self::TimeEmbedIn),
            "time_guidance_embed.timestep_embedder.linear_2" => return Some(Self::TimeEmbedOut),
            "double_stream_modulation_img.linear"
            | "double_stream_modulation_txt.linear"
            | "single_stream_modulation.linear" => return Some(Self::Modulation),
            "norm_out.linear" => return Some(Self::NormOut),
            "proj_out" => return Some(Self::TrunkHead),
            _ => {}
        }
        if let Some(rest) = name.strip_prefix("transformer_blocks.") {
            let (index, leaf) = rest.split_once('.')?;
            return Some(Self::Double {
                index: index.parse().ok()?,
                leaf: DoubleLeaf::from_leaf_key(leaf)?,
            });
        }
        let rest = name.strip_prefix("single_transformer_blocks.")?;
        let (index, leaf) = rest.split_once('.')?;
        Some(Self::Single {
            index: index.parse().ok()?,
            leaf: SingleLeaf::from_leaf_key(leaf)?,
        })
    }
}

/// Why a klein projection is served dense BF16 rather than packed W4A4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KleinDenseReason {
    /// Reads text-encoder-derived context (`context_embedder`, the double blocks' txt stream).
    ContextRead,
    /// Reads a post-nonlinearity intermediate (gated-FF outputs, attention outputs, the
    /// SiLU-conditioned modulation/AdaLN reads).
    PostNonlinearity,
    /// Sits in the first double block, or the first/last single block — the edges where
    /// caption-derived activations still reach the block inputs. A superset of what the old
    /// substring policy guarded: `blocks.0.` fired on the two index-0 tables, but nothing ever
    /// matched the last single block.
    EdgeBlock,
    /// The trunk head (`proj_out`).
    TrunkHead,
    /// The role table does not name this key at all — dense by default, never packed by a missed
    /// guard (sc-12140).
    Unclassified,
}

/// The execution role the role table assigns a klein projection — which of the two things
/// `Nvfp4Linear` can be it should be, **before** any device/checkpoint capability fact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KleinExecutionRole {
    /// Packed W4A4: the weight stays in its NVFP4 container and the cuBLASLt FP4 GEMM runs.
    PackedW4A4,
    /// Dense BF16 (W4A16) for the stated structural reason.
    DenseBf16(KleinDenseReason),
}

impl KleinExecutionRole {
    /// True iff this role is the packed W4A4 lane.
    pub fn is_packed_w4a4(self) -> bool {
        matches!(self, Self::PackedW4A4)
    }

    /// The activation precision this role asks [`candle_gen::quant::Nvfp4Linear`] for.
    pub fn act_precision(self) -> ActPrecision {
        match self {
            Self::PackedW4A4 => ActPrecision::W4A4,
            Self::DenseBf16(_) => ActPrecision::W4A16,
        }
    }
}

/// The role table bound to one klein trunk's block counts — the single source of the topology
/// facts (`is_edge_block` needs to name the **last** single block).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KleinRoleTable {
    num_double_layers: usize,
    num_single_layers: usize,
}

impl KleinRoleTable {
    /// Bind the table to the architecture the mapping declares.
    pub fn new(cfg: &Flux2Config) -> Self {
        Self {
            num_double_layers: cfg.num_double_layers,
            num_single_layers: cfg.num_single_layers,
        }
    }

    /// The number of double blocks this table is bound to.
    pub fn num_double_layers(&self) -> usize {
        self.num_double_layers
    }

    /// The number of single blocks this table is bound to.
    pub fn num_single_layers(&self) -> usize {
        self.num_single_layers
    }

    /// The execution role of the projection named by the diffusers dotted key `name` (without a
    /// `.weight` suffix). Total: an unnamed key is dense by [`KleinDenseReason::Unclassified`].
    pub fn execution_role(&self, name: &str) -> KleinExecutionRole {
        match KleinSite::classify(name) {
            Some(site) => self.role_for_site(site),
            None => KleinExecutionRole::DenseBf16(KleinDenseReason::Unclassified),
        }
    }

    /// The role a [`KleinSite`] carries on this trunk. Exhaustive over the enum, so adding a site
    /// is a compile error until its class is stated.
    ///
    /// Precedence is reporting order only — every named class selects dense BF16, so which is
    /// named cannot change the representation.
    pub fn role_for_site(&self, site: KleinSite) -> KleinExecutionRole {
        let dense = KleinExecutionRole::DenseBf16;
        match site {
            KleinSite::ImageIngest | KleinSite::TimeEmbedIn => KleinExecutionRole::PackedW4A4,
            KleinSite::ContextIngest => dense(KleinDenseReason::ContextRead),
            KleinSite::TimeEmbedOut | KleinSite::Modulation | KleinSite::NormOut => {
                dense(KleinDenseReason::PostNonlinearity)
            }
            KleinSite::TrunkHead => dense(KleinDenseReason::TrunkHead),
            KleinSite::Double { index, leaf } => {
                if leaf.reads_text_stream() {
                    dense(KleinDenseReason::ContextRead)
                } else if leaf.reads_post_nonlinearity() {
                    dense(KleinDenseReason::PostNonlinearity)
                } else if index == 0 {
                    dense(KleinDenseReason::EdgeBlock)
                } else {
                    KleinExecutionRole::PackedW4A4
                }
            }
            KleinSite::Single { index, leaf } => {
                if leaf.reads_post_nonlinearity() {
                    dense(KleinDenseReason::PostNonlinearity)
                } else if index == 0 || index + 1 == self.num_single_layers.max(1) {
                    dense(KleinDenseReason::EdgeBlock)
                } else {
                    KleinExecutionRole::PackedW4A4
                }
            }
        }
    }

    /// Every projection base key of the trunk this table is bound to, in loader order — the same
    /// surface `Flux2Transformer::new_planned` constructs through `qlinear`, enumerated from the
    /// role table so the coverage test can assert the two agree without a second spelling.
    pub fn all_projection_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = vec![
            "x_embedder".to_owned(),
            "context_embedder".to_owned(),
            "time_guidance_embed.timestep_embedder.linear_1".to_owned(),
            "time_guidance_embed.timestep_embedder.linear_2".to_owned(),
            "double_stream_modulation_img.linear".to_owned(),
            "double_stream_modulation_txt.linear".to_owned(),
            "single_stream_modulation.linear".to_owned(),
            "norm_out.linear".to_owned(),
            "proj_out".to_owned(),
        ];
        for index in 0..self.num_double_layers {
            for leaf in DoubleLeaf::ALL {
                keys.push(format!("transformer_blocks.{index}.{}", leaf.leaf_key()));
            }
        }
        for index in 0..self.num_single_layers {
            for leaf in SingleLeaf::ALL {
                keys.push(format!(
                    "single_transformer_blocks.{index}.{}",
                    leaf.leaf_key()
                ));
            }
        }
        keys
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Flux2Variant;

    fn table() -> KleinRoleTable {
        KleinRoleTable::new(&Flux2Variant::Klein9b.config())
    }

    /// **Exhaustive coverage**: every projection key the klein loader constructs classifies to a
    /// named site (nothing rides `Unclassified` on the real surface), and every classification
    /// round-trips through the parse.
    #[test]
    fn every_klein_projection_key_classifies_to_a_named_site() {
        let table = table();
        for key in table.all_projection_keys() {
            let site = KleinSite::classify(&key);
            assert!(
                site.is_some(),
                "`{key}` must classify — an unnamed real key would silently ride the \
                 Unclassified dense default and hide a topology drift"
            );
            // The role is total and never panics.
            let _ = table.execution_role(&key);
        }
    }

    /// **The review's named misses are guarded** (sc-11045 feature-end review, E5): the last
    /// single block, the post-nonlinearity outputs, and the context surface all resolve dense —
    /// and the guards are structural, not substring luck.
    #[test]
    fn the_reviews_named_misses_are_dense() {
        let table = table();
        let last = table.num_single_layers() - 1;
        for key in [
            // The unguarded last DiT block (old heuristic: `last_block` never matched).
            format!("single_transformer_blocks.{last}.attn.to_qkv_mlp_proj"),
            // Post-nonlinearity (old heuristic: no anchor fired on either spelling).
            "transformer_blocks.3.ff.linear_out".to_owned(),
            "transformer_blocks.3.attn.to_out.0".to_owned(),
            "single_transformer_blocks.7.attn.to_out".to_owned(),
            // The context surface (old heuristic: klein has no attn2/caption_projection).
            "context_embedder".to_owned(),
            "transformer_blocks.5.attn.add_k_proj".to_owned(),
            "transformer_blocks.5.ff_context.linear_out".to_owned(),
        ] {
            assert!(
                !table.execution_role(&key).is_packed_w4a4(),
                "`{key}` must be dense BF16 — this is one of the review's named unsafe-W4A4 keys"
            );
            assert_eq!(
                table.execution_role(&key).act_precision(),
                ActPrecision::W4A16
            );
        }
    }

    /// The interior compute bulk stays on the packed lane — the layers the throughput case rides.
    #[test]
    fn interior_block_input_reads_stay_packed() {
        let table = table();
        for key in [
            "transformer_blocks.3.attn.to_q",
            "transformer_blocks.5.ff.linear_in",
            "single_transformer_blocks.7.attn.to_qkv_mlp_proj",
            "x_embedder",
        ] {
            assert!(
                table.execution_role(key).is_packed_w4a4(),
                "`{key}` reads a normalized block input in the interior and stays W4A4"
            );
            assert_eq!(
                table.execution_role(key).act_precision(),
                ActPrecision::W4A4
            );
        }
    }

    /// **Mutation guard: change one role → red.** Pins the exact partition over one double and
    /// one single interior block plus every fixed site, so a role-table edit cannot pass unseen.
    #[test]
    fn the_partition_is_pinned_per_leaf() {
        let table = table();
        let expect_dense = |key: &str, reason: KleinDenseReason| {
            assert_eq!(
                table.execution_role(key),
                KleinExecutionRole::DenseBf16(reason),
                "`{key}`"
            );
        };
        // Fixed sites.
        assert!(table.execution_role("x_embedder").is_packed_w4a4());
        assert!(table
            .execution_role("time_guidance_embed.timestep_embedder.linear_1")
            .is_packed_w4a4());
        expect_dense("context_embedder", KleinDenseReason::ContextRead);
        expect_dense(
            "time_guidance_embed.timestep_embedder.linear_2",
            KleinDenseReason::PostNonlinearity,
        );
        expect_dense(
            "double_stream_modulation_img.linear",
            KleinDenseReason::PostNonlinearity,
        );
        expect_dense(
            "double_stream_modulation_txt.linear",
            KleinDenseReason::PostNonlinearity,
        );
        expect_dense(
            "single_stream_modulation.linear",
            KleinDenseReason::PostNonlinearity,
        );
        expect_dense("norm_out.linear", KleinDenseReason::PostNonlinearity);
        expect_dense("proj_out", KleinDenseReason::TrunkHead);
        // An interior double block: img-stream input reads packed, txt stream and
        // post-nonlinearity dense.
        for (leaf, packed) in [
            ("attn.to_q", true),
            ("attn.to_k", true),
            ("attn.to_v", true),
            ("ff.linear_in", true),
            ("attn.add_q_proj", false),
            ("attn.add_k_proj", false),
            ("attn.add_v_proj", false),
            ("attn.to_add_out", false),
            ("ff_context.linear_in", false),
            ("ff_context.linear_out", false),
            ("attn.to_out.0", false),
            ("ff.linear_out", false),
        ] {
            assert_eq!(
                table
                    .execution_role(&format!("transformer_blocks.4.{leaf}"))
                    .is_packed_w4a4(),
                packed,
                "transformer_blocks.4.{leaf}"
            );
        }
        // Edges: the first double block and the first/last single block guard their
        // block-input reads too.
        expect_dense(
            "transformer_blocks.0.attn.to_q",
            KleinDenseReason::EdgeBlock,
        );
        expect_dense(
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj",
            KleinDenseReason::EdgeBlock,
        );
        let last = table.num_single_layers() - 1;
        expect_dense(
            &format!("single_transformer_blocks.{last}.attn.to_qkv_mlp_proj"),
            KleinDenseReason::EdgeBlock,
        );
        // Unclassified keys are dense by refusal, never packed by default.
        expect_dense("some.novel.key", KleinDenseReason::Unclassified);
        expect_dense(
            "transformer_blocks.4.attn.mystery",
            KleinDenseReason::Unclassified,
        );
    }
}
