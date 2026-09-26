//! Native YuE2 precision tiers and the V2 precision policy (sc-22995, epic E8).
//!
//! Upstream publishes YuE2-3B in BF16 only and offers no weight tiers; the `bf16` / `q8` / `q4`
//! tiers here are SceneWorks engineering. A tier is a statement about the **whole** MoT: every
//! matmul weight of the model follows it — both Mixture-of-Transformers paths, `lm_head` and the
//! NAR auxiliary heads — using the workspace's one GGML tier scheme (`candle_llm`'s
//! [`QuantSpec`]: Q8 = GGML Q8_0, Q4 = GGML Q4_K with the Q4_0 fallback for an input width that
//! is 32- but not 256-aligned; stored as GGML block tensors, see [`crate::tier`]).
//!
//! # The V2 precision map
//!
//! Per tensor class ([`classify`]), what each tier stores ([`storage`]). "Compute" is the
//! activation dtype the stages run in: BF16 on an accelerator, F32 on the CPU (Candle's CPU backend
//! has no BF16 matmul, so a CPU load upcasts BF16 weights exactly); a GGML weight is multiplied by
//! Candle's quantized matmul, which quantizes the activation to its 8-bit block format on the fly.
//!
//! | class | tensors | `bf16` | `q8` | `q4` |
//! |---|---|---|---|---|
//! | [`TensorClass::ArProjection`] | `layers.N.self_attn.{q,k,v,o}_proj`, `layers.N.mlp.{gate,up,down}_proj` | BF16 | Q8_0 | Q4_K |
//! | [`TensorClass::NarProjection`] | the `nar_self_attn` / `nar_mlp` twins | BF16 | Q8_0 | Q4_K |
//! | [`TensorClass::LmHead`] | `lm_head.weight` | BF16 | Q8_0 | Q4_K |
//! | [`TensorClass::NarHead`] | `vae2llm`, `llm2vae`, `time_embedder.mlp.{0,2}` weights | BF16 | Q8_0 | Q4_K (`vae2llm`: Q4_0, its 64-wide input is not 256-aligned) |
//! | [`TensorClass::TokenEmbedding`] | `model.embed_tokens.weight` | BF16 | BF16 | BF16 |
//! | [`TensorClass::LatentPositions`] | `latent_pos_embed.pe` | BF16 | BF16 | BF16 |
//! | [`TensorClass::Norm`] | every RMSNorm weight (`*_layernorm`, `q_norm`, `k_norm`, `model.norm`) | BF16 | BF16 | BF16 |
//! | [`TensorClass::Bias`] | the NAR heads' biases | BF16 | BF16 | BF16 |
//! | VAE (both decoders) | every tensor | FP32 | FP32 | FP32 |
//!
//! ## Explicit V2 choices (owner-visible)
//!
//! These are YuE2's own decisions, recorded in [`OWNER_DECISIONS`] so none is silent. **YuE1's
//! approved xcodec / Vocos fp16 carve-out (epic sc-19373) does not carry over to YuE2**: YuE2 has
//! no xcodec or Vocos, and nothing here runs a component in FP16.
//!
//! * **VAE at FP32 at every tier.** Upstream's reference decode is FP32 (`vae_dtype: float32` in
//!   its effective configuration; epic E8), and the pinned Candle has no quantized convolution, so
//!   no `q8` / `q4` VAE target exists to measure. The decoders therefore run above a `q8` / `q4`
//!   tier — an exception to the whole-render tier contract that an owner must approve (or replace
//!   with a quantized-VAE requirement). It is [`OWNER_DECISIONS`]' `vae_fp32_at_every_tier`.
//! * **Lookup tables and vectors stay BF16.** The token embedding and the latent position table are
//!   gathered, not multiplied, and the workspace's GGML tier convention (`candle_llm::prepare`)
//!   stores embeddings and norms exactly as read; Candle has no quantized gather. Norms and biases
//!   are vectors below one GGML block. Together they are 0.86 GB of the 7.26 GB checkpoint.
//!   Recorded as `embedding_tables_bf16`.
//!
//! ## Shared AR / NAR weights
//!
//! One MoT serves every stage: the ABC and semantic stages run the AR path; the acoustic stage runs
//! the AR path over the song prefix (its keys/values) **and** the NAR twins over the latents. For
//! `bf16` / `q8` / `q4` the tier is fixed at load — the AR path the acoustic stage prefills with is
//! the same resident (quantized) weight the AR stages decoded with, and the NAR twins are quantized
//! once like every other projection — so **no restore or re-quantization happens between stages**.
//! The experimental FP8 AR mode ([`crate::fp8`]) is the one exception: it swaps the AR projections
//! for FP8 copies during the AR stages only, and restores the exact BF16 originals before the
//! acoustic stage (whose AR-path prefill must be BF16, as upstream) and before anything could save
//! the model; see that module.
//!
//! # Residency
//!
//! [`weight_residency`] prices a tier's resident weights from the checkpoint's tensor table alone
//! (weights-free), per backend, with the FP8 mode's CPU-resident BF16 originals counted separately;
//! [`crate::model::Yue2Lm::weight_residency`] measures the same thing on a loaded model.

use candle_audio::candle_core::quantized::GgmlDType;
use candle_audio::candle_core::DType;
use candle_audio::gen_core::{self, Quant};
use candle_llm::primitives::QuantSpec;

/// A native YuE2 weight tier (see the [module docs](self)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tier {
    /// The released BF16 checkpoint, loaded as is (the pinned upstream snapshot).
    Bf16,
    /// Every matmul weight GGML Q8_0 (a derived tier snapshot, [`crate::tier`]).
    Q8,
    /// Every matmul weight GGML Q4_K (Q4_0 where the input width is not 256-aligned).
    Q4,
}

impl Tier {
    /// Every tier, densest first.
    pub const ALL: [Tier; 3] = [Tier::Bf16, Tier::Q8, Tier::Q4];

    /// The tier's name (`bf16` / `q8` / `q4`), as recorded in manifests and configurations.
    pub fn name(self) -> &'static str {
        match self {
            Tier::Bf16 => "bf16",
            Tier::Q8 => "q8",
            Tier::Q4 => "q4",
        }
    }

    /// The tier named `name`.
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.name() == name)
    }

    /// The tier a `LoadSpec::quantize` asserts: `None` asserts nothing (the staged snapshot's tier
    /// is used), `Q8` / `Q4` assert those tiers. NVFP4 is not a YuE2 tier and is refused by name
    /// rather than folded onto `q4`.
    pub fn from_quant(quant: Option<Quant>) -> gen_core::Result<Option<Self>> {
        match quant {
            None => Ok(None),
            Some(Quant::Q8) => Ok(Some(Tier::Q8)),
            Some(Quant::Q4) => Ok(Some(Tier::Q4)),
            Some(other) => Err(gen_core::Error::Unsupported(format!(
                "yue2: quantize={other:?} is not a YuE2 tier; YuE2 has bf16 (the released \
                 checkpoint), q8 and q4"
            ))),
        }
    }

    /// The workspace GGML spec of a quantized tier (`None` for `bf16`).
    pub fn quant_spec(self) -> Option<QuantSpec> {
        match self {
            Tier::Bf16 => None,
            Tier::Q8 => Some(QuantSpec::q8()),
            Tier::Q4 => Some(QuantSpec::q4()),
        }
    }
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// What a checkpoint tensor is, for the precision policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorClass {
    /// `model.embed_tokens.weight` (a gathered table).
    TokenEmbedding,
    /// `latent_pos_embed.pe` (a gathered table).
    LatentPositions,
    /// An RMSNorm weight.
    Norm,
    /// A bias vector of a NAR head.
    Bias,
    /// An AR-path attention or MLP projection.
    ArProjection,
    /// A NAR-twin attention or MLP projection.
    NarProjection,
    /// `lm_head.weight`.
    LmHead,
    /// A NAR head's weight matrix (`vae2llm`, `llm2vae`, `time_embedder.mlp.{0,2}`).
    NarHead,
}

impl TensorClass {
    /// Stable lower-case label (manifests, reports).
    pub fn name(self) -> &'static str {
        match self {
            TensorClass::TokenEmbedding => "token_embedding",
            TensorClass::LatentPositions => "latent_positions",
            TensorClass::Norm => "norm",
            TensorClass::Bias => "bias",
            TensorClass::ArProjection => "ar_projection",
            TensorClass::NarProjection => "nar_projection",
            TensorClass::LmHead => "lm_head",
            TensorClass::NarHead => "nar_head",
        }
    }

    /// Whether the class is a matmul weight — the classes a quantized tier stores quantized.
    pub fn follows_tier(self) -> bool {
        matches!(
            self,
            TensorClass::ArProjection
                | TensorClass::NarProjection
                | TensorClass::LmHead
                | TensorClass::NarHead
        )
    }
}

/// The class of checkpoint tensor `name`, or `None` for a name the released architecture does not
/// have (a precision plan never guesses: an unknown tensor is an error there).
pub fn classify(name: &str) -> Option<TensorClass> {
    const AR: [&str; 7] = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ];
    const NORMS: [&str; 8] = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "nar_input_layernorm.weight",
        "nar_pre_mlp_layernorm.weight",
        "nar_self_attn.q_norm.weight",
        "nar_self_attn.k_norm.weight",
    ];
    match name {
        "model.embed_tokens.weight" => return Some(TensorClass::TokenEmbedding),
        "latent_pos_embed.pe" => return Some(TensorClass::LatentPositions),
        "model.norm.weight" => return Some(TensorClass::Norm),
        "lm_head.weight" => return Some(TensorClass::LmHead),
        "vae2llm.weight"
        | "llm2vae.weight"
        | "time_embedder.mlp.0.weight"
        | "time_embedder.mlp.2.weight" => return Some(TensorClass::NarHead),
        "vae2llm.bias"
        | "llm2vae.bias"
        | "time_embedder.mlp.0.bias"
        | "time_embedder.mlp.2.bias" => return Some(TensorClass::Bias),
        _ => {}
    }
    let rest = name.strip_prefix("model.layers.")?;
    let (index, leaf) = rest.split_once('.')?;
    if index.is_empty() || !index.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if AR.contains(&leaf) {
        Some(TensorClass::ArProjection)
    } else if leaf
        .strip_prefix("nar_")
        .is_some_and(|twin| AR.contains(&twin))
    {
        Some(TensorClass::NarProjection)
    } else if NORMS.contains(&leaf) {
        Some(TensorClass::Norm)
    } else {
        None
    }
}

/// How a tier stores one tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Storage {
    /// The released BF16 values, byte for byte.
    Bf16,
    /// A GGML block-quantized matrix, stored as a GGML block tensor ([`crate::tier`]).
    Ggml(GgmlDType),
}

impl Storage {
    /// Stable label: `bf16`, `q8_0`, `q4_k`, `q4_0`.
    pub fn label(self) -> &'static str {
        match self {
            Storage::Bf16 => "bf16",
            Storage::Ggml(GgmlDType::Q8_0) => "q8_0",
            Storage::Ggml(GgmlDType::Q4K) => "q4_k",
            Storage::Ggml(GgmlDType::Q4_0) => "q4_0",
            Storage::Ggml(_) => "ggml_other",
        }
    }

    /// The safetensors dtype and shape a `logical`-shaped tensor is stored with: the BF16 tensor
    /// itself, or a `U8` `[rows, blocks_per_row, block_bytes]` GGML block tensor.
    pub fn stored(self, logical: &[usize]) -> (&'static str, Vec<usize>) {
        match self {
            Storage::Bf16 => ("BF16", logical.to_vec()),
            Storage::Ggml(d) => {
                let cols = logical.last().copied().unwrap_or(0);
                let rows: usize = logical[..logical.len().saturating_sub(1)].iter().product();
                ("U8", vec![rows, cols / d.block_size(), d.type_size()])
            }
        }
    }

    /// Bytes of the stored payload of a `logical`-shaped tensor (no device padding).
    pub fn stored_bytes(self, logical: &[usize]) -> u64 {
        let elems: u64 = logical.iter().map(|&d| d as u64).product();
        match self {
            Storage::Bf16 => elems * 2,
            Storage::Ggml(d) => elems / d.block_size() as u64 * d.type_size() as u64,
        }
    }
}

/// How `tier` stores a `class` tensor of `logical` shape. A matmul weight of a quantized tier must
/// be a matrix whose input width divides into the GGML blocks of [`Tier::quant_spec`]'s dtype for
/// that width; anything else is an error, never a silent dense fallback.
pub fn storage(tier: Tier, class: TensorClass, logical: &[usize]) -> gen_core::Result<Storage> {
    let Some(spec) = tier.quant_spec().filter(|_| class.follows_tier()) else {
        return Ok(Storage::Bf16);
    };
    let [_, in_dim] = logical else {
        return Err(gen_core::Error::Unsupported(format!(
            "yue2 {tier}: a {} weight must be a matrix, got shape {logical:?}",
            class.name()
        )));
    };
    let dtype = spec.dtype_for_in_dim(*in_dim);
    if *in_dim == 0 || !in_dim.is_multiple_of(dtype.block_size()) {
        return Err(gen_core::Error::Unsupported(format!(
            "yue2 {tier}: a {} weight with input width {in_dim} does not divide into {dtype:?} \
             blocks of {}",
            class.name(),
            dtype.block_size()
        )));
    }
    Ok(Storage::Ggml(dtype))
}

/// One tensor of a tier's precision plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedTensor {
    /// The checkpoint name.
    pub name: String,
    /// Its class.
    pub class: TensorClass,
    /// The released (logical) shape.
    pub logical_shape: Vec<usize>,
    /// How the tier stores it.
    pub storage: Storage,
}

/// The precision plan of `tier` over a released checkpoint's tensor table (`name`, BF16 shape):
/// every tensor classified ([`classify`]) and assigned its [`storage`]. A tensor the released
/// architecture does not have, or one that is not BF16 in the source, is an error.
pub fn plan<'a>(
    source: impl IntoIterator<Item = (&'a str, &'a str, &'a [usize])>,
    tier: Tier,
) -> gen_core::Result<Vec<PlannedTensor>> {
    let mut out = Vec::new();
    for (name, dtype, shape) in source {
        let class = classify(name).ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "yue2 {tier}: `{name}` is not a tensor of the released YuE2 architecture"
            ))
        })?;
        if dtype != "BF16" {
            return Err(gen_core::Error::Unsupported(format!(
                "yue2 {tier}: `{name}` is {dtype} in the source; tiers are derived from the \
                 released BF16 checkpoint only"
            )));
        }
        out.push(PlannedTensor {
            name: name.to_string(),
            class,
            logical_shape: shape.to_vec(),
            storage: storage(tier, class, shape)?,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// The VAE precision at every tier.
pub const VAE_DTYPE: &str = "float32";

/// The backend a residency is priced for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Candle CPU: dense weights held in F32 (no BF16 matmul), GGML blocks as stored.
    Cpu,
    /// Candle CUDA: dense weights in the compute dtype, GGML blocks plus Candle's CUDA row padding.
    Cuda,
    /// Candle Metal: dense weights in the compute dtype, GGML blocks as stored.
    Metal,
}

/// Resident weight bytes of a loaded MoT.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Residency {
    /// Bytes on the model device (host memory for a CPU model).
    pub device_bytes: u64,
    /// Bytes held in host memory **in addition** to `device_bytes` — the FP8 mode's retained BF16
    /// originals ([`crate::fp8`]). Always 0 without FP8.
    pub host_bytes: u64,
}

impl Residency {
    /// Everything the weights occupy.
    pub fn total(&self) -> u64 {
        self.device_bytes + self.host_bytes
    }
}

/// Candle's CUDA `MATRIX_ROW_PADDING` (elements): every GGML tensor Candle builds on a CUDA device
/// carries this many elements' worth of extra blocks (`candle_core::quantized::cuda`).
const CUDA_GGML_ROW_PADDING: u64 = 512;

/// Bytes a loaded `class` tensor of `logical` shape stored as `stored` occupies on `backend`
/// computing in `compute`.
pub fn loaded_bytes(stored: Storage, logical: &[usize], backend: Backend, compute: DType) -> u64 {
    let elems: u64 = logical.iter().map(|&d| d as u64).product();
    match stored {
        Storage::Bf16 => elems * compute.size_in_bytes() as u64,
        Storage::Ggml(d) => {
            let payload = stored.stored_bytes(logical);
            match backend {
                Backend::Cuda => {
                    payload + CUDA_GGML_ROW_PADDING * d.type_size() as u64 / d.block_size() as u64
                }
                Backend::Cpu | Backend::Metal => payload,
            }
        }
    }
}

/// Price the resident weights of `plan` (both MoT paths and the NAR heads — what the engine loads)
/// on `backend` in compute dtype `compute`. With `fp8_ar`, the AR projections are held on the
/// device as FP8 E4M3 plus one F32 scale each, and their BF16 originals in host memory
/// ([`Residency::host_bytes`]) — the upstream FP8 mode's layout ([`crate::fp8`]).
pub fn weight_residency(
    plan: &[PlannedTensor],
    backend: Backend,
    compute: DType,
    fp8_ar: bool,
) -> Residency {
    let mut r = Residency::default();
    for t in plan {
        let elems: u64 = t.logical_shape.iter().map(|&d| d as u64).product();
        if fp8_ar && t.class == TensorClass::ArProjection {
            r.device_bytes += elems + 4;
            r.host_bytes += elems * 2;
        } else {
            r.device_bytes += loaded_bytes(t.storage, &t.logical_shape, backend, compute);
        }
    }
    r
}

/// Whether an owner-visible precision / distribution item is settled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionStatus {
    /// Implemented as described; recorded so it is visible, no owner action needed.
    Recorded,
    /// Needs an owner decision; the implementation holds the stated interim behaviour.
    Unresolved,
}

/// An owner-visible precision or distribution item of the V2 policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwnerDecision {
    /// Stable id.
    pub id: &'static str,
    /// Status.
    pub status: DecisionStatus,
    /// What holds now, and what the owner would decide.
    pub summary: &'static str,
}

/// Every owner-visible item of the YuE2 precision policy (sc-22995). None is silently dropped: an
/// unsupported target or proposed exception is listed here as [`DecisionStatus::Unresolved`].
pub const OWNER_DECISIONS: &[OwnerDecision] = &[
    OwnerDecision {
        id: "vae_fp32_at_every_tier",
        status: DecisionStatus::Unresolved,
        summary: "Both VAE decoders run FP32 at bf16, q8 and q4 (upstream's reference decode is \
                  FP32; the pinned Candle has no quantized convolution, so no q8/q4 VAE target \
                  exists). This is an exception to the whole-render tier contract; the owner \
                  approves it or requires a quantized-VAE target.",
    },
    OwnerDecision {
        id: "embedding_tables_bf16",
        status: DecisionStatus::Unresolved,
        summary: "model.embed_tokens and latent_pos_embed.pe (0.86 GB) stay BF16 at q8/q4, per \
                  the workspace GGML-tier convention (embeddings are gathered, not multiplied; \
                  Candle has no quantized gather). Norms and biases (vectors) stay BF16 too. The \
                  owner approves this or requires quantized tables.",
    },
    OwnerDecision {
        id: "v1_fp16_exception_not_carried",
        status: DecisionStatus::Recorded,
        summary: "YuE1's approved xcodec/Vocos fp16 carve-out does not apply to YuE2: YuE2 has \
                  no xcodec or Vocos and runs no component in FP16.",
    },
    OwnerDecision {
        id: "shared_ar_nar_weights",
        status: DecisionStatus::Recorded,
        summary: "At bf16/q8/q4 the AR and NAR stages share one resident MoT whose tier is fixed \
                  at load: no restore or re-quantization between stages. Only the experimental \
                  FP8 AR mode swaps the AR projections, and restores the exact BF16 originals \
                  before the acoustic stage and before any save.",
    },
    OwnerDecision {
        id: "tier_snapshot_rehost",
        status: DecisionStatus::Unresolved,
        summary: "q8/q4 tier snapshots are derived locally from the verified pinned original \
                  (crate::tier). Rehosting them is redistribution of CC BY-NC 4.0 derived weights, \
                  which crate::license refuses until an owner records a distribution basis.",
    },
    OwnerDecision {
        id: "fp8_not_on_the_load_spec",
        status: DecisionStatus::Unresolved,
        summary: "The experimental FP8 AR mode is a native engine option (ArPrecision::Fp8). \
                  gen-core's LoadSpec has no FP8 quantize value, so the registered provider \
                  cannot request it; adding one is a cross-workspace contract change for the \
                  owner to decide.",
    },
    OwnerDecision {
        id: "metal_bf16_parity",
        status: DecisionStatus::Unresolved,
        summary: "BF16 whole-model device parity is measured on CUDA here; Metal BF16 parity is \
                  terminal-story evidence on the owner's GPU (never measured on the dev Mac's \
                  GPU by this story).",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    fn released_table() -> Vec<(String, String, Vec<usize>)> {
        let m = crate::inventory::ComponentId::Lm
            .component()
            .conversion_manifest()
            .unwrap();
        m.tensors
            .iter()
            .map(|t| (t.name.clone(), t.dtype.clone(), t.shape.clone()))
            .collect()
    }

    fn plan_of(tier: Tier) -> Vec<PlannedTensor> {
        let table = released_table();
        plan(
            table
                .iter()
                .map(|(n, d, s)| (n.as_str(), d.as_str(), s.as_slice())),
            tier,
        )
        .unwrap()
    }

    /// Every tensor of the released checkpoint has a class; the classes carve it the way the
    /// module table says (628 tensors: 28 layers × 2 paths × 7 projections, 8 norms per layer…).
    #[test]
    fn every_released_tensor_is_classified() {
        let table = released_table();
        let mut counts = std::collections::BTreeMap::new();
        for (name, _, _) in &table {
            let class = classify(name).unwrap_or_else(|| panic!("unclassified {name}"));
            *counts.entry(class.name()).or_insert(0usize) += 1;
        }
        let want = [
            ("ar_projection", 28 * 7),
            ("bias", 4),
            ("latent_positions", 1),
            ("lm_head", 1),
            ("nar_head", 4),
            ("nar_projection", 28 * 7),
            ("norm", 28 * 8 + 1),
            ("token_embedding", 1),
        ];
        assert_eq!(counts.into_iter().collect::<Vec<_>>(), want);
        for bad in [
            "model.layers.x.mlp.up_proj.weight",
            "model.layers.3.mlp.fc.weight",
            "model.layers.3.nar_mlp.gate_proj.bias",
            "lm_head.bias",
            "model.embed_tokens",
        ] {
            assert_eq!(classify(bad), None, "{bad}");
        }
    }

    /// The released checkpoint's precision map, per tier: every matmul weight follows the tier
    /// (Q8_0 / Q4_K, and Q4_0 for `vae2llm`'s 64-wide input), everything else stays BF16.
    #[test]
    fn released_precision_map_follows_the_tier_for_every_matmul_weight() {
        for tier in Tier::ALL {
            for t in plan_of(tier) {
                let want = match (tier, t.class.follows_tier()) {
                    (Tier::Bf16, _) | (_, false) => Storage::Bf16,
                    (Tier::Q8, true) => Storage::Ggml(GgmlDType::Q8_0),
                    (Tier::Q4, true) if t.name == "vae2llm.weight" => {
                        Storage::Ggml(GgmlDType::Q4_0)
                    }
                    (Tier::Q4, true) => Storage::Ggml(GgmlDType::Q4K),
                };
                assert_eq!(t.storage, want, "{tier} {}", t.name);
            }
        }
    }

    /// A matmul weight that does not divide into the tier's blocks is refused, never kept dense;
    /// a non-BF16 source or an unknown tensor is refused.
    #[test]
    fn unfit_tensors_are_refused() {
        let e = storage(Tier::Q8, TensorClass::ArProjection, &[64, 48]).unwrap_err();
        assert!(matches!(e, gen_core::Error::Unsupported(_)), "{e}");
        let e = storage(Tier::Q4, TensorClass::LmHead, &[64, 48]).unwrap_err();
        assert!(e.to_string().contains("does not divide"), "{e}");
        assert!(storage(Tier::Q4, TensorClass::LmHead, &[64]).is_err());
        assert_eq!(
            storage(Tier::Q4, TensorClass::Norm, &[48]).unwrap(),
            Storage::Bf16
        );
        let shape = [64usize, 64];
        assert!(plan([("lm_head.weight", "F32", &shape[..])], Tier::Q8).is_err());
        assert!(plan([("lm_head.extra", "BF16", &shape[..])], Tier::Q8).is_err());
    }

    /// The weights-free residency of the released checkpoint (CPU, F32 compute): bf16 is the whole
    /// checkpoint upcast, q8/q4 hold the matmul weights at 34/32 and 18/32 or 144/256 bytes per
    /// weight; FP8 counts one byte per AR weight on the device plus its two BF16 bytes on the host.
    #[test]
    fn residency_prices_every_tier_and_the_fp8_originals() {
        let bf16 = plan_of(Tier::Bf16);
        let params: u64 = bf16
            .iter()
            .map(|t| t.logical_shape.iter().map(|&d| d as u64).product::<u64>())
            .sum();
        let cpu = weight_residency(&bf16, Backend::Cpu, DType::F32, false);
        assert_eq!(cpu.device_bytes, params * 4);
        assert_eq!(cpu.host_bytes, 0);
        let gpu = weight_residency(&bf16, Backend::Cuda, DType::BF16, false);
        assert_eq!(gpu.device_bytes, params * 2);
        let ar: u64 = bf16
            .iter()
            .filter(|t| t.class == TensorClass::ArProjection)
            .map(|t| t.logical_shape.iter().map(|&d| d as u64).product::<u64>())
            .sum();
        let fp8 = weight_residency(&bf16, Backend::Cuda, DType::BF16, true);
        assert_eq!(fp8.host_bytes, ar * 2);
        assert_eq!(fp8.device_bytes, params * 2 - ar * 2 + ar + 4 * 28 * 7);
        let q8 = weight_residency(&plan_of(Tier::Q8), Backend::Metal, DType::BF16, false);
        let q4 = weight_residency(&plan_of(Tier::Q4), Backend::Metal, DType::BF16, false);
        assert!(q4.device_bytes < q8.device_bytes && q8.device_bytes < gpu.device_bytes);
        println!(
            "resident weights: bf16 {:.2} GB, q8 {:.2} GB, q4 {:.2} GB; fp8 device {:.2} GB + \
             host originals {:.2} GB",
            gpu.device_bytes as f64 / 1e9,
            q8.device_bytes as f64 / 1e9,
            q4.device_bytes as f64 / 1e9,
            fp8.device_bytes as f64 / 1e9,
            fp8.host_bytes as f64 / 1e9,
        );
    }

    #[test]
    fn quantize_mapping_refuses_nvfp4_and_keeps_none_distinct() {
        assert_eq!(Tier::from_quant(None).unwrap(), None);
        assert_eq!(Tier::from_quant(Some(Quant::Q8)).unwrap(), Some(Tier::Q8));
        assert_eq!(Tier::from_quant(Some(Quant::Q4)).unwrap(), Some(Tier::Q4));
        assert!(matches!(
            Tier::from_quant(Some(Quant::Nvfp4)),
            Err(gen_core::Error::Unsupported(_))
        ));
        for t in Tier::ALL {
            assert_eq!(Tier::parse(t.name()), Some(t));
        }
    }

    #[test]
    fn every_owner_decision_is_listed_once() {
        let mut ids: Vec<_> = OWNER_DECISIONS.iter().map(|d| d.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), OWNER_DECISIONS.len());
        assert!(OWNER_DECISIONS
            .iter()
            .any(|d| d.id == "vae_fp32_at_every_tier" && d.status == DecisionStatus::Unresolved));
    }
}
