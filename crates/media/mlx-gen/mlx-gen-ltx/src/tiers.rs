//! LTX-2.5 **packed quant tiers** — `q4/`, `q8/`, `bf16/` from the split component set (sc-18775).
//!
//! [`crate::convert::convert_and_assemble`] builds the shipped `SceneWorks/ltx-2.3-mlx` tiers from
//! LTX-2.3's single all-in-one checkpoint. LTX-2.5 changes the input side fundamentally: **eight
//! component files instead of one** ([`LtxComponent`]), a 42.0 GB dense-bf16 transformer, and a
//! 26.3 GB text encoder that is now its own component rather than a co-requisite HF snapshot. This
//! module is the 2.5 side of the same job.
//!
//! # A tier is a whole-pipeline contract
//!
//! `q4` means **every quantizable segment** of the pipeline at 4 bits, not a q4 transformer beside a
//! bf16 everything-else. Across a component split it is much easier to leave a component
//! accidentally dense, so the policy here is explicit rather than incidental: every emitted component
//! declares, in [`LtxTierReport`] and in the tier's `split_model.json`, how many of its tensors were
//! quantized, how many float tensors stayed dense, and — when a component carries **no** quantized
//! tensor inside a quantized tier — a [`DenseReason`] saying why. `tests/ltx_2_5_tiers.rs` and
//! `tests/ltx_2_5_tiers_real_weights.rs` re-derive those numbers **from the produced files**, so the
//! claim is measured rather than asserted from converter intent.
//!
//! ## What is quantized, and why
//!
//! | segment | q4/q8 | measured bf16 bytes | read path |
//! |---|---|---|---|
//! | DiT attention + FFN Linears (1344) | **yes** | 37.04 GB | [`crate::transformer::Linear`] (`.scales` predicate) |
//! | the two embeddings connectors' attention + FFN Linears (96) | **yes** | 4.02 GB | [`crate::connector::Connector`] (sc-18775) |
//! | the two `text_embedding_projection.*_aggregate_embed` Linears | **yes** | 2.31 GB | [`crate::text_encoder::LtxTextEncoder`] (sc-18775) |
//! | Gemma 4 attention + MLP projections (328) | **yes** | 21.85 GB | `mlx_llm::primitives::projection::Projection` (`config.quantization`) |
//! | `model.embed_tokens` | no | 2.01 GB | an embedding **lookup**, not a matmul — mlx-llm has no quantized-embedding read path, and the weight is tied to an LM head LTX never runs |
//! | the Gemma vision tower / audio + multimodal projectors | no | 0.08 GB | not on the LTX text path at all; carried verbatim so the pack stays self-contained |
//! | conv video VAE, audio VAE, vocoder, both latent upsamplers | no | 3.07 GB | **zero** rank-2 Linear weights between them — every weight is a Conv1d/2d/3d kernel, a norm, or a per-channel statistic, and MLX has no quantized convolution |
//! | the DiffVAE `NADiffusionDecoder` | no | 0.83 GB | its MLX port is sc-18766; quantizing a decoder no test in this crate can run would ship unverified weights |
//! | duration head | no | 0.004 GB | no MLX port exists yet (sc-18777); same reasoning, and it is 0.02 % of the smallest tier |
//! | biases, norms, `layer_scalar`, `scale_shift_table`s, learnable registers | no | — | not matmul weights; the affine grid is defined over a Linear's input axis |
//!
//! Every "no" row above is a **declared** exemption carried in the manifest, not an omission: the
//! validation tests fail on a component that is dense without one.
//!
//! # Layout
//!
//! Each tier directory is a SceneWorks split bundle in the shape [`crate::model::load`] already
//! consumes — the same per-component `.safetensors` + `embedded_config.json` + `split_model.json`
//! the 2.3 converter emits.
//!
//! One component boundary is deliberately **not** upstream's: the LTX-2.5 text-encoder file carries
//! `text_embedding_projection.*`, and the tier moves it into `connector.safetensors`. That is where
//! LTX-2.3 keeps it and where [`crate::text_encoder::LtxTextEncoder`]'s feature heads read it from,
//! so the tier stays a drop-in for the shipped ports instead of requiring a loader signature change
//! for a weight that is a text-stage weight either way.
//!
//! Two 2.5-specific additions on top of the 2.3 shape:
//!
//! * every component file **keeps its own `__metadata__`** (the upstream `config` slice,
//!   `model_version`, the transformer's `gemma_source_checkpoint`, the text encoder's `gemma_config`,
//!   and the embedded licence text) rather than being re-emitted bare. The loader reads config from
//!   metadata; a converter that drops it produces a bundle that loads as garbage or not at all;
//! * the text encoder's `gemma_config` gains a top-level `quantization` block in the quantized tiers,
//!   which is exactly the trigger `mlx_llm::config::ModelConfig::from_json` reads to bind
//!   pre-quantized projections.
//!
//! # Traps this module is written around
//!
//! * **This conversion runs on MLX's CPU stream** ([`CpuConversion`]). It is bandwidth-bound work
//!   with no kernel worth dispatching, and on the GPU stream the multi-GB materializations trip
//!   Metal's command-buffer watchdog — measured, twice, on the 26.3 GB text encoder.
//! * **`save_file` metadata order is not stable.** Two byte-identical conversions can produce
//!   different files, so nothing here — and nothing downstream — may verify a tier by hashing it.
//!   Verification is by *content measurement* (per-tensor dtype, quant geometry, key sets).
//! * **Non-contiguous writes corrupt silently.** Every emitted tensor goes through
//!   [`crate::convert`]'s established save path, which `eval`s before writing.
//! * **AppleDouble sidecars break `*.safetensors` globs.** The tier tree is written fresh and never
//!   copied through a resource-fork-preserving path; discovery skips dot-files
//!   (`gen_core::weightsmeta::is_hidden_file`).
//! * **The Gemma 4 checkpoint carries 48 per-layer `layer_scalar` `[1]` trained buffers** and no
//!   `v_proj` on its eight `full_attention` layers. Both survive here because the text-encoder
//!   emitter is a whole-file pass-through with a quantization predicate over it, never a rebuild
//!   from an expected key list.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use mlx_rs::ops::quantize;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Device, Dtype};

use mlx_gen::gen_core::ltx_checkpoint::{LtxBundle, LtxComponent, LtxResolvedComponent};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::convert::{
    build_connector_component, cast_component_floats_bf16, sanitize_audio_vae_component,
    sanitize_transformer_component, sanitize_vae_decoder_component, sanitize_vae_encoder_component,
    sanitize_vocoder_component, CONNECTOR_QUANT_SUFFIXES, TRANSFORMER_QUANT_SUFFIXES,
};

/// The affine-quant group width every shipped LTX tier uses (the reference `convert.py` default).
pub const DEFAULT_GROUP_SIZE: i32 = 64;

/// The `model_version` an LTX-2.5 tier declares, propagated from the source components.
pub const LTX_2_5_MODEL_VERSION: &str = "2.5.0";

/// The tier manifest file, beside the per-component `.safetensors`.
pub const TIER_MANIFEST_FILE: &str = "split_model.json";

/// The merged per-component config sidecar the shipped ports read
/// (`LtxConfig::from_model_dir` and friends).
pub const EMBEDDED_CONFIG_FILE: &str = "embedded_config.json";

/// The `__metadata__` key the tier converter stamps with the tier id, so a component file that has
/// been moved out of its directory still says which tier it came from.
pub const TIER_METADATA_KEY: &str = "sceneworks_tier";

// =================================================================================================
// Tiers
// =================================================================================================

/// One shipped precision tier.
///
/// The ids are the subdirectory names the 2.3 bundle established (`q4/`, `q8/`, `bf16/`) and that
/// the SceneWorks manifest's `download.subdir` selects between.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LtxTier {
    /// 4-bit affine-quantized Linears (the default tier).
    Q4,
    /// 8-bit affine-quantized Linears.
    Q8,
    /// Dense bf16 — every float tensor at bf16, nothing packed.
    Bf16,
}

impl LtxTier {
    /// Every tier, in descending compression order (the order they are built in).
    pub const ALL: &'static [LtxTier] = &[LtxTier::Q4, LtxTier::Q8, LtxTier::Bf16];

    /// The tier's directory name / manifest id.
    pub fn id(self) -> &'static str {
        match self {
            LtxTier::Q4 => "q4",
            LtxTier::Q8 => "q8",
            LtxTier::Bf16 => "bf16",
        }
    }

    /// Bits per quantized weight, or `None` for the dense tier.
    pub fn bits(self) -> Option<i32> {
        match self {
            LtxTier::Q4 => Some(4),
            LtxTier::Q8 => Some(8),
            LtxTier::Bf16 => None,
        }
    }

    /// Parse a tier id (the inverse of [`id`](Self::id)).
    pub fn from_id(id: &str) -> Option<LtxTier> {
        LtxTier::ALL.iter().copied().find(|t| t.id() == id)
    }
}

impl std::fmt::Display for LtxTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.id())
    }
}

/// Why a component carries no quantized tensor inside a quantized tier.
///
/// A component that is dense in a `q4`/`q8` tier **must** declare one of these; the tier validation
/// walks the produced files and fails on an undeclared dense component. That is the whole mechanism
/// preventing "q4 transformer, bf16 quietly everything else".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseReason {
    /// The component holds no rank-2 Linear weight at all — every weight is a convolution kernel, a
    /// norm, or a statistic. MLX quantizes Linears (and embeddings), not convolutions, so there is
    /// nothing here the affine grid is defined over. Measured, not assumed: the emitter counts the
    /// component's rank-2 weights and refuses this reason if any exist.
    NoLinearWeights,
    /// The component's Linears are real, but this crate has no MLX port that can *run* them, so a
    /// quantized emission could not be exercised by any test — it would be unverified weights on a
    /// rehost. Carries the story that lands the port.
    NoMlxPort(&'static str),
    /// A dense tier: nothing is quantized anywhere, by definition.
    DenseTier,
}

impl DenseReason {
    /// The stable id written into the tier manifest.
    pub fn id(self) -> &'static str {
        match self {
            DenseReason::NoLinearWeights => "no-linear-weights",
            DenseReason::NoMlxPort(_) => "no-mlx-port",
            DenseReason::DenseTier => "dense-tier",
        }
    }

    /// The human-readable justification written beside the id.
    pub fn describe(self) -> String {
        match self {
            DenseReason::NoLinearWeights => "no rank-2 Linear weights: every weight is a \
                 convolution kernel, a norm or a per-channel statistic, and MLX has no quantized \
                 convolution"
                .to_string(),
            DenseReason::NoMlxPort(story) => format!(
                "this crate has no MLX port that can run these weights yet ({story}); quantizing \
                 them would ship weights no test can exercise"
            ),
            DenseReason::DenseTier => "the bf16 tier is dense by definition".to_string(),
        }
    }
}

// =================================================================================================
// Per-component quantization policy
// =================================================================================================

/// The Gemma 4 decoder projections `mlx_llm`'s loader binds through
/// `Projection::load_quantized` — the exact set for which a `{key}.scales` sibling is read.
///
/// Keyed on the **suffix** so the layer index and the `model.` / `model.language_model.` prefix
/// variance both fall out. `k_norm`/`q_norm`/`layernorm`/`layer_scalar`/`embed_tokens` are
/// deliberately absent: the first four are not matmul weights, and the last is an embedding lookup.
const GEMMA_QUANT_SUFFIXES: &[&str] = &[
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
    ".mlp.gate_proj",
    ".mlp.up_proj",
    ".mlp.down_proj",
];

/// Key prefix of the LTX text-embedding projection. It rides inside the LTX-2.5 **text-encoder**
/// file; the tier moves it into `connector.safetensors`, where LTX-2.3 keeps it and where
/// [`crate::text_encoder::LtxTextEncoder`]'s feature heads read it from.
pub const TEXT_PROJECTION_PREFIX: &str = "text_embedding_projection.";

/// Whether `key` names a weight this component quantizes at a quantized tier.
///
/// `suffixes` are matched against the key with its `.weight` suffix removed, exactly as the
/// reference `_quantize_ltx_predicate` and `mlx_llm`'s `load_proj` do.
fn matches_quant_suffix(key: &str, suffixes: &[&str]) -> bool {
    key.strip_suffix(".weight")
        .is_some_and(|base| suffixes.iter().any(|s| base.ends_with(s)))
}

/// The text encoder's quantized set: the Gemma decoder projections, and nothing else.
///
/// Never the packed HF asset tensors (`tokenizer_json`, `hf_asset__*`) — they are `U8` payloads and
/// quantizing one would corrupt the tokenizer — and never `model.embed_tokens.weight`, which is an
/// embedding **lookup**. The two `text_embedding_projection` aggregate embeds are quantized too, but
/// under [`CONNECTOR_QUANT_SUFFIXES`]: the tier moves them into the connector component.
fn is_text_encoder_quantizable(key: &str) -> bool {
    if mlx_gen::gen_core::gemma_assets::is_gemma_asset_key(key) {
        return false;
    }
    matches_quant_suffix(key, GEMMA_QUANT_SUFFIXES)
}

// =================================================================================================
// Reports
// =================================================================================================

/// What one component of one tier actually contains, measured after it was written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LtxTierComponentReport {
    /// The component's file stem (`transformer`, `connector`, `text_encoder`, …).
    pub name: String,
    /// The written file.
    pub file: PathBuf,
    /// Total tensors in the emitted file, quantized triples counted as three.
    pub tensors: usize,
    /// Linears emitted as a packed `weight`/`scales`/`biases` triple.
    pub quantized_linears: usize,
    /// Float tensors left dense (biases, norms, statistics, conv kernels, exempt Linears).
    pub dense_float_tensors: usize,
    /// Non-float, non-quantized tensors passed through verbatim (the packed HF assets).
    pub passthrough_tensors: usize,
    /// Bytes the emitted file occupies on disk.
    pub bytes: u64,
    /// Set when the component carries no quantized tensor — never `None` in that case.
    pub dense_reason: Option<DenseReason>,
}

/// One built tier: where it is, what it declares, and what every component measured.
#[derive(Clone, Debug)]
pub struct LtxTierReport {
    /// Which tier this is.
    pub tier: LtxTier,
    /// The tier directory.
    pub dir: PathBuf,
    /// Bits per quantized weight (`None` for `bf16`).
    pub bits: Option<i32>,
    /// The affine group width.
    pub group_size: i32,
    /// Per-component measurements, in emission order.
    pub components: Vec<LtxTierComponentReport>,
    /// Total bytes of every emitted file (weights + sidecars).
    pub bytes: u64,
}

impl LtxTierReport {
    /// One component's report by name.
    pub fn component(&self, name: &str) -> Option<&LtxTierComponentReport> {
        self.components.iter().find(|c| c.name == name)
    }

    /// Every quantized Linear across the tier.
    pub fn quantized_linears(&self) -> usize {
        self.components.iter().map(|c| c.quantized_linears).sum()
    }

    /// The manifest value this report serializes to (also written to `split_model.json`).
    pub fn manifest(&self, source: &BTreeMap<String, PathBuf>) -> serde_json::Value {
        let components: Vec<serde_json::Value> = self
            .components
            .iter()
            .map(|c| {
                let mut entry = serde_json::json!({
                    "name": c.name,
                    "file": format!("{}.safetensors", c.name),
                    "tensors": c.tensors,
                    "quantized_linears": c.quantized_linears,
                    "dense_float_tensors": c.dense_float_tensors,
                    "passthrough_tensors": c.passthrough_tensors,
                    "bytes": c.bytes,
                });
                if let Some(reason) = c.dense_reason {
                    entry["dense_reason"] = serde_json::Value::from(reason.id());
                    entry["dense_reason_detail"] = serde_json::Value::from(reason.describe());
                }
                entry
            })
            .collect();
        let sources: serde_json::Map<String, serde_json::Value> = source
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::from(v.display().to_string())))
            .collect();
        serde_json::json!({
            "format": "split",
            "model_version": LTX_2_5_MODEL_VERSION,
            "variant": "distilled",
            "tier": self.tier.id(),
            // `quantized` / `quantization_bits` / `quantization_group_size` are the fields
            // `crate::config::SplitModel` reads; the tier detail rides alongside them.
            "quantized": self.bits.is_some(),
            // The dense tier still declares a well-defined geometry: `SplitModel::dense()` does the
            // same, so a downstream consumer never has to invent one.
            "quantization_bits": self.bits.unwrap_or(4),
            "quantization_group_size": self.group_size,
            "components": self.components.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
            "component_detail": components,
            // Weights only. The sidecars are written after this value is computed — a manifest
            // whose own size fed back into a total it declares could never be self-consistent — so
            // `LtxTierReport::bytes` (which adds them) is deliberately the larger number, and it is
            // the one sc-18781's `footprint.diskSizeBytes` should carry.
            "total_bytes": self.bytes,
            "source": serde_json::Value::Object(sources),
        })
    }
}

// =================================================================================================
// The emitter
// =================================================================================================

/// One component's weights plus the metadata that must travel with them.
struct Emit {
    name: &'static str,
    weights: HashMap<String, Array>,
    metadata: BTreeMap<String, String>,
    /// `None` ⇒ this component quantizes nothing at any tier, for this declared reason.
    exempt: Option<DenseReason>,
}

/// Quantize every `map` entry matching `is_quantizable`, in place, into the
/// `weight`(u32)/`scales`/`biases` triple MLX's `quantized_matmul` consumes.
///
/// A matched weight whose input axis is not a multiple of `group_size` is a **hard error**, not a
/// skip: silently leaving it dense is exactly the "a component quietly stayed bf16" failure this
/// module exists to prevent, and it would surface much later as a load that binds a dense weight
/// where the tier promised a packed one.
fn quantize_selected(
    map: HashMap<String, Array>,
    bits: i32,
    group_size: i32,
    component: &str,
    is_quantizable: impl Fn(&str) -> bool,
) -> Result<(HashMap<String, Array>, usize)> {
    let mut out = HashMap::with_capacity(map.len());
    let mut count = 0usize;
    for (key, value) in map {
        if !is_quantizable(&key) {
            out.insert(key, value);
            continue;
        }
        let base = key
            .strip_suffix(".weight")
            .expect("the quant predicates only match `.weight` keys")
            .to_string();
        let shape = value.shape();
        let last = *shape.last().unwrap_or(&0);
        if shape.len() != 2 || last % group_size != 0 {
            return Err(Error::Msg(format!(
                "ltx tiers: {component}: {key} is selected for quantization but its shape {shape:?} \
                 is not a rank-2 weight whose input axis divides the group size {group_size}"
            )));
        }
        let (packed, scales, biases) = quantize(&value, group_size, bits)?;
        out.insert(format!("{base}.weight"), packed);
        out.insert(format!("{base}.scales"), scales);
        out.insert(format!("{base}.biases"), biases);
        count += 1;
    }
    Ok((out, count))
}

/// Count the rank-2 float weight tensors in a map — the population
/// [`DenseReason::NoLinearWeights`] claims is empty.
fn rank2_float_weights(map: &HashMap<String, Array>) -> usize {
    map.iter()
        .filter(|(key, value)| {
            key.ends_with(".weight")
                && value.ndim() == 2
                && matches!(
                    value.dtype(),
                    Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
                )
        })
        .count()
}

/// Byte budget for one `eval` submission (measured against the arrays' logical size).
///
/// MLX is lazy, so a single `eval` over a whole component's tensor map materializes that component's
/// entire graph in one shot — 4349 tensors and 1344 `quantize` ops for the transformer, 26.3 GB of
/// Gemma weights for the text encoder. Batching bounds the transient working set to something that
/// leaves room for the source mapping and the write buffer, instead of asking the allocator for the
/// whole component at once.
const EVAL_BATCH_BYTES: usize = 512 * 1024 * 1024;

/// `eval` a component's tensors in bounded batches. See [`EVAL_BATCH_BYTES`].
///
/// Iteration order is the map's, which is arbitrary but irrelevant: every tensor is an independent
/// leaf of the write, so any partition into batches produces identical bytes.
fn eval_in_batches<'a>(arrays: impl IntoIterator<Item = &'a Array>) -> Result<()> {
    let mut batch: Vec<&Array> = Vec::new();
    let mut bytes = 0usize;
    for array in arrays {
        bytes = bytes.saturating_add(array.nbytes());
        batch.push(array);
        if bytes >= EVAL_BATCH_BYTES {
            eval(batch.iter().copied())?;
            batch.clear();
            bytes = 0;
        }
    }
    if !batch.is_empty() {
        eval(batch.iter().copied())?;
    }
    Ok(())
}

/// Runs a tier conversion on MLX's **CPU** stream, restoring the previous default device on drop.
///
/// A weight conversion has no business on the Metal command queue. It is an offline,
/// bandwidth-bound job — read a tensor, quantize or cast it, write it — with no kernel worth
/// dispatching and nothing to gain from the GPU, and putting it there costs two things that both
/// bit during this story:
///
/// 1. **The watchdog.** Metal kills a command buffer that runs too long. Materializing the 26.3 GB
///    text encoder came back as `kIOGPUCommandBufferCallbackErrorTimeout` — and once one buffer
///    fails, every later submission in the process returns
///    `kIOGPUCommandBufferCallbackErrorSubmissionsIgnored`, so the failure is not even local to the
///    component that caused it. Both were measured on this conversion, twice.
/// 2. **The device.** This machine's GPU is also what renders; a multi-minute conversion holding the
///    queue is a conversion that cannot run beside anything else.
///
/// Setting the default device is global to the process, so this is an RAII guard rather than a
/// one-way switch: a caller that converts and then runs a forward (every real-weights test here
/// does) gets its GPU back, panic or not.
struct CpuConversion {
    previous: Device,
}

impl CpuConversion {
    fn enter() -> Result<Self> {
        let previous = Device::try_default()?;
        Device::set_default(&Device::cpu());
        Ok(Self { previous })
    }
}

impl Drop for CpuConversion {
    fn drop(&mut self) {
        Device::set_default(&self.previous);
    }
}

/// Materialize and write one component, then measure the file that landed.
///
/// Deliberately measures **after** the write: the report is a statement about bytes on disk, which
/// is what the manifest's footprint numbers and the tier validation both need. There is no hash
/// here and there must not be one — `save_file` orders `__metadata__` nondeterministically, so two
/// correct conversions of the same input differ byte-for-byte.
fn write_component(
    dir: &Path,
    emit: &Emit,
    quantized_linears: usize,
    dense_reason: Option<DenseReason>,
) -> Result<LtxTierComponentReport> {
    let file = dir.join(format!("{}.safetensors", emit.name));
    eval_in_batches(emit.weights.values())?;
    let metadata: HashMap<String, String> = emit
        .metadata
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    Array::save_safetensors(
        emit.weights.iter().map(|(k, v)| (k.as_str(), v)),
        Some(&metadata),
        &file,
    )?;

    let mut dense_float = 0usize;
    let mut passthrough = 0usize;
    for (key, value) in &emit.weights {
        let is_quant_part = key.ends_with(".scales")
            || key.ends_with(".biases")
            || (key.ends_with(".weight") && value.dtype() == Dtype::Uint32);
        if is_quant_part {
            continue;
        }
        match value.dtype() {
            Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16 => dense_float += 1,
            _ => passthrough += 1,
        }
    }
    let bytes = std::fs::metadata(&file)?.len();
    Ok(LtxTierComponentReport {
        name: emit.name.to_string(),
        file,
        tensors: emit.weights.len(),
        quantized_linears,
        dense_float_tensors: dense_float,
        passthrough_tensors: passthrough,
        bytes,
        dense_reason,
    })
}

/// Cast, quantize (when the tier and the component call for it), write, and report one component.
///
/// This is the single place a component becomes a file, so the "nothing silently stays bf16" rule
/// has exactly one enforcement point: a quantized tier + a component with no declared exemption +
/// zero quantized Linears is an error here, not a surprise on the rehost.
fn emit_component(
    dir: &Path,
    mut emit: Emit,
    tier: LtxTier,
    group_size: i32,
    is_quantizable: impl Fn(&str) -> bool,
) -> Result<LtxTierComponentReport> {
    cast_component_floats_bf16(&mut emit.weights)?;

    let dense_reason = match (tier.bits(), emit.exempt) {
        (None, _) => Some(DenseReason::DenseTier),
        (Some(_), Some(reason)) => Some(reason),
        (Some(_), None) => None,
    };

    // Validate the structural exemption against the weights rather than trusting the declaration.
    if matches!(dense_reason, Some(DenseReason::NoLinearWeights)) {
        let linears = rank2_float_weights(&emit.weights);
        if linears != 0 {
            return Err(Error::Msg(format!(
                "ltx tiers: {} declares `no-linear-weights` but carries {linears} rank-2 float \
                 weight tensor(s) — the exemption is wrong, or the component gained Linears",
                emit.name
            )));
        }
    }

    let quantized = match (tier.bits(), emit.exempt) {
        (Some(bits), None) => {
            let (weights, count) = quantize_selected(
                std::mem::take(&mut emit.weights),
                bits,
                group_size,
                emit.name,
                is_quantizable,
            )?;
            emit.weights = weights;
            if count == 0 {
                return Err(Error::Msg(format!(
                    "ltx tiers: {} quantized nothing at {} but declares no exemption — a tier is a \
                     whole-pipeline contract, so a silently-dense component is a conversion bug",
                    emit.name,
                    tier.id()
                )));
            }
            count
        }
        _ => 0,
    };

    write_component(dir, &emit, quantized, dense_reason)
}

// =================================================================================================
// Metadata
// =================================================================================================

/// A source component's `__metadata__`, verbatim, plus the tier stamp.
///
/// Verbatim matters in three ways: the `config` slice is what the loader builds the module from,
/// `model_version` is what [`mlx_gen::gen_core::ltx_checkpoint`] keys the split layout on, and the
/// embedded LTX-2.x Community License text travels with the weights it licenses.
fn component_metadata(resolved: &LtxResolvedComponent, tier: LtxTier) -> BTreeMap<String, String> {
    let mut metadata = resolved.metadata().raw().clone();
    metadata.insert(TIER_METADATA_KEY.to_string(), tier.id().to_string());
    metadata
}

/// Metadata for the **secondary** half of a source file the tier splits in two — the connector out
/// of the transformer, the encoder out of each video VAE, the vocoder out of the audio VAE.
///
/// The `config` key is **dropped** rather than copied, so that exactly one emitted file declares
/// each [`LtxComponent`] slot. A `vae_encoder.safetensors` carrying `config.vae` would classify as a
/// second `conv_video_vae` and make [`mlx_gen::gen_core::ltx_checkpoint::discover_split_bundle`]
/// refuse the whole tree as ambiguous; a `connector.safetensors` carrying `config.transformer`
/// would do the same to the transformer slot. The configs are not lost — the primary half of the
/// same split keeps the whole slice ([`component_metadata`]), and the tier's merged
/// `embedded_config.json` carries every section, which is what the shipped ports read.
fn derived_metadata(
    resolved: &LtxResolvedComponent,
    tier: LtxTier,
    derived_from: &str,
) -> BTreeMap<String, String> {
    let mut metadata = component_metadata(resolved, tier);
    metadata.remove(mlx_gen::gen_core::ltx_checkpoint::CONFIG_METADATA_KEY);
    metadata.insert(
        "sceneworks_derived_from".to_string(),
        derived_from.to_string(),
    );
    metadata
}

/// Stamp the text encoder's `gemma_config` with the `quantization` block
/// `mlx_llm::config::ModelConfig::from_json` reads to bind pre-quantized projections.
///
/// Without this the packed `.scales` tensors are present but unread, and `mlx_llm`'s loader
/// explicitly refuses that state ("snapshot stores quantized tensor … but config.json has no
/// `quantization` block") rather than silently loading a dense weight — so this is the difference
/// between a tier that loads and one that fails loudly.
fn stamp_gemma_quantization(
    metadata: &mut BTreeMap<String, String>,
    tier: LtxTier,
    group_size: i32,
    source: &Path,
) -> Result<()> {
    let Some(bits) = tier.bits() else {
        return Ok(());
    };
    let key = mlx_gen::gen_core::ltx_checkpoint::GEMMA_CONFIG_METADATA_KEY;
    let raw = metadata.get(key).ok_or_else(|| {
        Error::Msg(format!(
            "ltx tiers: {} carries no __metadata__[{key:?}] — the text encoder must ship its \
             HuggingFace config",
            source.display()
        ))
    })?;
    let mut config: serde_json::Value = serde_json::from_str(raw).map_err(|e| {
        Error::Msg(format!(
            "ltx tiers: parse {key} from {}: {e}",
            source.display()
        ))
    })?;
    let object = config.as_object_mut().ok_or_else(|| {
        Error::Msg(format!(
            "ltx tiers: {key} in {} is not a JSON object",
            source.display()
        ))
    })?;
    object.insert(
        "quantization".to_string(),
        serde_json::json!({ "group_size": group_size, "bits": bits, "mode": "affine" }),
    );
    metadata.insert(
        key.to_string(),
        serde_json::to_string(&config)
            .map_err(|e| Error::Msg(format!("ltx tiers: serialize {key}: {e}")))?,
    );
    Ok(())
}

// =================================================================================================
// Public entry points
// =================================================================================================

/// Build one tier of an LTX-2.5 split bundle into `out_dir`.
///
/// `bundle` is a resolved 2.5 bundle ([`crate::bundle::resolve_split_bundle`] or
/// `gen_core::ltx_checkpoint::discover_split_bundle`). The transformer, the conv video VAE and the
/// text encoder are **required**; every other component is emitted when the bundle carries it, so a
/// video-only or upsampler-less install converts without a special mode.
///
/// Components are processed **one at a time and dropped**, with the MLX cache cleared between them:
/// the source transformer alone is 42 GB and the text encoder 26 GB, and holding two resident is
/// the difference between a conversion that runs on a 128 GB machine and one that does not.
pub fn convert_2_5_tier(
    bundle: &LtxBundle,
    out_dir: impl AsRef<Path>,
    tier: LtxTier,
    group_size: i32,
) -> Result<LtxTierReport> {
    let out = out_dir.as_ref();
    std::fs::create_dir_all(out)?;
    if group_size <= 0 {
        return Err(Error::Msg(format!(
            "ltx tiers: group size must be positive, got {group_size}"
        )));
    }
    // Everything below runs on the CPU stream and the caller's device is restored on the way out.
    let _cpu = CpuConversion::enter()?;

    let mut components: Vec<LtxTierComponentReport> = Vec::new();
    let mut embedded = serde_json::Map::new();
    let mut sources: BTreeMap<String, PathBuf> = BTreeMap::new();
    // Assembled from two source files (the connector towers from the transformer, the text-embedding
    // projection from the text encoder) and written once both are in hand.
    let mut connector: HashMap<String, Array>;
    let connector_meta: BTreeMap<String, String>;

    // ---- transformer + the two embeddings connectors -------------------------------------------
    {
        let resolved = bundle.require(LtxComponent::Transformer)?;
        sources.insert("transformer".into(), resolved.path().to_path_buf());
        let raw = Weights::from_file(resolved.path())?;
        embedded.insert("transformer".into(), resolved.config()?.clone());
        if let Some(scheduler) = resolved.optional_section("scheduler") {
            embedded.insert("scheduler".into(), scheduler.clone());
        }

        components.push(emit_component(
            out,
            Emit {
                name: "transformer",
                weights: sanitize_transformer_component(&raw),
                metadata: component_metadata(resolved, tier),
                exempt: None,
            },
            tier,
            group_size,
            |key| matches_quant_suffix(key, TRANSFORMER_QUANT_SUFFIXES),
        )?);
        // The two embeddings connectors are lifted out of the transformer file and emitted as their
        // own component, exactly as the 2.3 bundle does: the text-encoder stage runs them long
        // before the DiT is resident, and folding 4 GB of connector into an 11 GB (q4) transformer
        // file would force the whole DiT in just to produce text embeddings.
        //
        // Held (not yet written) because the LTX text-embedding projection belongs in the same file
        // and lives in a different source — see the text-encoder step below. These are unevaluated
        // memory-mapped views until then, so holding them costs page cache, not resident RAM.
        connector = build_connector_component(&raw);
        if connector.is_empty() {
            return Err(Error::Msg(format!(
                "ltx tiers: {} carries no `*_embeddings_connector.*` tensors — an LTX-2.5 \
                 transformer always ships both connectors",
                resolved.path().display()
            )));
        }
        connector_meta = derived_metadata(resolved, tier, "transformer");
        drop(raw);
        mlx_rs::memory::clear_cache();
    }

    // ---- text encoder + the text-embedding projection --------------------------------------------
    //
    // LTX-2.3 kept `text_embedding_projection.*` in `connector.safetensors`; LTX-2.5 ships it inside
    // the text-encoder file. The **tier puts it back with the connectors**, because that is the file
    // [`crate::text_encoder::LtxTextEncoder`] reads both from — its `video_head`/`audio_head` take
    // one `connector_w` carrying the `aggregate_embed` Linear *and* the connector tower. Leaving the
    // projection in the text-encoder file would make an otherwise drop-in tier need a signature
    // change in whatever loads it (sc-18778), for no benefit: the two aggregate embeds are a
    // text-stage weight either way, and this keeps every LTX bundle, 2.3 and 2.5, one shape.
    {
        let resolved = bundle.require(LtxComponent::TextEncoder)?;
        sources.insert("text_encoder".into(), resolved.path().to_path_buf());
        if resolved.path().is_dir() {
            return Err(Error::Msg(format!(
                "ltx tiers: the text encoder at {} is an HuggingFace snapshot directory; an LTX-2.5 \
                 tier packs the single-file encoder that carries its own assets",
                resolved.path().display()
            )));
        }
        let raw = Weights::from_file(resolved.path())?;
        let mut encoder = passthrough_map(&raw);
        let projection: Vec<String> = encoder
            .keys()
            .filter(|k| k.starts_with(TEXT_PROJECTION_PREFIX))
            .cloned()
            .collect();
        if projection.is_empty() {
            return Err(Error::Msg(format!(
                "ltx tiers: {} carries no `{TEXT_PROJECTION_PREFIX}*` tensors — the LTX-2.5 text \
                 encoder ships the video/audio aggregate embeds the feature heads project through",
                resolved.path().display()
            )));
        }
        for key in projection {
            let value = encoder.remove(&key).expect("key from keys()");
            connector.insert(key, value);
        }

        let mut metadata = component_metadata(resolved, tier);
        stamp_gemma_quantization(&mut metadata, tier, group_size, resolved.path())?;
        components.push(emit_component(
            out,
            Emit {
                name: "text_encoder",
                weights: encoder,
                metadata,
                exempt: None,
            },
            tier,
            group_size,
            is_text_encoder_quantizable,
        )?);
        drop(raw);
        mlx_rs::memory::clear_cache();
    }

    // ---- connector (both towers + the text-embedding projection) ----------------------------------
    components.push(emit_component(
        out,
        Emit {
            name: "connector",
            weights: connector,
            metadata: connector_meta,
            exempt: None,
        },
        tier,
        group_size,
        |key| matches_quant_suffix(key, CONNECTOR_QUANT_SUFFIXES),
    )?);
    mlx_rs::memory::clear_cache();

    // ---- conv video VAE (encoder + decoder) -----------------------------------------------------
    {
        let resolved = bundle.require(LtxComponent::ConvVideoVae)?;
        sources.insert("conv_video_vae".into(), resolved.path().to_path_buf());
        let raw = Weights::from_file(resolved.path())?;
        let vae_block = resolved.config()?.clone();
        // Parse through the real reader so an unknown `latent_log_var` or an unsupported padding
        // mode is an error here rather than at render time.
        let _ = crate::config::LtxVaeConfig::from_embedded_vae(&vae_block)?;
        embedded.insert("vae".into(), vae_block);

        // The **decoder** carries the `config.vae` slice and so is the file that classifies as
        // `conv_video_vae`; the encoder half carries none. Exactly one file per component slot may
        // declare itself, or a directory scan over the tier tree is ambiguous — and the decoder is
        // the right one to be it, because a "video VAE" that cannot decode is not the component a
        // caller resolving `conv_video_vae` is asking for.
        for (name, weights, metadata) in [
            (
                "vae_decoder",
                sanitize_vae_decoder_component(&raw)?,
                component_metadata(resolved, tier),
            ),
            (
                "vae_encoder",
                sanitize_vae_encoder_component(&raw)?,
                derived_metadata(resolved, tier, "conv_video_vae"),
            ),
        ] {
            if weights.is_empty() {
                return Err(Error::Msg(format!(
                    "ltx tiers: {} yielded no {name} tensors",
                    resolved.path().display()
                )));
            }
            components.push(emit_component(
                out,
                Emit {
                    name,
                    weights,
                    metadata,
                    exempt: Some(DenseReason::NoLinearWeights),
                },
                tier,
                group_size,
                |_| false,
            )?);
        }
        drop(raw);
        mlx_rs::memory::clear_cache();
    }

    // ---- diffusion video VAE ---------------------------------------------------------------------
    if let Some(resolved) = bundle.get(LtxComponent::DiffusionVideoVae) {
        sources.insert("diffusion_video_vae".into(), resolved.path().to_path_buf());
        let raw = Weights::from_file(resolved.path())?;
        let vae_block = resolved.config()?.clone();
        embedded.insert("diffusion_vae".into(), vae_block);

        // The DiffVAE's encoder is the same conv module as the conv VAE's, differing only in the
        // declared `latent_log_var`; its decoder is an `NADiffusionDecoder` of rank-2 Linears.
        let encoder = sanitize_vae_encoder_component(&raw)?;
        if encoder.is_empty() {
            return Err(Error::Msg(format!(
                "ltx tiers: {} yielded no encoder tensors",
                resolved.path().display()
            )));
        }
        components.push(emit_component(
            out,
            Emit {
                name: "diffusion_vae_encoder",
                weights: encoder,
                metadata: derived_metadata(resolved, tier, "diffusion_video_vae"),
                exempt: Some(DenseReason::NoLinearWeights),
            },
            tier,
            group_size,
            |_| false,
        )?);

        // As with the conv VAE, the decoder is the file that carries `config.vae` — here declaring
        // `CausalDiffusionVAE`, so it is the one that classifies as `diffusion_video_vae`. A DiffVAE
        // file carrying no diffusion decoder is not a DiffVAE, and leaving the slot unclaimed in
        // that case is the honest outcome rather than pointing it at a conv encoder.
        let decoder = sanitize_vae_decoder_component(&raw)?;
        let has_stages = decoder.keys().any(|k| k.starts_with("det_stages."));
        let has_blocks = decoder.keys().any(|k| k.starts_with("diff_blocks."));
        if has_stages && has_blocks {
            components.push(emit_component(
                out,
                Emit {
                    name: "vae_diffusion_decoder",
                    weights: decoder,
                    metadata: component_metadata(resolved, tier),
                    exempt: Some(DenseReason::NoMlxPort("sc-18766")),
                },
                tier,
                group_size,
                |_| false,
            )?);
        }
        drop(raw);
        mlx_rs::memory::clear_cache();
    }

    // ---- audio VAE + vocoder ---------------------------------------------------------------------
    if let Some(resolved) = bundle.get(LtxComponent::AudioVae) {
        sources.insert("audio_vae".into(), resolved.path().to_path_buf());
        let raw = Weights::from_file(resolved.path())?;
        for section in ["audio_vae", "vocoder"] {
            embedded.insert(section.into(), resolved.config_section(section)?.clone());
        }
        // Upstream packs the audio VAE and the vocoder in one file carrying both config sections;
        // the tier splits the weights but keeps the whole config on `audio_vae.safetensors`, which
        // is therefore the file that classifies as the `audio_vae` component. `vocoder` is not an
        // `LtxComponent` slot at all, so its file declaring nothing is correct rather than a loss.
        for (name, weights, metadata) in [
            (
                "audio_vae",
                sanitize_audio_vae_component(&raw)?,
                component_metadata(resolved, tier),
            ),
            (
                "vocoder",
                sanitize_vocoder_component(&raw)?,
                derived_metadata(resolved, tier, "audio_vae"),
            ),
        ] {
            if weights.is_empty() {
                return Err(Error::Msg(format!(
                    "ltx tiers: {} yielded no {name} tensors",
                    resolved.path().display()
                )));
            }
            components.push(emit_component(
                out,
                Emit {
                    name,
                    weights,
                    metadata,
                    exempt: Some(DenseReason::NoLinearWeights),
                },
                tier,
                group_size,
                |_| false,
            )?);
        }
        drop(raw);
        mlx_rs::memory::clear_cache();
    }

    // ---- latent upsamplers + duration head --------------------------------------------------------
    for (component, name, exempt) in [
        (
            LtxComponent::SpatialUpsampler,
            "spatial_upsampler",
            DenseReason::NoLinearWeights,
        ),
        (
            LtxComponent::TemporalUpsampler,
            "temporal_upsampler",
            DenseReason::NoLinearWeights,
        ),
        (
            LtxComponent::DurationHead,
            "duration_head",
            DenseReason::NoMlxPort("sc-18777"),
        ),
    ] {
        let Some(resolved) = bundle.get(component) else {
            continue;
        };
        sources.insert(name.into(), resolved.path().to_path_buf());
        let raw = Weights::from_file(resolved.path())?;
        embedded.insert(name.into(), resolved.config()?.clone());
        components.push(emit_component(
            out,
            Emit {
                name,
                weights: passthrough_map(&raw),
                metadata: component_metadata(resolved, tier),
                exempt: Some(exempt),
            },
            tier,
            group_size,
            |_| false,
        )?);
        drop(raw);
        mlx_rs::memory::clear_cache();
    }

    // ---- sidecars ---------------------------------------------------------------------------------
    let mut bytes: u64 = components.iter().map(|c| c.bytes).sum();
    let report = LtxTierReport {
        tier,
        dir: out.to_path_buf(),
        bits: tier.bits(),
        group_size,
        components,
        bytes,
    };
    write_json(
        out.join(EMBEDDED_CONFIG_FILE),
        &serde_json::Value::Object(embedded),
    )?;
    write_json(out.join(TIER_MANIFEST_FILE), &report.manifest(&sources))?;
    for sidecar in [EMBEDDED_CONFIG_FILE, TIER_MANIFEST_FILE] {
        bytes += std::fs::metadata(out.join(sidecar))?.len();
    }
    Ok(LtxTierReport { bytes, ..report })
}

/// Build every tier of an LTX-2.5 bundle under `out_root`, one subdirectory per tier.
///
/// Tiers are built **sequentially** and each one's arrays are released before the next starts. The
/// source components are re-read per tier rather than held: re-reading 42 GB from a memory-mapped
/// file costs page cache, holding it costs resident RAM, and this conversion runs beside nothing
/// else on a machine that has to survive it.
pub fn convert_2_5_tiers(
    bundle: &LtxBundle,
    out_root: impl AsRef<Path>,
    tiers: &[LtxTier],
    group_size: i32,
) -> Result<Vec<LtxTierReport>> {
    let root = out_root.as_ref();
    let mut reports = Vec::with_capacity(tiers.len());
    for tier in tiers {
        let report = convert_2_5_tier(bundle, root.join(tier.id()), *tier, group_size)?;
        mlx_rs::memory::clear_cache();
        reports.push(report);
    }
    Ok(reports)
}

/// Every tensor of a source file, keys and values verbatim — the shape for components that need no
/// key remap (the text encoder, the latent upsamplers, the duration head).
fn passthrough_map(raw: &Weights) -> HashMap<String, Array> {
    raw.keys()
        .map(|k| {
            (
                k.to_string(),
                raw.require(k).expect("key from keys()").clone(),
            )
        })
        .collect()
}

/// Pretty-print a JSON value to `path`.
fn write_json(path: PathBuf, value: &serde_json::Value) -> Result<()> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|e| Error::Msg(format!("serialize {}: {e}", path.display())))?;
    std::fs::write(&path, text)?;
    Ok(())
}

#[cfg(test)]
mod tests;
