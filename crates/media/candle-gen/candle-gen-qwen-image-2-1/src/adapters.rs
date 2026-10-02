//! LoRA / LoKr / LoHa on the Qwen-Image 2.1 DiT (sc-24157) — the candle twin of the MLX host's
//! adapter consumption.
//!
//! Two routes, chosen per adapter file from the file's **own** keys and metadata:
//!
//! * **LoRA and PEFT-stamped LoKr** ride as forward-time additive residuals through the shared
//!   [`candle_gen::quant::install_dotted_adapters`] over
//!   [`QwenImage21Transformer::visit_adaptable_mut`] — the FLUX.2 pattern. The base weight is never
//!   touched, so the same install serves the dense bf16 tier and the packed q8/q4 tiers at the base's
//!   footprint. Adapters stack in request order, each at its own `AdapterSpec::scale`.
//! * **LyCORIS LoHa** (`hada_w1_a/b`, `hada_w2_a/b`) has no deferred additive form — the Hadamard
//!   product of two low-rank pairs is not a low-rank matrix — so it is **folded into the dense
//!   weight** (`W ← W + δ`), the `candle-gen-qwen-image` precedent. That needs a dense weight, so it
//!   is served on the **bf16 tier only**: on a packed q8/q4 tier LoHa is refused with a typed
//!   `Unsupported` naming LoHa and the tier ([`loha_on_packed_tier_refusal`]) — never silently
//!   skipped and never served by dequantizing the base.
//!
//! **Strict.** Every target in every selected file must reach a DiT projection: an adapter key that
//! resolves to no projection (another model's module tree, a text-encoder LoRA, a mis-shaped factor)
//! is a load error naming the file and the key, not a silently un-adapted render. Two raw keys that
//! name the same projection (`transformer.X` beside `X`, or `X` beside `lora_unet_X`) are refused
//! too, never silently collapsed or applied twice.
//!
//! **Weight-free admission.** [`plan`] (and its [`preflight`] wrapper) runs before any weight is
//! read, under every offload policy — `load` reaches it through the memory contract, which prices
//! the stack from it (under `Sequential` the DiT — and so [`install`] — is deferred to the first
//! render). It reads only the safetensors **headers** of each file and resolves every key against
//! [`ProjectionTable`], the projection set `transformer/config.json` declares, checking LoRA and
//! LoHa factor shapes against each projection's `[out, in]` (LoKr is key-matched there; its
//! Kronecker factor shapes are checked at install).
//!
//! **Formats (sc-24158)** — every spelling the MLX twin's shared loader accepts for this DiT:
//!
//! * **PEFT / diffusers / ComfyUI / ai-toolkit LoRA** — `lora_A`/`lora_B` (optionally
//!   `.default`) or `lora_down`/`lora_up` factors plus an optional per-module `.alpha`, under no
//!   namespace, `transformer.`, `diffusion_model.` or a raw PEFT `base_model.model.` wrapper.
//! * **kohya LoRA** — `lora_unet_<dotted path with . → _>` (`lora_unet_txt_in_in_layer`,
//!   `lora_unet_transformer_blocks_0_attn_to_out_0`, `lora_unet_norm_out_linear`, …), resolved by
//!   exact match against the flattened projection table, so names that already contain `_`
//!   (`img_mlp.gate_layer`, `to_out.0`) are never mis-split.
//! * **SceneWorks / PEFT-stamped LoKr** (`networkType=lokr`, global `rank`/`alpha`) — the shared
//!   additive install.
//! * **Third-party LyCORIS LoKr** (`lokr_*` factors with no `networkType` stamp — lycoris-lib's
//!   `lycoris_…`, kohya's `lora_unet_…`, ai-toolkit's dotted `diffusion_model.…`) — this crate's
//!   [`install`] derives the LyCORIS **per-module** scale (`alpha / lora_dim`, forced 1 when both
//!   Kronecker factors are full), collapses a Linear tucker `lokr_t2` (`[r, r, 1, 1]`) into its
//!   right factor, and attaches the structured Kronecker residual — so it serves the dense and the
//!   packed tiers alike. Flattened keys resolve prefix-agnostically by the longest `_`-delimited
//!   projection stem, as the MLX `apply_lokr_thirdparty` does.
//! * **LyCORIS LoHa** — folded (bf16 tier only, see above), under the same key spellings.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::weightsmeta as wmeta;
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::quant::{AdaptLinear, LokrFactors};
use candle_gen::train::merge::{parse_loha_thirdparty, read_adapter, read_scalar_opt, AdapterFile};
use candle_gen::{CandleError as Error, Result};

use crate::config::TransformerConfig;
use crate::quant::Tier;
use crate::transformer::QwenImage21Transformer;
use crate::MODEL_ID;

/// What one install applied.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdapterInstallReport {
    /// LoRA / LoKr residuals attached (one per `(projection, file)` hit; stacked files count once
    /// each).
    pub residuals: usize,
    /// LoHa deltas folded into dense projection weights.
    pub loha_folds: usize,
}

/// The typed refusal for a LoHa adapter on a packed tier. Names LoHa, the tier and the way out.
pub fn loha_on_packed_tier_refusal(tier: Tier, adapter: &std::path::Path) -> String {
    format!(
        "{MODEL_ID}: LoHa adapter {} cannot apply on the packed {} tier — LoHa has no additive \
         form over quantized weights and is merged into dense weights only; load the bf16 tier to \
         use it",
        adapter.display(),
        tier.dir_name()
    )
}

/// Whether `file` is a LyCORIS LoHa (keyed by its own tensors, never by the declared kind).
fn is_loha(file: &AdapterFile) -> bool {
    wmeta::keys_contain_loha(file.tensors.keys().map(String::as_str))
}

fn check_spec(spec: &AdapterSpec) -> Result<()> {
    if !spec.scale.is_finite() {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: adapter {} scale must be finite, got {}",
            spec.path.display(),
            spec.scale
        )));
    }
    if let Some(expert) = spec.moe_expert {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: adapter {} targets the {expert:?} MoE expert, but this model has a single \
             denoiser",
            spec.path.display()
        )));
    }
    Ok(())
}

/// Every adaptable DiT projection — dotted key → `[out, in]` — plus the kohya-flattened spelling of
/// each, built from the transformer config alone
/// ([`QwenImage21Transformer::adaptable_projections`]).
#[derive(Clone, Debug)]
pub struct ProjectionTable {
    shapes: BTreeMap<String, (usize, usize)>,
    kohya: HashMap<String, String>,
    /// Flattened stem (dotted path with `.` → `_`, no prefix) → dotted path: the prefix-agnostic
    /// LyCORIS resolution table ([`wmeta::resolve_lokr_path`]).
    stems: BTreeMap<String, String>,
}

impl ProjectionTable {
    pub fn from_config(cfg: &TransformerConfig) -> Self {
        let shapes: BTreeMap<String, (usize, usize)> =
            QwenImage21Transformer::adaptable_projections(cfg)
                .into_iter()
                .collect();
        let kohya = shapes
            .keys()
            .map(|path| (kohya_spelling(path), path.clone()))
            .collect();
        let stems = wmeta::kohya_table(&shapes.keys().cloned().collect::<Vec<_>>());
        Self {
            shapes,
            kohya,
            stems,
        }
    }

    /// The table for the snapshot at `root` (`<root>/transformer/config.json`).
    pub fn from_snapshot(root: &Path) -> Result<Self> {
        Ok(Self::from_config(&TransformerConfig::from_json_file(
            &root.join("transformer").join("config.json"),
        )?))
    }

    /// The projection a namespace-stripped module name addresses, by its dotted or exact kohya
    /// (`lora_unet_`) spelling — what the shared LoRA / stamped-LoKr install matches. With
    /// `lycoris`, a third-party LyCORIS key additionally resolves prefix-agnostically
    /// (`lycoris_…`, `lora_unet_…`, any `<PREFIX>_<stem>`) by the longest `_`-delimited stem,
    /// exactly as the MLX `apply_lokr_thirdparty` / `apply_loha_thirdparty` resolve it.
    fn resolve(&self, module: &str, lycoris: bool) -> Option<(&str, (usize, usize))> {
        let path = if self.shapes.contains_key(module) {
            module
        } else if let Some(path) = self.kohya.get(module) {
            path.as_str()
        } else if lycoris {
            wmeta::resolve_lokr_path(module, &self.stems)?
        } else {
            return None;
        };
        self.shapes
            .get_key_value(path)
            .map(|(path, shape)| (path.as_str(), *shape))
    }
}

/// The weight-free admission verdict for a stack: which LoRA files ride as residuals, how many
/// Kronecker-factor elements every LoKr keeps resident, and the `[out, in]` of every projection a
/// LoHa folds into (the fold's load-time transient is sized from the largest of these).
#[derive(Clone, Debug, Default)]
pub struct AdapterPlan {
    /// LoRA files, in request order (resident at their safetensors bytes).
    pub additive: Vec<AdapterSpec>,
    /// Σ over every LoKr module (stamped or LyCORIS) of `a·c + b·d` — the elements of the two small
    /// Kronecker factors `[a, c]` / `[b, d]` the structured residual keeps on device
    /// ([`LokrFactors::resident_f32_bytes`] holds them in f32, and a compute-dtype prepared copy
    /// rides beside them). A low-rank leg is materialized to its full `[b, d]`, so this is not the
    /// file's byte count.
    pub lokr_factor_elements: u64,
    /// One entry per `(LoHa file, projection)` fold.
    pub loha_fold_shapes: Vec<(usize, usize)>,
}

/// Weight-free admission for `specs` against the snapshot at `root`: see [`plan`].
pub fn preflight(root: &Path, specs: &[AdapterSpec], tier: Tier) -> Result<()> {
    plan(&ProjectionTable::from_snapshot(root)?, specs, tier).map(|_| ())
}

/// Weight-free admission from safetensors **headers** only (no tensor data is read): refuses a LoHa
/// on a packed tier, a LoHa declared as LoKr, any key that is not a factor of the file's format,
/// any module that resolves to no projection of `table`, two raw keys naming one module or one
/// projection, a repeated factor, and a LoRA / LoHa factor whose shape does not reconstruct that
/// projection's `[out, in]` in orientation.
pub fn plan(table: &ProjectionTable, specs: &[AdapterSpec], tier: Tier) -> Result<AdapterPlan> {
    let mut plan = AdapterPlan::default();
    for spec in specs {
        check_spec(spec)?;
        let headers = wmeta::safetensors_path_tensor_headers(&spec.path)?;
        let entries: Vec<(&str, &[usize])> = headers
            .iter()
            .map(|header| (header.name.as_str(), header.shape.as_slice()))
            .collect();
        if wmeta::keys_contain_loha(entries.iter().map(|(name, _)| *name)) {
            admit_loha(spec, tier)?;
            for (key, shape, factors) in resolve_modules(spec, table, &entries, Format::Loha)? {
                if factors.contains_key("hada_t1") || factors.contains_key("hada_t2") {
                    return Err(Error::Msg(format!(
                        "{MODEL_ID}: LoHa `{key}` in {} is a conv/tucker LoHa (hada_t1/hada_t2); \
                         the DiT's projections are linear",
                        spec.path.display()
                    )));
                }
                let get = |factor: &str| factors.get(factor).copied();
                check_loha_orientation(
                    &key,
                    shape,
                    get("hada_w1_a"),
                    get("hada_w1_b"),
                    get("hada_w2_a"),
                    get("hada_w2_b"),
                )?;
                plan.loha_fold_shapes.push(shape);
            }
        } else if wmeta::keys_contain_lokr(entries.iter().map(|(name, _)| *name)) {
            let stamped = wmeta::is_lokr_network_type(
                wmeta::safetensors_file_metadata(&spec.path)?
                    .get("networkType")
                    .map(String::as_str),
            );
            if stamped && spec.kind == AdapterKind::Lora {
                return Err(Error::Msg(format!(
                    "{MODEL_ID}: adapter {} was declared LoRA but its metadata says \
                     networkType=lokr",
                    spec.path.display()
                )));
            }
            let format = if stamped {
                Format::Lokr
            } else {
                Format::LokrThirdParty
            };
            for (key, shape, factors) in resolve_modules(spec, table, &entries, format)? {
                check_lokr_factors(spec, &key, &factors)?;
                plan.lokr_factor_elements = plan
                    .lokr_factor_elements
                    .saturating_add(lokr_factor_elements(spec, &key, shape, &factors)?);
            }
        } else {
            for (key, (out_f, in_f), factors) in
                resolve_modules(spec, table, &entries, Format::Lora)?
            {
                let (Some(down), Some(up)) = (factors.get("down"), factors.get("up")) else {
                    return Err(Error::Msg(format!(
                        "{MODEL_ID}: LoRA `{key}` in {} is missing its down/A or up/B factor",
                        spec.path.display()
                    )));
                };
                let oriented = down.len() == 2
                    && up.len() == 2
                    && down[1] == in_f
                    && up[0] == out_f
                    && down[0] == up[1];
                if !oriented {
                    return Err(Error::Msg(format!(
                        "{MODEL_ID}: LoRA `{key}` in {} has down {down:?} / up {up:?}, which does \
                         not reconstruct the projection's [out={out_f}, in={in_f}] (expected down \
                         [r, in], up [out, r])",
                        spec.path.display()
                    )));
                }
            }
            plan.additive.push(spec.clone());
        }
    }
    Ok(plan)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Lora,
    /// A `networkType=lokr`-stamped (SceneWorks / PEFT) LoKr: global `rank`/`alpha`, shared install.
    Lokr,
    /// An unstamped LyCORIS LoKr: per-module `.alpha`, optional tucker `lokr_t2`, this crate's
    /// install.
    LokrThirdParty,
    Loha,
}

impl Format {
    fn name(self) -> &'static str {
        match self {
            Format::Lora => "LoRA",
            Format::Lokr => "LoKr",
            Format::LokrThirdParty => "LyCORIS LoKr",
            Format::Loha => "LoHa",
        }
    }

    /// Whether this format's install resolves keys prefix-agnostically (the LyCORIS routes) or
    /// only by the exact dotted / `lora_unet_` spelling (the shared additive install).
    fn lycoris(self) -> bool {
        matches!(self, Format::LokrThirdParty | Format::Loha)
    }

    /// Split `key` into its raw module path and factor name, mirroring what the install consumes
    /// (`candle_gen::quant::install_dotted_adapters` for LoRA / LoKr, [`parse_loha_thirdparty`]
    /// for LoHa); `None` is a key the install could not apply.
    fn split(self, key: &str) -> Option<(&str, &'static str)> {
        let table: &[(&str, &'static str)] = match self {
            Format::Lora => &[
                (".lora_A.default.weight", "down"),
                (".lora_B.default.weight", "up"),
                (".lora_A.weight", "down"),
                (".lora_B.weight", "up"),
                (".lora_down.weight", "down"),
                (".lora_up.weight", "up"),
                (".lora_down", "down"),
                (".lora_up", "up"),
                (".alpha", "alpha"),
            ],
            Format::Lokr => &[
                (".lokr_w1_a", "lokr_w1_a"),
                (".lokr_w1_b", "lokr_w1_b"),
                (".lokr_w1", "lokr_w1"),
                (".lokr_w2_a", "lokr_w2_a"),
                (".lokr_w2_b", "lokr_w2_b"),
                (".lokr_w2", "lokr_w2"),
            ],
            Format::LokrThirdParty => &[
                (".lokr_w1_a", "lokr_w1_a"),
                (".lokr_w1_b", "lokr_w1_b"),
                (".lokr_w1", "lokr_w1"),
                (".lokr_w2_a", "lokr_w2_a"),
                (".lokr_w2_b", "lokr_w2_b"),
                (".lokr_w2", "lokr_w2"),
                (".lokr_t2", "lokr_t2"),
                (".alpha", "alpha"),
            ],
            Format::Loha => &[
                (".hada_w1_a", "hada_w1_a"),
                (".hada_w1_b", "hada_w1_b"),
                (".hada_w2_a", "hada_w2_a"),
                (".hada_w2_b", "hada_w2_b"),
                (".hada_t1", "hada_t1"),
                (".hada_t2", "hada_t2"),
                (".alpha", "alpha"),
            ],
        };
        table
            .iter()
            .find_map(|(suffix, factor)| key.strip_suffix(suffix).map(|module| (module, *factor)))
    }
}

/// One file's modules, resolved: `(raw module key, projection [out, in], factor → shape)`.
type ResolvedModules<'a> = Vec<(String, (usize, usize), BTreeMap<&'static str, &'a [usize]>)>;

/// Group `entries` by module and resolve each against `table`, strictly: a key that is not a
/// factor of `format`, a module that addresses no projection, two raw keys collapsing to one
/// module (`transformer.X` / `X`) or one projection (`X` / `lora_unet_X`), and a repeated factor
/// are all refusals naming the file and key.
fn resolve_modules<'a>(
    spec: &AdapterSpec,
    table: &ProjectionTable,
    entries: &[(&'a str, &'a [usize])],
    format: Format,
) -> Result<ResolvedModules<'a>> {
    let file = spec.path.display();
    let kind = format.name();
    type Module<'k> = (&'k str, BTreeMap<&'static str, &'k [usize]>);
    let mut modules: BTreeMap<&str, Module<'a>> = BTreeMap::new();
    for &(key, shape) in entries {
        let Some((raw, factor)) = format.split(key) else {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: {kind} adapter {file} carries `{key}`, which is not a {kind} factor; \
                 refusing a partial apply"
            )));
        };
        let module = strip_namespace(raw);
        let (first_raw, factors) = modules
            .entry(module)
            .or_insert_with(|| (raw, BTreeMap::new()));
        if *first_raw != raw {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: {kind} adapter {file} carries both `{first_raw}` and `{raw}`, which \
                 name the same module `{module}`; refusing an ambiguous apply"
            )));
        }
        if factors.insert(factor, shape).is_some() {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: {kind} adapter {file} repeats the {factor} factor of `{raw}` \
                 (second: `{key}`)"
            )));
        }
    }
    let mut targeted: HashMap<&str, &str> = HashMap::new();
    let mut resolved = Vec::with_capacity(modules.len());
    for (module, (raw, factors)) in modules {
        let Some((path, shape)) = table.resolve(module, format.lycoris()) else {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: {kind} adapter {file} targets `{raw}`, which matches no DiT \
                 projection; every target must apply — expected diffusers/PEFT keys (or their \
                 kohya `lora_unet_` spelling) over `transformer_blocks.{{i}}.{{attn.to_q|to_k|\
                 to_v|to_out.0, img_mlp.gate_layer|proj|out}}` and the embedder / modulation / \
                 output projections"
            )));
        };
        if let Some(previous) = targeted.insert(path, raw) {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: {kind} adapter {file} carries both `{previous}` and `{raw}`, which \
                 target the same projection `{path}`; refusing a double apply"
            )));
        }
        resolved.push((raw.to_owned(), shape, factors));
    }
    Ok(resolved)
}

fn admit_loha(spec: &AdapterSpec, tier: Tier) -> Result<()> {
    if spec.kind == AdapterKind::Lokr {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: adapter {} was declared LoKr but its keys are LyCORIS LoHa (hada_*)",
            spec.path.display()
        )));
    }
    if tier != Tier::Bf16 {
        return Err(Error::Unsupported(loha_on_packed_tier_refusal(
            tier, &spec.path,
        )));
    }
    Ok(())
}

/// Which factors a LoKr module carries — the one factor-set admission both the header-only
/// [`plan`] and the install ([`LycorisLokr::factors`]) apply, so they cannot disagree.
fn check_lokr_factor_set(has: impl Fn(&str) -> bool) -> std::result::Result<(), &'static str> {
    if has("lokr_w1") && (has("lokr_w1_a") || has("lokr_w1_b")) {
        return Err("carries both a full lokr_w1 and a low-rank lokr_w1_a/lokr_w1_b spelling");
    }
    if has("lokr_w2") && (has("lokr_w2_a") || has("lokr_w2_b") || has("lokr_t2")) {
        return Err(
            "carries both a full lokr_w2 and a low-rank / tucker lokr_w2_a/lokr_w2_b/lokr_t2 \
             spelling",
        );
    }
    let w1 = has("lokr_w1") || (has("lokr_w1_a") && has("lokr_w1_b"));
    let w2 = has("lokr_w2") || (has("lokr_w2_a") && has("lokr_w2_b"));
    if !w1 || !w2 {
        return Err(
            "is missing a Kronecker factor (need lokr_w1 or lokr_w1_a+lokr_w1_b, and lokr_w2, \
             lokr_w2_a+lokr_w2_b, or lokr_t2+lokr_w2_a+lokr_w2_b)",
        );
    }
    if (has("lokr_w1_a") != has("lokr_w1_b")) || (has("lokr_w2_a") != has("lokr_w2_b")) {
        return Err("carries half of a low-rank lokr_w*_a/lokr_w*_b pair");
    }
    Ok(())
}

/// Header-level admission for one LoKr module (stamped or LyCORIS): an unambiguous factor set
/// ([`check_lokr_factor_set`]), every `lokr_w*` factor 2-D, a tucker `lokr_t2` in the
/// `[r, r, 1, 1]` Linear form — a spatial (`kH·kW > 1`) tucker is a conv LoKr this DiT has no
/// target for — and a one-element `alpha` (a present-but-empty one is a malformed file, never a
/// silent fall back to `alpha = rank`).
fn check_lokr_factors(
    spec: &AdapterSpec,
    key: &str,
    factors: &BTreeMap<&'static str, &[usize]>,
) -> Result<()> {
    let file = spec.path.display();
    check_lokr_factor_set(|factor| factors.contains_key(factor))
        .map_err(|what| Error::Msg(format!("{MODEL_ID}: LoKr `{key}` in {file} {what}")))?;
    for (factor, shape) in factors {
        if *factor == "alpha" && shape.iter().product::<usize>() == 0 {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: LoKr `{key}` in {file} has a present but empty alpha {shape:?} — a \
                 malformed file"
            )));
        }
        let ok = match *factor {
            "alpha" => shape.iter().product::<usize>() == 1,
            "lokr_t2" => shape.len() == 4 && shape[2] == 1 && shape[3] == 1,
            _ => shape.len() == 2,
        };
        if !ok {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: LyCORIS LoKr `{key}` in {file} has {factor} {shape:?}, which is not \
                 the Linear form (2-D lokr_w* factors, a [r, r, 1, 1] tucker lokr_t2, a scalar \
                 alpha); a conv LoKr has no target on this DiT"
            )));
        }
    }
    Ok(())
}

/// The Kronecker dimensions `w1 [a, c]` ⊗ `w2 [b, d]` of an admitted LoKr module
/// ([`check_lokr_factors`]) from its header shapes — `w1_a [a, r]`·`w1_b [r, c]`, `w2_a [b, r]`·
/// `w2_b [r, d]`, or the tucker `w2_a [r, b]` / `w2_b [r, d]` — refused unless `a·b × c·d` is the
/// projection's `[out, in]` (so a mis-shaped LoKr fails at the weight-free preflight, not mid-render
/// under `Sequential`). Returns `a·c + b·d`, the elements the structured residual keeps resident.
fn lokr_factor_elements(
    spec: &AdapterSpec,
    key: &str,
    (out_f, in_f): (usize, usize),
    factors: &BTreeMap<&'static str, &[usize]>,
) -> Result<u64> {
    let dim = |factor: &str, axis: usize| factors.get(factor).map(|shape| shape[axis]);
    let (a, c) = match factors.get("lokr_w1") {
        Some(w1) => (Some(w1[0]), Some(w1[1])),
        None => (dim("lokr_w1_a", 0), dim("lokr_w1_b", 1)),
    };
    let (b, d) = match (factors.get("lokr_w2"), factors.contains_key("lokr_t2")) {
        (Some(w2), _) => (Some(w2[0]), Some(w2[1])),
        (None, true) => (dim("lokr_w2_a", 1), dim("lokr_w2_b", 1)),
        (None, false) => (dim("lokr_w2_a", 0), dim("lokr_w2_b", 1)),
    };
    let (Some(a), Some(b), Some(c), Some(d)) = (a, b, c, d) else {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: LoKr `{key}` in {} is missing a Kronecker factor",
            spec.path.display()
        )));
    };
    if a * b != out_f || c * d != in_f {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: LoKr `{key}` in {} does not reconstruct the projection's [out={out_f}, \
             in={in_f}]: w1 [{a}, {c}] ⊗ w2 [{b}, {d}] is [{}, {}]",
            spec.path.display(),
            a * b,
            c * d
        )));
    }
    Ok((a * c + b * d) as u64)
}

/// A LoHa module's factors must reconstruct exactly the projection's `[out, in]` **in orientation**:
/// `hada_w{1,2}_a` `[out, r]`, `hada_w{1,2}_b` `[r, in]`. An element-count match alone is not
/// enough — a transposed module (`[in, r]`·`[r, out]`) reshapes into `[out, in]` and would fold
/// scrambled weights.
fn check_loha_orientation(
    key: &str,
    (out_f, in_f): (usize, usize),
    w1_a: Option<&[usize]>,
    w1_b: Option<&[usize]>,
    w2_a: Option<&[usize]>,
    w2_b: Option<&[usize]>,
) -> Result<()> {
    let (Some(w1_a), Some(w1_b), Some(w2_a), Some(w2_b)) = (w1_a, w1_b, w2_a, w2_b) else {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: LoHa `{key}` is missing a hada_w1/w2 a/b factor"
        )));
    };
    let pair = |a: &[usize], b: &[usize]| {
        a.len() == 2 && b.len() == 2 && a[0] == out_f && b[1] == in_f && a[1] == b[0]
    };
    if pair(w1_a, w1_b) && pair(w2_a, w2_b) {
        return Ok(());
    }
    Err(Error::Msg(format!(
        "{MODEL_ID}: LoHa `{key}` factors hada_w1_a {w1_a:?} / hada_w1_b {w1_b:?} / hada_w2_a \
         {w2_a:?} / hada_w2_b {w2_b:?} are not oriented for the projection's [out={out_f}, \
         in={in_f}] (expected hada_w*_a [out, r] and hada_w*_b [r, in])"
    )))
}

/// The kohya-flattened spelling of the projection at `path`.
fn kohya_spelling(path: &str) -> String {
    format!("lora_unet_{}", path.replace('.', "_"))
}

/// The wrapper a raw PEFT `PeftModel.save_pretrained` puts in front of every module path.
const PEFT_WRAPPER_PREFIX: &str = "base_model.model.";

/// Strip a leading PEFT `base_model.model.` wrapper, then a `transformer.` / `diffusion_model.`
/// namespace — the same normalisation the shared additive install applies.
fn strip_namespace(key: &str) -> &str {
    let key = key.strip_prefix(PEFT_WRAPPER_PREFIX).unwrap_or(key);
    for prefix in wmeta::COMMON_LORA_PREFIXES {
        if let Some(rest) = key.strip_prefix(prefix) {
            return rest;
        }
    }
    key
}

/// Resolve a LyCORIS file's raw module groups to DiT projections, strictly: two raw keys that
/// normalize to one module (`transformer.X` beside `X`) or that resolve to one projection (`X`
/// beside `lora_unet_X` / `lycoris_X`) are refused rather than collapsed or applied twice, and a
/// group that reaches no projection is refused by name. Returns projection path → (raw key, group).
fn resolve_lycoris_groups<G>(
    spec: &AdapterSpec,
    table: &ProjectionTable,
    kind: &str,
    groups: BTreeMap<String, G>,
) -> Result<HashMap<String, (String, G)>> {
    let file = spec.path.display();
    let mut modules: HashMap<String, String> = HashMap::new();
    let mut resolved: HashMap<String, (String, G)> = HashMap::new();
    for (raw, group) in groups {
        let module = strip_namespace(&raw).to_string();
        if let Some(previous) = modules.insert(module.clone(), raw.clone()) {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: {kind} adapter {file} carries both `{previous}` and `{raw}`, which \
                 name the same module `{module}`; refusing an ambiguous apply"
            )));
        }
        let Some((path, _)) = table.resolve(&module, true) else {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: {kind} adapter {file} targets `{raw}`, which matches no DiT \
                 projection; every target must apply"
            )));
        };
        if let Some((previous, _)) = resolved.get(path) {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: {kind} adapter {file} carries both `{previous}` and `{raw}`, which \
                 target the same projection `{path}`; refusing a double apply"
            )));
        }
        resolved.insert(path.to_owned(), (raw, group));
    }
    if resolved.is_empty() {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: {kind} adapter {file} matched no DiT projection"
        )));
    }
    Ok(resolved)
}

/// Fold one LoHa file into the dense DiT at `spec.scale`, strictly: every module group must reach a
/// projection, every key must be a LoHa factor or a per-module `.alpha`, no two raw keys may name
/// one module or one projection, and every module's factors must be oriented for its projection.
fn fold_loha(
    transformer: &mut QwenImage21Transformer,
    spec: &AdapterSpec,
    file: &AdapterFile,
) -> Result<usize> {
    let stray: Vec<&str> = file
        .tensors
        .keys()
        .map(String::as_str)
        .filter(|key| {
            !key.ends_with(".alpha")
                && wmeta::split_factor_key(key, &wmeta::LOHA_TP_SUFFIXES).is_none()
        })
        .collect();
    if let Some(key) = stray.first() {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: LoHa adapter {} carries {} key(s) that are not LoHa factors (first: `{key}`); \
             refusing a partial apply",
            spec.path.display(),
            stray.len()
        )));
    }
    let table = ProjectionTable::from_config(transformer.config());
    let resolved = resolve_lycoris_groups(spec, &table, "LoHa", parse_loha_thirdparty(file)?)?;
    let mut folded: HashSet<String> = HashSet::new();
    transformer.visit_adaptable_mut(&mut |path, linear| {
        let Some((raw, group)) = resolved.get(path) else {
            return Ok(());
        };
        fn dims(t: &Option<Tensor>) -> Option<&[usize]> {
            t.as_ref().map(|t| t.dims())
        }
        check_loha_orientation(
            raw,
            linear.base_shape(),
            dims(&group.w1_a),
            dims(&group.w1_b),
            dims(&group.w2_a),
            dims(&group.w2_b),
        )
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let delta = group
            .delta(linear.base_shape(), spec.scale)
            .map_err(|e| candle_core::Error::Msg(format!("LoHa `{raw}`: {e}")))?;
        linear.fold_dense_delta(&delta).map_err(|e| {
            candle_core::Error::Msg(format!(
                "{MODEL_ID}: LoHa `{raw}` cannot fold into `{path}`: {e}"
            ))
        })?;
        folded.insert(path.to_owned());
        Ok(())
    })?;
    require_all_visited(spec, "LoHa", &resolved, &folded)?;
    Ok(folded.len())
}

/// Every resolved module must have been reached by the visitor walk: the projection table is
/// derived from the config the DiT was built from (and pinned equal to the walk by a test), so a
/// miss is a table / visitor divergence — refused naming the raw keys it would have dropped, never
/// a silently partial apply.
fn require_all_visited<G>(
    spec: &AdapterSpec,
    kind: &str,
    resolved: &HashMap<String, (String, G)>,
    visited: &HashSet<String>,
) -> Result<()> {
    let mut missed: Vec<&str> = resolved
        .iter()
        .filter(|(path, _)| !visited.contains(*path))
        .map(|(_, (raw, _))| raw.as_str())
        .collect();
    if missed.is_empty() {
        return Ok(());
    }
    missed.sort_unstable();
    Err(Error::Msg(format!(
        "{MODEL_ID}: {kind} adapter {} resolved {} module(s) the DiT walk never reached ({}); \
         refusing a partial apply",
        spec.path.display(),
        missed.len(),
        missed.join(", ")
    )))
}

/// Whether `file` is a **third-party** LyCORIS LoKr: `lokr_*` factors without the SceneWorks /
/// PEFT `networkType=lokr` stamp (whose global `rank`/`alpha` the shared install scales by).
fn is_lycoris_lokr(file: &AdapterFile) -> bool {
    wmeta::keys_contain_lokr(file.tensors.keys().map(String::as_str)) && !file.declares_lokr()
}

/// One module of a third-party LyCORIS LoKr: each Kronecker factor full or low-rank, an optional
/// tucker `lokr_t2` on the right factor, and an optional per-module `.alpha`.
#[derive(Default)]
struct LycorisLokr {
    w1: Option<Tensor>,
    w1_a: Option<Tensor>,
    w1_b: Option<Tensor>,
    w2: Option<Tensor>,
    w2_a: Option<Tensor>,
    w2_b: Option<Tensor>,
    t2: Option<Tensor>,
    alpha: Option<f32>,
}

impl LycorisLokr {
    /// LyCORIS `lora_dim`, from whichever decomposed factor is present, in the MLX order:
    /// `lokr_w1_a` `[a, r]`, the tucker `lokr_t2` `[r, r, 1, 1]`, then `lokr_w2_a` `[b, r]`.
    /// `None` when both factors are full.
    fn rank(&self) -> Result<Option<f64>> {
        let (factor, tensor, axis) = if let Some(t) = &self.w1_a {
            ("lokr_w1_a", t, 1)
        } else if let Some(t) = &self.t2 {
            ("lokr_t2", t, 0)
        } else if let Some(t) = &self.w2_a {
            ("lokr_w2_a", t, 1)
        } else {
            return Ok(None);
        };
        match tensor.dims().get(axis) {
            Some(&rank) if rank > 0 => Ok(Some(rank as f64)),
            _ => Err(Error::Msg(format!(
                "LyCORIS LoKr {factor} {:?} carries no rank",
                tensor.dims()
            ))),
        }
    }

    /// LyCORIS `scale = alpha / lora_dim` (alpha defaulting to `lora_dim`), forced 1 when both
    /// Kronecker factors are full (`LokrModule.__init__`: `if use_w1 and use_w2: alpha = lora_dim`).
    fn scale(&self) -> Result<f64> {
        Ok(match self.rank()? {
            None => 1.0,
            Some(rank) => self.alpha.map_or(rank, f64::from) / rank,
        })
    }

    /// The structured Kronecker residual for a `[out, in]` projection at `strength`, with the
    /// per-module LyCORIS scale baked in. A Linear tucker right factor
    /// (`einsum("ijhw,ip,jr->prhw", t2, w2_a, w2_b)` with `h = w = 1`) collapses to the 2-D
    /// `w2_aᵀ · t2 · w2_b` first, so it is deferrable like any other Linear LoKr. `None` when the
    /// factors do not reconstruct `base_shape`.
    fn factors(&self, strength: f32, base_shape: (usize, usize)) -> Result<Option<LokrFactors>> {
        let has = |factor: &str| match factor {
            "lokr_w1" => self.w1.is_some(),
            "lokr_w1_a" => self.w1_a.is_some(),
            "lokr_w1_b" => self.w1_b.is_some(),
            "lokr_w2" => self.w2.is_some(),
            "lokr_w2_a" => self.w2_a.is_some(),
            "lokr_w2_b" => self.w2_b.is_some(),
            _ => self.t2.is_some(),
        };
        check_lokr_factor_set(has).map_err(|what| Error::Msg(format!("module {what}")))?;
        let scale = self.scale()? * f64::from(strength);
        let collapsed = match (&self.w2, &self.t2, &self.w2_a, &self.w2_b) {
            (None, Some(t2), Some(w2_a), Some(w2_b)) => {
                Some(collapse_linear_tucker(t2, w2_a, w2_b)?)
            }
            _ => None,
        };
        let (w2, w2_a, w2_b) = match &collapsed {
            Some(w2) => (Some(w2), None, None),
            None => (self.w2.as_ref(), self.w2_a.as_ref(), self.w2_b.as_ref()),
        };
        LokrFactors::build(
            scale,
            base_shape,
            self.w1.as_ref(),
            self.w1_a.as_ref(),
            self.w1_b.as_ref(),
            w2,
            None,
            w2_a,
            w2_b,
        )
    }
}

/// `w2_aᵀ · t2[:, :, 0, 0] · w2_b` — the LyCORIS tucker rebuild of a Linear (1×1-kernel) LoKr's
/// right factor. A spatial kernel is a conv LoKr, refused.
fn collapse_linear_tucker(t2: &Tensor, w2_a: &Tensor, w2_b: &Tensor) -> Result<Tensor> {
    let (core, wa, wb) = (t2.dims(), w2_a.dims(), w2_b.dims());
    let linear = core.len() == 4
        && core[2] == 1
        && core[3] == 1
        && wa.len() == 2
        && wb.len() == 2
        && wa[0] == core[0]
        && wb[0] == core[1];
    if !linear {
        return Err(Error::Msg(format!(
            "LyCORIS LoKr tucker lokr_t2 {core:?} / lokr_w2_a {wa:?} / lokr_w2_b {wb:?} is not the \
             Linear form ([i, j, 1, 1], [i, p], [j, r])"
        )));
    }
    let core = t2.to_dtype(DType::F32)?.reshape((core[0], core[1]))?;
    let wa = w2_a.to_dtype(DType::F32)?;
    let wb = w2_b.to_dtype(DType::F32)?;
    Ok(wa.t()?.matmul(&core)?.matmul(&wb)?)
}

/// Group a third-party LoKr file by raw module key, strictly: a key that is neither a LoKr factor
/// nor a per-module `.alpha` is refused, never skipped.
fn parse_lycoris_lokr(
    spec: &AdapterSpec,
    file: &AdapterFile,
) -> Result<BTreeMap<String, LycorisLokr>> {
    let mut groups: BTreeMap<String, LycorisLokr> = BTreeMap::new();
    for (key, tensor) in &file.tensors {
        if let Some(raw) = key.strip_suffix(".alpha") {
            let Some(alpha) = read_scalar_opt(key, "alpha", tensor)? else {
                return Err(Error::Msg(format!(
                    "{MODEL_ID}: LyCORIS LoKr adapter {} has a present but empty `{key}` — a \
                     malformed file, not an absent alpha",
                    spec.path.display()
                )));
            };
            groups.entry(raw.to_owned()).or_default().alpha = Some(alpha);
            continue;
        }
        let Some((raw, factor)) = wmeta::split_factor_key(key, &wmeta::LOKR_TP_SUFFIXES) else {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: LyCORIS LoKr adapter {} carries `{key}`, which is not a LoKr factor; \
                 refusing a partial apply",
                spec.path.display()
            )));
        };
        let group = groups.entry(raw.to_owned()).or_default();
        let slot = match factor {
            "lokr_w1" => &mut group.w1,
            "lokr_w1_a" => &mut group.w1_a,
            "lokr_w1_b" => &mut group.w1_b,
            "lokr_w2" => &mut group.w2,
            "lokr_w2_a" => &mut group.w2_a,
            "lokr_w2_b" => &mut group.w2_b,
            _ => &mut group.t2,
        };
        *slot = Some(tensor.clone());
    }
    Ok(groups)
}

/// Attach one third-party LyCORIS LoKr file as structured Kronecker residuals — over a dense or a
/// packed projection alike (the base is never touched). Strict: every module must resolve to a
/// projection and its factors must reconstruct that projection's `[out, in]`.
fn install_lycoris_lokr(
    transformer: &mut QwenImage21Transformer,
    spec: &AdapterSpec,
    file: &AdapterFile,
    device: &Device,
) -> Result<usize> {
    let table = ProjectionTable::from_config(transformer.config());
    let resolved = resolve_lycoris_groups(
        spec,
        &table,
        "LyCORIS LoKr",
        parse_lycoris_lokr(spec, file)?,
    )?;
    let mut attached: HashSet<String> = HashSet::new();
    transformer.visit_adaptable_mut(&mut |path, linear: &mut AdaptLinear| {
        let Some((raw, group)) = resolved.get(path) else {
            return Ok(());
        };
        let message = |e: Error| candle_core::Error::Msg(format!("LyCORIS LoKr `{raw}`: {e}"));
        let factors = group
            .factors(spec.scale, linear.base_shape())
            .map_err(message)?
            .ok_or_else(|| {
                let (out_f, in_f) = linear.base_shape();
                candle_core::Error::Msg(format!(
                    "{MODEL_ID}: LyCORIS LoKr `{raw}` in {} does not reconstruct `{path}`'s \
                     [out={out_f}, in={in_f}] as a Kronecker product",
                    spec.path.display()
                ))
            })?;
        linear
            .push_lokr_structured(factors.to_device(device).map_err(message)?)
            .map_err(message)?;
        attached.insert(path.to_owned());
        Ok(())
    })?;
    require_all_visited(spec, "LyCORIS LoKr", &resolved, &attached)?;
    Ok(attached.len())
}

/// Install `specs` on a loaded DiT: LoHa files fold into the dense weights (bf16 tier only),
/// third-party LyCORIS LoKr files attach as structured residuals, then every LoRA / stamped LoKr
/// file attaches as a stacked additive residual through the shared install. `tier` is the tier the
/// DiT was loaded from; `device` the DiT's device. A no-op for an empty stack.
pub fn install(
    transformer: &mut QwenImage21Transformer,
    specs: &[AdapterSpec],
    tier: Tier,
    device: &Device,
) -> Result<AdapterInstallReport> {
    let mut report = AdapterInstallReport::default();
    if specs.is_empty() {
        return Ok(report);
    }
    let mut additive = Vec::new();
    for spec in specs {
        check_spec(spec)?;
        let file = read_adapter(&spec.path)?;
        if is_loha(&file) {
            admit_loha(spec, tier)?;
            report.loha_folds += fold_loha(transformer, spec, &file)?;
        } else if is_lycoris_lokr(&file) {
            report.residuals += install_lycoris_lokr(transformer, spec, &file, device)?;
        } else {
            additive.push(spec.clone());
        }
    }
    if additive.is_empty() {
        return Ok(report);
    }
    let applied =
        candle_gen::quant::install_dotted_adapters(MODEL_ID, &additive, device, |visitor| {
            transformer.visit_adaptable_mut(visitor)
        })?;
    if let Some(first) = applied.skipped_targets.first() {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: {} adapter target(s) match no DiT projection (first: `{first}`); every \
             target must apply — expected diffusers/PEFT keys (or their kohya `lora_unet_` \
             spelling) over `transformer_blocks.{{i}}.{{attn.to_q|to_k|to_v|to_out.0, \
             img_mlp.gate_layer|proj|out}}` and the embedder / modulation / output projections",
            applied.skipped_targets.len()
        )));
    }
    if applied.skipped_keys > 0 {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: {} adapter key(s) were not applicable (non-factor keys, half pairs or \
             factors whose shape does not match the projection); refusing a partial apply",
            applied.skipped_keys
        )));
    }
    report.residuals += applied.applied;
    Ok(report)
}
