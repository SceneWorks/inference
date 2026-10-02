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

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use candle_core::Device;
use candle_gen::gen_core::weightsmeta as wmeta;
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::train::merge::{parse_loha_thirdparty, read_adapter, AdapterFile, ThirdPartyLoha};
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
        Self { shapes, kohya }
    }

    /// The table for the snapshot at `root` (`<root>/transformer/config.json`).
    pub fn from_snapshot(root: &Path) -> Result<Self> {
        Ok(Self::from_config(&TransformerConfig::from_json_file(
            &root.join("transformer").join("config.json"),
        )?))
    }

    /// The projection a namespace-stripped module name addresses, by its dotted or kohya spelling.
    fn resolve(&self, module: &str) -> Option<(&str, (usize, usize))> {
        let path = if self.shapes.contains_key(module) {
            module
        } else {
            self.kohya.get(module)?.as_str()
        };
        self.shapes
            .get_key_value(path)
            .map(|(path, shape)| (path.as_str(), *shape))
    }
}

/// The weight-free admission verdict for a stack: which files ride as residuals, and the
/// `[out, in]` of every projection a LoHa folds into (the fold's load-time transient is sized from
/// the largest of these).
#[derive(Clone, Debug, Default)]
pub struct AdapterPlan {
    /// LoRA / LoKr files, in request order.
    pub additive: Vec<AdapterSpec>,
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
            // Key-matched only: a LoKr's Kronecker factor shapes are checked at install.
            resolve_modules(spec, table, &entries, Format::Lokr)?;
            plan.additive.push(spec.clone());
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
    Lokr,
    Loha,
}

impl Format {
    fn name(self) -> &'static str {
        match self {
            Format::Lora => "LoRA",
            Format::Lokr => "LoKr",
            Format::Loha => "LoHa",
        }
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
        let Some((path, shape)) = table.resolve(module) else {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: {kind} adapter {file} targets `{raw}`, which matches no DiT \
                 projection; every target must apply — expected diffusers/PEFT keys over \
                 `transformer_blocks.{{i}}.{{attn.to_q|to_k|to_v|to_out.0, \
                 img_mlp.gate_layer|proj|out}}` and the embedder / modulation / output projections"
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

/// The dotted / kohya-flattened spellings an adapter key may use for the projection at `path`.
fn key_candidates(path: &str) -> [String; 2] {
    [path.to_string(), kohya_spelling(path)]
}

/// Strip a leading `transformer.` / `diffusion_model.` namespace.
fn strip_namespace(key: &str) -> &str {
    for prefix in wmeta::COMMON_LORA_PREFIXES {
        if let Some(rest) = key.strip_prefix(prefix) {
            return rest;
        }
    }
    key
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
    // Normalized module → (raw module key, factors). Two raw keys that normalize to one module
    // (`transformer.X` and `X`) are refused, never silently collapsed to whichever sorts last.
    let mut groups: BTreeMap<String, (String, ThirdPartyLoha)> = BTreeMap::new();
    for (raw, group) in parse_loha_thirdparty(file)? {
        let module = strip_namespace(&raw).to_string();
        if let Some((previous, _)) = groups.get(&module) {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: LoHa adapter {} carries both `{previous}` and `{raw}`, which name the \
                 same module `{module}`; refusing an ambiguous apply",
                spec.path.display()
            )));
        }
        groups.insert(module, (raw, group));
    }
    let mut matched = HashSet::new();
    let mut folded = 0usize;
    transformer.visit_adaptable_mut(&mut |path, linear| {
        let hits: Vec<String> = key_candidates(path)
            .into_iter()
            .filter(|key| groups.contains_key(key))
            .collect();
        let key = match hits.as_slice() {
            [] => return Ok(()),
            [key] => key,
            [first, second, ..] => {
                return Err(candle_core::Error::Msg(format!(
                    "{MODEL_ID}: LoHa adapter {} carries both `{first}` and `{second}`, which \
                     target the same projection `{path}`; refusing a double apply",
                    spec.path.display()
                )))
            }
        };
        let (raw, group) = &groups[key];
        matched.insert(key.clone());
        fn dims(t: &Option<candle_core::Tensor>) -> Option<&[usize]> {
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
        folded += 1;
        Ok(())
    })?;
    let unmatched: Vec<&String> = groups
        .iter()
        .filter(|(module, _)| !matched.contains(*module))
        .map(|(_, (raw, _))| raw)
        .collect();
    if let Some(first) = unmatched.first() {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: LoHa adapter {} has {} module(s) that match no DiT projection (first: \
             `{first}`); every target must apply",
            spec.path.display(),
            unmatched.len()
        )));
    }
    if folded == 0 {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: LoHa adapter {} matched no DiT projection",
            spec.path.display()
        )));
    }
    Ok(folded)
}

/// Install `specs` on a loaded DiT: LoHa files fold into the dense weights (bf16 tier only), then
/// every LoRA / LoKr file attaches as a stacked additive residual. `tier` is the tier the DiT was
/// loaded from; `device` the DiT's device. A no-op for an empty stack.
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
             target must apply — expected diffusers/PEFT keys over `transformer_blocks.{{i}}.\
             {{attn.to_q|to_k|to_v|to_out.0, img_mlp.gate_layer|proj|out}}` and the embedder / \
             modulation / output projections",
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
    report.residuals = applied.applied;
    Ok(report)
}
