//! LoRA / LoKr adapters on the Iris backbone (sc-25681), the MLX family convention: every adapter
//! file is installed through the shared strict loader
//! ([`mlx_gen::adapters::loader::apply_adapters_strict`]) as a **forward-time residual** on the
//! projection it targets — `y = base(x) + Σ adapter(x)` — so the base weight is never mutated. A
//! LoRA applies as `scale · (x·Aᵀ)·Bᵀ`; a LoKr / LoHa as its reconstructed `[out, in]` delta (the
//! projections are dense, so the shared loader materializes rather than taking its packed-base
//! structured form), built in the backbone's compute dtype — f32 under `Precision::Fp32`, never
//! rounded to bf16 below it (sc-25690, [`AdaptableHost::lycoris_dtype`]).
//!
//! Upstream Iris-3B ships no adapter code, so the surface is the repo's key layouts: PEFT/diffusers
//! LoRA (`transformer.` / `diffusion_model.` prefixes or bare), kohya flattened LoRA, PEFT-stamped
//! LoKr (`networkType=lokr`), LyCORIS-layout LoKr/LoHa factors — keyed by the upstream `IrisDiT`
//! module path (`blocks.3.attn_proj`, `y_embedder.refiner.proj`, …), which is also the checkpoint
//! key stem.
//!
//! **Identity first.** Before a tensor is read, the file's `__metadata__` must name the Iris family
//! and this route's task ([`check_adapter_identity`]): the three Iris task backbones share one
//! architecture, so a depth or restoration adapter would otherwise resolve onto the generation
//! backbone without complaint. This holds for **every** key layout above: a file exported by a
//! third-party trainer (kohya, LyCORIS) carries no `family` / `irisTask` stamps and is refused until
//! it is re-stamped; unstamped files are never installed. **Strict install:** an adapter target that resolves to no
//! projection, a file that lands nothing, and a ComfyUI diff-patch file (`.diff` / `.diff_b`, which
//! this residual path does not fold) are typed errors — never a partially adapted render. Each file's
//! outcome is returned as an [`AdapterApplyReport`] (the provider's provenance surface).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use mlx_gen::adapters::loader::apply_adapters_strict;
use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::gen_core::iris::{
    check_adapter_identity, IrisTask, ADAPTER_BASE_MODEL_KEY, ADAPTER_FAMILY_KEY, ADAPTER_TASK_KEY,
};
use mlx_gen::weights::Weights;
use mlx_gen::{AdapterApplyReport, AdapterSpec, Error, Result};
use mlx_rs::{Array, Dtype};

use crate::nn::AdaptedLinears;

#[path = "adapter_load.rs"]
mod load;
pub use load::{AdapterFileReadFacts, AdapterLoadAllocation, ProjectionAdapterLoadFacts};

/// Actual ordered adapter stack, keyed by the checkpoint's upstream projection path. Header
/// pricing and the built graph use this same inventory, never a maximum-rank/image-row proxy.
pub type ProjectionAdapterFacts = BTreeMap<String, Vec<mlx_gen::adapters::facts::AdapterFacts>>;

/// Header-only adapter inventory. `projections` is the retained subset after load; `files` keeps
/// reconstruction inputs separate so a load graph cannot be mistaken for persistent parameters.
#[derive(Clone, Debug, Default)]
pub struct AdapterHeaderInventory {
    pub projections: ProjectionAdapterFacts,
    pub files: Vec<AdapterFileFacts>,
}

#[derive(Clone, Debug)]
pub struct AdapterFileFacts {
    pub path: std::path::PathBuf,
    pub format: mlx_gen::adapters::loader::UnfusedAdapterFormat,
    pub sources: BTreeMap<String, mlx_gen::adapters::facts::TensorFacts>,
    /// Actual scalar-alpha reads selected by this file's loader format (including partial groups).
    pub scalar_reads: BTreeSet<String>,
    pub reconstructions: Vec<AdapterReconstructionFacts>,
}

/// Normalization performed by the actual strict loader before a residual is materialized. The
/// factors here are load-only inputs, not another retained copy of a full LoKr/LoHa delta.
#[derive(Clone, Debug)]
pub enum AdapterReconstructionFacts {
    Lora {
        projection: String,
        down: String,
        up: String,
        alpha_fold: bool,
    },
    Lokr {
        projection: String,
        factors: BTreeMap<String, String>,
    },
    Loha {
        projection: String,
        factors: BTreeMap<String, String>,
    },
}

/// Label of one adapter file's iOS pre-read check.
pub(crate) const ADAPTER_LOAD_QUANTUM: &str = "Iris adapter load";
/// Lazy residual reconstruction, guarded immediately before the Loader evaluates its payloads.
pub(crate) const ADAPTER_MATERIALIZE_QUANTUM: &str = "Iris adapter materialize";
pub(crate) const ADAPTER_HOST_QUANTUM: &str = "Iris adapter host";

/// The dense adapter install host uses one F32 scalar per packed projection as the backing of its
/// zero-stride logical placeholder. The BF16 cast/broadcast expressions are never evaluated by
/// the install; no full dense base or additional cast buffer is created. Count actual packed header
/// projections, including the host's unadapted candidates, not just adapter targets.
pub fn host_constructor_bytes(packed_projections: u64) -> Result<u64> {
    packed_projections
        .checked_mul(4)
        .ok_or_else(|| Error::Unsupported("iris: adapter host scalar byte sum overflow".into()))
}

/// Every 2-D `‹path›.weight` of the checkpoint as an [`AdaptableLinear`] (with its `‹path›.bias`),
/// addressable by dotted module path — the install's host.
struct ProjectionHost {
    linears: BTreeMap<String, AdaptableLinear>,
    /// The backbone's compute dtype — also the dtype a LoKr / LoHa residual is built in, so an
    /// FP32 load keeps the adapter f32 (as the Candle twin keeps its LoKr factors f32).
    compute: Dtype,
}

impl ProjectionHost {
    /// Weights and biases in `compute`, exactly as [`crate::nn::Loader::linear`] builds a bare
    /// projection, so an adapted projection differs from the bare one only by its adapters.
    ///
    /// The host is read through [`peek`], never `Weights::get`/`require`: those mark a key as
    /// consumed, and the backbone build's unused-key check (config vs checkpoint) must see only the
    /// keys the configured architecture reads — not every projection the install could reach.
    fn from_weights(weights: &mut Weights, compute: Dtype) -> Result<Self> {
        let mut keys: Vec<String> = weights
            .keys()
            .filter(|key| key.ends_with(".weight"))
            .map(str::to_owned)
            .collect();
        keys.sort_unstable();
        let mut linears = BTreeMap::new();
        for key in keys {
            let path = key.strip_suffix(".weight").expect("filtered");
            let Some(weight) = peek(weights, &key) else {
                continue;
            };
            if weight.ndim() != 2 {
                continue;
            }
            // A packed artifact's projection ([`crate::packed`]): the install sees a zero-stride
            // placeholder of its LOGICAL `[out, in]` shape (a broadcast scalar — never a dense
            // copy), and [`crate::nn::Loader::linear`] moves the installed adapters onto the
            // stored pack. The adapters never read the base, so this is the dense install exactly.
            if let Some(scales) = peek(weights, &format!("{path}.scales")) {
                let group = crate::nn::QUANT_GROUP_SIZE as i32;
                let padded_in = scales.shape()[1] * group;
                let in_features = match peek(
                    weights,
                    &format!("{path}{}", crate::packed::IN_FEATURES_SUFFIX),
                ) {
                    Some(width) => width.item::<i32>(),
                    None => padded_in,
                };
                let placeholder = mlx_rs::ops::broadcast_to(
                    Array::from_f32(0.0).as_dtype(compute)?,
                    &[weight.shape()[0], in_features],
                )?;
                let bias = peek(weights, &format!("{path}.bias"))
                    .map(|b| b.as_dtype(compute))
                    .transpose()?;
                linears.insert(path.to_owned(), AdaptableLinear::dense(placeholder, bias));
                continue;
            }
            let bias = peek(weights, &format!("{path}.bias"))
                .map(|b| b.as_dtype(compute))
                .transpose()?;
            linears.insert(
                path.to_owned(),
                AdaptableLinear::dense(weight.as_dtype(compute)?, bias),
            );
        }
        Ok(Self { linears, compute })
    }
}

/// A handle on `key`'s tensor that leaves `weights`' consumed-key accounting untouched (`remove`
/// and `insert` are not reads); the array is reference-counted, so this copies no data.
fn peek(weights: &mut Weights, key: &str) -> Option<Array> {
    let array = weights.remove(key)?;
    weights.insert(key, array.clone());
    Some(array)
}

/// Shapes the actual dense install host exposes, including the logical (unpadded) width of an
/// already-packed artifact. Only the packed width scalar is read; no base cast/placeholder is built.
fn logical_projection_shapes(weights: &mut Weights) -> Result<(BTreeMap<String, [i32; 2]>, u64)> {
    // Borrow metadata only: get() would mark consumption and clone key strings; neither a
    // cloned key list nor a tensor/shape allocation belongs ahead of this native check.
    let mut packed = 0_u64;
    for key in weights.keys() {
        let Some(path) = key.strip_suffix(".weight") else {
            continue;
        };
        let Some(weight) = weights.peek(key) else {
            continue;
        };
        if weight.ndim() == 2
            && weights
                .keys()
                .any(|name| name.strip_suffix(".scales") == Some(path))
        {
            packed = packed.checked_add(1).ok_or_else(|| {
                Error::Unsupported("iris: packed adapter-host projection count overflow".into())
            })?;
        }
    }
    if packed != 0 {
        // Packed-format width markers are scalar I32 (four bytes); validate that metadata below
        // before try_item. This guard protects the sum of their native source payloads.
        mlx_gen::ios_quantum::before(ADAPTER_HOST_QUANTUM, host_constructor_bytes(packed)?)?;
    }
    let keys: Vec<String> = weights
        .keys()
        .filter(|k| k.ends_with(".weight"))
        .map(str::to_owned)
        .collect();
    let mut shapes = BTreeMap::new();
    for key in keys {
        let Some(weight) = peek(weights, &key) else {
            continue;
        };
        if weight.ndim() != 2 {
            continue;
        }
        let path = key.strip_suffix(".weight").expect("filtered weight");
        let input = if let Some(scales) = peek(weights, &format!("{path}.scales")) {
            if scales.ndim() != 2 {
                return Err(Error::Unsupported(format!(
                    "iris: {path} has non-matrix packed scales"
                )));
            }
            let padded = scales.shape()[1]
                .checked_mul(crate::nn::QUANT_GROUP_SIZE as i32)
                .ok_or_else(|| {
                    Error::Unsupported(format!("iris: {path} packed width overflows MLX shape"))
                })?;
            match peek(
                weights,
                &format!("{path}{}", crate::packed::IN_FEATURES_SUFFIX),
            ) {
                Some(width) => {
                    if width.dtype() != Dtype::Int32 || !width.shape().is_empty() {
                        return Err(Error::Unsupported(format!(
                            "iris: {path} packed width must be a scalar I32"
                        )));
                    }
                    width.try_item::<i32>()?
                }
                None => padded,
            }
        } else {
            weight.shape()[1]
        };
        shapes.insert(path.to_owned(), [weight.shape()[0], input]);
    }
    Ok((shapes, packed))
}

impl AdaptableHost for ProjectionHost {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        self.linears.get_mut(&path.join("."))
    }

    fn adaptable_paths(&self) -> Vec<String> {
        self.linears.keys().cloned().collect()
    }

    fn lycoris_dtype(&self) -> Dtype {
        self.compute
    }
}

/// The identity stamps of an adapter file's `__metadata__`.
pub fn adapter_identity(path: &Path) -> Result<HashMap<String, String>> {
    let file = Weights::from_file(path)
        .map_err(|e| Error::Msg(format!("iris: reading adapter {}: {e}", path.display())))?;
    Ok(
        [ADAPTER_FAMILY_KEY, ADAPTER_BASE_MODEL_KEY, ADAPTER_TASK_KEY]
            .into_iter()
            .filter_map(|key| {
                file.metadata(key)
                    .map(|value| (key.to_owned(), value.to_owned()))
            })
            .collect(),
    )
}

/// Install `specs` (in order, stacking) onto the projections of `weights` at the `compute` dtype for
/// `task` on `base_model`. Returns the adapted projections for the model build and one report per
/// file. An empty `specs` is `Ok(None)` (the bare build).
pub fn install(
    weights: &mut Weights,
    compute: Dtype,
    specs: &[AdapterSpec],
    task: IrisTask,
    base_model: &str,
) -> Result<Option<(AdaptedLinears, Vec<AdapterApplyReport>)>> {
    if specs.is_empty() {
        return Ok(None);
    }
    // Same header normalization as pre-load pricing. No adapter tensor or MLX model is allocated
    // by this inventory; the base is already at its caller-admitted load boundary.
    let (shapes, packed) = logical_projection_shapes(weights)?;
    let inventory = header_inventory(specs, &shapes, compute, task, base_model)?;
    let load_facts = inventory.projection_loads()?;
    let file_reads = inventory.file_read_facts()?;
    let host_bytes = host_constructor_bytes(packed)?;
    // Only packed projections construct scalar-backed placeholders. A dense install creates no
    // native host payload here: its first allocating boundary is the file's ADAPTER_LOAD check.
    if host_bytes != 0 {
        mlx_gen::ios_quantum::before(ADAPTER_HOST_QUANTUM, host_bytes)?;
    }
    let mut host = ProjectionHost::from_weights(weights, compute)?;
    let mut reports = Vec::with_capacity(specs.len());
    for (spec, read) in specs.iter().zip(&file_reads) {
        // Guard immediately before the strict loader opens this file and creates its lazy graph
        // (including strong F32 alpha scalars). Earlier file payloads/base are already held, not
        // re-added here. The later per-projection guard owns reconstruction evaluation scratch.
        mlx_gen::ios_quantum::before(ADAPTER_LOAD_QUANTUM, read.read_guard_bytes()?)?;
        let report = apply_adapters_strict(&mut host, std::slice::from_ref(spec), base_model)?;
        reports.push(AdapterApplyReport {
            adapter_path: spec.path.clone(),
            applied: report.applied,
            skipped: report.unmatched_paths,
        });
    }
    let adapted: BTreeSet<String> = host
        .linears
        .iter()
        .filter(|(_, linear)| !linear.adapters().is_empty())
        .map(|(path, _)| path.clone())
        .collect();
    Ok(Some((
        AdaptedLinears::new(host.linears, adapted).with_load_facts(load_facts),
        reports,
    )))
}

/// Logical shapes of the same projections ProjectionHost exposes. Header-only except for the
/// packed format's optional four-byte `.in_features` scalar, read with the shared bounded payload
/// reader. No weights or MLX arrays are loaded before admission. The second result is the actual
/// number of packed candidates whose install host creates a scalar-backed placeholder.
pub fn projection_shapes_from_headers(path: &Path) -> Result<(BTreeMap<String, [i32; 2]>, u64)> {
    use mlx_gen::gen_core::weightsmeta as wm;
    let headers = wm::safetensors_path_tensor_headers(path).map_err(Error::from)?;
    let by_name: BTreeMap<&str, &wm::SafetensorsTensorHeader> =
        headers.iter().map(|h| (h.name.as_str(), h)).collect();
    let width_keys: BTreeSet<String> = headers
        .iter()
        .filter(|h| h.name.ends_with(".weight") && h.shape.len() == 2)
        .filter_map(|h| {
            let base = h.name.strip_suffix(".weight")?;
            by_name
                .contains_key(format!("{base}.scales").as_str())
                .then(|| format!("{base}{}", crate::packed::IN_FEATURES_SUFFIX))
        })
        .collect();
    let widths = wm::read_safetensors_tensor_payloads(path, |h| width_keys.contains(&h.name), 4)
        .map_err(Error::from)?;
    let to_i32 = |d: usize| {
        i32::try_from(d).map_err(|_| {
            Error::Unsupported("iris: projection dimension exceeds MLX's i32 shape".into())
        })
    };
    let mut projections = BTreeMap::new();
    let mut packed = 0_u64;
    for h in &headers {
        let Some(base) = h.name.strip_suffix(".weight") else {
            continue;
        };
        if h.shape.len() != 2 {
            continue;
        }
        let out = to_i32(h.shape[0])?;
        let input =
            if let Some(scales) = by_name.get(format!("{base}.scales").as_str()) {
                if scales.shape.len() != 2 {
                    return Err(Error::Unsupported(format!(
                        "iris: {base} has non-matrix scales"
                    )));
                }
                packed = packed.checked_add(1).ok_or_else(|| {
                    Error::Unsupported("iris: packed adapter-host projection count overflow".into())
                })?;
                let padded = scales.shape[1]
                    .checked_mul(crate::nn::QUANT_GROUP_SIZE)
                    .ok_or_else(|| {
                        Error::Unsupported("iris: packed projection width overflow".into())
                    })?;
                let padded = to_i32(padded)?;
                let key = format!("{base}{}", crate::packed::IN_FEATURES_SUFFIX);
                if let Some(bytes) = widths.get(&key) {
                    let header = by_name[key.as_str()];
                    if header.dtype != wm::Dtype::I32
                        || header.element_count().map_err(Error::from)? != 1
                        || bytes.len() != 4
                    {
                        return Err(Error::Unsupported(format!(
                            "iris: {key} must be one I32 logical-width scalar"
                        )));
                    }
                    let width =
                        i32::from_le_bytes(bytes.as_slice().try_into().map_err(|_| {
                            Error::Msg("iris: malformed logical-width payload".into())
                        })?);
                    if width <= 0 || width > padded {
                        return Err(Error::Unsupported(format!(
                            "iris: {base} logical width {width} is outside its {padded}-wide pack"
                        )));
                    }
                    width
                } else {
                    padded
                }
            } else {
                to_i32(h.shape[1])?
            };
        projections.insert(base.to_owned(), [out, input]);
    }
    Ok((projections, packed))
}

/// Read only JSON headers and metadata — never an adapter tensor payload or MLX array — and describe the
/// exact residual forms Iris installs on its dense projection host (also for a packed base).
/// `projections` names the actual backbone projections at their logical `[out,in]` shapes.
pub fn header_inventory(
    specs: &[AdapterSpec],
    projections: &BTreeMap<String, [i32; 2]>,
    compute: Dtype,
    task: IrisTask,
    base_model: &str,
) -> Result<AdapterHeaderInventory> {
    use mlx_gen::adapters::facts::{AdapterFacts, TensorFacts};
    use mlx_gen::adapters::loader::{
        claim_lora_factor, classify_unfused_adapter_format, kohya_factor_key, peft_lora_factor,
        strip_common_lora_prefix, KohyaRole, LoraFactorRole, UnfusedAdapterFormat,
    };
    use mlx_gen::gen_core::weightsmeta as wm;

    let table = wm::kohya_table(&projections.keys().cloned().collect::<Vec<_>>());
    let mut inventory = AdapterHeaderInventory::default();
    for spec in specs {
        let headers = wm::safetensors_path_tensor_headers(&spec.path).map_err(Error::from)?;
        let metadata = wm::safetensors_file_metadata(&spec.path).map_err(Error::from)?;
        let identity = metadata
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        check_adapter_identity(&identity, task, base_model, &spec.path).map_err(Error::from)?;
        if spec.pass_scales.is_some() || spec.moe_expert.is_some() {
            return Err(Error::Unsupported(format!(
                "{base_model}: adapter {} sets per-pass scales or an MoE expert; the Iris backbone \
                 is one single-pass denoiser, so only `scale` applies", spec.path.display()
            )));
        }
        if mlx_gen::adapters::loader::has_diff_patch_key_names(
            headers.iter().map(|h| h.name.as_str()),
        ) {
            return Err(Error::Unsupported(format!(
                "{base_model}: adapter {} is a ComfyUI diff-patch (`.diff`/`.diff_b`) file; the Iris \
                 adapter surface is LoRA / LoKr / LoHa low-rank factors", spec.path.display()
            )));
        }
        let format = classify_unfused_adapter_format(
            headers.iter().map(|h| h.name.as_str()),
            metadata.get("networkType").map(String::as_str),
            spec.kind,
        );
        let mut file = AdapterFileFacts {
            path: spec.path.clone(),
            format,
            sources: BTreeMap::new(),
            scalar_reads: BTreeSet::new(),
            reconstructions: Vec::new(),
        };
        for header in &headers {
            let shape = header
                .shape
                .iter()
                .map(|&d| {
                    i32::try_from(d).map_err(|_| {
                        Error::Msg(format!(
                            "iris: adapter tensor {} has a dimension outside MLX's i32 shape",
                            header.name
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let mut stride: usize = 1;
            let mut strides = vec![0; shape.len()];
            for (i, &d) in shape.iter().enumerate().rev() {
                strides[i] = stride;
                stride = stride.checked_mul(d as usize).ok_or_else(|| {
                    Error::Unsupported("iris: adapter header stride product overflow".into())
                })?;
            }
            file.sources.insert(
                header.name.clone(),
                TensorFacts {
                    shape,
                    dtype: native_header_dtype(header.dtype)?,
                    strides,
                },
            );
        }
        if format == UnfusedAdapterFormat::LoraKindMismatch {
            return Err(Error::Msg(format!(
                "adapter {} declared Lora but its metadata says networkType=lokr",
                spec.path.display()
            )));
        }
        if format == UnfusedAdapterFormat::PeftLokr {
            let (rank, alpha) = wm::parse_rank_alpha(
                metadata.get("rank").map(String::as_str),
                metadata.get("alpha").map(String::as_str),
            );
            mlx_gen::adapters::validate_lokr_metadata_scale(alpha, rank)?;
        }
        let disabled = spec.scale == 0.0;
        let before = file.reconstructions.len();
        match format {
            UnfusedAdapterFormat::PeftLora | UnfusedAdapterFormat::KohyaLora => {
                // A group with a missing half is skipped by install_lora_groups, not an installed
                // zero-rank residual. The strict install still rejects a file that lands nothing.
                let prefix =
                    wm::detect_lora_prefix(file.sources.keys().map(String::as_str)).unwrap_or("");
                let cfg = if format == UnfusedAdapterFormat::PeftLora {
                    wm::LoraAdapterMeta::from_metadata(
                        metadata
                            .get(wm::LORA_ADAPTER_METADATA_KEY)
                            .map(String::as_str),
                    )
                } else {
                    None
                };
                let mut groups: BTreeMap<String, (Option<String>, Option<String>, bool)> =
                    BTreeMap::new();
                let mut claimed = BTreeMap::new();
                for key in file.sources.keys() {
                    let parsed = if format == UnfusedAdapterFormat::KohyaLora {
                        if let Some((stem, role)) = kohya_factor_key(key) {
                            let path = table.get(stem).ok_or_else(|| {
                                Error::Msg(format!(
                                    "iris: adapter target {stem} resolves to no projection"
                                ))
                            })?;
                            match role {
                                KohyaRole::Down => {
                                    Some((path.as_str(), Some(LoraFactorRole::Down)))
                                }
                                KohyaRole::Up => Some((path.as_str(), Some(LoraFactorRole::Up))),
                                KohyaRole::Alpha => Some((path.as_str(), None)),
                            }
                        } else {
                            None
                        }
                    } else if let Some((path, role)) = peft_lora_factor(key, prefix) {
                        Some((path, Some(role)))
                    } else {
                        key.strip_prefix(prefix)
                            .and_then(|k| k.strip_suffix(".alpha"))
                            .or_else(|| key.strip_suffix(".alpha"))
                            .map(|path| (path, None))
                    };
                    let Some((path, role)) = parsed else { continue };
                    let group = groups.entry(path.to_owned()).or_default();
                    match role {
                        Some(role) => {
                            claim_lora_factor(&mut claimed, path, role, key)?;
                            let slot = if role == LoraFactorRole::Down {
                                &mut group.0
                            } else {
                                &mut group.1
                            };
                            *slot = Some(key.clone());
                        }
                        None => {
                            group.2 = true;
                            file.scalar_reads.insert(key.clone());
                        }
                    }
                }
                for (path, (down, up, alpha)) in groups {
                    let (Some(down), Some(up)) = (down, up) else {
                        continue;
                    };
                    if !projections.contains_key(&path) {
                        return Err(Error::Msg(format!(
                            "iris: adapter target {path} resolves to no projection"
                        )));
                    }
                    let a = &file.sources[&down];
                    let b = &file.sources[&up];
                    mlx_gen::adapters::loader::validate_lora_shapes(&path, &a.shape, &b.shape)?;
                    let fold =
                        alpha || cfg.as_ref().is_some_and(|m| m.effective(&path).0.is_some());
                    if fold {
                        mlx_gen::adapters::loader::validate_lora_alpha_rank(&path, a.shape[0])?;
                    }
                    // install_lora_groups folds alpha with an F32 Array (not a weak scalar).
                    let b_dtype = if fold {
                        Dtype::from_promoting_types(b.dtype, Dtype::Float32)
                    } else {
                        b.dtype
                    };
                    inventory.projections.entry(path.clone()).or_default().push(
                        AdapterFacts::Lora {
                            a: TensorFacts::materialized_matrix(a.shape[1], a.shape[0], a.dtype),
                            b: TensorFacts::materialized_matrix(b.shape[1], b.shape[0], b_dtype),
                            disabled,
                        },
                    );
                    file.reconstructions.push(AdapterReconstructionFacts::Lora {
                        projection: path,
                        down,
                        up,
                        alpha_fold: fold,
                    });
                }
            }
            UnfusedAdapterFormat::ThirdpartyLokr
            | UnfusedAdapterFormat::PeftLokr
            | UnfusedAdapterFormat::ThirdpartyLoha => {
                let loha = format == UnfusedAdapterFormat::ThirdpartyLoha;
                let suffixes = if loha {
                    &wm::LOHA_TP_SUFFIXES[..]
                } else if format == UnfusedAdapterFormat::PeftLokr {
                    &wm::LOKR_SUFFIXES[..]
                } else {
                    &wm::LOKR_TP_SUFFIXES[..]
                };
                let mut groups: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
                for key in file.sources.keys() {
                    if format != UnfusedAdapterFormat::PeftLokr {
                        if let Some(raw) = key.strip_suffix(".alpha") {
                            file.scalar_reads.insert(key.clone());
                            groups.entry(raw.to_owned()).or_default();
                            continue;
                        }
                    }
                    if let Some((raw, factor)) = wm::split_factor_key(key, suffixes) {
                        groups
                            .entry(raw.to_owned())
                            .or_default()
                            .insert(factor.to_owned(), key.clone());
                    }
                }
                let mut claimed = BTreeSet::new();
                for (raw, factors) in groups {
                    let path = if format == UnfusedAdapterFormat::PeftLokr {
                        raw.clone()
                    } else {
                        wm::resolve_lokr_path(&raw, &table)
                            .unwrap_or_else(|| strip_common_lora_prefix(&raw))
                            .to_owned()
                    };
                    if !claimed.insert(path.clone()) {
                        return Err(Error::Msg(format!(
                            "iris: adapter aliases claim projection {path} twice"
                        )));
                    }
                    let &[out, input] = projections.get(&path).ok_or_else(|| {
                        Error::Msg(format!(
                            "iris: adapter target {path} resolves to no projection"
                        ))
                    })?;
                    let shape = |role: &str| {
                        factors
                            .get(role)
                            .and_then(|key| file.sources.get(key))
                            .map(|t| t.shape.as_slice())
                    };
                    if format == UnfusedAdapterFormat::ThirdpartyLokr {
                        mlx_gen::adapters::loader::thirdparty_lokr_rank(
                            shape("lokr_w1_a"),
                            shape("lokr_t2"),
                            shape("lokr_w2_a"),
                        )?;
                    } else if loha {
                        if ["hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"]
                            .iter()
                            .any(|key| !factors.contains_key(*key))
                        {
                            return Err(Error::Msg(
                                "LoHa: a hada_w1/w2 a/b factor is missing".into(),
                            ));
                        }
                        mlx_gen::adapters::loader::thirdparty_loha_rank(shape("hada_w1_b"))?;
                    }
                    inventory.projections.entry(path.clone()).or_default().push(
                        AdapterFacts::Materialized {
                            delta: TensorFacts::materialized_matrix(out, input, compute),
                            disabled,
                        },
                    );
                    file.reconstructions.push(if loha {
                        AdapterReconstructionFacts::Loha {
                            projection: path,
                            factors,
                        }
                    } else {
                        AdapterReconstructionFacts::Lokr {
                            projection: path,
                            factors,
                        }
                    });
                }
            }
            UnfusedAdapterFormat::LoraKindMismatch => unreachable!("handled before dispatch"),
        }
        if file.reconstructions.len() == before {
            return Err(Error::Msg(format!(
                "iris: adapter {} matched no projection",
                spec.path.display()
            )));
        }
        inventory.files.push(file);
    }
    // Public header facts must not advertise a reconstruction the actual materialization would
    // reject. This shares native-shape/type lowering; it does not evaluate any array or payload.
    inventory.projection_loads()?;
    inventory.file_read_facts()?;
    Ok(inventory)
}

/// Value/path-free semantic facts for the REGISTERED generation adapter path. This uses the real
/// factory's resource/configuration/format validation and the same header inventory as loading;
/// it creates no model or MLX array and reads no adapter/weight values. The packed format's
/// optional four-byte logical-width companion is metadata, read by the existing bounded reader.
///
/// Ordering is explicit: projection/source maps are BTreeMaps and files/stacks retain input order.
/// Dtypes have stable names, never Debug strings or ABI tags. Nonzero continuous strengths and
/// alpha/rank multiplier VALUES are not architecture: factor dimensions, actual alpha-fold dtype
/// widening and the scale-zero graph branch are. These facts are NOT physical/additivity evidence.
pub fn generation_adapter_semantic_facts(spec: &mlx_gen::LoadSpec) -> Result<serde_json::Value> {
    use mlx_gen::adapters::facts::{dtype_name, AdapterFacts};
    use mlx_gen::adapters::loader::UnfusedAdapterFormat;
    use mlx_gen::gen_core::iris::{GenerationResources, BACKBONE_WEIGHTS_FILE};
    use serde_json::json;

    // Reuse the actual pure factory validation, including resources, config, tiers and component
    // refusals. Do not invent defaults or turn arbitrary Unsupported/I/O failures into a skip.
    crate::memory_strategy::memory_strategy_contract(crate::model::MODEL_ID, spec)?;
    let resources = GenerationResources::from_spec(spec, crate::model::MODEL_ID)?;
    let backbone = resources.backbone_dir.join(BACKBONE_WEIGHTS_FILE);
    let (shapes, _) = projection_shapes_from_headers(&backbone)?;
    let compute = crate::model::compute_dtype(spec);
    let inventory = header_inventory(
        &spec.adapters,
        &shapes,
        compute,
        IrisTask::Generation,
        crate::model::MODEL_ID,
    )?;
    inventory.projection_loads()?;
    if inventory.files.len() != spec.adapters.len() {
        return Err(Error::Msg(
            "iris: semantic file inventory does not match the requested adapter stack".into(),
        ));
    }
    let base_artifact_layout = match crate::packed::artifact_tier(&backbone)? {
        None => "dense",
        Some(mlx_gen::Quant::Q4) => "affine_q4",
        Some(mlx_gen::Quant::Q8) => "affine_q8",
        Some(_) => {
            return Err(Error::Unsupported(
                "iris: unknown base artifact layout in adapter facts".into(),
            ))
        }
    };
    let mut projections = Vec::new();
    for (path, stack) in &inventory.projections {
        let base_shape = shapes.get(path).ok_or_else(|| {
            Error::Msg("iris: semantic projection has no actual base shape".into())
        })?;
        let mut residuals = Vec::new();
        for residual in stack {
            residuals.push(match residual {
                AdapterFacts::Lora {a,b,disabled}=>json!({
                    "kind":"lora", "a":semantic_tensor(a,true)?, "b":semantic_tensor(b,true)?,
                    "rank":a.shape[1], "disabled":disabled,
                }),
                AdapterFacts::Materialized {delta,disabled}=>json!({
                    "kind":"materialized_delta", "delta":semantic_tensor(delta,true)?, "disabled":disabled,
                }),
                AdapterFacts::Structured {..}=>return Err(Error::Unsupported("iris: the dense install host does not produce structured adapter facts".into())),
            });
        }
        projections.push(
            json!({"projection":path,"base_logical_shape":base_shape,"residual_stack":residuals}),
        );
    }
    let mut files = Vec::new();
    for (file, adapter) in inventory.files.iter().zip(&spec.adapters) {
        let format = match file.format {
            UnfusedAdapterFormat::PeftLora => "peft_lora",
            UnfusedAdapterFormat::KohyaLora => "kohya_lora",
            UnfusedAdapterFormat::PeftLokr => "peft_lokr",
            UnfusedAdapterFormat::ThirdpartyLokr => "thirdparty_lokr",
            UnfusedAdapterFormat::ThirdpartyLoha => "thirdparty_loha",
            UnfusedAdapterFormat::LoraKindMismatch => {
                return Err(Error::Unsupported(
                    "iris: mismatched declared adapter kind has no semantic facts".into(),
                ))
            }
        };
        let kind = match adapter.kind {
            mlx_gen::AdapterKind::Lora => "lora",
            mlx_gen::AdapterKind::Lokr => "lokr",
        };
        let mut sources = Vec::new();
        for (name, tensor) in &file.sources {
            sources.push(json!({"name":name,"tensor":semantic_tensor(tensor,false)?}));
        }
        let mut reconstructions = Vec::new();
        for plan in &file.reconstructions {
            reconstructions.push(match plan {
                AdapterReconstructionFacts::Lora {projection,down,up,alpha_fold}=>json!({
                    "kind":"lora", "projection":projection, "down":down,"up":up,"alpha_fold":alpha_fold,
                    "a_source_view_only":true,"b_source_view_only":!*alpha_fold,
                }),
                AdapterReconstructionFacts::Lokr {projection,factors}=>json!({
                    "kind":"lokr_dense_delta","projection":projection,"factors":factors,
                    "w1_full_active":factors.contains_key("lokr_w1"),
                    "w2_full_active":factors.contains_key("lokr_w2"),
                    "w2_tucker_active":load::lokr_tucker_active(factors),
                }),
                AdapterReconstructionFacts::Loha {projection,factors}=>json!({
                    "kind":"loha_dense_delta","projection":projection,"factors":factors,
                    "tucker_active":load::loha_tucker_active(factors),
                }),
            });
        }
        // No file.path or scalar/weight values enter this DTO. Inactive factors remain in the
        // source inventory because the strict parser reads them even when reconstruction ignores
        // them; active branch flags come from the SAME predicates as the reconstruction helper.
        files.push(
            json!({"format":format,"declared_kind":kind,"disabled":adapter.scale==0.0,
            "sources":sources,"scalar_reads":file.scalar_reads,"reconstructions":reconstructions}),
        );
    }
    Ok(canonical_semantic_json(
        json!({"schema":"iris_generation_adapter_semantics_v1","compute_dtype":dtype_name(compute),
        "base_artifact_layout":base_artifact_layout,"adapter_storage":"dense_forward_residuals",
        "adapter_input":"logical_unpadded_compute","residual_output":"narrow_to_base_compute",
        "projections":projections,"files":files}),
    ))
}

// Sort recursively even if a consumer enables serde_json's preserve_order feature: a producer's
// Cargo feature union must not change a named-fact digest. Arrays intentionally retain stack order.
fn canonical_semantic_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(object) => {
            let ordered: BTreeMap<_, _> = object
                .into_iter()
                .map(|(k, v)| (k, canonical_semantic_json(v)))
                .collect();
            serde_json::Value::Object(ordered.into_iter().collect())
        }
        serde_json::Value::Array(array) => {
            serde_json::Value::Array(array.into_iter().map(canonical_semantic_json).collect())
        }
        scalar => scalar,
    }
}

fn semantic_tensor(
    tensor: &mlx_gen::adapters::facts::TensorFacts,
    matrix_payload: bool,
) -> Result<serde_json::Value> {
    use serde_json::json;
    let layout = if matrix_payload {
        if tensor.shape.len() != 2 || !tensor.strides.is_empty() {
            return Err(Error::Unsupported(
                "iris: unclassified normalized matrix payload layout".into(),
            ));
        }
        json!({"kind":"dense_matrix_either_order"})
    } else {
        json!({"kind":"exact_strides","strides":tensor.strides})
    };
    Ok(
        json!({"shape":tensor.shape,"dtype":mlx_gen::adapters::facts::dtype_name(tensor.dtype),"layout":layout}),
    )
}

#[cfg(test)]
#[path = "adapter_facts_tests.rs"]
mod adapter_facts_tests;

fn native_header_dtype(dtype: mlx_gen::gen_core::weightsmeta::Dtype) -> Result<Dtype> {
    use mlx_gen::gen_core::weightsmeta::Dtype as S;
    Ok(match dtype {
        S::BOOL => Dtype::Bool,
        S::U8 | S::F8_E4M3 | S::F8_E8M0 => Dtype::Uint8,
        S::I8 => Dtype::Int8,
        S::I16 => Dtype::Int16,
        S::U16 => Dtype::Uint16,
        S::I32 => Dtype::Int32,
        S::U32 => Dtype::Uint32,
        S::I64 => Dtype::Int64,
        S::U64 => Dtype::Uint64,
        S::F16 => Dtype::Float16,
        S::BF16 => Dtype::Bfloat16,
        S::F32 => Dtype::Float32,
        // The native safetensors reader rejects these strings too (not a tier conversion).
        S::F64 | S::F8_E5M2 => {
            return Err(Error::Unsupported(format!(
                "iris: native MLX safetensors does not load {dtype:?}"
            )))
        }
    })
}
