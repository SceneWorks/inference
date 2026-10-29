//! LoRA / LoKr adapters on the Iris backbone (sc-25681), installed as **forward-time residuals**
//! over the frozen base projection (`y = base(x) + r(x)`, [`Residual`]) — the MLX twin's
//! `AdaptableLinear` install, and (sc-25686) exactly the forward the Candle trainer optimizes, so a
//! trained adapter renders here what its training previews rendered. Never merged into the base
//! weight: under the release's bf16 compute a merge rounds every delta below the base's bf16 ulp
//! away. A LoRA is held as its f32 factors (`Aᵀ`, `Bᵀ · scale · alpha/rank` — [`lora_residual`]); a
//! LoKr as its Kronecker factors applied structurally, `w1 · X · w2ᵀ` with `scale · alpha/rank`
//! baked into `w2` ([`lokr_residual`] — the delta `kron(w1, w2)` is never formed); a LoHa as its
//! exact low-rank factorisation when that is smaller than its delta, else the delta
//! ([`crate::nn::Residual`] documents the costs).
//!
//! Upstream Iris-3B ships no adapter code, so the surface is the repo's, the same one the MLX twin
//! installs: PEFT/diffusers LoRA (`transformer.` / `diffusion_model.` prefixes or bare, `lora_A/B`
//! or `lora_down/up`, per-target `.alpha` or the `lora_adapter_metadata` blob), kohya flattened
//! (`lora_unet_…`), PEFT-stamped LoKr (`networkType=lokr`), and LyCORIS-layout LoKr/LoHa factors
//! (no `networkType` stamp) — keyed by the upstream `IrisDiT` module path, which is also the
//! checkpoint key stem.
//!
//! **Identity first** ([`check_adapter_identity`]): the file must name the Iris family and this
//! route's task, because the three task backbones share one architecture. This holds for **every**
//! key layout above: a file exported by a third-party trainer (kohya, LyCORIS) carries no `family`
//! / `irisTask` stamps and is refused until it is re-stamped; unstamped files are never installed. **Strict:** a target
//! that resolves to no 2-D projection, a factor whose delta does not match its base shape, a file
//! that merges nothing, and a diff-patch file are typed errors — never a partially adapted render.

use std::collections::{BTreeMap, HashMap};

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::gen_core::weightsmeta::{
    keys_contain_loha, keys_contain_lokr, kohya_table, resolve_kohya_stem, resolve_lokr_path,
    COMMON_LORA_PREFIXES, KOHYA_PREFIX, LOKR_SUFFIXES,
};
use candle_gen::gen_core::{AdapterApplyReport, AdapterKind, AdapterSpec};
use candle_gen::quant::LokrFactors;
use candle_gen::train::lora::{parse_lokr_metadata, reconstruct_lokr_delta, LoraAdapterMeta};
use candle_gen::train::merge::{
    has_diff_patch_keys, parse_loha_thirdparty, parse_lokr_thirdparty, read_adapter, read_scalar,
    AdapterFile, ThirdPartyLoha,
};
use candle_gen::{CandleError as Error, Result};
use gen_core_iris::{check_adapter_identity, IrisTask};

use crate::nn::Residual;

/// An adapter stack resolved against the backbone: the forward-time residuals of every file, keyed
/// by projection path (`blocks.3.attn_proj`, one entry per file that targets it, in file order),
/// plus one report per file.
pub struct MergedAdapters {
    pub residuals: BTreeMap<String, Vec<Residual>>,
    pub reports: Vec<AdapterApplyReport>,
}

impl MergedAdapters {
    /// The f32 weight delta each adapted projection receives, summed over the stack and keyed by
    /// checkpoint key (`‹path›.weight`) — the merged-weight view of the residuals.
    pub fn deltas(&self) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::with_capacity(self.residuals.len());
        for (path, stack) in &self.residuals {
            let mut sum: Option<Tensor> = None;
            for r in stack {
                let d = r.delta()?;
                sum = Some(match sum {
                    Some(s) => (s + d)?,
                    None => d,
                });
            }
            if let Some(sum) = sum {
                out.insert(format!("{path}.weight"), sum);
            }
        }
        Ok(out)
    }
}

/// A LoRA's residual from its PEFT factors `down = A` `[rank, in]` and `up = B` `[out, rank]`:
/// `a = Aᵀ`, `b = Bᵀ · ((alpha/rank) · scale)`, f32. The one formula the provider's loader and the
/// trainer's forward share (both scale `b` the same way, so a trained file renders exactly).
pub fn lora_residual(
    down: &Tensor,
    up: &Tensor,
    alpha: f32,
    rank: f32,
    scale: f32,
) -> Result<Residual> {
    let eff = (alpha as f64 / rank as f64) * scale as f64;
    Ok(Residual::Lora {
        a: down.to_dtype(DType::F32)?.t()?,
        b: (up.to_dtype(DType::F32)?.t()? * eff)?,
    })
}

/// A LoKr's residual from its factors (full `w1`/`w2` or low-rank `_a`/`_b` legs), scaled by
/// `(alpha/rank) · scale`: the structured Kronecker product ([`Residual::Kron`]) when the factors
/// decompose the `[out, in]` projection, else the reconstructed delta. The one builder the
/// provider's loader and the trainer's forward share, so a trained file renders exactly.
#[allow(clippy::too_many_arguments)]
pub fn lokr_residual(
    w1: Option<&Tensor>,
    w1_a: Option<&Tensor>,
    w1_b: Option<&Tensor>,
    w2: Option<&Tensor>,
    w2_a: Option<&Tensor>,
    w2_b: Option<&Tensor>,
    alpha: f32,
    rank: f32,
    scale: f32,
    shape: (usize, usize),
) -> Result<Residual> {
    if !rank.is_finite() || rank <= 0.0 || !alpha.is_finite() || !scale.is_finite() {
        return Err(Error::Msg(format!(
            "iris: LoKr scale is invalid (rank = {rank}, alpha = {alpha}, scale = {scale})"
        )));
    }
    let eff = (alpha as f64 / rank as f64) * scale as f64;
    match LokrFactors::build(eff, shape, w1, w1_a, w1_b, w2, None, w2_a, w2_b)? {
        Some(f) => Ok(Residual::Kron(f)),
        None => reconstruct_lokr_delta(w1, w1_a, w1_b, w2, w2_a, w2_b, alpha, rank, scale, shape)
            .map(Residual::Delta),
    }
}

/// A LyCORIS LoHa's residual, `s · (A₁B₁) ⊙ (A₂B₂)` with `A₍ᵢ₎` `[out, rᵢ]`, `B₍ᵢ₎` `[rᵢ, in]`.
/// The Hadamard product of two low-rank products is exactly rank `r₁r₂`: with the row-wise Kronecker
/// `U[o, (p, q)] = A₁[o, p]·A₂[o, q]` and `V[(p, q), i] = B₁[p, i]·B₂[q, i]`, `δ = s · U · V`. When
/// `r₁r₂·(in + out) < in·out` that factorisation is smaller than the delta (and so is its GEMM work)
/// and is installed as [`Residual::LowRank`]; otherwise the dense delta is cheaper and installed.
pub(crate) fn loha_residual(
    g: &ThirdPartyLoha,
    shape: (usize, usize),
    user_scale: f32,
) -> Result<Residual> {
    let (out_f, in_f) = shape;
    let factors = match (&g.w1_a, &g.w1_b, &g.w2_a, &g.w2_b) {
        (Some(a1), Some(b1), Some(a2), Some(b2)) => [a1, b1, a2, b2].map(|t| t.dims().to_vec()),
        _ => return g.delta(shape, user_scale).map(Residual::Delta),
    };
    let rank_of = |a: &[usize], b: &[usize]| {
        (a.len() == 2 && b.len() == 2 && a[0] == out_f && b[1] == in_f && a[1] == b[0])
            .then_some(a[1])
    };
    let (Some(r1), Some(r2)) = (
        rank_of(&factors[0], &factors[1]),
        rank_of(&factors[2], &factors[3]),
    ) else {
        // Misoriented factors: the dense reconstruction reports them by shape.
        return g.delta(shape, user_scale).map(Residual::Delta);
    };
    let k = r1 * r2;
    if k * (in_f + out_f) >= in_f * out_f {
        return g.delta(shape, user_scale).map(Residual::Delta);
    }
    let f32d = |t: &Option<Tensor>| -> Result<Tensor> {
        Ok(t.as_ref().expect("checked above").to_dtype(DType::F32)?)
    };
    let (a1, b1, a2, b2) = (
        f32d(&g.w1_a)?,
        f32d(&g.w1_b)?,
        f32d(&g.w2_a)?,
        f32d(&g.w2_b)?,
    );
    let u = a1
        .reshape((out_f, r1, 1))?
        .broadcast_mul(&a2.reshape((out_f, 1, r2))?)?
        .reshape((out_f, k))?;
    let v = b1
        .reshape((r1, 1, in_f))?
        .broadcast_mul(&b2.reshape((1, r2, in_f))?)?
        .reshape((k, in_f))?;
    Ok(Residual::LowRank {
        a: v.t()?.contiguous()?,
        b: (u.t()? * g.scale(user_scale) as f64)?.contiguous()?,
    })
}

/// Resolves an adapter's module names against the backbone's projections (every 2-D
/// `‹path›.weight` of the checkpoint, with its `[out, in]` shape).
struct Projections {
    shapes: BTreeMap<String, (usize, usize)>,
    kohya: BTreeMap<String, String>,
}

impl Projections {
    fn new(shapes: &BTreeMap<String, Vec<usize>>) -> Self {
        let shapes: BTreeMap<String, (usize, usize)> = shapes
            .iter()
            .filter_map(|(key, shape)| {
                let path = key.strip_suffix(".weight")?;
                (shape.len() == 2).then(|| (path.to_owned(), (shape[0], shape[1])))
            })
            .collect();
        let paths: Vec<String> = shapes.keys().cloned().collect();
        Self {
            kohya: kohya_table(&paths),
            shapes,
        }
    }

    /// A raw module name (prefixed, bare, kohya-flattened or third-party-flattened) → the dotted
    /// projection path, or `None`.
    fn resolve(&self, raw: &str) -> Option<String> {
        let bare = COMMON_LORA_PREFIXES
            .iter()
            .find_map(|p| raw.strip_prefix(p))
            .unwrap_or(raw);
        if self.shapes.contains_key(bare) {
            return Some(bare.to_owned());
        }
        if let Some(stem) = raw.strip_prefix(KOHYA_PREFIX) {
            if let Some(path) = resolve_kohya_stem(stem, &self.kohya) {
                return Some(path);
            }
        }
        resolve_lokr_path(raw, &self.kohya).map(str::to_owned)
    }
}

/// Which factor of a LoRA target a key carries.
enum Role {
    Down,
    Up,
    Alpha,
}

fn classify_lora_key(key: &str) -> Option<(&str, Role)> {
    for (suffix, role) in [
        (".lora_A.default.weight", Role::Down),
        (".lora_B.default.weight", Role::Up),
        (".lora_A.weight", Role::Down),
        (".lora_B.weight", Role::Up),
        (".lora_down.weight", Role::Down),
        (".lora_up.weight", Role::Up),
        (".alpha", Role::Alpha),
    ] {
        if let Some(stem) = key.strip_suffix(suffix) {
            return Some((stem, role));
        }
    }
    None
}

/// Accumulates one file's residuals, failing on anything it cannot place.
struct FileMerge<'a> {
    projections: &'a Projections,
    residuals: &'a mut BTreeMap<String, Vec<Residual>>,
    applied: usize,
    unresolved: Vec<String>,
}

impl FileMerge<'_> {
    /// Place one target's residual (built by `residual` from the projection's `[out, in]` shape).
    fn fold(
        &mut self,
        raw: &str,
        residual: impl FnOnce((usize, usize)) -> Result<Residual>,
    ) -> Result<()> {
        let Some(path) = self.projections.resolve(raw) else {
            self.unresolved.push(raw.to_owned());
            return Ok(());
        };
        let shape = self.projections.shapes[&path];
        let residual = residual(shape)?;
        let (dims, inner_ok) = residual.shape()?;
        if dims != shape || !inner_ok {
            return Err(Error::Msg(format!(
                "iris: adapter target {raw} reconstructs a {dims:?} delta for the {shape:?} \
                 projection {path} — the adapter was trained on a different architecture"
            )));
        }
        self.residuals.entry(path).or_default().push(residual);
        self.applied += 1;
        Ok(())
    }
}

fn merge_lora(merge: &mut FileMerge, af: &AdapterFile, scale: f32) -> Result<()> {
    #[derive(Default)]
    struct Triple {
        down: Option<Tensor>,
        up: Option<Tensor>,
        alpha: Option<f32>,
    }
    let mut triples: BTreeMap<&str, Triple> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match classify_lora_key(key) {
            Some((stem, Role::Down)) => triples.entry(stem).or_default().down = Some(t.clone()),
            Some((stem, Role::Up)) => triples.entry(stem).or_default().up = Some(t.clone()),
            Some((stem, Role::Alpha)) => {
                triples.entry(stem).or_default().alpha = Some(read_scalar(key, "alpha", t)?)
            }
            None => merge.unresolved.push(key.clone()),
        }
    }
    let blob = LoraAdapterMeta::from_file_metadata(&af.meta);
    for (stem, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            merge.unresolved.push(stem.to_owned());
            continue;
        };
        if down.dims().len() != 2 || up.dims().len() != 2 {
            return Err(Error::Msg(format!(
                "iris: adapter target {stem} is not a 2-D Linear LoRA (the Iris adapter surface \
                 is the backbone's projections)"
            )));
        }
        let path = merge.projections.resolve(stem).unwrap_or_default();
        let (blob_alpha, blob_rank) = blob.as_ref().map_or((None, None), |b| b.effective(&path));
        let rank = blob_rank.unwrap_or(down.dims()[0] as f32);
        let alpha = t.alpha.or(blob_alpha).unwrap_or(rank);
        merge.fold(stem, |_| lora_residual(&down, &up, alpha, rank, scale))?;
    }
    Ok(())
}

fn merge_stamped_lokr(merge: &mut FileMerge, af: &AdapterFile, scale: f32) -> Result<()> {
    let (rank, alpha) = parse_lokr_metadata(
        af.meta.get("rank").map(String::as_str),
        af.meta.get("alpha").map(String::as_str),
    )?;
    let mut groups: BTreeMap<&str, BTreeMap<&str, Tensor>> = BTreeMap::new();
    let mut alphas: Vec<(&str, &String, &Tensor)> = Vec::new();
    for (key, t) in &af.tensors {
        match LOKR_SUFFIXES
            .iter()
            .find_map(|suffix| key.strip_suffix(suffix).map(|stem| (stem, &suffix[1..])))
        {
            Some((stem, factor)) => {
                groups.entry(stem).or_default().insert(factor, t.clone());
            }
            None => match key.strip_suffix(".alpha") {
                Some(stem) => alphas.push((stem, key, t)),
                None => merge.unresolved.push(key.clone()),
            },
        }
    }
    // A per-target `‹path›.alpha` scalar (the Iris trainers' artifact layout, sc-25685/sc-25686)
    // restates the metadata alpha the delta is scaled by — the MLX loader scales by the metadata the
    // same way. It must name a LoKr target of this file and agree with the metadata; a disagreeing
    // one is refused rather than silently ignored or applied.
    for (stem, key, t) in alphas {
        if !groups.contains_key(stem) {
            merge.unresolved.push(key.clone());
            continue;
        }
        let a = read_scalar(key, "alpha", t)?;
        if a != alpha {
            return Err(Error::Msg(format!(
                "iris: LoKr target {stem} stores alpha {a} but the file metadata says {alpha}"
            )));
        }
    }
    for (stem, f) in groups {
        merge.fold(stem, |shape| {
            lokr_residual(
                f.get("lokr_w1"),
                f.get("lokr_w1_a"),
                f.get("lokr_w1_b"),
                f.get("lokr_w2"),
                f.get("lokr_w2_a"),
                f.get("lokr_w2_b"),
                alpha,
                rank,
                scale,
                shape,
            )
        })?;
    }
    Ok(())
}

/// Reconstruct every adapter in `specs` against the backbone whose checkpoint header carries
/// `shapes` (key → shape), for `task` on `base_model`. An empty `specs` merges nothing.
pub fn merge_adapters(
    shapes: &BTreeMap<String, Vec<usize>>,
    specs: &[AdapterSpec],
    task: IrisTask,
    base_model: &str,
) -> Result<MergedAdapters> {
    let projections = Projections::new(shapes);
    let mut residuals = BTreeMap::new();
    let mut reports = Vec::with_capacity(specs.len());
    for spec in specs {
        let label = spec.path.display();
        if spec.pass_scales.is_some() || spec.moe_expert.is_some() {
            return Err(Error::Unsupported(format!(
                "{base_model}: adapter {label} sets per-pass scales or an MoE expert; the Iris \
                 backbone is one single-pass denoiser, so only `scale` applies"
            )));
        }
        if has_diff_patch_keys(&spec.path)? {
            return Err(Error::Unsupported(format!(
                "{base_model}: adapter {label} is a ComfyUI diff-patch (`.diff`/`.diff_b`) file; the \
                 Iris adapter surface is LoRA / LoKr / LoHa low-rank factors"
            )));
        }
        let af = read_adapter(&spec.path)?;
        check_adapter_identity(&af.meta, task, base_model, &spec.path)?;
        let keys = || af.tensors.keys().map(String::as_str);
        let stamped_lokr = af.declares_lokr();
        if spec.kind == AdapterKind::Lora && stamped_lokr {
            return Err(Error::Msg(format!(
                "{base_model}: adapter {label} declared LoRA but its metadata says networkType=lokr"
            )));
        }
        let mut merge = FileMerge {
            projections: &projections,
            residuals: &mut residuals,
            applied: 0,
            unresolved: Vec::new(),
        };
        if stamped_lokr {
            merge_stamped_lokr(&mut merge, &af, spec.scale)?;
        } else if keys_contain_lokr(keys()) {
            for (raw, g) in parse_lokr_thirdparty(&af)? {
                merge.fold(&raw, |shape| match g.structured(shape, spec.scale)? {
                    Some(f) => Ok(Residual::Kron(f)),
                    None => g.delta(shape, spec.scale).map(Residual::Delta),
                })?;
            }
        } else if keys_contain_loha(keys()) {
            for (raw, g) in parse_loha_thirdparty(&af)? {
                merge.fold(&raw, |shape| loha_residual(&g, shape, spec.scale))?;
            }
        } else if spec.kind == AdapterKind::Lokr {
            return Err(Error::Msg(format!(
                "{base_model}: adapter {label} declared LoKr but carries no LoKr factors"
            )));
        } else {
            merge_lora(&mut merge, &af, spec.scale)?;
        }
        let (applied, unresolved) = (merge.applied, merge.unresolved);
        if !unresolved.is_empty() {
            return Err(Error::Msg(format!(
                "{base_model}: adapter {label} has {} target(s) that match no backbone projection \
                 (surfaced, not silently dropped): {unresolved:?}",
                unresolved.len()
            )));
        }
        if applied == 0 {
            return Err(Error::Msg(format!(
                "{base_model}: adapter {label} matched no backbone projection — expected PEFT/kohya \
                 LoRA or LoKr factors keyed by the IrisDiT module path"
            )));
        }
        reports.push(AdapterApplyReport {
            adapter_path: spec.path.clone(),
            applied,
            skipped: Vec::new(),
        });
    }
    Ok(MergedAdapters { residuals, reports })
}

/// Device bytes an adapter stack holds beside the backbone once installed, split by how the forward
/// holds each residual ([`Residual`]). Read from safetensors headers only ([`adapter_footprint`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdapterFootprint {
    /// Residuals applied as their factors: a LoRA's two f32 factors, a structured LoKr's f32
    /// Kronecker factors plus the kernel-dtype copies its forward caches, and a LoHa's low-rank
    /// factorisation in the compute dtype.
    pub forward_residual_bytes: u64,
    /// Residuals held as a full `[out, in]` delta in the compute dtype: a LoHa at or above its
    /// break-even rank, and a LoKr whose factors are not plain matrices.
    pub materialized_delta_bytes: u64,
}

impl AdapterFootprint {
    pub fn total(&self) -> u64 {
        self.forward_residual_bytes
            .saturating_add(self.materialized_delta_bytes)
    }
}

/// The shapes of one module's factors, by factor name.
type FactorShapes<'a> = BTreeMap<&'a str, &'a [usize]>;

fn elements(shape: &[usize]) -> u64 {
    shape.iter().map(|&d| d as u64).product()
}

/// One LoKr module's residency on an `[out, in]` projection: the structured Kronecker residual
/// when both factors are plain matrices that decompose the projection
/// (`LokrFactors::build`'s rule), else the dense delta the loader reconstructs.
fn lokr_footprint(
    factors: &FactorShapes,
    (out_f, in_f): (usize, usize),
    compute_width: u64,
    kernel_width: u64,
    footprint: &mut AdapterFootprint,
) {
    let matrix = |full: &str, a: &str, b: &str| -> Option<(u64, u64)> {
        match (factors.get(full), factors.get(a), factors.get(b)) {
            (Some(w), _, _) if w.len() == 2 => Some((w[0] as u64, w[1] as u64)),
            (None, Some(a), Some(b)) if a.len() == 2 && b.len() == 2 => {
                Some((a[0] as u64, b[1] as u64))
            }
            _ => None,
        }
    };
    let structured = matrix("lokr_w1", "lokr_w1_a", "lokr_w1_b")
        .zip(matrix("lokr_w2", "lokr_w2_a", "lokr_w2_b"))
        .filter(|((a, c), (b, d))| a * b == out_f as u64 && c * d == in_f as u64);
    match structured {
        Some(((a, c), (b, d))) => {
            let dims = candle_gen::gen_core::LokrKroneckerDims { a, b, c, d };
            footprint.forward_residual_bytes = footprint
                .forward_residual_bytes
                .saturating_add(dims.resident_bytes(kernel_width));
        }
        None => {
            footprint.materialized_delta_bytes = footprint
                .materialized_delta_bytes
                .saturating_add((out_f as u64 * in_f as u64).saturating_mul(compute_width));
        }
    }
}

/// One LoHa module's residency: its rank-`r₁r₂` factorisation below the break-even rank
/// ([`loha_residual`]'s rule), else the dense delta.
fn loha_footprint(
    factors: &FactorShapes,
    (out_f, in_f): (usize, usize),
    compute_width: u64,
    footprint: &mut AdapterFootprint,
) {
    let rank = |a: &str, b: &str| -> Option<usize> {
        let (a, b) = (factors.get(a)?, factors.get(b)?);
        (a.len() == 2 && b.len() == 2 && a[0] == out_f && b[1] == in_f && a[1] == b[0])
            .then_some(a[1])
    };
    let low_rank = rank("hada_w1_a", "hada_w1_b")
        .zip(rank("hada_w2_a", "hada_w2_b"))
        .map(|(r1, r2)| r1 * r2)
        .filter(|k| k * (in_f + out_f) < in_f * out_f);
    match low_rank {
        Some(k) => {
            footprint.forward_residual_bytes = footprint
                .forward_residual_bytes
                .saturating_add((k as u64 * (in_f + out_f) as u64).saturating_mul(compute_width));
        }
        None => {
            footprint.materialized_delta_bytes = footprint
                .materialized_delta_bytes
                .saturating_add((out_f as u64 * in_f as u64).saturating_mul(compute_width));
        }
    }
}

/// What [`merge_adapters`] would install for `specs`, priced from each file's safetensors **header**
/// (tensor names, shapes and `__metadata__`) — no tensor is read and no delta is reconstructed, so
/// a caller that only needs the stack's size (the memory contract, before every load) pays a few
/// kilobytes of I/O per file. `compute_width` is the backbone's compute dtype width and
/// `kernel_width` the width its GEMMs run at (they differ only on the Candle CPU backend under a
/// half-precision compute dtype).
///
/// Every refusal [`merge_adapters`] can make from a header is made here with the same condition:
/// per-pass scales / an MoE expert, a diff-patch file, a missing or wrong `family` / `irisTask`
/// stamp, a declared kind the file contradicts, a target that resolves to no backbone projection, a
/// LoRA whose factors do not fit its projection, and a file that adapts nothing. What needs tensor
/// values — a per-target LoKr `alpha` disagreeing with the metadata, a factor set whose
/// reconstruction fails — stays the loader's refusal.
pub fn adapter_footprint(
    shapes: &BTreeMap<String, Vec<usize>>,
    specs: &[AdapterSpec],
    task: IrisTask,
    base_model: &str,
    compute_width: u64,
    kernel_width: u64,
) -> Result<AdapterFootprint> {
    let projections = Projections::new(shapes);
    let mut footprint = AdapterFootprint::default();
    for spec in specs {
        let label = spec.path.display();
        if spec.pass_scales.is_some() || spec.moe_expert.is_some() {
            return Err(Error::Unsupported(format!(
                "{base_model}: adapter {label} sets per-pass scales or an MoE expert; the Iris \
                 backbone is one single-pass denoiser, so only `scale` applies"
            )));
        }
        let header = crate::nn::safetensors_shapes(&spec.path)?;
        if header
            .keys()
            .any(|key| key.ends_with(".diff") || key.ends_with(".diff_b"))
        {
            return Err(Error::Unsupported(format!(
                "{base_model}: adapter {label} is a ComfyUI diff-patch (`.diff`/`.diff_b`) file; the \
                 Iris adapter surface is LoRA / LoKr / LoHa low-rank factors"
            )));
        }
        let meta = gen_core_iris::downstream::weights_metadata(&spec.path)?.unwrap_or_default();
        check_adapter_identity(&meta, task, base_model, &spec.path)?;
        let keys = || header.keys().map(String::as_str);
        let stamped_lokr = candle_gen::gen_core::weightsmeta::is_lokr_network_type(
            meta.get("networkType").map(String::as_str),
        );
        if spec.kind == AdapterKind::Lora && stamped_lokr {
            return Err(Error::Msg(format!(
                "{base_model}: adapter {label} declared LoRA but its metadata says networkType=lokr"
            )));
        }
        let mut applied = 0usize;
        let mut unresolved: Vec<String> = Vec::new();
        // Group a factor file's keys by module, under the suffix set of its layout.
        let group = |suffixes: &[&str], strict: bool, unresolved: &mut Vec<String>| {
            let mut groups: BTreeMap<&str, FactorShapes> = BTreeMap::new();
            let mut alphas: Vec<&str> = Vec::new();
            for (key, shape) in &header {
                match candle_gen::gen_core::weightsmeta::split_factor_key(key, suffixes) {
                    Some((module, factor)) => {
                        groups
                            .entry(module)
                            .or_default()
                            .insert(factor, shape.as_slice());
                    }
                    None => match key.strip_suffix(".alpha") {
                        Some(module) => alphas.push(module),
                        None if strict => unresolved.push(key.clone()),
                        None => {}
                    },
                }
            }
            if strict {
                for module in alphas {
                    if !groups.contains_key(module) {
                        unresolved.push(format!("{module}.alpha"));
                    }
                }
            }
            groups
        };
        if stamped_lokr || keys_contain_lokr(keys()) {
            if stamped_lokr {
                parse_lokr_metadata(
                    meta.get("rank").map(String::as_str),
                    meta.get("alpha").map(String::as_str),
                )?;
            }
            let suffixes: &[&str] = if stamped_lokr {
                &LOKR_SUFFIXES
            } else {
                &candle_gen::gen_core::weightsmeta::LOKR_TP_SUFFIXES
            };
            for (module, factors) in group(suffixes, stamped_lokr, &mut unresolved) {
                match projections.resolve(module) {
                    Some(path) => {
                        lokr_footprint(
                            &factors,
                            projections.shapes[&path],
                            compute_width,
                            kernel_width,
                            &mut footprint,
                        );
                        applied += 1;
                    }
                    None => unresolved.push(module.to_owned()),
                }
            }
        } else if keys_contain_loha(keys()) {
            let suffixes = &candle_gen::gen_core::weightsmeta::LOHA_TP_SUFFIXES;
            for (module, factors) in group(suffixes, false, &mut unresolved) {
                match projections.resolve(module) {
                    Some(path) => {
                        loha_footprint(
                            &factors,
                            projections.shapes[&path],
                            compute_width,
                            &mut footprint,
                        );
                        applied += 1;
                    }
                    None => unresolved.push(module.to_owned()),
                }
            }
        } else if spec.kind == AdapterKind::Lokr {
            return Err(Error::Msg(format!(
                "{base_model}: adapter {label} declared LoKr but carries no LoKr factors"
            )));
        } else {
            type Pair<'a> = (Option<&'a [usize]>, Option<&'a [usize]>);
            let mut pairs: BTreeMap<&str, Pair> = BTreeMap::new();
            for (key, shape) in &header {
                match classify_lora_key(key) {
                    Some((stem, Role::Down)) => pairs.entry(stem).or_default().0 = Some(shape),
                    Some((stem, Role::Up)) => pairs.entry(stem).or_default().1 = Some(shape),
                    Some((stem, Role::Alpha)) => {
                        pairs.entry(stem).or_default();
                    }
                    None => unresolved.push(key.clone()),
                }
            }
            for (stem, pair) in pairs {
                let (Some(down), Some(up)) = pair else {
                    unresolved.push(stem.to_owned());
                    continue;
                };
                if down.len() != 2 || up.len() != 2 {
                    return Err(Error::Msg(format!(
                        "iris: adapter target {stem} is not a 2-D Linear LoRA (the Iris adapter \
                         surface is the backbone's projections)"
                    )));
                }
                let Some(path) = projections.resolve(stem) else {
                    unresolved.push(stem.to_owned());
                    continue;
                };
                let shape = projections.shapes[&path];
                if (up[0], down[1]) != shape || down[0] != up[1] {
                    return Err(Error::Msg(format!(
                        "iris: adapter target {stem} reconstructs a {:?} delta for the {shape:?} \
                         projection {path} — the adapter was trained on a different architecture",
                        (up[0], down[1])
                    )));
                }
                // Both factors are held in f32 whatever the compute dtype.
                footprint.forward_residual_bytes = footprint
                    .forward_residual_bytes
                    .saturating_add((elements(down) + elements(up)).saturating_mul(4));
                applied += 1;
            }
        }
        if !unresolved.is_empty() {
            return Err(Error::Msg(format!(
                "{base_model}: adapter {label} has {} target(s) that match no backbone projection \
                 (surfaced, not silently dropped): {unresolved:?}",
                unresolved.len()
            )));
        }
        if applied == 0 {
            return Err(Error::Msg(format!(
                "{base_model}: adapter {label} matched no backbone projection — expected PEFT/kohya \
                 LoRA or LoKr factors keyed by the IrisDiT module path"
            )));
        }
    }
    Ok(footprint)
}
