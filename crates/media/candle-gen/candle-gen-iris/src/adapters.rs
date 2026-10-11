//! LoRA / LoKr adapters on the Iris backbone (sc-25681), installed as **forward-time residuals**
//! over the frozen base projection (`y = base(x) + r(x)`, [`Residual`]) — the MLX twin's
//! `AdaptableLinear` install, and (sc-25686) exactly the forward the Candle trainer optimizes, so a
//! trained adapter renders here what its training previews rendered. Never merged into the base
//! weight: under the release's bf16 compute a merge rounds every delta below the base's bf16 ulp
//! away. A LoRA is held as its f32 factors (`Aᵀ`, `Bᵀ · scale · alpha/rank` — [`lora_residual`]); a
//! LoKr / LoHa as its reconstructed delta, the real Kronecker product
//! `δ = scale · (alpha/rank) · kron(w1, w2)` (low-rank `_a`/`_b` factors expanded first).
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
use candle_gen::gen_core::iris::{check_adapter_identity, IrisTask};
use candle_gen::gen_core::weightsmeta::{
    keys_contain_loha, keys_contain_lokr, kohya_table, resolve_kohya_stem, resolve_lokr_path,
    COMMON_LORA_PREFIXES, KOHYA_PREFIX, LOKR_SUFFIXES,
};
use candle_gen::gen_core::{AdapterApplyReport, AdapterKind, AdapterSpec};
use candle_gen::train::lora::{parse_lokr_metadata, reconstruct_lokr_delta, LoraAdapterMeta};
use candle_gen::train::merge::{
    has_diff_patch_keys, parse_loha_thirdparty, parse_lokr_thirdparty, read_adapter, read_scalar,
    AdapterFile,
};
use candle_gen::{CandleError as Error, Result};

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
        let dims = match &residual {
            Residual::Lora { a, b } => vec![b.dim(1)?, a.dim(0)?],
            Residual::Delta(d) => d.dims().to_vec(),
        };
        let inner_ok = match &residual {
            Residual::Lora { a, b } => a.dim(1)? == b.dim(0)?,
            Residual::Delta(_) => true,
        };
        if dims != [shape.0, shape.1] || !inner_ok {
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
            reconstruct_lokr_delta(
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
            .map(Residual::Delta)
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
                merge.fold(&raw, |shape| {
                    g.delta(shape, spec.scale).map(Residual::Delta)
                })?;
            }
        } else if keys_contain_loha(keys()) {
            for (raw, g) in parse_loha_thirdparty(&af)? {
                merge.fold(&raw, |shape| {
                    g.delta(shape, spec.scale).map(Residual::Delta)
                })?;
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
