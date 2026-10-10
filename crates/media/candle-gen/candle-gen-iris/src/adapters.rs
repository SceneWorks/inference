//! LoRA / LoKr adapters on the Iris backbone (sc-25681), the Candle family convention: every
//! adapter file's weight delta is reconstructed in f32 and **folded into the dense base weight**
//! (`W += δ`) at the safetensors-key level, before the projection is cast to the compute dtype —
//! the merge the other candle families use (`candle_gen::train::merge`), so the merged forward is
//! reproduced exactly with no per-step residual op. A LoKr's delta is the real Kronecker product
//! `δ = scale · (alpha/rank) · kron(w1, w2)` (low-rank `_a`/`_b` factors expanded first); a LoRA's
//! `scale · (alpha/rank) · B·A`.
//!
//! Upstream Iris-3B ships no adapter code, so the surface is the repo's, the same one the MLX twin
//! installs: PEFT/diffusers LoRA (`transformer.` / `diffusion_model.` prefixes or bare, `lora_A/B`
//! or `lora_down/up`, per-target `.alpha` or the `lora_adapter_metadata` blob), kohya flattened
//! (`lora_unet_…`), PEFT-stamped LoKr (`networkType=lokr`), and untagged third-party LyCORIS
//! LoKr/LoHa — keyed by the upstream `IrisDiT` module path, which is also the checkpoint key stem.
//!
//! **Identity first** ([`check_adapter_identity`]): the file must name the Iris family and this
//! route's task, because the three task backbones share one architecture. **Strict:** a target
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
use candle_gen::train::lora::{
    parse_lokr_metadata, reconstruct_lokr_delta, reconstruct_lora_delta, LoraAdapterMeta,
};
use candle_gen::train::merge::{
    has_diff_patch_keys, parse_loha_thirdparty, parse_lokr_thirdparty, read_adapter, read_scalar,
    AdapterFile,
};
use candle_gen::{CandleError as Error, Result};

/// The f32 weight deltas of an adapter stack, keyed by checkpoint key (`‹path›.weight`), summed
/// over every file that targets the same projection, plus one report per file.
pub struct MergedAdapters {
    pub deltas: HashMap<String, Tensor>,
    pub reports: Vec<AdapterApplyReport>,
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

/// Accumulates one file's deltas, failing on anything it cannot place.
struct FileMerge<'a> {
    projections: &'a Projections,
    deltas: &'a mut HashMap<String, Tensor>,
    applied: usize,
    unresolved: Vec<String>,
}

impl FileMerge<'_> {
    fn fold(
        &mut self,
        raw: &str,
        delta: impl FnOnce((usize, usize)) -> Result<Tensor>,
    ) -> Result<()> {
        let Some(path) = self.projections.resolve(raw) else {
            self.unresolved.push(raw.to_owned());
            return Ok(());
        };
        let shape = self.projections.shapes[&path];
        let delta = delta(shape)?.to_dtype(DType::F32)?;
        if delta.dims() != [shape.0, shape.1] {
            return Err(Error::Msg(format!(
                "iris: adapter target {raw} reconstructs a {:?} delta for the {shape:?} projection \
                 {path} — the adapter was trained on a different architecture",
                delta.dims()
            )));
        }
        let key = format!("{path}.weight");
        let summed = match self.deltas.remove(&key) {
            Some(prev) => (prev + delta)?,
            None => delta,
        };
        self.deltas.insert(key, summed);
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
        merge.fold(stem, |_| {
            reconstruct_lora_delta(&down, &up, alpha, rank, scale)
        })?;
    }
    Ok(())
}

fn merge_stamped_lokr(merge: &mut FileMerge, af: &AdapterFile, scale: f32) -> Result<()> {
    let (rank, alpha) = parse_lokr_metadata(
        af.meta.get("rank").map(String::as_str),
        af.meta.get("alpha").map(String::as_str),
    )?;
    let mut groups: BTreeMap<&str, BTreeMap<&str, Tensor>> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match LOKR_SUFFIXES
            .iter()
            .find_map(|suffix| key.strip_suffix(suffix).map(|stem| (stem, &suffix[1..])))
        {
            Some((stem, factor)) => {
                groups.entry(stem).or_default().insert(factor, t.clone());
            }
            None => merge.unresolved.push(key.clone()),
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
    let mut deltas = HashMap::new();
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
            deltas: &mut deltas,
            applied: 0,
            unresolved: Vec::new(),
        };
        if stamped_lokr {
            merge_stamped_lokr(&mut merge, &af, spec.scale)?;
        } else if keys_contain_lokr(keys()) {
            for (raw, g) in parse_lokr_thirdparty(&af)? {
                merge.fold(&raw, |shape| g.delta(shape, spec.scale))?;
            }
        } else if keys_contain_loha(keys()) {
            for (raw, g) in parse_loha_thirdparty(&af)? {
                merge.fold(&raw, |shape| g.delta(shape, spec.scale))?;
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
    Ok(MergedAdapters { deltas, reports })
}
