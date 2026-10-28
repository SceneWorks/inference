//! **BFL / ComfyUI** fused → split LoRA and LyCORIS: per-target row slices of a fused source
//! linear.
//!
//! Split out of `loader.rs` (sc-27427); every item is re-exported there, so
//! `mlx_gen::adapters::loader::…` paths are unchanged.

use super::*;

// ---- BFL / ComfyUI fused→split LoRA (sc-2743) ----------------------------------------------------

/// A row-slice over a raw LoRA factor (applied BEFORE the `[r,in]`/`[out,r]`→residual transpose),
/// porting the fork's `LoraTransforms` byte-for-byte (sc-2743). The BFL/ComfyUI checkpoints store a
/// block's q/k/v (and, for FLUX.1, the qkv+mlp) concatenated along dim-0 of a single *fused* linear;
/// each diffusers split target slices out its own rows. Indices/divisibility match `LoraTransforms`
/// exactly (verified against the fork venv).
#[derive(Clone, Debug)]
pub enum LoraRowSlice {
    /// Chunk `index` of `n` equal dim-0 chunks (`chunk = shape[0] / n`), ALWAYS sliced — the fork's
    /// `_split_qkv_up` (the up factor `[n·out, r]` → `[out, r]`). `n=3` for qkv.
    Chunk { n: i32, index: i32 },
    /// Chunk `index` of `n` equal dim-0 chunks IFF `shape[0] % n == 0`, else the whole tensor — the
    /// fork's `_split_qkv_down`/`_split_qkv_mlp_down` (the down factor is *shared* across q/k/v when
    /// the rank isn't divisible by `n`, which is the usual fused-qkv LoRA, and sliced when it is).
    ChunkIfDivisible { n: i32, index: i32 },
    /// The dim-0 slice `[Σdims[..index] .. Σdims[..=index]]` — the fork's `_split_qkv_mlp_up` with
    /// config-derived `dims` (FLUX.1 `linear1` = `[q,k,v,mlp]`, e.g. `[3072,3072,3072,12288]`). FLUX.2
    /// keeps qkv+mlp fused (`to_qkv_mlp_proj`) so it never uses this; FLUX.1 (sc-2657) will.
    Dims { dims: Vec<i32>, index: i32 },
}

impl LoraRowSlice {
    pub(super) fn apply(&self, t: &Array) -> Result<Array> {
        let rows = t.shape()[0];
        // `n`/`index`/`dims` come from a static `bfl_targets()` table built in code, not the file —
        // but a miswritten entry (n<=0, index out of [0,n), or an out-of-range `dims` index) would
        // divide-by-zero / index out of bounds and panic. Reject it with a typed error (sc-5283).
        let (start, end) = match self {
            LoraRowSlice::Chunk { n, index } => {
                if *n <= 0 || *index < 0 || *index >= *n {
                    return Err(Error::Msg(format!(
                        "LoraRowSlice::Chunk: invalid chunk spec (n={n}, index={index})"
                    )));
                }
                // sc-9099: `Chunk` ALWAYS slices (vs `ChunkIfDivisible`'s fall-through), so a non-divisible
                // `rows` silently mis-slices the fused qkv delta (truncating division drops the tail).
                // The fork's `_split_qkv_up` is only ever emitted for an exactly-divisible fused linear,
                // so a non-divisible shape means a mis-converted/mismatched adapter — reject it.
                if rows % n != 0 {
                    return Err(Error::Msg(format!(
                        "LoraRowSlice::Chunk: rows ({rows}) not divisible by n ({n}); \
                         the fused factor is mis-shaped for this split"
                    )));
                }
                let chunk = rows / n;
                (index * chunk, (index + 1) * chunk)
            }
            LoraRowSlice::ChunkIfDivisible { n, index } => {
                if *n <= 0 || *index < 0 || *index >= *n {
                    return Err(Error::Msg(format!(
                        "LoraRowSlice::ChunkIfDivisible: invalid chunk spec (n={n}, index={index})"
                    )));
                }
                if rows % n != 0 {
                    return Ok(t.clone());
                }
                let chunk = rows / n;
                (index * chunk, (index + 1) * chunk)
            }
            LoraRowSlice::Dims { dims, index } => {
                let i = *index as usize;
                if *index < 0 || i >= dims.len() {
                    return Err(Error::Msg(format!(
                        "LoraRowSlice::Dims: index {index} out of range for {} dims",
                        dims.len()
                    )));
                }
                let start: i32 = dims[..i].iter().sum();
                (start, start + dims[i])
            }
        };
        // `t[start:end, :]` — byte-identical to the fork's slicing.
        Ok(t.try_index((start..end, ..))?)
    }
}

/// One BFL/ComfyUI adapter target: a set of source key spellings (across the `lora_unet_` /
/// `diffusion_model.` / `base_model.model.` prefix conventions) mapping to a diffusers module
/// `target_path`, with an optional [`LoraRowSlice`] on the up/down factor. A *fused* source (BFL
/// `…img_attn.qkv`) is named by SEVERAL `BflTarget`s — one per split destination (`to_q`/`to_k`/`to_v`)
/// — that share its key spellings but slice different rows; the loader fans the one source tensor into
/// all of them. A plain rename (BFL `img_in` → `x_embedder`) is a `BflTarget` with no slice. Mirrors a
/// fork `LoRATarget` restricted to its BFL patterns + up/down transforms.
#[derive(Clone, Debug)]
pub struct BflTarget {
    /// Diffusers module path that [`AdaptableHost::adaptable_mut`] resolves (concrete, no `{block}`).
    pub target_path: String,
    /// Source up-factor (`lora_up`/`lora_B`) key spellings.
    pub up_keys: Vec<String>,
    /// Source down-factor (`lora_down`/`lora_A`) key spellings.
    pub down_keys: Vec<String>,
    /// Source `alpha` key spellings (no transform, no transpose).
    pub alpha_keys: Vec<String>,
    /// Row-slice applied to the up factor (the qkv split). `None` for a plain rename.
    pub up_slice: Option<LoraRowSlice>,
    /// Row-slice applied to the down factor (shared-or-split). `None` for a plain rename.
    pub down_slice: Option<LoraRowSlice>,
}

/// One contribution of a source key to a target: which target/role it feeds and how to slice it.
pub(super) struct BflSlot {
    pub(super) target: String,
    pub(super) role: KohyaRole,
    pub(super) slice: Option<LoraRowSlice>,
}

/// Invert `targets` into `source_key → [contribution, …]`. One fused source key (e.g. a qkv
/// `lora_up`) contributes to multiple targets (q/k/v) with different slices, so the value is a list.
pub(super) fn bfl_index(targets: &[BflTarget]) -> BTreeMap<String, Vec<BflSlot>> {
    let mut index: BTreeMap<String, Vec<BflSlot>> = BTreeMap::new();
    let mut push = |key: &str, target: &str, role: KohyaRole, slice: Option<LoraRowSlice>| {
        index.entry(key.to_string()).or_default().push(BflSlot {
            target: target.to_string(),
            role,
            slice,
        });
    };
    for t in targets {
        for k in &t.up_keys {
            push(k, &t.target_path, KohyaRole::Up, t.up_slice.clone());
        }
        for k in &t.down_keys {
            push(k, &t.target_path, KohyaRole::Down, t.down_slice.clone());
        }
        for k in &t.alpha_keys {
            push(k, &t.target_path, KohyaRole::Alpha, None);
        }
    }
    index
}

/// `true` if any key in `w` is a known BFL source key for `targets` — i.e. the file uses the BFL /
/// ComfyUI naming (`double_blocks`/`single_blocks`/`img_in`/… across the three prefix conventions),
/// which the diffusers/peft/standard-kohya paths cannot resolve. Precise: a standard diffusers/peft
/// or standard-kohya file shares none of these spellings, so it is never misrouted here.
pub fn is_bfl(w: &Weights, targets: &[BflTarget]) -> bool {
    if targets.is_empty() {
        return false;
    }
    let index = bfl_index(targets);
    w.keys().any(|k| index.contains_key(k))
}

/// Recognized LoRA factor suffixes — a key ending in one of these is adapter-shaped (vs. a base
/// weight or some bundled extra). Used to surface BFL-named keys that resolve to no target.
pub(super) const LORA_FACTOR_SUFFIXES: [&str; 5] = [
    ".lora_up.weight",
    ".lora_down.weight",
    ".lora_A.weight",
    ".lora_B.weight",
    ".alpha",
];

/// Install a BFL / ComfyUI fused→split LoRA onto `host` (sc-2743). Each file key is matched against
/// the inverted [`BflTarget`] index; a matched *fused* source is row-sliced per destination and fanned
/// into the diffusers split targets (`…img_attn.qkv` → `attn.to_q/to_k/to_v`), a plain rename is copied
/// through. Resolved factors feed the same `install_lora_groups` path as PEFT/kohya (transpose +
/// `alpha/rank` fold), so a BFL file yields the byte-identical adapter to the equivalent diffusers
/// split-target LoRA.
///
/// An adapter-shaped key that matches NO target — an off-surface BFL key (e.g. a block out of range)
/// — is surfaced in `unmatched_paths` (loud, never silently dropped). A bundled text-encoder key
/// (`lora_te_…`/`text_encoder.…`) is not a denoiser target and is ignored, matching the PEFT/kohya
/// loaders' treatment of out-of-namespace keys.
pub fn apply_lora_bfl(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    targets: &[BflTarget],
) -> Result<ApplyReport> {
    let index = bfl_index(targets);
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    let mut unresolved: BTreeSet<String> = BTreeSet::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        let Some(slots) = index.get(&key) else {
            // Not a BFL source key: surface it if it's an adapter-shaped denoiser key (off-surface),
            // ignore a bundled text-encoder adapter (`lora_te_…`/`…text_encoder.…`).
            let adapter_shaped = LORA_FACTOR_SUFFIXES.iter().any(|s| key.ends_with(s));
            let foreign_te = key.starts_with("lora_te") || key.contains("text_encoder");
            if adapter_shaped && !foreign_te {
                unresolved.insert(key);
            }
            continue;
        };
        let v = w.require(&key)?;
        for slot in slots {
            let parts = groups.entry(slot.target.clone()).or_default();
            match slot.role {
                KohyaRole::Down => {
                    parts.a = Some(match &slot.slice {
                        Some(s) => s.apply(v)?,
                        None => v.clone(),
                    });
                }
                KohyaRole::Up => {
                    parts.b = Some(match &slot.slice {
                        Some(s) => s.apply(v)?,
                        None => v.clone(),
                    });
                }
                KohyaRole::Alpha => {
                    // Detect a genuine conflict rather than nondeterministic last-wins
                    // (sc-11129): a file with two `.alpha` spellings mapping to one target resolved by
                    // HashMap visit order, installing at a nondeterministic scale across runs while
                    // reporting success. Mirror `apply_lora_peft`'s hard "alpha conflict" error. A
                    // repeated identical value (the same key visited twice) is not a conflict.
                    if let Some(new) = scalar_alpha(v)? {
                        match parts.alpha {
                            Some(existing) if existing != new => {
                                return Err(Error::Msg(format!(
                                    "LoRA alpha conflict for `{}`: {existing} vs {new} \
                                     (duplicate alpha keys for one BFL target)",
                                    slot.target
                                )));
                            }
                            _ => parts.alpha = Some(new),
                        }
                    }
                }
            }
        }
    }

    // kohya / BFL files carry a per-target `.alpha` tensor, not the `lora_adapter_metadata` blob — no
    // blob to honor here (sc-5513).
    let mut report = install_lora_groups(host, groups, scale, None)?;
    report.unmatched_paths.extend(unresolved);
    Ok(report)
}

// ---- BFL / ComfyUI fused→split LyCORIS LoKr / LoHa (sc-8345) --------------------------------------

/// One BFL/ComfyUI LyCORIS destination: a split diffusers `target_path` plus the row-slice that carves
/// this destination's rows out of the *fused* reconstructed `[out,in]` delta (`None` = a plain rename,
/// the whole delta). Unlike [`BflTarget`] — which slices the raw LoRA up/down *factors* before the
/// residual transpose — a LyCORIS adapter reconstructs the full Kronecker/Hadamard delta first and
/// then slices its rows, so only the out-dim (`up`) slice is relevant here.
#[derive(Clone, Debug)]
pub(super) struct BflLycorisTarget {
    pub(super) target_path: String,
    pub(super) out_slice: Option<LoraRowSlice>,
}

/// Build a `prefixed-module-path → [split target, …]` map from a host's [`BflTarget`] list, keyed by
/// the module path *as a LyCORIS file spells it* — every `up_key` minus its `.lora_up.weight` /
/// `.lora_B.weight` factor suffix, i.e. the `lora_unet_<flat>` / `diffusion_model.<dotted>` /
/// `base_model.model.<dotted>` BFL spellings. A fused qkv source maps to its three split targets (each
/// carrying its own out-dim slice); a rename maps to one target with no slice. The LyCORIS analog of
/// [`bfl_index`] (which keys by full factor key + role because LoRA slices factors, not the delta).
pub(super) fn bfl_lycoris_module_map(
    targets: &[BflTarget],
) -> BTreeMap<String, Vec<BflLycorisTarget>> {
    let mut map: BTreeMap<String, Vec<BflLycorisTarget>> = BTreeMap::new();
    for t in targets {
        for up in &t.up_keys {
            let Some(module) = up
                .strip_suffix(".lora_up.weight")
                .or_else(|| up.strip_suffix(".lora_B.weight"))
            else {
                continue;
            };
            let entry = map.entry(module.to_string()).or_default();
            // The same module key appears under both the `lora_up` and `lora_B` spellings; keep one
            // entry per destination.
            if entry.iter().all(|e| e.target_path != t.target_path) {
                entry.push(BflLycorisTarget {
                    target_path: t.target_path.clone(),
                    out_slice: t.up_slice.clone(),
                });
            }
        }
    }
    map
}

/// The LyCORIS module path a key belongs to — `key` minus a trailing `.lokr_*` / `.hada_*` / `.alpha`
/// factor suffix — or `None` if `key` is not a LyCORIS factor key.
pub(super) fn lycoris_module_of(key: &str) -> Option<&str> {
    if let Some(module) = key.strip_suffix(".alpha") {
        return Some(module);
    }
    LOKR_TP_SUFFIXES
        .iter()
        .chain(LOHA_TP_SUFFIXES.iter())
        .find_map(|suffix| key.strip_suffix(suffix))
}

/// `true` if any LyCORIS factor key in `w` names a module in the BFL map — i.e. the file uses the
/// BFL/ComfyUI fused naming a host's `bfl_targets()` covers. A diffusers/bare/standard-kohya LyCORIS
/// file (modules like `transformer.…` or a `lora_unet_<diffusers-flat>` that resolves through the
/// kohya table) shares none of these spellings, so it is never misrouted here and stays on the
/// existing third-party/peft path. Empty map (a host with no BFL surface — every engine but FLUX.1/
/// FLUX.2) ⇒ always `false`.
pub(super) fn is_bfl_lycoris(w: &Weights, map: &BTreeMap<String, Vec<BflLycorisTarget>>) -> bool {
    if map.is_empty() {
        return false;
    }
    w.keys()
        .any(|k| lycoris_module_of(k).is_some_and(|module| map.contains_key(module)))
}

/// Install grouped LyCORIS deltas onto a host's BFL fused→split targets (sc-8345). For each source
/// module the `reconstruct` closure rebuilds the FULL fused `[out,in]` delta (the host-fused qkv shape,
/// with the format's `alpha/rank` scale already baked in); each destination then row-slices its share
/// out of that delta and stacks it as an [`Adapter::Lokr`] residual at the user `scale`. The fused
/// `out` is the SUM of the destinations' out dims (3·inner for a qkv split, Σdims for FLUX.1's qkv+mlp,
/// the target's own out for a rename), so it is derived from the resolved targets rather than parsed
/// from the slice. A module absent from `map` is surfaced in `unmatched_paths`, never silently dropped.
pub(super) fn install_bfl_lycoris<I, F>(
    host: &mut impl AdaptableHost,
    map: &BTreeMap<String, Vec<BflLycorisTarget>>,
    groups: I,
    scale: f32,
) -> Result<ApplyReport>
where
    I: IntoIterator<Item = (String, F)>,
    F: FnOnce(&[i32]) -> Result<Array>,
{
    let mut report = ApplyReport::default();

    // One resolved BFL module: its split destinations, the fused `[Σout, in]` shape to reconstruct at,
    // and the (deferred) reconstruction closure — held so pass 1 can budget-check the total before any
    // pass-2 allocation.
    struct BflPlan<F> {
        targets: Vec<BflLycorisTarget>,
        fused_shape: [i32; 2],
        reconstruct: F,
    }

    // Pass 1 — resolve each module's fused shape and sum the bytes the reconstruction will materialize
    // (sc-11129). This BFL fused→split path has NO deferred/allocation-free form: each
    // destination row-slices the fully reconstructed `[Σout,in]` bf16 delta, so every install
    // materializes. Its plain-path twin (`install_lycoris_groups`) already applies the sc-10678 memory
    // guard; this sibling did not, so a full-coverage BFL LoKr on FLUX could hold multiple GB of
    // resident deltas with no up-front refusal and OOM the worker mid-load via an uncatchable SIGKILL.
    let mut plans: Vec<BflPlan<F>> = Vec::new();
    let mut projected_materialize: usize = 0;
    let dtype = host.lycoris_dtype();
    for (module, reconstruct) in groups {
        let Some(targets) = map.get(&module) else {
            report.unmatched_paths.push(module);
            continue;
        };
        // Fused reconstruction shape: rows = Σ destination out dims, cols = the shared in dim. Every
        // fused destination MUST share one in_dim — the reconstruction uses the first destination's
        // input dim for all rows, so a miswritten `bfl_targets()` entry fusing mismatched in-dims would
        // otherwise surface as an opaque kron/reshape error far from the cause (sc-11129).
        let mut fused_out = 0i32;
        let mut in_dim: Option<i32> = None;
        let mut resolvable = true;
        for tgt in targets {
            let parts: Vec<&str> = tgt.target_path.split('.').collect();
            // SC-18319 — shape-only resolution, so the PROBE half. See `install_lycoris_groups`.
            match host.adaptable_facts(&parts).map(|facts| facts.base_shape) {
                Some(shape) if shape.len() == 2 => {
                    fused_out += shape[0];
                    match in_dim {
                        None => in_dim = Some(shape[1]),
                        Some(existing) if existing != shape[1] => {
                            return Err(Error::Msg(format!(
                                "BFL LyCORIS module '{module}': fused destination '{}' has in_dim {} \
                                 but a prior destination in the same fused group had in_dim {existing} \
                                 — a fused qkv/mlp split must share one input dim",
                                tgt.target_path,
                                shape[1],
                            )));
                        }
                        Some(_) => {}
                    }
                }
                _ => {
                    resolvable = false;
                    break;
                }
            }
        }
        let (Some(in_dim), true) = (in_dim, resolvable) else {
            // A destination that didn't resolve (or a non-2-D linear) — surface the module rather than
            // install a partial, mis-shaped delta.
            report.unmatched_paths.push(module);
            continue;
        };
        projected_materialize += projected_delta_bytes(&[fused_out, in_dim], dtype);
        plans.push(BflPlan {
            targets: targets.clone(),
            fused_shape: [fused_out, in_dim],
            reconstruct,
        });
    }

    // sc-10678 / sc-11129 — pre-flight memory guard, the same policy as `install_lycoris_groups`.
    if projected_materialize > 0 {
        let active = get_active_memory();
        let limit = get_memory_limit();
        if materialization_exceeds_budget(active, projected_materialize, limit) {
            return Err(Error::Msg(format!(
                "This BFL/ComfyUI LyCORIS adapter must reconstruct a fused dense `[out,in]` delta for \
                 every target (the fused→split path has no allocation-free form) — ~{:.1} GB on top of \
                 the ~{:.1} GB already resident, over the ~{:.1} GB budget. Use a plain LoRA instead \
                 (near-zero extra memory on any tier). On a quantized tier, the bf16 tier merges this \
                 adapter into the dense weight in place — note bf16 renders differently.",
                projected_materialize as f64 / GIB,
                active as f64 / GIB,
                (limit as f64 * MATERIALIZE_BUDGET_HEADROOM) / GIB,
            )));
        }
    }

    // Pass 2 — reconstruct each fused delta and row-slice every destination's share.
    for plan in plans {
        let BflPlan {
            targets,
            fused_shape,
            reconstruct,
        } = plan;
        let delta = reconstruct(&fused_shape)?;
        for tgt in &targets {
            let parts: Vec<&str> = tgt.target_path.split('.').collect();
            let Some(lin) = host.adaptable_mut(&parts) else {
                report.unmatched_paths.push(tgt.target_path.clone());
                continue;
            };
            let piece = match &tgt.out_slice {
                Some(slice) => slice.apply(&delta)?,
                None => delta.clone(),
            };
            lin.push(Adapter::Lokr {
                delta: piece,
                scale,
            });
            report.applied += 1;
        }
    }
    Ok(report)
}

/// Install a metadata-stamped (peft) LoKr file in BFL/ComfyUI fused naming (sc-8345). Same Kronecker
/// reconstruction + `alpha/rank` fold as [`apply_lokr`], but the fused qkv source is rebuilt at the
/// host-fused shape and row-sliced into the split targets via [`install_bfl_lycoris`].
pub(super) fn apply_lokr_bfl(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    map: &BTreeMap<String, Vec<BflLycorisTarget>>,
) -> Result<ApplyReport> {
    let file = parse_lokr(w)?;
    let (alpha, rank) = (file.alpha, file.rank);
    let dtype = host.lycoris_dtype();
    let groups = file.groups.into_iter().map(|(module, factors)| {
        (module, move |base: &[i32]| {
            reconstruct_lokr_delta(
                alpha,
                rank,
                base,
                factors.get("lokr_w1"),
                factors.get("lokr_w1_a"),
                factors.get("lokr_w1_b"),
                factors.get("lokr_w2"),
                factors.get("lokr_w2_a"),
                factors.get("lokr_w2_b"),
                dtype,
            )
        })
    });
    install_bfl_lycoris(host, map, groups, scale)
}

/// Install a third-party LyCORIS **LoKr** file in BFL/ComfyUI fused naming (sc-8345). Per-module
/// lycoris scale + tucker-capable Kronecker reconstruction (same as [`apply_lokr_thirdparty`]), fused→
/// split via [`install_bfl_lycoris`].
pub(super) fn apply_lokr_thirdparty_bfl(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    map: &BTreeMap<String, Vec<BflLycorisTarget>>,
) -> Result<ApplyReport> {
    let dtype = host.lycoris_dtype();
    let groups = parse_lokr_thirdparty(w)?
        .into_iter()
        .map(|(module, g)| (module, move |base: &[i32]| g.delta(base, dtype)));
    install_bfl_lycoris(host, map, groups, scale)
}

/// Install a third-party LyCORIS **LoHa** file in BFL/ComfyUI fused naming (sc-8345). Hadamard
/// reconstruction (same as [`apply_loha_thirdparty`]), fused→split via [`install_bfl_lycoris`].
pub(super) fn apply_loha_thirdparty_bfl(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    map: &BTreeMap<String, Vec<BflLycorisTarget>>,
) -> Result<ApplyReport> {
    let dtype = host.lycoris_dtype();
    let groups = parse_loha_thirdparty(w)?
        .into_iter()
        .map(|(module, g)| (module, move |base: &[i32]| g.delta(base, dtype)));
    install_bfl_lycoris(host, map, groups, scale)
}
