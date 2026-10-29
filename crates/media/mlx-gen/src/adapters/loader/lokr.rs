//! Native (PEFT) and third-party LyCORIS **LoKr** files: detection, parsing and install.
//!
//! Split out of `loader.rs` (sc-27427); every item is re-exported there, so
//! `mlx_gen::adapters::loader::…` paths are unchanged.

use super::*;

/// PEFT LoKr per-module factor suffixes (gen-core's table) — each factor is full (`lokr_w1`/
/// `lokr_w2`) or low-rank (`_a`/`_b`). Exact-suffix matched.
pub(super) use gen_core::weightsmeta::LOKR_SUFFIXES;

/// `true` if the file's `networkType` metadata marks it a LoKr adapter.
pub fn is_lokr(w: &Weights) -> bool {
    wmeta::is_lokr_network_type(w.metadata("networkType"))
}

/// A parsed LoKr file: the global `(alpha, rank)` from metadata plus every module's Kronecker
/// factors grouped by path. The factor map is keyed by the bare factor name (`lokr_w1`,
/// `lokr_w1_a`, `lokr_w1_b`, `lokr_w2`, `lokr_w2_a`, `lokr_w2_b`); a module is full or low-rank.
///
/// This is the format-parsing half of a LoKr install, factored out of [`apply_lokr`] so the video
/// providers (LTX/Wan) — which install onto their crate-local `Linear`s as a forward-time residual
/// or an in-place weight merge, rather than the core [`AdaptableHost`] — reuse the exact same factor
/// grouping + metadata read and differ only in the install step. Each provider then maps the bare
/// module `path` through its own key→module table and calls [`reconstruct_lokr_delta`].
#[derive(Debug)]
pub struct LokrFile {
    pub alpha: f32,
    pub rank: f32,
    /// `module path → { factor name → tensor }`.
    pub groups: BTreeMap<String, BTreeMap<String, Array>>,
}

impl LokrFile {
    /// Reconstruct one module's `[out,in]` delta at `out_dtype` from its grouped factors, baking in
    /// `alpha/rank` (the user `strength` is applied separately). `base_shape` is the target linear's
    /// logical weight shape. Returns the [`reconstruct_lokr_delta`] result.
    pub fn delta(
        &self,
        factors: &BTreeMap<String, Array>,
        base_shape: &[i32],
        out_dtype: Dtype,
    ) -> Result<Array> {
        reconstruct_lokr_delta(
            self.alpha,
            self.rank,
            base_shape,
            factors.get("lokr_w1"),
            factors.get("lokr_w1_a"),
            factors.get("lokr_w1_b"),
            factors.get("lokr_w2"),
            factors.get("lokr_w2_a"),
            factors.get("lokr_w2_b"),
            out_dtype,
        )
    }

    /// The **structured** (deferred, allocation-free) counterpart to [`delta`](Self::delta) for a
    /// packed base (sc-10578, extending sc-10050 from the video providers to every family on the
    /// shared install path): build the small `[a,c]`/`[b,d]` Kronecker factors so the residual applies
    /// via the vec-trick without ever materializing the `[out,in]` delta.
    ///
    /// **Scale differs from [`delta`](Self::delta) by design.** `delta` bakes only `alpha/rank` and
    /// leaves the user `strength` to [`Adapter::Lokr`]'s `scale` field; the structured residual carries
    /// no separate scale (it is baked into `factors.w2`), so the FULL `alpha/rank · strength` is
    /// applied here. Mismatching these two is a silent mis-scale, not a crash.
    ///
    /// Returns `Ok(None)` when the module has no 2-D matrix form (a conv-shaped factor); the peft
    /// LoKr format never carries a tucker `lokr_t2` — that is third-party-only, hence the `None`
    /// passed through. The caller then materializes (dense base) or errors (packed base).
    pub fn factors(
        &self,
        factors: &BTreeMap<String, Array>,
        strength: f32,
        base_shape: &[i32],
        out_dtype: Dtype,
    ) -> Result<Option<LokrFactors>> {
        build_lokr_factors(
            (self.alpha / self.rank) * strength,
            base_shape,
            factors.get("lokr_w1"),
            factors.get("lokr_w1_a"),
            factors.get("lokr_w1_b"),
            factors.get("lokr_w2"),
            None, // peft LoKr never carries a tucker `lokr_t2` (lycoris conv-only).
            factors.get("lokr_w2_a"),
            factors.get("lokr_w2_b"),
            out_dtype,
        )
    }
}

/// Parse a LoKr `.safetensors` into [`LokrFile`]: read `rank`/`alpha` from metadata (alpha defaults
/// to rank, i.e. scale 1.0, matching PEFT) and group every `‹path›.lokr_*` tensor by module path.
/// Shared by [`apply_lokr`] (core `AdaptableHost` install) and the video providers' crate-local
/// residual/merge installers.
pub fn parse_lokr(w: &Weights) -> Result<LokrFile> {
    // rank/alpha (alpha defaults to rank ⇒ scale 1.0, matching PEFT) — parsed in gen-core.
    let (rank, alpha) = wmeta::parse_rank_alpha(w.metadata("rank"), w.metadata("alpha"));

    let keys: Vec<String> = w.keys().map(str::to_string).collect();
    let mut groups: BTreeMap<String, BTreeMap<String, Array>> = BTreeMap::new();
    for key in &keys {
        if let Some((path, factor)) = wmeta::split_factor_key(key, &LOKR_SUFFIXES) {
            groups
                .entry(path.to_string())
                .or_default()
                .insert(factor.to_string(), w.require(key)?.clone());
        }
    }
    Ok(LokrFile {
        alpha,
        rank,
        groups,
    })
}

/// Install a LoKr adapter file onto `host`. `scale` is the user-facing strength (the
/// `alpha/rank` factor is baked into the reconstructed delta, mirroring the fork).
pub fn apply_lokr(host: &mut impl AdaptableHost, w: &Weights, scale: f32) -> Result<ApplyReport> {
    let file = parse_lokr(w)?;
    let file = &file;
    // Fork-parity residual path keeps the delta at bf16 (PARITY-BF16, sc-2609) unless the host
    // computes wider ([`AdaptableHost::lycoris_dtype`]).
    let dtype = host.lycoris_dtype();
    let groups = file.groups.iter().map(|(path, factors)| {
        (
            path.as_str(),
            move |base: &[i32]| file.delta(factors, base, dtype),
            move |strength: f32, base: &[i32]| file.factors(factors, strength, base, dtype),
        )
    });
    install_lycoris_groups(host, groups, scale, LycorisKeyResolution::Dotted)
}

// ---- Third-party LyCORIS LoKr (sc-3642) ----------------------------------------------------------

/// Third-party LoKr factor suffixes — the PEFT set plus `lokr_t2` (the lycoris tucker/CP factor).
/// `.lokr_w1_a`/`_b` precede the bare `.lokr_w1` so exact-suffix matching never mis-binds. (gen-core.)
pub(super) use gen_core::weightsmeta::LOKR_TP_SUFFIXES;

/// `true` if any key is a LoKr factor (`*.lokr_w…`), regardless of `networkType` metadata. This is
/// how a **third-party** LyCORIS LoKr (kohya / ai-toolkit / lycoris-lib) is recognized — those files
/// ship the Kronecker factors but NOT SceneWorks' peft `networkType=lokr` stamp that [`is_lokr`]
/// keys off. (A `lokr_t2` tucker factor always co-occurs with `lokr_w2_a`, so `.lokr_w` suffices.)
pub fn is_lokr_keys(w: &Weights) -> bool {
    wmeta::keys_contain_lokr(w.keys())
}

/// One module's third-party LoKr factors. Unlike the peft [`LokrFile`] (one global `(alpha, rank)`
/// from metadata), a third-party file carries **per-module** factor shapes + an optional per-module
/// `.alpha` scalar, so rank/alpha/scale are derived per module here.
#[derive(Default)]
pub struct ThirdPartyLokr {
    pub(super) w1: Option<Array>,
    pub(super) w1_a: Option<Array>,
    pub(super) w1_b: Option<Array>,
    pub(super) w2: Option<Array>,
    pub(super) w2_a: Option<Array>,
    pub(super) w2_b: Option<Array>,
    pub(super) t2: Option<Array>,
    pub(super) alpha: Option<f32>,
}

/// Exact third-party rank/shape gate, shared by live tensors and pre-load header normalization.
/// The historical diagnostic says "2-D", but the actual accepted condition is AT LEAST two axes;
/// do not turn this into an exact-matrix restriction on otherwise legal batched factors.
pub fn thirdparty_lokr_rank(
    w1_a: Option<&[i32]>,
    t2: Option<&[i32]>,
    w2_a: Option<&[i32]>,
) -> Result<Option<f32>> {
    let selected = if let Some(shape) = w1_a {
        Some((shape, 1, "lokr_w1_a"))
    } else if let Some(shape) = t2 {
        Some((shape, 0, "lokr_t2"))
    } else {
        w2_a.map(|shape| (shape, 1, "lokr_w2_a"))
    };
    let Some((shape, axis, what)) = selected else {
        return Ok(None);
    };
    if shape.len() < 2 {
        return Err(Error::Msg(format!(
            "lokr adapter: {what} factor must be 2-D, got {}-D shape {:?}",
            shape.len(),
            shape
        )));
    }
    let rank = shape[axis] as f32;
    if rank <= 0.0 {
        return Err(Error::Msg(format!(
            "lokr adapter: {what} factor has zero rank (zero leading dim)"
        )));
    }
    Ok(Some(rank))
}

/// Exact shared third-party alpha/rank normalization, including the rank-free full-factor case.
/// Header-only configured-graph facts use the same branch as the executable loader.
pub fn thirdparty_lokr_scale(rank: Option<f32>, alpha: Option<f32>) -> f32 {
    match rank {
        None => 1.0,
        Some(rank) => alpha.unwrap_or(rank) / rank,
    }
}

impl ThirdPartyLokr {
    /// The factorization rank (`lora_dim`). lycoris lays the factors out inconsistently, so derive in
    /// a fixed order from whichever decomposed factor is present: `lokr_w1_a` is `[shape0, dim]`
    /// (dim = `shape[1]`); the tucker `lokr_t2` is `[dim, dim, kH, kW]` (dim = `shape[0]`); the
    /// non-tucker `lokr_w2_a` is `[shape0, dim]` (dim = `shape[1]`). `None` when **both** factors are
    /// full — lycoris then forces `alpha = lora_dim` ⇒ scale 1, so rank is unused.
    ///
    /// sc-9099: a corrupt/adversarial third-party file can carry a 1-D/0-D factor (the shape index used
    /// to panic) or a zero dim (→ `scale = alpha/0` = inf/NaN baked into the reconstructed delta and
    /// silently installed). `Result` so both are rejected with a typed error instead; mirrors the
    /// sc-5252 LoRA rank-0 guard.
    pub(super) fn rank(&self) -> Result<Option<f32>> {
        thirdparty_lokr_rank(
            self.w1_a.as_ref().map(Array::shape),
            self.t2.as_ref().map(Array::shape),
            self.w2_a.as_ref().map(Array::shape),
        )
    }

    /// LyCORIS `scale`: `alpha / lora_dim` (alpha defaulting to `lora_dim`), EXCEPT both-full forces
    /// scale 1 (mirrors `LokrModule.__init__`: `if use_w1 and use_w2: alpha = lora_dim`).
    pub(super) fn scale(&self) -> Result<f32> {
        Ok(thirdparty_lokr_scale(self.rank()?, self.alpha))
    }

    /// Reconstruct this module's `ΔW` (lycoris per-module scale baked in) at `out_dtype`. `pub` so the
    /// merge-path providers (SDXL/Wan/LTX, sc-3671) reuse the exact derivation + reconstruction and
    /// differ only in how they install it (in-place merge vs forward residual).
    pub fn delta(&self, base_shape: &[i32], out_dtype: Dtype) -> Result<Array> {
        reconstruct_lokr_delta_scaled(
            self.scale()?,
            base_shape,
            self.w1.as_ref(),
            self.w1_a.as_ref(),
            self.w1_b.as_ref(),
            self.w2.as_ref(),
            self.t2.as_ref(),
            self.w2_a.as_ref(),
            self.w2_b.as_ref(),
            out_dtype,
        )
    }

    /// The **structured** (deferred, allocation-free) counterpart to [`delta`](Self::delta) for a
    /// packed base (sc-10050): build the small `[a,c]`/`[b,d]` Kronecker factors (the FULL scale
    /// `lycoris_per_module_scale · strength` baked into `w2`) via [`build_lokr_factors`], so the residual
    /// applies the vec-trick without ever materializing the `[out,in]` delta. The user `strength` is
    /// baked in here (the structured residual carries no separate scale). Returns `Ok(None)` for a
    /// **tucker/CP** `w2` (`lokr_t2`, conv-only) that has no 2-D matrix form — the caller then falls back
    /// (materialize on dense, or a clear error on packed). `pub` so the video providers (Wan/LTX) reuse
    /// the exact derivation.
    pub fn factors(
        &self,
        strength: f32,
        base_shape: &[i32],
        out_dtype: Dtype,
    ) -> Result<Option<LokrFactors>> {
        build_lokr_factors(
            self.scale()? * strength,
            base_shape,
            self.w1.as_ref(),
            self.w1_a.as_ref(),
            self.w1_b.as_ref(),
            self.w2.as_ref(),
            self.t2.as_ref(),
            self.w2_a.as_ref(),
            self.w2_b.as_ref(),
            out_dtype,
        )
    }
}

/// Group a third-party LoKr file's tensors by raw module key (the part before `.lokr_*`/`.alpha`).
/// The raw key is whatever the trainer wrote — a `<PREFIX>_<flattened.path>` (kohya/lycoris) or, more
/// rarely, a dotted path; resolution to the host's module map happens in [`apply_lokr_thirdparty`]
/// (or the merge-path providers' own tables, sc-3671).
pub fn parse_lokr_thirdparty(w: &Weights) -> Result<BTreeMap<String, ThirdPartyLokr>> {
    let mut groups: BTreeMap<String, ThirdPartyLokr> = BTreeMap::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        if let Some(raw) = key.strip_suffix(".alpha") {
            if let Some(a) = scalar_alpha(w.require(&key)?)? {
                groups.entry(raw.to_string()).or_default().alpha = Some(a);
            }
            continue;
        }
        for suffix in LOKR_TP_SUFFIXES {
            if let Some(raw) = key.strip_suffix(suffix) {
                let g = groups.entry(raw.to_string()).or_default();
                let t = w.require(&key)?.clone();
                match &suffix[1..] {
                    "lokr_w1" => g.w1 = Some(t),
                    "lokr_w1_a" => g.w1_a = Some(t),
                    "lokr_w1_b" => g.w1_b = Some(t),
                    "lokr_w2" => g.w2 = Some(t),
                    "lokr_w2_a" => g.w2_a = Some(t),
                    "lokr_w2_b" => g.w2_b = Some(t),
                    "lokr_t2" => g.t2 = Some(t),
                    _ => {}
                }
                break;
            }
        }
    }
    Ok(groups)
}

// `resolve_lokr_path` — resolve a third-party LoKr raw module key (`<PREFIX>_<stem>`, the diffusers
// path flattened with `.`→`_`) to a host dotted path, longest-stem-wins — is defined in gen-core
// (`weightsmeta`) and re-exported at the top of this module so the merge-path providers (sc-3671)
// resolve third-party keys against their own module tables.

/// Strip a leading [`COMMON_LORA_PREFIXES`] namespace (`transformer.` / `diffusion_model.`) from a
/// dotted third-party key, so a host whose [`AdaptableHost::adaptable_mut`] routes from the bare
/// module path still resolves it. The PEFT path passes a detected prefix to the appliers, but the
/// third-party LoKr/LoHa dotted-fallback path has no equivalent — without this, e.g. ostris
/// ai-toolkit's Krea-2 LoKr (`diffusion_model.blocks.N.attn.wq…`, sc-8395) matches no target even
/// though the Krea host already aliases `blocks`/`txtfusion`/`wq…`/`mlp` (sc-8185). Returns the key
/// unchanged when it carries no such prefix.
pub fn strip_common_lora_prefix(raw: &str) -> &str {
    for prefix in COMMON_LORA_PREFIXES {
        if let Some(rest) = raw.strip_prefix(prefix) {
            return rest;
        }
    }
    raw
}

/// Install a third-party LyCORIS **LoKr** file (LoHa is sc-3643) onto `host`. Reconstructs each
/// module's Kronecker delta from its per-module factors (full / low-rank / tucker) at the lycoris
/// scale and stacks it as an [`Adapter::Lokr`] residual at the user `scale` — the same install as
/// peft [`apply_lokr`], differing only in (a) per-module rank/alpha derivation and (b) resolving the
/// trainer's flattened key names to the host's dotted module map. Unresolved paths are surfaced in
/// `unmatched_paths`, never silently dropped.
pub fn apply_lokr_thirdparty(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
) -> Result<ApplyReport> {
    let table = kohya_table(&host.adaptable_paths());
    let groups = parse_lokr_thirdparty(w)?;
    // The host's LyCORIS dtype (bf16 by default, PARITY-BF16, sc-2609) — same as peft `apply_lokr`.
    let dtype = host.lycoris_dtype();
    let groups = groups.iter().map(|(raw, g)| {
        (
            raw.as_str(),
            move |base: &[i32]| g.delta(base, dtype),
            move |strength: f32, base: &[i32]| g.factors(strength, base, dtype),
        )
    });
    install_lycoris_groups(
        host,
        groups,
        scale,
        LycorisKeyResolution::Thirdparty(&table),
    )
}
