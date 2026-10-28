//! The shared LyCORIS install machinery (LoKr / LoHa, dotted and BFL keyed): the delta-
//! materialization budget, the per-target plan, and `ApplyReport`.
//!
//! Split out of `loader.rs` (sc-27427); every item is re-exported there, so
//! `mlx_gen::adapters::loader::…` paths are unchanged.

use super::*;

/// Read a scalar adapter value (an `alpha`) as `f32`, regardless of its on-disk dtype. Trained
/// adapters store `alpha` in their compute dtype: real kohya/BFL FLUX LoRAs ship it **bf16** (sc-2657),
/// and `Array::as_slice::<f32>()` `unwrap`s a hard dtype-mismatch (it never casts), so reading a bf16
/// scalar that way panics. Cast to f32 first (exact for the small integer alphas these files carry, and
/// a no-op when already f32); a `[]`- or `[1]`-shaped scalar both read as a one-element slice.
///
/// A size-0 `alpha` tensor (a malformed third-party adapter file) has no data pointer to borrow, so
/// `as_slice` — `try_as_slice().unwrap()` — would panic rather than fall through to the trailing
/// `first()`. Guard it like [`crate::array::host_i32`] so one bad file fails its single job with a
/// typed error instead of aborting the worker.
pub(super) fn scalar_alpha(a: &Array) -> Result<Option<f32>> {
    if a.size() == 0 {
        // The callers only reach here for an `alpha` key that *exists* (`w.require(..)`), so a
        // present-but-empty tensor is a malformed file — NOT an absent alpha. Returning `Ok(None)`
        // here would be indistinguishable from "no alpha key", silently falling back to the
        // `alpha == rank ⇒ scale 1.0` default and mis-scaling the adapter while reporting success.
        // Fail this one job with a typed error instead (sc-5281, matching the contract documented above).
        return Err(Error::Msg(
            "scalar_alpha: alpha tensor present but empty (size 0) — malformed adapter file".into(),
        ));
    }
    Ok(a.as_dtype(Dtype::Float32)?
        .try_as_slice::<f32>()
        .map_err(|e| Error::Msg(format!("scalar_alpha: not a readable scalar array: {e}")))?
        .first()
        .copied())
}

/// Outcome of installing an adapter file: how many target modules were adapted, and any
/// adapter keys that matched no module in the host (surfaced, never silently dropped).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ApplyReport {
    pub applied: usize,
    pub unmatched_paths: Vec<String>,
    /// ComfyUI/lightx2v diff-patch (`.diff`/`.diff_b`) deltas that did **not** land — skipped
    /// (shape-incompatible, or a weight `.diff` aimed at a packed Linear) plus stems that matched no
    /// module (sc-15326). Always empty on [`apply_adapters_strict`], which does not read diff-patch
    /// keys at all; populated by [`apply_adapters_strict_with_diff_patch`].
    ///
    /// Carried on the report rather than left to `stderr` because a partial adapter install is a fact a
    /// *user* has to be told — the render is changed but not by the whole LoRA, which looks like a
    /// success. A provider stamps this into the asset's provenance; an engine log line is not a
    /// product surface.
    pub diff_patch_unapplied: Vec<String>,
}

/// How a LyCORIS applier resolves a file's raw module key to the host's dotted module path
/// (the one axis the non-BFL LyCORIS installers differ on — see [`install_lycoris_groups`]).
pub(super) enum LycorisKeyResolution<'t> {
    /// Keys are already bare dotted module paths (the peft LoKr convention, [`apply_lokr`]).
    Dotted,
    /// Trainer raw keys: a kohya/lycoris `<PREFIX>_<flattened>` stem resolves through the
    /// host-derived flattened→dotted table ([`kohya_table`]); anything else is treated as
    /// already-dotted after stripping a common `transformer.`/`diffusion_model.` namespace
    /// ([`strip_common_lora_prefix`], sc-8395).
    Thirdparty(&'t BTreeMap<String, String>),
}

/// Bytes → GiB base for the sc-10678 pre-flight guard's messages.
pub(super) const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The materialized `[out,in]` LoKr/LoHa delta dtype is bf16 by default (PARITY-BF16, sc-2609): 2
/// bytes/element; a host whose [`AdaptableHost::lycoris_dtype`] is f32 materializes 4.
pub(super) const DELTA_DTYPE_BYTES: usize = 2;

/// Fraction of the MLX memory limit the guard treats as usable, leaving headroom for the forward-pass
/// activation working set (mirrors the sc-4874 training pre-flight's 0.85).
pub(super) const MATERIALIZE_BUDGET_HEADROOM: f64 = 0.85;

/// Bytes a materialized `[out, in]` bf16 delta occupies — the memory a non-deferrable adapter (LoHa /
/// tucker-LoKr) forces onto a packed base per target (sc-10678). Pure (no MLX state) ⇒ deterministically
/// unit-testable.
pub(super) fn projected_delta_bytes(base_shape: &[i32], dtype: Dtype) -> usize {
    let element = match dtype {
        Dtype::Float32 => 4,
        _ => DELTA_DTYPE_BYTES,
    };
    base_shape
        .iter()
        .map(|&d| d.max(0) as usize)
        .product::<usize>()
        * element
}

/// Whether materializing `projected_new` more bytes on top of the `active` resident set would cross the
/// safe budget (`limit × headroom`). `limit == 0` means MLX has no cap set (unlimited) ⇒ never guard.
/// Pure (the caller supplies the live `get_active_memory()` / `get_memory_limit()`) ⇒ deterministically
/// unit-testable without touching global MLX memory state.
pub(super) fn materialization_exceeds_budget(
    active: usize,
    projected_new: usize,
    limit: usize,
) -> bool {
    if limit == 0 {
        return false;
    }
    let safe = (limit as f64 * MATERIALIZE_BUDGET_HEADROOM) as usize;
    active.saturating_add(projected_new) > safe
}

/// What one resolved LyCORIS group will do in the install pass — computed up front so the packed +
/// non-deferrable **materialization** (LoHa / tucker-LoKr) can be projected and budget-checked BEFORE any
/// `[out,in]` delta is allocated (sc-10678). Deferrable / dense / unmatched groups carry no such cost.
pub(super) enum LycorisPlan<F> {
    /// Key resolved to no module (surfaced in `unmatched_paths` under its raw spelling).
    Unmatched(String),
    /// Dense base → materialize `[out,in]` and stack as `Adapter::Lokr` (unchanged, fork-parity).
    Dense { dotted: String, delta: F },
    /// Packed base with a deferrable Kronecker form → stack `Adapter::LokrStructured` (allocation-free).
    Deferred {
        dotted: String,
        factors: LokrFactors,
    },
    /// Packed base, NOT deferrable → materialize `[out,in]` (the budget-guarded case).
    Materialize {
        dotted: String,
        base_shape: Vec<i32>,
        delta: F,
    },
}

/// Shared install skeleton for the non-BFL LyCORIS appliers ([`apply_lokr`],
/// [`apply_lokr_thirdparty`], [`apply_loha_thirdparty`] — sc-9099): resolve each module's raw key to
/// a host dotted path per `resolution`, then stack a residual onto the target. A key that resolves to
/// no module is surfaced in `unmatched_paths` under its raw spelling, never silently dropped. The BFL
/// fused→split twin is [`install_bfl_lycoris`].
///
/// **On a packed base, prefer the deferred residual** (sc-10578):
///   * **Dense base** — `delta(base_shape)` materializes the `[out,in]` `ΔW` (the closure bakes in the
///     format's `alpha/rank` scale and dtype) and it stacks as an [`Adapter::Lokr`] at the user
///     `scale`. Byte-for-byte the pre-sc-10578 behavior, which every family's fork-parity goldens
///     depend on.
///   * **Packed base, deferrable module** — `factors(scale, base_shape)` builds the small Kronecker
///     factors and it stacks as an [`Adapter::LokrStructured`], applying the residual via the vec-trick
///     **without ever allocating `[out,in]`**. Note the scale: the structured residual carries none of
///     its own (it is baked into `factors.w2`), so the user `scale` is passed *in* here, whereas the
///     dense path passes it to `Adapter::Lokr`. Getting that backwards is a silent mis-scale.
///   * **Packed base, NOT deferrable** (`factors` → `Ok(None)`: LoHa's Hadamard product, which has no
///     deferred form at all; a tucker/CP `lokr_t2`; a conv-shaped factor) — **fall back to the
///     materialized delta**, exactly as before. It is correct: [`Adapter::Lokr`]'s residual is
///     `scale · x·ΔWᵀ`, which never reads the base, so it rides over `quantized_matmul` unchanged.
///     It is merely memory-hungry.
///
/// **The materialized fallback is budget-guarded, not silent (sc-10678).** An earlier cut of sc-10578
/// rejected the non-deferrable packed combination outright, reasoning from sc-10051 (LoHa on quantized
/// Wan: steer to bf16, do not materialize). That was wrong as a blanket rule *here*: this shared path is
/// reached by eleven provider crates, adapters install **after** `.quantize()` in every one, and the
/// worker defaults `spec.quantize` to `Some(Q8)` for **every** MLX model — so `is_quantized()` is true on
/// the default path almost everywhere, and erroring unconditionally would turn a working third-party LoHa
/// that *fits* into a hard failure. So the fallback stays: a materialization that fits the memory budget
/// runs, on the user's chosen (quantized) tier. But a materialization that would **not** fit is refused
/// UP FRONT with a catchable, actionable error ([`materialization_exceeds_budget`]) rather than left to a
/// silent mid-run SIGKILL when the OS hard-kills the worker. The tier is never switched (it is a creative
/// choice, not a perf knob); bf16 is only *suggested*, framed as a different render. This is the sc-10678
/// decision (E: pre-flight memory guard), mirroring the sc-4874 training pre-flight.
pub(super) fn install_lycoris_groups<K, F, G>(
    host: &mut impl AdaptableHost,
    groups: impl IntoIterator<Item = (K, F, G)>,
    scale: f32,
    resolution: LycorisKeyResolution<'_>,
) -> Result<ApplyReport>
where
    K: AsRef<str> + Into<String>,
    F: FnOnce(&[i32]) -> Result<Array>,
    G: FnOnce(f32, &[i32]) -> Result<Option<LokrFactors>>,
{
    let mut report = ApplyReport::default();

    // Pass 1 — resolve every group to a concrete plan, consuming the `factors` probe (which is cheap:
    // it builds the small Kronecker factors or returns `None`, but never the `[out,in]` delta). Sum the
    // bytes EVERY **materialization** will allocate — the packed + non-deferrable case AND the dense-base
    // case, both of which stack a full `[out,in]` bf16 delta per target (sc-11129) — so we can
    // budget-check them before the (potentially multi-GB) reconstruction runs. Only deferred (structured,
    // allocation-free) and unmatched groups add nothing to `projected_materialize`.
    let mut plans: Vec<LycorisPlan<F>> = Vec::new();
    let mut projected_materialize: usize = 0;
    let dtype = host.lycoris_dtype();
    // Resolved module path → the raw key that claimed it. Two raw spellings of one module in one file
    // (`lycoris_X` beside `lora_unet_X`, `diffusion_model.X` beside `X`) would otherwise BOTH install
    // and double-apply the delta — refuse instead (sc-24158).
    let mut claimed: BTreeMap<String, String> = BTreeMap::new();
    for (raw, delta, factors) in groups {
        let dotted: String = match &resolution {
            LycorisKeyResolution::Dotted => raw.as_ref().to_string(),
            LycorisKeyResolution::Thirdparty(table) => resolve_lokr_path(raw.as_ref(), table)
                .unwrap_or_else(|| strip_common_lora_prefix(raw.as_ref()))
                .to_string(),
        };
        if let Some(previous) = claimed.insert(dotted.clone(), raw.as_ref().to_string()) {
            return Err(Error::Msg(format!(
                "LyCORIS adapter carries both `{previous}` and `{}`, which resolve to the same \
                 module `{dotted}`; refusing a double apply",
                raw.as_ref()
            )));
        }
        let parts: Vec<&str> = dotted.split('.').collect();
        // SC-18319 — pass 1 only *reads* (is the base packed, how big is it), so it goes through the
        // PROBE half of the host surface. Taking the `&mut` here would unfuse every `FusedQkvProjection`
        // it walked past — dismantling the fusion during a scan that installs nothing.
        match host.adaptable_facts(&parts) {
            // Packed base: take the deferred form when the module has one, else plan a materialization.
            Some(facts) if facts.is_quantized => {
                let base_shape = facts.base_shape;
                match factors(scale, &base_shape)? {
                    Some(f) => plans.push(LycorisPlan::Deferred { dotted, factors: f }),
                    None => {
                        projected_materialize += projected_delta_bytes(&base_shape, dtype);
                        plans.push(LycorisPlan::Materialize {
                            dotted,
                            base_shape,
                            delta,
                        });
                    }
                }
            }
            Some(facts) => {
                // Dense base still materializes the same `[out,in]` bf16 delta as a stacked residual —
                // the OOM the guard prevents on packed tiers is reachable here too (sc-11129). Project it.
                projected_materialize += projected_delta_bytes(&facts.base_shape, dtype);
                plans.push(LycorisPlan::Dense { dotted, delta });
            }
            None => plans.push(LycorisPlan::Unmatched(raw.into())),
        }
    }

    // sc-10678 — pre-flight memory guard. A non-deferrable adapter (LoHa / tucker-LoKr) must materialize
    // a dense `[out,in]` delta per target — on a PACKED (q4/q8) base (no deferred form) OR a dense base
    // (sc-11129: the stacked residual is the same footprint). Over a whole model that can dwarf the base
    // footprint; left unchecked it OOMs the worker via an uncatchable SIGKILL mid-run. Refuse UP FRONT —
    // but only when the projected materialization would actually exceed the budget, so an adapter that
    // fits still runs on the user's chosen tier (a plain LoRA or an all-deferred LoKr never reach this
    // branch: `projected_materialize == 0`).
    if projected_materialize > 0 {
        let active = get_active_memory();
        let limit = get_memory_limit();
        if materialization_exceeds_budget(active, projected_materialize, limit) {
            return Err(Error::Msg(format!(
                "This adapter (LoHa / tucker-LoKr) has no allocation-free form, so it must materialize a \
                 dense `[out,in]` delta for every target — ~{:.1} GB on top of the ~{:.1} GB already \
                 resident, over the ~{:.1} GB budget. Use a plain LoRA/LoKr instead (near-zero extra \
                 memory on any tier). On a quantized tier, the bf16 tier merges this adapter into the \
                 dense weight in place — note bf16 renders differently from the quantized tier.",
                projected_materialize as f64 / GIB,
                active as f64 / GIB,
                (limit as f64 * MATERIALIZE_BUDGET_HEADROOM) / GIB,
            )));
        }
    }

    // Pass 2 — install. Behaviour is identical to the pre-guard single loop: deferred residual on a
    // deferrable packed module, materialized `Adapter::Lokr` on a non-deferrable packed or dense module,
    // and `unmatched_paths` under the raw key. Re-resolving by `dotted` is a cheap map lookup (the host
    // is not structurally mutated between passes).
    // A pass-2 re-resolution that returns `None` (the module vanished between passes — unreachable on a
    // stable host today, but lazy/offloaded module trees are now a live pattern, epic 10834) must be
    // SURFACED in `unmatched_paths`, never silently dropped, honoring the function's contract
    // (sc-11129). The dotted path is the resolved spelling (the raw key is gone by pass 2).
    for plan in plans {
        match plan {
            LycorisPlan::Unmatched(raw) => report.unmatched_paths.push(raw),
            LycorisPlan::Deferred { dotted, factors } => {
                let parts: Vec<&str> = dotted.split('.').collect();
                if let Some(lin) = host.adaptable_mut(&parts) {
                    lin.push(Adapter::LokrStructured { factors });
                    report.applied += 1;
                } else {
                    report.unmatched_paths.push(dotted);
                }
            }
            LycorisPlan::Materialize {
                dotted,
                base_shape,
                delta,
            } => {
                let delta = delta(&base_shape)?;
                let parts: Vec<&str> = dotted.split('.').collect();
                if let Some(lin) = host.adaptable_mut(&parts) {
                    lin.push(Adapter::Lokr { delta, scale });
                    report.applied += 1;
                } else {
                    report.unmatched_paths.push(dotted);
                }
            }
            LycorisPlan::Dense { dotted, delta } => {
                let parts: Vec<&str> = dotted.split('.').collect();
                if let Some(lin) = host.adaptable_mut(&parts) {
                    let base_shape = lin.base_shape();
                    let delta = delta(&base_shape)?;
                    lin.push(Adapter::Lokr { delta, scale });
                    report.applied += 1;
                } else {
                    report.unmatched_paths.push(dotted);
                }
            }
        }
    }
    Ok(report)
}
