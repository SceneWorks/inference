//! LTX-2.3 LoRA application (sc-2687) — wires the reference `mlx_video/lora/` into the Rust
//! pipeline. The reference *defines* its `lora/` module but never invokes it from `generate_av.py`,
//! so this is the wiring plus the LTX key→module map.
//!
//! **Strategy: forward-time residual** (the reference `lora/apply.py::LoRALinear`,
//! `out + scale·strength·(x·Aᵀ·Bᵀ)`), not a merged weight. The reference also offers
//! `apply_loras_to_model` (dequant Q8 → merge → dense bf16); residual is chosen because the shipped
//! transformer is **Q8-only** at 22B — merging a full attn+ff LoRA would dequantize ~15 GB to bf16,
//! and the net-new per-pass strength would double it — and because residual leaves the bit-exact base
//! forward (sc-2842) untouched. Installed onto the model tree's [`crate::transformer::Linear`]s via
//! `LtxDiT::adaptable_mut`.
//!
//! **Format.** PEFT `lora_A`/`lora_B` (`.default` infix tolerated) and kohya `lora_down`/`lora_up`,
//! per-module `.alpha` (default = rank); real LTX-2.3 files ship PEFT, bf16, `diffusion_model.`-prefixed.
//! `scale = alpha/rank` (the reference `LoRAWeights.scale`).
//!
//! **LoKr** (sc-2393) is net-new — the reference `lora/` is LoRA-only, so this is parity-PLUS. A LoKr
//! file (`networkType=lokr`, `‹path›.lokr_w1/w2[_a/_b]`) is parsed by the core `parse_lokr`, its
//! per-module `[out,in]` delta reconstructed via `reconstruct_lokr_delta` (`alpha/rank` folded in),
//! mapped through this same LTX key→module table, and installed as a forward-time residual carrying
//! the same per-pass strength as LoRA.
//!
//! **Skips, never errors-on-skip.** Mirrors the reference (`apply_loras_to_weights` counts skipped
//! modules, never raises): audio / `av_ca` / `a2v` targets (the video-only port has no such modules)
//! and the PixArt-spelled adaLN embedder (`linear_1/2` ≠ the checkpoint's `linear1/2`) resolve to no
//! module and are reported, not dropped. We error only if a non-empty spec list matched *nothing*.

use std::collections::BTreeMap;

use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::loader::{
    is_loha_keys, is_lokr, is_lokr_keys, parse_loha_thirdparty, parse_lokr, parse_lokr_thirdparty,
};
use mlx_gen::gen_core::weightsmeta::{LoraAdapterMeta, LORA_ADAPTER_METADATA_KEY};
use mlx_gen::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::transformer::LtxAdaptable;

/// LoRA key namespace prefixes stripped (longest-first), matching the reference
/// `_normalize_ltx_lora_key`. SceneWorks' trained LTX LoRAs use `diffusion_model.`.
const PREFIXES: [&str; 3] = ["model.diffusion_model.", "diffusion_model.", "model."];

/// Outcome of applying the LTX adapter specs: residuals installed and the LoRA module paths that
/// resolved to no target (surfaced, never silently dropped — audio/av_ca/a2v and PixArt-spelled
/// adaLN embedder leaves).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct LtxLoraReport {
    pub applied: usize,
    pub skipped: Vec<String>,
}

#[derive(Clone, Copy)]
enum Role {
    Down, // lora_A / lora_down → A [rank, in]
    Up,   // lora_B / lora_up   → B [out, rank]
    Alpha,
}

#[derive(Default)]
struct LoraParts {
    down: Option<Array>,
    up: Option<Array>,
    alpha: Option<f32>,
}

/// Normalize a LoRA module path to the LTX checkpoint's naming (the reference
/// `_normalize_ltx_lora_key`): strip a known prefix, then the diffusers→checkpoint renames
/// `to_out.0`→`to_out`, `ff.net.0.proj`→`ff.proj_in`, `ff.net.2`→`ff.proj_out` (+ audio analogues,
/// which the video-only port then resolves to no module). The leading `.` in each pattern keeps
/// `.ff.net.*` from matching inside `audio_ff.net.*`, exactly as the reference relies on.
pub(crate) fn normalize_ltx_key(key: &str) -> String {
    let stripped = PREFIXES
        .iter()
        .find_map(|p| key.strip_prefix(p))
        .unwrap_or(key);
    let mut t = stripped.to_string();
    if let Some(head) = t.strip_suffix(".to_out.0") {
        t = format!("{head}.to_out");
    }
    t = t.replace(".to_out.0.", ".to_out.");
    t = t.replace(".ff.net.0.proj.", ".ff.proj_in.");
    t = t.replace(".ff.net.0.proj", ".ff.proj_in");
    t = t.replace(".ff.net.2.", ".ff.proj_out.");
    t = t.replace(".ff.net.2", ".ff.proj_out");
    t = t.replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.");
    t = t.replace(".audio_ff.net.0.proj", ".audio_ff.proj_in");
    t = t.replace(".audio_ff.net.2.", ".audio_ff.proj_out.");
    t = t.replace(".audio_ff.net.2", ".audio_ff.proj_out");
    t
}

/// Suffix → role, longest-first. PEFT `lora_A/B`, kohya `lora_down/up`, the peft-export `.default`
/// infix, and a bare `.alpha`. `lora_A`/`lora_down` are the A (down) factor; `lora_B`/`lora_up` the B.
const SUFFIXES: [(&str, Role); 9] = [
    (".lora_A.default.weight", Role::Down),
    (".lora_B.default.weight", Role::Up),
    (".lora_down.default.weight", Role::Down),
    (".lora_up.default.weight", Role::Up),
    (".lora_A.weight", Role::Down),
    (".lora_B.weight", Role::Up),
    (".lora_down.weight", Role::Down),
    (".lora_up.weight", Role::Up),
    (".alpha", Role::Alpha),
];

/// Surface inventory of the LoRA **targets a file carries** (sc-13019): unique normalized module
/// paths with BOTH a down and an up factor, split into `(video, audio_or_cross_modal)`. The
/// audio/cross bucket is classified textually from the LTX module namespace (`audio*` blocks,
/// `av_ca`/`a2v` cross-modal attention) — the same classification the reference's video-only skip
/// behavior produces.
///
/// This is the file-derived half of the structural adapter gates: the tests assert the *routing
/// report* (what [`apply_ltx_adapters`] resolved on a given model) against this *key-inventory*
/// expectation, so the counts follow whatever lora is under test instead of hardcoding one training
/// recipe's shape (the old gates baked in one character lora's 576/1632 counts, making the suites
/// un-runnable without that exact asset), while a routing regression that drops modules still trips
/// the comparison.
///
/// Scope notes: orphan down/up factors and `.alpha`-only keys are counted by NEITHER bucket (the
/// gates assume a well-formed lora; the apply path surfaces such keys as `skipped`, so a malformed
/// file fails the skipped-exact asserts rather than passing silently). Flattened kohya keys (no
/// dots) are inventoried as-is but resolve to no module — same failure direction.
pub fn lora_target_inventory(w: &Weights) -> (Vec<String>, Vec<String>) {
    use std::collections::BTreeSet;
    let mut downs: BTreeSet<String> = BTreeSet::new();
    let mut ups: BTreeSet<String> = BTreeSet::new();
    for key in w.keys() {
        let Some((stem, role)) = SUFFIXES
            .iter()
            .find_map(|(suf, role)| key.strip_suffix(suf).map(|s| (s, *role)))
        else {
            continue;
        };
        match role {
            Role::Down => {
                downs.insert(normalize_ltx_key(stem));
            }
            Role::Up => {
                ups.insert(normalize_ltx_key(stem));
            }
            Role::Alpha => {}
        }
    }
    let (mut video, mut audio) = (Vec::new(), Vec::new());
    for path in downs.intersection(&ups) {
        if path.contains("audio") || path.contains("av_ca") || path.contains("a2v") {
            audio.push(path.clone());
        } else {
            video.push(path.clone());
        }
    }
    (video, audio)
}

/// Read a scalar `.alpha` as f32 regardless of on-disk dtype (real files ship it bf16; a direct
/// `as_slice::<f32>()` would panic on a dtype mismatch). A `[]`- or `[1]`-shaped scalar both read.
fn read_alpha(a: &Array) -> Result<f32> {
    // Adapter files are external input; an empty `.alpha` tensor would panic on `[0]`. Read the first
    // element defensively and error on an empty tensor instead (F-010).
    a.as_dtype(Dtype::Float32)?
        .as_slice::<f32>()
        .first()
        .copied()
        .ok_or_else(|| Error::Msg("ltx_2_3 adapter: empty .alpha tensor".into()))
}

/// Per-pass user strengths for one adapter: `spec.pass_scales` (one per distilled stage, validated
/// to `num_passes`) or `spec.scale` (uniform — a length-1 vec the forward clamps into). The `strength`
/// is the user knob; `alpha/rank` is folded in separately (into B for LoRA via [`pass_scales`], into
/// the delta for LoKr via `reconstruct_lokr_delta`).
fn pass_strengths(spec: &AdapterSpec, num_passes: usize) -> Result<Vec<f32>> {
    let scales = match &spec.pass_scales {
        None => vec![spec.scale],
        Some(v) => {
            if v.len() != num_passes {
                return Err(Error::Msg(format!(
                    "ltx_2_3 adapter {}: pass_scales has {} entries but the distilled pipeline runs \
                     {num_passes} passes",
                    spec.path.display(),
                    v.len()
                )));
            }
            v.clone()
        }
    };
    // An empty pass_scale (reachable when num_passes == 0 and pass_scales = Some([])) would later
    // underflow `Linear::forward`'s per-pass index; reject it at load time instead (F-009).
    if scales.is_empty() {
        return Err(Error::Msg(format!(
            "ltx_2_3 adapter {}: pass_scales must have at least one entry",
            spec.path.display()
        )));
    }
    if let Some((pass, scale)) = scales
        .iter()
        .enumerate()
        .find(|(_, scale)| !scale.is_finite())
    {
        return Err(Error::Msg(format!(
            "ltx_2_3 adapter {}: effective scale for pass {pass} must be finite (got {scale})",
            spec.path.display()
        )));
    }
    Ok(scales)
}

/// LoRA per-pass effective scales for one resolved module: `(alpha/rank)·strength`. `strength` comes
/// from [`pass_strengths`]; the `alpha/rank·strength` product is computed in f64 then f32, matching
/// the reference's Python-float `scale * strength`. (LoKr bakes `alpha/rank` into the delta, so it
/// uses [`pass_strengths`] directly — no fold here.)
fn pass_scales(spec: &AdapterSpec, alpha: f32, rank: f32, num_passes: usize) -> Result<Vec<f32>> {
    let eff = |strength: f32| ((alpha as f64 / rank as f64) * strength as f64) as f32;
    let scales: Vec<f32> = pass_strengths(spec, num_passes)?
        .into_iter()
        .map(eff)
        .collect();
    if let Some((pass, scale)) = scales
        .iter()
        .enumerate()
        .find(|(_, scale)| !scale.is_finite())
    {
        return Err(Error::Msg(format!(
            "ltx_2_3 adapter {}: effective LoRA scale for pass {pass} must be finite (got {scale})",
            spec.path.display()
        )));
    }
    Ok(scales)
}

fn require_spec_applied(
    spec: &AdapterSpec,
    applied_before: usize,
    applied_after: usize,
) -> Result<()> {
    if applied_after == applied_before {
        return Err(Error::Msg(format!(
            "ltx_2_3 adapter {} matched no target module — every selected adapter must apply",
            spec.path.display()
        )));
    }
    Ok(())
}

/// LTX-2.5's published adapters carry a file-wide, explicit scale contract.  Unlike the older
/// 2.3 community formats, the split provider must not guess a rank from a factor or quietly
/// substitute a per-target default: doing so can make a formally accepted adapter inert or apply
/// it at the wrong strength.
#[derive(Clone, Copy, Debug)]
struct Ltx25Scale {
    rank: i32,
    alpha: f32,
}

fn ltx25_scale(
    rank: Option<&str>,
    alpha: Option<&str>,
    source: &std::path::Path,
) -> Result<Ltx25Scale> {
    let rank = rank.ok_or_else(|| {
        Error::Msg(format!(
            "ltx_2_5 adapter {} is missing required `lora_rank` safetensors metadata",
            source.display()
        ))
    })?;
    let rank = rank.parse::<i32>().map_err(|_| {
        Error::Msg(format!(
            "ltx_2_5 adapter {} has non-positive-integer `lora_rank` metadata `{rank}`",
            source.display()
        ))
    })?;
    if rank == 0 {
        return Err(Error::Msg(format!(
            "ltx_2_5 adapter {} has invalid `lora_rank` metadata 0",
            source.display()
        )));
    }
    let alpha = alpha.ok_or_else(|| {
        Error::Msg(format!(
            "ltx_2_5 adapter {} is missing required `lora_alpha` safetensors metadata",
            source.display()
        ))
    })?;
    let parse_alpha = |value: &str| -> Result<f32> {
        let value = value.parse::<f32>().map_err(|_| {
            Error::Msg(format!(
                "ltx_2_5 adapter {} has non-numeric `lora_alpha` metadata `{value}`",
                source.display()
            ))
        })?;
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::Msg(format!(
                "ltx_2_5 adapter {} has invalid `lora_alpha` metadata {value}",
                source.display()
            )));
        }
        Ok(value)
    };
    Ok(Ltx25Scale {
        rank,
        alpha: parse_alpha(alpha)?,
    })
}

/// Published LTX-2.5 distilled factors use the declared decomposition rank until a projection
/// dimension is smaller, at which point the physical factor width is capped to that dimension.
fn ltx25_factor_rank(declared_rank: i32, in_features: i32, out_features: i32) -> i32 {
    declared_rank.min(in_features).min(out_features)
}

/// Install one LoRA file's residuals onto `host` at `spec`'s strength, accumulating into `report`.
fn apply_one(
    host: &mut impl LtxAdaptable,
    w: &Weights,
    spec: &AdapterSpec,
    num_passes: usize,
    strict_25: Option<Ltx25Scale>,
    report: &mut LtxLoraReport,
) -> Result<()> {
    // Group factors by normalized module path.
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        let Some((stem, role)) = SUFFIXES
            .iter()
            .find_map(|(suf, role)| key.strip_suffix(suf).map(|s| (s, *role)))
        else {
            continue; // not a LoRA factor key (base weight / bundled extra) — ignore.
        };
        let path = normalize_ltx_key(stem);
        let parts = groups.entry(path).or_default();
        match role {
            Role::Down => parts.down = Some(w.require(&key)?.clone()),
            Role::Up => parts.up = Some(w.require(&key)?.clone()),
            Role::Alpha => parts.alpha = Some(read_alpha(w.require(&key)?)?),
        }
    }

    // PEFT/diffusers `save_lora_adapter` files carry no per-target `.alpha` tensor — `lora_alpha`/`r`
    // (+ per-module overrides) live in the `lora_adapter_metadata` header blob (sc-5513). `None` for a
    // file without it (kohya / trainer files ship a `.alpha` tensor), in which case the per-target
    // `.alpha` or the factor rank is used exactly as before.
    let cfg = LoraAdapterMeta::from_metadata(w.metadata(LORA_ADAPTER_METADATA_KEY));
    for (path, parts) in groups {
        let (Some(down), Some(up)) = (parts.down, parts.up) else {
            // A down/up whose partner targeted a non-LoRA key — skip the orphan, surface the path.
            report.skipped.push(path);
            continue;
        };
        if down.ndim() != 2 || up.ndim() != 2 || down.shape()[0] == 0 || up.shape()[0] == 0 {
            return Err(Error::Msg(format!(
                "ltx adapter {} target `{path}` must have non-empty rank-2 A/B factors",
                spec.path.display()
            )));
        }
        if down.shape()[0] != up.shape()[1] {
            return Err(Error::Msg(format!(
                "ltx adapter {} target `{path}` has incompatible A/B factor shapes {:?} / {:?}",
                spec.path.display(),
                down.shape(),
                up.shape()
            )));
        }
        let segs: Vec<&str> = path.split('.').collect();
        // Effective scaling: per-target `.alpha` tensor → `alpha_pattern`/`lora_alpha` blob → factor
        // rank (today's default). The denominator honors the blob `r`/`rank_pattern` when given
        // (always `> 0`), else the stored `down` leading dim (which equals it for a well-formed file).
        let (cfg_alpha, cfg_rank) = cfg.as_ref().map_or((None, None), |c| c.effective(&path));
        let (rank, alpha) = match strict_25 {
            Some(contract) => {
                let expected = ltx25_factor_rank(contract.rank, down.shape()[1], up.shape()[0]);
                if down.shape()[0] != expected {
                    return Err(Error::Msg(format!(
                        "ltx_2_5 adapter {} target `{path}` has factor rank {} but declares lora_rank {}; projection [{}, {}] requires factor rank {expected}",
                        spec.path.display(),
                        down.shape()[0],
                        contract.rank,
                        up.shape()[0],
                        down.shape()[1]
                    )));
                }
                (contract.rank as f32, contract.alpha)
            }
            None => {
                let rank = cfg_rank.unwrap_or(down.shape()[0] as f32);
                let alpha = parts.alpha.or(cfg_alpha).unwrap_or(rank);
                (rank, alpha)
            }
        };
        let scales = pass_scales(spec, alpha, rank, num_passes)?;
        match host.adaptable_mut(&segs) {
            Some(lin) => {
                if strict_25.is_some() {
                    let base = lin.base_shape();
                    if down.shape()[1] != base[1] || up.shape()[0] != base[0] {
                        return Err(Error::Msg(format!(
                            "ltx_2_5 adapter {} target `{path}` factor shapes {:?} / {:?} do not match base {:?}",
                            spec.path.display(),
                            down.shape(),
                            up.shape(),
                            base
                        )));
                    }
                }
                // Residual form: a = Aᵀ [in, rank], b = Bᵀ [rank, out]; factors keep their loaded
                // (bf16) dtype so the residual promotes against the activation like the reference.
                lin.push_lora(down.t(), up.t(), scales);
                report.applied += 1;
            }
            None => report.skipped.push(path),
        }
    }
    Ok(())
}

/// Install one LoKr file's residuals onto `host` at `spec`'s per-pass strength (sc-2393 — net-new,
/// the reference `lora/` has no LoKr). Each module's `[out,in]` delta is reconstructed from its
/// Kronecker factors via the core `reconstruct_lokr_delta` (`alpha/rank` baked in), keyed at the
/// target linear's base shape, then installed as a forward-time residual carrying the raw per-pass
/// strengths (no further alpha/rank fold). Skips/surfaces a path that resolves to no module, like
/// the LoRA path (audio/av_ca/a2v on the video-only port).
fn apply_one_lokr(
    host: &mut impl LtxAdaptable,
    w: &Weights,
    spec: &AdapterSpec,
    num_passes: usize,
    report: &mut LtxLoraReport,
) -> Result<()> {
    let file = parse_lokr(w)?;
    let strengths = pass_strengths(spec, num_passes)?;
    for (raw_path, factors) in &file.groups {
        let path = normalize_ltx_key(raw_path);
        let segs: Vec<&str> = path.split('.').collect();
        match host.adaptable_mut(&segs) {
            Some(lin) => {
                // Residual path keeps the delta bf16 (PARITY-BF16) like the core LoKr install; the
                // forward casts it to the activation dtype.
                let delta = file.delta(factors, &lin.base_shape(), Dtype::Bfloat16)?;
                lin.push_lokr(delta, strengths.clone());
                report.applied += 1;
            }
            None => report.skipped.push(path),
        }
    }
    Ok(())
}

/// Install one third-party LyCORIS file's residuals onto `host` (sc-3671). `reconstruct` produces a
/// module's `[out,in]` delta (LoKr or LoHa, lycoris per-module scale baked in) from its parsed
/// factors; keys resolve through the same [`normalize_ltx_key`] map as the peft path (a dotted
/// diffusers third-party file shares it). The delta is installed as a forward residual at the raw
/// per-pass strengths. A kohya-FLATTENED key (no dots) won't normalize to a module and is surfaced as
/// skipped (LTX exposes no module table to un-flatten against — dotted is the real LTX surface).
fn apply_one_thirdparty<G>(
    host: &mut impl LtxAdaptable,
    groups: &BTreeMap<String, G>,
    delta_at: impl Fn(&G, &[i32]) -> Result<Array>,
    spec: &AdapterSpec,
    num_passes: usize,
    report: &mut LtxLoraReport,
) -> Result<()> {
    let strengths = pass_strengths(spec, num_passes)?;
    for (raw, g) in groups {
        let path = normalize_ltx_key(raw);
        let segs: Vec<&str> = path.split('.').collect();
        match host.adaptable_mut(&segs) {
            Some(lin) => {
                let delta = delta_at(g, &lin.base_shape())?;
                lin.push_lokr(delta, strengths.clone());
                report.applied += 1;
            }
            None => report.skipped.push(path),
        }
    }
    Ok(())
}

/// Install every adapter in `specs` onto the LTX transformer, stacking in order (sc-2687 LoRA /
/// sc-2393 LoKr). `num_passes` is the distilled pipeline's denoise-pass count (for validating +
/// expanding `pass_scales`). LoRA (PEFT/kohya) and LoKr (`networkType=lokr`) are dispatched by the
/// file's metadata / the spec kind. Every selected file must match at least one target module;
/// a valid earlier file cannot mask a later zero-match file. Per-key skips are reported, not fatal.
pub fn apply_ltx_adapters(
    host: &mut impl LtxAdaptable,
    specs: &[AdapterSpec],
    num_passes: usize,
) -> Result<LtxLoraReport> {
    let mut report = LtxLoraReport::default();
    for spec in specs {
        let applied_before = report.applied;
        let w = Weights::from_file(&spec.path)?;
        // The file's metadata is authoritative; the spec kind is an additional hint. A spec that
        // declares Lora but whose file says `networkType=lokr` is a caller error (the LoRA loader
        // would find no `lora_A/B` and apply nothing) — route by the file so it is never mis-applied.
        if spec.kind == AdapterKind::Lokr || is_lokr(&w) {
            apply_one_lokr(host, &w, spec, num_passes, &mut report)?;
        } else if is_lokr_keys(&w) {
            // Third-party LyCORIS LoKr (sc-3671): lokr_* keys, no networkType stamp (is_lokr handled
            // above). Reconstruct per-module (bf16 residual, PARITY-BF16) and install like peft LoKr.
            apply_one_thirdparty(
                host,
                &parse_lokr_thirdparty(&w)?,
                |g, bs| g.delta(bs, Dtype::Bfloat16),
                spec,
                num_passes,
                &mut report,
            )?;
        } else if is_loha_keys(&w) {
            // Third-party LyCORIS LoHa (sc-3671).
            apply_one_thirdparty(
                host,
                &parse_loha_thirdparty(&w)?,
                |g, bs| g.delta(bs, Dtype::Bfloat16),
                spec,
                num_passes,
                &mut report,
            )?;
        } else {
            apply_one(host, &w, spec, num_passes, None, &mut report)?;
        }
        require_spec_applied(spec, applied_before, report.applied)?;
    }
    Ok(report)
}

/// Install a split LTX-2.5 LoRA stack.  This deliberately has a stricter contract than
/// [`apply_ltx_adapters`]: every selected file must be a LoRA file, declare `lora_rank` and
/// `lora_alpha`, have the exact factor rank implied by the declaration and projection dimensions,
/// and resolve every factor pair to the loaded DiT. Keeping the policy at the provider seam
/// prevents a valid 2.3 compatibility fallback from weakening the 2.5 route.
pub fn apply_ltx25_adapters(
    host: &mut impl LtxAdaptable,
    specs: &[AdapterSpec],
    num_passes: usize,
) -> Result<LtxLoraReport> {
    let mut report = LtxLoraReport::default();
    for spec in specs {
        if spec.kind != AdapterKind::Lora {
            return Err(Error::Msg(format!(
                "ltx_2_5 adapter {} must be declared LoRA; LoKr is not supported by this route",
                spec.path.display()
            )));
        }
        let w = Weights::from_file(&spec.path)?;
        if is_lokr(&w) || is_lokr_keys(&w) || is_loha_keys(&w) {
            return Err(Error::Msg(format!(
                "ltx_2_5 adapter {} is not a PEFT/Kohya LoRA file",
                spec.path.display()
            )));
        }
        let contract = ltx25_scale(
            w.metadata("lora_rank"),
            w.metadata("lora_alpha"),
            &spec.path,
        )?;
        let applied_before = report.applied;
        let skipped_before = report.skipped.len();
        apply_one(host, &w, spec, num_passes, Some(contract), &mut report)?;
        require_spec_applied(spec, applied_before, report.applied)?;
        if report.skipped.len() != skipped_before {
            return Err(Error::Msg(format!(
                "ltx_2_5 adapter {} contains target(s) that do not resolve on the loaded DiT: {}",
                spec.path.display(),
                report.skipped[skipped_before..].join(", ")
            )));
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ltx25_header_requires_declared_450_rank_and_alpha_without_fallback() {
        let source = std::path::Path::new("synthetic-450.safetensors");
        let scale = ltx25_scale(Some("450"), Some("450"), source).unwrap();
        assert_eq!(scale.rank, 450);
        assert_eq!(scale.alpha, 450.0);

        let missing_rank = ltx25_scale(None, Some("450"), source)
            .unwrap_err()
            .to_string();
        assert!(missing_rank.contains("lora_rank"), "{missing_rank}");
        let missing_alpha = ltx25_scale(Some("450"), None, source)
            .unwrap_err()
            .to_string();
        assert!(missing_alpha.contains("lora_alpha"), "{missing_alpha}");

        let fractional_rank = ltx25_scale(Some("450.5"), Some("450"), source)
            .unwrap_err()
            .to_string();
        assert!(
            fractional_rank.contains("non-positive-integer `lora_rank`"),
            "{fractional_rank}"
        );
    }

    #[test]
    fn ltx25_declared_rank_caps_to_projection_dimensions_exactly() {
        assert_eq!(ltx25_factor_rank(450, 4096, 4096), 450);
        assert_eq!(ltx25_factor_rank(450, 2048, 32), 32);
        assert_eq!(ltx25_factor_rank(450, 256, 2048), 256);
        assert_eq!(ltx25_factor_rank(450, 128, 4096), 128);
    }

    /// sc-13019: the file-derived inventory the `#[ignore]`d multi-surface gates assert against —
    /// synthetic keys so the pairing/normalization/classification logic itself runs in CI.
    #[test]
    fn lora_target_inventory_pairs_normalizes_and_classifies() {
        let mut w = Weights::empty();
        let t = || Array::zeros::<f32>(&[2, 2]).unwrap();
        // PEFT pair on a video attn leaf → one video target.
        w.insert(
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight",
            t(),
        );
        w.insert(
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight",
            t(),
        );
        // kohya + `.default` infix pair, with `to_out.0` normalization collapsing to `to_out`.
        w.insert(
            "diffusion_model.transformer_blocks.1.attn2.to_out.0.lora_down.default.weight",
            t(),
        );
        w.insert(
            "diffusion_model.transformer_blocks.1.attn2.to_out.0.lora_up.default.weight",
            t(),
        );
        // Audio-block pair → audio bucket.
        w.insert(
            "diffusion_model.transformer_blocks.0.audio_attn1.to_v.lora_A.weight",
            t(),
        );
        w.insert(
            "diffusion_model.transformer_blocks.0.audio_attn1.to_v.lora_B.weight",
            t(),
        );
        // Cross-modal (a2v) pair → audio bucket.
        w.insert(
            "diffusion_model.transformer_blocks.0.video_to_audio_attn.to_k.lora_A.weight",
            t(),
        );
        w.insert(
            "diffusion_model.transformer_blocks.0.video_to_audio_attn.to_k.lora_B.weight",
            t(),
        );
        // Orphan down (no up) and a bare `.alpha` — counted by neither bucket.
        w.insert(
            "diffusion_model.transformer_blocks.2.attn1.to_k.lora_A.weight",
            t(),
        );
        w.insert("diffusion_model.transformer_blocks.0.attn1.to_q.alpha", t());
        // Non-lora key — ignored.
        w.insert(
            "diffusion_model.transformer_blocks.0.attn1.to_q.weight",
            t(),
        );

        let (video, audio) = lora_target_inventory(&w);
        assert_eq!(
            video,
            vec![
                "transformer_blocks.0.attn1.to_q".to_string(),
                "transformer_blocks.1.attn2.to_out".to_string(),
            ]
        );
        assert_eq!(
            audio,
            vec![
                "transformer_blocks.0.audio_attn1.to_v".to_string(),
                "transformer_blocks.0.video_to_audio_attn.to_k".to_string(),
            ]
        );
    }

    #[test]
    fn normalize_strips_prefix_and_renames_to_out_and_ff() {
        assert_eq!(
            normalize_ltx_key("diffusion_model.transformer_blocks.0.attn1.to_out.0"),
            "transformer_blocks.0.attn1.to_out"
        );
        assert_eq!(
            normalize_ltx_key("diffusion_model.transformer_blocks.5.ff.net.0.proj"),
            "transformer_blocks.5.ff.proj_in"
        );
        assert_eq!(
            normalize_ltx_key("diffusion_model.transformer_blocks.5.ff.net.2"),
            "transformer_blocks.5.ff.proj_out"
        );
        // gate + q/k/v pass through unchanged (already checkpoint naming).
        assert_eq!(
            normalize_ltx_key("diffusion_model.transformer_blocks.0.attn2.to_gate_logits"),
            "transformer_blocks.0.attn2.to_gate_logits"
        );
        assert_eq!(
            normalize_ltx_key("diffusion_model.adaln_single.linear"),
            "adaln_single.linear"
        );
    }

    #[test]
    fn ic_lora_union_control_keys_map_to_av_blocks() {
        // The production LTX-2.3 IC-LoRA (Lightricks LTX-2.3-22b-IC-LoRA-Union-Control, used for
        // extend_clip / video_bridge / replace_person) ships 960 PEFT bf16 tensors named
        // `diffusion_model.transformer_blocks.N.{attn1,attn2}.to_{q,k,v}.lora_{A,B}.weight`,
        // `...to_out.0...`, and `...ff.net.{0.proj,2}...`. Confirmed against the real file: every key
        // strips to a (suffix, role) and normalizes to an AvDiT video-block module path — so the
        // IC-LoRA loads via the existing `apply_ltx_adapters` seam with no new code (epic 3040).
        let cases = [
            (
                "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight",
                "transformer_blocks.0.attn1.to_q",
            ),
            (
                "diffusion_model.transformer_blocks.0.attn1.to_out.0.lora_B.weight",
                "transformer_blocks.0.attn1.to_out",
            ),
            (
                "diffusion_model.transformer_blocks.27.attn2.to_k.lora_A.weight",
                "transformer_blocks.27.attn2.to_k",
            ),
            (
                "diffusion_model.transformer_blocks.27.ff.net.0.proj.lora_A.weight",
                "transformer_blocks.27.ff.proj_in",
            ),
            (
                "diffusion_model.transformer_blocks.27.ff.net.2.lora_B.weight",
                "transformer_blocks.27.ff.proj_out",
            ),
        ];
        for (key, want) in cases {
            let stem = SUFFIXES
                .iter()
                .find_map(|(suf, _)| key.strip_suffix(suf))
                .unwrap_or_else(|| panic!("no LoRA suffix matched {key}"));
            assert_eq!(normalize_ltx_key(stem), want, "key {key}");
        }
    }

    #[test]
    fn normalize_other_prefixes_and_audio_analogues() {
        assert_eq!(
            normalize_ltx_key("model.diffusion_model.transformer_blocks.0.attn1.to_q"),
            "transformer_blocks.0.attn1.to_q"
        );
        // `.ff.net.*` must NOT fire inside `audio_ff.net.*`; the audio rename handles it separately.
        assert_eq!(
            normalize_ltx_key("diffusion_model.transformer_blocks.0.audio_ff.net.0.proj"),
            "transformer_blocks.0.audio_ff.proj_in"
        );
        assert_eq!(
            normalize_ltx_key("diffusion_model.transformer_blocks.0.audio_ff.net.2"),
            "transformer_blocks.0.audio_ff.proj_out"
        );
    }

    #[test]
    fn pass_scales_uniform_and_per_pass() {
        let mut spec = AdapterSpec::new("x.safetensors".into(), 0.5, AdapterKind::Lora);
        // Uniform: one entry = (alpha/rank)·scale = (16/8)·0.5 = 1.0.
        let u = pass_scales(&spec, 16.0, 8.0, 2).unwrap();
        assert_eq!(u, vec![1.0]);
        // Per-pass: (16/8)·[0.5, 0.25] = [1.0, 0.5].
        spec.pass_scales = Some(vec![0.5, 0.25]);
        assert_eq!(pass_scales(&spec, 16.0, 8.0, 2).unwrap(), vec![1.0, 0.5]);
        // Wrong length errors.
        spec.pass_scales = Some(vec![0.5]);
        assert!(pass_scales(&spec, 16.0, 8.0, 2).is_err());
    }

    #[test]
    fn empty_pass_scales_rejected() {
        // num_passes == 0 + an empty pass_scales vec passes the length check but must still be
        // rejected at load — it would otherwise underflow Linear::forward's per-pass index (F-009).
        let mut spec = AdapterSpec::new("x.safetensors".into(), 0.5, AdapterKind::Lora);
        spec.pass_scales = Some(vec![]);
        assert!(pass_strengths(&spec, 0).is_err());
        // The default (no pass_scales) still yields the single uniform strength.
        let plain = AdapterSpec::new("x.safetensors".into(), 0.5, AdapterKind::Lora);
        assert_eq!(pass_strengths(&plain, 2).unwrap(), vec![0.5]);
    }

    #[test]
    fn non_finite_spec_and_pass_scales_are_rejected() {
        let nan = AdapterSpec::new("nan.safetensors".into(), f32::NAN, AdapterKind::Lora);
        assert!(pass_strengths(&nan, 2).is_err());

        let mut infinite = AdapterSpec::new("inf.safetensors".into(), 1.0, AdapterKind::Lora);
        infinite.pass_scales = Some(vec![1.0, f32::INFINITY]);
        assert!(pass_strengths(&infinite, 2).is_err());

        let overflow = AdapterSpec::new("overflow.safetensors".into(), f32::MAX, AdapterKind::Lora);
        assert!(pass_scales(&overflow, f32::MAX, 1.0, 2).is_err());
    }

    #[test]
    fn valid_adapter_cannot_mask_later_zero_match_spec() {
        let valid = AdapterSpec::new("valid.safetensors".into(), 1.0, AdapterKind::Lora);
        let unmatched = AdapterSpec::new("unmatched.safetensors".into(), 1.0, AdapterKind::Lora);

        require_spec_applied(&valid, 0, 1).unwrap();
        let err = require_spec_applied(&unmatched, 1, 1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unmatched.safetensors"), "{err}");
        assert!(err.contains("every selected adapter must apply"), "{err}");
    }

    /// sc-3671: the LTX crate reconstructs third-party LoKr/LoHa deltas (via the shared core pub
    /// helpers) against the lycoris reference fixtures (`<repo>/tests/fixtures`, generated through
    /// `~/mlx-flux-venv`), and detects them by keys. The residual install (`push_lokr`) is the
    /// existing peft-LoKr path; key resolution reuses `normalize_ltx_key`.
    #[test]
    fn thirdparty_lycoris_reconstructs_against_reference() {
        use mlx_rs::ops::all_close;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        for (dir, stem, is_loha) in [
            ("sc3642_lokr", "linear_bothlr", false),
            ("sc3643_loha", "linear", true),
        ] {
            let base = root.join("tests/fixtures").join(dir);
            let w = Weights::from_file(base.join(format!("{stem}.safetensors"))).unwrap();
            let exp =
                Weights::from_file(base.join(format!("{stem}.expected.safetensors"))).unwrap();
            let want = exp.require("proj").unwrap();
            let got = if is_loha {
                assert!(is_loha_keys(&w), "{stem}: not detected as LoHa");
                let g = parse_loha_thirdparty(&w).unwrap();
                g.values()
                    .next()
                    .unwrap()
                    .delta(want.shape(), Dtype::Float32)
                    .unwrap()
            } else {
                assert!(
                    is_lokr_keys(&w) && !is_lokr(&w),
                    "{stem}: not detected as 3rd-party LoKr"
                );
                let g = parse_lokr_thirdparty(&w).unwrap();
                g.values()
                    .next()
                    .unwrap()
                    .delta(want.shape(), Dtype::Float32)
                    .unwrap()
            };
            assert!(
                all_close(&got, want, 1e-4, 1e-5, false)
                    .unwrap()
                    .item::<bool>(),
                "{stem}: LTX third-party reconstruction diverged from lycoris reference"
            );
        }
    }
}
