//! Anima LoRA/LoKr adapter consumption on the candle lane (sc-10525, the candle twin of MLX sc-10521).
//!
//! **Merge, don't residual** at the safetensors-key level, before the DiT + conditioner are built: the
//! Anima DiT single-file checkpoint holds BOTH the Cosmos DiT (`{prefix}.blocks.*` + globals) AND the
//! bundled `AnimaTextConditioner` (`{prefix}.llm_adapter.blocks.*`) under one root (`prefix` = `net`
//! for the base cut, `model.diffusion_model` for turbo/aesthetic). A trained adapter carries BOTH
//! stacks under ComfyUI's `diffusion_model.` namespace (PEFT `lora_A.weight`/`lora_B.weight`), so after
//! stripping the namespace every target path (`blocks.*` for the DiT, `llm_adapter.blocks.*` for the
//! conditioner) folds into the **same** base key `{prefix}.{path}.weight` — no separate DiT/conditioner
//! routing is needed, because the conditioner already lives under `{prefix}.llm_adapter.` in the base.
//!
//! **The verified trap (sc-10274 class).** `anima-turbo-lora-v0.2` is **508** target pairs (**448** DiT
//! and **60** `llm_adapter.*`); `anima-greg-rutkowski-style` is **448** DiT-only. The merge is
//! **strict**: every target in the file MUST resolve to a base `{prefix}.{path}.weight` key, else it
//! hard-errors naming the unrouted targets — a DiT-only walk that skipped the 60 conditioner targets
//! cannot silently load at partial strength. (For `anima-turbo-lora-v0.2` specifically all 60
//! conditioner `lora_B` are zero-init, so their delta is numerically inert for THIS file — but the
//! guard is about the MECHANISM: a non-zero conditioner LoRA, e.g. the shipped `anima-rl-v0.1`, must
//! not silently fold at partial strength.) No `alpha`/`rank` in the PEFT metadata (`__metadata__ ==
//! {"format":"pt"}`, zero `.alpha` tensors) means α = r means scale 1.0, folded via the same f32 math
//! a candle trainer's forward would use.

use std::collections::{BTreeMap, HashMap};

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::train::lora::{reconstruct_lokr_delta, reconstruct_lora_delta};
use candle_gen::train::merge::{
    merge_into, no_target_matched, read_adapter, read_scalar, AdapterFile, LoraTriple, MergeReport,
    Role,
};
use candle_gen::{CandleError, Result};

/// LoRA-key namespace prefixes an Anima adapter may carry, longest-first so the more specific PEFT form
/// wins (the trained files use `diffusion_model.`; a bare-key candle-trained adapter matches `""`).
const LORA_PREFIXES: [&str; 5] = [
    "base_model.model.diffusion_model.",
    "base_model.model.",
    "diffusion_model.",
    "transformer.",
    "",
];

/// LoKr per-module factor suffixes, longest-first so `.lokr_w1_a` wins over `.lokr_w1`.
const LOKR_SUFFIXES: [&str; 6] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
];

fn strip_lora_prefix(key: &str) -> &str {
    for p in LORA_PREFIXES {
        if let Some(rem) = key.strip_prefix(p) {
            return rem;
        }
    }
    key
}

/// Map one LoRA key to `(module_path, role)` (path is namespace-stripped, e.g. `blocks.0.self_attn.q_proj`
/// or `llm_adapter.blocks.0.cross_attn.k_proj`). Splits on the `.lora_{A,B,down,up}.weight` / `.alpha`
/// suffix so `adaln_modulation_self_attn.1`'s trailing `.1` (a real path segment) survives.
fn classify_lora_key(key: &str) -> Option<(String, Role)> {
    let rem = strip_lora_prefix(key);
    for (suf, role) in [
        (".lora_A.weight", Role::Down),
        (".lora_B.weight", Role::Up),
        (".lora_A.default.weight", Role::Down),
        (".lora_B.default.weight", Role::Up),
        (".lora_down.weight", Role::Down),
        (".lora_up.weight", Role::Up),
        (".alpha", Role::Alpha),
    ] {
        if let Some(path) = rem.strip_suffix(suf) {
            return Some((path.to_string(), role));
        }
    }
    None
}

/// Map one LoKr factor key to `(module_path, factor_name)`.
fn classify_lokr_key(key: &str) -> Option<(String, &'static str)> {
    for suf in LOKR_SUFFIXES {
        if let Some(stem) = key.strip_suffix(suf) {
            return Some((strip_lora_prefix(stem).to_string(), &suf[1..]));
        }
    }
    None
}

/// Fold one LoRA file into `base` (keys `{prefix}.{path}.weight`) at `scale`. Every complete `(A,B)`
/// target is either routed (folded, path pushed to `routed`) or recorded in `unrouted` (its base key is
/// absent) — the caller turns a non-empty `unrouted` into a hard error (strict, no silent partial).
fn merge_lora_file(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    prefix: &str,
    scale: f32,
    report: &mut MergeReport,
    routed: &mut Vec<String>,
    unrouted: &mut Vec<String>,
) -> Result<()> {
    let mut triples: BTreeMap<String, LoraTriple> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match classify_lora_key(key) {
            Some((path, Role::Down)) => triples.entry(path).or_default().down = Some(t.clone()),
            Some((path, Role::Up)) => triples.entry(path).or_default().up = Some(t.clone()),
            Some((path, Role::Alpha)) => {
                triples.entry(path).or_default().alpha = Some(read_scalar(key, t)?)
            }
            None => report.skipped_keys += 1,
        }
    }

    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            report.skipped_keys += 1; // half-pair
            continue;
        };
        if down.dims().len() != 2 || up.dims().len() != 2 {
            report.skipped_keys += 1; // Anima adapts Linears only
            continue;
        }
        let base_key = format!("{prefix}.{path}.weight");
        if !base.contains_key(&base_key) {
            unrouted.push(path);
            continue;
        }
        let rank = down.dims()[0] as f32;
        // No PEFT `.alpha` ⇒ α = rank ⇒ (alpha/rank) = 1 ⇒ scale 1.0 fold.
        let alpha = t.alpha.unwrap_or(rank);
        let delta = reconstruct_lora_delta(&down, &up, alpha, rank, scale)?;
        merge_into(base, &base_key, &delta, report)?;
        routed.push(path);
    }
    Ok(())
}

/// Fold one LoKr file into `base` at `scale` — `rank`/`alpha` from the file's `__metadata__` (default
/// `rank = 1`, `alpha = rank`), per-module factors grouped, `δ = (alpha/rank)·kron(w1,w2)·scale`.
fn merge_lokr_file(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    prefix: &str,
    scale: f32,
    report: &mut MergeReport,
    routed: &mut Vec<String>,
    unrouted: &mut Vec<String>,
) -> Result<()> {
    let rank = af
        .meta
        .get("rank")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);
    let alpha = af
        .meta
        .get("alpha")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(rank);

    let mut grouped: BTreeMap<String, BTreeMap<&'static str, Tensor>> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match classify_lokr_key(key) {
            Some((path, factor)) => {
                grouped.entry(path).or_default().insert(factor, t.clone());
            }
            None => report.skipped_keys += 1,
        }
    }

    for (path, f) in grouped {
        let base_key = format!("{prefix}.{path}.weight");
        let Some(w) = base.get(&base_key) else {
            unrouted.push(path);
            continue;
        };
        if w.dims().len() != 2 {
            report.skipped_keys += 1;
            continue;
        }
        let (out_f, in_f) = (w.dims()[0], w.dims()[1]);
        let delta = reconstruct_lokr_delta(
            f.get("lokr_w1"),
            f.get("lokr_w1_a"),
            f.get("lokr_w1_b"),
            f.get("lokr_w2"),
            f.get("lokr_w2_a"),
            f.get("lokr_w2_b"),
            alpha,
            rank,
            scale,
            (out_f, in_f),
        )?;
        merge_into(base, &base_key, &delta, report)?;
        routed.push(path);
    }
    Ok(())
}

/// Fold every adapter in `specs` into the Anima DiT+conditioner base weight map `base` (CPU), keyed
/// `{prefix}.{path}.weight`, stacked and mixed LoRA/LoKr. **Strict**: a target whose base key is absent
/// (an unrouted `llm_adapter.*` / `blocks.*`) is a hard error naming the unrouted paths — never a silent
/// partial fold (the sc-10274 guard). A spec list that routes zero targets also hard-errors. Returns the
/// [`MergeReport`] (`report.merged` = the routed-target count: 508 for the turbo LoRA, 448 for a
/// DiT-only style LoRA).
pub fn apply_anima_adapters(
    base: &mut HashMap<String, Tensor>,
    prefix: &str,
    specs: &[AdapterSpec],
) -> Result<MergeReport> {
    let mut report = MergeReport::default();
    let mut routed: Vec<String> = Vec::new();
    let mut unrouted: Vec<String> = Vec::new();

    for spec in specs {
        let af = read_adapter(&spec.path)?;
        let is_lokr = matches!(spec.kind, AdapterKind::Lokr) || af.declares_lokr();
        if is_lokr {
            merge_lokr_file(
                base,
                &af,
                prefix,
                spec.scale,
                &mut report,
                &mut routed,
                &mut unrouted,
            )?;
        } else {
            merge_lora_file(
                base,
                &af,
                prefix,
                spec.scale,
                &mut report,
                &mut routed,
                &mut unrouted,
            )?;
        }
    }

    if !unrouted.is_empty() {
        unrouted.sort();
        return Err(CandleError::Msg(format!(
            "anima: {} adapter target(s) did not route to a base module under prefix {prefix:?} \
             (no silent partial fold — sc-10274). First unrouted: {}",
            unrouted.len(),
            unrouted
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    if routed.is_empty() {
        return Err(no_target_matched(
            "anima",
            "expected ComfyUI `diffusion_model.<path>.lora_A/B.weight` (DiT `blocks.*` + conditioner \
             `llm_adapter.blocks.*`)",
            specs.len(),
        ));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device};

    const PFX: &str = "net";

    /// A base map with one target of each of the three routing classes: a DiT attention projection, a
    /// DiT adaLN modulation `.1` (the `.N`-suffix path segment), and a conditioner (`llm_adapter.*`).
    fn base_map(out: usize, inp: usize) -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut m = HashMap::new();
        for path in [
            "blocks.0.self_attn.q_proj",
            "blocks.0.adaln_modulation_self_attn.1",
            "llm_adapter.blocks.0.cross_attn.k_proj",
        ] {
            m.insert(
                format!("{PFX}.{path}.weight"),
                Tensor::zeros((out, inp), DType::F32, &dev).unwrap(),
            );
        }
        m
    }

    fn rand_t(r: usize, c: usize, seed: u64) -> Tensor {
        // Deterministic small values so the fold is exactly reproducible in the assertion.
        let n = r * c;
        let mut s = seed;
        let data: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / u32::MAX as f32) * 0.02 - 0.01
            })
            .collect();
        Tensor::from_vec(data, (r, c), &Device::Cpu).unwrap()
    }

    fn write_lora(
        dir: &std::path::Path,
        paths: &[&str],
        out: usize,
        inp: usize,
        rank: usize,
    ) -> std::path::PathBuf {
        let mut m = HashMap::new();
        for (i, p) in paths.iter().enumerate() {
            m.insert(
                format!("diffusion_model.{p}.lora_A.weight"),
                rand_t(rank, inp, 100 + i as u64),
            );
            m.insert(
                format!("diffusion_model.{p}.lora_B.weight"),
                rand_t(out, rank, 500 + i as u64),
            );
        }
        let path = dir.join("lora.safetensors");
        candle_gen::candle_core::safetensors::save(&m, &path).unwrap();
        path
    }

    /// Injected weight == base + B·A **bit-exact** (f32) for a DiT target, an adaLN target, and a
    /// conditioner target — the weight-level LoRA property, provable with no GPU. Also asserts the
    /// routed count == number of targets (all three classes routed).
    #[test]
    fn lora_fold_is_base_plus_ba_for_all_three_target_classes() {
        let (out, inp, rank) = (8usize, 6usize, 2usize);
        let dir = std::env::temp_dir().join(format!("anima_lora_fold_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let targets = [
            "blocks.0.self_attn.q_proj",
            "blocks.0.adaln_modulation_self_attn.1",
            "llm_adapter.blocks.0.cross_attn.k_proj",
        ];
        let lora_path = write_lora(&dir, &targets, out, inp, rank);

        // Recover the exact A/B written, to compute the expected B@A.
        let af = read_adapter(&lora_path).unwrap();
        let mut base = base_map(out, inp);
        let spec = AdapterSpec::new(lora_path, 1.0, AdapterKind::Lora);
        let report = apply_anima_adapters(&mut base, PFX, std::slice::from_ref(&spec)).unwrap();
        assert_eq!(report.merged, 3, "all three target classes must route");

        for p in targets {
            let a = af
                .tensors
                .get(&format!("diffusion_model.{p}.lora_A.weight"))
                .unwrap();
            let b = af
                .tensors
                .get(&format!("diffusion_model.{p}.lora_B.weight"))
                .unwrap();
            let expected = b.matmul(a).unwrap(); // B[out,r] @ A[r,in] = [out,in], base is zeros
            let got = base.get(&format!("{PFX}.{p}.weight")).unwrap();
            let diff = (got - &expected)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(
                diff < 1e-6,
                "{p}: merged != base + B·A (max abs diff {diff})"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Strict routing: a target whose base key is absent (a DiT-only base missing the `llm_adapter.*`
    /// module) is a HARD ERROR — never a silent partial fold (the sc-10274 mutation guard).
    #[test]
    fn unrouted_target_is_a_hard_error() {
        let (out, inp, rank) = (8usize, 6usize, 2usize);
        let dir = std::env::temp_dir().join(format!("anima_lora_unrouted_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let lora_path = write_lora(
            &dir,
            &[
                "blocks.0.self_attn.q_proj",
                "llm_adapter.blocks.0.cross_attn.k_proj",
            ],
            out,
            inp,
            rank,
        );
        // DiT-only base: drop the conditioner module → the llm_adapter target cannot route.
        let mut base = base_map(out, inp);
        base.remove(&format!(
            "{PFX}.llm_adapter.blocks.0.cross_attn.k_proj.weight"
        ));

        let spec = AdapterSpec::new(lora_path, 1.0, AdapterKind::Lora);
        let err = apply_anima_adapters(&mut base, PFX, std::slice::from_ref(&spec))
            .expect_err("a DiT-only base must reject the llm_adapter target, not silently drop it");
        assert!(
            err.to_string().contains("did not route"),
            "expected an unrouted-target error, got: {err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A synthesized LoKr (no official Anima LoKr exists) loads + applies: `δ = kron(w1, w2)` folded
    /// into the base, and a stacked LoRA+LoKr sums both deltas.
    #[test]
    fn synthesized_lokr_loads_and_stacks_with_lora() {
        let dev = Device::Cpu;
        let dir = std::env::temp_dir().join(format!("anima_lokr_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path_target = "blocks.0.self_attn.q_proj";

        // A full-factor LoKr: w1 [2,3], w2 [4,2] ⇒ kron = [8,6]. No metadata ⇒ rank = alpha = 1 (the
        // merge defaults) ⇒ scale 1.0; the `AdapterKind::Lokr` spec below drives the LoKr dispatch (no
        // `networkType` stamp needed — candle's `save` writes no header metadata).
        let w1 = rand_t(2, 3, 7);
        let w2 = rand_t(4, 2, 9);
        let mut lm = HashMap::new();
        lm.insert(format!("diffusion_model.{path_target}.lokr_w1"), w1.clone());
        lm.insert(format!("diffusion_model.{path_target}.lokr_w2"), w2.clone());
        let lokr_path = dir.join("lokr.safetensors");
        candle_gen::candle_core::safetensors::save(&lm, &lokr_path).unwrap();

        let mut base = HashMap::new();
        base.insert(
            format!("{PFX}.{path_target}.weight"),
            Tensor::zeros((8usize, 6usize), DType::F32, &dev).unwrap(),
        );

        // LoKr alone folds kron(w1, w2).
        let lokr_spec = AdapterSpec::new(lokr_path.clone(), 1.0, AdapterKind::Lokr);
        let r = apply_anima_adapters(&mut base, PFX, std::slice::from_ref(&lokr_spec)).unwrap();
        assert_eq!(r.merged, 1, "the LoKr target must route");
        let after_lokr = base
            .get(&format!("{PFX}.{path_target}.weight"))
            .unwrap()
            .clone();
        let l2 = after_lokr
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            l2 > 0.0,
            "LoKr delta must be non-zero (kron of non-zero factors)"
        );

        // Stacked: fold a LoRA onto the SAME base on top of the LoKr → the two deltas sum.
        let lora_path = write_lora(&dir, &[path_target], 8, 6, 2);
        let lora_spec = AdapterSpec::new(lora_path, 1.0, AdapterKind::Lora);
        let r2 = apply_anima_adapters(&mut base, PFX, std::slice::from_ref(&lora_spec)).unwrap();
        assert_eq!(r2.merged, 1);
        let stacked = base.get(&format!("{PFX}.{path_target}.weight")).unwrap();
        // stacked - after_lokr == the LoRA delta (non-zero) ⇒ additive stacking, not overwrite.
        let added = (stacked - &after_lokr)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            added > 0.0,
            "stacked LoRA delta must add on top of the LoKr, not replace it"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
