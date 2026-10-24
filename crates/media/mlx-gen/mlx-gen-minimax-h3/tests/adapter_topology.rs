//! sc-26688 — adapters and turbo schedules across the full and pruned AdaLN topologies.
//!
//! # What this file is defending
//!
//! The sc-18724/19443 adapter surface was built and verified against the **full** checkpoint — the
//! one with a `time_embedder` timestep MLP. The pruned coordinate-table checkpoint (sc-26684) loads
//! the *same body leaves* through `from_weights_body_only` / `from_weights_curve` and the same
//! `adaptable_mut` arms, so a compatible attention/FFN adapter must fold **identically** on both
//! topologies, and an adapter target that belongs to the full AdaLN surface must be **refused
//! specifically** on a pruned checkpoint — never dropped, never folded onto the curve, never
//! rederived from `adaln_t_table`. The curve runtime stores `U·S` and `W·Vh.T` of the centered,
//! post-SiLU embedding curve; there is no validated full→pruned transformation for an adapter delta,
//! so the only correct outcome for such a target is a named error.
//!
//! Every numeric gate here is a max-abs comparison against an **independently computed** reference
//! (`x·Aᵀ·(fold·Bᵀ)·scale` from the actual file factors) or an exact cross-topology equality —
//! never a shape check, a hash, or a cosine.
//!
//! # Turbo schedules
//!
//! The published turbo recipes are `(num_inference_steps, video_shift)` pairs resolved through
//! `JointSchedule::with_shifts` (sc-18729); the pruned curve consumes the resulting coordinates
//! through `interpolation_position` — `clamp(t, 0, 1)·(grid−1)`, **no ×1000, no SiLU**. The
//! schedule-resolution arm below pins the 768p turbo recipe's four evaluations, the `t = 1 − σ`
//! coordinate convention, and the table lerp the curve runtime actually performs.

use crate::common;

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::ops::{multiply, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::gen_core::pruned_adaln::{interpolation_position, TABLE_KEY};
use mlx_gen::gen_core::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::adapters::convert_comfyui_key_space;
use mlx_gen_minimax_h3::{
    adaln_schedule, adapter_target_paths, alpha_rank_fold, apply_minimax_h3_adapters,
    is_comfyui_key_space, AdaLnSurface, JointSchedule, MiniMaxH3Dit, MiniMaxH3DitConfig,
    AUDIO_SIGMA_SHIFT, VIDEO_SIGMA_SHIFT,
};

use common::{dit_fixture_config, DIT_FIXTURE};

/// The module the numeric arms probe — `[96, 64]` at the fixture geometry.
const PROBE: &str = "transformer_blocks.0.attn.to_q";
/// The SwiGLU input projection, whose ComfyUI spelling (`mlp.fc1`) carries the row-half swap.
const PROBE_FFN: &str = "transformer_blocks.0.ff.net.0.proj";

/// The rank the cross-topology LoRA fixture uses — the *published* 128, so the folds asserted
/// below are the published numbers (1.0) rather than a scaled-down analogue.
const PUBLISHED_RANK: i32 = 128;

/// The fixture's pruned-curve table shape: `[grid, rank]`, F32.
const CURVE_GRID: usize = 8;
const CURVE_RANK: usize = 4;

/// A deterministic host-built tensor — no GPU RNG stream, so every arm sees identical bytes.
fn tensor(shape: &[i32], seed: f32) -> Array {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let v: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.013 + seed).sin()) * 0.25)
        .collect();
    Array::from_slice(&v, shape)
}

fn bf16(a: &Array) -> Array {
    a.as_dtype(Dtype::Bfloat16).expect("cast to bf16")
}

fn f16(a: &Array) -> Array {
    a.as_dtype(Dtype::Float16).expect("cast to f16")
}

fn f32s(a: &Array) -> Vec<f32> {
    a.as_dtype(Dtype::Float32)
        .expect("widen to f32")
        .as_slice::<f32>()
        .to_vec()
}

/// The committed tiny DiT — real crate types, real loader, full timestep-MLP topology.
fn full_dit(cfg: &MiniMaxH3DitConfig) -> MiniMaxH3Dit {
    let mut w = Weights::from_file(DIT_FIXTURE).expect("the committed DiT fixture");
    let dit = MiniMaxH3Dit::from_weights(&mut w, cfg, Dtype::Float32).expect("the whole tiny DiT");
    assert_eq!(
        dit.adapter_surface(),
        AdaLnSurface::FullTimestepMlp,
        "the committed fixture is the full timestep-MLP topology"
    );
    dit
}

/// The committed tiny DiT's tensors transformed into the pruned coordinate-table topology: the
/// full timestep MLP, the full-width per-block AdaLN projections and the final projection are
/// removed and replaced by `adaln_t_table` plus the reduced operands
/// `gen_core::pruned_adaln::inspect` requires — one entry per transformer block plus the final
/// one. **Every body tensor is carried over byte-identically**, which is what makes the
/// cross-topology delta comparison a one-variable experiment.
fn pruned_fixture_map(cfg: &MiniMaxH3DitConfig) -> HashMap<String, Array> {
    let full = Weights::from_file(DIT_FIXTURE).expect("the committed DiT fixture");
    let mut m: HashMap<String, Array> = HashMap::new();
    for key in full.keys().map(str::to_string).collect::<Vec<_>>() {
        let drop = key.starts_with("time_embedder.")
            || key.contains(".adaln_proj.linear.")
            || key.starts_with("norm_out.linear.");
        if !drop {
            let t = full.require(&key).expect("fixture tensor").clone();
            m.insert(key, t);
        }
    }
    m.insert(
        TABLE_KEY.to_string(),
        tensor(&[CURVE_GRID as i32, CURVE_RANK as i32], 0.37),
    );
    for i in 0..cfg.num_layers {
        let prefix = format!("transformer_blocks.{i}.adaln_proj.linear");
        let out = 18 * cfg.hidden_size; // 6 · MODALITY_NUM · hidden_size
        m.insert(
            format!("{prefix}.weight"),
            bf16(&tensor(&[out, CURVE_RANK as i32], 0.41 + i as f32)),
        );
        m.insert(
            format!("{prefix}.bias"),
            f16(&tensor(&[out], 0.43 + i as f32)),
        );
    }
    let final_out = 2 * cfg.hidden_size;
    m.insert(
        "norm_out.linear.weight".to_string(),
        bf16(&tensor(&[final_out, CURVE_RANK as i32], 0.47)),
    );
    m.insert(
        "norm_out.linear.bias".to_string(),
        f16(&tensor(&[final_out], 0.53)),
    );
    m
}

fn pruned_weights(cfg: &MiniMaxH3DitConfig) -> Weights {
    Weights::from_map(pruned_fixture_map(cfg))
}

/// A second body: every checkpoint tensor doubled — a different base, identical geometry. The
/// table is part of the map, so this is a genuinely different pruning basis as well.
fn doubled_pruned_map(cfg: &MiniMaxH3DitConfig) -> HashMap<String, Array> {
    let mut m = pruned_fixture_map(cfg);
    for key in m.keys().cloned().collect::<Vec<_>>() {
        let t = m[&key].clone();
        let two = Array::from_slice(&[2.0f32], &[1]);
        m.insert(key, multiply(&t, &two).expect("double"));
    }
    m
}

fn doubled_pruned_weights(cfg: &MiniMaxH3DitConfig) -> Weights {
    Weights::from_map(doubled_pruned_map(cfg))
}

/// The fixture DiT geometry as a `config.json` document, for staging a checkpoint directory the
/// production `MiniMaxH3Dit::load` path reads.
fn fixture_config_json() -> String {
    r#"{
        "num_attention_heads": 4, "attention_head_dim": 24, "hidden_size": 64,
        "num_layers": 2, "num_refiner_layers": 2, "ffn_dim": 32,
        "in_channels": 24, "audio_in_channels": 32, "patch_size": [1, 2, 2],
        "text_dim": 40, "freq_dim": 16, "time_embed_hidden_dim": 64,
        "time_embed_dim": 48, "rope_freq_dim": 2, "rope_theta": 10000.0,
        "norm_eps": 1e-05, "qk_norm_eps": 1e-05, "final_norm_eps": 1e-05
    }"#
    .to_string()
}

/// Stage one partition directory: `config.json` plus every tensor of `map` in one shard.
fn stage_partition(dir: &Path, partition: &str, map: HashMap<String, Array>) {
    let root = dir.join(partition);
    std::fs::create_dir_all(&root).expect("partition dir");
    std::fs::write(root.join("config.json"), fixture_config_json()).expect("config.json");
    let mut entries: Vec<(String, Array)> = map.into_iter().collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let refs: Vec<(&str, &Array)> = entries.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let no_meta: HashMap<String, String> = HashMap::new();
    Array::save_safetensors(refs, Some(&no_meta), root.join("model.safetensors"))
        .expect("partition shard");
}

fn pruned_dit(cfg: &MiniMaxH3DitConfig) -> MiniMaxH3Dit {
    let mut w = pruned_weights(cfg);
    let dit = MiniMaxH3Dit::from_weights(&mut w, cfg, Dtype::Float32).expect("the pruned tiny DiT");
    assert!(
        dit.pruning_identity().is_some(),
        "the transformed fixture must load as a pruned curve"
    );
    assert_eq!(
        dit.adapter_surface(),
        AdaLnSurface::PrunedCurve,
        "the transformed fixture must present the pruned curve surface"
    );
    dit
}

/// The `[out, in]` logical shape of each of the six adaptable leaves at `cfg`'s geometry.
fn target_shape(cfg: &MiniMaxH3DitConfig, target: &str) -> (i32, i32) {
    match target {
        t if t.ends_with("attn.to_q") || t.ends_with("attn.to_k") || t.ends_with("attn.to_v") => {
            (cfg.inner_dim(), cfg.hidden_size)
        }
        t if t.ends_with("attn.to_out.0") => (cfg.hidden_size, cfg.inner_dim()),
        // SwiGLU input emits `[value | gate]`, so twice `ffn_dim`.
        t if t.ends_with("ff.net.0.proj") => (2 * cfg.ffn_dim, cfg.hidden_size),
        t if t.ends_with("ff.net.2") => (cfg.hidden_size, cfg.ffn_dim),
        other => panic!("unknown target {other}"),
    }
}

/// One adapter target and the `[out, in]` shapes its factors carry.
type FixtureTarget = (String, i32, i32);

/// Every adaptable module at `cfg`'s geometry, with shapes — 312 targets at the fixture geometry.
fn all_targets(cfg: &MiniMaxH3DitConfig) -> Vec<FixtureTarget> {
    adapter_target_paths(cfg)
        .into_iter()
        .map(|t| {
            let (out, inn) = target_shape(cfg, &t);
            (t, out, inn)
        })
        .collect()
}

/// Write a tiny **diffusers-key-space** LoRA over `targets`, bf16 like every published turbo
/// tensor, with `alpha` stamped into the top-level `__metadata__` when given.
fn write_lora(
    dir: &Path,
    name: &str,
    targets: &[FixtureTarget],
    alpha: Option<&str>,
) -> std::path::PathBuf {
    let path = dir.join(name);
    let mut arrays: Vec<(String, Array)> = Vec::new();
    for (target, out, inn) in targets {
        // A distinct seed per module, so a routing bug that folds the right factors onto the wrong
        // module is observable rather than symmetric.
        let seed = target.len() as f32;
        arrays.push((
            format!("{target}.lora_A.default.weight"),
            bf16(&tensor(&[PUBLISHED_RANK, *inn], seed)),
        ));
        arrays.push((
            format!("{target}.lora_B.default.weight"),
            bf16(&tensor(&[*out, PUBLISHED_RANK], seed + 0.5)),
        ));
    }
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".into(), "pt".into());
    meta.insert("floating_dtype".into(), "bfloat16".into());
    if let Some(a) = alpha {
        meta.insert("alpha".into(), a.into());
    }
    let entries: Vec<(&str, &Array)> = arrays.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(entries, Some(&meta), &path).expect("write the tiny LoRA");
    path
}

fn spec(path: std::path::PathBuf, scale: f32) -> AdapterSpec {
    AdapterSpec::new(path, scale, AdapterKind::Lora)
}

fn segs(target: &str) -> Vec<&str> {
    target.split('.').collect()
}

/// `y_with_lora(x) − y_base(x)` at `target` — the residual the install actually added.
fn probe_residual(
    dit: impl Fn() -> MiniMaxH3Dit,
    lora: &Path,
    scale: f32,
    target: &str,
    x: &Array,
) -> Array {
    let s = segs(target);
    let mut base = dit();
    let y0 = base
        .adaptable_mut(&s)
        .unwrap_or_else(|| panic!("probe module {target}"))
        .forward(x)
        .expect("base forward");

    let mut adapted = dit();
    let report = apply_minimax_h3_adapters(&mut adapted, &[spec(lora.to_path_buf(), scale)])
        .expect("install");
    assert!(report.unmatched_paths.is_empty(), "install must be strict");
    let y1 = adapted
        .adaptable_mut(&s)
        .unwrap_or_else(|| panic!("probe module {target}"))
        .forward(x)
        .expect("adapted forward");
    subtract(&y1, &y0).expect("residual")
}

fn max_abs(a: &Array) -> f32 {
    a.abs()
        .unwrap()
        .max(None)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .item::<f32>()
}

/// Round-to-nearest-even f32 → bf16, matching every published turbo tensor's storage dtype and
/// the compute dtype the residual chain runs at.
fn bf16_round(v: f32) -> f32 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = (bits + 0x7fff + lsb) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

/// The stored factor bytes a reference computation reads: `a` is the `[rank, in]` lora_A and `b`
/// the `[out, rank]` lora_B, both widened from the actual file bytes.
struct LoraFactors<'a> {
    a: &'a [f32],
    b: &'a [f32],
    in_dim: usize,
    out: usize,
}

/// The mathematical residual `((x·a)·b_folded)·scale` in f64, over the **stored bf16 factor
/// bytes** (widened exactly) and the exact f32 probe input.
///
/// This is the contract-level reference: the factors are what the file stores (bf16 — every
/// published turbo tensor), the fold is the install's own `alpha/rank`, and the probe input is the
/// exact f32 vector. It deliberately does NOT model where each backend rounds inside the chain:
/// the declared compute dtype is the factors' bf16 (the sc-18724 dtype-matched fold exists so the
/// product runs at the fork's precision rather than being promoted), but the observable contract
/// does not pin whether a backend rounds `x`, an intermediate, or only the output to bf16 — and
/// the two lanes demonstrably place those roundings differently. The gate below derives the bound
/// from that declared bf16 chain instead of encoding one placement.
fn reference_residual(
    factors: LoraFactors<'_>,
    fold: f32,
    scale: f32,
    x: &[f32],
    rows: usize,
) -> Vec<f64> {
    let LoraFactors { a, b, in_dim, out } = factors;
    let rank = a.len() / in_dim;
    assert_eq!(b.len() % rank, 0, "lora_B must be [out, rank]");
    // The install folds `alpha/rank` into `b` with a dtype-matched scalar — one bf16 rounding,
    // identical on both backends because it happens at install, on the host, before any matmul.
    let b_inst: Vec<f64> = b.iter().map(|&v| bf16_round(v * fold) as f64).collect();
    let mut y = vec![0.0f64; rows * out];
    for r in 0..rows {
        let mut mid = vec![0.0f64; rank];
        for (k, mid_k) in mid.iter_mut().enumerate() {
            let acc: f64 = (0..in_dim)
                .map(|j| x[r * in_dim + j] as f64 * a[k * in_dim + j] as f64)
                .sum();
            *mid_k = acc;
        }
        for o in 0..out {
            let acc: f64 = (0..rank).map(|k| mid[k] * b_inst[o * rank + k]).sum();
            y[r * out + o] = acc * scale as f64;
        }
    }
    y
}

/// The factor values the fixture wrote for one target, read back from the written file so the
/// reference uses the actual bytes the install consumed.
fn fixture_factors(
    lora: &Path,
    target: &str,
    rank: i32,
    inn: i32,
    out: i32,
) -> (Vec<f32>, Vec<f32>) {
    let w = Weights::from_file(lora).expect("the written LoRA");
    let a = w
        .require(&format!("{target}.lora_A.default.weight"))
        .expect("lora_A");
    let b = w
        .require(&format!("{target}.lora_B.default.weight"))
        .expect("lora_B");
    assert_eq!(a.shape(), &[rank, inn], "{target} lora_A shape");
    assert_eq!(b.shape(), &[out, rank], "{target} lora_B shape");
    (f32s(a), f32s(b))
}

fn assert_matches_reference(got: &[f32], want: &[f64], what: &str) {
    let mut worst = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        worst = worst.max((*g as f64 - w).abs());
    }
    let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    // Derived bound, not an observed one. The reference is the exact f64 residual over the stored
    // bf16 factor bytes; what it cannot pin is WHERE the declared bf16 chain rounds. Each bf16
    // rounding of an operand/output is ≤ 2⁻⁹ relative and bf16×bf16 products are exact in f32, so
    // a chain that rounds `x`, an intermediate, and the output — the declared bf16 compute dtype's
    // worst placement — errs ≤ 3·2⁻⁹ ≈ 5.9e-3, plus f32 accumulation order (≲ 128·2⁻²⁴). The two
    // lanes demonstrably place the roundings differently (Metal vs CUDA/CPU gemm internals are
    // unpinned by the contract), so the gate is the union bound 1e-2, <2× the derived worst case —
    // NOT a percentage copied from an observed output. A wrong fold (16×/128×), a wrong module or
    // a dropped target moves the residual by orders of magnitude past this band, and the
    // cross-topology bit-exact equality in the same tests is independent of it.
    assert!(
        worst <= 1.0e-2 * peak,
        "{what}: the installed residual must match the independent bf16-chain reference \
         (worst {worst}, peak {peak})"
    );
}

/// **AC1/AC2 (sc-26688):** the same body LoRA folds *bit-identically* on the full and the pruned
/// topology, and each fold matches an independently computed `x·Aᵀ·(fold·Bᵀ)·scale` — the deltas
/// and the alpha fold are preserved across topologies, never reweighted, never rederived.
#[test]
fn body_deltas_are_identical_across_full_and_pruned_topologies() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let lora = write_lora(
        dir.path(),
        "body.safetensors",
        &all_targets(&cfg),
        Some("128"),
    );
    let x = tensor(&[3, cfg.hidden_size], 0.7);

    for target in [PROBE, PROBE_FFN] {
        let (out, inn) = target_shape(&cfg, target);
        let full_res = probe_residual(|| full_dit(&cfg), &lora, 1.0, target, &x);
        let pruned_res = probe_residual(|| pruned_dit(&cfg), &lora, 1.0, target, &x);

        assert!(
            max_abs(&full_res) > 0.0,
            "{target}: the residual must be nonzero or the test proves nothing"
        );
        let diff = subtract(&full_res, &pruned_res).unwrap();
        assert_eq!(
            max_abs(&diff),
            0.0,
            "{target}: the same factors must fold bit-identically on both topologies — the body \
             leaves are the same modules and the install must not reweight across topology"
        );

        let (a, b) = fixture_factors(&lora, target, PUBLISHED_RANK, inn, out);
        let fold = alpha_rank_fold(128.0, PUBLISHED_RANK as f32);
        let want = reference_residual(
            LoraFactors {
                a: &a,
                b: &b,
                in_dim: inn as usize,
                out: out as usize,
            },
            fold,
            1.0,
            &f32s(&x),
            3,
        );
        assert_matches_reference(&f32s(&full_res), &want, target);
    }
}

/// **AC2:** an adapter whose targets belong to the full AdaLN surface is refused **specifically**
/// on a pruned checkpoint — naming the topology and the absent full→pruned transformation — while
/// the same file on the full topology is refused as a deliberate exclusion. Neither surface folds
/// the file, and neither drops the targets.
#[test]
fn full_topology_adaln_targets_are_refused_specifically_per_topology() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let mut targets = all_targets(&cfg);
    // The three full-topology AdaLN stems, with the shapes the full checkpoint would give them.
    targets.push((
        "time_embedder.linear_1".to_string(),
        cfg.time_embed_dim,
        cfg.freq_dim,
    ));
    targets.push((
        "norm_out.linear".to_string(),
        2 * cfg.hidden_size,
        cfg.time_embed_dim,
    ));
    targets.push((
        "transformer_blocks.0.adaln_proj.linear".to_string(),
        18 * cfg.hidden_size,
        cfg.time_embed_dim,
    ));
    let lora = write_lora(
        dir.path(),
        "adaln_targets.safetensors",
        &targets,
        Some("128"),
    );

    let pruned_error = apply_minimax_h3_adapters(&mut pruned_dit(&cfg), &[spec(lora.clone(), 1.0)])
        .expect_err("the pruned curve must refuse full-topology AdaLN targets")
        .to_string();
    // One user-facing marker: the refusal must NAME the topology to be actionable (the story's
    // "specific errors"); the classification itself is the typed contract below.
    assert!(
        pruned_error.contains("pruned coordinate-table"),
        "the refusal must name the topology: {pruned_error}"
    );
    for stem in ["time_embedder", "adaln_proj.linear", "norm_out.linear"] {
        assert!(
            pruned_error.contains(stem),
            "the refusal must list {stem}: {pruned_error}"
        );
    }

    let full_error = apply_minimax_h3_adapters(&mut full_dit(&cfg), &[spec(lora, 1.0)])
        .expect_err("the full topology must refuse its own AdaLN modules too")
        .to_string();
    assert_ne!(
        pruned_error, full_error,
        "the two surfaces must refuse the same file with different classifications"
    );
    assert!(
        !full_error.contains("pruned coordinate-table"),
        "the topology-mismatch wording must not leak onto the full surface: {full_error}"
    );

    // The typed classification contract — asserted directly, never via message wording.
    use mlx_gen_minimax_h3::{classify_unmatched_target, AdaLnSurface, UnmatchedTargetClass};
    for stem in [
        "time_embedder.linear_1",
        "norm_out.linear",
        "transformer_blocks.0.adaln_proj.linear",
    ] {
        assert_eq!(
            classify_unmatched_target(stem, AdaLnSurface::PrunedCurve),
            UnmatchedTargetClass::PrunedCurveTopologyMismatch,
            "{stem} on the pruned curve is a topology mismatch"
        );
        assert_eq!(
            classify_unmatched_target(stem, AdaLnSurface::FullTimestepMlp),
            UnmatchedTargetClass::DeliberatelyExcluded,
            "{stem} on the full path is a real module the surface excludes"
        );
    }
}

/// An adapter targeting a real module no H3 adapter can ever attach to (the bias-Linear I/O heads,
/// the RMS norms) names the exclusion instead of a generic miss — actionable, on either topology.
#[test]
fn real_but_unaddressable_modules_name_the_exclusion() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let mut targets = all_targets(&cfg);
    targets.push((
        "context_embedder".to_string(),
        cfg.hidden_size,
        cfg.text_dim,
    ));
    targets.push((
        "transformer_blocks.0.norm1".to_string(),
        cfg.hidden_size,
        cfg.hidden_size,
    ));
    let lora = write_lora(dir.path(), "heads.safetensors", &targets, Some("128"));

    let error = apply_minimax_h3_adapters(&mut pruned_dit(&cfg), &[spec(lora, 1.0)])
        .expect_err("unaddressable modules must refuse")
        .to_string();
    assert!(
        error.contains("no MiniMax-H3 adapter can attach"),
        "the refusal must name the exclusion: {error}"
    );
    assert!(
        error.contains("context_embedder") && error.contains("norm1"),
        "the refusal must list the offending targets: {error}"
    );
    use mlx_gen_minimax_h3::{classify_unmatched_target, UnmatchedTargetClass};
    for stem in [
        "context_embedder",
        "transformer_blocks.0.norm1",
        "proj_out",
        "norm_out.norm",
    ] {
        assert_eq!(
            classify_unmatched_target(stem, mlx_gen_minimax_h3::AdaLnSurface::PrunedCurve),
            UnmatchedTargetClass::DeliberatelyExcluded,
            "{stem} is a real module no H3 adapter can attach to"
        );
    }
}

/// **AC1:** the published 768p turbo recipe (4 evaluations, video shift 6) resolves its
/// coordinates as `t = 1 − σ` — every one inside `[0, 1]`, so a ×1000 convention would clamp every
/// lookup to the table's final interval — and the pruned curve interpolates `adaln_t_table` at
/// exactly those coordinates, matching the pinned `torch.lerp` addressing elementwise.
#[test]
fn turbo_schedule_coordinates_drive_the_pruned_curve_without_a_thousand_scaling() {
    let cfg = dit_fixture_config();
    let dit = pruned_dit(&cfg);

    let joint = JointSchedule::with_shifts(5, 6.0, AUDIO_SIGMA_SHIFT).expect("turbo schedule");
    assert_eq!(
        joint.num_evals(),
        4,
        "the published 768p turbo recipe evaluates four times (5 requested incl. terminal σ=0)"
    );
    let schedule = adaln_schedule(&joint).expect("adaln schedule");
    let ts = schedule.distinct_timesteps().to_vec();
    assert!(
        !ts.is_empty(),
        "the turbo schedule must carry coordinates for the curve"
    );
    for t in &ts {
        assert!(
            (0.0..=1.0).contains(t),
            "every coordinate must be t = 1 − σ in [0, 1] — got {t}; a ×1000 convention or a SiLU \
             on the coordinate would leave this range or collapse every lookup"
        );
    }
    // The four video evaluation coordinates must be present exactly: t_i = 1 − σ_i.
    let video = joint.video();
    let want: Vec<f32> = video.sigmas()[..video.sigmas().len() - 1]
        .iter()
        .map(|&s| 1.0 - s)
        .collect();
    for w in &want {
        assert!(
            ts.iter().any(|t| (t - w).abs() < 1e-6),
            "the turbo video coordinate {w} must reach the AdaLN schedule"
        );
    }

    // The curve runtime's coordinates match the pinned interpolation elementwise.
    let coords = dit.embed_timesteps(&ts).expect("curve coordinates");
    let table = {
        let w = pruned_weights(&cfg);
        f32s(w.require(TABLE_KEY).expect("the table"))
    };
    let got = f32s(&coords);
    assert_eq!(
        got.len(),
        ts.len() * CURVE_RANK,
        "one curve row per distinct timestep"
    );
    for (i, t) in ts.iter().enumerate() {
        let (lower, fraction) = interpolation_position(*t, CURVE_GRID).expect("pinned addressing");
        for k in 0..CURVE_RANK {
            let start = table[lower * CURVE_RANK + k] as f64;
            let end = table[(lower + 1) * CURVE_RANK + k] as f64;
            let want = start + (fraction as f64) * (end - start);
            let g = got[i * CURVE_RANK + k] as f64;
            assert!(
                (g - want).abs() <= 1e-5 + 1e-5 * want.abs(),
                "coordinate ({t}, {k}): curve {g} vs lerp {want}"
            );
        }
    }
    // The endpoint clamp is the pinned one: t=1 selects the last interval with fraction 1.
    let (lower, fraction) = interpolation_position(1.0, CURVE_GRID).expect("endpoint");
    assert_eq!(lower, CURVE_GRID - 2, "t=1 selects the last interval");
    assert_eq!(fraction, 1.0, "t=1 sits at its end, not rescaled");
}

/// **AC2:** a genuine LoKr file routes to the shared LyCORIS seam on the pruned curve too, and its
/// residual is bit-identical to the full-topology fold — the factored path never materializes a
/// whole-model delta and never depends on the topology.
#[test]
fn a_genuine_lokr_installs_on_the_pruned_curve_with_the_same_delta() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    // kron(w1 [a,c], w2 [b,d]) reshapes to `attn.to_q`'s [96, 64]: a·b = 8·12, c·d = 8·8.
    let arrays = [
        (
            "transformer_blocks.0.attn.to_q.lokr_w1".to_string(),
            bf16(&tensor(&[8, 8], 1.0)),
        ),
        (
            "transformer_blocks.0.attn.to_q.lokr_w2".to_string(),
            bf16(&tensor(&[12, 8], 2.0)),
        ),
    ];
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".into(), "pt".into());
    meta.insert("networkType".into(), "lokr".into());
    meta.insert("rank".into(), "8".into());
    meta.insert("alpha".into(), "8".into());
    let path = dir.path().join("lokr.safetensors");
    let entries: Vec<(&str, &Array)> = arrays.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(entries, Some(&meta), &path).expect("write the LoKr");

    let x = tensor(&[3, cfg.hidden_size], 0.7);
    let full_res = probe_residual(|| full_dit(&cfg), &path, 1.0, PROBE, &x);
    let pruned_res = probe_residual(|| pruned_dit(&cfg), &path, 1.0, PROBE, &x);
    assert!(
        max_abs(&full_res) > 0.0,
        "the LoKr residual must be nonzero or the test proves nothing"
    );
    let diff = subtract(&full_res, &pruned_res).unwrap();
    assert_eq!(
        max_abs(&diff),
        0.0,
        "the LoKr delta must fold bit-identically across topologies"
    );
}

/// **AC1 (namespace):** the ComfyUI `_comfyui_` export shape converts — fused `attn.qkv_proj`
/// splits into three targets with the block-diagonal alpha division — and the converted install
/// folds identically on the pruned curve and against an independent reference. The namespace
/// mapping and the split-offset arithmetic are topology-independent.
#[test]
fn the_comfyui_converted_qkv_namespace_installs_identically_on_the_pruned_curve() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    // Exactly-divisible fused rank 6 = 3·2, so the down factor chunks and the up factor is
    // block-diagonal per q/k/v — the spelling whose alpha the conversion divides by three.
    let fused_rank = 6i32;
    let in_dim = cfg.hidden_size;
    let out_dim = cfg.inner_dim();
    let down = bf16(&tensor(&[fused_rank, in_dim], 0.61));
    // up [3·out, 3r]: slice i is nonzero only on rank columns [i·r, (i+1)·r) — the block-diagonal
    // spelling `is_block_diagonal` detects.
    let r = (fused_rank / 3) as usize;
    let mut up_v = vec![0.0f32; (3 * out_dim * fused_rank) as usize];
    for i in 0..3usize {
        let seed = [0.71f32, 0.73, 0.79][i];
        for row in 0..out_dim as usize {
            for k in 0..r {
                let index = (i * out_dim as usize + row) * fused_rank as usize + i * r + k;
                up_v[index] = ((index as f32 * 0.013 + seed).sin()) * 0.25;
            }
        }
    }
    let up = bf16(&Array::from_slice(&up_v, &[3 * out_dim, fused_rank]));
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".into(), "pt".into());
    meta.insert("floating_dtype".into(), "bfloat16".into());
    meta.insert("alpha".into(), "24".into());
    let arrays = [
        (
            "transformer_blocks.0.attn.qkv_proj.lora_down.weight".to_string(),
            down,
        ),
        (
            "transformer_blocks.0.attn.qkv_proj.lora_up.weight".to_string(),
            up,
        ),
    ];
    let comfy_path = dir.path().join("comfy.safetensors");
    let entries: Vec<(&str, &Array)> = arrays.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(entries, Some(&meta), &comfy_path).expect("write the ComfyUI file");

    let converted = convert_comfyui_key_space(&Weights::from_file(&comfy_path).expect("read back"))
        .expect("the ComfyUI conversion");
    assert!(
        !is_comfyui_key_space(&converted),
        "the conversion must leave no ComfyUI module behind"
    );
    // The split-adjusted alpha: fused 24 over fused rank 6; block-diagonal division emits
    // (24/3)/2 = 4.0 per projection — `alpha/rank` held across the split.
    let alpha = converted
        .require(&format!("{PROBE}.alpha"))
        .expect("the conversion emits a per-target alpha");
    let emitted_alpha = f32s(alpha)[0];
    assert_eq!(
        emitted_alpha, 8.0,
        "the fused 24 must divide by three across the split"
    );
    let converted_path = dir.path().join("converted.safetensors");
    let no_meta: HashMap<String, String> = HashMap::new();
    let entries: Vec<(String, Array)> = converted
        .keys()
        .map(|k| {
            (
                k.to_string(),
                converted.require(k).expect("converted tensor").clone(),
            )
        })
        .collect();
    let entries_ref: Vec<(&str, &Array)> = entries.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(entries_ref, Some(&no_meta), &converted_path).expect("write converted");

    let x = tensor(&[3, cfg.hidden_size], 0.7);
    let full_res = probe_residual(|| full_dit(&cfg), &converted_path, 1.0, PROBE, &x);
    let pruned_res = probe_residual(|| pruned_dit(&cfg), &converted_path, 1.0, PROBE, &x);
    assert!(
        max_abs(&pruned_res) > 0.0,
        "the converted qkv residual must be nonzero"
    );
    let diff = subtract(&full_res, &pruned_res).unwrap();
    assert_eq!(
        max_abs(&diff),
        0.0,
        "the converted namespace must fold bit-identically across topologies"
    );

    // Independent reference over the CONVERTED factor bytes at the emitted alpha's fold.
    let w = Weights::from_file(&converted_path).expect("read converted");
    let a = f32s(
        w.require(&format!("{PROBE}.lora_A.weight"))
            .expect("converted lora_A"),
    );
    let b = f32s(
        w.require(&format!("{PROBE}.lora_B.weight"))
            .expect("converted lora_B"),
    );
    let fold = alpha_rank_fold(emitted_alpha, 2.0);
    let want = reference_residual(
        LoraFactors {
            a: &a,
            b: &b,
            in_dim: in_dim as usize,
            out: out_dim as usize,
        },
        fold,
        1.0,
        &f32s(&x),
        3,
    );
    assert_matches_reference(
        &f32s(&pruned_res),
        &want,
        "converted qkv on the pruned curve",
    );
}

/// The delta is independent of the body it folds onto: the same adapter file on a **different**
/// body (every checkpoint weight doubled — a different base and a different pruning basis, identical
/// geometry) produces the same residual. This is the property that lets one adapter serve every
/// partition/topology of the selected checkpoint without coupling to the base weights — and why no
/// stock-weight substitution can masquerade as an applied adapter.
#[test]
fn the_delta_is_independent_of_the_body_weights_it_folds_onto() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let lora = write_lora(
        dir.path(),
        "body.safetensors",
        &all_targets(&cfg),
        Some("128"),
    );
    let x = tensor(&[3, cfg.hidden_size], 0.7);
    let s = segs(PROBE);

    let residual_on = |weights: Weights, what: &str| -> Array {
        let mut base = {
            let mut w = weights;
            MiniMaxH3Dit::from_weights(&mut w, &cfg, Dtype::Float32)
                .unwrap_or_else(|_| panic!("load {what}"))
        };
        let y0 = base.adaptable_mut(&s).expect("probe").forward(&x).unwrap();
        let mut adapted = {
            let mut w = if what == "pruned body" {
                pruned_weights(&cfg)
            } else {
                doubled_pruned_weights(&cfg)
            };
            MiniMaxH3Dit::from_weights(&mut w, &cfg, Dtype::Float32)
                .unwrap_or_else(|_| panic!("load {what}"))
        };
        apply_minimax_h3_adapters(&mut adapted, &[spec(lora.clone(), 1.0)]).expect("install");
        let y1 = adapted
            .adaptable_mut(&s)
            .expect("probe")
            .forward(&x)
            .unwrap();
        subtract(&y1, &y0).expect("residual")
    };

    let res_a = residual_on(pruned_weights(&cfg), "pruned body");
    let res_b = residual_on(doubled_pruned_weights(&cfg), "doubled body");
    assert!(max_abs(&res_a) > 0.0, "the residual must be nonzero");
    // f32 addition rounds, so adapted−base reproduces the residual to f32 rounding, not bit-exact —
    // but any wrong fold, dropped target or base-weight coupling moves it by orders of magnitude.
    let diff = subtract(&res_a, &res_b).unwrap();
    let worst = max_abs(&diff);
    let peak = max_abs(&res_a) as f64;
    assert!(
        worst as f64 <= 1.0e-4 * peak,
        "the delta must be independent of the body weights it folds onto (worst {worst}, peak {peak})"
    );
}

/// Ignored: this is the native numeric target the CI lane dispatches on an MLX (Metal) runner
/// (sc-26688) — `ci.yml` runs it `--ignored --exact`, so the lane reports it executed, never
/// skipped. It fails when no Metal device is available, because every `Array` here forces a device
/// evaluation through the production apply path.
///
/// What the dispatched arm proves beyond the committed suite: on the lane's own device, the
/// **production apply** (`apply_minimax_h3_adapters` through the real loader) folds a written
/// adapter file onto the pruned curve, the residual matches the independent f64 reference, the
/// same fold is bit-identical across the full and pruned topologies, and every incompatible
/// combination (full-topology AdaLN targets, a LoKr-declared file without factors) is refused
/// **at install — before any forward runs**.
#[test]
#[ignore = "native numeric target dispatched on the MLX CI lane"]
fn native_pruned_turbo_adapter_residual_matches_reference() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let x = tensor(&[3, cfg.hidden_size], 0.7);

    // 1. Production apply + native residual vs the independent reference, both topologies.
    let body = write_lora(
        dir.path(),
        "body.safetensors",
        &all_targets(&cfg),
        Some("128"),
    );
    for target in [PROBE, PROBE_FFN] {
        let (out, inn) = target_shape(&cfg, target);
        let full_res = probe_residual(|| full_dit(&cfg), &body, 1.0, target, &x);
        let pruned_res = probe_residual(|| pruned_dit(&cfg), &body, 1.0, target, &x);
        assert!(max_abs(&pruned_res) > 0.0, "{target}: nonzero residual");
        let diff = subtract(&full_res, &pruned_res).unwrap();
        assert_eq!(
            max_abs(&diff),
            0.0,
            "{target}: bit-identical across topologies"
        );
        let (a, b) = fixture_factors(&body, target, PUBLISHED_RANK, inn, out);
        let fold = alpha_rank_fold(128.0, PUBLISHED_RANK as f32);
        let want = reference_residual(
            LoraFactors {
                a: &a,
                b: &b,
                in_dim: inn as usize,
                out: out as usize,
            },
            fold,
            1.0,
            &f32s(&x),
            3,
        );
        assert_matches_reference(&f32s(&pruned_res), &want, target);
    }

    // 2. Wrong-topology targets refuse BEFORE execution: the install errors without a single
    //    adapted forward having run.
    let mut bad = all_targets(&cfg);
    bad.push((
        "time_embedder.linear_1".to_string(),
        cfg.time_embed_dim,
        cfg.freq_dim,
    ));
    let wrong_topology = write_lora(dir.path(), "wrong_topology.safetensors", &bad, Some("128"));
    let error = apply_minimax_h3_adapters(&mut pruned_dit(&cfg), &[spec(wrong_topology, 1.0)])
        .expect_err("full-topology targets refuse on the pruned curve before execution")
        .to_string();
    assert!(error.contains("pruned coordinate-table"), "{error}");
    assert!(
        error.contains("no validated full→pruned transformation"),
        "{error}"
    );

    // 3. A declared-LoKr file with no LoKr factors refuses by name, still before execution.
    let claimed = dir.path().join("claimed.safetensors");
    let factors = write_lora(
        dir.path(),
        "claimed_factors.safetensors",
        &all_targets(&cfg),
        None,
    );
    let w = Weights::from_file(&factors).expect("factors");
    let arrays: Vec<(String, Array)> = w
        .keys()
        .map(|k| (k.to_string(), w.require(k).expect("tensor").clone()))
        .collect();
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".into(), "pt".into());
    let entries: Vec<(&str, &Array)> = arrays.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(entries, Some(&meta), &claimed).expect("write claimed LoKr");
    let claimed_spec = AdapterSpec::new(claimed, 1.0, AdapterKind::Lokr);
    let error = apply_minimax_h3_adapters(&mut pruned_dit(&cfg), &[claimed_spec])
        .expect_err("a LoKr-declared file without factors refuses")
        .to_string();
    assert!(error.contains("declared LoKr"), "{error}");
}

/// **AC1 (partition selection):** the adapter install is a property of the SELECTED partition, and
/// the same file folds the same delta on every partition of the checkpoint — exercised through the
/// production `MiniMaxH3Dit::load(root, partition, …)` directory path, not `from_weights`, with
/// two genuinely different partitions (`transformer`, `transformer_ref`) staged side by side.
#[test]
fn the_same_adapter_installs_on_every_partition_of_the_selected_checkpoint() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    stage_partition(dir.path(), "transformer", pruned_fixture_map(&cfg));
    // The reference partition is a DIFFERENT checkpoint (different body, different pruning basis)
    // with identical geometry — exactly what the ref2va task resolves as its DiT.
    stage_partition(dir.path(), "transformer_ref", doubled_pruned_map(&cfg));

    let lora = write_lora(
        dir.path(),
        "body.safetensors",
        &all_targets(&cfg),
        Some("128"),
    );
    let x = tensor(&[3, cfg.hidden_size], 0.7);
    let s = segs(PROBE);
    let probe = |partition: &str| -> Array {
        let mut base = MiniMaxH3Dit::load(dir.path(), partition, Dtype::Float32)
            .unwrap_or_else(|_| panic!("load {partition}"));
        assert_eq!(
            base.adapter_surface(),
            AdaLnSurface::PrunedCurve,
            "{partition} must present the pruned curve surface"
        );
        let y0 = base.adaptable_mut(&s).expect("probe").forward(&x).unwrap();
        let mut adapted =
            MiniMaxH3Dit::load(dir.path(), partition, Dtype::Float32).expect("reload");
        let report = apply_minimax_h3_adapters(&mut adapted, &[spec(lora.clone(), 1.0)])
            .unwrap_or_else(|_| panic!("install on {partition}"));
        assert_eq!(
            report.applied,
            adapter_target_paths(&cfg).len(),
            "{partition} coverage"
        );
        assert!(report.unmatched_paths.is_empty(), "{partition}");
        let y1 = adapted
            .adaptable_mut(&s)
            .expect("probe")
            .forward(&x)
            .unwrap();
        subtract(&y1, &y0).expect("residual")
    };

    let res_base = probe("transformer");
    let res_ref = probe("transformer_ref");
    assert!(max_abs(&res_base) > 0.0, "the residual must be nonzero");
    // The delta is a property of the adapter factors, not of the partition it folds onto; f32
    // addition rounding is the only difference (see the body-independence arm's note).
    let diff = subtract(&res_base, &res_ref).unwrap();
    let worst = max_abs(&diff) as f64;
    let peak = max_abs(&res_base) as f64;
    assert!(
        worst <= 1.0e-4 * peak,
        "the same adapter must fold the same delta on both partitions (worst {worst}, peak {peak})"
    );
}

/// **AC1 (trained schedules):** every published turbo recipe the product resolves — the 50-NFE
/// base regime and the four trained turbo variants — produces its documented evaluation count and
/// video shift, keeps the audio shift at its pinned 3.0, and its coordinates stay the `t = 1 − σ`
/// convention the pruned curve consumes.
#[test]
fn every_published_turbo_recipe_resolves_its_trained_schedule() {
    // (name, requested steps, video shift, expected evaluations).
    let recipes: [(&str, usize, f32, usize); 5] = [
        ("base", 50, VIDEO_SIGMA_SHIFT, 50),
        ("turbo_8step", 8, VIDEO_SIGMA_SHIFT, 8),
        ("turbo_4step_768p", 4, 6.0, 4),
        ("turbo_4step_v01", 4, VIDEO_SIGMA_SHIFT, 4),
        ("ref2v_turbo_4step", 4, VIDEO_SIGMA_SHIFT, 4),
    ];
    for (name, requested, video_shift, evals) in recipes {
        let joint = JointSchedule::with_shifts(requested + 1, video_shift, AUDIO_SIGMA_SHIFT)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            joint.num_evals(),
            evals,
            "{name}: N requested steps drive N evaluations (terminal σ=0 is not evaluated)"
        );
        assert_eq!(
            joint.video().shift(),
            video_shift,
            "{name}: trained video shift"
        );
        assert_eq!(
            joint.audio().shift(),
            AUDIO_SIGMA_SHIFT,
            "{name}: audio stays pinned"
        );
        let video = joint.video();
        let sigmas = video.sigmas();
        let timesteps = video.timesteps();
        assert_eq!(timesteps.len(), evals, "{name}");
        for (t, s) in timesteps.iter().zip(sigmas) {
            assert!(
                (0.0..1.0).contains(t),
                "{name}: every coordinate is t = 1 − σ in [0, 1), got {t}"
            );
            assert!(
                (t - (1.0 - s)).abs() < 1e-6,
                "{name}: coordinate {t} must equal 1 − σ ({s}) — no ×1000, no SiLU"
            );
        }
        assert!(
            timesteps.windows(2).all(|w| w[1] > w[0]),
            "{name}: coordinates ascend across the run"
        );
    }
}
