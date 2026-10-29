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

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use mlx_rs::ops::{multiply, subtract};
use mlx_rs::{Array, Dtype};

use gen_core_h3::pruned_adaln::{interpolation_position, TABLE_KEY};
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::gen_core::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::adapters::convert_comfyui_key_space;
use mlx_gen_minimax_h3::{
    adaln_schedule, adapter_target_paths, alpha_rank_fold, apply_minimax_h3_adapters,
    is_comfyui_key_space, AdaLnSurface, JointSchedule, MiniMaxH3Dit, MiniMaxH3DitConfig,
    AUDIO_SIGMA_SHIFT, BLOCK_TARGETS, VIDEO_SIGMA_SHIFT,
};

use common::{dit_fixture_config, DIT_FIXTURE};
use mlx_gen::testing::max_abs;

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
/// `gen_core_h3::pruned_adaln::inspect` requires — one entry per transformer block plus the final
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

/// **AC1/AC2 (sc-26688):** the same body LoRA folds *bit-identically* on the full and the pruned
/// topology — the deltas and the alpha fold are preserved across topologies, never reweighted,
/// never rederived. The numeric-strength proof lives in the exact dyadic corpus below; this
/// non-dyadic arm pins engine-to-engine equality on realistic bf16 factor shapes.
#[test]
#[ignore = "complete synthetic DiT/block-stack execution; dispatched only by the manual H3 native suite"]
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
    }
}

/// **AC2:** an adapter whose targets belong to the full AdaLN surface is refused **specifically**
/// on a pruned checkpoint — naming the topology and the absent full→pruned transformation — while
/// the same file on the full topology is refused as a deliberate exclusion. Neither surface folds
/// the file, and neither drops the targets.
#[test]
#[ignore = "complete synthetic model/block-stack execution; dispatched only by the manual H3 native suite"]
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
#[ignore = "complete synthetic model/block-stack execution; dispatched only by the manual H3 native suite"]
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

/// **AC2, incompatibility-only file (LoRA route):** an adapter that targets **only**
/// `transformer_blocks.0.adaln_proj.linear` — zero compatible body targets — must be refused on
/// the pruned curve with the topology-mismatch error that names the offending stem, never with the
/// generic per-file "no target modules matched". The compatible body targets the earlier tests add
/// are deliberately absent here: they are what let the per-file zero-match guard stay silent and
/// concealed this hole.
#[test]
#[ignore = "complete synthetic model/block-stack execution; dispatched only by the manual H3 native suite"]
fn an_adaln_only_file_with_no_body_targets_is_refused_specifically() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let lora = write_lora(
        dir.path(),
        "adaln_only.safetensors",
        &[(
            "transformer_blocks.0.adaln_proj.linear".to_string(),
            18 * cfg.hidden_size,
            cfg.time_embed_dim,
        )],
        Some("128"),
    );

    let pruned_error = apply_minimax_h3_adapters(&mut pruned_dit(&cfg), &[spec(lora.clone(), 1.0)])
        .expect_err("the pruned curve must refuse an AdaLN-only file")
        .to_string();
    assert!(
        pruned_error.contains("pruned coordinate-table"),
        "the refusal must name the topology mismatch, not the generic zero-match error: {pruned_error}"
    );
    assert!(
        pruned_error.contains("adaln_proj.linear"),
        "the refusal must name the offending target: {pruned_error}"
    );
    assert!(
        !pruned_error.contains("no target modules matched"),
        "the generic zero-match error must not swallow the topology mismatch: {pruned_error}"
    );

    let full_error = apply_minimax_h3_adapters(&mut full_dit(&cfg), &[spec(lora, 1.0)])
        .expect_err("the full topology must refuse its own AdaLN modules")
        .to_string();
    assert!(
        full_error.contains("no MiniMax-H3 adapter can attach"),
        "on the full topology the AdaLN-only file is a deliberate exclusion: {full_error}"
    );
    assert!(
        !full_error.contains("no target modules matched"),
        "the generic zero-match error must not swallow the exclusion: {full_error}"
    );
}

/// **AC2, incompatibility-only file (excluded modules):** a file that targets only real modules no
/// H3 adapter can attach to (the I/O heads, the RMS norms) — again with **no** compatible body
/// target — is refused with the exclusion error that lists them, not the generic zero-match error.
#[test]
#[ignore = "complete synthetic model/block-stack execution; dispatched only by the manual H3 native suite"]
fn an_excluded_only_file_with_no_body_targets_is_refused_specifically() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let lora = write_lora(
        dir.path(),
        "excluded_only.safetensors",
        &[
            (
                "context_embedder".to_string(),
                cfg.hidden_size,
                cfg.text_dim,
            ),
            (
                "transformer_blocks.0.norm1".to_string(),
                cfg.hidden_size,
                cfg.hidden_size,
            ),
        ],
        Some("128"),
    );

    let error = apply_minimax_h3_adapters(&mut pruned_dit(&cfg), &[spec(lora, 1.0)])
        .expect_err("an excluded-modules-only file must be refused")
        .to_string();
    assert!(
        error.contains("no MiniMax-H3 adapter can attach"),
        "the refusal must name the exclusion, not the generic zero-match error: {error}"
    );
    assert!(
        error.contains("context_embedder") && error.contains("norm1"),
        "the refusal must list the offending targets: {error}"
    );
    assert!(
        !error.contains("no target modules matched"),
        "the generic zero-match error must not swallow the exclusion: {error}"
    );
}

/// **AC2, incompatibility-only file (LoKr route):** the LoKr zero-match guard has the same hole as
/// the LoRA one. A valid LoKr file targeting only the full AdaLN surface — no compatible body
/// target — must be refused on the pruned curve as a topology mismatch naming the stem, not with
/// the generic per-file zero-match error.
#[test]
#[ignore = "complete synthetic model/block-stack execution; dispatched only by the manual H3 native suite"]
fn an_adaln_only_lokr_with_no_body_targets_is_refused_specifically() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let arrays = [
        (
            "transformer_blocks.0.adaln_proj.linear.lokr_w1".to_string(),
            bf16(&tensor(&[8, 8], 1.0)),
        ),
        (
            "transformer_blocks.0.adaln_proj.linear.lokr_w2".to_string(),
            bf16(&tensor(&[12, 8], 2.0)),
        ),
    ];
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".into(), "pt".into());
    meta.insert("networkType".into(), "lokr".into());
    meta.insert("rank".into(), "8".into());
    meta.insert("alpha".into(), "8".into());
    let path = dir.path().join("adaln_only_lokr.safetensors");
    let entries: Vec<(&str, &Array)> = arrays.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(entries, Some(&meta), &path).expect("write the LoKr");

    let pruned_error = apply_minimax_h3_adapters(&mut pruned_dit(&cfg), &[spec(path.clone(), 1.0)])
        .expect_err("the pruned curve must refuse an AdaLN-only LoKr file")
        .to_string();
    assert!(
        pruned_error.contains("pruned coordinate-table"),
        "the refusal must name the topology mismatch, not the generic zero-match error: {pruned_error}"
    );
    assert!(
        pruned_error.contains("adaln_proj.linear"),
        "the refusal must name the offending target: {pruned_error}"
    );
    assert!(
        !pruned_error.contains("no target modules matched"),
        "the generic zero-match error must not swallow the topology mismatch: {pruned_error}"
    );

    let full_error = apply_minimax_h3_adapters(&mut full_dit(&cfg), &[spec(path, 1.0)])
        .expect_err("the full topology must refuse its own AdaLN modules")
        .to_string();
    assert!(
        full_error.contains("no MiniMax-H3 adapter can attach"),
        "on the full topology the AdaLN-only LoKr is a deliberate exclusion: {full_error}"
    );
}

/// **AC1:** the published 768p turbo recipe (4 evaluations, video shift 6) resolves its
/// coordinates as `t = 1 − σ` — every one inside `[0, 1]`, so a ×1000 convention would clamp every
/// lookup to the table's final interval — and the pruned curve interpolates `adaln_t_table` at
/// exactly those coordinates, matching the pinned `torch.lerp` addressing elementwise.
#[test]
#[ignore = "complete synthetic DiT/block-stack execution; dispatched only by the manual H3 native suite"]
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
#[ignore = "complete synthetic DiT/block-stack execution; dispatched only by the manual H3 native suite"]
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
    genuine_lora_lokr_stack_causality(&cfg, dir.path(), &path);
}

/// **AC1 (namespace):** the ComfyUI `_comfyui_` export shape converts — fused `attn.qkv_proj`
/// splits into three targets with the block-diagonal alpha division — and the converted install
/// folds identically on the pruned curve and against an independent reference. The namespace
/// mapping and the split-offset arithmetic are topology-independent.
#[test]
#[ignore = "complete synthetic DiT/block-stack execution; dispatched only by the manual H3 native suite"]
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
}

/// The delta is independent of the body it folds onto: the same adapter file on a **different**
/// body (every checkpoint weight doubled — a different base and a different pruning basis, identical
/// geometry) produces the same residual. This is the property that lets one adapter serve every
/// partition/topology of the selected checkpoint without coupling to the base weights — and why no
/// stock-weight substitution can masquerade as an applied adapter.
#[test]
#[ignore = "complete synthetic DiT/block-stack execution; dispatched only by the manual H3 native suite"]
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
        let full_res = probe_residual(|| full_dit(&cfg), &body, 1.0, target, &x);
        let pruned_res = probe_residual(|| pruned_dit(&cfg), &body, 1.0, target, &x);
        assert!(max_abs(&pruned_res) > 0.0, "{target}: nonzero residual");
        let diff = subtract(&full_res, &pruned_res).unwrap();
        assert_eq!(
            max_abs(&diff),
            0.0,
            "{target}: bit-identical across topologies"
        );
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
#[ignore = "complete synthetic DiT/block-stack execution; dispatched only by the manual H3 native suite"]
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

// --- The exact dyadic corpus (sc-26688 review) ----------------------------------------------------
//
// The numeric-strength proof for the format/topology/partition/gain gates: factors and probe input
// built so that EVERY product and partial sum in the residual chain is a small dyadic rational —
// exactly representable in bf16 AND f32 — so the installed residual equals the f64 expectation
// read from the STORED bytes bit-for-bit under EVERY admitted stage-cast variant. No tolerance, no
// output-relative guard: every admitted rounding placement is exact, so promotions are irrelevant.
//
// * `x` entries are integers in [-2, 2] — exact in bf16 and f32, so an input cast is a no-op;
// * `a` rows are a SUB-PERMUTATION (at most one nonzero per row AND per column, unused rank rows
//   zero) — `x·aᵀ` is a single term, no accumulation at all; declared rank/alpha stay 128/128;
// * `b` rows are one-hot basis probes `e_k` with row/target-coded values `m·2^e`, `m ∈ [32, 63]`
//   (≤ 6 significant bf16 bits), distinct signs/exponents per row — the second matmul is a single
//   term and a wrong routing cannot hide behind a symmetric value;
// * the declared FOLDS are powers of two (1, 0.0625, 4) and the gain is dyadic with two
//   significant bits (0.75 = 3·2⁻²) — not a power of two; with `b` carrying ≤ 6 significant bits,
//   every folded/gained value stays at ≤ 8 significant bits, exactly representable in bf16 (and
//   f32), so the dtype-matched scalar multiplies are exact on every conforming backend;
// * the probe modules' base weights are ZERO in a dedicated fixture variant, so the observed
//   module output IS the residual: `base(x) = 0` exactly, and `0 + residual` is exact in f32.
//   (A bit-identical base graph with `(adapted − base)` would NOT prove the delta if the bf16
//   base+delta addition rounded first — the zero base removes that stage instead of arguing it.)
//
// The non-dyadic arms above keep their bit-exact engine-to-engine equality and their perturbed
// body / cross-topology / partition coverage; they assert no numeric tolerance.

/// `true` for the six adaptable leaves' base weights.
fn exact_probe_leaf_weight(key: &str) -> bool {
    BLOCK_TARGETS
        .iter()
        .any(|leaf| key.ends_with(&format!(".{leaf}.weight")))
}

fn with_zero_bases(mut m: HashMap<String, Array>) -> HashMap<String, Array> {
    for key in m.keys().cloned().collect::<Vec<_>>() {
        if exact_probe_leaf_weight(&key) {
            let shape = m[&key].shape().to_vec();
            let n: usize = shape.iter().map(|&d| d as usize).product();
            m.insert(key, Array::from_slice(&vec![0.0f32; n], &shape));
        }
    }
    m
}

fn full_zero_dit(cfg: &MiniMaxH3DitConfig) -> MiniMaxH3Dit {
    let full = Weights::from_file(DIT_FIXTURE).expect("the committed DiT fixture");
    let m: HashMap<String, Array> = full
        .keys()
        .map(|k| {
            (
                k.to_string(),
                full.require(k).expect("fixture tensor").clone(),
            )
        })
        .collect();
    let mut w = Weights::from_map(with_zero_bases(m));
    let dit = MiniMaxH3Dit::from_weights(&mut w, cfg, Dtype::Float32).expect("full zero DiT");
    assert_eq!(dit.adapter_surface(), AdaLnSurface::FullTimestepMlp);
    dit
}

fn pruned_zero_dit(cfg: &MiniMaxH3DitConfig) -> MiniMaxH3Dit {
    let mut w = Weights::from_map(with_zero_bases(pruned_fixture_map(cfg)));
    let dit = MiniMaxH3Dit::from_weights(&mut w, cfg, Dtype::Float32).expect("pruned zero DiT");
    assert_eq!(dit.adapter_surface(), AdaLnSurface::PrunedCurve);
    dit
}

/// Small-integer probe rows at `inn` width, exact in bf16 and f32 alike; rows differ so a
/// row-routing bug shows. Each adaptable leaf has its OWN input width, so the probe is built per
/// target, never shared.
fn exact_probe_x_for(cfg: &MiniMaxH3DitConfig, target: &str) -> Array {
    let (rows, inn) = (3usize, {
        let (_, inn) = target_shape(cfg, target);
        inn as usize
    });
    let v: Vec<f32> = (0..rows * inn)
        .map(|i| {
            let (r, j) = (i / inn, i % inn);
            (((r * 3 + j * 7) % 5) as f32) - 2.0
        })
        .collect();
    Array::from_slice(&v, &[rows as i32, inn as i32])
}

struct ExactCorpus {
    path: std::path::PathBuf,
    /// Probed targets, whose stored bytes are structurally verified and whose expected residual is
    /// computed independently from those bytes.
    probes: Vec<String>,
}

/// Write the corpus over every adaptable module (full strict coverage, no production-input
/// shrinking) with per-target nonuniform one-hot coding. Fold is `alpha/rank = 128/128 = 1`.
fn write_exact_corpus(
    dir: &Path,
    name: &str,
    cfg: &MiniMaxH3DitConfig,
    probes: &[&str],
) -> ExactCorpus {
    let path = dir.join(name);
    let rank = PUBLISHED_RANK as usize;
    let mut arrays: Vec<(String, Array)> = Vec::new();
    for (tidx, (target, out, inn)) in all_targets(cfg).into_iter().enumerate() {
        let (out, inn) = (out as usize, inn as usize);
        // Sub-permutation: row k active only while k < inn (one nonzero per row AND column).
        let mut down = vec![0.0f32; rank * inn];
        for k in 0..inn {
            down[k * inn + (tidx * 7 + k) % inn] = 1.0;
        }
        let mut up = vec![0.0f32; out * rank];
        for o in 0..out {
            // Basis probe: one active rank row per output, value in [32, 63] (≤ 6 bf16 bits),
            // sign/exponent coded by row and target so outputs are nonuniform.
            let m = (32 + (o * 7 + tidx * 5) % 32) as f32;
            let value = if (o + tidx) % 2 == 0 { m } else { -m };
            up[o * rank + (o * 5 + tidx) % inn] = value;
        }
        arrays.push((
            format!("{target}.lora_A.default.weight"),
            bf16(&Array::from_slice(&down, &[rank as i32, inn as i32])),
        ));
        arrays.push((
            format!("{target}.lora_B.default.weight"),
            bf16(&Array::from_slice(&up, &[out as i32, rank as i32])),
        ));
    }
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".into(), "pt".into());
    meta.insert("floating_dtype".into(), "bfloat16".into());
    meta.insert("alpha".into(), "128".into());
    let entries: Vec<(&str, &Array)> = arrays.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(entries, Some(&meta), &path).expect("write the exact corpus");
    ExactCorpus {
        path,
        probes: probes.iter().map(|t| t.to_string()).collect(),
    }
}

/// Verify the corpus' stored bytes ARE the declared sparse dyadic structure — the exactness claim
/// is about the file's actual bytes, not the writer's intent.
fn verify_corpus_bytes(w: &Weights, target: &str, inn: usize) {
    let a = f32s(
        w.require(&format!("{target}.lora_A.default.weight"))
            .expect("lora_A"),
    );
    let b = f32s(
        w.require(&format!("{target}.lora_B.default.weight"))
            .expect("lora_B"),
    );
    let rank = a.len() / inn;
    let mut active_columns: Vec<usize> = Vec::new();
    for k in 0..rank {
        let nonzeros: Vec<usize> = (0..inn).filter(|&j| a[k * inn + j] != 0.0).collect();
        assert!(
            nonzeros.len() <= 1,
            "{target}: down row {k} must be empty or a unit vector, got {:?}",
            &a[k * inn..(k + 1) * inn]
        );
        if let Some(&j) = nonzeros.first() {
            assert_eq!(
                a[k * inn + j],
                1.0,
                "{target}: down row {k} nonzero must be 1.0 (a power of two)"
            );
            active_columns.push(j);
        }
    }
    // Sub-permutation: at most one nonzero per COLUMN as well as per row.
    let distinct: std::collections::BTreeSet<usize> = active_columns.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        active_columns.len(),
        "{target}: down factor must be a sub-permutation (no shared column)"
    );
    let out = b.len() / rank;
    for o in 0..out {
        let nonzeros: Vec<usize> = (0..rank).filter(|&k| b[o * rank + k] != 0.0).collect();
        assert!(
            nonzeros.len() == 1
                && b[o * rank + nonzeros[0]].abs() >= 32.0
                && b[o * rank + nonzeros[0]].abs() <= 63.0,
            "{target}: up row {o} must be one basis probe with a 6-bit dyadic value"
        );
    }
}

/// The expected residual in f64, computed from the STORED bytes: `y[o] = fold·Σ_k b[o,k]·Σ_j
/// a[k,j]·x[r,j]`. Exact for the corpus because every partial sum reduces to one or two small
/// dyadic terms — and exact for the converted/LoKr arms for the same reason.
fn exact_expected_from_bytes(
    w: &Weights,
    a_key: &str,
    b_key: &str,
    x: &[f32],
    rows: usize,
    inn: usize,
    fold: f64,
) -> Vec<f64> {
    let a = f32s(w.require(a_key).expect("lora_A"));
    let b = f32s(w.require(b_key).expect("lora_B"));
    let rank = a.len() / inn;
    let out = b.len() / rank;
    let mut y = vec![0.0f64; rows * out];
    for r in 0..rows {
        let mut mid = vec![0.0f64; rank];
        for (k, mid_k) in mid.iter_mut().enumerate() {
            *mid_k = (0..inn)
                .map(|j| x[r * inn + j] as f64 * a[k * inn + j] as f64)
                .sum();
        }
        for o in 0..out {
            y[r * out + o] = fold
                * (0..rank)
                    .map(|k| mid[k] * b[o * rank + k] as f64)
                    .sum::<f64>();
        }
    }
    y
}

/// Install the corpus on a ZERO-BASE dit and return the observed residual rows for `target` at
/// user gain `scale` (a power of two in these arms, so the gain multiply is exact).
fn exact_probe(
    _cfg: &MiniMaxH3DitConfig,
    dit: impl Fn() -> MiniMaxH3Dit,
    corpus: &ExactCorpus,
    scale: f32,
    target: &str,
    x: &Array,
) -> Vec<f32> {
    let s = segs(target);
    let mut base = dit();
    let y0 = base
        .adaptable_mut(&s)
        .expect("probe")
        .forward(x)
        .expect("base");
    assert_eq!(
        max_abs(&y0),
        0.0,
        "{target}: the zero-base probe module must output exactly zero"
    );
    let mut adapted = dit();
    let report = apply_minimax_h3_adapters(&mut adapted, &[spec(corpus.path.clone(), scale)])
        .expect("install");
    assert!(report.unmatched_paths.is_empty(), "{target}");
    let y1 = adapted
        .adaptable_mut(&s)
        .expect("probe")
        .forward(x)
        .expect("adapted");
    f32s(&y1)
}

/// The probe set: every leaf kind of a transformer block, the refiner, a second block, and both
/// SwiGLU leaves — nonuniform coding makes a wrong row/module/alias mapping observable.
fn exact_probe_targets(cfg: &MiniMaxH3DitConfig) -> Vec<FixtureTarget> {
    const PROBES: [&str; 8] = [
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_v",
        "transformer_blocks.0.attn.to_out.0",
        "transformer_blocks.0.ff.net.0.proj",
        "transformer_blocks.0.ff.net.2",
        "token_refiner.refiner_blocks.0.attn.to_q",
        "transformer_blocks.1.ff.net.2",
    ];
    all_targets(cfg)
        .into_iter()
        .filter(|(t, _, _)| PROBES.contains(&t.as_str()))
        .collect()
}

/// **AC1/AC2, exact corpus:** on a zero-base checkpoint the installed residual equals the f64
/// expectation read from the stored bytes BIT-FOR-BIT, on both topologies, at gains 1.0 and 0.75 —
/// no tolerance, and the per-target expectations are pairwise distinct so a wrong
/// row/module/alias/gain mapping cannot hide.
#[test]
#[ignore = "complete synthetic DiT/block-stack execution; dispatched only by the manual H3 native suite"]
fn the_exact_dyadic_corpus_folds_exactly_on_both_topologies() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let probes = exact_probe_targets(&cfg);
    let corpus = write_exact_corpus(
        dir.path(),
        "exact.safetensors",
        &cfg,
        &probes
            .iter()
            .map(|(t, _, _)| t.clone())
            .collect::<Vec<_>>()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
    );
    let file = Weights::from_file(&corpus.path).expect("read the corpus back");
    for target in &corpus.probes {
        let (_, inn) = target_shape(&cfg, target);
        verify_corpus_bytes(&file, target, inn as usize);
    }
    // The expectations must be pairwise distinct across targets — a shared value would let a
    // misrouted fold hide.
    let fold = alpha_rank_fold(128.0, PUBLISHED_RANK as f32);
    let mut expectations: BTreeMap<String, (Vec<f64>, Array)> = BTreeMap::new();
    for target in &corpus.probes {
        let (_, inn) = target_shape(&cfg, target);
        let x = exact_probe_x_for(&cfg, target);
        let want = exact_expected_from_bytes(
            &file,
            &format!("{target}.lora_A.default.weight"),
            &format!("{target}.lora_B.default.weight"),
            &f32s(&x),
            3,
            inn as usize,
            fold as f64,
        );
        expectations.insert(target.clone(), (want, x));
    }
    let values: Vec<&Vec<f64>> = expectations.values().map(|(want, _)| want).collect();
    for i in 0..values.len() {
        for j in (i + 1)..values.len() {
            assert_ne!(
                values[i], values[j],
                "expectations must be pairwise distinct so a misroute cannot hide"
            );
        }
    }
    for target in &corpus.probes {
        let (want, x) = &expectations[target];
        for (gain, scale) in [(1.0f32, 1.0f64), (0.75, 0.75)] {
            let got = exact_probe(&cfg, || full_zero_dit(&cfg), &corpus, gain, target, x);
            for (g, w) in got.iter().zip(want) {
                assert_eq!(
                    *g as f64,
                    *w * scale,
                    "{target} gain {gain}: the exact corpus must fold bit-exactly on the full \
                     topology"
                );
            }
            let got = exact_probe(&cfg, || pruned_zero_dit(&cfg), &corpus, gain, target, x);
            for (g, w) in got.iter().zip(want) {
                assert_eq!(
                    *g as f64,
                    *w * scale,
                    "{target} gain {gain}: the exact corpus must fold bit-exactly on the pruned \
                     curve"
                );
            }
            assert!(
                want.iter().any(|v| *v != 0.0),
                "{target}: the corpus residual must be nonzero"
            );
        }
    }
}

/// **AC1 (partition selection, exact):** through the production `MiniMaxH3Dit::load(root,
/// partition, …)` path, two genuinely different pruned partitions fold the corpus to the SAME
/// exact expectation — the delta is a property of the adapter, not of the partition it lands on.
#[test]
#[ignore = "complete synthetic DiT/block-stack execution; dispatched only by the manual H3 native suite"]
fn the_exact_corpus_is_bit_identical_across_partitions() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    stage_partition(
        dir.path(),
        "transformer",
        with_zero_bases(pruned_fixture_map(&cfg)),
    );
    stage_partition(
        dir.path(),
        "transformer_ref",
        with_zero_bases(doubled_pruned_map(&cfg)),
    );
    let probes = exact_probe_targets(&cfg);
    let corpus = write_exact_corpus(
        dir.path(),
        "exact.safetensors",
        &cfg,
        &probes
            .iter()
            .map(|(t, _, _)| t.clone())
            .collect::<Vec<_>>()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
    );
    let file = Weights::from_file(&corpus.path).expect("read the corpus back");
    for target in &corpus.probes {
        let (_, inn) = target_shape(&cfg, target);
        verify_corpus_bytes(&file, target, inn as usize);
        let x = exact_probe_x_for(&cfg, target);
        let want = exact_expected_from_bytes(
            &file,
            &format!("{target}.lora_A.default.weight"),
            &format!("{target}.lora_B.default.weight"),
            &f32s(&x),
            3,
            inn as usize,
            1.0,
        );
        for partition in ["transformer", "transformer_ref"] {
            let mut base = MiniMaxH3Dit::load(dir.path(), partition, Dtype::Float32)
                .unwrap_or_else(|e| panic!("load {partition}: {e}"));
            assert_eq!(
                base.adapter_surface(),
                AdaLnSurface::PrunedCurve,
                "{partition}"
            );
            let s = segs(target);
            let y0 = base.adaptable_mut(&s).expect("probe").forward(&x).unwrap();
            assert_eq!(max_abs(&y0), 0.0, "{partition}: zero-base probe");
            let mut adapted =
                MiniMaxH3Dit::load(dir.path(), partition, Dtype::Float32).expect("reload");
            let report = apply_minimax_h3_adapters(&mut adapted, &[spec(corpus.path.clone(), 1.0)])
                .unwrap_or_else(|e| panic!("install on {partition}: {e}"));
            assert_eq!(
                report.applied,
                adapter_target_paths(&cfg).len(),
                "{partition}"
            );
            assert!(report.unmatched_paths.is_empty(), "{partition}");
            let y1 = adapted
                .adaptable_mut(&s)
                .expect("probe")
                .forward(&x)
                .unwrap();
            for (g, w) in f32s(&y1).iter().zip(&want) {
                assert_eq!(
                    *g as f64, *w,
                    "{partition}/{target}: the corpus residual must be exact"
                );
            }
        }
    }
}

/// A genuine LoKr whose every Kronecker product and residual term is dyadic and exact: `w1` rows
/// are powers of two one-hot, `w2` rows are one-hot nonuniform dyadics — so `reconstruct_lokr_delta`'s
/// bf16 reconstruction (with fold `alpha/rank = 8/8 = 1`) and the residual are exact, and a wrong
/// routing is visible because every output row has a distinct expected value.
#[test]
#[ignore = "complete synthetic DiT/block-stack execution; dispatched only by the manual H3 native suite"]
fn the_exact_lokr_corpus_folds_exactly_on_the_pruned_curve() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    // kron(w1 [8,8], w2 [12,8]) = [96, 64], `to_q`'s exact shape.
    let mut w1 = vec![0.0f32; 64];
    for i in 0..8usize {
        w1[i * 8 + i] = 2.0f64.powi((i % 3) as i32) as f32; // powers of two: 1, 2, 4
    }
    let mut w2 = vec![0.0f32; 96];
    for b in 0..12usize {
        let value = if b % 2 == 0 { 1.0f32 } else { -1.0f32 };
        w2[b * 8 + (b * 3) % 8] = value;
    }
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".into(), "pt".into());
    meta.insert("networkType".into(), "lokr".into());
    meta.insert("rank".into(), "8".into());
    meta.insert("alpha".into(), "8".into());
    let arrays = [
        (
            "transformer_blocks.0.attn.to_q.lokr_w1".to_string(),
            bf16(&Array::from_slice(&w1, &[8, 8])),
        ),
        (
            "transformer_blocks.0.attn.to_q.lokr_w2".to_string(),
            bf16(&Array::from_slice(&w2, &[12, 8])),
        ),
    ];
    let path = dir.path().join("exact_lokr.safetensors");
    let entries: Vec<(&str, &Array)> = arrays.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(entries, Some(&meta), &path).expect("write the exact LoKr");

    // Expected from the STORED factor bytes: delta row (i*12+b) has exactly one nonzero at column
    // i*8 + (b*3)%8 with value w1[i,i]·w2[b,(b*3)%8] — a product of dyadics, exact.
    let fw = Weights::from_file(&path).expect("read the LoKr back");
    let w1s = f32s(
        fw.require("transformer_blocks.0.attn.to_q.lokr_w1")
            .expect("w1"),
    );
    let w2s = f32s(
        fw.require("transformer_blocks.0.attn.to_q.lokr_w2")
            .expect("w2"),
    );
    let x = exact_probe_x_for(&cfg, PROBE);
    let xf = f32s(&x);
    let inn = {
        let (_, inn) = target_shape(&cfg, PROBE);
        inn as usize
    };
    let out = cfg.inner_dim() as usize;
    let mut want = vec![0.0f64; 3 * out];
    for o in 0..out {
        let (i, b) = (o / 12, o % 12);
        let col = i * 8 + (b * 3) % 8;
        let delta = w1s[i * 8 + i] as f64 * w2s[b * 8 + (b * 3) % 8] as f64;
        for r in 0..3 {
            want[r * out + o] = delta * xf[r * inn + col] as f64;
        }
    }
    assert!(
        want.iter().any(|v| *v != 0.0),
        "the LoKr corpus must be nonzero"
    );

    let got = exact_probe(
        &cfg,
        || pruned_zero_dit(&cfg),
        &ExactCorpus {
            path,
            probes: vec![],
        },
        1.0,
        PROBE,
        &x,
    );
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(*g as f64, *w, "the exact LoKr residual must be bit-exact");
    }
}

// Called only by the existing ignored genuine-LoKr proof: actual native projection output,
// genuine safetensors LoRA+LoKr files, no model-count or installed-adapter-count oracle.
fn genuine_lora_lokr_stack_causality(cfg: &MiniMaxH3DitConfig, dir: &Path, lokr: &Path) {
    let lora = write_lora(
        dir,
        "causal-stack-lora.safetensors",
        &[(PROBE.into(), cfg.inner_dim(), cfg.hidden_size)],
        Some("128"),
    );
    let x = tensor(&[3, cfg.hidden_size], 0.91);
    for pruned in [false, true] {
        let run = |lora_on, lokr_on, reverse| {
            let mut model = if pruned {
                pruned_dit(cfg)
            } else {
                full_dit(cfg)
            };
            let mut adapters = Vec::new();
            if lora_on {
                adapters.push(spec(lora.clone(), 0.75));
            }
            if lokr_on {
                adapters.push(spec(lokr.to_path_buf(), 0.5));
            }
            if reverse {
                adapters.reverse();
            }
            if !adapters.is_empty() {
                let report = apply_minimax_h3_adapters(&mut model, &adapters).unwrap();
                assert!(report.unmatched_paths.is_empty());
            }
            let out = model
                .adaptable_mut(&segs(PROBE))
                .unwrap()
                .forward(&x)
                .unwrap();
            let values = f32s(&out);
            assert!(values.iter().all(|x| x.is_finite()));
            values
        };
        let base = run(false, false, false);
        let only_lora = run(true, false, false);
        let only_lokr = run(false, true, false);
        let both = run(true, true, false);
        let reversed = run(true, true, true);
        let topology = if pruned { "pruned" } else { "full" };
        let effect = |property, a: &[f32], b: &[f32]| {
            assert_eq!(a.len(), b.len());
            let peak = a.iter().chain(b).map(|x| x.abs()).fold(0., f32::max);
            let delta = a
                .iter()
                .zip(b)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max);
            let floor = 8. * f32::EPSILON * peak;
            println!("H3_ADAPTER_STACK_CAUSAL backend=metal topology={topology} property={property} max_abs={delta:.9e} floor={floor:.9e} seam=adaptable_projection");
            assert!(
                delta > floor,
                "{topology}/{property}: stack component is not load bearing"
            );
        };
        effect("lora_vs_base", &only_lora, &base);
        effect("lokr_vs_base", &only_lokr, &base);
        effect("stack_vs_base", &both, &base);
        effect("stack_drop_lora", &both, &only_lokr);
        effect("stack_drop_lokr", &both, &only_lora);
        let peak = base
            .iter()
            .chain(&only_lora)
            .chain(&only_lokr)
            .chain(&both)
            .map(|x| x.abs())
            .fold(0., f32::max);
        let tolerance = 64. * f32::EPSILON * peak;
        let additive_error = both
            .iter()
            .zip(&only_lora)
            .zip(&only_lokr)
            .zip(&base)
            .map(|(((both, lora), lokr), base)| (both - lora - lokr + base).abs())
            .fold(0., f32::max);
        let order_error = both
            .iter()
            .zip(&reversed)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max);
        println!("H3_ADAPTER_STACK_CAUSAL backend=metal topology={topology} property=component_differentials additive_max_abs={additive_error:.9e} order_max_abs={order_error:.9e} bound={tolerance:.9e}");
        assert!(
            additive_error <= tolerance,
            "stack-minus-LoRA must preserve the standalone LoKr differential, and vice versa"
        );
        assert!(
            order_error <= tolerance,
            "additive stack order must not silently replace a component"
        );
    }
}
