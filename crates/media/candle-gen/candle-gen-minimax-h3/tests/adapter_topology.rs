//! sc-26688 — adapters and turbo schedules across the full and pruned AdaLN topologies.
//!
//! # What this file is defending
//!
//! The sc-18724/19443 adapter surface was built and verified against the **full** checkpoint — the
//! one with a `time_embedder` timestep MLP. The pruned coordinate-table checkpoint (sc-26684) loads
//! the *same body leaves* through the curve path and the same `adaptable_mut` arms, so a compatible
//! attention/FFN adapter must fold **identically** on both topologies, and an adapter target that
//! belongs to the full AdaLN surface must be **refused specifically** on a pruned checkpoint — never
//! dropped, never folded onto the curve, never rederived from `adaln_t_table`. The curve runtime
//! stores `U·S` and `W·Vh.T` of the centered, post-SiLU embedding curve; there is no validated
//! full→pruned transformation for an adapter delta, so the only correct outcome for such a target is
//! a named, typed error.
//!
//! The refusal classification is asserted through the **typed** contract — [`classify_unmatched_target`]
//! → [`UnmatchedTargetClass`] — not by pinning message wording. One user-facing marker phrase per
//! class ("pruned coordinate-table" for the topology mismatch, "no MiniMax-H3 adapter can attach"
//! for a deliberate exclusion) plus the stem-listing are the only prose edges that pin the rendered
//! error to the classification.
//!
//! Every numeric gate here is a max-abs comparison against an **independently computed** reference
//! (`x·Aᵀ·(fold·Bᵀ)·scale` from the actual file factors) or an exact cross-topology equality —
//! never a shape check, a hash, or a cosine.
//!
//! # Turbo schedules
//!
//! The published turbo recipes are `(num_inference_steps, video_shift)` pairs resolved through
//! [`JointSchedule::with_shifts`] (sc-18729); the pruned curve consumes the resulting coordinates
//! through `interpolation_position` — `clamp(t, 0, 1)·(grid−1)`, **no ×1000, no SiLU**. The
//! schedule-resolution arm below pins the 768p turbo recipe's four evaluations, the `t = 1 − σ`
//! coordinate convention, and the table lerp the curve runtime actually performs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::pruned_adaln::{interpolation_position, TABLE_KEY};
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::Weights;
use candle_gen_minimax_h3::{
    adaln_schedule, adapter_target_paths, alpha_rank_fold, apply_minimax_h3_adapters,
    classify_unmatched_target, convert_comfyui_key_space, is_comfyui_key_space, AdaLnSurface,
    JointSchedule, MiniMaxH3Dit, MiniMaxH3DitConfig, UnmatchedTargetClass, AUDIO_SIGMA_SHIFT,
};

use crate::common::{dit_fixture_config, weights, Golden, DIT_FIXTURE};

/// The module the numeric arms probe — `[96, 64]` at the fixture geometry.
const PROBE: &str = "transformer_blocks.0.attn.to_q";
/// The SwiGLU input projection, whose ComfyUI spelling (`mlp.fc1`) carries the row-half swap.
const PROBE_FFN: &str = "transformer_blocks.0.ff.net.0.proj";

/// The rank the cross-topology LoRA fixture uses — the *published* 128, so the folds asserted
/// below are the published numbers (1.0) rather than a scaled-down analogue.
const PUBLISHED_RANK: usize = 128;

/// The fixture's pruned-curve table shape: `[grid, rank]`, F32.
const CURVE_GRID: usize = 8;
const CURVE_RANK: usize = 4;

/// Relative tolerance the native (CUDA) arm gates its residual against the independent f64
/// reference with. The factors are stored bf16 and every published fold is an exact power of two,
/// so the CPU arms below gate bit-exactly; the CUDA arm admits fp32-kernel accumulation error only,
/// two orders above the measured CPU noise floor.
const NATIVE_RELATIVE_TOL: f64 = 2.0e-2;

fn dev() -> Device {
    Device::Cpu
}

/// A deterministic host-built tensor — no RNG stream, so every arm sees identical bytes.
fn tensor(shape: &[usize], seed: f32) -> Tensor {
    tensor_on(shape, seed, &dev())
}

fn tensor_on(shape: &[usize], seed: f32, device: &Device) -> Tensor {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.013 + seed).sin()) * 0.25)
        .collect();
    Tensor::from_vec(v, shape, device).expect("deterministic tensor")
}

/// Cast to `bf16` — the dtype **every** published turbo factor carries.
fn bf16(t: &Tensor) -> Tensor {
    t.to_dtype(DType::BF16).expect("cast to bf16")
}

/// Cast to `f16` — the dtype the pruned fixture stores its folded biases at (sc-26684).
fn f16(t: &Tensor) -> Tensor {
    t.to_dtype(DType::F16).expect("cast to f16")
}

/// Widen a tensor to f32 and flatten it to logical (row-major) order.
fn f32s(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .expect("widen to f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("f32 values")
}

/// The `[out, in]` logical shape of each of the six adaptable leaves at `cfg`'s geometry.
fn target_shape(cfg: &MiniMaxH3DitConfig, target: &str) -> (usize, usize) {
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
type FixtureTarget = (String, usize, usize);

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

/// Serialize a tensor map plus a top-level `__metadata__` to a real safetensors file.
///
/// Mirrors [`turbo_lora`]'s helper: candle's own `safetensors::save` writes no metadata, and the
/// metadata is the thing under test, so this goes through the `safetensors` crate directly.
fn write_safetensors(path: &Path, arrays: &[(String, Tensor)], meta: &[(&str, &str)]) {
    // Own the byte buffers over the whole serialization rather than leaking: the `TensorView`s
    // borrow these, and `serialize_to_file` consumes them synchronously within this call.
    let buffers: Vec<Box<[u8]>> = arrays.iter().map(|(_, t)| tensor_bytes(t)).collect();
    let views: Vec<(String, safetensors::tensor::TensorView)> = arrays
        .iter()
        .enumerate()
        .map(|(i, (k, t))| {
            let dtype = match t.dtype() {
                DType::BF16 => safetensors::Dtype::BF16,
                DType::F32 => safetensors::Dtype::F32,
                other => panic!("unhandled fixture dtype {other:?}"),
            };
            let view = safetensors::tensor::TensorView::new(dtype, t.dims().to_vec(), &buffers[i])
                .expect("tensor view");
            (k.clone(), view)
        })
        .collect();
    let meta: HashMap<String, String> = meta
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    safetensors::serialize_to_file(views, &Some(meta), path).expect("write the fixture");
}

/// A tensor's little-endian on-disk bytes at its own dtype.
fn tensor_bytes(t: &Tensor) -> Box<[u8]> {
    let flat = t.flatten_all().expect("flatten");
    match t.dtype() {
        DType::BF16 => flat
            .to_vec1::<half::bf16>()
            .expect("bf16 values")
            .into_iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<u8>>()
            .into_boxed_slice(),
        DType::F32 => flat
            .to_vec1::<f32>()
            .expect("f32 values")
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<u8>>()
            .into_boxed_slice(),
        other => panic!("unhandled fixture dtype {other:?}"),
    }
}

/// Write a tiny **diffusers-key-space** LoRA over `targets`, bf16 like every published turbo
/// tensor, with `alpha` stamped into the top-level `__metadata__` when given.
fn write_lora(dir: &Path, name: &str, targets: &[FixtureTarget], alpha: Option<&str>) -> PathBuf {
    let path = dir.join(name);
    let mut arrays: Vec<(String, Tensor)> = Vec::new();
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
    let mut meta: Vec<(&str, &str)> = vec![("format", "pt"), ("floating_dtype", "bfloat16")];
    if let Some(a) = alpha {
        meta.push(("alpha", a));
    }
    write_safetensors(&path, &arrays, &meta);
    path
}

fn spec(path: PathBuf, scale: f32) -> AdapterSpec {
    AdapterSpec {
        path,
        scale,
        kind: AdapterKind::Lora,
        pass_scales: None,
        moe_expert: None,
    }
}

fn segs(target: &str) -> Vec<&str> {
    target.split('.').collect()
}

/// The committed tiny DiT — real crate types, real loader, full timestep-MLP topology.
fn full_dit(cfg: &MiniMaxH3DitConfig) -> MiniMaxH3Dit {
    full_dit_on(cfg, &dev())
}

fn full_dit_on(cfg: &MiniMaxH3DitConfig, device: &Device) -> MiniMaxH3Dit {
    let f = Golden::load(DIT_FIXTURE);
    let tensors = f
        .model_map(&["src.", "in.", "out.", "layout."])
        .into_iter()
        .map(|(name, tensor)| {
            (
                name,
                tensor
                    .to_device(device)
                    .expect("move tiny DiT fixture to requested device"),
            )
        })
        .collect();
    let w = weights(tensors);
    let dit = MiniMaxH3Dit::from_weights(&w, cfg, device, DType::F32).expect("the whole tiny DiT");
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
fn pruned_fixture_map(cfg: &MiniMaxH3DitConfig, device: &Device) -> HashMap<String, Tensor> {
    let f = Golden::load(DIT_FIXTURE);
    let mut m: HashMap<String, Tensor> = f
        .model_map(&["src.", "in.", "out.", "layout."])
        .into_iter()
        .map(|(name, tensor)| {
            (
                name,
                tensor
                    .to_device(device)
                    .expect("move tiny DiT fixture to requested device"),
            )
        })
        .filter(|(key, _)| {
            !key.starts_with("time_embedder.")
                && !key.contains(".adaln_proj.linear.")
                && !key.starts_with("norm_out.linear.")
        })
        .collect();
    m.insert(
        TABLE_KEY.to_string(),
        tensor_on(&[CURVE_GRID, CURVE_RANK], 0.37, device),
    );
    for i in 0..cfg.num_layers {
        let prefix = format!("transformer_blocks.{i}.adaln_proj.linear");
        let out = 18 * cfg.hidden_size; // 6 · MODALITY_NUM · hidden_size
        m.insert(
            format!("{prefix}.weight"),
            bf16(&tensor_on(&[out, CURVE_RANK], 0.41 + i as f32, device)),
        );
        m.insert(
            format!("{prefix}.bias"),
            f16(&tensor_on(&[out], 0.43 + i as f32, device)),
        );
    }
    let final_out = 2 * cfg.hidden_size;
    m.insert(
        "norm_out.linear.weight".to_string(),
        bf16(&tensor_on(&[final_out, CURVE_RANK], 0.47, device)),
    );
    m.insert(
        "norm_out.linear.bias".to_string(),
        f16(&tensor_on(&[final_out], 0.53, device)),
    );
    m
}

fn pruned_weights(cfg: &MiniMaxH3DitConfig, device: &Device) -> Weights {
    weights(pruned_fixture_map(cfg, device))
}

/// A second body: every checkpoint tensor doubled — a different base, identical geometry. The
/// table is part of the map, so this is a genuinely different pruning basis as well. Widening each
/// tensor to f32 before doubling keeps the map valid across the mixed F32/BF16/F16 operand dtypes
/// (the pruned contract accepts any of F16/BF16/F32 for a reduced operand).
fn doubled_pruned_weights(cfg: &MiniMaxH3DitConfig, device: &Device) -> Weights {
    let mut m = pruned_fixture_map(cfg, device);
    for key in m.keys().cloned().collect::<Vec<_>>() {
        let doubled = m[&key]
            .to_dtype(DType::F32)
            .expect("widen for doubling")
            .affine(2.0, 0.0)
            .expect("double");
        m.insert(key, doubled);
    }
    weights(m)
}

fn pruned_dit(cfg: &MiniMaxH3DitConfig) -> MiniMaxH3Dit {
    pruned_dit_on(cfg, &dev())
}

fn pruned_dit_on(cfg: &MiniMaxH3DitConfig, device: &Device) -> MiniMaxH3Dit {
    let w = pruned_weights(cfg, device);
    let dit = MiniMaxH3Dit::from_weights(&w, cfg, device, DType::F32).expect("the pruned tiny DiT");
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

/// `y_with_lora(x) − y_base(x)` at `target` — the residual the install actually added.
fn probe_residual(
    dit: impl Fn() -> MiniMaxH3Dit,
    lora: &Path,
    scale: f32,
    target: &str,
    x: &Tensor,
) -> Tensor {
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
    (y1 - y0).expect("residual")
}

fn max_abs(t: &Tensor) -> f32 {
    t.to_dtype(DType::F32)
        .expect("f32")
        .abs()
        .expect("abs")
        .flatten_all()
        .expect("flat")
        .max(0)
        .expect("max")
        .to_scalar::<f32>()
        .expect("scalar")
}

/// The stored factor bytes a reference computation reads: `a` is the `[rank, in]` lora_A and `b`
/// the `[out, rank]` lora_B, both widened from the actual file bytes.
struct LoraFactors<'a> {
    a: &'a [f32],
    b: &'a [f32],
    in_dim: usize,
    out: usize,
}

/// Independent `((x·a)·b)·scale` in f64 — computed from the **actual file bytes**, widened
/// exactly, so the only tolerance left absorbs accumulation order.
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
    let mut y = vec![0.0f64; rows * out];
    for r in 0..rows {
        let mut mid = vec![0.0f64; rank];
        for (k, mid_k) in mid.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for j in 0..in_dim {
                acc += x[r * in_dim + j] as f64 * a[k * in_dim + j] as f64;
            }
            *mid_k = acc;
        }
        for o in 0..out {
            let mut acc = 0.0f64;
            for k in 0..rank {
                acc += mid[k] * b[o * rank + k] as f64;
            }
            y[r * out + o] = acc * fold as f64 * scale as f64;
        }
    }
    y
}

/// The factor values the fixture wrote for one target, read back from the written file so the
/// reference uses the actual bytes the install consumed.
fn fixture_factors(
    lora: &Path,
    target: &str,
    rank: usize,
    inn: usize,
    out: usize,
) -> (Vec<f32>, Vec<f32>) {
    let af = candle_gen::train::merge::read_adapter(lora).expect("the written LoRA");
    let a = af
        .tensors
        .get(&format!("{target}.lora_A.default.weight"))
        .unwrap_or_else(|| panic!("{target} lora_A"));
    let b = af
        .tensors
        .get(&format!("{target}.lora_B.default.weight"))
        .unwrap_or_else(|| panic!("{target} lora_B"));
    assert_eq!(
        a.dims2().expect("2-D"),
        (rank, inn),
        "{target} lora_A shape"
    );
    assert_eq!(
        b.dims2().expect("2-D"),
        (out, rank),
        "{target} lora_B shape"
    );
    (f32s(a), f32s(b))
}

fn assert_matches_reference(got: &[f32], want: &[f64], what: &str) {
    let mut worst = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        worst = worst.max((*g as f64 - w).abs());
    }
    let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        worst <= 1.0e-3 * peak,
        "{what}: the installed residual must match the independent x·Aᵀ·(fold·Bᵀ)·scale reference \
         (worst {worst}, peak {peak})"
    );
}

/// Relative max-abs-diff against the independent reference — the only honest gate for the native
/// (CUDA) arm, whose fp32 kernels over bf16 factors accumulate past the CPU bit-exact floor.
fn assert_matches_reference_rel(got: &[f32], want: &[f64], tol: f64, what: &str) {
    let mut worst = 0.0f64;
    let mut peak = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        worst = worst.max((*g as f64 - w).abs());
        peak = peak.max(w.abs());
    }
    assert!(
        worst <= tol * peak,
        "{what}: the native residual must match the independent reference (worst {worst}, peak \
         {peak}, relative tolerance {tol})"
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
        let diff = (&full_res - &pruned_res).expect("difference");
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
                in_dim: inn,
                out,
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
/// on a pruned checkpoint — classified as a topology mismatch with no full→pruned transformation —
/// while the same stems on the full topology classify as a deliberate exclusion. Neither surface
/// folds the file, and neither drops the targets.
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

    // The typed contract: the same stems are a topology mismatch on the pruned curve and a
    // deliberate exclusion on the full surface — never a generic miss.
    for stem in [
        "time_embedder.linear_1",
        "norm_out.linear",
        "transformer_blocks.0.adaln_proj.linear",
    ] {
        assert_eq!(
            classify_unmatched_target(stem, AdaLnSurface::PrunedCurve),
            UnmatchedTargetClass::PrunedCurveTopologyMismatch,
            "{stem} must classify as a pruned topology mismatch"
        );
        assert_eq!(
            classify_unmatched_target(stem, AdaLnSurface::FullTimestepMlp),
            UnmatchedTargetClass::DeliberatelyExcluded,
            "{stem} must classify as a deliberate exclusion on the full surface"
        );
    }

    let pruned_error = apply_minimax_h3_adapters(&mut pruned_dit(&cfg), &[spec(lora.clone(), 1.0)])
        .expect_err("the pruned curve must refuse full-topology AdaLN targets")
        .to_string();
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
    assert!(
        full_error.contains("no MiniMax-H3 adapter can attach"),
        "on the full topology the same targets are a deliberate exclusion: {full_error}"
    );
    assert!(
        !full_error.contains("pruned coordinate-table"),
        "the topology-mismatch wording must not leak onto the full surface: {full_error}"
    );
}

/// An adapter targeting a real module no H3 adapter can ever attach to (the bias-Linear I/O heads,
/// the RMS norms) classifies as a deliberate exclusion — not a generic miss — on either topology.
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

    for stem in ["context_embedder", "transformer_blocks.0.norm1"] {
        assert_eq!(
            classify_unmatched_target(stem, AdaLnSurface::PrunedCurve),
            UnmatchedTargetClass::DeliberatelyExcluded,
            "{stem} must classify as a deliberate exclusion on the pruned curve"
        );
        assert_eq!(
            classify_unmatched_target(stem, AdaLnSurface::FullTimestepMlp),
            UnmatchedTargetClass::DeliberatelyExcluded,
            "{stem} must classify as a deliberate exclusion on the full surface"
        );
    }

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
    let coords = dit.curve_coordinates(&ts).expect("curve coordinates");
    let table = {
        let w = pruned_weights(&cfg, &dev());
        f32s(&w.require(TABLE_KEY).expect("the table"))
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
    let arrays = vec![
        (
            "transformer_blocks.0.attn.to_q.lokr_w1".to_string(),
            bf16(&tensor(&[8, 8], 1.0)),
        ),
        (
            "transformer_blocks.0.attn.to_q.lokr_w2".to_string(),
            bf16(&tensor(&[12, 8], 2.0)),
        ),
    ];
    let path = dir.path().join("lokr.safetensors");
    write_safetensors(
        &path,
        &arrays,
        &[
            ("format", "pt"),
            ("networkType", "lokr"),
            ("rank", "8"),
            ("alpha", "8"),
        ],
    );

    let x = tensor(&[3, cfg.hidden_size], 0.7);
    let full_res = probe_residual(|| full_dit(&cfg), &path, 1.0, PROBE, &x);
    let pruned_res = probe_residual(|| pruned_dit(&cfg), &path, 1.0, PROBE, &x);
    assert!(
        max_abs(&full_res) > 0.0,
        "the LoKr residual must be nonzero or the test proves nothing"
    );
    let diff = (full_res - pruned_res).expect("difference");
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
    let fused_rank = 6usize;
    let in_dim = cfg.hidden_size;
    let out_dim = cfg.inner_dim();
    let down = bf16(&tensor(&[fused_rank, in_dim], 0.61));
    // up [3·out, 3r]: slice i is nonzero only on rank columns [i·r, (i+1)·r) — the block-diagonal
    // spelling `is_block_diagonal` detects.
    let r = fused_rank / 3;
    let mut up_v = vec![0.0f32; 3 * out_dim * fused_rank];
    for i in 0..3usize {
        let seed = [0.71f32, 0.73, 0.79][i];
        for row in 0..out_dim {
            for k in 0..r {
                let index = (i * out_dim + row) * fused_rank + i * r + k;
                up_v[index] = ((index as f32 * 0.013 + seed).sin()) * 0.25;
            }
        }
    }
    let up = Tensor::from_vec(up_v, (3 * out_dim, fused_rank), &dev()).expect("block-diagonal up");
    let up = bf16(&up);
    let arrays = vec![
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
    write_safetensors(
        &comfy_path,
        &arrays,
        &[
            ("format", "pt"),
            ("floating_dtype", "bfloat16"),
            ("alpha", "24"),
        ],
    );

    let af = candle_gen::train::merge::read_adapter(&comfy_path).expect("read back");
    let converted =
        convert_comfyui_key_space(&af.tensors, &af.meta).expect("the ComfyUI conversion");
    assert!(
        !is_comfyui_key_space(converted.keys().map(String::as_str)),
        "the conversion must leave no ComfyUI module behind"
    );
    // The split-adjusted alpha: fused 24 over fused rank 6; block-diagonal division emits
    // (24/3)/2 = 4.0 per projection — `alpha/rank` held across the split.
    let emitted_alpha = converted
        .get(&format!("{PROBE}.alpha"))
        .expect("the conversion emits a per-target alpha")
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("f32 alpha")[0];
    assert_eq!(
        emitted_alpha, 8.0,
        "the fused 24 must divide by three across the split"
    );
    let converted_path = dir.path().join("converted.safetensors");
    let arrays: Vec<(String, Tensor)> = converted
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    write_safetensors(&converted_path, &arrays, &[]);

    let x = tensor(&[3, cfg.hidden_size], 0.7);
    let full_res = probe_residual(|| full_dit(&cfg), &converted_path, 1.0, PROBE, &x);
    let pruned_res = probe_residual(|| pruned_dit(&cfg), &converted_path, 1.0, PROBE, &x);
    assert!(
        max_abs(&pruned_res) > 0.0,
        "the converted qkv residual must be nonzero"
    );
    let diff = (&full_res - &pruned_res).expect("difference");
    assert_eq!(
        max_abs(&diff),
        0.0,
        "the converted namespace must fold bit-identically across topologies"
    );

    // Independent reference over the CONVERTED factor bytes at the emitted alpha's fold.
    let wconv = candle_gen::train::merge::read_adapter(&converted_path).expect("read converted");
    let a = f32s(
        wconv
            .tensors
            .get(&format!("{PROBE}.lora_A.weight"))
            .expect("converted lora_A"),
    );
    let b = f32s(
        wconv
            .tensors
            .get(&format!("{PROBE}.lora_B.weight"))
            .expect("converted lora_B"),
    );
    let fold = alpha_rank_fold(emitted_alpha, 2.0);
    let want = reference_residual(
        LoraFactors {
            a: &a,
            b: &b,
            in_dim,
            out: out_dim,
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

    let residual_on = |weights: Weights, what: &str| -> Tensor {
        let mut base = {
            let w = weights;
            MiniMaxH3Dit::from_weights(&w, &cfg, &dev(), DType::F32)
                .unwrap_or_else(|_| panic!("load {what}"))
        };
        let y0 = base
            .adaptable_mut(&s)
            .expect("probe")
            .forward(&x)
            .expect("base");
        let mut adapted = {
            let w = if what == "pruned body" {
                pruned_weights(&cfg, &dev())
            } else {
                doubled_pruned_weights(&cfg, &dev())
            };
            MiniMaxH3Dit::from_weights(&w, &cfg, &dev(), DType::F32)
                .unwrap_or_else(|_| panic!("load {what}"))
        };
        apply_minimax_h3_adapters(&mut adapted, &[spec(lora.clone(), 1.0)]).expect("install");
        let y1 = adapted
            .adaptable_mut(&s)
            .expect("probe")
            .forward(&x)
            .expect("adapted");
        (y1 - y0).expect("residual")
    };

    let res_a = residual_on(pruned_weights(&cfg, &dev()), "pruned body");
    let res_b = residual_on(doubled_pruned_weights(&cfg, &dev()), "doubled body");
    assert!(max_abs(&res_a) > 0.0, "the residual must be nonzero");
    // f32 addition rounds, so adapted−base reproduces the residual to f32 rounding, not bit-exact —
    // but any wrong fold, dropped target or base-weight coupling moves it by orders of magnitude.
    let diff = (&res_a - &res_b).expect("difference");
    let worst = max_abs(&diff);
    let peak = max_abs(&res_a) as f64;
    assert!(
        worst as f64 <= 1.0e-4 * peak,
        "the delta must be independent of the body weights it folds onto (worst {worst}, peak {peak})"
    );
}

/// Requires a real CUDA device. `absence is failure, not skip`: the integration lane must supply
/// the device, so an unavailable device fails the run rather than lowering coverage.
///
/// Builds the pruned fixture DiT **on** the CUDA device, installs the body LoRA through the
/// production [`apply_minimax_h3_adapters`] path, and probes the residual at [`PROBE`] and
/// [`PROBE_FFN`] against the same independent f64 reference — gated at
/// [`NATIVE_RELATIVE_TOL`] relative, because fp32 CUDA kernels over bf16 factors accumulate past
/// the CPU bit-exact floor. It also asserts, on the same device and **before any forward**, that a
/// full-topology AdaLN target file is refused at install time with the topology-mismatch wording.
#[test]
#[ignore = "requires a CUDA device; absence is failure, not skip (integration lane, sc-26688)"]
fn native_adapter_folds_on_the_pruned_curve() {
    let device = Device::new_cuda(0).expect(
        "the integration lane must provide a CUDA device — an unavailable device is a failure, not \
         a skip",
    );
    let cfg = dit_fixture_config();

    let dir = tempfile::tempdir().expect("fixture dir");
    let lora = write_lora(
        dir.path(),
        "body.safetensors",
        &all_targets(&cfg),
        Some("128"),
    );
    let x = tensor_on(&[3, cfg.hidden_size], 0.7, &device);

    for target in [PROBE, PROBE_FFN] {
        let (out, inn) = target_shape(&cfg, target);
        let residual = probe_residual(|| pruned_dit_on(&cfg, &device), &lora, 1.0, target, &x);
        assert!(
            max_abs(&residual) > 0.0,
            "{target}: the native residual must be nonzero"
        );
        let (a, b) = fixture_factors(&lora, target, PUBLISHED_RANK, inn, out);
        let fold = alpha_rank_fold(128.0, PUBLISHED_RANK as f32);
        let want = reference_residual(
            LoraFactors {
                a: &a,
                b: &b,
                in_dim: inn,
                out,
            },
            fold,
            1.0,
            &f32s(&x),
            3,
        );
        assert_matches_reference_rel(&f32s(&residual), &want, NATIVE_RELATIVE_TOL, target);
    }

    // A full-topology AdaLN file is refused at install — before any forward — naming the topology.
    let mut adaln_targets = all_targets(&cfg);
    adaln_targets.push((
        "time_embedder.linear_1".to_string(),
        cfg.time_embed_dim,
        cfg.freq_dim,
    ));
    adaln_targets.push((
        "norm_out.linear".to_string(),
        2 * cfg.hidden_size,
        cfg.time_embed_dim,
    ));
    adaln_targets.push((
        "transformer_blocks.0.adaln_proj.linear".to_string(),
        18 * cfg.hidden_size,
        cfg.time_embed_dim,
    ));
    let adaln_lora = write_lora(dir.path(), "adaln.safetensors", &adaln_targets, Some("128"));
    let mut host = pruned_dit_on(&cfg, &device);
    let err = apply_minimax_h3_adapters(&mut host, &[spec(adaln_lora, 1.0)])
        .expect_err("the pruned curve must refuse full-topology AdaLN targets on the native device")
        .to_string();
    assert!(
        err.contains("pruned coordinate-table"),
        "the refusal must name the topology: {err}"
    );
}
