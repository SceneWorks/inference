//! sc-18728 / sc-19443 — the lightx2v turbo LoRA folds at the **measured** strength on the candle
//! lane, over the committed tiny DiT, and a **ComfyUI** export converts to its diffusers twin.
//!
//! # What this file is defending
//!
//! `lightx2v/Minimax-h3-Turbo` stamps its LoRA alpha as a bare top-level `__metadata__` string and
//! ships **no** `rank` key, no per-target `.alpha` tensor and no `lora_adapter_metadata` blob. A
//! loader that falls back to `alpha = rank` folds the `alpha=8, rank=128` files **16× too strong**;
//! one that defaults a missing rank to 1.0 folds them **128×**. Neither failure has a shape, a
//! checksum or a key-coverage proof that can see it — the render simply comes out wrong.
//!
//! # Why all the alpha cases are here, not just one
//!
//! The published set disagrees *within one repo* — `alpha: "128"` on the 4-step 768p file,
//! `alpha: "8"` on the 8-step and ref2v ones, and **nothing at all** on `4step_v0.1`. A test
//! covering only the `alpha=128` file would pass against an implementation that hardcodes `1.0`, and
//! a test covering only an `alpha=8` file would pass against one that hardcodes `0.0625`. Every
//! constant-returning implementation fails at least one arm below.
//!
//! The absolute arm is checked against an **independently computed** `x·Aᵀ·Bᵀ` rather than against a
//! second call into the code under test, and the relative arms are ratios, so a shared error in the
//! fold formula cannot cancel out of both.
//!
//! # Numeric gate: relative max-abs-diff, never cosine
//!
//! Cosine is scale-invariant, so it is structurally blind to the exact defect class this whole
//! module exists to close — a 16× fold and a correct one are cosine 1.0 apart. Norm and checksum
//! were blind to real defects in this family seven times. Every parity assertion below is
//! `max|a-b| / max|b|`.

use crate::common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen_minimax_h3::adapters::{
    adapter_target_paths, alpha_rank_fold, apply_minimax_h3_adapters, convert_comfyui_key_space,
    is_comfyui_key_space, resolve_alpha, resolve_rank, unflatten_minimax_h3_trainer_tensors,
    DEFAULT_LORA_ALPHA,
};
use candle_gen_minimax_h3::{MiniMaxH3Dit, MiniMaxH3DitConfig};

use common::{dit_fixture_config, weights, Golden, DIT_FIXTURE};

/// The published rank — every one of the seven turbo files is rank 128. Used verbatim in the tiny
/// fixtures so the folds asserted below are the *published* numbers (1.0 and 0.0625) rather than a
/// scaled-down analogue of them.
const PUBLISHED_RANK: usize = 128;

/// The module the numeric arms probe. `attn.to_q` is `[96, 64]` at the fixture geometry, so its
/// factors are `A [128, 64]` / `B [96, 128]`.
const PROBE: &str = "transformer_blocks.0.attn.to_q";

fn dev() -> Device {
    Device::Cpu
}

/// A deterministic host-built tensor — no RNG stream, so every arm sees identical bytes.
fn tensor(shape: &[usize], seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.013 + seed).sin()) * 0.25)
        .collect();
    Tensor::from_vec(v, shape, &dev()).expect("fixture tensor")
}

/// Cast to `bf16` — the dtype **every** tensor in **every** published turbo file carries.
///
/// The fixtures are deliberately not f32: with f32 factors the install's dtype-preserving fold
/// (`affine`, which multiplies at the tensor's own dtype) is unobservable, and the sc-18724 review
/// found that exact hole on the MLX lane — a dropped cast survived all 428 tests. Every fold
/// asserted here is an exact power of two, so bf16 storage costs the assertions no exactness.
fn bf16(t: &Tensor) -> Tensor {
    t.to_dtype(DType::BF16).expect("cast to bf16")
}

/// The `[out, in]` logical shape of each of the six adaptable leaves at `cfg`'s geometry.
fn target_shape(cfg: &MiniMaxH3DitConfig, target: &str) -> (usize, usize) {
    match target {
        t if t.ends_with("attn.to_q") || t.ends_with("attn.to_k") || t.ends_with("attn.to_v") => {
            (cfg.inner_dim(), cfg.hidden_size)
        }
        t if t.ends_with("attn.to_out.0") => (cfg.hidden_size, cfg.inner_dim()),
        // The SwiGLU input emits `[value | gate]`, so twice `ffn_dim`.
        t if t.ends_with("ff.net.0.proj") => (2 * cfg.ffn_dim, cfg.hidden_size),
        t if t.ends_with("ff.net.2") => (cfg.hidden_size, cfg.ffn_dim),
        other => panic!("unknown target {other}"),
    }
}

/// Serialize a tensor map plus a top-level `__metadata__` to a real safetensors file.
///
/// The metadata is the thing under test, and candle's own `safetensors::save` writes none, so this
/// goes through the `safetensors` crate directly.
fn write_safetensors(path: &Path, arrays: &[(String, Tensor)], meta: &[(&str, &str)]) {
    let views: Vec<(String, safetensors::tensor::TensorView)> = arrays
        .iter()
        .map(|(k, t)| {
            let dtype = match t.dtype() {
                DType::BF16 => safetensors::Dtype::BF16,
                DType::F32 => safetensors::Dtype::F32,
                other => panic!("unhandled fixture dtype {other:?}"),
            };
            let bytes = tensor_bytes(t);
            let view =
                safetensors::tensor::TensorView::new(dtype, t.dims().to_vec(), Box::leak(bytes))
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

/// Write a tiny **diffusers-key-space** turbo LoRA covering every adaptable module at `cfg`'s
/// geometry, keyed exactly as the published files are — `.lora_A.default.weight` /
/// `.lora_B.default.weight`, no namespace prefix — with `alpha` stamped into the top-level
/// `__metadata__` only when `alpha` is `Some`.
fn write_lora(dir: &Path, name: &str, cfg: &MiniMaxH3DitConfig, alpha: Option<&str>) -> PathBuf {
    write_lora_with_meta(dir, name, cfg, alpha, &[])
}

/// [`write_lora`] plus arbitrary extra `__metadata__` entries — the `lora_adapter_metadata` blob
/// arms need one, and everything else about the file must stay identical so the arms differ in
/// exactly the metadata under test.
fn write_lora_with_meta(
    dir: &Path,
    name: &str,
    cfg: &MiniMaxH3DitConfig,
    alpha: Option<&str>,
    extra: &[(&str, &str)],
) -> PathBuf {
    let path = dir.join(name);
    let mut arrays: Vec<(String, Tensor)> = Vec::new();
    for target in adapter_target_paths(cfg) {
        let (out, inn) = target_shape(cfg, &target);
        // A distinct seed per module, so a routing bug that folds the right factors onto the wrong
        // module is observable rather than symmetric.
        let seed = target.len() as f32;
        arrays.push((
            format!("{target}.lora_A.default.weight"),
            bf16(&tensor(&[PUBLISHED_RANK, inn], seed)),
        ));
        arrays.push((
            format!("{target}.lora_B.default.weight"),
            bf16(&tensor(&[out, PUBLISHED_RANK], seed + 0.5)),
        ));
    }
    let mut meta: Vec<(&str, &str)> = vec![("format", "pt"), ("floating_dtype", "bfloat16")];
    if let Some(a) = alpha {
        meta.push(("alpha", a));
    }
    meta.extend_from_slice(extra);
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

/// The committed tiny DiT — real crate types, real loader, 2 transformer blocks + 2 refiner blocks.
fn tiny_dit(cfg: &MiniMaxH3DitConfig) -> MiniMaxH3Dit {
    let f = Golden::load(DIT_FIXTURE);
    let w = weights(f.model_map(&["src.", "in.", "out.", "layout."]));
    MiniMaxH3Dit::from_weights(&w, cfg, &dev(), DType::F32).expect("the whole tiny DiT")
}

/// **Relative max-abs-diff**, the only honest gate for this defect class. `max|a-b| / max|b|`.
fn rel_max_abs(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.dims(), b.dims(), "shape");
    let d = max_abs(&(a - b).expect("difference"));
    let scale = max_abs(b);
    if scale == 0.0 {
        return d;
    }
    d / scale
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

/// `y_with_lora(x) − y_base(x)` at [`PROBE`] — the residual the install actually added.
fn probe_residual(cfg: &MiniMaxH3DitConfig, lora: &Path, scale: f32, x: &Tensor) -> Tensor {
    let segs: Vec<&str> = PROBE.split('.').collect();
    let mut base = tiny_dit(cfg);
    let y0 = base
        .adaptable_mut(&segs)
        .expect("probe module")
        .forward(x)
        .expect("base forward");

    let mut adapted = tiny_dit(cfg);
    let report = apply_minimax_h3_adapters(&mut adapted, &[spec(lora.to_path_buf(), scale)])
        .expect("install");
    assert_eq!(report.applied, adapter_target_paths(cfg).len());
    assert!(report.unmatched_paths.is_empty());
    let y1 = adapted
        .adaptable_mut(&segs)
        .expect("probe module")
        .forward(x)
        .expect("adapted forward");
    (y1 - y0).expect("residual")
}

// ─── the target map ────────────────────────────────────────────────────────────────────────────

/// Every path [`adapter_target_paths`] enumerates **resolves through the model tree**, and the two
/// stacks are both covered.
///
/// This is the guard that keeps the enumeration and the tree from drifting apart. It constructs a
/// real DiT and calls the real `adaptable_mut`, so it cannot pass on a path that only exists in the
/// list.
#[test]
fn every_enumerated_target_resolves_through_the_model_tree() {
    let cfg = dit_fixture_config();
    let mut dit = tiny_dit(&cfg);
    let paths = adapter_target_paths(&cfg);
    assert_eq!(
        paths.len(),
        (cfg.num_layers + cfg.num_refiner_layers) * 6,
        "six leaves per block across both stacks"
    );
    for p in &paths {
        let segs: Vec<&str> = p.split('.').collect();
        assert!(
            dit.adaptable_mut(&segs).is_some(),
            "unresolvable target {p}"
        );
    }
    assert_eq!(
        paths
            .iter()
            .filter(|p| p.starts_with("token_refiner."))
            .count(),
        cfg.num_refiner_layers * 6,
        "24 of the published 624 tensors target the refiner — it cannot be stubbed"
    );
    // At the SHIPPED geometry the same enumeration is the published 312 modules / 624 tensors.
    let shipped = adapter_target_paths(&MiniMaxH3DitConfig::default());
    assert_eq!(shipped.len(), 312, "50·6 + 2·6");
    assert_eq!(
        shipped.len() * 2,
        624,
        "each module carries lora_A + lora_B"
    );
}

/// **`adaln_proj` is unreachable to adapters**, and so is every norm and every input/output
/// projection.
///
/// Not a stylistic exclusion: `crate::dit::adaln` precomputes the modulation tables and then evicts
/// the projection, so an adaptable `adaln_proj` would make one checkpoint behave as two models
/// depending on whether the eviction had already run.
#[test]
fn adaln_and_the_norms_are_not_addressable() {
    let cfg = dit_fixture_config();
    let mut dit = tiny_dit(&cfg);
    for path in [
        "transformer_blocks.0.adaln_proj.linear",
        "transformer_blocks.0.adaln_proj",
        "transformer_blocks.0.norm1",
        "transformer_blocks.0.attn.norm_q",
        "token_refiner.final_norm",
        "proj_in",
        "transformer_blocks.99.attn.to_q",
        "token_refiner.refiner_blocks.99.attn.to_q",
    ] {
        let segs: Vec<&str> = path.split('.').collect();
        assert!(
            dit.adaptable_mut(&segs).is_none(),
            "{path} must NOT be addressable by an adapter"
        );
    }
}

// ─── alpha resolution ──────────────────────────────────────────────────────────────────────────

/// The three published `__metadata__` spellings resolve to the three alphas the turbo set actually
/// ships — and the ABSENT case is 8, not the rank.
#[test]
fn alpha_resolution_covers_every_published_spelling() {
    let m = |pairs: &[(&str, &str)]| -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    };
    assert_eq!(resolve_alpha(&m(&[("alpha", "128")])).unwrap(), 128.0);
    assert_eq!(resolve_alpha(&m(&[("alpha", "8")])).unwrap(), 8.0);
    assert_eq!(
        resolve_alpha(&m(&[("format", "pt")])).unwrap(),
        DEFAULT_LORA_ALPHA,
        "an absent alpha must fall back to upstream's DEFAULT_LORA_ALPHA, never to the rank"
    );
    // A present-but-malformed alpha fails loudly rather than falling through to the default —
    // otherwise the correction is indistinguishable from the bug it replaces.
    for bad in ["", "eight", "nan", "inf"] {
        assert!(
            resolve_alpha(&m(&[("alpha", bad)])).is_err(),
            "alpha={bad:?} must be rejected, not silently treated as {DEFAULT_LORA_ALPHA}"
        );
    }
}

/// The fold each published file produces at rank 128. **The 0.0625 cases are the point**: a test
/// that only covered the alpha=128 file would assert the code's own do-nothing default.
#[test]
fn published_alphas_produce_the_measured_folds() {
    assert_eq!(alpha_rank_fold(128.0, 128.0), 1.0, "4step_v1.0_768p");
    assert_eq!(alpha_rank_fold(8.0, 128.0), 0.0625, "8step_v1.0");
    assert_eq!(
        alpha_rank_fold(DEFAULT_LORA_ALPHA, 128.0),
        0.0625,
        "4step_v0.1"
    );
    // The two failure modes, stated as the numbers they would produce.
    assert_eq!(
        alpha_rank_fold(128.0, 128.0) / alpha_rank_fold(8.0, 128.0),
        16.0,
        "falling back to alpha = rank folds the 8-step file 16x too strong"
    );
    assert_eq!(
        alpha_rank_fold(8.0, 1.0) / alpha_rank_fold(8.0, 128.0),
        128.0,
        "a rank defaulted to 1.0 folds it 128x too strong"
    );
}

/// Rank comes from the factor shapes. A metadata `rank` is a cross-check only, and a disagreement is
/// an error rather than a silent pick.
#[test]
fn rank_comes_from_the_factor_shapes() {
    let down = Tensor::zeros((128, 64), DType::F32, &dev()).unwrap();
    assert_eq!(resolve_rank("p", &down, None).unwrap(), 128.0);
    assert_eq!(resolve_rank("p", &down, Some("128")).unwrap(), 128.0);
    assert!(
        resolve_rank("p", &down, Some("1")).is_err(),
        "a metadata rank that disagrees with the shapes must not be silently preferred"
    );
    assert!(resolve_rank("p", &down, Some("oops")).is_err());
    let empty = Tensor::zeros((0, 8), DType::F32, &dev()).unwrap();
    assert!(resolve_rank("p", &empty, None).is_err(), "zero rank");
    let flat = Tensor::zeros(8, DType::F32, &dev()).unwrap();
    assert!(resolve_rank("p", &flat, None).is_err(), "1-D factor");
}

// ─── the install, numerically ──────────────────────────────────────────────────────────────────

/// **The installed residual is `scale · (alpha/rank) · x·Aᵀ·Bᵀ`**, checked against an independently
/// computed product rather than against a second call into the code under test.
///
/// The absolute arm pins the `alpha=128` file at fold 1.0; the ratio arms pin the two `0.0625`
/// spellings **relative to it**, so an error shared by the fold formula and the reference cannot
/// cancel out of both.
#[test]
fn the_install_folds_at_the_published_strength() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().unwrap();
    let x = tensor(&[1, 5, cfg.hidden_size], 0.7);

    // ── absolute: alpha 128 / rank 128 = fold 1.0, against a hand-computed x·Aᵀ·Bᵀ.
    let a128 = write_lora(dir.path(), "a128.safetensors", &cfg, Some("128"));
    let got = probe_residual(&cfg, &a128, 1.0, &x);

    let (out, inn) = target_shape(&cfg, PROBE);
    let seed = PROBE.len() as f32;
    let down = bf16(&tensor(&[PUBLISHED_RANK, inn], seed));
    let up = bf16(&tensor(&[out, PUBLISHED_RANK], seed + 0.5));
    let want = x
        .reshape((5, inn))
        .unwrap()
        .matmul(
            &down
                .to_dtype(DType::F32)
                .unwrap()
                .t()
                .unwrap()
                .contiguous()
                .unwrap(),
        )
        .unwrap()
        .matmul(
            &up.to_dtype(DType::F32)
                .unwrap()
                .t()
                .unwrap()
                .contiguous()
                .unwrap(),
        )
        .unwrap()
        .reshape((1, 5, out))
        .unwrap();
    let drift = rel_max_abs(&got, &want);
    println!("[alpha=128] rel-max-abs vs an independent x·Aᵀ·Bᵀ = {drift:.3e}");
    assert!(
        drift < 1e-2,
        "an alpha=128 / rank=128 file must fold at exactly 1.0; got rel-max-abs {drift:.3e}"
    );

    // ── ratio: alpha 8 is 1/16 of alpha 128, and an ABSENT alpha equals a declared 8.
    let a8 = write_lora(dir.path(), "a8.safetensors", &cfg, Some("8"));
    let none = write_lora(dir.path(), "none.safetensors", &cfg, None);
    let r128 = max_abs(&got);
    let r8 = max_abs(&probe_residual(&cfg, &a8, 1.0, &x));
    let rnone = max_abs(&probe_residual(&cfg, &none, 1.0, &x));
    assert!(
        ((r8 / r128) - 0.0625).abs() < 1e-3,
        "alpha=8 must fold at 1/16 of alpha=128; got {}",
        r8 / r128
    );
    assert!(
        ((rnone / r128) - 0.0625).abs() < 1e-3,
        "an ABSENT alpha must fold like a declared 8 (DEFAULT_LORA_ALPHA), not like the rank; got {}",
        rnone / r128
    );

    // ── the user's strength multiplies the fold, and a zero strength is exactly inert.
    let r_half = max_abs(&probe_residual(&cfg, &a128, 0.5, &x));
    assert!(
        ((r_half / r128) - 0.5).abs() < 1e-3,
        "AdapterSpec::scale must multiply the fold; got {}",
        r_half / r128
    );
    assert_eq!(
        max_abs(&probe_residual(&cfg, &a128, 0.0, &x)),
        0.0,
        "a zero-strength adapter must be byte-identical to the bare base"
    );
}

/// **The installed fold keeps the factor's dtype.**
///
/// The only guard that can see a stray widening cast in the install. Every published fold is an
/// exact power of two, so a bf16 factor and an f32 one produce **bit-identical** products — no
/// numeric assertion anywhere in this file can distinguish them. The sc-18724 review found exactly
/// this hole on the MLX lane: a dropped dtype cast survived all 428 tests.
///
/// It reads the dtype of the tensor `apply_minimax_h3_adapters` actually installed, through the real
/// entry point — not a reimplementation of the multiply.
#[test]
fn the_installed_fold_keeps_the_factor_dtype() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().unwrap();
    let lora = write_lora(dir.path(), "bf16.safetensors", &cfg, Some("8"));

    let mut dit = tiny_dit(&cfg);
    apply_minimax_h3_adapters(&mut dit, &[spec(lora, 1.0)]).expect("install");

    let segs: Vec<&str> = PROBE.split('.').collect();
    // The base DiT is f32 here, so the assertions below also pin that the residual dtype tracks the
    // FACTOR and not the host — an install that took the host's dtype would read F32 and pass a
    // weaker claim. Captured before the mutable borrow so the two facts sit side by side.
    let host_dtype = dit.dtype();
    let lin = dit.adaptable_mut(&segs).expect("probe module");
    let installed = lin.adapters();
    assert_eq!(installed.len(), 1, "one residual");
    assert_eq!(
        installed[0].dtype(),
        DType::BF16,
        "the fold must be applied AT THE FACTOR'S DTYPE — every published file is bf16, and a fold \
         that widened to f32 would run the low-rank matmuls at a precision the reference does not. \
         No numeric assertion can see this: every published fold is an exact power of two."
    );
    assert_eq!(host_dtype, DType::F32);
    let ((a_in, a_rank), (b_rank, b_out)) = installed[0].shapes();
    let (out, inn) = target_shape(&cfg, PROBE);
    assert_eq!((a_in, a_rank), (inn, PUBLISHED_RANK), "a is [in, rank]");
    assert_eq!((b_rank, b_out), (PUBLISHED_RANK, out), "b is [rank, out]");
    assert_eq!(installed[0].scale(), 1.0);
}

// ─── the strict install ────────────────────────────────────────────────────────────────────────

/// **A file that matched nothing is an error — per spec, not in aggregate.**
///
/// The aggregate form returns `Ok` for `[good, junk]`, silently ignoring the junk file, which
/// reaches a user as "my second LoRA did nothing, and there was no error". The MLX lane shipped that
/// bug and fixed it in review; this pins the candle lane against it, with junk BOTH after and before
/// the good file so an order-dependent check cannot pass.
#[test]
fn a_file_that_folds_nothing_is_refused_per_spec_not_in_aggregate() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().unwrap();
    let good = write_lora(dir.path(), "good.safetensors", &cfg, Some("8"));

    // A file with NO recognized LoRA suffix at all — it contributes neither an `applied` nor an
    // `unmatched_path`, so only a per-spec check can see it.
    let junk = dir.path().join("junk.safetensors");
    write_safetensors(
        &junk,
        &[("some.merged.weight".to_string(), tensor(&[2, 2], 1.0))],
        &[("format", "pt")],
    );

    // Control: the good file alone installs.
    let mut dit = tiny_dit(&cfg);
    apply_minimax_h3_adapters(&mut dit, &[spec(good.clone(), 1.0)]).expect("the control");

    for (specs, label) in [
        (
            vec![spec(good.clone(), 1.0), spec(junk.clone(), 1.0)],
            "junk after good",
        ),
        (
            vec![spec(junk.clone(), 1.0), spec(good.clone(), 1.0)],
            "junk before good",
        ),
        (vec![spec(junk.clone(), 1.0)], "junk alone"),
    ] {
        let mut dit = tiny_dit(&cfg);
        let err = apply_minimax_h3_adapters(&mut dit, &specs)
            .expect_err(&format!("{label}: a file matching nothing must be refused"))
            .to_string();
        assert!(
            err.contains("no target modules matched"),
            "{label}: wrong error: {err}"
        );
        assert!(err.contains("junk.safetensors"), "{label}: {err}");
    }
}

/// A target that resolves to no module is surfaced by NAME, never silently dropped.
#[test]
fn an_unmatched_target_is_surfaced_by_name() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stray.safetensors");
    let (out, inn) = target_shape(&cfg, PROBE);
    write_safetensors(
        &path,
        &[
            // One real target, so the per-spec zero-match guard is NOT what fires here.
            (
                format!("{PROBE}.lora_A.default.weight"),
                bf16(&tensor(&[PUBLISHED_RANK, inn], 1.0)),
            ),
            (
                format!("{PROBE}.lora_B.default.weight"),
                bf16(&tensor(&[out, PUBLISHED_RANK], 2.0)),
            ),
            // ...and one that names a block this DiT does not have.
            (
                "transformer_blocks.77.attn.to_q.lora_A.default.weight".to_string(),
                bf16(&tensor(&[PUBLISHED_RANK, inn], 3.0)),
            ),
            (
                "transformer_blocks.77.attn.to_q.lora_B.default.weight".to_string(),
                bf16(&tensor(&[out, PUBLISHED_RANK], 4.0)),
            ),
        ],
        &[("alpha", "8")],
    );

    let mut dit = tiny_dit(&cfg);
    let err = apply_minimax_h3_adapters(&mut dit, &[spec(path, 1.0)])
        .expect_err("an unmatched target must be an error")
        .to_string();
    assert!(err.contains("matched no module"), "{err}");
    assert!(err.contains("transformer_blocks.77.attn.to_q"), "{err}");
}

/// LoKr is refused **by file content**, not only by the declared `AdapterKind`. A file carrying
/// `lokr_*` factors while claiming to be a LoRA must not reach the `(x·A)·B` path.
#[test]
fn a_lokr_file_is_refused_even_when_declared_as_lora() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lokr.safetensors");
    write_safetensors(
        &path,
        &[
            (format!("{PROBE}.lokr_w1"), bf16(&tensor(&[8, 8], 1.0))),
            (format!("{PROBE}.lokr_w2"), bf16(&tensor(&[12, 8], 2.0))),
        ],
        &[("format", "pt")],
    );
    let mut dit = tiny_dit(&cfg);
    // Declared as a LoRA — the file is what refuses it.
    let err = apply_minimax_h3_adapters(&mut dit, &[spec(path, 1.0)])
        .expect_err("a lokr file must be refused")
        .to_string();
    assert!(err.contains("LoKr is not supported"), "{err}");
}

/// **The PEFT `lora_adapter_metadata` blob is the middle link of the alpha chain — and its `r` is a
/// cross-check, never an override.**
///
/// The MLX twin has enforced both halves since sc-18724; this lane gained the blob leg with the
/// sc-19443 review fix, because the ComfyUI conversion has to resolve the *same* chain the install
/// does, and a chain that differed per backend would make the quant tier decide the picture.
///
/// The shared loaders take `cfg_rank.unwrap_or(factor_rank)`, so a `{"r": 8}` blob over rank-128
/// factors folds at `8/8 = 1.0` instead of `8/128 = 0.0625` — the same 16× overshoot this module
/// exists to close, arriving through a different door. PEFT writes `r` equal to the factor rank, so
/// the consistent arm shows the check rejects nothing legitimate.
#[test]
fn a_peft_blob_supplies_the_alpha_and_its_rank_is_only_a_cross_check() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().unwrap();
    let x = tensor(&[1, 5, cfg.hidden_size], 0.7);

    let bad = write_lora_with_meta(
        dir.path(),
        "blob_r8.safetensors",
        &cfg,
        None,
        &[(
            "lora_adapter_metadata",
            r#"{"r": 8, "lora_alpha": 8, "peft_type": "LORA"}"#,
        )],
    );
    let mut dit = tiny_dit(&cfg);
    let msg = apply_minimax_h3_adapters(&mut dit, &[spec(bad, 1.0)])
        .expect_err("a blob rank that disagrees with the factors must be refused")
        .to_string();
    assert!(msg.contains("lora_adapter_metadata"), "got {msg}");
    assert!(msg.contains("declares rank 8"), "got {msg}");
    assert!(msg.contains("rank 128"), "got {msg}");
    assert!(msg.contains("shapes are authoritative"), "got {msg}");

    // A CONSISTENT blob is honored and supplies the alpha. `lora_alpha` is 128 here, NOT 8, so this
    // cannot pass by falling through to `DEFAULT_LORA_ALPHA`: it must match the `alpha = "128"`
    // sibling file (fold 1.0) and differ 16× from the no-alpha one.
    let blob = write_lora_with_meta(
        dir.path(),
        "blob_r128.safetensors",
        &cfg,
        None,
        &[("lora_adapter_metadata", r#"{"r": 128, "lora_alpha": 128}"#)],
    );
    let top = write_lora(dir.path(), "top_alpha128.safetensors", &cfg, Some("128"));
    let r_blob = probe_residual(&cfg, &blob, 1.0, &x);
    let r_top = probe_residual(&cfg, &top, 1.0, &x);
    let drift = rel_max_abs(&r_blob, &r_top);
    println!("[peft blob] rel-max-abs vs top-level alpha=128 = {drift:.3e}");
    assert!(
        drift < 1e-6,
        "a consistent PEFT blob must fold at alpha 128 / rank 128 like the top-level stamp; got \
         {drift:.3e}"
    );
    let none = write_lora(dir.path(), "blob_none.safetensors", &cfg, None);
    let ratio = max_abs(&probe_residual(&cfg, &none, 1.0, &x)) / max_abs(&r_blob);
    assert!(
        (ratio - 0.0625).abs() < 1e-5,
        "…and the blob alpha must WIN over DEFAULT_LORA_ALPHA, which would have given 1.0; got \
         {ratio}"
    );

    // An in-band `.alpha` outranks BOTH file-level sources — the chain's first link, pinned inside
    // one file so the ordering is asserted rather than inferred from two separate arms.
    let both = dir.path().join("inband_over_blob.safetensors");
    let mut arrays: Vec<(String, Tensor)> = Vec::new();
    for target in adapter_target_paths(&cfg) {
        let (o, i) = target_shape(&cfg, &target);
        let seed = target.len() as f32;
        arrays.push((
            format!("{target}.lora_A.default.weight"),
            bf16(&tensor(&[PUBLISHED_RANK, i], seed)),
        ));
        arrays.push((
            format!("{target}.lora_B.default.weight"),
            bf16(&tensor(&[o, PUBLISHED_RANK], seed + 0.5)),
        ));
        arrays.push((
            format!("{target}.alpha"),
            Tensor::new(&[128.0f32], &dev()).unwrap(),
        ));
    }
    write_safetensors(
        &both,
        &arrays,
        &[
            ("format", "pt"),
            ("alpha", "8"),
            ("lora_adapter_metadata", r#"{"r": 128, "lora_alpha": 8}"#),
        ],
    );
    let drift = rel_max_abs(&probe_residual(&cfg, &both, 1.0, &x), &r_top);
    assert!(
        drift < 1e-6,
        "an in-band .alpha of 128 must outrank BOTH a blob lora_alpha of 8 and a top-level alpha \
         of 8; got rel-max-abs {drift:.3e} against the fold-1.0 reference"
    );
}

// ─── sc-19443: the ComfyUI key space ───────────────────────────────────────────────────────────

/// **Where a ComfyUI export stamps the alpha of its fused `attn.qkv_proj`.**
///
/// The fixture generator used to emit the in-band `.alpha` tensor and nothing else, which made the
/// AC2 twin-equivalence gate — a well-built gate — span exactly **one of four** spellings. The other
/// three routed around the conversion's block-diagonal `÷3` and folded attention 3× too strong with
/// no error, and the gate could not see any of them. The generator is parameterized over this so
/// every arm below runs four times.
///
/// The `__metadata__` spelling is not a hypothetical: it is what **every published lightx2v file**
/// uses, and none of them carries an in-band `.alpha` at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AlphaSpelling {
    /// A per-target `.alpha` tensor beside the factors — the kohya / ComfyUI in-band convention,
    /// and the only spelling the original fixture generator could write.
    InBand,
    /// A PEFT `lora_adapter_metadata` JSON blob in the top-level `__metadata__` (sc-5374 / sc-5513).
    PeftBlob,
    /// A bare top-level `__metadata__["alpha"]` string — the lightx2v spelling.
    TopLevelMetadata,
    /// No alpha anywhere; resolution must land on `DEFAULT_LORA_ALPHA` **and still divide it**.
    Absent,
}

/// Every spelling, so an arm cannot quietly cover three of four.
const ALPHA_SPELLINGS: [AlphaSpelling; 4] = [
    AlphaSpelling::InBand,
    AlphaSpelling::PeftBlob,
    AlphaSpelling::TopLevelMetadata,
    AlphaSpelling::Absent,
];

/// The alpha the fused `qkv_proj` declares, in whichever spelling is under test.
///
/// **48, not the published 24.** `24/3 = 8` is `DEFAULT_LORA_ALPHA`, so the published pairing cannot
/// distinguish "divided the resolved alpha" from "dropped the alpha and defaulted". `48/3 = 16`
/// distinguishes them.
const FUSED_ALPHA: f32 = 48.0;

/// The in-band alpha the twin's `mlp.fc1` carries, in **every** arm.
///
/// `fc1` is not fused, so its alpha is never divided and its spelling is not what is under test. It
/// is pinned in-band — where it outranks every file-level source — so that varying the *qkv*
/// spelling changes exactly one thing. Non-default, and different from every qkv value, so a
/// conversion that crossed the two alphas over is visible rather than symmetric.
const FC1_ALPHA: f32 = 32.0;

/// What the converted per-target `.alpha` **must** be: the resolved fused alpha, divided by three
/// exactly when the block-diagonal un-fuse divided the rank by three.
///
/// Written out here rather than read back from the converter, so this is an independent statement of
/// the rule and not a restatement of the implementation.
fn expected_target_alpha(spelling: AlphaSpelling, block_diagonal: bool) -> f32 {
    let resolved = match spelling {
        AlphaSpelling::Absent => DEFAULT_LORA_ALPHA,
        _ => FUSED_ALPHA,
    };
    if block_diagonal {
        resolved / 3.0
    } else {
        resolved
    }
}

/// How a ComfyUI twin declares its fused `attn.qkv_proj`: the alpha, **where** that alpha lives, and
/// which of the two legitimate fused shapes to write.
#[derive(Clone, Copy, Debug)]
struct FusedQkvSpec {
    alpha: f32,
    spelling: AlphaSpelling,
    block_diagonal: bool,
}

impl FusedQkvSpec {
    /// The standard fixture: [`FUSED_ALPHA`] in `spelling`, at the chosen fused shape.
    fn new(spelling: AlphaSpelling, block_diagonal: bool) -> Self {
        Self {
            alpha: FUSED_ALPHA,
            spelling,
            block_diagonal,
        }
    }

    /// The same, at an explicit alpha — for the arms that pin the published `24 → 8` pairing.
    fn at(alpha: f32, block_diagonal: bool) -> Self {
        Self {
            alpha,
            spelling: AlphaSpelling::InBand,
            block_diagonal,
        }
    }
}

/// Build a **ComfyUI** twin of a diffusers module set, at one block, from the same underlying
/// factors — so the two files describe the *same* adapter and any difference in the folded result is
/// the conversion's fault.
///
/// `fused.block_diagonal` selects which of the two legitimate fused forms to write;
/// `fused.spelling` selects **where the fused alpha lives**, which is the axis the AC2 gate was
/// blind along.
fn write_comfyui_twin(
    dir: &Path,
    name: &str,
    cfg: &MiniMaxH3DitConfig,
    fused: FusedQkvSpec,
    qkv: &[(Tensor, Tensor); 3],
    fc1: (&Tensor, &Tensor),
) -> PathBuf {
    let FusedQkvSpec {
        alpha,
        spelling,
        block_diagonal,
    } = fused;
    let path = dir.join(name);
    let r = PUBLISHED_RANK;
    let out = cfg.inner_dim();
    let mut arrays: Vec<(String, Tensor)> = Vec::new();

    if block_diagonal {
        // A `[3r, in]` vertical concat, and a `[3·out, 3r]` block-diagonal B.
        let a = Tensor::cat(&[&qkv[0].0, &qkv[1].0, &qkv[2].0], 0).unwrap();
        let zero =
            |rows: usize, cols: usize| Tensor::zeros((rows, cols), DType::BF16, &dev()).unwrap();
        let rows: Vec<Tensor> = (0..3)
            .map(|i| {
                let parts: Vec<Tensor> = (0..3)
                    .map(|j| {
                        if i == j {
                            qkv[i].1.clone()
                        } else {
                            zero(out, r)
                        }
                    })
                    .collect();
                Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1).unwrap()
            })
            .collect();
        let b = Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0).unwrap();
        arrays.push(("blocks.0.attn.qkv_proj.lora_A.weight".into(), a));
        arrays.push(("blocks.0.attn.qkv_proj.lora_B.weight".into(), b));
    } else {
        // Shared `A`: one `[r, in]` down factor and a `[3·out, r]` stacked B.
        arrays.push((
            "blocks.0.attn.qkv_proj.lora_A.weight".into(),
            qkv[0].0.clone(),
        ));
        let b = Tensor::cat(&[&qkv[0].1, &qkv[1].1, &qkv[2].1], 0).unwrap();
        arrays.push(("blocks.0.attn.qkv_proj.lora_B.weight".into(), b));
    }
    // The fused alpha, written into exactly ONE of the four places it can live.
    if spelling == AlphaSpelling::InBand {
        arrays.push((
            "blocks.0.attn.qkv_proj.alpha".into(),
            Tensor::new(&[alpha], &dev()).unwrap(),
        ));
    }

    // `mlp.fc1` carries the SwiGLU halves the OTHER way round — `[gate | value]` where the DiT is
    // `[value | gate]` — so the twin's B is the diffusers B with its row halves swapped.
    let (fc1_a, fc1_b) = fc1;
    let half = fc1_b.dim(0).unwrap() / 2;
    let swapped = Tensor::cat(
        &[
            &fc1_b.narrow(0, half, half).unwrap(),
            &fc1_b.narrow(0, 0, half).unwrap(),
        ],
        0,
    )
    .unwrap();
    arrays.push(("blocks.0.mlp.fc1.lora_A.weight".into(), fc1_a.clone()));
    arrays.push(("blocks.0.mlp.fc1.lora_B.weight".into(), swapped));
    // `fc1` is unfused, so its alpha is never divided — pinned in-band in every arm so the only
    // thing `spelling` varies is the fused module's.
    arrays.push((
        "blocks.0.mlp.fc1.alpha".into(),
        Tensor::new(&[FC1_ALPHA], &dev()).unwrap(),
    ));

    let blob = format!(r#"{{"lora_alpha": {alpha}, "r": {PUBLISHED_RANK}}}"#);
    let alpha_str = alpha.to_string();
    let mut meta: Vec<(&str, &str)> = vec![("target_format", "ComfyUI generic LoRA")];
    match spelling {
        // A contradictory file-level value makes the precedence claim non-vacuous: the per-target
        // tensor must win independently for qkv and fc1.
        AlphaSpelling::InBand => meta.push(("alpha", "3")),
        // `r` equals the per-target factor rank on BOTH fused forms (a block-diagonal `[3r, in]` A
        // splits into three rank-`r` ones), so this blob is self-consistent and is not rejected by
        // the rank cross-check — it is the alpha, and only the alpha, that is under test.
        AlphaSpelling::PeftBlob => meta.push(("lora_adapter_metadata", &blob)),
        AlphaSpelling::TopLevelMetadata => meta.push(("alpha", &alpha_str)),
        AlphaSpelling::Absent => {}
    }
    write_safetensors(&path, &arrays, &meta);
    path
}

/// Re-key the small raw-module fixture into the exact trainer spelling. Production validates the
/// full 50×4 census before this rewrite; this weights-light fixture isolates the numerical seam.
fn trainer_keys_from_comfyui(tensors: &HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    tensors
        .iter()
        .map(|(source, tensor)| {
            let target = source
                .replace("blocks.0.attn.qkv_proj", "lora_unet_blocks_0_attn_qkv_proj")
                .replace("blocks.0.mlp.fc1", "lora_unet_blocks_0_mlp_fc1")
                .replace(".lora_A.weight", ".lora_down.weight")
                .replace(".lora_B.weight", ".lora_up.weight");
            (target, tensor.clone())
        })
        .collect()
}

/// Write the **diffusers** counterpart of [`write_comfyui_twin`]'s modules, from the same factors.
///
/// Every target carries an explicit in-band `.alpha`: `qkv_alpha` on q/k/v and [`FC1_ALPHA`] on the
/// feed-forward input. The header's `alpha` is deliberately a value **no target should ever reach**
/// — if the control itself started falling through to a file-level alpha, its residual would move
/// and the comparison would stop meaning what it claims.
fn write_diffusers_twin(
    dir: &Path,
    name: &str,
    qkv_alpha: f32,
    qkv: &[(Tensor, Tensor); 3],
    fc1: (&Tensor, &Tensor),
) -> PathBuf {
    let path = dir.join(name);
    let mut arrays: Vec<(String, Tensor)> = Vec::new();
    for (i, n) in ["to_q", "to_k", "to_v"].iter().enumerate() {
        arrays.push((
            format!("transformer_blocks.0.attn.{n}.lora_A.default.weight"),
            qkv[i].0.clone(),
        ));
        arrays.push((
            format!("transformer_blocks.0.attn.{n}.lora_B.default.weight"),
            qkv[i].1.clone(),
        ));
        arrays.push((
            format!("transformer_blocks.0.attn.{n}.alpha"),
            Tensor::new(&[qkv_alpha], &dev()).unwrap(),
        ));
    }
    arrays.push((
        "transformer_blocks.0.ff.net.0.proj.lora_A.default.weight".into(),
        fc1.0.clone(),
    ));
    arrays.push((
        "transformer_blocks.0.ff.net.0.proj.lora_B.default.weight".into(),
        fc1.1.clone(),
    ));
    arrays.push((
        "transformer_blocks.0.ff.net.0.proj.alpha".into(),
        Tensor::new(&[FC1_ALPHA], &dev()).unwrap(),
    ));
    write_safetensors(&path, &arrays, &[("alpha", "1")]);
    path
}

/// The factors both twins are built from — deterministic and distinct per projection, so a
/// conversion that mixes up q/k/v is observable rather than symmetric.
fn twin_factors(cfg: &MiniMaxH3DitConfig) -> ([(Tensor, Tensor); 3], (Tensor, Tensor)) {
    let r = PUBLISHED_RANK;
    let qkv = [
        (
            bf16(&tensor(&[r, cfg.hidden_size], 1.0)),
            bf16(&tensor(&[cfg.inner_dim(), r], 2.0)),
        ),
        (
            bf16(&tensor(&[r, cfg.hidden_size], 3.0)),
            bf16(&tensor(&[cfg.inner_dim(), r], 4.0)),
        ),
        (
            bf16(&tensor(&[r, cfg.hidden_size], 5.0)),
            bf16(&tensor(&[cfg.inner_dim(), r], 6.0)),
        ),
    ];
    let fc1 = (
        bf16(&tensor(&[r, cfg.hidden_size], 7.0)),
        bf16(&tensor(&[2 * cfg.ffn_dim, r], 8.0)),
    );
    (qkv, fc1)
}

/// The exact trainer spelling reaches independently constructed diffusers-shaped factors.
/// Q/K/V factors are position-distinct, FC1 halves are distinct, and qkv/fc1 alphas disagree, so a
/// wrong slice, either half-order swap, or crossed/default alpha independently makes this red.
#[test]
fn trainer_namespace_converts_to_independent_expected_factors_at_relative_max_abs() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let (qkv, fc1) = twin_factors(&cfg);
    let comfy_path = write_comfyui_twin(
        dir.path(),
        "trainer_source.safetensors",
        &cfg,
        FusedQkvSpec::new(AlphaSpelling::InBand, false),
        &qkv,
        (&fc1.0, &fc1.1),
    );
    let adapter = candle_gen::train::merge::read_adapter(&comfy_path).expect("raw fixture");
    let trainer = trainer_keys_from_comfyui(&adapter.tensors);
    let (raw, alphas) =
        unflatten_minimax_h3_trainer_tensors(&trainer).expect("exact trainer unflatten");
    assert_eq!(alphas, vec![FC1_ALPHA, FUSED_ALPHA]);

    let got = convert_comfyui_key_space(&raw, &adapter.meta).expect("trainer conversion");
    assert_eq!(got.len(), 12, "q/k/v and fc1 each carry A/B/alpha");
    for (index, name) in ["to_q", "to_k", "to_v"].iter().enumerate() {
        for (suffix, expected) in [
            ("lora_A.weight", &qkv[0].0),
            ("lora_B.weight", &qkv[index].1),
        ] {
            let key = format!("transformer_blocks.0.attn.{name}.{suffix}");
            let drift = rel_max_abs(got.get(&key).unwrap(), expected);
            assert!(
                drift <= 1e-6,
                "{key}: trainer conversion drifted at relative max-abs {drift:.3e}"
            );
        }
        let key = format!("transformer_blocks.0.attn.{name}.alpha");
        let expected = Tensor::new(&[FUSED_ALPHA], &dev()).unwrap();
        let drift = rel_max_abs(got.get(&key).unwrap(), &expected);
        assert!(
            drift <= 1e-6,
            "{key}: trainer/raw conversion drifted at relative max-abs {drift:.3e}"
        );
    }
    for (suffix, expected) in [("lora_A.weight", &fc1.0), ("lora_B.weight", &fc1.1)] {
        let key = format!("transformer_blocks.0.ff.net.0.proj.{suffix}");
        let drift = rel_max_abs(got.get(&key).unwrap(), expected);
        assert!(drift <= 1e-6, "{key}: relative max-abs {drift:.3e}");
    }
    let expected = Tensor::new(&[FC1_ALPHA], &dev()).unwrap();
    let drift = rel_max_abs(
        got.get("transformer_blocks.0.ff.net.0.proj.alpha").unwrap(),
        &expected,
    );
    assert!(drift <= 1e-6, "FC1 alpha drifted by {drift:.3e}");
}

/// **The detection is unchanged** — sc-19443 changed what happens after it, never the guard itself.
/// A file that reaches the fold path in the wrong key space is a silent-corruption bug.
#[test]
fn the_comfyui_key_space_is_still_detected_and_the_diffusers_one_is_not() {
    assert!(is_comfyui_key_space([
        "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"
    ]));
    assert!(is_comfyui_key_space([
        "diffusion_model.blocks.0.mlp.fc1.lora_A.weight"
    ]));
    assert!(is_comfyui_key_space([
        "diffusion_model.blocks.0.attn.out_proj.lora_B.weight"
    ]));
    assert!(
        !is_comfyui_key_space([
            "transformer_blocks.0.attn.to_out.0.lora_A.default.weight",
            "transformer_blocks.0.ff.net.0.proj.lora_B.default.weight",
        ]),
        "the diffusers export must not trip the comfyui guard"
    );
}

/// **A converted ComfyUI file folds to the SAME residual as its diffusers twin**, gated on relative
/// max-abs-diff.
///
/// That equivalence is the only honest gate for this conversion, and it is what makes each of the
/// three transforms individually load-bearing: un-fusing `qkv_proj` wrong mixes q/k/v, dropping the
/// alpha division folds 3× strong, and skipping the SwiGLU swap computes
/// `w2(silu(value)·gate)` — the sc-18740 defect, which shipped green at cosine 0.73–0.78.
///
/// **Both fused forms are covered.** Block-diagonal is the lightx2v twins' shape; shared-`A` is what
/// a LoRA trained natively on the fused module looks like. The two are told apart by measuring the
/// bytes, so a converter that assumed one form would get the other's rank — and therefore its fold —
/// wrong.
///
/// **And all four alpha spellings are covered**, which is the sc-19443 review's finding. The gate
/// itself was sound; its fixture generator only ever wrote the in-band `.alpha` tensor, so the
/// `__metadata__`, PEFT-blob and no-alpha routes — the first of which is the ONLY spelling any
/// published file for this family uses — never reached the block-diagonal `÷3` and folded attention
/// 3× too strong at rel-max-abs `2.007e0`, installing `Ok`.
///
/// Each spelling is **its own `#[test]`** (see the generated arms below) rather than an inner loop,
/// so a regression names the route it broke instead of only the gate, and a mutation can be shown to
/// red one route while leaving the others green.
fn assert_twin_equivalence(spelling: AlphaSpelling) {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().unwrap();
    let (qkv, fc1) = twin_factors(&cfg);

    // **The shared-`A` form describes a different adapter**, so its twin must too: all three
    // projections contract through ONE down factor there, where the block-diagonal form carries
    // three independent ones. Comparing the shared-`A` conversion against the three-`A` twin would
    // be comparing two different adapters and would red for a reason that is not the converter's.
    let shared_qkv = [
        (qkv[0].0.clone(), qkv[0].1.clone()),
        (qkv[0].0.clone(), qkv[1].1.clone()),
        (qkv[0].0.clone(), qkv[2].1.clone()),
    ];

    for (block_diagonal, twin_qkv, form) in [
        (true, &qkv, "block-diagonal (the lightx2v twin shape)"),
        (false, &shared_qkv, "shared-A (natively fused training)"),
    ] {
        // The diffusers control declares, per target, exactly what the conversion must arrive at:
        // `FUSED_ALPHA/3` (or `DEFAULT_LORA_ALPHA/3`) on a block-diagonal un-fuse, and the undivided
        // value when the rank was not split.
        let want_alpha = expected_target_alpha(spelling, block_diagonal);
        let label = format!("{form} / {spelling:?} (per-target alpha {want_alpha})");
        let diffusers = write_diffusers_twin(
            dir.path(),
            &format!("twin_{block_diagonal}_{spelling:?}.safetensors"),
            want_alpha,
            twin_qkv,
            (&fc1.0, &fc1.1),
        );
        let comfy = write_comfyui_twin(
            dir.path(),
            &format!("comfy_{block_diagonal}_{spelling:?}.safetensors"),
            &cfg,
            FusedQkvSpec::new(spelling, block_diagonal),
            twin_qkv,
            (&fc1.0, &fc1.1),
        );

        for probe in [
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.0.attn.to_k",
            "transformer_blocks.0.attn.to_v",
            "transformer_blocks.0.ff.net.0.proj",
        ] {
            let segs: Vec<&str> = probe.split('.').collect();
            let (_, inn) = target_shape(&cfg, probe);
            let x = tensor(&[1, 4, inn], 0.31);

            // The bare base is the same for both files, so it is computed once per probe.
            let mut base = tiny_dit(&cfg);
            let y0 = base.adaptable_mut(&segs).unwrap().forward(&x).unwrap();
            let residual = |file: &Path| -> Tensor {
                let mut adapted = tiny_dit(&cfg);
                apply_minimax_h3_adapters(&mut adapted, &[spec(file.to_path_buf(), 1.0)])
                    .expect("install");
                let y1 = adapted.adaptable_mut(&segs).unwrap().forward(&x).unwrap();
                (y1 - &y0).unwrap()
            };

            let want = residual(&diffusers);
            let got = residual(&comfy);
            assert!(
                max_abs(&want) > 1e-4,
                "{probe}: the twin's own residual must be non-trivial, else the comparison is \
                 vacuous"
            );
            let drift = rel_max_abs(&got, &want);
            // The **fold ratio** is the quantity the defect is stated in — an undivided alpha on a
            // block-diagonal split reads 3.007×, and the fix reads 1.000×. Printed alongside the
            // gated relative max-abs-diff (never cosine: a pure fold-ratio error is exactly a scale
            // error, and cosine is scale-invariant).
            let ratio = max_abs(&got) / max_abs(&want);
            println!(
                "[{label}] {probe}: fold ratio {ratio:.4}x, converted-vs-twin rel-max-abs = \
                 {drift:.3e}"
            );
            assert!(
                drift < 1e-2,
                "{label} / {probe}: a converted ComfyUI file must fold like its diffusers twin \
                 whatever spelling carried its alpha; got fold ratio {ratio:.4}x, rel-max-abs \
                 {drift:.3e}"
            );
        }
    }
}

/// One `#[test]` per alpha spelling, for both the residual gate and the emitted-alpha gate.
///
/// The point of the split is diagnostic resolution: the sc-19443 blocker made exactly three of the
/// four routes wrong, and a single test covering all four can only say "the gate failed".
macro_rules! per_alpha_spelling {
    ($($name:ident, $chain:ident => $spelling:expr;)+) => {
        /// Exactly the spellings the arms below were generated for — cross-checked against
        /// [`ALPHA_SPELLINGS`] by `every_alpha_spelling_has_generated_arms`.
        const GENERATED_SPELLINGS: &[AlphaSpelling] = &[$($spelling),+];
        $(
            #[test]
            fn $name() {
                assert_twin_equivalence($spelling);
            }

            #[test]
            fn $chain() {
                assert_alpha_resolves_before_the_split($spelling);
            }
        )+
    };
}

per_alpha_spelling! {
    a_converted_comfyui_file_folds_like_its_diffusers_twin_in_band,
        the_in_band_alpha_resolves_before_the_qkv_split => AlphaSpelling::InBand;
    a_converted_comfyui_file_folds_like_its_diffusers_twin_peft_blob,
        the_peft_blob_alpha_resolves_before_the_qkv_split => AlphaSpelling::PeftBlob;
    a_converted_comfyui_file_folds_like_its_diffusers_twin_metadata,
        the_metadata_alpha_resolves_before_the_qkv_split => AlphaSpelling::TopLevelMetadata;
    a_converted_comfyui_file_folds_like_its_diffusers_twin_absent,
        an_absent_alpha_resolves_before_the_qkv_split => AlphaSpelling::Absent;
}

/// **Every `AlphaSpelling` gets generated arms.** A fifth spelling added to the enum breaks the
/// exhaustive `match` below until it is listed in [`ALPHA_SPELLINGS`], and the membership check then
/// forces it into the `per_alpha_spelling!` list too — so the gate cannot silently go back to
/// covering a subset, which is precisely how it missed the `__metadata__` route.
#[test]
fn every_alpha_spelling_has_generated_arms() {
    for s in ALPHA_SPELLINGS {
        match s {
            AlphaSpelling::InBand
            | AlphaSpelling::PeftBlob
            | AlphaSpelling::TopLevelMetadata
            | AlphaSpelling::Absent => {}
        }
        assert!(
            GENERATED_SPELLINGS.contains(&s),
            "{s:?} has no generated twin-equivalence arm"
        );
    }
    assert_eq!(GENERATED_SPELLINGS.len(), ALPHA_SPELLINGS.len());
}

/// **The alpha is resolved through the WHOLE chain before the qkv split, in every spelling.**
///
/// The residual gate above proves the end-to-end fold; this one names the number, so a failure says
/// *which* link broke rather than only that the folds disagree. It reads the `.alpha` the converter
/// emitted, for all four spellings × both fused forms, and it is the guard that would have caught
/// the sc-19443 blocker on its own.
///
/// Three claims per arm:
///
/// * a per-target `.alpha` is emitted **at all** — the old converter emitted none unless the file
///   carried an in-band tensor, and "no alpha" is exactly how the undivided file-level value leaked
///   back in downstream;
/// * its value is the resolved alpha divided by three **iff** the rank was;
/// * `fc1`'s alpha is untouched by the qkv spelling, so the division is not a blanket file-level
///   rescale that happens to land right on q/k/v.
fn assert_alpha_resolves_before_the_split(spelling: AlphaSpelling) {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().unwrap();
    let (qkv, fc1) = twin_factors(&cfg);

    let read = |file: &PathBuf, key: &str| -> f32 {
        let af = candle_gen::train::merge::read_adapter(file).unwrap();
        let converted = convert_comfyui_key_space(&af.tensors, &af.meta).unwrap();
        converted
            .get(key)
            .unwrap_or_else(|| panic!("no converted alpha at {key}"))
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0]
    };

    for block_diagonal in [true, false] {
        let f = write_comfyui_twin(
            dir.path(),
            &format!("chain_{block_diagonal}_{spelling:?}.safetensors"),
            &cfg,
            FusedQkvSpec::new(spelling, block_diagonal),
            &qkv,
            (&fc1.0, &fc1.1),
        );
        let want = expected_target_alpha(spelling, block_diagonal);
        for n in ["to_q", "to_k", "to_v"] {
            let got = read(&f, &format!("transformer_blocks.0.attn.{n}.alpha"));
            assert_eq!(
                got, want,
                "{spelling:?} / block_diagonal={block_diagonal} / {n}: the fused alpha must be \
                 resolved through in-band → PEFT blob → __metadata__ → DEFAULT_LORA_ALPHA and THEN \
                 divided by three iff the rank was; got {got}, want {want}"
            );
        }
        // The unfused feed-forward alpha rides along untouched, in every arm.
        assert_eq!(
            read(&f, "transformer_blocks.0.ff.net.0.proj.alpha"),
            FC1_ALPHA,
            "{spelling:?}: an unfused module's alpha must not be divided"
        );
        // Non-vacuity: on the block-diagonal form no arm may land on the default alpha, or a
        // converter that dropped the alpha entirely would pass. (`FUSED_ALPHA/3 = 16` and
        // `DEFAULT_LORA_ALPHA/3 = 8/3`; neither is `DEFAULT_LORA_ALPHA`.)
        if block_diagonal {
            assert_ne!(
                want, DEFAULT_LORA_ALPHA,
                "{spelling:?}: this arm must not assert the default alpha"
            );
        }
    }
}

/// The published pairing, stated as the invariant the block-diagonal division exists to hold.
#[test]
fn the_alpha_division_holds_alpha_over_rank_fixed_across_the_unfuse() {
    assert_eq!(
        alpha_rank_fold(24.0, 3.0 * PUBLISHED_RANK as f32),
        alpha_rank_fold(8.0, PUBLISHED_RANK as f32),
        "the division exists to hold alpha/rank fixed across the un-fuse"
    );
    assert_eq!(
        alpha_rank_fold(FUSED_ALPHA / 3.0, PUBLISHED_RANK as f32),
        0.125
    );
    assert_ne!(
        alpha_rank_fold(FUSED_ALPHA / 3.0, PUBLISHED_RANK as f32),
        alpha_rank_fold(DEFAULT_LORA_ALPHA, PUBLISHED_RANK as f32),
        "FUSED_ALPHA is chosen so no arm asserts the default fold"
    );
}

/// **The alpha division is the block-diagonal case's, and it lands on a NON-default number.**
///
/// The published `24 → 8` is a trap for a test: `8` is also `DEFAULT_LORA_ALPHA`, so a converter
/// that dropped the alpha entirely would land on the same fold and pass. The first arm therefore
/// uses `48 → 16`, which is neither the default nor the fused value, and asserts the emitted
/// `.alpha` tensor directly. The published pairing is the second arm.
///
/// The shared-`A` arm is the other half of the claim: its rank is NOT divided, so dividing its alpha
/// would fold it three times too weak.
#[test]
fn the_alpha_division_tracks_the_rank_split_and_is_not_the_default() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().unwrap();
    let (qkv, fc1) = twin_factors(&cfg);

    let read_alpha = |file: &PathBuf| -> f32 {
        let af = candle_gen::train::merge::read_adapter(file).unwrap();
        let converted = convert_comfyui_key_space(&af.tensors, &af.meta).unwrap();
        converted
            .get("transformer_blocks.0.attn.to_q.alpha")
            .expect("the converted per-target alpha")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0]
    };

    // ── block-diagonal, a NON-default alpha: 48 / 3 = 16, and 16/128 = 0.125 ≠ DEFAULT's 0.0625.
    let f48 = write_comfyui_twin(
        dir.path(),
        "a48.safetensors",
        &cfg,
        FusedQkvSpec::at(48.0, true),
        &qkv,
        (&fc1.0, &fc1.1),
    );
    assert_eq!(
        read_alpha(&f48),
        16.0,
        "a block-diagonal un-fuse divides the rank by three, so the alpha must divide by three"
    );
    assert_eq!(alpha_rank_fold(16.0, PUBLISHED_RANK as f32), 0.125);
    assert_ne!(
        alpha_rank_fold(16.0, PUBLISHED_RANK as f32),
        alpha_rank_fold(DEFAULT_LORA_ALPHA, PUBLISHED_RANK as f32),
        "this arm must not land on the default fold, or a converter that dropped the alpha passes"
    );

    // ── the published pairing: 24 / 3 = 8, holding `alpha/rank` at 24/384 == 8/128 == 0.0625.
    let f24 = write_comfyui_twin(
        dir.path(),
        "a24.safetensors",
        &cfg,
        FusedQkvSpec::at(24.0, true),
        &qkv,
        (&fc1.0, &fc1.1),
    );
    assert_eq!(read_alpha(&f24), 8.0);
    assert_eq!(
        alpha_rank_fold(24.0, 3.0 * PUBLISHED_RANK as f32),
        alpha_rank_fold(8.0, PUBLISHED_RANK as f32),
        "the division exists to hold alpha/rank fixed across the un-fuse"
    );

    // ── shared-A keeps rank `r`, so its alpha is UNCHANGED. Dividing here would fold 3x too weak.
    let shared = write_comfyui_twin(
        dir.path(),
        "shared.safetensors",
        &cfg,
        FusedQkvSpec::at(48.0, false),
        &qkv,
        (&fc1.0, &fc1.1),
    );
    assert_eq!(
        read_alpha(&shared),
        48.0,
        "a shared-A fused LoRA is already at per-projection rank; dividing its alpha folds it 3x \
         too weak"
    );
}

/// The conversion renames every ComfyUI module, and the installer re-checks that — a conversion that
/// left a fused module behind must fail rather than half-fold.
#[test]
fn the_conversion_leaves_no_comfyui_module_behind() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().unwrap();
    let (qkv, fc1) = twin_factors(&cfg);
    let comfy = write_comfyui_twin(
        dir.path(),
        "c.safetensors",
        &cfg,
        FusedQkvSpec::at(24.0, true),
        &qkv,
        (&fc1.0, &fc1.1),
    );

    let af = candle_gen::train::merge::read_adapter(&comfy).unwrap();
    assert!(is_comfyui_key_space(af.tensors.keys().map(String::as_str)));
    let converted = convert_comfyui_key_space(&af.tensors, &af.meta).unwrap();
    assert!(
        !is_comfyui_key_space(converted.keys().map(String::as_str)),
        "the converted map must carry no ComfyUI module name at all"
    );
    // The trunk container is renamed too: `blocks.0` is ComfyUI's spelling of `transformer_blocks.0`.
    assert!(converted
        .keys()
        .all(|k| k.starts_with("transformer_blocks.") || k.starts_with("token_refiner.")));
    for want in [
        "transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer_blocks.0.attn.to_k.lora_B.weight",
        "transformer_blocks.0.attn.to_v.alpha",
        "transformer_blocks.0.ff.net.0.proj.lora_B.weight",
    ] {
        assert!(converted.contains_key(want), "missing {want}");
    }
    // The report says the conversion ran — a diffusers file must NOT be counted.
    let mut dit = tiny_dit(&cfg);
    let r = apply_minimax_h3_adapters(&mut dit, &[spec(comfy, 1.0)]).expect("install");
    assert_eq!(r.converted_from_comfyui, 1);

    let plain = write_lora(dir.path(), "plain.safetensors", &cfg, Some("8"));
    let mut dit = tiny_dit(&cfg);
    let r = apply_minimax_h3_adapters(&mut dit, &[spec(plain, 1.0)]).expect("install");
    assert_eq!(
        r.converted_from_comfyui, 0,
        "a diffusers file must not go through the conversion"
    );
}

/// **Block-diagonality is measured on the BYTES, not inferred from the shape.**
///
/// A shared-`A` fused LoRA whose rank happens to divide by three has exactly the shape a
/// block-diagonal one would, so a converter that took `r % 3 == 0` as proof would split it into
/// three rank-`r/3` LoRAs and divide its alpha — folding it three times too weak, on factors that
/// are not the ones it was trained with. The rank here is 96 precisely so the shape test alone
/// cannot tell the two forms apart.
#[test]
fn block_diagonality_is_measured_on_the_bytes_not_inferred_from_the_shape() {
    let cfg = dit_fixture_config();
    let r = 96usize; // divisible by three, so the shape is ambiguous and only the bytes decide
    assert_eq!(r % 3, 0);
    let out = cfg.inner_dim();

    let mut m: HashMap<String, Tensor> = HashMap::new();
    m.insert(
        "blocks.0.attn.qkv_proj.lora_A.weight".into(),
        bf16(&tensor(&[r, cfg.hidden_size], 1.0)),
    );
    // A DENSE `[3·out, r]` B: every entry is non-zero, so it is emphatically not block-diagonal.
    m.insert(
        "blocks.0.attn.qkv_proj.lora_B.weight".into(),
        bf16(&tensor(&[3 * out, r], 2.0)),
    );
    m.insert(
        "blocks.0.attn.qkv_proj.alpha".into(),
        Tensor::new(&[48.0f32], &dev()).unwrap(),
    );

    let converted = convert_comfyui_key_space(&m, &HashMap::new()).unwrap();
    let a = converted
        .get("transformer_blocks.0.attn.to_q.lora_A.weight")
        .expect("converted to_q down factor");
    assert_eq!(
        a.dims()[0],
        r,
        "a shared-A fused LoRA keeps its full rank {r}; splitting it into {} would use factors it \
         was never trained with",
        r / 3
    );
    let alpha = converted
        .get("transformer_blocks.0.attn.to_q.alpha")
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()[0];
    assert_eq!(
        alpha, 48.0,
        "the alpha must NOT be divided when the rank was not split — dividing folds 3x too weak"
    );
    // All three projections share the one down factor, and differ in their up factor.
    for n in ["to_k", "to_v"] {
        let other = converted
            .get(&format!("transformer_blocks.0.attn.{n}.lora_A.weight"))
            .unwrap();
        assert_eq!(other.dims(), a.dims());
    }
    assert_ne!(
        max_abs(
            &(converted
                .get("transformer_blocks.0.attn.to_q.lora_B.weight")
                .unwrap()
                - converted
                    .get("transformer_blocks.0.attn.to_k.lora_B.weight")
                    .unwrap())
            .unwrap()
        ),
        0.0,
        "q and k must take DIFFERENT row blocks of the fused B"
    );
}

/// **The render seam folds the staged adapter onto the DiT the render actually denoises with.**
///
/// Everything else in this file installs onto a DiT the test built itself. This is the one arm that
/// goes through `MiniMaxH3::load` → `load_task_dit`, i.e. the path a real job takes — which is the
/// difference between "the installer works" and "a staged LoRA reaches the render". A provider that
/// loaded the partition and forgot to call the installer passes every other test here.
///
/// The control is the same snapshot loaded with no adapters: it must come back un-adapted, so the
/// residual below is attributable to the spec and not to the fixture.
#[test]
fn the_render_seam_folds_the_staged_adapter_onto_the_dit() {
    let cfg = dit_fixture_config();
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    // `MiniMaxH3::load` validates every component dir before it reads anything — and since
    // sc-17157's review a dir must carry a **shard**, because an empty one loaded clean and failed
    // in the middle of the render. A zero-byte stub satisfies that check, which never opens the
    // file. `transformer/` is deliberately NOT stubbed: it gets the real fixture shard below, and
    // the DiT loader reads **every** `.safetensors` in its partition, so a stub there is a
    // "header too small" parse error rather than a satisfied precondition.
    std::fs::create_dir_all(root.join("transformer")).unwrap();
    for c in ["text_encoder", "vae", "audio_vae"] {
        let dir = root.join(c);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model-00001-of-00001.safetensors"), []).unwrap();
    }
    // The text encoder additionally needs its `config.json` (sc-20267): it is a *tiered* component, so
    // `MiniMaxH3::load` reads the `quantization` marker there to reconcile the staged tier. Dense — no
    // `quantization` block — which is the `bf16` shape this fixture is. `vae`/`audio_vae` are
    // tier-agnostic and are still satisfied by a shard alone.
    std::fs::write(
        root.join("text_encoder").join("config.json"),
        r#"{"num_layers": 50}"#,
    )
    .unwrap();
    // A real `transformer/` partition at the fixture geometry: the committed tensors plus the
    // diffusers `config.json` the loader parses.
    let f = Golden::load(DIT_FIXTURE);
    let map = f.model_map(&["src.", "in.", "out.", "layout."]);
    candle_gen::candle_core::safetensors::save(
        &map,
        root.join("transformer").join("dit.safetensors"),
    )
    .unwrap();
    std::fs::write(
        root.join("transformer").join("config.json"),
        fixture_config_json(&cfg),
    )
    .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let lora = write_lora(dir.path(), "turbo.safetensors", &cfg, Some("8"));

    let segs: Vec<&str> = PROBE.split('.').collect();
    let x = tensor(&[1, 3, cfg.hidden_size], 0.11);

    // Control: no adapters staged ⇒ the seam returns a bare DiT.
    let bare = candle_gen_minimax_h3::model::MiniMaxH3::load(&candle_gen::gen_core::LoadSpec::new(
        candle_gen::gen_core::WeightsSource::Dir(root.clone()),
    ))
    .unwrap();
    let mut bare_dit = bare.load_task_dit("transformer").unwrap();
    assert_eq!(
        bare_dit.adapted_module_count(),
        0,
        "the control must be un-adapted"
    );
    let y0 = bare_dit.adaptable_mut(&segs).unwrap().forward(&x).unwrap();

    // The real path: a spec carrying the adapter.
    let spec_loaded = candle_gen_minimax_h3::model::MiniMaxH3::load(
        &candle_gen::gen_core::LoadSpec::new(candle_gen::gen_core::WeightsSource::Dir(root))
            .with_adapters(vec![spec(lora, 1.0)]),
    )
    .unwrap();
    let mut adapted = spec_loaded.load_task_dit("transformer").unwrap();
    assert_eq!(
        adapted.adapted_module_count(),
        adapter_target_paths(&cfg).len(),
        "EVERY enumerated target must carry a residual after the render seam ran"
    );
    let y1 = adapted.adaptable_mut(&segs).unwrap().forward(&x).unwrap();
    let residual = max_abs(&(y1 - y0).unwrap());
    assert!(
        residual > 1e-4,
        "the staged adapter must CHANGE the render's DiT; got a residual of {residual:.3e}, which \
         is what a provider that accepted the spec and never installed it produces"
    );
}

/// The fixture geometry as the diffusers `config.json` the DiT loader parses.
fn fixture_config_json(cfg: &MiniMaxH3DitConfig) -> String {
    serde_json::json!({
        "num_attention_heads": cfg.num_attention_heads,
        "attention_head_dim": cfg.attention_head_dim,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "num_refiner_layers": cfg.num_refiner_layers,
        "ffn_dim": cfg.ffn_dim,
        "in_channels": cfg.in_channels,
        "audio_in_channels": cfg.audio_in_channels,
        "patch_size": cfg.patch_size.to_vec(),
        "text_dim": cfg.text_dim,
        "freq_dim": cfg.freq_dim,
        "time_embed_hidden_dim": cfg.time_embed_hidden_dim,
        "time_embed_dim": cfg.time_embed_dim,
        "rope_freq_dim": cfg.rope_freq_dim,
        "rope_theta": cfg.rope_theta,
        "norm_eps": cfg.norm_eps,
        "qk_norm_eps": cfg.qk_norm_eps,
        "final_norm_eps": cfg.final_norm_eps,
    })
    .to_string()
}

/// A fused `qkv_proj` whose factors do not compose, or whose `B` is not three equal projections, is
/// an error naming the module — never a guessed split.
#[test]
fn a_malformed_fused_qkv_is_refused_by_name() {
    let cfg = dit_fixture_config();
    let r = PUBLISHED_RANK;
    let out = cfg.inner_dim();

    // Ranks that disagree between A and B.
    let mut m: HashMap<String, Tensor> = HashMap::new();
    m.insert(
        "blocks.0.attn.qkv_proj.lora_A.weight".into(),
        bf16(&tensor(&[3 * r, cfg.hidden_size], 1.0)),
    );
    m.insert(
        "blocks.0.attn.qkv_proj.lora_B.weight".into(),
        bf16(&tensor(&[3 * out, r], 2.0)),
    );
    let err = convert_comfyui_key_space(&m, &HashMap::new())
        .unwrap_err()
        .to_string();
    assert!(err.contains("do not compose"), "{err}");
    assert!(err.contains("qkv_proj"), "{err}");

    // A `B` whose row count is not three equal projections.
    let mut m: HashMap<String, Tensor> = HashMap::new();
    m.insert(
        "blocks.0.attn.qkv_proj.lora_A.weight".into(),
        bf16(&tensor(&[r, cfg.hidden_size], 1.0)),
    );
    m.insert(
        "blocks.0.attn.qkv_proj.lora_B.weight".into(),
        bf16(&tensor(&[3 * out + 1, r], 2.0)),
    );
    let err = convert_comfyui_key_space(&m, &HashMap::new())
        .unwrap_err()
        .to_string();
    assert!(err.contains("three equal q/k/v projections"), "{err}");

    // A lone half-pair.
    let mut m: HashMap<String, Tensor> = HashMap::new();
    m.insert(
        "blocks.0.attn.qkv_proj.lora_A.weight".into(),
        bf16(&tensor(&[r, cfg.hidden_size], 1.0)),
    );
    let err = convert_comfyui_key_space(&m, &HashMap::new())
        .unwrap_err()
        .to_string();
    assert!(err.contains("needs both"), "{err}");
}
