//! sc-18724 — the lightx2v turbo LoRA folds at the **measured** strength, on the **published** key
//! space, over the committed tiny DiT.
//!
//! # What this file is defending
//!
//! `lightx2v/Minimax-h3-Turbo` stamps its LoRA alpha as a bare top-level `__metadata__` string and
//! ships **no** `rank` key, no per-target `.alpha` tensor and no `lora_adapter_metadata` blob. Both
//! pre-existing in-tree alpha paths mishandle that with no error at all — the PEFT one folds an
//! `alpha=8, rank=128` file **16× too strong**, the `parse_rank_alpha` one **128×**. Neither failure
//! has a shape, a checksum or a key-coverage proof that can see it: the render simply comes out
//! wrong. See [`mlx_gen_minimax_h3::adapters`].
//!
//! # Why all three alpha cases are here, not just one
//!
//! The set disagrees *within one repo* — `alpha: "128"` on the 4-step 768p file, `alpha: "8"` on the
//! 8-step and the ref2v ones, and **nothing at all** on `4step_v0.1`. A test covering only the
//! `alpha=128` file would pass against an implementation that hardcodes `1.0`, and a test covering
//! only an `alpha=8` file would pass against one that hardcodes `0.0625`. Every constant-returning
//! implementation fails at least one arm below.
//!
//! The absolute arm is checked against an **independently computed** `x·Aᵀ·Bᵀ` rather than against a
//! second call into the code under test, and the relative arms (`×16`, and absent == declared-8) are
//! ratios, so a shared error in the fold formula cannot cancel out of both.
//!
//! The `#[ignore]`d gate at the bottom is the same claim against the real published files:
//!
//! ```text
//! MINIMAX_H3_TURBO_LORA=<dir of lightx2v/Minimax-h3-Turbo> \
//!   cargo test -p mlx-gen-minimax-h3 --test integration turbo_lora:: -- --ignored --nocapture
//! ```

use crate::common;

use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_rs::ops::{concatenate_axis, matmul, subtract};
use mlx_rs::{Array, Device, DeviceType, Dtype};
use sha2::{Digest, Sha256};

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::gen_core::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::adapters::{
    adapter_target_paths, alpha_rank_fold, apply_minimax_h3_adapters, convert_comfyui_key_space,
    convert_minimax_h3_trainer_key_space, is_comfyui_key_space, resolve_alpha, resolve_rank,
    unflatten_minimax_h3_trainer_tensors, DEFAULT_LORA_ALPHA,
};
use mlx_gen_minimax_h3::{MiniMaxH3Dit, MiniMaxH3DitConfig, SMALLEST_LEGAL_FRAMES};

use common::{dit_fixture_config, DIT_FIXTURE};

/// The published rank — every one of the seven turbo files is rank 128. Used verbatim in the tiny
/// fixtures so the folds asserted below are the *published* numbers (1.0 and 0.0625) and not a
/// scaled-down analogue of them.
const PUBLISHED_RANK: i32 = 128;

/// The module the numeric arms probe. `attn.to_q` is `[96, 64]` at the fixture geometry, so its
/// factors are `A [128, 64]` / `B [96, 128]`.
const PROBE: &str = "transformer_blocks.0.attn.to_q";

/// A deterministic host-built tensor — no GPU RNG stream, so every arm sees identical bytes.
fn tensor(shape: &[i32], seed: f32) -> Array {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let v: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.013 + seed).sin()) * 0.25)
        .collect();
    Array::from_slice(&v, shape)
}

/// Cast to `bf16` — the dtype **every** tensor in **every** published turbo file carries.
fn bf16(a: &Array) -> Array {
    a.as_dtype(Dtype::Bfloat16).expect("cast to bf16")
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

/// Write a tiny **diffusers-key-space** turbo LoRA covering every adaptable module at `cfg`'s
/// geometry, keyed exactly as the published files are — `.lora_A.default.weight` /
/// `.lora_B.default.weight`, no namespace prefix — with `alpha` stamped into the top-level
/// `__metadata__` only when `alpha` is `Some`.
///
/// **The factors are written `bf16`, not `f32`.** All 624 tensors in every published turbo file are
/// bf16, and that is the only dtype under which the install's dtype-matched fold scalar
/// (`scalar(fold).as_dtype(up.dtype())`) has any effect at all — with f32 fixtures the cast is a
/// no-op and the property is unguarded (sc-18724 review). Every fold asserted below is an exact
/// power of two, so bf16 storage costs the assertions no exactness: scaling a bf16 factor by `1.0`
/// or `0.0625` is representable without rounding.
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
    let mut arrays: Vec<(String, Array)> = Vec::new();
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
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".into(), "pt".into());
    meta.insert("floating_dtype".into(), "bfloat16".into());
    if let Some(a) = alpha {
        meta.insert("alpha".into(), a.into());
    }
    for (k, v) in extra {
        meta.insert((*k).into(), (*v).into());
    }
    let entries: Vec<(&str, &Array)> = arrays.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(entries, Some(&meta), &path).expect("write the tiny turbo LoRA");
    path
}

fn spec(path: PathBuf, scale: f32) -> AdapterSpec {
    AdapterSpec::new(path, scale, AdapterKind::Lora)
}

/// The committed tiny DiT — real crate types, real loader, 2 transformer blocks + 2 refiner blocks.
fn tiny_dit(cfg: &MiniMaxH3DitConfig) -> MiniMaxH3Dit {
    let mut w = Weights::from_file(DIT_FIXTURE).expect("the committed DiT fixture");
    MiniMaxH3Dit::from_weights(&mut w, cfg, Dtype::Float32).expect("the whole tiny DiT")
}

/// `y_with_lora(x) − y_base(x)` at [`PROBE`] — the residual the install actually added.
fn probe_residual(cfg: &MiniMaxH3DitConfig, lora: &Path, scale: f32, x: &Array) -> Array {
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

/// **The alpha gate.** All three published spellings, over the committed fixture, at strength 1.0:
/// `alpha=128` folds at exactly 1.0, `alpha=8` at 0.0625, and an absent alpha at 0.0625 — the last
/// through [`DEFAULT_LORA_ALPHA`] and *not* through the rank.
#[test]
fn the_three_published_alphas_fold_at_one_and_one_sixteenth() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let x = tensor(&[3, cfg.hidden_size], 0.7);

    let a128 = write_lora(dir.path(), "alpha128.safetensors", &cfg, Some("128"));
    let a8 = write_lora(dir.path(), "alpha8.safetensors", &cfg, Some("8"));
    let none = write_lora(dir.path(), "noalpha.safetensors", &cfg, None);

    // The alpha=128 file must fold at exactly 1.0 — checked against an INDEPENDENTLY computed
    // `x·Aᵀ·Bᵀ` read straight off the fixture file, not against a second call into the loader.
    let w = Weights::from_file(&a128).expect("read back");
    let a = w
        .require(&format!("{PROBE}.lora_A.default.weight"))
        .unwrap()
        .clone();
    let b = w
        .require(&format!("{PROBE}.lora_B.default.weight"))
        .unwrap()
        .clone();
    assert_eq!(a.shape(), [PUBLISHED_RANK, cfg.hidden_size]);
    assert_eq!(resolve_rank(PROBE, &a, None).unwrap(), 128.0);
    let unscaled = matmul(matmul(&x, a.t()).unwrap(), b.t()).unwrap();

    let r128 = probe_residual(&cfg, &a128, 1.0, &x);
    let scale_ref = max_abs(&unscaled);
    assert!(scale_ref > 1e-3, "the probe residual must be non-trivial");
    let dev = max_abs(&subtract(&r128, &unscaled).unwrap()) / scale_ref;
    println!("[alpha=128] fold 1.0: rel-max-abs vs x·Aᵀ·Bᵀ = {dev:.3e}");
    assert!(
        dev < 1e-6,
        "alpha=128 at rank 128 must fold at exactly 1.0, got rel dev {dev:.3e}"
    );

    // The alpha=8 file must fold at 0.0625 — i.e. exactly one sixteenth of the alpha=128 one. Stated
    // as a RATIO so it cannot pass by sharing a wrong constant with the arm above.
    let r8 = probe_residual(&cfg, &a8, 1.0, &x);
    let sixteenth = max_abs(&r8) / max_abs(&r128);
    println!("[alpha=8]   |residual| / |alpha=128 residual| = {sixteenth:.6}");
    assert!(
        (sixteenth - 0.0625).abs() < 1e-5,
        "alpha=8 at rank 128 must fold 16x weaker than alpha=128, got {sixteenth}"
    );
    let want8 = max_abs(&unscaled) * alpha_rank_fold(8.0, 128.0);
    assert!(
        (max_abs(&r8) - want8).abs() / want8 < 1e-5,
        "alpha=8 must fold at 0.0625 absolutely, not merely relatively"
    );

    // And the file with NO alpha must be byte-identical to the one declaring 8 — the whole point of
    // `DEFAULT_LORA_ALPHA`. Falling back to the rank instead would make this arm equal to r128.
    let rnone = probe_residual(&cfg, &none, 1.0, &x);
    let drift = max_abs(&subtract(&rnone, &r8).unwrap()) / max_abs(&r8);
    println!("[no alpha]  rel-max-abs vs alpha=8 = {drift:.3e}");
    assert!(
        drift < 1e-6,
        "an absent alpha must fall back to DEFAULT_LORA_ALPHA = {DEFAULT_LORA_ALPHA}, not to the \
         rank; got rel dev {drift:.3e} against the declared-8 file"
    );
    let vs_rank_fallback = max_abs(&rnone) / max_abs(&r128);
    assert!(
        (vs_rank_fallback - 0.0625).abs() < 1e-5,
        "…and it must NOT equal the alpha = rank fallback (which would give 1.0), got \
         {vs_rank_fallback}"
    );
}

/// **The fold scalar is dtype-matched — and only an assertion on the INSTALLED adapter can see it.**
///
/// `apply_one_lora` folds `alpha/rank` into `b` with `scalar(fold).as_dtype(up.dtype())`. Dropping
/// the `as_dtype` is numerically **invisible**: every published fold is an exact power of two, so the
/// bf16 and f32 products carry identical values, and every numeric arm above would stay green. What
/// changes is the dtype of `b` — and with it, the dtype the low-rank `(x·A)·B` matmul runs in: f32
/// where the reference runs bf16, a silent fork-parity deviation. (It is *not* a host-widening
/// question: `AdaptableLinear::apply_adapters` narrows every residual to `out.dtype()` before the
/// add, sc-15265.)
///
/// So this test asserts the dtype of the array the loader actually installed, reached through
/// `apply_minimax_h3_adapters` — it never recomputes the multiply, because a test that reimplements
/// the thing under test validates only the reimplementation.
#[test]
fn the_installed_fold_keeps_the_bf16_factor_dtype() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let lora = write_lora(dir.path(), "bf16.safetensors", &cfg, Some("8"));

    // The fixture must really carry the published dtype, or the assertion below is vacuous — this is
    // exactly the hole the f32 fixtures left.
    let w = Weights::from_file(&lora).expect("read back");
    for suffix in ["lora_A", "lora_B"] {
        let f = w
            .require(&format!("{PROBE}.{suffix}.default.weight"))
            .unwrap();
        assert_eq!(
            f.dtype(),
            Dtype::Bfloat16,
            "{suffix}: the fixture must be bf16 like every published turbo tensor"
        );
    }

    let mut dit = tiny_dit(&cfg);
    let report = apply_minimax_h3_adapters(&mut dit, &[spec(lora, 1.0)]).expect("install");
    assert_eq!(report.applied, adapter_target_paths(&cfg).len());
    let segs: Vec<&str> = PROBE.split('.').collect();
    let installed = dit.adaptable_mut(&segs).expect("probe module").adapters();
    assert_eq!(installed.len(), 1, "exactly one residual per module");
    let Adapter::Lora { a, b, scale } = &installed[0] else {
        panic!("the diffusers path must install a LoRA residual, not a LoKr one");
    };
    assert_eq!(*scale, 1.0);
    assert_eq!(
        a.dtype(),
        Dtype::Bfloat16,
        "the down factor is installed untouched, so it keeps its loaded dtype"
    );
    assert_eq!(
        b.dtype(),
        Dtype::Bfloat16,
        "the alpha/rank fold must NOT promote the up factor to f32 — a bare `scalar(fold)` would, \
         running the low-rank matmul in f32 where the reference runs it in bf16"
    );
}

/// The user's `AdapterSpec::scale` multiplies the `alpha/rank` fold rather than replacing it — so a
/// strength-0.5 install of the 8-step file lands at 0.03125, and a strength-0 one is inert.
#[test]
fn the_user_strength_multiplies_the_alpha_fold() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let x = tensor(&[3, cfg.hidden_size], 0.7);
    let a8 = write_lora(dir.path(), "alpha8.safetensors", &cfg, Some("8"));

    let full = max_abs(&probe_residual(&cfg, &a8, 1.0, &x));
    let half = max_abs(&probe_residual(&cfg, &a8, 0.5, &x));
    assert!(
        (half / full - 0.5).abs() < 1e-5,
        "strength must scale the fold, got {}",
        half / full
    );
    let off = max_abs(&probe_residual(&cfg, &a8, 0.0, &x));
    assert_eq!(off, 0.0, "a scale-0 adapter must be exactly inert");
}

/// **Coverage.** Every module of a published-shaped file resolves, on both stacks — including the
/// token refiner, which 24 of the 624 published tensors target and which therefore cannot be a stub.
#[test]
fn every_target_resolves_across_both_stacks() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let lora = write_lora(dir.path(), "cover.safetensors", &cfg, Some("8"));

    let targets = adapter_target_paths(&cfg);
    assert_eq!(
        targets.len(),
        (cfg.num_layers + cfg.num_refiner_layers) as usize * 6
    );
    let mut dit = tiny_dit(&cfg);
    let report = apply_minimax_h3_adapters(&mut dit, &[spec(lora, 1.0)]).expect("install");
    assert_eq!(report.applied, targets.len());
    assert!(report.unmatched_paths.is_empty());

    // Every enumerated path really resolves through the host — the list and the module tree cannot
    // drift apart while this holds.
    for t in &targets {
        let segs: Vec<&str> = t.split('.').collect();
        assert!(dit.adaptable_mut(&segs).is_some(), "unreachable target {t}");
    }
    assert_eq!(
        targets
            .iter()
            .filter(|t| t.starts_with("token_refiner."))
            .count(),
        (cfg.num_refiner_layers * 6) as usize
    );

    // Modules that are deliberately NOT on the surface: the AdaLN projection (evicted mid-render),
    // the norms, and the 17 mixed-precision heads.
    for absent in [
        "transformer_blocks.0.adaln_proj.linear",
        "transformer_blocks.0.norm1",
        "transformer_blocks.0.attn.norm_q",
        "proj_in",
        "context_embedder",
        "proj_out",
        "token_refiner.final_norm",
    ] {
        let segs: Vec<&str> = absent.split('.').collect();
        assert!(
            dit.adaptable_mut(&segs).is_none(),
            "{absent} must not be an adapter target"
        );
    }
}

/// A key that resolves to no module is an **error**, not a quiet partial fold.
#[test]
fn an_unmatched_target_is_loud() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let path = dir.path().join("stray.safetensors");
    let a = tensor(&[8, cfg.hidden_size], 1.0);
    let b = tensor(&[cfg.hidden_size, 8], 2.0);
    // One real target so the install is not rejected for matching nothing, plus one stray.
    let entries: Vec<(&str, &Array)> = vec![
        ("transformer_blocks.0.attn.to_q.lora_A.default.weight", &a),
        ("transformer_blocks.0.attn.to_q.lora_B.default.weight", &b),
        (
            "transformer_blocks.0.adaln_proj.linear.lora_A.default.weight",
            &a,
        ),
        (
            "transformer_blocks.0.adaln_proj.linear.lora_B.default.weight",
            &b,
        ),
    ];
    Array::save_safetensors(entries, None, &path).expect("write");

    let mut dit = tiny_dit(&cfg);
    let err = apply_minimax_h3_adapters(&mut dit, &[spec(path, 1.0)])
        .expect_err("an unmatched target must fail the install");
    let msg = err.to_string();
    assert!(msg.contains("adaln_proj"), "got {msg}");
    assert!(msg.contains("matched no module"), "got {msg}");
}

/// `supports_lokr` is **reachable**, not merely declared: a genuine LoKr file (`networkType=lokr`
/// plus `lokr_w1`/`lokr_w2` factors) routes to the shared LyCORIS seam on this same host and
/// installs. Routing is by the FILE, which is what keeps the turbo LoRAs — an `alpha` string with no
/// `networkType` — off `parse_rank_alpha` and its 128x fold.
#[test]
fn a_genuine_lokr_routes_to_the_shared_seam_and_the_turbo_files_do_not() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let path = dir.path().join("lokr.safetensors");
    // `kron(w1 [a,c], w2 [b,d])` reshapes to the base's `[96, 64]`: a·b = 8·12, c·d = 8·8.
    let w1 = tensor(&[8, 8], 1.0);
    let w2 = tensor(&[12, 8], 2.0);
    let meta: HashMap<String, String> = [
        ("networkType".to_string(), "lokr".to_string()),
        ("rank".to_string(), "8".to_string()),
        ("alpha".to_string(), "8".to_string()),
    ]
    .into_iter()
    .collect();
    Array::save_safetensors(
        vec![
            ("transformer_blocks.0.attn.to_q.lokr_w1", &w1),
            ("transformer_blocks.0.attn.to_q.lokr_w2", &w2),
        ],
        Some(&meta),
        &path,
    )
    .expect("write the LoKr");

    let mut dit = tiny_dit(&cfg);
    let report = apply_minimax_h3_adapters(&mut dit, &[spec(path, 1.0)]).expect("install the LoKr");
    assert_eq!(report.applied, 1);
    assert!(report.unmatched_paths.is_empty());
    let segs: Vec<&str> = PROBE.split('.').collect();
    assert_eq!(dit.adaptable_mut(&segs).unwrap().adapters().len(), 1);

    // …and every published turbo file must NOT classify as LoKr: a `networkType` stamp or a
    // `lokr_*` key would route it to `parse_rank_alpha`, where its missing `rank` defaults to 1.0.
    let turbo = write_lora(dir.path(), "turbo.safetensors", &cfg, Some("8"));
    let w = Weights::from_file(&turbo).unwrap();
    assert!(
        w.metadata("networkType").is_none(),
        "a turbo LoRA carries no networkType stamp"
    );
    assert!(
        !w.keys().any(|k| k.contains(".lokr_w")),
        "a turbo LoRA carries no LoKr factors"
    );
    assert!(
        w.metadata("rank").is_none(),
        "…and no `rank` key, which is exactly why parse_rank_alpha would default it to 1.0"
    );
}

/// An exact H3 trainer namespace claim owns the whole file: a valid LoKr factor must not be allowed
/// to route around trainer validation and make the malformed mixed file look successfully applied.
#[test]
fn h3_trainer_metadata_cannot_route_mixed_lokr_factors_around_validation() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let path = dir.path().join("mixed-h3-trainer-lokr.safetensors");
    let w1 = tensor(&[8, 8], 1.0);
    let w2 = tensor(&[12, 8], 2.0);
    let meta: HashMap<String, String> = [
        (
            "ss_network_module".to_string(),
            "networks.lora_minimax_h3".to_string(),
        ),
        ("ss_h3_lora_token_refiner".to_string(), "False".to_string()),
        ("rank".to_string(), "8".to_string()),
        ("alpha".to_string(), "8".to_string()),
    ]
    .into_iter()
    .collect();
    Array::save_safetensors(
        vec![
            ("transformer_blocks.0.attn.to_q.lokr_w1", &w1),
            ("transformer_blocks.0.attn.to_q.lokr_w2", &w2),
        ],
        Some(&meta),
        &path,
    )
    .expect("write the mixed adapter");

    let mut dit = tiny_dit(&cfg);
    let err = apply_minimax_h3_adapters(&mut dit, &[spec(path, 1.0)])
        .expect_err("mixed H3 trainer and LoKr namespaces must fail before LoKr application");
    let msg = err.to_string();
    assert!(msg.contains("networks.lora_minimax_h3"), "got {msg}");
    assert!(msg.contains("LoKr"), "got {msg}");
    assert!(msg.contains("mixed"), "got {msg}");
}

/// A file that matches nothing at all names the key space it expected — **per file**, including when
/// it rides alongside a file that matched everything.
///
/// The zero-match check must be per-spec, not an aggregate `report.applied == 0` over the whole
/// list: with an aggregate, `[good, junk]` returns `Ok` and the junk file is silently ignored. The
/// hole is specifically files carrying **no recognized LoRA suffix at all** — a merged LoRA, a base
/// checkpoint, a textual inversion. A wrong-model file whose keys *do* end in `.lora_A.weight` is
/// caught either way, by the unmatched-target guard.
#[test]
fn a_file_matching_nothing_names_the_expected_key_space() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let path = dir.path().join("empty.safetensors");
    let t = tensor(&[2, 2], 1.0);
    // No recognized LoRA suffix at all, so this contributes neither an `applied` nor an
    // `unmatched_path` — only a per-spec count can see it.
    Array::save_safetensors(vec![("some.base.weight", &t)], None, &path).expect("write");
    let mut dit = tiny_dit(&cfg);
    let msg = apply_minimax_h3_adapters(&mut dit, &[spec(path.clone(), 1.0)])
        .expect_err("must fail")
        .to_string();
    assert!(msg.contains("no target modules matched"), "got {msg}");
    assert!(msg.contains("lora_A.default.weight"), "got {msg}");
    assert!(msg.contains("empty.safetensors"), "got {msg}");

    // Two specs, junk SECOND: the good file folds 24 modules, and an aggregate check would return
    // Ok(applied = 24) right here.
    let good = write_lora(dir.path(), "good.safetensors", &cfg, Some("8"));
    assert_eq!(adapter_target_paths(&cfg).len(), 24);
    let mut dit = tiny_dit(&cfg);
    let msg = apply_minimax_h3_adapters(
        &mut dit,
        &[spec(good.clone(), 1.0), spec(path.clone(), 1.0)],
    )
    .expect_err("a junk file riding alongside a good one must still fail")
    .to_string();
    assert!(msg.contains("no target modules matched"), "got {msg}");
    assert!(
        msg.contains("empty.safetensors") && !msg.contains("good.safetensors"),
        "the error must name the OFFENDING file, not the list; got {msg}"
    );

    // …and junk FIRST, so the check cannot be an end-of-loop artifact of spec order.
    let mut dit = tiny_dit(&cfg);
    let msg = apply_minimax_h3_adapters(&mut dit, &[spec(path, 1.0), spec(good.clone(), 1.0)])
        .expect_err("junk first must fail too")
        .to_string();
    assert!(msg.contains("empty.safetensors"), "got {msg}");

    // The control: two GOOD files stack without error, so the per-spec check is not simply rejecting
    // every multi-spec install.
    let good2 = write_lora(dir.path(), "good2.safetensors", &cfg, Some("128"));
    let mut dit = tiny_dit(&cfg);
    let report = apply_minimax_h3_adapters(&mut dit, &[spec(good, 1.0), spec(good2, 1.0)])
        .expect("two good files must stack");
    assert_eq!(report.applied, 2 * adapter_target_paths(&cfg).len());
    let segs: Vec<&str> = PROBE.split('.').collect();
    assert_eq!(dit.adaptable_mut(&segs).unwrap().adapters().len(), 2);
}

/// A PEFT `lora_adapter_metadata` blob whose `r` disagrees with the factor shapes is a **hard
/// error**, exactly like a disagreeing `__metadata__["rank"]` string — never a silent override.
///
/// The shared loader takes `cfg_rank.unwrap_or(factor_rank)`, so a `{"r": 8}` blob over rank-128
/// factors folds at `8/8 = 1.0` instead of `8/128 = 0.0625`: the same 16× overshoot this module
/// exists to close, arriving through a different door. PEFT writes `r` equal to the factor rank, so
/// the consistent arm below shows the check rejects nothing legitimate.
#[test]
fn a_peft_blob_rank_disagreeing_with_the_factors_is_refused() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");

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

    // A CONSISTENT blob is still honored, and still supplies the alpha. `lora_alpha` is 128 here,
    // NOT 8, so the arm cannot pass by falling through to `DEFAULT_LORA_ALPHA`: it must equal the
    // top-level `alpha = "128"` sibling (fold 1.0) and differ 16x from the `alpha = "8"` one.
    // Compared against those sibling files rather than against a recomputed scalar.
    let x = tensor(&[3, cfg.hidden_size], 0.7);
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
    let drift = max_abs(&subtract(&r_blob, &r_top).unwrap()) / max_abs(&r_top);
    println!("[peft blob] rel-max-abs vs top-level alpha=128 = {drift:.3e}");
    assert!(
        drift < 1e-6,
        "a consistent PEFT blob must fold at alpha 128 / rank 128 like the top-level stamp; got \
         {drift:.3e}"
    );
    let default_fallback = write_lora(dir.path(), "no_alpha.safetensors", &cfg, None);
    let ratio = max_abs(&probe_residual(&cfg, &default_fallback, 1.0, &x)) / max_abs(&r_blob);
    assert!(
        (ratio - 0.0625).abs() < 1e-5,
        "…and the blob alpha must WIN over DEFAULT_LORA_ALPHA, which would have given 1.0; got \
         {ratio}"
    );
}

// ─── sc-19443: the ComfyUI key space ───────────────────────────────────────────────────────────

/// **Where a ComfyUI export stamps the alpha of its fused `attn.qkv_proj`.**
///
/// The fixture generator used to emit the in-band `.alpha` tensor and nothing else, which made the
/// twin-equivalence gate — a well-built gate — span exactly **one of four** spellings. The other
/// three routed around the conversion's block-diagonal `÷3` and folded attention 3× too strong with
/// no error, and the gate could not see any of them. The generator is parameterized over this so
/// every arm below runs four times.
///
/// The `__metadata__` spelling is not a hypothetical: it is what **every published lightx2v file**
/// uses, and none of them carries an in-band `.alpha` at all. On this lane the PEFT blob is a second
/// bypass, because `apply_one_lora` honors it too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AlphaSpelling {
    /// A per-target `.alpha` tensor beside the factors — the kohya / ComfyUI in-band convention,
    /// and the only spelling the original fixture generator could write.
    InBand,
    /// A PEFT `lora_adapter_metadata` JSON blob in the top-level `__metadata__` (sc-5513).
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

/// Write a **ComfyUI** twin of one block's modules, built from the same factors a diffusers twin
/// would carry, so any difference in the folded result is the conversion's fault.
///
/// `fused.block_diagonal` selects which of the two legitimate fused forms to write;
/// `fused.spelling` selects **where the fused alpha lives**, which is the axis the equivalence gate
/// was blind along.
fn write_comfyui_twin(
    dir: &Path,
    name: &str,
    cfg: &MiniMaxH3DitConfig,
    fused: FusedQkvSpec,
    qkv: &[(Array, Array); 3],
    fc1: (&Array, &Array),
) -> PathBuf {
    let FusedQkvSpec {
        alpha,
        spelling,
        block_diagonal,
    } = fused;
    let path = dir.join(name);
    let r = PUBLISHED_RANK;
    let out = cfg.inner_dim();
    let mut arrays: Vec<(String, Array)> = Vec::new();

    if block_diagonal {
        let a = concatenate_axis(&[qkv[0].0.clone(), qkv[1].0.clone(), qkv[2].0.clone()], 0)
            .expect("concat A");
        let rows: Vec<Array> = (0..3usize)
            .map(|i| {
                let parts: Vec<Array> = (0..3usize)
                    .map(|j| {
                        if i == j {
                            qkv[i].1.clone()
                        } else {
                            bf16(&Array::zeros::<f32>(&[out, r]).unwrap())
                        }
                    })
                    .collect();
                concatenate_axis(&parts, 1).expect("concat row")
            })
            .collect();
        let b = concatenate_axis(&rows, 0).expect("concat B");
        arrays.push(("blocks.0.attn.qkv_proj.lora_A.weight".into(), a));
        arrays.push(("blocks.0.attn.qkv_proj.lora_B.weight".into(), b));
    } else {
        arrays.push((
            "blocks.0.attn.qkv_proj.lora_A.weight".into(),
            qkv[0].0.clone(),
        ));
        let b = concatenate_axis(&[qkv[0].1.clone(), qkv[1].1.clone(), qkv[2].1.clone()], 0)
            .expect("concat B");
        arrays.push(("blocks.0.attn.qkv_proj.lora_B.weight".into(), b));
    }
    // The fused alpha, written into exactly ONE of the four places it can live.
    if spelling == AlphaSpelling::InBand {
        arrays.push((
            "blocks.0.attn.qkv_proj.alpha".into(),
            Array::from_slice(&[alpha], &[1]),
        ));
    }

    // `mlp.fc1` carries the SwiGLU halves the OTHER way round — `[gate | value]` where the DiT is
    // `[value | gate]` — so the twin's B is the diffusers B with its row halves swapped.
    let (fc1_a, fc1_b) = fc1;
    let rows = fc1_b.shape()[0];
    let half = rows / 2;
    let top = mlx_gen_minimax_h3::tensor::slice_axis(fc1_b, 0, 0, half).expect("top half");
    let bottom = mlx_gen_minimax_h3::tensor::slice_axis(fc1_b, 0, half, rows).expect("bottom half");
    let swapped = concatenate_axis(&[bottom, top], 0).expect("swap halves");
    arrays.push(("blocks.0.mlp.fc1.lora_A.weight".into(), fc1_a.clone()));
    arrays.push(("blocks.0.mlp.fc1.lora_B.weight".into(), swapped));
    // `fc1` is unfused, so its alpha is never divided — pinned in-band in every arm so the only
    // thing `spelling` varies is the fused module's.
    arrays.push((
        "blocks.0.mlp.fc1.alpha".into(),
        Array::from_slice(&[FC1_ALPHA], &[1]),
    ));

    let mut meta: HashMap<String, String> = [(
        "target_format".to_string(),
        "ComfyUI generic LoRA".to_string(),
    )]
    .into_iter()
    .collect();
    match spelling {
        // A contradictory file-level value makes the precedence claim non-vacuous: the per-target
        // tensor must win independently for qkv and fc1.
        AlphaSpelling::InBand => {
            meta.insert("alpha".to_string(), "3".to_string());
        }
        AlphaSpelling::Absent => {}
        // `r` equals the per-target factor rank on BOTH fused forms (a block-diagonal `[3r, in]` A
        // splits into three rank-`r` ones), so this blob is self-consistent and is not rejected by
        // the rank cross-check — it is the alpha, and only the alpha, that is under test.
        AlphaSpelling::PeftBlob => {
            meta.insert(
                "lora_adapter_metadata".to_string(),
                format!(r#"{{"lora_alpha": {alpha}, "r": {PUBLISHED_RANK}}}"#),
            );
        }
        AlphaSpelling::TopLevelMetadata => {
            meta.insert("alpha".to_string(), alpha.to_string());
        }
    }
    let entries: Vec<(&str, &Array)> = arrays.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(entries, Some(&meta), &path).expect("write the comfyui twin");
    path
}

/// Re-key the small raw-module fixture into the exact trainer spelling. Production validates the
/// full 50×4 census before this rewrite; this weights-light fixture isolates the numerical seam.
fn trainer_keys_from_comfyui(w: &Weights) -> Weights {
    let mut trainer = Weights::empty();
    for source in w.keys().map(str::to_string).collect::<Vec<_>>() {
        let target = source
            .replace("blocks.0.attn.qkv_proj", "lora_unet_blocks_0_attn_qkv_proj")
            .replace("blocks.0.mlp.fc1", "lora_unet_blocks_0_mlp_fc1")
            .replace(".lora_A.weight", ".lora_down.weight")
            .replace(".lora_B.weight", ".lora_up.weight");
        trainer.insert(target, w.require(&source).unwrap().clone());
    }
    trainer
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
    qkv: &[(Array, Array); 3],
    fc1: (&Array, &Array),
) -> PathBuf {
    let path = dir.join(name);
    let mut arrays: Vec<(String, Array)> = Vec::new();
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
            Array::from_slice(&[qkv_alpha], &[1]),
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
        Array::from_slice(&[FC1_ALPHA], &[1]),
    ));
    let meta: HashMap<String, String> = [("alpha".to_string(), "1".to_string())]
        .into_iter()
        .collect();
    let entries: Vec<(&str, &Array)> = arrays.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(entries, Some(&meta), &path).expect("write the diffusers twin");
    path
}

/// The factors both twins are built from — distinct per projection, so a conversion that mixes up
/// q/k/v is observable rather than symmetric.
fn twin_factors(cfg: &MiniMaxH3DitConfig) -> ([(Array, Array); 3], (Array, Array)) {
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
    let comfy = Weights::from_file(&comfy_path).expect("raw fixture");
    let trainer = trainer_keys_from_comfyui(&comfy);
    let (raw, alphas) =
        unflatten_minimax_h3_trainer_tensors(&trainer).expect("exact trainer unflatten");
    assert_eq!(alphas, vec![FC1_ALPHA, FUSED_ALPHA]);

    let got = convert_comfyui_key_space(&raw).expect("trainer conversion");
    assert_eq!(got.keys().count(), 12, "q/k/v and fc1 each carry A/B/alpha");
    for (index, name) in ["to_q", "to_k", "to_v"].iter().enumerate() {
        for (suffix, expected) in [
            ("lora_A.weight", &qkv[0].0),
            ("lora_B.weight", &qkv[index].1),
        ] {
            let key = format!("transformer_blocks.0.attn.{name}.{suffix}");
            let actual = got.require(&key).unwrap();
            let drift =
                max_abs(&subtract(actual, expected).unwrap()) / max_abs(expected).max(1e-12);
            assert!(
                drift <= 1e-6,
                "{key}: trainer conversion drifted at relative max-abs {drift:.3e}"
            );
        }
        let key = format!("transformer_blocks.0.attn.{name}.alpha");
        let expected = Array::from_slice(&[FUSED_ALPHA], &[1]);
        let actual = got.require(&key).unwrap();
        let drift = max_abs(&subtract(actual, &expected).unwrap()) / max_abs(&expected);
        assert!(
            drift <= 1e-6,
            "{key}: trainer/raw conversion drifted at relative max-abs {drift:.3e}"
        );
    }
    for (suffix, expected) in [("lora_A.weight", &fc1.0), ("lora_B.weight", &fc1.1)] {
        let key = format!("transformer_blocks.0.ff.net.0.proj.{suffix}");
        let actual = got.require(&key).unwrap();
        let drift = max_abs(&subtract(actual, expected).unwrap()) / max_abs(expected).max(1e-12);
        assert!(drift <= 1e-6, "{key}: relative max-abs {drift:.3e}");
    }
    let expected = Array::from_slice(&[FC1_ALPHA], &[1]);
    let drift = max_abs(
        &subtract(
            got.require("transformer_blocks.0.ff.net.0.proj.alpha")
                .unwrap(),
            &expected,
        )
        .unwrap(),
    ) / max_abs(&expected);
    assert!(
        drift <= 1e-6,
        "FC1 alpha drifted at relative max-abs {drift:.3e}"
    );
}

/// **A converted ComfyUI file folds to the SAME residual as its diffusers twin**, gated on relative
/// max-abs-diff.
///
/// That equivalence is the only honest gate for this conversion, and it is what makes each of the
/// three transforms individually load-bearing: un-fusing `qkv_proj` wrong mixes q/k/v, dropping the
/// alpha division folds 3× strong, and skipping the SwiGLU swap computes `w2(silu(value)·gate)` —
/// the sc-18740 defect, which shipped green at cosine 0.73–0.78. Cosine cannot see any of them.
///
/// **Both fused forms are covered.** Block-diagonal is the lightx2v twins' shape; shared-`A` is what
/// a LoRA trained natively on the fused module looks like, and it carries a DIFFERENT adapter (one
/// down factor shared by all three projections), so it gets its own twin.
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
    let dir = tempfile::tempdir().expect("fixture dir");
    let (qkv, fc1) = twin_factors(&cfg);
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
            let x = tensor(&[1, 4, cfg.hidden_size], 0.31);

            // The bare base is the same for both files, so it is computed once per probe.
            let mut base = tiny_dit(&cfg);
            let y0 = base
                .adaptable_mut(&segs)
                .expect("probe")
                .forward(&x)
                .expect("base forward");
            let residual = |file: &Path| -> Array {
                let mut adapted = tiny_dit(&cfg);
                apply_minimax_h3_adapters(&mut adapted, &[spec(file.to_path_buf(), 1.0)])
                    .expect("install");
                let y1 = adapted
                    .adaptable_mut(&segs)
                    .expect("probe")
                    .forward(&x)
                    .expect("adapted forward");
                subtract(&y1, &y0).expect("residual")
            };

            let want = residual(&diffusers);
            let got = residual(&comfy);
            assert!(
                max_abs(&want) > 1e-4,
                "{probe}: the twin's own residual must be non-trivial, else this is vacuous"
            );
            let drift = max_abs(&subtract(&got, &want).unwrap()) / max_abs(&want);
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
    let dir = tempfile::tempdir().expect("fixture dir");
    let (qkv, fc1) = twin_factors(&cfg);

    let read = |file: &PathBuf, key: &str| -> f32 {
        let w = Weights::from_file(file).unwrap();
        let converted = convert_comfyui_key_space(&w).unwrap();
        converted
            .require(key)
            .unwrap_or_else(|_| panic!("no converted alpha at {key}"))
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()[0]
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

/// **The alpha division tracks the rank split, and lands on a NON-default number.**
///
/// The published `24 → 8` is a trap for a test: `8` is also `DEFAULT_LORA_ALPHA`, so a converter
/// that dropped the alpha entirely would land on the same fold and pass. The first arm therefore
/// uses `48 → 16`, which is neither the default nor the fused value.
///
/// The shared-`A` arm is the other half of the claim: its rank is NOT divided, so dividing its alpha
/// would fold it three times too weak.
#[test]
fn the_alpha_division_tracks_the_rank_split_and_is_not_the_default() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let (qkv, fc1) = twin_factors(&cfg);

    let read_alpha = |file: &PathBuf| -> f32 {
        let w = Weights::from_file(file).unwrap();
        let converted = convert_comfyui_key_space(&w).unwrap();
        converted
            .require("transformer_blocks.0.attn.to_q.alpha")
            .expect("the converted per-target alpha")
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()[0]
    };

    // Block-diagonal, a NON-default alpha: 48 / 3 = 16, and 16/128 = 0.125 ≠ the default's 0.0625.
    let f48 = write_comfyui_twin(
        dir.path(),
        "a48.safetensors",
        &cfg,
        FusedQkvSpec::at(48.0, true),
        &qkv,
        (&fc1.0, &fc1.1),
    );
    assert_eq!(read_alpha(&f48), 16.0);
    assert_eq!(alpha_rank_fold(16.0, PUBLISHED_RANK as f32), 0.125);
    assert_ne!(
        alpha_rank_fold(16.0, PUBLISHED_RANK as f32),
        alpha_rank_fold(DEFAULT_LORA_ALPHA, PUBLISHED_RANK as f32),
        "this arm must not land on the default fold, or a converter that dropped the alpha passes"
    );

    // The published pairing: 24 / 3 = 8, holding `alpha/rank` at 24/384 == 8/128 == 0.0625.
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
        alpha_rank_fold(8.0, PUBLISHED_RANK as f32)
    );

    // Shared-A keeps rank `r`, so its alpha is UNCHANGED.
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

/// The conversion renames every ComfyUI module, the trunk container included, and the report says it
/// ran — a diffusers file must NOT be counted.
#[test]
fn the_conversion_leaves_no_comfyui_module_behind() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let (qkv, fc1) = twin_factors(&cfg);
    let comfy = write_comfyui_twin(
        dir.path(),
        "c.safetensors",
        &cfg,
        FusedQkvSpec::at(24.0, true),
        &qkv,
        (&fc1.0, &fc1.1),
    );

    let w = Weights::from_file(&comfy).unwrap();
    assert!(is_comfyui_key_space(&w));
    let converted = convert_comfyui_key_space(&w).unwrap();
    assert!(
        !is_comfyui_key_space(&converted),
        "the converted map must carry no ComfyUI module name at all"
    );
    assert!(converted
        .keys()
        .all(|k| k.starts_with("transformer_blocks.") || k.starts_with("token_refiner.")));

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
/// three rank-`r/3` LoRAs and divide its alpha — folding it three times too weak, on factors it was
/// never trained with. The rank here is 96 precisely so the shape alone cannot decide.
#[test]
fn block_diagonality_is_measured_on_the_bytes_not_inferred_from_the_shape() {
    let cfg = dit_fixture_config();
    let r: i32 = 96;
    assert_eq!(r % 3, 0);
    let out = cfg.inner_dim();

    let mut w = Weights::empty();
    w.insert(
        "blocks.0.attn.qkv_proj.lora_A.weight",
        bf16(&tensor(&[r, cfg.hidden_size], 1.0)),
    );
    // A DENSE `[3·out, r]` B — emphatically not block-diagonal.
    w.insert(
        "blocks.0.attn.qkv_proj.lora_B.weight",
        bf16(&tensor(&[3 * out, r], 2.0)),
    );
    w.insert(
        "blocks.0.attn.qkv_proj.alpha",
        Array::from_slice(&[48.0f32], &[1]),
    );

    let converted = convert_comfyui_key_space(&w).unwrap();
    let a = converted
        .require("transformer_blocks.0.attn.to_q.lora_A.weight")
        .unwrap();
    assert_eq!(
        a.shape()[0],
        r,
        "a shared-A fused LoRA keeps its full rank; splitting it uses factors it never had"
    );
    let alpha = converted
        .require("transformer_blocks.0.attn.to_q.alpha")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()[0];
    assert_eq!(
        alpha, 48.0,
        "the alpha must NOT be divided when the rank was not split"
    );
}

/// `Weights` is not `Debug`, so `unwrap_err` is unavailable — match the error out by hand.
fn expect_conversion_error(w: &Weights) -> String {
    match convert_comfyui_key_space(w) {
        Ok(_) => panic!("a malformed fused qkv must be refused, not silently split"),
        Err(e) => e.to_string(),
    }
}

/// A fused `qkv_proj` whose factors do not compose, or whose `B` is not three equal projections, is
/// an error naming the module — never a guessed split.
#[test]
fn a_malformed_fused_qkv_is_refused_by_name() {
    let cfg = dit_fixture_config();
    let r = PUBLISHED_RANK;
    let out = cfg.inner_dim();

    let mut w = Weights::empty();
    w.insert(
        "blocks.0.attn.qkv_proj.lora_A.weight",
        bf16(&tensor(&[3 * r, cfg.hidden_size], 1.0)),
    );
    w.insert(
        "blocks.0.attn.qkv_proj.lora_B.weight",
        bf16(&tensor(&[3 * out, r], 2.0)),
    );
    let e = expect_conversion_error(&w);
    assert!(e.contains("do not compose"), "{e}");
    assert!(e.contains("qkv_proj"), "{e}");

    let mut w = Weights::empty();
    w.insert(
        "blocks.0.attn.qkv_proj.lora_A.weight",
        bf16(&tensor(&[r, cfg.hidden_size], 1.0)),
    );
    w.insert(
        "blocks.0.attn.qkv_proj.lora_B.weight",
        bf16(&tensor(&[3 * out + 1, r], 2.0)),
    );
    let e = expect_conversion_error(&w);
    assert!(e.contains("three equal q/k/v projections"), "{e}");

    let mut w = Weights::empty();
    w.insert(
        "blocks.0.attn.qkv_proj.lora_A.weight",
        bf16(&tensor(&[r, cfg.hidden_size], 1.0)),
    );
    let e = expect_conversion_error(&w);
    assert!(e.contains("needs both"), "{e}");
}

// ─── real weights ──────────────────────────────────────────────────────────────────────────────

/// The published `lightx2v/Minimax-h3-Turbo` set, from `MINIMAX_H3_TURBO_LORA`.
fn turbo_dir() -> PathBuf {
    let raw = std::env::var("MINIMAX_H3_TURBO_LORA").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_TURBO_LORA must point at a downloaded `lightx2v/Minimax-h3-Turbo` snapshot dir"
    );
    let dir = PathBuf::from(raw);
    assert!(
        dir.is_dir(),
        "MINIMAX_H3_TURBO_LORA={} is not a dir",
        dir.display()
    );
    dir
}

/// **The real-weight gate**: every published turbo file's 624 tensors collapse to 312 modules, all
/// of which are MiniMax-H3 DiT targets — 0 unmatched — and each file's alpha resolves to the value
/// it actually ships.
///
/// Proves it RAN rather than skipped: the file count is asserted, so an empty or wrong directory
/// fails instead of passing in 0.00s.
#[test]
#[ignore = "needs the published lightx2v/Minimax-h3-Turbo files (MINIMAX_H3_TURBO_LORA)"]
fn published_turbo_files_resolve_to_dit_targets_at_the_measured_alphas() {
    let dir = turbo_dir();
    let cfg = MiniMaxH3DitConfig::default();
    let targets: std::collections::BTreeSet<String> =
        adapter_target_paths(&cfg).into_iter().collect();
    assert_eq!(targets.len(), 312);

    // file stem → (expected alpha, whether the file DECLARES it). Measured off the published
    // headers.
    //
    // **The `declared` flag is what makes three of these four arms mean anything.** Three published
    // files resolve to `8.0`, which *is* `DEFAULT_LORA_ALPHA` — so `alpha == 8.0` passes whether
    // resolution read the header or fell straight through to the fallback, and the arms would be
    // vacuous on their own. The published alphas cannot be changed to non-default values without
    // making the fixture a lie about real files, so provenance is asserted instead: a `declared`
    // file must carry a parseable `__metadata__["alpha"]` that equals the resolved value, and the
    // undeclared one must carry no such key at all. Between the two claims, an implementation that
    // ignored the header and always returned the default fails every `declared` arm.
    let expected: [(&str, f32, bool); 4] = [
        ("minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16", 128.0, true),
        ("minimax_h3_fl2v_turbo_8step_v1.0_bf16", 8.0, true),
        ("minimax_h3_ref2v_turbo_4step_v0.1_bf16", 8.0, true),
        // Ships NO alpha — resolves through DEFAULT_LORA_ALPHA.
        (
            "minimax_h3_fl2v_turbo_4step_v0.1",
            DEFAULT_LORA_ALPHA,
            false,
        ),
    ];
    let comfy = [
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16",
        "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16",
        "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16",
    ];

    let mut examined = 0usize;
    for (stem, want_alpha, declared) in expected {
        let path = dir.join(format!("{stem}.safetensors"));
        assert!(path.is_file(), "missing {}", path.display());
        let w = Weights::from_file(&path).unwrap_or_else(|e| panic!("read {stem}: {e}"));
        assert!(
            !is_comfyui_key_space(&w),
            "{stem} must be the diffusers key space"
        );
        assert_eq!(w.len(), 624, "{stem} tensor count");

        let alpha = resolve_alpha(&w).unwrap_or_else(|e| panic!("{stem} alpha: {e}"));
        assert_eq!(alpha, want_alpha, "{stem} alpha");
        // Provenance, so the three `8.0` arms are not just re-asserting `DEFAULT_LORA_ALPHA`.
        let raw = w.metadata("alpha");
        assert_eq!(
            raw.is_some(),
            declared,
            "{stem}: expected __metadata__[\"alpha\"] to be {}, got {raw:?}",
            if declared { "present" } else { "absent" }
        );
        match raw {
            Some(s) => assert_eq!(
                s.trim().parse::<f32>().ok(),
                Some(want_alpha),
                "{stem}: the resolved alpha must come FROM the header, not from the fallback"
            ),
            None => assert_eq!(
                alpha, DEFAULT_LORA_ALPHA,
                "{stem} declares no alpha, so it must resolve through DEFAULT_LORA_ALPHA"
            ),
        }
        // No published file carries a per-target `.alpha` tensor or a PEFT blob — the two higher
        // links of the chain — which is exactly why the `__metadata__` route has to work.
        assert!(
            !w.keys().any(|k| k.ends_with(".alpha")),
            "{stem}: no published file carries an in-band .alpha"
        );
        assert!(
            w.metadata("lora_adapter_metadata").is_none(),
            "{stem}: no published file carries a PEFT blob"
        );
        assert!(
            w.metadata(mlx_gen_minimax_h3::adapters::RANK_METADATA_KEY)
                .is_none(),
            "{stem}: no published file carries a rank key, so the rank cross-check cannot be \
             relied on to catch an alpha error"
        );

        // Collapse the 624 factor keys to module paths through the real suffix table, and check each
        // against the DiT's target surface.
        let mut modules = std::collections::BTreeSet::new();
        let mut ranks = std::collections::BTreeSet::new();
        for key in w.keys() {
            let Some(stem_path) = key
                .strip_suffix(".lora_A.default.weight")
                .or_else(|| key.strip_suffix(".lora_B.default.weight"))
            else {
                panic!("{stem}: unexpected key spelling {key}");
            };
            if key.ends_with(".lora_A.default.weight") {
                let a = w.require(key).unwrap();
                ranks.insert(resolve_rank(stem_path, a, None).unwrap() as i32);
            }
            modules.insert(mlx_gen_minimax_h3::adapters::normalize_minimax_h3_key(
                stem_path,
            ));
        }
        assert_eq!(modules.len(), 312, "{stem} module count");
        assert_eq!(
            ranks,
            std::collections::BTreeSet::from([128]),
            "{stem} rank, derived from the lora_A shapes"
        );
        let unmatched: Vec<&String> = modules.difference(&targets).collect();
        let fold = alpha_rank_fold(alpha, 128.0);
        println!(
            "[turbo lora] {stem}: 624 tensors -> {} modules, {} unmatched, alpha {alpha} / rank 128 \
             => fold {fold}",
            modules.len(),
            unmatched.len()
        );
        assert!(
            unmatched.is_empty(),
            "{stem}: {} module(s) would fold onto nothing: {unmatched:?}",
            unmatched.len()
        );
        assert_eq!(
            modules
                .iter()
                .filter(|m| m.starts_with("token_refiner."))
                .count(),
            12,
            "{stem}: the refiner's 24 tensors must resolve"
        );
        examined += 1;
    }

    for stem in comfy {
        let path = dir.join(format!("{stem}.safetensors"));
        assert!(path.is_file(), "missing {}", path.display());
        let w = Weights::from_file(&path).unwrap_or_else(|e| panic!("read {stem}: {e}"));
        assert!(
            is_comfyui_key_space(&w),
            "{stem} must be detected as the comfyui key space and converted"
        );
        // sc-19443 converts these rather than refusing them. The alpha lives in `__metadata__` on
        // the real files too, so this is the exact route the review found unguarded.
        let converted = convert_comfyui_key_space(&w).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(
            !is_comfyui_key_space(&converted),
            "{stem}: the conversion must leave no ComfyUI module behind"
        );
        let fused_alpha = resolve_alpha(&w).unwrap_or_else(|e| panic!("{stem} alpha: {e}"));
        let per_target = converted
            .require("transformer_blocks.0.attn.to_q.alpha")
            .unwrap_or_else(|_| panic!("{stem}: no per-target alpha emitted"))
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()[0];
        println!(
            "[turbo lora] {stem}: comfyui key space — converted, fused alpha {fused_alpha} => \
             per-target {per_target}"
        );
        assert_eq!(
            per_target,
            fused_alpha / 3.0,
            "{stem}: these files are block-diagonal, so the resolved alpha must divide by three"
        );
        examined += 1;
    }

    println!("[turbo lora] EXAMINED {examined} published file(s)");
    assert_eq!(examined, 7, "the published set is 4 diffusers + 3 comfyui");
}

/// **The end-to-end gate**: the real 8-step turbo LoRA folds onto the real DiT partition — all 312
/// modules matched, at the measured 0.0625 — and the adapted forward differs from the base one.
///
/// `MINIMAX_H3_TURBO_DIT` points at a `transformer/` directory (any tier: the fold is
/// tier-independent). Budget the disk and the load before running it — **measured**, the `q4`
/// `transformer/` is **~17.5 GiB (18.8 GB)** across 14 shards (1.28–1.49 GB each) against bf16's
/// **~61.7 GiB (66.3 GB)**.
#[test]
#[ignore = "needs a real MiniMax-H3 transformer/ (MINIMAX_H3_TURBO_DIT) + the turbo LoRA + Metal"]
fn the_real_turbo_lora_folds_onto_the_real_dit() {
    let lora = turbo_dir().join("minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors");
    assert!(lora.is_file(), "missing {}", lora.display());
    let raw = std::env::var("MINIMAX_H3_TURBO_DIT").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_TURBO_DIT must point at a MiniMax-H3 `transformer/` directory"
    );
    let dit_dir = PathBuf::from(raw);

    let mut dit = MiniMaxH3Dit::load_dir(&dit_dir, Dtype::Bfloat16)
        .unwrap_or_else(|e| panic!("load {}: {e}", dit_dir.display()));
    println!(
        "[turbo e2e] loaded {} — {} blocks, hidden {}",
        dit_dir.display(),
        dit.num_layers(),
        dit.config().hidden_size
    );

    // A probe activation through one real block projection, before and after.
    let hidden = dit.config().hidden_size;
    let x = tensor(&[2, hidden], 0.3).as_dtype(Dtype::Bfloat16).unwrap();
    let segs: Vec<&str> = PROBE.split('.').collect();
    let y0 = dit
        .adaptable_mut(&segs)
        .expect("probe module")
        .forward(&x)
        .expect("base forward");
    mlx_rs::transforms::eval([&y0]).expect("force the base forward");

    let report = apply_minimax_h3_adapters(&mut dit, &[spec(lora.clone(), 1.0)])
        .unwrap_or_else(|e| panic!("install {}: {e}", lora.display()));
    println!(
        "[turbo e2e] applied {} module(s), {} unmatched",
        report.applied,
        report.unmatched_paths.len()
    );
    assert_eq!(report.applied, 312, "every published module must fold");
    assert!(report.unmatched_paths.is_empty());

    let probe = dit.adaptable_mut(&segs).expect("probe module");
    println!(
        "[turbo e2e] probe base: shape {:?}, packed={}, {} adapter(s) stacked",
        probe.base_shape(),
        probe.is_quantized(),
        probe.adapters().len()
    );
    assert_eq!(probe.adapters().len(), 1, "exactly one residual per module");
    let y1 = probe.forward(&x).expect("adapted forward");
    let delta = max_abs(&subtract(&y1, &y0).unwrap());
    let base = max_abs(&y0);
    println!("[turbo e2e] probe |Δy| = {delta:.4e} against |y| = {base:.4e}");
    assert!(
        delta > 0.0,
        "the fold must change the forward — a 0 delta means it landed on nothing"
    );

    // The refiner arm, run for real: `embed_context` is `context_embedder` then the whole 2-block
    // token refiner, so this exercises the 24 published tensors that land there — the ones a stubbed
    // refiner would have dropped. `context_embedder` itself is NOT adapted, so any difference here
    // comes from the refiner blocks.
    let context = tensor(&[1, 4, dit.config().text_dim], 1.7)
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let refined = dit.embed_context(&context).expect("embed + refine");
    mlx_rs::transforms::eval([&refined]).expect("force the refiner chain");
    let clean = MiniMaxH3Dit::load_dir(&dit_dir, Dtype::Bfloat16).expect("reload");
    let refined_base = clean.embed_context(&context).expect("embed + refine");
    let refiner_delta = max_abs(&subtract(&refined, &refined_base).unwrap());
    println!(
        "[turbo e2e] refiner chain |Δ| = {refiner_delta:.4e} against |base| = {:.4e}",
        max_abs(&refined_base)
    );
    assert!(
        refiner_delta > 0.0,
        "the refiner's 24 tensors must fold — a 0 delta means token_refiner took nothing"
    );
    // Keep `clean` alive to the end so the comparison arm cannot be optimized away.
    assert_eq!(clean.num_layers(), dit.num_layers());

    // The 8-step file declares alpha 8 at rank 128, so it must fold at 0.0625 here too.
    let w = Weights::from_file(&lora).expect("read the LoRA");
    assert_eq!(resolve_alpha(&w).unwrap(), 8.0);
    assert_eq!(alpha_rank_fold(8.0, 128.0), 0.0625);
    println!("[turbo e2e] EXAMINED the real DiT at fold 0.0625");
}

// ─── sc-18729: the measured render ─────────────────────────────────────────────────────────────
//
// The fold gates above prove the LoRA lands on the right 312 modules at the right strength. They
// say nothing about whether a 4-step schedule actually produces a clip, what it costs, or what it
// costs in quality — which is the whole reason the toggle exists. This section renders.
//
// **The sampling recipe is documented, not guessed.** `lightx2v/Minimax-h3-Turbo`'s README points
// at `github.com/ModelTC/Minimax-H3-Turbo`, whose model-specs table lists per-variant training
// shifts (video / audio) and recommended inference steps, and whose
// `DIFFUSERS_SETUP_AND_INFERENCE.md` gives the invocation verbatim:
//
// ```text
// python inference_minimax_h3.py --jobs-json examples/prompts_t2va_test.json \
//   --lora-path minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
//   --inference-steps 4 --video-shift 6 --lora-alpha 128 \
//   --megapixels 1.0 --aspect-ratio 16:9 --output-dir outputs/lora_4nfe_768p
// ```
//
// | variant | steps | video / audio shift |
// |---|---|---|
// | base | 50 | 12 / 3 |
// | `4step_v0.1`, `8step_v1.0` (544p) | 4 / 8 | 12 / 3 |
// | `4step_v1.0_768p` | 4 | **6** / 3 |
//
// `--lora-alpha 128` is the reference script working around its own hardcoded default of 8; this
// engine resolves alpha from the file's `__metadata__`, so it is not a knob here and must not
// become one (sc-18724). The audio shift never moves across the published set, which is why
// `scheduler_shift` overrides the video half only.

/// Env var, empty-as-absent.
fn render_env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

fn render_env_f32(key: &str) -> Option<f32> {
    render_env(key).map(|v| {
        v.parse()
            .unwrap_or_else(|e| panic!("{key}={v} is not a float: {e}"))
    })
}

fn render_env_u32(key: &str) -> Option<u32> {
    render_env(key).map(|v| {
        v.parse()
            .unwrap_or_else(|e| panic!("{key}={v} is not an integer: {e}"))
    })
}

/// What one render cost and produced.
struct RenderReceipt {
    frames: Vec<mlx_gen::gen_core::Image>,
    audio: mlx_gen::gen_core::AudioTrack,
    fps: u32,
    /// `generate` entry → **first** `Progress::Step`: the text-encoder map, the prompt encode, the
    /// TE release, the DiT map, the **LoRA fold**, the AdaLN precompute-and-evict — **and the first
    /// model evaluation**, because `render_latents` reports a step only once that evaluation has
    /// finished. One bucket because that is the granularity `Progress` exposes on the `Resident`
    /// path; H3 emits no `Loading` phases.
    setup_and_first_eval_s: f64,
    /// First → last `Progress::Step`. This spans `steps − 1` evaluations, **not** `steps` — the
    /// off-by-one that would otherwise understate `s_per_step` by 25 % on a 4-step render.
    denoise_tail_s: f64,
    /// `Progress::Decoding` → return: video VAE + audio VAE + the A/V fit.
    decode_s: f64,
    /// Seconds per model evaluation, over the `steps − 1` intervals actually bounded by two ticks.
    /// `None` for a single-step render, where no interval exists and any figure would be invented.
    s_per_step: Option<f64>,
    /// Wall-clock of each interval, in order — so a warm-up outlier is visible rather than averaged
    /// into a headline number.
    step_intervals_s: Vec<f64>,
    wall_s: f64,
    peak_bytes: u64,
    steps_seen: u32,
}

/// Everything one arm of the comparison varies. A struct rather than a parameter list because the
/// arms differ in one or two fields at a time, and a positional call site of eight values is how a
/// canvas ends up transposed against a frame count with nothing to catch it.
struct RenderRecipe<'a> {
    /// The upstream snapshot root — text encoder, tokenizer, both VAEs.
    root: &'a Path,
    /// The staged tier's `transformer/`, or `None` for a flat `root/transformer` install.
    dit: Option<&'a Path>,
    /// The turbo file, or `None` for the base arm.
    lora: Option<&'a Path>,
    steps: u32,
    /// The **video** sigma shift. `None` keeps the base checkpoint's published 12.0.
    shift: Option<f32>,
    width: u32,
    height: u32,
    frames: u32,
    seed: u64,
    prompt: &'a str,
}

/// Render once at an explicit recipe, staging the tier exactly as a split install does.
///
/// `recipe.lora` is `None` for the base arm. When `Some`, the alpha is resolved from the file header
/// here too — printed, not passed — so the receipt records the strength the engine independently
/// resolved rather than one this harness supplied.
fn render_once(recipe: &RenderRecipe<'_>) -> RenderReceipt {
    let &RenderRecipe {
        root,
        dit,
        lora,
        steps,
        shift,
        width,
        height,
        frames,
        seed,
        prompt,
    } = recipe;
    use mlx_gen::gen_core::{
        CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
    };
    use mlx_gen_minimax_h3::model::{load, DIT_COMPONENT};

    let mut load_spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()));
    if let Some(d) = dit {
        load_spec = load_spec.with_component(DIT_COMPONENT, WeightsSource::Dir(d.to_path_buf()));
    }
    // `spec.quantize` is deliberately left unset: `reconcile_tier` treats the on-disk marker as
    // authoritative, and asserting `Q4` would additionally demand a q4-marked `transformer_ref`,
    // which the `t2va` path never reads. The tier that loads is still whatever is staged.
    if let Some(p) = lora {
        let w = Weights::from_file(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
        let alpha = resolve_alpha(&w).unwrap_or_else(|e| panic!("alpha: {e}"));
        println!(
            "[render] adapter {} — {} tensors, resolved alpha {alpha}, fold {} at rank 128",
            p.file_name().unwrap_or_default().to_string_lossy(),
            w.len(),
            alpha_rank_fold(alpha, 128.0)
        );
        assert!(
            !is_comfyui_key_space(&w),
            "the ComfyUI twin fuses qkv and swaps the SwiGLU halves; use the diffusers file"
        );
        load_spec = load_spec.with_adapters(vec![spec(p.to_path_buf(), 1.0)]);
    }
    let model = load(&load_spec).unwrap_or_else(|e| panic!("load: {e}"));

    let req = GenerationRequest {
        prompt: prompt.into(),
        width,
        height,
        frames: Some(frames),
        steps: Some(steps),
        scheduler_shift: shift,
        seed: Some(seed),
        cancel: CancelFlag::default(),
        ..Default::default()
    };
    model.validate(&req).expect("the recipe must validate");

    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let started = Instant::now();
    // Every tick, not just the first and last: `s_per_step` is then an average over intervals that
    // really exist, and a warm-up outlier stays visible instead of being smeared into the headline.
    let mut ticks: Vec<Instant> = Vec::new();
    let mut decoding_at: Option<Instant> = None;
    let mut steps_seen = 0u32;
    let out = model
        .generate(&req, &mut |p| match p {
            Progress::Step { current, total } => {
                steps_seen += 1;
                let now = Instant::now();
                if let Some(prev) = ticks.last() {
                    println!(
                        "[render]   step {current}/{total} (+{:.1} s)",
                        now.duration_since(*prev).as_secs_f64()
                    );
                } else {
                    println!(
                        "[render]   step {current}/{total} — setup + first eval took {:.1} s",
                        now.duration_since(started).as_secs_f64()
                    );
                }
                ticks.push(now);
            }
            Progress::Decoding => decoding_at = Some(Instant::now()),
            Progress::Loading(_) => {}
        })
        .unwrap_or_else(|e| panic!("generate: {e}"));
    let wall_s = started.elapsed().as_secs_f64();
    // Read the peak only after `generate` has returned. Every product below is a **host** buffer —
    // `Vec<u8>` pixels and `Vec<f32>` samples — which cannot exist unless the whole graph was
    // materialized inside the timed region. That is what makes this peak meaningful: the epic's
    // recorded trap is a bare `MiniMaxH3Dit::load` reporting 33 KB because MLX mmaps lazily and
    // nothing forced the tensors. Nothing here is unforced.
    let peak_bytes = mlx_rs::memory::get_peak_memory() as u64;

    let (frames, fps, audio) = match out {
        GenerationOutput::Video { frames, fps, audio } => (frames, fps, audio),
        other => panic!("expected a Video output, got {other:?}"),
    };
    let audio = audio.expect("MiniMax-H3 always produces a soundtrack");

    let first = *ticks
        .first()
        .expect("the denoise must report at least one step");
    let last = *ticks
        .last()
        .expect("the denoise must report at least one step");
    let dec = decoding_at.expect("the decode phase must be reported");
    let step_intervals_s: Vec<f64> = ticks
        .windows(2)
        .map(|w| w[1].duration_since(w[0]).as_secs_f64())
        .collect();
    let denoise_tail_s = last.duration_since(first).as_secs_f64();
    RenderReceipt {
        setup_and_first_eval_s: first.duration_since(started).as_secs_f64(),
        denoise_tail_s,
        decode_s: wall_s - dec.duration_since(started).as_secs_f64(),
        // Divided by the interval count, which is `steps − 1`. Dividing by `steps` is the
        // 25 %-at-4-steps error this harness must not make.
        s_per_step: (!step_intervals_s.is_empty())
            .then(|| denoise_tail_s / step_intervals_s.len() as f64),
        step_intervals_s,
        wall_s,
        peak_bytes,
        steps_seen,
        frames,
        audio,
        fps,
    }
}

/// Per-frame stddev and inter-frame motion, on the 0-255 scale — the "is this a picture, and is it
/// one scene" evidence. Same statistic `quant_tiers_real` gates tiers on, so the rows compare.
fn clip_stats(frames: &[mlx_gen::gen_core::Image]) -> (f64, f64) {
    let mut sd_sum = 0.0;
    let mut motion_sum = 0.0;
    for (i, f) in frames.iter().enumerate() {
        let n = f.pixels.len() as f64;
        let mean = f.pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / n;
        sd_sum += (f
            .pixels
            .iter()
            .map(|&p| (f64::from(p) - mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        if i > 0 {
            let prev = &frames[i - 1].pixels;
            motion_sum += f
                .pixels
                .iter()
                .zip(prev.iter())
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                .sum::<f64>()
                / n;
        }
    }
    (
        sd_sum / frames.len() as f64,
        motion_sum / (frames.len() - 1) as f64,
    )
}

/// **The sc-18729 render.** One clip, fully measured, written to disk.
///
/// Every knob is an env var so the same binary produces each arm of the comparison in its **own
/// process** — which is not a convenience: MLX's peak counter is process-wide, and two renders in
/// one process would report the larger one's high-water for both.
///
/// ```sh
/// MINIMAX_H3_SNAPSHOT=<upstream root> \
/// MINIMAX_H3_DIT=<tier>/transformer \
/// MINIMAX_H3_TURBO_LORA=<lightx2v/Minimax-h3-Turbo dir> \
/// MINIMAX_H3_RENDER_LORA=minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16 \
/// MINIMAX_H3_RENDER_STEPS=4 MINIMAX_H3_RENDER_SHIFT=6 \
/// MINIMAX_H3_RENDER_WIDTH=1344 MINIMAX_H3_RENDER_HEIGHT=768 \
/// MINIMAX_H3_RENDER_OUT=<dir> MINIMAX_H3_RENDER_LABEL=turbo4-768p \
///   cargo test -p mlx-gen-minimax-h3 --test integration turbo_lora:: -- --ignored --nocapture \
///   --test-threads=1 turbo_render_records_a_measured_clip
/// ```
///
/// `MINIMAX_H3_RENDER_LORA=none` renders the base arm at the same geometry, which is what makes the
/// speedup a measurement rather than arithmetic.
///
/// The artifacts are raw (`frames.rgb`, `audio.f32le`) rather than an encoded container: this crate
/// has no muxer and must not grow one for a validation harness. `receipt.json` carries everything
/// needed to encode them and everything measured.
#[test]
#[ignore = "sc-18729: needs MINIMAX_H3_SNAPSHOT + the turbo LoRA + Metal; the 1344x768 4-step arm is ~15 min, the 50-step base at that canvas ~2 h"]
fn turbo_render_records_a_measured_clip() {
    let root = PathBuf::from(
        render_env("MINIMAX_H3_SNAPSHOT").expect("MINIMAX_H3_SNAPSHOT=<upstream snapshot root>"),
    );
    let dit = render_env("MINIMAX_H3_DIT").map(PathBuf::from);
    let lora_stem = render_env("MINIMAX_H3_RENDER_LORA").expect(
        "MINIMAX_H3_RENDER_LORA=<file stem>, or `none` for the base arm — an unset value would \
         silently render the base and be recorded as turbo",
    );
    let lora = if lora_stem == "none" {
        None
    } else {
        let p = turbo_dir().join(format!("{lora_stem}.safetensors"));
        assert!(p.is_file(), "missing {}", p.display());
        Some(p)
    };
    let steps = render_env_u32("MINIMAX_H3_RENDER_STEPS").unwrap_or(4);
    let shift = render_env_f32("MINIMAX_H3_RENDER_SHIFT");
    let width = render_env_u32("MINIMAX_H3_RENDER_WIDTH").unwrap_or(1344);
    let height = render_env_u32("MINIMAX_H3_RENDER_HEIGHT").unwrap_or(768);
    let frames_req =
        render_env_u32("MINIMAX_H3_RENDER_FRAMES").unwrap_or(SMALLEST_LEGAL_FRAMES as u32);
    let seed = render_env_u32("MINIMAX_H3_RENDER_SEED").unwrap_or(18_729) as u64;
    let label = render_env("MINIMAX_H3_RENDER_LABEL").unwrap_or_else(|| "render".into());
    let out_dir = PathBuf::from(
        render_env("MINIMAX_H3_RENDER_OUT").expect("MINIMAX_H3_RENDER_OUT=<output dir>"),
    );
    std::fs::create_dir_all(&out_dir).expect("create the output dir");

    // A prompt with picture AND sound in it — the model is a joint AV generator and an A/B that
    // only looks at pixels cannot speak to half of what distillation might have cost.
    let prompt = render_env("MINIMAX_H3_RENDER_PROMPT").unwrap_or_else(|| {
        "a lighthouse on a rocky coast at dusk, waves breaking against the rocks, seagulls calling"
            .into()
    });

    println!(
        "\n[render] {label}: {width}x{height} / {frames_req} frames / {steps} steps / video shift \
         {} / seed {seed}\n[render]   root {}\n[render]   dit  {}\n[render]   lora {}",
        shift.map_or("12 (default)".to_string(), |s| s.to_string()),
        root.display(),
        dit.as_ref().map_or("<flat root/transformer>".into(), |d| d
            .display()
            .to_string()),
        lora.as_ref()
            .map_or("<none — base arm>".into(), |l| l.display().to_string()),
    );

    let r = render_once(&RenderRecipe {
        root: &root,
        dit: dit.as_deref(),
        lora: lora.as_deref(),
        steps,
        shift,
        width,
        height,
        frames: frames_req,
        seed,
        prompt: &prompt,
    });

    // --- evidence the render happened ---------------------------------------------------------
    assert_eq!(r.frames.len(), frames_req as usize, "decoded frame count");
    assert_eq!(r.steps_seen, steps, "one progress tick per evaluation");
    assert_eq!(r.fps, 24);
    for (i, f) in r.frames.iter().enumerate() {
        assert_eq!((f.width, f.height), (width, height), "frame {i} size");
        assert_eq!(f.pixels.len(), (width * height * 3) as usize, "frame {i}");
    }
    assert!(
        r.peak_bytes > 1_000_000_000,
        "MLX peak {} B is too small for a real 33 B forward — nothing was materialized",
        r.peak_bytes
    );
    let (mean_sd, motion) = clip_stats(&r.frames);
    let rms = {
        let s: f64 = r
            .audio
            .samples
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum();
        (s / r.audio.samples.len() as f64).sqrt()
    };
    assert!(
        r.audio.samples.iter().all(|s| s.is_finite()),
        "the soundtrack carries NaN/Inf"
    );

    // --- the artifacts ------------------------------------------------------------------------
    let rgb_path = out_dir.join(format!("{label}.frames.rgb"));
    let pcm_path = out_dir.join(format!("{label}.audio.f32le"));
    {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(&rgb_path).expect("create the frame file"),
        );
        for frame in &r.frames {
            f.write_all(&frame.pixels).expect("write frames");
        }
        f.flush().expect("flush frames");
        let mut a =
            std::io::BufWriter::new(std::fs::File::create(&pcm_path).expect("create the pcm file"));
        for s in &r.audio.samples {
            a.write_all(&s.to_le_bytes()).expect("write pcm");
        }
        a.flush().expect("flush pcm");
    }

    let per_channel = r.audio.samples.len() / usize::from(r.audio.channels);
    let audio_seconds = per_channel as f64 / f64::from(r.audio.sample_rate);
    let video_seconds = f64::from(frames_req) / f64::from(r.fps);
    let intervals = r
        .step_intervals_s
        .iter()
        .map(|v| format!("{v:.3}"))
        .collect::<Vec<_>>()
        .join(", ");
    let s_per_step = r.s_per_step.map_or("null".into(), |v| format!("{v:.4}"));
    let receipt = format!(
        "{{\n  \"label\": \"{label}\",\n  \"width\": {width},\n  \"height\": {height},\n  \
         \"frames\": {frames_req},\n  \"fps\": {},\n  \"steps\": {steps},\n  \
         \"video_shift\": {},\n  \"audio_shift\": 3.0,\n  \"seed\": {seed},\n  \
         \"lora\": \"{lora_stem}\",\n  \"tier_dir\": \"{}\",\n  \
         \"wall_s\": {:.3},\n  \"setup_and_first_eval_s\": {:.3},\n  \
         \"denoise_tail_s\": {:.3},\n  \"denoise_tail_evals\": {},\n  \
         \"decode_s\": {:.3},\n  \"s_per_step\": {s_per_step},\n  \
         \"step_intervals_s\": [{intervals}],\n  \"peak_bytes\": {},\n  \
         \"peak_gb\": {:.3},\n  \"mean_frame_stddev\": {mean_sd:.4},\n  \
         \"inter_frame_motion\": {motion:.4},\n  \"audio_rms\": {rms:.6},\n  \
         \"audio_seconds\": {audio_seconds:.5},\n  \"video_seconds\": {video_seconds:.5},\n  \
         \"sample_rate\": {},\n  \"channels\": {},\n  \"prompt\": \"{}\"\n}}\n",
        r.fps,
        shift.unwrap_or(12.0),
        dit.as_ref()
            .map_or(String::new(), |d| d.display().to_string()),
        r.wall_s,
        r.setup_and_first_eval_s,
        r.denoise_tail_s,
        r.step_intervals_s.len(),
        r.decode_s,
        r.peak_bytes,
        r.peak_bytes as f64 / 1e9,
        r.audio.sample_rate,
        r.audio.channels,
        prompt.replace('"', "'"),
    );
    let receipt_path = out_dir.join(format!("{label}.receipt.json"));
    std::fs::write(&receipt_path, &receipt).expect("write the receipt");

    println!(
        "\n[render] {label} DONE\n  wall {:.1} s = (setup + eval 1) {:.1} + {} further eval(s) \
         {:.1} + decode {:.1}\n  s/step {} over {} interval(s): [{intervals}]\n  MLX peak {:.2} \
         GB\n  frame stddev {mean_sd:.2}, inter-frame motion {motion:.3}, audio rms {rms:.4} \
         ({audio_seconds:.4} s vs {video_seconds:.4} s of picture)\n  {}\n  {}\n  {}",
        r.wall_s,
        r.setup_and_first_eval_s,
        r.step_intervals_s.len(),
        r.denoise_tail_s,
        r.decode_s,
        s_per_step,
        r.step_intervals_s.len(),
        r.peak_bytes as f64 / 1e9,
        rgb_path.display(),
        pcm_path.display(),
        receipt_path.display(),
    );

    // Structure gates last, so the artifacts and the receipt survive a failing one and can be
    // inspected. A turbo arm that renders mush is a *result*, not a lost run.
    assert!(
        mean_sd > 8.0,
        "{label}: frames are nearly flat (stddev {mean_sd:.2}) — the render collapsed"
    );
    assert!(
        motion > 0.05,
        "{label}: frozen clip (inter-frame motion {motion:.4})"
    );
    assert!(
        motion < 60.0,
        "{label}: per-frame noise, not video (inter-frame motion {motion:.2})"
    );
    assert!(
        rms > 1e-4,
        "{label}: the soundtrack is silent (rms {rms:.3e})"
    );
}

// ─── exact external trainer receipt (sc-21028) ─────────────────────────────────────────────────

const TRAINER_LORA_ENV: &str = "MINIMAX_H3_TRAINER_LORA";
const TRAINER_LORA_SHA256_ENV: &str = "MINIMAX_H3_TRAINER_LORA_SHA256";
const TRAINER_LORA_BYTES_ENV: &str = "MINIMAX_H3_TRAINER_LORA_BYTES";
const TRAINER_DIT_ENV: &str = "MINIMAX_H3_DIT";
const SC21028_TRAINER_LORA_SHA256: &str =
    "1fd239662f6290255b0bb3a220764fb53aab2859378f7fd3024030c1e1991cb2";
const SC21028_TRAINER_LORA_BYTES: u64 = 298_263_792;
const SC21028_PROBE_DOWN: &str = "lora_unet_blocks_0_attn_qkv_proj.lora_down.weight";
const SC21028_PROBE_UP: &str = "lora_unet_blocks_0_attn_qkv_proj.lora_up.weight";
const SC21028_PROBE_ALPHA: &str = "lora_unet_blocks_0_attn_qkv_proj.alpha";
const SC21028_RUNTIME_ADAPTER_SCALE: f32 = 0.5;
const SC21028_RUNTIME_RESIDUAL_REL_MAX: f32 = 1e-3;

/// An explicitly selected exact-file receipt must fail closed when the proprietary file was not
/// supplied. The optional digest and byte count bind a local receipt to a known artifact when the
/// operator has them; absent values are reported, never invented.
fn trainer_lora_path() -> PathBuf {
    let raw = std::env::var(TRAINER_LORA_ENV).unwrap_or_else(|_| {
        panic!(
            "{TRAINER_LORA_ENV}=<exact networks.lora_minimax_h3 safetensors file> is required \
             when this ignored receipt test is selected"
        )
    });
    let path = PathBuf::from(raw);
    assert!(
        path.is_file(),
        "{TRAINER_LORA_ENV}={} is not a readable safetensors file",
        path.display()
    );
    path
}

fn sha256_of(path: &Path) -> String {
    let mut file =
        File::open(path).unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)
        .unwrap_or_else(|error| panic!("hash {}: {error}", path.display()));
    format!("{:x}", hasher.finalize())
}

fn assert_optional_trainer_artifact_identity(path: &Path, bytes: u64, sha256: &str) {
    if let Ok(expected) = std::env::var(TRAINER_LORA_BYTES_ENV) {
        let expected = expected.parse::<u64>().unwrap_or_else(|error| {
            panic!("{TRAINER_LORA_BYTES_ENV}={expected:?} is not an unsigned byte count: {error}")
        });
        assert_eq!(bytes, expected, "{} byte count", path.display());
    }
    if let Ok(expected) = std::env::var(TRAINER_LORA_SHA256_ENV) {
        let expected = expected.trim().to_ascii_lowercase();
        assert!(
            expected.len() == 64 && expected.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "{TRAINER_LORA_SHA256_ENV} must be a 64-character SHA-256 hex digest"
        );
        assert_eq!(sha256, expected, "{} SHA-256", path.display());
    }
}

fn assert_sc21028_trainer_artifact_identity(path: &Path, bytes: u64, sha256: &str) {
    assert_eq!(
        bytes,
        SC21028_TRAINER_LORA_BYTES,
        "{} is not the exact SC-21028 trainer artifact: byte count",
        path.display()
    );
    assert_eq!(
        sha256,
        SC21028_TRAINER_LORA_SHA256,
        "{} is not the exact SC-21028 trainer artifact: SHA-256",
        path.display()
    );
    assert_optional_trainer_artifact_identity(path, bytes, sha256);
}

fn trainer_dit_path() -> PathBuf {
    let raw = std::env::var(TRAINER_DIT_ENV).unwrap_or_else(|_| {
        panic!(
            "{TRAINER_DIT_ENV}=<real MiniMax-H3 transformer component directory> is required; \
             this ignored runtime receipt never skips"
        )
    });
    let path = PathBuf::from(raw);
    assert!(
        path.is_dir() && path.join("config.json").is_file(),
        "{TRAINER_DIT_ENV}={} must be a real transformer component directory with config.json",
        path.display()
    );
    path
}

/// Assert the runtime this receipt will actually dispatch through. MLX's public device model names
/// the Apple accelerator `Gpu`; on macOS that backend is Metal. Checking only availability would
/// still let a caller set the process default to CPU and publish a false Metal receipt.
fn require_default_mlx_metal_device() -> (String, i32) {
    assert!(
        cfg!(target_os = "macos"),
        "the MLX/Metal receipt is only valid on macOS"
    );
    let device = Device::try_default().expect("query the MLX process-default device");
    let device_type = device
        .get_type()
        .expect("query the MLX default device type");
    assert!(
        matches!(device_type, DeviceType::Gpu),
        "the MLX process-default device must be GPU/Metal, got {device}"
    );
    let index = device
        .get_index()
        .expect("query the MLX default device index");
    (device.to_string(), index)
}

/// Independent oracle for the exact probe residual. It reads the trainer's raw fused-QKV factors,
/// takes the Q row block itself, and evaluates
/// `x·down^T·up_q^T·alpha/rank·SC21028_RUNTIME_ADAPTER_SCALE`; it never calls either trainer
/// conversion or the adapter installer under test. The exact file's alpha/rank is the identity
/// 16/16, so the non-identity alpha-formula proof remains the independent mutation fixture
/// `trainer_namespace_converts_to_independent_expected_factors_at_relative_max_abs`; this oracle's
/// independent non-unit term is the runtime adapter scale.
fn exact_trainer_q_probe_residual(path: &Path, x: &Array, cfg: &MiniMaxH3DitConfig) -> Array {
    let raw = Weights::from_file(path).expect("read exact trainer factors for independent oracle");
    let down = raw
        .require(SC21028_PROBE_DOWN)
        .expect("exact QKV down factor");
    let fused_up = raw
        .require(SC21028_PROBE_UP)
        .expect("exact fused QKV up factor");
    let alpha = raw
        .require(SC21028_PROBE_ALPHA)
        .expect("exact fused QKV alpha")
        .as_dtype(Dtype::Float32)
        .unwrap()
        .item::<f32>();
    let rank = down.shape()[0];
    assert_eq!(down.shape(), &[16, cfg.hidden_size], "raw QKV down shape");
    assert_eq!(
        fused_up.shape(),
        &[3 * cfg.inner_dim(), 16],
        "raw fused QKV up shape"
    );
    assert_eq!((rank, alpha), (16, 16.0), "raw probe rank/alpha");
    let q_up = mlx_gen_minimax_h3::tensor::slice_axis(fused_up, 0, 0, cfg.inner_dim())
        .expect("take the Q row block independently");
    let unscaled = matmul(matmul(x, down.t()).unwrap(), q_up.t()).unwrap();
    unscaled
        .multiply(Array::from_slice(
            &[(alpha / rank as f32) * SC21028_RUNTIME_ADAPTER_SCALE],
            &[1],
        ))
        .expect("scale independent exact trainer residual")
}

fn converted_lora_module(key: &str) -> Option<&str> {
    key.strip_suffix(".lora_A.weight")
        .or_else(|| key.strip_suffix(".lora_B.weight"))
        .or_else(|| key.strip_suffix(".alpha"))
}

/// Header and conversion receipt for one exact `networks.lora_minimax_h3` artifact.
///
/// Run this only with the supplied proprietary file:
///
/// ```text
/// MINIMAX_H3_TRAINER_LORA=/absolute/path/adapter.safetensors \
///   cargo test -p mlx-gen-minimax-h3 --test integration \
///   turbo_lora::exact_h3_trainer_file_receipt -- --ignored --nocapture
/// ```
///
/// This is deliberately **not** a substitute for the full-model MLX/Metal receipt: that separate
/// run must install this same digest into a real 50-block transformer and exercise every target.
#[test]
#[ignore = "needs MINIMAX_H3_TRAINER_LORA=<exact trainer safetensors>; run with --ignored"]
fn exact_h3_trainer_file_receipt() {
    let path = trainer_lora_path();
    let bytes = std::fs::metadata(&path)
        .unwrap_or_else(|error| panic!("stat {}: {error}", path.display()))
        .len();
    let sha256 = sha256_of(&path);
    assert_optional_trainer_artifact_identity(&path, bytes, &sha256);

    let adapter = Weights::from_file(&path)
        .unwrap_or_else(|error| panic!("read exact trainer file {}: {error}", path.display()));
    let (converted, layout, mut alphas) = convert_minimax_h3_trainer_key_space(&adapter)
        .unwrap_or_else(|error| panic!("validate exact trainer file {}: {error}", path.display()));
    assert_eq!(layout.source_targets, 200, "50 blocks × four source leaves");
    assert_eq!(layout.ranks, vec![16], "unique source factor ranks");
    assert!(layout.trunk_only, "the exact trainer export is trunk-only");
    alphas.sort_by(f32::total_cmp);
    alphas.dedup_by(|left, right| *left == *right);
    assert_eq!(alphas, vec![16.0], "unique per-target alphas");

    let runtime_targets = adapter_target_paths(&MiniMaxH3DitConfig::default())
        .into_iter()
        .filter(|target| target.starts_with("transformer_blocks."))
        .collect::<BTreeSet<_>>();
    assert_eq!(runtime_targets.len(), 300, "50 blocks × six runtime leaves");
    let runtime_modules = converted
        .keys()
        .map(|key| {
            converted_lora_module(key)
                .unwrap_or_else(|| panic!("trainer conversion emitted unexpected key {key}"))
                .to_string()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(converted.len(), 900, "300 runtime leaves × down/up/alpha");
    assert_eq!(
        runtime_modules.len(),
        300,
        "fused QKV expands 200 source targets to 300 leaves"
    );
    let unmatched = runtime_modules
        .difference(&runtime_targets)
        .cloned()
        .collect::<Vec<_>>();
    assert!(
        unmatched.is_empty(),
        "unmatched runtime targets: {unmatched:?}"
    );

    println!(
        "SC21028_TRAINER_RECEIPT backend=mlx file={} bytes={bytes} sha256={sha256} \
         source_targets={} runtime_leaves={} unmatched={} unique_ranks={:?} unique_alphas={:?} \
         parser_layout_only=true",
        path.display(),
        layout.source_targets,
        runtime_modules.len(),
        unmatched.len(),
        layout.ranks,
        alphas,
    );
    println!(
        "SC21028_TRAINER_RUNTIME_RECEIPT_REQUIRED backend=mlx full_model=MINIMAX_H3 transformer \
         blocks=50 accelerator=Metal status=not_run"
    );
}

/// Manual MLX/Metal receipt for the exact SC-21028 trainer artifact against a real 50-block DiT.
///
/// This intentionally probes one real projection rather than rendering a clip: installation walks
/// and audits all 300 trunk projections, while the projection forward proves the installed factors
/// execute numerically on Metal. The change gate is relative max-abs-diff, never shape or cosine.
///
/// ```text
/// MINIMAX_H3_TRAINER_LORA=/absolute/path/deepthroat_v02.safetensors \
/// MINIMAX_H3_DIT=/absolute/path/to/a/real/tier/transformer \
/// cargo test --release --locked -p mlx-gen-minimax-h3 --test integration \
///   turbo_lora::manual_metal_exact_h3_trainer_runtime_receipt -- --ignored --exact --nocapture
/// ```
#[test]
#[ignore = "manual MLX/Metal real-weight receipt; needs exact trainer LoRA and a 50-block DiT"]
fn manual_metal_exact_h3_trainer_runtime_receipt() {
    let (runtime_device, runtime_device_index) = require_default_mlx_metal_device();
    let lora = trainer_lora_path();
    let bytes = std::fs::metadata(&lora)
        .unwrap_or_else(|error| panic!("stat {}: {error}", lora.display()))
        .len();
    let sha256 = sha256_of(&lora);
    assert_sc21028_trainer_artifact_identity(&lora, bytes, &sha256);

    let dit_dir = trainer_dit_path();
    let config_path = dit_dir.join("config.json");
    let config_text = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", config_path.display()));
    let config_sha256 = sha256_of(&config_path);
    let cfg = MiniMaxH3DitConfig::from_diffusers_json(&config_text)
        .expect("parse the real MiniMax-H3 transformer config");
    assert_eq!(
        cfg,
        MiniMaxH3DitConfig::default(),
        "the runtime receipt requires the published 50-block MiniMax-H3 geometry"
    );

    let tier = match mlx_gen::quant::packed_quant_bits_at(&dit_dir)
        .unwrap_or_else(|error| panic!("read staged DiT tier {}: {error}", dit_dir.display()))
    {
        Some(bits) => format!("q{bits}"),
        None => "bf16".to_owned(),
    };
    let mut dit = MiniMaxH3Dit::load_dir(&dit_dir, Dtype::Bfloat16)
        .unwrap_or_else(|error| panic!("load real DiT {}: {error}", dit_dir.display()));
    assert_eq!(
        dit.num_layers(),
        50,
        "the real DiT must contain all 50 blocks"
    );

    let probe = "transformer_blocks.0.attn.to_q";
    let probe_segments = probe.split('.').collect::<Vec<_>>();
    let x = tensor(&[1, 3, cfg.hidden_size], 0.21028)
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let (probe_packed, base_native) = {
        let projection = dit
            .adaptable_mut(&probe_segments)
            .expect("resolve real probe projection before install");
        (
            projection.is_quantized(),
            projection
                .forward(&x)
                .expect("real base projection forward"),
        )
    };
    let base = base_native.as_dtype(Dtype::Float32).unwrap();
    let base_peak = max_abs(&base);
    assert!(
        base_peak.is_finite() && base_peak > 1e-6,
        "real base projection is non-finite or vacuous: max|y|={base_peak:.3e}"
    );

    let report = apply_minimax_h3_adapters(
        &mut dit,
        &[spec(lora.clone(), SC21028_RUNTIME_ADAPTER_SCALE)],
    )
    .expect("install the exact trainer LoRA into the real 50-block DiT");
    assert_eq!(report.applied, 300, "50 blocks × six runtime leaves");
    assert!(report.unmatched_paths.is_empty(), "zero unmatched targets");
    assert_eq!(report.converted_from_trainer, 1, "exact trainer namespace");
    assert_eq!(
        report.trainer_source_targets_applied, 200,
        "50 × four source leaves"
    );
    assert_eq!(report.trainer_ranks, vec![16], "exact source rank");
    assert_eq!(report.trainer_alphas, vec![16.0], "exact per-target alpha");

    let mut adapted_paths = Vec::new();
    for path in adapter_target_paths(&cfg) {
        let segments = path.split('.').collect::<Vec<_>>();
        let linear = dit
            .adaptable_mut(&segments)
            .unwrap_or_else(|| panic!("declared adapter target {path} does not resolve"));
        if !linear.adapters().is_empty() {
            adapted_paths.push(path);
        }
    }
    assert_eq!(
        adapted_paths.len(),
        300,
        "every trunk runtime leaf is adapted"
    );
    assert!(
        adapted_paths
            .iter()
            .all(|path| path.starts_with("transformer_blocks.")),
        "the exact trunk-only artifact must not fabricate token-refiner targets"
    );

    let adapted_native = dit
        .adaptable_mut(&probe_segments)
        .expect("resolve real probe projection after install")
        .forward(&x)
        .expect("real adapted projection forward");
    let observed_delta = subtract(&adapted_native, &base_native)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let independent_residual = exact_trainer_q_probe_residual(&lora, &x, &cfg);
    // The production linear narrows the residual to the base output dtype before adding it. Build
    // the oracle's expected *observed* delta through the same public dtype boundary so BF16 add
    // rounding does not masquerade as a conversion error. QKV selection/orientation and the
    // non-unit runtime scale above remain independent; non-identity alpha/rank coverage belongs to
    // `trainer_namespace_converts_to_independent_expected_factors_at_relative_max_abs` because this
    // exact artifact's alpha/rank is 16/16 = 1.
    let expected_adapted = base_native
        .add(&independent_residual.as_dtype(base_native.dtype()).unwrap())
        .unwrap();
    let expected_delta = subtract(&expected_adapted, &base_native)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let expected_peak = max_abs(&expected_delta);
    let relative_max_abs_diff = expected_peak / base_peak;
    assert!(
        relative_max_abs_diff.is_finite() && relative_max_abs_diff > 1e-6,
        "the exact trainer LoRA did not measurably change the real {probe} forward: \
         relative-max-abs-diff={relative_max_abs_diff:.3e}"
    );
    let residual_correctness_rel_max =
        max_abs(&subtract(&observed_delta, &expected_delta).unwrap()) / expected_peak;
    assert!(
        residual_correctness_rel_max.is_finite()
            && residual_correctness_rel_max <= SC21028_RUNTIME_RESIDUAL_REL_MAX,
        "the installed {probe} residual disagrees with independent raw-factor math by \
         relative-max-abs={residual_correctness_rel_max:.3e} (limit \
         {SC21028_RUNTIME_RESIDUAL_REL_MAX:.3e}); wrong QKV slice/orientation or runtime scale \
         (non-identity alpha/rank is covered by the independent trainer mutation fixture)"
    );

    println!(
         "SC21028_TRAINER_RUNTIME_RECEIPT backend=mlx accelerator=Metal file={} bytes={bytes} \
         sha256={sha256} model_id={} dit={} config_sha256={} tier={} probe_packed={} \
         runtime_device={} runtime_device_index={} adapter_scale={} blocks={} source_targets={} applied={} adapted_modules={} \
         unmatched={} unique_ranks={:?} unique_alphas={:?} probe={} relative_max_abs_diff={:.6e}",
        lora.display(),
        mlx_gen_minimax_h3::MODEL_ID,
        dit_dir.display(),
        config_sha256,
        tier,
        probe_packed,
        runtime_device,
        runtime_device_index,
        SC21028_RUNTIME_ADAPTER_SCALE,
        dit.num_layers(),
        report.trainer_source_targets_applied,
        report.applied,
        adapted_paths.len(),
        report.unmatched_paths.len(),
        report.trainer_ranks,
        report.trainer_alphas,
        probe,
        relative_max_abs_diff,
    );
    println!(
        "SC21028_TRAINER_RUNTIME_CORRECTNESS backend=mlx probe={} \
         observed_vs_independent_expected_relative_max_abs={:.6e} limit={:.6e}",
        probe, residual_correctness_rel_max, SC21028_RUNTIME_RESIDUAL_REL_MAX
    );
}
