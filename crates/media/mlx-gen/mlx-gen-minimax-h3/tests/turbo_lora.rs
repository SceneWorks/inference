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
//!   cargo test -p mlx-gen-minimax-h3 --test turbo_lora -- --ignored --nocapture
//! ```

mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_rs::ops::{matmul, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::gen_core::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::adapters::{
    adapter_target_paths, alpha_rank_fold, apply_minimax_h3_adapters, is_comfyui_key_space,
    resolve_alpha, resolve_rank, DEFAULT_LORA_ALPHA,
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

/// **The `_comfyui_` twin is refused by name**, not half-applied. Its `qkv_proj` is fused and its
/// `mlp.fc1` carries the SwiGLU halves the other way round, so a shape-valid fold would be
/// numerically wrong in a way no key-coverage proof can see (the sc-18740 class).
#[test]
fn the_comfyui_export_is_refused_and_points_at_the_diffusers_twin() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let path = dir.path().join("turbo_comfyui_bf16.safetensors");
    // The real comfyui shape: rank·3 concatenated A, block-diagonal B, alpha ×3, in-band `.alpha`.
    let a = tensor(&[3 * PUBLISHED_RANK, cfg.hidden_size], 1.0);
    let b = tensor(&[3 * cfg.inner_dim(), 3 * PUBLISHED_RANK], 2.0);
    let alpha = Array::from_slice(&[24.0f32], &[]);
    let entries: Vec<(&str, &Array)> = vec![
        ("diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight", &a),
        ("diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight", &b),
        ("diffusion_model.blocks.0.attn.qkv_proj.alpha", &alpha),
    ];
    let meta: HashMap<String, String> = [(
        "target_format".to_string(),
        "ComfyUI generic LoRA".to_string(),
    )]
    .into_iter()
    .collect();
    Array::save_safetensors(entries, Some(&meta), &path).expect("write");

    let w = Weights::from_file(&path).unwrap();
    assert!(is_comfyui_key_space(&w));

    let mut dit = tiny_dit(&cfg);
    let msg = apply_minimax_h3_adapters(&mut dit, &[spec(path, 1.0)])
        .expect_err("a comfyui export must be refused")
        .to_string();
    assert!(msg.contains("_comfyui_"), "got {msg}");
    assert!(msg.contains("qkv_proj"), "got {msg}");
    assert!(
        msg.contains("diffusers twin"),
        "the refusal must name the file to use instead; got {msg}"
    );
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

    // file stem → (expected alpha, expected fold at rank 128). Measured off the published headers.
    let expected: [(&str, f32); 4] = [
        ("minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16", 128.0),
        ("minimax_h3_fl2v_turbo_8step_v1.0_bf16", 8.0),
        ("minimax_h3_ref2v_turbo_4step_v0.1_bf16", 8.0),
        // Ships NO alpha — resolves through DEFAULT_LORA_ALPHA.
        ("minimax_h3_fl2v_turbo_4step_v0.1", DEFAULT_LORA_ALPHA),
    ];
    let comfy = [
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16",
        "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16",
        "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16",
    ];

    let mut examined = 0usize;
    for (stem, want_alpha) in expected {
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
            "{stem} must be detected as the comfyui key space and refused"
        );
        println!("[turbo lora] {stem}: comfyui key space — refused, use the diffusers twin");
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
    /// `generate` entry → **first** `Progress::Step`. Text-encoder map + prompt encode + TE release
    /// + DiT map + **LoRA fold** + AdaLN precompute-and-evict — **and the first model evaluation**,
    /// because `render_latents` reports a step only once that evaluation has finished. One bucket
    /// because that is the granularity `Progress` exposes on the `Resident` path; H3 emits no
    /// `Loading` phases.
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

/// Render once at an explicit recipe, staging the tier exactly as a split install does.
///
/// `lora` is `None` for the base arm. When `Some`, the alpha is resolved from the file header here
/// too — printed, not passed — so the receipt records the strength the engine independently
/// resolved rather than one this harness supplied.
fn render_once(
    root: &Path,
    dit: Option<&Path>,
    lora: Option<&Path>,
    steps: u32,
    shift: Option<f32>,
    (width, height, frames): (u32, u32, u32),
    seed: u64,
    prompt: &str,
) -> RenderReceipt {
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
///   cargo test -p mlx-gen-minimax-h3 --test turbo_lora -- --ignored --nocapture \
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

    let r = render_once(
        &root,
        dit.as_deref(),
        lora.as_deref(),
        steps,
        shift,
        (width, height, frames_req),
        seed,
        &prompt,
    );

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
