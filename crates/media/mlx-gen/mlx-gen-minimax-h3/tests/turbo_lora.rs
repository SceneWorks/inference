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

use mlx_rs::ops::{matmul, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::gen_core::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::adapters::{
    adapter_target_paths, alpha_rank_fold, apply_minimax_h3_adapters, is_comfyui_key_space,
    resolve_alpha, resolve_rank, DEFAULT_LORA_ALPHA,
};
use mlx_gen_minimax_h3::{MiniMaxH3Dit, MiniMaxH3DitConfig};

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
fn write_lora(dir: &Path, name: &str, cfg: &MiniMaxH3DitConfig, alpha: Option<&str>) -> PathBuf {
    let path = dir.join(name);
    let mut arrays: Vec<(String, Array)> = Vec::new();
    for target in adapter_target_paths(cfg) {
        let (out, inn) = target_shape(cfg, &target);
        // A distinct seed per module, so a routing bug that folds the right factors onto the wrong
        // module is observable rather than symmetric.
        let seed = target.len() as f32;
        arrays.push((
            format!("{target}.lora_A.default.weight"),
            tensor(&[PUBLISHED_RANK, inn], seed),
        ));
        arrays.push((
            format!("{target}.lora_B.default.weight"),
            tensor(&[out, PUBLISHED_RANK], seed + 0.5),
        ));
    }
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".into(), "pt".into());
    meta.insert("floating_dtype".into(), "bfloat16".into());
    if let Some(a) = alpha {
        meta.insert("alpha".into(), a.into());
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

/// A file that matches nothing at all names the key space it expected.
#[test]
fn a_file_matching_nothing_names_the_expected_key_space() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let path = dir.path().join("empty.safetensors");
    let t = tensor(&[2, 2], 1.0);
    Array::save_safetensors(vec![("some.base.weight", &t)], None, &path).expect("write");
    let mut dit = tiny_dit(&cfg);
    let msg = apply_minimax_h3_adapters(&mut dit, &[spec(path, 1.0)])
        .expect_err("must fail")
        .to_string();
    assert!(msg.contains("no target modules matched"), "got {msg}");
    assert!(msg.contains("lora_A.default.weight"), "got {msg}");
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
/// tier-independent, and the `q4` one is ~9 GB against bf16's ~62 GB).
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
