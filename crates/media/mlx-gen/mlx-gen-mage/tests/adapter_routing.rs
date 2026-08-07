//! sc-14057 — the NR-MMDiT's **community**-adapter host surface: every `Linear` in the DiT is
//! reachable by its published checkpoint path, including the four globals a
//! `target_modules="all-linear"` PEFT run trains that our own trainer's default target set never
//! touches (`time_text_embed.timestep_embedder.linear_{1,2}`, `norm_out.linear`, `proj_out`).
//!
//! Runs in the default `cargo test`: it drives the committed 2-block f32 fixture
//! (`tests/fixtures/mage_flow_small.safetensors`, see `mage_flow_small.rs`) — no weights, no
//! gitignored golden.
//!
//! **The failure this guards is not a silent one, and that is the point.** The shared installer is
//! `apply_adapters_strict`, so before sc-14057 a third-party adapter naming any of those four
//! surfaced them in `unmatched_paths` and **errored the whole file** — loud, but it made an
//! otherwise-valid Mage adapter completely unusable rather than partially applied. Phase 2 of the
//! test below is the direct probe: revert the routing and it fails.

use std::collections::BTreeSet;
use std::path::PathBuf;

use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::weights::Weights;
use mlx_gen::{AdapterKind, AdapterSpec};
use mlx_gen_mage::config::MageFlowConfig;
use mlx_gen_mage::{apply_mage_adapters, ImgShape, MageTransformer, PackLayout};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/mage_flow_small.safetensors"
);

/// The four globals a `target_modules="all-linear"` community adapter trains and our own
/// default target set (`to_q`/`to_k`/`to_v`/`to_out.0`) does not.
const COMMUNITY_ONLY_TARGETS: [&str; 4] = [
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "norm_out.linear",
    "proj_out",
];

fn fixture() -> Weights {
    Weights::from_file(FIXTURE).expect(
        "tests/fixtures/mage_flow_small.safetensors is committed; regenerate with \
         `<ref-venv>/bin/python crates/media/mlx-gen/tools/dump_mage_flow_small.py`",
    )
}

/// The fixture stores the state dict under a `model.` prefix; strip it back to the checkpoint's
/// own key layout.
fn model_weights(w: &Weights) -> Weights {
    let mut out = Weights::empty();
    for key in w.keys().collect::<Vec<_>>() {
        if let Some(rest) = key.strip_prefix("model.") {
            out.insert(rest, w.require(key).unwrap().clone());
        }
    }
    out
}

fn config(w: &Weights) -> MageFlowConfig {
    let c = w
        .require("config")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    MageFlowConfig {
        in_channels: c[0],
        out_channels: c[1],
        context_in_dim: c[2],
        hidden_size: c[3],
        num_heads: c[4],
        depth: c[5] as usize,
        patch_size: c[6],
        axes_dim: c[7..10].to_vec(),
        checkpoint: false,
    }
}

fn model() -> (MageTransformer, Weights) {
    let w = fixture();
    let mw = model_weights(&w);
    let t = MageTransformer::from_weights(&mw, config(&w)).unwrap();
    (t, mw)
}

/// A fixed packed forward, so "the adapter actually changed something" is measurable.
fn probe(t: &MageTransformer, cfg: &MageFlowConfig) -> Array {
    let (grid, txt_len) = (4, 3);
    let layout = PackLayout::generation(vec![ImgShape::latent(grid, grid)], vec![txt_len]).unwrap();
    let ctx = t.pack_context(layout).unwrap();
    let img = mlx_rs::random::normal::<f32>(
        &[1, grid * grid, cfg.in_channels],
        None,
        None,
        Some(&mlx_rs::random::key(11).unwrap()),
    )
    .unwrap();
    let txt = mlx_rs::random::normal::<f32>(
        &[1, txt_len, cfg.context_in_dim],
        None,
        None,
        Some(&mlx_rs::random::key(12).unwrap()),
    )
    .unwrap();
    let sigma = Array::from_slice(&[0.5f32], &[1]);
    t.forward(&img, &txt, &sigma, &ctx).unwrap()
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let n: i32 = a.shape().iter().product();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    a.subtract(&b)
        .unwrap()
        .abs()
        .unwrap()
        .max(None)
        .unwrap()
        .item::<f32>()
}

/// Write a diffusers/PEFT adapter targeting `paths`, sized off the fixture's own base weights so
/// every factor is shape-correct.
fn write_peft_adapter(
    dir: &std::path::Path,
    name: &str,
    base: &Weights,
    paths: &[&str],
) -> PathBuf {
    let rank = 2i32;
    let mut owned: Vec<(String, Array)> = Vec::new();
    for path in paths {
        let shape = base.require(&format!("{path}.weight")).unwrap().shape();
        let (out_f, in_f) = (shape[0], shape[1]);
        // Non-zero on BOTH factors so the residual is a real delta, not the trainer's `B = 0`
        // no-op init — otherwise "the adapter applied" and "the adapter did nothing" look alike.
        owned.push((
            format!("{path}.lora_A.weight"),
            mlx_rs::random::normal::<f32>(&[rank, in_f], None, None, None).unwrap(),
        ));
        owned.push((
            format!("{path}.lora_B.weight"),
            mlx_rs::random::normal::<f32>(&[out_f, rank], None, None, None).unwrap(),
        ));
        owned.push((
            format!("{path}.alpha"),
            Array::from_slice(&[rank as f32], &[1]),
        ));
    }
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(name);
    Array::save_safetensors(
        owned
            .iter()
            .map(|(k, v)| (k.clone(), v))
            .collect::<Vec<_>>(),
        None,
        &path,
    )
    .unwrap();
    path
}

fn tmp_dir(tmp: &tempfile::TempDir, tag: &str) -> PathBuf {
    let dir = tmp.path().join(format!("sc14057-{tag}"));
    dir
}

/// The whole community-adapter surface, in three phases.
///
/// **Why one test and not three.** Each phase builds a DiT, installs adapters, and writes/reads a
/// safetensors file; that capture → install → re-read pattern strands a lazily-held array on
/// mlx-rs's single default Metal stream once a prior phase has churned it, and a second such test
/// in the same binary dies with "There is no Stream(gpu, N)" while passing in isolation — the
/// mlx-gen-anima precedent (`mlx-gen-anima/tests/common/mod.rs`, sc-10521), which split its suite
/// across binaries for exactly this. `RUST_TEST_THREADS=1` (forced in `.cargo/config.toml`) does
/// not help: the phases are sequential either way, and it is the accumulation, not concurrency,
/// that strands the stream. Folding them into one function keeps that to a single arc.
#[test]
fn the_dit_exposes_its_whole_linear_surface_to_community_adapters() {
    let tmp = tempfile::tempdir().unwrap();
    let (mut t, base) = model();
    let cfg = config(&fixture());

    // --- 1. The `AdaptableHost` contract ---------------------------------------------------
    // Every enumerated path must resolve, and the kohya flattening (`.`→`_`, the
    // `flattened → dotted` lookup the kohya loader builds from this list) must stay
    // collision-free. Adding the four globals to the enumeration is only safe if both hold.
    let paths = AdaptableHost::adaptable_paths(&t);
    assert!(!paths.is_empty());
    for path in &paths {
        let segs: Vec<&str> = path.split('.').collect();
        assert!(
            t.adaptable_mut(&segs).is_some(),
            "{path} is enumerated but does not resolve through adaptable_mut"
        );
    }
    let flattened: BTreeSet<String> = paths.iter().map(|p| p.replace('.', "_")).collect();
    assert_eq!(
        flattened.len(),
        paths.len(),
        "two enumerated paths collide once kohya-flattened"
    );
    for target in COMMUNITY_ONLY_TARGETS {
        assert!(
            paths.iter().any(|p| p == target),
            "{target} must be enumerated (kohya-reachable), not just routable"
        );
    }
    // The globals our own trainer already used stay reachable.
    for target in ["img_in", "txt_in"] {
        assert!(paths.iter().any(|p| p == target));
    }
    // …and a path the DiT does not own still does not resolve.
    assert!(t.adaptable_mut(&["txt_norm"]).is_none());
    assert!(t.adaptable_mut(&["time_text_embed", "time_proj"]).is_none());
    assert!(t.adaptable_mut(&["norm_out"]).is_none());

    // --- 2. 🔴 The trap-3 probe -------------------------------------------------------------
    // An adapter that trains ONLY the timestep embedder and the output head — the shape a
    // `target_modules="all-linear"` community run produces for the modules our default target set
    // skips — must install and change the velocity. Revert the `["time_text_embed", ..]` /
    // `["norm_out", ..] | ["proj_out", ..]` routing and `apply_mage_adapters` errors instead
    // ("adapter target(s) matched no module"), which is the pre-sc-14057 behaviour.
    let before = probe(&t, &cfg);
    let dir = tmp_dir(&tmp, "adapters");
    let head_only = write_peft_adapter(&dir, "head.safetensors", &base, &COMMUNITY_ONLY_TARGETS);
    let report = apply_mage_adapters(
        &mut t,
        &[AdapterSpec {
            path: head_only,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
    )
    .expect("a head/timestep-only community adapter must install, not error");
    assert_eq!(report.applied, COMMUNITY_ONLY_TARGETS.len());
    assert!(report.unmatched_paths.is_empty(), "{report:?}");

    let after = probe(&t, &cfg);
    let delta = max_abs_diff(&before, &after);
    assert!(
        delta > 1e-4,
        "the reloaded head/timestep adapter must change the velocity (max_abs {delta})"
    );

    // --- 3. The whole surface at once -------------------------------------------------------
    // An "all-linear" adapter covering every enumerated target installs as one file, exactly as a
    // community PEFT export would. A fresh DiT so phase 2's residuals do not stack.
    let (mut fresh, _) = model();
    let refs: Vec<&str> = paths.iter().map(String::as_str).collect();
    let all_linear = write_peft_adapter(&dir, "all.safetensors", &base, &refs);
    let report = apply_mage_adapters(
        &mut fresh,
        &[AdapterSpec {
            path: all_linear,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
    )
    .expect("an all-linear adapter must install");
    assert_eq!(report.applied, paths.len());
    assert!(report.unmatched_paths.is_empty(), "{report:?}");
    let _ = std::fs::remove_dir_all(&dir);
}
