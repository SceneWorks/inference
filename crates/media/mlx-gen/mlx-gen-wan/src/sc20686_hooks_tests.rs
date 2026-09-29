//! SC-20686 Metal-lane hook tests on the checked-in tiny 2-block Wan (S5) and Wan-VACE fixtures:
//! the observer is inert when no campaign is armed, the armed hooks leave the product numerics
//! bit-identical, and the armed transcript attributes the real cache tensors.

use super::*;
use mlx_gen::sc20686 as obs;
use mlx_rs::Dtype;
use serde_json::Value;

const SOURCE_REF: &str = "fedcba9876543210fedcba9876543210fedcba98";

fn tiny_cfg() -> crate::config::WanModelConfig {
    let mut c = crate::config::WanModelConfig::wan21_t2v_1_3b();
    c.dim = 128;
    c.num_heads = 1;
    c.num_layers = 2;
    c.ffn_dim = 256;
    c.freq_dim = 256;
    c.text_dim = 32;
    c.text_len = 8;
    c.in_dim = 16;
    c.out_dim = 16;
    c.vae_z_dim = 16;
    c
}

fn bytes(a: &Array) -> Vec<u8> {
    let a = mlx_gen::array::contiguous(&a.as_dtype(Dtype::Float32).expect("f32")).expect("contig");
    a.eval().expect("eval");
    a.as_slice::<f32>()
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

/// A minimal snapshot root with an immutable revision so activation can derive runtime identity.
fn snapshot() -> (tempfile::TempDir, std::path::PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let snapshot = root.path().join("0123456789abcdef0123456789abcdef01234567");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("config.json"), b"{}").unwrap();
    (root, snapshot)
}

fn activate(
    root: &std::path::Path,
    snapshot: &std::path::Path,
    variant: &str,
) -> (obs::Scope, std::rc::Rc<std::cell::RefCell<Vec<Value>>>) {
    let request = obs::request_output(root.join("unused.jsonl"), SOURCE_REF, "sequential")
        .unwrap()
        .arm();
    let (instruments, events) = obs::capture_instruments();
    let scope = obs::activate_with_instruments(
        snapshot,
        &CancelFlag::new(),
        variant,
        obs::RequestFacts {
            batch: 1,
            frames: 5,
            width: 64,
            height: 64,
            prompt_sha256: "a".repeat(64),
            guidance: "5".into(),
            reference_count: 0,
        },
        instruments,
    )
    .unwrap()
    .expect("armed");
    drop(request);
    (scope, events)
}

fn of(events: &[Value], phase: &str) -> Vec<Value> {
    events
        .iter()
        .filter(|event| event["phase"] == phase)
        .cloned()
        .collect()
}

#[test]
fn wan_cross_kv_hooks_are_inert_off_and_bit_identical_on() {
    let cfg = tiny_cfg();
    let weights = mlx_gen::weights::Weights::from_file(format!(
        "{}/tests/fixtures/s5_low.safetensors",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let transformer = WanTransformer::from_weights(&weights, &cfg).unwrap();
    let raw = weights.require("ctx_cond").unwrap().clone();
    let cond = transformer.embed_text(&raw).unwrap();
    let uncond = transformer
        .embed_text(&multiply(&raw, scalar(0.5)).unwrap())
        .unwrap();
    let latent = weights.require("init_noise").unwrap().clone();
    let grid = transformer.patch_grid(&latent);

    // Observer off: no ownership guard, no campaign identity for any tuple.
    let cache = build_cache(&transformer, &cond, Some(&uncond), grid).unwrap();
    assert!(cache._sc20686.is_none());
    assert!(obs::cross_kv_cache_id(&cache.cross_kv[0]).is_none());
    let off: Vec<Vec<u8>> = [999.0, 624.0]
        .iter()
        .map(|&t| bytes(&predict(&transformer, &latent, t, &cache, 5.0, None).unwrap()))
        .collect();
    drop(cache);

    // Observer on: the same forward, with read windows around every cached cross-attention.
    let (root, snapshot) = snapshot();
    let (scope, events) = activate(root.path(), &snapshot, "wan2_2_t2v_14b");
    let cache = build_cache(&transformer, &cond, Some(&uncond), grid).unwrap();
    assert!(cache._sc20686.is_some());
    let dense: Vec<u64> = cache
        .cross_kv
        .iter()
        .map(|(k, v)| (k.nbytes() + v.nbytes()) as u64)
        .collect();
    let on: Vec<Vec<u8>> = [999.0, 624.0]
        .iter()
        .map(|&t| bytes(&predict(&transformer, &latent, t, &cache, 5.0, None).unwrap()))
        .collect();
    assert_eq!(off, on, "campaign read windows changed Wan numerics");
    drop(cache);
    obs::observe_generation_end();
    drop(scope);

    let events = events.borrow();
    let created = of(&events, "cross-kv-created");
    assert_eq!(
        created.len(),
        cfg.num_layers,
        "one persistent cache per block"
    );
    for (event, dense) in created.iter().zip(&dense) {
        assert_eq!(event["persistent_bytes"], *dense, "exact retained nbytes");
        assert_eq!(
            event["kv_batch"], 2,
            "CFG cond+uncond stacked on the batch axis"
        );
        assert_eq!(event["operation"], SC20686_CROSS_KV_CREATE);
    }
    let reads = of(&events, "cross-kv-read");
    assert_eq!(
        reads.len(),
        2 * cfg.num_layers,
        "every block reads per predict"
    );
    let created_ids: Vec<_> = created.iter().map(|e| e["cache_id"].clone()).collect();
    assert!(reads
        .iter()
        .all(|read| created_ids.contains(&read["cache_id"])
            && read["operation"] == crate::transformer::SC20686_CROSS_KV_READ
            && read["allocator_measurement_available"] == true
            && read["transient_bytes"].as_u64().unwrap()
                == read["allocator_high_bytes"].as_u64().unwrap()
                    - read["allocator_before_bytes"].as_u64().unwrap()));
    let released = of(&events, "cross-kv-released");
    assert_eq!(
        released.len(),
        cfg.num_layers,
        "StepCache drop releases every block"
    );
    assert!(released
        .iter()
        .all(|event| event["operation"] == SC20686_CROSS_KV_RELEASE));
    let metrics = of(&events, "metrics").pop().unwrap();
    assert_eq!(
        metrics["current_persistent_bytes"],
        dense.iter().sum::<u64>()
    );
    assert_eq!(metrics["minimum_cache_reads"], 2);
    let metadata = of(&events, "metadata").pop().unwrap();
    assert_eq!(metadata["geometry"]["layers"], cfg.num_layers);
    assert_eq!(metadata["geometry"]["skv"], cfg.text_len);
    assert_eq!(
        metadata["geometry"]["sq"],
        (grid.0 * grid.1 * grid.2) as u64
    );
}

#[test]
fn vace_text_kv_is_recomputed_and_hooks_are_bit_identical() {
    let mut weights = mlx_gen::weights::Weights::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/wanvace_transformer_golden.safetensors"
    ))
    .unwrap();
    let aliases: Vec<_> = weights
        .keys()
        .filter_map(|key| {
            key.strip_prefix("model.")
                .map(|name| (key.to_string(), name.to_string()))
        })
        .collect();
    for (from, to) in aliases {
        weights.alias(&from, &to);
    }
    weights.cast_all(Dtype::Float32).unwrap();
    let config = crate::config::WanVaceConfig::from_config_json(&serde_json::json!({
        "model_type": "t2v", "model_version": "2.1", "dim": 64, "num_heads": 4,
        "num_layers": 4, "ffn_dim": 128, "freq_dim": 64, "text_dim": 32, "in_dim": 16,
        "out_dim": 16, "eps": 1e-6, "dual_model": true, "vace_layers": [0, 2],
        "vace_in_channels": 96
    }));
    let transformer =
        crate::vace::WanVaceTransformer::from_weights(&weights, &config, Dtype::Float32).unwrap();
    let init = weights
        .require("in.hidden_states")
        .unwrap()
        .reshape(&[16, 4, 8, 8])
        .unwrap();
    let control = weights
        .require("in.control_hidden_states")
        .unwrap()
        .reshape(&[96, 4, 8, 8])
        .unwrap();
    let context = weights.require("in.encoder_hidden_states").unwrap().clone();
    let run = || {
        crate::vace::denoise_vace(
            &transformer,
            &control,
            &[1.0, 0.5],
            crate::scheduler::SolverKind::UniPC,
            1000,
            2,
            1.0,
            3.0,
            &context,
            Some(&context),
            &init,
            &CancelFlag::new(),
            &mut |_| {},
        )
        .unwrap()
    };
    let off = bytes(&run());
    let (root, snapshot) = snapshot();
    let (scope, events) = activate(root.path(), &snapshot, "wan_vace");
    let on = bytes(&run());
    obs::observe_generation_end();
    drop(scope);
    assert_eq!(off, on, "campaign read windows changed VACE numerics");

    let events = events.borrow();
    let created = of(&events, "cross-kv-created");
    // 4 main + 2 VACE blocks project the text K/V in each of 2 CFG forwards × 2 steps.
    let layers = 4 + 2;
    assert_eq!(created.len(), layers * 2 * 2);
    assert!(created.iter().all(|event| event["persistent_bytes"] == 0
        && event["cache_id"] == 0
        && event["operation"] == crate::vace::SC20686_VACE_TEXT_KV_CREATE));
    assert!(
        of(&events, "cross-kv-released").is_empty(),
        "nothing persistent to release"
    );
    let metrics = of(&events, "metrics").pop().unwrap();
    assert_eq!(metrics["current_persistent_bytes"], 0);
    assert_eq!(metrics["reused_requests"], 4, "2 steps × 2 CFG forwards");
    assert_eq!(metrics["minimum_cache_reads"], 4);
    let dense = created[0]["transient_bytes"].as_u64().unwrap();
    assert_eq!(metrics["current_read_transient_bytes"], dense);
    let metadata = of(&events, "metadata").pop().unwrap();
    assert_eq!(metadata["geometry"]["layers"], layers);
    assert_eq!(metadata["geometry"]["heads"], 4);
    assert_eq!(metadata["geometry"]["dtype"], "F32");
    let skv = metadata["geometry"]["skv"].as_u64().unwrap();
    assert_eq!(
        dense,
        2 * 4 * skv * 16 * 4,
        "2·B·H·Skv·D·f32 for the recomputed text K/V"
    );
}

#[test]
fn a_sequential_expert_swap_is_its_own_load_window_and_windows_stay_unique() {
    let cfg = tiny_cfg();
    let weights = mlx_gen::weights::Weights::from_file(format!(
        "{}/tests/fixtures/s5_low.safetensors",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let raw = weights.require("ctx_cond").unwrap().clone();
    let latent = weights.require("init_noise").unwrap().clone();
    let (root, snapshot) = snapshot();
    let (scope, events) = activate(root.path(), &snapshot, "wan2_2_t2v_14b");
    let loads = std::cell::Cell::new(0);
    // Both "experts" are the tiny 2-block fixture; the swap mechanics are what is under test.
    let load = |_high: bool| -> Result<(WanTransformer, Array, Option<Array>, f32)> {
        loads.set(loads.get() + 1);
        let transformer = WanTransformer::from_weights(&weights, &cfg)?;
        let cond = transformer.embed_text(&raw)?;
        let uncond = transformer.embed_text(&multiply(&raw, scalar(0.5))?)?;
        Ok((transformer, cond, Some(uncond), 5.0))
    };
    denoise_moe_curated_swapped(
        500.0,
        "euler",
        1000,
        4,
        1.0,
        &latent,
        None,
        7,
        &CancelFlag::new(),
        &mut |progress| obs::observe_progress(&progress),
        load,
    )
    .unwrap();
    obs::observe_generation_end();
    drop(scope);
    assert_eq!(
        loads.get(),
        2,
        "the schedule must cross the expert boundary"
    );
    let windows: Vec<(String, u64)> = of(&events.borrow(), "phase-window")
        .iter()
        .map(|w| {
            (
                w["window"].as_str().unwrap().to_owned(),
                w["window_index"].as_u64().unwrap(),
            )
        })
        .collect();
    let unique: std::collections::BTreeSet<_> = windows.iter().cloned().collect();
    assert_eq!(
        unique.len(),
        windows.len(),
        "duplicate windows: {windows:?}"
    );
    assert_eq!(
        windows.iter().filter(|(name, _)| name == "load").count(),
        1,
        "the mid-denoise swap is one load window: {windows:?}"
    );
    let steps: Vec<u64> = windows
        .iter()
        .filter(|(name, _)| name == "denoise-step")
        .map(|(_, index)| *index)
        .collect();
    assert_eq!(steps, vec![1, 2, 3, 4], "{windows:?}");
}
