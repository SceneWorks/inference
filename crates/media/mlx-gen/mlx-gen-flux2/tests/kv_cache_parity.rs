//! sc-2347: exact correctness gates for the 9b-kv reference-K/V cache, on the committed tiny
//! synthetic transformer (`tests/fixtures/transformer_golden.safetensors`, the sc-2346 S3 fixture).
//! No real weights — these prove the *mechanism* (extract/cached splice, RoPE-position slice, and
//! per-stream layer bookkeeping) is wired correctly, via two exact invariants:
//!
//!   (a) **Extract is transparent.** The step-0 extract forward over `[txt, target, ref]` is
//!       byte-identical to the plain (no-cache) forward — extract only *stores* the trailing ref
//!       K/V, it does not change the attention math.
//!
//!   (b) **Cached reconstructs extract.** After the cache is populated by `extract(X)`, a cached
//!       forward on the *same* target `X` (with the reference tokens dropped, the ref K/V spliced
//!       from the cache) reproduces the target-token slice of `extract(X)` exactly. The fresh
//!       `[txt, target]` K/V are recomputed identically and the spliced ref K/V are the byte-exact
//!       stored arrays, so the `[txt, target]` queries attend over the identical `[txt, target,
//!       ref]` K/V. This is the non-circular proof that the cache splices the right tokens at the
//!       right positions through every block of both stacks.

use mlx_gen::weights::Weights;
use mlx_gen_flux2::{
    prepare_grid_ids, prepare_text_ids, CacheMode, CfgBranch, Flux2Config, Flux2ForwardInputs,
    Flux2KvCache, Flux2KvCfgCaches, Flux2Transformer,
};
use mlx_rs::ops::{array_eq, concatenate_axis};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/transformer_golden.safetensors"
);

const TS: f32 = 500.0;

/// The tiny config the dump script used (inner = 2·8 = 16), matching `transformer_parity.rs`.
fn tiny_config() -> Flux2Config {
    Flux2Config {
        num_double_layers: 1,
        num_single_layers: 1,
        num_heads: 2,
        head_dim: 8,
        in_channels: 4,
        out_channels: 4,
        joint_attention_dim: 12,
        mlp_ratio: 3.0,
        timestep_channels: 16,
        axes_dim: [2, 2, 2, 2],
        rope_theta: 2000.0,
        te_hidden_size: 4,
        te_intermediate_size: 12,
        te_out_layers: [0, 1, 2],
        max_sequence_length: 512,
        num_latent_channels: 1,
        vae_scale_factor: 8,
    }
}

fn exact_eq(a: &Array, b: &Array) -> bool {
    a.shape() == b.shape() && array_eq(a, b, false).unwrap().item::<bool>()
}

/// Leading `n` tokens of `a` along the sequence axis (axis 1 in `[B, S, C]`).
fn leading(a: &Array, n: i32) -> Array {
    let idx = Array::from_slice(&(0..n).collect::<Vec<i32>>(), &[n]);
    a.take_axis(&idx, 1).unwrap()
}

struct Fixture {
    t: Flux2Transformer,
    target: Array,     // [1, target_seq, in_channels]
    txt: Array,        // [1, txt_seq, joint]
    target_ids: Array, // [1, target_seq, 4]
    txt_ids: Array,    // [1, txt_seq, 4]
    ref_lat: Array,    // [1, ref_seq, in_channels]
    ref_ids: Array,    // [1, ref_seq, 4]
}

impl Fixture {
    fn load() -> Self {
        let w = Weights::from_file(FIXTURE).unwrap();
        let cfg = tiny_config();
        let t = Flux2Transformer::from_weights(&w, &cfg).unwrap();
        let target = w.require("hidden").unwrap().clone();
        let txt = w.require("encoder").unwrap().clone();
        let target_seq = target.shape()[1];
        let txt_seq = txt.shape()[1];
        // Synthetic reference tokens (the cache invariants don't depend on the values, only on
        // self-consistency across the three forwards). 3 ref tokens, in_channels = 4, t-offset 10.
        let ref_seq = 3i32;
        let ref_vals: Vec<f32> = (0..ref_seq * cfg.in_channels as i32)
            .map(|i| (i as f32) * 0.013 - 0.21)
            .collect();
        let ref_lat = Array::from_slice(&ref_vals, &[1, ref_seq, cfg.in_channels as i32]);
        Self {
            t,
            target,
            txt,
            target_ids: prepare_grid_ids(1, target_seq as usize, 0),
            txt_ids: prepare_text_ids(txt_seq as usize),
            ref_lat,
            ref_ids: prepare_grid_ids(1, ref_seq as usize, 10),
        }
    }

    fn ref_seq(&self) -> usize {
        self.ref_lat.shape()[1] as usize
    }

    /// Full `[txt, target, ref]` forward, optionally with the cache (extract or none).
    fn forward_full(&self, cache: Option<&Flux2KvCache>) -> Array {
        self.forward_full_with(&self.txt, cache)
    }

    fn forward_full_with(&self, txt: &Array, cache: Option<&Flux2KvCache>) -> Array {
        let img = concatenate_axis(&[&self.target, &self.ref_lat], 1).unwrap();
        let ids = concatenate_axis(&[&self.target_ids, &self.ref_ids], 1).unwrap();
        self.t
            .forward_with_cache(
                &Flux2ForwardInputs {
                    hidden_states: &img,
                    encoder_hidden_states: txt,
                    img_ids: &ids,
                    txt_ids: &self.txt_ids,
                    timestep: TS,
                    guidance: None,
                },
                cache,
            )
            .unwrap()
    }

    /// Target-only `[txt, target]` forward, with the cache splicing the stored ref K/V back in.
    fn forward_cached(&self, cache: &Flux2KvCache) -> Array {
        self.forward_cached_with(&self.txt, cache)
    }

    fn forward_cached_with(&self, txt: &Array, cache: &Flux2KvCache) -> Array {
        self.t
            .forward_with_cache(
                &Flux2ForwardInputs {
                    hidden_states: &self.target,
                    encoder_hidden_states: txt,
                    img_ids: &self.target_ids,
                    txt_ids: &self.txt_ids,
                    timestep: TS,
                    guidance: None,
                },
                Some(cache),
            )
            .unwrap()
    }
}

#[test]
fn extract_pass_equals_plain_forward() {
    let f = Fixture::load();
    let plain = f.forward_full(None);
    let cache = Flux2KvCache::new(1, 1);
    cache.configure(CacheMode::Extract, f.ref_seq());
    let extract = f.forward_full(Some(&cache));
    assert!(
        exact_eq(&plain, &extract),
        "extract mode must be byte-identical to the plain forward (it only stores K/V)"
    );
    assert!(
        cache.is_populated(),
        "extract must populate every layer slot"
    );
}

#[test]
fn cached_pass_reconstructs_extract_target_slice() {
    let f = Fixture::load();
    let cache = Flux2KvCache::new(1, 1);

    // Populate the cache from the full extract forward, then run the cached (target-only) forward
    // on the same target.
    cache.configure(CacheMode::Extract, f.ref_seq());
    let extract = f.forward_full(Some(&cache));

    cache.configure(CacheMode::Cached, f.ref_seq());
    let cached = f.forward_cached(&cache);

    let target_seq = f.target.shape()[1];
    // `forward` returns velocity over the image tokens: extract → [target, ref], cached → [target].
    assert_eq!(cached.shape()[1], target_seq);
    let extract_target = leading(&extract, target_seq);
    assert!(
        exact_eq(&cached, &extract_target),
        "cached forward must reproduce the target-token slice of the extract forward exactly"
    );
}

/// LoRA ⊥ cache: applying adapters to the transformer must not perturb the cache invariants. The
/// cache only depends on the (post-RoPE) K/V the linears produce, so `cached(X) == extract(X)[target]`
/// must still hold *exactly* with a LoRA installed on the attention projections — proving the
/// `-kv` variant's inherited LoRA path composes cleanly with the cache (sc-2646 + sc-2347).
#[test]
fn cache_invariants_hold_with_lora_installed() {
    use mlx_gen::adapters::{install_adapter, Adapter};

    let mut f = Fixture::load();
    // Install a non-trivial LoRA on a couple of double-block attention projections. The core
    // residual is `matmul(matmul(x, a), b)`, so a=[in,r], b=[r,out] (tiny inner = 16). Scale ≠ 0 so
    // it actually changes the K/V the cache stores.
    let r = 2i32;
    let a: Vec<f32> = (0..16 * r).map(|i| (i as f32) * 0.005 - 0.02).collect();
    let b: Vec<f32> = (0..r * 16).map(|i| (i as f32) * 0.004 - 0.015).collect();
    for proj in ["to_q", "to_v"] {
        install_adapter(
            &mut f.t,
            &format!("transformer_blocks.0.attn.{proj}"),
            Adapter::Lora {
                a: Array::from_slice(&a, &[16, r]),
                b: Array::from_slice(&b, &[r, 16]),
                scale: 0.7,
            },
        )
        .unwrap();
    }

    // Same two exact invariants as the dense case, now over the adapted transformer.
    let plain = f.forward_full(None);
    let cache = Flux2KvCache::new(1, 1);
    cache.configure(CacheMode::Extract, f.ref_seq());
    let extract = f.forward_full(Some(&cache));
    assert!(
        exact_eq(&plain, &extract),
        "with LoRA: extract must still be byte-identical to the plain forward"
    );

    cache.configure(CacheMode::Cached, f.ref_seq());
    let cached = f.forward_cached(&cache);
    let extract_target = leading(&extract, f.target.shape()[1]);
    assert!(
        exact_eq(&cached, &extract_target),
        "with LoRA: cached forward must still reproduce the extract target slice exactly"
    );
}

#[test]
fn cached_without_populated_cache_errors() {
    let f = Fixture::load();
    let cache = Flux2KvCache::new(1, 1);
    cache.configure(CacheMode::Cached, f.ref_seq());
    // No extract pass ran → the first cached attention layer finds an empty slot.
    let err =
        f.t.forward_with_cache(
            &Flux2ForwardInputs {
                hidden_states: &f.target,
                encoder_hidden_states: &f.txt,
                img_ids: &f.target_ids,
                txt_ids: &f.txt_ids,
                timestep: TS,
                guidance: None,
            },
            Some(&cache),
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("slot is empty"), "got: {err}");
}

// -------------------------------------------------------------------------------------------------
// True-CFG kv edit (guidance > 1 + negative prompt): each branch must splice ITS OWN reference K/V.
// -------------------------------------------------------------------------------------------------

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    assert_eq!(a.shape(), b.shape());
    mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap())
        .unwrap()
        .max(None)
        .unwrap()
        .item::<f32>()
}

/// The negative prompt: a deterministic transform of the positive text so the two branches'
/// prompt-dependent reference K/V differ.
fn cfg_negative(f: &Fixture) -> Array {
    mlx_rs::ops::add(
        mlx_rs::ops::multiply(&f.txt, Array::from_f32(-0.5)).unwrap(),
        Array::from_f32(0.1),
    )
    .unwrap()
}

/// `Flux2::generate`'s per-branch forward: `[target, ref]` on the extract pass, `[target]` plus the
/// branch cache afterwards (or the full no-cache forward for the non-kv oracle).
fn cfg_forward(
    f: &Fixture,
    negative: &Array,
    branch: CfgBranch,
    include_ref: bool,
    cache: Option<&Flux2KvCache>,
) -> Array {
    let txt = match branch {
        CfgBranch::Positive => &f.txt,
        CfgBranch::Negative => negative,
    };
    let out = match (include_ref, cache) {
        (false, Some(cache)) => f.forward_cached_with(txt, cache),
        _ => f.forward_full_with(txt, cache),
    };
    leading(&out, f.target.shape()[1])
}

const CFG_GUIDANCE: f32 = 2.0;

/// Non-kv oracle: both branches over the full `[txt, target, ref]` joint sequence, then
/// `neg + g·(pos − neg)` — the non-kv edit route at the same latent and timestep.
fn cfg_oracle(f: &Fixture, negative: &Array) -> Array {
    let v = cfg_forward(f, negative, CfgBranch::Positive, true, None);
    let vn = cfg_forward(f, negative, CfgBranch::Negative, true, None);
    mlx_rs::ops::add(
        &vn,
        mlx_rs::ops::multiply(
            mlx_rs::ops::subtract(&v, &vn).unwrap(),
            Array::from_f32(CFG_GUIDANCE),
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn cfg_kv_edit_matches_the_non_kv_forward_on_every_cached_step() {
    let f = Fixture::load();
    let negative = cfg_negative(&f);
    let oracle = cfg_oracle(&f, &negative);
    let mut caches = Flux2KvCfgCaches::new(1, 1, true, f.ref_seq());
    for step in 0..3 {
        let kv = caches
            .velocity(CFG_GUIDANCE, |branch, include_ref, cache| {
                Ok(cfg_forward(&f, &negative, branch, include_ref, Some(cache)))
            })
            .unwrap();
        let error = max_abs_diff(&kv, &oracle);
        eprintln!("cfg kv edit step {step}: max |kv - non-kv| = {error:e}");
        assert!(
            exact_eq(&kv, &oracle),
            "step {step}: per-branch kv CFG diverged from the non-kv forward (max {error:e})"
        );
    }

    // The shared single-cache shape (the mflux fork's, shipped before this fix) is measurably
    // wrong: the negative extract overwrites the positive slots, so the positive cached pass
    // attends over negative-branch reference K/V.
    let shared = Flux2KvCache::new(1, 1);
    shared.configure(CacheMode::Extract, f.ref_seq());
    let _ = cfg_forward(&f, &negative, CfgBranch::Positive, true, Some(&shared));
    let _ = cfg_forward(&f, &negative, CfgBranch::Negative, true, Some(&shared));
    shared.configure(CacheMode::Cached, f.ref_seq());
    let v = cfg_forward(&f, &negative, CfgBranch::Positive, false, Some(&shared));
    let vn = cfg_forward(&f, &negative, CfgBranch::Negative, false, Some(&shared));
    let broken = mlx_rs::ops::add(
        &vn,
        mlx_rs::ops::multiply(
            mlx_rs::ops::subtract(&v, &vn).unwrap(),
            Array::from_f32(CFG_GUIDANCE),
        )
        .unwrap(),
    )
    .unwrap();
    let error = max_abs_diff(&broken, &oracle);
    eprintln!("shared-cache (pre-fix) cached step: max |kv - non-kv| = {error:e}");
    assert!(
        error > 0.0,
        "the fixture must discriminate the shared-cache defect"
    );
}

#[test]
fn extract_materializes_every_slot_with_the_step_output() {
    let f = Fixture::load();
    let negative = cfg_negative(&f);
    let mut caches = Flux2KvCfgCaches::new(1, 1, true, f.ref_seq());
    let _ = caches
        .velocity(CFG_GUIDANCE, |branch, include_ref, cache| {
            Ok(cfg_forward(&f, &negative, branch, include_ref, Some(cache)))
        })
        .unwrap();
    let expected_slot_bytes = 2 * 8 * f.ref_seq() * 4; // one [1, 2, ref, 8] f32 array
    for branch in [CfgBranch::Positive, CfgBranch::Negative] {
        let slots = caches.cache(branch).unwrap().slot_arrays();
        assert_eq!(
            slots.len(),
            4,
            "K and V for one double and one single layer"
        );
        for slot in &slots {
            // A materialized slot holds only its own buffer; an unevaluated `take` would keep the
            // layer's whole post-RoPE `[txt, target, ref]` K/V alive until step 1.
            assert!(
                mlx_gen::array::is_materialized(slot),
                "{branch:?} slot left lazy after the extract step"
            );
            assert_eq!(slot.nbytes(), expected_slot_bytes);
        }
    }
}

// -------------------------------------------------------------------------------------------------
// SC-20686 Metal lane: the observer hooks at the reference-K/V boundaries are inert when no
// campaign is armed and leave the forward bit-identical when one is.
// -------------------------------------------------------------------------------------------------

mod sc20686 {
    use super::*;
    use mlx_gen::sc20686 as obs;
    use serde_json::Value;

    const SOURCE_REF: &str = "fedcba9876543210fedcba9876543210fedcba98";

    fn activate() -> (
        tempfile::TempDir,
        obs::Scope,
        std::rc::Rc<std::cell::RefCell<Vec<Value>>>,
    ) {
        let root = tempfile::tempdir().unwrap();
        let snapshot = root.path().join("0123456789abcdef0123456789abcdef01234567");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("config.json"), b"{}").unwrap();
        let request = obs::request_output(root.path().join("e.jsonl"), SOURCE_REF, "sequential")
            .unwrap()
            .arm();
        let (instruments, events) = obs::capture_instruments();
        let scope = obs::activate_with_instruments(
            &snapshot,
            &mlx_gen::CancelFlag::new(),
            "flux2_klein_9b_kv_edit",
            obs::RequestFacts {
                batch: 1,
                frames: 1,
                width: 16,
                height: 16,
                prompt_sha256: "a".repeat(64),
                guidance: "1".into(),
                reference_count: 1,
            },
            instruments,
        )
        .unwrap()
        .expect("armed");
        drop(request);
        (root, scope, events)
    }

    fn negative_text(f: &Fixture) -> Array {
        mlx_rs::ops::add(
            mlx_rs::ops::multiply(&f.txt, Array::from_f32(-0.5)).unwrap(),
            Array::from_f32(0.1),
        )
        .unwrap()
    }

    /// One branch forward exactly as `Flux2::generate`'s `run` builds it: `[target, ref]` on the
    /// extract pass, `[target]` plus the branch's cache afterwards; keep the target tokens.
    fn branch_forward(
        f: &Fixture,
        negative: &Array,
        branch: CfgBranch,
        include_ref: bool,
        cache: &Flux2KvCache,
    ) -> Array {
        let txt = match branch {
            CfgBranch::Positive => &f.txt,
            CfgBranch::Negative => negative,
        };
        let out = if include_ref {
            f.forward_full_with(txt, Some(cache))
        } else {
            f.forward_cached_with(txt, cache)
        };
        leading(&out, f.target.shape()[1])
    }

    fn of(events: &[Value], phase: &str) -> Vec<Value> {
        events
            .iter()
            .filter(|event| event["phase"] == phase)
            .cloned()
            .collect()
    }

    #[test]
    fn kv_cache_lifecycle_is_observed_without_changing_the_forward() {
        let f = Fixture::load();
        let negative = negative_text(&f);
        let run = |caches: &mut Flux2KvCfgCaches| {
            caches
                .velocity(2.0, |branch, include_ref, cache| {
                    Ok(branch_forward(&f, &negative, branch, include_ref, cache))
                })
                .unwrap()
        };
        let mut off = Flux2KvCfgCaches::new(1, 1, true, f.ref_seq());
        assert_eq!(
            off.cache(CfgBranch::Positive)
                .unwrap()
                .sc20686_cached_read(mlx_gen_flux2::Stream::Double, 0),
            None
        );
        let extract_off = run(&mut off);
        let cached_off = run(&mut off);
        drop(off);

        let (_root, scope, events) = activate();
        obs::bind_geometry(
            obs::KvGeometry {
                layers: 2,
                heads: 2,
                head_dimension: 8,
                sq: f.target.shape()[1] as u64,
                skv: f.ref_seq() as u64,
            },
            None,
            "joint-unmasked",
            "flux2-4-axis",
        );
        let mut caches = Flux2KvCfgCaches::new(1, 1, true, f.ref_seq());
        let extract_on = run(&mut caches);
        let cached_on = run(&mut caches);
        assert!(
            exact_eq(&extract_off, &extract_on),
            "extract changed under the observer"
        );
        assert!(
            exact_eq(&cached_off, &cached_on),
            "cached forward changed under the observer"
        );
        drop(caches);
        obs::observe_generation_end();
        drop(scope);

        let events = events.borrow();
        let created = of(&events, "cross-kv-created");
        // One cache per CFG branch: 2 branches × (1 double + 1 single) slots, each extracted once.
        assert_eq!(
            created.len(),
            4,
            "each branch extracts its own two slots once"
        );
        let slice_bytes = 2 * (2 * f.ref_seq() * 8 * 4) as u64; // K+V [1,2,ref,8] f32
        assert!(created.iter().all(|e| e["persistent_bytes"] == slice_bytes
            && e["kv_batch"] == 1
            && e["operation"] == "Flux2KvCache::extract"));
        let released = of(&events, "cross-kv-released");
        assert_eq!(released.len(), 4);
        assert!(
            released
                .iter()
                .all(|e| e["operation"] == "Flux2KvCache::drop"),
            "no branch overwrites another branch's slots"
        );
        let reads = of(&events, "cross-kv-read");
        assert_eq!(reads.len(), 4, "every slot is read once on the cached step");
        let ids: Vec<_> = created.iter().map(|e| e["cache_id"].clone()).collect();
        for id in &ids {
            assert_eq!(reads.iter().filter(|r| &r["cache_id"] == id).count(), 1);
        }
        assert!(reads
            .iter()
            .all(|read| read["operation"] == "Flux2KvCache::cached(joint-attention)"));
        let metrics = of(&events, "metrics").pop().unwrap();
        assert_eq!(metrics["current_persistent_bytes"], 4 * slice_bytes);
        assert_eq!(metrics["minimum_cache_reads"], 1);
        assert_eq!(metrics["reference_runtime_attribution_available"], false);
    }

    #[test]
    fn cached_read_window_contains_the_reference_splice() {
        // A long reference makes the spliced K/V dominate the attention output, so the window
        // placement is visible above allocator rounding: [1, 2, ~2k, 8] f32 K and V ≈ 260 KiB
        // (the extract step's attention over ~2k tokens stays well under 64 MiB).
        let mut f = Fixture::load();
        let long_ref = 2048;
        let values: Vec<f32> = (0..long_ref * 4)
            .map(|i| ((i % 97) as f32) * 0.01 - 0.4)
            .collect();
        f.ref_lat = Array::from_slice(&values, &[1, long_ref, 4]);
        f.ref_ids = prepare_grid_ids(1, long_ref as usize, 10);
        let negative = negative_text(&f);
        let (_root, scope, events) = activate();
        let mut caches = Flux2KvCfgCaches::new(1, 1, false, f.ref_seq());
        for _step in 0..2 {
            caches
                .velocity(1.0, |branch, include_ref, cache| {
                    Ok(branch_forward(&f, &negative, branch, include_ref, cache))
                })
                .unwrap();
        }
        drop(caches);
        drop(scope);
        let events = events.borrow();
        let reads = of(&events, "cross-kv-read");
        assert_eq!(reads.len(), 2);
        // The spliced K and V are [1, 2, txt + target + ref, 8] f32 each; the window must contain
        // their materialization, not open after it.
        let joint = f.txt.shape()[1] as u64 + f.target.shape()[1] as u64 + f.ref_seq() as u64;
        let splice = 2 * 2 * joint * 8 * 4;
        for read in &reads {
            assert!(
                read["transient_bytes"].as_u64().unwrap() >= splice,
                "read transient {} excludes the {splice}-byte cached-K/V splice",
                read["transient_bytes"]
            );
        }
    }

    #[test]
    fn edit_reference_slice_is_recomputed_on_double_stream_forwards_only() {
        let f = Fixture::load();
        let off = f.forward_full(None);
        let (_root, scope, events) = activate();
        obs::bind_geometry(
            obs::KvGeometry {
                layers: 1,
                heads: 2,
                head_dimension: 8,
                sq: f.target.shape()[1] as u64,
                skv: f.ref_seq() as u64,
            },
            None,
            "joint-unmasked",
            "flux2-4-axis",
        );
        let on = {
            let _reference = obs::reference_forward(true);
            f.forward_full(None)
        };
        // A forward that does not carry the reference tokens records nothing.
        let _ = f.forward_full(None);
        obs::observe_generation_end();
        drop(scope);
        assert!(
            exact_eq(&off, &on),
            "the reference-slice window changed the forward"
        );
        let events = events.borrow();
        let created = of(&events, "cross-kv-created");
        assert_eq!(
            created.len(),
            1,
            "one double-stream layer, one reference forward"
        );
        let dense = 2 * (2 * f.ref_seq() * 8 * 4) as u64;
        assert_eq!(created[0]["persistent_bytes"], 0);
        assert_eq!(created[0]["transient_bytes"], dense);
        assert_eq!(
            created[0]["operation"],
            "DoubleAttention::to_k/to_v(reference-slice)"
        );
        let reads = of(&events, "cross-kv-read");
        assert_eq!(reads.len(), 1);
        assert_eq!(reads[0]["transient_bytes"], dense);
        assert_eq!(
            reads[0]["operation"],
            "DoubleAttention::attention(joint-context-non-attributable)"
        );
        let metrics = of(&events, "metrics").pop().unwrap();
        assert_eq!(metrics["current_persistent_bytes"], 0);
        assert_eq!(metrics["reused_requests"], 1);
    }
}
