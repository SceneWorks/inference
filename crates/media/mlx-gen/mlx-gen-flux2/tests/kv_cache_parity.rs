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
    prepare_grid_ids, prepare_text_ids, CacheMode, Flux2Config, Flux2ForwardInputs, Flux2KvCache,
    Flux2Transformer,
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
        let img = concatenate_axis(&[&self.target, &self.ref_lat], 1).unwrap();
        let ids = concatenate_axis(&[&self.target_ids, &self.ref_ids], 1).unwrap();
        self.t
            .forward_with_cache(
                &Flux2ForwardInputs {
                    hidden_states: &img,
                    encoder_hidden_states: &self.txt,
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
        self.t
            .forward_with_cache(
                &Flux2ForwardInputs {
                    hidden_states: &self.target,
                    encoder_hidden_states: &self.txt,
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
        let off_cache = Flux2KvCache::new(1, 1);
        assert_eq!(
            off_cache.sc20686_cached_read(mlx_gen_flux2::Stream::Double, 0),
            None
        );
        off_cache.configure(CacheMode::Extract, f.ref_seq());
        let extract_off = f.forward_full(Some(&off_cache));
        off_cache.configure(CacheMode::Cached, f.ref_seq());
        let cached_off = f.forward_cached(&off_cache);
        drop(off_cache);

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
        let cache = Flux2KvCache::new(1, 1);
        cache.configure(CacheMode::Extract, f.ref_seq());
        let extract_on = f.forward_full(Some(&cache));
        // The CFG negative pass re-extracts every slot: a rebuild, not a second live cache.
        let _ = f.forward_full(Some(&cache));
        cache.configure(CacheMode::Cached, f.ref_seq());
        let cached_on = f.forward_cached(&cache);
        assert!(
            exact_eq(&extract_off, &extract_on),
            "extract changed under the observer"
        );
        assert!(
            exact_eq(&cached_off, &cached_on),
            "cached forward changed under the observer"
        );
        drop(cache);
        obs::observe_generation_end();
        drop(scope);

        let events = events.borrow();
        let created = of(&events, "cross-kv-created");
        assert_eq!(created.len(), 4, "two slots extracted twice");
        let slice_bytes = 2 * (2 * f.ref_seq() * 8 * 4) as u64; // K+V [1,2,ref,8] f32
        assert!(created.iter().all(|e| e["persistent_bytes"] == slice_bytes
            && e["kv_batch"] == 1
            && e["operation"] == "Flux2KvCache::extract"));
        let released = of(&events, "cross-kv-released");
        let operations: Vec<_> = released.iter().map(|e| e["operation"].clone()).collect();
        assert_eq!(
            operations,
            vec![
                "Flux2KvCache::extract(rebuild)",
                "Flux2KvCache::extract(rebuild)",
                "Flux2KvCache::drop",
                "Flux2KvCache::drop"
            ]
        );
        let reads = of(&events, "cross-kv-read");
        assert_eq!(reads.len(), 2, "one cached read per slot");
        let live_ids = [
            created[2]["cache_id"].clone(),
            created[3]["cache_id"].clone(),
        ];
        assert!(reads.iter().all(|read| live_ids.contains(&read["cache_id"])
            && read["operation"] == "Flux2KvCache::cached(joint-attention)"));
        let metrics = of(&events, "metrics").pop().unwrap();
        assert_eq!(metrics["current_persistent_bytes"], 2 * slice_bytes);
        assert_eq!(
            metrics["minimum_cache_reads"], 0,
            "the rebuilt slots were never read"
        );
        assert_eq!(metrics["reference_runtime_attribution_available"], false);
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
