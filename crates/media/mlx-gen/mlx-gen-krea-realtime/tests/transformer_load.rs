//! Krea Realtime 14B converter + load-path validation against the **S1 tensor inventory**, with
//! synthesized fixtures only — never the real 28.58 GB checkpoint (sc-8435 S2, non-gated).
//!
//! Ground truth (S1 audit): Krea Realtime 14B is Wan 2.1 T2V 14B weight-for-weight — a transformer-only
//! checkpoint of 1095 parameter tensors (+ the dropped `freqs` RoPE buffer), all F16, shipped in one of
//! two on-disk layouts:
//!
//!   * **single-file** — every key prefixed `model.`, and
//!   * **sharded** `transformer/` — the same names without the prefix.
//!
//! These tests build the native tensor **names + shapes** from independent literals (the audit
//! constants typed here, *not* read from the crate's own [`WanModelConfig`] preset) so that a bug in
//! either the preset or the sanitizer surfaces as a mismatch rather than being masked. MLX arrays are
//! lazily evaluated, so the synthesized `zeros` are metadata handles — building the full inventory
//! never materializes a byte, and no test forces evaluation.
//!
//! Coverage: (i) both layouts normalize to the same internal names; (ii) the representative audit
//! shapes survive the sanitizer's conv→Linear reshape and `ffn.0/2`→`fc1/fc2` renames; (iii) the
//! F16 checkpoint is cast to bf16; (iv) a missing / extra / wrong-shape tensor is rejected with the
//! offending key named. A final test loads a tiny full inventory into the real reused Wan DiT.

use std::collections::HashMap;

use mlx_gen_krea_realtime::{
    load_krea_realtime_transformer, sanitize_krea_realtime_transformer, verify_transformer_tensors,
    KreaRealtimeConfig, TRANSFORMER_DTYPE,
};
use mlx_rs::{Array, Dtype};

/// A Wan-2.1 transformer geometry, as independent literals (not read from the crate preset).
struct Dims {
    dim: i32,
    layers: usize,
    ffn: i32,
    text_dim: i32,
    freq: i32,
    in_dim: i32,
    out_dim: i32,
    patch: (i32, i32, i32),
}

/// The shipped Krea Realtime / Wan-2.1-T2V-14B geometry from the S1 audit, as literals.
fn audit_dims() -> Dims {
    Dims {
        dim: 5120,
        layers: 40,
        ffn: 13824,
        text_dim: 4096,
        freq: 256,
        in_dim: 16,
        out_dim: 16,
        patch: (1, 2, 2),
    }
}

/// A tiny but structurally-complete geometry for the full-inventory / load-wiring tests (no real
/// weights, cheap even if MLX were eager).
fn tiny_dims() -> Dims {
    Dims {
        dim: 128,
        layers: 2,
        ffn: 256,
        text_dim: 64,
        freq: 64,
        in_dim: 16,
        out_dim: 16,
        patch: (1, 2, 2),
    }
}

/// A [`KreaRealtimeConfig`] whose reused Wan DiT matches [`tiny_dims`].
fn tiny_config() -> KreaRealtimeConfig {
    let d = tiny_dims();
    let mut cfg = KreaRealtimeConfig::krea_realtime_14b();
    cfg.wan.dim = d.dim as usize;
    cfg.wan.num_heads = 2; // head_dim = 64
    cfg.wan.num_layers = d.layers;
    cfg.wan.ffn_dim = d.ffn as usize;
    cfg.wan.freq_dim = d.freq as usize;
    cfg.wan.text_dim = d.text_dim as usize;
    cfg
}

/// The full **native** (pre-sanitize) Wan-2.1 transformer tensor inventory for `d`: every key at its
/// native name + native shape, including the single-file/sharded-shared `freqs` buffer the sanitizer
/// drops. This is the inverse of the crate's internal expectation, written independently here.
fn native_specs(d: &Dims) -> Vec<(String, Vec<i32>)> {
    let (dim, ffn, text_dim, freq) = (d.dim, d.ffn, d.text_dim, d.freq);
    let prod = d.patch.0 * d.patch.1 * d.patch.2;
    let head_out = d.out_dim * prod;
    let mut v: Vec<(String, Vec<i32>)> = Vec::new();
    let mut push = |name: &str, shape: Vec<i32>| v.push((name.to_string(), shape));

    // Patch embed is a Conv3d on disk: [dim, in, kT, kH, kW].
    push(
        "patch_embedding.weight",
        vec![dim, d.in_dim, d.patch.0, d.patch.1, d.patch.2],
    );
    push("patch_embedding.bias", vec![dim]);
    // Text/time embeddings are Sequentials (`.0` / `.2`).
    push("text_embedding.0.weight", vec![dim, text_dim]);
    push("text_embedding.0.bias", vec![dim]);
    push("text_embedding.2.weight", vec![dim, dim]);
    push("text_embedding.2.bias", vec![dim]);
    push("time_embedding.0.weight", vec![dim, freq]);
    push("time_embedding.0.bias", vec![dim]);
    push("time_embedding.2.weight", vec![dim, dim]);
    push("time_embedding.2.bias", vec![dim]);
    // time_projection Sequential's Linear is `.1`.
    push("time_projection.1.weight", vec![6 * dim, dim]);
    push("time_projection.1.bias", vec![6 * dim]);
    push("head.head.weight", vec![head_out, dim]);
    push("head.head.bias", vec![head_out]);
    push("head.modulation", vec![1, 2, dim]);
    // The RoPE buffer the sanitizer drops (shape irrelevant).
    push("freqs", vec![1024, 64]);

    for i in 0..d.layers {
        let p = format!("blocks.{i}");
        push(&format!("{p}.modulation"), vec![1, 6, dim]);
        for attn in ["self_attn", "cross_attn"] {
            for proj in ["q", "k", "v", "o"] {
                push(&format!("{p}.{attn}.{proj}.weight"), vec![dim, dim]);
                push(&format!("{p}.{attn}.{proj}.bias"), vec![dim]);
            }
            push(&format!("{p}.{attn}.norm_q.weight"), vec![dim]);
            push(&format!("{p}.{attn}.norm_k.weight"), vec![dim]);
        }
        push(&format!("{p}.norm3.weight"), vec![dim]);
        push(&format!("{p}.norm3.bias"), vec![dim]);
        // FFN Sequential: Linear `.0` (dim→ffn), GELU, Linear `.2` (ffn→dim).
        push(&format!("{p}.ffn.0.weight"), vec![ffn, dim]);
        push(&format!("{p}.ffn.0.bias"), vec![ffn]);
        push(&format!("{p}.ffn.2.weight"), vec![dim, ffn]);
        push(&format!("{p}.ffn.2.bias"), vec![dim]);
    }
    v
}

/// One lazy F16 `zeros` tensor at `shape` (metadata handle; never materialized unless evaluated).
fn f16_zeros(shape: &[i32]) -> Array {
    Array::zeros::<f32>(shape)
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap()
}

/// Build a native tensor map from `specs`, applying `prefix` to every key (`"model."` for the
/// single-file layout, `""` for the sharded layout).
fn build_map(specs: &[(String, Vec<i32>)], prefix: &str) -> HashMap<String, Array> {
    specs
        .iter()
        .map(|(name, shape)| (format!("{prefix}{name}"), f16_zeros(shape)))
        .collect()
}

fn sorted_keys(map: &HashMap<String, Array>) -> Vec<String> {
    let mut ks: Vec<String> = map.keys().cloned().collect();
    ks.sort();
    ks
}

/// (i) Both on-disk layouts collapse to the identical internal name set, `freqs` is dropped, and each
/// verifies against the real `wan21_t2v_14b` preset (here at the tiny geometry).
#[test]
fn both_layouts_normalize_to_identical_internal_names() {
    let d = tiny_dims();
    let specs = native_specs(&d);

    let single = build_map(&specs, "model.");
    let sharded = build_map(&specs, "");
    assert_eq!(single.len(), sharded.len());

    let internal_single = sanitize_krea_realtime_transformer(single).unwrap();
    let internal_sharded = sanitize_krea_realtime_transformer(sharded).unwrap();

    assert_eq!(
        sorted_keys(&internal_single),
        sorted_keys(&internal_sharded),
        "single-file (model.-prefixed) and sharded layouts must produce the same internal names"
    );
    assert!(
        !internal_single.contains_key("freqs") && !internal_single.contains_key("model.freqs"),
        "the freqs RoPE buffer must be dropped, not carried through"
    );

    let cfg = tiny_config();
    verify_transformer_tensors(&internal_single, &cfg.wan).expect("single-file layout is complete");
    verify_transformer_tensors(&internal_sharded, &cfg.wan).expect("sharded layout is complete");
}

/// (ii) The representative S1-audit shapes survive the sanitizer's conv→Linear reshape and Sequential
/// renames. Uses a small 6-tensor map at the **real audit shapes** (independent of the crate preset).
#[test]
fn representative_shapes_match_the_audit() {
    let d = audit_dims();
    let prod = d.patch.0 * d.patch.1 * d.patch.2;
    let mut native: HashMap<String, Array> = HashMap::new();
    native.insert(
        "patch_embedding.weight".into(),
        f16_zeros(&[d.dim, d.in_dim, d.patch.0, d.patch.1, d.patch.2]),
    );
    native.insert(
        "blocks.0.self_attn.q.weight".into(),
        f16_zeros(&[d.dim, d.dim]),
    );
    native.insert("blocks.0.ffn.0.weight".into(), f16_zeros(&[d.ffn, d.dim]));
    native.insert(
        "head.head.weight".into(),
        f16_zeros(&[d.out_dim * prod, d.dim]),
    );
    native.insert(
        "time_projection.1.weight".into(),
        f16_zeros(&[6 * d.dim, d.dim]),
    );
    native.insert(
        "text_embedding.0.weight".into(),
        f16_zeros(&[d.dim, d.text_dim]),
    );

    let internal = sanitize_krea_realtime_transformer(native).unwrap();

    assert_eq!(
        internal["blocks.0.self_attn.q.weight"].shape(),
        &[5120, 5120]
    );
    assert_eq!(internal["blocks.0.ffn.fc1.weight"].shape(), &[13824, 5120]);
    // Conv `[5120,16,1,2,2]` flattens to the Linear `[5120,64]`.
    assert_eq!(internal["patch_embedding_proj.weight"].shape(), &[5120, 64]);
    assert_eq!(internal["head.head.weight"].shape(), &[64, 5120]);
    assert_eq!(internal["time_projection.weight"].shape(), &[30720, 5120]);
    assert_eq!(internal["text_embedding_0.weight"].shape(), &[5120, 4096]);
}

/// (iii) The F16 checkpoint is cast to bf16 on sanitize.
#[test]
fn f16_checkpoint_is_cast_to_bf16() {
    let specs = native_specs(&tiny_dims());
    let sharded = build_map(&specs, "");
    assert_eq!(
        sharded["blocks.0.self_attn.q.weight"].dtype(),
        Dtype::Float16,
        "the native Krea Realtime checkpoint is F16"
    );

    let internal = sanitize_krea_realtime_transformer(sharded).unwrap();
    assert_eq!(TRANSFORMER_DTYPE, Dtype::Bfloat16);
    assert_eq!(
        internal["blocks.0.self_attn.q.weight"].dtype(),
        Dtype::Bfloat16,
        "sanitize must cast the transformer to bf16"
    );
    assert_eq!(
        internal["patch_embedding_proj.weight"].dtype(),
        Dtype::Bfloat16
    );
}

/// (iv) A missing, extra, or wrong-shape tensor is rejected, with the offending key named; the clean
/// inventory passes.
#[test]
fn incomplete_or_mismatched_inventory_is_rejected() {
    let d = tiny_dims();
    let cfg = tiny_config();
    let specs = native_specs(&d);

    // Clean baseline passes.
    let clean = sanitize_krea_realtime_transformer(build_map(&specs, "")).unwrap();
    verify_transformer_tensors(&clean, &cfg.wan).expect("clean inventory verifies");

    // Missing.
    let mut missing = build_map(&specs, "");
    missing.remove("blocks.1.self_attn.k.weight");
    let missing = sanitize_krea_realtime_transformer(missing).unwrap();
    let err = verify_transformer_tensors(&missing, &cfg.wan)
        .expect_err("a missing tensor must be rejected")
        .to_string();
    assert!(
        err.contains("blocks.1.self_attn.k.weight"),
        "missing key must be named: {err}"
    );

    // Extra.
    let mut extra = build_map(&specs, "");
    extra.insert("blocks.0.bogus.weight".into(), f16_zeros(&[4, 4]));
    let extra = sanitize_krea_realtime_transformer(extra).unwrap();
    let err = verify_transformer_tensors(&extra, &cfg.wan)
        .expect_err("an extra tensor must be rejected")
        .to_string();
    assert!(
        err.contains("blocks.0.bogus.weight"),
        "extra key must be named: {err}"
    );

    // Wrong shape.
    let mut wrong = build_map(&specs, "");
    wrong.insert("blocks.0.self_attn.q.weight".into(), f16_zeros(&[8, 8]));
    let wrong = sanitize_krea_realtime_transformer(wrong).unwrap();
    let err = verify_transformer_tensors(&wrong, &cfg.wan)
        .expect_err("a wrong-shape tensor must be rejected")
        .to_string();
    assert!(
        err.contains("blocks.0.self_attn.q.weight") && err.contains("wrong-shape"),
        "wrong-shape key must be named: {err}"
    );
}

/// The load path builds the real reused Wan DiT from a tiny full inventory (both the sanitize and the
/// `WanTransformer::from_weights` wiring), for both on-disk layouts.
#[test]
fn load_builds_the_reused_wan_transformer() {
    let cfg = tiny_config();
    let specs = native_specs(&tiny_dims());

    for prefix in ["", "model."] {
        let dit = load_krea_realtime_transformer(build_map(&specs, prefix), &cfg)
            .unwrap_or_else(|e| panic!("load ({prefix:?} layout): {e}"));
        assert_eq!(dit.num_blocks(), cfg.wan.num_layers);
    }
}
