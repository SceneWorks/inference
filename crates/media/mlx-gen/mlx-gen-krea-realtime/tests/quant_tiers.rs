//! Krea Realtime 14B **quant tiers** (sc-15203, S19) — torch-free, tiny random-weight DiT fixtures
//! packed with the very same reused Wan packer the rehost uses (no real 28.58 GB checkpoint).
//!
//! Krea Realtime ships three tiers — bf16 / Q8 / Q4 — and the quantized ones ship **pre-quantized
//! (packed) on disk**, not quantized on the fly. Because Krea Realtime is Wan-2.1-14B weight-for-weight,
//! the whole packed path is reuse: `mlx_gen_wan::convert::quantize_wan_transformer` writes the tier and
//! `mlx_gen_wan::WanTransformer::from_weights` consumes it. These gates make a broken tier FAIL a
//! specific assertion rather than silently degrading:
//!
//!   * **The packed tier is really packed** — after loading a Q4/Q8 snapshot the quantize-predicate
//!     Linears report `is_quantized()` and the out-of-predicate ones (embeddings / `time_projection` /
//!     head) do **not**. A load that silently fell back to a dense base would fail.
//!   * **Pre-quantized ≡ load-time quantized (bit-exact)** — the AR forward of a snapshot packed on disk
//!     is bit-identical to the same dense snapshot quantized at load. This is the strongest available
//!     statement that the consume path is real: the packed codes are being used, at the right group
//!     scales, in the right Linears.
//!   * **The tier changes the numerics** — Q4 and Q8 forwards both differ from bf16, and differ from
//!     each other. A "packed" path that quietly served bf16 would pass the shape checks but fail here.
//!   * **The u32 codes survive the converter** — the bf16 cast must not touch the packed weight.
//!   * **LoRA works on a packed base** — a synthetic Wan-family LoRA measurably shifts the *packed*
//!     forward, every target resolves, scale-0 is a bit-exact no-op on every tier, and the base is
//!     **still packed** afterwards (never dequantized). This is the epic-10043 / sc-10578
//!     additive-on-packed property, for Krea.
//!   * **Verification is exact on a packed tier** — a missing `.scales` / a stray extra tensor fails.

use std::collections::HashMap;
use std::path::PathBuf;

use mlx_gen::adapters::loader::apply_adapters_strict;
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::{AdapterKind, AdapterSpec, Quant};
use mlx_gen_krea_realtime::{
    convert_krea_realtime_tier, convert_krea_realtime_tier_with_config,
    expected_transformer_tensors, load_krea_realtime_transformer,
    load_krea_realtime_transformer_with_quant, quantize_krea_realtime_transformer,
    resolve_load_time_quant, resolve_snapshot_quant, sanitize_krea_realtime_transformer,
    verify_transformer_tensors, CausalKreaTransformer, KreaRealtimeConfig, WanQuant, DIT_FILE,
    MODEL_ID, PACKED_LINEARS_PER_BLOCK,
};
use mlx_rs::{Array, Dtype};

// ── Tiny geometry + fixtures (mirrors tests/style_lora.rs) ───────────────────────────────────────

/// A tiny but structurally-complete Krea / Wan-2.1 geometry: dim 64, 2 heads, 2 layers. `dim` and
/// `ffn_dim` are both multiples of the group size 64, so every quantize-predicate Linear packs.
fn tiny_cfg() -> KreaRealtimeConfig {
    let mut c = KreaRealtimeConfig::krea_realtime_14b();
    c.wan.dim = 64;
    c.wan.num_heads = 2; // head_dim 32
    c.wan.num_layers = 2;
    c.wan.ffn_dim = 128;
    c.wan.freq_dim = 32;
    c.wan.text_dim = 32;
    c.wan.text_len = 8;
    c.wan.in_dim = 16;
    c.wan.out_dim = 16;
    c.wan.patch_size = (1, 2, 2);
    c.ar.num_frames_per_block = 2;
    c.ar.frame_seq_length = 4; // h·w
    c.ar.local_attn_size = -1; // global
    c.ar.sink_size = 0;
    c.ar.seq_length = 100_000;
    c
}

/// Deterministic (seedless, reproducible) fill: `bias + scale · U(-0.5, 0.5)` via xorshift64.
fn det_fill(shape: &[i32], seed: u64, scale: f32, bias: f32, dtype: Dtype) -> Array {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let mut s = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x1234_5678);
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = (s >> 11) as f64 / (1u64 << 53) as f64;
        data.push(bias + scale * (u as f32 - 0.5));
    }
    Array::from_slice(&data, shape).as_dtype(dtype).unwrap()
}

/// Full **native** (pre-sanitize) Wan-2.1 transformer inventory for the tiny geometry (F16).
fn native_random_map(cfg: &KreaRealtimeConfig) -> HashMap<String, Array> {
    let w = &cfg.wan;
    let dim = w.dim as i32;
    let ffn = w.ffn_dim as i32;
    let text_dim = w.text_dim as i32;
    let freq = w.freq_dim as i32;
    let (pt, ph, pw) = w.patch_size;
    let (pt, ph, pw) = (pt as i32, ph as i32, pw as i32);
    let head_out = w.out_dim as i32 * pt * ph * pw;

    let mut map: HashMap<String, Array> = HashMap::new();
    let mut seed = 1u64;
    let mut put =
        |map: &mut HashMap<String, Array>, name: &str, shape: &[i32], scale: f32, bias: f32| {
            seed += 1;
            map.insert(
                name.to_string(),
                det_fill(shape, seed, scale, bias, Dtype::Float16),
            );
        };

    put(
        &mut map,
        "patch_embedding.weight",
        &[dim, w.in_dim as i32, pt, ph, pw],
        0.1,
        0.0,
    );
    put(&mut map, "patch_embedding.bias", &[dim], 0.05, 0.0);
    put(
        &mut map,
        "text_embedding.0.weight",
        &[dim, text_dim],
        0.1,
        0.0,
    );
    put(&mut map, "text_embedding.0.bias", &[dim], 0.05, 0.0);
    put(&mut map, "text_embedding.2.weight", &[dim, dim], 0.1, 0.0);
    put(&mut map, "text_embedding.2.bias", &[dim], 0.05, 0.0);
    put(&mut map, "time_embedding.0.weight", &[dim, freq], 0.1, 0.0);
    put(&mut map, "time_embedding.0.bias", &[dim], 0.05, 0.0);
    put(&mut map, "time_embedding.2.weight", &[dim, dim], 0.1, 0.0);
    put(&mut map, "time_embedding.2.bias", &[dim], 0.05, 0.0);
    put(
        &mut map,
        "time_projection.1.weight",
        &[6 * dim, dim],
        0.1,
        0.0,
    );
    put(&mut map, "time_projection.1.bias", &[6 * dim], 0.05, 0.0);
    put(&mut map, "head.head.weight", &[head_out, dim], 0.1, 0.0);
    put(&mut map, "head.head.bias", &[head_out], 0.05, 0.0);
    put(&mut map, "head.modulation", &[1, 2, dim], 0.05, 0.0);
    put(&mut map, "freqs", &[1024, 32], 0.0, 0.0); // dropped on sanitize

    for i in 0..w.num_layers {
        let p = format!("blocks.{i}");
        put(
            &mut map,
            &format!("{p}.modulation"),
            &[1, 6, dim],
            0.05,
            0.0,
        );
        for attn in ["self_attn", "cross_attn"] {
            for proj in ["q", "k", "v", "o"] {
                put(
                    &mut map,
                    &format!("{p}.{attn}.{proj}.weight"),
                    &[dim, dim],
                    0.12,
                    0.0,
                );
                put(
                    &mut map,
                    &format!("{p}.{attn}.{proj}.bias"),
                    &[dim],
                    0.03,
                    0.0,
                );
            }
            put(
                &mut map,
                &format!("{p}.{attn}.norm_q.weight"),
                &[dim],
                0.1,
                1.0,
            );
            put(
                &mut map,
                &format!("{p}.{attn}.norm_k.weight"),
                &[dim],
                0.1,
                1.0,
            );
        }
        put(&mut map, &format!("{p}.norm3.weight"), &[dim], 0.1, 1.0);
        put(&mut map, &format!("{p}.norm3.bias"), &[dim], 0.03, 0.0);
        put(
            &mut map,
            &format!("{p}.ffn.0.weight"),
            &[ffn, dim],
            0.1,
            0.0,
        );
        put(&mut map, &format!("{p}.ffn.0.bias"), &[ffn], 0.03, 0.0);
        put(
            &mut map,
            &format!("{p}.ffn.2.weight"),
            &[dim, ffn],
            0.1,
            0.0,
        );
        put(&mut map, &format!("{p}.ffn.2.bias"), &[dim], 0.03, 0.0);
    }
    map
}

/// The MLX group size the tier converter + the load-time quantizer both default to.
const GROUP: i32 = 64;

/// A **pre-quantized** (packed) tier map for the tiny geometry at `bits`, produced exactly the way the
/// rehost produces one: sanitize the native checkpoint, then pack with the reused Wan packer.
fn packed_map(cfg: &KreaRealtimeConfig, bits: i32) -> HashMap<String, Array> {
    let sanitized = sanitize_krea_realtime_transformer(native_random_map(cfg)).expect("sanitize");
    quantize_krea_realtime_transformer(sanitized, bits, GROUP).expect("pack tier")
}

/// A deterministic f32 latent chunk `[C, F, H, W]`.
fn det_latent(c: i32, f: i32, h: i32, w: i32, seed: u64) -> Array {
    let n = (c * f * h * w) as usize;
    let mut s = seed.wrapping_mul(0xD1B5_4A32_D192_ED03).wrapping_add(7);
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = (s >> 11) as f64 / (1u64 << 53) as f64;
        data.push((u as f32 - 0.5) * 2.0);
    }
    Array::from_slice(&data, &[c, f, h, w])
}

fn max_abs(a: &Array) -> f32 {
    mlx_rs::ops::max(mlx_rs::ops::abs(a).unwrap(), None)
        .unwrap()
        .item::<f32>()
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    max_abs(&mlx_rs::ops::subtract(a, b).unwrap())
}

/// One deterministic AR chunk forward → the denoised velocity latent `[C, F, H, W]`.
fn forward_once(causal: &CausalKreaTransformer, cfg: &KreaRealtimeConfig) -> Array {
    let t5 = det_latent(cfg.wan.text_len as i32, cfg.wan.text_dim as i32, 1, 1, 99)
        .reshape(&[cfg.wan.text_len as i32, cfg.wan.text_dim as i32])
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let ctx = causal.inner().embed_text(&t5).expect("embed text");
    let xkv = causal.prepare_cross_kv(&ctx).expect("cross kv");

    let c = cfg.wan.in_dim as i32;
    let fpb = cfg.ar.num_frames_per_block as i32;
    let chunk = det_latent(c, fpb, 4, 4, 11);
    let mut cache = causal.new_cache();
    causal
        .forward_chunk(&chunk, 500.0, &xkv, 0, &mut cache)
        .expect("forward chunk")
}

/// The dense bf16 tier's transformer.
fn dense_transformer(cfg: &KreaRealtimeConfig) -> CausalKreaTransformer {
    let dit = load_krea_realtime_transformer(native_random_map(cfg), cfg).expect("load dense DiT");
    CausalKreaTransformer::new(dit, cfg)
}

/// A **pre-quantized** tier's transformer, loaded straight off the packed map.
fn packed_transformer(cfg: &KreaRealtimeConfig, bits: i32) -> CausalKreaTransformer {
    let (dit, resolved) =
        load_krea_realtime_transformer_with_quant(packed_map(cfg, bits), cfg).expect("load packed");
    assert_eq!(
        resolved,
        Some(WanQuant {
            bits,
            group_size: GROUP
        }),
        "the loader must report the tier it read off the packed weights"
    );
    CausalKreaTransformer::new(dit, cfg)
}

/// The same dense snapshot, quantized **at load** (the `LoadSpec::quantize` path over a bf16 snapshot).
fn load_time_quantized_transformer(cfg: &KreaRealtimeConfig, bits: i32) -> CausalKreaTransformer {
    let mut dit = load_krea_realtime_transformer(native_random_map(cfg), cfg).expect("load dense");
    dit.quantize(bits, None).expect("load-time quantize");
    CausalKreaTransformer::new(dit, cfg)
}

/// Is the Linear at `path` packed?
fn is_packed(t: &mut CausalKreaTransformer, path: &[&str]) -> bool {
    t.adaptable_mut(path)
        .unwrap_or_else(|| panic!("`{}` is not an adaptable target", path.join(".")))
        .is_quantized()
}

// ── Gates ────────────────────────────────────────────────────────────────────────────────────────

/// (1) A packed snapshot really loads packed: every quantize-predicate Linear is quantized, and the
/// out-of-predicate ones stay dense. Discriminating in both directions — a silent dense fallback fails
/// the first half; over-packing the precision-sensitive embeddings/head fails the second (and would
/// desync from `WanTransformer::from_weights`, which loads those with `quant = None` unconditionally).
#[test]
fn packed_snapshot_quantizes_exactly_the_predicate_linears() {
    let cfg = tiny_cfg();
    for bits in [4, 8] {
        let mut t = packed_transformer(&cfg, bits);
        for block in 0..cfg.wan.num_layers {
            let b = block.to_string();
            for attn in ["self_attn", "cross_attn"] {
                for proj in ["q", "k", "v", "o"] {
                    assert!(
                        is_packed(&mut t, &["blocks", &b, attn, proj]),
                        "Q{bits}: blocks.{b}.{attn}.{proj} must load packed"
                    );
                }
            }
            // The reference-named FFN keys route to the converted `fc1`/`fc2` Linears.
            assert!(is_packed(&mut t, &["blocks", &b, "ffn", "0"]));
            assert!(is_packed(&mut t, &["blocks", &b, "ffn", "2"]));
        }
        // Not adaptable targets, hence not reachable through the host — but they ARE reachable via the
        // expected-tensor inventory, which is asserted dense in the unit tests. Here, assert the dense
        // tier is genuinely different: nothing is packed on bf16.
        let mut dense = dense_transformer(&cfg);
        assert!(
            !is_packed(&mut dense, &["blocks", "0", "self_attn", "q"]),
            "the bf16 tier must not be quantized"
        );
    }
}

/// (2) **The load-bearing equivalence.** A tier packed on disk builds a *bit-identical* model to the
/// same dense snapshot quantized at load — same MLX affine codes, same group scales, same Linears. If
/// the packed consume path read the wrong tensor, assumed the wrong group size, or fell back to dense,
/// this diverges immediately.
#[test]
fn pre_quantized_tier_is_bit_identical_to_load_time_quantization() {
    let cfg = tiny_cfg();
    for bits in [4, 8] {
        let from_disk = forward_once(&packed_transformer(&cfg, bits), &cfg);
        let at_load = forward_once(&load_time_quantized_transformer(&cfg, bits), &cfg);
        assert_eq!(
            max_abs_diff(&from_disk, &at_load),
            0.0,
            "Q{bits}: a pre-quantized snapshot must build the same model as load-time quantization"
        );
    }
}

/// (3) The tier genuinely changes the numerics: Q4 ≠ bf16, Q8 ≠ bf16, and Q4 ≠ Q8 — with Q8 measurably
/// closer to bf16 than Q4 is (more bits ⇒ less quantization error). A "packed" path that silently
/// served the dense weights would pass every shape assertion but fail all three here.
#[test]
fn each_tier_changes_the_forward_and_q8_is_closer_to_bf16_than_q4() {
    let cfg = tiny_cfg();
    let bf16 = forward_once(&dense_transformer(&cfg), &cfg);
    let q8 = forward_once(&packed_transformer(&cfg, 8), &cfg);
    let q4 = forward_once(&packed_transformer(&cfg, 4), &cfg);

    let scale = max_abs(&bf16);
    assert!(scale > 0.0, "the baseline forward must be non-trivial");

    let d8 = max_abs_diff(&bf16, &q8);
    let d4 = max_abs_diff(&bf16, &q4);
    assert!(d8 > 0.0, "Q8 must differ from bf16 (it is quantized)");
    assert!(d4 > 0.0, "Q4 must differ from bf16 (it is quantized)");
    assert!(
        max_abs_diff(&q4, &q8) > 0.0,
        "Q4 and Q8 must not be the same model"
    );
    assert!(
        d8 < d4,
        "Q8 must be closer to bf16 than Q4 (Q8 err {d8}, Q4 err {d4})"
    );
}

/// (4) The bf16 cast in the converter must leave the u32 packed codes alone (sc-15203): a packed tier
/// survives the load path's re-sanitize with its codes intact and its tier still readable. Casting the
/// codes would reinterpret 8 (Q4) packed weights as one float — a silent, total corruption.
#[test]
fn packed_codes_survive_the_load_paths_re_sanitize() {
    let cfg = tiny_cfg();
    let packed = packed_map(&cfg, 4);
    let key = "blocks.0.self_attn.q.weight";
    let before: Vec<u32> = packed[key].as_slice::<u32>().to_vec();
    assert_eq!(packed[key].dtype(), Dtype::Uint32);

    // The pipeline re-sanitizes an already-converted `dit.safetensors` on every load.
    let again = sanitize_krea_realtime_transformer(packed).expect("re-sanitize a packed tier");
    assert_eq!(
        again[key].dtype(),
        Dtype::Uint32,
        "the u32 codes must not be cast to bf16"
    );
    assert_eq!(again[key].as_slice::<u32>(), before.as_slice());
    // And the tier is still recoverable from the shapes afterwards.
    assert_eq!(
        resolve_snapshot_quant(&again, &cfg.wan).unwrap(),
        Some(WanQuant {
            bits: 4,
            group_size: GROUP
        })
    );
}

/// (5) `verify_transformer_tensors` is exact on a packed tier: a missing `.scales` and a stray extra
/// tensor both fail. Without the quant-aware inventory this check would reject every valid Q4/Q8
/// snapshot (`.scales`/`.biases` as "extra", u32 codes as "wrong shape"), so it has to be exercised.
#[test]
fn verification_is_exact_on_a_packed_tier() {
    let cfg = tiny_cfg();
    let mut wan = cfg.wan.clone();
    wan.quantization = Some(WanQuant {
        bits: 8,
        group_size: GROUP,
    });

    let good = packed_map(&cfg, 8);
    verify_transformer_tensors(&good, &wan).expect("a well-formed Q8 tier must verify");
    // …and the inventory it verified against is the packed one (10 predicate Linears × 2 extra tensors
    // per block over the dense count).
    assert_eq!(
        expected_transformer_tensors(&wan).len() - expected_transformer_tensors(&cfg.wan).len(),
        cfg.wan.num_layers * 10 * 2
    );

    let mut missing = good.clone();
    missing.remove("blocks.0.ffn.fc1.scales");
    let err = verify_transformer_tensors(&missing, &wan)
        .expect_err("a missing scales tensor must be caught");
    assert!(err.to_string().contains("ffn.fc1.scales"), "got: {err}");

    let mut extra = good.clone();
    extra.insert(
        "blocks.0.self_attn.q.stray".into(),
        Array::zeros::<f32>(&[1]).unwrap(),
    );
    let err =
        verify_transformer_tensors(&extra, &wan).expect_err("a stray extra tensor must be caught");
    assert!(err.to_string().contains("stray"), "got: {err}");

    // A packed map verified against the *dense* inventory must fail — the discriminating half proving
    // the quant-awareness is load-bearing, not decorative.
    assert!(
        verify_transformer_tensors(&good, &cfg.wan).is_err(),
        "a packed tier must not verify against the dense inventory"
    );
}

/// Write a PEFT LoRA file (`diffusion_model.‹stem›.lora_A/B.weight`) over the tiny geometry's
/// attention + FFN targets — the FFN spelled in the reference `ffn.0`/`ffn.2` naming a real
/// Wan-family file carries, so the host's key normalization is exercised too.
///
/// `dt` is the **factor dtype**, and it matters (see
/// [`scale_zero_is_a_bit_exact_no_op_on_every_tier`]): the residual is added to the base's output at
/// the promoted dtype, so f32 factors widen the Linear's bf16 output to f32. That widening is
/// value-preserving at the Linear itself but changes downstream rounding, on a **dense** base exactly
/// as much as on a packed one.
fn write_lora(name: &str, seed: u64, mag: f32, dt: Dtype) -> PathBuf {
    const RANK: i32 = 4;
    let stems: [(&str, i32, i32); 5] = [
        ("blocks.0.self_attn.q", 64, 64),
        ("blocks.0.self_attn.o", 64, 64),
        ("blocks.1.cross_attn.v", 64, 64),
        ("blocks.1.ffn.0", 128, 64),
        ("blocks.1.ffn.2", 64, 128),
    ];
    let mut entries: Vec<(String, Array)> = Vec::new();
    for (i, (stem, out, inp)) in stems.iter().enumerate() {
        let a = det_fill(&[RANK, *inp], seed + i as u64 * 7, mag, 0.0, dt);
        let b = det_fill(&[*out, RANK], seed + 100 + i as u64 * 7, mag, 0.0, dt);
        entries.push((format!("diffusion_model.{stem}.lora_A.weight"), a));
        entries.push((format!("diffusion_model.{stem}.lora_B.weight"), b));
    }
    let dir = std::env::temp_dir().join("mlx_gen_krea_quant_tier_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    let refs: Vec<(&str, &Array)> = entries.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(refs, None, &path).unwrap();
    path
}

/// The number of targets [`write_lora`] writes — every one must resolve on every tier.
const LORA_TARGETS: usize = 5;

/// Build the transformer for a tier: `None` = dense bf16, `Some(bits)` = the pre-quantized tier.
fn tier_transformer(cfg: &KreaRealtimeConfig, bits: Option<i32>) -> CausalKreaTransformer {
    match bits {
        Some(b) => packed_transformer(cfg, b),
        None => dense_transformer(cfg),
    }
}

/// Every tier, named for assertion messages.
const TIERS: [(&str, Option<i32>); 3] = [("bf16", None), ("q8", Some(8)), ("q4", Some(4))];

/// (6) **LoRA on a packed base** (the epic-10043 / sc-10578 additive-on-packed property, for Krea): a
/// Wan-family LoRA installs onto a Q4 **and** a Q8 base, every target resolves, it measurably shifts
/// the packed forward, and the base is **still packed** afterwards — never dequantized, which would
/// blow the whole point of the tier (a 14B Q4 base dequantized back to bf16 is ~28 GB resident).
#[test]
fn lora_installs_additively_over_a_packed_base_without_dequantizing_it() {
    let cfg = tiny_cfg();
    let lora = write_lora("packed_base.safetensors", 5, 0.5, Dtype::Float32);

    for bits in [4, 8] {
        let baseline = forward_once(&packed_transformer(&cfg, bits), &cfg);

        let mut with_lora = packed_transformer(&cfg, bits);
        let report = apply_adapters_strict(
            &mut with_lora,
            &[AdapterSpec::new(lora.clone(), 1.0, AdapterKind::Lora)],
            MODEL_ID,
        )
        .expect("a Wan-family LoRA must install over a packed base");
        assert_eq!(
            report.applied, LORA_TARGETS,
            "Q{bits}: every LoRA target must resolve on the packed base"
        );
        assert!(report.unmatched_paths.is_empty());

        // The base is STILL packed after the install — the residual rides on top, no dequantization.
        assert!(
            is_packed(&mut with_lora, &["blocks", "0", "self_attn", "q"]),
            "Q{bits}: the attention base must stay packed after a LoRA install"
        );
        assert!(
            is_packed(&mut with_lora, &["blocks", "1", "ffn", "0"]),
            "Q{bits}: the FFN base must stay packed after a LoRA install"
        );

        assert!(
            max_abs_diff(&baseline, &forward_once(&with_lora, &cfg)) > 1e-4,
            "Q{bits}: the LoRA must measurably change the packed forward"
        );
    }
}

/// (7) A **scale-0** adapter is a bit-exact no-op on **every** tier — bf16, Q8 and Q4 alike. This is
/// what excludes "the forward changed by accident" from gate (6), and it proves the install never
/// mutates the base (which on a packed tier would mean an unpack/repack round trip).
///
/// The factors are **bf16** here deliberately. The shared loader adds the residual at the factors'
/// promoted dtype, so an *f32*-factor adapter widens the host Linear's bf16 output to f32 — a
/// value-preserving widening at the Linear, but one that changes downstream rounding (`rms_norm`,
/// SDPA, the FFN chain) and so is not bit-exact end-to-end even at scale 0. Measured, that artifact is
/// ~1.9e-4 on a dense base and ~2.9e-4 on Q4: present on **both**, i.e. a property of the shared
/// adapter path, not of the packed tier. Pinning the bf16 case keeps this gate about the tier.
#[test]
fn scale_zero_is_a_bit_exact_no_op_on_every_tier() {
    let cfg = tiny_cfg();
    let lora = write_lora("scale_zero.safetensors", 5, 0.5, Dtype::Bfloat16);

    for (name, bits) in TIERS {
        let baseline = forward_once(&tier_transformer(&cfg, bits), &cfg);
        let mut zero = tier_transformer(&cfg, bits);
        apply_adapters_strict(
            &mut zero,
            &[AdapterSpec::new(lora.clone(), 0.0, AdapterKind::Lora)],
            MODEL_ID,
        )
        .unwrap_or_else(|e| panic!("{name}: scale-0 install failed: {e}"));
        assert_eq!(
            max_abs_diff(&baseline, &forward_once(&zero, &cfg)),
            0.0,
            "{name}: a scale-0 LoRA must be a bit-exact no-op"
        );
        // …and a non-zero scale of the SAME file does move it, so the no-op is not vacuous.
        let mut on = tier_transformer(&cfg, bits);
        apply_adapters_strict(
            &mut on,
            &[AdapterSpec::new(lora.clone(), 1.0, AdapterKind::Lora)],
            MODEL_ID,
        )
        .unwrap();
        assert!(
            max_abs_diff(&baseline, &forward_once(&on, &cfg)) > 1e-4,
            "{name}: the same file at scale 1 must move the forward"
        );
    }
}

/// (8) The same LoRA file resolves on **every** tier — the capability the descriptor advertises
/// (`supports_lora` alongside `supported_quants = [Q4, Q8]`) has to hold at Q4/Q8, not only bf16.
#[test]
fn the_same_lora_installs_on_every_tier() {
    let cfg = tiny_cfg();
    let lora = write_lora("every_tier.safetensors", 77, 0.5, Dtype::Float32);
    let spec = AdapterSpec::new(lora, 1.0, AdapterKind::Lora);

    for (name, bits) in TIERS {
        let baseline = forward_once(&tier_transformer(&cfg, bits), &cfg);
        let mut host = tier_transformer(&cfg, bits);
        let report = apply_adapters_strict(&mut host, std::slice::from_ref(&spec), MODEL_ID)
            .unwrap_or_else(|e| panic!("{name}: LoRA install failed: {e}"));
        assert_eq!(
            report.applied, LORA_TARGETS,
            "{name}: all targets must resolve"
        );
        assert!(
            max_abs_diff(&baseline, &forward_once(&host, &cfg)) > 1e-4,
            "{name}: the LoRA must move this tier's forward"
        );
    }
}

/// (9) The load-time quant request is reconciled against the tier actually on disk. A Q4 request over a
/// packed Q8 snapshot is a hard error — the case that would otherwise be a *silent* wrong-tier serve,
/// because `AdaptableLinear::quantize` no-ops over packed weights.
#[test]
fn a_conflicting_load_time_request_is_rejected_against_the_real_tier() {
    let cfg = tiny_cfg();
    let (_, packed) = load_krea_realtime_transformer_with_quant(packed_map(&cfg, 8), &cfg)
        .expect("load a Q8 tier");

    let err = resolve_load_time_quant(MODEL_ID, packed, Some(Quant::Q4))
        .expect_err("Q4 over a packed Q8 snapshot must fail");
    assert!(err.to_string().contains("silently serve Q8"), "got: {err}");

    // The matching request, and no request at all, are both no-ops (already packed).
    assert_eq!(
        resolve_load_time_quant(MODEL_ID, packed, Some(Quant::Q8)).unwrap(),
        None
    );
    assert_eq!(
        resolve_load_time_quant(MODEL_ID, packed, None).unwrap(),
        None
    );

    // Over a dense snapshot the request is honored.
    let (_, dense) = load_krea_realtime_transformer_with_quant(
        sanitize_krea_realtime_transformer(native_random_map(&cfg)).unwrap(),
        &cfg,
    )
    .expect("load a dense tier");
    assert_eq!(dense, None);
    assert_eq!(
        resolve_load_time_quant(MODEL_ID, dense, Some(Quant::Q4)).unwrap(),
        Some(Quant::Q4)
    );
}

/// Write the tiny geometry's **native** (single-file, `model.`-prefixed) checkpoint — the layout the
/// real 14B ships in — under `base`, and return its path.
fn write_native_checkpoint(cfg: &KreaRealtimeConfig, base: &std::path::Path) -> PathBuf {
    std::fs::create_dir_all(base).unwrap();
    let native = base.join("native.safetensors");
    let map = native_random_map(cfg);
    let prefixed: Vec<(String, Array)> = map
        .iter()
        .map(|(k, v)| (format!("model.{k}"), v.clone()))
        .collect();
    let refs: Vec<(&str, &Array)> = prefixed.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(refs, None, &native).unwrap();
    native
}

/// (10) **The rehost emitter and the load path agree, end to end.** `convert_krea_realtime_tier` writes
/// a complete tier directory (`dit.safetensors` + `config.json`) from a *native* checkpoint; the
/// pipeline's own config reader then recovers the tier from that `config.json`, and the DiT loads
/// packed at exactly that width. This is the seam S2b's three-tier rehost drives, so a drift between
/// what we write and what we read would otherwise only surface on a 28 GB real-weight run.
#[test]
fn tier_converter_writes_a_snapshot_the_load_path_reads_back() {
    let cfg = tiny_cfg();
    let base = std::env::temp_dir().join(format!("krea_tier_convert_{}", std::process::id()));
    let native = write_native_checkpoint(&cfg, &base);

    for (tier, quantize) in [
        ("bf16", None),
        ("q8", Some((8, GROUP))),
        ("q4", Some((4, GROUP))),
    ] {
        let out = base.join(tier);
        // The tiny-geometry seam of the same emitter (`convert_krea_realtime_tier` is this with the
        // shipped 14B preset), so what is verified and what is declared are one config.
        convert_krea_realtime_tier_with_config(&native, &out, quantize, &cfg)
            .unwrap_or_else(|e| panic!("{tier}: convert failed: {e}"));
        assert!(out.join(DIT_FILE).is_file(), "{tier}: dit must be written");
        assert!(
            out.join("config.json").is_file(),
            "{tier}: config must be written"
        );

        // The emitted config declares exactly the tier that was packed (and nothing on bf16), over the
        // geometry it was converted against.
        let read_back = KreaRealtimeConfig::from_model_dir(&out).expect("read the emitted config");
        let expected = quantize.map(|(bits, group_size)| WanQuant { bits, group_size });
        assert_eq!(
            read_back.wan.quantization, expected,
            "{tier}: the emitted config.json must declare the tier that was written"
        );
        assert_eq!(
            (
                read_back.wan.dim,
                read_back.wan.ffn_dim,
                read_back.wan.num_layers
            ),
            (cfg.wan.dim, cfg.wan.ffn_dim, cfg.wan.num_layers),
            "{tier}: the emitted config.json must declare the converted geometry"
        );

        // …and the weights on disk resolve to the same tier, verified against the config that shipped
        // with them.
        let w = mlx_gen::weights::Weights::from_file(out.join(DIT_FILE)).unwrap();
        let raw: HashMap<String, Array> = w
            .keys()
            .map(|k| (k.to_string(), w.get(k).unwrap().clone()))
            .collect();
        let (dit, resolved) = load_krea_realtime_transformer_with_quant(raw, &read_back)
            .unwrap_or_else(|e| panic!("{tier}: load failed: {e}"));
        assert_eq!(
            resolved, expected,
            "{tier}: the written weights must resolve to the tier the config declares"
        );
        // A packed tier really loads packed; bf16 really loads dense.
        let mut host = CausalKreaTransformer::new(dit, &read_back);
        assert_eq!(
            is_packed(&mut host, &["blocks", "0", "self_attn", "q"]),
            quantize.is_some(),
            "{tier}: packed-ness must match the tier"
        );
    }

    std::fs::remove_dir_all(&base).ok();
}

/// (11) A source checkpoint whose geometry is **not** the config the snapshot would ship with is
/// rejected **at conversion time**, on every tier — before `dit.safetensors` is written. Without the
/// pre-write verification the emitter happily pairs any tensor set with the hard-coded 14B preset's
/// `config.json`, and the mismatch first surfaces at *load* — on the real rehost that is after a
/// 28.58 GB conversion and an HF upload.
///
/// Discriminating: the identical source converts fine against its own geometry (gate 10), and the
/// failure is asserted to leave **no output files** behind, so "it errored somewhere later" would not
/// pass.
#[test]
fn a_geometry_mismatched_source_is_rejected_before_anything_is_written() {
    let cfg = tiny_cfg();
    let base = std::env::temp_dir().join(format!("krea_tier_mismatch_{}", std::process::id()));
    let native = write_native_checkpoint(&cfg, &base);

    for (tier, quantize) in [
        ("bf16", None),
        ("q8", Some((8, GROUP))),
        ("q4", Some((4, GROUP))),
    ] {
        let out = base.join(format!("preset_{tier}"));
        // The preset entry point declares the shipped 14B geometry — which this tiny source is not.
        let err = convert_krea_realtime_tier(&native, &out, quantize)
            .expect_err("a tiny source under the 14B preset must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("does not match the config it would ship with"),
            "{tier}: got: {msg}"
        );
        assert!(
            !out.join(DIT_FILE).exists() && !out.join("config.json").exists(),
            "{tier}: nothing may be written when the geometry does not match"
        );

        // The same rejection when the geometry differs by ONE field (a near-miss the tensor-count
        // check alone would catch, but which the shape check must also flag).
        let mut wider = cfg.clone();
        wider.wan.ffn_dim = cfg.wan.ffn_dim * 2;
        let out = base.join(format!("near_miss_{tier}"));
        let err = convert_krea_realtime_tier_with_config(&native, &out, quantize, &wider)
            .expect_err("a one-field geometry mismatch must be rejected too");
        assert!(
            err.to_string().contains("wrong-shape"),
            "{tier}: got: {err}"
        );
    }

    std::fs::remove_dir_all(&base).ok();
}

/// (12) [`PACKED_LINEARS_PER_BLOCK`] is the **real** predicate surface, not a stale description of it:
/// the set of per-block Linears the reused Wan packer actually packs (every `.scales` it emits) is
/// exactly this list. The constant drives `expected_transformer_tensors`, so if the packer's predicate
/// ever moves (a fused `qkv`, a packed `time_projection`) this goes red here instead of every Q4/Q8
/// snapshot failing verification at load.
#[test]
fn the_packed_linear_list_is_exactly_what_the_packer_packs() {
    let cfg = tiny_cfg();
    let packed = packed_map(&cfg, 4);

    let mut packed_suffixes: Vec<String> = packed
        .keys()
        .filter_map(|k| k.strip_suffix(".scales"))
        .filter_map(|stem| stem.strip_prefix("blocks.0."))
        .map(str::to_string)
        .collect();
    packed_suffixes.sort();
    let mut declared: Vec<String> = PACKED_LINEARS_PER_BLOCK
        .iter()
        .map(|s| s.to_string())
        .collect();
    declared.sort();
    assert_eq!(
        packed_suffixes, declared,
        "PACKED_LINEARS_PER_BLOCK must be exactly the per-block Linears the Wan packer packs"
    );

    // Non-vacuous: the packer did pack a real surface, and it packed the SAME surface in every block.
    assert_eq!(declared.len(), 10);
    let per_block: usize = packed
        .keys()
        .filter(|k| k.ends_with(".scales") && k.starts_with("blocks.1."))
        .count();
    assert_eq!(per_block, declared.len());
    // …and nothing outside `blocks.` packed at all.
    assert!(
        packed
            .keys()
            .filter(|k| k.ends_with(".scales"))
            .all(|k| k.starts_with("blocks.")),
        "only per-block Linears are inside the quantize predicate"
    );
}
