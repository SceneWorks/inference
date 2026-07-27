//! Krea Realtime 14B **real-weight** conversion + tier-emission gate (sc-8435, S2b).
//!
//! The crate's other tests prove the converter / load path over synthesized fixtures at a tiny
//! geometry. This is the gated counterpart that runs them against the **real 28.58 GB
//! `krea/krea-realtime-video` checkpoint** and emits the tier snapshots published to
//! `SceneWorks/krea-realtime-14b-mlx` — the rehost driver, mirroring
//! `mlx-gen-scail2/tests/quantize_snapshot.rs`.
//!
//! **This driver emits and validates; it does not publish.** Uploading is a credentialed network
//! write, which does not belong inside a `cargo test` binary, so these tests stop at a verified local
//! snapshot and the operator publishes it with the `hf` CLI (below). Nothing here reaches the network.
//!
//! `#[ignore]` (needs the real checkpoint). Run on macOS:
//!
//! ```text
//! # 1. Inventory gate — settles the two S2 gated notes. Shapes only (MLX loads are lazy), so this
//! #    reads the safetensors header and never materializes the 28.58 GB of weights.
//! cargo test -p mlx-gen-krea-realtime --test rehost_real_weights -- --ignored --nocapture \
//!   real_checkpoint_matches_the_audited_inventory
//!
//! # 2. Emit ONE tier, then load it back. `KREA_REHOST_TIER` = bf16 | q8 | q4 (default q4).
//! KREA_REHOST_TIER=q4 cargo test -p mlx-gen-krea-realtime --test rehost_real_weights \
//!   -- --ignored --nocapture emit_tier_and_load_it_back
//!
//! # 2b. …or as a sharded `transformer/` dir (how the dense bf16 tier ships).
//! KREA_REHOST_TIER=bf16 cargo test -p mlx-gen-krea-realtime --test rehost_real_weights \
//!   -- --ignored --nocapture emit_tier_sharded_and_validate_every_shard
//! ```
//!
//! Publishing the validated snapshot (operator step, outside the test binary) — the whole tier
//! directory, so `config.json` and the stock-Wan companions go up with the DiT and the remote tier is
//! never left partial:
//!
//! ```text
//! hf upload SceneWorks/krea-realtime-14b-mlx ~/.cache/krea-realtime-mlx-q4 q4 --repo-type model
//! ```
//!
//! On a host too small to hold a whole tier, publish shard-by-shard instead: that is what the
//! `on_shard` callback on `convert_krea_realtime_tier_sharded` is for — upload the shard and delete it
//! inside the callback, then upload `config.json` + the companions once at the end.
//!
//! Env:
//!   * `KREA_REALTIME_CHECKPOINT` (**required**) — path to the native checkpoint: the single-file
//!     `krea-realtime-video-14b.safetensors` **or** the sharded `transformer/` dir. There is
//!     deliberately no default: inference never resolves a model out of an HF cache itself, every
//!     component is handed in as a path by the caller.
//!   * `KREA_REHOST_TIER` — `bf16` / `q8` / `q4` (default `q4`).
//!   * `KREA_REHOST_OUT` — tier snapshot destination (default `~/.cache/krea-realtime-mlx-<tier>`).
//!   * `KREA_REHOST_SHARD_GIB` — shard budget for the sharded emit (default 4 GiB).
//!   * `KREA_REHOST_COMPANIONS` — staged stock-Wan components (default
//!     `~/.cache/krea-realtime-mlx-companions`).
//!
//! **Disk:** the tiers are ~8.4 GB (Q4) / ~15.4 GB (Q8) / ~28.6 GB (bf16) and every test here keeps
//! the emitted tier on disk so it can be read back, so run **one tier at a time** and remove the
//! previous tier's directory first.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen_krea_realtime::{
    convert_krea_realtime_tier, convert_krea_realtime_tier_sharded,
    load_krea_realtime_transformer_with_quant, sanitize_krea_realtime_transformer,
    verify_transformer_tensors, KreaRealtimeConfig, WanQuant, DEFAULT_SHARD_BYTES, DIT_FILE,
};
use mlx_rs::ops::dequantize;
use mlx_rs::{Array, Dtype};

/// The S1 audit's transformer tensor count — **excluding** the `freqs` RoPE buffer (15 non-block +
/// 40 blocks × 27 = 1095). Asserted against the real checkpoint below.
const AUDIT_TENSOR_COUNT: usize = 1095;

/// The S1 audit's total parameter count (`safetensors.total`).
const AUDIT_PARAM_COUNT: usize = 14_288_491_584;

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

/// The native checkpoint source — single file or sharded dir; both layouts are handled by the
/// converter (`normalize_krea_keys` strips the single-file layout's `model.` prefix).
///
/// Required, with no fallback: inference does not resolve models out of a cache of its own, so the
/// caller passes the path in (the same boundary the engine's own load paths hold to).
fn checkpoint() -> PathBuf {
    let raw = std::env::var("KREA_REALTIME_CHECKPOINT").unwrap_or_else(|_| {
        panic!(
            "set KREA_REALTIME_CHECKPOINT to the native krea/krea-realtime-video checkpoint \
             (the single-file `krea-realtime-video-14b.safetensors` or the sharded `transformer/` dir)"
        )
    });
    let path = PathBuf::from(raw);
    assert!(
        path.exists(),
        "KREA_REALTIME_CHECKPOINT does not exist: {}",
        path.display()
    );
    path
}

/// `(label, quantize)` for the requested tier: `None` = dense bf16, else `(bits, group_size)`. Group
/// size 64 on both packed tiers — the Wan packer's shipped grouping.
fn tier() -> (String, Option<(i32, i32)>) {
    let label = std::env::var("KREA_REHOST_TIER").unwrap_or_else(|_| "q4".to_string());
    let quant = match label.as_str() {
        "bf16" => None,
        "q8" => Some((8, 64)),
        "q4" => Some((4, 64)),
        other => panic!("KREA_REHOST_TIER must be bf16 | q8 | q4, got {other:?}"),
    };
    (label, quant)
}

fn out_dir(label: &str) -> PathBuf {
    std::env::var("KREA_REHOST_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home().join(format!(".cache/krea-realtime-mlx-{label}")))
}

/// Read the native checkpoint into an owned map. MLX safetensors loads are **lazy**, so this reads the
/// header and holds unmaterialized handles — the 28.58 GB is only paged in if something forces it.
fn read_native(src: &Path) -> HashMap<String, Array> {
    let w = if src.is_dir() {
        Weights::from_dir(src)
    } else {
        Weights::from_file(src)
    }
    .unwrap_or_else(|e| panic!("open native checkpoint {}: {e}", src.display()));
    w.keys()
        .map(|k| {
            (
                k.to_string(),
                w.require(k).expect("key from keys()").clone(),
            )
        })
        .collect()
}

/// **Settles both sc-8435 gated notes against the real checkpoint**, from safetensors metadata only:
///
/// 1. `verify_transformer_tensors` rejects any tensor outside the audited set as "extra" — stricter
///    than `WanTransformer::from_weights`, which tolerates unused keys. If the real checkpoint carried
///    a benign extra buffer, the strict check would reject a perfectly good conversion. It does not:
///    the sanitized map matches the `wan21_t2v_14b` inventory **exactly**.
/// 2. The internal expected count of 1095 assumes the audit figure **excludes** the `freqs` RoPE
///    buffer. Confirmed here directly: the native checkpoint ships no `freqs` tensor at all (it is a
///    non-persistent buffer in the reference), and the raw count is already 1095 — so the sanitizer's
///    `freqs` drop is a no-op on this checkpoint rather than the thing that gets it to 1095.
#[test]
#[ignore = "real 28.58 GB checkpoint; run with --ignored on macOS (see module doc)"]
fn real_checkpoint_matches_the_audited_inventory() {
    let src = checkpoint();
    let raw = read_native(&src);

    // (2) The raw checkpoint is already at the audit count and carries no `freqs`.
    let freqs: Vec<&String> = raw.keys().filter(|k| k.contains("freqs")).collect();
    assert!(
        freqs.is_empty(),
        "the audit's 1095 must EXCLUDE `freqs`, but the checkpoint ships {freqs:?}"
    );
    assert_eq!(
        raw.len(),
        AUDIT_TENSOR_COUNT,
        "native tensor count must match the S1 audit"
    );
    let params: usize = raw.values().map(|a| a.size()).sum();
    assert_eq!(
        params, AUDIT_PARAM_COUNT,
        "total parameter count (S1 audit)"
    );
    assert!(
        raw.values().all(|a| a.dtype() == Dtype::Float16),
        "the shipped checkpoint is F16 throughout"
    );

    // (1) Sanitize, then assert the STRICT verify passes — no missing, no extra, no wrong shape.
    let cfg = KreaRealtimeConfig::krea_realtime_14b();
    let sanitized = sanitize_krea_realtime_transformer(raw).expect("sanitize real checkpoint");
    assert_eq!(sanitized.len(), AUDIT_TENSOR_COUNT);
    assert!(
        sanitized.values().all(|a| a.dtype() == Dtype::Bfloat16),
        "the converter casts the F16 checkpoint to bf16 (the reference DiT dtype)"
    );
    verify_transformer_tensors(&sanitized, &cfg.wan)
        .expect("the real checkpoint must match the audited inventory with no extra buffers");

    println!(
        "real checkpoint OK: {} tensors, {} params, F16 -> bf16, no `freqs`, strict verify clean ({})",
        sanitized.len(),
        params,
        src.display()
    );
}

/// Emit ONE tier snapshot from the real checkpoint through the shipped emitter
/// (`convert_krea_realtime_tier`, which runs `verify_transformer_tensors` before writing), then prove
/// the written snapshot loads back through the crate's own load path at the tier it claims. This is
/// the artifact published to `SceneWorks/krea-realtime-14b-mlx/<tier>/`.
#[test]
#[ignore = "real 28.58 GB checkpoint + a multi-GB write; run with --ignored on macOS (see module doc)"]
fn emit_tier_and_load_it_back() {
    let src = checkpoint();
    let (label, quant) = tier();
    let dst = out_dir(&label);

    let t0 = std::time::Instant::now();
    convert_krea_realtime_tier(&src, &dst, quant)
        .unwrap_or_else(|e| panic!("emit {label} tier from {}: {e}", src.display()));
    let emit = t0.elapsed();

    let dit = dst.join(DIT_FILE);
    let bytes = std::fs::metadata(&dit).expect("tier dit.safetensors").len();
    assert!(
        dst.join("config.json").is_file(),
        "tier config.json written"
    );

    // Load the WRITTEN snapshot back — not the in-memory map — through the production load path.
    let t1 = std::time::Instant::now();
    let raw = read_native(&dit);
    let cfg = KreaRealtimeConfig::krea_realtime_14b();
    let (_transformer, resolved) = load_krea_realtime_transformer_with_quant(raw, &cfg)
        .unwrap_or_else(|e| panic!("load back the emitted {label} tier: {e}"));
    let load = t1.elapsed();

    // The tier the snapshot resolves to must be the tier we asked for — a bf16 emit that silently
    // resolved packed (or vice versa) would ship the wrong artifact under the right name.
    let want = quant.map(|(bits, group_size)| WanQuant { bits, group_size });
    match (want, resolved) {
        (None, None) => {}
        (Some(w), Some(g)) => assert_eq!(
            (w.bits, w.group_size),
            (g.bits, g.group_size),
            "{label}: emitted tier must resolve to the requested quantization"
        ),
        (w, g) => panic!("{label}: requested {w:?} but the snapshot resolved to {g:?}"),
    }

    println!(
        "tier {label}: {} ({:.2} GB) emitted in {:.1?}, loaded back in {:.1?} -> {:?}",
        dit.display(),
        bytes as f64 / 1e9,
        emit,
        load,
        resolved,
    );

    check_tier_reconstructs_the_source(&src, &dit, &label, quant);
}

/// Emit a tier as a **sharded** `transformer/` directory — the layout the dense bf16 tier (~28.6 GB)
/// ships in, and the one a conversion host that cannot hold the whole tier needs.
///
/// This **emits and validates only; it does not publish.** Publishing is a credentialed network write
/// and does not belong inside a `cargo test` binary, so the driver stops at a verified local snapshot
/// and the operator uploads it with the `hf` CLI (see the module doc for the exact commands). The
/// disk-bounded publish flow is still fully available — it is the `on_shard` callback on
/// [`convert_krea_realtime_tier_sharded`], which the caller can use to upload and delete each shard —
/// it is simply the operator's callback, not this test's.
///
/// Validation runs at two levels, and **unconditionally**: every shard is read back off disk the
/// moment it is written and compared against the source (so the split + `save_map` path is checked
/// per shard, not just the pre-write in-memory inventory), and the finished directory is then loaded
/// through the production path and content-checked as a whole.
///
/// `KREA_REHOST_SHARD_GIB` overrides the 4 GiB shard budget.
#[test]
#[ignore = "real 28.58 GB checkpoint; run with --ignored on macOS (see module doc)"]
fn emit_tier_sharded_and_validate_every_shard() {
    let src = checkpoint();
    let (label, quant) = tier();
    let dst = out_dir(&label);
    let budget = std::env::var("KREA_REHOST_SHARD_GIB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(|g| g * 1024 * 1024 * 1024)
        .unwrap_or(DEFAULT_SHARD_BYTES);

    let cfg = KreaRealtimeConfig::krea_realtime_14b();
    // The sanitized source, to compare each shard against as it lands.
    let source = sanitize_krea_realtime_transformer(read_native(&src)).expect("sanitize source");

    let t0 = std::time::Instant::now();
    let mut checked = 0usize;
    let shards =
        convert_krea_realtime_tier_sharded(&src, &dst, quant, &cfg, budget, &mut |shard| {
            let bytes = std::fs::metadata(shard).expect("shard metadata").len();
            let compared = check_shard_against_source(shard, &source, &label);
            checked += compared;
            println!(
                "  shard {} ({:.2} GB): {compared} tensor(s) bit-exact vs. source",
                shard.file_name().unwrap().to_string_lossy(),
                bytes as f64 / 1e9,
            );
            Ok(())
        })
        .unwrap_or_else(|e| panic!("sharded emit of the {label} tier: {e}"));

    assert!(
        checked > 0,
        "{label}: no shard tensor was comparable against the source — the per-shard check is inert"
    );
    println!(
        "tier {label}: {} shard(s) emitted in {:.1?}, {checked} tensor(s) spot-checked -> {}",
        shards.len(),
        t0.elapsed(),
        dst.display()
    );

    // Hold the sharded tier to the SAME bar as the single-file one: it must load back through the
    // production path at the tier it claims, and reconstruct the source weights.
    let dir = dst.join(mlx_gen_krea_realtime::TRANSFORMER_DIR);
    let raw = read_native(&dir);
    assert_eq!(
        raw.len(),
        AUDIT_TENSOR_COUNT + if quant.is_some() { 800 } else { 0 },
        "{label}: the sharded set must be complete (and disjoint — from_dir rejects duplicates)"
    );
    let (_t, resolved) = load_krea_realtime_transformer_with_quant(raw, &cfg)
        .unwrap_or_else(|e| panic!("load back the sharded {label} tier: {e}"));
    let want = quant.map(|(bits, group_size)| WanQuant { bits, group_size });
    assert_eq!(
        resolved.map(|q| (q.bits, q.group_size)),
        want.map(|q| (q.bits, q.group_size)),
        "{label}: the sharded tier must resolve to the requested quantization"
    );
    check_tier_reconstructs_the_source(&src, &dir, &label, quant);
}

/// Read `shard` back off disk and bit-exact-compare its tensors against the sanitized `source`,
/// returning how many were compared.
///
/// Only tensors that are **pass-through** at this tier are comparable — on a packed tier the predicate
/// Linears hold u32 codes that by construction differ from the source's bf16, so the rule is "same key,
/// same dtype, same shape ⇒ must be identical". A handful per shard is enough: this exists to catch a
/// bad split or a bad write in *this* shard, and the whole-tier content check runs afterwards.
///
/// Every key must still be *accounted for*: it is either a source key, or a `.scales`/`.biases`
/// companion that quantization created for a source `{base}.weight`. Anything else means the shard
/// holds a tensor the source never had.
fn check_shard_against_source(shard: &Path, source: &HashMap<String, Array>, label: &str) -> usize {
    let w = Weights::from_file(shard).expect("read the shard back");
    let mut names: Vec<&str> = w.keys().collect();
    names.sort();

    let mut compared = 0;
    for name in names {
        let got = w.require(name).expect("key from keys()");
        let Some(want) = source.get(name) else {
            // A packed companion is legitimate iff its base weight came from the source.
            let base = name
                .strip_suffix(".scales")
                .or_else(|| name.strip_suffix(".biases"));
            let ok = base.is_some_and(|b| source.contains_key(&format!("{b}.weight")));
            assert!(
                ok,
                "{label}: shard {shard:?} holds `{name}`, which is neither a source tensor nor a \
                 quantization companion of one"
            );
            continue;
        };
        if compared >= 8 || got.dtype() != want.dtype() || got.shape() != want.shape() {
            continue; // packed at this tier — covered by the dequantize check on the whole tier
        }
        assert_eq!(
            max_abs(&got.subtract(want).expect("sub")),
            0.0,
            "{label}: shard {shard:?} tensor `{name}` does not match the source bit for bit"
        );
        compared += 1;
    }
    compared
}

/// The stock-Wan companion directory (`t5_encoder.safetensors` + `vae.safetensors` +
/// `tokenizer.json`) staged for the rehost.
fn companions_dir() -> PathBuf {
    std::env::var("KREA_REHOST_COMPANIONS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home().join(".cache/krea-realtime-mlx-companions"))
}

/// Krea Realtime ships **transformer-only**: the text encoder, VAE and tokenizer are the stock Wan 2.1
/// components, published into every tier subdirectory so each tier is a self-contained snapshot (the
/// `SceneWorks/scail2-mlx` convention). This proves the three staged companion files load through the
/// *same* `mlx_gen_wan` loaders `t2v.rs::load_components` drives, at the Krea config's geometry — so a
/// wrong-model or wrong-layout companion is caught here rather than at first generation.
///
/// A prompt is actually encoded (not merely loaded) so the UMT5 stack is exercised end to end, and the
/// VAE is round-tripped on a tiny clip so the z16 latent geometry is confirmed against the config.
#[test]
#[ignore = "needs the ~11.9 GB staged Wan companions; run with --ignored on macOS (see module doc)"]
fn stock_wan_companions_load_at_the_krea_geometry() {
    use mlx_gen_wan::{load_tokenizer, Umt5Encoder, WanVae};

    let root = companions_dir();
    let cfg = KreaRealtimeConfig::krea_realtime_14b();

    // Text encoder + tokenizer — the `encode_prompt` path.
    let tokenizer = load_tokenizer(root.join("tokenizer.json"), cfg.wan.text_len)
        .expect("stock Wan UMT5 tokenizer.json");
    let w =
        Weights::from_file(root.join("t5_encoder.safetensors")).expect("t5_encoder.safetensors");
    let enc = Umt5Encoder::from_weights(&w, &cfg.wan).expect("UMT5 encoder at the Krea geometry");
    let context = enc
        .encode(&tokenizer, "a cinematic shot of a city street at night")
        .expect("encode a prompt");
    mlx_rs::transforms::eval([&context]).expect("eval context");
    // `encode` returns the prompt's real token rows (the DiT pads/masks to `text_len`), so the
    // load-bearing assertion is the feature width — UMT5-XXL's 4096 — and a non-empty, in-bounds
    // sequence. A wrong text encoder would land on a different `text_dim` immediately.
    let shape = context.shape().to_vec();
    assert_eq!(shape.len(), 2, "prompt context must be [tokens, text_dim]");
    assert_eq!(
        shape[1], cfg.wan.text_dim as i32,
        "prompt context width must be UMT5-XXL's text_dim"
    );
    assert!(
        shape[0] > 0 && shape[0] <= cfg.wan.text_len as i32,
        "prompt token count {} must be in (0, text_len={}]",
        shape[0],
        cfg.wan.text_len
    );
    assert!(
        max_abs(&context) > 0.0,
        "the encoded context is all-zero — wrong or truncated text encoder"
    );

    // VAE — the `load_components` path, plus a latent round-trip at the z16 geometry.
    let w = Weights::from_file(root.join("vae.safetensors")).expect("vae.safetensors");
    let vae = WanVae::from_weights(&w).expect("stock Wan 2.1 z16 VAE");
    let (t, h, w_px) = (5, 64, 64);
    let video = Array::zeros::<f32>(&[1, 3, t, h, w_px]).expect("zeros");
    let z = vae.encode(&video).expect("vae encode");
    let stride = cfg.wan.vae_stride;
    assert_eq!(
        z.shape(),
        &[
            1,
            cfg.wan.vae_z_dim as i32,
            (t - 1) / stride.0 as i32 + 1,
            h / stride.1 as i32,
            w_px / stride.2 as i32,
        ],
        "z16 latent geometry must match the config's vae_stride / vae_z_dim"
    );

    println!(
        "companions OK at {}: UMT5 context {:?}, z16 latent {:?}",
        root.display(),
        context.shape(),
        z.shape()
    );
}

/// Largest absolute element of `a`, as f32.
fn max_abs(a: &Array) -> f32 {
    a.as_dtype(Dtype::Float32)
        .expect("f32")
        .abs()
        .expect("abs")
        .max(None)
        .expect("max")
        .item::<f32>()
}

/// **Content** fidelity of the emitted tier — the check that shape-level verification cannot make.
///
/// `verify_transformer_tensors` and the tier probe both read only names and shapes, so a converter
/// that wrote correctly-shaped *garbage* (a mis-strided read, a bad cast, quantizing the wrong axis)
/// would sail through both and only show up as a ruined generation after a multi-GB upload. This
/// reconstructs the emitted tensors and compares them against the source checkpoint:
///
///   * **pass-through** tensors (norms, modulation tables, embeddings, the head — everything outside
///     the quantize predicate) must be **bit-exact** with the source cast F16 → bf16, on every tier;
///   * **packed** Linears must dequantize back to the source weight within the group-affine
///     round-trip error for their width (Q8 strictly tighter than Q4), and must not be degenerate
///     (all-zero scales, or a constant reconstruction).
fn check_tier_reconstructs_the_source(
    src: &Path,
    dit: &Path,
    label: &str,
    quant: Option<(i32, i32)>,
) {
    let source = sanitize_krea_realtime_transformer(read_native(src)).expect("sanitize source");
    let emitted = read_native(dit);

    // Pass-through tensors are byte-identical on every tier — sample across the whole model.
    let passthrough = [
        "patch_embedding_proj.weight",
        "text_embedding_0.weight",
        "time_projection.weight",
        "head.head.weight",
        "head.modulation",
        "blocks.0.modulation",
        "blocks.39.norm3.weight",
    ];
    for name in passthrough {
        let want = source.get(name).unwrap_or_else(|| panic!("source {name}"));
        let got = emitted
            .get(name)
            .unwrap_or_else(|| panic!("emitted {name}"));
        assert_eq!(got.dtype(), Dtype::Bfloat16, "{label}/{name}: dense bf16");
        assert_eq!(got.shape(), want.shape(), "{label}/{name}: shape");
        let diff = max_abs(&got.subtract(want).expect("sub"));
        assert_eq!(
            diff, 0.0,
            "{label}/{name}: pass-through tensor must be bit-exact with the source"
        );
    }

    // The quantize-predicate Linears: dequantize and compare against the source weight.
    let probes = [
        "blocks.0.self_attn.q.weight",
        "blocks.20.cross_attn.v.weight",
        "blocks.39.ffn.fc2.weight",
    ];
    for base in probes {
        let want = source.get(base).unwrap_or_else(|| panic!("source {base}"));
        let got = emitted
            .get(base)
            .unwrap_or_else(|| panic!("emitted {base}"));
        let scale = max_abs(want);
        assert!(scale > 0.0, "{label}/{base}: source weight is all-zero");

        let Some((bits, group_size)) = quant else {
            // bf16 tier: the predicate Linears are dense too, so this is also bit-exact.
            assert_eq!(got.dtype(), Dtype::Bfloat16, "{label}/{base}: dense bf16");
            let diff = max_abs(&got.subtract(want).expect("sub"));
            assert_eq!(diff, 0.0, "{label}/{base}: dense tier must be bit-exact");
            continue;
        };

        assert_eq!(got.dtype(), Dtype::Uint32, "{label}/{base}: packed codes");
        let scales = emitted
            .get(&format!("{}.scales", base.trim_end_matches(".weight")))
            .unwrap_or_else(|| panic!("emitted scales for {base}"));
        let biases = emitted
            .get(&format!("{}.biases", base.trim_end_matches(".weight")))
            .unwrap_or_else(|| panic!("emitted biases for {base}"));
        assert!(
            max_abs(scales) > 0.0,
            "{label}/{base}: degenerate all-zero scales"
        );

        let recon = dequantize(got, scales, biases, group_size, bits).expect("dequantize");
        assert_eq!(recon.shape(), want.shape(), "{label}/{base}: recon shape");
        let err = max_abs(&recon.subtract(want).expect("sub")) / scale;
        // Group-affine round-trip: the reconstruction lands within one quantization step of the
        // group's range. Generous vs. theory (2^-bits) but far tighter than garbage, which would be
        // O(1) relative error.
        let tol = if bits == 8 { 0.02 } else { 0.25 };
        assert!(
            err < tol,
            "{label}/{base}: Q{bits} dequantized reconstruction is off by {err} (relative), \
             tolerance {tol} — the packed tier does not reconstruct the source weight"
        );
        // Not a constant: a collapsed reconstruction would still pass a loose error bound if the
        // source happened to be small, so require real spread. `max_abs > 0` would NOT show this — a
        // nonzero constant satisfies it — so compare the extremes.
        let r32 = recon.as_dtype(Dtype::Float32).expect("f32");
        let spread =
            r32.max(None).expect("max").item::<f32>() - r32.min(None).expect("min").item::<f32>();
        assert!(
            spread > 0.0,
            "{label}/{base}: reconstruction is constant (spread {spread}) — not a real weight"
        );
        println!("  {label}/{base}: Q{bits} relative reconstruction error {err:.4} (< {tol})");
    }
    println!("tier {label}: content fidelity vs. the source checkpoint OK");
}
