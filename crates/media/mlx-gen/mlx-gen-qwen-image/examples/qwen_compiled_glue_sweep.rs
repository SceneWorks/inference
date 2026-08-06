//! sc-17513 sweep harness: the producer behind every number in
//! `tests/perf.rs::COMPILED_GLUE_WHOLE_FWD_REL_TOL_T2I` and
//! `tests/perf.rs::COMPILED_GLUE_WHOLE_FWD_REL_TOL_EDIT` — one bound per arm, each derived from
//! that arm's own run of this sweep.
//!
//! It discriminates (a) the bf16/Metal + `mx.compile` f32 ULP floor amplified by a 60-layer stack
//! from (b) a genuine compiled-glue fusion divergence on the Qwen-Image MMDiT, and it derives each
//! arm's bound from that arm's `conditioning` mode measured envelope. It is an **example, not a
//! test**, on purpose: it is far too expensive to wire (the 13-shape sweep is ~10 minutes of 40
//! GiB forwards, and `depth` wants one process per depth), and adding four `#[ignore]` tests would
//! enlarge the per-variable test census `release/real-weight-models.toml` accounts for without
//! buying any CI coverage. The gate that CI runs is `tests/perf.rs`; this is how its numbers were
//! obtained and how to obtain them again.
//!
//! Run from the workspace root, one mode per process:
//! ```text
//! SNAP=…/qwen-image-mlx/snapshots/8080a417…/bf16
//! ESNAP=…/qwen-image-edit-2511-mlx/snapshots/0dfbf3a0…/q8
//!
//! MLX_GEN_QWEN_SNAPSHOT=$SNAP  cargo run --release -p mlx-gen-qwen-image \
//!     --example qwen_compiled_glue_sweep -- shapes
//! MLX_GEN_QWEN_SNAPSHOT=$SNAP  cargo run --release -p mlx-gen-qwen-image \
//!     --example qwen_compiled_glue_sweep -- conditioning
//! QWEN_IMAGE_EDIT_SNAPSHOT=$ESNAP cargo run --release -p mlx-gen-qwen-image \
//!     --example qwen_compiled_glue_sweep -- edit
//! QWEN_IMAGE_EDIT_SNAPSHOT=$ESNAP cargo run --release -p mlx-gen-qwen-image \
//!     --example qwen_compiled_glue_sweep -- conditioning-edit
//! for L in 1 2 4 8 15 30 45 60; do SC17513_LAYERS=$L MLX_GEN_QWEN_SNAPSHOT=$SNAP \
//!     cargo run --release -p mlx-gen-qwen-image --example qwen_compiled_glue_sweep -- depth; done
//! ```
//!
//! MEMORY DISCIPLINE, and it is not optional: the bf16 T2I model is ~38 GiB live. One mode per
//! process, never two concurrently. Within a mode the model is loaded **once** and shapes are
//! iterated in-process with `clear_cache()` between them; `depth` takes its layer count from the
//! environment so peak is bounded by the single largest model rather than by whatever the
//! allocator declines to release between rebuilds.

use mlx_gen::weights::Weights;
use mlx_gen_qwen_image::loader::remap_transformer_keys;
use mlx_gen_qwen_image::transformer::{set_compile_glue, QwenTransformer, QwenTransformerConfig};
use mlx_rs::memory::{clear_cache, get_active_memory, get_cache_memory, get_peak_memory};
use mlx_rs::{random, Array, Dtype};
use std::path::{Path, PathBuf};

/// One f32 ULP, relative. The unit both controls are expressed in.
const ULP: f32 = f32::EPSILON; // 1.1920929e-7

fn snapshot(env: &str) -> PathBuf {
    PathBuf::from(
        std::env::var(env)
            .unwrap_or_else(|_| panic!("set {env} to the snapshot TIER dir (…/bf16, …/q8)")),
    )
}

fn max_abs(a: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    mlx_rs::ops::max(mlx_rs::ops::abs(&a).unwrap(), None)
        .unwrap()
        .item::<f32>()
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    max_abs(&mlx_rs::ops::subtract(&a, &b).unwrap())
}

fn gib(b: usize) -> f64 {
    b as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn mem(tag: &str) {
    println!(
        "    [mem {tag}] active={:.2} GiB cache={:.2} GiB peak={:.2} GiB",
        gib(get_active_memory()),
        gib(get_cache_memory()),
        gib(get_peak_memory())
    );
}

/// f32 packed image latents `[1, seq, 64]` and bf16 text embeds `[1, txt, 3584]` — production
/// dtypes, and the same fixed key `perf.rs` uses, so the two agree shape for shape.
fn inputs(img_seq: i32, txt_seq: i32) -> (Array, Array) {
    let key = random::key(0).unwrap();
    let hidden = random::normal::<f32>(&[1, img_seq, 64], None, None, Some(&key)).unwrap();
    let encoder = random::normal::<f32>(&[1, txt_seq, 3584], None, None, Some(&key))
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    mlx_rs::transforms::eval([&hidden, &encoder]).unwrap();
    (hidden, encoder)
}

/// Load the real transformer at a chosen depth. `layers < 60` truncates the `blocks` Vec only
/// (`transformer.rs`); it does not touch `set_compile_glue`, which is a process-global atomic in
/// `mlx-gen/src/nn.rs` — so the depth series really is the same fusion at a shorter stack.
fn load(root: &Path, layers: usize, edit: bool) -> QwenTransformer {
    let mut w = Weights::from_dir(root.join("transformer")).expect("weights");
    remap_transformer_keys(&mut w);
    let mut cfg = if edit {
        QwenTransformerConfig::qwen_image_edit()
    } else {
        QwenTransformerConfig::qwen_image()
    };
    cfg.num_layers = layers;
    QwenTransformer::from_weights(&w, "", &cfg).expect("transformer")
}

/// One row of the shape / depth series: eager x2 and compiled x2, so the repeat-determinism
/// control rides along with every measurement instead of being a separate claim. Prints
/// `max|out|` (the denominator) next to `max|Δ|`, because a bare absolute figure is not comparable
/// across rows — at truncated depth the output magnitude is ~25x its trained value.
fn row(
    label: &str,
    t: &QwenTransformer,
    hidden: &Array,
    encoder: &Array,
    lat: usize,
    grids: &[(usize, usize)],
) {
    let run = || {
        t.forward(hidden, encoder, None, 1.0f32, lat, lat, grids)
            .unwrap()
    };
    set_compile_glue(false);
    let e1 = run();
    mlx_rs::transforms::eval([&e1]).unwrap();
    let e2 = run();
    mlx_rs::transforms::eval([&e2]).unwrap();
    set_compile_glue(true);
    let c1 = run();
    mlx_rs::transforms::eval([&c1]).unwrap();
    let c2 = run();
    mlx_rs::transforms::eval([&c2]).unwrap();
    set_compile_glue(false);

    let m = max_abs(&e1);
    let d = max_abs_diff(&c1, &e1);
    let rel = d / m;
    println!(
        "{label}  max|out|={m:.5}  max|D|={d:.4e}  rel={rel:.4e}  relULP={:.1}  \
         eager-repeat={:.3e}  compiled-repeat={:.3e}  c2-vs-e1={:.4e}",
        rel / ULP,
        max_abs_diff(&e2, &e1),
        max_abs_diff(&c2, &c1),
        max_abs_diff(&c2, &e1),
    );
}

/// A) T2I shape sweep at full depth, with per-shape repeat-determinism controls.
fn sweep_shapes() {
    let snap = snapshot("MLX_GEN_QWEN_SNAPSHOT");
    let t = load(&snap, 60, false);
    mem("after load");
    let txt_seq = 128;
    for size in [
        128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1152, 1280,
    ] {
        let lat = (size / 16) as usize;
        let img_seq = (lat * lat) as i32;
        let (hidden, encoder) = inputs(img_seq, txt_seq);
        row(
            &format!("size={size:<5} img_seq={img_seq:<5}"),
            &t,
            &hidden,
            &encoder,
            lat,
            &[],
        );
        drop(hidden);
        drop(encoder);
        clear_cache();
    }
    mem("end");
}

/// B) Depth probe at a fixed shape — does the divergence accumulate with layer count? ONE depth
/// per process (`SC17513_LAYERS`), driven from a shell loop.
fn sweep_depth() {
    let snap = snapshot("MLX_GEN_QWEN_SNAPSHOT");
    let layers: usize = std::env::var("SC17513_LAYERS")
        .expect("set SC17513_LAYERS")
        .parse()
        .unwrap();
    let txt_seq = 128;
    let size = 512;
    let lat = (size / 16) as usize;
    let img_seq = (lat * lat) as i32;
    let (hidden, encoder) = inputs(img_seq, txt_seq);
    let t = load(&snap, layers, false);
    row(
        &format!("layers={layers:<4} size={size} img_seq={img_seq}"),
        &t,
        &hidden,
        &encoder,
        lat,
        &[],
    );
    mem(&format!("{layers} layers"));
}

/// C) CONDITIONING CONTROL — the decisive discriminator, and the source of the gate's bound.
///
/// How far does the EAGER forward move when its OWN input is perturbed by k f32 ULP relative, with
/// compiled glue never switched on? If that envelope is the same order as the compiled-vs-eager
/// divergence, the divergence is the ULP floor rather than a fusion bug — and the envelope, not
/// the divergence, is the quantity a bound should be derived from, because it measures what the
/// stack does to *any* sub-ULP difference rather than what it did to one particular one.
///
/// k = 4 is measured alongside k = 1 to **test whether** the response scales with the perturbation
/// magnitude — and the answer is that it does not. It plateaus: across the 16 paired probes the
/// largest increase is 2.1× and 6 of 16 pairs *decreased*, so quadrupling the input does not
/// quadruple the output difference. That is why `tests/perf.rs` documents the ×4 in each bound as
/// margin for **shapes not sampled** and for the second host, and explicitly **not** as a magnitude
/// extrapolation — extrapolating linearly off k = 1 would overstate it. (k = 4 is worth probing at
/// all because 4 ULP is [`mlx_gen::nn::COMPILED_GLUE_F32_ULP_TOL`], the op-level budget the fused
/// glue is separately gated to.)
///
/// Run per arm: the T2I bf16 and Edit q8 tiers are different weight encodings and there is no
/// reason to assume one arm's envelope transfers to the other, so each arm's bound is derived from
/// its own.
fn sweep_conditioning(edit: bool) {
    let snap = snapshot(if edit {
        "QWEN_IMAGE_EDIT_SNAPSHOT"
    } else {
        "MLX_GEN_QWEN_SNAPSHOT"
    });
    let t = load(&snap, 60, edit);
    let txt_seq = 128;
    for size in [256, 384, 512, 1024] {
        let lat = (size / 16) as usize;
        // Edit is dual-latent: noise grid + one same-size reference, concatenated.
        let img_seq = (lat * lat) as i32 * if edit { 2 } else { 1 };
        let cond = [(lat, lat)];
        let grids: &[(usize, usize)] = if edit { &cond } else { &[] };
        let (hidden, encoder) = inputs(img_seq, txt_seq);
        set_compile_glue(false);
        let base = t
            .forward(&hidden, &encoder, None, 1.0f32, lat, lat, grids)
            .unwrap();
        mlx_rs::transforms::eval([&base]).unwrap();
        let m = max_abs(&base);
        println!("size={size} img_seq={img_seq} max|out|={m:.5}");

        let probe = |tag: &str, perturbed: Array| {
            let in_rel = max_abs_diff(&perturbed, &hidden) / max_abs(&hidden);
            let out = t
                .forward(&perturbed, &encoder, None, 1.0f32, lat, lat, grids)
                .unwrap();
            mlx_rs::transforms::eval([&out]).unwrap();
            let out_rel = max_abs_diff(&out, &base) / m;
            println!(
                "  {tag:<12}: in_rel={in_rel:.4e} ({:.1} inULP) out_rel={out_rel:.4e} \
                 ({:.0} outULP) amplification={:.3e}",
                in_rel / ULP,
                out_rel / ULP,
                out_rel / in_rel,
            );
        };

        // Uniform relative gain: every element moved the same signed fraction.
        for k in [1.0f32, 4.0] {
            probe(
                &format!("gain-{k:.0}ULP"),
                mlx_rs::ops::multiply(&hidden, Array::from_slice(&[1.0f32 + k * ULP], &[1]))
                    .unwrap(),
            );
        }
        // Random relative noise — closer in shape to a per-element rounding difference, which is
        // what a fused kernel actually produces.
        let k7 = random::key(7).unwrap();
        let u = random::normal::<f32>(&[1, img_seq, 64], None, None, Some(&k7)).unwrap();
        for k in [1.0f32, 4.0] {
            probe(
                &format!("noise-{k:.0}ULP"),
                mlx_rs::ops::multiply(
                    &hidden,
                    mlx_rs::ops::add(
                        Array::from_slice(&[1.0f32], &[1]),
                        mlx_rs::ops::multiply(&u, Array::from_slice(&[k * ULP], &[1])).unwrap(),
                    )
                    .unwrap(),
                )
                .unwrap(),
            );
        }
        clear_cache();
    }
    mem("end");
}

/// D) Edit arm at the CI tier (q8), exercising the dual-latent `modulate_index` route.
fn sweep_edit() {
    let snap = snapshot("QWEN_IMAGE_EDIT_SNAPSHOT");
    let t = load(&snap, 60, true);
    mem("after load");
    let txt_seq = 128;
    for size in [256, 384, 512, 768, 1024] {
        let lat = (size / 16) as usize;
        let noise_seq = (lat * lat) as i32;
        let img_seq = noise_seq * 2;
        let (hidden, encoder) = inputs(img_seq, txt_seq);
        row(
            &format!("size={size:<5} img_seq={img_seq:<5}"),
            &t,
            &hidden,
            &encoder,
            lat,
            &[(lat, lat)],
        );
        clear_cache();
    }
    mem("end");
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("shapes") => sweep_shapes(),
        Some("depth") => sweep_depth(),
        Some("conditioning") => sweep_conditioning(false),
        Some("conditioning-edit") => sweep_conditioning(true),
        Some("edit") => sweep_edit(),
        other => panic!(
            "usage: … --example qwen_compiled_glue_sweep -- <shapes|depth|conditioning|conditioning-edit|edit> \
             (got {other:?}); see this file's module doc for the env vars each mode needs"
        ),
    }
}
