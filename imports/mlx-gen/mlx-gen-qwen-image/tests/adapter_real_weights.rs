//! sc-2528: end-to-end Qwen-Image LoRA/LoKr adapter consumption against real weights.
//!
//! `#[ignore]`d — needs the real `Qwen/Qwen-Image` snapshot in the HF cache (env
//! `QWEN_IMAGE_SNAPSHOT`) and the adapter goldens from `tools/dump_qwen_adapter_golden.py`
//! (gitignored, local). Run:
//!   cargo test -p mlx-gen-qwen-image --release --test adapter_real_weights -- --ignored --nocapture
//!
//! Gates: (1) the key→module map resolves the FULL fork `QwenLoRAMapping` surface (60 blocks ×
//! attention + img/txt MLP) against the real module tree; (2) the public
//! `load(spec.with_adapters(…)).generate()` render matches the fork's LoRA *and* LoKr golden
//! (px>8); (3) a scale-0 adapter is a bit-exact no-op.

use std::path::PathBuf;

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use mlx_gen_qwen_image::{decoded_to_image, loader};

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN_IMAGE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/golden")
}

/// (1) The top-level `AdaptableHost` resolves every fork `QwenLoRAMapping` target (all per-block:
/// the joint attention + the two stream MLPs; no globals) across the real 60-block tree, and
/// rejects off-surface paths.
#[test]
#[ignore = "needs real Qwen-Image weights"]
fn routing_map_covers_full_fork_surface() {
    let mut t = loader::load_transformer(&snapshot()).unwrap();
    let resolves = |t: &mut _, p: &str| -> bool {
        let segs: Vec<&str> = p.split('.').collect();
        AdaptableHost::adaptable_mut(t, &segs).is_some()
    };

    let targets = [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "img_mlp.net.0.proj",
        "img_mlp.net.2",
        "txt_mlp.net.0.proj",
        "txt_mlp.net.2",
    ];
    for i in 0..60 {
        for tgt in targets {
            let p = format!("transformer_blocks.{i}.{tgt}");
            assert!(resolves(&mut t, &p), "expected {p} to resolve");
        }
    }
    for p in [
        "transformer_blocks.60.attn.to_q",    // out of range
        "transformer_blocks.0.attn.to_out",   // missing .0
        "transformer_blocks.0.attn.add_q",    // internal name, not the file's add_q_proj
        "transformer_blocks.0.img_mlp.net.1", // gelu slot
        "img_in",                             // not a trained target
    ] {
        assert!(!resolves(&mut t, p), "expected {p} NOT to resolve");
    }
    println!("✓ routing map covers the full fork QwenLoRAMapping surface (60 blocks × 12 targets)");
}

fn render(adapter: Option<(&str, AdapterKind, f32)>, golden_kind: &str) -> Vec<u8> {
    let g = Weights::from_file(golden_dir().join(format!("qwen_{golden_kind}_golden.safetensors")))
        .unwrap();
    let prompt = g.metadata("prompt").unwrap().to_string();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let steps: u32 = g.metadata("steps").unwrap().parse().unwrap();
    let w: u32 = g.metadata("width").unwrap().parse().unwrap();
    let h: u32 = g.metadata("height").unwrap().parse().unwrap();

    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    if let Some((file, kind, scale)) = adapter {
        spec = spec.with_adapters(vec![AdapterSpec {
            path: golden_dir().join(file),
            scale,
            kind,
        }]);
    }
    let generator = mlx_gen::load("qwen_image", &spec).unwrap();
    let req = GenerationRequest {
        prompt,
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        ..Default::default()
    };
    let out = generator.generate(&req, &mut |_| {}).unwrap();
    match out {
        GenerationOutput::Images(mut v) => v.pop().unwrap().pixels,
        other => panic!("expected Images, got {other:?}"),
    }
}

fn assert_matches_golden(kind: &str, my_kind: AdapterKind) {
    let pixels = render(
        Some((&format!("qwen_{kind}_adapter.safetensors"), my_kind, 1.0)),
        kind,
    );
    let g =
        Weights::from_file(golden_dir().join(format!("qwen_{kind}_golden.safetensors"))).unwrap();
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let differ = pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    let frac = differ as f64 / pixels.len() as f64;
    println!(
        "✓ qwen {kind} adapter render: {differ}/{} px differ by >8 from the fork ({:.4}%)",
        pixels.len(),
        frac * 100.0
    );
    assert!(
        differ < pixels.len() / 20,
        "qwen {kind} adapter render diverges from the fork: {differ} px ({:.3}%)",
        frac * 100.0
    );
}

#[test]
#[ignore = "needs real Qwen-Image weights + adapter golden"]
fn lora_render_matches_fork_golden() {
    assert_matches_golden("lora", AdapterKind::Lora);
}

#[test]
#[ignore = "needs real Qwen-Image weights + adapter golden"]
fn lokr_render_matches_fork_golden() {
    assert_matches_golden("lokr", AdapterKind::Lokr);
}

/// Diagnostic (per the divergence-is-not-rounding rule): the Rust base render (no adapter) vs the
/// fork base golden at the SAME 256² config — the floor the LoRA/LoKr px>8 numbers inherit. Needs
/// `QWEN_W=256 QWEN_H=256 dump_qwen_image_golden.py` (gitignored base golden).
#[test]
#[ignore = "needs real Qwen-Image weights + base golden @256"]
fn base_render_drift_attributes_adapter_gap() {
    let pixels = render(None, "lora"); // config (256², seed 42, 4 steps) read from the lora golden
    let g = Weights::from_file(golden_dir().join("qwen_image_golden.safetensors")).unwrap();
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let differ = pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    println!(
        "qwen base (no adapter) vs fork base @256²: {differ}/{} px>8 ({:.4}%)",
        pixels.len(),
        differ as f64 / pixels.len() as f64 * 100.0
    );
}

#[test]
#[ignore = "needs real Qwen-Image weights + adapter golden"]
fn scale_zero_adapter_is_noop() {
    let base = render(None, "lora");
    let zero = render(
        Some(("qwen_lora_adapter.safetensors", AdapterKind::Lora, 0.0)),
        "lora",
    );
    let differ = base.iter().zip(&zero).filter(|(a, b)| a != b).count();
    println!("✓ qwen scale-0 adapter no-op: {differ} px differ from the no-adapter render");
    assert_eq!(differ, 0, "scale-0 adapter must be a bit-exact no-op");
}
