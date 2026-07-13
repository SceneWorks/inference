//! sc-5917: FLUX.2-**dev** pre-quantization (producer + packed consumer) on the real checkpoint.
//! `#[ignore]`d — needs the real `black-forest-labs/FLUX.2-dev` snapshot (~60 GB DiT + ~45 GB TE):
//!
//!   cargo test -p mlx-gen-flux2 --release --test quant_prequantize_real_weights -- --ignored --nocapture
//!
//! Each test is the offline convert vehicle AND the integration proof: it pre-quantizes one
//! component to a temp dir (the producer, `convert::quantize_flux2_*`), then loads it back through
//! the *packed* loader (the consumer, `load_*_dev` → `from_weights_quant`) and runs a real-dimension
//! forward — proving the packed snapshot round-trips with no dense bf16 weight materialized. The
//! deterministic byte-parity of the packing vs the load-time `AdaptableLinear::quantize` op is
//! closed cheaply in the crate unit tests (`convert::tests`), so this stays memory-bounded (the only
//! dense transient is the one-off offline convert read).
//!
//! **Footprint** (the sc-5917 acceptance metric) is measured by wrapping the TEST BINARY directly:
//!   /usr/bin/time -l target/release/deps/quant_prequantize_real_weights-<hash> \
//!       dit_prequantize_loads_packed_and_forwards --ignored --nocapture
//! ("peak memory footprint" — MLX's Metal-wired allocation does not show in `ps` RSS; and wrap the
//! built binary, not `cargo test`, which would report the cargo parent's peak).

use std::path::{Path, PathBuf};

use mlx_gen_flux2::{
    create_noise, load_text_encoder_dev, load_transformer_dev, prepare_grid_ids, prepare_text_ids,
    quantize_flux2_dit, quantize_flux2_text_encoder_dir,
};
use mlx_rs::{random, Dtype};

const BITS: i32 = 4;
const GROUP_SIZE: i32 = 64;

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX2_DEV_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots");
    std::fs::read_dir(&snaps)
        .expect("snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir under models--black-forest-labs--FLUX.2-dev/snapshots")
}

/// A stable temp output root for the pre-quantized components, so a second run (e.g. the TE test
/// after the DiT test) reuses an already-converted component instead of re-packing.
fn out_root() -> PathBuf {
    std::env::temp_dir().join(format!("mlx_gen_flux2_dev_prequant_q{BITS}"))
}

#[test]
#[ignore = "needs real FLUX.2-dev snapshot (~60 GB DiT); writes a Q4 packed dir to TMPDIR"]
fn dit_prequantize_loads_packed_and_forwards() {
    let snap = snapshot();
    let dst = out_root();
    let dst_transformer = dst.join("transformer");

    // Producer: pack the dense bf16 DiT → Q4 on disk (skip if a prior run already did).
    if !dst_transformer
        .join("diffusion_pytorch_model.safetensors")
        .exists()
    {
        println!(
            "converting dev DiT → Q{BITS} (group {GROUP_SIZE}) at {}",
            dst_transformer.display()
        );
        quantize_flux2_dit(
            &snap.join("transformer"),
            &dst_transformer,
            BITS,
            GROUP_SIZE,
        )
        .expect("pre-quantize dev DiT");
    } else {
        println!("reusing pre-quantized DiT at {}", dst_transformer.display());
    }

    // Consumer: the manifest in transformer/config.json flips load_transformer_dev to the packed
    // path — every predicate Linear built from packed Q4 parts, no dense 60 GB transient.
    let t = load_transformer_dev(&dst).expect("packed dev transformer loads");

    // The probe Linear is packed (u32 codes), at the manifest's group_size/bits.
    let (_wq, _sc, _bi, gs, bits) = t.probe_quant_to_q().expect("to_q is packed Q4");
    assert_eq!(
        (bits, gs),
        (BITS, GROUP_SIZE),
        "packed at the manifest's Q{BITS}/group {GROUP_SIZE}"
    );

    // A real-dimension forward runs finite through the packed quantized stack.
    let (w, h) = (64u32, 64u32);
    let hidden = create_noise(0, w, h, 128).unwrap(); // [1,16,128]
    let key = random::key(1).unwrap();
    let encoder = random::normal::<f32>(&[1, 8, 15360][..], None, None, Some(&key)).unwrap();
    let img_ids = prepare_grid_ids((h / 16) as usize, (w / 16) as usize, 0);
    let txt_ids = prepare_text_ids(8);
    let out = t
        .forward(&hidden, &encoder, &img_ids, &txt_ids, 500.0)
        .unwrap();
    assert_eq!(out.shape(), &[1, 16, 128], "velocity shape");
    let total = out
        .as_dtype(Dtype::Float32)
        .unwrap()
        .sum(None)
        .unwrap()
        .item::<f32>();
    assert!(
        total.is_finite(),
        "packed dev DiT output is non-finite: {total}"
    );
    println!(
        "flux2-dev Q{BITS} DiT packed-load + forward OK: shape {:?}, finite",
        out.shape()
    );
    print_disk_size(&dst_transformer.join("diffusion_pytorch_model.safetensors"));
}

#[test]
#[ignore = "needs real FLUX.2-dev snapshot (~45 GB Mistral TE); writes a Q4 packed dir to TMPDIR"]
fn te_prequantize_loads_packed_and_encodes() {
    let snap = snapshot();
    let dst = out_root();
    let dst_te = dst.join("text_encoder");

    if !dst_te.join("model.safetensors").exists() {
        println!(
            "converting dev Mistral TE → Q{BITS} (group {GROUP_SIZE}) at {}",
            dst_te.display()
        );
        quantize_flux2_text_encoder_dir(&snap.join("text_encoder"), &dst_te, BITS, GROUP_SIZE)
            .expect("pre-quantize dev TE");
    } else {
        println!("reusing pre-quantized TE at {}", dst_te.display());
    }

    let te = load_text_encoder_dev(&dst).expect("packed dev TE loads");

    // The token embedding is packed (the unique Embedding case).
    let (_wq, _sc, _bi, gs, bits) = te.probe_quant_embed().expect("embed_tokens is packed Q4");
    assert_eq!(
        (bits, gs),
        (BITS, GROUP_SIZE),
        "TE packed at Q{BITS}/group {GROUP_SIZE}"
    );

    // A short prompt encodes finite through the packed quantized layers → the 15360-wide embeds.
    let input_ids = mlx_rs::Array::from_slice(&[1i32, 2, 3, 4, 5, 6], &[1, 6]);
    let mask = mlx_rs::Array::from_slice(&[1i32, 1, 1, 1, 1, 1], &[1, 6]);
    let embeds = te.prompt_embeds(&input_ids, &mask).unwrap();
    assert_eq!(
        embeds.shape(),
        &[1, 6, 15360],
        "dev prompt-embeds width 15360"
    );
    let total = embeds
        .as_dtype(Dtype::Float32)
        .unwrap()
        .sum(None)
        .unwrap()
        .item::<f32>();
    assert!(
        total.is_finite(),
        "packed dev TE embeds non-finite: {total}"
    );
    println!(
        "flux2-dev Q{BITS} Mistral TE packed-load + encode OK: shape {:?}, finite",
        embeds.shape()
    );
    print_disk_size(&dst_te.join("model.safetensors"));
}

fn print_disk_size(path: &Path) {
    if let Ok(m) = std::fs::metadata(path) {
        println!(
            "  on-disk {}: {:.2} GB",
            path.display(),
            m.len() as f64 / 1e9
        );
    }
}
