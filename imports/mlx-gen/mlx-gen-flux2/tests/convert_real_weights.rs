//! sc-3136: real-weights smoke for the native single-file → diffusers converter
//! ([`mlx_gen_flux2::convert_and_assemble`]). `#[ignore]`d — needs BOTH the base
//! `black-forest-labs/FLUX.2-klein-9b` snapshot AND the wikeeyang `Flux2-Klein-9B-True-V2`
//! `*-bf16.safetensors` single file:
//!
//!   cargo test -p mlx-gen-flux2 --test convert_real_weights -- --ignored --nocapture
//!
//! The committed `convert` unit tests prove the remap *math* (qkv split, adaLN half-swap, key
//! renames) on synthetic tensors; this proves the *whole assembly* on the real fine-tune: the
//! converter's internal base-validation guard passes (produced keyset+shapes == base's 233), the
//! borrowed vae/text_encoder/tokenizer/scheduler are symlinked to the base, and the assembled
//! `transformer/` loads through the production `load_transformer` and forwards finite — i.e. the
//! dir is exactly what the `flux2_klein_9b` loader consumes via the worker's `modelPath` seam.

use std::path::PathBuf;

use mlx_gen_flux2::{
    convert_and_assemble, create_noise, load_transformer, prepare_grid_ids, prepare_text_ids,
};
use mlx_rs::Dtype;

/// Base FLUX.2-klein-9b diffusers snapshot (env `MLX_GEN_FLUX2_SNAPSHOT` or the HF cache).
fn base_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX2_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9b/snapshots");
    std::fs::read_dir(&snaps)
        .expect("base snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a base snapshot dir")
}

/// wikeeyang true_v2 single-file transformer, bf16 (env `MLX_GEN_FLUX2_TRUE_V2_FILE` or the HF
/// cache). This is the exact file the SceneWorks manifest's `convertSourceFile` targets.
fn true_v2_bf16_file() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX2_TRUE_V2_FILE") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--wikeeyang--Flux2-Klein-9B-True-V2/snapshots");
    let snap = std::fs::read_dir(&snaps)
        .expect("true_v2 snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a true_v2 snapshot dir");
    let file = snap.join("Flux2-Klein-9B-True-v2-bf16.safetensors");
    assert!(
        file.is_file(),
        "missing bf16 single file: {}",
        file.display()
    );
    file
}

#[test]
#[ignore = "needs base FLUX.2-klein-9b snapshot + wikeeyang true_v2 bf16 single file (~35 GB total)"]
fn convert_assembles_loadable_diffusers_dir() {
    let base = base_snapshot();
    let source = true_v2_bf16_file();
    let out = std::env::temp_dir().join("mlx_gen_flux2_true_v2_convert_out");
    let _ = std::fs::remove_dir_all(&out); // idempotent: clear any prior run

    // Convert + assemble. The internal base-validation guard asserts produced keyset+shapes match
    // the base diffusers transformer exactly — so a returned Ok is already structural proof.
    let assembled = convert_and_assemble(&source, &base, &out).expect("convert + assemble");
    assert_eq!(assembled, out);

    // The converted transformer is a real file with its borrowed config.json.
    let tf = out.join("transformer");
    assert!(
        tf.join("diffusion_pytorch_model.safetensors").is_file(),
        "converted transformer safetensors written"
    );
    assert!(
        tf.join("config.json").is_file(),
        "transformer config.json copied"
    );

    // Borrowed components are symlinks into the base snapshot (no multi-GB duplication), and the
    // files the loader actually reads resolve through them.
    for sub in ["vae", "text_encoder", "tokenizer", "scheduler"] {
        let link = out.join(sub);
        assert!(
            std::fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink(),
            "{sub} is a symlink"
        );
        assert!(link.is_dir(), "{sub} symlink resolves to a dir");
    }
    assert!(
        out.join("model_index.json").is_file(),
        "model_index.json copied"
    );
    assert!(
        out.join("tokenizer/tokenizer.json").is_file(),
        "borrowed tokenizer.json resolves"
    );
    assert!(
        std::fs::read_dir(out.join("text_encoder"))
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("safetensors")),
        "borrowed text_encoder weights resolve"
    );

    // The converted transformer loads through the production loader (every diffusers key + the
    // loader's `to_out.0`/`timestep_embedder` remaps resolve) and forwards finite.
    let t = load_transformer(&out).expect("converted transformer loads via load_transformer");
    let (w, h) = (64u32, 64u32);
    let hidden = create_noise(0, w, h, 128).unwrap(); // [1,16,128]
    let key = mlx_rs::random::key(1).unwrap();
    let encoder =
        mlx_rs::random::normal::<f32>(&[1, 8, 12288][..], None, None, Some(&key)).unwrap();
    let img_ids = prepare_grid_ids((h / 16) as usize, (w / 16) as usize, 0);
    let txt_ids = prepare_text_ids(8);
    let v = t
        .forward(&hidden, &encoder, &img_ids, &txt_ids, 500.0)
        .unwrap();
    assert_eq!(v.shape(), &[1, 16, 128], "velocity shape");
    let total = v
        .as_dtype(Dtype::Float32)
        .unwrap()
        .sum(None)
        .unwrap()
        .item::<f32>();
    assert!(
        total.is_finite(),
        "converted transformer output non-finite: {total}"
    );

    println!(
        "flux2 true_v2 convert + assemble OK: loadable diffusers dir at {}",
        out.display()
    );
    let _ = std::fs::remove_dir_all(&out);
}
