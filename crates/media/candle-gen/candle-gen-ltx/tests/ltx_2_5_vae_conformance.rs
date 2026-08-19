//! sc-18765 — **the candle LTX VAE / audio-VAE / vocoder ports consume LTX-2.5's component
//! checkpoints unchanged.**
//!
//! The MLX ports need a conversion step for LTX-2.5 (their weights are channels-last, upstream's are
//! PyTorch). The candle ports need **none**: they already read PyTorch-layout `[O,I,kt,kh,kw]` convs
//! and the `-of-means` statistic names, so the only difference between an LTX-2.3 all-in-one
//! checkpoint and an LTX-2.5 video-VAE component file is *where the VarBuilder is rooted* — `vae.`
//! for 2.3, the file root for 2.5. The audio VAE and vocoder do not differ even in that: LTX-2.5
//! keys them `audio_vae.*` / `vocoder.*` exactly as 2.3 does.
//!
//! That is a claim about key names, and it is asserted here by **running the shipped loaders**
//! against a checkpoint carrying LTX-2.5's key set (recorded in
//! `docs/reference/sc-18765-vae-keysets/ltx-vae-keysets.json`). Tensor *contents* are irrelevant to
//! a key-name claim, so the stand-in checkpoint stores rank-preserving 2-per-axis dummies — the
//! loaders infer channel counts from the weights they are handed, so they exercise exactly the same
//! `get_unchecked` / `contains_tensor` calls at a millionth of the bytes. Real weights and real
//! numerics are `ltx_2_5_vae_real_weights.rs`.
//!
//! No CUDA, no GPU, no weights — this runs in ordinary CI on every platform.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen_ltx::audio_vae::AudioDecoder;
use candle_gen_ltx::config::{AudioVaeConfig, VocoderConfig, LATENT_CHANNELS};
use candle_gen_ltx::vae::LtxVideoVae;
use candle_gen_ltx::vocoder::LtxVocoder;

const KEYSETS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../../docs/reference/sc-18765-vae-keysets/ltx-vae-keysets.json"
);

/// The per-channel statistics are `reshape`d to `[1, LATENT_CHANNELS, 1, 1, 1]` at load, so they are
/// the one pair of tensors whose element count has to be real.
const STATS_SUFFIXES: [&str; 2] = ["mean-of-means", "std-of-means"];

fn keyset(component: &str) -> Vec<String> {
    let text = std::fs::read_to_string(KEYSETS).unwrap_or_else(|e| {
        panic!("read {KEYSETS}: {e} (regenerate with tools/dump_ltx_vae_keysets.py)")
    });
    let doc: serde_json::Value = serde_json::from_str(&text).expect("parse ltx-vae-keysets.json");
    doc["components"][component]["tensors"]
        .as_object()
        .unwrap_or_else(|| panic!("fixture component {component} has no tensors"))
        .keys()
        .cloned()
        .collect()
}

/// Write a checkpoint carrying `keys` (optionally re-namespaced under `prefix`) with rank-preserving
/// dummy tensors.
fn checkpoint(dir: &Path, file: &str, keys: &[String], prefix: &str) -> PathBuf {
    let mut tensors: BTreeMap<String, Tensor> = BTreeMap::new();
    for key in keys {
        // Every conv weight is `[O, I, ...]` and its bias is `[O]`, so a uniform 2-per-axis dummy
        // keeps the two consistent for the `reshape((1, out_c, 1, 1, 1))` the loaders do.
        let tensor = if STATS_SUFFIXES.iter().any(|s| key.ends_with(s)) {
            Tensor::zeros(LATENT_CHANNELS, DType::F32, &Device::Cpu)
        } else {
            let rank = 1.max(dummy_rank(key));
            Tensor::zeros(vec![2usize; rank], DType::F32, &Device::Cpu)
        }
        .expect("dummy tensor");
        tensors.insert(format!("{prefix}{key}"), tensor);
    }
    let path = dir.join(file);
    candle_gen::candle_core::safetensors::save(
        &tensors
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>(),
        &path,
    )
    .expect("write dummy checkpoint");
    path
}

/// The rank recorded for `key` in the fixture. Ranks are what the loaders read (channel counts and
/// kernel sizes come out of `dims()`), so they are the one shape property worth preserving.
fn dummy_rank(key: &str) -> usize {
    let text = std::fs::read_to_string(KEYSETS).expect("keysets");
    let doc: serde_json::Value = serde_json::from_str(&text).expect("parse");
    for component in doc["components"].as_object().expect("components").values() {
        if let Some(shape) = component["tensors"].get(key) {
            return shape.as_array().expect("shape").len();
        }
    }
    panic!("no recorded shape for {key}");
}

fn vb(path: &Path) -> VarBuilder<'static> {
    unsafe {
        VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &Device::Cpu)
            .expect("mmap dummy checkpoint")
    }
}

#[test]
fn video_vae_loads_straight_off_a_2_5_component_checkpoint_but_not_a_2_3_namespaced_one() {
    let tmp = tempfile::tempdir().unwrap();
    let keys = keyset("ltx_2_5_video_vae_conv");
    assert_eq!(keys.len(), 170, "the 2.5 conv VAE is a 170-tensor file");

    // LTX-2.5 shape: tensors at the file root. The shipped loader must find every one of them.
    let component = checkpoint(tmp.path(), "ltx25.safetensors", &keys, "");
    let builder = vb(&component);
    LtxVideoVae::new_with_encoder(builder.clone(), builder, LATENT_CHANNELS, 4).unwrap_or_else(
        |e| panic!("the candle LTX VAE port must load an LTX-2.5 conv VAE component verbatim: {e}"),
    );

    // The executed control: the SAME key set under LTX-2.3's `vae.` namespace must NOT load from the
    // root. Without it, a loader that ignored its VarBuilder prefix entirely would pass the
    // assertion above — and the one real structural difference between 2.3 and 2.5 would go
    // unrecorded.
    let bundled = checkpoint(tmp.path(), "ltx23.safetensors", &keys, "vae.");
    let builder = vb(&bundled);
    let err = match LtxVideoVae::new_with_encoder(builder.clone(), builder, LATENT_CHANNELS, 4) {
        Err(e) => format!("{e}"),
        Ok(_) => panic!(
            "a `vae.`-namespaced checkpoint must not load from the root — the 2.3/2.5 namespace \
             difference is real and must stay visible"
        ),
    };
    assert!(
        err.contains("decoder.up_blocks") || err.contains("cannot find tensor"),
        "the failure should name the missing tensor, got: {err}"
    );

    // …and it loads when the builder is rooted at `vae.`, which is the 2.3 path this crate ships.
    let builder = vb(&bundled).pp("vae");
    LtxVideoVae::new_with_encoder(builder.clone(), builder, LATENT_CHANNELS, 4)
        .expect("the 2.3 `vae.`-rooted path must keep working");
}

#[test]
fn audio_vae_and_vocoder_load_off_a_2_5_component_checkpoint_with_no_renaming_at_all() {
    let tmp = tempfile::tempdir().unwrap();
    let keys = keyset("ltx_2_5_audio_vae");
    assert_eq!(keys.len(), 1329, "the 2.5 audio VAE is a 1329-tensor file");
    let component = checkpoint(tmp.path(), "ltx25-audio.safetensors", &keys, "");

    // Exactly the roots `Pipeline::load_components` uses for a 2.3 all-in-one checkpoint: the audio
    // VAE under `audio_vae`, the vocoder at the checkpoint root (it carries its own `vocoder.`).
    let builder = vb(&component);
    AudioDecoder::load(&builder.pp("audio_vae"), &AudioVaeConfig::ltx_2_3())
        .expect("the candle audio VAE port must load an LTX-2.5 audio component verbatim");
    let vocoder = LtxVocoder::load(vb(&component), &Device::Cpu, &VocoderConfig::ltx_2_3())
        .expect("the candle vocoder port must load an LTX-2.5 audio component verbatim");
    assert!(
        matches!(vocoder, LtxVocoder::Bwe(_)),
        "LTX-2.5 still ships the bandwidth-extension stage"
    );
}
