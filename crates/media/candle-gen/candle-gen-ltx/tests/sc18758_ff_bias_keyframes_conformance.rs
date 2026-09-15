//! sc-18758 conformance (candle twin of mlx-gen-ltx's
//! `tests/sc18758_ff_bias_keyframes_conformance.rs`): the LTX-2.3→2.5 DiT config delta
//! (`ff_bias`/`use_keyframes_abs_pos_embedding`) changes exactly the tensor set the real shipped
//! checkpoint has.
//!
//! `tests/fixtures/ltx25_distilled_dit_tensors.json` is the same real LTX-2.5 distilled
//! `transformer.safetensors` header used by the mlx-gen-ltx twin (name + shape per tensor,
//! `model.diffusion_model.` prefix stripped, `*_embeddings_connector.*` dropped — 4091 DiT-only
//! tensors of the real 4349; see that file's module doc for the capture method and the real-header
//! measurement, including the correction that `audio_ff_bias` is **not** part of the delta).
//!
//! Unlike mlx-gen-ltx (whose `transformer.safetensors` is produced by `convert.rs::sanitize_transformer`,
//! which renames `to_out.0`→`to_out` etc), candle-gen-ltx's `AvDiT` reads the checkpoint's *raw*
//! reference naming directly (`to_out.0`, `linear_1`/`linear_2`, `ff.net.0.proj`/`ff.net.2` — see
//! `crate::transformer::Attention::load_with_dims` / `AdaLayerNormSingle::load` / `FeedForward::load`),
//! so this fixture needs no renaming, only the prefix strip already applied.
//!
//! candle's `VarBuilder` (unlike mlx-gen's `Weights`) has no "was this key read" introspection, so the
//! "no extra" half of the proof takes a missing-tensor-differential shape, as in the original harness:
//! a config that requires a tensor the map lacks must fail to load; a config that does not require it
//! must load regardless of whether the map happens to carry it.

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;

use candle_gen_ltx::config::AvConfig;
use candle_gen_ltx::transformer::AvDiT;

const FIXTURE_JSON: &str = include_str!("fixtures/ltx25_distilled_dit_tensors.json");

/// DiT depth the fixture header was captured at.
pub const SHARED_FIXTURE_NUM_LAYERS: usize = 48;
/// DiT-only tensor count the fixture carries (4349 real minus the 258 embeddings-connector keys).
pub const SHARED_FIXTURE_DIT_TENSOR_COUNT: usize = 4091;
/// Video FFN up-projection bias width, from the real header's `[16384, 4096]` weight out-dim.
pub const SHARED_FIXTURE_FF_UP_BIAS_DIM: usize = 16384;
/// Video FFN down-projection bias width, from the real header's `[4096, 16384]` weight out-dim.
pub const SHARED_FIXTURE_FF_DOWN_BIAS_DIM: usize = 4096;

struct RealTensor {
    name: String,
    shape: Vec<usize>,
}

/// Parse the `[{"name": "...", "shape": [..]}, ...]` fixture without a `serde::Deserialize` derive
/// (the crate depends on `serde_json` but not `serde` directly) — plain `serde_json::Value` walking.
fn real_2_5_tensors() -> Vec<RealTensor> {
    let v: serde_json::Value = serde_json::from_str(FIXTURE_JSON).expect("fixture is valid JSON");
    v.as_array()
        .expect("fixture is a JSON array")
        .iter()
        .map(|entry| {
            let name = entry["name"]
                .as_str()
                .expect("entry.name is a string")
                .to_string();
            let shape = entry["shape"]
                .as_array()
                .expect("entry.shape is a JSON array")
                .iter()
                .map(|n| n.as_i64().expect("shape entry is an integer") as usize)
                .collect();
            RealTensor { name, shape }
        })
        .collect()
}

/// Shrink large feature dims to `1` to keep the fixture's real (up to `[16384, 4096]`) shapes from
/// allocating hundreds of MB per tensor across 4091 tensors — `AvDiT::new` never validates a specific
/// dim *value* at load, only rank (`dims2()` on every `Linear` weight) and, for the cross-modal
/// `scale_shift_table_a2v_ca_{audio,video}` tables, that `dim0 ≥ 5` (`AvBlock::load`'s `narrow(0,0,4)`/
/// `narrow(0,4,1)` split). Preserving any dim ≤ 6 keeps those real `2`/`5`-row tables exact while
/// everything else collapses to a handful of elements.
fn shrink_shape(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![1];
    }
    shape
        .iter()
        .map(|&d| if d > 6 { 1 } else { d.max(1) })
        .collect()
}

fn placeholder(shape: &[usize], dev: &Device) -> Tensor {
    let shape = shrink_shape(shape);
    Tensor::zeros(shape, DType::F32, dev).unwrap()
}

/// The real LTX-2.5 DiT weights map (candle naming = raw reference naming, so only the prefix strip
/// already baked into the fixture applies — no rename needed).
fn real_25_weights(dev: &Device) -> HashMap<String, Tensor> {
    real_2_5_tensors()
        .into_iter()
        .map(|t| (t.name.clone(), placeholder(&t.shape, dev)))
        .collect()
}

/// A reconstructed "real 2.3-shaped" map: the real 2.5 set with the two confirmed deltas reversed —
/// `keyframes_abs_pos_embedding` removed, and the 96 video `ff.net.{0.proj,2}.bias` tensors added back
/// (shapes from the real header's `ff.net.0.proj.weight`/`ff.net.2.weight` out-dims: `[16384, 4096]`
/// ⇒ bias `[16384]`; `[4096, 16384]` ⇒ bias `[4096]`, shrunk the same way as every other tensor here).
fn reconstructed_23_weights(dev: &Device) -> HashMap<String, Tensor> {
    let mut m: HashMap<String, Tensor> = real_2_5_tensors()
        .into_iter()
        .filter(|t| t.name != "keyframes_abs_pos_embedding")
        .map(|t| (t.name.clone(), placeholder(&t.shape, dev)))
        .collect();
    for i in 0..SHARED_FIXTURE_NUM_LAYERS {
        let p = format!("transformer_blocks.{i}");
        m.insert(
            format!("{p}.ff.net.0.proj.bias"),
            placeholder(&[SHARED_FIXTURE_FF_UP_BIAS_DIM], dev),
        );
        m.insert(
            format!("{p}.ff.net.2.bias"),
            placeholder(&[SHARED_FIXTURE_FF_DOWN_BIAS_DIM], dev),
        );
    }
    m
}

/// The core proof: the 2.5 config loads the **real** LTX-2.5 tensor set (candle naming, no rename
/// needed) with no missing tensor.
#[test]
fn ltx25_config_loads_the_real_header() {
    let dev = Device::Cpu;
    let m = real_25_weights(&dev);
    assert_eq!(
        m.len(),
        SHARED_FIXTURE_DIT_TENSOR_COUNT,
        "fixture must be the real {SHARED_FIXTURE_DIT_TENSOR_COUNT}-tensor DiT-only set"
    );
    let vb = VarBuilder::from_tensors(m, DType::F32, &dev);
    AvDiT::new(vb, &AvConfig::ltx_2_5()).expect("ltx_2_5 config must load the real header");
}

/// The mutation proof: the SAME real 2.5 tensor set must **fail** to load under the 2.3 config
/// (`ff_bias:true` demands the video FFN bias the real 2.5 checkpoint does not carry) — the flag, not
/// checkpoint content, decides requiredness.
#[test]
fn ltx23_config_fails_on_the_real_2_5_header() {
    let dev = Device::Cpu;
    let m = real_25_weights(&dev);
    let vb = VarBuilder::from_tensors(m, DType::F32, &dev);
    let err = AvDiT::new(vb, &AvConfig::ltx_2_3());
    assert!(
        err.is_err(),
        "ltx_2_3 config (ff_bias:true) must error on the real LTX-2.5 header, which carries no \
         video ff.net.{{0.proj,2}}.bias"
    );
}

/// 2.3 conformance is unchanged: the reconstructed real-2.3-shaped set (real 2.5 data with the two
/// measured deltas reversed) loads cleanly under the 2.3 config.
#[test]
fn ltx23_config_loads_the_reconstructed_2_3_header() {
    let dev = Device::Cpu;
    let m = reconstructed_23_weights(&dev);
    let vb = VarBuilder::from_tensors(m, DType::F32, &dev);
    AvDiT::new(vb, &AvConfig::ltx_2_3())
        .expect("ltx_2_3 config must load the reconstructed 2.3-shaped header");
}

/// The 2.5 config also loads a checkpoint that (redundantly) carries the video FFN bias tensors on
/// top of everything 2.5 needs — `ff_bias:false` means the loader never reads `.bias`, whether or not
/// the file happens to still have it. This is the half of "no extra" candle's `VarBuilder` can express
/// (it cannot report *unread* keys the way mlx-gen's `Weights::unused_keys` does, but not-reading is
/// still directly observable: the load succeeds identically whether or not the unread tensor is
/// present).
#[test]
fn ltx25_config_tolerates_but_does_not_require_video_ff_bias_present() {
    let dev = Device::Cpu;
    // The real 2.5 set (has `keyframes_abs_pos_embedding`) PLUS the video ff bias tensors it does not
    // ship — a strict superset of what `ltx_2_5` needs.
    let mut m = real_25_weights(&dev);
    for i in 0..SHARED_FIXTURE_NUM_LAYERS {
        let p = format!("transformer_blocks.{i}");
        m.insert(
            format!("{p}.ff.net.0.proj.bias"),
            placeholder(&[SHARED_FIXTURE_FF_UP_BIAS_DIM], &dev),
        );
        m.insert(
            format!("{p}.ff.net.2.bias"),
            placeholder(&[SHARED_FIXTURE_FF_DOWN_BIAS_DIM], &dev),
        );
    }
    let vb = VarBuilder::from_tensors(m, DType::F32, &dev);
    AvDiT::new(vb, &AvConfig::ltx_2_5())
        .expect("ltx_2_5 config must still load even if the map happens to carry video ff bias");
}
