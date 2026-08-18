//! sc-18758 conformance (candle twin of mlx-gen-ltx's
//! `tests/sc18758_ff_bias_keyframes_conformance.rs`): the LTX-2.3→2.5 DiT config delta
//! (`ff_bias`/`audio_ff_bias`/`use_keyframes_abs_pos_embedding`) changes exactly the tensor set the
//! reference build logic implies.
//!
//! **BLOCKED (credential, Michael only):** as on the MLX side, `SceneWorks/ltx-2.3-mlx`-style weight
//! repos are gated on Hugging Face and no HF token exists on this development machine, so the real
//! LTX-2.5 `transformer.safetensors` header (4349 tensor names) could not be fetched. This harness
//! instead proves the mutation directly against `AvDiT::new` (`crate::transformer`), whose key layout
//! is transcribed here from the actual loader source (`AvStream::load`, `AvBlock::load`,
//! `Attention::load_with_dims`, `FeedForward::load`).
//!
//! candle's `VarBuilder` (unlike mlx-gen's `Weights`) has no "was this key read" introspection, so the
//! "no extra" half of the proof takes a different, still-strict shape: a **missing**-tensor
//! differential. A map shaped like the real 2.5 checkpoint (bias tensors genuinely absent, the
//! keyframes marker present) must load under the 2.5 config and must **fail to load** under the 2.3
//! config (which still requires the bias) — proving the flag, not checkpoint content, decides
//! requiredness in both directions.

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;

use candle_gen_ltx::config::AvConfig;
use candle_gen_ltx::transformer::AvDiT;

const NUM_LAYERS: usize = 48;

/// A rank-1 placeholder (bias / norm weight / raw table — none of these are `dims2()`-checked at
/// load time, only at `forward`, which this harness never calls).
fn put(map: &mut HashMap<String, Tensor>, key: impl Into<String>, dev: &Device) {
    map.insert(key.into(), Tensor::zeros(1, DType::F32, dev).unwrap());
}

/// A rank-2 `[1,1]` placeholder for a `Linear`'s `.weight` — `quant::qlinear`'s dense fallback calls
/// `w.dims2()` at load time (unlike mlx-gen's `Weights`, which never shape-validates until forward),
/// so every Linear weight in this fixture must actually be rank 2.
fn put_weight_2d(map: &mut HashMap<String, Tensor>, key: impl Into<String>, dev: &Device) {
    map.insert(key.into(), Tensor::zeros((1, 1), DType::F32, dev).unwrap());
}

/// A `[5,1]` placeholder for a cross-modal `scale_shift_table_a2v_ca_{audio,video}` table —
/// `AvBlock::load`'s `split` closure `narrow(0, 0, 4)`/`narrow(0, 4, 1)`s it at load time (splitting
/// the 4-row scale-shift block from the 1-row gate), so it needs at least 5 rows.
fn put_cross_table(map: &mut HashMap<String, Tensor>, key: impl Into<String>, dev: &Device) {
    map.insert(key.into(), Tensor::zeros((5, 1), DType::F32, dev).unwrap());
}

/// `weight` (rank-2) + optional `bias` (rank-1) for one `Linear`-shaped leaf under `prefix`.
fn put_linear(map: &mut HashMap<String, Tensor>, prefix: &str, with_bias: bool, dev: &Device) {
    put_weight_2d(map, format!("{prefix}.weight"), dev);
    if with_bias {
        put(map, format!("{prefix}.bias"), dev);
    }
}

/// Every key one `AdaLayerNormSingle::load(vb)` reads (transcribed from
/// `crate::transformer::AdaLayerNormSingle::load`).
fn put_adaln(map: &mut HashMap<String, Tensor>, prefix: &str, dev: &Device) {
    put_linear(
        map,
        &format!("{prefix}.emb.timestep_embedder.linear_1"),
        true,
        dev,
    );
    put_linear(
        map,
        &format!("{prefix}.emb.timestep_embedder.linear_2"),
        true,
        dev,
    );
    put_linear(map, &format!("{prefix}.linear"), true, dev);
}

/// Every key one `Attention::load_with_dims(vb, ...)` reads (transcribed from
/// `crate::transformer::Attention::load_with_dims`): q/k/v/out (always biased — `attention_bias` is
/// reference-hardcoded `True`) + q/k RMSNorm + the gate.
fn put_attention(map: &mut HashMap<String, Tensor>, prefix: &str, dev: &Device) {
    for sub in ["to_q", "to_k", "to_v", "to_out.0", "to_gate_logits"] {
        put_linear(map, &format!("{prefix}.{sub}"), true, dev);
    }
    put(map, format!("{prefix}.q_norm.weight"), dev);
    put(map, format!("{prefix}.k_norm.weight"), dev);
}

/// `FeedForward::load(vb, bias)` — `with_bias` controls whether `net.0.proj.bias`/`net.2.bias` are
/// inserted at all (the real-checkpoint shape: 2.5 genuinely lacks the tensor, not merely unread).
fn put_ff(map: &mut HashMap<String, Tensor>, prefix: &str, with_bias: bool, dev: &Device) {
    put_linear(map, &format!("{prefix}.net.0.proj"), with_bias, dev);
    put_linear(map, &format!("{prefix}.net.2"), with_bias, dev);
}

/// The full `AvDiT::new` key set, `ff_bias:with_bias` shaped — a real 2.3 checkpoint (`true`) or a
/// real 2.5 checkpoint (`false`, plus `keyframes_abs_pos_embedding` when `with_keyframes`).
fn dit_weights(with_bias: bool, with_keyframes: bool, dev: &Device) -> HashMap<String, Tensor> {
    let mut m = HashMap::new();

    put_linear(&mut m, "patchify_proj", true, dev);
    put_adaln(&mut m, "adaln_single", dev);
    put_adaln(&mut m, "prompt_adaln_single", dev);
    put_adaln(&mut m, "av_ca_video_scale_shift_adaln_single", dev);
    put_adaln(&mut m, "av_ca_a2v_gate_adaln_single", dev);
    put(&mut m, "scale_shift_table", dev);
    put_linear(&mut m, "proj_out", true, dev);
    if with_keyframes {
        put(&mut m, "keyframes_abs_pos_embedding", dev);
    }

    put_linear(&mut m, "audio_patchify_proj", true, dev);
    put_adaln(&mut m, "audio_adaln_single", dev);
    put_adaln(&mut m, "audio_prompt_adaln_single", dev);
    put_adaln(&mut m, "av_ca_audio_scale_shift_adaln_single", dev);
    put_adaln(&mut m, "av_ca_v2a_gate_adaln_single", dev);
    put(&mut m, "audio_scale_shift_table", dev);
    put_linear(&mut m, "audio_proj_out", true, dev);

    for i in 0..NUM_LAYERS {
        let p = format!("transformer_blocks.{i}");
        put_attention(&mut m, &format!("{p}.attn1"), dev);
        put_attention(&mut m, &format!("{p}.attn2"), dev);
        put_ff(&mut m, &format!("{p}.ff"), with_bias, dev);
        put(&mut m, format!("{p}.scale_shift_table"), dev);
        put(&mut m, format!("{p}.prompt_scale_shift_table"), dev);
        put_attention(&mut m, &format!("{p}.audio_attn1"), dev);
        put_attention(&mut m, &format!("{p}.audio_attn2"), dev);
        put_ff(&mut m, &format!("{p}.audio_ff"), with_bias, dev);
        put(&mut m, format!("{p}.audio_scale_shift_table"), dev);
        put(&mut m, format!("{p}.audio_prompt_scale_shift_table"), dev);
        put_attention(&mut m, &format!("{p}.audio_to_video_attn"), dev);
        put_attention(&mut m, &format!("{p}.video_to_audio_attn"), dev);
        put_cross_table(&mut m, format!("{p}.scale_shift_table_a2v_ca_audio"), dev);
        put_cross_table(&mut m, format!("{p}.scale_shift_table_a2v_ca_video"), dev);
    }

    m
}

/// The mutation proof, direction 1: a checkpoint genuinely shaped like real LTX-2.5 (no FFN bias
/// tensors at all, `keyframes_abs_pos_embedding` present) loads under `AvConfig::ltx_2_5` — `ff_bias`
/// is read as `false` and the loader never demands the absent tensor.
#[test]
fn ltx25_config_loads_from_a_bias_free_checkpoint() {
    let dev = Device::Cpu;
    let m = dit_weights(false, true, &dev);
    let vb = VarBuilder::from_tensors(m, DType::F32, &dev);
    AvDiT::new(vb, &AvConfig::ltx_2_5())
        .expect("ltx_2_5 config must load a checkpoint with no FFN bias tensors");
}

/// The mutation proof, direction 2: the SAME bias-free checkpoint must **fail** to load under
/// `AvConfig::ltx_2_3` — `ff_bias:true` still requires the tensor, so the flag (not the checkpoint's
/// actual content) decides whether the bias is demanded. This is what makes it a mutation test, not an
/// assertion-of-a-default: flipping only the config, holding the checkpoint fixed, flips the outcome.
#[test]
fn ltx23_config_fails_on_a_bias_free_checkpoint() {
    let dev = Device::Cpu;
    let m = dit_weights(false, true, &dev);
    let vb = VarBuilder::from_tensors(m, DType::F32, &dev);
    let err = AvDiT::new(vb, &AvConfig::ltx_2_3());
    assert!(
        err.is_err(),
        "ltx_2_3 config (ff_bias:true) must error on a checkpoint with no `.bias` tensors, not \
         silently proceed"
    );
}

/// 2.3 conformance is unchanged: the real 2.3-shaped checkpoint (bias tensors present, no keyframes
/// marker) still loads cleanly under `AvConfig::ltx_2_3` — sc-18758 introduced no regression on the
/// pre-existing path.
#[test]
fn ltx23_config_loads_from_the_real_2_3_shaped_checkpoint() {
    let dev = Device::Cpu;
    let m = dit_weights(true, false, &dev);
    let vb = VarBuilder::from_tensors(m, DType::F32, &dev);
    AvDiT::new(vb, &AvConfig::ltx_2_3())
        .expect("ltx_2_3 config must still load its own (biased, no-keyframes) checkpoint shape");
}

/// The 2.5 config also loads a checkpoint that (redundantly) carries bias tensors — `ff_bias:false`
/// means the loader never reads `.bias`, whether or not the file happens to still have it; this is the
/// half of "no extra" candle's `VarBuilder` can express (it cannot report *unread* keys the way
/// mlx-gen's `Weights::unused_keys` does, but not-reading is still directly observable: the load
/// succeeds identically whether or not the unread tensor is present).
#[test]
fn ltx25_config_tolerates_but_does_not_require_bias_tensors_present() {
    let dev = Device::Cpu;
    let m = dit_weights(true, true, &dev);
    let vb = VarBuilder::from_tensors(m, DType::F32, &dev);
    AvDiT::new(vb, &AvConfig::ltx_2_5())
        .expect("ltx_2_5 config must still load even if the map happens to carry bias tensors");
}
