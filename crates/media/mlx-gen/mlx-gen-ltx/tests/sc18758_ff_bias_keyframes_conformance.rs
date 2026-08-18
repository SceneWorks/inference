//! sc-18758 conformance: the LTX-2.3→2.5 DiT config delta (`ff_bias`/`audio_ff_bias`/
//! `use_keyframes_abs_pos_embedding`) changes **exactly** the tensor set the reference build logic
//! implies — no missing, no extra — and 2.3 conformance is untouched.
//!
//! **BLOCKED (credential, Michael only):** the acceptance criterion asks for a conformance check
//! against the *real* LTX-2.5 checkpoint's header (4349 tensor names). `SceneWorks/ltx-2.3-mlx`-style
//! weight repos are gated on Hugging Face and no HF token exists on this development machine, so the
//! real 2.5 `transformer.safetensors` header could not be fetched or inspected. Per the story's own
//! fallback plan, this harness instead:
//!   1. Derives a synthetic tensor-name set directly from the **actual loader source** in
//!      `crate::transformer` (`AvDiT::from_weights`, `AvBlock::load`, `Attention::load`,
//!      `FeedForward::load`, `AdaLayerNormSingle::load`) — not invented, transcribed from the
//!      `format!()` key patterns those functions read.
//!   2. Builds one placeholder-valued superset weights map (48 real layers, every optional leaf —
//!      attention gates, both FFN biases, the keyframes marker — present) so the *same* map can be
//!      fed to both the 2.3 and 2.5 configs.
//!   3. Uses [`mlx_gen::weights::Weights::unused_keys`] after each construction to observe exactly
//!      which keys each config's loader actually touched, and asserts the touched-set **delta**
//!      between 2.3 and 2.5 is precisely the measured two-key config change: 2.3 reads 4 bias tensors
//!      per block (`ff.proj_{in,out}.bias`, `audio_ff.proj_{in,out}.bias`) that 2.5 does not (192
//!      tensors over 48 layers), and 2.5 additionally reads `keyframes_abs_pos_embedding`, which 2.3
//!      does not.
//!
//! This proves the flag-driven parameter set is exactly right relative to the loader's own behavior,
//! independent of the real header. Someone with the gated-repo credential should re-run this class of
//! check against a real header dump (`safetensors` `__metadata__`/key list) to confirm the absolute
//! 4349 count; see the module docs above for what's still open.

use std::collections::HashMap;

use mlx_gen::weights::Weights;
use mlx_rs::Array;

use mlx_gen_ltx::config::LtxConfig;
use mlx_gen_ltx::transformer::{AvDiT, Precision};

const NUM_LAYERS: i32 = 48;

/// The 2.3 config the epic orientation cites as unaffected — [`LtxConfig::video_only_defaults`] is
/// already dimensioned as the shipped AV 2.3 checkpoint (32 heads × 128 head-dim video, 32×64 audio,
/// gated family, 48 layers); only `ff_bias`/`audio_ff_bias`/`use_keyframes_abs_pos_embedding` are
/// touched here, matching `LtxConfig::from_embedded_transformer` parsing an `embedded_config.json`
/// that omits both delta keys.
fn ltx23_cfg() -> LtxConfig {
    let mut cfg = LtxConfig::video_only_defaults();
    cfg.apply_gated_attention = true;
    cfg.adaln_embedding_coefficient = 9;
    cfg.cross_attention_adaln = true;
    cfg.num_layers = NUM_LAYERS;
    cfg
}

/// The measured 2.3→2.5 delta applied to the same base.
fn ltx25_cfg() -> LtxConfig {
    let mut cfg = ltx23_cfg();
    cfg.ff_bias = false;
    cfg.audio_ff_bias = false;
    cfg.use_keyframes_abs_pos_embedding = true;
    cfg
}

/// A `[1]` f32 placeholder — `AvDiT::from_weights` never shape-validates at load time (only a later
/// `forward` would), so every leaf can share this one dummy value; only tensor **names** matter here.
fn leaf() -> Array {
    Array::from_slice(&[0.0f32], &[1])
}

/// Every key one `AdaLayerNormSingle::load(w, prefix)` reads (transcribed from
/// `crate::transformer::AdaLayerNormSingle::load`): two timestep-embedder Linears + the output Linear.
fn insert_adaln(m: &mut HashMap<String, Array>, prefix: &str) {
    for leaf_key in [
        "emb.timestep_embedder.linear1.weight",
        "emb.timestep_embedder.linear1.bias",
        "emb.timestep_embedder.linear2.weight",
        "emb.timestep_embedder.linear2.bias",
        "linear.weight",
        "linear.bias",
    ] {
        m.insert(format!("{prefix}.{leaf_key}"), leaf());
    }
}

/// Every key one `Attention::load(w, prefix, ...)` reads (transcribed from
/// `crate::transformer::Attention::load`): q/k/v/out Linears (always biased — `attention_bias` is
/// reference-hardcoded `True`, independent of `ff_bias`), q/k RMSNorm weights, and the optional gate
/// Linear (included here — the superset is shared by both configs, so its presence/absence in the
/// real checkpoint cancels out of the 2.3-vs-2.5 delta this test asserts).
fn insert_attention(m: &mut HashMap<String, Array>, prefix: &str) {
    for sub in ["to_q", "to_k", "to_v", "to_out", "to_gate_logits"] {
        m.insert(format!("{prefix}.{sub}.weight"), leaf());
        m.insert(format!("{prefix}.{sub}.bias"), leaf());
    }
    m.insert(format!("{prefix}.q_norm.weight"), leaf());
    m.insert(format!("{prefix}.k_norm.weight"), leaf());
}

/// Every key one `FeedForward::load(w, prefix, prec, bias)` **could** read — always inserts the bias
/// tensors (the superset), so a config that sets `bias:false` demonstrably leaves them unaccessed
/// rather than merely being handed a map that lacks them.
fn insert_ff(m: &mut HashMap<String, Array>, prefix: &str) {
    m.insert(format!("{prefix}.proj_in.weight"), leaf());
    m.insert(format!("{prefix}.proj_in.bias"), leaf());
    m.insert(format!("{prefix}.proj_out.weight"), leaf());
    m.insert(format!("{prefix}.proj_out.bias"), leaf());
}

/// The full transformer.safetensors superset: every key `AvDiT::from_weights` could possibly read for
/// `NUM_LAYERS` blocks, including `keyframes_abs_pos_embedding` (present regardless of config — the
/// test proves *access*, not presence, differs between 2.3 and 2.5).
fn superset_weights() -> Weights {
    let mut m = HashMap::new();

    // Video stream globals.
    m.insert("patchify_proj.weight".into(), leaf());
    m.insert("patchify_proj.bias".into(), leaf());
    insert_adaln(&mut m, "adaln_single");
    insert_adaln(&mut m, "prompt_adaln_single");
    insert_adaln(&mut m, "av_ca_video_scale_shift_adaln_single");
    insert_adaln(&mut m, "av_ca_a2v_gate_adaln_single");
    m.insert("scale_shift_table".into(), leaf());
    m.insert("proj_out.weight".into(), leaf());
    m.insert("proj_out.bias".into(), leaf());
    m.insert("keyframes_abs_pos_embedding".into(), leaf());

    // Audio stream globals.
    m.insert("audio_patchify_proj.weight".into(), leaf());
    m.insert("audio_patchify_proj.bias".into(), leaf());
    insert_adaln(&mut m, "audio_adaln_single");
    insert_adaln(&mut m, "audio_prompt_adaln_single");
    insert_adaln(&mut m, "av_ca_audio_scale_shift_adaln_single");
    insert_adaln(&mut m, "av_ca_v2a_gate_adaln_single");
    m.insert("audio_scale_shift_table".into(), leaf());
    m.insert("audio_proj_out.weight".into(), leaf());
    m.insert("audio_proj_out.bias".into(), leaf());

    // Per-block (transformer_blocks.{i}.*), transcribed from `AvBlock::load`.
    for i in 0..NUM_LAYERS {
        let p = format!("transformer_blocks.{i}");
        insert_attention(&mut m, &format!("{p}.attn1"));
        insert_attention(&mut m, &format!("{p}.attn2"));
        insert_ff(&mut m, &format!("{p}.ff"));
        m.insert(format!("{p}.scale_shift_table"), leaf());
        m.insert(format!("{p}.prompt_scale_shift_table"), leaf());
        insert_attention(&mut m, &format!("{p}.audio_attn1"));
        insert_attention(&mut m, &format!("{p}.audio_attn2"));
        insert_ff(&mut m, &format!("{p}.audio_ff"));
        m.insert(format!("{p}.audio_scale_shift_table"), leaf());
        m.insert(format!("{p}.audio_prompt_scale_shift_table"), leaf());
        insert_attention(&mut m, &format!("{p}.audio_to_video_attn"));
        insert_attention(&mut m, &format!("{p}.video_to_audio_attn"));
        m.insert(format!("{p}.scale_shift_table_a2v_ca_audio"), leaf());
        m.insert(format!("{p}.scale_shift_table_a2v_ca_video"), leaf());
    }

    Weights::from_map(m)
}

/// The exact set of `ff.proj_{in,out}.bias` / `audio_ff.proj_{in,out}.bias` keys across all
/// `NUM_LAYERS` blocks — what 2.3 reads that 2.5 must not (192 tensors at `NUM_LAYERS=48`).
fn expected_ff_bias_keys() -> std::collections::HashSet<String> {
    let mut s = std::collections::HashSet::new();
    for i in 0..NUM_LAYERS {
        let p = format!("transformer_blocks.{i}");
        for ff_prefix in ["ff", "audio_ff"] {
            s.insert(format!("{p}.{ff_prefix}.proj_in.bias"));
            s.insert(format!("{p}.{ff_prefix}.proj_out.bias"));
        }
    }
    s
}

/// Zero missing, zero extra: both the 2.3 and the 2.5 config load cleanly from the shared superset
/// (a real checkpoint has no extra keys to begin with; here that direction is covered by
/// `ff_bias_false_never_requires_an_absent_bias_tensor` in `transformer.rs`'s unit tests, which builds
/// a map that genuinely lacks the bias tensors).
#[test]
fn ltx25_and_ltx23_both_load_from_the_documented_key_set() {
    let prec = Precision::dense_f32(4, 64);
    let w = superset_weights();

    AvDiT::from_weights(&w, &ltx23_cfg(), prec).expect("2.3 config loads from the documented set");
    AvDiT::from_weights(&w, &ltx25_cfg(), prec).expect("2.5 config loads from the documented set");
}

/// The core conformance/mutation proof: the accessed-tensor delta between the 2.3 and 2.5
/// constructions is **exactly** the measured config delta — no more, no less. This is what "no
/// missing, no extra" reduces to without a real header: the loader's *own* read pattern, observed via
/// [`Weights::unused_keys`], must match the two-key delta bit-for-bit.
#[test]
fn ff_bias_and_keyframes_change_exactly_the_measured_tensor_set() {
    let prec = Precision::dense_f32(4, 64);
    let w23 = superset_weights();
    let w25 = superset_weights();

    AvDiT::from_weights(&w23, &ltx23_cfg(), prec).unwrap();
    AvDiT::from_weights(&w25, &ltx25_cfg(), prec).unwrap();

    let unused23: std::collections::HashSet<String> =
        w23.unused_keys().into_iter().map(str::to_string).collect();
    let unused25: std::collections::HashSet<String> =
        w25.unused_keys().into_iter().map(str::to_string).collect();

    // 2.3 leaves `keyframes_abs_pos_embedding` unread; 2.5 does not.
    assert!(unused23.contains("keyframes_abs_pos_embedding"));
    assert!(!unused25.contains("keyframes_abs_pos_embedding"));

    // 2.5 leaves every `ff`/`audio_ff` bias tensor unread; 2.3 does not.
    let ff_bias_keys = expected_ff_bias_keys();
    assert_eq!(ff_bias_keys.len(), (NUM_LAYERS as usize) * 4);
    for key in &ff_bias_keys {
        assert!(
            !unused23.contains(key),
            "2.3 must read {key} (ff_bias defaults true)"
        );
        assert!(
            unused25.contains(key),
            "2.5 must NOT read {key} (ff_bias:false — the tensor doesn't exist on a real checkpoint)"
        );
    }

    // The delta is EXACTLY these two effects — nothing else changed between the two loads.
    let extra_unused_in_25: std::collections::HashSet<_> =
        unused25.difference(&unused23).cloned().collect();
    assert_eq!(
        extra_unused_in_25, ff_bias_keys,
        "2.5 vs 2.3 must leave exactly the ff/audio_ff bias tensors unread, nothing more"
    );
    let mut extra_unused_in_23: std::collections::HashSet<_> =
        unused23.difference(&unused25).cloned().collect();
    assert!(extra_unused_in_23.remove("keyframes_abs_pos_embedding"));
    assert!(
        extra_unused_in_23.is_empty(),
        "2.3 vs 2.5 must leave exactly `keyframes_abs_pos_embedding` unread, nothing more; got {extra_unused_in_23:?}"
    );
}

/// 2.3 conformance is unchanged (F-047-style guard): a config with neither delta key set (the
/// `LtxConfig::video_only_defaults` / `from_embedded_transformer` fallback) still reads every FFN bias
/// and never touches `keyframes_abs_pos_embedding` — i.e. sc-18758 introduced no regression on the
/// pre-existing 2.3 path.
#[test]
fn ltx23_conformance_reads_every_ff_bias_and_no_keyframes_marker() {
    let prec = Precision::dense_f32(4, 64);
    let w = superset_weights();
    AvDiT::from_weights(&w, &ltx23_cfg(), prec).unwrap();

    let unused: std::collections::HashSet<String> =
        w.unused_keys().into_iter().map(str::to_string).collect();
    for key in expected_ff_bias_keys() {
        assert!(!unused.contains(&key), "2.3 must still read {key}");
    }
    assert!(unused.contains("keyframes_abs_pos_embedding"));
}
