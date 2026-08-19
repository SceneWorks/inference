//! sc-18758 conformance: the LTX-2.3→2.5 DiT config delta (`ff_bias`/`use_keyframes_abs_pos_embedding`)
//! changes **exactly** the tensor set the real shipped checkpoint has — no missing, no extra.
//!
//! `tests/fixtures/ltx25_distilled_dit_tensors.json` is the **real** LTX-2.5 distilled
//! `transformer.safetensors` header (name + shape per tensor), captured locally via a header-only
//! read (the 8-byte length prefix + JSON header — zero weight bytes) of the distilled transformer
//! snapshot on the models volume cache, with the `model.diffusion_model.` prefix stripped and the
//! `*_embeddings_connector.*` tensors dropped (this story's subject is the DiT, not the connector;
//! connector.safetensors is a separate file mlx-gen-ltx loads independently). The
//! real header has **4349** total tensors, 258 of them the two embeddings connectors, 4091 DiT-only —
//! matching the acceptance criterion's cited count. The LTX-2.5 **dev** header
//! (`ltx-2.5-22b-dev-transformer-bf16.safetensors`) was cross-checked and is byte-identical in shape/
//! name (both distilled and dev share the same transformer architecture); the fixture captures either.
//!
//! **Measured from the real header (not assumed):** `__metadata__.config.transformer` carries
//! `ff_bias: false` and `use_keyframes_abs_pos_embedding: true`, and has **no** `audio_ff_bias` key at
//! all — so `audio_ff_bias` takes the reference's absent-key default (`True`), same as 2.3. Concretely:
//! zero `transformer_blocks.*.ff.net.{0.proj,2}.bias` (video FFN, removed) but all 96
//! `transformer_blocks.*.audio_ff.net.{0.proj,2}.bias` (audio FFN, present/unchanged across 48
//! blocks), plus one `keyframes_abs_pos_embedding` (added). The measured 2.3→2.5 tensor-set delta is
//! therefore exactly **96 removed + 1 added**, not "192 removed + 1 added" (an earlier draft of this
//! harness incorrectly folded `audio_ff_bias` into the delta before reading the real header).
//!
//! The real header's tensor names are the *reference* naming (`to_out.0`, `linear_1`/`linear_2`,
//! `ff.net.0.proj`/`ff.net.2`) — mlx-gen-ltx's own `transformer.safetensors` is produced by
//! [`mlx_gen_ltx::convert::LtxConvertOpts`]'s `sanitize_transformer`, which additionally renames
//! `.to_out.0.`→`.to_out.`, `.ff.net.0.proj.`→`.ff.proj_in.`, `.ff.net.2.`→`.ff.proj_out.` (+ audio),
//! and `.linear_1.`/`.linear_2.`→`.linear1.`/`.linear2.`. This harness applies that exact rename (kept
//! in lockstep with `convert.rs::sanitize_transformer` — see [`sanitize`]) before building the
//! [`Weights`] map, so `AvDiT::from_weights` sees precisely what a converted checkpoint would.

use std::collections::HashSet;

use mlx_gen::weights::Weights;
use mlx_rs::Array;

use mlx_gen_ltx::config::LtxConfig;
use mlx_gen_ltx::transformer::{AvDiT, Precision};

const FIXTURE_JSON: &str = include_str!("fixtures/ltx25_distilled_dit_tensors.json");
const NUM_LAYERS: i32 = 48;

struct RealTensor {
    name: String,
    shape: Vec<i32>,
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
                .map(|n| n.as_i64().expect("shape entry is an integer") as i32)
                .collect();
            RealTensor { name, shape }
        })
        .collect()
}

/// The exact rename `convert.rs::sanitize_transformer` applies to a `model.diffusion_model.`-stripped
/// reference key, converting it to mlx-gen-ltx's internal `transformer.safetensors` naming. Kept
/// byte-for-byte identical to that function so this harness tests the real converted key set, not an
/// invented one.
fn sanitize(name: &str) -> String {
    name.replace(".to_out.0.", ".to_out.")
        .replace(".ff.net.0.proj.", ".ff.proj_in.")
        .replace(".ff.net.2.", ".ff.proj_out.")
        .replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.")
        .replace(".audio_ff.net.2.", ".audio_ff.proj_out.")
        .replace(".linear_1.", ".linear1.")
        .replace(".linear_2.", ".linear2.")
}

/// Real per-tensor shapes range up to `[16384, 4096]` (~268 MB as f32 placeholders) — allocating 4091
/// of those would blow well past the "lightweight synthetic tensors" resource lane for a mere
/// name-conformance check. `AvDiT::from_weights` never shape-validates at load time, so the *value* of
/// a large dim is irrelevant to this test — only whether a dim is "big" (a feature/hidden width) or
/// "small" (a structural row count some loader constructs *do* slice, e.g. the cross-modal
/// `scale_shift_table_a2v_ca_{audio,video}` tables' `take_axis` for rows `0..4` + `4`, which needs
/// `dim0 ≥ 5`). Any dim ≤ 6 is preserved exactly (covers the real `2`- and `5`-row tables in this
/// fixture); anything larger shrinks to `1`. Total footprint: at most a handful of elements per tensor.
fn shrink_shape(shape: &[i32]) -> Vec<i32> {
    if shape.is_empty() {
        return vec![1];
    }
    shape
        .iter()
        .map(|&d| if d > 6 { 1 } else { d.max(1) })
        .collect()
}

fn placeholder(shape: &[i32]) -> Array {
    let shape = shrink_shape(shape);
    let n: i32 = shape.iter().product();
    Array::from_slice(&vec![0.0f32; n.max(1) as usize], &shape)
}

/// The real, `sanitize`-remapped LTX-2.5 DiT weights map: exactly the 4091 real DiT tensor names
/// (shapes shrunk per [`shrink_shape`] — `AvDiT::from_weights` never shape-validates at load time,
/// only names matter here).
fn real_25_weights() -> Weights {
    let mut m = std::collections::HashMap::new();
    for t in real_2_5_tensors() {
        m.insert(sanitize(&t.name), placeholder(&t.shape));
    }
    Weights::from_map(m)
}

/// A reconstructed "real 2.3-shaped" map: the real 2.5 set with the two confirmed deltas reversed —
/// `keyframes_abs_pos_embedding` removed, and the 96 video `ff.proj_{in,out}.bias` tensors added back
/// (shape taken from the real header's `ff.proj_in.weight`/`ff.proj_out.weight` out-dim: `[16384,
/// 4096]` ⇒ bias `[16384]`; `[4096, 16384]` ⇒ bias `[4096]` — shrunk the same way as every other
/// tensor here). This is not invented data — every tensor it adds/removes is one of the exact two
/// measured deltas, applied in reverse to real data.
fn reconstructed_23_weights() -> Weights {
    let mut m = std::collections::HashMap::new();
    for t in real_2_5_tensors() {
        if t.name == "keyframes_abs_pos_embedding" {
            continue;
        }
        m.insert(sanitize(&t.name), placeholder(&t.shape));
    }
    for i in 0..NUM_LAYERS {
        let p = format!("transformer_blocks.{i}");
        m.insert(format!("{p}.ff.proj_in.bias"), placeholder(&[16384]));
        m.insert(format!("{p}.ff.proj_out.bias"), placeholder(&[4096]));
    }
    Weights::from_map(m)
}

/// The 2.3 config: `ff_bias`/`audio_ff_bias` default `true` (both absent from a 2.3
/// `embedded_config.json`), `use_keyframes_abs_pos_embedding` defaults `false`.
fn ltx23_cfg() -> LtxConfig {
    let mut cfg = LtxConfig::video_only_defaults();
    cfg.apply_gated_attention = true;
    cfg.adaln_embedding_coefficient = 9;
    cfg.cross_attention_adaln = true;
    cfg.num_layers = NUM_LAYERS;
    cfg
}

/// The measured 2.3→2.5 delta, verified against the real header: `ff_bias:false` (video only) +
/// `use_keyframes_abs_pos_embedding:true`. `audio_ff_bias` is left at its default (`true`) — the real
/// header carries no such key.
fn ltx25_cfg() -> LtxConfig {
    let mut cfg = ltx23_cfg();
    cfg.ff_bias = false;
    cfg.use_keyframes_abs_pos_embedding = true;
    cfg
}

/// The core proof: the 2.5 config loads the **real** LTX-2.5 tensor set with zero missing and zero
/// extra (`unused_keys()` empty) — not a superset, the exact real 4091-tensor DiT set.
#[test]
fn ltx25_config_loads_the_real_header_exactly_no_missing_no_extra() {
    let prec = Precision::dense_f32(4, 64);
    let w = real_25_weights();
    let total = w.len();
    assert_eq!(
        total, 4091,
        "fixture must be the real 4091-tensor DiT-only set"
    );

    AvDiT::from_weights(&w, &ltx25_cfg(), prec).expect("ltx_2_5 config must load the real header");
    let unused = w.unused_keys();
    assert!(
        unused.is_empty(),
        "ltx_2_5 must read every real tensor — unread: {unused:?}"
    );
}

/// The mutation proof: the SAME real 2.5 tensor set must **fail** to load under the 2.3 config
/// (`ff_bias:true` demands the video FFN bias the real 2.5 checkpoint does not carry) — the flag, not
/// checkpoint content, decides requiredness.
#[test]
fn ltx23_config_fails_on_the_real_2_5_header() {
    let prec = Precision::dense_f32(4, 64);
    let w = real_25_weights();
    let err = AvDiT::from_weights(&w, &ltx23_cfg(), prec);
    assert!(
        err.is_err(),
        "ltx_2_3 config (ff_bias:true) must error on the real LTX-2.5 header, which carries no \
         video ff.proj_{{in,out}}.bias"
    );
}

/// 2.3 conformance is unchanged: the reconstructed real-2.3-shaped set (real 2.5 data with the two
/// measured deltas reversed) loads cleanly under the 2.3 config with zero missing, zero extra.
#[test]
fn ltx23_config_loads_the_reconstructed_2_3_header_exactly() {
    let prec = Precision::dense_f32(4, 64);
    let w = reconstructed_23_weights();
    AvDiT::from_weights(&w, &ltx23_cfg(), prec)
        .expect("ltx_2_3 config must load the reconstructed 2.3-shaped header");
    let unused = w.unused_keys();
    assert!(
        unused.is_empty(),
        "ltx_2_3 must read every tensor in its own header shape — unread: {unused:?}"
    );
}

/// The exact measured delta, spelled out against real tensor names: 96 video FFN bias tensors
/// removed (48 blocks × `ff.proj_in.bias` + `ff.proj_out.bias`), 1 `keyframes_abs_pos_embedding`
/// added — audio_ff bias is untouched (present in both).
#[test]
fn measured_delta_is_exactly_96_removed_and_1_added() {
    let real: HashSet<String> = real_2_5_tensors()
        .into_iter()
        .map(|t| sanitize(&t.name))
        .collect();
    let reconstructed: HashSet<String> = {
        let mut s: HashSet<String> = real
            .iter()
            .filter(|n| n.as_str() != "keyframes_abs_pos_embedding")
            .cloned()
            .collect();
        for i in 0..NUM_LAYERS {
            s.insert(format!("transformer_blocks.{i}.ff.proj_in.bias"));
            s.insert(format!("transformer_blocks.{i}.ff.proj_out.bias"));
        }
        s
    };

    let removed_going_25_to_23_direction: Vec<_> = reconstructed.difference(&real).collect();
    let added_going_23_to_25_direction: Vec<_> = real.difference(&reconstructed).collect();

    assert_eq!(
        removed_going_25_to_23_direction.len(),
        (NUM_LAYERS as usize) * 2,
        "reconstructed 2.3 adds back exactly 96 video ff bias tensors vs the real 2.5 set"
    );
    assert!(removed_going_25_to_23_direction
        .iter()
        .all(|k| k.contains(".ff.proj_") && k.contains(".bias") && !k.contains("audio_ff")));

    assert_eq!(added_going_23_to_25_direction.len(), 1);
    assert_eq!(
        added_going_23_to_25_direction[0].as_str(),
        "keyframes_abs_pos_embedding"
    );

    // Audio FFN bias is present and identical in both directions (untouched by the delta).
    for i in 0..NUM_LAYERS {
        let k = format!("transformer_blocks.{i}.audio_ff.proj_in.bias");
        assert!(real.contains(&k) && reconstructed.contains(&k));
        let k = format!("transformer_blocks.{i}.audio_ff.proj_out.bias");
        assert!(real.contains(&k) && reconstructed.contains(&k));
    }
}
