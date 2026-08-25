//! The MLX converter's pack predicates held against **this lane's** loaders — weights-free.
//!
//! The candle twin of `mlx-gen-minimax-h3/tests/quant_policy.rs`, and it exists for the same reason:
//! both halves of MiniMax-H3's tiering have the same failure mode and it is silent in both. A tensor
//! packed by `mlx_gen_minimax_h3::convert` that some loader reads with a raw `Weights::require` hands
//! back u32 codes where a float is expected and reports **no error at all** (sc-14980's Mage
//! `pos_embed`). The shapes are plausible, the render completes, the output is wrong.
//!
//! The only defence is that the converter's predicate and the loaders' split are the *same* set. This
//! file asserts that directly instead of trusting two comment blocks to agree.
//!
//! # Why the converter's constants are RESTATED here rather than imported
//!
//! The MLX sibling imports `convert::{DENSE_BY_POLICY, TE_PACK_SUFFIXES, …}` and compares against
//! them. This crate cannot: `mlx-gen-minimax-h3` is macOS-only (it links MLX), and a dependency from
//! the candle lane onto it would break every Linux and Windows build of this crate. So the tables
//! below are a **transcription**, pinned at the revision named on each one, and the property the
//! transcription buys is asserted against the real candle loaders rather than against another
//! comment.
//!
//! That makes this file a *one-directional* guard, and the direction is the safe one: it cannot
//! notice the converter growing a new pack target, but it does prove that every tensor this lane
//! reads either has a packed path or **refuses loudly**. A converter that packed something new would
//! hit a refusal at load rather than rendering garbage, which is the sc-14980 outcome that matters.
//! The MLX-side test owns the other direction.
//!
//! # Scope: this crate's OWN loaders
//!
//! The DiT and the Qwen3 decoder half of the text encoder load through [`lin`] / [`embed`] /
//! [`guard_dense`] here. The Qwen3-VL **vision tower** does not — it is
//! `candle_gen_boogu::vision::VisionTower`, with its own packed detect and its own dense-by-policy
//! refusals, tested at the tower in that crate. This file therefore covers the keys this crate
//! actually routes, and says so rather than reaching across the boundary to assert a guard the
//! production vision path never calls.
//!
//! # The sets, transcribed from `mlx_gen_minimax_h3::convert` (verified at `c7c215c25`)
//!
//! * `DENSE_BY_POLICY` — matched **exactly**, DiT only: `proj_in`, `proj_out`, `audio_proj_in`,
//!   `audio_proj_out`, `time_embedder.linear_1`, `time_embedder.linear_2`, `context_embedder`,
//!   `norm_out.linear`.
//! * `DENSE_NORM_SUFFIXES` — DiT, suffix-matched: `.norm1`, `.norm2`, `.norm_q`, `.norm_k`, `.norm`,
//!   `.final_norm`.
//! * `TE_DENSE_BY_POLICY` — text encoder, suffix-matched, and it **wins** over the pack list:
//!   `.input_layernorm`, `.post_attention_layernorm`, `.q_norm`, `.k_norm`, `.norm`, `.norm1`,
//!   `.norm2`, `.pos_embed`, `.patch_embed.proj`. The last four of those are **vision** keys, loaded
//!   by `candle_gen_boogu::vision::VisionTower` and guarded there rather than here — see
//!   [`TE_DENSE_KEYS`].
//! * `TE_PACK_SUFFIXES` — text encoder, suffix-matched: the seven Qwen3 decoder projections
//!   (`.self_attn.{q,k,v,o}_proj`, `.mlp.{gate,up,down}_proj`), `.embed_tokens`, and the four
//!   Qwen3-VL vision entries `.attn.qkv`, `.attn.proj`, `.linear_fc1`, `.linear_fc2`.

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::quant::MLX_GROUP_SIZE;
use candle_gen::Weights;

use candle_gen_minimax_h3::quant::{embed, guard_dense, lin, DIT, TEXT_ENCODER};

/// A deterministic MLX **Q4** affine pack at [`MLX_GROUP_SIZE`] of an `[out, in]` weight.
///
/// Returns `(wq [out, in/8] u32, scales, biases, grid [out, in])` — the packed triple a published
/// tier ships, plus the exact affine grid `scale·q + bias` it decodes to. Its own copy rather than
/// `crate::quant::testkit`'s, which is `#[cfg(test)] pub(crate)` and therefore invisible to an
/// integration test.
fn q4_pack(out_dim: usize, in_dim: usize, phase: usize) -> (Tensor, Tensor, Tensor, Tensor) {
    let dev = Device::Cpu;
    assert_eq!(in_dim % MLX_GROUP_SIZE, 0, "in_dim must be group-aligned");
    let codes: Vec<u8> = (0..out_dim * in_dim)
        .map(|i| (((i + phase) * 7 + i / 13) % 16) as u8)
        .collect();
    let groups_per_row = in_dim / MLX_GROUP_SIZE;
    let n_groups = out_dim * groups_per_row;
    let scales: Vec<f32> = (0..n_groups)
        .map(|k| 0.0625 * (((k + phase) % 7) as f32 + 1.0))
        .collect();
    let biases: Vec<f32> = (0..n_groups)
        .map(|k| -0.5 - 0.125 * ((k + phase) % 5) as f32)
        .collect();
    let grid: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| {
            let (row, col) = (i / in_dim, i % in_dim);
            let k = row * groups_per_row + col / MLX_GROUP_SIZE;
            scales[k] * f32::from(codes[i]) + biases[k]
        })
        .collect();
    let words: Vec<u32> = codes
        .chunks_exact(8)
        .map(|c| {
            c.iter()
                .enumerate()
                .fold(0u32, |acc, (i, &q)| acc | ((u32::from(q) & 0xF) << (4 * i)))
        })
        .collect();
    (
        Tensor::from_vec(words, (out_dim, in_dim / 8), &dev).unwrap(),
        Tensor::from_vec(scales, (out_dim, groups_per_row), &dev).unwrap(),
        Tensor::from_vec(biases, (out_dim, groups_per_row), &dev).unwrap(),
        Tensor::from_vec(grid, (out_dim, in_dim), &dev).unwrap(),
    )
}

/// Insert a packed triple for `{base}` into a key map, returning the grid it decodes to.
fn insert_packed(
    map: &mut HashMap<String, Tensor>,
    base: &str,
    out_dim: usize,
    in_dim: usize,
) -> Tensor {
    let (wq, scales, biases, grid) = q4_pack(out_dim, in_dim, base.len());
    map.insert(format!("{base}.weight"), wq);
    map.insert(format!("{base}.scales"), scales);
    map.insert(format!("{base}.biases"), biases);
    grid
}

/// Insert a dense `{base}.weight` — the shape a `bf16` tier ships.
fn insert_dense(map: &mut HashMap<String, Tensor>, base: &str, out_dim: usize, in_dim: usize) {
    map.insert(
        format!("{base}.weight"),
        Tensor::zeros((out_dim, in_dim), DType::F32, &Device::Cpu).unwrap(),
    );
}

/// **Relative max-abs-diff** — `max|a-b| / max|b|`, this crate's established measure.
///
/// Never cosine: cosine is scale-invariant and therefore structurally blind to a mis-decoded group
/// scale, which is precisely the defect class the packed path can produce.
fn rel_max_abs(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.dims(), b.dims(), "shape");
    let max_abs = |t: &Tensor| -> f32 {
        t.abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_vec0::<f32>()
            .unwrap()
    };
    let d = max_abs(&(a - b).unwrap());
    let scale = max_abs(b);
    if scale == 0.0 {
        d
    } else {
        d / scale
    }
}

/// Every base the converter's `TE_PACK_SUFFIXES` names, as a representative full dotted key on this
/// model — the keys a real packed text-encoder tier ships as u32 codes.
///
/// The vision entries carry `model.visual.` prefixes and the decoder entries
/// `model.language_model.layers.N.`, matching `VISION_PREFIX` / `LM_PREFIX`.
const TE_PACKED_KEYS: &[&str] = &[
    "model.language_model.layers.0.self_attn.q_proj",
    "model.language_model.layers.0.self_attn.k_proj",
    "model.language_model.layers.0.self_attn.v_proj",
    "model.language_model.layers.0.self_attn.o_proj",
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.0.mlp.up_proj",
    "model.language_model.layers.0.mlp.down_proj",
    "model.visual.blocks.0.attn.qkv",
    "model.visual.blocks.0.attn.proj",
    "model.visual.blocks.0.mlp.linear_fc1",
    "model.visual.merger.linear_fc1",
    "model.visual.deepstack_merger_list.0.linear_fc1",
];

/// Every base `TE_DENSE_BY_POLICY` keeps dense in **every** tier **that this crate's own loaders
/// read through [`guard_dense`]** — i.e. the Qwen3 decoder's norms.
///
/// # The vision keys are deliberately NOT in this list
///
/// `.pos_embed`, `.patch_embed.proj` and the vision `norm*` are also `TE_DENSE_BY_POLICY` members,
/// but this crate never calls `guard_dense` for them: they are loaded by
/// `candle_gen_boogu::vision::VisionTower`, which owns its own refusal (`require_dense`) at each of
/// the three sites. Asserting them here would prove a guard that production never reaches for those
/// keys — a green test over a path that does not exist, which is worse than no test because it reads
/// as coverage.
///
/// They are covered where they actually run, at the tower, by
/// `candle-gen-boogu/src/vision/mod.rs`'s `packed_pos_embed_is_refused_rather_than_silently_cast_to_floats`,
/// `packed_vision_norm_is_refused` and `packed_patch_embed_is_still_rejected` — all three driving
/// `VisionTower::load` rather than a standalone helper.
const TE_DENSE_KEYS: &[&str] = &[
    "model.language_model.layers.0.input_layernorm",
    "model.language_model.layers.0.post_attention_layernorm",
    "model.language_model.layers.0.self_attn.q_norm",
    "model.language_model.layers.0.self_attn.k_norm",
    "model.language_model.norm",
];

/// The DiT's `DENSE_BY_POLICY` — matched exactly, and dense in every tier.
const DIT_DENSE_KEYS: &[&str] = &[
    "proj_in",
    "proj_out",
    "audio_proj_in",
    "audio_proj_out",
    "time_embedder.linear_1",
    "time_embedder.linear_2",
    "context_embedder",
    "norm_out.linear",
];

/// The DiT bases the converter DOES pack — `is_dit_linear_target` is a negative predicate, so this is
/// everything that is neither dense-by-policy, nor a norm, nor an AdaLN target.
const DIT_PACKED_KEYS: &[&str] = &[
    "transformer_blocks.0.attn.to_q",
    "transformer_blocks.0.attn.to_k",
    "transformer_blocks.0.attn.to_v",
    "transformer_blocks.0.attn.to_out.0",
    "transformer_blocks.0.ff.net.0.proj",
    "transformer_blocks.0.ff.net.2",
];

/// **Every tensor the converter packs has a real packed path on this lane.**
///
/// The load must produce a *packed* base — not a dense one built by accident from the code stream,
/// which is the silent failure. Asserted on `is_packed()` and on the recovered `[out, in]`, which is
/// derived from the **scales** shape rather than the code column count (Q4 and Q8 pack a different
/// number of codes per u32 word, so the codes cannot answer the question).
#[test]
fn every_packed_target_loads_through_a_packed_path() {
    let (out, in_) = (32usize, 128usize);
    for (component, keys) in [(TEXT_ENCODER, TE_PACKED_KEYS), (DIT, DIT_PACKED_KEYS)] {
        for base in keys {
            let mut map = HashMap::new();
            insert_packed(&mut map, base, out, in_);
            let w = Weights::from_map(map);
            let loaded = lin(&w, component, base, false, DType::F32)
                .unwrap_or_else(|e| panic!("{component} {base}: packed load must succeed: {e}"));
            assert!(
                loaded.linear.is_packed(),
                "{component} {base}: the converter packs this, so the loader must build it PACKED — \
                 a dense base here means u32 codes were read as floats"
            );
            assert_eq!(
                loaded.linear.base_shape(),
                (out, in_),
                "{component} {base}: [out, in] must come from the scales"
            );
            assert!(
                loaded.base_bytes < out * in_ * 4,
                "{component} {base}: a packed base must hold fewer bytes than the dense weight"
            );
        }
    }
}

/// **The token table is a pack target too**, and it needs its own detect.
///
/// `.embed_tokens` is named in `TE_PACK_SUFFIXES`, so a packed tier ships it as u32 codes. A raw
/// `require` would hand those back as a float table and report nothing at all.
#[test]
fn the_token_table_loads_through_a_packed_path() {
    let mut map = HashMap::new();
    insert_packed(&mut map, "model.language_model.embed_tokens", 64, 128);
    let w = Weights::from_map(map);
    let loaded =
        embed(&w, "model.language_model.embed_tokens", DType::F32).expect("packed token table");
    assert_eq!(loaded.hidden, 128, "hidden comes from the scales");
    assert!(
        matches!(
            loaded.embedding,
            candle_gen::quant::QEmbedding::Quantized { .. }
        ),
        "a packed table must stay quantized, not be rebuilt densely"
    );
}

/// **Every dense-by-policy tensor is REFUSED if it ever arrives packed** — the sc-14980 class.
///
/// These are read with a raw `Weights::require` by design, so the guard is the only thing between a
/// converter change and silent garbage. Both lists are checked, and each refusal must cite its OWN
/// component's policy list: a text-encoder norm refused under a message reciting the DiT's list sends
/// the reader to the wrong converter constant.
#[test]
fn every_dense_by_policy_tensor_is_refused_when_packed() {
    for (component, keys, own, other) in [
        (
            TEXT_ENCODER,
            TE_DENSE_KEYS,
            "TE_DENSE_BY_POLICY",
            "DENSE_BY_POLICY keeps the float32 I/O heads",
        ),
        (DIT, DIT_DENSE_KEYS, "DENSE_BY_POLICY", "TE_DENSE_BY_POLICY"),
    ] {
        for base in keys {
            let mut map = HashMap::new();
            insert_packed(&mut map, base, 32, 128);
            let w = Weights::from_map(map);
            let err = match guard_dense(&w, component, base) {
                Ok(()) => panic!(
                    "{component} {base}: this tensor is dense in every tier and is read with a raw \
                     require — a packed one MUST be refused, not read as floats"
                ),
                Err(e) => e.to_string(),
            };
            assert!(
                err.contains(&format!("{base}.scales")),
                "the refusal must name the offending key: {err}"
            );
            assert!(err.contains("MLX-PACKED"), "{err}");
            assert!(
                err.contains(own),
                "{component} {base}: the refusal must cite its own policy list: {err}"
            );
            assert!(
                !err.contains(other),
                "{component} {base}: the refusal must NOT cite the other component's list: {err}"
            );
        }
    }
}

/// A **dense** tier passes the same guard untouched, so the guard gates on packedness rather than on
/// the key being in a list.
#[test]
fn a_dense_tier_passes_every_guard() {
    for (component, keys) in [(TEXT_ENCODER, TE_DENSE_KEYS), (DIT, DIT_DENSE_KEYS)] {
        for base in keys {
            let mut map = HashMap::new();
            insert_dense(&mut map, base, 32, 128);
            let w = Weights::from_map(map);
            guard_dense(&w, component, base)
                .unwrap_or_else(|e| panic!("{component} {base}: a dense tier must pass: {e}"));
        }
    }
}

/// **One loader serves every tier**, and the dense arm is reached by the ABSENCE of `.scales` rather
/// than by any manifest read.
#[test]
fn the_same_loader_serves_a_dense_tier_densely() {
    for base in DIT_PACKED_KEYS {
        let mut map = HashMap::new();
        insert_dense(&mut map, base, 32, 128);
        let w = Weights::from_map(map);
        let loaded = lin(&w, DIT, base, false, DType::F32).expect("dense load");
        assert!(
            !loaded.linear.is_packed(),
            "{base}: no `.scales` sibling must mean a dense base"
        );
        assert_eq!(loaded.base_bytes, 32 * 128 * 4, "{base}: dense f32 bytes");
    }
}

/// **Tier parity on relative max-abs**: the packed forward reproduces a dense projection built from
/// the same dequantized affine grid, at every packed base on both components.
///
/// The grid is the right reference because it is what the tier's producer quantized — a dense
/// checkpoint of the *unquantized* weight is a different tensor and comparing against it would only
/// measure quantization error, not decode correctness.
#[test]
fn the_packed_forward_matches_its_dense_grid_at_every_target() {
    let (out, in_) = (32usize, 128usize);
    let x = Tensor::from_vec(
        (0..3 * in_)
            .map(|i| (i as f32 * 0.13).sin())
            .collect::<Vec<_>>(),
        (3, in_),
        &Device::Cpu,
    )
    .unwrap();
    let mut worst = 0f32;
    for (component, keys) in [(TEXT_ENCODER, TE_PACKED_KEYS), (DIT, DIT_PACKED_KEYS)] {
        for base in keys {
            let mut map = HashMap::new();
            let grid = insert_packed(&mut map, base, out, in_);
            let packed = lin(&Weights::from_map(map), component, base, false, DType::F32)
                .expect("packed load");

            let mut dense_map = HashMap::new();
            dense_map.insert(format!("{base}.weight"), grid);
            let dense = lin(
                &Weights::from_map(dense_map),
                component,
                base,
                false,
                DType::F32,
            )
            .expect("dense load");

            let got = packed.linear.forward_upcast(&x).expect("packed forward");
            let want = dense.linear.forward_upcast(&x).expect("dense forward");
            let drift = rel_max_abs(&got, &want);
            assert!(
                drift < 5e-3,
                "{component} {base}: the Q4_1 repack is lossless up to the f16 scale/bias cast; got \
                 {drift:.3e}"
            );
            worst = worst.max(drift);
        }
    }
    println!("[quant-policy] worst packed-vs-dense-grid rel-max-abs = {worst:.3e}");
}

/// A `.scales` sibling whose `.weight` is **not** a u32 code stream is a typed load error, not a
/// silent repack of whatever floats happened to be there.
#[test]
fn a_packed_marker_over_a_float_weight_is_refused_on_both_components() {
    for component in [TEXT_ENCODER, DIT] {
        let mut map = HashMap::new();
        insert_packed(&mut map, "p", 32, 128);
        map.insert(
            "p.weight".to_string(),
            Tensor::zeros((32, 16), DType::F32, &Device::Cpu).unwrap(),
        );
        let err = match lin(&Weights::from_map(map), component, "p", false, DType::F32) {
            Ok(_) => panic!("{component}: a float weight under a packed marker must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("rather than U32"), "{component}: {err}");
    }
}

/// The group size this lane reads packed weights at is the **shared** constant, not a crate-local
/// re-declaration — a second declaration that drifted would derive a legal-looking, wrong bit width
/// from a perfectly good artifact (sc-15154).
#[test]
fn the_group_size_is_the_shared_constant() {
    assert_eq!(MLX_GROUP_SIZE, 64, "every published H3 tier packs at 64");
}
