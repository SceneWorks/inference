//! Community **single-file** Krea 2 DiT → in-memory diffusers-key remap (epic 14015, sc-14017 S0b).
//!
//! A ComfyUI-exported single-file Krea 2 checkpoint (e.g. `kreamania_variant5.safetensors`, a dense
//! bf16 merge) stores the DiT under **native-mmdit** tensor names beneath the
//! `model.diffusion_model.` prefix — `blocks.N.attn.{wq,wk,wv,wo,gate}`, `blocks.N.mlp.{gate,up,down}`,
//! `blocks.N.{prenorm,postnorm}.scale`, `blocks.N.attn.qknorm.{qnorm,knorm}.scale`, `blocks.N.mod.lin`,
//! the `txtfusion.{layerwise_blocks,refiner_blocks}.*` text-fusion stacks + `txtfusion.projector`,
//! `txtmlp.*`, `tmlp.*`, `tproj.*`, `first.*`, and `last.{linear,modulation,norm}.*`.
//!
//! The MLX [`crate::transformer::Krea2Transformer`] loads the **diffusers** key schema
//! (`transformer_blocks.N.attn.to_q`, `img_in`, `txt_in`, `time_embed`, `time_mod_proj`, `text_fusion`,
//! `final_layer.*`) — identity-keyed against the published `krea/Krea-2-Turbo` snapshot. So this module
//! renames every native key to its diffusers counterpart in memory, producing a [`Weights`] the existing
//! [`Krea2Transformer::from_weights`](crate::transformer::Krea2Transformer::from_weights) drops straight
//! into. Tensor **values and dtypes pass through untouched** — the community merge stores the norm/
//! modulation scales as bf16 (the published turnkey stores them f32; both upcast to f32 in the norm/
//! modulation forward), so a verbatim load is the faithful one.
//!
//! # Remap source
//!
//! The native↔diffusers correspondence is the **inverse** of candle's authoritative
//! `convrot_diffusers_to_native` in
//! `crates/media/candle-gen/candle-gen-krea/src/loader.rs` (sc-9300) — the map validated exhaustively
//! against the real native-mmdit header. It is replicated here (rather than shared) because that
//! function lives in the candle backend tree, is bolted to the INT8-ConvRot loader (int8 codes +
//! regular-Hadamard rotation), and runs diffusers→native; this MLX path wants the **pure key mapping
//! only** (no int8, no rotation) in the native→diffusers direction. Keep the two in lockstep: an edit to
//! the candle correspondence must be mirrored here.
//!
//! # Fail-closed
//!
//! [`remap_native_dit_to_diffusers`] fails closed (typed [`Error`], never a silent skip) on **any**
//! on-disk key it cannot map and on **any** two keys that would collide onto one diffusers name. The
//! complementary "every module weight the transformer needs is present" coverage + shape check is
//! [`crate::convert::validate_transformer`], run by the single-file loader after this remap.
//!
//! # Shape normalization
//!
//! The remap is a pure **key** rename. One diffusers-vs-native **shape** difference is normalized
//! separately by [`normalize_modulation_tables`] (a lossless row-major reshape of the per-block
//! `scale_shift_table` from the single file's flat `[6·hidden]` to the diffusers `[6, hidden]`), which
//! the single-file loader runs between the remap and `validate_transformer`.

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

/// Translate a **native-mmdit** single-file DiT tensor key to the **diffusers** key the MLX
/// [`Krea2Transformer`](crate::transformer::Krea2Transformer) module tree loads. Returns `None` for any
/// key that is not a recognized Krea DiT tensor (including a key missing the `model.diffusion_model.`
/// prefix) — the caller collects those and errors, so an unexpected/foreign tensor never slips through
/// silently.
///
/// This is the exact inverse of candle's `convrot_diffusers_to_native`
/// (`candle-gen-krea/src/loader.rs`), minus the int8/rotation coupling — see the module docs. Shapes
/// line up 1:1 (the only reshapes — `time_mod_proj` / `scale_shift_table` — are done by the DiT), so this
/// is a pure rename.
pub fn native_dit_key_to_diffusers(key: &str) -> Option<String> {
    // The ComfyUI single file namespaces the whole DiT under `model.diffusion_model.`; a DiT tensor
    // without it is unrecognized (returns `None` → fail-closed at the call site).
    let key = key.strip_prefix("model.diffusion_model.")?;

    // Top-level (non-block) tensors.
    let top = match key {
        "first.weight" => Some("img_in.weight"),
        "first.bias" => Some("img_in.bias"),
        "txtmlp.0.scale" => Some("txt_in.norm.weight"),
        "txtmlp.1.weight" => Some("txt_in.linear_1.weight"),
        "txtmlp.1.bias" => Some("txt_in.linear_1.bias"),
        "txtmlp.3.weight" => Some("txt_in.linear_2.weight"),
        "txtmlp.3.bias" => Some("txt_in.linear_2.bias"),
        "tmlp.0.weight" => Some("time_embed.linear_1.weight"),
        "tmlp.0.bias" => Some("time_embed.linear_1.bias"),
        "tmlp.2.weight" => Some("time_embed.linear_2.weight"),
        "tmlp.2.bias" => Some("time_embed.linear_2.bias"),
        "tproj.1.weight" => Some("time_mod_proj.weight"),
        "tproj.1.bias" => Some("time_mod_proj.bias"),
        "txtfusion.projector.weight" => Some("text_fusion.projector.weight"),
        "last.linear.weight" => Some("final_layer.linear.weight"),
        "last.linear.bias" => Some("final_layer.linear.bias"),
        "last.norm.scale" => Some("final_layer.norm.weight"),
        "last.modulation.lin" => Some("final_layer.scale_shift_table"),
        _ => None,
    };
    if let Some(t) = top {
        return Some(t.to_string());
    }

    // Per-block leaf remap (shared by the single-stream `blocks` and the two text-fusion stacks).
    let leaf = |rest: &str| -> Option<&'static str> {
        Some(match rest {
            "attn.qknorm.qnorm.scale" => "attn.norm_q.weight",
            "attn.qknorm.knorm.scale" => "attn.norm_k.weight",
            "attn.wq.weight" => "attn.to_q.weight",
            "attn.wk.weight" => "attn.to_k.weight",
            "attn.wv.weight" => "attn.to_v.weight",
            "attn.wo.weight" => "attn.to_out.0.weight",
            "attn.gate.weight" => "attn.to_gate.weight",
            "mlp.gate.weight" => "ff.gate.weight",
            "mlp.up.weight" => "ff.up.weight",
            "mlp.down.weight" => "ff.down.weight",
            "prenorm.scale" => "norm1.weight",
            "postnorm.scale" => "norm2.weight",
            "mod.lin" => "scale_shift_table",
            _ => return None,
        })
    };

    // `blocks.N.<leaf>` → `transformer_blocks.N.<diffusers-leaf>`.
    if let Some(rest) = key.strip_prefix("blocks.") {
        if let Some((idx, tail)) = rest.split_once('.') {
            if !idx.is_empty() && idx.chars().all(|c| c.is_ascii_digit()) {
                return leaf(tail).map(|dl| format!("transformer_blocks.{idx}.{dl}"));
            }
        }
    }

    // `txtfusion.{layerwise,refiner}_blocks.N.<leaf>` → `text_fusion.{...}.N.<diffusers-leaf>`.
    if let Some(rest) = key.strip_prefix("txtfusion.") {
        for kind in ["layerwise_blocks.", "refiner_blocks."] {
            if let Some(after) = rest.strip_prefix(kind) {
                if let Some((idx, tail)) = after.split_once('.') {
                    if !idx.is_empty() && idx.chars().all(|c| c.is_ascii_digit()) {
                        return leaf(tail).map(|dl| format!("text_fusion.{kind}{idx}.{dl}"));
                    }
                }
            }
        }
    }

    None
}

/// Rename every tensor in a native-mmdit single-file DiT weight set to its diffusers key, moving the
/// tensors into a fresh [`Weights`] the [`Krea2Transformer`](crate::transformer::Krea2Transformer) loads
/// directly. Values and dtypes are preserved verbatim.
///
/// Fails closed (typed [`Error::Msg`]) — never a silent skip — when:
/// * an on-disk key maps to `None` ([`native_dit_key_to_diffusers`]) — a foreign/unexpected tensor, or a
///   key missing the `model.diffusion_model.` prefix; or
/// * two distinct native keys collide onto the same diffusers key (a non-injective mapping).
///
/// Presence of every diffusers key the DiT needs (and the representative shape checks) is the separate
/// [`crate::convert::validate_transformer`] pass the loader runs on the returned set — an unmapped key
/// cannot be caught there because it never enters the returned map, which is why the unmapped case is
/// enforced here.
pub fn remap_native_dit_to_diffusers(mut native: Weights) -> Result<Weights> {
    let keys: Vec<String> = native.keys().map(str::to_string).collect();

    let mut out = Weights::empty();
    // diffusers key → the native key that produced it, for a precise collision diagnostic.
    let mut source: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut unmapped: Vec<String> = Vec::new();
    let mut collisions: Vec<String> = Vec::new();

    for native_key in keys {
        let Some(diffusers_key) = native_dit_key_to_diffusers(&native_key) else {
            unmapped.push(native_key);
            continue;
        };
        // `keys` came from `native.keys()`, so the remove is infallible.
        let tensor = native
            .remove(&native_key)
            .ok_or_else(|| Error::MissingTensor(native_key.clone()))?;
        if let Some(prev) = source.insert(diffusers_key.clone(), native_key.clone()) {
            collisions.push(format!("{prev} + {native_key} → {diffusers_key}"));
            continue;
        }
        out.insert(diffusers_key, tensor);
    }

    if !unmapped.is_empty() {
        unmapped.sort();
        return Err(Error::Msg(format!(
            "krea single-file remap: {} on-disk DiT key(s) have no diffusers mapping (unrecognized \
             checkpoint, wrong family, or a key outside the `model.diffusion_model.` DiT namespace): \
             [{}]",
            unmapped.len(),
            preview(&unmapped),
        )));
    }
    if !collisions.is_empty() {
        collisions.sort();
        return Err(Error::Msg(format!(
            "krea single-file remap: {} diffusers key collision(s) — the native→diffusers map is not \
             injective over this checkpoint: [{}]",
            collisions.len(),
            preview(&collisions),
        )));
    }
    Ok(out)
}

/// Normalize the modulation ("`scale_shift_table`") tables to the diffusers 2-D `[factors, hidden]`
/// shape (`6` factors per single-stream block, `2` for the final continuous-AdaLN layer).
///
/// The remap ([`remap_native_dit_to_diffusers`]) is a pure key rename — values/dtypes/shapes pass
/// through. But a ComfyUI single file may store the **per-block** modulation table FLAT
/// (`[factors·hidden]`, e.g. variant5's `blocks.N.mod.lin` is `[36864]`) where the published diffusers
/// `transformer/` snapshot stores it 2-D `[6, hidden]`. The flat form is row-major-identical to the 2-D
/// form (the DiT reshapes it to `[1, 1, 6·hidden]` either way, so the forward is unaffected), but the
/// shape check in [`crate::convert::validate_transformer`] expects the 2-D diffusers shape. Reshaping
/// flat→2-D (a lossless row-major view) makes the remapped set shape-identical to a snapshot load.
/// Already-2-D tables (variant5's `last.modulation.lin` is `[2, hidden]`) pass through untouched.
///
/// Errors (never silently reshapes to a wrong grid) if a flat table's element count is not divisible by
/// its factor — a truncated/foreign tensor.
pub fn normalize_modulation_tables(w: &mut Weights) -> Result<()> {
    let keys: Vec<String> = w
        .keys()
        .filter(|k| k.ends_with(".scale_shift_table"))
        .map(str::to_string)
        .collect();
    for key in keys {
        // The final continuous-AdaLN table is 2-factor (scale/shift); every single-stream block table is
        // 6-factor (pre/post × scale/shift/gate). Matches `Krea2Config::MOD_FACTORS` (6) and the DiT's
        // `final_layer` reshape to `[1, 2, hidden]`.
        let factors: i32 = if key == "final_layer.scale_shift_table" {
            2
        } else {
            6
        };
        // `remove` → own the tensor so the re-insert doesn't fight the read borrow.
        let tensor = w
            .remove(&key)
            .ok_or_else(|| Error::MissingTensor(key.clone()))?;
        if tensor.shape().len() == 2 {
            w.insert(key, tensor); // already `[factors, hidden]` — snapshot-shaped.
            continue;
        }
        let numel: i32 = tensor.shape().iter().product();
        if numel % factors != 0 {
            return Err(Error::Msg(format!(
                "krea single-file remap: `{key}` has {numel} elements, not divisible by {factors} \
                 modulation factors — cannot reshape to the diffusers [{factors}, hidden] table"
            )));
        }
        let reshaped = tensor.reshape(&[factors, numel / factors])?;
        w.insert(key, reshaped);
    }
    Ok(())
}

/// First few items of a diagnostic list (the full set can be hundreds of keys).
fn preview(items: &[String]) -> String {
    const HEAD: usize = 8;
    let shown = items
        .iter()
        .take(HEAD)
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(", ");
    if items.len() > HEAD {
        format!("{shown}, …")
    } else {
        shown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Krea2Config;
    use crate::convert::expected_transformer_keys;
    use mlx_rs::Array;
    use std::collections::BTreeSet;

    /// The real native-mmdit key set captured from `kreamania_variant5.safetensors` (430 tensors) — the
    /// committed fixture (the 26 GB weights file itself is not committed). Comment/blank lines dropped.
    fn variant5_native_keys() -> Vec<String> {
        let raw = include_str!("../tests/fixtures/variant5_native_keys.txt");
        raw.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(str::to_string)
            .collect()
    }

    /// A 1-element bf16 placeholder — the remap moves tensor handles by key; values/shape are irrelevant
    /// to key coverage, so a scalar stands in for each of the 430 real tensors.
    fn stub() -> Array {
        Array::from_slice(&[0.0f32], &[1])
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .unwrap()
    }

    /// The fixture is the real header: 430 native keys, all under the DiT prefix.
    #[test]
    fn fixture_is_the_real_variant5_header() {
        let keys = variant5_native_keys();
        assert_eq!(keys.len(), 430, "variant5 ships 430 DiT tensors");
        assert!(
            keys.iter().all(|k| k.starts_with("model.diffusion_model.")),
            "every variant5 DiT tensor is under the `model.diffusion_model.` prefix"
        );
    }

    /// **Every variant5 key template maps to a valid module-tree (diffusers) key, and the covered set is
    /// EXACTLY the set the transformer requires** — driven by the real header + the loader's own
    /// `expected_transformer_keys`. Set equality proves both directions at once: full coverage (no
    /// missing module weight) and no stray mapping (no diffusers key the module tree does not consume).
    #[test]
    fn remap_covers_every_variant5_key_and_matches_expected_module_keys() {
        let mapped: BTreeSet<String> = variant5_native_keys()
            .iter()
            .map(|k| {
                native_dit_key_to_diffusers(k)
                    .unwrap_or_else(|| panic!("variant5 key has no diffusers mapping: {k}"))
            })
            .collect();

        let expected: BTreeSet<String> = expected_transformer_keys(&Krea2Config::turbo())
            .into_iter()
            .collect();

        let missing: Vec<&String> = expected.difference(&mapped).collect();
        let extra: Vec<&String> = mapped.difference(&expected).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "remap ≠ module keys: missing {missing:?}, extra {extra:?}"
        );
    }

    /// **The mapping is a bijection over the covered set (no collisions):** 430 distinct native keys map
    /// to 430 distinct diffusers keys.
    #[test]
    fn remap_is_injective_over_variant5() {
        let native = variant5_native_keys();
        let mapped: BTreeSet<String> = native
            .iter()
            .filter_map(|k| native_dit_key_to_diffusers(k))
            .collect();
        assert_eq!(
            mapped.len(),
            native.len(),
            "collision: {} native keys collapsed to {} diffusers keys",
            native.len(),
            mapped.len()
        );
    }

    /// **`remap_native_dit_to_diffusers` renames the whole real header into a loadable diffusers set** —
    /// the end-to-end remap over a `Weights` built from every real key (stub tensors), asserting the
    /// output key set equals the module tree's expected set.
    #[test]
    fn remap_weights_end_to_end_over_real_header() {
        let mut w = Weights::empty();
        for k in variant5_native_keys() {
            w.insert(k, stub());
        }
        let out = remap_native_dit_to_diffusers(w).expect("real header remaps cleanly");

        let got: BTreeSet<String> = out.keys().map(str::to_string).collect();
        let expected: BTreeSet<String> = expected_transformer_keys(&Krea2Config::turbo())
            .into_iter()
            .collect();
        assert_eq!(
            got, expected,
            "remapped key set must equal the module tree's keys"
        );
    }

    /// **Fail-closed on an unmapped on-disk key.** A foreign/unexpected tensor (here a key without the
    /// `model.diffusion_model.` prefix) yields a typed error naming it — never a silent skip.
    #[test]
    fn unmapped_key_fails_closed() {
        let mut w = Weights::empty();
        // A valid key so the map is non-empty, plus one that cannot map.
        w.insert("model.diffusion_model.first.weight", stub());
        w.insert("unexpected.foreign.tensor", stub());

        // `Weights` is not `Debug`, so match rather than `expect_err`.
        let err = match remap_native_dit_to_diffusers(w) {
            Ok(_) => panic!("an unmapped key must fail closed"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("no diffusers mapping"),
            "unexpected error: {err}"
        );
        assert!(
            err.contains("unexpected.foreign.tensor"),
            "error must name the key: {err}"
        );
    }

    /// **Fail-closed on a native key inside the DiT namespace that is not a recognized leaf.** A
    /// truncated/garbage tensor under the prefix (`…blocks.0.attn.bogus`) still errors.
    #[test]
    fn unrecognized_leaf_under_prefix_fails_closed() {
        let mut w = Weights::empty();
        w.insert("model.diffusion_model.first.weight", stub());
        w.insert("model.diffusion_model.blocks.0.attn.bogus", stub());
        let err = match remap_native_dit_to_diffusers(w) {
            Ok(_) => panic!("an unrecognized leaf must fail closed"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("no diffusers mapping"),
            "unexpected error: {err}"
        );
    }

    /// **Fail-closed detects a missing module weight (via `validate_transformer` over the remap).**
    /// Dropping one native key (`first.bias`) yields a remapped set the transformer coverage check
    /// rejects — the complementary half of the fail-closed contract (unmapped is caught in the remap;
    /// missing is caught downstream). Proven here by wiring the two together as the loader does.
    #[test]
    fn missing_required_key_is_caught_by_validation() {
        let mut w = Weights::empty();
        for k in variant5_native_keys() {
            if k == "model.diffusion_model.first.bias" {
                continue; // simulate a truncated download missing img_in.bias
            }
            w.insert(k, stub());
        }
        let remapped =
            remap_native_dit_to_diffusers(w).expect("remap of a present subset is clean");
        // The coverage half is `validate_transformer`; shapes are stubbed, so assert the coverage
        // message specifically (it runs before the shape checks).
        let err = crate::convert::validate_transformer(&remapped, &Krea2Config::turbo())
            .expect_err("a missing module weight must fail closed")
            .to_string();
        assert!(
            err.contains("img_in.bias"),
            "error must name the missing key: {err}"
        );
    }

    /// The individual top-level and per-block correspondences, spot-checked against candle's
    /// `convrot_diffusers_to_native` inverse (the load-bearing renames the epic calls out).
    #[test]
    fn spot_check_representative_renames() {
        let cases = [
            ("model.diffusion_model.first.weight", "img_in.weight"),
            (
                "model.diffusion_model.last.modulation.lin",
                "final_layer.scale_shift_table",
            ),
            (
                "model.diffusion_model.last.norm.scale",
                "final_layer.norm.weight",
            ),
            ("model.diffusion_model.txtmlp.0.scale", "txt_in.norm.weight"),
            (
                "model.diffusion_model.txtfusion.projector.weight",
                "text_fusion.projector.weight",
            ),
            (
                "model.diffusion_model.tproj.1.weight",
                "time_mod_proj.weight",
            ),
            (
                "model.diffusion_model.blocks.7.attn.wq.weight",
                "transformer_blocks.7.attn.to_q.weight",
            ),
            (
                "model.diffusion_model.blocks.7.attn.qknorm.qnorm.scale",
                "transformer_blocks.7.attn.norm_q.weight",
            ),
            (
                "model.diffusion_model.blocks.7.attn.wo.weight",
                "transformer_blocks.7.attn.to_out.0.weight",
            ),
            (
                "model.diffusion_model.blocks.7.mod.lin",
                "transformer_blocks.7.scale_shift_table",
            ),
            (
                "model.diffusion_model.txtfusion.refiner_blocks.1.mlp.down.weight",
                "text_fusion.refiner_blocks.1.ff.down.weight",
            ),
        ];
        for (native, diffusers) in cases {
            assert_eq!(
                native_dit_key_to_diffusers(native).as_deref(),
                Some(diffusers),
                "wrong remap for {native}"
            );
        }
    }

    /// **`normalize_modulation_tables` reshapes a flat per-block table to the diffusers 2-D shape and
    /// leaves an already-2-D final table alone.** A flat `[6·h]` block table (variant5's `mod.lin` form)
    /// becomes `[6, h]`; the `[2, h]` final table (variant5's `modulation.lin` form) is unchanged; the
    /// reshape is a lossless row-major view so the flattened values are preserved.
    #[test]
    fn normalize_reshapes_flat_block_table_and_keeps_2d_final() {
        let hidden = 4i32;
        // Flat block table `[6·hidden]` = [24], row-major values 0..24.
        let flat: Vec<f32> = (0..(6 * hidden)).map(|i| i as f32).collect();
        let final_2d: Vec<f32> = (0..(2 * hidden)).map(|i| (100 + i) as f32).collect();

        let mut w = Weights::empty();
        w.insert(
            "transformer_blocks.0.scale_shift_table",
            Array::from_slice(&flat, &[6 * hidden]),
        );
        w.insert(
            "final_layer.scale_shift_table",
            Array::from_slice(&final_2d, &[2, hidden]),
        );

        normalize_modulation_tables(&mut w).expect("normalization is clean");

        let block = w.require("transformer_blocks.0.scale_shift_table").unwrap();
        assert_eq!(
            block.shape(),
            &[6, hidden],
            "flat block table reshaped to [6, hidden]"
        );
        // Row-major values preserved by the reshape (contiguous view → physical buffer in order).
        assert_eq!(block.as_slice::<f32>(), flat.as_slice());

        let fin = w.require("final_layer.scale_shift_table").unwrap();
        assert_eq!(
            fin.shape(),
            &[2, hidden],
            "already-2-D final table unchanged"
        );
    }

    /// **`normalize_modulation_tables` fails closed on a flat table whose element count is not divisible
    /// by its factor** — a truncated/foreign tensor, not silently reshaped to a wrong grid.
    #[test]
    fn normalize_fails_closed_on_indivisible_flat_table() {
        let mut w = Weights::empty();
        // 25 is not divisible by the 6 block modulation factors.
        w.insert(
            "transformer_blocks.0.scale_shift_table",
            Array::from_slice(&[0.0f32; 25], &[25]),
        );
        let err = normalize_modulation_tables(&mut w)
            .expect_err("indivisible flat table must fail closed")
            .to_string();
        assert!(err.contains("not divisible"), "unexpected error: {err}");
    }
}
