//! Offline builder for the **candle-native** packed Wan quant tiers (sc-10026, epic 9083) — hosted at
//! `SceneWorks/wan2.2-{ti2v-5b,t2v-a14b,i2v-a14b}-candle`.
//!
//! Option B of the sc-10026 design: rather than reverse-remap the native-Wan-keyed `SceneWorks/*-mlx`
//! tiers, candle hosts its OWN packed tiers on the **diffusers** keys it already reads. This builder
//! loads a `Wan-AI/Wan2.2-*-Diffusers` snapshot and rewrites each transformer component so every DiT
//! Linear weight is quantized on the same diffusers key. The result keeps the diffusers component
//! layout (`transformer/` [+ `transformer_2/`] + `text_encoder/` + `vae/` + `tokenizer/`), so the worker
//! load path resolves it with **zero change**.
//!
//! **Storage (SW-7285).** A new build emits a **Candle-native GGML tier**: every Linear is ONE stored
//! GGML block tensor `{key}.weight` = `U8 [out, in / 32, block_bytes]` (no `.scales` / `.biases`), the
//! exact blocks the load-time repack of an MLX triple builds, so the shared loader
//! (`candle_layers::stored_ggml`, behind [`crate::quant::QLinear::linear_detect`]) installs them as
//! stored — no repack, no saved-quantization cache entry:
//!
//! * **q4** — MLX-affine pack at group [`TIER_GROUP_SIZE`] ([`candle_gen::quant::pack_mlx_affine`]),
//!   then the lossless `Q4_1` repack (`to_ggml_block_tensor(repack_packed_weight_uncached(..))`) — the
//!   same blocks `candle_ggml_tier --tier q4=from-mlx` writes from a published MLX q4 tier;
//! * **q8** — `Q8_0` straight from the dense weight (candle's own quantizer), the same blocks
//!   `candle_ggml_tier --tier q8=from-bf16@..` writes; never the lossy MLX-q8 -> `Q8_0` requant.
//!
//! Both go through the family-generic builder's primitives ([`candle_gen::ggml_tier`]), so there is one
//! quantizer. The component's `quantize_config.json` keeps the top-level `bits` the Wan memory strategy
//! reads and carries `quantization = {"bits": N, "storage": "ggml"}`
//! ([`candle_gen::ggml_tier::ggml_quantize_config`], byte-identical to what the generic builder writes
//! when it converts an already-published MLX-affine Wan tier).
//!
//! [`pack_transformer_component`] (the MLX-affine triple form) stays for its other callers (the
//! Bernini converter and the packed-seam fixtures); tier builds use [`ggml_transformer_component`].
//!
//! **What quantizes, what stays dense.** Only rank-2 `.weight` tensors in the transformer component(s):
//! those are exactly the seam's Linear surface (attn `to_q/k/v/to_out.0`, ffn `net.0.proj/net.2`, the
//! condition-embedder Linears, `proj_out`). Everything else stays dense — the 1-D norms, the 3-D
//! `scale_shift_table`, the `patch_embedding` conv (rank >= 4), every `.bias`, and the whole T5 encoder
//! plus the z16 VAE (the MLX build keeps those dense too).
//!
//! The [`build_candle_wan_tier`] entry point is an `#[ignore]`d test — it needs the multi-GB diffusers
//! snapshot on disk. Run it on-device per model/tier, then `hf upload` the output to the `*-candle` repo.
//! The pure [`ggml_transformer_component`] / [`pack_transformer_component`] cores are unit-tested in CI.

use std::collections::HashMap;
use std::path::Path;

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::ggml_tier;
use candle_gen::quant::pack_mlx_affine;

/// The quant group size the candle Wan tiers pack at (matches the MLX tiers + the seam's default).
pub const TIER_GROUP_SIZE: usize = 64;

/// Rewrite a transformer component's CPU tensor `map`, MLX-affine-packing every **rank-2 `.weight`**
/// (the seam's Linear set) at `bits` (4 or 8) / [`TIER_GROUP_SIZE`] into the `{base}.weight` (u32) +
/// `{base}.scales` + `{base}.biases` triple, and leaving every other tensor untouched (dense norms /
/// conv / `scale_shift_table` / `.bias`). Returns the rewritten map + the number of Linears packed.
///
/// Rank-2 is the exact seam predicate here: in the Wan diffusers transformer the only rank-2 weights are
/// the Linears the seam routes through `linear_detect`; the `patch_embedding` conv is rank ≥ 4, the norms
/// are rank 1, `scale_shift_table` is rank 3. Packing a weight the seam would load densely would make it
/// load u32 codes as bf16 garbage — so this predicate must stay aligned with the seam (both key off "the
/// Linears"). Pure + deterministic (no I/O) so CI unit-tests it without weights.
pub fn pack_transformer_component(
    map: HashMap<String, Tensor>,
    bits: usize,
) -> Result<(HashMap<String, Tensor>, usize)> {
    let mut out: HashMap<String, Tensor> = HashMap::with_capacity(map.len());
    let mut packed = 0usize;
    for (key, value) in map {
        let is_linear_weight = key.ends_with(".weight") && value.rank() == 2;
        if is_linear_weight {
            let base = key.strip_suffix(".weight").unwrap();
            let (wq, scales, biases) =
                pack_mlx_affine(&value.to_dtype(DType::F32)?, bits, TIER_GROUP_SIZE)?;
            out.insert(format!("{base}.weight"), wq);
            out.insert(format!("{base}.scales"), scales);
            out.insert(format!("{base}.biases"), biases);
            packed += 1;
        } else {
            // Dense passthrough: norms (rank 1), scale_shift_table (rank 3), patch_embedding conv
            // (rank ≥ 4), every `.bias`, and any non-weight tensor. Kept at its on-disk dtype.
            out.insert(key, value);
        }
    }
    Ok((out, packed))
}

/// The **Candle-native GGML** form of [`pack_transformer_component`] (SW-7285): the same rank-2
/// `.weight` predicate, but each Linear becomes ONE stored GGML block tensor `{base}.weight` =
/// `U8 [out, in / 32, block_bytes]` — `Q4_1` from the group-[`TIER_GROUP_SIZE`] MLX q4 pack (lossless
/// against that pack) at `bits = 4`, `Q8_0` straight from the dense weight at `bits = 8`. No
/// `.scales` / `.biases`. Every other tensor is untouched. Returns the rewritten map + the number of
/// Linears quantized. The quantizers are [`candle_gen::ggml_tier`]'s, so the blocks are byte-for-byte
/// what `candle_ggml_tier` writes when it converts the MLX-affine tier [`pack_transformer_component`]
/// builds (q4) or quantizes the same dense source (q8).
pub fn ggml_transformer_component(
    map: HashMap<String, Tensor>,
    bits: usize,
) -> Result<(HashMap<String, Tensor>, usize)> {
    if bits != 4 && bits != 8 {
        candle_gen::candle_core::bail!("wan ggml tier: bits must be 4 or 8 (got {bits})");
    }
    let mut out: HashMap<String, Tensor> = HashMap::with_capacity(map.len());
    let mut packed = 0usize;
    for (key, value) in map {
        if key.ends_with(".weight") && value.rank() == 2 {
            let base = key.strip_suffix(".weight").unwrap();
            let blocks = if bits == 4 {
                let (wq, scales, biases) =
                    pack_mlx_affine(&value.to_dtype(DType::F32)?, 4, TIER_GROUP_SIZE)?;
                ggml_tier::mlx_triple_to_ggml_blocks(base, &wq, &scales, &biases, TIER_GROUP_SIZE)
            } else {
                ggml_tier::dense_to_q8_0_blocks(base, &value)
            }
            .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
            out.insert(key, blocks);
            packed += 1;
        } else {
            out.insert(key, value);
        }
    }
    Ok((out, packed))
}

/// The `quantize_config.json` text a Wan GGML transformer component carries: the top-level `bits`
/// the Wan memory strategy reads, plus `quantization = {"bits": N, "storage": "ggml"}`. Derived from
/// the MLX-affine tier's own marker through [`ggml_tier::ggml_quantize_config`], so it is exactly what
/// the generic builder writes when it converts a published MLX-affine Wan tier.
pub fn ggml_quantize_config_text(bits: usize) -> Result<String> {
    ggml_tier::ggml_quantize_config(&mlx_quantize_config_text(bits))
        .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))
}

/// The MLX-affine tier's `quantize_config.json` (the sc-10026 marker, still what the published
/// `*-candle` tiers carry).
fn mlx_quantize_config_text(bits: usize) -> String {
    format!(
        "{{\n  \"bits\": {bits},\n  \"quantization\": {{ \"group_size\": {TIER_GROUP_SIZE} }}\n}}\n"
    )
}

/// Load a transformer component dir's `.safetensors` shards, [`ggml_transformer_component`] them at
/// `bits`, and write the GGML component to `dst` as a single `model.safetensors` + the
/// [`ggml_quantize_config_text`] `quantize_config.json`. Any non-`.safetensors` sidecar (e.g. the
/// diffusers `config.json`) is copied verbatim.
fn pack_component_dir(src: &Path, dst: &Path, bits: usize) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    // `sorted_safetensors` returns candle-gen's error type; bridge it into `candle_core::Result`.
    let files = candle_gen::sorted_safetensors(src, "wan-candle-tier")
        .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
    let mut map: HashMap<String, Tensor> = HashMap::new();
    for f in &files {
        map.extend(candle_gen::candle_core::safetensors::load(f, &Device::Cpu)?);
    }
    let (packed, n) = ggml_transformer_component(map, bits)?;
    candle_gen::candle_core::safetensors::save(&packed, dst.join("model.safetensors"))?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        // Copy sidecars (e.g. the diffusers `config.json`), but NOT any `.safetensors` shard or its
        // `.safetensors.index.json` — we collapsed the shards into one `model.safetensors`, so a copied
        // shard index would dangle and mislead a sharded loader.
        if entry.path().is_file() && !name.contains(".safetensors") {
            std::fs::copy(entry.path(), dst.join(&name))?;
        }
    }
    std::fs::write(
        dst.join("quantize_config.json"),
        ggml_quantize_config_text(bits)?,
    )?;
    eprintln!(
        "[[CANDLE-TIER]] stored {n} Linears as GGML blocks: {} -> {}",
        src.display(),
        dst.display()
    );
    Ok(())
}

/// Copy a dense component dir (e.g. `text_encoder/`, `vae/`, `tokenizer/`) verbatim, recursively.
fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Build one candle GGML tier from a `Wan-AI/Wan2.2-*-Diffusers` snapshot at `diffusers_dir` into
/// `out_dir` at `bits` (4 or 8). Quantizes each transformer component (`transformer/`, plus `transformer_2/`
/// for the A14B dual-expert MoE) via `pack_component_dir`; copies every other component/file
/// (`text_encoder/`, `vae/`, `tokenizer/`, `scheduler/`, `model_index.json`, …) verbatim so the tier is
/// a complete, self-contained diffusers-layout snapshot the sc-10025 seam loads unchanged. Host the
/// result at `SceneWorks/wan2.2-*-candle`.
pub fn build_candle_wan_tier(diffusers_dir: &Path, out_dir: &Path, bits: usize) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    for entry in std::fs::read_dir(diffusers_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let src = entry.path();
        let dst = out_dir.join(&name);
        if src.is_dir() && (name == "transformer" || name == "transformer_2") {
            pack_component_dir(&src, &dst, bits)?;
        } else if src.is_dir() {
            copy_dir_recursive(&src, &dst)?;
        } else {
            std::fs::copy(&src, &dst)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::safetensors::MmapedSafetensors;
    use candle_gen::candle_core::Device;
    use candle_gen::candle_nn::{Linear, Module, VarBuilder};

    fn dense(out_dim: usize, in_dim: usize, seed: f32) -> Tensor {
        let data: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i as f32 + seed) * 0.013).sin() * 1.3)
            .collect();
        Tensor::from_vec(data, (out_dim, in_dim), &Device::Cpu).unwrap()
    }

    /// `pack_transformer_component` packs exactly the rank-2 `.weight`s (the Linear surface) and leaves
    /// the norm (rank 1), `scale_shift_table` (rank 3), `patch_embedding` conv (rank 5) and every `.bias`
    /// dense — the on-disk shape the seam loads.
    #[test]
    fn packs_only_rank2_weights() -> Result<()> {
        let dev = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("blocks.0.attn1.to_q.weight".into(), dense(64, 128, 1.0));
        map.insert(
            "blocks.0.attn1.to_q.bias".into(),
            Tensor::zeros(64, DType::F32, &dev)?,
        );
        map.insert("blocks.0.ffn.net.0.proj.weight".into(), dense(256, 64, 2.0));
        map.insert("proj_out.weight".into(), dense(64, 64, 3.0));
        // Dense-only leaves.
        map.insert(
            "blocks.0.norm_q.weight".into(),
            Tensor::ones(64, DType::F32, &dev)?,
        ); // rank 1
        map.insert(
            "scale_shift_table".into(),
            Tensor::zeros((1, 6, 64), DType::F32, &dev)?, // rank 3
        );
        map.insert(
            "patch_embedding.weight".into(),
            Tensor::zeros((64, 48, 1, 2, 2), DType::F32, &dev)?, // rank 5 conv
        );

        let (out, packed) = pack_transformer_component(map, 4)?;
        assert_eq!(packed, 3, "the three rank-2 weights pack");
        // Packed Linears gained the triple; the u32 codes replace the dense weight.
        for base in ["blocks.0.attn1.to_q", "blocks.0.ffn.net.0.proj", "proj_out"] {
            assert_eq!(out[&format!("{base}.weight")].dtype(), DType::U32);
            assert!(out.contains_key(&format!("{base}.scales")));
            assert!(out.contains_key(&format!("{base}.biases")));
        }
        // Dense leaves untouched (dtype + rank).
        assert_eq!(out["blocks.0.attn1.to_q.bias"].rank(), 1);
        assert_eq!(out["blocks.0.norm_q.weight"].dtype(), DType::F32);
        assert_eq!(out["scale_shift_table"].rank(), 3);
        assert_eq!(out["patch_embedding.weight"].rank(), 5);
        assert!(!out.contains_key("scale_shift_table.scales"));
        Ok(())
    }

    fn dense_component() -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("blocks.0.attn1.to_q.weight".into(), dense(64, 128, 1.0));
        map.insert(
            "blocks.0.attn1.to_q.bias".into(),
            Tensor::zeros(64, DType::F32, &dev).unwrap(),
        );
        map.insert("proj_out.weight".into(), dense(32, 64, 3.0));
        map.insert(
            "blocks.0.norm_q.weight".into(),
            Tensor::ones(64, DType::F32, &dev).unwrap(),
        );
        map.insert(
            "scale_shift_table".into(),
            Tensor::zeros((1, 6, 64), DType::F32, &dev).unwrap(),
        );
        map.insert(
            "patch_embedding.weight".into(),
            Tensor::zeros((64, 48, 1, 2, 2), DType::F32, &dev).unwrap(),
        );
        map
    }

    fn u8s(t: &Tensor) -> Vec<u8> {
        t.flatten_all().unwrap().to_vec1::<u8>().unwrap()
    }

    /// SW-7285: a new Wan tier build stores every rank-2 Linear as GGML blocks — `Q4_1` equal to the
    /// load-time repack of the group-64 MLX q4 pack, `Q8_0` equal to candle's own quantizer over the
    /// dense weight — with no `.scales` / `.biases`, and leaves every other tensor as it was.
    #[test]
    fn ggml_build_emits_stored_blocks_equal_to_the_repack_and_q8_0() -> Result<()> {
        use candle_gen::candle_core::quantized::{GgmlDType, QTensor};
        let dev = Device::Cpu;
        for bits in [4usize, 8] {
            let (out, n) = ggml_transformer_component(dense_component(), bits)?;
            assert_eq!(n, 2, "the two rank-2 weights quantize");
            assert!(out
                .keys()
                .all(|k| !k.ends_with(".scales") && !k.ends_with(".biases")));
            for (base, o, i) in [("blocks.0.attn1.to_q", 64, 128), ("proj_out", 32, 64)] {
                let stored = &out[&format!("{base}.weight")];
                let w = dense_component()[&format!("{base}.weight")].clone();
                let want = if bits == 4 {
                    let (wq, s, b) = pack_mlx_affine(&w, 4, TIER_GROUP_SIZE)?;
                    candle_gen::quant::repack_packed_weight_uncached(
                        &wq,
                        &s,
                        &b,
                        TIER_GROUP_SIZE,
                        &dev,
                    )?
                } else {
                    QTensor::quantize(&w, GgmlDType::Q8_0)?
                };
                let block_bytes = if bits == 4 { 20 } else { 34 };
                assert_eq!(stored.dtype(), DType::U8);
                assert_eq!(stored.dims(), [o, i / 32, block_bytes], "{base} q{bits}");
                assert_eq!(u8s(stored), want.data()?.into_owned(), "{base} q{bits}");
            }
            assert_eq!(out["blocks.0.norm_q.weight"].dtype(), DType::F32);
            assert_eq!(out["scale_shift_table"].rank(), 3);
            assert_eq!(out["patch_embedding.weight"].rank(), 5);
            assert_eq!(out["blocks.0.attn1.to_q.bias"].rank(), 1);
        }
        assert!(ggml_transformer_component(dense_component(), 3).is_err());
        Ok(())
    }

    /// SW-7285: a component written by the Wan builder and the generic `candle_ggml_tier` conversion
    /// of the published MLX-affine form of the same component are the same tier: identical q4 blocks,
    /// identical passthrough, identical `quantize_config.json` marker (top-level `bits` kept, the
    /// `quantization` block carrying `"storage": "ggml"`), the diffusers config copied verbatim.
    #[test]
    fn wan_builder_and_generic_converter_agree_on_the_ggml_tier() -> Result<()> {
        let root = tempfile::tempdir().unwrap();
        let (src, mlx, ours, generic) = (
            root.path().join("diffusers/transformer"),
            root.path().join("mlx/transformer"),
            root.path().join("ours/transformer"),
            root.path().join("generic/transformer"),
        );
        let cfg = r#"{"_class_name": "WanTransformer3DModel", "num_layers": 1}"#;
        // A Wan-AI diffusers component, and its published MLX-affine q4 tier (the sc-10026 format).
        std::fs::create_dir_all(&src)?;
        candle_gen::candle_core::safetensors::save(
            &dense_component(),
            src.join("diffusion_pytorch_model.safetensors"),
        )?;
        std::fs::write(src.join("config.json"), cfg)?;
        std::fs::create_dir_all(&mlx)?;
        let (packed, _) = pack_transformer_component(dense_component(), 4)?;
        candle_gen::candle_core::safetensors::save(&packed, mlx.join("model.safetensors"))?;
        std::fs::write(mlx.join("config.json"), cfg)?;
        std::fs::write(
            mlx.join("quantize_config.json"),
            mlx_quantize_config_text(4),
        )?;

        pack_component_dir(&src, &ours, 4)?;
        let report = ggml_tier::convert_component(
            &mlx,
            &generic,
            &ggml_tier::SourceMode::FromMlx,
            &mut |_, _| Ok(()),
        )
        .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
        assert_eq!(report.converted, 2);

        let load = |p: &Path| candle_gen::candle_core::safetensors::load(p, &Device::Cpu);
        let a = load(&ours.join("model.safetensors"))?;
        let b = load(&generic.join("model.safetensors"))?;
        let mut keys: Vec<_> = a.keys().cloned().collect();
        keys.sort();
        let mut bkeys: Vec<_> = b.keys().cloned().collect();
        bkeys.sort();
        assert_eq!(keys, bkeys);
        for k in &keys {
            assert_eq!(a[k].dtype(), b[k].dtype(), "{k}");
            assert_eq!(a[k].dims(), b[k].dims(), "{k}");
            if a[k].dtype() == DType::U8 {
                assert_eq!(u8s(&a[k]), u8s(&b[k]), "{k}");
            } else {
                let f = |t: &Tensor| t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>();
                assert_eq!(f(&a[k])?, f(&b[k])?, "{k}");
            }
        }
        let marker = std::fs::read_to_string(ours.join("quantize_config.json"))?;
        assert_eq!(
            marker,
            std::fs::read_to_string(generic.join("quantize_config.json"))?
        );
        let v: serde_json::Value = serde_json::from_str(&marker).unwrap();
        assert_eq!(v["bits"], 4);
        assert_eq!(
            v["quantization"],
            serde_json::json!({"bits": 4, "storage": "ggml"})
        );
        assert_eq!(std::fs::read_to_string(ours.join("config.json"))?, cfg);
        assert_eq!(std::fs::read_to_string(generic.join("config.json"))?, cfg);
        Ok(())
    }

    /// End-to-end: a packed component written by `pack_transformer_component` is loaded back through the
    /// **seam** (`QLinear::linear_detect`) — it fires the packed path and the dequantized forward matches
    /// the original dense Linear (Q4 cosine parity). Proves the producer ↔ consumer round-trip that lets
    /// candle host its own tiers.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn packed_component_loads_through_the_seam() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128, 256);
        let w = dense(out_dim, in_dim, 7.0);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("blocks.0.attn1.to_q.weight".into(), w.clone());

        let (packed_map, packed) = pack_transformer_component(map, 4)?;
        assert_eq!(packed, 1);

        let tmp_guard = tempfile::tempdir().unwrap();
        let tmp = tmp_guard.path().join("sc10026_component.safetensors");
        candle_gen::candle_core::safetensors::save(&packed_map, &tmp)?;
        // SAFETY: freshly written, single-reader for the test.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        // The crate seam loads it packed.
        let loaded = crate::quant::QLinear::linear_detect(
            in_dim,
            out_dim,
            &vb.pp("blocks.0.attn1"),
            "to_q",
            false,
        )?;
        assert!(
            loaded.is_packed(),
            "written tier must load through the packed seam path"
        );

        // Dequantized forward ≈ the original dense Linear (Q4 cosine parity). Q4 is genuinely lossy, so
        // this is a cosine tolerance (> 0.997), not bit-equality; the input is **deterministic** (not an
        // unseeded `randn`) so the parity is portable — an unseeded input made the cosine flap around the
        // Q4 loss floor and fail on CI's FP under a too-tight threshold ([[fp-byte-equality tests are
        // non-portable]]).
        let dense_lin = Linear::new(w, None);
        let x = dense(4, in_dim, 11.0);
        let a = loaded.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
        let b = __model_proof_output!(dense_lin.forward(&x))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(&b) {
            dot += *x as f64 * *y as f64;
            na += *x as f64 * *x as f64;
            nb += *y as f64 * *y as f64;
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
        assert!(
            cos > 0.997,
            "packed-load forward cosine {cos:.6} vs dense too low"
        );

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED candle_tier_build::tests::packed_component_loads_through_the_seam");
        Ok(())
    }

    /// On-device tier build (`#[ignore]`d — needs a `Wan-AI/Wan2.2-*-Diffusers` snapshot). Run per
    /// model/tier, then `hf upload` the output dir to `SceneWorks/wan2.2-<model>-candle`:
    ///
    /// ```sh
    /// export SCENEWORKS_CANDLE_WAN_DIFFUSERS_DIR=<Wan-AI/Wan2.2-T2V-A14B-Diffusers snapshot>
    /// export SCENEWORKS_CANDLE_WAN_TIER_OUT=<out-dir>          # the packed candle tier is written here
    /// export SCENEWORKS_CANDLE_WAN_BITS=4                      # 4 (q4) or 8 (q8)
    /// cargo test -p candle-gen-wan --release build_candle_wan_tier_from_env -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "on-device tier build: needs a Wan-AI/Wan2.2-*-Diffusers snapshot on disk"]
    fn build_candle_wan_tier_from_env() {
        let diffusers = std::env::var("SCENEWORKS_CANDLE_WAN_DIFFUSERS_DIR")
            .expect("set SCENEWORKS_CANDLE_WAN_DIFFUSERS_DIR to the diffusers snapshot dir");
        let out = std::env::var("SCENEWORKS_CANDLE_WAN_TIER_OUT")
            .expect("set SCENEWORKS_CANDLE_WAN_TIER_OUT to the output tier dir");
        let bits: usize = std::env::var("SCENEWORKS_CANDLE_WAN_BITS")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(4);
        assert!(
            bits == 4 || bits == 8,
            "SCENEWORKS_CANDLE_WAN_BITS must be 4 or 8"
        );
        super::build_candle_wan_tier(Path::new(&diffusers), Path::new(&out), bits)
            .expect("build candle wan tier");
        eprintln!("[[CANDLE-TIER]] done: q{bits} tier at {out}");
    }
}
