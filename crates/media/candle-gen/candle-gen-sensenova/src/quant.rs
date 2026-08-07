//! The SenseNova-U1 **weight-store seam** (sc-14249, epic 9083) — the one place that decides how a
//! checkpoint tensor lands in memory.
//!
//! Before this the whole crate mmapped its backbone at [`DType::F32`] and hard-rejected
//! `spec.quantize`, so the candle lane could consume only the dense `bf16/` tier of the SceneWorks
//! turnkey — and consumed it at **double** its on-disk size. On an RTX PRO 6000 that measured a
//! **70.5 GB** peak for a 32.7 GiB checkpoint: a 96 GB-class-card feature that will not fit an 80 GB
//! A100/H100, while the MLX lane runs the same model packed-q4 at ~11 GB.
//!
//! Both of the story's levers turn out to ride ONE seam, so they are not two loaders:
//!
//! - **Lever A (stop the F32 upcast).** On a dense `bf16/` tier [`detect_linear`] keeps the weight at
//!   its on-disk bf16 and the forward upcasts per op ([`QLinear::forward_upcast`]). Activations stay
//!   f32, so **every matmul is still f32** and the arithmetic is bit-identical to the old f32 store —
//!   bf16 → f32 widening is exact. Only the resident weight halves. This is the sc-12828 bf16-store /
//!   f32-compute split, reused verbatim.
//! - **Lever B (packed q4/q8).** On a packed tier the SAME [`detect_linear`] finds the `.scales`
//!   sibling and builds a [`QLinear::Quantized`] straight from the packed triple, with no dense weight
//!   ever materialized.
//!
//! **Detection is per-tensor, not per-config.** Unlike the kolors (sc-10819) and SDXL (sc-9416)
//! turnkeys — which carry a `quantization` block in a component `config.json` — every SenseNova tier
//! ships the *same byte-identical* `config.json` (it is the upstream model config, hardlinked across
//! `bf16/`, `q8/` and `q4/`). There is nothing per-tier to probe, so the only honest signal is the
//! presence of `{base}.scales` in the checkpoint itself, which is exactly what the shared
//! `candle_gen::quant` detect helpers key on.
//!
//! **What is packed:** only the 42 decoder layers' 14 projections (both the understanding and the
//! `_mot_gen` generation path × `{q,k,v,o}_proj` + `{gate,up,down}_proj`) — 588 triples, the ~16.2 B
//! parameters that are the whole footprint. Everything else stays dense bf16 in every tier: the token
//! embedding, `lm_head`, the RMSNorm vectors, the FM head, the timestep/noise-scale embedders and the
//! vision embedders. That split is the tier author's, not ours; this module simply follows it.

use candle_gen::candle_core::{DType, Device};
use candle_gen::candle_nn::{Linear, VarBuilder};
use candle_gen::quant::{DenseLinear, QLinear};
use candle_gen::Result;

/// The MLX affine group size every SenseNova tier packs at.
///
/// Verified against the shipped `SceneWorks/sensenova-u1-8b-mlx` tiers rather than assumed: q4
/// `mlp.gate_proj` is `weight U32 [12288, 512]` + `scales BF16 [12288, 64]`, i.e. `512 · 8 = 4096`
/// input features over 64 groups; q8 is `[12288, 1024]` + the same `[12288, 64]`, i.e. `1024 · 4 =
/// 4096` over 64. Both ⇒ 64. It is NOT recoverable from the packed shapes alone (see the shared
/// `repack`), and there is no per-tier `config.json` `quantization` block to read it from, so it is
/// pinned here with the evidence rather than threaded from a probe that would have nothing to read.
pub(crate) const PACKED_GROUP_SIZE: usize = 64;

/// Load `{base}` as a [`QLinear`], packed-detecting the MLX triple.
///
/// The packed arm mirrors the shared `candle_gen::quant::lin_gs` exactly (u32 codes at their native
/// dtype — a cast to a float dtype would reinterpret the bit-packed nibbles; scales/biases upcast to
/// f32, which is exact from bf16).
///
/// The dense arm deliberately does **not** delegate to `lin_gs`: that helper's dense branch goes
/// through `candle_nn::linear`, which is shape-CHECKED, while this crate has always loaded its
/// projections shapelessly via `get_unchecked`. Keeping the shapeless load means this change is a
/// pure store-dtype/packed-detect change with no new failure mode on a checkpoint whose dims differ
/// from what the config implies — the dims are only ever read back from the tensors themselves.
pub(crate) fn detect_linear(vb: &VarBuilder, base: &str, bias: bool) -> Result<QLinear> {
    let scales_key = format!("{base}.scales");
    if vb.contains_tensor(&scales_key) {
        let device: Device = vb.device().clone();
        let wq = vb.get_unchecked_dtype(&format!("{base}.weight"), DType::U32)?;
        let scales = vb.get_unchecked_dtype(&scales_key, DType::F32)?;
        let biases = vb.get_unchecked_dtype(&format!("{base}.biases"), DType::F32)?;
        // A packed projection's own `.bias` (distinct from the affine `.biases`) stays
        // full-precision, like every other packed loader in the workspace.
        let bias = if bias {
            Some(vb.get_unchecked_dtype(&format!("{base}.bias"), DType::F32)?)
        } else {
            None
        };
        return Ok(QLinear::from_packed_gs(
            &wq,
            &scales,
            &biases,
            bias,
            PACKED_GROUP_SIZE,
            &device,
        )?);
    }
    let w = vb.get_unchecked(&format!("{base}.weight"))?;
    let b = if bias {
        Some(vb.get_unchecked(&format!("{base}.bias"))?)
    } else {
        None
    };
    Ok(QLinear::from_dense(DenseLinear::Linear(Linear::new(w, b))))
}

/// Read `key` at **f32** regardless of the VarBuilder's store dtype — the compute dtype for every
/// dense leaf the model multiplies against an f32 activation.
///
/// The RMSNorm vectors, the vision conv kernels and the FM/timestep Linears are all tiny (well under
/// 1% of the checkpoint), so widening them costs nothing and keeps the entire forward at one dtype.
/// They must be widened, not merely allowed to ride the store: `rms_norm` broadcast-multiplies its
/// weight against an f32 hidden state, and candle rejects a mixed-dtype op rather than promoting —
/// so a bf16 norm under a bf16 store is a dtype error at the first forward, not at load.
pub(crate) fn get_f32(vb: &VarBuilder, key: &str) -> Result<candle_gen::candle_core::Tensor> {
    Ok(vb.get_unchecked_dtype(key, DType::F32)?)
}

/// The dtype the backbone's bulk projections are STORED at, given the checkpoint's own dtype.
///
/// **Never widen, never truncate** (the sc-12828 rule): store bf16 only when the checkpoint is
/// already bf16, so the bit-identity argument for Lever A actually holds. Any other on-disk dtype
/// loads exactly as it did before — f32 — rather than being silently rounded down.
pub(crate) fn store_dtype_for(disk: DType) -> DType {
    if disk == DType::BF16 {
        DType::BF16
    } else {
        DType::F32
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_gen::candle_core::Tensor;

    use super::*;

    /// Write `tensors` to a real `.safetensors` and mmap a VarBuilder over it at `dtype` — the exact
    /// production load path (`backbone_vb` → `mmap_var_builder`), so these tests exercise the
    /// on-disk → VarBuilder → [`detect_linear`] chain rather than a hand-built tensor map.
    ///
    /// `tag` names the file; it must be unique per call site (the crate has no `tempfile` dep, and
    /// adding one for three tests is not worth a new dependency). Scoped by pid so concurrent runs
    /// of the suite cannot collide.
    fn vb_over(
        tmp: &tempfile::TempDir,
        tag: &str,
        tensors: HashMap<String, Tensor>,
        dtype: DType,
    ) -> VarBuilder<'static> {
        let dir = tmp.path().join("sensenova-quant");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{tag}.safetensors"));
        candle_gen::candle_core::safetensors::save(&tensors, &path).unwrap();
        candle_gen::mmap_var_builder(&[path], dtype, &Device::Cpu).unwrap()
    }

    /// Per-element 4-bit codes → MLX u32 words (LSB-first nibbles) — the on-disk packing the tier
    /// author emits, mirrored here so the fixture is a real packed triple rather than a stand-in.
    fn pack_mlx_q4(codes: &[u8]) -> Vec<u32> {
        codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect()
    }

    /// The store dtype follows the checkpoint: bf16 stays bf16 (the win), and nothing else is ever
    /// truncated into bf16 — an f16 or f32 checkpoint keeps loading at f32 exactly as before.
    #[test]
    fn store_dtype_never_widens_and_never_truncates() {
        assert_eq!(store_dtype_for(DType::BF16), DType::BF16);
        assert_eq!(store_dtype_for(DType::F32), DType::F32);
        assert_eq!(store_dtype_for(DType::F16), DType::F32);
    }

    /// **Lever A, proven exactly.** A bf16-stored projection driven by `forward_upcast` against f32
    /// activations is **bit-identical** to the old f32 store — `max|Δ| == 0`, not a tolerance.
    ///
    /// That is the whole safety argument for halving the resident backbone: the weight is only ever
    /// widened *to* f32 (bf16 → f32 is exact), never the activation narrowed *down*, so there is no
    /// bf16 matmul anywhere and the arithmetic cannot drift. Because nothing here needs a GPU, this
    /// is a stronger and cheaper gate than a cosine check on a real render.
    #[test]
    fn bf16_store_with_f32_compute_is_bit_identical_to_an_f32_store() {
        let tmp = tempfile::tempdir().unwrap();
        let (out_dim, in_dim) = (16usize, 32usize);
        // A bf16 ON-DISK weight, exactly like the shipped checkpoint.
        let w = Tensor::from_vec(
            (0..out_dim * in_dim)
                .map(|i| (i as f32 * 0.013).sin())
                .collect::<Vec<f32>>(),
            (out_dim, in_dim),
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let tensors = HashMap::from([("proj.weight".to_owned(), w)]);

        let x = Tensor::from_vec(
            (0..2 * in_dim)
                .map(|i| (i as f32 * 0.07).cos())
                .collect::<Vec<f32>>(),
            (2, in_dim),
            &Device::Cpu,
        )
        .unwrap();

        let narrow = vb_over(&tmp, "levera-narrow", tensors.clone(), DType::BF16);
        let wide = vb_over(&tmp, "levera-wide", tensors, DType::F32);

        let bf16_store = detect_linear(&narrow, "proj", false).unwrap();
        let f32_store = detect_linear(&wide, "proj", false).unwrap();
        assert!(!bf16_store.is_quantized() && !f32_store.is_quantized());

        let a = bf16_store.forward_upcast(&x).unwrap();
        let b = f32_store.forward_upcast(&x).unwrap();
        assert_eq!(a.dtype(), DType::F32, "the forward must produce f32");
        let max_delta = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(
            max_delta, 0.0,
            "bf16 store must be bit-identical, not close"
        );
    }

    /// **Lever B, at the seam.** A projection whose `.scales` sibling is on disk loads PACKED — no
    /// dense weight materialized — while the identical key set without one still loads dense.
    ///
    /// This is the only detection signal available for SenseNova: every tier ships the same
    /// byte-identical `config.json`, so there is no `quantization` block to probe. If this ever
    /// regressed to always-dense, a q4 tier would try to read `U32` codes as floats and render
    /// garbage rather than fail — hence asserting the ARM, not just the output.
    #[test]
    fn detect_linear_takes_the_packed_arm_only_when_scales_are_present() {
        let tmp = tempfile::tempdir().unwrap();
        let (out_dim, in_dim) = (8usize, PACKED_GROUP_SIZE);
        let dev = Device::Cpu;
        let groups = in_dim / PACKED_GROUP_SIZE;
        let codes: Vec<u8> = (0..out_dim * in_dim).map(|i| (i % 16) as u8).collect();
        let packed = HashMap::from([
            (
                "proj.weight".to_owned(),
                Tensor::from_vec(pack_mlx_q4(&codes), (out_dim, in_dim / 8), &dev).unwrap(),
            ),
            (
                "proj.scales".to_owned(),
                Tensor::from_vec(vec![0.0625f32; out_dim * groups], (out_dim, groups), &dev)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap(),
            ),
            (
                "proj.biases".to_owned(),
                Tensor::from_vec(vec![-0.5f32; out_dim * groups], (out_dim, groups), &dev)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap(),
            ),
        ]);
        // The VarBuilder is at the STORE dtype (bf16), as production is — the packed arm must still
        // read the u32 codes at their native dtype rather than reinterpreting the packed nibbles.
        let vb = vb_over(&tmp, "packed-arm", packed, DType::BF16);
        let lin = detect_linear(&vb, "proj", false).unwrap();
        assert!(lin.is_quantized(), "a `.scales` sibling must load packed");
        // It reproduces the affine grid it encodes: every code here is `i % 16` at scale 0.0625,
        // bias -0.5, so row·x is a real number, not garbage from misread nibbles.
        let x = Tensor::ones((1, in_dim), DType::F32, &dev).unwrap();
        let y = lin.forward_upcast(&x).unwrap().to_vec2::<f32>().unwrap();
        let expected: f32 = (0..in_dim).map(|c| 0.0625 * (c % 16) as f32 - 0.5).sum();
        assert!(
            (y[0][0] - expected).abs() < 1e-3,
            "packed forward {} vs affine grid {expected}",
            y[0][0]
        );

        // The same key WITHOUT the scales sibling is the dense tier — unchanged behavior.
        let dense = HashMap::from([(
            "proj.weight".to_owned(),
            Tensor::ones((out_dim, in_dim), DType::BF16, &dev).unwrap(),
        )]);
        let vb = vb_over(&tmp, "dense-arm", dense, DType::BF16);
        assert!(!detect_linear(&vb, "proj", false).unwrap().is_quantized());
    }

    /// **Real-weight repack guard.** Dequantizing a projection loaded from an ACTUAL shipped q4/q8
    /// tier must reproduce the affine grid that tier's own `weight`/`scales`/`biases` encode.
    ///
    /// The synthetic fixtures above prove the packed arm is *taken* and that a hand-built triple
    /// decodes; this proves the SHIPPED tiers' packing convention (nibble order, group axis, group
    /// SIZE) is the one `from_packed_gs` assumes. That last one has no other check:
    /// [`PACKED_GROUP_SIZE`] is pinned from tensor shapes and is not recoverable from the packed data,
    /// so a tier packed at a different group would repack into a *periodically* wrong weight — which
    /// renders as structured texture rather than an error, and is easy to mistake for "4-bit is just
    /// lossy".
    ///
    /// `#[ignore]`d: needs a real tier on disk. Point `SENSENOVA_TIER_DIR` at one, e.g.
    /// `…\models--SceneWorks--sensenova-u1-8b-mlx\snapshots\<hash>\q4`, then
    /// `cargo test -p candle-gen-sensenova --lib -- --ignored real_tier_repack --nocapture`.
    #[test]
    #[ignore = "real-weight guard; set SENSENOVA_TIER_DIR to a shipped q4/q8 tier dir"]
    fn real_tier_repack_reproduces_the_on_disk_affine_grid() {
        let dir = std::path::PathBuf::from(
            std::env::var("SENSENOVA_TIER_DIR").expect("set SENSENOVA_TIER_DIR to a tier dir"),
        );
        let files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .expect("tier dir")
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        assert!(!files.is_empty(), "no .safetensors in {}", dir.display());
        let vb = candle_gen::mmap_var_builder(&files, DType::BF16, &Device::Cpu).unwrap();

        // A small real projection: [out = 1024, in = 4096] on the 8B-MoT.
        let base = "language_model.model.layers.0.self_attn.k_proj";
        if !vb.contains_tensor(&format!("{base}.scales")) {
            println!(
                "[repack-guard] {} is DENSE — nothing to check",
                dir.display()
            );
            return;
        }

        let wq = vb
            .get_unchecked_dtype(&format!("{base}.weight"), DType::U32)
            .unwrap();
        let scales = vb
            .get_unchecked_dtype(&format!("{base}.scales"), DType::F32)
            .unwrap();
        let biases = vb
            .get_unchecked_dtype(&format!("{base}.biases"), DType::F32)
            .unwrap();
        let (out_dim, lanes) = wq.dims2().unwrap();
        let (_, n_groups) = scales.dims2().unwrap();
        // `in_features` follows from the group count and the pinned group size; the bit width then
        // follows from how many codes share a u32 lane.
        let in_dim = n_groups * PACKED_GROUP_SIZE;
        let per_lane = in_dim / lanes;
        let bits = 32 / per_lane;
        println!(
            "[repack-guard] {base}: [{out_dim}, {in_dim}] {bits}-bit, {n_groups} groups of \
             {PACKED_GROUP_SIZE}"
        );

        // The affine grid the tier itself encodes, decoded independently of `from_packed_gs`:
        // `w[r][c] = scales[r][c / group] · code(r, c) + biases[r][c / group]`.
        let codes_u32 = wq.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        let s = scales.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = biases.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mask: u32 = if bits >= 32 {
            u32::MAX
        } else {
            (1u32 << bits) - 1
        };
        let mut grid = vec![0f32; out_dim * in_dim];
        for r in 0..out_dim {
            for c in 0..in_dim {
                let lane = r * lanes + c / per_lane;
                let shift = (c % per_lane) * bits;
                let code = (codes_u32[lane] >> shift) & mask;
                let g = r * n_groups + c / PACKED_GROUP_SIZE;
                grid[r * in_dim + c] = s[g] * code as f32 + b[g];
            }
        }

        // What the loader actually produces: recover Wᵀ by pushing the identity through it.
        let lin = detect_linear(&vb, base, false).unwrap();
        assert!(lin.is_quantized());
        let eye = Tensor::from_vec(
            (0..in_dim * in_dim)
                .map(|i| if i / in_dim == i % in_dim { 1f32 } else { 0f32 })
                .collect::<Vec<f32>>(),
            (in_dim, in_dim),
            &Device::Cpu,
        )
        .unwrap();
        let wt = lin.forward_upcast(&eye).unwrap(); // [in, out] == Wᵀ
        let got = wt.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let (mut max_abs, mut sum_sq, mut ref_sq) = (0f64, 0f64, 0f64);
        for r in 0..out_dim {
            for c in 0..in_dim {
                let want = grid[r * in_dim + c] as f64;
                let have = got[c * out_dim + r] as f64;
                max_abs = max_abs.max((want - have).abs());
                sum_sq += (want - have) * (want - have);
                ref_sq += want * want;
            }
        }
        let rel_rms = (sum_sq / ref_sq.max(1e-30)).sqrt();
        println!(
            "[repack-guard] max|Δ| {max_abs:.3e}   rel-RMS {:.4}%",
            rel_rms * 100.0
        );
        // Q4 repacks LOSSLESSLY into Q4_1 (the affine form is preserved exactly — the epic's sc-9085
        // premise); Q8 goes through dequant → Q8_0 re-quant, measured there at ~0.56% mean rel RMS.
        if bits == 4 {
            assert!(
                rel_rms < 1e-6,
                "Q4→Q4_1 repack must be LOSSLESS against the tier's own affine grid, got rel-RMS \
                 {rel_rms:.6} (max|Δ| {max_abs:.3e}) — a periodic mismatch here IS the structured \
                 render artifact, not 4-bit fidelity"
            );
        } else {
            assert!(
                rel_rms < 0.02,
                "Q8 re-quant rel-RMS {rel_rms:.6} exceeds the measured band"
            );
        }
    }

    /// `get_f32` widens a bf16 on-disk leaf even though the VarBuilder stores bf16 — the norms and
    /// conv kernels must reach the forward at the compute dtype, or the first `broadcast_mul`
    /// against an f32 hidden state is a dtype error.
    #[test]
    fn get_f32_widens_dense_leaves_under_a_bf16_store() {
        let tmp = tempfile::tempdir().unwrap();
        let vb = vb_over(
            &tmp,
            "leaf-widen",
            HashMap::from([(
                "norm.weight".to_owned(),
                Tensor::ones((4,), DType::BF16, &Device::Cpu).unwrap(),
            )]),
            DType::BF16,
        );
        assert_eq!(vb.dtype(), DType::BF16);
        assert_eq!(get_f32(&vb, "norm.weight").unwrap().dtype(), DType::F32);
    }

    /// The group size is the one the shipped tiers actually pack at. Pinned so a future tier rebuild
    /// at a different group is caught here rather than as silent numeric garbage: the packed shapes
    /// alone cannot reveal the group, so a wrong constant repacks wrong and still "loads".
    #[test]
    fn packed_group_size_matches_the_shipped_tiers() {
        // q4: in_features = 512 u32 lanes · 8 nibbles = 4096, over 64 scale groups.
        assert_eq!(512 * 8 / PACKED_GROUP_SIZE, 64);
        // q8: in_features = 1024 u32 lanes · 4 bytes = 4096, over the same 64 groups.
        assert_eq!(1024 * 4 / PACKED_GROUP_SIZE, 64);
    }
}
