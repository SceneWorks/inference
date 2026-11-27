//! SW-6128 GPU parity of the fused strided RoPE and QK-RMSNorm + RoPE kernels
//! (`candle_quant_kernels::media_rope`, served through `candle_gen::rope`) against the shared
//! reference op chains, on CUDA device 0.
//!
//! Declared tolerance: **bit identity in every dtype** (f32, bf16, f16) and both math modes. The
//! kernel performs the reference chain's operations in the reference's order with non-contracting
//! intrinsics (`media_rope.cu`'s numerics contract): f32 math rounds once to the dtype exactly where
//! the chain's final `to_dtype` does; native math rounds each product / sum to the dtype where the
//! chain's half-precision `mul` / `sub` / `add` kernels do (a correctly rounded half op equals the
//! f32 op rounded once, because the f32 result of two half operands is exact or far from a tie);
//! the RMSNorm reproduces candle's `rmsnorm` warp reduction. The E2 bound (f32 max rel err
//! <= 1e-5) is therefore met with zero error; the tests print the max rel err they observed.
//!
//! The mutation checks show the comparison is sensitive: a flipped sine, the other pairing, a
//! shifted rotary dim or a dropped norm weight each break parity.
//!
//! `#[ignore]`d CUDA compute (registered in `scripts/ci/test_suites.json` for the dispatch-only
//! exhaustive suite):
//! `cargo test -p candle-gen --features cuda --test integration -- fused_rope_gpu:: --ignored`.

#![cfg(feature = "cuda")]

use candle_gen::candle_core::{DType, Device, Result, Tensor, Var};
use candle_gen::cuda_mempool::MemPool;
use candle_gen::fused_path::{media_fused_policy_guard, Attempt};
use candle_gen::rope::media_rope::{fused_rms_norm_rope, fused_rope};
use candle_gen::rope::{
    apply_rope, apply_rope_in_place, apply_rope_reference, freqs_cis_cos_sin, rms_norm_rope,
    rms_norm_rope_in_place, rms_norm_rope_reference, FreqsCisLayout, RopeLayout, RopeMath,
    RopePairing, RopeSpec, RMS_NORM_ROPE_PATH, ROPE_PATH,
};

fn cuda() -> Device {
    Device::new_cuda(0).expect("CUDA device 0")
}

/// Deterministic xorshift values in [-scale, scale).
fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * scale
        })
        .collect()
}

fn tensor(dims: &[usize], seed: u64, scale: f32, dev: &Device) -> Result<Tensor> {
    Tensor::from_vec(values(dims.iter().product(), seed, scale), dims, dev)
}

fn host_f32(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

/// `(mismatching elements, max relative error)` of `got` against `want` (NaN-free inputs).
fn compare(got: &Tensor, want: &Tensor) -> Result<(usize, f32)> {
    assert_eq!(got.dims(), want.dims());
    assert_eq!(got.dtype(), want.dtype());
    let (g, w) = (host_f32(got)?, host_f32(want)?);
    let mut mismatches = 0;
    let mut max_rel = 0f32;
    for (a, b) in g.iter().zip(&w) {
        if a.to_bits() != b.to_bits() {
            mismatches += 1;
        }
        let rel = (a - b).abs() / b.abs().max(1e-6);
        max_rel = max_rel.max(rel);
    }
    Ok((mismatches, max_rel))
}

fn fused(a: Attempt<Tensor>, what: &str) -> Tensor {
    match a {
        Attempt::Fused(t) => t,
        Attempt::Reference(r) => panic!("{what}: the fused kernel declined: {r}"),
    }
}

/// `cos` / `sin` `[bt, s, w]` (f32) of angles up to ~±60 rad (large positions exercise every
/// quadrant).
fn tables(bt: usize, s: usize, w: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let angles = tensor(&[bt, s, w], 77, 60.0, dev)?;
    Ok((angles.cos()?, angles.sin()?))
}

/// One parity case: `x` (a possibly strided view), tables, spec.
struct Case {
    name: String,
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    spec: RopeSpec,
}

/// Geometries: contiguous BSHD, the permuted BHSD view of BSHD memory (no copy), q and k
/// narrowed out of a fused `[b, s, 3, h, d]` QKV buffer (gapped, offset views), partial rotary
/// with an odd pass-through tail, full-width per-lane tables, a per-batch table, an offset (narrowed)
/// table, FLUX's strided `freqs_cis` matrix views, and Krea's production shape.
fn cases(dev: &Device, dtype: DType, pairing: RopePairing, math: RopeMath) -> Result<Vec<Case>> {
    let mut out = Vec::new();
    let spec = |layout| RopeSpec::new(pairing, layout).with_math(math);
    let tag = |name: &str| format!("{name} {dtype:?} {pairing:?} {math:?}");
    let (b, s, h, d) = (2usize, 37usize, 3usize, 64usize);
    let (cos, sin) = tables(1, s, d / 2, dev)?;

    let x = tensor(&[b, s, h, d], 1, 3.0, dev)?.to_dtype(dtype)?;
    out.push(Case {
        name: tag("contiguous bshd"),
        x: x.clone(),
        cos: cos.clone(),
        sin: sin.clone(),
        spec: spec(RopeLayout::Bshd),
    });
    out.push(Case {
        name: tag("permuted bhsd view"),
        x: x.transpose(1, 2)?,
        cos: cos.clone(),
        sin: sin.clone(),
        spec: spec(RopeLayout::Bhsd),
    });

    let qkv = tensor(&[b, s, 3, h, d], 2, 3.0, dev)?.to_dtype(dtype)?;
    for (part, name) in [(0usize, "q of fused qkv"), (1, "k of fused qkv")] {
        out.push(Case {
            name: tag(name),
            x: qkv.narrow(2, part, 1)?.squeeze(2)?,
            cos: cos.clone(),
            sin: sin.clone(),
            spec: spec(RopeLayout::Bshd),
        });
    }

    // Partial rotary: head 74, rotate 48, 26 lanes (13 items) pass through; one odd head too.
    let xp = tensor(&[1, s, h, 74], 3, 3.0, dev)?.to_dtype(dtype)?;
    let (pc, ps) = tables(1, s, 24, dev)?;
    out.push(Case {
        name: tag("partial rotary 48/74"),
        x: xp,
        cos: pc.clone(),
        sin: ps.clone(),
        spec: spec(RopeLayout::Bshd).with_rotary_dim(48),
    });
    let xo = tensor(&[1, s, 2, 51], 4, 3.0, dev)?.to_dtype(dtype)?;
    out.push(Case {
        name: tag("partial rotary 48/51 (odd tail)"),
        x: xo,
        cos: pc,
        sin: ps,
        spec: spec(RopeLayout::Bshd).with_rotary_dim(48),
    });

    // Full-width (per-lane) tables, as the HF copies build them.
    let (fc, fs) = tables(1, s, d, dev)?;
    out.push(Case {
        name: tag("full-width table"),
        x: x.clone(),
        cos: fc,
        sin: fs,
        spec: spec(RopeLayout::Bshd),
    });

    // A per-batch table, and an offset view into a longer table (a position offset).
    let (bc, bs) = tables(b, s, d / 2, dev)?;
    out.push(Case {
        name: tag("per-batch table"),
        x: x.clone(),
        cos: bc,
        sin: bs,
        spec: spec(RopeLayout::Bshd),
    });
    let (lc, ls) = tables(1, s + 11, d / 2, dev)?;
    out.push(Case {
        name: tag("offset table view"),
        x: x.clone(),
        cos: lc.narrow(1, 11, s)?,
        sin: ls.narrow(1, 11, s)?,
        spec: spec(RopeLayout::Bshd),
    });

    // FLUX: BHSD x, `[b, 1, s, d/2, 2, 2]` rotation matrices -> lane-stride-4 cos/sin views.
    let xb = tensor(&[1, h, s, d], 5, 3.0, dev)?.to_dtype(dtype)?;
    let m = Tensor::stack(&[&cos, &sin.neg()?, &sin, &cos], 3)?
        .reshape((1, s, d / 2, 2, 2))?
        .unsqueeze(1)?;
    let (mc, ms) = freqs_cis_cos_sin(&m, FreqsCisLayout::Matrix)?;
    out.push(Case {
        name: tag("freqs_cis matrix views"),
        x: xb,
        cos: mc,
        sin: ms,
        spec: spec(RopeLayout::Bhsd),
    });

    // Krea 2 at 1024^2: 512 caption + 64x64 image tokens, 48 q heads x 128 (and 12 k heads).
    let ks = 512 + 64 * 64;
    let (kc, ksn) = tables(1, ks, 64, dev)?;
    for (heads, name) in [(48usize, "krea q 4608x48x128"), (12, "krea k 4608x12x128")] {
        out.push(Case {
            name: tag(name),
            x: tensor(&[1, ks, heads, 128], 6 + heads as u64, 3.0, dev)?.to_dtype(dtype)?,
            cos: kc.clone(),
            sin: ksn.clone(),
            spec: spec(RopeLayout::Bshd),
        });
    }
    Ok(out)
}

fn all_modes() -> Vec<(DType, RopePairing, RopeMath)> {
    let mut v = Vec::new();
    for dtype in [DType::F32, DType::BF16, DType::F16] {
        for pairing in [RopePairing::Interleaved, RopePairing::SplitHalf] {
            for math in [RopeMath::F32, RopeMath::Native] {
                if dtype == DType::F32 && math == RopeMath::Native {
                    continue; // identical to F32 math for an f32 x
                }
                v.push((dtype, pairing, math));
            }
        }
    }
    v
}

/// Assert `got` equals `want` bit for bit, tracking the worst relative error seen.
fn assert_bit_identical(name: &str, got: &Tensor, want: &Tensor, worst: &mut f32) -> Result<()> {
    let (mismatches, max_rel) = compare(got, want)?;
    *worst = worst.max(max_rel);
    assert_eq!(
        mismatches,
        0,
        "{name}: {mismatches} of {} elements differ (max rel err {max_rel:e})",
        got.elem_count()
    );
    Ok(())
}

#[test]
#[ignore = "CUDA compute: run by the validation workflow, not CI"]
fn fused_rope_is_bit_identical_to_the_reference() -> Result<()> {
    let dev = cuda();
    let mut worst = 0f32;
    let mut n = 0;
    for (dtype, pairing, math) in all_modes() {
        for case in cases(&dev, dtype, pairing, math)? {
            let want = apply_rope_reference(&case.x, &case.cos, &case.sin, &case.spec)?;
            let (cos, sin) = canonical(&case);
            let got = fused(
                fused_rope(&case.x, &cos, &sin, &case.spec, false)?,
                &case.name,
            );
            assert_bit_identical(&case.name, &got, &want, &mut worst)?;
            n += 1;
        }
    }
    println!("fused rope: {n} cases bit-identical (max rel err {worst:e})");
    Ok(())
}

/// The tables in the `[bt, s, w]` form the kernel-level entry point takes (the public API
/// canonicalizes for its callers).
fn canonical(case: &Case) -> (Tensor, Tensor) {
    let fix = |t: &Tensor| match t.rank() {
        4 => t.squeeze(case.spec.layout.head_axis()).unwrap(),
        _ => t.clone(),
    };
    (fix(&case.cos), fix(&case.sin))
}

/// The in-place form writes the reference result into x's own buffer — and only into x's elements:
/// the rest of a fused QKV buffer (v, and the other projection) is untouched.
#[test]
#[ignore = "CUDA compute: run by the validation workflow, not CI"]
fn in_place_rope_writes_only_its_view() -> Result<()> {
    let dev = cuda();
    let mut worst = 0f32;
    for (dtype, pairing, math) in all_modes() {
        let spec = RopeSpec::new(pairing, RopeLayout::Bshd).with_math(math);
        let (b, s, h, d) = (2usize, 29usize, 4usize, 96usize);
        let (cos, sin) = tables(1, s, d / 2, &dev)?;
        let base = tensor(&[b, s, 3, h, d], 9, 3.0, &dev)?.to_dtype(dtype)?;
        for (part, rd) in [(0usize, d), (1, 64)] {
            let spec = spec.with_rotary_dim(rd);
            let buf = base.copy()?;
            let view = buf.narrow(2, part, 1)?.squeeze(2)?;
            let (c, sn) = (cos.narrow(2, 0, rd / 2)?, sin.narrow(2, 0, rd / 2)?);
            let want = apply_rope_reference(&view, &c, &sn, &spec)?;
            let name = format!("in-place part {part} rd {rd} {dtype:?} {pairing:?} {math:?}");
            let got = fused(fused_rope(&view, &c, &sn, &spec, true)?, &name);
            assert_bit_identical(&name, &got, &want, &mut worst)?;
            // The buffer now holds the result in that view, and the original elsewhere.
            let after = buf.narrow(2, part, 1)?.squeeze(2)?;
            assert_bit_identical(&name, &after, &want, &mut worst)?;
            for other in (0..3).filter(|p| *p != part) {
                let untouched = buf.narrow(2, other, 1)?;
                let orig = base.narrow(2, other, 1)?;
                assert_eq!(
                    compare(&untouched, &orig)?.0,
                    0,
                    "{name}: part {other} changed"
                );
            }
            // The fused QK-norm + RoPE in place on the same strided view.
            let w = (tensor(&[d], 10, 0.5, &dev)? + 1.0)?;
            let buf = base.copy()?;
            let view = buf.narrow(2, part, 1)?.squeeze(2)?;
            let want = rms_norm_rope_reference(&view, &w, 1e-6, &c, &sn, &spec)?;
            let got = fused(
                fused_rms_norm_rope(&view, &w, 1e-6, &c, &sn, &spec, true)?,
                &name,
            );
            assert_bit_identical(&format!("{name} rms"), &got, &want, &mut worst)?;
            let after = buf.narrow(2, part, 1)?.squeeze(2)?;
            assert_bit_identical(&format!("{name} rms buffer"), &after, &want, &mut worst)?;
            for other in (0..3).filter(|p| *p != part) {
                let untouched = buf.narrow(2, other, 1)?;
                let orig = base.narrow(2, other, 1)?;
                assert_eq!(
                    compare(&untouched, &orig)?.0,
                    0,
                    "{name} rms: part {other} changed"
                );
            }
        }
    }
    println!("in-place rope: bit-identical, writes confined to the view (max rel err {worst:e})");
    Ok(())
}

#[test]
#[ignore = "CUDA compute: run by the validation workflow, not CI"]
fn fused_rms_norm_rope_is_bit_identical_to_rope_of_rmsnorm() -> Result<()> {
    let dev = cuda();
    let mut worst = 0f32;
    let mut n = 0;
    for (dtype, pairing, math) in all_modes() {
        for case in cases(&dev, dtype, pairing, math)? {
            let d = case.x.dim(3)?;
            // A learned-looking weight near 1 (Krea folds `scale + 1`).
            let w = (tensor(&[d], 31, 0.5, &dev)? + 1.0)?;
            for eps in [1e-6, 1e-5] {
                let want =
                    rms_norm_rope_reference(&case.x, &w, eps, &case.cos, &case.sin, &case.spec)?;
                let (cos, sin) = canonical(&case);
                let name = format!("{} eps {eps}", case.name);
                let got = fused(
                    fused_rms_norm_rope(&case.x, &w, eps, &cos, &sin, &case.spec, false)?,
                    &name,
                );
                assert_bit_identical(&name, &got, &want, &mut worst)?;
                // In place on a private copy of the same view geometry.
                let owned = case.x.copy()?;
                let got = fused(
                    fused_rms_norm_rope(&owned, &w, eps, &cos, &sin, &case.spec, true)?,
                    &name,
                );
                assert_bit_identical(&format!("{name} in place"), &got, &want, &mut worst)?;
                assert_bit_identical(&format!("{name} buffer"), &owned, &want, &mut worst)?;
                n += 1;
            }
        }
    }
    println!("fused rms_norm+rope: {n} cases bit-identical (max rel err {worst:e})");
    Ok(())
}

/// The in-place form allocates nothing of q/k size: the stream-ordered pool's high-water across
/// the call stays below one q (out of place, the output is exactly one q).
#[test]
#[ignore = "CUDA compute: run by the validation workflow, not CI"]
fn in_place_forms_allocate_no_q_sized_buffer() -> Result<()> {
    let dev = cuda();
    let pool = MemPool::device_current(0).expect("current CUDA pool");
    let _media = media_fused_policy_guard(Some(true));
    let _rope = ROPE_PATH.guard(Some(true));
    let _rms = RMS_NORM_ROPE_PATH.guard(Some(true));
    let (s, h, d) = (512 + 64 * 64, 48usize, 128usize);
    let (cos, sin) = tables(1, s, d / 2, &dev)?;
    let w = (tensor(&[d], 3, 0.5, &dev)? + 1.0)?;
    let q = tensor(&[1, s, h, d], 4, 3.0, &dev)?.to_dtype(DType::BF16)?;
    let q_bytes = (q.elem_count() * 2) as u64;
    let spec = RopeSpec::interleaved(RopeLayout::Bshd);

    let peak = |f: &mut dyn FnMut() -> Result<Tensor>| -> Result<u64> {
        dev.synchronize()?;
        let base = pool.used().expect("USED_MEM_CURRENT");
        assert!(pool.reset_used_high_water());
        let out = f()?;
        dev.synchronize()?;
        let high = pool.used_high().expect("USED_MEM_HIGH");
        drop(out);
        Ok(high.saturating_sub(base))
    };
    let start = (
        ROPE_PATH.thread_counts(),
        RMS_NORM_ROPE_PATH.thread_counts(),
    );
    let rope_in_place = peak(&mut || apply_rope_in_place(&q, &cos, &sin, &spec))?;
    let rms_in_place = peak(&mut || rms_norm_rope_in_place(&q, &w, 1e-6, &cos, &sin, &spec))?;
    let rope_out = peak(&mut || apply_rope(&q, &cos, &sin, &spec))?;
    let reference = peak(&mut || rms_norm_rope_reference(&q, &w, 1e-6, &cos, &sin, &spec))?;
    let d_rope = ROPE_PATH.thread_counts().since(&start.0);
    let d_rms = RMS_NORM_ROPE_PATH.thread_counts().since(&start.1);
    println!(
        "q {q_bytes} B: in-place rope peak {rope_in_place} B, in-place rms+rope {rms_in_place} B, \
         out-of-place rope {rope_out} B, reference rms+rope {reference} B; rope {d_rope}, rms {d_rms}"
    );
    assert_eq!((d_rope.fused, d_rope.reference_total()), (2, 0), "{d_rope}");
    assert_eq!((d_rms.fused, d_rms.reference_total()), (1, 0), "{d_rms}");
    assert!(
        rope_in_place < q_bytes,
        "in-place rope allocated {rope_in_place} B"
    );
    assert!(
        rms_in_place < q_bytes,
        "in-place rms+rope allocated {rms_in_place} B"
    );
    assert!(rope_out >= q_bytes, "the out-of-place output is one q");
    Ok(())
}

/// Mutations of the convention the parity is meant to pin must break it.
#[test]
#[ignore = "CUDA compute: run by the validation workflow, not CI"]
fn parity_catches_convention_mutations() -> Result<()> {
    let dev = cuda();
    let (s, h, d) = (33usize, 3usize, 64usize);
    let (cos, sin) = tables(1, s, d / 2, &dev)?;
    let x = tensor(&[1, s, h, d], 12, 3.0, &dev)?.to_dtype(DType::BF16)?;
    let w = (tensor(&[d], 13, 0.5, &dev)? + 1.0)?;
    let spec = RopeSpec::interleaved(RopeLayout::Bshd);
    let want = apply_rope_reference(&x, &cos, &sin, &spec)?;
    let want_rms = rms_norm_rope_reference(&x, &w, 1e-6, &cos, &sin, &spec)?;
    let differs = |got: &Tensor, want: &Tensor| compare(got, want).map(|(m, _)| m > 0);

    let neg_sin = sin.neg()?;
    let mutants: Vec<(&str, Tensor)> = vec![
        (
            "sin sign swapped",
            fused(fused_rope(&x, &cos, &neg_sin, &spec, false)?, "mutant"),
        ),
        (
            "split-half instead of interleaved",
            fused(
                fused_rope(
                    &x,
                    &cos,
                    &sin,
                    &RopeSpec::split_half(RopeLayout::Bshd),
                    false,
                )?,
                "mutant",
            ),
        ),
        (
            "cos and sin swapped",
            fused(fused_rope(&x, &sin, &cos, &spec, false)?, "mutant"),
        ),
    ];
    for (name, got) in &mutants {
        assert!(differs(got, &want)?, "mutation '{name}' passed parity");
    }
    let shifted = RopeSpec::interleaved(RopeLayout::Bshd).with_rotary_dim(d - 2);
    let got = fused(
        fused_rope(
            &x,
            &cos.narrow(2, 0, d / 2 - 1)?,
            &sin.narrow(2, 0, d / 2 - 1)?,
            &shifted,
            false,
        )?,
        "mutant",
    );
    assert!(
        differs(&got, &want)?,
        "mutation 'rotary dim - 2' passed parity"
    );
    let ones = Tensor::ones(d, DType::F32, &dev)?;
    let got = fused(
        fused_rms_norm_rope(&x, &ones, 1e-6, &cos, &sin, &spec, false)?,
        "mutant",
    );
    assert!(
        differs(&got, &want_rms)?,
        "mutation 'norm weight dropped' passed parity"
    );
    let got = fused(
        fused_rms_norm_rope(&x, &w, 1e-6, &cos, &neg_sin, &spec, false)?,
        "mutant",
    );
    assert!(
        differs(&got, &want_rms)?,
        "mutation 'rms+rope sin sign' passed parity"
    );
    println!("all {} rope mutations break parity", mutants.len() + 3);
    Ok(())
}

/// The public entry points report their path: fused on CUDA, `disabled` with the switch off (same
/// bits), and the typed reason for each input the kernels do not serve.
#[test]
#[ignore = "CUDA compute: run by the validation workflow, not CI"]
fn the_public_entry_points_report_their_path() -> Result<()> {
    let dev = cuda();
    let _media = media_fused_policy_guard(Some(true));
    let _rope = ROPE_PATH.guard(Some(true));
    let _rms = RMS_NORM_ROPE_PATH.guard(Some(true));
    let (s, h, d) = (21usize, 2usize, 32usize);
    let (cos, sin) = tables(1, s, d / 2, &dev)?;
    let w = (tensor(&[d], 14, 0.5, &dev)? + 1.0)?;
    let x = tensor(&[1, s, h, d], 15, 3.0, &dev)?.to_dtype(DType::BF16)?;
    let spec = RopeSpec::interleaved(RopeLayout::Bshd);
    let path = |p: &'static candle_gen::fused_path::FusedKernelPath,
                f: &mut dyn FnMut() -> Result<Tensor>| {
        let start = p.thread_counts();
        let out = f();
        (out, p.thread_counts().since(&start))
    };

    let (on, c) = path(&ROPE_PATH, &mut || apply_rope(&x, &cos, &sin, &spec));
    assert_eq!((c.fused, c.reference_total()), (1, 0), "{c}");
    ROPE_PATH.set_enabled(Some(false));
    let (off, c) = path(&ROPE_PATH, &mut || apply_rope(&x, &cos, &sin, &spec));
    assert_eq!(c.reference_for("disabled"), 1, "{c}");
    assert_eq!(compare(&on?, &off?)?.0, 0, "switch on and off agree");
    ROPE_PATH.set_enabled(Some(true));

    let (on, c) = path(&RMS_NORM_ROPE_PATH, &mut || {
        rms_norm_rope(&x, &w, 1e-6, &cos, &sin, &spec)
    });
    assert_eq!(c.fused, 1, "{c}");
    RMS_NORM_ROPE_PATH.set_enabled(Some(false));
    let (off, c) = path(&RMS_NORM_ROPE_PATH, &mut || {
        rms_norm_rope(&x, &w, 1e-6, &cos, &sin, &spec)
    });
    assert_eq!(c.reference_for("disabled"), 1, "{c}");
    assert_eq!(compare(&on?, &off?)?.0, 0, "switch on and off agree");
    RMS_NORM_ROPE_PATH.set_enabled(Some(true));

    // Unserved inputs: a gradient-tracking x (E9), f64, an in-place broadcast view, a head too wide
    // for the RMS kernel.
    let var = Var::from_tensor(&x.to_dtype(DType::F32)?)?;
    let (_, c) = path(&ROPE_PATH, &mut || {
        apply_rope(var.as_tensor(), &cos, &sin, &spec)
    });
    assert_eq!(c.reference_for("autograd"), 1, "{c}");
    let x64 = x.to_dtype(DType::F64)?;
    let (_, c) = path(&ROPE_PATH, &mut || apply_rope(&x64, &cos, &sin, &spec));
    assert_eq!(c.reference_for("dtype"), 1, "{c}");
    let bcast = x.narrow(2, 0, 1)?.broadcast_as((1, s, h, d))?;
    let before = host_f32(&x)?;
    let (out, c) = path(&ROPE_PATH, &mut || {
        apply_rope_in_place(&bcast, &cos, &sin, &spec)
    });
    assert_eq!(c.reference_for("layout"), 1, "{c}");
    assert_eq!(
        host_f32(&x)?,
        before,
        "an overlapping view is never written in place"
    );
    let want = apply_rope_reference(&bcast, &cos, &sin, &spec)?;
    assert_eq!(compare(&out?, &want)?.0, 0);
    let (wc, ws) = tables(1, s, 512, &dev)?;
    let wide = tensor(&[1, s, 1, 1024], 16, 3.0, &dev)?;
    let ww = Tensor::ones(1024, DType::F32, &dev)?;
    let (_, c) = path(&RMS_NORM_ROPE_PATH, &mut || {
        rms_norm_rope(&wide, &ww, 1e-6, &wc, &ws, &spec)
    });
    assert_eq!(c.reference_for("shape"), 1, "{c}");
    Ok(())
}

/// E3 / E8: the source compiles through the nvrtc seam once per device (a second lookup is the
/// cached module) and no entry point spills to per-thread local memory (candle-llm's
/// all-sources walk covers it too; this is the kernel's own focused check).
#[test]
#[ignore = "CUDA compute: run by the validation workflow, not CI"]
fn media_rope_kernels_compile_once_and_use_no_local_memory() -> Result<()> {
    use candle_gen::rope::media_rope::MEDIA_ROPE_SRC;
    let device = cuda();
    let dev = device.as_cuda_device()?;
    let kernel = MEDIA_ROPE_SRC.compiled(dev).expect("media rope compiles");
    let again = MEDIA_ROPE_SRC.compiled(dev).expect("cached");
    assert!(
        std::sync::Arc::ptr_eq(&kernel, &again),
        "one compiled module per device"
    );
    assert_eq!(MEDIA_ROPE_SRC.compile_attempts(dev), 1);
    let mut entries = kernel.entry_points().to_vec();
    entries.sort();
    assert_eq!(
        entries,
        [
            "media_rms_norm_rope_bf16",
            "media_rms_norm_rope_f16",
            "media_rms_norm_rope_f32",
            "media_rope_bf16",
            "media_rope_f16",
            "media_rope_f32",
        ]
    );
    for entry in &entries {
        let f = kernel.function(entry).expect("entry point");
        let local = f.local_size_bytes().expect("local size");
        println!(
            "{entry}: {local} B local, {} registers",
            f.num_regs().expect("registers")
        );
        assert_eq!(local, 0, "{entry} spills to local memory");
    }
    Ok(())
}
