//! sc-12438 — empirical MLX large-output write probes (RUN ON METAL, MLX 0.31.2).
//!
//! `#[ignore]`d: each allocates multi-GB single-buffer arrays straddling the `i32::MAX`-element
//! boundary. They exist to establish — on *this* pinned runtime (MLX core 0.31.2 via pmetal-mlx-rs
//! `38e1cc17`) — the exact per-operation behaviour the tiling fix depends on. Run explicitly:
//!
//! ```text
//! cargo test -p mlx-gen --test mlx_write_bound_probe -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Probe rules (avoid the false greens that hid this): **position-dependent** data (`i % 251`);
//! compare only sub-bound slices / individual elements (never a reduction or an op *over* the oversized
//! tensor); read back with `as_slice` (the one large-array operation MLX does correctly) and compare on
//! the host after a single `eval`.
//!
//! Findings feed the design of `mlx_gen::vae_tiling::tiled_decode`'s safe path and the doc rewrite.

use mlx_rs::ops::{add, conv3d, multiply, pad};
use mlx_rs::Array;

const I32_MAX: i64 = i32::MAX as i64; // 2_147_483_647

/// Position-dependent value for flat index `i`. Range 0..=250, exact in f32, non-constant along every
/// axis so a scrambled output index lands on a different value with prob. ~250/251.
fn pv(i: i64) -> f32 {
    (i % 251) as f32
}

/// **The core claim.** `conv3d` 8→128 whose OUTPUT crosses `i32::MAX` on MLX 0.31.2 (the story's exact
/// `128×D×480×848` geometry, D=41 below / D=42 above). A pointwise (1×1×1) conv with a permutation
/// weight makes `out[..,co] = in[..,co%8]`, so every output element is predictable. Below the bound the
/// decode must be exact; above it, MLX 0.31.2's per-thread output offset overflows int32 (fixed upstream
/// by MLX PR #3524 in 0.32.0) → this is what justifies keeping the convolution-stage write guard while
/// pinned to 0.31.2.
#[test]
#[ignore = "sc-12438 heavy MLX conv write probe (~9 GB); run with --ignored on Metal"]
fn conv3d_8to128_output_across_i32max() {
    let (h, w): (i64, i64) = (480, 848);
    let (cin, cout): (i64, i64) = (8, 128);
    // weight [cout,1,1,1,cin]: W[co,ci] = 1 iff ci == co%cin → out[..,co] = in[..,co%cin].
    let mut wbuf = vec![0f32; (cout * cin) as usize];
    for co in 0..cout {
        wbuf[(co * cin + co % cin) as usize] = 1.0;
    }
    let weight = Array::from_slice(&wbuf, &[cout as i32, 1, 1, 1, cin as i32]);
    weight.eval().unwrap();

    let run = |d: i64| -> (i64, i64, i64) {
        let in_elems = d * h * w * cin;
        let xhost: Vec<f32> = (0..in_elems).map(pv).collect();
        let x = Array::from_slice(&xhost, &[1, d as i32, h as i32, w as i32, cin as i32]);
        x.eval().unwrap();
        drop(xhost);
        let y = conv3d(&x, &weight, (1, 1, 1), (0, 0, 0), (1, 1, 1), 1).unwrap();
        y.eval().unwrap();
        assert_eq!(
            y.shape(),
            &[1, d as i32, h as i32, w as i32, cout as i32],
            "conv output shape"
        );
        let ys = y.as_slice::<f32>();
        let out_elems = d * h * w * cout;
        let mut first_bad = -1i64;
        let mut checked = 0i64;
        for dd in [0i64, d / 2, d - 2, d - 1] {
            for hh in [0i64, h / 2, h - 1] {
                for ww in [0i64, w / 2, w - 1] {
                    for co in [0i64, 1, 7, 63, 64, 127] {
                        let off = ((dd * h + hh) * w + ww) * cout + co;
                        let want = pv(((dd * h + hh) * w + ww) * cin + co % cin);
                        let got = ys[off as usize];
                        checked += 1;
                        if (got - want).abs() > 1e-3 && first_bad < 0 {
                            first_bad = off;
                        }
                    }
                }
            }
        }
        (out_elems, first_bad, checked)
    };

    let (be, bf, bc) = run(41); // 2_136_145_920 < 2^31
    eprintln!(
        "[conv D=41 below] out_elems={be} (={:.4}× 2^31)  checked={bc}  first_bad_offset={bf}",
        be as f64 / I32_MAX as f64
    );
    let (ae, af, ac) = run(42); // 2_188_247_040 > 2^31
    eprintln!(
        "[conv D=42 above] out_elems={ae} (={:.4}× 2^31)  checked={ac}  first_bad_offset={af} \
         (={:.4}× 2^31)",
        ae as f64 / I32_MAX as f64,
        if af < 0 { 0.0 } else { af as f64 / I32_MAX as f64 }
    );

    assert!(be < I32_MAX && ae > I32_MAX, "geometry must bracket 2^31");
    assert!(
        bf < 0,
        "conv BELOW the bound must be exact — corruption at offset {bf} breaks the premise"
    );
    assert!(
        af >= 0,
        "conv ABOVE the bound was EXACT on this runtime — MLX 0.31.2 conv did NOT corrupt at \
         128×42×480×848. If so, the convolution write guard's premise no longer holds here and must \
         be re-derived from measurement."
    );
    assert!(
        af >= I32_MAX,
        "conv corruption began BELOW 2^31 (offset {af}) — not the int32 output-index boundary"
    );
}

/// The story's exact `pad` geometry: `[128,41,480,848]` → pad the frame axis `+1` → `[128,42,480,848]`
/// (output 2.188e9 > 2^31). Front data at `c≈126,127` lands above 2^31. Reports whether pad corrupts
/// the above-boundary region **on this runtime** — an earlier `[2200,1024,1024]` pad at 1.074× was
/// exact, so this checks the specific geometry the story flagged rather than assuming universality.
#[test]
#[ignore = "sc-12438 heavy MLX pad write probe (~18 GB); run with --ignored on Metal"]
fn pad_story_geometry_128x42x480x848() {
    let (c, d, h, w): (i64, i64, i64, i64) = (128, 41, 480, 848);
    let in_elems = c * d * h * w;
    assert!(in_elems < I32_MAX, "input must stay under the bound (from_slice)");
    let xhost: Vec<f32> = (0..in_elems).map(pv).collect();
    let x = Array::from_slice(&xhost, &[c as i32, d as i32, h as i32, w as i32]);
    x.eval().unwrap();
    drop(xhost);

    // Pad the D (frame-like) axis by (0,1) → zeros at d=41.
    let y = pad(&x, &[(0, 0), (0, 1), (0, 0), (0, 0)][..], None, None).unwrap();
    y.eval().unwrap();
    let dout = d + 1;
    assert_eq!(y.shape(), &[c as i32, dout as i32, h as i32, w as i32]);
    let out_elems = c * dout * h * w;
    assert!(out_elems > I32_MAX, "padded output must cross the bound");
    let ys = y.as_slice::<f32>();

    // expected Y[cc,dd,hh,ww] = X[cc,dd,hh,ww] if dd<d else 0.
    let mut first_bad = -1i64;
    let mut bad = 0i64;
    let mut checked = 0i64;
    for cc in [0i64, 64, 120, 125, 126, 127] {
        for dd in [0i64, d / 2, d - 1, d /* = pad row (zeros) */] {
            for &(hh, ww) in &[(0i64, 0i64), (h / 2, w / 2), (h - 1, w - 1)] {
                let off = ((cc * dout + dd) * h + hh) * w + ww;
                let want = if dd < d {
                    pv(((cc * d + dd) * h + hh) * w + ww)
                } else {
                    0.0
                };
                let got = ys[off as usize];
                checked += 1;
                if (got - want).abs() > 1e-3 {
                    bad += 1;
                    if first_bad < 0 {
                        first_bad = off;
                    }
                }
            }
        }
    }
    eprintln!(
        "[pad 128×42×480×848] out_elems={out_elems} (={:.4}× 2^31)  checked={checked}  bad={bad}  \
         first_bad_offset={first_bad} (={:.4}× 2^31)",
        out_elems as f64 / I32_MAX as f64,
        if first_bad < 0 {
            0.0
        } else {
            first_bad as f64 / I32_MAX as f64
        }
    );
    // Stable regression assertion: whatever pad does ABOVE the bound (geometry-dependent — an earlier
    // 2200×1024×1024 pad at 1.074× was exact), it must be exact for every sampled position BELOW 2^31.
    // This is the invariant `check_output_writable` relies on (pad is safe under the bound), and it holds
    // regardless of whether this particular geometry corrupts above it.
    assert!(
        !(0..I32_MAX).contains(&first_bad),
        "pad corrupted a BELOW-2^31 position (offset {first_bad}) — the write-index boundary is not at \
         2^31 and check_output_writable's 'pad is safe under the bound' premise is broken"
    );
}

/// Does creating / reshaping a >i32::MAX array through the mlx-rs wrappers stay intact? `from_slice`
/// asserts `len == shape.product::<i32>()` (overflows), and `reshape(&[-1])` infers the flat dim as
/// i32 — both cap MLX arrays at i32::MAX elements *through this binding* even though MLX core can hold
/// the buffer (a `pad` here builds a 2.188e9-element array and reads back). This is why the safe path
/// must NEVER flatten or `from_slice` the full oversized output.
#[test]
#[ignore = "sc-12438 MLX >i32::MAX reshape/contiguous probe (~9 GB); run with --ignored on Metal"]
fn reshape_and_contiguous_on_oversized_array() {
    let (c, d, h, w): (i64, i64, i64, i64) = (128, 41, 480, 848);
    let xhost: Vec<f32> = (0..c * d * h * w).map(pv).collect();
    let x = Array::from_slice(&xhost, &[c as i32, d as i32, h as i32, w as i32]);
    x.eval().unwrap();
    drop(xhost);
    let y = pad(&x, &[(0, 0), (0, 1), (0, 0), (0, 0)][..], None, None).unwrap();
    y.eval().unwrap();
    let out_elems = c * (d + 1) * h * w;
    assert!(out_elems > I32_MAX);

    // (a) as_slice on the oversized array — expected to WORK (read side is fine).
    let read_ok = std::panic::catch_unwind(|| {
        let s = y.as_slice::<f32>();
        s[out_elems as usize - 1] // touch the last element
    })
    .is_ok();

    // (b) elementwise 2·y+1 over the oversized array — expected to WORK (elementwise is safe).
    let two = Array::from_slice(&[2.0f32], &[1]);
    let one = Array::from_slice(&[1.0f32], &[1]);
    let elementwise_ok = std::panic::catch_unwind(|| {
        let z = add(multiply(&y, &two).unwrap(), &one).unwrap();
        z.eval().unwrap();
        let zs = z.as_slice::<f32>();
        // sample an above-2^31 element: c=127,d=0,h=0,w=0 → should be 2·pv(127*d*h*w)+1.
        let off = (127i64 * (d + 1) * h * w) as usize;
        (zs[off] - (2.0 * pv(127 * d * h * w) + 1.0)).abs() < 1e-3
    })
    .unwrap_or(false);

    // (c) reshape(&[-1]) — expected to FAIL/overflow (i32 flat-dim inference) on this binding.
    let reshape_flat_ok =
        std::panic::catch_unwind(|| y.reshape(&[-1]).map(|r| r.shape().to_vec())).is_ok();

    eprintln!(
        "[oversized array ops] as_slice_read_ok={read_ok}  elementwise_ok={elementwise_ok}  \
         reshape(-1)_ok={reshape_flat_ok}  (out_elems={out_elems} = {:.4}× 2^31)",
        out_elems as f64 / I32_MAX as f64
    );
    assert!(read_ok, "reading back an oversized array must work");
    assert!(
        elementwise_ok,
        "elementwise over an oversized array must be correct (the safe path relies on this)"
    );
    // reshape(-1) is expected to be unsupported here; we only record it (no hard assert either way).
}
