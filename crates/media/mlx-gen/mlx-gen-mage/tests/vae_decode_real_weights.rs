//! sc-14039: Mage-VAE decode parity against the CPU torch goldens, at three geometries.
//!
//! `#[ignore]`d — needs the real `microsoft/Mage-Flow` `vae/` weights (~0.35 GB; **bit-identical
//! across all six repos**) and the gitignored goldens under `../tools/golden/`. Run:
//!
//! ```sh
//! MAGE_SNAPSHOT=/path/to/microsoft--Mage-Flow/snapshots/<rev> \
//!   cargo test -p mlx-gen-mage --release --test vae_decode_real_weights -- --ignored --nocapture
//! ```
//!
//! Weights come from `$MAGE_SNAPSHOT` (required); goldens from `$MAGE_GOLDEN_DIR` or
//! `../tools/golden`. `MAGE_DEVICE=cpu` is **mandatory** for any re-dump — MPS dumps are silently
//! corrupt (sc-14250).
//!
//! ## Two oracles, deliberately
//!
//! | file | dtype | what it measures |
//! |---|---|---|
//! | `mage_flow_vae_golden_{size}.safetensors` | bf16 | production fidelity — the dtype the shipping pipeline runs (`load_from_repo` hard-codes it), so the residual is dominated by the *reference's* rounding |
//! | `mage_flow_vae_f32_{size}.safetensors` | f32 | this port's own error, with the reference's bf16 noise removed |
//!
//! The bf16 bundle is `dump_mage_flow_golden.py --stage vae` (sc-14036); the f32 bundle is
//! `dump_mage_vae_sizes.py`, which exists because torch's CPU bf16 kernels are effectively
//! single-threaded here — a 1024² bf16 dump takes tens of minutes and 2048 hours, while the whole
//! f32 sweep takes ~30 s:
//!
//! ```sh
//! MAGE_VAE_SIZES=256,1024,2048 PYTHONPATH=_vendor python3 tools/dump_mage_vae_sizes.py
//! ```
//!
//! ## Why 2048 is a named gate, not just "another size"
//!
//! sc-8228 (SeedVR2) shipped a VAE decoder that was correct at test sizes and **silently
//! corrupted large decodes**. This suite therefore checks 2048 explicitly, and not only with a
//! whole-tensor error bound: [`no_large_decode_corruption`] looks for the *shape* that class of
//! bug takes — error concentrated at the 32-latent attention-tile seams or the 16-pixel patch
//! seams, or one spatial region degrading relative to the rest.

use std::path::{Path, PathBuf};

use mlx_rs::ops::{abs, max, mean, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::image::decoded_to_image;
use mlx_gen::weights::Weights;
use mlx_gen_mage::vae::{
    self, MageVae, VaePart, DECODER_PREFIX, ENCODER_PREFIX, SKIPPED_DECODER_SUBTREES,
};

/// The geometries this suite gates, as the golden-file suffix.
///
/// Not all square, deliberately — **every square geometry hides a transposed h/w**, and the epic's
/// native-resolution range explicitly admits aspects up to 4:1:
///
/// * `256` — a 16×16 latent, which pads to one 32×32 `AttnBlock` tile, so this is the
///   replicate-padding case. 1024² and 2048 divide evenly and never reach it.
/// * `1024`, `2048` — the production sizes; 2048 is the DoD's large-decode gate.
/// * `992` — the witness for the **precision step at 2^20 pixels** described on
///   [`RefDtype::tolerance`]: 992² is 0.98 MP and sits *below* it, 1024² is 1.05 MP and sits
///   above. Keeping both makes that step visible in the suite's own output rather than only in a
///   comment, so a future MLX bump that moves it is noticed.
/// * `512x2048` — the 4:1 extreme.
/// * `768x1280` — a 48×80 latent; non-square, but its pads are **equal** (16, 16).
/// * `768x1152` — a 48×72 latent, pads **(16, 24)**. The **only** geometry here whose two
///   attention pads differ, and therefore the only one that fails on a `pad_h`/`pad_w` swap:
///   256² and 768×1280 pad (16,16), and 1024²/2048/512×2048 pad nothing. Do not drop it without
///   replacing it with another unequal-pad geometry.
const GEOMETRIES: &[&str] = &[
    "256", "992", "1024", "2048", "512x2048", "768x1280", "768x1152",
];

/// The geometry the DoD names for the large-decode corruption check.
const LARGE_GEOMETRY: &str = "2048";

/// Which reference dtype a golden bundle was dumped in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RefDtype {
    /// What the shipping pipeline runs.
    Bf16,
    /// The tight oracle.
    F32,
}

impl RefDtype {
    /// `(max_abs, mean_abs)` bound for a decode against this oracle.
    ///
    /// ## Calibrate against 1024², not 256²
    ///
    /// MLX's decode error is **not flat in resolution**: it steps by ~2× at exactly
    /// `H·W = 2^20` pixels, and the step follows *pixel count*, not side length. Bisected on
    /// `dec_from_latent`:
    ///
    /// | geometry | MP | mean_abs | max_abs |
    /// |---|---|---|---|
    /// | 768² | 0.59 | 0.000255 | 0.00122 |
    /// | 768×1152 | 0.88 | 0.000251 | 0.00119 |
    /// | 768×1280 | 0.98 | 0.000250 | 0.00104 |
    /// | 992² | 0.98 | 0.000250 | 0.00104 |
    /// | **1024²** | **1.05** | **0.000535** | **0.00923** |
    /// | 1056² | 1.12 | 0.000519 | 0.00904 |
    /// | 512×2048 | 1.05 | 0.000526 | 0.00541 |
    /// | 2048² | 4.19 | 0.000460 | 0.00689 |
    ///
    /// `512x2048` is the tell: same 1.05 MP as 1024², wildly different shape, same stepped error.
    /// It is **benign** — the error stays spatially uniform across the step (16-px phase spread
    /// ≤1.11×, 512-px block spread ≤1.12×) and it is not padding-driven (1056² pads and stays
    /// high, 992² pads and stays low) — so this is MLX kernel selection, not a defect.
    ///
    /// **But it means the 256² row flatters the headroom.** Every production geometry is ≥1 MP,
    /// i.e. above the step, so the bounds must be read against that regime: worst observed there
    /// is mean_abs **0.000735** against 0.0015 (**2.0×** headroom) and max_abs **0.0203** against
    /// 0.04 (**2.0×**) — not the ~9× the 256² numbers suggest. Anyone re-tuning these should use
    /// **1024² as the reference point**.
    ///
    /// ## What each bound actually catches
    ///
    /// They are not interchangeable, and it is **`max_abs` that bounds localized defects**:
    /// a 512×512 interior region wrong by 0.015 gives mean_abs 0.00137, which passes. `mean_abs`
    /// bounds *global* drift only. `max_abs` is noisier (it is a tail statistic, so it grows with
    /// sample count for a fixed error distribution) which is why the suite also reports the tail
    /// fraction — but it is the one that would catch a region going wrong. Localized corruption at
    /// 2048 has its own dedicated probes in [`no_large_decode_corruption`]; these two bounds are
    /// the whole-tensor backstop.
    ///
    /// Either way a structural fault — transposed conv weight, swapped 8192-channel ordering, zero
    /// instead of replicate padding, mis-ordered adaLN chunk — moves the output by O(0.1–1);
    /// `vae_decode_fixture.rs` demonstrates that separation directly and without weights.
    fn tolerance(self) -> (f32, f32) {
        match self {
            Self::Bf16 => (0.06, 6e-3),
            Self::F32 => (4e-2, 1.5e-3),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
        }
    }
}

fn golden_dir() -> PathBuf {
    match std::env::var("MAGE_GOLDEN_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/golden"),
    }
}

/// The `vae/`-bearing snapshot directory, from `$MAGE_SNAPSHOT`.
///
/// Deriving it from the HF cache is forbidden by the epic-13657 workspace guard
/// (`scripts/check-workspace.py`) — `inference` never reads a cache path itself — so this mirrors
/// the `mlx-gen-z-image` sibling and takes the path from the caller. Either a repo root (with a
/// `vae/` subdirectory) or the `vae/` directory itself works.
fn snapshot() -> PathBuf {
    let p = std::env::var("MAGE_SNAPSHOT").expect(
        "set MAGE_SNAPSHOT to a Mage-Flow snapshot directory (or its vae/ subdirectory) — the \
         vae/ weights are bit-identical across all six repos",
    );
    PathBuf::from(p)
}

/// Every golden bundle available for `size`, newest-oracle first.
fn load_goldens(size: &str) -> Vec<(RefDtype, Weights)> {
    let dir = golden_dir();
    let mut candidates = vec![
        (
            RefDtype::F32,
            dir.join(format!("mage_flow_vae_f32_{size}.safetensors")),
        ),
        (
            RefDtype::Bf16,
            dir.join(format!("mage_flow_vae_golden_{size}.safetensors")),
        ),
    ];
    if size == "256" {
        // The sc-14036 bundle predates the per-geometry naming.
        candidates.push((RefDtype::Bf16, dir.join("mage_flow_vae_golden.safetensors")));
    }

    let mut out: Vec<(RefDtype, Weights)> = Vec::new();
    for (dt, path) in candidates {
        if !path.exists() || out.iter().any(|(d, _)| *d == dt) {
            continue;
        }
        out.push((
            dt,
            Weights::from_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
        ));
    }
    out
}

/// The tightest available oracle for `size` — f32 when dumped, else bf16.
fn best_golden(size: &str) -> Option<(RefDtype, Weights)> {
    load_goldens(size).into_iter().next()
}

fn vae(fold: bool) -> MageVae {
    let dir = snapshot();
    if fold {
        vae::load(&dir, VaePart::Decode, Dtype::Float32).expect("load Mage-VAE")
    } else {
        let vae_dir = pick_vae_dir(&dir);
        let mut w = Weights::from_dir(&vae_dir).expect("read vae weights");
        w.cast_all(Dtype::Float32).unwrap();
        MageVae::from_weights(&w, Dtype::Float32, false).expect("build unfolded Mage-VAE")
    }
}

fn pick_vae_dir(dir: &Path) -> PathBuf {
    if dir.join("vae").is_dir() {
        dir.join("vae")
    } else {
        dir.to_path_buf()
    }
}

fn max_abs(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    max(abs(subtract(&a, &b).unwrap()).unwrap(), None)
        .unwrap()
        .item::<f32>()
}

fn mean_abs(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    let d = abs(subtract(&a, &b).unwrap()).unwrap();
    scalar_mean(&d)
}

/// `mean(x)` as an `f32`.
fn scalar_mean(x: &Array) -> f32 {
    mean(x, None).unwrap().item::<f32>()
}

/// Fraction of elements whose absolute error exceeds `thresh`.
///
/// `max_abs` is a tail statistic that necessarily grows with pixel count — 2048² has 64x the
/// samples of 256², so its worst single pixel is worse for the same error *distribution*. This
/// reports how heavy that tail actually is, so the bound below is set against evidence rather than
/// against the largest number that happened to pass.
fn frac_above(a: &Array, b: &Array, thresh: f32) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    let d = abs(subtract(&a, &b).unwrap()).unwrap();
    let over = d
        .gt(Array::from_f32(thresh))
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    scalar_mean(&over)
}

/// Write a decode to `$MAGE_VAE_DUMP_DIR` as a binary PPM, for eyeballing a decode when a parity
/// number looks odd. No-op unless the variable is set; PPM so the test needs no image encoder.
fn maybe_dump_ppm(name: &str, decoded: &Array) {
    let Ok(dir) = std::env::var("MAGE_VAE_DUMP_DIR") else {
        return;
    };
    let img = decoded_to_image(&decoded.as_dtype(Dtype::Float32).unwrap()).unwrap();
    let mut out = format!("P6\n{} {}\n255\n", img.width, img.height).into_bytes();
    out.extend_from_slice(&img.pixels);
    let path = PathBuf::from(dir).join(format!("{name}.ppm"));
    std::fs::write(&path, out).unwrap();
    println!("    wrote {}", path.display());
}

fn check_decode(
    size: &str,
    dt: RefDtype,
    g: &Weights,
    key_latent: &str,
    key_expected: &str,
    v: &MageVae,
) {
    let latent = g.require(key_latent).unwrap().clone();
    let want = g.require(key_expected).unwrap();

    mlx_rs::memory::reset_peak_memory();
    let got = v.decode(&latent).unwrap();
    mlx_rs::transforms::eval([&got]).unwrap();
    let peak = mlx_rs::memory::get_peak_memory() as f64 / 1e9;
    assert_eq!(got.shape(), want.shape(), "{size} {key_expected} geometry");
    maybe_dump_ppm(&format!("{key_expected}_{size}_{}", dt.label()), &got);

    let (mx, mn) = (max_abs(&got, want), mean_abs(&got, want));
    let (tol_mx, tol_mn) = dt.tolerance();
    println!(
        "  {size} [{}] {key_expected}: max_abs {mx:.6} (< {tol_mx})  mean_abs {mn:.6} (< {tol_mn})  \
         frac>10x_mean {:.3e}  peak {peak:.2} GB",
        dt.label(),
        frac_above(&got, want, 10.0 * mn)
    );
    assert!(
        mx < tol_mx,
        "{size}² [{}] {key_expected} max_abs {mx}",
        dt.label()
    );
    assert!(
        mn < tol_mn,
        "{size}² [{}] {key_expected} mean_abs {mn}",
        dt.label()
    );
}

/// The headline gate: decode a known reference latent at every dumped geometry.
///
/// Both decodes in each bundle are checked — `dec_from_synth` (a seeded synthetic latent) isolates
/// the decoder, while `dec_from_latent` (the real encoded image) is the in-distribution case a
/// broken decoder is most likely to *pass* on by accident.
#[test]
#[ignore = "needs real Mage-Flow vae/ weights + the gitignored goldens"]
fn decode_matches_the_torch_golden_at_every_geometry() {
    let v = vae(true);
    assert!(v.is_adaln_folded(), "production load must fold adaLN");

    let mut checked = 0;
    for &size in GEOMETRIES {
        let bundles = load_goldens(size);
        if bundles.is_empty() {
            println!("{size}: no golden, skipped");
            continue;
        }
        println!("{size}:");
        for (dt, g) in &bundles {
            check_decode(size, *dt, g, "synth_latent", "dec_from_synth", &v);
            check_decode(size, *dt, g, "enc_latent", "dec_from_latent", &v);
        }
        checked += 1;
    }
    assert_eq!(
        checked,
        GEOMETRIES.len(),
        "only {checked} of {} geometries had goldens — the DoD needs 256², 1024² AND 2048; \
         dump them with the command in this file's module docs",
        GEOMETRIES.len()
    );
}

/// **sc-8228 regression class: no silent corruption at a large decode.**
///
/// A whole-tensor error bound can hide a locally-corrupt region inside an otherwise-correct image.
/// This checks the two seams the architecture actually has, at the largest supported geometry:
///
/// * the CoD decoder's **32-latent attention tiles** (512 pixels apart), and
/// * the denoiser's **16-pixel patch** boundaries, where the unfold/fold round trip lands.
///
/// Both are compared against the golden, so the claim is "the port has no seam the reference
/// lacks" rather than "the image looks smooth" — the reference's own tile structure, if any, is
/// subtracted out.
#[test]
#[ignore = "needs real Mage-Flow vae/ weights + the 2048 golden"]
fn no_large_decode_corruption() {
    let size = LARGE_GEOMETRY;
    let Some((dt, g)) = best_golden(size) else {
        panic!(
            "no {size} golden in {} — dump it with the command in this file's module docs; \
             the DoD requires an explicit large-decode check",
            golden_dir().display()
        );
    };
    println!("{size} oracle: {}", dt.label());
    let v = vae(true);
    let latent = g.require("enc_latent").unwrap().clone();
    let want = g.require("dec_from_latent").unwrap();

    // Report MLX's own peak allocation across the decode. `ps`-style RSS does not see Metal's
    // unified buffers, so this is the figure sc-14046's fit gate / memory coefficients want. The
    // dominant intermediate is the per-pixel stream: [B·L, P², in+hidden_x+max_freqs²] — at 2048
    // that is 16384 × 256 × 99 f32 ≈ 1.66 GB, which is why it is worth having a number for.
    // Peak only. `get_active_memory()` sampled here would read ~0 and print as if it were a
    // measurement: MLX loads weights lazily, so nothing is resident until the first `eval`.
    mlx_rs::memory::reset_peak_memory();
    let got = v.decode(&latent).unwrap();
    mlx_rs::transforms::eval([&got]).unwrap();
    println!(
        "{size} decode: peak {:.2} GB",
        mlx_rs::memory::get_peak_memory() as f64 / 1e9
    );

    let report = corruption_report(&got, want);
    report.print(size);
    report.assert_clean(size);
}

/// The per-region error statistics [`no_large_decode_corruption`] gates on.
struct CorruptionReport {
    overall: f32,
    /// All four quadrant means, so the verdict can compare the worst against the *others* rather
    /// than against a global mean that already contains the defect.
    quadrants: [f32; 4],
    /// `(label, seam mean, interior mean)` per architectural stride, **per axis**.
    seams: Vec<(String, f32, f32)>,
    /// `max / median` over 512×512 blocks — catches a single wrong interior tile, which neither
    /// the quadrant nor the seam probes see.
    block_max_over_median: f32,
    lo: f32,
    hi: f32,
    sd: f32,
}

impl CorruptionReport {
    /// Does the seam probe whose label contains `needle` itself flag a defect? Used where the
    /// point is that a *specific* probe has coverage, independent of which probe happens to win
    /// the race inside [`Self::verdict`] (which returns on the first one that fires).
    fn seam_fires(&self, needle: &str) -> bool {
        self.seams
            .iter()
            .any(|(label, s, i)| label.contains(needle) && *s >= 3.0 * i.max(1e-9))
    }

    fn worst_quadrant(&self) -> f32 {
        self.quadrants.iter().copied().fold(0.0f32, f32::max)
    }

    /// Median of the three quadrants that are not the worst — the baseline a localised defect is
    /// measured against.
    fn quiet_quadrant_median(&self) -> f32 {
        let mut v = self.quadrants;
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[1] // median of the lower three
    }

    fn print(&self, size: &str) {
        println!("{size} overall mean_abs {:.6}", self.overall);
        println!(
            "  quadrants {:?} -> worst {:.6} vs quiet-median {:.6} (ratio {:.2})",
            self.quadrants.map(|q| (q * 1e6).round() / 1e6),
            self.worst_quadrant(),
            self.quiet_quadrant_median(),
            self.worst_quadrant() / self.quiet_quadrant_median().max(1e-9)
        );
        for (label, s, i) in &self.seams {
            println!("  {label}: seam {s:.6} vs interior {i:.6}");
        }
        println!(
            "  512px blocks: max/median {:.2}",
            self.block_max_over_median
        );
        println!(
            "  range [{:.4}, {:.4}] std {:.4}",
            self.lo, self.hi, self.sd
        );
    }

    /// `Err(reason)` if any probe fires. Split out from the assertions so
    /// [`corruption_probes_reject_a_seeded_defect`] can drive the same code on a deliberately
    /// broken tensor — the DoD claim "no large-decode corruption" is only worth anything if the
    /// probes demonstrably reject corruption.
    fn verdict(&self) -> Result<(), String> {
        // Compared against the QUIET quadrants, not `overall`. `overall` includes the defect, so a
        // perfectly isolated bad quadrant can never exceed 4x it however wrong it is -- that
        // capped this probe's sensitivity at a 3.0 threshold it could only just reach.
        let (worst, quiet) = (self.worst_quadrant(), self.quiet_quadrant_median());
        if worst >= 3.0 * quiet.max(1e-9) {
            return Err(format!(
                "one quadrant ({worst}) is >=3x the median of the other three ({quiet})"
            ));
        }
        for (label, s, i) in &self.seams {
            if *s >= 3.0 * i.max(1e-9) {
                return Err(format!("{label}: seam ({s}) is >=3x the interior ({i})"));
            }
        }
        if self.block_max_over_median >= 3.0 {
            return Err(format!(
                "one 512px block is {}x the median block — localised interior corruption",
                self.block_max_over_median
            ));
        }
        if self.lo <= -1.2 || self.hi >= 1.2 {
            return Err(format!("decode left [-1,1]: [{}, {}]", self.lo, self.hi));
        }
        if self.sd <= 0.05 {
            return Err(format!("decode is nearly constant (std {})", self.sd));
        }
        Ok(())
    }

    fn assert_clean(&self, size: &str) {
        if let Err(why) = self.verdict() {
            panic!("{size} large-decode corruption: {why}");
        }
    }
}

/// Compare `got` against `want` and summarise where the error lives.
///
/// Every spatial probe is computed **on both axes**. The first version of this function indexed
/// only axis 2, which made it blind to column-direction defects — a vertical attention-tile seam
/// scored `seam == interior` exactly, and a bad replicate-pad on the right edge scored 0.000000.
/// Its self-test could not reveal that, because the self-test only injected row defects.
fn corruption_report(got: &Array, want: &Array) -> CorruptionReport {
    let g32 = got.as_dtype(Dtype::Float32).unwrap();
    let w32 = want.as_dtype(Dtype::Float32).unwrap();
    let err = abs(subtract(&g32, &w32).unwrap()).unwrap();
    let overall = scalar_mean(&err);

    let sh = g32.shape();
    let (h, w) = (sh[2], sh[3]);

    let pick = |a: &Array, idx: &[i32], axis: i32| -> Array {
        a.take_axis(Array::from_slice(idx, &[idx.len() as i32]), axis)
            .unwrap()
    };

    // Per-quadrant.
    let mut quadrants = [0.0f32; 4];
    for (n, (qy, qx)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
        let rows: Vec<i32> = (qy * h / 2..(qy + 1) * h / 2).collect();
        let cols: Vec<i32> = (qx * w / 2..(qx + 1) * w / 2).collect();
        quadrants[n] = scalar_mean(&pick(&pick(&err, &rows, 2), &cols, 3));
    }

    // Seams vs interior, at the two architectural strides, on BOTH axes.
    let mut seams = Vec::new();
    for (label, stride) in [
        ("attention tile (32 latent = 512 px)", 512usize),
        ("patch (16 px)", 16),
    ] {
        for (axis, extent, dir) in [(2i32, h, "rows"), (3, w, "cols")] {
            // The trailing edge is part of the seam set: a bad replicate-pad shows up on the
            // LAST row/column, and `step_by(stride)` from 0 never reaches it (at 2048 it sampled
            // 0/512/1024/1536 only, scoring 0.000000 for a bottom-edge defect).
            let mut seam: Vec<i32> = (0..extent).step_by(stride).collect();
            if let Some(last) = (extent - 1).checked_sub(0) {
                if !seam.contains(&last) {
                    seam.push(last);
                }
            }
            let interior: Vec<i32> = (0..extent)
                .filter(|r| r % stride as i32 == stride as i32 / 2)
                .collect();
            if seam.is_empty() || interior.is_empty() {
                continue;
            }
            seams.push((
                format!("{label} {dir}"),
                scalar_mean(&pick(&err, &seam, axis)),
                scalar_mean(&pick(&err, &interior, axis)),
            ));
        }
    }

    // Per-512px-block max/median: a single wrong interior tile is invisible to the quadrant probe
    // (it dilutes into a quarter of the image) and to the seam probes (it is not on a seam).
    let block = 512i32;
    let mut blocks: Vec<f32> = Vec::new();
    let mut by = 0;
    while by < h {
        let mut bx = 0;
        while bx < w {
            let rows: Vec<i32> = (by..(by + block).min(h)).collect();
            let cols: Vec<i32> = (bx..(bx + block).min(w)).collect();
            blocks.push(scalar_mean(&pick(&pick(&err, &rows, 2), &cols, 3)));
            bx += block;
        }
        by += block;
    }
    blocks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = blocks[blocks.len() / 2];
    let block_max_over_median = blocks.last().copied().unwrap_or(0.0) / median.max(1e-9);

    // And the decode must still be a picture: in range, and not a constant field.
    let mu = mean(&g32, None).unwrap();
    let centred = subtract(&g32, &mu).unwrap();
    let sq = mlx_rs::ops::multiply(&centred, &centred).unwrap();

    CorruptionReport {
        overall,
        quadrants,
        seams,
        block_max_over_median,
        lo: mlx_rs::ops::min(&g32, None).unwrap().item::<f32>(),
        hi: max(&g32, None).unwrap().item::<f32>(),
        sd: scalar_mean(&sq).sqrt(),
    }
}

/// Self-test: the corruption probes must **reject** a seeded defect.
///
/// Without this, `no_large_decode_corruption` passing would only mean "these ratios happened to be
/// small", which is also true of a probe that measures nothing. Each defect is injected into a
/// copy of the *golden* (so the un-corrupted control is exactly clean) and must be caught **by the
/// probe that is supposed to catch it** — asserted on the reason string, not just `is_err()`.
///
/// The row/column pairs matter. An earlier version of this test injected only row-direction
/// defects against a report that only indexed axis 2, so the two blind spots agreed with each
/// other and the suite looked healthy. Every seam defect here is therefore injected **both ways**.
#[test]
#[ignore = "needs the 2048 golden"]
fn corruption_probes_reject_a_seeded_defect() {
    let size = LARGE_GEOMETRY;
    let Some((_, g)) = best_golden(size) else {
        panic!("need the {size} golden");
    };
    let want = g.require("dec_from_latent").unwrap().clone();

    // Control: the golden against itself is clean by construction.
    let control = corruption_report(&want, &want);
    assert!(
        control.verdict().is_ok(),
        "control must be clean, got {:?}",
        control.verdict()
    );

    let sh = want.shape();
    let (h, w) = (sh[2], sh[3]);
    let corrupt = |mask: Vec<f32>| -> Array {
        let m = Array::from_slice(&mask, &[1, 1, h, w])
            .as_dtype(want.dtype())
            .unwrap();
        mlx_rs::ops::add(&want, m).unwrap()
    };
    // A realistic floor: without it every ratio is against ~0 and any defect trivially "fires".
    let baseline = 0.0005f32;

    let mut cases: Vec<(&str, &str, Vec<f32>)> = Vec::new();

    // (a) one corrupt quadrant — the SeedVR2 (sc-8228) failure shape.
    let mut m = vec![baseline; (h * w) as usize];
    for r in 0..h / 2 {
        for c in 0..w / 2 {
            m[(r * w + c) as usize] = 0.05;
        }
    }
    cases.push(("corrupt quadrant", "quadrant", m));

    // (b) patch seams — ROWS, then COLUMNS. The column case is the one the old row-only report
    // scored as seam == interior exactly.
    for (label, is_row) in [("patch seam rows", true), ("patch seam cols", false)] {
        let mut m = vec![baseline; (h * w) as usize];
        for i in (0..if is_row { h } else { w }).step_by(16) {
            for j in 0..if is_row { w } else { h } {
                let (r, c) = if is_row { (i, j) } else { (j, i) };
                m[(r * w + c) as usize] = 0.05;
            }
        }
        cases.push((label, "patch", m));
    }

    // (c) attention-tile seams — ROWS, then COLUMNS.
    for (label, is_row) in [("attn seam rows", true), ("attn seam cols", false)] {
        let mut m = vec![baseline; (h * w) as usize];
        for i in (0..if is_row { h } else { w }).step_by(512) {
            for j in 0..if is_row { w } else { h } {
                let (r, c) = if is_row { (i, j) } else { (j, i) };
                m[(r * w + c) as usize] = 0.05;
            }
        }
        cases.push((label, "attention tile", m));
    }

    // (e) one wrong interior 512×512 tile. Deliberately sized so the block probe is the *only*
    // one that can catch it: a 512² block is a quarter of a quadrant at 2048, so a defect `d`
    // against baseline `b` lifts its quadrant to `(d + 3b)/4` — for `3b <= d < 9b` the block ratio
    // clears 3x while the quadrant ratio does not. `d = 6b` sits in that window, so this case
    // proves the block probe adds coverage rather than merely agreeing with the quadrant probe.
    let mut m = vec![baseline; (h * w) as usize];
    for r in 512..1024 {
        for c in 512..1024 {
            m[(r * w + c) as usize] = 6.0 * baseline;
        }
    }
    cases.push(("interior 512px tile", "512px block", m));

    for (label, expect, mask) in cases {
        let r = corruption_report(&corrupt(mask), &want);
        let why = match r.verdict() {
            Err(why) => why,
            Ok(()) => {
                r.print(size);
                panic!("defect '{label}' was NOT detected");
            }
        };
        println!("  {label:22} -> rejected: {why}");
        assert!(
            expect.is_empty() || why.contains(expect),
            "defect '{label}' fired the wrong probe: expected a '{expect}' reason, got: {why}"
        );
    }

    // (d) a bad replicate-pad on the trailing edge — a 32-px band on the bottom, then the right.
    // The old report sampled rows 0/512/1024/1536 only and scored a bottom-edge defect 0.000000,
    // which is why the seam index set now includes the last row/column. Asserted on the seam
    // statistic itself rather than on the verdict string: a band this strong also trips the
    // quadrant probe, and `verdict()` returns on whichever fires first.
    for (label, is_bottom) in [("bottom edge band", true), ("right edge band", false)] {
        let mut m = vec![baseline; (h * w) as usize];
        for r in 0..h {
            for c in 0..w {
                let hit = if is_bottom { r >= h - 32 } else { c >= w - 32 };
                if hit {
                    m[(r * w + c) as usize] = 0.05;
                }
            }
        }
        let r = corruption_report(&corrupt(m), &want);
        assert!(r.verdict().is_err(), "defect '{label}' was NOT detected");
        let axis = if is_bottom { "rows" } else { "cols" };
        assert!(
            r.seam_fires(axis),
            "defect '{label}' did not trip the trailing-edge seam probe on {axis}: {:?}",
            r.seams
        );
        println!("  {label:22} -> rejected, trailing-edge seam probe fires on {axis}");
    }

    // (f) a constant field — the "decoder produced nothing" shape.
    let flat = mlx_rs::ops::zeros_dtype(sh, want.dtype()).unwrap();
    let why = corruption_report(&flat, &want)
        .verdict()
        .expect_err("a constant decode was not detected");
    assert!(why.contains("constant"), "wrong probe fired for (f): {why}");
}

/// Constant-folding adaLN at `t = 0` must be numerically identical on the **real** weights, not
/// just the tiny fixture — the production path frees ~18.6M parameters on the strength of this.
#[test]
#[ignore = "needs real Mage-Flow vae/ weights"]
fn adaln_folding_is_identical_on_real_weights() {
    let Some((_, g)) = best_golden("256") else {
        panic!("need a 256² golden for a latent to decode");
    };
    let latent = g.require("synth_latent").unwrap().clone();

    let folded = vae(true);
    let unfolded = vae(false);
    assert!(folded.is_adaln_folded());
    assert!(!unfolded.is_adaln_folded());

    let a = folded.decode(&latent).unwrap();
    let b = unfolded.decode(&latent).unwrap();
    let err = max_abs(&a, &b);
    println!("folded vs unfolded max_abs {err}");
    assert_eq!(err, 0.0, "adaLN folding changed the decode by {err}");
}

/// Decoding is deterministic and batch-invariant: the same latent decodes identically whether it
/// is run alone or as part of a batch. Anything stateful (a cached tile map, a shape-keyed buffer)
/// would show up here.
#[test]
#[ignore = "needs real Mage-Flow vae/ weights"]
fn decode_is_deterministic_and_batch_invariant() {
    let Some((_, g)) = best_golden("256") else {
        panic!("need a 256² golden for a latent to decode");
    };
    let latent = g.require("synth_latent").unwrap().clone();
    let v = vae(true);

    let once = v.decode(&latent).unwrap();
    let twice = v.decode(&latent).unwrap();
    assert_eq!(max_abs(&once, &twice), 0.0, "decode is not deterministic");

    let batched = v
        .decode(&mlx_rs::ops::concatenate_axis(&[latent.clone(), latent], 0).unwrap())
        .unwrap();
    let first = batched.take_axis(Array::from_int(0), 0).unwrap();
    let err = max_abs(&first.reshape(once.shape()).unwrap(), &once);
    println!("batch-of-2 vs batch-of-1 max_abs {err}");
    assert!(err < 1e-5, "batching changed the decode by {err}");
}

/// The state-dict layout constants are claims about the published checkpoint, so check them
/// against it rather than leaving them as decorative documentation.
///
/// In particular the two [`SKIPPED_DECODER_SUBTREES`] must **exist and be skipped**: they are the
/// discarded FLUX.2 VAE encoder side (`mage_vae.py:588`). This port loads by explicit name rather
/// than by walking the state dict, so skipping them is structural — but that only means anything
/// if they are actually present to be skipped, which is what this asserts.
#[test]
#[ignore = "needs real Mage-Flow vae/ weights"]
fn checkpoint_layout_matches_the_declared_prefixes() {
    let w = Weights::from_dir(pick_vae_dir(&snapshot())).expect("read vae weights");
    let keys: Vec<&str> = w.keys().collect();
    assert!(!keys.is_empty(), "no tensors in the vae/ directory");

    let count = |p: &str| keys.iter().filter(|k| k.starts_with(p)).count();

    let enc = count(&format!("{ENCODER_PREFIX}."));
    let dec = count(&format!("{DECODER_PREFIX}."));
    println!("{ENCODER_PREFIX}.*: {enc} tensors, {DECODER_PREFIX}.*: {dec} tensors");
    assert!(
        enc > 0,
        "no '{ENCODER_PREFIX}.*' keys — the encoder prefix is wrong (sc-14046)"
    );
    assert!(
        dec > 0,
        "no '{DECODER_PREFIX}.*' keys — the decoder prefix is wrong"
    );
    assert_eq!(
        enc + dec,
        keys.len(),
        "the checkpoint holds keys under neither declared prefix"
    );

    let mut skipped = 0;
    for sub in SKIPPED_DECODER_SUBTREES {
        let n = count(&format!("{DECODER_PREFIX}.{sub}"));
        println!("  skipped subtree '{sub}': {n} tensors");
        skipped += n;
    }
    assert!(
        skipped > 0,
        "none of {SKIPPED_DECODER_SUBTREES:?} exist — the skip rule is vacuous, so either the \
         checkpoint layout changed or the constant is wrong"
    );

    // And the model still builds without them, which is the point of the skip.
    let v = vae(true);
    assert!(v.is_adaln_folded());
}
