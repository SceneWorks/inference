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
/// * `512x2048` — the 4:1 extreme.
/// * `768x1280` — a 48×80 latent, where the attention tiling pads **both axes by different
///   amounts** (48→64, 80→96). No square geometry produces that.
const GEOMETRIES: &[&str] = &["256", "1024", "2048", "512x2048", "768x1280"];

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
    /// The bf16 bounds are wide because the *reference* carries ~3 decimal digits through a
    /// ~30-layer decoder; the f32 bounds are tight because both sides are f32 and only MLX's
    /// reduced-precision Metal matmul/conv accumulation separates them. Either way a structural
    /// fault — transposed conv weight, swapped 8192-channel ordering, zero instead of replicate
    /// padding, mis-ordered adaLN chunk — moves the output by O(0.1–1); `vae_decode_fixture.rs`
    /// demonstrates that separation directly and without weights.
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

/// Tolerance for a decode against the golden.
///
/// **The reference runs the whole codec in bf16** (`dump_mage_flow_golden.py`:
/// `.to(torch.bfloat16)`), while this port runs f32, so the residual is dominated by the
/// *reference's* own rounding, not the port's — bf16 carries ~3 decimal digits, and the decoder is
/// ~30 layers deep with a `[-1, 1]` output range. Anything structural (a transposed conv weight, a
/// swapped 8192-channel ordering, zero instead of replicate padding, a mis-ordered adaLN chunk)
/// moves the output by O(0.1–1), two to three orders above this bound — the weights-free suite in
/// `vae_decode_fixture.rs` demonstrates that separation directly, at f32 on both sides.
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
    mlx_rs::memory::reset_peak_memory();
    let before = mlx_rs::memory::get_active_memory();
    let got = v.decode(&latent).unwrap();
    mlx_rs::transforms::eval([&got]).unwrap();
    let peak = mlx_rs::memory::get_peak_memory();
    println!(
        "{size} decode: peak {:.2} GB (model resident {:.2} GB)",
        peak as f64 / 1e9,
        before as f64 / 1e9
    );

    let report = corruption_report(&got, want);
    report.print(size);
    report.assert_clean(size);
}

/// The per-region error statistics [`no_large_decode_corruption`] gates on.
struct CorruptionReport {
    overall: f32,
    worst_quadrant: f32,
    /// `(label, seam mean, interior mean)` per architectural stride.
    seams: Vec<(&'static str, f32, f32)>,
    lo: f32,
    hi: f32,
    sd: f32,
}

impl CorruptionReport {
    fn print(&self, size: &str) {
        println!("{size} overall mean_abs {:.6}", self.overall);
        println!("  worst quadrant mean_abs {:.6}", self.worst_quadrant);
        for (label, s, i) in &self.seams {
            println!("  {label}: seam {s:.6} vs interior {i:.6}");
        }
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
        if self.worst_quadrant >= 3.0 * self.overall.max(1e-9) {
            return Err(format!(
                "one quadrant ({}) is >=3x the overall mean error ({})",
                self.worst_quadrant, self.overall
            ));
        }
        for (label, s, i) in &self.seams {
            if *s >= 3.0 * i.max(1e-9) {
                return Err(format!("{label}: seam ({s}) is >=3x the interior ({i})"));
            }
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
fn corruption_report(got: &Array, want: &Array) -> CorruptionReport {
    let g32 = got.as_dtype(Dtype::Float32).unwrap();
    let w32 = want.as_dtype(Dtype::Float32).unwrap();
    let err = abs(subtract(&g32, &w32).unwrap()).unwrap();
    let overall = scalar_mean(&err);

    let sh = g32.shape();
    let (h, w) = (sh[2], sh[3]);

    // Per-quadrant: a corrupt region shows up as one quadrant far worse than the rest. The
    // reference itself is spatially uniform, so any localisation is the port's.
    let mut worst_quadrant = 0.0f32;
    for (qy, qx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        let rows: Vec<i32> = (qy * h / 2..(qy + 1) * h / 2).collect();
        let cols: Vec<i32> = (qx * w / 2..(qx + 1) * w / 2).collect();
        let q = err
            .take_axis(Array::from_slice(&rows, &[rows.len() as i32]), 2)
            .unwrap()
            .take_axis(Array::from_slice(&cols, &[cols.len() as i32]), 3)
            .unwrap();
        worst_quadrant = worst_quadrant.max(scalar_mean(&q));
    }

    // Seam rows vs interior rows, at the two strides the architecture actually has.
    let mut seams = Vec::new();
    for (label, stride) in [
        ("attention tile (32 latent = 512 px)", 512usize),
        ("patch (16 px)", 16),
    ] {
        let seam: Vec<i32> = (0..h).step_by(stride).collect();
        let interior: Vec<i32> = (0..h)
            .filter(|r| r % stride as i32 == stride as i32 / 2)
            .collect();
        let seam_rows = err
            .take_axis(Array::from_slice(&seam, &[seam.len() as i32]), 2)
            .unwrap();
        let interior_rows = err
            .take_axis(Array::from_slice(&interior, &[interior.len() as i32]), 2)
            .unwrap();
        seams.push((label, scalar_mean(&seam_rows), scalar_mean(&interior_rows)));
    }

    // And the decode must still be a picture: in range, and not a constant field.
    let mu = mean(&g32, None).unwrap();
    let centred = subtract(&g32, &mu).unwrap();
    let sq = mlx_rs::ops::multiply(&centred, &centred).unwrap();

    CorruptionReport {
        overall,
        worst_quadrant,
        seams,
        lo: mlx_rs::ops::min(&g32, None).unwrap().item::<f32>(),
        hi: max(&g32, None).unwrap().item::<f32>(),
        sd: scalar_mean(&sq).sqrt(),
    }
}

/// Self-test: the corruption probes must **reject** a seeded defect.
///
/// Without this, `no_large_decode_corruption` passing would only mean "these ratios happened to be
/// small", which is also true of a probe that measures nothing. Each defect is injected into a
/// copy of the *golden* (so the un-corrupted control is exactly clean) and must be caught.
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

    // (a) one corrupt quadrant — the SeedVR2 (sc-8228) failure shape.
    let mut mask = vec![0.0f32; (h * w) as usize];
    for r in 0..h / 2 {
        for c in 0..w / 2 {
            mask[(r * w + c) as usize] = 0.05;
        }
    }
    let mask = Array::from_slice(&mask, &[1, 1, h, w])
        .as_dtype(want.dtype())
        .unwrap();
    let quad = mlx_rs::ops::add(&want, mask).unwrap();
    let r = corruption_report(&quad, &want);
    r.print(size);
    let why = r
        .verdict()
        .expect_err("a corrupt quadrant was not detected");
    assert!(why.contains("quadrant"), "wrong probe fired for (a): {why}");

    // (b) corrupt patch seams — a broken unfold/fold round trip.
    let mut seam_mask = vec![0.0f32; (h * w) as usize];
    for r in (0..h).step_by(16) {
        for c in 0..w {
            seam_mask[(r * w + c) as usize] = 0.05;
        }
    }
    let seam_mask = Array::from_slice(&seam_mask, &[1, 1, h, w])
        .as_dtype(want.dtype())
        .unwrap();
    let seamy = mlx_rs::ops::add(&want, seam_mask).unwrap();
    let r = corruption_report(&seamy, &want);
    let why = r
        .verdict()
        .expect_err("corrupt patch seams were not detected");
    assert!(why.contains("patch"), "wrong probe fired for (b): {why}");

    // (c) a constant field — the "decoder produced nothing" shape.
    let flat = mlx_rs::ops::zeros_dtype(sh, want.dtype()).unwrap();
    let r = corruption_report(&flat, &want);
    let why = r.verdict().expect_err("a constant decode was not detected");
    assert!(why.contains("constant"), "wrong probe fired for (c): {why}");
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
