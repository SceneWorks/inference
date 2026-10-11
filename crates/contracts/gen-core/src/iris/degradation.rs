//! The **second-order Real-ESRGAN degradation** the Iris-3B restoration recipe trains on (paper
//! arXiv 2610.09450v1 §5.3: "degraded on the fly with the second-order Real-ESRGAN pipeline at 4×,
//! including its sharpening step"), ported from the HYPIR reference implementation the paper
//! follows (`XPixelGroup/HYPIR` @ [`HYPIR_REVISION`]: `HYPIR/dataset/realesrgan.py`,
//! `HYPIR/dataset/batch_transform.py`, `HYPIR/dataset/utils.py`, `HYPIR/dataset/diffjpeg.py`,
//! `HYPIR/utils/degradation.py` — themselves BasicSR's `degradations.py` / `diffjpeg.py` /
//! `img_process_util.py`), with HYPIR's `configs/sd2_train.yaml` values as [`DegradationConfig`]'s
//! defaults.
//!
//! Host-side and backend-neutral (planar `[3, H, W]` f32 in `[0, 1]`, torch's layout), so the MLX
//! and Candle trainers degrade an item identically. Two halves:
//!
//! * [`DegradationDraw::draw`] — every random choice of one item (flips, the two blur kernels, the
//!   final sinc filter, the two resize/noise/JPEG stages, the stage-2 ordering) drawn from one
//!   positional [`HostRng`] key, so an item's degradation is a pure function of
//!   `(seed, epoch, item)` and a resumed run re-degrades it bit-identically.
//! * [`degrade`] — the deterministic pipeline that applies a draw: USM sharpening (the GT the model
//!   learns to produce), blur → resize → noise → JPEG, blur → resize → noise → {resize back + sinc,
//!   JPEG} in the drawn order, bicubic back to the GT size, 8-bit rounding. The per-pixel noise
//!   fields come from a [`NoiseSource`] (the positional stream in training; the reference's own
//!   draws in the parity fixture).
//!
//! **Native mapping of the reference's batch-level choices.** HYPIR draws the stage resize factor
//! and interpolation mode once per *batch* and then shuffles samples through a 256-entry training
//! pool (`_dequeue_and_enqueue`) whose only purpose is to diversify those per-batch choices across a
//! batch. Here every choice is drawn per *item*, which achieves that diversity directly and keeps
//! each item's degradation a function of its own key (the pool would couple an item's training pair
//! to the order of earlier batches). The parameter ranges and probabilities are the reference's.
//! The numpy / torch random streams themselves are not reproduced (the distribution law is).

use super::train::{mix_seed, HostRng};
use crate::{Error, Result};

/// `XPixelGroup/HYPIR` commit the degradation (and the rest of the restoration recipe) is traced
/// to.
pub const HYPIR_REVISION: &str = "b61d107c6cef38f01a93c7833558869731cfa8c1";

/// The padded side of every blur / sinc kernel (`pad_size = (21 - kernel_size) // 2`; the
/// reference hard-codes 21 regardless of its `blur_kernel_size` key).
pub const KERNEL_SIDE: usize = 21;
/// `self.kernel_range = [2 * v + 1 for v in range(3, 11)]` — odd sides 7..=21.
pub const KERNEL_SIZES: [usize; 8] = [7, 9, 11, 13, 15, 17, 19, 21];
/// `USMSharp(radius=50)` → an odd 51-tap Gaussian (`cv2.getGaussianKernel(51, 0)`, σ = 8).
pub const USM_RADIUS: usize = 51;
/// `USMSharp.forward(weight=0.5, threshold=10)`.
pub const USM_WEIGHT: f32 = 0.5;
pub const USM_THRESHOLD: f32 = 10.0;

// =============================================================================================
// Configuration
// =============================================================================================

/// The six blur-kernel families of `random_mixed_kernels`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelType {
    Iso,
    Aniso,
    GeneralizedIso,
    GeneralizedAniso,
    PlateauIso,
    PlateauAniso,
}

impl KernelType {
    pub const ALL: [KernelType; 6] = [
        KernelType::Iso,
        KernelType::Aniso,
        KernelType::GeneralizedIso,
        KernelType::GeneralizedAniso,
        KernelType::PlateauIso,
        KernelType::PlateauAniso,
    ];
    pub fn name(self) -> &'static str {
        match self {
            KernelType::Iso => "iso",
            KernelType::Aniso => "aniso",
            KernelType::GeneralizedIso => "generalized_iso",
            KernelType::GeneralizedAniso => "generalized_aniso",
            KernelType::PlateauIso => "plateau_iso",
            KernelType::PlateauAniso => "plateau_aniso",
        }
    }
    pub fn parse(s: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|k| k.name() == s)
            .ok_or_else(|| Error::Msg(format!("degradation: unknown kernel type {s:?}")))
    }
    fn isotropic(self) -> bool {
        matches!(
            self,
            KernelType::Iso | KernelType::GeneralizedIso | KernelType::PlateauIso
        )
    }
}

/// One blur stage's kernel distribution (`kernel_list`, `kernel_prob`, `sinc_prob`, `blur_sigma`,
/// `betag_range`, `betap_range`).
#[derive(Clone, Debug, PartialEq)]
pub struct KernelConfig {
    pub kernel_list: Vec<KernelType>,
    pub kernel_prob: Vec<f64>,
    pub sinc_prob: f64,
    pub blur_sigma: (f64, f64),
    pub betag_range: (f64, f64),
    pub betap_range: (f64, f64),
}

/// One degradation stage's resize / noise / JPEG distribution.
#[derive(Clone, Debug, PartialEq)]
pub struct StageConfig {
    /// `[up, down, keep]` probabilities.
    pub resize_prob: [f64; 3],
    pub resize_range: (f64, f64),
    pub gaussian_noise_prob: f64,
    /// Gaussian σ range in 8-bit levels.
    pub noise_range: (f64, f64),
    pub poisson_scale_range: (f64, f64),
    pub gray_noise_prob: f64,
    pub jpeg_range: (f64, f64),
}

/// The full Real-ESRGAN degradation configuration (HYPIR `sd2_train.yaml` defaults).
#[derive(Clone, Debug, PartialEq)]
pub struct DegradationConfig {
    pub kernel1: KernelConfig,
    pub kernel2: KernelConfig,
    pub final_sinc_prob: f64,
    pub use_hflip: bool,
    pub use_rot: bool,
    /// USM-sharpen the clean image (the GT) before degrading it.
    pub use_sharpener: bool,
    pub stage1: StageConfig,
    pub second_blur_prob: f64,
    /// The degradation's total downscale (the restorer's `4×`).
    pub stage2_scale: f64,
    pub stage2: StageConfig,
    /// Bicubic the degraded image back to the GT size (the restorer upsamples its input first).
    pub resize_back: bool,
}

impl Default for DegradationConfig {
    fn default() -> Self {
        let kernels = || KernelConfig {
            kernel_list: KernelType::ALL.to_vec(),
            kernel_prob: vec![0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
            sinc_prob: 0.1,
            blur_sigma: (0.2, 3.0),
            betag_range: (0.5, 4.0),
            betap_range: (1.0, 2.0),
        };
        let mut kernel2 = kernels();
        kernel2.blur_sigma = (0.2, 1.5);
        Self {
            kernel1: kernels(),
            kernel2,
            final_sinc_prob: 0.8,
            use_hflip: true,
            use_rot: false,
            use_sharpener: true,
            stage1: StageConfig {
                resize_prob: [0.2, 0.7, 0.1],
                resize_range: (0.15, 1.5),
                gaussian_noise_prob: 0.5,
                noise_range: (1.0, 30.0),
                poisson_scale_range: (0.05, 3.0),
                gray_noise_prob: 0.4,
                jpeg_range: (30.0, 95.0),
            },
            second_blur_prob: 0.8,
            stage2_scale: 4.0,
            stage2: StageConfig {
                resize_prob: [0.3, 0.4, 0.3],
                resize_range: (0.3, 1.2),
                gaussian_noise_prob: 0.5,
                noise_range: (1.0, 25.0),
                poisson_scale_range: (0.05, 2.5),
                gray_noise_prob: 0.4,
                jpeg_range: (30.0, 95.0),
            },
            resize_back: true,
        }
    }
}

fn check_prob(name: &str, p: f64) -> Result<()> {
    if (0.0..=1.0).contains(&p) {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "degradation: {name} = {p} is not a probability"
        )))
    }
}

fn check_range(name: &str, (lo, hi): (f64, f64), min: f64) -> Result<()> {
    if lo.is_finite() && hi.is_finite() && lo >= min && lo <= hi {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "degradation: {name} = [{lo}, {hi}] must be an ordered range ≥ {min}"
        )))
    }
}

impl KernelConfig {
    fn validate(&self, stage: &str) -> Result<()> {
        if self.kernel_list.is_empty() || self.kernel_list.len() != self.kernel_prob.len() {
            return Err(Error::Msg(format!(
                "degradation: {stage} kernel_list and kernel_prob must be non-empty and the same \
                 length"
            )));
        }
        if self
            .kernel_prob
            .iter()
            .any(|p| !(p.is_finite() && *p >= 0.0))
            || self.kernel_prob.iter().sum::<f64>() <= 0.0
        {
            return Err(Error::Msg(format!(
                "degradation: {stage} kernel_prob must be non-negative weights with a positive sum"
            )));
        }
        check_prob(&format!("{stage} sinc_prob"), self.sinc_prob)?;
        check_range(&format!("{stage} blur_sigma"), self.blur_sigma, 1e-6)?;
        // `random_bivariate_*` draws β from U(lo, 1) or U(1, hi): the range must straddle 1.
        for (n, r) in [
            ("betag_range", self.betag_range),
            ("betap_range", self.betap_range),
        ] {
            check_range(&format!("{stage} {n}"), r, 1e-6)?;
            if r.0 > 1.0 || r.1 < 1.0 {
                return Err(Error::Msg(format!(
                    "degradation: {stage} {n} = [{}, {}] must contain 1 (β ~ U(lo, 1) or U(1, hi))",
                    r.0, r.1
                )));
            }
        }
        Ok(())
    }
}

impl StageConfig {
    fn validate(&self, stage: &str) -> Result<()> {
        if self
            .resize_prob
            .iter()
            .any(|p| !(p.is_finite() && *p >= 0.0))
            || self.resize_prob.iter().sum::<f64>() <= 0.0
        {
            return Err(Error::Msg(format!(
                "degradation: {stage} resize_prob must be non-negative weights with a positive sum"
            )));
        }
        check_range(&format!("{stage} resize_range"), self.resize_range, 1e-6)?;
        if self.resize_range.0 > 1.0 || self.resize_range.1 < 1.0 {
            return Err(Error::Msg(format!(
                "degradation: {stage} resize_range must contain 1 (down draws U(lo, 1), up U(1, hi))"
            )));
        }
        check_prob(
            &format!("{stage} gaussian_noise_prob"),
            self.gaussian_noise_prob,
        )?;
        check_prob(&format!("{stage} gray_noise_prob"), self.gray_noise_prob)?;
        check_range(&format!("{stage} noise_range"), self.noise_range, 0.0)?;
        check_range(
            &format!("{stage} poisson_scale_range"),
            self.poisson_scale_range,
            0.0,
        )?;
        check_range(&format!("{stage} jpeg_range"), self.jpeg_range, 1.0)?;
        if self.jpeg_range.1 > 100.0 {
            return Err(Error::Msg(format!(
                "degradation: {stage} jpeg_range must lie in [1, 100]"
            )));
        }
        Ok(())
    }
}

impl DegradationConfig {
    /// Range / probability checks (every draw must be well defined).
    pub fn validate(&self) -> Result<()> {
        self.kernel1.validate("stage-1")?;
        self.kernel2.validate("stage-2")?;
        check_prob("final_sinc_prob", self.final_sinc_prob)?;
        check_prob("second_blur_prob", self.second_blur_prob)?;
        self.stage1.validate("stage-1")?;
        self.stage2.validate("stage-2")?;
        if !(self.stage2_scale.is_finite() && self.stage2_scale >= 1.0) {
            return Err(Error::Msg(format!(
                "degradation: stage2_scale = {} must be ≥ 1",
                self.stage2_scale
            )));
        }
        Ok(())
    }

    /// The smallest side every reflect-padded filter sees for an `side × side` crop, checked
    /// against the 21-tap kernels (torch's `reflect` padding needs `pad < side`): the stage-1 blur
    /// runs on the crop, the stage-2 blur on the stage-1 resize at its smallest factor, the final
    /// sinc on the stage-2 size. A crop too small for the configured ranges is refused (the
    /// reference would raise inside `F.pad`), never silently degraded differently.
    pub fn check_crop(&self, side: usize) -> Result<()> {
        let pad = KERNEL_SIDE / 2;
        let usm_pad = USM_RADIUS / 2;
        let s1_min = (side as f64 * self.stage1.resize_range.0).floor() as usize;
        let s2 = (side as f64 / self.stage2_scale) as usize;
        let s2_min = (s2 as f64 * self.stage2.resize_range.0) as usize;
        let mut need: Vec<(&str, usize, usize)> = vec![("the crop", side, pad)];
        if self.use_sharpener {
            need.push(("the crop (USM sharpening)", side, usm_pad));
        }
        if self.second_blur_prob > 0.0 {
            need.push(("the stage-1 output", s1_min, pad));
        }
        need.push(("the stage-2 size", s2, pad));
        if s2_min == 0 {
            need.push(("the stage-2 resize", s2_min, 0));
        }
        for (what, size, pad) in need {
            if size <= pad {
                return Err(Error::Msg(format!(
                    "degradation: a {side}px crop leaves {what} at {size}px, too small for the \
                     {}-tap reflect-padded filter — use a larger resolution or narrow the resize \
                     ranges",
                    2 * pad + 1
                )));
            }
        }
        Ok(())
    }
}

// =============================================================================================
// Blur kernels (`HYPIR/utils/degradation.py`, f64 like numpy, stored as f32 like `FloatTensor`)
// =============================================================================================

/// A square kernel, row-major, side [`KERNEL_SIDE`] (or [`USM_RADIUS`]).
pub type Kernel = Vec<f32>;

fn mesh(k: usize) -> Vec<f64> {
    // `np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)`
    let lo = -((k as i64 + 1) / 2) + 1;
    (0..k).map(|i| (lo + i as i64) as f64).collect()
}

/// `sigma_matrix2` (rotated) or the isotropic `diag(σx², σx²)`, inverted.
fn inverse_sigma(sig_x: f64, sig_y: f64, theta: f64, isotropic: bool) -> [[f64; 2]; 2] {
    let s = if isotropic {
        [[sig_x * sig_x, 0.0], [0.0, sig_x * sig_x]]
    } else {
        let (c, sn) = (theta.cos(), theta.sin());
        let u = [[c, -sn], [sn, c]];
        let d = [sig_x * sig_x, sig_y * sig_y];
        // U · D · Uᵀ
        let mut m = [[0.0; 2]; 2];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = u[i][0] * d[0] * u[j][0] + u[i][1] * d[1] * u[j][1];
            }
        }
        m
    };
    let det = s[0][0] * s[1][1] - s[0][1] * s[1][0];
    [
        [s[1][1] / det, -s[0][1] / det],
        [-s[1][0] / det, s[0][0] / det],
    ]
}

/// The quadratic form `gᵀ Σ⁻¹ g` over the mesh, `[k, k]` row-major (`grid[i][j] = (ax[j], ax[i])`).
fn quadratic(k: usize, inv: [[f64; 2]; 2]) -> Vec<f64> {
    let ax = mesh(k);
    let mut out = Vec::with_capacity(k * k);
    for &gy in &ax {
        for &gx in &ax {
            let a = gx * inv[0][0] + gy * inv[1][0];
            let b = gx * inv[0][1] + gy * inv[1][1];
            out.push(a * gx + b * gy);
        }
    }
    out
}

fn normalize(v: Vec<f64>) -> Vec<f64> {
    let s: f64 = v.iter().sum();
    v.into_iter().map(|x| x / s).collect()
}

/// The parameters of one mixed blur kernel (what `random_mixed_kernels` draws).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlurKernelParams {
    pub kind: KernelType,
    pub size: usize,
    pub sigma_x: f64,
    pub sigma_y: f64,
    pub theta: f64,
    /// Shape parameter (generalized / plateau; 1 for the Gaussians).
    pub beta: f64,
}

/// `bivariate_Gaussian` / `bivariate_generalized_Gaussian` / `bivariate_plateau` — normalized,
/// unpadded, f64.
pub fn blur_kernel_f64(p: &BlurKernelParams) -> Vec<f64> {
    let inv = inverse_sigma(p.sigma_x, p.sigma_y, p.theta, p.kind.isotropic());
    let q = quadratic(p.size, inv);
    let raw: Vec<f64> = match p.kind {
        KernelType::Iso | KernelType::Aniso => q.iter().map(|&v| (-0.5 * v).exp()).collect(),
        KernelType::GeneralizedIso | KernelType::GeneralizedAniso => {
            q.iter().map(|&v| (-0.5 * v.powf(p.beta)).exp()).collect()
        }
        KernelType::PlateauIso | KernelType::PlateauAniso => {
            q.iter().map(|&v| 1.0 / (v.powf(p.beta) + 1.0)).collect()
        }
    };
    normalize(raw)
}

/// Bessel `J₁(x)` by the trapezoid rule on the periodic integrand of
/// `J₁(x) = (1/2π) ∫₀^{2π} cos(τ − x sin τ) dτ` — spectrally convergent (machine precision for the
/// `|x| < 50` the sinc kernels need; scipy's cephes `j1` agrees to ~1e-15).
pub fn bessel_j1(x: f64) -> f64 {
    const N: usize = 512;
    let mut acc = 0.0;
    for i in 0..N {
        let t = std::f64::consts::TAU * i as f64 / N as f64;
        acc += (t - x * t.sin()).cos();
    }
    acc / N as f64
}

/// `circular_lowpass_kernel(cutoff, kernel_size, pad_to=0)` — the 2-D sinc, normalized, f64.
pub fn sinc_kernel_f64(cutoff: f64, size: usize) -> Vec<f64> {
    let c = (size as f64 - 1.0) / 2.0;
    let mut out = Vec::with_capacity(size * size);
    for x in 0..size {
        for y in 0..size {
            let r = ((x as f64 - c).powi(2) + (y as f64 - c).powi(2)).sqrt();
            out.push(if r == 0.0 {
                cutoff * cutoff / (4.0 * std::f64::consts::PI)
            } else {
                cutoff * bessel_j1(cutoff * r) / (2.0 * std::f64::consts::PI * r)
            });
        }
    }
    normalize(out)
}

/// Zero-pad an odd `k × k` kernel to `side × side` (centered) and round to f32.
pub fn pad_kernel(k: &[f64], size: usize, side: usize) -> Kernel {
    let pad = (side - size) / 2;
    let mut out = vec![0f32; side * side];
    for y in 0..size {
        for x in 0..size {
            out[(y + pad) * side + x + pad] = k[y * size + x] as f32;
        }
    }
    out
}

/// The identity kernel (`pulse_tensor`: a 1 at the center of a 21×21 zero kernel).
pub fn pulse_kernel() -> Kernel {
    let mut k = vec![0f32; KERNEL_SIDE * KERNEL_SIDE];
    k[(KERNEL_SIDE / 2) * KERNEL_SIDE + KERNEL_SIDE / 2] = 1.0;
    k
}

/// `cv2.getGaussianKernel(ksize, 0)` outer product (σ = `0.3·((ksize−1)·0.5 − 1) + 0.8`), f32.
pub fn usm_kernel() -> Kernel {
    let k = USM_RADIUS;
    let sigma = 0.3 * ((k as f64 - 1.0) * 0.5 - 1.0) + 0.8;
    let c = (k as f64 - 1.0) / 2.0;
    let g: Vec<f64> = (0..k)
        .map(|i| (-((i as f64 - c).powi(2)) / (2.0 * sigma * sigma)).exp())
        .collect();
    let s: f64 = g.iter().sum();
    let g: Vec<f64> = g.iter().map(|v| v / s).collect();
    let mut out = Vec::with_capacity(k * k);
    for a in &g {
        for b in &g {
            out.push((a * b) as f32);
        }
    }
    out
}

// =============================================================================================
// Planar image + torch ops
// =============================================================================================

/// A planar `[C, H, W]` f32 image (the restoration module's type).
pub use super::restoration::Planes;

/// `F.pad(mode="reflect")` index map (no edge repeat; `pad < n`).
fn reflect(i: i64, n: i64) -> usize {
    let mut i = i;
    if i < 0 {
        i = -i;
    }
    if i >= n {
        i = 2 * (n - 1) - i;
    }
    i as usize
}

/// `filter2D(img, kernel)`: reflect-pad by `k // 2`, then a valid cross-correlation of every plane
/// with the (odd, square) kernel. f32 accumulation, kernel rows outer.
pub fn filter2d(img: &Planes, kernel: &[f32]) -> Result<Planes> {
    let k = (kernel.len() as f64).sqrt() as usize;
    if k * k != kernel.len() || k.is_multiple_of(2) {
        return Err(Error::Msg("filter2D: wrong kernel size".into()));
    }
    let (h, w) = (img.height, img.width);
    let r = k / 2;
    if r >= h || r >= w {
        return Err(Error::Msg(format!(
            "filter2D: a {k}×{k} reflect-padded kernel needs a side above {r}, got {w}×{h}"
        )));
    }
    let mut out = Planes::zeros(img.channels, h, w);
    let rows: Vec<Vec<usize>> = (0..h)
        .map(|y| {
            (0..k)
                .map(|ky| reflect(y as i64 + ky as i64 - r as i64, h as i64))
                .collect()
        })
        .collect();
    let cols: Vec<Vec<usize>> = (0..w)
        .map(|x| {
            (0..k)
                .map(|kx| reflect(x as i64 + kx as i64 - r as i64, w as i64))
                .collect()
        })
        .collect();
    for c in 0..img.channels {
        let src = &img.data[c * h * w..(c + 1) * h * w];
        let dst = &mut out.data[c * h * w..(c + 1) * h * w];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0f32;
                for (ky, &sy) in rows[y].iter().enumerate() {
                    let row = &src[sy * w..];
                    let kr = &kernel[ky * k..(ky + 1) * k];
                    for (kx, &sx) in cols[x].iter().enumerate() {
                        acc += row[sx] * kr[kx];
                    }
                }
                dst[y * w + x] = acc;
            }
        }
    }
    Ok(out)
}

/// `USMSharp()(img)` (weight 0.5, threshold 10): `residual = img − blur(img)`, a soft mask of the
/// pixels whose `|residual|·255 > 10` (blurred by the same 51-tap Gaussian), and
/// `mask·clip(img + 0.5·residual, 0, 1) + (1 − mask)·img`.
pub fn usm_sharpen(img: &Planes) -> Result<Planes> {
    let kernel = usm_kernel();
    let blur = filter2d(img, &kernel)?;
    let n = img.data.len();
    let mut residual = vec![0f32; n];
    let mut mask = Planes::zeros(img.channels, img.height, img.width);
    for (i, r) in residual.iter_mut().enumerate() {
        *r = img.data[i] - blur.data[i];
        mask.data[i] = if r.abs() * 255.0 > USM_THRESHOLD {
            1.0
        } else {
            0.0
        };
    }
    let soft = filter2d(&mask, &kernel)?;
    let mut out = img.clone();
    for (i, r) in residual.iter().enumerate() {
        let sharp = (img.data[i] + USM_WEIGHT * r).clamp(0.0, 1.0);
        out.data[i] = soft.data[i] * sharp + (1.0 - soft.data[i]) * img.data[i];
    }
    Ok(out)
}

/// The `F.interpolate` modes the pipeline draws from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interp {
    /// `area` = `adaptive_avg_pool2d` to the output size.
    Area,
    /// `bilinear`, `align_corners=False`.
    Bilinear,
    /// `bicubic`, `align_corners=False`, Keys `a = −0.75`, no antialias.
    Bicubic,
}

impl Interp {
    pub const ALL: [Interp; 3] = [Interp::Area, Interp::Bilinear, Interp::Bicubic];
    pub fn name(self) -> &'static str {
        match self {
            Interp::Area => "area",
            Interp::Bilinear => "bilinear",
            Interp::Bicubic => "bicubic",
        }
    }
}

/// `F.interpolate(x, scale_factor=scale, mode)`: output `floor(side · scale)`, coordinate scale
/// `1 / scale`.
pub fn interpolate_scale(img: &Planes, scale: f64, mode: Interp) -> Result<Planes> {
    let oh = (img.height as f64 * scale).floor() as usize;
    let ow = (img.width as f64 * scale).floor() as usize;
    let coord = (1.0 / scale) as f32;
    resize_with(img, oh, ow, coord, coord, mode)
}

/// `F.interpolate(x, size=(oh, ow), mode)`: coordinate scale `in / out` (f32).
pub fn interpolate_size(img: &Planes, oh: usize, ow: usize, mode: Interp) -> Result<Planes> {
    if oh == 0 || ow == 0 {
        return Err(Error::Msg(format!(
            "interpolate: output size {ow}×{oh} is empty"
        )));
    }
    let ch = img.height as f32 / oh as f32;
    let cw = img.width as f32 / ow as f32;
    resize_with(img, oh, ow, ch, cw, mode)
}

fn resize_with(
    img: &Planes,
    oh: usize,
    ow: usize,
    coord_h: f32,
    coord_w: f32,
    mode: Interp,
) -> Result<Planes> {
    if oh == 0 || ow == 0 {
        return Err(Error::Msg(format!(
            "interpolate: a {}×{} image resizes to an empty {ow}×{oh}",
            img.width, img.height
        )));
    }
    let (h, w) = (img.height, img.width);
    let mut out = Planes::zeros(img.channels, oh, ow);
    match mode {
        Interp::Area => {
            // adaptive_avg_pool2d: start = floor(i·in/out), end = ceil((i+1)·in/out).
            let span = |i: usize, inn: usize, outn: usize| {
                let s = (i * inn) / outn;
                let e = ((i + 1) * inn).div_ceil(outn);
                (s, e)
            };
            for c in 0..img.channels {
                let src = &img.data[c * h * w..(c + 1) * h * w];
                let dst = &mut out.data[c * oh * ow..(c + 1) * oh * ow];
                for oy in 0..oh {
                    let (ys, ye) = span(oy, h, oh);
                    for ox in 0..ow {
                        let (xs, xe) = span(ox, w, ow);
                        let mut sum = 0f32;
                        for y in ys..ye {
                            for x in xs..xe {
                                sum += src[y * w + x];
                            }
                        }
                        dst[oy * ow + ox] = sum / (ye - ys) as f32 / (xe - xs) as f32;
                    }
                }
            }
        }
        Interp::Bilinear => {
            let taps = |inn: usize, outn: usize, coord: f32| -> Vec<(usize, usize, f32, f32)> {
                (0..outn)
                    .map(|i| {
                        let real = (coord * (i as f32 + 0.5) - 0.5).max(0.0);
                        let i0 = real as usize;
                        let off = usize::from(i0 + 1 < inn);
                        let l1 = (real - i0 as f32).clamp(0.0, 1.0);
                        (i0.min(inn - 1), (i0 + off).min(inn - 1), 1.0 - l1, l1)
                    })
                    .collect()
            };
            let rows = taps(h, oh, coord_h);
            let cols = taps(w, ow, coord_w);
            for c in 0..img.channels {
                let src = &img.data[c * h * w..(c + 1) * h * w];
                let dst = &mut out.data[c * oh * ow..(c + 1) * oh * ow];
                for (oy, &(y0, y1, h0, h1)) in rows.iter().enumerate() {
                    for (ox, &(x0, x1, w0, w1)) in cols.iter().enumerate() {
                        let top = w0 * src[y0 * w + x0] + w1 * src[y0 * w + x1];
                        let bot = w0 * src[y1 * w + x0] + w1 * src[y1 * w + x1];
                        dst[oy * ow + ox] = h0 * top + h1 * bot;
                    }
                }
            }
        }
        Interp::Bicubic => {
            let rows = super::restoration::bicubic_taps(h, oh, coord_h);
            let cols = super::restoration::bicubic_taps(w, ow, coord_w);
            for c in 0..img.channels {
                let src = &img.data[c * h * w..(c + 1) * h * w];
                let dst = &mut out.data[c * oh * ow..(c + 1) * oh * ow];
                for (oy, (ri, rw)) in rows.iter().enumerate() {
                    for (ox, (ci, cw)) in cols.iter().enumerate() {
                        let row = |r: usize| {
                            let base = &src[r * w..];
                            let mut acc = base[ci[0]] * cw[0];
                            for (&i, &wt) in ci.iter().zip(cw).skip(1) {
                                acc += base[i] * wt;
                            }
                            acc
                        };
                        let mut acc = row(ri[0]) * rw[0];
                        for (&r, &wt) in ri.iter().zip(rw).skip(1) {
                            acc += row(r) * wt;
                        }
                        dst[oy * ow + ox] = acc;
                    }
                }
            }
        }
    }
    Ok(out)
}

fn clamp01(img: &mut Planes) {
    for v in &mut img.data {
        *v = v.clamp(0.0, 1.0);
    }
}

/// `torch.clamp((x * 255.0).round(), 0, 255) / 255.0` (round half to even).
pub fn quantize8(img: &mut Planes) {
    for v in &mut img.data {
        *v = (*v * 255.0).round_ties_even().clamp(0.0, 255.0) / 255.0;
    }
}

/// `torchvision rgb_to_grayscale`: `0.2989·r + 0.587·g + 0.114·b` (f32), one plane.
fn grayscale(img: &Planes) -> Vec<f32> {
    let n = img.height * img.width;
    (0..n)
        .map(|i| {
            0.2989f32 * img.data[i] + 0.587f32 * img.data[n + i] + 0.114f32 * img.data[2 * n + i]
        })
        .collect()
}

/// `2 ** ceil(log2(len(unique(x))))` of an 8-bit-rounded image.
fn poisson_vals(rounded: &[f32]) -> f32 {
    let mut seen = [false; 256];
    for &v in rounded {
        seen[((v * 255.0).round() as i64).clamp(0, 255) as usize] = true;
    }
    let count = seen.iter().filter(|&&b| b).count().max(1);
    2f64.powf((count as f64).log2().ceil()) as f32
}

// =============================================================================================
// JPEG (`DiffJPEG(differentiable=False)`)
// =============================================================================================

const Y_TABLE: [[f32; 8]; 8] = [
    [16., 11., 10., 16., 24., 40., 51., 61.],
    [12., 12., 14., 19., 26., 58., 60., 55.],
    [14., 13., 16., 24., 40., 57., 69., 56.],
    [14., 17., 22., 29., 51., 87., 80., 62.],
    [18., 22., 37., 56., 68., 109., 103., 77.],
    [24., 35., 55., 64., 81., 104., 113., 92.],
    [49., 64., 78., 87., 103., 121., 120., 101.],
    [72., 92., 95., 98., 112., 100., 103., 99.],
];

/// The reference's tables are the transposes of the standard ones (`np.array(...).T`).
fn quant_tables() -> ([f32; 64], [f32; 64]) {
    let mut y = [0f32; 64];
    let mut c = [99f32; 64];
    let c4: [[f32; 4]; 4] = [
        [17., 18., 24., 47.],
        [18., 21., 26., 66.],
        [24., 26., 56., 99.],
        [47., 66., 99., 99.],
    ];
    for u in 0..8 {
        for v in 0..8 {
            y[u * 8 + v] = Y_TABLE[v][u];
            if u < 4 && v < 4 {
                c[u * 8 + v] = c4[v][u];
            }
        }
    }
    (y, c)
}

/// `quality_to_factor`.
pub fn jpeg_quality_factor(quality: f32) -> f32 {
    let q = if quality < 50.0 {
        5000.0 / quality
    } else {
        200.0 - quality * 2.0
    };
    q / 100.0
}

/// The `[8,8,8,8]` DCT basis `cos((2x+1)uπ/16)·cos((2y+1)vπ/16)` (f64 → f32 like the reference).
fn dct_basis() -> Vec<f32> {
    let mut t = vec![0f32; 4096];
    for x in 0..8 {
        for y in 0..8 {
            for u in 0..8 {
                for v in 0..8 {
                    let a = ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0).cos();
                    let b = ((2 * y + 1) as f64 * v as f64 * std::f64::consts::PI / 16.0).cos();
                    t[((x * 8 + y) * 8 + u) * 8 + v] = (a * b) as f32;
                }
            }
        }
    }
    t
}

fn alpha_outer() -> [f32; 64] {
    let a = |i: usize| if i == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
    let mut o = [0f32; 64];
    for u in 0..8 {
        for v in 0..8 {
            o[u * 8 + v] = (a(u) * a(v)) as f32;
        }
    }
    o
}

/// One channel plane through block DCT → quantize (round half even) → dequantize → iDCT.
fn jpeg_plane(plane: &mut [f32], h: usize, w: usize, table: &[f32; 64], factor: f32) {
    let basis = dct_basis();
    let alpha = alpha_outer();
    let scale: Vec<f32> = alpha.iter().map(|a| ((*a as f64) * 0.25) as f32).collect();
    let mut block = [0f32; 64];
    let mut coef = [0f32; 64];
    for by in (0..h).step_by(8) {
        for bx in (0..w).step_by(8) {
            for i in 0..8 {
                for j in 0..8 {
                    block[i * 8 + j] = plane[(by + i) * w + bx + j] - 128.0;
                }
            }
            // DCT8x8: scale ⊙ tensordot(image, tensor, dims=2) over (x, y).
            for uv in 0..64 {
                let mut acc = 0f32;
                for xy in 0..64 {
                    acc += block[xy] * basis[xy * 64 + uv];
                }
                coef[uv] = scale[uv] * acc;
            }
            for uv in 0..64 {
                let q = table[uv] * factor;
                coef[uv] = (coef[uv] / q).round_ties_even() * q;
            }
            // iDCT8x8: 0.25 · tensordot(image ⊙ alpha, tensor, dims=2) + 128.
            for uv in 0..64 {
                coef[uv] *= alpha[uv];
            }
            // `tensor[x, y, u, v] = cos((2u+1)xπ/16)·cos((2v+1)yπ/16)` contracted over the
            // frequency pair (x, y) = the DCT basis read with its two index pairs swapped.
            for ab in 0..64 {
                let mut acc = 0f32;
                for ij in 0..64 {
                    acc += coef[ij] * basis[ab * 64 + ij];
                }
                block[ab] = 0.25 * acc + 128.0;
            }
            for i in 0..8 {
                for j in 0..8 {
                    plane[(by + i) * w + bx + j] = block[i * 8 + j];
                }
            }
        }
    }
}

/// `DiffJPEG(differentiable=False)(x, quality)` on one `[3, H, W]` image in `[0, 1]`: zero pad to a
/// multiple of 16, RGB → YCbCr (JPEG matrix, ×255), 2×2 average chroma subsampling, 8×8 DCT,
/// quantization by the (transposed) standard tables × `quality_to_factor`, the inverse path with
/// nearest chroma upsampling, clamp to `[0, 255]`, `/255`, crop.
pub fn jpeg(img: &Planes, quality: f32) -> Planes {
    let factor = jpeg_quality_factor(quality);
    let (h, w) = (img.height, img.width);
    let ph = h.div_ceil(16) * 16;
    let pw = w.div_ceil(16) * 16;
    let n = ph * pw;
    let at = |c: usize, y: usize, x: usize| -> f32 {
        if y < h && x < w {
            img.data[c * h * w + y * w + x] * 255.0
        } else {
            0.0
        }
    };
    let m = [
        [0.299f32, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ];
    let shift = [0f32, 128.0, 128.0];
    let mut ycc = vec![vec![0f32; n]; 3];
    for y in 0..ph {
        for x in 0..pw {
            let rgb = [at(0, y, x), at(1, y, x), at(2, y, x)];
            for k in 0..3 {
                ycc[k][y * pw + x] =
                    rgb[0] * m[k][0] + rgb[1] * m[k][1] + rgb[2] * m[k][2] + shift[k];
            }
        }
    }
    let (ch, cw) = (ph / 2, pw / 2);
    let sub = |plane: &[f32]| -> Vec<f32> {
        let mut o = vec![0f32; ch * cw];
        for y in 0..ch {
            for x in 0..cw {
                let s = plane[2 * y * pw + 2 * x]
                    + plane[2 * y * pw + 2 * x + 1]
                    + plane[(2 * y + 1) * pw + 2 * x]
                    + plane[(2 * y + 1) * pw + 2 * x + 1];
                o[y * cw + x] = s / 4.0;
            }
        }
        o
    };
    let (yt, ct) = quant_tables();
    let mut yp = std::mem::take(&mut ycc[0]);
    let mut cb = sub(&ycc[1]);
    let mut cr = sub(&ycc[2]);
    jpeg_plane(&mut yp, ph, pw, &yt, factor);
    jpeg_plane(&mut cb, ch, cw, &ct, factor);
    jpeg_plane(&mut cr, ch, cw, &ct, factor);
    let m2 = [
        [1.0f32, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0],
    ];
    let mut out = Planes::zeros(3, h, w);
    for y in 0..h {
        for x in 0..w {
            let c = (y / 2) * cw + x / 2;
            let v = [yp[y * pw + x], cb[c] - 128.0, cr[c] - 128.0];
            for (k, row) in m2.iter().enumerate() {
                let r = v[0] * row[0] + v[1] * row[1] + v[2] * row[2];
                out.data[k * h * w + y * w + x] = r.clamp(0.0, 255.0) / 255.0;
            }
        }
    }
    out
}

// =============================================================================================
// Noise
// =============================================================================================

/// Where the per-pixel noise of a degradation comes from: the item's positional stream in
/// training ([`PositionalNoise`]), or recorded draws in a parity test.
pub trait NoiseSource {
    /// `n` standard normals (`torch.randn`).
    fn normals(&mut self, n: usize) -> Vec<f32>;
    /// One Poisson draw per rate (`torch.poisson(rates)`).
    fn poisson(&mut self, rates: &[f32]) -> Vec<f32>;
}

/// The training noise stream: a [`HostRng`] (Box–Muller normals, PTRS / Knuth Poisson).
pub struct PositionalNoise(pub HostRng);

/// numpy's `random_loggam` (log Γ via the Stirling series with an upward shift below 7); the
/// coefficients are numpy's, verbatim.
#[allow(clippy::excessive_precision)]
fn loggam(x: f64) -> f64 {
    const A: [f64; 10] = [
        8.333333333333333e-02,
        -2.777777777777778e-03,
        7.936507936507937e-04,
        -5.952380952380952e-04,
        8.417508417508418e-04,
        -1.917526917526918e-03,
        6.410256410256410e-03,
        -2.955065359477124e-02,
        1.796443723688307e-01,
        -1.39243221690590e+00,
    ];
    if x == 1.0 || x == 2.0 {
        return 0.0;
    }
    let n = if x < 7.0 { (7.0 - x) as i64 } else { 0 };
    let mut x0 = x + n as f64;
    let x2 = (1.0 / x0) * (1.0 / x0);
    let mut gl0 = A[9];
    for k in (0..9).rev() {
        gl0 = gl0 * x2 + A[k];
    }
    let mut gl = gl0 / x0 + 0.5 * 1.8378770664093453 + (x0 - 0.5) * x0.ln() - x0;
    if x < 7.0 {
        for _ in 0..n {
            gl -= (x0 - 1.0).ln();
            x0 -= 1.0;
        }
    }
    gl
}

/// One Poisson(λ) draw: Knuth's product method below λ = 10, Hörmann's PTRS above (numpy's
/// `random_poisson`).
pub fn poisson_draw(rng: &mut HostRng, lam: f64) -> f64 {
    if lam <= 0.0 {
        return 0.0;
    }
    if lam < 10.0 {
        let enlam = (-lam).exp();
        let mut x = 0.0;
        let mut prod = 1.0;
        loop {
            prod *= rng.uniform();
            if prod > enlam {
                x += 1.0;
            } else {
                return x;
            }
        }
    }
    let slam = lam.sqrt();
    let loglam = lam.ln();
    let b = 0.931 + 2.53 * slam;
    let a = -0.059 + 0.02483 * b;
    let invalpha = 1.1239 + 1.1328 / (b - 3.4);
    let vr = 0.9277 - 3.6224 / (b - 2.0);
    loop {
        let u = rng.uniform() - 0.5;
        let v = rng.uniform();
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + lam + 0.43).floor();
        if us >= 0.07 && v <= vr {
            return k;
        }
        if k < 0.0 || (us < 0.013 && v > us) {
            continue;
        }
        if v.ln() + invalpha.ln() - (a / (us * us) + b).ln() <= -lam + k * loglam - loggam(k + 1.0)
        {
            return k;
        }
    }
}

impl NoiseSource for PositionalNoise {
    fn normals(&mut self, n: usize) -> Vec<f32> {
        self.0.normals_f32(n)
    }
    fn poisson(&mut self, rates: &[f32]) -> Vec<f32> {
        rates
            .iter()
            .map(|&r| poisson_draw(&mut self.0, r as f64) as f32)
            .collect()
    }
}

/// One stage's drawn noise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoiseDraw {
    /// σ in 8-bit levels; `gray` = one luminance field for all channels.
    Gaussian { sigma: f32, gray: bool },
    /// Shot-noise strength; `gray` as above.
    Poisson { scale: f32, gray: bool },
}

/// `random_add_gaussian_noise_pt` / `random_add_poisson_noise_pt` with the drawn parameters
/// (`clip=True, rounds=False`). Draw order from `src`: the gray field (when gray), then the color
/// field (always, as the reference computes it unconditionally).
pub fn add_noise(img: &Planes, draw: NoiseDraw, src: &mut dyn NoiseSource) -> Planes {
    let (h, w) = (img.height, img.width);
    let hw = h * w;
    let mut out = img.clone();
    match draw {
        NoiseDraw::Gaussian { sigma, gray } => {
            let gray_field = gray.then(|| src.normals(hw));
            let color = src.normals(3 * hw);
            for c in 0..3 {
                for i in 0..hw {
                    let n = match &gray_field {
                        Some(g) => g[i] * sigma / 255.0,
                        None => color[c * hw + i] * sigma / 255.0,
                    };
                    out.data[c * hw + i] = img.data[c * hw + i] + n;
                }
            }
        }
        NoiseDraw::Poisson { scale, gray } => {
            let gray_noise = gray.then(|| {
                let mut g = grayscale(img);
                for v in &mut g {
                    *v = (*v * 255.0).round_ties_even().clamp(0.0, 255.0) / 255.0;
                }
                let vals = poisson_vals(&g);
                let rates: Vec<f32> = g.iter().map(|v| v * vals).collect();
                let counts = src.poisson(&rates);
                g.iter()
                    .zip(counts)
                    .map(|(v, k)| k / vals - v)
                    .collect::<Vec<f32>>()
            });
            let rounded: Vec<f32> = img
                .data
                .iter()
                .map(|v| (v * 255.0).round_ties_even().clamp(0.0, 255.0) / 255.0)
                .collect();
            let vals = poisson_vals(&rounded);
            let rates: Vec<f32> = rounded.iter().map(|v| v * vals).collect();
            let counts = src.poisson(&rates);
            for c in 0..3 {
                for i in 0..hw {
                    let j = c * hw + i;
                    let n = match &gray_noise {
                        Some(g) => g[i],
                        None => counts[j] / vals - rounded[j],
                    };
                    out.data[j] = img.data[j] + n * scale;
                }
            }
        }
    }
    clamp01(&mut out);
    out
}

// =============================================================================================
// The draw
// =============================================================================================

/// One stage's drawn resize.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResizeDraw {
    /// The factor (1 for `keep`).
    pub scale: f64,
    pub mode: Interp,
}

/// One stage's draws.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StageDraw {
    pub resize: ResizeDraw,
    pub noise: NoiseDraw,
    pub jpeg_quality: f32,
}

/// Every random choice of one item's degradation.
#[derive(Clone, Debug, PartialEq)]
pub struct DegradationDraw {
    pub hflip: bool,
    pub vflip: bool,
    pub rot90: bool,
    /// `[21, 21]` padded kernels (f32).
    pub kernel1: Kernel,
    pub kernel2: Kernel,
    /// The final sinc filter, or the identity pulse.
    pub sinc: Kernel,
    pub stage1: StageDraw,
    pub second_blur: bool,
    pub stage2: StageDraw,
    /// `true`: [resize back + sinc] then JPEG; `false`: JPEG then [resize back + sinc].
    pub sinc_before_jpeg: bool,
    /// Interpolation of the stage-2 resize back.
    pub back_mode: Interp,
    /// Key of the item's per-pixel noise stream ([`PositionalNoise`]).
    pub noise_key: u64,
}

/// Stream tag of the degradation draws (`mix_seed(seed, TAG, epoch, item)`), distinct from every
/// other positional stream of a run.
pub const DEGRADATION_STREAM: u64 = 0x5245_5354_4f52_4531; // "RESTORE1"
const NOISE_STREAM: u64 = 0x4e4f_4953_4531; // "NOISE1"

/// The positional key of item `index` in `epoch`.
pub fn degradation_key(seed: u64, epoch: u32, index: usize) -> u64 {
    mix_seed(&[seed, DEGRADATION_STREAM, epoch as u64, index as u64])
}

fn weighted_choice(rng: &mut HostRng, weights: &[f64]) -> usize {
    let total: f64 = weights.iter().sum();
    let mut u = rng.uniform() * total;
    for (i, w) in weights.iter().enumerate() {
        if u < *w {
            return i;
        }
        u -= w;
    }
    weights.len() - 1
}

fn uniform(rng: &mut HostRng, (lo, hi): (f64, f64)) -> f64 {
    lo + (hi - lo) * rng.uniform()
}

/// `realesrgan.py`'s kernel draw for one stage (sinc with `sinc_prob`, else a mixed kernel),
/// padded to 21.
fn draw_kernel(rng: &mut HostRng, cfg: &KernelConfig) -> (Kernel, Option<BlurKernelParams>) {
    let size = KERNEL_SIZES[rng.below(KERNEL_SIZES.len())];
    if rng.uniform() < cfg.sinc_prob {
        let omega = if size < 13 {
            uniform(rng, (std::f64::consts::PI / 3.0, std::f64::consts::PI))
        } else {
            uniform(rng, (std::f64::consts::PI / 5.0, std::f64::consts::PI))
        };
        return (
            pad_kernel(&sinc_kernel_f64(omega, size), size, KERNEL_SIDE),
            None,
        );
    }
    let kind = cfg.kernel_list[weighted_choice(rng, &cfg.kernel_prob)];
    let sigma_x = uniform(rng, cfg.blur_sigma);
    let (sigma_y, theta) = if kind.isotropic() {
        (sigma_x, 0.0)
    } else {
        (
            uniform(rng, cfg.blur_sigma),
            uniform(rng, (-std::f64::consts::PI, std::f64::consts::PI)),
        )
    };
    let beta = match kind {
        KernelType::Iso | KernelType::Aniso => 1.0,
        KernelType::GeneralizedIso | KernelType::GeneralizedAniso => {
            beta_draw(rng, cfg.betag_range)
        }
        KernelType::PlateauIso | KernelType::PlateauAniso => beta_draw(rng, cfg.betap_range),
    };
    let p = BlurKernelParams {
        kind,
        size,
        sigma_x,
        sigma_y,
        theta,
        beta,
    };
    (pad_kernel(&blur_kernel_f64(&p), size, KERNEL_SIDE), Some(p))
}

fn beta_draw(rng: &mut HostRng, (lo, hi): (f64, f64)) -> f64 {
    if rng.uniform() < 0.5 {
        uniform(rng, (lo, 1.0))
    } else {
        uniform(rng, (1.0, hi))
    }
}

fn draw_stage(rng: &mut HostRng, cfg: &StageConfig) -> StageDraw {
    let scale = match weighted_choice(rng, &cfg.resize_prob) {
        0 => uniform(rng, (1.0, cfg.resize_range.1)),
        1 => uniform(rng, (cfg.resize_range.0, 1.0)),
        _ => 1.0,
    };
    let mode = Interp::ALL[rng.below(3)];
    let noise = if rng.uniform() < cfg.gaussian_noise_prob {
        let sigma = uniform(rng, cfg.noise_range) as f32;
        NoiseDraw::Gaussian {
            sigma,
            gray: rng.uniform() < cfg.gray_noise_prob,
        }
    } else {
        let scale = uniform(rng, cfg.poisson_scale_range) as f32;
        NoiseDraw::Poisson {
            scale,
            gray: rng.uniform() < cfg.gray_noise_prob,
        }
    };
    StageDraw {
        resize: ResizeDraw { scale, mode },
        noise,
        jpeg_quality: uniform(rng, cfg.jpeg_range) as f32,
    }
}

impl DegradationDraw {
    /// Draw item `key`'s degradation (order: flips, kernel 1, kernel 2, final sinc, stage 1,
    /// stage-2 blur, stage 2, stage-2 order, resize-back mode).
    pub fn draw(cfg: &DegradationConfig, key: u64) -> Self {
        let mut rng = HostRng::new(key);
        let hflip = cfg.use_hflip && rng.uniform() < 0.5;
        let vflip = cfg.use_rot && rng.uniform() < 0.5;
        let rot90 = cfg.use_rot && rng.uniform() < 0.5;
        let (kernel1, _) = draw_kernel(&mut rng, &cfg.kernel1);
        let (kernel2, _) = draw_kernel(&mut rng, &cfg.kernel2);
        let sinc = if rng.uniform() < cfg.final_sinc_prob {
            let size = KERNEL_SIZES[rng.below(KERNEL_SIZES.len())];
            let omega = uniform(&mut rng, (std::f64::consts::PI / 3.0, std::f64::consts::PI));
            pad_kernel(&sinc_kernel_f64(omega, size), size, KERNEL_SIDE)
        } else {
            pulse_kernel()
        };
        let stage1 = draw_stage(&mut rng, &cfg.stage1);
        let second_blur = rng.uniform() < cfg.second_blur_prob;
        let stage2 = draw_stage(&mut rng, &cfg.stage2);
        let sinc_before_jpeg = rng.uniform() < 0.5;
        let back_mode = Interp::ALL[rng.below(3)];
        Self {
            hflip,
            vflip,
            rot90,
            kernel1,
            kernel2,
            sinc,
            stage1,
            second_blur,
            stage2,
            sinc_before_jpeg,
            back_mode,
            noise_key: mix_seed(&[key, NOISE_STREAM]),
        }
    }

    /// The item's training noise stream.
    pub fn noise(&self) -> PositionalNoise {
        PositionalNoise(HostRng::new(self.noise_key))
    }
}

// =============================================================================================
// The pipeline
// =============================================================================================

/// `augment(img, hflip, rotation)`: horizontal flip, vertical flip, transpose — in that order.
pub fn augment(img: &Planes, hflip: bool, vflip: bool, rot90: bool) -> Planes {
    let (c, h, w) = (img.channels, img.height, img.width);
    let mut out = img.clone();
    if hflip || vflip {
        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let sy = if vflip { h - 1 - y } else { y };
                    let sx = if hflip { w - 1 - x } else { x };
                    out.data[ch * h * w + y * w + x] = img.data[ch * h * w + sy * w + sx];
                }
            }
        }
    }
    if rot90 {
        let src = out.clone();
        out = Planes::zeros(c, w, h);
        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    out.data[ch * h * w + x * h + y] = src.data[ch * h * w + y * w + x];
                }
            }
        }
    }
    out
}

/// One training pair: the GT (sharpened clean image) and the degraded input at the GT size, both
/// `[3, H, W]` in `[0, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct DegradedPair {
    pub gt: Planes,
    pub lq: Planes,
}

/// Apply `draw` to the clean (already flipped) `hq`, taking the per-pixel noise from `noise`
/// (the batch transform of `batch_transform.py` for one item).
pub fn degrade_with(
    cfg: &DegradationConfig,
    hq: &Planes,
    draw: &DegradationDraw,
    noise: &mut dyn NoiseSource,
) -> Result<DegradedPair> {
    if hq.channels != 3 {
        return Err(Error::Msg("degradation: expected an RGB image".into()));
    }
    let (ori_h, ori_w) = (hq.height, hq.width);
    let gt = if cfg.use_sharpener {
        usm_sharpen(hq)?
    } else {
        hq.clone()
    };
    // ---- the first degradation ----------------------------------------------------------------
    let mut out = filter2d(&gt, &draw.kernel1)?;
    out = interpolate_scale(&out, draw.stage1.resize.scale, draw.stage1.resize.mode)?;
    out = add_noise(&out, draw.stage1.noise, noise);
    clamp01(&mut out);
    out = jpeg(&out, draw.stage1.jpeg_quality);
    // ---- the second degradation ---------------------------------------------------------------
    if draw.second_blur {
        out = filter2d(&out, &draw.kernel2)?;
    }
    let s2h = (ori_h as f64 / cfg.stage2_scale) as usize;
    let s2w = (ori_w as f64 / cfg.stage2_scale) as usize;
    let sc = draw.stage2.resize.scale;
    out = interpolate_size(
        &out,
        (s2h as f64 * sc) as usize,
        (s2w as f64 * sc) as usize,
        draw.stage2.resize.mode,
    )?;
    out = add_noise(&out, draw.stage2.noise, noise);
    if draw.sinc_before_jpeg {
        out = interpolate_size(&out, s2h, s2w, draw.back_mode)?;
        out = filter2d(&out, &draw.sinc)?;
        clamp01(&mut out);
        out = jpeg(&out, draw.stage2.jpeg_quality);
    } else {
        clamp01(&mut out);
        out = jpeg(&out, draw.stage2.jpeg_quality);
        out = interpolate_size(&out, s2h, s2w, draw.back_mode)?;
        out = filter2d(&out, &draw.sinc)?;
    }
    if cfg.stage2_scale != 1.0 && cfg.resize_back {
        out = interpolate_size(&out, ori_h, ori_w, Interp::Bicubic)?;
    }
    quantize8(&mut out);
    Ok(DegradedPair { gt, lq: out })
}

/// The full per-item degradation: draw from `key`, flip, degrade with the item's positional noise.
pub fn degrade(cfg: &DegradationConfig, clean: &Planes, key: u64) -> Result<DegradedPair> {
    let draw = DegradationDraw::draw(cfg, key);
    let hq = augment(clean, draw.hflip, draw.vflip, draw.rot90);
    degrade_with(cfg, &hq, &draw, &mut draw.noise())
}

#[cfg(test)]
#[path = "degradation_parity.rs"]
mod parity;

#[cfg(test)]
mod tests {
    use super::*;

    fn wave(c: usize, h: usize, w: usize) -> Planes {
        let mut p = Planes::zeros(c, h, w);
        for (i, v) in p.data.iter_mut().enumerate() {
            *v = 0.5 + 0.45 * ((i as f32) * 0.37).sin() * ((i as f32) * 0.011).cos();
        }
        p
    }

    #[test]
    fn bessel_j1_matches_known_values() {
        // scipy.special.j1 reference values.
        for (x, want) in [
            (0.0, 0.0),
            (1.0, 0.440_050_585_744_933_5),
            (2.5, 0.497_094_102_464_274_4),
            (10.0, 0.043_472_746_168_861_44),
            (30.0, -0.118_751_062_616_622_98),
            (44.0, -0.082_803_359_376_029_19),
        ] {
            assert!(
                (bessel_j1(x) - want).abs() < 1e-13,
                "J1({x}) = {}",
                bessel_j1(x)
            );
        }
    }

    #[test]
    fn loggam_matches_ln_factorial() {
        let mut f = 0f64;
        for k in 1..40u32 {
            f += (k as f64).ln();
            assert!((loggam(k as f64 + 1.0) - f).abs() < 1e-10, "k={k}");
        }
    }

    #[test]
    fn poisson_draws_have_the_right_moments() {
        for lam in [0.5f64, 4.0, 37.0, 200.0] {
            let mut rng = HostRng::new(7);
            let n = 40_000;
            let xs: Vec<f64> = (0..n).map(|_| poisson_draw(&mut rng, lam)).collect();
            let mean = xs.iter().sum::<f64>() / n as f64;
            let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            assert!(
                (mean - lam).abs() < 0.05 * lam.max(1.0),
                "λ={lam} mean {mean}"
            );
            assert!((var - lam).abs() < 0.08 * lam.max(1.0), "λ={lam} var {var}");
            assert!(xs.iter().all(|x| *x >= 0.0 && x.fract() == 0.0));
        }
    }

    #[test]
    fn kernels_are_normalized_and_padded() {
        let p = BlurKernelParams {
            kind: KernelType::GeneralizedAniso,
            size: 9,
            sigma_x: 1.3,
            sigma_y: 0.7,
            theta: 0.4,
            beta: 1.7,
        };
        let k = pad_kernel(&blur_kernel_f64(&p), 9, KERNEL_SIDE);
        assert_eq!(k.len(), 441);
        assert!((k.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        // zero border outside the centered 9×9
        assert_eq!(k[0], 0.0);
        assert!(k[10 * 21 + 10] > 0.0);
        let s = sinc_kernel_f64(1.2, 13);
        assert!((s.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(usm_kernel().len(), USM_RADIUS * USM_RADIUS);
    }

    #[test]
    fn interpolation_identities_and_sizes() {
        let img = wave(3, 20, 24);
        for mode in Interp::ALL {
            let same = interpolate_scale(&img, 1.0, mode).unwrap();
            assert_eq!(same.height, 20);
            for (a, b) in same.data.iter().zip(&img.data) {
                assert!((a - b).abs() < 1e-6, "{mode:?}");
            }
            let down = interpolate_scale(&img, 0.37, mode).unwrap();
            assert_eq!((down.height, down.width), (7, 8));
            let to = interpolate_size(&img, 11, 5, mode).unwrap();
            assert_eq!((to.height, to.width), (11, 5));
        }
    }

    #[test]
    fn jpeg_is_close_at_high_quality_and_lossy_at_low() {
        let img = wave(3, 24, 40);
        let hi = jpeg(&img, 95.0);
        let lo = jpeg(&img, 10.0);
        let err = |o: &Planes| {
            o.data
                .iter()
                .zip(&img.data)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
        };
        assert!(err(&hi) < err(&lo));
        assert_eq!((hi.height, hi.width), (24, 40));
    }

    #[test]
    fn degradation_is_a_pure_function_of_its_key() {
        let cfg = DegradationConfig::default();
        let clean = wave(3, 192, 160);
        let a = degrade(&cfg, &clean, degradation_key(3, 1, 5)).unwrap();
        let b = degrade(&cfg, &clean, degradation_key(3, 1, 5)).unwrap();
        assert_eq!(a, b);
        let c = degrade(&cfg, &clean, degradation_key(3, 2, 5)).unwrap();
        assert_ne!(a.lq, c.lq);
        assert_eq!((a.lq.height, a.lq.width), (192, 160));
        assert!(a.lq.data.iter().all(|v| (0.0..=1.0).contains(v)));
        // 8-bit levels
        assert!(a
            .lq
            .data
            .iter()
            .all(|v| ((v * 255.0).round() - v * 255.0).abs() < 1e-3));
    }

    #[test]
    fn a_crop_too_small_for_the_kernels_is_refused() {
        let cfg = DegradationConfig::default();
        assert!(cfg.check_crop(1024).is_ok());
        assert!(cfg.check_crop(64).is_err());
        let mut small = cfg.clone();
        small.stage2_scale = 1.0;
        small.stage1.resize_range = (0.8, 1.2);
        small.stage2.resize_range = (0.8, 1.2);
        small.use_sharpener = false;
        assert!(small.check_crop(32).is_ok());
    }

    #[test]
    fn invalid_configs_are_refused() {
        let mut c = DegradationConfig::default();
        c.stage1.resize_range = (1.2, 1.5);
        assert!(c.validate().is_err());
        let mut c = DegradationConfig::default();
        c.kernel1.kernel_prob.pop();
        assert!(c.validate().is_err());
        assert!(DegradationConfig::default().validate().is_ok());
    }
}
