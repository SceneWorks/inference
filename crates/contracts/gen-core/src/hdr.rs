//! HDR colour science for the LTX-2.5 HDR path (sc-18790).
//!
//! A port of the LTX-2 v1.2.0 reference (`ltx_core.color.primaries`, `ltx_core.hdr`,
//! `ltx_core.color.hlg`, `ltx_core.color.yuv`, `ltx_pipelines.utils.media_io.color_config`,
//! `…media_io.range_map`) at upstream commit `d1511477`. Tensor-free like the rest of gen-core:
//! everything here operates on interleaved `f32` RGB (`HWC`), so the contract layer owns the
//! colour policy and each backend only supplies pixels.
//!
//! # The two directions
//!
//! **In (conditioning).** A scene-linear EXR still/sequence is rotated into the transfer's native
//! primaries, compressed to the `[0, 1]` working space ([`HdrTransfer::to_working_space`]) and
//! mapped to the VAE's `[-1, 1]` convention ([`to_vae_range`]). An [`HdrColorSpace::AcesCct`]
//! source is *already* working-space log code, so it skips the transfer and is only clamped —
//! this asymmetry is [`HdrColorSpace::is_log_working`] and is the single most load-bearing
//! distinction in this module.
//!
//! **Out (render).** VAE decode yields the same `[0, 1]` working-space signal. It feeds two
//! independent sinks: scene-linear EXR frames (via [`HdrTransfer::to_linear`], or the raw log
//! codes when `is_log_working`) and a BT.2020/HLG master (always Rec.709 scene-linear →
//! [`HlgConverter`] → [`Yuv420p10`], tagged with [`HLG_MASTER_TAGS`]).
//!
//! # SDR is the default and is untouched
//!
//! Nothing in this module runs unless a caller opts in with
//! [`GenerationRequest::hdr`](crate::GenerationRequest::hdr). The SDR path keeps its
//! `clip((x+1)/2, 0, 1)·255 → u8` quantization byte-for-byte; HDR branches at that same seam
//! rather than upstream of it, so a request without `hdr` cannot observe any of this.

use crate::media::HdrFrame;
use crate::Error;

// ---------------------------------------------------------------------------------------------
// Primaries
// ---------------------------------------------------------------------------------------------

/// ACEScg (AP1, D60-adapted) linear → linear sRGB / Rec.709 (D65) primaries.
// Transcribed from the reference at full precision on purpose: the extra digits carry no f32
// information, but they keep each entry diff-able against the upstream Python source, which is
// how a transposed or mistyped coefficient gets caught by eye. Truncating them to f32's actual
// resolution would break that audit for no numerical gain.
#[allow(clippy::excessive_precision)]
const ACESCG_TO_SRGB: [[f32; 3]; 3] = [
    [1.705_05, -0.621_79, -0.083_26],
    [-0.130_26, 1.140_8, -0.010_55],
    [-0.024_00, -0.128_97, 1.152_97],
];

/// Exact inverse of [`ACESCG_TO_SRGB`], inverted in `f64` then rounded to `f32` so the
/// round-trip is clean (upstream computes this at import time with `torch.linalg.inv`).
#[allow(clippy::excessive_precision)] // full-precision transcription — see ACESCG_TO_SRGB
const SRGB_TO_ACESCG: [[f32; 3]; 3] = [
    [0.613_098_5, 0.339_524_18, 0.047_380_731],
    [0.070_196_077, 0.916_359_01, 0.013_454_048],
    [0.020_614_197, 0.109_570_42, 0.869_816_48],
];

/// Rec.709 (D65) → Rec.2020 (D65), ITU-R BT.2087.
#[allow(clippy::excessive_precision)] // full-precision transcription — see ACESCG_TO_SRGB
const REC709_TO_2020: [[f32; 3]; 3] = [
    [0.627_403_89, 0.329_283_04, 0.043_313_07],
    [0.069_097_29, 0.919_540_4, 0.011_362_32],
    [0.016_391_44, 0.088_013_31, 0.895_595_25],
];

/// ACEScg (AP1, D60) → Rec.2020 (D65), Bradford CAT.
#[allow(clippy::excessive_precision)] // full-precision transcription — see ACESCG_TO_SRGB
const ACESCG_TO_2020: [[f32; 3]; 3] = [
    [1.025_824_75, -0.020_053_19, -0.005_771_56],
    [-0.002_234_37, 1.004_586_5, -0.002_352_13],
    [-0.005_013_35, -0.025_290_07, 1.030_303_42],
];

/// Authoring / working-space colour primaries (linear light).
///
/// Rec.2020 is deliberately **not** a member: it is a conversion *target* for the HLG master
/// ([`Primaries::matrix_to_rec2020`]), never an EXR authoring space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Primaries {
    /// Rec.709 / sRGB primaries, D65 white.
    Rec709,
    /// ACEScg (AP1) primaries, D60 white.
    AcesCg,
}

impl Primaries {
    /// The EXR `chromaticities` header attribute: `R.x R.y G.x G.y B.x B.y W.x W.y`.
    pub fn exr_chromaticities(self) -> [f32; 8] {
        match self {
            Primaries::Rec709 => [0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290],
            Primaries::AcesCg => [0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767],
        }
    }

    /// 3×3 linear map from this basis to Rec.2020 (the HLG master space).
    pub fn matrix_to_rec2020(self) -> [[f32; 3]; 3] {
        match self {
            Primaries::Rec709 => REC709_TO_2020,
            Primaries::AcesCg => ACESCG_TO_2020,
        }
    }

    /// 3×3 linear map from this basis into `target`, or `None` when they are the same basis
    /// (the identity — callers skip the multiply entirely).
    pub fn matrix_to(self, target: Primaries) -> Option<[[f32; 3]; 3]> {
        match (self, target) {
            (Primaries::Rec709, Primaries::Rec709) | (Primaries::AcesCg, Primaries::AcesCg) => None,
            (Primaries::Rec709, Primaries::AcesCg) => Some(SRGB_TO_ACESCG),
            (Primaries::AcesCg, Primaries::Rec709) => Some(ACESCG_TO_SRGB),
        }
    }

    /// Convert interleaved linear-light RGB from this basis into `target`, in place. A no-op
    /// when the bases match.
    pub fn convert_rgb_in_place(self, target: Primaries, rgb: &mut [f32]) {
        if let Some(m) = self.matrix_to(target) {
            apply_matrix_in_place(&m, rgb);
        }
    }
}

/// Apply a 3×3 matrix to every interleaved RGB triple in `rgb`, in place. A trailing partial
/// triple is left untouched; callers validate buffer length up front (see [`HdrFrame::validate`]).
fn apply_matrix_in_place(m: &[[f32; 3]; 3], rgb: &mut [f32]) {
    for px in rgb.chunks_exact_mut(3) {
        let (r, g, b) = (px[0], px[1], px[2]);
        px[0] = m[0][0] * r + m[0][1] * g + m[0][2] * b;
        px[1] = m[1][0] * r + m[1][1] * g + m[1][2] * b;
        px[2] = m[2][0] * r + m[2][1] * g + m[2][2] * b;
    }
}

// ---------------------------------------------------------------------------------------------
// Working-space transfer curves
// ---------------------------------------------------------------------------------------------

// ARRI LogC3 (EI 800).
const LOGC3_A: f32 = 5.555_556;
const LOGC3_B: f32 = 0.052_272;
const LOGC3_C: f32 = 0.247_190;
const LOGC3_D: f32 = 0.385_537;
const LOGC3_E: f32 = 5.367_655;
const LOGC3_F: f32 = 0.092_809;
const LOGC3_CUT: f32 = 0.010_591;

// AMPAS S-2016-001 ACEScct.
const ACESCCT_A_LIN: f32 = 10.540_238;
const ACESCCT_B_LIN: f32 = 0.072_905_53;
const ACESCCT_X_BRK: f32 = 0.0078125;
const ACESCCT_Y_BRK: f32 = 0.155_251_14;
const ACESCCT_LOG_M: f32 = 17.52;
const ACESCCT_LOG_B: f32 = 9.72;

/// The VAE HDR working-space transfer curve — a bijection between linear HDR `[0, ∞)` and a
/// compressed `[0, 1]` signal.
///
/// Mapping the `[0, 1]` signal to and from the VAE's `[-1, 1]` convention is the *caller's* job
/// ([`to_vae_range`] on the way in, [`from_vae_range`] on the way out), exactly as upstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HdrTransfer {
    /// ARRI LogC3 (EI 800) — the HDR IC-LoRA default; defined on Rec.709 primaries.
    LogC3,
    /// ACEScct log working space — the EXR conditioning round-trip; defined on ACEScg primaries.
    AcesCct,
}

impl HdrTransfer {
    /// Human / EXR `colorSpace` tag name.
    pub fn display_name(self) -> &'static str {
        match self {
            HdrTransfer::LogC3 => "LogC3",
            HdrTransfer::AcesCct => "ACEScct",
        }
    }

    /// The linear primaries this curve is defined on — [`decompress`](Self::decompress) lands
    /// here, and [`compress`](Self::compress) expects to be handed light already in this basis.
    pub fn native_primaries(self) -> Primaries {
        match self {
            HdrTransfer::LogC3 => Primaries::Rec709,
            HdrTransfer::AcesCct => Primaries::AcesCg,
        }
    }

    /// Compress one linear-HDR sample `[0, ∞)` to the working space `[0, 1]`.
    pub fn compress(self, x: f32) -> f32 {
        let x = x.max(0.0);
        let v = match self {
            HdrTransfer::LogC3 => {
                if x >= LOGC3_CUT {
                    LOGC3_C * (LOGC3_A * x + LOGC3_B).log10() + LOGC3_D
                } else {
                    LOGC3_E * x + LOGC3_F
                }
            }
            HdrTransfer::AcesCct => {
                if x > ACESCCT_X_BRK {
                    (x.max(1e-12).log2() + ACESCCT_LOG_B) / ACESCCT_LOG_M
                } else {
                    ACESCCT_A_LIN * x + ACESCCT_B_LIN
                }
            }
        };
        v.clamp(0.0, 1.0)
    }

    /// Decompress one working-space sample `[0, 1]` to linear HDR `[0, ∞)`.
    pub fn decompress(self, v: f32) -> f32 {
        let v = v.clamp(0.0, 1.0);
        match self {
            HdrTransfer::LogC3 => {
                let cut_log = LOGC3_E * LOGC3_CUT + LOGC3_F;
                if v >= cut_log {
                    (10f32.powf((v - LOGC3_D) / LOGC3_C) - LOGC3_B) / LOGC3_A
                } else {
                    (v - LOGC3_F) / LOGC3_E
                }
            }
            HdrTransfer::AcesCct => {
                if v > ACESCCT_Y_BRK {
                    2f32.powf(v * ACESCCT_LOG_M - ACESCCT_LOG_B)
                } else {
                    (v - ACESCCT_B_LIN) / ACESCCT_A_LIN
                }
            }
        }
    }

    /// Scene-linear HDR in `source_primaries` → this transfer's compressed `[0, 1]` working
    /// space, in place. Rotates into [`native_primaries`](Self::native_primaries) first.
    pub fn to_working_space(self, rgb: &mut [f32], source_primaries: Primaries) {
        source_primaries.convert_rgb_in_place(self.native_primaries(), rgb);
        for v in rgb.iter_mut() {
            *v = self.compress(v.max(0.0));
        }
    }

    /// This transfer's `[0, 1]` working space → scene-linear HDR in `out_primaries`, in place.
    /// Decompresses into [`native_primaries`](Self::native_primaries), rotates, then clamps
    /// negatives introduced by the gamut rotation back to zero.
    pub fn to_linear(self, working: &mut [f32], out_primaries: Primaries) {
        for v in working.iter_mut() {
            *v = self.decompress(*v);
        }
        self.native_primaries()
            .convert_rgb_in_place(out_primaries, working);
        for v in working.iter_mut() {
            *v = v.max(0.0);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// HDR colour-space policy
// ---------------------------------------------------------------------------------------------

/// The explicit HDR source / working colour space a request opts into.
///
/// `Option::<HdrColorSpace>::None` at a call site means **SDR** — the default everywhere. This is
/// the Rust spelling of upstream's `--hdr {SRGB_LINEAR,ACESCG,ACESCCT}` CLI flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HdrColorSpace {
    /// Scene-linear Rec.709 / sRGB-tagged EXR. Compressed to ACEScct on load.
    SrgbLinear,
    /// Scene-linear ACEScg EXR. Compressed to ACEScct on load.
    AcesCg,
    /// Already ACEScct log working codes — passed through on load (no transfer).
    AcesCct,
}

impl HdrColorSpace {
    /// Every variant, in declaration order — the parse/serialize round-trip surface.
    pub const ALL: [HdrColorSpace; 3] = [
        HdrColorSpace::SrgbLinear,
        HdrColorSpace::AcesCg,
        HdrColorSpace::AcesCct,
    ];

    /// True when the VAE signal is already ACEScct log code, so loading applies **no** transfer
    /// and writing EXR emits the raw working codes rather than scene-linear light.
    pub fn is_log_working(self) -> bool {
        matches!(self, HdrColorSpace::AcesCct)
    }

    /// The linear primaries the *source* media is authored in.
    pub fn source_primaries(self) -> Primaries {
        match self {
            HdrColorSpace::AcesCg | HdrColorSpace::AcesCct => Primaries::AcesCg,
            HdrColorSpace::SrgbLinear => Primaries::Rec709,
        }
    }

    /// The curve for the VAE working space. Always defined; whether it is *applied* depends on
    /// the call site (load compresses only when not [`is_log_working`](Self::is_log_working);
    /// EXR write decompresses only when writing linear; the HLG master always decompresses).
    pub fn transfer(self) -> HdrTransfer {
        HdrTransfer::AcesCct
    }

    /// `(primaries, colorSpace tag)` for the EXR header this colour space writes.
    pub fn exr_output_tags(self) -> (Primaries, &'static str) {
        match self {
            HdrColorSpace::AcesCct => (Primaries::AcesCg, "ACEScct"),
            HdrColorSpace::AcesCg => (Primaries::AcesCg, "ACEScg"),
            HdrColorSpace::SrgbLinear => (Primaries::Rec709, "sRGB"),
        }
    }

    /// The stable wire spelling (matches upstream's CLI enum values).
    pub fn as_str(self) -> &'static str {
        match self {
            HdrColorSpace::SrgbLinear => "srgb_linear",
            HdrColorSpace::AcesCg => "acescg",
            HdrColorSpace::AcesCct => "acescct",
        }
    }

    /// Parse the wire spelling, case-insensitively. `None` for an unknown value — callers turn
    /// that into a typed request rejection rather than silently defaulting to a colour space.
    pub fn parse(s: &str) -> Option<HdrColorSpace> {
        let s = s.trim().to_ascii_lowercase();
        HdrColorSpace::ALL.into_iter().find(|c| c.as_str() == s)
    }
}

impl std::fmt::Display for HdrColorSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------------------------
// VAE range mapping
// ---------------------------------------------------------------------------------------------

/// Map working-space codes `[0, 1]` to the VAE's `[-1, 1]` input convention.
///
/// Deliberately **does not clamp**: a caller handing this out-of-range values has skipped the
/// compress step (or is feeding non-working-space data), and silently clamping would turn that
/// into a quietly wrong render. Returns a typed error instead, matching upstream's `ValueError`.
pub fn to_vae_range(working: &mut [f32]) -> crate::Result<()> {
    for v in working.iter() {
        if !(0.0..=1.0).contains(v) {
            return Err(Error::Msg(format!(
                "to_vae_range expects working-space codes in [0, 1]; got {v}. Compress with \
                 HdrTransfer::to_working_space first, or pass already-log working codes."
            )));
        }
    }
    for v in working.iter_mut() {
        *v = *v * 2.0 - 1.0;
    }
    Ok(())
}

/// Map the VAE's `[-1, 1]` output convention back to working-space codes `[0, 1]`.
///
/// This is the **shared** first step of both the SDR and HDR output paths: SDR then scales by
/// 255 to `u8`, HDR then decompresses to scene-linear. Clamping here is correct (and matches
/// upstream) because a decoder legitimately overshoots the range slightly.
pub fn from_vae_range(z: f32) -> f32 {
    ((z + 1.0) / 2.0).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------------------------
// Hybrid Log-Gamma (ITU-R BT.2100 / ARIB STD-B67)
// ---------------------------------------------------------------------------------------------

/// ARIB STD-B67 (HLG) OETF constant `a`.
pub const HLG_A: f32 = 0.178_832_77;
/// ARIB STD-B67 (HLG) OETF constant `b`.
pub const HLG_B: f32 = 0.284_668_92;
/// ARIB STD-B67 (HLG) OETF constant `c`.
#[allow(clippy::excessive_precision)] // full-precision transcription — see ACESCG_TO_SRGB
pub const HLG_C: f32 = 0.559_910_73;

/// The HLG signal value diffuse white maps to, when the caller does not choose one. Upstream's
/// `white_signal` default: 75 % signal is the conventional HLG diffuse-white reference.
pub const HLG_DEFAULT_WHITE_SIGNAL: f32 = 0.75;

/// HLG OETF: scene-linear `[0, ∞)` → signal `[0, 1]`.
pub fn hlg_oetf(x: f32) -> f32 {
    let x = x.max(0.0);
    if x <= 1.0 / 12.0 {
        (3.0 * x).max(0.0).sqrt()
    } else {
        HLG_A * (12.0 * x - HLG_B).max(1e-12).ln() + HLG_C
    }
}

/// Inverse HLG OETF: signal `[0, 1]` → scene-linear.
pub fn hlg_inverse_oetf(v: f32) -> f32 {
    if v <= 0.5 {
        (v * v) / 3.0
    } else {
        (((v - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
    }
}

/// Scene-linear HDR → HLG signal, for one choice of source primaries and diffuse-white anchor.
///
/// Linear light at `1.0` is treated as **diffuse white** and lands on `white_signal`; everything
/// above it is rolled off exponentially toward `1.0` so specular highlights compress smoothly
/// instead of clipping. `rolloff_k` defaults to `white_x / (1 - white_x)`, which makes the
/// roll-off C¹-continuous with the linear segment at the diffuse-white knee.
#[derive(Clone, Copy, Debug)]
pub struct HlgConverter {
    /// Source basis → Rec.2020, applied before the transfer.
    prim_mat: [[f32; 3]; 3],
    /// Scene-linear value of `white_signal`, i.e. `hlg_inverse_oetf(white_signal)`.
    white_x: f32,
    /// Highlight roll-off rate above diffuse white.
    roll_k: f32,
}

impl HlgConverter {
    /// Build a converter from `primaries` at the default diffuse-white anchor.
    pub fn new(primaries: Primaries) -> Self {
        Self::with_white_signal(primaries, HLG_DEFAULT_WHITE_SIGNAL, None)
    }

    /// Build a converter, choosing the diffuse-white signal and (optionally) the roll-off rate.
    pub fn with_white_signal(
        primaries: Primaries,
        white_signal: f32,
        rolloff_k: Option<f32>,
    ) -> Self {
        let white_x = hlg_inverse_oetf(white_signal);
        let roll_k = rolloff_k.unwrap_or(white_x / (1.0 - white_x));
        Self {
            prim_mat: primaries.matrix_to_rec2020(),
            white_x,
            roll_k,
        }
    }

    /// Map one scene-linear Rec.2020 sample through the diffuse-white anchor + highlight
    /// roll-off, then the HLG OETF.
    fn signal_from_rec2020_linear(&self, lin: f32) -> f32 {
        // Upstream's `nan_to_num(nan=0.0, neginf=0.0)` leaves POSITIVE infinity alone, so it flows
        // through the roll-off below and lands on peak white. Flushing it to zero instead — as an
        // `is_finite()` guard does — turns a blown highlight into a black hole, which is both
        // wrong and the more alarming failure on screen.
        let lin = if lin.is_nan() {
            0.0
        } else if lin == f32::INFINITY {
            f32::MAX
        } else {
            lin.max(0.0)
        };
        let x = if lin <= 1.0 {
            lin * self.white_x
        } else {
            1.0 - (1.0 - self.white_x) * (-self.roll_k * (lin - 1.0)).exp()
        };
        hlg_oetf(x).clamp(0.0, 1.0)
    }

    /// Convert interleaved scene-linear RGB (in the converter's source primaries) to an HLG
    /// signal in `[0, 1]`, in place. NaN and negative infinity are flushed to zero; positive
    /// infinity rolls off to peak white (upstream `nan_to_num` semantics).
    pub fn to_hlg_signal_in_place(&self, rgb: &mut [f32]) {
        apply_matrix_in_place(&self.prim_mat, rgb);
        for v in rgb.iter_mut() {
            *v = self.signal_from_rec2020_linear(*v);
        }
    }

    /// Convert one scene-linear HDR frame to planar 10-bit YUV 4:2:0, BT.2020 non-constant
    /// luminance, limited ("MPEG") range — the exact pixel payload the HLG master is encoded
    /// from. See [`HLG_MASTER_TAGS`] for the tags that must accompany it.
    ///
    /// `Err` when the frame's dimensions are odd (4:2:0 needs even edges) or its buffer length
    /// disagrees with its dimensions.
    pub fn frame_to_yuv420p10(&self, frame: &HdrFrame) -> crate::Result<Yuv420p10> {
        frame.validate()?;
        let (w, h) = (frame.width as usize, frame.height as usize);
        if w % 2 != 0 || h % 2 != 0 {
            return Err(Error::Msg(format!(
                "frame_to_yuv420p10: 4:2:0 chroma subsampling needs even dimensions, got {w}×{h}"
            )));
        }
        let mut rgb = frame.rgb.clone();
        self.to_hlg_signal_in_place(&mut rgb);
        Ok(rgb_signal_to_yuv420p10_unchecked(&rgb, w, h))
    }
}

/// Planar 10-bit YUV 4:2:0 (`yuv420p10le`) code levels for one frame.
///
/// `y` is `width × height`; `u` and `v` are each `(width / 2) × (height / 2)`. Values are the
/// encoder's code levels already — clamped to `[0, 1023]`, limited-range scaled — so a consumer
/// hands these straight to `libx265` without a further conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Yuv420p10 {
    /// Luma plane width in samples (equals the frame width).
    pub width: u32,
    /// Luma plane height in samples (equals the frame height).
    pub height: u32,
    /// Luma plane, row-major, `width · height` samples.
    pub y: Vec<u16>,
    /// Cb plane, row-major, `(width / 2) · (height / 2)` samples.
    pub u: Vec<u16>,
    /// Cr plane, row-major, `(width / 2) · (height / 2)` samples.
    pub v: Vec<u16>,
}

// BT.2020 non-constant-luminance luma coefficients.
const KR_2020: f32 = 0.2627;
const KG_2020: f32 = 0.6780;
const KB_2020: f32 = 0.0593;

/// RGB **signal** (already transfer-encoded, `[0, 1]`) → planar 10-bit YUV 4:2:0, BT.2020 NCL,
/// **limited ("tv") range**. Chroma is box-averaged over 2×2 before the code-level scale (the
/// scale is affine, so averaging before or after is identical — this matches upstream's
/// `avg_pool2d`).
///
/// Public because the code-level scaling is a distinct, independently-wrong-able stage: it is what
/// puts `E' = 0` on luma **64** and `E' = 1` on luma **940** rather than on 0 and 1023. A payload
/// written full-range but tagged `tv` — the tags in [`HLG_MASTER_TAGS`] say limited — is re-stretched
/// by every player, crushing blacks and clipping highlights. `ffprobe` cannot catch that: it reads
/// container tags, not sample values. So the scaling is exposed here and asserted at the extremes
/// directly, on the signal, without the HLG curve in the way.
///
/// `Err` on odd dimensions (4:2:0 needs even edges) or a buffer inconsistent with them.
pub fn rgb_signal_to_yuv420p10(
    signal_rgb: &[f32],
    width: u32,
    height: u32,
) -> crate::Result<Yuv420p10> {
    let (w, h) = (width as usize, height as usize);
    if width == 0 || height == 0 || w % 2 != 0 || h % 2 != 0 {
        return Err(Error::Msg(format!(
            "rgb_signal_to_yuv420p10: 4:2:0 chroma subsampling needs non-zero even dimensions, got {width}×{height}"
        )));
    }
    if signal_rgb.len() != w * h * 3 {
        return Err(Error::Msg(format!(
            "rgb_signal_to_yuv420p10: buffer length {} disagrees with {width}×{height} RGB (need {})",
            signal_rgb.len(),
            w * h * 3
        )));
    }
    Ok(rgb_signal_to_yuv420p10_unchecked(signal_rgb, w, h))
}

fn rgb_signal_to_yuv420p10_unchecked(rgb: &[f32], w: usize, h: usize) -> Yuv420p10 {
    let (cw, ch) = (w / 2, h / 2);
    let mut y_plane = vec![0u16; w * h];
    // Full-resolution chroma, averaged into the subsampled planes below.
    let mut cb_full = vec![0f32; w * h];
    let mut cr_full = vec![0f32; w * h];

    for i in 0..w * h {
        let (r, g, b) = (rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
        let y = KR_2020 * r + KG_2020 * g + KB_2020 * b;
        // Cb/Cr are centred on 0 here; the code-level scale re-centres them on 512.
        cb_full[i] = (b - y) / 1.8814;
        cr_full[i] = (r - y) / 1.4746;
        // Limited-range 10-bit luma: (219·E' + 16) · 2^(10-8).
        y_plane[i] = quantize_code(y * 876.0 + 64.0);
    }

    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for cy in 0..ch {
        for cx in 0..cw {
            let mut cb = 0f32;
            let mut cr = 0f32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let i = (cy * 2 + dy) * w + (cx * 2 + dx);
                    cb += cb_full[i];
                    cr += cr_full[i];
                }
            }
            // Limited-range 10-bit chroma: (224·E' + 128) · 2^(10-8).
            u[cy * cw + cx] = quantize_code(cb / 4.0 * 896.0 + 512.0);
            v[cy * cw + cx] = quantize_code(cr / 4.0 * 896.0 + 512.0);
        }
    }

    Yuv420p10 {
        width: w as u32,
        height: h as u32,
        y: y_plane,
        u,
        v,
    }
}

/// Round to the nearest integer code and clamp into the 10-bit container.
///
/// The clamp is the full `[0, 1023]` container, not the limited-range `[64, 940]` window: the
/// limited-range *scale* is already applied, and BT.2100 keeps the head/footroom outside that
/// window legal. Clamping tighter would crush legal super-white/sub-black excursions.
fn quantize_code(v: f32) -> u16 {
    if !v.is_finite() {
        return 0;
    }
    v.round().clamp(0.0, 1023.0) as u16
}

// ---------------------------------------------------------------------------------------------
// HLG master stream tags
// ---------------------------------------------------------------------------------------------

/// The exact container/codec tags a BT.2020/HLG master must carry.
///
/// These are **not** decoration: HLG signal in an untagged (or BT.709-tagged) stream is displayed
/// with the wrong transfer and washes out everywhere. The pixel payload from
/// [`HlgConverter::frame_to_yuv420p10`] is only correct when muxed with exactly these values, so
/// they travel together as one declaration rather than being re-typed at each call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HlgMasterTags {
    /// FFmpeg `-pix_fmt`.
    pub pix_fmt: &'static str,
    /// FFmpeg `-color_primaries` (`AVCOL_PRI_BT2020` = 9).
    pub color_primaries: &'static str,
    /// FFmpeg `-color_trc` (`AVCOL_TRC_ARIB_STD_B67` = 18 — HLG).
    pub color_trc: &'static str,
    /// FFmpeg `-colorspace` (`AVCOL_SPC_BT2020_NCL` = 9).
    pub colorspace: &'static str,
    /// FFmpeg `-color_range` (`AVCOL_RANGE_MPEG` = 1 — limited).
    pub color_range: &'static str,
    /// The `libx265` parameter string that writes the same tags into the HEVC VUI, so the
    /// signalling survives remuxing out of the container.
    pub x265_params: &'static str,
    /// MP4 sample-entry tag. `hvc1` (not `hev1`) is what QuickTime/AVFoundation will play.
    pub codec_tag: &'static str,
}

/// The one true tag set for the HLG master (see [`HlgMasterTags`]).
pub const HLG_MASTER_TAGS: HlgMasterTags = HlgMasterTags {
    pix_fmt: "yuv420p10le",
    color_primaries: "bt2020",
    color_trc: "arib-std-b67",
    colorspace: "bt2020nc",
    color_range: "tv",
    x265_params: "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited",
    codec_tag: "hvc1",
};

/// `AVCOL_PRI_BT2020` — the numeric `color_primaries` `ffprobe` reports for a correct master.
pub const AVCOL_PRI_BT2020: u32 = 9;
/// `AVCOL_TRC_ARIB_STD_B67` — the numeric HLG `color_trc`.
pub const AVCOL_TRC_ARIB_STD_B67: u32 = 18;
/// `AVCOL_SPC_BT2020_NCL` — the numeric `colorspace`.
pub const AVCOL_SPC_BT2020_NCL: u32 = 9;

// ---------------------------------------------------------------------------------------------
// Decode / conditioning helpers
// ---------------------------------------------------------------------------------------------

/// One VAE-decoded frame in the **working space** (`[0, 1]` compressed codes) → the frame that
/// gets written to EXR, for `color_space`.
///
/// `working` is what an engine hands an
/// [`HdrFrameSink`](crate::runtime::HdrFrameSink): [`from_vae_range`] already applied, no
/// transfer yet. For a log working space the codes **are** the EXR payload (no transfer);
/// otherwise they are decompressed to scene-linear light in the authoring primaries.
pub fn working_frame_to_exr_payload(
    working: &HdrFrame,
    color_space: HdrColorSpace,
) -> crate::Result<HdrFrame> {
    working.validate()?;
    let mut rgb = working.rgb.clone();
    if !color_space.is_log_working() {
        color_space
            .transfer()
            .to_linear(&mut rgb, color_space.source_primaries());
    }
    Ok(HdrFrame {
        width: working.width,
        height: working.height,
        rgb,
    })
}

/// One VAE-decoded working-space frame → the scene-linear **Rec.709** frame the HLG master is
/// encoded from.
///
/// Always Rec.709 regardless of `color_space`'s authoring basis (upstream does the same): the
/// [`HlgConverter`] owns the rotation into Rec.2020, so pinning its input basis keeps exactly one
/// matrix in play on the master path.
pub fn working_frame_to_hlg_linear(
    working: &HdrFrame,
    color_space: HdrColorSpace,
) -> crate::Result<HdrFrame> {
    working.validate()?;
    let mut rgb = working.rgb.clone();
    color_space
        .transfer()
        .to_linear(&mut rgb, Primaries::Rec709);
    Ok(HdrFrame {
        width: working.width,
        height: working.height,
        rgb,
    })
}

/// An EXR conditioning frame → the VAE's `[-1, 1]` input range, for `color_space`.
///
/// The load-side counterpart of [`working_frame_to_exr_payload`], and the round-trip partner the
/// story's "an EXR conditioning input round-trips" acceptance rests on. A log working space is
/// only clamped (the codes are already the VAE signal); a scene-linear space is compressed
/// through the transfer first.
///
/// Geometry is the caller's: hand this a frame already at the target resolution
/// (`crate::imageops::resize_lanczos_f32`).
pub fn exr_conditioning_to_vae_range(
    frame: &HdrFrame,
    color_space: HdrColorSpace,
) -> crate::Result<Vec<f32>> {
    frame.validate()?;
    let mut rgb = frame.rgb.clone();
    if color_space.is_log_working() {
        // Already working-space codes; clamp file noise into the legal range.
        for v in rgb.iter_mut() {
            *v = v.clamp(0.0, 1.0);
        }
    } else {
        color_space
            .transfer()
            .to_working_space(&mut rgb, color_space.source_primaries());
    }
    to_vae_range(&mut rgb)?;
    Ok(rgb)
}
