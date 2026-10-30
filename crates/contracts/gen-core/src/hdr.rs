//! HDR colour policy for the LTX-2.5 HDR path (sc-18790): the colour space a request opts into,
//! its primaries and its working-space transfer. Tensor-free; everything operates on interleaved
//! `f32` RGB. The HLG master, the 10-bit YUV packing, the VAE-range maps and the EXR payload
//! conversions live in `gen_core_image::hdr` (sc-27316).

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
/// triple is left untouched; callers validate buffer length up front (see [`HdrFrame::validate`](crate::media::HdrFrame::validate)).
#[doc(hidden)]
pub fn apply_matrix_in_place(m: &[[f32; 3]; 3], rgb: &mut [f32]) {
    for px in rgb.chunks_exact_mut(3) {
        let (r, g, b) = (px[0], px[1], px[2]);
        px[0] = m[0][0] * r + m[0][1] * g + m[0][2] * b;
        px[1] = m[1][0] * r + m[1][1] * g + m[1][2] * b;
        px[2] = m[2][0] * r + m[2][1] * g + m[2][2] * b;
    }
}
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
/// (`gen_core_image::hdr::to_vae_range` on the way in, `gen_core_image::hdr::from_vae_range` on the way out), exactly as upstream.
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
