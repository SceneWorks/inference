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
//! [`GenerationRequest::hdr`](gen_core::GenerationRequest::hdr). The SDR path keeps its
//! `clip((x+1)/2, 0, 1)·255 → u8` quantization byte-for-byte; HDR branches at that same seam
//! rather than upstream of it, so a request without `hdr` cannot observe any of this.

use gen_core::hdr::apply_matrix_in_place;
pub use gen_core::hdr::{HdrColorSpace, HdrTransfer, Primaries};
use gen_core::media::HdrFrame;
use gen_core::Error;

// ---------------------------------------------------------------------------------------------
// Primaries
// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------
// Working-space transfer curves
// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------
// HDR colour-space policy
// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------
// VAE range mapping
// ---------------------------------------------------------------------------------------------

/// Map working-space codes `[0, 1]` to the VAE's `[-1, 1]` input convention.
///
/// Deliberately **does not clamp**: a caller handing this out-of-range values has skipped the
/// compress step (or is feeding non-working-space data), and silently clamping would turn that
/// into a quietly wrong render. Returns a typed error instead, matching upstream's `ValueError`.
pub fn to_vae_range(working: &mut [f32]) -> gen_core::Result<()> {
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
    pub fn frame_to_yuv420p10(&self, frame: &HdrFrame) -> gen_core::Result<Yuv420p10> {
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
) -> gen_core::Result<Yuv420p10> {
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
/// [`HdrFrameSink`](gen_core::runtime::HdrFrameSink): [`from_vae_range`] already applied, no
/// transfer yet. For a log working space the codes **are** the EXR payload (no transfer);
/// otherwise they are decompressed to scene-linear light in the authoring primaries.
pub fn working_frame_to_exr_payload(
    working: &HdrFrame,
    color_space: HdrColorSpace,
) -> gen_core::Result<HdrFrame> {
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
) -> gen_core::Result<HdrFrame> {
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
/// (`gen_core::imageops::resize_lanczos_f32`).
pub fn exr_conditioning_to_vae_range(
    frame: &HdrFrame,
    color_space: HdrColorSpace,
) -> gen_core::Result<Vec<f32>> {
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
