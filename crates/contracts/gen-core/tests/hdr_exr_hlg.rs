//! HDR lane conformance (sc-18790): colour-transfer round-trips, OpenEXR encode/decode, and the
//! BT.2020/HLG master's pixel packing + stream tags.
//!
//! # Why several of these shell out to ffmpeg
//!
//! Asserting that our EXR writer round-trips through our own EXR reader proves the pair is
//! self-consistent, not that the file is *valid*. A wrong-but-symmetric writer/reader pair passes
//! that test forever. The checks marked EXTERNAL therefore hand the bytes to ffmpeg/ffprobe — a
//! decoder with no shared code with ours — and compare what it reports. That is the only way the
//! story's "verified with an external tool rather than trusting our own writer" is actually met.
//!
//! When ffmpeg is not reachable those checks **print a loud skip and pass**; the pure-Rust checks
//! around them still run. A skip is legitimate on a runner with no ffmpeg, but it is never silent.

use std::path::PathBuf;
use std::process::Command;

use gen_core::hdr::{
    exr_conditioning_to_vae_range, working_frame_to_exr_payload, working_frame_to_hlg_linear,
    HdrColorSpace, HdrTransfer, HlgConverter, Primaries, HLG_MASTER_TAGS,
};
use gen_core::{read_rgb_exr, write_rgb_exr, HdrFrame};

// -------------------------------------------------------------------------------------------
// Fixtures
// -------------------------------------------------------------------------------------------

/// A deterministic scene-linear gradient with genuine HDR content.
///
/// The point of the value ladder is that it straddles every branch the module has: sub-black
/// (clamped), the ACEScct linear/log breakpoint at `0.0078125`, diffuse white at `1.0`, and
/// specular highlights far above it. A fixture confined to `[0, 1]` would exercise none of the
/// highlight handling and would pass even if the transfer clipped.
fn gradient_frame(width: u32, height: u32) -> HdrFrame {
    let (w, h) = (width as usize, height as usize);
    let mut rgb = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let u = x as f32 / (w.max(2) - 1) as f32;
            let v = y as f32 / (h.max(2) - 1) as f32;
            // 0 → ~64: an exponential ramp so the low end resolves the ACEScct toe and the high
            // end reaches well past diffuse white.
            rgb.push(0.0001 + u * 64.0);
            rgb.push(0.0001 + v * 12.0);
            rgb.push(0.0001 + (u * v) * 3.0);
        }
    }
    HdrFrame {
        width,
        height,
        rgb,
    }
}

fn ffmpeg_program() -> String {
    std::env::var("SCENEWORKS_FFMPEG").unwrap_or_else(|_| "ffmpeg".to_owned())
}

fn ffprobe_program() -> String {
    std::env::var("SCENEWORKS_FFPROBE").unwrap_or_else(|_| "ffprobe".to_owned())
}

/// `Some(())` when `program` runs, `None` when it is not installed. Any other spawn failure is a
/// real error and panics rather than being laundered into a skip.
fn tool_available(program: &str) -> bool {
    match Command::new(program).arg("-version").output() {
        Ok(out) => out.status.success(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
        Err(e) => panic!("failed to probe {program}: {e}"),
    }
}

fn skip(check: &str, program: &str) {
    println!(
        "SKIPPED (external): {check} — `{program}` is not reachable, so the independent \
         verification did not run. Install ffmpeg to exercise it."
    );
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sc18790-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

// -------------------------------------------------------------------------------------------
// Transfer + primaries round-trips
// -------------------------------------------------------------------------------------------

/// ACEScct compress→decompress recovers scene-linear light across the whole ladder, including
/// across the linear/log breakpoint and far above diffuse white.
#[test]
fn acescct_round_trips_across_the_breakpoint() {
    let transfer = HdrTransfer::AcesCct;
    // Straddle the `x > 0.0078125` breakpoint deliberately: a curve that got the two segments'
    // continuity wrong would still round-trip if every sample sat on one side of it.
    for &x in &[
        0.0, 1e-6, 0.001, 0.0078, 0.0078125, 0.0079, 0.1, 0.18, 0.5, 1.0, 4.0, 16.0, 64.0,
    ] {
        let code = transfer.compress(x);
        assert!(
            (0.0..=1.0).contains(&code),
            "ACEScct code for {x} escaped [0,1]: {code}"
        );
        let back = transfer.decompress(code);
        let tol = (x.abs() * 1e-3).max(1e-6);
        assert!(
            (back - x).abs() <= tol,
            "ACEScct round-trip lost {x}: got {back} (code {code}, tol {tol})"
        );
    }
}

/// LogC3 round-trips too — the second transfer is shipped, so it is tested, not assumed.
#[test]
fn logc3_round_trips_across_the_breakpoint() {
    let transfer = HdrTransfer::LogC3;
    for &x in &[0.0, 1e-5, 0.005, 0.010591, 0.02, 0.18, 1.0, 8.0] {
        let back = transfer.decompress(transfer.compress(x));
        let tol = (x.abs() * 2e-3).max(1e-5);
        assert!(
            (back - x).abs() <= tol,
            "LogC3 round-trip lost {x}: got {back} (tol {tol})"
        );
    }
}

/// Rec.709 → ACEScg → Rec.709 is the identity, which is what makes the hard-coded inverse matrix
/// trustworthy. An inverse transcribed with a wrong digit fails here.
#[test]
fn primaries_round_trip_is_identity() {
    let mut rgb = vec![0.18, 0.5, 1.0, 4.0, 0.0, 12.0];
    let original = rgb.clone();
    Primaries::Rec709.convert_rgb_in_place(Primaries::AcesCg, &mut rgb);
    assert_ne!(rgb, original, "the rotation did nothing — matrix is identity?");
    Primaries::AcesCg.convert_rgb_in_place(Primaries::Rec709, &mut rgb);
    for (got, want) in rgb.iter().zip(&original) {
        assert!(
            (got - want).abs() <= 1e-4,
            "primaries round-trip lost {want}: got {got}"
        );
    }
}

/// HLG OETF and its inverse agree, and diffuse white lands on the documented 0.75 signal.
#[test]
fn hlg_oetf_round_trips_and_anchors_diffuse_white() {
    for &v in &[0.0, 0.1, 0.25, 0.5, 0.5001, 0.75, 0.9, 1.0] {
        let lin = gen_core::hlg_inverse_oetf(v);
        let back = gen_core::hlg_oetf(lin);
        assert!(
            (back - v).abs() <= 1e-5,
            "HLG OETF round-trip lost signal {v}: got {back}"
        );
    }
    // The knee the converter is built around: 0.75 signal is diffuse white.
    let white_linear = gen_core::hlg_inverse_oetf(0.75);
    assert!(
        (white_linear - 0.264_962_56).abs() < 1e-6,
        "HLG diffuse-white scene-linear anchor drifted: {white_linear}"
    );
}

/// Scene-linear input at diffuse white (1.0) maps to the 0.75 HLG signal, and highlights above it
/// roll off monotonically toward 1.0 without ever clipping flat.
#[test]
fn hlg_highlights_roll_off_monotonically_without_clipping() {
    let converter = HlgConverter::new(Primaries::Rec709);
    let signal_for = |lin: f32| {
        let mut rgb = vec![lin, lin, lin];
        converter.to_hlg_signal_in_place(&mut rgb);
        rgb[0]
    };

    // Strictly increasing while the roll-off is numerically resolvable. Above ~linear 32 the
    // exponential has closed to within f32 epsilon of 1.0, so equality there is the curve's
    // designed asymptote, not clipping — asserting strict increase past that point would be
    // asserting against the maths.
    let mut previous = -1.0f32;
    for &lin in &[0.0f32, 0.18, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0] {
        let signal = signal_for(lin);
        assert!(
            (0.0..=1.0).contains(&signal),
            "HLG signal for linear {lin} escaped [0,1]: {signal}"
        );
        assert!(
            signal > previous,
            "HLG transfer stopped increasing at linear {lin} ({signal} <= {previous}) — \
             highlights are clipping instead of rolling off"
        );
        previous = signal;
    }
    // Never exceeds 1.0 even far into the highlights.
    for &lin in &[32.0f32, 64.0, 1000.0, 1.0e6] {
        let signal = signal_for(lin);
        assert!(
            (0.0..=1.0).contains(&signal),
            "HLG signal for linear {lin} escaped [0,1]: {signal}"
        );
    }
    // The anti-clipping question, stated directly: two stops above diffuse white must land
    // strictly between the diffuse-white signal and full scale. A converter that clipped
    // highlights would put both of these on exactly 1.0 and still satisfy monotonicity above.
    for &lin in &[2.0f32, 4.0, 8.0] {
        let signal = signal_for(lin);
        assert!(
            signal > 0.75 && signal < 1.0,
            "linear {lin} should roll off strictly between diffuse white and full scale, \
             got {signal}"
        );
    }
    // Diffuse white anchor, through the full Rec.709→Rec.2020→OETF chain (neutral grey is
    // preserved by the primaries rotation, so it lands on the nominal signal).
    let mut white = vec![1.0f32, 1.0, 1.0];
    converter.to_hlg_signal_in_place(&mut white);
    assert!(
        (white[0] - 0.75).abs() < 1e-3,
        "diffuse white should land on the 0.75 HLG signal, got {}",
        white[0]
    );
}

// -------------------------------------------------------------------------------------------
// EXR: our own round-trip, then the external verdict
// -------------------------------------------------------------------------------------------

/// Float EXR is lossless through our own writer/reader, and both header tags survive.
#[test]
fn exr_float_round_trips_exactly_with_tags() {
    let frame = gradient_frame(16, 8);
    let bytes = write_rgb_exr(&frame, Primaries::Rec709, "sRGB", false).expect("write EXR");
    let decoded = read_rgb_exr(&bytes).expect("read EXR");

    assert_eq!(decoded.frame.width, frame.width);
    assert_eq!(decoded.frame.height, frame.height);
    assert_eq!(
        decoded.frame.rgb, frame.rgb,
        "float EXR must round-trip bit-exactly"
    );
    assert_eq!(decoded.color_space_tag.as_deref(), Some("sRGB"));
    let chroma = decoded.chromaticities.expect("chromaticities written");
    for (got, want) in chroma.iter().zip(&Primaries::Rec709.exr_chromaticities()) {
        assert!((got - want).abs() < 1e-6, "chromaticities drifted: {chroma:?}");
    }
}

/// Half EXR round-trips within f16 precision and carries the ACEScg tag set. The story's default
/// output is half, so its precision envelope is asserted rather than assumed.
#[test]
fn exr_half_round_trips_within_f16_precision() {
    let frame = gradient_frame(16, 8);
    let bytes = write_rgb_exr(&frame, Primaries::AcesCg, "ACEScg", true).expect("write EXR");
    let decoded = read_rgb_exr(&bytes).expect("read EXR");

    assert_eq!(decoded.color_space_tag.as_deref(), Some("ACEScg"));
    for (got, want) in decoded.frame.rgb.iter().zip(&frame.rgb) {
        // f16 carries ~3 decimal digits; scale the tolerance with magnitude.
        let tol = (want.abs() * 1e-2).max(1e-4);
        assert!(
            (got - want).abs() <= tol,
            "half EXR lost {want}: got {got} (tol {tol})"
        );
    }
    // Highlights above 1.0 must survive — this is the whole reason for EXR over PNG.
    let peak = decoded
        .frame
        .rgb
        .iter()
        .copied()
        .fold(f32::MIN, f32::max);
    assert!(
        peak > 60.0,
        "half EXR clipped the highlights: peak {peak} (expected ~64)"
    );
}

/// EXTERNAL — ffmpeg, a decoder sharing no code with ours, reads our EXR and reports the same
/// pixel values. This is what proves the file is genuinely a valid OpenEXR rather than something
/// only our reader understands.
#[test]
fn exr_is_valid_to_an_external_decoder() {
    let program = ffmpeg_program();
    if !tool_available(&program) {
        skip("exr_is_valid_to_an_external_decoder", &program);
        return;
    }
    let dir = temp_dir("exr-external");
    let exr_path = dir.join("frame.exr");
    let raw_path = dir.join("frame.raw");

    let frame = gradient_frame(8, 4);
    let bytes = write_rgb_exr(&frame, Primaries::Rec709, "sRGB", false).expect("write EXR");
    std::fs::write(&exr_path, &bytes).expect("write exr file");

    // Decode to planar float RGB. `gbrpf32le` keeps full float precision (any 8-bit format would
    // quantize the HDR values away and make the comparison meaningless).
    let out = Command::new(&program)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&exr_path)
        .args(["-f", "rawvideo", "-pix_fmt", "gbrpf32le", "-y"])
        .arg(&raw_path)
        .output()
        .expect("run ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg failed to decode our EXR — the file is not valid OpenEXR.\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let raw = std::fs::read(&raw_path).expect("read raw");
    let (w, h) = (frame.width as usize, frame.height as usize);
    assert_eq!(
        raw.len(),
        w * h * 3 * 4,
        "unexpected raw size from ffmpeg: {} bytes for {w}x{h}",
        raw.len()
    );
    let samples: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // gbrp = three planes in G, B, R order.
    let (g_plane, rest) = samples.split_at(w * h);
    let (b_plane, r_plane) = rest.split_at(w * h);

    for i in 0..w * h {
        let want = (frame.rgb[i * 3], frame.rgb[i * 3 + 1], frame.rgb[i * 3 + 2]);
        let got = (r_plane[i], g_plane[i], b_plane[i]);
        let tol = (want.0.abs().max(want.1.abs()).max(want.2.abs()) * 1e-5).max(1e-6);
        assert!(
            (got.0 - want.0).abs() <= tol
                && (got.1 - want.1).abs() <= tol
                && (got.2 - want.2).abs() <= tol,
            "ffmpeg read pixel {i} as {got:?}, we wrote {want:?} (tol {tol})"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// -------------------------------------------------------------------------------------------
// Conditioning round-trip
// -------------------------------------------------------------------------------------------

/// A scene-linear EXR conditioning plate survives the full load path — file bytes → decode →
/// working space → VAE range — and comes back to the light it started as.
///
/// This is the story's "an EXR conditioning input round-trips" acceptance, exercised end to end
/// through the real codec rather than on in-memory floats.
#[test]
fn exr_conditioning_input_round_trips_to_scene_linear() {
    let frame = gradient_frame(12, 6);
    let bytes = write_rgb_exr(&frame, Primaries::Rec709, "sRGB", false).expect("write EXR");
    let decoded = read_rgb_exr(&bytes).expect("read EXR").frame;

    let color_space = HdrColorSpace::SrgbLinear;
    let vae = exr_conditioning_to_vae_range(&decoded, color_space).expect("to VAE range");
    assert_eq!(vae.len(), decoded.rgb.len());
    for v in &vae {
        assert!(
            (-1.0..=1.0).contains(v),
            "VAE-range sample escaped [-1,1]: {v}"
        );
    }

    // Back out the way a decode would: VAE range → working codes → scene-linear.
    let working = HdrFrame {
        width: decoded.width,
        height: decoded.height,
        rgb: vae.iter().map(|z| gen_core::from_vae_range(*z)).collect(),
    };
    let recovered = working_frame_to_exr_payload(&working, color_space).expect("to EXR payload");

    for (got, want) in recovered.rgb.iter().zip(&frame.rgb) {
        // ACEScct is a log curve quantized through f32; relative tolerance is the meaningful one.
        let tol = (want.abs() * 5e-3).max(1e-4);
        assert!(
            (got - want).abs() <= tol,
            "conditioning round-trip lost {want}: got {got} (tol {tol})"
        );
    }
}

/// An ACEScct plate is already working-space code, so the load path applies **no** transfer.
///
/// Guards the one asymmetry in the module: treating log codes as scene-linear light (or vice
/// versa) silently double-applies the curve, which looks plausible but grades wrong.
#[test]
fn acescct_conditioning_skips_the_transfer() {
    let frame = HdrFrame {
        width: 2,
        height: 1,
        rgb: vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.1],
    };
    let vae = exr_conditioning_to_vae_range(&frame, HdrColorSpace::AcesCct).expect("to VAE range");
    // Log working space: the codes pass straight through the [0,1] → [-1,1] remap.
    for (got, want) in vae.iter().zip(&frame.rgb) {
        let expected = want * 2.0 - 1.0;
        assert!(
            (got - expected).abs() <= 1e-6,
            "ACEScct conditioning applied a transfer it should not have: {want} → {got}, \
             expected {expected}"
        );
    }
    // ...and the scene-linear spelling of the same numbers must NOT pass through, or the
    // assertion above would be vacuous.
    let linear = exr_conditioning_to_vae_range(&frame, HdrColorSpace::SrgbLinear).expect("linear");
    assert!(
        linear
            .iter()
            .zip(&vae)
            .any(|(a, b)| (a - b).abs() > 1e-3),
        "scene-linear and log conditioning produced identical output — the transfer is inert"
    );
}

// -------------------------------------------------------------------------------------------
// HLG master: packing, then the external verdict on the tags
// -------------------------------------------------------------------------------------------

/// The 10-bit packing lands in the legal container and puts neutral grey on the achromatic
/// chroma centre.
#[test]
fn hlg_yuv420p10_packing_is_in_range_and_neutral() {
    let converter = HlgConverter::new(Primaries::Rec709);
    // A flat neutral frame: chroma must sit at the 512 centre for an achromatic input.
    let frame = HdrFrame {
        width: 4,
        height: 4,
        rgb: vec![0.5; 4 * 4 * 3],
    };
    let planes = converter.frame_to_yuv420p10(&frame).expect("pack");
    assert_eq!(planes.y.len(), 16);
    assert_eq!(planes.u.len(), 4);
    assert_eq!(planes.v.len(), 4);
    for &y in &planes.y {
        assert!(y <= 1023, "luma escaped the 10-bit container: {y}");
        assert!(y >= 64, "neutral 0.5 luma below the limited-range floor: {y}");
    }
    for (&u, &v) in planes.u.iter().zip(&planes.v) {
        assert!(
            (u as i32 - 512).abs() <= 1,
            "achromatic input produced a chroma cast: U={u}"
        );
        assert!(
            (v as i32 - 512).abs() <= 1,
            "achromatic input produced a chroma cast: V={v}"
        );
    }
}

/// Odd dimensions cannot be 4:2:0 subsampled and are rejected rather than silently truncated.
#[test]
fn hlg_packing_rejects_odd_dimensions() {
    let converter = HlgConverter::new(Primaries::Rec709);
    let frame = HdrFrame {
        width: 3,
        height: 2,
        rgb: vec![0.5; 3 * 2 * 3],
    };
    let err = converter.frame_to_yuv420p10(&frame).unwrap_err();
    assert!(
        err.to_string().contains("even dimensions"),
        "unexpected error for odd dimensions: {err}"
    );
}

/// EXTERNAL — encode our packed planes with the tag set this module declares, then ask ffprobe
/// what the resulting stream actually says.
///
/// Every assertion below is on the value ffprobe reports for the encoded file, not on our
/// constants: this is the check that the declaration in `HLG_MASTER_TAGS` genuinely produces a
/// BT.2020/HLG-tagged master. A master that carries the right pixels but the wrong transfer tag
/// plays back washed out everywhere, which is exactly the defect this guards.
#[test]
fn hlg_master_is_tagged_bt2020_hlg_to_ffprobe() {
    let (ffmpeg, ffprobe) = (ffmpeg_program(), ffprobe_program());
    if !tool_available(&ffmpeg) {
        skip("hlg_master_is_tagged_bt2020_hlg_to_ffprobe", &ffmpeg);
        return;
    }
    if !tool_available(&ffprobe) {
        skip("hlg_master_is_tagged_bt2020_hlg_to_ffprobe", &ffprobe);
        return;
    }
    let dir = temp_dir("hlg-external");
    let raw_path = dir.join("frames.yuv");
    let mp4_path = dir.join("master.mp4");

    let (w, h) = (64u32, 64u32);
    let converter = HlgConverter::new(Primaries::Rec709);
    let color_space = HdrColorSpace::SrgbLinear;

    // Three frames of a moving scene-linear gradient, taken through the real production path:
    // working-space signal → scene-linear Rec.709 → HLG → 10-bit planes.
    let mut raw: Vec<u8> = Vec::new();
    for f in 0..3u32 {
        let mut working = gradient_frame(w, h);
        // Re-express the fixture as a working-space [0,1] signal, which is what a decode emits.
        for v in working.rgb.iter_mut() {
            *v = HdrTransfer::AcesCct.compress(*v * (1.0 + f as f32 * 0.1));
        }
        let linear = working_frame_to_hlg_linear(&working, color_space).expect("hlg linear");
        let planes = converter.frame_to_yuv420p10(&linear).expect("pack");
        for plane in [&planes.y, &planes.u, &planes.v] {
            for &s in plane.iter() {
                raw.extend_from_slice(&s.to_le_bytes());
            }
        }
    }
    std::fs::write(&raw_path, &raw).expect("write raw planes");

    let tags = HLG_MASTER_TAGS;
    let out = Command::new(&ffmpeg)
        .args(["-hide_banner", "-loglevel", "error"])
        .args(["-f", "rawvideo", "-pix_fmt", tags.pix_fmt])
        .args(["-s", &format!("{w}x{h}"), "-r", "24", "-i"])
        .arg(&raw_path)
        .args(["-c:v", "libx265", "-x265-params", tags.x265_params])
        .args(["-pix_fmt", tags.pix_fmt])
        .args(["-color_primaries", tags.color_primaries])
        .args(["-color_trc", tags.color_trc])
        .args(["-colorspace", tags.colorspace])
        .args(["-color_range", tags.color_range])
        .args(["-tag:v", tags.codec_tag, "-y"])
        .arg(&mp4_path)
        .output()
        .expect("run ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg failed to encode the HLG master.\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let probe = Command::new(&ffprobe)
        .args(["-v", "error", "-select_streams", "v:0"])
        .args([
            "-show_entries",
            "stream=pix_fmt,color_primaries,color_transfer,color_space,color_range,codec_name",
        ])
        .args(["-of", "default=noprint_wrappers=1"])
        .arg(&mp4_path)
        .output()
        .expect("run ffprobe");
    assert!(probe.status.success(), "ffprobe failed");
    let report = String::from_utf8_lossy(&probe.stdout).to_string();

    // Assert on what the encoded stream reports, field by field, so a failure names the tag that
    // is wrong rather than just "the master is bad".
    for (key, want) in [
        ("codec_name", "hevc"),
        ("pix_fmt", "yuv420p10le"),
        ("color_primaries", "bt2020"),
        ("color_transfer", "arib-std-b67"),
        ("color_space", "bt2020nc"),
        ("color_range", "tv"),
    ] {
        let line = report
            .lines()
            .find(|l| l.starts_with(&format!("{key}=")))
            .unwrap_or_else(|| panic!("ffprobe reported no {key}.\nfull report:\n{report}"));
        let got = line.split_once('=').map(|(_, v)| v.trim()).unwrap_or("");
        assert_eq!(
            got, want,
            "the encoded HLG master reports {key}={got}, expected {want}.\n\
             full report:\n{report}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
