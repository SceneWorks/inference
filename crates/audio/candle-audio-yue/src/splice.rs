//! **Seam: low-band splice** — the reference post-process that replaces the 44.1 kHz Vocos mix's
//! band below [`CUTOFF_HZ`] with the energy-matched low band of the 16 kHz codec mix.
//!
//! A plain function seam (no weights). The production function is currently the stub
//! [`splice_stub`]; **sc-19378** replaces [`splice`] with the energy-matched band splice. The stub
//! stays as the end-to-end seam test's double.

use candle_audio::gen_core;

/// The splice crossover.
pub const CUTOFF_HZ: f32 = 5_500.0;

/// The splice seam's signature: `(mix at codec rate, mix at vocoder rate) → final mix at vocoder
/// rate`. The rates are [`crate::codec::SAMPLE_RATE`] and [`crate::vocoder::SAMPLE_RATE`].
pub type SpliceFn = fn(&[f32], &[f32]) -> gen_core::Result<Vec<f32>>;

/// Production splice. **Stub until sc-19378** (delegates to [`splice_stub`]).
pub fn splice(mix_codec_rate: &[f32], mix_vocoder_rate: &[f32]) -> gen_core::Result<Vec<f32>> {
    splice_stub(mix_codec_rate, mix_vocoder_rate)
}

/// **Stub splice** — replaced as the production stage by **sc-19378**. Returns the vocoder-rate mix
/// unchanged.
pub fn splice_stub(
    _mix_codec_rate: &[f32],
    mix_vocoder_rate: &[f32],
) -> gen_core::Result<Vec<f32>> {
    Ok(mix_vocoder_rate.to_vec())
}
