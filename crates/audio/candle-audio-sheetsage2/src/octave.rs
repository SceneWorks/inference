//! Octave evidence for the review artifact: a spectral check of each transcribed vocal note
//! against the audio the model was fed (the heuristic of sc-23003's `octave_check`, ported so every
//! transcription carries it).
//!
//! For each note: the interior `start + 50 ms .. end − 30 ms` (at least 2,048 samples), a
//! symmetric-Hann-windowed 65,536-point spectrum (longer interiors are truncated to 65,536 samples,
//! exactly as `numpy.fft.rfft(x, n)` does), and the peak magnitude within ±3 % of the transcribed
//! f0, of f0/2 and of 2·f0. A note whose f0/2 peak exceeds its f0 peak *looks* an octave high.
//! Accompaniment can also put energy at f0/2, so this is evidence for a reviewer, never a
//! correction: the model's pitches are left untouched.

/// Evidence for one vocal note.
#[derive(Clone, Debug, PartialEq)]
pub struct NoteEvidence {
    /// Note start, seconds.
    pub start: f64,
    /// Transcribed MIDI pitch.
    pub midi: i64,
    /// Peak magnitude around f0/2.
    pub energy_f0_half: f64,
    /// Peak magnitude around f0.
    pub energy_f0: f64,
    /// Peak magnitude around 2·f0.
    pub energy_2f0: f64,
}

/// The whole check.
#[derive(Clone, Debug, PartialEq)]
pub struct OctaveEvidence {
    /// Every transcribed vocal pitch's range `(min, max)`.
    pub transcribed_midi_range: Option<(i64, i64)>,
    /// Notes long enough to check.
    pub notes: Vec<NoteEvidence>,
}

impl OctaveEvidence {
    /// Notes whose f0/2 peak beats their f0 peak.
    pub fn f0_half_dominant(&self) -> usize {
        self.notes
            .iter()
            .filter(|n| n.energy_f0_half > n.energy_f0)
            .count()
    }

    /// Fraction of checked notes that are f0/2-dominant.
    pub fn fraction_f0_half_dominant(&self) -> Option<f64> {
        (!self.notes.is_empty()).then(|| self.f0_half_dominant() as f64 / self.notes.len() as f64)
    }
}

const SIZE: usize = 1 << 16;

struct Fft64 {
    twiddles: Vec<(f64, f64)>,
    bitrev: Vec<usize>,
}

impl Fft64 {
    fn new() -> Self {
        let bits = SIZE.trailing_zeros();
        Self {
            twiddles: (0..SIZE / 2)
                .map(|k| {
                    let a = -2.0 * std::f64::consts::PI * k as f64 / SIZE as f64;
                    (a.cos(), a.sin())
                })
                .collect(),
            bitrev: (0..SIZE)
                .map(|i| i.reverse_bits() >> (usize::BITS - bits))
                .collect(),
        }
    }

    /// Magnitudes of the first `SIZE / 2 + 1` bins of the real input (zero-padded / truncated).
    fn magnitudes(&self, input: &[f64]) -> Vec<f64> {
        let mut data = vec![(0.0f64, 0.0f64); SIZE];
        for (slot, &x) in data.iter_mut().zip(input) {
            slot.0 = x;
        }
        for i in 0..SIZE {
            let j = self.bitrev[i];
            if i < j {
                data.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= SIZE {
            let stride = SIZE / len;
            for start in (0..SIZE).step_by(len) {
                for k in 0..len / 2 {
                    let (wr, wi) = self.twiddles[k * stride];
                    let (ar, ai) = data[start + k];
                    let (br, bi) = data[start + k + len / 2];
                    let (tr, ti) = (br * wr - bi * wi, br * wi + bi * wr);
                    data[start + k] = (ar + tr, ai + ti);
                    data[start + k + len / 2] = (ar - tr, ai - ti);
                }
            }
            len <<= 1;
        }
        data[..SIZE / 2 + 1]
            .iter()
            .map(|(r, i)| r.hypot(*i))
            .collect()
    }
}

/// Run the check on mono `audio` at `rate` Hz for the vocal notes `(start, end, midi)`.
pub fn octave_evidence(audio: &[f32], rate: u32, notes: &[(f64, f64, i64)]) -> OctaveEvidence {
    let fft = Fft64::new();
    let rate_f = f64::from(rate);
    let bin_hz = rate_f / SIZE as f64;
    let mut checked = Vec::new();
    for &(start, end, midi) in notes {
        // Python int() truncates toward zero; a negative bound clamps to the start.
        let lo = (((start + 0.05) * rate_f) as i64).max(0) as usize;
        let hi = ((((end - 0.03) * rate_f) as i64).max(0) as usize).min(audio.len());
        if hi <= lo || hi - lo < 2048 {
            continue;
        }
        let segment = &audio[lo..hi];
        let m = segment.len();
        // numpy.hanning(M): symmetric, 0.5 − 0.5·cos(2πn / (M − 1)).
        let windowed: Vec<f64> = segment
            .iter()
            .enumerate()
            .take(SIZE)
            .map(|(n, &x)| {
                let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (m - 1) as f64).cos();
                f64::from(x) * w
            })
            .collect();
        let spectrum = fft.magnitudes(&windowed);
        let peak = |hz: f64| {
            let (lo, hi) = (hz * 0.97, hz * 1.03);
            spectrum
                .iter()
                .enumerate()
                .filter(|(k, _)| {
                    let f = *k as f64 * bin_hz;
                    f > lo && f < hi
                })
                .map(|(_, v)| *v)
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let f0 = 440.0 * 2f64.powf((midi as f64 - 69.0) / 12.0);
        checked.push(NoteEvidence {
            start,
            midi,
            energy_f0_half: peak(f0 / 2.0),
            energy_f0: peak(f0),
            energy_2f0: peak(2.0 * f0),
        });
    }
    let range = notes
        .iter()
        .map(|n| n.2)
        .fold(None, |acc: Option<(i64, i64)>, p| {
            Some(acc.map_or((p, p), |(a, b)| (a.min(p), b.max(p))))
        });
    OctaveEvidence {
        transcribed_midi_range: range,
        notes: checked,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pure tone transcribed an octave too high is flagged; transcribed correctly it is not.
    /// Mutation that must fail: compare against 2·f0 instead of f0/2.
    #[test]
    fn a_tone_labelled_an_octave_high_is_flagged() {
        let rate = 24_000u32;
        let hz = 220.0; // A3 = MIDI 57
        let audio: Vec<f32> = (0..rate as usize)
            .map(|n| (2.0 * std::f64::consts::PI * hz * n as f64 / f64::from(rate)).sin() as f32)
            .collect();
        let high = octave_evidence(&audio, rate, &[(0.0, 1.0, 69)]);
        assert_eq!(high.f0_half_dominant(), 1);
        let right = octave_evidence(&audio, rate, &[(0.0, 1.0, 57)]);
        assert_eq!(right.f0_half_dominant(), 0);
        assert_eq!(right.transcribed_midi_range, Some((57, 57)));
        // Too short to check.
        assert!(octave_evidence(&audio, rate, &[(0.0, 0.1, 57)])
            .notes
            .is_empty());
    }
}
