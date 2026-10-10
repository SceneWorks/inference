//! Deterministic helpers shared by the weights-free stage stubs (never by a real stage).

/// SplitMix64 finalizer over an ordered list of inputs — a stable, seedable hash so every stub's
/// output is a pure function of its inputs.
pub(crate) fn hash(parts: &[u64]) -> u64 {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for &p in parts {
        state = state.wrapping_add(p).wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        state = z ^ (z >> 31);
    }
    state
}

/// A deterministic tone of `len` samples in `[-0.25, 0.25]` whose pitch is keyed by `key`.
pub(crate) fn tone(key: u64, len: usize, sample_rate: u32) -> Vec<f32> {
    let freq = 110.0 + (key % 880) as f32;
    let step = std::f32::consts::TAU * freq / sample_rate as f32;
    (0..len).map(|i| 0.25 * (step * i as f32).sin()).collect()
}
