//! Native reproducible random streams. These preserve source distributions, not
//! PyTorch seed-stream identity. Stream domains prevent cache/previews consuming
//! the next training step's noise; checkpoint metadata names this algorithm.
use gen_core::{Error, Result};
use sha2::{Digest, Sha256};
pub const RNG_ALGORITHM: &str = "flux3-splitmix64-boxmuller-f32-v1";
pub struct NativeRng {
    state: u64,
}
impl NativeRng {
    pub fn keyed(seed: u64, domain: &str, index: u64) -> Self {
        let mut hash = Sha256::new();
        hash.update(RNG_ALGORITHM.as_bytes());
        hash.update(seed.to_le_bytes());
        hash.update((domain.len() as u64).to_le_bytes());
        hash.update(domain.as_bytes());
        hash.update(index.to_le_bytes());
        let digest = hash.finalize();
        Self {
            state: u64::from_le_bytes(digest[..8].try_into().expect("SHA256 prefix")),
        }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as u32) as f32 * (1.0 / 16_777_216.0)
    }
    pub fn normal(&mut self) -> f32 {
        let u = 1.0 - self.uniform();
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
    }
    pub fn bounded(&mut self, bound: usize) -> Result<usize> {
        if bound == 0 {
            return Err(Error::Unsupported("flux3 RNG: empty range".into()));
        }
        let bound = bound as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return Ok((value % bound) as usize);
            }
        }
    }
    pub fn shuffle<T>(&mut self, values: &mut [T]) -> Result<()> {
        for i in (1..values.len()).rev() {
            let j = self.bounded(i + 1)?;
            values.swap(i, j);
        }
        Ok(())
    }
}
/// A step uses independently keyed streams, so its complete next-state is this
/// algorithm/version + seed + next micro-step. Epoch/cursor are separate fields.
pub fn state_record(seed: u64, next_micro_step: u64) -> Vec<u8> {
    let mut bytes = RNG_ALGORITHM.as_bytes().to_vec();
    bytes.push(0);
    bytes.extend_from_slice(&seed.to_le_bytes());
    bytes.extend_from_slice(&next_micro_step.to_le_bytes());
    bytes
}
pub fn validate_state(bytes: &[u8], seed: u64, next_micro_step: u64) -> Result<()> {
    if bytes != state_record(seed, next_micro_step) {
        Err(Error::Unsupported(
            "flux3 resume RNG algorithm/seed/counter mismatch; refusing reset".into(),
        ))
    } else {
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streams_are_explicit_reproducible_and_resume_state_is_strict() {
        let mut a = NativeRng::keyed(73, "noise", 9);
        let mut b = NativeRng::keyed(73, "noise", 9);
        let mut other = NativeRng::keyed(73, "caption", 9);
        let values = (0..64).map(|_| a.normal().to_bits()).collect::<Vec<_>>();
        assert_eq!(
            values,
            (0..64).map(|_| b.normal().to_bits()).collect::<Vec<_>>()
        );
        assert_ne!(values[0], other.normal().to_bits());
        for _ in 0..100 {
            let u = a.uniform();
            assert!((0.0..1.0).contains(&u));
            assert!(a.normal().is_finite());
        }
        let record = state_record(73, 9);
        validate_state(&record, 73, 9).unwrap();
        assert!(validate_state(&record, 73, 10).is_err());
        assert!(validate_state(&record, 74, 9).is_err());
    }
}
