//! sc-11671: **reference-matching** host RNGs for the Bernini MAR reveal loop — a numpy-compatible
//! MT19937 (`np.random.shuffle`) for the reveal permutation and a torch-compatible CPU MT19937
//! (`torch.randn` / `normal_fill`) for the per-step flow-match noise. These are the *actual* draws the
//! reference `sample_vit_embed` performs (`_vendor/bernini/pipeline.py`): the reveal order via numpy's
//! legacy `RandomState.shuffle` (MT19937 + Fisher-Yates over `rk_interval`), and the per-step FM base
//! noise via `torch.randn` (torch CPU MT19937 → `normal_fill` Box–Muller). Reimplementing both bit-for-bit
//! lets the candle MAR trajectory match torch/numpy exactly (golden: `tests/fixtures/mar_mt19937_golden`,
//! dumped by `tools/dump_bernini_mar_mt19937_golden.py`).
//!
//! **One MT19937 core.** numpy's legacy `RandomState(seed)` (for an integer seed ≤ `2³²−1`) and torch's
//! `at::CPUGeneratorImpl` both seed the *same* standard MT19937 via `init_genrand` (constant
//! `1812433253`, low-32-bit seed) and both temper with the standard MT recurrence, so [`Mt19937`] serves
//! both. They differ only in what they draw on top: numpy's shuffle consumes bounded 32-bit integers
//! (`random_interval`), torch's `normal_fill` consumes 24-bit `float` uniforms fed through Box–Muller.
//!
//! A u64 request seed is reduced to its low 32 bits for both (numpy's legacy scalar seeding rejects
//! seeds ≥ 2³², and torch's MT19937 seeds from `seed & 0xffffffff` anyway).

/// Standard MT19937 (Matsumoto–Nishimura) — the shared core of numpy legacy `RandomState` and torch's
/// CPU generator.
const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// A standard MT19937 32-bit generator, seeded exactly as numpy's `mt19937_seed` / torch's
/// `MT19937RNGEngine::seed` do (`init_genrand`).
pub struct Mt19937 {
    key: [u32; N],
    pos: usize,
}

impl Mt19937 {
    /// Seed with the low 32 bits of `seed` via `init_genrand` (identical numpy/torch seeding).
    pub fn seed(seed: u32) -> Self {
        let mut key = [0u32; N];
        let mut s = seed;
        for (i, k) in key.iter_mut().enumerate() {
            *k = s;
            s = 1812433253u32
                .wrapping_mul(s ^ (s >> 30))
                .wrapping_add(i as u32 + 1);
        }
        Mt19937 { key, pos: N }
    }

    /// Regenerate the 624-word block (standard MT twist).
    fn regenerate(&mut self) {
        let mag01 = [0u32, MATRIX_A];
        let k = &mut self.key;
        for i in 0..(N - M) {
            let y = (k[i] & UPPER_MASK) | (k[i + 1] & LOWER_MASK);
            k[i] = k[i + M] ^ (y >> 1) ^ mag01[(y & 1) as usize];
        }
        for i in (N - M)..(N - 1) {
            let y = (k[i] & UPPER_MASK) | (k[i + 1] & LOWER_MASK);
            k[i] = k[i + M - N] ^ (y >> 1) ^ mag01[(y & 1) as usize];
        }
        let y = (k[N - 1] & UPPER_MASK) | (k[0] & LOWER_MASK);
        k[N - 1] = k[M - 1] ^ (y >> 1) ^ mag01[(y & 1) as usize];
        self.pos = 0;
    }

    /// Next tempered 32-bit word (`genrand_int32`).
    pub fn next_u32(&mut self) -> u32 {
        if self.pos >= N {
            self.regenerate();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// numpy `random_interval(max)` — a uniform integer in the **inclusive** range `[0, max]` via
    /// masked rejection on 32-bit words (the bound stays ≤ `u32::MAX` for every MAR reveal size).
    fn interval(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let v = self.next_u32() & mask;
            if v <= max {
                return v;
            }
        }
    }

    /// torch `uniform_real_distribution<float>(0,1)` — one 32-bit word → a 24-bit-mantissa `float` in
    /// `[0, 1)`.
    fn uniform_f32(&mut self) -> f32 {
        const FLOAT_MASK: u32 = (1 << 24) - 1;
        const FLOAT_DIVISOR: f32 = 1.0 / (1u32 << 24) as f32;
        (self.next_u32() & FLOAT_MASK) as f32 * FLOAT_DIVISOR
    }
}

/// numpy legacy `RandomState(seed)` then `np.random.shuffle(np.arange(n))`: an in-place Fisher–Yates
/// that walks `i` from `n−1` down to `1`, swapping `x[i]` with `x[random_interval(i)]`. Returns the
/// reveal permutation of `[0, n)`.
pub fn numpy_shuffle(n: usize, seed: u32) -> Vec<i32> {
    let mut mt = Mt19937::seed(seed);
    let mut x: Vec<i32> = (0..n as i32).collect();
    let mut i = n;
    while i > 1 {
        i -= 1;
        let j = mt.interval(i as u32) as usize;
        x.swap(i, j);
    }
    x
}

/// torch `torch.randn(size)` (CPU, `float32`) via `normal_fill`: fill `size` uniforms, Box–Muller each
/// aligned block of 16, then (when `size % 16 != 0`) **recompute** the final 16 from fresh uniforms.
/// Advances `mt` by exactly the number of words torch consumes (`size`, plus 16 more for the tail
/// recompute), so sequential `torch_randn` calls on one generator stay word-aligned with torch.
///
/// Requires `size >= 16` — the `normal_fill` regime torch takes for any non-tiny tensor, which every MAR
/// draw satisfies (`n_revealed · hidden`, `hidden` in the thousands). For `size < 16` (never reached in
/// the MAR loop) this pads to a 16-word draw and truncates — deterministic, but **not** torch-parity.
pub fn torch_randn(mt: &mut Mt19937, size: usize) -> Vec<f32> {
    if size < 16 {
        let mut buf = [0f32; 16];
        for b in buf.iter_mut() {
            *b = mt.uniform_f32();
        }
        normal_fill_16(&mut buf);
        return buf[..size].to_vec();
    }
    let mut data = vec![0f32; size];
    for d in data.iter_mut() {
        *d = mt.uniform_f32();
    }
    let mut i = 0;
    while i + 16 <= size {
        normal_fill_16(&mut data[i..i + 16]);
        i += 16;
    }
    if !size.is_multiple_of(16) {
        let base = size - 16;
        for k in 0..16 {
            data[base + k] = mt.uniform_f32();
        }
        normal_fill_16(&mut data[base..]);
    }
    data
}

/// torch `normal_fill_16`: in-place Box–Muller over a 16-element window — indices `0..8` carry the
/// uniforms `u_j`, `8..16` carry `u_{j+8}`; each pair emits `(radius·cosθ, radius·sinθ)`. Math mirrors
/// torch's `scalar_t=float` ordering (`θ = (float)(2·π_f64·u2)`, `radius = sqrtf(-2·lnf(1-u1))`).
fn normal_fill_16(data: &mut [f32]) {
    const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
    for j in 0..8 {
        let u1 = 1.0f32 - data[j];
        let u2 = data[j + 8];
        let radius = (-2.0f32 * u1.ln()).sqrt();
        let theta = (TWO_PI * u2 as f64) as f32;
        data[j] = radius * theta.cos();
        data[j + 8] = radius * theta.sin();
    }
}

/// Per-step reference FM base noise for the MAR loop, as raw row-major `f32` (torch `normal_fill`).
///
/// One torch generator is seeded once (low 32 bits of `seed`) and drawn **sequentially** across the
/// planning steps, mirroring the reference: `torch.randn(n_revealed, in_channels)` inside the
/// `revealed.sum() != 0` branch. A step whose revealed set is empty **or** is `{token 0}` alone
/// (`sum == 0`, the reference's `nonzero().sum()==0` skip) draws nothing and yields an empty `Vec`, so
/// the returned outer `Vec` is indexed by planning step and stays aligned with
/// [`crate::mar::sample_vit_embed`]'s `step_noise[step]` lookups. `schedule` is
/// [`crate::mar::mar_schedule`]'s output (sorted revealed positions per step).
pub fn torch_step_noise(schedule: &[Vec<i32>], in_channels: usize, seed: u32) -> Vec<Vec<f32>> {
    let mut mt = Mt19937::seed(seed);
    schedule
        .iter()
        .map(|revealed| {
            if revealed.iter().sum::<i32>() == 0 {
                Vec::new()
            } else {
                torch_randn(&mut mt, revealed.len() * in_channels)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// numpy `RandomState(1234).shuffle(arange(10))` (dumped once from numpy 2.4) — bit-exact pin so a
    /// core/seeding regression is caught without the safetensors golden.
    #[test]
    fn numpy_shuffle_matches_known_vector() {
        assert_eq!(numpy_shuffle(10, 1234), vec![7, 2, 9, 1, 0, 8, 4, 5, 6, 3]);
    }

    /// `torch.Generator().manual_seed(1234)` → `torch.randn(20, dtype=float32)` first values (dumped once
    /// from torch 2.13 CPU) — pins the `normal_fill` Box–Muller + tail recompute.
    #[test]
    fn torch_randn_matches_known_vector() {
        let mut mt = Mt19937::seed(1234);
        let r = torch_randn(&mut mt, 20);
        let expect = [
            -0.111_718_57f32,
            -0.496_590_1,
            0.163_073_7,
            -0.881_687_76,
            0.289_097_2,
            0.489_870_85,
        ];
        for (i, &e) in expect.iter().enumerate() {
            assert!((r[i] - e).abs() < 1e-6, "randn[{i}] {} vs {e}", r[i]);
        }
    }
}
