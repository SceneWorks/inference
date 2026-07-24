//! Gaussian-Shading watermarked initial noise + watermark detection — **owned by sc-14104**.
//!
//! Port of `_vendor/mage_flow/models/modules/mage_latent.py`. **Not optional for parity:** the
//! reference computes a plain `randn` via `get_noise` and then *discards* it, replacing the initial
//! latent with `encode_noise(shape, key=…, seed=…)` on both the generation (`pipeline.py:307`) and
//! edit (`:506`) paths. There is no toggle — every real Mage-Flow output carries the watermark, and
//! a port that starts from plain `randn` cannot match any golden (measured max_abs 5.99 apart).
//!
//! Epic posture (recorded on sc-14105): **keep provenance marking, drop blocking.** This module
//! ships; the reference's mandatory content classifier does not. The in-repo precedent is
//! `candle-audio-chatterbox/src/model.rs:530-531`, which applies the PerTh provenance watermark
//! unconditionally.
//!
//! # Mechanism
//!
//! 1. The payload [`GS_PAYLOAD`] is SHA-256-expanded into a
//!    [`GS_MESSAGE_BITS`]-bit message ([`payload_to_bits`]).
//! 2. The key seeds a per-entry XOR pad **and** a per-entry message index ([`pad_and_pos`]).
//! 3. `target_half[i] = msg[pos[i]] ^ pad[i] ∈ {0, 1}` picks which half of the standard normal
//!    entry `i` must land in.
//! 4. `u[i] ~ U(0,1)` comes from a *seed*-driven stream, independent of the key.
//! 5. `z[i] = Φ⁻¹(clamp((target_half[i] + u[i]) / 2, 1e-6, 1 − 1e-6))`.
//!
//! Because `u` is uniform on each half-interval, `z` is still exactly ~N(0,1): the watermark is
//! distribution-preserving, so nothing downstream changes shape or scale. Detection ([`decode_bits`])
//! reads only the **signs** — `z[i] > 0 ⟺ target_half[i] = 1` — which is what lets it survive the
//! lossy round trip back through the flow ODE ([`invert_to_noise`]).
//!
//! # Bit-exactness: two RNG streams, reproduced directly
//!
//! The reference mixes two *different* generators, and both had to be reproduced exactly. This is a
//! hash-like construction: one off-by-one draw yields an entirely different tensor rather than a
//! slightly different one, so "close" is indistinguishable from "wrong".
//!
//! | stream | reference | consumer |
//! |---|---|---|
//! | `np.random.default_rng(key)` | numpy `SeedSequence` → PCG64 (XSL-RR) → Lemire bounded ints | XOR pad + message-index map (`mage_latent.py:68-74`) |
//! | `torch.Generator(seed)` | torch CPU `at::mt19937` → `random64()` → 53-bit uniform | per-entry `u` (`mage_latent.py:83-84`) |
//!
//! **No crate was adopted for either.** Both algorithms are short and fully specified, and the
//! parity golden is a bit-exact oracle, so the private `numpy_rng` and `torch_rng` submodules
//! implement them directly: the dependency graph is unchanged, and correctness is established
//! against the reference's own output rather than against a crate's README.
//!
//! The unit tests below pin raw draws captured from numpy 2.4.3 and torch 2.13.0 — the versions
//! `_vendor/mage_flow/requirements.txt` pins — across **four keys** (single-limb, two-limb, the
//! eight-limb passphrase, and zero) and **four seeds**, each stream separately so a failure names
//! the culprit. `tests/gaussian_shading_parity.rs` then pins the assembled tensor, and its
//! `#[ignore]`d golden check compares all 32 768 entries: **0 mismatches at float32 and at
//! bfloat16**, while the discarded plain `randn` sits 5.98846 away in max_abs.
//!
//! # Divergences from the reference, deliberate
//!
//! 1. **Key provisioning.** `resolve_gs_key` (`mage_latent.py:38-52`) reads a `MAGEFLOW_GS_KEY` env
//!    var and a `~/.mageflow/gs_key` keyfile. **Neither is ported:** this workspace derives no paths
//!    and reads no production env side channels (the epic-13657 guardrail in
//!    `scripts/check-workspace.py`). [`resolve_gs_key`] takes the key as an explicit argument and
//!    falls back to [`GS_DEFAULT_KEY`]; sc-14041 threads it in from
//!    `LoadSpec`/the request. Every other part of key normalisation — integer, digit string,
//!    passphrase-via-SHA-256, arbitrary precision — is ported faithfully ([`GsKey`]).
//! 2. **Compute precision.** The reference derives `z` in float64 and casts once at the end
//!    (`mage_latent.py:83-91`). Metal has no float64, so [`encode_noise_host`] performs the whole
//!    derivation on the host in `f64` and only then hands a contiguous `f32` buffer to MLX — the
//!    same single-narrowing path torch takes for a double source, not a float32 re-derivation.

use mlx_rs::{Array, Dtype};
use sha2::{Digest, Sha256};

use mlx_gen::{Error, Result};

use crate::config::{GS_DEFAULT_KEY, GS_MESSAGE_BITS, GS_PAYLOAD};

/// One-sided binomial p-value below which [`decode_bits`] reports the watermark present
/// (`mage_latent.py:130`).
pub const GS_PRESENT_PVALUE: f64 = 1e-6;

/// Clamp applied to the inverse-normal-CDF argument (`mage_latent.py:86`).
const GS_ARG_CLAMP: f64 = 1e-6;

// =================================================================================================
// Key normalisation
// =================================================================================================

/// A normalised Gaussian-Shading key: a non-negative arbitrary-precision integer, held as
/// little-endian `u32` limbs in exactly the form numpy's `_int_to_uint32_array` produces.
///
/// Mirrors `_key_to_int` (`mage_latent.py:23-35`): an integer is used directly (absolute value), an
/// all-digits string is parsed as an integer, and anything else is treated as a passphrase and
/// hashed to a 256-bit integer with SHA-256. Arbitrary precision is load-bearing — a passphrase key
/// is eight limbs, and numpy's `SeedSequence` folds limbs beyond the fourth in through a *different*
/// pass than the first four, so truncating to 64 bits would silently change the watermark.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GsKey {
    /// Little-endian `u32` limbs. Always non-empty; zero is `[0]`, matching numpy.
    limbs: Vec<u32>,
}

impl GsKey {
    /// The key for a non-negative integer.
    #[must_use]
    pub fn from_u64(value: u64) -> Self {
        Self::trimmed(vec![
            (value & 0xFFFF_FFFF) as u32,
            u32::try_from(value >> 32).unwrap_or_default(),
        ])
    }

    /// The key for a passphrase: `int.from_bytes(sha256(s), "big")` (`mage_latent.py:35`).
    #[must_use]
    pub fn from_passphrase(passphrase: &str) -> Self {
        let digest = Sha256::digest(passphrase.as_bytes());
        // Big-endian bytes → a 256-bit integer → little-endian u32 limbs.
        Self::trimmed(
            digest
                .chunks_exact(4)
                .rev()
                .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }

    /// Normalise a caller-supplied key string exactly as `_key_to_int` does: an optionally-signed
    /// all-digits string is its absolute integer value (at arbitrary precision — Python ints are
    /// unbounded), and anything else is a passphrase.
    ///
    /// # Errors
    /// Returns [`Error::Msg`] for an empty or whitespace-only key, matching the reference's
    /// `ValueError` (`mage_latent.py:29`).
    pub fn parse(value: &str) -> Result<Self> {
        let s = value.trim();
        if s.is_empty() {
            return Err(Error::Msg("empty Gaussian-Shading key".into()));
        }
        let digits = s.strip_prefix('-').unwrap_or(s);
        if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
            return Ok(Self::from_decimal_digits(digits));
        }
        Ok(Self::from_passphrase(s))
    }

    /// The little-endian `u32` limbs that seed numpy's `SeedSequence`.
    #[must_use]
    pub fn limbs(&self) -> &[u32] {
        &self.limbs
    }

    /// Arbitrary-precision decimal → little-endian `u32` limbs (`limbs = limbs·10 + digit`).
    fn from_decimal_digits(digits: &str) -> Self {
        let mut limbs: Vec<u32> = vec![0];
        for d in digits.bytes().map(|b| u64::from(b - b'0')) {
            let mut carry = d;
            for limb in &mut limbs {
                let v = u64::from(*limb) * 10 + carry;
                *limb = (v & 0xFFFF_FFFF) as u32;
                carry = v >> 32;
            }
            if carry != 0 {
                limbs.push(u32::try_from(carry).unwrap_or_default());
            }
        }
        Self::trimmed(limbs)
    }

    /// Drop leading-zero limbs, keeping at least one (numpy's `_int_to_uint32_array(0) == [0]`).
    fn trimmed(mut limbs: Vec<u32>) -> Self {
        while limbs.len() > 1 && limbs.last() == Some(&0) {
            limbs.pop();
        }
        Self { limbs }
    }
}

impl Default for GsKey {
    fn default() -> Self {
        Self::from_u64(GS_DEFAULT_KEY)
    }
}

/// Resolve the Gaussian-Shading key from an explicit caller-supplied value, defaulting to
/// [`GS_DEFAULT_KEY`].
///
/// This is the **whole** of the ported `resolve_gs_key` (`mage_latent.py:38-52`): the reference's
/// `MAGEFLOW_GS_KEY` environment variable and `~/.mageflow/gs_key` keyfile fallbacks are
/// deliberately absent — see the module docs. sc-14041 supplies `explicit` from `LoadSpec`/the
/// request.
///
/// # Errors
/// Propagates [`GsKey::parse`] on a blank explicit key.
pub fn resolve_gs_key(explicit: Option<&str>) -> Result<GsKey> {
    match explicit {
        Some(value) => GsKey::parse(value),
        None => Ok(GsKey::default()),
    }
}

// =================================================================================================
// numpy `SeedSequence` + PCG64 (XSL-RR) + Lemire bounded integers
// =================================================================================================

/// A bit-exact reimplementation of the `np.random.default_rng(key)` draw sequence.
///
/// Covers numpy's `SeedSequence` entropy expansion (`numpy/random/bit_generator.pyx`), the PCG64
/// XSL-RR bit generator (`numpy/random/src/pcg64/pcg64.h`) and the unbiased Lemire bounded-integer
/// path `Generator.integers` takes (`numpy/random/src/distributions/random_bounded_integers.c`).
/// `RandomState`'s masked-rejection path is a *different* stream and is deliberately not
/// implemented.
mod numpy_rng {
    const INIT_A: u32 = 0x43B0_D7E5;
    const MULT_A: u32 = 0x931E_8875;
    const INIT_B: u32 = 0x8B51_F9DD;
    const MULT_B: u32 = 0x58F3_8DED;
    const MIX_MULT_L: u32 = 0xCA01_F9DD;
    const MIX_MULT_R: u32 = 0x4973_F715;
    const XSHIFT: u32 = 16;
    const POOL_SIZE: usize = 4;

    /// `pcg_setseq_128`'s default 128-bit multiplier.
    const PCG_MULTIPLIER: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

    /// `SeedSequence._hashmix`: a mixer whose constant advances on every call, so the *order* of
    /// mixes is part of the definition.
    struct HashConst(u32);

    impl HashConst {
        fn mix(&mut self, value: u32) -> u32 {
            let mut value = value ^ self.0;
            self.0 = self.0.wrapping_mul(MULT_A);
            value = value.wrapping_mul(self.0);
            value ^= value >> XSHIFT;
            value
        }
    }

    /// `SeedSequence._mix`.
    fn mix(x: u32, y: u32) -> u32 {
        let mut r = MIX_MULT_L
            .wrapping_mul(x)
            .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
        r ^= r >> XSHIFT;
        r
    }

    /// `SeedSequence.mix_entropy` over the assembled entropy (no spawn key — we never spawn).
    ///
    /// The trailing loop is why [`super::GsKey`] keeps arbitrary precision: entropy limbs beyond the
    /// pool size are folded in through a second, different pass.
    fn seed_pool(entropy: &[u32]) -> [u32; POOL_SIZE] {
        let mut mixer = [0u32; POOL_SIZE];
        let mut hc = HashConst(INIT_A);
        for (i, slot) in mixer.iter_mut().enumerate() {
            *slot = hc.mix(entropy.get(i).copied().unwrap_or(0));
        }
        for i_src in 0..POOL_SIZE {
            for i_dst in 0..POOL_SIZE {
                if i_src != i_dst {
                    let m = hc.mix(mixer[i_src]);
                    mixer[i_dst] = mix(mixer[i_dst], m);
                }
            }
        }
        for &word in entropy.iter().skip(POOL_SIZE) {
            for slot in &mut mixer {
                let m = hc.mix(word);
                *slot = mix(*slot, m);
            }
        }
        mixer
    }

    /// `SeedSequence.generate_state(4, np.uint64)` — eight `u32` words viewed as four little-endian
    /// `u64`s.
    fn generate_state_u64(pool: &[u32; POOL_SIZE]) -> [u64; 4] {
        let mut words = [0u32; 8];
        let mut hash_const = INIT_B;
        for (i, slot) in words.iter_mut().enumerate() {
            let mut v = pool[i % POOL_SIZE] ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            v = v.wrapping_mul(hash_const);
            v ^= v >> XSHIFT;
            *slot = v;
        }
        let mut out = [0u64; 4];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = (u64::from(words[2 * i + 1]) << 32) | u64::from(words[2 * i]);
        }
        out
    }

    /// numpy's PCG64 (XSL-RR 128→64) plus the 32-bit half-word buffer `Generator.integers` uses.
    pub(super) struct Pcg64 {
        state: u128,
        inc: u128,
        buffered: Option<u32>,
    }

    impl Pcg64 {
        pub(super) fn new(entropy: &[u32]) -> Self {
            let st = generate_state_u64(&seed_pool(entropy));
            // `pcg64_set_seed` reads `seed[0]` as the HIGH word and `seed[1]` as the LOW word;
            // swapping them is silent and produces a plausible-looking but wrong stream.
            let initstate = (u128::from(st[0]) << 64) | u128::from(st[1]);
            let initseq = (u128::from(st[2]) << 64) | u128::from(st[3]);
            let mut rng = Self {
                state: 0,
                inc: (initseq << 1) | 1,
                buffered: None,
            };
            rng.step();
            rng.state = rng.state.wrapping_add(initstate);
            rng.step();
            rng
        }

        fn step(&mut self) {
            self.state = self
                .state
                .wrapping_mul(PCG_MULTIPLIER)
                .wrapping_add(self.inc);
        }

        /// `pcg_setseq_128_xsl_rr_64_random_r`: advance first, then output.
        fn next_u64(&mut self) -> u64 {
            self.step();
            let s = self.state;
            let xored = ((s >> 64) as u64) ^ (s as u64);
            xored.rotate_right((s >> 122) as u32)
        }

        /// `pcg64_next32`: the LOW half of a 64-bit draw first, the high half on the next call.
        fn next_u32(&mut self) -> u32 {
            if let Some(hi) = self.buffered.take() {
                return hi;
            }
            let next = self.next_u64();
            self.buffered = Some((next >> 32) as u32);
            next as u32
        }

        /// `bounded_lemire_uint32`. `range` is `high − low − 1` and must be below `u32::MAX`.
        fn bounded(&mut self, range: u32) -> u32 {
            let range_excl = u64::from(range) + 1;
            let mut m = u64::from(self.next_u32()) * range_excl;
            let mut leftover = m & 0xFFFF_FFFF;
            if leftover < range_excl {
                let threshold = u64::from(u32::MAX - range) % range_excl;
                while leftover < threshold {
                    m = u64::from(self.next_u32()) * range_excl;
                    leftover = m & 0xFFFF_FFFF;
                }
            }
            (m >> 32) as u32
        }

        /// `rng.integers(0, high, size=out.len())`.
        pub(super) fn fill_integers(&mut self, high: u32, out: &mut [u32]) {
            let range = high.saturating_sub(1);
            if range == 0 {
                // numpy short-circuits a degenerate range without consuming any randomness.
                out.fill(0);
                return;
            }
            for slot in out.iter_mut() {
                *slot = self.bounded(range);
            }
        }
    }
}

// =================================================================================================
// torch CPU `mt19937` + the float64 uniform draw
// =================================================================================================

/// A bit-exact reimplementation of `torch.rand(n, generator=torch.Generator().manual_seed(s),
/// dtype=torch.float64)` on CPU.
///
/// `at::mt19937` (`aten/src/ATen/core/MT19937RNGEngine.h`) is standard MT19937 seeded by
/// `init_genrand`; `CPUGeneratorImpl::random64()` concatenates two 32-bit outputs **high word
/// first**; and `uniform_real_distribution<double>` (`ATen/core/DistributionsHelper.h` →
/// `transformation::uniform_real`) keeps the low 53 bits scaled by `2⁻⁵³`. The CPU uniform kernel is
/// a `cpu_serial_kernel`, so element `i` always consumes draws `2i` and `2i+1` regardless of tensor
/// size — checked against torch 2.13.0 at the production 32 768-element size, not only on short
/// vectors.
mod torch_rng {
    const N: usize = 624;
    const M: usize = 397;
    const MATRIX_A: u32 = 0x9908_B0DF;
    const UPPER_MASK: u32 = 0x8000_0000;
    const LOWER_MASK: u32 = 0x7FFF_FFFF;

    pub(super) struct Mt19937 {
        state: [u32; N],
        next: usize,
        left: usize,
    }

    impl Mt19937 {
        pub(super) fn new(seed: u32) -> Self {
            let mut state = [0u32; N];
            state[0] = seed;
            for j in 1..N {
                let prev = state[j - 1];
                state[j] = 1_812_433_253u32
                    .wrapping_mul(prev ^ (prev >> 30))
                    .wrapping_add(j as u32);
            }
            // `left = 1` is the sentinel that forces a regeneration on the first draw.
            Self {
                state,
                next: 0,
                left: 1,
            }
        }

        fn twist(u: u32, v: u32) -> u32 {
            (((u & UPPER_MASK) | (v & LOWER_MASK)) >> 1) ^ if v & 1 == 1 { MATRIX_A } else { 0 }
        }

        fn next_state(&mut self) {
            self.left = N;
            self.next = 0;
            for j in 0..(N - M) {
                self.state[j] = self.state[j + M] ^ Self::twist(self.state[j], self.state[j + 1]);
            }
            for j in (N - M)..(N - 1) {
                self.state[j] =
                    self.state[j + M - N] ^ Self::twist(self.state[j], self.state[j + 1]);
            }
            self.state[N - 1] = self.state[M - 1] ^ Self::twist(self.state[N - 1], self.state[0]);
        }

        fn next_u32(&mut self) -> u32 {
            self.left -= 1;
            if self.left == 0 {
                self.next_state();
            }
            let mut y = self.state[self.next];
            self.next += 1;
            y ^= y >> 11;
            y ^= (y << 7) & 0x9D2C_5680;
            y ^= (y << 15) & 0xEFC6_0000;
            y ^= y >> 18;
            y
        }

        /// `CPUGeneratorImpl::random64()` — the first output becomes the HIGH half.
        fn next_u64(&mut self) -> u64 {
            let hi = self.next_u32();
            let lo = self.next_u32();
            (u64::from(hi) << 32) | u64::from(lo)
        }

        /// One `torch.rand(..., dtype=torch.float64)` element.
        pub(super) fn next_uniform_f64(&mut self) -> f64 {
            const MASK: u64 = (1u64 << 53) - 1;
            #[allow(clippy::cast_precision_loss)]
            const DIVISOR: f64 = 1.0 / (1u64 << 53) as f64;
            (self.next_u64() & MASK) as f64 * DIVISOR
        }
    }
}

// =================================================================================================
// Cephes special functions
// =================================================================================================

/// Double-precision `ndtri` (inverse normal CDF) and `erf`/`erfc`, ported from the Cephes Math
/// Library — the same source PyTorch vendors for `torch.special.ndtri`
/// (`aten/src/ATen/native/Math.h::calc_ndtri`), and the same one CPython's `math.erfc` agrees with
/// to ~1e-16. Ported rather than approximated because the watermark's *value* depends on `ndtri`
/// pointwise; `tests/gaussian_shading_parity.rs` pins both against reference grids captured from
/// torch and CPython.
///
/// > Derived from the Cephes Math Library (Release 2.8), Copyright 1984–2000 Stephen L. Moshier,
/// > redistributed under the 3-clause BSD terms carried by PyTorch's vendored copy.
mod cephes {
    /// `polevl(x, A, len(A) − 1)` — Horner over every coefficient.
    fn polevl(x: f64, coeffs: &[f64]) -> f64 {
        coeffs.iter().fold(0.0, |acc, &c| acc * x + c)
    }

    /// `p1evl(x, A, len(A))` — Horner with an implicit leading coefficient of 1.
    fn p1evl(x: f64, coeffs: &[f64]) -> f64 {
        let mut acc = x + coeffs[0];
        for &c in &coeffs[1..] {
            acc = acc * x + c;
        }
        acc
    }

    /// `sqrt(2π)`.
    const S2PI: f64 = 2.506_628_274_631_000_5;
    /// `exp(−2)` — the branch point between the central and tail approximations.
    const EXP_M2: f64 = 0.135_335_283_236_612_7;
    /// Cephes `MAXLOG`: `exp` of anything below `−MAXLOG` underflows to zero.
    const MAXLOG: f64 = 7.097_827_128_933_84e2;

    #[rustfmt::skip]
    const P0: [f64; 5] = [
        -5.996_335_010_141_079e1, 9.800_107_541_859_997e1, -5.667_628_574_690_703e1,
        1.393_126_093_872_797e1, -1.239_165_838_673_812_6,
    ];
    #[rustfmt::skip]
    const Q0: [f64; 9] = [
        1.0, 1.954_488_583_381_417_6, 4.676_279_128_988_815, 8.636_024_213_908_906e1,
        -2.254_626_878_541_193_7e2, 2.002_602_123_800_606_6e2, -8.203_722_561_683_333e1,
        1.590_562_251_262_117e1, -1.183_316_211_213_3,
    ];
    #[rustfmt::skip]
    const P1: [f64; 9] = [
        4.055_448_923_059_624, 3.152_510_945_998_938_6e1, 5.716_281_922_464_213e1,
        4.408_050_738_932_008e1, 1.468_495_619_288_580_2e1, 2.186_633_068_507_902_7,
        -1.402_560_791_713_545e-1, -3.504_246_268_278_482e-2, -8.574_567_851_546_854e-4,
    ];
    #[rustfmt::skip]
    const Q1: [f64; 9] = [
        1.0, 1.577_998_832_564_667_5e1, 4.539_076_351_288_792e1, 4.131_720_382_546_72e1,
        1.504_253_856_929_075e1, 2.504_649_462_083_094, -1.421_829_228_547_877_9e-1,
        -3.808_064_076_915_783e-2, -9.332_594_808_954_574e-4,
    ];
    #[rustfmt::skip]
    const P2: [f64; 9] = [
        3.237_748_917_769_46, 6.915_228_890_689_842, 3.938_810_252_924_744_4,
        1.333_034_608_158_075_4, 2.014_853_895_491_790_8e-1, 1.237_166_348_178_200_2e-2,
        3.015_815_535_082_354e-4, 2.658_069_746_867_375_5e-6, 6.239_745_391_849_833e-9,
    ];
    #[rustfmt::skip]
    const Q2: [f64; 9] = [
        1.0, 6.024_270_393_647_42, 3.679_835_638_561_608_6, 1.377_020_994_890_813_3,
        2.162_369_935_944_966_4e-1, 1.342_040_060_885_431_9e-2, 3.280_144_646_821_277_4e-4,
        2.892_478_647_453_806_8e-6, 6.790_194_080_099_813e-9,
    ];

    /// Inverse of the standard normal CDF — `torch.special.ndtri` / `scipy.special.ndtri`.
    ///
    /// Returns ∓∞ at 0/1 and NaN outside `[0, 1]`, matching `calc_ndtri`.
    pub(super) fn ndtri(y0: f64) -> f64 {
        if y0 == 0.0 {
            return f64::NEG_INFINITY;
        }
        if y0 == 1.0 {
            return f64::INFINITY;
        }
        if !(0.0..=1.0).contains(&y0) {
            return f64::NAN;
        }
        let mut negate = true;
        let mut y = y0;
        if y > 1.0 - EXP_M2 {
            y = 1.0 - y;
            negate = false;
        }
        if y > EXP_M2 {
            y -= 0.5;
            let y2 = y * y;
            let x = y + y * (y2 * polevl(y2, &P0) / polevl(y2, &Q0));
            return x * S2PI;
        }
        let x = (-2.0 * y.ln()).sqrt();
        let x0 = x - x.ln() / x;
        let z = 1.0 / x;
        let x1 = if x < 8.0 {
            z * polevl(z, &P1) / polevl(z, &Q1)
        } else {
            z * polevl(z, &P2) / polevl(z, &Q2)
        };
        let x = x0 - x1;
        if negate {
            -x
        } else {
            x
        }
    }

    #[rustfmt::skip]
    const ERFC_P: [f64; 9] = [
        2.461_969_814_735_305e-10, 5.641_895_648_310_688e-1, 7.463_210_564_422_699,
        4.863_719_709_856_814e1, 1.965_208_329_560_771e2, 5.264_451_949_954_773e2,
        9.345_285_271_719_576e2, 1.027_551_886_895_157_1e3, 5.575_353_353_693_993e2,
    ];
    #[rustfmt::skip]
    const ERFC_Q: [f64; 8] = [
        1.322_819_511_547_449_9e1, 8.670_721_408_859_897e1, 3.549_377_788_878_199e2,
        9.757_085_017_432_055e2, 1.823_909_166_879_097_4e3, 2.246_337_608_187_109_8e3,
        1.656_663_091_941_613_4e3, 5.575_353_408_177_277e2,
    ];
    #[rustfmt::skip]
    const ERFC_R: [f64; 6] = [
        5.641_895_835_477_551e-1, 1.275_366_707_599_781, 5.019_050_422_511_805,
        6.160_210_979_930_536, 7.409_742_699_504_489, 2.978_866_653_721_002_4,
    ];
    #[rustfmt::skip]
    const ERFC_S: [f64; 6] = [
        2.260_528_632_201_173, 9.396_035_249_380_014, 1.204_895_398_080_966_6e1,
        1.708_144_507_475_659e1, 9.608_968_090_632_859, 3.369_076_451_000_815,
    ];
    #[rustfmt::skip]
    const ERF_T: [f64; 5] = [
        9.604_973_739_870_516, 9.002_601_972_038_427e1, 2.232_005_345_946_843e3,
        7.003_325_141_128_051e3, 5.559_230_130_103_95e4,
    ];
    #[rustfmt::skip]
    const ERF_U: [f64; 5] = [
        3.356_171_416_475_031e1, 5.213_579_497_801_527e2, 4.594_323_829_709_801e3,
        2.262_900_006_138_909e4, 4.926_739_426_086_359e4,
    ];

    /// Complementary error function.
    pub(super) fn erfc(a: f64) -> f64 {
        let x = a.abs();
        if x < 1.0 {
            return 1.0 - erf(a);
        }
        let z = -a * a;
        if z < -MAXLOG {
            return if a < 0.0 { 2.0 } else { 0.0 };
        }
        let z = z.exp();
        let (p, q) = if x < 8.0 {
            (polevl(x, &ERFC_P), p1evl(x, &ERFC_Q))
        } else {
            (polevl(x, &ERFC_R), p1evl(x, &ERFC_S))
        };
        let y = (z * p) / q;
        if a < 0.0 {
            2.0 - y
        } else {
            y
        }
    }

    /// Error function.
    pub(super) fn erf(x: f64) -> f64 {
        if x.abs() > 1.0 {
            return 1.0 - erfc(x);
        }
        let z = x * x;
        x * polevl(z, &ERF_T) / p1evl(z, &ERF_U)
    }
}

// =================================================================================================
// The watermark
// =================================================================================================

/// Deterministically expand a payload string into an `n_bits` bit vector (`_payload_to_bits`,
/// `mage_latent.py:55-65`).
///
/// `sha256("{payload}:{counter}")` blocks are emitted **LSB-first within each byte** and
/// concatenated until `n_bits` bits exist. At the production `n_bits = 256` this is exactly the
/// bytes of `sha256("MageFlow:0")`.
#[must_use]
pub fn payload_to_bits(payload: &str, n_bits: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_bits + 8);
    let mut counter: u64 = 0;
    while out.len() < n_bits {
        let digest = Sha256::digest(format!("{payload}:{counter}").as_bytes());
        for byte in digest {
            for k in 0..8 {
                out.push((byte >> k) & 1);
            }
        }
        counter += 1;
    }
    out.truncate(n_bits);
    out
}

/// The production watermark message: [`GS_PAYLOAD`] expanded to
/// [`GS_MESSAGE_BITS`] bits.
#[must_use]
pub fn message_bits() -> Vec<u8> {
    payload_to_bits(GS_PAYLOAD, GS_MESSAGE_BITS)
}

/// The key-seeded per-entry XOR pad and message-index map (`_pad_and_pos`, `mage_latent.py:68-74`).
///
/// Both come from **one** `np.random.default_rng(key)` stream, pad first — so the index map depends
/// on `n`, and a port that draws them in the other order, or restarts the stream between them,
/// produces a completely different watermark while still looking statistically identical.
///
/// # Errors
/// Returns [`Error::Msg`] if `n_bits` falls outside the range numpy's Lemire path covers
/// (`2 ..= u32::MAX − 1`).
pub fn pad_and_pos(n: usize, key: &GsKey, n_bits: usize) -> Result<(Vec<u8>, Vec<u32>)> {
    if !(2..u32::MAX as usize).contains(&n_bits) {
        return Err(Error::Msg(format!(
            "Gaussian-Shading message length {n_bits} is outside the supported range 2..{}",
            u32::MAX
        )));
    }
    let mut rng = numpy_rng::Pcg64::new(key.limbs());
    let mut pad_raw = vec![0u32; n];
    rng.fill_integers(2, &mut pad_raw);
    let mut pos = vec![0u32; n];
    rng.fill_integers(n_bits as u32, &mut pos);
    let pad = pad_raw.into_iter().map(|v| v as u8).collect();
    Ok((pad, pos))
}

/// Build the Gaussian-Shading watermarked initial noise on the host, in `f64` (`encode_noise`,
/// `mage_latent.py:77-91`).
///
/// `shape` is `(channels, height, width)` in **latent** units — the reference passes `x.shape[1:]`
/// of the `randn` it is about to discard. The returned buffer is row-major `C·H·W`, ready to be
/// viewed as `[1, C, H, W]`.
///
/// # Errors
/// Returns [`Error::Msg`] on an empty or overflowing shape; propagates [`pad_and_pos`].
pub fn encode_noise_host(shape: (usize, usize, usize), key: &GsKey, seed: i64) -> Result<Vec<f64>> {
    let (c, h, w) = shape;
    let n = c
        .checked_mul(h)
        .and_then(|v| v.checked_mul(w))
        .ok_or_else(|| Error::Msg(format!("Gaussian-Shading latent {shape:?} overflows usize")))?;
    if n == 0 {
        return Err(Error::Msg(format!(
            "Gaussian-Shading latent {shape:?} is empty"
        )));
    }

    let msg = message_bits();
    let (pad, pos) = pad_and_pos(n, key, GS_MESSAGE_BITS)?;

    // `torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)` (`mage_latent.py:83`).
    let mut gen = torch_rng::Mt19937::new((seed as u64 & 0x7FFF_FFFF) as u32);

    let hi = 1.0 - GS_ARG_CLAMP;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let target_half = f64::from(msg[pos[i] as usize] ^ pad[i]);
        let u = gen.next_uniform_f64();
        let arg = ((target_half + u) / 2.0).clamp(GS_ARG_CLAMP, hi);
        out.push(cephes::ndtri(arg));
    }
    Ok(out)
}

/// The Gaussian-Shading watermarked initial latent as an MLX array of shape `[1, C, H, W]`.
///
/// The reference feeds the denoise loop this tensor at bfloat16 (`pipeline.py:307-308`), which is
/// what the end-to-end goldens start from; pass [`Dtype::Float32`] for parity work against the
/// `gs_noise` golden.
///
/// # Errors
/// Propagates [`encode_noise_host`] and any MLX conversion failure.
pub fn encode_noise(
    shape: (usize, usize, usize),
    key: &GsKey,
    seed: i64,
    dtype: Dtype,
) -> Result<Array> {
    let host = encode_noise_host(shape, key, seed)?;
    // f64 → f32 → target dtype is exactly torch's conversion chain for a double source: `c10::convert`
    // narrows through `float` before constructing bfloat16/half.
    let f32s: Vec<f32> = host.into_iter().map(|v| v as f32).collect();
    let (c, h, w) = shape;
    let arr = Array::from_slice(&f32s, &[1, c as i32, h as i32, w as i32]);
    if dtype == Dtype::Float32 {
        Ok(arr)
    } else {
        Ok(arr.as_dtype(dtype)?)
    }
}

// =================================================================================================
// Detection
// =================================================================================================

/// The outcome of a Gaussian-Shading read (`decode_bits`, `mage_latent.py:94-135`).
#[derive(Clone, Debug)]
pub struct WatermarkReport {
    /// Fraction of latent entries whose sign matches the expected half. `0.5` is chance.
    pub raw_acc: f64,
    /// Fraction of the message bits recovered by majority vote.
    pub msg_acc: f64,
    /// Entries whose sign matched.
    pub matches: usize,
    /// Entries examined.
    pub n: usize,
    /// Normal-approximation z-score of `matches` under `Binomial(n, 0.5)`.
    pub z_score: f64,
    /// One-sided p-value, `0.5·erfc(z/√2)`.
    pub p_value: f64,
    /// Whether `p_value` clears [`GS_PRESENT_PVALUE`].
    pub present: bool,
    /// Majority-vote estimate of the message.
    pub msg_hat: Vec<u8>,
    /// The expected message.
    pub msg: Vec<u8>,
}

/// Read the watermark out of a host-side latent buffer (row-major, any shape).
///
/// Only the **signs** are used, exactly as the reference does after upcasting to float32 — so this
/// is dtype-insensitive and tolerates the residual [`invert_to_noise`] leaves behind.
///
/// # Errors
/// Returns [`Error::Msg`] on an empty buffer; propagates [`pad_and_pos`].
pub fn decode_bits_host(latent: &[f32], key: &GsKey) -> Result<WatermarkReport> {
    let n = latent.len();
    if n == 0 {
        return Err(Error::Msg("Gaussian-Shading decode: empty latent".into()));
    }
    let msg = message_bits();
    let (pad, pos) = pad_and_pos(n, key, GS_MESSAGE_BITS)?;

    let mut matches = 0usize;
    let mut votes = vec![[0u32; 2]; GS_MESSAGE_BITS];
    for (i, &value) in latent.iter().enumerate() {
        let observed_half = u8::from(value > 0.0);
        let expected_half = msg[pos[i] as usize] ^ pad[i];
        if observed_half == expected_half {
            matches += 1;
        }
        let implied = observed_half ^ pad[i];
        votes[pos[i] as usize][usize::from(implied)] += 1;
    }
    // `votes.argmax(axis=1)`: numpy returns the FIRST maximum, so a tie resolves to 0.
    let msg_hat: Vec<u8> = votes.iter().map(|v| u8::from(v[1] > v[0])).collect();
    let recovered = msg_hat.iter().zip(&msg).filter(|(a, b)| a == b).count();

    let n_f = n as f64;
    let z_score = (matches as f64 - 0.5 * n_f) / (0.5 * n_f.sqrt());
    let p_value = 0.5 * cephes::erfc(z_score / std::f64::consts::SQRT_2);
    Ok(WatermarkReport {
        raw_acc: matches as f64 / n_f,
        msg_acc: recovered as f64 / GS_MESSAGE_BITS as f64,
        matches,
        n,
        z_score,
        p_value,
        present: p_value < GS_PRESENT_PVALUE,
        msg_hat,
        msg,
    })
}

/// Read the watermark out of an MLX latent of any shape.
///
/// # Errors
/// Propagates the host readback and [`decode_bits_host`].
pub fn decode_bits(latent: &Array, key: &GsKey) -> Result<WatermarkReport> {
    let flat = mlx_gen::array::contiguous(&latent.as_dtype(Dtype::Float32)?)?;
    let host = flat
        .try_as_slice::<f32>()
        .map_err(|e| Error::Msg(format!("Gaussian-Shading decode: unreadable latent: {e}")))?;
    decode_bits_host(host, key)
}

// =================================================================================================
// Flow-ODE inversion — the detection primitive
// =================================================================================================

/// Reverse the rectified-flow ODE from a clean latent back to the initial noise
/// (`invert_to_noise`, `pipeline.py:577-629`).
///
/// The forward sampler advances `x_{i+1} = x_i + (σ_{i+1} − σ_i)·v(x_i, σ_i)`; this walks the ladder
/// backwards from the clean end, evaluating the velocity at the point in hand as the proxy for
/// `x_i`. That proxy is the reference's own deliberate approximation — the sign-only watermark
/// absorbs the residual, which is why [`decode_bits`] still reads a clean signal off the recovered
/// noise.
///
/// * `z0` — clean latent `[1, C, gh, gw]`. Use the VAE posterior **mean**, which is deterministic;
///   the sampled posterior the edit path uses would inject noise into the detector
///   (`pipeline.py:583-585`).
/// * `sigmas` — the scheduler's sigma ladder: `steps + 1` entries descending to a terminal `0`.
/// * `velocity` — the model. Called as `velocity(tokens, σ)` with `tokens` shaped `[1, gh·gw, C]` at
///   bfloat16, the layout and dtype the reference hands the transformer, and must return the
///   velocity in the same layout. **sc-14041 supplies the empty-prompt, cfg-1 single forward the
///   reference uses** (`pipeline.py:612-616`); this function owns the integration, the token layout
///   and the dtype contract, and is deliberately model-agnostic so detection does not have to wait
///   on the DiT.
///
/// Returns the recovered initial-noise latent `[1, C, gh, gw]` in float32, ready for
/// [`decode_bits`].
///
/// # Errors
/// Returns [`Error::Msg`] on a malformed `z0`/`sigmas` or a velocity of the wrong shape, and
/// propagates whatever `velocity` returns.
pub fn invert_to_noise<F>(z0: &Array, sigmas: &[f32], mut velocity: F) -> Result<Array>
where
    F: FnMut(&Array, f32) -> Result<Array>,
{
    let shape = z0.shape();
    if shape.len() != 4 || shape[0] != 1 {
        return Err(Error::Msg(format!(
            "invert_to_noise: expected a [1, C, gh, gw] latent, got {shape:?}"
        )));
    }
    if sigmas.len() < 2 {
        return Err(Error::Msg(format!(
            "invert_to_noise: need at least one step, but sigmas has {} entries",
            sigmas.len()
        )));
    }
    let (c, gh, gw) = (shape[1], shape[2], shape[3]);
    let tokens = gh * gw;

    // `rearrange(z0, "b c h w -> b (h w) c").to(torch.bfloat16)` (`pipeline.py:602`).
    let mut img = z0
        .transpose_axes(&[0, 2, 3, 1])?
        .reshape(&[1, tokens, c])?
        .as_dtype(Dtype::Bfloat16)?;

    // Forward step `si` is `x_{si+1} = x_si + (σ_{si+1} − σ_si)·v`; undo it from the clean end.
    for si in (0..sigmas.len() - 1).rev() {
        let d_sigma = sigmas[si + 1] - sigmas[si];
        let vel = velocity(&img, sigmas[si])?;
        if vel.shape() != img.shape() {
            return Err(Error::Msg(format!(
                "invert_to_noise: velocity shape {:?} does not match the token stream {:?}",
                vel.shape(),
                img.shape()
            )));
        }
        let scale = Array::from_slice(&[d_sigma], &[1]).as_dtype(vel.dtype())?;
        img = img
            .subtract(vel.multiply(&scale)?)?
            .as_dtype(Dtype::Bfloat16)?;
    }

    // `unpack(img.float(), height, width)` (`pipeline.py:629` → `utils.py:36-43`).
    Ok(img
        .as_dtype(Dtype::Float32)?
        .reshape(&[1, gh, gw, c])?
        .transpose_axes(&[0, 3, 1, 2])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_normalisation_matches_the_reference_rules() {
        assert_eq!(
            GsKey::parse("20260720").unwrap(),
            GsKey::from_u64(20_260_720)
        );
        assert_eq!(GsKey::default(), GsKey::from_u64(GS_DEFAULT_KEY));
        // `_key_to_int` takes the absolute value of a signed integer key.
        assert_eq!(
            GsKey::parse("-20260720").unwrap(),
            GsKey::from_u64(20_260_720)
        );
        // Zero is one zero limb, exactly like numpy's `_int_to_uint32_array(0)`.
        assert_eq!(GsKey::from_u64(0).limbs(), &[0]);
        // Two limbs, little-endian.
        assert_eq!(GsKey::from_u64((1 << 40) + 7).limbs(), &[7, 256]);
        // A non-numeric key is a passphrase, hashed to eight limbs.
        let phrase = GsKey::parse("hunter2").unwrap();
        assert_eq!(phrase, GsKey::from_passphrase("hunter2"));
        assert_eq!(phrase.limbs().len(), 8);
        // Arbitrary precision: a 40-digit decimal (≈1.23e39, i.e. 131 bits) is not truncated.
        let big = GsKey::parse("1234567890123456789012345678901234567890").unwrap();
        assert_eq!(big.limbs().len(), 5);
        assert_ne!(big, GsKey::from_u64(u64::MAX));
        assert!(GsKey::parse("   ").is_err());
    }

    #[test]
    fn resolve_gs_key_defaults_to_the_pinned_key() {
        assert_eq!(
            resolve_gs_key(None).unwrap(),
            GsKey::from_u64(GS_DEFAULT_KEY)
        );
        assert_eq!(resolve_gs_key(Some("7")).unwrap(), GsKey::from_u64(7));
        assert!(resolve_gs_key(Some("")).is_err());
    }

    #[test]
    fn message_bits_are_the_sha256_of_the_counter_zero_block() {
        let bits = message_bits();
        assert_eq!(bits.len(), GS_MESSAGE_BITS);
        // LSB-first packing means the 256 bits repack to exactly the digest bytes.
        let digest = Sha256::digest(b"MageFlow:0");
        let mut packed = [0u8; 32];
        for (i, &b) in bits.iter().enumerate() {
            packed[i / 8] |= b << (i % 8);
        }
        assert_eq!(&packed[..], &digest[..]);
        // A longer request rolls the counter rather than repeating the first block.
        let long = payload_to_bits(GS_PAYLOAD, 300);
        assert_eq!(long.len(), 300);
        assert_eq!(&long[..256], &bits[..]);
        assert_ne!(&long[256..264], &bits[..8]);
    }

    #[test]
    fn pad_and_pos_draw_from_one_stream_in_order() {
        let key = GsKey::default();
        let (pad32, pos32) = pad_and_pos(32, &key, GS_MESSAGE_BITS).unwrap();
        assert_eq!(pad32.len(), 32);
        assert!(pos32.iter().all(|&p| (p as usize) < GS_MESSAGE_BITS));
        // The pad is the head of the stream, so it is length-independent...
        let (pad_big, pos_big) = pad_and_pos(4096, &key, GS_MESSAGE_BITS).unwrap();
        assert_eq!(&pad_big[..32], &pad32[..]);
        // ...while the index map starts after `n` pad draws, so it is NOT.
        assert_ne!(&pos_big[..32], &pos32[..]);
        // A different key gives a different pad.
        let (other, _) = pad_and_pos(32, &GsKey::from_u64(1), GS_MESSAGE_BITS).unwrap();
        assert_ne!(other, pad32);
        assert!(pad_and_pos(8, &key, 1).is_err());
    }

    #[test]
    fn encode_noise_is_standard_normal_and_carries_its_own_watermark() {
        let key = GsKey::default();
        let host = encode_noise_host((16, 8, 8), &key, 42).unwrap();
        assert_eq!(host.len(), 1024);
        let mean = host.iter().sum::<f64>() / host.len() as f64;
        let var = host.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / host.len() as f64;
        assert!(mean.abs() < 0.15, "mean {mean} is not ~0");
        assert!((var - 1.0).abs() < 0.15, "variance {var} is not ~1");

        let f32s: Vec<f32> = host.iter().map(|&v| v as f32).collect();
        let report = decode_bits_host(&f32s, &key).unwrap();
        assert_eq!(
            report.raw_acc, 1.0,
            "the pristine latent must decode perfectly"
        );
        assert!(report.present);
        // Message recovery is a per-bit majority vote, so it needs *coverage*: 1024 entries over
        // 256 bits leaves some bits with no votes at all, which resolve to 0 by numpy's
        // first-maximum tie rule. Full recovery is asserted where the redundancy is real — the
        // production tensor gives 128 votes per bit — not claimed at every size.
        assert!(report.msg_acc > 0.95, "msg_acc {}", report.msg_acc);
        let big = encode_noise_host((128, 8, 8), &key, 42).unwrap();
        let big_f32: Vec<f32> = big.iter().map(|&v| v as f32).collect();
        let big_report = decode_bits_host(&big_f32, &key).unwrap();
        assert_eq!(big_report.msg_acc, 1.0);
        assert_eq!(big_report.msg_hat, big_report.msg);

        // The wrong key reads chance, not the message.
        let wrong = decode_bits_host(&f32s, &GsKey::from_u64(20_260_721)).unwrap();
        assert!(!wrong.present, "a wrong key must not detect: {wrong:?}");
        assert!((wrong.raw_acc - 0.5).abs() < 0.1);
    }

    #[test]
    fn the_seed_moves_magnitudes_and_the_key_moves_signs() {
        let a = encode_noise_host((4, 4, 4), &GsKey::default(), 42).unwrap();
        let b = encode_noise_host((4, 4, 4), &GsKey::default(), 43).unwrap();
        let c = encode_noise_host((4, 4, 4), &GsKey::from_u64(20_260_721), 42).unwrap();
        assert_ne!(a, b);
        assert_ne!(a, c);
        // The seed only supplies the magnitude within a half, so `a` and `b` agree on every sign
        // while `a` and `c` — different half-assignment — do not.
        assert!(a.iter().zip(&b).all(|(x, y)| (*x > 0.0) == (*y > 0.0)));
        assert!(!a.iter().zip(&c).all(|(x, y)| (*x > 0.0) == (*y > 0.0)));
    }

    #[test]
    fn empty_and_degenerate_inputs_are_typed_errors_not_panics() {
        let key = GsKey::default();
        assert!(encode_noise_host((0, 4, 4), &key, 0).is_err());
        assert!(decode_bits_host(&[], &key).is_err());
    }

    // ------------------------------------------------------------------------------------------
    // Pinned reference vectors.
    //
    // Every literal below was captured from the pinned reference environment
    // (`_vendor/mage_flow/requirements.txt`: numpy 2.4.3, torch 2.13.0) or from CPython's
    // `math.erfc`, on this machine, and is reproduced here so CI exercises the parity claim
    // without the gitignored golden bundle. These are the three streams that had to be right;
    // each is pinned independently so a failure names the culprit instead of only reporting
    // "the tensor is wrong".
    // ------------------------------------------------------------------------------------------

    /// `torch.rand(8, generator=torch.Generator().manual_seed(s), dtype=torch.float64)`.
    #[test]
    fn torch_uniform_stream_is_bit_exact() {
        #[rustfmt::skip]
        const CASES: [(i64, [f64; 8]); 4] = [
            (42, [
                0.058_154_485_961_429_69, 0.062_910_167_424_577_56, 0.123_586_072_774_409_03,
                0.052_580_164_361_077_04, 0.526_171_896_854_761_8, 0.476_784_753_787_479_64,
                0.955_235_700_249_141_6, 0.928_752_581_457_429_6,
            ]),
            (0, [
                0.970_053_001_806_553_1, 0.707_819_864_399_788, 0.459_382_943_127_450_87,
                0.920_747_684_121_960_3, 0.645_024_120_122_764_8, 0.791_147_892_180_303_7,
                0.178_606_175_200_750_95, 0.351_107_624_393_928_4,
            ]),
            (7, [
                0.279_380_429_906_404, 0.273_693_713_651_765_65, 0.862_092_961_417_625_5,
                0.656_688_430_588_201_4, 0.922_529_367_452_987_2, 0.839_545_375_762_386_6,
                0.294_712_108_012_883_7, 0.560_721_597_372_155_5,
            ]),
            (2_147_483_647, [
                0.665_031_381_611_226_3, 0.843_458_957_224_422, 0.723_553_514_275_463,
                0.057_976_206_743_579_62, 0.064_231_526_648_305_67, 0.926_094_780_492_096_1,
                0.911_943_764_871_210_5, 0.075_837_683_337_990_57,
            ]),
        ];
        for (seed, want) in CASES {
            let mut gen = torch_rng::Mt19937::new((seed as u64 & 0x7FFF_FFFF) as u32);
            let got: Vec<f64> = (0..8).map(|_| gen.next_uniform_f64()).collect();
            assert_eq!(
                got,
                want.to_vec(),
                "torch uniform stream diverged at seed {seed}"
            );
        }
    }

    /// `np.random.default_rng(key).integers(...)` — the pad, then the index map, from one stream.
    #[test]
    fn numpy_pcg64_stream_is_bit_exact() {
        #[rustfmt::skip]
        const PAD32: [u8; 32] = [
            1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
            0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0,
        ];
        #[rustfmt::skip]
        const POS32: [u32; 32] = [
            209, 165, 244, 153, 226, 74, 133, 136, 69, 14, 17, 73, 200, 2, 107, 74,
            127, 195, 133, 126, 99, 199, 164, 26, 211, 32, 106, 249, 238, 212, 232, 53,
        ];
        let key = GsKey::default();
        let (pad, pos) = pad_and_pos(32, &key, GS_MESSAGE_BITS).unwrap();
        assert_eq!(pad, PAD32.to_vec());
        assert_eq!(pos, POS32.to_vec());

        // The production size. The index map starts after 32 768 pad draws, so this catches an
        // implementation that gets the short case right by accident.
        const IDX: [usize; 13] = [
            0, 1, 2, 3, 31, 32, 100, 1000, 4095, 16384, 32765, 32766, 32767,
        ];
        const PAD_BIG: [u8; 13] = [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1];
        const POS_BIG: [u32; 13] = [46, 235, 9, 158, 225, 177, 161, 121, 8, 236, 24, 216, 169];
        let (pad, pos) = pad_and_pos(32_768, &key, GS_MESSAGE_BITS).unwrap();
        for (k, i) in IDX.into_iter().enumerate() {
            assert_eq!(pad[i], PAD_BIG[k], "pad[{i}]");
            assert_eq!(pos[i], POS_BIG[k], "pos[{i}]");
        }
        let pad_sum: u32 = pad.iter().map(|&v| u32::from(v)).sum();
        let pos_sum: u64 = pos.iter().map(|&v| u64::from(v)).sum();
        assert_eq!(pad_sum, 16_340, "whole-stream pad checksum");
        assert_eq!(pos_sum, 4_180_124, "whole-stream index-map checksum");
    }

    /// numpy's `SeedSequence` folds entropy limbs past the pool size through a second pass, so a
    /// key wider than 32 bits — and especially the eight-limb passphrase key — must be carried at
    /// full precision. A `u64`-truncating implementation passes the single-limb case and fails here.
    #[test]
    fn numpy_pcg64_multi_limb_keys_are_bit_exact() {
        let cases: [(GsKey, [u8; 12], [u32; 12]); 3] = [
            (
                GsKey::from_u64((1 << 40) + 7),
                [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                [172, 3, 165, 9, 178, 34, 62, 230, 87, 86, 211, 8],
            ),
            (
                GsKey::from_passphrase(GS_PAYLOAD),
                [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
                [121, 240, 204, 75, 168, 144, 147, 167, 28, 177, 103, 250],
            ),
            (
                GsKey::from_u64(0),
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [128, 155, 248, 186, 161, 139, 143, 239, 71, 208, 171, 0],
            ),
        ];
        for (key, want_pad, want_pos) in cases {
            let (pad, pos) = pad_and_pos(12, &key, GS_MESSAGE_BITS).unwrap();
            assert_eq!(pad, want_pad.to_vec(), "pad for {key:?}");
            assert_eq!(pos, want_pos.to_vec(), "pos for {key:?}");
        }
    }

    /// `torch.special.ndtri` in float64. The clamp in [`encode_noise_host`] keeps the production
    /// argument inside `[1e-6, 1-1e-6]`, i.e. the `P0/Q0` and `P1/Q1` branches; `P2/Q2` is pinned
    /// too so the port is correct rather than merely sufficient.
    #[test]
    fn ndtri_matches_torch_float64() {
        #[rustfmt::skip]
        const GRID: [(f64, f64); 24] = [
            (1e-100, -21.273_453_560_965_322),
            (1e-20, -9.262_340_089_798_409),
            (1e-14, -7.650_628_092_935_269_5),
            (1.366_416_554_9e-14, -7.610_383_733_489_052),
            (1e-6, -4.753_424_308_822_899),
            (1e-5, -4.264_890_793_922_825),
            (1e-4, -3.719_016_485_455_680_4),
            (0.001, -3.090_232_306_167_813),
            (0.01, -2.326_347_874_040_840_8),
            (0.05, -1.644_853_626_951_472_9),
            (0.1, -1.281_551_565_544_600_4),
            (0.135_335_283_236_612_7, -1.101_519_628_498_75),
            (0.2, -0.841_621_233_572_914_2),
            (0.3, -0.524_400_512_708_040_9),
            (0.4, -0.253_347_103_135_799_7),
            (0.499_999_9, -2.506_628_274_703_107e-7),
            (0.5, 0.0),
            (0.500_000_1, 2.506_628_273_311_648e-7),
            (0.6, 0.253_347_103_135_799_7),
            (0.75, 0.674_489_750_196_081_7),
            (0.864_664_716_763_387_3, 1.101_519_628_498_749_8),
            (0.9, 1.281_551_565_544_600_4),
            (0.99, 2.326_347_874_040_840_8),
            (0.999_999_9, 5.199_337_582_290_661),
        ];
        let mut worst = 0.0f64;
        for (arg, want) in GRID {
            let got = cephes::ndtri(arg);
            let err = if want == 0.0 {
                got.abs()
            } else {
                ((got - want) / want).abs()
            };
            worst = worst.max(err);
            assert!(
                err < 1e-15,
                "ndtri({arg}) = {got}, want {want} (rel {err:e})"
            );
        }
        println!("ndtri worst relative deviation vs torch float64: {worst:e}");
        assert!(cephes::ndtri(0.0).is_infinite() && cephes::ndtri(0.0) < 0.0);
        assert!(cephes::ndtri(1.0).is_infinite() && cephes::ndtri(1.0) > 0.0);
        assert!(cephes::ndtri(-0.1).is_nan());
        assert!(cephes::ndtri(1.1).is_nan());
    }

    /// `math.erfc` (CPython/platform libm), which is what produces the reference p-value.
    #[test]
    fn erfc_matches_cpython() {
        #[rustfmt::skip]
        const GRID: [(f64, f64); 16] = [
            (0.0, 1.0),
            (0.25, 0.723_673_609_831_763_1),
            (0.5, 0.479_500_122_186_953_5),
            (std::f64::consts::FRAC_1_SQRT_2, 0.317_310_507_862_914),
            (0.9, 0.203_091_787_577_167_84),
            (1.0, 0.157_299_207_050_285_16),
            (1.5, 0.033_894_853_524_689_274),
            (2.0, 0.004_677_734_981_047_264_5),
            (3.362_090_641_623_8, 1.987_273_601_947_579e-6),
            (4.0, 1.541_725_790_028_002e-8),
            (5.0, 1.537_459_794_428_035e-12),
            (8.0, 1.122_429_717_298_292_6e-29),
            (10.0, 2.088_487_583_762_544_6e-45),
            (20.0, 5.395_865_611_607_901e-176),
            (-0.5, 1.520_499_877_813_046_5),
            (-1.0, 1.842_700_792_949_714_8),
        ];
        let mut worst = 0.0f64;
        for (x, want) in GRID {
            let got = cephes::erfc(x);
            let err = ((got - want) / want).abs();
            worst = worst.max(err);
            assert!(err < 1e-13, "erfc({x}) = {got}, want {want} (rel {err:e})");
        }
        println!("erfc worst relative deviation vs CPython: {worst:e}");
        // Cephes floors to zero once `x² > MAXLOG` (|x| ≳ 26.64), where the platform libm still
        // returns subnormals. Documented, not accidental: the p-value is a diagnostic and the
        // decision boundary sits at z ≈ 4.75, twenty-two standard deviations below this floor.
        assert_eq!(cephes::erfc(27.0), 0.0);
        assert!(cephes::erfc(26.0) > 0.0);
    }
}
