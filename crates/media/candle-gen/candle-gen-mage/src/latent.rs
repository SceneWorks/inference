//! Exact Gaussian-Shading prior used by every Mage-Flow generation.

use candle_core::{DType, Device, Result, Tensor};
use sha2::{Digest, Sha256};

const KEY: u64 = 20_260_720;
const BITS: usize = 256;

struct HashConst(u32);
impl HashConst {
    fn mix(&mut self, mut value: u32) -> u32 {
        value ^= self.0;
        self.0 = self.0.wrapping_mul(0x931e_8875);
        value = value.wrapping_mul(self.0);
        value ^ (value >> 16)
    }
}
fn mix(x: u32, y: u32) -> u32 {
    let mut r = 0xca01_f9ddu32
        .wrapping_mul(x)
        .wrapping_sub(0x4973_f715u32.wrapping_mul(y));
    r ^= r >> 16;
    r
}
fn seed_pool(entropy: &[u32]) -> [u32; 4] {
    let mut pool = [0; 4];
    let mut hc = HashConst(0x43b0_d7e5);
    for (i, p) in pool.iter_mut().enumerate() {
        *p = hc.mix(entropy.get(i).copied().unwrap_or(0));
    }
    for src in 0..4 {
        for dst in 0..4 {
            if src != dst {
                pool[dst] = mix(pool[dst], hc.mix(pool[src]));
            }
        }
    }
    for word in entropy.iter().skip(4) {
        for p in &mut pool {
            *p = mix(*p, hc.mix(*word));
        }
    }
    pool
}
fn seed_state(pool: &[u32; 4]) -> [u64; 4] {
    let mut words = [0u32; 8];
    let mut hc = 0x8b51_f9dd;
    for (i, w) in words.iter_mut().enumerate() {
        let mut v = pool[i % 4] ^ hc;
        hc = hc.wrapping_mul(0x58f3_8ded);
        v = v.wrapping_mul(hc);
        *w = v ^ (v >> 16);
    }
    std::array::from_fn(|i| (u64::from(words[2 * i + 1]) << 32) | u64::from(words[2 * i]))
}
struct Pcg {
    state: u128,
    inc: u128,
    buffered: Option<u32>,
}
impl Pcg {
    fn new(key: u64) -> Self {
        let entropy = [key as u32, (key >> 32) as u32];
        let st = seed_state(&seed_pool(&entropy));
        let mut r = Self {
            state: 0,
            inc: (((u128::from(st[2]) << 64) | u128::from(st[3])) << 1) | 1,
            buffered: None,
        };
        r.step();
        r.state = r
            .state
            .wrapping_add((u128::from(st[0]) << 64) | u128::from(st[1]));
        r.step();
        r
    }
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645)
            .wrapping_add(self.inc);
    }
    fn u64(&mut self) -> u64 {
        self.step();
        let x = (self.state >> 64) as u64 ^ self.state as u64;
        x.rotate_right((self.state >> 122) as u32)
    }
    fn u32(&mut self) -> u32 {
        if let Some(v) = self.buffered.take() {
            return v;
        }
        let v = self.u64();
        self.buffered = Some((v >> 32) as u32);
        v as u32
    }
    fn bounded(&mut self, high: u32) -> u32 {
        let range = u64::from(high);
        let mut m = u64::from(self.u32()) * range;
        let mut low = m as u32 as u64;
        let threshold = u64::from(u32::MAX - (high - 1)) % range;
        while low < threshold {
            m = u64::from(self.u32()) * range;
            low = m as u32 as u64;
        }
        (m >> 32) as u32
    }
}

struct Mt {
    state: [u32; 624],
    next: usize,
    left: usize,
}
impl Mt {
    fn new(seed: u32) -> Self {
        let mut state = [0; 624];
        state[0] = seed;
        for i in 1..624 {
            let p = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(p ^ (p >> 30))
                .wrapping_add(i as u32);
        }
        Self {
            state,
            next: 0,
            left: 1,
        }
    }
    fn twist(u: u32, v: u32) -> u32 {
        (((u & 0x8000_0000) | (v & 0x7fff_ffff)) >> 1) ^ if v & 1 == 1 { 0x9908_b0df } else { 0 }
    }
    fn refresh(&mut self) {
        self.left = 624;
        self.next = 0;
        for j in 0..227 {
            self.state[j] = self.state[j + 397] ^ Self::twist(self.state[j], self.state[j + 1]);
        }
        for j in 227..623 {
            self.state[j] = self.state[j - 227] ^ Self::twist(self.state[j], self.state[j + 1]);
        }
        self.state[623] = self.state[396] ^ Self::twist(self.state[623], self.state[0]);
    }
    fn u32(&mut self) -> u32 {
        self.left -= 1;
        if self.left == 0 {
            self.refresh();
        }
        let mut y = self.state[self.next];
        self.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
    fn uniform(&mut self) -> f64 {
        let v = (u64::from(self.u32()) << 32) | u64::from(self.u32());
        (v & ((1u64 << 53) - 1)) as f64 / (1u64 << 53) as f64
    }
}

fn polevl(x: f64, c: &[f64]) -> f64 {
    c.iter().fold(0., |a, v| a * x + v)
}
fn ndtri(y0: f64) -> f64 {
    const P0: [f64; 5] = [
        -59.96335010141079,
        98.00107541859997,
        -56.67628574690703,
        13.93126093872797,
        -1.2391658386738126,
    ];
    const Q0: [f64; 9] = [
        1.,
        1.9544885833814176,
        4.676279128988815,
        86.36024213908906,
        -225.46268785411937,
        200.26021238006066,
        -82.03722561683333,
        15.9056225126217,
        -1.1833162112133,
    ];
    const P1: [f64; 9] = [
        4.055448923059624,
        31.525109459989386,
        57.16281922464213,
        44.08050738932008,
        14.68495619295802,
        2.1866330685079027,
        -0.1402560791713545,
        -0.03504246268278482,
        -0.0008574567851546854,
    ];
    const Q1: [f64; 9] = [
        1.,
        15.779988325646675,
        45.39076391288792,
        41.3172038254672,
        15.04253856929075,
        2.504649462083094,
        -0.14218292285478779,
        -0.03808064076915783,
        -0.0009332594808954574,
    ];
    const P2: [f64; 9] = [
        3.23774891776946,
        6.915228890689842,
        3.9388102529247444,
        1.3330346081580754,
        0.201485389549179,
        0.012371663481782,
        0.0003015815535082354,
        0.000002658069746867,
        0.00000000623974539185,
    ];
    const Q2: [f64; 9] = [
        1.,
        6.02427039364742,
        3.6798356385616086,
        1.3770209948908133,
        0.2162369935944966,
        0.0134204046093809,
        0.000328014464682128,
        0.000002892478647454,
        0.0000000067901940801,
    ];
    let mut negate = true;
    let mut y = y0;
    if y > 0.8646647167633873 {
        y = 1. - y;
        negate = false;
    }
    if y > 0.1353352832366127 {
        y -= 0.5;
        let z = y * y;
        return (y + y * (z * polevl(z, &P0) / polevl(z, &Q0))) * 2.5066282746310005;
    }
    let x = (-2. * y.ln()).sqrt();
    let x0 = x - x.ln() / x;
    let z = 1. / x;
    let x1 = if x < 8. {
        z * polevl(z, &P1) / polevl(z, &Q1)
    } else {
        z * polevl(z, &P2) / polevl(z, &Q2)
    };
    if negate {
        -(x0 - x1)
    } else {
        x0 - x1
    }
}

fn message() -> Vec<u8> {
    let digest = Sha256::digest(b"MageFlow:0");
    digest
        .iter()
        .flat_map(|b| (0..8).map(move |k| (b >> k) & 1))
        .collect()
}

pub fn watermarked_noise(
    channels: usize,
    height: usize,
    width: usize,
    seed: u64,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let n = channels * height * width;
    let mut pcg = Pcg::new(KEY);
    let pad: Vec<u8> = (0..n).map(|_| pcg.bounded(2) as u8).collect();
    let pos: Vec<usize> = (0..n).map(|_| pcg.bounded(BITS as u32) as usize).collect();
    let msg = message();
    let mut mt = Mt::new((seed & 0x7fff_ffff) as u32);
    let values: Vec<f32> = (0..n)
        .map(|i| {
            let half = f64::from(msg[pos[i]] ^ pad[i]);
            ndtri(((half + mt.uniform()) / 2.).clamp(1e-6, 1. - 1e-6)) as f32
        })
        .collect();
    Tensor::from_vec(values, (1, channels, height, width), &Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(device)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rng_streams_match_pinned_reference_heads() {
        let mut mt = Mt::new(42);
        assert_eq!(mt.uniform(), 0.058_154_485_961_429_69);
        let mut pcg = Pcg::new(KEY);
        let pad: Vec<u8> = (0..32).map(|_| pcg.bounded(2) as u8).collect();
        assert_eq!(
            pad,
            [
                1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0,
                1, 0, 1, 0
            ]
        );
    }

    #[test]
    fn inverse_normal_matches_torch_float64() {
        for (arg, want) in [
            (1e-6, -4.753_424_308_822_899),
            (0.1, -1.281_551_565_544_600_4),
            (0.5, 0.0),
            (0.9, 1.281_551_565_544_600_4),
            (0.999_999_9, 5.199_337_582_290_661),
        ] {
            let got = ndtri(arg);
            assert_eq!(got as f32, want as f32, "{arg}: {got} != {want}");
        }
    }
}
