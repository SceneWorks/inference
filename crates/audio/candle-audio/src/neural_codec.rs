//! Building blocks shared by the candle ports of **neural audio codecs** — the pieces several
//! providers' codec decoders have in common, lifted here so each port reuses one audited copy.
//!
//! - [`fold_weight_norm`] / [`resolve_weight_norm`] — old-style torch `weight_norm` pairs
//!   (`X.weight_g` / `X.weight_v`) folded to a plain `X.weight` at load.
//! - [`rvq_dequantize`] — residual-VQ decode: each quantizer's codebook lookup, summed.
//! - [`Snake`] and [`DacDecoder`] — the descript-audio-codec (DAC) convolutional decoder:
//!   `WNConv1d(k7) → N × DecoderBlock(stride r) → Snake → WNConv1d(→ 1, k7)`, each block
//!   `Snake → WNConvTranspose1d(k 2r, s r, p ⌈r/2⌉, op r mod 2) → 3 × ResidualUnit(dilation 1/3/9)`.
//!
//! Users: MOSS-SFX's continuous DAC VAE (sc-12841), MOSS-TTSD's XY_Tokenizer RVQ (sc-13518), and
//! YuE's xcodec (`SoundStream` RVQ + DAC `decoder_2`, sc-19377).
//!
//! Upstream candle's `dac.rs` decoder is deliberately **not** used: it hardcodes
//! `output_padding = 0`, which is wrong for odd strides (5 and 3 in these checkpoints).

use std::collections::HashMap;

use candle_core::{Device, Module, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};

/// Fold one weight-norm pair into a plain weight: `w = v · (g / ‖v‖)`, the norm taken over every
/// dim except 0 (torch `weight_norm(dim=0)`, whose `_weight_norm` computes exactly this order).
/// Valid for `Conv1d` and `ConvTranspose1d` alike — `g` is `[dim0, 1, …]` in both.
pub fn fold_weight_norm(v: &Tensor, g: &Tensor) -> Result<Tensor> {
    let mut sq = v.sqr()?;
    for d in 1..v.rank() {
        sq = sq.sum_keepdim(d)?;
    }
    v.broadcast_mul(&g.broadcast_div(&sq.sqrt()?)?)
}

/// Fold every `X.weight_g` / `X.weight_v` pair in `raw` into `X.weight` ([`fold_weight_norm`]);
/// every other tensor passes through unchanged. A `weight_g` without its `weight_v` is an error.
pub fn resolve_weight_norm(raw: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::with_capacity(raw.len());
    for (name, tensor) in &raw {
        if let Some(base) = name.strip_suffix(".weight_g") {
            let v = raw.get(&format!("{base}.weight_v")).ok_or_else(|| {
                candle_core::Error::Msg(format!("{base}.weight_g without {base}.weight_v"))
            })?;
            out.insert(format!("{base}.weight"), fold_weight_norm(v, tensor)?);
        } else if !name.ends_with(".weight_v") {
            out.insert(name.clone(), tensor.clone());
        }
    }
    Ok(out)
}

/// Residual-VQ dequantization: `Σ_q codebooks[q][codes[q]]` as `[1, dim, T]`.
///
/// `codebooks[q]` is quantizer `q`'s `[size, dim]` table and `codes[q]` its code row (all rows the
/// same length `T`). Rows beyond `codebooks.len()` are ignored, like the reference decoders that
/// consume only as many quantizers as they ship. Accumulation starts from zero and adds the
/// quantizers in order — the reference's `quantized_out = 0; quantized_out += …` — so the sum
/// rounds the same way. Codes must be in range (callers validate or clamp first); an out-of-range
/// code is an `index_select` error, not a silent wrap.
pub fn rvq_dequantize(codebooks: &[Tensor], codes: &[Vec<u32>], device: &Device) -> Result<Tensor> {
    let first = codebooks
        .first()
        .ok_or_else(|| candle_core::Error::Msg("rvq_dequantize: no codebooks".into()))?;
    let (_, dim) = first.dims2()?;
    let t = codes.first().map_or(0, Vec::len);
    let mut emb = Tensor::zeros((1, dim, t), first.dtype(), device)?;
    for (codebook, row) in codebooks.iter().zip(codes) {
        if row.len() != t {
            candle_core::bail!(
                "rvq_dequantize: code rows differ in length ({} vs {t})",
                row.len()
            );
        }
        let ids = Tensor::from_vec(row.clone(), (t,), device)?;
        let looked = codebook.index_select(&ids, 0)?; // [T, dim]
        emb = (emb + looked.t()?.unsqueeze(0)?.contiguous()?)?;
    }
    Ok(emb)
}

/// Snake activation: `x + (α + 1e-9)⁻¹ · sin²(αx)`, `α` per channel `[1, C, 1]`.
#[derive(Clone, Debug)]
pub struct Snake {
    alpha: Tensor,
}

impl Snake {
    /// Wrap a `[1, C, 1]` per-channel `alpha`.
    pub fn new(alpha: Tensor) -> Self {
        Self { alpha }
    }
}

impl Module for Snake {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let ax = x.broadcast_mul(&self.alpha)?;
        let s = ax.sin()?;
        let s2 = (&s * &s)?;
        x + s2.broadcast_div(&(&self.alpha + 1e-9)?)
    }
}

/// A name→tensor view over a resolved (weight-norm-folded) checkpoint map.
struct Weights<'a> {
    map: &'a HashMap<String, Tensor>,
}

impl Weights<'_> {
    fn get(&self, name: &str) -> Result<Tensor> {
        self.map
            .get(name)
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg(format!("DAC decoder: missing tensor {name:?}")))
    }

    fn snake(&self, name: &str) -> Result<Snake> {
        Ok(Snake::new(self.get(&format!("{name}.alpha"))?))
    }

    fn conv(&self, name: &str, cfg: Conv1dConfig) -> Result<Conv1d> {
        Ok(Conv1d::new(
            self.get(&format!("{name}.weight"))?,
            Some(self.get(&format!("{name}.bias"))?),
            cfg,
        ))
    }

    fn conv_transpose(&self, name: &str, cfg: ConvTranspose1dConfig) -> Result<ConvTranspose1d> {
        Ok(ConvTranspose1d::new(
            self.get(&format!("{name}.weight"))?,
            Some(self.get(&format!("{name}.bias"))?),
            cfg,
        ))
    }

    fn residual_unit(&self, base: &str, dilation: usize) -> Result<ResidualUnit> {
        let pad = (7 - 1) * dilation / 2;
        Ok(ResidualUnit {
            snake1: self.snake(&format!("{base}.block.0"))?,
            conv1: self.conv(
                &format!("{base}.block.1"),
                Conv1dConfig {
                    padding: pad,
                    dilation,
                    ..Default::default()
                },
            )?,
            snake2: self.snake(&format!("{base}.block.2"))?,
            conv2: self.conv(&format!("{base}.block.3"), Conv1dConfig::default())?,
        })
    }
}

struct ResidualUnit {
    snake1: Snake,
    conv1: Conv1d,
    snake2: Snake,
    conv2: Conv1d,
}

impl ResidualUnit {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv2.forward(
            &self
                .snake2
                .forward(&self.conv1.forward(&self.snake1.forward(x)?)?)?,
        )?;
        // Same-padding convs preserve length here; the reference crops symmetrically if not.
        let pad = (x.dim(D::Minus1)? - y.dim(D::Minus1)?) / 2;
        if pad > 0 {
            y.broadcast_add(&x.narrow(D::Minus1, pad, y.dim(D::Minus1)?)?)
        } else {
            y + x
        }
    }
}

struct DecoderBlock {
    snake: Snake,
    up: ConvTranspose1d,
    res: [ResidualUnit; 3],
}

impl DecoderBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.up.forward(&self.snake.forward(x)?)?;
        for r in &self.res {
            x = r.forward(&x)?;
        }
        Ok(x)
    }
}

/// The DAC convolutional decoder (`dac.model.dac.Decoder`), loaded from a weight-norm-resolved
/// tensor map under `<prefix>.model.N`. Output is the raw final conv — no `tanh`; a caller whose
/// reference ends in one (MOSS-SFX) applies it.
pub struct DacDecoder {
    conv_in: Conv1d,
    blocks: Vec<DecoderBlock>,
    snake_out: Snake,
    conv_out: Conv1d,
}

impl DacDecoder {
    /// Load `<prefix>.model.{0..=rates.len()+2}` from `map` for the given upsampling `rates`.
    /// Each transposed conv's kernel is checked to be `2·rate` (the DAC block geometry).
    pub fn load(map: &HashMap<String, Tensor>, prefix: &str, rates: &[usize]) -> Result<Self> {
        let w = Weights { map };
        let conv_in = w.conv(
            &format!("{prefix}.model.0"),
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
        )?;
        let mut blocks = Vec::with_capacity(rates.len());
        for (i, &stride) in rates.iter().enumerate() {
            let base = format!("{prefix}.model.{}", i + 1);
            let up_cfg = ConvTranspose1dConfig {
                stride,
                padding: stride.div_ceil(2),
                output_padding: stride % 2,
                ..Default::default()
            };
            let up = w.conv_transpose(&format!("{base}.block.1"), up_cfg)?;
            let k = up.weight().dim(2)?;
            if k != 2 * stride {
                candle_core::bail!(
                    "DAC decoder: {base} transposed-conv kernel {k} != 2 × stride {stride}"
                );
            }
            blocks.push(DecoderBlock {
                snake: w.snake(&format!("{base}.block.0"))?,
                up,
                res: [
                    w.residual_unit(&format!("{base}.block.2"), 1)?,
                    w.residual_unit(&format!("{base}.block.3"), 3)?,
                    w.residual_unit(&format!("{base}.block.4"), 9)?,
                ],
            });
        }
        let n = rates.len();
        let snake_out = w.snake(&format!("{prefix}.model.{}", n + 1))?;
        let conv_out = w.conv(
            &format!("{prefix}.model.{}", n + 2),
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
        )?;
        Ok(Self {
            conv_in,
            blocks,
            snake_out,
            conv_out,
        })
    }

    /// The input channel count (the latent width the first conv expects).
    pub fn input_dim(&self) -> Result<usize> {
        self.conv_in.weight().dim(1)
    }

    /// Decode `[B, input_dim, T]` → `[B, 1, T · Π rates]`. `cancel` is polled before each upsampling
    /// block (they dominate the cost) and before the output conv; a tripped poll returns `None`.
    pub fn decode(&self, x: &Tensor, cancel: &dyn Fn() -> bool) -> Result<Option<Tensor>> {
        let mut x = self.conv_in.forward(x)?;
        for block in &self.blocks {
            if cancel() {
                return Ok(None);
            }
            x = block.forward(&x)?;
        }
        if cancel() {
            return Ok(None);
        }
        Ok(Some(self.conv_out.forward(&self.snake_out.forward(&x)?)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn weight_norm_fold_matches_the_closed_form() {
        let dev = Device::Cpu;
        // v rows [3, 4] and [0, 2] (norms 5 and 2), g = [10, 3] → w = [[6, 8], [0, 3]].
        let v = Tensor::new(&[[[3f32], [4.]], [[0.], [2.]]], &dev).unwrap();
        let g = Tensor::new(&[[[10f32]], [[3.]]], &dev).unwrap();
        let w = fold_weight_norm(&v, &g).unwrap();
        assert_eq!(
            w.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            [6., 8., 0., 3.]
        );
        let mut raw = HashMap::new();
        raw.insert("c.weight_v".to_string(), v);
        raw.insert("c.weight_g".to_string(), g);
        raw.insert(
            "c.bias".to_string(),
            Tensor::zeros(2, DType::F32, &dev).unwrap(),
        );
        let out = resolve_weight_norm(raw).unwrap();
        let mut keys: Vec<_> = out.keys().cloned().collect();
        keys.sort();
        assert_eq!(keys, ["c.bias", "c.weight"]);
        let mut orphan = HashMap::new();
        orphan.insert(
            "x.weight_g".to_string(),
            Tensor::ones((1, 1, 1), DType::F32, &dev).unwrap(),
        );
        assert!(resolve_weight_norm(orphan).is_err());
    }

    #[test]
    fn rvq_dequantize_sums_the_per_quantizer_lookups() {
        let dev = Device::Cpu;
        let cb0 = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &dev).unwrap();
        let cb1 = Tensor::new(&[[10f32, 20.], [30., 40.], [50., 60.]], &dev).unwrap();
        // Frame 0: cb0[2] + cb1[0] = [15, 26]; frame 1: cb0[0] + cb1[1] = [31, 42].
        let codes = vec![vec![2u32, 0], vec![0u32, 1], vec![1u32, 1]]; // 3rd row: no codebook
        let e = rvq_dequantize(&[cb0.clone(), cb1], &codes, &dev).unwrap();
        assert_eq!(e.dims(), [1, 2, 2]);
        assert_eq!(
            e.squeeze(0).unwrap().to_vec2::<f32>().unwrap(),
            [[15., 31.], [26., 42.]]
        );
        assert!(rvq_dequantize(std::slice::from_ref(&cb0), &[vec![3u32]], &dev).is_err());
        assert!(rvq_dequantize(&[cb0.clone(), cb0], &[vec![0u32, 1], vec![0u32]], &dev).is_err());
    }

    #[test]
    fn upsample_geometry_is_exact_per_stage() {
        // (L−1)·s − 2·⌈s/2⌉ + 2s + (s mod 2) = s·L — the output-padding rule that makes each DAC
        // block exactly ×stride for odd strides too.
        for s in [2usize, 3, 4, 5, 8] {
            for l in [1usize, 7, 50] {
                let out = (l - 1) * s + 2 * s + (s % 2) - 2 * s.div_ceil(2);
                assert_eq!(out, s * l, "stride {s} at length {l}");
            }
        }
    }

    #[test]
    fn snake_is_identity_at_zero_and_bounded_growth() {
        let dev = Device::Cpu;
        let snake = Snake::new(Tensor::ones((1, 2, 1), DType::F32, &dev).unwrap());
        let x = Tensor::zeros((1, 2, 4), DType::F32, &dev).unwrap();
        let y = snake.forward(&x).unwrap();
        assert_eq!(
            y.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0; 8]
        );
        // snake(π/2) with α=1: x + sin²(x) = π/2 + 1.
        let x = Tensor::full(std::f32::consts::FRAC_PI_2, (1, 2, 1), &dev).unwrap();
        for got in snake
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
        {
            assert!((got - (std::f32::consts::FRAC_PI_2 + 1.0)).abs() < 2e-6);
        }
    }
}
