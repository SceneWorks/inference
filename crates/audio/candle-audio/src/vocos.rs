//! **Vocos** — the ConvNeXt backbone + ISTFT-head neural vocoder (Siuzdak, `gemelo-ai/vocos`),
//! shared by every candle audio port that ends in one.
//!
//! ```text
//!   features [1, C_in, T] ─▶ embed Conv1d(k7, p3) ─▶ LayerNorm ─▶ N × ConvNeXtBlock ─▶ LayerNorm
//!     ─▶ head.out Linear(dim → n_fft + 2) ─▶ (log-mag, phase) ─▶ S = min(e^mag, 100)·e^{iφ}
//!     ─▶ irfft(n_fft) · window ─▶ overlap-add (hop) ÷ Σwindow² ─▶ trim (n_fft − hop)/2 each side
//!                                                                    ─▶ T · hop samples
//! ```
//!
//! The ISTFT is the reference's custom `"same"`-padding transform (`vocos/spectral_ops.py`
//! `ISTFT(padding="same")`): `irfft` as a fixed backward-normalized inverse-DFT basis matmul (so
//! `n_fft` need not be a power of two), Hann-windowed overlap-add normalized by the summed squared
//! window, the `(n_fft − hop) / 2` edge pad trimmed.
//!
//! Users: MOSS-TTSD's XY_Tokenizer `enhanced_vocos` (80 mel → 24 kHz, sc-13518) and YuE's two
//! `xcodec_mini_infer` upsamplers (1024-d codec embedding → 44.1 kHz, sc-19378). Only the
//! non-conditional (`LayerNorm`, no `AdaLayerNorm`) backbone with an `ISTFTHead` is ported — the
//! only shape either checkpoint family uses.

use candle_core::{Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, VarBuilder};

/// A Vocos instance's widths and transform (the `backbone` + `head` `init_args` of its config).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VocosConfig {
    /// Feature channels in (`backbone.input_channels`).
    pub input_channels: usize,
    /// Backbone width (`backbone.dim`).
    pub dim: usize,
    /// ConvNeXt MLP width (`backbone.intermediate_dim`).
    pub intermediate_dim: usize,
    /// ConvNeXt block count (`backbone.num_layers`).
    pub num_layers: usize,
    /// ISTFT size (`head.n_fft`; the window length equals it).
    pub n_fft: usize,
    /// ISTFT hop (`head.hop_length`) — output samples per feature frame.
    pub hop: usize,
    /// LayerNorm epsilon (the reference's `1e-6`).
    pub ln_eps: f64,
}

fn linear(vb: &VarBuilder, name: &str, out: usize, inp: usize) -> Result<Linear> {
    let w = vb.get((out, inp), &format!("{name}.weight"))?;
    let b = vb.get(out, &format!("{name}.bias"))?;
    Ok(Linear::new(w, Some(b)))
}

fn layer_norm(vb: &VarBuilder, name: &str, dim: usize, eps: f64) -> Result<LayerNorm> {
    let w = vb.get(dim, &format!("{name}.weight"))?;
    let b = vb.get(dim, &format!("{name}.bias"))?;
    Ok(LayerNorm::new(w, b, eps))
}

fn conv1d(
    vb: &VarBuilder,
    name: &str,
    (out, inp, k): (usize, usize, usize),
    cfg: Conv1dConfig,
) -> Result<Conv1d> {
    let w = vb.get((out, inp, k), &format!("{name}.weight"))?;
    let b = vb.get(out, &format!("{name}.bias"))?;
    Ok(Conv1d::new(w, Some(b), cfg))
}

/// One ConvNeXt block: depthwise conv(k7) → LayerNorm → Linear → GELU(erf) → Linear → ·γ, residual.
struct ConvNeXtBlock {
    dwconv: Conv1d,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn load(vb: &VarBuilder, cfg: &VocosConfig) -> Result<Self> {
        let dwconv = conv1d(
            vb,
            "dwconv",
            (cfg.dim, 1, 7),
            Conv1dConfig {
                padding: 3,
                groups: cfg.dim,
                ..Default::default()
            },
        )?;
        Ok(Self {
            dwconv,
            norm: layer_norm(vb, "norm", cfg.dim, cfg.ln_eps)?,
            pwconv1: linear(vb, "pwconv1", cfg.intermediate_dim, cfg.dim)?,
            pwconv2: linear(vb, "pwconv2", cfg.dim, cfg.intermediate_dim)?,
            gamma: vb.get(cfg.dim, "gamma")?,
        })
    }

    /// `[1, dim, T]` → `[1, dim, T]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let h = self.dwconv.forward(x)?;
        let h = h.transpose(1, 2)?.contiguous()?; // [1, T, dim]
        let h = self.norm.forward(&h)?;
        let h = self.pwconv1.forward(&h)?.gelu_erf()?;
        let h = self.pwconv2.forward(&h)?;
        let h = h.broadcast_mul(&self.gamma)?;
        let h = h.transpose(1, 2)?.contiguous()?; // [1, dim, T]
        residual + h
    }
}

/// A loaded Vocos vocoder (backbone + ISTFT head).
pub struct Vocos {
    embed: Conv1d,
    norm: LayerNorm,
    blocks: Vec<ConvNeXtBlock>,
    final_layer_norm: LayerNorm,
    head_out: Linear,
    /// Inverse-DFT basis for the ISTFT: `[n_fft, n_bins]` cos / sin, backward-normalized.
    idft_cos: Tensor,
    idft_sin: Tensor,
    window: Vec<f32>,
    cfg: VocosConfig,
}

impl Vocos {
    /// Load from a builder rooted at the module holding `backbone.*` and `head.out.*`, with the
    /// synthesis `window` (length [`VocosConfig::n_fft`] — the checkpoint's own
    /// `head.istft.window` buffer where it ships one, else [`hann_window`]).
    pub fn load(
        vb: &VarBuilder,
        cfg: VocosConfig,
        window: Vec<f32>,
        device: &Device,
    ) -> Result<Self> {
        if window.len() != cfg.n_fft || cfg.hop == 0 || cfg.hop > cfg.n_fft {
            candle_core::bail!(
                "vocos: window of {} samples, n_fft {}, hop {} are inconsistent",
                window.len(),
                cfg.n_fft,
                cfg.hop
            );
        }
        let bb = vb.pp("backbone");
        let embed = conv1d(
            &bb,
            "embed",
            (cfg.dim, cfg.input_channels, 7),
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
        )?;
        let blocks = (0..cfg.num_layers)
            .map(|i| ConvNeXtBlock::load(&bb.pp("convnext").pp(i), &cfg))
            .collect::<Result<Vec<_>>>()?;
        let head_out = linear(&vb.pp("head"), "out", cfg.n_fft + 2, cfg.dim)?;
        let (idft_cos, idft_sin) = idft_basis(cfg.n_fft, device)?;
        Ok(Self {
            embed,
            norm: layer_norm(&bb, "norm", cfg.dim, cfg.ln_eps)?,
            blocks,
            final_layer_norm: layer_norm(&bb, "final_layer_norm", cfg.dim, cfg.ln_eps)?,
            head_out,
            idft_cos,
            idft_sin,
            window,
            cfg,
        })
    }

    /// The loaded configuration.
    pub fn config(&self) -> &VocosConfig {
        &self.cfg
    }

    /// `[1, input_channels, T]` → mono waveform of `T · hop` samples.
    pub fn forward(&self, features: &Tensor) -> Result<Vec<f32>> {
        Ok(self
            .forward_cancellable(features, &|| false)?
            .unwrap_or_default())
    }

    /// [`Self::forward`], polling `cancel` between the backbone blocks and before the head;
    /// `None` when it trips.
    pub fn forward_cancellable(
        &self,
        features: &Tensor,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>> {
        let mut x = self.embed.forward(features)?; // [1, dim, T]
        x = self.norm.forward(&x.transpose(1, 2)?.contiguous()?)?;
        x = x.transpose(1, 2)?.contiguous()?;
        for block in &self.blocks {
            if cancel() {
                return Ok(None);
            }
            x = block.forward(&x)?;
        }
        if cancel() {
            return Ok(None);
        }
        let x = self
            .final_layer_norm
            .forward(&x.transpose(1, 2)?.contiguous()?)?; // [1, T, dim]
        let coeffs = self.head_out.forward(&x)?.i(0)?; // [T, n_fft + 2]
        let n_bins = self.cfg.n_fft / 2 + 1;
        let mag = coeffs.narrow(1, 0, n_bins)?;
        let phase = coeffs.narrow(1, n_bins, n_bins)?;
        let mag = mag.exp()?.clamp(f32::NEG_INFINITY, 1e2)?;
        let re = (mag.clone() * phase.cos()?)?;
        let im = (mag * phase.sin()?)?;
        // irfft via the fixed basis: frames[T, n_fft] = re @ cosᵀ − im @ sinᵀ.
        let frames = (re.matmul(&self.idft_cos.t()?)? - im.matmul(&self.idft_sin.t()?)?)?;
        Ok(Some(self.overlap_add(&frames.to_vec2::<f32>()?)))
    }

    /// Windowed overlap-add with summed-window-square envelope normalization and `"same"` edge
    /// trimming (`pad = (n_fft − hop) / 2`), exactly the reference custom ISTFT.
    pub fn overlap_add(&self, frames: &[Vec<f32>]) -> Vec<f32> {
        let (n, hop) = (self.cfg.n_fft, self.cfg.hop);
        let t = frames.len();
        if t == 0 {
            return Vec::new();
        }
        let out_len = (t - 1) * hop + n;
        let mut y = vec![0f32; out_len];
        let mut env = vec![0f32; out_len];
        for (f, frame) in frames.iter().enumerate() {
            let base = f * hop;
            for i in 0..n {
                let w = self.window[i];
                y[base + i] += frame[i] * w;
                env[base + i] += w * w;
            }
        }
        let pad = (n - hop) / 2;
        let mut out = Vec::with_capacity(out_len - 2 * pad);
        for i in pad..out_len - pad {
            let e = env[i];
            out.push(if e > 1e-11 { y[i] / e } else { 0.0 });
        }
        out
    }
}

/// A periodic Hann window of length `n` (`0.5 − 0.5·cos(2π i / n)`, the cosine in f64 — torch's
/// default `hann_window`).
pub fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos() as f32)
        .collect()
}

/// The backward-normalized inverse real-DFT basis `[N, n_bins]`: `cos[n,k] = a_k·cos(2πkn/N)/N`
/// and `sin[n,k] = a_k·sin(2πkn/N)/N`, with the one-sided fold factor `a_k = 1` at DC/Nyquist and
/// `2` in between — so `frames = re @ cosᵀ − im @ sinᵀ` reproduces `torch.fft.irfft(norm="backward")`.
pub fn idft_basis(n: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let n_bins = n / 2 + 1;
    let inv_n = 1.0 / n as f64;
    let mut cos = vec![0f32; n * n_bins];
    let mut sin = vec![0f32; n * n_bins];
    for idx in 0..n {
        for k in 0..n_bins {
            let a = if k == 0 || k == n / 2 { 1.0 } else { 2.0 };
            let theta = 2.0 * std::f64::consts::PI * (k as f64) * (idx as f64) * inv_n;
            cos[idx * n_bins + k] = (a * theta.cos() * inv_n) as f32;
            sin[idx * n_bins + k] = (a * theta.sin() * inv_n) as f32;
        }
    }
    Ok((
        Tensor::from_vec(cos, (n, n_bins), device)?,
        Tensor::from_vec(sin, (n, n_bins), device)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use std::collections::HashMap;

    #[test]
    fn hann_window_is_periodic() {
        let w = hann_window(8);
        assert_eq!(w.len(), 8);
        assert!(w[0].abs() < 1e-6, "periodic Hann starts at 0");
        for k in 1..8 {
            assert!((w[k] - w[8 - k]).abs() < 1e-6, "periodic symmetry at {k}");
        }
    }

    /// The basis reproduces `irfft(rfft(x))` for a pure cosine (a single real spike at bin 2).
    #[test]
    fn idft_basis_reconstructs_a_cosine() {
        let dev = Device::Cpu;
        let n = 16usize;
        let n_bins = n / 2 + 1;
        let mut re = vec![0f32; n_bins];
        re[2] = (n as f32) / 2.0;
        let (cos, sin) = idft_basis(n, &dev).unwrap();
        let re_t = Tensor::from_vec(re, (1, n_bins), &dev).unwrap();
        let im_t = Tensor::zeros((1, n_bins), DType::F32, &dev).unwrap();
        let frames = (re_t.matmul(&cos.t().unwrap()).unwrap()
            - im_t.matmul(&sin.t().unwrap()).unwrap())
        .unwrap();
        let got = frames.i(0).unwrap().to_vec1::<f32>().unwrap();
        for (t, g) in got.iter().enumerate() {
            let want = (2.0 * std::f64::consts::PI * 2.0 * t as f64 / n as f64).cos() as f32;
            assert!((g - want).abs() < 1e-4, "sample {t}: {g} vs {want}");
        }
    }

    /// A zero-weight Vocos of the given widths (every tensor the loader reads).
    fn zero_vocos(cfg: VocosConfig) -> Vocos {
        let dev = Device::Cpu;
        let z = |s: &[usize]| Tensor::zeros(s, DType::F32, &dev).unwrap();
        let mut m = HashMap::new();
        let ln = |m: &mut HashMap<String, Tensor>, p: &str| {
            m.insert(format!("{p}.weight"), z(&[cfg.dim]));
            m.insert(format!("{p}.bias"), z(&[cfg.dim]));
        };
        m.insert(
            "backbone.embed.weight".into(),
            z(&[cfg.dim, cfg.input_channels, 7]),
        );
        m.insert("backbone.embed.bias".into(), z(&[cfg.dim]));
        ln(&mut m, "backbone.norm");
        ln(&mut m, "backbone.final_layer_norm");
        for i in 0..cfg.num_layers {
            let p = format!("backbone.convnext.{i}");
            m.insert(format!("{p}.dwconv.weight"), z(&[cfg.dim, 1, 7]));
            m.insert(format!("{p}.dwconv.bias"), z(&[cfg.dim]));
            ln(&mut m, &format!("{p}.norm"));
            m.insert(
                format!("{p}.pwconv1.weight"),
                z(&[cfg.intermediate_dim, cfg.dim]),
            );
            m.insert(format!("{p}.pwconv1.bias"), z(&[cfg.intermediate_dim]));
            m.insert(
                format!("{p}.pwconv2.weight"),
                z(&[cfg.dim, cfg.intermediate_dim]),
            );
            m.insert(format!("{p}.pwconv2.bias"), z(&[cfg.dim]));
            m.insert(format!("{p}.gamma"), z(&[cfg.dim]));
        }
        m.insert("head.out.weight".into(), z(&[cfg.n_fft + 2, cfg.dim]));
        m.insert("head.out.bias".into(), z(&[cfg.n_fft + 2]));
        let vb = VarBuilder::from_tensors(m, DType::F32, &dev);
        Vocos::load(&vb, cfg, hann_window(cfg.n_fft), &dev).unwrap()
    }

    const TINY: VocosConfig = VocosConfig {
        input_channels: 3,
        dim: 4,
        intermediate_dim: 6,
        num_layers: 2,
        n_fft: 960,
        hop: 240,
        ln_eps: 1e-6,
    };

    #[test]
    fn overlap_add_and_forward_emit_hop_samples_per_frame() {
        let vocos = zero_vocos(TINY);
        let frames = vec![vec![0f32; TINY.n_fft]; 5];
        assert_eq!(vocos.overlap_add(&frames).len(), 5 * TINY.hop);
        let feats = Tensor::zeros((1, TINY.input_channels, 7), DType::F32, &Device::Cpu).unwrap();
        let wav = vocos.forward(&feats).unwrap();
        assert_eq!(wav.len(), 7 * TINY.hop);
        assert!(wav.iter().all(|v| v.is_finite()));
    }

    /// Zero head weights give `mag = e⁰ = 1`, `φ = 0` on every bin — an impulse at each frame's
    /// start, which the window zeroes; the result must stay finite and exactly `T · hop` long for a
    /// non-power-of-two `n_fft` (YuE's 3528 / 882 geometry).
    #[test]
    fn non_power_of_two_transforms_keep_the_same_padding_length() {
        let cfg = VocosConfig {
            n_fft: 3528,
            hop: 882,
            ..TINY
        };
        let vocos = zero_vocos(cfg);
        let feats = Tensor::zeros((1, cfg.input_channels, 3), DType::F32, &Device::Cpu).unwrap();
        let wav = vocos.forward(&feats).unwrap();
        assert_eq!(wav.len(), 3 * 882);
        assert!(wav.iter().all(|v| v.is_finite()));
    }

    /// `max|got − want| / max|want|`; a length mismatch is an infinite error.
    fn max_rel(got: &[f32], want: &[f32]) -> f64 {
        if got.len() != want.len() {
            return f64::INFINITY;
        }
        let peak = want.iter().fold(0f64, |m, &v| m.max(v.abs() as f64));
        let diff = got
            .iter()
            .zip(want)
            .fold(0f64, |m, (&a, &b)| m.max((a as f64 - b as f64).abs()));
        diff / peak
    }

    /// Upstream `VocosBackbone` + `ISTFTHead` at toy widths and MOSS-TTSD's transform (n_fft 960,
    /// hop 240), run through the shared [`Vocos`] with the computed [`hann_window`] — the exact
    /// load path `candle-audio-moss-tts` uses. The fixture
    /// (`scripts/reference/yue_vocos_reference.py tiny`) randomizes every layer scale and LayerNorm
    /// and drives some log-magnitudes past the `1e2` clamp, so the ConvNeXt block (GELU-erf, γ),
    /// the head's clamp and the inverse DFT's sign are all observable. Measured max relative
    /// difference: 1.3e-6 (CPU/f32), bound 1e-5.
    #[test]
    fn vocos_matches_the_upstream_reference_at_moss_geometry() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/vocos_tiny_moss_reference.safetensors");
        let dev = Device::Cpu;
        let t = candle_core::safetensors::load(&path, &dev).unwrap();
        let features = t["ref.features"].clone();
        let want = t["ref.wave"].to_vec1::<f32>().unwrap();
        let cfg = VocosConfig {
            input_channels: 5,
            dim: 8,
            intermediate_dim: 12,
            num_layers: 2,
            n_fft: 960,
            hop: 240,
            ln_eps: 1e-6,
        };
        let vb = VarBuilder::from_tensors(t, DType::F32, &dev);
        let vocos = Vocos::load(&vb, cfg, hann_window(cfg.n_fft), &dev).unwrap();
        let got = vocos.forward(&features).unwrap();
        assert_eq!(got.len(), 6 * cfg.hop);
        let rel = max_rel(&got, &want);
        println!("tiny MOSS-geometry vocos: max|Δ|/max|ref| = {rel:.3e}");
        assert!(rel <= 1e-5, "vocos diverges from the reference: {rel:.3e}");
    }

    #[test]
    fn cancel_trips_between_blocks_and_bad_geometry_is_refused() {
        let vocos = zero_vocos(TINY);
        let feats = Tensor::zeros((1, TINY.input_channels, 2), DType::F32, &Device::Cpu).unwrap();
        assert!(vocos
            .forward_cancellable(&feats, &|| true)
            .unwrap()
            .is_none());
        let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu);
        assert!(Vocos::load(&vb, TINY, hann_window(8), &Device::Cpu).is_err());
    }
}
