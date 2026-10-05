//! **TAEHV tiny video decoder** (madebyollin/taehv `taehv.py`) — the candle twin of
//! `mlx_gen::train::taehv`: the small, frozen, differentiable latent → pixel decoder the shared
//! perceptual-loss path runs a **video/Wan-VAE-family** trainer's x0 prediction through (epic 2123
//! E8, sc-24830).
//!
//! Faithful port of the reference's non-`_super` decoder (upstream rev
//! `011dfc2112197741c540e0bdd5b7b67bcc930771`). Three checkpoints are wired, each the upstream
//! name-guessed architecture:
//!
//! | variant | base VAE | latent ch | patch | `decoder_time_upscale` | t-upscale | `frames_to_trim` | px / latent |
//! |---|---|---|---|---|---|---|---|
//! | `taew2_1` | Wan 2.1, Wan 2.2 14B, Qwen-Image | 16 | 1 | (F, T, T) | 4 | 3 | 8 |
//! | `taew2_2` | Wan 2.2 TI2V 5B | 48 | 2 | (F, T, T) | 4 | 3 | 16 |
//! | `taeltx2_3` | LTX-2.3 / LTX-2.5 | 128 | 4 | (T, T, T) | 8 | 7 | 32 |
//!
//! **Latent space.** TAEHV applies no latent scale/shift: it decodes the diffusion model's own
//! per-channel-**normalized** latent (`(z − latents_mean) / latents_std`) — what the trainers cache.
//!
//! Graph, key layout (`decoder.{i}.…`, torch OIHW, cast to f32) and the per-frame `T = 1` decode
//! are documented on the MLX twin; this module runs the same math in candle's native NCHW layout,
//! built only from ops with a backward (`conv2d`, `upsample_nearest2d`, `relu`, `tanh`, `cat`,
//! `reshape`/`permute`/`narrow`, `clamp`), so the decode is differentiable in its input. Shipped
//! checkpoints also carry `encoder.*` keys, which are ignored; any unread `decoder.*` key is an
//! error (wrong variant, e.g. a `_super` checkpoint).

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Error, Result, Tensor};

use super::perceptual::{AuxModelFootprint, X0Decoder};

/// TAEHV decoder hyperparameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaehvConfig {
    /// Checkpoint name (`taew2_1`, `taew2_2`, `taeltx2_3`); the file is `<name>.safetensors`.
    pub name: &'static str,
    /// Latent channels the decoder consumes.
    pub latent_channels: usize,
    /// Output pixel-shuffle patch size.
    pub patch_size: usize,
    /// Per-stage temporal ×2 upsampling (TGrow stride 2 when true).
    pub decoder_time_upscale: [bool; 3],
    /// Per-stage spatial ×2 nearest upsampling.
    pub decoder_space_upscale: [bool; 3],
    /// Stage widths `n_f` (`[256, 128, 64, 64]` for every shipped non-super checkpoint).
    pub channels: [usize; 4],
}

impl TaehvConfig {
    /// `taew2_1` — Wan 2.1 / Wan 2.2 14B / Qwen-Image VAE (16 channels).
    pub fn taew2_1() -> Self {
        Self {
            name: "taew2_1",
            latent_channels: 16,
            patch_size: 1,
            decoder_time_upscale: [false, true, true],
            decoder_space_upscale: [true, true, true],
            channels: [256, 128, 64, 64],
        }
    }

    /// `taew2_2` — Wan 2.2 TI2V 5B VAE (48 channels, 16× spatial).
    pub fn taew2_2() -> Self {
        Self {
            name: "taew2_2",
            latent_channels: 48,
            patch_size: 2,
            ..Self::taew2_1()
        }
    }

    /// `taeltx2_3` — LTX-2.3 / LTX-2.5 VAE (128 channels, 32× spatial, 8× temporal).
    pub fn taeltx2_3() -> Self {
        Self {
            name: "taeltx2_3",
            latent_channels: 128,
            patch_size: 4,
            decoder_time_upscale: [true, true, true],
            ..Self::taew2_1()
        }
    }

    /// The Wan-VAE-family variant for a Wan/Qwen-Image VAE `z_dim` (16 → `taew2_1`,
    /// 48 → `taew2_2`); `None` for any other channel count.
    pub fn for_wan_z_dim(z_dim: usize) -> Option<Self> {
        match z_dim {
            16 => Some(Self::taew2_1()),
            48 => Some(Self::taew2_2()),
            _ => None,
        }
    }

    /// Human-readable decoder name for errors (`TAEW2.1`, `TAEW2.2`, `TAELTX2.3`).
    pub fn display_name(&self) -> String {
        match self.name {
            "taew2_1" => "TAEW2.1".into(),
            "taew2_2" => "TAEW2.2".into(),
            "taeltx2_3" => "TAELTX2.3".into(),
            other => other.to_uppercase(),
        }
    }

    /// The checkpoint file name (`<name>.safetensors`, the upstream `safetensors/` file).
    pub fn file_name(&self) -> String {
        format!("{}.safetensors", self.name)
    }

    fn time_stride(&self, stage: usize) -> usize {
        if self.decoder_time_upscale[stage] {
            2
        } else {
            1
        }
    }

    fn space_scale(&self, stage: usize) -> usize {
        if self.decoder_space_upscale[stage] {
            2
        } else {
            1
        }
    }

    /// Output frames per latent frame (`2^#time-upscaling stages`).
    pub fn t_upscale(&self) -> usize {
        (0..3).map(|s| self.time_stride(s)).product()
    }

    /// Leading output frames the reference drops (`t_upscale − 1`).
    pub fn frames_to_trim(&self) -> usize {
        self.t_upscale() - 1
    }

    /// Pixels per latent along each spatial axis (`2^#space-upscaling stages · patch`).
    pub fn spatial_upscale(&self) -> usize {
        (0..3).map(|s| self.space_scale(s)).product::<usize>() * self.patch_size
    }

    /// Exact parameter count of the decoder graph.
    pub fn param_count(&self) -> u64 {
        let conv3 = |i: usize, o: usize, bias: bool| (o * i * 9 + if bias { o } else { 0 }) as u64;
        let nf = self.channels;
        let mut n = conv3(self.latent_channels, nf[0], true);
        for s in 0..3 {
            let c = nf[s];
            n += 3 * (conv3(2 * c, c, true) + 2 * conv3(c, c, true));
            n += (c * c * self.time_stride(s)) as u64;
            n += conv3(c, nf[s + 1], false);
        }
        n + conv3(nf[3], 3 * self.patch_size * self.patch_size, true)
    }

    /// Pre-load memory figures for decoding ONE latent frame to an `out_h × out_w` image (resident
    /// f32 weights + one differentiable per-frame decode).
    pub fn footprint(&self, out_h: u32, out_w: u32) -> AuxModelFootprint {
        AuxModelFootprint {
            param_bytes: self.param_count() * 4,
            working_set_bytes: self.training_working_set_bytes(out_h, out_w),
            reference_bytes_per_image: 0,
        }
    }

    /// Conservative upper bound on the training working set of one per-frame
    /// ([`TaehvDecoder::decode_frames`]) differentiable decode to `out_h × out_w`, in bytes (f32,
    /// ×2 for the gradients) — the same estimate as the MLX twin. Not a measured value. A trainer
    /// decoding `k` frames per step scales it by `k`.
    pub fn training_working_set_bytes(&self, out_h: u32, out_w: u32) -> u64 {
        let up = self.spatial_upscale() as u64;
        let (lh, lw) = (
            (out_h as u64).div_ceil(up).max(1),
            (out_w as u64).div_ceil(up).max(1),
        );
        let nf = self.channels.map(|c| c as u64);
        let mut area = lh * lw;
        let mut elems = 3 * area * nf[0].max(self.latent_channels as u64);
        let mut frames = 1u64;
        for s in 0..3 {
            let c = nf[s];
            elems += 3 * 11 * frames * area * c;
            let sc = self.space_scale(s) as u64;
            let up_area = area * sc * sc;
            if s < 2 {
                let stride = self.time_stride(s) as u64;
                elems += frames * up_area * c;
                elems += 2 * frames * stride * up_area * c;
                frames *= stride;
                elems += frames * up_area * nf[s + 1];
            } else {
                let p2 = (self.patch_size * self.patch_size) as u64;
                elems += area * c;
                elems += 4 * up_area * nf[3];
                elems += 3 * up_area * 3 * p2;
            }
            area = up_area;
        }
        elems * 4 * 2
    }
}

/// One TAEHV MemBlock (identity skip: `n_in == n_out` everywhere in the decoder).
struct MemBlock {
    convs: [(Tensor, Tensor); 3],
}

fn conv3x3(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let y = x.conv2d(w, 1, 1, 1, 1)?;
    match b {
        Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0)?, 1, 1))?),
        None => Ok(y),
    }
}

impl MemBlock {
    /// `x`: `[n·T, C, h, w]` (clip-major); `past` = the same block's input one frame earlier in the
    /// clip, zeros for each clip's first frame.
    fn forward(&self, x: &Tensor, clips: usize) -> Result<Tensor> {
        let (nt, c, h, w) = x.dims4()?;
        let t = nt / clips;
        let past = if t == 1 {
            x.zeros_like()?
        } else {
            let xr = x.reshape((clips, t, c, h, w))?;
            let zero = Tensor::zeros((clips, 1, c, h, w), x.dtype(), x.device())?;
            Tensor::cat(&[&zero, &xr.narrow(1, 0, t - 1)?], 1)?.reshape((nt, c, h, w))?
        };
        let cat = Tensor::cat(&[x, &past], 1)?;
        let y = conv3x3(&cat, &self.convs[0].0, Some(&self.convs[0].1))?.relu()?;
        let y = conv3x3(&y, &self.convs[1].0, Some(&self.convs[1].1))?.relu()?;
        let y = conv3x3(&y, &self.convs[2].0, Some(&self.convs[2].1))?;
        (y + x)?.relu()
    }
}

struct Stage {
    blocks: [MemBlock; 3],
    /// TGrow 1×1 conv, OIHW `[C·stride, C, 1, 1]`.
    tgrow: Tensor,
    stride: usize,
    space: usize,
    /// Stage-closing 3×3 conv (no bias), OIHW.
    conv: Tensor,
}

/// TGrow: 1×1 conv to `C·stride` channels, then torch `reshape(-1, C, H, W)` (NCHW contiguous).
fn tgrow(x: &Tensor, w: &Tensor, stride: usize) -> Result<Tensor> {
    let y = x.conv2d(w, 0, 1, 1, 1)?;
    if stride == 1 {
        return Ok(y);
    }
    let (nt, cs, h, wd) = y.dims4()?;
    y.reshape((nt * stride, cs / stride, h, wd))
}

/// `F.pixel_shuffle(x, p)` on NCHW.
fn pixel_shuffle(x: &Tensor, p: usize) -> Result<Tensor> {
    if p == 1 {
        return Ok(x.clone());
    }
    let (b, cp, h, w) = x.dims4()?;
    let c = cp / (p * p);
    x.reshape((b, c, p, p, h, w))?
        .permute((0, 1, 4, 2, 5, 3))?
        .reshape((b, c, h * p, w * p))
}

/// The loaded, frozen TAEHV decoder.
pub struct TaehvDecoder {
    cfg: TaehvConfig,
    conv_in: (Tensor, Tensor),
    stages: [Stage; 3],
    conv_out: (Tensor, Tensor),
    param_bytes: u64,
}

impl TaehvDecoder {
    /// Load from a checkpoint file, or from a directory holding `<name>.safetensors` directly, under
    /// `vae/` (the `Kijai/LTX2.3_comfy` mirror layout) or under `safetensors/` (the upstream
    /// `madebyollin/taehv` repo layout). The file is picked by name only — never "the only
    /// safetensors in the dir". A missing checkpoint is an error naming the decoder.
    pub fn from_path(path: impl AsRef<Path>, cfg: TaehvConfig, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        let missing = || {
            Error::Msg(format!(
                "{} decoder checkpoint ({}) not found at {}",
                cfg.display_name(),
                cfg.file_name(),
                path.display()
            ))
        };
        let file = if path.is_file() {
            path.to_path_buf()
        } else {
            // By NAME only: a snapshot root (e.g. a multi-component mirror repo) can hold other
            // safetensors, so "the only/first safetensors in the dir" would load the wrong file.
            ["", "vae", "safetensors"]
                .iter()
                .map(|sub| path.join(sub).join(cfg.file_name()))
                .find(|p| p.is_file())
                .ok_or_else(missing)?
        };
        let w = candle_core::safetensors::load(&file, device)
            .map_err(|e| Error::Msg(format!("{}: {e}", cfg.display_name())))?;
        Self::from_weights(&w, cfg)
    }

    /// Build from tensors in the reference `decoder.{i}.…` layout. Every tensor's shape is checked
    /// against `cfg`; an unread `decoder.*` key is an error.
    pub fn from_weights(w: &HashMap<String, Tensor>, cfg: TaehvConfig) -> Result<Self> {
        let name = cfg.display_name();
        let mut read = std::collections::HashSet::new();
        let mut bytes = 0u64;
        let mut get = |key: &str, shape: &[usize]| -> Result<Tensor> {
            let a = w
                .get(key)
                .ok_or_else(|| Error::Msg(format!("{name}: missing tensor `{key}`")))?;
            if a.dims() != shape {
                return Err(Error::Msg(format!(
                    "{name}: `{key}` has shape {:?}, expected {shape:?} (wrong TAEHV variant?)",
                    a.dims()
                )));
            }
            read.insert(key.to_string());
            let a = a.to_dtype(DType::F32)?.contiguous()?;
            bytes += (a.elem_count() * 4) as u64;
            Ok(a)
        };
        let nf = cfg.channels;
        let conv_in = (
            get("decoder.1.weight", &[nf[0], cfg.latent_channels, 3, 3])?,
            get("decoder.1.bias", &[nf[0]])?,
        );
        let mut stages = Vec::with_capacity(3);
        for s in 0..3 {
            let base = 3 + 6 * s;
            let c = nf[s];
            let mut blocks = Vec::with_capacity(3);
            for b in 0..3 {
                let p = format!("decoder.{}.conv", base + b);
                let mut leg = |j: usize, cin: usize| -> Result<(Tensor, Tensor)> {
                    Ok((
                        get(&format!("{p}.{j}.weight"), &[c, cin, 3, 3])?,
                        get(&format!("{p}.{j}.bias"), &[c])?,
                    ))
                };
                let convs = [leg(0, 2 * c)?, leg(2, c)?, leg(4, c)?];
                blocks.push(MemBlock { convs });
            }
            let stride = cfg.time_stride(s);
            let tgrow = get(
                &format!("decoder.{}.conv.weight", base + 4),
                &[c * stride, c, 1, 1],
            )?;
            let conv = get(
                &format!("decoder.{}.weight", base + 5),
                &[nf[s + 1], c, 3, 3],
            )?;
            let blocks: [MemBlock; 3] = blocks
                .try_into()
                .map_err(|_| Error::Msg("taehv: three MemBlocks per stage".into()))?;
            stages.push(Stage {
                blocks,
                tgrow,
                stride,
                space: cfg.space_scale(s),
                conv,
            });
        }
        let out_c = 3 * cfg.patch_size * cfg.patch_size;
        let conv_out = (
            get("decoder.22.weight", &[out_c, nf[3], 3, 3])?,
            get("decoder.22.bias", &[out_c])?,
        );
        let mut stray: Vec<&str> = w
            .keys()
            .map(String::as_str)
            .filter(|k| k.starts_with("decoder.") && !read.contains(*k))
            .collect();
        if !stray.is_empty() {
            stray.sort_unstable();
            return Err(Error::Msg(format!(
                "{name}: checkpoint carries decoder keys this architecture does not read \
                 (wrong TAEHV variant?): {stray:?}"
            )));
        }
        let stages: [Stage; 3] = stages
            .try_into()
            .map_err(|_| Error::Msg("taehv: three stages".into()))?;
        Ok(Self {
            cfg,
            conv_in,
            stages,
            conv_out,
            param_bytes: bytes,
        })
    }

    /// The loaded configuration.
    pub fn config(&self) -> &TaehvConfig {
        &self.cfg
    }

    /// Resident parameter bytes (f32).
    pub fn param_bytes(&self) -> u64 {
        self.param_bytes
    }

    /// Clamp `tanh(x/3)·3` → conv_in → ReLU on NCHW.
    fn head(&self, x: &Tensor) -> Result<Tensor> {
        let h = ((x / 3.0)?.tanh()? * 3.0)?;
        conv3x3(&h, &self.conv_in.0, Some(&self.conv_in.1))?.relu()
    }

    /// ReLU → output conv → pixel-shuffle → clamp [0, 1] on NCHW.
    fn tail(&self, x: &Tensor) -> Result<Tensor> {
        let h = conv3x3(&x.relu()?, &self.conv_out.0, Some(&self.conv_out.1))?;
        pixel_shuffle(&h, self.cfg.patch_size)?.clamp(0f32, 1f32)
    }

    fn check_channels(&self, c: usize, dims: &[usize]) -> Result<()> {
        if c != self.cfg.latent_channels {
            return Err(Error::Msg(format!(
                "{} expects {} latent channels, got shape {dims:?}",
                self.cfg.display_name(),
                self.cfg.latent_channels
            )));
        }
        Ok(())
    }

    /// The reference `decode_video` (parallel mode): latents `[N, T, C, h, w]` (NTCHW) → frames
    /// NTHWC `[N, T·t_upscale − frames_to_trim, H, W, 3]` in `[0, 1]`. Differentiable.
    pub fn decode_video(&self, latents: &Tensor) -> Result<Tensor> {
        let dims = latents.dims().to_vec();
        if dims.len() != 5 {
            return Err(Error::Msg(format!(
                "{} decode_video expects NTCHW latents, got shape {dims:?}",
                self.cfg.display_name()
            )));
        }
        self.check_channels(dims[2], &dims)?;
        let (n, t, c, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let mut x = self.head(&latents.reshape((n * t, c, h, w))?)?;
        for st in &self.stages {
            for b in &st.blocks {
                x = b.forward(&x, n)?;
            }
            if st.space > 1 {
                let (_, _, hh, ww) = x.dims4()?;
                x = x.upsample_nearest2d(hh * st.space, ww * st.space)?;
            }
            x = tgrow(&x, &st.tgrow, st.stride)?;
            x = conv3x3(&x, &st.conv, None)?;
        }
        let x = self.tail(&x)?;
        let (nt, oc, oh, ow) = x.dims4()?;
        let frames = nt / n;
        let trim = self.cfg.frames_to_trim();
        x.reshape((n, frames, oc, oh, ow))?
            .narrow(1, trim, frames - trim)?
            .permute((0, 1, 3, 4, 2))?
            .contiguous()
    }

    /// Decode `N` latent frames **each as an independent `T = 1` clip** (the reference's
    /// `decode_video(z[:, None])` after its trim — the last of the `t_upscale` grown frames):
    /// NCHW `[N, C, h, w]` → NHWC `[N, H, W, 3]` in `[0, 1]`. Differentiable in `latents`.
    /// Only the surviving frame's full-resolution tail is computed (the last stage-2 frame through
    /// the last `n_f[2]` TGrow output channels — no MemBlock follows the final TGrow).
    pub fn decode_frames(&self, latents: &Tensor) -> Result<Tensor> {
        let dims = latents.dims().to_vec();
        if dims.len() != 4 {
            return Err(Error::Msg(format!(
                "{} decode_frames expects NCHW latents, got shape {dims:?}",
                self.cfg.display_name()
            )));
        }
        self.check_channels(dims[1], &dims)?;
        let n = dims[0];
        let mut x = self.head(latents)?;
        let last = self.stages.len() - 1;
        for (s, st) in self.stages.iter().enumerate() {
            for b in &st.blocks {
                x = b.forward(&x, n)?;
            }
            if s == last {
                let (nf, c, h, w) = x.dims4()?;
                let f = nf / n;
                x = x
                    .reshape((n, f, c, h, w))?
                    .narrow(1, f - 1, 1)?
                    .reshape((n, c, h, w))?;
            }
            if st.space > 1 {
                let (_, _, hh, ww) = x.dims4()?;
                x = x.upsample_nearest2d(hh * st.space, ww * st.space)?;
            }
            if s == last {
                let c = self.cfg.channels[s];
                let w_last = st.tgrow.narrow(0, (st.stride - 1) * c, c)?;
                x = x.conv2d(&w_last, 0, 1, 1, 1)?;
            } else {
                x = tgrow(&x, &st.tgrow, st.stride)?;
            }
            x = conv3x3(&x, &st.conv, None)?;
        }
        self.tail(&x)?.permute((0, 2, 3, 1))?.contiguous()
    }
}

impl X0Decoder for TaehvDecoder {
    fn decode(&self, latents: &Tensor) -> crate::Result<Tensor> {
        Ok(self.decode_frames(latents)?)
    }
}

/// splitmix64(`seed`·2³² + j) → uniform [−1, 1) (top 53 bits), times `scale·√3` (unit-variance
/// uniform · `scale`) plus `offset`, computed in f64 — deterministic on every host and backend.
pub fn splitmix_uniform(
    shape: &[usize],
    seed: u64,
    scale: f64,
    offset: f64,
    device: &Device,
) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n as u64)
        .map(|j| {
            let mut z = j
                .wrapping_add(seed << 32)
                .wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            (u * scale * 3f64.sqrt() + offset) as f32
        })
        .collect();
    Tensor::from_vec(v, shape, device)
}

/// A complete deterministic random-init TAEHV decoder checkpoint for `cfg` in the reference key
/// layout (torch OIHW, decoder keys only) — for tests, which must never download real weights.
/// Weights `U·√(1/fan_in)`, biases `U·0.05` (`decoder.22.bias` centred on 0.5 so outputs sit inside
/// the clamp), per-key seed = Σ key bytes mod 997 offset by `seed`.
pub fn synthetic_taehv_weights(
    cfg: &TaehvConfig,
    seed: u64,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let mut shapes: Vec<(String, Vec<usize>)> = Vec::new();
    let nf = cfg.channels;
    shapes.push((
        "decoder.1.weight".into(),
        vec![nf[0], cfg.latent_channels, 3, 3],
    ));
    shapes.push(("decoder.1.bias".into(), vec![nf[0]]));
    for s in 0..3 {
        let base = 3 + 6 * s;
        let c = nf[s];
        for b in 0..3 {
            for (j, cin) in [(0, 2 * c), (2, c), (4, c)] {
                let p = format!("decoder.{}.conv.{j}", base + b);
                shapes.push((format!("{p}.weight"), vec![c, cin, 3, 3]));
                shapes.push((format!("{p}.bias"), vec![c]));
            }
        }
        shapes.push((
            format!("decoder.{}.conv.weight", base + 4),
            vec![c * cfg.time_stride(s), c, 1, 1],
        ));
        shapes.push((
            format!("decoder.{}.weight", base + 5),
            vec![nf[s + 1], c, 3, 3],
        ));
    }
    let out_c = 3 * cfg.patch_size * cfg.patch_size;
    shapes.push(("decoder.22.weight".into(), vec![out_c, nf[3], 3, 3]));
    shapes.push(("decoder.22.bias".into(), vec![out_c]));
    let mut w = HashMap::new();
    for (k, shape) in shapes {
        let key_seed = k.bytes().map(u64::from).sum::<u64>() % 997 + seed * 1000;
        let fan_in: usize = shape[1..].iter().product::<usize>().max(1);
        let t = match k.as_str() {
            "decoder.22.bias" => splitmix_uniform(&shape, key_seed, 0.05, 0.5, device)?,
            _ if k.ends_with("bias") => splitmix_uniform(&shape, key_seed, 0.05, 0.0, device)?,
            _ => splitmix_uniform(&shape, key_seed, (1.0 / fan_in as f64).sqrt(), 0.0, device)?,
        };
        w.insert(k, t);
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Var;

    fn tiny(cfg: TaehvConfig) -> TaehvConfig {
        TaehvConfig {
            channels: [8, 6, 4, 4],
            ..cfg
        }
    }

    fn load(cfg: &TaehvConfig, seed: u64) -> TaehvDecoder {
        TaehvDecoder::from_weights(
            &synthetic_taehv_weights(cfg, seed, &Device::Cpu).unwrap(),
            cfg.clone(),
        )
        .unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn variant_hyperparameters_match_the_reference() {
        let w21 = TaehvConfig::taew2_1();
        assert_eq!((w21.latent_channels, w21.patch_size), (16, 1));
        assert_eq!(
            (w21.t_upscale(), w21.frames_to_trim(), w21.spatial_upscale()),
            (4, 3, 8)
        );
        let w22 = TaehvConfig::taew2_2();
        assert_eq!((w22.latent_channels, w22.patch_size), (48, 2));
        assert_eq!(
            (w22.t_upscale(), w22.frames_to_trim(), w22.spatial_upscale()),
            (4, 3, 16)
        );
        let ltx = TaehvConfig::taeltx2_3();
        assert_eq!((ltx.latent_channels, ltx.patch_size), (128, 4));
        assert_eq!(
            (ltx.t_upscale(), ltx.frames_to_trim(), ltx.spatial_upscale()),
            (8, 7, 32)
        );
        assert_eq!(TaehvConfig::for_wan_z_dim(16), Some(w21));
        assert_eq!(TaehvConfig::for_wan_z_dim(48), Some(w22));
        assert_eq!(TaehvConfig::for_wan_z_dim(128), None);
    }

    #[test]
    fn full_size_key_layout_matches_the_shipped_checkpoints_and_is_fully_consumed() {
        // Shapes from the upstream safetensors headers (rev 011dfc21).
        for (cfg, tgrow0, out_c) in [
            (TaehvConfig::taew2_1(), 256, 3),
            (TaehvConfig::taew2_2(), 256, 12),
            (TaehvConfig::taeltx2_3(), 512, 48),
        ] {
            let w = synthetic_taehv_weights(&cfg, 1, &Device::Cpu).unwrap();
            assert_eq!(w.len(), 64, "{}", cfg.name);
            assert_eq!(w["decoder.7.conv.weight"].dims(), &[tgrow0, 256, 1, 1]);
            assert_eq!(w["decoder.13.conv.weight"].dims(), &[256, 128, 1, 1]);
            assert_eq!(w["decoder.19.conv.weight"].dims(), &[128, 64, 1, 1]);
            assert_eq!(w["decoder.22.weight"].dims(), &[out_c, 64, 3, 3]);
            assert!(
                !w.contains_key("decoder.8.bias"),
                "stage convs carry no bias"
            );
            let dec = TaehvDecoder::from_weights(&w, cfg.clone()).unwrap();
            assert_eq!(dec.param_bytes(), cfg.param_count() * 4);
        }
    }

    #[test]
    fn encoder_keys_are_ignored_but_stray_decoder_keys_and_wrong_shapes_refuse() {
        let cfg = tiny(TaehvConfig::taew2_1());
        let mut w = synthetic_taehv_weights(&cfg, 3, &Device::Cpu).unwrap();
        w.insert(
            "encoder.0.weight".into(),
            Tensor::zeros((8, 3, 3, 3), DType::F32, &Device::Cpu).unwrap(),
        );
        assert!(TaehvDecoder::from_weights(&w, cfg.clone()).is_ok());
        w.insert(
            "decoder.23.weight".into(),
            Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap(),
        );
        let err = TaehvDecoder::from_weights(&w, cfg.clone()).err().unwrap();
        assert!(err.to_string().contains("decoder.23.weight"), "{err}");
        let w22 = synthetic_taehv_weights(&tiny(TaehvConfig::taew2_2()), 3, &Device::Cpu).unwrap();
        let err = TaehvDecoder::from_weights(&w22, cfg).err().unwrap();
        assert!(err.to_string().contains("TAEW2.1"), "{err}");
    }

    #[test]
    fn missing_checkpoint_names_the_decoder() {
        let dir = tempfile::tempdir().unwrap();
        let err = TaehvDecoder::from_path(dir.path(), TaehvConfig::taeltx2_3(), &Device::Cpu)
            .err()
            .unwrap()
            .to_string();
        assert!(
            err.contains("TAELTX2.3") && err.contains("taeltx2_3.safetensors"),
            "{err}"
        );
    }

    #[test]
    fn checkpoint_is_picked_by_name_never_by_being_the_only_file() {
        let cfg = tiny(TaehvConfig::taew2_1());
        let w = synthetic_taehv_weights(&cfg, 2, &Device::Cpu).unwrap();
        let dir = tempfile::tempdir().unwrap();
        // A mirror snapshot root holding an unrelated component: not taken as the decoder.
        let other: HashMap<String, Tensor> = [(
            "x".to_string(),
            Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap(),
        )]
        .into();
        candle_core::safetensors::save(&other, dir.path().join("other_model.safetensors")).unwrap();
        let err = TaehvDecoder::from_path(dir.path(), cfg.clone(), &Device::Cpu)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("taew2_1.safetensors"), "{err}");
        for sub in ["", "vae", "safetensors"] {
            let d = dir.path().join(format!("layout{sub}"));
            std::fs::create_dir_all(d.join(sub)).unwrap();
            candle_core::safetensors::save(&w, d.join(sub).join("taew2_1.safetensors")).unwrap();
            let dec = TaehvDecoder::from_path(&d, cfg.clone(), &Device::Cpu).unwrap();
            assert_eq!(dec.param_bytes(), cfg.param_count() * 4, "layout {sub:?}");
        }
        let f = dir.path().join("layout/taew2_1.safetensors");
        assert!(TaehvDecoder::from_path(&f, cfg, &Device::Cpu).is_ok());
    }

    #[test]
    fn per_frame_shapes_per_variant() {
        for (cfg, up) in [
            (tiny(TaehvConfig::taew2_1()), 8),
            (tiny(TaehvConfig::taew2_2()), 16),
            (tiny(TaehvConfig::taeltx2_3()), 32),
        ] {
            let dec = load(&cfg, 4);
            let z = splitmix_uniform(&[3, cfg.latent_channels, 2, 3], 9, 1.0, 0.0, &Device::Cpu)
                .unwrap();
            let px = dec.decode_frames(&z).unwrap();
            assert_eq!(px.dims(), &[3, 2 * up, 3 * up, 3], "{}", cfg.name);
            let v = px.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(v.iter().all(|x| (0.0..=1.0).contains(x)));
        }
    }

    #[test]
    fn per_frame_decode_equals_the_reference_t1_clip_with_its_trim() {
        for cfg in [tiny(TaehvConfig::taew2_1()), tiny(TaehvConfig::taeltx2_3())] {
            let dec = load(&cfg, 5);
            let z = splitmix_uniform(&[2, cfg.latent_channels, 2, 2], 11, 1.0, 0.0, &Device::Cpu)
                .unwrap();
            let fast = dec.decode_frames(&z).unwrap();
            let (n, c, h, w) = z.dims4().unwrap();
            let video = dec
                .decode_video(&z.reshape((n, 1, c, h, w)).unwrap())
                .unwrap();
            assert_eq!(
                video.dim(1).unwrap(),
                1,
                "{}: T=1 ⇒ one frame after the trim",
                cfg.name
            );
            let video = video.squeeze(1).unwrap();
            assert!(max_abs_diff(&fast, &video) < 1e-5, "{}", cfg.name);
        }
    }

    #[test]
    fn video_decode_grows_and_trims_and_memory_crosses_frames() {
        let cfg = tiny(TaehvConfig::taew2_1());
        let dec = load(&cfg, 6);
        let z = splitmix_uniform(
            &[1, 3, cfg.latent_channels, 2, 2],
            12,
            1.0,
            0.0,
            &Device::Cpu,
        )
        .unwrap();
        let v = dec.decode_video(&z).unwrap();
        assert_eq!(v.dims(), &[1, 9, 16, 16, 3]);
        // MemBlocks carry the earlier frames' state: the clip's last frame ≠ the last latent alone.
        let alone = dec
            .decode_frames(&z.narrow(1, 2, 1).unwrap().squeeze(1).unwrap())
            .unwrap();
        let last = v.narrow(1, 8, 1).unwrap().squeeze(1).unwrap();
        assert!(max_abs_diff(&alone, &last) > 1e-2);
    }

    #[test]
    fn per_frame_decode_backpropagates_to_its_input() {
        let cfg = tiny(TaehvConfig::taew2_2());
        let dec = load(&cfg, 7);
        let z = Var::from_tensor(
            &splitmix_uniform(&[2, cfg.latent_channels, 2, 2], 13, 1.0, 0.0, &Device::Cpu).unwrap(),
        )
        .unwrap();
        let loss = X0Decoder::decode(&dec, z.as_tensor())
            .unwrap()
            .sum_all()
            .unwrap();
        let grads = loss.backward().unwrap();
        let g = grads
            .get(z.as_tensor())
            .expect("gradient for the latent Var");
        assert_eq!(g.dims(), z.dims());
        let mag = g
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(mag > 0.0 && mag.is_finite());
    }

    #[test]
    fn working_set_scales_with_area() {
        for cfg in [
            TaehvConfig::taew2_1(),
            TaehvConfig::taew2_2(),
            TaehvConfig::taeltx2_3(),
        ] {
            let a = cfg.training_working_set_bytes(512, 512);
            assert_eq!(
                cfg.training_working_set_bytes(1024, 1024),
                a * 4,
                "{}",
                cfg.name
            );
            assert!(a > 0);
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn full_size_decode_matches_upstream_taehv_py_golden_values() {
        // Same golden values as the MLX twin, from the UPSTREAM reference (taehv.py @ 011dfc21,
        // `TAEHV(checkpoint_path=None, arch_name=name).double()`, `decode_video(parallel=True)`)
        // with the `synthetic_taehv_weights(cfg, 0)` weights; latents `splitmix_uniform` seed 7,
        // shape [1, 2, C, 2, 3]. Rows: (variant, video mean, video samples (flat NTHWC index,
        // value), frame-1-alone mean, frame-1-alone samples).
        #[allow(clippy::type_complexity)]
        let golden: &[(&str, f32, &[(usize, f32)], f32, &[(usize, f32)])] = &[
            (
                "taew2_1",
                0.5453534,
                &[
                    (0, 0.5278884),
                    (383, 0.5084928),
                    (767, 0.6362164),
                    (1151, 0.4839259),
                    (1535, 0.2605670),
                    (1919, 0.2949821),
                    (2303, 0.4986200),
                    (2687, 0.7761875),
                    (3071, 0.6646734),
                    (3455, 0.3471640),
                    (3839, 0.6241932),
                    (4223, 0.4906073),
                    (4607, 0.4165053),
                    (4991, 0.5359723),
                    (5375, 0.6930664),
                    (5759, 0.4709628),
                ],
                0.5575177,
                &[
                    (0, 0.4963038),
                    (76, 0.5129942),
                    (153, 0.4917771),
                    (230, 0.5424019),
                    (306, 0.5916426),
                    (383, 0.4937346),
                    (460, 0.5383758),
                    (537, 0.6685046),
                    (613, 0.6094354),
                    (690, 0.4107044),
                    (767, 0.6319227),
                    (844, 0.6082115),
                    (920, 0.7683413),
                    (997, 0.6360722),
                    (1074, 0.3444287),
                    (1151, 0.4861233),
                ],
            ),
            (
                "taew2_2",
                0.4599152,
                &[
                    (0, 0.5360567),
                    (1535, 0.1798162),
                    (3071, 0.5886377),
                    (4607, 0.5701227),
                    (6143, 0.4623908),
                    (7679, 0.3602138),
                    (9215, 0.3585008),
                    (10751, 0.4661903),
                    (12287, 0.6765762),
                    (13823, 0.4666952),
                    (15359, 0.5489842),
                    (16895, 0.1881633),
                    (18431, 0.4183862),
                    (19967, 0.1838195),
                    (21503, 0.3284745),
                    (23039, 0.5554451),
                ],
                0.4340227,
                &[
                    (0, 0.5102840),
                    (307, 0.4135604),
                    (614, 0.4558109),
                    (921, 0.5233188),
                    (1228, 0.7189682),
                    (1535, 0.2424411),
                    (1842, 0.4115649),
                    (2149, 0.3088579),
                    (2457, 0.1860975),
                    (2764, 0.3281009),
                    (3071, 0.3731965),
                    (3378, 0.6654694),
                    (3685, 0.2662813),
                    (3992, 0.5445945),
                    (4299, 0.6135635),
                    (4607, 0.5529289),
                ],
            ),
            (
                "taeltx2_3",
                0.4776780,
                &[
                    (0, 0.5787073),
                    (11059, 0.7197583),
                    (22118, 0.3439939),
                    (33177, 0.4524901),
                    (44236, 0.4893995),
                    (55295, 0.2780566),
                    (66354, 0.5553743),
                    (77413, 0.8660297),
                    (88473, 0.6473224),
                    (99532, 0.3645075),
                    (110591, 0.4368899),
                    (121650, 0.1053346),
                    (132709, 0.8250001),
                    (143768, 0.4557823),
                    (154827, 0.7226046),
                    (165887, 0.3213555),
                ],
                0.4809216,
                &[
                    (0, 0.5433555),
                    (1228, 0.0000000),
                    (2457, 0.3800583),
                    (3686, 0.4027431),
                    (4914, 0.5836142),
                    (6143, 0.6946127),
                    (7372, 0.4173975),
                    (8601, 0.1996691),
                    (9829, 0.7288970),
                    (11058, 0.5623746),
                    (12287, 0.6670548),
                    (13516, 0.6050438),
                    (14744, 0.5309826),
                    (15973, 0.3135534),
                    (17202, 0.2649713),
                    (18431, 0.3214565),
                ],
            ),
        ];
        for &(name, mean, samples, mean1, samples1) in golden {
            let cfg = match name {
                "taew2_1" => TaehvConfig::taew2_1(),
                "taew2_2" => TaehvConfig::taew2_2(),
                _ => TaehvConfig::taeltx2_3(),
            };
            let dec = load(&cfg, 0);
            let c = cfg.latent_channels;
            let z = splitmix_uniform(&[1, 2, c, 2, 3], 7, 1.0, 0.0, &Device::Cpu).unwrap();
            let video = dec.decode_video(&z).unwrap();
            let frame1 = dec
                .decode_frames(&z.narrow(1, 1, 1).unwrap().squeeze(1).unwrap())
                .unwrap();
            for (out, m, smp, what) in [
                (&video, mean, samples, "video"),
                (&frame1, mean1, samples1, "frame 1 alone"),
            ] {
                let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let got_mean = v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
                assert!(
                    (got_mean as f32 - m).abs() < 1e-4,
                    "{name} {what} mean {got_mean} vs {m}"
                );
                for &(i, want) in smp {
                    assert!(
                        (v[i] - want).abs() < 2e-4,
                        "{name} {what} [{i}] {} vs {want}",
                        v[i]
                    );
                }
            }
        }
    }
}
