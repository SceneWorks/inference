//! CLAP audio tower (sc-12851): a faithful HTSAT (Swin-transformer) port on `candle-nn`, matching
//! `transformers` `ClapAudioEncoder` — patch embed → 4 windowed-attention stages with patch
//! merging → LayerNorm → mean pool. Batch-of-one.
//!
//! Ported module-for-module from `ClapAudioPatchEmbed` / `ClapAudioSelfAttention` (relative-position
//! bias + shifted-window mask) / `ClapAudioLayer` (window partition, cyclic shift) /
//! `ClapAudioStage` / `ClapAudioPatchMerging`. The final `avgpool(flatten(...))` over the rearranged
//! last hidden state is, by inspection, a plain **mean over all spatial tokens per channel**
//! (the rearrangement steps only permute the token set, then average all of it) — so we pool with
//! `last_hidden_state.mean(tokens)`, which is numerically identical and reshape-risk-free.

use crate::config;
use candle_audio::candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{
    conv2d, layer_norm, linear, linear_no_bias, Conv2d, Conv2dConfig, LayerNorm, Linear, Module,
    VarBuilder,
};

/// Pairwise relative-position index for a `window × window` grid, flattened `[WW*WW]` — the buffer
/// `ClapAudioSelfAttention` registers (and does not save), recomputed here.
fn relative_position_index(window: usize) -> Vec<u32> {
    let ww = window * window;
    // coords_flatten: (2, WW): [h_coords; w_coords] in row-major (i = h*window + w) order.
    let mut hs = vec![0i64; ww];
    let mut wsv = vec![0i64; ww];
    for h in 0..window {
        for w in 0..window {
            hs[h * window + w] = h as i64;
            wsv[h * window + w] = w as i64;
        }
    }
    let stride = (2 * window - 1) as i64;
    let mut idx = vec![0u32; ww * ww];
    for a in 0..ww {
        for b in 0..ww {
            let dh = hs[a] - hs[b] + (window as i64 - 1);
            let dw = wsv[a] - wsv[b] + (window as i64 - 1);
            idx[a * ww + b] = (dh * stride + dw) as u32;
        }
    }
    idx
}

/// The shifted-window attention mask `[nW, WW, WW]` (0 / -100), or `None` when `shift == 0`.
/// Mirrors `ClapAudioLayer.get_attn_mask` for a resolution that is an exact multiple of `window`.
fn shift_attn_mask(height: usize, width: usize, window: usize, shift: usize) -> Option<Vec<f32>> {
    if shift == 0 {
        return None;
    }
    // Region id per (h, w), from the 3×3 slice pattern.
    let region = |x: usize, n: usize| -> usize {
        if x < n - window {
            0
        } else if x < n - shift {
            1
        } else {
            2
        }
    };
    let mut img = vec![0usize; height * width];
    for h in 0..height {
        for w in 0..width {
            img[h * width + w] = region(h, height) * 3 + region(w, width);
        }
    }
    let nwh = height / window;
    let nww = width / window;
    let n_windows = nwh * nww;
    let ww = window * window;
    let mut mask = vec![0f32; n_windows * ww * ww];
    for wh in 0..nwh {
        for wv in 0..nww {
            let win = wh * nww + wv;
            // Collect the WW region ids in window-partition order (i = row, j = col).
            let mut ids = vec![0usize; ww];
            for i in 0..window {
                for j in 0..window {
                    ids[i * window + j] = img[(wh * window + i) * width + (wv * window + j)];
                }
            }
            for a in 0..ww {
                for b in 0..ww {
                    if ids[a] != ids[b] {
                        mask[win * ww * ww + a * ww + b] = -100.0;
                    }
                }
            }
        }
    }
    Some(mask)
}

/// Cyclic shift by `-s` (torch `roll(shifts=-s)`) along `dim`: `cat(x[s:], x[:s])`.
fn roll_neg(x: &Tensor, s: usize, dim: usize) -> Result<Tensor> {
    let n = x.dim(dim)?;
    if s == 0 {
        return Ok(x.clone());
    }
    Tensor::cat(&[x.narrow(dim, s, n - s)?, x.narrow(dim, 0, s)?], dim)
}

/// Cyclic shift by `+s` (torch `roll(shifts=+s)`) along `dim`: `cat(x[n-s:], x[:n-s])`.
fn roll_pos(x: &Tensor, s: usize, dim: usize) -> Result<Tensor> {
    let n = x.dim(dim)?;
    if s == 0 {
        return Ok(x.clone());
    }
    Tensor::cat(&[x.narrow(dim, n - s, s)?, x.narrow(dim, 0, n - s)?], dim)
}

/// (1, H, W, C) → (nW, window, window, C).
fn window_partition(x: &Tensor, window: usize) -> Result<Tensor> {
    let (b, h, w, c) = x.dims4()?;
    x.reshape((b, h / window, window, w / window, window, c))?
        .permute((0, 1, 3, 2, 4, 5))?
        .contiguous()?
        .reshape((b * (h / window) * (w / window), window, window, c))
}

/// (nW, window, window, C) → (1, H, W, C).
fn window_reverse(x: &Tensor, window: usize, h: usize, w: usize, c: usize) -> Result<Tensor> {
    x.reshape((1, h / window, w / window, window, window, c))?
        .permute((0, 1, 3, 2, 4, 5))?
        .contiguous()?
        .reshape((1, h, w, c))
}

struct SwinAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    proj: Linear,
    rel_bias_table: Tensor, // ((2W-1)^2, heads)
    rel_index: Tensor,      // (WW,) u32
    num_heads: usize,
    head_dim: usize,
    window: usize,
}

impl SwinAttention {
    fn load(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        window: usize,
        device: &Device,
    ) -> Result<Self> {
        let self_vb = vb.pp("self");
        let rel_bias_table = self_vb.get(
            ((2 * window - 1) * (2 * window - 1), num_heads),
            "relative_position_bias_table",
        )?;
        let rel_index = Tensor::from_vec(
            relative_position_index(window),
            window * window * window * window,
            device,
        )?;
        Ok(Self {
            query: linear(dim, dim, self_vb.pp("query"))?,
            key: linear(dim, dim, self_vb.pp("key"))?,
            value: linear(dim, dim, self_vb.pp("value"))?,
            proj: linear(dim, dim, vb.pp("output").pp("dense"))?,
            rel_bias_table,
            rel_index,
            num_heads,
            head_dim: dim / num_heads,
            window,
        })
    }

    fn heads(&self, x: &Tensor) -> Result<Tensor> {
        let (nw, seq, _) = x.dims3()?;
        x.reshape((nw, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// `hidden`: (nW, WW, dim). `mask`: optional (nW, WW, WW).
    fn forward(&self, hidden: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (nw, seq, dim) = hidden.dims3()?;
        let q = self.heads(&self.query.forward(hidden)?)?;
        let k = self.heads(&self.key.forward(hidden)?)?;
        let v = self.heads(&self.value.forward(hidden)?)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? * scale)?;

        // Relative position bias: table[index] → (WW*WW, heads) → (heads, WW, WW).
        let ww = self.window * self.window;
        let bias = self
            .rel_bias_table
            .index_select(&self.rel_index, 0)?
            .reshape((ww, ww, self.num_heads))?
            .permute((2, 0, 1))?
            .contiguous()?
            .unsqueeze(0)?; // (1, heads, WW, WW)
        scores = scores.broadcast_add(&bias)?;

        if let Some(mask) = mask {
            // scores: (nW, heads, WW, WW); mask: (nW, WW, WW) → (nW, 1, WW, WW).
            let mask = mask.unsqueeze(1)?;
            scores = scores.broadcast_add(&mask)?;
        }

        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // (nW, heads, WW, head_dim)
        let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((nw, seq, dim))?;
        self.proj.forward(&ctx)
    }
}

struct SwinBlock {
    ln_before: LayerNorm,
    attention: SwinAttention,
    ln_after: LayerNorm,
    intermediate: Linear,
    output: Linear,
    shift: usize,
    window: usize,
    resolution: usize,
}

impl SwinBlock {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        window: usize,
        shift: usize,
        resolution: usize,
        device: &Device,
    ) -> Result<Self> {
        let eps = config::AUDIO_LN_EPS;
        let hidden_mlp = (config::AUDIO_MLP_RATIO * dim as f64) as usize;
        // When the resolution is ≤ window, Swin collapses to a single window with no shift.
        let (window, shift) = if resolution <= window {
            (resolution, 0)
        } else {
            (window, shift)
        };
        Ok(Self {
            ln_before: layer_norm(dim, eps, vb.pp("layernorm_before"))?,
            attention: SwinAttention::load(vb.pp("attention"), dim, num_heads, window, device)?,
            ln_after: layer_norm(dim, eps, vb.pp("layernorm_after"))?,
            intermediate: linear(dim, hidden_mlp, vb.pp("intermediate").pp("dense"))?,
            output: linear(hidden_mlp, dim, vb.pp("output").pp("dense"))?,
            shift,
            window,
            resolution,
        })
    }

    fn forward(&self, hidden: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (_b, _l, c) = hidden.dims3()?;
        let res = self.resolution;
        let shortcut = hidden.clone();
        let x = self.ln_before.forward(hidden)?;
        let x = x.reshape((1, res, res, c))?;

        let x = if self.shift > 0 {
            roll_neg(&roll_neg(&x, self.shift, 1)?, self.shift, 2)?
        } else {
            x
        };
        let windows = window_partition(&x, self.window)?; // (nW, win, win, C)
        let (nw, _, _, _) = windows.dims4()?;
        let windows = windows.reshape((nw, self.window * self.window, c))?;

        let attn = self.attention.forward(&windows, mask)?;
        let attn = attn.reshape((nw, self.window, self.window, c))?;
        let x = window_reverse(&attn, self.window, res, res, c)?;

        let x = if self.shift > 0 {
            roll_pos(&roll_pos(&x, self.shift, 1)?, self.shift, 2)?
        } else {
            x
        };
        let x = x.reshape((1, res * res, c))?;
        let hidden = (shortcut + x)?;

        let y = self.ln_after.forward(&hidden)?;
        let y = self.intermediate.forward(&y)?.gelu_erf()?;
        let y = self.output.forward(&y)?;
        hidden + y
    }
}

struct PatchMerging {
    norm: LayerNorm,
    reduction: Linear,
    resolution: usize,
}

impl PatchMerging {
    fn load(vb: VarBuilder, dim: usize, resolution: usize) -> Result<Self> {
        Ok(Self {
            norm: layer_norm(4 * dim, config::AUDIO_LN_EPS, vb.pp("norm"))?,
            reduction: linear_no_bias(4 * dim, 2 * dim, vb.pp("reduction"))?,
            resolution,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let (_b, _l, c) = hidden.dims3()?;
        let r = self.resolution;
        let x = hidden.reshape((1, r, r, c))?;
        // Even/odd row/col strided gather.
        let idx_even: Vec<u32> = (0..r).step_by(2).map(|v| v as u32).collect();
        let idx_odd: Vec<u32> = (1..r).step_by(2).map(|v| v as u32).collect();
        let dev = hidden.device();
        let ie = Tensor::from_vec(idx_even.clone(), idx_even.len(), dev)?;
        let io = Tensor::from_vec(idx_odd.clone(), idx_odd.len(), dev)?;
        let rows_e = x.index_select(&ie, 1)?;
        let rows_o = x.index_select(&io, 1)?;
        let x0 = rows_e.index_select(&ie, 2)?; // (1, r/2, r/2, C) even rows, even cols
        let x1 = rows_o.index_select(&ie, 2)?; // odd rows, even cols
        let x2 = rows_e.index_select(&io, 2)?; // even rows, odd cols
        let x3 = rows_o.index_select(&io, 2)?; // odd rows, odd cols
        let cat = Tensor::cat(&[&x0, &x1, &x2, &x3], D::Minus1)?; // (1, r/2, r/2, 4C)
        let merged = cat.reshape((1, (r / 2) * (r / 2), 4 * c))?;
        let merged = self.norm.forward(&merged)?;
        self.reduction.forward(&merged)
    }
}

struct Stage {
    blocks: Vec<SwinBlock>,
    downsample: Option<PatchMerging>,
    masks: Vec<Option<Tensor>>,
}

impl Stage {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: VarBuilder,
        dim: usize,
        depth: usize,
        num_heads: usize,
        window: usize,
        resolution: usize,
        has_downsample: bool,
        device: &Device,
    ) -> Result<Self> {
        let blocks_vb = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(depth);
        let mut masks = Vec::with_capacity(depth);
        for i in 0..depth {
            let shift = if i % 2 == 0 { 0 } else { window / 2 };
            let block = SwinBlock::load(
                blocks_vb.pp(i),
                dim,
                num_heads,
                window,
                shift,
                resolution,
                device,
            )?;
            // Precompute the (possibly None) shift mask for this block's effective window/shift.
            let mask = match shift_attn_mask(resolution, resolution, block.window, block.shift) {
                Some(v) => {
                    let nw = (resolution / block.window) * (resolution / block.window);
                    let ww = block.window * block.window;
                    Some(Tensor::from_vec(v, (nw, ww, ww), device)?)
                }
                None => None,
            };
            blocks.push(block);
            masks.push(mask);
        }
        let downsample = if has_downsample {
            Some(PatchMerging::load(vb.pp("downsample"), dim, resolution)?)
        } else {
            None
        };
        Ok(Self {
            blocks,
            downsample,
            masks,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let mut x = hidden.clone();
        for (block, mask) in self.blocks.iter().zip(&self.masks) {
            x = block.forward(&x, mask.as_ref())?;
        }
        match &self.downsample {
            Some(ds) => ds.forward(&x),
            None => Ok(x),
        }
    }
}

/// The full HTSAT audio encoder. `forward(mel)` → the `(1, num_features)` pooled latent that
/// `audio_projection` consumes.
pub struct AudioTower {
    bn_weight: Tensor,
    bn_bias: Tensor,
    bn_mean: Tensor,
    bn_var: Tensor,
    patch_proj: Conv2d,
    patch_norm: LayerNorm,
    stages: Vec<Stage>,
    final_norm: LayerNorm,
}

impl AudioTower {
    pub fn load(vb: VarBuilder, device: &Device) -> Result<Self> {
        let n_mel = config::AUDIO_NUM_MEL_BINS;
        let bn = vb.pp("batch_norm");
        let bn_weight = bn.get(n_mel, "weight")?;
        let bn_bias = bn.get(n_mel, "bias")?;
        let bn_mean = bn.get(n_mel, "running_mean")?;
        let bn_var = bn.get(n_mel, "running_var")?;

        let patch_cfg = Conv2dConfig {
            padding: 0,
            stride: config::AUDIO_PATCH_SIZE,
            ..Default::default()
        };
        let patch_proj = conv2d(
            1,
            config::AUDIO_EMBED_DIM,
            config::AUDIO_PATCH_SIZE,
            patch_cfg,
            vb.pp("patch_embed").pp("proj"),
        )?;
        let patch_norm = layer_norm(
            config::AUDIO_EMBED_DIM,
            config::AUDIO_LN_EPS,
            vb.pp("patch_embed").pp("norm"),
        )?;

        let grid = config::AUDIO_SPEC_SIZE / config::AUDIO_PATCH_SIZE; // 64
        let layers_vb = vb.pp("layers");
        let mut stages = Vec::with_capacity(4);
        for i in 0..4 {
            let dim = config::AUDIO_EMBED_DIM * (1 << i);
            let resolution = grid >> i;
            let has_downsample = i < 3;
            stages.push(Stage::load(
                layers_vb.pp(i),
                dim,
                config::AUDIO_DEPTHS[i],
                config::AUDIO_HEADS[i],
                config::AUDIO_WINDOW_SIZE,
                resolution,
                has_downsample,
                device,
            )?);
        }
        let num_features = config::AUDIO_EMBED_DIM * (1 << 3); // 768
        let final_norm = layer_norm(num_features, config::AUDIO_LN_EPS, vb.pp("norm"))?;

        Ok(Self {
            bn_weight,
            bn_bias,
            bn_mean,
            bn_var,
            patch_proj,
            patch_norm,
            stages,
            final_norm,
        })
    }

    /// `mel`: flat `[TARGET_FRAMES * n_mels]` host vector (time-major). Returns `(1, num_features)`.
    pub fn forward(&self, mel: &[f32], device: &Device) -> Result<Tensor> {
        let t = config::TARGET_FRAMES;
        let f = config::AUDIO_NUM_MEL_BINS;
        // (1, 1, T, F)
        let x = Tensor::from_vec(mel.to_vec(), (1, 1, t, f), device)?.to_dtype(DType::F32)?;

        // BatchNorm2d over the mel-bin channel (== last dim F here): (x-mean)/sqrt(var+eps)*w + b.
        let eps = config::AUDIO_BN_EPS;
        let denom = (self.bn_var.clone() + eps)?.sqrt()?;
        let scale = (self.bn_weight.clone() / &denom)?; // (F,)
        let shift = (self.bn_bias.clone() - (self.bn_mean.clone() * &scale)?)?; // (F,)
        let scale = scale.reshape((1, 1, 1, f))?;
        let shift = shift.reshape((1, 1, 1, f))?;
        let x = x.broadcast_mul(&scale)?.broadcast_add(&shift)?;

        // reshape_mel2img (no interpolation: T == spec_width, F == spec_heigth):
        // (1,1,T,F) → (1, freq_ratio, T/freq_ratio, F) → permute(0,1,3,2) → (1, 1, F*freq_ratio, T/freq_ratio)
        let fr = config::AUDIO_FREQ_RATIO;
        let img = x
            .reshape((1, fr, t / fr, f))?
            .permute((0, 1, 3, 2))?
            .contiguous()?
            .reshape((1, 1, f * fr, t / fr))?; // (1,1,256,256)

        // Patch embed: conv2d(1→96, k4,s4) → (1,96,64,64) → flatten → (1,4096,96) → norm.
        let x = self.patch_proj.forward(&img)?; // (1, 96, 64, 64)
        let (_b, c, gh, gw) = x.dims4()?;
        let x = x.reshape((1, c, gh * gw))?.transpose(1, 2)?.contiguous()?; // (1, 4096, 96)
        let mut hidden = self.patch_norm.forward(&x)?;

        for stage in &self.stages {
            hidden = stage.forward(&hidden)?;
        }

        let hidden = self.final_norm.forward(&hidden)?; // (1, 64, 768)
                                                        // avgpool over all spatial tokens per channel (see module doc — identical to HF pooling).
        hidden.mean(1)
    }
}
