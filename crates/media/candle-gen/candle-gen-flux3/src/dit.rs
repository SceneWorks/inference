//! `Flux3Dit` — the image DiT of upstream `src/flux3/dit.py` (`DiT`), the Candle twin of
//! `mlx_gen_flux3::dit`: per-mode input projections, the timestep (and, for a flow map, the
//! target-time) MLP embedders, per-mode `early` and `single` modulations, `DEPTH` per-mode blocks
//! for `image` / `image_cond` / `text`, then `DEPTH_SINGLE` shared single-stream blocks over the
//! per-sample `[image, image_cond, text]` concatenation, and the `LastLayer` velocity head over
//! the image tokens.
//!
//! Every module mirrors its upstream class one-to-one under the upstream state-dict key names, so
//! the released `model.safetensors` loads unchanged. Precision: see [`crate::nn`].
//!
//! # Packing
//!
//! Upstream packs a batch **mode-major** (`Packed`): every mode's tokens of every sample are
//! concatenated, with `mode_indices[mode][sample] = (start, end)`, and attention runs per sample
//! inside each mode stream and per sample over the concatenated single stream. [`SampleTokens`]
//! is one sample of that pack; [`Flux3Dit::forward`] packs the samples itself. An empty mode of a
//! sample (`image_cond` of a text-to-image sample) takes **no** token: upstream gives the empty
//! stream one phantom token so its projections run, but a phantom is never attended (its bound is
//! empty) and never placed in the single stream, so the stream's output never reaches the image
//! velocity — skipping it is the same forward with less work.

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::flux3::{
    timestep_freqs, DitConfig, MODES, TIMESTEP_EMBED_DIM, TIMESTEP_MAX_PERIOD,
};
use candle_gen::{CandleError as Error, Result};

use crate::nn::{
    cat, chunk_last, expect_shape, layernorm_modulation, mul, packed_attention, rmsnorm_rope, silu,
    to, Linear, Loader, Rope,
};

/// `MLPEmbedder`: `out_layer(silu(in_layer(x)))`, bias-free.
struct MlpEmbedder {
    in_layer: Linear,
    out_layer: Linear,
}

impl MlpEmbedder {
    fn load(l: &Loader<'_>, prefix: &str) -> Result<Self> {
        Ok(Self {
            in_layer: l.linear(&format!("{prefix}.in_layer"))?,
            out_layer: l.linear(&format!("{prefix}.out_layer"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.out_layer.forward(&silu(&self.in_layer.forward(x)?)?)
    }
}

/// `ModeModulation`: `lin(silu(vec)).chunk(3)` → (shift, scale, gate).
struct ModeModulation {
    lin: Linear,
}

impl ModeModulation {
    fn load(l: &Loader<'_>, prefix: &str) -> Result<Self> {
        Ok(Self {
            lin: l.linear(&format!("{prefix}.lin"))?,
        })
    }

    /// `vec` `[1, hidden]` → three `[1, hidden]` rows.
    fn forward(&self, vec: &Tensor) -> Result<[Tensor; 3]> {
        let mut parts = chunk_last(&self.lin.forward(&silu(vec)?)?, 3)?;
        let gate = parts.pop().expect("three chunks");
        let scale = parts.pop().expect("three chunks");
        let shift = parts.pop().expect("three chunks");
        Ok([shift, scale, gate])
    }
}

/// `Block`: parallel attention + SwiGLU MLP under one modulation, one gated residual.
struct Block {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    mlp_in: Linear,
    attn_out: Linear,
    mlp_out: Linear,
    query_norm: Tensor,
    key_norm: Tensor,
}

/// One token run's modulation: (shift, scale, gate), each `[N, hidden]` or `[1, hidden]`.
struct Modulation<'a> {
    shift: &'a Tensor,
    scale: &'a Tensor,
    gate: &'a Tensor,
}

impl Block {
    fn load(l: &Loader<'_>, prefix: &str, cfg: &DitConfig) -> Result<Self> {
        let hidden = cfg.hidden;
        let head_dim = cfg.head_dim();
        let mlp_hidden = cfg.mlp_hidden();
        let block = Self {
            q_proj: l.linear(&format!("{prefix}.q_proj"))?,
            k_proj: l.linear(&format!("{prefix}.k_proj"))?,
            v_proj: l.linear(&format!("{prefix}.v_proj"))?,
            mlp_in: l.linear(&format!("{prefix}.mlp_in"))?,
            attn_out: l.linear(&format!("{prefix}.attn_out"))?,
            mlp_out: l.linear(&format!("{prefix}.mlp_out"))?,
            query_norm: l.f32(&format!("{prefix}.norm.query_norm.scale"))?,
            key_norm: l.f32(&format!("{prefix}.norm.key_norm.scale"))?,
        };
        for (name, lin, want) in [
            ("q_proj", &block.q_proj, (hidden, hidden)),
            ("k_proj", &block.k_proj, (hidden, hidden)),
            ("v_proj", &block.v_proj, (hidden, hidden)),
            ("mlp_in", &block.mlp_in, (2 * mlp_hidden, hidden)),
            ("attn_out", &block.attn_out, (hidden, hidden)),
            ("mlp_out", &block.mlp_out, (hidden, mlp_hidden)),
        ] {
            let got = lin.out_in()?;
            if got != want {
                return Err(Error::Msg(format!(
                    "flux3: {prefix}.{name}.weight is {got:?}, the config implies {want:?}"
                )));
            }
        }
        expect_shape(
            &format!("{prefix}.norm.query_norm.scale"),
            &block.query_norm,
            &[head_dim],
        )?;
        expect_shape(
            &format!("{prefix}.norm.key_norm.scale"),
            &block.key_norm,
            &[head_dim],
        )?;
        Ok(block)
    }

    /// `x` `[N, hidden]` (compute dtype); `rope` the `[N, head_dim/2]` tables; `bounds` the
    /// per-sample token spans of this run.
    fn forward(
        &self,
        x: &Tensor,
        rope: &Rope,
        bounds: &[(usize, usize)],
        m: Modulation<'_>,
        heads: usize,
        compute: DType,
    ) -> Result<Tensor> {
        let n = x.dim(0)?;
        let x_mod = layernorm_modulation(x, m.scale, m.shift)?;
        // L (H D) -> 1 H L D
        let split = |t: Tensor| -> Result<Tensor> {
            Ok(t.reshape((1, n, heads, ()))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let q = rmsnorm_rope(
            &split(self.q_proj.forward(&x_mod)?)?,
            &self.query_norm,
            rope,
        )?;
        let k = rmsnorm_rope(&split(self.k_proj.forward(&x_mod)?)?, &self.key_norm, rope)?;
        let v = split(self.v_proj.forward(&x_mod)?)?;
        let attn = packed_attention(&q, &k, &v, bounds, compute)?;
        let mut gate_up = chunk_last(&self.mlp_in.forward(&x_mod)?, 2)?;
        let up = gate_up.pop().expect("two chunks");
        let gate_in = gate_up.pop().expect("two chunks");
        let mlp = self.mlp_out.forward(&mul(&silu(&gate_in)?, &up)?)?;
        let delta = self.attn_out.forward(&attn)?.add(&mlp)?;
        Ok(x.add(&mul(m.gate, &delta)?)?)
    }
}

/// `LastLayer`: adaLN (shift, scale) from `vec`, modulated LayerNorm, linear to the out channels.
struct LastLayer {
    linear: Linear,
    adaln: Linear,
}

impl LastLayer {
    fn load(l: &Loader<'_>, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear: l.linear(&format!("{prefix}.linear"))?,
            // nn.Sequential(nn.SiLU(), nn.Linear(...)) → index 1 holds the Linear.
            adaln: l.linear(&format!("{prefix}.adaLN_modulation.1"))?,
        })
    }

    fn forward(&self, x: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let mut parts = chunk_last(&self.adaln.forward(&silu(vec)?)?, 2)?;
        let scale = parts.pop().expect("two chunks");
        let shift = parts.pop().expect("two chunks");
        self.linear
            .forward(&layernorm_modulation(x, &scale, &shift)?)
    }
}

/// One sample of a packed forward: its text tokens, image tokens and optional reference tokens,
/// each with their ids, and its timesteps.
pub struct SampleTokens<'a> {
    /// `[Lt, text_in_channels]` — the text conditioning (bf16 from the encoder).
    pub text: &'a Tensor,
    /// `Lt` ids `(0, 0, 0, position)`.
    pub text_ids: &'a [[f32; 4]],
    /// `[Ni, image_in_channels]` — the noisy image tokens (f32 state; cast by the projection).
    pub image: &'a Tensor,
    /// `Ni` scaled ids `(0, row·0.5, col·0.5, 0)`.
    pub image_ids: &'a [[f32; 4]],
    /// Reference tokens and their ids (`image_cond`; sc-26236). `None` for text-to-image.
    pub image_cond: Option<(&'a Tensor, &'a [[f32; 4]])>,
    /// The image tokens' time `t` (text and reference tokens sit at time 0, as upstream's
    /// `pack` initializes every mode's timesteps to zero and `denoise` fills only `image`).
    pub timestep: f32,
    /// Flow map only: the time the sample jumps to (`timesteps_ref`).
    pub target_timestep: Option<f32>,
}

/// The loaded DiT.
pub struct Flux3Dit {
    cfg: DitConfig,
    compute: DType,
    device: Device,
    emb_in_image: Linear,
    emb_in_image_cond: Linear,
    txt_in: Linear,
    time_in: MlpEmbedder,
    ref_time_in: Option<MlpEmbedder>,
    /// Indexed as [`MODES`]: image, image_cond, text.
    early_mod: [ModeModulation; 3],
    single_mod: [ModeModulation; 3],
    image_blocks: Vec<Block>,
    image_cond_blocks: Vec<Block>,
    txt_blocks: Vec<Block>,
    single_blocks: Vec<Block>,
    final_layer: LastLayer,
    /// `timestep_embedding`'s frequencies `[1, 128]` f32.
    freqs: Tensor,
}

/// Append one sample's tokens of `mode` to the per-mode pack and record its `(start, end)` run
/// (an empty run for `None`).
fn push_mode<'a>(
    mode: usize,
    data: &mut [Vec<&'a Tensor>; 3],
    ids: &mut [Vec<[f32; 4]>; 3],
    bounds: &mut [Vec<(usize, usize)>; 3],
    tokens: Option<(&'a Tensor, &[[f32; 4]])>,
) -> Result<()> {
    let start = ids[mode].len();
    if let Some((t, i)) = tokens {
        if t.dim(0)? != i.len() {
            return Err(Error::Msg(format!(
                "flux3: {} ids for {} {} tokens",
                i.len(),
                t.dim(0)?,
                MODES[mode]
            )));
        }
        data[mode].push(t);
        ids[mode].extend_from_slice(i);
    }
    bounds[mode].push((start, ids[mode].len()));
    Ok(())
}

impl Flux3Dit {
    /// Install one preflight-validated projection; unknown paths remain typed refusals.
    pub fn install_lora(
        &mut self,
        target: &str,
        down: Tensor,
        up: Tensor,
        scale: f64,
    ) -> Result<()> {
        let (parent, leaf) = target
            .rsplit_once('.')
            .ok_or_else(|| Error::Unsupported(format!("flux3 LoRA: invalid target {target}")))?;
        let (blocks, index) = if let Some(index) = parent.strip_prefix("content_mode_blocks.image.")
        {
            (&mut self.image_blocks, index)
        } else if let Some(index) = parent.strip_prefix("content_mode_blocks.image_cond.") {
            (&mut self.image_cond_blocks, index)
        } else if let Some(index) = parent.strip_prefix("txt_mode_blocks.") {
            (&mut self.txt_blocks, index)
        } else if let Some(index) = parent.strip_prefix("single_blocks.") {
            (&mut self.single_blocks, index)
        } else {
            return Err(Error::Unsupported(format!(
                "flux3 LoRA: unknown target {target}"
            )));
        };
        let index: usize = index
            .parse()
            .map_err(|_| Error::Unsupported(format!("flux3 LoRA: invalid block {target}")))?;
        let block = blocks.get_mut(index).ok_or_else(|| {
            Error::Unsupported(format!("flux3 LoRA: block out of range {target}"))
        })?;
        let linear = match leaf {
            "q_proj" => &mut block.q_proj,
            "k_proj" => &mut block.k_proj,
            "v_proj" => &mut block.v_proj,
            "attn_out" => &mut block.attn_out,
            "mlp_in" => &mut block.mlp_in,
            "mlp_out" => &mut block.mlp_out,
            _ => {
                return Err(Error::Unsupported(format!(
                    "flux3 LoRA: unknown projection {target}"
                )))
            }
        };
        linear.push_lora(down, up, scale)
    }

    /// Load from the package's `model.safetensors` (upstream key names; an mmap read one tensor
    /// at a time, each cast to `compute` as it is consumed, so the load peak is the compute-dtype
    /// model plus a single stored tensor) onto `device`.
    pub fn load(path: &Path, cfg: &DitConfig, compute: DType, device: &Device) -> Result<Self> {
        if !path.is_file() {
            return Err(Error::Msg(format!(
                "flux3: the DiT weights file {} does not exist",
                path.display()
            )));
        }
        let vb = candle_gen::mmap_var_builder(&[path.to_path_buf()], compute, device)
            .map_err(|e| Error::Msg(format!("flux3: loading {}: {e}", path.display())))?;
        Self::from_var_builder(&vb, cfg, compute, device)
    }

    /// Build from an already-mapped checkpoint at `compute` dtype.
    pub fn from_var_builder(
        vb: &VarBuilder<'static>,
        cfg: &DitConfig,
        compute: DType,
        device: &Device,
    ) -> Result<Self> {
        let l = Loader {
            vb,
            compute,
            device,
        };
        let hidden = cfg.hidden;
        let emb_in_image = l.linear("emb_in.image")?;
        let emb_in_image_cond = l.linear("emb_in.image_cond")?;
        let txt_in = l.linear("txt_in")?;
        for (name, lin, want) in [
            (
                "emb_in.image",
                &emb_in_image,
                (hidden, cfg.image_in_channels),
            ),
            (
                "emb_in.image_cond",
                &emb_in_image_cond,
                (hidden, cfg.image_cond_in_channels),
            ),
            ("txt_in", &txt_in, (hidden, cfg.text_in_channels)),
        ] {
            let got = lin.out_in()?;
            if got != want {
                return Err(Error::Msg(format!(
                    "flux3: {name}.weight is {got:?}, the config implies {want:?}"
                )));
            }
        }
        let time_in = MlpEmbedder::load(&l, "time_in")?;
        let has_ref = l.contains("ref_time_in.in_layer.weight");
        if has_ref != cfg.flow_map {
            return Err(Error::Unsupported(format!(
                "flux3: config.json says flow_map = {} but the checkpoint {} `ref_time_in` (the \
                 flow-map key `checkpoint.FLOW_MAP_KEY`)",
                cfg.flow_map,
                if has_ref { "carries" } else { "lacks" }
            )));
        }
        let ref_time_in = if has_ref {
            Some(MlpEmbedder::load(&l, "ref_time_in")?)
        } else {
            None
        };
        let mods = |prefix: &str| -> Result<[ModeModulation; 3]> {
            Ok([
                ModeModulation::load(&l, &format!("{prefix}.image"))?,
                ModeModulation::load(&l, &format!("{prefix}.image_cond"))?,
                ModeModulation::load(&l, &format!("{prefix}.txt"))?,
            ])
        };
        let blocks = |prefix: &str, depth: usize| -> Result<Vec<Block>> {
            (0..depth)
                .map(|i| Block::load(&l, &format!("{prefix}.{i}"), cfg))
                .collect()
        };
        let freqs = timestep_freqs(TIMESTEP_EMBED_DIM, TIMESTEP_MAX_PERIOD);
        let half = freqs.len();
        let dit = Self {
            cfg: cfg.clone(),
            compute,
            device: device.clone(),
            emb_in_image,
            emb_in_image_cond,
            txt_in,
            time_in,
            ref_time_in,
            early_mod: mods("early_stream_modulations")?,
            single_mod: mods("single_stream_modulations")?,
            image_blocks: blocks("content_mode_blocks.image", cfg.depth)?,
            image_cond_blocks: blocks("content_mode_blocks.image_cond", cfg.depth)?,
            txt_blocks: blocks("txt_mode_blocks", cfg.depth)?,
            single_blocks: blocks("single_blocks", cfg.depth_single)?,
            final_layer: LastLayer::load(&l, "final_layer.image")?,
            freqs: Tensor::from_vec(freqs, (1, half), device)?,
        };
        let (out, _) = dit.final_layer.linear.out_in()?;
        if out != cfg.image_in_channels {
            return Err(Error::Msg(format!(
                "flux3: final_layer.image.linear emits {out} channels, the config implies {}",
                cfg.image_in_channels
            )));
        }
        Ok(dit)
    }

    pub fn config(&self) -> &DitConfig {
        &self.cfg
    }

    pub fn compute_dtype(&self) -> DType {
        self.compute
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn is_flow_map(&self) -> bool {
        self.ref_time_in.is_some()
    }

    /// `timestep_embedding(t, 256, 10000)` for one scalar time, in the compute dtype: `1000 · t`
    /// is formed **in the compute dtype** (upstream's timesteps tensor is bf16, so the product
    /// rounds to bf16 before the f32 sinusoid), then `cat[cos, sin]` cast back to it.
    pub fn timestep_embedding(&self, t: f32) -> Result<Tensor> {
        let t = to(&Tensor::new(&[[t]], &self.device)?, self.compute)?;
        let thousand = to(&Tensor::new(&[[1000.0f32]], &self.device)?, self.compute)?;
        let scaled = to(&t.mul(&thousand)?, DType::F32)?;
        let args = scaled.broadcast_mul(&self.freqs)?;
        to(&Tensor::cat(&[args.cos()?, args.sin()?], 1)?, self.compute)
    }

    /// `vec[mode]` of one sample: `time_in(emb(t))` plus, for a flow map, `ref_time_in(emb(t_ref))`;
    /// `[1, hidden]` in the compute dtype.
    fn mode_vec(&self, t: f32, t_ref: Option<f32>) -> Result<Tensor> {
        let mut vec = self.time_in.forward(&self.timestep_embedding(t)?)?;
        if let Some(r) = &self.ref_time_in {
            let t_ref = t_ref.ok_or_else(|| {
                Error::Msg("flux3: a flow-map DiT needs the target timestep of every sample".into())
            })?;
            vec = vec.add(&r.forward(&self.timestep_embedding(t_ref)?)?)?;
        } else if t_ref.is_some() {
            return Err(Error::Msg(
                "flux3: a CFG DiT takes no target timestep (timesteps_ref is a flow-map input)"
                    .into(),
            ));
        }
        Ok(vec)
    }

    fn rope(&self, ids: &[[f32; 4]]) -> Result<Rope> {
        Rope::new(ids, &self.cfg.axes_dim, self.cfg.theta, &self.device)
    }

    /// Velocity prediction for the image tokens of every sample, `[Σ Ni, image_in_channels]` in
    /// the compute dtype, in sample order (the image tokens of sample 0, then sample 1, …).
    pub fn forward(&self, samples: &[SampleTokens<'_>]) -> Result<Tensor> {
        if samples.is_empty() {
            return Err(Error::Msg("flux3: forward over no samples".into()));
        }
        let heads = self.cfg.heads;
        let compute = self.compute;
        // --- per-mode packs -------------------------------------------------------------------
        // mode index: 0 image, 1 image_cond, 2 text (MODES order).
        let mut data: [Vec<&Tensor>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut ids: [Vec<[f32; 4]>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut bounds: [Vec<(usize, usize)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        // vec / modulation rows per (mode, sample), [1, hidden] each
        let mut early: Vec<[[Tensor; 3]; 3]> = Vec::with_capacity(samples.len());
        let mut single: Vec<[[Tensor; 3]; 3]> = Vec::with_capacity(samples.len());
        let mut vecs: Vec<[Tensor; 3]> = Vec::with_capacity(samples.len());
        for s in samples {
            let (n_img, c_img) = s.image.dims2()?;
            let (n_txt, c_txt) = s.text.dims2()?;
            if s.image_ids.len() != n_img || s.text_ids.len() != n_txt {
                return Err(Error::Msg(format!(
                    "flux3: ids ({} image, {} text) do not match the tokens ({n_img}, {n_txt})",
                    s.image_ids.len(),
                    s.text_ids.len()
                )));
            }
            if c_img != self.cfg.image_in_channels || c_txt != self.cfg.text_in_channels {
                return Err(Error::Msg(format!(
                    "flux3: token widths {:?} / {:?} do not match the DiT ({} / {})",
                    s.image.dims(),
                    s.text.dims(),
                    self.cfg.image_in_channels,
                    self.cfg.text_in_channels
                )));
            }
            push_mode(
                0,
                &mut data,
                &mut ids,
                &mut bounds,
                Some((s.image, s.image_ids)),
            )?;
            push_mode(1, &mut data, &mut ids, &mut bounds, s.image_cond)?;
            push_mode(
                2,
                &mut data,
                &mut ids,
                &mut bounds,
                Some((s.text, s.text_ids)),
            )?;
            // timesteps: image at t, the other modes at 0; the target time reaches every mode.
            let v_img = self.mode_vec(s.timestep, s.target_timestep)?;
            let v_zero = self.mode_vec(0.0, s.target_timestep)?;
            let v = [v_img, v_zero.clone(), v_zero];
            let mut e: Vec<[Tensor; 3]> = Vec::with_capacity(3);
            let mut g: Vec<[Tensor; 3]> = Vec::with_capacity(3);
            for (mode, vec) in v.iter().enumerate() {
                e.push(self.early_mod[mode].forward(vec)?);
                g.push(self.single_mod[mode].forward(vec)?);
            }
            early.push([e.remove(0), e.remove(0), e.remove(0)]);
            single.push([g.remove(0), g.remove(0), g.remove(0)]);
            vecs.push(v);
        }

        // --- per-mode streams ------------------------------------------------------------------
        let mut streams: [Option<Tensor>; 3] = [None, None, None];
        for mode in 0..3 {
            if data[mode].is_empty() {
                continue;
            }
            let x = cat(&data[mode], 0)?;
            let x = match mode {
                0 => self.emb_in_image.forward(&x)?,
                1 => self.emb_in_image_cond.forward(&x)?,
                _ => self.txt_in.forward(&x)?,
            };
            let rope = self.rope(&ids[mode])?;
            let m = self.broadcast_modulation(&early, mode, &bounds[mode])?;
            let blocks = match mode {
                0 => &self.image_blocks,
                1 => &self.image_cond_blocks,
                _ => &self.txt_blocks,
            };
            let mut x = x;
            for block in blocks {
                x = block.forward(
                    &x,
                    &rope,
                    &bounds[mode],
                    Modulation {
                        shift: &m[0],
                        scale: &m[1],
                        gate: &m[2],
                    },
                    heads,
                    compute,
                )?;
            }
            streams[mode] = Some(x);
        }

        // --- single stream: per sample, [image, image_cond, text] -------------------------------
        let hidden = self.cfg.hidden;
        let mut chunks: Vec<Tensor> = Vec::new();
        let mut id_chunks: Vec<[f32; 4]> = Vec::new();
        // Each non-empty (sample, mode) run's modulation rows and its length, in stream order.
        let mut mods: [Vec<Tensor>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut run_lengths: Vec<usize> = Vec::new();
        let mut sample_bounds: Vec<(usize, usize)> = Vec::with_capacity(samples.len());
        let mut placement: Vec<(usize, usize)> = Vec::with_capacity(samples.len()); // image rows
        let mut offset = 0usize;
        for (s, single_rows) in single.iter().enumerate() {
            let start = offset;
            for (mode, stream) in streams.iter().enumerate() {
                let (a, b) = bounds[mode][s];
                if b > a {
                    let stream = stream.as_ref().expect("a non-empty mode ran");
                    chunks.push(stream.narrow(0, a, b - a)?);
                    id_chunks.extend_from_slice(&ids[mode][a..b]);
                    for (i, m) in mods.iter_mut().enumerate() {
                        m.push(single_rows[mode][i].clone());
                    }
                    run_lengths.push(b - a);
                    if mode == 0 {
                        placement.push((offset, offset + b - a));
                    }
                    offset += b - a;
                }
            }
            sample_bounds.push((start, offset));
        }
        let refs: Vec<&Tensor> = chunks.iter().collect();
        let mut h = cat(&refs, 0)?;
        let rope = self.rope(&id_chunks)?;
        // Expand each [1, hidden] modulation row over its run, so the three are [N, hidden].
        let expand = |rows: &[Tensor]| -> Result<Tensor> {
            let mut parts = Vec::with_capacity(rows.len());
            for (row, &len) in rows.iter().zip(&run_lengths) {
                parts.push(row.broadcast_as((len, hidden))?.contiguous()?);
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            cat(&refs, 0)
        };
        let m = [expand(&mods[0])?, expand(&mods[1])?, expand(&mods[2])?];
        for block in &self.single_blocks {
            h = block.forward(
                &h,
                &rope,
                &sample_bounds,
                Modulation {
                    shift: &m[0],
                    scale: &m[1],
                    gate: &m[2],
                },
                heads,
                compute,
            )?;
        }

        // --- the image tokens, in sample order, through the velocity head ----------------------
        let mut outs = Vec::with_capacity(samples.len());
        for (s, &(a, b)) in placement.iter().enumerate() {
            let image = h.narrow(0, a, b - a)?;
            outs.push(self.final_layer.forward(&image, &vecs[s][0])?);
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        cat(&refs, 0)
    }

    /// The early-stream modulation of `mode` over its packed tokens: each sample's `[1, hidden]`
    /// row broadcast over that sample's span, concatenated → three `[N, hidden]`.
    fn broadcast_modulation(
        &self,
        rows: &[[[Tensor; 3]; 3]],
        mode: usize,
        bounds: &[(usize, usize)],
    ) -> Result<[Tensor; 3]> {
        let hidden = self.cfg.hidden;
        let one = |i: usize| -> Result<Tensor> {
            let mut parts = Vec::with_capacity(bounds.len());
            for (s, &(a, b)) in bounds.iter().enumerate() {
                if b > a {
                    parts.push(
                        rows[s][mode][i]
                            .broadcast_as((b - a, hidden))?
                            .contiguous()?,
                    );
                }
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            cat(&refs, 0)
        };
        Ok([one(0)?, one(1)?, one(2)?])
    }
}
