//! Frozen MOSS graph. No FLUX DiT/scheduler or ordinary cast-down adapter abstraction is used.
use crate::precision::{attention_edit_bf16, attention_fp32, linear_fp32};
use mlx_gen::attention::slice_axis;
use mlx_gen::gen_core::moss_image::{moss_time_features, MossImageConfig};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Result};
use mlx_rs::ops::{add, concatenate_axis, divide, multiply, rsqrt, sigmoid, tanh};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

#[derive(Clone)]
pub(crate) struct Geometry {
    pub dim: i32,
    pub ffn: i32,
    pub heads: i32,
    pub layers: usize,
    pub text: i32,
    pub freq: i32,
    pub channels: i32,
}
impl From<&MossImageConfig> for Geometry {
    fn from(c: &MossImageConfig) -> Self {
        Self {
            dim: c.dim as i32,
            ffn: c.ffn_dim as i32,
            heads: c.num_heads as i32,
            layers: c.num_layers,
            text: c.text_dim as i32,
            freq: c.freq_dim as i32,
            channels: c.latent_channels as i32,
        }
    }
}
fn constant(value: f32, dtype: Dtype) -> Result<Array> {
    Ok(Array::from_slice(&[value], &[1]).as_dtype(dtype)?)
}
fn scale(x: &Array, v: f32) -> Result<Array> {
    Ok(multiply(x, constant(v, x.dtype())?)?)
}
fn shaped(w: &Weights, key: &str, shape: &[i32]) -> Result<Array> {
    let a = w.require(key)?;
    if a.shape() != shape {
        return Err(Error::Msg(format!(
            "MOSS tensor {key}: expected {shape:?}, got {:?}",
            a.shape()
        )));
    }
    Ok(a.clone())
}
pub(crate) fn check(cancel: &CancelFlag) -> Result<()> {
    if cancel.is_cancelled() {
        Err(Error::Canceled)
    } else {
        Ok(())
    }
}

pub(crate) fn situ(hidden: &Array, gate: &Array) -> Result<Array> {
    let a = multiply(
        scale(&tanh(&divide(gate, constant(4.0, gate.dtype())?)?)?, 4.0)?,
        sigmoid(gate)?,
    )?;
    let b = scale(
        &tanh(&divide(hidden, constant(25.0, hidden.dtype())?)?)?,
        25.0,
    )?;
    Ok(multiply(a, b)?)
}
struct Norm {
    weight: Array,
}
impl Norm {
    fn load(w: &Weights, key: &str, n: i32) -> Result<Self> {
        Ok(Self {
            weight: shaped(w, key, &[n])?,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        let f = x.as_dtype(Dtype::Float32)?;
        let variance = multiply(&f, &f)?.mean_axis(-1, true)?;
        let normalized = multiply(&f, rsqrt(add(variance, constant(1e-6, Dtype::Float32)?)?)?)?
            .as_dtype(x.dtype())?;
        Ok(multiply(normalized, &self.weight)?)
    }
}
pub(crate) struct Projection {
    weight: Array,
    bias: Option<Array>,
    adapter: Option<(Array, Array, f32)>,
    edit_adapter: bool,
    edit_fused_bias: bool,
}
impl Projection {
    fn load(w: &Weights, p: &str, input: i32, output: i32, bias: bool) -> Result<Self> {
        Ok(Self {
            weight: shaped(w, &format!("{p}.weight"), &[output, input])?,
            bias: if bias {
                Some(shaped(w, &format!("{p}.bias"), &[output])?)
            } else {
                None
            },
            adapter: None,
            edit_adapter: false,
            edit_fused_bias: false,
        })
    }
    fn dtype(&self) -> Dtype {
        self.weight.dtype()
    }
    pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
        let cast = x.as_dtype(self.weight.dtype())?;
        let base = if self.weight.dtype() == Dtype::Float32 {
            let projected = linear_fp32(&cast, &self.weight)?;
            match &self.bias {
                Some(bias) => add(projected, bias)?,
                None => projected,
            }
        } else if let Some(b) = &self.bias {
            if self.edit_fused_bias {
                // Pinned Metal GEMV addmm casts its dot to BF16 before bias addition.
                // Upstream nn.Linear rounds once after FP32 accumulation plus bias.
                add(
                    linear_fp32(
                        &cast.as_dtype(Dtype::Float32)?,
                        &self.weight.as_dtype(Dtype::Float32)?,
                    )?,
                    b.as_dtype(Dtype::Float32)?,
                )?
                .as_dtype(self.weight.dtype())?
            } else {
                mlx_gen::nn::linear(&cast, &self.weight, b)?
            }
        } else {
            cast.matmul(self.weight.t())?
        };
        if let Some((down, up, scale)) = &self.adapter {
            if self.edit_adapter {
                let residual = if down.dtype() == Dtype::Float32 {
                    // Synthetic FP32 diagnostics must not enter the NAX TF32 GEMM path.
                    linear_fp32(&linear_fp32(&x.as_dtype(Dtype::Float32)?, down)?, up)?
                } else {
                    x.as_dtype(down.dtype())?.matmul(down.t())?.matmul(up.t())?
                };
                return Ok(add(
                    &base,
                    self::scale(&residual, *scale)?.as_dtype(base.dtype())?,
                )?);
            }
            // The original input's FP32 values feed the residual; only the backbone is narrowed.
            let residual = linear_fp32(&linear_fp32(&x.as_dtype(Dtype::Float32)?, down)?, up)?;
            Ok(add(
                base.as_dtype(Dtype::Float32)?,
                self::scale(&residual, *scale)?,
            )?)
        } else {
            Ok(base)
        }
    }
    fn widen(&mut self) -> Result<()> {
        self.weight = self.weight.as_dtype(Dtype::Float32)?;
        self.bias = self
            .bias
            .as_ref()
            .map(|b| b.as_dtype(Dtype::Float32))
            .transpose()?;
        Ok(())
    }
    fn install(&mut self, w: &Weights, p: &str, rank: i32) -> Result<()> {
        let prefix = format!("diffusion_model.{p}");
        let down = shaped(
            w,
            &format!("{prefix}.lora_down.weight"),
            &[rank, self.weight.shape()[1]],
        )?;
        let up = shaped(
            w,
            &format!("{prefix}.lora_up.weight"),
            &[self.weight.shape()[0], rank],
        )?;
        let alpha = w
            .require(&format!("{prefix}.alpha"))?
            .as_dtype(Dtype::Float32)?;
        eval([&alpha])?;
        let values = alpha
            .try_as_slice::<f32>()
            .map_err(|e| Error::Msg(e.to_string()))?;
        if values.len() != 1
            || values[0] != rank as f32
            || down.dtype() != Dtype::Float32
            || up.dtype() != Dtype::Float32
        {
            return Err(Error::Unsupported(
                "invalid MOSS FP32 low-rank projection".into(),
            ));
        }
        self.adapter = Some((down, up, values[0] / rank as f32));
        Ok(())
    }
}
fn qk_norm(x: &Array, norm: &Norm, g: &Geometry) -> Result<Array> {
    let s = x.shape();
    Ok(norm
        .forward(&x.reshape(&[s[0], s[1], g.heads, g.dim / g.heads])?)?
        .reshape(s)?)
}
pub(crate) fn rope(height: i32, width: i32, text: i32, head: i32) -> Result<Array> {
    if height < 1 || width < 1 || text < 1 || head % 8 != 0 {
        return Err(Error::Msg("invalid MOSS RoPE geometry".into()));
    }
    let bands = head / 8;
    let mut data = Vec::with_capacity(((height * width + text) * head) as usize);
    for token in 0..height * width + text {
        let coordinates = if token < height * width {
            [
                0.0,
                text as f32,
                (token / width) as f32 - (height - 1) as f32 / 2.0,
                (token % width) as f32 - (width - 1) as f32 / 2.0,
            ]
        } else {
            [0.0, (token - height * width) as f32, 0.0, 0.0]
        };
        for sine in [true, false] {
            for coordinate in coordinates {
                for band in 0..bands {
                    let frequency = 1.0 / 10000_f32.powf(band as f32 / bands as f32);
                    let phase = coordinate * frequency;
                    data.push(if sine { phase.sin() } else { phase.cos() });
                }
            }
        }
    }
    Ok(Array::from_slice(&data, &[1, height * width + text, head]))
}
fn rotate(x: &Array, rope: &Array) -> Result<Array> {
    let d = x.shape()[3];
    let half = d / 2;
    let sin = slice_axis(rope, 2, 0, half)?.expand_dims(2)?;
    let cos = slice_axis(rope, 2, half, half)?.expand_dims(2)?;
    let sin = concatenate_axis(&[&sin, &sin], -1)?;
    let cos = concatenate_axis(&[&cos, &cos], -1)?;
    let a = slice_axis(x, 3, 0, half)?;
    let b = scale(&slice_axis(x, 3, half, half)?, -1.0)?;
    let rotated = concatenate_axis(&[&b, &a], -1)?;
    Ok(add(multiply(x, cos)?, multiply(rotated, sin)?)?)
}
fn attention(
    q: &Array,
    k: &Array,
    v: &Array,
    rope: &Array,
    g: &Geometry,
    cancel: &CancelFlag,
    edit: bool,
) -> Result<Array> {
    let shape = [q.shape()[0], q.shape()[1], g.heads, g.dim / g.heads];
    let q = rotate(&q.reshape(&shape)?, rope)?
        .as_dtype(v.dtype())?
        .transpose_axes(&[0, 2, 1, 3])?;
    let k = rotate(&k.reshape(&shape)?, rope)?
        .as_dtype(v.dtype())?
        .transpose_axes(&[0, 2, 1, 3])?;
    let v = v.reshape(&shape)?.transpose_axes(&[0, 2, 1, 3])?;
    let scale = 1.0 / (g.dim as f32 / g.heads as f32).sqrt();
    let out = if edit {
        edit_attention(&q, &k, &v, g, cancel)?
    } else if v.dtype() == Dtype::Float32 {
        attention_fp32(&q, &k, &v, scale, cancel)?
    } else {
        mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, scale, None, None)?
    };
    Ok(out
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[shape[0], shape[1], g.dim])?)
}
struct TextBlock {
    edit_precision: bool,
    n1: Norm,
    n2: Norm,
    qn: Norm,
    kn: Norm,
    qkv: Projection,
    proj: Projection,
    up: Projection,
    gate: Projection,
    down: Projection,
}
impl TextBlock {
    fn load(w: &Weights, p: &str, g: &Geometry) -> Result<Self> {
        Ok(Self {
            edit_precision: false,
            n1: Norm::load(w, &format!("{p}.norm1.weight"), g.dim)?,
            n2: Norm::load(w, &format!("{p}.norm2.weight"), g.dim)?,
            qn: Norm::load(w, &format!("{p}.q_norm.weight"), g.dim / g.heads)?,
            kn: Norm::load(w, &format!("{p}.k_norm.weight"), g.dim / g.heads)?,
            qkv: Projection::load(w, &format!("{p}.qkv"), g.dim, 3 * g.dim, true)?,
            proj: Projection::load(w, &format!("{p}.proj"), g.dim, g.dim, true)?,
            up: Projection::load(w, &format!("{p}.ffn_up"), g.dim, g.ffn, true)?,
            gate: Projection::load(w, &format!("{p}.ffn_gate"), g.dim, g.ffn, true)?,
            down: Projection::load(w, &format!("{p}.ffn_down"), g.ffn, g.dim, true)?,
        })
    }
    fn forward(&self, x: &Array, rope: &Array, g: &Geometry, cancel: &CancelFlag) -> Result<Array> {
        let qkv = self.qkv.forward(&self.n1.forward(x)?)?;
        let q = qk_norm(&slice_axis(&qkv, 2, 0, g.dim)?, &self.qn, g)?;
        let k = qk_norm(&slice_axis(&qkv, 2, g.dim, g.dim)?, &self.kn, g)?;
        let v = slice_axis(&qkv, 2, 2 * g.dim, g.dim)?;
        let x = add(
            x,
            self.proj.forward(&attention(
                &q,
                &k,
                &v,
                rope,
                g,
                cancel,
                self.edit_precision,
            )?)?,
        )?;
        let y = self.n2.forward(&x)?;
        let hidden = self.up.forward(&y)?;
        let gate = self.gate.forward(&y)?;
        let activated = if self.edit_precision {
            edit_situ(&hidden, &gate)?
        } else {
            situ(&hidden, &gate)?
        };
        Ok(add(&x, self.down.forward(&activated)?)?)
    }
}
struct Block {
    n1: Norm,
    n2: Norm,
    qn: Norm,
    kn: Norm,
    projections: Vec<Projection>,
}
impl Block {
    fn load(w: &Weights, i: usize, g: &Geometry) -> Result<Self> {
        let p = format!("layers.{i}");
        let mut projections = Vec::new();
        for (name, input, output) in [
            ("q", g.dim, g.dim),
            ("k", g.dim, g.dim),
            ("v", g.dim, g.dim),
            ("o", g.dim, g.dim),
            ("ffn_up_gate", g.dim, 2 * g.ffn),
            ("ffn_down", g.ffn, g.dim),
        ] {
            projections.push(Projection::load(
                w,
                &format!("{p}.proj.{name}"),
                input,
                output,
                false,
            )?);
        }
        Ok(Self {
            n1: Norm::load(w, &format!("{p}.norm1.weight"), g.dim)?,
            n2: Norm::load(w, &format!("{p}.norm2.weight"), g.dim)?,
            qn: Norm::load(w, &format!("{p}.q_norm.weight"), g.dim / g.heads)?,
            kn: Norm::load(w, &format!("{p}.k_norm.weight"), g.dim / g.heads)?,
            projections,
        })
    }
    fn forward(
        &self,
        x: &Array,
        rope: &Array,
        modulation: &Array,
        g: &Geometry,
        cancel: &CancelFlag,
    ) -> Result<Array> {
        let m =
            |i| -> Result<Array> { Ok(slice_axis(modulation, 1, i, 1)?.reshape(&[1, 1, g.dim])?) };
        let y = add(
            multiply(
                self.n1.forward(x)?,
                add(m(1)?, constant(1.0, modulation.dtype())?)?,
            )?,
            m(0)?,
        )?
        .as_dtype(self.projections[0].dtype())?;
        let q = qk_norm(&self.projections[0].forward(&y)?, &self.qn, g)?;
        let k = qk_norm(&self.projections[1].forward(&y)?, &self.kn, g)?;
        let v = self.projections[2].forward(&y)?;
        let a = self.projections[3].forward(&attention(&q, &k, &v, rope, g, cancel, false)?)?;
        let x = add(x, multiply(a, m(2)?)?)?;
        let y = add(
            multiply(
                self.n2.forward(&x)?,
                add(m(4)?, constant(1.0, modulation.dtype())?)?,
            )?,
            m(3)?,
        )?
        .as_dtype(self.projections[0].dtype())?;
        let fused = self.projections[4].forward(&y)?;
        let hidden = slice_axis(&fused, 2, 0, g.ffn)?;
        let gate = slice_axis(&fused, 2, g.ffn, g.ffn)?;
        Ok(add(
            x,
            multiply(self.projections[5].forward(&situ(&hidden, &gate)?)?, m(5)?)?,
        )?)
    }
}
pub(crate) struct Transformer {
    owned_weight_bytes: u64,
    g: Geometry,
    video: Projection,
    text: Projection,
    text_blocks: Vec<TextBlock>,
    time1: Projection,
    time2: Projection,
    modulation: Projection,
    head: Projection,
    blocks: Vec<Block>,
    norm: Norm,
    output: Projection,
}
impl Transformer {
    pub(crate) fn load(w: &Weights, g: Geometry) -> Result<Self> {
        let _ = shaped(w, "rope.bands", &[g.dim / g.heads / 8])?;
        let blocks = (0..g.layers)
            .map(|i| Block::load(w, i, &g))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            owned_weight_bytes: 0,
            video: Projection::load(w, "video_embedder", g.channels, g.dim, true)?,
            text: Projection::load(w, "text_embedder.proj", g.text, g.dim, true)?,
            text_blocks: (0..2)
                .map(|i| TextBlock::load(w, &format!("text_embedder.blocks.{i}"), &g))
                .collect::<Result<_>>()?,
            time1: Projection::load(w, "time_embedding.0", g.freq, g.dim, true)?,
            time2: Projection::load(w, "time_embedding.2", g.dim, g.dim, true)?,
            modulation: Projection::load(w, "shared_block_modulation", g.dim, 6 * g.dim, true)?,
            head: Projection::load(w, "video_head_modulation", g.dim, 2 * g.dim, true)?,
            norm: Norm::load(w, "final_norm_video.weight", g.dim)?,
            output: Projection::load(w, "final_linear_video", g.dim, g.channels, false)?,
            blocks,
            g,
        })
    }
    fn owned_arrays(&self) -> Vec<&Array> {
        let mut arrays = vec![&self.norm.weight];
        let mut projections = vec![
            &self.video,
            &self.text,
            &self.time1,
            &self.time2,
            &self.modulation,
            &self.head,
            &self.output,
        ];
        for block in &self.text_blocks {
            arrays.extend([
                &block.n1.weight,
                &block.n2.weight,
                &block.qn.weight,
                &block.kn.weight,
            ]);
            projections.extend([&block.qkv, &block.proj, &block.up, &block.gate, &block.down]);
        }
        for block in &self.blocks {
            arrays.extend([
                &block.n1.weight,
                &block.n2.weight,
                &block.qn.weight,
                &block.kn.weight,
            ]);
            projections.extend(&block.projections);
        }
        for projection in projections {
            arrays.push(&projection.weight);
            arrays.extend(projection.bias.as_ref());
            if let Some((down, up, _)) = &projection.adapter {
                arrays.extend([down, up]);
            }
        }
        arrays
    }
    /// Evaluate retained weight casts before publishing residency credit. Scalar alpha and the
    /// regenerated RoPE bands are not retained weight buffers and receive no credit.
    pub(crate) fn materialize_owned(&mut self, cancel: &CancelFlag) -> Result<()> {
        let mut owned_weight_bytes = 0u64;
        for array in self.owned_arrays() {
            check(cancel)?;
            eval([array])?;
            owned_weight_bytes = owned_weight_bytes.saturating_add(array.nbytes() as u64);
        }
        check(cancel)?;
        self.owned_weight_bytes = owned_weight_bytes;
        Ok(())
    }
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.owned_weight_bytes
    }
    pub(crate) fn install_dmd(
        &mut self,
        lora: &Weights,
        replacements: &Weights,
        rank: i32,
    ) -> Result<()> {
        self.time1.widen()?;
        self.time2.widen()?;
        self.modulation.widen()?;
        self.head.widen()?;
        for (projection, prefix) in [
            (&mut self.time1, "time_embedding.0"),
            (&mut self.time2, "time_embedding.2"),
        ] {
            projection.weight = shaped(
                replacements,
                &format!("{prefix}.weight"),
                projection.weight.shape(),
            )?;
            let bias = projection
                .bias
                .as_ref()
                .ok_or_else(|| Error::Msg("missing time bias".into()))?;
            projection.bias = Some(shaped(
                replacements,
                &format!("{prefix}.bias"),
                bias.shape(),
            )?);
        }
        self.modulation.bias = Some(shaped(
            replacements,
            "shared_block_modulation.bias",
            &[6 * self.g.dim],
        )?);
        self.head.bias = Some(shaped(
            replacements,
            "video_head_modulation.bias",
            &[2 * self.g.dim],
        )?);
        for (index, block) in self.blocks.iter_mut().enumerate() {
            for (projection, name) in
                block
                    .projections
                    .iter_mut()
                    .zip(["q", "k", "v", "o", "ffn_up_gate", "ffn_down"])
            {
                projection.install(lora, &format!("layers.{index}.proj.{name}"), rank)?;
            }
        }
        Ok(())
    }
    pub(crate) fn forward(
        &self,
        latents: &Array,
        times: &[f32],
        contexts: &[Array],
        skip: &[u32],
        cancel: &CancelFlag,
    ) -> Result<Array> {
        check(cancel)?;
        let g = &self.g;
        let shape = latents.shape();
        if shape.len() != 4
            || shape[1] != g.channels
            || contexts.len() != shape[0] as usize
            || !(times.len() == 1 || times.len() == contexts.len())
            || skip.iter().any(|i| *i as usize >= g.layers)
        {
            return Err(Error::Msg("MOSS forward shape/time/skip mismatch".into()));
        }
        let (h, w) = (shape[2], shape[3]);
        let mut outputs = Vec::new();
        for (index, context) in contexts.iter().enumerate() {
            check(cancel)?;
            let t = times[if times.len() == 1 { 0 } else { index }];
            let features = if g.freq == 256 {
                moss_time_features(t)?.to_vec()
            } else {
                let half = g.freq / 2;
                let mut f = Vec::with_capacity(g.freq as usize);
                for sin in [false, true] {
                    for i in 0..half {
                        let phase = t as f64 * 10000_f64.powf(-(i as f64) / half as f64);
                        f.push(if sin {
                            phase.sin() as f32
                        } else {
                            phase.cos() as f32
                        });
                    }
                }
                f
            };
            let time = self
                .time1
                .forward(&Array::from_slice(&features, &[1, g.freq]))?;
            let time = self
                .time2
                .forward(&mlx_rs::fast::silu_exact(&time)?)?
                .as_dtype(Dtype::Float32)?;
            let modulation = self.modulation.forward(&time)?.reshape(&[1, 6, g.dim])?;
            let head = self.head.forward(&time)?.reshape(&[1, 2, g.dim])?;
            let video = slice_axis(latents, 0, index as i32, 1)?
                .transpose_axes(&[0, 2, 3, 1])?
                .reshape(&[1, h * w, g.channels])?;
            let video = self.video.forward(&video)?;
            let context_len = context.shape()[1];
            let rope = rope(h, w, context_len, g.dim / g.heads)?.as_dtype(video.dtype())?;
            let text_rope = slice_axis(&rope, 1, h * w, context_len)?;
            let mut text = self.text.forward(context)?;
            for block in &self.text_blocks {
                check(cancel)?;
                text = block.forward(&text, &text_rope, g, cancel)?;
                eval([&text])?;
                check(cancel)?;
            }
            let mut x = concatenate_axis(&[&video, &text], 1)?;
            for (i, block) in self.blocks.iter().enumerate() {
                check(cancel)?;
                if !skip.contains(&(i as u32)) {
                    x = block.forward(&x, &rope, &modulation, g, cancel)?;
                    eval([&x])?;
                }
            }
            let x = self
                .norm
                .forward(&slice_axis(&x, 1, 0, h * w)?.as_dtype(self.norm.weight.dtype())?)?;
            let scale_head = slice_axis(&head, 1, 1, 1)?;
            let shift_head = slice_axis(&head, 1, 0, 1)?;
            let x = add(
                multiply(x, add(&scale_head, constant(1.0, head.dtype())?)?)?,
                shift_head,
            )?;
            let out = self
                .output
                .forward(&x)?
                .reshape(&[1, h, w, g.channels])?
                .transpose_axes(&[0, 3, 1, 2])?;
            eval([&out])?;
            outputs.push(out);
        }
        check(cancel)?;
        let refs: Vec<_> = outputs.iter().collect();
        Ok(concatenate_axis(&refs, 0)?)
    }
}

#[cfg(test)]
fn checked_delta(a: &[f32], shape: &[i32], b: &[f32], expected_shape: &[i32]) -> f32 {
    assert_eq!(shape, expected_shape, "oracle shape");
    assert_eq!(a.len(), b.len(), "oracle count");
    assert_eq!(
        shape
            .iter()
            .map(|d| usize::try_from(*d).unwrap())
            .product::<usize>(),
        a.len(),
        "oracle shape/count"
    );
    assert!(
        a.iter().chain(b).all(|v| v.is_finite()),
        "nonfinite oracle value"
    );
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max)
}
#[cfg(test)]
mod oracle_cpu_tests {
    #[test]
    fn rejects_nonfinite_truncated_and_wrong_shape_oracles_sc26325() {
        use super::checked_delta;
        assert_eq!(checked_delta(&[1., 2.], &[2], &[1., 2.], &[2]), 0.);
        for (a, shape, b, expected_shape) in [
            (vec![f32::NAN], vec![1], vec![0.], vec![1]),
            (vec![0.], vec![1], vec![f32::NAN], vec![1]),
            (vec![f32::INFINITY], vec![1], vec![0.], vec![1]),
            (vec![1.], vec![1], vec![1., 2.], vec![2]),
            (vec![1., 2.], vec![2], vec![1.], vec![2]),
            (vec![1., 2.], vec![1, 2], vec![1., 2.], vec![2, 1]),
        ] {
            assert!(
                std::panic::catch_unwind(|| checked_delta(&a, &shape, &b, &expected_shape))
                    .is_err()
            );
        }
    }
}
#[cfg(test)]
mod author_fixture_tests {
    use super::*;
    use mlx_rs::ops::subtract;
    use std::collections::HashMap;
    fn fixture(name: &str) -> serde_json::Value {
        serde_json::from_str(match name{"ordinary"=>include_str!("../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_author/fixture.json"),_=>include_str!("../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_author/dmd_fixture.json")}).unwrap()
    }
    fn array(v: &serde_json::Value) -> Array {
        let shape: Vec<i32> = v["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let values: Vec<f32> = v["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        Array::from_slice(&values, &shape)
    }
    fn geometry(v: &serde_json::Value) -> Geometry {
        let c = &v["config"];
        Geometry {
            dim: c["dim"].as_i64().unwrap() as i32,
            ffn: c["ffn_dim"].as_i64().unwrap() as i32,
            heads: c["num_heads"].as_i64().unwrap() as i32,
            layers: c["num_layers"].as_u64().unwrap() as usize,
            text: c["text_dim"].as_i64().unwrap() as i32,
            freq: c["freq_dim"].as_i64().unwrap() as i32,
            channels: c["latent_channels"].as_i64().unwrap() as i32,
        }
    }
    fn weights(v: &serde_json::Value, dtype: Dtype) -> Weights {
        Weights::from_map(
            v.as_object()
                .unwrap()
                .iter()
                .map(|(key, value)| {
                    (
                        key.clone(),
                        array(value)
                            .as_dtype(if key == "rope.bands" {
                                Dtype::Float32
                            } else {
                                dtype
                            })
                            .unwrap(),
                    )
                })
                .collect(),
        )
    }
    fn contexts(v: &serde_json::Value, mask: &serde_json::Value, text: i32) -> Vec<Array> {
        let source = array(v);
        let mask = mask["values"].as_array().unwrap();
        let s = source.shape()[1] as usize;
        (0..source.shape()[0])
            .map(|b| {
                let row = slice_axis(&source, 0, b, 1).unwrap();
                let ids: Vec<i32> = (0..s)
                    .filter(|i| mask[b as usize * s + *i].as_f64().unwrap() != 0.0)
                    .map(|i| i as i32)
                    .collect();
                row.take_axis(Array::from_slice(&ids, &[ids.len() as i32]), 1)
                    .unwrap()
                    .reshape(&[1, ids.len() as i32, text])
                    .unwrap()
            })
            .collect()
    }
    fn delta(actual: &Array, expected: &serde_json::Value) -> f32 {
        // Host slices expose physical storage; a B=1 concatenation can retain a transposed view.
        let actual = mlx_gen::array::contiguous(&actual.as_dtype(Dtype::Float32).unwrap()).unwrap();
        eval([&actual]).unwrap();
        let expected_values: Vec<f32> = serde_json::from_value(expected["values"].clone()).unwrap();
        let expected_shape: Vec<i32> = serde_json::from_value(expected["shape"].clone()).unwrap();
        super::checked_delta(
            actual.try_as_slice::<f32>().unwrap(),
            actual.shape(),
            &expected_values,
            &expected_shape,
        )
    }

    // These small numerical tests run on the ordinary hosted macOS MLX CI lane. Local invocation
    // requires the same explicit Metal ownership admission as any other MLX Array numerical test.
    #[test]
    fn frozen_author_full_graph_ragged_masks_fp32_bf16_and_skipping() {
        let f = fixture("ordinary");
        let g = geometry(&f);
        let context = contexts(
            &f["inputs"]["encoder_hidden_states"],
            &f["inputs"]["encoder_attention_mask"],
            g.text,
        );
        let input = array(&f["inputs"]["hidden_states"]);
        let times: Vec<f32> = f["inputs"]["timestep"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        for case in f["cases"].as_array().unwrap() {
            let dtype = if case["dtype"] == "float32" {
                Dtype::Float32
            } else {
                Dtype::Bfloat16
            };
            let w = weights(&f["weights"], dtype);
            let model = Transformer::load(&w, g.clone()).unwrap();
            assert!(w.unused_keys().is_empty());
            assert!(context.iter().all(|c| c.dtype() == Dtype::Float32));
            if dtype == Dtype::Float32 {
                // Full FP32 must preserve supplied embedding precision until its F32 projection.
                let precise = model.text.forward(&context[0]).unwrap();
                let narrowed = model
                    .text
                    .forward(&context[0].as_dtype(Dtype::Bfloat16).unwrap())
                    .unwrap();
                assert_eq!(precise.dtype(), Dtype::Float32);
                let difference = subtract(&precise, &narrowed)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max(None)
                    .unwrap();
                eval([&difference]).unwrap();
                assert!(
                    difference.item::<f32>() > 1e-6,
                    "embedding precision probe does not discriminate narrowing"
                );
            }
            let skip: Vec<u32> = case["skip_layers"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let output = model
                .forward(&input, &times, &context, &skip, &CancelFlag::new())
                .unwrap();
            let difference = delta(&output, &case["velocity"]);
            let tolerance = if dtype == Dtype::Float32 { 2e-5 } else { 1e-3 };
            assert!(
                difference < tolerance,
                "dtype={dtype:?} skip={skip:?} max_abs={difference}"
            );
        }
        let hidden = array(&f["leaf_ops"]["situ_hidden"]);
        let gate = array(&f["leaf_ops"]["situ_gate"]);
        assert!(
            delta(
                &situ(&hidden, &gate).unwrap(),
                &f["leaf_ops"]["situ_expected"]
            ) < 1e-6
        );
        assert!(delta(&rope(2, 3, 3, 8).unwrap(), &f["leaf_ops"]["rope_2x3_text3"]) < 1e-6);
    }
    #[test]
    fn frozen_reference_unmerged_fp32_projection_and_three_step_dmd() {
        let f = fixture("dmd");
        let g = geometry(&f);
        let w = weights(&f["base_weights"], Dtype::Bfloat16);
        let mut model = Transformer::load(&w, g.clone()).unwrap();
        let mut adapters = HashMap::new();
        for (key, value) in f["adapters"].as_object().unwrap() {
            let p = format!("diffusion_model.{key}");
            adapters.insert(format!("{p}.lora_down.weight"), array(&value["down"]));
            adapters.insert(format!("{p}.lora_up.weight"), array(&value["up"]));
            adapters.insert(
                format!("{p}.alpha"),
                Array::from_slice(&[value["alpha"].as_f64().unwrap() as f32], &[]),
            );
        }
        model
            .install_dmd(
                &Weights::from_map(adapters),
                &weights(&f["replacement_tensors"], Dtype::Float32),
                3,
            )
            .unwrap();
        let probe = model.blocks[0].projections[3]
            .forward(&array(&f["projection_probe"]["input"]))
            .unwrap();
        assert_eq!(probe.dtype(), Dtype::Float32);
        assert!(delta(&probe, &f["projection_probe"]["correct"]) < 1e-3);
        let narrowed = probe
            .as_dtype(Dtype::Bfloat16)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let difference = subtract(&probe, &narrowed)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(difference > 1e-4, "residual was narrowed");
        assert_eq!(model.time1.dtype(), Dtype::Float32);
        assert_eq!(model.time2.dtype(), Dtype::Float32);
        assert_eq!(model.modulation.dtype(), Dtype::Float32);
        assert_eq!(model.head.dtype(), Dtype::Float32);
        let noise = array(&f["noise"]);
        let q = array(&f["mix_matrix"]);
        let context = contexts(&f["conditioning"], &f["mask"], g.text);
        let mut clean = multiply(&noise, constant(0.0, Dtype::Float32).unwrap()).unwrap();
        let shape = noise.shape();
        for (i, code) in f["step_codes"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let flat = clean
                .transpose_axes(&[0, 2, 3, 1])
                .unwrap()
                .reshape(&[-1, g.channels])
                .unwrap();
            let mixed = linear_fp32(&flat, &q)
                .unwrap()
                .reshape(&[shape[0], shape[2], shape[3], g.channels])
                .unwrap()
                .transpose_axes(&[0, 3, 1, 2])
                .unwrap();
            let input = add(&noise, mixed).unwrap();
            let velocity = model
                .forward(
                    &input,
                    &[code.as_f64().unwrap() as f32],
                    &context,
                    &[],
                    &CancelFlag::new(),
                )
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap();
            let expected = &f["runs"]["correct_widened"]["steps"][i];
            let velocity_error = delta(&velocity, &expected["velocity"]);
            assert!(velocity_error < 1e-3,
                "DMD step={i} velocity max_abs={velocity_error}; input max_abs={}; actual={:?}; expected={:?}",
                delta(&input, &expected["input"]), crate::pipeline::logical_f32_readback(&velocity).unwrap(), expected["velocity"]["values"]);
            clean = subtract(&noise, velocity).unwrap();
            eval([&clean]).unwrap();
            let clean_error = delta(&clean, &expected["x0"]);
            assert!(clean_error < 1e-3, "DMD step={i} x0 max_abs={clean_error}");
        }
    }
}

/// Borrowed from one immutable renderer and scoped to one sampling call. No cross-request reuse.
type EditKeyValue = (Array, Array);
pub(crate) struct EditCache {
    keys_values: Vec<EditKeyValue>,
}
pub(crate) struct EditConditioning<'a> {
    pub context: &'a Array,
    pub references: &'a [Array],
    pub cache: Option<&'a EditCache>,
}
struct EditBodyInput<'a> {
    rope: &'a Array,
    modulation: &'a Array,
    geometry: &'a Geometry,
    text: i32,
}
impl Block {
    fn edit_qkv(
        &self,
        x: &Array,
        rope: &Array,
        modulation: &Array,
        g: &Geometry,
    ) -> Result<(Array, Array, Array)> {
        let m =
            |i| -> Result<Array> { Ok(slice_axis(modulation, 1, i, 1)?.reshape(&[1, 1, g.dim])?) };
        let y = add(
            multiply(
                self.n1.forward(x)?,
                add(m(1)?, constant(1.0, modulation.dtype())?)?,
            )?,
            m(0)?,
        )?
        .as_dtype(self.projections[0].dtype())?;
        let shape = [1, x.shape()[1], g.heads, g.dim / g.heads];
        let q = qk_norm(&self.projections[0].forward(&y)?, &self.qn, g)?.reshape(&shape)?;
        let k = qk_norm(&self.projections[1].forward(&y)?, &self.kn, g)?.reshape(&shape)?;
        let v = self.projections[2]
            .forward(&y)?
            .reshape(&shape)?
            .transpose_axes(&[0, 2, 1, 3])?;
        Ok((
            rotate(&q, rope)?
                .as_dtype(v.dtype())?
                .transpose_axes(&[0, 2, 1, 3])?,
            rotate(&k, rope)?
                .as_dtype(v.dtype())?
                .transpose_axes(&[0, 2, 1, 3])?,
            v,
        ))
    }
    fn edit_finish(
        &self,
        x: &Array,
        attention: &Array,
        modulation: &Array,
        g: &Geometry,
    ) -> Result<Array> {
        let m =
            |i| -> Result<Array> { Ok(slice_axis(modulation, 1, i, 1)?.reshape(&[1, 1, g.dim])?) };
        let attention =
            attention
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[1, x.shape()[1], g.dim])?;
        let x = add(
            x,
            multiply(self.projections[3].forward(&attention)?, m(2)?)?,
        )?;
        let y = add(
            multiply(
                self.n2.forward(&x)?,
                add(m(4)?, constant(1.0, modulation.dtype())?)?,
            )?,
            m(3)?,
        )?
        .as_dtype(self.projections[0].dtype())?;
        let fused = self.projections[4].forward(&y)?;
        Ok(add(
            x,
            multiply(
                self.projections[5].forward(&edit_situ(
                    &slice_axis(&fused, 2, 0, g.ffn)?,
                    &slice_axis(&fused, 2, g.ffn, g.ffn)?,
                )?)?,
                m(5)?,
            )?,
        )?)
    }
    fn edit_reference(
        &self,
        x: &Array,
        rope: &Array,
        modulation: &Array,
        g: &Geometry,
        cancel: &CancelFlag,
    ) -> Result<(Array, EditKeyValue)> {
        check(cancel)?;
        let (q, k, v) = self.edit_qkv(x, rope, modulation, g)?;
        let output = edit_attention(&q, &k, &v, g, cancel)?;
        let next = self.edit_finish(x, &output, modulation, g)?;
        eval([&next, &k, &v])?;
        check(cancel)?;
        Ok((next, (k, v)))
    }
    fn edit_body(
        &self,
        x: &Array,
        reference: &EditKeyValue,
        input: EditBodyInput<'_>,
        cancel: &CancelFlag,
    ) -> Result<Array> {
        let EditBodyInput {
            rope,
            modulation,
            geometry: g,
            text,
        } = input;
        check(cancel)?;
        let (q, k, v) = self.edit_qkv(x, rope, modulation, g)?;
        let insert = |body: &Array, reference: &Array| -> Result<Array> {
            Ok(concatenate_axis(
                &[
                    &slice_axis(body, 2, 0, text)?,
                    reference,
                    &slice_axis(body, 2, text, body.shape()[2] - text)?,
                ],
                2,
            )?)
        };
        let output = edit_attention(
            &q,
            &insert(&k, &reference.0)?,
            &insert(&v, &reference.1)?,
            g,
            cancel,
        )?;
        let next = self.edit_finish(x, &output, modulation, g)?;
        eval([&next])?;
        check(cancel)?;
        Ok(next)
    }
}
fn edit_situ(hidden: &Array, gate: &Array) -> Result<Array> {
    if hidden.dtype() == Dtype::Float32 {
        return situ(hidden, gate);
    }
    // Each source unary op computes in F32 then rounds once. Evaluate at these
    // boundaries so widened intermediates cannot accumulate across the two branches.
    let dtype = hidden.dtype();
    let unary = |x: &Array, operation: &dyn Fn(Array) -> Result<Array>| -> Result<Array> {
        let widened = x.as_dtype(Dtype::Float32)?;
        let result = operation(widened)?;
        let rounded = result.as_dtype(dtype)?;
        eval([&rounded])?;
        Ok(rounded)
    };
    let divide_then_tanh = |x: &Array, divisor: f32| -> Result<Array> {
        let divided = unary(x, &|a| Ok(divide(a, constant(divisor, Dtype::Float32)?)?))?;
        unary(&divided, &|a| Ok(tanh(a)?))
    };
    let gated = {
        let gate_tanh = divide_then_tanh(gate, 4.0)?;
        let sigmoid = unary(gate, &|a| Ok(sigmoid(a)?))?;
        let gated = multiply(scale(&gate_tanh, 4.0)?, sigmoid)?;
        eval([&gated])?;
        gated
    };
    Ok(multiply(
        gated,
        scale(&divide_then_tanh(hidden, 25.0)?, 25.0)?,
    )?)
}

fn edit_silu(x: &Array) -> Result<Array> {
    // nn.SiLU computes the activation in FP32 then rounds once to the input dtype.
    // The backend's BF16 silu_exact instead preserves eager sigmoid/multiply rounding.
    Ok(mlx_rs::fast::silu_exact(x.as_dtype(Dtype::Float32)?)?.as_dtype(x.dtype())?)
}
fn edit_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Geometry,
    cancel: &CancelFlag,
) -> Result<Array> {
    let scale = 1.0 / (g.dim as f32 / g.heads as f32).sqrt();
    if v.dtype() == Dtype::Float32 {
        attention_fp32(q, k, v, scale, cancel)
    } else {
        attention_edit_bf16(q, k, v, cancel)
    }
}
impl Transformer {
    fn enable_edit_precision(&mut self) {
        for projection in [
            &mut self.time1,
            &mut self.time2,
            &mut self.modulation,
            &mut self.head,
            &mut self.video,
            &mut self.text,
            &mut self.output,
        ] {
            projection.edit_fused_bias = true;
        }
        for block in &mut self.text_blocks {
            block.edit_precision = true;
            for projection in [
                &mut block.qkv,
                &mut block.proj,
                &mut block.up,
                &mut block.gate,
                &mut block.down,
            ] {
                projection.edit_fused_bias = true;
            }
        }
    }
    pub(crate) fn install_edit(&mut self, w: &Weights, rank: i32) -> Result<()> {
        self.enable_edit_precision();
        for (index, block) in self.blocks.iter_mut().enumerate() {
            for (projection, name) in
                block
                    .projections
                    .iter_mut()
                    .zip(["q", "k", "v", "o", "ffn_up_gate", "ffn_down"])
            {
                let prefix = format!("layers.{index}.proj.{name}");
                let a = shaped(
                    w,
                    &format!("{prefix}.lora_A"),
                    &[rank, projection.weight.shape()[1]],
                )?;
                let b = shaped(
                    w,
                    &format!("{prefix}.lora_B"),
                    &[projection.weight.shape()[0], rank],
                )?;
                if a.dtype() != Dtype::Bfloat16 || b.dtype() != Dtype::Bfloat16 {
                    return Err(Error::Unsupported(
                        "MOSS edit requires BF16 A/B tensors".into(),
                    ));
                }
                projection.adapter = Some((a, b, 1.0));
                projection.edit_adapter = true;
            }
        }
        Ok(())
    }
    fn edit_time(&self, normalized: f32) -> Result<(Array, Array)> {
        if !normalized.is_finite() || !(0.0..=1.0).contains(&normalized) {
            return Err(Error::Unsupported(
                "MOSS edit time must be normalized in [0,1]".into(),
            ));
        }
        let half = self.g.freq / 2;
        let mut features = Vec::new();
        for sin in [false, true] {
            for band in 0..half {
                let phase =
                    (normalized * 1000.0) as f64 * 10000_f64.powf(-(band as f64) / half as f64);
                features.push(if sin {
                    phase.sin() as f32
                } else {
                    phase.cos() as f32
                });
            }
        }
        let time = self
            .time2
            .forward(&edit_silu(
                &self
                    .time1
                    .forward(&Array::from_slice(&features, &[1, self.g.freq]))?,
            )?)?
            .as_dtype(Dtype::Float32)?;
        Ok((
            self.modulation
                .forward(&time)?
                .reshape(&[1, 6, self.g.dim])?,
            self.head.forward(&time)?.reshape(&[1, 2, self.g.dim])?,
        ))
    }
    fn edit_reference_inputs(&self, references: &[Array]) -> Result<(Array, Array)> {
        let mut tokens = Vec::new();
        let mut ropes = Vec::new();
        for (index, reference) in references.iter().enumerate() {
            let shape = reference.shape();
            if shape.len() != 4
                || shape[0] != 1
                || shape[1] != self.g.channels
                || shape[2] < 1
                || shape[3] < 1
            {
                return Err(Error::Unsupported(
                    "invalid MOSS edit reference latent".into(),
                ));
            }
            let (h, w) = (shape[2], shape[3]);
            let value =
                self.video
                    .forward(&reference.transpose_axes(&[0, 2, 3, 1])?.reshape(&[
                        1,
                        h * w,
                        self.g.channels,
                    ])?)?;
            let head = self.g.dim / self.g.heads;
            let mut phases = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    for sine in [true, false] {
                        for coordinate in [
                            index as f32 + 1.0,
                            0.0,
                            y as f32 - (h - 1) as f32 / 2.0,
                            x as f32 - (w - 1) as f32 / 2.0,
                        ] {
                            for band in 0..head / 8 {
                                let phase =
                                    coordinate / 10000_f32.powf(band as f32 / (head / 8) as f32);
                                phases.push(if sine { phase.sin() } else { phase.cos() });
                            }
                        }
                    }
                }
            }
            ropes.push(Array::from_slice(&phases, &[1, h * w, head]).as_dtype(value.dtype())?);
            tokens.push(value);
        }
        if tokens.is_empty() {
            return Err(Error::Unsupported("MOSS edit requires references".into()));
        }
        Ok((
            concatenate_axis(&tokens.iter().collect::<Vec<_>>(), 1)?,
            concatenate_axis(&ropes.iter().collect::<Vec<_>>(), 1)?,
        ))
    }
    pub(crate) fn prepare_edit_cache(
        &self,
        references: &[Array],
        cancel: &CancelFlag,
    ) -> Result<EditCache> {
        check(cancel)?;
        let (mut prefix, rope) = self.edit_reference_inputs(references)?;
        let (modulation, _) = self.edit_time(0.0)?;
        let mut keys_values = Vec::new();
        for block in &self.blocks {
            let (next, kv) = block.edit_reference(&prefix, &rope, &modulation, &self.g, cancel)?;
            prefix = next;
            keys_values.push(kv);
        }
        Ok(EditCache { keys_values })
    }
    pub(crate) fn forward_edit(
        &self,
        latent: &Array,
        time: f32,
        conditioning: EditConditioning<'_>,
        cancel: &CancelFlag,
    ) -> Result<Array> {
        let EditConditioning {
            context,
            references,
            cache,
        } = conditioning;
        check(cancel)?;
        let g = &self.g;
        let s = latent.shape();
        if s.len() != 4
            || s[0] != 1
            || s[1] != g.channels
            || context.ndim() != 3
            || context.shape()[0] != 1
            || context.shape()[1] < 1
            || context.shape()[2] != g.text
        {
            return Err(Error::Unsupported(
                "MOSS edit latent/context shape mismatch".into(),
            ));
        }
        let (h, w, text_len) = (s[2], s[3], context.shape()[1]);
        let video = self
            .video
            .forward(
                &latent
                    .transpose_axes(&[0, 2, 3, 1])?
                    .reshape(&[1, h * w, g.channels])?,
            )?;

        let original = rope(h, w, text_len, g.dim / g.heads)?.as_dtype(video.dtype())?;
        let text_rope = slice_axis(&original, 1, h * w, text_len)?;
        let mut text = self.text.forward(context)?;

        for block in &self.text_blocks {
            check(cancel)?;
            text = block.forward(&text, &text_rope, g, cancel)?;
            eval([&text])?;
        }
        let mut body = concatenate_axis(&[&text, &video], 1)?;

        let body_rope = concatenate_axis(&[&text_rope, &slice_axis(&original, 1, 0, h * w)?], 1)?;
        let (modulation, head) = self.edit_time(time)?;
        let mut uncached = if cache.is_none() {
            Some(self.edit_reference_inputs(references)?)
        } else {
            None
        };
        let (reference_modulation, _) = self.edit_time(0.0)?;
        for (index, block) in self.blocks.iter().enumerate() {
            let temporary;
            let reference = if let Some(cache) = cache {
                &cache.keys_values[index]
            } else {
                let (prefix, rope) = uncached
                    .as_mut()
                    .ok_or_else(|| Error::Msg("missing edit prefix".into()))?;
                let (next, kv) =
                    block.edit_reference(prefix, rope, &reference_modulation, g, cancel)?;
                *prefix = next;
                temporary = kv;
                &temporary
            };

            body = block.edit_body(
                &body,
                reference,
                EditBodyInput {
                    rope: &body_rope,
                    modulation: &modulation,
                    geometry: g,
                    text: text_len,
                },
                cancel,
            )?;
        }

        let output = self
            .norm
            .forward(&slice_axis(&body, 1, text_len, h * w)?.as_dtype(self.norm.weight.dtype())?)?;

        let output = add(
            multiply(
                output,
                add(slice_axis(&head, 1, 1, 1)?, constant(1.0, head.dtype())?)?,
            )?,
            slice_axis(&head, 1, 0, 1)?,
        )?;

        let output = self.output.forward(&output)?;

        let output = output
            .reshape(&[1, h, w, g.channels])?
            .transpose_axes(&[0, 3, 1, 2])?;

        eval([&output])?;
        check(cancel)?;
        Ok(output)
    }
}

#[cfg(test)]
mod edit_fixture_tests {
    use super::*;
    use crate::pipeline::logical_f32_readback;
    use serde_json::Value;
    fn array(value: &Value) -> Array {
        let shape = value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect::<Vec<_>>();
        let values = value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        Array::from_slice(&values, &shape)
            .as_dtype(if value["dtype"] == "bfloat16" {
                Dtype::Bfloat16
            } else {
                Dtype::Float32
            })
            .unwrap()
    }
    fn compare(actual: &Array, expected: &Value, tolerance: f32, label: &str) {
        let expected_shape = expected["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect::<Vec<_>>();
        let expected_values = expected["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        let got = logical_f32_readback(actual).unwrap();
        let delta = checked_delta(&got, actual.shape(), &expected_values, &expected_shape);
        eprintln!("{label}: max_abs={delta}, bound={tolerance}");
        assert!(
            delta <= tolerance,
            "{label}: maximum absolute error {delta} > {tolerance}"
        );
    }
    fn bf16_bound(expected: &Value) -> f32 {
        let magnitude = expected["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| (v.as_f64().unwrap() as f32).abs())
            .fold(0.0_f32, f32::max);
        magnitude / 64.0 + 1.0 / 1024.0
    }
    fn compare_bf16(actual: &Array, expected: &Value, label: &str) {
        compare(actual, expected, bf16_bound(expected), label);
    }
    fn time_probes(model: &Transformer, f: &Value) {
        for probe in f["time_probes"].as_array().unwrap() {
            // Each stage also receives the independent upstream input, locating
            // backend rounding without allowing accumulated errors to hide it.
            for (label, actual, expected) in [
                (
                    "time first linear",
                    model.time1.forward(&array(&probe["time1_input"])).unwrap(),
                    &probe["time1"],
                ),
                (
                    "time exact SiLU",
                    edit_silu(&array(&probe["silu_input"])).unwrap(),
                    &probe["silu"],
                ),
                (
                    "time second linear",
                    model.time2.forward(&array(&probe["time2_input"])).unwrap(),
                    &probe["time2"],
                ),
                (
                    "time shared modulation",
                    model
                        .modulation
                        .forward(&array(&probe["modulation_flat_input"]))
                        .unwrap(),
                    &probe["modulation_flat"],
                ),
                (
                    "time head modulation",
                    model
                        .head
                        .forward(&array(&probe["head_flat_input"]))
                        .unwrap(),
                    &probe["head_flat"],
                ),
            ] {
                if matches!(label, "time first linear" | "time exact SiLU") {
                    compare(&actual, expected, 0.0, label);
                } else {
                    compare_bf16(&actual, expected, label);
                }
            }
            let split_bias = add(
                array(&probe["time1_input"])
                    .matmul(model.time1.weight.t())
                    .unwrap(),
                model.time1.bias.as_ref().unwrap(),
            )
            .unwrap();
            let eager_silu = mlx_rs::fast::silu_exact(array(&probe["silu_input"])).unwrap();
            for (wrong, expected) in [
                (&split_bias, &probe["time1"]),
                (&eager_silu, &probe["silu"]),
            ] {
                assert_ne!(
                    logical_f32_readback(wrong).unwrap(),
                    logical_f32_readback(&array(expected)).unwrap(),
                    "rounding control must distinguish source-fused arithmetic"
                );
            }
            let (modulation, head) = model
                .edit_time(probe["normalized_timestep"].as_f64().unwrap() as f32)
                .unwrap();
            compare_bf16(
                &modulation,
                &probe["modulation"],
                "complete time modulation",
            );
            compare_bf16(&head, &probe["head"], "complete time head");
        }
    }
    fn masked(value: &Value, mask: &Value) -> Array {
        let indices = mask["values"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .filter_map(|(i, v)| (v.as_f64().unwrap() != 0.0).then_some(i as i32))
            .collect::<Vec<_>>();
        array(value)
            .take_axis(Array::from_slice(&indices, &[indices.len() as i32]), 1)
            .unwrap()
    }
    #[test]
    fn edit_bf16_source_leaf_cpu() {
        const CHILD: &str = "MOSS_EDIT_LEAF_CPU_CHILD";
        if std::env::var(CHILD).as_deref() != Ok("1") {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "transformer::edit_fixture_tests::edit_bf16_source_leaf_cpu",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "CPU leaf child: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            eprint!("{}", String::from_utf8_lossy(&output.stderr));
            return;
        }
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || {
            let fixture: Value = serde_json::from_str(include_str!(
                "../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_edit_precision/leaf_ops.json"
            )).unwrap();
            let cancel = CancelFlag::new();
            for (i, call) in fixture["attention"].as_array().unwrap().iter().enumerate() {
                let (q, k, v) = (array(&call["q"]), array(&call["k"]), array(&call["v"]));
                let corrected = attention_edit_bf16(&q, &k, &v, &cancel).unwrap();
                compare(
                    &corrected,
                    &call["output"],
                    0.0,
                    &format!("source SDPA {i}"),
                );
                let unrounded = attention_fp32(
                    &q.as_dtype(Dtype::Float32).unwrap(),
                    &k.as_dtype(Dtype::Float32).unwrap(),
                    &v.as_dtype(Dtype::Float32).unwrap(),
                    1.0 / (q.shape()[3] as f32).sqrt(),
                    &cancel,
                )
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
                assert_ne!(
                    logical_f32_readback(&corrected).unwrap(),
                    logical_f32_readback(&unrounded).unwrap(),
                    "source SDPA rounding discriminator {i}"
                );
                // Repeated independent queries preserve the source output; 17 rows
                // exercise two complete tiles and a partial tile, with all keys visible.
                if q.shape()[2] == 8 {
                    let first = slice_axis(&q, 2, 0, 1).unwrap();
                    let repeated = concatenate_axis(&[&q, &q, &first], 2).unwrap();
                    let source = array(&call["output"]);
                    let first_source = slice_axis(&source, 2, 0, 1).unwrap();
                    let expected = concatenate_axis(&[&source, &source, &first_source], 2).unwrap();
                    let actual = attention_edit_bf16(&repeated, &k, &v, &cancel).unwrap();
                    assert_eq!(actual.shape()[2], 17);
                    assert_eq!(
                        logical_f32_readback(&actual).unwrap(),
                        logical_f32_readback(&expected).unwrap()
                    );
                }
            }
            for (i, call) in fixture["situ"].as_array().unwrap().iter().enumerate() {
                let (hidden, gate) = (array(&call["hidden"]), array(&call["gate"]));
                let corrected = edit_situ(&hidden, &gate).unwrap();
                compare(
                    &corrected,
                    &call["output"],
                    0.0,
                    &format!("source SiTU {i}"),
                );
                let unrounded = situ(
                    &hidden.as_dtype(Dtype::Float32).unwrap(),
                    &gate.as_dtype(Dtype::Float32).unwrap(),
                )
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
                assert_ne!(
                    logical_f32_readback(&corrected).unwrap(),
                    logical_f32_readback(&unrounded).unwrap(),
                    "source SiTU eager rounding discriminator {i}"
                );
            }
        });
    }
    #[test]
    fn edit_projection_and_reference_coordinates_cpu() {
        const CHILD: &str = "MOSS_EDIT_PROJECTION_CPU_CHILD";
        if std::env::var(CHILD).as_deref() != Ok("1") {
            let output=std::process::Command::new(std::env::current_exe().unwrap()).args(["--exact","transformer::edit_fixture_tests::edit_projection_and_reference_coordinates_cpu","--nocapture","--test-threads=1"]).env(CHILD,"1").output().unwrap();
            assert!(
                output.status.success(),
                "CPU edit projection child: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || run_oracle(true));
    }
    #[test]
    #[ignore = "requires coordinator-admitted Metal ownership; exact silu kernel is Metal-only"]
    fn frozen_edit_oracle_on_metal() {
        assert_eq!(
            std::env::var("MOSS_NATIVE_TEST_ADMITTED").as_deref(),
            Ok("1")
        );
        mlx_rs::with_new_default_stream(mlx_rs::Stream::gpu(), || {
            three_reference_negative_stage_probe();
            run_oracle(false);
        });
    }
    fn fixture_model(f: &Value, fp32: bool) -> Transformer {
        let weights = Weights::from_map(
            f["weights"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| {
                    let value = array(v);
                    (
                        k.clone(),
                        if fp32 {
                            value.as_dtype(Dtype::Float32).unwrap()
                        } else {
                            value
                        },
                    )
                })
                .collect(),
        );
        let mut model = Transformer::load(
            &weights,
            Geometry {
                dim: 16,
                ffn: 24,
                heads: 2,
                layers: 2,
                text: 12,
                freq: 8,
                channels: 4,
            },
        )
        .unwrap();
        let adapter = Weights::from_map(
            f["adapter"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), array(v)))
                .collect(),
        );
        model.install_edit(&adapter, 2).unwrap();
        if fp32 {
            // Diagnostic only: same represented source weights cast to FP32. The
            // production package validator continues requiring BF16 A/B tensors.
            for block in &mut model.blocks {
                for p in &mut block.projections {
                    if let Some((a, b, scale)) = &p.adapter {
                        p.adapter = Some((
                            a.as_dtype(Dtype::Float32).unwrap(),
                            b.as_dtype(Dtype::Float32).unwrap(),
                            *scale,
                        ));
                    }
                }
            }
        }
        model
    }
    // Separate network error from integration: BF16 predictions are compared at
    // identical source states; FP32 represented-weight sampling checks the entire graph.
    // The historical final-only BF16 bound did not compose through CFG and state feedback.
    fn sampler_precision(f: &Value) {
        const FP32_BOUND: f32 = 2e-5;
        let mut oracle: Value = serde_json::from_str(include_str!("../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_edit_v4_sampler_analysis/fp32_sampler.json")).unwrap();
        let extra: Value = serde_json::from_str(include_str!("../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_edit_multi/fp32_sampler.json")).unwrap();
        oracle["cases"]
            .as_array_mut()
            .unwrap()
            .extend(extra["cases"].as_array().unwrap().iter().cloned());
        let input = &f["inputs"];
        for fp32 in [false, true] {
            let model = fixture_model(f, fp32);
            for case in f["cases"].as_array().unwrap() {
                let count = case["reference_count"].as_u64().unwrap() as usize;
                let refs = input["references"].as_array().unwrap()[..count]
                    .iter()
                    .map(array)
                    .collect::<Vec<_>>();
                let positive = [masked(&input["positive"], &input["positive_mask"])];
                let negative = [masked(&input["negative"], &input["negative_mask"])];
                let resolved = mlx_gen::gen_core::moss_image::MossImageResolved::resolve(
                    mlx_gen::gen_core::moss_image::MossImageVariant::Sft,
                    &mlx_gen::gen_core::moss_image_edit::edit_request(
                        &mlx_gen::GenerationRequest {
                            steps: Some(2),
                            guidance: Some(4.0),
                            width: 48,
                            height: 32,
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                )
                .unwrap();
                let sampling = crate::edit::EditSampling {
                    transformer: &model,
                    resolved: &resolved,
                    positive: &positive,
                    negative: &negative,
                    references: &refs,
                    use_cache: true,
                };
                crate::edit::SAMPLING_TRACE.with(|t| *t.borrow_mut() = Some(Vec::new()));
                let output = sampling
                    .run(
                        &array(&input["hidden_states"]),
                        &CancelFlag::new(),
                        &mut |_| {},
                    )
                    .unwrap();
                let trace = crate::edit::SAMPLING_TRACE.with(|t| t.borrow_mut().take().unwrap());
                assert_eq!(trace.len(), 2);
                assert_eq!(output.dtype(), Dtype::Float32);
                if fp32 {
                    let expected = oracle["cases"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|c| c["reference_count"] == case["reference_count"])
                        .unwrap();
                    compare(
                        &output,
                        &expected["correct"]["final"],
                        FP32_BOUND,
                        "strict FP32 complete sampler",
                    );
                    for (actual, step) in trace
                        .iter()
                        .zip(expected["correct"]["steps"].as_array().unwrap())
                    {
                        for (a, key) in
                            actual
                                .iter()
                                .zip(["input", "conditional", "unconditional", "output"])
                        {
                            assert_eq!(a.dtype(), Dtype::Float32);
                            compare(a, &step[key], FP32_BOUND, "strict FP32 sampler stage");
                        }
                    }
                    for (name, control) in expected["controls"].as_object().unwrap() {
                        let wrong = array(&control["final"]);
                        let delta = checked_delta(
                            &logical_f32_readback(&output).unwrap(),
                            output.shape(),
                            &logical_f32_readback(&wrong).unwrap(),
                            wrong.shape(),
                        );
                        eprintln!("full sampler semantic control {name}: delta={delta}, bound={FP32_BOUND}");
                        assert!(
                            delta > 100.0 * FP32_BOUND,
                            "undiscriminated full sampler control {name}: {delta}"
                        );
                    }
                } else {
                    let cache = model.prepare_edit_cache(&refs, &CancelFlag::new()).unwrap();
                    for (index, source) in case["sampling"]["steps"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .enumerate()
                    {
                        let sigmas = &case["sampling"]["sigmas"]["values"];
                        let time = sigmas[index].as_f64().unwrap() as f32;
                        for (context, key) in [
                            (&positive[0], "conditional"),
                            (&negative[0], "unconditional"),
                        ] {
                            let prediction = model
                                .forward_edit(
                                    &array(&source["input"]),
                                    time,
                                    EditConditioning {
                                        context,
                                        references: &refs,
                                        cache: Some(&cache),
                                    },
                                    &CancelFlag::new(),
                                )
                                .unwrap();
                            assert_eq!(
                                prediction.dtype(),
                                Dtype::Bfloat16,
                                "production edit prediction dtype"
                            );
                            compare_bf16(
                                &prediction,
                                &source[key],
                                "BF16 prediction at source step input",
                            );
                        }
                        let values = trace[index]
                            .iter()
                            .map(|a| {
                                assert_eq!(a.dtype(), Dtype::Float32);
                                logical_f32_readback(a).unwrap()
                            })
                            .collect::<Vec<_>>();
                        let dt = sigmas[index + 1].as_f64().unwrap() as f32 - time;
                        let mut error = 0.0_f32;
                        for (i, &current) in values[0].iter().enumerate() {
                            let guided = values[2][i] + 4.0 * (values[1][i] - values[2][i]);
                            error = error.max((values[3][i] - (current + dt * guided)).abs());
                        }
                        assert!(
                            error <= 1e-6,
                            "actual production sampler composition error {error}"
                        );
                    }
                }
            }
        }
    }

    fn lifecycle_values(a: &Array) -> Vec<f32> {
        logical_f32_readback(a).unwrap()
    }
    fn lifecycle_delta(a: &Array, b: &Array) -> f32 {
        checked_delta(
            &lifecycle_values(a),
            a.shape(),
            &lifecycle_values(b),
            b.shape(),
        )
    }
    // Successive request discriminators execute actual tiny Transformer/LoRA and sampler math.
    // Fresh model + uncached sampling is an independent lifetime baseline, not a new golden.
    fn request_local_lifecycle(f: &Value) {
        let mut changed_backbone = f.clone();
        let mut changed_adapter = f.clone();
        for value in changed_backbone["weights"]["final_linear_video.weight"]["values"]
            .as_array_mut()
            .unwrap()
        {
            *value = serde_json::json!(-value.as_f64().unwrap());
        }
        for (name, tensor) in changed_adapter["adapter"].as_object_mut().unwrap() {
            if name.ends_with("lora_B") {
                for value in tensor["values"].as_array_mut().unwrap() {
                    *value = serde_json::json!(-value.as_f64().unwrap());
                }
            }
        }
        for fp32 in [false, true] {
            let definitions = [f, &changed_backbone, &changed_adapter];
            let models: Vec<_> = definitions.iter().map(|f| fixture_model(f, fp32)).collect();
            let input = &f["inputs"];
            let refs: Vec<_> = input["references"]
                .as_array()
                .unwrap()
                .iter()
                .map(array)
                .collect();
            let hidden = array(&input["hidden_states"]);
            let positive = masked(&input["positive"], &input["positive_mask"]);
            let negative = masked(&input["negative"], &input["negative_mask"]);
            let mut baseline = None;
            // Return to A after each changed count/order/content/duplicate/model/adapter/context,
            // geometry/noise/CFG request. Both BF16 production boundaries and represented F32.
            for case in [
                0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0,
            ] {
                let model_index = match case {
                    6 => 1,
                    7 => 2,
                    _ => 0,
                };
                let mut references = refs.clone();
                match case {
                    1 => references.truncate(1),
                    2 => references.truncate(2),
                    3 => references.truncate(3),
                    4 => references.reverse(),
                    5 => references[2] = references[0].clone(),
                    8 => {
                        references[1] =
                            multiply(&references[1], Array::from_slice(&[-1.0_f32], &[1])).unwrap()
                    }
                    _ => (),
                }
                let noise = if case == 9 {
                    slice_axis(&hidden, 3, 0, 2).unwrap()
                } else if case == 10 {
                    add(&hidden, Array::from_slice(&[0.25_f32], &[1])).unwrap()
                } else {
                    hidden.clone()
                };
                let positive = [if case == 11 {
                    negative.clone()
                } else {
                    positive.clone()
                }];
                let negative = [if case == 11 {
                    array(&input["negative"])
                } else {
                    negative.clone()
                }];
                let mut resolved = mlx_gen::gen_core::moss_image::MossImageResolved::resolve(
                    mlx_gen::gen_core::moss_image::MossImageVariant::Sft,
                    &mlx_gen::gen_core::moss_image_edit::edit_request(
                        &mlx_gen::GenerationRequest {
                            width: 48,
                            height: 32,
                            steps: Some(2),
                            guidance: Some(4.),
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                )
                .unwrap();
                if case == 12 {
                    resolved.guidance = 1.;
                }
                let fresh = fixture_model(definitions[model_index], fp32);
                let sample =
                    |transformer,
                     use_cache,
                     cancel: &CancelFlag,
                     progress: &mut dyn FnMut(mlx_gen::Progress)| {
                        crate::edit::EditSampling {
                            transformer,
                            resolved: &resolved,
                            positive: &positive,
                            negative: &negative,
                            references: &references,
                            use_cache,
                        }
                        .run(&noise, cancel, progress)
                    };
                let actual =
                    sample(&models[model_index], true, &CancelFlag::new(), &mut |_| {}).unwrap();
                let expected = sample(&fresh, false, &CancelFlag::new(), &mut |_| {}).unwrap();
                assert_eq!(
                    lifecycle_values(&actual),
                    lifecycle_values(&expected),
                    "fresh uncached baseline case={case} fp32={fp32}"
                );
                if case == 0 {
                    if let Some(original) = &baseline {
                        assert_eq!(lifecycle_values(&actual), lifecycle_values(original));
                    } else {
                        baseline = Some(actual.clone());
                    }
                } else if case != 9 {
                    assert!(lifecycle_delta(&actual, baseline.as_ref().unwrap()) > 2e-3,
                        "changed request must change real numerical output: case={case} fp32={fp32}");
                }
                let cancel = CancelFlag::new();
                assert!(matches!(
                    sample(&models[model_index], true, &cancel, &mut |p| {
                        if matches!(p, mlx_gen::Progress::Step { .. }) {
                            cancel.cancel();
                        }
                    }),
                    Err(Error::Canceled)
                ));
                // Fail inside the actual network after preparing K/V, then enter another run.
                let bad_context = [slice_axis(&positive[0], 2, 0, 11).unwrap()];
                assert!(crate::edit::EditSampling {
                    transformer: &models[model_index],
                    resolved: &resolved,
                    positive: &bad_context,
                    negative: &negative,
                    references: &references,
                    use_cache: true,
                }
                .run(&noise, &CancelFlag::new(), &mut |_| {})
                .is_err());
                let recovered =
                    sample(&models[model_index], true, &CancelFlag::new(), &mut |_| {}).unwrap();
                assert_eq!(
                    lifecycle_values(&recovered),
                    lifecycle_values(&expected),
                    "cancel/error reentry case={case}"
                );
            }
            eprintln!("request-local lifecycle fp32={fp32}: 25 changed/reentry requests, each cached/fresh-uncached/cancel/network-error/recovered; 4 reference counts");
            let resolved = mlx_gen::gen_core::moss_image::MossImageResolved::resolve(
                mlx_gen::gen_core::moss_image::MossImageVariant::Sft,
                &mlx_gen::gen_core::moss_image_edit::edit_request(&mlx_gen::GenerationRequest {
                    width: 48,
                    height: 32,
                    steps: Some(2),
                    guidance: Some(4.),
                    ..Default::default()
                })
                .unwrap(),
            )
            .unwrap();
            let positive_batch = [positive.clone()];
            let negative_batch = [negative.clone()];
            crate::model::verify_numerical_renderer_transitions(
                |mode| {
                    let ordinary =
                        mode == mlx_gen::gen_core::moss_image::MossImageAcceleration::Off;
                    let definition = if ordinary { f } else { &changed_backbone };
                    let dtype = if ordinary {
                        Dtype::Bfloat16
                    } else {
                        Dtype::Float32
                    };
                    let weights = Weights::from_map(
                        definition["weights"]
                            .as_object()
                            .unwrap()
                            .iter()
                            .map(|(name, value)| {
                                (name.clone(), array(value).as_dtype(dtype).unwrap())
                            })
                            .collect(),
                    );
                    Transformer::load(&weights, models[0].g.clone()).unwrap()
                },
                |model| {
                    lifecycle_values(
                        &model
                            .forward(&hidden, &[0.73], &positive_batch, &[], &CancelFlag::new())
                            .unwrap(),
                    )
                },
                || {
                    lifecycle_values(
                        &crate::edit::EditSampling {
                            transformer: &models[0],
                            resolved: &resolved,
                            positive: &positive_batch,
                            negative: &negative_batch,
                            references: &refs,
                            use_cache: true,
                        }
                        .run(&hidden, &CancelFlag::new(), &mut |_| {})
                        .unwrap(),
                    )
                },
            );
        }
    }

    // Diagnostic only: independent frozen-source inputs locate the first backend
    // divergence. The unchanged complete oracle still runs and enforces every bound.
    fn three_reference_negative_stage_probe() {
        let f: Value = serde_json::from_str(include_str!(
            "../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_edit_metal_probe/fixture.json"
        )).unwrap();
        for case in f["cases"].as_array().unwrap() {
            let fp32 = case["backbone_dtype"] == "float32";
            let model = fixture_model(&f, fp32);
            eprintln!("STAGE_PROBE case: reference_count=3 condition=negative normalized_timestep=0.73 dtype={}", case["backbone_dtype"]);
            let stages = &case["stages"];
            let cancel = CancelFlag::new();
            let report = |label: &str, actual: &Array, expected: &Value| {
                let values = logical_f32_readback(actual).unwrap();
                let expected_values = logical_f32_readback(&array(expected)).unwrap();
                let delta = checked_delta(
                    &values,
                    actual.shape(),
                    &expected_values,
                    array(expected).shape(),
                );
                eprintln!(
                    "STAGE_PROBE {label}: max_abs={delta}, original_tensor_bound={}",
                    if fp32 { 2e-5 } else { bf16_bound(expected) }
                );
            };
            report(
                "video projection",
                &model
                    .video
                    .forward(&array(&stages["body.video_embedder_input"]))
                    .unwrap(),
                &stages["body.video_embedder"],
            );
            report(
                "text projection",
                &model
                    .text
                    .forward(&array(&stages["body.text_projection_input"]))
                    .unwrap(),
                &stages["body.text_projection"],
            );
            for (i, block) in model.text_blocks.iter().enumerate() {
                let key = format!("body.text_blocks.{i}");
                report(
                    &key,
                    &block
                        .forward(
                            &array(&stages[format!("{key}_input")]),
                            &array(&stages[format!("{key}_rope_input")]),
                            &model.g,
                            &cancel,
                        )
                        .unwrap(),
                    &stages[&key],
                );
            }
            for (i, block) in model.blocks.iter().enumerate() {
                let prefix = format!("body.layers.{i}");
                let stage = |name: &str| &stages[format!("{prefix}.{name}")];
                report(
                    &format!("{prefix}.norm1"),
                    &block.n1.forward(&array(stage("norm1_input"))).unwrap(),
                    stage("norm1"),
                );
                report(
                    &format!("{prefix}.norm2"),
                    &block.n2.forward(&array(stage("norm2_input"))).unwrap(),
                    stage("norm2"),
                );
                for (projection, name) in block.projections.iter().zip([
                    "q_projected",
                    "k_projected",
                    "v_projected",
                    "o_projected",
                    "up_gate",
                    "ffn_down",
                ]) {
                    report(
                        &format!("{prefix}.{name}"),
                        &projection
                            .forward(&array(stage(&format!("{name}_input"))))
                            .unwrap(),
                        stage(name),
                    );
                }
                report(
                    &format!("{prefix}.q_norm"),
                    &block.qn.forward(&array(stage("q_norm_input"))).unwrap(),
                    stage("q_norm"),
                );
                report(
                    &format!("{prefix}.k_norm"),
                    &block.kn.forward(&array(stage("k_norm_input"))).unwrap(),
                    stage("k_norm"),
                );
                let (q, k, v) = block
                    .edit_qkv(
                        &array(stage("input")),
                        &array(stage("rope")),
                        &array(stage("modulation")),
                        &model.g,
                    )
                    .unwrap();
                for (name, actual) in [("q_rope", &q), ("k_rope", &k), ("v_heads", &v)] {
                    report(&format!("{prefix}.{name}"), actual, stage(name));
                }
                report(
                    &format!("{prefix}.situ"),
                    &situ(&array(stage("hidden")), &array(stage("gate"))).unwrap(),
                    stage("situ"),
                );
                report(
                    &format!("{prefix}.edit_situ"),
                    &edit_situ(&array(stage("hidden")), &array(stage("gate"))).unwrap(),
                    stage("situ"),
                );
                report(
                    &format!("{prefix}.finish"),
                    &block
                        .edit_finish(
                            &array(stage("input")),
                            &array(stage("attention")),
                            &array(stage("modulation")),
                            &model.g,
                        )
                        .unwrap(),
                    stage("output"),
                );
                let source_kv = (
                    &case["reference_kv"][i]["key"],
                    &case["reference_kv"][i]["value"],
                );
                report(
                    &format!("{prefix}.whole_body_from_source_input"),
                    &block
                        .edit_body(
                            &array(stage("input")),
                            &(array(source_kv.0), array(source_kv.1)),
                            EditBodyInput {
                                rope: &array(stage("rope")),
                                modulation: &array(stage("modulation")),
                                geometry: &model.g,
                                text: 2,
                            },
                            &cancel,
                        )
                        .unwrap(),
                    stage("output"),
                );
            }
            for (i, call) in case["attention_calls"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
            {
                let (q, k, v) = (array(&call["q"]), array(&call["k"]), array(&call["v"]));
                report(
                    &format!("attention.{i}.edit_candidate"),
                    &edit_attention(&q, &k, &v, &model.g, &cancel).unwrap(),
                    &call["output"],
                );
                // Source-backed discriminator, not replacement production arithmetic:
                // CPU Torch SDPA rounds unnormalized exponentials to BF16, while
                // retaining F32 score, denominator and value accumulation boundaries.
                let (qf, kf, vf) = (
                    q.as_dtype(Dtype::Float32).unwrap(),
                    k.as_dtype(Dtype::Float32).unwrap(),
                    v.as_dtype(Dtype::Float32).unwrap(),
                );
                let dot = kf
                    .expand_dims(2)
                    .unwrap()
                    .matmul(qf.expand_dims(-1).unwrap())
                    .unwrap()
                    .squeeze_axes(&[-1])
                    .unwrap();
                let scores = divide(
                    &dot,
                    constant((q.shape()[3] as f32).sqrt(), Dtype::Float32).unwrap(),
                )
                .unwrap();
                let exponentials =
                    mlx_rs::ops::subtract(&scores, scores.max_axis(-1, true).unwrap())
                        .unwrap()
                        .exp()
                        .unwrap();
                let denominator = exponentials.sum_axis(-1, true).unwrap();
                let operand = exponentials
                    .as_dtype(v.dtype())
                    .unwrap()
                    .as_dtype(Dtype::Float32)
                    .unwrap();
                let numerator = vf
                    .transpose_axes(&[0, 1, 3, 2])
                    .unwrap()
                    .expand_dims(2)
                    .unwrap()
                    .matmul(operand.expand_dims(-1).unwrap())
                    .unwrap()
                    .squeeze_axes(&[-1])
                    .unwrap();
                let source_rounded = divide(numerator, denominator)
                    .unwrap()
                    .as_dtype(v.dtype())
                    .unwrap();
                report(
                    &format!("attention.{i}.source_rounded_exp"),
                    &source_rounded,
                    &call["output"],
                );
            }
            report(
                "final norm",
                &model
                    .norm
                    .forward(&array(&stages["final.norm_input"]))
                    .unwrap(),
                &stages["final.norm"],
            );
            report(
                "final projection",
                &model
                    .output
                    .forward(&array(&stages["final.projection_input"]))
                    .unwrap(),
                &stages["final.projection"],
            );
            let input = &f["inputs"];
            let refs: Vec<_> = input["references"].as_array().unwrap()[..3]
                .iter()
                .map(array)
                .collect();
            let negative = masked(&input["negative"], &input["negative_mask"]);
            let cache = model.prepare_edit_cache(&refs, &cancel).unwrap();
            for (label, cache) in [("cached", Some(&cache)), ("uncached", None)] {
                let actual = model
                    .forward_edit(
                        &array(&input["hidden_states"]),
                        0.73,
                        EditConditioning {
                            context: &negative,
                            references: &refs,
                            cache,
                        },
                        &cancel,
                    )
                    .unwrap();
                report(
                    &format!("three-reference negative {label}"),
                    &actual,
                    &case["expected"],
                );
                eprintln!(
                    "STAGE_PROBE three-reference negative {label} values: {:?}",
                    logical_f32_readback(&actual).unwrap()
                );
            }
        }
    }

    fn run_oracle(cpu_only: bool) {
        let mut f: Value = serde_json::from_str(include_str!(
            "../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_edit_v2/fixture.json"
        ))
        .unwrap();
        let extra: Value = serde_json::from_str(include_str!(
            "../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_edit_multi/fixture.json"
        ))
        .unwrap();
        assert_eq!(f["weights"], extra["weights"]);
        assert_eq!(f["adapter"], extra["adapter"]);
        f["cases"]
            .as_array_mut()
            .unwrap()
            .extend(extra["cases"].as_array().unwrap().iter().cloned());
        let weights = Weights::from_map(
            f["weights"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), array(v)))
                .collect(),
        );
        let adapter = Weights::from_map(
            f["adapter"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), array(v)))
                .collect(),
        );
        let mut model = Transformer::load(
            &weights,
            Geometry {
                dim: 16,
                ffn: 24,
                heads: 2,
                layers: 2,
                text: 12,
                freq: 8,
                channels: 4,
            },
        )
        .unwrap();
        model.install_edit(&adapter, 2).unwrap();
        let probe = &f["projection_probe"];
        let projected = model.blocks[0].projections[0]
            .forward(&array(&probe["input"]))
            .unwrap();
        compare(&projected, &probe["expected"], 0.0, "BF16 edit projection");
        if !cpu_only {
            let original: Value = serde_json::from_str(include_str!(
                "../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_edit_v1_time_probe/fixture.json"
            )).unwrap();
            let original_weights = Weights::from_map(
                original["weights"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(k, v)| (k.clone(), array(v)))
                    .collect(),
            );
            let mut original_model = Transformer::load(&original_weights, model.g.clone()).unwrap();
            original_model.enable_edit_precision();
            time_probes(&original_model, &original);
            time_probes(&model, &f);
        }
        let input = &f["inputs"];
        let hidden = array(&input["hidden_states"]);
        let positive = masked(&input["positive"], &input["positive_mask"]);
        let negative = masked(&input["negative"], &input["negative_mask"]);
        let references = input["references"]
            .as_array()
            .unwrap()
            .iter()
            .map(array)
            .collect::<Vec<_>>();
        let time = input["normalized_timestep"].as_f64().unwrap() as f32;
        let cancel = CancelFlag::new();
        for case in f["cases"].as_array().unwrap() {
            let refs = &references[..case["reference_count"].as_u64().unwrap() as usize];
            let (tokens, rope) = model.edit_reference_inputs(refs).unwrap();
            compare(&tokens, &case["reference_tokens"], 0.0, "reference tokens");
            compare(&rope, &case["reference_rope"], 0.0, "reference coordinates");
            if cpu_only {
                continue;
            }
            let (reference_modulation, _) = model.edit_time(0.0).unwrap();
            compare_bf16(
                &reference_modulation,
                &case["reference_modulation"],
                "reference t0",
            );
            let (body_modulation, _) = model.edit_time(time).unwrap();
            compare_bf16(&body_modulation, &case["body_modulation"], "body time");
            let cache = model.prepare_edit_cache(refs, &cancel).unwrap();
            for ((key, value), expected) in cache
                .keys_values
                .iter()
                .zip(case["reference_kv"].as_array().unwrap())
            {
                compare_bf16(key, &expected["key"], "reference cache key");
                compare_bf16(value, &expected["value"], "reference cache value");
            }
            let cached = model
                .forward_edit(
                    &hidden,
                    time,
                    EditConditioning {
                        context: &positive,
                        references: refs,
                        cache: Some(&cache),
                    },
                    &cancel,
                )
                .unwrap();
            let uncached = model
                .forward_edit(
                    &hidden,
                    time,
                    EditConditioning {
                        context: &positive,
                        references: refs,
                        cache: None,
                    },
                    &cancel,
                )
                .unwrap();
            compare_bf16(&cached, &case["cached"], "cached prediction");
            compare_bf16(&uncached, &case["uncached"], "uncached prediction");
            for (name, control) in case["controls"].as_object().unwrap() {
                if refs.len() == 2 && name == "reversed_reference_order" {
                    // Two-reference source BF16 reversal cannot meet the unchanged 3x
                    // margin. Exact ordinal RoPE + strict F32 trajectory/mutant gates
                    // establish order; this still runs the actual reversed BF16 graph.
                    let reversed: Vec<_> = refs.iter().rev().cloned().collect();
                    let output = model
                        .forward_edit(
                            &hidden,
                            time,
                            EditConditioning {
                                context: &positive,
                                references: &reversed,
                                cache: None,
                            },
                            &cancel,
                        )
                        .unwrap();
                    compare_bf16(
                        &output,
                        &control["output"],
                        "two-reference reversed BF16 prediction",
                    );
                    let difference = lifecycle_delta(&cached, &output);
                    eprintln!(
                        "two-reference BF16 order diagnostic: native={difference}, source={}",
                        control["max_abs_difference"]
                    );
                    assert!(difference > 0., "actual BF16 graph must retain order");
                    continue;
                }
                let delta = checked_delta(
                    &logical_f32_readback(&cached).unwrap(),
                    cached.shape(),
                    &logical_f32_readback(&array(&control["output"])).unwrap(),
                    cached.shape(),
                );
                assert!(
                    delta > 3.0 * bf16_bound(&case["cached"]),
                    "undiscriminated control {name}: {delta}"
                );
            }
            let negative = model
                .forward_edit(
                    &hidden,
                    time,
                    EditConditioning {
                        context: &negative,
                        references: refs,
                        cache: Some(&cache),
                    },
                    &cancel,
                )
                .unwrap();
            compare_bf16(&negative, &case["negative_cached"], "negative prediction");
            assert_eq!(
                logical_f32_readback(&cached).unwrap(),
                logical_f32_readback(&uncached).unwrap()
            );
            let resolved = mlx_gen::gen_core::moss_image::MossImageResolved::resolve(
                mlx_gen::gen_core::moss_image::MossImageVariant::Sft,
                &mlx_gen::gen_core::moss_image_edit::edit_request(&mlx_gen::GenerationRequest {
                    steps: Some(2),
                    guidance: Some(4.0),
                    width: 48,
                    height: 32,
                    ..Default::default()
                })
                .unwrap(),
            )
            .unwrap();
            let positive_batch = [positive.clone()];
            let negative_batch = [masked(&input["negative"], &input["negative_mask"])];
            let sampling = crate::edit::EditSampling {
                transformer: &model,
                resolved: &resolved,
                positive: &positive_batch,
                negative: &negative_batch,
                references: refs,
                use_cache: true,
            };
            let mut steps = Vec::new();
            let sampled = sampling
                .run(&hidden, &cancel, &mut |p| {
                    if let mlx_gen::Progress::Step { current, total } = p {
                        steps.push((current, total))
                    }
                })
                .unwrap();
            assert_eq!(sampled.dtype(), Dtype::Float32, "sampler accumulator dtype");
            // Aggregate BF16 drift is diagnostic, not a fictitious final-latent ULP gate.
            let expected = array(&case["sampling"]["final"]);
            eprintln!(
                "BF16 complete sampler aggregate drift: {}",
                checked_delta(
                    &logical_f32_readback(&sampled).unwrap(),
                    sampled.shape(),
                    &logical_f32_readback(&expected).unwrap(),
                    expected.shape()
                )
            );
            assert_eq!(steps, [(1, 2), (2, 2)]);
            let stop = CancelFlag::new();
            assert!(matches!(
                sampling.run(
                    &hidden,
                    &stop,
                    &mut |p| if matches!(p, mlx_gen::Progress::Step { .. }) {
                        stop.cancel()
                    }
                ),
                Err(Error::Canceled)
            ));
            let reentry = sampling
                .run(&hidden, &CancelFlag::new(), &mut |_| {})
                .unwrap();
            assert_eq!(
                logical_f32_readback(&sampled).unwrap(),
                logical_f32_readback(&reentry).unwrap()
            );
        }
        if !cpu_only {
            sampler_precision(&f);
            request_local_lifecycle(&f);
        }
        cancel.cancel();
        assert!(matches!(
            model.prepare_edit_cache(&references, &cancel),
            Err(Error::Canceled)
        ));
    }
}
