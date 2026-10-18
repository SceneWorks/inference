//! Frozen MOSS graph. No FLUX DiT/scheduler or ordinary cast-down adapter abstraction is used.
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
        })
    }
    fn dtype(&self) -> Dtype {
        self.weight.dtype()
    }
    pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
        let cast = x.as_dtype(self.weight.dtype())?;
        let base = if let Some(b) = &self.bias {
            mlx_gen::nn::linear(&cast, &self.weight, b)?
        } else {
            cast.matmul(self.weight.t())?
        };
        if let Some((down, up, scale)) = &self.adapter {
            // The original input's FP32 values feed the residual; only the backbone is narrowed.
            let residual = x
                .as_dtype(Dtype::Float32)?
                .matmul(down.t())?
                .matmul(up.t())?;
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
fn attention(q: &Array, k: &Array, v: &Array, rope: &Array, g: &Geometry) -> Result<Array> {
    let shape = [q.shape()[0], q.shape()[1], g.heads, g.dim / g.heads];
    let q = rotate(&q.reshape(&shape)?, rope)?
        .as_dtype(v.dtype())?
        .transpose_axes(&[0, 2, 1, 3])?;
    let k = rotate(&k.reshape(&shape)?, rope)?
        .as_dtype(v.dtype())?
        .transpose_axes(&[0, 2, 1, 3])?;
    let v = v.reshape(&shape)?.transpose_axes(&[0, 2, 1, 3])?;
    let out = mlx_rs::fast::scaled_dot_product_attention(
        &q,
        &k,
        &v,
        1.0 / (g.dim as f32 / g.heads as f32).sqrt(),
        None,
        None,
    )?;
    Ok(out
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[shape[0], shape[1], g.dim])?)
}
struct TextBlock {
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
    fn forward(&self, x: &Array, rope: &Array, g: &Geometry) -> Result<Array> {
        let qkv = self.qkv.forward(&self.n1.forward(x)?)?;
        let q = qk_norm(&slice_axis(&qkv, 2, 0, g.dim)?, &self.qn, g)?;
        let k = qk_norm(&slice_axis(&qkv, 2, g.dim, g.dim)?, &self.kn, g)?;
        let v = slice_axis(&qkv, 2, 2 * g.dim, g.dim)?;
        let x = add(x, self.proj.forward(&attention(&q, &k, &v, rope, g)?)?)?;
        let y = self.n2.forward(&x)?;
        Ok(add(
            &x,
            self.down
                .forward(&situ(&self.up.forward(&y)?, &self.gate.forward(&y)?)?)?,
        )?)
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
    fn forward(&self, x: &Array, rope: &Array, modulation: &Array, g: &Geometry) -> Result<Array> {
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
        let a = self.projections[3].forward(&attention(&q, &k, &v, rope, g)?)?;
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
                text = block.forward(&text, &text_rope, g)?;
                eval([&text])?;
                check(cancel)?;
            }
            let mut x = concatenate_axis(&[&video, &text], 1)?;
            for (i, block) in self.blocks.iter().enumerate() {
                check(cancel)?;
                if !skip.contains(&(i as u32)) {
                    x = block.forward(&x, &rope, &modulation, g)?;
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
            let mixed = clean
                .transpose_axes(&[0, 2, 3, 1])
                .unwrap()
                .reshape(&[-1, g.channels])
                .unwrap()
                .matmul(q.t())
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
