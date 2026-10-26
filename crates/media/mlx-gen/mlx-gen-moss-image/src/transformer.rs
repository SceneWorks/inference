//! Frozen MOSS graph. No FLUX DiT/scheduler or ordinary cast-down adapter abstraction is used.
use crate::adapters::Lokr;
use crate::precision::{attention_edit_bf16, attention_fp32, linear_fp32};
use mlx_gen::adapters::{AdaptableHost, AdaptableLinear, Adapter};
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
#[derive(Clone)]
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
#[derive(Clone)]
pub(crate) struct Projection {
    weight: Array,
    bias: Option<Array>,
    adapter: Option<(Array, Array, f32)>,
    // Adapter routing only; its shared base handle is never used or mutated by this graph.
    custom: AdaptableLinear,
    lokr: Vec<Lokr>,
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
            lokr: Vec::new(),
            custom: AdaptableLinear::dense(
                shaped(w, &format!("{p}.weight"), &[output, input])?,
                None,
            ),
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
        let mut base = base;
        if let Some((down, up, scale)) = &self.adapter {
            if self.edit_adapter {
                let residual = if down.dtype() == Dtype::Float32 {
                    // Synthetic FP32 diagnostics must not enter the NAX TF32 GEMM path.
                    linear_fp32(&linear_fp32(&x.as_dtype(Dtype::Float32)?, down)?, up)?
                } else {
                    x.as_dtype(down.dtype())?.matmul(down.t())?.matmul(up.t())?
                };
                base = add(
                    &base,
                    self::scale(&residual, *scale)?.as_dtype(base.dtype())?,
                )?;
            } else {
                // The original input's FP32 values feed the residual; only the backbone is narrowed.
                let residual = linear_fp32(&linear_fp32(&x.as_dtype(Dtype::Float32)?, down)?, up)?;
                base = add(
                    base.as_dtype(Dtype::Float32)?,
                    self::scale(&residual, *scale)?,
                )?;
            }
        }
        for adapter in self.custom.adapters() {
            if adapter.is_disabled() {
                continue;
            }
            let input = x.as_dtype(Dtype::Float32)?;
            // Match training/reload FP32 residual arithmetic, not generic host narrowing or TF32.
            // The supplied DMD residual remains an independent frozen stack.
            let residual = match adapter {
                Adapter::Lora { a, b, scale } => {
                    self::scale(&linear_fp32(&linear_fp32(&input, &a.t())?, &b.t())?, *scale)?
                }
                Adapter::Lokr { delta, scale } => {
                    self::scale(&linear_fp32(&input, delta)?, *scale)?
                }
                Adapter::LokrStructured { .. } => {
                    return Err(Error::Unsupported(
                        "MOSS custom adapters require dense FP32 LoKr reconstruction".into(),
                    ))
                }
            };
            base = add(base.as_dtype(Dtype::Float32)?, residual)?;
        }
        for adapter in &self.lokr {
            if let Some(residual) = adapter.residual(x)? {
                base = add(base.as_dtype(Dtype::Float32)?, residual)?;
            }
        }
        Ok(base)
    }
    fn cast(&mut self, dtype: Dtype) -> Result<()> {
        self.weight = self.weight.as_dtype(dtype)?;
        self.bias = self.bias.as_ref().map(|b| b.as_dtype(dtype)).transpose()?;
        // Cast before installing trainable factors; keep the host's shared handle current so
        // FP32 training does not retain a second idle BF16 copy of every backbone projection.
        self.custom = AdaptableLinear::dense(self.weight.clone(), None);
        Ok(())
    }
    fn sync_routing_base(&mut self) {
        // Routing and execution share one frozen allocation, including DMD replacements.
        let adapters = self.custom.adapters().to_vec();
        self.custom = AdaptableLinear::dense(self.weight.clone(), None);
        self.custom.set_adapters(adapters);
    }
    fn replace_weight_bias(&mut self, weight: Array, bias: Option<Array>) {
        self.weight = weight;
        self.bias = bias;
        self.sync_routing_base();
    }
    fn widen(&mut self) -> Result<()> {
        self.weight = self.weight.as_dtype(Dtype::Float32)?;
        self.bias = self
            .bias
            .as_ref()
            .map(|b| b.as_dtype(Dtype::Float32))
            .transpose()?;
        self.sync_routing_base();
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
#[allow(clippy::too_many_arguments)]
fn attention(
    q: &Array,
    k: &Array,
    v: &Array,
    rope: &Array,
    g: &Geometry,
    cancel: &CancelFlag,
    training: bool,
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
    let value_dtype = v.dtype();
    // CPU BF16 fast SDPA and explicit FP32 SDPA do not round identically. Use
    // the same differentiable computation for CPU deployment and training, then
    // restore the value dtype at their shared O-projection boundary. The Metal
    // inference fast path below is deliberately unchanged.
    let cpu_bf16 =
        value_dtype == Dtype::Bfloat16 && mlx_rs::Device::try_default()? == mlx_rs::Device::cpu();
    let out = if edit {
        edit_attention(&q, &k, &v, g, cancel)?
    } else if training || cpu_bf16 {
        // Explicit differentiable SDPA: no evaluation of traced adapter activations.
        let q = q.as_dtype(Dtype::Float32)?;
        let k = k.as_dtype(Dtype::Float32)?;
        let v = v.as_dtype(Dtype::Float32)?;
        let scores = k
            .expand_dims(2)?
            .matmul(&self::scale(&q, scale)?.expand_dims(-1)?)?
            .squeeze_axes(&[-1])?;
        let p = mlx_rs::ops::softmax_axis(&scores, -1, true)?;
        v.transpose_axes(&[0, 1, 3, 2])?
            .expand_dims(2)?
            .matmul(&p.expand_dims(-1)?)?
            .squeeze_axes(&[-1])?
            .as_dtype(value_dtype)?
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
    fn forward(
        &self,
        x: &Array,
        rope: &Array,
        g: &Geometry,
        cancel: &CancelFlag,
        training: bool,
    ) -> Result<Array> {
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
                training,
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
#[derive(Clone)]
struct Block {
    n1: Norm,
    n2: Norm,
    qn: Norm,
    kn: Norm,
    projections: Vec<Projection>,
}
#[cfg(test)]
static TRAINING_BLOCK_FORWARDS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
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
    fn forward_checkpointed(
        &self,
        x: &Array,
        rope: &Array,
        modulation: &Array,
        g: &Geometry,
        cancel: &CancelFlag,
    ) -> Result<Array> {
        let mut inputs = vec![x.clone(), rope.clone(), modulation.clone()];
        for projection in &self.projections {
            for adapter in projection.custom.adapters() {
                match adapter {
                    Adapter::Lora { a, b, .. } => inputs.extend([a.clone(), b.clone()]),
                    _ => {
                        return Err(Error::Unsupported(
                            "MOSS checkpoints require raw LoKr factors, never dense deltas".into(),
                        ))
                    }
                }
            }
            for lokr in &projection.lokr {
                inputs.extend(lokr.arrays().into_iter().cloned());
            }
        }
        let mut segment = mlx_rs::transforms::checkpoint(|arrays: &[Array]| {
            let mut block = self.clone();
            let mut cursor = 3;
            for projection in &mut block.projections {
                let mut adapters = Vec::new();
                for adapter in projection.custom.adapters() {
                    match adapter {
                        Adapter::Lora { scale, .. } => {
                            adapters.push(Adapter::Lora {
                                a: arrays[cursor].clone(),
                                b: arrays[cursor + 1].clone(),
                                scale: *scale,
                            });
                            cursor += 2;
                        }
                        _ => {
                            return Err(mlx_rs::error::Exception::custom(
                                "dense LoKr entered MOSS checkpoint",
                            ))
                        }
                    }
                }
                projection.custom.set_training_adapters(adapters);
                projection.lokr = projection
                    .lokr
                    .iter()
                    .map(|lokr| lokr.rebound(arrays, &mut cursor))
                    .collect();
            }
            if cursor != arrays.len() {
                return Err(mlx_rs::error::Exception::custom(
                    "MOSS raw factor checkpoint arity mismatch",
                ));
            }
            Ok(vec![block
                .forward(&arrays[0], &arrays[1], &arrays[2], g, cancel, true)
                .map_err(|e| {
                    mlx_rs::error::Exception::custom(e.to_string())
                })?])
        });
        let mut output = segment(&inputs)?;
        if output.len() != 1 {
            return Err(Error::Msg(
                "MOSS checkpoint returned wrong output count".into(),
            ));
        }
        Ok(output.remove(0))
    }
    fn forward(
        &self,
        x: &Array,
        rope: &Array,
        modulation: &Array,
        g: &Geometry,
        cancel: &CancelFlag,
        training: bool,
    ) -> Result<Array> {
        #[cfg(test)]
        if training {
            TRAINING_BLOCK_FORWARDS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
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
        let a = self.projections[3]
            .forward(&attention(&q, &k, &v, rope, g, cancel, training, false)?)?;
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
    pub(crate) training: bool,
    pub(crate) training_checkpoint: bool,
    pub(crate) custom_reports: Vec<mlx_gen::gen_core::AdapterApplyReport>,
    pub(crate) custom_owned_bytes: u64,
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
            training: false,
            training_checkpoint: false,
            custom_reports: Vec::new(),
            custom_owned_bytes: 0,
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
    fn projection_mut(&mut self, path: &[&str]) -> Option<&mut Projection> {
        match path {
            ["layers", index, "proj", name] => {
                let index = index.parse::<usize>().ok()?;
                let projection = ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"]
                    .iter()
                    .position(|n| n == name)?;
                self.blocks.get_mut(index)?.projections.get_mut(projection)
            }
            _ => None,
        }
    }
    pub(crate) fn push_lokr(&mut self, path: &str, lokr: Lokr) -> Result<()> {
        let parts: Vec<_> = path.split('.').collect();
        self.projection_mut(&parts)
            .ok_or_else(|| Error::Unsupported(format!("unknown MOSS LoKr projection {path}")))?
            .lokr
            .push(lokr);
        Ok(())
    }
    pub(crate) fn install_training_adapter(
        &mut self,
        adapter: &mlx_gen::train::lora::TrainAdapter,
        params: &mlx_gen::train::lora::LoraParams,
        alpha: f32,
        rank: f32,
    ) -> Result<()> {
        if !rank.is_finite() || rank <= 0. || !alpha.is_finite() {
            return Err(Error::Unsupported(
                "MOSS training adapter scaling is invalid".into(),
            ));
        }
        match adapter {
            mlx_gen::train::lora::TrainAdapter::Lora { .. } => {
                for path in adapter.paths() {
                    let parts: Vec<_> = path.split('.').collect();
                    self.projection_mut(&parts)
                        .ok_or_else(|| Error::Msg("MOSS training path disappeared".into()))?
                        .lokr
                        .clear();
                }
                adapter.install(self, params, alpha, rank, Dtype::Float32)?;
            }
            mlx_gen::train::lora::TrainAdapter::Lokr { .. } => {
                for path in adapter.paths() {
                    let parts: Vec<_> = path.split('.').collect();
                    let projection = self
                        .projection_mut(&parts)
                        .ok_or_else(|| Error::Msg("MOSS training path disappeared".into()))?;
                    let mut group = std::collections::BTreeMap::new();
                    for suffix in mlx_gen::gen_core::weightsmeta::LOKR_SUFFIXES {
                        if let Some(array) = params.get(format!("{path}{suffix}").as_str()) {
                            group.insert(suffix.trim_start_matches('.').to_owned(), array.clone());
                        }
                    }
                    let lokr = Lokr::from_group(&group, projection.weight.shape(), alpha / rank)?;
                    projection.custom.set_training_adapters(Vec::new());
                    projection.lokr = vec![lokr];
                }
            }
        }
        Ok(())
    }
    pub(crate) fn materialize_custom(&mut self) -> Result<()> {
        let mut bytes = 0u64;
        for block in &self.blocks {
            for projection in &block.projections {
                projection.custom.materialize_adapters()?;
                for adapter in projection.custom.adapters() {
                    crate::adapters::validate_lora(adapter)?;
                    if let Adapter::Lora { a, b, .. } = adapter {
                        bytes = bytes
                            .checked_add((a.nbytes() + b.nbytes()) as u64)
                            .ok_or_else(|| Error::Msg("MOSS adapter residency overflow".into()))?;
                    }
                }
                for lokr in &projection.lokr {
                    lokr.materialize_and_validate()?;
                    for array in lokr.arrays() {
                        bytes = bytes
                            .checked_add(array.nbytes() as u64)
                            .ok_or_else(|| Error::Msg("MOSS adapter residency overflow".into()))?;
                    }
                }
            }
        }
        self.custom_owned_bytes = bytes;
        Ok(())
    }
    #[cfg(test)]
    fn raw_lokr_bytes(&self) -> usize {
        self.blocks
            .iter()
            .flat_map(|b| &b.projections)
            .flat_map(|p| &p.lokr)
            .flat_map(|l| l.arrays())
            .map(|a| a.nbytes())
            .sum()
    }
    #[cfg(test)]
    pub(crate) fn cast_training_weights(&mut self, dtype: Dtype) -> Result<()> {
        self.cast_training_weights_with_cancel(dtype, &CancelFlag::new())
    }
    pub(crate) fn cast_training_weights_with_cancel(
        &mut self,
        dtype: Dtype,
        cancel: &CancelFlag,
    ) -> Result<()> {
        check(cancel)?;
        for p in [
            &mut self.video,
            &mut self.text,
            &mut self.time1,
            &mut self.time2,
            &mut self.modulation,
            &mut self.head,
            &mut self.output,
        ] {
            check(cancel)?;
            p.cast(dtype)?;
        }
        self.norm.weight = self.norm.weight.as_dtype(dtype)?;
        for block in &mut self.text_blocks {
            for norm in [&mut block.n1, &mut block.n2, &mut block.qn, &mut block.kn] {
                check(cancel)?;
                norm.weight = norm.weight.as_dtype(dtype)?;
            }
            for p in [
                &mut block.qkv,
                &mut block.proj,
                &mut block.up,
                &mut block.gate,
                &mut block.down,
            ] {
                check(cancel)?;
                p.cast(dtype)?;
            }
        }
        for block in &mut self.blocks {
            for norm in [&mut block.n1, &mut block.n2, &mut block.qn, &mut block.kn] {
                check(cancel)?;
                norm.weight = norm.weight.as_dtype(dtype)?;
            }
            for p in &mut block.projections {
                check(cancel)?;
                p.cast(dtype)?;
            }
        }
        self.materialize_owned(cancel)
    }
    fn owned_arrays(&self) -> Vec<&Array> {
        self.weight_arrays(true)
    }
    #[cfg(feature = "test-support")]
    pub(crate) fn frozen_training_arrays(&self) -> Vec<Array> {
        self.weight_arrays(false).into_iter().cloned().collect()
    }
    fn weight_arrays(&self, include_adapter: bool) -> Vec<&Array> {
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
            if include_adapter {
                if let Some((down, up, _)) = &projection.adapter {
                    arrays.extend([down, up]);
                }
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
            .saturating_add(self.custom_owned_bytes)
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
            let weight = shaped(
                replacements,
                &format!("{prefix}.weight"),
                projection.weight.shape(),
            )?;
            let bias = projection
                .bias
                .as_ref()
                .ok_or_else(|| Error::Msg("missing time bias".into()))?;
            let bias = Some(shaped(
                replacements,
                &format!("{prefix}.bias"),
                bias.shape(),
            )?);
            projection.replace_weight_bias(weight, bias);
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
                .forward(&if self.training
                    || mlx_rs::Device::try_default()? == mlx_rs::Device::cpu()
                {
                    multiply(&time, sigmoid(&time)?)?
                } else {
                    mlx_rs::fast::silu_exact(&time)?
                })?
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
                text = block.forward(&text, &text_rope, g, cancel, self.training)?;
                if !self.training {
                    eval([&text])?;
                }
                check(cancel)?;
            }
            let mut x = concatenate_axis(&[&video, &text], 1)?;
            for (i, block) in self.blocks.iter().enumerate() {
                check(cancel)?;
                if !skip.contains(&(i as u32)) {
                    x = if self.training_checkpoint {
                        block.forward_checkpointed(&x, &rope, &modulation, g, cancel)?
                    } else {
                        block.forward(&x, &rope, &modulation, g, cancel, self.training)?
                    };
                    if !self.training {
                        eval([&x])?;
                    }
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
            if !self.training {
                eval([&out])?;
            }
            outputs.push(out);
        }
        check(cancel)?;
        let refs: Vec<_> = outputs.iter().collect();
        Ok(concatenate_axis(&refs, 0)?)
    }
}

impl AdaptableHost for Transformer {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        self.projection_mut(path).map(|p| &mut p.custom)
    }
    fn adaptable_paths(&self) -> Vec<String> {
        (0..self.blocks.len())
            .flat_map(|i| {
                ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"]
                    .into_iter()
                    .map(move |name| format!("layers.{i}.proj.{name}"))
            })
            .collect()
    }
    fn lycoris_dtype(&self) -> Dtype {
        Dtype::Float32
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
    pub(super) fn fixture(name: &str) -> serde_json::Value {
        serde_json::from_str(match name{"ordinary"=>include_str!("../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_author/fixture.json"),_=>include_str!("../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_author/dmd_fixture.json")}).unwrap()
    }
    pub(super) fn array(v: &serde_json::Value) -> Array {
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
    pub(super) fn geometry(v: &serde_json::Value) -> Geometry {
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
    pub(super) fn weights(v: &serde_json::Value, dtype: Dtype) -> Weights {
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
    pub(super) fn contexts(
        v: &serde_json::Value,
        mask: &serde_json::Value,
        text: i32,
    ) -> Vec<Array> {
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

    // These original default-Metal numerical proofs run only on the explicit hosted
    // moss-metal-fixture profile. Their device contract is not converted to CPU.
    #[test]
    #[ignore = "manual MOSS Metal model proof; CI lanes=moss-metal-fixture"]
    fn frozen_author_full_graph_ragged_masks_fp32_bf16_and_skipping() {
        #[cfg(feature = "test-support")]
        crate::test_support::assert_default_metal_profile_device();
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
    #[ignore = "manual MOSS Metal model proof; CI lanes=moss-metal-fixture"]
    fn frozen_reference_unmerged_fp32_projection_and_three_step_dmd() {
        #[cfg(feature = "test-support")]
        crate::test_support::assert_default_metal_profile_device();
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
            text = block.forward(&text, &text_rope, g, cancel, false)?;
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
            run_oracle(false);
            trained_custom_adapters_compose_with_edit();
        });
    }
    // Reached only through the existing ignored Metal edit oracle. Exercise actual learning,
    // portable export and the production custom-before-edit installation order for both methods.
    fn trained_custom_adapters_compose_with_edit() {
        use mlx_gen::gen_core::{moss_image::MossImageVariant, moss_image_training as contract};
        use mlx_gen::train::lora::{build_lokr_targets, build_lora_targets, TrainAdapter};
        use mlx_gen::{NetworkType, TrainingConfig};
        let f: Value = serde_json::from_str(include_str!(
            "../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_edit_v2/fixture.json"
        ))
        .unwrap();
        let backbone = |fp32| {
            let weights = Weights::from_map(
                f["weights"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(key, value)| {
                        let value = array(value);
                        (
                            key.clone(),
                            if fp32 {
                                value.as_dtype(Dtype::Float32).unwrap()
                            } else {
                                value
                            },
                        )
                    })
                    .collect(),
            );
            Transformer::load(
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
            .unwrap()
        };
        let edit_weights = || {
            Weights::from_map(
                f["adapter"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(key, value)| (key.clone(), array(value)))
                    .collect(),
            )
        };
        let input = &f["inputs"];
        let x = array(&input["hidden_states"]);
        let context = masked(&input["positive"], &input["positive_mask"]);
        let references: Vec<_> = input["references"]
            .as_array()
            .unwrap()
            .iter()
            .map(array)
            .collect();
        let noise = add(&x, Array::from_f32(0.37)).unwrap();
        let cancel = CancelFlag::new();
        let predict = |model: &Transformer, cached: bool| {
            let cache =
                cached.then(|| model.prepare_edit_cache(&references[..1], &cancel).unwrap());
            logical_f32_readback(
                &model
                    .forward_edit(
                        &x,
                        0.37,
                        EditConditioning {
                            context: &context,
                            references: &references[..1],
                            cache: cache.as_ref(),
                        },
                        &cancel,
                    )
                    .unwrap(),
            )
            .unwrap()
        };
        let mut edit_only = backbone(false);
        edit_only.install_edit(&edit_weights(), 2).unwrap();
        let baseline = predict(&edit_only, true);
        let paths = vec!["layers.0.proj.o".to_owned()];
        for network in [NetworkType::Lora, NetworkType::Lokr] {
            let mut trained = backbone(true);
            let (adapter, mut params) = match network {
                NetworkType::Lora => {
                    let (targets, params) =
                        build_lora_targets(&mut trained, &paths, 2, 17).unwrap();
                    (TrainAdapter::Lora { targets }, params)
                }
                NetworkType::Lokr => {
                    let (targets, params) =
                        build_lokr_targets(&mut trained, &paths, 2, 2, 17).unwrap();
                    (TrainAdapter::Lokr { targets }, params)
                }
            };
            let config = TrainingConfig {
                rank: 2,
                alpha: 3.,
                network_type: network,
                decompose_factor: 2,
                ..Default::default()
            };
            let mut optimizer = mlx_gen::TrainOptimizer::from_config("adamw", 0.1, 0.0).unwrap();
            for update in 0..3 {
                let (loss, gradient) = crate::training::loss_grads(
                    &mut trained,
                    &params,
                    &adapter,
                    &config,
                    &x,
                    &context,
                    &noise,
                    0.37,
                    None,
                    &cancel,
                )
                .unwrap();
                assert!(loss.is_finite());
                assert!(gradient
                    .values()
                    .flat_map(|array| logical_f32_readback(array).unwrap())
                    .any(|value| value != 0.));
                mlx_gen::train::lora::adapter_optimizer_update(
                    &mut optimizer,
                    &mut params,
                    &gradient,
                    &config,
                    update,
                    17,
                )
                .unwrap();
            }
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join("trained.safetensors");
            let metadata = contract::metadata(MossImageVariant::Sft, contract::SAMPLER);
            let extra: Vec<_> = metadata
                .iter()
                .map(|(key, value)| (key.as_str(), value.as_str()))
                .collect();
            crate::adapters::save_validated(
                &mut trained,
                &adapter,
                &params,
                &config,
                &extra,
                &path,
            )
            .unwrap();
            let kind = match network {
                NetworkType::Lora => mlx_gen::runtime::AdapterKind::Lora,
                NetworkType::Lokr => mlx_gen::runtime::AdapterKind::Lokr,
            };
            for strength in [0., 1., -1.] {
                let mut model = backbone(false);
                let spec = mlx_gen::runtime::AdapterSpec::new(path.clone(), strength, kind);
                let pin = mlx_gen::gen_core::PinnedWeightsFile::pin(&path).unwrap();
                let reports = crate::adapters::apply_pinned(
                    &mut model,
                    MossImageVariant::Sft,
                    &[spec],
                    &[pin],
                )
                .unwrap();
                assert_eq!(reports[0].applied, 1);
                model.install_edit(&edit_weights(), 2).unwrap();
                let output = predict(&model, true);
                assert!(output.iter().all(|value| value.is_finite()));
                assert_eq!(
                    output,
                    predict(&model, false),
                    "custom adapters changed cached/uncached edit equivalence"
                );
                if strength == 0. {
                    assert_eq!(
                        output, baseline,
                        "disabled custom adapter changed edit arithmetic"
                    );
                } else {
                    assert_ne!(output, baseline, "trained {network:?} residual disappeared during edit at strength {strength}");
                }
            }
        }
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

#[cfg(test)]
mod training_projection_tests {
    use super::*;
    use mlx_gen::adapters::{AdaptableLinear, Adapter};
    #[test]
    #[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
    fn custom_stack_is_fp32_scaled_and_keeps_dmd_separate() {
        let _device = crate::cpu_test_device();
        let weight = Array::from_slice(&[1f32, 0., 0., 1.], &[2, 2]);
        let mut p = Projection {
            lokr: Vec::new(),
            weight: weight.clone(),
            bias: None,
            adapter: Some((
                Array::from_slice(&[1f32, 2.], &[1, 2]),
                Array::from_slice(&[3f32, 4.], &[2, 1]),
                0.5,
            )),
            custom: AdaptableLinear::dense(weight, None),
            edit_adapter: false,
            edit_fused_bias: false,
        };
        let x = Array::from_slice(&[2f32, 1.], &[1, 2]);
        let bare_dmd = p.forward(&x).unwrap();
        p.custom.set_training_adapters(vec![Adapter::Lora {
            a: Array::from_slice(&[2f32, -1.], &[2, 1]),
            b: Array::from_slice(&[0.25f32, -0.5], &[1, 2]),
            scale: 2.,
        }]);
        let both = p.forward(&x).unwrap();
        assert_eq!(
            crate::pipeline::logical_f32_readback(&bare_dmd).unwrap(),
            vec![8., 9.]
        );
        assert_eq!(
            crate::pipeline::logical_f32_readback(&both).unwrap(),
            vec![9.5, 6.]
        );
        p.custom.set_adapters(Vec::new());
        assert_eq!(
            crate::pipeline::logical_f32_readback(&p.forward(&x).unwrap()).unwrap(),
            vec![8., 9.]
        );
        assert_eq!(
            crate::pipeline::logical_f32_readback(&p.weight).unwrap(),
            vec![1., 0., 0., 1.]
        );
        p.adapter = None;
        p.weight = p.weight.as_dtype(Dtype::Bfloat16).unwrap();
        p.custom.set_adapters(vec![Adapter::Lora {
            a: Array::from_slice(&[2f32, -1.], &[2, 1]),
            b: Array::from_slice(&[1f32, 1.], &[1, 2]),
            scale: 0.,
        }]);
        assert_eq!(
            p.forward(&x).unwrap().dtype(),
            Dtype::Bfloat16,
            "disabled adapter widened a BF16 backbone"
        );
    }
}

#[cfg(test)]
mod training_graph_tests {
    use super::author_fixture_tests::*;
    use super::*;
    use mlx_gen::gen_core::{moss_image::MossImageVariant, moss_image_training as contract};
    use mlx_gen::train::lora::{build_lokr_targets, build_lora_targets, TrainAdapter};
    use mlx_gen::{NetworkType, TrainingConfig};
    fn read(a: &Array) -> Vec<f32> {
        crate::pipeline::logical_f32_readback(a).unwrap()
    }
    #[test]
    #[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
    fn cpu_native_backward_updates_only_factors_and_reload_changes_generation() {
        let _device = crate::cpu_test_device();
        let f = fixture("ordinary");
        let g = geometry(&f);
        let x = slice_axis(&array(&f["inputs"]["hidden_states"]), 0, 0, 1).unwrap();
        let contexts = contexts(
            &f["inputs"]["encoder_hidden_states"],
            &f["inputs"]["encoder_attention_mask"],
            g.text,
        );
        let noise = mlx_rs::ops::add(&x, Array::from_f32(0.37)).unwrap();
        let cancel = CancelFlag::new();
        for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
            for network in [NetworkType::Lora, NetworkType::Lokr] {
                let mut model =
                    Transformer::load(&weights(&f["weights"], Dtype::Float32), g.clone()).unwrap();
                let frozen: Vec<_> = model.owned_arrays().iter().map(|a| read(a)).collect();
                let paths =
                    contract::target_paths(g.layers, &["q".into(), "ffn_up_gate".into()]).unwrap();
                assert_eq!(paths.len(), g.layers * 2);
                assert!(model.adaptable_mut(&["text_embedder", "proj"]).is_none());
                let (adapter, mut params) = match network {
                    NetworkType::Lora => {
                        let (t, p) = build_lora_targets(&mut model, &paths, 2, 17).unwrap();
                        (TrainAdapter::Lora { targets: t }, p)
                    }
                    NetworkType::Lokr => {
                        let (t, p) = build_lokr_targets(&mut model, &paths, 2, 2, 17).unwrap();
                        (TrainAdapter::Lokr { targets: t }, p)
                    }
                };
                let cfg = TrainingConfig {
                    rank: 2,
                    alpha: 3.,
                    network_type: network,
                    decompose_factor: 2,
                    ..Default::default()
                };
                let baseline = model
                    .forward(
                        &x,
                        &[370.],
                        std::slice::from_ref(&contexts[0]),
                        &[],
                        &cancel,
                    )
                    .unwrap();
                model
                    .install_training_adapter(&adapter, &params, 3., 2.)
                    .unwrap();
                assert_eq!(
                    read(&baseline),
                    read(
                        &model
                            .forward(
                                &x,
                                &[370.],
                                std::slice::from_ref(&contexts[0]),
                                &[],
                                &cancel
                            )
                            .unwrap()
                    ),
                    "initial adapter must be identity"
                );
                TRAINING_BLOCK_FORWARDS.store(0, std::sync::atomic::Ordering::SeqCst);
                let (loss, grads) = crate::training::loss_grads(
                    &mut model,
                    &params,
                    &adapter,
                    &cfg,
                    &x,
                    &contexts[0],
                    &noise,
                    0.37,
                    None,
                    &cancel,
                )
                .unwrap();
                let dense_calls = TRAINING_BLOCK_FORWARDS.load(std::sync::atomic::Ordering::SeqCst);
                assert!(loss.is_finite());
                let norm: f32 = grads.values().flat_map(read).map(|v| v * v).sum();
                assert!(
                    norm.is_finite() && norm > 1e-12,
                    "backward detached adapter leaves: {norm}"
                );
                // A nonzero gradient selected from the actual adapter map, independently checked
                // against symmetric differences of the production loss (not a toy residual).
                let (key, gradient) = grads
                    .iter()
                    .max_by(|(_, a), (_, b)| {
                        let max = |a: &Array| read(a).into_iter().map(f32::abs).fold(0., f32::max);
                        max(a).total_cmp(&max(b))
                    })
                    .unwrap();
                let gv = read(gradient);
                let index = gv
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
                    .unwrap()
                    .0;
                let epsilon = 0.01;
                let mut evaluate = |sign: f32| {
                    let mut p = params.clone();
                    let mut v = read(&p[key]);
                    v[index] += sign * epsilon;
                    p.insert(key.clone(), Array::from_slice(&v, p[key].shape()));
                    crate::training::loss_grads(
                        &mut model,
                        &p,
                        &adapter,
                        &cfg,
                        &x,
                        &contexts[0],
                        &noise,
                        0.37,
                        None,
                        &cancel,
                    )
                    .unwrap()
                    .0
                };
                let fd = (evaluate(1.) - evaluate(-1.)) / (2. * epsilon);
                assert!(
                    (fd - gv[index]).abs() < 1e-6 + gv[index].abs() * 0.02,
                    "finite difference {fd} vs backward {}",
                    gv[index]
                );
                let cfg_checkpoint = TrainingConfig {
                    gradient_checkpointing: true,
                    ..cfg.clone()
                };
                TRAINING_BLOCK_FORWARDS.store(0, std::sync::atomic::Ordering::SeqCst);
                let (recomputed, rg) = crate::training::loss_grads(
                    &mut model,
                    &params,
                    &adapter,
                    &cfg_checkpoint,
                    &x,
                    &contexts[0],
                    &noise,
                    0.37,
                    None,
                    &cancel,
                )
                .unwrap();
                assert!(
                    TRAINING_BLOCK_FORWARDS.load(std::sync::atomic::Ordering::SeqCst) > dense_calls,
                    "gradient checkpoint toggle did not recompute blocks"
                );
                assert!((loss - recomputed).abs() < 1e-6);
                for (k, v) in &grads {
                    assert!(read(v)
                        .iter()
                        .zip(read(&rg[k]))
                        .all(|(a, b)| (a - b).abs() < 1e-5));
                }
                let mut optimizer =
                    mlx_gen::TrainOptimizer::from_config("adamw", 0.01, 0.0).unwrap();
                for update in 0..3 {
                    let (_, grad) = crate::training::loss_grads(
                        &mut model,
                        &params,
                        &adapter,
                        &cfg,
                        &x,
                        &contexts[0],
                        &noise,
                        0.37,
                        None,
                        &cancel,
                    )
                    .unwrap();
                    mlx_gen::train::lora::adapter_optimizer_update(
                        &mut optimizer,
                        &mut params,
                        &grad,
                        &cfg,
                        update,
                        17,
                    )
                    .unwrap();
                }
                assert_eq!(
                    frozen,
                    model
                        .owned_arrays()
                        .iter()
                        .map(|a| read(a))
                        .collect::<Vec<_>>(),
                    "base/conditioning weights mutated"
                );
                let temp = tempfile::tempdir().unwrap();
                let path = temp.path().join("adapter.safetensors");
                let meta = contract::metadata(variant, contract::SAMPLER);
                let extra: Vec<_> = meta.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
                adapter
                    .save_with_meta(&params, 3., 2., 2, "", &extra, &path)
                    .unwrap();
                model
                    .install_training_adapter(&adapter, &params, 3., 2.)
                    .unwrap();
                let trained = model
                    .forward(
                        &x,
                        &[370.],
                        std::slice::from_ref(&contexts[0]),
                        &[],
                        &cancel,
                    )
                    .unwrap();
                let mut reloaded =
                    Transformer::load(&weights(&f["weights"], Dtype::Float32), g.clone()).unwrap();
                let kind = match network {
                    NetworkType::Lora => mlx_gen::runtime::AdapterKind::Lora,
                    NetworkType::Lokr => mlx_gen::runtime::AdapterKind::Lokr,
                };
                let spec = mlx_gen::runtime::AdapterSpec::new(path, 1., kind);
                let reports =
                    crate::adapters::apply(&mut reloaded, variant, std::slice::from_ref(&spec))
                        .unwrap();
                assert_eq!(reports[0].applied, paths.len());
                let got = reloaded
                    .forward(
                        &x,
                        &[370.],
                        std::slice::from_ref(&contexts[0]),
                        &[],
                        &cancel,
                    )
                    .unwrap();
                assert!(
                    read(&got)
                        .iter()
                        .zip(read(&trained))
                        .all(|(a, b)| (a - b).abs() < 2e-5),
                    "trained/reloaded projection mismatch"
                );
                assert!(
                    read(&got)
                        .iter()
                        .zip(read(&baseline))
                        .any(|(a, b)| (a - b).abs() > 1e-5),
                    "adapter did not affect generation {variant:?} {network:?}"
                );
                let mut disabled =
                    Transformer::load(&weights(&f["weights"], Dtype::Float32), g.clone()).unwrap();
                let mut zero = spec.clone();
                zero.scale = 0.;
                crate::adapters::apply(&mut disabled, variant, &[zero]).unwrap();
                let disabled_output = disabled
                    .forward(
                        &x,
                        &[370.],
                        std::slice::from_ref(&contexts[0]),
                        &[],
                        &cancel,
                    )
                    .unwrap();
                assert_eq!(
                    read(&disabled_output),
                    read(&baseline),
                    "scale zero changed the backbone"
                );
                let (tensors, metadata) =
                    Array::load_safetensors_with_metadata(&spec.path).unwrap();
                let first = tensors.keys().next().unwrap().clone();
                for defect in ["nan", "extra", "bf16", "missing", "shape"] {
                    let mut corrupt = tensors.clone();
                    match defect {
                        "nan" => {
                            corrupt.insert(
                                first.clone(),
                                multiply(&corrupt[&first], Array::from_f32(f32::NAN)).unwrap(),
                            );
                        }
                        "extra" => {
                            corrupt
                                .insert("layers.0.proj.q.unrecognized".into(), Array::from_f32(1.));
                        }
                        "bf16" => {
                            corrupt.insert(
                                first.clone(),
                                corrupt[&first].as_dtype(Dtype::Bfloat16).unwrap(),
                            );
                        }
                        "shape" => {
                            let key = corrupt
                                .keys()
                                .find(|k| k.ends_with("lora_A.weight") || k.ends_with("lokr_w1"))
                                .expect("required factor")
                                .clone();
                            corrupt.insert(key, Array::zeros::<f32>(&[1, 1]).unwrap());
                        }
                        "missing" => {
                            let key = corrupt
                                .keys()
                                .find(|k| {
                                    k.ends_with("lora_B.weight")
                                        || k.ends_with("lokr_w2_b")
                                        || k.ends_with("lokr_w2")
                                })
                                .expect("required factor")
                                .clone();
                            corrupt.remove(&key);
                        }
                        _ => unreachable!(),
                    }
                    let bad_path = temp.path().join(format!("{defect}.safetensors"));
                    Array::save_safetensors(
                        corrupt.iter().map(|(k, v)| (k.as_str(), v)),
                        Some(&metadata),
                        &bad_path,
                    )
                    .unwrap();
                    let mut bad = spec.clone();
                    bad.path = bad_path;
                    let mut fresh =
                        Transformer::load(&weights(&f["weights"], Dtype::Float32), g.clone())
                            .unwrap();
                    assert!(
                        crate::adapters::apply(&mut fresh, variant, &[bad]).is_err(),
                        "accepted malformed {defect} {network:?} adapter"
                    );
                }
                let leased = temp.path().join("leased.safetensors");
                Array::save_safetensors(
                    tensors.iter().map(|(k, v)| (k.as_str(), v)),
                    Some(&metadata),
                    &leased,
                )
                .unwrap();
                let pin = mlx_gen::gen_core::PinnedWeightsFile::pin(&leased).unwrap();
                let mut changed = tensors.clone();
                changed.insert(
                    first.clone(),
                    multiply(&changed[&first], Array::from_f32(2.)).unwrap(),
                );
                Array::save_safetensors(
                    changed.iter().map(|(k, v)| (k.as_str(), v)),
                    Some(&metadata),
                    &leased,
                )
                .unwrap();
                let mut lease_spec = spec.clone();
                lease_spec.path = leased;
                let mut fresh =
                    Transformer::load(&weights(&f["weights"], Dtype::Float32), g.clone()).unwrap();
                assert!(
                    crate::adapters::apply_pinned(&mut fresh, variant, &[lease_spec], &[pin])
                        .is_err(),
                    "changed adapter bypassed its exact source token"
                );
                let wrong = if variant == MossImageVariant::Base {
                    MossImageVariant::Sft
                } else {
                    MossImageVariant::Base
                };
                assert!(crate::adapters::apply(&mut reloaded, wrong, &[spec]).is_err());
            }
        }
    }
}

#[cfg(test)]
mod training_bf16_cpu_tests {
    use super::author_fixture_tests::*;
    use super::*;
    use mlx_gen::train::lora::{build_lokr_targets, build_lora_targets, TrainAdapter};
    use mlx_gen::{
        gen_core::{moss_image::MossImageVariant, moss_image_training as contract},
        NetworkType, TrainingConfig,
    };
    #[test]
    #[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
    fn production_bf16_backbone_keeps_fp32_adapter_gradients_and_portable_reload() {
        let _device = crate::cpu_test_device();
        let f = fixture("ordinary");
        let g = geometry(&f);
        let x = slice_axis(&array(&f["inputs"]["hidden_states"]), 0, 0, 1).unwrap();
        let context = contexts(
            &f["inputs"]["encoder_hidden_states"],
            &f["inputs"]["encoder_attention_mask"],
            g.text,
        );
        for targets in [
            vec!["v".into(), "o".into(), "ffn_down".into()],
            vec!["o".into()],
        ] {
            for network in [NetworkType::Lora, NetworkType::Lokr] {
                let mut model =
                    Transformer::load(&weights(&f["weights"], Dtype::Bfloat16), g.clone()).unwrap();
                let frozen_values =
                    crate::pipeline::logical_f32_readback(&model.video.weight).unwrap();
                model.cast_training_weights(Dtype::Float32).unwrap();
                assert!(model
                    .owned_arrays()
                    .iter()
                    .all(|a| a.dtype() == Dtype::Float32));
                assert_eq!(
                    crate::pipeline::logical_f32_readback(&model.video.weight).unwrap(),
                    frozen_values
                );
                model.cast_training_weights(Dtype::Bfloat16).unwrap();
                assert!(model
                    .owned_arrays()
                    .iter()
                    .all(|a| a.dtype() == Dtype::Bfloat16));
                let paths = contract::target_paths(g.layers, &targets).unwrap();
                let (adapter, mut params) = match network {
                    NetworkType::Lora => {
                        let (t, p) = build_lora_targets(&mut model, &paths, 2, 71).unwrap();
                        (TrainAdapter::Lora { targets: t }, p)
                    }
                    NetworkType::Lokr => {
                        let (t, p) = build_lokr_targets(&mut model, &paths, 2, 2, 71).unwrap();
                        (TrainAdapter::Lokr { targets: t }, p)
                    }
                };
                let cfg = TrainingConfig {
                    rank: 2,
                    alpha: 3.,
                    network_type: network,
                    ..Default::default()
                };
                let noise = add(&x, Array::from_f32(0.7)).unwrap();
                let (_, grad) = crate::training::loss_grads(
                    &mut model,
                    &params,
                    &adapter,
                    &cfg,
                    &x,
                    &context[0],
                    &noise,
                    0.41,
                    None,
                    &CancelFlag::new(),
                )
                .unwrap();
                assert!(grad.values().all(|g| g.dtype() == Dtype::Float32));
                assert!(
                    grad.values()
                        .map(|a| crate::pipeline::logical_f32_readback(a)
                            .unwrap()
                            .iter()
                            .map(|v| v * v)
                            .sum::<f32>())
                        .sum::<f32>()
                        > 1e-12
                );
                let mut optimizer =
                    mlx_gen::TrainOptimizer::from_config("adamw", 0.01, 0.).unwrap();
                mlx_gen::train::lora::adapter_optimizer_update(
                    &mut optimizer,
                    &mut params,
                    &grad,
                    &cfg,
                    0,
                    71,
                )
                .unwrap();
                model
                    .install_training_adapter(&adapter, &params, 3., 2.)
                    .unwrap();
                model.training = true;
                let before = model
                    .forward(
                        &x,
                        &[410.],
                        std::slice::from_ref(&context[0]),
                        &[],
                        &CancelFlag::new(),
                    )
                    .unwrap();
                let temp = tempfile::tempdir().unwrap();
                let file = temp.path().join("adapter.safetensors");
                let metadata = contract::metadata(MossImageVariant::Base, contract::SAMPLER);
                let extra: Vec<_> = metadata
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.as_str()))
                    .collect();
                adapter
                    .save_with_meta(&params, 3., 2., 2, "", &extra, &file)
                    .unwrap();
                let kind = match network {
                    NetworkType::Lora => mlx_gen::runtime::AdapterKind::Lora,
                    NetworkType::Lokr => mlx_gen::runtime::AdapterKind::Lokr,
                };
                let mut reloaded =
                    Transformer::load(&weights(&f["weights"], Dtype::Bfloat16), g.clone()).unwrap();
                crate::adapters::apply(
                    &mut reloaded,
                    MossImageVariant::Base,
                    &[mlx_gen::runtime::AdapterSpec::new(file, 1., kind)],
                )
                .unwrap();
                let after = reloaded
                    .forward(
                        &x,
                        &[410.],
                        std::slice::from_ref(&context[0]),
                        &[],
                        &CancelFlag::new(),
                    )
                    .unwrap();
                assert_eq!(
                    crate::pipeline::logical_f32_readback(&before).unwrap(),
                    crate::pipeline::logical_f32_readback(&after).unwrap(),
                    "BF16 selected-target training/reload diverged: {:?}",
                    targets
                );
                assert!(reloaded.custom_owned_bytes > 0);
                assert_eq!(
                    reloaded.resident_bytes(),
                    reloaded.owned_weight_bytes + reloaded.custom_owned_bytes
                );
            }
        }
    }
}

#[cfg(test)]
mod review_fix_tests {
    use super::author_fixture_tests::*;
    use super::*;
    use mlx_gen::gen_core::{moss_image::MossImageVariant, moss_image_training as contract};
    use mlx_gen::runtime::{AdapterKind, AdapterSpec};
    fn tiny() -> Transformer {
        let f = fixture("ordinary");
        Transformer::load(&weights(&f["weights"], Dtype::Float32), geometry(&f)).unwrap()
    }
    fn file(
        temp: &tempfile::TempDir,
        values: std::collections::HashMap<String, Array>,
        meta: std::collections::BTreeMap<String, String>,
        kind: AdapterKind,
        scale: f32,
    ) -> AdapterSpec {
        let path = temp.path().join("adapter.safetensors");
        let meta: std::collections::HashMap<_, _> = meta.into_iter().collect();
        Array::save_safetensors(
            values.iter().map(|(k, v)| (k.as_str(), v)),
            Some(&meta),
            &path,
        )
        .unwrap();
        AdapterSpec::new(path, scale, kind)
    }
    fn full(shape: &[i32], value: f32) -> Array {
        Array::from_slice(
            &vec![value; shape.iter().map(|v| *v as usize).product()],
            shape,
        )
    }
    fn lokr_values() -> std::collections::HashMap<String, Array> {
        [
            ("layers.0.proj.q.lokr_w1".into(), full(&[2, 2], 0.5)),
            ("layers.0.proj.q.lokr_w2".into(), full(&[8, 8], 0.25)),
        ]
        .into_iter()
        .collect()
    }
    fn lokr_meta() -> std::collections::BTreeMap<String, String> {
        let mut m = contract::metadata(MossImageVariant::Base, contract::SAMPLER);
        m.insert("networkType".into(), "lokr".into());
        m.insert("rank".into(), "4".into());
        m.insert("alpha".into(), "6".into());
        m
    }
    #[test]
    #[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
    fn lokr_real_loader_rejects_crossed_axes_unused_factors_and_bad_scaling() {
        let _device = crate::cpu_test_device();
        for defect in [
            "axes",
            "orphan",
            "competing",
            "rank_zero",
            "rank_garbage",
            "alpha_garbage",
            "alpha_missing",
        ] {
            let mut values = lokr_values();
            let mut meta = lokr_meta();
            match defect {
                "axes" => {
                    values.insert("layers.0.proj.q.lokr_w2".into(), full(&[4, 16], 0.25));
                }
                "orphan" => {
                    values.insert("layers.0.proj.q.lokr_w2_a".into(), full(&[8, 1], 0.25));
                }
                "competing" => {
                    values.insert("layers.0.proj.q.lokr_w2_a".into(), full(&[8, 1], 0.25));
                    values.insert("layers.0.proj.q.lokr_w2_b".into(), full(&[1, 8], 0.25));
                }
                "rank_zero" => {
                    meta.insert("rank".into(), "0".into());
                }
                "rank_garbage" => {
                    meta.insert("rank".into(), "garbage".into());
                }
                "alpha_garbage" => {
                    meta.insert("alpha".into(), "garbage".into());
                }
                "alpha_missing" => {
                    meta.remove("alpha");
                }
                _ => unreachable!(),
            }
            let temp = tempfile::tempdir().unwrap();
            let spec = file(&temp, values, meta, AdapterKind::Lokr, 1.);
            assert!(
                crate::adapters::apply(&mut tiny(), MossImageVariant::Base, &[spec]).is_err(),
                "accepted {defect}"
            );
        }
    }
    #[test]
    #[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
    fn installed_adapter_validation_rejects_finite_sources_that_overflow() {
        let _device = crate::cpu_test_device();
        for defect in [
            "kron_product",
            "decomposed_factor",
            "lora_alpha_fold",
            "lora_strength",
        ] {
            let mut meta = lokr_meta();
            let mut values = lokr_values();
            let mut kind = AdapterKind::Lokr;
            let mut strength = 1.;
            match defect {
                "kron_product" => {
                    values.insert("layers.0.proj.q.lokr_w1".into(), full(&[2, 2], 1e30));
                    values.insert("layers.0.proj.q.lokr_w2".into(), full(&[8, 8], 1e30));
                }
                "decomposed_factor" => {
                    values.remove("layers.0.proj.q.lokr_w2");
                    values.insert("layers.0.proj.q.lokr_w2_a".into(), full(&[8, 1], 1e30));
                    values.insert("layers.0.proj.q.lokr_w2_b".into(), full(&[1, 8], 1e30));
                }
                "lora_alpha_fold" | "lora_strength" => {
                    meta.insert("networkType".into(), "lora".into());
                    kind = AdapterKind::Lora;
                    values = [
                        ("layers.0.proj.q.lora_A.weight".into(), full(&[2, 16], 1.)),
                        ("layers.0.proj.q.lora_B.weight".into(), full(&[16, 2], 1e30)),
                        (
                            "layers.0.proj.q.alpha".into(),
                            full(
                                &[1],
                                if defect == "lora_alpha_fold" {
                                    1e30
                                } else {
                                    2.
                                },
                            ),
                        ),
                    ]
                    .into_iter()
                    .collect();
                    if defect == "lora_strength" {
                        strength = 1e30;
                    }
                }
                _ => unreachable!(),
            }
            assert!(
                values.values().all(|array| array
                    .is_finite()
                    .unwrap()
                    .all(None)
                    .unwrap()
                    .item::<bool>()),
                "fixture must have finite source tensors"
            );
            let temp = tempfile::tempdir().unwrap();
            let spec = file(&temp, values, meta, kind, strength);
            assert!(
                crate::adapters::apply(&mut tiny(), MossImageVariant::Base, &[spec]).is_err(),
                "accepted derived overflow {defect}"
            );
        }
    }
    #[test]
    #[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
    fn all_projection_lokr_keeps_raw_factors_in_dense_and_checkpointed_backward() {
        let _device = crate::cpu_test_device();
        use mlx_gen::train::lora::{build_lokr_targets, TrainAdapter};
        let f = fixture("ordinary");
        let g = geometry(&f);
        let mut model = tiny();
        let paths = contract::target_paths(g.layers, &[]).unwrap();
        let (targets, params) = build_lokr_targets(&mut model, &paths, 2, 2, 19).unwrap();
        let adapter = TrainAdapter::Lokr { targets };
        let x = slice_axis(&array(&f["inputs"]["hidden_states"]), 0, 0, 1).unwrap();
        let context = contexts(
            &f["inputs"]["encoder_hidden_states"],
            &f["inputs"]["encoder_attention_mask"],
            g.text,
        );
        let noise = add(&x, Array::from_f32(0.7)).unwrap();
        let expected: usize = params.values().map(|a| a.nbytes()).sum();
        for checkpoint in [false, true] {
            let cfg = mlx_gen::TrainingConfig {
                rank: 2,
                alpha: 3.,
                network_type: mlx_gen::NetworkType::Lokr,
                gradient_checkpointing: checkpoint,
                ..Default::default()
            };
            let (_, grads) = crate::training::loss_grads(
                &mut model,
                &params,
                &adapter,
                &cfg,
                &x,
                &context[0],
                &noise,
                0.41,
                None,
                &CancelFlag::new(),
            )
            .unwrap();
            assert_eq!(
                model.raw_lokr_bytes(),
                expected,
                "retained reconstructed delta collection instead of raw trainable factors"
            );
            assert!(
                model
                    .blocks
                    .iter()
                    .flat_map(|b| &b.projections)
                    .all(|p| p.custom.adapters().is_empty()),
                "generic dense LoKr installation reached production graph"
            );
            assert!(
                grads
                    .values()
                    .map(|g| crate::pipeline::logical_f32_readback(g)
                        .unwrap()
                        .iter()
                        .map(|v| v * v)
                        .sum::<f32>())
                    .sum::<f32>()
                    > 0.
            );
        }
    }
    #[test]
    #[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
    fn routing_owner_tracks_widening_and_dmd_weight_replacement() {
        let _device = crate::cpu_test_device();
        let original = full(&[2, 2], 0.25).as_dtype(Dtype::Bfloat16).unwrap();
        let mut p = Projection {
            weight: original.clone(),
            bias: None,
            adapter: None,
            custom: AdaptableLinear::dense(original, None),
            lokr: Vec::new(),
            edit_adapter: false,
            edit_fused_bias: false,
        };
        p.widen().unwrap();
        assert_eq!(
            p.custom.dense_weight().unwrap().0.dtype(),
            Dtype::Float32,
            "routing retained original BF16 allocation"
        );
        p.replace_weight_bias(full(&[2, 2], 0.75), None);
        assert_eq!(
            crate::pipeline::logical_f32_readback(p.custom.dense_weight().unwrap().0).unwrap(),
            crate::pipeline::logical_f32_readback(&p.weight).unwrap(),
            "routing retained replaced time weight"
        );
    }
}

#[cfg(test)]
mod review_publication_tests {
    use super::author_fixture_tests::*;
    use super::*;
    #[test]
    #[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
    fn final_optimizer_values_are_validated_before_adapter_file_is_written() {
        let _device = crate::cpu_test_device();
        let f = fixture("ordinary");
        let mut model =
            Transformer::load(&weights(&f["weights"], Dtype::Float32), geometry(&f)).unwrap();
        let (targets, mut params) = mlx_gen::train::lora::build_lora_targets(
            &mut model,
            &["layers.0.proj.q".into()],
            2,
            13,
        )
        .unwrap();
        let key = params
            .keys()
            .find(|key| key.ends_with(".lora_b"))
            .unwrap()
            .clone();
        params.insert(key, Array::from_slice(&[1e30f32; 16 * 2], &[16, 2]));
        let adapter = mlx_gen::train::lora::TrainAdapter::Lora { targets };
        let config = mlx_gen::TrainingConfig {
            rank: 2,
            alpha: 1e30,
            ..Default::default()
        };
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("must-not-publish.safetensors");
        assert!(crate::adapters::save_validated(
            &mut model,
            &adapter,
            &params,
            &config,
            &[],
            &path
        )
        .is_err());
        assert!(
            !path.exists(),
            "invalid effective adapter was published as a trained artifact"
        );
    }
}

#[cfg(test)]
mod review_structured_math_tests {
    use super::author_fixture_tests::*;
    use super::*;
    use mlx_gen::gen_core::{moss_image::MossImageVariant, moss_image_training as contract};
    use std::collections::{BTreeMap, HashMap};
    fn values(rows: usize, cols: usize, phase: f32) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| ((i as f32 + 1.) * 0.31 + phase).sin() * 0.4)
            .collect()
    }
    fn product(a: &[f32], rows: usize, rank: usize, b: &[f32], cols: usize) -> Vec<f64> {
        (0..rows)
            .flat_map(|r| {
                (0..cols).map(move |c| {
                    (0..rank)
                        .map(|k| a[r * rank + k] as f64 * b[k * cols + c] as f64)
                        .sum()
                })
            })
            .collect()
    }
    #[test]
    #[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
    fn raw_structured_full_and_decomposed_factors_match_independent_kronecker_math() {
        let _device = crate::cpu_test_device();
        for ((a, c), (b, d)) in [((2usize, 3usize), (4usize, 2usize)), ((4, 2), (2, 3))] {
            for decomposed in [false, true] {
                let mut group = BTreeMap::new();
                let mut effective = Vec::new();
                for (name, rows, cols, phase) in [("lokr_w1", a, c, 0.2), ("lokr_w2", b, d, 0.7)] {
                    if decomposed {
                        let rank = 2;
                        let left = values(rows, rank, phase);
                        let right = values(rank, cols, phase + 0.4);
                        effective.push(product(&left, rows, rank, &right, cols));
                        group.insert(
                            format!("{name}_a"),
                            Array::from_slice(&left, &[rows as i32, rank as i32]),
                        );
                        group.insert(
                            format!("{name}_b"),
                            Array::from_slice(&right, &[rank as i32, cols as i32]),
                        );
                    } else {
                        let full = values(rows, cols, phase);
                        effective.push(full.iter().map(|v| *v as f64).collect());
                        group.insert(
                            name.into(),
                            Array::from_slice(&full, &[rows as i32, cols as i32]),
                        );
                    }
                }
                let input = values(6, c * d, 1.3);
                let x = Array::from_slice(&input, &[2, 3, (c * d) as i32]);
                let scale = -0.75f32;
                let lokr = crate::adapters::Lokr::from_group(
                    &group,
                    &[(a * b) as i32, (c * d) as i32],
                    scale,
                )
                .unwrap();
                let output = lokr.residual(&x).unwrap().expect("nonzero explicit scale");
                assert_eq!(output.shape(), &[2, 3, (a * b) as i32]);
                let actual = crate::pipeline::logical_f32_readback(&output).unwrap();
                for row in 0..6 {
                    for p in 0..a {
                        for q in 0..b {
                            let expected: f64 = (0..c)
                                .flat_map(|r| (0..d).map(move |s| (r, s)))
                                .map(|(r, s)| {
                                    input[row * c * d + r * d + s] as f64
                                        * effective[0][p * c + r]
                                        * effective[1][q * d + s]
                                        * scale as f64
                                })
                                .sum();
                            let error = (actual[row * a * b + p * b + q] as f64 - expected).abs();
                            assert!(error<1e-6+expected.abs()*1e-5,"Kronecker axis/order/scale mismatch: {error}, decomposed={decomposed}");
                        }
                    }
                }
                let zero = crate::adapters::Lokr::from_group(
                    &group,
                    &[(a * b) as i32, (c * d) as i32],
                    0.,
                )
                .unwrap();
                assert!(zero.residual(&x).unwrap().is_none());
            }
        }
    }
    #[test]
    #[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
    fn real_loader_uses_nondefault_rank_alpha_and_user_strength_on_the_projection() {
        let _device = crate::cpu_test_device();
        let fixture = fixture("ordinary");
        let geometry = geometry(&fixture);
        let mut metadata = contract::metadata(MossImageVariant::Base, contract::SAMPLER);
        metadata.insert("networkType".into(), "lokr".into());
        metadata.insert("rank".into(), "4".into());
        metadata.insert("alpha".into(), "6".into());
        let metadata: HashMap<_, _> = metadata.into_iter().collect();
        let w1 = Array::from_slice(&[0.5f32; 4], &[2, 2]);
        let w2 = Array::from_slice(&[0.25f32; 64], &[8, 8]);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("scaled.safetensors");
        Array::save_safetensors(
            [
                ("layers.0.proj.q.lokr_w1", &w1),
                ("layers.0.proj.q.lokr_w2", &w2),
            ],
            Some(&metadata),
            &path,
        )
        .unwrap();
        let x = Array::from_slice(&(1..=16).map(|v| v as f32).collect::<Vec<_>>(), &[1, 1, 16]);
        for strength in [1f32, -2.] {
            let mut model = Transformer::load(
                &weights(&fixture["weights"], Dtype::Float32),
                geometry.clone(),
            )
            .unwrap();
            crate::adapters::apply(
                &mut model,
                MossImageVariant::Base,
                &[mlx_gen::runtime::AdapterSpec::new(
                    path.clone(),
                    strength,
                    mlx_gen::runtime::AdapterKind::Lokr,
                )],
            )
            .unwrap();
            let residual = model.blocks[0].projections[0].lokr[0]
                .residual(&x)
                .unwrap()
                .unwrap();
            let expected = (1..=16).sum::<i32>() as f32 * 0.5 * 0.25 * (6. / 4.) * strength;
            assert_eq!(
                crate::pipeline::logical_f32_readback(&residual).unwrap(),
                vec![expected; 16]
            );
        }
        // Mirrored Candle/MLX parity oracle: rank=2, global alpha=4, no target alpha
        // MUST produce residual 17 at strength 1, not the unity-fallback value 8.5.
        // Per-target alpha overrides global; malformed native rank is a refusal.
        let a = Array::from_slice(&[0.125f32; 32], &[2, 16]);
        let b = Array::from_slice(&[0.25f32; 32], &[16, 2]);
        for (native_rank, target_alpha, expected_alpha) in
            [(2, None, 4.), (2, Some(1f32), 1.), (3, None, 4.)]
        {
            let mut metadata = contract::metadata(MossImageVariant::Base, contract::SAMPLER);
            metadata.extend([
                ("networkType".into(), "lora".into()),
                ("rank".into(), native_rank.to_string()),
                ("alpha".into(), "4".into()),
            ]);
            let metadata: HashMap<_, _> = metadata.into_iter().collect();
            let mut tensors = vec![
                ("layers.0.proj.q.lora_A.weight", a.clone()),
                ("layers.0.proj.q.lora_B.weight", b.clone()),
            ];
            if let Some(alpha) = target_alpha {
                tensors.push(("layers.0.proj.q.alpha", Array::from_slice(&[alpha], &[1])));
            }
            let file = temp.path().join("lora-alpha.safetensors");
            Array::save_safetensors(tensors.iter().map(|(k, v)| (*k, v)), Some(&metadata), &file)
                .unwrap();
            for strength in [1f32, -2.] {
                let mut model = Transformer::load(
                    &weights(&fixture["weights"], Dtype::Float32),
                    geometry.clone(),
                )
                .unwrap();
                let result = crate::adapters::apply(
                    &mut model,
                    MossImageVariant::Base,
                    &[mlx_gen::runtime::AdapterSpec::new(
                        file.clone(),
                        strength,
                        mlx_gen::runtime::AdapterKind::Lora,
                    )],
                );
                if native_rank != 2 {
                    assert!(
                        result.is_err(),
                        "LoRA tensor/native metadata rank mismatch accepted"
                    );
                    continue;
                }
                result.unwrap();
                let Adapter::Lora { a, b, scale } =
                    &model.blocks[0].projections[0].custom.adapters()[0]
                else {
                    panic!("actual loaded LoRA missing")
                };
                let residual = self::scale(
                    &linear_fp32(&linear_fp32(&x, &a.t()).unwrap(), &b.t()).unwrap(),
                    *scale,
                )
                .unwrap();
                let expected = 136. * 0.125 * 0.25 * 2. * (expected_alpha / 2.) * strength;
                assert_eq!(crate::pipeline::logical_f32_readback(&residual).unwrap(), vec![expected;16],
                    "native LoRA global alpha fallback/target precedence differs from Candle parity oracle");
            }
        }
    }
}
