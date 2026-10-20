//! Frozen MOSS graph. No FLUX DiT/scheduler or ordinary cast-down adapter abstraction is used.
use crate::adapters::Lokr;
use crate::precision::{attention_fp32, linear_fp32};
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
            mlx_gen::nn::linear(&cast, &self.weight, b)?
        } else {
            cast.matmul(self.weight.t())?
        };
        let mut base = base;
        if let Some((down, up, scale)) = &self.adapter {
            // The original input's FP32 values feed the residual; only the backbone is narrowed.
            let residual = linear_fp32(&linear_fp32(&x.as_dtype(Dtype::Float32)?, down)?, up)?;
            base = add(
                base.as_dtype(Dtype::Float32)?,
                self::scale(&residual, *scale)?,
            )?;
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
fn attention(
    q: &Array,
    k: &Array,
    v: &Array,
    rope: &Array,
    g: &Geometry,
    cancel: &CancelFlag,
    training: bool,
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
    let out = if training {
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
            self.proj
                .forward(&attention(&q, &k, &v, rope, g, cancel, training)?)?,
        )?;
        let y = self.n2.forward(&x)?;
        Ok(add(
            &x,
            self.down
                .forward(&situ(&self.up.forward(&y)?, &self.gate.forward(&y)?)?)?,
        )?)
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
        let a = self.projections[3].forward(&attention(&q, &k, &v, rope, g, cancel, training)?)?;
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
    pub(crate) fn cast_training_weights(&mut self, dtype: Dtype) -> Result<()> {
        for p in [
            &mut self.video,
            &mut self.text,
            &mut self.time1,
            &mut self.time2,
            &mut self.modulation,
            &mut self.head,
            &mut self.output,
        ] {
            p.cast(dtype)?;
        }
        self.norm.weight = self.norm.weight.as_dtype(dtype)?;
        for block in &mut self.text_blocks {
            for norm in [&mut block.n1, &mut block.n2, &mut block.qn, &mut block.kn] {
                norm.weight = norm.weight.as_dtype(dtype)?;
            }
            for p in [
                &mut block.qkv,
                &mut block.proj,
                &mut block.up,
                &mut block.gate,
                &mut block.down,
            ] {
                p.cast(dtype)?;
            }
        }
        for block in &mut self.blocks {
            for norm in [&mut block.n1, &mut block.n2, &mut block.qn, &mut block.kn] {
                norm.weight = norm.weight.as_dtype(dtype)?;
            }
            for p in &mut block.projections {
                p.cast(dtype)?;
            }
        }
        self.materialize_owned(&CancelFlag::new())
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

#[cfg(test)]
mod training_projection_tests {
    use super::*;
    use mlx_gen::adapters::{AdaptableLinear, Adapter};
    #[test]
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
        for network in [NetworkType::Lora, NetworkType::Lokr] {
            let mut model =
                Transformer::load(&weights(&f["weights"], Dtype::Bfloat16), g.clone()).unwrap();
            let frozen_values = crate::pipeline::logical_f32_readback(&model.video.weight).unwrap();
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
            let paths =
                contract::target_paths(g.layers, &["v".into(), "o".into(), "ffn_down".into()])
                    .unwrap();
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
            let mut optimizer = mlx_gen::TrainOptimizer::from_config("adamw", 0.01, 0.).unwrap();
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
                crate::pipeline::logical_f32_readback(&after).unwrap()
            );
            assert!(reloaded.custom_owned_bytes > 0);
            assert_eq!(
                reloaded.resident_bytes(),
                reloaded.owned_weight_bytes + reloaded.custom_owned_bytes
            );
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
    fn routing_owner_tracks_widening_and_dmd_weight_replacement() {
        let _device = crate::cpu_test_device();
        let original = full(&[2, 2], 0.25).as_dtype(Dtype::Bfloat16).unwrap();
        let mut p = Projection {
            weight: original.clone(),
            bias: None,
            adapter: None,
            custom: AdaptableLinear::dense(original, None),
            lokr: Vec::new(),
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
    }
}
