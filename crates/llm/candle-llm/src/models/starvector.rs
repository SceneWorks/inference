//! Native StarVector-1B tensor modules.

use candle_core::{DType, Tensor};
use candle_nn::ops::softmax_last_dim;

use crate::error::{Error, Result};
use crate::primitives::nn::{conv2d, gelu, layer_norm, linear};
use crate::primitives::Weights;

const CLIP_WIDTH: usize = 1024;
const CLIP_LAYERS: usize = 23;
const CLIP_HEADS: usize = 16;
const CLIP_PATCH: usize = 14;
pub const STARVECTOR_IMAGE_TOKENS: usize = 257;
pub const STARVECTOR_HIDDEN: usize = 2048;

fn tensor(w: &Weights, prefix: &str, leaf: &str) -> Result<Tensor> {
    Ok(w.require(&format!("{prefix}.{leaf}"))?.clone())
}

// OpenAI CLIP uses QuickGELU; GPTBigCode below retains its distinct GELU activation.
fn clip_quick_gelu(value: &Tensor) -> Result<Tensor> {
    Ok((value * candle_nn::ops::sigmoid(&(value * 1.702)?)?)?)
}

struct ClipBlock {
    ln1w: Tensor,
    ln1b: Tensor,
    qkvw: Tensor,
    qkvb: Tensor,
    outw: Tensor,
    outb: Tensor,
    ln2w: Tensor,
    ln2b: Tensor,
    fcw: Tensor,
    fcb: Tensor,
    projw: Tensor,
    projb: Tensor,
}

impl ClipBlock {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        Ok(Self {
            ln1w: tensor(w, p, "ln_1.weight")?,
            ln1b: tensor(w, p, "ln_1.bias")?,
            qkvw: tensor(w, p, "attn.in_proj_weight")?,
            qkvb: tensor(w, p, "attn.in_proj_bias")?,
            outw: tensor(w, p, "attn.out_proj.weight")?,
            outb: tensor(w, p, "attn.out_proj.bias")?,
            ln2w: tensor(w, p, "ln_2.weight")?,
            ln2b: tensor(w, p, "ln_2.bias")?,
            fcw: tensor(w, p, "mlp.c_fc.weight")?,
            fcb: tensor(w, p, "mlp.c_fc.bias")?,
            projw: tensor(w, p, "mlp.c_proj.weight")?,
            projb: tensor(w, p, "mlp.c_proj.bias")?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let h = layer_norm(input, &self.ln1w, &self.ln1b, 1e-5)?;
        let qkv = linear(&h, &self.qkvw, Some(&self.qkvb))?;
        let (batch, seq, _) = qkv.dims3()?;
        let head = CLIP_WIDTH / CLIP_HEADS;
        let split = |start| -> Result<Tensor> {
            Ok(qkv
                .narrow(2, start, CLIP_WIDTH)?
                .reshape((batch, seq, CLIP_HEADS, head))?
                .transpose(1, 2)?)
        };
        let dtype = input.dtype();
        let q = split(0)?.to_dtype(DType::F32)?;
        let k = split(CLIP_WIDTH)?.to_dtype(DType::F32)?;
        let v = split(CLIP_WIDTH * 2)?.to_dtype(DType::F32)?;
        let attn = softmax_last_dim(
            &((q.matmul(&k.transpose(2, 3)?.contiguous()?)?) * (head as f64).powf(-0.5))?,
        )?
        .matmul(&v)?
        .to_dtype(dtype)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch, seq, CLIP_WIDTH))?;
        let residual = (input + linear(&attn, &self.outw, Some(&self.outb))?)?;
        let mlp = clip_quick_gelu(&linear(
            &layer_norm(&residual, &self.ln2w, &self.ln2b, 1e-5)?,
            &self.fcw,
            Some(&self.fcb),
        )?)?;
        Ok((&residual + linear(&mlp, &self.projw, Some(&self.projb))?)?)
    }
}

/// OpenAI CLIP ViT-L/14 with the exact packed-QKV naming used by the checkpoint.
pub struct StarVectorClip {
    conv: Tensor,
    class: Tensor,
    positions: Tensor,
    ln_pre_w: Tensor,
    ln_pre_b: Tensor,
    blocks: Vec<ClipBlock>,
    ln_out_w: Tensor,
    ln_out_b: Tensor,
}

fn align_clip_pixels(pixels: &Tensor, convolution: &Tensor) -> Result<Tensor> {
    Ok(pixels.to_dtype(convolution.dtype())?)
}

impl StarVectorClip {
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let p = "model.image_encoder.visual_encoder";
        Ok(Self {
            conv: tensor(w, p, "conv1.weight")?,
            class: tensor(w, p, "class_embedding")?,
            positions: tensor(w, p, "positional_embedding")?,
            ln_pre_w: tensor(w, p, "ln_pre.weight")?,
            ln_pre_b: tensor(w, p, "ln_pre.bias")?,
            blocks: (0..CLIP_LAYERS)
                .map(|i| ClipBlock::load(w, &format!("{p}.transformer.resblocks.{i}")))
                .collect::<Result<Vec<_>>>()?,
            ln_out_w: tensor(w, "model.image_encoder.ln_vision", "weight")?,
            ln_out_b: tensor(w, "model.image_encoder.ln_vision", "bias")?,
        })
    }

    pub fn forward(&self, pixels: &Tensor) -> Result<Tensor> {
        let pixels = align_clip_pixels(pixels, &self.conv)?;
        let patches = conv2d(&pixels, &self.conv, None, CLIP_PATCH, 0)?;
        let (batch, _, height, width) = patches.dims4()?;
        if (height, width) != (16, 16) {
            return Err(Error::Msg(format!(
                "starvector CLIP expected 16x16 patch grid, got {height}x{width}"
            )));
        }
        let patches = patches.flatten_from(2)?.transpose(1, 2)?;
        let class = self
            .class
            .reshape((1, 1, CLIP_WIDTH))?
            .broadcast_as((batch, 1, CLIP_WIDTH))?;
        let mut h = Tensor::cat(&[&class, &patches], 1)?.broadcast_add(
            &self
                .positions
                .reshape((1, STARVECTOR_IMAGE_TOKENS, CLIP_WIDTH))?,
        )?;
        h = layer_norm(&h, &self.ln_pre_w, &self.ln_pre_b, 1e-5)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        layer_norm(&h, &self.ln_out_w, &self.ln_out_b, 1e-5)
    }
}

/// Checkpoint adapter (`Linear → Swish → Linear → eval BatchNorm1d`).
pub struct StarVectorAdapter {
    fcw: Tensor,
    fcb: Tensor,
    projw: Tensor,
    projb: Tensor,
    normw: Tensor,
    normb: Tensor,
    mean: Tensor,
    var: Tensor,
}
impl StarVectorAdapter {
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let p = "model.image_projection";
        Ok(Self {
            fcw: tensor(w, p, "c_fc.weight")?,
            fcb: tensor(w, p, "c_fc.bias")?,
            projw: tensor(w, p, "c_proj.weight")?,
            projb: tensor(w, p, "c_proj.bias")?,
            normw: tensor(w, p, "norm.weight")?,
            normb: tensor(w, p, "norm.bias")?,
            mean: tensor(w, p, "norm.running_mean")?,
            var: tensor(w, p, "norm.running_var")?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = linear(x, &self.fcw, Some(&self.fcb))?;
        let h = (&h * candle_nn::ops::sigmoid(&h)?)?;
        let h = linear(&h, &self.projw, Some(&self.projb))?;
        let (_, tokens, hidden) = h.dims3()?;
        if tokens != STARVECTOR_IMAGE_TOKENS || hidden != STARVECTOR_HIDDEN {
            return Err(Error::Msg("starvector adapter shape mismatch".into()));
        }
        let inv_std = self.var.affine(1.0, 1e-5)?.sqrt()?.recip()?;
        let scale = self
            .normw
            .broadcast_mul(&inv_std)?
            .reshape((1, tokens, 1))?;
        let offset = self
            .normb
            .broadcast_sub(
                &self
                    .mean
                    .broadcast_mul(&self.normw.broadcast_mul(&inv_std)?)?,
            )?
            .reshape((1, tokens, 1))?;
        h.broadcast_mul(&scale)?
            .broadcast_add(&offset)
            .map_err(Into::into)
    }
}

/// GPTBigCode decoder for the checkpoint's `inputs_embeds` StarVector prefill.
///
/// It intentionally owns a per-layer MQA cache, so a provider serializes one generation at a time
/// behind a mutex and can reset it on cancellation/unload.
pub struct StarVectorDecoder {
    wte: Tensor,
    wpe: Tensor,
    layers: Vec<BigCodeBlock>,
    lnw: Tensor,
    lnb: Tensor,
    head: Tensor,
}
struct BigCodeBlock {
    ln1w: Tensor,
    ln1b: Tensor,
    qkvw: Tensor,
    qkvb: Tensor,
    outw: Tensor,
    outb: Tensor,
    ln2w: Tensor,
    ln2b: Tensor,
    fcw: Tensor,
    fcb: Tensor,
    projw: Tensor,
    projb: Tensor,
    cache: Option<Tensor>,
}
impl BigCodeBlock {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        Ok(Self {
            ln1w: tensor(w, p, "ln_1.weight")?,
            ln1b: tensor(w, p, "ln_1.bias")?,
            qkvw: decoder_projection(w, p, "attn.c_attn.weight")?,
            qkvb: tensor(w, p, "attn.c_attn.bias")?,
            outw: decoder_projection(w, p, "attn.c_proj.weight")?,
            outb: tensor(w, p, "attn.c_proj.bias")?,
            ln2w: tensor(w, p, "ln_2.weight")?,
            ln2b: tensor(w, p, "ln_2.bias")?,
            fcw: decoder_projection(w, p, "mlp.c_fc.weight")?,
            fcb: tensor(w, p, "mlp.c_fc.bias")?,
            projw: decoder_projection(w, p, "mlp.c_proj.weight")?,
            projb: tensor(w, p, "mlp.c_proj.bias")?,
            cache: None,
        })
    }
    fn reset(&mut self) {
        self.cache = None;
    }
    fn forward(&mut self, input: &Tensor, past: usize) -> Result<Tensor> {
        let h = layer_norm(input, &self.ln1w, &self.ln1b, 1e-5)?;
        let qkv = linear(&h, &self.qkvw, Some(&self.qkvb))?;
        let (b, s, _) = qkv.dims3()?;
        let q = qkv
            .narrow(2, 0, STARVECTOR_HIDDEN)?
            .reshape((b, s, 16, 128))?;
        let mut kv = qkv.narrow(2, STARVECTOR_HIDDEN, 256)?;
        if let Some(old) = &self.cache {
            kv = Tensor::cat(&[old, &kv], 1)?;
        }
        self.cache = Some(kv.clone());
        let attn = multi_query_attention(&q, &kv, past)?;
        let residual = (input + linear(&attn, &self.outw, Some(&self.outb))?)?;
        let mlp = gelu(&linear(
            &layer_norm(&residual, &self.ln2w, &self.ln2b, 1e-5)?,
            &self.fcw,
            Some(&self.fcb),
        )?)?;
        Ok((&residual + linear(&mlp, &self.projw, Some(&self.projb))?)?)
    }
}

fn multi_query_attention(q: &Tensor, kv: &Tensor, past: usize) -> Result<Tensor> {
    let (b, s, heads, head) = q.dims4()?;
    let (kv_batch, total, kv_width) = kv.dims3()?;
    if kv_batch != b || total != past + s || kv_width != head * 2 {
        return Err(Error::Msg(format!(
            "starvector MQA shape mismatch: query={:?}, kv={:?}, past={past}",
            q.dims(),
            kv.dims()
        )));
    }

    // GPTBigCode's multi-query attention shares one K/V head across all query heads. Fold the
    // query-head axis into the row axis for both matrix products so the shared K/V tensors are
    // consumed once without materializing sixteen copies. Restore `[batch, tokens, hidden]` only
    // after the per-head value product.
    let query = q
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b, heads * s, head))?;
    let keys = kv.narrow(2, 0, head)?.transpose(1, 2)?;
    // The V half has a `2 * head` row stride after `narrow`; CUDA GEMM only accepts a dense
    // minor matrix, so materialize this view just as we do for the transposed K half.
    let values = kv.narrow(2, head, head)?.contiguous()?;
    let scores = (query.matmul(&keys.contiguous()?)? * (head as f64).powf(-0.5))?
        .reshape((b, heads, s, total))?;
    let mut allow = vec![0u8; s * total];
    for row in 0..s {
        for col in 0..=past + row {
            allow[row * total + col] = 1;
        }
    }
    let allow =
        Tensor::from_vec(allow, (1, 1, s, total), scores.device())?.broadcast_as(scores.dims())?;
    let neg = Tensor::new(f32::NEG_INFINITY, scores.device())?.broadcast_as(scores.dims())?;
    Ok(softmax_last_dim(&allow.where_cond(&scores, &neg)?)?
        .reshape((b, heads * s, total))?
        .matmul(&values)?
        .reshape((b, heads, s, head))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b, s, heads * head))?)
}

/// The published StarVector GPTBigCode tensors are already `[out, in]`, exactly as Candle's
/// shared `linear` leaf expects. Keeping this boundary explicit prevents accidentally applying
/// the older Transformers `Conv1D` `[in, out]` convention to these safetensors.
fn decoder_projection(w: &Weights, prefix: &str, suffix: &str) -> Result<Tensor> {
    tensor(w, prefix, suffix)
}

fn tied_token_embedding(w: &Weights, prefix: &str) -> Result<(Tensor, Tensor)> {
    let wte = tensor(w, prefix, "wte.weight")?;
    // GPTBigCode ties the output projection to the token embedding. The published
    // StarVector-1B snapshot therefore intentionally has no separate `lm_head.weight`.
    let head = wte.clone();
    Ok((wte, head))
}

impl StarVectorDecoder {
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let p = "model.svg_transformer.transformer.transformer";
        let (wte, head) = tied_token_embedding(w, p)?;
        Ok(Self {
            wte,
            wpe: tensor(w, p, "wpe.weight")?,
            layers: (0..24)
                .map(|i| BigCodeBlock::load(w, &format!("{p}.h.{i}")))
                .collect::<Result<Vec<_>>>()?,
            lnw: tensor(w, p, "ln_f.weight")?,
            lnb: tensor(w, p, "ln_f.bias")?,
            head,
        })
    }
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
    pub fn embeddings(&self, ids: &Tensor) -> Result<Tensor> {
        crate::primitives::nn::embed(&self.wte, ids)
    }
    pub fn forward_embeds(&mut self, input: &Tensor, past: usize) -> Result<Tensor> {
        let (b, s, h) = input.dims3()?;
        if h != STARVECTOR_HIDDEN || past + s > 8192 {
            return Err(Error::Msg("starvector decoder context limit".into()));
        }
        let pos = self
            .wpe
            .narrow(0, past, s)?
            .reshape((1, s, h))?
            .broadcast_as((b, s, h))?;
        let mut out = input.broadcast_add(&pos)?;
        for layer in &mut self.layers {
            out = layer.forward(&out, past)?;
        }
        let out = layer_norm(&out, &self.lnw, &self.lnb, 1e-5)?;
        linear(&out.narrow(1, s - 1, 1)?.squeeze(1)?, &self.head, None)
    }
}

/// Loaded native tensor stack.  The provider owns this behind a mutex because decoder KV state is
/// request-local; [`reset`](Self::reset) is called on every terminal path.
pub struct StarVectorModel {
    pub vision: StarVectorClip,
    pub adapter: StarVectorAdapter,
    pub decoder: StarVectorDecoder,
}
impl StarVectorModel {
    pub fn from_weights(weights: &Weights) -> Result<Self> {
        Ok(Self {
            vision: StarVectorClip::from_weights(weights)?,
            adapter: StarVectorAdapter::from_weights(weights)?,
            decoder: StarVectorDecoder::from_weights(weights)?,
        })
    }
    pub fn image_embeddings(&self, pixels: &Tensor) -> Result<Tensor> {
        self.adapter.forward(&self.vision.forward(pixels)?)
    }
    pub fn reset(&mut self) {
        self.decoder.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap;

    #[test]
    fn clip_quick_gelu_matches_upstream_numeric_fixture() {
        let values = [-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let input = Tensor::from_slice(&values, (1, 5), &Device::Cpu).unwrap();
        let actual = clip_quick_gelu(&input)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Frozen scalar outputs from x * sigmoid(1.702 * x), including both tails.
        let expected = [
            -0.06434137685579186f64,
            -0.1542042340671787,
            0.0,
            0.8457957659328212,
            1.9356586231442083,
        ];
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((*actual as f64 - expected).abs() < 2e-7);
        }
        assert!(
            (gelu(&input)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[4]
                - actual[4])
                .abs()
                > 0.01
        );
    }

    #[test]
    fn clip_aligns_f32_preprocessing_with_f16_checkpoint_weights() {
        let device = Device::Cpu;
        let pixels = Tensor::zeros((1,), DType::F32, &device).unwrap();
        let conv = Tensor::zeros((1,), DType::F16, &device).unwrap();

        let aligned = align_clip_pixels(&pixels, &conv).unwrap();

        assert_eq!(aligned.dtype(), DType::F16);
    }

    #[test]
    fn multi_query_attention_preserves_head_and_token_axes() {
        let device = Device::Cpu;
        let query = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0],
            (1, 2, 2, 2),
            &device,
        )
        .unwrap();
        let kv = Tensor::from_vec(
            vec![1.0f32, 0.0, 10.0, 20.0, 0.0, 1.0, 30.0, 40.0],
            (1, 2, 4),
            &device,
        )
        .unwrap();

        let output = multi_query_attention(&query, &kv, 0).unwrap();

        assert_eq!(output.dims(), &[1, 2, 4]);
        let output = output.to_vec3::<f32>().unwrap();
        assert_eq!(output[0][0], vec![10.0, 20.0, 10.0, 20.0]);
        assert!((output[0][1][0] - 20.0).abs() < 1e-5);
        assert!((output[0][1][1] - 30.0).abs() < 1e-5);
        assert!((output[0][1][2] - 26.088594).abs() < 1e-4);
        assert!((output[0][1][3] - 36.088596).abs() < 1e-4);
    }

    #[test]
    fn starvector_1b_prefill_mqa_accepts_the_native_axis_contract() {
        let device = Device::Cpu;
        let query = Tensor::zeros((1, 259, 16, 128), DType::F32, &device).unwrap();
        let kv = Tensor::zeros((1, 259, 256), DType::F32, &device).unwrap();

        let output = multi_query_attention(&query, &kv, 0).unwrap();

        assert_eq!(output.dims(), &[1, 259, STARVECTOR_HIDDEN]);
        assert_eq!(output.sum_all().unwrap().to_scalar::<f32>().unwrap(), 0.0);
    }

    #[test]
    fn multi_query_attention_decode_reads_the_complete_shared_cache() {
        let device = Device::Cpu;
        let query = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (1, 1, 2, 2), &device).unwrap();
        let kv = Tensor::from_vec(
            vec![
                1.0f32, 0.0, 10.0, 20.0, 0.0, 1.0, 30.0, 40.0, 1.0, 1.0, 50.0, 60.0,
            ],
            (1, 3, 4),
            &device,
        )
        .unwrap();

        let output = multi_query_attention(&query, &kv, 2)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();

        let exp_scaled = (1.0f32 / 2.0f32.sqrt()).exp();
        let high = exp_scaled / (2.0 * exp_scaled + 1.0);
        let low = 1.0 / (2.0 * exp_scaled + 1.0);
        let expected = [
            60.0 * high + 30.0 * low,
            80.0 * high + 40.0 * low,
            80.0 * high + 10.0 * low,
            100.0 * high + 20.0 * low,
        ];
        for (actual, expected) in output[0][0].iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn multi_query_attention_rejects_cache_geometry_that_disagrees_with_past() {
        let device = Device::Cpu;
        let query = Tensor::zeros((1, 1, 2, 2), DType::F32, &device).unwrap();
        let short_cache = Tensor::zeros((1, 2, 4), DType::F32, &device).unwrap();

        let error = multi_query_attention(&query, &short_cache, 2).unwrap_err();

        assert!(error
            .to_string()
            .contains("query=[1, 1, 2, 2], kv=[1, 2, 4], past=2"));
    }

    #[test]
    fn decoder_projection_preserves_published_out_in_layout() {
        let device = Device::Cpu;
        let projection =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &device).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("fixture.attn.c_attn.weight".into(), projection);
        let weights = Weights::from_map(tensors, device.clone());

        let projection = decoder_projection(&weights, "fixture", "attn.c_attn.weight").unwrap();
        assert_eq!(projection.dims(), &[3, 2]);
        let input = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &device).unwrap();
        assert_eq!(
            linear(&input, &projection, None)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![1.0, 3.0, 5.0]]
        );
    }

    #[test]
    fn tied_token_embedding_projects_without_a_separate_lm_head() {
        let device = Device::Cpu;
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let wte = Tensor::from_vec(values.clone(), (3, 2), &device).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.svg_transformer.transformer.transformer.wte.weight".into(),
            wte,
        );
        let weights = Weights::from_map(tensors, device.clone());

        let (wte, head) =
            tied_token_embedding(&weights, "model.svg_transformer.transformer.transformer")
                .unwrap();
        assert_eq!(
            head.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            values
        );
        assert_eq!(
            wte.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            head.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        let hidden = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &device).unwrap();
        assert_eq!(
            linear(&hidden, &head, None)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![1.0, 3.0, 5.0]]
        );
    }
}
