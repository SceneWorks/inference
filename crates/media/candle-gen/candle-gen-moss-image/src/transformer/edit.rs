//! Request-owned edit graph: references evolve independently at t=0, with per-layer K/V.
use super::*;

type KeyValue = (Tensor, Tensor);
pub(crate) struct EditCache {
    keys_values: Vec<KeyValue>,
}
pub(crate) struct EditConditioning<'a> {
    pub context: &'a Tensor,
    pub references: &'a [Tensor],
    pub cache: Option<&'a EditCache>,
}

fn silu(x: &Tensor) -> Result<Tensor> {
    // nn.SiLU has one rounding boundary, after its FP32 activation evaluation.
    Ok(candle_nn::ops::silu(&x.to_dtype(DType::F32)?)?.to_dtype(x.dtype())?)
}
pub(super) fn edit_situ(hidden: &Tensor, gate: &Tensor) -> Result<Tensor> {
    // PyTorch evaluates each sigmoid/scalar division in FP32, then rounds that operation's
    // result. Candle CPU's BF16 sigmoid rounds exp/add/reciprocal separately, and affine scalar
    // division rounds the reciprocal early. Retain the source's eager BF16 boundaries here.
    let divide = |x: &Tensor, scalar: f64| -> Result<Tensor> {
        Ok((x.to_dtype(DType::F32)? / scalar)?.to_dtype(x.dtype())?)
    };
    let sigmoid = candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?.to_dtype(gate.dtype())?;
    let gated = (divide(gate, 4.)?.tanh()? * 4.)?.mul(&sigmoid)?;
    Ok(gated.mul(&(divide(hidden, 25.)?.tanh()? * 25.)?)?)
}
fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    // K is already rotated. Pack heads before the cuBLAS-supported transpose, including F32.
    let q = q.to_dtype(DType::F32)?.contiguous()?;
    let k = k.to_dtype(DType::F32)?.contiguous()?.transpose(2, 3)?;
    let scores = (q.matmul(&k)? / (q.dim(3)? as f64).sqrt())?;
    let values = v.to_dtype(DType::F32)?.contiguous()?;
    let output = if v.dtype() == DType::BF16 {
        // The frozen source's BF16 SDPA uses a rounded, unnormalized exponential operand.
        // Accumulate both the value numerator and normalization in F32; rounding normalized
        // probabilities instead changes the source operation boundaries.
        let exponentials = scores.broadcast_sub(&scores.max_keepdim(3)?)?.exp()?;
        let denominator = exponentials.sum_keepdim(3)?;
        exponentials
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?
            .matmul(&values)?
            .broadcast_div(&denominator)?
    } else {
        candle_nn::ops::softmax_last_dim(&scores)?.matmul(&values)?
    };
    Ok(output.to_dtype(v.dtype())?)
}
pub(super) fn text_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    rope: &Tensor,
    heads: usize,
) -> Result<Tensor> {
    let (b, s, d) = q.dims3()?;
    let (q, kt) = attention_qk_operands(q, k, v.dtype(), rope, heads)?;
    let v = v.reshape((b, s, heads, d / heads))?.transpose(1, 2)?;
    Ok(attention(&q, &kt.transpose(2, 3)?, &v)?
        .transpose(1, 2)?
        .reshape((b, s, d))?)
}
impl Block {
    fn edit_qkv(&self, x: &Tensor, rope: &Tensor, m: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let part = |i| m.i((.., i, ..))?.unsqueeze(1);
        let y = self
            .n1
            .forward(x)?
            .broadcast_mul(&(part(1)? + 1.0)?)?
            .broadcast_add(&part(0)?)?
            .to_dtype(self.q.dtype())?;
        let (b, s, d) = y.dims3()?;
        let shape = (b, s, self.heads, d / self.heads);
        let q = self.qn.forward(&self.q.forward(&y)?.reshape(shape)?)?;
        let k = self.kn.forward(&self.k.forward(&y)?.reshape(shape)?)?;
        let v = self.v.forward(&y)?.reshape(shape)?.transpose(1, 2)?;
        Ok((
            rotate(&q, rope)?.to_dtype(v.dtype())?.transpose(1, 2)?,
            rotate(&k, rope)?.to_dtype(v.dtype())?.transpose(1, 2)?,
            v,
        ))
    }
    fn edit_finish(&self, x: &Tensor, a: &Tensor, m: &Tensor) -> Result<Tensor> {
        let part = |i| m.i((.., i, ..))?.unsqueeze(1);
        let a = a.transpose(1, 2)?.reshape(x.shape())?;
        let x = x.add(&self.o.forward(&a)?.broadcast_mul(&part(2)?)?)?;
        let y = self
            .n2
            .forward(&x)?
            .broadcast_mul(&(part(4)? + 1.0)?)?
            .broadcast_add(&part(3)?)?
            .to_dtype(self.q.dtype())?;
        let hg = self.up_gate.forward(&y)?;
        let f = hg.dim(2)? / 2;
        let y = self
            .down
            .forward(&edit_situ(&hg.narrow(2, 0, f)?, &hg.narrow(2, f, f)?)?)?;
        Ok(x.add(&y.broadcast_mul(&part(5)?)?)?)
    }
    fn edit_reference(
        &self,
        x: &Tensor,
        rope: &Tensor,
        m: &Tensor,
        cancel: &CancelFlag,
    ) -> Result<(Tensor, KeyValue)> {
        check_cancel(cancel)?;
        let (q, k, v) = self.edit_qkv(x, rope, m)?;
        let next = self.edit_finish(x, &attention(&q, &k, &v)?, m)?;
        x.device().synchronize()?;
        check_cancel(cancel)?;
        Ok((next, (k, v)))
    }
    fn edit_body(
        &self,
        x: &Tensor,
        reference: &KeyValue,
        rope: &Tensor,
        m: &Tensor,
        text: usize,
        cancel: &CancelFlag,
    ) -> Result<Tensor> {
        check_cancel(cancel)?;
        let (q, k, v) = self.edit_qkv(x, rope, m)?;
        let insert = |body: &Tensor, reference: &Tensor| -> Result<Tensor> {
            Ok(Tensor::cat(
                &[
                    &body.narrow(2, 0, text)?,
                    reference,
                    &body.narrow(2, text, body.dim(2)? - text)?,
                ],
                2,
            )?)
        };
        let next = self.edit_finish(
            x,
            &attention(&q, &insert(&k, &reference.0)?, &insert(&v, &reference.1)?)?,
            m,
        )?;
        x.device().synchronize()?;
        check_cancel(cancel)?;
        Ok(next)
    }
}
impl Transformer {
    fn enable_edit_precision(&mut self) {
        for p in [
            &mut self.video,
            &mut self.text,
            &mut self.time1,
            &mut self.time2,
            &mut self.shared,
            &mut self.head,
            &mut self.final_linear,
        ] {
            p.edit_fused_bias = true;
        }
        for block in &mut self.text_blocks {
            for p in [
                &mut block.qkv,
                &mut block.o,
                &mut block.gate,
                &mut block.up,
                &mut block.down,
            ] {
                p.edit_fused_bias = true;
            }
        }
    }
    pub(crate) fn install_edit(&mut self, weights: &Weights, rank: usize) -> Result<()> {
        self.enable_edit_precision();
        let mut consumed = BTreeSet::new();
        for (i, block) in self.layers.iter_mut().enumerate() {
            for (p, name) in [
                &mut block.q,
                &mut block.k,
                &mut block.v,
                &mut block.o,
                &mut block.up_gate,
                &mut block.down,
            ]
            .into_iter()
            .zip(["q", "k", "v", "o", "ffn_up_gate", "ffn_down"])
            {
                if p.low_rank.is_some() {
                    return Err(CandleError::Msg("MOSS edit cannot compose with DMD".into()));
                }
                let a = format!("layers.{i}.proj.{name}.lora_A");
                let b = format!("layers.{i}.proj.{name}.lora_B");
                let down = weights.require(&a)?;
                let up = weights.require(&b)?;
                let shape = p.base.weight().dims2()?;
                if down.dims() != [rank, shape.1]
                    || up.dims() != [shape.0, rank]
                    || down.dtype() != DType::BF16
                    || up.dtype() != DType::BF16
                    || rank == 0
                {
                    return Err(CandleError::Msg(format!(
                        "MOSS edit BF16 A/B shape mismatch: {name}"
                    )));
                }
                self.resident_bytes += ((down.elem_count() + up.elem_count()) * 2) as u64;
                consumed.extend([a, b]);
                p.low_rank = Some(LowRank {
                    down,
                    up,
                    scale: 1.0,
                });
                p.edit_adapter = true;
            }
        }
        if weights.keys().any(|key| !consumed.contains(key)) {
            return Err(CandleError::Msg(
                "MOSS edit unconsumed adapter tensors".into(),
            ));
        }
        Ok(())
    }
    fn edit_time(&self, normalized: f32) -> Result<(Tensor, Tensor)> {
        if !normalized.is_finite() || !(0.0..=1.0).contains(&normalized) {
            return Err(CandleError::Msg(
                "MOSS edit time must be normalized in [0,1]".into(),
            ));
        }
        let half = self.g.freq / 2;
        let phase: Vec<_> = (0..half)
            .map(|i| f64::from(normalized * 1000.) * 10000_f64.powf(-(i as f64) / half as f64))
            .collect();
        let features: Vec<_> = phase
            .iter()
            .map(|p| p.cos() as f32)
            .chain(phase.iter().map(|p| p.sin() as f32))
            .collect();
        let t = Tensor::from_vec(features, (1, self.g.freq), self.device())?
            .to_dtype(self.time1.dtype())?;
        let t = self
            .time2
            .forward(&silu(&self.time1.forward(&t)?)?)?
            .to_dtype(DType::F32)?;
        Ok((
            self.shared
                .forward(&t.to_dtype(self.shared.dtype())?)?
                .reshape((1, 6, self.g.dim))?,
            self.head
                .forward(&t.to_dtype(self.head.dtype())?)?
                .reshape((1, 2, self.g.dim))?,
        ))
    }
    fn edit_reference_inputs(&self, references: &[Tensor]) -> Result<(Tensor, Tensor)> {
        let mut tokens = Vec::new();
        let mut ropes = Vec::new();
        for (index, reference) in references.iter().enumerate() {
            let (b, c, h, w) = reference.dims4()?;
            if b != 1 || c != self.g.channels || h == 0 || w == 0 {
                return Err(CandleError::Msg(
                    "invalid MOSS edit reference latent".into(),
                ));
            }
            let x = reference
                .reshape((1, c, h * w))?
                .transpose(1, 2)?
                .to_dtype(self.video.dtype())?
                .contiguous()?;
            let x = self.video.forward(&x)?;
            let head = self.g.dim / self.g.heads;
            let mut phases = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    for sine in [true, false] {
                        for coordinate in [
                            index as f32 + 1.,
                            0.,
                            y as f32 - (h - 1) as f32 / 2.,
                            x as f32 - (w - 1) as f32 / 2.,
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
            ropes.push(
                Tensor::from_vec(phases, (1, h * w, head), self.device())?.to_dtype(x.dtype())?,
            );
            tokens.push(x);
        }
        if tokens.is_empty() {
            return Err(CandleError::Msg("MOSS edit requires references".into()));
        }
        Ok((Tensor::cat(&tokens, 1)?, Tensor::cat(&ropes, 1)?))
    }
    pub(crate) fn prepare_edit_cache(
        &self,
        references: &[Tensor],
        cancel: &CancelFlag,
    ) -> Result<EditCache> {
        check_cancel(cancel)?;
        let (mut prefix, rope) = self.edit_reference_inputs(references)?;
        let (m, _) = self.edit_time(0.)?;
        let mut keys_values = Vec::new();
        for block in &self.layers {
            let (next, kv) = block.edit_reference(&prefix, &rope, &m, cancel)?;
            prefix = next;
            keys_values.push(kv);
        }
        Ok(EditCache { keys_values })
    }
    pub(crate) fn forward_edit(
        &self,
        latent: &Tensor,
        time: f32,
        conditioning: EditConditioning<'_>,
        cancel: &CancelFlag,
    ) -> Result<Tensor> {
        let EditConditioning {
            context,
            references,
            cache,
        } = conditioning;
        check_cancel(cancel)?;
        let (b, c, h, w) = latent.dims4()?;
        let (cb, text_len, td) = context.dims3()?;
        if b != 1
            || c != self.g.channels
            || cb != 1
            || text_len == 0
            || td != self.g.text
            || h == 0
            || w == 0
            || cache.is_some_and(|c| c.keys_values.len() != self.layers.len())
        {
            return Err(CandleError::Msg(
                "MOSS edit latent/context/cache shape mismatch".into(),
            ));
        }
        let video = self.video.forward(
            &latent
                .reshape((1, c, h * w))?
                .transpose(1, 2)?
                .to_dtype(self.video.dtype())?
                .contiguous()?,
        )?;
        let original = rope(
            h,
            w,
            text_len,
            self.g.dim / self.g.heads,
            video.dtype(),
            self.device(),
        )?;
        let text_rope = original.narrow(1, h * w, text_len)?;
        let mut text = self.project_context(context)?;
        for block in &self.text_blocks {
            check_cancel(cancel)?;
            text = block.forward_with_edit_precision(&text, &text_rope, true)?;
            self.device().synchronize()?;
            check_cancel(cancel)?;
        }
        let mut body = Tensor::cat(&[text, video], 1)?;
        let body_rope = Tensor::cat(&[&text_rope, &original.narrow(1, 0, h * w)?], 1)?;
        let (m, head) = self.edit_time(time)?;
        let (ref_m, _) = self.edit_time(0.)?;
        let mut uncached = if cache.is_none() {
            Some(self.edit_reference_inputs(references)?)
        } else {
            None
        };
        for (index, block) in self.layers.iter().enumerate() {
            let temporary;
            let kv = if let Some(cache) = cache {
                &cache.keys_values[index]
            } else {
                let (prefix, rope) = uncached.as_mut().expect("uncached prefix");
                let (next, kv) = block.edit_reference(prefix, rope, &ref_m, cancel)?;
                *prefix = next;
                temporary = kv;
                &temporary
            };
            body = block.edit_body(&body, kv, &body_rope, &m, text_len, cancel)?;
        }
        let output = self
            .norm
            .forward(
                &body
                    .narrow(1, text_len, h * w)?
                    .to_dtype(self.norm.0.dtype())?,
            )?
            .broadcast_mul(&(head.i((.., 1, ..))?.unsqueeze(1)? + 1.)?)?
            .broadcast_add(&head.i((.., 0, ..))?.unsqueeze(1)?)?;
        let output = self
            .final_linear
            .forward(&output.to_dtype(self.final_linear.dtype())?)?
            .transpose(1, 2)?
            .reshape((1, c, h, w))?;
        self.device().synchronize()?;
        check_cancel(cancel)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests;
