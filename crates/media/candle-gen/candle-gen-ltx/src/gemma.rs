//! Gemma-3-12B text encoder — the LTX-2.3 text-encoder backbone. Port of mlx-gen-ltx `gemma.rs`
//! (`GemmaModel::forward`), itself a port of `mlx_vlm` Gemma-3. Returns the **49 hidden states**
//! (scaled embedding + each of 48 layer outputs, the last final-normed) that the LTX feature
//! extractor concatenates and projects.
//!
//! Gemma specifics: RMSNorm scales by **(1 + weight)** (eps 1e-6); token embeddings ×√hidden_size
//! (bf16); **per-layer RoPE base** (local 1e4 on sliding layers `(i+1)%6 != 0`, global 1e6
//! otherwise); **q/k RMSNorm over head_dim** (256); GQA (16 q / 8 kv heads); attention scale
//! `256^-0.5`; MLP `down(gelu_tanh(gate(x)) * up(x))`; norm-sandwich block. Our checkpoint is dense
//! bf16 (no quant). The prompt is ≤ the sliding-window size (1024), so one full causal+padding mask
//! serves every layer (only the RoPE base differs) — the window never truncates, so it is not
//! modeled. Runs bf16; RoPE + attention compute in f32 for fidelity.

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{ops::rms_norm as candle_rms_norm, ops::softmax_last_dim, VarBuilder};

use crate::config::GemmaConfig;
use crate::quant::{qembedding, qlinear, QLinear};

/// Finite large-negative mask value (bf16 min, as f32) — used instead of -∞ so an all-masked row
/// (a left-padding query position) softmaxes to a finite uniform vector rather than NaN. Those
/// positions are zeroed downstream by the attention-mask multiply in the aggregator.
const MASK_NEG: f32 = -3.389_531_4e38;

/// `weight + 1.0` (Gemma RMSNorm scale), kept bf16.
fn norm_alpha(vb: &VarBuilder, key: &str) -> Result<Tensor> {
    let w = vb.get_unchecked(key)?;
    (w + 1.0)?.to_dtype(DType::BF16)
}

/// Packed-detecting bias-less Linear (sc-9417): loads the MLX-packed projection triple when a
/// `{key}.scales` sibling is present, else the dense bf16 weight unchanged. The Gemma TE is **dense**
/// in the hosted `SceneWorks/ltx-2.3-mlx` tier (no `.scales`), so this future-proofs the surface + is
/// covered by the guard; the dense arm is byte-identical to the legacy read. Gemma projections are
/// bias-less.
fn linear(vb: &VarBuilder, key: &str) -> Result<QLinear> {
    qlinear(vb, key, false)
}

struct GemmaLayer {
    input_ln: Tensor,
    post_attn_ln: Tensor,
    pre_ff_ln: Tensor,
    post_ff_ln: Tensor,
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
}

pub struct GemmaEncoder {
    embed: Tensor, // [vocab, hidden] bf16
    layers: Vec<GemmaLayer>,
    norm: Tensor,
    embed_scale: Tensor, // bf16 scalar √hidden
    cfg: GemmaConfig,
    device: Device,
}

impl GemmaEncoder {
    /// Build from a VarBuilder rooted at `language_model.model.` of a gemma-3-12b-it snapshot.
    pub fn new(vb: VarBuilder, cfg: &GemmaConfig) -> Result<Self> {
        let device = vb.device().clone();
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let lb = vb.pp(format!("layers.{i}"));
            let attn = lb.pp("self_attn");
            layers.push(GemmaLayer {
                input_ln: norm_alpha(&lb, "input_layernorm.weight")?,
                post_attn_ln: norm_alpha(&lb, "post_attention_layernorm.weight")?,
                pre_ff_ln: norm_alpha(&lb, "pre_feedforward_layernorm.weight")?,
                post_ff_ln: norm_alpha(&lb, "post_feedforward_layernorm.weight")?,
                q_proj: linear(&attn, "q_proj")?,
                k_proj: linear(&attn, "k_proj")?,
                v_proj: linear(&attn, "v_proj")?,
                o_proj: linear(&attn, "o_proj")?,
                q_norm: norm_alpha(&attn, "q_norm.weight")?,
                k_norm: norm_alpha(&attn, "k_norm.weight")?,
                gate_proj: linear(&lb.pp("mlp"), "gate_proj")?,
                up_proj: linear(&lb.pp("mlp"), "up_proj")?,
                down_proj: linear(&lb.pp("mlp"), "down_proj")?,
            });
        }
        // Packed-detecting `embed_tokens` (sc-9417): dense in the hosted tier, but routed through the
        // shared packed-detect so a future packed tier loads the table straight from the packed parts.
        // The encoder scales + index-selects the raw table, so keep the resolved table tensor.
        let embed = qembedding(&vb, "embed_tokens", cfg.vocab_size, cfg.hidden_size)?
            .weight()
            .to_dtype(DType::BF16)?;
        let scale = (cfg.hidden_size as f64).sqrt();
        let embed_scale = Tensor::new(scale as f32, &device)?.to_dtype(DType::BF16)?;
        Ok(Self {
            embed,
            layers,
            norm: norm_alpha(&vb, "norm.weight")?,
            embed_scale,
            cfg: cfg.clone(),
            device,
        })
    }

    fn rms(&self, x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
        candle_rms_norm(&x.contiguous()?, alpha, self.cfg.rms_eps as f32)
    }

    /// NeoX rotate-half RoPE in f32: `x` `[B,H,L,D]`, `cos`/`sin` `[1,1,L,D/2]` → rotated, cast back.
    fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let half = x.dim(D::Minus1)? / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let o1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
        let o2 = (x2.broadcast_mul(cos)? + x1.broadcast_mul(sin)?)?;
        Tensor::cat(&[&o1, &o2], D::Minus1)?.to_dtype(in_dtype)
    }

    /// Build `(cos, sin)` `[1,1,L,head_dim/2]` (f32) for a given RoPE base.
    fn rope_tables(&self, l: usize, base: f64) -> Result<(Tensor, Tensor)> {
        let d = self.cfg.head_dim;
        let half = d / 2;
        let mut cos = vec![0f32; l * half];
        let mut sin = vec![0f32; l * half];
        for p in 0..l {
            for i in 0..half {
                let inv_freq = base.powf(-(2.0 * i as f64) / d as f64);
                let theta = p as f64 * inv_freq;
                cos[p * half + i] = theta.cos() as f32;
                sin[p * half + i] = theta.sin() as f32;
            }
        }
        Ok((
            Tensor::from_vec(cos, (1, 1, l, half), &self.device)?,
            Tensor::from_vec(sin, (1, 1, l, half), &self.device)?,
        ))
    }

    fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let (b, kv, l, d) = x.dims4()?;
        x.unsqueeze(2)?
            .broadcast_as((b, kv, n_rep, l, d))?
            .reshape((b, kv * n_rep, l, d))
    }

    #[allow(clippy::too_many_arguments)]
    fn attn(
        &self,
        layer: &GemmaLayer,
        x: &Tensor,
        mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let (h, kv, d) = (self.cfg.num_heads, self.cfg.num_kv_heads, self.cfg.head_dim);
        let q = layer
            .q_proj
            .forward(x)?
            .reshape((b, l, h, d))?
            .transpose(1, 2)?;
        let k = layer
            .k_proj
            .forward(x)?
            .reshape((b, l, kv, d))?
            .transpose(1, 2)?;
        let v = layer
            .v_proj
            .forward(x)?
            .reshape((b, l, kv, d))?
            .transpose(1, 2)?;
        // q/k RMSNorm over head_dim, then per-layer RoPE.
        let q = self.rms(&q.contiguous()?, &layer.q_norm)?;
        let k = self.rms(&k.contiguous()?, &layer.k_norm)?;
        let q = Self::apply_rope(&q, cos, sin)?;
        let k = Self::apply_rope(&k, cos, sin)?;
        // GQA + attention in f32.
        let k = Self::repeat_kv(&k, h / kv)?;
        let v = Self::repeat_kv(&v, h / kv)?;
        let qf = q.to_dtype(DType::F32)?.contiguous()?;
        let kf = k.to_dtype(DType::F32)?.contiguous()?;
        let vf = v.to_dtype(DType::F32)?.contiguous()?;
        let scale = self.cfg.query_pre_attn_scalar.powf(-0.5);
        let scores = (qf.matmul(&kf.transpose(2, 3)?)? * scale)?;
        let scores = scores.broadcast_add(mask)?;
        let probs = softmax_last_dim(&scores)?;
        let out = probs.matmul(&vf)?; // (b,h,l,d) f32
        let out = out
            .transpose(1, 2)?
            .reshape((b, l, h * d))?
            .to_dtype(DType::BF16)?;
        layer.o_proj.forward(&out)
    }

    fn mlp(&self, layer: &GemmaLayer, x: &Tensor) -> Result<Tensor> {
        let gate = layer.gate_proj.forward(x)?.gelu()?; // tanh-approx gelu
        let up = layer.up_proj.forward(x)?;
        layer.down_proj.forward(&(gate * up)?)
    }

    fn layer_forward(
        &self,
        layer: &GemmaLayer,
        x: &Tensor,
        mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let r = self.attn(layer, &self.rms(x, &layer.input_ln)?, mask, cos, sin)?;
        let h = (x + self.rms(&r, &layer.post_attn_ln)?)?;
        let r = self.mlp(layer, &self.rms(&h, &layer.pre_ff_ln)?)?;
        &h + self.rms(&r, &layer.post_ff_ln)?
    }

    /// Additive causal + left-padding mask `[1,1,L,L]` f32. `valid(i,j) = j<=i && mask01[j]`.
    fn causal_padding_mask(&self, mask01: &[u32], l: usize) -> Result<Tensor> {
        let mut data = vec![0f32; l * l];
        for i in 0..l {
            for j in 0..l {
                let valid = j <= i && mask01[j] != 0;
                data[i * l + j] = if valid { 0.0 } else { MASK_NEG };
            }
        }
        Tensor::from_vec(data, (1, 1, l, l), &self.device)
    }

    /// Run the encoder over `input_ids` `[1,L]` (u32) + `mask01` (1 for valid, left-padded) → the
    /// **49 hidden states** `[1,L,3840]` (bf16).
    pub fn forward(&self, input_ids: &Tensor, mask01: &[u32]) -> Result<Vec<Tensor>> {
        let (b, l) = input_ids.dims2()?;
        let ids = input_ids.reshape((b * l,))?;
        let mut h = self
            .embed
            .index_select(&ids, 0)?
            .reshape((b, l, self.cfg.hidden_size))?;
        h = h.broadcast_mul(&self.embed_scale)?;

        let mask = self.causal_padding_mask(mask01, l)?;
        // Two RoPE tables (local / global base); pick per layer.
        let (cos_l, sin_l) = self.rope_tables(l, self.cfg.rope_theta_local)?;
        let (cos_g, sin_g) = self.rope_tables(l, self.cfg.rope_theta_global)?;

        let mut hiddens = Vec::with_capacity(self.cfg.num_layers + 1);
        hiddens.push(h.clone());
        for (i, layer) in self.layers.iter().enumerate() {
            let (cos, sin) = if self.cfg.is_global_layer(i) {
                (&cos_g, &sin_g)
            } else {
                (&cos_l, &sin_l)
            };
            h = self.layer_forward(layer, &h, &mask, cos, sin)?;
            if i < self.cfg.num_layers - 1 {
                hiddens.push(h.clone());
            }
        }
        hiddens.push(self.rms(&h, &self.norm)?);
        Ok(hiddens)
    }
}

/// sc-18763: pin exactly which Gemma hidden states [`GemmaEncoder::forward`] returns — the count
/// and which slot carries the final norm. The LTX caption feature extractor concatenates ALL
/// returned states (`flat_dim = hidden_size * (num_layers+1)`, `text_encoder.rs`'s
/// `normed_hidden`); an off-by-one here changes every downstream number by a small amount a smoke
/// render would not catch. Synthetic tiny weights only — no real Gemma checkpoint needed. Port of
/// the equivalent mlx-gen-ltx `gemma.rs` test module.
///
/// Requires an accelerator: [`GemmaEncoder`] runs bf16 unconditionally (matching the reference),
/// and candle's plain CPU backend has no bf16 matmul (`gemm-bf16` is not part of this candle
/// fork's CPU gemm surface) — the same reason `tests/conformance.rs` is `cuda`-only. `device()`
/// below prefers `cuda`, else `metal` (both support bf16 matmul; this Mac exercises the `metal`
/// arm). Neither feature is in this crate's `default` set, so `cargo test` with no `--features`
/// compiles this module out entirely rather than failing on an unsupported dtype.
#[cfg(all(test, any(feature = "cuda", feature = "metal")))]
mod hidden_state_pinning_tests {
    use super::*;
    use std::collections::HashMap;

    fn device() -> Device {
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(0).expect("cuda device")
        }
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        {
            Device::new_metal(0).expect("metal device")
        }
    }

    fn tiny_cfg(num_layers: usize) -> GemmaConfig {
        GemmaConfig {
            num_layers,
            hidden_size: 8,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            intermediate_size: 8,
            rms_eps: 1e-6,
            rope_theta_global: 1_000_000.0,
            rope_theta_local: 10_000.0,
            sliding_window_pattern: 6,
            query_pre_attn_scalar: 4.0,
            vocab_size: 6,
        }
    }

    /// Deterministic, non-degenerate (non-zero, non-uniform) fill so RMSNorm/attention behave
    /// generically rather than hitting a zero-weight special case.
    fn fill(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + seed) * 0.037).sin() * 0.1)
            .collect()
    }

    fn put(
        m: &mut HashMap<String, Tensor>,
        dev: &Device,
        key: &str,
        shape: (usize, usize),
        seed: f32,
    ) {
        let n = shape.0 * shape.1;
        let t = Tensor::from_vec(fill(n, seed), shape, dev).unwrap();
        m.insert(key.to_string(), t);
    }

    fn put1(m: &mut HashMap<String, Tensor>, dev: &Device, key: &str, n: usize, seed: f32) {
        let t = Tensor::from_vec(fill(n, seed), n, dev).unwrap();
        m.insert(key.to_string(), t);
    }

    /// A tiny, fully-synthetic Gemma weight set covering every tensor `GemmaEncoder::new` requires.
    fn tiny_vb(cfg: &GemmaConfig, seed: f32, dev: &Device) -> VarBuilder<'static> {
        let mut m: HashMap<String, Tensor> = HashMap::new();
        put(
            &mut m,
            dev,
            "embed_tokens.weight",
            (cfg.vocab_size, cfg.hidden_size),
            seed,
        );
        for i in 0..cfg.num_layers {
            let b = format!("layers.{i}.");
            let s = seed + i as f32;
            put1(
                &mut m,
                dev,
                &format!("{b}input_layernorm.weight"),
                cfg.hidden_size,
                s + 1.0,
            );
            put1(
                &mut m,
                dev,
                &format!("{b}post_attention_layernorm.weight"),
                cfg.hidden_size,
                s + 2.0,
            );
            put1(
                &mut m,
                dev,
                &format!("{b}pre_feedforward_layernorm.weight"),
                cfg.hidden_size,
                s + 3.0,
            );
            put1(
                &mut m,
                dev,
                &format!("{b}post_feedforward_layernorm.weight"),
                cfg.hidden_size,
                s + 4.0,
            );
            put(
                &mut m,
                dev,
                &format!("{b}self_attn.q_proj.weight"),
                (cfg.num_heads * cfg.head_dim, cfg.hidden_size),
                s + 5.0,
            );
            put(
                &mut m,
                dev,
                &format!("{b}self_attn.k_proj.weight"),
                (cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size),
                s + 6.0,
            );
            put(
                &mut m,
                dev,
                &format!("{b}self_attn.v_proj.weight"),
                (cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size),
                s + 7.0,
            );
            put(
                &mut m,
                dev,
                &format!("{b}self_attn.o_proj.weight"),
                (cfg.hidden_size, cfg.num_heads * cfg.head_dim),
                s + 8.0,
            );
            put1(
                &mut m,
                dev,
                &format!("{b}self_attn.q_norm.weight"),
                cfg.head_dim,
                s + 9.0,
            );
            put1(
                &mut m,
                dev,
                &format!("{b}self_attn.k_norm.weight"),
                cfg.head_dim,
                s + 10.0,
            );
            put(
                &mut m,
                dev,
                &format!("{b}mlp.gate_proj.weight"),
                (cfg.intermediate_size, cfg.hidden_size),
                s + 11.0,
            );
            put(
                &mut m,
                dev,
                &format!("{b}mlp.up_proj.weight"),
                (cfg.intermediate_size, cfg.hidden_size),
                s + 12.0,
            );
            put(
                &mut m,
                dev,
                &format!("{b}mlp.down_proj.weight"),
                (cfg.hidden_size, cfg.intermediate_size),
                s + 13.0,
            );
        }
        put1(&mut m, dev, "norm.weight", cfg.hidden_size, seed + 100.0);
        // BF16, matching production (the Gemma dense arm always casts to `vb.dtype()` = bf16); the
        // rest of `GemmaEncoder` hardcodes bf16 for the embedding/embed_scale/norm regardless.
        VarBuilder::from_tensors(m, DType::BF16, dev)
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.to_dtype(DType::F32).unwrap();
        let b = b.to_dtype(DType::F32).unwrap();
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn forward_returns_exactly_num_layers_plus_one_hidden_states() {
        let dev = device();
        for num_layers in [1usize, 2, 3, 5] {
            let cfg = tiny_cfg(num_layers);
            let vb = tiny_vb(&cfg, 0.0, &dev);
            let model = GemmaEncoder::new(vb, &cfg).unwrap();
            let ids = Tensor::from_vec(vec![0u32, 1, 2], (1, 3), &dev).unwrap();
            let hiddens = model.forward(&ids, &[1, 1, 1]).unwrap();
            assert_eq!(
                hiddens.len(),
                num_layers + 1,
                "num_layers={num_layers}: the LTX feature extractor concatenates ALL returned \
                 hidden states (flat_dim = hidden_size · (num_layers+1)); an off-by-one here \
                 silently drops or duplicates a layer's contribution with no visible failure in a \
                 smoke render"
            );
        }
    }

    #[test]
    fn hidden_state_zero_is_exactly_the_scaled_embedding_lookup() {
        let dev = device();
        let cfg = tiny_cfg(2);
        let vb = tiny_vb(&cfg, 0.0, &dev);
        let model = GemmaEncoder::new(vb, &cfg).unwrap();
        let ids = Tensor::from_vec(vec![3u32, 1], (1, 2), &dev).unwrap();
        let hiddens = model.forward(&ids, &[1, 1]).unwrap();

        let want_ids = Tensor::from_vec(vec![3u32, 1], (2,), &dev).unwrap();
        let embed_rows = model.embed.index_select(&want_ids, 0).unwrap();
        let want = embed_rows
            .broadcast_mul(&model.embed_scale)
            .unwrap()
            .reshape((1, 2, cfg.hidden_size))
            .unwrap();
        assert!(
            max_abs_diff(&hiddens[0], &want) < 5e-3,
            "hidden_states[0] must be exactly the scaled embedding lookup (index 0 pinning)"
        );
    }

    #[test]
    fn only_the_last_hidden_state_carries_the_final_norm() {
        // Two models identical except `norm.weight` (the FINAL norm, applied once after the last
        // layer). If it's wired to exactly one slot (the last), perturbing it must change
        // `hiddens.last()` and MUST NOT change any earlier slot.
        let dev = device();
        let cfg = tiny_cfg(2);
        let ids = Tensor::from_vec(vec![0u32, 1], (1, 2), &dev).unwrap();

        let vb_a = tiny_vb(&cfg, 0.0, &dev);
        let model_a = GemmaEncoder::new(vb_a, &cfg).unwrap();
        let hiddens_a = model_a.forward(&ids, &[1, 1]).unwrap();

        // Rebuild with the SAME seed, then hand-perturb only `norm.weight` on the resulting model
        // (bypassing the loader so nothing else can change).
        let vb_b = tiny_vb(&cfg, 0.0, &dev);
        let mut model_b = GemmaEncoder::new(vb_b, &cfg).unwrap();
        model_b.norm = Tensor::from_vec(vec![9.0f32; cfg.hidden_size], cfg.hidden_size, &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let hiddens_b = model_b.forward(&ids, &[1, 1]).unwrap();

        assert_eq!(hiddens_a.len(), hiddens_b.len());
        let last = hiddens_a.len() - 1;
        for i in 0..last {
            assert!(
                max_abs_diff(&hiddens_a[i], &hiddens_b[i]) < 1e-6,
                "perturbing the FINAL norm changed hidden_states[{i}], which is not the final index"
            );
        }
        assert!(
            max_abs_diff(&hiddens_a[last], &hiddens_b[last]) > 1e-3,
            "perturbing the FINAL norm must change the LAST hidden state (the final-norm slot)"
        );
    }
}
