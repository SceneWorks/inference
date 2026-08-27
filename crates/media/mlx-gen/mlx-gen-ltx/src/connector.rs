//! `Embeddings1DConnector` — the LTX-2.3 text-feature connector (S1).
//!
//! Port of `text_encoder.py`'s `Embeddings1DConnector` as configured for the LTX-2.3 models
//! (`connector.safetensors`): an **8-layer** pre-norm transformer over the Gemma feature-extractor
//! output, dim **4096** (32 heads × 128), **gated** attention (`2·sigmoid`, zero-init identity)
//! with q/k RMSNorm, a tanh-GELU MLP (inner 16384), **128 learnable registers** that replace
//! left-padding, and a connector-specific **1-D SPLIT RoPE** (positions `arange(seq)/4096`,
//! double-precision). The semantic authority is `ltx_core`'s `Embeddings1DConnector` (the stack
//! the checkpoints were trained with) — NOT mlx_video's port, whose connector drops the gate's
//! `2·` and uses exact GELU (sc-21663; both fixed here for 2.3 and 2.5 alike).
//!
//! Two connectors exist in the checkpoint (`video_embeddings_connector.*`,
//! `audio_embeddings_connector.*`); this core uses the video one. Compute dtype is a parameter:
//! **bf16** to match the reference pipeline end-to-end, **f32** for the isolated bit-exact gate.
//! The fused SDPA is always run in **f32** regardless — the pmetal bf16 maskless-SDPA kernel
//! returns garbage at this shape (see `tests/bf16_sdpa_bug.rs`); the reference's wheel MLX has a
//! correct bf16 SDPA, so f32 matches it to bf16 rounding.

use std::f64::consts::PI;

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, multiply, sigmoid, sum, tile};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::gelu_tanh;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::LtxConfig;
use crate::rope::apply_split_rotary_emb;
use crate::transformer::{Linear, Precision};

const CONNECTOR_EPS: f32 = 1e-6;

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// One connector transformer block (attn1 + gelu FF, both pre-normed with a unit-weight RMSNorm).
///
/// Every projection is a [`Linear`], the same dense-or-quantized carrier the DiT uses: an LTX-2.5
/// `q4`/`q8` tier packs the connector's attention and FFN Linears (sc-18775 — 4.03 GB dense across
/// both towers, 21 % of a q4 tier), and a `Linear` binds the packed `weight`/`scales`/`biases`
/// triple or a dense weight from the **same** call, selected by whether `{prefix}.scales` is in the
/// checkpoint. LTX-2.3's dense `connector.safetensors` therefore takes the identical path it always
/// did.
///
/// `ff_in`/`ff_out` carry no bias when `connector_ff_bias:false` (sc-18758 — reference
/// `Embeddings1DConnectorConfigurator`/`AudioEmbeddings1DConnectorConfigurator`, independent of the
/// DiT's own `ff_bias`); neither shipped checkpoint sets it, so a bias is present in practice today.
struct ConnectorBlock {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    q_norm_w: Array,
    k_norm_w: Array,
    gate: Linear,
    ff_in: Linear,
    ff_out: Linear,
}

/// The video text-feature connector.
pub struct Connector {
    blocks: Vec<ConnectorBlock>,
    registers: Array, // (num_registers, dim)
    num_heads: i32,
    head_dim: i32,
    theta: f64,
    max_pos: i32,
    ones: Array, // unit RMSNorm weight (dim,)
    /// Activation compute dtype — **always f32** (sc-21663). With the correct `2·sigmoid` gates
    /// the 8-layer stack is expansive (~2x/layer): per-op rounding differences are amplified
    /// ~50-500x at the output, so bf16 activations cost ~5e-2 peak-rel of conditioning fidelity
    /// against the f32 reference (measured: 5.57e-2 bf16 vs 1.29e-2 f32 on the LTX-2.5 video
    /// golden). The connector is 256 tokens × 8 layers — f32 activations are numerically decisive
    /// and computationally free next to the TE/DiT; weights stay at the tier's own dtype/packing.
    /// Same precedent as the f32 SDPA below.
    dtype: Dtype,
    /// The pipeline dtype the output is returned in (`prec.dtype()`), so downstream consumers see
    /// exactly the interface they always did.
    out_dtype: Dtype,
}

impl Connector {
    /// Build the **video** connector from a `Weights` map (e.g. `connector.safetensors`) under
    /// `prefix` (`"video_embeddings_connector."`).
    ///
    /// `prec` supplies both the compute dtype (bf16 to match the reference pipeline end-to-end; f32
    /// for the isolated bit-exact gate) **and** the checkpoint's quant geometry, exactly as it does
    /// for the DiT. Whether any given Linear is actually packed is decided per-Linear by the
    /// presence of `{prefix}.scales`, so a dense LTX-2.3 connector and a quantized LTX-2.5 tier
    /// connector both load through this one call.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &LtxConfig,
        prec: Precision,
    ) -> Result<Self> {
        Self::from_weights_dims(
            w,
            prefix,
            cfg.connector_num_layers,
            cfg.connector_num_attention_heads,
            cfg.connector_attention_head_dim,
            cfg.positional_embedding_theta,
            cfg.connector_positional_embedding_max_pos,
            cfg.connector_ff_bias,
            prec,
        )
    }

    /// Build a connector with **explicit** dims — used for both the video connector (32×128) and the
    /// audio connector (32×64, sc-2684), which share the checkpoint's layer count / theta / max_pos
    /// but differ in `head_dim` (hence `dim`).
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights_dims(
        w: &Weights,
        prefix: &str,
        num_layers: i32,
        num_heads: i32,
        head_dim: i32,
        theta: f64,
        max_pos: i32,
        ff_bias: bool,
        prec: Precision,
    ) -> Result<Self> {
        let n = num_layers as usize;
        let dim = num_heads * head_dim;
        // Activations always compute in f32 (see the `dtype` field doc); the pipeline's own dtype
        // is only the output interface.
        let dtype = Dtype::Float32;
        let out_dtype = prec.dtype();
        // A non-Linear parameter (norm weight, registers) cast to the compute dtype.
        let w_at_dtype = |key: &str| -> Result<Array> {
            w.get(key)
                .ok_or_else(|| Error::MissingTensor(key.into()))?
                .as_dtype(dtype)
                .map_err(Error::from)
        };
        let mut blocks = Vec::with_capacity(n);
        for i in 0..n {
            let b = format!("{prefix}transformer_1d_blocks.{i}.");
            blocks.push(ConnectorBlock {
                to_q: Linear::load(w, &format!("{b}attn1.to_q"), prec)?,
                to_k: Linear::load(w, &format!("{b}attn1.to_k"), prec)?,
                to_v: Linear::load(w, &format!("{b}attn1.to_v"), prec)?,
                to_out: Linear::load(w, &format!("{b}attn1.to_out.0"), prec)?,
                q_norm_w: w_at_dtype(&format!("{b}attn1.q_norm.weight"))?,
                k_norm_w: w_at_dtype(&format!("{b}attn1.k_norm.weight"))?,
                gate: Linear::load(w, &format!("{b}attn1.to_gate_logits"), prec)?,
                // The `ff.net.{0.proj,2}.bias` tensors are absent when `connector_ff_bias:false`
                // (sc-18758); `has_bias:false` must not `require` a tensor that was never shipped.
                ff_in: Linear::load_with_bias(w, &format!("{b}ff.net.0.proj"), prec, ff_bias)?,
                ff_out: Linear::load_with_bias(w, &format!("{b}ff.net.2"), prec, ff_bias)?,
            });
        }
        let registers = w_at_dtype(&format!("{prefix}learnable_registers"))?;
        Ok(Self {
            blocks,
            registers,
            num_heads,
            head_dim,
            theta,
            max_pos,
            ones: Array::ones::<f32>(&[dim])?.as_dtype(dtype)?,
            dtype,
            out_dtype,
        })
    }

    /// Connector-specific 1-D SPLIT RoPE (double-precision): positions `arange(seq)`, scaled by
    /// `max_pos`. Returns `(cos, sin)`, each `(1, num_heads, seq, head_dim/2)`.
    fn rope(&self, seq: usize) -> Result<(Array, Array)> {
        let heads = self.num_heads as usize;
        let head_half = (self.head_dim / 2) as usize;
        let dim = heads * (self.head_dim as usize);
        let n_elem = 2usize; // 2 * len([max_pos])
        let num_indices = dim / n_elem; // 2048 (= heads * head_half, no padding)
        let step = if num_indices == 1 {
            0.0
        } else {
            1.0 / (num_indices - 1) as f64
        };
        // f64 exponentials rounded to f32 BEFORE the position multiply — exactly upstream's
        // `generate_freq_grid_np` (its "double precision" covers only the log-spaced grid; the
        // returned indices are f32, and `generate_freqs` forms the angles in f32). Keeping f64
        // through `cos`/`sin` — what mlx_video and this port used to do — perturbs the top
        // frequencies by ~1e-3 rad; the connector's 2·sigmoid gates amplify that identical
        // per-layer table delta coherently (~2×/layer over 8 layers), which alone holds the
        // parity gate at 1.3e-2 video / 8.8e-2 audio vs the ltx_core golden (sc-21663).
        let indices: Vec<f32> = (0..num_indices)
            .map(|i| (self.theta.powf(i as f64 * step) * (PI / 2.0)) as f32)
            .collect();

        let mut cos = vec![0f32; heads * seq * head_half];
        let mut sin = vec![0f32; heads * seq * head_half];
        for t in 0..seq {
            let scaled = ((t as f64 / self.max_pos as f64) * 2.0 - 1.0) as f32;
            for h in 0..heads {
                for p in 0..head_half {
                    let ang = scaled * indices[h * head_half + p];
                    let o = (h * seq + t) * head_half + p;
                    cos[o] = ang.cos();
                    sin[o] = ang.sin();
                }
            }
        }
        let shape = [1, heads as i32, seq as i32, head_half as i32];
        Ok((
            Array::from_slice(&cos, &shape),
            Array::from_slice(&sin, &shape),
        ))
    }

    /// Replace left-padding with learnable registers (batch=1). Valid tokens (the trailing
    /// `num_valid` of a left-padded sequence) move to the front; registers fill the tail.
    fn replace_with_registers(&self, x: &Array, mask01: &Array) -> Result<Array> {
        let sh = x.shape();
        let (s, dim) = (sh[1], sh[2]);
        let nv = sum(mask01, None)?.item::<i32>();
        let num_reg = self.registers.shape()[0];
        // A sequence shorter than the register block makes `num_tiles == 0`, so `tile` yields an
        // empty register grid and the `[1, s, dim]` reshape shape-errors. Surface it clearly (F-050).
        if num_reg == 0 || s < num_reg {
            return Err(Error::Msg(format!(
                "ltx connector: sequence length {s} is smaller than the register count {num_reg}"
            )));
        }
        // F-113: `tile(registers, [num_tiles, 1])` produces `(num_tiles·num_reg, dim)`; the following
        // `reshape([1, s, dim])` only succeeds when `s` is an exact multiple of `num_reg`. A
        // non-divisible length would otherwise reshape-error opaquely — reject it up front.
        if s % num_reg != 0 {
            return Err(Error::Msg(format!(
                "ltx connector: sequence length {s} is not a multiple of the register count {num_reg}"
            )));
        }
        let num_tiles = s / num_reg;
        let reg_full = tile(&self.registers, &[num_tiles, 1])? // (s, dim)
            .reshape(&[1, s, dim])?
            .as_dtype(x.dtype())?;
        if nv >= s {
            return Ok(x.clone());
        }
        let valid_idx: Vec<i32> = (s - nv..s).collect();
        let tail_idx: Vec<i32> = (nv..s).collect();
        let valid = x.take_axis(Array::from_slice(&valid_idx, &[valid_idx.len() as i32]), 1)?;
        let reg_tail =
            reg_full.take_axis(Array::from_slice(&tail_idx, &[tail_idx.len() as i32]), 1)?;
        Ok(concatenate_axis(&[&valid, &reg_tail], 1)?)
    }

    fn attn(&self, blk: &ConnectorBlock, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let (h, d) = (self.num_heads, self.head_dim);
        let q = rms_norm(&blk.to_q.forward(x)?, &blk.q_norm_w, CONNECTOR_EPS)?;
        let k = rms_norm(&blk.to_k.forward(x)?, &blk.k_norm_w, CONNECTOR_EPS)?;
        let v = blk.to_v.forward(x)?;
        let q = q.reshape(&[b, s, h, d])?.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, s, h, d])?.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, s, h, d])?.transpose_axes(&[0, 2, 1, 3])?;
        let q = apply_split_rotary_emb(&q, cos, sin)?;
        let k = apply_split_rotary_emb(&k, cos, sin)?;
        let scale = 1.0 / (d as f32).sqrt();
        // SDPA in f32: the pmetal bf16 fused-SDPA kernel returns garbage at this shape (mask=None,
        // 32 heads, head_dim 128) — a sibling of the bf16-GEMM bug, NOT fixed by sc-2714 (which
        // patched matmul.cpp only). The reference's wheel MLX has a correct bf16 SDPA, so an f32
        // SDPA matches it to bf16 rounding. (No-op when the connector already runs f32.) See
        // tests/bf16_sdpa_bug.rs.
        let out = scaled_dot_product_attention(
            &q.as_dtype(Dtype::Float32)?,
            &k.as_dtype(Dtype::Float32)?,
            &v.as_dtype(Dtype::Float32)?,
            scale,
            None,
            None,
        )?
        .as_dtype(self.dtype)?; // (b,h,s,d)
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, -1])?;
        // Gated: out_head *= 2·sigmoid(to_gate_logits(x)) — zero-init identity, the same
        // convention the DiT's own gated attention uses (`transformer.rs`). ltx_core's
        // `PytorchGatedAttention` is the authority; the `2·` was missing here because this port
        // was originally pinned against mlx_video's connector, which dropped it (sc-21663).
        let logits = blk.gate.forward(x)?;
        let gates = multiply(&sigmoid(&logits)?, scalar(2.0).as_dtype(logits.dtype())?)?
            .reshape(&[b, s, h, 1])?;
        let out = multiply(&out.reshape(&[b, s, h, d])?, &gates)?.reshape(&[b, s, -1])?;
        blk.to_out.forward(&out)
    }

    fn block(&self, blk: &ConnectorBlock, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let n = rms_norm(x, &self.ones, CONNECTOR_EPS)?;
        let x = add(x, &self.attn(blk, &n, cos, sin)?)?;
        let n = rms_norm(&x, &self.ones, CONNECTOR_EPS)?;
        // tanh-approximate GELU — ltx_core's `GELUApprox` (`gelu(x, approximate="tanh")`), the same
        // activation the DiT FFN uses; mlx_video's connector used exact erf-GELU (sc-21663).
        let ff = blk.ff_out.forward(&gelu_tanh(&blk.ff_in.forward(&n)?)?)?;
        Ok(add(&x, &ff)?)
    }

    /// Run the connector. `x` = `(1, seq, dim)` feature-extractor output (f32); `mask01` = `(1, seq)`
    /// 1/0 attention mask (1 = valid; left-padded). Returns video embeddings `(1, seq, dim)`.
    pub fn forward(&self, x: &Array, mask01: &Array) -> Result<Array> {
        let mut h = self.replace_with_registers(&x.as_dtype(self.dtype)?, mask01)?;
        let (cos, sin) = self.rope(h.shape()[1] as usize)?;
        for blk in &self.blocks {
            h = self.block(blk, &h, &cos, &sin)?;
        }
        // Back to the pipeline dtype at the interface (a no-op for the isolated f32 gates).
        Ok(rms_norm(&h, &self.ones, CONNECTOR_EPS)?.as_dtype(self.out_dtype)?)
    }
}
