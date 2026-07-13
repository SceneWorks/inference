//! SigLIP vision tower (story 7262 — the Candle port of mlx-llm's 7157 tower).
//!
//! LLaVA-family VLMs (JoyCaption, the llava-* checkpoints) use a SigLIP image encoder: a patch conv
//! embedding + a learned position embedding, a stack of pre-norm transformer encoder layers
//! (bias-ful QKV, `gelu_pytorch_tanh` MLP), and a final post-layernorm. The decoder reads a chosen
//! intermediate hidden state (JoyCaption: layer `-2`, all patch tokens), so [`SiglipVisionTower::forward`]
//! returns the HF-style `hidden_states` list (embeddings + one per layer) in addition to the
//! post-normed `last_hidden_state`.
//!
//! The tower runs in **f32** against the f32 preprocessed pixels (the weights are cast to f32 on
//! load), so the vision features are dtype-stable before the projector casts them into the bf16/f32
//! decoder — matching the reference engine, which promotes the bf16 vision weights to f32.
//!
//! [`SiglipVisionConfig::default`] is the `so400m-patch14-384` geometry; nothing here is
//! VLM-specific (the feature-layer choice lives with the VLM in [`crate::llava`]).

use candle_core::{DType, Tensor};

use crate::error::{Error, Result};
use crate::primitives::attention::AttnMask;
use crate::primitives::nn::{conv2d, gelu, layer_norm, linear};
use crate::primitives::{sdpa, Weights};

/// Geometry of a SigLIP vision tower.
#[derive(Clone, Copy, Debug)]
pub struct SiglipVisionConfig {
    /// Square input edge in pixels.
    pub image_size: usize,
    /// Patch (and conv kernel/stride) size.
    pub patch_size: usize,
    /// Input channels (3 = RGB).
    pub num_channels: usize,
    /// Hidden width.
    pub hidden_size: usize,
    /// MLP inner width.
    pub intermediate_size: usize,
    /// Number of encoder layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// LayerNorm epsilon.
    pub layer_norm_eps: f64,
}

impl Default for SiglipVisionConfig {
    /// `siglip2-so400m-patch14-384` (the JoyCaption tower).
    fn default() -> Self {
        Self {
            image_size: 384,
            patch_size: 14,
            num_channels: 3,
            hidden_size: 1152,
            intermediate_size: 4304,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            layer_norm_eps: 1e-6,
        }
    }
}

impl SiglipVisionConfig {
    /// Parse the `vision_config` object of a LLaVA `config.json`, falling back to the so400m defaults
    /// for any absent field.
    pub fn from_json(v: &serde_json::Value) -> Self {
        let d = Self::default();
        let u = |key: &str, fallback: usize| -> usize {
            v.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(fallback)
        };
        Self {
            image_size: u("image_size", d.image_size),
            patch_size: u("patch_size", d.patch_size),
            num_channels: u("num_channels", d.num_channels),
            hidden_size: u("hidden_size", d.hidden_size),
            intermediate_size: u("intermediate_size", d.intermediate_size),
            num_hidden_layers: u("num_hidden_layers", d.num_hidden_layers),
            num_attention_heads: u("num_attention_heads", d.num_attention_heads),
            layer_norm_eps: v
                .get("layer_norm_eps")
                .and_then(|x| x.as_f64())
                .unwrap_or(d.layer_norm_eps),
        }
    }

    /// Per-head dimension.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Patches per side (`image_size / patch_size`).
    pub fn grid(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Total patch tokens (`grid²`).
    pub fn num_patches(&self) -> usize {
        self.grid() * self.grid()
    }
}

/// Output of the vision tower.
pub struct SiglipVisionOutput {
    /// `[b, num_patches, hidden]` after the final post-layernorm.
    pub last_hidden_state: Tensor,
    /// HF-style hidden states: the embeddings output followed by one output per encoder layer
    /// (before post-layernorm). Length is `num_hidden_layers + 1`.
    pub hidden_states: Vec<Tensor>,
}

/// A loaded SigLIP vision tower (weights in f32).
pub struct SiglipVisionTower {
    patch_embedding: Tensor,
    patch_bias: Option<Tensor>,
    position_embedding: Tensor,
    layers: Vec<SiglipEncoderLayer>,
    post_ln_w: Tensor,
    post_ln_b: Tensor,
    cfg: SiglipVisionConfig,
}

impl SiglipVisionTower {
    /// Load from a checkpoint. `prefix` points at the HF `vision_model` module — e.g.
    /// `vision_tower.vision_model` for a LLaVA checkpoint. All weights are cast to f32 (the tower
    /// runs in f32 for numeric fidelity, then the projector casts the features into the decoder).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: SiglipVisionConfig) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        let f32w = |key: &str| -> Result<Tensor> { Ok(w.require(key)?.to_dtype(DType::F32)?) };
        let f32_opt = |key: &str| -> Result<Option<Tensor>> {
            match w.get(key) {
                Some(t) => Ok(Some(t.to_dtype(DType::F32)?)),
                None => Ok(None),
            }
        };
        // HF stores the patch conv `[out, in, kH, kW]` — exactly Candle's `conv2d` kernel layout.
        let patch_embedding = f32w(&p("embeddings.patch_embedding.weight"))?;
        let patch_bias = f32_opt(&p("embeddings.patch_embedding.bias"))?;
        let position_embedding = f32w(&p("embeddings.position_embedding.weight"))?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| SiglipEncoderLayer::from_weights(w, &p(&format!("encoder.layers.{i}")), &cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embedding,
            patch_bias,
            position_embedding,
            layers,
            post_ln_w: f32w(&p("post_layernorm.weight"))?,
            post_ln_b: f32w(&p("post_layernorm.bias"))?,
            cfg,
        })
    }

    /// Patch + position embeddings of preprocessed NCHW `pixel_values` → `[b, num_patches, hidden]`.
    pub fn embeddings(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let b = pixel_values.dim(0)?;
        let np = self.cfg.num_patches();
        let h = self.cfg.hidden_size;
        // [b, 3, S, S] -> [b, hidden, grid, grid] -> [b, hidden, num_patches] -> [b, num_patches, hidden].
        let patches = conv2d(
            pixel_values,
            &self.patch_embedding,
            self.patch_bias.as_ref(),
            self.cfg.patch_size,
            0,
        )?;
        let patches = patches.reshape((b, h, np))?.transpose(1, 2)?.contiguous()?;
        let pos = self.position_embedding.reshape((1, np, h))?;
        Ok(patches.broadcast_add(&pos)?)
    }

    /// Run the tower over preprocessed NCHW `pixel_values`, collecting the per-layer hidden states.
    pub fn forward(&self, pixel_values: &Tensor) -> Result<SiglipVisionOutput> {
        let mut hidden = self.embeddings(pixel_values)?;
        let mut hidden_states = Vec::with_capacity(self.layers.len() + 1);
        hidden_states.push(hidden.clone());
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
            hidden_states.push(hidden.clone());
        }
        let last_hidden_state = layer_norm(
            &hidden,
            &self.post_ln_w,
            &self.post_ln_b,
            self.cfg.layer_norm_eps,
        )?;
        Ok(SiglipVisionOutput {
            last_hidden_state,
            hidden_states,
        })
    }

    /// The tower geometry.
    pub fn config(&self) -> &SiglipVisionConfig {
        &self.cfg
    }
}

/// Select a hidden state from a [`SiglipVisionOutput`] by HF-style index (negatives count from the
/// end; `-2` = the penultimate state = the layer the LLaVA decoder reads).
pub fn select_vision_feature(output: &SiglipVisionOutput, layer: i32) -> Result<Tensor> {
    let len = output.hidden_states.len() as i32;
    let idx = if layer < 0 { len + layer } else { layer };
    if idx < 0 || idx >= len {
        return Err(Error::Msg(format!(
            "siglip: vision feature layer {layer} out of range for {len} hidden states"
        )));
    }
    Ok(output.hidden_states[idx as usize].clone())
}

struct SiglipEncoderLayer {
    ln1_w: Tensor,
    ln1_b: Tensor,
    ln2_w: Tensor,
    ln2_b: Tensor,
    attn: SiglipAttention,
    mlp: SiglipMlp,
    eps: f64,
}

impl SiglipEncoderLayer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &SiglipVisionConfig) -> Result<Self> {
        let f32w = |leaf: &str| -> Result<Tensor> {
            Ok(w.require(&join(prefix, leaf))?.to_dtype(DType::F32)?)
        };
        Ok(Self {
            ln1_w: f32w("layer_norm1.weight")?,
            ln1_b: f32w("layer_norm1.bias")?,
            ln2_w: f32w("layer_norm2.weight")?,
            ln2_b: f32w("layer_norm2.bias")?,
            attn: SiglipAttention::from_weights(w, &join(prefix, "self_attn"), cfg)?,
            mlp: SiglipMlp::from_weights(w, &join(prefix, "mlp"))?,
            eps: cfg.layer_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = layer_norm(x, &self.ln1_w, &self.ln1_b, self.eps)?;
        let x = (x + self.attn.forward(&y)?)?;
        let y = layer_norm(&x, &self.ln2_w, &self.ln2_b, self.eps)?;
        Ok((&x + self.mlp.forward(&y)?)?)
    }
}

struct SiglipAttention {
    q_w: Tensor,
    q_b: Option<Tensor>,
    k_w: Tensor,
    k_b: Option<Tensor>,
    v_w: Tensor,
    v_b: Option<Tensor>,
    out_w: Tensor,
    out_b: Option<Tensor>,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl SiglipAttention {
    fn from_weights(w: &Weights, prefix: &str, cfg: &SiglipVisionConfig) -> Result<Self> {
        let f32w = |leaf: &str| -> Result<Tensor> {
            Ok(w.require(&join(prefix, leaf))?.to_dtype(DType::F32)?)
        };
        let bias = |leaf: &str| -> Result<Option<Tensor>> {
            match w.get(&join(prefix, leaf)) {
                Some(t) => Ok(Some(t.to_dtype(DType::F32)?)),
                None => Ok(None),
            }
        };
        let head_dim = cfg.head_dim();
        Ok(Self {
            q_w: f32w("q_proj.weight")?,
            q_b: bias("q_proj.bias")?,
            k_w: f32w("k_proj.weight")?,
            k_b: bias("k_proj.bias")?,
            v_w: f32w("v_proj.weight")?,
            v_b: bias("v_proj.bias")?,
            out_w: f32w("out_proj.weight")?,
            out_b: bias("out_proj.bias")?,
            num_heads: cfg.num_attention_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let to_heads = |a: Tensor| -> Result<Tensor> {
            Ok(a.reshape((b, n, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let q = to_heads(linear(x, &self.q_w, self.q_b.as_ref())?)?;
        let k = to_heads(linear(x, &self.k_w, self.k_b.as_ref())?)?;
        let v = to_heads(linear(x, &self.v_w, self.v_b.as_ref())?)?;
        // SigLIP attention is fully bidirectional — no mask, no softcap.
        let out = sdpa(&q, &k, &v, self.scale, None, AttnMask::None)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, n, self.num_heads * self.head_dim))?;
        linear(&out, &self.out_w, self.out_b.as_ref())
    }
}

struct SiglipMlp {
    fc1_w: Tensor,
    fc1_b: Option<Tensor>,
    fc2_w: Tensor,
    fc2_b: Option<Tensor>,
}

impl SiglipMlp {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let f32w = |leaf: &str| -> Result<Tensor> {
            Ok(w.require(&join(prefix, leaf))?.to_dtype(DType::F32)?)
        };
        let bias = |leaf: &str| -> Result<Option<Tensor>> {
            match w.get(&join(prefix, leaf)) {
                Some(t) => Ok(Some(t.to_dtype(DType::F32)?)),
                None => Ok(None),
            }
        };
        Ok(Self {
            fc1_w: f32w("fc1.weight")?,
            fc1_b: bias("fc1.bias")?,
            fc2_w: f32w("fc2.weight")?,
            fc2_b: bias("fc2.bias")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = linear(x, &self.fc1_w, self.fc1_b.as_ref())?;
        let x = gelu(&x)?; // gelu_pytorch_tanh
        linear(&x, &self.fc2_w, self.fc2_b.as_ref())
    }
}

/// Join a weight-key `prefix` and `leaf` with `.` (no leading dot when empty).
fn join(prefix: &str, leaf: &str) -> String {
    if prefix.is_empty() {
        leaf.to_owned()
    } else {
        format!("{prefix}.{leaf}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn so400m_config_geometry() {
        let cfg = SiglipVisionConfig::default();
        assert_eq!(cfg.num_patches(), 729); // 27*27
        assert_eq!(cfg.grid(), 27);
        assert_eq!(cfg.head_dim(), 72); // 1152/16
    }

    #[test]
    fn config_from_json_overrides_then_defaults() {
        let v: serde_json::Value =
            serde_json::from_str(r#"{"image_size": 256, "patch_size": 16}"#).unwrap();
        let cfg = SiglipVisionConfig::from_json(&v);
        assert_eq!(cfg.image_size, 256);
        assert_eq!(cfg.patch_size, 16);
        assert_eq!(cfg.grid(), 16);
        // Absent fields keep the so400m defaults.
        assert_eq!(cfg.hidden_size, 1152);
        assert_eq!(cfg.num_hidden_layers, 27);
    }

    #[test]
    fn feature_layer_negative_index() {
        use candle_core::Device;
        let mk = |v: f32| Tensor::from_vec(vec![v], (1, 1, 1), &Device::Cpu).unwrap();
        let hs = vec![mk(1.0), mk(2.0), mk(3.0)];
        let out = SiglipVisionOutput {
            last_hidden_state: hs[2].clone(),
            hidden_states: hs,
        };
        let val = |t: Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert_eq!(val(select_vision_feature(&out, -2).unwrap()), 2.0);
        assert!(select_vision_feature(&out, -4).is_err());
        assert_eq!(val(select_vision_feature(&out, 0).unwrap()), 1.0);
    }
}
