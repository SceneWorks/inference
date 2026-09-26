//! The YuE2-3B Mixture-of-Transformers backbone, native on Candle (sc-22991).
//!
//! Ported from the pinned upstream `src/yue2/modeling_yue2.py` (Apache-2.0, commit
//! [`YUE2_SOURCE_COMMIT`](crate::inventory::YUE2_SOURCE_COMMIT)). Every decoder layer of the
//! released checkpoint carries **two** complete transformer paths over one residual stream:
//!
//! * the **AR** path — `input_layernorm`, `self_attn` (Q/K/V/O with per-head `q_norm`/`k_norm`),
//!   `post_attention_layernorm`, `mlp` (SwiGLU) — which the autoregressive score (ABC) and
//!   semantic-token stages run alone, over a causal KV cache; and
//! * the **NAR** path — the `nar_`-prefixed twins of the same four modules — which the acoustic
//!   flow-matching stage runs for its latent positions while attending the AR path's cached
//!   keys/values of the song prefix.
//!
//! Upstream's generation forward (`DecoderLayer.forward` with `ar_mask=None`) is the AR path only,
//! which is what [`Yue2Lm::prefill`] / [`Yue2Lm::decode`] run. [`MotPaths::ArAndNar`] also loads
//! the NAR twins (same [`Attention`] / [`Mlp`] types) for the acoustic stage to drive; with
//! [`MotPaths::Ar`] they are never read from the checkpoint.
//!
//! # Numerics
//!
//! The model runs in the dtype it was loaded in. The checkpoint is BF16; the upstream release runs
//! BF16 on an accelerator. Candle's CPU backend has no BF16 matmul, so a CPU load upcasts the BF16
//! weights to F32 (exactly representable) and computes in F32 — the same thing the F32 torch
//! reference does. Two leaves are written here rather than reused because their rounding sequence
//! is part of the released arithmetic in BF16:
//!
//! * [`rms_norm`] is upstream's `x * rsqrt(mean(x.float()²) + eps).to(x.dtype) * weight`: the
//!   reciprocal is rounded to the model dtype and then multiplied in the model dtype (two roundings),
//!   where `candle_llm::primitives::rms_norm` scales in F32 and rounds once.
//! * RoPE uses upstream's F32 angle table cast to the model dtype, which is exactly what
//!   `candle_llm::primitives::Rope::cos_sin` + `apply_rope` (reused) compute.
//!
//! Attention (`candle_llm::primitives::sdpa_gqa`, grouped-query, un-expanded K/V, bottom-right
//! causal) and the preallocated KV cache (`candle_llm::primitives::StaticKvCache`, fails before
//! writing past its capacity — upstream's `StaticKVCache`) are reused unchanged.
//!
//! # Positions and context
//!
//! RoPE positions are the physical cache slots `0, 1, 2, …` of the one sequence a cache holds —
//! upstream's `position_ids = cache_position` for an unpadded single request. There is no
//! windowing, shortening or re-prefill: a sequence either fits the cache it was given (at most
//! [`CONTEXT`](crate::protocol::CONTEXT) positions) or the step fails before writing (see
//! [`crate::generate`] for the request-level budget check).

use std::collections::BTreeMap;
use std::sync::Arc;

use candle_audio::candle_core::{DType, Device, Tensor, D};
use candle_audio::gen_core;
use candle_llm::primitives::{
    apply_rope, embed, sdpa_gqa, swiglu, AttnMask, KvCache, Rope, StaticKvCache,
};
use candle_nn::VarBuilder;
use serde_json::Value;

use crate::inventory::ComponentId;
use crate::precision::{Residency, Storage, Tier};
use crate::snapshot::{self, SnapshotDirs};
use crate::weights::{Loader, Proj};

/// Query positions per prefill forward. Bounds a long prefix's per-forward work so cancellation is
/// observed between chunks (see [`crate::generate`]); attention itself is additionally tiled by
/// `sdpa_gqa`. Chunking changes neither the visible key set nor any position (bottom-right causal
/// over the cache), so a chunked prefill computes the same function as one forward.
pub const PREFILL_CHUNK: usize = 512;

pub(crate) fn backend<E: std::error::Error + Send + Sync + 'static>(
    what: &'static str,
) -> impl Fn(E) -> gen_core::Error {
    move |e| gen_core::Error::Msg(format!("candle-audio-yue2 {what}: {e}"))
}

/// The YuE2 model configuration — upstream `YuE2Config`'s inference fields.
#[derive(Clone, Debug, PartialEq)]
pub struct Yue2Config {
    /// Residual width (2048).
    pub hidden_size: usize,
    /// Decoder layers (28).
    pub num_hidden_layers: usize,
    /// Query heads (16).
    pub num_attention_heads: usize,
    /// Key/value heads (8, grouped-query).
    pub num_key_value_heads: usize,
    /// Per-head width (128).
    pub head_dim: usize,
    /// SwiGLU inner width (6144).
    pub intermediate_size: usize,
    /// Vocabulary (184 704: text, ABC, protocol specials, 32 768 codec ids, latent markers).
    pub vocab_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// RoPE base.
    pub rope_theta: f64,
    /// Positions the model was trained for (24 576).
    pub max_position_embeddings: usize,
}

impl Yue2Config {
    /// Parse a YuE2 `config.json`. Every inference field is required; a checkpoint that is not the
    /// released YuE2 architecture (tied embeddings, a non-`vae` latent type, a head layout that
    /// does not group) is [`gen_core::Error::Unsupported`].
    pub fn from_json(text: &str) -> gen_core::Result<Self> {
        let v: Value = serde_json::from_str(text)
            .map_err(|e| gen_core::Error::Msg(format!("YuE2 config.json: {e}")))?;
        let uint = |key: &str| -> gen_core::Result<usize> {
            v.get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| gen_core::Error::Msg(format!("YuE2 config.json: missing `{key}`")))
        };
        let float = |key: &str| -> gen_core::Result<f64> {
            v.get(key)
                .and_then(Value::as_f64)
                .ok_or_else(|| gen_core::Error::Msg(format!("YuE2 config.json: missing `{key}`")))
        };
        if v.get("model_type").and_then(Value::as_str) != Some("yue2") {
            return Err(gen_core::Error::Unsupported(
                "config.json is not a YuE2 (`model_type: yue2`) checkpoint".into(),
            ));
        }
        if v.get("tie_word_embeddings").and_then(Value::as_bool) != Some(false) {
            return Err(gen_core::Error::Unsupported(
                "YuE2 inference requires an untied `lm_head` (`tie_word_embeddings: false`)".into(),
            ));
        }
        if v.get("latent_type").and_then(Value::as_str) != Some("vae") {
            return Err(gen_core::Error::Unsupported(
                "YuE2 inference supports only `latent_type: vae`".into(),
            ));
        }
        let config = Self {
            hidden_size: uint("hidden_size")?,
            num_hidden_layers: uint("num_hidden_layers")?,
            num_attention_heads: uint("num_attention_heads")?,
            num_key_value_heads: uint("num_key_value_heads")?,
            head_dim: uint("head_dim")?,
            intermediate_size: uint("intermediate_size")?,
            vocab_size: uint("vocab_size")?,
            rms_norm_eps: float("rms_norm_eps")?,
            rope_theta: float("rope_theta")?,
            max_position_embeddings: uint("max_position_embeddings")?,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> gen_core::Result<()> {
        let dims = [
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.intermediate_size,
            self.vocab_size,
            self.max_position_embeddings,
        ];
        if dims.contains(&0) || !self.head_dim.is_multiple_of(2) {
            return Err(gen_core::Error::Unsupported(format!(
                "YuE2 config has an empty dimension or an odd head width: {self:?}"
            )));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "YuE2 config: {} query heads do not group over {} key/value heads",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.vocab_size <= crate::sampling::VOCAB_MAX_PROTOCOL_ID as usize {
            return Err(gen_core::Error::Unsupported(format!(
                "YuE2 config: vocabulary of {} does not reach the protocol's token ids",
                self.vocab_size
            )));
        }
        Ok(())
    }
}

/// Refuse a compute dtype/device pair YuE2 cannot run **before** any verification or weight I/O:
/// the model computes in F32 or BF16 only, and Candle's CPU backend has no BF16 matmul (a CPU
/// load computes in F32 — the BF16 checkpoint upcasts exactly).
fn check_compute(dtype: DType, device: &Device) -> gen_core::Result<()> {
    match dtype {
        DType::F32 => Ok(()),
        DType::BF16 if !device.is_cpu() => Ok(()),
        DType::BF16 => Err(gen_core::Error::Unsupported(
            "YuE2 on the CPU computes in F32: Candle's CPU backend has no BF16 matmul".into(),
        )),
        other => Err(gen_core::Error::Unsupported(format!(
            "YuE2 computes in F32 or BF16 (the released precision), not {other:?}"
        ))),
    }
}

/// The verified YuE2-3B snapshot, opened for loading (see [`open_verified`]).
pub(crate) struct VerifiedLmFiles {
    /// The verified `config.json`, read.
    pub(crate) config_json: String,
    /// A loader over exactly the verified weights file, converting dense tensors to the compute
    /// dtype and reading the tier's GGML block tensors as stored.
    pub(crate) loader: Loader<'static>,
    /// SHA-256 of that weights file (checked against its pin or its tier manifest).
    pub(crate) weights_sha256: String,
    /// The tier the weights file holds.
    pub(crate) tier: Tier,
}

/// Refuse an unsupported dtype/device pair, then resolve and verify the YuE2-3B snapshot
/// **immediately before loading** (the crate's load-boundary rule) and open exactly the verified
/// `config.json` and weights file. Every loader of the MoT goes through here.
///
/// The `m-a-p/YuE2-3B` directory is either the pinned original (the `bf16` tier, verified against
/// its pins) or a derived tier snapshot ([`crate::tier`], verified with
/// [`crate::tier::verify_tier`]). `expect`, when set, is the tier the caller asserts; any other
/// staged tier is refused rather than loaded.
pub(crate) fn open_verified(
    dirs: &SnapshotDirs,
    dtype: DType,
    device: &Device,
    expect: Option<Tier>,
) -> gen_core::Result<VerifiedLmFiles> {
    check_compute(dtype, device)?;
    let dir = dirs.snapshot_dir(&ComponentId::Lm.component().repo)?;
    if crate::tier::is_tier_snapshot(&dir) {
        let verified = crate::tier::verify_tier(&dir)?;
        check_tier(expect, verified.tier(), &dir)?;
        return open_tier(&verified, dtype, device);
    }
    check_tier(expect, Tier::Bf16, &dir)?;
    let verified = snapshot::resolve_component(ComponentId::Lm, dirs)?;
    let config_path = verified.path("config.json").ok_or_else(|| {
        gen_core::Error::Msg("verified YuE2-3B snapshot has no config.json".into())
    })?;
    let weights = verified.weights_path().ok_or_else(|| {
        gen_core::Error::Msg("verified YuE2-3B snapshot has no weights file".into())
    })?;
    let config_json = std::fs::read_to_string(config_path)?;
    // SAFETY: the file is the snapshot's verified weights file, opened read-only; the mapping
    // lives only for the duration of the caller's load (every tensor is copied out in `dtype`).
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device) }
        .map_err(backend("open weights"))?;
    Ok(VerifiedLmFiles {
        config_json,
        loader: Loader::new(vb, None),
        weights_sha256: verified.manifest().native.sha256.clone(),
        tier: Tier::Bf16,
    })
}

/// An asserted tier must be the staged one: a request for `q8` over the BF16 original (or any
/// other mismatch) is refused, never served at another precision.
pub(crate) fn check_tier(expect: Option<Tier>, staged: Tier, dir: &std::path::Path) -> gen_core::Result<()> {
    match expect {
        Some(want) if want != staged => Err(gen_core::Error::Unsupported(format!(
            "yue2: the {want} tier was requested but {} holds the {staged} tier{}",
            dir.display(),
            if staged == Tier::Bf16 {
                " (the released checkpoint); derive the tier snapshot locally first \
                 (candle_audio_yue2::tier::convert, or the audio-lane snapshot preparer)"
            } else {
                ""
            }
        ))),
        _ => Ok(()),
    }
}

/// Open a verified tier snapshot for loading.
pub(crate) fn open_tier(
    verified: &crate::tier::VerifiedTier,
    dtype: DType,
    device: &Device,
) -> gen_core::Result<VerifiedLmFiles> {
    check_compute(dtype, device)?;
    let config_json = std::fs::read_to_string(verified.config_path())?;
    // SAFETY: the verified tier weights file, opened read-only; every tensor is copied out.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[verified.weights_path()], dtype, device)
    }
    .map_err(backend("open tier weights"))?;
    let storage: BTreeMap<String, Storage> = verified.storage_map();
    Ok(VerifiedLmFiles {
        config_json,
        loader: Loader::new(vb, Some(Arc::new(storage))),
        weights_sha256: verified.weights_sha256().to_string(),
        tier: verified.tier(),
    })
}

/// Which Mixture-of-Transformers paths to load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotPaths {
    /// The AR path only — everything the ABC and semantic stages run.
    Ar,
    /// Both paths — the acoustic (NAR) stage also needs the `nar_` twins.
    ArAndNar,
}

/// Upstream `RMSNorm.forward`: `x * rsqrt(mean(x.float()²) + eps).to(x.dtype) * weight`.
///
/// In F32 this is the textbook RMSNorm; in BF16 the reciprocal is rounded to BF16 before it scales
/// `x`, and the product is rounded again before the weight — the released arithmetic.
pub fn rms_norm(
    x: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> candle_audio::candle_core::Result<Tensor> {
    let dtype = x.dtype();
    let mean = x.to_dtype(DType::F32)?.sqr()?.mean_keepdim(D::Minus1)?;
    let inv = (mean + eps)?.sqrt()?.recip()?.to_dtype(dtype)?;
    x.broadcast_mul(&inv)?.broadcast_mul(weight)
}

/// One attention block (`self_attn` or `nar_self_attn`): bias-free Q/K/V/O projections and the
/// per-head `q_norm` / `k_norm` applied before RoPE.
#[derive(Debug)]
pub struct Attention {
    q_proj: Proj,
    k_proj: Proj,
    v_proj: Proj,
    o_proj: Proj,
    q_norm: Tensor,
    k_norm: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f64,
}

impl Attention {
    fn load(w: &Loader, prefix: &str, cfg: &Yue2Config) -> candle_audio::candle_core::Result<Self> {
        let (h, hd, nq, nkv) = (
            cfg.hidden_size,
            cfg.head_dim,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
        );
        let name = |leaf: &str| format!("{prefix}.{leaf}");
        Ok(Self {
            q_proj: w.matrix(&name("q_proj.weight"), (nq * hd, h))?,
            k_proj: w.matrix(&name("k_proj.weight"), (nkv * hd, h))?,
            v_proj: w.matrix(&name("v_proj.weight"), (nkv * hd, h))?,
            o_proj: w.matrix(&name("o_proj.weight"), (h, nq * hd))?,
            q_norm: w.tensor(&name("q_norm.weight"), hd)?,
            k_norm: w.tensor(&name("k_norm.weight"), hd)?,
            num_heads: nq,
            num_kv_heads: nkv,
            head_dim: hd,
            eps: cfg.rms_norm_eps,
        })
    }

    /// Upstream `Attention.project_qkv`: project, per-head normalize and rotate. `x` is
    /// `[batch, seq, hidden]` (already layer-normed); `cos`/`sin` are `[1, seq, head_dim]`.
    /// Returns Q `[batch, heads, seq, head_dim]` and K/V `[batch, kv_heads, seq, head_dim]`,
    /// contiguous and head-major (the KV cache's layout; K is RoPE'd, V raw).
    pub fn project_qkv(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> gen_core::Result<(Tensor, Tensor, Tensor)> {
        let err = backend("attention");
        let (b, t, _) = x.dims3().map_err(&err)?;
        let q = self.q_proj.forward(x, None).map_err(backend("q_proj"))?;
        let k = self.k_proj.forward(x, None).map_err(backend("k_proj"))?;
        let v = self.v_proj.forward(x, None).map_err(backend("v_proj"))?;
        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))
            .map_err(&err)?;
        let k = k
            .reshape((b, t, self.num_kv_heads, self.head_dim))
            .map_err(&err)?;
        let v = v
            .reshape((b, t, self.num_kv_heads, self.head_dim))
            .map_err(&err)?;
        let q = rms_norm(&q, &self.q_norm, self.eps).map_err(&err)?;
        let k = rms_norm(&k, &self.k_norm, self.eps).map_err(&err)?;
        let q = apply_rope(&q, cos, sin, false).map_err(backend("rope"))?;
        let k = apply_rope(&k, cos, sin, false).map_err(backend("rope"))?;
        let head_major = |x: Tensor| x.transpose(1, 2).and_then(|x| x.contiguous());
        Ok((
            head_major(q).map_err(&err)?,
            head_major(k).map_err(&err)?,
            head_major(v).map_err(&err)?,
        ))
    }

    /// The output projection of an attention result `[batch, heads, seq, head_dim]`.
    pub fn project_out(&self, attn: &Tensor) -> gen_core::Result<Tensor> {
        let err = backend("attention");
        let (b, _, t, _) = attn.dims4().map_err(&err)?;
        let merged = attn
            .transpose(1, 2)
            .and_then(|x| x.reshape((b, t, self.num_heads * self.head_dim)))
            .map_err(&err)?;
        self.o_proj.forward(&merged, None).map_err(backend("o_proj"))
    }

    pub(crate) fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    fn to_device(&self, device: &Device) -> candle_audio::candle_core::Result<Self> {
        Ok(Self {
            q_proj: self.q_proj.to_device(device)?,
            k_proj: self.k_proj.to_device(device)?,
            v_proj: self.v_proj.to_device(device)?,
            o_proj: self.o_proj.to_device(device)?,
            q_norm: self.q_norm.to_device(device)?,
            k_norm: self.k_norm.to_device(device)?,
            ..*self
        })
    }

    /// The four projections, in `q, k, v, o` order.
    pub(crate) fn projections(&self) -> [&Proj; 4] {
        [&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj]
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn projections_mut(&mut self) -> [&mut Proj; 4] {
        [
            &mut self.q_proj,
            &mut self.k_proj,
            &mut self.v_proj,
            &mut self.o_proj,
        ]
    }

    fn norms(&self) -> [&Tensor; 2] {
        [&self.q_norm, &self.k_norm]
    }
}

/// One SwiGLU MLP (`mlp` or `nar_mlp`): `down(silu(gate(x)) * up(x))`.
#[derive(Debug)]
pub struct Mlp {
    gate_proj: Proj,
    up_proj: Proj,
    down_proj: Proj,
}

impl Mlp {
    fn load(w: &Loader, prefix: &str, cfg: &Yue2Config) -> candle_audio::candle_core::Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate_proj: w.matrix(&format!("{prefix}.gate_proj.weight"), (i, h))?,
            up_proj: w.matrix(&format!("{prefix}.up_proj.weight"), (i, h))?,
            down_proj: w.matrix(&format!("{prefix}.down_proj.weight"), (h, i))?,
        })
    }

    /// `down(silu(gate(x)) * up(x))`.
    pub fn forward(&self, x: &Tensor) -> gen_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x, None).map_err(backend("gate_proj"))?;
        let up = self.up_proj.forward(x, None).map_err(backend("up_proj"))?;
        let act = swiglu(&gate, &up).map_err(backend("swiglu"))?;
        self.down_proj.forward(&act, None).map_err(backend("down_proj"))
    }

    /// The three projections, in `gate, up, down` order.
    pub(crate) fn projections(&self) -> [&Proj; 3] {
        [&self.gate_proj, &self.up_proj, &self.down_proj]
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn projections_mut(&mut self) -> [&mut Proj; 3] {
        [&mut self.gate_proj, &mut self.up_proj, &mut self.down_proj]
    }

    fn to_device(&self, device: &Device) -> candle_audio::candle_core::Result<Self> {
        Ok(Self {
            gate_proj: self.gate_proj.to_device(device)?,
            up_proj: self.up_proj.to_device(device)?,
            down_proj: self.down_proj.to_device(device)?,
        })
    }
}

/// One transformer path of a layer: pre-attention norm, attention, pre-MLP norm, MLP.
#[derive(Debug)]
pub struct MotPath {
    /// `input_layernorm` / `nar_input_layernorm`.
    pub attn_norm: Tensor,
    /// `self_attn` / `nar_self_attn`.
    pub attn: Attention,
    /// `post_attention_layernorm` / `nar_pre_mlp_layernorm`.
    pub mlp_norm: Tensor,
    /// `mlp` / `nar_mlp`.
    pub mlp: Mlp,
}

impl MotPath {
    fn to_device(&self, device: &Device) -> candle_audio::candle_core::Result<Self> {
        Ok(Self {
            attn_norm: self.attn_norm.to_device(device)?,
            attn: self.attn.to_device(device)?,
            mlp_norm: self.mlp_norm.to_device(device)?,
            mlp: self.mlp.to_device(device)?,
        })
    }

    /// The device of every tensor of this path (the first one's; a path is moved as a whole).
    fn device(&self) -> Device {
        self.attn_norm.device().clone()
    }

    /// Every matmul weight of the path: the attention's `q, k, v, o`, then the MLP's
    /// `gate, up, down` — upstream's AR-linear order.
    pub(crate) fn projections(&self) -> Vec<&Proj> {
        let mut out: Vec<&Proj> = self.attn.projections().to_vec();
        out.extend(self.mlp.projections());
        out
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn projections_mut(&mut self) -> Vec<&mut Proj> {
        let mut out: Vec<&mut Proj> = self.attn.projections_mut().into_iter().collect();
        out.extend(self.mlp.projections_mut());
        out
    }

    /// Every vector (norm) weight of the path.
    fn norms(&self) -> Vec<&Tensor> {
        let mut out = vec![&self.attn_norm, &self.mlp_norm];
        out.extend(self.attn.norms());
        out
    }

    fn bytes(&self) -> usize {
        let dense: usize = self
            .norms()
            .iter()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum();
        dense
            + self
                .projections()
                .iter()
                .map(|p| p.resident_bytes() as usize)
                .sum::<usize>()
    }
}

/// One decoder layer: the AR path and, when loaded, the NAR path.
#[derive(Debug)]
pub struct MotLayer {
    /// The AR path (generation).
    pub ar: MotPath,
    /// The NAR path (acoustic flow matching); `None` under [`MotPaths::Ar`].
    pub nar: Option<MotPath>,
}

/// The YuE2 MoT language model: token embedding, the decoder layers, the final norm and the
/// untied `lm_head` (the NAR auxiliary heads — `llm2vae`, `vae2llm`, the time and latent-position
/// embeddings — belong to the acoustic stage, [`crate::nar`], and are not loaded here).
///
/// The model holds one precision tier ([`Tier`], [`crate::precision`]) fixed at load: every matmul
/// weight is dense (`bf16`) or GGML block-quantized (`q8` / `q4`); the embedding and the norms stay
/// dense. The experimental FP8 AR mode ([`crate::fp8`]) temporarily swaps the AR projections.
///
/// # AR offload
///
/// Upstream's `nar._offload_ar` moves the modules the acoustic ODE never reads — the token
/// embedding, `lm_head` and every layer's AR path — to host memory while one chunk is solved, and
/// moves them back afterwards. [`Yue2Lm::offload_ar`] / [`Yue2Lm::restore_ar`] are that
/// transition. While offloaded, every AR entry point ([`Yue2Lm::prefill`], [`Yue2Lm::decode`])
/// refuses with an error instead of computing on host copies; the NAR twins and the final norm
/// never move. A GGML weight moves as its exact blocks.
#[derive(Debug)]
pub struct Yue2Lm {
    config: Yue2Config,
    embed_tokens: Tensor,
    layers: Vec<MotLayer>,
    norm: Tensor,
    lm_head: Proj,
    rope: Rope,
    dtype: DType,
    device: Device,
    tier: Tier,
    ar_offloaded: bool,
    pub(crate) fp8: Option<crate::fp8::Fp8State>,
}

impl Yue2Lm {
    /// Resolve and verify the YuE2-3B snapshot from `dirs` **immediately before loading** (see the
    /// crate's load-boundary rule), then load exactly the verified `config.json` and weights file:
    /// the pinned original (`bf16`) or a derived tier snapshot ([`crate::tier`]). `dtype` is the
    /// compute dtype (the BF16 checkpoint is converted on load; use `DType::F32` on a CPU device,
    /// which has no BF16 matmul).
    pub fn load(
        dirs: &SnapshotDirs,
        paths: MotPaths,
        dtype: DType,
        device: &Device,
    ) -> gen_core::Result<Self> {
        let opened = open_verified(dirs, dtype, device, None)?;
        Self::from_loader(
            Yue2Config::from_json(&opened.config_json)?,
            &opened.loader,
            paths,
            opened.tier,
        )
    }

    /// Build the released-precision model from an arbitrary [`VarBuilder`]. Crate-private:
    /// production loads go through [`Yue2Lm::load`], which verifies the bytes first.
    #[cfg(test)]
    pub(crate) fn from_var_builder(
        config: Yue2Config,
        vb: VarBuilder,
        paths: MotPaths,
    ) -> gen_core::Result<Self> {
        Self::from_loader(config, &Loader::new(vb, None), paths, Tier::Bf16)
    }

    /// Build from a tier-aware [`Loader`].
    pub(crate) fn from_loader(
        config: Yue2Config,
        w: &Loader,
        paths: MotPaths,
        tier: Tier,
    ) -> gen_core::Result<Self> {
        check_compute(w.dtype(), w.device())?;
        config.validate()?;
        let err = backend("load");
        let (h, v) = (config.hidden_size, config.vocab_size);
        let embed_tokens = w
            .tensor("model.embed_tokens.weight", (v, h))
            .map_err(&err)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let ar = MotPath {
                attn_norm: w
                    .tensor(&format!("{p}.input_layernorm.weight"), h)
                    .map_err(&err)?,
                attn: Attention::load(w, &format!("{p}.self_attn"), &config).map_err(&err)?,
                mlp_norm: w
                    .tensor(&format!("{p}.post_attention_layernorm.weight"), h)
                    .map_err(&err)?,
                mlp: Mlp::load(w, &format!("{p}.mlp"), &config).map_err(&err)?,
            };
            let nar = match paths {
                MotPaths::Ar => None,
                MotPaths::ArAndNar => Some(MotPath {
                    attn_norm: w
                        .tensor(&format!("{p}.nar_input_layernorm.weight"), h)
                        .map_err(&err)?,
                    attn: Attention::load(w, &format!("{p}.nar_self_attn"), &config)
                        .map_err(&err)?,
                    mlp_norm: w
                        .tensor(&format!("{p}.nar_pre_mlp_layernorm.weight"), h)
                        .map_err(&err)?,
                    mlp: Mlp::load(w, &format!("{p}.nar_mlp"), &config).map_err(&err)?,
                }),
            };
            layers.push(MotLayer { ar, nar });
        }
        let norm = w.tensor("model.norm.weight", h).map_err(&err)?;
        let lm_head = w.matrix("lm_head.weight", (v, h)).map_err(&err)?;
        let rope = Rope::standard(config.head_dim as i32, config.rope_theta as f32);
        Ok(Self {
            dtype: embed_tokens.dtype(),
            device: embed_tokens.device().clone(),
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            tier,
            ar_offloaded: false,
            fp8: None,
        })
    }

    /// The parsed configuration.
    pub fn config(&self) -> &Yue2Config {
        &self.config
    }

    /// The compute dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The device the weights live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The precision tier the weights were loaded at.
    pub fn tier(&self) -> Tier {
        self.tier
    }

    /// Whether the AR-only weights are offloaded (see the [type docs](Yue2Lm#ar-offload)).
    pub fn ar_offloaded(&self) -> bool {
        self.ar_offloaded
    }

    /// `lm_head`.
    #[cfg(test)]
    pub(crate) fn lm_head(&self) -> &Proj {
        &self.lm_head
    }

    /// The layers, mutably (the FP8 AR mode swaps AR projections in place).
    #[cfg(feature = "cuda")]
    pub(crate) fn layers_mut(&mut self) -> &mut [MotLayer] {
        &mut self.layers
    }

    /// Measured resident weight bytes of this model (both paths, embedding, norms, `lm_head`; the
    /// NAR heads are counted by [`crate::nar::Yue2Nar::weight_residency`]). While the FP8 AR mode
    /// is active its CPU-held BF16 originals are [`Residency::host_bytes`].
    pub fn weight_residency(&self) -> Residency {
        let dense = |t: &Tensor| (t.elem_count() * t.dtype().size_in_bytes()) as u64;
        let mut device = dense(&self.embed_tokens) + dense(&self.norm);
        device += self.lm_head.resident_bytes();
        for layer in &self.layers {
            for path in std::iter::once(&layer.ar).chain(layer.nar.as_ref()) {
                device += path.norms().into_iter().map(dense).sum::<u64>();
                device += path
                    .projections()
                    .iter()
                    .map(|p| p.resident_bytes())
                    .sum::<u64>();
            }
        }
        Residency {
            device_bytes: device,
            host_bytes: self.fp8.as_ref().map_or(0, |f| f.original_bytes()),
        }
    }

    /// Move the AR-only weights — `embed_tokens`, `lm_head` and every layer's AR path — to host
    /// memory, upstream's `_offload_ar(model, True)` entry. Returns the bytes that left the device:
    /// `0` for a model that already lives in host memory (upstream moves only modules whose device
    /// is not the CPU), in which case this only marks the AR path unavailable. An error part-way
    /// leaves the model marked offloaded; [`Yue2Lm::restore_ar`] brings every tensor back. Refused
    /// while the FP8 AR mode is active (restore the BF16 AR path first — the acoustic stage always
    /// does).
    pub fn offload_ar(&mut self) -> gen_core::Result<usize> {
        if self.fp8.is_some() {
            return Err(gen_core::Error::Msg(
                "YuE2: the AR path is in the FP8 mode; restore its BF16 originals before offloading"
                    .into(),
            ));
        }
        self.ar_offloaded = true;
        if self.device.is_cpu() {
            return Ok(0);
        }
        let host = Device::Cpu;
        let err = backend("offload AR path");
        let mut moved = 0;
        if !self.embed_tokens.device().is_cpu() {
            moved += self.embed_tokens.elem_count() * self.embed_tokens.dtype().size_in_bytes();
            self.embed_tokens = self.embed_tokens.to_device(&host).map_err(&err)?;
        }
        if !self.lm_head.device().is_cpu() {
            moved += self.lm_head.resident_bytes() as usize;
            self.lm_head = self.lm_head.to_device(&host).map_err(&err)?;
        }
        for layer in &mut self.layers {
            if !layer.ar.device().is_cpu() {
                moved += layer.ar.bytes();
                layer.ar = layer.ar.to_device(&host).map_err(&err)?;
            }
        }
        Ok(moved)
    }

    /// Move every AR-only weight back to the model device and make the AR path available again —
    /// upstream's `_offload_ar` exit. Idempotent: tensors already on the device are not copied, so
    /// it also recovers from an [`Yue2Lm::offload_ar`] or an earlier restore that failed part-way.
    pub fn restore_ar(&mut self) -> gen_core::Result<()> {
        if !self.device.is_cpu() {
            let device = self.device.clone();
            let err = backend("restore AR path");
            if !self.embed_tokens.device().same_device(&device) {
                self.embed_tokens = self.embed_tokens.to_device(&device).map_err(&err)?;
            }
            if !self.lm_head.device().same_device(&device) {
                self.lm_head = self.lm_head.to_device(&device).map_err(&err)?;
            }
            for layer in &mut self.layers {
                if !layer.ar.device().same_device(&device) {
                    layer.ar = layer.ar.to_device(&device).map_err(&err)?;
                }
            }
        }
        self.ar_offloaded = false;
        Ok(())
    }

    /// The decoder layers (both MoT paths when loaded with [`MotPaths::ArAndNar`]).
    pub fn layers(&self) -> &[MotLayer] {
        &self.layers
    }

    /// The final RMSNorm weight (`model.norm`), shared by both paths' outputs.
    pub fn final_norm(&self) -> &Tensor {
        &self.norm
    }

    /// RoPE `(cos, sin)` tables `[1, len, head_dim]` for positions `start..start + len`, in the
    /// model dtype (upstream builds them in F32 and casts at use).
    pub fn rope_tables(&self, start: usize, len: usize) -> gen_core::Result<(Tensor, Tensor)> {
        self.rope
            .cos_sin(len as i32, start as i32, self.dtype, &self.device)
            .map_err(backend("rope"))
    }

    /// A fresh single-sequence KV cache for `capacity` positions in the model dtype — upstream's
    /// `StaticKVCache(batch_size=1, max_seq_len=capacity)`.
    pub fn new_cache(&self, capacity: usize) -> gen_core::Result<StaticKvCache> {
        StaticKvCache::new(
            self.config.num_hidden_layers,
            1,
            self.config.num_key_value_heads,
            self.config.head_dim,
            capacity,
            self.dtype,
            &self.device,
        )
        .map_err(backend("kv cache"))
    }

    /// Run the AR path over `ids` (appended to `cache` at its current offset) and return the
    /// **last** position's logits `[vocab]` in the model dtype — upstream's
    /// `model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)`. Positions continue
    /// from the cache offset; queries attend every cached key plus the causal part of `ids`.
    fn forward_last(&self, ids: &[u32], cache: &mut StaticKvCache) -> gen_core::Result<Tensor> {
        let err = backend("forward");
        if self.ar_offloaded {
            return Err(gen_core::Error::Msg(
                "YuE2 forward: the AR path is offloaded (restore it with `Yue2Lm::restore_ar`)"
                    .into(),
            ));
        }
        if ids.is_empty() {
            return Err(gen_core::Error::Msg(
                "YuE2 forward: input must contain at least one token".into(),
            ));
        }
        if let Some(&bad) = ids.iter().find(|&&t| t as usize >= self.config.vocab_size) {
            return Err(gen_core::Error::Msg(format!(
                "YuE2 forward: token id {bad} is outside the {}-token vocabulary",
                self.config.vocab_size
            )));
        }
        let start = cache.offset() as usize;
        let t = ids.len();
        let input = Tensor::from_slice(ids, (1, t), &self.device).map_err(&err)?;
        let mut x = embed(&self.embed_tokens, &input).map_err(backend("embed"))?;
        let (cos, sin) = self.rope_tables(start, t)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let p = &layer.ar;
            let normed = rms_norm(&x, &p.attn_norm, self.config.rms_norm_eps).map_err(&err)?;
            let (q, k, v) = p.attn.project_qkv(&normed, &cos, &sin)?;
            let (keys, values) = cache.update(i, &k, &v).map_err(backend("kv cache"))?;
            let attn = sdpa_gqa(&q, &keys, &values, p.attn.scale(), AttnMask::Causal)
                .map_err(backend("attention"))?;
            x = (x + p.attn.project_out(&attn)?).map_err(&err)?;
            let normed = rms_norm(&x, &p.mlp_norm, self.config.rms_norm_eps).map_err(&err)?;
            x = (&x + p.mlp.forward(&normed)?).map_err(&err)?;
        }
        let last = x.narrow(1, t - 1, 1).map_err(&err)?;
        let last = rms_norm(&last, &self.norm, self.config.rms_norm_eps).map_err(&err)?;
        let logits = self
            .lm_head
            .forward(&last, None)
            .map_err(backend("lm_head"))?;
        logits.flatten_all().map_err(err)
    }

    /// Prefill `ids` into `cache` in [`PREFILL_CHUNK`]-position forwards, calling `between_chunks`
    /// before each forward (the cancellation seam: an `Err` aborts before any further work), and
    /// return the last position's logits `[vocab]`.
    pub fn prefill(
        &self,
        ids: &[u32],
        cache: &mut StaticKvCache,
        mut between_chunks: impl FnMut() -> gen_core::Result<()>,
    ) -> gen_core::Result<Tensor> {
        if ids.is_empty() {
            return Err(gen_core::Error::Msg(
                "YuE2 prefill: the prefix must contain at least one token".into(),
            ));
        }
        let mut last = None;
        for chunk in ids.chunks(PREFILL_CHUNK) {
            between_chunks()?;
            last = Some(self.forward_last(chunk, cache)?);
        }
        Ok(last.expect("a non-empty prefix has at least one chunk"))
    }

    /// Append one token to `cache` and return the next position's logits `[vocab]`.
    pub fn decode(&self, token: u32, cache: &mut StaticKvCache) -> gen_core::Result<Tensor> {
        self.forward_last(&[token], cache)
    }
}

#[cfg(all(test, any(feature = "cuda", feature = "metal")))]
impl Yue2Lm {
    /// Where every weight lives, for the offload tests: `(AR-only tensors, tensors that never
    /// move)` — the latter being every NAR twin and the final norm.
    pub(crate) fn placement(&self) -> (Vec<Device>, Vec<Device>) {
        let path = |p: &MotPath| {
            let mut t: Vec<Device> = p.norms().iter().map(|t| t.device().clone()).collect();
            t.extend(p.projections().iter().map(|p| p.device()));
            t
        };
        let mut ar = vec![self.embed_tokens.device().clone(), self.lm_head.device()];
        let mut fixed = vec![self.norm.device().clone()];
        for layer in &self.layers {
            ar.extend(path(&layer.ar));
            if let Some(nar) = &layer.nar {
                fixed.extend(path(nar));
            }
        }
        (ar, fixed)
    }
}

#[cfg(test)]
pub(crate) mod synthetic {
    //! A deterministic, tiny-width YuE2 with the real architecture: GQA (2 query heads per KV
    //! head), per-head Q/K norm, a non-square Q projection, RoPE θ = 10⁶, and the **full** protocol
    //! vocabulary (so every token-range mask and stop id is real). Weights are an integer hash of
    //! (tensor name, element index), bit-identical to `scripts/reference/yue2/ar_fixtures.py`'s
    //! `synthetic_state_dict`, so the committed reference logits of that script's upstream
    //! `YuE2ForCausalLM` apply to this model with nothing but the fixture committed.
    use std::collections::HashMap;

    use super::*;

    pub(crate) const HIDDEN: usize = 32;
    pub(crate) const LAYERS: usize = 2;

    pub(crate) fn config() -> Yue2Config {
        Yue2Config {
            hidden_size: HIDDEN,
            num_hidden_layers: LAYERS,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            intermediate_size: 64,
            vocab_size: crate::protocol::VOCAB_SIZE as usize,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: crate::protocol::CONTEXT,
        }
    }

    fn fnv1a(name: &str) -> u32 {
        name.bytes().fold(0x811C_9DC5u32, |h, b| {
            (h ^ b as u32).wrapping_mul(0x0100_0193)
        })
    }

    fn fmix(mut x: u32) -> u32 {
        x ^= x >> 16;
        x = x.wrapping_mul(0x85EB_CA6B);
        x ^= x >> 13;
        x = x.wrapping_mul(0xC2B2_AE35);
        x ^ (x >> 16)
    }

    /// Element `i` of tensor `name`: an integer in `[-1000, 1000]`, scaled.
    pub(crate) fn values(name: &str, n: usize, scale: f32, offset: f32) -> Vec<f32> {
        let seed = fnv1a(name);
        let step = scale / 1000.0;
        (0..n)
            .map(|i| {
                let x = fmix(seed ^ (i as u32).wrapping_mul(0x9E37_79B1));
                offset + ((x % 2001) as f32 - 1000.0) * step
            })
            .collect()
    }

    /// A fixed logits row for sampler parity: `((h % 2000001) - 1000000) · scale / 10⁶` — two
    /// million levels, so ties are rare (`row_values` in the fixture generator).
    pub(crate) fn row_values(name: &str, n: usize, scale: f32) -> Vec<f32> {
        let seed = fnv1a(name);
        let step = scale / 1_000_000.0;
        (0..n)
            .map(|i| {
                let x = fmix(seed ^ (i as u32).wrapping_mul(0x9E37_79B1));
                ((x % 2_000_001) as f32 - 1_000_000.0) * step
            })
            .collect()
    }

    /// Every tensor the synthetic checkpoint holds (both MoT paths), by upstream name.
    pub(crate) fn state_dict(cfg: &Yue2Config) -> Vec<(String, Vec<usize>, f32, f32)> {
        let (h, hd, nq, nkv, i, v) = (
            cfg.hidden_size,
            cfg.head_dim,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.intermediate_size,
            cfg.vocab_size,
        );
        let proj = |fan_in: usize| 1.7 / (fan_in as f32).sqrt();
        let mut out = vec![
            (
                "model.embed_tokens.weight".to_string(),
                vec![v, h],
                1.0,
                0.0,
            ),
            ("model.norm.weight".to_string(), vec![h], 0.25, 1.0),
            ("lm_head.weight".to_string(), vec![v, h], proj(h), 0.0),
        ];
        for l in 0..cfg.num_hidden_layers {
            for (attn, norm_a, norm_m, mlp) in [
                (
                    "self_attn",
                    "input_layernorm",
                    "post_attention_layernorm",
                    "mlp",
                ),
                (
                    "nar_self_attn",
                    "nar_input_layernorm",
                    "nar_pre_mlp_layernorm",
                    "nar_mlp",
                ),
            ] {
                let p = format!("model.layers.{l}");
                out.push((format!("{p}.{norm_a}.weight"), vec![h], 0.25, 1.0));
                out.push((format!("{p}.{norm_m}.weight"), vec![h], 0.25, 1.0));
                out.push((
                    format!("{p}.{attn}.q_proj.weight"),
                    vec![nq * hd, h],
                    proj(h),
                    0.0,
                ));
                out.push((
                    format!("{p}.{attn}.k_proj.weight"),
                    vec![nkv * hd, h],
                    proj(h),
                    0.0,
                ));
                out.push((
                    format!("{p}.{attn}.v_proj.weight"),
                    vec![nkv * hd, h],
                    proj(h),
                    0.0,
                ));
                out.push((
                    format!("{p}.{attn}.o_proj.weight"),
                    vec![h, nq * hd],
                    proj(nq * hd),
                    0.0,
                ));
                out.push((format!("{p}.{attn}.q_norm.weight"), vec![hd], 0.25, 1.0));
                out.push((format!("{p}.{attn}.k_norm.weight"), vec![hd], 0.25, 1.0));
                out.push((
                    format!("{p}.{mlp}.gate_proj.weight"),
                    vec![i, h],
                    proj(h),
                    0.0,
                ));
                out.push((
                    format!("{p}.{mlp}.up_proj.weight"),
                    vec![i, h],
                    proj(h),
                    0.0,
                ));
                out.push((
                    format!("{p}.{mlp}.down_proj.weight"),
                    vec![h, i],
                    proj(i),
                    0.0,
                ));
            }
        }
        out
    }

    pub(crate) fn tensors(cfg: &Yue2Config) -> HashMap<String, Tensor> {
        state_dict(cfg)
            .into_iter()
            .map(|(name, shape, scale, offset)| {
                let n = shape.iter().product();
                let data = values(&name, n, scale, offset);
                let t = Tensor::from_vec(data, shape, &Device::Cpu).expect("synthetic tensor");
                (name, t)
            })
            .collect()
    }

    /// The synthetic model on the CPU in F32.
    pub(crate) fn model(paths: MotPaths) -> Yue2Lm {
        model_on(paths, &Device::Cpu, DType::F32)
    }

    /// The synthetic model on `device` computing in `dtype` (the F32 values converted on load, as
    /// a released BF16 checkpoint is).
    pub(crate) fn model_on(paths: MotPaths, device: &Device, dtype: DType) -> Yue2Lm {
        let cfg = config();
        let vb = VarBuilder::from_tensors(tensors(&cfg), dtype, device);
        Yue2Lm::from_var_builder(cfg, vb, paths).expect("synthetic YuE2 loads")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_vec(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32).unwrap().to_vec1().unwrap()
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn released_config_parses_and_refuses_other_architectures() {
        let released = r#"{"model_type":"yue2","tie_word_embeddings":false,"latent_type":"vae",
            "hidden_size":2048,"num_hidden_layers":28,"num_attention_heads":16,
            "num_key_value_heads":8,"head_dim":128,"intermediate_size":6144,"vocab_size":184704,
            "rms_norm_eps":1e-06,"rope_theta":1000000,"max_position_embeddings":24576}"#;
        let cfg = Yue2Config::from_json(released).unwrap();
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.rope_theta, 1e6);
        for (from, to) in [
            (
                r#""tie_word_embeddings":false"#,
                r#""tie_word_embeddings":true"#,
            ),
            (r#""latent_type":"vae""#, r#""latent_type":"codec""#),
            (r#""model_type":"yue2""#, r#""model_type":"qwen3""#),
            (r#""num_key_value_heads":8"#, r#""num_key_value_heads":6"#),
            (r#""vocab_size":184704"#, r#""vocab_size":151936"#),
        ] {
            let bad = released.replace(from, to);
            assert!(Yue2Config::from_json(&bad).is_err(), "accepted {to}");
        }
        let missing = released.replace(r#""head_dim":128,"#, "");
        assert!(Yue2Config::from_json(&missing).is_err());
    }

    /// An unsupported compute dtype (F16), or BF16 on the CPU (no BF16 matmul), is refused before
    /// any weight is read; F32 on the CPU loads.
    #[test]
    fn unsupported_compute_dtypes_are_refused_before_loading() {
        let cfg = synthetic::config();
        for dtype in [DType::F16, DType::BF16, DType::F64] {
            // An empty builder: reaching any tensor lookup would be a "missing tensor" error.
            let vb = VarBuilder::from_tensors(Default::default(), dtype, &Device::Cpu);
            let err = Yue2Lm::from_var_builder(cfg.clone(), vb, MotPaths::Ar).unwrap_err();
            assert!(
                matches!(err, gen_core::Error::Unsupported(_)),
                "{dtype:?}: {err}"
            );
        }
        let dirs = SnapshotDirs::new();
        let err = Yue2Lm::load(&dirs, MotPaths::Ar, DType::F16, &Device::Cpu).unwrap_err();
        assert!(matches!(err, gen_core::Error::Unsupported(_)), "{err}");
        // F32 passes the check and only then reaches verification (a cold cache here).
        let err = Yue2Lm::load(&dirs, MotPaths::Ar, DType::F32, &Device::Cpu).unwrap_err();
        assert!(!matches!(err, gen_core::Error::Unsupported(_)), "{err}");
    }

    #[test]
    fn ar_only_load_never_reads_the_nar_twins() {
        let cfg = synthetic::config();
        let mut t = synthetic::tensors(&cfg);
        t.retain(|name, _| !name.contains(".nar_"));
        let vb = VarBuilder::from_tensors(t.clone(), DType::F32, &Device::Cpu);
        let lm = Yue2Lm::from_var_builder(cfg.clone(), vb, MotPaths::Ar).unwrap();
        assert!(lm.layers().iter().all(|l| l.nar.is_none()));
        let vb = VarBuilder::from_tensors(t, DType::F32, &Device::Cpu);
        let err = Yue2Lm::from_var_builder(cfg, vb, MotPaths::ArAndNar).unwrap_err();
        assert!(err.to_string().contains("nar_"), "{err}");
        let both = synthetic::model(MotPaths::ArAndNar);
        assert!(both.layers().iter().all(|l| l.nar.is_some()));
    }

    #[test]
    fn rms_norm_is_upstreams_two_rounding_formula() {
        let x = Tensor::new(&[[1.5f32, -2.25, 0.125, 3.0]], &Device::Cpu).unwrap();
        let w = Tensor::new(&[1.0f32, 0.5, 2.0, -1.0], &Device::Cpu).unwrap();
        let got = to_vec(&rms_norm(&x, &w, 1e-6).unwrap().flatten_all().unwrap());
        let ms = (1.5f32 * 1.5 + 2.25 * 2.25 + 0.125 * 0.125 + 9.0) / 4.0;
        let r = 1.0 / (ms + 1e-6).sqrt();
        let want = [1.5 * r, -2.25 * r * 0.5, 0.125 * r * 2.0, -3.0 * r];
        assert!(max_abs(&got, &want) < 1e-6, "{got:?} vs {want:?}");
    }

    /// Prefill-then-decode equals a full recompute of the same sequence: the cache holds the right
    /// keys at the right positions, and a decode step's RoPE position is its cache slot.
    #[test]
    fn cached_decode_equals_full_recompute() {
        let lm = synthetic::model(MotPaths::Ar);
        let seq: Vec<u32> = vec![151643, 40, 1234, 99, 151847, 5, 777, 151848, 151851, 160000];
        let split = 6;
        let mut cache = lm.new_cache(seq.len()).unwrap();
        let mut logits = lm.prefill(&seq[..split], &mut cache, || Ok(())).unwrap();
        for &tok in &seq[split..] {
            logits = lm.decode(tok, &mut cache).unwrap();
        }
        let mut fresh = lm.new_cache(seq.len()).unwrap();
        let full = lm.prefill(&seq, &mut fresh, || Ok(())).unwrap();
        let (a, b) = (to_vec(&logits), to_vec(&full));
        // Same function, different matmul shapes: F32 reduction-order noise only.
        let d = max_abs(&a, &b);
        assert!(d < 1e-4, "cached vs recompute max |Δ| = {d}");
    }

    /// A prefix longer than one prefill chunk is prefilled in bounded chunks and equals the same
    /// prefix prefilled as one forward (bottom-right causal over the cache: same keys, same
    /// positions).
    #[test]
    fn chunked_prefill_equals_single_forward() {
        let lm = synthetic::model(MotPaths::Ar);
        let seq: Vec<u32> = (0..PREFILL_CHUNK as u32 + 37)
            .map(|i| (i * 7919) % 151_643)
            .collect();
        let mut calls = 0;
        let mut cache = lm.new_cache(seq.len()).unwrap();
        let chunked = lm
            .prefill(&seq, &mut cache, || {
                calls += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(calls, 2, "one cancellation check per chunk");
        assert_eq!(cache.offset() as usize, seq.len());
        let mut one = lm.new_cache(seq.len()).unwrap();
        let single = lm.forward_last(&seq, &mut one).unwrap();
        let d = max_abs(&to_vec(&chunked), &to_vec(&single));
        assert!(d < 1e-4, "chunked vs single prefill max |Δ| = {d}");
    }

    #[test]
    fn cache_capacity_is_a_hard_bound() {
        let lm = synthetic::model(MotPaths::Ar);
        let mut cache = lm.new_cache(3).unwrap();
        lm.prefill(&[1, 2, 3], &mut cache, || Ok(())).unwrap();
        assert!(lm.decode(4, &mut cache).is_err());
        let mut cache = lm.new_cache(4).unwrap();
        assert!(lm.prefill(&[1, 2, 3], &mut cache, || Ok(())).is_ok());
        assert!(lm.decode(u32::MAX, &mut cache).is_err(), "out-of-vocab id");
    }
}
