//! The full Ideogram 4 DiT: token composition (`[text ; image]`), scalar-`t` AdaLN conditioning, 34
//! blocks, and the affine-less final layer. Port of `Ideogram4Transformer.forward`.
//!
//! Token roles (`indicator`): `LLM_TOKEN_INDICATOR = 3` (text), `OUTPUT_IMAGE_INDICATOR = 2`
//! (image). Text positions carry the projected Qwen3-VL features (`llm_cond_proj`); image positions
//! carry the patchified noise latents (`input_proj`). Both streams live in one sequence, mixed every
//! block by full (segment-masked) attention + interleaved 3D MRoPE.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use candle_gen::candle_core::{DType, Device, DeviceLocation, Result, Tensor, TensorId, D};
use candle_gen::gen_core::{AdapterSpec, CancelFlag};

use super::block::Ideogram4Block;
use super::mrope::Ideogram4MRoPE;
use super::rmsnorm;
use crate::config::Ideogram4DitConfig;
use crate::loader::{embedding_detect, linear_detect, Weights};
use crate::quant::{QEmbedding, QLinear};

/// Token role constants (upstream `ideogram4.constants`).
const OUTPUT_IMAGE_INDICATOR: i64 = 2;
const LLM_TOKEN_INDICATOR: i64 = 3;

/// `llm_cond_norm` and the final LayerNorm both use eps 1e-6 (upstream).
const COND_NORM_EPS: f64 = 1e-6;
const FINAL_NORM_EPS: f64 = 1e-6;

static NEXT_IDEOGRAM_LOAD_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_IDEOGRAM_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

fn next_ideogram_load_id() -> u64 {
    NEXT_IDEOGRAM_LOAD_ID.fetch_add(1, Ordering::Relaxed)
}

fn host_slice_fingerprint(values: &[i64]) -> u64 {
    values.iter().fold(0xcbf29ce484222325, |hash, value| {
        value.to_le_bytes().iter().fold(hash, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
    })
}

/// O(1) denoise-loop identity for the immutable host-owned role and segment packing. Construct this
/// only after `Packing` is complete, then pass the same value to preparation and every prepared
/// forward. A new request always receives a new nonce, even when its host arrays have identical data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HostConditioningIdentity {
    request_id: u64,
    indicator_len: usize,
    segment_len: usize,
    indicator_fingerprint: u64,
    segment_fingerprint: u64,
}

impl HostConditioningIdentity {
    pub(crate) fn new(indicator: &[i64], segment_ids: &[i64]) -> Self {
        Self {
            request_id: NEXT_IDEOGRAM_REQUEST_ID.fetch_add(1, Ordering::Relaxed),
            indicator_len: indicator.len(),
            segment_len: segment_ids.len(),
            indicator_fingerprint: host_slice_fingerprint(indicator),
            segment_fingerprint: host_slice_fingerprint(segment_ids),
        }
    }

    fn matches_host(&self, indicator: &[i64], segment_ids: &[i64]) -> bool {
        self.indicator_len == indicator.len()
            && self.segment_len == segment_ids.len()
            && self.indicator_fingerprint == host_slice_fingerprint(indicator)
            && self.segment_fingerprint == host_slice_fingerprint(segment_ids)
    }
}

pub struct Ideogram4Transformer {
    input_proj: QLinear,
    llm_cond_norm: Tensor,
    llm_cond_proj: QLinear,
    t_mlp_in: QLinear,
    t_mlp_out: QLinear,
    adaln_proj: QLinear,
    embed_image_indicator: QEmbedding,
    rotary_emb: Ideogram4MRoPE,
    layers: TransformerLayers,
    final_adaln: QLinear,
    final_linear: QLinear,
    /// Sinusoidal frequencies for the `t` embedding (`[1, emb_dim/2]`, f32).
    t_freqs: Tensor,
    dtype: DType,
    load_id: u64,
}

enum TransformerLayers {
    Resident(Vec<Ideogram4Block>),
    Streamed(StreamedLayers),
}

struct StreamedLayers {
    weights: Arc<Weights>,
    config: Ideogram4DitConfig,
    window_size: usize,
    turbo_adapter: Option<PathBuf>,
    user_adapters: Vec<AdapterSpec>,
}

struct MaterializedWindow {
    layers: Vec<(usize, Ideogram4Block)>,
    device: Device,
}

impl Drop for MaterializedWindow {
    fn drop(&mut self) {
        // Transfers and kernels may still reference the current window. Always synchronize before the
        // device tensors are released, including error, cancellation, and panic unwinding.
        let _ = self.device.synchronize();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorSpec {
    shape: Vec<usize>,
    dtype: DType,
    device: DeviceLocation,
}

impl TensorSpec {
    fn new(tensor: &Tensor) -> Self {
        Self {
            shape: tensor.dims().to_vec(),
            dtype: tensor.dtype(),
            device: tensor.device().location(),
        }
    }

    fn matches(&self, tensor: &Tensor) -> bool {
        tensor.dims() == self.shape
            && tensor.dtype() == self.dtype
            && tensor.device().location() == self.device
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorIdentity {
    id: TensorId,
    spec: TensorSpec,
}

impl TensorIdentity {
    fn new(tensor: &Tensor) -> Self {
        Self {
            id: tensor.id(),
            spec: TensorSpec::new(tensor),
        }
    }

    fn matches(&self, tensor: &Tensor) -> bool {
        tensor.id() == self.id && self.spec.matches(tensor)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedConditioningRequest {
    model_load_id: u64,
    host: HostConditioningIdentity,
    input: TensorSpec,
    llm_features: TensorIdentity,
    position_ids: TensorIdentity,
}

impl PreparedConditioningRequest {
    fn new(
        model_load_id: u64,
        host: HostConditioningIdentity,
        input: &Tensor,
        llm_features: &Tensor,
        position_ids: &Tensor,
    ) -> Result<Self> {
        let (b, l, _) = input.dims3()?;
        let (llm_b, llm_l, _) = llm_features.dims3()?;
        let (position_b, position_l, axes) = position_ids.dims3()?;
        if (llm_b, llm_l) != (b, l) || (position_b, position_l, axes) != (b, l, 3) {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "ideogram: conditioning geometry does not match input [batch={b}, tokens={l}]"
            )));
        }
        let device = input.device().location();
        if llm_features.device().location() != device || position_ids.device().location() != device
        {
            return Err(candle_gen::candle_core::Error::Msg(
                "ideogram: conditioning tensors and input must share a device".into(),
            ));
        }
        Ok(Self {
            model_load_id,
            host,
            input: TensorSpec::new(input),
            llm_features: TensorIdentity::new(llm_features),
            position_ids: TensorIdentity::new(position_ids),
        })
    }

    fn validate(
        &self,
        model_load_id: u64,
        host: HostConditioningIdentity,
        input: &Tensor,
        llm_features: &Tensor,
        position_ids: &Tensor,
    ) -> Result<()> {
        if model_load_id != self.model_load_id
            || host != self.host
            || !self.input.matches(input)
            || !self.llm_features.matches(llm_features)
            || !self.position_ids.matches(position_ids)
        {
            return Err(candle_gen::candle_core::Error::Msg(
                "ideogram: prepared conditioning request identity does not match model, geometry, position grid, dtype, or device".into(),
            ));
        }
        Ok(())
    }
}

/// The step-invariant conditioning tensors prepared once per render (sc-11280). `seg_mask = None` when
/// every token shares a segment id — the additive mask is provably all-zeros, so the per-block add is
/// skipped entirely (softmax over `scores + 0` == softmax over `scores`, so the step is byte-identical).
pub(crate) struct PreparedConditioning {
    request: PreparedConditioningRequest,
    img_mask: Tensor,
    cos: Tensor,
    sin: Tensor,
    seg_mask: Option<Tensor>,
    llm: Tensor,
    indicator_emb: Tensor,
}

impl Ideogram4Transformer {
    /// Load a DiT from a component dir of `.safetensors` (top-level keys: `input_proj.*`,
    /// `layers.{i}.*`, `final_layer.*`, …). `w`'s dtype is the DiT compute dtype (bf16).
    pub fn load(w: &Weights, cfg: &Ideogram4DitConfig) -> Result<Self> {
        let head_dim = cfg.emb_dim / cfg.num_heads;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(Ideogram4Block::load(
                w,
                &format!("layers.{i}"),
                cfg.num_heads,
                head_dim,
                cfg.norm_eps,
            )?);
        }
        // Sinusoidal freqs: half = emb_dim/2, lf = ln(1e4)/(half-1), f[d] = exp(-lf·d).
        let half = cfg.emb_dim / 2;
        let lf = (1e4f32).ln() / (half as f32 - 1.0);
        let t_freqs: Vec<f32> = (0..half).map(|d| (-lf * d as f32).exp()).collect();
        let t_freqs = Tensor::from_vec(t_freqs, (1, half), w.device())?;

        let embed_image_indicator = embedding_detect(w, "embed_image_indicator")?;

        Ok(Self {
            input_proj: linear_detect(w, "input_proj", true)?,
            llm_cond_norm: w.get("llm_cond_norm.weight")?,
            llm_cond_proj: linear_detect(w, "llm_cond_proj", true)?,
            t_mlp_in: linear_detect(w, "t_embedding.mlp_in", true)?,
            t_mlp_out: linear_detect(w, "t_embedding.mlp_out", true)?,
            adaln_proj: linear_detect(w, "adaln_proj", true)?,
            embed_image_indicator,
            rotary_emb: Ideogram4MRoPE::new(
                head_dim,
                cfg.rope_theta,
                cfg.mrope_section,
                w.device(),
            )?,
            layers: TransformerLayers::Resident(layers),
            final_adaln: linear_detect(w, "final_layer.adaln_modulation", true)?,
            final_linear: linear_detect(w, "final_layer.linear", true)?,
            t_freqs,
            dtype: w.dtype(),
            load_id: next_ideogram_load_id(),
        })
    }

    /// Build the exact deferred-materialization form: top-level projections remain resident for the
    /// denoise phase, while the 34 trunk blocks stay mmap-backed and are materialized in request-sized
    /// windows. Adapter factors are reattached to every materialized subset in their original order.
    pub(crate) fn load_streamed(
        weights: Weights,
        cfg: &Ideogram4DitConfig,
        window_size: usize,
        turbo_adapter: Option<&Path>,
        user_adapters: &[AdapterSpec],
    ) -> Result<Self> {
        if window_size == 0 || window_size > cfg.num_layers {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "ideogram: invalid transformer window {window_size} for {} layers",
                cfg.num_layers
            )));
        }
        let weights = Arc::new(weights);
        let mut transformer = Self::load_top(
            &weights,
            cfg,
            TransformerLayers::Streamed(StreamedLayers {
                weights: Arc::clone(&weights),
                config: *cfg,
                window_size,
                turbo_adapter: turbo_adapter.map(Path::to_path_buf),
                user_adapters: user_adapters.to_vec(),
            }),
        )?;
        transformer.install_top_adapters()?;
        Ok(transformer)
    }

    fn load_top(w: &Weights, cfg: &Ideogram4DitConfig, layers: TransformerLayers) -> Result<Self> {
        let head_dim = cfg.emb_dim / cfg.num_heads;
        let half = cfg.emb_dim / 2;
        let lf = (1e4f32).ln() / (half as f32 - 1.0);
        let t_freqs = Tensor::from_vec(
            (0..half)
                .map(|d| (-lf * d as f32).exp())
                .collect::<Vec<_>>(),
            (1, half),
            w.device(),
        )?;
        Ok(Self {
            input_proj: linear_detect(w, "input_proj", true)?,
            llm_cond_norm: w.get("llm_cond_norm.weight")?,
            llm_cond_proj: linear_detect(w, "llm_cond_proj", true)?,
            t_mlp_in: linear_detect(w, "t_embedding.mlp_in", true)?,
            t_mlp_out: linear_detect(w, "t_embedding.mlp_out", true)?,
            adaln_proj: linear_detect(w, "adaln_proj", true)?,
            embed_image_indicator: embedding_detect(w, "embed_image_indicator")?,
            rotary_emb: Ideogram4MRoPE::new(
                head_dim,
                cfg.rope_theta,
                cfg.mrope_section,
                w.device(),
            )?,
            layers,
            final_adaln: linear_detect(w, "final_layer.adaln_modulation", true)?,
            final_linear: linear_detect(w, "final_layer.linear", true)?,
            t_freqs,
            dtype: w.dtype(),
            load_id: next_ideogram_load_id(),
        })
    }

    fn visit_top_adaptable_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        f("input_proj", &mut self.input_proj)?;
        f("llm_cond_proj", &mut self.llm_cond_proj)?;
        f("t_embedding.mlp_in", &mut self.t_mlp_in)?;
        f("t_embedding.mlp_out", &mut self.t_mlp_out)?;
        f("adaln_proj", &mut self.adaln_proj)?;
        f("final_layer.adaln_modulation", &mut self.final_adaln)?;
        f("final_layer.linear", &mut self.final_linear)
    }

    fn install_top_adapters(&mut self) -> Result<()> {
        let TransformerLayers::Streamed(streamed) = &self.layers else {
            return Ok(());
        };
        let turbo = streamed
            .turbo_adapter
            .as_ref()
            .filter(|path| adapter_targets(path, None).unwrap_or(true))
            .cloned();
        let adapters = streamed
            .user_adapters
            .iter()
            .filter(|adapter| adapter_targets(&adapter.path, None).unwrap_or(true))
            .cloned()
            .collect::<Vec<_>>();
        let device = self.device();
        if let Some(path) = turbo {
            crate::adapters::install_turbo_lora_additive_for_visitor(
                &device,
                &path,
                crate::config::TURBO_LORA_SCALE,
                |visitor| self.visit_top_adaptable_mut(visitor),
            )?;
        }
        if !adapters.is_empty() {
            candle_gen::quant::install_dotted_adapters(
                "ideogram streamed top-level",
                &adapters,
                &device,
                |visitor| self.visit_top_adaptable_mut(visitor),
            )
            .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))?;
        }
        Ok(())
    }

    /// The DiT's compute device (every weight loaded onto it) — the device a resolved additive residual
    /// factor is moved to before it is pushed onto a projection (sc-11104).
    pub fn device(&self) -> candle_gen::candle_core::Device {
        self.t_freqs.device().clone()
    }

    /// Walk **every** adaptable projection in the DiT with its canonical diffusers-style dotted path
    /// (sc-11104): the top-level `input_proj` / `llm_cond_proj` / `t_embedding.mlp_{in,out}` /
    /// `adaln_proj`, each block's `layers.{i}.*`, and the final layer's `final_layer.{adaln_modulation,
    /// linear}`. These are the keys a prefix-stripped TurboTime-LoRA module resolves against, so the
    /// additive installer ([`crate::adapters::install_turbo_lora_additive`]) can push a residual onto any
    /// matched projection while leaving the base packed. Ordered by walk, top-level first.
    pub fn visit_adaptable_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        f("input_proj", &mut self.input_proj)?;
        f("llm_cond_proj", &mut self.llm_cond_proj)?;
        f("t_embedding.mlp_in", &mut self.t_mlp_in)?;
        f("t_embedding.mlp_out", &mut self.t_mlp_out)?;
        f("adaln_proj", &mut self.adaln_proj)?;
        if let TransformerLayers::Resident(layers) = &mut self.layers {
            for (i, layer) in layers.iter_mut().enumerate() {
                layer.visit_adaptable_mut(&format!("layers.{i}"), f)?;
            }
        }
        f("final_layer.adaln_modulation", &mut self.final_adaln)?;
        f("final_layer.linear", &mut self.final_linear)?;
        Ok(())
    }

    /// Sinusoidal scalar-`t` embedding → MLP. `t`: `[B]` in `[0,1]` → `[B, emb_dim]`.
    fn t_embedding(&self, t: &Tensor) -> Result<Tensor> {
        let scaled = (t.to_dtype(DType::F32)? * 1e4)?; // [B]
        let emb = scaled.unsqueeze(1)?.broadcast_mul(&self.t_freqs)?; // [B, half]
        let emb = Tensor::cat(&[emb.sin()?, emb.cos()?], D::Minus1)?.to_dtype(self.dtype)?;
        let h = self.t_mlp_in.forward(&emb)?.silu()?;
        self.t_mlp_out.forward(&h)
    }

    /// Velocity prediction `[B, L, in_channels]` (f32). Inputs follow the upstream packing:
    /// `llm_features [B,L,llm_dim]`, `x [B,L,in_ch]`, `t [B]`, `position_ids [B,L,3]`,
    /// `segment_ids [B,L]`, `indicator [B,L]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        llm_features: &Tensor,
        x: &Tensor,
        t: &Tensor,
        position_ids: &Tensor,
        segment_ids: &Tensor,
        indicator: &Tensor,
        attention_budget: usize,
        cancel: &CancelFlag,
    ) -> Result<Tensor> {
        let prepared = self.prepare(llm_features, x, indicator, segment_ids, position_ids)?;
        let host = prepared.request.host;
        self.forward_prepared(
            x,
            t,
            llm_features,
            position_ids,
            host,
            &prepared,
            attention_budget,
            cancel,
        )
    }

    /// Materialize request-scoped geometry conditioning. The caller owns the handle, rather than the
    /// model retaining a mutable cache, so repeated forwards never need a device-to-host key readback.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_from_host(
        &self,
        llm_features: &Tensor,
        input: &Tensor,
        indicator: &[i64],
        segment_ids: &[i64],
        position_ids: &Tensor,
        host: HostConditioningIdentity,
    ) -> Result<PreparedConditioning> {
        let (b, l, _) = input.dims3()?;
        if indicator.len() != b * l || segment_ids.len() != b * l {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "ideogram: host conditioning lengths ({}, {}) must equal batch*tokens {}",
                indicator.len(),
                segment_ids.len(),
                b * l
            )));
        }
        if !host.matches_host(indicator, segment_ids) {
            return Err(candle_gen::candle_core::Error::Msg(
                "ideogram: host conditioning identity does not match role or segment packing"
                    .into(),
            ));
        }
        let request = PreparedConditioningRequest::new(
            self.load_id,
            host,
            input,
            llm_features,
            position_ids,
        )?;
        let (llm_mask, img_mask, img_idx) =
            role_tensors(indicator, b, l, self.dtype, position_ids.device())?;
        let (cos, sin) = self.rotary_emb.forward(position_ids)?;
        let seg_mask = segment_mask(segment_ids, b, l, position_ids.device())?;
        let llm_features = llm_features
            .to_dtype(self.dtype)?
            .broadcast_mul(&llm_mask)?;
        let llm = rmsnorm(&llm_features, &self.llm_cond_norm, COND_NORM_EPS)?;
        let llm = self.llm_cond_proj.forward(&llm)?.broadcast_mul(&llm_mask)?;
        let indicator_emb = self.embed_image_indicator.forward(&img_idx)?;
        Ok(PreparedConditioning {
            request,
            img_mask,
            cos,
            sin,
            seg_mask,
            llm,
            indicator_emb,
        })
    }

    /// Compatibility preparation entry for one-off callers. Render loops must use
    /// [`prepare_from_host`](Self::prepare_from_host) while the packing is still host-owned.
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        &self,
        llm_features: &Tensor,
        input: &Tensor,
        indicator: &Tensor,
        segment_ids: &Tensor,
        position_ids: &Tensor,
    ) -> Result<PreparedConditioning> {
        let ind = indicator
            .to_dtype(DType::I64)?
            .flatten_all()?
            .to_vec1::<i64>()?;
        let seg = segment_ids
            .to_dtype(DType::I64)?
            .flatten_all()?
            .to_vec1::<i64>()?;
        let host = HostConditioningIdentity::new(&ind, &seg);
        self.prepare_from_host(llm_features, input, &ind, &seg, position_ids, host)
    }

    /// Run one denoise forward against request-scoped prepared conditioning.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_prepared(
        &self,
        x: &Tensor,
        t: &Tensor,
        llm_features: &Tensor,
        position_ids: &Tensor,
        host: HostConditioningIdentity,
        prepared: &PreparedConditioning,
        attention_budget: usize,
        cancel: &CancelFlag,
    ) -> Result<Tensor> {
        prepared
            .request
            .validate(self.load_id, host, x, llm_features, position_ids)?;
        let img_mask = &prepared.img_mask;
        let cos = &prepared.cos;
        let sin = &prepared.sin;
        let seg_mask = &prepared.seg_mask;

        let x = x.to_dtype(self.dtype)?.broadcast_mul(img_mask)?;
        let x = self.input_proj.forward(&x)?.broadcast_mul(img_mask)?;

        let t_cond = self.t_embedding(t)?.unsqueeze(1)?; // [B,1,emb]
        let adaln_input = self.adaln_proj.forward(&t_cond)?.silu()?; // [B,1,adaln]

        let mut h = (&x + &prepared.llm)?;
        h = (h + &prepared.indicator_emb)?;

        match &self.layers {
            TransformerLayers::Resident(layers) => {
                for layer in layers {
                    if cancel.is_cancelled() {
                        return Err(candle_gen::candle_core::Error::Msg(
                            "ideogram: generation canceled".into(),
                        ));
                    }
                    h = layer.forward(
                        &h,
                        cos,
                        sin,
                        seg_mask.as_ref(),
                        &adaln_input,
                        attention_budget,
                    )?;
                }
            }
            TransformerLayers::Streamed(streamed) => {
                for first in (0..streamed.config.num_layers).step_by(streamed.window_size) {
                    if cancel.is_cancelled() {
                        return Err(candle_gen::candle_core::Error::Msg(
                            "ideogram: generation canceled".into(),
                        ));
                    }
                    let count = streamed.window_size.min(streamed.config.num_layers - first);
                    let mut window = MaterializedWindow {
                        layers: Vec::with_capacity(count),
                        device: streamed.weights.device().clone(),
                    };
                    for index in first..first + count {
                        window.layers.push((
                            index,
                            Ideogram4Block::load(
                                &streamed.weights,
                                &format!("layers.{index}"),
                                streamed.config.num_heads,
                                streamed.config.head_dim,
                                streamed.config.norm_eps,
                            )?,
                        ));
                    }
                    install_window_adapters(streamed, &mut window)?;
                    for (_, layer) in &window.layers {
                        h = layer.forward(
                            &h,
                            cos,
                            sin,
                            seg_mask.as_ref(),
                            &adaln_input,
                            attention_budget,
                        )?;
                    }
                    // The drop guard synchronizes before releasing this window.
                    drop(window);
                }
            }
        }

        // Final layer: scale = 1 + adaln(silu(c)); linear(layernorm_no_affine(h) · scale).
        let scale = (self.final_adaln.forward(&adaln_input.silu()?)? + 1.0)?;
        let normed = layer_norm_no_affine(&h, FINAL_NORM_EPS)?;
        let out = self.final_linear.forward(&normed.broadcast_mul(&scale)?)?;
        out.to_dtype(DType::F32)
    }
}

fn adapter_layer_index(name: &str) -> Option<usize> {
    for marker in ["layers.", "layers_"] {
        if let Some(rest) = name.split(marker).nth(1) {
            let digits = rest
                .chars()
                .take_while(|character| character.is_ascii_digit())
                .collect::<String>();
            if !digits.is_empty() {
                return digits.parse().ok();
            }
        }
    }
    None
}

fn adapter_targets(path: &Path, window: Option<(usize, usize)>) -> Result<bool> {
    let headers = candle_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(path)
        .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))?;
    Ok(headers
        .iter()
        .any(|header| match (window, adapter_layer_index(&header.name)) {
            (None, None) => true,
            (Some((first, count)), Some(index)) => (first..first + count).contains(&index),
            _ => false,
        }))
}

fn install_window_adapters(
    streamed: &StreamedLayers,
    window: &mut MaterializedWindow,
) -> Result<()> {
    let Some(first) = window.layers.first().map(|(first, _)| *first) else {
        return Ok(());
    };
    let count = window.layers.len();
    let device = window.device.clone();
    if let Some(path) = streamed
        .turbo_adapter
        .as_ref()
        .filter(|path| adapter_targets(path, Some((first, count))).unwrap_or(true))
    {
        crate::adapters::install_turbo_lora_additive_for_visitor(
            &device,
            path,
            crate::config::TURBO_LORA_SCALE,
            |visitor| {
                for (index, layer) in &mut window.layers {
                    layer.visit_adaptable_mut(&format!("layers.{index}"), visitor)?;
                }
                Ok(())
            },
        )?;
    }
    let adapters = streamed
        .user_adapters
        .iter()
        .filter(|adapter| adapter_targets(&adapter.path, Some((first, count))).unwrap_or(true))
        .cloned()
        .collect::<Vec<_>>();
    if !adapters.is_empty() {
        candle_gen::quant::install_dotted_adapters(
            "ideogram streamed transformer window",
            &adapters,
            &device,
            |visitor| {
                for (index, layer) in &mut window.layers {
                    layer.visit_adaptable_mut(&format!("layers.{index}"), visitor)?;
                }
                Ok(())
            },
        )
        .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))?;
    }
    Ok(())
}

/// No-affine LayerNorm over the last dim (computed in f32 for stability, cast back to `x`'s dtype).
fn layer_norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    normed.to_dtype(dt)
}

/// From `indicator` (host `[B·L]`, row-major): `(llm_mask [B,L,1], img_mask [B,L,1]` at `dtype`,
/// `img_idx [B,L]` u32). `img_idx` = 1 at image tokens, 0 elsewhere (the `embed_image_indicator`
/// lookup index). Takes the already-host-read `indicator` slice so the device→host round-trip happens
/// once per render, not once per step (sc-8992).
fn role_tensors(
    ind: &[i64],
    b: usize,
    l: usize,
    dtype: DType,
    dev: &candle_gen::candle_core::Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let n = b * l;
    let mut llm = vec![0f32; n];
    let mut img = vec![0f32; n];
    let mut idx = vec![0u32; n];
    for (p, &v) in ind.iter().enumerate().take(n) {
        if v == LLM_TOKEN_INDICATOR {
            llm[p] = 1.0;
        }
        if v == OUTPUT_IMAGE_INDICATOR {
            img[p] = 1.0;
            idx[p] = 1;
        }
    }
    Ok((
        Tensor::from_vec(llm, (b, l, 1), dev)?.to_dtype(dtype)?,
        Tensor::from_vec(img, (b, l, 1), dev)?.to_dtype(dtype)?,
        Tensor::from_vec(idx, (b, l), dev)?,
    ))
}

/// Additive attention mask `[B, 1, L, L]` (f32): `0` where two tokens share a `segment_id`, `-inf`
/// otherwise (full bidirectional attention within a packed sample — not causal). Takes the
/// already-host-read `seg` slice (sc-8992).
///
/// Returns `None` when **every** token shares one segment id — the mask would be all-zeros, so the
/// caller skips the per-block additive step entirely (`softmax(scores + 0) == softmax(scores)`, so the
/// step is byte-identical). This pipeline always packs a single uniform segment, so `None` is the hot
/// path and the ~`B·L²`-element allocation + per-block broadcast-add are avoided.
fn segment_mask(
    seg: &[i64],
    b: usize,
    l: usize,
    dev: &candle_gen::candle_core::Device,
) -> Result<Option<Tensor>> {
    let uniform = seg.iter().all(|&s| Some(s) == seg.first().copied());
    if uniform {
        return Ok(None);
    }
    let mut data = vec![0f32; b * l * l];
    for bi in 0..b {
        for i in 0..l {
            for j in 0..l {
                if seg[bi * l + i] != seg[bi * l + j] {
                    data[(bi * l + i) * l + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Ok(Some(Tensor::from_vec(data, (b, 1, l, l), dev)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    /// A uniform segment (every token shares one id) yields `None` — the caller skips the per-block
    /// additive mask, which is the always-taken path in this pipeline (sc-8992).
    #[test]
    fn segment_mask_uniform_is_none() {
        let dev = Device::Cpu;
        assert!(segment_mask(&[7, 7, 7, 7], 1, 4, &dev).unwrap().is_none());
        // A single-token sequence is trivially uniform.
        assert!(segment_mask(&[3], 1, 1, &dev).unwrap().is_none());
    }

    /// A non-uniform segment builds the additive `[B,1,L,L]` mask: `0` within a segment, `-inf` across.
    #[test]
    fn segment_mask_non_uniform_places_neg_inf_across_segments() {
        let dev = Device::Cpu;
        // Tokens 0,1 in segment 0; tokens 2,3 in segment 1.
        let m = segment_mask(&[0, 0, 1, 1], 1, 4, &dev).unwrap().unwrap();
        assert_eq!(m.dims(), &[1, 1, 4, 4]);
        let v = m.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let at = |i: usize, j: usize| v[i * 4 + j];
        // Same-segment pairs are 0; cross-segment pairs are -inf.
        assert_eq!(at(0, 1), 0.0);
        assert_eq!(at(2, 3), 0.0);
        assert!(at(0, 2).is_infinite() && at(0, 2) < 0.0);
        assert!(at(3, 1).is_infinite() && at(3, 1) < 0.0);
    }

    #[test]
    fn prepared_conditioning_request_reuses_once_and_rejects_stale_identity() -> Result<()> {
        let device = Device::Cpu;
        let input = Tensor::zeros((1, 4, 8), DType::F32, &device)?;
        let llm = Tensor::zeros((1, 4, 12), DType::F32, &device)?;
        let positions = Tensor::zeros((1, 4, 3), DType::I64, &device)?;
        let host = HostConditioningIdentity::new(&[3, 3, 2, 2], &[0, 0, 0, 0]);
        let prepared = PreparedConditioningRequest::new(23, host, &input, &llm, &positions)?;

        // The latent value changes each step, but the prepared request remains valid.
        for _ in 0..2 {
            let next = Tensor::ones((1, 4, 8), DType::F32, &device)?;
            prepared.validate(23, host, &next, &llm, &positions)?;
        }

        // Tensor identity catches a changed same-shape position grid; rebuilding accepts it.
        let changed_positions = Tensor::ones((1, 4, 3), DType::I64, &device)?;
        assert!(prepared
            .validate(23, host, &input, &llm, &changed_positions)
            .is_err());
        let rebuilt = PreparedConditioningRequest::new(23, host, &input, &llm, &changed_positions)?;
        rebuilt.validate(23, host, &input, &llm, &changed_positions)?;

        let changed_llm = Tensor::ones((1, 4, 12), DType::F32, &device)?;
        assert!(prepared
            .validate(23, host, &input, &changed_llm, &positions)
            .is_err());
        let wrong_geometry = Tensor::zeros((1, 5, 8), DType::F32, &device)?;
        assert!(prepared
            .validate(23, host, &wrong_geometry, &llm, &positions)
            .is_err());
        let wrong_dtype = input.to_dtype(DType::F16)?;
        assert!(prepared
            .validate(23, host, &wrong_dtype, &llm, &positions)
            .is_err());
        let mut wrong_device = prepared.clone();
        wrong_device.input.device = DeviceLocation::Cuda { gpu_id: 9 };
        assert!(wrong_device
            .validate(23, host, &input, &llm, &positions)
            .is_err());
        let new_host = HostConditioningIdentity::new(&[3, 2, 2, 2], &[0, 0, 0, 0]);
        assert!(prepared
            .validate(23, new_host, &input, &llm, &positions)
            .is_err());
        let rebuilt_host =
            PreparedConditioningRequest::new(23, new_host, &input, &llm, &positions)?;
        rebuilt_host.validate(23, new_host, &input, &llm, &positions)?;
        assert!(prepared
            .validate(24, host, &input, &llm, &positions)
            .is_err());
        Ok(())
    }

    #[test]
    fn streamed_adapter_routing_covers_top_first_middle_and_last_windows() {
        use std::collections::HashMap;

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("adapter.safetensors");
        let mut tensors = HashMap::new();
        for name in [
            "input_proj.lora_A.weight",
            "layers.0.attention.qkv.lora_A.weight",
            "layers.16.feed_forward.w1.lora_A.weight",
            "layers.33.attention.o.lora_A.weight",
        ] {
            tensors.insert(
                name.to_owned(),
                Tensor::ones((1, 1), DType::F32, &Device::Cpu).unwrap(),
            );
        }
        candle_gen::candle_core::safetensors::save(&tensors, &path).unwrap();
        assert!(adapter_targets(&path, None).unwrap());
        assert!(adapter_targets(&path, Some((0, 4))).unwrap());
        assert!(adapter_targets(&path, Some((16, 4))).unwrap());
        assert!(adapter_targets(&path, Some((32, 2))).unwrap());
        assert!(!adapter_targets(&path, Some((4, 4))).unwrap());
        assert_eq!(
            adapter_layer_index("lora_unet_layers_33_attention_o"),
            Some(33)
        );
    }
}
