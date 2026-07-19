//! Krea 2's **Qwen3-VL-4B-Instruct** condition encoder (text path only — the vision tower is unused
//! for text-to-image). A 36-layer decoder-only LM; the hidden states at the 12 evenly-spaced indices
//! `text_encoder_select_layers = [2,5,…,35]` are **stacked** (not aggregated here) into
//! `[B, L, 12, 2560]` — the exact `context` the DiT's `TextFusionTransformer` consumes (sc-7569). The
//! learned aggregation lives in the DiT, NOT here. Port of `mlx-gen-krea`'s `text_encoder/`,
//! structured like `candle-gen-boogu`'s Qwen3-VL text encoder.
//!
//! GQA (32 query / 8 kv heads, **decoupled** head_dim 128 so q_proj is 4096-wide while hidden is
//! 2560), bias-less q/k/v/o, **per-head q/k RMSNorm**, HF half-split RoPE (θ = 5e6), SwiGLU MLP,
//! pre-norm causal decoder blocks. **Computes in f32** — the parity-grade precision for this exact
//! encoder in the sibling boogu/ideogram ports; the DiT casts the features down to bf16. The weights
//! are **stored bf16** (sc-12828): the embedding is upcast to f32 and each projection runs
//! `QLinear::forward_upcast` (bf16 weight → f32 per matmul), with the RMSNorm weights loaded f32, so
//! the forward is bit-identical to an f32 store at half the resident footprint (the disk weights are
//! bf16, so an f32 store only widened them). The `load_text` render path stores bf16; the
//! training/ControlNet paths load f32, where `forward_upcast` is inert.
//!
//! HF `hidden_states` indexing: `hidden_states[i]` is the state after running `i` decoder layers
//! (`hidden_states[0]` = the raw embedding), so the reference's `select_hidden = [2,5,…,35]` capture
//! the OUTPUT of 0-indexed layers `[1,4,…,34]`. The final `language_model.norm` is never applied (all
//! selected layers are pre-final-norm), and only `max+1` layers are run.

use std::path::Path;

use candle_gen::candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::rotary_emb::rope;
// Shared Qwen3-VL grounding helpers (sc-11205 / F-118) — the MRoPE / vision-splice machinery this
// encoder previously defined inline ("ported verbatim from candle_gen_boogu"), byte-identical to
// Boogu's copy. Now one shared home in `candle_gen::grounding`.
use candle_gen::grounding::{
    causal_mask, image_blocks, mrope_cos_sin, mrope_positions, repeat_kv, replace_seq, slice_seq,
    Rotary,
};

use crate::loader::{embedding_detect, linear_detect, rmsnorm, Weights};
use crate::quant::{QEmbedding, QLinear};

/// Qwen3-VL-4B text-tower architecture (verified from the published `text_encoder/config.json`:
/// `qwen3_vl_text`, hidden 2560, 36 layers, GQA 32/8, head_dim 128, FFN 9728, eps 1e-6) + the Krea
/// conditioning policy (which hidden-state layers to stack, how many template-prefix tokens to drop).
#[derive(Debug, Clone, PartialEq)]
pub struct KreaTeConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    /// HF `output_hidden_states` indices the pipeline stacks (`model_index.json`
    /// `text_encoder_select_layers`): `hidden_states[i]` = the LM state after running `i` layers.
    pub select_hidden: Vec<usize>,
    /// Leading template-prefix tokens dropped from the conditioning (`Qwen3VLConditioner`'s
    /// `prompt_template_encode_start_idx`); the system-instruction prefix tokenizes to this many.
    pub prefix_tokens: usize,
    /// `<|image_pad|>` id (the vision-embed splice placeholder) — image-grounded edit path only
    /// (epic 10871 / sc-10880). Standard Qwen3-VL `151655` (confirmed for Krea, sc-10875).
    pub image_token_id: u32,
    /// Qwen3-VL `text_config.rope_parameters.mrope_section` — the per-axis (T/H/W) frequency counts over
    /// `head_dim/2 = 64` used by the 3-D interleaved MRoPE on the image-grounded path (`[24, 20, 20]`).
    pub mrope_section: [usize; 3],
}

impl KreaTeConfig {
    pub fn qwen3_vl_4b() -> Self {
        Self {
            num_layers: 36,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            select_hidden: vec![2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
            prefix_tokens: 34,
            image_token_id: 151655,
            mrope_section: [24, 20, 20],
        }
    }

    /// Parse `<root>/text_encoder/config.json` (`text_config`) + `<root>/model_index.json`
    /// (`text_encoder_select_layers`); missing scalars fall back to [`Self::qwen3_vl_4b`].
    pub fn from_snapshot(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let path = root.join("text_encoder").join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| {
            candle_gen::candle_core::Error::Msg(format!("krea te: read {}: {e}", path.display()))
        })?;
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
            candle_gen::candle_core::Error::Msg(format!("krea te: parse {}: {e}", path.display()))
        })?;
        let tc = v.get("text_config").unwrap_or(&v);
        let d = Self::qwen3_vl_4b();
        let u = |k: &str, dflt: usize| {
            tc.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(dflt)
        };

        let mut cfg = Self {
            num_layers: u("num_hidden_layers", d.num_layers),
            num_heads: u("num_attention_heads", d.num_heads),
            num_kv_heads: u("num_key_value_heads", d.num_kv_heads),
            head_dim: u("head_dim", d.head_dim),
            rms_norm_eps: tc
                .get("rms_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(d.rms_norm_eps),
            // `text_config.rope_theta` is null on disk; honor `rope_parameters`/`rope_scaling` if set,
            // else the qwen3_vl_text default (5e6).
            rope_theta: tc
                .get("rope_parameters")
                .or_else(|| tc.get("rope_scaling"))
                .and_then(|r| r.get("rope_theta"))
                .or_else(|| tc.get("rope_theta"))
                .and_then(serde_json::Value::as_f64)
                .map(|n| n as f32)
                .unwrap_or(d.rope_theta),
            select_hidden: d.select_hidden.clone(),
            prefix_tokens: d.prefix_tokens,
            // `image_token_id` is a top-level config field; `mrope_section` lives under
            // `rope_parameters`/`rope_scaling`. Both fall back to the standard Qwen3-VL values (sc-10875).
            image_token_id: v
                .get("image_token_id")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as u32)
                .unwrap_or(d.image_token_id),
            mrope_section: tc
                .get("rope_parameters")
                .or_else(|| tc.get("rope_scaling"))
                .and_then(|r| r.get("mrope_section"))
                .and_then(serde_json::Value::as_array)
                .and_then(|a| read_mrope_section(a))
                .unwrap_or(d.mrope_section),
        };

        // `text_encoder_select_layers` lives in the pipeline manifest. A genuinely-absent
        // `model_index.json` keeps the reference `select_hidden` default; a *present-but-corrupt*
        // manifest (I/O error or malformed JSON) errors loudly rather than silently downgrading to the
        // default on a damaged snapshot (sc-9010 / F-073).
        if let Some(mv) = read_optional_model_index(&root.join("model_index.json"))? {
            if let Some(arr) = mv
                .get("text_encoder_select_layers")
                .and_then(|a| a.as_array())
            {
                let sel: Vec<usize> = arr
                    .iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect();
                if !sel.is_empty() {
                    cfg.select_hidden = sel;
                }
            }
        }
        Ok(cfg)
    }
}

/// Read the **optional** `model_index.json`, distinguishing "genuinely absent" (→ `Ok(None)`, keep
/// the reference default) from "present but corrupt" (→ `Err`, name the file). A missing manifest is a
/// legitimate snapshot shape; an I/O error or malformed JSON on a manifest that *is* present signals a
/// damaged/partial download that must surface rather than silently downgrade behavior (sc-9010 /
/// F-073).
fn read_optional_model_index(path: &Path) -> Result<Option<serde_json::Value>> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea te: read {}: {e}",
                path.display()
            )))
        }
    };
    let v = serde_json::from_str(&text).map_err(|e| {
        candle_gen::candle_core::Error::Msg(format!(
            "krea te: parse {} (corrupt snapshot?): {e}",
            path.display()
        ))
    })?;
    Ok(Some(v))
}

/// Parse a JSON `mrope_section` array into `[t, h, w]` (exactly three positive counts); any other shape
/// falls back (`None`) to the [`KreaTeConfig::qwen3_vl_4b`] default.
fn read_mrope_section(a: &[serde_json::Value]) -> Option<[usize; 3]> {
    if a.len() != 3 {
        return None;
    }
    let mut out = [0usize; 3];
    for (i, x) in a.iter().enumerate() {
        out[i] = x.as_u64()? as usize;
    }
    Some(out)
}

struct Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    eps: f64,
}

impl Attention {
    fn load(w: &Weights, prefix: &str, cfg: &KreaTeConfig) -> Result<Self> {
        Ok(Self {
            q_proj: linear_detect(w, &format!("{prefix}.q_proj"), false)?,
            k_proj: linear_detect(w, &format!("{prefix}.k_proj"), false)?,
            v_proj: linear_detect(w, &format!("{prefix}.v_proj"), false)?,
            o_proj: linear_detect(w, &format!("{prefix}.o_proj"), false)?,
            // f32 store (sc-12828): the norm weights are tiny, always kept f32 regardless of the
            // encoder's bf16 weight store, so `rmsnorm` runs on the f32 hidden state with an f32
            // weight — bit-identical to the old all-f32 load (the disk weights are bf16, so f32 just
            // widens them). The bulk projections stay bf16 and upcast per matmul (`forward_upcast`).
            q_norm: w.get_f32(&format!("{prefix}.q_norm.weight"))?,
            k_norm: w.get_f32(&format!("{prefix}.k_norm.weight"))?,
            n_heads: cfg.num_heads,
            n_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.n_heads, self.n_kv_heads, self.head_dim);

        // `forward_upcast`, not `forward` (sc-12828): the projections are stored bf16 but the hidden
        // state is f32, so each weight is upcast to f32 per matmul — bit-identical to an f32-stored
        // projection (bf16→f32 is exact) at half the resident footprint. Inert (an `Arc` clone, no
        // copy) on the f32-stored training/control paths, where `x` already matches the weight dtype.
        let q = self.q_proj.forward_upcast(x)?.reshape((b, s, nh, hd))?;
        let k = self.k_proj.forward_upcast(x)?.reshape((b, s, nkv, hd))?;
        let v = self.v_proj.forward_upcast(x)?.reshape((b, s, nkv, hd))?;
        // Per-head q/k RMSNorm over the head dim, then transpose to [B, H, S, D].
        let q = rmsnorm(&q, &self.q_norm, self.eps)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = rmsnorm(&k, &self.k_norm, self.eps)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let q = rope(&q, cos, sin)?;
        let k = rope(&k, cos, sin)?;
        let k = repeat_kv(&k, nh / nkv)?;
        let v = repeat_kv(&v, nh / nkv)?;

        let scale = (hd as f64).powf(-0.5);
        // i32-overflow guard (sc-11154 / F-081): image-grounded edit prompts run right up to the
        // inclusive `MAX_EDIT_TOKENS = 8192` cap, so the `[B, nh, S, S]` scores tensor reaches
        // `32·8192² = 2^31 > i32::MAX` — candle's CUDA kernels index scores with i32 and silently
        // corrupt the tail, subtly wrong grounding on the krea edit engine. Chunk over the query rows
        // via the shared helper (the additive causal mask is `[B,1,S,S]`, narrowed per chunk); single
        // un-chunked pass (byte-identical) below budget, exact fused `softmax_last_dim` preserved.
        let o = candle_gen::sdpa_budgeted_bhsd(
            &q,
            &k,
            &v,
            scale,
            Some(mask),
            softmax_last_dim,
            candle_gen::ATTN_SCORES_BUDGET,
        )?; // [B, nh, S, D]
        let o = o.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * hd))?;
        self.o_proj.forward_upcast(&o)
    }
}

struct Mlp {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

impl Mlp {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: linear_detect(w, &format!("{prefix}.gate_proj"), false)?,
            up: linear_detect(w, &format!("{prefix}.up_proj"), false)?,
            down: linear_detect(w, &format!("{prefix}.down_proj"), false)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // `forward_upcast` (sc-12828): bf16-stored projections, f32 hidden — see `Attention::forward`.
        let gated = (self.gate.forward_upcast(x)?.silu()? * self.up.forward_upcast(x)?)?;
        self.down.forward_upcast(&gated)
    }
}

struct DecoderLayer {
    input_ln: Tensor,
    post_ln: Tensor,
    attn: Attention,
    mlp: Mlp,
    eps: f64,
}

impl DecoderLayer {
    fn load(w: &Weights, prefix: &str, cfg: &KreaTeConfig) -> Result<Self> {
        Ok(Self {
            // f32 norm weights (sc-12828): tiny, kept f32 so `rmsnorm` runs f32-on-f32 regardless of
            // the encoder's bf16 weight store — bit-identical to the old all-f32 load.
            input_ln: w.get_f32(&format!("{prefix}.input_layernorm.weight"))?,
            post_ln: w.get_f32(&format!("{prefix}.post_attention_layernorm.weight"))?,
            attn: Attention::load(w, &format!("{prefix}.self_attn"), cfg)?,
            mlp: Mlp::load(w, &format!("{prefix}.mlp"))?,
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let h = (x + self
            .attn
            .forward(&rmsnorm(x, &self.input_ln, self.eps)?, cos, sin, mask)?)?;
        &h + self.mlp.forward(&rmsnorm(&h, &self.post_ln, self.eps)?)?
    }
}

/// The Krea Qwen3-VL-4B text-path condition encoder.
pub struct KreaTextEncoder {
    embed_tokens: QEmbedding,
    layers: Vec<DecoderLayer>,
    rotary: Rotary,
    /// 0-indexed decoder-layer OUTPUT indices to capture (= `select_hidden[i] - 1`), in stack order.
    out_layers: Vec<usize>,
    prefix_tokens: usize,
    // ── image-grounded edit path (epic 10871 / sc-10880) ────────────────────────────────────────
    head_dim: usize,
    rms_norm_eps: f64,
    rope_theta: f32,
    /// Qwen3-VL MRoPE per-axis (T/H/W) frequency counts over `head_dim/2` for the 3-D interleaved rope.
    mrope_section: [usize; 3],
    /// `<|image_pad|>` id — the vision-embed splice placeholder.
    image_token_id: u32,
    device: Device,
}

impl KreaTextEncoder {
    /// Load from the `text_encoder` weights under `prefix` (`"language_model"`). The final
    /// `{prefix}.norm.weight` is intentionally not loaded. `max_seq` sizes the RoPE table.
    pub fn load(w: &Weights, prefix: &str, cfg: &KreaTeConfig, max_seq: usize) -> Result<Self> {
        let out_layers: Vec<usize> = cfg
            .select_hidden
            .iter()
            .map(|&s| {
                s.checked_sub(1).ok_or_else(|| {
                    candle_gen::candle_core::Error::Msg(
                        "krea te: select_hidden index 0 has no layer output".into(),
                    )
                })
            })
            .collect::<Result<_>>()?;
        let max_layer = *out_layers.iter().max().unwrap_or(&0);
        if max_layer >= cfg.num_layers {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea te: select_hidden needs layer {max_layer} but the encoder has {} layers",
                cfg.num_layers
            )));
        }

        let embed_tokens = embedding_detect(w, &format!("{prefix}.embed_tokens"))?;

        let mut layers = Vec::with_capacity(max_layer + 1);
        for i in 0..=max_layer {
            layers.push(DecoderLayer::load(w, &format!("{prefix}.layers.{i}"), cfg)?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            rotary: Rotary::new(cfg.head_dim, cfg.rope_theta, max_seq.max(1), w.device())?,
            out_layers,
            prefix_tokens: cfg.prefix_tokens,
            head_dim: cfg.head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
            rope_theta: cfg.rope_theta,
            mrope_section: cfg.mrope_section,
            image_token_id: cfg.image_token_id,
            device: w.device().clone(),
        })
    }

    /// `input_ids`: `[1, S]` u32. Returns the stacked conditioning `[1, S - prefix_tokens, num_select,
    /// hidden]` (the DiT's `context`), f32. The final norm is never applied; only layers up to
    /// `max(out_layers)` are run. Causal (decoder-only); no padding (the candle tokenizer emits none).
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        let (cos, sin) = self.rotary.text(s)?;
        let mask = causal_mask(b, s, &self.device)?;

        let mut hidden = self.embed_tokens.forward(input_ids)?.to_dtype(DType::F32)?;
        let mut saved: Vec<(usize, Tensor)> = Vec::with_capacity(self.out_layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
            if self.out_layers.contains(&i) {
                saved.push((i, hidden.clone()));
            }
        }
        self.stack_and_trim(&saved)
    }

    /// Image-grounded condition encoding for the edit path (epic 10871 / sc-10880). The image-only
    /// sibling of [`forward`](Self::forward): the vision tower's merged embeds are spliced over each
    /// reference's `<|image_pad|>` block, the decoder runs under the 3-D **interleaved MRoPE** (each
    /// reference's grid advancing the shared position counter), and each reference's `deepstack[k]`
    /// features are injected at its block after LM layers 0/1/2 — mirroring `Qwen3VLTextModel` with one
    /// `<|image_pad|>` block per reference. The same select-layer stack + template-prefix drop tail as
    /// [`forward`](Self::forward) produces the DiT's `context` `[1, S − prefix, num_select, hidden]`.
    ///
    /// - `input_ids`: `[1, S]` u32 from [`crate::tokenizer::KreaTokenizer::encode_with_images`] (the edit
    ///   template with `<|vision_start|><|image_pad|>×n<|vision_end|>` per reference).
    /// - `image_embeds[k]`: the k-th reference's merged vision embeds `[n_k, hidden]`.
    /// - `deepstack[k]`: its 3 deepstack features (each `[n_k, hidden]`).
    /// - `grids[k]`: its patch grid `[t, h, w]`.
    ///
    /// The block order must match the reference order (the template emits the references' vision blocks
    /// in order, before the instruction). `b = 1`.
    pub fn forward_with_images(
        &self,
        input_ids: &Tensor,
        image_embeds: &[Tensor],
        deepstack: &[Vec<Tensor>],
        grids: &[[i32; 3]],
    ) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        let ids: Vec<u32> = input_ids.i(0)?.to_dtype(DType::U32)?.to_vec1::<u32>()?;

        // Contiguous `<|image_pad|>` blocks, in order; block k carries reference k.
        let blocks = image_blocks(&ids, self.image_token_id);
        if blocks.len() != image_embeds.len() {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea te: {} image-token blocks in input_ids but {} reference embeds",
                blocks.len(),
                image_embeds.len()
            )));
        }

        // Token embeddings (f32), then splice each reference's vision embeds over its block. Each
        // replacement is the same length as the block, so earlier splices don't shift later indices.
        let mut hidden = self.embed_tokens.forward(input_ids)?.to_dtype(DType::F32)?;
        for (k, &(start, len)) in blocks.iter().enumerate() {
            if image_embeds[k].dim(0)? != len {
                return Err(candle_gen::candle_core::Error::Msg(format!(
                    "krea te: reference {k} has {} vision tokens but its image block is {len}",
                    image_embeds[k].dim(0)?
                )));
            }
            let img = image_embeds[k].unsqueeze(0)?.to_dtype(hidden.dtype())?; // [1, n_k, hidden]
            hidden = replace_seq(&hidden, &img, start, start + len, s)?;
        }

        // 3-D interleaved MRoPE (per-image grids) + causal mask (shared grounding helpers, sc-11205).
        let (pt, ph, pw) = mrope_positions(&ids, self.image_token_id, grids);
        let (cos, sin) = mrope_cos_sin(
            self.head_dim,
            self.mrope_section,
            self.rope_theta,
            &pt,
            &ph,
            &pw,
            &self.device,
        )?;
        let mask = causal_mask(b, s, &self.device)?;

        let mut saved: Vec<(usize, Tensor)> = Vec::with_capacity(self.out_layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
            // Deepstack: after LM layers 0/1/2, add each reference's layer-i feature at its block.
            for (k, &(start, len)) in blocks.iter().enumerate() {
                if i < deepstack[k].len() {
                    let ds = deepstack[k][i].unsqueeze(0)?.to_dtype(hidden.dtype())?; // [1, n_k, hidden]
                    let mid = (slice_seq(&hidden, start, start + len)? + ds)?;
                    hidden = replace_seq(&hidden, &mid, start, start + len, s)?;
                }
            }
            if self.out_layers.contains(&i) {
                saved.push((i, hidden.clone()));
            }
        }
        self.stack_and_trim(&saved)
    }

    /// Stack the captured select layers on a NEW axis 2 → `[b, s, n, hidden]` (reference
    /// `torch.stack([hidden_states[i] for i in select], dim=2)`), then drop the leading template-prefix
    /// tokens. Shared by [`forward`](Self::forward) and [`forward_with_images`](Self::forward_with_images)
    /// so the text and image-grounded paths stack identically.
    fn stack_and_trim(&self, saved: &[(usize, Tensor)]) -> Result<Tensor> {
        let pick = |idx: usize| -> Result<Tensor> {
            saved
                .iter()
                .find(|(k, _)| *k == idx)
                .map(|(_, v)| v.clone())
                .ok_or_else(|| {
                    candle_gen::candle_core::Error::Msg(format!(
                        "krea te: hidden state {idx} not captured"
                    ))
                })
        };
        let expanded: Vec<Tensor> = self
            .out_layers
            .iter()
            .map(|&idx| pick(idx)?.unsqueeze(2))
            .collect::<Result<_>>()?;
        let stacked = Tensor::cat(&expanded, 2)?; // [b, s, n, hidden]

        // Drop the leading template-prefix tokens (the system instruction).
        let n = stacked.dim(1)?;
        if self.prefix_tokens >= n {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea te: prompt has {n} tokens but the {} template-prefix tokens leave nothing",
                self.prefix_tokens
            )));
        }
        stacked.narrow(1, self.prefix_tokens, n - self.prefix_tokens)
    }

    /// The RMS-norm eps (exposed for the parity harness); the config value threaded at load.
    pub fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_layers_map_to_zero_indexed_outputs() {
        let cfg = KreaTeConfig::qwen3_vl_4b();
        assert_eq!(cfg.select_hidden.len(), 12);
        assert_eq!(cfg.select_hidden.first().copied(), Some(2));
        assert_eq!(cfg.select_hidden.last().copied(), Some(35));
        // The OUTPUT-of-layer mapping is `select - 1`: captures layers 1..34.
        let out: Vec<usize> = cfg.select_hidden.iter().map(|s| s - 1).collect();
        assert_eq!(out.first().copied(), Some(1));
        assert_eq!(out.last().copied(), Some(34));
        assert!(*out.iter().max().unwrap() < cfg.num_layers);
    }

    fn te_snapshot_tmp(name: &str) -> std::path::PathBuf {
        let tmp = std::env::temp_dir().join(format!(
            "krea_te_{name}_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("text_encoder")).unwrap();
        // A minimal valid text_encoder/config.json (missing scalars default to qwen3_vl_4b).
        std::fs::write(
            tmp.join("text_encoder").join("config.json"),
            br#"{"text_config": {}}"#,
        )
        .unwrap();
        tmp
    }

    #[test]
    fn from_snapshot_defaults_select_when_model_index_absent() {
        // No model_index.json → keep the reference select_hidden default.
        let tmp = te_snapshot_tmp("idx_absent");
        let cfg = KreaTeConfig::from_snapshot(&tmp).unwrap();
        assert_eq!(cfg.select_hidden, KreaTeConfig::qwen3_vl_4b().select_hidden);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn from_snapshot_reads_present_select_layers() {
        let tmp = te_snapshot_tmp("idx_present");
        std::fs::write(
            tmp.join("model_index.json"),
            br#"{"text_encoder_select_layers": [1, 2, 3]}"#,
        )
        .unwrap();
        let cfg = KreaTeConfig::from_snapshot(&tmp).unwrap();
        assert_eq!(cfg.select_hidden, vec![1, 2, 3]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn from_snapshot_errors_on_corrupt_model_index() {
        // model_index.json present but malformed (partial download) → error, NOT silent default.
        let tmp = te_snapshot_tmp("idx_corrupt");
        std::fs::write(tmp.join("model_index.json"), b"{ not json").unwrap();
        assert!(KreaTeConfig::from_snapshot(&tmp).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn te_config_carries_grounding_defaults() {
        let c = KreaTeConfig::qwen3_vl_4b();
        assert_eq!(c.image_token_id, 151655);
        assert_eq!(c.mrope_section, [24, 20, 20]);
    }

    const IMG: u32 = 151655;

    #[test]
    fn image_blocks_finds_runs_in_order() {
        // text, text, [4 image], text, [2 image], text.
        let ids = [9u32, 9, IMG, IMG, IMG, IMG, 9, IMG, IMG, 9];
        assert_eq!(image_blocks(&ids, IMG), vec![(2, 4), (7, 2)]);
    }

    #[test]
    fn mrope_positions_advance_across_two_images() {
        // Block 0 ↔ grid [1,4,4] (merged 2×2 = 4 tokens, t-step max(4,4)/2 = 2);
        // block 1 ↔ grid [1,4,2] (merged 2×1 = 2 tokens, t-step max(4,2)/2 = 2). The image-1-then-image-2
        // fixed order (sc-10878) is exactly the grid order fed here.
        let ids = [9u32, 9, IMG, IMG, IMG, IMG, 9, IMG, IMG, 9];
        let grids = [[1, 4, 4], [1, 4, 2]];
        let (pt, ph, pw) = mrope_positions(&ids, IMG, &grids);
        assert_eq!(pt.len(), ids.len());
        assert_eq!((pt[0], pt[1]), (0, 1));
        assert_eq!(&pt[2..6], &[2, 2, 2, 2]);
        assert_eq!(&ph[2..6], &[2, 2, 3, 3]);
        assert_eq!(&pw[2..6], &[2, 3, 2, 3]);
        assert_eq!(pt[6], 4);
        assert_eq!(&pt[7..9], &[5, 5]);
        assert_eq!(&ph[7..9], &[5, 6]);
        assert_eq!(&pw[7..9], &[5, 5]);
        assert_eq!(pt[9], 7);
    }

    // ── sc-12828: bf16 weight store, f32 compute — bit-identical to an f32 store ──────────────────

    /// A tiny valid Qwen3-VL text-encoder weight map (2 layers, hidden 6, GQA 2/1, decoupled head_dim
    /// 4) drawn as **bf16** — modelling the hosted TE, whose weights ship bf16 on disk. Serialized so it
    /// can be loaded at either store dtype through the shipping `Weights` path.
    fn tiny_te_map() -> (std::collections::HashMap<String, Tensor>, KreaTeConfig) {
        let cfg = KreaTeConfig {
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            select_hidden: vec![1, 2], // out layers 0, 1
            prefix_tokens: 1,
            image_token_id: IMG,
            mrope_section: [24, 20, 20],
        };
        let (hidden, inter, vocab) = (6usize, 8usize, 12usize);
        let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        let bf16 = |shape: &[usize]| {
            Tensor::randn(0f32, 0.5f32, shape, &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let mut t = std::collections::HashMap::new();
        t.insert(
            "language_model.embed_tokens.weight".to_string(),
            bf16(&[vocab, hidden]),
        );
        for i in 0..cfg.num_layers {
            let p = format!("language_model.layers.{i}");
            t.insert(format!("{p}.input_layernorm.weight"), bf16(&[hidden]));
            t.insert(
                format!("{p}.post_attention_layernorm.weight"),
                bf16(&[hidden]),
            );
            t.insert(
                format!("{p}.self_attn.q_proj.weight"),
                bf16(&[nh * hd, hidden]),
            );
            t.insert(
                format!("{p}.self_attn.k_proj.weight"),
                bf16(&[nkv * hd, hidden]),
            );
            t.insert(
                format!("{p}.self_attn.v_proj.weight"),
                bf16(&[nkv * hd, hidden]),
            );
            t.insert(
                format!("{p}.self_attn.o_proj.weight"),
                bf16(&[hidden, nh * hd]),
            );
            t.insert(format!("{p}.self_attn.q_norm.weight"), bf16(&[hd]));
            t.insert(format!("{p}.self_attn.k_norm.weight"), bf16(&[hd]));
            t.insert(format!("{p}.mlp.gate_proj.weight"), bf16(&[inter, hidden]));
            t.insert(format!("{p}.mlp.up_proj.weight"), bf16(&[inter, hidden]));
            t.insert(format!("{p}.mlp.down_proj.weight"), bf16(&[hidden, inter]));
        }
        (t, cfg)
    }

    fn save_tiny_te(
        map: &std::collections::HashMap<String, Tensor>,
        tag: &str,
    ) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "krea_te_bf16_{tag}_{}_{:?}.safetensors",
            std::process::id(),
            std::thread::current().id()
        ));
        candle_gen::candle_core::safetensors::save(map, &path).unwrap();
        path
    }

    /// The parity gate (sc-12828): a bf16 weight **store** with f32 **compute** is bit-identical to an
    /// f32 store. The hosted TE ships bf16 on disk, so an f32 store only widens those exact values and
    /// every matmul still runs f32 (the projections upcast via `forward_upcast`, the RMSNorm weights
    /// load f32). Reverting `forward_upcast` → `forward` or the f32 norm load turns the bf16 path into a
    /// dtype-mismatch error, so this test goes RED — it is not a tautology that passes with the win
    /// ripped out. Runs on CPU precisely because the compute never leaves f32 (no bf16 matmul).
    #[test]
    fn bf16_store_forward_is_bit_identical_to_f32_store() {
        let (map, cfg) = tiny_te_map();
        let path = save_tiny_te(&map, "biteq");
        let ids = Tensor::from_vec(vec![1u32, 5, 3, 9], (1, 4), &Device::Cpu).unwrap();

        let w_f32 = Weights::from_file(&path, &Device::Cpu, DType::F32).unwrap();
        let te_f32 = KreaTextEncoder::load(&w_f32, "language_model", &cfg, 64).unwrap();
        let out_f32 = te_f32.forward(&ids).unwrap();

        let w_bf16 = Weights::from_file(&path, &Device::Cpu, DType::BF16).unwrap();
        let te_bf16 = KreaTextEncoder::load(&w_bf16, "language_model", &cfg, 64).unwrap();
        let out_bf16 = te_bf16.forward(&ids).unwrap();

        // The encoder computes f32 either way; output is the prefix-trimmed [1, S-1, n_select, hidden].
        assert_eq!(out_f32.dtype(), DType::F32);
        assert_eq!(out_bf16.dtype(), DType::F32);
        assert_eq!(out_f32.dims(), &[1, 3, 2, 6]);
        let a = out_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = out_bf16.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(a.iter().all(|x| x.is_finite()), "context must be finite");
        assert_eq!(
            a, b,
            "bf16-store forward must be bit-identical to the f32-store forward"
        );

        // The footprint win is real, not just an output cast: at the bf16 store the bulk projection
        // weight is bf16 (half the f32 resident); `linear_detect`'s dense arm stores exactly
        // `Weights::get`, so this pins the storage dtype the resident-VRAM saving rides on.
        let q = "language_model.layers.0.self_attn.q_proj.weight";
        assert_eq!(w_bf16.get(q).unwrap().dtype(), DType::BF16);
        assert_eq!(w_f32.get(q).unwrap().dtype(), DType::F32);

        let _ = std::fs::remove_file(&path);
    }

    /// The **image-grounded edit** path (`forward_with_images`) is bit-identical under the bf16 store
    /// too (sc-12828): same shared decoder layers as the text path (`forward_upcast`), plus the
    /// vision-embed splice + deepstack injection on the f32 hidden. Pins that the edit encode stays f32.
    #[test]
    fn bf16_store_grounded_is_bit_identical_to_f32_store() {
        let (map, mut cfg) = tiny_te_map();
        cfg.image_token_id = 0; // in-vocab (vocab 12) so the tiny embed table is valid
        let path = save_tiny_te(&map, "grounded");
        let dev = Device::Cpu;

        // 1 text, a 4-token image block (id 0, a 2×2 merged grid → grid [1,4,4]), 1 text. S=6 > prefix 1.
        let ids = Tensor::from_vec(vec![1u32, 0, 0, 0, 0, 1], (1, 6), &dev).unwrap();
        let (n, hidden) = (4usize, 6usize);
        let embeds = vec![(Tensor::ones((n, hidden), DType::F32, &dev).unwrap() * 0.5).unwrap()];
        let deepstack = vec![(0..3)
            .map(|k| {
                (Tensor::ones((n, hidden), DType::F32, &dev).unwrap() * (0.01 * (k + 1) as f64))
                    .unwrap()
            })
            .collect::<Vec<_>>()];
        let grids = [[1i32, 4, 4]];

        let run = |store: DType| -> Tensor {
            let w = Weights::from_file(&path, &dev, store).unwrap();
            KreaTextEncoder::load(&w, "language_model", &cfg, 64)
                .unwrap()
                .forward_with_images(&ids, &embeds, &deepstack, &grids)
                .unwrap()
        };
        let out_f32 = run(DType::F32);
        let out_bf16 = run(DType::BF16);

        assert_eq!(out_bf16.dtype(), DType::F32);
        assert_eq!(out_f32.dims(), &[1, 5, 2, 6]);
        let a = out_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = out_bf16.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(a.iter().all(|x| x.is_finite()));
        assert_eq!(
            a, b,
            "bf16-store grounded context must be bit-identical to the f32-store forward"
        );

        let _ = std::fs::remove_file(&path);
    }
}
