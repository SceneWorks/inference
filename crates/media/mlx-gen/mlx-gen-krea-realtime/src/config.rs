//! Krea Realtime Video 14B configuration (epic 8431, sc-8435 S2).
//!
//! Krea Realtime 14B is **Wan 2.1 T2V 14B, weight-for-weight** (verified in the S1 audit): its DiT
//! dimensions are exactly [`WanModelConfig::wan21_t2v_14b`] — dim 5120, 40 layers, 40 heads, head_dim
//! 128, ffn_dim 13824, in/out 16, freq_dim 256, text_len 512, text_dim 4096, patch (1,2,2), eps 1e-6.
//! The model reuses the stock Wan 2.1 text encoder (UMT5-XXL), z16 VAE, and tokenizer — the shipped
//! checkpoint is **transformer-only**.
//!
//! What makes Krea Realtime distinct from stock Wan is not its weights but its **inference regime**:
//! an autoregressive, self-forcing, real-time denoiser that runs a short `denoising_step_list` per
//! frame-block with a KV cache and (optional) local/block-sparse causal attention. Those knobs are
//! carried on [`KreaArConfig`] here so the preset is complete, but they are **not consumed until
//! S3–S5** (causal attention / KV cache / the AR loop). For S2 the DiT loads into the reused
//! `mlx_gen_wan` transformer using only the [`WanModelConfig`] half.
//!
//! The AR knob defaults and the token-geometry helpers (`max_attention_size` / `sink_tokens` /
//! `block_size`) are **adapted from** the reference `causal_model.py` (`krea-ai/realtime-video`): the
//! shipped values are the reference model's published config (facts), and the geometry formulas are
//! reimplemented from the reference's token-window algebra.

use std::path::Path;

use mlx_gen::{Error, Result};
use mlx_gen_wan::config::WanModelConfig;
use serde_json::Value;

/// Engine / provider id for Krea Realtime 14B. Distinct from the unrelated **image** crate
/// `mlx-gen-krea` (Krea 2 Turbo, engine `krea_2_turbo`) — this is the autoregressive **video** model.
/// The registered [`Generator`](crate::KreaRealtime) is composed under this id (sc-8439 S6).
pub const MODEL_ID: &str = "krea_realtime_14b";

/// Bytes MLX spends on one affine-quantization group's `scale` **and** `bias`. `mlx_rs::ops::quantize`
/// emits both in the **input's** dtype, and the KV cache quantizes bf16 post-RoPE K / raw V, so each
/// group costs 2 + 2 bytes on top of its packed payload. This is the term that makes Q8 KV `0.53×`
/// bf16 rather than a clean half — see [`KvCacheQuant::row_bytes`].
const QUANT_GROUP_OVERHEAD_BYTES: usize = 4;

/// Bits per element of the shipped (unquantized) KV cache — bf16 post-RoPE keys + raw values.
const DENSE_KV_BITS: usize = 16;

/// Group-wise affine quantization of the **persistent self-attention KV cache** (sc-17807).
///
/// The KV cache holds *activations*, not weights, so a Q4 DiT does not shrink it: for the Wan-14B
/// backbone the shipped bf16 cache costs `2 (K and V) × 40 layers × 5120 dim × 2 bytes` =
/// **819,200 bytes (800 KiB) per DiT token** (see [`KreaRealtimeConfig::kv_bytes_per_token`]).
///
/// Unlike the ~9 GiB of Q4 weights, that term **scales with the clip**, which is what makes it the
/// lever. Precisely: at the shipped Mac bounded window (6 latent frames = 9,360 tokens at 832×480)
/// the KV is 7.14 GiB — *comparable to* the weights, not yet dominant. It overtakes them at the
/// first wider window and runs away from there: 17.9 GiB at 15 frames, 35.7 at 30, and 53.6 for the
/// checkpoint's global window over a 45-frame clip. So "the KV dominates" is a statement about
/// where this model goes, not about its smallest configuration.
///
/// **How it is spent, and why not a quantized attention kernel.** MLX at the pinned revision exposes
/// **no fused quantized SDPA**: `mlx_rs::fast::scaled_dot_product_attention` takes dense arrays, and
/// the only quantized primitives are `quantize` / `dequantize` / `quantized_matmul`. Consuming a packed
/// cache directly therefore means the decomposed `quantized_matmul → softmax → quantized_matmul` form
/// (mlx-lm's `quantized_scaled_dot_product_attention`), which **materializes the whole `Sq × Sk` score
/// matrix** that the fused kernel never builds. That is fine for LLM decode, where `Sq = 1`; for one
/// Krea AR chunk `Sq` is a full frame-block (4,680 tokens at 832×480), and *one layer's* bf16 score
/// matrix (3.50 GB) is already the size of the Q8 saving on the *whole 40-layer cache* (3.59 GB) —
/// while costing **37×** what dequantizing that layer's read window costs (0.096 GB). A `precise` f32
/// softmax, which is what mlx-lm uses, doubles the transient again. Both figures are computed by
/// `decomposed_quantized_attention_costs_far_more_than_dequantizing_the_window` in `causal.rs`. So the
/// cache stores packed and **dequantizes the read window per layer**, keeping the fused SDPA path. The
/// dequantized window is an anonymous graph intermediate (built and dropped inside one chunk forward,
/// before the AR loop's per-step `eval`), so MLX frees it as each layer's attention completes; what
/// stays resident across chunks is the packed cache.
///
/// Grouping runs along the **last** axis of the cached `[B, n, S, head_dim]` K/V — i.e. within a head,
/// exactly the layout `take_axis`/`concatenate_axis` on the token axis leave untouched — so
/// `head_dim` must be a multiple of [`group_size`](Self::group_size) (128 / 64 for this backbone).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvCacheQuant {
    /// Bits per packed element. MLX's affine mode supports 2, 3, 4, 5, 6 and 8.
    pub bits: i32,
    /// Elements sharing one scale/bias. MLX's affine mode supports 32, 64 and 128.
    pub group_size: i32,
}

impl KvCacheQuant {
    /// Q8 KV at MLX's default group size — the tier sc-17807 measured (`0.53×` the bf16 cache).
    pub const Q8: Self = Self {
        bits: 8,
        group_size: 64,
    };

    /// Q4 KV at MLX's default group size (`0.28×` the bf16 cache). Materially lossier than
    /// [`Q8`](Self::Q8) and **not** measured by the sc-17807 sweep — treat it as unvalidated.
    pub const Q4: Self = Self {
        bits: 4,
        group_size: 64,
    };

    /// Reject a tier MLX's affine quantization cannot express, before it becomes an opaque MLX
    /// exception several layers down inside a chunk forward.
    pub fn validate(&self) -> Result<()> {
        if !matches!(self.bits, 2 | 3 | 4 | 5 | 6 | 8) {
            return Err(Error::Msg(format!(
                "krea KV cache: {} bits is not an MLX affine quantization width (2, 3, 4, 5, 6, 8)",
                self.bits
            )));
        }
        if !matches!(self.group_size, 32 | 64 | 128) {
            return Err(Error::Msg(format!(
                "krea KV cache: group size {} is not an MLX affine quantization group (32, 64, 128)",
                self.group_size
            )));
        }
        Ok(())
    }

    /// Bytes one row of `dim` cached elements occupies at this tier: the packed payload plus one
    /// scale **and** one bias per group. `dim` must be a multiple of
    /// [`group_size`](Self::group_size) — the same divisibility MLX's `quantize` enforces on the
    /// grouped axis.
    ///
    /// The per-group overhead is why Q8 is `0.53×` bf16 and not `0.50×`: at `group_size = 64` a group
    /// costs 64 packed bytes plus 4 bytes of bf16 scale/bias against bf16's 128.
    pub fn row_bytes(&self, dim: usize) -> Result<usize> {
        self.validate()?;
        let group_size = self.group_size as usize;
        if dim == 0 || !dim.is_multiple_of(group_size) {
            return Err(Error::Msg(format!(
                "krea KV cache: {dim} elements per row is not a positive multiple of the \
                 quantization group size {group_size}"
            )));
        }
        // Exact integer arithmetic: `bits · dim` is a whole number of bits because `dim` is a
        // multiple of the group size and every supported group size is a multiple of 8.
        let packed = dim * self.bits as usize / 8;
        Ok(packed + dim / group_size * QUANT_GROUP_OVERHEAD_BYTES)
    }
}

/// The autoregressive / self-forcing inference knobs Krea Realtime carries on top of the shared
/// Wan-2.1-T2V-14B DiT. These describe the AR **denoise regime**, not the weights, and are consumed by
/// S3 (causal attention), S4 (KV cache), and S5 (the self-forcing AR loop) — **not** by the S2 load
/// path. Values are the reference defaults from the Krea Realtime 14B release (S1 audit, 2026-07-26).
#[derive(Clone, Debug, PartialEq)]
pub struct KreaArConfig {
    /// Local (sliding-window) causal-attention span in **latent frames**, or `-1` for full causal
    /// attention over the whole generated history (the shipped default). The read window is
    /// `local_attn_size × frame_seq_length` tokens (reference `causal_model.py:192`). Consumed by S3/S5.
    pub local_attn_size: i64,
    /// Number of always-attended "sink" **latent frames** retained at the start of the KV cache (0 =
    /// none; the shipped default). The reference keeps the first `sink_size` frames unchanged when
    /// rolling the KV cache (`causal_model.py:359,584`). Consumed by S3/S5.
    pub sink_size: usize,
    /// Frame-blocks denoised together as one autoregressive chunk (3). Consumed by S5.
    pub num_frames_per_block: usize,
    /// Frame-blocks of context kept resident in the rolling KV cache (3). Consumed by S4.
    pub kv_cache_num_frames: usize,
    /// DiT token count contributed by a single latent frame at the model's canonical geometry (1560).
    /// Consumed by S4 to size the KV cache and by S5 for per-block token slicing.
    pub frame_seq_length: usize,
    /// Total DiT sequence length across the full generated clip at the canonical geometry (32760 =
    /// 21 latent frames × [`Self::frame_seq_length`]). Consumed by S4/S5.
    pub seq_length: usize,
    /// Offline self-forcing denoising timestep values. The product/release path uses this list's
    /// length, not its integer values, to derive the shifted float schedule; the direct offline
    /// scheduler API can still consume the values verbatim. One AR block runs that many forwards.
    pub denoising_step_list: Vec<u32>,
    /// Flow-match sigma shift applied to the schedule (5.0). Consumed by S5.
    pub timestep_shift: f32,
    /// Whether to run the **clean-context KV-cache recompute** after each chunk's denoise: rerun the
    /// transformer on the chunk's clean `x0` output at [`context_noise`](Self::context_noise) and commit
    /// *that* (clean-context) K/V for the chunk, instead of the near-clean final-denoise-step K/V. The
    /// shipped reference default is **on** (`configs/self_forcing_server_14b.yaml: do_kv_recomp: true`);
    /// turning it off falls back to the S4 behaviour (final denoise step commits) — the A/B baseline.
    /// Consumed by S5.
    pub do_kv_recomp: bool,
    /// The timestep the clean-context recompute forward runs at — the reference's `args.context_noise`,
    /// applied as an int64 timestep (`context_timestep = ones_like(timestep) * context_noise`,
    /// `pipeline/causal_inference.py:228`). The shipped default is `0` (`configs/default_config.yaml`),
    /// which the few-step schedule's argmin maps to the smallest tabled sigma `≈ 0.00498` — near-clean,
    /// not exactly clean. Consumed by S5.
    pub context_noise: f32,
    /// Group-wise affine quantization of the persistent self-attention KV cache, or `None` — the
    /// shipped default — to retain post-RoPE K / raw V as **bf16** (sc-17807). The reference has no
    /// such knob; this is a memory lever for the bounded-window Mac path, where the KV, not the Q4
    /// weights, is the dominant term (800 KiB per DiT token at bf16 —
    /// [`KreaRealtimeConfig::kv_bytes_per_token`]). Consumed by
    /// [`CausalKvCache`](crate::CausalKvCache).
    ///
    /// **Opt-in on purpose.** Quantizing the cache perturbs the same long-clip coherence sc-15571 /
    /// sc-15127 measure, so it is off by default and turning it on is a measured decision, not a free
    /// one — see the sc-17807 arm of the S18 sweep.
    pub kv_cache_quant: Option<KvCacheQuant>,
}

impl Default for KreaArConfig {
    fn default() -> Self {
        Self::krea_realtime_14b()
    }
}

impl KreaArConfig {
    /// Tokens per **attention block** = [`frame_seq_length`](Self::frame_seq_length) ×
    /// [`num_frames_per_block`](Self::num_frames_per_block) — the block-causal unit (a query attends to
    /// all tokens up to the end of its own block, intra-block bidirectional, inter-block strictly
    /// causal). At the shipped geometry: `1560 × 3 = 4680`. Consumed by S3's mask + KV cache.
    pub fn block_size(&self) -> usize {
        self.frame_seq_length * self.num_frames_per_block
    }

    /// The self-attention read window in **tokens**: `k[max(0, end - max_attention_size):end]`. Global
    /// (= [`seq_length`](Self::seq_length), `32760`) when [`local_attn_size`](Self::local_attn_size) is
    /// `-1` — the shipped checkpoint — else [`local_attn_size`](Self::local_attn_size) **frames** ×
    /// [`frame_seq_length`](Self::frame_seq_length). Mirrors the 14B reference exactly:
    /// `max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560`
    /// (`wan/modules/causal_model.py:192`, `CausalWanSelfAttention.__init__`). Note the unit is
    /// **frames × frame_seq_length**, *not* blocks — an earlier port multiplied by
    /// [`block_size`](Self::block_size) (= `frame_seq_length × num_frames_per_block`), overcounting the
    /// window by `num_frames_per_block`×.
    pub fn max_attention_size(&self) -> usize {
        if self.local_attn_size < 0 {
            self.seq_length
        } else {
            self.local_attn_size as usize * self.frame_seq_length
        }
    }

    /// Always-attended "sink" prefix in **tokens** = [`sink_size`](Self::sink_size) **frames** ×
    /// [`frame_seq_length`](Self::frame_seq_length) (`0` for the shipped checkpoint). Retained regardless
    /// of the sliding window. Mirrors the reference's `sink_tokens = self.sink_size * frame_seqlen`
    /// (`wan/modules/causal_model.py:359`) — frames × frame_seq_length, *not* blocks; the reference
    /// docstring is explicit: "we keep the first `sink_size` frames unchanged when rolling the KV cache".
    pub fn sink_tokens(&self) -> usize {
        self.sink_size * self.frame_seq_length
    }

    /// The bounded (streaming) local-attention window in **frames** for memory-feasible long-clip
    /// generation on Mac: [`kv_cache_num_frames`](Self::kv_cache_num_frames) +
    /// [`num_frames_per_block`](Self::num_frames_per_block) — the reference server's `attn_size`
    /// (`release_server.py:543`, `kv_cache_num_frames + num_frame_per_block`). Setting
    /// [`local_attn_size`](Self::local_attn_size) to this value bounds the KV read/store to
    /// `× frame_seq_length` tokens (≈ `6 · 1560 = 9360`) instead of the shipped global
    /// [`seq_length`](Self::seq_length) (`32760` ≈ 27 GB of KV on Mac). The shipped checkpoint itself is
    /// global (`local_attn_size = -1`); this is the Mac streaming bound the S6 pipeline selects.
    pub fn streaming_local_attn_frames(&self) -> usize {
        self.kv_cache_num_frames + self.num_frames_per_block
    }

    /// The shipped Krea Realtime 14B AR defaults (S1 audit).
    pub fn krea_realtime_14b() -> Self {
        Self {
            local_attn_size: -1,
            sink_size: 0,
            num_frames_per_block: 3,
            kv_cache_num_frames: 3,
            frame_seq_length: 1560,
            seq_length: 32760,
            denoising_step_list: vec![1000, 937, 833, 625, 0],
            timestep_shift: 5.0,
            do_kv_recomp: true,
            context_noise: 0.0,
            kv_cache_quant: None,
        }
    }
}

/// Krea Realtime 14B model configuration: the shared Wan-2.1-T2V-14B DiT dimensions
/// ([`WanModelConfig::wan21_t2v_14b`]) plus the [`KreaArConfig`] autoregressive knobs the base Wan
/// config has no home for.
#[derive(Clone, Debug, PartialEq)]
pub struct KreaRealtimeConfig {
    /// The Wan-2.1-T2V-14B DiT / VAE / scheduler dimensions Krea Realtime shares weight-for-weight.
    pub wan: WanModelConfig,
    /// The autoregressive / self-forcing inference knobs (carried now, consumed S3–S5).
    pub ar: KreaArConfig,
}

impl Default for KreaRealtimeConfig {
    fn default() -> Self {
        Self::krea_realtime_14b()
    }
}

impl KreaRealtimeConfig {
    /// The shipped Krea Realtime 14B config: `wan21_t2v_14b` weight-for-weight + the AR defaults.
    pub fn krea_realtime_14b() -> Self {
        Self {
            wan: WanModelConfig::wan21_t2v_14b(),
            ar: KreaArConfig::krea_realtime_14b(),
        }
    }

    /// Load from a model directory. Krea Realtime ships transformer-only (no bundled TE/VAE/tokenizer,
    /// which reuse stock Wan), and typically no `config.json`, so this starts from the shipped
    /// [`krea_realtime_14b`](Self::krea_realtime_14b) preset and overlays any Wan DiT fields present in
    /// an optional `config.json` (the Wan native layout). The AR knobs stay at their shipped defaults
    /// unless a `config.json` carries them.
    pub fn from_model_dir(root: &Path) -> Result<Self> {
        let mut cfg = Self::krea_realtime_14b();
        let path = root.join("config.json");
        if path.exists() {
            let text = std::fs::read_to_string(&path)?;
            let v: Value = serde_json::from_str(&text)
                .map_err(|e| Error::Msg(format!("krea-realtime: parse config.json: {e}")))?;
            // The Wan DiT half auto-detects/overlays via the shared Wan loader, then we force the
            // Wan2.1-dense identity Krea is (a bare transformer config would otherwise resolve to a
            // 2.2 preset). Only structural DiT fields present in the JSON take effect.
            cfg.wan = WanModelConfig::from_config_json(&v);
            cfg.wan.model_version = "2.1".into();
            cfg.wan.dual_model = false;
            overlay_ar(&v, &mut cfg.ar)?;
        }
        Ok(cfg)
    }

    /// Bytes of **persistent self-attention KV per DiT token**, across every layer — the quantity that
    /// actually sizes an autoregressive video run (sc-17807).
    ///
    /// `2 (K and V) × num_layers × row_bytes(dim)`, where a bf16 row costs `2·dim` bytes and a
    /// quantized row costs its packed payload plus a bf16 scale **and** bias per group
    /// ([`KvCacheQuant::row_bytes`]). At the shipped Wan-14B geometry (40 layers, dim 5120):
    ///
    /// | KV tier | bytes/token | KiB/token | vs bf16 |
    /// |---|---|---|---|
    /// | bf16 (shipped default) | 819,200 | 800 | 1.00× |
    /// | Q8 / group 64 | 435,200 | 425 | 0.53× |
    /// | Q4 / group 64 | 230,400 | 225 | 0.28× |
    ///
    /// Multiply by the retained window to get the cache: the Mac bounded window
    /// ([`streaming_local_attn_frames`](KreaArConfig::streaming_local_attn_frames) = 6 latent frames)
    /// at 832×480 is `6 × 1560 = 9,360` tokens ⇒ **7.14 GiB** bf16, 3.79 GiB at Q8; the checkpoint's
    /// global window over a 45-latent-frame clip is 70,200 tokens ⇒ **53.6 GiB** bf16.
    ///
    /// Errors when [`kv_cache_quant`](KreaArConfig::kv_cache_quant) names a tier MLX cannot express,
    /// or one whose group size does not divide the model's `head_dim` (the grouped axis).
    pub fn kv_bytes_per_token(&self) -> Result<usize> {
        let row = match self.ar.kv_cache_quant {
            None => self.wan.dim * DENSE_KV_BITS / 8,
            Some(q) => {
                // Grouping runs along `head_dim` (the cached tensor's last axis), so that — not the
                // full `dim` — is the axis the group size has to divide. Once it does, the whole
                // `dim` row is an exact multiple of the group too, which is what `row_bytes` costs.
                q.row_bytes(self.wan.head_dim())?;
                q.row_bytes(self.wan.dim)?
            }
        };
        Ok(2 * self.wan.num_layers * row)
    }

    /// Reject a [`kv_cache_quant`](KreaArConfig::kv_cache_quant) tier this backbone cannot express.
    ///
    /// Called on the request path so a snapshot `config.json` naming an impossible tier fails where
    /// the caller can see it, rather than several layers down inside the first chunk forward — by
    /// which point a long clip has already been paid for.
    pub fn validate_kv_cache_quant(&self) -> Result<()> {
        self.kv_bytes_per_token().map(|_| ())
    }

    /// Serialize to the `config.json` schema [`from_model_dir`](Self::from_model_dir) reads back — the
    /// shared Wan DiT half ([`WanModelConfig::to_json`], which carries the `quantization` block of a
    /// pre-quantized tier) plus the AR knobs `overlay_ar` consumes. Round-trips by construction (the
    /// test below asserts it), so a rehosted tier snapshot's config is generated from the preset rather
    /// than hand-written — the sc-15203 tier converter's config emitter.
    pub fn to_json(&self) -> Value {
        let mut v = self.wan.to_json();
        v["local_attn_size"] = serde_json::json!(self.ar.local_attn_size);
        v["sink_size"] = serde_json::json!(self.ar.sink_size);
        v["num_frames_per_block"] = serde_json::json!(self.ar.num_frames_per_block);
        v["kv_cache_num_frames"] = serde_json::json!(self.ar.kv_cache_num_frames);
        v["frame_seq_length"] = serde_json::json!(self.ar.frame_seq_length);
        v["seq_length"] = serde_json::json!(self.ar.seq_length);
        v["denoising_step_list"] = serde_json::json!(self.ar.denoising_step_list);
        v["timestep_shift"] = serde_json::json!(self.ar.timestep_shift);
        v["do_kv_recomp"] = serde_json::json!(self.ar.do_kv_recomp);
        v["context_noise"] = serde_json::json!(self.ar.context_noise);
        // Emitted only when the cache is quantized, mirroring the Wan half's `quantization` block: a
        // bf16-KV snapshot must not read as one carrying an (identity) KV tier.
        if let Some(q) = self.ar.kv_cache_quant {
            v["kv_cache_quant"] = serde_json::json!({
                "bits": q.bits,
                "group_size": q.group_size,
            });
        }
        v
    }
}

/// Overlay any AR knobs explicitly present in a `config.json` onto the shipped defaults.
///
/// Every knob but one degrades silently on a malformed value — a non-numeric `sink_size` simply keeps
/// the shipped default, which is the historical behaviour and stays that way. [`kv_cache_quant`] is
/// the exception, and deliberately: silently degrading it means running at **1.9× the intended
/// memory** while every log line and every recorded measurement says otherwise, which is the exact
/// failure [`CausalKvCache::new`](crate::CausalKvCache::new) takes an explicit tier parameter to
/// prevent. A snapshot that names the key must therefore name it correctly (sc-17807).
fn overlay_ar(v: &Value, ar: &mut KreaArConfig) -> Result<()> {
    if let Some(n) = v.get("local_attn_size").and_then(Value::as_i64) {
        ar.local_attn_size = n;
    }
    if let Some(n) = v.get("sink_size").and_then(Value::as_u64) {
        ar.sink_size = n as usize;
    }
    if let Some(n) = v.get("num_frames_per_block").and_then(Value::as_u64) {
        ar.num_frames_per_block = n as usize;
    }
    if let Some(n) = v.get("kv_cache_num_frames").and_then(Value::as_u64) {
        ar.kv_cache_num_frames = n as usize;
    }
    if let Some(n) = v.get("frame_seq_length").and_then(Value::as_u64) {
        ar.frame_seq_length = n as usize;
    }
    if let Some(n) = v.get("seq_length").and_then(Value::as_u64) {
        ar.seq_length = n as usize;
    }
    if let Some(arr) = v.get("denoising_step_list").and_then(Value::as_array) {
        let steps: Vec<u32> = arr
            .iter()
            .filter_map(|x| x.as_u64().map(|n| n as u32))
            .collect();
        if !steps.is_empty() {
            ar.denoising_step_list = steps;
        }
    }
    if let Some(n) = v.get("timestep_shift").and_then(Value::as_f64) {
        ar.timestep_shift = n as f32;
    }
    if let Some(b) = v.get("do_kv_recomp").and_then(Value::as_bool) {
        ar.do_kv_recomp = b;
    }
    if let Some(n) = v.get("context_noise").and_then(Value::as_f64) {
        ar.context_noise = n as f32;
    }
    // An explicit `null` selects the shipped bf16 cache, so the key being *present* is what decides —
    // a snapshot can turn the cache tier back off, not only on. Anything else present must be a
    // well-formed tier: a half-written or mistyped one is an error, NOT a quiet fall back to bf16.
    if let Some(q) = v.get("kv_cache_quant") {
        ar.kv_cache_quant = if q.is_null() {
            None
        } else {
            let field = |name: &str| -> Result<i32> {
                q.get(name)
                    .and_then(Value::as_i64)
                    .and_then(|n| i32::try_from(n).ok())
                    .ok_or_else(|| {
                        Error::Msg(format!(
                            "krea-realtime: config.json `kv_cache_quant` is present but its \
                             `{name}` is missing or not a 32-bit integer ({q}). Write both `bits` \
                             and `group_size`, or use `null` for the shipped bf16 cache — a \
                             malformed tier must not silently become bf16 and run at ~1.9x the \
                             intended KV memory."
                        ))
                    })
            };
            let quant = KvCacheQuant {
                bits: field("bits")?,
                group_size: field("group_size")?,
            };
            quant.validate()?;
            Some(quant)
        };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen_wan::config::GuideScale;

    #[test]
    fn shipped_14b_is_wan21_t2v_14b_weight_for_weight() {
        let c = KreaRealtimeConfig::krea_realtime_14b();
        // Identical to the Wan 2.1 T2V 14B DiT preset (S1: Krea Realtime is Wan2.1-14B weight-for-weight).
        assert_eq!(c.wan, WanModelConfig::wan21_t2v_14b());
        assert_eq!(c.wan.dim, 5120);
        assert_eq!(c.wan.num_layers, 40);
        assert_eq!(c.wan.num_heads, 40);
        assert_eq!(c.wan.head_dim(), 128);
        assert_eq!(c.wan.ffn_dim, 13824);
        assert_eq!(c.wan.in_dim, 16);
        assert_eq!(c.wan.out_dim, 16);
        assert_eq!(c.wan.freq_dim, 256);
        assert_eq!(c.wan.text_len, 512);
        assert_eq!(c.wan.text_dim, 4096);
        assert_eq!(c.wan.patch_size, (1, 2, 2));
        assert_eq!(c.wan.eps, 1e-6);
        assert_eq!(c.wan.model_version, "2.1");
        assert!(!c.wan.dual_model);
        assert_eq!(c.wan.vae_z_dim, 16);
        assert_eq!(c.wan.vae_stride, (4, 8, 8));
        assert_eq!(c.wan.sample_guide_scale, GuideScale::Single(5.0));
    }

    #[test]
    fn ar_defaults_match_reference() {
        let ar = KreaArConfig::krea_realtime_14b();
        assert_eq!(ar.local_attn_size, -1);
        assert_eq!(ar.sink_size, 0);
        assert_eq!(ar.num_frames_per_block, 3);
        assert_eq!(ar.kv_cache_num_frames, 3);
        assert_eq!(ar.frame_seq_length, 1560);
        assert_eq!(ar.seq_length, 32760);
        assert_eq!(ar.denoising_step_list, vec![1000, 937, 833, 625, 0]);
        assert_eq!(ar.timestep_shift, 5.0);
        // KV-cache recompute is on by default (self_forcing_server_14b.yaml: do_kv_recomp: true), and
        // the recompute runs at context_noise 0 (configs/default_config.yaml: context_noise: 0).
        assert!(ar.do_kv_recomp);
        assert_eq!(ar.context_noise, 0.0);
        // The canonical seq_length is 21 latent frames × frame_seq_length.
        assert_eq!(ar.seq_length, 21 * ar.frame_seq_length);
    }

    /// Reference-anchored token geometry (replaces a prior tautological test that asserted the code's
    /// own `× block_size` formula). The 14B reference measures the read window / sink in **frames ×
    /// frame_seq_length**, NOT blocks: `causal_model.py:192`
    /// `max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560`, and `:359`
    /// `sink_tokens = self.sink_size * frame_seqlen`. The independent numeric literals here (9360, 3120,
    /// and the rejected 28080) discriminate the earlier `× block_size` overcount.
    #[test]
    fn ar_token_geometry_matches_reference_units() {
        let ar = KreaArConfig::krea_realtime_14b();
        // Block size stays the block-causal unit: frame_seq_length × num_frames_per_block = 1560 × 3.
        assert_eq!(ar.block_size(), 4680);
        assert_eq!(
            ar.block_size(),
            ar.frame_seq_length * ar.num_frames_per_block
        );

        // Global checkpoint (local_attn_size = -1): the read window is the whole clip (reference's
        // else-branch constant 32760).
        assert_eq!(ar.max_attention_size(), 32760);
        assert_eq!(ar.max_attention_size(), ar.seq_length);
        // No attention sink on the shipped checkpoint.
        assert_eq!(ar.sink_tokens(), 0);

        // A windowed (local) config: local_attn_size = 6 frames, sink_size = 2 frames.
        // Reference: max_attention_size = 6 × 1560 = 9360; sink_tokens = 2 × 1560 = 3120.
        let mut local = ar.clone();
        local.local_attn_size = 6;
        local.sink_size = 2;
        assert_eq!(
            local.max_attention_size(),
            9360,
            "6 frames × frame_seq_length 1560 (reference: local_attn_size * frame_seqlen)"
        );
        assert_eq!(local.max_attention_size(), 6 * local.frame_seq_length);
        assert_eq!(
            local.sink_tokens(),
            3120,
            "2 frames × frame_seq_length 1560"
        );
        assert_eq!(local.sink_tokens(), 2 * local.frame_seq_length);
        // The block-based overcount the earlier port shipped would have been 6 × 4680 = 28080 — reject it.
        assert_ne!(local.max_attention_size(), 6 * local.block_size());
        assert_ne!(local.sink_tokens(), 2 * local.block_size());

        // The Mac streaming bound: kv_cache_num_frames (3) + num_frames_per_block (3) = 6 frames → 9360
        // tokens (release_server.py:543 attn_size), an order of magnitude under the global 32760.
        assert_eq!(ar.streaming_local_attn_frames(), 6);
        let mut streaming = ar.clone();
        streaming.local_attn_size = ar.streaming_local_attn_frames() as i64;
        assert_eq!(streaming.max_attention_size(), 9360);
        assert!(streaming.max_attention_size() < ar.seq_length);
    }

    /// **The per-token KV figure, computed rather than recited (sc-17807).**
    ///
    /// The crate previously documented the cache at 546 kB a token, which is wrong by 1.5× — the correct bf16
    /// cost at this backbone's geometry is `2 (K and V) × 40 layers × 5120 dim × 2 bytes` =
    /// **819,200 bytes = 800 KiB**. That is not a rounding quibble: at the checkpoint's global window
    /// over a 45-latent-frame 832×480 clip it is the difference between a documented "≈38 GiB of KV"
    /// and the 53.6 GiB the run actually holds, which is the gap between "tight on a 128 GiB host"
    /// and "does not fit". The literals here are derived independently of `kv_bytes_per_token`'s
    /// implementation, so the two have to agree.
    #[test]
    fn kv_bytes_per_token_is_800_kib_dense_and_425_kib_at_q8() {
        let cfg = KreaRealtimeConfig::krea_realtime_14b();
        assert_eq!(cfg.ar.kv_cache_quant, None, "the shipped cache is bf16");
        assert_eq!(cfg.kv_bytes_per_token().unwrap(), 819_200);
        assert_eq!(cfg.kv_bytes_per_token().unwrap(), 800 * 1024);
        // The withdrawn figure, named so it cannot quietly return.
        assert_ne!(
            cfg.kv_bytes_per_token().unwrap(),
            546 * 1000,
            "the stale estimate priced a token at 546 kB; the derived cost is 800 KiB"
        );

        let mut q8 = cfg.clone();
        q8.ar.kv_cache_quant = Some(KvCacheQuant::Q8);
        // 5120 packed bytes + 80 groups x (bf16 scale + bf16 bias), x2 for K and V, x40 layers.
        assert_eq!(q8.kv_bytes_per_token().unwrap(), 2 * 40 * (5120 + 80 * 4));
        assert_eq!(q8.kv_bytes_per_token().unwrap(), 435_200);
        // Q8 is 0.53x, NOT the clean 0.5x a "halves the KV" claim would imply — the per-group
        // scale/bias is a real 6% of the tier.
        let ratio =
            q8.kv_bytes_per_token().unwrap() as f64 / cfg.kv_bytes_per_token().unwrap() as f64;
        assert!((ratio - 0.53125).abs() < 1e-9, "Q8 KV is {ratio} of bf16");

        let mut q4 = cfg.clone();
        q4.ar.kv_cache_quant = Some(KvCacheQuant::Q4);
        assert_eq!(q4.kv_bytes_per_token().unwrap(), 230_400);

        // The window figures the docs quote, derived from the same function.
        let shipped_window_tokens = 6 * 1560; // the Mac bounded window at 832x480
        assert_eq!(
            shipped_window_tokens * cfg.kv_bytes_per_token().unwrap(),
            7_667_712_000
        );
        let global_window_tokens = 45 * 1560; // the checkpoint's global window over a 45-frame clip
        let global_gib =
            (global_window_tokens * cfg.kv_bytes_per_token().unwrap()) as f64 / 1024f64.powi(3);
        assert!(
            (global_gib - 53.6).abs() < 0.1,
            "the global window over a 45-latent-frame 832x480 clip is {global_gib:.1} GiB of KV"
        );
    }

    /// A tier MLX cannot express must be rejected by the config, not deep inside a chunk forward —
    /// and the `head_dim` divisibility (the axis grouping actually runs along) must be part of it.
    #[test]
    fn kv_bytes_per_token_rejects_a_tier_the_backbone_cannot_express() {
        let mut cfg = KreaRealtimeConfig::krea_realtime_14b();
        assert_eq!(cfg.wan.head_dim(), 128);
        // group 128 divides head_dim 128 — allowed.
        cfg.ar.kv_cache_quant = Some(KvCacheQuant {
            bits: 8,
            group_size: 128,
        });
        assert!(cfg.kv_bytes_per_token().is_ok());
        // A model whose head_dim is not a multiple of the group is refused, even though `dim` is.
        cfg.wan.num_heads = 80; // head_dim = 5120 / 80 = 64 < 128, while dim stays a multiple of 128
        let err = cfg
            .kv_bytes_per_token()
            .expect_err("head_dim 64 cannot carry a 128-wide group");
        assert!(format!("{err}").contains("group size"), "{err}");
    }

    /// The knob must survive `to_json` → `from_model_dir`, and — the discriminating half — a snapshot
    /// with no `kv_cache_quant` key must stay bf16 rather than inheriting whatever the preset had.
    #[test]
    fn kv_cache_quant_round_trips_and_is_absent_from_a_dense_config() {
        let mut cfg = KreaRealtimeConfig::krea_realtime_14b();
        cfg.ar.kv_cache_quant = Some(KvCacheQuant {
            bits: 4,
            group_size: 32,
        });
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        std::fs::write(
            root.join("config.json"),
            serde_json::to_string_pretty(&cfg.to_json()).unwrap(),
        )
        .unwrap();
        let back = KreaRealtimeConfig::from_model_dir(&root).unwrap();
        assert_eq!(back.ar.kv_cache_quant, cfg.ar.kv_cache_quant);
        assert_eq!(back, cfg);

        // A bf16 config emits no key at all, so a dense snapshot never reads as quantized.
        let dense = KreaRealtimeConfig::krea_realtime_14b();
        assert!(dense.to_json().get("kv_cache_quant").is_none());

        // ...and a JSON that omits the key leaves the shipped bf16 default in place.
        let mut ar = KreaArConfig::krea_realtime_14b();
        overlay_ar(
            &serde_json::from_str("{\"sink_size\": 1}").unwrap(),
            &mut ar,
        )
        .unwrap();
        assert_eq!(ar.kv_cache_quant, None);
        // An explicit null turns a quantized preset back off — the key's presence is what decides.
        let mut ar = KreaArConfig::krea_realtime_14b();
        ar.kv_cache_quant = Some(KvCacheQuant::Q8);
        overlay_ar(
            &serde_json::from_str("{\"kv_cache_quant\": null}").unwrap(),
            &mut ar,
        )
        .unwrap();
        assert_eq!(ar.kv_cache_quant, None);
    }

    /// **A malformed `kv_cache_quant` must be an ERROR, not a quiet fall back to bf16.**
    ///
    /// Every other AR knob degrades silently on a bad value, and that is fine — a non-numeric
    /// `sink_size` keeping the shipped default costs nothing. This one is different: degrading it
    /// silently means the run holds **1.9× the intended KV** while the config, the logs and any
    /// recorded measurement all say `q8`. `CausalKvCache::new` takes an explicit tier parameter to
    /// stop exactly that, and `config.json` is the only surface a product operator can reach the
    /// knob through — so the parse has to be as strict as the constructor.
    ///
    /// Every case here returns `None` under the obvious `and_then(as_i64).zip(...)` form, which is
    /// what makes them discriminating rather than decorative.
    #[test]
    fn a_malformed_kv_cache_quant_is_an_error_not_a_silent_bf16_downgrade() {
        let parse = |json: &str| -> Result<Option<KvCacheQuant>> {
            let mut ar = KreaArConfig::krea_realtime_14b();
            overlay_ar(&serde_json::from_str(json).unwrap(), &mut ar)?;
            Ok(ar.kv_cache_quant)
        };

        // Well formed: accepted.
        assert_eq!(
            parse(r#"{"kv_cache_quant": {"bits": 8, "group_size": 64}}"#).unwrap(),
            Some(KvCacheQuant::Q8)
        );
        // Absent, or an explicit null: the shipped bf16 cache, no error.
        assert_eq!(parse("{}").unwrap(), None);
        assert_eq!(parse(r#"{"kv_cache_quant": null}"#).unwrap(), None);

        // Half-written, mistyped, wrong JSON shape, or a tier MLX cannot express — all errors.
        for bad in [
            r#"{"kv_cache_quant": {"bits": 8}}"#,
            r#"{"kv_cache_quant": {"group_size": 64}}"#,
            r#"{"kv_cache_quant": {"bits": "8", "group_size": 64}}"#,
            r#"{"kv_cache_quant": {"bits": 8.5, "group_size": 64}}"#,
            r#"{"kv_cache_quant": 8}"#,
            r#"{"kv_cache_quant": {}}"#,
            r#"{"kv_cache_quant": {"bits": 7, "group_size": 64}}"#,
            r#"{"kv_cache_quant": {"bits": 8, "group_size": 48}}"#,
        ] {
            let err = parse(bad)
                .err()
                .unwrap_or_else(|| panic!("`{bad}` must not parse to a silent bf16 cache"));
            assert!(matches!(err, Error::Msg(_)), "`{bad}` -> {err:?}");
        }

        // ...and the same strictness reaches `from_model_dir`, which is the surface a snapshot uses.
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        std::fs::write(
            root.join("config.json"),
            r#"{"kv_cache_quant": {"bits": 8}}"#,
        )
        .unwrap();
        assert!(
            KreaRealtimeConfig::from_model_dir(&root).is_err(),
            "a snapshot naming a half-written KV tier must fail to load, not load as bf16"
        );
    }

    #[test]
    fn model_id_is_distinct_from_image_krea() {
        assert_eq!(MODEL_ID, "krea_realtime_14b");
        assert_ne!(MODEL_ID, "krea_2_turbo");
    }

    #[test]
    fn from_model_dir_overlays_present_ar_keys_and_keeps_defaults_for_absent() {
        // A config.json carrying a *subset* of the AR knobs. Keys present in the JSON must overlay the
        // shipped defaults; keys absent from the JSON must retain them.
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        let json = r#"{
            "local_attn_size": 6,
            "sink_size": 1,
            "num_frames_per_block": 4,
            "denoising_step_list": [900, 700, 0],
            "timestep_shift": 3.5
        }"#;
        std::fs::write(root.join("config.json"), json).unwrap();

        let cfg = KreaRealtimeConfig::from_model_dir(&root).unwrap();

        // Present keys are overlaid.
        assert_eq!(cfg.ar.local_attn_size, 6);
        assert_eq!(cfg.ar.sink_size, 1);
        assert_eq!(cfg.ar.num_frames_per_block, 4);
        assert_eq!(cfg.ar.denoising_step_list, vec![900, 700, 0]);
        assert_eq!(cfg.ar.timestep_shift, 3.5);
        // Absent keys fall back to the shipped defaults.
        let def = KreaArConfig::krea_realtime_14b();
        assert_eq!(cfg.ar.kv_cache_num_frames, def.kv_cache_num_frames);
        assert_eq!(cfg.ar.frame_seq_length, def.frame_seq_length);
        assert_eq!(cfg.ar.seq_length, def.seq_length);
        // The Wan half is forced to the dense 2.1 identity regardless of what the JSON implies.
        assert_eq!(cfg.wan.model_version, "2.1");
        assert!(!cfg.wan.dual_model);
    }

    #[test]
    fn overlay_ar_leaves_defaults_when_no_keys_present() {
        // An empty JSON object must not disturb any shipped AR default.
        let v: Value = serde_json::from_str("{}").unwrap();
        let mut ar = KreaArConfig::krea_realtime_14b();
        overlay_ar(&v, &mut ar).unwrap();
        assert_eq!(ar, KreaArConfig::krea_realtime_14b());
    }

    /// `to_json` → `from_model_dir` round-trips the whole config, **including** a pre-quantized tier's
    /// `quantization` block (sc-15203) — the property the tier converter's config emitter relies on.
    /// Discriminating: the AR knobs and the quant manifest are perturbed away from the preset first, so
    /// a `to_json` that dropped either would fail rather than accidentally matching the default.
    #[test]
    fn to_json_round_trips_through_from_model_dir_including_the_quant_tier() {
        let mut cfg = KreaRealtimeConfig::krea_realtime_14b();
        cfg.wan.quantization = Some(mlx_gen_wan::config::WanQuant {
            bits: 4,
            group_size: 64,
        });
        // Perturb the AR half away from the shipped defaults so the round-trip is not a tautology.
        cfg.ar.local_attn_size = 6;
        cfg.ar.sink_size = 2;
        cfg.ar.do_kv_recomp = false;
        cfg.ar.context_noise = 0.25;
        cfg.ar.denoising_step_list = vec![900, 500, 0];

        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        std::fs::write(
            root.join("config.json"),
            serde_json::to_string_pretty(&cfg.to_json()).unwrap(),
        )
        .unwrap();

        let back = KreaRealtimeConfig::from_model_dir(&root).unwrap();
        assert_eq!(back, cfg);
        assert_eq!(
            back.wan.quantization,
            Some(mlx_gen_wan::config::WanQuant {
                bits: 4,
                group_size: 64
            }),
            "the pre-quantized tier manifest must survive the round-trip"
        );

        // A bf16 tier emits no `quantization` block at all (so a dense snapshot never reads as packed).
        let dense = KreaRealtimeConfig::krea_realtime_14b();
        assert!(dense.to_json().get("quantization").is_none());
    }

    #[test]
    fn from_model_dir_without_config_json_is_the_shipped_preset() {
        // No config.json at all → the untouched shipped preset.
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        let cfg = KreaRealtimeConfig::from_model_dir(&root).unwrap();
        assert_eq!(cfg, KreaRealtimeConfig::krea_realtime_14b());
    }
}
