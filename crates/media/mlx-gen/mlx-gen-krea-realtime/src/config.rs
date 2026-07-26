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

use std::path::Path;

use mlx_gen::{Error, Result};
use mlx_gen_wan::config::WanModelConfig;
use serde_json::Value;

/// Engine / provider id for Krea Realtime 14B. Distinct from the unrelated **image** crate
/// `mlx-gen-krea` (Krea 2 Turbo, engine `krea_2_turbo`) — this is the autoregressive **video** model.
/// The registered [`Generator`](crate::KreaRealtime) is composed under this id (sc-8439 S6).
pub const MODEL_ID: &str = "krea_realtime_14b";

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
    /// The self-forcing denoising timestep schedule (descending; the terminal `0` is the clean step).
    /// One AR block is denoised through these steps. Consumed by S5.
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
            overlay_ar(&v, &mut cfg.ar);
        }
        Ok(cfg)
    }
}

/// Overlay any AR knobs explicitly present in a `config.json` onto the shipped defaults.
fn overlay_ar(v: &Value, ar: &mut KreaArConfig) {
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

    #[test]
    fn model_id_is_distinct_from_image_krea() {
        assert_eq!(MODEL_ID, "krea_realtime_14b");
        assert_ne!(MODEL_ID, "krea_2_turbo");
    }

    #[test]
    fn from_model_dir_overlays_present_ar_keys_and_keeps_defaults_for_absent() {
        // A config.json carrying a *subset* of the AR knobs. Keys present in the JSON must overlay the
        // shipped defaults; keys absent from the JSON must retain them.
        let root = std::env::temp_dir().join(format!("krea_realtime_cfg_{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
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

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn overlay_ar_leaves_defaults_when_no_keys_present() {
        // An empty JSON object must not disturb any shipped AR default.
        let v: Value = serde_json::from_str("{}").unwrap();
        let mut ar = KreaArConfig::krea_realtime_14b();
        overlay_ar(&v, &mut ar);
        assert_eq!(ar, KreaArConfig::krea_realtime_14b());
    }

    #[test]
    fn from_model_dir_without_config_json_is_the_shipped_preset() {
        // No config.json at all → the untouched shipped preset.
        let root = std::env::temp_dir().join(format!("krea_realtime_nocfg_{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let cfg = KreaRealtimeConfig::from_model_dir(&root).unwrap();
        assert_eq!(cfg, KreaRealtimeConfig::krea_realtime_14b());
        std::fs::remove_dir_all(&root).ok();
    }
}
