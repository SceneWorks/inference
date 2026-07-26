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
/// Registration under this id is deliberately deferred to S6 (the crate ships no generator yet).
pub const MODEL_ID: &str = "krea_realtime_14b";

/// The autoregressive / self-forcing inference knobs Krea Realtime carries on top of the shared
/// Wan-2.1-T2V-14B DiT. These describe the AR **denoise regime**, not the weights, and are consumed by
/// S3 (causal attention), S4 (KV cache), and S5 (the self-forcing AR loop) — **not** by the S2 load
/// path. Values are the reference defaults from the Krea Realtime 14B release (S1 audit, 2026-07-26).
#[derive(Clone, Debug, PartialEq)]
pub struct KreaArConfig {
    /// Local (sliding-window) causal-attention span in frame-blocks, or `-1` for full causal attention
    /// over the whole generated history (the shipped default). Consumed by S3.
    pub local_attn_size: i64,
    /// Number of always-attended "sink" frame-blocks retained at the start of the KV cache (0 =
    /// none). Consumed by S3/S4.
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
}

impl Default for KreaArConfig {
    fn default() -> Self {
        Self::krea_realtime_14b()
    }
}

impl KreaArConfig {
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
        // The canonical seq_length is 21 latent frames × frame_seq_length.
        assert_eq!(ar.seq_length, 21 * ar.frame_seq_length);
    }

    #[test]
    fn model_id_is_distinct_from_image_krea() {
        assert_eq!(MODEL_ID, "krea_realtime_14b");
        assert_ne!(MODEL_ID, "krea_2_turbo");
    }
}
