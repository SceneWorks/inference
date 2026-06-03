//! LTX-2.3 model configuration — **config-driven** from the model's `embedded_config.json`,
//! mirroring the reference `generate_av.py` build logic (lines 1464–1529 of the
//! `mlx-video-with-audio` package).
//!
//! The shipped SceneWorks model (`ltx_2_3_eros`) is an `AVTransformer3DModel`: gated attention,
//! adaLN coefficient 9, **no** PixArt caption-projection linears (so `caption_channels` is the
//! connector output `connector_heads × connector_head_dim = 4096`, not the 2.0 default 3840), and
//! an 8-layer learnable-register connector. The 2.0 `generate.py` path hardcodes a different
//! (non-gated, coeff-6, caption-proj-true, 3840) config and cannot run against this checkpoint —
//! hence "read `embedded_config.json`, don't hardcode 2.0 values" (sc-2679 S0).
//!
//! This core is **VideoOnly**: only the video-stack transformer fields are consumed by the
//! denoise path. The audio + connector fields are read here (so the reader is complete and the
//! sibling slices reuse it) but are inert for T2V.

use std::path::Path;

use serde_json::Value;

use mlx_gen::{Error, Result};

/// Rotary-embedding layout. LTX-2.3 uses [`RopeType::Split`] (the 2.0 default is interleaved).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeType {
    Interleaved,
    Split,
}

impl RopeType {
    fn from_str(s: &str) -> RopeType {
        match s {
            "split" => RopeType::Split,
            _ => RopeType::Interleaved,
        }
    }
}

/// The full LTX transformer config. Dimension-parametric: every field is read from
/// `embedded_config.json` where present, falling back to the reference's hardcoded defaults.
#[derive(Clone, Debug)]
pub struct LtxConfig {
    // --- Video transformer ---
    pub num_attention_heads: i32,
    pub attention_head_dim: i32,
    pub in_channels: i32,
    pub out_channels: i32,
    pub num_layers: i32,
    pub cross_attention_dim: i32,
    /// Input dim of the text features entering cross-attention. When both caption-projection
    /// linears are absent (LTX-2.3) this equals the connector output `conn_heads × conn_head_dim`.
    pub caption_channels: i32,
    pub caption_projection_first_linear: bool,
    pub caption_projection_second_linear: bool,
    /// adaLN-single scale_shift_table row count: **9** for the gated family, **6** otherwise.
    pub adaln_embedding_coefficient: i32,
    pub apply_gated_attention: bool,
    /// `cross_attention_adaln=true` for 2.3 (the per-block `scale_shift_table` carries the extra
    /// rows 6..9 used by the text cross-attention; see transformer.py `v_has_ca_ada`).
    pub cross_attention_adaln: bool,
    pub norm_eps: f64,

    // --- Positional / RoPE ---
    pub positional_embedding_theta: f64,
    pub positional_embedding_max_pos: [i32; 3],
    pub use_middle_indices_grid: bool,
    pub rope_type: RopeType,
    pub double_precision_rope: bool,
    pub timestep_scale_multiplier: i32,

    // --- Connector (S1: Embeddings1DConnector) — read here, consumed by the TE slice ---
    pub use_embeddings_connector: bool,
    pub connector_num_layers: i32,
    pub connector_num_attention_heads: i32,
    pub connector_attention_head_dim: i32,
    pub connector_num_learnable_registers: i32,
    pub connector_positional_embedding_max_pos: i32,
    pub connector_apply_gated_attention: bool,

    // --- Audio stack (sibling sc-2684) — read here, inert for VideoOnly ---
    pub audio_num_attention_heads: i32,
    pub audio_attention_head_dim: i32,
    pub audio_cross_attention_dim: i32,
    pub audio_caption_channels: i32,
}

impl LtxConfig {
    /// Video inner dimension `heads × head_dim` (4096 for LTX-2.3).
    pub fn inner_dim(&self) -> i32 {
        self.num_attention_heads * self.attention_head_dim
    }

    /// The reference 2.0 `generate.py` defaults (non-gated, coeff-6, caption-proj present,
    /// caption_channels 3840, interleaved-default overridden to split by `generate.py`). Used as
    /// the fallback when no `embedded_config.json` is present; LTX-2.3 overrides most of these.
    pub fn video_only_defaults() -> Self {
        LtxConfig {
            num_attention_heads: 32,
            attention_head_dim: 128,
            in_channels: 128,
            out_channels: 128,
            num_layers: 48,
            cross_attention_dim: 4096,
            caption_channels: 3840,
            caption_projection_first_linear: true,
            caption_projection_second_linear: true,
            adaln_embedding_coefficient: 6,
            apply_gated_attention: false,
            cross_attention_adaln: false,
            norm_eps: 1e-6,
            positional_embedding_theta: 10000.0,
            positional_embedding_max_pos: [20, 2048, 2048],
            use_middle_indices_grid: true,
            rope_type: RopeType::Split,
            double_precision_rope: true,
            timestep_scale_multiplier: 1000,
            use_embeddings_connector: false,
            connector_num_layers: 8,
            connector_num_attention_heads: 32,
            connector_attention_head_dim: 128,
            connector_num_learnable_registers: 128,
            connector_positional_embedding_max_pos: 4096,
            connector_apply_gated_attention: false,
            audio_num_attention_heads: 32,
            audio_attention_head_dim: 64,
            audio_cross_attention_dim: 2048,
            audio_caption_channels: 3840,
        }
    }

    /// Build the config from the `transformer` block of a parsed `embedded_config.json`,
    /// reproducing `generate_av.py`'s field resolution exactly.
    pub fn from_embedded_transformer(t: &Value) -> Self {
        let mut cfg = Self::video_only_defaults();

        // Plain dimension-parametric reads (default = the reference's hardcoded value).
        cfg.num_attention_heads = get_i32(t, "num_attention_heads", cfg.num_attention_heads);
        cfg.attention_head_dim = get_i32(t, "attention_head_dim", cfg.attention_head_dim);
        cfg.in_channels = get_i32(t, "in_channels", cfg.in_channels);
        cfg.out_channels = get_i32(t, "out_channels", cfg.out_channels);
        cfg.num_layers = get_i32(t, "num_layers", cfg.num_layers);
        cfg.cross_attention_dim = get_i32(t, "cross_attention_dim", cfg.cross_attention_dim);
        cfg.norm_eps = get_f64(t, "norm_eps", cfg.norm_eps);
        cfg.positional_embedding_theta = get_f64(
            t,
            "positional_embedding_theta",
            cfg.positional_embedding_theta,
        );
        cfg.positional_embedding_max_pos = get_i32_3(
            t,
            "positional_embedding_max_pos",
            cfg.positional_embedding_max_pos,
        );
        cfg.use_middle_indices_grid =
            get_bool(t, "use_middle_indices_grid", cfg.use_middle_indices_grid);
        if let Some(s) = t.get("rope_type").and_then(Value::as_str) {
            cfg.rope_type = RopeType::from_str(s);
        }
        // `frequencies_precision: "float64"` ⇒ double-precision RoPE (generate_av hardcodes true).
        cfg.double_precision_rope = t
            .get("frequencies_precision")
            .and_then(Value::as_str)
            .map(|s| s == "float64")
            .unwrap_or(true);
        cfg.timestep_scale_multiplier = get_i32(
            t,
            "timestep_scale_multiplier",
            cfg.timestep_scale_multiplier,
        );

        // Caption-projection / gated-attention resolution (generate_av.py lines 1480–1498).
        cfg.caption_projection_first_linear = get_bool(t, "caption_projection_first_linear", true);
        cfg.caption_projection_second_linear =
            get_bool(t, "caption_projection_second_linear", true);
        cfg.apply_gated_attention = get_bool(t, "apply_gated_attention", false);
        cfg.adaln_embedding_coefficient = if cfg.apply_gated_attention { 9 } else { 6 };
        cfg.cross_attention_adaln = get_bool(t, "cross_attention_adaln", cfg.apply_gated_attention);

        // Connector dims (used to derive caption_channels when caption-proj is absent).
        cfg.use_embeddings_connector = get_bool(t, "use_embeddings_connector", false);
        cfg.connector_num_layers = get_i32(t, "connector_num_layers", cfg.connector_num_layers);
        cfg.connector_num_attention_heads = get_i32(
            t,
            "connector_num_attention_heads",
            cfg.connector_num_attention_heads,
        );
        cfg.connector_attention_head_dim = get_i32(
            t,
            "connector_attention_head_dim",
            cfg.connector_attention_head_dim,
        );
        cfg.connector_num_learnable_registers = get_i32(
            t,
            "connector_num_learnable_registers",
            cfg.connector_num_learnable_registers,
        );
        cfg.connector_positional_embedding_max_pos = t
            .get("connector_positional_embedding_max_pos")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(Value::as_i64)
            .map(|n| n as i32)
            .unwrap_or(cfg.connector_positional_embedding_max_pos);
        cfg.connector_apply_gated_attention = get_bool(
            t,
            "connector_apply_gated_attention",
            cfg.apply_gated_attention,
        );

        // Audio connector dims (for the audio-caption derivation, mirrored below).
        cfg.audio_num_attention_heads = get_i32(
            t,
            "audio_num_attention_heads",
            cfg.audio_num_attention_heads,
        );
        cfg.audio_attention_head_dim =
            get_i32(t, "audio_attention_head_dim", cfg.audio_attention_head_dim);
        cfg.audio_cross_attention_dim = get_i32(
            t,
            "audio_cross_attention_dim",
            cfg.audio_cross_attention_dim,
        );

        // caption_channels derivation (generate_av.py lines 1484–1498).
        let no_caption_proj =
            !cfg.caption_projection_first_linear && !cfg.caption_projection_second_linear;
        if no_caption_proj {
            cfg.caption_channels =
                cfg.connector_num_attention_heads * cfg.connector_attention_head_dim;
            let audio_conn_heads = get_i32(t, "audio_connector_num_attention_heads", 32);
            let audio_conn_head_dim = get_i32(t, "audio_connector_attention_head_dim", 64);
            cfg.audio_caption_channels = audio_conn_heads * audio_conn_head_dim;
        } else {
            cfg.caption_channels = get_i32(t, "caption_channels", cfg.caption_channels);
            cfg.audio_caption_channels =
                get_i32(t, "audio_caption_channels", cfg.audio_caption_channels);
        }

        cfg
    }

    /// Load the config from a model directory's `embedded_config.json` (the `transformer` block).
    /// Falls back to [`video_only_defaults`](Self::video_only_defaults) if the file is absent.
    pub fn from_model_dir(root: &Path) -> Result<Self> {
        let path = root.join("embedded_config.json");
        if !path.exists() {
            return Ok(Self::video_only_defaults());
        }
        let text = std::fs::read_to_string(&path)?;
        let root_cfg: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Msg(format!("ltx: parse embedded_config.json: {e}")))?;
        let t = root_cfg
            .get("transformer")
            .ok_or_else(|| Error::Msg("ltx: embedded_config.json missing `transformer`".into()))?;
        Ok(Self::from_embedded_transformer(t))
    }
}

fn get_i32(v: &Value, key: &str, default: i32) -> i32 {
    v.get(key)
        .and_then(Value::as_i64)
        .map(|n| n as i32)
        .unwrap_or(default)
}

fn get_f64(v: &Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap_or(default)
}

fn get_bool(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn get_i32_3(v: &Value, key: &str, default: [i32; 3]) -> [i32; 3] {
    match v.get(key).and_then(Value::as_array) {
        Some(a) if a.len() == 3 => {
            let mut out = default;
            for (i, slot) in out.iter_mut().enumerate() {
                if let Some(n) = a[i].as_i64() {
                    *slot = n as i32;
                }
            }
            out
        }
        _ => default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact `transformer` block of `ltx_2_3_eros/embedded_config.json`.
    fn eros_transformer() -> Value {
        serde_json::json!({
            "_class_name": "AVTransformer3DModel",
            "attention_head_dim": 128,
            "caption_channels": 3840,
            "cross_attention_dim": 4096,
            "in_channels": 128,
            "norm_eps": 1e-06,
            "num_attention_heads": 32,
            "num_layers": 48,
            "out_channels": 128,
            "audio_num_attention_heads": 32,
            "audio_attention_head_dim": 64,
            "audio_cross_attention_dim": 2048,
            "use_embeddings_connector": true,
            "connector_attention_head_dim": 128,
            "connector_num_attention_heads": 32,
            "connector_num_layers": 8,
            "connector_positional_embedding_max_pos": [4096],
            "connector_num_learnable_registers": 128,
            "use_middle_indices_grid": true,
            "apply_gated_attention": true,
            "connector_apply_gated_attention": true,
            "caption_projection_first_linear": false,
            "caption_projection_second_linear": false,
            "audio_connector_attention_head_dim": 64,
            "audio_connector_num_attention_heads": 32,
            "cross_attention_adaln": true,
            "text_encoder_norm_type": "per_token_rms",
            "rope_type": "split",
            "frequencies_precision": "float64",
            "positional_embedding_theta": 10000.0,
            "positional_embedding_max_pos": [20, 2048, 2048],
            "timestep_scale_multiplier": 1000,
            "av_ca_timestep_scale_multiplier": 1000.0
        })
    }

    #[test]
    fn eros_config_matches_reference_build_logic() {
        let cfg = LtxConfig::from_embedded_transformer(&eros_transformer());
        // Gated family → adaLN coeff 9.
        assert!(cfg.apply_gated_attention);
        assert_eq!(cfg.adaln_embedding_coefficient, 9);
        assert!(cfg.cross_attention_adaln);
        // No caption projection → caption_channels = connector_heads × connector_head_dim = 4096.
        assert!(!cfg.caption_projection_first_linear);
        assert!(!cfg.caption_projection_second_linear);
        assert_eq!(cfg.caption_channels, 4096);
        assert_eq!(cfg.audio_caption_channels, 32 * 64);
        // Core dims.
        assert_eq!(cfg.inner_dim(), 4096);
        assert_eq!(cfg.num_layers, 48);
        assert_eq!(cfg.cross_attention_dim, 4096);
        assert_eq!(cfg.rope_type, RopeType::Split);
        assert!(cfg.double_precision_rope);
        assert_eq!(cfg.positional_embedding_max_pos, [20, 2048, 2048]);
        assert!(cfg.use_middle_indices_grid);
        assert_eq!(cfg.timestep_scale_multiplier, 1000);
        // Connector.
        assert!(cfg.use_embeddings_connector);
        assert_eq!(cfg.connector_num_layers, 8);
        assert_eq!(cfg.connector_num_attention_heads, 32);
        assert_eq!(cfg.connector_attention_head_dim, 128);
        assert_eq!(cfg.connector_num_learnable_registers, 128);
        assert_eq!(cfg.connector_positional_embedding_max_pos, 4096);
        assert!(cfg.connector_apply_gated_attention);
    }

    #[test]
    fn defaults_are_the_2_0_values() {
        let cfg = LtxConfig::video_only_defaults();
        assert_eq!(cfg.adaln_embedding_coefficient, 6);
        assert!(!cfg.apply_gated_attention);
        assert_eq!(cfg.caption_channels, 3840);
        assert!(cfg.caption_projection_first_linear);
    }
}
