//! Chatterbox hyperparameters (sc-13222), transcribed exactly from the upstream Python source
//! (`resemble-ai/chatterbox` @ the pinned revision) so the native candle port matches the
//! reference numerically.
//!
//! The two model stacks each carry their own config block:
//!
//! - [`T3Config`] — the text→speech-token LM (a Llama-520M backbone with custom embeddings and
//!   heads). Mirrors `models/t3/modules/t3_config.py` + `models/t3/llama_configs.py`
//!   (`LLAMA_520M_CONFIG_DICT`).
//! - [`S3GenConfig`] — the speech-token→waveform stack (s3tokenizer + CAMPPlus x-vector + the
//!   CosyVoice-derived flow-matching decoder + the HiFTNet vocoder). Mirrors
//!   `models/s3gen/configs.py` + the call-site kwargs in `models/s3gen/s3gen.py`.
//!
//! These are compile-time constants rather than parsed JSON: the pinned Chatterbox snapshot ships
//! no per-model `config.json` for either stack (the reference hard-codes them in Python), so the
//! honest port hard-codes the same values here, pinned by the revision the weights are pinned to.

/// Reference-audio input rate for the s3tokenizer and the voice/speaker encoders (Hz).
pub const S3_SR: u32 = 16_000;

/// Synthesis output rate — the S3Gen mel extractor and HiFTNet vocoder rate (Hz).
pub const S3GEN_SR: u32 = 24_000;

/// S3 speech-token rate (Hz): 25 tokens/s (`S3_TOKEN_HOP = 640` samples at 16 kHz).
pub const S3_TOKEN_RATE: u32 = 25;

/// Speech-token→mel upsample ratio (`token_mel_ratio`): 25 Hz tokens → 50 Hz mel frames.
pub const TOKEN_MEL_RATIO: usize = 2;

/// Valid S3 speech-token codebook size (`SPEECH_VOCAB_SIZE`): ids `0..6560` are real speech
/// codes; ids `>= 6561` are T3's special/BOS/EOS speech tokens, dropped before S3Gen.
pub const SPEECH_VOCAB_SIZE: usize = 6561;

/// Cap on the speech-token *prompt* (encoder conditioning) reference: 6 s at 16 kHz.
pub const ENC_COND_LEN: usize = 6 * S3_SR as usize;

/// Cap on the S3Gen mel/x-vector *decoder* reference: 10 s at 24 kHz.
pub const DEC_COND_LEN: usize = 10 * S3GEN_SR as usize;

/// The T3 conditioning prompt length in speech tokens (`speech_cond_prompt_len`).
pub const SPEECH_COND_PROMPT_LEN: usize = 150;

/// The Llama-3 RoPE scaling block used by the T3 backbone (`rope_scaling` in
/// `LLAMA_520M_CONFIG_DICT`) — the standard Llama-3 long-context frequency remap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Llama3RopeScaling {
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_max_position_embeddings: usize,
}

/// The T3 LM configuration — a Llama-520M backbone with Chatterbox's custom text/speech
/// embeddings, learned positional embeddings, conditioning encoder, and speech head.
///
/// Field values are exactly `LLAMA_520M_CONFIG_DICT` (backbone) and `T3Config` (I/O surface)
/// from the upstream source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct T3Config {
    // --- Llama-520M backbone (`LLAMA_520M_CONFIG_DICT`) ---
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: Llama3RopeScaling,
    pub max_position_embeddings: usize,

    // --- Chatterbox T3 I/O surface (`T3Config`) ---
    /// Text token vocab (base English model; the multilingual model uses 2454).
    pub text_tokens_dict_size: usize,
    /// Speech token vocab (includes the special/BOS/EOS speech tokens above 6561).
    pub speech_tokens_dict_size: usize,
    pub start_text_token: u32,
    pub stop_text_token: u32,
    pub start_speech_token: u32,
    pub stop_speech_token: u32,
    pub max_text_tokens: usize,
    pub max_speech_tokens: usize,
    /// Length of the speech-token conditioning prompt fed through the Perceiver resampler.
    pub speech_cond_prompt_len: usize,
    /// Speaker-embedding width consumed by the conditioning encoder (the `chatterbox_ve` vector).
    pub speaker_embed_size: usize,
    /// Whether the conditioning prompt tokens pass through a Perceiver resampler
    /// (`use_perceiver_resampler = True`).
    pub use_perceiver_resampler: bool,
    /// Whether an emotion-advisor scalar is part of the conditioning prefix (`emotion_adv = True`).
    pub emotion_adv: bool,
}

impl Default for T3Config {
    fn default() -> Self {
        Self::LLAMA_520M
    }
}

impl T3Config {
    /// The base English Chatterbox T3 configuration.
    pub const LLAMA_520M: Self = Self {
        hidden_size: 1024,
        intermediate_size: 4096,
        num_hidden_layers: 30,
        num_attention_heads: 16,
        num_key_value_heads: 16,
        head_dim: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 500_000.0,
        rope_scaling: Llama3RopeScaling {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        },
        max_position_embeddings: 131_072,
        text_tokens_dict_size: 704,
        speech_tokens_dict_size: 8194,
        start_text_token: 255,
        stop_text_token: 0,
        start_speech_token: 6561,
        stop_speech_token: 6562,
        max_text_tokens: 2048,
        max_speech_tokens: 4096,
        speech_cond_prompt_len: SPEECH_COND_PROMPT_LEN,
        speaker_embed_size: 256,
        use_perceiver_resampler: true,
        emotion_adv: true,
    };

    /// Learned text positional-embedding table length (`max_text_tokens + 2`).
    pub const fn text_pos_len(&self) -> usize {
        self.max_text_tokens + 2
    }

    /// Learned speech positional-embedding table length (`max_speech_tokens + 2 + 2`).
    pub const fn speech_pos_len(&self) -> usize {
        self.max_speech_tokens + 4
    }
}

/// The default sampling knobs of `TTS.generate` (reference call defaults).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationDefaults {
    pub exaggeration: f32,
    pub cfg_weight: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub min_p: f32,
    pub top_p: f32,
}

impl Default for GenerationDefaults {
    fn default() -> Self {
        Self {
            exaggeration: 0.5,
            cfg_weight: 0.5,
            temperature: 0.8,
            repetition_penalty: 1.2,
            min_p: 0.05,
            top_p: 1.0,
        }
    }
}

/// The S3Gen speech-token→waveform configuration (s3tokenizer + CAMPPlus + flow + HiFTNet).
///
/// Values are transcribed from `models/s3gen/configs.py`, the `CFM_PARAMS` block, and the
/// `HiFTGenerator` / `CausalMaskedDiffWithXvec` call-site kwargs in `models/s3gen/s3gen.py`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct S3GenConfig {
    // --- mel extractor (`utils/mel.py`) ---
    pub mel_n_fft: usize,
    pub mel_num_mels: usize,
    pub mel_hop: usize,
    pub mel_win: usize,
    pub mel_fmin: f32,
    pub mel_fmax: f32,

    // --- flow (`CausalMaskedDiffWithXvec`) ---
    /// Token embedding width feeding the conformer encoder (`input_size`).
    pub flow_input_size: usize,
    /// Conformer encoder output width (`output_size`).
    pub flow_encoder_dim: usize,
    /// Output mel width (`out_channels`, 80).
    pub mel_dim: usize,
    /// CAMPPlus x-vector width before the `spk_embed_affine_layer` (192 → 80).
    pub xvector_dim: usize,

    // --- CFM (`CFM_PARAMS`) ---
    pub cfm_sigma_min: f64,
    pub cfm_inference_cfg_rate: f64,
    /// Default flow-matching solver steps (`n_cfm_timesteps`).
    pub cfm_steps: usize,

    // --- HiFTNet vocoder (`HiFTGenerator`) ---
    pub hift_sampling_rate: u32,
    pub hift_upsample_rates: [usize; 3],
    pub hift_upsample_kernel_sizes: [usize; 3],
    pub hift_istft_n_fft: usize,
    pub hift_istft_hop: usize,
}

impl Default for S3GenConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl S3GenConfig {
    /// The shipped Chatterbox S3Gen configuration.
    pub const DEFAULT: Self = Self {
        mel_n_fft: 1920,
        mel_num_mels: 80,
        mel_hop: 480,
        mel_win: 1920,
        mel_fmin: 0.0,
        mel_fmax: 8000.0,
        flow_input_size: 512,
        flow_encoder_dim: 512,
        mel_dim: 80,
        xvector_dim: 192,
        cfm_sigma_min: 1e-6,
        cfm_inference_cfg_rate: 0.7,
        cfm_steps: 10,
        hift_sampling_rate: S3GEN_SR,
        hift_upsample_rates: [8, 5, 3],
        hift_upsample_kernel_sizes: [16, 11, 7],
        hift_istft_n_fft: 16,
        hift_istft_hop: 4,
    };

    /// Waveform samples produced per mel frame: `prod(upsample_rates) · istft_hop`.
    pub const fn samples_per_mel_frame(&self) -> usize {
        self.hift_upsample_rates[0]
            * self.hift_upsample_rates[1]
            * self.hift_upsample_rates[2]
            * self.hift_istft_hop
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t3_backbone_matches_llama_520m() {
        let c = T3Config::LLAMA_520M;
        assert_eq!(c.hidden_size, 1024);
        assert_eq!(c.num_hidden_layers, 30);
        assert_eq!(c.num_attention_heads, 16);
        assert_eq!(c.num_key_value_heads, 16); // no GQA
        assert_eq!(c.head_dim, 64);
        assert_eq!(c.num_attention_heads * c.head_dim, c.hidden_size);
        assert_eq!(c.rope_theta, 500_000.0);
        assert_eq!(c.rope_scaling.factor, 8.0);
    }

    #[test]
    fn t3_io_surface_matches_reference() {
        let c = T3Config::LLAMA_520M;
        assert_eq!(c.text_tokens_dict_size, 704);
        assert_eq!(c.speech_tokens_dict_size, 8194);
        assert_eq!(c.start_text_token, 255);
        assert_eq!(c.stop_text_token, 0);
        assert_eq!(c.start_speech_token, 6561);
        assert_eq!(c.stop_speech_token, 6562);
        // Learned positional-embedding table lengths.
        assert_eq!(c.text_pos_len(), 2050);
        assert_eq!(c.speech_pos_len(), 4100);
    }

    #[test]
    fn s3gen_vocoder_frame_math_is_consistent() {
        let s = S3GenConfig::DEFAULT;
        // 8·5·3·4 = 480 samples per mel frame → 24000/480 = 50 Hz mel frame rate,
        // which is TOKEN_MEL_RATIO × the 25 Hz token rate.
        assert_eq!(s.samples_per_mel_frame(), 480);
        assert_eq!(
            s.hift_sampling_rate as usize / s.samples_per_mel_frame(),
            S3_TOKEN_RATE as usize * TOKEN_MEL_RATIO
        );
        assert_eq!(s.mel_hop, s.samples_per_mel_frame());
    }

    #[test]
    fn generation_defaults_match_reference() {
        let d = GenerationDefaults::default();
        assert_eq!(d.cfg_weight, 0.5);
        assert_eq!(d.temperature, 0.8);
        assert_eq!(d.repetition_penalty, 1.2);
        assert_eq!(d.min_p, 0.05);
    }
}
