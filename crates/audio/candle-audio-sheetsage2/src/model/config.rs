//! `config.json` of `m-a-p/SheetSage2` (`SheetSage2Config`, with its `backbone_config`) and of
//! `m-a-p/MERT-v2-FullSong` (`MERT2Config`), validated the way upstream validates them.

use serde_json::Value;

use crate::Error;

fn num(v: &Value, key: &str) -> Result<f64, Error> {
    v.get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| Error::Config(format!("config field `{key}` is missing or not a number")))
}

fn int(v: &Value, key: &str) -> Result<usize, Error> {
    let x = num(v, key)?;
    if x < 0.0 || x.fract() != 0.0 {
        return Err(Error::Config(format!(
            "config field `{key}` must be a non-negative integer"
        )));
    }
    Ok(x as usize)
}

fn text(v: &Value, key: &str) -> Result<String, Error> {
    v.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| Error::Config(format!("config field `{key}` is missing or not a string")))
}

fn triple(v: &Value, key: &str) -> Result<[usize; 3], Error> {
    let items = v
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Config(format!("config field `{key}` is missing or not a list")))?;
    if items.len() != 3 {
        return Err(Error::Config(format!(
            "config field `{key}` must have three entries"
        )));
    }
    let mut out = [0usize; 3];
    for (slot, item) in out.iter_mut().zip(items) {
        *slot = item
            .as_u64()
            .ok_or_else(|| Error::Config(format!("config field `{key}` must hold integers")))?
            as usize;
    }
    Ok(out)
}

/// The MERT2 encoder architecture (`MERT2Config`).
#[derive(Clone, Debug, PartialEq)]
pub struct MertConfig {
    /// Model width.
    pub hidden_size: usize,
    /// Macaron FFN width.
    pub intermediate_size: usize,
    /// Conformer blocks.
    pub num_hidden_layers: usize,
    /// Attention heads.
    pub num_attention_heads: usize,
    /// Mel bins.
    pub num_mel_bins: usize,
    /// Input sample rate.
    pub sampling_rate: usize,
    /// FFT size.
    pub n_fft: usize,
    /// Analysis window length.
    pub win_length: usize,
    /// STFT hop.
    pub hop_length: usize,
    /// Subsampler stage widths.
    pub subsampling_channels: [usize; 3],
    /// ConvNeXt layers per stage.
    pub subsampling_depths: [usize; 3],
    /// Depthwise kernel of the Conformer convolution module.
    pub conv_depthwise_kernel_size: usize,
    /// RoPE base.
    pub rotary_embedding_base: f64,
    /// Conformer LayerNorm epsilon.
    pub layer_norm_eps: f64,
    /// Subsampler LayerNorm epsilon.
    pub subsampling_layer_norm_eps: f64,
    /// `30s` or `fs`.
    pub variant: String,
}

impl MertConfig {
    /// Parse and validate (upstream `MERT2Config.__init__` checks).
    pub fn from_json(v: &Value) -> Result<Self, Error> {
        let config = Self {
            hidden_size: int(v, "hidden_size")?,
            intermediate_size: int(v, "intermediate_size")?,
            num_hidden_layers: int(v, "num_hidden_layers")?,
            num_attention_heads: int(v, "num_attention_heads")?,
            num_mel_bins: int(v, "num_mel_bins")?,
            sampling_rate: int(v, "sampling_rate")?,
            n_fft: int(v, "n_fft")?,
            win_length: int(v, "win_length")?,
            hop_length: int(v, "hop_length")?,
            subsampling_channels: triple(v, "subsampling_channels")?,
            subsampling_depths: triple(v, "subsampling_depths")?,
            conv_depthwise_kernel_size: int(v, "conv_depthwise_kernel_size")?,
            rotary_embedding_base: num(v, "rotary_embedding_base")?,
            layer_norm_eps: num(v, "layer_norm_eps")?,
            subsampling_layer_norm_eps: num(v, "subsampling_layer_norm_eps")?,
            variant: text(v, "variant")?,
        };
        let bad = |m: &str| Err(Error::Config(m.to_string()));
        if config.hidden_size == 0
            || config.intermediate_size == 0
            || config.num_hidden_layers == 0
            || config.num_attention_heads == 0
        {
            return bad("encoder dimensions and layer counts must be positive");
        }
        if config.hidden_size % config.num_attention_heads != 0
            || (config.hidden_size / config.num_attention_heads) % 2 != 0
        {
            return bad("hidden_size must divide into an even head dimension");
        }
        if config.subsampling_channels[0] != config.num_mel_bins
            || config.subsampling_channels[2] != config.hidden_size
        {
            return bad("subsampling widths must start at num_mel_bins and end at hidden_size");
        }
        if config.n_fft < config.win_length || config.win_length == 0 || config.n_fft % 2 != 0 {
            return bad("n_fft must be even and at least win_length > 0");
        }
        if config.conv_depthwise_kernel_size % 2 != 1 {
            return bad("conv_depthwise_kernel_size must be positive and odd");
        }
        if !matches!(config.variant.as_str(), "30s" | "fs") {
            return bad("variant must be '30s' or 'fs'");
        }
        Ok(config)
    }

    /// Samples per output frame (`hop_length * 4`).
    pub fn inputs_to_logits_ratio(&self) -> usize {
        self.hop_length * 4
    }

    /// Smallest accepted waveform (`n_fft / 2 + 1`).
    pub fn minimum_input_samples(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Head width.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// The SheetSage2 head and decoder (`SheetSage2Config`).
#[derive(Clone, Debug, PartialEq)]
pub struct SheetSage2Config {
    /// Vocabulary size (must equal the computed tokenizer's).
    pub vocab_size: usize,
    /// Decoder width.
    pub hidden_size: usize,
    /// Decoder layers.
    pub decoder_layers: usize,
    /// Decoder heads.
    pub num_attention_heads: usize,
    /// Decoder FFN width.
    pub intermediate_size: usize,
    /// Fixed window length, seconds.
    pub input_audio_length: f64,
    /// Decoder positional capacity.
    pub max_output_seq_len: usize,
    /// Time tokens per second.
    pub time_hz: u32,
    /// Sample rate.
    pub sampling_rate: usize,
    /// LoRA rank.
    pub lora_rank: usize,
    /// LoRA alpha.
    pub lora_alpha: f64,
    /// `adapter` (the only format the pinned checkpoint uses and this port loads).
    pub weights_format: String,
    /// The pinned MERT parent revision.
    pub base_model_revision: String,
    /// The pinned MERT parent `model.safetensors` SHA-256.
    pub base_model_sha256: String,
    /// The pinned vocabulary fingerprint.
    pub tokenizer_fingerprint: String,
    /// The encoder architecture the head was trained on.
    pub backbone: MertConfig,
}

impl SheetSage2Config {
    /// Parse and validate (upstream `SheetSage2Config.__init__` checks, plus the port's own: schema
    /// `v1` and the adapter format only).
    pub fn from_json(v: &Value) -> Result<Self, Error> {
        let backbone = MertConfig::from_json(
            v.get("backbone_config")
                .ok_or_else(|| Error::Config("config has no backbone_config".into()))?,
        )?;
        let config = Self {
            vocab_size: int(v, "vocab_size")?,
            hidden_size: int(v, "hidden_size")?,
            decoder_layers: int(v, "decoder_layers")?,
            num_attention_heads: int(v, "num_attention_heads")?,
            intermediate_size: int(v, "intermediate_size")?,
            input_audio_length: num(v, "input_audio_length")?,
            max_output_seq_len: int(v, "max_output_seq_len")?,
            time_hz: int(v, "time_hz")? as u32,
            sampling_rate: int(v, "sampling_rate")?,
            lora_rank: int(v, "lora_rank")?,
            lora_alpha: num(v, "lora_alpha")?,
            weights_format: text(v, "weights_format")?,
            base_model_revision: text(v, "base_model_revision")?,
            base_model_sha256: text(v, "base_model_sha256")?,
            tokenizer_fingerprint: text(v, "tokenizer_fingerprint")?,
            backbone,
        };
        if text(v, "tokenizer_schema_version")? != "v1" {
            return Err(Error::Config("only tokenizer schema v1 is ported".into()));
        }
        if config.weights_format != "adapter" {
            return Err(Error::Config(format!(
                "weights_format {:?}: only the pinned adapter layout (MERT parent + LoRA \
                 adapters, merged in fp32 at load) is supported",
                config.weights_format
            )));
        }
        if config.hidden_size % config.num_attention_heads != 0 {
            return Err(Error::Config(
                "hidden_size must be divisible by num_attention_heads".into(),
            ));
        }
        if config.sampling_rate != config.backbone.sampling_rate {
            return Err(Error::Config(
                "processor and encoder sampling rates must match".into(),
            ));
        }
        if config.lora_rank == 0 || !(config.lora_alpha > 0.0) {
            return Err(Error::Config("lora rank and alpha must be positive".into()));
        }
        Ok(config)
    }

    /// Samples in one fixed window.
    pub fn window_samples(&self) -> usize {
        (self.input_audio_length * self.sampling_rate as f64).round() as usize
    }
}
