//! Qwen3-VL text conditioning — the port of `iris3b/text/qwen3_vl.py` `Qwen3VLTextEncoder`, the
//! Candle twin of `mlx_gen_iris::text_encoder`.
//!
//! * The chat-template prefix, the caption and the suffix are tokenized **separately**
//!   (`add_special_tokens=False`) and concatenated; the caption alone is cut to
//!   `max_length − len(suffix)` so the assistant-turn marker always survives
//!   ([`gen_core::iris::assemble_window`]).
//! * The language tower is the shared generic decoder [`candle_llm::CausalLm`] (the `qwen3_vl`
//!   architecture: GQA, per-head q/k RMSNorm, SwiGLU, θ = 5e6; interleaved mRoPE collapses to 1-D
//!   for a text-only prompt). On a GPU it runs in the dtype the backbone's `text_encoder.dtype`
//!   names (`bfloat16` in the release); on the Candle CPU backend, which has no half-precision
//!   GEMM, in f32 (see [`tower_dtype`]).
//! * The conditioning is the stack of the 1-based post-block hidden states `hidden_layers`
//!   (HF `output_hidden_states` indexing) over the window `[len(prefix), len(prefix) + max_length)`,
//!   with every pad row zeroed — `[1, max_length, len(hidden_layers), dim]` f32.
//!
//! Upstream right-pads the batch with the pad token and runs the padded sequence; with a causal
//! tower and trailing pads the real rows never see a pad and the pad rows are zeroed by the mask,
//! so this port runs only the real tokens and writes zeros — the same conditioning without the pad
//! compute.
//!
//! [`gen_core::iris::assemble_window`]: candle_gen::gen_core::iris::assemble_window

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
pub use candle_gen::gen_core::iris::caption_overflow_warning;
use candle_gen::gen_core::iris::{
    assemble_window, TextEncoderConfig, TextWindow, PROMPT_PREFIX, PROMPT_SUFFIX,
};
use candle_gen::gen_core::safetensors_shards::{
    resolve_indexed_safetensors_shards, snapshot_shard_roots,
};
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::{CandleError as Error, Result};
use candle_llm::models::CausalLm;
use candle_llm::primitives::weights::Weights as LlmWeights;
use candle_llm::ModelConfig;

/// One prompt's conditioning, as `TextEncoding` upstream.
pub struct TextConditioning {
    /// `[1, max_length, layers, dim]` f32, pad rows zero.
    pub states: Tensor,
    /// `[max_length]` 0/1 flags.
    pub mask: Vec<i32>,
    /// Caption tokens dropped by the budget (warned at encode under `on_caption_overflow: warn`).
    pub truncated_tokens: usize,
}

/// The loaded Qwen3-VL language tower + tokenizer + the backbone's conditioning contract.
pub struct IrisTextEncoder {
    lm: CausalLm,
    tokenizer: TextTokenizer,
    prefix_ids: Vec<i32>,
    suffix_ids: Vec<i32>,
    hidden_layers: Vec<usize>,
    max_length: usize,
    dim: usize,
    device: Device,
    /// `text_encoder.on_caption_overflow == "warn"` (the release policy).
    warn_on_overflow: bool,
}

fn from_llm(e: candle_llm::Error) -> Error {
    match e {
        candle_llm::Error::Candle(c) => Error::Candle(c),
        candle_llm::Error::Unsupported(m) => Error::Unsupported(m),
        candle_llm::Error::Canceled => Error::Canceled,
        other => Error::Msg(format!("iris text encoder: {other}")),
    }
}

/// The tower's compute dtype: the backbone's `text_encoder.dtype` on a GPU device; f32 on the
/// Candle CPU backend, which has no half-precision GEMM (the same CPU policy every `candle_llm`
/// decoder follows — [`candle_llm::compute_dtype`]).
pub fn tower_dtype(cfg: &TextEncoderConfig, device: &Device) -> Result<DType> {
    let declared = match cfg.dtype.as_str() {
        "bfloat16" | "bf16" => DType::BF16,
        "float16" | "fp16" | "half" => DType::F16,
        "float32" | "fp32" | "float" => DType::F32,
        other => {
            return Err(Error::Unsupported(format!(
                "iris: text_encoder.dtype = {other} is not a dtype the Qwen3-VL tower can run in"
            )))
        }
    };
    Ok(if device.is_cpu() {
        DType::F32
    } else {
        declared
    })
}

/// The text-encoder shard files of a `Qwen/Qwen3-VL-4B-Instruct` snapshot: the indexed shards when
/// an index is present, else the single `model.safetensors`. Every listed file must exist.
pub fn shard_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let roots = snapshot_shard_roots(dir).map_err(Error::from)?;
    let files: Vec<PathBuf> =
        match resolve_indexed_safetensors_shards(dir, "model.safetensors.index.json", &roots)
            .map_err(Error::from)?
        {
            Some(shards) => shards.into_iter().map(|s| s.loader_path).collect(),
            None => vec![dir.join("model.safetensors")],
        };
    for file in &files {
        if !file.is_file() {
            return Err(Error::Msg(format!(
                "iris: the text encoder resource is incomplete — shard {} is missing",
                file.display()
            )));
        }
    }
    Ok(files)
}

impl IrisTextEncoder {
    /// Load from a `Qwen/Qwen3-VL-4B-Instruct` snapshot directory under the backbone's
    /// `text_encoder` config section, onto `device`. Only the language tower's tensors reach the
    /// device; the vision tower (`model.visual.*`) is dropped on the host, one shard at a time.
    pub fn load(dir: &Path, cfg: &TextEncoderConfig, device: &Device) -> Result<Self> {
        let model_cfg = ModelConfig::from_dir(dir).map_err(from_llm)?;
        if !model_cfg.architecture.is_qwen3_vl() {
            return Err(Error::Unsupported(format!(
                "iris: {} is not a Qwen3-VL checkpoint (the backbone was trained on {})",
                dir.display(),
                cfg.pretrained
            )));
        }
        if model_cfg.hidden_size as usize != cfg.dim {
            return Err(Error::Msg(format!(
                "iris: the text encoder at {} is {} wide, but the backbone's text_encoder.dim is {}",
                dir.display(),
                model_cfg.hidden_size,
                cfg.dim
            )));
        }
        let deepest = *cfg.hidden_layers.last().unwrap_or(&0);
        if deepest > model_cfg.num_layers {
            return Err(Error::Msg(format!(
                "iris: text_encoder.hidden_layers needs layer {deepest}, but the text encoder at {} \
                 has {} layers",
                dir.display(),
                model_cfg.num_layers
            )));
        }
        let dtype = tower_dtype(cfg, device)?;

        let mut tensors = HashMap::new();
        for file in shard_files(dir)? {
            let part =
                candle_gen::candle_core::safetensors::load(&file, &Device::Cpu).map_err(|e| {
                    Error::Msg(format!(
                        "iris: loading text-encoder shard {}: {e}",
                        file.display()
                    ))
                })?;
            for (key, tensor) in part {
                if key.starts_with("model.visual.") {
                    continue;
                }
                tensors.insert(key, tensor.to_device(device)?);
            }
        }
        let weights = LlmWeights::from_map(tensors, device.clone());
        let lm =
            CausalLm::from_weights_dtype(&weights, "", model_cfg, None, dtype).map_err(from_llm)?;
        drop(weights);

        let tokenizer = TextTokenizer::from_file(
            dir.join("tokenizer.json"),
            TokenizerConfig {
                max_length: usize::MAX,
                pad_token_id: 0,
                chat_template: ChatTemplate::None,
                pad_to_max_length: false,
            },
        )
        .map_err(Error::from)?;
        let prefix_ids = tokenizer
            .encode_ids(PROMPT_PREFIX, false)
            .map_err(Error::from)?;
        let suffix_ids = tokenizer
            .encode_ids(PROMPT_SUFFIX, false)
            .map_err(Error::from)?;
        Ok(Self {
            lm,
            tokenizer,
            prefix_ids,
            suffix_ids,
            hidden_layers: cfg.hidden_layers.clone(),
            max_length: cfg.max_length,
            dim: cfg.dim,
            device: device.clone(),
            warn_on_overflow: cfg.on_caption_overflow == "warn",
        })
    }

    pub fn prefix_ids(&self) -> &[i32] {
        &self.prefix_ids
    }

    pub fn suffix_ids(&self) -> &[i32] {
        &self.suffix_ids
    }

    /// The tower's compute dtype.
    pub fn dtype(&self) -> DType {
        self.lm.compute_dtype()
    }

    /// Tokenize and assemble one prompt's window.
    pub fn window(&self, prompt: &str) -> Result<TextWindow> {
        let caption = self
            .tokenizer
            .encode_ids(prompt, false)
            .map_err(Error::from)?;
        assemble_window(
            &self.prefix_ids,
            &caption,
            &self.suffix_ids,
            self.max_length,
        )
        .map_err(Error::from)
    }

    /// `encode([prompt])` (and `null(negative_prompt)`, which is the same computation).
    pub fn encode(&self, prompt: &str) -> Result<TextConditioning> {
        let window = self.window(prompt)?;
        if self.warn_on_overflow {
            if let Some(msg) =
                caption_overflow_warning(&window, self.max_length, self.suffix_ids.len())
            {
                eprintln!("{msg}");
            }
        }
        self.encode_window(&window)
    }

    /// Run the tower over an assembled window.
    pub fn encode_window(&self, window: &TextWindow) -> Result<TextConditioning> {
        let ids: Vec<u32> = window.input_ids.iter().map(|&i| i as u32).collect();
        let n = ids.len();
        let ids = Tensor::from_vec(ids, (1, n), &self.device)?;
        let mut cache = self.lm.new_cache();
        let states = self
            .lm
            .hidden_states(&ids, &mut cache, 0)
            .map_err(from_llm)?;
        let start = window.prefix_len;
        let real = window.window_tokens();
        let picked: Vec<Tensor> = self
            .hidden_layers
            .iter()
            .map(|&k| -> Result<Tensor> {
                let state = states
                    .get(k)
                    .ok_or_else(|| Error::Msg(format!("iris: hidden state {k} not returned")))?;
                Ok(state.narrow(1, start, real)?.to_dtype(DType::F32)?)
            })
            .collect::<Result<_>>()?;
        let stacked = Tensor::stack(&picked, 2)?; // [1, real, layers, dim]
        let pad = self.max_length - real;
        let states = if pad > 0 {
            let zeros = Tensor::zeros(
                (1, pad, self.hidden_layers.len(), self.dim),
                DType::F32,
                &self.device,
            )?;
            Tensor::cat(&[&stacked, &zeros], 1)?
        } else {
            stacked
        };
        Ok(TextConditioning {
            states,
            mask: window.mask.clone(),
            truncated_tokens: window.truncated_tokens,
        })
    }
}
