//! Qwen3-VL-4B text conditioning — the port of upstream `src/flux3/text_encoder.py`
//! (`TextEncoder.encode`).
//!
//! * The prompt is wrapped in the chat template (single user turn + generation prompt, no system
//!   turn: [`gen_core::flux3::chat_text`]) and tokenized **as one string** with the pinned
//!   tokenizer (`tokenizer(text, truncation=True, max_length=4096)`).
//! * The window is right-padded with `<|endoftext|>` to the next multiple of 80 (never beyond
//!   4096); the pads carry `attention_mask = 0` and position id **1**
//!   ([`gen_core::flux3::assemble_window`]).
//! * The language tower is the shared generic decoder [`mlx_llm::CausalLm`] (the `qwen3_vl`
//!   architecture: GQA 32/8, per-head q/k RMSNorm, SwiGLU, θ = 5e6; interleaved mRoPE collapses
//!   to 1-D for a text-only prompt). It runs in bf16, the release's `text_encoder.dtype`.
//! * Upstream runs the **padded** sequence: every query (the pads included) attends the real
//!   keys causally and no pad key ([`gen_core::flux3::padded_causal_additive`]). The pad rows are
//!   therefore NOT zero — they are the pad token's state at position 1 over the real keys — and
//!   the DiT attends them as ordinary text tokens, so the port computes them exactly the same
//!   way rather than zeroing or dropping them.
//! * The conditioning is the eight hidden states `[4, 8, …, 32]` (HF `output_hidden_states`
//!   indexing: entry `k` is the output of block `k`) concatenated **per token** in layer order —
//!   `[L, 8 · 2560]` bf16 — with ids `(0, 0, 0, position)`.

use std::collections::HashMap;
use std::path::Path;

use gen_core::flux3::{
    assemble_window, chat_text, padded_causal_additive, text_ids, TextEncoderConfig, TextWindow,
    PAD_TOKEN,
};
use gen_core::safetensors_shards::{resolve_indexed_safetensors_shards, snapshot_shard_roots};
use gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::gen_core;
use mlx_gen::{Error, Result};
use mlx_llm::primitives::attention::AttnMask;
use mlx_llm::{CausalLm, ModelConfig};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

/// One prompt's conditioning, as `TextEncoder.encode` returns it.
pub struct TextConditioning {
    /// `[L, layers · hidden]`: selected states per token; BF16 for generation,
    /// explicitly selected precision for the frozen training teacher.
    pub tokens: Array,
    /// `[L]` ids `(0, 0, 0, position)`.
    pub ids: Vec<[f32; 4]>,
    /// The window the tower ran (ids, mask, positions, truncation).
    pub window: TextWindow,
}

/// The loaded Qwen3-VL language tower + tokenizer + the package's conditioning contract.
pub struct Flux3TextEncoder {
    lm: CausalLm,
    tokenizer: TextTokenizer,
    pad_id: i32,
    layers: Vec<usize>,
    pad_multiple: usize,
    max_length: usize,
    output_dtype: Dtype,
    hidden: usize,
}

fn from_llm(e: mlx_llm::Error) -> Error {
    match e {
        mlx_llm::Error::Unsupported(m) => Error::Unsupported(m),
        mlx_llm::Error::MissingTensor(k) => Error::MissingTensor(k),
        mlx_llm::Error::Canceled => Error::Canceled,
        other => Error::Msg(format!("flux3 text encoder: {other}")),
    }
}

/// The safetensors files [`Flux3TextEncoder::load`] opens for the snapshot at `dir`: the shards
/// its `model.safetensors.index.json` names, else the single `model.safetensors`. Shared with the
/// memory contract's asset pricing so both read one file list.
pub(crate) fn weight_shards(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let roots = snapshot_shard_roots(dir).map_err(Error::from)?;
    Ok(
        match resolve_indexed_safetensors_shards(dir, "model.safetensors.index.json", &roots)
            .map_err(Error::from)?
        {
            Some(shards) => shards.into_iter().map(|s| s.loader_path).collect(),
            None => vec![dir.join("model.safetensors")],
        },
    )
}

impl Flux3TextEncoder {
    /// Load from the Qwen3-VL snapshot directory under the package's `text_encoder` section. Only
    /// the language tower's tensors are read. `text_in_channels` is the DiT's `txt_in` width,
    /// checked against `layers × hidden` so a mismatched encoder fails at load, by number.
    pub fn load(dir: &Path, cfg: &TextEncoderConfig, text_in_channels: usize) -> Result<Self> {
        Self::load_internal(dir, cfg, text_in_channels, None)
    }

    /// Frozen training teacher with selected compute/output precision, HF Module.to
    /// frequency-buffer rounding, and source training window cap (default512).
    pub fn load_for_training(
        dir: &Path,
        cfg: &TextEncoderConfig,
        text_in_channels: usize,
        compute: Dtype,
        max_sequence_length: usize,
    ) -> Result<Self> {
        if !matches!(compute, Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16)
            || max_sequence_length == 0
            || max_sequence_length > i32::MAX as usize
        {
            return Err(Error::Unsupported(
                "flux3 training teacher requires F32/F16/BF16 and a positive bounded text window"
                    .into(),
            ));
        }
        Self::load_internal(
            dir,
            cfg,
            text_in_channels,
            Some((compute, max_sequence_length)),
        )
    }

    fn load_internal(
        dir: &Path,
        cfg: &TextEncoderConfig,
        text_in_channels: usize,
        training: Option<(Dtype, usize)>,
    ) -> Result<Self> {
        let model_cfg = ModelConfig::from_dir(dir).map_err(from_llm)?;
        if !model_cfg.architecture.is_qwen3_vl() {
            return Err(Error::Unsupported(format!(
                "flux3: {} is not a Qwen3-VL checkpoint (the DiT was trained on {})",
                dir.display(),
                gen_core::flux3::TEXT_ENCODER_UPSTREAM_REPO
            )));
        }
        let hidden = model_cfg.hidden_size as usize;
        if cfg.layers.len() * hidden != text_in_channels {
            return Err(Error::Msg(format!(
                "flux3: the text encoder at {} is {hidden} wide and the package stacks {} layers \
                 ({} channels), but the DiT's txt_in reads {text_in_channels}",
                dir.display(),
                cfg.layers.len(),
                cfg.layers.len() * hidden
            )));
        }
        let deepest = *cfg.layers.last().unwrap_or(&0);
        if deepest > model_cfg.num_layers {
            return Err(Error::Msg(format!(
                "flux3: text_encoder.layers needs layer {deepest}, but the text encoder at {} has \
                 {} layers",
                dir.display(),
                model_cfg.num_layers
            )));
        }
        if training.is_none() && cfg.dtype != "bfloat16" {
            return Err(Error::Unsupported(format!(
                "flux3: text_encoder.dtype {} is not the bf16 the shared tower computes in",
                cfg.dtype
            )));
        }

        let (output_dtype, max_length) = match training {
            Some((dtype, max_length)) => {
                if cfg.pad_multiple == 0 || max_length > model_cfg.max_position_embeddings as usize
                {
                    return Err(Error::Unsupported("flux3 training: text window exceeds teacher positional capacity or has invalid padding".into()));
                }
                (dtype, max_length)
            }
            None => (Dtype::Bfloat16, cfg.window_max_length()),
        };
        let files = weight_shards(dir)?;
        let mut tensors = HashMap::new();
        for file in &files {
            if !file.is_file() {
                return Err(Error::Msg(format!(
                    "flux3: the text encoder resource is incomplete — shard {} is missing",
                    file.display()
                )));
            }
            let part = Array::load_safetensors(file).map_err(|e| {
                Error::Msg(format!(
                    "flux3: loading text-encoder shard {}: {e}",
                    file.display()
                ))
            })?;
            tensors.extend(part);
        }
        let weights = mlx_llm::primitives::weights::Weights::from_map(tensors);
        // `from_weights` evaluates every parameter group it built (the load boundary), so no
        // forward ever reads a weight from disk.
        let lm = match training {
            Some((dtype, _)) => CausalLm::from_weights_dtype(&weights, "", model_cfg, dtype)
                .map_err(from_llm)?
                .with_training_frequency_storage()
                .map_err(from_llm)?,
            None => CausalLm::from_weights(&weights, "", model_cfg).map_err(from_llm)?,
        };
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
        let pad_id = tokenizer.token_to_id(PAD_TOKEN).ok_or_else(|| {
            Error::Msg(format!(
                "flux3: the tokenizer at {} has no {PAD_TOKEN:?} token (the pinned pad token)",
                dir.display()
            ))
        })? as i32;
        Ok(Self {
            lm,
            tokenizer,
            pad_id,
            layers: cfg.layers.clone(),
            pad_multiple: cfg.pad_multiple,
            max_length,
            output_dtype,
            hidden,
        })
    }

    pub fn pad_id(&self) -> i32 {
        self.pad_id
    }

    pub fn compute_dtype(&self) -> Dtype {
        self.lm.compute_dtype()
    }
    pub fn output_dtype(&self) -> Dtype {
        self.output_dtype
    }
    pub fn precision_profile_key(&self) -> String {
        format!(
            "{};output={:?};max={};layers={:?}",
            self.lm.cache_fingerprint(),
            self.output_dtype,
            self.max_length,
            self.layers
        )
    }

    /// `layers × hidden`: the width of one conditioning row.
    pub fn text_in_channels(&self) -> usize {
        self.layers.len() * self.hidden
    }

    /// Tokenize the template-wrapped prompt and assemble its window.
    pub fn window(&self, prompt: &str) -> Result<TextWindow> {
        let ids = self
            .tokenizer
            .encode_ids(&chat_text(prompt), false)
            .map_err(Error::from)?;
        assemble_window(&ids, self.pad_id, self.pad_multiple, self.max_length).map_err(Error::from)
    }

    /// `TextEncoder.encode(prompt)`.
    pub fn encode(&self, prompt: &str) -> Result<TextConditioning> {
        let window = self.window(prompt)?;
        self.encode_window(window)
    }

    /// Run the tower over an assembled window.
    pub fn encode_window(&self, window: TextWindow) -> Result<TextConditioning> {
        let n = window.length() as i32;
        let ids = Array::from_slice(&window.input_ids, &[1, n]);
        let mask = Array::from_slice(&padded_causal_additive(&window.mask), &[1, 1, n, n])
            .as_dtype(self.lm.compute_dtype())?;
        let mut cache = self.lm.new_cache();
        let states = self
            .lm
            .hidden_states_with_mask_at(
                &ids,
                &mut cache,
                &window.positions,
                AttnMask::Additive(&mask),
            )
            .map_err(from_llm)?;
        // One command buffer per layer: the returned handles are a single lazy graph.
        let deepest = *self.layers.last().unwrap_or(&0);
        for state in states.iter().take(deepest + 1) {
            eval([state])?;
        }
        let picked: Vec<Array> = self
            .layers
            .iter()
            .map(|&k| -> Result<Array> {
                let state = states
                    .get(k)
                    .ok_or_else(|| Error::Msg(format!("flux3: hidden state {k} not returned")))?;
                Ok(state.reshape(&[n, self.hidden as i32])?)
            })
            .collect::<Result<_>>()?;
        let refs: Vec<&Array> = picked.iter().collect();
        // `stack(dim=1).permute(0, 2, 1, 3).reshape(L, -1)`: layer-major within each token row.
        let tokens = concatenate_axis(&refs, 1)?.as_dtype(self.output_dtype)?;
        eval([&tokens])?;
        Ok(TextConditioning {
            tokens,
            ids: text_ids(window.length()),
            window,
        })
    }
}
