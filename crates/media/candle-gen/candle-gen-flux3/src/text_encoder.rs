//! Qwen3-VL-4B text conditioning — the port of upstream `src/flux3/text_encoder.py`
//! (`TextEncoder.encode`), the Candle twin of `mlx_gen_flux3::text_encoder`.
//!
//! * The prompt is wrapped in the chat template (single user turn + generation prompt, no system
//!   turn: [`gen_core::flux3::chat_text`]) and tokenized **as one string** with the pinned
//!   tokenizer (`tokenizer(text, truncation=True, max_length=4096)`).
//! * The window is right-padded with `<|endoftext|>` to the next multiple of 80 (never beyond
//!   4096); the pads carry `attention_mask = 0` and position id **1**
//!   ([`gen_core::flux3::assemble_window`]).
//! * The language tower is the shared generic decoder [`candle_llm::CausalLm`] (the `qwen3_vl`
//!   architecture: GQA 32/8, per-head q/k RMSNorm, SwiGLU, θ = 5e6; interleaved mRoPE collapses to
//!   1-D for a text-only prompt). On a GPU it runs in the package's `text_encoder.dtype` (bf16 in
//!   the release); on the Candle CPU backend, which has no half-precision GEMM, in f32 (the same
//!   CPU policy every `candle_llm` decoder follows — [`tower_dtype`]).
//! * Upstream runs the **padded** sequence: every query (the pads included) attends the real keys
//!   causally and no pad key ([`gen_core::flux3::padded_causal_additive`]), at the explicit
//!   positions `0..real` for the real tokens and `1` for every pad
//!   (`CausalLm::hidden_states_with_mask_at`). The pad rows are therefore NOT zero — they are the
//!   pad token's state at position 1 over the real keys — and the DiT attends them as ordinary
//!   text tokens, so the port computes them exactly the same way rather than zeroing or dropping
//!   them.
//! * The conditioning is the eight hidden states `[4, 8, …, 32]` (HF `output_hidden_states`
//!   indexing: entry `k` is the output of block `k`) concatenated **per token** in layer order —
//!   `[L, 8 · 2560]` bf16 (upstream's `.to(torch.bfloat16)`, whatever the tower computed in) —
//!   with ids `(0, 0, 0, position)`.
//!
//! [`gen_core::flux3::chat_text`]: candle_gen::gen_core::flux3::chat_text
//! [`gen_core::flux3::assemble_window`]: candle_gen::gen_core::flux3::assemble_window
//! [`gen_core::flux3::padded_causal_additive`]: candle_gen::gen_core::flux3::padded_causal_additive

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::flux3::{
    assemble_window, chat_text, padded_causal_additive, text_ids, TextEncoderConfig, TextWindow,
    PAD_TOKEN, TEXT_ENCODER_UPSTREAM_REPO,
};
use candle_gen::gen_core::safetensors_shards::{
    resolve_indexed_safetensors_shards, snapshot_shard_roots,
};
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::{CandleError as Error, Result};
use candle_llm::primitives::attention::AttnMask;
use candle_llm::primitives::weights::Weights as LlmWeights;
use candle_llm::{CausalLm, ModelConfig};

/// One prompt's conditioning, as `TextEncoder.encode` returns it.
pub struct TextConditioning {
    /// `[L, layers · hidden]`: selected states per token; BF16 for generation,
    /// explicitly selected compute/output dtype for the frozen training teacher.
    pub tokens: Tensor,
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
    output_dtype: DType,
    hidden: usize,
    device: Device,
}

fn from_llm(e: candle_llm::Error) -> Error {
    match e {
        candle_llm::Error::Candle(c) => Error::Candle(c),
        candle_llm::Error::Unsupported(m) => Error::Unsupported(m),
        candle_llm::Error::Canceled => Error::Canceled,
        other => Error::Msg(format!("flux3 text encoder: {other}")),
    }
}

/// The dtype the package's `text_encoder.dtype` names — what the tower runs in on a GPU device
/// ([`tower_dtype`]). The release names `bfloat16`; anything else is refused by name.
pub fn declared_tower_dtype(cfg: &TextEncoderConfig) -> Result<DType> {
    match cfg.dtype.as_str() {
        "bfloat16" => Ok(DType::BF16),
        other => Err(Error::Unsupported(format!(
            "flux3: text_encoder.dtype = {other} is not a dtype the Qwen3-VL tower can run in"
        ))),
    }
}

/// The tower's compute dtype: the package's `text_encoder.dtype` on a GPU device; f32 on the
/// Candle CPU backend, which has no half-precision GEMM.
pub fn tower_dtype(cfg: &TextEncoderConfig, device: &Device) -> Result<DType> {
    let declared = declared_tower_dtype(cfg)?;
    Ok(if device.is_cpu() {
        DType::F32
    } else {
        declared
    })
}

/// The safetensors files [`Flux3TextEncoder::load`] opens for the snapshot at `dir`: the shards
/// its `model.safetensors.index.json` names, else the single `model.safetensors`. Every listed file
/// must exist. Shared with the memory contract's asset pricing so both read one file list.
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
                "flux3: the text encoder resource is incomplete — shard {} is missing",
                file.display()
            )));
        }
    }
    Ok(files)
}

/// Cast each floating checkpoint tensor on the host before the upload boundary. Keeping an
/// F32 device map beside the constructed BF16 tower would exceed the header-priced load shape.
fn upload_text_weight(
    tensor: Tensor,
    dtype: DType,
    upload: impl FnOnce(Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    let tensor = if tensor.dtype().is_float() {
        tensor.to_dtype(dtype)?
    } else {
        tensor
    };
    upload(tensor)
}

impl Flux3TextEncoder {
    /// Load from the Qwen3-VL snapshot directory under the package's `text_encoder` section, onto
    /// `device`. Only the language tower's tensors reach the device; the vision tower
    /// (`model.visual.*`) is dropped on the host, one shard at a time. `text_in_channels` is the
    /// DiT's `txt_in` width, checked against `layers × hidden` so a mismatched encoder fails at
    /// load, by number.
    pub fn load(
        dir: &Path,
        cfg: &TextEncoderConfig,
        text_in_channels: usize,
        device: &Device,
    ) -> Result<Self> {
        Self::load_internal(dir, cfg, text_in_channels, device, None)
    }

    /// Frozen training teacher with explicit selected compute/output dtype and
    /// source training token cap (default512), independent of inference's policy.
    pub fn load_for_training(
        dir: &Path,
        cfg: &TextEncoderConfig,
        text_in_channels: usize,
        device: &Device,
        compute: DType,
        max_sequence_length: usize,
    ) -> Result<Self> {
        if !matches!(compute, DType::F32 | DType::F16 | DType::BF16)
            || max_sequence_length == 0
            || max_sequence_length > i32::MAX as usize
        {
            return Err(Error::Unsupported(
                "flux3 training teacher requires F32/F16/BF16 and a positive bounded text window"
                    .into(),
            ));
        }
        if device.is_cpu() && compute == DType::BF16 {
            return Err(Error::Unsupported("flux3 training: the Candle CPU teacher has no BF16 GEMM; choose F32 or native CUDA, never a silent F32 teacher".into()));
        }
        Self::load_internal(
            dir,
            cfg,
            text_in_channels,
            device,
            Some((compute, max_sequence_length)),
        )
    }

    fn load_internal(
        dir: &Path,
        cfg: &TextEncoderConfig,
        text_in_channels: usize,
        device: &Device,
        training: Option<(DType, usize)>,
    ) -> Result<Self> {
        let model_cfg = ModelConfig::from_dir(dir).map_err(from_llm)?;
        if !model_cfg.architecture.is_qwen3_vl() {
            return Err(Error::Unsupported(format!(
                "flux3: {} is not a Qwen3-VL checkpoint (the DiT was trained on \
                 {TEXT_ENCODER_UPSTREAM_REPO})",
                dir.display()
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
        let (dtype, output_dtype, max_length) = match training {
            Some((dtype, max_length)) => {
                if cfg.pad_multiple == 0 || max_length > model_cfg.max_position_embeddings as usize
                {
                    return Err(Error::Unsupported("flux3 training: text window exceeds the teacher's positional capacity or has invalid padding".into()));
                }
                (dtype, dtype, max_length)
            }
            None => (
                tower_dtype(cfg, device)?,
                DType::BF16,
                cfg.window_max_length(),
            ),
        };
        let files = shard_files(dir)?;
        if training.is_some() {
            // Header-only dense-format refusal before the first device upload.
            for file in &files {
                for header in
                    candle_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(file)
                        .map_err(Error::from)?
                {
                    let key = header.name.as_str();
                    if !(key.starts_with("model.language_model.") || key.starts_with("lm_head.")) {
                        continue;
                    }
                    if key.ends_with(".scales")
                        || key.ends_with(".biases")
                        || ((key.ends_with(".weight") || key.ends_with(".bias"))
                            && !header.is_float())
                    {
                        return Err(Error::Unsupported("flux3 training teacher requires dense floating weights; packed/quantized teachers are not source-supported".into()));
                    }
                }
            }
        }
        let mut tensors = HashMap::new();
        for file in files {
            let part =
                candle_gen::candle_core::safetensors::load(&file, &Device::Cpu).map_err(|e| {
                    Error::Msg(format!(
                        "flux3: loading text-encoder shard {}: {e}",
                        file.display()
                    ))
                })?;
            for (key, tensor) in part {
                if key.starts_with("model.visual.") {
                    continue;
                }
                tensors.insert(
                    key,
                    upload_text_weight(tensor, dtype, |tensor| Ok(tensor.to_device(device)?))?,
                );
            }
        }
        let weights = LlmWeights::from_map(tensors, device.clone());
        let lm =
            CausalLm::from_weights_dtype(&weights, "", model_cfg, None, dtype).map_err(from_llm)?;
        let lm = if training.is_some() {
            lm.with_training_frequency_storage().map_err(from_llm)?
        } else {
            lm
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
            device: device.clone(),
        })
    }

    pub fn pad_id(&self) -> i32 {
        self.pad_id
    }

    /// `layers × hidden`: the width of one conditioning row.
    pub fn text_in_channels(&self) -> usize {
        self.layers.len() * self.hidden
    }

    /// The tower's compute dtype.
    pub fn dtype(&self) -> DType {
        self.lm.compute_dtype()
    }

    pub fn output_dtype(&self) -> DType {
        self.output_dtype
    }

    pub fn precision_profile_key(&self) -> String {
        format!(
            "{};output={:?};max={};layers={:?}",
            self.lm.precision_profile_key(),
            self.output_dtype,
            self.max_length,
            self.layers
        )
    }

    pub fn device(&self) -> &Device {
        &self.device
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
        let n = window.length();
        let ids: Vec<u32> = window.input_ids.iter().map(|&i| i as u32).collect();
        let ids = Tensor::from_vec(ids, (1, n), &self.device)?;
        let mask = Tensor::from_vec(
            padded_causal_additive(&window.mask),
            (1, 1, n, n),
            &self.device,
        )?
        .to_dtype(self.lm.compute_dtype())?;
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
        let picked: Vec<Tensor> = self
            .layers
            .iter()
            .map(|&k| -> Result<Tensor> {
                let state = states
                    .get(k)
                    .ok_or_else(|| Error::Msg(format!("flux3: hidden state {k} not returned")))?;
                Ok(state.reshape((n, self.hidden))?)
            })
            .collect::<Result<_>>()?;
        // `stack(dim=1).permute(0, 2, 1, 3).reshape(L, -1)`: layer-major within each token row.
        let tokens = __model_proof_output!(Tensor::cat(&picked, 1)?.to_dtype(self.output_dtype))?;
        Ok(TextConditioning {
            tokens,
            ids: text_ids(n),
            window,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_stored_weights_are_bf16_before_the_upload_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("f32-stored.safetensors");
        let source =
            Tensor::from_vec(vec![1.00391_f32, -0.001, 2.125], (3,), &Device::Cpu).unwrap();
        candle_gen::candle_core::safetensors::save(
            &HashMap::from([("weight".to_owned(), source.clone())]),
            &file,
        )
        .unwrap();
        let mut stored = candle_gen::candle_core::safetensors::load(&file, &Device::Cpu).unwrap();
        let input = stored.remove("weight").unwrap();
        assert_eq!(input.dtype(), DType::F32);
        let output = upload_text_weight(input, DType::BF16, |staged| {
            assert!(
                staged.device().is_cpu(),
                "conversion must occur while still host-resident"
            );
            assert_eq!(
                staged.dtype(),
                DType::BF16,
                "no full F32 map may cross the upload boundary"
            );
            assert_eq!(staged.elem_count() * staged.dtype().size_in_bytes(), 6);
            Ok(staged)
        })
        .unwrap();
        assert_eq!(
            output
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            source
                .to_dtype(DType::BF16)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
        let integer = Tensor::from_vec(vec![2_u32, 129], (2,), &Device::Cpu).unwrap();
        let integer = upload_text_weight(integer, DType::BF16, |staged| {
            assert_eq!(staged.dtype(), DType::U32);
            Ok(staged)
        })
        .unwrap();
        assert_eq!(integer.to_vec1::<u32>().unwrap(), [2, 129]);
    }
}
