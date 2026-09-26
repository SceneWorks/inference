//! The Candle SheetSage2 model: MERT2 encoder (LoRA merged) → softmax layer mix → projection →
//! BART decoder, decoded greedily under the grammar (`modeling_sheetsage2.py` /
//! `generation_sheetsage2.constrained_prompt_generate_batch` at `4f89269`, batch size 1).

pub mod config;
pub mod decoder;
pub mod frontend;
pub mod mert;
pub(crate) mod weights;

use std::sync::atomic::{AtomicUsize, Ordering};

use candle_audio::candle_core::{Device, Tensor};

use crate::grammar::{argmax_first, mask_logits, GrammarState};
use crate::tokenizer::{TokenType, Tokenizer, EOS, OUT, SOS};
use crate::Error;
use config::{MertConfig, SheetSage2Config};
use decoder::Decoder;
use mert::{Linear, MertEncoder};
use weights::Weights;

/// One greedy step, as seen by a [`SheetSage2Model::generate`] observer.
#[derive(Debug)]
pub struct GenerationStep<'a> {
    /// Sequence length before this step's token.
    pub position: usize,
    /// Raw logits of the next token.
    pub logits: &'a [f32],
    /// The grammar-masked choice.
    pub token: u32,
    /// Masked top-1 minus top-2 logit (`inf` when only one token was allowed).
    pub margin: f32,
}

static LIVE_MODELS: AtomicUsize = AtomicUsize::new(0);

/// SheetSage2 models alive in this process: every successful [`SheetSage2Model::load`] not yet
/// dropped. The cover path requires zero before it loads a generator.
pub fn live_models() -> usize {
    LIVE_MODELS.load(Ordering::SeqCst)
}

/// The loaded, merged model.
pub struct SheetSage2Model {
    config: SheetSage2Config,
    tokenizer: Tokenizer,
    encoder: MertEncoder,
    layer_weights: Vec<f32>,
    projection: Linear,
    decoder: Decoder,
    device: Device,
}

/// Encoder outputs kept for parity measurement.
#[derive(Debug)]
pub struct AudioFeatures {
    /// Normalized mel frames.
    pub mel: Tensor,
    /// Subsampler output (mix input 0).
    pub input_hidden: Tensor,
    /// Every block output.
    pub blocks: Vec<Tensor>,
    /// The softmax-weighted mix of the 25 states.
    pub mixed: Tensor,
    /// The decoder memory.
    pub memory: Tensor,
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

impl SheetSage2Model {
    /// Build the model from SheetSage2's `config.json`, the MERT parent's `config.json`, the
    /// SheetSage2 head/adapter weights and the MERT parent weights. The parent's architecture must
    /// equal SheetSage2's `backbone_config` (upstream's "parent architecture mismatch" check), the
    /// vocabulary fingerprint must match, and every tensor of both files must be consumed.
    pub(crate) fn load(
        sheetsage2_config: &serde_json::Value,
        mert_config: &serde_json::Value,
        mut head: Weights,
        mut parent: Weights,
        device: &Device,
    ) -> Result<Self, Error> {
        let config = SheetSage2Config::from_json(sheetsage2_config)?;
        let parent_config = MertConfig::from_json(mert_config)?;
        if parent_config != config.backbone {
            return Err(Error::Config(
                "MERT-v2 parent architecture mismatch with SheetSage2's backbone_config".into(),
            ));
        }
        let tokenizer = Tokenizer::new(
            config.input_audio_length,
            config.time_hz,
            Some(&config.tokenizer_fingerprint),
        )?;
        if tokenizer.n_tokens() as usize != config.vocab_size {
            return Err(Error::Config(
                "tokenizer vocabulary size does not match the model".into(),
            ));
        }
        let encoder = MertEncoder::load(
            &config.backbone,
            &mut parent,
            &mut head,
            config.lora_rank,
            config.lora_alpha,
            device,
        )?;
        parent.finish()?;
        let layers = config.backbone.num_hidden_layers + 1;
        let layer_weight = head.take("layer_weight", &[layers])?.to_vec1::<f32>()?;
        let projection = Linear::new(
            &head.take(
                "encoder_projection.weight",
                &[config.hidden_size, config.backbone.hidden_size],
            )?,
            Some(head.take("encoder_projection.bias", &[config.hidden_size])?),
            device,
        )?;
        let decoder = Decoder::load(&mut head, &config, device)?;
        head.finish()?;
        let model = Self {
            layer_weights: softmax(&layer_weight),
            config,
            tokenizer,
            encoder,
            projection,
            decoder,
            device: device.clone(),
        };
        LIVE_MODELS.fetch_add(1, Ordering::SeqCst);
        Ok(model)
    }

    /// The configuration.
    pub fn config(&self) -> &SheetSage2Config {
        &self.config
    }

    /// The vocabulary.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Parameter bytes held on the device.
    pub fn parameter_bytes(&self) -> usize {
        self.encoder.bytes() + self.projection.bytes() + self.decoder.bytes()
    }

    /// The device the model runs on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Upstream `_prepare_audio` for one waveform: finite, at least `n_fft/2 + 1` and at most one
    /// window of samples, zero-padded to the fixed window (the encoder always attends to the whole
    /// window, silence included).
    pub fn prepare_window(&self, samples: &[f32]) -> Result<Vec<f32>, Error> {
        let minimum = self.config.backbone.minimum_input_samples();
        let window = self.config.window_samples();
        if samples.iter().any(|s| !s.is_finite()) {
            return Err(Error::Request(
                "audio must contain only finite samples".into(),
            ));
        }
        if samples.len() < minimum {
            return Err(Error::Request(format!(
                "each waveform must contain at least {minimum} samples"
            )));
        }
        if samples.len() > window {
            return Err(Error::Request(
                "audio exceeds one model window; use the whole-song transcriber".into(),
            ));
        }
        let stride = self.config.backbone.inputs_to_logits_ratio();
        let mut padded = samples.to_vec();
        padded.resize(window, 0.0);
        if !window.is_multiple_of(stride) {
            padded.resize(window + stride - window % stride, 0.0);
        }
        Ok(padded)
    }

    /// Encode one window into the decoder memory `[1, frames, width]`, streaming the layer mix.
    pub fn encode(&self, samples: &[f32]) -> Result<Tensor, Error> {
        let window = self.prepare_window(samples)?;
        let mut mixed: Option<Tensor> = None;
        self.encoder.forward_states(&window, |index, state| {
            let term = (state * f64::from(self.layer_weights[index]))?;
            mixed = Some(match mixed.take() {
                None => term,
                Some(m) => (m + term)?,
            });
            Ok(())
        })?;
        self.projection
            .forward(&mixed.expect("at least the subsampler state"))
    }

    /// Encode one window keeping every intermediate state (parity measurement).
    pub fn audio_features(&self, samples: &[f32]) -> Result<AudioFeatures, Error> {
        let window = self.prepare_window(samples)?;
        let states = self.encoder.forward_all(&window)?;
        let mut mixed = (&states.input_hidden * f64::from(self.layer_weights[0]))?;
        for (i, block) in states.blocks.iter().enumerate() {
            mixed = (mixed + (block * f64::from(self.layer_weights[i + 1]))?)?;
        }
        let memory = self.projection.forward(&mixed)?;
        Ok(AudioFeatures {
            mel: states.mel,
            input_hidden: states.input_hidden,
            blocks: states.blocks,
            mixed,
            memory,
        })
    }

    /// Grammar-constrained greedy decoding from `prefix` (must start with `<|sos|>` and contain
    /// `<|out|>`), up to `max_length` tokens. When `stop_time` is set, generating a time token at
    /// or past it appends `<|eos|>` and stops (upstream's window stop rule). The result always ends
    /// with `<|eos|>`. `observe` sees every step and may abort (cancellation) by returning an error.
    pub fn generate(
        &self,
        memory: &Tensor,
        prefix: &[u32],
        max_length: usize,
        stop_time: Option<f64>,
        mut observe: impl FnMut(&GenerationStep) -> Result<(), Error>,
    ) -> Result<Vec<u32>, Error> {
        let mut prefix = prefix.to_vec();
        if prefix.first() != Some(&SOS) {
            return Err(Error::Request(
                "generation prefix must begin with <|sos|>".into(),
            ));
        }
        if prefix.last() == Some(&EOS) {
            prefix.pop();
        }
        let out_index = prefix
            .iter()
            .position(|&t| t == OUT)
            .ok_or_else(|| Error::Request("generation prefix has no <|out|>".into()))?;
        if max_length <= prefix.len() || max_length > self.config.max_output_seq_len {
            return Err(Error::Request(
                "the token limit must exceed the prefix length and fit the decoder context".into(),
            ));
        }
        let tokenizer = &self.tokenizer;
        let mut state = GrammarState::new();
        for &token in &prefix[out_index + 1..] {
            state.update(tokenizer, token)?;
        }
        let mut output = prefix.clone();
        let mut cache = self.decoder.start(memory)?;
        let mut input = prefix;
        while output.len() < max_length {
            let logits = self.decoder.step(&mut cache, &input)?.to_vec1::<f32>()?;
            let mut masked = logits.clone();
            mask_logits(&mut masked, &state.allowed(tokenizer));
            let token = argmax_first(&masked);
            let best = masked[token as usize];
            let second = masked
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != token as usize)
                .map(|(_, v)| *v)
                .fold(f32::NEG_INFINITY, f32::max);
            observe(&GenerationStep {
                position: output.len(),
                logits: &logits,
                token,
                margin: best - second,
            })?;
            output.push(token);
            let mut finished = state.update(tokenizer, token)?;
            if !finished {
                if let (Some(stop), Some(id)) = (stop_time, tokenizer.time_id(token)) {
                    if tokenizer.token_type(token)? == TokenType::Time
                        && f64::from(id) / f64::from(tokenizer.time_hz()) >= stop
                    {
                        output.push(EOS);
                        finished = true;
                    }
                }
            }
            if finished {
                break;
            }
            input = vec![token];
        }
        if output.last() != Some(&EOS) {
            output.push(EOS);
        }
        Ok(output)
    }
}

impl Drop for SheetSage2Model {
    fn drop(&mut self) {
        LIVE_MODELS.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
pub(crate) mod tests;
