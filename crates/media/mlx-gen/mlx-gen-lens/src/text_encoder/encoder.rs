//! The Lens **encoder-only** gpt-oss-20b stack (sc-3171): `embed_tokens` → 24 decoder layers with
//! per-layer sliding/full masks → capture the hidden states at the selected layers `[5, 11, 17, 23]`
//! → early-exit after the last selected layer. A faithful port of
//! `_vendor/lens/text_encoder.py::LensGptOssEncoder.forward` (the Lens feature-extraction path).
//!
//! ## Parity-critical details (from the reference)
//! - **Captured = layer *output*.** `captured[pos] = hidden_states` is taken *after* running decoder
//!   layer `i` (not the embedding-offset `hidden_states[i]` of HF's stock `output_hidden_states`). So
//!   the default selection `[5, 11, 17, 23]` is the output of decoder indices 5/11/17/23.
//! - **Per-layer mask by `layer_types[i]`.** Even layers are sliding-window (window 128), odd layers
//!   are full causal ([`GptOssConfig::is_sliding`]). Both masks are built once for the sequence and
//!   reused; for the un-padded single prompt the Lens encoder runs this is pure causal ±the window.
//! - **`position_ids = arange(L)`**, RoPE computed once (the YaRN `inv_freq` + `attention_scaling`).
//! - **No final `norm`, no LM head, no KV cache, no generation** — the feature path stops at the max
//!   selected layer.
//!
//! The encoder runs the *whole* token sequence (the 97-token harmony preamble is real causal
//! context); the DiT later consumes the captured features sliced at `txt_offset = 97` (sc-3173).

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Quant, Result, WeightsSource};

use crate::config::GptOssConfig;
use crate::text_encoder::gpt_oss::{attention_mask, GptOssDecoderLayer};

/// The Lens default multi-layer capture indices (`selected_layer_index` in the DiT config /
/// `set_selected_layers` default).
pub const DEFAULT_SELECTED_LAYERS: [usize; 4] = [5, 11, 17, 23];

/// The Lens gpt-oss-20b text encoder, run encoder-only with multi-layer hidden capture.
pub struct LensTextEncoder {
    /// `model.embed_tokens.weight`, `[vocab, hidden]`.
    embed_tokens: Array,
    /// Decoder layers `0..=max_selected` (the stack is truncated at the last captured layer — the
    /// remaining layers, the final `norm`, and the LM head are never built).
    layers: Vec<GptOssDecoderLayer>,
    /// YaRN RoPE frequencies `[head_dim/2]` and the `attention_scaling` (mscale), computed once.
    inv_freq: Array,
    attn_scaling: f32,
    /// Layer indices whose outputs are captured, in the order the DiT expects them.
    selected_layers: Vec<usize>,
    sliding_window: i32,
    dtype: Dtype,
    /// `is_sliding` mapping is config-driven; kept for the per-layer mask choice.
    cfg: GptOssConfig,
    /// Rung 4's Sequential-only, re-openable layer source. A streamable encoder deliberately owns no
    /// resident layers; [`Self::encode`] dispatches through this source so an unscoped call cannot
    /// silently return the bare token embedding.
    stream: Option<super::stream::TextEncoderBlockStream>,
}

impl LensTextEncoder {
    /// Load the encoder from the full `text_encoder` weights at `dtype` (bf16 production / f32 gate),
    /// capturing the [`DEFAULT_SELECTED_LAYERS`]. Only layers `0..=max(selected)` are constructed —
    /// for the default selection that is all 24, but a smaller selection loads (and dequantizes the
    /// MXFP4 experts of) only the needed prefix.
    pub fn from_weights(w: Weights, cfg: &GptOssConfig, dtype: Dtype) -> Result<Self> {
        Self::with_selected_layers(w, cfg, dtype, DEFAULT_SELECTED_LAYERS.to_vec(), None)
    }

    /// As [`from_weights`](Self::from_weights) but quantizes the MoE experts to Q4/Q8 (sc-3172) so the
    /// encoder loads at `~12 GB` instead of `~40 GB` bf16. Attention / router / embedding stay dense.
    pub fn from_weights_quant(
        w: Weights,
        cfg: &GptOssConfig,
        dtype: Dtype,
        quant: Option<Quant>,
    ) -> Result<Self> {
        Self::with_selected_layers(w, cfg, dtype, DEFAULT_SELECTED_LAYERS.to_vec(), quant)
    }

    /// As [`from_weights`](Self::from_weights) but with an explicit (non-empty, unique, in-range)
    /// capture-index list (`set_selected_layers`) and optional MoE-expert quantization.
    ///
    /// **Consumes `w`** (sc-11030): each layer's source tensors are dropped from the map as soon as
    /// the layer is built, so the 13 GB gpt-oss source and the growing 21/63 GB built encoder don't
    /// both stay resident. This bounds the load-time unified-memory transient to ~the built size,
    /// which is what lets `OffloadPolicy::Sequential` actually reduce peak memory (otherwise the
    /// source+built LOAD spike, not the resident component sum, is the peak — no staging win).
    pub fn with_selected_layers(
        mut w: Weights,
        cfg: &GptOssConfig,
        dtype: Dtype,
        selected_layers: Vec<usize>,
        quant: Option<Quant>,
    ) -> Result<Self> {
        let max_layer = validate_selected_layers(cfg, &selected_layers)?;

        let embed_tokens = w.require("model.embed_tokens.weight")?.as_dtype(dtype)?;
        // Source is dropped from the map as each component is built (sc-11030) — the load transient
        // stays ~= the built encoder rather than source(13 GB) + built.
        w.remove("model.embed_tokens.weight");
        let mut layers = Vec::with_capacity(max_layer + 1);
        for i in 0..=max_layer {
            layers.push(GptOssDecoderLayer::from_weights(
                &w,
                &format!("model.layers.{i}"),
                cfg,
                dtype,
                quant,
            )?);
            // Free this layer's source tensors now that the layer is built (its Linears/experts were
            // copied/quantized into fresh Arrays, so the source is unreferenced). MLX returns the
            // buffers to its reuse pool for the next layer's allocations.
            w.remove_prefix(&format!("model.layers.{i}."));
        }

        let (inv_freq, attn_scaling) = cfg.yarn_rope();
        Ok(Self {
            embed_tokens,
            layers,
            inv_freq: Array::from_slice(&inv_freq, &[inv_freq.len() as i32]),
            attn_scaling,
            selected_layers,
            sliding_window: cfg.sliding_window,
            dtype,
            cfg: *cfg,
            stream: None,
        })
    }

    /// Build the Sequential-only rung-4 form: keep the token embedding resident and leave the
    /// decoder stack in a re-openable source, materializing it through the shared block-window driver
    /// during [`Self::encode_windowed`]. `quant` is replayed for every materialized layer so dense,
    /// load-time Q4/Q8, and packed-turnkey paths use the same constructor as the resident encoder.
    pub fn from_streamable_source(
        mut w: Weights,
        source: WeightsSource,
        cfg: &GptOssConfig,
        dtype: Dtype,
        selected_layers: Vec<usize>,
        quant: Option<Quant>,
    ) -> Result<Self> {
        validate_selected_layers(cfg, &selected_layers)?;
        let embed_tokens = w.require("model.embed_tokens.weight")?.as_dtype(dtype)?;
        // The view owns a refcounted handle to every tensor it returned. Drain the embedding handle
        // before dropping the otherwise-lazy view; the layer views use the same load-bearing rule.
        w.remove_accessed();
        let (inv_freq, attn_scaling) = cfg.yarn_rope();
        Ok(Self {
            embed_tokens,
            layers: Vec::new(),
            inv_freq: Array::from_slice(&inv_freq, &[inv_freq.len() as i32]),
            attn_scaling,
            selected_layers: selected_layers.clone(),
            sliding_window: cfg.sliding_window,
            dtype,
            cfg: *cfg,
            stream: Some(super::stream::TextEncoderBlockStream::new(
                source,
                *cfg,
                dtype,
                selected_layers,
                quant,
            )),
        })
    }

    /// Whether this encoder can execute rung 4's text-encoder component scope.
    pub fn is_streamable(&self) -> bool {
        self.stream.is_some()
    }

    /// The capture indices, in DiT order.
    pub fn selected_layers(&self) -> &[usize] {
        &self.selected_layers
    }

    /// Encode `input_ids` `[B, L]` (int32) → the captured hidden states, one `[B, L, hidden]` per
    /// selected layer in selection order (== `LensGptOssEncoder.forward`'s returned list). Runs
    /// `position_ids = arange(L)` and stops after the max selected layer.
    ///
    /// `cancel` is the cooperative cancellation handle (F-019/F-029): checked before each of the ~24
    /// gpt-oss MoE decoder layers (the dominant cost on a 4-step turbo run) so a cancel during the
    /// encode is honored. Returns [`Error::Canceled`] on trip.
    ///
    /// F-029: under lazy MLX execution the per-layer check alone is a *false green* — the original
    /// premise ("routing forces a host sync per layer") went stale when sc-9500 moved MoE routing
    /// on-device, so all ~24 checks executed in microseconds during graph *construction* while the
    /// entire 20B compute ran later in one uninterruptible `eval`. We now force `eval([&hidden])`
    /// after each layer **when a cancel handle is present**, so the next iteration's check observes
    /// materialized state and a cancel is honored within one layer's compute rather than the whole
    /// encode. The eval is skipped entirely when `cancel` is `None` (the graph stays lazy).
    pub fn encode(&self, input_ids: &Array, cancel: Option<&CancelFlag>) -> Result<Vec<Array>> {
        // The streamable form intentionally has no resident layers. One full-width window preserves
        // ordinary unscoped behavior and closes the empty-stack hazard: returning the embedding here
        // would produce plausible images that silently ignore the prompt.
        if self.layers.is_empty() {
            if let Some(stream) = self.stream.as_ref() {
                return self.encode_windowed(
                    input_ids,
                    stream.n_blocks(),
                    cancel.unwrap_or(&CancelFlag::default()),
                );
            }
        }
        let l = input_ids.shape()[1];

        // Both per-layer masks, built once for the sequence (full causal + sliding-window causal).
        let full_mask = attention_mask(l, None, self.dtype)?;
        let sliding_mask = attention_mask(l, Some(self.sliding_window), self.dtype)?;

        let mut hidden = self.embed_tokens.take_axis(input_ids, 0)?; // [B, L, hidden]

        // Capture slots, filled in selection order (matches the reference's `index_lookup`).
        let mut captured: Vec<Option<Array>> = vec![None; self.selected_layers.len()];
        for (i, layer) in self.layers.iter().enumerate() {
            if cancel.is_some_and(CancelFlag::is_cancelled) {
                return Err(Error::Canceled);
            }
            let mask = if self.cfg.is_sliding(i) {
                &sliding_mask
            } else {
                &full_mask
            };
            hidden = layer.forward(&hidden, &self.inv_freq, self.attn_scaling, mask)?;
            if let Some(pos) = self.selected_layers.iter().position(|&s| s == i) {
                captured[pos] = Some(hidden.clone());
            }
            // F-029: materialize this layer's work so the next iteration's cancel check observes real
            // progress. Without this the loop's checks are graph-time-only (a no-op cancel). Only pay
            // the per-layer sync when a cancel handle is actually threaded.
            if cancel.is_some() {
                mlx_rs::transforms::eval([&hidden])?;
            }
        }

        Ok(captured
            .into_iter()
            .map(|c| c.expect("every selected layer captured"))
            .collect())
    }

    /// Encode through the shared rung-4 block-window driver. Errors when the encoder has no
    /// re-openable source rather than silently running the resident stack under a selected strategy.
    pub fn encode_windowed(
        &self,
        input_ids: &Array,
        window: usize,
        cancel: &CancelFlag,
    ) -> Result<Vec<Array>> {
        let stream = self.stream.as_ref().ok_or_else(|| {
            Error::Msg(
                "lens: the rung-4 text-encoder scope needs a Sequential loader with a re-openable \
                 snapshot source"
                    .to_owned(),
            )
        })?;
        let l = input_ids.shape()[1];
        let full_mask = attention_mask(l, None, self.dtype)?;
        let sliding_mask = attention_mask(l, Some(self.sliding_window), self.dtype)?;
        let hidden = self.embed_tokens.take_axis(input_ids, 0)?;
        let plan = mlx_gen::block_residency::BlockPlan::new(stream.n_blocks(), window)?;
        super::stream::run_windowed_layers(
            stream,
            &plan,
            cancel,
            hidden,
            &self.inv_freq,
            self.attn_scaling,
            &full_mask,
            &sliding_mask,
        )
    }
}

fn validate_selected_layers(cfg: &GptOssConfig, selected_layers: &[usize]) -> Result<usize> {
    // Reachable from `Result`-returning public APIs, so error rather than panic the worker on a bad
    // capture list. Duplicates would otherwise leave a capture slot unfilled at the end of encode.
    let max_layer = *selected_layers
        .iter()
        .max()
        .ok_or_else(|| Error::Msg("lens encoder: selected_layers must be non-empty".into()))?;
    if max_layer >= cfg.num_layers {
        return Err(Error::Msg(format!(
            "lens encoder: selected layer {max_layer} out of range (model has {} layers)",
            cfg.num_layers
        )));
    }
    for (j, &layer) in selected_layers.iter().enumerate() {
        if selected_layers[..j].contains(&layer) {
            return Err(Error::Msg(format!(
                "lens encoder: selected_layers must be unique (layer {layer} repeated)"
            )));
        }
    }
    Ok(max_layer)
}
