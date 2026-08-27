//! Decoder-stack residency: hold every layer, or stream them one at a time (sc-18798).
//!
//! [`EncoderResidency::Sequential`] materializes decoder layer `i` out of a freshly opened
//! [`Weights`] view, runs it, evaluates the carry, drops the layer, and drains the view of exactly
//! the tensors that layer read — then advances. Peak weight residency over a stack pass is one
//! layer instead of all of them. [`EncoderResidency::Resident`] is the historical behaviour and
//! stays the default everywhere.
//!
//! # Why this lives here and not in the media crates
//!
//! LTX-2.5's text encoder is a Gemma 4 decoder, and on LTX the **text phase binds the peak** — the
//! encoder is 26.3 GB against a q4 DiT's ~10.6 GiB, so bounding the DiT harder cannot move it.
//! Every other consumer of this crate's decoder has the same lever available for the same reason,
//! which is why the stream is a property of [`CausalLm`](crate::CausalLm) rather than a loader inside
//! `mlx-gen-ltx`. Epic 18755 R9 names the provider-local streamed loader as the trap to avoid
//! (sc-15958); this is the shared-primitive side of that rule.
//!
//! # The two hazards, both load-bearing
//!
//! 1. **Evaluate before dropping.** MLX is lazy. Dropping a layer whose output is still an
//!    unevaluated graph frees nothing — the graph holds the weights alive — so the stream would
//!    cost a re-open per layer and bound nothing. `SequentialStack::run_layer` evaluates the
//!    carry before the layer goes out of scope. This is the same rule Z-Image's encoder stream
//!    documents on its `EncoderCarry`.
//! 2. **Drain the view.** `Array` is refcounted and the [`Weights`] map keeps its own handle on
//!    every tensor it handed out, so dropping the layer alone still leaves the map holding the
//!    bytes. [`Weights::remove_accessed`] drops exactly the keys this layer read. Measured on the
//!    DiT twin of this primitive, the distinction was 8.0 MiB vs 238.4 MiB.
//!
//! Neither hazard is visible in the output: a stream with hazard 1 or 2 unfixed returns numbers
//! identical to a correct one and merely fails to save any memory.
//!
//! # What it does not bound
//!
//! Only the **layer stack**. Token embeddings, the final norm and the LM head stay resident — on
//! Gemma 4 that is `model.embed_tokens` at ~2.01 GB against ~21.85 GB of layer projections, so
//! streaming the stack is the lever that matters. The KV cache is also untouched; an encoder
//! forward fills it for every layer, and bounding that is a separate axis (`PagedKvCache`).
//!
//! # Per-tier text-encoder quantization interacts with this
//!
//! A tier's text encoder is not always packed at the tier's width. `q4` ships the LTX-2.5 encoder
//! **dense**, on measured evidence — see
//! `mlx_gen_ltx::tiers::TEXT_ENCODER_Q4_QUALITY` for the decision, the numbers behind it and the
//! whole-pipeline-tier exception it declares. The consequence for residency, which is this
//! module's concern: on `q4` the text phase carries a *bf16* encoder, so `q4` is the tier where the
//! text phase binds hardest and where streaming buys the most — the opposite of the intuition that
//! the smallest tier has the smallest text phase. Sizes bear that out: `q4` is only about 2 %
//! smaller than `q8` overall, because the encoder it ships is the larger of the two.
//!
//! # Loader identity
//!
//! [`CausalLm::stream_observation`](crate::CausalLm::stream_observation) returns `Some` only when
//! this stack is engaged, and counts what
//! the stream actually did. That is the observation AC1 asks for: comparing outputs cannot
//! distinguish a streamed pass from a resident one — they are numerically identical by
//! construction, which is the point — so the claim "the streamed loader ran" has to be observed on
//! the loader.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use mlx_rs::transforms::eval;
use mlx_rs::Array;

use crate::config::ModelConfig;
use crate::error::Result;
use crate::models::llama::{LayerPlan, RopeTables, SharedKv};
use crate::primitives::projection::QuantSpec;
use crate::primitives::{AttnMask, KvCache, Weights};

/// Where a decoder's layers live for the lifetime of the model.
///
/// The names mirror `gen_core::OffloadPolicy`, which is the enum a SceneWorks provider receives on
/// its `LoadSpec`. They are separate types on purpose: `mlx-gen` depends on `mlx-llm`, never the
/// reverse, so this crate cannot name that one. A provider maps
/// `OffloadPolicy::Sequential -> EncoderResidency::Sequential` at its own load seam.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EncoderResidency {
    /// Every layer built at load time and held — today's behaviour, and the default.
    #[default]
    Resident,
    /// Materialize each layer from the source file when it runs, then drop it.
    Sequential,
}

impl EncoderResidency {
    /// Whether this policy streams the layer stack.
    pub fn is_sequential(self) -> bool {
        matches!(self, Self::Sequential)
    }
}

/// What the streaming loader actually did — the loader-identity record.
///
/// Counters are cumulative across forwards and shared with every clone of the `Arc`, so a caller
/// can snapshot before a pass and diff after. A resident model has no `StreamObservation` at all
/// ([`CausalLm::stream_observation`] returns `None`), which is a stronger statement than a counter
/// reading zero: it says the streaming loader was never constructed, not that it ran and did
/// nothing.
///
/// [`CausalLm::stream_observation`]: crate::CausalLm::stream_observation
#[derive(Debug, Default)]
pub struct StreamObservation {
    layers_materialized: AtomicUsize,
    view_drains: AtomicUsize,
    passes: AtomicUsize,
    layers_per_pass: AtomicUsize,
}

impl StreamObservation {
    /// Layers built out of a reopened view and then dropped, over every forward so far.
    pub fn layers_materialized(&self) -> usize {
        self.layers_materialized.load(Ordering::Relaxed)
    }

    /// [`Weights::remove_accessed`] calls — one per materialized layer when the stream is correct.
    /// A drain count below the materialization count means layers were built without releasing the
    /// view's handle on their tensors, i.e. the stream is bounding nothing.
    pub fn view_drains(&self) -> usize {
        self.view_drains.load(Ordering::Relaxed)
    }

    /// Completed stack passes.
    pub fn passes(&self) -> usize {
        self.passes.load(Ordering::Relaxed)
    }

    /// Layers the most recent completed pass covered.
    ///
    /// The discriminator for a stream that ran but covered an empty range: `passes()` advances,
    /// `layers_materialized()` does not, and this reads 0.
    pub fn layers_in_last_pass(&self) -> usize {
        self.layers_per_pass.load(Ordering::Relaxed)
    }

    pub(crate) fn record_layer(&self) {
        self.layers_materialized.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_drain(&self) {
        self.view_drains.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_pass(&self, layers: usize) {
        self.passes.fetch_add(1, Ordering::Relaxed);
        self.layers_per_pass.store(layers, Ordering::Relaxed);
    }
}

/// A decoder layer stack materialized one layer at a time from a re-openable source.
pub(crate) struct SequentialStack {
    /// The single `.safetensors` file the layers are re-read from. Must stay readable for the
    /// model's lifetime — the stream reopens it on every forward.
    source: PathBuf,
    plan: LayerPlan,
    /// Replayed per materialized layer so a streamed layer is byte-identical to its resident twin.
    quant: Option<QuantSpec>,
    observation: Arc<StreamObservation>,
}

impl std::fmt::Debug for SequentialStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SequentialStack")
            .field("source", &self.source)
            .field(
                "layers_materialized",
                &self.observation.layers_materialized(),
            )
            .finish()
    }
}

impl SequentialStack {
    pub(crate) fn new(source: PathBuf, plan: LayerPlan, quant: Option<QuantSpec>) -> Self {
        Self {
            source,
            plan,
            quant,
            observation: Arc::new(StreamObservation::default()),
        }
    }

    pub(crate) fn observation(&self) -> &Arc<StreamObservation> {
        &self.observation
    }

    /// Open a fresh lazy view of the source. Called once per stack pass.
    pub(crate) fn open(&self) -> Result<Weights> {
        Weights::from_file(&self.source)
    }

    /// Materialize layer `i` out of `view`, run it over `h`, force the result, then release the
    /// layer and drain the view of exactly the tensors it read.
    ///
    /// The ordering is the contract, and every step of it is load-bearing:
    ///
    /// 1. build the layer (lazy graph nodes over `view`'s tensors);
    /// 2. forward (more lazy graph);
    /// 3. `eval` the carry — this is what actually reads the layer's bytes and, crucially, what
    ///    leaves `out` independent of the layer. Skip it and the drop below frees nothing, because
    ///    the unevaluated result still references the weights (hazard 1);
    /// 4. drop the layer;
    /// 5. `remove_accessed` — the view is still holding its own refcount on every tensor the layer
    ///    read, so without this the drop in (4) frees nothing either (hazard 2).
    ///
    /// Evaluating the carry also covers the K/V this layer pushed into `cache` and any shared K/V
    /// it stored: both are ancestors of `out` in the same graph, so one `eval` materializes them
    /// all. That matters — a cache entry left lazy would pin this layer's projections for the rest
    /// of the pass.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_layer(
        &self,
        view: &mut Weights,
        cfg: &ModelConfig,
        i: usize,
        h: &Array,
        ropes: &RopeTables,
        mask: AttnMask<'_>,
        cache: &mut dyn KvCache,
        shared: &mut SharedKv,
    ) -> Result<Array> {
        let out = {
            let layer = self.plan.load(view, cfg, self.quant, i)?;
            self.observation.record_layer();
            let out = layer.forward(h, ropes, mask, cache, i, shared)?;
            eval([&out])?;
            out
            // `layer` drops here, after its output is materialized.
        };
        view.remove_accessed();
        self.observation.record_drain();
        Ok(out)
    }
}
