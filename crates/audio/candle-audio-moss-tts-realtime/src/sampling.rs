//! Deterministic RVQ-token sampling for the MOSS-TTS-Realtime AR loop (sc-13392).
//!
//! The reference `streaming_mossttsrealtime.py` samples each codebook token — it is **not** a greedy
//! model. Greedy (argmax) decoding collapses within a few frames into a repeating loop whose codec
//! decode is silent; the model card ships `do_sample=True` with `temperature=0.8`, `top_k=30`,
//! `top_p=0.6`, and a per-codebook cross-frame `repetition_penalty=1.1` over a 50-frame window. This
//! module reproduces that pipeline faithfully, but with a **seeded** PRNG so the gen-core
//! reproducibility law holds: the same `seed` yields byte-identical frames (and thus audio) run to
//! run, so `generate` and `generate_streaming` agree.
//!
//! Order (matching the reference `generate_local_transformer` → `sample_token`): repetition penalty
//! on the raw logits → divide by temperature → top-k mask → top-p (nucleus) mask → softmax →
//! multinomial draw. Parity with PyTorch's audio is not required (a different RNG) — a valid,
//! deterministic speech distribution is.
//!
//! ## RNG-parity decision (sc-13433)
//!
//! sc-13392 flagged that this sampler's token *sequence* diverges from a reference PyTorch run
//! because the draw is not bit-parity with Torch's Philox `multinomial`, and asked whether a
//! Philox-parity RNG (option a) would restore reference-matching, prompt-faithful output. The
//! measured answer is **no — Philox parity is not worth it**, on two independent grounds:
//!
//! 1. **Necessary but not sufficient.** Reference-matching a Torch run needs bit-identical *logits*,
//!    not just an identical RNG stream. This port runs the AR head in F32 on CPU (candle); the
//!    reference runs Torch on a different dtype/backend, so the logits differ by ~1e-3+ (the standard
//!    cross-framework matmul gap). With ~1027-way codebooks under top-k 30 / top-p 0.6 there are many
//!    near-ties, so the sampled sequence diverges at the first close race regardless of the RNG. The
//!    sufficient condition (bit-identical logits) is unattainable across frameworks.
//! 2. **Not the fidelity lever.** An ASR round-trip sweep (prompt → TTS → `whisper_base` → CER, see
//!    the crate's `moss_tts_realtime_asr_roundtrip_fidelity` conformance gate) showed prompt-following
//!    is limited by **spurious early audio-EOS** (a full sentence collapsing to a sub-second
//!    fragment), *not* by RNG divergence. Greedy (temperature 0) collapses to silence; temperatures
//!    below 0.8 stop *earlier*, not better; temperature 0.8 (the reference default) is kept. sc-13433
//!    first mitigated the early-EOS symptom with a minimum-length floor, but sc-13570 traced the root
//!    cause to the port's prompt conditioning (a fabricated chat turn + `text_pad`-only generation)
//!    and restored the reference delay-pattern ([`crate::decode::build_prompt_frames`]); with correct
//!    conditioning the model reaches its natural EOS on its own, so the floor is off by default
//!    ([`crate::decode::DEFAULT_MIN_EOS_FRAMES`] = 0) and prompts that used to collapse (e.g. "The
//!    weather is very nice this afternoon." 0.95 → 0.00) now render from conditioning alone.
//!
//! The only RNG property that matters here is **reproducibility** (same seed ⇒ byte-identical
//! frames), which the shared seeded splitmix64 ([`Rng`]) already guarantees and the determinism gate enforces.

// The pipeline itself (PRNG, penalty → temperature → top-k → top-p → multinomial) is shared with
// the sibling MOSS crate through the audio commons (sc-27272); this module keeps the model's own
// reference defaults.
pub use candle_audio::sampling::{sample, Rng, SamplingParams};

/// Reference decoding defaults (`MossTTSRealtimeStreamingSession`).
pub const TEMPERATURE: f32 = 0.8;
pub const TOP_P: f32 = 0.6;
pub const TOP_K: usize = 30;
pub const REPETITION_PENALTY: f32 = 1.1;
pub const REPETITION_WINDOW: usize = 50;

/// The reference streaming-session defaults as one [`SamplingParams`] value —
/// what the AR loop samples with.
pub const DEFAULT_PARAMS: SamplingParams = SamplingParams {
    temperature: TEMPERATURE,
    top_p: TOP_P,
    top_k: TOP_K,
    repetition_penalty: REPETITION_PENALTY,
    repetition_window: REPETITION_WINDOW,
};
