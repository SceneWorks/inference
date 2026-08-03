# sc-16601 Stable Audio 3 provider polish

This record closes the F-208/F-209/F-239/F-243/F-244 follow-up on all six Candle Stable Audio 3
providers. It changes cold-load work, bounded reference preparation, and request validation without
changing provider ids, checkpoint pins, sampler numerics, or reference-audio output semantics.

## Adapter planning happens once

`load_variant` parses adapter metadata and tensors on CPU, matches every module against the pinned
checkpoint header, and retains the resulting `AdapterPlan` in `StableAudio3Generator`. A malformed
or key-mismatched adapter therefore still fails from `load_variant`, before first generation.

Lazy pipeline construction calls `AdapterPlan::to_device`; it copies the retained factors to the
selected device and does not retain or revisit `AdapterSpec` paths. The weight-free regression
removes the adapter file after planning and requires transfer to preserve the plan and values. The
real-weight production-route regression removes the file after `load_variant` and requires first
generation to succeed. Those tests fail if parsing, tensor loading, or plan construction happens a
second time.

This does **not** correct or optimize Jacobi SVD. SVD was never duplicated by the two planning
calls: `-xs` bases are computed only while an adapter is folded into a base checkpoint weight.

## Reference preparation is a bounded global-output window

The public `prepare_reference_pcm` remains defensive and validates source rate, channel shape, and
sample finiteness. The generator validates the request once and then enters a crate-private
validated synthesis path, so its source samples are not scanned for finiteness again.

Preparation asks the sc-16602 shared resampler for
`0..min(full_resampled_frames, target_frames)` on the complete clip's global output timeline. It
does not resample a sliced input, so rational phase, leading/trailing FIR context, and every retained
sample remain bit-identical to slicing the historical whole-buffer result. Only retained frames are
channel-conformed (mono duplicated, stereo retained, channels above two reduced to their first two)
and written into the fixed stereo output; the remainder is zero padding. There is no complete
resampled output and no target-sized padded buffer at the caller's channel count.

Why the bound matters: one hour of 48 kHz stereo resampled to 44.1 kHz produces about 158.76 million
frames. At the current 215-tap ratio that is about 68 billion multiply-accumulates and
1,270,080,000 bytes (about 1.27 GB) of interleaved F32 output before the former trim. Work-counter
coverage now requires the production preparation helper to evaluate only the requested output
window, while byte comparisons cover mono, stereo, more-than-two channels, short clips, trimming,
and right padding.

## Validation and pinned evidence

`guidance_eta` is accepted only when finite and in the inclusive interval `0..=1`, in addition to
requiring `guidance_method=apg`. Both endpoints pass; NaN, both infinities, and values on either side
of the interval fail in weight-free provider validation.

The six-file `stable_audio_3_medium` artifact pin set is exactly **10,443,755,936 bytes**:

- root config and weights: 10,360 + 9,222,116,660 bytes;
- T5Gemma config, weights, and tokenizers: 2,540 + 1,183,022,944 + 34,362,429 + 4,241,003 bytes.

The pin-set test asserts that exact sum. Earlier `10.4 GB` prose is descriptive rounding only.
Likewise, the 10–16 minute CPU timing for a 380-second medium render is an **estimate inferred**
from small measured CPU runs and measured Metal throughput; it is not a measured 380-second CPU
render.

Finally, T5Gemma's text-side dtype policy is independent of reference preparation. Text compute and
conditioned embeddings remain F32 on every backend. The raw encoder output remains F32 on CPU, while
Metal and CUDA apply one BF16 rounding at the raw-output boundary before returning to F32
conditioning. This is a backend dtype policy, not a reference-padding behavior.
