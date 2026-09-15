# sc-16602 shared resampler boundary

This record fixes the compatibility boundary for the shared Candle audio resampler after the
F-210/F-211/F-242 hardening. It applies to `candle_audio::dsp::{resample, resample_range,
resample_mono_range}` and to the Chatterbox T3 and CLAP front ends that consume bounded windows.

## Exact numerical compatibility

`resample` remains a centered rational polyphase Kaiser-windowed-sinc FIR. Its output length is
`round(input_frames * dst_rate / src_rate)`, with integer ties rounded upward and every non-empty
clip producing at least one frame. The filter has at least 197 taps per phase; downsampling expands
support in proportion to the rate reduction. This change removes repeated bounds and normalization
work for interior frames, not the FIR dot-product cost.

The coefficient and output arithmetic are a compatibility surface:

- phase coefficients are designed and normalized in the same `f64` tap order as before;
- each phase's normalized divisor is accumulated in that same order;
- edge outputs retain the old checked tap walk, omitted-tap divisor, and nearest-source fallback;
- interior dot products retain the old per-channel, tap-order `f64` accumulation and division;
- final samples are converted to `f32` at the same point.

A test-only copy of the old loop compares `f32::to_bits` across leading and trailing boundaries,
interiors, mono/stereo/multichannel input, common and uncommon ratios, short clips, constants,
impulses, and deterministic random data. The asserted boundary is byte identity, not tolerance.

Phase kernels now occupy one contiguous allocation. Rates requiring more than
`RESAMPLE_MAX_PRECOMPUTED_COEFFICIENTS` are rejected with a typed audio error before output
allocation or evaluation. There is no on-demand phase fallback.

## Bounded global-output windows

`resample_range` selects a half-open frame range on the output timeline of the complete source
clip. It does not resample a sliced source. Therefore the range keeps the whole clip's rational
phase and leading/trailing FIR boundary context, and its result is bit-identical to slicing the
same range from `resample`.

The range must be ordered and contained by the complete resampled frame count. Empty source admits
only `0..0`. Equal-rate ranges are byte-identical frame slices. Multichannel `resample_range`
filters channels independently. `resample_mono_range` averages each interleaved source frame in
channel-order `f32` arithmetic before its FIR contribution, matching the former provider downmix
without allocating a complete mono clip.

Chatterbox requests `0..min(full_output_frames, 96_000)` at 16 kHz for its first-six-second T3
prompt. CLAP requests the centered `TARGET_SAMPLES` range at 48 kHz when long, or the complete
resampled result when short so existing repeat-padding remains unchanged. Both call sites downmix
only through the bounded API. Center rounding remains `(full_len - target) / 2`.

## Cached embedding and vector reproducibility

For rate pairs accepted before and after this change, whole-buffer PCM and bounded-window PCM are
bit-identical to the historical whole-buffer path. Chatterbox conditioning tokens, CLAP mel input,
and deterministic embeddings/vectors derived from those samples therefore require no cache
invalidation solely because of sc-16602.

Consumers should invalidate or refuse a cached artifact only when its preprocessing identity
already includes an input/rate pair that is newly rejected by the coefficient-table ceiling, or
when they change the selected output window, channel layout, rates, or upstream PCM. A rejection
does not produce a replacement embedding/vector and must not be cached as one.
