# SW-7156: completed text and understanding execution reports

The additive report APIs return an output and its attempt-local execution identity together.
They do not read a mutable provider-wide last-report field. The existing generation APIs and
stream events remain source compatible.

## Production routes

- Candle LlamaProvider (including its VLM routes), LLaVA, StarVector 1B and 8B implement
  `TextLlm::generate_with_execution_report`.
- MLX LlamaProvider (including its VLM routes), JoyCaption, StarVector 1B and 8B implement
  the same method.
- All four StarVector providers implement `generate_svg_with_execution_report` directly,
  preserving their bounded SVG output and streaming behavior.
- Candle `SenseNovaUnderstanding` and MLX `SenseNova` expose
  `run_vqa_with_execution_report` and `run_interleave_with_execution_report`. These take the
  same arguments as the progress-capable existing methods and return `(output, ExecutionIdentity)`.

The inherited text report batch method invokes each request independently and returns its own
identity. It does not claim continuous batching. Existing `generate_batch` behavior is unchanged.

## Evidence sources

`GenerationConfig::execution_seed` observes the sampler controls and effective seed at the
native decoder's RNG construction. Omitted seeds retain the decoder's existing default resolver;
reports carry its actual resolved value. The completion carries measured decode/cache choices,
including speculative fallbacks, prefix reuse, graph/attention selection and sampler path.
Floating point sampler controls retain IEEE-754 bits; the generated-only presence-penalty boundary is observed when the sampler resets it after prefill. Greedy token decoding reports no RNG draws. Decode token budgets and stop-token IDs
come from the consumed configuration. SVG byte/time/root limits come from the bounded stream.

SenseNova's direct runtime retains each completed internal decode report. Interleave records
its actual ordered text/image streams and every exact denoise schedule digest. Image draws use
the seed supplied to the native noise producer. MLX explicit noise is hashed from the effective
f32 bytes instead of being mislabeled with the options seed; this diagnostic path adds a host
readback only when report capture is active. Effective memory/attention and flow policies are
recorded after request-scope memory configuration. One output unit is the complete interleaved
document; `text_segment_N` and `image_N` draws refer to document index zero, while output policy
contains the actual image count and shapes.

Scopes are thread-bound and nestable. Errors, cancellation and panic drop their observations.
Streaming/progress callbacks suspend the enclosing observation so a reentrant legacy decode
cannot contaminate it; a nested report-aware call can capture its own attempt.

## Explicit limits

A partial or canceled result cannot publish a resolved identity. Missing measured decode,
missing/ambiguous RNG mapping, or an inconsistent document-to-stream mapping produces a typed
Unavailable reason. A zero-budget direct route that never decodes cannot invent a completed
decode report. These reasons replace the compatibility-only provider default.

Pre-execution plan methods explicitly report that runtime decode choices are unavailable before
execution. They do not promise strict pre-admission replay for a path whose numerical choices
are only observed after decoding. Completed token and SenseNova document reports retain the
observed settings and draws but classify replay as non-deterministic until a route, model, device
and precision scope has a bitwise replay attestation. Live prefix reuse, adaptive speculation,
and SVG wall-time bounds name their additional dynamic dependency in the refusal reason. No
learned model execution is part of ordinary tests.

## Worker integration

SceneWorks `vector_jobs.rs` must use `generate_svg_with_execution_report` and retain both fields
through artifact publication. `sensenova_jobs.rs` must consume the tuple returned by the new direct
methods. Text consumers use `generate_with_execution_report`. Consumers must preserve typed
Unavailable and non-deterministic classifications, and must never replace them with submitted
request values. The SceneWorks worker changes belong to SW-7148.
