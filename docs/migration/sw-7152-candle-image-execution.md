# SW-7152: Candle image execution reports

The registered image providers override `Generator::generate_with_execution_report`.
The existing normal execution body still owns validation, execution, and output. The
report observes that body in a fresh thread-local scope and binds observations to
its returned output only after the final cancellation check. The trait's streaming
report method delegates to the same override. Failed and cancelled calls return no
completed report. Nested calls, unwinding, and concurrent threads cannot reuse an
older scope; consumer progress callbacks execute outside their provider's scope.

`candle-gen/src/execution.rs` implements this scope. It has no mutable last-report
slot and does not reconstruct choices from `GenerationRequest`. It records the
selected sampler at the factory/loop which uses it, hashes the consumed numeric
schedule's IEEE bits, records actual seed constructors and per-output seed order,
and observes loaded precision/quantization, resolved controls, attention budgets,
block windows and decoder policies at their owning execution seams. The output
provides the actual dimensions, count, RGB/RGBA format and channels. The consuming
application must also bind the exact inference revision and model/component bytes.
`candle-execution-v1` names this numerical-observation implementation.

Missing sampler, random-stream algorithm, quality, runtime, output, or output-binding facts
produce `Unavailable(candle_execution_observation_incomplete)`, with the missing
facts listed. A new branch cannot inherit observations from a previous attempt.
The ordinary compatibility default is not used by these registered providers.

## Registered route inventory

Paths in this table are relative to `crates/media/candle-gen/`.

| Family | Registered entry owner | Actual integration observation |
| --- | --- | --- |
| SDXL | `candle-gen-sdxl/src/lib.rs`, `control_provider.rs` | Shared curated driver (omitted default DDIM; unknown curated names fall back to Euler), native Lightning and conditioned ancestral loops |
| SD3 | `candle-gen-sd3/src/lib.rs` | Shared curated/flow driver; normal and staged per-image seeds |
| FLUX | `candle-gen-flux/src/lib.rs` | Shared flow driver; resident/sequential execution |
| Krea | `candle-gen-krea/src/lib.rs`, `control_registry.rs` | Shared flow driver, including phased, edit and control calls |
| Z-Image | `candle-gen-z-image/src/lib.rs`, `base.rs` | Shared flow driver; native Turbo control/edit loops observe their f64 schedule and edit tail |
| Qwen | `candle-gen-qwen-image/src/lib.rs`, `candle-gen-qwen-image-2-1/src/lib.rs` | Shared flow driver, including saved/native schedule selection and Qwen 2.1 RGB/RGBA output |
| Sana | `candle-gen-sana/src/model.rs` | Base flow and Sprint SCM drivers; normal/staged seeds |
| Chroma | `candle-gen-chroma/src/lib.rs` | Shared flow driver; normal/staged per-image seeds |
| Lens | `candle-gen-lens/src/lib.rs` | Shared flow driver |
| Mage | `candle-gen-mage/src/lib.rs` | Native Euler ladder in generation/edit; actual MT19937 masked seed and PCG watermark key |
| Boogu | `candle-gen-boogu/src/lib.rs` | Shared flow driver or native Turbo DMD tail and salted renoise streams |
| Anima | `candle-gen-anima/src/lib.rs` | Shared flow driver |
| Ideogram | `candle-gen-ideogram/src/lib.rs` | Native f64 logit-normal/Euler schedule, resolution and edit-selected tail |
| Iris | `candle-gen-iris/src/model.rs` | Native f64 FlowDPM plan, coefficients/prediction, per-prompt and repetition seeds |
| Supra2 | `candle-gen-supra2/src/model.rs` | Native Euler, SplitMix64/Box-Muller per-image seeds; explicit noise uses per-image hashes |
| SenseNova | `candle-gen-sensenova/src/lib.rs` | Native text-to-image and reference-image Euler timesteps and resolved T2I options |
| Kolors | `candle-gen-kolors/src/lib.rs` | Shared curated driver or native Euler schedule/timesteps |

Zero-step source roundtrips state that no denoiser schedule applies. They do not
invent a one-step solve. PiD is an additional observed decoder stage: its actual
SDE/ODE schedule, sigma, scale, selected tile edge/overlap and initial-plus-epsilon
continuing `StdRng` stream are recorded. Candle PiD draws full-image noise even
when it tiles the neural forward; it does not draw independent noise per tile.
Staged decode explicitly restores each output index before recording decoder draws.
Boogu's existing staged PiD path independently resolves a second default seed when
no request seed is supplied. Its report preserves both actual streams and marks
exact replay `NonDeterministic`: the current single seed input cannot reproduce
both independently selected values. Supplying a seed avoids that limitation.

## Standalone production entry points

SceneWorks' `crates/sceneworks-worker/src/image_jobs/` has direct Candle callers:
`sdxl_ipadapter.rs`, `sdxl_edit_candle.rs` (including masked edits),
`flux_ipadapter.rs`, `flux1_control_candle.rs`, `qwen_control.rs`,
`qwen_edit_candle.rs` (memory context), `kolors_ipadapter.rs` (memory context),
`kolors_control.rs` (memory context), `zimage_control.rs`,
`krea_control_candle.rs`, `zimage_edit_candle.rs`, and `detail.rs`.

Their engine types expose additive `*_with_execution_report` methods, forwarding
the same arguments through the original execution body. The FLUX injector and
memory-context variants have matching wrappers. The wrapper checks cancellation
before publishing and preserves the actual selected runtime observations.
SceneWorks must select these additive methods when it integrates the reports.

The exported Anima and Mage image pipelines and both Sana pipelines also expose
completed-report methods. Sana's callback-bearing form is
`generate_with_progress_and_execution_report`. Mage covers ordinary edits and
`edit_from_reference_tokens`. Supra2's
`generate_with_noise_and_execution_report` retains its original latents, images,
prompt diagnostics and stored-unconditional flag in `CompletedSupra2Output`.

Krea's `pipeline::render*`, Ideogram's `pipeline::render*`, Iris's `pipeline::denoise`
and SenseNova's `T2iModel` Tensor helpers are called by the registered generation
bodies, rather than by separate SceneWorks image-job entry points. Their executed
operations contribute to the enclosing report. SenseNova's separate understanding
and interleaved-document API belongs to SW-7156.

## Validation boundaries

The new attempt tests use raw tensor arithmetic and metadata only: actual shared
solver fallback, exact schedule digests, omitted and wrapping seeds, streaming,
missing facts, nested/concurrent scopes, panic/error retry, terminal cancellation,
staged decoder output binding, explicit-noise hashes, and compute dtype transitions.
Existing SDXL default-routing and FLUX control-route source tests remain applicable.
No learned inference or training was added to normal CI. All affected targets must
compile and pass strict lint in their platform lanes. These checks do not substitute
for separately dispatched model/device evidence or application replay acceptance.
