# SW-7150: completed MLX image execution reports

The bespoke MLX image cohort now publishes same-attempt execution observations through
`Generator::generate_with_execution_report`. Ordinary generation and progress callbacks suspend
capture. The shared streaming wrapper calls the completed boundary. Existing rich reports keep
their warnings. No provider stores a mutable last report.

The cohort includes Flux1/Flux2 base, edit and control; Flux3 variants; Krea base, turbo, edit,
multiphase and control; Qwen Image base, edit and control plus Qwen Image 2.1; SD3; SDXL; Chroma;
Ideogram; Kolors; Iris; Mage; Supra2; registered SenseNova image routes; Lens; PuLID; and Boogu
base, turbo and edit. Krea realtime belongs to SW-7151. SenseNova understanding and interleave
belong to SW-7156.

## Observations

- Shared FLOW calls observe the selected solver and consumed sigmas after success. Native loops
  observe their actual coefficients/times, dtype and output seed. Krea scopes multiple stages to
  the correct output. SDXL/Kolors scope native global initial noise and selected step RNG schemes.
- SDXL native identities include the exact sigma source table, compute dtype and consumed time
  pairs. PiD truncation keeps its actual nonzero endpoint. This avoids pretending host-interpolated
  sigmas equal the native f16 interpolation. LCM/Lightning/TCD/FlowMatch expose actual scalar
  coefficients and step/input dtypes; Kolors exposes actual leading times/sigmas and start scale.
- Control strengths for Flux1/Flux2/Qwen/Krea, Krea control guidance, Iris CFG scale and its actual
  per-step activation, and Krea's shared decoder seam tiling/override are runtime observations.
- Flux2 caption enhancement retains the real `generate_speculative` decode report and scopes the
  actual `CaptionSampler` seed/settings with SW-7156's token collector. Flux3 retains the
  upsampler's `TextLlm::generate_with_execution_report` identity through the portable prompt-stage
  driver. Each image report carries the nested identity and actual enhancement outcome/effective
  caption digest. Missing auxiliary execution or failed Flux2 enhancement fails closed with a
  typed unavailable reason; absence/unsupported-variant bypass remains distinguishable.
- PiD auxiliary decoder observations come from SW-7149. Explicit unobserved noise overrides and
  missing producer seeds fail closed. The shared collector conservatively declares MLX replay
  nondeterministic; this work does not attest cross-device or bitwise replay.

## Validation and dependency order

This isolated branch intentionally does not import unmerged sibling commits. Native build, tests
and strict lint require the reviewed SW-7149 image collector (`c546389f36d...`) and SW-7156 token
collector/provider observations (`2591b2479fdb...`) to reach the integration branch first. The
coordinator owns that integration and its exact-head validation. SenseNova `t2i.rs` observations
from SW-7150 and SW-7156 are additive and both must survive integration.

Portable evidence on this branch:

- `cargo test -p sceneworks-gen-core-flux3 --test flux3_pup`: 20 passed, including a scripted
  completed provider proving that typed auxiliary absence belongs to its own attempt across
  streaming cancellation and retry. No weights or learned compute.
- `cargo clippy --locked -p sceneworks-gen-core-flux3 --all-targets -- -D warnings`: passed.
- Formatting, workspace policy, model-inference source/workflow policy and clock-assertion
  ratchet passed. The policy checker used isolated `uv --with pyyaml` because system Python lacked
  its YAML dependency. The ratchet remains at 288; no baseline changed.

Prepared native metadata-only regressions (no arrays, weights or learned compute):

- `execution_policy_tests` in Flux1/Flux2 control, Qwen control, Krea, Iris, Flux2 enhancement and
  Flux3 enhancement: distinguish strengths, CFG, native/tiled/override decode, caption outcomes,
  unavailable auxiliary identity and a fresh retry.
- SDXL `execution_schedule_preserves_truncated_endpoint_and_sigma_source` checks the actual
  native sampler's scalar state; `execution_rng_identifies_actual_default_acceleration_and_cfgpp_fallbacks`
  exercises the production resolver; `externally_supplied_noise_cannot_inherit_a_previous_native_output_seed`
  exercises completed capture and missing producer refusal.

After reviewed dependency integration, run these exact test filters and strict all-targets Clippy
for all touched MLX provider crates. Set `CARGO_BUILD_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2` for native
builds. Linux/Windows/macOS/iOS contract, bundle and platform lanes remain required. Any learned
model proof belongs to a separately dispatched evidence workflow; no such proof ran locally.
