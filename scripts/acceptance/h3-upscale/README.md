# H3 x2 offline prototype (sc-25328)

This is an experiment, absent from product capability/worker registration. Ordinary
H3 generation still enforces its existing frame/duration lattice. The executable
is native Candle/CUDA; Python only prepares bounded inputs and independent upstream
reference fixtures. No downloaded weights or source video belongs in this repo.

The completed bounded experiment concludes **NO-GO**. See [READOUT.md](READOUT.md)
and [readout.json](readout.json) for measured timings, hashes, reference receipts
and preservation findings. Production slices remain unstarted under E11.

Recipe identity: ComfyUI-H3-Video-Upsampler commit
`36cb612ec3df30094eeb4fec66f1528925ebba73`. That pack does not pin its ComfyUI
dependency; this experiment additionally pins ComfyUI
`7a5dad695fe1cae25efcb2550530fb20ef68da3d` (the last preceding commit). Installed
LoRA `sandpies/ComfyUI-H3-Video-Upsampler@9337fe00d0753d249aba6181b81d3b6a62bccb52`,
`lora/h3upscale_v2_x2.safetensors`, SHA256
`ff671fa94e377fd277319a89c366a67f49b97ae6528c8ac3b0c5e0a3b254c56c`.
Installed latent upscaler `LBH-123-AI/Minimax_h3_latent_Upscaler@3f941d5d182014dd5c0a5e16330420ee2d4aa0c6`,
`minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`,
SHA256 `4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6`.

Build `cargo build --locked --release -p candle-gen-minimax-h3 --features cuda
--example h3_upscale_prototype`. On Windows initialize a CUDA-compatible MSVC
toolset first. The measured setup uses MSVC 14.44 with CUDA 12.9; newer 14.51 is
incompatible and no compiler/STL override is needed or accepted as evidence.

Run `run.py --binary <exe> --root <immutable H3 snapshot> --upscaler <installed
file> --lora <installed file> --noise <seed444 safetensors> --manifest <input
provenance json> --out <external evidence directory> --gpu-uuid <assigned UUID>`.
It serializes guided, guide-disabled and denoise-zero controls on three
explicitly normalized 39-frame 512x288@24 cases. Input provenance distinguishes
original sources from these test fixtures. Only source AAC is muxed; frozen
internal clean-zero audio is never decoded or delivered. Do not use an occupied
GPU. The UUID maps to ordinal zero inside the native child process.

The learned network runs FP16. VAE and unaccelerated Ref2VA transformer run BF16.
The exact latent chain is posterior raw -> VAE normalization -> learned network
statistics normalization -> network -> reverse network statistics. Its output is
already normalized VAE/DiT state. Applying VAE normalization after the learned
network, or feeding posterior raw state to it, corrupts the result; a composed
boundary golden rejects both mistakes.
Guide pixels are source latent decode -> clamped antialiased bicubic target ->
clamped antialiased bicubic target/2 -> source-size VAE re-encode. Guide rows
cover all 12 latent times and every second target patch. The target canvas is
64px aligned; the half-size 512x288 source is 32px aligned. The default denoise .1 selects the final nonzero simple-grid
sigma .5714286 and a single Euler update. Denoise zero bypasses refinement.
All noise comes from the committed export recipe's Torch CPU seed444 fixtures;
there is one full-window video field, and guide augmentation restarts seed444.
No Turbo or other LoRA is accepted.

`export_reference.py` executes AST-extracted actual upstream definitions to
produce the committed tiny nontrivial graph/interpolation/guide layout/pixel
fixtures. It records source/fixture hashes and tolerances. Supply `--upstream
<tiled_upsampler.py> --comfy-model <pinned comfy model.py> --guide-source
<upscale_guide.py> --comfy-sampling <pinned model_sampling.py>
--comfy-samplers <pinned samplers.py> --comfy-vae <pinned minimax/vae.py>
--out <fixture safetensors>
--runtime-noise <external noise
file>`. The native integration tests compare the actual independent goldens;
wrong full-grid/stretch guides and raw-sigma denoise are discriminated. Tiny
fixtures are algorithm evidence, never quality or real-weight performance proof.
`compare_real_upscaler.py` additionally compares captured actual source/upscaled
latents with the pinned FP16 network under declared tolerances. Default CPU
execution can be slow; `--device cuda` is permitted as serial reference tooling
on the assigned GPU outside the product. Its temporal-tap and intermediate
resize rounding deviations must pass before GO.

Every run writes per-stage synchronized timings, sampled peak process RSS and
assigned-device memory, source/output hashes, probes and a machine-readable
report. Sampling is 200ms, so peaks are sampled measurements, not exact allocation
maxima. A report stays INCOMPLETE until all three cases, controls, installed
SeedVR2 comparison, numerical receipts and the visual rubric are present. GO
requires useful visible improvement over bicubic on two cases including non-H3,
no material identity/motion/timing regression, reference fidelity and a measured
feasible path on named hardware. An executed negative finding is NO-GO; a missing
run/dependency remains incomplete and must never masquerade as measured NO-GO.

The current fixed-case executable deliberately bounds state to one short window;
arbitrary frame-rate, odd dimensions, long-window/tile policy and cancellation
belong to the production slices gated by this experiment. No general support,
performance claim or production capability follows from the fixed-case success.

See THIRD_PARTY_NOTICES.md and UPSTREAM_LICENSE for original code notices.
Learned weights are Apache-2.0. The LoRA metadata/dataset declare `license=other`
and the notice defers to H3 base/dataset terms for redistribution/commercial use.
The user confirmed written authorization obtained through the online form for
the requested use. This is user-provided authorization evidence; it is not an
independent audit or a permissive public license grant. No weights are
redistributed. Existing H3 notices and controls still apply.

Build the owning-crate comparator with `cargo build --locked --release -p
candle-gen-seedvr2 --features cuda --example h3_upscale_baseline`. After the H3
campaign completes, run `baseline.py --binary <seedvr2 exe> --root <installed
SeedVR2 snapshot> --evidence <same directory> --gpu-uuid <assigned UUID>`. It
records installed weight hashes, per-stage timing and sampled memory, and copies
the fixture soundtrack. Never overlap reference or native GPU workloads.

Run `review.py --evidence <same directory>` after the comparator. It verifies
every decoded fixture soundtrack, writes beginning/middle/tail contact sheets and
full-size bicubic/guided/SeedVR2 pairs, and records all-frame RGB/change departures.
Those metrics are descriptive comparisons with bicubic, not automatic quality
scores or high-resolution truth. A reviewer applies the preservation rubric and
records the final GO/NO-GO decision; the runner itself never grants a GO.

Delivery checks: `python -m unittest discover -s scripts/acceptance/h3-upscale
-p test_run.py -v` verifies silent, early-ended and full soundtrack fixtures,
all39 frame timestamps and decoded soundtrack hashes. The CUDA-only tiny BF16
refinement fixture is ignored by default and must be invoked explicitly on the
assigned UUID; it distinguishes a missing BF16-to-F32 Euler boundary cast.
