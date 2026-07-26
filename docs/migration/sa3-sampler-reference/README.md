# Stable Audio 3 sampler reference

This directory locks the corrected `sc-14542` sampler math to frozen upstream commit
`124e8a799f57a1f665495ecb72e547d0a62867f1`.

`sampler.json` is a weight-free independent oracle for Euler, corrected batched RK4, rectified-flow
DPM++, Pingpong, shared/per-example LogSNR schedules, corrected partial init strength, and the eager
terminal Pingpong noise draw. `manifest.json` locks the seven upstream source files, the artifact
payload, the authoritative P0 manifest, all six P0 artifact payloads, and each connected checkpoint
revision/config hash. Generate and verify it with:

```sh
python3 scripts/reference/sa3_sampler_reference.py \
  --generate --upstream /private/tmp/sa3-upstream-review
python3 scripts/reference/sa3_sampler_reference.py
```

The six pre-existing P0 artifacts under `docs/migration/sa3-reference/` remain the real-weight,
eight-step Pingpong oracle. `tests/sampler_oracle.rs` compares every frozen `x`, denoised value,
sigma, and final latent using reconstructed explicit draws. Runtime seeded reproducibility is a
separate test; no cross-backend byte identity is claimed.

Deliberate corrections from frozen Python are:

- per-example RK4 iterates over the schedule time axis and retains the scalar formula;
- partial init strength shifts a normalized schedule before scaling the full trajectory;
- schedule shape is typed, so `batch == steps + 1` is never ambiguous.
- mixed duration batches adapt tensor geometry from the maximum present duration while a missing
  duration forces only the distribution-shift schedule to its global fallback;
- solver schedules are strictly decreasing to one terminal zero; strength zero is accepted only by
  the separate initialized no-call path;
- the live callback observes the frozen pre-update state once per solver step and cancels by
  propagating its error before any update or Pingpong draw;
- CFG interval decisions use typed host timestep values for every model evaluation, including all
  four RK4 stages. A heterogeneous per-example decision fails closed because the DiT requires one
  mandatory `2B` CFG forward.

Sampling remains unregistered. k/v samplers are unreachable, `rescale_cfg` is not advertised,
`Guidance.scale_phi` owns real CFG rescale, and the already-reviewed DiT owns batched CFG/APG.
The advanced DiT entry validates every guidance scalar and rejects the unreachable V objective.
The sampler forwards padding masks to the DiT but does not duplicate its frozen V-zero boundary.

Request-local seeded noise is generated on the host. One stream owns the initial noise and every
Pingpong step draw, including the eager terminal draw, so concurrent requests cannot perturb draw
ordering. Each draw is a full-latent host allocation and, on an accelerator, a real host-to-device
transfer; CPU execution has no H2D transfer. The resource estimator reports solver-only cost after
the initial latent exists: an eight-step Pingpong solve performs eight DiT calls and eight draws,
with eight transfers only on an accelerator. A full text-to-audio request adds one initial
full-latent draw and an additional transfer only on an accelerator. Euler and RF DPM++ perform
eight calls; RK4 performs 32. CUDA numerical runtime remains owned by `sc-14552` when CUDA hardware
is absent; ordinary CUDA compilation is not presented as numerical proof.

## Real-weight numerical evidence

The final source was run against the immutable revisions recorded in `snapshot-files.json`.
All-six P0 trajectory comparison passed on CPU in 90.32 seconds and on Metal in 4.69 seconds.
Every step checks `x`, denoised, and sigma; the final latent is checked separately.

| Backend | Minimum cosine | Largest small-model absolute error | Largest medium-model absolute error |
| --- | ---: | ---: | ---: |
| CPU | 0.999999998 | 0.214843750 | 0.000073552 |
| Metal | 0.999999999 | 0.214843750 | 0.000074863 |

The real one-step CFG/APG fixture covers post-trained and base DiTs with vanilla CFG, APG, and
blended/rescaled guidance. All twelve CPU/Metal comparisons had cosine `1.0`. Worst absolute error
was `0.003721237` on CPU and `0.003684998` on Metal.

The reference verifiers independently lock:

- the frozen upstream commit and source payloads;
- the sampler JSON and guidance safetensors payload digests;
- the P0 manifest digest and every connected P0 artifact;
- the snapshot-lock digest, repository, revision, config/model hashes, and diffusion objective;
- the exact runtime, input schedule, padding, negative-conditioning contract, tensor inventory,
  shapes, and dtypes.

Mutation tests include coupled artifact/manifest edits and coupled snapshot-lock/guidance-manifest
model-hash edits.

## Fresh-process resource evidence

Each row ran in a fresh process. Elapsed time is measured only after a scalar checksum synchronizes
queued accelerator work. RSS is the process peak reported by `/usr/bin/time -l`. `Host-created
bytes (accelerator H2D)` is the sampler-only host-materialized byte count: one nine-element typed
schedule plus eight full-latent Pingpong draws, or only the schedule for Euler. These bytes are
actual H2D traffic on Metal/CUDA; CPU actual H2D is zero. A full text-to-audio request adds one
initial full-latent host allocation and, on an accelerator, its transfer. Checksums are
synchronization/finite-output witnesses, not a cross-backend equality claim.

### CPU

| Model/default | Seconds | Latent | Calls/draws | Host-created bytes (accelerator H2D) | Synchronized seconds | Peak RSS bytes | Checksum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small post / Pingpong | 0.25 | 68 | 8 / 8 | 557,092 | 12.008731 | 3,684,040,704 | 6,661,003.5 |
| small post / Pingpong | 30 | 388 | 8 / 8 | 3,178,532 | 32.420715 | 3,690,315,776 | -3,685.3662 |
| small post / Pingpong | 120 (max) | 1,292 | 8 / 8 | 10,584,100 | 84.614448 | 3,684,040,704 | -5,468.7471 |
| small base / Euler | 0.25 | 68 | 8 / 0 | 36 | 10.846712 | 3,684,057,088 | 475,598 |
| small base / Euler | 30 | 388 | 8 / 0 | 36 | 27.424626 | 3,684,139,008 | 6,714.7715 |
| small base / Euler | 120 (max) | 1,300 | 8 / 0 | 36 | 79.863068 | 3,684,155,392 | 11,636.5996 |
| medium post / Pingpong | 0.25 | 68 | 8 / 8 | 557,092 | 33.445775 | 11,638,046,720 | -753.0635 |
| medium post / Pingpong | 30 | 388 | 8 / 8 | 3,178,532 | 88.532797 | 11,638,226,944 | -4,729.6816 |
| medium post / Pingpong | 380.435737 (max) | 4,096 | 8 / 8 | 33,554,468 | 964.959421 | 11,638,243,328 | -29,249.0508 |
| medium base / Euler | 0.25 | 68 | 8 / 0 | 36 | 33.750183 | 11,638,177,792 | -838.1959 |
| medium base / Euler | 30 | 388 | 8 / 0 | 36 | 97.339883 | 11,638,145,024 | -4,658.6338 |
| medium base / Euler | 380.435737 (max) | 4,096 | 8 / 0 | 36 | 1,064.076907 | 11,638,259,712 | -47,460.0859 |

### Metal

| Model/default | Seconds | Latent | Calls/draws | Host-created bytes (accelerator H2D) | Synchronized seconds | Peak RSS bytes | Checksum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small post / Pingpong | 0.25 | 68 | 8 / 8 | 557,092 | 0.358470 | 3,696,918,528 | 6,660,978.5 |
| small post / Pingpong | 30 | 388 | 8 / 8 | 3,178,532 | 0.781208 | 3,696,836,608 | -3,685.3682 |
| small post / Pingpong | 120 (max) | 1,292 | 8 / 8 | 10,584,100 | 2.229690 | 3,696,902,144 | -1,900.9646 |
| small base / Euler | 0.25 | 68 | 8 / 0 | 36 | 0.349776 | 3,696,902,144 | 475,602.0625 |
| small base / Euler | 30 | 388 | 8 / 0 | 36 | 0.780774 | 3,696,934,912 | 6,714.7266 |
| small base / Euler | 120 (max) | 1,300 | 8 / 0 | 36 | 2.248285 | 3,696,869,376 | 16,306.6221 |
| medium post / Pingpong | 0.25 | 68 | 8 / 8 | 557,092 | 0.910261 | 11,650,400,256 | -753.0646 |
| medium post / Pingpong | 30 | 388 | 8 / 8 | 3,178,532 | 2.232162 | 11,650,334,720 | -4,729.6807 |
| medium post / Pingpong | 380.435737 (max) | 4,096 | 8 / 8 | 33,554,468 | 25.394811 | 11,650,236,416 | -29,248.9453 |
| medium base / Euler | 0.25 | 68 | 8 / 0 | 36 | 0.910013 | 11,650,416,640 | -838.1956 |
| medium base / Euler | 30 | 388 | 8 / 0 | 36 | 2.230458 | 11,650,367,488 | -4,658.6348 |
| medium base / Euler | 380.435737 (max) | 4,096 | 8 / 0 | 36 | 25.451844 | 11,650,318,336 | -47,458.7031 |
