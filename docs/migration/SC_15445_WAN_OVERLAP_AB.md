# sc-15445: Wan temporal-overlap A/B

## Verdict

Restore the candidate grid's historical temporal overlap for the Wan z16 and z48 product selectors.
Keep half-tile overlap for Krea Realtime and SCAIL-2, which share the z16 VAE but enter through the
quality-oriented selector added for their sc-15325 correction. Keep LTX's separately measured
half-tile policy.

Across four real-weight cells, half-tile overlap added **21.1–48.5%** VAE decode wall time. It reduced
error against the high-context reference by **0.10–0.60/255**, did not materially change clipping, and
did not change peak memory. That is material Wan wall time for a marginal quality gain.

## Controlled experiment

- Hardware: Apple M5 Max MacBook Pro, 18 CPU cores, 128 GB unified memory.
- OS: macOS 26.5.2 (25F84).
- Timed source: inference `1e9e45d2`, plus
  `mlx-gen-wan/tests/wan_overlap_ab.rs`; release profile, MLX 0.32.0 fork `932beb4e`.
- MLX memory context: default 121.600 GiB limit; cache cleared and peak reset before every decode; no
  concurrent Metal workload during timing. A duplicate task briefly overlapped the final cell's
  post-timing quality phase. Those quality materializations were discarded and rerun uncontended;
  the clean rerun reproduced them exactly.
- Stimulus: identical deterministic normalized Gaussian latent per A/B/reference, seed 15445.
  This is representative decoder input, not generated semantic content. It isolates the decode
  policy and makes the timing repeatable; source-dependent clipping is therefore reported but not
  generalized.
- Spatial policy: 192 px tile / 64 px overlap, a shipping candidate, fixed across A/B/reference.
- Temporal A/B: candidate overlap versus half-tile overlap on the same tile.
- Reference: 96-frame tile / 48-frame overlap. It is temporally single-pass for the 81-frame cell and
  the highest-context affordable reference for the 121-frame cell.
- Timing: one warm-up per A/B arm, then three alternating materialized decodes. Reported wall time is
  median with min–max range.
- Accuracy: mean absolute difference from the reference on the viewer's 0–255 scale.
- Clipping: mean/worst frame percentage of pixels with any channel at or above 250/255.
- z16 lengths: 81/121 product frames map to 84/124 decoded frames before the normal product trim
  because z16 is non-causal (`T_lat * 4`).
- z48 lengths: causal decode emits the exact 81/121 product frames. We cast the VAE to bf16 exactly as
  the shipping TI2V-5B path does.

Weights:

| family | snapshot / file | SHA-256 | bytes |
|---|---|---|---:|
| z16 | `SceneWorks/wan2.2-t2v-a14b-mlx@991eb255…/bf16/vae.safetensors` | `4bcdb28b031fe96df6b61cc6c8f61c4563a9e0fad50f0ebb51089bf739865983` | 507,591,260 |
| z48 | `SceneWorks/wan2.2-ti2v-5b-mlx@bb1b0552…/vae.safetensors` | `d78f8b02e0058ec717672a916a7cb150fa868d6cd2311c7db1541db3274d6b81` | 2,818,778,910 |

## Results

`calls` is `(temporal tiles / all temporal×height×width tiles)`.

| family / bucket | temporal A/B | calls old → half | median seconds old → half (range) | wall | MAE/255 old → half | clip mean old → half | clip worst old → half | peak GiB old → half |
|---|---|---|---|---:|---:|---:|---:|---:|
| z16 640×384×81 (84 decoded) | 32/8 → 32/16 | 4/60 → 5/75 | 66.325 (65.999–74.315) → 98.481 (92.060–100.976) | +48.48% | 2.4579 → 1.8535 | 0.0155% → 0.0139% | 0.2669% → 0.2669% | 7.793 → 7.582 |
| z16 832×480×121 (124 decoded) | 48/16 → 48/24 | 4/96 → 5/120 | 164.202 (155.473–165.560) → 216.347 (213.282–229.452) | +31.76% | 1.1873 → 0.8875 | 0.0102% → 0.0104% | 0.1635% → 0.1635% | 11.495 → 11.811 |
| z48 bf16 640×384×81 | 32/8 → 32/16 | 4/60 → 5/75 | 87.143 (85.697–89.064) → 121.351 (120.933–121.557) | +39.25% | 0.9496 → 0.7999 | 0.1723% → 0.1695% | 3.3777% → 3.3777% | 4.725 → 4.725 |
| z48 bf16 832×480×121 | 48/16 → 48/24 | 4/96 → 5/120 | 241.198 (240.296–242.399) → 292.123 (289.300–305.715) | +21.11% | 0.5729 → 0.4760 | 0.1504% → 0.1488% | 2.7236% → 2.7236% | 6.785 → 6.785 |

The half-tile policy adds exactly 25% more tile calls at these points. Wall time rises by more than
the call ratio in three cells and slightly less in the longest z48 cell; repeated ranges do not
overlap. Peak is flat within measurement noise, confirming that overlap is recomputation rather than
a larger live tile. Clipping direction remains source-dependent and negligible here, matching the
warning carried from sc-15325.

Raw logs:

| log | SHA-256 | use |
|---|---|---|
| `/private/tmp/sc15445-z16-640.log` | `648c5006f72e1e47e613291dfa457b16ec92658309a451c8bc6722a8c372c898` | z16 short timing + quality |
| `/private/tmp/sc15445-z16-832.log` | `0f66caa2e964ed0c60cdf7ea096754b181134affb66a2bdbd818820900e5ed4a` | z16 long timing + quality |
| `/private/tmp/sc15445-z48-640.log` | `dfb42bb0de1ac901229a1f2214318b91cd5b394475fefac048c3d63fbbdcbbd3` | z48 short timing + quality |
| `/private/tmp/sc15445-z48-832.log` | `13449abf7606be22501a1bd1035572b2dc52d8cbdddb7854c1818cc261c55012` | z48 long timing; quality discarded after duplicate started |
| `/private/tmp/sc15445-z48-832-quality-clean.log` | `23a35a7a9f04d1736c78c58e1fbddf12d073069d6f71b506e13066a52760a79d` | uncontended z48 long quality rerun |

## Durable gate

The decision is mutation-gated at three levels:

1. gen-core asserts candidate versus half-tile overlap on every shipping temporal row and reproduces
   the exact 60→75 / 96→120 iteration counts from all four measured cells;
2. `mlx-gen-wan` asserts Wan product selectors emit candidate overlap, the z16 quality selector emits
   half-tile overlap, and the recorded cost/quality conjunction remains the rationale;
3. Krea Realtime and SCAIL-2 each assert their live decode seam still emits 32/16 at the measured
   640×384 point, so either consumer accidentally switching back to Wan's product selector is red.

The ignored `wan_overlap_ab` test remains the rerunnable real-weight harness. Set
`WAN_OVERLAP_AB_QUALITY_ONLY=1` to rerun reference/quality without repeating wall-time collection.
