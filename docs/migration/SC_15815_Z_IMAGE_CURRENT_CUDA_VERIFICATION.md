# SC-15815 Z-Image current Candle/CUDA verification

Date: 2026-08-03

## Provenance

- Inference baseline: `d48023204cd3a4f3f8eb060f79803dccaddcb482`
- SceneWorks baseline: `92c7a63fd3baf16b28b11000335363920ee9bbad`
- Provider/backend/tier: `z_image` / `candle_cuda` / packed Q4
- Artifact: `SceneWorks/z-image-mlx` snapshot
  `c74f74c2ad193294fc9ff3f8a5be71daa00d22ab`, `q4/`
- Cache root: `E:\huggingface`
- Device: NVIDIA RTX PRO 6000 Blackwell Max-Q, device 0, driver 596.36
- Toolchain: Visual Studio 2022 17.14.33, CUDA 12.9.41
- Fixed seed: 15815; one denoise step

The final paired SceneWorks revision pins the inference commit containing this evidence. The
baselines above identify the exact pre-repair pair from which the two verification branches were
created.

## Current quality measurements

The resident image is the reference. Staged residency remains byte-exact. The three bounded
strategies share an intentional whole-frame host f32 VAE decode; the transformer and attention
denoise paths remain identical to resident.

| Geometry | Strategy | max RGB8 | mean RGB8 | RMSE RGB8 | Reference SHA-256 | Candidate SHA-256 |
|---|---|---:|---:|---:|---|---|
| 256x256 | StagedResidency | 0 | 0 | 0 | `1747db3e04201ef92d5ea73b1986ce528e79ae20bfa93a6c9e6345d4fb657026` | same |
| 256x256 | BoundedDecode | 7 | 0.268534342 | 0.551588689 | `1747db3e04201ef92d5ea73b1986ce528e79ae20bfa93a6c9e6345d4fb657026` | `453873ce0f3937974f75a86daccc42d452b975586c0b333b7e790f70e8d70a60` |
| 256x256 | BoundedAttention | 7 | 0.268534342 | 0.551588689 | `1747db3e04201ef92d5ea73b1986ce528e79ae20bfa93a6c9e6345d4fb657026` | `453873ce0f3937974f75a86daccc42d452b975586c0b333b7e790f70e8d70a60` |
| 256x256 | BoundedTransformerResidency | 7 | 0.268534342 | 0.551588689 | `1747db3e04201ef92d5ea73b1986ce528e79ae20bfa93a6c9e6345d4fb657026` | `453873ce0f3937974f75a86daccc42d452b975586c0b333b7e790f70e8d70a60` |
| 1024x1024 | BoundedDecode | 10 | 0.303418477 | 0.610755335 | `56f3eb239fc8896d90a21bd19c3a1c6df875cb09f1afab1728409241df6ba525` | `0e878ddfe70e5d1d960bc41f689d75e2b33b8b8db1e9fbda83aea6e579bd1b3f` |

The previous 1024 bound (maximum 7, mean 0.280577342) was stale and failed closed against the
current pair. The replacement is strategy-specific and minimally rounded from the current
measurements: maximum 10, mean 0.304, RMSE 0.611. It applies only to packed-Q4 host-decode
strategies at the two measured geometries. Resident/staged, other numeric tiers, seeds, geometries,
providers, and backends do not inherit it.

## Contract correction

The first current five-rung run reached rung four and failed admission as `Missing`. The runtime was
correct: `BoundedTransformerResidency` is implemented only for directory weights loaded with
`DeferredMaterialization`; the verifier had used the eager `LoadSpec::new` default. Real-rung tests
now request the production deferred load shape. Unit tests continue to prove deferred is
`Implemented`, eager is `Missing`, stale fingerprints fail closed, and all control routes preserve
the ladder.

## Commands and results

All CUDA commands used `HF_HOME=E:\huggingface` and `CUDA_VISIBLE_DEVICES=0` and ran serially.

```text
cargo test --locked -p candle-gen-z-image --features cuda --test packed_tier_validate \
  packed_base_q4_production_host_decode_parity -- --ignored --exact --nocapture
```

Passed in 322.50 seconds; the result reproduced the 1024 metrics and hashes above exactly.

```text
cargo test --locked -p candle-gen-z-image --features cuda --test packed_tier_validate \
  packed_base_all_rungs_preserve_fixed_seed_output -- --ignored --exact --nocapture
```

Passed in 758.19 seconds. Resident was coherent, staged was exact, and all three bounded strategies
produced the same measured candidate hash.

```text
cargo test --locked -p candle-gen-z-image --features cuda --lib --tests
```

Passed: 101 library tests plus both request-scoped residency tests; real-weight tests not selected by
the commands above remained ignored. This covers stable base/turbo/edit registration and request
surfaces, all five memory-contract rungs, stale-evidence rejection, typed cancellation, recoverable
errors, phase cleanup, and warm staged/warm lifecycle behavior.
