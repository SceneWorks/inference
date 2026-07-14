# Model Catalog Reference

The complete provider surface shipped by each platform bundle, as of release
`runtime-2026.07.0`. Every id below is a **load key**: `registry.load("<id>", &spec)`
(and the per-kind `load_trainer` / `load_captioner` / `load_image_embedder` /
`load_text_embedder`).

These lists are **authoritative** — they are asserted verbatim by the exact-surface tests
in [`mlx-gen-catalog`](../../crates/media/mlx-gen/mlx-gen-catalog/src/lib.rs) and
[`candle-gen-catalog`](../../crates/media/candle-gen/candle-gen-catalog/src/lib.rs), which
also run the weights-free descriptor conformance sweep. If a bundle's shipped surface
changes, those tests fail until updated, and this document should be updated with them. To
read the surface programmatically at runtime, call `catalog().snapshot()` (see the
[Getting Started guide](../guide/getting-started.md#4-inspect-the-surface-without-loading-weights)).

## Platforms at a glance

| Kind                | `runtime-macos` (MLX) | `runtime-cuda` / `runtime-cpu` (Candle) |
| ------------------- | --------------------: | --------------------------------------: |
| Generators          |                    57 |                                      43 |
| Trainers            |                    14 |                                       6 |
| Transforms          |                     0 |                                       0 |
| Captioners          |                     1 |                                       1 |
| Image embedders     |                     1 |                                       1 |
| Text embedders      |                     1 |                                       1 |
| Text LLMs           |                     2 |                                       2 |
| Snapshot preparers  |             `mlx`     |                               `candle`  |

`runtime-cuda` and `runtime-cpu` ship the **same Candle media/LLM surface**; they differ
in target triples and native prerequisites, not in the provider list.

### Id suffix conventions

Ids are self-documenting. Recurring suffixes:

| Suffix                                   | Meaning                                                             |
| ---------------------------------------- | ------------------------------------------------------------------ |
| `_turbo`, `_flash`, `_sprint`, `_schnell`| Distilled / few-step fast variant                                  |
| `_raw`                                   | Base, un-distilled variant                                         |
| `_edit`                                  | Reference-image edit (image + text → image)                        |
| `_control`                               | ControlNet control-branch variant (requires `LoadSpec.control`)    |
| `_kv_edit`                               | KV-cache edit path (FLUX.2)                                         |
| `t2v` / `i2v` / `ti2v`                   | Text- / image- / (text+image)-to-video (Wan)                       |
| `_vace`, `vace_fun`                      | VACE control-video variant                                         |
| `_1600m`, `_5b`, `_7b`, `_8b`, `_9b`, `_14b` | Approximate parameter scale                                    |

## Generators

Grouped by provider family. `✓` = shipped on that platform; `—` = not shipped.

| Family (crate) | Model id | MLX | Candle |
| --- | --- | :---: | :---: |
| **anima** | `anima_base` | ✓ | ✓ |
| | `anima_aesthetic` | ✓ | ✓ |
| | `anima_turbo` | ✓ | ✓ |
| **bernini** | `bernini_renderer` | ✓ | ✓ |
| | `bernini` | ✓ | ✓ |
| **boogu** | `boogu_image` | ✓ | ✓ |
| | `boogu_image_turbo` | ✓ | ✓ |
| | `boogu_image_edit` | ✓ | ✓ |
| **chroma** | `chroma1_hd` | ✓ | ✓ |
| | `chroma1_base` | ✓ | ✓ |
| | `chroma1_flash` | ✓ | ✓ |
| **flux** (FLUX.1) | `flux1_schnell` | ✓ | ✓ |
| | `flux1_dev` | ✓ | ✓ |
| | `flux1_dev_control` | ✓ | — |
| **flux2** (FLUX.2) | `flux2_klein_9b` | ✓ | ✓ |
| | `flux2_klein_9b_edit` | ✓ | — |
| | `flux2_klein_9b_kv_edit` | ✓ | — |
| | `flux2_dev` | ✓ | ✓ |
| | `flux2_dev_edit` | ✓ | — |
| | `flux2_dev_control` | ✓ | — |
| **ideogram** | `ideogram_4` | ✓ | ✓ |
| | `ideogram_4_turbo` | ✓ | ✓ |
| **kolors** | `kolors` | ✓ | ✓ |
| **krea** | `krea_2_turbo` | ✓ | ✓ |
| | `krea_2_raw` | ✓ | ✓ |
| | `krea_2_edit` | ✓ | ✓ |
| | `krea_2_turbo_edit` | ✓ | — |
| | `krea_2_turbo_control` | ✓ | — |
| **lens** | `lens_turbo` | ✓ | ✓ |
| | `lens` | ✓ | ✓ |
| **ltx** (video + audio) | `ltx_2_3` | ✓ | — |
| | `ltx_2_3_distilled` | — | ✓ |
| **pulid** (identity, FLUX) | `pulid_flux` | ✓ | — · *(bespoke — see below)* |
| **qwen-image** (Qwen-Image) | `qwen_image` | ✓ | ✓ |
| | `qwen_image_control` | ✓ | — |
| | `qwen_image_edit` | ✓ | — |
| **sana** (SANA) | `sana_1600m` | ✓ | ✓ |
| | `sana_sprint_1600m` | ✓ | — |
| **scail2** | `scail2_14b` | ✓ | ✓ |
| **sd3** (Stable Diffusion 3.5) | `sd3_5_large` | ✓ | ✓ |
| | `sd3_5_large_turbo` | ✓ | ✓ |
| | `sd3_5_medium` | ✓ | ✓ |
| **sdxl** (SDXL) | `sdxl` | ✓ | ✓ |
| **seedvr2** | `seedvr2` | ✓ | ✓ |
| | `seedvr2_3b` | ✓ | ✓ |
| | `seedvr2_7b` | ✓ | ✓ |
| **sensenova** | `sensenova_u1_8b` | ✓ | ✓ |
| | `sensenova_u1_8b_fast` | ✓ | ✓ |
| **svd** (Stable Video Diffusion, image → video) | `svd_xt` | ✓ | ✓ |
| **wan** (Wan 2.2, video) | `wan2_2_ti2v_5b` | ✓ | ✓ |
| | `wan2_2_t2v_14b` | ✓ | ✓ |
| | `wan2_2_i2v_14b` | ✓ | ✓ |
| | `wan_vace` | ✓ | ✓ |
| | `wan2_2_vace_fun_14b` | ✓ | — |
| **z-image** | `z_image_turbo` | ✓ | ✓ |
| | `z_image` | ✓ | ✓ |
| | `z_image_control` | ✓ | — |
| | `z_image_turbo_control` | ✓ | — |

## Trainers

LoRA/LoKr fine-tuning is available for a subset of generator families. Load with
`registry.load_trainer("<id>", &spec)`.

| Trainer id | MLX | Candle |
| --- | :---: | :---: |
| `anima_base` | ✓ | — |
| `anima_aesthetic` | ✓ | — |
| `anima_turbo` | ✓ | — |
| `kolors` | ✓ | — |
| `krea_2_raw` | ✓ | ✓ |
| `krea_2_control` | — | ✓ |
| `lens` | ✓ | ✓ |
| `ltx_2_3` | ✓ | — |
| `sd3_5_large` | ✓ | — |
| `sd3_5_medium` | ✓ | — |
| `sdxl` | ✓ | ✓ |
| `wan2_2_t2v_14b` | ✓ | ✓ |
| `wan2_2_i2v_14b` | ✓ | — |
| `wan2_2_ti2v_5b` | ✓ | — |
| `z_image_turbo` | ✓ | ✓ |

## Captioners, embedders

Identical on both platforms.

| Kind | Id |
| --- | --- |
| Captioner | `fancyfeast/llama-joycaption-beta-one-hf-llava` |
| Image embedder | `clip_vit_l14` |
| Text embedder | `clip_vit_l14_text` |

Neither platform ships any standalone `Transform` provider at this release.

## Text LLMs

Loaded through `catalog.text()` (a `core_llm::TextLlmRegistry`); see the
[Getting Started guide, §6](../guide/getting-started.md#6-serve-an-llm).

| Platform | Text LLM ids |
| --- | --- |
| `runtime-macos` (MLX) | `mlx-llama`, `mlx-joycaption` |
| `runtime-cuda` / `runtime-cpu` (Candle) | `candle-llama`, `candle-llava` |

## Bespoke utility crates

Some crates in a platform's `providers` module are **not** registered generators — they
are consumed through provider-specific APIs (depth maps, face analysis, segmentation,
identity conditioning, the PiD latent decoder) rather than the `load(id, spec)` path.
They are still owned and re-exported by the catalog so their platform membership is
explicit.

| Platform | Bespoke utility crates |
| --- | --- |
| `runtime-macos` (MLX) | `depth`, `face`, `instantid`, `pid`, `sam2`, `sam3` |
| `runtime-cuda` / `runtime-cpu` (Candle) | `depth`, `face`, `instantid`, `pid`, `pulid`, `sam3` |

## Platform differences (why the surfaces diverge)

The catalogs are **deliberately allowed to differ** where the implementations genuinely
differ; each difference is pinned in the exact-surface tests rather than papered over
(an ADR invariant). The notable deltas at this release:

- **Edit / control / distilled variants are further along on MLX.** MLX ships several
  reference-edit and ControlNet variants that Candle does not yet: `flux1_dev_control`,
  the FLUX.2 `_edit` / `_kv_edit` / `_dev_control` set, `krea_2_turbo_edit` /
  `krea_2_turbo_control`, `qwen_image_control` / `qwen_image_edit`, `sana_sprint_1600m`,
  `wan2_2_vace_fun_14b`, and `z_image_control` / `z_image_turbo_control`.
- **PuLID is a registered generator on MLX, a bespoke utility on Candle.** MLX exposes
  `pulid_flux` as a loadable generator id; on Candle, `pulid` is a bespoke identity
  utility crate consumed through its own API (no registered generator id).
- **SAM 2 is MLX-only.** MLX ships both `sam2` and `sam3` as bespoke utilities; Candle
  ships `sam3` only.
- **LTX ships a different variant per backend.** MLX registers `ltx_2_3`; Candle
  registers `ltx_2_3_distilled`.
- **The trainer sets differ substantially.** MLX offers 14 trainers (including the whole
  `anima` family, `kolors`, `ltx_2_3`, `sd3_5_large` / `sd3_5_medium`, and the Wan
  video trainers); Candle offers 6, and is the only backend with a `krea_2_control`
  trainer.

## See also

- [Getting Started](../guide/getting-started.md) — how to load and run any of these.
- [Architecture rationale](../architecture/inference-rearchitecture.md) — why explicit,
  per-platform catalogs (and their deltas) are the design, not an accident.
- [`mlx-gen/docs/MODEL_ARCHITECTURE.md`](../../crates/media/mlx-gen/docs/MODEL_ARCHITECTURE.md)
  — the `Generator` / `Transform` contract and per-family model internals.
