# sc-25679 — Iris-3B licence evidence

Companion to `docs/licensing/sc-16665-checkpoint-licence-evidence.md`: the primary-source reads
behind the `iris_3b` and `qwen3_vl_4b_instruct` component rows
(`crates/contracts/gen-core/src/license/components.rs`) that the Iris-3B MLX provider
(`mlx-gen-iris`) introduced. Both land in the existing `apache-2-0` family; no family is added.

**PROVISIONAL — gathered by an agent on 2026-10-10, not yet signed off by a human.** Same status
as the sibling notes.

## `iris_3b` — `speridlabs/iris-3b`

* Declaration: the model card front matter at the pinned revision reads `license: apache-2.0`
  (<https://huggingface.co/speridlabs/iris-3b/raw/7445443349bc9abe3c96f01ff793e2098ca012b3/README.md>).
* Code: the GitHub repository at `a8d15239dea469aba042cfa56ca3bb4e450d5ebc` ships the Apache
  License 2.0 text as `LICENSE` and a `NOTICE` naming the holder ("Copyright 2026 Speridlabs").
* Gating: public (`gated: false` from the Hub API on 2026-10-10).
* One row covers the generation backbone at the repository root and the `depth/` and `upscaler/`
  backbones, which the same card and licence govern.

## `qwen3_vl_4b_instruct` — `Qwen/Qwen3-VL-4B-Instruct`

* Declaration: the model card front matter at the pinned revision reads `license: apache-2.0`
  (<https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/raw/ebb281ec70b05090aa6165b016eac8ec08e71b17/README.md>).
* Gating: public (`gated: false` from the Hub API on 2026-10-10).
* Iris's own `NOTICE` lists it as a downloaded, not redistributed, runtime dependency under the
  Apache License 2.0; the provider loads it from the upstream repository itself (pinned revision),
  so the upstream declaration governs it.

## Judgement calls, flagged

1. `retrieved` is `2026-10-10`, the day both pinned revisions were read.
2. Outputs: Apache-2.0 places no restriction on outputs; nothing is transcribed beyond the family.
