# sc-25900 — Qwen-Image 2.1 Turbo licence evidence

Companion to `docs/licensing/sc-24108-qwen-research-licence-evidence.md`: the primary-source read
behind the `qwen_image_2_1_turbo` component row
(`crates/contracts/gen-core/src/license/components.rs`) that the `qwen_image_2_1_turbo` engine id
introduced on both Qwen-Image 2.1 providers — MLX (`mlx-gen-qwen-image-2-1`, sc-25900) and Candle
(`candle-gen-qwen-image-2-1`, sc-25901), each mapping it in its catalog's licence table
(`mlx-gen-catalog` / `candle-gen-catalog` `src/licenses.rs`). The row lands in the existing
`qwen-research` family; no family is added and no term is transcribed anew.

**PROVISIONAL — gathered by an agent on 2026-10-13, not yet signed off by a human.** Same status
as the sibling notes.

## `qwen_image_2_1_turbo` — `Qwen/Qwen-Image-2.1-Turbo`

* Declaration: the model card front matter at the pinned revision reads `license: other`,
  `license_name: qwen-research`, `license_link: LICENSE`, `base_model: Qwen/Qwen-Image-2.1`
  (<https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/raw/fc13ba1986fcb905854dc8de2f52eee395f49b5b/README.md>).
  `declared` is therefore `qwen-research`, verbatim — the same declaration as the base row.
* Text: `LICENSE` beside the pinned weights —
  <https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/raw/fc13ba1986fcb905854dc8de2f52eee395f49b5b/LICENSE>
  — is **byte-identical** to the base repository's `LICENSE` at its pinned revision
  (<https://huggingface.co/Qwen/Qwen-Image-2.1/raw/790c92633540aa0cb11d9abf19eb46d861714758/LICENSE>):
  both are 7831 bytes with SHA-256
  `8dc973f024ff95966bea25866efa443fd16776dcb1001e681e3d467ea572b28d`. Every term sc-24108
  transcribed for the `qwen-research` family therefore applies unchanged, and the §3(c) attribution
  notice is the same sentence.
* Gating: public (`gated: false` from the Hub API on 2026-10-13).
* One row covers the DiT, the RGBA VAE and the bundled Qwen3-VL-8B text tower, which ship together
  in the one repository under the one declaration — the same shape as `qwen_image_2_1`.

## Judgement calls, flagged

1. A separate row rather than reusing `qwen_image_2_1`: the weights are a different repository,
   and a row's `source_url` is what a drift job re-reads. The family, declaration and attribution
   are shared; the provenance is not.
2. `retrieved` is `2026-10-13`, the day the pinned revision was read.
