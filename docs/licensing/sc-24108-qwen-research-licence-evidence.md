# sc-24108 — Qwen Research License evidence (Qwen-Image 2.1)

Companion to `docs/licensing/sc-16662-licence-family-evidence.md` and
`docs/licensing/sc-16665-checkpoint-licence-evidence.md`: the primary-source quotes behind the
`qwen-research` family (`crates/contracts/gen-core/src/license/families.rs`) and the
`qwen_image_2_1` component row (`crates/contracts/gen-core/src/license/components.rs`) that the
Qwen-Image 2.1 MLX provider (`mlx-gen-qwen-image-2-1`) introduced.

**PROVISIONAL — gathered by an agent on 2026-09-22, not yet signed off by a human.** Same status
as the two sibling notes.

## Source

* Text: `LICENSE` beside the pinned weights —
  <https://huggingface.co/Qwen/Qwen-Image-2.1/raw/790c92633540aa0cb11d9abf19eb46d861714758/LICENSE>
  (title "Qwen RESEARCH LICENSE AGREEMENT", "Release Date: September 20, 2026").
* Declaration: the model card's front matter reads `license: other`, `license_name: qwen-research`,
  `license_link: LICENSE` (same revision). `declared` is therefore `qwen-research`, verbatim.
* Gating: the repository is public (`gated: false` from the Hub API on 2026-09-22).

## Terms transcribed

| term | clause | quote |
| --- | --- | --- |
| `NonCommercialWeights` | §1(i), §2(a) | "Non-Commercial" shall mean for research or evaluation purposes only. … a non-exclusive, worldwide, non-transferable and royalty-free limited license … to use, reproduce, distribute, copy, create derivative works of, and make modifications to the Materials FOR NON-COMMERCIAL PURPOSES ONLY. |
| `RegistrationRequired { contact: model-business@notice.qwencloud.com }` | §2(b) | You shall not use the Materials for any commercial purpose without obtaining a separate commercial license from us. If you wish to use the Materials commercially, you shall request a license from us at model-business@notice.qwencloud.com. |
| `DownstreamLicenseCopy { family: qwen-research }` | §3(a) | You shall give any other recipients of the Materials or derivative works a copy of this Agreement; |
| `NoticeFileRequired` | §3(b), §3(c) | You shall cause any modified files to carry prominent notices stating that you changed the files; … You shall retain in all copies of the Materials that you distribute the following attribution notices within a "Notice" text file distributed as a part of such copies: |
| `AttributionRequired` | §3(c) | "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved." |
| `DeployerObligation` | §4(b) | If you use the Materials or any outputs or results therefrom to create, train, fine-tune, or improve an AI model that is distributed or made available, you shall prominently display "Built with Qwen" or "Improved using Qwen" in the related product documentation. |
| `DeployerObligation` | §4(c) | You shall not use "Qwen" as the primary name or identifier of any derivative works or products; reasonable descriptive use (e.g., "fine-tuned from Qwen Image") is permitted. |

## Silence recorded as silence

* **Outputs.** §2(a) restricts the *Materials* (§1(f): the model weights, code and documentation).
  §6(b)–(c) disclaim warranty and liability for "any output therefrom" but state no use
  restriction on outputs, so no `NonCommercialOutputs` is transcribed — the reading sc-16662's U3
  set for every other silent family.
* **Acceptable-use policy.** §4(a) is an export-control compliance duty, not a named policy
  document, and no URL or policy is referenced anywhere in the text; no `AcceptableUsePolicy`.
* **Revenue ceiling.** None named.

## Judgement calls, flagged

1. §3(b) and §3(c) are transcribed as one `NoticeFileRequired` plus one `AttributionRequired`
   (the §3(c) notice names itself an attribution notice) — the same U11 reading the landed
   `flux-non-commercial-v2-1` and `ideogram-4-non-commercial` families take.
2. §2(b)'s "request a license from us" is carried as `RegistrationRequired` with the named address,
   following the landed `chatglm3-model-license` reading of a commercial-use registration.
3. `retrieved` is `2026-09-22`, the day the pinned revision was read; the licence's own release
   date (2026-09-20) is recorded in the family doc comment, not as the retrieval date.
