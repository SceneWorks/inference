# sc-22989 — Tongyi Qianwen License evidence (YuE2 `qwen.tiktoken`)

This note sits beside `docs/licensing/sc-16662-licence-family-evidence.md`,
`docs/licensing/sc-16665-checkpoint-licence-evidence.md` and
`docs/licensing/sc-24108-qwen-research-licence-evidence.md`. It records the primary-source quotes
behind two things:

- the `tongyi-qianwen` family in `crates/contracts/gen-core/src/license/families.rs`;
- the `yue2_qwen_tiktoken` component row in `crates/audio/candle-audio-yue2/src/license.rs`.

Both were introduced by the YuE2 asset closure (epic sc-22988).

**PROVISIONAL — gathered by an agent on 2026-09-26 and not yet signed off by a human.** This note
has the same status as its siblings.

## Why a Tongyi Qianwen row exists at all

- YuE2's text/ABC tokenizer is `qwen.tiktoken` from `m-a-p/YuE2-3B` at revision
  `1a96eca688d6ae5d7f0feb88573fec89920fcd19`: 2,561,218 bytes, SHA-256
  `b2b1b8dfb5cc5f024bafc373121c6aba3f66f9a5a0269e243470a1de16a33186`, git blob
  `9b9b0e0416d84d7c88333eb261c77e5fe2d7f7be`.
- On 2026-09-26 the Hub API reported that git blob id for the `qwen.tiktoken` in each of these
  repositories: `Qwen/Qwen-7B` (revision `ef3c5c9c57b252f3149c1408daf4d649ec8b6c85`),
  `Qwen/Qwen-1_8B` and `Qwen/Qwen-14B`. It is the same file, byte for byte.
- The YuE2-3B card declares `license: cc-by-nc-4.0`. YuE2's own `MODEL_LICENSE`, however, scopes
  that licence to the checkpoint weights and says: "This weight license does not replace
  separately applicable licenses for code, text tokenization files, evaluation assets or other
  bundled material."
- The row therefore declares the file's **origin** licence, not the redistributor's tag.
- The YuE2-3B repository ships neither the Tongyi Qianwen Agreement nor its §3(c) Notice.

## Source

- **Text:** the `LICENSE` beside the Qwen-7B weights, at
  <https://huggingface.co/Qwen/Qwen-7B/raw/ef3c5c9c57b252f3149c1408daf4d649ec8b6c85/LICENSE>.
  Its title is "Tongyi Qianwen LICENSE AGREEMENT", with "Tongyi Qianwen Release Date: August 3, 2023".
  It is vendored verbatim at `crates/audio/candle-audio-yue2/licenses/qwen-tiktoken/LICENSE`
  (SHA-256 `7c7b8e244f6aa1ac8c32b74f56d42c41a0364dd2dabed8d9c6030a862e805b54`).
- **NOTICE:** the `NOTICE` at the same revision is vendored alongside it. It carries third-party
  *code* licences: NVIDIA Megatron-LM, OpenAI tiktoken (MIT), stanford_alpaca and AutoGPTQ.
- **Declaration:** the Qwen-7B card front matter reads `license: other` and
  `license_name: tongyi-qianwen-license-agreement`. `declared` is therefore
  `tongyi-qianwen-license-agreement`, verbatim, following the sc-24108 convention for
  `license: other` cards.
- **Gating:** the Qwen-7B and YuE2-3B repositories are both public (`gated: false` on 2026-09-26).

## Terms transcribed

| term | clause | quote |
| --- | --- | --- |
| `DownstreamLicenseCopy { family: tongyi-qianwen }` | §3(a) | You shall give any other recipients of the Materials or derivative works a copy of this Agreement; |
| `NoticeFileRequired` | §3(b), §3(c) | You shall cause any modified files to carry prominent notices stating that You changed the files; … You shall retain in all copies of the Materials that You distribute the following attribution notices within a "Notice" text file distributed as a part of such copies: |
| `AttributionRequired` | §3(c) | "Tongyi Qianwen is licensed under the Tongyi Qianwen LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved." |
| `DeployerObligation` | §4 | If you are commercially using the Materials, and your product or service has more than 100 million monthly active users, You shall request a license from Us. You cannot exercise your rights under this Agreement without our express authorization. |
| `DeployerObligation` | §5(b) | You can not use the Materials or any output therefrom to improve any other large language model (excluding Tongyi Qianwen or derivative works thereof). |

## Silence recorded as silence

- **Noncommercial.** §2 grants a licence "to use, reproduce, distribute, copy, create derivative
  works of, and make modifications to the Materials" with no noncommercial limit. No
  `NonCommercialWeights` is transcribed. This is the difference from the `qwen-research` family.
- **Outputs.** No outputs restriction beyond §5(b), which is transcribed verbatim.
- **Revenue ceiling.** The §4 threshold is counted in monthly active users, not revenue. It is
  carried as a `DeployerObligation` rather than a `RevenueCeiling`, following the landed
  `llama-3-1-community` reading.
- **Acceptable-use policy.** None is named. §5(a) is an export-control compliance duty.

## Policy consequence (YuE2 crate, not gen-core)

These are recorded in `candle-audio-yue2::license::POLICIES`; the gen-core surface stays
disclosure-only.

- **Noncommercial local experimentation is permitted**, on the §2 grant.
- **Commercial use is gated.** §4 and §5(b) have not been reviewed for this product, and the YuE2
  weights the tokenizer serves are noncommercial regardless.
- **Redistribution is gated.** §3 requires the Agreement and a Notice file with every copy, and no
  owner-recorded basis exists.

## Judgement calls, flagged

1. §3(b) and §3(c) are transcribed as one `NoticeFileRequired` plus one `AttributionRequired`. This
   is the same U11 reading that `qwen-research`, `flux-non-commercial-v2-1` and
   `ideogram-4-non-commercial` use.
2. The component row's `source_url` is the **origin** repository (`Qwen/Qwen-7B`), not the
   redistributing `m-a-p/YuE2-3B`. This follows the `ComponentLicense::source_url` rule for an
   artifact bundled inside another party's repository.
3. `retrieved` is `2026-09-26`, the day both repositories were read. The licence's own release
   date (2023-08-03) is recorded in the family doc comment.
