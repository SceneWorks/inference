# MiniMax-H3-Image-PE licence evidence (SW-7099)

Read on 2026-11-10. Neither repository is gated.

| Item | Primary evidence | SHA-256 of the file read |
| --- | --- | --- |
| Family `minimax-h3-image-community` | [LICENSE at 492fd8c4](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/492fd8c4eca666f1085f57e30738c1e0d86b3439/LICENSE), dated November 10, 2026 | `edfab8ece9d0ccd39106c4fe44105ce599b27ca7dd22601b736a8e9219cb4d26` |
| Component `minimax_h3_image_pe` | [README at 35603d4b](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE/blob/35603d4b63949afa12cb098227fffefedf081afa/README.md): front matter `license: other`, `license_name: minimax-h3-image-community-license-agreement` | `b215c9ff2fcc401d34515559cd520f63bfe705df6a9538c9f048e1a6ecaf6e1c` |

The PE weights are a fine-tune of Apache-2.0 Qwen3.8-27B, but the LICENSE's closing note puts
the text-to-image prompt-enhancer model under the Agreement "for the avoidance of doubt", so one
row covers the weights and the `prompt/` files.

Diffing the LICENSE against the MiniMax-H3 LICENSE (`minimax-h3-community`, read 2026-08-12)
after renaming the model leaves these material differences:

- **§IV.1** — prior written authorization through `https://platform.minimax.io/license` when the
  Annualized Revenue of the licensee and its affiliates *exceeds* 1 Million USD (H3: more than
  20 million USD, by email), **or** when the licensee provides Hosted Service(s) to any Third
  Parties. Recorded as `revenue_ceiling:1000000:exclusive`, `registration_required` with the URL,
  and a `deployer_obligation` quoting the hosted-service trigger.
- **§I.6 / §I.8** — "MiniMax H3" is the image model and expressly includes prompt-enhancement
  models; "Hosted Service" gains a definition.

Unchanged and transcribed as for H3: §III.1 licence copy, §III.4 NOTICE file, §IV.2 prominent
"MiniMax-H3-Image" display, §V.2 downstream restrictions, §V.4 Excluded Territories (EU, UK,
Republic of Korea, USA), §V.5 safeguards, Exhibit A acceptable use (inside the LICENSE).

No MiniMax prompt text or weight is committed, and no converted tier (bf16 / Q8 / Q4, local or
published) bundles the `prompt/` files: `modelopt_nvfp4_to_bf16` drops the directory. The adapter
(`core_llm::minimax_h3_image_pe`) reads the prompts from the upstream
`MiniMaxAI/MiniMax-H3-Image-PE` snapshot at the pinned revision and checks them against
`prompt/manifest.json`, whose
SHA-256 at the pinned revision is `6b0969322669d9713e33148d36c401f73b18939337ea2384d90dc457430b6a5b`.
Its tests and quality prompt set use this repository's own text.
