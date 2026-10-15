# sc-26234 — FLUX Kommunity licence evidence

Companion to `docs/licensing/sc-16662-licence-family-evidence.md`: the primary-source read behind
the `flux-kommunity-v1-0` family (`crates/contracts/gen-core/src/license/families.rs`) and the
`flux3_image_klein` component row (`crates/contracts/gen-core/src/license/components.rs`) that
the `flux3_klein` engine id introduced on the MLX FLUX 3 provider (`mlx-gen-flux3`, epic
sc-26233), mapped in `mlx-gen-catalog/src/licenses.rs`.

**PROVISIONAL — gathered by an agent on 2026-10-15, not yet signed off by a human.** Same status
as the sibling notes.

## `flux3_image_klein` — `black-forest-labs/flux-3-image-klein`

* Declaration: the model card front matter at the pinned revision
  `8454017f9ebaada30c998fccd9f67a80c7580f68` reads `license: other`,
  `license_name: flux-kommunity-license`, `license_link: LICENSE.md`. `declared` is therefore
  `flux-kommunity-license`, verbatim.
* Text: `LICENSE.md` beside the pinned weights (19,593 bytes), titled **FLUX Kommunity License
  v.1.0**, read through <https://huggingface.co/black-forest-labs/flux-3-image-klein/resolve/8454017f9ebaada30c998fccd9f67a80c7580f68/LICENSE.md>
  (served ungated; the weights are gated — `gated: auto` from the Hub API on 2026-10-15).
* One row covers the DiT (`model.safetensors`, sha256
  `c3e9813cdd049c831c836b2e9db61150e49a93fc1a4c63b256494642fee0bd7c`) and the autoencoder
  (`ae.safetensors`, sha256 `868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3`),
  which ship together in the one repository under the one declaration.
* The bundled `text_encoder/` is **not** this row. Its two shards carry the same LFS sha256 as
  `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17`
  (`30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9`,
  `046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6`), and every non-LFS
  file has the same git blob oid as that revision's tree (Hub `/tree` API vs the local
  `blob <size>\0` SHA-1, 2026-10-15): `config.json` `22be8bb79043d39840ca5e70d7a2dab80311602c`,
  `tokenizer.json` `c6cc1014128b19d1fc46b1d30a23e3b1d35db421`, `tokenizer_config.json`
  `d3d3763207692c78780f4bf42d4dadf49a5c8012`, `chat_template.json`
  `e49e75bdda104c8de941eb20e0b3765184fb27bc`, `generation_config.json` `e116347d…`,
  `merges.txt` `20024bfe…`, `vocab.json` `4783fe10…`, `model.safetensors.index.json`
  `3b52de97…`, `preprocessor_config.json` `2ea84a43…`, `video_preprocessor_config.json`
  `3ba673a5…`, `README.md` `54241709…`. The one file without a twin is the bundled
  `text_encoder/LICENSE` (the Apache License 2.0 text; the Qwen tree carries its licence at
  the repository root). The provider therefore maps the existing
  `qwen3_vl_4b_instruct` row (sc-25679) for it.

## The family — transcribed terms, each with its clause

| term | clause |
| --- | --- |
| `non_commercial_weights` | §2(b) "You may only access, use, Distribute, or create Derivatives of the FLUX Model or Derivatives for Non-Commercial Purposes, except as set forth in Section 2.a." |
| `revenue_ceiling:5000000:inclusive` | §1(e) "'Qualifying User' means you, if at any time you together with your affiliates' gross annualized revenue is less than US$5,000,000." — "less than", so the amount itself is past the threshold (Inclusive). §2(a)(ii) grants a Qualifying User the right to "use Outputs, including Outputs from Derivatives, for commercial purposes, and you may access and use the FLUX Model and Derivatives in a production environment solely for the purpose of generating such Outputs". The exception is for **Outputs** only: §1(c)(c) excludes "to train, fine tune, or distill other models for commercial use" from Non-Commercial Purposes and §4(a) forbids commercial Distribution and Derivatives beyond §2(a), so `non_commercial_weights` stands and no `non_commercial_outputs` is recorded. |
| `registration_required:https://bfl.ai/licensing` | §2(b) "you must request a license from Company, which Company may grant to you in Company's sole discretion … Please see https://bfl.ai/licensing if you would like a commercial license." |
| `downstream_license_copy:flux-kommunity-v1-0` | §3(a) "you must make available a copy of this License to third-party recipients of the FLUX Model and/or Derivatives you Distribute, and specify that any rights to use the FLUX Model and/or Derivatives shall be directly granted by Company to said third-party recipients pursuant to this License". |
| `notice_file_required`, `attribution_required` | §3(b) "you must prominently display the following notice alongside the Distribution of the FLUX Model or Derivative (such as via a 'Notice' text file distributed as part of such FLUX Model or Derivative) (the 'Attribution Notice')". The notice text is the row's `attribution`. |
| `downstream_restrictions:flux-kommunity-v1-0` | §3(c)(ii) "any terms and conditions you impose on any third-party recipients relating to Derivatives made by or for you shall neither limit such third-party recipients' use of the FLUX Model or any Derivatives made by or for Company in accordance with this License nor conflict with any of its terms and conditions and must include disclaimer of warranties and limitation of liability provisions that are at least as protective of Company as those set forth herein". |
| `deployer_obligation` (content filters) | §2(e)(i)(A) "implement and maintain content filtering measures ('Content Filters') for your use of the FLUX Model or Derivatives to prevent the creation, display, transmission, generation, or dissemination of unlawful, unsafe or infringing content". |
| `deployer_obligation` (AI disclosure) | §2(e)(ii) "ensure Output includes disclosures (or other indications) that the Output was generated or modified using artificial intelligence technologies to the extent required under applicable law". |
| `acceptable_use_policy:<none>` | §4 enumerates the prohibited uses inline (commercial/production beyond §2(a), military, surveillance, biometric processing, infringing and unlawful uses, Export Laws) and names no separate policy document; the only address in the text is the licensing page above. |

## Judgement calls, flagged

1. A new family rather than reusing `flux-non-commercial-v2-1`: the Kommunity text adds the
   Qualifying-User revenue ceiling, the licensing-page registration route, the §3(c) flow-down of
   restrictions and the Robotics-Uses carve-outs, and its §4 restrictions differ, so the two texts
   must not share one row of terms.
2. `retrieved` is `2026-10-15`, the day the pinned revision was read.
3. The base checkpoint (`flux-3-image-klein-base`) and the prompt upsampler
   (`flux-3-image-klein-pup`) are later stories' rows; nothing here reads them.
