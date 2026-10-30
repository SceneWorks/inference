# sc-26238 — FLUX 3 Image Klein prompt upsampler licence evidence

Companion to `docs/licensing/sc-26234-flux-kommunity-licence-evidence.md`: the primary-source
read behind the `flux3_image_klein_pup` component row
(`crates/contracts/gen-core/src/license/components.rs`) that story sc-26238 adds to the
`flux3_klein` provider mapping (`mlx-gen-catalog/src/licenses.rs`). The upsampler is an
**optional** co-requisite of the FLUX 3 provider (epic sc-26233, E5): the engine loads it only
when `LoadSpec::components["prompt_upsampler"]` names its package, and it is priced and licensed
separately from the image model and its text encoder.

**PROVISIONAL — gathered by an agent on 2026-10-15, not yet signed off by a human.** Same status
as the sibling notes.

## `flux3_image_klein_pup` — `black-forest-labs/flux-3-image-klein-pup`

* Declaration: the model card front matter at the pinned revision
  `8b069751eb5dcab23829d9c5eccb470eb8b8a4ed` reads `license: apache-2.0` and
  `base_model: [Qwen/Qwen3.5-9B]`. `declared` is therefore `apache-2.0`, verbatim; the family is
  `apache-2-0`.
* The card's own licence section: *"This model is released under the Apache License 2.0. It is
  derived from Qwen3.5-9B by the Qwen team at Alibaba Cloud, released under the Apache License
  2.0."* The inference repository's README (`black-forest-labs/flux-3-image-klein` @
  `45e57c9fb06f9751cdced29a65c5d2171003be05`) says the same in its licence paragraph: *"the prompt
  upsampler is Apache 2.0"*, in the same sentence that puts the image weights under the FLUX
  Kommunity License — which is why this is a row of its own and not part of
  `flux3_image_klein`.
* Distribution: the Hub API reported `gated: false`, `private: false` on 2026-10-15; the files
  were read with the configured HF credential after an unauthenticated IP rate-limit response. `model.safetensors` is 18,819,722,392 bytes, Hub linked-object
  sha256 `f5062a1fba3e63afe5c88349a311903fec06c959b362f1247127e63d5c860ed4`
  (`gen_core_flux3::pup::UPSAMPLER_WEIGHTS_SHA256`); `system_prompts.json` sha256
  `b5da3ed8bc04ab59dbdb747930f7dbff847e868903fed8028f85ee1a4ac82fb4`, byte-identical to the
  inference repository's `src/flux3/pup_prompts.json` at the pinned commit, and embedded in the
  contract crate (`gen-core/src/flux3/pup_system_prompts.json`) under the same licence.
* Attribution carried by the row: *"FLUX 3 Image Klein Prompt Upsampler © Black Forest Labs —
  licensed under Apache-2.0; derived from Qwen3.5-9B © Alibaba Cloud (Apache-2.0)"*. The card's
  trademark paragraph (no implied sponsorship through the FLUX / Black Forest Labs marks) applies
  to redistributions, not to loading the weights.
* Nothing in the row asserts an outputs restriction: Apache-2.0 places none, and the card names
  none.

## What the provider mapping says

`flux3_klein` now lists `flux3_image_klein`, `qwen3_vl_4b_instruct` and `flux3_image_klein_pup`.
The third is reached only by a request with `enhance_prompt` on a generator loaded with the
component; a generator loaded without it refuses such a request with a typed error rather than
silently sending the prompt as typed, so the licence surface a consumer sees is exactly the set of
weights it provisioned.
