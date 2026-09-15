# Licence family table — SIGN-OFF DOCUMENT (sc-16662)

**Status: PROVISIONAL. The values below are LANDED IN CODE and have NOT been read by a human.**

| | |
| --- | --- |
| Story | sc-16662 — Licence family table, human legal read of the ~14 families |
| Epic | 16660 |
| Landed in | `crates/contracts/gen-core/src/license/families.rs` — 16 `LicenseFamily` consts plus `LICENSE_FAMILIES` |
| Evidence gathered by | Claude (Opus 5), automated agent, on behalf of Michael Trefry |
| Transcribed into code by | Claude (Opus 5), automated agent, **2026-08-02** |
| Retrieval date for **every** quote below | **2026-08-02** (unless a row says otherwise) |
| Method | `curl` of the canonical licence file (upstream HF `raw/main/LICENSE*`, the licensor's own domain, or the licensor's own GitHub), then verbatim extraction of the operative clause |
| Gated repos | Files behind an HF gate were retrieved **2026-08-02** with the authenticated Hugging Face CLI (`hf download <repo> <file>`) by Claude (Opus 5) on Michael Trefry's Windows host, under HF account **`SceneWorks`**. See "Authenticated reads" below. |
| **Read and signed off by a human** | **_(NOT YET — Michael)_** |
| Sign-off date | **_(pending)_** |

## Who read what, and when — plainly

The story's definition of done asks that a note record who read what and when. The honest record is:

> An **automated agent** fetched sixteen primary-source licence texts on 2026-08-02, extracted a
> verbatim quote for every term it proposed, and transcribed those terms into Rust consts. **No
> human being has yet read a licence text or checked a quote.** Nothing here has been reviewed by a
> lawyer and none of it is legal advice.

The values are landed rather than held back because the surface is disclosure-only (below), so a
wrong value shows a user a wrong sentence rather than blocking anything — and because a landed table
carrying an explicit PROVISIONAL marker is easier to review than a branch nobody can compile.
Michael's review is a **quote check**: for each family, read the quote, decide whether it supports
the term, tick the box. When every box is ticked, change the status line at the top of this file and
delete the PROVISIONAL banner from the module docs in `families.rs`.

## What this document is, and what it is not

This records **facts**: the canonical text URL for each licence family, the identifier the upstream
declares verbatim, and a short verbatim quote of the operative clause behind every term landed for
that family. Per the epic's governing principle R2, it deliberately contains **no legal
conclusions** — no row says "therefore you may use this commercially". A quote is a fact; an
entitlement is not.

Every quote is reproduced under fair use for the purpose of review, is limited to the operative
clause relied on, and is attributed to the URL it came from. No licence is reproduced in full.

## Disclosure only — the constraint that governs every entry

**No licence term in this surface blocks, gates, or degrades any functionality.** The data exists so
a consumer can SHOW it to a user. Two consequences run through everything below:

1. A term records what a text **names**, never what a user may or may not do.
2. **Where a licence is silent, the surface is silent.** A non-commercial restriction on *use of the
   weights* is not transcribed as a restriction on *outputs* unless the text says so. Four families
   restrict weights and say nothing about outputs (`insightface-research-only`, `nvidia-nsclv1`,
   `apple-mlr`, `cc-by-nc-4-0`); none of them carries `NonCommercialOutputs`, and **no family in the
   table does**. That is a deliberate reversal of the story's draft for insightface and CC-BY-NC —
   see **U3**.

The typed vocabulary is fixed by the landed contract (`crates/contracts/gen-core/src/license.rs`,
`LicenseTerm`) and terms may only be drawn from it:
`AttributionRequired`, `NoticeFileRequired`, `NonCommercialWeights`, `NonCommercialOutputs`,
`RevenueCeiling{amount_usd, boundary}`, `RegistrationRequired{contact: Option}`,
`AcceptableUsePolicy{url: Option}`, `DeployerObligation{text}`, `DownstreamLicenseCopy{family}`,
`DownstreamRestrictions{family}`, `GatedAccess`.

> **The vocabulary changed after this note's first draft**, in each case because the texts forced
> it. sc-16898 split the bare `DownstreamFlowDown` into `DownstreamLicenseCopy` /
> `DownstreamRestrictions` (Q2), moved `GatedAccess` off `LicenseFamily` onto
> `ComponentLicense::gated` (U7), and gave `RevenueCeiling` a `CeilingBoundary` (U1 vs family 8).
> sc-16662 then made `AcceptableUsePolicy::url` and `RegistrationRequired::contact` `Option` (U2,
> U10). The family rows below have been rewritten onto the current shape: where the first pass wrote
> `DownstreamFlowDown`, the landed value names which of the two kinds it is.

---

# UNRESOLVED — needs Michael

Every item below is either **AMBIGUOUS** (the text does not settle it) or **NOT FOUND** (no
canonical text was reachable). None of them has been resolved by guessing. Where landing the consts
forced a choice, the choice is stated here and is **reversible in one commit** — none of it is
buried in the family rows.

**Update 2026-08-02 (second pass, authenticated).** U1 is **RESOLVED**; U2 is **narrowed** — one of
the four URLs was found and the other three are now positively established as having no address. A
new item **U10** was opened by the authenticated read.

**Update 2026-08-02 (review pass, sc-16662).** Three review findings closed against this note and
`families.rs`, none of them changing a term list: LTX-2's `DeployerObligation` string now carries the
source's own "…" instead of joining across it (family 8); `cc-by-nc-4-0`'s `NonCommercialWeights` is
now supported by §2(a)(1), the operative grant, rather than §1(i), which is only the definition of
the word (family 3); and **U11 is widened** from a Gemma-only question to the three-row decision it
actually is — Gemma, `nvidia-open-model` and Krea — with the candidate distinguisher written down.
The landed values stay **PROVISIONAL**.

**Update 2026-08-02 (transcription pass, sc-16662).** The consts landed. Every open item and the
disposition taken for it:

| # | item | disposition taken when landing | still needs Michael? |
| --- | --- | --- | --- |
| U1 | SD3.5's gated text | **CLOSED** — read under authentication; `stability-ai-community` sources from SD3.5's own file, ceiling `$1,000,000` **Exclusive** | no |
| U2 | `AcceptableUsePolicy` with no URL ×3 | **Contract amended.** `AcceptableUsePolicy::url` is now `Option<&'static str>`; OpenRAIL++, FLUX.1 [dev] and LTX-2 land `url: None` — "the licence names a policy and gives no address" is now representable. **No URL was invented**, and `https://blackforestlabs.ai/aup` (404, in no licence text) is asserted absent by a test | **confirm the convention** |
| U3 | `NonCommercialOutputs` from a use restriction | **Term NOT landed on any family.** No quote supports it anywhere. Reverses the draft for `insightface-research-only` and `cc-by-nc-4-0`. A test fails if it is re-added | **decide whether silence should read as restriction** |
| U4 | `nvidia-open-model` may have no checkpoint | Family **landed** (its text is real and was read). Whether any shipped checkpoint declares it is a *component-row* question, i.e. sc-16665 | **confirm before component rows land** |
| U5 | `candle-gen-sana/NOTICE` contradicted by source | Untouched — out of scope, flagged only | **needs a fix story** |
| U6 | Kolors repo-level licence | Untouched — a component question, not a family one | at sc-16665 |
| U7 | `GatedAccess` on family vs component | **CLOSED by sc-16898** — moved to `ComponentLicense::gated`. No family declares it; conformance rejects it and a test asserts it | no |
| U8 | Llama's 700M-MAU threshold has no variant | **Landed as `DeployerObligation`**, quoted verbatim, per the contract's own guidance. It is emphatically *not* laundered into a `RevenueCeiling` | **confirm** |
| U9 | Apache-2.0 §4(a) as flow-down | **Landed as `DownstreamLicenseCopy`.** The sc-16898 split answers the question the item posed: §4(a) is a copy-of-licence duty and Apache states no use restrictions to flow down, so the lighter variant fits exactly and the heavier one does not | **confirm** |
| U10 | LTX-2 ships two different texts | **Defaulted to the copy shipped beside the weights** (`Lightricks/LTX-2.3` `LICENSE`). Consequence: `RegistrationRequired{contact: None}` — the shipped text names no address. The GitHub copy's `https://ltx.io/model/licensing` is **not** transcribed. A test pins the choice | **DECIDE which text governs** |
| U11 | Is a notice clause *also* `AttributionRequired`? — Gemma **and** NVIDIA, plus Krea's naming duty | **Answered opposite ways by transcription order, which is the defect.** `gemma-terms` carries `NoticeFileRequired` only; `nvidia-open-model` derives **both** terms from its one §3 clause; `krea-2-community` carries `AttributionRequired` on a model-name-prefix duty. The candidate distinguisher is textual — NVIDIA says "attribution notice", Gemma says only "notice" — and it is stated in U11 rather than assumed | **DECIDE all three together** |
| Q2 | do the flow-downs differ? | **CLOSED by sc-16898** — two variants, each carrying its family. Four families state the heavier "restrictions as enforceable provisions" shape and eleven state the lighter "copy of the licence" shape; three state both | no |

## U1. SD3.5's own `LICENSE.md` — **RESOLVED. Read under authentication; cosmetic differences only.**

**Status: closed.** The first pass could not read the file (HTTP 401 anonymous) and *asserted* that
SD3.5's gated `LICENSE.md` was the same text as the ungated SVD-XT copy. That assertion has now been
tested directly and it **holds**.

| | |
| --- | --- |
| Read by | Claude (Opus 5), automated agent, on Michael Trefry's Windows host |
| HF account | **`SceneWorks`** (verified with `hf auth whoami` → `user: SceneWorks`) |
| Date | **2026-08-02** |
| Command | `hf download stabilityai/stable-diffusion-3.5-large LICENSE.md` |
| Local path | `E:\huggingface\hub\models--stabilityai--stable-diffusion-3.5-large\snapshots\ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f\LICENSE.md` |
| Revision | snapshot `ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f` — **confirmed equal to the repo's current `main` `sha`** via `https://huggingface.co/api/models/stabilityai/stable-diffusion-3.5-large` |
| Card metadata (unchanged) | `license_name: stabilityai-ai-community`, `license_link: LICENSE.md`, `gated: auto` |

Both files were then compared byte-for-byte against
`stabilityai/stable-video-diffusion-img2vid-xt` `LICENSE.md` (also re-fetched with the same CLI,
snapshot `9e43909513c6714f1bc78bcb44d96e733cd242aa`).

- SD3.5 `LICENSE.md`: **11,726 bytes**, LF-only, no BOM.
- SVD-XT `LICENSE.md`: **11,852 bytes**, LF-only, no BOM.
- Both carry the identical title line and date: `STABILITY AI COMMUNITY LICENSE AGREEMENT` /
  `Last Updated: July 5, 2024`.

### Verdict: **COSMETIC_DIFF — no substantive divergence**

The 126-byte delta is entirely typographic. After normalising curly quotes to straight, collapsing
whitespace, and mapping the section headings, both documents reduce to **36 lines each**, of which
only **3 still differ** — rows 4–6 below: two are punctuation around an **identical** URL, and the
third is a missing space. No word of operative text differs. Full inventory, including the three
differences that normalisation absorbs:

| # | difference | character of it |
| --- | --- | --- |
| 1 | headings: SD3.5 `I.`–`V.` (Roman) vs SVD-XT `1.`–`5.` (Arabic) | Formatting. Note both bodies cross-reference "Section III", "Section IV(a)", "Section V below", so **SD3.5's Roman numbering is the internally consistent one**; SVD-XT's Arabic headings disagree with its own cross-references |
| 2 | SD3.5 uses straight quotes `"` `'`; SVD-XT uses curly `“” ’` | Typographic |
| 3 | SVD-XT blank-line-separates the definitions; SD3.5 does not | Whitespace |
| 4 | AUP: SVD-XT `available at (https://stability.ai/use-policy),` vs SD3.5 `available at https://stability.ai/use-policy,` | **Same URL**, parentheses vs bare |
| 5 | Core Models: SVD-XT `available at (https://stability.ai/core-models)` vs SD3.5 `available at, https://stability.ai/core-models,` | **Same URL**, punctuation only |
| 6 | SD3.5 has a missing space: `including"fine tune"` vs SVD-XT `including “fine tune”` | Typo; same words |

### The revenue-ceiling clause is **byte-identical** in both files

Tested with case-sensitive string equality on the extracted sentence (887 characters in both):

> "If You are using or distributing the Stability AI Materials for a Commercial Purpose, You must
> register with Stability AI at (https://stability.ai/community-license). If at any time You or Your
> Affiliate(s), either individually or in aggregate, generate **more than USD $1,000,000** in annual
> revenue (or the equivalent thereof in Your local currency), regardless of whether that revenue is
> generated directly or indirectly from the Stability AI Materials or Derivative Works, any licenses
> granted to You under this Agreement shall terminate as of such date."

— `stabilityai/stable-diffusion-3.5-large` `LICENSE.md` §III, retrieved 2026-08-02 under HF account
`SceneWorks`. The same sentence, character for character, is in the SVD-XT file.

**Boundary wording confirmed EXCLUSIVE.** SD3.5 says **"generate more than USD $1,000,000 in annual
revenue"**. It does **not** say "at least". The string `at least` does not occur anywhere in either
file. The introductory paragraph is likewise identical in both and states the complementary side:

> "this Agreement  preserves free access to the Models for people or organizations  generating annual
> revenue of **less than US $1,000,000** (or local currency equivalent)."

— same source. So $1,000,000 exactly sits **below** the threshold under this licence — contrast
LTX-2's "at least $10,000,000", which is inclusive (family 8).

### Consequence

**The #5/#6 single-family merge stands.** `stable-video-diffusion-community` and
`stabilityai-ai-community` are two declared identifier strings over one licence text, and family 5's
terms are now sourced from **SD3.5's own file**, not from a proxy. Family 5's `text_url` has been
updated accordingly. No term in family 5 changes as a result of this read.

## U2. `AcceptableUsePolicy{url}` for four families — **NARROWED. One URL found; three confirmed to have none.**

The typed variant requires a `url`. On the second pass each of the four families was re-checked
against the upstream repo itself under authentication — licence text, `README.md`, the model card's
`extra_gated_prompt`, and the licensor's own GitHub. Result:

| family | outcome | address |
| --- | --- | --- |
| `krea-2-community` | ✅ **RESOLVED** | `https://www.krea.ai/krea-2-use-policy` |
| `creativeml-openrail-pp-m` | **inline, no canonical URL** (positively established) | — |
| `flux-1-dev-non-commercial` | **inline, no canonical URL**; the card's own cited policy file **does not exist** | — |
| `ltx-2-community` | **referenced but never defined and no address anywhere** | — |

### ✅ Krea 2 — RESOLVED: `https://www.krea.ai/krea-2-use-policy`

The address is not in the licence PDF; it is in the model card's gate prompt.
`krea/Krea-2-Turbo` `README.md` (retrieved 2026-08-02 with the authenticated CLI), YAML
`extra_gated_prompt`, verbatim:

> "…and acknowledge the [Acceptable Use Policy](https://www.krea.ai/krea-2-use-policy)."

That URL was fetched and **resolves: HTTP 200**. Its title is **"Krea Acceptable Use Policy"** and
its opening sentence scopes it to exactly the weights this repo loads:

> "This Acceptable Use Policy applies to all use of Krea 2 Raw model weights and Krea 2 Turbo model
> weights obtained through download, including any Derivatives and any Outputs generated from such
> weights."

— `https://www.krea.ai/krea-2-use-policy`, retrieved 2026-08-02. The page is dated **June 22, 2026**,
the same date as the Krea 2 Community License Agreement v.1, which corroborates that this is the
document §4.4 incorporates by reference. Family 7's row is updated.

### `creativeml-openrail-pp-m` (SDXL) — inline, no canonical URL

`stabilityai/stable-diffusion-xl-base-1.0` `LICENSE.md` (14,105 chars, retrieved 2026-08-02) contains
**zero `http(s)://` URLs of any kind** — verified by regex over the whole file. The restrictions are
enumerated in the licence's own Attachment A, and the body designates them as the operative
restriction set:

> "Use-based restrictions. The restrictions set forth in Attachment A are considered Use-based
> restrictions. Therefore You cannot use the Model and the Derivatives of the Model for the specified
> restricted uses."

— same source. There is no external policy to point at. **"Inline, no canonical URL" is the finding**,
not a gap.

### `flux-1-dev-non-commercial` — inline; and the card's cited `POLICY.md` **does not exist**

The licence text itself was re-read under authentication (see U10 note below: the gated HF copy is
**identical** to the GitHub copy the first pass used). Grepping the full 18,491-char text for
`acceptable use` / `prohibited use` / `use polic` / `aup` returns **no match at all** — the prohibited
uses are enumerated inline in §4.

`black-forest-labs/FLUX.1-dev` `README.md` YAML `extra_gated_prompt` does cite a policy, verbatim:

> "…and acknowledge the [Acceptable Use Policy](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/POLICY.md)."

**That file does not exist.** Three independent confirmations, all 2026-08-02:

1. `hf download black-forest-labs/FLUX.1-dev POLICY.md` → `Entry Not Found for url:
   https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/POLICY.md` (authenticated, so this
   is a genuine absence and not a gate refusal).
2. The repo's **complete** file listing from the HF API at `main` contains `LICENSE.md` and
   `README.md` and **no `POLICY.md`**.
3. `https://raw.githubusercontent.com/black-forest-labs/flux/main/POLICY.md` → **404**, as does
   `.../model_licenses/POLICY.md`.

So FLUX's gate asks you to acknowledge a document that is not published at the address it names.
**AMBIGUOUS** — recorded, not resolved. The honest entry for this family is a blank.

### `ltx-2-community` — referenced, never defined, no address

`Acceptable Use Policy` occurs **exactly once** in the whole LTX-2 licence, in Attachment A:

> "When using the Outputs, LTX-2 and any Derivatives thereof, you will comply with the Acceptable Use
> Policy. In addition, you agree not to use the Outputs, LTX-2 or its Derivatives in any of the
> following ways: …"

— `Lightricks/LTX-2.3` `LICENSE`, retrieved 2026-08-02. Note **"In addition"**: the enumerated list
that follows is presented as *additional to* the Acceptable Use Policy, so the AUP is not simply a
label for Attachment A — it reads as a separate document. But the term is capitalised without ever
being defined, and:

- the licence's definitions section never defines it;
- the only URL in the HF copy of the licence is `https://github.com/Lightricks/LTX-2`;
- the `Lightricks/LTX-2` GitHub repo root contains only `LICENSE`, `README.md`, `pyproject.toml`,
  `uv.lock` and dotfiles — **no policy document**;
- `https://raw.githubusercontent.com/Lightricks/LTX-2/main/POLICY.md` and
  `.../ACCEPTABLE_USE_POLICY.md` both **404**;
- `Lightricks/LTX-2.3` `README.md` never mentions a policy.

**AMBIGUOUS — dangling defined-term with no address.** Blank is the honest entry.

### The sc-16661 fixture's FLUX URL

The fixture guesses `AcceptableUsePolicy { url: "https://blackforestlabs.ai/aup" }`. Re-checked
2026-08-02: `https://blackforestlabs.ai/aup` → **HTTP 404**, and `https://bfl.ai/aup` → **HTTP 404**.
Neither string appears anywhere in the FLUX.1 [dev] licence. **There is no correct URL to substitute**
— see the FLUX subsection above. It is a fixture, not evidence, and must not be copied into the real
table.

### What is still Michael's to decide

For the three families with no address, the convention question is unchanged: point `url` at the
licence text itself (the restrictions *are* the policy), leave `AcceptableUsePolicy` off those
families, or relax the variant so the URL is optional. **Do not invent a URL** — a 404 in this table
is worse than an honest blank.

## U3. Whether `NonCommercialOutputs` follows from a non-commercial *use* restriction (AMBIGUOUS ×4)

Four families restrict *use of the model* to non-commercial/research purposes but say **nothing
about Outputs** — unlike FLUX.1 [dev] and CircleStone, which address Outputs explicitly and permit
commercial use of them. Whether silence means the restriction reaches renders is a legal read, not a
transcription:

- `insightface-research-only` — "available for non-commercial research purposes only"
- `nvidia-nsclv1` (PiD) — "The Work and any derivative works thereof only may be used or intended for use non-commercially"
- `apple-mlr` — "exclusively for Research Purposes"
- `cc-by-nc-4-0` — restricts Sharing/use of the Licensed Material; generated audio is arguably not "Adapted Material"

The draft table asserts `NonCommercialOutputs` for `insightface-research-only`. The evidence neither
supports nor refutes it. Note that `crates/media/mlx-gen/mlx-gen-pid/src/lib.rs` already states the
repo's working assumption for PiD — "The NC restriction flows to PiD-decoded output" — but that is a
SceneWorks engineering note, not a quote from NVIDIA.

## U4. The `nvidia-open-model` family may have **no checkpoint pointing at it** (NOT FOUND)

The draft assigns family 12 to "Cosmos-Predict2, PiD". Neither survives contact with the repo:

- **PiD is a different licence.** `nvidia/PiD`'s model card says it is **NSCLv1** (non-commercial),
  not the NVIDIA Open Model License (which is commercially usable). See family 12b below. These are
  two distinct texts and must not share a family id.
- **Cosmos-Predict2 appears only as an architecture reference.** `candle-gen-anima/src/config.rs`
  transcribes "the diffusers `Cosmos-2.0-Diffusion-2B-Text2Image` transformer config"; the weights
  actually loaded are `circlestone-labs/Anima`, under the CircleStone licence. No grep in
  `crates/media/` found a load path for NVIDIA Cosmos *weights*.

So `nvidia-open-model` is documented below (the text is real and was read), but **Michael must
confirm whether any shipped checkpoint is actually governed by it.** If not, it should not be in the
table — a family with no component is dead surface.

## U5. `candle-gen-sana/NOTICE` is contradicted by the primary source (NOT FOUND / in-repo defect)

`crates/media/candle-gen/candle-gen-sana/NOTICE` states SANA-1.6B and SANA-Sprint are "Licensed under
the NVIDIA License (the NVIDIA Open Model License family)" and links
`.../Sana_1600M_1024px_diffusers/blob/main/LICENSE.txt`.

- That URL **404s** (2026-08-02) — the file is `LICENSE`, not `LICENSE.txt`.
- `https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers/raw/main/LICENSE` is the
  **Apache License, Version 2.0** (HTTP 200, first line "Apache License / Version 2.0, January 2004").
- `https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers/raw/main/LICENSE`
  is likewise **Apache-2.0**. Both HF cards declare `license: apache-2.0`.

On the evidence, SANA's weights are `apache-2-0`, not an NVIDIA family. This is an existing in-repo
statement that needs correcting, and it is exactly the class of error the family table exists to
prevent. Flagged, not fixed — out of scope for this story.

## U6. Kolors declares `apache-2.0` at repo level while shipping a `MODEL_LICENSE` (AMBIGUOUS)

`Kwai-Kolors/Kolors` HF card metadata declares `license: apache-2.0`. But
`crates/media/mlx-gen/mlx-gen-kolors/src/convert.rs` copies a file named `MODEL_LICENSE` verbatim,
and the text encoder is **ChatGLM3-6B**, whose own `MODEL_LICENSE` requires commercial registration
(family 14). Which document governs the Kolors *checkpoint* (as opposed to the ChatGLM3 component)
was not settled from primary sources. The draft's assignment of `chatglm3-model-license` to "Kolors
text encoder" is supported for the **text encoder component**; the Kolors UNet/VAE component is
unresolved.

## U7. `GatedAccess` is a per-checkpoint distribution fact, not a licence-text fact — **CLOSED**

> **Resolved by sc-16898**, in the direction this item argued for: `GatedAccess` moved to
> `ComponentLicense::gated`, `license_table_conformance_errors` rejects it on a family, and
> `provider_terms` still raises it into the derived union. No family in the landed table declares
> it. The per-checkpoint table below stands as the evidence for the component rows (sc-16665).

Nothing in any licence text says "the weights are gated". Gating is a Hugging Face repo setting.
Measured 2026-08-02 via the HF API:

| checkpoint | `gated` |
| --- | --- |
| `stabilityai/stable-diffusion-3.5-large` | `auto` |
| `stabilityai/stable-video-diffusion-img2vid-xt` | **`False`** |
| `black-forest-labs/FLUX.1-dev` | `auto` |
| `krea/Krea-2-Turbo`, `krea/Krea-2-Raw` | `auto` |
| `nvidia/Cosmos-Predict2-2B-Text2Image` | `auto` |
| `google/gemma-2-2b-it` | `manual` |
| `Lightricks/LTX-2.3` | `False` |
| `circlestone-labs/Anima` | `False` |
| `nvidia/PiD` | `False` |

This is a problem for the type: SVD-XT and SD3.5 share **one licence text** but differ on gating, so
`GatedAccess` cannot be a property of a *family* without splitting families that are otherwise
identical. Michael must decide whether `GatedAccess` belongs on `LicenseFamily` at all or should move
to `ComponentLicense`. Below, gating is recorded per-checkpoint and **not** asserted as a family term.

## U8. Llama 3.1's threshold is 700M monthly active users — no typed variant fits (AMBIGUOUS, contract gap)

Family 16 (JoyCaption) gates additional commercial terms on *monthly active users*, not revenue.
`RevenueCeiling{amount_usd}` would be a false transcription. There is no `UserCeiling` variant.
Michael must decide: omit the term, widen the variant, or accept that this fact is unrepresentable.

## U9. Does Apache-2.0 §4(a) constitute a flow-down? — **landed as the copy variant; confirm**

> The sc-16898 split dissolves the dilemma this item poses. §4(a) is textually a
> *copy-of-licence* duty, and Apache-2.0 has no use restrictions for a downstream agreement to
> reproduce — so `DownstreamLicenseCopy` fits exactly and `DownstreamRestrictions` cannot apply.
> `apache-2-0` therefore landed with `DownstreamLicenseCopy{"apache-2-0"}`. **Michael: confirm,
> since it puts a flow-down on the most widely used family in the catalog.**

Apache-2.0 §4(a): *"You must give any other recipients of the Work or Derivative Works a copy of this
License"* — textually the same shape as the flow-down clauses in families 4/5/7/8/9/10/11/12/12b/15/16.
The draft assigns `DownstreamFlowDown` to none of the permissive families. If the term means "a copy
of the licence must reach downstream recipients", Apache-2.0 qualifies; if it means "you must impose
the *use restrictions* on downstream users as enforceable provisions", it does not (Apache has no use
restrictions). See Q2 below — this distinction is the crux.

## U10. LTX-2 ships **two different licence texts**, and they differ substantively (NEW — AMBIGUOUS)

Opened by the second (authenticated) pass. The first pass sourced family 8 from the GitHub copy,
because that is what the HF card's `license_link` points at. But `Lightricks/LTX-2.3` — the repo whose
weights this codebase actually loads — **commits its own `LICENSE`, and it is not the same text**.

| | HF: `Lightricks/LTX-2.3` `LICENSE` | GitHub: `Lightricks/LTX-2` `main` `LICENSE` |
| --- | --- | --- |
| size | 21,393 bytes | 21,461 bytes |
| licence date | `License date: January 5, 2026` | `License date: January 5, 2026` (same) |
| revenue ceiling | "at least $10,000,000" | "at least $10,000,000" (same) |
| **registration address** | **no URL** | `https://ltx.io/model/licensing` |
| **"Control" threshold** | **50% or more** (inclusive) | **more than 50%** (exclusive) |

Both differences are operative, not cosmetic.

**1. The `RegistrationRequired` URL is absent from the shipped copy.** Verbatim, same sentence in each:

> HF `LTX-2.3`: "…Commercial Entities interested in such a commercial license are required to contact
> Licensor."

> GitHub `LTX-2`: "…Commercial Entities interested in such a commercial license are required to
> [contact Licensor](https://ltx.io/model/licensing)."

— both retrieved 2026-08-02. So family 8's `RegistrationRequired{"https://ltx.io/model/licensing"}`
is supported **only** by the GitHub copy. The text distributed alongside the checkpoint names no
address at all. (`https://ltx.io/model/licensing` itself resolves, HTTP 200.)

**2. The `"Control"` definition flips its boundary**, which changes which entities aggregate as
Affiliates and therefore who is a "Commercial Entity" against the $10,000,000 ceiling:

> HF `LTX-2.3`: "\"Control\" means the direct or indirect ownership of **fifty percent (50%) or more**
> of the voting securities or other ownership interests…"

> GitHub `LTX-2`: "\"Control\" means the direct or indirect ownership of **more than fifty percent
> (50%)** of the voting securities or other ownership interests…"

— both retrieved 2026-08-02. At exactly 50%, the two texts disagree about whether an entity is
controlled.

**Which one governs is not settled by the evidence.** Both are upstream Lightricks publications: the
HF card's `license_link` field points at the GitHub blob, while the HF repo simultaneously commits a
different `LICENSE` beside the weights. **AMBIGUOUS** — Michael must decide which text family 8 is
transcribed from, and the answer determines whether `RegistrationRequired` has a URL at all. This is
the same defect class as U1, caught the same way; it is recorded rather than resolved.

## U11. Is a notice clause *also* `AttributionRequired`? — `gemma-terms`, `nvidia-open-model` and Krea's naming duty (AMBIGUOUS ×3)

**Widened by the sc-16662 review pass; decide all three rows here, in one read.**

The item was first opened as a Gemma-only question. It is not one. `nvidia-open-model` makes the
*same* modelling move in the *opposite* direction, from a clause of the same shape, and neither the
consts nor this note argued why. So U11 is currently answered one way for NVIDIA and deferred for
Gemma — by transcription order, not by a reading anyone took. A third row, Krea's, rests on a
different clause but the same class of judgement. **One decision settles three rows.**

Nothing here gates anything: `AttributionRequired` is a disclosure that the licence names an
attribution duty. The question is only whether the table says that about these texts.

### The two notice clauses, side by side

| | `nvidia-open-model` §3 | `gemma-terms` §3.1 |
| --- | --- | --- |
| **the clause, verbatim** | "include the following **attribution notice** within a \"Notice\" text file with such copies: \"Licensed by NVIDIA Corporation under the NVIDIA Open Model License\"" | "All Distributions (other than through a Hosted Service) must be accompanied by a \"Notice\" text file that contains the following **notice**: \"Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms\"." |
| **what landed** | `NoticeFileRequired` **and** `AttributionRequired` — two terms from the one clause | `NoticeFileRequired` **only** — the second term withheld |
| **the string the clause names** | "Licensed by NVIDIA Corporation under the NVIDIA Open Model License" — names the licensor | "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms" — names the terms, not a party |
| **carve-out** | none | Hosted Service distributions are excluded |

**The candidate distinguisher — stated so it can be accepted or rejected, not asserted.** NVIDIA's
clause calls the thing an *"attribution notice"*; Gemma's calls it a *"notice"* and never uses the
word attribution. On that hook the two treatments are defensible exactly as landed: the table would
be reporting each text's own vocabulary. The two supporting facts above point the same way — NVIDIA's
string names a party (the shape of attribution) while Gemma's names a document. **This argument is
made nowhere in the code and was made nowhere in this note before the review pass; it is offered
here as the leading candidate, not as a finding.** Both family consts now carry a doc comment
pointing at this item, so a reader of either lands on the open decision instead of on a silent
inconsistency.

The competing reading, equally available: "attribution notice" and "notice" are the same duty in
different words, one clause is one obligation, and the table should either derive both terms from
both clauses or from neither.

### The third row: Krea §3.1(b)

`krea-2-community` carries `AttributionRequired` on §3.1(b) — "include \"Krea\" at the beginning of
any such AI model name" — a **model-naming** duty, not a notice string. It is quoted, and
`AttributionRequired` is the closest variant the vocabulary carries, but "prefix the model name" is
not obviously the same disclosure as "reproduce this attribution string". (Krea's `NoticeFileRequired`
is a *separate* clause, §3.1(c), so this is not the one-clause-two-terms question — it is the
adjacent one: how far the variant stretches.) Same class of judgement, so it is decided here rather
than separately.

### The four ways this can be settled

| option | effect on the consts |
| --- | --- |
| **A — accept as landed** (the "attribution notice" hook governs) | no code change; U11 closes with the textual argument above recorded as the reason |
| **B — add `AttributionRequired` to `gemma-terms`** | one term added; the §3.1 Notice string is then read as attribution too |
| **C — drop `AttributionRequired` from `nvidia-open-model`** | one term removed; a notice clause yields only `NoticeFileRequired`, uniformly |
| **D — narrow Krea** | `AttributionRequired` comes off `krea-2-community`, leaving §3.1(c)'s `NoticeFileRequired`; or the variant's doc widens to say a naming duty counts |

Options B, C and D each change a term list, so each fails the **term census** in
`families.rs` by design and lands as a deliberate, reviewable edit. Option A costs nothing but a
tick.

---

# What landed in code — the 16 families at a glance

`crates/contracts/gen-core/src/license/families.rs`, one `pub const LicenseFamily` per row plus
`LICENSE_FAMILIES`. Every term below is backed by a quote in this document; the source comments in
`families.rs` carry the same quote beside the term. **All 16 landed; none was withheld.**

| # | family id | terms landed |
| --- | --- | --- |
| 1 | `apache-2-0` | `AttributionRequired`, `NoticeFileRequired`, `DownstreamLicenseCopy` |
| 2 | `mit` | `AttributionRequired` |
| 3 | `cc-by-nc-4-0` | `AttributionRequired`, `NonCommercialWeights` |
| 4 | `creativeml-openrail-pp-m` | `DownstreamRestrictions`, `AttributionRequired`, `NoticeFileRequired`, `AcceptableUsePolicy{None}` |
| 5 | `stability-ai-community` | `RevenueCeiling{1_000_000, Exclusive}`, `RegistrationRequired{Some}`, `AttributionRequired`, `NoticeFileRequired`, `DownstreamLicenseCopy`, `AcceptableUsePolicy{Some}` |
| 6 | `flux-1-dev-non-commercial` | `NonCommercialWeights`, `DownstreamLicenseCopy`, `DeployerObligation{content filters}`, `AcceptableUsePolicy{None}` |
| 7 | `krea-2-community` | `RevenueCeiling{1_000_000, Inclusive}`, `RegistrationRequired{Some}`, `DownstreamLicenseCopy`, `AttributionRequired`, `NoticeFileRequired`, `DeployerObligation` ×2, `AcceptableUsePolicy{Some}` |
| 8 | `ltx-2-community` | `RevenueCeiling{10_000_000, Inclusive}`, `RegistrationRequired{None}`, `DownstreamRestrictions`, `DownstreamLicenseCopy`, `AttributionRequired`, `NoticeFileRequired`, `DeployerObligation{exclusive relicensing}`, `AcceptableUsePolicy{None}` |
| 9 | `circlestone-labs-non-commercial` | `NonCommercialWeights`, `DownstreamLicenseCopy`, `AttributionRequired` |
| 10 | `gemma-terms` | `DownstreamRestrictions`, `DownstreamLicenseCopy`, `NoticeFileRequired`, `AcceptableUsePolicy{Some}` |
| 11 | `nvidia-open-model` | `DeployerObligation{guardrails}`, `DownstreamLicenseCopy`, `NoticeFileRequired`, `AttributionRequired`, `AcceptableUsePolicy{Some}` |
| 12 | `nvidia-nsclv1` | `NonCommercialWeights`, `DownstreamLicenseCopy`, `DownstreamRestrictions`, `AttributionRequired` |
| 13 | `insightface-research-only` | `NonCommercialWeights` |
| 14 | `chatglm3-model-license` | `RegistrationRequired{Some}`, `AttributionRequired` |
| 15 | `apple-mlr` | `NonCommercialWeights`, `DownstreamLicenseCopy`, `AttributionRequired` |
| 16 | `llama-3-1-community` | `DownstreamLicenseCopy`, `AttributionRequired`, `NoticeFileRequired`, `AcceptableUsePolicy{Some}`, `DeployerObligation{700M MAU}` |

## Terms deliberately NOT landed

| term | where the draft or a first-pass row proposed it | why it is absent |
| --- | --- | --- |
| `NonCommercialOutputs` | `insightface-research-only`, `cc-by-nc-4-0` (story draft); `nvidia-nsclv1`, `apple-mlr` (first pass, flagged) | **No quote, anywhere.** All four restrict *use of the weights* and are silent on outputs. Inferring a restriction from silence is a legal reading, not a transcription — **U3** |
| `GatedAccess` | FLUX, Krea, SD3.5, Gemma (story draft) | Not a licence-text fact. sc-16898 moved it to `ComponentLicense::gated`; conformance now rejects it on a family — **U7** |
| `AttributionRequired` on `gemma-terms` | first pass, flagged `(?)` | The §3.1 Notice string is the only attribution-shaped obligation; naming it twice is a modelling choice — **U11**. Note the mirror: `nvidia-open-model` **did** land both terms from its one §3 clause, and `krea-2-community` landed the term on a model-naming duty. U11 decides all three |
| `RegistrationRequired{Some("https://ltx.io/model/licensing")}` | first pass, from the GitHub copy | Not in the text shipped with the weights — **U10** |
| any `AcceptableUsePolicy` URL for OpenRAIL++, FLUX, LTX-2 | the sc-16661 fixture guessed `https://blackforestlabs.ai/aup` | That URL 404s and appears in no licence text. Landed as `url: None` — **U2** |

## SPDX ids

Three are already committed in `release/model-weight-licenses.json` and were reused verbatim:
`LicenseRef-Stability-AI-Community`, `LicenseRef-Gemma-Terms`, `LicenseRef-Apple-MLR`. `Apache-2.0`,
`MIT` and `CC-BY-NC-4.0` are real SPDX identifiers. The remaining ten have no SPDX entry, so a
`LicenseRef-…` id was minted for each (`LicenseRef-CreativeML-OpenRAIL-PP-M`,
`LicenseRef-FLUX-1-dev-Non-Commercial`, `LicenseRef-Krea-2-Community`, `LicenseRef-LTX-2-Community`,
`LicenseRef-CircleStone-Labs-Non-Commercial`, `LicenseRef-NVIDIA-Open-Model`,
`LicenseRef-NVIDIA-NSCLv1`, `LicenseRef-InsightFace-Research-Only`,
`LicenseRef-ChatGLM3-6B-Model-License`, `LicenseRef-Llama-3.1-Community`). A minted `LicenseRef-` is
a naming choice, not a claim about a text — but they are a compatibility boundary once published, so
**say now if you want different strings.**

## What the tests enforce

`crates/contracts/gen-core/src/license/families.rs` `mod tests` — nine tests:

- every const is in `LICENSE_FAMILIES`, the slice is sorted by id, and every id resolves;
- `license_table_conformance_errors` is empty over the whole family set;
- no family declares `GatedAccess`;
- **no family declares `NonCommercialOutputs`** (with a message pointing back at this document);
- Stability `Exclusive` vs LTX-2 `Inclusive` vs Krea `Inclusive` at the same amount as Stability;
- the two flow-down shapes are held apart by exact family lists, and every flow-down names its own
  family;
- the addressless policies/registrations are exactly the four expected, and
  `blackforestlabs.ai` appears nowhere in the emitted manifest;
- LTX-2's `text_url` is the shipped copy and it carries **no** registration address;
- a **term census** pinning every family's exact term list.

The census is the tripwire for "no term without a quote". Full linkage is not mechanisable from the
crate — the quotes live in this Markdown file and the contracts crate reads no files at test time —
so adding, removing or re-parameterising any term fails the census and sends the author back here.

---

# Q1 — Does the Stable Video Diffusion community licence carry a revenue ceiling?

## ANSWER: **YES — USD $1,000,000 annual revenue.** SVD-XT is governed by the *Stability AI Community License Agreement*, the same text as SD3.5.

The SVD-XT model card points at `stability.ai/license` without stating a threshold, but the repo
**does** commit the governing text, and it is ungated:

- `https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/raw/main/LICENSE.md`
  — HTTP 200, 11,852 bytes, retrieved **2026-08-02**.

First lines, verbatim: `STABILITY AI COMMUNITY LICENSE AGREEMENT` / `Last Updated: July 5, 2024`.

The operative clause, verbatim from that file:

> "If You are using or distributing the Stability AI Materials for a Commercial Purpose, You must
> register with Stability AI at (https://stability.ai/community-license). If at any time You or Your
> Affiliate(s), either individually or in aggregate, generate more than USD $1,000,000 in annual
> revenue (or the equivalent thereof in Your local currency), regardless of whether that revenue is
> generated directly or indirectly from the Stability AI Materials or Derivative Works, any licenses
> granted to You under this Agreement shall terminate as of such date."

The same sentence (minus the inline URLs) appears at
`https://stability.ai/community-license-agreement`, also Last Updated July 5, 2024, retrieved
2026-08-02.

**On the original non-commercial research release.** The question anticipated that SVD's first
release used a non-commercial research licence and asked whether that was superseded. On the evidence
as of 2026-08-02, the `LICENSE.md` currently served from that repo's `main` **is** the Community
License, not a research licence. The transition itself was not researched — only the current state
was read. If the provenance of the change matters, that is a separate question.

## Consequence for the table: draft families #5 and #6 are ONE family, not two

`stable-video-diffusion-community` and `stabilityai-ai-community` are two **declared identifier
strings** for what is, on the evidence, the same licence *text*. A third, `stable-audio-community`,
is declared by `stabilityai/SAME-L`, and the six `stable-audio-3-*` rows already committed in
`release/model-weight-licenses.json` use `LicenseRef-Stability-AI-Community`. All of these should
collapse to **one** `LicenseFamily`, with the differing strings recorded per-component in
`ComponentLicense::declared` — which is exactly what that field is for.

**No longer subject to U1.** SD3.5's own gated `LICENSE.md` has since been read under authentication
(HF account `SceneWorks`, 2026-08-02) and its revenue-ceiling clause is **byte-identical** to
SVD-XT's. The merge is confirmed on primary sources for both members. See U1.

---

# Q2 — Are the three `DownstreamFlowDown` obligations the same obligation?

> **RESOLVED and shipped.** sc-16898 split the bare variant into `DownstreamLicenseCopy{family}`
> and `DownstreamRestrictions{family}` along exactly the line this section draws. Four families
> state the heavier restrictions-as-enforceable-provisions shape (`creativeml-openrail-pp-m`,
> `gemma-terms`, `ltx-2-community`, `nvidia-nsclv1`) and eleven state the lighter copy-of-licence
> shape; three state **both**. Carrying the family id means eleven distinct duties stay eleven
> elements of a union instead of deduping to one.

## ANSWER: **MATERIALLY_DIFFERENT.** And the problem is bigger than three families — **eleven** of the sixteen impose a flow-down, in at least three structurally distinct shapes.

The bare `DownstreamFlowDown` variant cannot tell them apart. The landed contract already
anticipates this in its doc comment ("several licences impose this with materially different texts,
so a union containing it more than once is not redundant at the product layer") — but a union
literally *cannot* contain it more than once, because the variant carries no payload and dedupes.
**The evidence points at `DownstreamFlowDown` needing to carry the family id** (or an equivalent
discriminator), so that a product union preserves one entry per distinct obligation.

## The three clauses the story asked about

**#4 CreativeML Open RAIL++-M** — source:
`https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/LICENSE.md`, Section III,
retrieved 2026-08-02:

> "Use-based restrictions as referenced in paragraph 5 MUST be included as an enforceable provision
> by You in any type of legal agreement (e.g. a license) governing the use and/or distribution of
> the Model or Derivatives of the Model, and You shall give notice to subsequent users You Distribute
> to, that the Model or Derivatives of the Model are subject to paragraph 5."

**#9 LTX-2 Community** — source:
`https://raw.githubusercontent.com/Lightricks/LTX-2/main/LICENSE` (License date: January 5, 2026),
§3(a)–(b), retrieved 2026-08-02:

> "(a) Use-based restrictions as referenced in paragraph 4 and all provisions of Attachment A MUST be
> included as an enforceable provision by you in any type of legal agreement (e.g. a license)
> governing the use and/or distribution of LTX-2 or Derivatives of LTX-2 …"

> "(b) … Any Derivative of LTX-2 … must be distributed exclusively under the terms of this Agreement
> with a complete copy of this license included;"

**#11 Gemma Terms** — source: `https://ai.google.dev/gemma/terms`, §3.1, retrieved 2026-08-02:

> "You must include the use restrictions referenced in Section 3.2 as an enforceable provision in any
> agreement (e.g., license agreement, terms of use, etc.) governing the use and/or distribution of
> Gemma or Model Derivatives and you must provide notice to subsequent users you Distribute to that
> Gemma or Model Derivatives are subject to the use restrictions in Section 3.2."

## What actually differs

1. **The text that must be passed on is different in each case.** OpenRAIL++ requires paragraph 5's
   restrictions; LTX-2 requires paragraph 4 **plus the whole of Attachment A**; Gemma requires
   §3.2, which incorporates the *Gemma Prohibited Use Policy* at
   `ai.google.dev/gemma/prohibited_use_policy` by reference — an externally hosted, unilaterally
   updatable document. A single downstream agreement satisfying one does not satisfy the others;
   SceneWorks must reproduce three different bodies of restriction.
2. **LTX-2 additionally forbids relicensing derivatives.** "must be distributed exclusively under the
   terms of this Agreement" is a copyleft-shaped constraint that neither OpenRAIL++ nor Gemma
   imposes — OpenRAIL++ and Gemma both explicitly permit "additional or different license terms …
   provided" the restrictions survive.
3. **The scope of "downstream" differs.** OpenRAIL++ and LTX-2 define Distribution to include hosted
   service / API access ("You may host for third parties remote access purposes (e.g.
   software-as-a-service)"), so an API consumer counts. Gemma §3.1 carves Hosted Services **out** of
   the Notice-file requirement ("All Distributions (other than through a Hosted Service) must be
   accompanied by a 'Notice' text file"), so the two obligations bind different populations of
   SceneWorks users.

## The other eight

Flow-down is far more widespread than the draft assumes. Each of these was read on 2026-08-02 and
each imposes a flow-down the draft table does not record:

| family | clause shape | verbatim fragment |
| --- | --- | --- |
| `stability-ai-community` (5) | copy of agreement | "you shall: (i) provide a copy of this Agreement to that third party" |
| `flux-1-dev-non-commercial` (7) | copy + direct grant | "you must make available a copy of this License to third-party recipients … and specify that any rights … shall be directly granted by Company to said third-party recipients" |
| `krea-2-community` (8) | copy + bind recipient | "provide a copy of this Agreement and require each recipient to be bound by the Terms of this Agreement" |
| `circlestone-labs-non-commercial` (10) | copy + direct grant (FLUX-shaped) | "you must make available a copy of this License to third-party recipients of the CircleStone Models and/or Derivatives you Distribute" |
| `nvidia-open-model` (11) | copy + Notice | "You must give any other recipients of the Model a copy of this Agreement and include the following attribution notice within a 'Notice' text file" |
| `nvidia-nsclv1` (12b) | copy + use-limit must survive | "(b) you include a complete copy of this license with your distribution" / "Your Terms provide that the use limitation in Section 3.3 applies to your derivative works" |
| `apple-mlr` (15) | copy + attribution string | "you must provide a copy of this Agreement to such third party, and ensure that the following attribution notice be provided" |
| `llama-3-1-community` (16) | copy + "Built with Llama" | "you shall (A) provide a copy of this Agreement with any such Llama Materials" |
| `apache-2-0` (1) | copy only — see U9 | "You must give any other recipients of the Work or Derivative Works a copy of this License" |

Two structurally different obligations are visible across that set:

- **"copy-of-licence" flow-down** (Stability, FLUX, Krea, CircleStone, NVIDIA ×2, Apple, Llama,
  Apache) — hand the downstream recipient the licence.
- **"restrictions-as-enforceable-provisions" flow-down** (OpenRAIL++, LTX-2, Gemma) — write the use
  restrictions into *your own* agreement with your users, as terms you must be able to enforce.

The second is a materially heavier engineering and legal obligation than the first, and a bare
variant erases the distinction. **Recommendation for Michael's decision, not a decision:** either
`DownstreamFlowDown { family: &'static str }`, or split into two variants along the line above.

---

# The families

Count: **16** (the draft's 14, minus 1 merge, plus 3 new). Reconciliation:

| change | detail |
| --- | --- |
| **−1 merge** | draft #6 `stable-video-diffusion-community` collapses into #5 `stability-ai-community` — same text (Q1) |
| **−1 split-out, +2** | draft #12 `nvidia-open-model` covers two different licences: NVIDIA Open Model License (Cosmos-Predict2, see U4) **and** NVIDIA NSCLv1 (PiD). Recorded as 11 and 12b |
| **+1 new** | `apple-mlr` — already shipping in `release/model-weight-licenses.json` (MMAudio's DFN5B-CLIP conditioner, 2 rows) and absent from the draft |
| **+1 new** | `llama-3-1-community` — JoyCaption, declared in `release/real-weight-models.toml` and absent from the draft |

14 − 1 + 1 + 1 + 1 = **16**.

For each family: `[ ]` = unreviewed. Michael ticks when the quote supports the term.

---

## 1. `apache-2-0` — Apache License 2.0

- **text_url**: `https://www.apache.org/licenses/LICENSE-2.0.txt` (HTTP 200, retrieved 2026-08-02)
- **declared upstream**: `apache-2.0` (HF card `license` field)
- **In-catalog checkpoints (partial)**: Z-Image-Turbo, Qwen-Image family, Mochi-1, SANA + SANA-Sprint
  (see U5), Kokoro-82M, Whisper, CLAP, the MOSS family, SmolLM2, Qwen3, IP-Adapter, krea-realtime-video

| ✔ | term | verbatim support |
| --- | --- | --- |
| [ ] | `AttributionRequired` | §4(c) "You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the Source form of the Work" |
| [ ] | `NoticeFileRequired` | §4(d) "If the Work includes a \"NOTICE\" text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file" |
| [ ] | `DownstreamLicenseCopy{"apache-2-0"}` | §4(a) "You must give any other recipients of the Work or Derivative Works a copy of this License" — **LANDED.** The sc-16898 split settles **U9**: this is a copy-of-licence duty, and Apache-2.0 states no use restrictions for a downstream agreement to reproduce, so the heavier `DownstreamRestrictions` does not fit. **Confirm.** |

Matches the draft, plus the §4(a) flow-down the draft omitted (U9, now landed as the copy variant).

---

## 2. `mit` — MIT License

- **text_url**: `https://raw.githubusercontent.com/spdx/license-list-data/main/text/MIT.txt`
  (SPDX canonical text, HTTP 200, retrieved 2026-08-02)
- **declared upstream**: `mit` (HF card `license` field)
- **In-catalog checkpoints (partial)**: ACE-Step v1.5 family, Chatterbox + Chatterbox-VE, OpenVoice V2,
  Mage-Flow family, CLIP ViT-L/14, CLIP ViT-bigG, sdxl-vae-fp16-fix, Synchformer, NVIDIA BigVGAN v2

| ✔ | term | verbatim support |
| --- | --- | --- |
| [ ] | `AttributionRequired` | "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software." |

Matches draft.

---

## 3. `cc-by-nc-4-0` — Creative Commons Attribution-NonCommercial 4.0 International

- **text_url**: `https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt` (HTTP 200, retrieved 2026-08-02)
- **declared upstream**: `cc-by-nc-4.0`
- **In-catalog checkpoints**: the five MMAudio rows already committed in
  `release/model-weight-licenses.json` (MM-DiT large_44k_v2 and small_16k, VAE 16k/44k, BigVGAN 16k)

| ✔ | term | verbatim support |
| --- | --- | --- |
| [ ] | `AttributionRequired` | §3(a)(1) "If You Share the Licensed Material (including in modified form), You must: … retain … identification of the creator(s) of the Licensed Material …; a copyright notice; a notice that refers to this Public License" |
| [ ] | `NonCommercialWeights` | §2(a)(1) — **the operative grant, bounded on its face**: "the Licensor hereby grants You a worldwide, royalty-free, non-sublicensable, non-exclusive, irrevocable license to exercise the Licensed Rights in the Licensed Material to: a. reproduce and Share the Licensed Material, in whole or in part, **for NonCommercial purposes only**; and b. produce, reproduce, and Share Adapted Material **for NonCommercial purposes only**." Supporting definition, §1(i): "NonCommercial means not primarily intended for or directed towards commercial advantage or monetary compensation." **§1(i) alone was the first pass's only support and is not a restriction** — it defines a word; §2(a)(1) is the clause that bounds the grant, so it is what `families.rs` now quotes beside the term. Every other `NonCommercialWeights` row in this document quotes an operative restriction (FLUX §2(b), CircleStone §2(b), NSCLv1 §3.3, Apple MLR §1, InsightFace README), and this row now does too |
| — | ⚠ `NonCommercialOutputs` | **NOT LANDED** — **no supporting quote found, see U3.** The licence governs Sharing and use of the Licensed Material; it says nothing about material generated by running a model |

Draft asserts both NC terms; only `NonCommercialWeights` is quotable, and only it is landed.

---

## 4. `creativeml-openrail-pp-m` — CreativeML Open RAIL++-M License

- **text_url**: `https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/LICENSE.md`
  (HTTP 200, 14,109 bytes, retrieved 2026-08-02). Header: "Copyright (c) 2023 Stability AI /
  CreativeML Open RAIL++-M License dated July 26, 2023"
- **declared upstream**: `openrail++` (HF card `license` field)
- **In-catalog checkpoints**: `stabilityai/stable-diffusion-xl-base-1.0`, `SG161222/RealVisXL_V5.0`
  (both in `release/real-weight-models.toml`); consumed by `mlx-gen-sdxl` / `candle-gen-sdxl` and
  transitively by `mlx-gen-instantid`, `mlx-gen-kolors`
- **gated**: `False`

| ✔ | term | verbatim support |
| --- | --- | --- |
| [ ] | `DownstreamRestrictions{"creativeml-openrail-pp-m"}` | Section III "Use-based restrictions as referenced in paragraph 5 MUST be included as an enforceable provision by You in any type of legal agreement … and You shall give notice to subsequent users You Distribute to, that the Model or Derivatives of the Model are subject to paragraph 5." |
| [ ] | `AttributionRequired` | Section III "You must retain all copyright, patent, trademark, and attribution notices excluding those notices that do not pertain to any part of the Model" |
| [ ] | `NoticeFileRequired` | Section III "You must cause any modified files to carry prominent notices stating that You changed the files" |
| [ ] | `AcceptableUsePolicy{url: None}` | **LANDED.** **Inline, no canonical URL.** The file contains **zero URLs of any kind**. Body: "The restrictions set forth in Attachment A are considered Use-based restrictions. Therefore You cannot use the Model and the Derivatives of the Model for the specified restricted uses." — **see U2** |

Draft has `AcceptableUsePolicy, DownstreamFlowDown`. Evidence adds `AttributionRequired` and
`NoticeFileRequired`; the AUP URL is unresolved. The draft's note "commercial permitted" is a
conclusion, not a term — correctly absent from `terms`.

---

## 5. `stability-ai-community` — Stability AI Community License Agreement

- **text_url**: `https://huggingface.co/stabilityai/stable-diffusion-3.5-large/raw/main/LICENSE.md`
  — **SD3.5's own file, 11,726 bytes, read 2026-08-02 with the authenticated HF CLI under account
  `SceneWorks`** at revision `ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f` (= current `main`).
  Corroborated by two independent copies with **byte-identical operative clauses**:
  `https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/raw/main/LICENSE.md`
  (HTTP 200, 11,852 bytes, ungated) and `https://stability.ai/community-license-agreement`
  (HTTP 200). All three: "STABILITY AI COMMUNITY LICENSE AGREEMENT / Last Updated: July 5, 2024".
  Differences between the SD3.5 and SVD-XT files are typographic only — **see U1** for the full diff.
- **declared upstream — three different strings, one text**:
  - `stable-video-diffusion-community` (`stabilityai/stable-video-diffusion-img2vid-xt`)
  - `stabilityai-ai-community` (`stabilityai/stable-diffusion-3.5-large`) — ✅ **U1 resolved**
  - `stable-audio-community` (`stabilityai/SAME-L`)
  - plus `LicenseRef-Stability-AI-Community` already committed for the six `stable-audio-3-*` rows
- **In-catalog checkpoints**: SVD-XT (`mlx-gen-svd`, `candle-gen-svd`); SD3.5 large / large-turbo /
  medium (`mlx-gen-sd3`, `candle-gen-sd3`); the six `stable-audio-3-*` audio providers; SAME-S/SAME-L
- **gated**: SVD-XT `False`; SD3.5-large `auto` — **see U7**

| ✔ | term | verbatim support |
| --- | --- | --- |
| [ ] | `RevenueCeiling{1_000_000, Exclusive}` | §III "If at any time You or Your Affiliate(s), either individually or in aggregate, generate **more than** USD $1,000,000 in annual revenue … any licenses granted to You under this Agreement shall terminate as of such date." — **boundary is EXCLUSIVE**; the string "at least" appears nowhere in either file. Verified byte-identical in SD3.5's own file and SVD-XT's (U1). Cf. the intro: "free access … for people or organizations  generating annual revenue of less than US $1,000,000" |
| [ ] | `RegistrationRequired{Some("https://stability.ai/community-license")}` | §III "If You are using or distributing the Stability AI Materials for a Commercial Purpose, You must register with Stability AI at (https://stability.ai/community-license)." |
| [ ] | `AttributionRequired` | §IV(a) "prominently display \"Powered by Stability AI\" on a related website, user interface, blogpost, about page, or product documentation" |
| [ ] | `NoticeFileRequired` | §IV(a) "retain the following attribution notice within a \"Notice\" text file distributed as a part of such copies: \"This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved\"" |
| [ ] | `DownstreamLicenseCopy{"stability-ai-community"}` | §IV(a) "If You distribute or make available the Stability AI Materials or a Derivative Work to a third party, or a product or service that uses any portion of them, You shall: (i) provide a copy of this Agreement to that third party" |
| [ ] | `AcceptableUsePolicy{Some("https://stability.ai/use-policy")}` | §V "\"AUP\" means the Stability AI Acceptable Use Policy available at (https://stability.ai/use-policy), as may be updated from time to time." |

**Draft correction — the largest in the table.** Draft #5 (SD3.5) had
`RevenueCeiling{1_000_000}, AcceptableUsePolicy, AttributionRequired` and draft #6 (SVD-XT) had
`AcceptableUsePolicy, AttributionRequired` with the ceiling flagged as unknown. On the evidence:
the two are one family, both carry the ceiling, and **both additionally carry
`RegistrationRequired`, `NoticeFileRequired`, and `DownstreamFlowDown`, which neither draft row had.**

---

## 6. `flux-1-dev-non-commercial` — FLUX.1 [dev] Non-Commercial License v1.1.1

- **text_url**: `https://raw.githubusercontent.com/black-forest-labs/flux/main/model_licenses/LICENSE-FLUX1-dev`
  (HTTP 200, 18,491 bytes, retrieved 2026-08-02). First line: "FLUX.1 [dev] Non-Commercial License v1.1.1".
  The HF copy at `black-forest-labs/FLUX.1-dev/raw/main/LICENSE.md` returns **HTTP 401** anonymously,
  but was **read 2026-08-02 with the authenticated HF CLI** (account `SceneWorks`, revision
  `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`) and is **identical to the GitHub copy** — same 18,491
  bytes, zero differences after whitespace normalisation. The gated-vs-proxy risk flagged for SD3.5
  in U1 does **not** materialise here either.
- **declared upstream**: `flux-1-dev-non-commercial-license` (HF card `license_name`)
- **In-catalog checkpoints**: `black-forest-labs/FLUX.1-dev` (`mlx-gen-flux`, `candle-gen-flux`),
  FLUX.1-Krea-dev (same `license_name`), and transitively `mlx-gen-pulid` / `candle-gen-pulid`
  (PuLID-FLUX rides the FLUX.1-dev backbone)
- **gated**: `auto`

| ✔ | term | verbatim support |
| --- | --- | --- |
| [ ] | `NonCommercialWeights` | §2(b) "You may only access, use, Distribute, or create Derivatives of the FLUX.1 [dev] Model or Derivatives for Non-Commercial Purposes." |
| [ ] | *(NOT `NonCommercialOutputs`)* | §2(d) "You may use Output for any purpose (including for commercial purposes), except as expressly prohibited herein." — draft is **correct** to exclude it |
| [ ] | `DownstreamLicenseCopy{"flux-1-dev-non-commercial"}` | §3(a) "you must make available a copy of this License to third-party recipients of the FLUX.1 [dev] Models and/or Derivatives you Distribute, and specify that any rights to use the FLUX.1 [dev] Models and/or Derivatives shall be directly granted by Company to said third-party recipients pursuant to this License" |
| [ ] | `DeployerObligation{...}` *(landed text = the quote at right, verbatim)* | §2(e) "implement and maintain content filtering measures (\"Content Filters\") for your use of the FLUX.1 [dev] Model or Derivatives to prevent the creation, display, transmission, generation, or dissemination of unlawful or infringing content" |
| [ ] | `AcceptableUsePolicy{url: None}` | **LANDED.** **Inline, no canonical URL.** Prohibited uses are inline in §4; the strings "acceptable use" / "use policy" / "AUP" do **not occur anywhere** in the 18,491-char text. The model card's gate prompt cites `.../FLUX.1-dev/blob/main/POLICY.md`, but **that file does not exist** (authenticated `hf download` → "Entry Not Found"; absent from the repo's complete file list) — **see U2** |
| — | ⚠ `GatedAccess` | **NOT LANDED** **on the family** — not a licence-text fact; HF `gated: auto`. Moved to `ComponentLicense::gated` by sc-16898; **U7 closed** |

**Draft correction.** Draft #7 had `NonCommercialWeights, GatedAccess, AcceptableUsePolicy`. Evidence
adds **`DownstreamFlowDown`** and **`DeployerObligation{content filtering}`** — the draft assigned
content filtering only to Krea. The draft's note that outputs are commercial-OK is confirmed.

---

## 7. `krea-2-community` — Krea 2 Community License Agreement v.1

- **text_url**: `https://cdn.jsdelivr.net/gh/krea-ai/krea-2@db3984fbc6e13b34c0064990fc2d95ac64d00058/assets/hf_samples/LICENSE.pdf`
  (HTTP 200, 137,711 bytes, PDF, retrieved 2026-08-02; this exact URL is the `license_link` in the HF
  card metadata). Header: "KREA 2 COMMUNITY LICENSE AGREEMENT / KREA 2 Community License Agreement
  v.1, Date: June 22, 2026". The HF `LICENSE.pdf` in-repo returns **HTTP 401** (gated).
- **declared upstream**: `krea-2-community-license` (HF card `license_name`)
- **In-catalog checkpoints**: `krea/Krea-2-Turbo`, `krea/Krea-2-Raw` (`mlx-gen-krea`, `candle-gen-krea`;
  registered in both catalogs)
- **gated**: `auto`

| ✔ | term | verbatim support |
| --- | --- | --- |
| [ ] | `RevenueCeiling{1_000_000, Inclusive}` | §2.3 "Commercial Use under this Agreement of the Krea Model, Derivatives, or Outputs is permitted only if you … have total company-wide annual revenue of **less than** one million United States dollars ($1,000,000 USD), calculated on a trailing twelve-month basis", and from the other side: "If you **meet or exceed** this threshold, you must obtain a separate enterprise license". "meet or exceed" puts $1,000,000 exactly **at** the threshold, so the boundary is **INCLUSIVE** — the same amount as Stability with the opposite reading. **This boundary determination is new in the transcription pass; check it first.** |
| [ ] | `RegistrationRequired{Some("opensource@krea.ai")}` | §2.3 "If you meet or exceed this threshold, you must obtain a separate enterprise license from Krea prior to any Commercial Use. … Enterprise license inquiries may be directed to opensource@krea.ai." |
| [ ] | `DownstreamLicenseCopy{"krea-2-community"}` | §3.1 "you shall (a) provide a copy of this Agreement and require each recipient to be bound by the Terms of this Agreement" |
| [ ] | ⚠ `AttributionRequired` | §3.1(b) "include \"Krea\" at the beginning of any such AI model name" — a **model-naming** duty, not a notice string, recorded as the closest variant the vocabulary carries. Whether the variant stretches that far is the same class of judgement as the NVIDIA/Gemma notice clauses — **decided with them in U11** |
| [ ] | `NoticeFileRequired` | §3.1(c) "retain the following attribution notice within a \"Notice\" text file distributed as part of such copies: \"Krea 2 is licensed under the Krea 2 Community License Agreement.\"" |
| [ ] | `DeployerObligation{...}` *(landed text = the quote at right, verbatim)* | §4.2 "You must implement reasonable and appropriate Content Filter measures to detect, prevent, and mitigate the generation or distribution of prohibited, harmful, or unlawful content through your deployment of the Krea Model or any Derivative." |
| [ ] | `DeployerObligation{...}` *(landed text = the quote at right, verbatim)* | §4.3 "Where required by applicable law, regulation, or platform policy, you must clearly disclose that Outputs were generated using artificial intelligence." |
| [ ] | ✅ `AcceptableUsePolicy{Some("https://www.krea.ai/krea-2-use-policy")}` | §4.4 "You must comply with the Acceptable Use Policy, which is incorporated herein by reference." The address is **not** in the PDF but **is** in `krea/Krea-2-Turbo`'s model-card gate prompt: "acknowledge the [Acceptable Use Policy](https://www.krea.ai/krea-2-use-policy)". **URL resolves, HTTP 200**, titled "Krea Acceptable Use Policy", dated June 22, 2026 (= the licence's own date), and scoped to "all use of Krea 2 Raw model weights and Krea 2 Turbo model weights obtained through download". **U2 resolved for this family** |
| — | ⚠ `GatedAccess` | **NOT LANDED** **on the family** — HF `gated: auto`; moved to `ComponentLicense::gated`, U7 closed |

**Draft correction.** Draft #8 had `GatedAccess, AcceptableUsePolicy, DeployerObligation{content
filtering}`. Evidence adds **`RevenueCeiling{1_000_000}`** (the draft had no ceiling for Krea at all),
**`RegistrationRequired`**, **`DownstreamFlowDown`**, **`AttributionRequired`**, **`NoticeFileRequired`**,
and a second `DeployerObligation`.

> Note: `candle-gen-krea/src/lib.rs` currently states "Krea 2 Community License (non-commercial use
> satisfies it)". That is a conclusion, and an incomplete one — §2.3 permits *commercial* use below
> the $1M threshold. Not fixed here; flagged.

---

## 8. `ltx-2-community` — LTX-2 Community License Agreement

- **text_url — ⚠ TWO upstream texts exist and they differ; see U10. LANDED VALUE:**
  **`https://huggingface.co/Lightricks/LTX-2.3/raw/main/LICENSE`** (21,393 bytes, retrieved
  2026-08-02) — the copy committed **beside the weights this repo loads**, i.e. the text a user who
  downloads the checkpoint actually receives. That is the tie-breaker applied; it is a choice, not a
  finding, and **Michael must confirm or overrule it**.
  - The rejected alternative: `https://raw.githubusercontent.com/Lightricks/LTX-2/main/LICENSE`
    (HTTP 200, 21,461 bytes, retrieved 2026-08-02) — the `license_link` in the HF card metadata.
    Same title and date. It differs in exactly two operative places: it links the registration
    address the shipped copy omits, and it inverts the `"Control"` threshold.
  - **Consequence of the choice**: `RegistrationRequired` lands with **no address**
    (`contact: None`), because the shipped text names none. `https://ltx.io/model/licensing` is
    **not** transcribed. Choosing the GitHub copy instead would add that URL and flip `"Control"`
    from "50% or more" to "more than 50%".
  - Every other quote below is unaffected: U10 established the two files agree everywhere else.

  Both headers: "LTX-2 Community License Agreement / License date: January 5, 2026"
- **declared upstream**: `ltx-2-community-license-agreement` (HF card `license_name`)
- **In-catalog checkpoints**: `Lightricks/LTX-2.3` (`mlx-gen-ltx`, `candle-gen-ltx`). Note the crate
  targets LTX-**2.3**, which declares the LTX-**2** community licence.
- **gated**: `False`

| ✔ | term | verbatim support |
| --- | --- | --- |
| [ ] | `RevenueCeiling{10_000_000, Inclusive}` | §2 "Entities with annual revenues of **at least** $10,000,000 (the \"Commercial Entities\") are required to obtain a paid commercial use license in order to use LTX-2 and Derivatives of LTX-2" — **boundary is INCLUSIVE**, and this clause is **identical in both upstream copies** (U10), so the ceiling itself is not in doubt |
| [ ] | ⚠ `RegistrationRequired{contact: None}` | **LANDED.** The shipped `LTX-2.3` copy reads §2 "Commercial Entities interested in such a commercial license are required to contact Licensor." — **no URL**. The GitHub copy reads "…required to [contact Licensor](https://ltx.io/model/licensing)" and that URL resolves (HTTP 200), but it is **not in the text shipped with the weights**, so it is not transcribed. **See U10 — this is the term the text choice decides.** |
| [ ] | `DownstreamRestrictions{"ltx-2-community"}` | §3(a) "Use-based restrictions as referenced in paragraph 4 and all provisions of Attachment A MUST be included as an enforceable provision by you in any type of legal agreement … governing the use and/or distribution of LTX-2" |
| [ ] | `DownstreamLicenseCopy{"ltx-2-community"}` | §3(b) "You must provide any third party recipients of LTX-2 or Derivatives of LTX-2 a copy of this Agreement, including all attachments and use policies." — LTX-2 imposes **both** flow-down shapes, which stay two elements of a union |
| [ ] | `DeployerObligation{"Any Derivative of LTX-2 … must be distributed exclusively under the terms of this Agreement with a complete copy of this license included"}` | §3(b), **quoted with the source's own elision preserved**: "(b) … Any Derivative of LTX-2 … must be distributed exclusively under the terms of this Agreement with a complete copy of this license included". The landed string is that fragment from "Any" through "included", word for word, with the source's internal "…" preserved and only the clause-final semicolon dropped — so note and const now read identically. **It first landed without the "…"**, which silently joined across the elided subject clause and presented two fragments as one continuous sentence; since a consumer eventually shows this string to a user, the join misrepresented the source, and the ellipsis is restored. (Cf. `nvidia-open-model` §2 and `llama-3-1-community` §2, whose landed strings already carried theirs.) A copyleft-shaped constraint neither OpenRAIL++ nor Gemma states, so it is carried separately rather than folded into the flow-down |
| [ ] | `AttributionRequired` | §3(d) "You must retain all copyright, patent, trademark, and attribution notices excluding those notices that do not pertain to any part of LTX-2" |
| [ ] | `NoticeFileRequired` | §3(c) "You must cause any modified files to carry prominent notices stating that you changed the files" |
| [ ] | `AcceptableUsePolicy{url: None}` | **LANDED.** **Referenced, never defined, no address anywhere.** Attachment A "When using the Outputs, LTX-2 and any Derivatives thereof, you will comply with the Acceptable Use Policy. **In addition**, you agree not to use the Outputs…" — the "In addition" implies the AUP is a *separate* document from the enumerated list, yet the term is never defined, the licence's only URL is the GitHub repo, and that repo root holds no policy file — **see U2** |

**Draft correction.** Draft #9 had `RevenueCeiling{10_000_000}, AcceptableUsePolicy,
DownstreamFlowDown`. The ceiling amount is confirmed, but note the framing is **inverted** relative
to Stability's: Stability terminates the licence *above* $1,000,000; LTX-2 requires a paid licence
at *"at least"* $10,000,000 — i.e. inclusive of exactly $10M. If `RevenueCeiling` is documented as
"permitted below the ceiling", $10,000,000 exactly is *not* permitted here but $1,000,000 exactly
*is* permitted under Stability. Evidence adds `RegistrationRequired`, `AttributionRequired`,
`NoticeFileRequired`.

---

## 9. `circlestone-labs-non-commercial` — CircleStone Labs Non-Commercial License v1.2

- **text_url**: `https://huggingface.co/circlestone-labs/Anima/raw/main/LICENSE.md`
  (HTTP 200, 18,259 bytes, retrieved 2026-08-02). First line: "CircleStone Labs Non-Commercial License v1.2"
- **declared upstream**: `circlestone-labs-non-commercial-license` (HF card `license_name`)
- **In-catalog checkpoints**: `circlestone-labs/Anima` (`mlx-gen-anima`, `candle-gen-anima`; registered
  in both catalogs)
- **gated**: `False`

| ✔ | term | verbatim support |
| --- | --- | --- |
| [ ] | `NonCommercialWeights` | §2(b) "You may only access, use, Distribute, or create Derivatives of the CircleStone Model or Derivatives for Non-Commercial Purposes, unless otherwise expressly granted by this License." |
| [ ] | *(NOT `NonCommercialOutputs`)* | §2(e) "You may use Outputs for any purpose (including for commercial purposes), except as expressly prohibited herein." |
| [ ] | `DownstreamLicenseCopy{"circlestone-labs-non-commercial"}` | §3(a) "you must make available a copy of this License to third-party recipients of the CircleStone Models and/or Derivatives you Distribute, and specify that any rights to use the CircleStone Models and/or Derivatives shall be directly granted by Company to said third-party recipients pursuant to this License" |
| [ ] | `AttributionRequired` | §3 attribution string: "The CircleStone Model is licensed by CircleStone Labs LLC under the CircleStone Non-Commercial License. Copyright CircleStone Labs LLC." |

**Draft correction.** Draft #10 had only `NonCommercialWeights`. Evidence adds `DownstreamFlowDown`
and `AttributionRequired`, and confirms outputs are **not** restricted (this licence is textually a
close relative of FLUX.1 [dev]'s).

---

## 10. `gemma-terms` — Gemma Terms of Use

- **text_url**: `https://ai.google.dev/gemma/terms` (HTTP 200, retrieved 2026-08-02)
- **declared upstream**: `gemma` (HF card `license` field); the audio manifest already uses
  `LicenseRef-Gemma-Terms`
- **In-catalog checkpoints**: `google/gemma-3-12b-it` (LTX text encoder), `gemma-2-2b-it` (SANA CHI
  encoder and PiD caption encoder), `t5gemma` inside the six `stable-audio-3-*` providers (already
  committed, 6 rows)
- **gated**: `google/gemma-2-2b-it` → `manual`

| ✔ | term | verbatim support |
| --- | --- | --- |
| [ ] | `DownstreamRestrictions{"gemma-terms"}` | §3.1 "You must include the use restrictions referenced in Section 3.2 as an enforceable provision in any agreement … and you must provide notice to subsequent users you Distribute to that Gemma or Model Derivatives are subject to the use restrictions in Section 3.2." |
| [ ] | `DownstreamLicenseCopy{"gemma-terms"}` | §3.1 "You must provide all third party recipients of Gemma or Model Derivatives a copy of this Agreement." — the same section imposes **both** shapes |
| [ ] | `NoticeFileRequired` | §3.1 "All Distributions (other than through a Hosted Service) must be accompanied by a \"Notice\" text file that contains the following notice: \"Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms\"." |
| [ ] | `AcceptableUsePolicy{Some("https://ai.google.dev/gemma/prohibited_use_policy")}` | §3.2 "You must not use any of the Gemma Services: for the restricted uses set forth in the Gemma Prohibited Use Policy at ai.google.dev/gemma/prohibited_use_policy (\"Prohibited Use Policy\"), which is hereby incorporated by reference into this Agreement" |
| — | *(?)* `AttributionRequired` | **NOT LANDED** — **U11.** The §3.1 Notice string is the only attribution-shaped obligation; calling one obligation two terms is a modelling choice, not a transcription, so `gemma-terms` carries `NoticeFileRequired` only. **`nvidia-open-model` §3 makes the opposite call from a clause of the same shape** (family 11 below) — the two are decided together in U11, where the "attribution notice" vs "notice" hook is set out. **Decide.** |
| — | ⚠ `GatedAccess` | **NOT LANDED** **on the family** — `gemma-2-2b-it` is `gated: manual`; moved to `ComponentLicense::gated`, U7 closed |
| | *(no `NonCommercial*`)* | §3.3 "Google claims no rights in Outputs you generate using Gemma." |

**Matches the draft** (`AcceptableUsePolicy`, flow-down, `NoticeFileRequired`) — the only draft row
that survives intact, though the single draft flow-down turns out to be two. Note the Hosted-Service carve-out in §3.1, which is the scope
difference discussed in Q2.

---

## 11. `nvidia-open-model` — NVIDIA Open Model License Agreement

- **text_url**: `https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/`
  (HTTP 200, retrieved 2026-08-02; this is the `license_link` in the Cosmos-Predict2 card metadata)
- **declared upstream**: `nvidia-open-model-license` (HF card `license_name`)
- **Checkpoints**: ⚠ **NONE CONFIRMED IN THIS REPO — see U4.** `nvidia/Cosmos-Predict2-*` is
  `gated: auto` and declares this licence, but the only Cosmos reference in `crates/media/` is
  `candle-gen-anima/src/config.rs` transcribing the Cosmos DiT *config*; the weights loaded are
  Anima's. Michael must confirm whether anything ships under this family.

| ✔ | term | verbatim support |
| --- | --- | --- |
| [ ] | `DeployerObligation{...}` *(landed text = the quote at right, verbatim)* | §2 "If You bypass, disable, reduce the efficacy of, or circumvent any technical limitation, safety guardrail or associated safety guardrail hyperparameter, encryption, security, digital rights management, or authentication mechanism … contained in the Model without a substantially similar Guardrail appropriate for your use case, your rights under this Agreement will automatically terminate." |
| [ ] | `DownstreamLicenseCopy{"nvidia-open-model"}` | §3 "If you distribute the Model, You must give any other recipients of the Model a copy of this Agreement" |
| [ ] | `NoticeFileRequired` | §3 "include the following attribution notice within a \"Notice\" text file with such copies: \"Licensed by NVIDIA Corporation under the NVIDIA Open Model License\"" |
| [ ] | ⚠ `AttributionRequired` | **Same §3 clause as the row above** — the text calls it "the following **attribution** notice", so one clause yields two terms here. **That is the move withheld from `gemma-terms` §3.1, whose clause says only "notice" — see U11, which now decides both.** |
| [ ] | `AcceptableUsePolicy{Some("https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/")}` | §3 "AI Ethics. Use of the Models under the Agreement must be consistent with NVIDIA's Trustworthy AI terms found at https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/." |
| | *(commercially usable)* | §1 "Models are commercially usable." — a fact, no term |

**Draft correction.** Draft #12 had only `DeployerObligation{safety guardrails}`. Evidence adds four
more terms — and, more importantly, this family covers **only** the Open Model License, not PiD.

---

## 12b. `nvidia-nsclv1` — NVIDIA License (NSCLv1) — **NEW, split out of draft #12**

- **text_url**: `https://huggingface.co/nvidia/PixelDiT-1300M-1024px/raw/main/LICENSE`
  (HTTP 200, 3,997 bytes, retrieved 2026-08-02; this is the URL `nvidia/PiD`'s model card links as
  its licence). First line: "NVIDIA License"
- **declared upstream**: `nvidia/PiD`'s README, §"License/Terms of Use": "This model is released under
  the [NSCLv1](https://huggingface.co/nvidia/PixelDiT-1300M-1024px/blob/main/LICENSE) License. The
  work and any derivative works may only be used for non-commercial
  (research or evaluation) purposes." The HF card carries **no** `license` metadata field at all.
- **In-catalog checkpoints**: `nvidia/PiD` (`mlx-gen-pid`, `candle-gen-pid`) — a bespoke utility crate
  in both catalogs, i.e. present and compiled but not registry-registered
- **gated**: `False`

| ✔ | term | verbatim support |
| --- | --- | --- |
| [ ] | `NonCommercialWeights` | §3.3 "The Work and any derivative works thereof only may be used or intended for use non-commercially. … As used herein, \"non-commercially\" means for research or evaluation purposes only." |
| [ ] | `DownstreamLicenseCopy{"nvidia-nsclv1"}` | §3.1 "You may reproduce or distribute the Work only if (a) you do so under this license, (b) you include a complete copy of this license with your distribution" |
| [ ] | `DownstreamRestrictions{"nvidia-nsclv1"}` | §3.2 "Your Terms provide that the use limitation in Section 3.3 applies to your derivative works" — the restriction must survive into the deployer's own terms, which is the heavier shape, imposed here **alongside** the copy duty |
| [ ] | `AttributionRequired` | §3.1(c) "you retain without modification any copyright, patent, trademark, or attribution notices that are present in the Work" |
| — | ⚠ `NonCommercialOutputs` | **NOT LANDED** — **no supporting quote, see U3.** `mlx-gen-pid/src/lib.rs` asserts "The NC restriction flows to PiD-decoded output", but that is SceneWorks' own reading, not NVIDIA's words |

**This family does not exist in the draft.** Merging PiD into `nvidia-open-model` would attach a
commercially-usable licence's terms to non-commercial weights — the single most consequential
correction in this note after Q1.

---

## 13. `insightface-research-only` — InsightFace non-commercial research use only

- **text_url**: `https://github.com/deepinsight/insightface` — §License of the repository README
  (fetched as `https://raw.githubusercontent.com/deepinsight/insightface/master/README.md`, HTTP 200,
  16,158 bytes, retrieved 2026-08-02)
- **declared upstream**: ⚠ **there is no licence document for the models.** The only statement is
  README prose. The code is MIT; the models are carved out by that prose.
- **In-catalog checkpoints**: the antelopev2 stack — ArcFace `glintr100`, SCRFD-10g detector, BiSeNet
  parsing — consumed by `mlx-gen-face` / `candle-gen-face` and transitively by `mlx-gen-instantid`,
  `mlx-gen-pulid`. Bespoke utility crates in both catalogs; **no HF repo id is pinned anywhere in
  the repo**, so there is no `source_url` of record.

| ✔ | term | verbatim support |
| --- | --- | --- |
| [ ] | `NonCommercialWeights` | README §License "The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only." |
| — | ⚠ `NonCommercialOutputs` | **NOT LANDED** — **no supporting quote, see U3.** The prose restricts model availability/use; it says nothing about generated images. **This reverses the story's draft table for insightface, deliberately.** |
| | *(context)* | README §License "The code of InsightFace is released under the MIT License. There is no limitation for both academic and commercial usage." — the MIT grant is **code only** |

**Draft asserts both NC terms; only the first is quotable, and only it is landed.** Additionally flag the *evidence quality*:
a README sentence is the weakest source in this whole table, and it is attached to the checkpoints
the story describes as carrying the strictest terms in the catalog. Michael may want a stronger
basis (or a decision to stop shipping them) rather than a table row.

---

## 14. `chatglm3-model-license` — The ChatGLM3-6B License

- **text_url**: `https://raw.githubusercontent.com/THUDM/ChatGLM3/main/MODEL_LICENSE`
  (HTTP 200, 5,178 bytes, retrieved 2026-08-02). First line: "The ChatGLM3-6B License"
- **declared upstream**: `THUDM/chatglm3-6b` carries **no** `license` field in its HF card metadata;
  the governing document is the `MODEL_LICENSE` file in the upstream GitHub repo.
  `mlx-gen-kolors/src/convert.rs` copies a file literally named `MODEL_LICENSE`.
- **In-catalog checkpoints**: the ChatGLM3-6B text encoder inside `Kwai-Kolors/Kolors-diffusers`
  (`mlx-gen-kolors`, `candle-gen-kolors`; registered in both catalogs). See **U6** for the
  Kolors-repo-level ambiguity.

| ✔ | term | verbatim support |
| --- | --- | --- |
| [ ] | `RegistrationRequired{Some("https://open.bigmodel.cn/mla/form")}` | §2 "This license permits you to use all open-source models in this repository for academic research free. Users who wish to use the models for commercial purposes must register [here](https://open.bigmodel.cn/mla/form)." |
| [ ] | `AttributionRequired` | §2 "The license notice shall be included in all copies or substantial portions of the Software." |
| | *(context)* | §2 "Registered users may use the models for commercial activities free of charge, but must comply with all terms and conditions of this license." |

Draft had `RegistrationRequired` only; evidence adds `AttributionRequired`.

---

## 15. `apple-mlr` — Apple Machine Learning Research Model License Agreement — **NEW**

- **text_url**: `https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378/raw/main/LICENSE`
  (HTTP 200, 5,820 bytes, retrieved 2026-08-13; immutable repo revision
  `01b771ed0d1395ca5ffdd279897d665ebe00dfd2`)
- **declared upstream**: already recorded in this repo as `LicenseRef-Apple-MLR` /
  "Apple Machine Learning Research Model License Agreement"
- **In-catalog checkpoints**: canonical `apple/DFN5B-CLIP-ViT-H-14-378`, the CLIP conditioner
  inside both MMAudio providers — **2 rows already shipping** in
  `release/model-weight-licenses.json`. MMAudio feeds this 378-native checkpoint at 384×384; its
  patch-14/stride-14 visual stem yields the same 27×27 grid at both resolutions. The stable internal
  component key therefore remains `dfn5b_clip_vit_h14_384`.
- **gated**: `False`

| ✔ | term | verbatim support |
| --- | --- | --- |
| [ ] | `NonCommercialWeights` | §1 "limited license, to use, copy, modify, distribute, and create Model Derivatives … exclusively for Research Purposes. … \"Research Purposes\" does not include any commercial exploitation, product development or use in any commercial product or service." |
| [ ] | `DownstreamLicenseCopy{"apple-mlr"}` | §2 "If you choose to redistribute Apple Machine Learning Research Model or its Model Derivatives, you must provide a copy of this Agreement to such third party" |
| [ ] | `AttributionRequired` | §2 "ensure that the following attribution notice be provided: \"Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement.\"" |
| — | ⚠ `NonCommercialOutputs` | **NOT LANDED** — no supporting quote, see U3 |

**Absent from the draft entirely**, despite already being committed weight-licence data. This is the
strictest licence in the audio catalog and the reason both MMAudio composite rows carry
`commercial_use: false`.

---

## 16. `llama-3-1-community` — Llama 3.1 Community License Agreement — **NEW**

- **text_url**: `https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE`
  (HTTP 200, 7,680 bytes, retrieved 2026-08-02)
- **declared upstream**: `release/real-weight-models.toml` records
  `license = "Llama 3.1 Community License"` for `joycaption-beta-one`
- **In-catalog checkpoints**: `fancyfeast/llama-joycaption-beta-one-hf-llava` (the Llama-3.1-8B
  backbone inside JoyCaption) — `mlx-gen-joycaption`, `candle-gen-joycaption`, registered in both
  catalogs

| ✔ | term | verbatim support |
| --- | --- | --- |
| [ ] | `DownstreamLicenseCopy{"llama-3-1-community"}` | §1(b)(i) "you shall (A) provide a copy of this Agreement with any such Llama Materials" |
| [ ] | `AttributionRequired` | §1(b)(i) "prominently display \"Built with Llama\" on a related website, user interface, blogpost, about page, or product documentation" |
| [ ] | `NoticeFileRequired` | §1(b)(iii) "You must retain in all copies of the Llama Materials that you distribute the following attribution notice within a \"Notice\" text file distributed as a part of such copies: \"Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.\"" |
| [ ] | `AcceptableUsePolicy{Some("https://llama.meta.com/llama3_1/use-policy")}` | §1(b)(iv) "adhere to the Acceptable Use Policy for the Llama Materials (available at https://llama.meta.com/llama3_1/use-policy), which is hereby incorporated by reference into this Agreement" |
| [ ] | `DeployerObligation{"If, on the Llama 3.1 version release date, the monthly active users … is greater than 700 million monthly active users in the preceding calendar month, you must request a license from Meta"}` | §2, quoted with the source's own elision. **LANDED.** No typed variant carries a user-count threshold, and `RevenueCeiling` would be a false transcription — the contract directs such conditions to `DeployerObligation` verbatim. **U8: confirm.** |

Note the crates' own doc comments describe JoyCaption's *prompt table* as Apache-2.0. That is the
source licence of a data file, not the weights. Absent from the draft entirely.

---

# Coverage — draft families vs. what the catalogs actually reference

Grounding sources read on 2026-08-02:
`release/model-weight-licenses.json` (schema 2, 43 rows, 18 provider ids — audio only),
`release/real-weight-models.toml` (schema 1, 49 model rows, captured 2026-07-12),
`crates/media/mlx-gen/mlx-gen-catalog/src/lib.rs` (33 crates present, 27 registered),
`crates/media/candle-gen/candle-gen-catalog/src/lib.rs` (31 present, 25 registered),
`crates/audio/candle-audio-catalog/src/lib.rs` (12 present, all 12 registered), and the `src/lib.rs`
doc headers of every media provider crate.

## Draft families with no checkpoint pointing at them

- **`nvidia-open-model`** — see U4. Cosmos-Predict2 is an architecture reference only.
- (No other draft family is unattached. `insightface-research-only` is attached but has no pinned
  repo id — see family 13.)

## Checkpoints whose licence the 14 draft families do not cover

Beyond the three new families above, the following registered providers reference upstream
checkpoints whose licences were **not** read for this note and are **not** represented in the draft.
Each needs either an existing family assignment or a new one before the table is complete:

| provider crate(s) | upstream | why it is open |
| --- | --- | --- |
| `mlx-gen-flux2`, `candle-gen-flux2` | `black-forest-labs/FLUX.2-klein-9B`, `FLUX.2-dev` | FLUX.2 klein and dev are believed to carry **different** licences from each other and from FLUX.1 [dev]; not read |
| `mlx-gen-seedvr2`, `candle-gen-seedvr2` | `numz/SeedVR2_comfyUI` | no licence stated in-repo; a community re-host, provenance unclear |
| `mlx-gen-sensenova`, `candle-gen-sensenova` | `sensenova/SenseNova-U1-8B-MoT` | no licence stated in-repo |
| `mlx-gen-bernini`, `candle-gen-bernini` | `ByteDance/Bernini-Diffusers` (+ Wan2.2-T2V-A14B) | no licence stated in-repo |
| `mlx-gen-chroma`, `candle-gen-chroma` | `lodestones/Chroma1-Base/-HD/-Flash` | no licence stated in-repo |
| `mlx-gen-lens`, `candle-gen-lens` | `microsoft/Lens`, `microsoft/Lens-Turbo` | no licence stated in-repo |
| `mlx-gen-scail2`, `candle-gen-scail2` | `zai-org/SCAIL-2` | no licence stated in-repo |
| `mlx-gen-wan`, `candle-gen-wan` | `Wan-AI/Wan2.2-*`, `Wan-AI/Wan2.1-VACE-*` | no licence stated in-repo |
| `mlx-gen-ideogram`, `candle-gen-ideogram` | `ideogram-ai/ideogram-4-fp8` + ostris LoRA | "gated turnkey publish"; licence not named |
| `mlx-gen-boogu`, `candle-gen-boogu` | `krea/boogu` (+ Qwen3-VL-8B TE, FLUX.1 VAE) | candle side claims Apache-2.0; the FLUX.1 VAE component is not obviously Apache |
| `mlx-gen-sam2` (MLX only), `mlx-gen-sam3`/`candle-gen-sam3` | `facebook/sam2.1-hiera-*`, `facebook/sam3` | in-repo "Apache-2.0" refers to the `transformers` **code**, not the weights |
| `mlx-gen-mage`, `candle-gen-mage` | `microsoft/Mage-Flow*` | MIT per `_vendor/VENDORED.md`; also ships **Cephes BSD-3-Clause** code inside `mlx-gen-mage/src/latent.rs` (see `crates/media/mlx-gen/NOTICE` + `LICENSE-CEPHES`) — a *code* licence, arguably out of scope for a weight table |

That list is the honest scope of "the count settles near 16": **16 families are evidenced here**, and
the dozen rows above are the work still needed before every registered provider resolves. They are
listed, not guessed at.

## Families already committed in `release/model-weight-licenses.json` (schema 2)

Six real families, all covered above: `MIT` (10 rows), `Apache-2.0` (6), `CC-BY-NC-4.0` (5),
`LicenseRef-Stability-AI-Community` (12), `LicenseRef-Gemma-Terms` (6), `LicenseRef-Apple-MLR` (2),
plus two synthetic composite roll-ups (`LicenseRef-MMAudio-large-44k-composite`,
`LicenseRef-MMAudio-small-16k-composite`) which are **compositions, not families** and should resolve
into per-component rows under the v3 schema rather than becoming `LicenseFamily` entries.

---

# Authenticated reads (gated repos)

Everything in this section was retrieved on **2026-08-02** by Claude (Opus 5), automated agent, on
Michael Trefry's Windows host, using the Hugging Face CLI logged in as **`SceneWorks`**
(`hf auth whoami` → `user: SceneWorks`). Command form: `hf download <repo> <file>` — single text
files only, no weights. These reads closed U1 and narrowed U2.

| repo | file | anonymous | authenticated result |
| --- | --- | --- | --- |
| `stabilityai/stable-diffusion-3.5-large` | `LICENSE.md` | 401 | **200** — 11,726 bytes, rev `ceddf0a7…` (= `main`). Closes **U1** |
| `stabilityai/stable-video-diffusion-img2vid-xt` | `LICENSE.md` | 200 | 200 — 11,852 bytes, rev `9e439095…` (re-fetched for the byte comparison) |
| `black-forest-labs/FLUX.1-dev` | `LICENSE.md` | 401 | **200** — 18,491 bytes, rev `3de623fc…`; identical to the GitHub copy |
| `black-forest-labs/FLUX.1-dev` | `README.md` | — | 200 — gate prompt cites a `POLICY.md` |
| `black-forest-labs/FLUX.1-dev` | `POLICY.md` | — | ❌ **Entry Not Found** — the file does not exist. Confirms **U2/FLUX** |
| `krea/Krea-2-Turbo` | `README.md` | — | 200 — gate prompt yields the AUP URL. Resolves **U2/Krea** |
| `krea/Krea-2-Turbo` | `LICENSE.pdf` | 401 | ❌ **still refused** — "Access to model krea/Krea-2-Turbo is restricted and you are not in the authorized list." The `SceneWorks` account is **not** on Krea's allow-list, so the licence text remains sourced from the `license_link` CDN mirror in the card metadata (unchanged from the first pass) |
| `Lightricks/LTX-2.3` | `LICENSE`, `README.md` | 200 | 200 — surfaced the two-texts divergence, **U10** |
| `stabilityai/stable-diffusion-xl-base-1.0` | `LICENSE.md`, `README.md` | 200 | 200 — confirmed zero URLs in the licence |

Gating status (`gated` field) and card metadata were read from `https://huggingface.co/api/models/<id>`.

---

# Retrieval log

Every URL below was fetched on **2026-08-02** with `curl -sSL`, except the corrected canonical
`apple-mlr` URL, which was re-fetched on **2026-08-13**. HTTP status recorded as observed.

| family | URL | status |
| --- | --- | --- |
| apache-2-0 | `https://www.apache.org/licenses/LICENSE-2.0.txt` | 200 |
| mit | `https://raw.githubusercontent.com/spdx/license-list-data/main/text/MIT.txt` | 200 |
| cc-by-nc-4-0 | `https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt` | 200 |
| creativeml-openrail-pp-m | `https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/LICENSE.md` | 200 |
| stability-ai-community | `https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/raw/main/LICENSE.md` | 200 |
| stability-ai-community (cross-check) | `https://stability.ai/community-license-agreement` | 200 |
| stability-ai-community (SD3.5's own copy) | `https://huggingface.co/stabilityai/stable-diffusion-3.5-large/raw/main/LICENSE.md` | 401 anon → **200 authenticated — U1 CLOSED** |
| flux-1-dev-non-commercial | `https://raw.githubusercontent.com/black-forest-labs/flux/main/model_licenses/LICENSE-FLUX1-dev` | 200 |
| flux-1-dev-non-commercial (HF copy) | `https://huggingface.co/black-forest-labs/FLUX.1-dev/raw/main/LICENSE.md` | 401 anon → **200 authenticated; identical to GitHub copy** |
| flux AUP cited by FLUX's own gate prompt | `https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/POLICY.md` | **does not exist — U2** |
| flux AUP guess in sc-16661 fixture | `https://blackforestlabs.ai/aup` | **404 — U2** |
| flux AUP guess, redirect target | `https://bfl.ai/aup` | **404 — U2** |
| krea-2-community AUP (from card gate prompt) | `https://www.krea.ai/krea-2-use-policy` | **200 — U2 RESOLVED for Krea** |
| ltx-2-community (copy shipped with the weights) | `https://huggingface.co/Lightricks/LTX-2.3/raw/main/LICENSE` | 200 — 21,393 bytes, **differs from GitHub — U10** |
| ltx-2-community registration URL | `https://ltx.io/model/licensing` | 200 (present in GitHub copy only) |
| ltx-2 AUP probes | `.../LTX-2/main/POLICY.md`, `.../ACCEPTABLE_USE_POLICY.md` | **404 both — U2** |
| krea-2-community | `https://cdn.jsdelivr.net/gh/krea-ai/krea-2@db3984fb…/assets/hf_samples/LICENSE.pdf` | 200 |
| ltx-2-community | `https://raw.githubusercontent.com/Lightricks/LTX-2/main/LICENSE` | 200 |
| circlestone-labs-non-commercial | `https://huggingface.co/circlestone-labs/Anima/raw/main/LICENSE.md` | 200 |
| gemma-terms | `https://ai.google.dev/gemma/terms` | 200 |
| nvidia-open-model | `https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/` | 200 |
| nvidia-nsclv1 | `https://huggingface.co/nvidia/PixelDiT-1300M-1024px/raw/main/LICENSE` | 200 |
| nvidia-nsclv1 (declaration) | `https://huggingface.co/nvidia/PiD/raw/main/README.md` | 200 |
| insightface-research-only | `https://raw.githubusercontent.com/deepinsight/insightface/master/README.md` | 200 |
| chatglm3-model-license | `https://raw.githubusercontent.com/THUDM/ChatGLM3/main/MODEL_LICENSE` | 200 |
| apple-mlr | `https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378/raw/main/LICENSE` | 200 |
| llama-3-1-community | `https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE` | 200 |
| SANA 1600M (U5) | `https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers/raw/main/LICENSE` | 200 — **Apache-2.0** |
| SANA 1600M, URL cited in `candle-gen-sana/NOTICE` (U5) | `.../Sana_1600M_1024px_diffusers/raw/main/LICENSE.txt` | **404** |
| SANA-Sprint (U5) | `https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers/raw/main/LICENSE` | 200 — **Apache-2.0** |

Gating status was read from the Hugging Face model API (`https://huggingface.co/api/models/<id>`,
`gated` and `cardData` fields), also on 2026-08-02.

---

# Sign-off

**Nothing here is signed off.** The `LicenseFamily` consts are landed and marked PROVISIONAL in
their own module docs. They are landed rather than held because the surface is disclosure-only — a
wrong value shows a user a wrong sentence, it does not block a render — and because a compiled table
is easier to review than a branch. Everything downstream of this story (the component rows sc-16665,
the schema-3 manifest sc-16664, the drift job sc-16670) should wait on the ticks below.

## 1. Per-family quote check

For each family: read the quote in its section above, decide whether it supports the term, tick.

| family | reviewed | notes |
| --- | --- | --- |
| 1 `apache-2-0` | [ ] | flow-down is new vs the draft — U9 |
| 2 `mit` | [ ] | |
| 3 `cc-by-nc-4-0` | [ ] | `NonCommercialOutputs` dropped — U3 |
| 4 `creativeml-openrail-pp-m` | [ ] | AUP has no address — U2 |
| 5 `stability-ai-community` | [ ] | merged #5/#6; ceiling **Exclusive** |
| 6 `flux-1-dev-non-commercial` | [ ] | AUP has no address — U2 |
| 7 `krea-2-community` | [ ] | **ceiling boundary Inclusive is a new determination**; `AttributionRequired` rests on a model-naming duty — U11 |
| 8 `ltx-2-community` | [ ] | **two upstream texts — U10 is the decision**; the `DeployerObligation` string carries the source's "…" |
| 9 `circlestone-labs-non-commercial` | [ ] | |
| 10 `gemma-terms` | [ ] | `AttributionRequired` withheld — U11 (decided with #11 and #7) |
| 11 `nvidia-open-model` | [ ] | may have no checkpoint — U4; **two terms from one §3 clause — U11 (decided with #10 and #7)** |
| 12 `nvidia-nsclv1` | [ ] | `NonCommercialOutputs` dropped — U3 |
| 13 `insightface-research-only` | [ ] | README prose is the whole evidence; `NonCommercialOutputs` dropped — U3 |
| 14 `chatglm3-model-license` | [ ] | |
| 15 `apple-mlr` | [ ] | `NonCommercialOutputs` dropped — U3 |
| 16 `llama-3-1-community` | [ ] | 700M-MAU as `DeployerObligation` — U8 |

## 2. Decisions

Items marked **DECIDE** were taken by the transcription pass and are reversible in one commit; items
marked **confirm** are readings the evidence supports but a human has not endorsed.

| decision | outcome |
| --- | --- |
| U1 SD3.5 gated text | ✅ **CLOSED 2026-08-02** — read under HF account `SceneWorks`; cosmetic differences only, revenue clause byte-identical, ceiling "more than USD $1,000,000" (**Exclusive**). #5/#6 merge stands. No decision needed |
| U2 `AcceptableUsePolicy` with no URL | **DECIDE.** Krea resolved → `https://www.krea.ai/krea-2-use-policy`. For OpenRAIL++ / FLUX / LTX-2 the **contract was amended**: `AcceptableUsePolicy::url` is now `Option`, and those three land `None`. No URL was invented; `https://blackforestlabs.ai/aup` is asserted absent by a test. Confirm the convention, or say you would rather point `url` at the licence text itself |
| U3 `NonCommercialOutputs` from a use restriction | **DECIDE.** Not landed on any family — no quote exists for it. Reverses the story's draft for `insightface-research-only` and `cc-by-nc-4-0`. If you read silence as reaching outputs, that is a legal determination and it should be recorded here as one, with your name on it |
| U4 `nvidia-open-model` has no checkpoint | **OPEN.** Family landed (its text is real and was read); whether any shipped checkpoint declares it is a component question. Settle before sc-16665 |
| U5 `candle-gen-sana/NOTICE` correction | **OPEN.** SANA's weights read Apache-2.0 on primary sources, not an NVIDIA family. Needs its own story; untouched here |
| U6 Kolors repo-level licence | **OPEN.** A component question, deferred to sc-16665 |
| U7 `GatedAccess` on family vs component | ✅ **CLOSED by sc-16898** — moved to `ComponentLicense::gated`; conformance rejects it on a family; a test asserts no family declares it |
| U8 Llama 700M MAU has no variant | **confirm.** Landed as a verbatim `DeployerObligation`, per the contract's own guidance. Not laundered into a `RevenueCeiling`, and a contract test asserts that |
| U9 Apache-2.0 §4(a) as flow-down | **confirm.** Landed as `DownstreamLicenseCopy{"apache-2-0"}`. The sc-16898 split makes the answer mechanical: §4(a) is a copy duty and Apache states no restrictions to flow down |
| U10 which of LTX-2's two texts governs | **DECIDE — the most consequential open item.** Landed from the copy **shipped beside the weights** (`Lightricks/LTX-2.3` `LICENSE`), so `RegistrationRequired{contact: None}`. Choosing the GitHub copy instead adds `https://ltx.io/model/licensing` and flips `"Control"` from "50% or more" to "more than 50%" |
| U11 notice clause → `AttributionRequired`? Gemma, NVIDIA and Krea (widened by the review pass) | **DECIDE — one decision, three rows.** `gemma-terms` carries `NoticeFileRequired` only; `nvidia-open-model` derives **both** it and `AttributionRequired` from a single §3 clause; `krea-2-community` carries `AttributionRequired` on §3.1(b)'s model-name-prefix duty. The divergence is currently an artefact of transcription order, not a reading. Candidate distinguisher, stated in U11 and asserted nowhere in the code: NVIDIA's clause says "attribution notice", Gemma's says only "notice". Options A–D are tabulated in U11; three of the four change a term list and so trip the term census by design |
| Krea ceiling boundary (new) | **confirm.** §2.3's "meet or exceed" reads **Inclusive** — the same $1,000,000 as Stability with the opposite boundary. This determination was made in the transcription pass, not the evidence pass |
| Q1 SVD-XT revenue ceiling | ✅ **ANSWERED** — yes, $1,000,000, same text as SD3.5 |
| Q2 do the flow-downs differ? | ✅ **ANSWERED and shipped** — materially different; two variants, each carrying its family |

## 3. Sign-off

| | |
| --- | --- |
| Reviewed by | **_(pending — Michael)_** |
| Date | **_(pending)_** |
| Outcome | **_(pending)_** |
