# SC-20686 persistent K/V campaign transport

The campaign adapter runs the one registered FLUX.2 Klein edit route and all five registered Wan
routes. For every normal and cancellation arm it creates a private `events.jsonl` file, passes that
path to the product entrypoint with `--sc20686-events`, and seals the exact event transcript as a
separate bundle artifact. Provider stdout and stderr are retained only as diagnostics: progress
output, including carriage-return updates, is never parsed as campaign evidence.

Every child process runs inside its own adapter-owned `sealed-run` directory. The adapter passes an
absolute `--out` below that directory, so images and video frames cannot escape the run closure via
an entrypoint default. FLUX writes `media.png`; Wan writes frames below `media`. Before deleting the
private run directory, the publisher copies every normal-arm output into the final campaign bundle.
Each run has a canonical media manifest that preserves output kind, relative path, byte count, and
content hash; both that metadata hash and every media content hash are bound by the row and campaign
receipts. Cancellation arms must seal an `absent` manifest and are rejected if they leave partial
media behind. The event transcript and media output therefore share one isolated parent, while the
final evidence remains independently reproducible after that private directory is removed.

## Sealed provenance and snapshot layouts

Invoke the adapter with `--inference-revision <40-hex-commit>` from the exact inference checkout
being measured. The adapter verifies that value against `git rev-parse HEAD`, passes it separately
as `--sc20686-source-ref`, and requires observer metadata to reproduce it. This repository revision
is never inferred from a model path.

Model identity has two independent fields: `model_snapshot_revision` is the immutable Hugging Face
revision, while `model_snapshot_sha256` hashes only the selected model/tier root. A selected root may
be either a component/tier directory with `config.json`, or a Diffusers pipeline root with
`model_index.json` and at least one component `config.json`. Nested tier roots such as
`<snapshot-revision>/q4` resolve the revision from the nearest two ancestors while hashing only the
`q4` contents. This preserves exact tier identity without confusing `q4` with a revision or widening
the hash to unrelated siblings.

## Product-equivalent residency

Residency is a sealed route axis, is passed to the entrypoint as `--sc20686-residency`, and is
applied to the real `LoadSpec` (plus request-scoped generation staging for FLUX.2 edit). The frozen
SceneWorks-equivalent strategies are:

| Product route | Strategy |
| --- | --- |
| `flux2_klein_9b_edit` | `sequential` |
| `wan2_2_ti2v_5b` | `sequential` |
| `wan2_2_t2v_14b` | `sequential` |
| `wan2_2_i2v_14b` | `sequential` |
| `wan_vace` | `resident` |
| `wan2_2_vace_fun_14b` | `sequential` |

The adapter, entrypoints, observer metadata, resolved-input manifest, row receipts, and reducer all
reject a different strategy rather than measuring a non-product residency shape. This includes the
Wan 14B ComfyUI-expert route: campaign residency is passed explicitly through its external-expert
loader, rather than falling back to that loader's ordinary resident default.

The Wan entrypoints are wired to the product-owned observer after each real route has bound its
snapshot-backed geometry. With observation off, the ownership hooks remain inactive and do not
allocate campaign evidence or retain cache ids. The adapter rejects a missing, non-JSONL, or
carriage-return-containing event transcript before reducing a campaign row.

Normal/cancellation pairs are inseparable decision evidence. The reducer independently requires the
cancel arm's product-owned cancellation identity, exactly one `cancelled` terminal, then metrics,
invalidation, and release in product order; each coordinate decision records that verification.

FLUX.2 Klein edit is an evidence-based no-go for persistent-reference-K/V productization in this
campaign. Its `DoubleAttention` path projects reference K/V for each denoise evaluation and joins
it into dense attention; there is no persistent reference K/V boundary or packed reader to promote.
The observer records that transient reference-slice work so a live campaign can establish the
no-go without representing an ordinary attention allocation as a promotable cache.
