# LTX-2.5 terminal quant campaign and promotion (sc-18777)

This is a terminal evidence apparatus, not a catalog route. Production `Quant::Q8` remains
INT8-ConvRot; packed q8 exists only in this comparison campaign. Production INT8-ConvRot and
NVFP4 remain fail-closed while `accepted_quant_receipts.allowlist` is empty.

## Physical matrix and immutable inputs

The active CUDA pool is consumer Blackwell `sm_120`, so the immutable matrix has nine rows:

- distilled: bf16, packed q4, packed q8, INT8-ConvRot, NVFP4;
- dev: bf16, packed q4, packed q8, INT8-ConvRot.

There is no dev NVFP4 row because the upstream NVFP4 transformer is distilled-only. Ada,
datacenter Blackwell, unknown CUDA generations, and non-CUDA devices remain explicitly classified
and refused.

Each manifest row binds an exact transformer variant separately from the weight bytes. Current
upstream LTX-2.5 transformer headers do not contain `variant`; do not rewrite or hand-stamp those
headers. The controller uses the reviewed `transformerVariant` only when metadata is absent, and
fails if present metadata disagrees. This preserves the upstream object IDs while preventing a dev
transformer from silently taking the distilled schedule.

Every advanced row (distilled INT8-ConvRot, distilled NVFP4, and dev INT8-ConvRot) must also name
the same upstream all-BF16 Gemma text-encoder file through `bf16TextEncoderSubpath`. The controller
inspects every safetensors tensor dtype before CUDA loading and refuses a directory, mixed dtype, or
Comfy/I8 encoder. Packed and BF16 comparison rows must not declare this field.

## Autonomous campaign materialization and workflow

The terminal workflow has no operator-provisioned manifest or snapshot prerequisite. Set the
repository variable `CANDLE_LTX25_HF_CACHE_ROOT` to one absolute persistent cache root on the
Windows pool. The workflow then accepts only the exact public `SceneWorks/ltx-2.5-mlx` 40-hex
revision and one physical GPU ordinal.

`scripts/release/prepare_ltx25_quant_campaign.py` runs under the repository's pinned uv, reviewed
CPython 3.12.10, and hash-locked Windows `huggingface_hub` dependency set. It requests the
repository with `token=False`, disables implicit tokens, rejects anything except
`private=false,gated=false`, and performs a full `snapshot_download` into this canonical layout:

```text
<CANDLE_LTX25_HF_CACHE_ROOT>\models--SceneWorks--ltx-2.5-mlx\snapshots\<public40hex>
```

The full repository is required (about 464 GiB logical at the SC-18777 publication), because
promotion compares its local inventory with every sibling in the raw public API readback. A
partial `allow_patterns` download cannot pass that check.

Before it writes the campaign manifest or starts the GPU producer, the helper walks the canonical
snapshot without following directory symlinks and compares its exact logical path set and sizes to
the expanded readback. Every available LFS SHA-256 is recomputed from the resolved bytes. A file
symlink is accepted only when it resolves inside that repository's canonical `blobs` directory;
missing, extra, mutated, special, parent-symlinked, or escaping entries fail before measurement.

The helper generates all nine v1 rows from a fixed table. BF16/packed rows use the public
`distilled/{bf16,q4,q8}` and `dev/{bf16,q4,q8}` bundles. Advanced rows use the three
`bundles/**` directories and bind the all-BF16 Gemma file by its snapshot-relative path inside the
same selected bundle. Every row retains the complete public snapshot as its inventory boundary.

Dispatch `.github/workflows/ltx25-quant-campaign.yml` at the exact committed inference ref with
only `public_revision` and one numeric `physical_gpu`. The workflow checks exact `sm_120`, builds
once, runs both BF16 references before all seven candidates in one serial process, and preserves:

```text
controller/ltx25-quant-measure.exe
controller/campaign-manifest.json
controller/campaign-public-readback.json
```

The artifact name includes the inference SHA, GPU ordinal, and run attempt. The campaign,
promotion, and `real-weights.yml` share the repository-wide
`inference-real-weights-physical-host` concurrency group; do not bypass that lock.

## Reviewed public promotion and real replay

Upload only to the canonical public repository `SceneWorks/ltx-2.5-mlx`. The promotion workflow
re-materializes that same exact revision anonymously into the persistent canonical cache and
captures the raw `?blobs=true` API response itself. The validator requires the canonical ID, exact
revision, `private=false`, `gated=false`, the complete sibling set and sizes, and matching LFS
SHA-256 values where supplied. A private/gated repository, wrong cache layout, partial download,
local mutation, or stale readback fails before replay.

A reviewer chooses only the passing production winner(s). Selection is explicit, may contain one
winner per transformer variant, and is not forced to promote all three advanced candidates. Supply
this compact reviewed selection JSON; the workflow constructs the path-bearing v2 document only
after it knows the runner's canonical snapshot and raw-readback paths:

```json
{
  "policyId": "sc-18777-reviewed-selection-v1",
  "reviewedBy": "<reviewer identity>",
  "selectedCaseIds": ["ltx25-int8-convrot-blackwell-v1"],
  "minimumReferencePsnr": 20.0,
  "minimumReferenceSsim": 0.8,
  "maximumTemporalBoundaryDrift": 0.1,
  "minimumReplayPsnr": 100.0,
  "minimumReplaySsim": 0.99,
  "maximumReplayTemporalBoundaryDrift": 0.001,
  "requireReplayOutputHashMatch": true
}
```

Dispatch `.github/workflows/ltx25-quant-promotion.yml` with the original inference revision,
successful campaign run ID and attempt, public model revision, compact reviewed selection JSON,
and the same physical GPU ordinal (six inputs total). The workflow verifies through the GitHub API
that the named run is a successful `ltx25-quant-campaign.yml` dispatch at the exact inference SHA,
then downloads the exact attempt-named artifact with the pinned `actions/download-artifact` action.
It checks out the exact revision but does not rebuild: it runs the preserved campaign executable
and performs a real generation from every selected public winner on exact consumer `sm_120`.

The producer copies the exact reviewed input bytes and newly fetched raw readback into
`promotion-sources/`. `promotion-manifest.json` records their paths, lengths, and SHA-256 values
alongside the replay receipts, runtime bindings, and staged allowlist hash. These source artifacts
and the manifest are fully written before the final atomic rename publishes
`accepted_quant_receipts.allowlist`; no workflow post-processing may add or reseal them.

Promotion re-inventories the complete public snapshot before and after generation; revalidates the
unchanged source receipt, code, executable, GPU/driver, selected components, external variant,
BF16 text encoder, and operator evidence; compares the replay with both the BF16 reference and the
measured winner; requires the reviewed quality thresholds and exact output hash; and seals a public
replay receipt containing those identities. Each replay directory retains the complete inventories
as `pre-public-inventory.json` and `post-public-inventory.json`; the replay receipt seals both files
and requires them to be byte-identical. A failed selection or replay can leave diagnostic evidence
but cannot write an allowlist.

Only after every explicitly selected winner passes does the controller create
`accepted_quant_receipts.allowlist`. Production maps the ordinary nested SceneWorks tier to the
reviewed full public snapshot, pins every selected component plus the BF16 text encoder explicitly,
reconstructs the live identity, and admits it only when the public repository/readback/replay and
source-to-public copy proof match. Model snapshots never authorize themselves and must not be
modified with post-public sidecars.
