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

## Campaign manifest and workflow

Each row names two storage boundaries:

- `snapshotRoot` is an exact `<repo>/snapshots/<40-hex-revision>` directory. Its complete logical
  file inventory, including followed HF blob targets, is hashed before and after generation.
- `bundleSubdir` is a traversal-free relative directory containing exactly one selected split
  bundle. The controller discovers there, then pins every resolved component explicitly while
  retaining the full snapshot as the inventory boundary.

The v1 manifest must contain all nine cases exactly once. The shape below shows the required fields;
repeat it for the complete matrix listed above.

```json
{
  "schemaVersion": "sceneworks-ltx25-quant-campaign-v1",
  "cases": [
    {
      "caseId": "ltx25-bf16-blackwell-v1",
      "transformerVariant": "distilled",
      "snapshotRoot": "D:\\hf\\source-bf16\\snapshots\\<40hex>",
      "modelRevision": "<40hex>",
      "bundleSubdir": "bundles/distilled/bf16"
    },
    {
      "caseId": "ltx25-int8-convrot-blackwell-v1",
      "transformerVariant": "distilled",
      "snapshotRoot": "D:\\hf\\upstream\\snapshots\\<40hex>",
      "modelRevision": "<40hex>",
      "bundleSubdir": "bundles/distilled/int8-convrot",
      "bf16TextEncoderSubpath": "shared/gemma4-bf16.safetensors"
    },
    {
      "caseId": "ltx25-int8-convrot-blackwell-dev-v1",
      "transformerVariant": "dev",
      "snapshotRoot": "D:\\hf\\upstream\\snapshots\\<40hex>",
      "modelRevision": "<40hex>",
      "bundleSubdir": "bundles/dev/int8-convrot",
      "bf16TextEncoderSubpath": "shared/gemma4-bf16.safetensors"
    }
  ]
}
```

Dispatch `.github/workflows/ltx25-quant-campaign.yml` at the exact committed inference revision
with the absolute manifest path, a new absolute evidence root, and one numeric physical GPU
ordinal. The workflow checks exact `sm_120`, builds once, runs both BF16 references before all
seven candidates in one serial process, and copies the exact executable to
`<evidenceRoot>/controller/ltx25-quant-measure.exe`. The campaign, promotion, and
`real-weights.yml` share the repository-wide `inference-real-weights-physical-host` concurrency
group; do not bypass that lock.

## Reviewed public promotion and real replay

Upload only to the canonical public repository `SceneWorks/ltx-2.5-mlx`. Resolve its immutable
revision into the canonical Hugging Face cache path:

```text
D:\hf\hub\models--SceneWorks--ltx-2.5-mlx\snapshots\<public40hex>
```

Capture the raw public API response after upload, including expanded blob metadata, and retain it
as an absolute runner-local file. For example:

```powershell
$revision = "<public40hex>"
Invoke-WebRequest -UseBasicParsing `
  -Uri "https://huggingface.co/api/models/SceneWorks/ltx-2.5-mlx/revision/$revision?blobs=true" `
  -OutFile "D:\campaign\ltx25-public-readback.json"
```

The validator requires `id=SceneWorks/ltx-2.5-mlx`, the exact revision, `private=false`, the exact
full sibling set and sizes, and matching LFS SHA-256 values where supplied. A private repository,
wrong cache layout, partial download, local mutation, or stale readback fails before replay.

A reviewer chooses only the passing production winner(s). Selection is explicit, may contain one
winner per transformer variant, and is not forced to promote all three advanced candidates. Record
the reviewer and concrete quality policy in the v2 promotion input:

```json
{
  "schemaVersion": "sceneworks-ltx25-quant-promotion-v2",
  "publicRepository": "SceneWorks/ltx-2.5-mlx",
  "selection": {
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
  },
  "cases": [
    {
      "caseId": "ltx25-int8-convrot-blackwell-v1",
      "transformerVariant": "distilled",
      "publicSnapshotRoot": "D:\\hf\\hub\\models--SceneWorks--ltx-2.5-mlx\\snapshots\\<public40hex>",
      "publicModelRevision": "<public40hex>",
      "publicBundleSubdir": "bundles/distilled/int8-convrot",
      "bf16TextEncoderSubpath": "shared/gemma4-bf16.safetensors",
      "publicReadback": "D:\\campaign\\ltx25-public-readback.json"
    }
  ]
}
```

Dispatch `.github/workflows/ltx25-quant-promotion.yml` with the original exact 40-hex inference
revision, campaign manifest, evidence root, reviewed promotion input, a new output directory, and
the same physical GPU ordinal. It checks out the exact revision but does not rebuild: it runs the
preserved campaign executable and performs a real generation from every selected public winner on
the exact consumer `sm_120` device.

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
