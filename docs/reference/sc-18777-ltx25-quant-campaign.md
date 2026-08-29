# LTX-2.5 terminal quant campaign and promotion (sc-18777)

This is a terminal evidence apparatus, not a catalog route. Production `Quant::Q8` remains
INT8-ConvRot; packed q8 exists only in this comparison campaign. Production INT8-ConvRot and
NVFP4 remain fail-closed while `accepted_quant_receipts.allowlist` is empty.

## Physical matrix

The active CUDA pool is consumer Blackwell `sm_120`, so the immutable matrix has nine rows:

- distilled: bf16, packed q4, packed q8, INT8-ConvRot, NVFP4;
- dev: bf16, packed q4, packed q8, INT8-ConvRot.

There is no dev NVFP4 row because the upstream NVFP4 transformer is distilled-only. Ada,
datacenter Blackwell, unknown CUDA generations, and non-CUDA devices remain explicitly classified
and refused; they are not speculative campaign rows.

## Campaign manifest

Each row names two separate boundaries:

- `snapshotRoot` is an exact `<repo>/snapshots/<40-hex-revision>` directory. Its complete logical
  file inventory (including followed HF blob targets) is hashed before and after generation.
- `bundleSubdir` is a relative nested directory containing exactly one selected split bundle. The
  controller discovers there, then copies every resolved component into an explicit `LoadSpec`
  component slot while keeping the full snapshot as `LoadSpec.weights`.

The manifest must contain every case exactly once. Candidate and bf16 rows may use different
snapshots and bundle subdirectories.

```json
{
  "schemaVersion": "sceneworks-ltx25-quant-campaign-v1",
  "cases": [
    {"caseId":"ltx25-bf16-blackwell-v1","snapshotRoot":"D:\\hf\\source-bf16\\snapshots\\<40hex>","modelRevision":"<40hex>","bundleSubdir":"bundles/distilled/bf16"},
    {"caseId":"ltx25-packed-q4-blackwell-v1","snapshotRoot":"D:\\hf\\source-q4\\snapshots\\<40hex>","modelRevision":"<40hex>","bundleSubdir":"bundles/distilled/q4"},
    {"caseId":"ltx25-packed-q8-blackwell-v1","snapshotRoot":"D:\\hf\\source-q8\\snapshots\\<40hex>","modelRevision":"<40hex>","bundleSubdir":"bundles/distilled/q8"},
    {"caseId":"ltx25-int8-convrot-blackwell-v1","snapshotRoot":"D:\\hf\\upstream\\snapshots\\<40hex>","modelRevision":"<40hex>","bundleSubdir":"bundles/distilled/int8-convrot"},
    {"caseId":"ltx25-nvfp4-blackwell-v1","snapshotRoot":"D:\\hf\\upstream\\snapshots\\<40hex>","modelRevision":"<40hex>","bundleSubdir":"bundles/distilled/nvfp4"},
    {"caseId":"ltx25-bf16-blackwell-dev-v1","snapshotRoot":"D:\\hf\\source-bf16\\snapshots\\<40hex>","modelRevision":"<40hex>","bundleSubdir":"bundles/dev/bf16"},
    {"caseId":"ltx25-packed-q4-blackwell-dev-v1","snapshotRoot":"D:\\hf\\source-q4\\snapshots\\<40hex>","modelRevision":"<40hex>","bundleSubdir":"bundles/dev/q4"},
    {"caseId":"ltx25-packed-q8-blackwell-dev-v1","snapshotRoot":"D:\\hf\\source-q8\\snapshots\\<40hex>","modelRevision":"<40hex>","bundleSubdir":"bundles/dev/q8"},
    {"caseId":"ltx25-int8-convrot-blackwell-dev-v1","snapshotRoot":"D:\\hf\\upstream\\snapshots\\<40hex>","modelRevision":"<40hex>","bundleSubdir":"bundles/dev/int8-convrot"}
  ]
}
```

Dispatch `.github/workflows/ltx25-quant-campaign.yml` with the absolute manifest path, a new
absolute evidence root, and one numeric physical GPU ordinal. The workflow checks `sm_120`, builds
the producer once, then one process runs both bf16 references first and all seven candidates
serially with exactly that one `CUDA_VISIBLE_DEVICES` ordinal. Do not dispatch while another
real-weight job owns the same physical GPU.

## Public promotion without receipt restamping

Publish the final artifacts publicly first, then resolve the immutable public HF revision and make
a promotion input containing only the three production-advanced rows:

```json
{
  "schemaVersion": "sceneworks-ltx25-quant-promotion-v1",
  "cases": [
    {"caseId":"ltx25-int8-convrot-blackwell-v1","publicSnapshotRoot":"D:\\hf\\public\\snapshots\\<public40hex>","publicModelRevision":"<public40hex>","publicBundleSubdir":"bundles/distilled/int8-convrot"},
    {"caseId":"ltx25-nvfp4-blackwell-v1","publicSnapshotRoot":"D:\\hf\\public\\snapshots\\<public40hex>","publicModelRevision":"<public40hex>","publicBundleSubdir":"bundles/distilled/nvfp4"},
    {"caseId":"ltx25-int8-convrot-blackwell-dev-v1","publicSnapshotRoot":"D:\\hf\\public\\snapshots\\<public40hex>","publicModelRevision":"<public40hex>","publicBundleSubdir":"bundles/dev/int8-convrot"}
  ]
}
```

Run the already-built producer without CUDA/model generation:

```text
ltx25-quant-measure.exe \
  --acknowledgement I_ACKNOWLEDGE_SC18777_TERMINAL_MEASUREMENT_ONLY \
  --materialize-promotion \
  --campaign-manifest D:\campaign\campaign.json \
  --promotion-input D:\campaign\public-promotion.json \
  --evidence-root D:\evidence\sc-18777 \
  --output-dir D:\evidence\sc-18777-promotion
```

Promotion re-inventories both source and public roots. Full snapshot inventories may differ, but
every selected component id, bundle-relative path, byte length, and SHA-256 must match. It keeps the
original measured receipt unchanged, records the final public revision/full inventory/bundle hash,
and seals the source-to-public mapping. A selected-byte or path mutation fails promotion.

The generated `accepted_quant_receipts.allowlist` is the only repository payload copied into the
existing non-Rust allowlist file. It is deliberately outside the measured Rust source contract;
no `.rs` change is required after measurement. Production reconstructs the live public snapshot
identity and compares it with that reviewed mapping. The generated runtime-binding JSON files are
audit renderings; model snapshots do not authorize themselves and must not be modified to add a
sidecar after their immutable public revision is known.
