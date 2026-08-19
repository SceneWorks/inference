# SC-16594: `MEMORY_EVIDENCE_V1` calibration observations

Real-weight memory harnesses now have one persisted observation boundary:

```text
MEMORY_EVIDENCE_V1 {"schema_version":1,...}
```

The payload is compact JSON emitted by `gen_core::MemoryEvidenceLogRecord::to_json_line`. It carries
the complete `MemoryEvidenceKey` (including backend, numeric tier and component floors, load shape,
mode, overlay, geometry with `reference_count`, exact engaged composition, and parameters), the
declared and observed calibration identities, predicted and observed absolute live-allocation peaks,
exact inference and SceneWorks Git revisions, harness version, and parity contract/result.
Each record also carries the exact model revision, the SHA-256 of the canonical dereferenced model
file inventory persisted with the run, and the SHA-256 of the exact output bytes written by that
probe. A log separated from its model inventory is therefore incomplete promotion evidence.

Both peak fields are live-allocation high-waters. MLX probes use its active/peak allocator counters;
Candle CUDA probes reset and read `CU_MEMPOOL_ATTR_USED_MEM_HIGH`. The accompanying sampled
`nvidia-smi`/reserved-footprint report is diagnostic only and is never written into either V1 peak
field. This preserves the budget currency defined by SC-16784 and prevents allocator cache from
being mistaken for irreducible request demand.

`scripts/release/verify_residency_ab.py` accepts no legacy `SEQ_AB` fallback. It rejects missing,
duplicate, malformed, extra-field, and wrongly typed records. For the resident/staged A/B it also
requires every non-strategy identity axis to match, verifies the record fingerprint and ABI against
the provider's exported values supplied by the caller, verifies each output file against its bound
SHA-256, executes exact cross-strategy parity, and then applies the requested byte reduction.
One log may contain both records; separate resident and staged logs remain supported.

A single-process Candle probe emits `parity_result=not_run`; it cannot truthfully know the other
strategy's output. The A/B verifier owns the cross-process byte comparison. An MLX harness that runs
both strategies itself may promote both records to `passed` only after its in-process exact comparison
succeeds. Neither path accepts an unbound, self-attested parity result.

## Fingerprint grammar

A calibration content fingerprint is lowercase ASCII kebab tokens with exactly one positive `vN`
token and no leading zeroes, for example `z-image-mlx-independent-materialization-v4`.

The fingerprint names content whose change invalidates a calibration. It is not a second encoding of
typed evidence axes: backend, tier, load shape, mode, geometry, strategy composition, and strategy
parameters remain in `MemoryEvidenceKey`. `MemoryProviderContract::conformance_errors` and the V1
writer both enforce the grammar, so a new provider cannot silently introduce an unlintable spelling.

## Calibration-point predictions

A promotion harness may set `predicted_peak_bytes` to the measured high-water at the exact calibration
cell: that value is the table prediction for the cell being established. This is not out-of-sample
validation. A later run that validates a previously promoted formula must emit the formula prediction
instead. Both values use the contract's absolute request live-allocation currency, not allocator cache
retention or process footprint.

## Revisions

Both repository revisions and the model revision are mandatory lowercase 40-character Git commit
IDs. Labels such as
`main`, dirty-tree hashes, abbreviated SHAs, and branch names are rejected. If a harness runs from a
dirty tree, it may be useful during development, but its output is not a promotable V1 record.

Every Candle probe also requires `MEMORY_EXPECTED_FINGERPRINT` and `MEMORY_EXPECTED_ABI`. Those are
the workflow/operator's pre-load declaration; the provider's exported constant or executable
contract supplies the independently observed identity. The writer rejects disagreement. The checked-in
`run-residency-ab.ps1` sets both values per provider, binds `INFERENCE_REVISION` to a clean checked-out
HEAD, and requires the exact SceneWorks revision plus each model's exact revision and canonical
inventory SHA-256 as explicit operator inputs.

The `memory-evidence-v1` dispatch profile in `.github/workflows/real-weights.yml` is the promotable
MLX path. It resolves the explicit `sceneworks_revision` through an actual SceneWorks checkout in
the runner's temporary directory, then re-verifies the inference worktree is the clean `github.sha`
immediately before measurement. It inventories every loader-visible model file by relative path,
size, and dereferenced SHA-256 (independent of symlink versus materialized-file storage), binds that
inventory hash into both records, and
requires an identical post-probe inventory so runner-local content cannot change mid-measurement. It
then runs exactly one serialized resident/staged probe, requires at least a 512 MiB live-allocation
reduction, feeds its two records and persisted RGB artifacts through the strict verifier, and only on
success uploads the verifier receipt, model inventory, log, and both bound outputs under a
revision-keyed artifact name.
