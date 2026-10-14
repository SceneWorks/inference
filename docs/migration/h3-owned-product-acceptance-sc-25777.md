# H3 native product acceptance authority

`config/acceptance/h3-owned-product.json` describes implemented source surfaces;
it is not a device acceptance receipt. Both providers use ordinary
`Generator::generate` and `VedaExecutionReport::outcome_json`. Absent
`GenerationRequest::h3_terminal_acceptance` leaves capture and faults off.
Authored Veda controls, weights, tolerances and normal resource gates are unchanged.

The SceneWorks worker independently validates the private authority file, SHA,
ownership, original canonical request projection against its actual job,
source freeze, campaign project/marker, and owned oracle process before calling
`gen_core::veda::acceptance::Binding::from_json`. Constructor arguments are the
UTF8 authority, `Context` (string fields `campaign_id`, `request_key`,
`request_sha256`, `source_freeze_sha256`, `proof_id`), and actual job ID. The
constructor rejects malformed, duplicate and unknown fields, compares Context,
and hashes exact `canonicalRequestJson` bytes without reserializing floats.
It appends actual worker PID/job ID to the binding digest. Clones share one
consumed generation lifetime; another process, reuse, post-failure append,
and existing capture files are rejected. The report is cleared before binding.

Closed authority keys: `schemaVersion` (1), `campaignId`, `requestKey`,
`requestSha256`, `sourceFreezeSha256`, `canonicalRequestJson`, `proofId`, `kind`,
`capturePath`, `captureScopes`, `captureTransport`. Kinds are
`self_test_output_fault`, `unsupported_capability`, `capture`.
`captureScopes` has exactly integer lists `steps`, `layers`, `heads`; duplicates,
negative/out-of-domain indices and partially empty declarations are rejected.
Faults require all lists empty. The terminal owner declares the finite capture
scope explicitly; native finalization rejects missing declared scope tuples.
It does not silently sample or drop any declared layer/head/step/batch input.

`captureTransport` has exactly `kind: "spool_ack_v1"`, `spoolDirectory` (an owned
absolute sibling directory), `oracleIdentitySha256`, `ackTimeoutMs` (1–30000),
`ownedDeadlineUnixMs` (the original guardian deadline, never more than an hour
away), `maxPendingRecords: 1`. The immutable oracle identity must bind original
frozen source descriptors and actual Python/torch/safetensors/interpreter/tie
probe receipts. Its validation belongs to the SceneWorks guardian/worker.

Self-test fault executes the existing sparse device self-test, materializes
its actual output, then adds 1 to that device output only. The existing numeric
verifier and tolerance remain unchanged. Receipts contain actual error,
applied flag, verifier identity and rejection. Unsupported capability presents
an incompatible probe dtype to the actual capability guard with unchanged
production QKV; it rejects before scorer/sparse production work. MLX defers
its preparation self-test only for this incompatible probe, which cannot reach
compatible sparse execution. Receipts distinguish actual/probe dtype and
injected proof from actual hardware failure. A requested fault that does not
reach and fail its corresponding guard/verifier cannot finalize successfully.

The native compact JSONL capture is never truncated. Every event carries
`bindingSha256` and monotonic zero-based `recordIndex`. `binding` records
campaign/request/source/proof/job/PID, declared scopes, oracle identity, and
compiled provider/model/DiT/packing/pipeline/selector/acceptance source hashes.
`attention_call` observes sparse calls after successful native execution;
`dense_attention` observes successful dense exceptions. Both cover actual
step/layer/shape. `native_output` hashes every decoded RGB frame and all audio
sample bits (little-endian IEEE f32, preserving signed zero).

Each scoped selection is serialized as one actual head's raw JSON artifact,
`<bindingSha256>-<recordIndex>.raw.json`, created new and closed/synced before
an atomically published `.ready.json` notice. Raw `event: "selection"` records
actual `layer`, `step`, `batch`, `heads: [head]`, `plan`, `grid`, `tileShape`,
`dtype`, `seq`, `video`, `reference`, ordered actual `spans: [{start,grid,shape}]`
from the production layout constructor (references first, target last; unprovable
reference segments remain global), complete `valid`/`perm`,
`refTokens`/`targetTokens`, actual `qTiled`/`kTiled` objects (`encoding: "f32_le_hex"`, `shape: [1,tiles,128,dim]`,
`data`: exact little-endian IEEE float-bit hex, `sha256`: binary float-bit SHA),
folded BF16 `projectionQ`/`projectionK` arrays promoted to f32 (head,3*dim,dim),
`predictorSha256`, `dim`, flat head-major scores, complete selected rows,
`generatedBudget`/`referenceBudget` (`ratio` or `tiles`), and `captureBound`.
The actual original managed predictor matching its SHA remains retained once
on each host; it independently authenticates FP8/scales→BF16 projection folds.
There is no fabricated checkpoint or synthetic replacement for product inputs.

Ready notice keys: `schemaVersion: 1`, `bindingSha256`, `recordIndex`,
`rawSha256`, `rawBytes`, `rawPath`, `ackPath`, `oracleIdentitySha256`.
The separate admitted CPU oracle reads the entire raw artifact, runs original
TripPool/projection/residual/selector arithmetic and exact mask checks, closes
all raw handles, and seals an immutable ledger entry before atomically publishing
`.ack.json`. Exact ACK keys: `schemaVersion: 1`, `bindingSha256`, `recordIndex`,
`rawSha256`, `oracleIdentitySha256`, `verdict: "verified"`, `ledgerEntrySha256`.
Native checks every binding/index/hash/identity/verdict, independently rehashes
raw bytes, records `raw_record` with hashes and `transportAcknowledged: true`,
then alone deletes raw/ready/ack. This is a transport observation; native never
claims independent oracle acceptance. The terminal consumer must separately
validate oracle source/runtime, ledger entry SHA/content, coverage, and deletion.
A foreign/error/missing ACK, timeout, mutation or incomplete scope seals failure,
retaining raw diagnostic artifacts. At most one raw record is pending.

Native finalization requires one real generation output, all declared scopes
and applicable fault rejection, and no pending raw. It emits explicit
`finish`, `finished: true`, `valid: true`, `recordsBeforeFinish: N`, syncs and
independently streams back compact capture with a 64 KiB buffer. Failures emit
`abort`, `valid: false`, error, and seal the generation lifetime. EOF or process
termination is never successful closure. The report's `acceptanceProof` includes
binding/context IDs, actual job/PID, `injectedProof`, compact path/SHA,
record/captured tuple counts, pending state, finished/aborted/error, observations.
The terminal consumer rehashes capture and requires its explicit final closure.

Capture overhead is confined to designated proof requests; all 11118 ordinary
E9 baseline identities keep capture absent. `capture_bound` is a conservative
one-head serialization/readback/projection transient bound, separate from device
allocator peaks; the oracle process requires separate admission. Raw retention
is the independently validated compact ledger/hash evidence after acknowledged
raw deletion. CPU tests establish protocol/source discrimination, not device
acceptance. Real MLX/CUDA proofs remain the single final product campaign.
