# Inference whole-branch pre-merge review, round 1 of at most 2

Fresh reviewer /root/moss_inference_whole_branch_r1, gpt-6-astra/high. Verdict CHANGES_REQUIRED, four accepted P2 findings. READONLY: no builds/tests/models/large payload reads/GPU/source/external mutations. Complete 70-file delta reviewed from main5e07e57b1a0bbe741fe67eb2723e888b7fc91fe1 to source candidate9be3a9497656d1c16bd41d12dfc3bd443ba2e18b; tree e1eecf852f69ff156073f188a97fe8ca5ed7cf17 remained unchanged. This is a frozen production SOURCE candidate, not the actual feature ref or main. Actual feature/source tree identity must be checked before accepting the eventual merge.

## [F-280] P2 — Candle repeats full source payload authentication for warm header facts

Candle loader.rs568 and lib.rs138: self.sources has selectionNone and never becomes selected. selected_sources rebuilds for validation, memory estimate and generation, calling validate_source_dtypes which reads entire safetensors Vecs and hashes whole payloads for headers. Ordinary SFT preparation reads approximately27.2GB including largest shard >5.3GB; validate/estimate/generate repeats even for resident model, before admission and without cancellation in preparation loop. Cache authenticated header/source facts by immutable approved identity, preserving pin validation and authentication of exact bytes used for actual tensors. Avoid full payload reauthentication solely for cached headers; propagate cancellation for unavoidable cold work. Byte counts/call paths source-established; no timing claim.

## [F-281] P2 — MLX advanced smoke uses image index for prompt receipt

MLX tests/native_smoke.rs272 requests B2/count2, production conditioninglen2 and sampleslen4. Loop computes b=i/2 but indexes conditioning[i], comparing wrong hash at1 and out-of-range at2. Use conditioning[b], assertlen2/imagesperprompt2, retain samples[i] noise check. CPU-only receipt-order regression can discriminate without Metal.

## [F-282] P2 — Authenticated explicit optional file rejected only in final receipt

gen-core moss_image.rs1045 via Candle loader.rs65: optional_file accepts WeightsSource::File arbitrarybasename; Asset::new verifies canonical logical identity/size/hash. Exact dataspace bytes stored as /cache/<sha>.safetensors pass auth/load, but final MossImageAssetReceipt::from_pin additionally requires canonical physical suffix, discarding valid output after sampling/decode. LoRA/time explicit files similar. Build receipt from authenticated logical identity and retained pin independent of physical basename, preserving digest/repo/revision; if canonical physical names intentionally required fail upfront instead. Prefer logical identity to preserve native explicit-file capability.

## [F-283] P2 — Numerical oracle comparators falsely accept NaNs

MLX transformer.rs655 and Candle tests.rs48 reduce absolute error using fold(0,f32::max), which ignores NaN against numeric operands; allNaN output yields0 error and passes. MLX also zip-truncates without shape/count equality. Assert finite actual/expected plus exact shape/count before errors; discriminating NaN/truncated comparator regressions. Principal ordinary/DMD graph oracle evidence depends on this. Production final-output validation does not repair false-green tests.

## Coverage and limits

Reviewed complete both-provider implementations; shared options/advanced inputs/schedules/receipts/manifest; Qwen headless/block callbacks; smallVAE config; catalogs/licenses/memory/residency; fixture generators/frozen author source; tests/CI/Windows ownership/sourcefreeze/protocol; docs/review records. Numeric payloads assessed via generating code/schema/consumers/hashes, not every number. All41 manifest entries match approved repo/rev/path/size/hash and checked-in author/fixture provenance hashes match files. Ordinary vsDMD distinct; fullFP32 nosecondLoRA. No additional concrete graph-equation or source-selection defects found. Native CUDA/MLX success, configuredVAE Metal fixture and actualPowerShell execution remain separate pending gates, not findings. Previously reported tests not rerun.

Confidence high for four findings, moderate for complete runtime correctness until final frozen hardware/platform gates execute. One full R1 fix batch assigned; fresh R2 whole-branch reviewer follows, no third review.

## Sole whole-branch round-one remediation batch

The reviewed source HEAD/tree, complete original report, accepted findings and
pending hardware boundaries above are preserved. This whole-branch R1 pass is
separate from the completed story-local reviews; there is no third review pass.
F-280 through F-283 are allocated in the append-only permanent registry.

F-280: the existing weightsmeta header parser now accepts an already-open reader;
its path API delegates to the same parser with unchanged header limit and file
size/range checks. Candle authenticates the exact prefix plus complete payload on
one open handle in at most 64KiB read chunks, retaining only header facts. Only
successful facts are cached by complete frozen repository/revision/logical path/
size/digest identity; every hit rechecks its retained pin. Request-bearing
validation, estimates and generation propagate cancellation through cold parsing
and hashing, using a typed Canceled result rather than retryable Interrupted IO.
The requestless APIs remain unchanged. Optional FP32 finite/shape proofs are also
retained only after complete approved-byte validation; cancellation is checked
through reads and finite scans. Actual tensor materialization still independently
authenticates the exact owned bytes it consumes, and conservative source-buffer
memory bounds are unchanged.

The counted tiny-stream CPU regression proves one cold complete authentication
and zero warm payload reads, while checking actual returned dtype/shape/byte
facts. Additional fixtures prove corrupt and canceled misses are not cached,
the same identity can later succeed, mutated pins cannot use cached facts, and
truncated/oversized prefixes fail before a large allocation.

F-281: the MLX advanced terminal helper indexes conditioning by prompt `i/2`
while noise receipts remain sample-indexed, and requires two conditioning rows,
four samples and images_per_prompt=2. The same helper is exercised without MLX
arrays by distinct prompt/noise receipts, plus wrong-order and truncated-row
refusals. No Metal is invoked by this regression.

F-282: from_pin verifies the approved frozen logical identity, size and digest
against the retained pin without requiring a physical suffix. Its CPU regression
uses real approved 240-byte config bytes under a content-addressed filename;
foreign logical content, revision mismatch, changed pin and wrong actual digest
are rejected. No model weights or authority bypass are used.

F-283: both numerical oracle comparators require exact shapes and counts and
finite actual/expected values before reducing errors. Pure CPU helper regressions
reject NaN, infinity, truncation and same-count wrong shapes. The complete Candle
ordinary FP32/BF16 and DMD fixtures are rerun through these stricter comparators;
MLX numerical fixtures are compiled only and run later on its admitted lane.

## Actual self-verification for this batch

CPU-only, `CARGO_BUILD_JOBS=2`; no local CUDA, Metal, real weights, model run,
Windows dispatch or comparison campaign was executed. Logs are retained under
`/Users/michael/.codex/epic-runs/sc-26321/implementation/native-integration/logs/`.

- Complete gen-core suite: 1199 passed, 2 ignored, including doctests. The final
  same-size foreign-digest and crossed-revision receipt assertions also passed in
  the focused receipt regression after the complete suite; production code did
  not change between these runs.
- Complete Candle MOSS package: 24 unit tests and one registry test passed; the
  real-weight terminal test remained ignored. This includes independent ordinary
  FP32/BF16 and DMD numerical fixtures, strict finite/shape checks and all three
  counted cold/warm/corrupt/cancel/header-bound regressions.
- MLX pure CPU oracle helper: one passed. MLX pure CPU supplied-receipt ordering
  integration test: one passed. Other MLX array/Metal and real-weight tests were
  compiled only and were not executed locally.
- Strict all-target Clippy and Rustdoc (`-D warnings`) passed for gen-core, both
  MOSS packages and Candle catalog. CPU runtime check and all 45 Candle catalog
  tests passed. Workflow policy: 65 passed. Workspace: 109 members passed.
  Append-only finding registry, 281-entry clock baseline, formatting and whitespace
  checks passed.

Initial diagnostic failures are retained: the new receipt-order helper initially
used usize instead of the contract's u32 images-per-prompt; the stricter Candle
comparator exposed an old synthetic sampler expectation without shape metadata.
The helper now uses u32 and that independent expectation explicitly declares its
known `[4, 4, 2, 3]` shape. Corrected checks above passed.

The separate MLX catalog lineage CI repair will be integrated before the final
round-two source freeze. Fresh whole-branch R2 review, Windows PowerShell CPU
self-test/CUDA CI execution and the admitted stationary-pin native terminal remain
pending; this record does not claim those gates passed.
