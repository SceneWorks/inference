# StarVector terminal campaign contract

`starvector-terminal-corpus-v1.json` and `starvector-terminal-receipt-v1.schema.json` are the
cross-repository contract for SC-22261's one permanent-pin campaign. They are deliberately small:
the corpus identifies upstream SVG-Bench rows and checksums rather than committing source SVGs or
raster binaries.

The `starvector-terminal` profile in this repository is a native-provider preflight only. It
materializes the immutable 1B/8B snapshots, records their inventories, and invokes all four
ignored MLX/Candle real-weight conformance hooks in serial MLX → Candle/CUDA order. Its artifacts
must be bound into the final receipt, but do not establish quality or catalog admission alone.
The final CUDA job uploads `starvector-terminal-preflight-<sha>-<run>-<attempt>`, containing the two
canonical inventory files, all four distinct native-hook logs, and
`starvector-terminal-preflight.json`. The index records the exact inference head, workflow run and
attempt, plus relative paths and SHA-256 digests accepted by SceneWorks' fail-closed preflight.

After the permanent inference pin, SceneWorks owns the single end-to-end execution: it resolves
the same case identities into rasters, runs its 200 hostile-sanitizer and 60 prompt-composition
suites, performs rendering/SSIM/LPIPS measurement, records memory and lifecycle outcomes, and
publishes the receipt. The text-only upstream source remains excluded from image-quality acceptance
because it can intentionally exercise sanitizer rejects.

The receipt validator is intentionally stricter than a summary report. Each native backend records
exactly 120 ordered case records (`case_index` 0 through 119), with the immutable source row,
source-SVG/input-PNG hashes, provider-event transcript, typed finish reason, canonical SVG/preview
hashes, raw latency, and raw SSIM/LPIPS. A rejected row has null output hashes and metrics. Hardware
records include the raw accelerator probe hash and the exact total/peak bytes used to recompute the
headroom threshold. The validator recomputes validity (at least 95%), median SSIM (at least 0.85),
median LPIPS (at most 0.20), and nearest-rank p95 latency (at most 120 seconds), so a producer
cannot substitute an aggregate. Every run also records 20 ordered deterministic parity cases with
both preview hashes and rendered SSIM, each at least 0.995.

The checked-in corpus deterministically defines the exact content identities of all 200 hostile
SVG inputs and 60 prompt-composition inputs, not merely their counts. The receipt carries every
ordered hostile outcome; all must be `rejected` or `sanitized_inert`, and every case proves zero
partial artifact and zero inline SVG response. It also carries every prompt's raster and vector
lineage, raw CLIP cosines, and recomputed alignment loss. At least 57 prompts must be accepted and
their median alignment loss must not exceed 0.02. Top-level execution, preflight, metric, producer,
and artifact-manifest hashes bind one clean campaign run; the manifest must include every referenced
input, output, transcript, metric artifact, and raw hardware probe.

For each backend, 8B admission is recomputed from the aligned 1B/8B case records: at least 114 cases
must be accepted by both tiers, the observed relative median-LPIPS improvement must be at least 10%,
and validity may fall by no more than 0.02. The lower bound is a deterministic one-sided 95% paired
bootstrap: 10,000 full-size resamples, Numerical Recipes LCG seed `0x5a17c0de`, statistic
`(median(1B) - median(8B)) / median(1B)`, and the sorted sample at index 499. The bound must be
strictly positive; no producer-supplied bootstrap, median, validity, or p95 field is trusted.

## Failed-campaign lineage (receipt V2)

`starvector-terminal-receipt-v2.schema.json` keeps every V1 current-campaign requirement and adds
an explicit `campaign_lineage` closure. V1 remains accepted unchanged for historical clean receipts;
its schema and validator semantics are not retroactively tightened. New receipts use V2, including a
clean first attempt (`kind: clean` with both lineage arrays empty).

A replacement campaign uses `kind: failed_campaign_supersession`. `failed_predecessors` is ordered
oldest to newest by strictly increasing GitHub workflow run ID. Every predecessor records its safe
campaign ID, exact historical inference pin and SceneWorks head, non-successful SceneWorks workflow
repository/path/run/attempt/head/conclusion, typed failure code/phase/four-tuple identity, exact
campaign- and tuple-marker copies, and one or more exact GitHub source artifacts. GitHub artifact IDs
are decimal strings and digests use the API's `sha256:<hex>` representation. Cross-pin predecessors
are valid: history binds each pin rather than pretending that old evidence ran at the current pin.

The predecessor's quarantine root is exactly `quarantine/<campaign-id>`. Its entries are the sorted,
complete copies under these namespaces:

- `quarantine/<campaign-id>/markers/campaign/...`
- `quarantine/<campaign-id>/markers/tuple/...`
- `quarantine/<campaign-id>/source-artifacts/<role>/<artifact-id>/<artifact-name>`
- `quarantine/<campaign-id>/workflow-run.json`

The quarantine aggregate is SHA-256 over canonical key-sorted JSON containing exactly `{root,
entries}`. The workflow-run entry likewise hashes canonical key-sorted JSON for the workflow object.
Paths are relative, slash-separated, traversal-free canonical paths; copy and authority byte sizes
are positive integers.

There is exactly one append-only supersession record per failed predecessor. The records follow the
same order and form a single chain from each predecessor to the next predecessor, then to the current
campaign. Each record binds both campaign IDs, both SceneWorks heads, both inference pins, and an
authority at
`lineage/supersession-records/<old-campaign>-to-<new-campaign>.json`. Duplicate campaign IDs,
workflow executions, GitHub artifacts, quarantine paths/digests, or authority paths/digests are
replay failures. Forks, cycles, reordering, a current campaign among its own predecessors, and a
successful predecessor are also rejected.

The producer stores SHA-256 of canonical key-sorted `campaign_lineage` JSON in
`producer.campaign_lineage_sha256`. That digest appears at `lineage/campaign-lineage.json` in the
artifact manifest. The manifest also contains every quarantine entry and aggregate plus every
append-only supersession authority. Unlike V1 containment validation, V2 requires the complete entry
list to be sorted and to have exact path-and-digest equality with the recomputed closure: extra,
missing, duplicate, mixed-run, or mutated lineage entries fail closed. Quarantine digests are
disjoint from current metric, preflight, producer-transcript, run, hostile-suite, and prompt-suite
references, and all four fresh current tuple records remain mandatory. Failed evidence is retained
for provenance only; it can never satisfy current acceptance.
