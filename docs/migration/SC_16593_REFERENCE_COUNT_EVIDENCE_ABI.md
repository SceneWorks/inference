# SC-16593: reference-count evidence ABI 3

`MemoryGeometry::reference_count` is now a required typed request-geometry axis. The memory
calibration ABI is **3**; ABI-1 and ABI-2 records are stale and cannot authorize a request.

## Contract changes

- Callers must set `MemoryGeometry::reference_count` to the number of reference images consumed by
  the provider. Zero means that the request has no reference input.
- `MemoryRunContext::has_reference` remains as a compatibility summary and must equal
  `geometry.reference_count > 0`. The shared safety gate rejects inconsistent contexts.
- `MemoryRunContext::overlay` remains an evidence-identity axis. The shared safety gate rejects
  structured key/value payloads such as `references=2`; request data belongs in typed fields.
- Evidence must match the exact reference count. A record for two references does not authorize
  four references, and a reference-sensitive peak must not be extrapolated from unrelated tiers or
  geometry.
- `StructurallyNotApplicable` is invalid when a contract declares the corresponding lifecycle
  implementation hook. Implementable but unavailable ladder rungs must use `Missing`.

## FLUX.2-dev edit migration

The MLX provider no longer owns measured peak coefficients or overwrites the caller's
`predicted_peak_bytes`. It validates the loaded tier, the typed multi-reference route, and the
caller-supplied live budget against the incremental demand derived from the exact evidence-owned
absolute peak after removing only request-resident bytes already charged in the committed budget
snapshot. Non-resident strategies are reported as `Missing`.

SceneWorks must advance its inference pin and migrate atomically:

1. add `referenceCount` to the serialized evidence geometry and exact-match query;
2. populate `MemoryRunContext::geometry.reference_count` from the actual request and stop encoding
   it in `overlay`;
3. select a peak only from current-ABI, matching-fingerprint, matching-tier, matching-artifact,
   exact-geometry evidence; and
4. fail closed when no such record exists. In particular, Q4 measurements do not authorize Q8 or
   BF16, and two-reference measurements do not authorize four-reference requests.

Existing ABI-2 evidence remains useful as historical measurement provenance but is not admissible
for runtime selection.
