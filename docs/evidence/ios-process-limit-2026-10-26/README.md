# Entitlement A/B: observed OS process limit

On 2026-10-26 the physical iPhone 16 Pro Max (`iPhone17,2`), iOS 26.6.2 (`23G90`),
reported the following current process limits. Each value is the sum of
`os_proc_available_memory` and current `phys_footprint` in all three before/resident/after
pairs from that launch. Restoring the signed entitled app restored the first limit.

| Signed launch | Entitlement | Observed process limit |
|---|---|---:|
| First on | YES | 6,442,450,944 B / 6,144 MiB |
| Off control | NO | 3,539,992,576 B / 3,376 MiB |
| Restored on | YES | 6,442,450,944 B / 6,144 MiB |

The runtime ceiling now uses the **observed entitled OS process limit**. It is not a
validated safe render working set, a model-fit result, or a universal limit on other devices/OS
builds. The off control is retained as an A/B observation, not used as the entitled ceiling.
All raw rows still say `cap_validated:false`, `model_admission_evidence:false` and
`production_success:false`. Live headroom, reviewed scoped demand plus margin and quantum
guards remain necessary; empty anchor DATA and a null engineering policy still refuse admission.

These are exact copies of the original `phone-session/` receipts from source
`e49f5345bd254936f20c15c5ea84e76656faf01f`. [provenance.json](provenance.json) records
their SHA-256 digests. [receipt.md](receipt.md) retains the signed-app/staticlib/metallib
identities, nonces, launch/exit checks, all nine sample pairs and the A/B restoration history.
The three JSONL rows, raw reports and launch-memory files are unchanged. Large signed app
bundles remain in the original external receipt directory and are identified by its hashes.

Changing the runtime source constant requires one final SDK/harness Rust rebuild before
weighted captures, using the final source/build identity. It does not change the already
observed phone/OS/entitlement A/B or relabel the original source SHA. No model loaded and no
weighted inference or simulator run formed any of these process-limit observations.

Confidence: high for this observed device/OS/signing configuration. The OS can change its
current limit, and separately sampled footprint/headroom pairs can move; the live checks
remain active. Historical 6,142/3,374 MiB values were launch headroom alone.
