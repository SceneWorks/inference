# SC-23935 CPU `context_64` diagnostic baseline

This file records the completed Windows CPU comparison before the MTP and
pending prefill optimization. It is diagnostic evidence, not full campaign
acceptance. The one-off workflow on this branch is **NEVER MERGE**.

- Run: [35593206473](https://github.com/SceneWorks/inference/actions/runs/35593206473), successful on 2026-09-21.
- Diagnostic source: `214a3abb94b284ac77d045a0dfc2ca0dafa64c22`.
- Frozen parent model revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Artifact: `/Volumes/Models/Codex-builds/sc-23935/cpu-context-probe-35593206473/sc23935-cpu-context64-214a3abb94b284ac77d045a0dfc2ca0dafa64c22-35593206473-1`.
- Receipt SHA-256: `295b325c0f02289d9aade479a101bb20ebf15d5d070e93d2e10d9b4843e8e7ac`.

The selected `context_64` case had 1,187 prompt tokens and generated 6. It
returned exactly `NEBULA-47`; quality, stream, and functional acceptance flags
were true. Model load took 50.615 seconds. The selected case took 669.805
seconds: 630.771 seconds of prefill and 38.758 seconds of decode. The native
receipt elapsed 780.938 seconds, and the outer wrapper elapsed 818.453
seconds. Neither watchdog fired.

The sampled process RSS high-water mark over the full native lifetime was
166,932,815,872 bytes. Within the selected case, sampled RSS peaked at
112,225,685,504 bytes, 868,020,224 bytes above its first selected-case sample.
These are process RSS observations, not a native allocator peak claim.

The frozen snapshot inventory hashes matched before and after. The native PID
was 21832, exited zero, and was independently proven absent. GPU0's exact UUID
was rechecked with no compute tenants before the owned reservation was
released. The downloaded outer 25-file manifest, inner 4-file manifest, and
row seals independently verified without hash errors. The verification record
sets `diagnostic_valid=true`, `context_case_passed=true`, and
`full_campaign_accepted=false`.

The comparison after optimization must use the same frozen model, single case,
oracle, hardware, process boundaries, and watchdogs. Any time improvement is
measured against the 630.771-second prefill baseline above; this run alone does
not predict the optimized Windows result.
