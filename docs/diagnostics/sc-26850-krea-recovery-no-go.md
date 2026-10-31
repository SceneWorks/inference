# Krea bounded recovery: measured NO-GO

The candidate remains **NO-GO** and the default remains Reference. The 25% speed
target is false; acceptance of that miss is unresolved. This record does not
complete the epic or authorize further Krea optimizer work. The conservative
user budget is exhausted: 14 prior charges + 6 new updates = **20/20**, zero left.

| Observation | Source / run (attempt 1) | Steady mean |
| --- | --- | --- |
| Original Reference, segment 0 | M3 `69c25a1d87f761464cb200802dfbb04125e9f54c`, [41127490702](https://github.com/SceneWorks/inference/actions/runs/41127490702) | 8.907143 s |
| Shared Reference, segment 1 | Same M3 source/run | 9.09337525 s |
| Actual recovery candidate, segment 2 | M4 `39219b9e6550bc33f340ca2533c2d3422a5579a6`, [41144582520](https://github.com/SceneWorks/inference/actions/runs/41144582520) | 27.997881325 s |

Each segment has two warmup and four timed updates. M3 remains an unaccepted
partial run; its 12 original reference updates/scalars are reused with the
separate unchanged-source/input qualification and 0.5 microsecond mean rounding
uncertainty. M4 is a cross-run six-update recovery, not a new matched 18-update
invocation. Consumer snapshot: `30e3ae50ea4050066f6be4510dc6e47ac2b387d1`.

M4 job `124366759673` on `cuda-windows-2` completed all six optimizer updates,
with 224/224 finite gradients each, 28 candidate dispatches each (168 total),
and source-defined loss parity true. The actual candidate adapter changed and
was saved, reloaded and applied to one 1024 RGB, 28-denoise-step, seed-7 sample.
There is no recovered shared/default Reference adapter. Rust reported one PASS,
zero failures/ignored tests, 307.28 seconds, and BODY_COMPLETED. These are bounded
smoke observations, without a convergence or isolated-benchmark claim.

The runtime fit used a 91 GiB cap plus 2 GiB reserve. The sampled 92,830 MiB peak
is global device usage during training/preparation, not attributable process
usage. The source estimate of 3.96 GiB less retention is not measured A/B memory
savings.

The original outer workflow failed at collector scalar validation: Rust's f32
learning rate serialized as `0.0010000000474974513`, while Python expected the
f64 literal `.001`. The repaired validator checks the exact canonical f32 wire
value, without general tolerance. A trusted offline audit of immutable artifact
`13020738927` used the clean M4 registry/configuration and original source
verifier, then the repaired scalar validator; original Rust invocation PASS and
corrected metrics VALID are separate observations. No artifact code or model was
executed. All original output hashes verified (10 files, 72,646,976 bytes), and
the evidence tree remained unchanged. Audit output is
`krea-recovery-corrected-audit-41144582520-1.json` in the external epic evidence
directory; it records artifact hashes and the separate auditor identity/hash.

Original workflow failure, receipt/result/metrics, `accepted_model_run=false`,
case acceptance false and `unaccepted-incomplete` remain unchanged. Input
provenance remains successful-reader-binding; no named revision, checkpoint
byte-integrity attestation or terminal-model-evidence claim is established.
