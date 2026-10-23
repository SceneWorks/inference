# iOS runtime — remaining integration work

The operational source of truth is
[epic 16774](https://app.shortcut.com/trefry/epic/16774), re-planned 2026-10-15.
This replaces the old phase diary and contradictory fit verdicts; it is not a
second implementation plan. Current implementation context is
`feature/sc-16774-ios-runtime` at `a54ad6c3a` (2026-10-16), not main.

- Architecture: [strategy](architecture/ios-strategy.md)
- Components/contracts: [project specification](architecture/ios-project-spec.md)
- Reproduction and evidence: [device runbook](ios-device-runbook.md)
- Operations: [device harness guide](guide/ios-device-runner.md)

## Outstanding dependencies

| Work | State at this documentation snapshot | Completion evidence still needed |
|---|---|---|
| sc-26267 Iris iOS ladder/catalog | Registration PR [#1277](https://github.com/SceneWorks/inference/pull/1277) open; q8/q4 engine fixes already merged | Merged public facade/modes, including the separate explicitly diagnostic depth surface; do not infer target fit from Mac tests |
| sc-26268 SDXL-family iOS catalog | Engine fixes merged; registration PR [#1278](https://github.com/SceneWorks/inference/pull/1278) open | Final admitted facade and all family routes on feature; no current phone capture authorizes product loads |
| sc-26269 Supra2-IMG iOS catalog | Not started; depends on epic 24212 implementation reaching inference main | Actual implementation dependency, iOS registration and coverage, not removal from the model set |
| sc-16781 documentation | Draft against merged feature plus explicitly labeled pending registrations | Reconcile final merged interfaces before claiming final-state documentation |
| sc-26270 terminal integration | Not complete | Device campaign, feature-end E1–E7 review, inference integration, single SceneWorks pin bump, SceneWorks integration and tracker readback |

## Target-device evidence still required

The target is iPhone 16 Pro Max (8 GB), not the historical 12 GB phone. Launch
available memory was 6142 MiB with increased-memory-limit and 3374 MiB without,
three readings each on iOS 26.6.2. This does not validate a safe rendering cap;
the provisional runtime budget stays unchanged. Source-bound SANA admission and
packed Gemma q4/q8 support are implemented, but an artifact-bound target-phone
working-set capture and successful image are still absent.

The target SANA 1024/tile192 attempt terminated with cause unconfirmed. A new
explicit UNADMITTED capture is diagnostic, never an admitted production success.
Mac Iris fidelity/resident-byte tests and final SDXL VAE parity have passed;
they do not close the phone campaign or establish target fit. The runbook records
their exact source provenance and the older full-pipeline measurement's limits.

The terminal campaign covers SANA, Anima Turbo, Iris 3B, Z-Image Turbo, the
complete SDXL family and Supra2-IMG across served tiers and modes as ANCHORS (the
recorded memory-anchor rule): per family the fully resident and the fully
optimized configuration at 1024² (Supra2: 256²), the optimized one three times,
every other rung and size derived (runbook: *Anchors and derivation*). Record
elapsed time, MLX peak, kernel footprint per phase and minimum headroom with
artifact/device identity.
Do not substitute old active+cache observations or launch readings for that
campaign. Keep missing cells visibly outstanding rather than inventing results.

No product FFI, Swift package, app-side provisioning, SenseNova registration or
Anima Base/Aesthetic work is included in this epic. Supra2's implementation is
owned by epic 24212, but its iOS registration remains this epic's dependency.
No docs-only change performs a pin bump, main merge or measurement campaign.
