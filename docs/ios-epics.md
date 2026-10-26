# iOS runtime — remaining integration work

The operational source of truth is
[epic 16774](https://app.shortcut.com/trefry/epic/16774), re-planned 2026-10-15.
This replaces the old phase diary and contradictory fit verdicts; it is not a
second implementation plan. Current implementation context is
`feature/sc-16774-ios-runtime` plus the sc-26270 terminal candidate, reconciled
2026-10-26. Final main integration and device evidence remain pending.

- Architecture: [strategy](architecture/ios-strategy.md)
- Components/contracts: [project specification](architecture/ios-project-spec.md)
- Reproduction and evidence: [device runbook](ios-device-runbook.md)
- Operations: [device harness guide](guide/ios-device-runner.md)

## Outstanding dependencies

| Work | State at this documentation snapshot | Completion evidence still needed |
|---|---|---|
| sc-26267 Iris iOS ladder/catalog | Registration PR [#1277](https://github.com/SceneWorks/inference/pull/1277) merged at `481b04d6`; q8/q4 engine fixes merged | Target-phone admission and image/task evidence remain terminal work |
| sc-26268 SDXL-family iOS catalog | Engine fixes and registration PR [#1278](https://github.com/SceneWorks/inference/pull/1278) merged at `be196d97` | No current phone capture authorizes product loads |
| sc-26269 Supra2-IMG iOS catalog | Base provider delivered separately; iOS registration explicitly user-deferred and non-blocking | Remains open with its full registration acceptance; does not gate general closeout |
| sc-16781 documentation | Draft against merged feature plus explicitly labeled pending registrations | Reconcile final merged interfaces before claiming final-state documentation |
| sc-26270 terminal integration | Not complete | Device endpoints and derived claims, feature-end E1–E8 review, inference integration, single permanent SceneWorks pin bump, SceneWorks integration and tracker readback |

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
complete SDXL family across served tiers and modes as ANCHORS (the
recorded memory-anchor rule): per family the fully resident and the fully
optimized configuration at 1024², the optimized one twice,
every other rung and size derived (runbook: *Anchors and derivation*). Record
elapsed time, MLX peak, kernel footprint per phase and minimum headroom with
artifact/device identity.
Do not substitute old active+cache observations or launch readings for that
campaign. Keep missing cells visibly outstanding rather than inventing results.

No product FFI, Swift package, app-side provisioning, SenseNova registration or
Anima Base/Aesthetic work is included in this epic. Supra2's implementation was
delivered separately; its iOS registration remains open in sc-26269 as the
explicitly user-approved non-blocking follow-up.
No docs-only change performs a pin bump, main merge or measurement campaign.
