# Operating the iOS device harness

The current per-change automated protection is the `ios_build` selection in
[ci.yml](../../.github/workflows/ci.yml): hosted macOS cross-builds the device and
simulator targets and checks the metallib's platform triple. It is not a phone
run. This feature does **not** install an `ios-device` CI lane or a six-hour
heartbeat workflow; do not register a dedicated runner on the assumption those
workflows exist.

For the actual connected-phone procedure, use the
[device runbook](../ios-device-runbook.md). It contains checkout prerequisites,
automatic signing, snapshot transfer, entitlement A/B, admitted SANA refusal,
explicit UNADMITTED capture and failed-run evidence retrieval. The
[project specification](../architecture/ios-project-spec.md) defines the bundle
and diagnostic host boundaries.

## Operator checklist

1. Use a paired iPhone 16 Pro Max (8 GB, current evidence OS iOS 26.6.2), enable
   Developer Mode, and keep it unlocked through launch. Do not treat a simulator
   as the target phone. The phone need not be dedicated to CI.
2. Confirm Xcode is signed in to the selected developer team. The current local
   automatic-signing settings are `DEVELOPMENT_TEAM=R2526H7YN9` and
   `SMOKE_BUNDLE_ID=com.michaeltrefry.SceneWorksSmoke`; other developers use their
   own values. XcodeGen 2.46 is installed locally; projects are generated.
3. Record the source SHA, target, OS and entitlement before running. Do not
   rewrite platform deployment settings to match a stale guide.
4. Install the app before pushing a snapshot. Use `push_model.sh`, not a manual
   directory copy that can flatten paths or leave dangling symlinks. Bundle-ID
   changes and uninstalling require re-provisioning the app container.
5. Keep one evidence directory per session and copy fresh `/tmp` reports before
   the next run overwrites them. Inspect named checks for skipped/refused cases:
   PASS without an admitted image is not a demonstrated model fit.
6. Keep diagnostics separate from admitted operation. `--unadmitted` and
   `IOS_SMOKE_SANA_CAPTURE` are explicit evidence collection, not production
   access. Do not accidentally inherit image-only, soak or old tile overrides.
7. A missing report plus a progress breadcrumb is not a termination diagnosis.
   Match the device crash/jetsam record before assigning a cause.

Michael's standing phone permission does not authorize a host GPU campaign.
State a bounded loan and obtain approval each session before Mac real-weight
measurements; use an external RSS/physical-footprint guard, and return the GPU
at the agreed deadline. This documentation change runs neither devices nor GPUs.

## Separately dispatched SANA simulator image

S3's simulator image is a manual `real-weights.yml` profile on a provisioned
Mac runner. Use an already provisioned immutable `sana-1600m-mlx` snapshot and an
exact iPhone simulator model name available on that runner. The workflow selects
its runtime there, creates and later deletes its own simulator without touching
the template's apps or data. Until the dedicated reusable
workflow is on the default branch, dispatch through the registered workflow at
the exact feature ref:

```sh
gh workflow run real-weights.yml --ref <feature-ref> \
  -f profile=ios-simulator-sana \
  -f ios_pricing_compile_binding='{"source_sha":"<40-hex-feature-head>","simulator":"iPhone 17 Pro Max","snapshot":"~/.cache/huggingface/hub/models--SceneWorks--Sana_1600M_1024px_mlx/snapshots/ac421696dd6eb0b41f446f4c45a53ccc057d82a1"}'
```

The route checks out that SHA, verifies the canonical snapshot root and the BF16
component files against the pinned public Hub metadata, then copies its
BF16 component tree into a fresh simulator app container and compares every file
against the source inventory. It builds a simulator metallib and installs without
launching. The controller then claims the exact app PID before releasing its
model-load gate, samples physical footprint and host reserve, and accepts only a
nonce-bound SANA report with a nondegenerate 256×256 PNG. Its artifact includes
the source, snapshot inventory, binary and PNG identities. This witness proves
simulator image functionality only; it establishes no physical-phone memory
admission, fit, kernel correctness, or performance claim.

## Optional tooling is not campaign evidence

`scripts/ios/soak.sh --secs 300 --device <identifier>` exists for sustained LLM
diagnostics, but loads unadmitted. It is not an image campaign or a source of
measured battery-energy claims. Its Instruments trace is best-effort and does
not establish mWh per token. Do not run it automatically during S3 reproduction.
There is no device scheduler or heartbeat to recover here; the terminal epic
story owns the deliberate phone campaign and its recorded results.
