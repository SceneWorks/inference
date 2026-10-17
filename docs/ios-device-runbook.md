# iOS device runbook

Updated 2026-10-17 against inference feature base `481b04d6f` plus the sc-26270
campaign harness. These are implemented harness commands, not a claim the epic
is on main or that all models render on the phone. See [strategy](architecture/ios-strategy.md),
[specification](architecture/ios-project-spec.md), and [remaining work](ios-epics.md).

## Current model and evidence status

Target: **iPhone 16 Pro Max, 8 GB, iOS 26.6.2**. No successful target-device image
has been recorded. Registration is not admission or demonstrated fit.

| Model set | Feature status at this snapshot | Target-phone status |
|---|---|---|
| SANA base/sprint | Registered; source-bound admission and packed Gemma embeddings merged in #1272 (`23c7e3953`) | Admitted loader refuses without applicable capture; 1024/tile192 attempt terminated, cause unconfirmed |
| Anima Turbo | Registered without training; Base/Aesthetic excluded | No demonstrated phone fit |
| Iris 3B | q8/q4 engine fixes merged #1275 (`2f3265c71`); iOS registration #1277 merged | No demonstrated phone fit; depth access is explicitly diagnostic |
| SDXL, Illustrious-XL v1/v2, RealVisXL, RealVisXL Lightning | Low-memory engine fixes merged #1274 (`a54ad6c3a`); iOS registration #1278 merged | No demonstrated phone fit |
| Supra2-IMG | sc-26269 requires epic 24212 on inference main; not started | No implementation/fit claim here |

The legacy S3 image check drives SANA. The sc-26270 campaign protocol below now
selects every catalog provider/route/tier/geometry/mode/strategy explicitly. There
are still no `--iris`, `--sdxl`, `--anima` or `--supra2` shortcut flags. Legacy
Z-Image diagnostics are not part of this epic's supported model set.

## Campaign protocol and command matrix (sc-26270)

**Mechanism only; no new device measurements are recorded here.** No full-model
diagnostic is cleared to execute. Independently bounded primitive characterization
must precede any such experiment; a diagnostic flag, launch headroom, tensor byte
sum, engineering reserve, MLX allocator limit or a cap increase is not a permit.
The current empty artifact-bound evidence remains fail-closed. Anima/Iris have no
admitted iPhone footprint facade; their cases explicitly refuse before loading,
not via a raw registry bypass. Unsupported strategy/tier/task combinations remain
visible instead of being silently omitted. Malformed/missing snapshots are
`diagnostic_failure`, **not** next-up admission-refusal evidence.

Weights-free listing uses the actual `runtime-ios` catalog (including transforms
and composed task memory registrations), with the SDXL provider's route constant.
It creates no arrays and loads no model. A new registered provider/task becomes
visible automatically. Supra2 currently appears in the dependency record, not as
a fictitious supported provider; once registered it enumerates at 256².

```sh
export PATH="$HOME/.cargo/bin:$PATH"
# CPU-only catalog exporter; --list alone NEVER launches a device campaign.
python3 scripts/ios/campaign.py list > /tmp/ios-campaign-catalog.json
# This is the full, executable menu, not a handwritten campaign population.
python3 scripts/ios/campaign.py list --catalog /tmp/ios-campaign-catalog.json \
  --provider iris_3b --tier q8 --edge 768 --rung 3
```

| Axis | Values / source |
|---|---|
| Provider | Catalog generators SANA base/sprint, Anima Turbo, SDXL, Iris; registered restoration and composed depth tasks |
| SDXL route | `sdxl`, `illustrious_xl_v1`, `illustrious_xl_v2`, `realvisxl`, `realvisxl_lightning` (provider constant) |
| Tier | `q4`, `q8`, `bf16`; unsupported task/tier combinations stay explicit refusals |
| Geometry | 512², 768², 1024²; future registered Supra2: 256² |
| Mode | Descriptor conditioning kinds plus text, advertised prompt batch, registered restoration/depth; no edit-to-text downgrade |
| Strategy | `--rung 0..4`, translated through the provider's shared contract, not a copied cumulative ladder |

Provision complete caller-owned snapshots into a distinct slot for every
route/tier. The slot name is not proof of stored precision. Iris generation needs
its Qwen3-VL encoder, depth/restoration their task exports; SDXL may need its
fp16-fix VAE. A `components.json` beside the snapshot maps component IDs to
confined relative directories within that slot, e.g.
`{"text_encoder":"qwen3_vl","vae":"fp16_fix_vae"}`. Nothing downloads or
silently resolves a sibling artifact. Existing source inventories, deferred-read
checks and admitted request binding are preserved.

```sh
# After the host app has been installed (no campaign is started by pushing):
scripts/ios/push_model.sh "$SNAPSHOT" --campaign-route iris_3b --tier q8 --device "$DEVICE"
# Same mechanism for every route and tier in the actual listing:
scripts/ios/push_model.sh "$SDXL_SNAPSHOT" --campaign-route illustrious_xl_v2 --tier q4 --device "$DEVICE"
```

The following are **future session recipes, not instructions to run now**. Phone
availability alone does not authorize a full-model diagnostic. Use a fresh evidence
directory and save the checkout SHA, OS, entitlement, conversion/source receipts
and catalog JSON. Every selected case starts a fresh app process, before legacy
LLM checks or kernel smokes consume memory. The default is admitted, never
inherited `IOS_SMOKE_UNADMITTED`. Execute only after the relevant safety criteria
are independently met. The session builds/installs **once**, then launches one-shot
cases without `--terminate-existing`. `--timeout-seconds` (default 1800) is passed
to the device's cooperative `CancelFlag` deadline and controls host polling, not
an unrelated 600-second media loop. After a case returns, the app writes its
report and exits itself; the host verifies exit before advancing or finishing.
On deadline/cancellation failure it records a harness failure and stops further
launches if exit cannot be proved. It never hard-kills an active device render.
Timed-out host commands own a new process group, so cargo/xcodebuild/devicectl
children are terminated together rather than orphaned. Cleanup/exit verification
has a separate bounded grace window. Launch exit codes are retained.

`--memory-entitlement on|off` applies to the once-installed build. Every case
requires a fresh matching `launch-memory.txt`; missing copies are failures.
Each case/probe has its own host artifact directory alongside the JSONL file.
`--max-cases` counts both renders and next-up probes; exhausted cells are explicit
skips. Direct `run_smoke.sh --campaign-case` flags delegate to this same driver,
never to the legacy reinstall/terminate-existing path.

```sh
# One exact cold case (JSONL output must not already exist):
python3 scripts/ios/campaign.py run --catalog /tmp/ios-campaign-catalog.json \
  --case sana_1600m/sana_1600m/q4/512/text/2 --device "$DEVICE" \
  --output "$EVIDENCE/sana-512.jsonl"
# One selected model/tier ladder, including all its registered modes and rungs:
python3 scripts/ios/campaign.py run --catalog /tmp/ios-campaign-catalog.json \
  --provider iris_3b --tier q8 --device "$DEVICE" --output "$EVIDENCE/iris-q8.jsonl"
# Full catalog-driven campaign; no hand-maintained model list:
python3 scripts/ios/campaign.py run --catalog /tmp/ios-campaign-catalog.json \
  --all-cases --device "$DEVICE" --output "$EVIDENCE/all.jsonl"
# Explicit next-up REFUSAL-ONLY probe after an admitted completion. This NEVER
# loads weights at the larger geometry, even if its metadata preflight admits it.
python3 scripts/ios/campaign.py run --catalog /tmp/ios-campaign-catalog.json \
  --case sdxl/sdxl/q4/1024/text/2 --next-edge 1280 --device "$DEVICE" \
  --output "$EVIDENCE/sdxl-next-up.jsonl"
# Diagnostic selection is supported, but currently refuses every full-model case
# until an independently bounded primitive permit exists. Never reports prod PASS.
python3 scripts/ios/campaign.py run --catalog /tmp/ios-campaign-catalog.json \
  --case anima_turbo/anima_turbo/q4/512/text/4 --diagnostic --device "$DEVICE" \
  --output "$EVIDENCE/anima-diagnostic.jsonl"
python3 scripts/ios/campaign.py table --input "$EVIDENCE/all.jsonl"
```

Each case emits `CAMPAIGN_JSON` and JSONL with elapsed seconds, sampled MLX
active/cache/active+cache peaks, MLX allocator peak, sampled Darwin
`TASK_VM_INFO.phys_footprint` peak (words 36–37), the kernel's exact lifetime
`ledger_phys_footprint_peak` (words 42–43, requested count 44), and minimum
`os_proc_available_memory()` headroom. Rows also carry SANA layout identity or
SDXL profile/request key, selected strategy parameters, evidence revision,
entitlement, refusal-only marker, per-launch nonce, compiled build SHA and iOS
version. The nonce prevents a previous report with the same case/SHA from passing
freshness checks. Render
filenames are case-specific and registered in `renders.txt`, including each
image of a real two-prompt batch. **RSS and MLX active+cache are not kernel
physical footprint.** Sampling is 20 ms and can miss transients; these are
observations, not an independent instantaneous working-set proof. Unavailable
readings are null, never fabricated zeros. Statuses are `successful` (admitted
image only), `admitted_refused`, `diagnostic_success` (never production),
`diagnostic_failure`, or `skipped` (explicit host case budget). Missing/stale
reports, timeout, malformed output or a terminated process cannot prove admission
refusal. Missing physical-footprint/headroom telemetry cannot yield production
success. Noncompletion returns nonzero; a successful listing is not a render.
The table includes the largest **admitted** completed edge per route/tier/mode/
strategy, active/cache/combined columns and probe markers. Refusals have typed
`refusal_kind`: `capacity`, `no_evidence`, `mode`, or `strategy`. Only an
artifact-bound **capacity** refusal proves the next-up condition; missing
footprint evidence, unsupported modes/rungs, malformed sources and termination
never prove it. SANA profile errors and SDXL profile/request-binding errors are
failures before admission, not capacity refusals.
Commit the actual measured table only after the terminal campaign, not this
harness change. The 4096 MiB budget remains provisional; 6142/3374 MiB launch
readings remain unvalidated diagnostics, and no cap/table is invented here.

### Bounded synthetic primitive characterization (sc-16780)

**Primitive characterization only: not a model render, not admission evidence
for any model, not a cap.** The reserved key `primitive/sana_block_quantum` is
not a catalog case and opens no model file or snapshot. One cold launch runs one
SANA-block quantum boundary: a synthetic f32 activation `[1,8,8,2240]` (256 px
latent) against a resident F16 1x1 convolution weight `[11200,1,1,2240]` shaped
like `conv_inverted`, so MLX materializes the F32-promoted weight copy inside
`ios_quantum::run` (live headroom recheck, eval, stream synchronize, zero-cache
drain). Before **any** allocation it refuses unless live
`os_proc_available_memory()` covers the requirement derived from the budget
module: `sana_block_bytes(configured_block_geometry(sana_1600m, 1, 8, 8, 300))`
+ the probe's resident activation/weight bytes + `PROCESS_RESERVE_BYTES` =
416.876 + 48.398 + 512 = **977.275 MiB** (1,024,746,752 bytes). Residents are
charged because they are allocated before the quantum's own live recheck of
floor + reserve. The floor counts the loaded block's conv biases; a bias-free
count gives 416.791.

The row (`evidence_class: primitive_characterization`,
`model_admission_evidence: false`, `diagnostic: true`) records `before`,
`resident` (weights materialized) and `after` readings — available bytes, current
phys_footprint, exact kernel lifetime peak, MLX active/cache/peak — plus the
sampled extrema, entitlement, build SHA, iOS version and nonce.
`quantum_peak_over_baseline_bytes` is the new lifetime peak minus the pre-probe
footprint (null when an earlier excursion masks it), and
`unaccounted_by_mlx_peak_bytes` subtracts MLX's own peak: the device/driver
overhead the 512 MiB engineering reserve currently guesses. The host validates
the label, nonce and every reading, or rejects the row.

```sh
# ~5-10 min install (cargo ios build + xcodebuild), then one launch of seconds.
# Requires the phone unlocked; the increased-memory-limit entitlement is on.
SMOKE_BUNDLE_ID=com.michaeltrefry.SceneWorksSmoke DEVELOPMENT_TEAM=R2526H7YN9 \
  python3 scripts/ios/campaign.py primitive \
  --device 2EB6F380-7996-5CA7-9C12-39199D18199F --memory-entitlement on \
  --output "/tmp/ios-primitive-$(date +%Y%m%d-%H%M%S).jsonl"
```

## 1. Prerequisites and checkout

Use an Apple-silicon Mac, Xcode with iPhoneOS SDK and Metal toolchain, the
committed Rust toolchain, a paired/unlocked phone with Developer Mode enabled,
and an Apple Developer account signed into Xcode. XcodeGen 2.46 is installed on
the development Mac. Do not use a simulator for memory evidence.

```sh
# From the isolated checkout of feature/sc-16774-ios-runtime (or its docs story).
export PATH="$HOME/.cargo/bin:$PATH"
git rev-parse HEAD
rustc --version
xcrun -f metal
xcodegen --version
rustup target add aarch64-apple-ios
xcrun devicectl list devices

# Use the identifier returned above, not a device name or a guessed field number.
export DEVICE='<paired-device-identifier>'
export DEVELOPMENT_TEAM=R2526H7YN9
export SMOKE_BUNDLE_ID=com.michaeltrefry.SceneWorksSmoke
# Use a fresh evidence directory for this session.
export EVIDENCE="$HOME/ios-evidence/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$EVIDENCE"
git rev-parse HEAD > "$EVIDENCE/source-sha.txt"

# Record BEFORE invoking the harness, including attempts that fail before launch.
# This wrapper deliberately returns normally so recovery is not skipped by set -e.
# Its saved exit-status file, NOT the wrapper's return, is the harness outcome.
run_attempt() {
  ATTEMPT=$1; EXPECTED_LIMIT=$2; shift 2
  ATTEMPT_STARTED=$(date +%s)
  printf '%s\n' "$ATTEMPT_STARTED" > "$EVIDENCE/$ATTEMPT.started-epoch"
  printf '%s\n' "$EXPECTED_LIMIT" > "$EVIDENCE/$ATTEMPT.expected-entitlement"
  if "$@" > "$EVIDENCE/$ATTEMPT.log" 2>&1; then
    ATTEMPT_STATUS=0
  else
    ATTEMPT_STATUS=$?
  fi
  printf '%s\n' "$ATTEMPT_STATUS" > "$EVIDENCE/$ATTEMPT.exit-status"
  printf 'Harness exit %s; log: %s\n' "$ATTEMPT_STATUS" "$EVIDENCE/$ATTEMPT.log"
}
```

Signing is automatic. Other developers supply their own team/bundle ID. Changing
the bundle ID or uninstalling the app loses access to the existing model
container and requires a re-push. Keep the phone unlocked through launch.
Michael permits phone use anytime; that is not a standing Mac GPU loan. Any
host real-weight/GPU measurement needs a separately agreed bounded window and
external RSS/physical-footprint guard. No such loan is active for this docs work.

## 2. Install and reproduce the entitlement A/B

Run without `--media` to install the host and measure launch headroom without
requesting SANA renders. The LLM load is refused by default when no applicable
footprint estimate exists; model absence may also yield skipped checks.

```sh
# Start with a clean diagnostic environment, not inherited unadmitted/soak knobs.
unset IOS_SMOKE_UNADMITTED IOS_SMOKE_SANA_CAPTURE IOS_SMOKE_IMAGE_ONLY IOS_SMOKE_ONLY
unset IOS_SMOKE_SOAK_SECS MLX_GEN_SANA_DECODE_TILE
for entitlement in on off; do
  expected=YES; [ "$entitlement" = on ] || expected=NO
  for n in 1 2 3; do
    run_attempt "launch-$entitlement-$n" "$expected" \
      scripts/ios/run_smoke.sh --device "$DEVICE" --memory-entitlement "$entitlement"
    # Stop the sequence, but keep the failed status/log and enter section 6 recovery.
    if [ "$ATTEMPT_STATUS" -ne 0 ]; then break 2; fi
    cp /tmp/ios-smoke-launch-memory.txt "$EVIDENCE/$ATTEMPT.launch-memory.txt"
    cp /tmp/ios-smoke-report.txt "$EVIDENCE/$ATTEMPT.report.txt"
  done
done
```

Check each command's exit status and fresh report before proceeding; on failure
use the recovery steps below, not a prior `/tmp` report. `run_smoke.sh` checks
staticlib/report freshness and verifies the launch record's entitlement tag.
The control entitlement retains extended virtual addressing and removes only
increased-memory-limit. Use `--memory-entitlement off`, not an invented
`SMOKE_ENTITLEMENTS` override.

Observed S3 launch readings: **6142 MiB on**, **3374 MiB off**, three identical
readings each. These are `os_proc_available_memory()` at launch, not a validated
safe working-set cap. The runtime's provisional budget constants remain
unchanged. Do not replace them with 6142 or derive a safe render budget from the
entitlement delta. MLX allocator limits are not a proven jetsam barrier.

## 3. Provision and inspect the actual SANA snapshot

Inference downloads nothing. Supply an already fetched/converted complete SANA
component tree (`transformer/`, `text_encoder/`, `vae/`, plus required configs
and tokenizers). Do not rely on a directory's q4 name. The failed target attempt
had a dense **1,179,648,000-byte embedding**. Packed q4/q8 Gemma embedding support
is implemented, but that does not retroactively change the pushed artifact.

```sh
export SNAPSHOT='/absolute/path/to/provisioned/sana-component-tree'
# Header-only profiler: no model construction or tensor payload evaluation.
cargo run --locked -p mlx-gen-ios-catalog --example snapshot_profile -- "$SNAPSHOT" \
  > "$EVIDENCE/sana-profile.txt"
scripts/ios/push_model.sh "$SNAPSHOT" sana --device "$DEVICE" \
  > "$EVIDENCE/push-sana.log" 2>&1
```

Inspect `layout_sha256`, all three component byte totals and embedding dtype and
shape. Keep the conversion source and component precision with the receipt.
The profile is layout identity, not a hash of weight values or proof of image
quality. The helper stages symlinks as real files and checks transferred sizes.
The installed app must exist first; the remote tree is `Documents/sana/`.

## 4. Admitted SANA check

```sh
run_attempt sana-admitted YES env IOS_SMOKE_ONLY=sana \
  scripts/ios/run_smoke.sh --media --device "$DEVICE" --memory-entitlement on
if [ "$ATTEMPT_STATUS" -eq 0 ]; then
  cp /tmp/ios-smoke-report.txt "$EVIDENCE/$ATTEMPT.report.txt"
fi
# If nonzero, retain the saved status/log and use section 6 before another attempt.
```

The regular SANA harness configurations are 1024/tile192, 1024/tile128 and
512/tile256. The 512 row is off-distribution for the 1024 checkpoint and is not
a quality benchmark. On the current feature, the artifact-bound admitted path
refuses before loading: historical observations cannot authorize this snapshot
on this phone. Inspect the named checks for refusal or skip even when the report
header says PASS. That header is not a successful admitted image result.

## 5. Explicit UNADMITTED evidence capture

This is a diagnostic experiment, **not production admission or a successful
production render**, even if a PNG is produced. It requires a real device and
explicit `--unadmitted`. The capture path selects only 256/tile128, not the
legacy untiled experiment; there is no arbitrary size flag for SANA.

**Mandatory BEFORE capture:** independently establish and record a conservative
**total process working-set bound for the actual snapshot and exact capture
configuration**. The record must identify the layout/component precisions,
source, phone/OS/entitlement, 256/tile128 execution configuration, derivation and
margin, and account for weight conversions, simultaneously resident tensors,
activations, allocator/cache and driver/process overhead. Compare that bound
conservatively with the available capacity for this attempt. The header-only
payload/profile and the harness's automated payload/cap/headroom preflight
**DO NOT satisfy this prerequisite**: they cannot bound these transients.
Neither launch headroom nor the provisional constant establishes such a bound.
**Do not execute capture if this independent bound is missing or unsupported.**
No numerical safe bound is supplied by this runbook, and none is asserted to
exist for the target snapshot. Merely creating a receipt file is not evidence.

Only after the operator has reviewed and recorded that independently supported
bound, set `CAPTURE_BOUND_RECORD` to its receipt and run:

```sh
# Leave unset unless the mandatory total-working-set prerequisite above is met.
if [ -n "${CAPTURE_BOUND_RECORD:-}" ] && [ -s "$CAPTURE_BOUND_RECORD" ]; then
  cp "$CAPTURE_BOUND_RECORD" "$EVIDENCE/capture-total-working-set-bound.txt"
  run_attempt sana-UNADMITTED-capture YES env \
    IOS_SMOKE_ONLY=sana IOS_SMOKE_SANA_CAPTURE=1 \
    scripts/ios/run_smoke.sh --media --unadmitted \
    --device "$DEVICE" --memory-entitlement on
else
  printf '%s\n' 'NOT RUN: independent total-working-set bound is missing.' >&2
fi
# If attempted, retain its saved exit status even on failure and enter section 6.
```

Keep the exit status, launch-memory record, progress trace, full report and
render manifest. Generated PNGs are pulled to `~/Desktop/sana-ios/` by default
(`ARTIFACT_DIR` overrides this). Copy the current session's artifacts into its
evidence directory. Do not set `IOS_SMOKE_IMAGE_ONLY`: it selects a different,
legacy diagnostic sequence beginning with untiled decode.

No completion of this capture or target image success is asserted here. The
previous 1024/tile192 target attempt ended before a final report; termination
cause remains **unconfirmed**, not proven jetsam.

## 6. Recover evidence from a failed/terminated run

A missing report can mean launch failure, crash, termination or slow execution.
A progress file proves how far execution got, not why it stopped. Retrieve the
matching timestamped device crash/jetsam diagnostic before assigning a cause.

Use the `ATTEMPT` recorded **before that invocation**, never the current time or
another attempt's receipt. A failure before launch or before progress truncation
can leave both device files from an older run. A local copy's creation/mtime is
not freshness evidence. Preserve the device-reported modification timestamps in
the metadata JSON below; if metadata is unavailable, keep files **unverified**.

```sh
# Retrieval failures must not abort the remaining recovery steps under set -e.
unset DEVICE_PROGRESS_MTIME DEVICE_LAUNCH_MTIME  # Never reuse a prior transcription.
if xcrun devicectl device info files --device "$DEVICE" \
  --domain-type appDataContainer --domain-identifier "$SMOKE_BUNDLE_ID" \
  --subdirectory Documents --no-recurse \
  --json-output "$EVIDENCE/$ATTEMPT.device-files.json" \
  > "$EVIDENCE/$ATTEMPT.device-files.log" 2>&1; then
  printf '%s\n' 'Device metadata saved; inspect each source modification timestamp.'
else
  printf '%s\n' 'Metadata unavailable: recovered files remain UNVERIFIED.' >&2
fi
for name in sana-progress.txt launch-memory.txt; do
  if xcrun devicectl device copy from --device "$DEVICE" \
    --domain-type appDataContainer --domain-identifier "$SMOKE_BUNDLE_ID" \
    --source "Documents/$name" \
    --destination "$EVIDENCE/$ATTEMPT.$name.unverified" \
    > "$EVIDENCE/$ATTEMPT.$name.copy.log" 2>&1; then
    printf '%s\n' "Copied $name as UNVERIFIED, not current-attempt evidence."
  else
    printf '%s\n' "Unavailable: $name; preserve the copy log." >&2
  fi
done
cat "$EVIDENCE/$ATTEMPT.exit-status"  # Original harness outcome, never overwritten.
```

Inspect the saved device metadata for **each exact source file**, and record its
modification time as Unix epoch seconds, retaining the original device timestamp
and timezone alongside that transcription. Do not substitute the copy timestamp.
The following classifier takes that device-derived value, not a guessed JSON
field (Xcode metadata schemas may differ). Unknown/missing timestamps, times at
or before the pre-invocation boundary, and mismatched entitlement are not current
evidence. The equality case is conservatively ambiguous at one-second precision.

```sh
classify_recovered() {
  file=$1; device_mtime=$2; kind=$3
  started=$(cat "$EVIDENCE/$ATTEMPT.started-epoch")
  expected=$(cat "$EVIDENCE/$ATTEMPT.expected-entitlement")
  case "$device_mtime" in
    ''|*[!0-9]*) printf '%s\n' UNVERIFIED; return ;;
  esac
  if [ ! -f "$file" ]; then printf '%s\n' MISSING; return; fi
  if [ "$device_mtime" -le "$started" ]; then
    printf '%s\n' STALE_OR_AMBIGUOUS; return
  fi
  if [ "$kind" = launch ]; then
    case "$expected" in YES|NO) ;; *) printf '%s\n' UNVERIFIED; return ;; esac
    if ! grep -q "increased-memory-limit = $expected\$" "$file"; then
      printf '%s\n' WRONG_ENTITLEMENT; return
    fi
  fi
  printf '%s\n' CURRENT
}
# Leave unknown until transcribed from the preserved DEVICE metadata, separately.
classify_recovered "$EVIDENCE/$ATTEMPT.sana-progress.txt.unverified" \
  "${DEVICE_PROGRESS_MTIME:-}" progress > "$EVIDENCE/$ATTEMPT.progress-freshness"
classify_recovered "$EVIDENCE/$ATTEMPT.launch-memory.txt.unverified" \
  "${DEVICE_LAUNCH_MTIME:-}" launch > "$EVIDENCE/$ATTEMPT.launch-freshness"
```

Only `CURRENT` files may be attributed to this attempt. Fresh progress alone
does not establish the entitlement: require a `CURRENT` launch record matching
YES/NO as well; otherwise keep build/entitlement attribution unverified. Store
all stale/mismatched files as historical evidence, not this run's measurements.
The script also attempts `/tmp/ios-sana-progress.txt` on failure; apply the same
attribution discipline. Preserve logs immediately: `/tmp/ios-smoke-report.txt`
and launch-memory filenames are reused. Retrieve crash/termination reports through Xcode Devices and Simulators
or the phone's Analytics Data, matching app, time, device and source receipt.
Do not label a timeout or a breadcrumb as a jetsam diagnosis.

Common failures: locked device (`FBSOpenApplicationErrorDomain`), missing Xcode
account for the signing team, changed bundle/container, incomplete snapshot,
stale staticlib/report, or a macOS metallib packaged for iOS. Fix the specific
failure; never bypass freshness/platform checks to get a green report.

## 7. Evidence ledger: Mac engineering, not phone fit

Existing authorized Mac runs establish engine behavior, not E4 target admission.
No GPU measurements were run for this documentation update.

| Evidence | Exact source/provenance | Result and limitation |
|---|---|---|
| Iris conditioning, fixed-input denoiser, renders, resident bytes | `1d64ac261df315000ef68a81d19db2ee4ac95455`, merged via #1275 `2f3265c71`; local logs `/tmp/sc16774-rerun3` | All passed; q8 backbone+tower 7.283 GiB, q4 4.200 GiB. Not process footprint or phone fit. |
| SDXL/Illustrious sequential CFG and all-three low-memory pipeline | Earlier `770fa87c28a9c5efcbe8e942c10d36e86dd79827` run, `/tmp/sc16774-rerun2` | Sequential CFG byte-identical; all-three MLX peak 3.2289 GiB, NOT process footprint (sampled suite footprint reached 12.736 GiB). Predates final fp16 accumulation fix; NOT a remeasured final pipeline claim. |
| Final VAE parity | `7b5684a1bdc9bf9ec5273351be9cb05febbc962e`, merged via #1274 `a54ad6c3a`; `/tmp/sc16774-vae-final/source-and-build.txt`, `metrics.json`, `full.log` | 26/26 comparisons passed: absolute pixel error max ≤3 and mean ≤0.25, both in 8-bit pixel units (equivalently normalized max ≤3/255 and mean ≤0.25/255). fp16 tile256 overflow fixed by narrowly widening convolution accumulation to f32 with f16 storage. |
| Final VAE memory | Same final source/run | Genuinely untiled 1024 f32 reference peak 7.69 GiB; sweep maximum MLX peak 10.48 GiB; sampled process footprint 8.971 GiB. VAE test, not full-pipeline or target-phone evidence. |

Historical SANA data on the 12 GB iPhone 17 Pro Max is unbound planning data:
its active+cache values are **not** kernel `phys_footprint`, and it is not the
8 GB target. Do not import its old fit/no-fit verdicts into the target table.

## 8. Terminal device campaign — not yet measured

sc-26270 owns the model × tier campaign at 512²/768²/1024² (Supra2: 256²), the
largest completed size and next-size refusal, feature review, pin and main
integration. Retain every registered mode; report explicit refusals, not missing
cells. Add real rows here only after captures, with this schema:

| Source SHA / artifact layout and component tiers | Phone / OS / entitlement | Provider / mode / size / strategy | Time | MLX peak | Kernel footprint | Min headroom | Completed / refused / unconfirmed termination | Image / diagnostic receipt |
|---|---|---|---|---|---|---|---|---|
| Not yet captured | iPhone 16 Pro Max / iOS 26.6.2 | Full required model/tier set remains outstanding | — | — | — | — | No target image success recorded | — |

A successful diagnostic capture does not populate an admitted-success row.
Do not invent measurements, infer safe caps from launch headroom, or silently
mark Supra2 complete while epic 24212 is outstanding.
