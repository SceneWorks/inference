# iOS device runbook

Updated 2026-10-16 against inference feature head `a54ad6c3a`. These are the S3
harness commands that exist there, not a claim the epic is on main or that all
models render on the phone. See [strategy](architecture/ios-strategy.md),
[specification](architecture/ios-project-spec.md), and [remaining work](ios-epics.md).

## Current model and evidence status

Target: **iPhone 16 Pro Max, 8 GB, iOS 26.6.2**. No successful target-device image
has been recorded. Registration is not admission or demonstrated fit.

| Model set | Feature status at this snapshot | Target-phone status |
|---|---|---|
| SANA base/sprint | Registered; source-bound admission and packed Gemma embeddings merged in #1272 (`23c7e3953`) | Admitted loader refuses without applicable capture; 1024/tile192 attempt terminated, cause unconfirmed |
| Anima Turbo | Registered without training; Base/Aesthetic excluded | No demonstrated phone fit |
| Iris 3B | q8/q4 engine fixes merged #1275 (`2f3265c71`); iOS registration #1277 open | No demonstrated phone fit; depth access is explicitly diagnostic |
| SDXL, Illustrious-XL v1/v2, RealVisXL, RealVisXL Lightning | Low-memory engine fixes merged #1274 (`a54ad6c3a`); iOS registration #1278 open | No demonstrated phone fit |
| Supra2-IMG | sc-26269 requires epic 24212 on inference main; not started | No implementation/fit claim here |

The S3 image harness currently drives SANA. Do not invent `--iris`, `--sdxl`,
`--anima` or `--supra2` flags: catalog registration does not add harness commands.
Legacy Z-Image diagnostic support is not part of this epic's supported model set.
The terminal campaign must cover every required model, tier and mode; missing
harness coverage is not permission to omit a model.

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
set -e  # Stop immediately rather than copying a stale file after a failed run.
for entitlement in on off; do
  for n in 1 2 3; do
    scripts/ios/run_smoke.sh --device "$DEVICE" --memory-entitlement "$entitlement" \
      > "$EVIDENCE/launch-$entitlement-$n.log" 2>&1
    cp /tmp/ios-smoke-launch-memory.txt "$EVIDENCE/launch-$entitlement-$n.txt"
    cp /tmp/ios-smoke-report.txt "$EVIDENCE/report-$entitlement-$n.txt"
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
IOS_SMOKE_ONLY=sana scripts/ios/run_smoke.sh --media \
  --device "$DEVICE" --memory-entitlement on \
  > "$EVIDENCE/sana-admitted.log" 2>&1
cp /tmp/ios-smoke-report.txt "$EVIDENCE/sana-admitted-report.txt"
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

```sh
IOS_SMOKE_ONLY=sana IOS_SMOKE_SANA_CAPTURE=1 \
  scripts/ios/run_smoke.sh --media --unadmitted \
  --device "$DEVICE" --memory-entitlement on \
  > "$EVIDENCE/sana-UNADMITTED-capture.log" 2>&1
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

```sh
xcrun devicectl device copy from --device "$DEVICE" \
  --domain-type appDataContainer --domain-identifier "$SMOKE_BUNDLE_ID" \
  --source Documents/sana-progress.txt --destination "$EVIDENCE/sana-progress.txt"
xcrun devicectl device copy from --device "$DEVICE" \
  --domain-type appDataContainer --domain-identifier "$SMOKE_BUNDLE_ID" \
  --source Documents/launch-memory.txt --destination "$EVIDENCE/launch-memory-recovery.txt"
```

The script also attempts `/tmp/ios-sana-progress.txt` on failure. Preserve the
logs immediately: `/tmp/ios-smoke-report.txt` and launch-memory filenames are
reused. Retrieve crash/termination reports through Xcode Devices and Simulators
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
| Final VAE parity | `7b5684a1bdc9bf9ec5273351be9cb05febbc962e`, merged via #1274 `a54ad6c3a`; `/tmp/sc16774-vae-final/source-and-build.txt`, `metrics.json`, `full.log` | 26/26 comparisons passed at max ≤3/255, mean ≤0.25. fp16 tile256 overflow fixed by narrowly widening convolution accumulation to f32 with f16 storage. |
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
