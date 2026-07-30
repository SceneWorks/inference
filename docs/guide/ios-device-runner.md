# Setting up the iOS device runner

The `ios-device` CI lane and the `ios-device-heartbeat` workflow both run on a **self-hosted macOS
runner with a physical iPhone tethered to it**. This is how to set one up.

This tier exists because the two things that matter most about on-device inference — that the
cross-compiled Metal kernels are numerically correct, and what the memory and thermal behaviour
actually is — cannot be established anywhere else. The simulator has no Apple Neural Engine and a
different Metal implementation; a hosted runner has no device at all.

## What the tier does and does not gate

| Lane | Trigger | Runner | Gates PRs? |
|---|---|---|---|
| `ios-build` | every PR touching iOS paths | hosted `macos-15` | **yes** |
| `ios-device` | manual dispatch | self-hosted + device | no |
| `ios-device-heartbeat` | every 6 h | self-hosted + device | no (but fails loudly) |

`ios-device` is **manual on purpose**. One runner with one phone is a single point of failure, and
making it a required check would mean a sleeping Mac blocks every merge. `ios-build` — which
cross-compiles both triples and asserts the artifacts really target iOS — is the per-PR guard.

## Requirements

- macOS 26.2+ on Apple silicon (matches the shipping deployment target)
- Xcode 16+ with the Metal toolchain
- [`xcodegen`](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`)
- An iPhone running iOS 18.0+, **dedicated to the runner** — not a daily driver
- An Apple Developer account signed in to Xcode

## Setup

**1. Register the runner** with the labels the workflows select on:

```
self-hosted, macOS, ARM64, ios-device
```

**2. Sign in to Xcode** (Settings → Accounts). This is not optional and not the same as having a
certificate in the keychain: automatic signing resolves the team from the *account*, and a
certificate whose team has no matching account fails with `No Account for Team`. The heartbeat
checks for both.

**3. Prepare the device:**

- Settings → Privacy & Security → **Developer Mode: on**
- Pair it: `xcrun devicectl list devices` should show `available (paired)`
- **Disable auto-lock** (Settings → Display & Brightness → Auto-Lock → Never). A locked device
  builds and installs fine and then refuses to *launch* — the failure appears late and reads as a
  launch bug rather than a lock.

**4. Provision a model snapshot.** This workspace never fetches weights; a caller provisions every
path. Prepare a Q4 snapshot on the runner and push it into the app container:

```sh
# Dense source -> Q4 (7.5 GiB -> 2.64 GiB). Community *-MLX-4bit snapshots do NOT work: they
# quantize the embedding table, which the engine loads dense. Start from a dense bf16 export.
cargo run --release -p mlx-llm --example prepare_snapshot -- \
  <dense-snapshot-dir> <out-dir> q4

# Install the app once so its container exists, then push the snapshot in (~80 s for 2.6 GB).
./scripts/ios/run_smoke.sh
xcrun devicectl device copy to \
  --device <udid> \
  --domain-type appDataContainer --domain-identifier com.idkplay.SceneWorksSmoke \
  --source <out-dir> --destination Documents/
```

`copy to` **flattens** the directory: files land in `Documents/`, not `Documents/<name>/`. The
loader accepts both layouts, so either is fine.

Without a snapshot the lane still runs — the model checks skip and the Metal kernel checks
execute, so it degrades rather than lying.

**5. Verify:**

```sh
./scripts/ios/run_smoke.sh
```

Expected:

```
SMOKE: PASS
  [ok] metallib resolves + elementwise kernel -- sum(ones[4,4]) = 16
  [ok] f32 GEMM (steel) -- sum(64x64 matmul) = 262144 (expected 262144)
  [ok] bf16 GEMM (steel) -- sum(64x64 matmul) = 262144 (expected 262144)
  [ok] softmax reduction kernel -- sum(softmax(ones[4,8])) = 4
  [ok] runtime-ios generation -- ... 20.6 tok/s ... "Paris"
  [ok] core-llm conformance suite -- all always-on checks passed in 6.6s
```

## Why the heartbeat exists

A lane that never runs looks exactly like a lane that always passes. If the runner sleeps, loses
the device, or drops off the network, `ios-device` silently stops happening and nothing goes red.

`ios-device-heartbeat` converts that silence into a failure. Every six hours it checks only what
the device lane needs — runner reachable, device paired and unlocked, Developer Mode on, signing
identity and Xcode account present — in a few seconds, with no build. It deliberately does not
test inference; that is the device lane's job.

## Failure modes worth recognising

| Symptom | Cause |
|---|---|
| `No Account for Team "XXXX"` | A keychain certificate whose team has no Xcode account. Sign in; do not just install a certificate. |
| Builds and installs, then `FBSOpenApplicationErrorDomain error 7` | Device locked. Disable auto-lock. |
| Report shows results that do not match the current tree | A stale staticlib or a stale report. `run_smoke.sh` fails on both — if you see it, you bypassed the script. |
| `mlx.metallib is missing from the bundle` | The packaging phase did not run. The app *will* launch and fail at the first Metal op. |
| Model checks say `skipped` | No snapshot in the container (step 4). |
