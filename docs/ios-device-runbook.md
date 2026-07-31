# iOS device runbook — running the smoke harness on hardware you own

Everything here has been run end-to-end on an iPhone 17 Pro Max (12 GB). Nothing in the repo is tied
to that device or to one developer's Apple account: the signing team is read from your Xcode
accounts, the Xcode project is generated (not committed), and the bundle id is a variable.

**The open question this exists to answer.** Every memory verdict in
[`ios-epics.md`](ios-epics.md) about an **8 GB** device is *derived from a 12 GB device and a Mac* —
there is no 8 GB hardware in the loop. It is the largest unverified claim in the initiative, and one
run on an 8 GB phone settles it. See [What to record](#what-to-record) for the three numbers that
matter.

---

## 1. Prerequisites

| | | check |
|---|---|---|
| macOS + Xcode 16+ | with the **Metal Toolchain** component (Xcode → Settings → Components) | `xcrun -f metal` |
| Rust 1.96.0 | pinned by `rust-toolchain.toml`, installed automatically by rustup | `rustc --version` |
| iOS target | `rustup target add aarch64-apple-ios` | `rustup target list --installed \| grep ios` |
| XcodeGen | `brew install xcodegen` | `xcodegen --version` |
| An Apple developer account | free/personal is enough — the memory entitlement is self-serve | Xcode → Settings → Accounts |
| iPhone with **Developer Mode** | Settings → Privacy & Security → Developer Mode, then pair over USB | `xcrun devicectl list devices` |

`mlx-sys` builds MLX from source with cmake on the first build (~5 minutes, cached afterwards). The
first `run_smoke.sh` is therefore slow and later ones are not.

> **Do not lower `MACOSX_DEPLOYMENT_TARGET`.** `.cargo/config.toml` pins it and the comments there are
> load-bearing — read them before changing anything about deployment target or `RUST_TEST_THREADS`.

## 2. Get the code

```sh
git clone https://github.com/SceneWorks/inference.git
cd inference
git checkout claude/ios-strategy      # not yet merged; blocked on SceneWorks/mlx-rs#23
```

## 3. Fetch the model snapshots

Inference **never downloads anything** — it takes caller-provisioned local paths (the epic-13657
boundary, enforced by `scripts/check-workspace.py`). Fetching is your job.

Both tiers are published in the SceneWorks org. **Use the `q4` subdirectory**, not the repo root:

```sh
pip install -U "huggingface_hub[cli]"

# SANA — the deciding measurement. ~5.4 GB.
hf download SceneWorks/Sana_1600M_1024px_mlx --include "q4/*" --local-dir ~/models/sana

# Z-Image-Turbo — optional, and expensive. ~5.9 GB.
hf download SceneWorks/z-image-turbo-mlx --include "q4/*" --local-dir ~/models/zimage
```

Verify before pushing — a partial LFS fetch produces pointer files of a few hundred bytes and fails
on device with a load error that reads nothing like "incomplete download":

| snapshot | component | expected |
|---|---|---|
| `sana/q4` | `text_encoder/gemma-2-2b-it.safetensors` | 2.16 GiB |
| | `transformer/diffusion_pytorch_model.safetensors` | 1.85 GiB |
| | `vae/diffusion_pytorch_model.safetensors` | 1.16 GiB |
| `zimage/q4` | `text_encoder/model.safetensors` | 2.11 GiB |
| | `transformer/model.safetensors` | 3.23 GiB |
| | `vae/model.safetensors` | 157 MiB |

```sh
find ~/models/sana/q4 -name '*.safetensors' -exec ls -lLh {} \;
```

## 4. Push the snapshots to the device

The app reads models from its own `Documents` container, so they are pushed once and persist across
reinstalls — **but not across an uninstall, and not across a bundle-id change.** Either wipes the
container and costs the full re-push.

```sh
scripts/ios/push_model.sh ~/models/sana/q4   sana
scripts/ios/push_model.sh ~/models/zimage/q4 zimage-q4     # only if running z-image
```

The directory name matters: the harness looks for a `Documents` subdirectory containing `sana` for
one lane and `zimage` for the other. It verifies each component's size after transfer.

Pushing ~5 GB over USB takes several minutes and the script prints per-component progress.

## 5. Run

```sh
# LLM lane only — fastest, no models needed beyond an optional LLM snapshot
scripts/ios/run_smoke.sh

# SANA. THIS IS THE ONE THAT ANSWERS THE 8 GB QUESTION.
IOS_SMOKE_ONLY=sana scripts/ios/run_smoke.sh --media

# Z-Image at 1024 (slow: ~4 minutes on a 12 GB device) and at 512 (~1 minute)
IOS_SMOKE_ONLY=zimage scripts/ios/run_smoke.sh --zimage
IOS_SMOKE_ONLY=zimage IOS_SMOKE_ZIMAGE_SIZE=512 scripts/ios/run_smoke.sh --zimage
```

**The device must be unlocked** when the app launches or SpringBoard denies it
(`FBSOpenApplicationErrorDomain error 7`). Rendered PNGs are pulled to `~/Desktop/sana-ios/`.

## 6. What to record

Three numbers, and the first one is a finding on its own.

**a. The cap.** The first line of every run:

```
per-app memory limit (OS-reported) -- os_proc_available_memory = 6136 MiB available at start of run
```

On a 12 GB device this is 6136 MiB. **Nobody has seen an 8 GB device's value** — every "8 GB" verdict
in `ios-epics.md` assumes ~4096 MiB. If the real number differs, several conclusions move.

**b. `peak MLX footprint` and `min headroom`, not `MLX peak`.** The report prints all three:

```
1024 tile128: 33.7s, MLX peak 2733 MiB, ... | peak MLX footprint 2860 MiB (active+cache), min headroom 3234 MiB
```

`MLX peak` tracks *live* allocation; jetsam reads `phys_footprint`, which also counts MLX's reuse
cache. That difference killed Z-Image four times while `MLX peak` said it had 3 GB to spare. **`min
headroom` is the ground truth** — how close the process actually came to being killed.

Reference values from the 12 GB device, for comparison:

| run | MLX peak | peak footprint | min headroom |
|---|---:|---:|---:|
| SANA 1024 tile128 | 2733 | 2860 | 3234 |
| SANA 512 tile256 | 3093 | 4113 | 1982 |
| Z-Image 1024 tile256 | 2901 | 3925 | 2146 |

**c. Whether it survived at all.** A jetsam kill shows up as `no report was produced` plus the
breadcrumbs written before death. If that happens, pull the trace — it names the phase and the
headroom it had:

```sh
xcrun devicectl device copy from --device <udid> --domain-type appDataContainer \
  --domain-identifier com.idkplay.SceneWorksSmoke \
  --source Documents/sana-progress.txt --destination /tmp/trace.txt
```

**Worth doing at 8 GB specifically:** the entitlement A/B. `com.apple.developer.kernel.increased-memory-limit`
is worth much more when the margin is small. Build the control with `SMOKE_ENTITLEMENTS= ` (empty)
and compare the reported cap:

```sh
SMOKE_ENTITLEMENTS= IOS_SMOKE_ONLY=sana scripts/ios/run_smoke.sh --media
```

## 7. Troubleshooting

**`No Account for Team` / signing fails.** The script reads the team from Xcode's signed-in accounts,
not the keychain — a certificate's OU can be a team your Xcode has no account for. Override
explicitly:

```sh
DEVELOPMENT_TEAM=XXXXXXXXXX scripts/ios/run_smoke.sh --media
```

**The bundle id is unavailable to your team.** `com.idkplay.SceneWorksSmoke` is the default. If
automatic signing cannot claim it:

```sh
export SMOKE_BUNDLE_ID=com.yourteam.SceneWorksSmoke
```

Every script follows it. **Only do this if signing actually fails** — a different bundle id means a
fresh `Documents` container and a full re-push of several GB.

**"device was not found".** `xcrun devicectl list devices` must show it as `available` or
`connected`; pass `--device <udid>` to disambiguate more than one.

**Install fails after switching teams.** A previously-installed copy signed by a different team must
be uninstalled first — **which deletes the pushed models.** The script prints the exact command.

**`stale staticlib`.** A Rust edit did not rebuild; check the cargo output above it for the real
failure. The guard exists because Xcode silently links whatever `.a` is on disk, which produces a
green run reporting the *previous* build's results.

---

## What this is measuring, in one paragraph

`ios-host/smoke` links `runtime-ios` — the shipped bundle — and drives it through
`provider_registry()` and the `Generator` contract rather than calling a loader directly, so a device
run exercises the composition a product actually consumes. On iOS it also calls
`runtime_ios::bound_mlx_to_platform_limits()` first, which binds MLX's memory and cache limits to
`os_proc_available_memory()`; without it MLX sizes them from device RAM (~11 GB on a 12 GB phone),
lets its reuse cache grow to fill the jetsam cap, and the app is killed holding gigabytes of
reclaimable memory. That call is why Z-Image runs at all. See
[`ios-epics.md`](ios-epics.md) for the full account.
