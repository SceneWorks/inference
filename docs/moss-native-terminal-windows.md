# Native MOSS Windows terminal acceptance (sc-26325)

Run once after the final inference-main pin is merged into the frozen SceneWorks
epic feature head, before the product feature merges to main. The
workflow is dispatched on inference main, refuses feature/story workflow refs, and never invokes a Python
model, comparison/calibration corpus, model downloader, or adoption gate.

`moss-native-terminal-windows.yml` selects the `cuda-windows-2` listener on
MICHAEL-TRX50. Labels and GitHub concurrency are not host ownership: the coordinator
must reconcile both repositories, all CI/listeners and external users before
claiming physical GPU0. Existing GPU1 policies do not authorize GPU0 work.

Dispatch on inference `main`, supplying the exact frozen product feature-head SHA
and strict data JSON:

```sh
gh workflow run moss-native-terminal-windows.yml -R SceneWorks/inference --ref main --json < native-dispatch.json
```

The input file has `sceneworks_revision` and `control_json` (a JSON string) only.
The control object's exact fields are:

```json
{
  "campaign": "sc-26325-native-final-unique-id",
  "inference_revision": "<exact final inference main ancestor SHA matching product Cargo/pin>",
  "snapshot_base": "E:\\sceneworks-evaluation\\moss\\sc-26322-reference-v9\\hub\\models--OpenMOSS-Team--MOSS-Image-Base\\snapshots\\1ceede80badddd6a7812069619edaa5633587192",
  "snapshot_shared": "E:\\sceneworks-evaluation\\moss\\sc-26322-reference-v9\\hub\\models--OpenMOSS-Team--MOSS-Image-SFT\\snapshots\\fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
  "snapshot_dmd": "E:\\sceneworks-evaluation\\moss\\sc-26322-reference-v9\\hub\\models--lodestones--MOSS-Image-SFT-8-steps-dmd\\snapshots\\5ff314b4c1470876b114b810146085db8a181a6e",
  "physical_gpu_index": 0,
  "gpu_uuid": "<fresh nvidia-smi UUID actually at physical index0>",
  "owner": "<responsible coordinator and fresh external lease identity>",
  "external_owners_reconciled": true,
  "deadline_utc": "<fresh UTC offset timestamp within six hours>",
  "timeout_seconds": 1800,
  "desktop_baseline": [],
  "baseline_max_memory_mib": 0,
  "baseline_max_utilization_percent": 0
}
```

Baseline values must come from fresh device queries, not the zero placeholders.
If a desktop process is explicitly admitted, each baseline row contains exactly
`pid` (integer), `name`, and `created_utc` matching fresh Win32_Process identity.
Missing/inaccessible or reused PID is refused. No process is implicitly exempted.
The host-local UUID lock is created exclusively under the existing
`E:\sceneworks-evaluation\moss\device-locks` authority. Stale or conflicting locks
are refused; they are not stolen. Every child ancestry edge requires a matching immutable parent identity and
creation-time ordering. Retained handles are rechecked before GPU allowlisting;
cleanup terminates only those exact held process objects, never a reopened PID. A new campaign directory is required; failed or
old evidence is never overwritten or automatically retried.

Snapshots above are historical candidates. The CPU-only Rust `native_inventory`
example rehashes all 41 shared frozen entries before admission, including four
published full-FP32 shards. It fails rather than acquiring missing files. The
actual loader then revalidates retained pins and authenticates the exact owned
bytes consumed. No model payload is trusted from a directory label or old receipt.
Build and inventory are locked/offline and use a fresh owned target; missing
toolchains or dependency preparation failures stop before a device lease. The installed CUDA12.9/MSVC2022 toolchain is
recorded; the inference architecture floor stays80 (native Blackwell rung comes
from its existing kernel builder). Build/inventory are bounded before the lease.

The trusted main workflow checks out exactly `control.inference_revision` and
verifies it is reachable from `origin/main` before executing its controller or
SelfTest. `GITHUB_SHA` is recorded separately as the workflow revision: unrelated
newer inference main commits do not force a product pin change. The product
checkout must equal both the supplied exact revision and the frozen
`origin/feature/sc-26321-moss-image-evaluation` head; product main is deliberately
not required at this pre-main terminal stage. Cargo.toml, lock and generated pin
must all match the exact selected inference execution source.

After those source/pin checks and before any build or lease, the controller runs
bounded supervised CPU dependency setup in that exact product checkout:

```powershell
cargo fetch --locked --target x86_64-pc-windows-msvc
```

Only that command temporarily sets `CARGO_NET_OFFLINE=false`; `finally` restores
it to true even on failure. This acquires ordinary Cargo source dependencies only,
with no models, reference runner, comparison corpus or dependency path patches.
The controller checks unchanged source, HEAD, pin and Cargo.lock afterwards, then
records and validates the exact offline resolution:

```powershell
cargo metadata --locked --offline --format-version 1 --filter-platform x86_64-pc-windows-msvc
```

Resolved inference package sources must all equal
`git+https://github.com/SceneWorks/inference?rev=<final SHA>#<final SHA>`.
Preparation stdout/stderr, source/lock hashes and failure status are retained.
A preparation failure stops before the lease; no separate remote endpoint,
lockfile rewrite or second transport mode is required.

Both backend and `sceneworks-worker` ignored test executables are compiled before
the lease, selected from actual Cargo JSON, hashed, and run directly
with fixed ignored exact filters under the same physical index0/UUID lease.
The product filter is
`image_jobs::moss_consumer_tests::moss_real_native_product_worker_campaign`;
it exercises preparation, standard boxed loader, typed admission, cache,
HTTP progress/events, real artifact writes/receipts and warm cancellation through
the actual worker. Product state/model caches are excluded from artifact capture. Logical
CUDA0 exposes only that UUID. Progress, Base/SFT ordinary, both DMD precisions,
OFF reentry, native advanced latents and typed cancellation are covered by the
existing Rust test. The checked-in `scripts/ci/moss-native-terminal-protocol.json` declares all
required backend and actual product-worker rows. The controller validates each
completed row against its actual mode/device/steps/output/sample/adapter receipt,
and each cancellation against the typed no-output/no-receipt result. Backend PNG
writes are disabled in the combined run; actual decoding and OFF equality checks
still execute. Product normal rows own the four 64px PNG outputs.
Advanced latent values are checked in memory; its receipt is saved. Existing
full-FP32 supplied-conditioning precision regression also remains CPU-covered.

The always-upload artifact `moss-native-cuda-<run>-<attempt>` contains source/pin/
compiler/binary/asset identity, stdout/stderr, lease/before-after device identity,
status/error/exit evidence, PNGs/receipts and a SHA manifest. The same evidence is
copied to the persistent new campaign. Weights, build targets and credentials are
never uploaded. Windows CI invokes controller `-SelfTest` with fake CPU children;
ordinary CI never runs the ignored native model test.


Warm worker admission calls `Generator::moss_image_owned_weight_bytes()` within
the serialized cache transaction. The compatibility default is None; both MOSS
providers return retained materialized weight counts only. MLX stores immutable
scalar counts after materializing exact arrays; the getter creates no Array/Vec,
evaluates no tensor and reads no weights. Candle uses its existing scalar census.
Credit remains bounded by the current committed snapshot; source/workspace and
other providers' allocations do not earn credit.
