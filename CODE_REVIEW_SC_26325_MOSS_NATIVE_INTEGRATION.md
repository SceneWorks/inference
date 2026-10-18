# sc-26325 inference story review

Reviewer: /root/native_integration_story_review, fresh gpt-6-astra/high, read-only source review.
Head: f8189bc1e521f0362a0fd8a09f37a9ebaaf2e642
Base: e0eb3d7c363d22eded33041a917af696a7ee8985
Verdict: CHANGES_REQUIRED. One finding; one batched self-verified fix assigned to original implementer. No routine re-review.

Selected Candle admission, authenticated source selection, largest owned shard transient accounting, both allocation/evaluation-free residency getters and MLX advanced smoke assertions were reviewed without concrete findings. Windows execution and final native acceptance remain pending; no builds, tests, GPU operations, dispatches or mutations were performed by the reviewer.

## [F-277] PID reuse can cause the supervisor to kill a foreign process

Priority: **P1**.

scripts/ci/moss-native-terminal.ps1:66, destructive consequence line89 at reviewed head.

Descendant discovery admits every process whose ParentProcessId matches any discovered PID. Windows retains a child's original parent PID after that parent exits. On a long-lived runner an unrelated surviving process can therefore have a parent PID subsequently reused by the new Cargo/test process. That foreign process enters the descendant IDs, is recorded as owned, passes the GPU allowlist, and is terminated by unconditional finally cleanup even after a successful build. Same-Identity confirms only the same foreign process originally misclassified; it does not prove ancestry.

Require creation-time ordering and verified parent identity at each ancestry edge before admission; keep immutable identities through discovery/cleanup and use held handles for termination where possible. Add a CPU supervisor fixture with a surviving process whose recorded parent PID matches a newer root, and prove it is neither admitted nor terminated while genuine descendants remain supported.

Confidence: high for the concrete discovery-to-termination path. Windows runtime behavior was not executed.

## [F-278] Terminal incorrectly requires the product main merge first

Priority: **P1**. Confidence: **high**. Coordinator's targeted integration
correction at reviewed HEAD `f8189bc1e521f0362a0fd8a09f37a9ebaaf2e642`,
`scripts/ci/moss-native-terminal.ps1:284` and workflow input description.

The terminal controller required the product revision to be an ancestor of
`origin/main`, and described its input as a SceneWorks-main revision. Approved
delivery ordering runs terminal acceptance after the final inference-main pin
lands in the product epic feature, before that feature merges to product main.
The check therefore makes the required pre-main acceptance impossible.

Require the exact checked-out product HEAD and input revision to equal the frozen
`origin/feature/sc-26321-moss-image-evaluation` head, while preserving all exact
inference-main pin checks. Positive and negative CPU SelfTest fixtures must cover
the frozen feature revision; product main is not an admission prerequisite.
This is the coordinator's concrete ordering finding, not a second independent
review. It is permanently allocated immediately after F-277.

## [F-279] Selected full-FP32 source closure fails at native admission and construction

Priority: **P1**. Confidence: **high**. This is accepted C2 from the single paired
consumer review, followed by the coordinator's bounded actual-constructor trace
at native HEAD `f8189bc1e521f0362a0fd8a09f37a9ebaaf2e642`.

The SAME filtered LoadSpec reaches selected admission AND actual normal generic
constructors: worker moss.rs212 prepares pins,279 moves spec; stream.rs753
forwards; cache.rs1389 invokes loader; base.rs10019/12675 supplies
inference_runtime::load. No parent/superset loader specification exists.

Candle selected admission is mode-aware (memory_strategy196), but load_on_device
lib59 calls Sources::new loader149 forcing OFF and demanding missing ordinary
pins. MLX static admission and model127 construction both prepare ordinary
sources; restoring pins then eagerly creates OFF renderer at165, later released
at310 before selected render. No simultaneous-renderer residency is alleged.
Both constructors must accept selected full-FP32 prepared source closure including
its published config; MLX must defer renderer until actual validated mode or
explicitly construct selected mode. Constructor facts must reflect actual
allocation. Generic loader/cache transaction and selected identity remain. Core
install prerequisite may remain catalog policy without requiring unused ordinary
weight materialization. Assigned inference F279, same paired integration fix
batch. Confidence high, source-only clarification.

## Sole remediation batch — sc-26325

The original independent verdict, reviewed HEAD, concrete scenario and confidence
above are preserved. F-277 through F-279 are permanently allocated in the append-only registry.
This is implementer remediation evidence, not a second independent review.

Descendant discovery now requires an immutable owned parent identity in the same
snapshot and child creation at or after that parent's creation. Every admitted
child retains its exact Process handle, then validates both held identities and
fresh Win32_Process identities before transferring ownership. A reused observed
PID never overwrites the original identity. Live GPU allowlisting rechecks the
actual current PID identity against the retained handle. Creation timestamps are
normalized to Win32_Process microsecond precision, without a fuzzy reuse window. Exact held-handle creation ticks additionally enforce
ordering below the Win32_Process timestamp precision; its CPU leaf fixture rejects
an earlier child even when both snapshot timestamp keys round to the same unit.
Cleanup uses only retained Process handles; there is no Stop-Process-by-PID path.

The production discovery and cleanup helpers are exercised by CPU-only SelfTest
fixtures: a foreign survivor with an older creation time and a reused parent PID
(and its child) is never admitted or terminated; genuine child/grandchild cleanup
still succeeds; a PID reused between snapshots cannot overwrite ownership or
admit another child; a parent replaced between snapshot and handle retention is
refused; an absent snapshot parent cannot authenticate a new ancestry edge.
Existing real CPU child success/failure/timeout fixtures remain wired.

Local PowerShell is unavailable. Source checks and workflow policy are run
locally; actual PowerShell CPU SelfTest execution remains explicitly pending
current-head Windows CI, where the existing windows-cuda-check job invokes it.
No GPU, Metal, model, dispatch, publication or tracker mutation occurs in this fix.
F-277/F-278 do not change model math or public APIs. F-279 changes private source
preparation and constructor residency; its affected CPU and compile gates are
rerun below. All prior evidence remains retained alongside the new fix evidence
under the epic implementation native-integration directory.

F-278 remediation: the controller checks exact product checkout/input/frozen
feature-head equality, with CPU fixture rejection of a different feature head.
Inference execution now selects the exact pinned main ancestor even if main has
advanced; trusted workflow-source guards run before any pinned controller code,
and execution and workflow revisions are recorded separately. A CPU fixture
accepts an exact ancestor and rejects off-main or unpinned execution sources.

The coordinator additionally authorized supervised, locked Cargo dependency fetch
inside this existing transport, scoped to CPU pre-lease setup after exact source
and pin checks. CPU SelfTest covers successful and failed setup restoring offline mode. Offline
mode is restored in finally; source/pin/lock cleanliness
is rechecked before the existing locked/offline metadata verification and builds.
This necessary preparation correction is documented as an implementation
decision, not a new code-review finding or another review pass.

F-279 remediation: both normal constructors prepare only authenticated shared
Qwen/tokenizer/scheduler/VAE sources and truthful common facts. MLX initializes
its renderer as None; Candle already deferred tensor allocation. Neither
constructor opens ordinary or optional DiT files. The exact retained LoadSpec is
used to prepare the explicit validated request's ordinary/mixed/full closure,
including the published selected configuration; MLX static selected admission
uses that same preparation. Default OFF still requires ordinary files when the
request is validated. Optional availability never enables acceleration and no
public load policy or registry redesign was introduced.

CPU regressions use the exact approved 240-byte frozen config and real pins.
They exercise the normal constructors' shared-preparation refusal with zero
ordinary pins, actual full-config ingestion/authentication, corrupt selected
config rejection, ignored invalid ordinary config on full, and the unchanged
ordinary/mixed source requirement. A successful full authenticated weight-header
admission/constructor still requires real frozen weights; these fixtures do not
bypass SHA authority or claim successful native model execution. The existing
terminal campaign exercises the actual successful filtered consumer spec.

The F-277 Windows SelfTest also runs a harmless real child/grandchild process
chain through production Owned-Run, releasing it only after verified live
ancestry is observed. This exercises real CIM/held-handle timestamp conversion
and valid descendant admission, in addition to synthetic PID-reuse fixtures.
Its execution remains explicitly pending Windows CI because local PowerShell is
unavailable.

## Actual affected self-verification

- Candle package: **20 unit + 1 registry passed**, 1 real-weight terminal ignored.
- MLX explicitly tensor-free CPU filters: **3 loader + 2 memory + 3 lifecycle
  passed**; numerical MLX/Metal tests were not executed.
- Candle catalog: **45 passed**.
- Strict all-target Clippy and `RUSTDOCFLAGS=-D warnings` rustdoc for both MOSS
  providers and Candle catalog: passed. CPU runtime check: passed.
- Workflow policy: **65 passed**; review registry unit tests: **17 passed**;
  append-only F-279 registry check, workspace **109**, clock baseline **281**,
  formatting and diff whitespace: passed.
- The unchanged full gen-core evidence (**1199 passed / 2 ignored**) remains
  retained from f818. No duplicate full-core run was needed for private loader,
  controller and documentation changes.

The first new fixture attempt omitted an explicitly configured file-component
pin and was correctly refused by LoadSpec finalization; the fixture now pins that
unselected placeholder and preserves strict production finalization. Both the
failure and passing rerun are retained. Local PowerShell is absent, so no claim
is made that the synthetic or actual-process Windows SelfTests have executed.
Current-head Windows CI must execute them and compile CUDA; final hardware
acceptance remains coordinator-owned. No model/GPU/Metal execution occurred.

Remediation source status: **F-277/F-278/F-279 addressed in the sole paired fix
batch and locally self-verified within the explicit CPU boundary**. This preserves
the original independent CHANGES_REQUIRED verdict as historical review evidence;
it is not a second review or a claim of final native acceptance.

## Minimal workflow CI validation repair after whole-branch R2

GitHub rejected workflow run `39140431560` before any job at published source
`3843e6e3052f20085542572930e2dec4591b09d0`: line31 used `runner.temp` in
job-level env, where the runner context is unavailable. GitHub's annotation was
`Line31 Col37: Unrecognized named-value: 'runner'`. The existing 65 policy
tests had passed but did not validate this new workflow's expression context.
This is a minimum self-verified CI repair, not another source review.

The first PowerShell step now derives the unique outside-checkout evidence root
from runtime RUNNER_TEMP and run/attempt environment values, creates it, exports
it through GITHUB_ENV and writes initialization evidence before either checkout.
Subsequent steps and always-upload share that exported root; checkout or source
guard failure therefore retains initialization evidence. Exact trusted-main/pin,
ancestry, device ownership and campaign protocols are unchanged.

Actionlint 1.7.12 with the explicit native self-hosted label config reproduces the
original invalid runner-context rejection and accepts the repaired workflow. The
new structural policy regression rejects direct, bracket and wrapped runner
expressions in job env while accepting runner expressions in step env, and checks
pre-checkout evidence initialization plus always-upload wiring. All **66 policy
tests passed** with Python3.14; whitespace checks passed. Initial local tooling
attempts using system Python3.9 (no tomllib) and Python3.12 (no PyYAML) are retained
as environment diagnostics. PowerShell is not installed locally; its Windows
SelfTest, CUDA compile and native terminal remain pending platform gates. No
model/Rust production source, GPU/Metal execution or Windows dispatch changed.
