# Routine and exhaustive test execution

Ordinary PR, merge-group, main and tag CI compile the original selected test
targets and execute bounded regressions. `scripts/ci/test_suites.json` identifies
167 exhaustive synthetic training and numerical cases by package, Cargo target,
exact libtest name, source and body/helper rationale. These cases execute only
when an operator selects `test_suite: exhaustive`. Existing real-weight ignores,
scheduled/dispatch real-weight routes and the dedicated H3/MOSS numerical routes
keep their own selection and admission contracts.

## Classification boundary

The registry covers 15 packages on both backends:

| Area | Exhaustive execution | Retained routine regression examples |
| --- | --- | --- |
| MLX/Candle Iris | 80-step depth/restoration overfits; repeated interrupted/resumed trajectories; rank replication, reshard and failure campaigns; dtype/checkpoint/adapter/shape matrices | Two-update AdamW/Muon/LoRA/LoKr committed-oracle cases; two-step export/load/render contracts; missing-text-encoder and malformed-mesh refusals; tiny optimizer arithmetic |
| MLX/Candle LLM | Statistical sampling campaigns; attention/GatedDelta/Prism shape and dtype matrices; long prefills and compressed-cache/materialization measurements | Fixed-seed and argmax selection; short recurrence; dense paging lifecycle; fixed geometry and dispatch/refusal regressions; RoPE schedule/rounding checks |
| Shared MLX and media providers | Production QKV/P3 geometry matrices; whole-model checkpoint-gradient parity; convergence runs (including misleadingly named five-step `one_optimizer_step_descends` helpers); Qwen and Krea multistep convergence; FLUX3 repeated resume and trained-export/render runs; FLUX3 all-factor/precision oracles | Primitive one-backward/optimizer state contracts; per-family fixed-geometry prologue regressions; one-microbatch FLUX3 workflow/optimizer partition contracts; stored-gradient optimizer replay; scalar CPU oracles |

This is a reviewed execution classification, not a name-based filter. A test called
`training_precision` may only compare short RoPE schedules, while a wrapper called
`cpu_process` can spawn a complete multi-rank or 80-step training run. Each moved
case retains its original body and attributes. The JSON records the traced reason;
no blanket module ignore, new early return or correctness-test deletion is used.
The single necessary test-body exception is Iris' CUDA rank-device refusal case:
its existing child override selected literal physical device `0`, replacing the
lane's GPU1 UUID. The case now preserves and validates that incoming single-card
mapping, is compiled only with CUDA, and fails when its required device is absent.
A pure string-helper regression exercises valid UUID/index-1 and malformed,
absent, physical-index-0 and multi-device mappings without touching a GPU.
The two FLUX3 `real_loop_interrupts_with_pending_gradient_resumes_identically_and_reloads`
cases run five continuous microsteps against four interrupted plus one resumed
microstep; their `trained_export_loads_through_the_public_base_provider_and_changes_generation`
companions train five microsteps and compare three three-step public renders.
They belong to exhaustive validation on both backends. The two stored-gradient
AdamW replay cases only update small factor tensors twice and remain ordinary,
as do Candle's fixed tiny chunked-attention and additive-mask/stride contracts.
Large memory/artifact contract fixtures remain ordinary even when they are slow.
Iris' MLX `restoration_train_lifecycle::activation_checkpointing_leaves_the_run_unchanged`
is exhaustive: full/LoRA/LoKr × plain/checkpointed runs six four-step training
trajectories, then checks export/reload and whole-tensor equality. Its same-named
Candle counterpart compares only two two-step trajectories and remains routine.

## Operator validation

To execute every registered exhaustive case on its applicable backend, dispatch
the existing CI workflow against the desired ref:

```sh
gh workflow run ci.yml --repo SceneWorks/inference --ref YOUR_REF \
  -f lanes=all -f test_suite=exhaustive -f synthetic_w4a8=none
```

`lanes=windows-cuda` with the same `test_suite=exhaustive` selects the Candle CUDA
subset, preserving physical GPU1 binding, device sharing for rank fixtures and
the offline synthetic cache. A full `lanes=all` run includes hosted Candle CPU,
hosted MLX, and the existing manual NAX/CUDA jobs. The explicit exhaustive mode
executes every registry entry present in each selected package/target/feature
graph by exact name, including the portable sampler cases in Candle's
`device_sampler` integration target. It does not substitute a default
`cargo test -- --ignored` invocation. Run ordinary CI as well for the retained
regressions; exhaustive mode selects the separated cases rather than repeating
the routine suite. Dedicated H3/MOSS and real-weight acceptance remain separate.

For a focused local invocation on a suitable, available host:

```sh
python3 scripts/ci/test_suites.py routine -- cargo test --locked --lib --tests -p mlx-gen-iris
python3 scripts/ci/test_suites.py exhaustive -- cargo test --locked --lib --tests -p mlx-gen-iris
python3 scripts/ci/test_suites.py check
```

The runner compiles with `--no-run --message-format=json`, then validates actual
libtest membership and ignored status before executing anything. Target renames,
missing cases, stale ignore changes and substring collisions fail visibly. CUDA
cases are selected using the compiled artifact's features; MLX cases require
macOS. Direct execution retains Cargo's package working directory, manifest and
loader environment, and forces one test thread, including child-process fixtures.
`--no-fail-fast` retains aggregate failure collection without turning a failed
test into a successful suite.

## Budgets and evidence

Healthy retained Candle CPU artifact/memory targets took 411s (`candle-gen-flux2`),
320s (`candle-gen-krea`) and 300s (`candle-gen-qwen-image`) in main run
40240078690. Routine target execution has a 900s ceiling, above those observations;
exhaustive cases have a 600s ceiling each. Listing and metadata calls are bounded
to 60s/120s, compilation to 2400s, and whole suite invocations to 2700s routine or
7200s exhaustive by default. The two Candle CPU matrix members and hosted
`Test MLX packages` step explicitly allow 3600s routine and 7200s exhaustive per
invocation for their full native package graphs. A timeout terminates
the child process tree, including rank fixtures. All CI jobs also have explicit
deadlines; ordinary Candle CPU members and hosted Metal are bounded to 75 minutes,
with larger explicit exhaustive budgets of 150 and 180 minutes respectively.

The CPU split preserves the existing full graph: 20 base native targets and 76 gen
native targets, with the original package selectors, `--lib`/`--tests` flags and
`-j 1` linker limit. They run independently on hosted runners under the existing
`candle-cpu-test` job ID. Matrix fail-fast is disabled, and either member's failure
fails that required dependency of `CI gate`; neither member masks failures.

In run 40310828863, the former serial job's base group took 15m05s (9m11s
compilation, 5m53s execution). The gen group's compilation took another 11m27s;
the 60-minute job limit canceled it while its healthy Qwen Image target had run
for 4m32s. Compared with completed baseline run 40240078690, 63 of 76 gen targets
had finished and 13 remained or were active; those 13 baseline runtimes sum to
337.90s. The matched completed targets ran at 1.509 times the baseline aggregate
duration. Applying that observed ratio to the remaining work estimates about
64 minutes for the serial job and 48 minutes for gen's invocation. These are
estimates, not completion measurements. The independent 60-minute invocation
budgets and 75-minute job bounds allow headroom above that gen estimate, including
setup and cache restoration, while retaining the 900s target watchdog. Metal job
and explicit exhaustive budgets are unchanged.

The pinned `Swatinem/rust-cache` action includes `key: ${{ matrix.group }}` in
both primary and restore prefixes, alongside the job, platform and Rust environment.
Base and gen therefore save distinct main-only dependency caches: a fast base run
cannot fill gen's immutable key with only base dependencies. This changes the
previous cache prefix, so the first runs will be cold until main saves each group;
PR runs still only restore. Gen also loses dependencies compiled by the old prior
base step. The first split run must measure that extra compilation and confirm the
bounds; no cache-hit or exact wall-clock reduction is claimed in advance.

Hosted Metal job 121596995674 passed in 54m29s; `Test MLX packages` ran from
13:32:14 to 14:13:21 (41m07s), compared with 36m55s in the prior run. That leaves
only 3m53s below the default 45-minute routine suite limit. Its explicit 60-minute
routine invocation budget provides headroom for the full native graph while
preserving the 75-minute job bound, 900s target watchdog and 120-minute exhaustive
invocation budget. The override belongs only to that step; other callers retain
their existing budgets.

These bounds prevent an accidental campaign/hang from consuming the former
six-hour job default. They are not performance claims. This change was validated
with source checks and synthetic runner tests, without local model/GPU execution.

The PR package/dependency scope remains owned by `test_scope.py`; main, tags and
dispatches still select all packages. Broad all-target Clippy, rustdoc, CUDA
compile-only coverage, and package globs remain intact. Edits to the suite runner
or registry force full package selection so execution-policy changes cannot hide
behind an empty component scope.
