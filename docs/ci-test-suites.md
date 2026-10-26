# Routine and exhaustive test execution

Ordinary PR, merge-group, main and tag CI compile the original selected test
targets and execute bounded regressions. `scripts/ci/test_suites.json` identifies
166 exhaustive synthetic training and numerical cases by package, Cargo target,
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
| Shared MLX and media providers | Production QKV/P3 geometry matrices; whole-model checkpoint-gradient parity; convergence runs (including misleadingly named five-step `one_optimizer_step_descends` helpers); Qwen and Krea multistep convergence; FLUX3 all-factor/precision oracles | Primitive one-backward/optimizer state contracts; per-family fixed-geometry prologue regressions; short FLUX3 cancellation/resume/export controller contracts; scalar CPU oracles |

This is a reviewed execution classification, not a name-based filter. A test called
`training_precision` may only compare short RoPE schedules, while a wrapper called
`cpu_process` can spawn a complete multi-rank or 80-step training run. Each moved
case retains its original body and attributes. The JSON records the traced reason;
no blanket module ignore, new early return or correctness-test deletion is used.
Large memory/artifact contract fixtures remain ordinary even when they are slow.

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
7200s exhaustive. A timeout terminates the child process tree, including rank
fixtures. All CI jobs also have explicit deadlines; the ordinary Candle CPU and
hosted Metal jobs are bounded to 60 and 75 minutes, with larger explicit
exhaustive budgets of 150 and 180 minutes.

These bounds prevent an accidental campaign/hang from consuming the former
six-hour job default. They are not performance claims. Run 40249723473 recorded
62.3 minutes for hosted Metal, including 51.2 minutes in the broad MLX test step;
the expected reduction needs fresh Actions evidence after this classification.
This change was validated with source checks and synthetic runner tests, without
local model/GPU execution.

The PR package/dependency scope remains owned by `test_scope.py`; main, tags and
dispatches still select all packages. Broad all-target Clippy, rustdoc, CUDA
compile-only coverage, and package globs remain intact. Edits to the suite runner
or registry force full package selection so execution-policy changes cannot hide
behind an empty component scope.
