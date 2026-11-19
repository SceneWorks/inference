# The release profile unstripped (rust-lang/rust#157750): Rust 1.96/LLVM 22 mis-strips macOS
# proc-macro dylibs, and `castaway` then failed with E0463 on `rustversion`. The profile change
# also rebuilds the stripped artifacts the persistent target already holds.
export CARGO_PROFILE_RELEASE_STRIP=none
export CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_STRIP=none
# Report-only: the host the loan ran on and its memory before the first build.
{ sysctl -n machdep.cpu.brand_string; sysctl -n hw.memsize; sw_vers -productVersion; } > "$DECODE_BENCH_OUT/host.txt" || true
# Every benchmark process runs under the external phys_footprint guard at its row's
# `memory_cap_gib` (SIGTERM only, 45 min wall); the budget starts after the build.
python3.12 -I scripts/release/speculative_bench_campaign.py run --plan "$DECODE_BENCH_OUT/plan.json" --epic-root . --epic-sha "$GITHUB_SHA" --epic-target-dir "$DECODE_BENCH_TARGET_DIR" --output "$DECODE_BENCH_OUT" --stop-starting-after-minutes "$DECODE_BENCH_BUDGET_MINUTES" --guard "python3.12 -I scripts/ci/mlx_tree_guard.py {cap_gib} 2700 --"
