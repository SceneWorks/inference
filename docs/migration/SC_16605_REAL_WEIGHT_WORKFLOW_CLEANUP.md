# Real-weight workflow cleanup (sc-16605)

The Stable Audio 3 macOS and Windows jobs now derive snapshot environment variables and revisions
from `release/real-weight-models.toml`. `scripts/release/export_model_snapshot_paths.py` uses the
same manifest loader as snapshot provisioning and writes stable runner-local paths under
`RUNNER_TEMP/model-snapshots/<model-key>/<manifest-revision>` to `GITHUB_ENV`. This removes the
workflow's second copy of every pinned SA3 revision while preserving separate runner-local caches.

Real-weight concurrency is scoped by both Git ref and dispatch profile. The ordinary CI lane
selector also exposes and summarizes its computed `real_weights` impact, but that signal is
informational: ordinary CI still has no job capable of launching the privileged self-hosted
real-weight runners. Those runs remain schedule- or operator-dispatch-only.

The historical SA3 test commentary removed from `ci.yml` remains in the migration evidence that
introduced each gate:

- `SC_14545_MEDIUM_PROVIDER.md` records provider, dtype, variant-quality, sampler, SAME, and
  primitive coverage.
- `SC_14547_REFERENCE_AUDIO_RESTYLE.md` records reference-audio sign and forwarding mutations.
- `SC_14548_AUDIO_EDIT_INPAINT.md` and `SC_14549_MULTI_REGION_AUDIO_EDIT.md` record editing geometry,
  conditioning, and multi-region mutations.
- `SC_14550_ADAPTER_FAMILY.md` records adapter-family and ordering evidence.
- `SC_15178_SA3_LISTENING_PROTOCOL.md` records the listening-stimulus controls.

The durable executable invariant is `scripts/tests/test_sa3_ci_target_coverage.py`: every SA3
integration target with a non-ignored case must be named in the weight-free CI command. The policy
test intentionally parses only the command, never comments, so coverage cannot be satisfied by
documentation drift.
