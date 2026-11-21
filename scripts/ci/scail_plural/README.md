# SCAIL-2 plural memory qualification

Run the manual `model-inference-proofs.yml` workflow on the published candidate that includes the native fit profile and proof. The existing workflow stages and authenticates the six files in `SceneWorks/scail2-mlx@ce88cfdb1008f395e9c820e525e6db7b6695f7b3/q4` on each runner. A cold cache requires 23,993,093,306 bytes plus build space. Shader execution takes place under the host's existing Metal or physical CUDA execution lease.

On `nax-macos-2`, the MLX proof stores its Hub snapshot in `~/Library/Caches/SceneWorks/model-proof/huggingface/hub` under the runner account. The setup probes hard-link publication in the selected tier before any download. A cold cache needs 23,993,093,306 free bytes above the 20 GiB staging reserve. The authenticated model directory is `models--SceneWorks--scail2-mlx/snapshots/ce88cfdb1008f395e9c820e525e6db7b6695f7b3/q4` beneath that hub. To reuse it in the SceneWorks Tier 2 product proof, include the absolute `hub_root` from the staging receipt in the dispatch option `hubs`. Keep the other roots needed for product inputs in that ordered list.

Generate the exact dispatch input for a backend from the checked-out candidate and submit once per backend. Set `BACKEND` to `mlx` or `candle-cuda` and `QUALIFIED_REF` to the reviewed feature branch. Check the resulting run's `headSha` against the approved commit before treating its evidence as the candidate.

```bash
REQUEST=$(python3.12 -c 'import json,sys; from scripts.ci.scail_plural.contract import SELECTION,policy; print(json.dumps({"selection":SELECTION,"policy":policy(sys.argv[1])}))' "$BACKEND")
gh workflow run model-inference-proofs.yml --repo SceneWorks/inference --ref "$QUALIFIED_REF" -f backend="$BACKEND" -f request="$REQUEST"
```

Both arms render a 45-frame 832×480 driving clip and all its color masks with distinct ordered character references and masks at full resolution. Each reference paints its primary subject blue on white. Driving masks use the worker’s left-to-right person palette on black. Both positive and negative UMT5 branches must produce 512 tokens. The native registered provider consumes the bound video plan, runs four UniPC updates with guidance and shift 5 and seed 37, and returns RGB at 16 fps. MLX returns 48 frames. Candle returns 45. Each child reports the prepared file identities, actual axes and native execution report, including codec policy, per-character and driving carrier digests, synchronized render duration and output digest.

| Backend | Counts in execution order | Child limit | Host cap / reserve | Device cap / reserve |
| --- | --- | --- | --- | --- |
| Secondary M5 Max | 6, 2 | 40 min each | 96 / 12 GiB unified | 28 GiB native VAE planner ceiling |
| Windows CUDA GPU 1 | 6, 2, 3, 4, 5 | 40 min each | 64 / 8 GiB | 84 / 6 GiB |

Each arm compiles the release probe once. Each count gets a fresh child and an independently guarded and released lease. Watchdog cadence is 250 ms and termination grace is 2 seconds. Model time is capped at 80 minutes on MLX and 200 minutes on CUDA. The workflow's 360-minute outer limit includes file staging, builds and claims. A failed count stops that arm and preserves the diagnostic receipt.

The supervisor records Darwin kernel lifetime process footprint or Windows Job commit, per-process working-set/private-commit peaks, host samples and free-memory admission. MLX also records active and cached allocator bytes at each step, with lifetime active high-water. CUDA also reports pool used/reserved high-water, owned WDDM dedicated/shared samples, and physical card free-memory samples on the fixed GPU UUID. The worker applies its live external reserve separately. Inspect the physical counters, known native floors, claim release and correct completed output for every measured count. The `candidate_row_pending_physical_review` field is an input to review. WDDM samples leave between-poll non-pool usage unresolved. A completed count alone does not bound a different shape, driver, hardware or allocator regime. The packaging owner assigns the immutable evidence SHA-256 after reviewing the envelope.
