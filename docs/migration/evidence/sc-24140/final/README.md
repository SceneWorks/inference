# sc-24140 — final epic evidence (epic sc-24128, fast decode on Blackwell)

**Hardware: RTX Pro 6000 / sm_120 only. Nothing here extrapolates to any other GPU.**

This directory is the terminal, authoritative evidence bundle for epic sc-24128. It supersedes
every earlier decode-perf / parity / perplexity measurement taken during the epic (see
[Superseded](#superseded) below). All numbers in the tables below are copied verbatim from the
sealed files in this directory (`at2-campaign/INDEX.md`, `at2-campaign/index.json`,
`at4/*.json`, `at3/*.log`, `admission/*.json`) — nothing here is invented; re-derive any figure
from the copied JSON/MD/log files to check it.

## What was measured, and where

* **Runtime commit:** `907e51114e7afa28211f99b126525e6e5ebdc9b1` (tag `runtime-2026.09.1-rc.1`,
  merge of PR #1045), clean tree. This is **R′** — the release candidate re-measured after
  inference #1044 fixed load-admission honesty (see [Superseded](#superseded)).
* **Head binary:** `decode_bench-1a9956edc819f0d9.exe`, sha256
  `5067faa60c68dd5abb5e5bdf65b4fc138c1e0e12a6f6bdd9e0066d97455a3321`, built from R′
  (`run.json.binary.build.git_sha == 907e51114e7afa28211f99b126525e6e5ebdc9b1`, `git_dirty: false`).
* **S1 baseline binary:** `decode_bench-a589f4c981fe54a8.exe`, sha256
  `88b88acce9c08732132f787cd61902e504cedb7088894e57ec26a5281aca974c`, built at
  `d2b8cb3352e1b43983efd8ac833122e95db53ae7` (pre-epic; the binary embeds no runtime SHA, per the
  epic's build-provenance rule for baselines with no build script).
* **Models:** Qwen/Qwen3.8-27B (`bonsai-qwen38-parent` @ `1d4bf0f2ff60`, config sha256
  `191e0af23210…`, qwen35 family) and Qwen/Qwen3-8B (`qwen3-8b` @ `b968826d9c46`, config sha256
  `f7c4eadfbbf5…`, llama family).
* **Idle-box procedure.** The campaign only starts once the box is quiet: `idle-poll.log` polls
  the self-hosted-runner queue and both GPUs every ~2 minutes; the run waited for `workers=0` and
  no in-progress/queued CI dispatch, reaching `IDLE_REACHED` at `10:10:31 EDT`. The campaign
  (`at2-campaign.driver.log`) started 26 seconds later at `10:10:57` and exited `CAMPAIGN_EXIT=0`
  / `VERIFY_EXIT=0` at `10:37:30`. Through that whole window, `at2-contamination-watch.log` polls
  every ~31s and shows `workers=0 gh_active=0` on every line — no CI worker and no other GitHub
  Actions job touched either GPU while the campaign ran.

## AT2 — decode-perf campaign

Full sealed campaign in [`at2-campaign/`](at2-campaign/) (`INDEX.md`, `index.json`, `SEAL.json`,
`baselines/`, `runs/`), 12 cells (2 models × 3 formats × 2 CUDA-graph settings). All rows: greedy,
256 new tokens (27B: 97 prompt tokens; 8B: 53 prompt tokens). `n/a` = not applicable to that row
(reference/StepModel rows have no acceptance/proposer). Peak memory is `cuMemGetInfo` total-free
at the row's last generated token (device-wide, weights included), as recorded in `INDEX.md`.

**Headline (bf16, the stable comparator — see caveat below):**

| comparison | tok/s | vs S1 baseline (10.71 tok/s) |
|---|---|---|
| 27B bf16 MTP K=2, graphs off | 25.04 | **2.34×** |
| 27B bf16 MTP K=2, graphs on | 26.81 | **2.50×** |
| 27B best row overall: NVFP4 MTP K=2, graphs on | 28.93 | **2.70×** |
| 8B bf16 reference (no speculation), graphs off, vs S1 baseline 27.18 tok/s | 72.54 | **2.67×** |

**Caveat (recorded, not smoothed over):** single-sample NVFP4/Q8 rows varied by up to ±30%
between identical binaries and identical token outputs run at rc.0 (`e9faeabdb`) and rc.1
(`907e5111…`) — the generated tokens are bit-identical between the two runs, only the timing
differs (run-to-run GPU clocking / thermal noise on a single-sample bench, not a regression). The
**bf16 figures are the stable headline**; NVFP4/Q8 tok/s below should be read as one noisy sample
each, not a precise ranking.

### Qwen3.8-27B (`bonsai-qwen38-parent`) — S1 baseline vs head, by format × CUDA graphs

S1 baseline (bf16, no CUDA-graph runner in this binary): **MTP off (reference) 10.71 tok/s**,
53.34 GiB. (The baseline binary also ran its own MTP K=1..5 rows — 12.74 / 16.06 / 14.65 / 11.45 /
12.88 tok/s — see `at2-campaign/index.json` `baselines[0]`; the head-vs-S1 speedups above use the
non-speculative 10.71 tok/s row, matching how the epic reported the headline.)

| format | graphs | row | tok/s | acceptance | syncs/tok | peak mem |
|---|---|---|---|---|---|---|
| bf16 | off | MTP off (reference) | 15.74 | n/a | 1.00 | 53.31 GiB |
| bf16 | off | MTP off (StepModel) | 15.63 | n/a | 1.00 | 53.31 GiB |
| bf16 | off | MTP K=1 | 21.15 | 0.827 | 0.55 | 53.31 GiB |
| bf16 | off | MTP K=2 | 25.04 | 0.757 | 0.40 | 53.47 GiB |
| bf16 | off | MTP K=3 | 23.94 | 0.569 | 0.38 | 53.56 GiB |
| bf16 | off | MTP K=4 | 24.29 | 0.467 | 0.36 | 53.75 GiB |
| bf16 | off | MTP K=5 | 23.77 | 0.465 | 0.30 | 54.03 GiB |
| bf16 | off | n-gram K=3 | 15.04 | 0.189 | 0.79 | 53.56 GiB |
| bf16 | on | MTP off (reference) | 15.67 | n/a | 1.00 | 53.25 GiB |
| bf16 | on | MTP off (StepModel) | 15.47 | n/a | 1.00 | 53.25 GiB |
| bf16 | on | MTP K=1 | 22.08 | 0.827 | 0.55 | 53.34 GiB |
| bf16 | on | MTP K=2 | 26.81 | 0.757 | 0.40 | 53.44 GiB |
| bf16 | on | MTP K=3 | 25.23 | 0.569 | 0.38 | 53.59 GiB |
| bf16 | on | MTP K=4 | 24.18 | 0.467 | 0.36 | 53.81 GiB |
| bf16 | on | MTP K=5 | 26.06 | 0.465 | 0.30 | 53.94 GiB |
| bf16 | on | n-gram K=3 | 15.71 | 0.189 | 0.79 | 53.59 GiB |
| Q8 | off | MTP off (reference) | 10.12 | n/a | 1.00 | 40.25 GiB |
| Q8 | off | MTP off (StepModel) | 11.08 | n/a | 1.00 | 40.25 GiB |
| Q8 | off | MTP K=1 | 14.93 | 0.796 | 0.56 | 40.25 GiB |
| Q8 | off | MTP K=2 | 15.89 | 0.592 | 0.46 | 40.25 GiB |
| Q8 | off | MTP K=3 | 20.69 | 0.569 | 0.38 | 40.25 GiB |
| Q8 | off | MTP K=4 | 16.08 | 0.426 | 0.38 | 40.25 GiB |
| Q8 | off | MTP K=5 | 17.62 | 0.405 | 0.34 | 40.25 GiB |
| Q8 | off | n-gram K=3 | 10.68 | 0.190 | 0.81 | 40.25 GiB |
| Q8 | on | MTP off (reference) | 13.92 | n/a | 1.00 | 40.53 GiB |
| Q8 | on | MTP off (StepModel) | 14.92 | n/a | 1.00 | 40.53 GiB |
| Q8 | on | MTP K=1 | 18.14 | 0.796 | 0.56 | 40.53 GiB |
| Q8 | on | MTP K=2 | 20.02 | 0.592 | 0.46 | 40.53 GiB |
| Q8 | on | MTP K=3 | 22.13 | 0.569 | 0.38 | 40.53 GiB |
| Q8 | on | MTP K=4 | 22.57 | 0.426 | 0.38 | 40.53 GiB |
| Q8 | on | MTP K=5 | 20.67 | 0.405 | 0.34 | 40.53 GiB |
| Q8 | on | n-gram K=3 | 19.12 | 0.190 | 0.81 | 40.53 GiB |
| NVFP4 | off | MTP off (reference) | 16.04 | n/a | 1.00 | 23.94 GiB |
| NVFP4 | off | MTP off (StepModel) | 22.03 | n/a | 1.00 | 23.94 GiB |
| NVFP4 | off | MTP K=1 | 25.86 | 0.776 | 0.57 | 23.94 GiB |
| NVFP4 | off | MTP K=2 | 22.20 | 0.621 | 0.45 | 24.00 GiB |
| NVFP4 | off | MTP K=3 | 21.90 | 0.505 | 0.40 | 24.07 GiB |
| NVFP4 | off | MTP K=4 | 24.73 | 0.391 | 0.39 | 24.22 GiB |
| NVFP4 | off | MTP K=5 | 19.81 | 0.304 | 0.40 | 24.47 GiB |
| NVFP4 | off | n-gram K=3 | 18.19 | 0.150 | 0.84 | 24.07 GiB |
| NVFP4 | on | MTP off (reference) | 16.95 | n/a | 1.00 | 24.07 GiB |
| NVFP4 | on | MTP off (StepModel) | 24.14 | n/a | 1.00 | 24.07 GiB |
| NVFP4 | on | MTP K=1 | 24.81 | 0.776 | 0.57 | 24.07 GiB |
| **NVFP4** | **on** | **MTP K=2 (best row)** | **28.93** | 0.621 | 0.45 | 24.13 GiB |
| NVFP4 | on | MTP K=3 | 28.40 | 0.505 | 0.40 | 24.16 GiB |
| NVFP4 | on | MTP K=4 | 24.85 | 0.391 | 0.39 | 24.35 GiB |
| NVFP4 | on | MTP K=5 | 21.88 | 0.304 | 0.40 | 24.53 GiB |
| NVFP4 | on | n-gram K=3 | 17.35 | 0.150 | 0.84 | 24.16 GiB |

### Qwen3-8B (`qwen3-8b`, llama family — no MTP head)

S1 baseline (bf16): **MTP off (reference) 27.18 tok/s**, 17.16 GiB.

| format | graphs | row | tok/s | acceptance | syncs/tok | peak mem |
|---|---|---|---|---|---|---|
| bf16 | off | MTP off (reference) | 72.54 | n/a | 1.00 | 17.13 GiB |
| bf16 | off | MTP off (StepModel) | 72.47 | n/a | 1.00 | 17.09 GiB |
| bf16 | off | n-gram K=3 | 74.42 | 0.087 | 0.89 | 17.09 GiB |
| bf16 | on | MTP off (reference) | 71.65 | n/a | 1.00 | 17.13 GiB |
| bf16 | on | MTP off (StepModel) | 72.31 | n/a | 1.00 | 17.09 GiB |
| bf16 | on | n-gram K=3 | 75.63 | 0.087 | 0.89 | 17.09 GiB |
| Q8 | off | MTP off (reference) | 37.81 | n/a | 1.00 | 16.31 GiB |
| Q8 | off | MTP off (StepModel) | 43.77 | n/a | 1.00 | 16.31 GiB |
| Q8 | off | n-gram K=3 | 33.84 | 0.101 | 0.88 | 16.31 GiB |
| Q8 | on | MTP off (reference) | 78.07 | n/a | 1.00 | 16.56 GiB |
| Q8 | on | MTP off (StepModel) | 79.28 | n/a | 1.00 | 16.56 GiB |
| Q8 | on | n-gram K=3 | 47.38 | 0.101 | 0.88 | 16.56 GiB |
| NVFP4 | off | MTP off (reference) | 102.51 | n/a | 1.00 | 10.00 GiB |
| NVFP4 | off | MTP off (StepModel) | 116.03 | n/a | 1.00 | 10.00 GiB |
| NVFP4 | off | n-gram K=3 | 75.49 | 0.123 | 0.84 | 10.00 GiB |
| NVFP4 | on | MTP off (reference) | 87.01 | n/a | 1.00 | 10.00 GiB |
| NVFP4 | on | MTP off (StepModel) | 127.25 | n/a | 1.00 | 10.00 GiB |
| NVFP4 | on | n-gram K=3 | 93.45 | 0.123 | 0.84 | 10.00 GiB |

("sampled" rows — the temperature/top-p/seed-fixed verification rows — are also sealed in
`at2-campaign/INDEX.md` but omitted here for brevity; they carry no additional speedup
information over the greedy rows above.)

## AT3 — decode-step and speculative-engine parity

All parity suites pass on both families. Full logs: [`at3/parity-qwen38-27b.log`](at3/parity-qwen38-27b.log),
[`at3/parity-qwen3-8b.log`](at3/parity-qwen3-8b.log), [`at3/cuda-graphs-ac1-qwen38-27b.log`](at3/cuda-graphs-ac1-qwen38-27b.log),
[`at3/provider-off-qwen38-27b.json`](at3/provider-off-qwen38-27b.json).

* **Exact rows: 256/256 tokens identical, both families.**
  * qwen35 (27B): `decode_step_parity::ac1_step_model_greedy_fixture_is_token_identical_to_reference`
    — "256 tokens identical"; `static_kv_parity::ac1_static_kv_greedy_fixture_is_token_identical_to_attn_kv`
    — "256 tokens identical across reference / static / growing (gqa)".
  * llama (8B): `speculative_engine_parity::llama_family_qwen3_8b_exact_rows_and_teacher_forced_knife_edge_gate`
    — "gate holds: exact rows identical over 256 tokens".
* **Teacher-forced gate (the E1 rule — decided at the epic's feature-end review, finding D):** a
  verify-shaped forward (M = 2..6) may only disagree with the single-token argmax at a reference
  position whose top-2 logit gap is ≤ 1 bf16 ULP of the top logit. Holds on both families:
  27B — "largest reference top-2 gap at any flipped position: 1.0 bf16 ULP"; 8B — "30
  teacher-forced disagreements, all at ≤ 1 bf16 ULP knife-edges".
  Every disagreement position and its ULP gap is printed in the logs (not just asserted).
* **Free-running speculative rows are recorded, not gated** (compounding rounding makes a fixed
  ULP bound arbitrary): first-divergence position and reference top-2 gap logged for every
  MTP/n-gram K on both families, e.g. 8B free-running n-gram K=3: "first divergence at 51,
  reference top-2 gap 2.00 bf16 ULP (not a knife-edge; recorded, not gated)".
* **`static_kv_parity.rs` (qwen35/27B only): 4/4 tests pass**, including the pointer-stability
  test (`ac3_static_kv_device_pointers_are_stable_across_the_fixture_and_a_rollback`): "16
  attention buffers pinned across 256 steps and a rollback to 349; preallocated 792854528 bytes
  (23134208 KV + 769720320 checkpoint ring)". The other three: `ac1` (exact-row identity, above),
  `provider_off_path_is_token_identical_to_the_reference_loop` ("engine 256 tokens / 255 events,
  reference 256 tokens / 255 events, identical=true"), and
  `teacher_forced_static_vs_attn_kv_logit_parity_report` ("256 positions; argmax agrees at 255").

## AT4 — perplexity gate

Full data: [`at4/at4-qwen38-27b.json`](at4/at4-qwen38-27b.json),
[`at4/at4-qwen3-8b.json`](at4/at4-qwen3-8b.json). Both scored on the same 2048-token plain-text
slice (`text_sha256 f238f90bf67cd7…`), greedy, mean NLL over 2047 scored tokens.

| model | bf16 ppl | NVFP4 ppl | Δ | gate (≤ 2%) |
|---|---|---|---|---|
| Qwen3.8-27B | 7.836568 | 8.414024 | **+7.37%** | **fails** |
| Qwen3-8B | 26.037581 | 27.971121 | **+7.43%** | **fails** |

**NVFP4 is not default-eligible.** Both families miss the ≤2% perplexity-regression gate by a
wide margin (recorded per acceptance test 4 of the epic's acceptance criteria). NVFP4 stays an
explicit opt-in weight format; it is not proposed as a default for either family.

## Admission vs peak (load-admission honesty)

Full data: [`admission/`](admission/) (`load-residency-*.json`/`.log`, one per model × format,
run standalone on GPU 1 with `cuda_graphs: Some(false)`). Each JSON's `admission.device_required_bytes`
is the load-time bound `LlamaProvider::load_memory_estimate` admits with (the bound fixed by
inference #1044 — see [Superseded](#superseded)); "peak" is the campaign's own
`cuMemGetInfo`-based "device used @ last token" maximum for that model × format, across both
CUDA-graph settings, from `at2-campaign/INDEX.md` above (converted GiB → GB, ×1.073741824, to
match the admission unit).

| model | format | admission (device_required_bytes) | peak, graphs off | peak, graphs on | holds? |
|---|---|---|---|---|---|
| Qwen3-8B | bf16 | 20.48 GB | 18.39 GB | 18.39 GB | yes / yes |
| Qwen3-8B | Q8 | 27.86 GB | 17.52 GB | 17.78 GB | yes / yes |
| Qwen3-8B | NVFP4 | 24.73 GB | 10.74 GB | 10.74 GB | yes / yes |
| Qwen3.8-27B | bf16 | 69.45 GB | 58.02 GB | 57.92 GB | yes / yes |
| Qwen3.8-27B | Q8 | 97.10 GB | 43.22 GB | 43.51 GB | yes / yes |
| Qwen3.8-27B | NVFP4 | 84.09 GB | 26.28 GB | 26.35 GB | yes / yes |

**12/12 admission ≥ peak** (2 models × 3 formats × 2 CUDA-graph settings = 12 cells; every cell's
admitted bound covers its measured peak, with margin to spare in every case).

## Decisions recorded during the epic

* **E1 knife-edge rule** (feature-end review, finding D): exact (non-speculative) rows are gated
  token-identical; speculative rows are gated on the teacher-forced verify-shaped forward at ≤ 1
  bf16 ULP of the reference's top-2 logit gap; free-running speculative rows are recorded, not
  gated, because their compounding rounding makes any fixed ULP bound arbitrary. Applied
  identically to both the qwen35 and llama families in `speculative_engine_parity.rs`.
* **GQA reference:** the provider's off-path (no proposer, static KV) and the reference `Decode`
  loop (growing KV, `set_decode_path(DecodePath::Reference)`) both attend through un-expanded GQA
  and both decode one token per step (M = 1) — so the knife-edge exception does not apply there;
  the two paths are required to match exactly, which they do (`at3/provider-off-qwen38-27b.json`).
* **CUDA graphs — closed with findings, not a capture path:** every graphs-on cell in the AT2
  campaign and every AT3 graphs-on parity run reports `0 replayed / N eager, 0 captured`, fallback
  reason `positions_host_scalar`. No cell in this evidence bundle captured a CUDA graph; "CUDA
  graphs on" here means the runner is enabled and takes its (working, correctness-preserving)
  eager fallback on every step, not that graph replay was exercised. This is the state the epic's
  CUDA-graph work closed on.
  * **AT1 (round 2 of the feature-end review)** did separately close the graph **capture race**:
    the CUDA-graph switch's lock is now re-entrant on its own thread, so a unit test or a capturing
    test holding the device lock can't deadlock the switch. That fix is upstream of this evidence
    but does not change the "no captures were exercised in this campaign" fact above.
* **NVFP4 is lossy and stays opt-in:** AT4 perplexity regressions of +7.37% (27B) and +7.43% (8B)
  against a ≤2% gate (see [AT4](#at4--perplexity-gate)) — NVFP4 is not proposed as a default
  weight format for either family.

## Superseded

* **rc.0 campaign at `e9faeabdbf8ded296ec8083828e717cd951240ae` (tag `runtime-2026.09.1-rc.0`)** —
  superseded by the rc.1 measurement in this directory. The generated tokens were identical
  between rc.0 and rc.1 (same fixture, same greedy/sampled settings); rc.1 exists because
  inference #1044 fixed load-admission honesty (the E6 finding: quantizing and GGUF loads could
  peak above their admitted bound — see the admission table above, now 12/12 holds) and the epic
  re-measured decode-perf, parity and perplexity against the corrected runtime rather than mix
  pre-fix admission numbers with post-fix decode numbers. The rc.0 campaign, AT3 logs and AT4 JSON
  are **not copied into this evidence bundle**; they remain at `C:/evidence/sc-24128/at2-campaign`,
  `at3/`, `at4-*.json` (sealed, outside the repo) for anyone who needs to diff rc.0 vs rc.1
  directly.
* **Aborted campaign, `aborted-campaign-20260924T0434/`** — a campaign run started on rc.1 that
  overlapped a CI dispatch to the same self-hosted runner (the idle-box precondition was violated
  mid-run). It was discarded rather than trusted or repaired in place; the campaign captured in
  this directory is the clean re-run that followed once the box was confirmed idle end-to-end (see
  the idle-box procedure above). The aborted campaign is **not copied**; it remains at
  `C:/evidence/sc-24128/aborted-campaign-20260924T0434/` (sealed, outside the repo).

## Provenance note (absolute local paths in sealed files)

The sealed `run.json`, `.log` and admission `.json` files under this directory embed the absolute
build/checkout/output paths from the machine that took the measurements — e.g.
`D:\repos\inference\.claude\worktrees\at-head-rc1\...`, `C:\cargo-targets\at-head-rc1\...`,
`C:\evidence\sc-24128\...`, and `E:\huggingface\hub\models--...\snapshots\...`. These are build
and dataset provenance fields, not user-profile or home-directory paths (no `Users\<name>` path
was found anywhere under this directory — verified with a recursive search before copying). They
are left exactly as measured because rewriting any byte of a sealed run directory would invalidate
its `SEAL.json` sha256 and the campaign's own `run_seal_sha256` entries; `campaign-verify` (below)
checks those hashes against the files as committed.

## Verification

```
python scripts/release/decode_bench.py campaign-verify docs/migration/evidence/sc-24140/final/at2-campaign
```

prints `campaign docs/migration/evidence/sc-24140/final/at2-campaign: 12 cells, seals verified, complete`
and exits 0 against this copy.
