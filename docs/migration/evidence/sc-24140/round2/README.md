# sc-24140 — feature-end review, round 2: evidence

Epic sc-24128, feature-end fix story, round 2. Host: Windows 11, CUDA lane on **GPU 1** only
(`CUDA_VISIBLE_DEVICES=1`, RTX PRO 6000 Blackwell Max-Q, sm_120), CUDA 12.9, MSVC 14.44,
`CUDA_COMPUTE_CAP=120`. No real-weight runs, and the AT2 campaign was **not** run.

## The five findings

1. **AT2 / E8 — a campaign without its S1 baseline.** Every model in a campaign now needs a bf16
   `baseline` run of its own. If one is missing, the index lists `<model> / S1 baseline (bf16)` as
   missing. In run mode the campaign refuses before any run starts, unless `--allow-partial` is
   given. The index records `allow_partial`. INDEX.md shows a partial banner and a
   `model | S1 baseline` table.
   * `comparison_basis` leaves the label empty only when the basis is a `baseline` run. When the
     table's first run is used as a fallback, the rows read `vs <run> ref, no S1 baseline`.
   * `campaign-verify` recomputes the missing list from the cells and the loaded baselines, and
     refuses an index that disagrees. It prints every missing cell. It exits non-zero unless the
     index was sealed with `allow_partial: true`; an older index with no flag also fails.
   * The committed `campaign-smoke*` indexes still verify as complete.
2. **E5 — the gate passed packed MLX-affine snapshots.** `nvfp4_model_gate` now refuses a
   snapshot that has a `quantization` block (top level or `text_config`) and names a
   `<stem>.weight` + `<stem>.scales` triple.
   * The tensor names come from `model.safetensors.index.json`, or else from the shard headers.
     No tensor data is read.
   * A prepared Q4/Q8 snapshot (the block over dense weights) is still served.
   * The probe and the load refuse with the same text, before any weight is read.
   * The loader's own refusal at `llama.rs:95` is one sentence again: its line continuation had
     lost its `\`.
3. **E2 — the refusal wording.** LLaVA, StarVector-1B and StarVector-8B now say
   `NVFP4 projections are not served for <model>`. None of them says "qwen3_5 family only".
4. **E8 / AT2 — the bench's run identity.**
   * **Probed device.** Both families record the device the binary probed from the driver:
     `device`, `device_name`, and `compute_capability` (for example `12.0`). This code sits
     outside the head-only block, so a baseline binary records it too. The label must have the
     form `<GPU> / sm_<NN>`. `run` refuses a document unless the device is `cuda`, the compute
     capability matches `sm_<NN>`, and the probed name contains `<GPU>`.
   * **Build provenance.** A new `candle-llm/build.rs` embeds the checkout's `HEAD` and a dirty
     flag, but only when the build runs with `CANDLE_LLM_BUILD_PROVENANCE=1`. Without that
     variable it sets both values empty, runs no `git`, and never triggers a rebuild. A head run
     must name `--runtime-sha` and a clean tree. A baseline binary embeds nothing, because its
     pre-epic commit has no build script. If a baseline binary does embed a SHA, it must be the
     runtime SHA.
   * **Comparison key.** `prompt_token_ids` is recorded, and its sha256 and the sampling seed are
     now part of the table's comparison key.
   * **Collected runs.** A campaign re-checks the probe and the provenance of every collected
     head run and every baseline.
5. **AT1 — the capture race.** The CUDA-graph switch's lock is now re-entrant on its own thread.
   This lock is the one CUDA test lock.
   * A unit test that opens a CUDA device holds the lock until its thread exits. This covers
     `select_device`, `backend_capabilities` (the lock is taken before its one-time probe), and
     `new_cuda_for_test`, which now replaces every direct `Device::new_cuda(0)` in the test code.
   * Every test that captures already holds the lock through its graph guard.
   * A source-scan test fails if a direct CUDA device constructor appears anywhere in `src/`
     outside `device.rs`.

## The 20× CUDA lib loop (finding 5)

`candle-llm` lib tests (`--features cuda`), run with `<exe> --test-threads=8` 20 times on GPU 1:

| binary | runs | failures |
|---|---|---|
| this branch ([`cuda-lib-loop-fixed.txt`](cuda-lib-loop-fixed.txt)) | 20 | **0** (402 passed each) |
| control: feature head `022246245`, no lock ([`cuda-lib-loop-control.txt`](cuda-lib-loop-control.txt)) | 20 | **9** |

Every control failure is `poc_contiguous_ops_replay_bit_exact_and_index_select_is_refused`, which
panicked with `capture abandoned: capture_invalidated` at `graph.rs:2093`. Serializing the CUDA
tests raises the suite's time from about 1.9 s to about 6.5 s.

## Mutations

[`mutations.log`](mutations.log) lists one mutation for each added or changed assertion. Each
mutation was applied alone, the named test was run, and the source was restored and touched.

* Python: P1–P34.
* Rust on the CPU: R1–R12.
* Rust on CUDA, GPU 1: R7, R10, R11.

All 59 mutations turned their tests **RED**. The provenance test was also run **GREEN** on a build with
`CANDLE_LLM_BUILD_PROVENANCE=1`, which embedded this checkout's `HEAD`, before R12b was run
against it.
