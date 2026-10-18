# sc-26322 independent foundation review

Verdict: **CHANGES REQUIRED — two P2 findings.** Confidence: high.

Reviewed inference HEAD `d32c781302e3712653743f66cd4b316ae0d726c5` against base `5e07e57b1a0bbe741fe67eb2723e888b7fc91fe1`, in `/Users/michael/.codex/worktrees/sc-26322-moss-native-inference`. All 14 changed files were reviewed. The worktree was clean at review start and after source inspection. No repository edits, compilation, accelerator activity, model execution, or weight payload loading were performed by this review. CPU-only scalar scheduler comparison, small-file hashes, and safetensors header inspection were performed.

The current implementation story and inherited epic requirements were read from `implementation/tracker-approved-implementation.json`, including the approved change from historical evaluation to native implementation. Missing production execution/registration in this foundation story is intentional and is not a finding. Backend execution belongs to sc-26323/sc-26324; SceneWorks integration belongs to sc-26325.

## Finding reference convention

Read `AGENTS.md` and `CONTRIBUTING.md`. `python3 scripts/check-review-findings.py --next` returned `F-274` at the reviewed HEAD. The two identifiers below are **proposed, not allocated**: this read-only review does not change the repository registry. When persisting this review in the remediation change, use a `CODE_REVIEW_*.md` document, register the range `274–275` in `docs/code-review-finding-allocations.tsv`, and recheck allocation against the live merge base per CONTRIBUTING. Durable remediation references should use `sc-26322` plus the qualified review context, not bare F identifiers.

## Findings

#### [F-274] [P2] Reproduce author FP32 linspace before snapping to the training grid

Location: `crates/contracts/gen-core/src/moss_image.rs:365–374` (particularly line 372).

The new grid interpolates each value in FP64 and casts it once, whereas the frozen author uses `torch.linspace` with its FP32 CPU grid construction. The difference can change the selected training index, rather than merely changing the final sigma by one ULP. A valid ordinary request with `steps=7`, `scheduler_shift=1`, and default training shift 1 selects index **167** at step index 5 here, versus author index **166**. Its sigma becomes approximately **0.168 instead of 0.167**, so both the transformer timestep and Euler interval differ from the author schedule. The existing golden only covers 50 steps/shift 5 and therefore misses this case.

Source authority: the frozen local `pipeline_moss_image.py`, SHA256 `1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31`, lines 272–287; snapshot `OpenMOSS-Team/MOSS-Image-SFT@fc5de0d56fc910cbca6f6625a70f5f582b26deb1`. The grid is built on CPU before the scheduler transfers timesteps to the execution device.

Executed CPU-only reproduction with local torch 2.9.1 and NumPy, mirroring the Rust arithmetic exactly for shift 1:

```python
import numpy as np
import torch

def author_grid(n):
    return torch.linspace(1, 1 / 1000, n, device="cpu").numpy()

def native_grid(n):
    lower = np.float32(1) / np.float32(1000)
    return np.array([
        1.0 + (float(lower) - 1.0) * i / (n - 1)
        for i in range(n)
    ], dtype=np.float32)

for name, grid in [("author", author_grid), ("native", native_grid)]:
    train, sigmas = grid(1000)[::-1], grid(7)
    hi = np.minimum(np.searchsorted(train, sigmas), 999)
    lo = np.maximum(hi - 1, 0)
    index = np.where(abs(sigmas - train[lo]) <= abs(train[hi] - sigmas), lo, hi)
    print(name, index.tolist(), train[index].tolist())
```

Observed:

```text
author indices: [999, 833, 666, 500, 333, 166, 0]
native indices: [999, 833, 666, 500, 333, 167, 0]
author sigma[5]: 0.16700001060962677
native sigma[5]: 0.1679999977350235
```

The sixth inference-grid input happens to be identical (`0.16750000417232513`); the differently rounded training-grid neighbors cause the wrong snap. Additional checks also found differing snap indices for other supported step/shift values. Implement the frozen author linspace arithmetic, including its endpoint handling, and add non-default fixtures that assert snapped indices, including this seven-step case. Do not loosen the existing tolerance: the error is a whole training-grid point.

Confidence: high (0.99). Impact: ordinary Base/SFT sampling fidelity for supported non-default controls.

#### [F-275] [P2] Carry the normal progress callback through the advanced generation contract

Location: `crates/contracts/gen-core/src/moss_image.rs:587–592`.

`MossImageAdvancedGenerator::generate_moss_advanced` accepts only a request and advanced inputs. There is no `on_progress` callback and `GenerationRequest` has no progress sink. A backend implementing this exact contract cannot stream normal step/decode progress for supplied embeddings, supplied noise, or latent output. The ordinary `Generator::generate` requires `&mut dyn FnMut(Progress)` at `generator.rs:146–151`; other additive generation entry points preserve that argument. Cancellation is correctly available through `request.cancel`, but progress cannot be recovered from it. The new documentation promises that the advanced native route preserves cancellation/progress through normal runtime contracts, and the inherited epic E2 requires that behavior.

Reproduction is the public signature itself: a caller holding `&dyn MossImageAdvancedGenerator` has no parameter through which to provide its job's progress callback; neither `MossImageAdvancedInputs` nor `GenerationRequest` supplies an equivalent callback. Calling ordinary `generate` instead cannot carry the advanced inputs.

Add the normal progress callback to this trait now, while neither backend implements it, and carry it into both ports. A small implementation/caller fixture can check that the callback is forwarded with a real `Progress` variant without model execution. This is a missing contract parameter, not a request to implement provider execution early.

Confidence: high (0.98). Impact: the declared advanced routes cannot satisfy existing progress/lifecycle behavior without changing or bypassing the foundation contract.

## Coverage and positive observations

- Contracts/module export/generic request integration: inspected typed variant and exact identities, default-OFF acceleration, ordinary and DMD defaults, incompatible sampler/scheduler/CFG/negative controls, bounds on steps/context/dimensions/count, finite checks, layer-index validation, and shared refusal of MOSS controls on unrelated provider IDs. No extra actionable finding identified.
- Advanced input validation: inspected positive text versus embedding exclusivity, positive/negative batch agreement, negative embedding and negative-string-batch exclusivity, DMD negative-input refusal, mask dimensions/binary entries/nonempty rows, finite embedding/noise values, checked count/shape multiplication, prompt-batch expansion, and latent shape/output representation. Generic prompt-versus-prompt_batch conflict and provider capacity limits remain enforced by the existing capability floor; downstream providers must retain it and apply returned expanded batch sizes when admitting advanced embeddings.
- Sampling: checked actual-step SLG interval semantics, CFG=1 SLG special case, CFG=0 behavior, FP64 time features, DMD prefix length, fixed-noise recurrence and nonsymmetric matrix orientation against the frozen author source/README and the existing reference harness. F-274 is the identified numerical-contract defect.
- DMD asset validation: inspected exact key-set coverage, rank/alpha, all six projections across 42 blocks, replacement biases, finite FP32 checks, parent/package identity, and separate immutable mode/component guidance. Parsed only safetensors headers from the actual frozen package: all **756 LoRA shapes/dtypes** exactly match the generated contract map; all **six replacement tensors** and **two dataspace tensors** match. No payload or model evaluation was needed.
- Fixtures: all four checked-in transformer/scheduler/text-encoder/VAE configs are byte-identical to the frozen cached source. All six small-file hashes in `goldens.json` match the frozen cached files, and the pipeline hash matches the published constant. The existing scalar time/DMD fixtures are honestly described; the one author schedule golden is too narrow to expose F-274.
- Tokenizer: compared the single-user renderer against the frozen Jinja path, including trimming, role delimiters, generation prefix, and open thinking prefix. JSON/tag/multilingual content is preserved inside the trimmed user message. No additional actionable finding identified.
- Both Qwen3.5 seams: inspected all changed loader/head/forward paths and regression tests. `from_weights_headless` skips the LM projection; `forward_hidden` applies the same final norm as the normal path and preserves existing cache ownership. Verified actual frozen text-encoder index has `language_model.*` keys and no requirement for a language-model head. MLX suppresses unrelated MTP loading for the headless encoder. Existing dense/Prism/tied-head constructors preserve their original head-loading branches. No actionable compatibility defect identified in changed code.
- Both FLUX.2 VAE seams: inspected independent decoder widths, checkpoint shape checks, standard defaults, optional encoder handling, attention dimensions, and decode-only behavior. Actual MOSS VAE header confirms decoder input width 384, output width 96, FP32 attention tensors `[384,384]`, and standard 32/128 raw/packed latent channels. Candle construction consumes explicit decoder widths; MLX construction derives channel widths from validated tensors and retains the fixed four stages. Existing standard callers keep their original architecture and behavior. No actionable compatibility defect identified in changed code.
- `git diff --check` passed for the exact base/head comparison. The root-reported compile/clippy/unit results were not independently rerun. The ignored MLX numerical decode fixture remains an explicit unverified accelerator acceptance item; this review did not run it or treat the CPU-stream escape/timeout as a pass. It is not a newly introduced defect by itself.
- No placeholder executor, provider registration, production Python dependency, or unrelated edits were found in this foundation diff.

## Limits

This is the one independent story review and reports both findings together for the authorized fix batch. It does not claim native real-weight parity, GPU memory/lifecycle acceptance, provider readiness, or completed terminal smoke tests. Those remain in their scoped backend/terminal stories. No CodeGraph tools were available in this agent session and no root CODEGRAPH.md existed; discovery used the exact 14-file diff and targeted source reads. Historical memory was consulted only for orientation and superseded by the current approved implementation requirements.

## Remediation batch (sc-26322)

The repository allocated this review's identifiers274–275 in the accompanying append-only registry row, after checking that274 was still next. This is the sole authorized fix batch; no second independent review is claimed.

- sc-26322 (F-274, this review): native linspace now uses the author FP32 step and two endpoint anchors. New CPU torch2.9.1 reference fixtures cover seven-step shift1 and additional nondefault inference/training shifts, with explicit lower-tie/index and endpoint regressions. The default schedule tolerance is not relaxed.
- sc-26322 (F-275, this review): the advanced trait now requires the same progress callback as ordinary generation. A dynamic-trait fake verifies actual step forwarding and typed cancellation without model execution.
- The same batch adds the agreed canonical strict JSON settings/resolved-controls and actual-execution receipt contract; it does not register an executor or claim native model acceptance.

Self-verification of the remediation batch: all18 focused MOSS tests passed; complete gen-core CPU package suite passed1144 unit,49 integration and3 doc tests (1196 total), with2 existing real-weight tests ignored. Strict all-target Clippy with warnings denied, formatting/diff checks,107-member workspace gate,281-baseline clock ratchet and review-allocation checks passed. Component compile checks and current-head PR CI are owned by the canonical story coordinator; no new independent review or accelerator/model acceptance is asserted here.
