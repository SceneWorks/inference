# Training comparisons: measured choices and final 414 budget

The completed supplement contains **17 source-accepted experiments: 15 qualified Reference update-timing wins and two Legacy fallbacks** (SD3.5 Medium and FLUX3). There are 28 retained observations and 414 conservative updates, including 64 charged to failed attempts. Complete experiments account for 350 starts/350 commits; all attempts record 394 starts/390 commits. Those counters are distinct.

All three final C75 workflows and original source verifiers passed: Flux 12, MOSS 12 and depth 4 completed updates. No further campaign budget remains. A qualified timing win means a positive reduction plus the original scalar numerical/state/path checks; it is not statistical superiority, release throughput or convergence.

## Evidence and budget amendment

The original campaign cap 404 and family caps Flux 24 / MOSS 24 / depth 12 are historical and unchanged. The user explicitly authorized the final 28 updates and effective cap 414:

> Go ahead with the remaining three runs and 28 updates. Raise the total cap to 414, allow the family-cap exceptions, and finish the epic.

The effective family caps are Flux 34, MOSS 28 and depth 14. The original report still flags global404 and those three original family caps as exceeded; the separately authenticated effective-budget view accepts414. All failed charges/flags remain. There is no transfer, automatic retry, unspent reservation or authority for further training. The appended [sourcebook amendment](../../scripts/ci/training-comparison-cases.json) preserves every original entry/proposal and records the actual capture/report hashes.

| Final evidence record | SHA256 |
|---|---|
| Original multiproducer report (28 observations) | `3db6eeb22a10cc41c3999b32af2984351d99fbc50f9188962171f08271f2f30f` |
| Separate authorized effective-budget view | `a19f7ebc5ecd0037dc7b0f806ed868535152da3eda99afcfc5b410c89a214026` |
| Actual user authorization capture record | `d344a27ed6724e212a929b9574583190ca8c0096fa8726f44f9ad0b9863f2f70` |

The report is `comparison-results-final-three-414-iris-depth-41297158841-1/results.json`; the effective view is `final-three-414-effective-budget-FINAL.json` in the retained epic evidence. Downloadable workflow artifacts below retain original receipts, source-bound state joins, timings and logs. Measurement producers remain [ce37](https://github.com/SceneWorks/inference/commit/ce37f3d054136b0298cb38e07185e095200f6b39), [H2](https://github.com/SceneWorks/inference/commit/2e45c46a931baf2e30803b41dbac1c4b1e2f6569) and [C75](https://github.com/SceneWorks/inference/commit/c75fe5fa74035a8c4a8654ffa55dfd95a2dac36e); they are not restamped to a final production commit.

## Measured adoption choices

All listed attention choices are **Reference**. Configurations use `train_dtype=bf16`, batch 1, accumulation 1 and seed 7; native F32 arithmetic islands and frozen source dtypes remain as recorded. Displayed means are seconds per timed physical optimizer update, rounded to 6 decimals; reductions are rounded to 4 decimals. Artifacts retain exact floats/loss arrays. Each row is one complete matched experiment; failed prefixes or arms from different runs are never combined.

| Family / measured workload | Measured profile choice | Legacy -> chosen s/update | Reduction | Original source / run / artifact |
|---|---|---|---|---|
| Lens / 512px LoRA, r8 | Current | 1.500913 -> 1.431379 | 4.6328% | ce37 [run41238141467](https://github.com/SceneWorks/inference/actions/runs/41238141467) / [artifact](https://github.com/SceneWorks/inference/actions/runs/41238141467/artifacts/13057068036) |
| Z-Image Turbo / 512px LoRA, r8 | WithoutE3 | 1.110722 -> 1.061617 | 4.4210% | ce37 [run41238144458](https://github.com/SceneWorks/inference/actions/runs/41238144458) / [artifact](https://github.com/SceneWorks/inference/actions/runs/41238144458/artifacts/13059132977) |
| Qwen Image 2.1 / 512px LoRA, r4 | Current | 1.514728 -> 1.462565 | 3.4437% | ce37 [run41238146895](https://github.com/SceneWorks/inference/actions/runs/41238146895) / [artifact](https://github.com/SceneWorks/inference/actions/runs/41238146895/artifacts/13057854378) |
| Wan 14B dual / 512px LoRA, r8 | WithoutE3 | 2.604278 -> 2.550423 | 2.0679% | ce37 [run41238149358](https://github.com/SceneWorks/inference/actions/runs/41238149358) / [artifact](https://github.com/SceneWorks/inference/actions/runs/41238149358/artifacts/13061010060) |
| Wan 5B single / 512px LoRA, r2 | WithoutE1 | 0.766640 -> 0.739887 | 3.4896% | ce37 [run41238152270](https://github.com/SceneWorks/inference/actions/runs/41238152270) / [artifact](https://github.com/SceneWorks/inference/actions/runs/41238152270/artifacts/13057059304) |
| LTX 2.3 / 256px LoRA, r4 | Current | 1.193124 -> 1.082997 | 9.2301% | ce37 [run41238154786](https://github.com/SceneWorks/inference/actions/runs/41238154786) / [artifact](https://github.com/SceneWorks/inference/actions/runs/41238154786/artifacts/13058499293) |
| LTX 2.5 / 256px LoRA, r4,9 frames+audio | WithoutE2 | 3.358707 -> 2.946835 | 12.2628% | ce37 [run41238157269](https://github.com/SceneWorks/inference/actions/runs/41238157269) / [artifact](https://github.com/SceneWorks/inference/actions/runs/41238157269/artifacts/13057941314) |
| SDXL / 512px LoRA, r8 | Current | 1.770335 -> 1.645246 | 7.0658% | ce37 [run41238159851](https://github.com/SceneWorks/inference/actions/runs/41238159851) / [artifact](https://github.com/SceneWorks/inference/actions/runs/41238159851/artifacts/13059644464) |
| Kolors / 512px LoRA, r2 | WithoutE3 | 1.662907 -> 1.555027 | 6.4874% | ce37 [run41238162116](https://github.com/SceneWorks/inference/actions/runs/41238162116) / [artifact](https://github.com/SceneWorks/inference/actions/runs/41238162116/artifacts/13058219397) |
| SD3.5 Medium / 512px LoRA, r2 | Legacy | 0.924941 -> 0.924941 | no qualified gain | ce37 [run41238164493](https://github.com/SceneWorks/inference/actions/runs/41238164493) / [artifact](https://github.com/SceneWorks/inference/actions/runs/41238164493/artifacts/13059165933) |
| Anima Base / 512px LoRA, r2 | WithoutE1 | 1.776166 -> 1.352614 | 23.8465% | H2 [run41270377984](https://github.com/SceneWorks/inference/actions/runs/41270377984) / [artifact](https://github.com/SceneWorks/inference/actions/runs/41270377984/artifacts/13070023415) |
| MOSS Base / 256px LoRA, r2 | WithoutE1 | 0.843595 -> 0.811005 | 3.8632% | C75 [run41297143598](https://github.com/SceneWorks/inference/actions/runs/41297143598) / [artifact](https://github.com/SceneWorks/inference/actions/runs/41297143598/artifacts/13080800742) |
| Mage Adapter / 512px LoRA Adapter, r2 | WithoutE3 | 0.351938 -> 0.328532 | 6.6507% | ce37 [run41238171879](https://github.com/SceneWorks/inference/actions/runs/41238171879) / [artifact](https://github.com/SceneWorks/inference/actions/runs/41238171879/artifacts/13062680159) |
| FLUX3 Klein Base / 64px LoRA PairedEdit, r2 | Legacy | 0.711037 -> 0.711037 | no qualified gain | C75 [run41297127692](https://github.com/SceneWorks/inference/actions/runs/41297127692) / [artifact](https://github.com/SceneWorks/inference/actions/runs/41297127692/artifacts/13080377076) |
| Iris generation / 512px LoRA, r8 | WithoutE3 | 2.323233 -> 2.026090 | 12.7901% | ce37 [run41238177523](https://github.com/SceneWorks/inference/actions/runs/41238177523) / [artifact](https://github.com/SceneWorks/inference/actions/runs/41238177523/artifacts/13060173664) |
| Iris depth / 512px Full | Current | 4.079068 -> 3.973020 | 2.5998% | C75 [run41297158841](https://github.com/SceneWorks/inference/actions/runs/41297158841) / [artifact](https://github.com/SceneWorks/inference/actions/runs/41297158841/artifacts/13080489577) |
| Iris restoration / 512px LoRA G+D, r8 | WithoutE3 | 25.994761 -> 25.426797 | 2.1849% | H2 [run41269138019](https://github.com/SceneWorks/inference/actions/runs/41269138019) / [artifact](https://github.com/SceneWorks/inference/actions/runs/41269138019/artifacts/13069887771) |

Profile names map to the new groups: **E1** combined LoRA arithmetic and normalization, **E2** frozen conditioning/geometry preparation, **E3** frozen projection packing. `WithoutE1/E2/E3` removes that one group from the shared configuration; `Legacy` disables new groups while retaining common correctness/safety repairs and preexisting family optimizations. Preexisting Iris factor scaling/fold3 paths remain common. No new configuration combination was synthesized from separate wins.

The measured choices are the source targets for the final policy batch; the [finite selector](../../crates/media/candle-gen/candle-gen/src/train/policy.rs) and native mode/resume scopes bind production adoption separately. MOSS is the measured 256px BaseLoRA recipe, Flux is 64px BaseLoRA PairedEdit, depth is mandatory Full. Mage Full, Wan I2V14B, SD3 Large, other modes/devices/resolutions and long jobs are not comparative measurements here.

## Strategy findings

- MOSS and Anima choose `WithoutE1`; the actual direct pairs, rather than an assumed benefit for every new operation, supply their positive Reference result. MOSS is 0.843595 -> 0.811005s/update (+3.8632%); its scalar check passes.
- Several families choose `WithoutE3`; LTX 2.5 chooses `WithoutE2`. These are conditional/post-hoc choices among measured arms, not independent additive speed contributions. Primary pair failures and slowdowns remain in the raw receipts.
- Depth retains `Current` with E2 ON: 4.079068 -> 3.973020s/update (+2.5998%). Its E1/E3/reference-owner controls are common or N/A; streaming is source eligible but was not remeasured in its two-arm recovery.
- SD3 and Flux retain `Legacy`: all tested Reference alternatives fail their direct numerical check against Legacy. SD3 Current has a timing-only improvement; Flux Reference arms are slower. Neither is a qualified improvement.
- Eligible E4 streaming shows some conditional wins and other numerical failures/slowdowns. This table deliberately selects qualified Reference arms; it does not claim Reference is fastest across every tested arm or promote streaming to a production default.

## Short-run setup tradeoffs

**Eight of the 15 update-timing wins have a slower observed preparation+counted-update subtotal.** Positive candidate-minus-Legacy below is a loss. This subtotal includes native preparation/cache/load, proof initialization diagnostics and warm/timed updates; it excludes compilation, admission and export. It is not whole-job or production end-to-end latency.

| Family | Legacy -> chosen subtotal seconds | Additional time (loss) |
|---|---|---|
| Wan 5B single | 35.479569 -> 39.513387 | +4.033818s |
| SDXL | 13.756858 -> 25.850160 | +12.093302s |
| Anima Base | 8.712067 -> 12.293705 | +3.581637s |
| MOSS Base | 80.481537 -> 82.825240 | +2.343703s |
| Mage Adapter | 12.909680 -> 19.033251 | +6.123571s |
| Iris generation | 29.310484 -> 30.260983 | +0.950499s |
| Iris depth | 310.727405 -> 382.508347 | +71.780942s |
| Iris restoration | 183.302170 -> 186.271067 | +2.968896s |

Restoration shares only an immutable, source/run-owned frozen checkpoint decode store across fresh children. First-build/reuse costs cannot be assigned to E2 or called a measured strategy gain. Mutable parameters, optimizer moments, EMA/critic/SN state and data/RNG construction remain fresh per stage.

## Measurement limits and historical functional coverage

All complete compared executables report Cargo test **opt_level0**, with debug assertions. They ran serial stages on shared GPUs, not an isolated device. Small samples, stage order/interference and post-hoc best-arm selection limit interpretation. Most ordinary arms have one warmup plus three timed updates; Anima/MOSS/Flux/depth have one warmup plus **one timed update**. Wan dual mixes two owners; restoration has one warm G+D cycle plus two timed cycles/four timed physical updates. Both optimizers count; mixed-owner means are not identical-work microbenchmarks.

The original numerical criterion is matched scalar loss trajectories at tolerance0.001 relative to max(1,loss), plus finite reached gradients and source-owned initialization/optimizer/owner/path gates. It is not full-gradient parity, convergence, quality, whole-model checkpoint-byte integrity, a named weight-revision attestation or release certification. Necessary correctness/memory repairs were common to both arms; Legacy is an optimization control, not an identical historical binary.

The separate **19 historical functional implementation rows** cover bounded save/reload/apply and viewed samples on their original configurations: 18 accepted source cases plus the separately qualified original MOSS product/body. Original MOSS product run[41195927630](https://github.com/SceneWorks/SceneWorks/actions/runs/41195927630) retains whole-workflow/postExecutionVerified=false; its body, 4 updates and offline lifecycle/source qualification are separate. The new engine MOSS timing experiment is not a replay or replacement consumer-product/sample campaign. Preserve preview cadence 500, caller prompts/settings and original serialization/source identities.

Krea Raw and Control are outside this 17-family supplement. Raw remains a separate 32-spent NO-GO: its matched[997run41214639210](https://github.com/SceneWorks/inference/actions/runs/41214639210) measured 8.805355s Reference versus 27.590252s candidate (slowdown), with original fields/workflows retained. The user rule is any positive qualified speedup wins/any slowdown loses; the historical 25% threshold is inactive. The [older Krea recovery note](sc-26850-krea-recovery-no-go.md) is an earlier 20-budget/cross-run snapshot, not the current 32-budget paired result. No additional Krea work is authorized.

The old 19-row functional source bridge and the selected ce37/H2/C75 math/API/export/private-state/preview joins are retained separately. This summary does not attest whole-binary equivalence or the final protected engine-main/consumer-pin pair; those delivery records must identify their actual final revisions. Source/CI success alone does not create runtime or budget credit.
