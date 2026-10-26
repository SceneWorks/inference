# sc-26584 inference integration-merge review

Reviewed revision: `da54be262b56e809e90e65b351205a6c1d3371ea`.

Scope: semantic conflict resolutions joining MOSS adapter training with main's
MOSS edit support, provider/catalog unions and shared test seams. Existing epic
and main reviews remain authoritative outside that merge delta. No native
execution, build or CI dispatch was performed during this review/repair.

#### [F-284] P1: MLX edit projections skip installed custom adapters

At the reviewed revision, `transformer.rs:150-163` returns directly from the edit
adapter branch before the custom LoRA and structured LoKR loops at lines 174-205.
`model.rs:466-475` installs custom adapters before installing edit weights, and
`install_edit` marks all six block projections as edit adapters. Consequently
valid custom adapters are reported installed but have no effect during editing.

Repair: retain the existing edit residual arithmetic, then continue through the
custom residual loops, matching Candle's composition. The existing ignored
`transformer::edit_fixture_tests::frozen_edit_oracle_on_metal` now additionally
trains, exports, pins and reloads LoRA and LoKR before edit installation, checks
zero and signed nonzero strengths, and checks cached/uncached edit equivalence.
The unchanged exact-test driver in `scripts/ci/moss_edit_metal.py` reaches this
regression. Ordinary CI does not execute the learned graph.

#### [F-285] P2: failed custom-pin authentication retains a stale edit receipt

At the reviewed revision, `model.rs:383` and `model.rs:776-781` authenticate custom
pins before clearing the previous edit receipt. Removing/changing a custom file
after successful editing returns early while leaving the old edit execution and
application reports visible. This violates main's failed-entry receipt cleanup.

Repair: clear the edit receipt before the fallible attempt-start pin check on
both ordinary and edit entry paths. A pure regression removes a pinned source
and checks that both receipts and the owned renderer are cleared.

#### [F-286] P2: edit application reports omit the custom adapter stack

At the reviewed revision, MLX `model.rs:853-859` and Candle `lib.rs:570-576`
return only the edit adapter when an edit receipt exists. The base execution
receipt already carries the custom adapter reports, so this loses the consumer's
per-file application evidence even on Candle where residual math composes.

Repair: preserve the base receipt's custom reports and append the edit report in
both providers. The existing Candle edit receipt test checks both entries.

## Validation state

Source formatting and whitespace checks pass. Native compilation, the pure Rust
regressions, and the affected dedicated Metal edit oracle remain pending on the
assembled integration head. These repairs neither satisfy nor change the sealed
eight-cell real-weight training campaign, which remains terminal work.
