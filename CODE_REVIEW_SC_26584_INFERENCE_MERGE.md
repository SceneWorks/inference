# sc-26584 inference integration-merge review

Reviewed revision: `da54be262b56e809e90e65b351205a6c1d3371ea`.

Scope: semantic conflict resolutions joining MOSS adapter training with main's
MOSS edit support, provider/catalog unions and shared test seams. Existing epic
and main reviews remain authoritative outside that merge delta. The initial
review was source-only; subsequent focused native validation is recorded below.
No learned execution or CI dispatch was performed locally during repair.

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

#### [F-286] P1: custom completion reports violate the sealed source receipt contract

Correction after native verification: the original review incorrectly claimed
that custom reports belonged in `edit.base.data().adapter_reports`. They do not.
`gen-core/src/moss_image.rs:1398-1409` reserves that field for exactly one published
unmerged DMD application, and requires it empty for ordinary/full-FP32 execution.
The native Candle test at `d320acdf39ddfaf41ffc29ac4ec3f09a3c89cdd8` rejected the
first repair's invalid fixture with `ordinary/full-FP32 receipt must not claim an
unmerged adapter installation`. That failed repair and its assumed storage model
are withdrawn; no validator restriction was relaxed.

The source defect is broader than the original report omission: both generators
fed user-trained adapter reports into the sealed source receipt, causing real
custom ordinary/edit completion to fail validation. The synthetic graph fixtures
returned before sealed receipt construction and did not expose that defect.

Repair: keep completed provider application reports separately in lifecycle state.
Only the built-in unmerged DMD report enters the sealed source receipt. Completed
provider reports contain custom applications, the unmerged DMD application when
present, and the edit application when present. Full-FP32 DMD remains folded and
does not claim an unmerged application. Publish completion reports after successful
validation and clear them on every new entry or failure, including pin failures.
The pure Candle regression uses a valid ordinary/edit receipt, verifies that custom
injection is still rejected by the real validator, checks independent custom/edit
reports, DMD/custom composition, and failed-entry cleanup. The ignored learned
Metal helper uses native adapter-loader reports directly, not sealed receipts.

## Validation state

After correcting F-286, strict combined all-targets/test-support Clippy passed for
both MOSS providers. The exact Candle edit receipt/composition test passed through
the real sealed receipt validator, and the exact MLX pin-failure lifecycle test
passed with completed application reports included in cleanup. Both native test
runs used the existing target directory, four build jobs, disabled incremental
compilation, repository Rust 1.96.0, default debug profiles and strip-none including
build overrides. Formatting and whitespace checks pass. The affected dedicated
Metal edit oracle remains pending. These repairs neither satisfy nor change the
sealed eight-cell real-weight training campaign, which remains terminal work.
