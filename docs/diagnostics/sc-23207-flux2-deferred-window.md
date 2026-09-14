# Flux2 deferred load and Resident execution (sc-23207)

The installed-app job `job_ab984800b1377e6f9df6dfed536092f2` selected
`flux2_klein_9b_true_v2`, 1024x1024, count 2. At 2026-09-12T14:46:36Z it loaded
Klein under Sequential/deferred policy, then admitted Resident at 54.49 GiB against
126 GiB. Generation failed with `a deferred transformer requires an explicit block
window` before producing an image. The user prompt remains in local evidence only.

The provider admitted Resident on a stream-capable load, but heavy loading attached
and finalized the block stream solely from the load specification. Finalization
removed the full block vectors even when the request carried no streaming window.
The existing request-scoped residency interface already carries the selected
streamable flag; Flux2 passed false into it and discarded the callback argument.

Thread the validated request window through that interface and use it to decide
whether to attach/finalize the block stream. A shallow request constructs complete
blocks once for its render lifecycle. A bounded request keeps the existing sealed
windowed path. Continue verifying the exact artifact inventory for every deferred
heavy load, including shallow execution. Explicit request staging overrides the
load default for all Klein variants; an absent memory selection preserves defaults.
Reject unsupported load/window/staging combinations before loading components.

Regression coverage includes callback flags, warm/staged/streamed transitions,
cancellation, and invalid explicit windows. The installed True-V2 generation check
exercises the previously missing load-to-forward path for both preserved defaults
and explicit Resident requests; it is an ignored hardware check, separate from
metadata-only admission checks.
