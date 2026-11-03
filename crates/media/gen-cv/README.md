# Native CV engines

This crate retains its existing AGPL-3.0-or-later licensing and explicit runtime `cv` opt-in; see [NOTICE](NOTICE).

YOLO production detection returns its final output without retaining parity-only block references. The `parity-captures` feature exposes all five existing `YoloForward` block fields for the unchanged external reference-oracle test. The macOS runtime forwards it as `cv-parity-captures`; consumers can enable that on a development dependency so shipping builds keep the normal feature surface. Unit-test builds also retain the captures.

The block4 and block10 local values remain in every forward because the neck concatenations consume them. The feature changes only returned references to intermediate arrays; model operations, ordering, weights, output decode and detection behavior are unchanged. Existing learned proof bodies, fixtures and manual selectors remain unchanged and are never executed in ordinary CI.

This is the owning-source portion of SceneWorks SW-6773 (SceneWorks 2026-10-27 review F-136), after CV engines moved from worker source into this crate. No model-runtime pass or new measurement is claimed for this cfg change.
