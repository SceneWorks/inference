# MOSS edit package revision correction

The R4 installer failed before rendering: Hugging Face returned 404 RevisionNotFound for
`chinoll/Moss-Image-Edit@e19b6ab0fb8fd010cb6f1c7f5b9bfe152252df11`.
The accessible immutable revision is
`3be9c40037e1b89b17aa85ff6f8ee805872aa558`. The captured public metadata is in
`crates/contracts/gen-core/tests/fixtures/moss_image/tiny_edit/upstream_identity.json`.
The native execution identity now names that revision; consumer manifests and fresh package
inventories must use it in lockstep after the native-main repair merges.

The 297382216-byte adapter still has SHA256
`17a48b8c605dcffdedb92a2b887d18cf316505be90802717669fb6556098dcdf`.
The 53712-byte config still has SHA256
`b6b471cf7bc77b0874af03be618f8a06b6997d92b35de12c47a06b292ade2669` and is byte-identical
to the existing config fixture. No weight/config fixture, tensor calculation, model control,
reference/cache behavior, validation policy, or serialized field changes.

Historical numerical receipts retain their original source and package identities. They were
not rerun or relabeled by this correction. Current execution-report validation continues to
require the current exact revision, so old-revision reports remain historical evidence rather
than valid new execution receipts. The source mapping is a single EDIT_REVISION replacement;
all other bytes of the production contract module are preserved. A new native build/source
identity and consumer pin are still required even though the model-content digests and math
are unchanged. Refresh only the affected evidence, retaining the original measurements.
