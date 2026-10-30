# gen-core-moss-image

> Package: `sceneworks-gen-core-moss-image` · library: `gen_core_moss_image`

The tensor-free **MOSS-Image** helpers only the MOSS providers need: the sampling schedule,
guidance and DMD math, package/config validation, the edit adapter's reference sizing and request
mapping, and the training admission/recovery contract. The request options and execution receipts a
`GenerationRequest` and the `Generator` MOSS entry points carry stay in `gen_core::moss_image` /
`gen_core::moss_image_edit` and are re-exported from the matching modules here.

Split out of [`gen-core`](../gen-core/README.md) in sc-27316. It depends on gen-core (gen-core
never depends on it) and, like gen-core, on no tensor library. Its tests read the shared MOSS
fixtures in `gen-core/tests/fixtures/moss_image`.

## License

Apache-2.0.
