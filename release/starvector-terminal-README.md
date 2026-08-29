# StarVector terminal campaign contract

`starvector-terminal-corpus-v1.json` and `starvector-terminal-receipt-v1.schema.json` are the
cross-repository contract for SC-22261's one permanent-pin campaign. They are deliberately small:
the corpus identifies upstream SVG-Bench rows and checksums rather than committing source SVGs or
raster binaries.

The `starvector-terminal` profile in this repository is a native-provider preflight only. It
materializes the immutable 1B/8B snapshots, records their inventories, and invokes all four
ignored MLX/Candle real-weight conformance hooks in serial MLX → Candle/CUDA order. Its artifacts
must be bound into the final receipt, but do not establish quality or catalog admission alone.

After the permanent inference pin, SceneWorks owns the single end-to-end execution: it resolves
the same case identities into rasters, runs its 200 hostile-sanitizer and 60 prompt-composition
suites, performs rendering/SSIM/LPIPS measurement, records memory and lifecycle outcomes, and
publishes the receipt. The text-only upstream source remains excluded from image-quality acceptance
because it can intentionally exercise sanitizer rejects.
