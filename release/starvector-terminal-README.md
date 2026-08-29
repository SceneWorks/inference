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

The receipt validator is intentionally stricter than a summary report: each native backend records
120 cases with median SSIM at least 0.85, median LPIPS at most 0.20, and p95 latency at most 120
seconds. Each backend also carries all 20 rendered parity SSIM values, each at least 0.995. The 8B
record recomputes (rather than trusts) a per-backend median-LPIPS improvement of at least 10% over
1B, a validity delta no worse than -0.02, and a positive 95% bootstrap lower bound.
