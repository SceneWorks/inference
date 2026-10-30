# gen-core-family

> Package: `sceneworks-gen-core-family` · library: `gen_core_family`

Small tensor-free **single-family host policy**, each module used by one model family's MLX and
Candle providers: schedulers (Ideogram, Krea multiphase/schedule), resolution and geometry (Lens,
SeedVR2 video, SDXL LDM), keypoint and face-alignment math (InstantID, face/FX programs), the
Qwen-Image 2.1 config/reference/scheduler, and the SD3 text-encoder artifact resolution and
seeded-output mapping.

Split out of [`gen-core`](../gen-core/README.md) in sc-27316. It depends on gen-core (gen-core
never depends on it) and, like gen-core, on no tensor library. Grouped into one crate so a change
to one of them rebuilds only these families, never gen-core.

## License

Apache-2.0.
