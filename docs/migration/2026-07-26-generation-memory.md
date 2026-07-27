# Generation request memory adaptations

`GenerationRequest` now carries an optional `GenerationMemory` value for
provider-specific, quality-preserving memory adaptations:

- tiled VAE decode;
- attention chunking; and
- transformer block streaming.

The compatibility default is `None`, so existing request construction and
provider behavior are unchanged. The initial implementation applies only to
ordinary Krea 2 Turbo text-to-image generation under sequential CUDA residency.
Raw, edit, control, PiD, ConvRot, adapter, and resident paths retain their
existing behavior.

Consumers that use exhaustive `GenerationRequest` literals must initialize the
new `memory` field. Consumers using `..Default::default()` require no change.
SceneWorks selects these adaptations from live free VRAM and the model
manifest's measured, per-tier phase curves.
