# candle-audio-yue2 test fixtures

| Fixture | Generator | Used by |
| --- | --- | --- |
| `vae_tiny/{standard,legacy}/{config.json,model.safetensors}`, `vae_tiny_reference.{safetensors,json}` | `scripts/reference/yue2/vae_reference.py tiny` | lib tests of `vae`, `decode`, `latent` (always run in CI) |
| `vae_real_reference.json` | `scripts/reference/yue2/vae_reference.py real` | `tests/vae_real_weights.rs` (`#[ignore]`d) |
| `protocol/` | `scripts/reference/yue2/protocol_fixtures.py` | tokenizer / protocol / plan tests (see `protocol/README.md`) |

Provenance: upstream `github.com/multimodal-art-projection/YuE` @
`92a73cc7652fcc1f937855e4b765e0a0edd7ff2e` (the installed pinned `yue2` package, never a copy), run
on the CPU in FP32 in the shared reference environment (`scripts/reference/yue2/setup_reference_env.sh`);
each JSON's `reference` block records Python, torch, numpy, safetensors and transformers versions.

- The `vae_tiny` models are **synthetic**: the released VAE topology (strides `[2, 2, 4, 4, 5, 6]`,
  64 latent channels, SnakeBeta, weight-norm, stereo, no final tanh) at toy widths with seeded random
  weights. They contain no YuE2 weights.
- The real-weight reference uses `m-a-p/YuE2-Vae` @ `95535e72a97bc0f09b8ada125d26b4009428c0e8` and
  `m-a-p/YuE2-Vae-legacy` @ `b54118f0fc462f08999d1ec07e88817f4ee3f770`. Its waveforms derive from
  CC BY-NC 4.0 weights, so the tensors stay outside the repository
  (`~/.cache/sceneworks-yue2-fixtures/vae/vae_real_reference.safetensors`); only their SHA-256,
  shapes and statistics are committed here, and the test refuses a reference file with another hash.
