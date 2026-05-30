# mlx-gen

Rust-native inference for generative **image and video** models on Apple [MLX](https://github.com/ml-explore/mlx), built on [`mlx-rs`](https://crates.io/crates/mlx-rs).

> **Status:** name reserved / work in progress — not yet usable.

A from-scratch Rust reimplementation of the MLX image/video model stack (a divergence from the Python `mflux` / `mlx-video` lineage), collapsing on-device inference into a single statically-linked binary with no Python sidecar.

**Planned scope**

- **Image:** FLUX.1, FLUX.2-klein (incl. KV-cache), Qwen-Image, Z-Image (incl. ControlNet)
- **Video:** Wan2.2, LTX-2.3
- **Adapters:** LoRA, LoKr (reconstruct + residual + stacking), ControlNet
- **Quantization:** Q4 / Q8

Requires a Mac with full Xcode + the Metal Toolchain (MLX's Metal kernels compile from source).
