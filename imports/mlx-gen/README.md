# mlx-gen

Rust-native inference for generative **image and video** models on Apple [MLX](https://github.com/ml-explore/mlx), built on [`mlx-rs`](https://crates.io/crates/mlx-rs).

> **Status:** active — 16 model provider crates with merged, parity-validated engines (image, video, identity, and understanding models). Built as a Rust library workspace consumed in-process; not yet published to crates.io. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

A from-scratch Rust reimplementation of the MLX image/video model stack (a divergence from the Python `mflux` / `mlx-video` lineage), collapsing on-device inference into a single statically-linked component with no Python sidecar. Each model family is its own provider crate registered through the core `mlx-gen` `Generator` contract.

**Supported models**

- **Image:** FLUX.1 (schnell/dev, incl. Hyper few-step), FLUX.2-klein (Qwen3 text encoder + KV-cache), Chroma (`chroma1_hd`/`base`/`flash`), Qwen-Image (+ Qwen-Image-Edit), Stable Diffusion XL (+ inpaint/outpaint, IP-Adapter, tile-ControlNet, LCM/Lightning/Hyper), Kolors (bilingual, ChatGLM3 text encoder), Z-Image (incl. ControlNet), SenseNova-U1 (unified understanding + generation: T2I, image-edit, VQA, interleaved document)
- **Video:** Wan2.2 (text/image-to-video, incl. VACE), LTX-2.3 (text-to-video), Stable Video Diffusion (image-to-video)
- **Identity:** PuLID-FLUX and InstantID, over a native MLX face stack (SCRFD + ArcFace + BiSeNet)
- **Understanding:** JoyCaption (captioning), SAM2 (segmentation)
- **Adapters:** LoRA, LoKr (reconstruct + forward-time residual + stacking, quant-safe), ControlNet, IP-Adapter
- **Training:** native MLX LoRA / LoKr fine-tuning for SDXL, Z-Image, Kolors, Wan2.2, and LTX-2.3 (adamw / adam / rose / prodigy optimizers, dataset + checkpoint plumbing)
- **Quantization:** group-wise affine Q4 / Q8 (byte-identical to the reference packing)
- **Weight converters:** native Rust weight-format converters for FLUX.2, Wan2.2 (T2V/I2V/TI2V + VAE), and LTX-2.3 — no Python conversion step

Requires a Mac with full Xcode + the Metal Toolchain (MLX's Metal kernels compile from source).

## Usage

mlx-gen is a Rust library workspace consumed in-process. Each model family lives in its own
provider crate that self-registers into the core `mlx-gen` registry at link time — so you depend
on `mlx-gen` plus whichever provider crates you want, then resolve models by id:

```toml
# Cargo.toml
[dependencies]
mlx-gen = { git = "https://github.com/michaeltrefry/mlx-gen" }
mlx-gen-z-image = { git = "https://github.com/michaeltrefry/mlx-gen" }
```

```rust
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource};

// A provider crate registers itself only when it is actually linked. Reference it once
// so the linker keeps its `inventory::submit!` registration.
use mlx_gen_z_image as _;

fn main() -> mlx_gen::Result<()> {
    // Load a model by id from a Hugging Face snapshot directory.
    let spec = LoadSpec::new(WeightsSource::Dir("/path/to/Z-Image-Turbo".into()));
    let model = mlx_gen::load("z_image_turbo", &spec)?;

    let req = GenerationRequest {
        prompt: "a red fox in a snowy forest".into(),
        width: 1024,
        height: 1024,
        seed: Some(42),
        ..Default::default()
    };

    let out = model.generate(&req, &mut |p| {
        if let Progress::Step { current, total } = p {
            println!("step {current}/{total}");
        }
    })?;

    if let GenerationOutput::Images(images) = out {
        let img = &images[0];
        // `img.pixels` is interleaved RGB (`img.width` × `img.height`); encode with any
        // image crate (e.g. `image::save_buffer`) to write a PNG.
        println!("generated {}×{}", img.width, img.height);
    }
    Ok(())
}
```

Discover what is registered at runtime with `mlx_gen::registry::generators()`. The same pattern
applies to the other entry points: `load_trainer` (LoRA/LoKr fine-tuning), `load_captioner`
(JoyCaption), and `load_transform` (SAM2).

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). You are free to
use, modify, and distribute mlx-gen, **including commercially**, under those terms.

## Acknowledgements

mlx-gen is an independent Rust reimplementation and includes no copied source,
but it stands on the work of others:

- [Apple MLX](https://github.com/ml-explore/mlx) (MIT) and [mlx-rs](https://crates.io/crates/mlx-rs) (Apache-2.0 OR MIT) — the on-device tensor stack
- [mflux](https://github.com/filipstrand/mflux) (MIT) — the MLX diffusion lineage mlx-gen diverged from and validates parity against
- [Apple mlx-examples](https://github.com/ml-explore/mlx-examples) (MIT)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) (Apache-2.0) — the upstream model architectures

See [NOTICE](NOTICE) for full attribution.
