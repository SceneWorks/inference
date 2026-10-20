//! The registered provider's source resolution refuses bundles not converted from the pinned
//! checkpoint (E3) — tensor-free, so it runs in every lane that builds this crate.

use std::path::Path;

use mlx_gen::gen_core::supra2::{
    convert_checkpoint, ConvertOptions, Supra2Config, TEXT_ENCODER_COMPONENT, VAE_COMPONENT,
};
use mlx_gen::{LoadSpec, WeightsSource};
use mlx_gen_supra2::Supra2Sources;

#[test]
fn provider_refuses_an_unpinned_bundle_naming_the_pin() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../contracts/gen-core/tests/fixtures/supra2/tiny_model_with_uncond.pt");
    let out = tempfile::Builder::new()
        .prefix(&format!("supra2-unpinned-{}-", std::process::id()))
        .tempdir()
        .unwrap();
    convert_checkpoint(
        &src,
        out.path(),
        &ConvertOptions {
            config: Supra2Config {
                latent_channels: 4,
                latent_size: 16,
                patch: 2,
                d_model: 32,
                depth: 2,
                num_heads: 2,
                mlp_hidden: 128,
                ctx_dim: 24,
                ctx_len: 8,
                freq_dim: 256,
            },
            require_pinned_source: false,
        },
    )
    .unwrap();
    // Components point at directories that do not exist: the bundle check must refuse first.
    let spec = LoadSpec::new(WeightsSource::Dir(out.path().to_path_buf()))
        .with_component(
            TEXT_ENCODER_COMPONENT,
            WeightsSource::Dir("/nonexistent-t5".into()),
        )
        .with_component(VAE_COMPONENT, WeightsSource::Dir("/nonexistent-vae".into()));
    let err = Supra2Sources::resolve(&spec).unwrap_err().to_string();
    assert!(err.contains("unpinned source"), "{err}");
    assert!(
        err.contains("SupraLabs/Supra2-IMG@10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b"),
        "{err}"
    );
}
