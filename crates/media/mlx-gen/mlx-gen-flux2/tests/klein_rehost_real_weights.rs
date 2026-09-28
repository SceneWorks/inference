//! sc-22760: the production MLX worker hands `mlx-gen-flux2` the raw HF cache snapshot plus tier
//! subdir (`models--SceneWorks--flux2-klein-9b[-kv]-mlx/snapshots/<rev>/<bf16|q4|q8>`), where every
//! entry is a symlink into the sibling `blobs/` and the bf16 tier is sharded. This is the one
//! real-weight row that loads that exact shape through the public registry for both rehosts at all
//! three tiers; the unit fixtures in `artifact_inventory.rs` only model it.
//!
//! Every tier ships the Qwen3 text encoder and the VAE dense; only the transformer packs. The
//! pinned revisions (`1902693279` base, `bbf22de8d6` kv) are the corrected rehosts: their q4/q8
//! `text_encoder/config.json` carries no `quantization` marker, so the artifact inventory admits
//! every tier as published (an earlier upload carried a stale marker over the dense tensors, which
//! the inventory refuses).
//!
//! `#[ignore]`d — needs both rehosts in an explicit models root and Apple/Metal:
//!
//! ```sh
//! MLX_GEN_MODELS_ROOT=/path/to/hub \
//! cargo test -p mlx-gen-flux2 --release --test integration klein_rehost_real_weights:: -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};
use mlx_gen_flux2::{FLUX2_KLEIN_9B_ID, FLUX2_KLEIN_9B_KV_EDIT_ID};

const BASE_REHOST: (&str, &str) = (
    "models--SceneWorks--flux2-klein-9b-mlx",
    "1902693279fcfb828919370dfac2b8922d99499a",
);
const KV_REHOST: (&str, &str) = (
    "models--SceneWorks--flux2-klein-9b-kv-mlx",
    "bbf22de8d654789de3b177632d2e283cc4f77729",
);
const TIERS: [(&str, Option<Quant>); 3] = [
    ("bf16", None),
    ("q4", Some(Quant::Q4)),
    ("q8", Some(Quant::Q8)),
];

fn models_root() -> PathBuf {
    PathBuf::from(std::env::var("MLX_GEN_MODELS_ROOT").expect(
        "set MLX_GEN_MODELS_ROOT to the explicit models root (holds models--*/snapshots); \
         inference never self-fetches or derives a cache location (epic 13657)",
    ))
}

fn tier_root((cache_dir, revision): (&str, &str), tier: &str) -> PathBuf {
    let root = models_root()
        .join(cache_dir)
        .join("snapshots")
        .join(revision)
        .join(tier);
    assert!(
        root.is_dir(),
        "missing rehost tier {}; the worker loads exactly this path",
        root.display()
    );
    root
}

fn load(provider_id: &str, root: PathBuf) -> Box<dyn mlx_gen::Generator> {
    mlx_gen_flux2::provider_registry()
        .expect("FLUX.2 registry")
        .load(
            provider_id,
            &LoadSpec::new(WeightsSource::Dir(root.clone())),
        )
        .unwrap_or_else(|error| panic!("{provider_id} refused {}: {error}", root.display()))
}

/// A load that reached the registry admitted the exact turnkey inventory: the memory contract is
/// built on every Klein load and refuses before the generator exists, so its presence is the gate's
/// receipt. The shipped markers are re-read so the row also documents the artifact truth the gate
/// encodes: only the transformer packs; the text encoder and the VAE stay dense at every tier.
fn assert_admitted_tier(
    generator: &dyn mlx_gen::Generator,
    provider_id: &str,
    root: &std::path::Path,
    quant: Option<Quant>,
) {
    let contract = generator
        .memory_strategy_contract()
        .expect("Klein publishes a memory contract on every load");
    assert_eq!(contract.provider_id, provider_id);
    assert_eq!(
        mlx_gen::quant::packed_quant_bits(root, "transformer").unwrap(),
        quant.map(Quant::bits),
        "{provider_id} transformer at {}",
        root.display()
    );
    for component in ["text_encoder", "vae"] {
        assert_eq!(
            mlx_gen::quant::packed_quant_bits(root, component).unwrap(),
            None,
            "{provider_id} {component} at {}",
            root.display()
        );
    }
}

fn render_probe(generator: &dyn mlx_gen::Generator) {
    let request = GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        width: 256,
        height: 256,
        count: 1,
        steps: Some(1),
        seed: Some(1234),
        ..Default::default()
    };
    let output = generator
        .generate(&request, &mut |_| {})
        .expect("one-step probe render");
    match output {
        GenerationOutput::Images(images) => assert_eq!(images.len(), 1),
        other => panic!("expected image output, got {other:?}"),
    }
}

#[test]
#[ignore = "needs both SceneWorks Klein rehosts in MLX_GEN_MODELS_ROOT and Apple/Metal"]
fn every_rehost_tier_loads_from_the_hf_cache_shape() {
    for (tier, quant) in TIERS {
        let started = std::time::Instant::now();
        let root = tier_root(BASE_REHOST, tier);
        let generator = load(FLUX2_KLEIN_9B_ID, root.clone());
        assert_admitted_tier(generator.as_ref(), FLUX2_KLEIN_9B_ID, &root, quant);
        render_probe(generator.as_ref());
        drop(generator);
        eprintln!(
            "base {tier}: loaded + probed in {:.1}s",
            started.elapsed().as_secs_f64()
        );

        let started = std::time::Instant::now();
        let root = tier_root(KV_REHOST, tier);
        let generator = load(FLUX2_KLEIN_9B_KV_EDIT_ID, root.clone());
        assert_admitted_tier(generator.as_ref(), FLUX2_KLEIN_9B_KV_EDIT_ID, &root, quant);
        drop(generator);
        eprintln!(
            "kv {tier}: loaded in {:.1}s",
            started.elapsed().as_secs_f64()
        );
    }
}
