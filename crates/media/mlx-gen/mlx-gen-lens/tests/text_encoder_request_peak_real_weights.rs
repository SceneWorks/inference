//! SC-15800 end-to-end Lens request-peak A/B. The isolated encoder sweep proves the conditioning
//! bound; this test determines whether the production request peak moves after all phases run.

use mlx_gen::gen_core::{GenerationMemory, TransformerComponent};
use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadShape, LoadSpec, OffloadPolicy, Quant,
    WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn snapshot() -> std::path::PathBuf {
    let path = std::path::PathBuf::from(
        std::env::var("LENS_DIR").expect("set LENS_DIR to an explicit local Lens tier directory"),
    );
    assert!(
        path.is_dir(),
        "Lens tier does not exist: {}",
        path.display()
    );
    path
}

fn tier(root: &std::path::Path) -> Option<Quant> {
    match root.file_name().and_then(|name| name.to_str()) {
        Some("q4") => Some(Quant::Q4),
        Some("q8") => Some(Quant::Q8),
        _ => None,
    }
}

struct Run {
    request: u64,
    image: Image,
}

fn run(window: Option<u32>) -> Run {
    let root = snapshot();
    assert!(
        tier(&root).is_none(),
        "the production request A/B is dense BF16-only; Q4's measured request peak did not improve"
    );
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LoadShape::DeferredMaterialization);
    if let Some(quant) = tier(&root) {
        spec = spec.with_quant(quant);
    }
    let memory = window.map(|window| GenerationMemory {
        stream_transformer_blocks: true,
        transformer_window_size: Some(window),
        transformer_window_component: Some(TransformerComponent::TextEncoder),
        ..Default::default()
    });
    let request = GenerationRequest {
        prompt: "a red fox crossing a snowy clearing at dawn, documentary photograph".into(),
        width: 256,
        height: 256,
        count: 1,
        steps: Some(1),
        guidance: Some(1.0),
        seed: Some(15800),
        memory,
        ..Default::default()
    };
    let model = mlx_gen_lens::provider_registry()
        .expect("Lens registry")
        .load("lens_turbo", &spec)
        .expect("load Lens-Turbo");
    clear_cache();
    reset_peak_memory();
    let output = model
        .generate(&request, &mut |_| {})
        .expect("generate Lens image");
    // One uninterrupted counter is the actual request peak. The isolated companion test measures the
    // conditioning phase; resetting at progress callbacks would turn this into a stitched estimate.
    let request_peak = get_peak_memory() as u64;
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("one image"),
        other => panic!("expected image output, got {other:?}"),
    };
    drop(model);
    clear_cache();
    Run {
        request: request_peak,
        image,
    }
}

fn image_delta(a: &Image, b: &Image) -> u8 {
    assert_eq!((a.width, a.height), (b.width, b.height));
    a.pixels
        .iter()
        .zip(&b.pixels)
        .fold(0_u8, |worst, (a, b)| worst.max(a.abs_diff(*b)))
}

#[test]
#[ignore = "needs a complete real Lens tier and Apple/Metal"]
fn encoder_scope_reports_the_request_peak_effect_without_changing_the_image() {
    // The control still uses the streamable Sequential encoder, but as one all-covering unscoped
    // window. This isolates the tunable component-scope effect from component staging itself.
    let unscoped = run(None);
    let scoped = run(Some(1));
    println!(
        "\nSC-15800 Lens request peak A/B ({})",
        snapshot().display()
    );
    println!("  {:<10} {:>11}", "arm", "request");
    for (name, value) in [("unscoped", &unscoped), ("text-w=1", &scoped)] {
        println!("  {:<10} {:>10.3}G", name, value.request as f64 / GIB);
    }
    let request_cut = 1.0 - scoped.request as f64 / unscoped.request.max(1) as f64;
    println!(
        "  resulting request-peak change: {:.1}%",
        request_cut * 100.0
    );
    assert_eq!(
        image_delta(&unscoped.image, &scoped.image),
        0,
        "text-encoder streaming changed the generated image"
    );
    assert!(
        scoped.request <= unscoped.request + unscoped.request / 20,
        "the bounded scope raised request peak by more than 5% ({:.3} -> {:.3} GiB)",
        unscoped.request as f64 / GIB,
        scoped.request as f64 / GIB
    );
}
