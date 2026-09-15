//! sc-16628 — real-weight Krea Turbo preview evidence.
//!
//! Loads the locally selected Krea snapshot through the public `Generator` seam, captures every
//! per-step preview, checks schedule numbering, and preserves a nearest-neighbour frame strip plus
//! the final render and run metadata under `SC16628_ARTIFACT_DIR` (default `/tmp/sc16628-krea-preview`).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use mlx_gen::{
    GenerationOutput, GenerationRequest, LoadSpec, PreviewFrame, PreviewSink, Quant, WeightsSource,
};

const WIDTH: u32 = 512;
const HEIGHT: u32 = 512;
const STEPS: u32 = 8;
const SEED: u64 = 16628;
const PROMPT: &str =
    "A red fox sitting in a snowy pine forest at golden hour, detailed photograph.";

fn artifact_dir() -> PathBuf {
    std::env::var_os("SC16628_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/sc16628-krea-preview"))
}

fn save_rgb(path: &Path, image: &mlx_gen::Image) {
    image::save_buffer(
        path,
        &image.pixels,
        image.width,
        image.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap_or_else(|error| panic!("save {}: {error}", path.display()));
}

fn adjacent_delta(image: &mlx_gen::Image) -> f32 {
    let stride = image.width as usize * 3;
    let mut sum = 0_u64;
    let mut count = 0_u64;
    for (index, &value) in image.pixels.iter().enumerate() {
        if index >= 3 && index % stride >= 3 {
            sum += (value as i32 - image.pixels[index - 3] as i32).unsigned_abs() as u64;
            count += 1;
        }
    }
    sum as f32 / count.max(1) as f32
}

fn distinct_levels(image: &mlx_gen::Image) -> usize {
    let mut seen = [false; 256];
    for &value in &image.pixels {
        seen[value as usize] = true;
    }
    seen.into_iter().filter(|present| *present).count()
}

fn save_strip(path: &Path, frames: &[PreviewFrame]) {
    let scale = 4;
    let tile_w = frames[0].image.width * scale;
    let tile_h = frames[0].image.height * scale;
    let mut strip = image::RgbImage::new(tile_w * frames.len() as u32, tile_h);
    for (index, frame) in frames.iter().enumerate() {
        let tile = image::RgbImage::from_raw(
            frame.image.width,
            frame.image.height,
            frame.image.pixels.clone(),
        )
        .expect("preview RGB buffer dimensions");
        let tile =
            image::imageops::resize(&tile, tile_w, tile_h, image::imageops::FilterType::Nearest);
        image::imageops::replace(&mut strip, &tile, index as i64 * tile_w as i64, 0);
    }
    strip
        .save(path)
        .unwrap_or_else(|error| panic!("save {}: {error}", path.display()));
}

#[test]
#[ignore = "needs KREA_TURBO_DIR pointing at a real Krea Turbo snapshot and a Metal device"]
fn turbo_generator_emits_numbered_developing_frame_strip() {
    let root = PathBuf::from(std::env::var_os("KREA_TURBO_DIR").expect("set KREA_TURBO_DIR"));
    let artifact_dir = artifact_dir();
    std::fs::create_dir_all(&artifact_dir).expect("create artifact directory");

    let spec = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(Quant::Q4);
    let load_started = Instant::now();
    let generator = mlx_gen_krea::load(&spec).expect("load Krea Turbo q4");
    let load_seconds = load_started.elapsed().as_secs_f32();

    let frames = Arc::new(Mutex::new(Vec::<PreviewFrame>::new()));
    let captured = Arc::clone(&frames);
    let preview = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));
    let request = GenerationRequest {
        prompt: PROMPT.into(),
        width: WIDTH,
        height: HEIGHT,
        steps: Some(STEPS),
        seed: Some(SEED),
        preview,
        ..Default::default()
    };

    let render_started = Instant::now();
    let output = generator
        .generate(&request, &mut |_| {})
        .expect("render Krea Turbo with previews");
    let render_seconds = render_started.elapsed().as_secs_f32();
    let GenerationOutput::Images(images) = output else {
        panic!("expected image output");
    };
    assert_eq!(images.len(), 1);

    let frames = frames.lock().unwrap();
    assert_eq!(frames.len(), STEPS as usize);
    assert_eq!(
        frames
            .iter()
            .map(|frame| (frame.current, frame.total))
            .collect::<Vec<_>>(),
        (1..=STEPS)
            .map(|current| (current, STEPS))
            .collect::<Vec<_>>()
    );
    assert!(frames
        .iter()
        .all(|frame| (frame.image.width, frame.image.height) == (WIDTH / 8, HEIGHT / 8)));

    let first_delta = adjacent_delta(&frames[0].image);
    let final_delta = adjacent_delta(&frames.last().unwrap().image);
    let first_distinct = distinct_levels(&frames[0].image);
    let final_distinct = distinct_levels(&frames.last().unwrap().image);
    assert_ne!(
        frames[0].image.pixels,
        frames.last().unwrap().image.pixels,
        "the preview trajectory must develop rather than repeat one latent"
    );
    assert!(
        first_distinct > 16 && final_distinct > 16,
        "preview frames must be non-degenerate: first distinct={first_distinct}, final distinct={final_distinct}"
    );

    let strip_path = artifact_dir.join("krea_turbo_q4_512_s8_preview_strip.png");
    let final_path = artifact_dir.join("krea_turbo_q4_512_s8_final.png");
    let metadata_path = artifact_dir.join("evidence.txt");
    save_strip(&strip_path, &frames);
    save_rgb(&final_path, &images[0]);
    std::fs::write(
        &metadata_path,
        format!(
            "story=sc-16628\nsnapshot={}\ntier=q4 (transformer/config.json bits=4 group_size=64)\ndevice=Metal\ngeometry={}x{}\nsteps={}\nseed={}\nframes={}\nnumbering=1..{} total={}\nfirst_distinct={}\nfinal_distinct={}\nfirst_adjacent_delta={:.3}\nfinal_adjacent_delta={:.3}\nload_seconds={:.3}\nrender_seconds={:.3}\nstrip={}\nfinal={}\n",
            root.display(),
            WIDTH,
            HEIGHT,
            STEPS,
            SEED,
            frames.len(),
            STEPS,
            STEPS,
            first_distinct,
            final_distinct,
            first_delta,
            final_delta,
            load_seconds,
            render_seconds,
            strip_path.display(),
            final_path.display(),
        ),
    )
    .expect("write evidence metadata");

    eprintln!(
        "sc-16628 real-weight preview: snapshot={} tier=q4(bits=4,group=64) device=Metal geometry={}x{} steps={} frames=1..{}/{} distinct={}→{} adjacent_delta={:.2}→{:.2} strip={} final={} load={:.1}s render={:.1}s",
        root.display(),
        WIDTH,
        HEIGHT,
        STEPS,
        STEPS,
        STEPS,
        first_distinct,
        final_distinct,
        first_delta,
        final_delta,
        strip_path.display(),
        final_path.display(),
        load_seconds,
        render_seconds,
    );
}
