//! sc-16629 — real-weight Anima Turbo inference preview evidence.
//!
//! Loads a local Anima snapshot through the public `Generator` seam, captures every denoise-step
//! preview, and preserves individual frames, a nearest-neighbour strip, the final render, and run
//! metadata under `SC16629_ARTIFACT_DIR`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use mlx_gen::{
    GenerationOutput, GenerationRequest, LoadSpec, PreviewFrame, PreviewSink, WeightsSource,
};

const WIDTH: u32 = 512;
const HEIGHT: u32 = 512;
const STEPS: u32 = 8;
const SEED: u64 = 16629;
const PROMPT: &str =
    "Anime illustration of a silver-haired traveler beneath cherry blossoms, detailed, cinematic.";

fn artifact_dir() -> PathBuf {
    std::env::var_os("SC16629_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/sc16629-anima-preview"))
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
#[ignore = "needs ANIMA_SNAPSHOT pointing at a real split_files directory and a Metal device"]
fn turbo_generator_emits_numbered_developing_frame_strip() {
    let root = PathBuf::from(std::env::var_os("ANIMA_SNAPSHOT").expect("set ANIMA_SNAPSHOT"));
    let artifact_dir = artifact_dir();
    std::fs::create_dir_all(&artifact_dir).expect("create artifact directory");

    let load_started = Instant::now();
    let generator = mlx_gen_anima::load_turbo(&LoadSpec::new(WeightsSource::Dir(root.clone())))
        .expect("load Anima Turbo");
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
        .expect("render Anima Turbo with previews");
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
    assert_ne!(
        frames[0].image.pixels,
        frames.last().unwrap().image.pixels,
        "the preview trajectory must develop rather than repeat one latent"
    );
    let first_distinct = distinct_levels(&frames[0].image);
    let final_distinct = distinct_levels(&frames.last().unwrap().image);
    assert!(
        first_distinct > 16 && final_distinct > 16,
        "preview frames must be non-degenerate: first distinct={first_distinct}, final distinct={final_distinct}"
    );

    for frame in frames.iter() {
        save_rgb(
            &artifact_dir.join(format!("anima_turbo_512_s8_frame_{:02}.png", frame.current)),
            &frame.image,
        );
    }
    let strip_path = artifact_dir.join("anima_turbo_512_s8_preview_strip.png");
    let final_path = artifact_dir.join("anima_turbo_512_s8_final.png");
    let metadata_path = artifact_dir.join("evidence.txt");
    save_strip(&strip_path, &frames);
    save_rgb(&final_path, &images[0]);
    std::fs::write(
        &metadata_path,
        format!(
            "story=sc-16629\nsnapshot={}\ntier=dense bf16\ndevice=Metal\nmodel=anima_turbo\ngeometry={}x{}\nsteps={}\nseed={}\nframes={}\nnumbering=1..{} total={}\nfirst_distinct={}\nfinal_distinct={}\nload_seconds={:.3}\nrender_seconds={:.3}\nstrip={}\nfinal={}\n",
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
            load_seconds,
            render_seconds,
            strip_path.display(),
            final_path.display(),
        ),
    )
    .expect("write evidence metadata");

    eprintln!(
        "sc-16629 real-weight preview: snapshot={} tier=dense-bf16 device=Metal model=anima_turbo geometry={}x{} steps={} frames=1..{}/{} distinct={}→{} strip={} final={} load={:.1}s render={:.1}s",
        root.display(),
        WIDTH,
        HEIGHT,
        STEPS,
        STEPS,
        STEPS,
        first_distinct,
        final_distinct,
        strip_path.display(),
        final_path.display(),
        load_seconds,
        render_seconds,
    );
}
