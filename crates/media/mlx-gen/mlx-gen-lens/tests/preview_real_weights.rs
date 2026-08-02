//! Real-weight Lens-Turbo active-preview transfer evidence.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use mlx_gen::{CancelFlag, Image, PreviewSink};
use mlx_gen_lens::pipeline::{LensHeavy, LensText, DEFAULT_DATE};
use mlx_rs::Dtype;

fn snapshot() -> PathBuf {
    PathBuf::from(std::env::var("LENS_PREVIEW_SNAPSHOT").unwrap_or_else(|_| {
        panic!("set LENS_PREVIEW_SNAPSHOT to the required real Lens-Turbo snapshot")
    }))
}

#[test]
#[ignore = "needs real Lens-Turbo weights and a Metal-capable macOS host"]
fn active_preview_emits_native_four_step_strip_without_changing_rgb() {
    mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
    let root = snapshot();
    let text = LensText::load(&root, Dtype::Bfloat16, None).unwrap();
    let cancel = CancelFlag::default();
    let (encoder_features, encoder_mask) = text
        .encode_prompt(
            "a red fox beside a turquoise alpine lake at golden hour",
            "blurry, low quality",
            DEFAULT_DATE,
            Some(&cancel),
        )
        .unwrap();
    let mut encoded = encoder_features.iter().collect::<Vec<_>>();
    encoded.push(&encoder_mask);
    mlx_rs::transforms::eval(encoded).unwrap();
    drop(text);
    mlx_rs::memory::clear_cache();

    let heavy = LensHeavy::load(&root, Dtype::Bfloat16).unwrap();
    let render = |preview: &PreviewSink| -> Image {
        heavy
            .render_with_preview(
                &encoder_features,
                &encoder_mask,
                16,
                16,
                4,
                1.0,
                None,
                None,
                1663088,
                None,
                None,
                &cancel,
                &mut |_| {},
                preview,
            )
            .unwrap()
    };

    // The first dense Lens render materializes lazy weights. Discard it so both parity samples use
    // the same warmed generator state and differ only in whether the preview sink is active.
    let _warmup = render(&PreviewSink::default());

    let frames = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&frames);
    let active = render(&PreviewSink::new(move |frame| {
        captured.lock().unwrap().push(frame)
    }));

    let frames = frames.lock().unwrap();
    // Retain the active strip before checking cadence or parity so a failure remains inspectable.
    if let Ok(path) = std::env::var("LENS_PREVIEW_ARTIFACT_DIR") {
        let dir = PathBuf::from(path);
        std::fs::create_dir_all(&dir).unwrap();
        if let Some(first) = frames.first() {
            let (frame_width, frame_height) =
                (first.image.width as usize, first.image.height as usize);
            let mut strip = vec![0u8; frame_width * frames.len() * frame_height * 3];
            for (frame_index, frame) in frames.iter().enumerate() {
                for y in 0..frame_height {
                    let src = y * frame_width * 3;
                    let dst = (y * frame_width * frames.len() + frame_index * frame_width) * 3;
                    strip[dst..dst + frame_width * 3]
                        .copy_from_slice(&frame.image.pixels[src..src + frame_width * 3]);
                }
            }
            image::save_buffer(
                dir.join("lens_turbo_preview_strip.png"),
                &strip,
                (frame_width * frames.len()) as u32,
                frame_height as u32,
                image::ColorType::Rgb8,
            )
            .unwrap();
        }
        image::save_buffer(
            dir.join("lens_turbo_preview_final.png"),
            &active.pixels,
            active.width,
            active.height,
            image::ColorType::Rgb8,
        )
        .unwrap();
    }

    assert_eq!(frames.len(), 4);
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!((frame.current, frame.total), (index as u32 + 1, 4));
        assert_eq!((frame.image.width, frame.image.height), (32, 32));
    }
    drop(frames);

    let inert = render(&PreviewSink::default());
    assert!(
        active.width == inert.width
            && active.height == inert.height
            && active.pixels == inert.pixels,
        "warm active preview changed RGB8 bytes"
    );
}
