//! Real-CUDA acceptance for caller-owned native Krea DiTs (base, Kontext edit, and strict pose).
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, Conditioning, GenerationOutput, GenerationRequest, Generator, Image,
};
use candle_gen_krea::Krea2ControlRequest;

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
}

fn read_image(path: &Path) -> Image {
    let rgb = image::open(path).expect("decode fixture image").to_rgb8();
    let (width, height) = rgb.dimensions();
    Image {
        width,
        height,
        pixels: rgb.into_raw(),
    }
}

fn imported_adapter() -> AdapterSpec {
    AdapterSpec::new(env_path("KREA_IMPORTED_ADAPTER"), 1.0, AdapterKind::Lora)
}

fn render_one(generator: &dyn Generator, request: &GenerationRequest) -> Image {
    let GenerationOutput::Images(images) = generator.generate(request, &mut |_| {}).unwrap() else {
        panic!("expected image output")
    };
    assert_eq!(images.len(), 1);
    assert_eq!(
        images[0].pixels.len(),
        (images[0].width * images[0].height * 3) as usize
    );
    images.into_iter().next().unwrap()
}

#[test]
#[ignore = "requires explicitly scheduled CUDA and local imported Krea/base/adapter assets"]
fn imported_native_krea_adapter_renders_base_and_kontext_edit() {
    let native = env_path("KREA_IMPORTED_DIT");
    let base = env_path("KREA_BASE_SNAPSHOT");
    let adapter = imported_adapter();
    let request = GenerationRequest {
        prompt: "a cinematic portrait".to_owned(),
        width: 512,
        height: 512,
        steps: Some(2),
        seed: Some(7),
        ..Default::default()
    };
    let plain = candle_gen_krea::load_from_native_dit_file(
        &native,
        &base,
        &[],
        candle_gen_krea::descriptor(),
    )
    .expect("native Krea base loads without adapters");
    let plain_image = render_one(plain.as_ref(), &request);
    drop(plain);
    let adapted = candle_gen_krea::load_from_native_dit_file(
        &native,
        &base,
        std::slice::from_ref(&adapter),
        candle_gen_krea::descriptor(),
    )
    .expect("native Krea base + selected user adapter loads through production seam");
    let adapted_image = render_one(adapted.as_ref(), &request);
    assert_ne!(
        plain_image.pixels, adapted_image.pixels,
        "selected imported-base adapter must change deterministic output"
    );
    drop(adapted);

    let request = GenerationRequest {
        prompt: "turn the scene into warm golden-hour light".to_owned(),
        width: 512,
        height: 512,
        steps: Some(2),
        seed: Some(9),
        conditioning: vec![Conditioning::Reference {
            image: read_image(&env_path("KREA_IMPORTED_EDIT_SOURCE")),
            strength: None,
        }],
        ..Default::default()
    };
    let plain = candle_gen_krea::load_from_native_dit_file(
        &native,
        &base,
        &[],
        candle_gen_krea::edit_descriptor(),
    )
    .expect("native Krea Kontext edit loads without adapters");
    let plain_image = render_one(plain.as_ref(), &request);
    drop(plain);
    let adapted = candle_gen_krea::load_from_native_dit_file(
        native,
        base,
        &[adapter],
        candle_gen_krea::edit_descriptor(),
    )
    .expect("native Krea Kontext edit + selected user adapter loads through production seam");
    let adapted_image = render_one(adapted.as_ref(), &request);
    assert_ne!(
        plain_image.pixels, adapted_image.pixels,
        "selected imported-edit adapter must change deterministic output"
    );
}

#[test]
#[ignore = "requires explicitly scheduled CUDA and local imported Krea/control/adapter assets"]
fn imported_native_krea_adapter_renders_strict_pose() {
    let pose = read_image(&env_path("KREA_IMPORTED_POSE_IMAGE"));
    let native = env_path("KREA_IMPORTED_DIT");
    let base = env_path("KREA_BASE_SNAPSHOT");
    let control = env_path("KREA_CONTROL_BRANCH");
    let request = Krea2ControlRequest {
        prompt: "a cinematic portrait following the pose".to_owned(),
        width: pose.width,
        height: pose.height,
        steps: 2,
        seed: 11,
        ..Default::default()
    };
    let plain = candle_gen_krea::load_control_from_native_dit_file(&native, &base, &control, &[])
        .expect("native Krea strict-pose loads without adapters");
    let plain_image = plain
        .generate(&request, &pose, &mut |_| {})
        .expect("plain strict-pose render succeeds");
    drop(plain);
    let adapted = candle_gen_krea::load_control_from_native_dit_file(
        native,
        base,
        control,
        &[imported_adapter()],
    )
    .expect("native Krea strict-pose + selected user adapter loads through production seam");
    let output = adapted
        .generate(&request, &pose, &mut |_| {})
        .expect("adapted strict-pose render succeeds");
    assert_eq!((output.width, output.height), (pose.width, pose.height));
    assert_eq!(output.pixels.len(), (pose.width * pose.height * 3) as usize);
    assert_ne!(
        plain_image.pixels, output.pixels,
        "selected imported strict-control adapter must change deterministic output"
    );
}
