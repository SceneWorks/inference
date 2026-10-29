//! Independent frozen-Pillow file modes. No renderer, request, reference-plan, or tensor API.
use gen_core_flux3::{
    image_input::{decode_reference_file, DecodedReference},
    pup::{policy_reference, ReferenceImage},
};
use std::path::PathBuf;

fn fixtures() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/flux3-image-input")
}

#[test]
fn file_modes_match_frozen_pillow_ae_and_pup() {
    let root = fixtures();
    // NUL-free raw pixels are still binary; EOL guessing must never rewrite source samples.
    let attributes = std::fs::read_to_string(root.join(".gitattributes")).unwrap();
    assert!(attributes.lines().any(|line| line.trim() == "*.rgb binary"));
    assert!(attributes
        .lines()
        .any(|line| line.trim() == "*.rgba binary"));
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("image-input.json")).unwrap()).unwrap();
    assert_eq!(metadata["upstream"], gen_core_flux3::UPSTREAM_CODE_REVISION);
    assert_eq!(
        metadata["source_sha256"],
        "33c323ebaeaf56790f3d0022a57827b5348446a8a4b372ab2bcc606d722e76a9"
    );
    assert_eq!(
        metadata["pup_source_sha256"],
        "e3561ae3cbff1a080ad8854c0b89bd0687c1ea9d84c74f5203b44844cbd4684d"
    );
    let cases = metadata["encoded_formats"].as_array().unwrap();
    assert_eq!(cases.len(), 9);
    for row in cases {
        let name = row["name"].as_str().unwrap();
        let path = root.join(row["file"].as_str().unwrap());
        let decoded = decode_reference_file(&path).unwrap();
        let (width, height, ae_rgb, input) = match decoded {
            DecodedReference::Rgb(image) => {
                assert_eq!(row["carrier"], "rgb", "{name}: original source mode");
                let input = ReferenceImage::Rgb8 {
                    width: image.width,
                    height: image.height,
                    pixels: image.pixels.clone(),
                };
                (image.width, image.height, image.pixels, input)
            }
            DecodedReference::Rgba(image) => {
                assert_eq!(row["carrier"], "rgba", "{name}: original source alpha mode");
                assert_eq!(
                    image.pixels,
                    std::fs::read(root.join(format!("{name}.decoded.rgba"))).unwrap(),
                    "{name}: hidden RGB and alpha before either fork"
                );
                let ae_rgb = image
                    .pixels
                    .chunks_exact(4)
                    .flat_map(|p| p[..3].iter().copied())
                    .collect::<Vec<_>>();
                let input = ReferenceImage::Rgba8 {
                    width: image.width,
                    height: image.height,
                    pixels: image.pixels,
                };
                (image.width, image.height, ae_rgb, input)
            }
        };
        assert_eq!(
            (width as u64, height as u64),
            (
                row["size"][0].as_u64().unwrap(),
                row["size"][1].as_u64().unwrap()
            )
        );
        assert_eq!(
            ae_rgb,
            std::fs::read(root.join(format!("{name}.ae.rgb"))).unwrap(),
            "{name}: exact Pillow RGB for AE (including gray16 clamp/color16 high byte)"
        );
        let expected = std::fs::read(root.join(format!("{name}.pup.rgb"))).unwrap();
        assert_eq!(
            input.to_rgb8().unwrap().2,
            expected,
            "{name}: raw normalized carrier PUP fork"
        );
        let path_input = ReferenceImage::Path(path);
        assert_eq!(
            path_input.to_rgb8().unwrap().2,
            expected,
            "{name}: actual PUP Path original-mode policy"
        );
        for count in [1, 10] {
            let expected = std::fs::read(root.join(format!("{name}.policy{count}.rgb"))).unwrap();
            for input in [&input, &path_input] {
                let actual = policy_reference(input, count).unwrap();
                assert_eq!(
                    actual.pixels, expected,
                    "{name}: full frozen PUP RGB/LANCZOS/JPEG q92 at {count} refs"
                );
                assert_eq!(
                    actual.width as u64,
                    row[format!("policy{count}_size")][0].as_u64().unwrap()
                );
                assert_eq!(
                    actual.height as u64,
                    row[format!("policy{count}_size")][1].as_u64().unwrap()
                );
            }
        }
    }
}

#[test]
fn invalid_file_and_bad_png_crc_fail_without_a_fallback() {
    let dir = tempfile::tempdir().unwrap();
    assert!(decode_reference_file(&dir.path().join("missing.png")).is_err());
    let empty = dir.path().join("empty.png");
    std::fs::write(&empty, []).unwrap();
    assert!(decode_reference_file(&empty).is_err());
    let mut bytes = std::fs::read(fixtures().join("mode-luma16.png")).unwrap();
    bytes[29] ^= 0xff; // corrupt critical IHDR CRC, not the mode hint itself
    let corrupt = dir.path().join("bad-crc.png");
    std::fs::write(&corrupt, bytes).unwrap();
    assert!(
        decode_reference_file(&corrupt).is_err(),
        "IHDR hints never bypass the normal PNG decoder's CRC validation"
    );
    assert!(ReferenceImage::Path(corrupt).to_rgb8().is_err());
}
