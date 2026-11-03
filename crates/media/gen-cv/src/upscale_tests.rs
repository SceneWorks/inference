use super::*;
use image::Rgb;
#[test]
fn tile_slices_single_when_image_fits() {
    // tile >= max(w,h) → one tile covering the whole image (upscalers.py).
    let t = tile_slices(400, 300, 512);
    assert_eq!(
        t,
        vec![Tile {
            x0: 0,
            y0: 0,
            x1: 400,
            y1: 300
        }]
    );
    // exact-fit edge: tile == max dim still single (>= guard).
    assert_eq!(tile_slices(512, 512, 512).len(), 1);
}

#[test]
fn tile_slices_grid_row_major_clamped() {
    // 768x768 @ tile 512 → 2x2 grid, edge tiles clamped to bounds.
    let t = tile_slices(768, 768, 512);
    assert_eq!(t.len(), 4);
    assert_eq!(
        t[0],
        Tile {
            x0: 0,
            y0: 0,
            x1: 512,
            y1: 512
        }
    );
    assert_eq!(
        t[1],
        Tile {
            x0: 512,
            y0: 0,
            x1: 768,
            y1: 512
        }
    );
    assert_eq!(
        t[2],
        Tile {
            x0: 0,
            y0: 512,
            x1: 512,
            y1: 768
        }
    );
    assert_eq!(
        t[3],
        Tile {
            x0: 512,
            y0: 512,
            x1: 768,
            y1: 768
        }
    );
    // full coverage, no gaps/overlaps in the (unpadded) inner grid
    let covered: usize = t.iter().map(|s| (s.x1 - s.x0) * (s.y1 - s.y0)).sum();
    assert_eq!(covered, 768 * 768);
}

#[test]
fn tile_slices_zero_tile_is_single() {
    assert_eq!(tile_slices(100, 50, 0).len(), 1);
}

#[test]
fn crop_to_chw_layout_and_normalization() {
    let mut img = RgbImage::new(3, 2);
    img.put_pixel(0, 0, Rgb([255, 0, 0]));
    img.put_pixel(1, 0, Rgb([0, 255, 0]));
    img.put_pixel(2, 0, Rgb([0, 0, 255]));
    img.put_pixel(0, 1, Rgb([0, 0, 0]));
    img.put_pixel(1, 1, Rgb([128, 128, 128]));
    img.put_pixel(2, 1, Rgb([255, 255, 255]));

    let (data, cw, ch) = crop_to_chw(&img, 0, 0, 3, 2);
    assert_eq!((cw, ch), (3, 2));
    assert_eq!(data.len(), 3 * 3 * 2);
    // CHW: R plane first. R[0,0]=1.0, G[0,0]=0.0, B[2,0]=1.0
    assert!((data[0] - 1.0).abs() < 1e-6); // R(0,0)
    let g_plane = cw * ch;
    assert!((data[g_plane + 1] - 1.0).abs() < 1e-6); // G(1,0)
    let b_plane = 2 * cw * ch;
    assert!((data[b_plane + 2] - 1.0).abs() < 1e-6); // B(2,0)
                                                     // mid-gray (1,1).G normalizes to 128/255
    assert!((data[g_plane + 4] - 128.0 / 255.0).abs() < 1e-6);
    assert!((data[g_plane - 1] - 1.0).abs() < 1e-6); // R(2,1)=255 last in R plane
}

#[test]
fn crop_to_chw_subregion() {
    let mut img = RgbImage::new(4, 4);
    for y in 0..4 {
        for x in 0..4 {
            img.put_pixel(x, y, Rgb([(x * 10) as u8, (y * 10) as u8, 0]));
        }
    }
    let (data, cw, ch) = crop_to_chw(&img, 1, 1, 3, 3);
    assert_eq!((cw, ch), (2, 2));
    // R plane top-left = pixel (1,1).R = 10/255
    assert!((data[0] - 10.0 / 255.0).abs() < 1e-6);
    // G plane bottom-right = pixel (2,2).G = 20/255
    let g = cw * ch;
    assert!((data[g + 3] - 20.0 / 255.0).abs() < 1e-6);
}

#[test]
fn pad_chw_even_replicates_to_even_dims_for_pixel_unshuffle() {
    // odd width (the 323px-wide regression): 3x2 CHW → padded to 4x2, the new column replicates
    // the last (no dark seam to bleed into the readable region via the x2 unshuffle conv).
    let (data, cw, ch) = crop_chw_const_gradient(3, 2);
    let (padded, pw, ph) = pad_chw_even(data.clone(), cw, ch);
    assert_eq!((pw, ph), (4, 2));
    assert_eq!(padded.len(), 3 * 4 * 2);
    for c in 0..3 {
        for y in 0..2 {
            // the appended column (x=3) equals the original last column (x=2)
            let last = data[c * ch * cw + y * cw + 2];
            let appended = padded[c * ph * pw + y * pw + 3];
            assert!((appended - last).abs() < 1e-6);
            // existing columns are preserved
            for x in 0..3 {
                assert!(
                    (padded[c * ph * pw + y * pw + x] - data[c * ch * cw + y * cw + x]).abs()
                        < 1e-6
                );
            }
        }
    }

    // odd height too → row replicated; even×even is returned untouched (no copy).
    let (d2, _, _) = crop_chw_const_gradient(2, 3);
    let (_, pw2, ph2) = pad_chw_even(d2, 2, 3);
    assert_eq!((pw2, ph2), (2, 4));
    let (d3, _, _) = crop_chw_const_gradient(4, 2);
    let (out3, pw3, ph3) = pad_chw_even(d3.clone(), 4, 2);
    assert_eq!((pw3, ph3), (4, 2));
    assert_eq!(out3, d3, "already-even dims are returned unchanged");
}

fn crop_chw_const_gradient(cw: usize, ch: usize) -> (Vec<f32>, usize, usize) {
    let mut data = vec![0.0f32; 3 * cw * ch];
    for c in 0..3 {
        for y in 0..ch {
            for x in 0..cw {
                data[c * ch * cw + y * cw + x] = (c * 100 + y * 10 + x) as f32 / 255.0;
            }
        }
    }
    (data, cw, ch)
}

#[test]
fn validate_upscale_output_rejects_mismatched_shapes() {
    // Input tile: 8×8, factor 4 → a valid export returns (1, 3, 32, 32) with 3·32·32 values.
    let (cw, ch, factor) = (8usize, 8usize, 4usize);
    let good_len = 3 * 32 * 32;

    // Happy path: exact contract → Ok((32, 32)).
    let ok = validate_upscale_output(&[1, 3, 32, 32], good_len, cw, ch, factor)
        .expect("valid (1,3,32,32) output must pass");
    assert_eq!(ok, (32, 32));

    // Rank != 4 → Engine (would have panicked on `oshape[2]`/`oshape[3]`).
    let rank = validate_upscale_output(&[1, 3, 32], good_len, cw, ch, factor);
    assert!(
        matches!(rank, Err(Error::Engine(ref m)) if m.contains("rank") && m.contains("!= 4")),
        "rank<4 must be an Engine error, got {rank:?}"
    );

    // Dims not input·factor (an x2 export: 16×16 instead of 32×32) → Engine with expected-vs-actual.
    let scale = validate_upscale_output(&[1, 3, 16, 16], 3 * 16 * 16, cw, ch, factor);
    assert!(
        matches!(scale, Err(Error::Engine(ref m))
            if m.contains("16×16") && m.contains("32×32") && m.contains("factor 4")),
        "a half-scale export must be an Engine error naming expected vs actual, got {scale:?}"
    );

    // Right dims but a short `odata` buffer → Engine (would have sliced OOB on `odata[…]`).
    let short = validate_upscale_output(&[1, 3, 32, 32], good_len - 1, cw, ch, factor);
    assert!(
        matches!(short, Err(Error::Engine(ref m)) if m.contains("needs") && m.contains(&good_len.to_string())),
        "a short output buffer must be an Engine error, got {short:?}"
    );
}
