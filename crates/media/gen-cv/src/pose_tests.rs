use super::*;
fn approx(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}
#[test]
fn wholebody_to_openpose_maps_indices_and_computes_neck() {
    // 133 keypoints; set shoulders (coco 5,6) and nose (0) to known values.
    let mut kps = vec![[0.0f32; 3]; 133];
    kps[0] = [100.0, 10.0, 5.0]; // nose -> op 0
    kps[5] = [80.0, 40.0, 4.0]; // l_sho -> op 5
    kps[6] = [120.0, 50.0, 6.0]; // r_sho -> op 6
    let rec = wholebody_to_openpose(&kps, 200.0, 100.0).expect("133 keypoints convert");
    // nose normalized
    assert!(approx(rec.keypoints[0][0], 0.5, 1e-6));
    assert!(approx(rec.keypoints[0][1], 0.1, 1e-6));
    // neck (op 1) = midpoint of shoulders, conf = min(4,6) = 4
    assert!(approx(
        rec.keypoints[1][0],
        (80.0 + 120.0) / 2.0 / 200.0,
        1e-6
    ));
    assert!(approx(
        rec.keypoints[1][1],
        (40.0 + 50.0) / 2.0 / 100.0,
        1e-6
    ));
    assert!(approx(rec.keypoints[1][2], 4.0, 1e-6));
    // shapes
    assert_eq!(rec.keypoints.len(), 18);
    assert_eq!(rec.hands[0].len(), 21);
    assert_eq!(rec.hands[1].len(), 21);
    assert_eq!(rec.face.len(), 68);
}

#[test]
fn squareify_pads_short_axis_centered() {
    // landscape 200x100 -> square side 200; y padded by (200-100)/2 = 50.
    let rec = PoseRecord {
        keypoints: vec![[0.5, 0.5, 1.0]; 18],
        hands: [vec![[0.5, 0.5, 1.0]; 21], vec![[0.5, 0.5, 1.0]; 21]],
        face: vec![[0.5, 0.5, 1.0]; 68],
    };
    let sq = squareify(&rec, 200.0, 100.0);
    // x: (0.5*200 + 0)/200 = 0.5 ; y: (0.5*100 + 50)/200 = 0.5  (center stays center)
    assert!(approx(sq.keypoints[0][0], 0.5, 1e-6));
    assert!(approx(sq.keypoints[0][1], 0.5, 1e-6));
    // a top point y=0 -> (0 + 50)/200 = 0.25 (letterboxed inward)
    let rec2 = PoseRecord {
        keypoints: vec![[0.5, 0.0, 1.0]; 18],
        hands: [vec![[0.0, 0.0, 0.0]; 21], vec![[0.0, 0.0, 0.0]; 21]],
        face: vec![[0.0, 0.0, 0.0]; 68],
    };
    let sq2 = squareify(&rec2, 200.0, 100.0);
    assert!(approx(sq2.keypoints[0][1], 0.25, 1e-6));
}

#[test]
fn yolox_decode_embedded_nms_branch() {
    // dets (1, 2, 5): one above 0.3, one below; ratio 0.5 -> boxes *2.
    let dets = vec![
        10.0, 20.0, 30.0, 40.0, 0.9, // keep
        1.0, 2.0, 3.0, 4.0, 0.1, // drop (score <= 0.3)
    ];
    let boxes = yolox_decode(&dets, &[1, 2, 5], 0.5).expect("valid (N,5) decodes");
    assert_eq!(boxes.len(), 1);
    assert!(approx(boxes[0].x1, 20.0, 1e-6));
    assert!(approx(boxes[0].y2, 80.0, 1e-6));
    // A non-NMS head is a mis-pinned export, not a valid "no people" response.
    assert!(matches!(
        yolox_decode(&[0.0; 85], &[1, 1, 85], 1.0),
        Err(Error::Engine(_))
    ));
}

#[test]
fn yolox_decode_rejects_truncated_output() {
    // Last dim is 5 (the embedded-NMS branch) but the data buffer is shorter than N*5:
    // (1,2,5) needs 10 values, hand it 5 → Engine error instead of an OOB panic (F-102).
    let short = yolox_decode(&[0.0; 5], &[1, 2, 5], 1.0);
    // `Box4` isn't `Debug`, so match the error shape directly rather than formatting.
    assert!(
        matches!(short, Err(Error::Engine(_))),
        "truncated yolox buffer must be rejected"
    );
}

#[test]
fn pose_decode_argmax_and_rescale() {
    // 1 keypoint, Wx=4, Wy=4 ; peak at x-bin 2, y-bin 2.
    let k = 1usize;
    let wx = 4usize;
    let wy = 4usize;
    let mut sx = vec![0.0f32; k * wx];
    let mut sy = vec![0.0f32; k * wy];
    sx[2] = 9.0;
    sy[2] = 7.0;
    // crop geometry: center (50,60), scale (PW, PH) so px = (loc/2)/PW*PW + x0.
    let (cx, cy, sw, sh) = (50.0f32, 60.0f32, PW as f32, PH as f32);
    let kp = pose_decode(
        &sx,
        &[1, k as i64, wx as i64],
        &sy,
        &[1, k as i64, wy as i64],
        (cx, cy, sw, sh),
    )
    .expect("valid simcc decodes");
    assert_eq!(kp.len(), 1);
    // x: loc 2 -> 2/2=1 ; (1)/PW*sw + (cx - sw/2) = 1 + (50 - PW/2)
    let x0 = cx - sw / 2.0;
    let y0 = cy - sh / 2.0;
    assert!(approx(kp[0][0], 1.0 + x0, 1e-3));
    assert!(approx(kp[0][1], 1.0 + y0, 1e-3));
    // score = 0.5*(9+7) = 8
    assert!(approx(kp[0][2], 8.0, 1e-6));
}

#[test]
fn pose_decode_negative_value_marks_missing() {
    let sx = vec![-1.0f32; 4];
    let sy = vec![-1.0f32; 4];
    let kp = pose_decode(
        &sx,
        &[1, 1, 4],
        &sy,
        &[1, 1, 4],
        (50.0, 60.0, PW as f32, PH as f32),
    )
    .expect("valid simcc decodes");
    // val <= 0 -> loc (-1,-1) -> negative rescaled coords, score <= 0
    assert!(kp[0][2] <= 0.0);
}

#[test]
fn pose_decode_rejects_malformed_simcc() {
    // Rank < 3 simcc_x shape → Engine error, not an OOB panic on sx_shape[2] (F-102).
    let rank2 = pose_decode(
        &[0.0; 4],
        &[1, 4],
        &[0.0; 4],
        &[1, 1, 4],
        (0.0, 0.0, 1.0, 1.0),
    );
    assert!(
        matches!(rank2, Err(Error::Engine(_))),
        "rank-2 simcc_x must be rejected, got {rank2:?}"
    );
    // Well-formed rank but simcc_x is shorter than K*Wx → Engine error.
    let short = pose_decode(
        &[0.0; 2],
        &[1, 2, 4],
        &[0.0; 8],
        &[1, 2, 4],
        (0.0, 0.0, 1.0, 1.0),
    );
    assert!(
        matches!(short, Err(Error::Engine(_))),
        "truncated simcc_x must be rejected, got {short:?}"
    );
    // A well-sized Y buffer cannot override the reported shape. A mismatched K or a truncated
    // buffer relative to Wy must fail rather than manufacturing Wy from len/K.
    let mismatched_k = pose_decode(
        &[0.0; 8],
        &[1, 2, 4],
        &[0.0; 12],
        &[1, 3, 4],
        (0.0, 0.0, 1.0, 1.0),
    );
    assert!(matches!(mismatched_k, Err(Error::Engine(_))));
    let short_y = pose_decode(
        &[0.0; 8],
        &[1, 2, 4],
        &[0.0; 6],
        &[1, 2, 4],
        (0.0, 0.0, 1.0, 1.0),
    );
    assert!(matches!(short_y, Err(Error::Engine(_))));
}

#[test]
fn wholebody_to_openpose_rejects_short_keypoint_set() {
    // Fewer than 133 keypoints (a mis-pinned RTMW export) → Engine error, not an OOB panic
    // on the body/hand/face slices (F-102).
    let short = vec![[0.0f32; 3]; 100];
    let result = wholebody_to_openpose(&short, 200.0, 100.0);
    // `PoseRecord` isn't `Debug`, so match the error shape directly rather than formatting.
    assert!(
        matches!(result, Err(Error::Engine(msg)) if msg.contains("133")),
        "100-keypoint buffer must be rejected with a 133-keypoint Engine error"
    );
}
