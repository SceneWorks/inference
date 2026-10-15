//! sc-26214 Metal device gate. `#[ignore]`d: each test needs an actual Metal device and the real
//! Veda predictor file at `VEDA_PREDICTOR`; the `veda-mlx` dispatch of `real-weights.yml` runs
//! them on nax-macos-2 (`veda::device_gate::`).
//!
//! The mask checks never compare the runtime against itself: their oracle tiles each head's Q/K
//! on the host from the native rows, then calls only gen-core's frozen host contract
//! (`pool_video_tiles`, `tile_logits`, `select_with_budgets`) one head at a time.
use super::*;
use crate::denoise::{JointGeometry, TEXT_TAG};
use crate::dit::positions::{KeyframeAnchor, ReferenceLatentGeometry};
use crate::reference::ReferenceKind;
use std::time::Instant;

const LAYERS: usize = 50;
const HEADS: usize = 56;
const DIM: usize = 128;

fn bundle() -> Arc<Bundle> {
    let path = std::env::var("VEDA_PREDICTOR")
        .expect("VEDA_PREDICTOR must name the Veda predictor .safetensors file");
    Bundle::load_cached(Path::new(&path), LAYERS, HEADS, DIM).expect("real Veda predictor")
}

fn gpu() {
    mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
    assert!(on_gpu().unwrap(), "the device gate must run on Metal");
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    T2va,
    Fl2va,
    R2va,
}

/// Production 1344x768 / 124-frame layouts. Text rows are sized so each packed sequence matches
/// the one the 2026-10-14 real-weight A/B executed (`denseTokenPairs / (heads * calls)`): 37,742
/// (t2va), 41,778 (fl2va, both keyframes) and 52,076 (r2va, one image reference).
fn production(mode: Mode) -> PackedLayout {
    let geometry = JointGeometry::new(124, 48, 84).unwrap();
    let build = |text: usize| {
        match mode {
            Mode::T2va => PackedLayout::build(geometry, [1, 2, 2], &vec![TEXT_TAG; text], 2, &[]),
            Mode::Fl2va => PackedLayout::build(
                geometry,
                [1, 2, 2],
                &vec![TEXT_TAG; text],
                2,
                &[KeyframeAnchor::First, KeyframeAnchor::Last],
            ),
            Mode::R2va => PackedLayout::build_ref2va(
                geometry,
                [1, 2, 2],
                &vec![TEXT_TAG; text],
                2,
                &[ReferenceLatentGeometry {
                    kind: ReferenceKind::Image,
                    num_latent_frames: 1,
                    latent_height: 48,
                    latent_width: 84,
                    num_audio_latents: 0,
                }],
            ),
        }
        .unwrap()
    };
    let target = match mode {
        Mode::T2va => 37_742,
        Mode::Fl2va => 41_778,
        Mode::R2va => 52_076,
    };
    let base = build(1).seq_len() as usize - 1;
    let packed = build(target - base);
    assert_eq!(packed.seq_len() as usize, target);
    packed
}

fn request() -> VedaRequest {
    VedaRequest {
        enabled: true,
        ..Default::default()
    }
}

fn runtime(bundle: Arc<Bundle>, packed: &PackedLayout) -> VedaRuntime {
    VedaRuntime::from_bundle(bundle, &request(), packed, ReportSink::default(), || Ok(()))
        .expect("Veda runtime")
}

/// `[1, heads, seq, dim]` BF16 views of native `[1, seq, heads, dim]` rows, as the DiT passes them.
fn qkv(seq: usize, seed: u64) -> [Array; 3] {
    let make = |i: u64| {
        let key = mlx_rs::random::key(seed * 3 + i).unwrap();
        let x = mlx_rs::random::normal::<f32>(
            &[1, seq as i32, HEADS as i32, DIM as i32][..],
            None,
            None,
            &key,
        )
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        mlx_rs::transforms::eval([&x]).unwrap();
        x.transpose_axes(&[0, 2, 1, 3]).unwrap()
    };
    [make(0), make(1), make(2)]
}

/// One head's native rows `[seq, dim]` as host f32.
fn head_rows(x: &Array, head: usize) -> Vec<f32> {
    let h = x
        .take_axis(Array::from_slice(&[head as i32], &[1]), 1)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[-1])
        .unwrap();
    mlx_rs::transforms::eval([&h]).unwrap();
    h.as_slice::<f32>().to_vec()
}

/// Host tile order `[1, tiles, 128, dim]`, padding zero — built without any runtime code.
fn host_tiles(rows: &[f32], layout: &Layout) -> Vec<f32> {
    let mut out = vec![0f32; layout.perm.len() * DIM];
    for (slot, &row) in layout.perm.iter().enumerate() {
        if row >= 0 {
            out[slot * DIM..(slot + 1) * DIM]
                .copy_from_slice(&rows[row as usize * DIM..(row as usize + 1) * DIM]);
        }
    }
    out
}

/// The independent per-head oracle: host contract scores and frozen selection of head `head`.
fn oracle(
    bundle: &Bundle,
    layer: usize,
    head: usize,
    layout: &Layout,
    q: &Array,
    k: &Array,
) -> (Vec<f32>, Vec<Vec<usize>>) {
    let pool = |x: &Array| {
        contract::pool_video_tiles(&host_tiles(&head_rows(x, head), layout), layout, 1, DIM)
            .unwrap()
    };
    let ids = [head as i32];
    let scores = contract::tile_logits(
        &pool(q),
        &pool(k),
        &bundle.projection(layer, &ids, true),
        &bundle.projection(layer, &ids, false),
        1,
        layout.video,
        DIM,
    )
    .unwrap();
    let r = request();
    let rows = layout
        .select_with_budgets(&scores, 1, budget(&r.generated), budget(&r.reference))
        .unwrap();
    (scores, rows)
}

fn run_layer(rt: &VedaRuntime, layer: usize, [q, k, v]: &[Array; 3]) -> Array {
    let out = rt
        .attention(layer, 0, q, k, v, BoundedAttention::UNBOUNDED)
        .unwrap();
    mlx_rs::transforms::eval([&out]).unwrap();
    let r = rt.report.borrow();
    assert!(r.fallback_reason.is_none(), "{:?}", r.fallback_reason);
    assert_eq!(r.kernel.as_deref(), Some(kernel_id().unwrap()));
    out
}

/// A plan whose heads alternate over three tile shapes, so every batch is a strict subset of the
/// layer's heads and three layouts are live at once.
fn mixed(bundle: &Bundle, grid: [usize; 3]) -> Arc<Bundle> {
    let shapes = vec![
        TileShape([2, 8, 8]),
        TileShape([1, 8, 16]),
        TileShape([4, 4, 8]),
    ];
    Arc::new(Bundle {
        identity: "sc-26214-mixed-shape".into(),
        layers: bundle.layers,
        heads: bundle.heads,
        dim: bundle.dim,
        plans: vec![Plan {
            name: "mixed".into(),
            grid,
            shapes,
            heads: vec![(0..HEADS).map(|h| h % 3).collect(); LAYERS],
            provenance: None,
        }],
        q: bundle.q.clone(),
        k: bundle.k.clone(),
    })
}

/// AC: on Metal the runtime's selected masks equal the independent host oracle bit for bit, for
/// t2va, fl2va (keyframe reference spans) and a mixed-shape plan, at every batch size: the
/// default batches, and batches cut to three heads by a copy allowance of three minimum heads.
#[test]
#[ignore = "requires a Metal device and VEDA_PREDICTOR"]
fn runtime_masks_equal_the_host_oracle_bit_for_bit() {
    gpu();
    let bundle = bundle();
    let layer = 7;
    for (name, packed, plan_bundle) in [
        ("t2va", production(Mode::T2va), bundle.clone()),
        ("fl2va", production(Mode::Fl2va), bundle.clone()),
        (
            "mixed-t2va",
            production(Mode::T2va),
            mixed(&bundle, [37, 24, 42]),
        ),
    ] {
        let seq = packed.seq_len() as usize;
        let inputs = qkv(seq, 26214);
        let mut by_batch = Vec::new();
        for batch in ["default", "three-heads"] {
            let mut rt = runtime(plan_bundle.clone(), &packed);
            if batch == "three-heads" {
                let sizes = rt
                    .layouts
                    .iter()
                    .map(|(&s, l)| (s, LayoutSize::from(l)))
                    .collect();
                let b = workspace::bound(
                    &rt.request,
                    Backend::Mlx,
                    &rt.plan,
                    rt.grid,
                    &sizes,
                    LAYERS,
                    HEADS,
                    DIM,
                    1,
                    2,
                    rt.limits,
                )
                .unwrap();
                let min = b
                    .chunks
                    .values()
                    .map(|c| c.minimum_copies_bytes)
                    .max()
                    .unwrap();
                rt.limits.copies = 3 * min;
            }
            run_layer(&rt, layer, &inputs);
            let masks = rt.masks.borrow().clone();
            assert_eq!(masks.len(), HEADS, "{name}/{batch}: one mask per head");
            let calls = rt.report.borrow().predictor_score_calls;
            if batch == "three-heads" {
                assert!(
                    calls >= HEADS as u64 / 3,
                    "{name}: batches were not cut ({calls})"
                );
            }
            println!(
                "VEDA_GATE masks mode={name} batch={batch} score_calls={calls} plan={}",
                rt.plan.name
            );
            by_batch.push(masks);
        }
        assert_eq!(
            by_batch[0]
                .iter()
                .map(|m| m.1)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            HEADS
        );
        let sorted = |m: &Vec<ExecutedMask>| {
            let mut m = m.clone();
            m.sort_by_key(|x| x.1);
            m
        };
        let (a, b) = (sorted(&by_batch[0]), sorted(&by_batch[1]));
        assert_eq!(a, b, "{name}: masks depend on the batch size");
        let rt = runtime(plan_bundle.clone(), &packed);
        let mut checked = 0;
        for (l, head, rows) in &a {
            let shape = rt.plan.shapes[rt.plan.heads[*l][*head]];
            let (_, want) = oracle(
                &plan_bundle,
                *l,
                *head,
                &rt.layouts[&shape],
                &inputs[0],
                &inputs[1],
            );
            assert_eq!(
                rows, &want,
                "{name}: head {head} mask differs from the host oracle"
            );
            checked += 1;
        }
        println!("VEDA_GATE masks mode={name} heads_checked={checked} result=exact");
    }
}

/// The Metal scorer's logits equal the host contract's bit for bit (not only their masks), for
/// one head alone and inside a batch of every head.
#[test]
#[ignore = "requires a Metal device and VEDA_PREDICTOR"]
fn metal_scorer_logits_are_the_host_contract_bit_for_bit() {
    gpu();
    let bundle = bundle();
    for mode in [Mode::T2va, Mode::Fl2va, Mode::R2va] {
        let packed = production(mode);
        let seq = packed.seq_len() as usize;
        let [q, k, _] = qkv(seq, 7);
        let rt = runtime(bundle.clone(), &packed);
        let layer = 31;
        let (pq, pk) = rt.projection(layer).unwrap();
        let native = |a: &Array| {
            a.transpose_axes(&[0, 2, 1, 3])
                .unwrap()
                .reshape(&[seq as i32, HEADS as i32, DIM as i32])
                .unwrap()
        };
        let (nq, nk) = (native(&q), native(&k));
        let shape = rt.plan.shapes[0];
        let layout = &rt.layouts[&shape];
        let all: Vec<i32> = (0..HEADS as i32).collect();
        let batched = metal::scores(&rt.device[&shape], &nq, &nk, &pq, &pk, &all, 0, DIM).unwrap();
        let per = layout.video * layout.video;
        for head in [0usize, 13, 55] {
            let alone = metal::scores(
                &rt.device[&shape],
                &nq,
                &nk,
                &pq,
                &pk,
                &[head as i32],
                0,
                DIM,
            )
            .unwrap();
            let (want, _) = oracle(&bundle, layer, head, layout, &q, &k);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&alone),
                bits(&want),
                "{mode:?} head {head}: single-head logits"
            );
            assert_eq!(
                bits(&batched[head * per..(head + 1) * per]),
                bits(&want),
                "{mode:?} head {head}: batched logits"
            );
        }
        println!(
            "VEDA_GATE logits mode={mode:?} video_tiles={} result=bit-exact",
            layout.video
        );
    }
}

/// The frozen upstream capture fixtures select their recorded masks through the Metal scorer.
#[test]
#[ignore = "requires a Metal device"]
fn frozen_upstream_device_scores_select_exact_masks() {
    gpu();
    let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/gen-core/tests/fixtures/veda/frozen.json"
    )))
    .unwrap();
    let uv = |v: &serde_json::Value| {
        v.as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap() as usize)
            .collect::<Vec<_>>()
    };
    let fv = |v: &serde_json::Value| {
        v.as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_f64().unwrap() as f32)
            .collect::<Vec<_>>()
    };
    let mut cases = 0;
    for c in fixture["scorers"].as_array().unwrap() {
        let spans = c["layout"]["spans"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| contract::TiledSpan {
                start: s["start"].as_u64().unwrap() as usize,
                grid: uv(&s["grid"]).try_into().unwrap(),
                shape: TileShape(uv(&s["shape"]).try_into().unwrap()),
            })
            .collect::<Vec<_>>();
        let seq = c["layout"]["seq"].as_u64().unwrap() as usize;
        let layout = Layout::from_spans(&spans, seq).unwrap();
        let heads = c["heads"].as_u64().unwrap() as usize;
        let dim = c["dim"].as_u64().unwrap() as usize;
        let pq = c["proj"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as u16)
            .collect::<Vec<_>>();
        let mut pk = vec![0; pq.len()];
        for h in 0..heads {
            for i in 0..3 * dim {
                for d in 0..dim {
                    pk[(h * 3 * dim + i) * dim + d] = pq[(h * 3 * dim + 3 * dim - 1 - i) * dim + d];
                }
            }
        }
        let bf = |p: &[u16]| {
            let f: Vec<f32> = p
                .iter()
                .map(|&b| f32::from_bits((b as u32) << 16))
                .collect();
            Array::from_slice(&f, &[heads as i32, 3 * dim as i32, dim as i32])
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
        };
        // The fixture's operands are in tile order; scatter them back to native rows.
        let native = |tiled: Vec<f32>| {
            let mut rows = vec![0f32; seq * heads * dim];
            for h in 0..heads {
                for (slot, &row) in layout.perm.iter().enumerate() {
                    if row >= 0 {
                        for d in 0..dim {
                            rows[(row as usize * heads + h) * dim + d] =
                                tiled[(h * layout.perm.len() + slot) * dim + d];
                        }
                    }
                }
            }
            Array::from_slice(&rows, &[seq as i32, heads as i32, dim as i32])
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
        };
        let ids: Vec<i32> = (0..heads as i32).collect();
        let scores = metal::scores(
            &metal::DeviceLayout::new(&layout),
            &native(fv(&c["q"])),
            &native(fv(&c["k"])),
            &bf(&pq),
            &bf(&pk),
            &ids,
            0,
            dim,
        )
        .unwrap();
        let masks = layout
            .select_with_budgets(&scores, heads, Budget::Tiles(2.3), Budget::Tiles(1.5))
            .unwrap();
        let expected = c["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(&uv)
            .collect::<Vec<_>>();
        assert_eq!(
            masks, expected,
            "device logits must select frozen masks exactly"
        );
        cases += 1;
    }
    assert!(cases > 0);
    println!("VEDA_GATE frozen_fixtures cases={cases} result=exact");
}

/// The Metal sparse kernel against the host masked-attention oracle and the CPU gather route,
/// on layouts with partial tiles, a tiled reference span, dense global rows and non-trivial
/// selections — plus the production first-use self-test on this device.
#[test]
#[ignore = "requires a Metal device"]
fn metal_sparse_attention_matches_the_host_oracle() {
    gpu();
    device_self_test().unwrap();
    let cases = [
        // (spans, seq): globals before, a partial reference span, a partial target grid.
        (
            vec![contract::TiledSpan {
                start: 17,
                grid: [2, 9, 17],
                shape: TileShape([2, 8, 8]),
            }],
            323,
        ),
        (
            vec![
                contract::TiledSpan {
                    start: 5,
                    grid: [1, 9, 13],
                    shape: TileShape([1, 8, 16]),
                },
                contract::TiledSpan {
                    start: 130,
                    grid: [3, 10, 18],
                    shape: TileShape([2, 8, 8]),
                },
            ],
            700,
        ),
    ];
    for (spans, seq) in cases {
        let l = Layout::from_spans(&spans, seq).unwrap();
        let heads = 3usize;
        let drive = |offset: usize| {
            let values: Vec<f32> = (0..heads * seq * DIM)
                .map(|i| ((i * 31 + offset) % 127) as f32 / 127.0 - 0.5)
                .collect();
            Array::from_slice(&values, &[1, heads as i32, seq as i32, DIM as i32])
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
        };
        let (q, k, v) = (drive(1), drive(5), drive(11));
        // A different selection per head and row, always including the diagonal and globals.
        let mut rows = Vec::new();
        for h in 0..heads {
            for r in 0..l.valid.len() {
                rows.push(if r < l.video {
                    let mut keys: Vec<usize> = (0..l.video)
                        .filter(|&c| c == r || (c * 7 + r + h) % 3 == 0)
                        .collect();
                    keys.extend(l.video..l.valid.len());
                    keys
                } else {
                    (0..l.valid.len()).collect()
                });
            }
        }
        let ids: Vec<i32> = (0..heads as i32).collect();
        let native = |a: &Array| {
            a.transpose_axes(&[0, 2, 1, 3])
                .unwrap()
                .reshape(&[seq as i32, heads as i32, DIM as i32])
                .unwrap()
        };
        let got = metal_batch(
            &metal::DeviceLayout::new(&l),
            &rows,
            [&q, &k, &v],
            DIM,
            BoundedAttention::UNBOUNDED,
        )
        .unwrap();
        let host = |a: &Array| {
            let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[-1]).unwrap();
            mlx_rs::transforms::eval([&a]).unwrap();
            a.as_slice::<f32>().to_vec()
        };
        let mut mask = vec![false; heads * seq * seq];
        for head in 0..heads {
            for row in 0..l.valid.len() {
                for qi in 0..l.valid[row] {
                    let qr = l.perm[row * TILE + qi] as usize;
                    for &kt in &rows[head * l.valid.len() + row] {
                        for ki in 0..l.valid[kt] {
                            mask[(head * seq + qr) * seq + l.perm[kt * TILE + ki] as usize] = true;
                        }
                    }
                }
            }
        }
        let want = host_masked_attention(
            &host(&q),
            &host(&k),
            &host(&v),
            &mask,
            heads,
            seq,
            DIM,
            1.0 / (DIM as f32).sqrt(),
        );
        let got = host(&got);
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "VEDA_GATE attend seq={seq} tiles={} max_abs={err} tolerance={SELF_TEST_ATOL}",
            l.valid.len()
        );
        assert!(err.is_finite() && err <= SELF_TEST_ATOL, "{err}");
        // Both Metal kernels on their video query rows, whichever one this device routes to.
        let routes: &[bool] = if metal::nax_available() {
            &[true, false]
        } else {
            &[false]
        };
        for &nax in routes {
            let kmax = rows.iter().map(Vec::len).max().unwrap();
            let tiles = l.valid.len();
            // Every head at once, and the last head alone (a strict subset written by head id
            // into the layer-sized output, its key lists at list position 0).
            let all = metal::attend_route(
                &metal::DeviceLayout::new(&l),
                &native(&q),
                &native(&k),
                &native(&v),
                &ids,
                &rows,
                kmax,
                nax,
            )
            .unwrap();
            let last = (heads - 1) as i32;
            let one = metal::attend_route(
                &metal::DeviceLayout::new(&l),
                &native(&q),
                &native(&k),
                &native(&v),
                &[last],
                &rows[(heads - 1) * tiles..],
                kmax,
                nax,
            )
            .unwrap();
            let one = host(&one);
            let tiled = host(&all);
            let span = l.video * TILE * DIM;
            assert_eq!(
                &one[(heads - 1) * span..heads * span],
                &tiled[(heads - 1) * span..heads * span],
                "nax={nax}: a single-head launch differs from the same head in an all-heads launch"
            );
            let mut worst = 0f32;
            for h in 0..heads {
                for slot in 0..l.video * TILE {
                    let row = l.perm[slot];
                    for d in 0..DIM {
                        let got = tiled[(h * l.video * TILE + slot) * DIM + d];
                        if row < 0 {
                            assert_eq!(got, 0.0, "padding rows are zero");
                            continue;
                        }
                        worst = worst.max((got - want[(h * seq + row as usize) * DIM + d]).abs());
                    }
                }
            }
            println!("VEDA_GATE attend_route nax={nax} seq={seq} max_abs={worst}");
            assert!(
                worst.is_finite() && worst <= SELF_TEST_ATOL,
                "nax={nax}: {worst}"
            );
        }
    }
}

/// The non-finite guard on Metal: a NaN sparse output is recomputed densely and reported.
#[test]
#[ignore = "requires a Metal device and VEDA_PREDICTOR"]
fn metal_non_finite_output_falls_back_to_dense_and_is_reported() {
    gpu();
    let packed = production(Mode::T2va);
    let seq = packed.seq_len() as usize;
    let [q, k, _] = qkv(seq, 3);
    let poisoned = k
        .multiply(Array::from_f32(f32::NAN))
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let rt = runtime(bundle(), &packed);
    rt.attention(2, 0, &q, &k, &poisoned, BoundedAttention::UNBOUNDED)
        .unwrap();
    let r = rt.report.borrow();
    assert_eq!((r.sparse_calls, r.dense_calls), (0, 1));
    assert!(
        r.fallback_reason
            .as_deref()
            .is_some_and(|e| e.contains("non-finite output")),
        "{:?}",
        r.fallback_reason
    );
}

/// Per-layer wall time of one sparse Veda attention call against the dense SDPA it replaces, on
/// the production layouts (real predictor, real plans, synthetic Q/K/V). Prints one
/// `VEDA_BENCH {json}` line per mode. Report-only.
#[test]
#[ignore = "requires a Metal device and VEDA_PREDICTOR"]
fn per_layer_bench() {
    gpu();
    let bundle = bundle();
    let reps = 3;
    for mode in [Mode::T2va, Mode::Fl2va, Mode::R2va] {
        let packed = production(mode);
        let seq = packed.seq_len() as usize;
        let inputs = qkv(seq, 11);
        let [q, k, v] = &inputs;
        let time = |f: &dyn Fn() -> Array| {
            let _ = f();
            let mut best = f64::MAX;
            for _ in 0..reps {
                let t = Instant::now();
                let out = f();
                mlx_rs::transforms::eval([&out]).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let dense = time(&|| {
            sdpa_bounded_bhsd(
                q,
                k,
                v,
                1.0 / (DIM as f32).sqrt(),
                None,
                BoundedAttention::UNBOUNDED,
            )
            .unwrap()
        });
        let rt = runtime(bundle.clone(), &packed);
        let sparse = time(&|| run_layer(&rt, 3, &inputs));
        let r = rt.report.borrow();
        let fraction = r.selected_token_pairs as f64 / r.dense_token_pairs as f64;
        let calls = r.predictor_score_calls as f64 / r.sparse_calls as f64;
        let profile = rt
            .profile
            .borrow()
            .iter()
            .map(|(k, v)| (k.to_string(), v / r.sparse_calls as f64))
            .collect::<BTreeMap<_, _>>();
        println!(
            "VEDA_BENCH {}",
            serde_json::json!({
                "mode": format!("{mode:?}"), "seq": seq, "dense_s": dense, "sparse_s": sparse,
                "speedup": dense / sparse, "attention_fraction": fraction,
                "batches_per_call": calls, "profile_s_per_call": profile,
                "workspace_bytes": r.workspace_bytes,
            })
        );
    }
}

/// E8: the first-use self-test's actual MLX allocator peak fits its declared bound.
#[test]
#[ignore = "requires a Metal device"]
fn full_self_test_lifetime_fits_declared_allocator_bound() {
    gpu();
    let layout = Layout::t2va(17, [2, 9, 17], TileShape([2, 8, 8]), 323).unwrap();
    let shape = TileShape([2, 8, 8]);
    let plan = Plan {
        name: "allocator-fixture".into(),
        grid: [2, 9, 17],
        shapes: vec![shape],
        heads: vec![vec![0, 0]],
        provenance: None,
    };
    let bound = workspace::bound(
        &VedaRequest {
            enabled: true,
            ..Default::default()
        },
        Backend::Mlx,
        &plan,
        plan.grid,
        &BTreeMap::from([(shape, LayoutSize::from(&layout))]),
        1,
        2,
        128,
        1,
        2,
        Limits::for_backend(Backend::Mlx),
    )
    .unwrap();
    mlx_rs::memory::clear_cache();
    let baseline = mlx_rs::memory::get_active_memory() as u64;
    mlx_rs::memory::reset_peak_memory();
    device_self_test().unwrap();
    let peak = (mlx_rs::memory::get_peak_memory() as u64).saturating_sub(baseline);
    assert!(
        peak <= bound.self_test_bytes,
        "self-test owned peak {peak} > {}",
        bound.self_test_bytes
    );
    assert!(peak <= bound.total_bytes);
    println!("Veda allocator receipt: self_test_peak_bytes={peak}, self_test_bound_bytes={}, total_lifetime_bound_bytes={}",bound.self_test_bytes,bound.total_bytes);
}

/// Fifty consecutive sparse layers (one denoise step) at the fl2va production layout must reach a
/// steady allocator state: MLX's live bytes plus its buffer cache may not keep growing once the
/// first layers have warmed the cache. A route whose allocation sizes never repeat grows the cache
/// to MLX's GC limit (~100 GB on a 128 GB Mac) even with a small live peak.
#[test]
#[ignore = "requires a Metal device and VEDA_PREDICTOR"]
fn memory_reaches_a_steady_state_across_layers() {
    gpu();
    let packed = production(Mode::Fl2va);
    let inputs = qkv(packed.seq_len() as usize, 5);
    let rt = runtime(bundle(), &packed);
    let gb = |b: usize| b as f64 / 1e9;
    let mut held = Vec::new();
    for layer in 0..LAYERS {
        run_layer(&rt, layer, &inputs);
        let (active, cache) = (
            mlx_rs::memory::get_active_memory(),
            mlx_rs::memory::get_cache_memory(),
        );
        println!(
            "VEDA_MEM layer={layer} active_gb={:.2} cache_gb={:.2} peak_gb={:.2}",
            gb(active),
            gb(cache),
            gb(mlx_rs::memory::get_peak_memory())
        );
        held.push(active + cache);
    }
    let (warm, last) = (held[9], *held.last().unwrap());
    assert!(
        last <= warm + 2_000_000_000,
        "allocator held {:.2} GB after layer 9 but {:.2} GB after layer 49",
        gb(warm),
        gb(last)
    );
}
