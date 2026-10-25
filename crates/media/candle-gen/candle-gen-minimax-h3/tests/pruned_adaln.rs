use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::pruned_adaln::{PrunedAdaLn, PruningIdentity, TABLE_KEY};
use candle_gen::Weights;
use candle_gen_minimax_h3::dit::{
    AdaLayerNormOut, CurveAdaLn, DitBlock, LinearBias, MmRope, TimestepSchedule,
};
use candle_gen_minimax_h3::MiniMaxH3DitConfig;
use serde_json::Value;

/// The shared pinned oracle, generated from the reference sources and committed at the repository
/// root (not in this crate).
const FIXTURE: &str = include_str!("../../../../../tests/fixtures/h3_pruned_adaln.json");

const COORDS_ABS: f64 = 2e-6;
const PROJECTION_ABS: f64 = 2e-5;
const BLOCK_ABS: f64 = 2e-5;

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).expect("the shared pruned-AdaLN fixture parses")
}

fn floats(v: &Value) -> Vec<f32> {
    serde_json::from_value(v.clone()).expect("fixture floats")
}

fn config() -> MiniMaxH3DitConfig {
    // Every config key the oracle recorded is one `from_diffusers_json` consumes — including the
    // in_channels / audio_in_channels / patch_size the final heads are shaped by — so the config
    // is parsed from the fixture bytes rather than re-typed here and allowed to drift.
    let f = fixture();
    MiniMaxH3DitConfig::from_diffusers_json(&f["config"].to_string()).expect("fixture config")
}

/// One fixture tensor as a candle tensor **at its stored dtype**.
///
/// The JSON carries each stored tensor's values widened losslessly to f32; casting back to the
/// declared storage dtype reproduces the exact stored bytes — that cast is the fixture format's
/// contract, not a loader shortcut.
fn stored_tensor(name: &str, f: &Value, device: &Device) -> Tensor {
    let entry = &f["tensors"][name];
    let shape: Vec<usize> = entry["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d.as_u64().unwrap() as usize)
        .collect();
    let data = floats(&entry["data"]);
    let dtype = entry["dtype"].as_str().unwrap();
    let t = Tensor::from_vec(data, shape, device).expect(name);
    match dtype {
        "F32" => t,
        "BF16" => t.to_dtype(DType::BF16).expect(name),
        "F16" => t.to_dtype(DType::F16).expect(name),
        other => panic!("fixture tensor {name}: unsupported stored dtype {other}"),
    }
}

/// The whole canonical tensor set as a weight map, curve operands at their stored dtypes.
fn weight_map(f: &Value, device: &Device) -> HashMap<String, Tensor> {
    f["tensors"]
        .as_object()
        .unwrap()
        .keys()
        .map(|k| (k.clone(), stored_tensor(k, f, device)))
        .collect()
}

fn curve(f: &Value, device: &Device) -> CurveAdaLn {
    let cfg = config();
    let map = weight_map(f, device);
    let w = Weights::from_map(map);
    let topology = CurveAdaLn::inspect(&w, &cfg)
        .expect("the complete pruned topology must inspect cleanly")
        .expect("the fixture carries the curve table");
    let operands: HashMap<String, Tensor> = topology
        .operands()
        .iter()
        .map(|o| {
            (
                o.name.clone(),
                w.require(&o.name).expect("curve operand present"),
            )
        })
        .collect();
    CurveAdaLn::assemble(topology, operands, &cfg, device).expect("assembly")
}

fn dev() -> Device {
    Device::Cpu
}

fn assert_close(got: &[f32], want: &[f32], tol: f64, what: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: element count {} vs {}",
        got.len(),
        want.len()
    );
    assert!(
        got.iter().chain(want).all(|v| v.is_finite()),
        "{what}: nonfinite numeric result"
    );
    let peak = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() as f64)
        .fold(0.0f64, f64::max);
    assert!(
        peak <= tol,
        "{what}: peak |Δ| {peak:.3e} exceeds the pinned {tol:.3e}"
    );
}

// -------------------------------------------------------------------------------------------
// Refusal before execution — the fail-closed half of the contract
// -------------------------------------------------------------------------------------------

fn lookup_of(
    f: &Value,
    extra: &[(&str, &str, Vec<usize>)],
) -> impl Fn(&str) -> Option<(candle_gen::gen_core::weightsmeta::Dtype, Vec<usize>)> {
    let mut map: HashMap<String, (String, Vec<usize>)> = f["tensors"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                (
                    v["dtype"].as_str().unwrap().to_string(),
                    v["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|d| d.as_u64().unwrap() as usize)
                        .collect(),
                ),
            )
        })
        .collect();
    for (name, dtype, shape) in extra {
        map.insert(name.to_string(), (dtype.to_string(), shape.clone()));
    }
    move |name: &str| {
        map.get(name).map(|(d, s)| {
            let d = match d.as_str() {
                "F32" => candle_gen::gen_core::weightsmeta::Dtype::F32,
                "BF16" => candle_gen::gen_core::weightsmeta::Dtype::BF16,
                "F16" => candle_gen::gen_core::weightsmeta::Dtype::F16,
                "I8" => candle_gen::gen_core::weightsmeta::Dtype::I8,
                other => panic!("lookup dtype {other}"),
            };
            (d, s.clone())
        })
    }
}

fn inspect(f: &Value, extra: &[(&str, &str, Vec<usize>)]) -> Result<Option<PrunedAdaLn>, String> {
    let cfg = config();
    PrunedAdaLn::inspect(
        cfg.hidden_size,
        cfg.num_layers,
        cfg.time_embed_dim,
        lookup_of(f, extra),
    )
}

#[test]
fn a_missing_table_with_no_timestep_mlp_is_refused() {
    let mut f = fixture();
    f["tensors"].as_object_mut().unwrap().remove(TABLE_KEY);
    let err = inspect(&f, &[]).unwrap_err();
    assert!(
        err.contains(TABLE_KEY) && err.contains("time_embedder.linear_1.weight"),
        "the refusal must name both halves of the inconsistency: {err}"
    );
}

#[test]
fn a_reduced_projection_without_a_table_is_refused() {
    let mut without = fixture();
    // Drop the table but keep the reduced block projections: the reduced width is a statement
    // that the checkpoint was pruned, and nothing may execute it against a curve that is not
    // there.
    without["tensors"]
        .as_object_mut()
        .unwrap()
        .remove(TABLE_KEY);
    let err = inspect(&without, &[]).unwrap_err();
    assert!(
        err.contains("reduced width") || err.contains(TABLE_KEY),
        "{err}"
    );
}

#[test]
fn a_table_alongside_any_timestep_mlp_tensor_is_refused() {
    let f = fixture();
    let cfg = config();
    for name in [
        "time_embedder.linear_1.weight",
        "time_embedder.linear_2.bias",
    ] {
        let shape = match name {
            "time_embedder.linear_1.weight" => vec![cfg.time_embed_hidden_dim, cfg.freq_dim],
            _ => vec![cfg.time_embed_dim],
        };
        let err = inspect(&f, &[(name, "F32", shape)]).unwrap_err();
        assert!(
            err.contains("conflicts") && err.contains(name),
            "table + {name} must be refused: {err}"
        );
    }
}

#[test]
fn a_missing_folded_bias_or_wrong_width_is_refused() {
    let f = fixture();
    // A reduced weight whose width disagrees with the table's rank.
    let err = inspect(
        &f,
        &[(
            "transformer_blocks.0.adaln_proj.linear.weight",
            "BF16",
            vec![144, 7],
        )],
    )
    .unwrap_err();
    assert!(
        err.contains("transformer_blocks.0.adaln_proj.linear.weight")
            && err.contains("dense F16/BF16/F32"),
        "{err}"
    );
    // An operand dtype that is not a float curve operand (packed codes).
    let err = inspect(
        &f,
        &[(
            "transformer_blocks.1.adaln_proj.linear.bias",
            "I8",
            vec![144],
        )],
    )
    .unwrap_err();
    assert!(
        err.contains("transformer_blocks.1.adaln_proj.linear.bias"),
        "a packed-code curve operand is refused: {err}"
    );
    // A missing final operand.
    let mut without = f.clone();
    without["tensors"]
        .as_object_mut()
        .unwrap()
        .remove("norm_out.linear.bias");
    let err = inspect(&without, &[]).unwrap_err();
    assert!(
        err.contains("missing pruned operand norm_out.linear.bias"),
        "{err}"
    );
}

// -------------------------------------------------------------------------------------------
// Source-precision storage, identity, and the pinned arithmetic
// -------------------------------------------------------------------------------------------

#[test]
fn curve_operands_keep_their_stored_dtypes() {
    let f = fixture();
    let device = dev();
    let c = curve(&f, &device);
    assert_eq!(c.table().dtype(), DType::F32, "the table stays F32");
    assert_eq!(c.grid(), 1025);
    assert_eq!(c.rank(), 8);
    for layer in 0..config().num_layers {
        let p = c.block_projection(layer).unwrap();
        assert_eq!(
            p.source_dtype(),
            DType::BF16,
            "block {layer} source weight is BF16"
        );
        assert_eq!(p.compute_dtype(), DType::F32);
        assert_eq!(
            p.nbytes(),
            (18 * c.topology().hidden_size) * (c.rank() + 1) * 4
        );
    }
    let n = c.norm_out_projection().unwrap();
    assert_eq!(
        n.source_dtype(),
        DType::BF16,
        "the final source weight is BF16"
    );
    assert_eq!(n.compute_dtype(), DType::F32);
    let compute_elements: usize = c
        .topology()
        .operands()
        .iter()
        .map(|operand| operand.shape.iter().product::<usize>())
        .sum();
    assert_eq!(c.nbytes(), compute_elements * 4);
}

#[test]
fn interpolation_matches_the_pinned_lerp_on_every_case() {
    let f = fixture();
    let device = dev();
    let c = curve(&f, &device);
    for (i, case) in f["cases"].as_array().unwrap().iter().enumerate() {
        let timesteps = floats(&case["timesteps"]);
        let got = c.coordinates(&timesteps, &device).unwrap();
        assert_eq!(got.dtype(), DType::F32);
        assert_close(
            &got.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            &floats(&case["coordinates"]),
            COORDS_ABS,
            &format!("case {i} coordinates"),
        );
    }
}

#[test]
fn the_reduced_projections_match_the_pinned_modulation_without_silu() {
    let f = fixture();
    let device = dev();
    let c = curve(&f, &device);
    for (i, case) in f["cases"].as_array().unwrap().iter().enumerate() {
        let coords = c.coordinates(&floats(&case["timesteps"]), &device).unwrap();
        for (layer, want) in case["block_modulations"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let flat = c
                .block_projection(layer)
                .unwrap()
                .forward(&coords)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_close(
                &flat,
                &floats(want),
                PROJECTION_ABS,
                &format!("case {i} block {layer} modulation"),
            );
        }
        // The pinned curve path has NO activation: pushing the coordinates through SiLU before
        // the projection moves the modulation far beyond the tolerance, so a port that kept the
        // stock MLP's SiLU cannot pass this gate by tolerance luck.
        let activated = candle_gen_minimax_h3::nn::silu(&coords).unwrap();
        let silu_first = c.block_projection(0).unwrap().forward(&activated).unwrap();
        let peak = silu_first
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .zip(
                case["block_modulations"][0]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32),
            )
            .map(|(g, w)| (g - w).abs() as f64)
            .fold(0.0f64, f64::max);
        assert!(
            peak > PROJECTION_ABS * 10.0,
            "a SiLU'd projection would be indistinguishable from the pinned one: peak {peak:.3e}"
        );
    }
}

#[test]
fn norm_out_modulation_matches_and_is_shift_first() {
    let f = fixture();
    let device = dev();
    let c = curve(&f, &device);
    for (i, case) in f["cases"].as_array().unwrap().iter().enumerate() {
        let coords = c.coordinates(&floats(&case["timesteps"]), &device).unwrap();
        let m = c.norm_out_modulation(&coords).unwrap();
        let flat = Tensor::cat(&[&m.shift, &m.scale], 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_close(
            &flat,
            &floats(&case["norm_modulation"]),
            PROJECTION_ABS,
            &format!("case {i} norm_out modulation"),
        );
    }
}

// -------------------------------------------------------------------------------------------
// Whole native blocks and the final seams against the pinned outputs
// -------------------------------------------------------------------------------------------

struct Native {
    blocks: Vec<DitBlock>,
    norm_out: AdaLayerNormOut,
    proj_out: LinearBias,
    audio_proj_out: LinearBias,
    rope: MmRope,
}

fn native(f: &Value, device: &Device) -> Native {
    let cfg = config();
    let map = weight_map(f, device);
    let w = Weights::from_map(map);
    let c = curve(f, device);
    let blocks = (0..cfg.num_layers)
        .map(|i| {
            DitBlock::from_weights_curve(
                &w,
                &format!("transformer_blocks.{i}"),
                &cfg,
                DType::F32,
                c.block_projection(i).unwrap(),
            )
            .unwrap()
        })
        .collect();
    let norm_out = AdaLayerNormOut::from_weights_curve(&w, "norm_out", &cfg).unwrap();
    let proj_out =
        LinearBias::from_weights(&w, "proj_out", cfg.hidden_size, cfg.video_patch_dim()).unwrap();
    let audio_proj_out =
        LinearBias::from_weights(&w, "audio_proj_out", cfg.hidden_size, cfg.audio_in_channels)
            .unwrap();
    // The fixture's positions are all zero, so the rope tables are the identity — assert that
    // rather than trusting the generator, then build the tables the way `JointDit` does.
    assert!(
        floats(&f["positions"]).iter().all(|p| *p == 0.0),
        "the oracle pins identity rope; a nonzero position needs its own case"
    );
    let rope = MmRope::new(cfg.rope_freq_dim, cfg.rope_theta).unwrap();
    Native {
        blocks,
        norm_out,
        proj_out,
        audio_proj_out,
        rope,
    }
}

#[test]
#[ignore = "complete synthetic DiT/block-stack execution; dispatched only by the manual H3 native suite"]
fn native_blocks_and_final_outputs_match_the_pinned_reference() {
    let f = fixture();
    let device = dev();
    let cfg = config();
    let c = curve(&f, &device);
    let n = native(&f, &device);
    let hidden =
        Tensor::from_vec(floats(&f["input_hidden"]), (1, 5, cfg.hidden_size), &device).unwrap();
    let adaln_indices = Tensor::from_vec(
        floats(&f["adaln_indices"])
            .into_iter()
            .map(|v| v as u32)
            .collect(),
        (5,),
        &device,
    )
    .unwrap();
    let timestep_indices = Tensor::from_vec(
        floats(&f["timestep_indices"])
            .into_iter()
            .map(|v| v as u32)
            .collect(),
        (5,),
        &device,
    )
    .unwrap();
    // All-fixture positions are zero, so the rope is the identity — the tables are built the same
    // way `JointDit` builds them, from the packed sequence's position grid.
    let tables = n
        .rope
        .tables(&Tensor::from_vec(vec![0.0f32; 15], (5, 3), &device).unwrap())
        .unwrap();

    for (i, case) in f["cases"].as_array().unwrap().iter().enumerate() {
        let coords = c.coordinates(&floats(&case["timesteps"]), &device).unwrap();
        let norm_out = c.norm_out_modulation(&coords).unwrap();
        let mut x = hidden.clone();
        for (layer, block) in n.blocks.iter().enumerate() {
            let modulation = block.curve_modulation(&coords).unwrap();
            x = block
                .forward(&x, &modulation, &adaln_indices, &n.rope, &tables)
                .unwrap_or_else(|e| panic!("case {i} block {layer}: {e}"));
            assert_close(
                &x.to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                &floats(&case["block_outputs"][layer]),
                BLOCK_ABS,
                &format!("case {i} block {layer} output"),
            );
        }
        let x = n.norm_out.apply(&x, &norm_out, &timestep_indices).unwrap();
        // The final head picks the target video row (4) and target audio row (3).
        let video = n.proj_out.forward(&x.narrow(1, 4, 1).unwrap()).unwrap();
        let audio = n
            .audio_proj_out
            .forward(&x.narrow(1, 3, 1).unwrap())
            .unwrap();
        assert_close(
            &video
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            &floats(&case["video_output"]),
            BLOCK_ABS,
            &format!("case {i} video output"),
        );
        assert_close(
            &audio
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            &floats(&case["audio_output"]),
            BLOCK_ABS,
            &format!("case {i} audio output"),
        );
    }
    let _ = cfg;
}

// -------------------------------------------------------------------------------------------
// Content identity: complete, and refusing same-shape strangers
// -------------------------------------------------------------------------------------------

#[test]
fn the_identity_is_complete_and_stable() {
    let f = fixture();
    let device = dev();
    let c = curve(&f, &device);
    let identity = c.identity();
    assert_eq!(identity.0.len(), 32);
    // Same bytes → same identity.
    let again = curve(&f, &device);
    assert_eq!(again.identity(), identity);
}

#[test]
fn a_same_shaped_different_basis_never_matches() {
    let f = fixture();
    let device = dev();
    let baseline = curve(&f, &device).identity();
    let cfg = config();

    let mutate = |name: &str, element: usize| -> PruningIdentity {
        let mut map = weight_map(&f, &device);
        let t = map.get_mut(name).unwrap();
        let dtype = t.dtype();
        let mut v = t
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        v[element] += 0.25;
        let t = Tensor::from_vec(v, t.dims().to_vec(), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        map.insert(name.to_string(), t);
        let w = Weights::from_map(map);
        let topology = CurveAdaLn::inspect(&w, &cfg).unwrap().unwrap();
        let operands: HashMap<String, Tensor> = topology
            .operands()
            .iter()
            .map(|o| (o.name.clone(), w.require(&o.name).unwrap()))
            .collect();
        CurveAdaLn::assemble(topology, operands, &cfg, &device)
            .unwrap()
            .identity()
    };

    // The table alone: same shape, one value moved.
    assert_ne!(
        mutate(TABLE_KEY, 0),
        baseline,
        "a different table is a different model"
    );
    // A reduced block weight and a folded bias each identify the basis too.
    assert_ne!(
        mutate("transformer_blocks.0.adaln_proj.linear.weight", 3),
        baseline,
        "a different reduced projection is a different model"
    );
    assert_ne!(
        mutate("transformer_blocks.1.adaln_proj.linear.bias", 7),
        baseline,
        "a different folded bias is a different model"
    );
    assert_ne!(
        mutate("norm_out.linear.weight", 11),
        baseline,
        "a different final projection is a different model"
    );
}

#[test]
fn an_incomplete_identity_is_refused() {
    let f = fixture();
    let cfg = config();
    let map = weight_map(&f, &dev());
    let w = Weights::from_map(map);
    let topology = CurveAdaLn::inspect(&w, &cfg).unwrap().unwrap();
    // Stop after the table: a table alone cannot identify its centered projection basis.
    let mut builder = topology.identity_builder();
    let table = w.require(TABLE_KEY).unwrap();
    let values = table
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    builder.operand_values(TABLE_KEY, values).unwrap();
    let err = builder.finish().unwrap_err();
    assert!(
        err.contains("incomplete table/basis identity"),
        "partial identity must be refused: {err}"
    );
    // Out-of-order and duplicate operand feeds are refused too.
    let mut builder = topology.identity_builder();
    let second = &topology.operands()[1].name;
    assert!(builder
        .operand_values(second, std::iter::empty::<f32>())
        .is_err());
}

// -------------------------------------------------------------------------------------------
// AdaLN cache identity and reuse on the curve path
// -------------------------------------------------------------------------------------------

#[test]
#[ignore = "complete synthetic model/block-stack execution; dispatched only by the manual H3 native suite"]
fn the_curve_cache_keys_on_the_schedule_and_the_basis() {
    let f = fixture();
    let device = dev();
    let cfg = config();
    let c = curve(&f, &device);
    let map = weight_map(&f, &device);
    let w = Weights::from_map(map.clone());
    let blocks = (0..cfg.num_layers)
        .map(|i| {
            DitBlock::from_weights_curve(
                &w,
                &format!("transformer_blocks.{i}"),
                &cfg,
                DType::F32,
                c.block_projection(i).unwrap(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    // One step, the fixture's four row-class timesteps — the contract's canonical order
    // [video, conditioned-video, audio, conditioned-audio], as the oracle recorded them.
    let timesteps = floats(&f["cases"][0]["timesteps"]);
    let schedule = TimestepSchedule::new(vec![timesteps.clone()]).unwrap();
    let (cache, _) = c.precompute(&blocks, &schedule).unwrap();
    assert_eq!(cache.num_layers(), cfg.num_layers);
    assert_eq!(cache.pruning_identity(), Some(c.identity()));
    assert!(
        cache.is_current_for_pruning(&schedule, c.identity()),
        "the freshly built cache must serve its own schedule and basis"
    );
    cache
        .check_current_for_pruning(&schedule, c.identity())
        .expect("the freshly built cache passes its own reuse check");

    // A different basis with the SAME schedule is refused, with a reason that names the mismatch.
    let mutated = {
        let mut map2 = map.clone();
        let t = map2
            .get("transformer_blocks.0.adaln_proj.linear.weight")
            .unwrap();
        let dtype = t.dtype();
        let shape = t.dims().to_vec();
        let mut v = t
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        v[0] += 0.5;
        map2.insert(
            "transformer_blocks.0.adaln_proj.linear.weight".to_string(),
            Tensor::from_vec(v, shape, &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap(),
        );
        let w2 = Weights::from_map(map2);
        let topology = CurveAdaLn::inspect(&w2, &cfg).unwrap().unwrap();
        let operands: HashMap<String, Tensor> = topology
            .operands()
            .iter()
            .map(|o| (o.name.clone(), w2.require(&o.name).unwrap()))
            .collect();
        CurveAdaLn::assemble(topology, operands, &cfg, &device).unwrap()
    };
    let identity = mutated.identity();
    assert_ne!(identity, c.identity());
    assert!(
        !cache.is_current_for_pruning(&schedule, identity),
        "a same-shaped different basis must not reuse the cache"
    );
    let err = cache
        .check_current_for_pruning(&schedule, identity)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("different pruning basis"),
        "the refusal must say why: {err}"
    );

    // A different schedule on the same basis is refused as well.
    let other = TimestepSchedule::new(vec![vec![0.1, 0.2, 0.3, 1.0]]).unwrap();
    assert!(!cache.is_current_for_pruning(&other, c.identity()));
}

#[test]
#[ignore = "complete synthetic model/block-stack execution; dispatched only by the manual H3 native suite"]
fn a_mixed_basis_block_stack_is_refused_before_any_projection() {
    let f = fixture();
    let device = dev();
    let cfg = config();
    let basis_a = curve(&f, &device);
    // Basis B: same shapes, one reduced block weight moved.
    let mut map = weight_map(&f, &device);
    let t = map
        .get("transformer_blocks.0.adaln_proj.linear.weight")
        .unwrap();
    let dtype = t.dtype();
    let shape = t.dims().to_vec();
    let mut v = t
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    v[1] += 0.5;
    map.insert(
        "transformer_blocks.0.adaln_proj.linear.weight".to_string(),
        Tensor::from_vec(v, shape, &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap(),
    );
    let w_b = Weights::from_map(map);
    let topology_b = CurveAdaLn::inspect(&w_b, &cfg).unwrap().unwrap();
    let operands_b: HashMap<String, Tensor> = topology_b
        .operands()
        .iter()
        .map(|o| (o.name.clone(), w_b.require(&o.name).unwrap()))
        .collect();
    let basis_b = CurveAdaLn::assemble(topology_b, operands_b, &cfg, &device).unwrap();
    assert_ne!(basis_a.identity(), basis_b.identity());

    // Blocks carrying basis B's projections, precomputed under basis A: the cache would be
    // labelled with A's identity while holding B's modulation, so the precompute refuses before
    // any projection runs.
    let blocks_b = (0..cfg.num_layers)
        .map(|i| {
            DitBlock::from_weights_curve(
                &w_b,
                &format!("transformer_blocks.{i}"),
                &cfg,
                DType::F32,
                basis_b.block_projection(i).unwrap(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let schedule = TimestepSchedule::new(vec![vec![0.5, 0.6, 0.7, 1.0]]).unwrap();
    let err = basis_a
        .precompute(&blocks_b, &schedule)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("different pruning basis"),
        "mixed-basis precompute must be refused: {err}"
    );
}

// -------------------------------------------------------------------------------------------
// The model-level seams: fail-closed embedding, identity exposure, curve precompute
// -------------------------------------------------------------------------------------------

#[test]
#[ignore = "complete synthetic model/block-stack execution; dispatched only by the manual H3 native suite"]
fn the_stock_path_is_untouched_and_refuses_curve_coordinates() {
    use crate::common::{dit_fixture_config, weights, Golden, DIT_FIXTURE};
    let device = dev();
    // A stock checkpoint — full timestep MLP, full-width projections, no table — must load exactly
    // as before the curve work existed, and refuse the curve accessors: the two embedding sources
    // are mutually exclusive, never a fallback pair.
    let golden = Golden::load(DIT_FIXTURE);
    let cfg = dit_fixture_config();
    let w = weights(golden.model_map(&["src.", "in.", "out.", "layout."]));
    let model =
        candle_gen_minimax_h3::dit::MiniMaxH3Dit::from_weights(&w, &cfg, &device, DType::BF16)
            .expect("the stock checkpoint still loads");
    assert!(
        model.holds_adaln(),
        "the stock checkpoint still holds its full-width projections"
    );
    assert!(
        model.pruning_identity().is_none(),
        "the stock path carries no pruning identity"
    );
    assert!(
        model.embed_timesteps(&[0.5]).is_ok(),
        "the stock timestep MLP is intact"
    );
    let err = model.curve_coordinates(&[0.5]).unwrap_err().to_string();
    assert!(
        err.contains("full timestep MLP"),
        "curve access is refused on the stock path: {err}"
    );
}

#[test]
#[ignore = "complete synthetic DiT/block-stack execution; dispatched only by the manual H3 native suite"]
fn load_from_dir_builds_the_pruned_model_with_its_identity_at_stored_precision() {
    // A synthetic **complete** pruned `transformer/` partition: the fixture's canonical curve
    // operands plus the top-level tensors the fixture omits (the input projections, the text
    // embedder, and a token-refiner body copied from block 0). Written to a real safetensors file
    // with the fixture config, this exercises the changed loader path (`load_from_dir` →
    // stored-precision native load → `from_weights`) and the public identity end to end, without
    // a real checkpoint campaign.
    let f = fixture();
    let cfg = config();
    let dir = tempfile::tempdir().unwrap();
    let mut tensors: HashMap<String, (safetensors::Dtype, Vec<usize>, Vec<u8>)> = HashMap::new();
    for (name, entry) in f["tensors"].as_object().unwrap() {
        let dtype = match entry["dtype"].as_str().unwrap() {
            "F32" => safetensors::Dtype::F32,
            "BF16" => safetensors::Dtype::BF16,
            "F16" => safetensors::Dtype::F16,
            other => panic!("fixture dtype {other}"),
        };
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect();
        let t = stored_tensor(name, &f, &Device::Cpu);
        // Write the STORED dtype's bytes back — the loader must see a real BF16/F16 tensor, not
        // an F32 stand-in, or the source-precision assertions below would be vacuous.
        let values = t
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let bytes: Vec<u8> = match dtype {
            safetensors::Dtype::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            safetensors::Dtype::BF16 => values
                .iter()
                .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                .collect(),
            safetensors::Dtype::F16 => values
                .iter()
                .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                .collect(),
            other => panic!("fixture dtype {other:?}"),
        };
        tensors.insert(name.clone(), (dtype, shape, bytes));
    }
    // The omitted top-level inputs: deterministic fillers at the published shapes.
    let filler = |bytes: usize| -> Vec<u8> {
        assert!(bytes.is_multiple_of(4));
        (0..bytes / 4)
            .flat_map(|i| (((i % 7) as f32 - 3.0) / 32.0).to_le_bytes())
            .collect()
    };
    let (patch, hidden, text) = (cfg.video_patch_dim(), cfg.hidden_size, cfg.text_dim);
    let mut push = |name: String, dtype: safetensors::Dtype, shape: Vec<usize>, data: Vec<u8>| {
        tensors.insert(name, (dtype, shape, data));
    };
    push(
        "proj_in.weight".to_string(),
        safetensors::Dtype::F32,
        vec![hidden, patch],
        filler(hidden * patch * 4),
    );
    push(
        "proj_in.bias".to_string(),
        safetensors::Dtype::F32,
        vec![hidden],
        filler(hidden * 4),
    );
    push(
        "audio_proj_in.weight".to_string(),
        safetensors::Dtype::F32,
        vec![hidden, cfg.audio_in_channels],
        filler(hidden * cfg.audio_in_channels * 4),
    );
    push(
        "audio_proj_in.bias".to_string(),
        safetensors::Dtype::F32,
        vec![hidden],
        filler(hidden * 4),
    );
    push(
        "context_embedder.weight".to_string(),
        safetensors::Dtype::F32,
        vec![hidden, text],
        filler(hidden * text * 4),
    );
    push(
        "context_embedder.bias".to_string(),
        safetensors::Dtype::F32,
        vec![hidden],
        filler(hidden * 4),
    );
    for name in [
        "norm1.weight",
        "norm2.weight",
        "attn.to_q.weight",
        "attn.to_k.weight",
        "attn.to_v.weight",
        "attn.norm_q.weight",
        "attn.norm_k.weight",
        "attn.to_out.0.weight",
        "ff.net.0.proj.weight",
        "ff.net.2.weight",
    ] {
        let src = format!("transformer_blocks.0.{name}");
        let entry = &f["tensors"][&src];
        let dtype = match entry["dtype"].as_str().unwrap() {
            "F32" => safetensors::Dtype::F32,
            other => panic!("refiner body dtype {other}"),
        };
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect();
        let t = stored_tensor(&src, &f, &Device::Cpu);
        push(
            format!("token_refiner.refiner_blocks.0.{name}"),
            dtype,
            shape,
            t.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        );
    }
    push(
        "token_refiner.final_norm.weight".to_string(),
        safetensors::Dtype::F32,
        vec![hidden],
        filler(hidden * 4),
    );

    let views: HashMap<String, safetensors::tensor::TensorView> = tensors
        .iter()
        .map(|(k, (d, s, data))| {
            (
                k.clone(),
                safetensors::tensor::TensorView::new(*d, s.clone(), data).unwrap(),
            )
        })
        .collect();
    let file = dir.path().join("transformer.safetensors");
    safetensors::tensor::serialize_to_file(views, &None, &file).unwrap();
    std::fs::write(dir.path().join("config.json"), f["config"].to_string()).unwrap();

    let model = candle_gen_minimax_h3::dit::MiniMaxH3Dit::load_from_dir(
        dir.path(),
        &Device::Cpu,
        DType::F32,
    )
    .expect("the synthetic pruned partition loads");
    let identity = model
        .pruning_identity()
        .expect("the pruned model exposes its identity");
    model.ensure_pruning_identity(&identity).unwrap();
    let mut other = identity;
    other.0[0] ^= 1;
    assert!(model.ensure_pruning_identity(&other).is_err());
    assert_eq!(
        identity,
        curve(&f, &Device::Cpu).identity(),
        "the identity binds the loaded bytes"
    );
    let mlp_error = model.embed_timesteps(&[0.5]).unwrap_err().to_string();
    assert!(
        mlp_error.contains("modulation-pruned") && mlp_error.contains("curve_coordinates"),
        "{mlp_error}"
    );
    assert!(model.curve_coordinates(&[0.5]).is_ok());
    assert!(model
        .curve_coordinates(&[f32::NAN])
        .unwrap_err()
        .to_string()
        .contains("finite"));

    use candle_gen_minimax_h3::dit::model::{BlockModulation, PackedForward};
    let device = Device::Cpu;
    let loaded_curve = model.curve().unwrap();
    let times = floats(&f["cases"][4]["timesteps"]);
    let coords = loaded_curve.coordinates(&times, &device).unwrap();
    let norm = loaded_curve.norm_out_modulation(&coords).unwrap();
    let schedule = TimestepSchedule::new(vec![times]).unwrap();
    let own_blocks = native(&f, &device);
    let (cache, _) = loaded_curve
        .precompute(&own_blocks.blocks, &schedule)
        .unwrap();
    let video = Tensor::from_vec(vec![0.1f32; 8], (1, 2, 4), &device).unwrap();
    let audio = Tensor::from_vec(vec![0.2f32; 8], (1, 2, 4), &device).unwrap();
    let text = Tensor::from_vec(vec![0.3f32; 8], (1, 1, 8), &device).unwrap();
    let adaln = Tensor::from_vec(vec![1u32, 3, 11, 8, 0], (5,), &device).unwrap();
    let ts = Tensor::from_vec(vec![0u32, 1, 3, 2, 0], (5,), &device).unwrap();
    let positions = Tensor::zeros((5, 3), DType::F32, &device).unwrap();
    let tables = own_blocks.rope.tables(&positions).unwrap();
    let packed = PackedForward {
        video_rows: &video,
        audio_rows: &audio,
        text_rows: &text,
        adaln_indices: &adaln,
        timestep_indices: &ts,
        tables: &tables,
        text_indices: &[0],
        video_indices: &[1, 4],
        audio_indices: &[2, 3],
    };
    let cached = model
        .forward_packed(&packed, BlockModulation::Cached(&cache), &norm)
        .unwrap();
    let resident = model
        .forward_packed(&packed, BlockModulation::CurveCoords(&coords), &norm)
        .unwrap();
    for (got, want) in [(&resident.0, &cached.0), (&resident.1, &cached.1)] {
        assert_close(
            &got.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            &want.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            BLOCK_ABS,
            "loaded source resident/cache equivalence",
        );
    }
    let mut other_fixture = f.clone();
    let value = other_fixture["tensors"][TABLE_KEY]["data"][3]
        .as_f64()
        .unwrap();
    other_fixture["tensors"][TABLE_KEY]["data"][3] = (value + 0.5).into();
    let other_curve = curve(&other_fixture, &device);
    let other_blocks = native(&other_fixture, &device);
    let (foreign, _) = other_curve
        .precompute(&other_blocks.blocks, &schedule)
        .unwrap();
    let error = model
        .forward_packed(&packed, BlockModulation::Cached(&foreign), &norm)
        .unwrap_err()
        .to_string();
    assert!(error.contains("pruning identity"), "{error}");
    println!("S5 source-preserving synthetic partition load + resident/cache forward: native CPU video/audio outputs match; foreign basis refused");
    assert!(
        !model.holds_adaln(),
        "no full-width projection exists on the pruned path"
    );
    let curve = model.curve().expect("the curve is resident");
    assert_eq!(
        curve.table().dtype(),
        DType::F32,
        "the table stays F32 through the loader"
    );
    assert_eq!(
        curve.block_projection(0).unwrap().source_dtype(),
        DType::BF16
    );
    assert_eq!(
        curve.block_projection(0).unwrap().compute_dtype(),
        DType::F32
    );
}

// -------------------------------------------------------------------------------------------
// The CUDA integration target — wired to the same shared oracle, run by the CI lane
// -------------------------------------------------------------------------------------------

#[ignore = "requires a CUDA device; absence is failure, not skip (integration lane, sc-26684)"]
#[test]
fn native_pruned_adaln_matches_pinned_reference() {
    let device = Device::new_cuda(0).expect(
        "the integration lane must provide a CUDA device — an unavailable device is a failure, \
         not a skip",
    );
    let f = fixture();
    let cfg = config();
    let c = curve(&f, &device);
    let n = native(&f, &device);
    let hidden =
        Tensor::from_vec(floats(&f["input_hidden"]), (1, 5, cfg.hidden_size), &device).unwrap();
    let to_indices = |v: &Value| -> Tensor {
        Tensor::from_vec(
            floats(v).into_iter().map(|x| x as u32).collect(),
            (5,),
            &device,
        )
        .unwrap()
    };
    let adaln_indices = to_indices(&f["adaln_indices"]);
    let timestep_indices = to_indices(&f["timestep_indices"]);
    let tables = n
        .rope
        .tables(&Tensor::from_vec(vec![0.0f32; 15], (5, 3), &device).unwrap())
        .unwrap();
    let flatten = |tensor: &Tensor| {
        tensor
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    assert_eq!(
        f["cases"].as_array().unwrap().len(),
        10,
        "native proof must execute every pinned sweep case"
    );
    for (i, case) in f["cases"].as_array().unwrap().iter().enumerate() {
        let coords = c.coordinates(&floats(&case["timesteps"]), &device).unwrap();
        assert_close(
            &flatten(&coords),
            &floats(&case["coordinates"]),
            COORDS_ABS,
            &format!("CUDA case {i} coordinates"),
        );
        let final_projected = c.norm_out_projection().unwrap().forward(&coords).unwrap();
        assert_close(
            &flatten(&final_projected),
            &floats(&case["norm_modulation"]),
            PROJECTION_ABS,
            &format!("CUDA case {i} final reduced projection"),
        );
        let norm_out = c.norm_out_modulation(&coords).unwrap();
        let mut x = hidden.clone();
        for (layer, block) in n.blocks.iter().enumerate() {
            let projected = c.block_projection(layer).unwrap().forward(&coords).unwrap();
            assert_close(
                &flatten(&projected),
                &floats(&case["block_modulations"][layer]),
                PROJECTION_ABS,
                &format!("CUDA case {i} block {layer} reduced projection"),
            );
            let modulation = block.curve_modulation(&coords).unwrap();
            x = block
                .forward(&x, &modulation, &adaln_indices, &n.rope, &tables)
                .unwrap_or_else(|e| panic!("case {i} block {layer}: {e}"));
            assert_close(
                &x.to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                &floats(&case["block_outputs"][layer]),
                BLOCK_ABS,
                &format!("case {i} block {layer} output"),
            );
        }
        let x = n.norm_out.apply(&x, &norm_out, &timestep_indices).unwrap();
        let video = n.proj_out.forward(&x.narrow(1, 4, 1).unwrap()).unwrap();
        let audio = n
            .audio_proj_out
            .forward(&x.narrow(1, 3, 1).unwrap())
            .unwrap();
        assert_close(
            &video
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            &floats(&case["video_output"]),
            BLOCK_ABS,
            &format!("case {i} video output"),
        );
        assert_close(
            &audio
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            &floats(&case["audio_output"]),
            BLOCK_ABS,
            &format!("case {i} audio output"),
        );
    }
}

// Manual-only complete native-model proof. The pinned source fixture supplies every learned
// pruned operand unchanged; only its already-omitted input heads/refiner receive deterministic
// fillers, exactly as the complete-partition loader proof above does. No new producer/oracle.
fn manual_conditioning_model(
    f: &Value,
    cfg: &MiniMaxH3DitConfig,
    d: &Device,
) -> candle_gen_minimax_h3::dit::MiniMaxH3Dit {
    let mut map = weight_map(f, d);
    for (name, shape) in [
        (
            "proj_in.weight",
            vec![cfg.hidden_size, cfg.video_patch_dim()],
        ),
        ("proj_in.bias", vec![cfg.hidden_size]),
        (
            "audio_proj_in.weight",
            vec![cfg.hidden_size, cfg.audio_in_channels],
        ),
        ("audio_proj_in.bias", vec![cfg.hidden_size]),
        (
            "context_embedder.weight",
            vec![cfg.hidden_size, cfg.text_dim],
        ),
        ("context_embedder.bias", vec![cfg.hidden_size]),
        ("token_refiner.final_norm.weight", vec![cfg.hidden_size]),
    ] {
        let count: usize = shape.iter().product();
        let values: Vec<f32> = (0..count).map(|i| ((i % 7) as f32 - 3.) / 32.).collect();
        map.insert(name.into(), Tensor::from_vec(values, shape, d).unwrap());
    }
    for body in [
        "norm1.weight",
        "norm2.weight",
        "attn.to_q.weight",
        "attn.to_k.weight",
        "attn.to_v.weight",
        "attn.norm_q.weight",
        "attn.norm_k.weight",
        "attn.to_out.0.weight",
        "ff.net.0.proj.weight",
        "ff.net.2.weight",
    ] {
        let value = map[&format!("transformer_blocks.0.{body}")].clone();
        map.insert(format!("token_refiner.refiner_blocks.0.{body}"), value);
    }
    let model = candle_gen_minimax_h3::dit::MiniMaxH3Dit::from_weights(
        &Weights::from_map(map),
        cfg,
        d,
        DType::F32,
    )
    .unwrap();
    let curve = model.curve().unwrap();
    assert_eq!(curve.table().dtype(), DType::F32);
    assert_eq!(
        curve.block_projection(0).unwrap().source_dtype(),
        DType::BF16
    );
    model
}

#[test]
#[ignore = "explicit manual full-model numerical proof of selected unit patches and ordered mono/stereo reference conditioning; never ordinary CI"]
fn manual_unit_patch_and_mixed_reference_full_forward_matches_resident_control() {
    use candle_gen_minimax_h3::conditioning::{
        keyframe_condition_rows, reference_audio_rows, KeyframeNoise,
    };
    use candle_gen_minimax_h3::denoise::{
        JointGeometry, JointStep, JointVelocity, PackedLayout, TEXT_TAG,
    };
    use candle_gen_minimax_h3::dit::adaln::AdaLnResidency;
    use candle_gen_minimax_h3::dit::model::JointDit;
    use candle_gen_minimax_h3::dit::positions::{KeyframeAnchor, ReferenceLatentGeometry};
    use candle_gen_minimax_h3::pipeline::{
        fl2va_layout, prepend_condition_audio_rows, prepend_condition_rows,
        ref2va_layout_with_audio_channels, RequestGeometry,
    };
    use candle_gen_minimax_h3::reference::ReferenceKind;
    let f = fixture();
    let cfg = config();
    assert_eq!(cfg.patch_size, [1, 1, 1]);
    #[cfg(feature = "cuda")]
    let d = Device::new_cuda(0)
        .expect("explicit CUDA proof requires the workflow-selected CUDA device; no CPU fallback");
    #[cfg(feature = "cuda")]
    assert!(d.is_cuda(), "the explicit proof must execute on CUDA");
    #[cfg(not(feature = "cuda"))]
    let d = Device::Cpu;
    println!(
        "H3 unit-patch/mixed-reference full forward device={d:?}; cuda={}",
        d.is_cuda()
    );
    // The core's minimum exact 17n+5 lattice is enough for a complete two-block forward.
    let geometry = RequestGeometry {
        width: 32,
        height: 32,
        joint: JointGeometry::new(5, 2, 2).unwrap(),
    };
    let target_video = Tensor::full(
        0.1f32,
        (
            1,
            geometry.joint.num_latent_frames * 4,
            cfg.video_patch_dim(),
        ),
        &d,
    )
    .unwrap();
    let target_audio = Tensor::full(
        0.2f32,
        (
            1,
            geometry.joint.num_audio_latents * 2,
            cfg.audio_in_channels,
        ),
        &d,
    )
    .unwrap();
    let condition = Tensor::full(0.7f32, (1, cfg.in_channels, 1, 2, 2), &d).unwrap();
    let visual =
        keyframe_condition_rows(&condition, cfg.patch_size, &KeyframeNoise::Seeded, 0, &d).unwrap();
    assert_eq!(visual.dims(), &[1, 4, cfg.in_channels]);
    let context = Tensor::full(0.3f32, (1, 1, cfg.text_dim), &d).unwrap();
    let times = PackedLayout::row_timesteps(0.6, 0.4);
    let schedule = TimestepSchedule::new(vec![times.to_vec()]).unwrap();
    let completed = std::cell::Cell::new(0usize);
    let check = |layout: PackedLayout, video: Tensor, audio: Tensor| {
        assert_eq!(video.dims()[1], layout.video_indices().len());
        assert_eq!(audio.dims()[1], layout.audio_indices().len());
        assert_eq!(
            layout.audio_channels(),
            2,
            "generated stereo is independent of references"
        );
        let indices = schedule
            .adaln_indices(0, layout.row_classes(), layout.token_tags(), &d)
            .unwrap();
        let mut outputs = Vec::new();
        for residency in [AdaLnResidency::Resident, AdaLnResidency::PrecomputeAndEvict] {
            let mut driver = JointDit::new(
                manual_conditioning_model(&f, &cfg, &d),
                layout.clone(),
                &context,
                schedule.clone(),
                residency,
            )
            .unwrap();
            let step = JointStep {
                index: 0,
                layout: &layout,
                video_rows: &video,
                audio_rows: &audio,
                adaln_indices: &indices,
                row_timesteps: times,
            };
            let output = driver
                .forward(&step)
                .expect("actual full native joint model forward");
            #[cfg(feature = "cuda")]
            assert!(
                output.0.device().is_cuda() && output.1.device().is_cuda(),
                "both full-model outputs must remain on the selected CUDA device"
            );
            assert_eq!(output.0.dims(), video.dims());
            assert_eq!(output.1.dims(), audio.dims());
            outputs.push(output);
        }
        for (cached, resident) in [
            (&outputs[1].0, &outputs[0].0),
            (&outputs[1].1, &outputs[0].1),
        ] {
            let a = cached.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let b = resident.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(a.iter().chain(&b).all(|v| v.is_finite()));
            assert!(a.iter().any(|v| v.abs() > 1e-12));
            assert_close(
                &a,
                &b,
                BLOCK_ABS,
                "unit-patch conditioned full-model cached/resident control",
            );
        }
        completed.set(completed.get() + 1);
    };
    for anchors in [
        vec![KeyframeAnchor::First],
        vec![KeyframeAnchor::Last],
        vec![KeyframeAnchor::First, KeyframeAnchor::Last],
    ] {
        let layout = fl2va_layout(&geometry, &[TEXT_TAG], &anchors, cfg.patch_size, &d).unwrap();
        let refs: Vec<&Tensor> = anchors.iter().map(|_| &visual).collect();
        let rows = Tensor::cat(&refs, 1).unwrap();
        assert_eq!(layout.num_condition_video_rows(), 4 * anchors.len());
        let video = prepend_condition_rows(&layout, Some(&rows), &target_video).unwrap();
        check(layout, video, target_audio.clone());
    }
    for (a, b, reverse) in (1usize..=2)
        .flat_map(|a| (1usize..=2).flat_map(move |b| [false, true].map(move |r| (a, b, r))))
    {
        let mut references = vec![
            ReferenceLatentGeometry {
                kind: ReferenceKind::Audio,
                num_latent_frames: 0,
                latent_height: 0,
                latent_width: 0,
                num_audio_latents: 3,
            },
            ReferenceLatentGeometry {
                kind: ReferenceKind::Image,
                num_latent_frames: 1,
                latent_height: 2,
                latent_width: 2,
                num_audio_latents: 0,
            },
            ReferenceLatentGeometry {
                kind: ReferenceKind::Video,
                num_latent_frames: 1,
                latent_height: 2,
                latent_width: 2,
                num_audio_latents: 5,
            },
        ];
        let mut channels = vec![a, 0, b];
        let standalone =
            reference_audio_rows(&Tensor::full(0.5f32, (a, 3, cfg.audio_in_channels), &d).unwrap())
                .unwrap();
        let soundtrack =
            reference_audio_rows(&Tensor::full(0.8f32, (b, 5, cfg.audio_in_channels), &d).unwrap())
                .unwrap();
        let audio = if reverse {
            references.reverse();
            channels.reverse();
            Tensor::cat(&[&soundtrack, &standalone], 1).unwrap()
        } else {
            Tensor::cat(&[&standalone, &soundtrack], 1).unwrap()
        };
        let layout = ref2va_layout_with_audio_channels(
            &geometry,
            &[TEXT_TAG],
            &references,
            &channels,
            cfg.patch_size,
            &d,
        )
        .unwrap();
        assert_eq!(layout.num_condition_audio_rows(), a * 3 + b * 5);
        assert_eq!(layout.num_condition_video_rows(), 8);
        let video = prepend_condition_rows(
            &layout,
            Some(&Tensor::cat(&[&visual, &visual], 1).unwrap()),
            &target_video,
        )
        .unwrap();
        let audio = prepend_condition_audio_rows(&layout, Some(&audio), &target_audio).unwrap();
        check(layout, video, audio);
    }
    assert_eq!(
        completed.get(),
        11,
        "every keyframe and mixed-channel ordered case must execute"
    );
    #[cfg(feature = "cuda")]
    println!("H3_GEOMETRY_AUDIO_FULL_FORWARD backend=cuda cases=11");
}
