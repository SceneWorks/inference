use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::Weights;
use candle_gen_minimax_h3::dit::{
    AdaLayerNormOut, CurveAdaLn, DitBlock, LinearBias, MmRope, TimestepSchedule,
};
use candle_gen_minimax_h3::MiniMaxH3DitConfig;
use gen_core_h3::pruned_adaln::{PrunedAdaLn, PruningIdentity, TABLE_KEY};
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
    println!("H3_PRUNED_ORACLE {what}: max_abs={peak:.9e} tolerance={tol:.9e}");
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
    #[cfg(feature = "cuda")]
    assert!(
        !candle_gen::candle_core::cuda_backend::gemm_reduced_precision_f32(),
        "the pinned curve oracle requires CUBLAS_COMPUTE_32F, never FAST_TF32"
    );
    let device = Device::new_cuda(0).expect(
        "the integration lane must provide a CUDA device — an unavailable device is a failure, \
         not a skip",
    );
    assert!(
        device.is_cuda(),
        "a CPU fallback cannot prove CUDA precision"
    );
    println!("H3_PRUNED_ORACLE_DEVICE {device:?}");
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
fn manual_conditioning_map(
    f: &Value,
    cfg: &MiniMaxH3DitConfig,
    d: &Device,
) -> HashMap<String, Tensor> {
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
    map
}

fn manual_conditioning_model(
    f: &Value,
    cfg: &MiniMaxH3DitConfig,
    d: &Device,
) -> candle_gen_minimax_h3::dit::MiniMaxH3Dit {
    let map = manual_conditioning_map(f, cfg, d);
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
    manual_conditioning_causality(&f, &cfg, &d);
}

// Called ONLY by the ignored manual full-forward proof above. Compare GENERATED rows, never
// the preserved conditioning prefix: dropping the conditioning in both executions must fail.
fn causal_signal(shape: &[usize], salt: usize, scale: f32, d: &Device) -> Tensor {
    let n: usize = shape.iter().product();
    let values: Vec<f32> = (0..n)
        .map(|i| (((i * 7 + salt) % 29) as f32 - 14.) * scale)
        .collect();
    Tensor::from_vec(values, shape, d).unwrap()
}

#[allow(clippy::too_many_arguments)]
fn causal_generated_forward(
    f: &Value,
    cfg: &MiniMaxH3DitConfig,
    d: &Device,
    layout: candle_gen_minimax_h3::denoise::PackedLayout,
    context: &Tensor,
    condition_video: Option<&Tensor>,
    condition_audio: Option<&Tensor>,
    selected: Option<candle_gen_minimax_h3::dit::MiniMaxH3Dit>,
) -> Vec<f32> {
    use candle_gen_minimax_h3::denoise::{JointStep, JointVelocity, PackedLayout};
    use candle_gen_minimax_h3::dit::adaln::AdaLnResidency;
    use candle_gen_minimax_h3::dit::model::JointDit;
    use candle_gen_minimax_h3::pipeline::{prepend_condition_audio_rows, prepend_condition_rows};
    let cv = layout.num_condition_video_rows();
    let ca = layout.num_condition_audio_rows();
    let nv = layout.video_indices().len() - cv;
    let na = layout.audio_indices().len() - ca;
    let video = causal_signal(&[1, nv, cfg.video_patch_dim()], 5, 0.03, d);
    let audio = causal_signal(&[1, na, cfg.audio_in_channels], 9, 0.04, d);
    let video = prepend_condition_rows(&layout, condition_video, &video).unwrap();
    let audio = prepend_condition_audio_rows(&layout, condition_audio, &audio).unwrap();
    let times = PackedLayout::row_timesteps(0.6, 0.4);
    let schedule = TimestepSchedule::new(vec![times.to_vec()]).unwrap();
    let indices = schedule
        .adaln_indices(0, layout.row_classes(), layout.token_tags(), d)
        .unwrap();
    let model = selected.unwrap_or_else(|| manual_conditioning_model(f, cfg, d));
    let mut driver = JointDit::new(
        model,
        layout.clone(),
        context,
        schedule,
        AdaLnResidency::Resident,
    )
    .unwrap();
    let output = driver
        .forward(&JointStep {
            index: 0,
            layout: &layout,
            video_rows: &video,
            audio_rows: &audio,
            adaln_indices: &indices,
            row_timesteps: times,
        })
        .unwrap();
    let mut generated = Vec::new();
    for (rows, start, count, features) in [
        (&output.0, cv, nv, cfg.video_patch_dim()),
        (&output.1, ca, na, cfg.audio_in_channels),
    ] {
        let tail = rows
            .narrow(1, start, count)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        assert_eq!(tail.dims(), &[1, count, features]);
        generated.extend(tail.flatten_all().unwrap().to_vec1::<f32>().unwrap());
    }
    assert!(generated.iter().all(|x| x.is_finite()));
    generated
}

fn assert_generated_cause(d: &Device, property: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    assert!(!a.is_empty());
    assert!(a.iter().chain(b).all(|x| x.is_finite()));
    let peak = a.iter().chain(b).map(|x| x.abs()).fold(0., f32::max);
    let delta = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max);
    let floor = 8. * f32::EPSILON * peak;
    let backend = if d.is_cuda() { "cuda" } else { "cpu" };
    println!("H3_CONDITIONING_CAUSAL backend={backend} property={property} max_abs={delta:.9e} floor={floor:.9e} generated_only=true");
    assert!(
        peak > 0. && delta > floor,
        "{property}: generated outputs are causally indistinguishable: delta={delta} floor={floor}"
    );
}

fn manual_conditioning_causality(f: &Value, cfg: &MiniMaxH3DitConfig, d: &Device) {
    use candle_gen_minimax_h3::conditioning::{
        keyframe_condition_rows, reference_audio_rows, KeyframeNoise,
    };
    use candle_gen_minimax_h3::denoise::{JointGeometry, TEXT_TAG};
    use candle_gen_minimax_h3::dit::positions::{KeyframeAnchor, ReferenceLatentGeometry};
    use candle_gen_minimax_h3::model::MiniMaxH3Task;
    use candle_gen_minimax_h3::pipeline::{
        fl2va_layout, ref2va_layout_with_audio_channels, RequestGeometry,
    };
    use candle_gen_minimax_h3::reference::ReferenceKind;
    let geometry = RequestGeometry {
        width: 32,
        height: 32,
        joint: JointGeometry::new(5, 2, 2).unwrap(),
    };
    let context_a = causal_signal(&[1, 1, cfg.text_dim], 13, 0.2, d);
    let context_b = causal_signal(&[1, 1, cfg.text_dim], 17, 0.35, d);
    let visual_a = keyframe_condition_rows(
        &causal_signal(&[1, cfg.in_channels, 1, 2, 2], 3, 0.12, d),
        cfg.patch_size,
        &KeyframeNoise::Seeded,
        0,
        d,
    )
    .unwrap();
    let visual_b = keyframe_condition_rows(
        &causal_signal(&[1, cfg.in_channels, 1, 2, 2], 19, 0.25, d),
        cfg.patch_size,
        &KeyframeNoise::Seeded,
        0,
        d,
    )
    .unwrap();
    let run_fl = |anchors: &[KeyframeAnchor], rows: Option<&Tensor>, context: &Tensor| {
        causal_generated_forward(
            f,
            cfg,
            d,
            fl2va_layout(&geometry, &[TEXT_TAG], anchors, cfg.patch_size, d).unwrap(),
            context,
            rows,
            None,
            None,
        )
    };
    let plain_a = run_fl(&[], None, &context_a);
    let plain_b = run_fl(&[], None, &context_b);
    assert_eq!(
        plain_a,
        run_fl(&[], None, &context_a),
        "same-input generated control must be deterministic"
    );
    assert_generated_cause(d, "context_values", &plain_a, &plain_b);
    assert_generated_cause(
        d,
        "context_vs_zero_same_shape",
        &plain_a,
        &run_fl(&[], None, &causal_signal(&[1, 1, cfg.text_dim], 0, 0., d)),
    );
    let first_a = run_fl(&[KeyframeAnchor::First], Some(&visual_a), &context_a);
    let first_b = run_fl(&[KeyframeAnchor::First], Some(&visual_b), &context_a);
    let last_a = run_fl(&[KeyframeAnchor::Last], Some(&visual_a), &context_a);
    let last_b = run_fl(&[KeyframeAnchor::Last], Some(&visual_b), &context_a);
    assert_generated_cause(d, "first_only_values", &first_a, &first_b);
    let zero_visual = causal_signal(&[1, 4, cfg.in_channels], 0, 0., d);
    assert_generated_cause(
        d,
        "first_only_vs_zero_same_shape",
        &first_a,
        &run_fl(&[KeyframeAnchor::First], Some(&zero_visual), &context_a),
    );
    assert_generated_cause(d, "last_only_values", &last_a, &last_b);
    assert_generated_cause(d, "first_vs_last_placement", &first_a, &last_a);
    assert_generated_cause(d, "first_only_vs_drop", &first_a, &plain_a);
    assert_generated_cause(d, "last_only_vs_drop", &last_b, &plain_a);
    let ab = Tensor::cat(&[&visual_a, &visual_b], 1).unwrap();
    let ba = Tensor::cat(&[&visual_b, &visual_a], 1).unwrap();
    let anchors = [KeyframeAnchor::First, KeyframeAnchor::Last];
    let both_ab = run_fl(&anchors, Some(&ab), &context_a);
    let both_ba = run_fl(&anchors, Some(&ba), &context_a);
    assert_generated_cause(d, "first_last_swap_values", &both_ab, &both_ba);
    // Keep BOTH slots, positions and row counts fixed: removing a slot would also change
    // attention/rotary geometry and could pass even if that slot's VALUES were discarded.
    let zero_first = Tensor::cat(&[&zero_visual, &visual_b], 1).unwrap();
    let zero_last = Tensor::cat(&[&visual_a, &zero_visual], 1).unwrap();
    assert_generated_cause(
        d,
        "first_last_drop_first",
        &both_ab,
        &run_fl(&anchors, Some(&zero_first), &context_a),
    );
    assert_generated_cause(
        d,
        "first_last_drop_last",
        &both_ab,
        &run_fl(&anchors, Some(&zero_last), &context_a),
    );
    let image = ReferenceLatentGeometry {
        kind: ReferenceKind::Image,
        num_latent_frames: 1,
        latent_height: 2,
        latent_width: 2,
        num_audio_latents: 0,
    };
    let image_layout = ref2va_layout_with_audio_channels(
        &geometry,
        &[TEXT_TAG],
        &[image],
        &[0],
        cfg.patch_size,
        d,
    )
    .unwrap();
    let ref_a = causal_generated_forward(
        f,
        cfg,
        d,
        image_layout.clone(),
        &context_a,
        Some(&visual_a),
        None,
        None,
    );
    let ref_b = causal_generated_forward(
        f,
        cfg,
        d,
        image_layout,
        &context_a,
        Some(&visual_b),
        None,
        None,
    );
    assert_generated_cause(d, "reference_image_values", &ref_a, &ref_b);
    assert_generated_cause(d, "fl2va_vs_ref2va_positioning", &first_a, &ref_a);
    let ordered = ref2va_layout_with_audio_channels(
        &geometry,
        &[TEXT_TAG],
        &[image, image],
        &[0, 0],
        cfg.patch_size,
        d,
    )
    .unwrap();
    let refs_ab = causal_generated_forward(
        f,
        cfg,
        d,
        ordered.clone(),
        &context_a,
        Some(&ab),
        None,
        None,
    );
    let refs_ba = causal_generated_forward(f, cfg, d, ordered, &context_a, Some(&ba), None, None);
    assert_generated_cause(d, "ordered_reference_values", &refs_ab, &refs_ba);
    let video_geometry = ReferenceLatentGeometry {
        kind: ReferenceKind::Video,
        num_latent_frames: 2,
        ..image
    };
    let video_a = keyframe_condition_rows(
        &causal_signal(&[1, cfg.in_channels, 2, 2, 2], 4, 0.16, d),
        cfg.patch_size,
        &KeyframeNoise::Seeded,
        0,
        d,
    )
    .unwrap();
    let video_b = keyframe_condition_rows(
        &causal_signal(&[1, cfg.in_channels, 2, 2, 2], 21, 0.23, d),
        cfg.patch_size,
        &KeyframeNoise::Seeded,
        0,
        d,
    )
    .unwrap();
    let video_layout = ref2va_layout_with_audio_channels(
        &geometry,
        &[TEXT_TAG],
        &[video_geometry],
        &[0],
        cfg.patch_size,
        d,
    )
    .unwrap();
    let video_out_a = causal_generated_forward(
        f,
        cfg,
        d,
        video_layout.clone(),
        &context_a,
        Some(&video_a),
        None,
        None,
    );
    let video_out_b = causal_generated_forward(
        f,
        cfg,
        d,
        video_layout,
        &context_a,
        Some(&video_b),
        None,
        None,
    );
    assert_generated_cause(d, "reference_video_only_values", &video_out_a, &video_out_b);
    // Isolated audio CONDITIONING_ROWS is a low-level packing/DiT witness, NOT an accepted
    // audio-only Generator request (which requires a companion visual reference).
    let audio_geometry = ReferenceLatentGeometry {
        kind: ReferenceKind::Audio,
        num_latent_frames: 0,
        latent_height: 0,
        latent_width: 0,
        num_audio_latents: 3,
    };
    let audio_a =
        reference_audio_rows(&causal_signal(&[1, 3, cfg.audio_in_channels], 2, 0.14, d)).unwrap();
    let audio_b =
        reference_audio_rows(&causal_signal(&[1, 3, cfg.audio_in_channels], 23, 0.27, d)).unwrap();
    let audio_layout = ref2va_layout_with_audio_channels(
        &geometry,
        &[TEXT_TAG],
        &[audio_geometry],
        &[1],
        cfg.patch_size,
        d,
    )
    .unwrap();
    let audio_out_a = causal_generated_forward(
        f,
        cfg,
        d,
        audio_layout.clone(),
        &context_a,
        None,
        Some(&audio_a),
        None,
    );
    let audio_out_b = causal_generated_forward(
        f,
        cfg,
        d,
        audio_layout,
        &context_a,
        None,
        Some(&audio_b),
        None,
    );
    assert_generated_cause(
        d,
        "reference_audio_only_CONDITIONING_ROWS",
        &audio_out_a,
        &audio_out_b,
    );
    let valid_audio_layout = ref2va_layout_with_audio_channels(
        &geometry,
        &[TEXT_TAG],
        &[image, audio_geometry],
        &[0, 1],
        cfg.patch_size,
        d,
    )
    .unwrap();
    let valid_audio_a = causal_generated_forward(
        f,
        cfg,
        d,
        valid_audio_layout.clone(),
        &context_a,
        Some(&visual_a),
        Some(&audio_a),
        None,
    );
    let valid_audio_b = causal_generated_forward(
        f,
        cfg,
        d,
        valid_audio_layout,
        &context_a,
        Some(&visual_a),
        Some(&audio_b),
        None,
    );
    assert_generated_cause(
        d,
        "reference_audio_values_visual_fixed",
        &valid_audio_a,
        &valid_audio_b,
    );
    // Post-encoder zero controls keep EVERY layout/position/count fixed. These are deliberate
    // conditioning-drop mutations at the DiT seam, not a bypass of media degeneracy validation.
    let run_reference = |refs: &[ReferenceLatentGeometry],
                         channels: &[usize],
                         visual: Option<&Tensor>,
                         sound: Option<&Tensor>| {
        let layout = ref2va_layout_with_audio_channels(
            &geometry,
            &[TEXT_TAG],
            refs,
            channels,
            cfg.patch_size,
            d,
        )
        .unwrap();
        causal_generated_forward(f, cfg, d, layout, &context_a, visual, sound, None)
    };
    let zero_video = causal_signal(&[1, 8, cfg.video_patch_dim()], 0, 0., d);
    let zero_audio = causal_signal(&[1, 3, cfg.audio_in_channels], 0, 0., d);
    assert_generated_cause(
        d,
        "reference_image_drop_same_layout",
        &ref_a,
        &run_reference(&[image], &[0], Some(&zero_visual), None),
    );
    assert_generated_cause(
        d,
        "reference_video_drop_same_layout",
        &video_out_a,
        &run_reference(&[video_geometry], &[0], Some(&zero_video), None),
    );
    assert_generated_cause(
        d,
        "reference_audio_only_drop_same_layout_CONDITIONING_ROWS",
        &audio_out_a,
        &run_reference(&[audio_geometry], &[1], None, Some(&zero_audio)),
    );
    // Same geometry per carrier on each side: only its encoded VALUES are reassigned to the
    // advancing reference clock. Dropping or sorting the ordered values makes these A/B equal.
    let videos_ab = Tensor::cat(&[&video_a, &video_b], 1).unwrap();
    let videos_ba = Tensor::cat(&[&video_b, &video_a], 1).unwrap();
    let ordered_videos_a = run_reference(
        &[video_geometry, video_geometry],
        &[0, 0],
        Some(&videos_ab),
        None,
    );
    let ordered_videos_b = run_reference(
        &[video_geometry, video_geometry],
        &[0, 0],
        Some(&videos_ba),
        None,
    );
    assert_generated_cause(
        d,
        "ordered_video_references_AB_vs_BA",
        &ordered_videos_a,
        &ordered_videos_b,
    );
    let audios_ab = Tensor::cat(&[&audio_a, &audio_b], 1).unwrap();
    let audios_ba = Tensor::cat(&[&audio_b, &audio_a], 1).unwrap();
    let ordered_audios_a = run_reference(
        &[image, audio_geometry, audio_geometry],
        &[0, 1, 1],
        Some(&visual_a),
        Some(&audios_ab),
    );
    let ordered_audios_b = run_reference(
        &[image, audio_geometry, audio_geometry],
        &[0, 1, 1],
        Some(&visual_a),
        Some(&audios_ba),
    );
    assert_generated_cause(
        d,
        "ordered_audio_references_AB_vs_BA_visual_fixed",
        &ordered_audios_a,
        &ordered_audios_b,
    );
    // A mixed list has one image, one video carrying its own stereo soundtrack and one mono
    // standalone audio reference. Reverse BOTH descriptor order and the corresponding modality
    // streams, never geometry alone. All generated target/context values remain unchanged.
    let video_with_soundtrack = ReferenceLatentGeometry {
        num_audio_latents: 5,
        ..video_geometry
    };
    let soundtrack =
        reference_audio_rows(&causal_signal(&[2, 5, cfg.audio_in_channels], 7, 0.18, d)).unwrap();
    let zero_soundtrack = causal_signal(&[1, 10, cfg.audio_in_channels], 0, 0., d);
    let mixed_video = Tensor::cat(&[&visual_a, &video_a], 1).unwrap();
    let mixed_audio = Tensor::cat(&[&soundtrack, &audio_a], 1).unwrap();
    let mixed = run_reference(
        &[image, video_with_soundtrack, audio_geometry],
        &[0, 2, 1],
        Some(&mixed_video),
        Some(&mixed_audio),
    );
    let reversed_video = Tensor::cat(&[&video_a, &visual_a], 1).unwrap();
    let reversed_audio = Tensor::cat(&[&audio_a, &soundtrack], 1).unwrap();
    let reversed_mixed = run_reference(
        &[audio_geometry, video_with_soundtrack, image],
        &[1, 2, 0],
        Some(&reversed_video),
        Some(&reversed_audio),
    );
    assert_generated_cause(
        d,
        "mixed_image_video_soundtrack_audio_reference_order",
        &mixed,
        &reversed_mixed,
    );
    let no_image = Tensor::cat(&[&zero_visual, &video_a], 1).unwrap();
    assert_generated_cause(
        d,
        "mixed_image_drop_same_layout",
        &mixed,
        &run_reference(
            &[image, video_with_soundtrack, audio_geometry],
            &[0, 2, 1],
            Some(&no_image),
            Some(&mixed_audio),
        ),
    );
    let no_video = Tensor::cat(&[&visual_a, &zero_video], 1).unwrap();
    let no_soundtrack = Tensor::cat(&[&zero_soundtrack, &audio_a], 1).unwrap();
    assert_generated_cause(
        d,
        "mixed_video_carrier_drop_same_layout",
        &mixed,
        &run_reference(
            &[image, video_with_soundtrack, audio_geometry],
            &[0, 2, 1],
            Some(&no_video),
            Some(&no_soundtrack),
        ),
    );
    let no_standalone = Tensor::cat(&[&soundtrack, &zero_audio], 1).unwrap();
    assert_generated_cause(
        d,
        "mixed_standalone_audio_drop_same_layout",
        &mixed,
        &run_reference(
            &[image, video_with_soundtrack, audio_geometry],
            &[0, 2, 1],
            Some(&mixed_video),
            Some(&no_standalone),
        ),
    );
    assert_generated_cause(
        d,
        "mixed_video_soundtrack_drop_same_layout",
        &mixed,
        &run_reference(
            &[image, video_with_soundtrack, audio_geometry],
            &[0, 2, 1],
            Some(&mixed_video),
            Some(&no_soundtrack),
        ),
    );
    // One declared hybrid fixture root, with two distinct imported portions. No stock fallback.
    // The public task partition resolver and production directory loader drive the numeric A/B.
    let dir = tempfile::tempdir().unwrap();
    for task in [MiniMaxH3Task::Fl2va, MiniMaxH3Task::Ref2va] {
        let root = dir.path().join(task.partition());
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("config.json"), f["config"].to_string()).unwrap();
        let mut map = manual_conditioning_map(f, cfg, &Device::Cpu);
        if task == MiniMaxH3Task::Ref2va {
            let bias = map["proj_out.bias"].clone();
            map.insert(
                "proj_out.bias".into(),
                bias.add(&causal_signal(
                    &[cfg.video_patch_dim()],
                    11,
                    0.4,
                    &Device::Cpu,
                ))
                .unwrap(),
            );
        }
        candle_gen::candle_core::safetensors::save(&map, root.join("model.safetensors")).unwrap();
    }
    let run_portion = |portion: &str| {
        let path = dir.path().join(portion);
        let source = candle_gen::candle_core::safetensors::load(
            path.join("model.safetensors"),
            &Device::Cpu,
        )
        .unwrap();
        let actual_bias = source["proj_out.bias"]
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected_map = manual_conditioning_map(f, cfg, &Device::Cpu);
        let expected_bias = if portion == "transformer_ref" {
            expected_map["proj_out.bias"]
                .add(&causal_signal(
                    &[cfg.video_patch_dim()],
                    11,
                    0.4,
                    &Device::Cpu,
                ))
                .unwrap()
        } else {
            expected_map["proj_out.bias"].clone()
        };
        assert_eq!(
            actual_bias,
            expected_bias.to_vec1::<f32>().unwrap(),
            "loaded portion must retain its own parameter identity"
        );
        let model =
            candle_gen_minimax_h3::dit::MiniMaxH3Dit::load_from_dir(&path, d, DType::F32).unwrap();
        causal_generated_forward(
            f,
            cfg,
            d,
            fl2va_layout(
                &geometry,
                &[TEXT_TAG],
                &[KeyframeAnchor::First],
                cfg.patch_size,
                d,
            )
            .unwrap(),
            &context_a,
            Some(&visual_a),
            None,
            Some(model),
        )
    };
    let expected_fl = run_portion("transformer");
    let expected_ref = run_portion("transformer_ref");
    let run_task = |keyframes, references, expected_portion: &str| {
        let task = MiniMaxH3Task::resolve(keyframes, references).unwrap();
        assert_eq!(
            dir.path().join(task.partition()),
            dir.path().join(expected_portion)
        );
        run_portion(task.partition())
    };
    let selected_fl = run_task(true, false, "transformer");
    let selected_ref = run_task(false, true, "transformer_ref");
    assert_eq!(
        selected_fl, expected_fl,
        "hybrid-checkpoint FL task must match explicit FL portion baseline"
    );
    assert_eq!(
        selected_ref, expected_ref,
        "hybrid-checkpoint Ref task must match explicit Ref portion baseline"
    );
    assert_generated_cause(
        d,
        "hybrid_checkpoint_FL_portion_vs_wrong_REF_portion",
        &selected_fl,
        &expected_ref,
    );
    assert_generated_cause(
        d,
        "hybrid_checkpoint_REF_portion_vs_wrong_FL_portion",
        &selected_ref,
        &expected_fl,
    );
    println!("H3_CONDITIONING_CAUSAL_BOUNDARY backend=cuda hybrid_checkpoint=two_distinct_portions selected_by=native_task_partition audio_only_generator=refused prompt_seam=context_embedding");
}
