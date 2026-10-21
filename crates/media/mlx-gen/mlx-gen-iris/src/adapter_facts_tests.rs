//! Production-loader / native-forward witnesses for adapter facts (sc-26270).
//!
//! These tests must run only in their CPU child. Every assertion is source-only until the
//! coordinator admits the native run AND its individual failing mutation / restored green rerun.

use std::collections::{BTreeMap, HashMap};

use mlx_gen::adapters::facts::AdapterFacts;
use mlx_gen::gen_core::iris::{adapter_provenance, IrisTask};
use mlx_gen::weights::Weights;
use mlx_gen::{AdapterKind, AdapterSpec, Quant};
use mlx_rs::{transforms::eval, Array, Dtype};

use super::header_inventory;
use crate::nn::{BackboneQuant, Loader};
use crate::retention::{allocated, Kind, LinearShape, Retention};

fn cpu_child(test: &str, body: fn()) {
    const CHILD: &str = "IRIS_ADAPTER_FACTS_CHILD";
    const DONE: &str = "IRIS_ADAPTER_FACTS_DONE";
    if std::env::var(CHILD).as_deref() == Ok(test) {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        assert_eq!(
            mlx_rs::Device::try_default().expect("read back native CPU default before fixtures"),
            mlx_rs::Device::cpu()
        );
        assert!(crate::test_metal_guard::forbid_metal());
        assert!(crate::test_metal_guard::metal_forbidden());
        assert_eq!(
            unsafe { mlx_sys::mlx_set_default_stream(mlx_rs::Stream::cpu().as_ptr()) },
            0
        );
        assert_eq!(
            mlx_rs::Stream::new().get_index().unwrap(),
            mlx_rs::Stream::cpu().get_index().unwrap()
        );
        // No Array, fixture loader, random key or model is constructed before the above setup.
        body();
        std::fs::write(std::env::var(DONE).unwrap(), b"ok").unwrap();
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let marker = temp.path().join("done");
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture", "--test-threads", "1"])
        .env(CHILD, test)
        .env(DONE, &marker)
        .status()
        .unwrap();
    assert!(status.success(), "{test}: CPU child failed");
    assert!(marker.is_file(), "{test}: CPU child executed no witness");
}

#[test]
fn checked_adapter_load_overflow_is_refused() {
    // HOST-ONLY: these are byte/shape metadata values, never MLX Array fixtures or allocations.
    // Each fault is placed on the exact value the production byte API operates on.
    use super::{
        AdapterFileFacts, AdapterFileReadFacts, AdapterHeaderInventory, AdapterLoadAllocation,
        AdapterReconstructionFacts, ProjectionAdapterLoadFacts,
    };
    use mlx_gen::adapters::facts::TensorFacts;
    let allocation = |shape, dtype, minimum_bytes, retained_output| AdapterLoadAllocation {
        operation: "overflow fixture",
        shape,
        dtype,
        minimum_bytes,
        retained_output,
    };
    // Individual mutants: unchecked shape product; unchecked allocator-rounding multiplication;
    // negative-dimension-to-zero masking. Each must fail its own corresponding assertion.
    assert!(allocation(vec![i32::MAX; 3], Dtype::Float32, 0, false)
        .bytes()
        .is_err());
    assert!(allocation(vec![1], Dtype::Bool, u64::MAX, false)
        .bytes()
        .is_err());
    assert!(allocation(vec![-1, 0], Dtype::Bool, 0, false)
        .bytes()
        .is_err());
    assert_eq!(
        allocation(vec![i32::MAX, i32::MAX, i32::MAX, 0], Dtype::Bool, 0, false)
            .bytes()
            .unwrap(),
        0
    );
    let huge = allocation(vec![i32::MAX, i32::MAX], Dtype::Float32, 0, true);
    huge.bytes().unwrap(); // Each individual request is representable; the TWO-site sum is not.
    let mut projection = ProjectionAdapterLoadFacts {
        retained: vec![
            AdapterFacts::Materialized {
                delta: TensorFacts::materialized_matrix(i32::MAX, i32::MAX, Dtype::Float32),
                disabled: false,
            };
            2
        ],
        allocations: vec![huge.clone(), huge],
        graph_constructor_bytes: 0,
    };
    // Individual mutants: wrap/saturate each production sum instead of returning its error.
    assert!(projection.materialize_scratch_bytes().is_err());
    assert!(projection.new_retained_bytes().is_err());
    assert!(projection.retained_bytes().is_err());
    for a in &mut projection.allocations {
        a.retained_output = false;
    }
    assert!(projection.transient_extra_bytes().is_err());
    assert!(super::host_constructor_bytes(u64::MAX).is_err());
    assert!(AdapterFileReadFacts {
        path: std::path::PathBuf::new(),
        source_bytes: 8,
        file_floor_bytes: 0,
        alpha_read_scratch_bytes: u64::MAX - 7,
        graph_constructor_bytes: 0
    }
    .read_guard_bytes()
    .is_err());

    // All source/output tensor byte counts below fit. Only the native greedy path's NAIVE COST
    // (3 * 2^63) overflows. Mutating that checked cost to wrapping arithmetic must make this
    // specific negative metadata case incorrectly succeed; no native tensor is ever instantiated.
    let n = 1_i32 << 16;
    let sources = BTreeMap::from([
        (
            "p.lokr_w1".to_owned(),
            TensorFacts::matrix(1, 1, Dtype::Float16, false),
        ),
        (
            "p.lokr_t2".to_owned(),
            TensorFacts {
                shape: vec![n, n, 1, 1],
                dtype: Dtype::Uint8,
                strides: vec![n as usize, 1, 1, 1],
            },
        ),
        (
            "p.lokr_w2_a".to_owned(),
            TensorFacts::matrix(n, n, Dtype::Float16, false),
        ),
        (
            "p.lokr_w2_b".to_owned(),
            TensorFacts::matrix(n, n / 2, Dtype::Float16, false),
        ),
    ]);
    let factors = ["lokr_w1", "lokr_t2", "lokr_w2_a", "lokr_w2_b"]
        .into_iter()
        .map(|name| (name.to_owned(), format!("p.{name}")))
        .collect();
    let inventory = AdapterHeaderInventory {
        projections: BTreeMap::from([(
            "p".to_owned(),
            vec![AdapterFacts::Materialized {
                delta: TensorFacts::materialized_matrix(n, n / 2, Dtype::Bfloat16),
                disabled: false,
            }],
        )]),
        files: vec![AdapterFileFacts {
            path: std::path::PathBuf::new(),
            format: mlx_gen::adapters::loader::UnfusedAdapterFormat::ThirdpartyLokr,
            sources,
            scalar_reads: Default::default(),
            reconstructions: vec![AdapterReconstructionFacts::Lokr {
                projection: "p".to_owned(),
                factors,
            }],
        }],
    };
    let error = inventory.projection_loads().unwrap_err().to_string();
    assert!(
        error.contains("Tucker naive cost overflow"),
        "wrong rejection site: {error}"
    );
}

fn tensor(shape: &[i32], dtype: Dtype) -> Array {
    let a = Array::ones::<f32>(shape).unwrap().as_dtype(dtype).unwrap();
    eval([&a]).unwrap();
    a
}

/// Generate a tiny real safetensors file, not a mocked attached residual. Every variant reaches
/// the production strict adapter dispatch through Iris's install and Loader::linear.
fn file(dir: &std::path::Path, variant: &str, dtype: Dtype, rank: i32, scale: f32) -> AdapterSpec {
    let (kind, named): (_, Vec<(&str, Vec<i32>)>) = match variant {
        "peft" => (
            AdapterKind::Lora,
            vec![
                ("p.lora_A.default.weight", vec![rank, 65]),
                ("p.lora_B.default.weight", vec![64, rank]),
                ("p.alpha", vec![1]),
            ],
        ),
        "peft_wide_alpha" => (
            AdapterKind::Lora,
            vec![
                ("p.lora_A.weight", vec![rank, 65]),
                ("p.lora_B.weight", vec![64, rank]),
                ("p.alpha", vec![8192]),
            ],
        ),
        "kohya" => (
            AdapterKind::Lora,
            vec![
                ("lora_unet_p.lora_down.default.weight", vec![rank, 65]),
                ("lora_unet_p.lora_up.default.weight", vec![64, rank]),
            ],
        ),
        "kohya_duplicate" => (
            AdapterKind::Lora,
            vec![
                ("lora_unet_p.lora_down.weight", vec![rank, 65]),
                ("lora_unet_p.lora_down.default.weight", vec![rank, 65]),
                ("lora_unet_p.lora_up.weight", vec![64, rank]),
            ],
        ),
        "lokr" | "lokr_peft" => (
            AdapterKind::Lokr,
            vec![
                ("p.lokr_w1_a", vec![8, 2]),
                ("p.lokr_w1_b", vec![2, 5]),
                ("p.lokr_w2_a", vec![8, 3]),
                ("p.lokr_w2_b", vec![3, 13]),
            ],
        ),
        "mixed_lora" => (
            AdapterKind::Lora,
            vec![
                ("p.lora_A.weight", vec![rank, 65]),
                ("p.lora_B.weight", vec![64, rank]),
            ],
        ),
        "lokr_vector" | "lokr_vector_peft" => (
            AdapterKind::Lokr,
            vec![
                ("p.lokr_w1_a", vec![3]),
                ("p.lokr_w1_b", vec![3]),
                ("p.lokr_w2", vec![64, 65]),
            ],
        ),
        "lokr_batched" => (
            AdapterKind::Lokr,
            vec![
                ("p.lokr_w1_a", vec![1, 8, 2]),
                ("p.lokr_w1_b", vec![1, 2, 5]),
                ("p.lokr_w2", vec![8, 13]),
            ],
        ),
        "lokr_full" => (
            AdapterKind::Lokr,
            vec![("p.lokr_w1", vec![8, 5]), ("p.lokr_w2", vec![8, 13])],
        ),
        "lokr_mixed_broadcast" => (
            AdapterKind::Lokr,
            vec![
                // J-first greedy contraction requires an I64 transpose/reshape copy BEFORE native
                // I64+F16->F16 promotion; w2_b broadcasts its contracted J axis. Target remains64x65.
                ("p.lokr_w1", vec![1, 1]),
                ("p.lokr_t2", vec![128, 128, 2, 2]),
                ("p.lokr_w2_a", vec![128, 65]),
                ("p.lokr_w2_b", vec![1, 16]),
            ],
        ),
        "lokr_tucker_scalar" => (
            AdapterKind::Lokr,
            vec![
                ("p.lokr_w1", vec![64, 65]),
                ("p.lokr_t2", vec![1, 1, 1, 1]),
                ("p.lokr_w2_a", vec![1, 1]),
                ("p.lokr_w2_b", vec![1, 1]),
            ],
        ),
        "lokr_tucker" => (
            AdapterKind::Lokr,
            vec![
                ("p.lokr_w1", vec![8, 5]),
                ("p.lokr_t2", vec![2, 3, 1, 1]),
                ("p.lokr_w2_a", vec![2, 8]),
                ("p.lokr_w2_b", vec![3, 13]),
            ],
        ),
        "loha_tucker" => (
            AdapterKind::Lora,
            vec![
                ("p.hada_t1", vec![2, 3, 1, 1]),
                ("p.hada_t2", vec![4, 5, 1, 1]),
                ("p.hada_w1_a", vec![2, 64]),
                ("p.hada_w1_b", vec![3, 65]),
                ("p.hada_w2_a", vec![4, 64]),
                ("p.hada_w2_b", vec![5, 65]),
            ],
        ),
        "loha" | "loha_lone_t1" | "loha_lone_t2" => {
            let mut factors = vec![
                ("p.hada_w1_a", vec![64, 3]),
                ("p.hada_w1_b", vec![3, 65]),
                ("p.hada_w2_a", vec![64, 2]),
                ("p.hada_w2_b", vec![2, 65]),
            ];
            if variant == "loha_lone_t1" {
                factors.push(("p.hada_t1", vec![1, 1, 1, 1]));
            }
            if variant == "loha_lone_t2" {
                factors.push(("p.hada_t2", vec![1, 1, 1, 1]));
            }
            (AdapterKind::Lora, factors)
        }
        _ => panic!("unknown fixture variant {variant}"),
    };
    let arrays: Vec<_> = named
        .iter()
        .map(|(name, shape)| {
            let dtype = if variant == "lokr_mixed_broadcast" {
                if name.ends_with("lokr_t2") {
                    Dtype::Int64
                } else {
                    Dtype::Float16
                }
            } else if variant == "mixed_lora" && name.contains("lora_B") {
                Dtype::Float16
            } else {
                dtype
            };
            ((*name).to_owned(), tensor(shape, dtype))
        })
        .collect();
    let metadata: HashMap<String, String> = adapter_provenance(IrisTask::Generation, "iris_3b")
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .chain(
            matches!(variant, "lokr_peft" | "lokr_vector_peft")
                .then(|| ("networkType".to_owned(), "lokr".to_owned())),
        )
        .collect();
    let path = dir.join(format!("{variant}-{dtype:?}-{rank}-{scale}.safetensors"));
    Array::save_safetensors(
        arrays.iter().map(|(k, v)| (k.as_str(), v)),
        Some(&metadata),
        &path,
    )
    .unwrap();
    AdapterSpec::new(path, scale, kind)
}

fn base(compute: Dtype) -> Weights {
    Weights::from_map(HashMap::from([
        ("p.weight".to_owned(), tensor(&[64, 65], compute)),
        ("p.bias".to_owned(), tensor(&[64], compute)),
    ]))
}

#[test]
fn generated_headers_match_actual_retained_stacks() {
    cpu_child(
        "adapters::adapter_facts_tests::generated_headers_match_actual_retained_stacks",
        || {
            let temp = tempfile::tempdir().unwrap();
            let shapes = BTreeMap::from([("p".to_owned(), [64, 65])]);
            for compute in [Dtype::Float32, Dtype::Bfloat16] {
                for dtype in [Dtype::Float32, Dtype::Bfloat16, Dtype::Float16] {
                    for quant in [None, Some(Quant::Q8), Some(Quant::Q4)] {
                        let specs: Vec<_> = [
                            "peft",
                            "kohya",
                            "mixed_lora",
                            "lokr",
                            "lokr_peft",
                            "lokr_vector_peft",
                            "lokr_batched",
                            "lokr_full",
                            "lokr_tucker",
                            "lokr_tucker_scalar",
                            "loha",
                            "loha_lone_t1",
                            "loha_lone_t2",
                            "loha_tucker",
                        ]
                        .into_iter()
                        .map(|v| {
                            file(
                                temp.path(),
                                v,
                                dtype,
                                7,
                                if v == "kohya" { 0.0 } else { 0.75 },
                            )
                        })
                        .collect();
                        let headers = header_inventory(
                            &specs,
                            &shapes,
                            compute,
                            IrisTask::Generation,
                            "iris_3b",
                        )
                        .unwrap();
                        let mut weights = base(compute);
                        let (installed, reports) = super::install(
                            &mut weights,
                            compute,
                            &specs,
                            IrisTask::Generation,
                            "iris_3b",
                        )
                        .unwrap()
                        .unwrap();
                        let linear = Loader {
                            weights: &weights,
                            compute,
                            adapted: Some(&installed),
                            quant: BackboneQuant::for_load(quant).unwrap(),
                        }
                        .linear("p", true)
                        .unwrap();
                        // Mutation: remove alpha-fold promotion or a retained stack entry from header_inventory.
                        assert_eq!(
                            headers.projections["p"],
                            linear.adapter_facts(),
                            "{compute:?}/{dtype:?}/{quant:?}"
                        );
                        let actual_deltas: u64 = linear
                            .adapter_facts()
                            .iter()
                            .map(|a| match a {
                                AdapterFacts::Materialized { delta, .. } => delta.elements(),
                                _ => 0,
                            })
                            .sum();
                        // Mutation: restore first-axis low_rank / lone-t1 Tucker inference in the
                        // contract instead of its normalized inventory. Vector/batch/lone/both cases
                        // above must agree with this ACTUAL loaded residual payload, not a mock.
                        assert_eq!(
                            crate::memory_strategy::materialized_delta_elements(&headers).unwrap(),
                            actual_deltas
                        );
                        assert_eq!(reports.len(), specs.len());
                        assert!(installed.unconsumed().is_empty());
                        eval(linear.arrays()).unwrap();
                        let adapter_bytes: u64 = headers.projections["p"]
                            .iter()
                            .flat_map(AdapterFacts::tensors)
                            .map(|t| t.bytes())
                            .sum();
                        let bare = Loader {
                            weights: &weights,
                            compute,
                            adapted: None,
                            quant: BackboneQuant::for_load(quant).unwrap(),
                        }
                        .linear("p", true)
                        .unwrap();
                        let actual: u64 = linear.arrays().iter().map(|a| a.nbytes() as u64).sum();
                        let base: u64 = bare.arrays().iter().map(|a| a.nbytes() as u64).sum();
                        // Mutation: omit adapters from Linear::arrays. Scale-zero payloads must STILL count.
                        assert_eq!(actual - base, adapter_bytes, "retained payload inventory");
                        let x = tensor(&[13, 65], compute);
                        eval([&linear.forward(&x).unwrap()]).unwrap();
                    }
                }
            }
        },
    );
}

#[test]
fn contract_factor_allowance_floors_at_actual_promoted_lora_payloads() {
    cpu_child("adapters::adapter_facts_tests::contract_factor_allowance_floors_at_actual_promoted_lora_payloads", || {
        let temp=tempfile::tempdir().unwrap();
        let spec=file(temp.path(),"peft",Dtype::Bfloat16,193,0.75);
        let headers=header_inventory(std::slice::from_ref(&spec),&BTreeMap::from([("p".to_owned(),[64,65])]),Dtype::Bfloat16,IrisTask::Generation,"iris_3b").unwrap();
        let file_allowance=mlx_gen::gen_core::adapter_stack_resident_bytes(std::slice::from_ref(&spec),mlx_gen::gen_core::AdapterResidencyMode::Additive).unwrap();
        let mut weights=base(Dtype::Bfloat16);
        let (installed,_)=super::install(&mut weights,Dtype::Bfloat16,&[spec],IrisTask::Generation,"iris_3b").unwrap().unwrap();
        let linear=Loader {weights:&weights,compute:Dtype::Bfloat16,adapted:Some(&installed),quant:None}.linear("p",true).unwrap();
        let actual:u64=linear.adapter_facts().iter().flat_map(AdapterFacts::tensors).map(|t|t.bytes()).sum();
        // Mutation: remove max(normalized retained) from the PRODUCTION contract helper. The
        // optional alpha fold's strong F32 B is measurably larger than this BF16 file allowance.
        assert!(actual>file_allowance,"fixture must discriminate the file-only undercount");
        assert_eq!(crate::memory_strategy::adapter_factor_bytes(file_allowance,&headers).unwrap(),actual);
        assert_eq!(crate::memory_strategy::retained_lora_bytes(&headers).unwrap(),actual);
    });
}

#[test]
fn production_lora_forward_peak_requires_rank_and_dtype_terms() {
    cpu_child(
        "adapters::adapter_facts_tests::production_lora_forward_peak_requires_rank_and_dtype_terms",
        || {
            let temp = tempfile::tempdir().unwrap();
            for compute in [Dtype::Float32, Dtype::Bfloat16] {
                for quant in [None, Some(Quant::Q8), Some(Quant::Q4)] {
                    let specs = [
                        file(temp.path(), "peft", Dtype::Float32, 384, 0.5),
                        file(temp.path(), "kohya", Dtype::Bfloat16, 193, 0.25),
                    ];
                    let mut weights = base(compute);
                    let (installed, _) = super::install(
                        &mut weights,
                        compute,
                        &specs,
                        IrisTask::Generation,
                        "iris_3b",
                    )
                    .unwrap()
                    .unwrap();
                    let linear = Loader {
                        weights: &weights,
                        compute,
                        adapted: Some(&installed),
                        quant: BackboneQuant::for_load(quant).unwrap(),
                    }
                    .linear("p", true)
                    .unwrap();
                    eval(linear.arrays()).unwrap();
                    let x = tensor(&[128, 65], compute);
                    let facts = linear.adapter_facts();
                    let mut r = Retention::on(crate::nn::dtype_bytes(compute), true);
                    r.adapted_linear(
                        LinearShape {
                            rows: 128,
                            input: 65,
                            output: 64,
                        },
                        false,
                        linear.input_pad(),
                        Kind::Named,
                        &facts,
                    );
                    let formula = r.price();
                    let measured =
                        mlx_gen::ios_quantum::measured_peak_bytes(|| Ok(vec![linear.forward(&x)?]))
                            .unwrap();
                    eprintln!("adapter native lower bound {compute:?}/{quant:?}: formula={formula} measured={measured}");
                    // Mutation (run individually): remove LoRA rows*rank intermediate from adapter_residual;
                    // then separately remove the activation/factor promotion casts. No print-only bypass.
                    assert!(
                        formula >= measured,
                        "adapter formula {formula} under native peak {measured}"
                    );
                    if formula as f64 > 1.5 * measured.max(1) as f64 {
                        eprintln!("REPORT ONLY: adapter allocation ratio exceeds 1.5");
                    }
                    // This witness really exercises the missing-adapter defect class, not just plumbing:
                    // an unadapted-only budget must be below the native adapted forward peak.
                    let mut missing = Retention::on(crate::nn::dtype_bytes(compute), true);
                    missing.linear(128, 65, 64, false, linear.input_pad(), Kind::Named);
                    assert!(
                        measured > missing.price(),
                        "native witness did not discriminate missing adapter scratch"
                    );
                    // Mutation: remove a retained parameter contribution from the actual inventory.
                    let owned: u64 = facts
                        .iter()
                        .flat_map(AdapterFacts::tensors)
                        .map(|t| allocated(t.bytes()))
                        .sum();
                    assert!(owned >= allocated(384 * 65 * 4) + allocated(384 * 64 * 4));
                }
            }
        },
    );
}

#[test]
fn format_specific_vector_gate_matches_actual_strict_loader() {
    cpu_child(
        "adapters::adapter_facts_tests::format_specific_vector_gate_matches_actual_strict_loader",
        || {
            let temp = tempfile::tempdir().unwrap();
            let shapes = BTreeMap::from([("p".to_owned(), [64, 65])]);
            let refused = file(temp.path(), "lokr_vector", Dtype::Float16, 3, 0.75);
            let header_error = header_inventory(
                std::slice::from_ref(&refused),
                &shapes,
                Dtype::Bfloat16,
                IrisTask::Generation,
                "iris_3b",
            )
            .unwrap_err()
            .to_string();
            let mut weights = base(Dtype::Bfloat16);
            let mut host =
                super::ProjectionHost::from_weights(&mut weights, Dtype::Bfloat16).unwrap();
            let native_error =
                mlx_gen::adapters::loader::apply_adapters_strict(&mut host, &[refused], "iris_3b")
                    .unwrap_err()
                    .to_string();
            // Mutation: remove the SHARED thirdparty_lokr_rank len<2 gate. Both the pure header
            // and actual strict runtime must reject this TP format, rather than porting vector support.
            assert!(
                header_error.contains("lokr_w1_a factor must be 2-D"),
                "{header_error}"
            );
            assert!(
                native_error.contains("lokr_w1_a factor must be 2-D"),
                "{native_error}"
            );
            // This same factorization has a DIFFERENT existing format path when PEFT-stamped: global
            // metadata rank, not the third-party rank-source gate. Positive validity still depends on
            // reaching and successfully materializing the real loader, not native matmul in isolation.
            let peft = file(temp.path(), "lokr_vector_peft", Dtype::Float16, 3, 0.75);
            let headers = header_inventory(
                std::slice::from_ref(&peft),
                &shapes,
                Dtype::Bfloat16,
                IrisTask::Generation,
                "iris_3b",
            )
            .unwrap();
            headers.projection_loads().unwrap();
            let (installed, _) = super::install(
                &mut weights,
                Dtype::Bfloat16,
                &[peft],
                IrisTask::Generation,
                "iris_3b",
            )
            .unwrap()
            .unwrap();
            let linear = Loader {
                weights: &weights,
                compute: Dtype::Bfloat16,
                adapted: Some(&installed),
                quant: None,
            }
            .linear("p", true)
            .unwrap();
            eval(linear.arrays()).unwrap();
            assert_eq!(headers.projections["p"], linear.adapter_facts());
        },
    );
}

#[test]
fn generation_semantic_export_is_normalized_and_path_value_free() {
    cpu_child("adapters::adapter_facts_tests::generation_semantic_export_is_normalized_and_path_value_free", || {
        let temp=tempfile::tempdir().unwrap();
        let a=tensor(&[193,48],Dtype::Bfloat16);
        let b=tensor(&[64,193],Dtype::Bfloat16);
        let metadata:HashMap<_,_>=adapter_provenance(IrisTask::Generation,"iris_3b").into_iter().map(|(k,v)|(k.to_owned(),v)).collect();
        let write=|name:&str,value:f32| {
            let path=temp.path().join(name);
            let alpha=Array::from_slice(&[value],&[1]).as_dtype(Dtype::Bfloat16).unwrap();
            Array::save_safetensors([
                ("s_embedder.proj.lora_A.weight",&a),("s_embedder.proj.lora_B.weight",&b),
                ("s_embedder.proj.alpha",&alpha),
            ],Some(&metadata),&path).unwrap();
            path
        };
        let fixtures=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-snapshot");
        let mut spec=mlx_gen::LoadSpec::new(mlx_gen::gen_core::WeightsSource::Dir(fixtures.join("iris")));
        spec.components.insert(mlx_gen::gen_core::iris::TEXT_ENCODER_COMPONENT.to_owned(),
            mlx_gen::gen_core::WeightsSource::Dir(fixtures.join("text_encoder")));
        spec.adapters=vec![AdapterSpec::new(write("alpha-one.safetensors",1.0),0.25,AdapterKind::Lora)];
        let first=super::generation_adapter_semantic_facts(&spec).unwrap();
        spec.adapters=vec![AdapterSpec::new(write("alpha-two.safetensors",2.0),0.75,AdapterKind::Lora)];
        let second=super::generation_adapter_semantic_facts(&spec).unwrap();
        // Mutations: include file.path or exact continuous scale / scalar values in the DTO.
        assert_eq!(first,second,"path/value changes with identical allocation semantics must not change named facts");
        assert_eq!(first["schema"],"iris_generation_adapter_semantics_v1");
        // Mutation: serialize raw BF16 B instead of its actual strong-alpha-fold F32 payload.
        assert_eq!(first["projections"][0]["residual_stack"][0]["b"]["dtype"],"F32");
        assert!(first["files"][0]["reconstructions"][0]["alpha_fold"].as_bool().unwrap());
        assert!(!serde_json::to_string(&first).unwrap().contains(temp.path().to_str().unwrap()));
        spec.adapters[0].scale=0.0;
        let disabled=super::generation_adapter_semantic_facts(&spec).unwrap();
        // Mutation: omit the ACTUAL scale-zero graph branch or drop its retained payload facts.
        assert_ne!(first,disabled);
        assert!(disabled["projections"][0]["residual_stack"][0]["disabled"].as_bool().unwrap());
        assert_eq!(disabled["projections"][0]["residual_stack"][0]["b"]["dtype"],"F32");
    });
}

#[test]
fn actual_file_guard_covers_full_alpha_cast_before_read() {
    cpu_child(
        "adapters::adapter_facts_tests::actual_file_guard_covers_full_alpha_cast_before_read",
        || {
            use std::{cell::Cell, rc::Rc};
            let temp = tempfile::tempdir().unwrap();
            let spec = file(temp.path(), "peft_wide_alpha", Dtype::Bfloat16, 1, 1.0);
            let mut weights = base(Dtype::Bfloat16);
            let checked = Rc::new(Cell::new(None));
            let witness = checked.clone();
            let measured = mlx_gen::ios_quantum::with_cpu_diagnostic_headroom(1 << 34, || {
                mlx_gen::ios_quantum::with_cpu_diagnostic_observer(
                    move |event| {
                        if let mlx_gen::ios_quantum::QuantumEvent::Checked { label, scratch } =
                            event
                        {
                            if label == super::ADAPTER_LOAD_QUANTUM {
                                witness.set(Some(*scratch));
                            }
                        }
                    },
                    || {
                        mlx_gen::ios_quantum::measured_peak_bytes(|| {
                            let installed = super::install(
                                &mut weights,
                                Dtype::Bfloat16,
                                std::slice::from_ref(&spec),
                                IrisTask::Generation,
                                "iris_3b",
                            )?;
                            drop(installed);
                            Ok(Vec::new())
                        })
                    },
                )
            })
            .unwrap();
            // Mutations: remove the ACTUAL file guard; then separately remove selected alpha's full
            // F32 cast term from file_read_facts. The 8192-element alpha is legal: scalar_alpha reads
            // its first value but as_dtype casts the ENTIRE tensor before that read.
            let formula = checked
                .get()
                .expect("actual Iris adapter file boundary emitted no guard");
            assert!(
                formula >= measured,
                "actual file guard {formula} < native read/cast {measured}"
            );
            if formula as f64 > 1.5 * measured.max(1) as f64 {
                eprintln!("REPORT ONLY: adapter file ratio exceeds 1.5");
            }
        },
    );
}

#[test]
fn actual_materialization_guard_covers_lazy_reconstruction_peak() {
    cpu_child("adapters::adapter_facts_tests::actual_materialization_guard_covers_lazy_reconstruction_peak", || {
        use std::{cell::Cell,rc::Rc};
        let temp=tempfile::tempdir().unwrap();
        for variant in ["peft","mixed_lora","lokr","lokr_vector_peft","lokr_batched","lokr_full","lokr_tucker","lokr_tucker_scalar","loha","loha_lone_t1","loha_lone_t2","loha_tucker","lokr_mixed_broadcast"] {
            let spec=file(temp.path(),variant,Dtype::Float16,193,0.75);
            let mut weights=base(Dtype::Bfloat16);
            let (installed,_)=super::install(&mut weights,Dtype::Bfloat16,std::slice::from_ref(&spec),IrisTask::Generation,"iris_3b").unwrap().unwrap();
            let checked=Rc::new(Cell::new(None));
            let witness=checked.clone();
            let measured=mlx_gen::ios_quantum::with_cpu_diagnostic_headroom(1<<34,|| {
                mlx_gen::ios_quantum::with_cpu_diagnostic_observer(move |event| {
                    if let mlx_gen::ios_quantum::QuantumEvent::Checked {label,scratch}=event {
                        if label==super::ADAPTER_MATERIALIZE_QUANTUM {witness.set(Some(*scratch));}
                    }
                },||mlx_gen::ios_quantum::measured_peak_bytes(|| {
                    let linear=Loader {weights:&weights,compute:Dtype::Bfloat16,adapted:Some(&installed),quant:None}.linear("p",true)?;
                    Ok(linear.arrays().into_iter().cloned().collect())
                }))
            }).unwrap();
            // Mutations: delete Loader's materialize guard (must fail presence); separately replace
            // projection.materialize_scratch_bytes with zero (must fail native hard lower bound).
            let formula=checked.get().expect("actual lazy adapter materialization emitted no guard");
            eprintln!("ADAPTER_MATERIALIZE {variant}: actual_guard={formula} native_peak={measured}");
            assert!(formula>=measured,"{variant}: actual reconstruction guard {formula} < native {measured}");
            let headers=header_inventory(&[spec],&BTreeMap::from([("p".to_owned(),[64,65])]),Dtype::Bfloat16,IrisTask::Generation,"iris_3b").unwrap();
            let loads=headers.projection_loads().unwrap();
            assert_eq!(formula,loads["p"].materialize_scratch_bytes().unwrap(),"same header/runtime operation law");
            assert_eq!(formula,loads["p"].new_retained_bytes().unwrap()+loads["p"].transient_extra_bytes().unwrap());
            if variant=="lokr_mixed_broadcast" {
                // Counterfactual OLD raw law priced the pre-cast I64 reshape copy at promoted
                // F16 width. Compare that actual defect model to the native measured operation,
                // not a committed machine number or a tolerance. This is not a physical basis.
                let missing_copy_bytes:u64=loads["p"].allocations.iter().filter(|a|
                    a.operation=="Tucker transpose_reshape/matmul operand copy upper" && a.dtype==Dtype::Int64)
                    .map(|a| {
                        let mut old=a.clone(); old.dtype=Dtype::Float16;
                        a.bytes().unwrap().checked_sub(old.bytes().unwrap()).unwrap()
                    }).sum();
                let legacy=formula.checked_sub(missing_copy_bytes).unwrap();
                eprintln!("ADAPTER_MATERIALIZE {variant}: old_promoted_copy_budget={legacy}");
                assert!(missing_copy_bytes>0 && measured>legacy,"fixture must discriminate original-width pre-cast copy: native={measured}, old={legacy}");
            }
            if formula as f64>1.5*measured.max(1) as f64 {eprintln!("REPORT ONLY: {variant} reconstruction ratio exceeds 1.5");}
        }
    });
}

#[test]
fn actual_parent_embed_guard_prices_installed_projection() {
    cpu_child(
        "adapters::adapter_facts_tests::actual_parent_embed_guard_prices_installed_projection",
        || {
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("patch-lora.safetensors");
            let a = tensor(&[1536, 48], Dtype::Float32);
            let b = tensor(&[64, 1536], Dtype::Float32);
            let metadata: HashMap<_, _> = adapter_provenance(IrisTask::Generation, "iris_3b")
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v))
                .collect();
            Array::save_safetensors(
                [
                    ("s_embedder.proj.lora_A.weight", &a),
                    ("s_embedder.proj.lora_B.weight", &b),
                ],
                Some(&metadata),
                &path,
            )
            .unwrap();
            let spec = AdapterSpec::new(path, 0.25, AdapterKind::Lora);
            let backbone = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/tiny-snapshot/iris");
            let config = mlx_gen::gen_core::iris::IrisConfig::from_dir(&backbone).unwrap();
            let (dit, _) = crate::model::load_backbone_with_adapters(
                &backbone,
                &config,
                Dtype::Float32,
                &[spec],
                IrisTask::Generation,
                "iris_3b",
            )
            .unwrap();
            eval(dit.arrays()).unwrap();
            let x = tensor(&[2, 3, 16, 16], Dtype::Float32);
            let t = tensor(&[2], Dtype::Float32);
            // This is the production parent guard's actual getter, not an adapter helper called only
            // by a test. Removing the installed inventory from the DiT, or dropping with_adapters in
            // embed_scratch, must red the native hard lower bound below.
            let formula = dit.embed_scratch(2, 16);
            let measured = mlx_gen::ios_quantum::measured_peak_bytes(|| {
                let embedded = dit.embed_body(&x, &t)?;
                let mut arrays = vec![embedded.s, embedded.t_emb, embedded.core_img];
                arrays.extend(embedded.core_txt);
                Ok(arrays)
            })
            .unwrap();
            assert!(
                formula >= measured,
                "actual embed guard {formula} < native {measured}"
            );
            if formula as f64 > 1.5 * measured.max(1) as f64 {
                eprintln!("REPORT ONLY: actual adapted embed ratio exceeds 1.5");
            }
        },
    );
}

#[test]
fn materialized_lokr_loha_native_peak_requires_adapter_budget() {
    cpu_child(
        "adapters::adapter_facts_tests::materialized_lokr_loha_native_peak_requires_adapter_budget",
        || {
            let temp = tempfile::tempdir().unwrap();
            for variant in ["lokr", "lokr_tucker", "loha", "loha_tucker"] {
                for quant in [None, Some(Quant::Q8), Some(Quant::Q4)] {
                    let specs = [file(temp.path(), variant, Dtype::Float16, 3, 0.5)];
                    let mut weights = base(Dtype::Float32);
                    let (installed, _) = super::install(
                        &mut weights,
                        Dtype::Float32,
                        &specs,
                        IrisTask::Generation,
                        "iris_3b",
                    )
                    .unwrap()
                    .unwrap();
                    let linear = Loader {
                        weights: &weights,
                        compute: Dtype::Float32,
                        adapted: Some(&installed),
                        quant: BackboneQuant::for_load(quant).unwrap(),
                    }
                    .linear("p", true)
                    .unwrap();
                    eval(linear.arrays()).unwrap();
                    let x = tensor(&[512, 65], Dtype::Float32);
                    let facts = linear.adapter_facts();
                    let mut r = Retention::on(4, true);
                    r.adapted_linear(
                        LinearShape {
                            rows: 512,
                            input: 65,
                            output: 64,
                        },
                        false,
                        linear.input_pad(),
                        Kind::Named,
                        &facts,
                    );
                    let formula = r.price();
                    let measured =
                        mlx_gen::ios_quantum::measured_peak_bytes(|| Ok(vec![linear.forward(&x)?]))
                            .unwrap();
                    // Mutation: replace actual Materialized adapter facts with an empty stack at this
                    // production-loader/forward witness's counted projection. No mocked residual.
                    assert!(
                        formula >= measured,
                        "{variant}/{quant:?}: {formula} < {measured}"
                    );
                    let mut missing = Retention::on(4, true);
                    missing.linear(512, 65, 64, false, linear.input_pad(), Kind::Named);
                    assert!(measured > missing.price(), "{variant}/{quant:?}: native witness did not discriminate missing adapter scratch");
                    if formula as f64 > 1.5 * measured.max(1) as f64 {
                        eprintln!(
                            "REPORT ONLY: {variant}/{quant:?} adapter allocation ratio exceeds 1.5"
                        );
                    }
                }
            }
        },
    );
}

#[test]
fn kohya_ambiguous_roles_fail_independent_of_key_order() {
    use mlx_gen::adapters::loader::{claim_lora_factor, LoraFactorRole};
    let keys = [
        "lora_unet_p.lora_down.default.weight",
        "lora_unet_p.lora_down.weight",
    ];
    let mut errors = Vec::new();
    for order in [keys, [keys[1], keys[0]]] {
        let mut claims = BTreeMap::new();
        claim_lora_factor(&mut claims, "p", LoraFactorRole::Down, order[0]).unwrap();
        errors.push(
            claim_lora_factor(&mut claims, "p", LoraFactorRole::Down, order[1])
                .unwrap_err()
                .to_string(),
        );
    }
    // Mutation: allow overwrite in shared claim_lora_factor; separately reverse its error ordering.
    assert_eq!(errors[0], errors[1]);
    assert!(keys.iter().all(|key| errors[0].contains(key)));
    cpu_child(
        "adapters::adapter_facts_tests::kohya_ambiguous_roles_fail_independent_of_key_order",
        || {
            let temp = tempfile::tempdir().unwrap();
            let spec = file(temp.path(), "kohya_duplicate", Dtype::Float32, 3, 1.0);
            let shapes = BTreeMap::from([("p".to_owned(), [64, 65])]);
            let header_error = header_inventory(
                std::slice::from_ref(&spec),
                &shapes,
                Dtype::Float32,
                IrisTask::Generation,
                "iris_3b",
            )
            .unwrap_err()
            .to_string();
            let mut weights = base(Dtype::Float32);
            let native_error = match super::install(
                &mut weights,
                Dtype::Float32,
                &[spec],
                IrisTask::Generation,
                "iris_3b",
            ) {
                Err(error) => error.to_string(),
                Ok(_) => panic!("production loader accepted ambiguous kohya aliases"),
            };
            for key in [
                "lora_unet_p.lora_down.default.weight",
                "lora_unet_p.lora_down.weight",
            ] {
                assert!(header_error.contains(key), "{header_error}");
                assert!(native_error.contains(key), "{native_error}");
            }
        },
    );
}

#[test]
fn scale_zero_keeps_payloads_but_executes_no_residual() {
    cpu_child(
        "adapters::adapter_facts_tests::scale_zero_keeps_payloads_but_executes_no_residual",
        || {
            let temp = tempfile::tempdir().unwrap();
            let specs = [
                file(temp.path(), "peft", Dtype::Float32, 31, 0.0),
                file(temp.path(), "lokr", Dtype::Float32, 1, 0.0),
            ];
            let mut weights = base(Dtype::Float32);
            let (installed, _) = super::install(
                &mut weights,
                Dtype::Float32,
                &specs,
                IrisTask::Generation,
                "iris_3b",
            )
            .unwrap()
            .unwrap();
            let linear = Loader {
                weights: &weights,
                compute: Dtype::Float32,
                adapted: Some(&installed),
                quant: None,
            }
            .linear("p", true)
            .unwrap();
            let bare = Loader {
                weights: &weights,
                compute: Dtype::Float32,
                adapted: None,
                quant: None,
            }
            .linear("p", true)
            .unwrap();
            let facts = linear.adapter_facts();
            let mut actual = Retention::on(4, true);
            actual.adapted_linear(
                LinearShape {
                    rows: 17,
                    input: 65,
                    output: 64,
                },
                false,
                0,
                Kind::Named,
                &facts,
            );
            let mut unchanged = Retention::on(4, true);
            unchanged.linear(17, 65, 64, false, 0, Kind::Named);
            // Mutation: stop skipping disabled residual facts in adapted_linear.
            assert_eq!(actual.price(), unchanged.price());
            let x = tensor(&[17, 65], Dtype::Float32);
            let out = linear.forward(&x).unwrap();
            let want = bare.forward(&x).unwrap();
            eval([&out, &want]).unwrap();
            assert_eq!(out.as_slice::<f32>(), want.as_slice::<f32>());
        },
    );
}
