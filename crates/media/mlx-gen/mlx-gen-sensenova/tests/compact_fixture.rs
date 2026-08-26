//! Shared loader for the losslessly partitioned SenseNova synthetic parity fixtures.

use mlx_gen::weights::Weights;

const COMMON: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sensenova_common_golden.safetensors"
);

/// Load one case's metadata and reconstruct its original tensor map from the shared model weights
/// plus its case-specific inputs and expected outputs.
pub(crate) fn load(case: &str) -> (Weights, Weights) {
    let common = Weights::from_file(COMMON).expect("load shared SenseNova fixture weights");
    let case = Weights::from_file(case).expect("load SenseNova fixture case");
    let mut merged = Weights::empty();
    for source in [&common, &case] {
        for key in source.keys() {
            assert!(
                merged.get(key).is_none(),
                "compact SenseNova fixtures must have disjoint tensor keys: {key}"
            );
            merged.insert(key, source.require(key).unwrap().clone());
        }
    }
    (merged, case)
}
