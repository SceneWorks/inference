//! SW-6140 AC3: the RoPE/modulate block micro-bench (`examples/rope_modulate_block_bench.rs`,
//! included verbatim) as a dispatch-only model proof, so it runs on an Apple Silicon Mac through
//! `model-inference-proofs.yml` (`runner: "nax"`). Never part of PR/push CI (`#[ignore]`). The
//! bench prints its own table (median ms/iter, MLX peak MiB, max rel diff, chip brand string).

#[path = "../examples/rope_modulate_block_bench.rs"]
mod bench;

#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn rope_modulate_block_bench() {
    bench::main();
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED rope_modulate_block_bench_proof::rope_modulate_block_bench"
    );
}
