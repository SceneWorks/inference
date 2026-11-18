# NVFP4 -> bf16 on the CPU (SW-7099). Scratch only here: the Windows job publishes the bf16 tier.
"$CARGO_TARGET_DIR/release/examples/modelopt_nvfp4_to_bf16" "$MINIMAX_PE_SNAPSHOT" "$PUBLISH_SCRATCH/pe/bf16"
