//! Metal training allocation envelope (MLX v0.31.1, sc-24163).
//!
//! This is deliberately tensor-free, so its structural allocation tests also run on Windows:
//! `rustc --edition 2021 --test src/training_memory.rs -o <temp>/training-memory-tests`.
//! Metal SDPA uses its unfused fallback during grad tracing, and its VJP is also unfused:
//! https://github.com/ml-explore/mlx/blob/v0.31.1/mlx/fast.cpp (SDPA fallback / VJP)
//! https://github.com/ml-explore/mlx/blob/v0.31.1/mlx/primitives.cpp (Softmax::vjp)
//! `Softmax::vjp` holds s, cotangent, s*cotangent, s*sum(s*cotangent), and their difference.
//! The synchronous count alone is insufficient: metal/eval.cpp retains op inputs in completion
//! handlers. transforms.cpp permits ten outstanding command buffers; device.cpp commits after
//! 50 MiB or 50 ops on Max/Ultra, with the allocation crossing the threshold still in that buffer.
//! Price both input and output, the largest score call and the f32 wide SwiGLU stream, plus that
//! buffer threshold, for every outstanding buffer and the one currently being encoded. This is
//! a conservative shape-derived envelope, not a coefficient fitted to the two recovered peaks.

pub const SOFTMAX_VJP_SCORE_BUFFERS: u64 = 5;
pub const MAX_ACTIVE_TASKS: u64 = 10;
pub const BUFFER_INPUT_OUTPUTS: u64 = 2;
// device.cpp compares (buffer_sizes >> 20) > 50: the first triggering byte is 51 MiB.
pub const METAL_BUFFER_BYTES: u64 = 51 << 20;

pub const ATTENTION_SAVED_HIDDEN: u64 = 14;
pub const FFN_SAVED_HIDDEN_FIXED: u64 = 3;
pub const FFN_SAVED_PER_MLP_RATIO: u64 = 4;

#[derive(Clone, Copy)]
pub struct StepShape {
    pub sequence: u64,
    pub inner: u64,
    pub layers: u64,
    pub heads: u64,
    pub mlp_ratio: u64,
    pub width: u64,
    pub target_scores: u64,
    pub prefix_scores: u64,
    pub largest_prefix_call: u64,
    pub lokr_delta_elements: u64,
    pub checkpointed: bool,
}

/// One step's activation envelope; resident weights, optimizer and caches are priced by caller.
pub fn step_bytes(s: StepShape) -> u64 {
    let hidden = s.sequence * s.inner * s.width;
    let block_saved =
        (ATTENTION_SAVED_HIDDEN + FFN_SAVED_HIDDEN_FIXED + FFN_SAVED_PER_MLP_RATIO * s.mlp_ratio)
            * hidden;
    let retained = if s.checkpointed {
        s.layers * hidden + 2 * block_saved
    } else {
        s.layers * block_saved
    };
    let deltas = s.lokr_delta_elements * 2;
    retained
        + attention_backward_bytes(s.heads * (s.target_scores + s.prefix_scores), s.width)
        + pipelined_bytes(
            s.heads * s.target_scores.max(s.largest_prefix_call),
            s.sequence * s.inner,
            s.mlp_ratio,
            s.width,
        )
        + if s.checkpointed {
            2 * deltas / s.layers.max(1)
        } else {
            deltas
        }
}

pub fn attention_backward_bytes(score_elements: u64, compute_width: u64) -> u64 {
    SOFTMAX_VJP_SCORE_BUFFERS * score_elements * compute_width
}

pub fn pipelined_bytes(largest_call: u64, hidden_elements: u64, mlp_ratio: u64, width: u64) -> u64 {
    BUFFER_INPUT_OUTPUTS
        * (MAX_ACTIVE_TASKS + 1)
        * (largest_call * width + hidden_elements * mlp_ratio * 4 + METAL_BUFFER_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_scores_and_retired_buffers_are_both_priced() {
        // One operation, not a borrowed production byte figure.
        assert_eq!(
            attention_backward_bytes(32 * 7 * 11, 2),
            5 * 32 * 7 * 11 * 2
        );
        let pipeline = pipelined_bytes(32 * 7 * 11, 11 * 4096, 3, 2);
        assert_eq!(
            pipeline,
            22 * (32 * 7 * 11 * 2 + 11 * 4096 * 3 * 4 + (51 << 20))
        );
        assert!(pipelined_bytes(32 * 7 * 22, 22 * 4096, 3, 2) > pipeline);
    }

    #[test]
    fn recovered_measured_peaks_fit_structural_envelope() {
        // Exact production model geometry; the old caption is conservatively 64 rows. Both
        // references are fitted to 1024² = 4096 tokens. Run 37127726908's actual active peaks
        // are regression floors, never coefficients in the prediction. Price only the DiT
        // resident set here: optimizer, caches and the evaluation slack can only increase it.
        let base = StepShape {
            sequence: 1024 + 64,
            inner: 4096,
            layers: 32,
            heads: 32,
            mlp_ratio: 3,
            width: 2,
            target_scores: 1024 * (1024 + 64),
            prefix_scores: 64 * 64,
            largest_prefix_call: 64 * 64,
            lokr_delta_elements: 0,
            checkpointed: true,
        };
        let resident = 2 * (7_115_112_448 + 24_576 / 2);
        // Rank-16 LoRA: seven target linears in each block; f32 masters/grad/update + Adam m/v.
        let lora_state = 32 * 16 * (4 * 2 * 4096 + 3 * (4096 + 12288)) * 5 * 4;
        let t2i_cache = 8 * (64 * 4096 + 1024 * 64) * 4;
        let eval_slack = 1 << 30;
        assert!(
            resident + lora_state + t2i_cache + eval_slack + step_bytes(base) >= 21_131_967_008,
            "T2I 19.68 GiB peak"
        );
        let edit = StepShape {
            sequence: 1024 + 64 + 8192,
            target_scores: 1024 * (1024 + 64 + 8192),
            // Lower bounds: omit the text calls, retain both complete reference calls.
            prefix_scores: 4096 * 4096 + 4096 * 8192,
            largest_prefix_call: 4096 * 8192,
            lokr_delta_elements: 32 * (4 * 4096 * 4096 + 3 * 4096 * 12288),
            ..base
        };
        let prediction = resident + eval_slack + step_bytes(edit);
        assert!(prediction >= 79_026_698_990, "two-ref edit 73.60 GiB peak");
        // It remains usable on the 128 GiB Max; a giant constant cannot satisfy this gate.
        assert!(prediction < 100 * (1 << 30));
        assert!(step_bytes(StepShape { width: 4, ..edit }) > step_bytes(edit));
        assert!(
            step_bytes(StepShape {
                checkpointed: false,
                ..edit
            }) > step_bytes(edit)
        );
    }
}
