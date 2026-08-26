// Backend-neutral SAM3 box-prompt validation cases (sc-21669).
//
// Both MLX and Candle include this table in their geometry unit tests so their
// request-boundary contract cannot drift.

pub(super) enum ExpectedBoxPromptResult {
    Accept { n: usize },
    Reject { message: &'static str },
}

pub(super) struct BoxPromptCase {
    pub(super) name: &'static str,
    pub(super) shape: &'static [usize],
    pub(super) values: &'static [f32],
    pub(super) labels: &'static [i32],
    pub(super) expected: ExpectedBoxPromptResult,
}

pub(super) const BOX_PROMPT_CASES: &[BoxPromptCase] = &[
    BoxPromptCase {
        name: "valid_two_boxes",
        shape: &[1, 2, 4],
        values: &[0.5, 0.5, 0.2, 0.2, 0.0, 1.0, 1.0, 0.0],
        labels: &[1, 0],
        expected: ExpectedBoxPromptResult::Accept { n: 2 },
    },
    BoxPromptCase {
        name: "wrong_last_dimension",
        shape: &[1, 2, 2],
        values: &[0.5; 4],
        labels: &[1, 1],
        expected: ExpectedBoxPromptResult::Reject {
            message: "[1, n, 4]",
        },
    },
    BoxPromptCase {
        name: "wrong_batch_dimension",
        shape: &[2, 1, 4],
        values: &[0.5; 8],
        labels: &[1],
        expected: ExpectedBoxPromptResult::Reject {
            message: "[1, n, 4]",
        },
    },
    BoxPromptCase {
        name: "wrong_rank",
        shape: &[1, 4],
        values: &[0.5; 4],
        labels: &[1],
        expected: ExpectedBoxPromptResult::Reject {
            message: "[1, n, 4]",
        },
    },
    BoxPromptCase {
        name: "zero_boxes",
        shape: &[1, 0, 4],
        values: &[],
        labels: &[],
        expected: ExpectedBoxPromptResult::Reject {
            message: "[1, n, 4]",
        },
    },
    BoxPromptCase {
        name: "mismatched_label_count",
        shape: &[1, 1, 4],
        values: &[0.5, 0.5, 0.2, 0.2],
        labels: &[1, 0],
        expected: ExpectedBoxPromptResult::Reject {
            message: "label(s)",
        },
    },
    BoxPromptCase {
        name: "negative_label",
        shape: &[1, 1, 4],
        values: &[0.5, 0.5, 0.2, 0.2],
        labels: &[-1],
        expected: ExpectedBoxPromptResult::Reject {
            message: "out of range",
        },
    },
    BoxPromptCase {
        name: "out_of_range_label",
        shape: &[1, 1, 4],
        values: &[0.5, 0.5, 0.2, 0.2],
        labels: &[2],
        expected: ExpectedBoxPromptResult::Reject {
            message: "out of range",
        },
    },
    BoxPromptCase {
        name: "nan_coordinate",
        shape: &[1, 1, 4],
        values: &[0.5, f32::NAN, 0.2, 0.2],
        labels: &[1],
        expected: ExpectedBoxPromptResult::Reject {
            message: "finite and in [0, 1]",
        },
    },
    BoxPromptCase {
        name: "infinite_coordinate",
        shape: &[1, 1, 4],
        values: &[0.5, f32::INFINITY, 0.2, 0.2],
        labels: &[1],
        expected: ExpectedBoxPromptResult::Reject {
            message: "finite and in [0, 1]",
        },
    },
    BoxPromptCase {
        name: "negative_coordinate",
        shape: &[1, 1, 4],
        values: &[-0.1, 0.5, 0.2, 0.2],
        labels: &[1],
        expected: ExpectedBoxPromptResult::Reject {
            message: "finite and in [0, 1]",
        },
    },
    BoxPromptCase {
        name: "oversized_extent",
        shape: &[1, 1, 4],
        values: &[0.5, 0.5, 1_000_000.0, 0.2],
        labels: &[1],
        expected: ExpectedBoxPromptResult::Reject {
            message: "finite and in [0, 1]",
        },
    },
];
