//! Seeded synthetic checkpoints for the crate's unit suites (story sc-24434): a builder that fills
//! a [`Weights`] map with small random tensors, and a word-level tokenizer, so a provider can be
//! driven end to end on a shape-valid fixture without real weights.

use std::collections::HashMap;

use mlx_rs::Array;

use core_llm::Tokenizer;

use crate::primitives::sampler::{SplitMix64, TokenRng};
use crate::primitives::Weights;

/// A seeded synthetic checkpoint under construction.
pub(crate) struct Synth {
    rng: SplitMix64,
    map: HashMap<String, Array>,
}

impl Synth {
    /// An empty checkpoint whose random tensors are drawn from `seed`.
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            rng: SplitMix64::new(seed),
            map: HashMap::new(),
        }
    }

    /// A uniform `[-0.4, 0.4)` tensor.
    pub(crate) fn randn(&mut self, key: impl Into<String>, shape: &[i32]) -> &mut Self {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| (self.rng.next_f32() - 0.5) * 0.8).collect();
        self.map.insert(key.into(), Array::from_slice(&data, shape));
        self
    }

    /// A constant tensor.
    pub(crate) fn fill(&mut self, key: impl Into<String>, shape: &[i32], value: f32) -> &mut Self {
        let n: i32 = shape.iter().product();
        self.map.insert(
            key.into(),
            Array::from_slice(&vec![value; n as usize], shape),
        );
        self
    }

    /// A LayerNorm pair under `prefix`: `weight` ones, `bias` zeros.
    pub(crate) fn layer_norm(&mut self, prefix: &str, width: i32) -> &mut Self {
        self.fill(format!("{prefix}.weight"), &[width], 1.0).fill(
            format!("{prefix}.bias"),
            &[width],
            0.0,
        )
    }

    /// A biased linear `[out, in]` under `prefix`.
    pub(crate) fn linear(&mut self, prefix: &str, out: i32, input: i32) -> &mut Self {
        self.randn(format!("{prefix}.weight"), &[out, input])
            .randn(format!("{prefix}.bias"), &[out])
    }

    /// Scale the tensor at `key` by `factor` (sharpens a fixture's attention so its output
    /// depends on positions).
    pub(crate) fn scale(&mut self, key: &str, factor: f32) -> &mut Self {
        let a = &self.map[key];
        let shape = a.shape().to_vec();
        let data: Vec<f32> = a.as_slice::<f32>().iter().map(|x| x * factor).collect();
        self.map
            .insert(key.to_string(), Array::from_slice(&data, &shape));
        self
    }

    /// Zero row `row` of the 2-D tensor at `key`, so a tied head scores that id exactly `0`.
    pub(crate) fn zero_row(&mut self, key: &str, row: i32) -> &mut Self {
        let a = &self.map[key];
        let shape = a.shape().to_vec();
        let mut data = a.as_slice::<f32>().to_vec();
        let width = shape[1] as usize;
        data[row as usize * width..(row as usize + 1) * width].fill(0.0);
        self.map
            .insert(key.to_string(), Array::from_slice(&data, &shape));
        self
    }

    /// The finished checkpoint.
    pub(crate) fn weights(&mut self) -> Weights {
        Weights::from_map(std::mem::take(&mut self.map))
    }
}

/// A whitespace word-level tokenizer over `t0 … t{vocab-1}` (unknown words are `t0`), plus
/// `extra` words at explicit ids.
pub(crate) fn word_tokenizer(vocab: usize, extra: &[(&str, u32)]) -> Tokenizer {
    let mut entries: Vec<String> = (0..vocab).map(|i| format!("\"t{i}\": {i}")).collect();
    entries.extend(extra.iter().map(|(word, id)| format!("\"{word}\": {id}")));
    Tokenizer::from_json(&format!(
        r#"{{"version": "1.0", "added_tokens": [], "normalizer": null,
            "pre_tokenizer": {{ "type": "Whitespace" }}, "post_processor": null,
            "decoder": null,
            "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }} }}"#,
        entries.join(", ")
    ))
    .unwrap()
}
