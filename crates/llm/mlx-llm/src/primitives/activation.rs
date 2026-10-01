//! The activation-dtype policy — the **one** place that decides in which dtype a tanh-approximate
//! GELU hands its result on (sc-24446).
//!
//! `mlx_rs::nn::gelu_approximate` builds its constants as `f32` arrays, so a BF16 input comes back
//! `f32`. In a GeGLU decoder (every Gemma generation) that `f32` then flows through `gate · up`,
//! the down projection, the residual stream and every later layer: each dense matmul promotes its
//! BF16 weight to an `f32` copy every forward (the LM head's alone is 2.4 GB on Gemma 2 and 4.0 GB
//! on the 262K-token Gemma 4), and the decoder reads every weight twice over.
//! [`GeluPrecision::ActivationDtype`] returns the GELU in its input's dtype instead — computed
//! with the same `f32` constants and rounded once, as PyTorch's BF16 `gelu(approximate="tanh")`
//! does — so the stream stays BF16, as the reference models run.
//!
//! Every tanh-GELU in this crate states its [`ActivationRole`]; [`gelu_precision`] maps the role
//! to a precision. Switching a role is a numerics change: it moves that role's outputs within the
//! tolerance its parity gate pins (`docs/reference/qwen38/native-memory-admission.md`).

use mlx_rs::Array;

use crate::error::Result;

/// What a tanh-GELU's output feeds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActivationRole {
    /// An LLM decoder generating tokens: chat serving ([`crate::LlamaProvider`]), draft models,
    /// prompt enhancement, StarVector's SVG decoders.
    #[default]
    LlmDecode,
    /// The LTX-2.5 text encoder (a Gemma 4 decoder whose hidden states condition the video
    /// model). Its pinned goldens are real-weight connector-input and tier-quality captures taken
    /// with `f32` GeGLU activations; until they are re-run with BF16 it stays on the `f32` path.
    LtxTextEncoder,
    /// A vision tower or projector (SigLIP, the Qwen3-VL ViT, JoyCaption's projector): an
    /// embedding producer, not a token decoder, held to its own vision parity fixtures.
    VisionEncoder,
}

/// The dtype a tanh-GELU returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeluPrecision {
    /// The input's dtype (BF16 in, BF16 out).
    ActivationDtype,
    /// `f32` whatever the input — `mlx_rs::nn::gelu_approximate` as it is.
    F32,
}

/// The policy.
pub const fn gelu_precision(role: ActivationRole) -> GeluPrecision {
    match role {
        ActivationRole::LlmDecode => GeluPrecision::ActivationDtype,
        ActivationRole::LtxTextEncoder | ActivationRole::VisionEncoder => GeluPrecision::F32,
    }
}

/// Tanh-approximate GELU for `role` (do not unify with the exact [`crate::primitives::nn::gelu`]:
/// the two are numerically distinct and both appear in the JoyCaption VLM).
pub fn gelu_tanh(x: &Array, role: ActivationRole) -> Result<Array> {
    let y = mlx_rs::nn::gelu_approximate(x)?;
    #[cfg(test)]
    let precision = parity::OVERRIDE
        .with(std::cell::Cell::get)
        .unwrap_or(gelu_precision(role));
    #[cfg(not(test))]
    let precision = gelu_precision(role);
    Ok(match precision {
        GeluPrecision::ActivationDtype if y.dtype() != x.dtype() => y.as_dtype(x.dtype())?,
        _ => y,
    })
}

/// The parity gate a role must pass to switch to [`GeluPrecision::ActivationDtype`]
/// (sc-24446): the same decoder run under both precisions on a fixture.
///
/// **Tolerance: `|Δlogit| ≤ 2⁻⁵ · max(|logit|, 1)` at every step** — eight BF16 ULPs at the
/// logit's magnitude. Rounding a GeGLU output to BF16 (an 8-bit significand) moves it by at most
/// 2⁻⁹ relative; the residual stream then stays BF16 too, so each later op rounds where the `f32`
/// path did not. Over the fixtures' 2–4 layers and the LM head that measures 1–2 % (the Gemma 4
/// decoder golden: 1.1e-2 against an `f32` oracle); 2⁻⁵ keeps ~1.5–3x headroom and is still far
/// below what a structural error moves (a dropped layer scalar: 1.1e-1). **Greedy tokens** must
/// agree at every step whose `f32` top-2 margin exceeds twice that step's `|Δlogit|` — a closer
/// race may legitimately flip, and is counted, never silently accepted.
#[cfg(test)]
pub(crate) mod parity {
    use std::cell::Cell;
    use std::collections::HashMap;

    use mlx_rs::{Array, Dtype};

    use super::GeluPrecision;
    use crate::decode::Decode;
    use crate::primitives::sampler::{SplitMix64, TokenRng};

    thread_local! {
        pub(super) static OVERRIDE: Cell<Option<GeluPrecision>> = const { Cell::new(None) };
    }

    /// Run `f` with every tanh-GELU forced to `precision`, whatever its role.
    pub(crate) fn with_gelu_precision<R>(precision: GeluPrecision, f: impl FnOnce() -> R) -> R {
        struct Restore(Option<GeluPrecision>);
        impl Drop for Restore {
            fn drop(&mut self) {
                OVERRIDE.with(|o| o.set(self.0));
            }
        }
        let _restore = Restore(OVERRIDE.with(|o| o.replace(Some(precision))));
        f()
    }

    /// The relative logit budget (see the module docs).
    pub(crate) const LOGIT_TOL: f32 = 1.0 / 32.0;

    /// A BF16 weight map for `shapes`: norm weights near one, everything else small and random.
    pub(crate) fn random_bf16(shapes: &[(String, Vec<i32>)], seed: u64) -> HashMap<String, Array> {
        let mut rng = SplitMix64::new(seed);
        shapes
            .iter()
            .map(|(key, shape)| {
                let n: i32 = shape.iter().product();
                let norm = key.contains("norm") || key.contains(".ln_");
                let data: Vec<f32> = (0..n)
                    .map(|_| {
                        let r = (rng.next_f32() - 0.5) * 0.4;
                        if norm && key.ends_with(".weight") {
                            1.0 + r
                        } else {
                            r
                        }
                    })
                    .collect();
                let a = Array::from_slice(&data, shape)
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap();
                (key.clone(), a)
            })
            .collect()
    }

    fn logits(model: &dyn Decode, ids: &[i32]) -> Vec<f32> {
        let mut cache = model.make_cache();
        let ids = Array::from_slice(ids, &[1, ids.len() as i32]);
        let out = model.step(&ids, cache.as_mut(), 0).unwrap();
        out.as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    }

    fn top2(v: &[f32]) -> (usize, f32) {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
        (idx[0], v[idx[0]] - v[idx[1]])
    }

    /// What one family's gate measured.
    #[derive(Debug)]
    pub(crate) struct Parity {
        /// Worst `|Δlogit| / max(|logit|, 1)` over every step.
        pub(crate) worst: f32,
        /// Steps whose greedy token differs (each one a race closer than twice its `|Δlogit|`).
        pub(crate) flips: usize,
    }

    /// Greedy-decode `steps` tokens after `prompt` on the `f32` path, then teacher-force the
    /// activation-dtype path along the same tokens and hold every step to [`LOGIT_TOL`] and the
    /// greedy-margin rule (module docs).
    pub(crate) fn assert_geglu_parity(
        label: &str,
        model: &dyn Decode,
        prompt: &[i32],
        steps: usize,
    ) -> Parity {
        let mut ids = prompt.to_vec();
        let mut report = Parity {
            worst: 0.0,
            flips: 0,
        };
        for step in 0..steps {
            let wide = with_gelu_precision(GeluPrecision::F32, || logits(model, &ids));
            let narrow =
                with_gelu_precision(GeluPrecision::ActivationDtype, || logits(model, &ids));
            let scale = wide.iter().fold(1.0f32, |m, x| m.max(x.abs()));
            let delta = wide
                .iter()
                .zip(&narrow)
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            let rel = delta / scale;
            report.worst = report.worst.max(rel);
            assert!(
                rel <= LOGIT_TOL,
                "{label} step {step}: |Δlogit| {delta} is {rel:.2e} of {scale}, over {LOGIT_TOL}"
            );
            let (token, margin) = top2(&wide);
            if top2(&narrow).0 != token {
                report.flips += 1;
                assert!(
                    margin <= 2.0 * delta,
                    "{label} step {step}: greedy token flipped on a decisive margin {margin} \
                     (|Δlogit| {delta})"
                );
            }
            ids.push(token as i32);
        }
        assert!(
            report.worst > 0.0,
            "{label}: the two precisions agree exactly — the gate is comparing a path with itself"
        );
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Dtype;

    /// The policy, and what each precision returns for a BF16 input: an LLM decoder keeps BF16
    /// (the GELU rounded once from its `f32` evaluation), a pinned role keeps `f32`.
    #[test]
    fn llm_decode_keeps_the_activation_dtype_and_pinned_roles_keep_f32() {
        assert_eq!(
            gelu_precision(ActivationRole::LlmDecode),
            GeluPrecision::ActivationDtype
        );
        assert_eq!(
            gelu_precision(ActivationRole::LtxTextEncoder),
            GeluPrecision::F32
        );
        assert_eq!(
            gelu_precision(ActivationRole::VisionEncoder),
            GeluPrecision::F32
        );
        let x = Array::from_slice(&[-2.0f32, -0.5, 0.0, 0.7, 3.0], &[5])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let wide = gelu_tanh(&x, ActivationRole::LtxTextEncoder).unwrap();
        let narrow = gelu_tanh(&x, ActivationRole::LlmDecode).unwrap();
        assert_eq!(wide.dtype(), Dtype::Float32);
        assert_eq!(narrow.dtype(), Dtype::Bfloat16);
        assert_eq!(
            narrow.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>(),
            wide.as_dtype(Dtype::Bfloat16)
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap()
                .as_slice::<f32>(),
            "the BF16 result is the f32 GELU rounded once"
        );
        let f = Array::from_slice(&[0.3f32], &[1]);
        assert_eq!(
            gelu_tanh(&f, ActivationRole::LlmDecode).unwrap().dtype(),
            Dtype::Float32,
            "an f32 input is unchanged"
        );
    }
}
