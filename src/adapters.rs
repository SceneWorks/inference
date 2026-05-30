//! Adapter framework — LoRA + LoKr applied as forward-time residuals over a shared
//! base. Quantized-safe: the base is never fused/mutated. Ported from the sc-2338
//! spike; mirrors the Python mflux fork's `LoKrLinear` / `FusedLoRALinear` (sc-2216).

use mlx_rs::{
    ops::{add, kron, matmul, multiply},
    Array, Dtype,
};

/// Crate-local result type. Refined into a typed error enum in a later story.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Reconstruct a LoKr weight delta `ΔW = (alpha/rank) · kron(w1, w2)`, reshaped to the
/// base weight's logical `[out, in]` and stored at bf16. Each Kronecker factor is either
/// full (`w1` / `w2`) or a low-rank product (`w1_a @ w1_b` / `w2_a @ w2_b`). Mirrors
/// PEFT/LyCORIS `LoKrLayer.get_delta_weight` (pending the sc-2324 cross-impl parity check).
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_lokr_delta(
    alpha: f32,
    rank: f32,
    base_shape: &[i32],
    w1: Option<&Array>,
    w1_a: Option<&Array>,
    w1_b: Option<&Array>,
    w2: Option<&Array>,
    w2_a: Option<&Array>,
    w2_b: Option<&Array>,
) -> Result<Array> {
    let factor1 = match (w1, w1_a, w1_b) {
        (Some(w), _, _) => w.clone(),
        (_, Some(a), Some(b)) => matmul(a, b)?,
        _ => return Err("LoKr: w1 missing (need full w1 or w1_a@w1_b)".into()),
    };
    let factor2 = match (w2, w2_a, w2_b) {
        (Some(w), _, _) => w.clone(),
        (_, Some(a), Some(b)) => matmul(a, b)?,
        _ => return Err("LoKr: w2 missing (need full w2 or w2_a@w2_b)".into()),
    };
    let delta = multiply(&kron(&factor1, &factor2)?, &scalar(alpha / rank))?;
    Ok(delta.reshape(base_shape)?.as_dtype(Dtype::Bfloat16)?)
}

/// One adapter's contribution WITHOUT the base, so a host can sum stacked adapters over
/// a single base application.
pub enum Adapter {
    /// LoRA: `residual = scale · x·A·B`.
    Lora { a: Array, b: Array, scale: f32 },
    /// LoKr: `residual = scale · x·ΔWᵀ`; `delta` stored bf16 (see [`reconstruct_lokr_delta`]).
    Lokr { delta: Array, scale: f32 },
}

impl Adapter {
    pub fn residual(&self, x: &Array) -> Result<Array> {
        Ok(match self {
            Adapter::Lora { a, b, scale } => multiply(&matmul(&matmul(x, a)?, b)?, &scalar(*scale))?,
            Adapter::Lokr { delta, scale } => {
                // Reconcile the bf16 delta with the activation dtype (no-op when x is bf16).
                let d = delta.as_dtype(x.dtype())?;
                multiply(&matmul(x, &d.t())?, &scalar(*scale))?
            }
        })
    }
}

/// A base linear plus a stack of adapters, applied as `base(x) + Σ adapter.residual(x)`.
/// Quantized-safe: the base weight is never mutated. The base is a plain `[out, in]`
/// weight for now; it generalizes to `nn::Linear` / `nn::QuantizedLinear` when the first
/// model tree is ported (sc-2344), at which point the path-addressed install lands too.
pub struct AdaptableLinear {
    weight: Array,
    adapters: Vec<Adapter>,
}

impl AdaptableLinear {
    pub fn new(weight: Array) -> Self {
        Self { weight, adapters: Vec::new() }
    }

    /// Stack a new adapter (LoRA or LoKr) on top of any already installed.
    pub fn push(&mut self, adapter: Adapter) {
        self.adapters.push(adapter);
    }

    pub fn adapters(&self) -> &[Adapter] {
        &self.adapters
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut out = matmul(x, &self.weight.t())?;
        for adapter in &self.adapters {
            out = add(&out, &adapter.residual(x)?)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{all_close, array_eq};

    fn lokr_2x2() -> Array {
        reconstruct_lokr_delta(
            8.0,
            4.0,
            &[2, 2],
            Some(&Array::from_slice(&[0.5f32, 0.6], &[2, 1])),
            None,
            None,
            Some(&Array::from_slice(&[0.7f32, 0.8], &[1, 2])),
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn lokr_delta_stored_bf16() {
        assert_eq!(lokr_2x2().dtype(), Dtype::Bfloat16);
    }

    #[test]
    fn scale_zero_lokr_is_bit_exact_noop() {
        let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        let mut lin = AdaptableLinear::new(w);
        let base = lin.forward(&x).unwrap();
        lin.push(Adapter::Lokr { delta: lokr_2x2(), scale: 0.0 });
        let out = lin.forward(&x).unwrap();
        assert!(array_eq(&out, &base, false).unwrap().item::<bool>());
    }

    #[test]
    fn stacks_mixed_lora_and_lokr_summing_residuals() {
        let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        let mut lin = AdaptableLinear::new(w);
        let base = lin.forward(&x).unwrap();
        let lora = Adapter::Lora {
            a: Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]),
            b: Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]),
            scale: 0.5,
        };
        let lokr = Adapter::Lokr { delta: lokr_2x2(), scale: 0.7 };
        let lora_r = lora.residual(&x).unwrap();
        let lokr_r = lokr.residual(&x).unwrap();
        lin.push(lora);
        lin.push(lokr);
        assert_eq!(lin.adapters().len(), 2);
        let expected = add(&add(&base, &lora_r).unwrap(), &lokr_r).unwrap();
        assert!(all_close(&lin.forward(&x).unwrap(), &expected, 1e-4, 1e-2, false).unwrap().item::<bool>());
    }
}
