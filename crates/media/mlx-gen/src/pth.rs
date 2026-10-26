//! Torch `.pth` (zip-of-pickle) loading into MLX arrays — first written for the native Wan converter
//! (sc-3237 / sc-3224), shared from mlx-gen since sc-24833 (the E-LatentLPIPS checkpoints ship as
//! `torch.save` state dicts too, BatchNorm `num_batches_tracked` int64 counters included).
//!
//! Since sc-24213 the pickle VM and archive access are the tensor-free [`gen_core::pth`] reader,
//! shared with the Candle backend (Supra2-IMG converts its `.pt` on both install paths). This module
//! is the MLX adapter: the reference Python loads these with `torch.load(...).float()` (every tensor
//! → f32), so [`load_pth_f32`] mirrors that exactly and returns each tensor as an **f32** MLX
//! [`Array`] in PyTorch layout (the converter's sanitizers then transpose conv weights to
//! channels-last and cast per component).

use std::collections::HashMap;
use std::path::Path;

use crate::Result;
use gen_core::pth::PthArchive;
use mlx_rs::Array;

/// Load a torch `.pth` checkpoint, returning every tensor as an f32 MLX [`Array`] in PyTorch layout
/// (`torch.load(...).float()` semantics). Conv-weight transposes + key renames are the caller's job.
pub fn load_pth_f32(path: impl AsRef<Path>) -> Result<HashMap<String, Array>> {
    load_pth_f32_filtered(path, |_| true)
}

/// [`load_pth_f32`] keeping only the tensors whose name passes `keep`: the others' storage blobs
/// are never read (a multi-tower checkpoint loads one tower without materializing the rest).
pub fn load_pth_f32_filtered(
    path: impl AsRef<Path>,
    keep: impl Fn(&str) -> bool,
) -> Result<HashMap<String, Array>> {
    let path = path.as_ref();
    let mut archive = PthArchive::open(path)?;
    let mut out = HashMap::new();
    archive.visit_state_dict(keep, |name, tensor| {
        let floats = tensor.to_f32()?;
        // Dims are validated non-negative and ≤ i32::MAX by the reader, so this cannot truncate.
        let shape: Vec<i32> = tensor.shape.iter().map(|&d| d as i32).collect();
        out.insert(name, Array::from_slice(&floats, &shape));
        Ok(())
    })?;
    gen_core::__model_input_result!(
        Ok(out),
        "mlx-gen::pth::load_pth_f32_filtered",
        ModelWeights,
        &[path]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sc-24833: a real `torch.save`d module state dict with a BatchNorm (float tensors, the int64
    /// `num_batches_tracked` counter, the `_metadata` attribute) — the format of the published
    /// E-LatentLPIPS checkpoints — loads with every float tensor bit-identical to the producer's
    /// formula values and the counter decoded. Mutation: drop the `LongStorage` arm ⇒ the load
    /// errors ⇒ red.
    #[test]
    fn loads_a_torch_state_dict_with_batchnorm_counters() {
        use crate::train::formula::{formula_values, Role};
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../contracts/gen-core/tests/fixtures/latent_perceptual/tiny_bn_state_dict.pth",
        );
        let sd = load_pth_f32(path).unwrap();
        assert_eq!(sd.len(), 7, "{:?}", sd.keys().collect::<Vec<_>>());
        for (k, shape, role) in [
            ("0.weight", vec![2, 4, 3, 3], Role::Conv),
            ("0.bias", vec![2], Role::Bias),
            ("1.weight", vec![2], Role::NormWeight),
            ("1.bias", vec![2], Role::Bias),
            ("1.running_mean", vec![2], Role::BnMean),
            ("1.running_var", vec![2], Role::BnVar),
        ] {
            let a = &sd[k];
            assert_eq!(a.shape(), shape.as_slice(), "{k}");
            assert_eq!(
                a.as_slice::<f32>(),
                formula_values(k, &shape, role).as_slice(),
                "{k}"
            );
        }
        assert_eq!(sd["1.num_batches_tracked"].item::<f32>(), 7.0);
    }
}
