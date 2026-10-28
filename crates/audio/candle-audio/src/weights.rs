//! The audio lane's one audited `unsafe` safetensors memory-map surface (sc-27259 / sc-27358).
//!
//! Every candle audio provider maps its caller-provisioned weight files through these helpers
//! instead of restating its own `unsafe { MmapedSafetensors::new(..) }` /
//! `unsafe { VarBuilder::from_mmaped_safetensors(..) }` block and its own safety justification.
//!
//! # The contract
//!
//! Inference receives every model component as a **caller-provisioned local path**
//! (`WeightsSource::Dir`/`File`, epic 13657); user-supplied models at arbitrary paths must load, so
//! nothing here can promise the file is a pinned, immutable snapshot. Memory-mapping is still the
//! right mechanism for multi-GB weights (reading the whole file would double resident memory), so
//! the supported contract is stated instead of assumed:
//!
//! - the mapping is **read-only**; nothing in this process writes or truncates a mapped file;
//! - the caller must not truncate or rewrite the files while they are mapped (for a `VarBuilder`,
//!   until the builder and every tensor borrowing it are dropped; tensors loaded to an owned
//!   device buffer stop borrowing the map). External mutation during that window is **outside the
//!   supported contract**: the process may observe torn data or be killed by `SIGBUS`.
//!
//! Small, request-supplied files (e.g. Stable Audio 3 LoRA/DoRA adapters) should not be mapped at
//! all — read them into an owned buffer (`BufferedSafetensors`) so a concurrent rewrite cannot
//! fault the process (sc-27352).

use std::path::Path;

use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

/// Memory-map one caller-provisioned safetensors file. See the [module contract](self).
pub fn mmap_safetensors(path: impl AsRef<Path>) -> candle_core::Result<MmapedSafetensors> {
    // SAFETY: read-only map of a caller-provisioned weight file; external mutation while mapped is
    // outside the supported contract documented on this module.
    unsafe { MmapedSafetensors::new(path) }
}

/// Memory-map caller-provisioned safetensors shard(s) into a `'static` [`VarBuilder`] that
/// materializes tensors at `dtype` on `device`. See the [module contract](self).
pub fn mmap_var_builder<P: AsRef<Path>>(
    paths: &[P],
    dtype: DType,
    device: &Device,
) -> candle_core::Result<VarBuilder<'static>> {
    // SAFETY: read-only map of caller-provisioned weight files; external mutation while mapped is
    // outside the supported contract documented on this module.
    unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    fn write_st(path: &Path, name: &str, values: &[f32]) {
        let mut map = std::collections::HashMap::new();
        map.insert(name.to_string(), Tensor::new(values, &Device::Cpu).unwrap());
        candle_core::safetensors::save(&map, path).unwrap();
    }

    #[test]
    fn mmap_helpers_read_tensors_and_report_missing_files() {
        let tmp = tempfile::tempdir().unwrap();
        let a = tmp.path().join("a.safetensors");
        let b = tmp.path().join("b.safetensors");
        write_st(&a, "x", &[1.0, 2.0]);
        write_st(&b, "y", &[3.0]);

        let st = mmap_safetensors(&a).unwrap();
        assert_eq!(st.get("x").unwrap().shape(), &[2]);

        let vb = mmap_var_builder(&[&a, &b], DType::F32, &Device::Cpu).unwrap();
        let y = vb.get(1, "y").unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(y, vec![3.0]);
        assert!(vb.contains_tensor("x"));

        assert!(mmap_safetensors(tmp.path().join("missing.safetensors")).is_err());
        assert!(mmap_var_builder(&[tmp.path().join("missing")], DType::F32, &Device::Cpu).is_err());
    }
}
