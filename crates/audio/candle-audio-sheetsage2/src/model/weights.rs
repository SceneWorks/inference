//! Checkpoint tensors, consumed by name so a missing *or* unexpected tensor is an error (upstream
//! refuses `missing_keys` / `unexpected_keys` the same way).

use std::collections::HashMap;
use std::path::Path;

use candle_audio::candle_core::{DType, Device, Tensor};

use crate::Error;

/// Tensors of one safetensors file on the CPU, taken one by one.
pub struct Weights {
    label: &'static str,
    tensors: HashMap<String, Tensor>,
}

impl Weights {
    /// Load every tensor of `path` onto the CPU (float32 is required downstream; nothing is cast
    /// here, so the LoRA merge sees the stored values).
    pub fn load(label: &'static str, path: &Path) -> Result<Self, Error> {
        let tensors = candle_audio::candle_core::safetensors::load(path, &Device::Cpu)
            .map_err(|e| Error::Config(format!("{label}: cannot read {}: {e}", path.display())))?;
        Ok(Self { label, tensors })
    }

    /// Wrap an in-memory map (tests).
    pub fn from_map(label: &'static str, tensors: HashMap<String, Tensor>) -> Self {
        Self { label, tensors }
    }

    /// Take tensor `name`, which must be float32 with shape `shape`.
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Tensor, Error> {
        let tensor = self
            .tensors
            .remove(name)
            .ok_or_else(|| Error::Config(format!("{}: missing tensor {name}", self.label)))?;
        if tensor.dtype() != DType::F32 {
            return Err(Error::Config(format!(
                "{}: {name} is {:?}; the pinned checkpoints are float32 and the LoRA merge must \
                 run in float32",
                self.label,
                tensor.dtype()
            )));
        }
        if tensor.dims() != shape {
            return Err(Error::Config(format!(
                "{}: {name} has shape {:?}, expected {shape:?}",
                self.label,
                tensor.dims()
            )));
        }
        Ok(tensor)
    }

    /// Refuse any tensor that was not consumed.
    pub fn finish(self) -> Result<(), Error> {
        if self.tensors.is_empty() {
            return Ok(());
        }
        let mut names: Vec<&String> = self.tensors.keys().collect();
        names.sort();
        Err(Error::Config(format!(
            "{}: unexpected tensors {:?}",
            self.label,
            names.iter().take(8).collect::<Vec<_>>()
        )))
    }
}
