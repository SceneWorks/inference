//! A tiny key→tensor map (the candle twin of `mlx_gen::weights::Weights`) so the SeedVR2 modules can
//! be ported 1:1 from the MLX reference — looking weights up by dotted key and reading shapes off the
//! tensors, rather than threading explicit shapes through `VarBuilder::get`. The converter
//! ([`crate::convert`]) produces one of these (DiT key-renames / VAE pass-through), then the modules
//! consume it.

use std::collections::HashMap;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::{CandleError, Result as CResult};

/// An owned `name → Tensor` table.
#[derive(Default)]
pub struct Weights {
    map: HashMap<String, Tensor>,
}

impl Weights {
    pub fn empty() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Load a safetensors file onto `device` (every tensor kept at its on-disk dtype).
    pub fn from_file(path: impl AsRef<std::path::Path>, device: &Device) -> CResult<Self> {
        let map = candle_gen::candle_core::safetensors::load(path.as_ref(), device)?;
        Ok(Self { map })
    }

    pub fn insert(&mut self, key: impl Into<String>, t: Tensor) {
        self.map.insert(key.into(), t);
    }

    pub fn get(&self, key: &str) -> Option<&Tensor> {
        self.map.get(key)
    }

    /// Look up `key` or fail with a clear message (the candle twin of mlx `Weights::require`).
    pub fn require(&self, key: &str) -> CResult<&Tensor> {
        self.map
            .get(key)
            .ok_or_else(|| CandleError::Msg(format!("seedvr2: missing weight tensor `{key}`")))
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(String::as_str)
    }

    /// Drain `self` into `(key, tensor)` pairs, consuming the map so each raw tensor can be dropped by
    /// the caller as it is cast (streaming convert+cast keeps peak load memory to ~1× rather than
    /// holding the whole raw fp16 set and the whole cast copy at once — sc-9042/F-058).
    pub fn into_iter_entries(self) -> impl Iterator<Item = (String, Tensor)> {
        self.map.into_iter()
    }
}

/// Model construction accepts either already converted dense tensors or a mapped, source-preserving checkpoint.
pub trait ModelWeights {
    fn tensor(&self, key: &str) -> CResult<Tensor>;
    fn contains(&self, key: &str) -> bool;
    fn dtype(&self, key: &str) -> CResult<candle_gen::candle_core::DType> {
        Ok(self.tensor(key)?.dtype())
    }
    fn native_projection(
        &self,
        _base: &str,
        _bias: bool,
    ) -> CResult<Option<candle_gen::candle_nn::Linear>> {
        Ok(None)
    }
}
impl ModelWeights for Weights {
    fn tensor(&self, key: &str) -> CResult<Tensor> {
        self.require(key).cloned()
    }
    fn contains(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }
}

/// Renaming a shared `.all` projection must preserve its codes, scales and shared allocation.
pub(crate) struct SourceWeights {
    source: candle_gen::Weights,
    physical: HashMap<String, String>,
    dtype: candle_gen::candle_core::DType,
    layers: std::cell::RefCell<HashMap<(String, bool), candle_gen::candle_nn::Linear>>,
}
impl SourceWeights {
    pub(crate) fn open(
        path: &std::path::Path,
        device: &Device,
        dtype: candle_gen::candle_core::DType,
        targets: impl Fn(&str) -> Vec<String>,
    ) -> CResult<Self> {
        let source = candle_gen::Weights::from_file(path, device, dtype)?;
        let mut physical = HashMap::new();
        for key in source.keys().filter(|k| !source.is_checkpoint_companion(k)) {
            for target in targets(key) {
                if physical.insert(target.clone(), key.clone()).is_some() {
                    return Err(CandleError::Msg(format!(
                        "duplicate converted SeedVR2 key {target}"
                    )));
                }
            }
        }
        Ok(Self {
            source,
            physical,
            dtype,
            layers: Default::default(),
        })
    }
    fn key(&self, key: &str) -> CResult<&str> {
        self.physical
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| CandleError::Msg(format!("seedvr2: missing weight tensor `{key}`")))
    }
}
impl ModelWeights for SourceWeights {
    fn tensor(&self, key: &str) -> CResult<Tensor> {
        Ok(self.source.require(self.key(key)?)?.to_dtype(self.dtype)?)
    }
    fn contains(&self, key: &str) -> bool {
        self.physical.contains_key(key)
    }
    fn dtype(&self, key: &str) -> CResult<candle_gen::candle_core::DType> {
        self.key(key)?;
        Ok(self.dtype)
    }
    fn native_projection(
        &self,
        base: &str,
        bias: bool,
    ) -> CResult<Option<candle_gen::candle_nn::Linear>> {
        let raw = self.key(&format!("{base}.weight"))?;
        if self.source.source_format_id(raw).is_none() {
            return Ok(None);
        }
        let physical = raw.strip_suffix(".weight").ok_or_else(|| {
            CandleError::Msg(format!("SeedVR2 projection key {raw} has no weight suffix"))
        })?;
        let key = (physical.to_owned(), bias);
        if let Some(layer) = self.layers.borrow().get(&key) {
            return Ok(Some(layer.clone()));
        }
        let layer = self.source.native_linear(physical, bias)?;
        if let Some(ref layer) = layer {
            self.layers.borrow_mut().insert(key, layer.clone());
        }
        Ok(layer)
    }
}
