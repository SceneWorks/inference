//! Safetensors weight loading.
//!
//! [`Weights`] is a flat name → `Tensor` map loaded from a single file or a sharded HF snapshot
//! directory (`model-00001-of-0000N.safetensors`, …) onto a chosen [`Device`]. Models look tensors
//! up by their HF key via [`Weights::require`] / [`Weights::get`]. The Candle port of `mlx-llm`'s
//! `Weights` (Candle reads safetensors into device tensors directly).

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};

use crate::error::{Error, Result};

/// A loaded set of named weight tensors, plus the device they live on.
#[derive(Debug)]
pub struct Weights {
    tensors: HashMap<String, Tensor>,
    device: Device,
}

impl Weights {
    /// Construct directly from an in-memory map (used by converters and tests).
    pub fn from_map(tensors: HashMap<String, Tensor>, device: Device) -> Self {
        Self { tensors, device }
    }

    /// Load every tensor from a single `.safetensors` file onto `device`.
    pub fn from_file(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        let tensors = candle_core::safetensors::load(path, device)
            .map_err(|e| Error::Msg(format!("load_safetensors {}: {e}", path.display())))?;
        Ok(Self {
            tensors,
            device: device.clone(),
        })
    }

    /// Load and merge every `*.safetensors` shard in a snapshot directory onto `device`.
    pub fn from_dir(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let dir = dir.as_ref();
        let mut shards: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        if shards.is_empty() {
            return Err(Error::Msg(format!(
                "no .safetensors files in {}",
                dir.display()
            )));
        }
        shards.sort(); // deterministic merge order
        let mut tensors = HashMap::new();
        for shard in shards {
            let part = candle_core::safetensors::load(&shard, device)
                .map_err(|e| Error::Msg(format!("load_safetensors {}: {e}", shard.display())))?;
            tensors.extend(part);
        }
        Ok(Self {
            tensors,
            device: device.clone(),
        })
    }

    /// Load a snapshot directory **pipeline-sharded**: each tensor is placed on the device chosen by
    /// `place` (keyed by its HF name), so a model too large for one GPU can be split across several
    /// without ever staging the whole thing on a single card. Each shard file is read to host memory,
    /// its tensors are moved to their target devices, and the host copy is dropped — so peak *device*
    /// memory is per-GPU-shard and peak *host* memory is one shard file. `device` records the model's
    /// home device (where embeddings / the first layer live); it is the value [`Weights::device`]
    /// reports and the decoder treats as its input device.
    pub fn from_dir_sharded(
        dir: impl AsRef<Path>,
        device: Device,
        place: impl Fn(&str) -> Device,
    ) -> Result<Self> {
        let dir = dir.as_ref();
        let mut shards: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        if shards.is_empty() {
            return Err(Error::Msg(format!(
                "no .safetensors files in {}",
                dir.display()
            )));
        }
        shards.sort(); // deterministic merge order
        let mut tensors = HashMap::new();
        for shard in shards {
            // Read the shard onto the host, then hand each tensor to its target device. The host map
            // is dropped at the end of the iteration, so only one shard is resident on the host at once.
            let part = candle_core::safetensors::load(&shard, &Device::Cpu)
                .map_err(|e| Error::Msg(format!("load_safetensors {}: {e}", shard.display())))?;
            for (key, t) in part {
                let target = place(&key);
                let t = t.to_device(&target)?;
                tensors.insert(key, t);
            }
        }
        Ok(Self { tensors, device })
    }

    /// Fetch a tensor by key, erroring if absent.
    pub fn require(&self, key: &str) -> Result<&Tensor> {
        self.tensors
            .get(key)
            .ok_or_else(|| Error::MissingTensor(key.to_string()))
    }

    /// Fetch a tensor by key if present.
    pub fn get(&self, key: &str) -> Option<&Tensor> {
        self.tensors.get(key)
    }

    /// Whether a key is present.
    pub fn contains(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    /// Number of loaded tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether no tensors are loaded.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// All loaded tensor keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// The device the tensors live on (the decoder reads this to pick its compute dtype).
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Consume into the underlying `name → Tensor` map (used by the snapshot writer, which mutates
    /// then re-serializes the tensors via `candle_core::safetensors::save`).
    pub fn into_map(self) -> HashMap<String, Tensor> {
        self.tensors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn require_and_get_on_in_memory_map() {
        let mut m = HashMap::new();
        m.insert(
            "a.weight".to_string(),
            Tensor::from_vec(vec![1.0f32, 2.0], (2,), &Device::Cpu).unwrap(),
        );
        let w = Weights::from_map(m, Device::Cpu);
        assert_eq!(w.len(), 1);
        assert!(w.contains("a.weight"));
        assert!(w.require("a.weight").is_ok());
        assert!(w.get("missing").is_none());
        assert!(matches!(w.require("missing"), Err(Error::MissingTensor(_))));
    }

    #[test]
    fn save_then_load_roundtrip() {
        let dir =
            std::env::temp_dir().join(format!("candle-llm-weights-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu).unwrap();
        let mut map = HashMap::new();
        map.insert("w".to_string(), a);
        candle_core::safetensors::save(&map, &path).unwrap();

        let w = Weights::from_file(&path, &Device::Cpu).unwrap();
        assert_eq!(w.require("w").unwrap().dims(), &[2, 2]);

        let w2 = Weights::from_dir(&dir, &Device::Cpu).unwrap();
        assert!(w2.contains("w"));

        std::fs::remove_dir_all(&dir).ok();
    }
}
