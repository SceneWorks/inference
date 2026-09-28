//! Safetensors weight loading.
//!
//! [`Weights`] is a flat name → `Array` map loaded from a single file or a sharded HF snapshot
//! directory (`model-00001-of-0000N.safetensors`, …). Models look tensors up by their HF key via
//! [`Weights::require`] / [`Weights::get`]. MLX reads safetensors on the CPU stream by default; the
//! arrays are lifted to the GPU lazily on first use.
//!
//! # The access set, and why a lazy map needs one
//!
//! Every successful [`Weights::require`] / [`Weights::get`] records its key. That set is what lets a
//! *streaming* loader ([`crate::residency`]) release a decoder layer's weights the moment the layer
//! has run: `Array` is refcounted, so dropping the built layer frees nothing while this map still
//! holds its own handle on the same buffers. [`Weights::remove_accessed`] drops exactly the handles
//! the last layer read — not a prefix sweep, so a key the layer *should* have read and did not is
//! left behind as a discriminator rather than deleted along with the rest.
//!
//! This mirrors `mlx_gen::weights::Weights`, whose block-window loaders established the primitive
//! (sc-15750). The two crates cannot share a type — `mlx-gen` depends on `mlx-llm`, not the reverse
//! — so the semantics are mirrored deliberately and the names kept identical.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::Array;

use crate::error::{Error, Result};

/// A loaded set of named weight tensors.
#[derive(Debug, Default)]
pub struct Weights {
    tensors: HashMap<String, Array>,
    /// Keys read through [`Weights::require`] / [`Weights::get`] since the last
    /// [`Weights::remove_accessed`]. See the module docs.
    accessed: RefCell<HashSet<String>>,
}

impl Weights {
    /// Construct directly from an in-memory map (used by converters and tests).
    pub fn from_map(tensors: HashMap<String, Array>) -> Self {
        Self {
            tensors,
            accessed: RefCell::new(HashSet::new()),
        }
    }

    /// Load every tensor from a single `.safetensors` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let tensors = Array::load_safetensors(path)
            .map_err(|e| Error::Msg(format!("load_safetensors {}: {e}", path.display())))?;
        Ok(Self::from_map(tensors))
    }

    /// Load and merge every `*.safetensors` shard in a snapshot directory.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
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
            let part = Array::load_safetensors(&shard)
                .map_err(|e| Error::Msg(format!("load_safetensors {}: {e}", shard.display())))?;
            tensors.extend(part);
        }
        Ok(Self::from_map(tensors))
    }

    /// Fetch a tensor by key, erroring if absent. Records the key in the access set.
    pub fn require(&self, key: &str) -> Result<&Array> {
        let value = self
            .tensors
            .get(key)
            .ok_or_else(|| Error::MissingTensor(key.to_string()))?;
        self.accessed.borrow_mut().insert(key.to_owned());
        Ok(value)
    }

    /// Fetch a tensor by key if present. Records the key in the access set.
    pub fn get(&self, key: &str) -> Option<&Array> {
        let value = self.tensors.get(key)?;
        self.accessed.borrow_mut().insert(key.to_owned());
        Some(value)
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

    /// Evaluate only the tensors read since the last [`Weights::remove_accessed`].
    ///
    /// A streaming loader calls this **before** draining, so the layer it just built has consumed
    /// its source bytes while the map still holds them — without evaluating the rest of the
    /// checkpoint, which would defeat the bounded residency the stream exists for.
    ///
    /// This is also the [`mlx_rs::transforms::eval`] that makes the subsequent drop a real release:
    /// MLX is lazy, so an unevaluated graph over a dropped tensor keeps the buffer alive anyway.
    pub fn materialize_accessed(&self) -> Result<()> {
        let accessed = self.accessed.borrow();
        mlx_rs::transforms::eval(accessed.iter().filter_map(|key| self.tensors.get(key)))?;
        Ok(())
    }

    /// Materialize the tensors read since the last [`Weights::remove_accessed`] in bounded batches
    /// and verify that the GPU reads each one as the bytes the CPU holds (sc-22414).
    ///
    /// This is the load boundary every model constructor and the streaming loader cross before a
    /// graph consumes the bytes: a lazy `Load` is forced here, on its own (CPU) stream, in batches
    /// of at most [`Weights::VERIFY_BATCH_BYTES`] so a cold multi-gigabyte file never becomes one
    /// submission, and each batch is then checked through
    /// [`coherence::verify_gpu_view`](crate::primitives::coherence::verify_gpu_view). Keys are
    /// visited in sorted order so the batching is deterministic.
    ///
    /// Only the *accessed* set is touched, for the same reason [`Weights::materialize_accessed`]
    /// restricts itself: evaluating the rest of a checkpoint would defeat bounded residency.
    pub fn verify_accessed_gpu_view(&self) -> Result<()> {
        let accessed = self.accessed.borrow();
        let mut keys: Vec<&str> = accessed.iter().map(String::as_str).collect();
        keys.sort_unstable();
        let mut batch: Vec<(&str, &Array)> = Vec::new();
        let mut bytes = 0usize;
        for key in keys {
            let Some(array) = self.tensors.get(key) else {
                continue;
            };
            bytes = bytes.saturating_add(array.nbytes());
            batch.push((key, array));
            if bytes >= Self::VERIFY_BATCH_BYTES {
                Self::verify_batch(&batch)?;
                batch.clear();
                bytes = 0;
            }
        }
        if !batch.is_empty() {
            Self::verify_batch(&batch)?;
        }
        Ok(())
    }

    /// Upper bound on the bytes one [`Weights::verify_accessed_gpu_view`] batch evaluates at once.
    pub const VERIFY_BATCH_BYTES: usize = 512 * 1024 * 1024;

    fn verify_batch(batch: &[(&str, &Array)]) -> Result<()> {
        mlx_rs::transforms::eval(batch.iter().map(|(_, a)| *a))?;
        crate::primitives::coherence::verify_gpu_view(batch.iter().copied())
    }

    /// Drop every tensor read through [`Weights::require`] / [`Weights::get`] since the previous
    /// call, and reset the access set.
    ///
    /// LOAD-BEARING, not decorative: `Array` is refcounted, so dropping a built decoder layer frees
    /// nothing while this map still holds its own handle on the same buffers. Draining *exactly the
    /// accessed keys* — rather than sweeping a `model.layers.{i}.` prefix — is what leaves a key the
    /// layer should have read but did not behind as an observable discriminator
    /// ([`Weights::unused_keys`]) instead of deleting it along with the rest.
    pub fn remove_accessed(&mut self) {
        let accessed = std::mem::take(self.accessed.get_mut());
        for key in accessed {
            self.tensors.remove(&key);
        }
    }

    /// Every stored key **not** yet read — the complement of the access set. A loader-conformance
    /// test constructs a model against a candidate map and asserts this is empty, proving no tensor
    /// was silently ignored.
    pub fn unused_keys(&self) -> Vec<&str> {
        let accessed = self.accessed.borrow();
        self.tensors
            .keys()
            .map(String::as_str)
            .filter(|k| !accessed.contains(*k))
            .collect()
    }

    /// Consume into the underlying `name → Array` map (used by the snapshot writer, which drains the
    /// loaded tensor set into its safetensors output).
    pub fn into_map(self) -> HashMap<String, Array> {
        self.tensors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixture::{assert_fixture_is_self_removing, Fixture};

    #[test]
    fn require_and_get_on_in_memory_map() {
        let mut m = HashMap::new();
        m.insert(
            "a.weight".to_string(),
            Array::from_slice(&[1.0f32, 2.0], &[2]),
        );
        let w = Weights::from_map(m);
        assert_eq!(w.len(), 1);
        assert!(w.contains("a.weight"));
        assert!(w.require("a.weight").is_ok());
        assert!(w.get("missing").is_none());
        assert!(matches!(w.require("missing"), Err(Error::MissingTensor(_))));
    }

    #[test]
    fn save_then_load_roundtrip() {
        // sc-17768: a guarded fixture root, so the tree leaves on `Drop` even when an assertion
        // below panics. It also has to outlive every `Weights` read here — MLX's
        // `load_safetensors` is lazy, so the tensors are still bound to this directory.
        let dir = Fixture::new("mlx-llm-weights-test-", None);
        let path = dir.join("model.safetensors");
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        Array::save_safetensors([("w", &a)], None, &path).unwrap();

        let w = Weights::from_file(&path).unwrap();
        assert_eq!(w.require("w").unwrap().shape(), &[2, 2]);

        let w2 = Weights::from_dir(&dir).unwrap();
        assert!(w2.contains("w"));
    }

    /// Drop-regression for this suite's fixture helper: the root leaves with the value. Flip
    /// [`Fixture::new`]'s builder to `disable_cleanup(true)` and this goes RED.
    #[test]
    fn weights_fixture_is_self_removing() {
        assert_fixture_is_self_removing(Fixture::new("mlx-llm-weights-test-", None));
    }

    /// The view-drain primitive the sequential decoder stack is built on (sc-18798).
    ///
    /// Two properties, and the second is the one that makes it worth having over a prefix sweep:
    ///
    /// 1. `remove_accessed` drops exactly what was read since the last drain, and resets the set —
    ///    so draining after layer 0 must not touch layer 1's tensors.
    /// 2. A key under the drained prefix that was **not** read survives, and `unused_keys` names it.
    ///    `remove_prefix("model.layers.0.")` would delete it along with the rest, turning an omitted
    ///    constructor read into silence.
    ///
    /// MUTATION: make `remove_accessed` sweep by prefix instead of by access set, and the
    /// `never_read` assertion goes RED. Make `require`/`get` stop recording, and the first
    /// `len()` assertion goes RED.
    #[test]
    fn remove_accessed_drains_exactly_what_was_read() {
        let t = |v: f32| Array::from_slice(&[v], &[1]);
        let mut m = HashMap::new();
        m.insert("model.layers.0.q".to_string(), t(0.0));
        m.insert("model.layers.0.k".to_string(), t(1.0));
        m.insert("model.layers.0.never_read".to_string(), t(2.0));
        m.insert("model.layers.1.q".to_string(), t(3.0));
        let mut w = Weights::from_map(m);
        assert_eq!(w.unused_keys().len(), 4, "nothing has been read yet");

        // "Build layer 0" — read its q and k, but not `never_read`.
        w.require("model.layers.0.q").expect("q");
        w.get("model.layers.0.k").expect("k");
        w.remove_accessed();

        assert_eq!(w.len(), 2, "exactly the two read tensors were dropped");
        assert!(
            w.contains("model.layers.0.never_read"),
            "a key under the same prefix that the layer did NOT read must survive the drain — a \
             prefix sweep would delete it and hide the omitted read"
        );
        assert!(
            w.contains("model.layers.1.q"),
            "the next layer's tensors must be untouched"
        );

        // The access set reset with the drain: reading layer 1 and draining again must not
        // retroactively remove anything else.
        w.require("model.layers.1.q").expect("layer 1 q");
        w.remove_accessed();
        assert_eq!(w.keys().collect::<Vec<_>>(), ["model.layers.0.never_read"]);
        assert_eq!(w.unused_keys(), ["model.layers.0.never_read"]);
    }
}
