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
    /// Keys whose handle [`Weights::materialize_groups`] dropped once the model's own arrays had
    /// consumed them (sc-24446). Still [`contained`](Weights::contains) — layout probes keep
    /// answering — but no longer readable.
    released: HashSet<String>,
}

/// What [`Weights::materialize_groups`] did, for the load's tests and probes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Materialized {
    /// Output groups evaluated (empty groups are skipped).
    pub groups: usize,
    /// Source tensors read, verified and released group by group.
    pub released: usize,
    /// Accessed sources no output group consumed: read, verified and kept, exactly as
    /// [`Weights::verify_accessed_gpu_view`] would have. A complete model enumeration leaves none.
    pub leftover: usize,
    /// The most MLX's buffer cache held right after any group's `clear_cache` (zero: every
    /// consumed buffer left MLX's cache before the next group read its sources).
    pub cache_after_groups: usize,
    /// Time spent waiting for the Metal driver to return released buffers (pacing).
    pub paced: std::time::Duration,
}

/// Footprint headroom the pacing allows beyond what the model has built and the previous group
/// released: host-heap and driver noise.
pub const MATERIALIZE_PACE_SLACK_BYTES: u64 = 128 * 1024 * 1024;

/// The process's `phys_footprint` (macOS `rusage_info_v4`), for pacing a materialization against
/// the Metal driver's asynchronous buffer release.
mod footprint {
    use std::time::{Duration, Instant};

    extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
    }

    /// The current footprint, or `0` if the kernel will not say (pacing then never waits).
    pub(super) fn current() -> u64 {
        let mut info = [0u64; 64];
        // SAFETY: `info` outlives the call and is larger than `rusage_info_v4`.
        let rc = unsafe { proc_pid_rusage(std::process::id() as i32, 4, info.as_mut_ptr()) };
        if rc == 0 {
            info[9]
        } else {
            0
        }
    }

    /// Wait (bounded) until `read` reports at most `limit`; how long it waited.
    pub(super) fn wait_until_at_most(read: &mut dyn FnMut() -> u64, limit: u64) -> Duration {
        let start = Instant::now();
        while read() > limit && start.elapsed() < MAX_WAIT {
            std::thread::sleep(Duration::from_millis(2));
        }
        start.elapsed()
    }

    /// The longest one group waits: the driver's release was measured at ≤ 1.25 s.
    const MAX_WAIT: Duration = Duration::from_secs(3);
}

impl Weights {
    /// Construct directly from an in-memory map (used by converters and tests).
    pub fn from_map(tensors: HashMap<String, Array>) -> Self {
        Self {
            tensors,
            accessed: RefCell::new(HashSet::new()),
            released: HashSet::new(),
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
        if self.released.contains(key) {
            return Err(released(key));
        }
        let value = self
            .tensors
            .get(key)
            .ok_or_else(|| Error::MissingTensor(key.to_string()))?;
        self.accessed.borrow_mut().insert(key.to_owned());
        Ok(value)
    }

    /// Fetch a tensor by key if present. Records the key in the access set. A released key reads
    /// as absent: its bytes now live only in the model that consumed them.
    pub fn get(&self, key: &str) -> Option<&Array> {
        let value = self.tensors.get(key)?;
        self.accessed.borrow_mut().insert(key.to_owned());
        Some(value)
    }

    /// Whether a key is present (a released key still is: the checkpoint carries it).
    pub fn contains(&self, key: &str) -> bool {
        self.tensors.contains_key(key) || self.released.contains(key)
    }

    /// Number of loaded tensors, released ones included.
    pub fn len(&self) -> usize {
        self.tensors.len() + self.released.len()
    }

    /// Whether no tensors are loaded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// All loaded tensor keys, released ones included.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tensors
            .keys()
            .chain(self.released.iter())
            .map(|s| s.as_str())
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

    /// Materialize a freshly built model **group by group**, releasing each group's consumed
    /// sources as it goes (sc-24446) — the load order MLX load admission prices.
    ///
    /// `groups` are the model's own arrays (dense weights, casts, quantized triples, stacked
    /// expert banks, norm results) in build order, one group per decoder layer plus the arrays
    /// outside the layer stack ([`crate::models::CausalLm::param_groups`],
    /// [`crate::models::Qwen35Model::param_groups`]). For each group:
    ///
    /// 1. its pending safetensors `Load`s are read on the CPU stream
    ///    ([`mlx_rs::transforms::eval_pending_loads`]) — so no GPU op below ever waits on the disk
    ///    (sc-24245);
    /// 2. every source that read made resident is checked for a coherent GPU view
    ///    ([`coherence::verify_gpu_view`](crate::primitives::coherence::verify_gpu_view), sc-22414);
    /// 3. the group is evaluated (quantize, cast, split, stack — all GPU work over resident
    ///    inputs);
    /// 4. this map's handles on those sources are dropped and MLX's buffer cache is cleared, so a
    ///    source consumed into a different array (a quantized projection, a BF16 cast, a stacked
    ///    bank) is returned to the system before the next group reads its own.
    ///
    /// Peak resident memory is therefore the arrays already built plus **one group's** sources and
    /// conversions — never the whole payload beside the whole derived set, which is what verifying
    /// every source up front and deriving lazily on the first request held.
    ///
    /// **Pacing.** The Metal driver returns a released buffer to the system asynchronously
    /// (measured 0.1–1.25 s after `clear_cache`), so a fast load could read group after group
    /// while earlier groups' sources still count against the process. Before each group reads,
    /// the process's `phys_footprint` is held (waiting, at most 3 s) to what it had at the start
    /// plus the arrays built so far, the previous group's released sources (those consumed into
    /// different arrays — a source the model keeps as stored shares its buffer and is not
    /// released) and [`MATERIALIZE_PACE_SLACK_BYTES`] — so at most one released group is
    /// outstanding.
    ///
    /// Every accessed source is still evaluated and verified before this returns, so the load
    /// boundary guarantee of [`Weights::verify_accessed_gpu_view`] holds: a source no group consumed
    /// (an enumeration gap) is read, verified and kept, and counted in
    /// [`Materialized::leftover`]. Sources that were already resident when this was called (an
    /// in-memory map) are verified and kept. Released keys stay [`contained`](Weights::contains).
    pub fn materialize_groups(&mut self, groups: &[Vec<Array>]) -> Result<Materialized> {
        self.materialize_groups_paced(groups, &mut footprint::current)
    }

    /// [`Weights::materialize_groups`] with the footprint reader injectable — the seam the pacing
    /// test drives a slow driver release through.
    fn materialize_groups_paced(
        &mut self,
        groups: &[Vec<Array>],
        read_footprint: &mut dyn FnMut() -> u64,
    ) -> Result<Materialized> {
        let mut accessed: Vec<String> = self.accessed.borrow().iter().cloned().collect();
        accessed.sort_unstable();
        let (resident, mut pending): (Vec<String>, Vec<String>) = accessed
            .into_iter()
            .filter(|key| self.tensors.contains_key(key))
            .partition(|key| is_available(&self.tensors[key]));
        crate::primitives::coherence::verify_gpu_view(
            resident
                .iter()
                .map(|key| (key.as_str(), &self.tensors[key])),
        )?;
        let mut report = Materialized::default();
        // Pacing (see the method docs): what the footprint may hold before the next group reads.
        let start = read_footprint();
        let mut built = 0u64;
        let mut released_last = 0u64;
        for group in groups.iter().filter(|g| !g.is_empty()) {
            let waited = footprint::wait_until_at_most(
                read_footprint,
                start
                    .saturating_add(built)
                    .saturating_add(released_last)
                    .saturating_add(MATERIALIZE_PACE_SLACK_BYTES),
            );
            report.paced = report.paced.saturating_add(waited);
            mlx_rs::transforms::eval_pending_loads(group.iter())?;
            let (read, rest): (Vec<String>, Vec<String>) = pending
                .into_iter()
                .partition(|key| is_available(&self.tensors[key]));
            pending = rest;
            crate::primitives::coherence::verify_gpu_view(
                read.iter().map(|key| (key.as_str(), &self.tensors[key])),
            )?;
            mlx_rs::transforms::eval(group.iter())?;
            built = group
                .iter()
                .fold(built, |b, a| b.saturating_add(a.nbytes() as u64));
            // What the driver now owes back: the sources consumed into different arrays. A
            // source the model keeps as one of its own arrays (a BF16 matrix used as stored)
            // shares its buffer and is not released.
            let kept: HashSet<usize> = group.iter().map(data_address).collect();
            released_last = read
                .iter()
                .map(|key| &self.tensors[key])
                .filter(|a| !kept.contains(&data_address(a)))
                .map(|a| a.nbytes() as u64)
                .sum();
            for key in read {
                self.tensors.remove(&key);
                self.released.insert(key);
                report.released += 1;
            }
            mlx_rs::memory::clear_cache();
            report.cache_after_groups = report
                .cache_after_groups
                .max(mlx_rs::memory::get_cache_memory());
            report.groups += 1;
        }
        report.leftover = pending.len();
        let mut batch: Vec<(&str, &Array)> = Vec::new();
        let mut bytes = 0usize;
        for key in &pending {
            let array = &self.tensors[key];
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
        Ok(report)
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

/// The error a read of a [released](Weights::materialize_groups) key returns.
fn released(key: &str) -> Error {
    Error::Msg(format!(
        "tensor `{key}` was released once the model's load materialized it; read it before \
         materializing, or from a fresh `Weights`"
    ))
}

/// The address of `a`'s data (evaluated): equal for two handles on one buffer.
fn data_address(a: &Array) -> usize {
    // SAFETY: `a` is a live, evaluated array; the accessor only reads its data pointer (MLX's
    // `array::data<T>` reinterprets the pointer whatever the dtype).
    unsafe { mlx_sys::mlx_array_data_uint8(a.as_ptr()) as usize }
}

/// Whether `a` holds its data (evaluated, or built from host memory) — MLX's `is_available`,
/// which mlx-rs does not expose.
pub(crate) fn is_available(a: &Array) -> bool {
    let mut available = false;
    // SAFETY: `a.as_ptr()` is a live `mlx_array` for the duration of the call, and the out
    // pointer is a valid `bool`.
    let status = unsafe { mlx_sys::_mlx_array_is_available(&mut available, a.as_ptr()) };
    status == 0 && available
}

/// Evaluate a model's own arrays group by group, clearing MLX's buffer cache after each, once its
/// constructor has verified every source ([`Weights::verify_accessed_gpu_view`]). The
/// non-releasing twin of [`Weights::materialize_groups`] for a caller-owned map: every derived
/// array (quantized projection, cast, split, stacked bank) exists before the constructor returns,
/// so a forward never builds one — and never reads a source — inside its command stream
/// (sc-24446, sc-24245).
pub(crate) fn eval_groups(groups: &[Vec<Array>]) -> Result<()> {
    for group in groups.iter().filter(|g| !g.is_empty()) {
        mlx_rs::transforms::eval(group.iter())?;
        mlx_rs::memory::clear_cache();
    }
    Ok(())
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

    /// `layers` f32 `[rows, 1024]` "layers" (consumed into a half-size BF16 cast) and one BF16
    /// matrix of the same element count kept as stored, saved to a file and lazily reopened.
    fn materialize_fixture(dir: &Fixture, layers: usize, rows: i32) -> Weights {
        let path = dir.join("model.safetensors");
        let layer = |i: usize| {
            let data: Vec<f32> = (0..rows as usize * 1024)
                .map(|j| ((i * 7 + j) % 97) as f32 * 0.01)
                .collect();
            Array::from_slice(&data, &[rows, 1024])
        };
        let mut tensors: Vec<(String, Array)> =
            (0..layers).map(|i| (format!("l{i}.w"), layer(i))).collect();
        tensors.push((
            "keep".into(),
            layer(99).as_dtype(mlx_rs::Dtype::Bfloat16).unwrap(),
        ));
        Array::save_safetensors(tensors.iter().map(|(k, v)| (k.as_str(), v)), None, &path).unwrap();
        Weights::from_file(&path).unwrap()
    }

    fn cast_groups(w: &Weights, layers: usize) -> Vec<Vec<Array>> {
        let mut groups: Vec<Vec<Array>> = (0..layers)
            .map(|i| {
                vec![w
                    .require(&format!("l{i}.w"))
                    .unwrap()
                    .as_dtype(mlx_rs::Dtype::Bfloat16)
                    .unwrap()]
            })
            .collect();
        groups.push(vec![w.require("keep").unwrap().clone()]);
        groups
    }

    /// sc-24446: [`Weights::materialize_groups`] reads, verifies, converts and **releases** one
    /// group at a time — measured as what macOS charges the process, its exact `phys_footprint`
    /// peak — so it holds the arrays already built plus one group's sources and the previous
    /// group's not-yet-returned ones, never every source beside every conversion; MLX's cache is
    /// empty after every group, and nothing stays lazy.
    ///
    /// Eight 64 MiB F32 layers cast to BF16: holding every source beside every cast would be
    /// 512 + 256 + 32 MiB; the paced order peaks under the 288 MiB built, two layers' sources and
    /// the pacing slack.
    ///
    /// MUTATION: keep the map's handle (`self.tensors.remove(&key)` skipped) or skip the
    /// `clear_cache`, and this goes RED. (Pacing is pinned deterministically by
    /// `a_group_waits_for_the_driver_to_return_the_previous_groups_sources`: on an idle machine
    /// the driver happens to release fast enough here that its absence moves this peak by only
    /// ~30 MiB.)
    #[test]
    fn materialize_groups_releases_each_groups_consumed_sources() {
        use crate::test_fixture::footprint;
        const MIB: u64 = 1024 * 1024;
        let dir = Fixture::new("mlx-llm-materialize-", None);
        let mut w = materialize_fixture(&dir, 8, 16 * 1024);
        let groups = cast_groups(&w, 8);
        mlx_rs::memory::clear_cache();
        footprint::settle();
        let base = footprint::current();
        footprint::reset_peak();
        let report = w.materialize_groups(&groups).unwrap();
        let peak = footprint::peak_since_reset().saturating_sub(base);
        assert_eq!((report.groups, report.released, report.leftover), (9, 9, 0));
        assert_eq!(
            report.cache_after_groups, 0,
            "MLX's cache kept a consumed buffer"
        );
        assert!(
            groups.iter().flatten().all(is_available),
            "a lazy array left"
        );
        let built = 8 * 32 * MIB + 32 * MIB;
        eprintln!(
            "materialize: footprint peak {} MiB, paced {:?}",
            peak / MIB,
            report.paced
        );
        assert!(
            peak <= built + 2 * 64 * MIB + MATERIALIZE_PACE_SLACK_BYTES,
            "footprint peak {} MiB: more than one group's sources beside the built arrays \
             (paced {:?})",
            peak / MIB,
            report.paced
        );
        // A released key is still in the checkpoint, but no longer readable.
        assert!(w.contains("l0.w") && w.keys().any(|k| k == "l0.w"));
        assert!(w.require("l0.w").is_err());
        assert!(w.get("l0.w").is_none());
    }

    /// The pacing (see [`Weights::materialize_groups`]): with a driver that has not yet returned
    /// the previous group's sources — a footprint that stays high for 40 reads — the next group
    /// waits until it drops, and only then reads.
    ///
    /// MUTATION: make `wait_until_at_most` return at once and this goes RED.
    #[test]
    fn a_group_waits_for_the_driver_to_return_the_previous_groups_sources() {
        let dir = Fixture::new("mlx-llm-materialize-", None);
        let mut w = materialize_fixture(&dir, 2, 256);
        let groups = cast_groups(&w, 2);
        let reads = std::cell::Cell::new(0u32);
        // The first read is the start; the first group's pacing check passes; then the driver
        // "holds" 1 TiB for 40 reads before releasing.
        let mut fake = || {
            reads.set(reads.get() + 1);
            if (3..43).contains(&reads.get()) {
                1 << 40
            } else {
                1 << 30
            }
        };
        let report = w.materialize_groups_paced(&groups, &mut fake).unwrap();
        assert_eq!(report.groups, 3);
        // Clock-free: the second group polled the footprint until the driver released (read 43),
        // and only then read its sources.
        assert!(
            reads.get() >= 44,
            "the second group read while the first group's sources were outstanding ({} reads)",
            reads.get()
        );
    }

    /// A source no group consumed is still read and verified before the load returns (the
    /// sc-24245 load boundary), kept in the map, and counted as a leftover.
    #[test]
    fn a_source_no_group_consumes_is_read_verified_and_kept() {
        let dir = Fixture::new("mlx-llm-materialize-", None);
        let mut w = materialize_fixture(&dir, 4, 256);
        let consumed = w
            .require("l0.w")
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .unwrap();
        let orphan = w.require("l1.w").unwrap().clone();
        assert!(!is_available(&orphan));
        let report = w.materialize_groups(&[vec![consumed]]).unwrap();
        assert_eq!((report.released, report.leftover), (1, 1));
        assert!(is_available(w.require("l1.w").unwrap()));
        assert!(is_available(&orphan));
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
