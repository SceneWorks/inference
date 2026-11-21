//! Per-loaded-model collection of checkpoint receipts. Readers retain only their measurement
//! state in the collector. The currently executing load or inference enters its own scope.
use crate::{CheckpointWeightFacts, LoadSpec, PinnedWeightsFile};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

type Snapshot = Arc<dyn Fn() -> crate::Result<CheckpointWeightFacts> + Send + Sync>;
type Sources = BTreeMap<Vec<PathBuf>, Snapshot>;
thread_local! {
    static CURRENT: RefCell<Option<CheckpointReceiptScope>> = const { RefCell::new(None) };
}

#[derive(Clone, Default)]
pub struct CheckpointReceiptScope {
    sources: Arc<Mutex<Sources>>,
    primary: Option<PathBuf>,
    pin: Option<PinnedWeightsFile>,
}

impl CheckpointReceiptScope {
    pub fn for_load(spec: &LoadSpec) -> crate::Result<Self> {
        match &spec.weights {
            crate::WeightsSource::Dir(_) => Ok(Self::default()),
            crate::WeightsSource::File(path) => Ok(Self {
                primary: Some(absolute(path)),
                pin: spec.prepared_file_pin_for(path)?.cloned(),
                ..Self::default()
            }),
        }
    }

    /// Install this load only for the dynamic extent of the callback, restoring its parent on unwind.
    pub fn with<T>(&self, f: impl FnOnce() -> T) -> T {
        struct Restore(Option<CheckpointReceiptScope>);
        impl Drop for Restore {
            fn drop(&mut self) {
                CURRENT.with(|s| s.replace(self.0.take()));
            }
        }
        let _restore = Restore(CURRENT.with(|s| s.replace(Some(self.clone()))));
        f()
    }

    /// Explicit propagation for a provider that opens a new reader on a worker thread.
    pub fn current() -> Option<Self> {
        CURRENT.with(|s| s.borrow().clone())
    }

    /// Register a lightweight snapshot callback, replacing the previous load of the same shards.
    pub fn register(
        paths: &[PathBuf],
        snapshot: impl Fn() -> crate::Result<CheckpointWeightFacts> + Send + Sync + 'static,
    ) {
        if let Some(scope) = Self::current() {
            let mut paths: Vec<_> = paths.iter().map(|p| absolute(p)).collect();
            paths.sort();
            paths.dedup();
            let snapshot: Snapshot =
                if let Some(primary) = scope.primary.as_ref().filter(|p| paths.contains(p)) {
                    let pin = match &scope.pin {
                        Some(pin) => Ok(pin.clone()),
                        None => PinnedWeightsFile::pin(primary),
                    }
                    .map_err(|e| e.to_string());
                    Arc::new(move || {
                        snapshot()?.with_verified_source(
                            pin.as_ref().map_err(|e| crate::Error::Msg(e.clone()))?,
                        )
                    })
                } else {
                    Arc::new(snapshot)
                };
            scope
                .sources
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(paths, snapshot);
        }
    }

    /// Every component with a measured read. The primary single-file source is reverified.
    pub fn try_snapshots(&self) -> crate::Result<Vec<CheckpointWeightFacts>> {
        let readers = self
            .sources
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let mut facts = Vec::new();
        for snapshot in readers.into_values() {
            let value = snapshot()?;
            if value.receipt().tensor_count == 0 {
                continue;
            }
            facts.push(value);
        }
        Ok(facts)
    }

    pub fn snapshots(&self) -> Vec<CheckpointWeightFacts> {
        self.try_snapshots().unwrap_or_default()
    }

    /// The exact File primary, or the unique measured checkpoint for a directory load.
    /// A multi-component directory has no implicit primary; all components remain available in snapshots.
    pub fn facts(&self) -> Option<CheckpointWeightFacts> {
        let mut snapshots = self.try_snapshots().ok()?;
        if self.primary.is_some() {
            snapshots
                .into_iter()
                .find(|value| value.source_binding().is_some())
        } else if snapshots.len() == 1 {
            snapshots.pop()
        } else {
            None
        }
    }
}

fn absolute(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| path.to_owned())
}
