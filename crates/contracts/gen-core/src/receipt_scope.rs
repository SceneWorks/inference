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
type Readers = BTreeMap<Vec<PathBuf>, Vec<Arc<crate::CheckpointReadAccounting>>>;
type Sources = BTreeMap<Vec<PathBuf>, Snapshot>;
thread_local! {
    static CURRENT: RefCell<Option<CheckpointReceiptScope>> = const { RefCell::new(None) };
}

#[derive(Clone, Default)]
pub struct CheckpointReceiptScope {
    sources: Arc<Mutex<Sources>>,
    readers: Arc<Mutex<Readers>>,

    prepared: Option<Arc<LoadSpec>>,
    primary: Option<PathBuf>,
    pin: Option<PinnedWeightsFile>,
    observed_pin: Arc<std::sync::OnceLock<std::result::Result<PinnedWeightsFile, String>>>,
}

impl CheckpointReceiptScope {
    pub fn for_load(spec: &LoadSpec) -> crate::Result<Self> {
        let prepared = spec.has_source_guard().then(|| Arc::new(spec.clone()));
        match &spec.weights {
            crate::WeightsSource::Dir(_) => Ok(Self {
                prepared,
                ..Self::default()
            }),
            crate::WeightsSource::File(path) => Ok(Self {
                prepared,
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

    /// Hold the authoritative preparation/inventory guard across deferred opens and execution.
    /// This preserves the same before/after source checks as the initial registry constructor.
    pub fn with_prepared<T>(&self, f: impl FnOnce() -> crate::Result<T>) -> crate::Result<T> {
        self.with(|| match &self.prepared {
            Some(spec) => spec.read_prepared_files_unchanged(f),
            None => f(),
        })
    }
    /// Explicit propagation for a provider that opens a new reader on a worker thread.
    pub fn current() -> Option<Self> {
        CURRENT.with(|s| s.borrow().clone())
    }

    pub(crate) fn share_reader(
        paths: &[PathBuf],
        plan: Arc<crate::LogicalWeightPlan>,
        capability: crate::NativeExecutionCapability,
    ) -> Arc<crate::CheckpointReadAccounting> {
        let Some(scope) = Self::current() else {
            return crate::CheckpointReadAccounting::create(plan, capability);
        };
        let paths = source_paths(paths);
        scope.pin_primary(&paths);
        let mut readers = scope.readers.lock().unwrap_or_else(|e| e.into_inner());
        let group = readers.entry(paths.clone()).or_default();
        if let Some(existing) = group.iter().find(|r| r.same_plan(&plan, &capability)) {
            return existing.clone();
        }
        let reader = crate::CheckpointReadAccounting::create(plan, capability);
        group.push(reader.clone());
        reader
    }

    fn pin_primary(&self, paths: &[PathBuf]) {
        if let Some(path) = self.primary.as_ref().filter(|p| paths.contains(p)) {
            self.observed_pin.get_or_init(|| {
                self.pin
                    .clone()
                    .map(Ok)
                    .unwrap_or_else(|| PinnedWeightsFile::pin(path))
                    .map_err(|e| e.to_string())
            });
        }
    }
    fn bind(
        &self,
        paths: &[PathBuf],
        facts: CheckpointWeightFacts,
    ) -> crate::Result<CheckpointWeightFacts> {
        if self
            .primary
            .as_ref()
            .is_some_and(|primary| paths.contains(primary))
        {
            let pin = self
                .observed_pin
                .get()
                .ok_or_else(|| crate::Error::Msg("missing observed checkpoint pin".into()))?
                .as_ref()
                .map_err(|e| crate::Error::Msg(e.clone()))?;
            facts.with_verified_source(pin)
        } else {
            Ok(facts)
        }
    }
    /// Register a lightweight snapshot callback, replacing the previous load of the same shards.
    pub fn register(
        paths: &[PathBuf],
        snapshot: impl Fn() -> crate::Result<CheckpointWeightFacts> + Send + Sync + 'static,
    ) {
        if let Some(scope) = Self::current() {
            let paths = source_paths(paths);
            scope.pin_primary(&paths);
            let snapshot: Snapshot = Arc::new(snapshot);
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
        for (paths, snapshot) in readers {
            let value = snapshot()?;
            if value.receipt().tensor_count == 0 {
                continue;
            }
            facts.push(self.bind(&paths, value)?);
        }
        let readers = self
            .readers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        for (paths, group) in readers {
            let value = crate::CheckpointReadAccounting::combined(&group)?;
            if value.receipt().tensor_count != 0 {
                facts.push(self.bind(&paths, value)?);
            }
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

fn source_paths(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut paths: Vec<_> = paths.iter().map(|p| absolute(p)).collect();
    paths.sort();
    paths.dedup();
    paths
}
