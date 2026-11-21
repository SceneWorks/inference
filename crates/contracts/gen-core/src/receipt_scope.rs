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
    pin: Option<std::result::Result<PinnedWeightsFile, String>>,
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
                // Capture an unprepared primary before a reader mmaps it. A domain-only load may
                // name a missing file; deliver that error only if a real reader describes it.
                pin: Some(match spec.prepared_file_pin_for(path)? {
                    Some(pin) => Ok(pin.clone()),
                    None => PinnedWeightsFile::pin(path).map_err(|e| e.to_string()),
                }),
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
        let mut readers = scope.readers.lock().unwrap_or_else(|e| e.into_inner());
        let group = readers.entry(paths.clone()).or_default();
        if let Some(existing) = group.iter().find(|r| r.same_plan(&plan, &capability)) {
            return existing.clone();
        }
        let reader = crate::CheckpointReadAccounting::create(plan, capability);
        group.push(reader.clone());
        reader
    }

    fn bind(
        &self,
        paths: &[PathBuf],
        facts: CheckpointWeightFacts,
    ) -> crate::Result<CheckpointWeightFacts> {
        let mut pins = Vec::with_capacity(paths.len());
        for path in paths {
            let mut pin = None;
            if let Some(spec) = &self.prepared {
                pin = spec.prepared_file_pins().get(path);
                if pin.is_none() {
                    if let Some(inventory) = spec.source_inventory() {
                        pin = inventory.member_pin(path)?;
                    }
                }
            }
            if pin.is_none() && self.primary.as_ref() == Some(path) {
                pin = Some(
                    self.pin
                        .as_ref()
                        .ok_or_else(|| crate::Error::Msg("missing observed checkpoint pin".into()))?
                        .as_ref()
                        .map_err(|e| crate::Error::Msg(e.clone()))?,
                );
            }
            if let Some(pin) = pin {
                pins.push(pin);
            }
        }
        if pins.is_empty() {
            return Ok(facts);
        }
        if pins.len() != paths.len() {
            return Err(crate::Error::Unsupported("a measured checkpoint shard set is only partially bound to its prepared source inventory".into()));
        }
        facts.with_verified_sources(pins)
    }
    /// Register a lightweight snapshot callback, replacing the previous load of the same shards.
    pub fn register(
        paths: &[PathBuf],
        snapshot: impl Fn() -> crate::Result<CheckpointWeightFacts> + Send + Sync + 'static,
    ) {
        if let Some(scope) = Self::current() {
            let paths = source_paths(paths);
            let snapshot: Snapshot = Arc::new(snapshot);
            scope
                .sources
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(paths, snapshot);
        }
    }

    fn snapshot_groups(&self) -> crate::Result<Vec<(Vec<PathBuf>, CheckpointWeightFacts)>> {
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
            let bound = self.bind(&paths, value)?;
            facts.push((paths, bound));
        }
        let readers = self
            .readers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        for (paths, group) in readers {
            let value = crate::CheckpointReadAccounting::combined(&group)?;
            if value.receipt().tensor_count != 0 {
                let bound = self.bind(&paths, value)?;
                facts.push((paths, bound));
            }
        }
        Ok(facts)
    }

    /// Every component with a measured read, bound to its original prepared source shards.
    pub fn try_snapshots(&self) -> crate::Result<Vec<CheckpointWeightFacts>> {
        Ok(self
            .snapshot_groups()?
            .into_iter()
            .map(|(_, facts)| facts)
            .collect())
    }

    pub fn snapshots(&self) -> Vec<CheckpointWeightFacts> {
        self.try_snapshots().unwrap_or_default()
    }

    /// The exact File primary, or the unique measured checkpoint for a directory load.
    /// A multi-component directory has no implicit primary; all components remain available in snapshots.
    pub fn facts(&self) -> Option<CheckpointWeightFacts> {
        let mut snapshots = self.snapshot_groups().ok()?;
        if let Some(primary) = &self.primary {
            snapshots
                .into_iter()
                .find(|(paths, _)| paths.contains(primary))
                .map(|(_, facts)| facts)
        } else if snapshots.len() == 1 {
            snapshots.pop().map(|(_, facts)| facts)
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
