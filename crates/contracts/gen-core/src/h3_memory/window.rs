//! MLX native H3 deferred-DiT residency. Source: `block_stream.rs::{DitBlockStream,
//! precompute_adaln_windowed}` and `dit/model.rs::{load_dir_deferred,new_windowed,
//! forward_windowed}`. Fresh weights views and forced carried state bound each ACTUAL aligned
//! BlockPlan window; refiner/root tensors are not streamed. This is structural arithmetic, not a
//! measured peak and not permission to advertise Candle streaming.

use std::collections::{BTreeMap, BTreeSet};

use super::{NativeBackend, NativeProjection, NativeRows, NativeTaskAllocation, RowDomain};
use crate::block_window::BlockPlan;
use crate::{Error, Result};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Partition {
    payload: u64,
    /// Subset forced by native descriptor validation even before block source verification.
    eager_payload: u64,
    /// Largest one source-to-compute cast, not a sum of all cast outputs.
    forward_cast: u64,
    descriptors: u64,
    rotations: BTreeSet<u64>,
}
impl Partition {
    fn resident(&self) -> Result<u64> {
        plus(self.payload, rotation_bytes(&self.rotations)?)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Layer {
    body: Partition,
    adaln: Partition,
}

/// Payload-only partition, built from the selected task's actual materialized tensors. Descriptor
/// blobs and synthesized shared rotations must NOT appear in `resident_by_name`. Root adjustment
/// is only the existing S5 compute/retained-source/cache DELTA over these payloads. Pruned reduced
/// projections remain root-resident, even though their source names contain block indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowAllocation {
    root: Partition,
    layers: Vec<Layer>,
    pruned: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Location {
    Root,
    Body(usize),
    Adaln(usize),
}
fn invalid() -> Error {
    Error::Msg(
        "H3 native window allocation: inconsistent, unsupported, or overflowing facts".into(),
    )
}
fn plus(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(invalid)
}
fn times(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(invalid)
}
fn rotation_bytes(groups: &BTreeSet<u64>) -> Result<u64> {
    groups
        .iter()
        .try_fold(0, |bytes, &g| plus(bytes, times(times(g, g)?, 4)?))
}

fn locate(name: &str) -> Result<Location> {
    let Some(tail) = name.strip_prefix("transformer_blocks.") else {
        return Ok(Location::Root);
    };
    let (index, suffix) = tail.split_once('.').ok_or_else(invalid)?;
    let layer = index.parse::<usize>().map_err(|_| invalid())?;
    if index != layer.to_string() || suffix.is_empty() {
        return Err(invalid());
    }
    if suffix.starts_with("adaln_proj.linear") {
        Ok(Location::Adaln(layer))
    } else {
        Ok(Location::Body(layer))
    }
}

impl WindowAllocation {
    pub fn from_resident_tensors(
        resident_by_name: &[(String, u64)],
        projections: &[NativeProjection],
        root_adjustment: u64,
        pruned: bool,
    ) -> Result<Self> {
        let mut root = Partition {
            payload: root_adjustment,
            ..Default::default()
        };
        let mut layers = BTreeMap::<usize, Layer>::new();
        // Constructors validate these small operands through as_slice BEFORE staged source
        // verification. Codes and ordinary dense bodies remain lazy; do not force them into
        // precompute merely because their source Array handles already exist.
        let eager_names = projections
            .iter()
            .flat_map(|p| {
                let suffixes: &[&str] = if p.w4a8 {
                    &["weight_s_rel", "weight_s_channel", "weight_codebook"]
                } else {
                    &["weight_scale"]
                };
                suffixes
                    .iter()
                    .map(move |suffix| format!("{}.{suffix}", p.base))
            })
            .collect::<BTreeSet<_>>();
        let mut names = BTreeSet::new();
        for (name, bytes) in resident_by_name {
            if name.ends_with(".comfy_quant") || !names.insert(name) {
                return Err(invalid());
            }
            let destination = match locate(name)? {
                Location::Root => &mut root,
                Location::Adaln(i) if pruned => {
                    layers.entry(i).or_default();
                    &mut root
                }
                Location::Adaln(i) => &mut layers.entry(i).or_default().adaln,
                Location::Body(i) => &mut layers.entry(i).or_default().body,
            };
            destination.payload = plus(destination.payload, *bytes)?;
            if eager_names.contains(name) {
                destination.eager_payload = plus(destination.eager_payload, *bytes)?;
            }
        }
        if layers.is_empty() || layers.keys().copied().ne(0..layers.len()) {
            return Err(invalid());
        }
        for p in projections {
            let destination = match locate(&p.base)? {
                Location::Root => &mut root,
                Location::Adaln(i) if pruned => {
                    if !layers.contains_key(&i) {
                        return Err(invalid());
                    }
                    &mut root
                }
                Location::Adaln(i) => &mut layers.get_mut(&i).ok_or_else(invalid)?.adaln,
                Location::Body(i) => &mut layers.get_mut(&i).ok_or_else(invalid)?.body,
            };
            destination.descriptors = plus(destination.descriptors, p.descriptor_bytes)?;
            // ConvRot matrices are shared by group inside one Weights view. W4A8 uses a
            // radix FHT and has no dense resident rotation, regardless of rotation_group.
            if !p.w4a8 {
                if p.rotation_group == 0 {
                    return Err(invalid());
                }
                destination.rotations.insert(p.rotation_group);
            }
        }
        Ok(Self {
            root,
            layers: layers.into_values().collect(),
            pruned,
        })
    }

    /// Add only actual current source-to-compute cast OUTPUT sizes supplied by the selected
    /// loader policy. Endpoint payloads are already max(source, compute), not their sum. A cast
    /// can first evaluate during forward; it therefore overlaps operation work, not just load.
    pub fn with_source_casts(mut self, casts: &[(String, u64)]) -> Result<Self> {
        for (name, bytes) in casts {
            let partition = match locate(name)? {
                Location::Root => &mut self.root,
                Location::Adaln(i) if self.pruned => {
                    if i >= self.layers.len() {
                        return Err(invalid());
                    }
                    &mut self.root
                }
                Location::Adaln(i) => &mut self.layers.get_mut(i).ok_or_else(invalid)?.adaln,
                Location::Body(i) => &mut self.layers.get_mut(i).ok_or_else(invalid)?.body,
            };
            partition.forward_cast = partition.forward_cast.max(*bytes);
        }
        Ok(self)
    }

    pub fn n_blocks(&self) -> usize {
        self.layers.len()
    }

    /// Resident-only counterpart for the ordinary native weight floor. Includes root/refiner,
    /// actual fresh-view groups and retained modulation cache, but NO operation scratch or
    /// descriptor blobs. This avoids placing a full-component ordinary floor underneath a real
    /// window=1 allocation prediction. Full-component single-view group storage is not forced to
    /// equal deferred root+fresh-window storage; those are genuinely different lifetimes.
    pub fn resident_peak_bytes(
        &self,
        task: &NativeTaskAllocation,
        backend: NativeBackend,
        rows: NativeRows,
        window: usize,
    ) -> Result<u64> {
        let plan = self.plan(backend, window)?;
        let root = self.root.resident()?;
        let table = self.table_bytes(task, rows)?;
        let mut peak = plus(root, table)?;
        for range in plan.windows() {
            let body = self.window_partition(range.clone(), false)?.resident()?;
            peak = peak.max(plus(plus(root, table)?, body)?);
            if !self.pruned {
                let adaln = self.window_partition(range.clone(), true)?.resident()?;
                let produced = times(table / self.layers.len() as u64, range.end as u64)?;
                peak = peak.max(plus(plus(root, produced)?, adaln)?);
            }
        }
        Ok(peak)
    }

    fn plan(&self, backend: NativeBackend, window: usize) -> Result<BlockPlan> {
        if !matches!(backend, NativeBackend::Mlx { .. }) {
            return Err(invalid());
        }
        BlockPlan::new(self.layers.len(), window)
    }

    fn table_bytes(&self, task: &NativeTaskAllocation, rows: NativeRows) -> Result<u64> {
        // Pruned source operands and the admitted-domain retained cache already live in the S5
        // root adjustment; those projections are never evicted or rematerialized per window.
        if self.pruned {
            if task.table_bytes_per_timestep != 0 || task.evicted_bytes != 0 {
                return Err(invalid());
            }
            return Ok(0);
        }
        if !task
            .table_bytes_per_timestep
            .is_multiple_of(self.layers.len() as u64)
        {
            return Err(invalid());
        }
        times(task.table_bytes_per_timestep, rows.schedule)
    }

    fn is_root(&self, projection: &NativeProjection) -> Result<bool> {
        Ok(match locate(&projection.base)? {
            Location::Root => true,
            Location::Adaln(_) => self.pruned,
            Location::Body(_) => false,
        })
    }

    fn root_working(
        &self,
        task: &NativeTaskAllocation,
        backend: NativeBackend,
        rows: NativeRows,
        conditioning: bool,
    ) -> Result<u64> {
        let mut peak = 0;
        for p in &task.projections {
            if self.is_root(p)? && (p.rows == RowDomain::Conditioning) == conditioning {
                let work = p.working_bytes(backend, rows);
                // S5 already counts the pruned modulation cache backing these output arrays.
                let work = if self.pruned && p.rows == RowDomain::Schedule {
                    work.saturating_sub(
                        p.output
                            .saturating_mul(p.activation_width)
                            .saturating_mul(rows.schedule),
                    )
                } else {
                    work
                };
                peak = peak.max(work);
            }
        }
        plus(peak, self.root.forward_cast)
    }

    /// Exact aligned window's payload and fresh-view shared matrices. Root and window groups
    /// intentionally do NOT deduplicate against each other: their Weights views are distinct and
    /// both are live. Consecutive windows release their view, so they take a maximum, not a sum.
    fn window_partition(&self, range: std::ops::Range<usize>, adaln: bool) -> Result<Partition> {
        let mut result = Partition::default();
        for layer in &self.layers[range] {
            let p = if adaln { &layer.adaln } else { &layer.body };
            result.payload = plus(result.payload, p.payload)?;
            result.eager_payload = plus(result.eager_payload, p.eager_payload)?;
            result.forward_cast = result.forward_cast.max(p.forward_cast);
            result.descriptors = plus(result.descriptors, p.descriptors)?;
            result.rotations.extend(p.rotations.iter().copied());
        }
        Ok(result)
    }

    /// A staged full stack has ONE shared Weights context, unlike deferred root/fresh-window
    /// views. Its ConvRot group matrices deduplicate across root, body and AdaLN together.
    fn full_view(&self, backend: NativeBackend) -> Result<(u64, Partition, Partition)> {
        self.plan(backend, self.layers.len())?;
        let body = self.window_partition(0..self.layers.len(), false)?;
        let adaln = self.window_partition(0..self.layers.len(), true)?;
        let mut groups = self.root.rotations.clone();
        groups.extend(body.rotations.iter().copied());
        groups.extend(adaln.rotations.iter().copied());
        Ok((
            plus(self.root.payload, rotation_bytes(&groups)?)?,
            body,
            adaln,
        ))
    }

    /// Ordinary resident-weight counterpart for the no-window MLX schedule. The full AdaLN
    /// stack is precomputed and evicted BEFORE body source verification. Native scale/codebook
    /// operands are the exception: constructors already read them for validation, and their
    /// backing arrays remain live throughout precompute. Descriptors and casts are work, not
    /// endpoint residency, and are intentionally excluded here.
    pub fn resident_peak_bytes_full(
        &self,
        task: &NativeTaskAllocation,
        backend: NativeBackend,
        rows: NativeRows,
    ) -> Result<u64> {
        let (root, body, adaln) = self.full_view(backend)?;
        let table = self.table_bytes(task, rows)?;
        let precompute = plus(plus(plus(root, body.eager_payload)?, adaln.payload)?, table)?;
        let denoise = plus(plus(root, body.payload)?, table)?;
        Ok(precompute.max(denoise))
    }

    pub fn precompute_peak_bytes_full(
        &self,
        task: &NativeTaskAllocation,
        backend: NativeBackend,
        rows: NativeRows,
    ) -> Result<u64> {
        let (root, body, adaln) = self.full_view(backend)?;
        let table = self.table_bytes(task, rows)?;
        // Root verification/refiner precedes AdaLN verification. Both native companion sets
        // are eagerly read by constructors; unverified AdaLN/body descriptor handles still exist.
        let eager = plus(body.eager_payload, adaln.eager_payload)?;
        let pending_descriptors = plus(body.descriptors, adaln.descriptors)?;
        let cold_or_refiner = plus(
            plus(plus(root, eager)?, pending_descriptors)?,
            self.root
                .descriptors
                .max(self.root_working(task, backend, rows, true)?),
        )?;
        // Final/root modulation runs after dense eviction but BEFORE body verification; body
        // companions/descriptors still coexist with this work and the complete retained cache.
        let retained_body = plus(plus(root, body.eager_payload)?, body.descriptors)?;
        let after_projection = plus(
            plus(retained_body, table)?,
            self.root_working(task, backend, rows, false)?,
        )?;
        let mut peak = cold_or_refiner.max(after_projection);
        if self.pruned {
            return Ok(peak);
        }
        let mut work = 0;
        for p in &task.projections {
            if matches!(locate(&p.base)?, Location::Adaln(_)) {
                work = work.max(
                    p.working_bytes(backend, rows).saturating_sub(
                        p.output
                            .saturating_mul(p.activation_width)
                            .saturating_mul(rows.schedule),
                    ),
                );
            }
        }
        let weights = plus(retained_body, adaln.payload)?;
        // AdaLN descriptors clear on source verification BEFORE projection work. The body
        // descriptors do not clear until the later body verification, so they are NOT maxed
        // away against AdaLN work. Cast output overlaps the active projection only once.
        let verifying = plus(weights, adaln.descriptors)?;
        let forwarding = plus(plus(weights, table)?, plus(work, adaln.forward_cast)?)?;
        peak = peak.max(verifying).max(forwarding);
        Ok(peak)
    }

    pub fn denoise_peak_bytes_full(
        &self,
        task: &NativeTaskAllocation,
        backend: NativeBackend,
        rows: NativeRows,
    ) -> Result<u64> {
        let (root, body, _) = self.full_view(backend)?;
        let base = plus(plus(root, body.payload)?, self.table_bytes(task, rows)?)?;
        let mut body_work = 0;
        for p in &task.projections {
            if matches!(locate(&p.base)?, Location::Body(_)) {
                body_work = body_work.max(p.working_bytes(backend, rows));
            }
        }
        // Verification keeps descriptor handles but does not evaluate derived dtype casts.
        // Forward starts after those handles clear, and may first evaluate one such cast.
        let working = body
            .descriptors
            .max(plus(body_work, body.forward_cast)?)
            .max(self.root_working(task, backend, rows, false)?);
        plus(base, working)
    }

    pub fn precompute_peak_bytes(
        &self,
        task: &NativeTaskAllocation,
        backend: NativeBackend,
        rows: NativeRows,
        window: usize,
    ) -> Result<u64> {
        let plan = self.plan(backend, window)?;
        let root = self.root.resident()?;
        let table = self.table_bytes(task, rows)?;
        // Root constructor descriptors are transient at load; refiner runs before AdaLN windows.
        // Root/head operations after precompute retain the full modulation table but no window.
        let cold_or_refiner = plus(
            root,
            self.root
                .descriptors
                .max(self.root_working(task, backend, rows, true)?),
        )?;
        let mut peak = cold_or_refiner.max(plus(
            plus(root, table)?,
            self.root_working(task, backend, rows, false)?,
        )?);
        if self.pruned {
            return Ok(peak);
        }
        for range in plan.windows() {
            let partition = self.window_partition(range.clone(), true)?;
            let mut work = 0;
            for p in &task.projections {
                if let Location::Adaln(index) = locate(&p.base)? {
                    if range.contains(&index) {
                        // This output becomes a retained modulation table, not a second copy.
                        work = work.max(
                            p.working_bytes(backend, rows).saturating_sub(
                                p.output
                                    .saturating_mul(p.activation_width)
                                    .saturating_mul(rows.schedule),
                            ),
                        );
                    }
                }
            }
            // H3's one common hidden/modality width gives equal table bytes per block. The
            // cache grows until THIS aligned range's end; do not give the first window the final
            // window's cache, or combine one window's large weights with another's large work.
            let produced = times(table / self.layers.len() as u64, range.end as u64)?;
            let live = plus(
                plus(plus(root, partition.resident()?)?, produced)?,
                partition
                    .descriptors
                    .max(plus(work, partition.forward_cast)?),
            )?;
            peak = peak.max(live);
        }
        Ok(peak)
    }

    pub fn denoise_peak_bytes(
        &self,
        task: &NativeTaskAllocation,
        backend: NativeBackend,
        rows: NativeRows,
        window: usize,
    ) -> Result<u64> {
        let plan = self.plan(backend, window)?;
        let base = plus(self.root.resident()?, self.table_bytes(task, rows)?)?;
        let mut peak = plus(base, self.root_working(task, backend, rows, false)?)?;
        for range in plan.windows() {
            let partition = self.window_partition(range.clone(), false)?;
            let mut work = 0;
            for p in &task.projections {
                if let Location::Body(index) = locate(&p.base)? {
                    if range.contains(&index) {
                        work = work.max(p.working_bytes(backend, rows));
                    }
                }
            }
            peak = peak.max(plus(
                plus(base, partition.resident()?)?,
                partition
                    .descriptors
                    .max(plus(work, partition.forward_cast)?),
            )?);
        }
        Ok(peak)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const BACKEND: NativeBackend = NativeBackend::Mlx { tile_columns: 16 };
    fn rows() -> NativeRows {
        NativeRows {
            packed: 2,
            conditioning: 1,
            schedule: 2,
            visual_elements: 0,
            global: 0,
        }
    }
    fn task(projections: Vec<NativeProjection>, table: u64, pruned: bool) -> NativeTaskAllocation {
        NativeTaskAllocation {
            resident_bytes: u64::MAX / 2,
            constructor_bytes: 0,
            patch_size: [1, 2, 2],
            reference_task: false,
            evicted_bytes: if pruned { 0 } else { 10 },
            table_bytes_per_timestep: table,
            projections,
            window: None,
        }
    }
    fn projection(layer: usize, adaln: bool, group: u64, w4a8: bool) -> NativeProjection {
        NativeProjection {
            base: format!(
                "transformer_blocks.{layer}.{}",
                if adaln {
                    "adaln_proj.linear"
                } else {
                    "attn.to_q"
                }
            ),
            input: 16,
            output: 16,
            rotation_group: group,
            activation_width: 2,
            w4a8,
            bias_width: None,
            bias_dtype: None,
            rows: if adaln {
                RowDomain::Schedule
            } else {
                RowDomain::Packed
            },
            descriptor_bytes: 0,
        }
    }
    fn payloads(bodies: &[u64], adaln: &[u64]) -> Vec<(String, u64)> {
        let mut v = vec![("token_refiner.proj.weight".into(), 10)];
        for (i, bytes) in bodies.iter().enumerate() {
            v.push((format!("transformer_blocks.{i}.attn.to_q.weight"), *bytes));
            v.push((
                format!("transformer_blocks.{i}.adaln_proj.linear.weight"),
                adaln[i],
            ));
        }
        v
    }
    #[test]
    fn dense_window_uses_aligned_actual_payload_not_full_component() {
        let data = payloads(&[100, 200, 300, 400, 500], &[10, 20, 30, 40, 50]);
        let p = WindowAllocation::from_resident_tensors(&data, &[], 5, false).unwrap();
        let task = task(vec![], 50, false);
        assert_eq!(p.n_blocks(), 5);
        // [0,2)=300, [2,4)=700, [4,5)=500: never sliding [3,5)=900.
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 2).unwrap(),
            15 + 100 + 700
        );
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 1).unwrap(),
            15 + 100 + 500
        );
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 99).unwrap(),
            15 + 100 + 1500
        );
        assert!(p.denoise_peak_bytes(&task, BACKEND, rows(), 1).unwrap() < task.resident_bytes);
    }
    #[test]
    fn dense_precompute_cache_grows_by_actual_window_end_and_evicts_adaln() {
        let data = payloads(&[100, 100, 100], &[1000, 10, 10]);
        let p = WindowAllocation::from_resident_tensors(&data, &[], 0, false).unwrap();
        let task = task(vec![], 300, false);
        // First window: root10 + adaln1000 + 1/3 of 600byte cache =1210.
        assert_eq!(
            p.precompute_peak_bytes(&task, BACKEND, rows(), 1).unwrap(),
            1210
        );
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 1).unwrap(),
            710
        );
    }
    #[test]
    fn root_and_fresh_window_rotations_coexist_but_group_duplicates_share() {
        let data = payloads(&[100, 100], &[10, 10]);
        let mut projections = vec![
            projection(0, false, 16, false),
            projection(1, false, 16, false),
        ];
        let mut root = projection(0, false, 16, false);
        root.base = "token_refiner.proj".into();
        root.rows = RowDomain::Conditioning;
        projections.push(root);
        let p = WindowAllocation::from_resident_tensors(&data, &projections, 0, false).unwrap();
        let task = task(projections, 0, false);
        let work = task.projections[0].working_bytes(BACKEND, rows());
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 2).unwrap(),
            10 + 1024 + 200 + 1024 + work
        );
    }
    #[test]
    fn w4a8_fht_has_no_resident_group_square_and_work_is_per_operation() {
        let data = payloads(&[100_000, 200], &[10, 10]);
        let mut a = projection(0, false, 16, true);
        let mut b = projection(1, false, 16, true);
        b.input = 4096;
        // Heavy weights and heavy scratch are deliberately in DIFFERENT windows.
        a.descriptor_bytes = 0;
        let projections = vec![a, b];
        let p = WindowAllocation::from_resident_tensors(&data, &projections, 0, false).unwrap();
        let task = task(projections, 0, false);
        let expected = (110 + task.projections[0].working_bytes(BACKEND, rows()))
            .max(210 + task.projections[1].working_bytes(BACKEND, rows()));
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 1).unwrap(),
            expected
        );
        let mut changed = task.projections.clone();
        changed[1].rotation_group = 8192;
        let no_dense_matrix =
            WindowAllocation::from_resident_tensors(&data, &changed, 0, false).unwrap();
        assert_eq!(p, no_dense_matrix);
    }
    #[test]
    fn pruned_operands_compute_and_retained_cache_are_root_no_eviction() {
        let data = payloads(&[100, 200], &[30, 40]);
        let p = WindowAllocation::from_resident_tensors(&data, &[], 1000, true).unwrap();
        let task = task(vec![], 0, true);
        assert_eq!(
            p.precompute_peak_bytes(&task, BACKEND, rows(), 1).unwrap(),
            1080
        );
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 1).unwrap(),
            1280
        );
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 2).unwrap(),
            1380
        );
    }
    #[test]
    fn descriptors_are_transient_max_not_permanent_or_all_layer_sum() {
        let data = payloads(&[100, 100], &[10, 10]);
        let mut a = projection(0, false, 16, true);
        a.descriptor_bytes = 9000;
        let mut b = projection(1, false, 16, true);
        b.descriptor_bytes = 8000;
        let projections = vec![a, b];
        let p = WindowAllocation::from_resident_tensors(&data, &projections, 0, false).unwrap();
        let task = task(projections, 0, false);
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 1).unwrap(),
            10 + 100 + 9000
        );
        assert_eq!(
            p.denoise_peak_bytes(&task, BACKEND, rows(), 2).unwrap(),
            10 + 200 + 17000
        );
    }
    #[test]
    fn ordinary_resident_floor_has_actual_cache_groups_but_no_descriptors_or_work() {
        let data = payloads(&[100, 200], &[1000, 10]);
        let mut a = projection(0, false, 16, false);
        a.descriptor_bytes = 9000;
        let mut b = projection(1, false, 16, false);
        b.input = 4096;
        let projections = vec![a, b];
        let p = WindowAllocation::from_resident_tensors(&data, &projections, 0, false).unwrap();
        let task = task(projections, 200, false);
        // Body1 200+matrix1024+fullcache400 exceeds AdaLN0 1000+prefixcache200.
        // Root10 is held throughout. The heavy body's scratch/descriptor is NOT a weight floor.
        let expected = 10 + 200 + 1024 + 400;
        assert_eq!(
            p.resident_peak_bytes(&task, BACKEND, rows(), 1).unwrap(),
            expected
        );
        assert!(p.denoise_peak_bytes(&task, BACKEND, rows(), 1).unwrap() > expected);
        let mut more_schedule = rows();
        more_schedule.schedule = 4;
        assert!(
            p.resident_peak_bytes(&task, BACKEND, more_schedule, 1)
                .unwrap()
                > expected
        );
    }
    #[test]
    fn staged_full_dense_precompute_does_not_hold_lazy_body_stack() {
        let data = payloads(&[100, 200], &[1000, 20]);
        let profile = WindowAllocation::from_resident_tensors(&data, &[], 0, false).unwrap();
        let task = task(vec![], 200, false);
        assert_eq!(
            profile
                .precompute_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            10 + 1020 + 400
        );
        assert_eq!(
            profile
                .denoise_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            10 + 300 + 400
        );
        assert_eq!(
            profile
                .resident_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            10 + 1020 + 400
        );
        assert!(
            profile
                .resident_peak_bytes_full(&task, BACKEND, rows())
                .unwrap()
                < 10 + 1020 + 300 + 400
        );
    }
    #[test]
    fn staged_full_eager_companions_pending_descriptors_and_single_view_groups() {
        let mut data = payloads(&[100, 200], &[1000, 20]);
        data.extend([
            ("transformer_blocks.0.attn.to_q.weight_scale".into(), 7),
            ("transformer_blocks.1.attn.to_q.weight_scale".into(), 11),
            (
                "transformer_blocks.0.adaln_proj.linear.weight_scale".into(),
                5,
            ),
        ]);
        let mut a = projection(0, false, 16, false);
        a.descriptor_bytes = 3;
        let mut b = projection(1, false, 16, false);
        b.descriptor_bytes = 4;
        let mut adaln = projection(0, true, 16, false);
        adaln.descriptor_bytes = 6;
        let mut root_projection = projection(0, false, 16, false);
        root_projection.base = "token_refiner.proj".into();
        root_projection.rows = RowDomain::Conditioning;
        root_projection.descriptor_bytes = 2;
        let projections = vec![a, b, adaln, root_projection];
        let profile =
            WindowAllocation::from_resident_tensors(&data, &projections, 0, false).unwrap();
        let task = task(projections, 200, false);
        let root = 10 + 4 * 16 * 16; // ONE whole-view group shared by root, both bodies, AdaLN.
        let table = 400;
        let adaln_work = task.projections[2].working_bytes(BACKEND, rows()) - 16 * 2 * 2;
        let expected_precompute = root + (7 + 11) + (3 + 4) + (1000 + 20 + 5) + table + adaln_work;
        assert_eq!(
            profile
                .precompute_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            expected_precompute
        );
        assert_eq!(
            profile
                .resident_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            root + (7 + 11) + (1000 + 20 + 5) + table
        );
        let body_work = task.projections[0].working_bytes(BACKEND, rows());
        assert_eq!(
            profile
                .denoise_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            root + (100 + 200 + 7 + 11) + table + body_work.max(3 + 4)
        );
    }
    #[test]
    fn staged_full_w4a8_companions_eager_but_codes_and_dense_bodies_lazy() {
        let mut data = payloads(&[100, 200], &[10, 20]);
        data.extend([
            ("transformer_blocks.0.attn.to_q.weight_s_rel".into(), 5),
            ("transformer_blocks.0.attn.to_q.weight_s_channel".into(), 7),
            ("transformer_blocks.0.attn.to_q.weight_codebook".into(), 11),
        ]);
        let mut projection = projection(0, false, 16, true);
        projection.descriptor_bytes = 3;
        let projections = vec![projection];
        let profile =
            WindowAllocation::from_resident_tensors(&data, &projections, 0, false).unwrap();
        let task = task(projections, 0, false);
        assert_eq!(
            profile
                .precompute_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            10 + 30 + (5 + 7 + 11) + 3
        );
        // Eager companions are a subset of body payload, never another copy at denoise.
        assert_eq!(
            profile
                .resident_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            10 + 300 + 5 + 7 + 11
        );
    }
    #[test]
    fn actual_source_casts_overlap_forward_once_but_never_resident_endpoints() {
        let mut data = payloads(&[100, 200], &[10, 20]);
        data.extend([
            ("transformer_blocks.0.ff.net.2.weight".into(), 32),
            ("transformer_blocks.1.ff.net.2.weight".into(), 32),
        ]);
        let projections = vec![
            projection(0, false, 16, true),
            projection(1, false, 16, true),
            projection(0, true, 16, true),
        ];
        let base = WindowAllocation::from_resident_tensors(&data, &projections, 0, false).unwrap();
        let casted = base
            .clone()
            .with_source_casts(&[
                ("token_refiner.proj.weight".into(), 13),
                ("transformer_blocks.0.ff.net.2.weight".into(), 17),
                ("transformer_blocks.1.ff.net.2.weight".into(), 11),
                ("transformer_blocks.1.adaln_proj.linear.weight".into(), 23),
            ])
            .unwrap();
        let task = task(projections, 64, false);
        assert_eq!(
            casted
                .resident_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            base.resident_peak_bytes_full(&task, BACKEND, rows())
                .unwrap()
        );
        assert_eq!(
            casted
                .resident_peak_bytes(&task, BACKEND, rows(), 2)
                .unwrap(),
            base.resident_peak_bytes(&task, BACKEND, rows(), 2).unwrap()
        );
        assert_eq!(
            casted
                .denoise_peak_bytes_full(&task, BACKEND, rows())
                .unwrap()
                - base
                    .denoise_peak_bytes_full(&task, BACKEND, rows())
                    .unwrap(),
            17
        );
        assert_eq!(
            casted
                .precompute_peak_bytes_full(&task, BACKEND, rows())
                .unwrap()
                - base
                    .precompute_peak_bytes_full(&task, BACKEND, rows())
                    .unwrap(),
            23
        );
        assert_eq!(
            casted
                .denoise_peak_bytes(&task, BACKEND, rows(), 2)
                .unwrap()
                - base.denoise_peak_bytes(&task, BACKEND, rows(), 2).unwrap(),
            17
        );
        assert!(base
            .with_source_casts(&[("transformer_blocks.9.ff.weight".into(), 1)])
            .is_err());
    }
    #[test]
    fn pruned_full_operands_stay_root_with_zero_eviction() {
        let data = payloads(&[100, 200], &[30, 40]);
        let profile = WindowAllocation::from_resident_tensors(&data, &[], 1000, true).unwrap();
        let task = task(vec![], 0, true);
        assert_eq!(
            profile
                .precompute_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            1080
        );
        assert_eq!(
            profile
                .denoise_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            1380
        );
        assert_eq!(
            profile
                .resident_peak_bytes_full(&task, BACKEND, rows())
                .unwrap(),
            1380
        );
    }
    #[test]
    fn invalid_windows_gaps_descriptors_and_overflow_are_refused() {
        let data = payloads(&[100], &[10]);
        let p = WindowAllocation::from_resident_tensors(&data, &[], 0, false).unwrap();
        let task = task(vec![], 0, false);
        assert!(p.denoise_peak_bytes(&task, BACKEND, rows(), 0).is_err());
        assert!(p
            .denoise_peak_bytes(
                &task,
                NativeBackend::Candle {
                    tile_rows: 16,
                    workspace_bytes: 0
                },
                rows(),
                1
            )
            .is_err());
        assert!(WindowAllocation::from_resident_tensors(
            &[("transformer_blocks.2.attn.weight".into(), 1)],
            &[],
            0,
            false
        )
        .is_err());
        assert!(WindowAllocation::from_resident_tensors(
            &[("transformer_blocks.X.attn.weight".into(), 1)],
            &[],
            0,
            false
        )
        .is_err());
        assert!(WindowAllocation::from_resident_tensors(
            &[("root.comfy_quant".into(), 1)],
            &[],
            0,
            false
        )
        .is_err());
        assert!(WindowAllocation::from_resident_tensors(&data, &[], u64::MAX, false).is_err());
    }
}
