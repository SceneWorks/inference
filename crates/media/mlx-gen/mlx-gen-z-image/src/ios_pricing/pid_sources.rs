//! SCRATCH candidate: Z's actual PiD source consumer. NOT applied, compiled, or qualified.
//!
//! Fixed PiD/Gemma source recipes below produce ordered source-capacity windows. They never run
//! a tensor kernel, read a tensor/scalar payload, change an engine/cache policy, or use budget.rs's
//! empirical 6/3/8 coefficients. A descriptor/view is not a native Data allocation certificate.
//! Native workspaces, queue/Data ownership and host allocator domains stay EXPLICIT obligations;
//! these logical source windows alone cannot certify a physical generation envelope.

use super::pid_work_basis::PidWorkBasis;
use super::{price_source_quanta, Facts, Model, Priced, Route};
use mlx_gen::adapters::facts::{dtype_width, native_header_dtype};
use mlx_gen::adapters::loader::source_facts::{
    visit_configured_adapter_source_cases, AdapterSourceCase, AdapterSourceFailure,
    AdapterSourceOutcome, AdapterSourcePrefix,
};
use mlx_gen::gen_core::weightsmeta as wm;
use mlx_gen::gen_core::{LoadSpec, WeightsSource};
use mlx_gen::ios_pricing::{IosPhase, IosPricing, IosWindowPrice};
use mlx_gen::ios_quantum::Backend;
use mlx_gen::sampler::flow_sampling::ResolvedFlowControls;
use mlx_gen::{Error, Result};
use mlx_gen_pid::{ConvPadding, Gemma2Config, PidConfig, SampleType, SamplerConfig};
use mlx_rs::Dtype;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[path = "pid_sources/native_allocations.rs"]
mod native_allocations;

/// Source-enumerated CPU-stream requests charged to the linked Metal active counter.
/// This deliberately reuses the native request records and cache-length law, NOT the Metal
/// operation selectors. No Data retirement or physical task-footprint inference is made here.
#[derive(Clone, Debug)]
pub(super) struct ActiveCounterRequests {
    requests: Vec<native_allocations::Request>,
    upper: u64,
    page: u64,
}
impl ActiveCounterRequests {
    pub(super) fn new(page: u64) -> Result<Self> {
        if !page.is_power_of_two() {
            return Err(Error::Msg(
                "CPU counter certificate: invalid allocator page".into(),
            ));
        }
        Ok(Self {
            requests: Vec::new(),
            upper: 0,
            page,
        })
    }
    pub(super) fn charge(&mut self, rule: &'static str, site: &str, bytes: u64) -> Result<()> {
        use native_allocations::CacheLengthBound;
        let charge = match native_allocations::cached_length_bound(bytes, Some(self.page), 64)? {
            CacheLengthBound::Empty => 0,
            CacheLengthBound::Cached {
                upper_exclusive, ..
            } => upper_exclusive,
            CacheLengthBound::Unbound(reason) => return Err(Error::Unsupported(reason.into())),
        };
        self.upper = self
            .upper
            .checked_add(charge)
            .ok_or_else(|| Error::Msg("CPU active-counter cumulative upper overflow".into()))?;
        self.requests.push(native_allocations::Request {
            domain: "native096/CPU-stream/linked-Metal/active-counter",
            workspace: None,
            source_value: None,
            rule,
            predicate: site.into(),
            bytes: Some(bytes),
            existing_output: None,
            data_size_product: None,
        });
        // Every CPU-counter record is numeric; an unknown size must refuse, never become zero.
        debug_assert!(self.requests.iter().all(|request| request.bytes.is_some()));
        Ok(())
    }
    pub(super) fn upper(&self) -> u64 {
        self.upper
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Slot(usize);
#[derive(Clone, Debug)]
struct Tensor {
    shape: Vec<i32>,
    dtype: Dtype,
    /// Source ancestry, not context pointers or a last-native-Data-owner assertion.
    roots: BTreeSet<Slot>,
    value: usize, // source descriptor occurrence, NOT observed native pointer identity
}
impl Tensor {
    fn elements(&self) -> Result<u64> {
        elements(&self.shape)
    }
}
#[derive(Clone, Debug)]
struct Capacity {
    name: String,
    shape: Vec<i32>,
    dtype: Dtype,
    bytes: u64,
    /// True only for an actual immediate source factory (host table/scalar), not a lazy op output.
    immediate: bool,
    possible_view_copy: bool,
    /// DAG edges avoid cloning a transitive ancestry set into every descriptor.
    parents: BTreeSet<Slot>,
    graph_attached: bool,
    value: usize,
}
#[derive(Clone, Debug)]
struct Operation {
    site: String,
    operands: Vec<(Vec<i32>, Dtype)>,
    output: Option<(Vec<i32>, Dtype)>,
    operand_values: Vec<usize>,
    output_value: Option<usize>,
    completed: bool,
}
/// A SOURCE rule, not a runtime observation or a physical/allocator certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
enum GatherRowLayout {
    /// Native096 Gather::eval_gpu calls set_data(out.nbytes()), setting row_contiguous=true;
    /// take(axis=0) does not transpose, and Squeeze::eval forwards the SAME input flags/Data.
    Native096GatherSetDataThenSqueeze,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct GatherRowSource {
    table_shape: Vec<i32>,
    indices_shape: Vec<i32>,
    gather_shape: Vec<i32>,
    squeezed_shape: Vec<i32>,
    gather_slot: Slot,
    squeeze_axis: usize,
    layout: GatherRowLayout,
}
#[derive(Clone, Debug)]
struct AffineRowsSource {
    group_size: i32,
    bits: i32,
    gathers: [GatherRowSource; 3],
    /// The EXISTING dequant output slot, not a second workspace allocation/charge.
    output_slot: Slot,
    output_shape: Vec<i32>,
    output_dtype: Dtype,
    /// The source rule is conditional: normal Metal forward uses fast::Quantize; CPU and
    /// Custom::{vjp,jvp,vmap} use the composed fallback. No flag/physical cost is guessed.
    dispatch_condition: &'static str,
}
/// Actual native factory attributes; unknown backend/stride/transform selectors are NOT defaults.
#[derive(Clone, Copy, Debug)]
enum NativeCall {
    Unbound,
    Matmul,
    Addmm,
    Qmm {
        group_size: i32,
        bits: i32,
        transpose: bool,
        bias_output_slot: Option<Slot>,
    },
    /// PiD Conv2d source fixes stride/dilation1, !flip; pre-edge-pad chooses padding0 vs1.
    Conv {
        padding: i32,
        groups: i32,
        has_bias: bool,
        plain_output: Slot,
    },
    GroupNorm {
        groups: i32,
        normalized_output: Slot,
    },
    SdpaNoMaskNoSinks,
    NormalNoLocScale,
}
#[derive(Clone, Debug)]
struct NativeWorkspace {
    site: String,
    inputs: Vec<(Vec<i32>, Dtype)>,
    input_values: Vec<usize>,
    /// A native factory/backend source obligation, NEVER an invented zero-byte workspace.
    family: &'static str,
    affine_rows: Option<AffineRowsSource>,
    output_slot: Option<Slot>,
    call: NativeCall,
}
#[derive(Clone, Debug)]
enum HostDemand {
    Bytes {
        site: String,
        bytes: u64,
    },
    /// Token replay has no user caption text. Keep its variable, do not silently bind it to zero.
    Symbolic {
        site: String,
        symbol: &'static str,
        coefficient: u64,
    },
}
#[derive(Clone, Debug, Default)]
struct Prefix {
    values: Vec<native_allocations::ValueFact>,
    boundaries: Vec<native_allocations::BoundaryFact>,
    capacities: Vec<Capacity>,
    operations: Vec<Operation>,
    workspaces: Vec<NativeWorkspace>,
    host: Vec<HostDemand>,
    windows: Vec<SourceWindow>,
    source_files: Vec<PathBuf>,
}
#[derive(Clone, Debug)]
struct SourceWindow {
    phase: IosPhase,
    site: String,
    resident_roots: BTreeSet<Slot>,
    created_roots: BTreeSet<Slot>,
    host: Vec<HostDemand>,
    workspace_start: usize,
    workspace_end: usize,
    operation_start: usize,
    operation_end: usize,
    requests: native_allocations::WindowRequests,
    owners: native_allocations::OwnerWindow,
    value_start: usize,
    value_end: usize,
    serial_completed: u32,
    step: Option<usize>,
    tile: Option<mlx_gen_pid::tiling::SpatialTile>,
}
#[derive(Clone, Debug, Default)]
struct Work {
    roots: BTreeSet<Slot>,
    operation_start: usize,
    workspace_start: usize,
    host_start: usize,
    value_start: usize,
}
#[derive(Clone, Default)]
struct Program {
    prefix: Prefix,
    work: Work,
    host_live: BTreeMap<String, HostDemand>,
    raw_io_frontiers: Vec<Arc<Prefix>>,
}

fn elements(shape: &[i32]) -> Result<u64> {
    shape.iter().try_fold(1u64, |n, &d| {
        if d < 0 {
            return Err(Error::Msg("negative PiD source dimension".into()));
        }
        n.checked_mul(d as u64)
            .ok_or_else(|| Error::Msg("PiD source element overflow".into()))
    })
}
fn mul(a: i32, b: i32) -> Result<i32> {
    a.checked_mul(b)
        .ok_or_else(|| Error::Msg("PiD source shape overflow".into()))
}
fn broadcast(a: &[i32], b: &[i32]) -> Result<Vec<i32>> {
    let mut shape = Vec::new();
    for i in 0..a.len().max(b.len()) {
        let x = a.len().checked_sub(i + 1).map_or(1, |j| a[j]);
        let y = b.len().checked_sub(i + 1).map_or(1, |j| b[j]);
        if x != y && x != 1 && y != 1 {
            return Err(Error::Msg(
                "PiD native broadcast dimensions disagree".into(),
            ));
        }
        shape.push(if x == 1 { y } else { x });
    }
    shape.reverse();
    Ok(shape)
}
fn union<'a>(tensors: impl IntoIterator<Item = &'a Tensor>) -> BTreeSet<Slot> {
    tensors
        .into_iter()
        .flat_map(|t| t.roots.iter().copied())
        .collect()
}

impl Program {
    fn begin(&mut self) {
        self.work = Work {
            roots: BTreeSet::new(),
            operation_start: self.prefix.operations.len(),
            workspace_start: self.prefix.workspaces.len(),
            host_start: self.prefix.host.len(),
            value_start: self.prefix.values.len(),
        };
    }
    fn attempt(&mut self, site: &str, inputs: &[&Tensor]) -> usize {
        let index = self.prefix.operations.len();
        self.prefix.operations.push(Operation {
            site: site.into(),
            operands: inputs.iter().map(|t| (t.shape.clone(), t.dtype)).collect(),
            output: None,
            operand_values: inputs.iter().map(|t| t.value).collect(),
            output_value: None,
            completed: false,
        });
        index
    }
    fn complete(&mut self, index: usize, output: &Tensor) {
        self.prefix.operations[index].completed = true;
        self.prefix.operations[index].output = Some((output.shape.clone(), output.dtype));
        self.prefix.operations[index].output_value = Some(output.value);
    }
    fn storage(
        &mut self,
        name: &str,
        shape: Vec<i32>,
        dtype: Dtype,
        parents: BTreeSet<Slot>,
        immediate: bool,
        copy: bool,
    ) -> Result<Tensor> {
        let bytes = elements(&shape)?
            .checked_mul(dtype_width(dtype))
            .ok_or_else(|| Error::Msg("PiD source bytes overflow".into()))?;
        let slot = Slot(self.prefix.capacities.len());
        let value = self.owner_value(
            shape.clone(),
            dtype,
            native_allocations::ValueRule::Opaque(slot),
        );
        self.prefix.capacities.push(Capacity {
            name: name.into(),
            shape: shape.clone(),
            dtype,
            bytes,
            immediate,
            possible_view_copy: copy,
            parents,
            graph_attached: true,
            value,
        });
        self.work.roots.insert(slot);
        Ok(Tensor {
            shape,
            dtype,
            roots: BTreeSet::from([slot]),
            value,
        })
    }
    fn owner_value(
        &mut self,
        shape: Vec<i32>,
        dtype: Dtype,
        rule: native_allocations::ValueRule,
    ) -> usize {
        let id = self.prefix.values.len();
        self.prefix
            .values
            .push(native_allocations::ValueFact { shape, dtype, rule });
        id
    }
    fn owner_rule(&mut self, value: usize, rule: native_allocations::ValueRule) -> Result<()> {
        self.prefix
            .values
            .get_mut(value)
            .ok_or_else(|| Error::Msg("PiD owner source value missing".into()))?
            .rule = rule;
        Ok(())
    }
    fn host_input(&mut self, site: &str, shape: Vec<i32>, dtype: Dtype) -> Result<Tensor> {
        let out = self.immediate(site, shape, dtype)?;
        self.owner_rule(out.value, native_allocations::ValueRule::HostInput)?;
        Ok(out)
    }
    fn external(&mut self, shape: Vec<i32>, dtype: Dtype) -> Tensor {
        let value = self.owner_value(
            shape.clone(),
            dtype,
            native_allocations::ValueRule::External,
        );
        Tensor {
            shape,
            dtype,
            roots: BTreeSet::new(),
            value,
        }
    }
    fn owner_boundary(&mut self, kind: native_allocations::BoundaryKind, site: &str) {
        self.prefix
            .boundaries
            .push(native_allocations::BoundaryFact {
                kind,
                site: site.into(),
                after_window: self.prefix.windows.len(),
            });
    }
    fn immediate(&mut self, site: &str, shape: Vec<i32>, dtype: Dtype) -> Result<Tensor> {
        self.storage(site, shape, dtype, BTreeSet::new(), true, false)
    }
    fn host(&mut self, site: &str, bytes: u64) {
        self.prefix.host.push(HostDemand::Bytes {
            site: site.into(),
            bytes,
        });
    }
    fn hold_host(&mut self, site: &str, bytes: u64) {
        let demand = HostDemand::Bytes {
            site: site.into(),
            bytes,
        };
        self.host_live.insert(site.into(), demand); // same source owner appears once in each frontier snapshot
    }
    fn end_host_source_scope(&mut self, site: &str) {
        if self.host_live.remove(site).is_some() {
            // Rust source scope ends; libc/allocator/process release is explicitly NOT inferred.
            self.prefix.host.push(HostDemand::Symbolic {
                site: format!("{site}/retired-host-allocator"),
                symbol: "host_allocator_retained_capacity",
                coefficient: 1,
            });
        }
    }
    fn workspace(&mut self, site: &str, family: &'static str, inputs: &[&Tensor]) -> usize {
        let index = self.prefix.workspaces.len();
        self.prefix.workspaces.push(NativeWorkspace {
            site: site.into(),
            family,
            inputs: inputs.iter().map(|t| (t.shape.clone(), t.dtype)).collect(),
            input_values: inputs.iter().map(|t| t.value).collect(),
            affine_rows: None,
            output_slot: None,
            call: NativeCall::Unbound,
        });
        index
    }
    fn native_output(&mut self, index: usize, output: &Tensor, call: NativeCall) -> Result<()> {
        if output.roots.len() != 1 {
            return Err(Error::Msg(
                "PiD native allocation output needs SAME existing source slot".into(),
            ));
        }
        let record = self
            .prefix
            .workspaces
            .get_mut(index)
            .ok_or_else(|| Error::Msg("PiD native workspace index missing".into()))?;
        record.output_slot = output.roots.iter().next().copied();
        record.call = call;
        let family = record.family;
        let values = record.input_values.clone();
        // Bind only explicit native set_data output factories. Composed/compiled donation is
        // represented separately, never inferred from a logical storage slot or metadata Array.
        if matches!(
            family,
            "native-matmul-backend-and-queue"
                | "native-addmm-fused-epilogue"
                | "native-quantized-matmul-backend-and-packed-layout"
        ) {
            self.owner_rule(
                output.value,
                native_allocations::ValueRule::FreshMetal(*output.roots.iter().next().unwrap()),
            )?;
        } else if family == "native-fast-rms-kernel-or-fallback" {
            self.owner_rule(
                output.value,
                native_allocations::ValueRule::NormLike(values[0], true),
            )?;
        } else if family == "native-explicit-softmax-not-flash" {
            self.owner_rule(
                output.value,
                native_allocations::ValueRule::NormLike(values[0], false),
            )?;
        } else if matches!(
            family,
            "source-eager-or-compiled-gated-epilogue"
                | "source-eager-or-compiled-modulate-epilogue"
                | "source-exact-epilogue-or-eager-gelu"
        ) {
            self.owner_rule(
                output.value,
                native_allocations::ValueRule::UnboundCompiled(values),
            )?;
        } else if family == "native-rng-keys-splits-and-normal-workspace" {
            self.owner_rule(output.value, native_allocations::ValueRule::NormalF32)?;
        }
        Ok(())
    }
    fn embedding_take_axis_zero(
        &mut self,
        site: &str,
        table: &Tensor,
        ids: &Tensor,
    ) -> Result<(Tensor, GatherRowSource)> {
        if table.shape.len() != 2 || ids.shape.len() != 2 || ids.shape[0] != 1 {
            return Err(Error::Msg(
                "Gemma source take requires actual table[vocab,C] and ids[1,L]".into(),
            ));
        }
        // Actual native take(table, ids, 0): Gather has indices.shape + slice_sizes[1,C].
        // There is NO ids flatten and NO axis transpose. Squeeze removes axis ids.ndim()==2.
        let op = self.attempt(&format!("{site}/native Gather axis0"), &[table, ids]);
        let mut gather_shape = ids.shape.clone();
        gather_shape.extend([1, table.shape[1]]);
        let gathered = self.storage(
            &format!("{site}/native Gather buffer"),
            gather_shape.clone(),
            table.dtype,
            union([table, ids]),
            false,
            false,
        )?;
        self.owner_rule(
            gathered.value,
            native_allocations::ValueRule::FreshMetal(*gathered.roots.iter().next().unwrap()),
        )?;
        self.complete(op, &gathered);
        let gather_slot = *gathered
            .roots
            .iter()
            .next()
            .ok_or_else(|| Error::Msg("Gemma gather has no source slot".into()))?;
        let mut squeezed_shape = ids.shape.clone();
        squeezed_shape.push(table.shape[1]);
        let rows = self.view(
            &format!("{site}/native Squeeze axis2 SAME flags and Data"),
            &gathered,
            squeezed_shape.clone(),
            false,
        )?;
        self.owner_rule(
            rows.value,
            native_allocations::ValueRule::Squeeze(gathered.value, 2),
        )?;
        Ok((
            rows,
            GatherRowSource {
                table_shape: table.shape.clone(),
                indices_shape: ids.shape.clone(),
                gather_shape,
                squeezed_shape,
                gather_slot,
                squeeze_axis: 2,
                layout: GatherRowLayout::Native096GatherSetDataThenSqueeze,
            },
        ))
    }
    fn gathered_affine_rows(
        &mut self,
        codes: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        gathers: [GatherRowSource; 3],
        bits: i32,
    ) -> Result<Tensor> {
        let inputs = [codes, scales, biases];
        if !matches!(bits, 4 | 8)
            || codes.dtype != Dtype::Uint32
            || !matches!(
                scales.dtype,
                Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32
            )
            || scales.dtype != biases.dtype
            || scales.shape != biases.shape
            || codes.shape.len() != 3
            || scales.shape.len() != 3
            || codes.shape[..2] != scales.shape[..2]
        {
            return Err(Error::Msg(
                "Gemma gathered affine source inputs do not match native validated layout".into(),
            ));
        }
        for (input, source) in inputs.iter().zip(gathers.iter()) {
            if input.shape != source.squeezed_shape
                || input.roots != BTreeSet::from([source.gather_slot])
                || source.table_shape.len() != 2
                || source.table_shape[1] != input.shape[2]
                || source.indices_shape != codes.shape[..2]
                || source.squeeze_axis != 2
                || source.gather_shape != vec![1, codes.shape[1], 1, *input.shape.last().unwrap()]
                || source.layout != GatherRowLayout::Native096GatherSetDataThenSqueeze
            {
                return Err(Error::Msg(
                    "Gemma gathered affine source lost actual take/squeeze layout provenance"
                        .into(),
                ));
            }
        }
        let hidden = mul(scales.shape[2], 64)?;
        if mul(codes.shape[2], 32)? != mul(hidden, bits)? {
            return Err(Error::Msg(
                "Gemma gathered affine packed width differs from native group/bit geometry".into(),
            ));
        }
        let op = self.attempt("Gemma selected rows affine dequantization", &inputs);
        let out = self.storage(
            "Gemma selected rows dequantized",
            vec![1, codes.shape[1], hidden],
            scales.dtype,
            union(inputs),
            false,
            false,
        )?;
        let output_slot = *out
            .roots
            .iter()
            .next()
            .ok_or_else(|| Error::Msg("Gemma dequant output missing source slot".into()))?;
        // Separate allocator events: each Gather owns its own existing source slot. fast::Quantize
        // requests the output once plus conditional copies of NON-row-contiguous inputs. The
        // proven Gather/set_data/Squeeze flags settle that predicate only on the matched normal
        // Metal path; CPU/Custom fallback, driver/JIT, queue/allocator/physical terms remain open.
        self.prefix.workspaces.push(NativeWorkspace {site:"Gemma row dequantization".into(),
            family:"native-affine-dequantize-selected-rows",inputs:inputs.iter().map(|t|(t.shape.clone(),t.dtype)).collect(),input_values:inputs.iter().map(|t|t.value).collect(),
            affine_rows:Some(AffineRowsSource {group_size:64,bits,gathers,output_slot,output_shape:out.shape.clone(),output_dtype:out.dtype,
                dispatch_condition:"CONDITIONAL Metal normal forward fast::Quantize; CPU or Custom transformation composed fallback; NOT physical"}),output_slot:Some(output_slot),call:NativeCall::Unbound});
        self.owner_rule(
            out.value,
            native_allocations::ValueRule::FreshMetal(output_slot),
        )?;
        self.complete(op, &out);
        Ok(out)
    }
    fn view(
        &mut self,
        site: &str,
        x: &Tensor,
        shape: Vec<i32>,
        possible_copy: bool,
    ) -> Result<Tensor> {
        let op = self.attempt(site, &[x]);
        if elements(&shape)? != x.elements()? {
            return Err(Error::Msg(format!(
                "PiD native reshape cannot preserve elements at {site}"
            )));
        }
        let result = if possible_copy && shape != x.shape {
            self.storage(site, shape, x.dtype, x.roots.clone(), false, true)?
        } else {
            let value = self.owner_value(
                shape.clone(),
                x.dtype,
                native_allocations::ValueRule::Reshape(x.value),
            );
            Tensor {
                shape,
                dtype: x.dtype,
                roots: x.roots.clone(),
                value,
            }
        };
        self.owner_rule(
            result.value,
            native_allocations::ValueRule::Reshape(x.value),
        )?;
        self.complete(op, &result);
        Ok(result)
    }
    fn transpose(&mut self, site: &str, x: &Tensor, axes: &[usize]) -> Result<Tensor> {
        let op = self.attempt(site, &[x]);
        let unique = axes.iter().copied().collect::<BTreeSet<_>>();
        if axes.len() != x.shape.len()
            || unique.len() != axes.len()
            || axes.iter().any(|&a| a >= x.shape.len())
        {
            return Err(Error::Msg(format!("PiD native transpose axes at {site}")));
        }
        let shape = axes.iter().map(|&a| x.shape[a]).collect::<Vec<_>>();
        let value = self.owner_value(
            shape.clone(),
            x.dtype,
            native_allocations::ValueRule::Transpose(x.value, axes.to_vec()),
        );
        let result = Tensor {
            shape,
            dtype: x.dtype,
            roots: x.roots.clone(),
            value,
        };
        self.complete(op, &result);
        Ok(result)
    }
    fn cast(&mut self, site: &str, x: &Tensor, dtype: Dtype) -> Result<Tensor> {
        let op = self.attempt(site, &[x]);
        let result = if x.dtype == dtype {
            x.clone()
        } else {
            self.storage(site, x.shape.clone(), dtype, x.roots.clone(), false, false)?
        };
        if x.dtype != dtype {
            self.owner_rule(result.value, native_allocations::ValueRule::Cast(x.value))?;
        }
        self.complete(op, &result);
        Ok(result)
    }
    fn scalar(&mut self, site: &str, dtype: Dtype) -> Result<Tensor> {
        let raw = self.immediate(site, vec![1], Dtype::Float32)?;
        // Rust helper scalars use [1], native fallback literals may be rank0. Only
        // extent1 is common proof; this does not guess a unified native descriptor rank.
        self.owner_rule(raw.value, native_allocations::ValueRule::Scalar)?;
        self.cast(&format!("{site}/typed"), &raw, dtype)
    }
    fn unary(&mut self, site: &str, x: &Tensor) -> Result<Tensor> {
        let op = self.attempt(site, &[x]);
        let result = self.storage(
            site,
            x.shape.clone(),
            x.dtype,
            x.roots.clone(),
            false,
            false,
        )?;
        self.owner_rule(result.value, native_allocations::ValueRule::Unary(x.value))?;
        self.complete(op, &result);
        Ok(result)
    }
    fn binary(&mut self, site: &str, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let op = self.attempt(site, &[a, b]);
        let dtype = promote_dtype(a.dtype, b.dtype);
        // Native elementwise casts precede broadcast validation, including a refused prefix.
        let a = self.cast(&format!("{site}/a-cast"), a, dtype)?;
        let b = self.cast(&format!("{site}/b-cast"), b, dtype)?;
        let result = self.storage(
            site,
            broadcast(&a.shape, &b.shape)?,
            dtype,
            union([&a, &b]),
            false,
            false,
        )?;
        self.owner_rule(
            result.value,
            native_allocations::ValueRule::Binary(a.value, b.value),
        )?;
        self.complete(op, &result);
        Ok(result)
    }
    fn concat(&mut self, site: &str, inputs: &[&Tensor], axis: usize) -> Result<Tensor> {
        let op = self.attempt(site, inputs);
        let first = inputs
            .first()
            .ok_or_else(|| Error::Msg("empty PiD concatenate".into()))?;
        let dtype = inputs
            .iter()
            .fold(first.dtype, |d, t| promote_dtype(d, t.dtype));
        let mut shape = first.shape.clone();
        if axis >= shape.len() {
            return Err(Error::Msg("PiD concatenate axis".into()));
        }
        shape[axis] = 0;
        for input in inputs {
            if input.shape.len() != shape.len()
                || input
                    .shape
                    .iter()
                    .enumerate()
                    .any(|(i, &d)| i != axis && d != first.shape[i])
            {
                return Err(Error::Msg("PiD concatenate shape".into()));
            }
            self.cast(&format!("{site}/input-cast"), input, dtype)?;
            shape[axis] = shape[axis]
                .checked_add(input.shape[axis])
                .ok_or_else(|| Error::Msg("PiD concatenate overflow".into()))?;
        }
        let result = self.storage(
            site,
            shape,
            dtype,
            union(inputs.iter().copied()),
            false,
            false,
        )?;
        self.complete(op, &result);
        Ok(result)
    }
    fn matmul(&mut self, site: &str, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let op = self.attempt(site, &[a, b]);
        if a.shape.is_empty() || b.shape.is_empty() {
            return Err(Error::Msg("PiD native matmul scalar".into()));
        }
        let na = a.shape.len();
        let nb = b.shape.len();
        let k = a.shape[na - 1];
        let kb = b.shape[if nb == 1 { 0 } else { nb - 2 }];
        if k != kb {
            return Err(Error::Msg(format!(
                "PiD native matmul contraction at {site}"
            )));
        }
        let dtype = promote_dtype(a.dtype, b.dtype);
        if !matches!(
            dtype,
            Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32 | Dtype::Float64 | Dtype::Complex64
        ) {
            return Err(Error::Msg("PiD native matmul promoted exact dtype".into()));
        }
        let a = if a.dtype != dtype {
            self.cast(&format!("{site}/a-cast"), a, dtype)?
        } else {
            a.clone()
        };
        let b = if b.dtype != dtype {
            self.cast(&format!("{site}/b-cast"), b, dtype)?
        } else {
            b.clone()
        };
        let mut shape = broadcast(
            &a.shape[..na.saturating_sub(2)],
            &b.shape[..nb.saturating_sub(2)],
        )?;
        if na != 1 {
            shape.push(a.shape[na - 2]);
        }
        if nb != 1 {
            shape.push(b.shape[nb - 1]);
        }
        let native = self.workspace(site, "native-matmul-backend-and-queue", &[&a, &b]);
        let result = self.storage(site, shape, dtype, union([&a, &b]), false, false)?;
        self.native_output(native, &result, NativeCall::Matmul)?;
        self.complete(op, &result);
        Ok(result)
    }
    fn linear(
        &mut self,
        weights: &mut Sources,
        site: &str,
        x: &Tensor,
        bias_policy: bool,
        packed: bool,
    ) -> Result<Tensor> {
        let w = weights.require(self, &format!("{site}.weight"))?;
        let bias = if bias_policy {
            weights.get(self, &format!("{site}.bias"))?
        } else {
            None
        };
        if packed {
            if let Some(scales) = weights.get(self, &format!("{site}.scales"))? {
                let biases = weights.require(self, &format!("{site}.biases"))?;
                let mut shape = x.shape.clone();
                let last = shape
                    .last_mut()
                    .ok_or_else(|| Error::Msg("packed PiD scalar projection".into()))?;
                *last = w.shape[0];
                let bits = mlx_gen::quant::packed_bits_from_metadata(
                    &w.shape,
                    w.dtype,
                    &scales.shape,
                    scales.dtype,
                    64,
                )?;
                // native096 ops.cpp: affine QMM promotes x/scales/quant-biases TOGETHER,
                // then casts these three before the primitive. Stored scale dtype is not x.dtype.
                let dtype = promote_dtype(promote_dtype(x.dtype, scales.dtype), biases.dtype);
                let x = self.cast(&format!("{site}/native-qmm-x-cast"), x, dtype)?;
                let scales =
                    self.cast(&format!("{site}/native-qmm-scales-cast"), &scales, dtype)?;
                let biases =
                    self.cast(&format!("{site}/native-qmm-biases-cast"), &biases, dtype)?;
                let mut native_inputs = vec![&x, &w, &scales, &biases];
                if let Some(bias) = &bias {
                    native_inputs.push(bias);
                }
                let native = self.workspace(
                    site,
                    "native-quantized-matmul-backend-and-packed-layout",
                    &native_inputs,
                );
                let out = self.storage(
                    site,
                    shape,
                    dtype,
                    union([&x, &w, &scales, &biases]),
                    false,
                    false,
                )?;
                let has_output_bias = native_inputs.len() == 5;
                let final_out = if let Some(bias) = &bias {
                    self.binary(&format!("{site}/bias"), &out, bias)?
                } else {
                    out.clone()
                };
                let bias_output_slot =
                    has_output_bias.then(|| *final_out.roots.iter().next().unwrap());
                self.native_output(
                    native,
                    &out,
                    NativeCall::Qmm {
                        group_size: 64,
                        bits,
                        transpose: true,
                        bias_output_slot,
                    },
                )?;
                return Ok(final_out);
            }
        }
        let mut axes = (0..w.shape.len()).collect::<Vec<_>>();
        axes.reverse();
        let wt = self.transpose(&format!("{site}/transpose"), &w, &axes)?;
        // addmm is a fused native projection, not a numeric matmul+add claim. The positive
        // descriptor-work envelope records its promoted operands/output and backend obligation.
        if let Some(bias) = bias {
            let dtype = promote_dtype(promote_dtype(x.dtype, wt.dtype), bias.dtype);
            let op = self.attempt(&format!("{site}/native-addmm"), &[&bias, x, &wt]);
            if x.shape.is_empty() || wt.shape.is_empty() {
                return Err(Error::Msg("PiD native addmm scalar input".into()));
            }
            let nx = x.shape.len();
            let nw = wt.shape.len();
            if x.shape[nx - 1] != wt.shape[if nw == 1 { 0 } else { nw - 2 }] {
                return Err(Error::Msg("PiD native addmm contraction".into()));
            }
            if !matches!(
                dtype,
                Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32 | Dtype::Float64
            ) {
                // Existing native complex64 takes its composed branch; presence is not new served support.
                if dtype == Dtype::Complex64 {
                    let mm = self.matmul(&format!("{site}/existing-complex-matmul"), x, &wt)?;
                    let alpha =
                        self.scalar(&format!("{site}/existing-complex-alpha"), Dtype::Float32)?;
                    let beta =
                        self.scalar(&format!("{site}/existing-complex-beta"), Dtype::Float32)?;
                    let mm = self.binary(
                        &format!("{site}/existing-complex-scaled-product"),
                        &mm,
                        &alpha,
                    )?;
                    let c = self.binary(
                        &format!("{site}/existing-complex-scaled-bias"),
                        &bias,
                        &beta,
                    )?;
                    let out = self.binary(&format!("{site}/existing-complex-add"), &mm, &c)?;
                    self.complete(op, &out);
                    return Ok(out);
                }
                return Err(Error::Msg(
                    "PiD native addmm promoted nonfloating dtype".into(),
                ));
            }
            let x = self.cast(&format!("{site}/addmm-x"), x, dtype)?;
            let wt = self.cast(&format!("{site}/addmm-weight"), &wt, dtype)?;
            let bias = self.cast(&format!("{site}/addmm-bias"), &bias, dtype)?;
            let mut shape = broadcast(
                &x.shape[..nx.saturating_sub(2)],
                &wt.shape[..nw.saturating_sub(2)],
            )?;
            if nx != 1 {
                shape.push(x.shape[nx - 2]);
            }
            if nw != 1 {
                shape.push(wt.shape[nw - 1]);
            }
            let shape = broadcast(&shape, &bias.shape)?;
            let native = self.workspace(site, "native-addmm-fused-epilogue", &[&bias, &x, &wt]);
            let out = self.storage(
                &format!("{site}/addmm-output"),
                shape,
                dtype,
                union([&bias, &x, &wt]),
                false,
                false,
            )?;
            self.native_output(native, &out, NativeCall::Addmm)?;
            self.complete(op, &out);
            Ok(out)
        } else {
            self.matmul(site, x, &wt)
        }
    }
    fn rms(&mut self, site: &str, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let xf = self.cast(&format!("{site}/x-f32"), x, Dtype::Float32)?;
        let wf = self.cast(&format!("{site}/w-f32"), w, Dtype::Float32)?;
        broadcast(&xf.shape, &wf.shape)?;
        let native = self.workspace(site, "native-fast-rms-kernel-or-fallback", &[&xf, &wf]);
        let out = self.storage(
            site,
            xf.shape.clone(),
            Dtype::Float32,
            union([&xf, &wf]),
            false,
            false,
        )?;
        self.native_output(native, &out, NativeCall::Unbound)?;
        self.cast(&format!("{site}/round-trip"), &out, x.dtype)
    }
    fn gated(&mut self, site: &str, x: &Tensor, gate: &Tensor, y: &Tensor) -> Result<Tensor> {
        let product = self.binary(&format!("{site}/gate-product"), gate, y)?;
        let native = self.workspace(
            site,
            "source-eager-or-compiled-gated-epilogue",
            &[x, gate, y],
        );
        let out = self.binary(&format!("{site}/residual"), x, &product)?;
        self.native_output(native, &out, NativeCall::Unbound)?;
        Ok(out)
    }
    fn modulate(
        &mut self,
        site: &str,
        x: &Tensor,
        scale: &Tensor,
        shift: &Tensor,
    ) -> Result<Tensor> {
        let one = self.scalar(&format!("{site}/one"), Dtype::Float32)?;
        let scale = self.binary(&format!("{site}/one-plus-scale"), scale, &one)?;
        let out = self.binary(&format!("{site}/scaled"), x, &scale)?;
        let native = self.workspace(
            site,
            "source-eager-or-compiled-modulate-epilogue",
            &[x, &scale, shift],
        );
        let out = self.binary(&format!("{site}/shifted"), &out, shift)?;
        self.native_output(native, &out, NativeCall::Unbound)?;
        Ok(out)
    }
    fn silu(&mut self, site: &str, x: &Tensor) -> Result<Tensor> {
        let sigmoid = self.unary(&format!("{site}/sigmoid"), x)?;
        self.binary(site, x, &sigmoid)
    }
    fn gelu(&mut self, site: &str, x: &Tensor, tanh: bool) -> Result<Tensor> {
        // Source calls are retained individually, not a measured activation multiplier. Exact
        // fused/native fallback workspace remains an explicit qualifier beside this eager upper.
        let half = self.scalar(&format!("{site}/half"), x.dtype)?;
        let one = self.scalar(&format!("{site}/one"), x.dtype)?;
        let halfx = self.binary(&format!("{site}/half-x"), x, &half)?;
        let inner = if tanh {
            let x2 = self.binary(&format!("{site}/x-square"), x, x)?;
            let x3 = self.binary(&format!("{site}/x-cube"), &x2, x)?;
            let coefficient = self.scalar(&format!("{site}/cubic-coefficient"), x.dtype)?;
            let cubic = self.binary(&format!("{site}/cubic"), &x3, &coefficient)?;
            let sum = self.binary(&format!("{site}/inner"), x, &cubic)?;
            let scale = self.scalar(&format!("{site}/sqrt-two-over-pi"), x.dtype)?;
            let scaled = self.binary(&format!("{site}/scaled-inner"), &sum, &scale)?;
            self.unary(&format!("{site}/tanh"), &scaled)?
        } else {
            let c = self.scalar(&format!("{site}/inv-sqrt-two"), x.dtype)?;
            let scaled = self.binary(&format!("{site}/scaled-inner"), x, &c)?;
            self.unary(&format!("{site}/erf"), &scaled)?
        };
        let shifted = self.binary(&format!("{site}/one-plus"), &inner, &one)?;
        let native = self.workspace(site, "source-exact-epilogue-or-eager-gelu", &[x]);
        let out = self.binary(site, &halfx, &shifted)?;
        self.native_output(native, &out, NativeCall::Unbound)?;
        Ok(out)
    }
    fn split(&mut self, site: &str, x: &Tensor, parts: i32, axis: usize) -> Result<Vec<Tensor>> {
        let op = self.attempt(site, &[x]);
        if parts <= 0 || axis >= x.shape.len() || x.shape[axis] % parts != 0 {
            return Err(Error::Msg("PiD native equal split".into()));
        }
        let mut shape = x.shape.clone();
        shape[axis] /= parts;
        let out = (0..parts)
            .map(|part| {
                let value = self.owner_value(
                    shape.clone(),
                    x.dtype,
                    native_allocations::ValueRule::Split(x.value, axis, part * shape[axis]),
                );
                Tensor {
                    shape: shape.clone(),
                    dtype: x.dtype,
                    roots: x.roots.clone(),
                    value,
                }
            })
            .collect::<Vec<_>>();
        self.complete(op, &out[0]);
        Ok(out)
    }
    fn sdpa(&mut self, site: &str, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let op = self.attempt(site, &[q, k, v]);
        if q.shape.len() != 4 || k.shape.len() != 4 || v.shape.len() != 4 {
            return Err(Error::Msg("PiD SDPA head rank".into()));
        }
        let hd = q.shape[3];
        let padded = [64, 80, 128].into_iter().find(|&n| n >= hd).unwrap_or(hd);
        let mut inputs = Vec::new();
        for (index, t) in [q, k, v].into_iter().enumerate() {
            if padded != hd {
                let mut shape = t.shape.clone();
                shape[3] = padded;
                inputs.push(self.storage(
                    &format!("{site}/head-pad-{index}"),
                    shape,
                    t.dtype,
                    t.roots.clone(),
                    false,
                    false,
                )?);
            } else {
                inputs.push(t.clone());
            }
        }
        // Actual served Z PiD geometry: validator min_size256*sr4 and >=1024 tiles yield patch
        // and pixel query length >=4096 (patch MMDiT also adds text). Vector qL<=8 branch is
        // unreachable for THIS caller; full/NAX/fallback selection still binds native source.
        // fast.cpp989/metal SDPA use_fallback591: training (grad tracing), logsumexp output, or a
        // CPU stream force the composed fallback BEFORE head/qL checks. The qL>=4096 vector-branch
        // elimination holds ONLY for the priced Metal, non-grad-tracing inference path. Production
        // PiD decode is not under value_and_grad, but that is a source condition carried here, and
        // the CPU diagnostic fixture path is the fallback (materialized scores) obligation.
        let dispatch=self.workspace(site,"CONDITION metal-nongrad-nologsumexp else native-sdpa-composed-fallback-materialized-scores",&[&inputs[0],&inputs[1],&inputs[2]]);
        if q.shape[2] <= 8 {
            return Err(Error::Msg(
                "PiD served geometry reached unproved vector-SDPA branch".into(),
            ));
        }
        let workspace = if padded == 80 {
            "native-sdpa-full-noNAX-padded-head80-attention-workspace"
        } else {
            "native-sdpa-full-or-NAX-gated-workspace"
        };
        let native = self.workspace(site, workspace, &[&inputs[0], &inputs[1], &inputs[2]]);
        let output = self.storage(
            site,
            inputs[0].shape.clone(),
            inputs[0].dtype,
            union(inputs.iter()),
            false,
            false,
        )?;
        self.native_output(dispatch, &output, NativeCall::SdpaNoMaskNoSinks)?;
        self.native_output(native, &output, NativeCall::SdpaNoMaskNoSinks)?;
        let mut shape = output.shape.clone();
        shape[3] = hd;
        let value = self.owner_value(
            shape.clone(),
            output.dtype,
            native_allocations::ValueRule::Split(output.value, 3, 0),
        );
        let result = Tensor {
            shape,
            dtype: output.dtype,
            roots: output.roots,
            value,
        }; // fixed zero-copy tail split
        self.complete(op, &result);
        Ok(result)
    }
    fn window(
        &mut self,
        phase: IosPhase,
        site: &str,
        resident: &BTreeSet<Slot>,
        serial: u32,
        step: Option<usize>,
        tile: Option<mlx_gen_pid::tiling::SpatialTile>,
    ) -> Result<()> {
        // Freeze this JOINT source frontier now. Later native graph-input detachment must not
        // retroactively erase a prior caption/parent/window from its accounting evidence.
        let resident = closure(&self.prefix, resident)?;
        let created = closure(&self.prefix, &self.work.roots)?;
        let mut window = SourceWindow {
            phase,
            site: site.into(),
            resident_roots: resident,
            created_roots: created,
            host: self
                .host_live
                .values()
                .cloned()
                .chain(self.prefix.host[self.work.host_start..].iter().cloned())
                .collect(),
            workspace_start: self.work.workspace_start,
            workspace_end: self.prefix.workspaces.len(),
            operation_start: self.work.operation_start,
            operation_end: self.prefix.operations.len(),
            requests: Default::default(),
            owners: Default::default(),
            value_start: self.work.value_start,
            value_end: self.prefix.values.len(),
            serial_completed: serial,
            step,
            tile,
        };
        // Evaluate THIS exact chronology range before resetting work. Constructors, refusal
        // prefixes, every tile/step and serial endpoints use this SAME connected consumer.
        window.owners = native_allocations::join_owners(&self.prefix, &window)?;
        window.requests = native_allocations::evaluate(&self.prefix, &window)?;
        self.prefix.windows.push(window);
        self.begin();
        Ok(())
    }
    fn evaluated_graph_frontier(&mut self, site: &str, outputs: &[&Tensor]) -> Result<()> {
        self.evaluated_graph_frontier_kind(
            site,
            outputs,
            native_allocations::BoundaryKind::BareEval,
        )
    }
    fn evaluated_graph_frontier_kind(
        &mut self,
        site: &str,
        outputs: &[&Tensor],
        kind: native_allocations::BoundaryKind,
    ) -> Result<()> {
        self.owner_boundary(kind, site);
        let reachable = closure(&self.prefix, &union(outputs.iter().copied()))?;
        let inputs = reachable
            .iter()
            .map(|slot| {
                let c = &self.prefix.capacities[slot.0];
                (c.shape.clone(), c.dtype)
            })
            .collect();
        // Pinned transforms.cpp342-343 + array.cpp117-129: non-tracer scheduled ArrayDesc
        // inputs/siblings are detached. This is ONLY a graph fact. No last-Data-owner, queued
        // completion, allocator-cache, process or Darwin-physical release credit follows.
        self.prefix.workspaces.push(NativeWorkspace {
            site: site.into(),
            inputs,
            family:
                "native-retired-Data-queue-cache-process-ownership-NOT-deduced-from-graph-detach",
            input_values: outputs.iter().map(|t| t.value).collect(),
            affine_rows: None,
            output_slot: None,
            call: NativeCall::Unbound,
        });
        self.owner_boundary(native_allocations::BoundaryKind::GraphDetach, site);
        for slot in reachable {
            self.prefix.capacities[slot.0].graph_attached = false;
        }
        Ok(())
    }
}

#[derive(Clone, Default)]
struct Sources {
    tensors: BTreeMap<String, Tensor>,
    accessed: BTreeSet<String>,
    files: Vec<PathBuf>,
    conv_views: BTreeMap<String, Tensor>,
    headers: BTreeMap<String, wm::SafetensorsTensorHeader>,
}
impl Sources {
    fn file(program: &mut Program, path: &Path, domain: &str) -> Result<Self> {
        let mut sources = Self::default();
        sources.append(program, path, domain)?;
        Ok(sources)
    }
    fn append(&mut self, program: &mut Program, path: &Path, domain: &str) -> Result<()> {
        program.prefix.source_files.push(path.to_owned());
        let headers = wm::safetensors_file_tensor_locations(path).map_err(Error::from)?;
        let ordinal = self.files.len();
        self.files.push(path.to_owned());
        for location in headers.tensors {
            let header = location.header;
            self.headers.insert(header.name.clone(), header.clone());
            let shape = header
                .shape
                .iter()
                .map(|&d| {
                    i32::try_from(d)
                        .map_err(|_| Error::Msg("PiD header dimension exceeds native range".into()))
                })
                .collect::<Result<Vec<_>>>()?;
            let dtype = native_header_dtype(header.dtype)?;
            let t = program.storage(
                &format!("{domain}/file-{ordinal}/{}", header.name),
                shape,
                dtype,
                BTreeSet::new(),
                false,
                false,
            )?;
            program.owner_rule(
                t.value,
                native_allocations::ValueRule::HeaderRoot(*t.roots.iter().next().unwrap()),
            )?;
            for slot in &t.roots {
                program.work.roots.remove(slot);
            } // raw lazy map shell != read payload
            if self.tensors.insert(header.name.clone(), t).is_some() {
                return Err(Error::Msg(format!(
                    "duplicate tensor key `{}` across PiD shards",
                    header.name
                )));
            }
        }
        Ok(())
    }
    fn get(&mut self, _program: &mut Program, key: &str) -> Result<Option<Tensor>> {
        if let Some(t) = self.tensors.get(key) {
            self.accessed.insert(key.into());
            Ok(Some(t.clone()))
        } else {
            Ok(None)
        }
    }
    fn require(&mut self, program: &mut Program, key: &str) -> Result<Tensor> {
        self.get(program, key)?
            .ok_or_else(|| Error::MissingTensor(key.into()))
    }
    fn raw_accessed(&self) -> BTreeSet<Slot> {
        self.accessed
            .iter()
            .flat_map(|k| self.tensors[k].roots.iter().copied())
            .collect()
    }
    fn constructor_linear(
        &mut self,
        p: &mut Program,
        key: &str,
        bias: bool,
        packed: bool,
    ) -> Result<()> {
        self.require(p, &format!("{key}.weight"))?;
        if packed && self.get(p, &format!("{key}.scales"))?.is_some() {
            self.require(p, &format!("{key}.biases"))?;
            // SAME existing pure header selector as native quant::lin; no guessed packed width.
            mlx_gen::adapters::facts::linear_header_facts(
                &self.headers,
                key,
                mlx_gen::quant::DEFAULT_GROUP_SIZE,
                None,
            )?;
        }
        if bias {
            self.get(p, &format!("{key}.bias"))?;
        }
        Ok(())
    }
}

// Pinned native7a1d4f pure promotion table; enum presence is NOT new backend capability.
fn promote_dtype(a: Dtype, b: Dtype) -> Dtype {
    use Dtype::*;
    let index = |dtype| match dtype {
        Bool => 0,
        Uint8 => 1,
        Uint16 => 2,
        Uint32 => 3,
        Uint64 => 4,
        Int8 => 5,
        Int16 => 6,
        Int32 => 7,
        Int64 => 8,
        Float16 => 9,
        Float32 => 10,
        Float64 => 11,
        Bfloat16 => 12,
        Complex64 => 13,
    };
    const TABLE: [[Dtype; 14]; 14] = [
        [
            Bool, Uint8, Uint16, Uint32, Uint64, Int8, Int16, Int32, Int64, Float16, Float32,
            Float64, Bfloat16, Complex64,
        ],
        [
            Uint8, Uint8, Uint16, Uint32, Uint64, Int16, Int16, Int32, Int64, Float16, Float32,
            Float64, Bfloat16, Complex64,
        ],
        [
            Uint16, Uint16, Uint16, Uint32, Uint64, Int32, Int32, Int32, Int64, Float16, Float32,
            Float64, Bfloat16, Complex64,
        ],
        [
            Uint32, Uint32, Uint32, Uint32, Uint64, Int64, Int64, Int64, Int64, Float16, Float32,
            Float64, Bfloat16, Complex64,
        ],
        [
            Uint64, Uint64, Uint64, Uint64, Uint64, Float32, Float32, Float32, Float32, Float16,
            Float32, Float64, Bfloat16, Complex64,
        ],
        [
            Int8, Int16, Int32, Int64, Float32, Int8, Int16, Int32, Int64, Float16, Float32,
            Float64, Bfloat16, Complex64,
        ],
        [
            Int16, Int16, Int32, Int64, Float32, Int16, Int16, Int32, Int64, Float16, Float32,
            Float64, Bfloat16, Complex64,
        ],
        [
            Int32, Int32, Int32, Int64, Float32, Int32, Int32, Int32, Int64, Float16, Float32,
            Float64, Bfloat16, Complex64,
        ],
        [
            Int64, Int64, Int64, Int64, Float32, Int64, Int64, Int64, Int64, Float16, Float32,
            Float64, Bfloat16, Complex64,
        ],
        [
            Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float16,
            Float16, Float32, Float64, Float32, Complex64,
        ],
        [
            Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32,
            Float32, Float32, Float64, Float32, Complex64,
        ],
        [
            Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64,
            Float64, Float64, Float64, Float64, Complex64,
        ],
        [
            Bfloat16, Bfloat16, Bfloat16, Bfloat16, Bfloat16, Bfloat16, Bfloat16, Bfloat16,
            Bfloat16, Float32, Float32, Float64, Bfloat16, Complex64,
        ],
        [
            Complex64, Complex64, Complex64, Complex64, Complex64, Complex64, Complex64, Complex64,
            Complex64, Complex64, Complex64, Complex64, Complex64, Complex64,
        ],
    ];
    TABLE[index(a)][index(b)]
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum CacheKey {
    ImageRope {
        config: String,
        h: i32,
        w: i32,
    },
    PixelRope {
        config: String,
        h: i32,
        w: i32,
    },
    TextRope {
        config: String,
        tokens: i32,
    },
    PixelPosition {
        config: String,
        h: i32,
        w: i32,
    },
    Feather {
        h: i32,
        w: i32,
        top: bool,
        bottom: bool,
        left: bool,
        right: bool,
        overlap: i32,
    },
}
#[derive(Clone, Default)]
struct Caches {
    slots: BTreeMap<CacheKey, Vec<Tensor>>,
}
impl Caches {
    fn get_or_build(
        &mut self,
        p: &mut Program,
        key: CacheKey,
        build: impl FnOnce(&mut Program) -> Result<Vec<Tensor>>,
    ) -> Result<Vec<Tensor>> {
        if let Some(found) = self.slots.get(&key) {
            let op = p.attempt(
                "pid/cache-hit SAME endpoint slots",
                &found.iter().collect::<Vec<_>>(),
            );
            if let Some(first) = found.first() {
                p.complete(op, first);
            }
            return Ok(found.clone());
        }
        let values = build(p)?;
        self.slots.insert(key, values.clone());
        Ok(values)
    }
    fn roots(&self) -> BTreeSet<Slot> {
        self.slots
            .values()
            .flat_map(|v| v.iter())
            .flat_map(|t| t.roots.iter().copied())
            .collect()
    }
}
#[derive(Clone, Debug)]
struct Controls {
    use_pid: bool,
    capture_sigma: Option<f32>,
    selected_tile: Option<(i32, i32)>,
}
impl Controls {
    fn from_source(sampling: &ResolvedFlowControls, route: &Route) -> Result<Self> {
        // Serialize the EXISTING validated wire; do not build a synthetic GenerationRequest or
        // add a shared accessor/DTO just to project these private controls for token replay.
        let wire =
            serde_json::to_value(sampling.recipe()).map_err(|e| Error::Msg(e.to_string()))?;
        let use_pid = wire
            .get("use_pid")
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| Error::Msg("PiD recipe missing use_pid".into()))?;
        let capture = wire
            .get("pid_capture_sigma")
            .ok_or_else(|| Error::Msg("PiD recipe missing capture sigma".into()))?;
        let capture_sigma = if capture.is_null() {
            None
        } else {
            Some(
                capture
                    .as_f64()
                    .ok_or_else(|| Error::Msg("PiD capture scalar type".into()))?
                    as f32,
            )
        };
        let selected_tile = if route.execution.tile_vae_decode {
            route.execution.decode_tile_edge.map(|edge| {
                (
                    edge as i32,
                    route
                        .execution
                        .decode_overlap
                        .unwrap_or(mlx_gen_pid::budget::DEFAULT_TILE_OVERLAP as u32)
                        as i32,
                )
            })
        } else {
            None
        };
        Ok(Self {
            use_pid,
            capture_sigma,
            selected_tile,
        })
    }
}

#[derive(Clone)]
struct EngineSources {
    student: Sources,
    gemma: Sources,
    cfg: PidConfig,
    gemma_cfg: Gemma2Config,
    norms: BTreeMap<String, Tensor>,
    raw_engine: BTreeSet<Slot>,
    chi_tokens: i32,
    caches: Caches,
    config_domain: String,
}

fn constructor_norm(p: &mut Program, w: &mut Sources, key: &str) -> Result<Tensor> {
    let raw = w.require(p, key)?;
    let one = p.scalar(&format!("gemma-constructor/{key}/one"), raw.dtype)?;
    p.binary(
        &format!("gemma-constructor/{key}/one-plus-weight"),
        &raw,
        &one,
    )
}
fn construct_gemma(
    p: &mut Program,
    w: &mut Sources,
    cfg: &Gemma2Config,
) -> Result<BTreeMap<String, Tensor>> {
    let mut norms = BTreeMap::new();
    for layer in 0..cfg.num_layers {
        let base = format!("model.layers.{layer}");
        let key = format!("{base}.input_layernorm.weight");
        norms.insert(key.clone(), constructor_norm(p, w, &key)?);
        for leaf in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            w.constructor_linear(p, &format!("{base}.self_attn.{leaf}"), false, true)?;
        }
        for leaf in ["post_attention_layernorm", "pre_feedforward_layernorm"] {
            let key = format!("{base}.{leaf}.weight");
            norms.insert(key.clone(), constructor_norm(p, w, &key)?);
        }
        for leaf in ["gate_proj", "up_proj", "down_proj"] {
            w.constructor_linear(p, &format!("{base}.mlp.{leaf}"), false, true)?;
        }
        let key = format!("{base}.post_feedforward_layernorm.weight");
        norms.insert(key.clone(), constructor_norm(p, w, &key)?);
    }
    // Actual Gemma2::from_weights order is layers -> final norm -> embedding.
    let key = "model.norm.weight";
    norms.insert(key.into(), constructor_norm(p, w, key)?);
    let embedding = w.require(p, "model.embed_tokens.weight")?;
    if let Some(scales) = w.get(p, "model.embed_tokens.scales")? {
        let biases = w.require(p, "model.embed_tokens.biases")?;
        if scales.shape.len() != 2
            || biases.shape != scales.shape
            || biases.dtype != scales.dtype
            || mul(scales.shape[1], 64)? != cfg.hidden_size
        {
            return Err(Error::Msg("Gemma-2 actual packed embedding layout".into()));
        }
        mlx_gen::adapters::facts::linear_header_facts(
            &w.headers,
            "model.embed_tokens",
            mlx_gen::quant::DEFAULT_GROUP_SIZE,
            None,
        )?;
        p.workspace(
            "Gemma embedding constructor",
            "native-packed-embedding-layout-and-backend-domain",
            &[&embedding, &scales, &biases],
        );
    } else {
        if w.get(p, "model.embed_tokens.biases")?.is_some()
            || embedding.shape.len() != 2
            || embedding.shape[1] != cfg.hidden_size
            || !matches!(
                embedding.dtype,
                Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32
            )
        {
            return Err(Error::Msg("Gemma-2 actual dense embedding layout".into()));
        }
    }
    Ok(norms)
}
fn construct_student(p: &mut Program, w: &mut Sources, cfg: &PidConfig) -> Result<()> {
    // All source gets below match the native constructors, including optional biases and norm.
    for i in 0..cfg.patch_depth {
        let b = format!("patch_blocks.{i}");
        for leaf in ["norm_x1", "norm_y1"] {
            w.require(p, &format!("{b}.{leaf}.weight"))?;
        }
        for leaf in ["qkv_x", "qkv_y"] {
            w.constructor_linear(p, &format!("{b}.attn.{leaf}"), true, false)?;
        }
        for leaf in ["q_norm_x", "k_norm_x", "q_norm_y", "k_norm_y"] {
            w.require(p, &format!("{b}.attn.{leaf}.weight"))?;
        }
        for leaf in ["proj_x", "proj_y"] {
            w.constructor_linear(p, &format!("{b}.attn.{leaf}"), true, false)?;
        }
        for leaf in ["norm_x2", "norm_y2"] {
            w.require(p, &format!("{b}.{leaf}.weight"))?;
        }
        for stream in ["mlp_x", "mlp_y"] {
            for leaf in ["w1", "w2", "w3"] {
                w.constructor_linear(p, &format!("{b}.{stream}.{leaf}"), true, false)?;
            }
        }
        for leaf in ["adaLN_modulation_img.0", "adaLN_modulation_txt.0"] {
            w.constructor_linear(p, &format!("{b}.{leaf}"), true, false)?;
        }
    }
    for i in 0..cfg.pixel_depth {
        let b = format!("pixel_blocks.{i}");
        for leaf in ["compress_to_attn", "expand_from_attn"] {
            w.constructor_linear(p, &format!("{b}.{leaf}"), true, false)?;
        }
        w.require(p, &format!("{b}.norm1.weight"))?;
        w.constructor_linear(p, &format!("{b}.attn.qkv"), true, false)?;
        for leaf in ["q_norm", "k_norm"] {
            w.require(p, &format!("{b}.attn.{leaf}.weight"))?;
        }
        w.constructor_linear(p, &format!("{b}.attn.proj"), true, false)?;
        w.require(p, &format!("{b}.norm2.weight"))?;
        for leaf in ["mlp.fc1", "mlp.fc2", "adaLN_modulation.0"] {
            w.constructor_linear(p, &format!("{b}.{leaf}"), true, false)?;
        }
    }
    for leaf in [
        "pixel_embedder.proj",
        "s_embedder.proj",
        "t_embedder.mlp.0",
        "t_embedder.mlp.2",
        "y_embedder.proj",
    ] {
        w.constructor_linear(p, leaf, true, false)?;
    }
    w.get(p, "s_embedder.norm.weight")?;
    w.get(p, "y_embedder.norm.weight")?;
    w.require(p, "y_pos_embedding")?;
    w.require(p, "final_layer.norm.weight")?;
    w.constructor_linear(p, "final_layer.linear", true, false)?;
    let input = w.require(p, "lq_proj.latent_proj.0.weight")?;
    if input.shape.len() < 2 {
        return Err(Error::Msg("PiD actual LQ first-conv shape".into()));
    }
    let n = cfg.lq_latent_channels;
    let cin = input.shape[1];
    if cin <= 0 || n % cin != 0 {
        return Err(Error::Msg("PiD native LQ unpatchify channel ratio".into()));
    }
    let factor = ((n / cin) as f64).sqrt().round() as i32;
    if mul(factor, factor)? != n / cin {
        return Err(Error::Msg("PiD native LQ non-square unpatchify".into()));
    }
    for leaf in ["0", "2"] {
        construct_conv(p, w, &format!("lq_proj.latent_proj.{leaf}"))?;
    }
    for i in 0..cfg.lq_num_res_blocks {
        let b = format!("lq_proj.latent_proj.{}.block", i + 3);
        for leaf in ["0", "3"] {
            for suffix in ["weight", "bias"] {
                w.require(p, &format!("{b}.{leaf}.{suffix}"))?;
            }
        }
        for leaf in ["2", "5"] {
            construct_conv(p, w, &format!("{b}.{leaf}"))?;
        }
    }
    for i in 0..cfg.num_lq_outputs() {
        w.constructor_linear(p, &format!("lq_proj.output_heads.{i}"), true, false)?;
    }
    if cfg.pit_lq_inject {
        w.constructor_linear(p, "lq_proj.pit_head", true, false)?;
    }
    for i in 0..cfg.num_lq_outputs() {
        construct_gate(p, w, &format!("lq_proj.gate_modules.{i}"))?;
    }
    if cfg.pit_lq_inject {
        construct_gate(p, w, "pit_lq_gate")?;
    }
    Ok(())
}
fn construct_gate(p: &mut Program, w: &mut Sources, site: &str) -> Result<()> {
    w.constructor_linear(p, &format!("{site}.content_proj"), true, false)?;
    w.require(p, &format!("{site}.log_alpha"))?;
    Ok(())
}
fn construct_conv(p: &mut Program, w: &mut Sources, site: &str) -> Result<()> {
    let raw = w.require(p, &format!("{site}.weight"))?;
    let view = p.transpose(&format!("{site}/NCHW-to-NHWC"), &raw, &[0, 2, 3, 1])?;
    w.conv_views.insert(site.into(), view);
    w.get(p, &format!("{site}.bias"))?;
    Ok(())
}
fn materialize_accessed(
    p: &mut Program,
    w: &Sources,
    site: &str,
    held: &BTreeSet<Slot>,
) -> Result<()> {
    // Actual Weights sorts keys and batches at512MiB. Retain ordered source batch attempts;
    // no Adapter.materialize, and no claim that raw eval detaches norm/conv endpoints.
    let mut bytes = 0u64;
    let mut batch = BTreeSet::new();
    let mut resident = held.clone();
    for key in &w.accessed {
        let t = &w.tensors[key];
        batch.extend(t.roots.iter().copied());
        bytes = bytes
            .checked_add(
                t.elements()?
                    .checked_mul(dtype_width(t.dtype))
                    .ok_or_else(|| Error::Msg("PiD raw batch bytes overflow".into()))?,
            )
            .ok_or_else(|| Error::Msg("PiD raw batch sum overflow".into()))?;
        if bytes >= 512 * 1024 * 1024 {
            p.work.roots.extend(batch.iter().copied());
            p.window(IosPhase::Load, site, &resident, 0, None, None)?;
            p.raw_io_frontiers.push(Arc::new(p.prefix.clone()));
            resident.extend(batch.iter().copied());
            batch.clear();
            bytes = 0;
        }
    }
    if !batch.is_empty() {
        p.work.roots.extend(batch);
        p.window(IosPhase::Load, site, &resident, 0, None, None)?;
        p.raw_io_frontiers.push(Arc::new(p.prefix.clone()));
    }
    Ok(())
}

fn open_engine(p: &mut Program, spec: &LoadSpec, scope: u64) -> Result<EngineSources> {
    let pid = spec
        .pid
        .as_ref()
        .ok_or_else(|| Error::Msg("use_pid requested without configured PiD weights".into()))?;
    let WeightsSource::File(checkpoint) = &pid.checkpoint else {
        return Err(Error::Msg("PiD checkpoint must be File".into()));
    };
    let WeightsSource::Dir(gemma_dir) = &pid.gemma else {
        return Err(Error::Msg("PiD Gemma must be snapshot Dir".into()));
    };
    p.hold_host(
        "PiD engine SamplerConfig t_list",
        SamplerConfig::distill_4step().student_t_list.len() as u64 * 4,
    );
    let mut student = Sources::file(p, checkpoint, &format!("pid-scope-{scope}/student"))?;
    let gate = student.require(p, "lq_proj.gate_modules.0.content_proj.weight")?;
    let rows = *gate
        .shape
        .first()
        .ok_or_else(|| Error::Msg("PiD version gate has no native row dimension".into()))?;
    let v1pt5 = mlx_gen_pid::engine::detect_v1pt5_signals(
        rows,
        student.get(p, "pit_lq_gate.content_proj.weight")?.is_some(),
    )?;
    let space = mlx_gen_pid::lookup(crate::model::PID_BACKBONE)
        .ok_or_else(|| Error::Msg("unknown Z PiD backbone".into()))?;
    let mut cfg = if v1pt5 {
        PidConfig::sr4x_v1pt5()
    } else {
        PidConfig::sr4x()
    };
    cfg.lq_latent_channels = space.latent_channels;
    cfg.latent_spatial_down_factor = space.latent_spatial_down_factor;
    cfg.sr_scale = space.pid_scale;
    let merged = gemma_dir.join(mlx_gen_pid::engine::GEMMA_MERGED_FILE);
    let mut gemma = Sources::default();
    if merged.is_file() {
        gemma.append(p, &merged, &format!("pid-scope-{scope}/gemma"))?;
    } else {
        let mut files = std::fs::read_dir(gemma_dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .filter(|path| !wm::is_hidden_file(path))
            .collect::<Vec<_>>();
        files.sort();
        if files.is_empty() {
            return Err(Error::Msg("PiD Gemma has no safetensors files".into()));
        }
        for file in files {
            gemma.append(p, &file, &format!("pid-scope-{scope}/gemma"))?;
        }
    }
    let gemma_cfg = Gemma2Config::gemma_2_2b();
    let norms = construct_gemma(p, &mut gemma, &gemma_cfg)?;
    let norm_roots = norms
        .values()
        .flat_map(|t| t.roots.iter().copied())
        .collect::<BTreeSet<_>>();
    p.window(
        IosPhase::Load,
        "PiD/Gemma actual constructor prefix",
        &BTreeSet::new(),
        0,
        None,
        None,
    )?;
    materialize_accessed(p, &gemma, "PiD/Gemma RAW accessed batch", &norm_roots)?;
    // This host tokenizer operation is actually reached only AFTER the Gemma raw materialize.
    let tokenizer = mlx_gen::tokenizer::TextTokenizer::from_file(
        gemma_dir.join("tokenizer.json"),
        mlx_gen::tokenizer::TokenizerConfig {
            max_length: 4096,
            pad_token_id: 0,
            chat_template: mlx_gen::tokenizer::ChatTemplate::None,
            pad_to_max_length: false,
        },
    )?;
    // tokenizers0.21.4 cache.rs53-55: BPE cache table allocated EAGERLY at engine load and
    // retained (grows across requests up to 10000 entries of <256B words) for engine lifetime.
    p.hold_host(
        "PiD Gemma BPE retained cache entries (<=10000 x (key<256B + Word<=255x32B))",
        10_000u64 * (255 + 255 * 32),
    );
    p.prefix.host.push(HostDemand::Symbolic {
        site: "PiD Gemma BPE AHashMap::with_capacity(10000) table layout".into(),
        symbol: "hashbrown_ahash_table_bytes_for_10000",
        coefficient: 1,
    });
    let chi_tokens = i32::try_from(
        tokenizer
            .encode_ids(mlx_gen_pid::caption::CHI_PROMPT, true)?
            .len(),
    )
    .map_err(|_| Error::Msg("PiD CHI token count exceeds native range".into()))?;
    construct_student(p, &mut student, &cfg)?;
    p.window(
        IosPhase::Load,
        "PiD/student actual probe constructor",
        &norm_roots,
        0,
        None,
        None,
    )?;
    let mut raw_engine = norm_roots;
    for key in &gemma.accessed {
        if !norms.contains_key(key) {
            raw_engine.extend(gemma.tensors[key].roots.iter().copied());
        }
    }
    p.prefix.host.push(HostDemand::Symbolic {
        site: "PiD retained student dictionary/ArrayDesc shells".into(),
        symbol: "native_ArrayDesc_and_dictionary_entry_bytes",
        coefficient: student.tensors.len() as u64,
    });
    materialize_accessed(p, &student, "PiD/student RAW accessed batch", &raw_engine)?;
    raw_engine.extend(student.raw_accessed());
    let config_domain = format!("pid-scope-{scope}/{cfg:?}");
    Ok(EngineSources {
        student,
        gemma,
        cfg,
        gemma_cfg,
        norms,
        raw_engine,
        chi_tokens,
        caches: Caches::default(),
        config_domain,
    })
}

fn gemma_caption(
    p: &mut Program,
    e: &mut EngineSources,
    work_basis: Option<PidWorkBasis>,
) -> Result<Tensor> {
    let c = e.gemma_cfg.clone();
    let len = e
        .chi_tokens
        .checked_add(298)
        .ok_or_else(|| Error::Msg("PiD CHI padding length overflow".into()))?;
    let held = e.raw_engine.clone();
    let ids = p.host_input("PiD caption input ids", vec![1, len], Dtype::Int32)?;
    let mask_ids = p.host_input("PiD caption padding ids", vec![1, len], Dtype::Int32)?;
    p.host(
        "PiD caption host ids/mask",
        elements(&[2, len])?
            .checked_mul(4)
            .ok_or_else(|| Error::Msg("PiD caption host bytes".into()))?,
    );
    let work = PidWorkBasis::require(work_basis)?;
    let bytes = work
        .prompt_utf8_bytes
        .checked_add(mlx_gen_pid::caption::CHI_PROMPT.len() as u64)
        .ok_or_else(|| Error::Msg("PiD CHI+caption UTF8 work overflow".into()))?;
    p.host(
        "PiD actual CHI+wholepretruncate caption string length",
        bytes,
    );
    // This is exact input length, not a claim that String/malloc capacity equals length.
    // The separate source-bound format/tokenizer allocator terms consume this SAME dimension.
    p.prefix.host.push(HostDemand::Symbolic {
        site: "PiD format/string allocator capacity over actualUTF8 length".into(),
        symbol: "format_allocation_capacity_over_exact_utf8_bytes",
        coefficient: bytes,
    });
    p.prefix.host.push(HostDemand::Symbolic {
        site: "PiD tokenizer arena before truncation".into(),
        symbol: "tokenizer_pretruncate_host_arena",
        coefficient: 1,
    });
    let weight = e.gemma.require(p, "model.embed_tokens.weight")?;
    let scales = e.gemma.get(p, "model.embed_tokens.scales")?;
    let dtype = scales.as_ref().map_or(weight.dtype, |t| t.dtype);
    let rows = if let Some(scales) = scales {
        let biases = e.gemma.require(p, "model.embed_tokens.biases")?;
        let bits = mlx_gen::quant::packed_bits_from_metadata(
            &weight.shape,
            weight.dtype,
            &scales.shape,
            scales.dtype,
            64,
        )?;
        let (codes, codes_source) =
            p.embedding_take_axis_zero("Gemma gather packed rows", &weight, &ids)?;
        let (s, scales_source) =
            p.embedding_take_axis_zero("Gemma gather row scales", &scales, &ids)?;
        let (b, biases_source) =
            p.embedding_take_axis_zero("Gemma gather row biases", &biases, &ids)?;
        p.gathered_affine_rows(
            &codes,
            &s,
            &b,
            [codes_source, scales_source, biases_source],
            bits,
        )?
    } else {
        p.embedding_take_axis_zero("Gemma dense row gather", &weight, &ids)?
            .0
    };
    let widened = p.cast("Gemma common embedding F32 result", &rows, Dtype::Float32)?;
    let native = p.cast("Gemma embedding stored-dtype return", &widened, dtype)?;
    let scalar = p.scalar("Gemma embedding sqrt-hidden scalar", dtype)?;
    let mut x = p.binary("Gemma scaled token embedding", &native, &scalar)?;
    p.host("Gemma host RoPE inv F64", (c.head_dim / 2) as u64 * 8);
    p.host(
        "Gemma host RoPE cos+sin F32",
        elements(&[2, len, c.head_dim])? * 4,
    );
    let cos = p.host_input("Gemma RoPE cos", vec![len, c.head_dim], Dtype::Float32)?;
    let sin = p.host_input("Gemma RoPE sin", vec![len, c.head_dim], Dtype::Float32)?;
    p.host("Gemma host causal mask", elements(&[len, len])? * 4);
    let causal = p.host_input("Gemma causal mask", vec![1, 1, len, len], Dtype::Float32)?;
    let maskf = p.cast("Gemma padding mask F32", &mask_ids, Dtype::Float32)?;
    let one = p.scalar("Gemma padding one", Dtype::Float32)?;
    let diff = p.binary("Gemma padding subtraction", &one, &maskf)?;
    let negative = p.scalar("Gemma padding negative", Dtype::Float32)?;
    let padding = p.binary("Gemma padding logits", &diff, &negative)?;
    let padding = p.view(
        "Gemma padding mask view",
        &padding,
        vec![1, 1, 1, len],
        false,
    )?;
    let mask = p.binary("Gemma combined causal and padding mask", &causal, &padding)?;
    p.window(
        IosPhase::Denoise,
        "PiD/Gemma embedding-mask",
        &held,
        0,
        None,
        None,
    )?;
    p.evaluated_graph_frontier_kind(
        "Gemma embedding/mask source eval",
        &[&x, &cos, &sin, &mask],
        native_allocations::BoundaryKind::ConditionalQuantumFinish,
    )?;
    for layer in 0..c.num_layers {
        let b = format!("model.layers.{layer}");
        let norm = e.norms[&format!("{b}.input_layernorm.weight")].clone();
        let normalized = p.rms(&format!("{b}/input-rms"), &x, &norm)?;
        let q = p.linear(
            &mut e.gemma,
            &format!("{b}.self_attn.q_proj"),
            &normalized,
            false,
            true,
        )?;
        let k = p.linear(
            &mut e.gemma,
            &format!("{b}.self_attn.k_proj"),
            &normalized,
            false,
            true,
        )?;
        let v = p.linear(
            &mut e.gemma,
            &format!("{b}.self_attn.v_proj"),
            &normalized,
            false,
            true,
        )?;
        let q = gemma_heads_rope(
            p,
            &format!("{b}/q"),
            &q,
            c.num_heads,
            c.head_dim,
            &cos,
            &sin,
        )?;
        let k = gemma_heads_rope(
            p,
            &format!("{b}/k"),
            &k,
            c.num_kv_heads,
            c.head_dim,
            &cos,
            &sin,
        )?;
        let v = p.view(
            &format!("{b}/v-head-view"),
            &v,
            vec![1, len, c.num_kv_heads, c.head_dim],
            false,
        )?;
        let v = p.transpose(&format!("{b}/v-head-major"), &v, &[0, 2, 1, 3])?;
        let repeats = c.num_heads / c.num_kv_heads;
        let k = gemma_repeat(p, &format!("{b}/repeat-k"), &k, repeats)?;
        let v = gemma_repeat(p, &format!("{b}/repeat-v"), &v, repeats)?;
        let kt = p.transpose(&format!("{b}/k-transpose"), &k, &[0, 1, 3, 2])?;
        let scores = p.matmul(&format!("{b}/scores"), &q, &kt)?;
        let scale = p.scalar(&format!("{b}/query-scale"), q.dtype)?;
        let scores = p.binary(&format!("{b}/scaled-scores"), &scores, &scale)?;
        let cap = p.scalar(&format!("{b}/softcap"), scores.dtype)?;
        let inverse = p.scalar(&format!("{b}/inverse-softcap"), scores.dtype)?;
        let inner = p.binary(&format!("{b}/softcap-inner"), &scores, &inverse)?;
        let tanh = p.unary(&format!("{b}/softcap-tanh"), &inner)?;
        let scores = p.binary(&format!("{b}/softcapped"), &tanh, &cap)?;
        let scores = p.cast(&format!("{b}/logits-F32"), &scores, Dtype::Float32)?;
        let maskf = p.cast(&format!("{b}/mask-F32"), &mask, Dtype::Float32)?;
        let scores = p.binary(&format!("{b}/masked-logits"), &scores, &maskf)?;
        let native = p.workspace(
            &format!("{b}/softmax"),
            "native-explicit-softmax-not-flash",
            &[&scores],
        );
        let attn = p.unary(&format!("{b}/softmax"), &scores)?;
        p.native_output(native, &attn, NativeCall::Unbound)?;
        let attn = p.cast(&format!("{b}/softmax-native-dtype"), &attn, x.dtype)?;
        let out = p.matmul(&format!("{b}/attention-value"), &attn, &v)?;
        let out = p.transpose(&format!("{b}/attention-token-major"), &out, &[0, 2, 1, 3])?;
        let out = p.view(
            &format!("{b}/attention-merge"),
            &out,
            vec![1, len, mul(c.num_heads, c.head_dim)?],
            true,
        )?;
        let out = p.linear(
            &mut e.gemma,
            &format!("{b}.self_attn.o_proj"),
            &out,
            false,
            true,
        )?;
        let norm = e.norms[&format!("{b}.post_attention_layernorm.weight")].clone();
        let out = p.rms(&format!("{b}/post-attn-rms"), &out, &norm)?;
        x = p.binary(&format!("{b}/attention-residual"), &x, &out)?;
        let norm = e.norms[&format!("{b}.pre_feedforward_layernorm.weight")].clone();
        let normalized = p.rms(&format!("{b}/pre-ff-rms"), &x, &norm)?;
        let gate = p.linear(
            &mut e.gemma,
            &format!("{b}.mlp.gate_proj"),
            &normalized,
            false,
            true,
        )?;
        let gate = p.gelu(&format!("{b}/gelu-tanh"), &gate, true)?;
        let up = p.linear(
            &mut e.gemma,
            &format!("{b}.mlp.up_proj"),
            &normalized,
            false,
            true,
        )?;
        let product = p.binary(&format!("{b}/ff-product"), &gate, &up)?;
        let down = p.linear(
            &mut e.gemma,
            &format!("{b}.mlp.down_proj"),
            &product,
            false,
            true,
        )?;
        let norm = e.norms[&format!("{b}.post_feedforward_layernorm.weight")].clone();
        let down = p.rms(&format!("{b}/post-ff-rms"), &down, &norm)?;
        x = p.binary(&format!("{b}/ff-residual"), &x, &down)?;
        let mut resident = held.clone();
        resident.extend(
            cos.roots
                .iter()
                .chain(sin.roots.iter())
                .chain(mask.roots.iter())
                .copied(),
        );
        p.window(
            IosPhase::Denoise,
            &format!("PiD/Gemma layer-{layer}"),
            &resident,
            0,
            None,
            None,
        )?;
        p.evaluated_graph_frontier_kind(
            "Gemma layer source eval",
            &[&x],
            native_allocations::BoundaryKind::ConditionalQuantumFinish,
        )?;
    }
    let norm = e.norms["model.norm.weight"].clone();
    x = p.rms("PiD/Gemma final norm", &x, &norm)?;
    p.window(
        IosPhase::Denoise,
        "PiD/Gemma final norm",
        &held,
        0,
        None,
        None,
    )?;
    p.evaluated_graph_frontier_kind(
        "Gemma final norm source eval",
        &[&x],
        native_allocations::BoundaryKind::ConditionalQuantumFinish,
    )?;
    p.host("PiD caption select-index host", 300 * 4);
    p.host("PiD selected-mask host", 300 * 4);
    let indices = p.immediate("PiD caption select-index", vec![300], Dtype::Int32)?;
    let _selected_mask = p.immediate("PiD selected mask", vec![1, 300], Dtype::Float32)?;
    let selected = p.storage(
        "PiD caption gather300",
        vec![1, 300, c.hidden_size],
        x.dtype,
        union([&x, &indices]),
        false,
        false,
    )?;
    let selected = p.cast("PiD caption BF16 lazy endpoint", &selected, Dtype::Bfloat16)?;
    // Selection/cast is lazy. Its roots STILL name final-hidden/selection ancestry; eval is not
    // being reinterpreted here as a last-native-Data release or an allocator-cache purge.
    p.window(
        IosPhase::Denoise,
        "PiD/caption gather and BF16 construction",
        &held,
        0,
        None,
        None,
    )?;
    Ok(selected)
}
fn gemma_repeat(p: &mut Program, site: &str, x: &Tensor, repeats: i32) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(x.clone());
    }
    let mut shape = x.shape.clone();
    shape[1] = mul(shape[1], repeats)?;
    p.storage(site, shape, x.dtype, x.roots.clone(), false, true)
}
fn gemma_heads_rope(
    p: &mut Program,
    site: &str,
    x: &Tensor,
    heads: i32,
    dim: i32,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor> {
    let len = x.shape[1];
    let x = p.view(
        &format!("{site}/head-view"),
        x,
        vec![1, len, heads, dim],
        false,
    )?;
    let x = p.transpose(&format!("{site}/head-major"), &x, &[0, 2, 1, 3])?;
    let cos = p.view(
        &format!("{site}/cos-view"),
        cos,
        vec![1, 1, len, dim],
        false,
    )?;
    let sin = p.view(
        &format!("{site}/sin-view"),
        sin,
        vec![1, 1, len, dim],
        false,
    )?;
    let cos = p.cast(&format!("{site}/cos-cast"), &cos, x.dtype)?;
    let sin = p.cast(&format!("{site}/sin-cast"), &sin, x.dtype)?;
    let halves = p.split(&format!("{site}/rotate-half-split"), &x, 2, 3)?;
    let negative = p.unary(&format!("{site}/negative-half"), &halves[1])?;
    let rotated = p.concat(
        &format!("{site}/rotate-half-concat"),
        &[&negative, &halves[0]],
        3,
    )?;
    let xc = p.binary(&format!("{site}/cos-product"), &x, &cos)?;
    let xs = p.binary(&format!("{site}/sin-product"), &rotated, &sin)?;
    p.binary(&format!("{site}/rotated-sum"), &xc, &xs)
}

fn conv(
    p: &mut Program,
    w: &mut Sources,
    site: &str,
    x: &Tensor,
    mode: ConvPadding,
) -> Result<Tensor> {
    let weight = w
        .conv_views
        .get(site)
        .cloned()
        .ok_or_else(|| Error::Msg(format!("PiD constructor endpoint missing: {site}")))?;
    let bias = w.get(p, &format!("{site}.bias"))?;
    let op = p.attempt(site, &[x, &weight]);
    if x.shape.len() != 4 || weight.shape.len() != 4 || x.shape[3] != weight.shape[3] {
        return Err(Error::Msg("PiD native LQ convolution shape".into()));
    }
    let mut input = x.clone();
    if mode == ConvPadding::Replicate {
        let mut shape = x.shape.clone();
        shape[1] = shape[1]
            .checked_add(2)
            .ok_or_else(|| Error::Msg("PiD edge pad overflow".into()))?;
        shape[2] = shape[2]
            .checked_add(2)
            .ok_or_else(|| Error::Msg("PiD edge pad overflow".into()))?;
        input = p.storage(
            &format!("{site}/replicate-pad"),
            shape,
            x.dtype,
            x.roots.clone(),
            false,
            false,
        )?;
    }
    let dtype = promote_dtype(input.dtype, weight.dtype);
    let input = p.cast(&format!("{site}/input-cast"), &input, dtype)?;
    let weight = p.cast(&format!("{site}/weight-cast"), &weight, dtype)?;
    let mut shape = x.shape.clone();
    shape[3] = weight.shape[0];
    shape[1] = shape[1]
        .checked_add(3)
        .and_then(|n| n.checked_sub(weight.shape[1]))
        .ok_or_else(|| Error::Msg("PiD native convolution output height".into()))?;
    shape[2] = shape[2]
        .checked_add(3)
        .and_then(|n| n.checked_sub(weight.shape[2]))
        .ok_or_else(|| Error::Msg("PiD native convolution output width".into()))?;
    let mut native_inputs = vec![&input, &weight];
    if let Some(bias) = &bias {
        native_inputs.push(bias);
    }
    let native = p.workspace(
        site,
        "native-convolution-dispatch-and-exact-or-composed-bias",
        &native_inputs,
    );
    let out = p.storage(site, shape, dtype, union([&input, &weight]), false, false)?;
    let plain_output = *out
        .roots
        .iter()
        .next()
        .ok_or_else(|| Error::Msg("PiD plain convolution output missing".into()))?;
    p.owner_rule(
        out.value,
        native_allocations::ValueRule::FreshMetal(plain_output),
    )?;
    let has_bias = bias.is_some();
    let out = if let Some(bias) = bias {
        p.binary(&format!("{site}/bias"), &out, &bias)?
    } else {
        out
    };
    p.native_output(
        native,
        &out,
        NativeCall::Conv {
            padding: if mode == ConvPadding::Replicate { 0 } else { 1 },
            groups: 1,
            has_bias,
            plain_output,
        },
    )?;
    p.complete(op, &out);
    Ok(out)
}
fn group_norm(p: &mut Program, w: &mut Sources, site: &str, x: &Tensor) -> Result<Tensor> {
    let weight = w.require(p, &format!("{site}.weight"))?;
    let bias = w.require(p, &format!("{site}.bias"))?;
    let op = p.attempt(site, &[x, &weight, &bias]);
    if x.shape.len() < 2 || x.shape.last().unwrap() % 4 != 0 {
        return Err(Error::Msg("PiD native GroupNorm four-group gate".into()));
    }
    let native = p.workspace(
        site,
        "native-group-norm-layernorm-and-exact-epilogue",
        &[x, &weight, &bias],
    );
    // Norm kernel output plus actual affine source operations; native workspace is not assigned0.
    let normalized = p.storage(
        &format!("{site}/normalized"),
        x.shape.clone(),
        x.dtype,
        union([x, &weight]),
        false,
        false,
    )?;
    let normalized_output = *normalized
        .roots
        .iter()
        .next()
        .ok_or_else(|| Error::Msg("PiD composed norm source output missing".into()))?;
    let scaled = p.binary(&format!("{site}/affine-scale"), &normalized, &weight)?;
    let out = p.binary(&format!("{site}/affine-shift"), &scaled, &bias)?;
    // Native exact GroupNormAffine returns the final affine dtype/output. The composed
    // normalized/scale slots remain in the source tape, but are not a second native output.
    p.native_output(
        native,
        &out,
        NativeCall::GroupNorm {
            groups: 4,
            normalized_output,
        },
    )?;
    p.complete(op, &out);
    Ok(out)
}
fn lq_features(
    p: &mut Program,
    e: &mut EngineSources,
    input: &Tensor,
    h: i32,
    w: i32,
) -> Result<(Vec<Tensor>, Option<Tensor>)> {
    let c = e.cfg.clone();
    let first = e.student.require(p, "lq_proj.latent_proj.0.weight")?;
    let cin = first.shape[1];
    let factor = ((c.lq_latent_channels / cin) as f64).sqrt().round() as i32;
    let mut input = input.clone();
    if factor > 1 {
        let sh = input.shape.clone();
        let unpack = p.view(
            "PiD LQ unpatchify split",
            &input,
            vec![
                sh[0],
                sh[1] / mul(factor, factor)?,
                factor,
                factor,
                sh[2],
                sh[3],
            ],
            false,
        )?;
        let unpack = p.transpose("PiD LQ unpatchify transpose", &unpack, &[0, 1, 4, 2, 5, 3])?;
        input = p.view(
            "PiD LQ unpatchify collapse",
            &unpack,
            vec![
                sh[0],
                sh[1] / mul(factor, factor)?,
                mul(sh[2], factor)?,
                mul(sh[3], factor)?,
            ],
            true,
        )?;
    }
    let mut x = p.transpose("PiD LQ latent NHWC", &input, &[0, 2, 3, 1])?;
    let ratio = (c.sr_scale * c.latent_spatial_down_factor / (c.patch_size * factor)).max(1);
    if ratio > 1 {
        let mut shape = x.shape.clone();
        shape[1] = mul(shape[1], ratio)?;
        shape[2] = mul(shape[2], ratio)?;
        x = p.storage(
            "PiD nearest LQ upsample",
            shape,
            x.dtype,
            x.roots.clone(),
            false,
            true,
        )?;
    }
    x = conv(
        p,
        &mut e.student,
        "lq_proj.latent_proj.0",
        &x,
        c.lq_conv_padding,
    )?;
    x = p.silu("PiD LQ first SiLU", &x)?;
    x = conv(
        p,
        &mut e.student,
        "lq_proj.latent_proj.2",
        &x,
        c.lq_conv_padding,
    )?;
    for i in 0..c.lq_num_res_blocks {
        let b = format!("lq_proj.latent_proj.{}.block", i + 3);
        let n = group_norm(p, &mut e.student, &format!("{b}.0"), &x)?;
        let n = p.silu(&format!("{b}/silu0"), &n)?;
        let a = conv(p, &mut e.student, &format!("{b}.2"), &n, c.lq_conv_padding)?;
        let n = group_norm(p, &mut e.student, &format!("{b}.3"), &a)?;
        let n = p.silu(&format!("{b}/silu3"), &n)?;
        let a = conv(p, &mut e.student, &format!("{b}.5"), &n, c.lq_conv_padding)?;
        x = p.binary(&format!("{b}/residual"), &x, &a)?;
    }
    let hidden = x.shape[3];
    let tokens = p.view(
        "PiD LQ patch token view",
        &x,
        vec![1, mul(h, w)?, hidden],
        true,
    )?;
    let mut feats = Vec::new();
    for i in 0..c.num_lq_outputs() {
        feats.push(p.linear(
            &mut e.student,
            &format!("lq_proj.output_heads.{i}"),
            &tokens,
            true,
            false,
        )?);
    }
    let pit = if c.pit_lq_inject {
        Some(p.linear(&mut e.student, "lq_proj.pit_head", &tokens, true, false)?)
    } else {
        None
    };
    Ok((feats, pit))
}
fn sigma_gate(
    p: &mut Program,
    w: &mut Sources,
    site: &str,
    x: &Tensor,
    lq: &Tensor,
    sigma: &Tensor,
) -> Result<Tensor> {
    let joined = p.concat(
        &format!("{site}/content-concat"),
        &[x, lq],
        x.shape.len() - 1,
    )?;
    let logit = p.linear(w, &format!("{site}.content_proj"), &joined, true, false)?;
    let logalpha = w.require(p, &format!("{site}.log_alpha"))?;
    let exp = p.unary(&format!("{site}/exp-logalpha"), &logalpha)?;
    let s = p.view(
        &format!("{site}/sigma-view"),
        sigma,
        vec![sigma.shape[0], 1, 1],
        false,
    )?;
    let offset = p.binary(&format!("{site}/sigma-offset"), &exp, &s)?;
    let logits = p.binary(&format!("{site}/logit-minus-offset"), &logit, &offset)?;
    let gate = p.unary(&format!("{site}/sigmoid"), &logits)?;
    p.gated(&format!("{site}/gated-residual"), x, &gate, lq)
}
fn rope_cache(
    p: &mut Program,
    e: &mut EngineSources,
    pixel: bool,
    h: i32,
    w: i32,
) -> Result<(Tensor, Tensor)> {
    let c = e.cfg.clone();
    let dim = if pixel {
        c.pixel_head_dim()
    } else {
        c.head_dim()
    };
    let key = if pixel {
        CacheKey::PixelRope {
            config: e.config_domain.clone(),
            h,
            w,
        }
    } else {
        CacheKey::ImageRope {
            config: e.config_domain.clone(),
            h,
            w,
        }
    };
    let v = e.caches.get_or_build(p, key, |p| {
        let n = mul(h, w)?;
        p.host("PiD 2D RoPE host invF64", (dim / 4) as u64 * 8);
        p.host("PiD 2D RoPE host cos/sin", elements(&[2, n, dim / 2])? * 4);
        Ok(vec![
            p.immediate("PiD 2D RoPE cached cos", vec![n, dim / 2], Dtype::Float32)?,
            p.immediate("PiD 2D RoPE cached sin", vec![n, dim / 2], Dtype::Float32)?,
        ])
    })?;
    Ok((v[0].clone(), v[1].clone()))
}
fn text_rope_cache(
    p: &mut Program,
    e: &mut EngineSources,
    tokens: i32,
) -> Result<(Tensor, Tensor)> {
    let key = CacheKey::TextRope {
        config: e.config_domain.clone(),
        tokens,
    };
    let dim = e.cfg.head_dim();
    let v = e.caches.get_or_build(p, key, |p| {
        p.host(
            "PiD text RoPE host cos/sin",
            elements(&[2, tokens, dim / 2])? * 4,
        );
        Ok(vec![
            p.immediate("PiD cached text cos", vec![tokens, dim / 2], Dtype::Float32)?,
            p.immediate("PiD cached text sin", vec![tokens, dim / 2], Dtype::Float32)?,
        ])
    })?;
    Ok((v[0].clone(), v[1].clone()))
}
fn adjacent_rope(
    p: &mut Program,
    site: &str,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor> {
    let sh = x.shape.clone();
    let half = sh[3] / 2;
    let pair = p.view(
        &format!("{site}/complex-pairs"),
        x,
        vec![sh[0], sh[1], sh[2], half, 2],
        true,
    )?;
    let lanes = p.split(&format!("{site}/actual adjacent-pair lanes"), &pair, 2, 4)?;
    let real_input = p.view(
        &format!("{site}/actual real lane reshape"),
        &lanes[0],
        vec![sh[0], sh[1], sh[2], half],
        false,
    )?;
    let imag_input = p.view(
        &format!("{site}/actual imag lane reshape"),
        &lanes[1],
        vec![sh[0], sh[1], sh[2], half],
        false,
    )?;
    let cos = p.view(
        &format!("{site}/cos-view"),
        cos,
        vec![1, 1, sh[2], half],
        false,
    )?;
    let sin = p.view(
        &format!("{site}/sin-view"),
        sin,
        vec![1, 1, sh[2], half],
        false,
    )?;
    let cos = p.cast(&format!("{site}/cos-dtype"), &cos, x.dtype)?;
    let sin = p.cast(&format!("{site}/sin-dtype"), &sin, x.dtype)?;
    let ac = p.binary(&format!("{site}/even-cos"), &real_input, &cos)?;
    let bs = p.binary(&format!("{site}/odd-sin"), &imag_input, &sin)?;
    let real = p.binary(&format!("{site}/real-subtract"), &ac, &bs)?;
    let bc = p.binary(&format!("{site}/odd-cos"), &imag_input, &cos)?;
    let as_ = p.binary(&format!("{site}/even-sin"), &real_input, &sin)?;
    let imag = p.binary(&format!("{site}/imag-add"), &bc, &as_)?;
    let native = p.workspace(
        site,
        "shared-qkv-adjacent-pair-rotation-layout",
        &[x, &cos, &sin],
    );
    let out = p.storage(
        &format!("{site}/adjacent-pair-interleave"),
        sh,
        x.dtype,
        union([&real, &imag]),
        false,
        false,
    )?;
    p.native_output(native, &out, NativeCall::Unbound)?;
    Ok(out)
}
struct HeadSpec<'a> {
    heads: i32,
    dim: i32,
    qn: &'a str,
    kn: &'a str,
    cos: &'a Tensor,
    sin: &'a Tensor,
}
fn prepare_heads(
    p: &mut Program,
    w: &mut Sources,
    site: &str,
    packed: &Tensor,
    args: HeadSpec<'_>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let HeadSpec {
        heads,
        dim,
        qn,
        kn,
        cos,
        sin,
    } = args;
    let parts = p.split(&format!("{site}/packed-qkv-split"), packed, 3, 2)?;
    let mut tensors = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        let t = p.view(
            &format!("{site}/heads-{i}"),
            part,
            vec![part.shape[0], part.shape[1], heads, dim],
            true,
        )?;
        tensors.push(p.transpose(&format!("{site}/head-major-{i}"), &t, &[0, 2, 1, 3])?);
    }
    let qn = w.require(p, qn)?;
    let kn = w.require(p, kn)?;
    let q = p.rms(&format!("{site}/q-F32-roundtrip-rms"), &tensors[0], &qn)?;
    let k = p.rms(&format!("{site}/k-F32-roundtrip-rms"), &tensors[1], &kn)?;
    let q = adjacent_rope(p, &format!("{site}/q-rope"), &q, cos, sin)?;
    let k = adjacent_rope(p, &format!("{site}/k-rope"), &k, cos, sin)?;
    Ok((q, k, tensors[2].clone()))
}
fn merge_heads(p: &mut Program, site: &str, x: &Tensor) -> Result<Tensor> {
    let sh = x.shape.clone();
    let t = p.transpose(&format!("{site}/token-major"), x, &[0, 2, 1, 3])?;
    p.view(site, &t, vec![sh[0], sh[2], mul(sh[1], sh[3])?], true)
}
fn feed_forward(p: &mut Program, w: &mut Sources, site: &str, x: &Tensor) -> Result<Tensor> {
    let a = p.linear(w, &format!("{site}.w1"), x, true, false)?;
    let a = p.silu(&format!("{site}/silu"), &a)?;
    let b = p.linear(w, &format!("{site}.w3"), x, true, false)?;
    let product = p.binary(&format!("{site}/swiglu-product"), &a, &b)?;
    p.linear(w, &format!("{site}.w2"), &product, true, false)
}
struct RopePair<'a> {
    ri: &'a (Tensor, Tensor),
    rt: &'a (Tensor, Tensor),
}
fn patch_block(
    p: &mut Program,
    e: &mut EngineSources,
    index: i32,
    x: &Tensor,
    y: &Tensor,
    condition: &Tensor,
    args: RopePair<'_>,
) -> Result<(Tensor, Tensor)> {
    let RopePair { ri, rt } = args;
    let b = format!("patch_blocks.{index}");
    let c = e.cfg.clone();
    let mx = p.linear(
        &mut e.student,
        &format!("{b}.adaLN_modulation_img.0"),
        condition,
        true,
        false,
    )?;
    let my = p.linear(
        &mut e.student,
        &format!("{b}.adaLN_modulation_txt.0"),
        condition,
        true,
        false,
    )?;
    let mx = p.split(&format!("{b}/img-modulation-split"), &mx, 6, 2)?;
    let my = p.split(&format!("{b}/txt-modulation-split"), &my, 6, 2)?;
    let nx = e.student.require(p, &format!("{b}.norm_x1.weight"))?;
    let ny = e.student.require(p, &format!("{b}.norm_y1.weight"))?;
    let nx = p.rms(&format!("{b}/x-rms1"), x, &nx)?;
    let ny = p.rms(&format!("{b}/y-rms1"), y, &ny)?;
    let nx = p.modulate(&format!("{b}/x-modulate1"), &nx, &mx[1], &mx[0])?;
    let ny = p.modulate(&format!("{b}/y-modulate1"), &ny, &my[1], &my[0])?;
    let qx = p.linear(&mut e.student, &format!("{b}.attn.qkv_x"), &nx, true, false)?;
    let qy = p.linear(&mut e.student, &format!("{b}.attn.qkv_y"), &ny, true, false)?;
    let img = prepare_heads(
        p,
        &mut e.student,
        &format!("{b}/img-qkv"),
        &qx,
        HeadSpec {
            heads: c.num_groups,
            dim: c.head_dim(),
            qn: &format!("{b}.attn.q_norm_x.weight"),
            kn: &format!("{b}.attn.k_norm_x.weight"),
            cos: &ri.0,
            sin: &ri.1,
        },
    )?;
    let txt = prepare_heads(
        p,
        &mut e.student,
        &format!("{b}/txt-qkv"),
        &qy,
        HeadSpec {
            heads: c.num_groups,
            dim: c.head_dim(),
            qn: &format!("{b}.attn.q_norm_y.weight"),
            kn: &format!("{b}.attn.k_norm_y.weight"),
            cos: &rt.0,
            sin: &rt.1,
        },
    )?;
    let q = p.concat(&format!("{b}/text-first-q"), &[&txt.0, &img.0], 2)?;
    let k = p.concat(&format!("{b}/text-first-k"), &[&txt.1, &img.1], 2)?;
    let v = p.concat(&format!("{b}/text-first-v"), &[&txt.2, &img.2], 2)?;
    let out = p.sdpa(&format!("{b}/flash-sdpa"), &q, &k, &v)?;
    let sy = slice(
        p,
        &format!("{b}/actual text-first split"),
        &out,
        2,
        0,
        y.shape[1],
    )?;
    let sx = slice(
        p,
        &format!("{b}/actual image-second split"),
        &out,
        2,
        y.shape[1],
        y.shape[1] + x.shape[1],
    )?;
    let sx = merge_heads(p, &format!("{b}/merge-img"), &sx)?;
    let sy = merge_heads(p, &format!("{b}/merge-txt"), &sy)?;
    let ax = p.linear(
        &mut e.student,
        &format!("{b}.attn.proj_x"),
        &sx,
        true,
        false,
    )?;
    let ay = p.linear(
        &mut e.student,
        &format!("{b}.attn.proj_y"),
        &sy,
        true,
        false,
    )?;
    let x = p.gated(&format!("{b}/x-attn-residual"), x, &mx[2], &ax)?;
    let y = p.gated(&format!("{b}/y-attn-residual"), y, &my[2], &ay)?;
    let nx = e.student.require(p, &format!("{b}.norm_x2.weight"))?;
    let ny = e.student.require(p, &format!("{b}.norm_y2.weight"))?;
    let nx = p.rms(&format!("{b}/x-rms2"), &x, &nx)?;
    let ny = p.rms(&format!("{b}/y-rms2"), &y, &ny)?;
    let nx = p.modulate(&format!("{b}/x-modulate2"), &nx, &mx[4], &mx[3])?;
    let ny = p.modulate(&format!("{b}/y-modulate2"), &ny, &my[4], &my[3])?;
    let fx = feed_forward(p, &mut e.student, &format!("{b}.mlp_x"), &nx)?;
    let x = p.gated(&format!("{b}/x-ff-residual"), &x, &mx[5], &fx)?;
    let fy = feed_forward(p, &mut e.student, &format!("{b}.mlp_y"), &ny)?;
    let y = p.gated(&format!("{b}/y-ff-residual"), &y, &my[5], &fy)?;
    Ok((x, y))
}

struct TokenGrid {
    batch: i32,
    tokens: i32,
}
fn pixel_block(
    p: &mut Program,
    e: &mut EngineSources,
    index: i32,
    x: &Tensor,
    s: &Tensor,
    rope: &(Tensor, Tensor),
    args: TokenGrid,
) -> Result<Tensor> {
    let TokenGrid { batch, tokens } = args;
    let c = e.cfg.clone();
    let b = format!("pixel_blocks.{index}");
    let sh = x.shape.clone();
    let cond = p.linear(
        &mut e.student,
        &format!("{b}.adaLN_modulation.0"),
        s,
        true,
        false,
    )?;
    let cond = p.view(
        &format!("{b}/modulation-shape"),
        &cond,
        vec![sh[0], sh[1], mul(6, c.pixel_hidden_size)?],
        true,
    )?;
    let m = p.split(&format!("{b}/modulation-split"), &cond, 6, 2)?;
    let nw = e.student.require(p, &format!("{b}.norm1.weight"))?;
    let nx = p.rms(&format!("{b}/rms1"), x, &nw)?;
    let nx = p.modulate(&format!("{b}/modulate1"), &nx, &m[1], &m[0])?;
    let flat = p.view(
        &format!("{b}/flatten-pixels"),
        &nx,
        vec![sh[0], mul(sh[1], c.pixel_hidden_size)?],
        true,
    )?;
    let compact = p.linear(
        &mut e.student,
        &format!("{b}.compress_to_attn"),
        &flat,
        true,
        false,
    )?;
    let compact = p.view(
        &format!("{b}/attention-token-shape"),
        &compact,
        vec![batch, tokens, c.pixel_attn_hidden_size],
        true,
    )?;
    let qkv = p.linear(
        &mut e.student,
        &format!("{b}.attn.qkv"),
        &compact,
        true,
        false,
    )?;
    let h = prepare_heads(
        p,
        &mut e.student,
        &format!("{b}/qkv"),
        &qkv,
        HeadSpec {
            heads: c.pixel_num_groups,
            dim: c.pixel_head_dim(),
            qn: &format!("{b}.attn.q_norm.weight"),
            kn: &format!("{b}.attn.k_norm.weight"),
            cos: &rope.0,
            sin: &rope.1,
        },
    )?;
    let attn = p.sdpa(&format!("{b}/pixel-flash-sdpa"), &h.0, &h.1, &h.2)?;
    let attn = merge_heads(p, &format!("{b}/attention-merge"), &attn)?;
    let attn = p.linear(
        &mut e.student,
        &format!("{b}.attn.proj"),
        &attn,
        true,
        false,
    )?;
    let attn = p.view(
        &format!("{b}/flat-attention"),
        &attn,
        vec![mul(batch, tokens)?, c.pixel_attn_hidden_size],
        true,
    )?;
    let expanded = p.linear(
        &mut e.student,
        &format!("{b}.expand_from_attn"),
        &attn,
        true,
        false,
    )?;
    let expanded = p.view(
        &format!("{b}/pixel-expand-shape"),
        &expanded,
        sh.clone(),
        true,
    )?;
    let x = p.gated(&format!("{b}/attention-residual"), x, &m[2], &expanded)?;
    let nw = e.student.require(p, &format!("{b}.norm2.weight"))?;
    let nx = p.rms(&format!("{b}/rms2"), &x, &nw)?;
    let nx = p.modulate(&format!("{b}/modulate2"), &nx, &m[4], &m[3])?;
    let fc1 = p.linear(&mut e.student, &format!("{b}.mlp.fc1"), &nx, true, false)?;
    let gelu = p.gelu(&format!("{b}/gelu-exact"), &fc1, false)?;
    let fc2 = p.linear(&mut e.student, &format!("{b}.mlp.fc2"), &gelu, true, false)?;
    p.gated(&format!("{b}/ff-residual"), &x, &m[5], &fc2)
}
fn net_forward(
    p: &mut Program,
    e: &mut EngineSources,
    x: &Tensor,
    t: &Tensor,
    caption: &Tensor,
    lq: &Tensor,
    sigma: &Tensor,
) -> Result<Tensor> {
    let c = e.cfg.clone();
    let sh = x.shape.clone();
    let (batch, h, w) = (sh[0], sh[2], sh[3]);
    let hs = h / c.patch_size;
    let ws = w / c.patch_size;
    let tokens = mul(hs, ws)?;
    let p2 = mul(c.patch_size, c.patch_size)?;
    let (features, pit_feature) = lq_features(p, e, lq, hs, ws)?;
    let unfolded = p.view(
        "PiD unfold source split",
        x,
        vec![batch, c.in_channels, hs, c.patch_size, ws, c.patch_size],
        false,
    )?;
    let unfolded = p.transpose(
        "PiD unfold source transpose",
        &unfolded,
        &[0, 2, 4, 1, 3, 5],
    )?;
    let patches = p.view(
        "PiD unfold flattened token patches",
        &unfolded,
        vec![batch, tokens, mul(c.in_channels, p2)?],
        true,
    )?;
    p.host("PiD timestep frequencies host F32", 128 * 4);
    let frequencies = p.immediate("PiD timestep frequencies", vec![1, 128], Dtype::Float32)?;
    let time = p.view("PiD timestep column", t, vec![batch, 1], false)?;
    let args = p.binary("PiD timestep arguments", &time, &frequencies)?;
    let cos = p.unary("PiD timestep cos", &args)?;
    let sin = p.unary("PiD timestep sin", &args)?;
    let emb = p.concat("PiD timestep embedding", &[&cos, &sin], 1)?;
    let emb = p.linear(&mut e.student, "t_embedder.mlp.0", &emb, true, false)?;
    let emb = p.silu("PiD timestep SiLU", &emb)?;
    let emb = p.linear(&mut e.student, "t_embedder.mlp.2", &emb, true, false)?;
    let emb = p.view(
        "PiD timestep conditioning view",
        &emb,
        vec![batch, 1, c.hidden_size],
        false,
    )?;
    let txt = caption.shape[1].min(c.txt_max_length);
    let prefix = slice(
        p,
        "PiD caption actual prefix_axis1 split",
        caption,
        1,
        0,
        txt,
    )?;
    let mut y = p.linear(&mut e.student, "y_embedder.proj", &prefix, true, false)?;
    if let Some(norm) = e.student.get(p, "y_embedder.norm.weight")? {
        y = p.rms("PiD caption embedding norm", &y, &norm)?;
    }
    let pos = e.student.require(p, "y_pos_embedding")?;
    let pos = slice(
        p,
        "PiD positional actual prefix_axis1 split",
        &pos,
        1,
        0,
        txt,
    )?;
    let pos = p.cast("PiD caption positional dtype", &pos, y.dtype)?;
    y = p.binary("PiD caption positional add", &y, &pos)?;
    let condition = p.silu("PiD patch conditioning SiLU", &emb)?;
    let ri = rope_cache(p, e, false, hs, ws)?;
    let rt = text_rope_cache(p, e, txt)?;
    let mut main = p.linear(&mut e.student, "s_embedder.proj", &patches, true, false)?;
    if let Some(norm) = e.student.get(p, "s_embedder.norm.weight")? {
        main = p.rms("PiD patch embedding norm", &main, &norm)?;
    }
    for index in 0..c.patch_depth {
        if c.lq_interval <= 1 || index % c.lq_interval == 0 {
            let idx = if c.lq_interval > 1 {
                index / c.lq_interval
            } else {
                index
            };
            main = sigma_gate(
                p,
                &mut e.student,
                &format!("lq_proj.gate_modules.{idx}"),
                &main,
                &features[idx as usize],
                sigma,
            )?;
        }
        let out = patch_block(
            p,
            e,
            index,
            &main,
            &y,
            &condition,
            RopePair { ri: &ri, rt: &rt },
        )?;
        main = out.0;
        y = out.1;
    }
    let s = p.binary("PiD timestep plus patch conditioning", &emb, &main)?;
    let mut s = p.silu("PiD pixel conditioning SiLU", &s)?;
    if let Some(feature) = pit_feature {
        s = sigma_gate(p, &mut e.student, "pit_lq_gate", &s, &feature, sigma)?;
    }
    let s = p.view(
        "PiD pixel conditioning flatten",
        &s,
        vec![mul(batch, tokens)?, c.hidden_size],
        true,
    )?;
    let xt = p.transpose("PiD pixel input NHWC", x, &[0, 2, 3, 1])?;
    let xp = p.linear(&mut e.student, "pixel_embedder.proj", &xt, true, false)?;
    let key = CacheKey::PixelPosition {
        config: e.config_domain.clone(),
        h,
        w,
    };
    let dim = c.pixel_hidden_size;
    let values = e.caches.get_or_build(p, key, |p| {
        p.host("PiD pixel-position host omega F64", (dim / 4) as u64 * 8);
        p.host(
            "PiD pixel-position host table F32",
            elements(&[h, w, dim])? * 4,
        );
        Ok(vec![p.immediate(
            "PiD cached RAW F32 pixel position",
            vec![mul(h, w)?, dim],
            Dtype::Float32,
        )?])
    })?;
    let pos = p.view(
        "PiD pixel-position view",
        &values[0],
        vec![1, h, w, dim],
        false,
    )?;
    let pos = p.cast("PiD per-call pixel-position dtype", &pos, xp.dtype)?;
    let xp = p.binary("PiD pixel positional add", &xp, &pos)?;
    let xp = p.view(
        "PiD pixel patch split",
        &xp,
        vec![batch, hs, c.patch_size, ws, c.patch_size, dim],
        false,
    )?;
    let xp = p.transpose("PiD pixel patch transpose", &xp, &[0, 1, 3, 2, 4, 5])?;
    let mut xp = p.view(
        "PiD per-patch pixels",
        &xp,
        vec![mul(batch, tokens)?, p2, dim],
        true,
    )?;
    let rp = rope_cache(p, e, true, hs, ws)?;
    for index in 0..c.pixel_depth {
        xp = pixel_block(p, e, index, &xp, &s, &rp, TokenGrid { batch, tokens })?;
    }
    let norm = e.student.require(p, "final_layer.norm.weight")?;
    xp = p.rms("PiD final pixel RMS", &xp, &norm)?;
    xp = p.linear(&mut e.student, "final_layer.linear", &xp, true, false)?;
    let xp = p.view(
        "PiD final patch shape",
        &xp,
        vec![batch, tokens, p2, c.in_channels],
        false,
    )?;
    let xp = p.transpose("PiD final patch channel-major", &xp, &[0, 3, 2, 1])?;
    let xp = p.view(
        "PiD fold split",
        &xp,
        vec![batch, c.in_channels, c.patch_size, c.patch_size, hs, ws],
        true,
    )?;
    let xp = p.transpose("PiD fold transpose", &xp, &[0, 1, 4, 2, 5, 3])?;
    p.view(
        "PiD fold pixel output",
        &xp,
        vec![batch, c.in_channels, h, w],
        true,
    )
}

fn slice(
    p: &mut Program,
    site: &str,
    x: &Tensor,
    axis: usize,
    start: i32,
    end: i32,
) -> Result<Tensor> {
    let op = p.attempt(&format!("{site}/axis-{axis}/[{start},{end})"), &[x]);
    if axis >= x.shape.len() || start < 0 || end < start || end > x.shape[axis] {
        return Err(Error::Msg("PiD native strided tile boundary".into()));
    }
    let mut out = x.clone();
    out.shape[axis] = end - start;
    out.value = p.owner_value(
        out.shape.clone(),
        out.dtype,
        native_allocations::ValueRule::Split(x.value, axis, start),
    );
    p.complete(op, &out);
    Ok(out) // split_sections aliases the WHOLE x/LQ parent, never a tile payload copy
}
struct Placement {
    axis: usize,
    start: i32,
    extent: i32,
    total: i32,
}
fn accumulate(
    p: &mut Program,
    site: &str,
    previous: Option<Tensor>,
    x: &Tensor,
    args: Placement,
) -> Result<Tensor> {
    let Placement {
        axis,
        start,
        extent,
        total,
    } = args;
    if x.shape[axis] != extent {
        return Err(Error::Msg("PiD actual accumulate extent gate".into()));
    }
    let placed = if start == 0 && extent == total {
        x.clone()
    } else {
        let mut shape = x.shape.clone();
        shape[axis] = total;
        p.storage(
            &format!("{site}/pad-along-ONE-axis"),
            shape,
            x.dtype,
            x.roots.clone(),
            false,
            false,
        )?
    };
    if let Some(previous) = previous {
        p.binary(&format!("{site}/accumulate-add"), &previous, &placed)
    } else {
        Ok(placed)
    }
}
struct TiledVelocityRequest<'a> {
    time: &'a Tensor,
    caption: &'a Tensor,
    lq: &'a Tensor,
    sigma: &'a Tensor,
    edge: i32,
    overlap: i32,
    held: &'a BTreeSet<Slot>,
    serial: u32,
    step: usize,
}
fn tiled_velocity(
    p: &mut Program,
    e: &mut EngineSources,
    x: &Tensor,
    args: TiledVelocityRequest<'_>,
) -> Result<Tensor> {
    let TiledVelocityRequest {
        time,
        caption,
        lq,
        sigma,
        edge,
        overlap,
        held,
        serial,
        step,
    } = args;
    let (h, w) = (x.shape[2], x.shape[3]);
    let align = (h / lq.shape[2]).max(1);
    let tiles = mlx_gen_pid::tiling::plan_tiles(h, w, edge, overlap, align);
    if tiles.len() == 1 {
        return net_forward(p, e, x, time, caption, lq, sigma);
    }
    // Exact host loop/data owners; no whole-image tensor gets resized to the tile here.
    p.hold_host("PiD tiling host sum_y/sum_x", (h as u64 + w as u64) * 4);
    p.host(
        "PiD tiling host plan metadata",
        tiles.len() as u64 * std::mem::size_of::<mlx_gen_pid::tiling::SpatialTile>() as u64,
    );
    let mut acc: Option<Tensor> = None;
    let mut row: Option<Tensor> = None;
    for (index, tile) in tiles.iter().copied().enumerate() {
        let xt = slice(p, "PiD x tile Y (full-x ancestry)", x, 2, tile.y0, tile.y1)?;
        let xt = slice(
            p,
            "PiD x tile X (full-x ancestry)",
            &xt,
            3,
            tile.x0,
            tile.x1,
        )?;
        let lt = slice(p, "PiD LQ tile Y", lq, 2, tile.y0 / align, tile.y1 / align)?;
        let lt = slice(p, "PiD LQ tile X", &lt, 3, tile.x0 / align, tile.x1 / align)?;
        let velocity = net_forward(p, e, &xt, time, caption, &lt, sigma)?;
        let th = tile.y1 - tile.y0;
        let tw = tile.x1 - tile.x0;
        let key = CacheKey::Feather {
            h: th,
            w: tw,
            top: tile.y0 > 0,
            bottom: tile.y1 < h,
            left: tile.x0 > 0,
            right: tile.x1 < w,
            overlap,
        };
        let values = e.caches.get_or_build(p, key, |p| {
            p.host(
                "PiD feather host ry/rx/output",
                (th as u64 + tw as u64 + th as u64 * tw as u64) * 4,
            );
            Ok(vec![p.immediate(
                "PiD cached RAW F32 feather",
                vec![1, 1, th, tw],
                Dtype::Float32,
            )?])
        })?;
        let feather = p.cast("PiD per-call feather dtype", &values[0], velocity.dtype)?;
        let weighted = p.binary("PiD feathered tile", &velocity, &feather)?;
        row = Some(accumulate(
            p,
            "PiD row strip",
            row.take(),
            &weighted,
            Placement {
                axis: 3,
                start: tile.x0,
                extent: tw,
                total: w,
            },
        )?);
        let last_in_row = tiles.get(index + 1).is_none_or(|next| next.y0 != tile.y0);
        if last_in_row {
            let strip = row
                .take()
                .ok_or_else(|| Error::Msg("PiD source missing completed row".into()))?;
            acc = Some(accumulate(
                p,
                "PiD full image accumulator",
                acc.take(),
                &strip,
                Placement {
                    axis: 2,
                    start: tile.y0,
                    extent: th,
                    total: h,
                },
            )?);
        }
        let mut resident = held.clone();
        resident.extend(e.caches.roots());
        if let Some(a) = &acc {
            resident.extend(a.roots.iter().copied());
        }
        if let Some(a) = &row {
            resident.extend(a.roots.iter().copied());
        }
        // Actual eval([row,acc]) frontier. This is NOT native Data/cache/process release evidence.
        p.window(
            IosPhase::Decode,
            "PiD tiled velocity ordered tile/eval frontier",
            &resident,
            serial,
            Some(step),
            Some(tile),
        )?;
        let outputs = row.iter().chain(acc.iter()).collect::<Vec<_>>();
        p.evaluated_graph_frontier("PiD tile row/full-acc source eval", &outputs)?;
    }
    let acc = acc.ok_or_else(|| Error::Msg("PiD source empty tile grid".into()))?;
    let result = full_normalize(p, &acc, h, w)?;
    let mut resident = held.clone();
    resident.extend(e.caches.roots());
    resident.extend(acc.roots.iter().copied());
    p.window(
        IosPhase::Decode,
        "PiD full normalizer and divide",
        &resident,
        serial,
        Some(step),
        None,
    )?;
    p.end_host_source_scope("PiD tiling host sum_y/sum_x");
    Ok(result)
}

fn full_normalize(p: &mut Program, acc: &Tensor, h: i32, w: i32) -> Result<Tensor> {
    let norm_y = p.immediate("PiD normalizer Y", vec![1, 1, h, 1], Dtype::Float32)?;
    let norm_x = p.immediate("PiD normalizer X", vec![1, 1, 1, w], Dtype::Float32)?;
    let wsum = p.binary("PiD FULL HxW weight sum", &norm_y, &norm_x)?;
    let wsum = p.cast("PiD normalizer accumulator dtype", &wsum, acc.dtype)?;
    p.binary("PiD FULL accumulator divide", acc, &wsum)
}

#[derive(Clone, Debug)]
enum TileSelection {
    Whole,
    Tiled { edge: i32, overlap: i32 },
}
#[derive(Clone, Debug)]
enum Condition {
    EngineUsed(bool),
    /// A native resource-policy predicate, not a memory-proof coefficient. No OS query here.
    NativeMinimumTileGuardPasses(bool),
    AutoPlannerReturns {
        edge: Option<i32>,
    },
    ExplicitTile {
        edge: i32,
        overlap: i32,
    },
    InputLatentDtype(Dtype),
    ZModelForward(bool),
    CaptureControl {
        requested: Option<f32>,
    },
    RawMaterializeBatchCompletes {
        ordinal: usize,
        completes: bool,
    },
}
fn selections(
    c: &Controls,
    cfg: &PidConfig,
    route: &Route,
) -> Result<Vec<(TileSelection, Vec<Condition>)>> {
    let h = mul(route.height as i32, cfg.sr_scale)?;
    let w = mul(route.width as i32, cfg.sr_scale)?;
    if let Some((edge, overlap)) = c.selected_tile {
        // Actual mint validates explicit geometry AFTER net+caption construction. This function
        // only carries it; run_source_case performs the validation at that reached frontier.
        return Ok(vec![(
            if h > edge || w > edge {
                TileSelection::Tiled { edge, overlap }
            } else {
                TileSelection::Whole
            },
            vec![Condition::ExplicitTile { edge, overlap }],
        )]);
    }
    let mut cases = Vec::new();
    if h.max(w) <= mlx_gen_pid::budget::WATCHDOG_SAFE_EDGE {
        cases.push((
            TileSelection::Whole,
            vec![Condition::AutoPlannerReturns { edge: None }],
        ));
    }
    // Enumerate the ACTUAL bounded policy's possible outputs, not its empirical estimate as a
    // storage bound, a sorted winner, or a fabricated safe budget. Runtime conformance binds the
    // AutoPlannerReturns predicate against the unchanged native planner and injected safe_gib.
    for edge in mlx_gen_pid::budget::tile_edge_candidates() {
        if edge <= mlx_gen_pid::budget::PREFERRED_TILE_EDGE && edge < h.max(w) {
            cases.push((
                TileSelection::Tiled {
                    edge,
                    overlap: mlx_gen_pid::budget::DEFAULT_TILE_OVERLAP,
                },
                vec![Condition::AutoPlannerReturns { edge: Some(edge) }],
            ));
        }
    }
    Ok(cases)
}
fn draw_noise_sources(
    p: &mut Program,
    shape: Vec<i32>,
    sampler: &SamplerConfig,
) -> Result<(Tensor, Vec<Tensor>)> {
    let noise = p.storage(
        "PiD initial FULL noise factory",
        shape.clone(),
        Dtype::Float32,
        BTreeSet::new(),
        false,
        false,
    )?;
    let native = p.workspace(
        "PiD RNG initial key/split/normal",
        "native-rng-keys-splits-and-normal-workspace",
        &[&noise],
    );
    p.native_output(native, &noise, NativeCall::NormalNoLocScale)?;
    let num_eps = if sampler.sample_type == SampleType::Sde {
        sampler
            .student_t_list
            .iter()
            .skip(1)
            .filter(|&&t| t > 0.0)
            .count()
    } else {
        0
    };
    let mut eps = Vec::new();
    for draw in 0..num_eps {
        let value = p.storage(
            &format!("PiD FULL epsilon draw-{draw}"),
            shape.clone(),
            Dtype::Float32,
            BTreeSet::new(),
            false,
            false,
        )?;
        let native = p.workspace(
            &format!("PiD RNG epsilon-{draw}"),
            "native-rng-keys-splits-and-normal-workspace",
            &[&value],
        );
        p.native_output(native, &value, NativeCall::NormalNoLocScale)?;
        eps.push(value);
    }
    p.host(
        "PiD host epsilon Vec handles",
        num_eps as u64 * std::mem::size_of::<mlx_rs::Array>() as u64,
    );
    Ok((noise, eps))
}

fn sample_decode(
    p: &mut Program,
    e: &mut EngineSources,
    caption: &Tensor,
    lq_input: &Tensor,
    selection: &TileSelection,
    serial: u32,
    sampler: &SamplerConfig,
) -> Result<Tensor> {
    let c = e.cfg.clone();
    let (h, w) = (
        mul(
            lq_input.shape[2],
            mul(c.latent_spatial_down_factor, c.sr_scale)?,
        )?,
        mul(
            lq_input.shape[3],
            mul(c.latent_spatial_down_factor, c.sr_scale)?,
        )?,
    );
    let lq = p.cast("PiD prep BF16 latent", lq_input, Dtype::Bfloat16)?;
    p.host("PiD prep sigma host", 4);
    let sigma = p.immediate("PiD prep sigma", vec![1], Dtype::Float32)?;
    let (noise, eps) = draw_noise_sources(p, vec![1, 3, h, w], sampler)?;
    let mut held = e.raw_engine.clone();
    held.extend(caption.roots.iter().copied());
    held.extend(lq.roots.iter().copied());
    held.extend(sigma.roots.iter().copied());
    held.extend(noise.roots.iter().copied());
    for value in &eps {
        held.extend(value.roots.iter().copied());
    }
    p.window(
        IosPhase::Decode,
        "PiD prep+ALL FULL RNG draws",
        &held,
        serial,
        None,
        None,
    )?;
    let mut x = noise.clone();
    let mut ei = 0usize;
    for step in 0..sampler.student_t_list.len().saturating_sub(1) {
        let next = sampler.student_t_list[step + 1];
        p.host("PiD scaled timestep host", 4);
        let time = p.immediate("PiD scaled timestep", vec![1], Dtype::Float32)?;
        let mut step_held = held.clone();
        step_held.extend(x.roots.iter().copied());
        step_held.extend(e.caches.roots());
        let velocity = match selection {
            TileSelection::Whole => net_forward(p, e, &x, &time, caption, &lq, &sigma)?,
            TileSelection::Tiled { edge, overlap } => tiled_velocity(
                p,
                e,
                &x,
                TiledVelocityRequest {
                    time: &time,
                    caption,
                    lq: &lq,
                    sigma: &sigma,
                    edge: *edge,
                    overlap: *overlap,
                    held: &step_held,
                    serial,
                    step,
                },
            )?,
        };
        if next > 0.0 && sampler.sample_type == SampleType::Ode {
            let delta = p.scalar("PiD ODE dt", Dtype::Float32)?;
            let update = p.binary("PiD ODE velocity product", &velocity, &delta)?;
            x = p.binary("PiD ODE full-state update", &x, &update)?;
        } else {
            let t = p.scalar("PiD velocity-to-x0 t", Dtype::Float32)?;
            let product = p.binary("PiD FULL t*velocity", &velocity, &t)?;
            let x0 = p.binary("PiD FULL velocity-to-x0", &x, &product)?;
            if next > 0.0 {
                let left = p.scalar("PiD SDE one-minus-next", Dtype::Float32)?;
                let right = p.scalar("PiD SDE next", Dtype::Float32)?;
                let x0 = p.binary("PiD FULL SDE x0 product", &x0, &left)?;
                let epsilon = eps
                    .get(ei)
                    .ok_or_else(|| Error::Msg("PiD native epsilon-count gate".into()))?;
                ei += 1;
                let fresh = p.binary("PiD FULL SDE epsilon product", epsilon, &right)?;
                x = p.binary("PiD FULL SDE state update", &x0, &fresh)?;
            } else {
                x = x0;
            }
        }
        step_held.extend(e.caches.roots());
        // This source frontier is the real run_inner eval([x]). It is NOT treated as proof of
        // native Data release, allocator reuse, process demand, or physical headroom.
        p.window(
            IosPhase::Decode,
            "PiD sampler FULL state/eval frontier",
            &step_held,
            serial,
            Some(step),
            None,
        )?;
        p.evaluated_graph_frontier("PiD per-step full-state source eval", &[&x])?;
    }
    let lower = p.scalar("PiD clamp lower", Dtype::Float32)?;
    let upper = p.scalar("PiD clamp upper", Dtype::Float32)?;
    let out = p.storage(
        "PiD FULL output clamp",
        x.shape.clone(),
        x.dtype,
        union([&x, &lower, &upper]),
        false,
        false,
    )?;
    held.extend(e.caches.roots());
    p.window(
        IosPhase::Decode,
        "PiD output clamp",
        &held,
        serial,
        None,
        None,
    )?;
    Ok(out)
}

fn closure(prefix: &Prefix, roots: &BTreeSet<Slot>) -> Result<BTreeSet<Slot>> {
    let mut todo = roots.iter().copied().collect::<Vec<_>>();
    let mut seen = BTreeSet::new();
    while let Some(slot) = todo.pop() {
        if !seen.insert(slot) {
            continue;
        }
        let capacity = prefix
            .capacities
            .get(slot.0)
            .ok_or_else(|| Error::Msg("PiD source slot missing".into()))?;
        if capacity.graph_attached {
            todo.extend(capacity.parents.iter().copied());
        }
    }
    Ok(seen)
}
fn capacity_sum(prefix: &Prefix, roots: &BTreeSet<Slot>) -> Result<u64> {
    roots.iter().try_fold(0u64, |sum, slot| {
        sum.checked_add(mlx_gen::ios_quantum::allocated_bytes(
            prefix.capacities[slot.0].bytes,
        ))
        .ok_or_else(|| Error::Msg("PiD positive source union overflow".into()))
    })
}
fn source_window_price(
    prefix: &Prefix,
    window: &SourceWindow,
    base_resident: u64,
    base_held: u64,
    class: &str,
) -> Result<IosWindowPrice> {
    let resident = window.resident_roots.clone();
    let created = window.created_roots.clone();
    // Positive disjoint partitions of SAME source slots, not upper-minus-upper/common labels.
    let current = created
        .difference(&resident)
        .copied()
        .collect::<BTreeSet<_>>();
    Ok(IosWindowPrice {phase:window.phase,class:Some(format!("{class}/source-op-{}-{}",window.operation_start,window.operation_end)),
        label:window.site.clone(),resident_bytes:base_resident.checked_add(capacity_sum(prefix,&resident)?).ok_or_else(||Error::Msg("PiD source resident overflow".into()))?,
        held_bytes:window.host.iter().try_fold(base_held,|sum,demand|match demand {
            HostDemand::Bytes {bytes,..}=>sum.checked_add(*bytes).ok_or_else(||Error::Msg("PiD coincident concrete HOST source overflow".into())),
            HostDemand::Symbolic {..}=>Ok(sum), // preserved explicit UNBOUND term in source prefix, NEVER asserted zero physical cost
        })?,scratch_bytes:capacity_sum(prefix,&current)?.checked_add(window.requests.known_extra_requested)
            .ok_or_else(||Error::Msg("PiD conditional extra requested allocation overflow".into()))?,basis:None,
        owner_domain:"z-image/pid/source-slots-plus-DOMAIN-TAGGED-conditional-components/INCOMPLETE-NOT-native-Data-or-physical-evidence".into()})
}

#[derive(Debug)]
pub(super) struct PidFailure {
    error: Error,
    adapter_prefix: AdapterSourcePrefix,
    source: Arc<Prefix>,
    committed: Vec<IosWindowPrice>,
    attempted_case_id: Option<u64>,
    adapter_attempted_case_id: Option<u64>,
}
impl PidFailure {
    fn into_legacy_error(self) -> Error {
        mlx_gen::ios_quantum::diagnostic_metadata(
            "z_image",
            "pid-source-pricing-abort",
            &[
                ("schema", 1),
                ("case_present", u64::from(self.attempted_case_id.is_some())),
                ("case_id", self.attempted_case_id.unwrap_or(0)),
                (
                    "adapter_case_present",
                    u64::from(self.adapter_attempted_case_id.is_some()),
                ),
                (
                    "adapter_case_id",
                    self.adapter_attempted_case_id.unwrap_or(0),
                ),
                ("adapter_work", self.adapter_prefix.work.len() as u64),
                ("pid_work", self.source.operations.len() as u64),
                ("pid_source_slots", self.source.capacities.len() as u64),
                ("committed_windows", self.committed.len() as u64),
            ],
        );
        for (ordinal, window) in self.committed.iter().enumerate() {
            mlx_gen::ios_quantum::diagnostic_metadata(
                "z_image",
                "pid-source-pricing-prior-window",
                &[
                    ("ordinal", ordinal as u64),
                    ("resident_bytes", window.resident_bytes),
                    ("held_bytes", window.held_bytes),
                    ("scratch_bytes", window.scratch_bytes),
                ],
            );
        }
        self.error
    }
}
struct SourceSummary {
    case_id: u64,
    source_digest: String,
    condition_count: usize,
    operations: usize,
    source_slots: usize,
    workspace_obligations: usize,
    evaluated_native_workspaces: usize,
    requested_components: usize,
    native_unknowns: usize,
    host_symbolic_owners: usize,
}
pub(super) struct PidReport {
    pricing: IosPricing,
    /// Small summaries only. The observer receives EVERY full ordered case prefix before commit;
    /// do not eagerly retain a Cartesian collection of all graph tapes in the admission process.
    source_cases: Vec<SourceSummary>,
    refused_cases: u64,
    complete_cases: u64,
}

#[derive(Clone)]
struct BaseContext {
    load_resident: u64,
    load_held: u64,
    mint_resident: u64,
    mint_held: u64,
    decode_resident: u64,
    decode_held: u64,
}
fn base_context(
    model: &Model,
    route: &Route,
    sampling: &ResolvedFlowControls,
    quanta: &[Priced],
    forward: bool,
    adapter_case: &AdapterSourceCase,
    adapter_ledger: &super::adapter_sources::Ledger,
) -> Result<BaseContext> {
    let f = &model.facts;
    let staged = route.execution.stage_residency;
    let te = f.te.resident(|key| model.te_constructor_keys.contains(key));
    let heavy = model
        .dit_resident()
        .checked_add(model.control_resident())
        .ok_or_else(|| Error::Msg("PiD base heavy source overflow".into()))?;
    let structural = heavy
        .checked_add(if staged { 0 } else { te })
        .ok_or_else(|| Error::Msg("PiD base structural overflow".into()))?;
    let last_load = quanta
        .iter()
        .rev()
        .find(|q| q.window.phase == IosPhase::Load)
        .ok_or_else(|| Error::Msg("PiD source missing base load operation".into()))?;
    let mut load = super::QuantizationCursor::new(super::QuantizationHistory::RequestFirst, false);
    let (lr, lc, _) = load
        .window_with_vae(
            f,
            IosPhase::Load,
            mlx_gen::adapters::loader::ADAPTER_LOAD_QUANTUM,
            crate::ios_pricing::WindowPoint {
                point: 0,
                structural,
                staged,
            },
            last_load.source_vae.as_deref(),
        )?
        .ok_or_else(|| Error::Msg("PiD base postVAE source operation".into()))?;
    let adapters = adapter_ledger.held()?;
    let cap = crate::quantum::ab(
        if route.mode.img2img() { 4 } else { 2 },
        &[
            f.caption_tokens.min(f.window_tokens) as u64,
            f.te_cfg.hidden_size as u64,
        ],
    );
    let history = if forward {
        super::QuantizationHistory::RequestFirst
    } else {
        super::QuantizationHistory::NoModelForward
    };
    let mut cursor = super::QuantizationCursor::new(history, false);
    for q in quanta.iter().filter(|q| q.window.phase != IosPhase::Decode) {
        cursor.window_with_vae(
            f,
            q.window.phase,
            q.label,
            crate::ios_pricing::WindowPoint {
                point: q.source_point,
                structural: q.source_structural,
                staged,
            },
            q.source_vae.as_deref(),
        )?;
    }
    // This is a PiD source operation, NOT a native VAE decoder consumption. Its label cannot mark
    // unused VAE decoder weights warm. SAME VAE constructor/remap world remains attached.
    let (dr, dc, _) = cursor
        .window_with_vae(
            f,
            IosPhase::Decode,
            "PiD source decode",
            crate::ios_pricing::WindowPoint {
                point: 0,
                structural: if staged { 0 } else { structural },
                staged,
            },
            last_load.source_vae.as_deref(),
        )?
        .ok_or_else(|| Error::Msg("PiD base decode ownership operation".into()))?;
    let deferred =
        super::adapter_sources::pid_decode_source_owners(adapter_case, adapter_ledger, forward)?;
    Ok(BaseContext {
        load_resident: lr
            .checked_add(adapters)
            .ok_or_else(|| Error::Msg("PiD postadapter source overflow".into()))?,
        load_held: last_load
            .source_held
            .checked_add(lc)
            .ok_or_else(|| Error::Msg("PiD base load-created overflow".into()))?,
        mint_resident: lr
            .checked_add(adapters)
            .ok_or_else(|| Error::Msg("PiD mint structural overflow".into()))?,
        // Native model.rs mints BEFORE VAE init/latent creation, so do not reuse a later
        // DIT_EMBED held subtotal (which already owns those future tensors).
        mint_held: cap
            .checked_add(sampling.host_schedule_bytes()?)
            .ok_or_else(|| Error::Msg("PiD mint caption/schedule overflow".into()))?,
        decode_resident: dr
            .checked_add(adapters)
            .and_then(|v| v.checked_add(deferred))
            .ok_or_else(|| Error::Msg("PiD decode base source overflow".into()))?,
        decode_held: dc,
    })
}
fn add_source_owner(
    pricing: &mut IosPricing,
    prefix: &Prefix,
    roots: &BTreeSet<Slot>,
    predicate: impl Fn(&IosWindowPrice) -> bool,
) -> Result<()> {
    let roots = closure(prefix, roots)?;
    let bytes = capacity_sum(prefix, &roots)?;
    for window in &mut pricing.windows {
        if predicate(window) {
            window.resident_bytes = window
                .resident_bytes
                .checked_add(bytes)
                .ok_or_else(|| Error::Msg("PiD retained source union overflow".into()))?;
        }
    }
    Ok(())
}
fn pid_source_prices(
    prefix: &Prefix,
    base: &BaseContext,
    route: &Route,
    scale: i32,
    class: &str,
) -> Result<IosPricing> {
    let mut pricing = IosPricing::default();
    for window in &prefix.windows {
        let (resident, held) = match window.phase {
            IosPhase::Load => (base.load_resident, base.load_held),
            IosPhase::Denoise => (base.mint_resident, base.mint_held),
            IosPhase::Decode => (base.decode_resident, base.decode_held),
            _ => return Err(Error::Msg("unexpected PiD source factory phase".into())),
        };
        let mut price = source_window_price(prefix, window, resident, held, class)?;
        if window.phase == IosPhase::Decode {
            let remaining = u64::from(route.count - window.serial_completed)
                .checked_mul(
                    elements(&[1, 16, route.height as i32 / 8, route.width as i32 / 8])? * 4,
                )
                .ok_or_else(|| Error::Msg("PiD serial remaining latent overflow".into()))?;
            let completed = u64::from(window.serial_completed)
                .checked_mul(elements(&[
                    3,
                    mul(route.height as i32, scale)?,
                    mul(route.width as i32, scale)?,
                ])?)
                .ok_or_else(|| Error::Msg("PiD serial completed RGB overflow".into()))?;
            price.held_bytes = price
                .held_bytes
                .checked_add(remaining)
                .and_then(|n| n.checked_add(completed))
                .ok_or_else(|| Error::Msg("PiD coincident serial source overflow".into()))?;
        }
        pricing.take_window(price);
    }
    Ok(pricing)
}
fn decoded_conversion(
    p: &mut Program,
    e: &EngineSources,
    decoded: &Tensor,
    serial: u32,
) -> Result<()> {
    let decoded = p.cast("Z PiD decoded F32 conversion", decoded, Dtype::Float32)?;
    // Four actual elementwise source steps and their immediate source scalars. Their values do
    // not enter any identity/price. No numerical output or fake Image carrier is constructed.
    let half = p.scalar("Z decoded half", Dtype::Float32)?;
    let scaled = p.binary("Z decoded half multiply", &decoded, &half)?;
    let shifted = p.binary("Z decoded half add", &scaled, &half)?;
    let lower = p.scalar("Z decoded clip lower", Dtype::Float32)?;
    let upper = p.scalar("Z decoded clip upper", Dtype::Float32)?;
    let clamped = p.storage(
        "Z decoded clip",
        shifted.shape.clone(),
        shifted.dtype,
        union([&shifted, &lower, &upper]),
        false,
        false,
    )?;
    let byte_scale = p.scalar("Z decoded byte scale", Dtype::Float32)?;
    let scaled = p.binary("Z decoded255 multiply", &clamped, &byte_scale)?;
    p.host("Z decoded F32 CPU readback", scaled.elements()? * 4);
    p.host("Z decoded CURRENT RGB bytes", decoded.elements()?);
    let mut roots = e.raw_engine.clone();
    roots.extend(e.caches.roots());
    roots.extend(decoded.roots.iter().copied());
    p.owner_boundary(
        native_allocations::BoundaryKind::ReadbackWait,
        "existing decoded_to_image CPU-readback wait, NOT completion-capture release",
    );
    p.window(
        IosPhase::Decode,
        crate::quantum::DECODED_TO_IMAGE,
        &roots,
        serial,
        None,
        None,
    )?;
    Ok(())
}

type ObserveFn<'a> = dyn FnMut(u64, &[Condition], &Prefix, &IosPricing, bool) -> Result<()> + 'a;

struct RefusalContext<'a> {
    base: &'a BaseContext,
    route: &'a Route,
    scale: i32,
}

struct Unified<'a> {
    next: u64,
    pricing: IosPricing,
    source_cases: Vec<SourceSummary>,
    observe: &'a mut ObserveFn<'a>,
    complete: u64,
    refused: u64,
    last_failure: Option<(Error, AdapterSourcePrefix, Arc<Prefix>)>,
    pending: Arc<Prefix>,
    attempted_case_id: Option<u64>,
}
impl<'a> Unified<'a> {
    fn new(observe: &'a mut ObserveFn<'a>) -> Self {
        Unified {
            next: 0,
            pricing: IosPricing::default(),
            source_cases: Vec::new(),
            complete: 0,
            refused: 0,
            last_failure: None,
            pending: Arc::new(Prefix::default()),
            attempted_case_id: None,
            observe,
        }
    }
    fn commit(
        &mut self,
        mut local: IosPricing,
        conditions: Vec<Condition>,
        prefix: Arc<Prefix>,
        complete: bool,
    ) -> Result<()> {
        self.pending = Arc::clone(&prefix);
        self.attempted_case_id = None;
        let id = self.next;
        self.next = self
            .next
            .checked_add(1)
            .ok_or_else(|| Error::Msg("PiD composite source-case ordinal overflow".into()))?;
        // Whole local case is already priced; this is the ONLY charging channel and only commit.
        // A probe/consumer error here preserves pending prefix/prior commits and never replays it.
        self.pending = prefix.clone();
        self.attempted_case_id = Some(id);
        (self.observe)(id, &conditions, &prefix, &local, complete)?;
        let digest = source_digest(&prefix, &conditions);
        let summary = SourceSummary {
            case_id: id,
            source_digest: digest.clone(),
            condition_count: conditions.len(),
            operations: prefix.operations.len(),
            source_slots: prefix.capacities.len(),
            workspace_obligations: prefix.workspaces.len(),
            evaluated_native_workspaces: prefix.windows.iter().try_fold(0usize, |n, w| {
                n.checked_add(w.requests.evaluated_workspaces)
                    .ok_or_else(|| Error::Msg("PiD native report count overflow".into()))
            })?,
            requested_components: prefix.windows.iter().try_fold(0usize, |n, w| {
                n.checked_add(w.requests.requests.len())
                    .ok_or_else(|| Error::Msg("PiD native report count overflow".into()))
            })?,
            native_unknowns: prefix.windows.iter().try_fold(0usize, |n, w| {
                n.checked_add(w.requests.unknowns.len())
                    .ok_or_else(|| Error::Msg("PiD native report count overflow".into()))
            })?,
            host_symbolic_owners: prefix
                .host
                .iter()
                .filter(|d| matches!(d, HostDemand::Symbolic { .. }))
                .count(),
        };
        for window in &mut local.windows {
            window.owner_domain = format!(
                "{}/pid-source-case-{id}/recipe-{digest}",
                window.owner_domain
            );
        }
        for window in local.windows {
            self.pricing.take_window(window);
        }
        self.pricing.absent_phases.extend(local.absent_phases);
        self.source_cases.push(summary);
        self.pending = Arc::new(Prefix::default());
        self.attempted_case_id = None;
        if complete {
            self.complete += 1;
        } else {
            self.refused += 1;
        }
        Ok(())
    }
    fn refuse(
        &mut self,
        error: Error,
        adapter: &AdapterSourcePrefix,
        p: Program,
        context: RefusalContext<'_>,
        conditions: Vec<Condition>,
    ) -> Result<()> {
        let RefusalContext { base, route, scale } = context;
        let prefix = Arc::new(p.prefix);
        self.pending = Arc::clone(&prefix);
        let local = pid_source_prices(
            &prefix,
            base,
            route,
            scale,
            "z-image/pid/source-refusal-prefix",
        )?;
        self.last_failure = Some((error, adapter.clone(), prefix.clone()));
        self.commit(local, conditions, prefix, false)
    }
    fn finish(self) -> std::result::Result<PidReport, Box<PidFailure>> {
        if self.complete != 0 {
            return Ok(PidReport {
                pricing: self.pricing,
                source_cases: self.source_cases,
                refused_cases: self.refused,
                complete_cases: self.complete,
            });
        }
        let (error, adapter_prefix, source) = self.last_failure.unwrap_or_else(|| {
            (
                Error::Msg("all PiD source cases refused".into()),
                AdapterSourcePrefix::default(),
                Arc::new(Prefix::default()),
            )
        });
        Err(Box::new(PidFailure {
            error,
            adapter_prefix,
            source,
            committed: self.pricing.windows,
            attempted_case_id: None,
            adapter_attempted_case_id: None,
        }))
    }
}

fn emit_raw_io_refusals(
    unified: &mut Unified,
    adapter: &AdapterSourcePrefix,
    p: &Program,
    base: &BaseContext,
    route: &Route,
    scale: i32,
) -> Result<()> {
    for (ordinal, prefix) in p.raw_io_frontiers.iter().enumerate() {
        let mut conditions = (0..ordinal)
            .map(|prior| Condition::RawMaterializeBatchCompletes {
                ordinal: prior,
                completes: true,
            })
            .collect::<Vec<_>>();
        conditions.push(Condition::RawMaterializeBatchCompletes {
            ordinal,
            completes: false,
        });
        unified.pending = Arc::clone(prefix);
        let local = pid_source_prices(
            prefix,
            base,
            route,
            scale,
            "z-image/pid/RAW-materialize-refused-prefix",
        )?;
        unified.last_failure = Some((
            Error::Msg("conditional native PiD RAW materialization IO refusal".into()),
            adapter.clone(),
            Arc::clone(prefix),
        ));
        unified.commit(local, conditions, Arc::clone(prefix), false)?;
    }
    Ok(())
}

struct CaseSources<'a> {
    adapter: &'a AdapterSourceCase,
    ledger: &'a super::adapter_sources::Ledger,
    quanta: &'a [Priced],
    base_pricing: &'a IosPricing,
    controls: &'a Controls,
    work_basis: Option<PidWorkBasis>,
}
fn source_case(
    unified: &mut Unified,
    model: &Model,
    route: &Route,
    sampling: &ResolvedFlowControls,
    spec: &LoadSpec,
    args: CaseSources<'_>,
) -> Result<()> {
    let CaseSources {
        adapter,
        ledger,
        quanta,
        base_pricing,
        controls,
        work_basis,
    } = args;
    let (positive, zero) = super::source_callback_presence(route, sampling)?;
    let forwards = [(true, positive), (false, zero)]
        .into_iter()
        .filter_map(|(v, on)| on.then_some(v))
        .collect::<Vec<_>>();
    // Control's native engine NEVER loads/mints PiD. Do not accidentally serve a new overlay on
    // that path merely because a base LoadSpec field or serialized request flag is present.
    let engine_needed = !route.mode.control()
        && spec.pid.is_some()
        && (!route.execution.stage_residency || controls.use_pid);
    if !engine_needed {
        if controls.use_pid && !route.mode.control() {
            let mut p = Program::default();
            p.begin();
            p.window(
                IosPhase::Denoise,
                "PiD missing configured engine refusal",
                &BTreeSet::new(),
                0,
                None,
                None,
            )?;
            let base = base_context(model, route, sampling, quanta, true, adapter, ledger)?;
            return unified.refuse(
                Error::Msg(
                    "use_pid was requested but no PiD decoder is loaded (load with LoadSpec::pid)"
                        .into(),
                ),
                &adapter.prefix,
                p,
                RefusalContext {
                    base: &base,
                    route,
                    scale: 1,
                },
                vec![Condition::EngineUsed(false)],
            );
        }
        return unified.commit(
            base_pricing.clone(),
            vec![Condition::EngineUsed(false)],
            Arc::new(Prefix::default()),
            true,
        );
    }
    let mut loaded = Program::default();
    loaded.begin();
    let first_base = base_context(model, route, sampling, quanta, true, adapter, ledger)?;
    let engine = match open_engine(&mut loaded, spec, adapter.case_id) {
        Ok(engine) => engine,
        Err(error) => {
            // Even an early map/header/tokenizer failure preserves attempted file/source records.
            emit_raw_io_refusals(unified, &adapter.prefix, &loaded, &first_base, route, 1)?;
            loaded.window(
                IosPhase::Load,
                "PiD reached source IO/constructor refusal",
                &BTreeSet::new(),
                0,
                None,
                None,
            )?;
            return unified.refuse(
                error,
                &adapter.prefix,
                loaded,
                RefusalContext {
                    base: &first_base,
                    route,
                    scale: 1,
                },
                vec![Condition::EngineUsed(true)],
            );
        }
    };
    emit_raw_io_refusals(
        unified,
        &adapter.prefix,
        &loaded,
        &first_base,
        route,
        engine.cfg.sr_scale,
    )?;
    if !controls.use_pid {
        let mut local = base_pricing.clone();
        add_source_owner(&mut local, &loaded.prefix, &engine.raw_engine, |window| {
            window.phase != IosPhase::Load
        })?;
        let extra = pid_source_prices(
            &loaded.prefix,
            &first_base,
            route,
            engine.cfg.sr_scale,
            "z-image/pid/configured-engine-native-VAE-route",
        )?;
        for window in extra.windows {
            local.take_window(window);
        }
        return unified.commit(
            local,
            vec![Condition::EngineUsed(false)],
            Arc::new(loaded.prefix),
            true,
        );
    }
    // Actual native guard refusal is a separate closed resource-policy case BEFORE decoder mint.
    let mut refused = loaded.clone();
    refused.window(
        IosPhase::Denoise,
        "PiD native minimum-tile guard refusal",
        &engine.raw_engine,
        0,
        None,
        None,
    )?;
    unified.refuse(
        Error::Msg("conditional native PiD minimum-tile guard refusal".into()),
        &adapter.prefix,
        refused,
        RefusalContext {
            base: &first_base,
            route,
            scale: engine.cfg.sr_scale,
        },
        vec![Condition::NativeMinimumTileGuardPasses(false)],
    )?;
    unified.pending = Arc::new(loaded.prefix.clone());
    let choices = selections(controls, &engine.cfg, route)?;
    for (selection, conditions) in choices {
        for &forward in &forwards {
            for input_dtype in [Dtype::Bfloat16, Dtype::Float32] {
                let mut p = loaded.clone();
                let mut e = engine.clone();
                p.begin();
                let base = base_context(model, route, sampling, quanta, forward, adapter, ledger)?;
                let mut conditions = conditions.clone();
                conditions.extend((0..loaded.raw_io_frontiers.len()).map(|ordinal| {
                    Condition::RawMaterializeBatchCompletes {
                        ordinal,
                        completes: true,
                    }
                }));
                conditions.extend([
                    Condition::NativeMinimumTileGuardPasses(true),
                    Condition::InputLatentDtype(input_dtype),
                    Condition::ZModelForward(forward),
                    Condition::CaptureControl {
                        requested: controls.capture_sigma,
                    },
                ]);
                // SAME retained student map. Rebuilding a decoder is not a source-file reopening.
                let built = (|| -> Result<(IosPricing, Tensor)> {
                    construct_student(&mut p, &mut e.student, &e.cfg)?;
                    p.hold_host(
                        "PiD per-generation Sampler t_list clone",
                        SamplerConfig::distill_4step().student_t_list.len() as u64 * 4,
                    );
                    p.window(
                        IosPhase::Denoise,
                        "PiD decoder constructor SAME student roots",
                        &e.raw_engine,
                        0,
                        None,
                        None,
                    )?;
                    let caption = gemma_caption(&mut p, &mut e, work_basis)?;
                    if let Some((edge, overlap)) = controls.selected_tile {
                        // Native validate_tile is AFTER engine.decoder (net + full caption).
                        mlx_gen_pid::budget::validate_tile(
                            crate::model::MODEL_ID,
                            edge,
                            overlap,
                            mul(route.height as i32, e.cfg.sr_scale)?,
                            mul(route.width as i32, e.cfg.sr_scale)?,
                        )?;
                    }
                    let mut local = base_pricing.clone();
                    local
                        .windows
                        .retain(|window| window.phase != IosPhase::Decode);
                    local.windows.retain(|window| {
                        !matches!(window.phase, IosPhase::Denoise)
                            || (window
                                .class
                                .as_deref()
                                .is_some_and(|class| class.contains("/no-model-forward/")))
                                != forward
                    });
                    let mut owners = e.raw_engine.clone();
                    owners.extend(caption.roots.iter().copied());
                    add_source_owner(&mut local, &p.prefix, &e.raw_engine, |window| {
                        window.phase == IosPhase::Conditioning && !route.execution.stage_residency
                    })?;
                    add_source_owner(&mut local, &p.prefix, &owners, |window| {
                        matches!(window.phase, IosPhase::Encode | IosPhase::Denoise)
                    })?;
                    Ok((local, caption))
                })();
                let (mut local, caption) = match built {
                    Ok(value) => value,
                    Err(error) => {
                        p.window(
                            IosPhase::Denoise,
                            "PiD reached decoder/caption/selected-tile refusal",
                            &e.raw_engine,
                            0,
                            None,
                            None,
                        )?;
                        unified.refuse(
                            error,
                            &adapter.prefix,
                            p,
                            RefusalContext {
                                base: &base,
                                route,
                                scale: e.cfg.sr_scale,
                            },
                            conditions,
                        )?;
                        continue;
                    }
                };
                // First and last serial endpoints; do not replay requested N and do NOT reset
                // SAME PidNet/Sampler caches before the later endpoint. Fresh RNG epochs still
                // exist per decode even though native PidDecoder.seed has the same value.
                let endpoints = if route.count > 1 {
                    vec![0, route.count - 1]
                } else {
                    vec![0]
                };
                let mut failed = None;
                for serial in endpoints {
                    let input = p.external(
                        vec![
                            1,
                            e.cfg.lq_latent_channels,
                            route.height as i32 / 8,
                            route.width as i32 / 8,
                        ],
                        input_dtype,
                    ); // backing may alias a larger caller latent; never infer from frame shape
                    match sample_decode(
                        &mut p,
                        &mut e,
                        &caption,
                        &input,
                        &selection,
                        serial,
                        &SamplerConfig::distill_4step(),
                    ) {
                        Ok(decoded) => {
                            if let Err(error) = decoded_conversion(&mut p, &e, &decoded, serial) {
                                failed = Some(error);
                                break;
                            }
                        }
                        Err(error) => {
                            failed = Some(error);
                            break;
                        }
                    }
                }
                if let Some(error) = failed {
                    p.window(
                        IosPhase::Decode,
                        "PiD reached sampler/tile/conversion refusal",
                        &e.raw_engine,
                        0,
                        None,
                        None,
                    )?;
                    unified.refuse(
                        error,
                        &adapter.prefix,
                        p,
                        RefusalContext {
                            base: &base,
                            route,
                            scale: e.cfg.sr_scale,
                        },
                        conditions,
                    )?;
                    continue;
                }
                let prefix = Arc::new(p.prefix);
                unified.pending = Arc::clone(&prefix);
                let extra = pid_source_prices(
                    &prefix,
                    &base,
                    route,
                    e.cfg.sr_scale,
                    "z-image/pid/actual-fixed-source-case",
                )?;
                for window in extra.windows {
                    local.take_window(window);
                }
                unified.commit(local, conditions, prefix, true)?;
            }
        }
    }
    Ok(())
}

pub(super) fn price_request(
    provider: &str,
    spec: &LoadSpec,
    route: &Route,
    sampling: &ResolvedFlowControls,
    initial_width: u64,
    work_basis: Option<PidWorkBasis>,
) -> std::result::Result<PidReport, Box<PidFailure>> {
    price_request_using(
        provider,
        spec,
        route,
        sampling,
        initial_width,
        work_basis,
        &mut |_, _, _, _, _| Ok(()),
    )
}

fn price_request_using<'a>(
    provider: &str,
    spec: &LoadSpec,
    route: &Route,
    sampling: &ResolvedFlowControls,
    initial_width: u64,
    work_basis: Option<PidWorkBasis>,
    observe: &'a mut ObserveFn<'a>,
) -> std::result::Result<PidReport, Box<PidFailure>> {
    let setup = (|| -> Result<_> {
        let controls = Controls::from_source(sampling, route)?;
        if controls.use_pid && !route.mode.control() {
            PidWorkBasis::require(work_basis)?;
        }
        let mut component_spec = spec.clone();
        component_spec.pid = None;
        component_spec.adapters.clear();
        let facts = Facts::from_spec(provider, &component_spec)?;
        let model = Model::new(facts)?;
        let WeightsSource::Dir(root) = &spec.weights else {
            return Err(Error::Unsupported(
                "Z PiD source pricing requires snapshot directory".into(),
            ));
        };
        let mut headers = wm::safetensors_path_tensor_headers(root.join("transformer"))
            .map_err(Error::from)?
            .into_iter()
            .map(|header| (header.name.clone(), header))
            .collect::<BTreeMap<_, _>>();
        crate::loader::remap_transformer_header_keys(&mut headers);
        let cfg = model.facts.dit_cfg.clone();
        let kohya = crate::transformer::adapter_header_targets(&cfg, false)
            .into_iter()
            .map(|(path, _)| path)
            .collect::<Vec<_>>();
        let mut projections = BTreeMap::new();
        for (path, header) in crate::transformer::adapter_header_targets(&cfg, true) {
            projections.insert(
                path,
                mlx_gen::adapters::facts::linear_header_facts(
                    &headers,
                    &header,
                    mlx_gen::quant::DEFAULT_GROUP_SIZE,
                    spec.quantize.map(|q| q.bits()),
                )?,
            );
        }
        Ok((controls, model, cfg, kohya, projections))
    })();
    let (controls, mut model, cfg, kohya, projections) = setup.map_err(|error| {
        Box::new(PidFailure {
            error,
            adapter_prefix: AdapterSourcePrefix::default(),
            source: Arc::new(Prefix::default()),
            committed: Vec::new(),
            attempted_case_id: None,
            adapter_attempted_case_id: None,
        })
    })?;
    let mut projection = |path: &str| -> Result<_> {
        let Some(canonical) = crate::transformer::canonical_adapter_path(&cfg, path) else {
            return Ok(None);
        };
        Ok(projections
            .get(&canonical)
            .cloned()
            .map(|facts| (canonical, facts)))
    };
    let mut unified = Unified::new(observe);
    let mut previous = BTreeSet::new();
    let mut consume = |adapter: &AdapterSourceCase| -> Result<()> {
        unified.pending = Arc::new(Prefix::default());
        let mut ledger = super::adapter_sources::Ledger::default();
        ledger.apply(&adapter.prefix)?;
        super::adapter_sources::install_metadata(&mut model, adapter, &mut previous)?;
        model.visit_quanta(route, Backend::Metal, &mut |quanta| {
            if matches!(adapter.outcome, AdapterSourceOutcome::Refused(_)) {
                // Native adapter failure is BEFORE PiD opening. Do not eagerly touch a future
                // PiD file and hide this earlier source history or make a refused generation valid.
                let mut local = IosPricing::default();
                for window in super::adapter_sources::load_windows(
                    &model,
                    adapter,
                    quanta,
                    &ledger,
                    route.execution.stage_residency,
                )? {
                    local.take_window(window);
                }
                unified.last_failure = Some((
                    Error::Unsupported(format!(
                        "{provider}: every native adapter source case refuses"
                    )),
                    adapter.prefix.clone(),
                    Arc::new(Prefix::default()),
                ));
                return unified.commit(
                    local,
                    vec![Condition::EngineUsed(false)],
                    Arc::new(Prefix::default()),
                    false,
                );
            }
            let mut base =
                price_source_quanta(&model.facts, quanta, route, sampling, initial_width)?;
            super::adapter_sources::decorate(&model, adapter, &ledger, quanta, &mut base, route)?;
            source_case(
                &mut unified,
                &model,
                route,
                sampling,
                spec,
                CaseSources {
                    adapter,
                    ledger: &ledger,
                    quanta,
                    base_pricing: &base,
                    controls: &controls,
                    work_basis,
                },
            )
        })
    };
    let visited = visit_configured_adapter_source_cases(
        &spec.adapters,
        &kohya,
        mlx_gen::adapters::DEFAULT_LYCORIS_DTYPE,
        &mut projection,
        &mut consume,
    );
    match visited {
        Ok(summary) if summary.enumeration_complete => unified.finish(),
        Ok(_) => Err(Box::new(PidFailure {
            error: Error::Msg("PiD source enumeration did not finish".into()),
            adapter_prefix: AdapterSourcePrefix::default(),
            source: unified.pending,
            committed: unified.pricing.windows,
            attempted_case_id: None,
            adapter_attempted_case_id: None,
        })),
        Err(failure) => {
            let AdapterSourceFailure {
                error,
                prefix,
                attempted_case_id,
                ..
            } = *failure;
            Err(Box::new(PidFailure {
                error,
                adapter_prefix: prefix,
                source: unified.pending,
                committed: unified.pricing.windows,
                attempted_case_id: unified.attempted_case_id,
                adapter_attempted_case_id: attempted_case_id,
            }))
        }
    }
}

/// Actual public request/token entrypoint integration target. The original full spec feeds both
/// visitors; only base-component metadata extraction clears overlays. The existing outward PiD
/// refusal stays TEMPORARILY until source/native/host/workspace/physical qualification, not as a
/// permanent NoEvidence implementation. Report/source probes use the rich channel above first.
pub(super) fn price_before_existing_guard(
    req: &mlx_gen::ios_pricing::IosPricingRequest<'_>,
    route: &Route,
    sampling: &ResolvedFlowControls,
    initial_width: u64,
    work_basis: Option<PidWorkBasis>,
) -> Result<IosPricing> {
    let actual = req.generation.map(PidWorkBasis::actual);
    if actual
        .zip(work_basis)
        .is_some_and(|(left, right)| left != right)
    {
        return Err(Error::Msg(
            "PiD replay work basis differs from actual request".into(),
        ));
    }
    let report = price_request(
        req.provider,
        req.spec,
        route,
        sampling,
        initial_width,
        actual.or(work_basis),
    )
    .map_err(|failure| failure.into_legacy_error())?;
    let source_windows = report.pricing.windows.len();
    let workspaces = report
        .source_cases
        .iter()
        .map(|summary| summary.workspace_obligations as u64)
        .sum::<u64>();
    let host_symbols = report
        .source_cases
        .iter()
        .map(|summary| summary.host_symbolic_owners as u64)
        .sum::<u64>();
    for summary in &report.source_cases {
        let mut digest = [0u8; 8];
        for (index, byte) in summary.source_digest.as_bytes().iter().take(8).enumerate() {
            digest[index] = *byte;
        }
        mlx_gen::ios_quantum::diagnostic_metadata(
            "z_image",
            "pid-fixed-source-recipe-summary",
            &[
                ("schema", 1),
                ("case_id", summary.case_id),
                ("recipe_digest_prefix", u64::from_le_bytes(digest)),
                ("conditions", summary.condition_count as u64),
                ("operations", summary.operations as u64),
                ("source_slots", summary.source_slots as u64),
                (
                    "native_evaluated_workspaces",
                    summary.evaluated_native_workspaces as u64,
                ),
                (
                    "native_requested_components",
                    summary.requested_components as u64,
                ),
                ("native_unbound_terms", summary.native_unknowns as u64),
            ],
        );
    }
    mlx_gen::ios_quantum::diagnostic_metadata(
        "z_image",
        "pid-fixed-source-pricing-checkpoint",
        &[
            ("schema", 1),
            ("source_cases", report.source_cases.len() as u64),
            ("complete_cases", report.complete_cases),
            ("refused_cases", report.refused_cases),
            ("priced_source_windows", source_windows as u64),
            ("native_workspace_obligations", workspaces),
            ("host_symbolic_owners", host_symbols),
        ],
    );
    // This is the SAME existing public compatibility guard, not permission to admit incomplete
    // workspace/host/nativeData/physical bounds. Nothing in this candidate calls an OS safe cap.
    Err(Error::Unsupported(format!(
        "{}: a PiD decode overlay is not an iOS load",
        req.provider
    )))
}

fn dtype_tag(dtype: Dtype) -> u64 {
    match dtype {
        Dtype::Bool => 0,
        Dtype::Uint8 => 1,
        Dtype::Uint16 => 2,
        Dtype::Uint32 => 3,
        Dtype::Uint64 => 4,
        Dtype::Int8 => 5,
        Dtype::Int16 => 6,
        Dtype::Int32 => 7,
        Dtype::Int64 => 8,
        Dtype::Float16 => 9,
        Dtype::Float32 => 10,
        Dtype::Float64 => 11,
        Dtype::Bfloat16 => 12,
        Dtype::Complex64 => 13,
    }
}
fn source_digest(prefix: &Prefix, conditions: &[Condition]) -> String {
    use sha2::{Digest, Sha256};
    fn word(h: &mut Sha256, value: u64) {
        h.update(value.to_le_bytes());
    }
    fn text(h: &mut Sha256, value: &str) {
        word(h, value.len() as u64);
        h.update(value.as_bytes());
    }
    fn tensor(h: &mut Sha256, shape: &[i32], dtype: Dtype) {
        word(h, shape.len() as u64);
        for &d in shape {
            h.update(d.to_le_bytes());
        }
        word(h, dtype_tag(dtype));
    }
    fn roots(h: &mut Sha256, slots: &BTreeSet<Slot>) {
        word(h, slots.len() as u64);
        for slot in slots {
            word(h, slot.0 as u64);
        }
    }
    fn host(h: &mut Sha256, demand: &HostDemand) {
        match demand {
            HostDemand::Bytes { site, bytes } => {
                word(h, 0);
                text(h, site);
                word(h, *bytes);
            }
            HostDemand::Symbolic {
                site,
                symbol,
                coefficient,
            } => {
                word(h, 1);
                text(h, site);
                text(h, symbol);
                word(h, *coefficient);
            }
        }
    }
    let mut h = Sha256::new();
    text(&mut h, "z-image-pid-fixed-source-recipe-v4");
    text(&mut h, native_allocations::AUTHORITY);
    for c in conditions {
        match c {
            Condition::EngineUsed(value) => {
                word(&mut h, 0);
                word(&mut h, u64::from(*value));
            }
            Condition::NativeMinimumTileGuardPasses(value) => {
                word(&mut h, 1);
                word(&mut h, u64::from(*value));
            }
            Condition::AutoPlannerReturns { edge } => {
                word(&mut h, 2);
                word(&mut h, u64::from(edge.is_some()));
                word(&mut h, edge.unwrap_or(0) as u64);
            }
            Condition::ExplicitTile { edge, overlap } => {
                word(&mut h, 3);
                h.update(edge.to_le_bytes());
                h.update(overlap.to_le_bytes());
            }
            Condition::InputLatentDtype(dtype) => {
                word(&mut h, 4);
                word(&mut h, dtype_tag(*dtype));
            }
            Condition::ZModelForward(value) => {
                word(&mut h, 5);
                word(&mut h, u64::from(*value));
            }
            Condition::CaptureControl { requested } => {
                word(&mut h, 6);
                word(&mut h, u64::from(requested.is_some()));
                word(&mut h, requested.unwrap_or(0.0).to_bits() as u64);
            }
            Condition::RawMaterializeBatchCompletes { ordinal, completes } => {
                word(&mut h, 7);
                word(&mut h, *ordinal as u64);
                word(&mut h, u64::from(*completes));
            }
        }
    }
    for value in &prefix.values {
        tensor(&mut h, &value.shape, value.dtype);
        text(&mut h, &format!("{:?}", value.rule));
    }
    for boundary in &prefix.boundaries {
        word(&mut h, boundary.after_window as u64);
        text(&mut h, &format!("{:?}", boundary.kind));
        text(&mut h, &boundary.site);
    }
    for c in &prefix.capacities {
        text(&mut h, &c.name);
        tensor(&mut h, &c.shape, c.dtype);
        word(&mut h, c.bytes);
        word(&mut h, u64::from(c.immediate));
        word(&mut h, u64::from(c.possible_view_copy));
        word(&mut h, u64::from(c.graph_attached));
        word(&mut h, c.value as u64);
        roots(&mut h, &c.parents);
    }
    for op in &prefix.operations {
        text(&mut h, &op.site);
        word(&mut h, u64::from(op.completed));
        word(&mut h, op.operands.len() as u64);
        for (shape, dtype) in &op.operands {
            tensor(&mut h, shape, *dtype);
        }
        word(&mut h, u64::from(op.output.is_some()));
        if let Some((shape, dtype)) = &op.output {
            tensor(&mut h, shape, *dtype);
        }
        for value in &op.operand_values {
            word(&mut h, *value as u64);
        }
        word(&mut h, u64::from(op.output_value.is_some()));
        if let Some(value) = op.output_value {
            word(&mut h, value as u64);
        }
    }
    for w in &prefix.workspaces {
        text(&mut h, &w.site);
        text(&mut h, w.family);
        word(&mut h, w.inputs.len() as u64);
        for (shape, dtype) in &w.inputs {
            tensor(&mut h, shape, *dtype);
        }
        for value in &w.input_values {
            word(&mut h, *value as u64);
        }
        text(&mut h, &format!("{:?}", w.call));
        word(&mut h, u64::from(w.output_slot.is_some()));
        if let Some(slot) = w.output_slot {
            word(&mut h, slot.0 as u64);
        }
        word(&mut h, u64::from(w.affine_rows.is_some()));
        if let Some(source) = &w.affine_rows {
            h.update(source.group_size.to_le_bytes());
            h.update(source.bits.to_le_bytes());
            word(&mut h, source.output_slot.0 as u64);
            tensor(&mut h, &source.output_shape, source.output_dtype);
            text(&mut h, source.dispatch_condition);
            for gather in &source.gathers {
                for shape in [
                    &gather.table_shape,
                    &gather.indices_shape,
                    &gather.gather_shape,
                    &gather.squeezed_shape,
                ] {
                    word(&mut h, shape.len() as u64);
                    for dim in shape {
                        h.update(dim.to_le_bytes());
                    }
                }
                word(&mut h, gather.gather_slot.0 as u64);
                word(&mut h, gather.squeeze_axis as u64);
                match gather.layout {
                    GatherRowLayout::Native096GatherSetDataThenSqueeze => word(&mut h, 1),
                }
            }
        }
    }
    for demand in &prefix.host {
        host(&mut h, demand);
    }
    for w in &prefix.windows {
        text(&mut h, w.phase.as_key());
        text(&mut h, &w.site);
        roots(&mut h, &w.resident_roots);
        roots(&mut h, &w.created_roots);
        word(&mut h, w.workspace_start as u64);
        word(&mut h, w.workspace_end as u64);
        word(&mut h, w.operation_start as u64);
        word(&mut h, w.operation_end as u64);
        word(&mut h, w.value_start as u64);
        word(&mut h, w.value_end as u64);
        word(&mut h, w.owners.boundary_count as u64);
        for descriptor in &w.owners.descriptors {
            word(&mut h, descriptor.descriptor as u64);
            word(&mut h, u64::from(descriptor.native_shape_bound));
            tensor(&mut h, &descriptor.shape, descriptor.dtype);
            text(&mut h, &format!("{:?}", descriptor.data));
            text(&mut h, &format!("{:?}", descriptor.leases));
            word(&mut h, u64::from(descriptor.data_size.is_some()));
            if let Some(n) = descriptor.data_size {
                word(&mut h, n);
            }
            text(&mut h, &format!("{:?}", descriptor.strides));
            text(&mut h, &format!("{:?}", descriptor.row_contiguous));
            text(&mut h, &format!("{:?}", descriptor.col_contiguous));
            text(&mut h, &format!("{:?}", descriptor.contiguous));
            text(&mut h, &format!("{:?}", descriptor.offset_bytes));
            text(&mut h, &format!("{:?}", descriptor.fresh_request_bytes));
            word(&mut h, u64::from(descriptor.same_data_parent.is_some()));
            if let Some(parent) = descriptor.same_data_parent {
                word(&mut h, parent as u64);
            }
            if let Some(rule) = descriptor.donation_rule {
                text(&mut h, rule);
            }
        }
        for lease in &w.owners.leases {
            text(&mut h, &format!("{:?}", lease.key));
            match &lease.length {
                native_allocations::LeaseLength::Empty => word(&mut h, 0),
                native_allocations::LeaseLength::Unbound { requested } => {
                    word(&mut h, 1);
                    word(&mut h, u64::from(requested.is_some()));
                    if let Some(n) = requested {
                        word(&mut h, *n);
                    }
                }
            }
            match &lease.cached_reuse {
                native_allocations::CacheLengthBound::Empty => word(&mut h, 0),
                native_allocations::CacheLengthBound::Cached {
                    aligned_request,
                    upper_exclusive,
                } => {
                    word(&mut h, 1);
                    word(&mut h, *aligned_request);
                    word(&mut h, *upper_exclusive);
                }
                native_allocations::CacheLengthBound::Unbound(reason) => {
                    word(&mut h, 2);
                    text(&mut h, reason);
                }
            }
            word(&mut h, u64::from(lease.may_be_queued));
            word(
                &mut h,
                u64::from(lease.may_enter_cache_after_unproved_retirement),
            );
        }
        for unknown in &w.owners.unknowns {
            text(&mut h, unknown);
        }
        word(&mut h, w.requests.known_extra_requested);
        word(&mut h, w.requests.evaluated_workspaces as u64);
        for request in &w.requests.requests {
            word(&mut h, u64::from(request.workspace.is_some()));
            if let Some(index) = request.workspace {
                word(&mut h, index as u64);
            }
            word(&mut h, u64::from(request.source_value.is_some()));
            if let Some(value) = request.source_value {
                word(&mut h, value as u64);
            }
            text(&mut h, request.domain);
            text(&mut h, request.rule);
            text(&mut h, &request.predicate);
            word(&mut h, u64::from(request.bytes.is_some()));
            if let Some(bytes) = request.bytes {
                word(&mut h, bytes);
            }
            word(&mut h, u64::from(request.existing_output.is_some()));
            if let Some(slot) = request.existing_output {
                word(&mut h, slot.0 as u64);
            }
            word(&mut h, u64::from(request.data_size_product.is_some()));
            if let Some(product) = &request.data_size_product {
                word(&mut h, product.itemsize);
                word(&mut h, product.output_slot.0 as u64);
                word(&mut h, u64::from(product.native_input_mapping_bound));
                word(&mut h, product.source_input_candidates.len() as u64);
                for input in &product.source_input_candidates {
                    word(&mut h, *input as u64);
                }
            }
        }
        for unknown in &w.requests.unknowns {
            text(&mut h, unknown);
        }
        word(&mut h, w.serial_completed as u64);
        word(&mut h, u64::from(w.step.is_some()));
        word(&mut h, w.step.unwrap_or(0) as u64);
        word(&mut h, u64::from(w.tile.is_some()));
        if let Some(tile) = w.tile {
            for n in [tile.y0, tile.y1, tile.x0, tile.x1] {
                h.update(n.to_le_bytes());
            }
        }
        for demand in &w.host {
            host(&mut h, demand);
        }
    }
    // Paths are source observations, not part of the machine-independent recipe identity.
    word(&mut h, prefix.source_files.len() as u64);
    format!("{:x}", h.finalize())
}

pub(super) fn requested(sampling: &ResolvedFlowControls) -> Result<bool> {
    let value =
        serde_json::to_value(sampling.recipe()).map_err(|error| Error::Msg(error.to_string()))?;
    value
        .get("use_pid")
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| Error::Msg("validated flow recipe missing PiD control".into()))
}

#[cfg(test)]
mod source_tests {
    use super::*;
    #[test]
    fn gemma_gathered_affine_preserves_actual_batch_rank_layout_and_single_output_source() {
        // Independent caller geometry: actual native ids[1,L], tables[vocab,C].
        // No native Array, tokenizer, source file, model weight or runtime device is constructed.
        for bits in [4, 8] {
            for dtype in [Dtype::Float16, Dtype::Bfloat16, Dtype::Float32] {
                let mut p = Program::default();
                p.begin();
                let ids = p
                    .immediate("fixture ids", vec![1, 13], Dtype::Int32)
                    .unwrap();
                let codes = p
                    .immediate(
                        "fixture packed table",
                        vec![19, 128 * bits / 32],
                        Dtype::Uint32,
                    )
                    .unwrap();
                let scales = p
                    .immediate("fixture scales table", vec![19, 2], dtype)
                    .unwrap();
                let biases = p
                    .immediate("fixture biases table", vec![19, 2], dtype)
                    .unwrap();
                let (codes, codes_source) = p
                    .embedding_take_axis_zero("Gemma gather packed rows", &codes, &ids)
                    .unwrap();
                let (scales, scales_source) = p
                    .embedding_take_axis_zero("Gemma gather row scales", &scales, &ids)
                    .unwrap();
                let (biases, biases_source) = p
                    .embedding_take_axis_zero("Gemma gather row biases", &biases, &ids)
                    .unwrap();
                for (source, columns) in [
                    (&codes_source, 128 * bits / 32),
                    (&scales_source, 2),
                    (&biases_source, 2),
                ] {
                    assert_eq!(
                        source.indices_shape,
                        vec![1, 13],
                        "Gemma native take must preserve ids[1,L]"
                    );
                    assert_eq!(
                        source.gather_shape,
                        vec![1, 13, 1, columns],
                        "Gemma native Gather output keeps singleton slice axis"
                    );
                    assert_eq!(
                        source.squeezed_shape,
                        vec![1, 13, columns],
                        "Gemma native dequant operands preserve batch rank"
                    );
                    assert_eq!(
                        source.squeeze_axis, 2,
                        "Gemma axis0 take squeezes ids.ndim axis2 only"
                    );
                    assert_eq!(
                        source.layout,
                        GatherRowLayout::Native096GatherSetDataThenSqueeze,
                        "Gemma layout proof must retain native Gather/Squeeze flag rule"
                    );
                }
                let gather_slots = BTreeSet::from([
                    codes_source.gather_slot,
                    scales_source.gather_slot,
                    biases_source.gather_slot,
                ]);
                assert_eq!(
                    gather_slots.len(),
                    3,
                    "three gathers remain separate requested-allocation sources"
                );
                let out = p
                    .gathered_affine_rows(
                        &codes,
                        &scales,
                        &biases,
                        [codes_source, scales_source, biases_source],
                        bits,
                    )
                    .unwrap();
                assert_eq!(
                    out.shape,
                    vec![1, 13, 128],
                    "Gemma affine output preserves actual batch rank"
                );
                let workspace = p.prefix.workspaces.last().unwrap();
                assert_eq!(
                    workspace
                        .inputs
                        .iter()
                        .map(|(shape, _)| shape.clone())
                        .collect::<Vec<_>>(),
                    vec![vec![1, 13, 128 * bits / 32], vec![1, 13, 2], vec![1, 13, 2]],
                    "Gemma workspace binds exact native input ranks"
                );
                let source = workspace.affine_rows.as_ref().unwrap();
                assert_eq!(source.group_size, 64);
                assert_eq!(source.bits, bits);
                assert_eq!(
                    out.roots,
                    BTreeSet::from([source.output_slot]),
                    "dequant workspace must refer to existing output slot exactly once"
                );
                assert!(
                    !gather_slots.contains(&source.output_slot),
                    "dequant output cannot alias a gathered input in this source allocator rule"
                );
                assert_eq!(
                    p.prefix
                        .capacities
                        .iter()
                        .filter(|c| c.name == "Gemma selected rows dequantized")
                        .count(),
                    1,
                    "dequant output must not be added again as workspace allocation"
                );
                let width = if dtype == Dtype::Float32 { 4 } else { 2 };
                assert_eq!(
                    p.prefix.capacities[source.output_slot.0].bytes,
                    13 * 128 * width
                );
                assert!(
                    source.dispatch_condition.contains("Metal")
                        && source.dispatch_condition.contains("CPU")
                        && source.dispatch_condition.contains("Custom"),
                    "affine source must preserve CPU/Custom composed fallback condition"
                );
            }
        }
    }
    #[test]
    fn gemma_affine_shape_only_or_wrong_slot_provenance_is_not_a_layout_certificate() {
        let mut p = Program::default();
        p.begin();
        let ids = p.immediate("ids", vec![1, 13], Dtype::Int32).unwrap();
        let w = p.immediate("codes", vec![19, 16], Dtype::Uint32).unwrap();
        let s = p.immediate("scales", vec![19, 2], Dtype::Bfloat16).unwrap();
        let b = p.immediate("biases", vec![19, 2], Dtype::Bfloat16).unwrap();
        let (w, wp) = p.embedding_take_axis_zero("w", &w, &ids).unwrap();
        let (s, sp) = p.embedding_take_axis_zero("s", &s, &ids).unwrap();
        let (b, bp) = p.embedding_take_axis_zero("b", &b, &ids).unwrap();
        let prior = p.prefix.capacities.len();
        let mut normalized = sp.clone();
        normalized.indices_shape = vec![13];
        assert!(
            p.gathered_affine_rows(&w, &s, &b, [wp.clone(), normalized, bp.clone()], 4)
                .is_err(),
            "same byte products cannot replace actual ids rank/layout provenance"
        );
        let mut wrong_slot = sp;
        wrong_slot.gather_slot = wp.gather_slot;
        assert!(
            p.gathered_affine_rows(&w, &s, &b, [wp, wrong_slot, bp], 4)
                .is_err(),
            "same shape/dtype cannot substitute a different gather source slot"
        );
        assert_eq!(
            p.prefix.capacities.len(),
            prior,
            "refused provenance must not fabricate a successful dequant output"
        );
    }
    #[test]
    fn same_cache_endpoint_survives_steps_and_count_but_geometry_config_and_domain_do_not_alias() {
        let mut p = Program::default();
        p.begin();
        let mut cache = Caches::default();
        let key = CacheKey::PixelPosition {
            config: "cfg-A".into(),
            h: 2,
            w: 3,
        };
        let first = cache
            .get_or_build(&mut p, key.clone(), |p| {
                Ok(vec![p.immediate(
                    "independent raw position",
                    vec![6, 4],
                    Dtype::Float32,
                )?])
            })
            .unwrap();
        let hit = cache
            .get_or_build(&mut p, key, |_| {
                panic!("same PidNet cache hit cannot recreate a table")
            })
            .unwrap();
        assert_eq!(first[0].roots, hit[0].roots);
        assert_eq!(p.prefix.capacities.len(), 1);
        for key in [
            CacheKey::PixelPosition {
                config: "cfg-B".into(),
                h: 2,
                w: 3,
            },
            CacheKey::PixelPosition {
                config: "cfg-A".into(),
                h: 3,
                w: 2,
            },
            CacheKey::ImageRope {
                config: "cfg-A".into(),
                h: 2,
                w: 3,
            },
            CacheKey::PixelRope {
                config: "cfg-A".into(),
                h: 2,
                w: 3,
            },
        ] {
            let next = cache
                .get_or_build(&mut p, key, |p| {
                    Ok(vec![p.immediate(
                        "independent other table",
                        vec![6, 4],
                        Dtype::Float32,
                    )?])
                })
                .unwrap();
            assert_ne!(
                first[0].roots, next[0].roots,
                "different native cache/config/geometry domain is not a clone"
            );
        }
        assert_eq!(cache.slots.len(), 5);
    }
    #[test]
    fn lazy_caption_keeps_full_hidden_parent_and_frozen_old_window_survives_later_graph_detach() {
        let mut p = Program::default();
        p.begin();
        let hidden = p
            .immediate("independent full hidden", vec![1, 5, 4], Dtype::Float32)
            .unwrap();
        let index = p
            .immediate("independent gather indices", vec![2], Dtype::Int32)
            .unwrap();
        let selected = p
            .storage(
                "independent lazy gather",
                vec![1, 2, 4],
                Dtype::Float32,
                union([&hidden, &index]),
                false,
                false,
            )
            .unwrap();
        p.window(
            IosPhase::Denoise,
            "independent lazy-caption frontier",
            &BTreeSet::new(),
            0,
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            capacity_sum(&p.prefix, &p.prefix.windows[0].created_roots).unwrap(),
            120,
            "full80+indices8+selected32 coexist"
        );
        p.evaluated_graph_frontier("independent successful nontracer eval", &[&selected])
            .unwrap();
        p.window(
            IosPhase::Decode,
            "independent later graph frontier",
            &selected.roots,
            0,
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            capacity_sum(&p.prefix, &p.prefix.windows[0].created_roots).unwrap(),
            120,
            "do not retroactively erase a prior source window"
        );
        assert_eq!(
            capacity_sum(&p.prefix, &p.prefix.windows[1].resident_roots).unwrap(),
            32
        );
        assert!(
            p.prefix
                .workspaces
                .iter()
                .any(|w| w.family.contains("NOT-deduced-from-graph-detach")),
            "logical detach is NOT lastData/cache/process/physical release evidence"
        );
    }
    #[test]
    fn initial_noise_and_all_eps_are_distinct_full_resolution_source_factories() {
        let mut p = Program::default();
        p.begin();
        let (noise, eps) =
            draw_noise_sources(&mut p, vec![1, 3, 8, 12], &SamplerConfig::distill_4step()).unwrap();
        assert_eq!(noise.shape, vec![1, 3, 8, 12]);
        assert_eq!(eps.len(), 3);
        assert!(eps
            .iter()
            .all(|e| e.shape == [1, 3, 8, 12] && e.dtype == Dtype::Float32));
        let mut roots = noise.roots.clone();
        for e in &eps {
            roots.extend(e.roots.iter().copied());
        }
        assert_eq!(
            roots.len(),
            4,
            "same seed/shape is not source-factory alias or per-step-only eps"
        );
        assert_eq!(
            capacity_sum(&p.prefix, &roots).unwrap(),
            4608,
            "four FULL1152B draws, never tile-sized"
        );
        assert_eq!(
            p.prefix.workspaces.len(),
            4,
            "all RNG key/split native obligations remain explicit, not priced0"
        );
    }
    #[test]
    fn row_strip_full_accumulator_and_full_normalizer_remain_coincident_not_tile_sized() {
        let mut p = Program::default();
        p.begin();
        let tile = p
            .immediate(
                "independent velocity tile",
                vec![1, 3, 2, 3],
                Dtype::Bfloat16,
            )
            .unwrap();
        let row = accumulate(
            &mut p,
            "independent row",
            None,
            &tile,
            Placement {
                axis: 3,
                start: 1,
                extent: 3,
                total: 7,
            },
        )
        .unwrap();
        let full = accumulate(
            &mut p,
            "independent full",
            None,
            &row,
            Placement {
                axis: 2,
                start: 1,
                extent: 2,
                total: 5,
            },
        )
        .unwrap();
        assert_eq!(row.shape, vec![1, 3, 2, 7]);
        assert_eq!(full.shape, vec![1, 3, 5, 7]);
        let before = p.prefix.capacities.len();
        let out = full_normalize(&mut p, &full, 5, 7).unwrap();
        assert_eq!(out.shape, vec![1, 3, 5, 7]);
        assert_eq!(out.dtype, Dtype::Bfloat16);
        let allocated = p.prefix.capacities[before..]
            .iter()
            .map(|c| (c.shape.clone(), c.dtype, c.bytes))
            .collect::<Vec<_>>();
        assert_eq!(
            allocated,
            vec![
                (vec![1, 1, 5, 1], Dtype::Float32, 20),
                (vec![1, 1, 1, 7], Dtype::Float32, 28),
                (vec![1, 1, 5, 7], Dtype::Float32, 140),
                (vec![1, 1, 5, 7], Dtype::Bfloat16, 70),
                (vec![1, 3, 5, 7], Dtype::Bfloat16, 210)
            ]
        );
    }
    #[test]
    fn composite_consumer_abort_keeps_same_prefix_and_prior_whole_case_only_once() {
        let calls = std::cell::Cell::new(0);
        let mut observer = |id: u64, _: &[Condition], _: &Prefix, _: &IosPricing, _: bool| {
            calls.set(calls.get() + 1);
            if id == 1 {
                Err(Error::Io(std::io::Error::from_raw_os_error(13)))
            } else {
                Ok(())
            }
        };
        let mut unified = Unified::new(&mut observer);
        let mut first = Program::default();
        first.begin();
        first
            .immediate("independent first", vec![3], Dtype::Float32)
            .unwrap();
        first
            .window(
                IosPhase::Load,
                "independent first window",
                &BTreeSet::new(),
                0,
                None,
                None,
            )
            .unwrap();
        let base = BaseContext {
            load_resident: 0,
            load_held: 0,
            mint_resident: 0,
            mint_held: 0,
            decode_resident: 0,
            decode_held: 0,
        };
        let route = Route {
            mode: super::super::Mode::Text,
            width: 16,
            height: 16,
            count: 1,
            execution: Default::default(),
        };
        let pricing = pid_source_prices(&first.prefix, &base, &route, 4, "independent").unwrap();
        unified
            .commit(pricing, vec![], Arc::new(first.prefix), false)
            .unwrap();
        assert_eq!(unified.pricing.windows.len(), 1);
        assert_eq!(unified.pricing.windows[0].scratch_bytes, 12);
        assert_eq!(unified.pricing.windows[0].label, "independent first window");
        let expected = unified.pricing.windows.clone(); // frozen successful prior commit, not the failing producer's result
        let mut current = Program::default();
        current.begin();
        current
            .immediate("independent pending", vec![5], Dtype::Float32)
            .unwrap();
        current
            .window(
                IosPhase::Denoise,
                "independent current prefix",
                &BTreeSet::new(),
                0,
                None,
                None,
            )
            .unwrap();
        let error = unified
            .commit(
                IosPricing::default(),
                vec![],
                Arc::new(current.prefix),
                true,
            )
            .unwrap_err();
        assert!(matches!(error,Error::Io(ref io) if io.raw_os_error()==Some(13)));
        assert_eq!(calls.get(), 2);
        assert_eq!(unified.attempted_case_id, Some(1));
        assert_eq!(
            unified.pricing.windows, expected,
            "failed current case never commits/replays; prior whole windows remain"
        );
        assert_eq!(
            unified.pending.windows[0].site,
            "independent current prefix"
        );
    }
}
