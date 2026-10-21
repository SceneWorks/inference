//! Private native096 requested-allocation evaluator. This is NOT a physical upper.
//! Conditional request branches are evaluated from the SAME operation inputs/call attributes.
//! Unknown selectors/ABI/host/retirement/driver costs remain named obligations, never numeric zero.
use super::*;

pub(super) const AUTHORITY: &str =
    "1c1396ab6122ef23a06d7341ab306486403c77ef885268c5e4adace1b4b28bc3";
// Native096 source-conditional descriptor/Data/lease relations, NEVER observed pointer IDs.
// These metadata records own no native Array/shared_ptr, so collecting them cannot change donation.
#[derive(Clone, Debug)]
pub(super) enum ValueRule {
    Opaque(Slot),
    HeaderRoot(Slot),
    External,
    Scalar,
    HostInput,
    FreshMetal(Slot),
    NormalF32,
    Cast(usize),
    Unary(usize),
    NormLike(usize, bool),
    Binary(usize, usize),
    UnboundCompiled(Vec<usize>),
    Reshape(usize),
    Transpose(usize, Vec<usize>),
    Squeeze(usize, usize),
    Split(usize, usize, i32),
    #[cfg(test)]
    WeakBorrow(usize),
}
#[derive(Clone, Debug)]
pub(super) struct ValueFact {
    pub shape: Vec<i32>,
    pub dtype: Dtype,
    pub rule: ValueRule,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BoundaryKind {
    BareEval,
    ConditionalQuantumFinish,
    ReadbackWait,
    GraphDetach,
    #[cfg(test)]
    EndpointDrop,
}
#[derive(Clone, Debug)]
pub(super) struct BoundaryFact {
    pub kind: BoundaryKind,
    pub after_window: usize,
    pub site: String,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum DataKey {
    Opaque(usize),
    Allocated(usize),
    BorrowedWrapper(usize),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum LeaseKey {
    Opaque(usize),
    AllocatorRequest(usize),
}
#[derive(Clone, Debug)]
pub(super) struct DescriptorState {
    pub shape: Vec<i32>,
    pub native_shape_bound: bool,
    pub dtype: Dtype,
    pub descriptor: usize,
    pub data: BTreeSet<DataKey>,
    pub leases: BTreeSet<LeaseKey>,
    pub data_size: Option<u64>,
    pub strides: Option<Vec<u64>>,
    pub row_contiguous: Option<bool>,
    pub col_contiguous: Option<bool>,
    pub contiguous: Option<bool>,
    pub offset_bytes: Option<u64>,
    pub donation_rule: Option<&'static str>,
    pub same_data_parent: Option<usize>,
    pub fresh_request_bytes: Option<u64>,
}
#[derive(Clone, Copy, Debug)]
pub(super) enum LeaseLength {
    Empty,
    Unbound { requested: Option<u64> },
}
#[derive(Clone, Debug)]
pub(super) struct LeaseState {
    pub key: LeaseKey,
    pub length: LeaseLength,
    pub cached_reuse: CacheLengthBound,
    /// Alternative lifetime states, not two independent charges. NO retirement is inferred.
    pub may_be_queued: bool,
    pub may_enter_cache_after_unproved_retirement: bool,
}
#[derive(Clone, Debug, Default)]
pub(super) struct OwnerWindow {
    pub descriptors: Vec<DescriptorState>,
    pub leases: Vec<LeaseState>,
    pub boundary_count: usize,
    pub unknowns: BTreeSet<String>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CacheLengthBound {
    Empty,
    Cached {
        aligned_request: u64,
        upper_exclusive: u64,
    },
    Unbound(&'static str),
}

#[derive(Clone, Debug)]
pub(super) struct DataSizeProduct {
    pub source_input_candidates: Vec<usize>,
    pub native_input_mapping_bound: bool,
    pub itemsize: u64,
    pub output_slot: Slot,
}
#[derive(Clone, Debug)]
pub(super) struct Request {
    pub domain: &'static str,
    pub workspace: Option<usize>,
    pub source_value: Option<usize>,
    pub rule: &'static str,
    pub predicate: String,
    pub bytes: Option<u64>,
    pub existing_output: Option<Slot>,
    pub data_size_product: Option<DataSizeProduct>,
}
#[derive(Clone, Debug, Default)]
pub(super) struct WindowRequests {
    /// Positive upper of enumerated EXTRA components only. Request.domain distinguishes
    /// source-composed nbytes from explicit native Metal requests. Not a native TOTAL bound.
    pub known_extra_requested: u64,
    pub requests: Vec<Request>,
    pub unknowns: BTreeSet<String>,
    pub evaluated_workspaces: usize,
}
#[derive(Default)]
struct WorkRequests {
    common: u64,
    alternatives: BTreeMap<String, u64>,
    requests: Vec<Request>,
    unknowns: BTreeSet<String>,
    index: usize,
}
fn fail(message: &str) -> Error {
    Error::Msg(format!("PiD native requested allocation: {message}"))
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(|| fail("addition overflow"))
}
fn times(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| fail("multiplication overflow"))
}
fn div_up(a: u64, b: u64) -> Result<u64> {
    if b == 0 {
        return Err(fail("zero divisor"));
    }
    Ok(a / b + u64::from(a % b != 0))
}
fn count(shape: &[i32]) -> Result<u64> {
    shape.iter().try_fold(1, |n, &d| {
        let d = u64::try_from(d).map_err(|_| fail("negative shape"))?;
        times(n, d)
    })
}
fn bytes(shape: &[i32], dtype: Dtype) -> Result<u64> {
    times(count(shape)?, dtype_width(dtype))
}
fn data_size_product(elements: Option<u64>, itemsize: u64) -> Result<Option<u64>> {
    elements.map(|n| times(n, itemsize)).transpose()
}
fn input(w: &NativeWorkspace, i: usize) -> Result<&(Vec<i32>, Dtype)> {
    w.inputs
        .get(i)
        .ok_or_else(|| fail("missing actual operand"))
}
impl WorkRequests {
    fn common(&mut self, rule: &'static str, predicate: &str, n: u64) -> Result<()> {
        self.common = add(self.common, n)?;
        self.requests.push(Request {
            domain: "native096/Metal/requested-bytes",
            workspace: Some(self.index),
            source_value: None,
            rule,
            predicate: predicate.into(),
            bytes: Some(n),
            existing_output: None,
            data_size_product: None,
        });
        Ok(())
    }
    fn branch(&mut self, group: &str, rule: &'static str, predicate: &str, n: u64) {
        self.alternatives
            .entry(group.into())
            .and_modify(|v| *v = (*v).max(n))
            .or_insert(n);
        self.requests.push(Request {
            domain: "native096/Metal/requested-bytes",
            workspace: Some(self.index),
            source_value: None,
            rule,
            predicate: predicate.into(),
            bytes: Some(n),
            existing_output: None,
            data_size_product: None,
        });
    }
    fn composed_branch(&mut self, group: &str, rule: &'static str, predicate: &str, n: u64) {
        self.branch(group, rule, predicate, n);
        self.requests.last_mut().unwrap().domain =
            "source-composed-output-nbytes/native-allocation-UNBOUND";
    }
    fn composed_common(&mut self, rule: &'static str, predicate: &str, n: u64) -> Result<()> {
        self.common(rule, predicate, n)?;
        self.requests.last_mut().unwrap().domain =
            "source-composed-output-nbytes/native-allocation-UNBOUND";
        Ok(())
    }
    fn unknown(&mut self, rule: &'static str, reason: &str) {
        self.unknowns.insert(format!("{rule}: {reason}"));
        self.requests.push(Request {
            domain: "unbound-native-host-driver-physical",
            workspace: Some(self.index),
            source_value: None,
            rule,
            predicate: reason.into(),
            bytes: None,
            existing_output: None,
            data_size_product: None,
        });
    }
    fn unknown_data_size(
        &mut self,
        rule: &'static str,
        reason: &str,
        candidates: Vec<usize>,
        itemsize: u64,
        output_slot: Slot,
        mapping_bound: bool,
    ) -> Result<()> {
        // A typed partial evaluation, tied to THIS native call's input positions. No shape-only
        // substitution of numel for data_size, and None cannot silently become Some(0).
        let value = data_size_product(None, itemsize)?;
        self.unknown(rule, reason);
        let request = self.requests.last_mut().unwrap();
        request.bytes = value;
        request.data_size_product = Some(DataSizeProduct {
            source_input_candidates: candidates,
            native_input_mapping_bound: mapping_bound,
            itemsize,
            output_slot,
        });
        Ok(())
    }
    fn output(&mut self, prefix: &Prefix, w: &NativeWorkspace) -> Result<Option<Slot>> {
        if let Some(slot) = w.output_slot {
            let c = prefix
                .capacities
                .get(slot.0)
                .ok_or_else(|| fail("missing existing output slot"))?;
            self.requests.push(Request {
                domain: "existing-logical-source-slot/NOT-native-backing-certificate",
                workspace: Some(self.index),
                source_value: None,
                rule: "existing-output",
                predicate:
                    "SAME operation output already charged by source slot; not extra workspace"
                        .into(),
                bytes: Some(c.bytes),
                existing_output: Some(slot),
                data_size_product: None,
            });
            Ok(Some(slot))
        } else {
            self.unknown("output-correspondence","attempt/constructor has no successful output slot; never invent a successful allocation");
            Ok(None)
        }
    }
    fn layout_copies(&mut self, w: &NativeWorkspace, rule: &'static str) -> Result<()> {
        // Each native call tests its OWN actual strides. Do not mark a branch selected merely
        // because shape products fit. The positive branch is a conditional requested upper.
        for (i, (shape, dtype)) in w.inputs.iter().enumerate() {
            // AddMM's actual C/bias operand is passed strided without check_transpose/copy.
            if matches!(w.call, NativeCall::Addmm) && i == 0 {
                continue;
            }
            if matches!(w.call, NativeCall::Qmm { .. }) && i >= 4 {
                continue;
            }
            if matches!(w.call, NativeCall::Conv { .. }) && i >= 2 {
                continue;
            }
            self.common(rule,&format!("operand{i}: only IF native actual layout predicate fails; otherwise no copy request"),bytes(shape,*dtype)?)?;
        }
        self.unknown("layout-selector","actual strides/donation/compiled grouping not observed by shape-only source inputs; conditional copy components are not a selected native trace");
        Ok(())
    }
    fn finish(self) -> Result<(u64, Vec<Request>, BTreeSet<String>)> {
        let total = self
            .alternatives
            .values()
            .try_fold(self.common, |n, &v| add(n, v))?;
        Ok((total, self.requests, self.unknowns))
    }
}
fn simd_split(m: u64, n: u64, k: u64) -> Result<u64> {
    let tiles = times(div_up(m, 32)?, div_up(n, 32)?)?;
    if tiles == 0 {
        return Err(fail("empty split tile grid"));
    }
    let ratio = (k / 16) / tiles;
    let power = ratio
        .checked_next_power_of_two()
        .ok_or_else(|| fail("split power overflow"))?;
    Ok(power.max(2).min(32))
}
fn nax_split(k: u64) -> Result<u64> {
    let partition = if k <= 1024 {
        k / 2
    } else if k <= 2048 {
        1024
    } else if k <= 4096 {
        2048
    } else {
        4096
    };
    div_up(k, partition)
}
fn qmv_limits(k: u64, n: u64) -> [u64; 3] {
    // Native get_qmv_batch_limit: architecture 'd', gen13/14 other, later other.
    if k <= 2048 && n <= 2048 {
        [32, 14, 18]
    } else if k <= 4096 && n <= 4096 {
        [18, 10, 12]
    } else {
        [12, 6, 10]
    }
}
fn qmm_split(m: u64, n: u64, k: u64, group: u64) -> Result<u64> {
    let tiles = times(div_up(n, 32)?, div_up(m, 32)?)?;
    if tiles == 0 || group == 0 {
        return Err(fail("empty quantized tile/group"));
    }
    let mut p = (512 / tiles).max(1).min(k / group);
    while p > 1 && k % times(p, group)? != 0 {
        p -= 1;
    }
    Ok(p)
}
fn gemm_extra(
    r: &mut WorkRequests,
    m: u64,
    n: u64,
    k: u64,
    batch: u64,
    complex: bool,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        r.unknown(
            "gemm-output",
            "empty/K0 fill, scalar/beta and copy branch require actual call binding",
        );
        return Ok(());
    }
    // Preserve actual batched form AND the source-conditional contiguous batch collapse.
    for (cm, cb, form) in [
        (m, batch, "actual batch form"),
        (
            times(m, batch)?,
            1,
            "only IF native contiguous batch-collapse predicates hold",
        ),
    ] {
        if cm.min(n) == 1 {
            r.branch("gemm-dispatch","gemm-output",&format!("{form}; native min(M,N)==1 routes GEMV before all split-K selectors; no explicit split-buffer malloc in gemv_axbpy"),0);
            continue;
        }
        if cb == 1 {
            let small = times(div_up(cm, 16)?, div_up(n, 16)?)?;
            if k / 16 >= 8 && k >= cm.max(n) && small <= 2048 {
                let p = simd_split(cm, n, k)?;
                r.branch(
                    "gemm-dispatch",
                    "gemm-simd-splitk",
                    &format!(
                        "{form}; !use_nax && device threshold (1024 or2048) admits {small} tiles"
                    ),
                    times(times(times(p, cm)?, n)?, if complex { 8 } else { 4 })?,
                )
            }
            if !complex
                && (k >= times(3, cm.max(n))? || (cm.max(n) <= 1024 && k > times(2, cm.max(n))?))
            {
                let p = nax_split(k)?;
                r.branch(
                    "gemm-dispatch",
                    "gemm-nax-splitk",
                    &format!("{form}; metal::is_nax_available && TF32/dtype selector admits NAX"),
                    times(times(times(p, cm)?, n)?, if complex { 8 } else { 4 })?,
                )
            }
        }
    }
    r.unknown("gemm-selection","hardware/TF32/actual strides remain conditional; all eligible explicit split branches above are bounded, not guessed selected");
    Ok(())
}
fn matrix(w: &NativeWorkspace) -> Result<(u64, u64, u64, u64)> {
    let a = &input(w, usize::from(matches!(w.call, NativeCall::Addmm)))?.0;
    let b = &input(w, 1 + usize::from(matches!(w.call, NativeCall::Addmm)))?.0;
    if a.is_empty() || b.is_empty() {
        return Err(fail("scalar matmul operand"));
    }
    let k = u64::try_from(*a.last().unwrap()).map_err(|_| fail("negative K"))?;
    let kb = if b.len() == 1 { b[0] } else { b[b.len() - 2] };
    if u64::try_from(kb).ok() != Some(k) {
        return Err(fail("actual matmul contraction differs"));
    }
    let m = if a.len() == 1 {
        1
    } else {
        u64::try_from(a[a.len() - 2]).map_err(|_| fail("negative M"))?
    };
    let n = if b.len() == 1 {
        1
    } else {
        u64::try_from(*b.last().unwrap()).map_err(|_| fail("negative N"))?
    };
    let batch = broadcast(
        &a[..a.len().saturating_sub(2)],
        &b[..b.len().saturating_sub(2)],
    )?;
    Ok((m, n, k, count(&batch)?))
}
fn reduction(
    r: &mut WorkRequests,
    output_bytes: u64,
    input_bytes: u64,
    reduced: u64,
    general_possible: bool,
) -> Result<()> {
    // Source conditional branches of the SAME reduction, not the sum of mutually exclusive
    // long-column/strided/all-reduce intermediates. GeneralReduce copy can precede replanning.
    let group = format!("reduction-plan-{}", r.requests.len());
    let blocks = if reduced >= 32768 { 128 } else { 32 };
    r.branch(
        &group,
        "reductions",
        "only IF this reduction executes native Metal long-column plan",
        times(blocks, output_bytes)?,
    );
    r.branch(
        &group,
        "reductions",
        "only IF this reduction executes native Metal strided two-pass plan",
        times(32, output_bytes)?,
    );
    if general_possible && output_bytes == 4 && input_bytes / 4 > 4096 {
        let rows = if input_bytes <= 1 << 26 { 128 } else { 4096 };
        r.branch(
            &group,
            "reductions",
            "only IF native Metal scalar F32 all-reduce",
            times(rows, 4)?,
        );
    }
    if general_possible {
        r.common(
            "reductions",
            "only IF GeneralReduce first copies input before replanning",
            input_bytes,
        )?;
    }
    if general_possible {
        r.common("reductions","only IF composed sum reaches native Metal Reduce::eval_gpu (mean division is a separate composed source output)",output_bytes.max(4))?;
    }
    r.unknown("reduction-plan","actual Metal contiguity/reduction plan/output accumulator dtype require caller binding; CPU reduction allocator expressions are NOT imported from these Metal formulas");
    Ok(())
}
fn conv_bias_static_possible(w: &NativeWorkspace, out: &Capacity, groups: i32) -> Result<bool> {
    let a = &input(w, 0)?.0;
    let f = &input(w, 1)?.0;
    let bias = input(w, 2)?;
    if a.len() != 4 || f.len() != 4 || out.shape.len() != 4 {
        return Err(fail("conv bias actual rank"));
    }
    if groups != 1 || bias.0 != vec![f[0]] || bias.1 != input(w, 0)?.1 {
        return Ok(false);
    }
    let c = count(&a[3..])?;
    let o = count(&f[..1])?;
    let large = times(times(count(&a[..1])?, count(&a[1..2])?)?, count(&a[2..3])?)? >= 4096;
    if f[1] == 3 && f[2] == 3 && c % 32 == 0 && o % 32 == 0 && large && add(c, o)? >= 256 {
        return Ok(false);
    }
    let out_large = times(
        times(count(&out.shape[..1])?, count(&out.shape[1..2])?)?,
        count(&out.shape[2..3])?,
    )? >= 256;
    Ok(((c <= 4 || c % 16 == 0) && (o <= 16 || o % 16 == 0))
        || (c % 16 == 0 && o % 16 == 0)
        || out_large)
}
fn conv_extra(
    r: &mut WorkRequests,
    w: &NativeWorkspace,
    out: &Capacity,
    padding: i32,
    groups: i32,
) -> Result<()> {
    let a = &input(w, 0)?.0;
    let f = &input(w, 1)?.0;
    if a.len() != 4 || f.len() != 4 || out.shape.len() != 4 || groups <= 0 {
        return Err(fail("actual conv rank/groups"));
    }
    let n = count(&a[..1])?;
    let ih = count(&a[1..2])?;
    let iw = count(&a[2..3])?;
    let c = count(&a[3..])?;
    let o = count(&f[..1])?;
    let oh = count(&out.shape[1..2])?;
    let ow = count(&out.shape[2..3])?;
    let pad = u64::try_from(padding).map_err(|_| fail("negative conv padding"))?;
    let t = dtype_width(input(w, 0)?.1);
    let wt = dtype_width(input(w, 1)?.1);
    if groups == 1
        && f[1] == 3
        && f[2] == 3
        && c % 32 == 0
        && o % 32 == 0
        && times(times(n, ih)?, iw)? >= 4096
        && add(c, o)? >= 256
    {
        // Caller binding fixes groups1,stride/dilation1,!flip and padding mode, so native
        // dispatcher Winograd geometry is actually source-selected (bias exact path refuses
        // then uses eager conv+add, whose source output slots are already separate).
        let hp = add(
            times(
                6,
                div_up(
                    add(ih, times(2, pad)?)?
                        .checked_sub(2)
                        .ok_or_else(|| fail("Winograd H"))?,
                    6,
                )?,
            )?,
            2,
        )?;
        let wp = add(
            times(
                6,
                div_up(
                    add(iw, times(2, pad)?)?
                        .checked_sub(2)
                        .ok_or_else(|| fail("Winograd W"))?,
                    6,
                )?,
            )?,
            2,
        )?;
        let tiles = times(times(n, div_up(oh, 6)?)?, div_up(ow, 6)?)?;
        r.common(
            "conv-winograd",
            "actual padded input; padded slice shares this SAME allocation",
            times(times(times(times(n, hp)?, wp)?, c)?, t)?,
        )?;
        r.common(
            "conv-winograd",
            "filter transform [64,C,O]",
            times(times(times(64, c)?, o)?, wt)?,
        )?;
        r.common(
            "conv-winograd",
            "input transform [64,Ntiles,C]",
            times(times(times(64, tiles)?, c)?, t)?,
        )?;
        r.common(
            "conv-winograd",
            "output transform [64,Ntiles,O]",
            times(times(times(64, tiles)?, o)?, t)?,
        )?;
    } else {
        let out_large = times(times(n, oh)?, ow)? >= 256;
        let implicit = (c <= 4 || c % 16 == 0) && (o <= 16 || o % 16 == 0);
        let general = c % 16 == 0 && o % 16 == 0 || out_large;
        if groups != 1 || (!implicit && !general) {
            let m = if o == 0 {
                return Err(fail("zero conv output channels"));
            } else {
                count(&out.shape)? / o
            };
            let k = count(f)? / o;
            r.common(
                "conv-unfold",
                "actual explicit dispatcher: unfolded [M,K*groups]",
                times(
                    times(
                        times(m, k)?,
                        u64::try_from(groups).map_err(|_| fail("groups"))?,
                    )?,
                    t,
                )?,
            )?;
            if groups != 1 {
                r.common(
                    "conv-unfold",
                    "grouped filter-transpose contiguous copy",
                    bytes(f, input(w, 1)?.1)?,
                )?;
            }
            gemm_extra(r, m, o, k, 1, false)?;
        }
    }
    r.unknown("conv-native","nested kernel/driver/JIT and exact input-layout prep remain separate; no physical charge/release bound");
    Ok(())
}
pub(super) fn evaluate(prefix: &Prefix, window: &SourceWindow) -> Result<WindowRequests> {
    let work = prefix
        .workspaces
        .get(window.workspace_start..window.workspace_end)
        .ok_or_else(|| fail("invalid window workspace interval"))?;
    let mut result = WindowRequests::default();
    for (offset, w) in work.iter().enumerate() {
        let mut r = WorkRequests {
            index: window.workspace_start + offset,
            ..Default::default()
        };
        let output = r.output(prefix, w)?;
        let out = output.map(|s| &prefix.capacities[s.0]);
        if output.is_none() && !matches!(w.family,"native-packed-embedding-layout-and-backend-domain"|"native-retired-Data-queue-cache-process-ownership-NOT-deduced-from-graph-detach") {
            r.unknown("refused-prefix","operation has no successful output source slot; do NOT fabricate later workspace allocations");
            let (n,requests,unknowns)=r.finish()?;
            result.known_extra_requested=add(result.known_extra_requested,n)?;
            result.requests.extend(requests);result.unknowns.extend(unknowns);result.evaluated_workspaces+=1;continue;
        }
        match w.family {
            "native-affine-dequantize-selected-rows"=>{
                let source=w.affine_rows.as_ref().ok_or_else(||fail("affine output lacks exact gathered-row source proof"))?;
                if Some(source.output_slot)!=output {return Err(fail("affine output counted under different source slot"));}
                for (i,gather) in source.gathers.iter().enumerate() {
                    let actual=input(w,i)?;
                    if gather.squeezed_shape!=actual.0 || gather.layout!=GatherRowLayout::Native096GatherSetDataThenSqueeze {return Err(fail("affine input rank/flag proof changed"));}
                    r.requests.push(Request {domain:"native096/Metal/requested-bytes",workspace:Some(r.index),source_value:None,rule:"affine-gathered-row-copy",predicate:format!("input{i}: matched native Gather/set_data then axis2 Squeeze preserves row_contiguous; NO conditional copy request on normal Metal branch"),bytes:Some(0),existing_output:Some(gather.gather_slot),data_size_product:None});
                }
                let codes=input(w,0)?;let scales=input(w,1)?;let out=out.ok_or_else(||fail("affine missing output"))?;
                let packed=count(&codes.0)?;let e=count(&out.shape)?;
                let parts=32/u64::try_from(source.bits).map_err(|_|fail("affine bits"))?;
                let shifts=times(times(times(parts,2)?,packed)?,4)?; // left+right UInt32 outputs
                let concat=times(e,4)?;
                let promoted=promote_dtype(Dtype::Uint32,scales.1);
                let cast=if promoted!=Dtype::Uint32 {times(e,dtype_width(promoted))?}else{0};
                let product=times(e,dtype_width(promoted))?;
                let mut fallback=add(add(shifts,concat)?,add(cast,product)?)?;
                if promoted!=out.dtype {fallback=add(fallback,times(e,dtype_width(promoted))?)?;}
                r.composed_branch("affine-dispatch","affine-composed","IF CPU primal composed fallback: per-pack left/right shifts, concat, promoted codes cast/product; final add/cast is SAME existing output",fallback);
                r.unknown("affine-CPU-Custom","fallback scalar shift/mask/shape objects and Custom transformed derivative graphs remain unbound; conditional primal allocator components above do NOT certify transformations/physical memory");
            }
            "native-matmul-backend-and-queue"|"native-addmm-fused-epilogue"=>{
                if !matches!(w.call,NativeCall::Matmul|NativeCall::Addmm) {return Err(fail("unbound matmul call attributes"));}
                r.layout_copies(w,"gemm-layout-copy")?;let (m,n,k,b)=matrix(w)?;gemm_extra(&mut r,m,n,k,b,out.is_some_and(|o|o.dtype==Dtype::Complex64))?;
            }
            "native-quantized-matmul-backend-and-packed-layout"=>{
                let NativeCall::Qmm {group_size,bits,transpose,bias_output_slot}=w.call else {return Err(fail("unbound QMM group/bits/transpose"));};
                if !matches!(bits,4|8)||group_size!=64 {return Err(fail("QMM differs from actual source factory"));}
                r.layout_copies(w,"qmm-preparation")?;
                let x=&input(w,0)?.0;let weight=&input(w,1)?.0;let out=out.ok_or_else(||fail("QMM missing output"))?;
                let k=u64::try_from(*x.last().ok_or_else(||fail("QMM scalar"))?).map_err(|_|fail("QMM K"))?;
                if k==0||weight.len()!=2 {return Err(fail("QMM source weight rank/K"));}
                let n=u64::try_from(*out.shape.last().unwrap()).map_err(|_|fail("QMM N"))?;
                if n==0 {return Err(fail("empty QMM N"));}
                if let Some(slot)=bias_output_slot {
                    let bias=input(w,4)?;let final_out=prefix.capacities.get(slot.0).ok_or_else(||fail("QMM final bias output source missing"))?;
                    r.requests.push(Request {domain:"existing-logical-source-slot/NOT-native-backing-certificate",workspace:Some(r.index),source_value:None,rule:"qmm-preparation",predicate:"IF exact output-bias factory is accepted: final SAME output slot; otherwise generic QMM + existing bias add slots".into(),bytes:Some(final_out.bytes),existing_output:Some(slot),data_size_product:None});
                    if bias.0==vec![i32::try_from(n).map_err(|_|fail("QMM bias N"))?] && bias.1==out.dtype {
                        r.common("qmm-preparation","only IF exact epilogue is enabled/accepted, native output_bias_ true, and bias row-contiguity fails",bytes(&bias.0,bias.1)?)?;
                    }
                    r.unknown("qmm-bias-factory","exact native output-bias admission additionally checks actual M/B/tile and int ABI; output-bias branch bypasses split-K. Generic split/reduction components below are conditional on exact path omitted/rejected, never assumed bias absent");
                }
                let flat_m=count(x)?/k;
                let row_m=if x.len()>=2 {u64::try_from(x[x.len()-2]).map_err(|_|fail("QMM M"))?}else{1};
                let mut split_extra=0;let mut max_parts=0;
                for (m,form) in [(flat_m,"IF row-contiguous non_batched flatten"),(row_m,"actual last-two-axis M when non_batched predicate fails")] {
                    if m==0 {return Err(fail("empty QMM M"));}
                    let batch=count(&out.shape)?/m/n;
                    let p=qmm_split(m,n,k,64)?;
                    let limits=qmv_limits(k,n);
                    if transpose && batch==1 && p>1 && limits.iter().any(|&limit|m>=limit) {
                        let extra=times(p,out.bytes)?;split_extra=split_extra.max(extra);max_parts=max_parts.max(p);
                        r.branch("quantized-dispatch","qmm-splitk",&format!("{form}; M >= device vector_limit and B==1, transpose=true, AND generic no-fused-output-bias path selected"),extra);
                    } else if !transpose && m<4 && k>=1024 {
                        let p=if k>8192 {32}else{8};let extra=times(p,out.bytes)?;
                        split_extra=split_extra.max(extra);max_parts=max_parts.max(p);
                        r.branch("quantized-dispatch","qvm-splitk",&format!("{form}; vector M<vector_limit, transpose=false"),extra);
                    }
                }
                if split_extra!=0 {reduction(&mut r,out.bytes,split_extra,max_parts,false)?;}
                r.unknown("QMM-selector","actual row-contiguity determines non_batched M; device vector_limit/hardware and downstream reduction plan remain conditional");
            }
            "native-convolution-dispatch-and-exact-or-composed-bias"=>{
                let NativeCall::Conv {padding,groups,has_bias,plain_output}=w.call else {return Err(fail("unbound actual stride1/dilation1/no-flip conv call"));};
                r.layout_copies(w,"conv-output-layout")?;
                let plain=prefix.capacities.get(plain_output.0).ok_or_else(||fail("missing plain conv source output"))?;
                if Some(plain_output)!=output {
                    r.requests.push(Request {domain:"existing-logical-source-slot/NOT-native-backing-certificate",workspace:Some(r.index),source_value:None,rule:"conv-output-layout",predicate:"IF eager native conv+bias fallback: plain conv and final bias output are EXISTING distinct source slots".into(),bytes:Some(plain.bytes),existing_output:Some(plain_output),data_size_product:None});
                }
                if has_bias {
                    let bias=input(w,2)?;
                    // Native conv promotes input/filter ONLY; mismatched bias dtype rejects
                    // exact epilogue and takes eager add. Do not invent bias-driven input casts.
                    if conv_bias_static_possible(w,plain,groups)? {
                        r.common("conv-output-layout","only IF exact epilogue enabled, host/int ABI + GPU acceptance pass, and actual output-bias row-contiguity fails",bytes(&bias.0,bias.1)?)?;
                    }
                    r.unknown("conv-bias-factory","runtime exact-epilogue flag/host int ABI/GPU acceptance remain conditional; native dtype/shape, Winograd and explicit dispatch rejection is respected, with plain conv+add existing slots");
                }
                if let Some(out)=out {conv_extra(&mut r,w,out,padding,groups)?;}
            }
            "CONDITION metal-nongrad-nologsumexp else native-sdpa-composed-fallback-materialized-scores"=>{
                if !matches!(w.call,NativeCall::SdpaNoMaskNoSinks) {return Err(fail("unbound SDPA mask/sinks/caller"));}
                let q=input(w,0)?;let k=input(w,1)?;
                if q.0.len()!=4||k.0.len()!=4||q.0[2]<=8 {return Err(fail("served SDPA rank/full-path geometry"));}
                let score=times(times(times(count(&q.0[..2])?,count(&q.0[2..3])?)?,count(&k.0[2..3])?)?,dtype_width(q.1))?;
                // Composed fallback q-scale, score matmul and non-donated softmax outputs.
                // Final attention output is the already-recorded output slot.
                let fallback=add(bytes(&q.0,q.1)?,times(score,2)?)?;
                r.composed_branch("SDPA-dispatch","sdpa-composed","IF CPU OR grad tracing OR logsumexp/failed-fast selector; qscale+scores+non-donated softmax",fallback);
                let fast=w.inputs.iter().try_fold(0,|n,(s,d)|add(n,bytes(s,*d)?))?;
                r.branch("SDPA-dispatch","sdpa-output-copies","IF fast full Metal path: max of three actual-last-stride copy branches",fast);
                r.unknown("sdpa-composed","GQA reshape/materialization, fallback nested GEMM/reductions, selected strides, compiled grouping and transforms require full source binding; vector path not imported here");
            }
            "native-sdpa-full-noNAX-padded-head80-attention-workspace"|"native-sdpa-full-or-NAX-gated-workspace"=>{
                r.unknown("sdpa-shader-storage","same SDPA output/copies already handled by dispatch record; threadgroup residency/private spills/NAX/driver remain unknown, not zero");
            }
            "native-fast-rms-kernel-or-fallback"|"native-explicit-softmax-not-flash"=>{
                let x=input(w,0)?;let n=bytes(&x.0,x.1)?;
                if let Some(out)=out {
                    r.requests.push(Request {domain:"native096/Metal/requested-bytes",workspace:Some(r.index),source_value:None,rule:"rmsnorm-softmax",predicate:"IF native no-copy predicate fails: contiguous_copy_gpu(x), output SHARES this allocation; SAME existing output slot".into(),bytes:Some(n),existing_output:output,data_size_product:None});
                    r.unknown_data_size("rmsnorm-softmax",&format!("IF actual contiguous/last-stride predicate passes AND non-donatable: malloc(actual input0.data_size*{}), not numel={}; logical output{} is not backing",dtype_width(x.1),count(&x.0)?,out.bytes),vec![0],dtype_width(x.1),output.unwrap(),true)?;
                }
                if w.family=="native-fast-rms-kernel-or-fallback" && x.1==Dtype::Float32 {
                    let last=count(&x.0[x.0.len().checked_sub(1).ok_or_else(||fail("RMS scalar"))?..])?;
                    if last==0 {return Err(fail("RMS empty axis"));}
                    let rows=count(&x.0)?/last;
                    r.composed_branch("normalization-dispatch","rmsnorm-softmax","IF F32 composed CPU fallback: square + mean + eps + rsqrt + normalized product; final weight output is existing slot",add(times(n,2)?,times(times(rows,4)?,3)?)?);
                    reduction(&mut r,times(rows,4)?,n,last,true)?;
                }
            }
            "native-group-norm-layernorm-and-exact-epilogue"=>{
                let NativeCall::GroupNorm {groups:4,normalized_output}=w.call else {return Err(fail("unbound source GroupNorm group count"));};
                let x=input(w,0)?;let n=bytes(&x.0,x.1)?;
                if x.0.len()<2||*x.0.last().unwrap()%4!=0 {return Err(fail("actual group4 normalization shape"));}
                let groups=times(u64::try_from(x.0[0]).map_err(|_|fail("norm batch"))?,4)?;
                let f32bytes=times(count(&x.0)?,4)?;
                // Exact caller: group4; contiguous(x) before native exact epilogue, or
                // reshape/transpose/reshape -> LayerNorm(None,None) -> restore -> affine.
                let companions=add(bytes(&input(w,1)?.0,input(w,1)?.1)?,bytes(&input(w,2)?.0,input(w,2)?.1)?)?;
                let normalized=prefix.capacities.get(normalized_output.0).ok_or_else(||fail("composed norm output slot missing"))?;
                r.requests.push(Request {domain:"existing-logical-source-slot/NOT-native-backing-certificate",workspace:Some(r.index),source_value:None,rule:"rmsnorm-softmax",predicate:"composed LayerNorm restore output is SAME existing normalized slot; final affine output remains separately existing".into(),bytes:Some(normalized.bytes),existing_output:Some(normalized_output),data_size_product:None});
                let channels=*x.0.last().unwrap();
                let fast_possible=matches!(x.1,Dtype::Float16|Dtype::Bfloat16|Dtype::Float32)
                    && input(w,1)?.1==x.1 && input(w,2)?.1==x.1
                    && input(w,1)?.0==vec![channels] && input(w,2)?.0==vec![channels] && count(&x.0)?!=0;
                if fast_possible {
                    // The real factory REQUIRES same dtype. There is no bias-driven promotion.
                    r.unknown_data_size("groupnorm-layernorm","IF exact fast non-donated output: data_size of contiguous(x); original source input0 is not silently the normalized native descriptor",vec![0],dtype_width(x.1),output.unwrap(),false)?;
                    r.branch("groupnorm-dispatch","rmsnorm-softmax","IF exact GroupNormAffine passes runtime Metal/int ABI checks: conditional contiguous(x), weight/bias prep copies",add(n,companions)?);
                }
                r.unknown_data_size("groupnorm-layernorm","IF composed fast LayerNorm: backing data_size of grouped/reshaped native input; source input0 alone is not a bound native descriptor",vec![0],dtype_width(normalized.dtype),normalized_output,false)?;
                r.branch("groupnorm-dispatch","rmsnorm-softmax","IF composed group4: up to three reshape/LN backing requests not represented by final normalized slot",times(n,3)?);
                let full=times(f32bytes,if x.1==Dtype::Float32 {2}else{4})?;
                let small=times(groups,16)?; // mu,var,eps-add,rsqrt: four F32 group scalars
                r.composed_branch("groupnorm-CPU-components","rmsnorm-softmax","IF LayerNorm CPU fallback: F32 cast/center/square/product and group moments; LN T output already covered above",add(full,small)?);
                let axis=if groups==0 {return Err(fail("empty norm group"));}else{count(&x.0)?/groups};
                reduction(&mut r,times(groups,4)?,f32bytes,axis,true)?;
                reduction(&mut r,times(groups,4)?,f32bytes,axis,true)?;
                r.unknown("groupnorm-layernorm","actual exact-epilogue flag/fallback, reshape-copy selectors, fast data_size/donation and transformed graphs remain unbound; affine outputs are existing eager slots");
            }
            "native-rng-keys-splits-and-normal-workspace"=>{
                if !matches!(w.call,NativeCall::NormalNoLocScale) {return Err(fail("unbound actual RNG loc/scale overload"));}
                let x=input(w,0)?;if x.1!=Dtype::Float32 {return Err(fail("actual PiD RNG factory must be F32"));}
                let n=bytes(&x.0,x.1)?;
                r.common("rng","IF native Metal RandomBits eval: allocator::malloc(out.nbytes()) for UInt32 bits output",n)?;
                for name in ["bits-to-F32 promotion","uniform divide","uniform minimum","uniform range product","uniform low add","normal erfinv"] {
                    r.composed_common("rng",&format!("IF eager/non-donated source output: {name}; final sqrt2 product is EXISTING output; exact per-op data_size/donation remains unbound"),n)?;
                }
                r.unknown("rng-key-scalars","key/split objects, scalar/low/high/range Data, fusion/donation/retirement and driver terms remain unknown; None loc/scale source overload omission is not zero total RNG cost");
            }
            "source-eager-or-compiled-gated-epilogue"|"source-eager-or-compiled-modulate-epilogue"|"source-exact-epilogue-or-eager-gelu"=>{
                if let Some(out)=out {
                    r.requests.push(Request {domain:"native096/Metal/requested-bytes",workspace:Some(r.index),source_value:None,rule:"compiled-output",predicate:"IF noncontiguous/non-donated compiled output: malloc(out.nbytes) maps to EXISTING output, not extra".into(),bytes:Some(out.bytes),existing_output:output,data_size_product:None});
                    let out_elements=count(&out.shape)?;
                    let mut candidates=Vec::new();for (i,(shape,_)) in w.inputs.iter().enumerate() {if count(shape)?==out_elements {candidates.push(i);}}
                    r.unknown_data_size("compiled-output",&format!("IF contiguous representative in.size==out.size and non-donated: request=actual representative.data_size*{}; no first-label/numel substitution; actual fusion/representative predicate remains unbound",dtype_width(out.dtype)),candidates,dtype_width(out.dtype),output.unwrap(),false)?;
                }
                r.unknown("compiled-source","actual compiled input-list/representative/fusion/donation, C compile closure and graph caches remain unbound; source argument positions are NOT silently equated to native compiled input positions; no new output is charged twice");
            }
            "native-packed-embedding-layout-and-backend-domain"=>r.unknown("array-host-abi","packed constructor retains existing raw codes/scales/biases; C handles/ArrayDesc/Primitive heap not a new proven Data allocation"),
            "shared-qkv-adjacent-pair-rotation-layout"=>r.unknown("array-host-abi","actual rotation eager buffers are source slots; reshape/split aliases, host handles and compiled output backing need source correspondence"),
            "native-retired-Data-queue-cache-process-ownership-NOT-deduced-from-graph-detach"=>r.unknown("retirement-driver","source graph detach is NOT native Data/queue/cache/process release; retain all pending retirement obligations"),
            _=>return Err(fail(&format!("workspace family has no evaluator: {}",w.family))),
        }
        r.unknown("always-attached","CAPI/ArrayDesc/Data/SmallVector/closure/host allocator, shader occupancy/spills, command-buffer retention/JIT/driver/physical charge are NOT bounded by these request components");
        let (n, requests, unknowns) = r.finish()?;
        result.known_extra_requested = add(result.known_extra_requested, n)?;
        result.requests.extend(requests);
        result.unknowns.extend(unknowns);
        result.evaluated_workspaces += 1;
    }
    let mut excess_by_output = BTreeMap::<Slot, u64>::new();
    for request in &mut result.requests {
        let Some(product) = &request.data_size_product else {
            continue;
        };
        if !product.native_input_mapping_bound || product.source_input_candidates.len() != 1 {
            continue;
        }
        let Some(workspace) = request.workspace.and_then(|i| prefix.workspaces.get(i)) else {
            return Err(fail("owner resolution workspace"));
        };
        let Some(value) = workspace
            .input_values
            .get(product.source_input_candidates[0])
        else {
            continue;
        };
        let Some(descriptor) = window.owners.descriptors.get(*value) else {
            continue;
        };
        let Some(size) = descriptor.data_size else {
            continue;
        };
        let actual = data_size_product(Some(size), product.itemsize)?
            .ok_or_else(|| fail("lost proven native data_size"))?;
        request.bytes = Some(actual);
        request.existing_output = Some(product.output_slot);
        let nominal = prefix
            .capacities
            .get(product.output_slot.0)
            .ok_or_else(|| fail("owner resolved output slot"))?
            .bytes;
        // SAME output exact request versus already-counted logical nbytes, NOT a physical
        // upper-minus-upper or anchor lower. Alternative rules for one output max once.
        let extra = actual.checked_sub(nominal).unwrap_or(0);
        excess_by_output
            .entry(product.output_slot)
            .and_modify(|n| *n = (*n).max(extra))
            .or_insert(extra);
    }
    for extra in excess_by_output.values() {
        result.known_extra_requested = add(result.known_extra_requested, *extra)?;
    }
    result.unknowns = result
        .requests
        .iter()
        .filter(|r| r.bytes.is_none())
        .map(|r| format!("{}: {}", r.rule, r.predicate))
        .collect();
    result
        .unknowns
        .extend(window.owners.unknowns.iter().cloned());
    for lease in &window.owners.leases {
        let LeaseKey::AllocatorRequest(value) = lease.key else {
            continue;
        };
        if value < window.value_start || value >= window.value_end {
            continue;
        }
        if prefix.capacities.iter().any(|c| c.value == value) {
            continue;
        }
        // A native reshape-copy branch may have no separate logical capacity slot. Its actual
        // source request is charged once at creation, not every later queued/cache frontier.
        if let LeaseLength::Unbound { requested: Some(n) } = lease.length {
            result.known_extra_requested = add(result.known_extra_requested, n)?;
            result.requests.push(Request {domain:"native096/Metal/requested-bytes",workspace:None,source_value:Some(value),rule:"owner-view-copy",predicate:format!("IF prepare_reshape chooses copy for source value{value}; no logical output slot exists; later queue/cache lifetime remains unbound"),bytes:Some(n),existing_output:None,data_size_product:None});
        }
    }
    Ok(result)
}

// All extents below describe the named native096 operation AFTER successful evaluation on its
// specified path. They are conditional source facts, not pointer observations or scheduling proof.
fn row_strides(shape: &[i32]) -> Result<Vec<u64>> {
    let mut strides = vec![0; shape.len()];
    let mut n = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = n;
        n = times(
            n,
            u64::try_from(shape[i]).map_err(|_| fail("negative owner shape"))?,
        )?;
    }
    Ok(strides)
}
fn row_flag(shape: &[i32], strides: &[u64], column: bool) -> Result<bool> {
    if shape.len() != strides.len() {
        return Err(fail("owner stride rank"));
    }
    let mut n = 1;
    let order = if column {
        (0..shape.len()).collect::<Vec<_>>()
    } else {
        (0..shape.len()).rev().collect()
    };
    for i in order {
        if shape[i] != 1 && strides[i] != n {
            return Ok(false);
        }
        n = times(
            n,
            u64::try_from(shape[i]).map_err(|_| fail("negative owner dimension"))?,
        )?;
    }
    Ok(true)
}
fn owner_opaque(id: usize, f: &ValueFact) -> DescriptorState {
    DescriptorState {
        shape: f.shape.clone(),
        native_shape_bound: false,
        dtype: f.dtype,
        descriptor: id,
        data: BTreeSet::from([DataKey::Opaque(id)]),
        leases: BTreeSet::from([LeaseKey::Opaque(id)]),
        data_size: None,
        strides: None,
        row_contiguous: None,
        col_contiguous: None,
        contiguous: None,
        offset_bytes: None,
        donation_rule: None,
        same_data_parent: None,
        fresh_request_bytes: None,
    }
}
fn owner_fresh(id: usize, shape: Vec<i32>, dtype: Dtype) -> Result<DescriptorState> {
    let n = count(&shape)?;
    let strides = row_strides(&shape)?;
    let col = row_flag(&shape, &strides, true)?;
    Ok(DescriptorState {
        shape,
        native_shape_bound: true,
        dtype,
        descriptor: id,
        data: BTreeSet::from([DataKey::Allocated(id)]),
        leases: BTreeSet::from([LeaseKey::AllocatorRequest(id)]),
        data_size: Some(n),
        strides: Some(strides),
        row_contiguous: Some(true),
        col_contiguous: Some(col),
        contiguous: Some(true),
        offset_bytes: Some(0),
        donation_rule: None,
        same_data_parent: None,
        fresh_request_bytes: Some(times(n, dtype_width(dtype))?),
    })
}
fn same_data(mut state: DescriptorState, id: usize, parent: usize) -> DescriptorState {
    state.descriptor = id;
    state.same_data_parent = Some(parent);
    state.fresh_request_bytes = None;
    state.donation_rule = None;
    state
}
fn may_donate(
    mut state: DescriptorState,
    id: usize,
    predicate: &'static str,
    allowed: bool,
) -> DescriptorState {
    if !allowed {
        state.data.clear();
        state.leases.clear();
    }
    state.data.insert(DataKey::Allocated(id));
    state.leases.insert(LeaseKey::AllocatorRequest(id));
    state.descriptor = id;
    state.same_data_parent = None;
    state.donation_rule = allowed.then_some(predicate);
    if state.offset_bytes != Some(0) {
        state.offset_bytes = None;
    }
    state
}
fn weak_descriptor(parent: &DescriptorState, id: usize) -> DescriptorState {
    let mut state = parent.clone();
    state.descriptor = id;
    state.data = BTreeSet::from([DataKey::BorrowedWrapper(id)]);
    // unsafe_weak_copy constructs a DIFFERENT Data with no-op deleter over the SAME buffer.
    // Data-key inequality does not imply a new allocator lease or physical disjointness.
    state.same_data_parent = None;
    state.fresh_request_bytes = None;
    state.donation_rule = None;
    state
}
fn owner_broadcast(parent: &DescriptorState, shape: &[i32]) -> Result<DescriptorState> {
    if parent.native_shape_bound && parent.shape == shape {
        return Ok(parent.clone());
    }
    if !parent.native_shape_bound {
        let mut out = parent.clone();
        out.shape = shape.to_vec();
        if parent.data_size == Some(1) && count(shape)? > 1 {
            out.strides = Some(vec![0; shape.len()]);
            out.row_contiguous = Some(false);
            out.col_contiguous = Some(false);
            out.native_shape_bound = true;
        } else {
            out.strides = None;
            out.row_contiguous = None;
            out.col_contiguous = None;
        }
        return Ok(out);
    }
    if shape.len() < parent.shape.len() {
        return Err(fail("owner broadcast rank"));
    }
    let mut out = parent.clone();
    out.shape = shape.to_vec();
    if let Some(strides) = &parent.strides {
        let mut result = vec![0; shape.len()];
        let diff = shape.len() - parent.shape.len();
        for i in 0..parent.shape.len() {
            if parent.shape[i] != 1 && parent.shape[i] != shape[diff + i] {
                return Err(fail("owner broadcast dimensions"));
            }
            result[diff + i] = if parent.shape[i] == 1 { 0 } else { strides[i] };
        }
        out.strides = Some(result);
    }
    if count(shape)? > count(&parent.shape)? {
        out.row_contiguous = Some(false);
        out.col_contiguous = Some(false);
    }
    Ok(out)
}
fn inherited_output(
    parent: &DescriptorState,
    id: usize,
    dtype: Dtype,
    allowed: bool,
    rule: &'static str,
) -> Result<DescriptorState> {
    let mut out = parent.clone();
    out.dtype = dtype;
    out.fresh_request_bytes = parent
        .data_size
        .map(|n| times(n, dtype_width(dtype)))
        .transpose()?;
    Ok(may_donate(out, id, rule, allowed))
}
fn owner_binary(
    a: &DescriptorState,
    b: &DescriptorState,
    id: usize,
    dtype: Dtype,
) -> Result<DescriptorState> {
    let shape = if !a.native_shape_bound
        && a.data_size == Some(1)
        && b.native_shape_bound
        && !b.shape.is_empty()
    {
        b.shape.clone()
    } else if !b.native_shape_bound
        && b.data_size == Some(1)
        && a.native_shape_bound
        && !a.shape.is_empty()
    {
        a.shape.clone()
    } else {
        broadcast(&a.shape, &b.shape)?
    };
    let a = owner_broadcast(a, &shape)?;
    let b = owner_broadcast(b, &shape)?;
    let rule="native common binary: exact refcounts==1, equal itemsize, buffer_size<=out.nbytes+16384 plus branch layout/size; UNBOUND not assumed";
    if a.data_size == Some(1) && b.data_size == Some(1) {
        let mut out = owner_fresh(id, shape, dtype)?;
        out.native_shape_bound = a.native_shape_bound && b.native_shape_bound;
        out.data_size = Some(1);
        out.fresh_request_bytes = Some(dtype_width(dtype));
        out.strides = a.strides;
        out.contiguous = a.contiguous;
        out.row_contiguous = a.row_contiguous;
        out.col_contiguous = a.col_contiguous;
        return Ok(out);
    }
    if a.data_size == Some(1) && b.contiguous == Some(true) {
        return inherited_output(
            &b,
            id,
            dtype,
            dtype_width(b.dtype) == dtype_width(dtype),
            rule,
        );
    }
    if b.data_size == Some(1) && a.contiguous == Some(true) {
        return inherited_output(
            &a,
            id,
            dtype,
            dtype_width(a.dtype) == dtype_width(dtype),
            rule,
        );
    }
    let mut out = owner_fresh(id, shape, dtype)?;
    if (a.row_contiguous == Some(true) && b.row_contiguous == Some(true))
        || (a.col_contiguous == Some(true) && b.col_contiguous == Some(true))
    {
        out.data_size = if a.data_size == b.data_size {
            a.data_size
        } else {
            None
        };
        out.strides = if a.strides == b.strides {
            a.strides.clone()
        } else {
            None
        };
        out.row_contiguous = if a.row_contiguous == b.row_contiguous {
            a.row_contiguous
        } else {
            None
        };
        out.col_contiguous = if a.col_contiguous == b.col_contiguous {
            a.col_contiguous
        } else {
            None
        };
        out.contiguous = if a.contiguous == b.contiguous {
            a.contiguous
        } else {
            None
        };
        out.fresh_request_bytes = a
            .data_size
            .map(|n| times(n, dtype_width(dtype)))
            .transpose()?;
        for candidate in [&a, &b] {
            if dtype_width(candidate.dtype) == dtype_width(dtype) {
                out.data.extend(candidate.data.iter().cloned());
                out.leases.extend(candidate.leases.iter().cloned());
            }
        }
    } else {
        let e = count(&out.shape)?;
        for candidate in [&a, &b] {
            if candidate.row_contiguous != Some(false)
                && count(&candidate.shape)? == e
                && dtype_width(candidate.dtype) == dtype_width(dtype)
            {
                out.data.extend(candidate.data.iter().cloned());
                out.leases.extend(candidate.leases.iter().cloned());
                if candidate.data_size != Some(e) {
                    out.data_size = None;
                }
                if candidate.row_contiguous != Some(true) {
                    out.row_contiguous = None;
                    out.col_contiguous = None;
                    out.contiguous = None;
                    out.strides = None;
                }
            }
        }
    }
    out.donation_rule = Some(rule);
    out.offset_bytes = None;
    Ok(out)
}
pub(super) fn cached_length_bound(
    request: u64,
    page: Option<u64>,
    native_bits: u32,
) -> Result<CacheLengthBound> {
    if request == 0 {
        return Ok(CacheLengthBound::Empty);
    }
    if native_bits != 64 {
        return Ok(CacheLengthBound::Unbound(
            "native size_t domain not source-bound to64",
        ));
    }
    let Some(page) = page else {
        return Ok(CacheLengthBound::Unbound(
            "actual native vm_page_size unknown; no guessed4096/16384",
        ));
    };
    if page == 0 {
        return Err(fail("zero native page size"));
    }
    let aligned = if request > page {
        times(div_up(request, page)?, page)?
    } else {
        request
    };
    let Some(twice) = aligned.checked_mul(2) else {
        return Ok(CacheLengthBound::Unbound(
            "native2R overflow guard does not hold",
        ));
    };
    let Some(two_pages) = page.checked_mul(2) else {
        return Ok(CacheLengthBound::Unbound(
            "native2page overflow guard does not hold",
        ));
    };
    let Some(plus) = aligned.checked_add(two_pages) else {
        return Ok(CacheLengthBound::Unbound(
            "native R+2page overflow guard does not hold",
        ));
    };
    Ok(CacheLengthBound::Cached {
        aligned_request: aligned,
        upper_exclusive: twice.min(plus),
    })
}
fn owner_parent(states: &[DescriptorState], parent: usize) -> Result<&DescriptorState> {
    states
        .get(parent)
        .ok_or_else(|| fail("owner edge not prior source value"))
}
pub(super) fn join_owners(prefix: &Prefix, window: &SourceWindow) -> Result<OwnerWindow> {
    if window.value_start > window.value_end {
        return Err(fail("owner value chronology reversed"));
    }
    let facts = prefix
        .values
        .get(..window.value_end)
        .ok_or_else(|| fail("owner window value frontier"))?;
    let mut report = OwnerWindow::default();
    for (id, fact) in facts.iter().enumerate() {
        let state = match &fact.rule {
            ValueRule::Opaque(slot) => {
                let _ = prefix
                    .capacities
                    .get(slot.0)
                    .ok_or_else(|| fail("opaque owner source slot"))?;
                owner_opaque(id, fact)
            }
            ValueRule::HeaderRoot(slot) => {
                let source = prefix
                    .capacities
                    .get(slot.0)
                    .ok_or_else(|| fail("header owner root source slot"))?;
                if source.value != id {
                    return Err(fail("header source root mapped to another descriptor role"));
                }
                let mut out = owner_opaque(id, fact);
                out.native_shape_bound = true;
                out
            }
            ValueRule::External => owner_opaque(id, fact),
            ValueRule::Scalar => {
                // Shared Rust scalar uses [1]; native fallback literals can be rank0. BOTH
                // have source-proved data_size1, but don't invent a common native rank/stride.
                let mut out = owner_fresh(id, fact.shape.clone(), fact.dtype)?;
                out.native_shape_bound = false;
                out.data_size = Some(1);
                out.strides = None;
                out.data = BTreeSet::from([DataKey::Opaque(id)]);
                out.leases = BTreeSet::from([LeaseKey::Opaque(id)]);
                out.fresh_request_bytes = None;
                out
            }
            ValueRule::HostInput => {
                let mut out = owner_fresh(id, fact.shape.clone(), fact.dtype)?;
                // Constructor set_data proves descriptor extent, NOT malloc-vs-external backing.
                out.data = BTreeSet::from([DataKey::Opaque(id)]);
                out.leases = BTreeSet::from([LeaseKey::Opaque(id)]);
                out.fresh_request_bytes = None;
                out
            }
            ValueRule::FreshMetal(slot) => {
                let c = prefix
                    .capacities
                    .get(slot.0)
                    .ok_or_else(|| fail("fresh owner source slot"))?;
                if c.value != id {
                    return Err(fail(
                        "fresh native output mapped to another source descriptor",
                    ));
                }
                owner_fresh(id, fact.shape.clone(), fact.dtype)?
            }
            ValueRule::NormalF32 => {
                let mut out = owner_fresh(id, fact.shape.clone(), fact.dtype)?;
                out.data.insert(DataKey::Opaque(id));
                out.leases.insert(LeaseKey::Opaque(id));
                out.donation_rule=Some("native F32 normal bits/elementwise chain has source row extent; internal donation/data identity remains unbound");
                out
            }
            ValueRule::Cast(parent) => {
                let source = owner_parent(&report.descriptors, *parent)?;
                if source.dtype == fact.dtype {
                    source.clone()
                } else if source.contiguous == Some(true) {
                    inherited_output(source,id,fact.dtype,dtype_width(source.dtype)==dtype_width(fact.dtype),"AsType Vector: common is_donatable refcounts/itemsize/buffer_size guard; no assumed donation")?
                } else if source.contiguous == Some(false) {
                    let mut out = owner_fresh(id, source.shape.clone(), fact.dtype)?;
                    out.native_shape_bound = source.native_shape_bound;
                    if !source.native_shape_bound {
                        out.data_size = None;
                        out.strides = None;
                        out.fresh_request_bytes = None;
                    }
                    out
                } else {
                    let mut out = owner_fresh(id, source.shape.clone(), fact.dtype)?;
                    out.native_shape_bound = source.native_shape_bound;
                    if source.data_size != out.data_size {
                        out.data_size = None;
                        out.fresh_request_bytes = None;
                    }
                    out.row_contiguous = None;
                    out.col_contiguous = None;
                    out.contiguous = None;
                    out.strides = None;
                    if dtype_width(source.dtype) == dtype_width(fact.dtype) {
                        out.data.extend(source.data.iter().cloned());
                        out.leases.extend(source.leases.iter().cloned());
                        out.donation_rule = Some(
                            "AsType unknown Vector/General plus exact common donation predicate",
                        );
                    }
                    out
                }
            }
            ValueRule::Unary(parent) => {
                let source = owner_parent(&report.descriptors, *parent)?;
                if source.contiguous == Some(true) {
                    inherited_output(
                        source,
                        id,
                        fact.dtype,
                        true,
                        "unary common donation exact refcounts/itemsize/buffer_size guard",
                    )?
                } else if source.contiguous == Some(false) {
                    owner_fresh(id, source.shape.clone(), fact.dtype)?
                } else {
                    let mut out = owner_opaque(id, fact);
                    out.data.extend(source.data.iter().cloned());
                    out.leases.extend(source.leases.iter().cloned());
                    out.data.insert(DataKey::Allocated(id));
                    out.leases.insert(LeaseKey::AllocatorRequest(id));
                    out.donation_rule = Some("unary contiguity/donation unresolved");
                    out
                }
            }
            ValueRule::NormLike(parent, rms) => {
                let source = owner_parent(&report.descriptors, *parent)?;
                let mut no_copy = source.contiguous;
                if let Some(strides) = &source.strides {
                    let last = *strides
                        .last()
                        .ok_or_else(|| fail("native norm scalar source"))?;
                    if last != 1 {
                        no_copy = Some(false);
                    }
                    if *rms && source.shape.len() > 1 {
                        let n = source.shape.len();
                        if strides[n - 2] != 0
                            && strides[n - 2]
                                != u64::try_from(source.shape[n - 1])
                                    .map_err(|_| fail("norm dimension"))?
                            && source.shape[n - 2] != 1
                        {
                            no_copy = Some(false);
                        }
                    }
                } else {
                    no_copy = None;
                }
                if no_copy == Some(true) {
                    inherited_output(source,id,fact.dtype,true,"RMS/softmax direct array.is_donatable: ArrayDesc.use_count==1 AND Data.use_count==1; no common buffer-slack assumption")?
                } else if no_copy == Some(false) {
                    owner_fresh(id, source.shape.clone(), fact.dtype)?
                } else {
                    owner_opaque(id, fact)
                }
            }
            ValueRule::Binary(a, b) => owner_binary(
                owner_parent(&report.descriptors, *a)?,
                owner_parent(&report.descriptors, *b)?,
                id,
                fact.dtype,
            )?,
            ValueRule::UnboundCompiled(inputs) => {
                let mut out = owner_opaque(id, fact);
                for parent in inputs {
                    let source = owner_parent(&report.descriptors, *parent)?;
                    let weak = weak_descriptor(source, id);
                    out.data.extend(source.data.iter().cloned());
                    out.data.extend(weak.data);
                    out.leases.extend(source.leases.iter().cloned());
                }
                out.donation_rule=Some("actual compiled input-list/CSE/constants/weak wrappers/representative unknown; never identify source args with native inputs by shape");
                out
            }
            ValueRule::Reshape(parent) => {
                let source = owner_parent(&report.descriptors, *parent)?;
                if count(&source.shape)? != count(&fact.shape)? {
                    return Err(fail(
                        "owner reshape native elements differ; no singleton guess",
                    ));
                }
                if source.row_contiguous == Some(true) || count(&source.shape)? == 0 {
                    let mut out = same_data(source.clone(), id, *parent);
                    out.shape = fact.shape.clone();
                    out.strides = Some(row_strides(&out.shape)?);
                    out.col_contiguous =
                        Some(row_flag(&out.shape, out.strides.as_ref().unwrap(), true)?);
                    out
                } else {
                    let mut out = owner_fresh(id, fact.shape.clone(), fact.dtype)?;
                    out.data.extend(source.data.iter().cloned());
                    out.leases.extend(source.leases.iter().cloned());
                    if source.data_size != out.data_size {
                        out.data_size = None;
                    }
                    out.row_contiguous = None;
                    out.col_contiguous = None;
                    out.contiguous = None;
                    out.strides = None;
                    out.offset_bytes = None;
                    out.donation_rule=Some("prepare_reshape copy vs shared_buffer_reshape unresolved; not automatically a fresh allocation");
                    out
                }
            }
            ValueRule::Transpose(parent, axes) => {
                let source = owner_parent(&report.descriptors, *parent)?;
                if axes.len() != source.shape.len() {
                    return Err(fail("native owner transpose rank"));
                }
                let mut out = same_data(source.clone(), id, *parent);
                out.shape = axes
                    .iter()
                    .map(|&i| {
                        source
                            .shape
                            .get(i)
                            .copied()
                            .ok_or_else(|| fail("native owner transpose axis"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                out.strides = source
                    .strides
                    .as_ref()
                    .map(|s| axes.iter().map(|&i| s[i]).collect());
                if source.contiguous == Some(true)
                    && source.data_size == Some(count(&source.shape)?)
                {
                    if let Some(strides) = &out.strides {
                        out.row_contiguous = Some(row_flag(&out.shape, strides, false)?);
                        out.col_contiguous = Some(row_flag(&out.shape, strides, true)?);
                    }
                } else {
                    out.row_contiguous = None;
                    out.col_contiguous = None;
                }
                out
            }
            ValueRule::Squeeze(parent, axis) => {
                let source = owner_parent(&report.descriptors, *parent)?;
                if source.shape.get(*axis) != Some(&1) {
                    return Err(fail("native owner squeeze nonunit axis"));
                }
                let mut out = same_data(source.clone(), id, *parent);
                out.shape.remove(*axis);
                if let Some(strides) = &mut out.strides {
                    strides.remove(*axis);
                }
                out
            }
            ValueRule::Split(parent, axis, start) => {
                let source = owner_parent(&report.descriptors, *parent)?;
                if *axis >= source.shape.len()
                    || fact.shape.len() != source.shape.len()
                    || *start < 0
                {
                    return Err(fail("native owner split axis/start"));
                }
                let mut out = same_data(source.clone(), id, *parent);
                out.shape = fact.shape.clone();
                if let Some(strides) = &source.strides {
                    let mut data_size = 1;
                    for (&d, &stride) in out.shape.iter().zip(strides) {
                        if stride > 0 {
                            data_size = times(
                                data_size,
                                u64::try_from(d).map_err(|_| fail("negative split shape"))?,
                            )?;
                        }
                    }
                    out.data_size = Some(data_size);
                    out.row_contiguous = Some(row_flag(&out.shape, strides, false)?);
                    out.col_contiguous = Some(row_flag(&out.shape, strides, true)?);
                    out.offset_bytes = source
                        .offset_bytes
                        .map(|offset| {
                            add(
                                offset,
                                times(
                                    times(
                                        u64::try_from(*start).map_err(|_| fail("split start"))?,
                                        strides[*axis],
                                    )?,
                                    dtype_width(source.dtype),
                                )?,
                            )
                        })
                        .transpose()?;
                    out.contiguous = if data_size == 1 {
                        Some(true)
                    } else if source.data_size == Some(data_size) {
                        source.contiguous
                    } else {
                        source.contiguous.map(|v| {
                            v && (out.row_contiguous == Some(true)
                                || out.col_contiguous == Some(true))
                        })
                    };
                } else {
                    out.data_size = None;
                    out.offset_bytes = None;
                    out.row_contiguous = None;
                    out.col_contiguous = None;
                    out.contiguous = None;
                }
                out
            }
            #[cfg(test)]
            ValueRule::WeakBorrow(parent) => {
                weak_descriptor(owner_parent(&report.descriptors, *parent)?, id)
            }
        };
        report.descriptors.push(state);
    }
    let mut leases = BTreeMap::new();
    for state in &report.descriptors {
        for key in &state.leases {
            leases.entry(key.clone()).or_insert_with(|| {
                let requested = match key {
                    LeaseKey::AllocatorRequest(id) => report
                        .descriptors
                        .get(*id)
                        .and_then(|s| s.fresh_request_bytes),
                    LeaseKey::Opaque(_) => None,
                };
                let length = if requested == Some(0) {
                    LeaseLength::Empty
                } else {
                    LeaseLength::Unbound { requested }
                };
                let cached_reuse = match requested {
                    Some(n) => cached_length_bound(n, None, 64).unwrap_or(
                        CacheLengthBound::Unbound("native arithmetic/domain not established"),
                    ),
                    None => {
                        CacheLengthBound::Unbound("actual request and native page size unbound")
                    }
                };
                LeaseState {
                    key: key.clone(),
                    length,
                    cached_reuse,
                    may_be_queued: true,
                    may_enter_cache_after_unproved_retirement: true,
                }
            });
        }
    }
    report.leases = leases.into_values().collect();
    report.boundary_count = prefix
        .boundaries
        .iter()
        .filter(|b| b.after_window <= prefix.windows.len())
        .count();
    report.unknowns.insert("Source IDs/after-eval facts are NOT observed ArrayDesc/Data/buffer pointer identities, native strong counts, or all-instant scheduling witnesses".into());
    report.unknowns.insert("Declared source shapes still require original caller/source-epoch correspondence; after-eval extent rules are conditional on that match, not metadata-carrier observations".into());
    report.unknowns.insert("Lease lengths remain unbound: descriptor data_size and requested bytes are NOT actual buf.length; fresh/heap/external memory is NOT cached-reuse bound".into());
    report.unknowns.insert("GPU input/sibling Data keepalives and encoder temporaries can outlive graph detach/bare eval/drop; conditional existing quantum sync+bounded drain is NOT all-capture destruction or cache0 certificate".into());
    report.unknowns.insert("External latent/header roots may share larger caller buffers or retained cache values; logical frame numel is NOT their native data_size/length".into());
    report.unknowns.insert("Weak/no-op Data wrappers share a resource without owning its allocator lease; Data.use_count alone is NOT resource lifetime/disjointness proof".into());
    report.unknowns.insert("First/last serial source endpoints do not prove omitted middle native allocation/retirement multiplicity; warm epochs and baseline inventory still require correlated coverage".into());
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    // Synthetic source records test evaluator arithmetic/integrity only. They are NOT native,
    // process, physical, capture-correspondence or successful-model certificates.
    fn fixture(
        family: &'static str,
        inputs: Vec<(Vec<i32>, Dtype)>,
        shape: Vec<i32>,
        dtype: Dtype,
        call: NativeCall,
    ) -> (Prefix, SourceWindow) {
        let mut prefix = Prefix::default();
        prefix.capacities.push(Capacity {
            name: "literal output".into(),
            bytes: bytes(&shape, dtype).unwrap(),
            shape,
            dtype,
            immediate: false,
            possible_view_copy: false,
            parents: BTreeSet::new(),
            graph_attached: true,
            value: 0,
        });
        prefix.workspaces.push(NativeWorkspace {
            site: "literal site".into(),
            family,
            inputs,
            input_values: Vec::new(),
            affine_rows: None,
            output_slot: Some(Slot(0)),
            call,
        });
        let window = SourceWindow {
            phase: IosPhase::Decode,
            site: "literal window".into(),
            resident_roots: BTreeSet::new(),
            created_roots: BTreeSet::from([Slot(0)]),
            host: Vec::new(),
            workspace_start: 0,
            workspace_end: 1,
            operation_start: 0,
            operation_end: 0,
            requests: Default::default(),
            owners: Default::default(),
            value_start: 0,
            value_end: 0,
            serial_completed: 0,
            step: None,
            tile: None,
        };
        (prefix, window)
    }
    #[test]
    fn literal_gemm_and_qmm_partitions_are_source_arithmetic_not_coefficients() {
        assert_eq!(
            simd_split(8, 16, 1024).unwrap(),
            32,
            "SIMD native partitions must clamp next-power-of-two to32"
        );
        assert_eq!(nax_split(1024).unwrap(), 2);
        assert_eq!(nax_split(2048).unwrap(), 2);
        assert_eq!(nax_split(4096).unwrap(), 2);
        assert_eq!(nax_split(12288).unwrap(), 3);
        assert_eq!(
            qmm_split(16, 64, 1024, 64).unwrap(),
            16,
            "QMM split must respect native quantization-group divisibility"
        );
        assert_eq!(qmm_split(16, 64, 960, 64).unwrap(), 15);
        assert_eq!(qmv_limits(1024,64),[32,14,18],"QMM matrix/vector conditions must follow actual native hardware-conditional threshold domain");
        let (prefix, window) = fixture(
            "native-matmul-backend-and-queue",
            vec![
                (vec![8, 1024], Dtype::Bfloat16),
                (vec![1024, 16], Dtype::Bfloat16),
            ],
            vec![8, 16],
            Dtype::Bfloat16,
            NativeCall::Matmul,
        );
        let report = evaluate(&prefix, &window).unwrap();
        let mut reference = WorkRequests::default();
        reference.output(&prefix, &prefix.workspaces[0]).unwrap();
        assert_eq!(
            reference.finish().unwrap().0,
            0,
            "an existing output slot must never become an extra workspace charge"
        );
        // Two possible input copies:16384+32768; max SIMD16384 vs NAX1024, NOT sum.
        assert_eq!(
            report.known_extra_requested, 65536,
            "mutually exclusive native split branches must max once"
        );
        assert!(
            !report.unknowns.is_empty(),
            "conditional requested subtotal must not certify a complete native/physical bound"
        );
        let (vector, window) = fixture(
            "native-matmul-backend-and-queue",
            vec![
                (vec![1, 8192], Dtype::Bfloat16),
                (vec![8192, 2048], Dtype::Bfloat16),
            ],
            vec![1, 2048],
            Dtype::Bfloat16,
            NativeCall::Matmul,
        );
        let vector = evaluate(&vector, &window).unwrap();
        assert!(
            !vector
                .requests
                .iter()
                .any(|r| matches!(r.rule, "gemm-simd-splitk" | "gemm-nax-splitk")),
            "actual GEMV early route must not invent GEMM split intermediates"
        );
        let (prefix, window) = fixture(
            "native-quantized-matmul-backend-and-packed-layout",
            vec![
                (vec![16, 1024], Dtype::Bfloat16),
                (vec![64, 128], Dtype::Uint32),
                (vec![64, 16], Dtype::Bfloat16),
                (vec![64, 16], Dtype::Bfloat16),
            ],
            vec![16, 64],
            Dtype::Bfloat16,
            NativeCall::Qmm {
                group_size: 64,
                bits: 4,
                transpose: true,
                bias_output_slot: None,
            },
        );
        let report = evaluate(&prefix, &window).unwrap();
        // Four matrix-layout copies69632, intermediate32768, strided reduction65536.
        assert_eq!(
            report.known_extra_requested, 167936,
            "QMM reduction workspace must remain joined with its split intermediate"
        );
        let mut program = Program::default();
        program.begin();
        let mut weights = Sources::default();
        weights.tensors.insert(
            "mixed.weight".into(),
            program
                .immediate("mixed codes", vec![64, 128], Dtype::Uint32)
                .unwrap(),
        );
        weights.tensors.insert(
            "mixed.scales".into(),
            program
                .immediate("mixed scales", vec![64, 16], Dtype::Float32)
                .unwrap(),
        );
        weights.tensors.insert(
            "mixed.biases".into(),
            program
                .immediate("mixed biases", vec![64, 16], Dtype::Float32)
                .unwrap(),
        );
        let x = program
            .immediate("mixed input", vec![2, 1024], Dtype::Float16)
            .unwrap();
        let y = program
            .linear(&mut weights, "mixed", &x, false, true)
            .unwrap();
        assert_eq!(
            y.dtype,
            Dtype::Float32,
            "affine QMM must promote actual x/scales/biases before native allocation pricing"
        );
        let call = program.prefix.workspaces.last().unwrap();
        assert!(
            [0, 2, 3]
                .iter()
                .all(|&i| call.inputs[i].1 == Dtype::Float32),
            "native QMM input dtypes must match promoted factory casts"
        );
        weights.tensors.insert(
            "mixed.bias".into(),
            program
                .immediate("mixed output bias", vec![64], Dtype::Float16)
                .unwrap(),
        );
        let biased = program
            .linear(&mut weights, "mixed", &x, true, true)
            .unwrap();
        assert_eq!(
            biased.dtype,
            Dtype::Float32,
            "native fused output bias cannot silently change affine QMM promotion"
        );
        program
            .window(
                IosPhase::Denoise,
                "mixed QMM literal",
                &BTreeSet::new(),
                0,
                None,
                None,
            )
            .unwrap();
        let report = &program.prefix.windows.last().unwrap().requests;
        assert!(
            !report
                .requests
                .iter()
                .any(|r| r.predicate.contains("M >= device vector_limit")),
            "M2 cannot select a native QMM split when every actual vector limit is at least14"
        );
        assert!(
            !report
                .requests
                .iter()
                .any(|r| r.predicate.contains("native output_bias_ true")),
            "mismatched bias dtype statically rejects fused bias factory; do not invent bias copy"
        );
    }
    #[test]
    fn literal_winograd_sums_coincident_transforms_and_dispatches_by_actual_geometry() {
        let (prefix, window) = fixture(
            "native-convolution-dispatch-and-exact-or-composed-bias",
            vec![
                (vec![1, 64, 64, 128], Dtype::Bfloat16),
                (vec![128, 3, 3, 128], Dtype::Bfloat16),
            ],
            vec![1, 64, 64, 128],
            Dtype::Bfloat16,
            NativeCall::Conv {
                padding: 1,
                groups: 1,
                has_bias: false,
                plain_output: Slot(0),
            },
        );
        let report = evaluate(&prefix, &window).unwrap();
        let mut transforms = report
            .requests
            .iter()
            .filter(|r| r.rule == "conv-winograd")
            .map(|r| r.bytes.unwrap())
            .collect::<Vec<_>>();
        transforms.sort();
        assert_eq!(
            transforms,
            vec![1183744, 1982464, 1982464, 2097152],
            "Winograd source has FOUR simultaneous transform allocations"
        );
        assert_eq!(
            report.known_extra_requested, 8589312,
            "Winograd must SUM coexisting transforms, not take one maximum"
        );
        let (prefix, window) = fixture(
            "native-convolution-dispatch-and-exact-or-composed-bias",
            vec![
                (vec![1, 8, 8, 64], Dtype::Bfloat16),
                (vec![64, 3, 3, 64], Dtype::Bfloat16),
            ],
            vec![1, 8, 8, 64],
            Dtype::Bfloat16,
            NativeCall::Conv {
                padding: 1,
                groups: 1,
                has_bias: false,
                plain_output: Slot(0),
            },
        );
        let report = evaluate(&prefix, &window).unwrap();
        assert!(
            !report
                .requests
                .iter()
                .any(|r| matches!(r.rule, "conv-winograd" | "conv-unfold")),
            "3x3 alone cannot select Winograd or unfold"
        );
    }
    #[test]
    fn literal_rng_and_affine_fallbacks_keep_final_output_slot_once() {
        let (prefix, window) = fixture(
            "native-rng-keys-splits-and-normal-workspace",
            vec![(vec![1, 3, 2, 2], Dtype::Float32)],
            vec![1, 3, 2, 2],
            Dtype::Float32,
            NativeCall::NormalNoLocScale,
        );
        let report = evaluate(&prefix, &window).unwrap();
        assert_eq!(
            report.known_extra_requested, 336,
            "normal primal has seven possible source components beyond final source slot"
        );
        assert_eq!(
            report
                .requests
                .iter()
                .filter(|r| r.rule == "rng" && r.domain == "native096/Metal/requested-bytes")
                .count(),
            1,
            "only RandomBits allocator request is directly source-bound here"
        );
        assert_eq!(
            report
                .requests
                .iter()
                .filter(|r| r.rule == "rng" && r.domain.starts_with("source-composed"))
                .count(),
            6,
            "ordinary eager output sizes must not masquerade as bound native data_size allocations"
        );
        assert_eq!(
            report
                .requests
                .iter()
                .filter(|r| r.rule == "existing-output")
                .count(),
            1
        );
        let mut p = Program::default();
        p.begin();
        let ids = p.immediate("ids", vec![1, 13], Dtype::Int32).unwrap();
        let w = p.immediate("codes", vec![19, 16], Dtype::Uint32).unwrap();
        let s = p.immediate("scales", vec![19, 2], Dtype::Bfloat16).unwrap();
        let b = p.immediate("biases", vec![19, 2], Dtype::Bfloat16).unwrap();
        let (w, wp) = p.embedding_take_axis_zero("w", &w, &ids).unwrap();
        let (s, sp) = p.embedding_take_axis_zero("s", &s, &ids).unwrap();
        let (b, bp) = p.embedding_take_axis_zero("b", &b, &ids).unwrap();
        let out = p.gathered_affine_rows(&w, &s, &b, [wp, sp, bp], 4).unwrap();
        p.window(
            IosPhase::Denoise,
            "affine literal window",
            &BTreeSet::new(),
            0,
            None,
            None,
        )
        .unwrap();
        let report = &p.prefix.windows.last().unwrap().requests;
        assert_eq!(
            report.known_extra_requested, 26624,
            "affine CPU primal shift/concat/cast/product components must be evaluated"
        );
        assert!(
            report
                .requests
                .iter()
                .any(|r| r.rule == "affine-composed" && r.domain.starts_with("source-composed")),
            "CPU graph sizes must remain distinct from bound Metal allocation requests"
        );
        assert_eq!(
            report
                .requests
                .iter()
                .find(|r| r.rule == "existing-output")
                .unwrap()
                .existing_output,
            out.roots.iter().next().copied(),
            "affine evaluator must reuse SAME already-counted output"
        );
        assert_eq!(
            report
                .requests
                .iter()
                .filter(|r| r.rule == "affine-gathered-row-copy")
                .count(),
            3
        );
        assert!(
            report
                .requests
                .iter()
                .filter(|r| r.rule == "affine-gathered-row-copy")
                .all(|r| r.bytes == Some(0)),
            "zero is allowed ONLY for source-proved absent layout copy, never an unknown"
        );
        assert!(report
            .requests
            .iter()
            .any(|r| r.rule == "affine-CPU-Custom" && r.bytes.is_none()));
    }
    #[test]
    fn window_consumer_joins_requests_once_and_refused_prefix_never_prices_future_output() {
        let mut p = Program::default();
        p.begin();
        let a = p.immediate("a", vec![8, 1024], Dtype::Bfloat16).unwrap();
        let b = p.immediate("b", vec![1024, 16], Dtype::Bfloat16).unwrap();
        p.matmul("literal actual matmul", &a, &b).unwrap();
        p.window(
            IosPhase::Decode,
            "serial0/tile0/step0",
            &BTreeSet::new(),
            0,
            Some(0),
            None,
        )
        .unwrap();
        let first = p.prefix.windows[0].clone();
        let logical = capacity_sum(&p.prefix, &first.created_roots).unwrap();
        let price = source_window_price(&p.prefix, &first, 0, 0, "literal").unwrap();
        assert_eq!(
            price.scratch_bytes,
            logical + 65536,
            "REAL window pricing must join native extra request components exactly once"
        );
        p.workspace(
            "refused matrix attempt",
            "native-matmul-backend-and-queue",
            &[&a, &b],
        );
        p.window(
            IosPhase::Decode,
            "refused-prefix",
            &BTreeSet::new(),
            1,
            Some(0),
            None,
        )
        .unwrap();
        let last = &p.prefix.windows[1].requests;
        assert_eq!(
            last.known_extra_requested, 0,
            "refused output-less attempt must not price future successful split allocations"
        );
        assert!(last
            .requests
            .iter()
            .any(|r| r.rule == "refused-prefix" && r.bytes.is_none()));
        assert_eq!(
            p.prefix.windows[0].requests.known_extra_requested, 65536,
            "later refusal must preserve frozen earlier request frontier"
        );
    }
    #[test]
    fn unknown_native_data_size_is_not_numel_or_zero_and_output_is_not_added_twice() {
        for family in [
            "native-fast-rms-kernel-or-fallback",
            "native-explicit-softmax-not-flash",
            "source-eager-or-compiled-gated-epilogue",
            "source-eager-or-compiled-modulate-epilogue",
            "source-exact-epilogue-or-eager-gelu",
        ] {
            let (prefix, window) = fixture(
                family,
                vec![(vec![2, 4], Dtype::Float32), (vec![4], Dtype::Float32)],
                vec![2, 4],
                Dtype::Float32,
                NativeCall::Unbound,
            );
            let report = evaluate(&prefix, &window).unwrap();
            let expected = if family.starts_with("source-") {
                "compiled-output"
            } else {
                "rmsnorm-softmax"
            };
            assert!(
                report.requests.iter().any(|r| r.rule == expected
                    && r.bytes.is_none()
                    && r.domain == "unbound-native-host-driver-physical"),
                "actual native data_size term must not be replaced by numel or zero"
            );
            assert!(
                report
                    .requests
                    .iter()
                    .any(|r| r.existing_output == Some(Slot(0))),
                "native output must retain SAME source-slot identity"
            );
            assert!(report.requests.iter().any(|r|r.rule==expected&&r.data_size_product.as_ref().is_some_and(|p|p.itemsize==4&&p.output_slot==Slot(0))),"unbound native product must retain actual input positions/itemsize/output-slot binding");
            assert!(!report.unknowns.is_empty());
        }
    }
    #[test]
    fn requested_expression_overflow_and_unknown_family_fail_before_a_valid_price() {
        assert!(times(u64::MAX, 2).is_err());
        assert!(add(u64::MAX, 1).is_err());
        assert_eq!(
            data_size_product(None, 4).unwrap(),
            None,
            "unknown data_size must not collapse to zero"
        );
        assert_eq!(data_size_product(Some(136), 4).unwrap(), Some(544));
        assert!(data_size_product(Some(u64::MAX), 4).is_err());
        assert!(count(&[i32::MAX, i32::MAX, i32::MAX]).is_err());
        assert!(qmm_split(0, 1, 64, 64).is_err());
        assert!(nax_split(0).is_err());
        assert_eq!(div_up(u64::MAX, 2).unwrap(), 9223372036854775808);
        let (prefix, window) = fixture(
            "not-a-priced-family",
            vec![],
            vec![1],
            Dtype::Float32,
            NativeCall::Unbound,
        );
        assert!(
            evaluate(&prefix, &window).is_err(),
            "new workspace family must not silently get zero request cost"
        );
    }
    #[test]
    fn every_remaining_reached_family_has_explicit_requests_or_unbound_native_terms() {
        let family="CONDITION metal-nongrad-nologsumexp else native-sdpa-composed-fallback-materialized-scores";
        let heads = vec![(vec![1, 1, 16, 64], Dtype::Bfloat16); 3];
        let (prefix, window) = fixture(
            family,
            heads.clone(),
            vec![1, 1, 16, 64],
            Dtype::Bfloat16,
            NativeCall::SdpaNoMaskNoSinks,
        );
        let report = evaluate(&prefix, &window).unwrap();
        assert_eq!(report.known_extra_requested,6144,"SDPA must max fast-copy6144 versus composed3072, not sum mutually exclusive dispatches");
        for family in [
            "native-sdpa-full-noNAX-padded-head80-attention-workspace",
            "native-sdpa-full-or-NAX-gated-workspace",
        ] {
            let (prefix, window) = fixture(
                family,
                heads.clone(),
                vec![1, 1, 16, 64],
                Dtype::Bfloat16,
                NativeCall::SdpaNoMaskNoSinks,
            );
            let report = evaluate(&prefix, &window).unwrap();
            assert_eq!(
                report.known_extra_requested, 0,
                "SDPA detail record must not double-count same dispatch/output"
            );
            assert!(report
                .requests
                .iter()
                .any(|r| r.rule == "sdpa-shader-storage" && r.bytes.is_none()));
        }
        let (prefix, window) = fixture(
            "native-addmm-fused-epilogue",
            vec![
                (vec![16], Dtype::Bfloat16),
                (vec![8, 1024], Dtype::Bfloat16),
                (vec![1024, 16], Dtype::Bfloat16),
            ],
            vec![8, 16],
            Dtype::Bfloat16,
            NativeCall::Addmm,
        );
        assert_eq!(
            evaluate(&prefix, &window).unwrap().known_extra_requested,
            65536,
            "AddMM strided C has no invented bias layout-copy request"
        );
        let (prefix, window) = fixture(
            "native-group-norm-layernorm-and-exact-epilogue",
            vec![
                (vec![1, 8, 8, 64], Dtype::Float32),
                (vec![64], Dtype::Float32),
                (vec![64], Dtype::Float32),
            ],
            vec![1, 8, 8, 64],
            Dtype::Float32,
            NativeCall::GroupNorm {
                groups: 4,
                normalized_output: Slot(0),
            },
        );
        let report = evaluate(&prefix, &window).unwrap();
        assert!(
            report
                .requests
                .iter()
                .any(|r| r.rule == "reductions" && r.bytes.is_some()),
            "group4 norm must evaluate real reduction components, not only count metadata"
        );
        for family in [
            "native-packed-embedding-layout-and-backend-domain",
            "native-retired-Data-queue-cache-process-ownership-NOT-deduced-from-graph-detach",
            "shared-qkv-adjacent-pair-rotation-layout",
        ] {
            let (mut prefix, window) = fixture(
                family,
                vec![(vec![1, 8], Dtype::Float32)],
                vec![1, 8],
                Dtype::Float32,
                NativeCall::Unbound,
            );
            if family != "shared-qkv-adjacent-pair-rotation-layout" {
                prefix.workspaces[0].output_slot = None;
            }
            let report = evaluate(&prefix, &window).unwrap();
            assert_eq!(report.evaluated_workspaces, 1);
            assert!(
                report.requests.iter().any(|r| r.bytes.is_none()),
                "constructor/rotation/retirement native heap terms cannot become implicit zero"
            );
        }
    }

    fn source_fresh(p: &mut Program, name: &str, shape: Vec<i32>, dtype: Dtype) -> Tensor {
        // Explicit literal set_data(malloc(nbytes)) SOURCE rule, never a real Array/observation.
        let out = p
            .storage(name, shape, dtype, BTreeSet::new(), false, false)
            .unwrap();
        p.owner_rule(
            out.value,
            ValueRule::FreshMetal(*out.roots.iter().next().unwrap()),
        )
        .unwrap();
        out
    }
    #[test]
    fn owner_split_transpose_and_squeeze_share_data_not_descriptor_or_buffer_extent() {
        let mut p = Program::default();
        p.begin();
        let root = source_fresh(&mut p, "80-wide backing", vec![1, 1, 1, 80], Dtype::Float16);
        let narrow = slice(&mut p, "actual split view", &root, 3, 0, 72).unwrap();
        let transposed = p
            .transpose("view permutation", &narrow, &[0, 2, 1, 3])
            .unwrap();
        p.window(
            IosPhase::Decode,
            "alias literal window",
            &BTreeSet::new(),
            0,
            None,
            None,
        )
        .unwrap();
        let owner = &p.prefix.windows[0].owners;
        let a = &owner.descriptors[root.value];
        let b = &owner.descriptors[narrow.value];
        assert_eq!(
            b.data, a.data,
            "definite split view must unify native Data source root"
        );
        assert_ne!(
            b.descriptor, a.descriptor,
            "source view descriptor role is distinct, not a second Data allocation"
        );
        assert_eq!(
            b.data_size,
            Some(72),
            "Split descriptor data_size is not full buffer length"
        );
        assert_eq!(owner.descriptors[transposed.value].data, a.data);
        assert_eq!(
            owner.leases.len(),
            1,
            "views must not duplicate allocator lease"
        );
        assert!(
            matches!(
                owner.leases[0].length,
                LeaseLength::Unbound {
                    requested: Some(160)
                }
            ),
            "80-wide allocation request must not shrink to72-view or become known buf.length"
        );
        let ids = p
            .host_input("actual host ids", vec![1, 13], Dtype::Int32)
            .unwrap();
        let table = source_fresh(&mut p, "actual table", vec![19, 16], Dtype::Uint32);
        let (rows, proof) = p
            .embedding_take_axis_zero("actual gather", &table, &ids)
            .unwrap();
        p.window(
            IosPhase::Denoise,
            "squeeze literal window",
            &BTreeSet::new(),
            0,
            None,
            None,
        )
        .unwrap();
        let owner = &p.prefix.windows.last().unwrap().owners;
        assert_eq!(
            owner.descriptors[rows.value].same_data_parent,
            Some(p.prefix.capacities[proof.gather_slot.0].value),
            "actual axis2 Squeeze must share Gather Data"
        );
        assert_eq!(owner.descriptors[rows.value].data_size, Some(208));
    }
    #[test]
    fn owner_donation_is_conditional_and_weak_data_wrapper_adds_no_allocator_lease() {
        let mut p = Program::default();
        p.begin();
        let root = source_fresh(&mut p, "root", vec![2, 4], Dtype::Float32);
        let unary = p.unary("conditional sigmoid", &root).unwrap();
        let cast = p
            .cast("different element width", &root, Dtype::Bfloat16)
            .unwrap();
        let weak = p.owner_value(
            root.shape.clone(),
            root.dtype,
            ValueRule::WeakBorrow(root.value),
        );
        p.window(
            IosPhase::Denoise,
            "donation/weak literal",
            &BTreeSet::new(),
            0,
            None,
            None,
        )
        .unwrap();
        let owner = &p.prefix.windows[0].owners;
        let raw = &owner.descriptors[root.value];
        let result = &owner.descriptors[unary.value];
        assert!(
            result.data.contains(&DataKey::Allocated(root.value))
                && result.data.contains(&DataKey::Allocated(unary.value)),
            "unknown refcounts/buffer guard must retain donation AND fresh alternatives"
        );
        assert!(result.donation_rule.unwrap().contains("refcounts"));
        assert_eq!(
            owner.descriptors[cast.value].data,
            BTreeSet::from([DataKey::Allocated(cast.value)]),
            "unequal itemsize must exclude common donation branch"
        );
        assert_ne!(
            owner.descriptors[weak].data, raw.data,
            "weak wrapper is distinct Data object, not same Data identity"
        );
        assert_eq!(
            owner.descriptors[weak].leases, raw.leases,
            "unsafe weak wrapper must reuse parent lease without fresh allocator charge"
        );
        assert_eq!(
            owner.leases.len(),
            3,
            "weak Data identity must not imply fourth resource lease"
        );
    }
    #[test]
    fn owner_graph_detach_eval_drop_and_conditional_finish_do_not_retire_leases() {
        let mut p = Program::default();
        p.begin();
        let root = source_fresh(&mut p, "retained root", vec![2, 4], Dtype::Float32);
        p.window(
            IosPhase::Denoise,
            "before boundary",
            &BTreeSet::new(),
            0,
            None,
            None,
        )
        .unwrap();
        let before = p.prefix.windows[0].owners.leases.len();
        p.evaluated_graph_frontier("existing bare eval", &[&root])
            .unwrap();
        p.owner_boundary(
            BoundaryKind::EndpointDrop,
            "one logical endpoint scope ends",
        );
        p.owner_boundary(
            BoundaryKind::ConditionalQuantumFinish,
            "existing IF-enabled eval/sync/bounded drain",
        );
        p.window(
            IosPhase::Decode,
            "after boundaries",
            &BTreeSet::new(),
            1,
            None,
            None,
        )
        .unwrap();
        let after = &p.prefix.windows[1].owners;
        assert_eq!(
            after.leases.len(),
            before,
            "graph detach/bare eval/drop/conditional finish cannot become native retirement proof"
        );
        assert!(after.leases.iter().all(|l|l.may_be_queued&&l.may_enter_cache_after_unproved_retirement),"queue and later cache are alternatives for same lease, not disappearance or two charges");
        assert!(after
            .unknowns
            .iter()
            .any(|s| s.contains("NOT all-capture destruction")));
        assert_eq!(
            p.prefix.windows[0].owners.boundary_count, 0,
            "later lifecycle facts cannot rewrite frozen earlier owner window"
        );
    }
    #[test]
    fn owner_proven_extent_resolves_actual_product_but_not_compiled_input_mapping() {
        let mut p = Program::default();
        p.begin();
        let x = source_fresh(&mut p, "native fresh input", vec![2, 4], Dtype::Float32);
        let weight = source_fresh(&mut p, "native weight", vec![4], Dtype::Float32);
        p.rms("actual source RMS", &x, &weight).unwrap();
        let gate = source_fresh(&mut p, "gate", vec![2, 1], Dtype::Float32);
        p.gated("conditional compiled call", &x, &gate, &x).unwrap();
        p.window(
            IosPhase::Denoise,
            "product literal",
            &BTreeSet::new(),
            0,
            None,
            None,
        )
        .unwrap();
        let report = &p.prefix.windows[0].requests;
        assert!(
            report.requests.iter().any(|r| r.rule == "rmsnorm-softmax"
                && r.data_size_product.is_some()
                && r.bytes == Some(32)),
            "real input value source extent must resolve mapped data_size product"
        );
        assert!(
            report.requests.iter().any(|r| r.rule == "compiled-output"
                && r.data_size_product
                    .as_ref()
                    .is_some_and(|p| !p.native_input_mapping_bound)
                && r.bytes.is_none()),
            "source argument extents cannot fabricate actual compiled input mapping"
        );
        assert!(
            !report.unknowns.is_empty(),
            "resolving one product is not a total native/process/physical certificate"
        );
        let mut opaque = Program::default();
        opaque.begin();
        let x = opaque
            .immediate(
                "unbound native descriptor extent",
                vec![2, 4],
                Dtype::Float32,
            )
            .unwrap();
        let w = opaque
            .immediate("unbound native norm weight", vec![4], Dtype::Float32)
            .unwrap();
        opaque.rms("unbound actual RMS input", &x, &w).unwrap();
        opaque
            .window(
                IosPhase::Denoise,
                "unbound product literal",
                &BTreeSet::new(),
                0,
                None,
                None,
            )
            .unwrap();
        assert!(
            opaque.prefix.windows[0]
                .requests
                .requests
                .iter()
                .any(|r| r.rule == "rmsnorm-softmax"
                    && r.data_size_product.is_some()
                    && r.bytes.is_none()),
            "opaque native input extent must remain unbound rather than logical numel"
        );
    }
    #[test]
    fn owner_uncovered_view_copy_joins_only_its_creation_window_once() {
        let mut p = Program::default();
        p.begin();
        let source = p
            .immediate("opaque layout", vec![2, 3], Dtype::Float32)
            .unwrap();
        p.view(
            "actual layout-conditional reshape",
            &source,
            vec![3, 2],
            false,
        )
        .unwrap();
        p.window(
            IosPhase::Denoise,
            "created conditional view",
            &BTreeSet::new(),
            0,
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            p.prefix.windows[0]
                .requests
                .requests
                .iter()
                .filter(|r| r.rule == "owner-view-copy")
                .map(|r| r.bytes.unwrap())
                .collect::<Vec<_>>(),
            vec![24],
            "conditional native view copy without logical slot must be joined once"
        );
        p.window(
            IosPhase::Decode,
            "later owner frontier",
            &BTreeSet::new(),
            0,
            None,
            None,
        )
        .unwrap();
        assert!(
            !p.prefix.windows[1]
                .requests
                .requests
                .iter()
                .any(|r| r.rule == "owner-view-copy"),
            "retained queued/cache lease cannot be charged as a new allocation each window"
        );
        assert_eq!(p.prefix.windows[1].owners.leases.len(), 2);
    }
    #[test]
    fn owner_cache_strict_bounds_guard_native_overflow_and_never_bound_fresh_length() {
        assert_eq!(
            cached_length_bound(5000, Some(4096), 64).unwrap(),
            CacheLengthBound::Cached {
                aligned_request: 8192,
                upper_exclusive: 16384
            },
            "cache reuse must keep exact strict source bound after alignment"
        );
        assert_eq!(
            cached_length_bound(0, None, 64).unwrap(),
            CacheLengthBound::Empty
        );
        assert!(
            matches!(
                cached_length_bound(5000, None, 64).unwrap(),
                CacheLengthBound::Unbound(_)
            ),
            "unknown native page cannot default to4096/16384"
        );
        assert!(matches!(
            cached_length_bound(5000, Some(4096), 32).unwrap(),
            CacheLengthBound::Unbound(_)
        ));
        assert!(
            matches!(
                cached_length_bound(u64::MAX, Some(1), 64).unwrap(),
                CacheLengthBound::Unbound(_)
            ),
            "native2R overflow must prevent cached bound"
        );
        assert!(cached_length_bound(u64::MAX, Some(4096), 64).is_err());
        assert!(cached_length_bound(1, Some(0), 64).is_err());
        let mut p = Program::default();
        p.begin();
        source_fresh(
            &mut p,
            "fresh or cached allocation",
            vec![5000],
            Dtype::Uint8,
        );
        p.window(
            IosPhase::Load,
            "length literal",
            &BTreeSet::new(),
            0,
            None,
            None,
        )
        .unwrap();
        let lease = &p.prefix.windows[0].owners.leases[0];
        assert!(
            matches!(
                lease.length,
                LeaseLength::Unbound {
                    requested: Some(5000)
                }
            ),
            "requested bytes or cached upper must not become actual fresh/heap buf.length"
        );
        assert!(matches!(lease.cached_reuse, CacheLengthBound::Unbound(_)));
    }
}
