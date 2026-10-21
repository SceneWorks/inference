//! Actual fused source operations and load-floor facts. No order enumeration, alias rejection,
//! synthetic directory, tensor payload/model graph or guessed HashMap order.
use mlx_gen::gen_core::sdxl_ldm::{remap_sdxl_ldm_key, SdxlComponent, TensorRemap};
use mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader;
use mlx_gen::{Error, Result};
use std::collections::BTreeMap;

/// One actual source operation produces at most three correlated target headers. The callback
/// is immediate: no Cartesian source-world list, no independently chosen Q/K/V maxima.
pub(crate) fn visit_transformed_header(
    source: &SafetensorsTensorHeader,
    mut emit: impl FnMut(SdxlComponent, SafetensorsTensorHeader) -> Result<()>,
) -> Result<()> {
    let Some(remap) = remap_sdxl_ldm_key(&source.name) else {
        return Ok(());
    };
    let mut put = |component, name: String, shape: Vec<usize>| -> Result<()> {
        let elements = shape
            .iter()
            .try_fold(1u64, |n, &d| n.checked_mul(d as u64))
            .ok_or_else(|| Error::Msg("sdxl LDM header element count overflow".into()))?;
        let data_bytes = elements
            .checked_mul(source.dtype.size() as u64)
            .ok_or_else(|| Error::Msg("sdxl LDM header byte count overflow".into()))?;
        emit(
            component,
            SafetensorsTensorHeader {
                name,
                dtype: source.dtype,
                shape,
                data_bytes,
            },
        )
    };
    match remap {
        TensorRemap::Rename(component, name) => put(component, name, source.shape.clone()),
        TensorRemap::Transpose(component, name) => {
            let mut shape = source.shape.clone();
            shape.reverse();
            put(component, name, shape)
        }
        TensorRemap::Squeeze(component, name) => {
            put(component, name, super::squeezed_dimensions(&source.shape))
        }
        TensorRemap::SplitQkv(component, names) => {
            let mut shape = source.shape.clone();
            let rows = shape
                .first_mut()
                .ok_or_else(|| Error::Msg("sdxl LDM QKV source is scalar".into()))?;
            if *rows % 3 != 0 {
                return Err(Error::Msg(
                    "sdxl LDM QKV source does not split evenly".into(),
                ));
            }
            *rows /= 3;
            for name in names {
                put(component, name, shape.clone())?;
            }
            Ok(())
        }
    }
}

/// Full split fact maps. Values are metadata, not parameter/allocator owners. All four components
/// are transformed before the native UNet→CLIP-L→bigG→VAE nonempty checks.
#[derive(Default)]
pub(crate) struct FusedHeaders {
    pub unet: BTreeMap<String, SafetensorsTensorHeader>,
    pub clip_l: BTreeMap<String, SafetensorsTensorHeader>,
    pub clip_bigg: BTreeMap<String, SafetensorsTensorHeader>,
    pub vae: BTreeMap<String, SafetensorsTensorHeader>,
}
impl FusedHeaders {
    fn map(&mut self, component: SdxlComponent) -> &mut BTreeMap<String, SafetensorsTensorHeader> {
        match component {
            SdxlComponent::Unet => &mut self.unet,
            SdxlComponent::ClipL => &mut self.clip_l,
            SdxlComponent::ClipBigG => &mut self.clip_bigg,
            SdxlComponent::Vae => &mut self.vae,
        }
    }
    fn require_components(&self) -> Result<()> {
        for (name, empty) in [
            ("UNet", self.unet.is_empty()),
            ("CLIP-L", self.clip_l.is_empty()),
            ("OpenCLIP-bigG", self.clip_bigg.is_empty()),
            ("VAE", self.vae.is_empty()),
        ] {
            if empty {
                return Err(Error::Msg(format!(
                    "sdxl LDM checkpoint is missing the {name} component"
                )));
            }
        }
        Ok(())
    }
}

/// Replay an ACTUAL producer order supplied by a caller/witness. This method DOES NOT invent the
/// future native HashMap order; source admission has not yet supplied that witness. O(H+E) space,
/// one visit/source, E<=3H, including ignored raw tensors. No factorial/exponential search.
#[cfg(test)]
pub(crate) fn split_in_exact_source_order(
    headers: &[SafetensorsTensorHeader],
    order: &[usize],
) -> Result<FusedHeaders> {
    for header in headers {
        crate::quant::header_dtype(header.dtype)?;
    }
    if order.len() != headers.len() {
        return Err(Error::Msg(
            "sdxl LDM source order witness cardinality differs".into(),
        ));
    }
    let mut seen = vec![false; headers.len()];
    let mut result = FusedHeaders::default();
    for &index in order {
        let flag = seen.get_mut(index).ok_or_else(|| {
            Error::Msg("sdxl LDM source order witness index is out of range".into())
        })?;
        if *flag {
            return Err(Error::Msg(
                "sdxl LDM source order witness repeats a tensor".into(),
            ));
        }
        *flag = true;
        visit_transformed_header(&headers[index], |component, header| {
            // Exact native insert/overwrite, NOT the old component_headers duplicate refusal.
            result.map(component).insert(header.name.clone(), header);
            Ok(())
        })?;
    }
    result.require_components()?;
    Ok(result)
}

/// A source operation is kept once even when it writes three Q/K/V destinations. Target
/// references point back to this operation: they are not independent winner choices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SourceTransform {
    Ignored,
    Rename,
    Transpose,
    Squeeze,
    SplitQkv,
}

pub(crate) struct SourceOutput {
    pub component: SdxlComponent,
    pub header: SafetensorsTensorHeader,
}

pub(crate) struct SourceOperation {
    pub source_index: usize,
    pub source_name: String,
    pub raw_shape: Vec<usize>,
    pub raw_dtype: mlx_gen::gen_core::weightsmeta::Dtype,
    pub raw_data_bytes: u64,
    pub transform: SourceTransform,
    pub outputs: Vec<SourceOutput>,
    pub transform_failure: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WriterId {
    pub source: usize,
    pub output: usize,
}

fn component_key(component: SdxlComponent) -> u8 {
    match component {
        SdxlComponent::Unet => 0,
        SdxlComponent::ClipL => 1,
        SdxlComponent::ClipBigG => 2,
        SdxlComponent::Vae => 3,
    }
}

/// O(H+E) operation/writer records, E<=3H. Shape/name storage is proportional to the
/// corresponding source and emitted metadata, not to tensor payload or a set of worlds.
/// BTree indexing costs O(log(H+E)) per insertion; no claim of strictly linear runtime.
/// This IR does not establish a native backing size, future HashMap order or retirement.
/// It is shared by exact collision-equivalent projection and the all-orders analysis input.
pub(crate) struct FusedSourceIr {
    pub operations: Vec<SourceOperation>,
    pub target_writers: BTreeMap<(u8, String), Vec<WriterId>>,
}
impl FusedSourceIr {
    pub(crate) fn from_headers(headers: &[SafetensorsTensorHeader]) -> Result<Self> {
        // Native source decoding accepts/rejects every raw dtype before constructor lookup.
        for header in headers {
            crate::quant::header_dtype(header.dtype)?;
        }
        let mut result = Self {
            operations: Vec::with_capacity(headers.len()),
            target_writers: BTreeMap::new(),
        };
        for (source_index, source) in headers.iter().enumerate() {
            let transform = match remap_sdxl_ldm_key(&source.name) {
                None => SourceTransform::Ignored,
                Some(TensorRemap::Rename(..)) => SourceTransform::Rename,
                Some(TensorRemap::Transpose(..)) => SourceTransform::Transpose,
                Some(TensorRemap::Squeeze(..)) => SourceTransform::Squeeze,
                Some(TensorRemap::SplitQkv(..)) => SourceTransform::SplitQkv,
            };
            let mut outputs = Vec::new();
            let transformed = visit_transformed_header(source, |component, header| {
                outputs.push(SourceOutput { component, header });
                Ok(())
            });
            for (output, value) in outputs.iter().enumerate() {
                result
                    .target_writers
                    .entry((component_key(value.component), value.header.name.clone()))
                    .or_default()
                    .push(WriterId {
                        source: source_index,
                        output,
                    });
            }
            result.operations.push(SourceOperation {
                source_index,
                source_name: source.name.clone(),
                raw_shape: source.shape.clone(),
                raw_dtype: source.dtype,
                raw_data_bytes: source.data_bytes,
                transform,
                outputs,
                transform_failure: transformed.err().map(|error| error.to_string()),
            });
        }
        Ok(result)
    }

    pub(crate) fn output(&self, writer: WriterId) -> &SourceOutput {
        &self.operations[writer.source].outputs[writer.output]
    }

    pub(crate) fn project_equivalent(&self) -> Result<OrderInvariantProjection> {
        // Per-component packed flag from target NAMES only: identical in every source order.
        let mut packed = [false; 4];
        for (component, name) in self.target_writers.keys() {
            if name.ends_with(".scales") {
                packed[usize::from(*component)] = true;
            }
        }
        let mut result = FusedHeaders::default();
        let mut ambiguous = vec![false; self.operations.len()];
        let mut transform_failures = Vec::new();
        for operation in &self.operations {
            if let Some(error) = &operation.transform_failure {
                transform_failures.push((operation.source_index, error.clone()));
            }
        }
        for writers in self.target_writers.values() {
            let Some(&first) = writers.first() else {
                continue;
            };
            let value = self.output(first);
            // Compare every fact a header pricing consumer reads, never independent maxima.
            // Equal targets may come from different raw parents: ALL raw operations remain in
            // this IR and in the raw load floor/parent terms, which never use a winner.
            let component_packed = packed[usize::from(component_key(value.component))];
            let mut ambiguous_target = false;
            for &writer in writers {
                if !pricing_equivalent(
                    component_packed,
                    &self.output(writer).header,
                    &value.header,
                )? {
                    ambiguous_target = true;
                }
            }
            if ambiguous_target {
                for writer in writers {
                    ambiguous[writer.source] = true;
                }
            }
            result
                .map(value.component)
                .insert(value.header.name.clone(), value.header.clone());
        }
        let source_indices = ambiguous
            .into_iter()
            .enumerate()
            .filter_map(|(source, ambiguous)| ambiguous.then_some(source))
            .collect::<Vec<_>>();
        if !source_indices.is_empty() || !transform_failures.is_empty() {
            return Ok(OrderInvariantProjection::NeedsActualOrder {
                source_indices,
                transform_failures,
            });
        }
        result.require_components()?;
        Ok(OrderInvariantProjection::Resolved(result))
    }
}

/// Whether two writers of the SAME target are indistinguishable to every SDXL header pricing
/// consumer. Those consumers read a component header only as: its name; its shape; its loaded
/// dtype (`quant::loaded_header_dtype`, via `clip_header_tensor`, `linear_from_headers`,
/// `HeaderWeights`, adapter `header_targets` and the fused footprint); raw dtypes only where the
/// loaded dtype IS the stored dtype (packed codes/scales/biases); raw dtype validation, which
/// `FusedSourceIr::from_headers` already applies to every source; and `text_pack_facts`, which
/// takes `data_bytes` only when the stored dtype equals the loaded one and otherwise the loaded
/// materialized size. Transformed `data_bytes` is elements x stored width by construction, so equal
/// shapes and equal loaded dtypes give equal bytes on both branches. Fused sources never arm the
/// rung-4 block stream (`memory_strategy::streamable` requires a Directory). Hence equal shape and
/// equal loaded dtype make every pricing input identical in every actual source order.
fn pricing_equivalent(
    packed: bool,
    a: &SafetensorsTensorHeader,
    b: &SafetensorsTensorHeader,
) -> Result<bool> {
    Ok(a.shape == b.shape
        && crate::quant::loaded_dtype_in_component(packed, a)?
            == crate::quant::loaded_dtype_in_component(packed, b)?)
}

/// O(H+E) proof result, not an arbitrary winner or capability refusal. If an alternative changes
/// pricing facts, return its compact source ambiguity; never independently maximize tensor cells.
pub(crate) enum OrderInvariantProjection {
    Resolved(FusedHeaders),
    NeedsActualOrder {
        source_indices: Vec<usize>,
        transform_failures: Vec<(usize, String)>,
    },
}
pub(crate) fn order_invariant_projection(
    headers: &[SafetensorsTensorHeader],
) -> Result<OrderInvariantProjection> {
    FusedSourceIr::from_headers(headers)?.project_equivalent()
}

/// One genuine fused input: raw source parents plus all remapped components from the SAME
/// fullsplit. No component filtering before sibling validation, synthetic directory or payload.
pub(crate) struct FusedHeaderSource {
    pub raw: Vec<SafetensorsTensorHeader>,
    pub components: FusedHeaders,
    ir: FusedSourceIr,
}
impl FusedHeaderSource {
    pub(crate) fn read(path: &std::path::Path) -> Result<Self> {
        let raw = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(path)
            .map_err(|error| Error::Msg(error.to_string()))?;
        Self::from_headers(raw)
    }
    pub(crate) fn from_headers(raw: Vec<SafetensorsTensorHeader>) -> Result<Self> {
        let ir = FusedSourceIr::from_headers(&raw)?;
        match ir.project_equivalent()? {
            OrderInvariantProjection::Resolved(components) => {
                // The same IR that established source-op/target provenance supplies the actual
                // raw parent records consumed by the shared native load-floor expression.
                // These are lossless original records, NOT joined/maximized checkpoint headers.
                let raw = ir
                    .operations
                    .iter()
                    .map(|op| SafetensorsTensorHeader {
                        name: op.source_name.clone(),
                        shape: op.raw_shape.clone(),
                        dtype: op.raw_dtype,
                        data_bytes: op.raw_data_bytes,
                    })
                    .collect();
                Ok(Self {
                    raw,
                    components,
                    ir,
                })
            }
            OrderInvariantProjection::NeedsActualOrder {
                source_indices,
                transform_failures,
            } => {
                // Native split transforms EVERY mapped source before any component check, so a
                // failing transform fails in every source order: a certain native failure.
                if let Some((_, error)) = transform_failures.first() {
                    return Err(Error::Msg(format!(
                        "sdxl LDM fused source transform failure in every source order: {error}"
                    )));
                }
                // Remaining: writers whose shape or loaded dtype differ, so the priced program
                // depends on native source order. Not a native restriction or Fused completion.
                Err(Error::Unsupported(format!(
                    "sdxl fused pricing for order-dependent source aliases remains incomplete ({} source alternatives)",
                    source_indices.len()
                )))
            }
        }
    }
    pub(crate) fn headers(&self, component: SdxlComponent) -> Vec<SafetensorsTensorHeader> {
        let map = match component {
            SdxlComponent::Unet => &self.components.unet,
            SdxlComponent::ClipL => &self.components.clip_l,
            SdxlComponent::ClipBigG => &self.components.clip_bigg,
            SdxlComponent::Vae => &self.components.vae,
        };
        map.values().cloned().collect()
    }
    pub(crate) fn raw_parent_bytes(&self, components: &[SdxlComponent]) -> u64 {
        self.ir
            .operations
            .iter()
            .filter(|op| {
                if op.transform == SourceTransform::Ignored {
                    return false;
                }
                op.outputs
                    .first()
                    .is_some_and(|output| components.contains(&output.component))
            })
            .fold(0u64, |sum, op| {
                sum.saturating_add(mlx_gen::ios_quantum::allocated_bytes(op.raw_data_bytes))
            })
    }
}

/// MLX ACTIVE-ALLOCATOR counter charge for one materialized request of `n` bytes (audit
/// sc26270-fused-native-backing-source-authority-v2, sha 3c577d52…): a new MTLBuffer has the
/// page-rounded length r(n); a cache-reused one is below min(2*r(n), r(n)+2*page). This is a
/// native counter law, NOT task RSS/process footprint (activity-26569 premise stays separate).
pub(crate) fn active_counter_charge(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let rounded = mlx_gen::ios_quantum::allocated_bytes(n);
    rounded
        .saturating_mul(2)
        .min(rounded.saturating_add(2 * mlx_gen::ios_quantum::PAGE_BYTES))
}

/// Lower/upper MLX active-counter backing of one actual fused split over every realizable native
/// source order. Counter-scoped only: no physical mapping, retirement or parent subtraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FusedCounterBacking {
    pub lower: u64,
    pub upper: u64,
}

impl FusedHeaderSource {
    /// The component-wide packed flag (`quant::loaded_header_dtype`) from target NAMES only.
    pub(crate) fn component_packed(&self, component: SdxlComponent) -> bool {
        let key = component_key(component);
        self.ir
            .target_writers
            .keys()
            .any(|(c, name)| *c == key && name.ends_with(".scales"))
    }

    /// Native split moves each mapped source's lazy Load into its component map; an overwritten
    /// writer and every ignored or unconstructed-component source is dropped unevaluated. A
    /// winning source is evaluated once (Load allocates its nbytes); split/transpose/reshape views
    /// share that buffer, so a source is charged ONCE however many targets it wins. Each winning
    /// target whose loaded dtype differs from its stored dtype adds its AsType output. Alias groups
    /// are connected components of (source, ambiguous target); their realizable winner assignments
    /// are exactly those whose "winner after every other writer" constraints are acyclic, and
    /// distinct groups share no source, so their extremes add.
    pub(crate) fn active_counter_backing(
        &self,
        components: &[SdxlComponent],
        loaded: &dyn Fn(SdxlComponent, &SafetensorsTensorHeader) -> Result<mlx_rs::Dtype>,
    ) -> Result<FusedCounterBacking> {
        let cast = |writer: WriterId| -> Result<u64> {
            let output = self.ir.output(writer);
            let target = loaded(output.component, &output.header)?;
            if crate::quant::header_dtype(output.header.dtype)? == target {
                return Ok(0);
            }
            let bytes = output
                .header
                .materialized_bytes(crate::quantum::dtype_bytes(target))
                .map_err(|error| Error::Msg(error.to_string()))?;
            Ok(active_counter_charge(bytes))
        };
        let raw = |source: usize| active_counter_charge(self.ir.operations[source].raw_data_bytes);
        let selected = self
            .ir
            .target_writers
            .iter()
            .filter(|((component, _), writers)| {
                !writers.is_empty() && components.iter().any(|c| component_key(*c) == *component)
            })
            .map(|(_, writers)| writers.as_slice())
            .collect::<Vec<_>>();
        let mut fixed_sources = std::collections::BTreeSet::new();
        let mut base = 0u64;
        for writers in selected.iter().filter(|writers| writers.len() == 1) {
            fixed_sources.insert(writers[0].source);
            base = base.saturating_add(cast(writers[0])?);
        }
        for &source in &fixed_sources {
            base = base.saturating_add(raw(source));
        }
        // Union sources through ambiguous targets.
        let ambiguous = selected
            .iter()
            .filter(|writers| writers.len() > 1)
            .copied()
            .collect::<Vec<_>>();
        let mut parent = (0..self.ir.operations.len()).collect::<Vec<_>>();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        for writers in &ambiguous {
            let first = find(&mut parent, writers[0].source);
            for writer in &writers[1..] {
                let other = find(&mut parent, writer.source);
                parent[other] = first;
            }
        }
        let mut groups: BTreeMap<usize, Vec<&[WriterId]>> = BTreeMap::new();
        for writers in ambiguous {
            let root = find(&mut parent, writers[0].source);
            groups.entry(root).or_default().push(writers);
        }
        let (mut lower, mut upper) = (base, base);
        for targets in groups.values() {
            let mut extreme: Option<(u64, u64)> = None;
            let mut choice = vec![0usize; targets.len()];
            loop {
                if realizable(targets, &choice) {
                    let mut charged = std::collections::BTreeSet::new();
                    let mut cost = 0u64;
                    for (writers, &pick) in targets.iter().zip(&choice) {
                        let winner = writers[pick];
                        cost = cost.saturating_add(cast(winner)?);
                        if !fixed_sources.contains(&winner.source) && charged.insert(winner.source)
                        {
                            cost = cost.saturating_add(raw(winner.source));
                        }
                    }
                    extreme =
                        Some(extreme.map_or((cost, cost), |(lo, hi)| (lo.min(cost), hi.max(cost))));
                }
                // Odometer over this group's per-target writer lists only.
                let mut index = 0;
                loop {
                    if index == choice.len() {
                        break;
                    }
                    choice[index] += 1;
                    if choice[index] < targets[index].len() {
                        break;
                    }
                    choice[index] = 0;
                    index += 1;
                }
                if index == choice.len() {
                    break;
                }
            }
            // Some order always exists, so every group has a realizable assignment.
            let (lo, hi) = extreme.ok_or_else(|| {
                Error::Msg("sdxl fused alias group has no realizable source order".into())
            })?;
            lower = lower.saturating_add(lo);
            upper = upper.saturating_add(hi);
        }
        Ok(FusedCounterBacking { lower, upper })
    }
}

/// A winner assignment is realizable iff "winner follows every other writer of its target" is an
/// acyclic relation over the group's sources (then some native visit order produces it).
fn realizable(targets: &[&[WriterId]], choice: &[usize]) -> bool {
    let mut edges: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut nodes = std::collections::BTreeSet::new();
    for (writers, &pick) in targets.iter().zip(choice) {
        let winner = writers[pick].source;
        for writer in writers.iter() {
            nodes.insert(writer.source);
            if writer.source != winner {
                edges.entry(writer.source).or_default().push(winner);
            }
        }
    }
    let mut indegree: BTreeMap<usize, usize> = nodes.iter().map(|&n| (n, 0)).collect();
    for targets in edges.values() {
        for target in targets {
            *indegree.get_mut(target).expect("edge endpoint is a node") += 1;
        }
    }
    let mut ready = indegree
        .iter()
        .filter(|(_, &d)| d == 0)
        .map(|(&n, _)| n)
        .collect::<Vec<_>>();
    let mut visited = 0usize;
    while let Some(node) = ready.pop() {
        visited += 1;
        for target in edges.get(&node).into_iter().flatten() {
            let degree = indegree.get_mut(target).expect("edge endpoint is a node");
            *degree -= 1;
            if *degree == 0 {
                ready.push(*target);
            }
        }
    }
    visited == nodes.len()
}

#[cfg(test)]
mod source_ir_tests {
    use super::*;
    use mlx_gen::gen_core::weightsmeta::Dtype as Stored;

    fn header(name: &str, shape: &[usize], dtype: Stored) -> SafetensorsTensorHeader {
        SafetensorsTensorHeader {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            data_bytes: shape.iter().product::<usize>() as u64 * dtype.size() as u64,
        }
    }

    fn source() -> Vec<SafetensorsTensorHeader> {
        vec![
            header(
                "model.diffusion_model.time_embed.0.weight",
                &[4, 4],
                Stored::F32,
            ),
            header(
                "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
                &[3, 4],
                Stored::F16,
            ),
            header(
                "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
                &[6, 4],
                Stored::F32,
            ),
            header(
                "conditioner.embedders.1.model.text_projection",
                &[4, 2],
                Stored::F32,
            ),
            header(
                "first_stage_model.decoder.mid.attn_1.q.weight",
                &[4, 4, 1, 1],
                Stored::F16,
            ),
            header("ignored_payload", &[17], Stored::U8),
        ]
    }

    #[test]
    fn source_ir_keeps_qkv_parent_and_all_equivalent_writers_host() {
        let mut headers = source();
        // Separate alias has the SAME post-transform facts as the Q part. No raw parent
        // disappears when the exact route collapses a target's equivalent winner facts.
        headers.push(header(
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.q_proj.weight",
            &[2, 4],
            Stored::F32,
        ));
        let ir = FusedSourceIr::from_headers(&headers).unwrap();
        assert_eq!(ir.operations.len(), 7);
        assert_eq!(ir.operations[2].transform, SourceTransform::SplitQkv);
        assert_eq!(ir.operations[2].outputs.len(), 3);
        assert_eq!(ir.operations[2].raw_shape, [6, 4]);
        assert_eq!(ir.operations[2].raw_dtype, Stored::F32);
        assert_eq!(ir.operations[2].raw_data_bytes, 96);
        assert_eq!(ir.operations[5].transform, SourceTransform::Ignored);
        assert_eq!(ir.operations[5].source_name, "ignored_payload");
        assert!(ir.operations[5].outputs.is_empty());
        let key = (
            2,
            "text_model.encoder.layers.0.self_attn.q_proj.weight".to_owned(),
        );
        assert_eq!(
            ir.target_writers[&key],
            [
                WriterId {
                    source: 2,
                    output: 0
                },
                WriterId {
                    source: 6,
                    output: 0
                }
            ]
        );
        for role in ["k_proj", "v_proj"] {
            let writers = &ir.target_writers[&(
                2,
                format!("text_model.encoder.layers.0.self_attn.{role}.weight"),
            )];
            assert_eq!(writers.len(), 1);
            assert_eq!(writers[0].source, 2);
        }
        assert_eq!(ir.target_writers.values().map(Vec::len).sum::<usize>(), 8);
        let OrderInvariantProjection::Resolved(projected) = ir.project_equivalent().unwrap() else {
            panic!("equal complete target facts are exact despite raw aliases");
        };
        assert_eq!(projected.clip_bigg[&key.1].shape, [2, 4]);
        // Same source-count bound regardless of declared payload extents.
        headers[5].shape = vec![usize::MAX];
        headers[5].data_bytes = u64::MAX;
        let large = FusedSourceIr::from_headers(&headers).unwrap();
        assert_eq!(large.operations.len(), 7);
        assert_eq!(
            large.target_writers.values().map(Vec::len).sum::<usize>(),
            8
        );
    }

    #[test]
    fn dtype_only_aliases_price_identically_in_every_actual_order_host() {
        let mut headers = source();
        // Same post-split shape as the Q part, different stored dtype (F16 vs split F32).
        headers.push(header(
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.q_proj.weight",
            &[2, 4],
            Stored::F16,
        ));
        let OrderInvariantProjection::Resolved(_) = FusedSourceIr::from_headers(&headers)
            .unwrap()
            .project_equivalent()
            .unwrap()
        else {
            panic!("equal shape and loaded dtype are pricing-equivalent");
        };
        // Both actual native orders: the split writer last, then the alias writer last.
        let split_last = split_in_exact_source_order(&headers, &[6, 0, 1, 2, 3, 4, 5]).unwrap();
        let alias_last = split_in_exact_source_order(&headers, &[0, 1, 2, 3, 4, 5, 6]).unwrap();
        let q = "text_model.encoder.layers.0.self_attn.q_proj.weight";
        assert_eq!(split_last.clip_bigg[q].dtype, Stored::F32);
        assert_eq!(alias_last.clip_bigg[q].dtype, Stored::F16);
        let a: Vec<_> = split_last.clip_bigg.values().cloned().collect();
        let b: Vec<_> = alias_last.clip_bigg.values().cloned().collect();
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(
                crate::text_encoder::clip_header_tensor(&a, &x.name).unwrap(),
                crate::text_encoder::clip_header_tensor(&b, &y.name).unwrap()
            );
        }
        let cfg = crate::config::ClipTextConfig::sdxl_te2();
        assert_eq!(
            crate::text_encoder::text_pack_facts(&a, &cfg, None)
                .unwrap()
                .retained,
            crate::text_encoder::text_pack_facts(&b, &cfg, None)
                .unwrap()
                .retained
        );
        // Every raw parent, including the losing alias, stays in the source-parent term.
        let source = FusedHeaderSource::from_headers(headers).unwrap();
        assert_eq!(
            source.raw_parent_bytes(&[SdxlComponent::ClipBigG]),
            96 + 32 + 16
        );
    }

    #[test]
    fn packed_companion_dtype_alias_stays_order_dependent_host() {
        let mut headers = source();
        let base = "model.diffusion_model.output_blocks.2";
        // Both upsampler slots remap to up_blocks.0.upsamplers.0.conv.scales; packed
        // companions load at their STORED dtype, so F16 vs BF16 changes priced facts.
        headers.push(header(
            &format!("{base}.1.conv.scales"),
            &[4, 1],
            Stored::F16,
        ));
        headers.push(header(
            &format!("{base}.2.conv.scales"),
            &[4, 1],
            Stored::BF16,
        ));
        let OrderInvariantProjection::NeedsActualOrder {
            source_indices,
            transform_failures,
        } = FusedSourceIr::from_headers(&headers)
            .unwrap()
            .project_equivalent()
            .unwrap()
        else {
            panic!("loaded-dtype difference must not be collapsed");
        };
        assert_eq!(source_indices, [6, 7]);
        assert!(transform_failures.is_empty());
        let error = FusedHeaderSource::from_headers(headers).err().unwrap();
        assert!(matches!(&error, Error::Unsupported(text) if text.contains("order-dependent")));
    }

    #[test]
    fn fused_transform_failure_is_a_certain_native_failure_host() {
        let mut headers = source();
        headers.push(header(
            "conditioner.embedders.1.model.transformer.resblocks.1.attn.in_proj_weight",
            &[7, 4],
            Stored::F32,
        ));
        let error = FusedHeaderSource::from_headers(headers).err().unwrap();
        assert!(
            matches!(&error, Error::Msg(text) if text.contains("every source order") && text.contains("split evenly"))
        );
    }

    #[test]
    fn fused_active_counter_backing_charges_unique_winning_sources_and_casts_host() {
        let mut headers = source();
        // Pricing-equivalent alias (same shape, loaded F16) but different raw bytes/cast.
        headers.push(header(
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.q_proj.weight",
            &[2, 4],
            Stored::BF16,
        ));
        let source = FusedHeaderSource::from_headers(headers).unwrap();
        let loaded = |_: SdxlComponent, _: &SafetensorsTensorHeader| Ok(crate::model::DTYPE);
        let backing = source
            .active_counter_backing(&[SdxlComponent::ClipBigG], &loaded)
            .unwrap();
        // Fixed: split source U(96)=192 once for its three views + text_projection U(32)=64,
        // casts of K, V and the projection U(16)=32 each. Q: split wins (cast 32, raw already
        // charged) or the BF16 alias wins (raw U(16)=32 + cast 32). U(n)=2n below one page.
        assert_eq!(
            backing,
            FusedCounterBacking {
                lower: 352 + 32,
                upper: 352 + 64
            }
        );
        assert_eq!(active_counter_charge(0), 0);
        let page = mlx_gen::ios_quantum::PAGE_BYTES;
        assert_eq!(active_counter_charge(10 * page + 1), 11 * page + 2 * page);
    }

    #[test]
    fn alias_winner_assignments_need_an_acyclic_source_order_host() {
        let w = |source| WriterId { source, output: 0 };
        let q = [w(1), w(2)];
        let k = [w(1), w(2)];
        // Two split sources writing Q and K: Q from 1 and K from 2 needs 1 after 2 and 2 after 1.
        assert!(realizable(&[&q, &k], &[0, 0]));
        assert!(realizable(&[&q, &k], &[1, 1]));
        assert!(!realizable(&[&q, &k], &[0, 1]));
    }

    #[test]
    fn source_ir_retains_ambiguity_and_transform_failure_prefixes_host() {
        let mut headers = source();
        headers.push(header(
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.q_proj.weight",
            &[3, 4],
            Stored::F32,
        ));
        headers.push(header(
            "conditioner.embedders.1.model.transformer.resblocks.1.attn.in_proj_weight",
            &[7, 4],
            Stored::F32,
        ));
        let ir = FusedSourceIr::from_headers(&headers).unwrap();
        let OrderInvariantProjection::NeedsActualOrder {
            source_indices,
            transform_failures,
        } = ir.project_equivalent().unwrap()
        else {
            panic!("do not choose a source-order world");
        };
        assert_eq!(source_indices, [2, 6]);
        assert_eq!(transform_failures.len(), 1);
        assert_eq!(transform_failures[0].0, 7);
        assert!(transform_failures[0].1.contains("split evenly"));
        assert_eq!(ir.operations[2].outputs.len(), 3);
        assert!(ir.operations[7].outputs.is_empty());
        assert_eq!(ir.operations.len(), 8);
    }
}
