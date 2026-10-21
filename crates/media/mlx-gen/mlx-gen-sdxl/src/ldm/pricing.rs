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
            // Compare complete transformed facts, not independent shape cells. Equal targets
            // may come from different raw parents: ALL raw operations remain in this IR/floor.
            if writers
                .iter()
                .any(|&writer| self.output(writer).header != value.header)
            {
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
                // Mandatory all-orders walker is being developed separately. This unfinished
                // seam is not a permanent alias policy or a claim of full Fused support.
                Err(Error::Unsupported(format!("sdxl fused all-orders pricing remains incomplete ({} source alternatives, {} transform failure alternatives)",
                    source_indices.len(),transform_failures.len())))
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
