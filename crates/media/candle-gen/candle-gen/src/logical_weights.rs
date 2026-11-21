//! Media projection constructors over the Candle checkpoint reader shared with audio and LLMs.
use candle_checkpoint::logical_weights as shared;
pub use candle_checkpoint::logical_weights::{
    baseline_codec_registry, nvfp4_layout_is_native, read_logical_weights,
    register_checkpoint_codecs, CandleCodecResidency, LogicalTensor, LogicalWeights,
    BASELINE_CODECS, CODEC_IMPLEMENTATION_IDS,
};
use candle_core::{Device, Tensor};
use gen_core::checkpoint_codec::{CodecResidencyPolicy, LogicalKeyMapping, LogicalWeightPlan};
use gen_core::checkpoint_facts::{CheckpointWeightFacts, NativeExecutionCapability};
use std::path::Path;

pub fn plan_logical_weights(
    path: &Path,
    mapping: &dyn LogicalKeyMapping,
    policy: &dyn CodecResidencyPolicy,
) -> crate::Result<LogicalWeightPlan> {
    Ok(shared::plan_logical_weights(path, mapping, policy)?)
}

pub struct LogicalWeightReader(shared::LogicalWeightReader);

impl std::ops::Deref for LogicalWeightReader {
    type Target = shared::LogicalWeightReader;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl LogicalWeightReader {
    pub fn open(path: &Path, plan: LogicalWeightPlan, device: &Device) -> crate::Result<Self> {
        Ok(Self(shared::LogicalWeightReader::open(path, plan, device)?))
    }

    pub fn open_with_capability(
        path: &Path,
        plan: LogicalWeightPlan,
        device: &Device,
        capability: NativeExecutionCapability,
    ) -> crate::Result<Self> {
        Ok(Self(shared::LogicalWeightReader::open_with_capability(
            path, plan, device, capability,
        )?))
    }

    pub fn checkpoint_weight_facts(&self) -> crate::Result<CheckpointWeightFacts> {
        Ok(self.0.checkpoint_weight_facts()?)
    }

    /// Seed a model's shared cuBLASLt context after the provider checked the device.
    pub fn set_int8_context(&mut self, context: crate::quant::Int8Context) {
        self.0.set_int8_context(context);
    }

    /// Materialize and bind a planned ConvRot projection to this reader's shared device resources.
    pub fn convrot_projection(
        &self,
        key: &str,
        bias: Option<Tensor>,
    ) -> crate::Result<Option<crate::quant::AdaptLinear>> {
        Ok(self
            .0
            .convrot_projection(key, bias)?
            .map(crate::quant::convrot_int8::ConvRotLinear::from_native)
            .map(crate::quant::AdaptLinear::from_convrot))
    }

    pub fn has_native_projection(&self, key: &str) -> bool {
        self.0.has_native_projection(key)
            || self
                .planned(key)
                .is_some_and(|p| matches!(p.codec, gen_core::TensorCodecSpec::ConvRotInt8 { .. }))
    }
    pub fn native_projection(
        &self,
        key: &str,
        bias: Option<Tensor>,
    ) -> crate::Result<Option<crate::quant::AdaptLinear>> {
        if let Some(layer) = self.convrot_projection(key, bias.clone())? {
            return Ok(Some(layer));
        }
        #[cfg(feature = "cuda")]
        if let Some(layer) = self.0.fp8_projection(key, bias)? {
            let (out_dim, in_dim) = layer.dims();
            return Ok(Some(crate::quant::AdaptLinear::from_packed(
                crate::quant::QLinear::Fp8(std::sync::Arc::new(layer)),
                in_dim,
                out_dim,
            )));
        }
        #[cfg(not(feature = "cuda"))]
        if self.has_native_projection(key) {
            let _ = bias;
            return Err(crate::CandleError::Unsupported(format!(
                "{key}: packed FP8 requires CUDA"
            )));
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_quant_kernels::Nvfp4Tensor;
    use gen_core::checkpoint_codec::*;
    use gen_core::checkpoint_codec::{LogicalTensorTransform, PlannedResidency, RowRange};
    use gen_core::checkpoint_facts::ExecutionRepresentation;
    use gen_core::ProviderRegistryBuilder;
    use shared::apply_transform;
    use std::collections::BTreeMap;

    struct StripModel;

    impl LogicalKeyMapping for StripModel {
        fn mapping_id(&self) -> &'static str {
            "strip-model-test"
        }
        fn logical_key(&self, physical_key: &str) -> Option<String> {
            physical_key.strip_prefix("model.").map(str::to_owned)
        }
    }

    /// [`StripModel`] that also **declares** one logical shape for every key it maps.
    ///
    /// A block-padded codec row (MXFP8/NVFP4) is only materializable when the adapter states the
    /// layer's true geometry — otherwise the plan can do nothing but carry the padded stored grid
    /// forward, and decoding it would promote padding to weights (`gen_core` refuses by name). The
    /// NVFP4 fixtures below are built UNPADDED on purpose, so declaring the grid they were built at
    /// is the adapter making a true statement about them, not a restatement of the storage.
    struct StripModelDeclaring(Vec<usize>);

    impl LogicalKeyMapping for StripModelDeclaring {
        fn mapping_id(&self) -> &'static str {
            "strip-model-declaring-test"
        }
        fn logical_key(&self, physical_key: &str) -> Option<String> {
            physical_key.strip_prefix("model.").map(str::to_owned)
        }
        fn logical_shape(&self, _logical_key: &str) -> Option<Vec<usize>> {
            Some(self.0.clone())
        }
    }

    fn fixture_dir() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix(&format!("candle-logical-weights-{}-", std::process::id()))
            .tempdir()
            .expect("fixture dir")
    }

    /// sc-20385 review: gen-core's own 256-code sweeps check `fp8_e4m3fn_to_f32` against a
    /// `spec_e4m3fn` helper written from the same OCP bit fields in the same file — a
    /// transliteration, so the two share any misreading of the table. `float8::F8E4M3` is a
    /// third-party implementation of the same standard; agreeing with it on all 256 codes is
    /// independent evidence. E5M2 gets the same treatment (its existing second oracle is the
    /// binary16-top-byte identity, which is genuinely independent, so this is belt and braces).
    ///
    /// NaN is compared by class: E4M3FN has no infinities and two NaN codes (`0x7F`, `0xFF`).
    #[test]
    fn fp8_references_agree_with_the_float8_crate_on_all_256_codes() {
        for code in 0..=u8::MAX {
            let ours = gen_core::fp8_e4m3fn_to_f32(code);
            let theirs = float8::F8E4M3::from_bits(code).to_f32();
            assert_eq!(
                ours.is_nan(),
                theirs.is_nan(),
                "e4m3 code {code:#04x}: NaN class disagrees ({ours} vs {theirs})"
            );
            if !ours.is_nan() {
                assert_eq!(
                    ours.to_bits(),
                    theirs.to_bits(),
                    "e4m3 code {code:#04x}: {ours} vs {theirs}"
                );
            }

            let ours = gen_core::fp8_e5m2_to_f32(code);
            let theirs = float8::F8E5M2::from_bits(code).to_f32();
            assert_eq!(
                ours.is_nan(),
                theirs.is_nan(),
                "e5m2 code {code:#04x}: NaN class disagrees ({ours} vs {theirs})"
            );
            if !ours.is_nan() {
                assert_eq!(
                    ours.to_bits(),
                    theirs.to_bits(),
                    "e5m2 code {code:#04x}: {ours} vs {theirs}"
                );
            }
        }
        // The sweep is only meaningful if the oracle actually distinguishes codes.
        assert_eq!(float8::F8E4M3::from_bits(0x38).to_f32(), 1.0);
        assert_eq!(float8::F8E5M2::from_bits(0x3C).to_f32(), 1.0);
    }

    /// sc-20385 review: the residency policy must not re-derive the locked-decision-7 sm_89
    /// threshold. `CublasLt::meets_fp8_floor` and `CandleCodecResidency::probe`'s device predicate
    /// both go through this one function, so the grid below pins the single definition. (The
    /// `probe` side is `cfg(cuda)`; the shared predicate is not, so this runs on every lane.)
    #[test]
    fn the_eight_bit_track_floor_has_one_definition_at_sm89() {
        assert_eq!(crate::quant::FP8_COMPUTE_CAP_FLOOR, (8, 9));
        for (cap, expected) in [
            ((7, 5), false),
            ((8, 0), false),
            ((8, 6), false),
            ((8, 8), false),
            ((8, 9), true),
            ((8, 10), true),
            ((9, 0), true),
            ((12, 0), true),
        ] {
            assert_eq!(
                crate::quant::compute_cap_meets_fp8_floor(cap),
                expected,
                "compute capability {cap:?}"
            );
        }
    }

    /// Write a minimal safetensors file from `(name, dtype, shape, little-endian payload)` rows.
    fn write_safetensors(path: &Path, tensors: &[(&str, &str, &[usize], Vec<u8>)]) {
        let mut header_entries = Vec::new();
        let mut body = Vec::new();
        for (name, dtype, shape, payload) in tensors {
            let start = body.len();
            body.extend_from_slice(payload);
            let end = body.len();
            header_entries.push(format!(
                "{:?}:{{\"dtype\":{:?},\"shape\":{:?},\"data_offsets\":[{start},{end}]}}",
                name, dtype, shape
            ));
        }
        let mut header = format!("{{{}}}", header_entries.join(",")).into_bytes();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut file = Vec::with_capacity(8 + header.len() + body.len());
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(&body);
        std::fs::write(path, file).expect("write safetensors fixture");
    }

    /// [`write_safetensors`] plus a file-level `__metadata__` map — the route a header-declared
    /// NVFP4 checkpoint takes (sc-20641).
    fn write_safetensors_with_metadata(
        path: &Path,
        tensors: &[(&str, &str, &[usize], Vec<u8>)],
        metadata: &[(&str, &str)],
    ) {
        let mut header_entries = vec![format!(
            "\"__metadata__\":{{{}}}",
            metadata
                .iter()
                .map(|(key, value)| format!("{key:?}:{value:?}"))
                .collect::<Vec<_>>()
                .join(",")
        )];
        let mut body = Vec::new();
        for (name, dtype, shape, payload) in tensors {
            let start = body.len();
            body.extend_from_slice(payload);
            let end = body.len();
            header_entries.push(format!(
                "{:?}:{{\"dtype\":{:?},\"shape\":{:?},\"data_offsets\":[{start},{end}]}}",
                name, dtype, shape
            ));
        }
        let mut header = format!("{{{}}}", header_entries.join(",")).into_bytes();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut file = Vec::with_capacity(8 + header.len() + body.len());
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(&body);
        std::fs::write(path, file).expect("write safetensors fixture");
    }

    // ---- an independent NVFP4 reference -------------------------------------------------------
    //
    // Written from the format specification and ComfyUI's own quantizer
    // (`comfy.quant_ops.TensorCoreNVFP4Layout.quantize`, `comfy.float.to_blocked` /
    // `stochastic_float_to_fp4_e2m1`), NOT by calling the codec under test. The FP8 E4M3 half goes
    // through the third-party `float8` crate; the E2M1 grid and the `to_blocked` swizzle are
    // derived here from their definitions.

    /// `comfy.float.to_blocked`, transliterated as an explicit index walk: view the padded
    /// `[R, B]` scale matrix as `(R/128, 128, B/4, 4)`, permute to `(R/128, B/4, 128, 4)`, reshape
    /// to `(-1, 4, 32, 4)`, transpose axes 1 and 2, flatten. Returns the flat destination index of
    /// logical `(row, block)`.
    fn reference_to_blocked_index(blocks: usize, row: usize, block: usize) -> usize {
        let n_col_blocks = blocks.div_ceil(4);
        // (R/128, B/4, 128, 4) — atoms walked row-major, then 128 rows × 4 block-cols inside.
        let atom = (row / 128) * n_col_blocks + block / 4;
        let (row_in_atom, col_in_atom) = (row % 128, block % 4);
        // (128, 4) viewed as (4, 32, 4) then transposed to (32, 4, 4): source (a, b, c) with
        // row_in_atom = a * 32 + b lands at b * 16 + a * 4 + c.
        let (a, b) = (row_in_atom / 32, row_in_atom % 32);
        atom * 512 + b * 16 + a * 4 + col_in_atom
    }

    /// The eight non-negative E2M1 magnitudes, derived from the bit fields (2 exponent bits at
    /// bias 1, 1 mantissa bit) rather than copied from a table.
    fn reference_e2m1_magnitudes() -> [f32; 8] {
        let mut grid = [0.0_f32; 8];
        for (code, slot) in grid.iter_mut().enumerate() {
            let (e, m) = ((code >> 1) as i32, (code & 1) as f32);
            *slot = if e == 0 {
                m / 2.0
            } else {
                (1.0 + m / 2.0) * 2f32.powi(e - 1)
            };
        }
        grid
    }

    /// Nearest E2M1 code (sign in bit 3) for a value already divided by its block scale.
    fn reference_e2m1_code(value: f32) -> u8 {
        let grid = reference_e2m1_magnitudes();
        let magnitude = value.abs().min(6.0);
        let mut best = 0_usize;
        for (index, candidate) in grid.iter().enumerate() {
            if (candidate - magnitude).abs() < (grid[best] - magnitude).abs() {
                best = index;
            }
        }
        (if value.is_sign_negative() { 0x08 } else { 0 }) | best as u8
    }

    /// Quantize a dense row-major `[rows, cols]` f32 matrix to the three stored NVFP4 artifacts,
    /// and return the values a correct decoder must reproduce.
    #[allow(clippy::type_complexity)]
    fn nvfp4_reference_quantize(
        data: &[f32],
        rows: usize,
        cols: usize,
    ) -> (Vec<u8>, Vec<u8>, f32, Vec<f32>) {
        assert_eq!(data.len(), rows * cols);
        assert!(cols.is_multiple_of(16) && rows.is_multiple_of(16));
        let blocks = cols / 16;
        // `scale = amax(|W|) / (F8_E4M3_MAX * F4_E2M1_MAX)` — quant_ops.py line 134.
        let amax = data.iter().fold(0.0_f32, |max, v| max.max(v.abs()));
        let global = amax / (448.0 * 6.0);

        let scale_rows = rows.div_ceil(128) * 128;
        let scale_cols = blocks.div_ceil(4) * 4;
        let mut blocked_scales = vec![0_u8; scale_rows * scale_cols];
        let mut packed = vec![0_u8; rows * cols / 2];
        let mut reference = vec![0.0_f32; rows * cols];

        for row in 0..rows {
            for block in 0..blocks {
                let span = &data[row * cols + block * 16..row * cols + block * 16 + 16];
                // `clamp(amax(|blk|) / F4_E2M1_MAX / per_tensor_scale, max=448).to(e4m3)`.
                let block_amax = span.iter().fold(0.0_f32, |max, v| max.max(v.abs()));
                let sf = (block_amax / 6.0 / global).min(448.0);
                let sf_byte = float8::F8E4M3::from_f32(sf).to_bits();
                blocked_scales[reference_to_blocked_index(blocks, row, block)] = sf_byte;

                let element_scale = float8::F8E4M3::from_bits(sf_byte).to_f32() * global;
                for (offset, value) in span.iter().enumerate() {
                    let col = block * 16 + offset;
                    let code = if element_scale > 0.0 {
                        reference_e2m1_code(value / element_scale)
                    } else {
                        0
                    };
                    // `packed = (even << 4) | odd` — float.py line 95: even element, high nibble.
                    let byte = &mut packed[(row * cols + col) / 2];
                    if col.is_multiple_of(2) {
                        *byte = (*byte & 0x0F) | (code << 4);
                    } else {
                        *byte = (*byte & 0xF0) | code;
                    }
                    let signed = if code & 0x08 != 0 { -1.0 } else { 1.0 };
                    reference[row * cols + col] = signed
                        * reference_e2m1_magnitudes()[(code & 0x07) as usize]
                        * element_scale;
                }
            }
        }
        (packed, blocked_scales, global, reference)
    }

    /// A deterministic spread of weights with a wide dynamic range across blocks, so per-block
    /// scales genuinely differ (a decode that used one scale everywhere would fail).
    fn nvfp4_fixture_values(rows: usize, cols: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(rows * cols);
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        for row in 0..rows {
            for col in 0..cols {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = ((state >> 40) as f32 / (1_u64 << 24) as f32) * 2.0 - 1.0;
                // Block-dependent decade so each 16-block gets its own exponent range.
                let decade = 2f32.powi(((row / 8 + col / 16) % 7) as i32 - 3);
                values.push(unit * decade);
            }
        }
        values
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// bf16 value of `x` (round-to-nearest-even) — the reference resident form.
    fn to_bf16(x: f32) -> f32 {
        if x.is_nan() {
            return x;
        }
        let bits = x.to_bits();
        f32::from_bits(bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000)
    }

    fn dense_f32(weights: &LogicalWeights, key: &str) -> Vec<f32> {
        match weights.tensors.get(key).expect("logical tensor") {
            LogicalTensor::Dense(tensor) => tensor
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            LogicalTensor::PackedConvRot { .. }
            | LogicalTensor::PackedMxFp8 { .. }
            | LogicalTensor::PackedScalarFp8 { .. }
            | LogicalTensor::PackedNvfp4 { .. } => {
                panic!("{key} unexpectedly packed")
            }
        }
    }

    fn plan_dense(path: &Path) -> LogicalWeightPlan {
        plan_logical_weights(path, &StripModel, &CandleCodecResidency::DENSE).expect("plan")
    }

    #[test]
    fn catalog_registration_carries_the_shared_baseline_codec_table() {
        let registry = baseline_codec_registry();
        let mut registered: Vec<&str> = registry.codecs().map(|codec| codec.codec_id).collect();
        registered.sort_unstable();
        let mut implemented = CODEC_IMPLEMENTATION_IDS.to_vec();
        implemented.sort_unstable();
        assert_eq!(registered, implemented);
        let catalog = register_checkpoint_codecs(ProviderRegistryBuilder::new())
            .build()
            .expect("codec table registers into an empty catalog");
        assert_eq!(
            catalog
                .checkpoint_codecs()
                .codecs()
                .copied()
                .collect::<Vec<_>>(),
            BASELINE_CODECS
        );
        // The table itself is gen-core's, not this crate's: a codec is a backend-portable
        // declaration, and two hand-kept copies could drift into a checkpoint one engine plans and
        // the other refuses with nothing comparing them.
        assert_eq!(
            BASELINE_CODECS,
            gen_core::checkpoint_codec::BASELINE_CHECKPOINT_CODECS
        );
    }

    #[test]
    fn raw_fp8_decode_and_companion_payload_reads_emit_source_progress() {
        let dir = fixture_dir();
        let path = dir.path().join("raw-fp8.safetensors");
        let descriptor = br#"{"format":"float8_e4m3fn"}"#;
        write_safetensors(
            &path,
            &[
                ("model.q.weight", "F8_E4M3", &[2, 2], vec![0x38; 4]),
                (
                    "model.q.weight_scale",
                    "F32",
                    &[],
                    1f32.to_le_bytes().to_vec(),
                ),
                (
                    "model.q.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
            ],
        );
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let out = events.clone();
        let _observer =
            gen_core::LoadActivityObserver::new(move |v| crate::lock_recover(&out).push(v));
        let plan = plan_dense(&path);
        assert!(
            crate::lock_recover(&events)
                .iter()
                .flatten()
                .any(|v| v.bytes_read >= descriptor.len() as u64),
            "planning reads the descriptor through the neutral source reader"
        );
        crate::lock_recover(&events).clear();
        let weights = read_logical_weights(&path, &plan, &Device::Cpu).unwrap();
        assert_eq!(dense_f32(&weights, "q.weight"), vec![1.; 4]);
        let reports = crate::lock_recover(&events);
        let file = std::fs::read(&path).unwrap();
        let header = 8 + u64::from_le_bytes(file[..8].try_into().unwrap());
        assert_eq!(
            reports.iter().flatten().map(|v| v.bytes_read).max(),
            Some(header + 8),
            "raw FP8 codes and scalar companion are source reads even without TensorView::load"
        );
        assert_eq!(reports.last(), Some(&None));
    }

    /// Golden: `fp8-e4m3-scalar-v1` dense fallback decodes exactly to the gen-core reference
    /// (spec-derived; the fixture covers 1.0, max 448, min normal, min subnormal, negatives, ±0).
    #[test]
    fn fp8_e4m3_scalar_golden_decodes_exactly_on_the_dense_fallback() {
        let dir = fixture_dir();
        let path = dir.path().join("e4m3.safetensors");
        let codes: Vec<u8> = vec![0x38, 0x7E, 0x08, 0x01, 0xB9, 0xC0, 0x00, 0x80];
        let scale = 0.03125_f32;
        let descriptor = br#"{"format": "float8_e4m3fn"}"#;
        write_safetensors(
            &path,
            &[
                ("model.q.weight", "F8_E4M3", &[2, 4], codes.clone()),
                (
                    "model.q.weight_scale",
                    "F32",
                    &[],
                    scale.to_le_bytes().to_vec(),
                ),
                (
                    "model.q.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
            ],
        );
        let plan = plan_dense(&path);
        assert_eq!(plan.codec_ids(), ["fp8-e4m3-scalar-v1"]);
        let weights = read_logical_weights(&path, &plan, &Device::Cpu).expect("read");
        let mut expected = Vec::new();
        gen_core::decode_fp8_e4m3fn_scalar(&codes, scale, &mut expected);
        let expected: Vec<f32> = expected.into_iter().map(to_bf16).collect();
        assert_eq!(dense_f32(&weights, "q.weight"), expected);
        assert_eq!(weights.receipt.resident_bytes(), plan.resident_bytes());
        assert_eq!(weights.receipt.resident_bytes(), 16);
    }

    #[test]
    fn scalar_dense_f32_preserves_source_precision_and_prices_the_actual_operand() {
        let dir = fixture_dir();
        let path = dir.path().join("source-f32.safetensors");
        let scale = 1.0_f32 + 1.0 / 512.;
        let desc = br#"{"format":"float8_e4m3fn"}"#;
        write_safetensors(
            &path,
            &[
                ("model.q.weight", "F8_E4M3", &[4, 4], vec![0x38; 16]),
                (
                    "model.q.weight_scale",
                    "F32",
                    &[],
                    scale.to_le_bytes().to_vec(),
                ),
                ("model.q.comfy_quant", "U8", &[desc.len()], desc.to_vec()),
                (
                    "model.direct.weight",
                    "F32",
                    &[1, 1],
                    scale.to_le_bytes().to_vec(),
                ),
            ],
        );
        let reader = LogicalWeightReader::open(&path, plan_dense(&path), &Device::Cpu).unwrap();
        for (name, count) in [("q.weight", 16), ("direct.weight", 1)] {
            let weight = reader.read_dense(name, DType::F32, &Device::Cpu).unwrap();
            assert_eq!(
                weight.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                vec![scale; count]
            );
        }
        let facts = reader.checkpoint_weight_facts().unwrap();
        assert_eq!(facts.receipt().resident_bytes(), 68);
    }

    /// Golden: `fp8-e5m2-scalar-v1` (scaled) plus the undescribed plain cast (unit scale).
    #[test]
    fn fp8_e5m2_scalar_and_plain_cast_golden_decode_exactly() {
        let dir = fixture_dir();
        let path = dir.path().join("e5m2.safetensors");
        let codes: Vec<u8> = vec![0x3C, 0x7B, 0x04, 0x01, 0xBF, 0x00, 0x80, 0x42];
        let scale = 0.25_f32;
        let descriptor = br#"{"format": "float8_e5m2"}"#;
        let plain: Vec<u8> = vec![0x3C, 0x40, 0xC2, 0x01];
        write_safetensors(
            &path,
            &[
                ("model.k.weight", "F8_E5M2", &[2, 4], codes.clone()),
                (
                    "model.k.weight_scale",
                    "F32",
                    &[1],
                    scale.to_le_bytes().to_vec(),
                ),
                (
                    "model.k.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
                ("model.k.bias", "F8_E5M2", &[4], plain.clone()),
            ],
        );
        let plan = plan_dense(&path);
        assert_eq!(plan.codec_ids(), ["fp8-e5m2-scalar-v1"]);
        let weights = read_logical_weights(&path, &plan, &Device::Cpu).expect("read");
        let mut expected = Vec::new();
        gen_core::decode_fp8_e5m2_scalar(&codes, scale, &mut expected);
        let expected: Vec<f32> = expected.into_iter().map(to_bf16).collect();
        assert_eq!(dense_f32(&weights, "k.weight"), expected);
        let mut expected_plain = Vec::new();
        gen_core::decode_fp8_e5m2_scalar(&plain, 1.0, &mut expected_plain);
        let expected_plain: Vec<f32> = expected_plain.into_iter().map(to_bf16).collect();
        assert_eq!(dense_f32(&weights, "k.bias"), expected_plain);
        assert_eq!(weights.receipt.resident_bytes(), plan.resident_bytes());
    }

    /// Golden: `mxfp8-v1` with a declared logical shape [37, 70] inside stored [64, 96] — padding
    /// rows, a non-block-aligned column tail, poison in the padding, swizzled scales.
    ///
    /// Run for **both** on-disk spellings of the block-scale companion. `F8_E8M0` is what ComfyUI
    /// actually writes (`torch.float8_e8m0fnu`) and is the case that matters here: candle reads
    /// through `MmapedSafetensors` — the `safetensors` crate's parser, not gen-core's header
    /// reader — so a fixture that only ever stored `U8` proved nothing about the real dtype
    /// surviving that second parser. `U8` stays covered because re-serialized checkpoints in the
    /// wild carry the scales that way.
    #[test]
    fn mxfp8_golden_unswizzles_unpads_and_decodes_exactly() {
        for scale_dtype in ["F8_E8M0", "U8"] {
            mxfp8_golden_case(scale_dtype);
        }
    }

    fn mxfp8_golden_case(scale_dtype: &str) {
        struct DeclaredShape;
        impl LogicalKeyMapping for DeclaredShape {
            fn mapping_id(&self) -> &'static str {
                "declared-shape-test"
            }
            fn logical_key(&self, physical_key: &str) -> Option<String> {
                physical_key.strip_prefix("model.").map(str::to_owned)
            }
            fn logical_shape(&self, logical_key: &str) -> Option<Vec<usize>> {
                (logical_key == "v.weight").then(|| vec![37, 70])
            }
        }
        let dir = fixture_dir();
        let path = dir.path().join(format!("mxfp8-{scale_dtype}.safetensors"));
        // 64 rows exercises BOTH 32-row halves of a 128-row swizzle tile; 96 columns = 3 blocks,
        // so the swizzled scale matrix carries a padded fourth column; logical [37, 70] leaves a
        // non-block-aligned column tail (70 = 2 full blocks + 6) and 27 padded rows.
        let stored = [64_usize, 96];
        let mut values = vec![0x7E_u8; stored[0] * stored[1]];
        for row in 0..37 {
            for col in 0..70 {
                values[row * stored[1] + col] = 0x38 + ((row + col) % 4) as u8;
            }
        }
        let scale_shape = gen_core::mxfp8_scale_shape([stored[0], stored[1]]);
        let mut scales = vec![0xFF_u8; scale_shape[0] * scale_shape[1]];
        for row in 0..37 {
            for block in 0..3 {
                scales[gen_core::mxfp8_swizzled_scale_index([stored[0], stored[1]], row, block)] =
                    126 + ((row + block) % 3) as u8;
            }
        }
        let descriptor = br#"{"format": "mxfp8"}"#;
        write_safetensors(
            &path,
            &[
                ("model.v.weight", "F8_E4M3", &stored, values.clone()),
                (
                    "model.v.weight_scale",
                    scale_dtype,
                    &scale_shape,
                    scales.clone(),
                ),
                (
                    "model.v.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
            ],
        );
        let plan = plan_logical_weights(&path, &DeclaredShape, &CandleCodecResidency::DENSE)
            .unwrap_or_else(|error| panic!("plan ({scale_dtype} scales): {error}"));
        assert_eq!(plan.codec_ids(), ["mxfp8-v1"], "{scale_dtype}");
        assert_eq!(plan.tensors[0].shape, vec![37, 70], "{scale_dtype}");
        let weights = read_logical_weights(&path, &plan, &Device::Cpu)
            .unwrap_or_else(|error| panic!("read ({scale_dtype} scales): {error}"));
        let mut expected = Vec::new();
        gen_core::decode_mxfp8(&values, &scales, stored, [37, 70], &mut expected).unwrap();
        let expected: Vec<f32> = expected.into_iter().map(to_bf16).collect();
        let got = dense_f32(&weights, "v.weight");
        assert_eq!(got, expected, "{scale_dtype}");
        assert!(
            got.iter().all(|value| value.is_finite() && *value < 400.0),
            "poison from the padding region leaked into the logical tensor ({scale_dtype})"
        );
        assert_eq!(
            weights.receipt.resident_bytes(),
            plan.resident_bytes(),
            "{scale_dtype}"
        );
        assert_eq!(
            weights.receipt.resident_bytes(),
            37 * 70 * 2,
            "{scale_dtype}"
        );
    }

    /// The int8 per-row row plus a mixed file: dense bf16 + e4m3 + int8 dispatch per layer.
    #[test]
    fn mixed_checkpoint_dispatches_per_layer_including_int8_per_row() {
        let dir = fixture_dir();
        let path = dir.path().join("mixed.safetensors");
        let e4m3_descriptor = br#"{"format": "float8_e4m3fn"}"#;
        let int8_descriptor = br#"{"format": "int8_tensorwise", "per_row": true}"#;
        write_safetensors(
            &path,
            &[
                ("model.dense.weight", "BF16", &[2], bf16_bytes(&[1.0, -1.0])),
                (
                    "model.q.weight",
                    "F8_E4M3",
                    &[1, 4],
                    vec![0x38, 0x40, 0x48, 0xB8],
                ),
                (
                    "model.q.weight_scale",
                    "F32",
                    &[],
                    2.0_f32.to_le_bytes().to_vec(),
                ),
                (
                    "model.q.comfy_quant",
                    "U8",
                    &[e4m3_descriptor.len()],
                    e4m3_descriptor.to_vec(),
                ),
                (
                    "model.o.weight",
                    "I8",
                    &[2, 3],
                    vec![1, 0xFE, 3, 0xFC, 5, 0xFA],
                ),
                (
                    "model.o.weight_scale",
                    "F32",
                    &[2, 1],
                    [0.5_f32, 2.0]
                        .iter()
                        .flat_map(|scale| scale.to_le_bytes())
                        .collect(),
                ),
                (
                    "model.o.comfy_quant",
                    "U8",
                    &[int8_descriptor.len()],
                    int8_descriptor.to_vec(),
                ),
            ],
        );
        let plan = plan_dense(&path);
        assert_eq!(
            plan.codec_ids(),
            ["dense-bf16-v1", "fp8-e4m3-scalar-v1", "int8-per-row-v1"]
        );
        let weights = read_logical_weights(&path, &plan, &Device::Cpu).expect("read");
        assert_eq!(dense_f32(&weights, "dense.weight"), [1.0, -1.0]);
        assert_eq!(dense_f32(&weights, "q.weight"), [2.0, 4.0, 8.0, -2.0]);
        assert_eq!(
            dense_f32(&weights, "o.weight"),
            [0.5, -1.0, 1.5, -8.0, 10.0, -12.0]
        );
        assert_eq!(weights.receipt.residency.len(), 3);
        assert_eq!(weights.receipt.resident_bytes(), plan.resident_bytes());
        let total: u64 = weights
            .receipt
            .residency
            .iter()
            .map(|report| report.source_bytes)
            .sum();
        assert_eq!(total, plan.source_bytes);
    }

    /// sc-20651 feature-end review (minor): the three **matrix** decode arms (MXFP8, NVFP4, int8
    /// per-row) index `tensor.shape[0]` / `[1]` positionally and hand the pair to `gen_core`'s
    /// `[rows, cols]` reference decoders. The plan compiler enforces rank 2 for every
    /// descriptor-bearing layer, so a compiler-built plan cannot reach them at another rank — but
    /// [`read_logical_weights`] is a public entry over a [`LogicalWeightPlan`] whose fields are
    /// public, and a plan built or edited outside the compiler used to **panic** on the index
    /// rather than refuse by name.
    ///
    /// Each arm is driven with a rank-1 and a rank-3 logical shape over a real, otherwise-valid
    /// file — the unmutated plan is read first, so a refusal below is the rank guard and nothing
    /// else.
    #[test]
    fn a_wrong_rank_matrix_plan_refuses_by_name_instead_of_panicking() {
        let dir = fixture_dir();

        // int8 per-row: a [2, 3] I8 layer with two row scales.
        let int8_path = dir.path().join("rank-int8.safetensors");
        let int8_descriptor = br#"{"format": "int8_tensorwise", "per_row": true}"#;
        write_safetensors(
            &int8_path,
            &[
                (
                    "model.o.weight",
                    "I8",
                    &[2, 3],
                    vec![1, 0xFE, 3, 0xFC, 5, 0xFA],
                ),
                (
                    "model.o.weight_scale",
                    "F32",
                    &[2, 1],
                    [0.5_f32, 2.0]
                        .iter()
                        .flat_map(|scale| scale.to_le_bytes())
                        .collect(),
                ),
                (
                    "model.o.comfy_quant",
                    "U8",
                    &[int8_descriptor.len()],
                    int8_descriptor.to_vec(),
                ),
            ],
        );
        let int8_plan = plan_dense(&int8_path);
        assert_eq!(int8_plan.codec_ids(), ["int8-per-row-v1"]);

        // MXFP8: a 32-aligned [32, 32] grid the adapter declares unpadded (so the undeclared-padding
        // refusal is not what fires below).
        let mxfp8_path = dir.path().join("rank-mxfp8.safetensors");
        let mxfp8_scales = gen_core::mxfp8_scale_shape([32, 32]);
        let mxfp8_descriptor = br#"{"format": "mxfp8"}"#;
        write_safetensors(
            &mxfp8_path,
            &[
                (
                    "model.v.weight",
                    "F8_E4M3",
                    &[32, 32],
                    vec![0x38_u8; 32 * 32],
                ),
                (
                    "model.v.weight_scale",
                    "F8_E8M0",
                    &mxfp8_scales,
                    vec![127_u8; mxfp8_scales[0] * mxfp8_scales[1]],
                ),
                (
                    "model.v.comfy_quant",
                    "U8",
                    &[mxfp8_descriptor.len()],
                    mxfp8_descriptor.to_vec(),
                ),
            ],
        );
        let mxfp8_plan = plan_logical_weights(
            &mxfp8_path,
            &StripModelDeclaring(vec![32, 32]),
            &CandleCodecResidency::DENSE,
        )
        .expect("mxfp8 plan");
        assert_eq!(mxfp8_plan.codec_ids(), ["mxfp8-v1"]);

        // NVFP4: the unpadded [256, 128] fixture the dense golden uses.
        let nvfp4_path = dir.path().join("rank-nvfp4.safetensors");
        let (rows, cols) = (256_usize, 128_usize);
        let (packed, scales, global, _) =
            nvfp4_reference_quantize(&nvfp4_fixture_values(rows, cols), rows, cols);
        write_safetensors_with_metadata(
            &nvfp4_path,
            &[
                ("model.q.weight", "U8", &[rows, cols / 2], packed),
                ("model.q.weight_scale", "F8_E4M3", &[256, 8], scales),
                (
                    "model.q.weight_scale_2",
                    "F32",
                    &[],
                    global.to_le_bytes().to_vec(),
                ),
            ],
            &[(
                "_quantization_metadata",
                r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4"}}}"#,
            )],
        );
        let nvfp4_plan = plan_logical_weights(
            &nvfp4_path,
            &StripModelDeclaring(vec![rows, cols]),
            &CandleCodecResidency::DENSE,
        )
        .expect("nvfp4 plan");
        assert_eq!(nvfp4_plan.codec_ids(), ["nvfp4-v1"]);

        for (codec, path, plan) in [
            ("int8-per-row-v1", int8_path.as_path(), &int8_plan),
            ("mxfp8-v1", mxfp8_path.as_path(), &mxfp8_plan),
            ("nvfp4-v1", nvfp4_path.as_path(), &nvfp4_plan),
        ] {
            read_logical_weights(path, plan, &Device::Cpu).unwrap_or_else(|error| {
                panic!("{codec}: the unmutated fixture must read: {error}")
            });

            for wrong in [vec![6_usize], vec![1_usize, 2, 3]] {
                let mut forced = plan.clone();
                forced.tensors[0].shape = wrong.clone();
                let error = read_logical_weights(path, &forced, &Device::Cpu)
                    .err()
                    .unwrap_or_else(|| panic!("{codec}: a rank-{} plan must refuse", wrong.len()))
                    .to_string();
                assert!(
                    error.contains(&format!("codec {codec}:"))
                        && error.contains("expected rank 2")
                        && error.contains(&format!("observed rank {}", wrong.len())),
                    "{codec}: the rank-{} refusal must name the codec, the expected rank and the \
                     observed rank, got: {error}",
                    wrong.len()
                );
            }

            // sc-21547 review (major): the arms index `source_shape()`, not `shape`. A plan whose
            // `shape` is a perfectly good rank-2 grid but whose `transform.source_shape` is rank 1
            // slips past a `shape`-only guard and then panics on `source_shape[1]`. The transform
            // is a public field on a public struct, so this plan is constructible by any caller.
            let mut forced = plan.clone();
            forced.tensors[0].transform = Some(LogicalTensorTransform {
                source_shape: vec![2_usize],
                rows: RowRange { start: 0, len: 1 },
                half_swap: false,
            });
            assert_eq!(
                forced.tensors[0].shape.len(),
                2,
                "{codec}: the shape stays rank 2, so only the source-shape rank can refuse below"
            );
            let error = read_logical_weights(path, &forced, &Device::Cpu)
                .err()
                .unwrap_or_else(|| {
                    panic!("{codec}: a rank-1 transform.source_shape must refuse, not panic")
                })
                .to_string();
            assert!(
                error.contains(&format!("codec {codec}:"))
                    && error.contains("expected rank 2")
                    && error.contains("observed rank 1"),
                "{codec}: the source-shape rank refusal must name the codec and both ranks, got: \
                 {error}"
            );
        }
    }

    /// sc-21547 review (major): a `half_swap` on a **packed** decode must be a real `Err`, not a
    /// `debug_assert!`.
    ///
    /// The plan compiler refuses that combination, but [`LogicalTensorPlan::transform`] is a public
    /// field on a public struct reached through the public [`read_logical_weights`] entry, so a
    /// plan that did not come from the compiler can carry one. A `debug_assert!` compiles out under
    /// `--release`, and the packed arms would then return the rows **unswapped** under the swapped
    /// logical key — plausible-but-wrong weights that every downstream shape check agrees with.
    ///
    /// This drives [`apply_transform`] directly with hand-made packed decodes (both packed
    /// variants) because that is the only way to reach the arm without a CUDA device, and it
    /// asserts on the refusal rather than on a debug-only panic, so it is a real assertion in both
    /// profiles.
    #[test]
    fn a_half_swap_on_a_packed_decode_refuses_in_every_profile() {
        fn packed_plan(codec_id: &'static str, logical_key: &str) -> LogicalTensorPlan {
            LogicalTensorPlan {
                logical_key: logical_key.to_string(),
                physical_key: "model.qkv.weight".to_string(),
                encoding: WeightEncoding::Fp8E4M3,
                shape: vec![128, 64],
                source_bytes: 128 * 64,
                codec_id,
                resident_encoding: WeightEncoding::Fp8E4M3,
                codec: TensorCodecSpec::Dense,
                residency: PlannedResidency {
                    mode: ResidencyMode::Packed,
                    resident_bytes: 128 * 64,
                },
                transform: Some(LogicalTensorTransform {
                    source_shape: vec![256, 64],
                    rows: RowRange { start: 0, len: 128 },
                    half_swap: true,
                }),
            }
        }

        // Packed fp8: the codes tensor's dtype is irrelevant to the transform arm.
        let fp8_plan = packed_plan(FP8_E4M3_SCALAR_CODEC.codec_id, "q.weight");
        let bytes: Vec<u8> = (0..256)
            .flat_map(|row| std::iter::repeat_n(row as u8, 64))
            .collect();
        let original = Tensor::from_vec(bytes, (256, 64), &Device::Cpu).unwrap();
        let LogicalTensor::PackedScalarFp8 {
            codes,
            weight_scale,
            ..
        } = apply_transform(
            &fp8_plan,
            LogicalTensor::PackedScalarFp8 {
                codes: original,
                weight_scale: 2.0,
                weight_scale_from_companion: true,
                input_scale: None,
            },
        )
        .unwrap()
        else {
            panic!("scalar code bytes must stay packed");
        };
        let result = codes.to_vec2::<u8>().unwrap();
        for (row, col) in result.iter().enumerate() {
            assert!(col.iter().all(|&c| c == ((row + 64) % 128) as u8));
        }
        assert_eq!(weight_scale, 2.0);

        // Packed NVFP4: the host container, so this leg runs on every lane.
        let nvfp4_plan = packed_plan(NVFP4_CODEC.codec_id, "k.weight");
        let packed = Nvfp4Tensor::pack_from_slice(&vec![0.5_f32; 256 * 64], 256, 64)
            .expect("pack nvfp4 fixture");
        let error = apply_transform(
            &nvfp4_plan,
            LogicalTensor::PackedNvfp4 {
                tensor: Box::new(packed),
                input_scale: None,
            },
        )
        .err()
        .expect("a half swap on a packed NVFP4 decode must refuse")
        .to_string();
        assert!(
            error.contains(&format!("codec {}:", NVFP4_CODEC.codec_id))
                && error.contains("\"model.qkv.weight\"")
                && error.contains("\"k.weight\"")
                && error.contains("half swap")
                && error.contains("packed-native residency"),
            "the packed-NVFP4 refusal must name the codec, the physical key, the logical key and \
             the half swap, got: {error}"
        );

        // The same transform on a DENSE decode is honoured, not refused — the refusal is about the
        // packed representation, not about half swaps.
        let mut dense_plan = packed_plan(DENSE_BF16_CODEC.codec_id, "v.weight");
        dense_plan.residency.mode = ResidencyMode::Dense;
        let rows = Tensor::from_vec(
            (0..256_u32).map(|row| row as f32).collect::<Vec<f32>>(),
            (256, 1),
            &Device::Cpu,
        )
        .expect("dense rows");
        let LogicalTensor::Dense(swapped) =
            apply_transform(&dense_plan, LogicalTensor::Dense(rows)).expect("dense half swap")
        else {
            panic!("a dense decode stays dense");
        };
        let values = swapped
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("values");
        assert_eq!(values.len(), 128);
        assert_eq!(values[0], 64.0, "the swap puts the second half first");
        assert_eq!(values[64], 0.0);
    }

    /// The packed-native policy is a pure layout+hardware predicate: it selects `Packed` only for
    /// the E4M3 scalar row without `full_precision_matrix_mult`, and the plan prices packed
    /// entries at stored bytes + retained scales instead of dense bf16.
    #[test]
    fn candle_residency_policy_packs_only_the_eligible_e4m3_layers_and_prices_them() {
        let dir = fixture_dir();
        let path = dir.path().join("packed-pricing.safetensors");
        let e4m3 = br#"{"format": "float8_e4m3fn"}"#;
        let e4m3_fpmm = br#"{"format": "float8_e4m3fn", "full_precision_matrix_mult": true}"#;
        let e5m2 = br#"{"format": "float8_e5m2"}"#;
        write_safetensors(
            &path,
            &[
                ("model.a.weight", "F8_E4M3", &[16, 16], vec![0x38; 256]),
                (
                    "model.a.weight_scale",
                    "F32",
                    &[],
                    2.0_f32.to_le_bytes().to_vec(),
                ),
                ("model.a.comfy_quant", "U8", &[e4m3.len()], e4m3.to_vec()),
                ("model.b.weight", "F8_E4M3", &[2, 4], vec![0x38; 8]),
                (
                    "model.b.weight_scale",
                    "F32",
                    &[],
                    2.0_f32.to_le_bytes().to_vec(),
                ),
                (
                    "model.b.comfy_quant",
                    "U8",
                    &[e4m3_fpmm.len()],
                    e4m3_fpmm.to_vec(),
                ),
                ("model.c.weight", "F8_E5M2", &[2, 4], vec![0x3C; 8]),
                (
                    "model.c.weight_scale",
                    "F32",
                    &[],
                    2.0_f32.to_le_bytes().to_vec(),
                ),
                ("model.c.comfy_quant", "U8", &[e5m2.len()], e5m2.to_vec()),
            ],
        );
        let native = CandleCodecResidency {
            nvfp4_native: false,
            mxfp8_native: false,
            fp8_e4m3_native: true,
        };
        let plan = plan_logical_weights(&path, &StripModel, &native).expect("plan");
        let by_key: BTreeMap<&str, &LogicalTensorPlan> = plan
            .tensors
            .iter()
            .map(|tensor| (tensor.logical_key.as_str(), tensor))
            .collect();
        // Eligible: packed at stored bytes (8), its 4-byte scale retained on the companion row.
        assert_eq!(by_key["a.weight"].residency.mode, ResidencyMode::Packed);
        assert_eq!(by_key["a.weight"].residency.resident_bytes, 256);
        let scale_row = plan
            .companions
            .iter()
            .find(|companion| companion.physical_key == "model.a.weight_scale")
            .unwrap();
        assert_eq!(scale_row.resident_bytes, 4);
        // full_precision_matrix_mult: dense by contract, 2 × 4 × bf16 = 16.
        assert_eq!(by_key["b.weight"].residency.mode, ResidencyMode::Dense);
        assert_eq!(by_key["b.weight"].residency.resident_bytes, 16);
        // E5M2 has no native leg: dense.
        assert_eq!(by_key["c.weight"].residency.mode, ResidencyMode::Dense);
        // Packed vs dense pricing differs end to end.
        let dense_plan = plan_dense(&path);
        assert!(plan.resident_bytes() < dense_plan.resident_bytes());
        assert_eq!(dense_plan.resident_bytes(), 512 + 2 * 16);
        assert_eq!(plan.resident_bytes(), (256 + 4) + 16 + 16);

        // The probed policy on this machine's devices: CPU (and Metal) are dense-only.
        assert_eq!(
            CandleCodecResidency::probe(&Device::Cpu),
            CandleCodecResidency::DENSE
        );

        // On a build without the CUDA fp8 leg, reading a packed plan is a typed refusal — never a
        // silent dense substitution of what admission priced as packed.
        #[cfg(not(feature = "cuda"))]
        {
            let error = match read_logical_weights(&path, &plan, &Device::Cpu) {
                Ok(_) => panic!("packed plan must refuse on a non-cuda build"),
                Err(error) => error.to_string(),
            };
            assert!(error.contains("no CUDA fp8 leg"), "{error}");
        }
        // On a CUDA build the packed read keeps F8E4M3 codes + scales resident; the read itself is
        // exercised by the windows-cuda lane (this lane's CPU device cannot hold F8E4M3 GEMM
        // operands). The *accounting* the receipt would then report is not cuda-gated, so pin it
        // here against the plan rows above: the receipt measures the retained scales off the
        // decoded value, so this number is derived independently of `plan.companions`.
        let codes = Tensor::zeros((2, 4), DType::F8E4M3, &Device::Cpu).expect("codes");
        assert_eq!(
            LogicalTensor::PackedScalarFp8 {
                codes: codes.clone(),
                weight_scale: 1.0,
                weight_scale_from_companion: true,
                input_scale: None,
            }
            .resident_bytes(),
            8 + 4,
            "packed codes + the retained weight_scale"
        );
        assert_eq!(
            LogicalTensor::PackedScalarFp8 {
                codes,
                weight_scale: 1.0,
                weight_scale_from_companion: true,
                input_scale: Some(0.5),
            }
            .resident_bytes(),
            8 + 4 + 4,
            "a retained input_scale is priced too"
        );
    }

    /// AC1 (Candle). A header-declared NVFP4 golden fixture — packed E2M1 values, block-16 FP8
    /// scales in the `to_blocked` swizzle, a global tensor scale — decodes to exactly the values an
    /// independent reference (`nvfp4_reference_quantize`, written from the format spec and the
    /// third-party `float8` crate) says it holds, on the dense fallback.
    ///
    /// Shape `[256, 128]` is deliberately **multi-atom** in both scale dimensions (2 row atoms ×
    /// 2 block atoms), so an atom-order mistake in the un-swizzle is a wrong value, not a no-op.
    /// The fixture also carries no `.comfy_quant` tensors at all: the declaration lives in
    /// `__metadata__._quantization_metadata` under a *relative* layer name, the form the ComfyUI
    /// Kitchen converters write, so the prefix resolver is on the path too.
    #[test]
    fn nvfp4_golden_decodes_exactly_on_the_dense_fallback() {
        let dir = fixture_dir();
        let path = dir.path().join("nvfp4.safetensors");
        let (rows, cols) = (256_usize, 128_usize);
        let values = nvfp4_fixture_values(rows, cols);
        let (packed, scales, global, reference) = nvfp4_reference_quantize(&values, rows, cols);
        assert_eq!(scales.len(), 256 * 8, "2 row atoms × 2 block atoms");

        write_safetensors_with_metadata(
            &path,
            &[
                ("model.q.weight", "U8", &[rows, cols / 2], packed.clone()),
                ("model.q.weight_scale", "F8_E4M3", &[256, 8], scales.clone()),
                (
                    "model.q.weight_scale_2",
                    "F32",
                    &[],
                    global.to_le_bytes().to_vec(),
                ),
            ],
            &[(
                "_quantization_metadata",
                // Relative layer name: the tensors are `model.q.*`, the declaration says `q`.
                r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4"}}}"#,
            )],
        );

        // The fixture is built at exactly [rows, cols] — no ComfyUI padding — so the adapter
        // declaring that grid is a true statement about the layer, and it is REQUIRED: without a
        // declaration the plan could not tell this grid from a padded one and refuses to
        // materialize it (sc-20651).
        let mapping = StripModelDeclaring(vec![rows, cols]);
        let plan =
            plan_logical_weights(&path, &mapping, &CandleCodecResidency::DENSE).expect("plan");
        assert_eq!(plan.codec_ids(), vec!["nvfp4-v1"]);
        let tensor = &plan.tensors[0];
        assert_eq!(tensor.shape, vec![rows, cols], "the logical element grid");
        assert_eq!(tensor.residency.mode, ResidencyMode::Dense);
        // Dense residency is priced on the logical shape × bf16, not on the stored 4-bit bytes.
        assert_eq!(
            tensor.residency.resident_bytes,
            (rows * cols * 2) as u64,
            "dense fallback holds bf16"
        );
        // Both scale levels are consumed by a dense decode, so neither is priced resident.
        assert_eq!(plan.companions.len(), 2);
        assert!(plan
            .companions
            .iter()
            .all(|companion| companion.resident_bytes == 0));
        assert_eq!(
            plan.resident_bytes(),
            (rows * cols * 2) as u64,
            "dense plan prices only the decoded weight"
        );

        let weights = read_logical_weights(&path, &plan, &Device::Cpu).expect("read");
        let decoded = dense_f32(&weights, "q.weight");
        assert_eq!(decoded.len(), rows * cols);
        for (index, expected) in reference.iter().enumerate() {
            assert_eq!(
                decoded[index],
                to_bf16(*expected),
                "element {index} (row {}, col {})",
                index / cols,
                index % cols
            );
        }
        // The receipt measures what was materialized and agrees with the plan's pricing.
        assert_eq!(weights.receipt.resident_bytes(), plan.resident_bytes());

        // Supplementary: round-trip error against the *original* f32 matrix. NVFP4 is lossy, so
        // this is a quantization-error bound, not a correctness proof — the exactness assertions
        // above are what pin the decode. Checked as a relative RMS; never a cosine.
        //
        // The band is the MEASURED error of this fixture (0.103) with headroom, not a spec figure:
        // uniform-random values are the worst case for E2M1's non-uniform 8-magnitude grid, whose
        // coarsest step (4 -> 6) is a third of its own magnitude. It exists to catch a decode that
        // drifts wholesale — a dropped scale level, the wrong global — which lands orders of
        // magnitude away, not to certify a precision claim.
        let (mut error, mut energy) = (0.0_f64, 0.0_f64);
        for (original, got) in values.iter().zip(decoded.iter()) {
            error += ((*original - *got) as f64).powi(2);
            energy += (*original as f64).powi(2);
        }
        let relative_rms = (error / energy).sqrt();
        assert!(
            (0.05..0.15).contains(&relative_rms),
            "NVFP4 relative RMS error {relative_rms} is outside the band this fixture measures \
             (0.103): far below means it stopped being quantized, far above means the decode drifted"
        );
    }

    /// AC2 (Candle, host half). The `sm_120` residency decision and the repack it selects.
    ///
    /// The hardware predicate is exercised through the one floor definition; the *repack* is host
    /// code, so a policy that reports the floor as met drives the real packed path on this lane and
    /// the resulting container decodes back to the same golden values. Live cuBLASLt execution of
    /// the resulting operands remains the windows-cuda box's job.
    #[test]
    fn nvfp4_packed_residency_repacks_and_prices_independently_of_the_dense_fallback() {
        let dir = fixture_dir();
        let path = dir.path().join("nvfp4-packed.safetensors");
        let (rows, cols) = (256_usize, 128_usize);
        let values = nvfp4_fixture_values(rows, cols);
        let (packed, scales, global, reference) = nvfp4_reference_quantize(&values, rows, cols);
        write_safetensors_with_metadata(
            &path,
            &[
                ("model.q.weight", "U8", &[rows, cols / 2], packed.clone()),
                ("model.q.weight_scale", "F8_E4M3", &[256, 8], scales),
                (
                    "model.q.weight_scale_2",
                    "F32",
                    &[],
                    global.to_le_bytes().to_vec(),
                ),
            ],
            &[(
                "_quantization_metadata",
                r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4"}}}"#,
            )],
        );

        let native = CandleCodecResidency {
            fp8_e4m3_native: false,
            nvfp4_native: true,
            mxfp8_native: false,
        };
        // Unpadded fixture; the adapter declares the grid it was built at (see above).
        let mapping = StripModelDeclaring(vec![rows, cols]);
        let plan = plan_logical_weights(&path, &mapping, &native).expect("plan");
        let tensor = &plan.tensors[0];
        assert_eq!(tensor.residency.mode, ResidencyMode::Packed);
        // Packed prices the stored 4-bit bytes; both scale levels are retained and priced on their
        // own companion rows — the two residencies are independent quantities, not a scaled copy.
        assert_eq!(tensor.residency.resident_bytes, (rows * cols / 2) as u64);
        let companions: u64 = plan
            .companions
            .iter()
            .map(|companion| companion.resident_bytes)
            .sum();
        assert_eq!(
            companions,
            (256 * 8) as u64 + 4,
            "block scales + F32 global"
        );
        assert_eq!(
            plan.resident_bytes(),
            (rows * cols / 2) as u64 + 256 * 8 + 4
        );
        let dense_plan =
            plan_logical_weights(&path, &mapping, &CandleCodecResidency::DENSE).expect("plan");
        assert!(plan.resident_bytes() < dense_plan.resident_bytes());

        // The read repacks into the container `Nvfp4Linear::from_packed_in` consumes...
        let weights = read_logical_weights(&path, &plan, &Device::Cpu).expect("read");
        let LogicalTensor::PackedNvfp4 {
            tensor: container,
            input_scale,
        } = weights.tensors.get("q.weight").expect("logical tensor")
        else {
            panic!("an sm_120-eligible NVFP4 layer must repack, not decode dense");
        };
        assert_eq!((container.rows, container.cols), (rows, cols));
        assert_eq!(container.global_scale, global);
        assert!(input_scale.is_none());
        // ...and that container still holds the golden values: the nibble swap and the block-scale
        // atom permutation are both lossless.
        let recovered = container.dequantize_to_vec();
        for (index, expected) in reference.iter().enumerate() {
            assert_eq!(recovered[index], *expected, "element {index}");
        }
        // Residency measured off the container, independently of the plan rows above.
        assert_eq!(weights.receipt.resident_bytes(), plan.resident_bytes());
    }

    /// One NVFP4 layer plus a dense sibling — the fixture the handoff-contract tests read as both a
    /// native and a non-native host.
    fn nvfp4_and_dense_fixture(path: &Path, rows: usize, cols: usize) {
        let values = nvfp4_fixture_values(rows, cols);
        let (packed, scales, global, _) = nvfp4_reference_quantize(&values, rows, cols);
        write_safetensors_with_metadata(
            path,
            &[
                ("model.q.weight", "U8", &[rows, cols / 2], packed),
                ("model.q.weight_scale", "F8_E4M3", &[256, 8], scales),
                (
                    "model.q.weight_scale_2",
                    "F32",
                    &[],
                    global.to_le_bytes().to_vec(),
                ),
                (
                    "model.norm.weight",
                    "F32",
                    &[4],
                    vec![0u8; 4 * std::mem::size_of::<f32>()],
                ),
            ],
            &[(
                "_quantization_metadata",
                r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4"}}}"#,
            )],
        );
    }

    /// **sc-21484 handoff contract: the source codec and the executed representation are separate,
    /// correlated facts.**
    ///
    /// The same file, read on two hosts. Both report the source as `nvfp4-v1` with the same tensor
    /// count and source bytes — that fact is device-independent. Only the *receipt* differs, and
    /// only the native host's receipt carries a
    /// [`ExecutionRepresentation::NativePacked`] row. This is exactly what the SceneWorks half
    /// reads to tell "stored NVFP4" from "ran NVFP4".
    #[test]
    fn checkpoint_facts_separate_the_nvfp4_source_from_the_representation_executed() {
        let dir = fixture_dir();
        let path = dir.path().join("nvfp4-facts.safetensors");
        let (rows, cols) = (256_usize, 128_usize);
        nvfp4_and_dense_fixture(&path, rows, cols);
        let mapping = StripModelDeclaring(vec![rows, cols]);

        let facts_for = |residency: CandleCodecResidency| {
            let plan = plan_logical_weights(&path, &mapping, &residency).expect("plan");
            let reader = LogicalWeightReader::open_with_capability(
                &path,
                plan.clone(),
                &Device::Cpu,
                residency.native_execution_capability(),
            )
            .expect("open");
            for tensor in &plan.tensors {
                reader.read(&tensor.logical_key).expect("read");
            }
            (plan, reader)
        };

        // ---- a non-native host: the declared dense fallback (AC3) ---------------------------
        let (dense_plan, dense_reader) = facts_for(CandleCodecResidency::DENSE);
        let dense = dense_reader.checkpoint_weight_facts().expect("facts");
        assert!(
            dense.source().declares(NVFP4_CODEC.codec_id),
            "the source is stored nvfp4-v1 on every host"
        );
        let dense_entry = dense.source().entry(NVFP4_CODEC.codec_id).expect("row");
        assert_eq!(dense_entry.tensor_count, 1);
        assert_eq!(dense_entry.planned_native_packed_tensors, 0);
        assert!(
            !dense.executes_natively(NVFP4_CODEC.codec_id),
            "a host below the sm_120 floor never labels its run native NVFP4"
        );
        assert!(dense.capability().is_dense_only());
        assert!(dense
            .materialized_as(NVFP4_CODEC.codec_id, ExecutionRepresentation::NativePacked)
            .is_none());
        let dense_row = dense
            .materialized_as(NVFP4_CODEC.codec_id, ExecutionRepresentation::DenseFallback)
            .expect("a dense-fallback row");
        assert_eq!(dense_row.tensor_count, 1);
        assert!(dense.is_complete());
        assert_eq!(dense.resident_bytes(), dense_plan.resident_bytes());

        // ---- a native host: the same source, executed packed --------------------------------
        let (packed_plan, packed_reader) = facts_for(CandleCodecResidency {
            fp8_e4m3_native: false,
            nvfp4_native: true,
            mxfp8_native: false,
        });
        let packed = packed_reader.checkpoint_weight_facts().expect("facts");
        // Fact 2 is identical across the two hosts — it is a property of the file.
        let packed_entry = packed.source().entry(NVFP4_CODEC.codec_id).expect("row");
        assert_eq!(packed_entry.tensor_count, dense_entry.tensor_count);
        assert_eq!(packed_entry.source_bytes, dense_entry.source_bytes);
        // Fact 3 is not.
        assert_eq!(packed_entry.planned_native_packed_tensors, 1);
        assert!(packed.executes_natively(NVFP4_CODEC.codec_id));
        assert!(packed
            .materialized_as(NVFP4_CODEC.codec_id, ExecutionRepresentation::DenseFallback)
            .is_none());
        assert_eq!(packed.resident_bytes(), packed_plan.resident_bytes());
        assert!(
            packed.resident_bytes() < dense.resident_bytes(),
            "packed execution is resident at the NVFP4 footprint, not the bf16 one"
        );
        // The dense sibling is its own codec row on both hosts and is never confused with either.
        assert!(packed.source().declares(DENSE_F32_CODEC.codec_id));
        assert!(!packed.executes_natively(DENSE_F32_CODEC.codec_id));
    }

    /// **sc-11045 fix round (epic E5): a `Packed`-priced row the provider demotes to the
    /// dequant-bf16 regime is reported `DenseFallback` at the measured dense bytes**, with its
    /// typed `RegimeDemotion` reconciling the receipt against the plan's packed pricing — never a
    /// native row, never packed byte counts, and the companion scales the dense decode consumed
    /// leave the packed expectation too.
    #[test]
    fn a_demoted_packed_row_reports_the_dense_fallback_truthfully() {
        use gen_core::checkpoint_codec::LogicalReadMaterialization;

        let dir = fixture_dir();
        let path = dir.path().join("nvfp4-demotion.safetensors");
        let (rows, cols) = (256_usize, 128_usize);
        nvfp4_and_dense_fixture(&path, rows, cols);
        let mapping = StripModelDeclaring(vec![rows, cols]);
        let residency = CandleCodecResidency {
            fp8_e4m3_native: false,
            nvfp4_native: true,
            mxfp8_native: false,
        };
        let plan = plan_logical_weights(&path, &mapping, &residency).expect("plan");
        let reader = LogicalWeightReader::open_with_capability(
            &path,
            plan.clone(),
            &Device::Cpu,
            residency.native_execution_capability(),
        )
        .expect("open");
        for tensor in &plan.tensors {
            reader.read(&tensor.logical_key).expect("read");
        }

        // A dense-priced row has no packed pricing to demote from — typed refusal.
        assert!(reader
            .demote_to_dense_fallback("norm.weight", 16)
            .unwrap_err()
            .to_string()
            .contains("priced Dense"));
        // An unknown row refuses by name.
        assert!(reader
            .demote_to_dense_fallback("ghost.weight", 16)
            .unwrap_err()
            .to_string()
            .contains("no entry in the compiled plan"));

        // The provider's W4A16 construction: the packed container dequantized to a resident dense
        // bf16 [rows, cols] weight.
        let dense_bytes = (rows * cols * 2) as u64;
        reader
            .demote_to_dense_fallback("q.weight", dense_bytes)
            .expect("demote the W4A16 row");
        // Idempotent at the same measurement; a contradicting re-demotion refuses.
        reader
            .demote_to_dense_fallback("q.weight", dense_bytes)
            .expect("re-demoting at the same bytes is idempotent");
        assert!(reader
            .demote_to_dense_fallback("q.weight", dense_bytes + 1)
            .unwrap_err()
            .to_string()
            .contains("contradicts the first measurement"));

        let receipt = reader.receipt();
        assert_eq!(
            receipt.materialization,
            LogicalReadMaterialization::Materialized
        );
        assert_eq!(receipt.demotions.len(), 1);
        assert_eq!(receipt.demotions[0].logical_key, "q.weight");
        assert_eq!(receipt.demotions[0].resident_bytes, dense_bytes);

        let facts = reader.checkpoint_weight_facts().expect(
            "a demoted row with its typed accounting must reconcile against the packed plan",
        );
        assert!(facts.source().declares(NVFP4_CODEC.codec_id));
        assert!(
            !facts.executes_natively(NVFP4_CODEC.codec_id),
            "the demoted run executed nothing natively"
        );
        assert!(facts
            .materialized_as(NVFP4_CODEC.codec_id, ExecutionRepresentation::NativePacked)
            .is_none());
        let row = facts
            .materialized_as(NVFP4_CODEC.codec_id, ExecutionRepresentation::DenseFallback)
            .expect("the demoted row reports dense-fallback");
        assert_eq!(row.tensor_count, 1);
        assert_eq!(
            row.resident_bytes, dense_bytes,
            "the receipt carries the measured dense bytes, not the plan's packed pricing"
        );
        assert!(facts.is_complete());
    }

    /// **sc-21484 mutation gate.** Two doctored receipts the handoff contract must reject, run
    /// against the engine's own dense-host read:
    ///
    /// * relabel the dense-fallback NVFP4 row `native-packed` — the lie AC3 forbids;
    /// * alias the source codec to `int8-per-row-v1` — the "call it q4" lie AC2 forbids.
    ///
    /// Both are rejected by [`CheckpointWeightFacts::new`], which is the only constructor, so a
    /// consumer cannot be handed either.
    #[test]
    fn a_doctored_receipt_cannot_relabel_the_representation_or_alias_the_source() {
        use gen_core::checkpoint_facts::CheckpointWeightFactsError;

        let dir = fixture_dir();
        let path = dir.path().join("nvfp4-mutation.safetensors");
        let (rows, cols) = (256_usize, 128_usize);
        nvfp4_and_dense_fixture(&path, rows, cols);
        let mapping = StripModelDeclaring(vec![rows, cols]);
        let plan =
            plan_logical_weights(&path, &mapping, &CandleCodecResidency::DENSE).expect("plan");
        let capability = CandleCodecResidency::DENSE.native_execution_capability();
        let reader = LogicalWeightReader::open_with_capability(
            &path,
            plan.clone(),
            &Device::Cpu,
            capability.clone(),
        )
        .expect("open");
        for tensor in &plan.tensors {
            reader.read(&tensor.logical_key).expect("read");
        }
        let honest = reader.receipt();
        // The honest receipt validates.
        CheckpointWeightFacts::new(&plan, capability.clone(), honest.clone()).expect("honest");

        let nvfp4_row = |receipt: &LogicalWeightReceipt| {
            receipt
                .residency
                .iter()
                .position(|row| row.codec_id == NVFP4_CODEC.codec_id)
                .expect("the read has an nvfp4 row")
        };

        // Mutation 1: this dense decode, labelled native.
        let mut relabelled = honest.clone();
        let index = nvfp4_row(&relabelled);
        relabelled.residency[index].representation = ExecutionRepresentation::NativePacked;
        let error = CheckpointWeightFacts::new(&plan, capability.clone(), relabelled)
            .expect_err("a dense fallback must not be labelled native");
        assert_eq!(
            error,
            CheckpointWeightFactsError::NativeWithoutCapability {
                codec_id: NVFP4_CODEC.codec_id,
                tensor_count: 1,
            }
        );

        // Mutation 2: this NVFP4 source, aliased to the int-affine row a `q4` product fact would
        // carry.
        let mut aliased = honest;
        let index = nvfp4_row(&aliased);
        aliased.residency[index].codec_id = INT8_PER_ROW_CODEC.codec_id;
        let error = CheckpointWeightFacts::new(&plan, capability, aliased)
            .expect_err("a receipt must not re-label the source codec");
        assert_eq!(
            error,
            CheckpointWeightFactsError::UnplannedCodec {
                codec_id: INT8_PER_ROW_CODEC.codec_id,
            }
        );
    }

    /// **AC3: `sm_100` is excluded from the native gate, and the capability says so.**
    ///
    /// Datacenter Blackwell reports compute capability `(10, 0)`, which is numerically *below* the
    /// consumer `sm_120` floor this leg's `VEC16_UE4M3` kernel needs. The exclusion is the floor
    /// comparison itself — asserted here on the shared predicate and on the rendered capability, so
    /// a future floor edit that let `sm_100` through reds this test rather than shipping a host
    /// that labels a run it cannot execute.
    #[test]
    fn the_nvfp4_native_gate_excludes_sm_100_and_the_capability_reflects_the_floor() {
        use crate::quant::{compute_cap_meets_fp8_floor, compute_cap_meets_nvfp4_floor};

        assert!(
            !compute_cap_meets_nvfp4_floor((10, 0)),
            "datacenter sm_100 is outside this leg"
        );
        assert!(!compute_cap_meets_nvfp4_floor((11, 9)));
        assert!(compute_cap_meets_nvfp4_floor((12, 0)));

        // A host at sm_100 probes `nvfp4_native: false`, so its capability lists nothing and no
        // receipt it produces can carry a native NVFP4 row.
        let sm_100 = CandleCodecResidency {
            fp8_e4m3_native: compute_cap_meets_fp8_floor((10, 0)),
            nvfp4_native: compute_cap_meets_nvfp4_floor((10, 0)),
            mxfp8_native: true,
        };
        let capability = sm_100.native_execution_capability();
        assert!(!capability.executes_natively(NVFP4_CODEC.codec_id));
        // sm_100 clears the fp8 sm_89 floor, so the capability is not vacuously empty — the NVFP4
        // absence is a real, codec-specific exclusion.
        assert!(capability.executes_natively(FP8_E4M3_SCALAR_CODEC.codec_id));

        let sm_120 = CandleCodecResidency {
            fp8_e4m3_native: compute_cap_meets_fp8_floor((12, 0)),
            nvfp4_native: compute_cap_meets_nvfp4_floor((12, 0)),
            mxfp8_native: true,
        };
        assert!(sm_120
            .native_execution_capability()
            .executes_natively(NVFP4_CODEC.codec_id));
        assert!(CandleCodecResidency::DENSE
            .native_execution_capability()
            .is_dense_only());
    }

    /// sc-20641 review. ComfyUI pads NVFP4 storage to 16, so a layer can be stored WIDER than it
    /// is — `in_features = 60` stores `K = 64`, which passes the FP4 leg's 32-alignment rule. The
    /// packed container holds the stored grid and carries no unpad, so repacking such a layer hands
    /// `Nvfp4Linear` four columns of padding as real contraction elements.
    ///
    /// Two independent guards, tested separately: the residency policy plans **Dense** on sm_120
    /// hardware, and the decoder's Packed arm **refuses** if a plan ever asks for it anyway.
    #[test]
    fn a_padded_nvfp4_layer_plans_dense_and_a_forced_packed_decode_refuses() {
        struct DeclaresSixty;
        impl LogicalKeyMapping for DeclaresSixty {
            fn mapping_id(&self) -> &'static str {
                "declares-sixty-test"
            }
            fn logical_key(&self, physical_key: &str) -> Option<String> {
                physical_key.strip_prefix("model.").map(str::to_owned)
            }
            fn logical_shape(&self, logical_key: &str) -> Option<Vec<usize>> {
                (logical_key == "q.weight").then(|| vec![256, 60])
            }
        }
        /// Ignores every layout fact — stands in for a residency policy that regresses.
        struct ForcePacked;
        impl CodecResidencyPolicy for ForcePacked {
            fn residency(
                &self,
                _codec: &CheckpointCodecRegistration,
                _spec: &TensorCodecSpec,
                _stored_shape: &[usize],
            ) -> ResidencyMode {
                ResidencyMode::Packed
            }
        }

        let dir = fixture_dir();
        let path = dir.path().join("nvfp4-padded.safetensors");
        let (rows, stored_cols, logical_cols) = (256_usize, 64_usize, 60_usize);
        // The pad columns must hold the zero code, which is what the geometry validator demands.
        let mut values = nvfp4_fixture_values(rows, stored_cols);
        for row in 0..rows {
            for col in logical_cols..stored_cols {
                values[row * stored_cols + col] = 0.0;
            }
        }
        let (packed, scales, global, _) = nvfp4_reference_quantize(&values, rows, stored_cols);
        assert_eq!(scales.len(), 256 * 4, "2 row atoms × 1 block atom");
        write_safetensors_with_metadata(
            &path,
            &[
                (
                    "model.q.weight",
                    "U8",
                    &[rows, stored_cols / 2],
                    packed.clone(),
                ),
                ("model.q.weight_scale", "F8_E4M3", &[256, 4], scales),
                (
                    "model.q.weight_scale_2",
                    "F32",
                    &[],
                    global.to_le_bytes().to_vec(),
                ),
            ],
            &[(
                "_quantization_metadata",
                r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4"}}}"#,
            )],
        );

        let native = CandleCodecResidency {
            fp8_e4m3_native: false,
            nvfp4_native: true,
            mxfp8_native: false,
        };
        let plan = plan_logical_weights(&path, &DeclaresSixty, &native).expect("plan");
        let tensor = &plan.tensors[0];
        assert_eq!(tensor.shape, vec![rows, logical_cols], "the unpadded layer");
        assert!(
            matches!(
                tensor.codec,
                TensorCodecSpec::Nvfp4 {
                    stored_shape: [256, 64],
                    logical_shape: [256, 60],
                    ..
                }
            ),
            "the spec carries BOTH shapes: {:?}",
            tensor.codec
        );
        assert_eq!(
            tensor.residency.mode,
            ResidencyMode::Dense,
            "a padded layer takes the dense fallback even on sm_120 with an aligned stored grid"
        );
        // Dense is the honest plan and it reads back correctly at the logical width.
        let weights = read_logical_weights(&path, &plan, &Device::Cpu).expect("dense read");
        assert_eq!(
            dense_f32(&weights, "q.weight").len(),
            rows * logical_cols,
            "the dense fallback unpads"
        );

        // The decoder does not trust the policy: a Packed plan for this layer is refused, naming
        // both shapes, rather than silently widening the layer.
        let forced = plan_logical_weights(&path, &DeclaresSixty, &ForcePacked).expect("plan");
        assert_eq!(forced.tensors[0].residency.mode, ResidencyMode::Packed);
        let Err(error) = read_logical_weights(&path, &forced, &Device::Cpu) else {
            panic!("a packed decode of a padded NVFP4 layer must refuse");
        };
        let error = error.to_string();
        assert!(error.contains("model.q.weight"), "{error}");
        assert!(error.contains("[256, 60]"), "names the logical: {error}");
        assert!(error.contains("[256, 64]"), "names the stored: {error}");
    }

    /// AC2 (Candle). The `sm_120` floor has one definition, and a layer the FP4 leg cannot run
    /// takes the dense fallback rather than a packed plan that would fail at the first forward.
    #[test]
    fn the_nvfp4_floor_has_one_definition_at_sm120_and_the_layout_gate_is_separate() {
        assert_eq!(crate::quant::NVFP4_COMPUTE_CAP_FLOOR, (12, 0));
        for (cap, expected) in [
            ((8, 9), false),
            ((9, 0), false),
            ((10, 0), false),
            ((11, 8), false),
            ((12, 0), true),
            ((12, 1), true),
            ((13, 0), true),
        ] {
            assert_eq!(
                crate::quant::compute_cap_meets_nvfp4_floor(cap),
                expected,
                "compute capability {cap:?}"
            );
        }
        // The layout gate reads the same K/N alignment the GEMM enforces.
        assert_eq!(crate::quant::NVFP4_K_ALIGN, 32);
        assert_eq!(crate::quant::NVFP4_N_ALIGN, 16);
        assert!(nvfp4_layout_is_native([256, 128]));
        assert!(
            !nvfp4_layout_is_native([256, 48]),
            "K = 48 is NOT_SUPPORTED"
        );
        assert!(
            !nvfp4_layout_is_native([24, 128]),
            "N must be a multiple of 16"
        );

        // And the policy needs all three: sm_120 hardware, an accepted layout, AND an unpadded
        // layer (`spec_padded` below covers the last one).
        let spec_of = |stored_shape: [usize; 2],
                       logical_shape: [usize; 2],
                       full_precision: bool| TensorCodecSpec::Nvfp4 {
            block_scale: "q.weight_scale".to_owned(),
            global_scale: "q.weight_scale_2".to_owned(),
            input_scale: None,
            stored_shape,
            logical_shape,
            logical_shape_declared: false,
            full_precision_matrix_mult: full_precision,
        };
        let spec = |stored_shape: [usize; 2], full_precision: bool| {
            spec_of(stored_shape, stored_shape, full_precision)
        };
        let native = CandleCodecResidency {
            fp8_e4m3_native: false,
            nvfp4_native: true,
            mxfp8_native: false,
        };
        let stored_bytes = [256_usize, 64];
        assert_eq!(
            native.residency(&NVFP4_CODEC, &spec([256, 128], false), &stored_bytes),
            ResidencyMode::Packed
        );
        assert_eq!(
            native.residency(&NVFP4_CODEC, &spec([256, 48], false), &stored_bytes),
            ResidencyMode::Dense,
            "an unaligned K falls back rather than planning a GEMM that would refuse"
        );
        assert_eq!(
            native.residency(&NVFP4_CODEC, &spec([256, 128], true), &stored_bytes),
            ResidencyMode::Dense,
            "`full_precision_matrix_mult` never runs packed"
        );
        // sc-20641 review: ComfyUI pads to 16, so `in_features = 60` stores a 32-aligned K = 64.
        // Alignment alone would plan Packed and hand `Nvfp4Linear` 4 columns of padding as real
        // contraction elements; the logical-equals-stored condition is what keeps it dense.
        assert!(nvfp4_layout_is_native([256, 64]), "the PADDED grid aligns");
        assert_eq!(
            native.residency(
                &NVFP4_CODEC,
                &spec_of([256, 64], [256, 60], false),
                &[256, 32]
            ),
            ResidencyMode::Dense,
            "a padded layer must not repack, however well the padded grid aligns"
        );
        assert_eq!(
            CandleCodecResidency::DENSE.residency(
                &NVFP4_CODEC,
                &spec([256, 128], false),
                &stored_bytes
            ),
            ResidencyMode::Dense,
            "below the floor every device takes the dense fallback"
        );
        // This machine's CPU (and Metal) are below both floors.
        assert_eq!(
            CandleCodecResidency::probe(&Device::Cpu),
            CandleCodecResidency::DENSE
        );
    }

    /// **sc-20651 major 3: an undeclared block-padded layer plans but must not materialize.**
    ///
    /// NVFP4 storage is 16-padded on both axes and the file records no true geometry, so with no
    /// adapter-declared logical shape the plan can only carry the *stored* grid forward. Decoding
    /// that grid hands the model the pad rows/columns as real weights — silent corruption, not a
    /// shape error, because every downstream check that trusts the plan agrees with it.
    ///
    /// Planning stays legal: a plan is also the pricing artifact admission reads, the padded grid is
    /// a conservative over-estimate, and the memory-strategy paths that compile a plan never
    /// materialize a tensor. Materialization is where it stops, and the refusal names the tensor.
    #[test]
    fn an_undeclared_block_padded_layer_plans_but_refuses_to_materialize() {
        let (rows, cols) = (32_usize, 64_usize);
        let dir = fixture_dir();
        let path = dir.path().join("nvfp4-undeclared.safetensors");
        let packed = vec![0x11_u8; rows * cols / 2];
        let block_scales =
            vec![0x38_u8; gen_core::nvfp4_scale_shape([rows, cols]).iter().product()];
        write_safetensors_with_metadata(
            &path,
            &[
                ("model.q.weight", "U8", &[rows, cols / 2], packed),
                (
                    "model.q.weight_scale",
                    "F8_E4M3",
                    &gen_core::nvfp4_scale_shape([rows, cols]),
                    block_scales,
                ),
                (
                    "model.q.weight_scale_2",
                    "F32",
                    &[],
                    1.0_f32.to_le_bytes().to_vec(),
                ),
            ],
            &[(
                "_quantization_metadata",
                r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4"}}}"#,
            )],
        );

        // Plans — and prices — with nothing declared.
        let undeclared = plan_logical_weights(&path, &StripModel, &CandleCodecResidency::DENSE)
            .expect("an undeclared padded checkpoint still PLANS: pricing is not materialization");
        assert_eq!(undeclared.codec_ids(), vec!["nvfp4-v1"]);
        assert_eq!(
            undeclared.tensors[0].undeclared_padded_storage(),
            Some([rows, cols]),
            "the plan records that its logical shape is only the stored grid"
        );

        // ...but the read refuses, naming the tensor.
        let error = match read_logical_weights(&path, &undeclared, &Device::Cpu) {
            Ok(_) => panic!("materializing an undeclared padded grid must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("model.q.weight"), "{error}");
        assert!(error.contains("block-padded"), "{error}");
        assert!(error.contains("declares no logical shape"), "{error}");

        // The SAME file, with the adapter declaring the layer's true geometry, reads.
        let declared = plan_logical_weights(
            &path,
            &StripModelDeclaring(vec![rows, cols]),
            &CandleCodecResidency::DENSE,
        )
        .expect("plan");
        assert_eq!(declared.tensors[0].undeclared_padded_storage(), None);
        read_logical_weights(&path, &declared, &Device::Cpu)
            .expect("a declared layer materializes exactly as before");
    }

    /// **sc-20651 blocker 2: the fp8 packed leg is rank-2 only.**
    ///
    /// `CublasLt::matmul_fp8`'s weight-side operand is a matrix, and this module's header has
    /// always said so ("a rank-2 E4M3 weight"). The policy did not enforce it, and the plan
    /// compiler deliberately applies **no** rank constraint to *undescribed* fp8 (ComfyUI's
    /// `weight_dtype=fp8_e4m3fn` cast covers biases and modulation vectors too), so a rank-1 bias
    /// in such a checkpoint planned `Packed` on an sm_89 CUDA device — into a container the fp8 leg
    /// cannot read — while MLX dense-decoded the same tensor correctly. That is two backends
    /// disagreeing about one file.
    #[test]
    fn fp8_packed_residency_is_rank_two_only() {
        let native = CandleCodecResidency {
            fp8_e4m3_native: true,
            nvfp4_native: false,
            mxfp8_native: false,
        };
        let scalar_fp8 = TensorCodecSpec::ScalarFp8 {
            scale: ScalarScaleSource::Unit,
            input_scale: None,
            full_precision_matrix_mult: false,
        };
        assert_eq!(
            native.residency(&FP8_E4M3_SCALAR_CODEC, &scalar_fp8, &[2048, 512]),
            ResidencyMode::Packed,
            "a rank-2 projection is the packed leg's operand"
        );
        for stored in [
            vec![16, 7],
            vec![7, 16],
            vec![0, 16],
            vec![2048_usize], // a bias / modulation vector — the real rank-1 case
            vec![],           // a scalar
            vec![4_usize, 2048, 512], // a rank-3 payload
        ] {
            assert_eq!(
                native.residency(&FP8_E4M3_SCALAR_CODEC, &scalar_fp8, &stored),
                ResidencyMode::Dense,
                "stored rank {} must take the dense fallback both backends share",
                stored.len()
            );
        }
    }

    /// **sc-20651 minor: a packed fp8 receipt counts a retained scale only when one exists.**
    ///
    /// The plan prices a retained `weight_scale` on its companion row, and
    /// [`ScalarScaleSource::Unit`] — the plain undescribed cast — has no such row, because there is
    /// no tensor in the file to price. `resident_bytes` added four bytes unconditionally, so the
    /// receipt exceeded the plan by exactly `4 x (undescribed fp8 layers)` on the one pairing the
    /// two exist to cross-check.
    #[test]
    fn packed_fp8_resident_bytes_counts_only_companions_that_exist() {
        let codes = Tensor::zeros((8, 8), DType::F8E4M3, &Device::Cpu).expect("codes");
        let unit = LogicalTensor::PackedScalarFp8 {
            codes: codes.clone(),
            weight_scale: 1.0,
            weight_scale_from_companion: false,
            input_scale: None,
        };
        let companion = LogicalTensor::PackedScalarFp8 {
            codes: codes.clone(),
            weight_scale: 0.5,
            weight_scale_from_companion: true,
            input_scale: None,
        };
        let both = LogicalTensor::PackedScalarFp8 {
            codes,
            weight_scale: 0.5,
            weight_scale_from_companion: true,
            input_scale: Some(0.25),
        };
        assert_eq!(
            unit.resident_bytes(),
            64,
            "the undescribed cast retains no scale ROW, so it prices none"
        );
        assert_eq!(companion.resident_bytes(), 64 + 4);
        assert_eq!(both.resident_bytes(), 64 + 4 + 4);
    }

    /// AC1 (Candle). Every NVFP4 layout defect fails closed, naming the layer.
    #[test]
    fn nvfp4_layout_defects_refuse_with_layer_specific_diagnostics() {
        let (rows, cols) = (32_usize, 64_usize);
        let values = nvfp4_fixture_values(rows, cols);
        let (packed, scales, global, _) = nvfp4_reference_quantize(&values, rows, cols);
        let scale_shape = [128_usize, 4];
        assert_eq!(scales.len(), scale_shape[0] * scale_shape[1]);

        let dir = fixture_dir();
        let case = |name: &str, tensors: &[(&str, &str, &[usize], Vec<u8>)]| -> String {
            let path = dir.path().join(format!("{name}.safetensors"));
            write_safetensors_with_metadata(
                &path,
                tensors,
                &[(
                    "_quantization_metadata",
                    r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4"}}}"#,
                )],
            );
            match plan_logical_weights(&path, &StripModel, &CandleCodecResidency::DENSE) {
                Ok(_) => panic!("{name}: must refuse"),
                Err(error) => error.to_string(),
            }
        };
        let weight = (
            "model.q.weight",
            "U8",
            &[rows, cols / 2][..],
            packed.clone(),
        );
        let block = (
            "model.q.weight_scale",
            "F8_E4M3",
            &scale_shape[..],
            scales.clone(),
        );
        let global_row = (
            "model.q.weight_scale_2",
            "F32",
            &[][..],
            global.to_le_bytes().to_vec(),
        );

        // A missing block scale, and a missing global scale — each named as the absent companion.
        let error = case("no-block-scale", &[weight.clone(), global_row.clone()]);
        assert!(
            error.contains("model.q.weight") && error.contains("weight_scale"),
            "{error}"
        );
        let error = case("no-global-scale", &[weight.clone(), block.clone()]);
        assert!(
            error.contains("model.q.weight_scale_2"),
            "a missing second scale level must name it: {error}"
        );
        // A mis-shaped block-scale surface (un-swizzled `[rows, blocks]` instead of the padded
        // swizzle) — the exact expected shape is named.
        let error = case(
            "unswizzled-scale",
            &[
                weight.clone(),
                (
                    "model.q.weight_scale",
                    "F8_E4M3",
                    &[rows, cols / 16],
                    scales[..rows * (cols / 16)].to_vec(),
                ),
                global_row.clone(),
            ],
        );
        assert!(
            error.contains("model.q.weight") && error.contains("[128, 4]"),
            "{error}"
        );
        // A global scale that is not a scalar F32.
        let error = case(
            "global-not-scalar",
            &[
                weight.clone(),
                block.clone(),
                ("model.q.weight_scale_2", "F32", &[4], vec![0_u8; 16]),
            ],
        );
        assert!(
            error.contains("model.q.weight_scale_2") && error.contains("scalar F32"),
            "{error}"
        );
        // Truncated nibbles: the byte matrix does not match its declared shape.
        let error = case(
            "truncated-nibbles",
            &[
                (
                    "model.q.weight",
                    "U8",
                    &[rows, cols / 2],
                    packed[..packed.len() - 8].to_vec(),
                ),
                block.clone(),
                global_row.clone(),
            ],
        );
        assert!(error.contains("model.q.weight"), "{error}");
        // Invalid padding: a logical column count that is not a multiple of 16.
        let error = case(
            "unpadded",
            &[
                ("model.q.weight", "U8", &[rows, 20], vec![0_u8; rows * 20]),
                (
                    "model.q.weight_scale",
                    "F8_E4M3",
                    &[128, 4],
                    vec![0_u8; 512],
                ),
                global_row.clone(),
            ],
        );
        assert!(
            error.contains("model.q.weight") && error.contains("multiple of 16"),
            "{error}"
        );
        // A shape mismatch between the weight and its block scales: a weight twice as wide still
        // has a valid 16-padded geometry, but needs 8 blocks per row, not 4.
        let error = case(
            "shape-mismatch",
            &[
                (
                    "model.q.weight",
                    "U8",
                    &[rows, cols],
                    vec![0_u8; rows * cols],
                ),
                block.clone(),
                global_row.clone(),
            ],
        );
        assert!(
            error.contains("model.q.weight") && error.contains("weight_scale"),
            "{error}"
        );
        // A `_quantization_metadata` layer name that matches nothing refuses rather than being
        // silently ignored.
        let path = dir.path().join("unmatched.safetensors");
        write_safetensors_with_metadata(
            &path,
            &[weight.clone(), block.clone(), global_row.clone()],
            &[(
                "_quantization_metadata",
                r#"{"format_version": "1.0", "layers": {"absent": {"format": "nvfp4"}}}"#,
            )],
        );
        let error = plan_logical_weights(&path, &StripModel, &CandleCodecResidency::DENSE)
            .expect_err("an unmatched declaration must refuse")
            .to_string();
        assert!(error.contains("absent"), "{error}");
    }

    #[test]
    fn descriptor_defects_and_source_drift_refuse_by_tensor() {
        let dir = fixture_dir();
        let path = dir.path().join("bad.safetensors");
        let descriptor = br#"{"format": "int4_awq"}"#;
        write_safetensors(
            &path,
            &[
                ("model.q.weight", "F8_E4M3", &[1, 2], vec![0x38, 0x40]),
                (
                    "model.q.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
            ],
        );
        let error = plan_logical_weights(&path, &StripModel, &CandleCodecResidency::DENSE)
            .expect_err("nvfp4 must refuse")
            .to_string();
        assert!(
            error.contains("\"model.q\"") && error.contains("nvfp4"),
            "{error}"
        );

        // Drift between planning and reading refuses.
        let drift = dir.path().join("drift.safetensors");
        write_safetensors(
            &drift,
            &[("model.w", "BF16", &[2], bf16_bytes(&[1.0, 2.0]))],
        );
        let plan = plan_dense(&drift);
        write_safetensors(
            &drift,
            &[
                ("model.w", "BF16", &[2], bf16_bytes(&[1.0, 2.0])),
                ("model.extra", "BF16", &[1], bf16_bytes(&[3.0])),
            ],
        );
        let error = match read_logical_weights(&drift, &plan, &Device::Cpu) {
            Ok(_) => panic!("drift must refuse"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("tensor set changed since planning"),
            "{error}"
        );
    }

    // ---- adapter-declared logical transforms (sc-21547) ---------------------------------------

    /// A mapping that declares the two fused layouts this seam exists for: a fused-QKV projection
    /// split into three logical weights, and an AdaLN modulation whose two halves are exchanged.
    /// The Klein shapes are fixtures only — nothing below knows what family it is reading.
    struct FusedLayouts {
        /// The physical tensor's declared logical shape, if any (NVFP4 needs one).
        source_logical_shape: Option<Vec<usize>>,
        /// Rows per fused-QKV part.
        part_rows: usize,
    }

    impl LogicalKeyMapping for FusedLayouts {
        fn mapping_id(&self) -> &'static str {
            "fused-layouts-test"
        }
        fn logical_key(&self, physical_key: &str) -> Option<String> {
            physical_key.strip_prefix("model.").map(str::to_owned)
        }
        fn logical_transform(
            &self,
            physical_key: &str,
        ) -> Option<gen_core::checkpoint_codec::LogicalTransformDeclaration> {
            use gen_core::checkpoint_codec::{LogicalTransformDeclaration, LogicalTransformOutput};
            let part = self.part_rows;
            let declaration = match physical_key {
                "model.qkv.weight" => LogicalTransformDeclaration::new(vec![
                    LogicalTransformOutput::row_slice("attn.q.weight", 0, part),
                    LogicalTransformOutput::row_slice("attn.k.weight", part, part),
                    LogicalTransformOutput::row_slice("attn.v.weight", 2 * part, part),
                ]),
                "model.norm.modulation" => {
                    LogicalTransformDeclaration::new(vec![LogicalTransformOutput::half_swap(
                        "norm.modulation",
                    )])
                }
                _ => return None,
            };
            Some(match &self.source_logical_shape {
                Some(shape) => declaration.with_source_logical_shape(shape.clone()),
                None => declaration,
            })
        }
    }

    /// Dense fused layouts: the reader publishes the compiled plan's logical keys and shapes, the
    /// row splits carry the source's own rows, the half swap exchanges them, and the receipt
    /// measured off the materialized tensors equals the plan's pricing.
    #[test]
    fn dense_row_splits_and_half_swaps_produce_the_compiled_logical_keys_and_shapes() {
        let dir = fixture_dir();
        let path = dir.path().join("fused-dense.safetensors");
        // Row `r` of the fused QKV matrix is `[10*r, 10*r + 1]`, so every row is identifiable.
        let qkv: Vec<f32> = (0..6)
            .flat_map(|row| [10.0 * row as f32, 10.0 * row as f32 + 1.0])
            .collect();
        // A 4-row modulation whose halves are `shift` (rows 0..2) and `scale` (rows 2..4).
        let modulation: Vec<f32> = (0..4).flat_map(|row| [100.0 + row as f32]).collect();
        write_safetensors(
            &path,
            &[
                (
                    "model.norm.modulation",
                    "BF16",
                    &[4, 1],
                    bf16_bytes(&modulation),
                ),
                ("model.qkv.weight", "BF16", &[6, 2], bf16_bytes(&qkv)),
            ],
        );

        let mapping = FusedLayouts {
            source_logical_shape: None,
            part_rows: 2,
        };
        let plan =
            plan_logical_weights(&path, &mapping, &CandleCodecResidency::DENSE).expect("plan");
        assert_eq!(
            plan.logical_keys().collect::<Vec<_>>(),
            [
                "attn.k.weight",
                "attn.q.weight",
                "attn.v.weight",
                "norm.modulation"
            ]
        );

        let weights = read_logical_weights(&path, &plan, &Device::Cpu).expect("read");
        // Produced keys and shapes equal the compiled plan's, entry by entry.
        for tensor in &plan.tensors {
            let LogicalTensor::Dense(materialized) = weights
                .tensors
                .get(&tensor.logical_key)
                .expect("logical tensor")
            else {
                panic!("{} is dense in this plan", tensor.logical_key);
            };
            assert_eq!(
                materialized.dims(),
                tensor.shape.as_slice(),
                "{}",
                tensor.logical_key
            );
        }
        assert_eq!(dense_f32(&weights, "attn.q.weight"), [0.0, 1.0, 10.0, 11.0]);
        assert_eq!(
            dense_f32(&weights, "attn.k.weight"),
            [20.0, 21.0, 30.0, 31.0]
        );
        assert_eq!(
            dense_f32(&weights, "attn.v.weight"),
            [40.0, 41.0, 50.0, 51.0]
        );
        // The halves are exchanged, and nothing else moves.
        assert_eq!(
            dense_f32(&weights, "norm.modulation"),
            [102.0, 103.0, 100.0, 101.0]
        );

        // Source bytes counted once for the transformed tensor, and the whole file accounted for.
        assert_eq!(weights.receipt.source_bytes, plan.source_bytes);
        assert_eq!(weights.receipt.source_bytes, (6 * 2 + 4) * 2);
        // Measured residency equals the plan's pricing, per codec row and in total.
        assert_eq!(weights.receipt.resident_bytes(), plan.resident_bytes());
        assert_eq!(plan.resident_bytes(), (6 * 2 + 4) * 2);
    }

    /// The packed half of the same mechanism: an NVFP4 fused-QKV matrix is split into three packed
    /// operands **without dequantizing anything**. Each output is still an [`Nvfp4Tensor`], holding
    /// its own share of the nibble payload and its own whole scale-factor atoms; dequantizing each
    /// afterwards reproduces exactly the rows of the golden reference it was cut from.
    #[test]
    fn a_packed_nvfp4_fused_qkv_splits_into_packed_operands_without_dequantizing() {
        let dir = fixture_dir();
        let path = dir.path().join("fused-nvfp4.safetensors");
        let (rows, cols, part) = (384_usize, 128_usize, 128_usize);
        let values = nvfp4_fixture_values(rows, cols);
        let (packed, scales, global, reference) = nvfp4_reference_quantize(&values, rows, cols);
        write_safetensors_with_metadata(
            &path,
            &[
                ("model.qkv.weight", "U8", &[rows, cols / 2], packed),
                ("model.qkv.weight_scale", "F8_E4M3", &[384, 8], scales),
                (
                    "model.qkv.weight_scale_2",
                    "F32",
                    &[],
                    global.to_le_bytes().to_vec(),
                ),
            ],
            &[(
                "_quantization_metadata",
                r#"{"format_version": "1.0", "layers": {"qkv": {"format": "nvfp4"}}}"#,
            )],
        );

        let native = CandleCodecResidency {
            fp8_e4m3_native: false,
            nvfp4_native: true,
            mxfp8_native: false,
        };
        let mapping = FusedLayouts {
            source_logical_shape: Some(vec![rows, cols]),
            part_rows: part,
        };
        let plan = plan_logical_weights(&path, &mapping, &native).expect("plan");
        assert_eq!(plan.tensor_count(), 3);
        for tensor in &plan.tensors {
            assert_eq!(tensor.residency.mode, ResidencyMode::Packed);
            assert_eq!(tensor.shape, vec![part, cols]);
            assert_eq!(tensor.source_shape(), [rows, cols]);
            assert_eq!(tensor.residency.resident_bytes, (part * cols / 2) as u64);
        }

        let weights = read_logical_weights(&path, &plan, &Device::Cpu).expect("read");
        for (index, key) in ["attn.q.weight", "attn.k.weight", "attn.v.weight"]
            .into_iter()
            .enumerate()
        {
            let LogicalTensor::PackedNvfp4 { tensor, .. } =
                weights.tensors.get(key).expect("logical tensor")
            else {
                panic!("{key} must stay packed: the split never dequantizes");
            };
            assert_eq!((tensor.rows, tensor.cols), (part, cols));
            assert_eq!(tensor.global_scale, global);
            let recovered = tensor.dequantize_to_vec();
            let expected = &reference[index * part * cols..(index + 1) * part * cols];
            for (element, want) in recovered.iter().zip(expected) {
                assert_eq!(element, want, "{key}");
            }
        }

        // Planned vs measured: the nibble payload partitions across the three operands, the
        // swizzled block scales partition with it, and each operand owns one copy of the F32
        // per-tensor scale.
        assert_eq!(
            plan.resident_bytes(),
            (rows * cols / 2) as u64 + 384 * 8 + 3 * 4
        );
        assert_eq!(weights.receipt.resident_bytes(), plan.resident_bytes());
        // One physical tensor, its source bytes counted once across three logical outputs.
        assert_eq!(weights.receipt.source_bytes, plan.source_bytes);
        assert_eq!(
            plan.tensors
                .iter()
                .map(|tensor| tensor.source_bytes)
                .sum::<u64>(),
            (rows * cols / 2) as u64
        );
        assert_eq!(weights.receipt.tensor_count, 3);
        let nvfp4 = weights
            .receipt
            .residency
            .iter()
            .find(|report| report.codec_id == NVFP4_CODEC.codec_id)
            .expect("one nvfp4 codec row");
        assert_eq!(nvfp4.tensor_count, 3);
        assert_eq!(nvfp4.source_bytes, plan.source_bytes);
    }

    /// The reader's incremental entry point resolves each logical output of a transformed tensor
    /// independently, and its receipt covers only what has materialized so far — the property a
    /// layer-by-layer provider depends on. Re-reading one output does not double-count it.
    #[test]
    fn a_transformed_tensor_materializes_one_logical_output_at_a_time() {
        let dir = fixture_dir();
        let path = dir.path().join("fused-incremental.safetensors");
        let qkv: Vec<f32> = (0..6).map(|row| row as f32).collect();
        write_safetensors(
            &path,
            &[("model.qkv.weight", "BF16", &[6, 1], bf16_bytes(&qkv))],
        );
        let mapping = FusedLayouts {
            source_logical_shape: None,
            part_rows: 2,
        };
        let plan =
            plan_logical_weights(&path, &mapping, &CandleCodecResidency::DENSE).expect("plan");
        let reader = LogicalWeightReader::open(&path, plan.clone(), &Device::Cpu).expect("open");
        assert_eq!(reader.receipt().tensor_count, 0);

        let LogicalTensor::Dense(k) = reader.read("attn.k.weight").expect("read k") else {
            panic!("dense");
        };
        assert_eq!(k.dims(), [2, 1]);
        assert_eq!(
            k.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            [2.0, 3.0]
        );
        let _ = reader.read("attn.k.weight").expect("re-read k");
        let receipt = reader.receipt();
        assert_eq!(
            receipt.tensor_count, 1,
            "a re-read overwrites, never accumulates"
        );
        assert_eq!(receipt.resident_bytes(), 2 * 2);

        for key in ["attn.q.weight", "attn.v.weight"] {
            let _ = reader.read(key).expect("read");
        }
        let receipt = reader.receipt();
        assert_eq!(receipt.tensor_count, 3);
        assert_eq!(receipt.resident_bytes(), plan.resident_bytes());
        assert_eq!(receipt.source_bytes, plan.source_bytes);
    }

    /// **A real fused load produces valid [`CheckpointWeightFacts`] (sc-21484 review).**
    ///
    /// The review's major finding: the receipt producer and `SourceCodecSummary::of` must count
    /// source bytes by the *same* rule, or a perfectly valid fused load hard-errors
    /// `SourceBytesExceedPlan` and `checkpoint_weight_facts()` becomes unusable the moment
    /// sc-21547's transforms are in play.
    ///
    /// Both now simply **sum** plan entries, which is correct because the compiler carries the
    /// physical tensor's `source_bytes` on exactly one output and zeroes its siblings. This drives
    /// that end to end on a real fused-QKV plan compiled by [`FusedLayouts`] — no hand-built
    /// receipt, no hand-built plan.
    ///
    /// # Mutation
    ///
    /// De-duplicate by physical key in either accounting (`receipt()` or `SourceCodecSummary::of`)
    /// — the shape an earlier draft of this story had. `plan.tensors` is sorted by logical key, so
    /// the first sighting of `model.qkv.weight` is `attn.k.weight`, whose `source_bytes` is ZERO;
    /// the deduping side loses the real bytes and the two accountings disagree. Dedup in `of()`
    /// alone reds this at `CodecSourceBytesExceedEntry`; dedup in `receipt()` alone reds the
    /// `receipt.source_bytes` assertion below.
    #[test]
    fn a_real_fused_plan_yields_valid_checkpoint_weight_facts() {
        let dir = fixture_dir();
        let path = dir.path().join("fused-facts.safetensors");
        let qkv: Vec<f32> = (0..6)
            .flat_map(|row| [10.0 * row as f32, 10.0 * row as f32 + 1.0])
            .collect();
        write_safetensors(
            &path,
            &[("model.qkv.weight", "BF16", &[6, 2], bf16_bytes(&qkv))],
        );
        let mapping = FusedLayouts {
            source_logical_shape: None,
            part_rows: 2,
        };
        let plan =
            plan_logical_weights(&path, &mapping, &CandleCodecResidency::DENSE).expect("plan");

        // The premise this test rests on: three logical outputs of ONE physical tensor, sorted so
        // that the byte-carrying output is NOT the first sighting of the fused key.
        assert_eq!(plan.tensors.len(), 3);
        assert!(plan
            .tensors
            .iter()
            .all(|tensor| tensor.physical_key == "model.qkv.weight"));
        assert_eq!(plan.tensors[0].logical_key, "attn.k.weight");
        assert_eq!(
            plan.tensors[0].source_bytes, 0,
            "the first entry of the fused key is a zero-carrying sibling"
        );
        assert_eq!(
            plan.tensors.iter().map(|t| t.source_bytes).sum::<u64>(),
            plan.source_bytes,
            "summing plan entries totals the file exactly once"
        );

        let reader = LogicalWeightReader::open_with_capability(
            &path,
            plan.clone(),
            &Device::Cpu,
            NativeExecutionCapability::dense_only(),
        )
        .expect("a fused plan names exactly the file's tensor set");
        for tensor in &plan.tensors {
            reader.read(&tensor.logical_key).expect("logical output");
        }

        let receipt = reader.receipt();
        assert_eq!(
            receipt.tensor_count, 3,
            "three LOGICAL outputs materialized"
        );
        assert_eq!(
            receipt.source_bytes, plan.source_bytes,
            "one PHYSICAL tensor's source bytes, totalled once"
        );

        // The whole point: this receipt validates against the plan it came from.
        let facts = reader
            .checkpoint_weight_facts()
            .expect("the fused load is valid");
        assert!(facts.is_complete());
        assert_eq!(
            facts
                .source()
                .entry(DENSE_BF16_CODEC.codec_id)
                .expect("the bf16 row")
                .source_bytes,
            plan.source_bytes,
            "both accountings of the fused tensor's source bytes agree"
        );
    }
}
