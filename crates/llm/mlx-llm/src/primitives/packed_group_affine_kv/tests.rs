use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn rejects_mlx_dimension_overflow_before_storage_allocation() {
    let oversized = usize::try_from(i32::MAX).unwrap() + 1;
    let error = PackedGroupAffineKvCache::new("overflow", 1, oversized, 1, 64, 32).unwrap_err();
    assert!(error.to_string().contains("exceeds MLX i32 range"));
}

#[derive(Debug)]
struct TestPackedKernel {
    calls: AtomicUsize,
    fail_on: Option<usize>,
    cancel_on_failure: bool,
}

impl RetainedPackedKernel for TestPackedKernel {
    fn cache_identity(&self) -> &str {
        "test-packed"
    }
    fn backend(&self) -> &str {
        "mlx-metal"
    }
    fn retained_host_bytes_estimate(&self) -> usize {
        0
    }
    fn dispatch(&self, args: &PackedAttentionArgs<'_>) -> Result<Array> {
        let call = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        if self.fail_on == Some(call) {
            if self.cancel_on_failure {
                return Err(Error::Canceled);
            }
            return Err(Error::Msg("injected packed dispatch fault".into()));
        }
        Ok(args.query.clone())
    }
    fn kernel_selection(&self, _args: &PackedAttentionArgs<'_>) -> Option<PackedKernelSelection> {
        Some(PackedKernelSelection {
            kernel: crate::primitives::PACKED_PER_ROW_KERNEL,
            selection: crate::primitives::PACKED_SELECTION_BELOW_MULTI_ROW,
            reason: "test reader",
            query_dtype: "float32",
        })
    }
}

fn hook_cache_geometry(
    layers: usize,
    batch: usize,
    kv_heads: usize,
    head_dimension: usize,
    fail_on: Option<usize>,
) -> (Box<dyn KvCache>, Arc<TestPackedKernel>) {
    let kernel = Arc::new(TestPackedKernel {
        calls: AtomicUsize::new(0),
        fail_on,
        cancel_on_failure: false,
    });
    let selection = select_decoder_cache_with_reader(
        PackedCacheRequest {
            enabled: true,
            backend: "mlx-metal".into(),
            identity: "test-packed".into(),
            layers,
            batch,
            kv_heads,
            head_dimension,
            group_size: PACKED_METAL_QUANT_GROUP_SIZE,
            bits: PackedCodeBits::Two,
            query_length: 1,
            has_mask: false,
        },
        CompiledKernelHandle::new(kernel.clone()),
    );
    (selection.into_cache(), kernel)
}

fn hook_cache(layers: usize, fail_on: Option<usize>) -> Box<dyn KvCache> {
    hook_cache_geometry(layers, 1, 1, 64, fail_on).0
}
fn data(step: usize, rows: usize, width: usize, bias: f32) -> Vec<f32> {
    (0..step * rows * width)
        .map(|i| bias + i as f32 * 0.25)
        .collect()
}

fn bhst_data(batch: usize, heads: usize, step: usize, width: usize, bias: f32) -> Vec<f32> {
    (0..batch * heads)
        .flat_map(|row| {
            (0..step).flat_map(move |token| {
                (0..width).map(move |channel| {
                    bias + (row * 10_000 + token * 100 + channel) as f32 * 0.03125
                })
            })
        })
        .collect()
}

fn pseudo_random_outliers(
    batch: usize,
    heads: usize,
    step: usize,
    width: usize,
    seed: u64,
) -> Vec<f32> {
    let mut state = seed;
    (0..batch * heads * step * width)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let base = ((state >> 40) as i32 - (1 << 23)) as f32 / (1 << 20) as f32;
            match index % 29 {
                0 => -10_000.0,
                1 => 10_000.0,
                2 => 0.0,
                _ => base,
            }
        })
        .collect()
}

fn token_range(
    source: &[f32],
    batch: usize,
    heads: usize,
    full_step: usize,
    width: usize,
    start: usize,
    len: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(batch * heads * len * width);
    for row in 0..batch * heads {
        for token in start..start + len {
            let offset = (row * full_step + token) * width;
            out.extend_from_slice(&source[offset..offset + width]);
        }
    }
    out
}

fn assert_same_rows(left: &PackedGroupAffineKvCache, right: &PackedGroupAffineKvCache) {
    assert_eq!(left.logical_len(), right.logical_len());
    for token in 0..left.logical_len() {
        for row in 0..left.rows() {
            assert_eq!(
                left.read_row(0, token, row).unwrap(),
                right.read_row(0, token, row).unwrap()
            );
        }
    }
}

fn assert_same_device_layer(
    actual: &Option<DevicePackedLayer>,
    expected: &Option<DevicePackedLayer>,
) {
    let (Some(actual), Some(expected)) = (actual, expected) else {
        assert_eq!(actual.is_some(), expected.is_some());
        return;
    };
    let counts = |device: &DevicePackedLayer| {
        (
            device.key_packed_tokens,
            device.value_packed_tokens,
            device.key_tail_rows,
            device.value_tail_rows,
        )
    };
    assert_eq!(counts(actual), counts(expected));
    // Only the live extents are state; block padding past them is not.
    let live = |device: &DevicePackedLayer| {
        let groups = device.key_packed_tokens / PACKED_METAL_QUANT_GROUP_SIZE;
        let tokens = device.value_packed_tokens;
        [
            live_rows(&device.key_codes, groups),
            live_rows(&device.key_scales, groups),
            live_rows(&device.key_zeros, groups),
            live_rows(&device.value_codes, tokens),
            live_rows(&device.value_scales, tokens),
            live_rows(&device.value_zeros, tokens),
            live_rows(&device.key_tail, device.key_tail_rows),
            live_rows(&device.value_tail, device.value_tail_rows),
        ]
        .map(|array| {
            let array = array.unwrap().as_dtype(Dtype::Float32).unwrap();
            if array.size() == 0 {
                return Vec::new();
            }
            array.eval().unwrap();
            array.as_slice::<f32>().to_vec()
        })
    };
    assert_eq!(live(actual), live(expected));
}
#[test]
fn packed_append_is_quantized_and_non_mirroring() {
    let mut c = PackedGroupAffineKvCache::new("model", 1, 1, 2, 7, 4).unwrap();
    let k = data(3, 2, 7, -2.0);
    c.append(0, &k, &k, 3).unwrap();
    assert_eq!(c.logical_len(), 3);
    assert!(c.allocated_bytes() > 0);
    assert!(c.no_dense_mirror());
    assert_eq!(c.representation().bits, 2);
}
#[test]
fn chunking_offsets_and_rollback() {
    let mut c = PackedGroupAffineKvCache::new("m", 1, 1, 1, 5, 4).unwrap();
    c.set_absolute_offset(19);
    let a = data(2, 1, 5, 0.0);
    c.append(0, &a, &a, 2).unwrap();
    let b = data(3, 1, 5, 9.0);
    c.append(0, &b, &b, 3).unwrap();
    assert_eq!(
        (c.logical_len(), c.allocated_len(), c.absolute_offset()),
        (5, 8, 19)
    );
    c.rollback(2).unwrap();
    assert_eq!(c.logical_len(), 2);
}
#[test]
fn snapshot_rejects_identity_and_round_trips() {
    let mut c = PackedGroupAffineKvCache::new("m", 1, 1, 1, 8, 4).unwrap();
    let x = data(1, 1, 8, 1.0);
    c.append(0, &x, &x, 1).unwrap();
    c.packed_mlx_arguments(0).unwrap();
    assert!(c.device_layers[0].is_some());
    let bytes = c.save().unwrap();
    let mut other = PackedGroupAffineKvCache::new("wrong", 1, 1, 1, 8, 4).unwrap();
    assert!(other.restore(&bytes).is_err());
    let mut restored = PackedGroupAffineKvCache::new("m", 1, 1, 1, 8, 4).unwrap();
    restored.restore(&bytes).unwrap();
    assert!(restored.device_layers[0].is_none());
    // Snapshot semantics include the versioned layout and logical/token capacity, but not the
    // allocator-dependent capacity of each backing Vec. Restore must report the physical
    // allocation it actually received instead of replaying a stale byte count from another
    // process/allocation history.
    let restored_allocated = restored.allocated_payload_bytes();
    assert_eq!(
        restored.representation().allocated_bytes,
        restored_allocated
    );
    assert!(restored_allocated >= restored.logical_stored_bytes());
    assert_eq!(
        RepresentationMetadata {
            allocated_bytes: 0,
            host_allocated_payload_bytes: 0,
            retained_device_packed_logical_bytes: 0,
            ..restored.representation()
        },
        RepresentationMetadata {
            allocated_bytes: 0,
            host_allocated_payload_bytes: 0,
            retained_device_packed_logical_bytes: 0,
            ..c.representation()
        }
    );
    assert_same_rows(&restored, &c);
    restored.packed_mlx_arguments(0).unwrap();
    assert!(restored.device_layers[0].is_some());
    restored.restore(&bytes).unwrap();
    assert!(restored.device_layers[0].is_none());
    assert_eq!(
        restored
            .dispatch_telemetry()
            .retained_device_packed_logical_bytes,
        0
    );
}

#[test]
fn qualified_group32_snapshot_restore_and_trim_preserve_evaluated_rows() {
    const BATCH: usize = 2;
    const HEADS: usize = 2;
    const WIDTH: usize = 64;
    const TOKENS: usize = 35;
    let keys = bhst_data(BATCH, HEADS, TOKENS, WIDTH, -11.0);
    let values = bhst_data(BATCH, HEADS, TOKENS, WIDTH, 13.0);
    let mut source = PackedGroupAffineKvCache::new(
        "group32-snapshot",
        1,
        BATCH,
        HEADS,
        WIDTH,
        PACKED_METAL_QUANT_GROUP_SIZE,
    )
    .unwrap();
    source.append(0, &keys, &values, TOKENS).unwrap();
    let expected = source.evaluated_dense_layer(0).unwrap();
    let snapshot = source.save().unwrap();

    let mut restored = PackedGroupAffineKvCache::new(
        "group32-snapshot",
        1,
        BATCH,
        HEADS,
        WIDTH,
        PACKED_METAL_QUANT_GROUP_SIZE,
    )
    .unwrap();
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.representation().bits, 2);
    assert_eq!(
        restored.representation().group_size,
        PACKED_METAL_QUANT_GROUP_SIZE
    );
    assert_eq!(restored.evaluated_dense_layer(0).unwrap(), expected);

    restored.trim(33).unwrap();
    assert_eq!(restored.logical_len(), 33);
    let trimmed = restored.evaluated_dense_layer(0).unwrap();
    let trimmed_snapshot = restored.save().unwrap();
    let mut second = PackedGroupAffineKvCache::new(
        "group32-snapshot",
        1,
        BATCH,
        HEADS,
        WIDTH,
        PACKED_METAL_QUANT_GROUP_SIZE,
    )
    .unwrap();
    second.restore(&trimmed_snapshot).unwrap();
    assert_eq!(second.evaluated_dense_layer(0).unwrap(), trimmed);
    assert!(second.device_layers.iter().all(Option::is_none));
}

#[test]
fn snapshot_rejects_exact_quantization_parameter_mismatch_without_mutation() {
    let mut source = PackedGroupAffineKvCache::new("m", 1, 1, 1, 8, 4).unwrap();
    let payload = data(3, 1, 8, -2.0);
    source.append(0, &payload, &payload, 3).unwrap();
    let snapshot = source.save().unwrap();
    let mut incompatible = PackedGroupAffineKvCache::new("m", 1, 1, 1, 8, 3).unwrap();
    let before = incompatible.representation();
    assert!(incompatible.restore(&snapshot).is_err());
    assert_eq!(incompatible.representation(), before);
}

#[test]
fn code_width_is_recorded_and_a_snapshot_of_another_width_is_refused_without_mutation() {
    assert_eq!(PackedCodeBits::from_bits(2).unwrap(), PackedCodeBits::Two);
    assert_eq!(PackedCodeBits::from_bits(4).unwrap(), PackedCodeBits::Four);
    for unsupported in [0, 1, 3, 16] {
        assert!(PackedCodeBits::from_bits(unsupported).is_err());
    }
    assert_eq!(
        (
            PackedCodeBits::Four.codes_per_byte(),
            PackedCodeBits::Four.max_code()
        ),
        (2, 15)
    );
    assert_eq!(
        (
            PackedCodeBits::Eight.codes_per_byte(),
            PackedCodeBits::Eight.max_code()
        ),
        (1, 255)
    );
    for (stored, other) in PackedCodeBits::ALL
        .into_iter()
        .flat_map(|stored| PackedCodeBits::ALL.map(|other| (stored, other)))
        .filter(|(stored, other)| stored != other)
    {
        let payload = data(3, 1, 8, -2.0);
        let mut source = PackedGroupAffineKvCache::with_bits("m", 1, 1, 1, 8, 4, stored).unwrap();
        source.append(0, &payload, &payload, 3).unwrap();
        assert_eq!(source.representation().bits, stored.bits());
        let snapshot = source.save().unwrap();
        let mut same = PackedGroupAffineKvCache::with_bits("m", 1, 1, 1, 8, 4, stored).unwrap();
        same.restore(&snapshot).unwrap();
        assert_same_rows(&source, &same);
        let mut incompatible =
            PackedGroupAffineKvCache::with_bits("m", 1, 1, 1, 8, 4, other).unwrap();
        let resident = data(2, 1, 8, 5.0);
        incompatible.append(0, &resident, &resident, 2).unwrap();
        let (before, rows) = (
            incompatible.representation(),
            incompatible.read_row(0, 1, 0).unwrap(),
        );
        let error = incompatible.restore(&snapshot).unwrap_err().to_string();
        assert!(error.contains("quantization or shape mismatch"), "{error}");
        assert_eq!(incompatible.representation(), before);
        assert_eq!(incompatible.read_row(0, 1, 0).unwrap(), rows);
    }
}

/// A reader built for another code width is refused at binding, and the decoder factory falls
/// back to dense before any packed mutation.
#[test]
fn a_reader_of_another_code_width_is_refused_before_mutation() {
    let two_bit_reader = CompiledKernelHandle::new(Arc::new(TestPackedKernel {
        calls: AtomicUsize::new(0),
        fail_on: None,
        cancel_on_failure: false,
    }));
    assert_eq!(two_bit_reader.code_bits(), PackedCodeBits::Two);
    let mut cache = PackedGroupAffineKvCache::with_bits(
        "test-packed",
        1,
        1,
        1,
        64,
        PACKED_METAL_QUANT_GROUP_SIZE,
        PackedCodeBits::Four,
    )
    .unwrap();
    let error = cache
        .bind_compiled_handle(two_bit_reader.clone())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("reads 2-bit codes but the cache stores 4-bit codes"),
        "{error}"
    );
    assert!(cache.compiled_handle().is_none());
    let selection = select_decoder_cache_with_reader(
        PackedCacheRequest {
            enabled: true,
            backend: "mlx-metal".into(),
            identity: "test-packed".into(),
            layers: 1,
            batch: 1,
            kv_heads: 1,
            head_dimension: 64,
            group_size: PACKED_METAL_QUANT_GROUP_SIZE,
            bits: PackedCodeBits::Four,
            query_length: 1,
            has_mask: false,
        },
        two_bit_reader,
    );
    let CacheRoute::DenseFallback { reason } = selection.route() else {
        panic!("a cross-width reader must not select the packed route");
    };
    assert!(
        reason.contains("packed reader rejected before mutation"),
        "{reason}"
    );
}

/// Every cross-width pairing is refused: binding a Metal reader to a cache of another width,
/// and dispatching a reader on another width's buffers (before anything is encoded).
#[cfg(target_os = "macos")]
#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn every_cross_width_reader_bind_and_dispatch_fails_closed() {
    use crate::primitives::packed_metal::{PackedMask, PackedMetalGpuFamily, PackedMetalKernel};
    for stored in PackedCodeBits::ALL {
        let mut cache = device_cache_bits(1, 1, 64, stored);
        let kv = bhsd(&pseudo_random_outliers(1, 1, 40, 64, 3), 1, 1, 40, 64);
        cache.append_device(0, &kv, &kv).unwrap();
        let query = bhsd(&[0.1; 64], 1, 1, 1, 64);
        for reader in PackedCodeBits::ALL
            .into_iter()
            .filter(|bits| *bits != stored)
        {
            let kernel = PackedMetalKernel::for_identity_family_and_bits(
                "test-packed",
                PackedMetalGpuFamily::Apple7OrNewer,
                reader,
            )
            .unwrap();
            let layer = cache.device_layers[0].as_ref().unwrap();
            let error = kernel
                .dispatch(&layer.args(&query, PackedMask::Causal))
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(&format!(
                    "reader for {}-bit codes cannot read {}-bit codes",
                    reader.bits(),
                    stored.bits()
                )),
                "{error}"
            );
            let error = cache
                .bind_compiled_handle(CompiledKernelHandle::new(Arc::new(kernel)))
                .unwrap_err()
                .to_string();
            assert!(error.contains("compiled handle reads"), "{error}");
            assert!(cache.compiled_handle().is_none());
        }
    }
}

/// Codes are computed against the STORED f16 zero and scale: on adversarial groups (a large
/// `|min|` with a tiny range, zero range, sub-normal-f16 ranges) every width's round trip stays
/// within half the stored scale, and the scale stays the group's range plus at most one f16
/// ulp of `min`, over the levels. Quantizing against the f32 statistics instead dequantizes
/// with the rounded zero, an error of up to half an f16 ulp of `min` (e.g. 0.25 at 1000).
#[test]
fn round_trip_error_is_within_half_the_stored_scale_on_adversarial_groups() {
    let (width, group, tokens) = (64, 32, 32);
    let cases: [(f32, f32); 7] = [
        (1000.3, 1e-2),
        (-777.7, 1e-3),
        (60_000.0, 1.0),
        (3.0, 0.0),
        (0.0, 1e-6),
        (-1e-4, 2e-4),
        (-2.5, 5.0),
    ];
    for bits in PackedCodeBits::ALL {
        for (min, range) in cases {
            let mut rng = 0x9e37_79b9_u32;
            let values = (0..tokens * width)
                .map(|index| {
                    rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    // Both K (per channel over tokens) and V (per token over channels) groups
                    // hit their exact min and max.
                    let fraction = match index % 7 {
                        0 => 0.0,
                        1 => 1.0,
                        _ => (rng >> 8) as f32 / (1u32 << 24) as f32,
                    };
                    min + range * fraction
                })
                .collect::<Vec<_>>();
            let mut cache =
                PackedGroupAffineKvCache::with_bits("m", 1, 1, 1, width, group, bits).unwrap();
            cache.append(0, &values, &values, tokens).unwrap();
            let storage = cache.layers[0].as_ref().unwrap();
            let stored_scale = storage
                .keys
                .scales
                .iter()
                .chain(&storage.values.scales)
                .map(|scale| scale.to_f32())
                .fold(0.0f32, f32::max);
            let mut worst = 0.0f32;
            for token in 0..tokens {
                let (keys, dequantized) = cache.read_row(0, token, 0).unwrap();
                for (channel, (key, value)) in keys.iter().zip(&dequantized).enumerate() {
                    let input = values[token * width + channel];
                    worst = worst.max((key - input).abs()).max((value - input).abs());
                }
            }
            let magnitude = min.abs() + range;
            let bound = 0.5 * stored_scale + 4.0 * f32::EPSILON * magnitude;
            eprintln!(
                    "{bits:?} min {min} range {range}: max round-trip error {worst:e} (stored scale {stored_scale:e})"
                );
            assert!(worst <= bound, "{bits:?} {min}/{range}: {worst} > {bound}");
            // The stored scale is the range plus at most one f16 ulp of `min`, over the
            // levels (floored at the smallest normal f16).
            let ulp = (f16::from_f32(min).to_f32() - f16_next_down(f16::from_f32(min)).to_f32())
                .abs()
                .max(f16::from_f32(min.abs()).to_f32() * f32::EPSILON);
            let ceiling = ((range + 2.0 * ulp) / bits.levels()).max(f16::MIN_POSITIVE.to_f32())
                * (1.0 + 2.0f32.powi(-9));
            assert!(
                stored_scale <= ceiling,
                "{bits:?} {min}/{range}: scale {stored_scale} > {ceiling}"
            );
        }
    }
}

/// Stored bytes per resident token and KV head, read from the live storage, are exactly the
/// representation's arithmetic: `D·b/8` code bytes for K and for V, plus two f16 per channel per
/// 32-token K group and two f16 per 32-channel V group.
#[test]
fn stored_bytes_per_token_are_exact_for_each_code_width() {
    let (width, group, heads, tokens) = (128, 32, 2, 4096 + 13);
    let keys = pseudo_random_outliers(1, heads, tokens, width, 0x0b17_5eed);
    let values = pseudo_random_outliers(1, heads, tokens, width, 0x5eed_0b17);
    for (bits, per_token) in [
        (PackedCodeBits::Two, 96),
        (PackedCodeBits::Four, 160),
        (PackedCodeBits::Eight, 288),
    ] {
        let mut cache =
            PackedGroupAffineKvCache::with_bits("m", 1, 1, heads, width, group, bits).unwrap();
        cache.append(0, &keys, &values, tokens).unwrap();
        let [key_codes, key_metadata, pending, value_codes, value_metadata] =
            cache.layer_component_bytes(0).unwrap();
        let (complete, tail) = (4096, 13);
        let code_bytes = width * usize::from(bits.bits()) / 8;
        assert_eq!(key_codes, heads * complete * code_bytes, "{bits:?}");
        assert_eq!(
            key_metadata,
            heads * (complete / group) * width * 4,
            "{bits:?}"
        );
        assert_eq!(pending, heads * tail * width * 4, "{bits:?}");
        assert_eq!(value_codes, heads * tokens * code_bytes, "{bits:?}");
        assert_eq!(
            value_metadata,
            heads * tokens * (width / group) * 4,
            "{bits:?}"
        );
        assert_eq!(
            cache.logical_stored_bytes(),
            key_codes + key_metadata + pending + value_codes + value_metadata
        );
        // Per quantized token and KV head (the dense residual excluded): K + V codes and metadata.
        let quantized = key_codes
            + key_metadata
            + value_codes / tokens * complete
            + value_metadata / tokens * complete;
        assert_eq!(quantized, per_token * complete * heads, "{bits:?}");
        #[cfg(target_os = "macos")]
        {
            // The device-resident decoder layout stores the same bytes per quantized token;
            // the incomplete group stays a dense residual of the input dtype (f16 here).
            let mut device = device_cache_bits(1, heads, width, bits);
            device
                .append_device(
                    0,
                    &bhsd(&keys, 1, heads, tokens, width)
                        .as_dtype(Dtype::Float16)
                        .unwrap(),
                    &bhsd(&values, 1, heads, tokens, width)
                        .as_dtype(Dtype::Float16)
                        .unwrap(),
                )
                .unwrap();
            let (codes, metadata) = device.retained_device_component_bytes();
            assert_eq!(codes, heads * complete * 2 * code_bytes, "{bits:?}");
            assert_eq!(
                metadata,
                heads * (complete * 32 + tail * width * 2 * 2),
                "{bits:?}"
            );
            assert_eq!(
                codes + metadata - heads * tail * width * 4,
                per_token * complete * heads
            );
            let layer = device.device_layers[0].as_ref().unwrap();
            let capacity = layer.capacity_tokens;
            assert_eq!(
                array_bytes(&layer.key_codes) + array_bytes(&layer.value_codes),
                (heads * capacity * 2 * code_bytes) as u64,
                "{bits:?}: physical code arrays"
            );
        }
        // bf16 dense K+V is 4·D = 512 bytes per token and head.
        eprintln!(
            "{bits:?}: {per_token} B/token/head vs dense bf16 512 ({:.2}% reduction)",
            100.0 * (1.0 - per_token as f64 / 512.0)
        );
    }
}
#[test]
fn fallback_is_explicit_and_cancel_is_safe() {
    let mut c = PackedGroupAffineKvCache::new("m", 1, 1, 1, 3, 2).unwrap();
    let event = c.dense_read_fallback("read", "unsupported mask");
    assert_eq!(event.reason, "unsupported mask");
    c.cancel();
    assert!(c.append(0, &[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 1).is_err());
}

#[test]
fn deterministic_extremes_zero_tail_and_non_aligned_width() {
    let mut c = PackedGroupAffineKvCache::new("m", 1, 1, 1, 11, 4).unwrap();
    let mut values = vec![0.0; 11];
    values[0] = -1000.0;
    values[4] = 1000.0;
    values[10] = 7.0;
    c.append(0, &values, &values, 1).unwrap();
    let (row, _) = c.read_row(0, 0, 0).unwrap();
    assert_eq!(row.len(), 11);
    assert!(row.iter().all(|value| value.is_finite()));
    assert_eq!(c.representation().logical_len, 1);
}

#[test]
fn corrupted_snapshot_version_and_trailing_bytes_reject_without_mutation() {
    let mut c = PackedGroupAffineKvCache::new("m", 1, 1, 1, 8, 4).unwrap();
    let x = data(2, 1, 8, -3.0);
    c.append(0, &x, &x, 2).unwrap();
    let before = c.representation();
    let mut version = c.save().unwrap();
    version[8] = 3;
    assert!(c.restore(&version).is_err());
    assert_eq!(c.representation(), before);
    let mut trailing = c.save().unwrap();
    trailing.push(0);
    assert!(c.restore(&trailing).is_err());
    assert_eq!(c.representation(), before);
}

/// sc-20688 review (findings 10 and round-2 5): a crafted snapshot whose header declares a
/// capacity the cache's doubling growth never produces is refused with `Err` (never a
/// capacity-overflow panic), leaving the cache untouched; a huge power-of-two capacity is
/// restored with only the reservation growth gives its logical length, not an overcommitted
/// one.
#[test]
fn a_crafted_snapshot_capacity_is_refused_not_reserved() {
    let mut c = PackedGroupAffineKvCache::new("m", 1, 1, 1, 8, 4).unwrap();
    let x = data(2, 1, 8, -3.0);
    c.append(0, &x, &x, 2).unwrap();
    let before = c.representation();
    let honest = c.save().unwrap();
    // magic, version, identity length and bytes, then group size, batch, KV heads and head
    // dimension before the capacity.
    let capacity_at = 8 + 4 + 4 + "m".len() + 4 * 8;
    assert_eq!(
        u64::from_le_bytes(honest[capacity_at..capacity_at + 8].try_into().unwrap()),
        c.capacity as u64
    );
    let craft = |capacity: u64| {
        let mut crafted = honest[..honest.len() - 8].to_vec();
        crafted[capacity_at..capacity_at + 8].copy_from_slice(&capacity.to_le_bytes());
        let sum = checksum(&crafted);
        crafted.extend(sum.to_le_bytes());
        crafted
    };
    for capacity in [u64::MAX, 3, (1 << 34) + 1] {
        let crafted = craft(capacity);
        let restored =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| c.restore(&crafted)));
        let error = restored
            .unwrap_or_else(|_| panic!("capacity {capacity} panicked"))
            .expect_err("a crafted capacity is refused");
        assert!(error.to_string().contains("capacity"), "{error}");
        assert_eq!(c.representation(), before);
    }
    // Huge powers of two parse, but reserve only what growth gives the two restored tokens.
    let mut reference = PackedGroupAffineKvCache::new("m", 1, 1, 1, 8, 4).unwrap();
    reference.restore(&honest).unwrap();
    for capacity in [1_u64 << 34, 1 << 62] {
        let mut restored = PackedGroupAffineKvCache::new("m", 1, 1, 1, 8, 4).unwrap();
        restored.restore(&craft(capacity)).unwrap();
        assert_eq!(restored.capacity, 2, "capacity {capacity}");
        assert_eq!(
            restored.representation(),
            reference.representation(),
            "capacity {capacity}: the honest snapshot's reservation"
        );
    }
    c.restore(&honest).unwrap();
    assert_eq!(
        c.save().unwrap(),
        honest,
        "the honest snapshot still restores"
    );
}

#[test]
fn atomic_layers_and_preflight_are_reachable() {
    let mut c =
        PackedGroupAffineKvCache::new("m", 2, 1, 1, 64, PACKED_METAL_QUANT_GROUP_SIZE).unwrap();
    let x = data(2, 1, 64, 0.0);
    let updates = [(&x[..], &x[..]), (&x[..x.len() - 1], &x[..x.len() - 1])];
    assert!(c.append_all_layers(&updates, 2).is_err());
    assert_eq!(c.logical_len(), 0);
    assert!(matches!(
        c.preflight("cpu", 1, false),
        crate::primitives::CacheRoute::DenseFallback { .. }
    ));
    assert!(matches!(
        c.preflight("mlx-metal", 1, false),
        crate::primitives::CacheRoute::DenseFallback { .. }
    ));
    let before_handle = c.process_visible_bytes_estimate();
    let handle = CompiledKernelHandle::new(Arc::new(OpaqueCompiledKernel::new(
        "m",
        "mlx-metal",
        64,
        Arc::new(()),
    )));
    c.bind_compiled_handle(handle).unwrap();
    assert_eq!(c.compiled_handle().unwrap().backend(), "mlx-metal");
    assert_eq!(c.process_visible_bytes_estimate(), before_handle + 64);
    assert!(matches!(
        c.preflight("mlx-metal", 1, false),
        crate::primitives::CacheRoute::ExperimentalPacked
    ));
    let mismatch = CompiledKernelHandle::new(Arc::new(OpaqueCompiledKernel::new(
        "other",
        "mlx-metal",
        0,
        Arc::new(()),
    )));
    assert!(c.bind_compiled_handle(mismatch).is_err());
}

#[test]
fn decoder_factory_is_opt_in_and_falls_back_before_packed_mutation() {
    let mut disabled = select_decoder_cache(PackedCacheRequest::disabled(2));
    assert!(matches!(
        disabled.route(),
        CacheRoute::DenseFallback { reason } if reason.contains("disabled")
    ));
    assert!(disabled.cache.as_any_mut().is::<ContiguousKvCache>());

    let mut unsupported = select_decoder_cache(PackedCacheRequest {
        enabled: true,
        backend: "mlx-metal".into(),
        identity: "m".into(),
        layers: 2,
        batch: 1,
        kv_heads: 1,
        head_dimension: 0,
        group_size: 32,
        bits: PackedCodeBits::Two,
        query_length: 1,
        has_mask: false,
    });
    assert!(unsupported.cache.as_any_mut().is::<ContiguousKvCache>());

    let mut staged = select_decoder_cache(PackedCacheRequest {
        enabled: true,
        backend: "mlx-metal".into(),
        identity: "m".into(),
        layers: 2,
        batch: 1,
        kv_heads: 2,
        head_dimension: 64,
        group_size: 32,
        bits: PackedCodeBits::Two,
        query_length: 1,
        has_mask: false,
    });
    assert!(matches!(staged.route(), CacheRoute::DenseFallback { .. }));
    let adapter = staged
        .cache
        .as_any_mut()
        .downcast_mut::<DenseFallbackPackedDecoderCache>()
        .expect("supported override stages storage through the decoder factory");
    assert_eq!(adapter.staged_representation().logical_len, 0);
    assert!(adapter.fallback_events().is_empty());
    assert!(matches!(
        adapter.preflight_packed(1, false),
        CacheRoute::DenseFallback { .. }
    ));
}

#[test]
fn decoder_factory_binds_reader_before_experimental_route() {
    let handle = CompiledKernelHandle::new(Arc::new(OpaqueCompiledKernel::new(
        "m",
        "mlx-metal",
        32,
        Arc::new(()),
    )));
    let selection = select_decoder_cache_with_reader(
        PackedCacheRequest {
            enabled: true,
            backend: "mlx-metal".into(),
            identity: "m".into(),
            layers: 1,
            batch: 1,
            kv_heads: 1,
            head_dimension: 64,
            group_size: 32,
            bits: PackedCodeBits::Two,
            query_length: 1,
            has_mask: false,
        },
        handle,
    );
    // An opaque handle is intentionally non-dispatchable, but factory binding and route
    // selection are still observable before any K/V mutation.
    assert_eq!(selection.route(), &CacheRoute::ExperimentalPacked);
    let mut cache = selection.into_cache();
    let packed = cache
        .as_any_mut()
        .downcast_mut::<DenseFallbackPackedDecoderCache>()
        .unwrap();
    assert_eq!(packed.staged_representation().logical_len, 0);
    assert!(packed
        .dispatch_packed(
            0,
            &Array::from_slice(&[0.0f32; 64], &[1, 1, 1, 64]),
            crate::primitives::packed_metal::PackedMask::Causal
        )
        .is_err());
    assert_eq!(packed.staged_representation().logical_len, 0);
}

#[test]
fn decoder_preflight_reports_exact_nonmutating_capability_reasons() {
    let mut cache = hook_cache(1, None);
    let packed = cache
        .as_any_mut()
        .downcast_mut::<DenseFallbackPackedDecoderCache>()
        .unwrap();
    assert_eq!(
        packed.preflight_packed(1, false),
        CacheRoute::ExperimentalPacked
    );
    assert!(matches!(
        packed.preflight_packed(0, false),
        CacheRoute::DenseFallback { reason }
            if reason == "packed attention requires a non-empty query"
    ));
    assert!(matches!(
        packed.preflight_packed(1, true),
        CacheRoute::DenseFallback { reason }
            if reason == "additive attention masks require dense fallback"
    ));
    assert_eq!(packed.offset(), 0);
    assert!(packed.fallback_events().is_empty());
}

#[test]
fn decoder_factory_rejects_mask_before_packed_mutation() {
    let handle = CompiledKernelHandle::new(Arc::new(OpaqueCompiledKernel::new(
        "m",
        "mlx-metal",
        1,
        Arc::new(()),
    )));
    let mut selection = select_decoder_cache_with_reader(
        PackedCacheRequest {
            enabled: true,
            backend: "mlx-metal".into(),
            identity: "m".into(),
            layers: 1,
            batch: 1,
            kv_heads: 1,
            head_dimension: 64,
            group_size: 32,
            bits: PackedCodeBits::Two,
            query_length: 1,
            has_mask: true,
        },
        handle,
    );
    assert!(matches!(
        selection.route(),
        CacheRoute::DenseFallback { .. }
    ));
    assert!(selection.cache.as_any_mut().is::<ContiguousKvCache>());
}

#[test]
fn retained_metal_reader_rejects_non_qualified_quant_group_before_mutation() {
    let handle = CompiledKernelHandle::new(Arc::new(TestPackedKernel {
        calls: AtomicUsize::new(0),
        fail_on: None,
        cancel_on_failure: false,
    }));
    let selection = select_decoder_cache_with_reader(
        PackedCacheRequest {
            enabled: true,
            backend: "mlx-metal".into(),
            identity: "test-packed".into(),
            layers: 1,
            batch: 1,
            kv_heads: 1,
            head_dimension: 64,
            group_size: 4,
            bits: PackedCodeBits::Two,
            query_length: 1,
            has_mask: false,
        },
        handle,
    );
    assert!(matches!(
        selection.route(),
        CacheRoute::DenseFallback { reason }
            if reason.contains("quantization group size must be 32")
    ));
    assert_eq!(selection.cache.offset(), 0);
}

#[test]
fn packed_hook_commits_only_after_all_layers_and_preserves_offset() {
    let mut cache = hook_cache(2, None);
    let packed = cache
        .as_any_mut()
        .downcast_mut::<DenseFallbackPackedDecoderCache>()
        .unwrap();
    let q = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
    let kv = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
    assert!(packed
        .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
        .unwrap()
        .is_some());
    assert_eq!(packed.offset(), 0);
    assert!(packed
        .try_packed_attention(1, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
        .unwrap()
        .is_some());
    assert_eq!(packed.offset(), 1);
    assert_eq!(packed.staged_representation().logical_len, 1);
    assert!(packed.staged_representation().allocated_bytes > 0);
    assert!(packed
        .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
        .unwrap()
        .is_some());
    assert_eq!(packed.offset(), 1);
    assert!(packed
        .try_packed_attention(1, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
        .unwrap()
        .is_some());
    assert_eq!(packed.offset(), 2);
}

#[test]
fn sc20676_cloned_retained_handle_reports_one_real_cold_dispatch_across_caches() {
    let kernel = Arc::new(TestPackedKernel {
        calls: AtomicUsize::new(0),
        fail_on: None,
        cancel_on_failure: false,
    });
    let handle = CompiledKernelHandle::new(kernel);
    let request = || PackedCacheRequest {
        enabled: true,
        backend: "mlx-metal".into(),
        identity: "test-packed".into(),
        layers: 1,
        batch: 1,
        kv_heads: 1,
        head_dimension: 64,
        group_size: PACKED_METAL_QUANT_GROUP_SIZE,
        bits: PackedCodeBits::Two,
        query_length: 1,
        has_mask: false,
    };
    let dispatch = |handle: CompiledKernelHandle| {
        let mut cache = select_decoder_cache_with_reader(request(), handle).into_cache();
        let packed = cache
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap();
        let query = Array::from_slice(&[1.0_f32; 64], &[1, 1, 1, 64]);
        let kv = Array::from_slice(&[2.0_f32; 64], &[1, 1, 1, 64]);
        assert!(packed
            .try_packed_attention(
                0,
                &query,
                &kv,
                &kv,
                PackedAttentionMask::Causal,
                0.125,
                false,
            )
            .unwrap()
            .is_some());
        packed.model_evidence()
    };

    let cold = dispatch(handle.clone());
    assert_eq!(cold.compile_jit_attempts, 1);
    assert_eq!(cold.cold_dispatches, 1);
    assert_eq!(cold.steady_dispatches, 0);
    let warm = dispatch(handle);
    assert_eq!(warm.compile_jit_attempts, 0);
    assert_eq!(warm.cold_dispatches, 0);
    assert_eq!(warm.steady_dispatches, 1);
    assert!(warm.kernel_warmed);
}

#[test]
fn packed_hook_fault_discards_pending_step_and_lifecycle_stays_consistent() {
    let mut cache = hook_cache(2, Some(2));
    let packed = cache
        .as_any_mut()
        .downcast_mut::<DenseFallbackPackedDecoderCache>()
        .unwrap();
    let q = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
    let kv = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
    assert!(packed
        .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
        .unwrap()
        .is_some());
    assert_eq!(packed.offset(), 0);
    assert!(packed
        .try_packed_attention(1, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
        .is_err());
    assert_eq!(packed.offset(), 0);
    packed.reset().unwrap();
    assert_eq!(packed.offset(), 0);
    assert!(packed.retain_sequences(&[0]).is_ok());
}

#[test]
fn later_layer_fault_restores_existing_device_residency_and_accepted_receipt_then_retries() {
    let (mut cache, kernel) = hook_cache_geometry(2, 1, 1, 64, Some(4));
    let packed = cache
        .as_any_mut()
        .downcast_mut::<DenseFallbackPackedDecoderCache>()
        .unwrap();
    let query = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
    let values = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
    for layer in 0..2 {
        assert!(packed
            .try_packed_attention(
                layer,
                &query,
                &values,
                &values,
                PackedAttentionMask::Causal,
                0.125,
                false,
            )
            .unwrap()
            .is_some());
    }
    assert_eq!(packed.offset(), 1);
    let prior_devices = packed.staged.device_layers.clone();
    let prior_accepted = packed.staged.accepted_dispatch_snapshot();
    let prior_physical = packed.staged.dispatch_telemetry();
    // Kernel paths are accepted-call evidence: one per layer so far, restored with the step.
    assert_eq!(prior_accepted.kernel_paths.len(), 1);
    assert_eq!(prior_accepted.kernel_paths[0].1, 2);

    assert!(packed
        .try_packed_attention(
            0,
            &query,
            &values,
            &values,
            PackedAttentionMask::Causal,
            0.125,
            false,
        )
        .unwrap()
        .is_some());
    assert_eq!(
        packed
            .packed_evidence()
            .unwrap()
            .retained_device_packed_logical_bytes,
        packed
            .staged
            .dispatch_telemetry()
            .retained_device_packed_logical_bytes,
        "a step without a group flush holds nothing beside the in-place store for rollback"
    );
    assert!(packed
        .try_packed_attention(
            1,
            &query,
            &values,
            &values,
            PackedAttentionMask::Causal,
            0.125,
            false,
        )
        .is_err());
    assert_eq!(packed.offset(), 1);
    assert!(packed.pending_step.is_none());
    for (actual, expected) in packed.staged.device_layers.iter().zip(prior_devices.iter()) {
        assert_same_device_layer(actual, expected);
    }
    assert_eq!(packed.staged.accepted_dispatch_snapshot(), prior_accepted);
    let failed = packed.staged.dispatch_telemetry();
    assert_eq!(
        failed.dispatch_attempts,
        prior_physical.dispatch_attempts + 2
    );
    assert_eq!(
        failed.failed_dispatches,
        prior_physical.failed_dispatches + 1
    );
    assert!(failed.kernel_warmed);
    assert!(failed.uploaded_packed_bytes > prior_physical.uploaded_packed_bytes);
    assert!(
        failed.peak_packed_transient_logical_bytes
            >= prior_physical.peak_packed_transient_logical_bytes
    );
    assert_eq!(
        failed.accepted_uploaded_packed_bytes,
        prior_physical.accepted_uploaded_packed_bytes
    );

    for layer in 0..2 {
        assert!(packed
            .try_packed_attention(
                layer,
                &query,
                &values,
                &values,
                PackedAttentionMask::Causal,
                0.125,
                false,
            )
            .unwrap()
            .is_some());
    }
    assert_eq!(kernel.calls.load(Ordering::Relaxed), 6);
    assert_eq!(packed.offset(), 2);
    assert_eq!(packed.staged.direct_dispatches(), 4);
    let retried = packed.staged.dispatch_telemetry();
    assert_eq!(
        retried.dispatch_attempts,
        prior_physical.dispatch_attempts + 4
    );
    assert_eq!(
        retried.failed_dispatches,
        prior_physical.failed_dispatches + 1
    );
    assert_eq!(retried.cold_dispatches, prior_physical.cold_dispatches);
    assert_eq!(
        retried.steady_dispatches,
        prior_physical.steady_dispatches + 2
    );
    assert!(retried.accepted_uploaded_packed_bytes > prior_physical.accepted_uploaded_packed_bytes);
    let evidence = packed.packed_evidence().unwrap();
    assert_eq!(evidence.dispatch_attempts, retried.dispatch_attempts);
    assert_eq!(evidence.failed_dispatches, 1);
    assert_eq!(evidence.accepted_direct_calls, 4);
    assert_eq!(
        evidence.accepted_uploaded_packed_bytes,
        retried.accepted_uploaded_packed_bytes
    );
}

#[test]
fn typed_first_dispatch_cancellation_rolls_back_and_propagates_then_retry_succeeds() {
    let kernel = Arc::new(TestPackedKernel {
        calls: AtomicUsize::new(0),
        fail_on: Some(1),
        cancel_on_failure: true,
    });
    let mut selection = select_decoder_cache_with_reader(
        PackedCacheRequest {
            enabled: true,
            backend: "mlx-metal".into(),
            identity: "test-packed".into(),
            layers: 1,
            batch: 1,
            kv_heads: 1,
            head_dimension: 64,
            group_size: PACKED_METAL_QUANT_GROUP_SIZE,
            bits: PackedCodeBits::Two,
            query_length: 1,
            has_mask: false,
        },
        CompiledKernelHandle::new(kernel.clone()),
    );
    let packed = selection
        .cache
        .as_any_mut()
        .downcast_mut::<DenseFallbackPackedDecoderCache>()
        .unwrap();
    let query = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
    let values = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
    let error = packed
        .try_packed_attention(
            0,
            &query,
            &values,
            &values,
            PackedAttentionMask::Causal,
            0.125,
            false,
        )
        .unwrap_err();
    assert!(matches!(error, Error::Canceled));
    assert_eq!(packed.offset(), 0);
    assert!(packed.pending_step.is_none());
    assert!(packed.staged.device_layers[0].is_none());
    assert!(packed.staged.compiled_handle().is_some());
    assert!(packed.fallback_events().is_empty());
    let cancelled = packed.staged.dispatch_telemetry();
    assert_eq!(cancelled.dispatch_attempts, 1);
    assert_eq!(cancelled.failed_dispatches, 1);
    assert_eq!(cancelled.cold_dispatches, 0);
    assert_eq!(packed.staged.direct_dispatches(), 0);
    assert!(cancelled.uploaded_packed_bytes > 0);
    assert_eq!(cancelled.accepted_uploaded_packed_bytes, 0);

    assert!(packed
        .try_packed_attention(
            0,
            &query,
            &values,
            &values,
            PackedAttentionMask::Causal,
            0.125,
            false,
        )
        .unwrap()
        .is_some());
    assert_eq!(kernel.calls.load(Ordering::Relaxed), 2);
    assert_eq!(packed.offset(), 1);
    assert_eq!(packed.staged.direct_dispatches(), 1);
    let retried = packed.staged.dispatch_telemetry();
    assert_eq!(retried.dispatch_attempts, 2);
    assert_eq!(retried.failed_dispatches, 1);
    assert_eq!(retried.cold_dispatches, 1);
    assert!(retried.kernel_warmed);
    assert!(retried.uploaded_packed_bytes > retried.accepted_uploaded_packed_bytes);
    assert!(retried.accepted_uploaded_packed_bytes > 0);
}

#[test]
fn retained_dispatch_fault_keeps_host_append_private_device_and_truthful_retry_evidence() {
    let kernel = Arc::new(TestPackedKernel {
        calls: AtomicUsize::new(0),
        fail_on: Some(1),
        cancel_on_failure: false,
    });
    let mut cache = PackedGroupAffineKvCache::new("test-packed", 1, 1, 1, 64, 32).unwrap();
    cache
        .bind_compiled_handle(CompiledKernelHandle::new(kernel.clone()))
        .unwrap();
    let values = vec![2.0f32; 64];
    cache.append(0, &values, &values, 1).unwrap();
    let query = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
    assert!(cache
        .dispatch_packed(
            0,
            &query,
            crate::primitives::packed_metal::PackedMask::Causal,
        )
        .is_err());
    assert!(cache.device_layers[0].is_none());
    assert_eq!(cache.logical_len(), 1, "host append remains caller-owned");
    assert!(cache.layers[0].is_some());
    assert_eq!(cache.direct_dispatches(), 0);
    let failed = cache.dispatch_telemetry();
    assert_eq!(failed.dispatch_attempts, 1);
    assert_eq!(failed.failed_dispatches, 1);
    assert_eq!(failed.compile_jit_attempts, 1);
    assert!(!failed.kernel_warmed);
    assert!(failed.uploaded_packed_bytes > 0);
    assert_eq!(failed.accepted_uploaded_packed_bytes, 0);
    assert_eq!(failed.retained_device_packed_logical_bytes, 0);

    cache
        .dispatch_packed(
            0,
            &query,
            crate::primitives::packed_metal::PackedMask::Causal,
        )
        .unwrap();
    assert_eq!(kernel.calls.load(Ordering::Relaxed), 2);
    assert!(cache.device_layers[0].is_some());
    assert_eq!(cache.logical_len(), 1);
    assert_eq!(cache.direct_dispatches(), 1);
    let retried = cache.dispatch_telemetry();
    assert_eq!(retried.dispatch_attempts, 2);
    assert_eq!(retried.failed_dispatches, 1);
    assert_eq!(retried.compile_jit_attempts, 2);
    assert!(retried.kernel_warmed);
    assert!(retried.uploaded_packed_bytes > retried.accepted_uploaded_packed_bytes);
    assert!(retried.accepted_uploaded_packed_bytes > 0);
    assert!(retried.retained_device_packed_logical_bytes > 0);
}

#[test]
fn host_mirror_extents_advance_only_after_a_successful_dispatch() {
    let kernel = Arc::new(TestPackedKernel {
        calls: AtomicUsize::new(0),
        fail_on: Some(2),
        cancel_on_failure: false,
    });
    let mut cache =
        PackedGroupAffineKvCache::new("test-packed", 1, 2, 2, 64, PACKED_METAL_QUANT_GROUP_SIZE)
            .unwrap();
    cache
        .bind_compiled_handle(CompiledKernelHandle::new(kernel))
        .unwrap();
    let query = Array::from_slice(&vec![0.5f32; 2 * 2 * 64], &[2, 2, 1, 64]);
    let first = bhst_data(2, 2, 33, 64, -1.5);
    cache.append(0, &first, &first, 33).unwrap();
    cache
        .dispatch_packed(0, &query, crate::primitives::packed_metal::PackedMask::None)
        .unwrap();
    let accepted = cache.device_layers[0].clone();
    let accepted_telemetry = cache.dispatch_telemetry();
    let extent = |cache: &PackedGroupAffineKvCache| {
        let device = cache.device_layers[0].as_ref().unwrap();
        (
            device.key_packed_tokens,
            device.value_packed_tokens,
            device.key_tail_rows,
        )
    };
    assert_eq!(extent(&cache), (32, 33, 1));

    let second = bhst_data(2, 2, 40, 64, 3.0);
    cache.append(0, &second, &second, 40).unwrap();
    assert!(cache
        .dispatch_packed(0, &query, crate::primitives::packed_metal::PackedMask::None)
        .is_err());
    assert_same_device_layer(&cache.device_layers[0], &accepted);
    let failed = cache.dispatch_telemetry();
    assert!(failed.uploaded_packed_bytes > accepted_telemetry.uploaded_packed_bytes);
    assert_eq!(
        failed.accepted_uploaded_packed_bytes,
        accepted_telemetry.accepted_uploaded_packed_bytes
    );
    assert_eq!(
        failed.retained_device_packed_logical_bytes,
        accepted_telemetry.retained_device_packed_logical_bytes
    );

    cache
        .dispatch_packed(0, &query, crate::primitives::packed_metal::PackedMask::None)
        .unwrap();
    assert_eq!(extent(&cache), (64, 73, 9));
}

#[test]
fn first_packed_dispatch_fault_disables_route_and_falls_back_before_mutation() {
    let mut cache = hook_cache(2, Some(1));
    let packed = cache
        .as_any_mut()
        .downcast_mut::<DenseFallbackPackedDecoderCache>()
        .unwrap();
    let q = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
    let kv = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
    assert!(packed
        .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
        .unwrap()
        .is_none());
    assert_eq!(packed.offset(), 0);
    assert!(packed.staged.compiled_handle().is_none());
    assert!(packed
        .fallback_events()
        .iter()
        .any(|event| event.operation == "dispatch-fault"));
    assert!(packed
        .try_packed_attention(1, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
        .unwrap()
        .is_none());
    packed.update(0, &kv, &kv).unwrap();
    packed.update(1, &kv, &kv).unwrap();
    assert_eq!(packed.dense.offset(), 1);
}

#[test]
fn pending_packed_step_rejects_retain_and_is_observably_discarded_by_truncate_and_reset() {
    let q = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
    let kv = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
    let mut cache = hook_cache(2, None);
    let packed = cache
        .as_any_mut()
        .downcast_mut::<DenseFallbackPackedDecoderCache>()
        .unwrap();
    packed
        .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
        .unwrap();
    assert!(packed.retain_sequences(&[0]).is_err());
    packed.truncate(0).unwrap();
    assert_eq!(packed.offset(), 0);
    assert!(packed
        .staged
        .fallback_events()
        .iter()
        .any(|event| event.operation == "truncate"));
    packed
        .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
        .unwrap();
    packed.reset().unwrap();
    assert_eq!(packed.offset(), 0);
    assert!(packed
        .staged
        .fallback_events()
        .iter()
        .any(|event| event.operation == "reset"));
}

#[test]
fn committed_packed_history_reconstructs_exact_evaluated_dense_cache_before_batch_compaction() {
    const BATCH: usize = 3;
    const KV_HEADS: usize = 2;
    const QUERY_HEADS: usize = 4;
    const STEP: usize = 2;
    const WIDTH: usize = 64;
    let (mut cache, _) = hook_cache_geometry(1, BATCH, KV_HEADS, WIDTH, None);
    let packed = cache
        .as_any_mut()
        .downcast_mut::<DenseFallbackPackedDecoderCache>()
        .unwrap();
    let query_values = bhst_data(BATCH, QUERY_HEADS, STEP, WIDTH, -0.5);
    let key_values = bhst_data(BATCH, KV_HEADS, STEP, WIDTH, -3.0);
    let value_values = bhst_data(BATCH, KV_HEADS, STEP, WIDTH, 7.0);
    let query = Array::from_slice(
        &query_values,
        &[BATCH as i32, QUERY_HEADS as i32, STEP as i32, WIDTH as i32],
    )
    .as_dtype(mlx_rs::Dtype::Bfloat16)
    .unwrap();
    let keys = Array::from_slice(
        &key_values,
        &[BATCH as i32, KV_HEADS as i32, STEP as i32, WIDTH as i32],
    )
    .as_dtype(mlx_rs::Dtype::Bfloat16)
    .unwrap();
    let values = Array::from_slice(
        &value_values,
        &[BATCH as i32, KV_HEADS as i32, STEP as i32, WIDTH as i32],
    )
    .as_dtype(mlx_rs::Dtype::Bfloat16)
    .unwrap();
    assert!(packed
        .try_packed_attention(
            0,
            &query,
            &keys,
            &values,
            PackedAttentionMask::Causal,
            (WIDTH as f32).powf(-0.5),
            false,
        )
        .unwrap()
        .is_some());
    assert_eq!(packed.offset(), STEP as i32);

    let keep = [2, 0];
    let (_, all_evaluated_keys, all_evaluated_values) =
        packed.staged.evaluated_dense_layer(0).unwrap();
    let mut expected_keys = Vec::new();
    let mut expected_values = Vec::new();
    for &batch in &keep {
        for head in 0..KV_HEADS {
            let row = batch * KV_HEADS + head;
            for token in 0..STEP {
                let start = (row * STEP + token) * WIDTH;
                expected_keys.extend_from_slice(&all_evaluated_keys[start..start + WIDTH]);
                expected_values.extend_from_slice(&all_evaluated_values[start..start + WIDTH]);
            }
        }
    }
    packed
        .retain_sequences(&keep.map(|value| value as i32))
        .unwrap();
    assert_eq!(packed.batch_size(), keep.len() as i32);
    let evidence = packed.packed_evidence().unwrap();
    assert_eq!(evidence.accepted_direct_calls, 1);
    assert_eq!(evidence.full_cache_dequantizations, 1);
    assert!(evidence.dense_active);
    assert_eq!(evidence.bits, 2);
    assert_eq!(evidence.quantization_group_size, 32);
    assert!(evidence.fallback_reasons.iter().any(|(operation, reason)| {
        operation == "retain_sequences"
            && reason == "batch compaction requires exact dense reconstruction"
    }));

    let expected_keys = Array::from_slice(
        &expected_keys,
        &[
            keep.len() as i32,
            KV_HEADS as i32,
            STEP as i32,
            WIDTH as i32,
        ],
    )
    .as_dtype(mlx_rs::Dtype::Bfloat16)
    .unwrap()
    .as_dtype(mlx_rs::Dtype::Float32)
    .unwrap();
    let expected_values = Array::from_slice(
        &expected_values,
        &[
            keep.len() as i32,
            KV_HEADS as i32,
            STEP as i32,
            WIDTH as i32,
        ],
    )
    .as_dtype(mlx_rs::Dtype::Bfloat16)
    .unwrap()
    .as_dtype(mlx_rs::Dtype::Float32)
    .unwrap();
    expected_keys.eval().unwrap();
    expected_values.eval().unwrap();

    let next = Array::from_slice(
        &vec![0.0f32; keep.len() * KV_HEADS * WIDTH],
        &[keep.len() as i32, KV_HEADS as i32, 1, WIDTH as i32],
    )
    .as_dtype(mlx_rs::Dtype::Bfloat16)
    .unwrap();
    let (actual_keys, actual_values) = packed.update(0, &next, &next).unwrap();
    let actual_keys = actual_keys.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    let actual_values = actual_values.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    actual_keys.eval().unwrap();
    actual_values.eval().unwrap();
    assert_eq!(
        actual_keys.shape(),
        &mlx_shape([keep.len(), KV_HEADS, STEP + 1, WIDTH]).unwrap()
    );
    for (actual, expected) in [
        (
            actual_keys.as_slice::<f32>(),
            expected_keys.as_slice::<f32>(),
        ),
        (
            actual_values.as_slice::<f32>(),
            expected_values.as_slice::<f32>(),
        ),
    ] {
        for row in 0..keep.len() * KV_HEADS {
            for token in 0..STEP {
                let actual_start = (row * (STEP + 1) + token) * WIDTH;
                let expected_start = (row * STEP + token) * WIDTH;
                assert_eq!(
                    &actual[actual_start..actual_start + WIDTH],
                    &expected[expected_start..expected_start + WIDTH],
                );
            }
        }
    }
    assert_eq!(packed.offset(), (STEP + 1) as i32);
}

#[test]
fn interleaved_caches_share_a_reader_without_sharing_transaction_or_receipt_state() {
    let shared = Arc::new(TestPackedKernel {
        calls: AtomicUsize::new(0),
        fail_on: None,
        cancel_on_failure: false,
    });
    let make_cache = || {
        select_decoder_cache_with_reader(
            PackedCacheRequest {
                enabled: true,
                backend: "mlx-metal".into(),
                identity: "test-packed".into(),
                layers: 1,
                batch: 1,
                kv_heads: 1,
                head_dimension: 64,
                group_size: PACKED_METAL_QUANT_GROUP_SIZE,
                bits: PackedCodeBits::Two,
                query_length: 1,
                has_mask: false,
            },
            CompiledKernelHandle::new(shared.clone()),
        )
        .into_cache()
    };
    let mut left = make_cache();
    let mut right = make_cache();
    let q = Array::from_slice(&vec![0.25f32; 64], &[1, 1, 1, 64]);
    let left_kv = Array::from_slice(&vec![-1.0f32; 64], &[1, 1, 1, 64]);
    let right_kv = Array::from_slice(&vec![3.0f32; 64], &[1, 1, 1, 64]);
    for (cache, kv) in [(&mut left, &left_kv), (&mut right, &right_kv)] {
        assert!(cache
            .try_packed_attention(0, &q, kv, kv, PackedAttentionMask::Causal, 0.125, false,)
            .unwrap()
            .is_some());
    }
    assert_eq!(shared.calls.load(Ordering::Relaxed), 2);
    assert_eq!(left.offset(), 1);
    assert_eq!(right.offset(), 1);
    assert_eq!(left.packed_evidence().unwrap().accepted_direct_calls, 1);
    assert_eq!(right.packed_evidence().unwrap().accepted_direct_calls, 1);
    left.truncate(0).unwrap();
    assert_eq!(left.offset(), 0);
    assert_eq!(right.offset(), 1);
}

#[test]
fn hard_pre_dispatch_errors_roll_back_the_pending_whole_step() {
    let q = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
    let kv = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
    let q_two = Array::from_slice(&vec![1.0f32; 128], &[1, 1, 2, 64]);
    let kv_two = Array::from_slice(&vec![2.0f32; 128], &[1, 1, 2, 64]);
    let assert_one_hard_error = |layer: usize, query: &Array, key: &Array, value: &Array| {
        let mut cache = hook_cache(2, None);
        let packed = cache
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap();
        assert!(packed
            .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false,)
            .unwrap()
            .is_some());
        assert!(packed.pending_step.is_some());
        assert_eq!(packed.offset(), 0);
        assert!(packed
            .try_packed_attention(
                layer,
                query,
                key,
                value,
                PackedAttentionMask::Causal,
                0.125,
                false,
            )
            .is_err());
        assert!(packed.pending_step.is_none());
        assert_eq!(packed.offset(), 0);
        assert_eq!(packed.staged.layer_tokens(0), 0);
    };
    assert_one_hard_error(0, &q, &kv, &kv);
    assert_one_hard_error(1, &q_two, &kv_two, &kv_two);
}

#[test]
fn late_capability_decline_preserves_partial_step_in_dense_cache() {
    let q = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
    let kv = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
    let mut cache = hook_cache(2, None);
    let packed = cache
        .as_any_mut()
        .downcast_mut::<DenseFallbackPackedDecoderCache>()
        .unwrap();

    for layer in 0..2 {
        assert!(packed
            .try_packed_attention(
                layer,
                &q,
                &kv,
                &kv,
                PackedAttentionMask::Causal,
                0.125,
                false,
            )
            .unwrap()
            .is_some());
    }
    assert_eq!(packed.offset(), 1);

    assert!(packed
        .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false,)
        .unwrap()
        .is_some());
    assert!(packed.pending_step.is_some());
    assert!(packed
        .try_packed_attention(1, &q, &kv, &kv, PackedAttentionMask::Additive, 0.125, false,)
        .unwrap()
        .is_none());
    assert!(packed.dense_active);
    assert!(packed.pending_step.is_none());
    assert_eq!(packed.staged.logical_len(), 0);
    assert_eq!(packed.dense.peek(0).unwrap().unwrap().0.shape()[2], 2);
    assert_eq!(packed.dense.peek(1).unwrap().unwrap().0.shape()[2], 1);

    packed.update(1, &kv, &kv).unwrap();
    assert_eq!(packed.offset(), 2);
    assert_eq!(packed.dense.peek(1).unwrap().unwrap().0.shape()[2], 2);
    assert_eq!(packed.staged.full_cache_dequantizations(), 1);
}

#[test]
fn every_late_capability_branch_transitions_the_partial_step_to_dense() {
    let q = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
    let unsupported_q = Array::from_slice(&[1u8; 64], &[1, 1, 1, 64]);
    let kv = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
    for (query, mask, scale, retained_for_sharing) in [
        (&q, PackedAttentionMask::Causal, 0.125, true),
        (&q, PackedAttentionMask::Causal, 1.0, false),
        (&q, PackedAttentionMask::SlidingWindow(0), 0.125, false),
        (&q, PackedAttentionMask::Additive, 0.125, false),
        (&unsupported_q, PackedAttentionMask::Causal, 0.125, false),
    ] {
        let mut cache = hook_cache(2, None);
        let packed = cache
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap();
        assert!(packed
            .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false,)
            .unwrap()
            .is_some());
        assert!(packed
            .try_packed_attention(1, query, &kv, &kv, mask, scale, retained_for_sharing,)
            .unwrap()
            .is_none());
        assert!(packed.dense_active);
        assert!(packed.pending_step.is_none());
        assert_eq!(packed.dense.peek(0).unwrap().unwrap().0.shape()[2], 1);
        assert!(packed.dense.peek(1).unwrap().is_none());
        packed.update(1, &kv, &kv).unwrap();
        assert_eq!(packed.offset(), 1);
    }
}

#[test]
fn direct_dense_update_after_a_packed_layer_preserves_that_partial_step() {
    let q = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
    let kv = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
    let mut cache = hook_cache(2, None);
    let packed = cache
        .as_any_mut()
        .downcast_mut::<DenseFallbackPackedDecoderCache>()
        .unwrap();
    assert!(packed
        .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false,)
        .unwrap()
        .is_some());
    packed.update(1, &kv, &kv).unwrap();
    assert!(packed.dense_active);
    assert_eq!(packed.offset(), 1);
    assert_eq!(packed.dense.peek(0).unwrap().unwrap().0.shape()[2], 1);
    assert_eq!(packed.dense.peek(1).unwrap().unwrap().0.shape()[2], 1);
}

#[test]
fn factory_rejects_zero_query_unsupported_dimension_and_group_before_staging() {
    for (query_length, head_dimension, group_size) in [(0, 64, 32), (1, 32, 32), (1, 64, 4)] {
        let mut selection = select_decoder_cache(PackedCacheRequest {
            enabled: true,
            backend: "mlx-metal".into(),
            identity: "test-packed".into(),
            layers: 2,
            batch: 1,
            kv_heads: 1,
            head_dimension,
            group_size,
            bits: PackedCodeBits::Two,
            query_length,
            has_mask: false,
        });
        assert!(selection.cache.as_any_mut().is::<ContiguousKvCache>());
        assert!(matches!(
            selection.route(),
            CacheRoute::DenseFallback { .. }
        ));
    }
}

#[test]
fn unsupported_geometry_dtype_and_window_decline_before_first_append() {
    let valid_q = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
    let valid_kv = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
    let unsupported_q = Array::from_slice(&[1u8; 64], &[1, 1, 1, 64]);
    for (query, mask) in [
        (&unsupported_q, PackedAttentionMask::Causal),
        (&valid_q, PackedAttentionMask::SlidingWindow(0)),
        (&valid_q, PackedAttentionMask::SlidingWindow(usize::MAX)),
    ] {
        let mut cache = hook_cache(2, None);
        let packed = cache
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap();
        assert!(packed
            .try_packed_attention(0, query, &valid_kv, &valid_kv, mask, 0.125, false)
            .unwrap()
            .is_none());
        assert_eq!(packed.offset(), 0);
        assert!(packed.pending_step.is_none());
        assert!(packed.staged.layers.iter().all(Option::is_none));
        assert!(packed.staged.compiled_handle().is_none());
    }
}

#[test]
fn key_token_groups_and_value_channel_groups_are_chunk_invariant() {
    let (batch, heads, step, width, group) = (2, 2, 7, 5, 3);
    let keys = bhst_data(batch, heads, step, width, -9.0);
    let values = bhst_data(batch, heads, step, width, 4.0);
    let mut one = PackedGroupAffineKvCache::new("m", 1, batch, heads, width, group).unwrap();
    one.append(0, &keys, &values, step).unwrap();
    let mut chunks = PackedGroupAffineKvCache::new("m", 1, batch, heads, width, group).unwrap();
    for (start, len) in [(0, 1), (1, 2), (3, 1), (4, 3)] {
        chunks
            .append(
                0,
                &token_range(&keys, batch, heads, step, width, start, len),
                &token_range(&values, batch, heads, step, width, start, len),
                len,
            )
            .unwrap();
    }
    assert_same_rows(&one, &chunks);
    let store = one.layers[0].as_ref().unwrap();
    assert_eq!(store.keys.complete_tokens, 6);
    assert_eq!(store.keys.pending_tokens, 1);
    assert_eq!(store.keys.scales.len(), batch * heads * 2 * width);
    assert_eq!(
        store.values.scales.len(),
        batch * heads * step * width.div_ceil(group)
    );
    assert_eq!(
        one.representation().key_grouping,
        "token-axis groups [B,H,ceil(S/group_size),D]"
    );
}

#[test]
fn packed_argument_adapter_reorders_multi_batch_multi_head_storage_row_major() {
    let (batch, heads, step, width, group) = (2, 3, 8, 8, 4);
    let keys = bhst_data(batch, heads, step, width, -17.0);
    let values = bhst_data(batch, heads, step, width, 23.0);
    let mut cache = PackedGroupAffineKvCache::new("layout", 1, batch, heads, width, group).unwrap();
    cache.append(0, &keys, &values, step).unwrap();
    let (kc, ks, kz, vc, vs, vz) = cache.packed_mlx_arguments(0).unwrap();
    for array in [&kc, &ks, &kz, &vc, &vs, &vz] {
        array.eval().unwrap();
    }
    let storage = cache.layers[0].as_ref().unwrap();
    let key_bytes = storage.keys.code_bytes_per_group();
    let value_bytes = cache.code_bits().code_bytes(width);
    let value_groups = width.div_ceil(group);
    let groups = step.div_ceil(group);
    let expected_kc =
        row_major_outer_rows(&storage.keys.codes, groups, batch * heads, key_bytes).unwrap();
    let expected_ks =
        row_major_outer_rows(&storage.keys.scales, groups, batch * heads, width).unwrap();
    let expected_kz =
        row_major_outer_rows(&storage.keys.zeros, groups, batch * heads, width).unwrap();
    let expected_vc =
        row_major_outer_rows(&storage.values.codes, step, batch * heads, value_bytes).unwrap();
    let expected_vs =
        row_major_outer_rows(&storage.values.scales, step, batch * heads, value_groups).unwrap();
    let expected_vz =
        row_major_outer_rows(&storage.values.zeros, step, batch * heads, value_groups).unwrap();
    assert_eq!(kc.as_slice::<u8>(), expected_kc);
    assert_eq!(ks.as_slice::<f16>(), expected_ks);
    assert_eq!(kz.as_slice::<f16>(), expected_kz);
    assert_eq!(vc.as_slice::<u8>(), expected_vc);
    assert_eq!(vs.as_slice::<f16>(), expected_vs);
    assert_eq!(vz.as_slice::<f16>(), expected_vz);
    let device = cache.device_layers[0].as_ref().unwrap();
    assert_eq!(device.key_packed_tokens, groups * group);
    assert_eq!(device.value_packed_tokens, step);
    let initial = cache.dispatch_telemetry();
    assert_eq!(
        initial.uploaded_packed_bytes,
        initial.retained_device_packed_logical_bytes
    );
    let initial_representation = cache.representation();
    assert_eq!(
        initial_representation.allocated_bytes,
        initial_representation.host_allocated_payload_bytes
            + cache.retained_device_physical_bytes() as usize
    );
    assert!(
        cache.retained_device_physical_bytes()
            >= initial_representation.retained_device_packed_logical_bytes as u64,
        "block-preallocated device arrays are at least their live extents"
    );

    let extra_keys = bhst_data(batch, heads, 4, width, 101.0);
    let extra_values = bhst_data(batch, heads, 4, width, -203.0);
    cache.append(0, &extra_keys, &extra_values, 4).unwrap();
    cache.packed_mlx_arguments(0).unwrap();
    let extended = cache.dispatch_telemetry();
    assert!(
        extended.retained_device_packed_logical_bytes
            > initial.retained_device_packed_logical_bytes
    );
    assert_eq!(
        extended.uploaded_packed_bytes - initial.uploaded_packed_bytes,
        extended.retained_device_packed_logical_bytes
            - initial.retained_device_packed_logical_bytes,
        "extending a resident cache host-uploads only the new packed group/token payload"
    );

    cache.trim(5).unwrap();
    let extents = |cache: &PackedGroupAffineKvCache| {
        let device = cache.device_layers[0].as_ref().unwrap();
        (
            device.key_packed_tokens,
            device.value_packed_tokens,
            device.key_tail_rows,
        )
    };
    assert_eq!(
        extents(&cache),
        (4, 5, 0),
        "trim keeps exactly the mirror prefix that still matches the host representation"
    );
    let uploaded_before_rebuild = cache.dispatch_telemetry().uploaded_packed_bytes;
    cache.packed_mlx_arguments(0).unwrap();
    let after_trim = cache.dispatch_telemetry();
    assert_eq!(extents(&cache), (4, 5, 1));
    let pending_key_bytes = batch * heads * width * std::mem::size_of::<f32>();
    assert_eq!(
        after_trim.uploaded_packed_bytes - uploaded_before_rebuild,
        pending_key_bytes as u64,
        "after trim only the re-staged pending key row is uploaded; the kept prefix is not"
    );
}

#[cfg(target_os = "macos")]
#[allow(clippy::arc_with_non_send_sync)]
fn assert_real_metal_dense_sdpa_case(
    dtype: mlx_rs::Dtype,
    head_dimension: usize,
    kv_heads: usize,
    query_heads: usize,
    mask: PackedAttentionMask,
    family: crate::primitives::packed_metal::PackedMetalGpuFamily,
    bits: PackedCodeBits,
) {
    use crate::primitives::attention::{sdpa, AttnMask};
    use crate::primitives::packed_attention::{attention_f32_masked, PackedAttentionShape};
    use crate::primitives::packed_metal::{PackedMask, PackedMetalKernel};

    const BATCH: usize = 2;
    const QUERY_LEN: usize = 2;
    const KV_LEN: usize = 5;
    let identity = format!(
        "oracle-{dtype:?}-{head_dimension}-{kv_heads}-{query_heads}-{mask:?}-{family:?}-{bits:?}"
    );
    let keys = (0..BATCH * kv_heads * KV_LEN * head_dimension)
        .map(|index| ((index * 17 + 11) % 67) as f32 * 0.015625 - 0.5)
        .collect::<Vec<_>>();
    let values = (0..BATCH * kv_heads * KV_LEN * head_dimension)
        .map(|index| ((index * 29 + 7) % 79) as f32 * 0.01171875 - 0.4)
        .collect::<Vec<_>>();
    let queries = (0..BATCH * query_heads * QUERY_LEN * head_dimension)
        .map(|index| ((index * 13 + 5) % 53) as f32 * 0.0078125 - 0.2)
        .collect::<Vec<_>>();

    let mut cache = PackedGroupAffineKvCache::with_bits(
        identity.clone(),
        1,
        BATCH,
        kv_heads,
        head_dimension,
        PACKED_METAL_QUANT_GROUP_SIZE,
        bits,
    )
    .unwrap();
    cache.append(0, &keys, &values, KV_LEN).unwrap();
    let (_, dense_keys, dense_values) = cache.evaluated_dense_layer(0).unwrap();
    let query = Array::from_slice(
        &queries,
        &[
            BATCH as i32,
            query_heads as i32,
            QUERY_LEN as i32,
            head_dimension as i32,
        ],
    )
    .as_dtype(dtype)
    .unwrap();
    let dense_key = Array::from_slice(
        &dense_keys,
        &[
            BATCH as i32,
            kv_heads as i32,
            KV_LEN as i32,
            head_dimension as i32,
        ],
    )
    .as_dtype(dtype)
    .unwrap();
    let dense_value = Array::from_slice(
        &dense_values,
        &[
            BATCH as i32,
            kv_heads as i32,
            KV_LEN as i32,
            head_dimension as i32,
        ],
    )
    .as_dtype(dtype)
    .unwrap();
    let query_f32 = query.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    let key_f32 = dense_key.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    let value_f32 = dense_value.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    query_f32.eval().unwrap();
    key_f32.eval().unwrap();
    value_f32.eval().unwrap();
    let query_values = query_f32.as_slice::<f32>();
    let key_values = key_f32.as_slice::<f32>();
    let value_values = value_f32.as_slice::<f32>();
    let shape = PackedAttentionShape {
        batch: BATCH,
        query_heads,
        kv_heads,
        query_len: QUERY_LEN,
        kv_len: KV_LEN,
        head_dim: head_dimension,
    };
    let reference = attention_f32_masked(
        shape,
        query_values,
        |batch, head, token, channel| {
            key_values[((batch * kv_heads + head) * KV_LEN + token) * head_dimension + channel]
        },
        |batch, head, token, channel| {
            value_values[((batch * kv_heads + head) * KV_LEN + token) * head_dimension + channel]
        },
        (head_dimension as f32).powf(-0.5),
        mask,
    )
    .unwrap();
    assert!(reference.iter().any(|value| value.abs() > 1e-4));

    let dense_mask = match mask {
        PackedAttentionMask::None => AttnMask::None,
        PackedAttentionMask::Causal => AttnMask::Causal,
        PackedAttentionMask::SlidingWindow(window) => AttnMask::SlidingCausal {
            window: window as i32,
        },
        PackedAttentionMask::Additive => unreachable!("additive is a deliberate fallback"),
    };
    let dense_output = sdpa(
        &query,
        &dense_key,
        &dense_value,
        (head_dimension as f32).powf(-0.5),
        dense_mask,
    )
    .unwrap()
    .as_dtype(mlx_rs::Dtype::Float32)
    .unwrap();
    dense_output.eval().unwrap();

    cache
        .bind_compiled_handle(CompiledKernelHandle::new(Arc::new(
            PackedMetalKernel::for_identity_family_and_bits(identity, family, bits).unwrap(),
        )))
        .unwrap();
    let packed_mask = match mask {
        PackedAttentionMask::None => PackedMask::None,
        PackedAttentionMask::Causal => PackedMask::Causal,
        PackedAttentionMask::SlidingWindow(window) => PackedMask::SlidingWindow(window),
        PackedAttentionMask::Additive => unreachable!("additive is a deliberate fallback"),
    };
    let packed_output = cache
        .dispatch_packed(0, &query, packed_mask)
        .unwrap()
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap();
    packed_output.eval().unwrap();
    let tolerance = match dtype {
        mlx_rs::Dtype::Float32 => 8e-3,
        mlx_rs::Dtype::Float16 => 4e-2,
        mlx_rs::Dtype::Bfloat16 => 8e-2,
        _ => unreachable!(),
    };
    for (index, ((actual, dense), reference)) in packed_output
        .as_slice::<f32>()
        .iter()
        .zip(dense_output.as_slice::<f32>())
        .zip(reference.iter())
        .enumerate()
    {
        assert!(actual.is_finite(), "non-finite packed output at {index}");
        assert!(
                (actual - dense).abs() <= tolerance,
                "packed/dense mismatch at {index}: {actual} != {dense}; dtype={dtype:?} D={head_dimension} KVH={kv_heads} QH={query_heads} mask={mask:?}"
            );
        assert!(
                (actual - reference).abs() <= tolerance,
                "packed/independent mismatch at {index}: {actual} != {reference}; dtype={dtype:?} D={head_dimension} KVH={kv_heads} QH={query_heads} mask={mask:?}"
            );
    }
    assert_eq!(cache.direct_dispatches(), 1);
    assert_eq!(cache.full_cache_dequantizations(), 0);
    let telemetry = cache.dispatch_telemetry();
    assert_eq!(telemetry.cold_dispatches, 1);
    assert_eq!(telemetry.steady_dispatches, 0);
    assert!(telemetry.uploaded_packed_bytes > 0);
    assert!(telemetry.retained_device_packed_logical_bytes > 0);
    // The residual key group is a separate argument that stays resident; nothing transient
    // (no padded or concatenated copy) is built for the reader.
    assert_eq!(
        telemetry.peak_packed_argument_logical_bytes,
        telemetry.retained_device_packed_logical_bytes
    );
    assert!(
        telemetry.peak_packed_transient_logical_bytes
            >= telemetry.retained_device_packed_logical_bytes
    );
}

#[cfg(target_os = "macos")]
#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn real_metal_group32_matches_dense_sdpa_and_independent_oracle_across_full_surface() {
    use crate::primitives::packed_metal::PackedMetalGpuFamily;

    for bits in PackedCodeBits::ALL {
        let dtypes = [
            mlx_rs::Dtype::Float16,
            mlx_rs::Dtype::Bfloat16,
            mlx_rs::Dtype::Float32,
        ];
        let dimensions = [64, 128, 256];
        let masks = [
            PackedAttentionMask::None,
            PackedAttentionMask::Causal,
            PackedAttentionMask::SlidingWindow(3),
        ];
        for (dtype_index, dtype) in dtypes.into_iter().enumerate() {
            for (dimension_index, dimension) in dimensions.into_iter().enumerate() {
                let case = dtype_index * dimensions.len() + dimension_index;
                assert_real_metal_dense_sdpa_case(
                    dtype,
                    dimension,
                    2,
                    4,
                    masks[case % masks.len()],
                    if case.is_multiple_of(2) {
                        PackedMetalGpuFamily::ConservativeUnknownApple
                    } else {
                        PackedMetalGpuFamily::Apple7OrNewer
                    },
                    bits,
                );
            }
        }
        for (kv_heads, query_heads) in [(2, 2), (1, 4), (2, 4)] {
            assert_real_metal_dense_sdpa_case(
                mlx_rs::Dtype::Float32,
                64,
                kv_heads,
                query_heads,
                PackedAttentionMask::Causal,
                PackedMetalGpuFamily::Apple7OrNewer,
                bits,
            );
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn real_metal_adapter_dispatch_matches_nonuniform_fp32_reference_with_gqa_and_tail() {
    use crate::primitives::packed_metal::{PackedMask, PackedMetalKernel};
    for bits in PackedCodeBits::ALL {
        let (batch, kv_heads, q_heads, step, width, group) = (2, 2, 4, 5, 64, 32);
        let mut keys = Vec::with_capacity(batch * kv_heads * step * width);
        let mut values = Vec::with_capacity(batch * kv_heads * step * width);
        for row in 0..batch * kv_heads {
            for token in 0..step {
                for channel in 0..width {
                    keys.push(
                        row as f32 * 32.0
                            + (token / group) as f32 * 8.0
                            + (token % group) as f32 * 2.0
                            + channel as f32,
                    );
                    values.push(
                        row as f32 * 32.0
                            + token as f32 * 8.0
                            + (channel / group) as f32 * 4.0
                            + (channel % group) as f32,
                    );
                }
            }
        }
        let mut cache = PackedGroupAffineKvCache::with_bits(
            "device-layout",
            1,
            batch,
            kv_heads,
            width,
            group,
            bits,
        )
        .unwrap();
        cache.append(0, &keys, &values, step).unwrap();
        let (_, dequantized_keys, dequantized_values) = cache.evaluated_dense_layer(0).unwrap();
        let query_values = (0..batch * q_heads * width)
            .map(|index| (index as i32 % 7 - 3) as f32 * 0.01)
            .collect::<Vec<_>>();
        let query = Array::from_slice(
            &query_values,
            &[batch as i32, q_heads as i32, 1, width as i32],
        );
        let mut expected = vec![0.0f32; batch * q_heads * width];
        for b in 0..batch {
            for qh in 0..q_heads {
                let kh = qh / (q_heads / kv_heads);
                let row = b * kv_heads + kh;
                let q_base = (b * q_heads + qh) * width;
                let mut scores = Vec::with_capacity(step);
                for token in 0..step {
                    let mut dot = 0.0f32;
                    for channel in 0..width {
                        dot += query_values[q_base + channel]
                            * dequantized_keys[(row * step + token) * width + channel];
                    }
                    scores.push(dot / (width as f32).sqrt());
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights = scores
                    .iter()
                    .map(|score| (*score - maximum).exp())
                    .collect::<Vec<_>>();
                let norm = weights.iter().sum::<f32>();
                for channel in 0..width {
                    let mut total = 0.0f32;
                    for token in 0..step {
                        total += weights[token]
                            * dequantized_values[(row * step + token) * width + channel];
                    }
                    expected[q_base + channel] = total / norm;
                }
            }
        }
        cache
            .bind_compiled_handle(CompiledKernelHandle::new(Arc::new(
                PackedMetalKernel::for_identity_family_and_bits(
                    "device-layout",
                    crate::primitives::packed_metal::PackedMetalGpuFamily::ConservativeUnknownApple,
                    bits,
                )
                .unwrap(),
            )))
            .unwrap();
        let output = cache
            .dispatch_packed(0, &query, PackedMask::Causal)
            .unwrap();
        let actual = output.as_slice::<f32>();
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= 2e-3,
                "index {index}: {actual} != {expected}"
            );
        }
        let steady = cache
            .dispatch_packed(0, &query, PackedMask::Causal)
            .unwrap();
        assert_eq!(steady.as_slice::<f32>(), actual);
        assert_eq!(cache.direct_dispatches(), 2);
        assert_eq!(cache.full_cache_dequantizations(), 0);
        let telemetry = cache.dispatch_telemetry();
        assert_eq!(telemetry.cold_dispatches, 1);
        assert_eq!(telemetry.steady_dispatches, 1);
    }
}

#[test]
fn deterministic_pseudorandom_outliers_pack_identically_across_arbitrary_chunks() {
    for bits in PackedCodeBits::ALL {
        let (batch, heads, step, width, group) = (2, 3, 11, 7, 4);
        let keys = pseudo_random_outliers(batch, heads, step, width, 0x5eed_beef);
        let values = pseudo_random_outliers(batch, heads, step, width, 0x0123_4567);
        let mut one =
            PackedGroupAffineKvCache::with_bits("m", 1, batch, heads, width, group, bits).unwrap();
        one.append(0, &keys, &values, step).unwrap();
        let mut chunks =
            PackedGroupAffineKvCache::with_bits("m", 1, batch, heads, width, group, bits).unwrap();
        for (start, len) in [(0, 2), (2, 1), (3, 4), (7, 1), (8, 3)] {
            chunks
                .append(
                    0,
                    &token_range(&keys, batch, heads, step, width, start, len),
                    &token_range(&values, batch, heads, step, width, start, len),
                    len,
                )
                .unwrap();
        }
        assert_same_rows(&one, &chunks);
        for token in 0..step {
            for row in 0..batch * heads {
                let (k, v) = one.read_row(0, token, row).unwrap();
                assert!(k.iter().chain(v.iter()).all(|value| value.is_finite()));
            }
        }
    }
}

#[test]
fn rollback_reappend_clears_packed_tails_and_requantizes_cut_key_groups() {
    for bits in PackedCodeBits::ALL {
        let (batch, heads, width, group) = (1, 2, 5, 4);
        let original_keys = bhst_data(batch, heads, 6, width, -1000.0);
        let original_values = bhst_data(batch, heads, 6, width, 1000.0);
        let replacement_keys = bhst_data(batch, heads, 3, width, 700.0);
        let replacement_values = bhst_data(batch, heads, 3, width, -700.0);
        let mut rolled =
            PackedGroupAffineKvCache::with_bits("m", 1, batch, heads, width, group, bits).unwrap();
        rolled
            .append(0, &original_keys, &original_values, 6)
            .unwrap();
        rolled.rollback(3).unwrap();
        rolled
            .append(0, &replacement_keys, &replacement_values, 3)
            .unwrap();

        let mut expected_keys = Vec::new();
        let mut expected_values = Vec::new();
        for row in 0..batch * heads {
            let prefix = row * 6 * width;
            let suffix = row * 3 * width;
            expected_keys.extend_from_slice(&original_keys[prefix..prefix + 3 * width]);
            expected_keys.extend_from_slice(&replacement_keys[suffix..suffix + 3 * width]);
            expected_values.extend_from_slice(&original_values[prefix..prefix + 3 * width]);
            expected_values.extend_from_slice(&replacement_values[suffix..suffix + 3 * width]);
        }
        let mut expected =
            PackedGroupAffineKvCache::with_bits("m", 1, batch, heads, width, group, bits).unwrap();
        expected
            .append(0, &expected_keys, &expected_values, 6)
            .unwrap();
        assert_same_rows(&rolled, &expected);
        assert_eq!(rolled.logical_len(), 6);
    }
}

#[test]
fn pending_key_groups_snapshot_and_byte_accounting_are_strict() {
    for bits in PackedCodeBits::ALL {
        let (batch, heads, step, width, group) = (2, 1, 5, 7, 3);
        let keys = bhst_data(batch, heads, step, width, 0.0);
        let mut values = bhst_data(batch, heads, step, width, 0.0);
        values[0] = -1000.0;
        let last = values.len() - 1;
        values[last] = 1000.0;
        let mut cache =
            PackedGroupAffineKvCache::with_bits("m", 1, batch, heads, width, group, bits).unwrap();
        cache.append(0, &keys, &values, step).unwrap();
        let bytes_before = cache.logical_stored_bytes();
        assert!(cache.allocated_vec_bytes() >= bytes_before);
        assert!(cache.process_visible_bytes_estimate() >= cache.allocated_vec_bytes());
        let snapshot = cache.save().unwrap();
        let mut restored =
            PackedGroupAffineKvCache::with_bits("m", 1, batch, heads, width, group, bits).unwrap();
        restored.restore(&snapshot).unwrap();
        assert_same_rows(&cache, &restored);
        assert_eq!(
            cache.logical_stored_bytes(),
            restored.logical_stored_bytes()
        );
        let mut corrupt = snapshot.clone();
        let midpoint = corrupt.len() / 2;
        corrupt[midpoint] ^= 0x01;
        assert!(restored.restore(&corrupt).is_err());
        assert_same_rows(&cache, &restored);
        cache.clear();
        assert_eq!(cache.logical_stored_bytes(), 0);
    }
}

#[test]
fn long_context_capacity_including_pending_tail_reduces_physical_payload_vs_dense_fp16() {
    let (step, width, group) = (4097, 128, 32);
    let keys = pseudo_random_outliers(1, 1, step, width, 0xfeed_face);
    let values = pseudo_random_outliers(1, 1, step, width, 0xface_feed);
    let mut cache = PackedGroupAffineKvCache::new("m", 1, 1, 1, width, group).unwrap();
    cache.append(0, &keys, &values, step).unwrap();
    let layer = cache.layers[0].as_ref().unwrap();
    assert_eq!(layer.keys.pending_tokens, 1);
    assert!(layer.keys.pending.len() >= width);
    assert!(cache.allocated_payload_bytes() * 4 <= cache.dense_fp16_equivalent_bytes() * 3);
    assert!(cache.process_visible_bytes_estimate() >= cache.allocated_payload_bytes());
}

fn bhsd(values: &[f32], batch: usize, heads: usize, tokens: usize, width: usize) -> Array {
    Array::from_slice(
        values,
        &[batch as i32, heads as i32, tokens as i32, width as i32],
    )
}

fn device_cache_bits(
    batch: usize,
    heads: usize,
    width: usize,
    bits: PackedCodeBits,
) -> PackedGroupAffineKvCache {
    PackedGroupAffineKvCache::with_bits(
        "test-packed",
        1,
        batch,
        heads,
        width,
        PACKED_METAL_QUANT_GROUP_SIZE,
        bits,
    )
    .unwrap()
}

fn host_readback<T: mlx_rs::ArrayElement + Copy>(array: &Array) -> Vec<T> {
    let array = materialized(array).unwrap();
    array.eval().unwrap();
    array.as_slice::<T>().to_vec()
}

/// GPU quantize-on-append is bit-identical to the CPU reference (codes, f16 scales and zeros),
/// across chunked appends that complete groups mid-step, and the residual holds the exact
/// not-yet-quantized input. Block growth past 256 tokens preserves the resident history.
#[cfg(target_os = "macos")]
#[test]
fn device_append_quantizes_bit_identically_to_the_cpu_reference_across_chunks_and_growth() {
    for bits in PackedCodeBits::ALL {
        for width in [64, 128] {
            assert_device_append_matches_cpu_reference(bits, width);
        }
    }
}

#[cfg(target_os = "macos")]
fn assert_device_append_matches_cpu_reference(bits: PackedCodeBits, width: usize) {
    let (batch, heads) = (2, 2);
    let step = 301;
    let keys = pseudo_random_outliers(batch, heads, step, width, 0xabad_cafe);
    let values = pseudo_random_outliers(batch, heads, step, width, 0x1dea_f00d);
    let mut host = device_cache_bits(batch, heads, width, bits);
    host.append(0, &keys, &values, step).unwrap();
    let mut device = device_cache_bits(batch, heads, width, bits);
    for (start, len) in [(0, 1), (1, 30), (31, 5), (36, 41), (77, 224)] {
        let k = token_range(&keys, batch, heads, step, width, start, len);
        let v = token_range(&values, batch, heads, step, width, start, len);
        device
            .append_device(
                0,
                &bhsd(&k, batch, heads, len, width),
                &bhsd(&v, batch, heads, len, width),
            )
            .unwrap();
    }
    assert_eq!(device.logical_len(), step);
    assert!(device.layers.iter().all(Option::is_none), "no host copy");
    assert_eq!(device.host_allocated_payload_bytes(), 0);
    let storage = host.layers[0].as_ref().unwrap();
    let layer = device.device_layers[0].as_ref().unwrap();
    assert!(layer.authoritative);
    assert_eq!(
        layer.capacity_tokens, 512,
        "grown by whole 256-token blocks"
    );
    let (rows, groups) = (batch * heads, step / 32);
    assert_eq!(layer.key_packed_tokens, groups * 32);
    assert_eq!(layer.value_packed_tokens, groups * 32);
    assert_eq!(layer.key_tail_rows, step - groups * 32);
    let key_words = 32 * width / bits.codes_per_byte();
    assert_eq!(
        layer.key_codes.shape()[3] as usize,
        key_words,
        "{bits:?}: packed key row width"
    );
    assert_eq!(
        layer.value_codes.shape()[3] as usize,
        width / bits.codes_per_byte()
    );
    assert_eq!(
        host_readback::<u8>(&live_rows(&layer.key_codes, groups).unwrap()),
        row_major_outer_rows(&storage.keys.codes, groups, rows, key_words).unwrap()
    );
    for (device_array, host_values) in [
        (&layer.key_scales, &storage.keys.scales),
        (&layer.key_zeros, &storage.keys.zeros),
    ] {
        assert_eq!(
            host_readback::<f16>(&live_rows(device_array, groups).unwrap()),
            row_major_outer_rows(host_values, groups, rows, width).unwrap()
        );
    }
    let packed = groups * 32;
    let value_rows = |source: &[u8]| {
        let words = width / bits.codes_per_byte();
        row_major_outer_rows(&source[..packed * rows * words], packed, rows, words).unwrap()
    };
    assert_eq!(
        host_readback::<u8>(&live_rows(&layer.value_codes, packed).unwrap()),
        value_rows(&storage.values.codes)
    );
    for (device_array, host_values) in [
        (&layer.value_scales, &storage.values.scales),
        (&layer.value_zeros, &storage.values.zeros),
    ] {
        let groups_per_row = width / 32;
        assert_eq!(
            host_readback::<f16>(&live_rows(device_array, packed).unwrap()),
            row_major_outer_rows(
                &host_values[..packed * rows * groups_per_row],
                packed,
                rows,
                groups_per_row
            )
            .unwrap()
        );
    }
    let tail = step - packed;
    assert_eq!(
        host_readback::<f32>(&live_rows(&layer.key_tail, tail).unwrap()),
        token_range(&keys, batch, heads, step, width, packed, tail)
    );
    assert_eq!(
        host_readback::<f32>(&live_rows(&layer.value_tail, tail).unwrap()),
        token_range(&values, batch, heads, step, width, packed, tail)
    );
}

/// The fused reader against the independent fp32 oracle: decode over two K groups plus a
/// residual tail, a causal prefill chunk, and a split-KV history, at the campaign tolerance.
#[cfg(target_os = "macos")]
#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn fused_reader_matches_independent_fp32_oracle_at_campaign_tolerance() {
    use crate::primitives::packed_metal::{PackedMetalGpuFamily, PackedMetalKernel};
    for bits in PackedCodeBits::ALL {
        for family in [
            PackedMetalGpuFamily::ConservativeUnknownApple,
            PackedMetalGpuFamily::Apple7OrNewer,
        ] {
            let reader = CompiledKernelHandle::new(Arc::new(
                PackedMetalKernel::for_identity_family_and_bits("parity", family, bits).unwrap(),
            ));
            let errors = group_affine_kernel_fp32_parity_errors(&reader).unwrap();
            assert_eq!(errors.len(), 4 * 128 + 4 * 5 * 128 + 4 * 128);
            let max = errors.iter().copied().fold(0.0, f64::max);
            assert!(
                max <= crate::decode::observe::COMPRESSED_PARITY_MAX_ERROR,
                "{family:?}: max parity error {max}"
            );
            eprintln!("campaign parity {bits:?} {family:?} D=128: max abs {max}");
            // The SC-20676 gate runs at the model's own head dimension.
            for width in [64, 256] {
                let errors = group_affine_kernel_fp32_parity_errors_at(&reader, width).unwrap();
                assert_eq!(errors.len(), (4 + 4 * 5 + 4) * width);
                let max = errors.iter().copied().fold(0.0, f64::max);
                assert!(
                    max <= crate::decode::observe::COMPRESSED_PARITY_MAX_ERROR,
                    "{family:?} at {width}: max parity error {max}"
                );
            }
            assert!(group_affine_kernel_fp32_parity_errors_at(&reader, 100).is_err());
        }
    }
}

/// Kernel surface sweep: dtypes, head dimensions, masks, GQA, prefill rows, and forced split
/// counts (including more splits than blocks) all agree with the fp32 oracle over the device
/// representation; single-pass and split-KV outputs agree with each other.
#[cfg(target_os = "macos")]
#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn fused_reader_single_and_split_kv_paths_match_the_oracle_across_the_surface() {
    use crate::primitives::packed_attention::{attention_f32_masked, PackedAttentionShape};
    use crate::primitives::packed_metal::{PackedMask, PackedMetalGpuFamily, PackedMetalKernel};
    for bits in PackedCodeBits::ALL {
        let cases = [
            (Dtype::Float32, 64, 2, 4, 77, 1, PackedAttentionMask::None),
            (
                Dtype::Float16,
                128,
                2,
                6,
                100,
                3,
                PackedAttentionMask::Causal,
            ),
            (
                Dtype::Bfloat16,
                256,
                1,
                2,
                70,
                1,
                PackedAttentionMask::Causal,
            ),
            (
                Dtype::Float32,
                128,
                2,
                4,
                131,
                4,
                PackedAttentionMask::SlidingWindow(45),
            ),
            (
                Dtype::Float32,
                64,
                4,
                4,
                32,
                32,
                PackedAttentionMask::Causal,
            ),
        ];
        for (index, (dtype, width, kv_heads, query_heads, tokens, query_len, mask)) in
            cases.into_iter().enumerate()
        {
            let batch = 2;
            // Clamped outliers, then a distinct range per 32-channel group and per token so every
            // K channel group and V channel-group scale/zero differs (a wrong metadata index is
            // numerically visible).
            let shaped = |seed: u64| {
                pseudo_random_outliers(batch, kv_heads, tokens, width, seed)
                    .into_iter()
                    .enumerate()
                    .map(|(i, value)| {
                        let (channel, token) = (i % width, (i / width) % tokens);
                        value.clamp(-4.0, 4.0) * (0.25 + (channel / 32) as f32 * 0.5)
                            + (channel / 32) as f32 * 0.75
                            - (token % 7) as f32 * 0.125
                    })
                    .collect::<Vec<_>>()
            };
            let keys = shaped(11 + index as u64);
            let values = shaped(97 + index as u64);
            let queries = (0..batch * query_heads * query_len * width)
                .map(|i| ((i * 37 + 3) % 61) as f32 * 0.02 - 0.6)
                .collect::<Vec<_>>();
            let mut cache = device_cache_bits(batch, kv_heads, width, bits);
            cache
                .append_device(
                    0,
                    &bhsd(&keys, batch, kv_heads, tokens, width)
                        .as_dtype(dtype)
                        .unwrap(),
                    &bhsd(&values, batch, kv_heads, tokens, width)
                        .as_dtype(dtype)
                        .unwrap(),
                )
                .unwrap();
            let query = bhsd(&queries, batch, query_heads, query_len, width)
                .as_dtype(dtype)
                .unwrap();
            let query_f32 = host_readback::<f32>(&query.as_dtype(Dtype::Float32).unwrap());
            let (_, dense_keys, dense_values) = cache.evaluated_dense_layer(0).unwrap();
            let reference = attention_f32_masked(
                PackedAttentionShape {
                    batch,
                    query_heads,
                    kv_heads,
                    query_len,
                    kv_len: tokens,
                    head_dim: width,
                },
                &query_f32,
                |b, h, t, d| dense_keys[((b * kv_heads + h) * tokens + t) * width + d],
                |b, h, t, d| dense_values[((b * kv_heads + h) * tokens + t) * width + d],
                (width as f32).powf(-0.5),
                mask,
            )
            .unwrap();
            let packed_mask = match mask {
                PackedAttentionMask::None => PackedMask::None,
                PackedAttentionMask::Causal => PackedMask::Causal,
                PackedAttentionMask::SlidingWindow(window) => PackedMask::SlidingWindow(window),
                PackedAttentionMask::Additive => unreachable!(),
            };
            let tolerance = match dtype {
                Dtype::Float32 => 1e-4,
                Dtype::Float16 => 4e-2,
                _ => 8e-2,
            };
            let layer = cache.device_layers[0].as_ref().unwrap();
            let args = layer.args(&query, packed_mask);
            let mut single = None;
            let mut case_max = 0.0f32;
            for family in [
                PackedMetalGpuFamily::ConservativeUnknownApple,
                PackedMetalGpuFamily::Apple7OrNewer,
            ] {
                let kernel =
                    PackedMetalKernel::for_identity_family_and_bits("sweep", family, bits).unwrap();
                for splits in [1, 2, 3, 7, 64] {
                    let output = kernel
                        .dispatch_with_splits(&args, Some(splits))
                        .unwrap()
                        .as_dtype(Dtype::Float32)
                        .unwrap();
                    let output = host_readback::<f32>(&output);
                    for (i, (actual, expected)) in output.iter().zip(&reference).enumerate() {
                        case_max = case_max.max((actual - expected).abs());
                        assert!(
                            (actual - expected).abs() <= tolerance,
                            "{bits:?} case {index} {family:?} splits {splits} element {i}: {actual} != {expected}"
                        );
                    }
                    let single = single.get_or_insert_with(|| output.clone());
                    for (actual, expected) in output.iter().zip(single.iter()) {
                        assert!((actual - expected).abs() <= tolerance / 4.0);
                    }
                }
            }
            eprintln!(
                "per-row parity {bits:?} case {index} {dtype:?} D={width}: max abs {case_max}"
            );
        }
    }
}

/// Whole-step rollback across a group flush: a later-layer fault after layer 0 quantized a
/// completed group restores both layers' extents and residuals exactly, and the retry produces
/// the same representation as an uninterrupted cache.
#[cfg(target_os = "macos")]
#[test]
fn rollback_after_a_group_flush_restores_the_exact_device_representation() {
    let width = 64;
    let history = pseudo_random_outliers(1, 1, 31, width, 5);
    let fresh = pseudo_random_outliers(1, 1, 2, width, 6);
    let run = |fail_on: Option<usize>| {
        let (mut cache, _) = hook_cache_geometry(2, 1, 1, width, fail_on);
        let packed = cache
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap();
        let seed = bhsd(&history, 1, 1, 31, width);
        assert!(packed
            .import_prefix(&[(seed.clone(), seed.clone()), (seed.clone(), seed)])
            .unwrap());
        let before = packed.staged.device_layers.clone();
        let q = bhsd(&vec![0.1; 2 * width], 1, 1, 2, width);
        let kv = bhsd(&fresh, 1, 1, 2, width);
        let first = packed.try_packed_attention(
            0,
            &q,
            &kv,
            &kv,
            PackedAttentionMask::Causal,
            (width as f32).powf(-0.5),
            false,
        );
        assert!(first.unwrap().is_some());
        assert_eq!(
            packed.staged.device_layers[0]
                .as_ref()
                .unwrap()
                .key_packed_tokens,
            32,
            "layer 0 flushed a completed group"
        );
        // The flush replaced layer 0's residual arrays; the pending step holds the originals
        // for rollback and the receipt counts them.
        let held = packed.pending_device_snapshot_overhead();
        assert!(held > 0);
        assert_eq!(
            packed.model_evidence().retained_device_metadata_bytes,
            packed.staged.retained_device_component_bytes().1 as u64 + held
        );
        let second = packed.try_packed_attention(
            1,
            &q,
            &kv,
            &kv,
            PackedAttentionMask::Causal,
            (width as f32).powf(-0.5),
            false,
        );
        (cache, before, second.map(|output| output.is_some()))
    };
    let (mut failed, before, outcome) = run(Some(2));
    assert!(outcome.is_err());
    let packed = failed
        .as_any_mut()
        .downcast_mut::<DenseFallbackPackedDecoderCache>()
        .unwrap();
    assert_eq!(packed.offset(), 31);
    for (actual, expected) in packed.staged.device_layers.iter().zip(before.iter()) {
        assert_same_device_layer(actual, expected);
    }
    let (mut clean, _, outcome) = run(None);
    assert!(outcome.unwrap());
    let q = bhsd(&vec![0.1; 2 * width], 1, 1, 2, width);
    let kv = bhsd(&fresh, 1, 1, 2, width);
    for layer in 0..2 {
        assert!(packed
            .try_packed_attention(
                layer,
                &q,
                &kv,
                &kv,
                PackedAttentionMask::Causal,
                (width as f32).powf(-0.5),
                false,
            )
            .unwrap()
            .is_some());
    }
    let clean = clean
        .as_any_mut()
        .downcast_mut::<DenseFallbackPackedDecoderCache>()
        .unwrap();
    for (actual, expected) in packed
        .staged
        .device_layers
        .iter()
        .zip(clean.staged.device_layers.iter())
    {
        assert_same_device_layer(actual, expected);
    }
}

/// sc-20681: speculative rollback on the contiguous compressed cache is exact and stays
/// compressed. After a 20-row verify that completes and quantizes a group is cut back to
/// `kept` rows, every layer's live codes, metadata and residual equal a cache that appended
/// only the kept rows. Without the armed window the same cut re-stages the group from its
/// codes, which is not exact (the control).
#[cfg(target_os = "macos")]
#[test]
fn speculative_rollback_on_the_contiguous_cache_is_exact_and_stays_compressed() {
    let width = 64;
    let history = pseudo_random_outliers(1, 1, 50, width, 31);
    let draft = pseudo_random_outliers(1, 1, 20, width, 32);
    let scale = (width as f32).powf(-0.5);
    let cache = |arm: bool, rows: usize| {
        let (mut cache, _) = hook_cache_geometry(2, 1, 1, width, None);
        let packed = cache
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap();
        let seed = bhsd(&history, 1, 1, 50, width);
        assert!(packed
            .import_prefix(&[(seed.clone(), seed.clone()), (seed.clone(), seed)])
            .unwrap());
        if arm {
            packed.begin_speculation().unwrap();
        }
        let q = bhsd(&vec![0.1; rows * width], 1, 1, rows, width);
        let kv = bhsd(&draft[..rows * width], 1, 1, rows, width);
        for layer in 0..2 {
            assert!(packed
                .try_packed_attention(
                    layer,
                    &q,
                    &kv,
                    &kv,
                    PackedAttentionMask::Causal,
                    scale,
                    false
                )
                .unwrap()
                .is_some());
        }
        cache
    };
    fn packed(cache: &mut Box<dyn KvCache>) -> &mut DenseFallbackPackedDecoderCache {
        cache
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap()
    }
    for kept in [1, 9, 14, 19] {
        let mut speculated = cache(true, 20);
        let spec = packed(&mut speculated);
        assert_eq!(
            spec.staged.device_layers[0]
                .as_ref()
                .unwrap()
                .key_packed_tokens,
            64,
            "the verify completed a group"
        );
        spec.truncate((50 + kept) as i32).unwrap();
        assert!(!spec.dense_active, "speculation stays compressed");
        assert!(spec.compressed_storage().unwrap().is_some());
        let mut reference = cache(false, kept);
        let reference = packed(&mut reference);
        assert_eq!(spec.offset(), reference.offset());
        for (actual, expected) in spec
            .staged
            .device_layers
            .iter()
            .zip(reference.staged.device_layers.iter())
        {
            assert_same_device_layer(actual, expected);
        }

        if 50 + kept < 64 {
            let mut control = cache(false, 20);
            let control = packed(&mut control);
            control.truncate((50 + kept) as i32).unwrap();
            assert_ne!(
                control.staged.evaluated_dense_layer(0).unwrap(),
                reference.staged.evaluated_dense_layer(0).unwrap(),
                "kept {kept}: re-staging the cut group from its codes is not exact"
            );
        }
    }
}

/// Trim into a completed group re-stages the kept prefix of that group from its quantized
/// values, so the reconstructed history is exactly the kept prefix of the previous one.
#[cfg(target_os = "macos")]
#[test]
fn device_trim_into_a_completed_group_restages_the_quantized_prefix() {
    for bits in PackedCodeBits::ALL {
        let (batch, heads, width, tokens) = (1, 2, 64, 77);
        let keys = pseudo_random_outliers(batch, heads, tokens, width, 21);
        let values = pseudo_random_outliers(batch, heads, tokens, width, 22);
        let mut cache = device_cache_bits(batch, heads, width, bits);
        cache
            .append_device(
                0,
                &bhsd(&keys, batch, heads, tokens, width),
                &bhsd(&values, batch, heads, tokens, width),
            )
            .unwrap();
        let (_, full_keys, full_values) = cache.evaluated_dense_layer(0).unwrap();
        for keep in [70, 40, 32, 5] {
            cache.trim(keep).unwrap();
            let (resident, kept_keys, kept_values) = cache.evaluated_dense_layer(0).unwrap();
            assert_eq!(resident, keep);
            let layer = cache.device_layers[0].as_ref().unwrap();
            assert_eq!(layer.key_packed_tokens, keep / 32 * 32);
            assert_eq!(layer.key_tail_rows, keep % 32);
            for row in 0..batch * heads {
                let prefix = |full: &[f32]| {
                    full[row * tokens * width..(row * tokens + keep) * width].to_vec()
                };
                let kept =
                    |dense: &[f32]| dense[row * keep * width..(row + 1) * keep * width].to_vec();
                assert_eq!(
                    kept(&kept_keys),
                    prefix(&full_keys),
                    "keep {keep} row {row}"
                );
                assert_eq!(
                    kept(&kept_values),
                    prefix(&full_values),
                    "keep {keep} row {row}"
                );
            }
        }
        assert!(
            cache.save().is_err(),
            "a dense value residual cannot be snapshotted"
        );
    }
}

/// The decoder route end to end: a long first step attends densely over its own fresh K/V
/// (exactly the dense SDPA result) and appends packed; the following decode steps use the
/// fused reader lazily against the oracle; nothing falls back or reconstructs.
#[cfg(target_os = "macos")]
#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn decoder_first_step_is_dense_over_fresh_kv_then_decode_is_fused_and_lazy() {
    use crate::primitives::attention::{sdpa, AttnMask};
    use crate::primitives::packed_attention::{attention_f32_masked, PackedAttentionShape};
    use crate::primitives::packed_metal::PackedMetalKernel;
    for bits in PackedCodeBits::ALL {
        let (heads, query_heads, width, prompt) = (2, 4, 128, 40);
        let reader = CompiledKernelHandle::new(Arc::new(
            PackedMetalKernel::for_identity_family_and_bits(
                "decoder-e2e",
                crate::primitives::packed_metal::PackedMetalGpuFamily::ConservativeUnknownApple,
                bits,
            )
            .unwrap(),
        ));
        let mut cache = select_decoder_cache_with_reader(
            PackedCacheRequest {
                enabled: true,
                backend: "mlx-metal".into(),
                identity: "decoder-e2e".into(),
                layers: 1,
                batch: 1,
                kv_heads: heads,
                head_dimension: width,
                group_size: PACKED_METAL_QUANT_GROUP_SIZE,
                bits,
                query_length: 1,
                has_mask: false,
            },
            reader,
        )
        .into_cache();
        let scale = (width as f32).powf(-0.5);
        let clamp = |values: Vec<f32>| {
            values
                .into_iter()
                .map(|v| v.clamp(-3.0, 3.0))
                .collect::<Vec<_>>()
        };
        let keys = bhsd(
            &clamp(pseudo_random_outliers(1, heads, prompt, width, 31)),
            1,
            heads,
            prompt,
            width,
        );
        let values = bhsd(
            &clamp(pseudo_random_outliers(1, heads, prompt, width, 32)),
            1,
            heads,
            prompt,
            width,
        );
        let queries = bhsd(
            &(0..query_heads * prompt * width)
                .map(|i| ((i * 13 + 1) % 41) as f32 * 0.03 - 0.6)
                .collect::<Vec<_>>(),
            1,
            query_heads,
            prompt,
            width,
        );
        let first = cache
            .try_packed_attention(
                0,
                &queries,
                &keys,
                &values,
                PackedAttentionMask::Causal,
                scale,
                false,
            )
            .unwrap()
            .unwrap();
        let dense = sdpa(&queries, &keys, &values, scale, AttnMask::Causal).unwrap();
        assert_eq!(host_readback::<f32>(&first), host_readback::<f32>(&dense));
        let evidence = cache.packed_evidence().unwrap();
        assert_eq!(
            evidence.accepted_direct_calls, 0,
            "no reader dispatch for the fresh-only step"
        );
        assert_eq!(cache.offset(), prompt as i32);

        for step in 0..3 {
            let q = bhsd(
                &vec![0.05 * (step + 1) as f32; query_heads * width],
                1,
                query_heads,
                1,
                width,
            );
            let fresh = clamp(pseudo_random_outliers(1, heads, 1, width, 40 + step as u64));
            let kv = bhsd(&fresh, 1, heads, 1, width);
            let output = cache
                .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, scale, false)
                .unwrap()
                .unwrap();
            let packed = cache
                .as_any_mut()
                .downcast_mut::<DenseFallbackPackedDecoderCache>()
                .unwrap();
            let (tokens, dense_keys, dense_values) =
                packed.staged.evaluated_dense_layer(0).unwrap();
            assert_eq!(tokens, prompt + step + 1);
            let reference = attention_f32_masked(
                PackedAttentionShape {
                    batch: 1,
                    query_heads,
                    kv_heads: heads,
                    query_len: 1,
                    kv_len: tokens,
                    head_dim: width,
                },
                &host_readback::<f32>(&q),
                |_, h, t, d| dense_keys[(h * tokens + t) * width + d],
                |_, h, t, d| dense_values[(h * tokens + t) * width + d],
                scale,
                PackedAttentionMask::Causal,
            )
            .unwrap();
            for (actual, expected) in host_readback::<f32>(&output).iter().zip(&reference) {
                assert!((actual - expected).abs() <= 1e-4, "{actual} != {expected}");
            }
        }
        let evidence = cache.packed_evidence().unwrap();
        assert_eq!(evidence.accepted_direct_calls, 3);
        assert_eq!(evidence.full_cache_dequantizations, 0);
        assert!(!evidence.dense_active && evidence.fallback_reasons.is_empty());
        assert!(evidence.accepted_uploaded_packed_bytes > 0);
        let storage = cache.compressed_storage().unwrap().unwrap();
        assert_eq!(
            storage.host_payload_bytes, 0,
            "the decoder route keeps no host copy"
        );
        assert_eq!(storage.tokens, (prompt + 3) as u64);
    }
}

/// Prefix import quantizes on append into the packed representation (no reconstruction); it
/// declines without mutation when the route is not live and empty or the geometry differs.
#[cfg(target_os = "macos")]
#[test]
fn prefix_import_appends_packed_and_declines_without_mutation() {
    let width = 64;
    let seed = bhsd(&pseudo_random_outliers(1, 1, 45, width, 3), 1, 1, 45, width);
    let wrong = bhsd(&pseudo_random_outliers(1, 1, 45, 128, 3), 1, 1, 45, 128);
    let mut cache = hook_cache(2, None);
    assert!(!cache
        .import_prefix(&[(seed.clone(), seed.clone())])
        .unwrap());
    assert!(!cache
        .import_prefix(&[(seed.clone(), seed.clone()), (wrong.clone(), wrong)])
        .unwrap());
    assert_eq!(cache.offset(), 0);
    assert!(cache
        .import_prefix(&[(seed.clone(), seed.clone()), (seed.clone(), seed.clone())])
        .unwrap());
    assert_eq!(cache.offset(), 45);
    assert!(!cache
        .import_prefix(&[(seed.clone(), seed.clone()), (seed.clone(), seed)])
        .unwrap());
    let evidence = cache.packed_evidence().unwrap();
    assert_eq!(evidence.full_cache_dequantizations, 0);
    assert!(evidence.fallback_reasons.is_empty() && !evidence.dense_active);
    assert!(evidence.accepted_uploaded_packed_bytes > 0);
    let q = bhsd(&vec![0.2; width], 1, 1, 1, width);
    let kv = bhsd(&vec![0.3; width], 1, 1, 1, width);
    for layer in 0..2 {
        assert!(cache
            .try_packed_attention(
                layer,
                &q,
                &kv,
                &kv,
                PackedAttentionMask::Causal,
                0.125,
                false
            )
            .unwrap()
            .is_some());
    }
    assert_eq!(cache.offset(), 46);
    assert!(!ContiguousKvCache::new(1)
        .import_prefix(&[(q.clone(), q)])
        .unwrap());
}

/// A reader that never produced an output faulting on its first dispatch over resident history
/// (the first step went dense over fresh K/V, or a prefix was imported) rolls the step back and
/// hands the exact packed history to the dense cache through the observable transition.
#[cfg(target_os = "macos")]
#[test]
fn cold_reader_fault_over_resident_history_transitions_to_dense() {
    let width = 64;
    let (mut cache, kernel) = hook_cache_geometry(2, 1, 1, width, Some(1));
    let seed = bhsd(&pseudo_random_outliers(1, 1, 40, width, 8), 1, 1, 40, width);
    assert!(cache
        .import_prefix(&[(seed.clone(), seed.clone()), (seed.clone(), seed)])
        .unwrap());
    let q = bhsd(&vec![0.1; width], 1, 1, 1, width);
    let kv = bhsd(&vec![0.2; width], 1, 1, 1, width);
    assert!(cache
        .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
        .unwrap()
        .is_none());
    assert_eq!(kernel.calls.load(Ordering::Relaxed), 1);
    let evidence = cache.packed_evidence().unwrap();
    assert!(evidence.dense_active);
    assert_eq!(evidence.full_cache_dequantizations, 1);
    assert!(evidence
        .fallback_reasons
        .iter()
        .any(|(operation, _)| operation == "dispatch-fault"));
    assert_eq!(cache.offset(), 40, "the faulted step was rolled back");
    for layer in 0..2 {
        let (keys, _) = cache.update(layer, &kv, &kv).unwrap();
        assert_eq!(keys.shape()[2], 41);
    }
}

/// sc-20682: a dense transition over resident history is admitted before it allocates. The
/// admission sees exactly the dense cache it would build (every layer at block-rounded
/// capacity) plus one layer's reconstruction transient; a refusal fails the step typed and
/// leaves the packed history intact, and an admitted transition proceeds as before.
#[cfg(target_os = "macos")]
#[test]
fn a_dense_transition_is_admitted_before_it_reconstructs_history() {
    use std::cell::Cell;
    use std::rc::Rc;
    let width = 64;
    let (mut cache, _kernel) = hook_cache_geometry(2, 1, 1, width, Some(1));
    let seed = bhsd(&pseudo_random_outliers(1, 1, 40, width, 8), 1, 1, 40, width);
    assert!(cache
        .import_prefix(&[(seed.clone(), seed.clone()), (seed.clone(), seed)])
        .unwrap());
    let packed = cache
        .as_any_mut()
        .downcast_mut::<DenseFallbackPackedDecoderCache>()
        .unwrap();
    // Two layers x 256-token block x 1 row x 64 channels x (f32 K + f32 V), plus one layer's
    // 40 tokens of f32 dequantization (K, V) beside the cast K/V.
    let expected = 2 * 256 * 64 * 8 + 40 * 64 * (8 + 8);
    assert_eq!(packed.dense_reconstruction_bytes(), expected);
    let seen = Rc::new(Cell::new(0));
    let refusal = core_llm::RequestResourceExhausted {
        prompt_tokens: 40,
        max_new_tokens: 1,
        max_context_tokens: 64,
        required_bytes: 1,
        available_bytes: 0,
    };
    let observed = seen.clone();
    packed.set_dense_transition_admission(DenseTransitionAdmission::new(move |bytes| {
        observed.set(bytes);
        Err(Error::ResourceExhausted(refusal))
    }));
    let q = bhsd(&vec![0.1; width], 1, 1, 1, width);
    let kv = bhsd(&vec![0.2; width], 1, 1, 1, width);
    let error = packed
        .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
        .unwrap_err();
    assert!(matches!(error, Error::ResourceExhausted(evidence) if evidence == refusal));
    assert_eq!(seen.get(), expected);
    let evidence = packed.packed_evidence().unwrap();
    assert!(
        !evidence.dense_active,
        "a refused transition reconstructs nothing"
    );
    assert_eq!(evidence.full_cache_dequantizations, 0);
    assert_eq!(packed.offset(), 40);

    packed.set_dense_transition_admission(DenseTransitionAdmission::new(|_| Ok(())));
    let (keys, _) = packed.update(0, &kv, &kv).unwrap();
    assert_eq!(keys.shape()[2], 41);
    let evidence = packed.packed_evidence().unwrap();
    assert!(evidence.dense_active);
    assert_eq!(evidence.full_cache_dequantizations, 1);
}

/// sc-20682: a transition with nothing resident to rebuild is still admitted (with zero
/// reconstruction bytes) — the generation continues dense from there.
#[cfg(target_os = "macos")]
#[test]
fn an_empty_dense_transition_is_still_admitted() {
    use std::cell::Cell;
    use std::rc::Rc;
    let width = 64;
    let (mut cache, _kernel) = hook_cache_geometry(2, 1, 1, width, None);
    let packed = cache
        .as_any_mut()
        .downcast_mut::<DenseFallbackPackedDecoderCache>()
        .unwrap();
    assert_eq!(packed.dense_reconstruction_bytes(), 0);
    let seen = Rc::new(Cell::new(None));
    let observed = seen.clone();
    packed.set_dense_transition_admission(DenseTransitionAdmission::new(move |bytes| {
        observed.set(Some(bytes));
        Err(Error::Msg("refused".into()))
    }));
    let kv = bhsd(&vec![0.2; width], 1, 1, 1, width);
    assert!(packed.update(0, &kv, &kv).is_err());
    assert_eq!(seen.get(), Some(0));
    assert!(!packed.packed_evidence().unwrap().dense_active);
}

/// Oracle output over the device representation `cache` holds for layer 0.
#[cfg(target_os = "macos")]
fn tiled_oracle(
    cache: &PackedGroupAffineKvCache,
    query_f32: &[f32],
    shape: crate::primitives::packed_attention::PackedAttentionShape,
    mask: PackedAttentionMask,
) -> Vec<f32> {
    let (tokens, keys, values) = cache.evaluated_dense_layer(0).unwrap();
    assert_eq!(tokens, shape.kv_len);
    let (heads, width) = (shape.kv_heads, shape.head_dim);
    crate::primitives::packed_attention::attention_f32_masked(
        shape,
        query_f32,
        |b, h, t, d| keys[((b * heads + h) * tokens + t) * width + d],
        |b, h, t, d| values[((b * heads + h) * tokens + t) * width + d],
        (width as f32).powf(-0.5),
        mask,
    )
    .unwrap()
}

/// The tiled multi-row reader against the independent fp32 oracle at the existing tolerances:
/// `S_q` 9/33/128/257 (partial last tiles, several tiles, rows spanning GQA heads), K and V
/// residual tails beside packed groups, causal rows offset to the end of the KV range, GQA 1–3,
/// a sliding window narrower than the chunk (also at D = 256), the host-mirror layout whose V rows
/// are fully packed while K keeps a residual tail (`value_packed != key_packed`), head
/// dimensions 64/128/256, all query dtypes, and
/// forced KV split counts (the partial + reduction path), each agreeing with the single pass.
#[cfg(target_os = "macos")]
#[test]
fn tiled_reader_matches_independent_fp32_oracle_for_multi_row_steps() {
    use crate::primitives::packed_attention::PackedAttentionShape;
    use crate::primitives::packed_metal::{PackedMask, PackedMetalGpuFamily, PackedMetalKernel};
    for bits in PackedCodeBits::ALL {
        let cases = [
            (
                Dtype::Float32,
                64,
                2,
                6,
                300,
                9,
                PackedAttentionMask::Causal,
            ),
            (
                Dtype::Float32,
                128,
                2,
                4,
                333,
                33,
                PackedAttentionMask::Causal,
            ),
            (
                Dtype::Float32,
                128,
                1,
                3,
                401,
                128,
                PackedAttentionMask::SlidingWindow(77),
            ),
            (
                Dtype::Float32,
                64,
                2,
                2,
                530,
                257,
                PackedAttentionMask::Causal,
            ),
            (
                Dtype::Float32,
                256,
                1,
                2,
                110,
                33,
                PackedAttentionMask::Causal,
            ),
            (Dtype::Float32, 64, 1, 3, 150, 33, PackedAttentionMask::None),
            (
                Dtype::Float16,
                128,
                2,
                6,
                290,
                33,
                PackedAttentionMask::Causal,
            ),
            (
                Dtype::Bfloat16,
                64,
                1,
                4,
                270,
                128,
                PackedAttentionMask::SlidingWindow(200),
            ),
            (
                Dtype::Float32,
                256,
                1,
                2,
                300,
                33,
                PackedAttentionMask::SlidingWindow(90),
            ),
            // Host reference cache: V rows quantize per token, K keeps its incomplete group dense.
            (
                Dtype::Float32,
                128,
                2,
                4,
                333,
                33,
                PackedAttentionMask::Causal,
            ),
        ];
        const HOST_MIRROR_CASE: usize = 9;
        let kernel = PackedMetalKernel::for_identity_family_and_bits(
            "tiled",
            PackedMetalGpuFamily::Apple7OrNewer,
            bits,
        )
        .unwrap();
        let mut maxima = Vec::new();
        for (index, (dtype, width, kv_heads, query_heads, tokens, query_len, mask)) in
            cases.into_iter().enumerate()
        {
            let batch = if index % 2 == 0 { 2 } else { 1 };
            // Distinct ranges per token and per 32-channel group so a wrong K group, V row, or
            // metadata index is numerically visible.
            let shaped = |seed: u64| {
                pseudo_random_outliers(batch, kv_heads, tokens, width, seed)
                    .into_iter()
                    .enumerate()
                    .map(|(i, value)| {
                        let (channel, token) = (i % width, (i / width) % tokens);
                        value.clamp(-4.0, 4.0) * (0.25 + (channel / 32) as f32 * 0.5)
                            + (channel / 32) as f32 * 0.75
                            - (token % 7) as f32 * 0.125
                    })
                    .collect::<Vec<_>>()
            };
            let keys = shaped(211 + index as u64);
            let values = shaped(307 + index as u64);
            let queries = (0..batch * query_heads * query_len * width)
                .map(|i| ((i * 37 + 3) % 61) as f32 * 0.02 - 0.6)
                .collect::<Vec<_>>();
            let mut cache = device_cache_bits(batch, kv_heads, width, bits);
            if index == HOST_MIRROR_CASE {
                cache.append(0, &keys, &values, tokens).unwrap();
            } else {
                cache
                    .append_device(
                        0,
                        &bhsd(&keys, batch, kv_heads, tokens, width)
                            .as_dtype(dtype)
                            .unwrap(),
                        &bhsd(&values, batch, kv_heads, tokens, width)
                            .as_dtype(dtype)
                            .unwrap(),
                    )
                    .unwrap();
            }
            let query = bhsd(&queries, batch, query_heads, query_len, width)
                .as_dtype(dtype)
                .unwrap();
            let reference = tiled_oracle(
                &cache,
                &host_readback::<f32>(&query.as_dtype(Dtype::Float32).unwrap()),
                PackedAttentionShape {
                    batch,
                    query_heads,
                    kv_heads,
                    query_len,
                    kv_len: tokens,
                    head_dim: width,
                },
                mask,
            );
            let packed_mask = match mask {
                PackedAttentionMask::None => PackedMask::None,
                PackedAttentionMask::Causal => PackedMask::Causal,
                PackedAttentionMask::SlidingWindow(window) => PackedMask::SlidingWindow(window),
                PackedAttentionMask::Additive => unreachable!(),
            };
            let tolerance = match dtype {
                Dtype::Float32 => 1e-4,
                Dtype::Float16 => 4e-2,
                _ => 8e-2,
            };
            let staged = cache.staged_reader_arguments(0).unwrap();
            let args = staged.args(&query, packed_mask);
            assert!(
                args.key_packed_tokens > 0 && args.key_packed_tokens < tokens,
                "{bits:?} case {index} exercises packed K groups and the K residual"
            );
            assert_eq!(
                args.value_packed_tokens == tokens,
                index == HOST_MIRROR_CASE,
                "only the host-mirror case packs every V row ahead of K"
            );
            let mut single: Option<Vec<f32>> = None;
            let mut case_max = 0.0f32;
            for splits in [1, 2, 5] {
                let output = host_readback::<f32>(
                    &kernel
                        .dispatch_tiled(&args, Some(splits))
                        .unwrap()
                        .as_dtype(Dtype::Float32)
                        .unwrap(),
                );
                assert_eq!(output.len(), reference.len());
                for (i, (actual, expected)) in output.iter().zip(&reference).enumerate() {
                    let error = (actual - expected).abs();
                    case_max = case_max.max(error);
                    assert!(
                        error <= tolerance,
                        "{bits:?} case {index} splits {splits} element {i}: {actual} != {expected}"
                    );
                }
                let single = single.get_or_insert_with(|| output.clone());
                for (actual, expected) in output.iter().zip(single.iter()) {
                    assert!((actual - expected).abs() <= tolerance / 4.0);
                }
            }
            maxima.push((index, dtype, case_max));
        }
        eprintln!("tiled parity max abs error per case: {maxima:?}");
    }
}

/// The NAX tiled reader against the independent fp32 oracle at the tolerances the dense bf16/f16
/// SDPA path is held to (`assert_real_metal_dense_sdpa_case`): `S_q` 16/33/128/2048, K and V
/// residual tails (the pending K group) beside packed groups, causal rows offset to the end of
/// the KV range, GQA 1–3, sliding windows, `None` masks, the host-mirror layout
/// (`value_packed != key_packed`), D = 64/128, bf16 and f16 queries, and forced KV split counts
/// (partial + reduction), each agreeing with the single pass. Each case also matches the fp32
/// tiled kernel over the same inputs at 2e-2 (relative above 1), a bound a one-token mask or
/// offset error in short rows exceeds. D = 256 and f32 queries are routed to the fp32 tiled
/// kernel with the recorded reason. Requires a Neural-Accelerator Mac (ignored elsewhere).
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires a Neural-Accelerator Mac (mlx::core::metal::is_nax_available()); run with --ignored"]
fn nax_tiled_reader_matches_independent_fp32_oracle_for_multi_row_steps() {
    use crate::primitives::packed_attention::PackedAttentionShape;
    use crate::primitives::packed_metal::{
        mlx_nax_available, PackedKernelPath, PackedMask, PackedMetalGpuFamily, PackedMetalKernel,
    };
    for bits in PackedCodeBits::ALL {
        assert!(
            mlx_nax_available().unwrap(),
            "mlx::core::metal::is_nax_available() is false on this host"
        );
        let cases = [
            (
                Dtype::Bfloat16,
                128,
                2,
                6,
                333,
                33,
                PackedAttentionMask::Causal,
            ),
            (
                Dtype::Float16,
                64,
                2,
                4,
                300,
                16,
                PackedAttentionMask::Causal,
            ),
            (
                Dtype::Bfloat16,
                64,
                1,
                3,
                401,
                128,
                PackedAttentionMask::SlidingWindow(77),
            ),
            (
                Dtype::Float16,
                128,
                2,
                2,
                530,
                128,
                PackedAttentionMask::None,
            ),
            (
                Dtype::Bfloat16,
                64,
                1,
                2,
                2100,
                2048,
                PackedAttentionMask::Causal,
            ),
            (
                Dtype::Float16,
                128,
                1,
                3,
                290,
                33,
                PackedAttentionMask::SlidingWindow(200),
            ),
            (
                Dtype::Float16,
                128,
                1,
                2,
                2090,
                2048,
                PackedAttentionMask::SlidingWindow(700),
            ),
            // Host reference cache: V rows quantize per token, K keeps its incomplete group dense.
            (
                Dtype::Bfloat16,
                128,
                2,
                4,
                333,
                33,
                PackedAttentionMask::Causal,
            ),
        ];
        const HOST_MIRROR_CASE: usize = 7;
        let kernel = PackedMetalKernel::for_identity_family_and_bits(
            "nax",
            PackedMetalGpuFamily::Apple7OrNewer,
            bits,
        )
        .unwrap();
        assert!(kernel.nax_available());
        let mut maxima = Vec::new();
        for (index, (dtype, width, kv_heads, query_heads, tokens, query_len, mask)) in
            cases.into_iter().enumerate()
        {
            let batch = if index % 2 == 0 && query_len < 2048 {
                2
            } else {
                1
            };
            let shaped = |seed: u64| {
                pseudo_random_outliers(batch, kv_heads, tokens, width, seed)
                    .into_iter()
                    .enumerate()
                    .map(|(i, value)| {
                        let (channel, token) = (i % width, (i / width) % tokens);
                        value.clamp(-4.0, 4.0) * (0.25 + (channel / 32) as f32 * 0.5)
                            + (channel / 32) as f32 * 0.75
                            - (token % 7) as f32 * 0.125
                    })
                    .collect::<Vec<_>>()
            };
            let keys = shaped(411 + index as u64);
            let values = shaped(507 + index as u64);
            let queries = (0..batch * query_heads * query_len * width)
                .map(|i| ((i * 37 + 3) % 61) as f32 * 0.02 - 0.6)
                .collect::<Vec<_>>();
            let mut cache = device_cache_bits(batch, kv_heads, width, bits);
            if index == HOST_MIRROR_CASE {
                cache.append(0, &keys, &values, tokens).unwrap();
            } else {
                cache
                    .append_device(
                        0,
                        &bhsd(&keys, batch, kv_heads, tokens, width)
                            .as_dtype(dtype)
                            .unwrap(),
                        &bhsd(&values, batch, kv_heads, tokens, width)
                            .as_dtype(dtype)
                            .unwrap(),
                    )
                    .unwrap();
            }
            let query = bhsd(&queries, batch, query_heads, query_len, width)
                .as_dtype(dtype)
                .unwrap();
            let reference = tiled_oracle(
                &cache,
                &host_readback::<f32>(&query.as_dtype(Dtype::Float32).unwrap()),
                PackedAttentionShape {
                    batch,
                    query_heads,
                    kv_heads,
                    query_len,
                    kv_len: tokens,
                    head_dim: width,
                },
                mask,
            );
            let packed_mask = match mask {
                PackedAttentionMask::None => PackedMask::None,
                PackedAttentionMask::Causal => PackedMask::Causal,
                PackedAttentionMask::SlidingWindow(window) => PackedMask::SlidingWindow(window),
                PackedAttentionMask::Additive => unreachable!(),
            };
            let tolerance = match dtype {
                Dtype::Float16 => 4e-2,
                _ => 8e-2,
            };
            let staged = cache.staged_reader_arguments(0).unwrap();
            let args = staged.args(&query, packed_mask);
            assert!(
                args.key_packed_tokens > 0 && args.key_packed_tokens < tokens,
                "{bits:?} case {index} exercises packed K groups and the K residual"
            );
            assert_eq!(
                args.value_packed_tokens == tokens,
                index == HOST_MIRROR_CASE,
                "only the host-mirror case packs every V row ahead of K"
            );
            assert!(
                matches!(
                    kernel.planned_path(&args).unwrap(),
                    PackedKernelPath::NaxTiled { .. }
                ),
                "{bits:?} case {index} plans the NAX kernel"
            );
            let mut single: Option<Vec<f32>> = None;
            let (mut case_abs, mut case_rel) = (0.0f32, 0.0f32);
            for splits in [1, 2, 5] {
                let output = host_readback::<f32>(
                    &kernel
                        .dispatch_nax(&args, Some(splits))
                        .unwrap()
                        .as_dtype(Dtype::Float32)
                        .unwrap(),
                );
                assert_eq!(output.len(), reference.len());
                for (i, (actual, expected)) in output.iter().zip(&reference).enumerate() {
                    let error = (actual - expected).abs();
                    case_abs = case_abs.max(error);
                    case_rel = case_rel.max(error / expected.abs().max(1e-2));
                    assert!(
                        error <= tolerance,
                        "{bits:?} case {index} splits {splits} element {i}: {actual} != {expected}"
                    );
                }
                let single = single.get_or_insert_with(|| output.clone());
                // Split and single pass round the same fp32 result to the 16-bit output, so
                // they may differ by one output ULP, which grows with |x| (bf16 2^-5 from 4):
                // the bound is relative above 1, the form the NAX-vs-tiled check below uses.
                for (actual, expected) in output.iter().zip(single.iter()) {
                    assert!(
                        (actual - expected).abs() <= tolerance / 4.0 * expected.abs().max(1.0),
                        "{bits:?} case {index} splits {splits}: {actual} != single {expected}"
                    );
                }
            }
            // Same inputs through the fp32 tiled kernel: both read the identical representation, so
            // they differ only by 16-bit tile rounding and output rounding.
            let tiled = host_readback::<f32>(
                &kernel
                    .dispatch_tiled(&args, None)
                    .unwrap()
                    .as_dtype(Dtype::Float32)
                    .unwrap(),
            );
            let mut case_tiled = 0.0f32;
            for (i, (nax, tiled)) in single.as_ref().unwrap().iter().zip(&tiled).enumerate() {
                let error = (nax - tiled).abs() / tiled.abs().max(1.0);
                case_tiled = case_tiled.max(error);
                assert!(
                    error <= 2e-2,
                    "{bits:?} case {index} element {i}: NAX {nax} != tiled {tiled}"
                );
            }
            maxima.push((
                index, dtype, width, query_len, case_abs, case_rel, case_tiled,
            ));
        }
        eprintln!("NAX parity (case, dtype, D, S_q, max abs, max rel, max vs tiled): {maxima:?}");

        // D = 256 and f32 queries keep the fp32 tiled kernel, with the recorded reason.
        for (width, dtype) in [(256, Dtype::Bfloat16), (128, Dtype::Float32)] {
            let selection = kernel.nax_selection(width, dtype);
            assert!(!selection.selected, "D {width} {dtype:?}");
            let descriptor = kernel.kernel_descriptor(64, width, dtype).unwrap();
            assert_eq!(
                descriptor.kernel,
                "sc20676_tiled_multi_row_simdgroup_matrix"
            );
            assert_eq!(descriptor.selection, selection.selection);
            assert_eq!(descriptor.selection_reason, selection.reason);
            let mut cache = device_cache_bits(1, 1, width, bits);
            let kv = bhsd(
                &pseudo_random_outliers(1, 1, 80, width, 3)
                    .into_iter()
                    .map(|value| value.clamp(-3.0, 3.0))
                    .collect::<Vec<_>>(),
                1,
                1,
                80,
                width,
            )
            .as_dtype(dtype)
            .unwrap();
            cache.append_device(0, &kv, &kv).unwrap();
            let query = bhsd(&vec![0.1; 2 * 64 * width], 1, 2, 64, width)
                .as_dtype(dtype)
                .unwrap();
            let staged = cache.staged_reader_arguments(0).unwrap();
            let args = staged.args(&query, PackedMask::Causal);
            assert!(matches!(
                kernel.planned_path(&args).unwrap(),
                PackedKernelPath::Tiled { .. }
            ));
            let refused = kernel.dispatch_nax(&args, None).unwrap_err().to_string();
            assert!(refused.contains(selection.reason), "{refused}");
        }
    }
}

/// `dispatch` runs exactly the kernel `planned_path` names: on the qualified family a tiled
/// kernel from the per-D threshold (16 for D = 64/128, 32 for D = 256) and the per-row kernel
/// below it — the NAX tiled kernel for bf16/f16 queries at D = 64/128 where MLX reports the
/// Neural Accelerator, the fp32 tiled kernel otherwise (f32 queries, D = 256, or no NAX); on the
/// conservative family always the per-row kernel (and both tiled seams are refused). The routed
/// output is bit-identical to the named path's output, and the descriptor names the same kernel
/// with the selection reason.
#[cfg(target_os = "macos")]
#[test]
fn dispatch_runs_the_path_planned_path_names() {
    use crate::primitives::packed_metal::{
        mlx_nax_available, packed_tiled_min_query_tokens, PackedKernelPath, PackedMask,
        PackedMetalGpuFamily, PackedMetalKernel,
    };
    for bits in PackedCodeBits::ALL {
        assert_eq!(packed_tiled_min_query_tokens(64), 16);
        assert_eq!(packed_tiled_min_query_tokens(128), 16);
        assert_eq!(packed_tiled_min_query_tokens(256), 32);
        let nax = mlx_nax_available().unwrap();
        let conservative = PackedMetalKernel::for_identity_family_and_bits(
            "route",
            crate::primitives::packed_metal::PackedMetalGpuFamily::ConservativeUnknownApple,
            bits,
        )
        .unwrap();
        assert!(!conservative.nax_available());
        let recent = PackedMetalKernel::for_identity_family_and_bits(
            "route",
            PackedMetalGpuFamily::Apple7OrNewer,
            bits,
        )
        .unwrap();
        assert_eq!(recent.nax_available(), nax);
        let recent_without_nax = PackedMetalKernel::for_identity_family_and_bits(
            "route",
            PackedMetalGpuFamily::Apple7OrNewer,
            bits,
        )
        .unwrap()
        .without_nax();
        for width in [128, 256] {
            let tokens = 96;
            let data = pseudo_random_outliers(1, 2, tokens, width, 9)
                .into_iter()
                .map(|value| value.clamp(-3.0, 3.0))
                .collect::<Vec<_>>();
            let mut cache = device_cache_bits(1, 2, width, bits);
            let kv = bhsd(&data, 1, 2, tokens, width);
            cache.append_device(0, &kv, &kv).unwrap();
            let layer = cache.device_layers[0].as_ref().unwrap();
            let threshold = packed_tiled_min_query_tokens(width);
            for (query_len, dtype) in [1, threshold - 1, threshold]
                .into_iter()
                .flat_map(|len| [(len, Dtype::Float32), (len, Dtype::Bfloat16)])
            {
                let query = bhsd(
                    &(0..6 * query_len * width)
                        .map(|i| ((i * 7 + 1) % 23) as f32 * 0.04 - 0.4)
                        .collect::<Vec<_>>(),
                    1,
                    6,
                    query_len,
                    width,
                )
                .as_dtype(dtype)
                .unwrap();
                let args = layer.args(&query, PackedMask::Causal);
                for (kernel, qualified, kernel_nax) in [
                    (&conservative, false, false),
                    (&recent, true, nax),
                    (&recent_without_nax, true, false),
                ] {
                    let path = kernel.planned_path(&args).unwrap();
                    let multi_row = qualified && query_len >= threshold;
                    let nax_expected =
                        multi_row && kernel_nax && width == 128 && dtype == Dtype::Bfloat16;
                    let expected = match (multi_row, nax_expected) {
                        (false, _) => "sc20676_split_kv_simdgroup",
                        (true, false) => "sc20676_tiled_multi_row_simdgroup_matrix",
                        (true, true) => "sc20676_nax_tiled_matmul2d",
                    };
                    let context = format!(
                        "D {width} S_q {query_len} {dtype:?} qualified {qualified} nax {kernel_nax}"
                    );
                    assert_eq!(
                        (
                            matches!(path, PackedKernelPath::Tiled { .. }),
                            matches!(path, PackedKernelPath::NaxTiled { .. })
                        ),
                        (multi_row && !nax_expected, nax_expected),
                        "{context}: {path:?}"
                    );
                    let (PackedKernelPath::Tiled { splits }
                    | PackedKernelPath::PerRow { splits }
                    | PackedKernelPath::NaxTiled { splits }) = path;
                    assert_eq!(kernel.planned_splits(&args).unwrap(), splits);
                    let descriptor = kernel.kernel_descriptor(query_len, width, dtype).unwrap();
                    assert_eq!(descriptor.kernel, expected, "{context}");
                    if multi_row {
                        let nax_selection = kernel.nax_selection(width, dtype);
                        assert_eq!(descriptor.selection, nax_selection.selection, "{context}");
                        assert_eq!(
                            descriptor.selection_reason, nax_selection.reason,
                            "{context}"
                        );
                    }
                    // The dispatch-time selection the cache records is this descriptor's.
                    let recorded = kernel.kernel_selection(&args).unwrap();
                    assert_eq!(
                        (recorded.kernel, recorded.selection, recorded.reason),
                        (
                            descriptor.kernel,
                            descriptor.selection,
                            descriptor.selection_reason
                        ),
                        "{context}"
                    );
                    assert_eq!(
                        Some(recorded.query_dtype),
                        crate::primitives::packed_query_dtype_name(dtype)
                    );
                    assert!(crate::primitives::packed_kernel_path_valid(
                        kernel.gpu_family().as_str(),
                        recorded.kernel,
                        recorded.selection,
                        recorded.query_dtype
                    ));
                    let readback = |array: Array| {
                        host_readback::<f32>(&array.as_dtype(Dtype::Float32).unwrap())
                    };
                    let routed = readback(kernel.dispatch(&args).unwrap());
                    let named = readback(match path {
                        PackedKernelPath::Tiled { splits } => {
                            kernel.dispatch_tiled(&args, Some(splits)).unwrap()
                        }
                        PackedKernelPath::PerRow { splits } => {
                            kernel.dispatch_with_splits(&args, Some(splits)).unwrap()
                        }
                        PackedKernelPath::NaxTiled { splits } => {
                            kernel.dispatch_nax(&args, Some(splits)).unwrap()
                        }
                    });
                    assert_eq!(routed, named, "{context} {path:?}");
                    assert_eq!(
                        kernel.dispatch_nax(&args, None).is_ok(),
                        kernel_nax && width == 128 && dtype == Dtype::Bfloat16,
                        "{context}: the NAX seam follows nax_selection"
                    );
                }
                assert!(conservative.dispatch_tiled(&args, None).is_err());
            }
        }
    }
}

/// Chunked prefill through the decoder route: after the dense first chunk, each later
/// multi-row chunk attends over the packed history plus its own fresh K/V through the tiled
/// reader — the NAX tiled kernel for bf16 activations where MLX reports the Neural Accelerator,
/// the fp32 tiled kernel for f32 — matches the fp32 oracle over the device representation, and
/// never reconstructs the cache densely.
#[cfg(target_os = "macos")]
#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn decoder_chunked_prefill_uses_the_tiled_reader_without_dense_reconstruction() {
    use crate::primitives::packed_attention::PackedAttentionShape;
    use crate::primitives::packed_metal::{
        mlx_nax_available, PackedKernelPath, PackedMask, PackedMetalGpuFamily, PackedMetalKernel,
    };
    for bits in PackedCodeBits::ALL {
        let (heads, query_heads, width) = (2, 6, 128);
        let nax = mlx_nax_available().unwrap();
        for (dtype, tolerance) in [(Dtype::Float32, 1e-4), (Dtype::Bfloat16, 8e-2)] {
            let kernel = Arc::new(
                PackedMetalKernel::for_identity_family_and_bits(
                    "decoder-tiled",
                    PackedMetalGpuFamily::Apple7OrNewer,
                    bits,
                )
                .unwrap(),
            );
            let reader = CompiledKernelHandle::new(kernel.clone());
            let mut cache = select_decoder_cache_with_reader(
                PackedCacheRequest {
                    enabled: true,
                    backend: "mlx-metal".into(),
                    identity: "decoder-tiled".into(),
                    layers: 1,
                    batch: 1,
                    kv_heads: heads,
                    head_dimension: width,
                    group_size: PACKED_METAL_QUANT_GROUP_SIZE,
                    bits,
                    query_length: 40,
                    has_mask: false,
                },
                reader,
            )
            .into_cache();
            let scale = (width as f32).powf(-0.5);
            let mut offset = 0;
            for (chunk, rows) in [40usize, 33, 128].into_iter().enumerate() {
                let fresh = |seed: u64| {
                    bhsd(
                        &pseudo_random_outliers(1, heads, rows, width, seed)
                            .into_iter()
                            .map(|value| value.clamp(-3.0, 3.0))
                            .collect::<Vec<_>>(),
                        1,
                        heads,
                        rows,
                        width,
                    )
                    .as_dtype(dtype)
                    .unwrap()
                };
                let (keys, values) = (fresh(50 + chunk as u64), fresh(60 + chunk as u64));
                let query = bhsd(
                    &(0..query_heads * rows * width)
                        .map(|i| ((i * 13 + chunk) % 41) as f32 * 0.03 - 0.6)
                        .collect::<Vec<_>>(),
                    1,
                    query_heads,
                    rows,
                    width,
                )
                .as_dtype(dtype)
                .unwrap();
                let query_values = host_readback::<f32>(&query.as_dtype(Dtype::Float32).unwrap());
                let output = cache
                    .try_packed_attention(
                        0,
                        &query,
                        &keys,
                        &values,
                        PackedAttentionMask::Causal,
                        scale,
                        false,
                    )
                    .unwrap()
                    .unwrap();
                offset += rows;
                if chunk == 0 {
                    continue;
                }
                let packed = cache
                    .as_any_mut()
                    .downcast_mut::<DenseFallbackPackedDecoderCache>()
                    .unwrap();
                let reference = tiled_oracle(
                    &packed.staged,
                    &query_values,
                    PackedAttentionShape {
                        batch: 1,
                        query_heads,
                        kv_heads: heads,
                        query_len: rows,
                        kv_len: offset,
                        head_dim: width,
                    },
                    PackedAttentionMask::Causal,
                );
                let output = host_readback::<f32>(&output.as_dtype(Dtype::Float32).unwrap());
                for (actual, expected) in output.iter().zip(&reference) {
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "{dtype:?}: {actual} != {expected}"
                    );
                }
                // The step ran the planned multi-row kernel: its output is bit-identical to a
                // direct dispatch of the kernel planned_path names over the same representation.
                let staged = packed.staged.staged_reader_arguments(0).unwrap();
                let args = staged.args(&query, PackedMask::Causal);
                let direct = match kernel.planned_path(&args).unwrap() {
                    PackedKernelPath::NaxTiled { splits } if dtype == Dtype::Bfloat16 && nax => {
                        kernel.dispatch_nax(&args, Some(splits)).unwrap()
                    }
                    PackedKernelPath::Tiled { splits } if dtype == Dtype::Float32 || !nax => {
                        kernel.dispatch_tiled(&args, Some(splits)).unwrap()
                    }
                    path => panic!("{dtype:?} chunk {chunk} of {rows} rows planned {path:?}"),
                };
                assert_eq!(
                    output,
                    host_readback::<f32>(&direct.as_dtype(Dtype::Float32).unwrap()),
                    "{dtype:?} chunk {chunk} output is not the planned kernel's"
                );
            }
            let evidence = cache.packed_evidence().unwrap();
            assert_eq!(evidence.accepted_direct_calls, 2);
            // Both packed chunks are recorded under the path they actually ran.
            let (kernel, selection, query_dtype) = match (dtype, nax) {
                (Dtype::Bfloat16, true) => (
                    crate::primitives::PACKED_NAX_KERNEL,
                    crate::primitives::PACKED_SELECTION_NAX,
                    "bfloat16",
                ),
                (Dtype::Bfloat16, false) => (
                    crate::primitives::PACKED_TILED_KERNEL,
                    crate::primitives::PACKED_SELECTION_NAX_UNAVAILABLE,
                    "bfloat16",
                ),
                _ => (
                    crate::primitives::PACKED_TILED_KERNEL,
                    if nax {
                        crate::primitives::PACKED_SELECTION_F32_QUERY
                    } else {
                        crate::primitives::PACKED_SELECTION_NAX_UNAVAILABLE
                    },
                    "float32",
                ),
            };
            assert_eq!(
                evidence
                    .kernel_paths
                    .iter()
                    .map(|path| (
                        path.kernel.as_str(),
                        path.selection.as_str(),
                        path.query_dtype.as_str(),
                        path.calls
                    ))
                    .collect::<Vec<_>>(),
                vec![(kernel, selection, query_dtype, 2)],
                "{dtype:?}"
            );
            assert_eq!(evidence.full_cache_dequantizations, 0);
            assert!(!evidence.dense_active && evidence.fallback_reasons.is_empty());
        }
    }
}

/// Llama-3.2-3B-shaped synthetic long-context K/V (bf16, `[1, H, S, D]`) with the features that
/// stress the stored representation: rotating high-magnitude key channels (RoPE-like, slow and
/// fast, so per-channel key groups carry large offsets and wide ranges), constant key-channel
/// biases, and a few "needle" tokens whose keys align with the query so attention mass
/// concentrates far back in the history instead of averaging to noise. Every value is
/// bf16-exact: `keys`/`values` are the host copies of exactly what the dense arm holds.
#[cfg(target_os = "macos")]
struct LongContextKv {
    kv_heads: usize,
    query_heads: usize,
    tokens: usize,
    width: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
    key_array: Array,
    value_array: Array,
}

#[cfg(target_os = "macos")]
fn long_context_uniform(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

/// The ±1 direction query head `head` points along (needle keys of its KV head align with it).
#[cfg(target_os = "macos")]
fn long_context_direction(head: usize, channel: usize) -> f32 {
    if (channel * 7 + head * 13 + channel / 5).is_multiple_of(3) {
        -1.0
    } else {
        1.0
    }
}

#[cfg(target_os = "macos")]
fn long_context_needles(tokens: usize) -> [usize; 4] {
    [17, tokens / 3, tokens / 2 + 5, tokens - 700]
}

#[cfg(target_os = "macos")]
fn long_context_kv(
    kv_heads: usize,
    query_heads: usize,
    tokens: usize,
    width: usize,
) -> LongContextKv {
    let gqa = query_heads / kv_heads;
    let needles = long_context_needles(tokens);
    let mut state = 0x5eed_1234_abcd_ef01u64;
    let mut keys = Vec::with_capacity(kv_heads * tokens * width);
    let mut values = Vec::with_capacity(kv_heads * tokens * width);
    for head in 0..kv_heads {
        for token in 0..tokens {
            let needle = needles.contains(&token);
            for channel in 0..width {
                let noise = long_context_uniform(&mut state) * 1.7;
                // Rotating outlier pairs: channel c and c + D/2 carry a magnitude-6 rotation at a
                // slow (1e-3 rad/token) or fast (0.9 rad/token) frequency, like RoPE'd outliers.
                let pair = channel % (width / 2);
                let angle = token as f32 * if pair.is_multiple_of(2) { 1e-3 } else { 0.9 };
                let rotation = if pair % 11 == 3 {
                    if channel < width / 2 {
                        6.0 * angle.cos()
                    } else {
                        6.0 * angle.sin()
                    }
                } else {
                    0.0
                };
                let bias = if channel % 37 == 5 { 4.5 } else { 0.0 };
                let key = if needle {
                    3.0 * long_context_direction(head * gqa, channel) + noise * 0.1 + bias
                } else {
                    noise + rotation + bias
                };
                keys.push(key);
                let value = if needle {
                    if channel % 8 == head % 8 {
                        3.0
                    } else {
                        -0.5
                    }
                } else {
                    long_context_uniform(&mut state) * 1.7 + (channel % 5) as f32 * 0.1
                };
                values.push(value);
            }
        }
    }
    let shape = [1, kv_heads as i32, tokens as i32, width as i32];
    let key_array = Array::from_slice(&keys, &shape)
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let value_array = Array::from_slice(&values, &shape)
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let keys = host_readback::<f32>(&key_array.as_dtype(Dtype::Float32).unwrap());
    let values = host_readback::<f32>(&value_array.as_dtype(Dtype::Float32).unwrap());
    LongContextKv {
        kv_heads,
        query_heads,
        tokens,
        width,
        keys,
        values,
        key_array,
        value_array,
    }
}

/// bf16 query rows for the last `rows` positions (`[1, Hq, rows, D]`) and their exact values.
#[cfg(target_os = "macos")]
fn long_context_query(kv: &LongContextKv, rows: usize) -> (Array, Vec<f32>) {
    let mut state = 0x0bad_5eed_0000_0001u64 ^ rows as u64;
    let mut query = Vec::with_capacity(kv.query_heads * rows * kv.width);
    for head in 0..kv.query_heads {
        for _ in 0..rows {
            for channel in 0..kv.width {
                // Aligned with the KV head's needle direction (the group's first head exactly,
                // the others partially), plus noise.
                let aligned =
                    long_context_direction(head - head % (kv.query_heads / kv.kv_heads), channel);
                let weight = if head % (kv.query_heads / kv.kv_heads) == 0 {
                    0.35
                } else {
                    0.2
                };
                query.push(weight * aligned + long_context_uniform(&mut state) * 0.15);
            }
        }
    }
    let array = Array::from_slice(
        &query,
        &[1, kv.query_heads as i32, rows as i32, kv.width as i32],
    )
    .as_dtype(Dtype::Bfloat16)
    .unwrap();
    let exact = host_readback::<f32>(&array.as_dtype(Dtype::Float32).unwrap());
    (array, exact)
}

/// fp64 causal attention of the last `rows` positions over host `[H, S, D]` K/V (the oracle).
#[cfg(target_os = "macos")]
fn long_context_oracle(
    kv: &LongContextKv,
    query: &[f32],
    rows: usize,
    keys: &[f32],
    values: &[f32],
) -> Vec<f64> {
    let (tokens, width) = (kv.tokens, kv.width);
    let gqa = kv.query_heads / kv.kv_heads;
    let scale = 1.0 / (width as f64).sqrt();
    let mut out = vec![0.0f64; kv.query_heads * rows * width];
    let mut scores = vec![0.0f64; tokens];
    for head in 0..kv.query_heads {
        let kv_head = head / gqa;
        let k = &keys[kv_head * tokens * width..(kv_head + 1) * tokens * width];
        let v = &values[kv_head * tokens * width..(kv_head + 1) * tokens * width];
        for row in 0..rows {
            let position = tokens - rows + row;
            let q = &query[(head * rows + row) * width..(head * rows + row + 1) * width];
            let mut max = f64::NEG_INFINITY;
            for (token, score) in scores.iter_mut().enumerate().take(position + 1) {
                let dot = k[token * width..(token + 1) * width]
                    .iter()
                    .zip(q)
                    .map(|(k, q)| f64::from(*k) * f64::from(*q))
                    .sum::<f64>();
                *score = dot * scale;
                max = max.max(*score);
            }
            let mut norm = 0.0;
            let target = &mut out[(head * rows + row) * width..(head * rows + row + 1) * width];
            for (token, score) in scores.iter().enumerate().take(position + 1) {
                let weight = (score - max).exp();
                norm += weight;
                for (accumulator, value) in target
                    .iter_mut()
                    .zip(&v[token * width..(token + 1) * width])
                {
                    *accumulator += weight * f64::from(*value);
                }
            }
            for accumulator in target.iter_mut() {
                *accumulator /= norm;
            }
        }
    }
    out
}

/// Relative L2 error of `actual` against `expected`.
#[cfg(target_os = "macos")]
fn relative_l2(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let (mut error, mut norm) = (0.0, 0.0);
    for (actual, expected) in actual.iter().zip(expected) {
        error += (actual - expected).powi(2);
        norm += expected.powi(2);
    }
    (error / norm).sqrt()
}

/// The token ranges one fill appends, in order. `chunk` 0 is one whole-history append (the
/// prefix-import shape); otherwise chunks of `chunk` tokens, with the last `decode_tail` tokens
/// appended one at a time (decode steps after a chunked prefill).
#[cfg(target_os = "macos")]
fn long_context_fill(tokens: usize, chunk: usize, decode_tail: usize) -> Vec<(usize, usize)> {
    if chunk == 0 {
        return vec![(0, tokens)];
    }
    let prefill = tokens - decode_tail;
    let mut ranges = (0..prefill)
        .step_by(chunk)
        .map(|start| (start, (start + chunk).min(prefill)))
        .collect::<Vec<_>>();
    ranges.extend((prefill..tokens).map(|token| (token, token + 1)));
    ranges
}

/// One long-context fill and dispatch: the cache filled by `ranges` of device appends exactly as
/// the decoder appends (bf16 K/V, quantize-on-append, dense residual), its dequantized readback,
/// and the production reader's decode (`S_q = 1`, per-row split-KV) and multi-row (`S_q = 16`,
/// causal; NAX/tiled) outputs with the planned paths.
#[cfg(target_os = "macos")]
#[allow(clippy::type_complexity, clippy::arc_with_non_send_sync)]
fn long_context_dispatch(
    kv: &LongContextKv,
    bits: PackedCodeBits,
    ranges: &[(usize, usize)],
    queries: [&Array; 2],
) -> (Vec<f32>, Vec<f32>, [(Vec<f64>, String); 2]) {
    use crate::primitives::packed_metal::{PackedMask, PackedMetalGpuFamily, PackedMetalKernel};
    let mut cache = device_cache_bits(1, kv.kv_heads, kv.width, bits);
    for &(start, end) in ranges {
        cache
            .append_device(
                0,
                &rows_range(&kv.key_array, start, end).unwrap(),
                &rows_range(&kv.value_array, start, end).unwrap(),
            )
            .unwrap();
        // Bound the lazy graph exactly as the decoder's per-step evaluation does.
        cache.evaluate_device_store().unwrap();
    }
    assert_eq!(cache.logical_len(), kv.tokens);
    let kernel = || {
        PackedMetalKernel::for_identity_family_and_bits(
            "test-packed",
            PackedMetalGpuFamily::Apple7OrNewer,
            bits,
        )
        .unwrap()
    };
    cache
        .bind_compiled_handle(CompiledKernelHandle::new(Arc::new(kernel())))
        .unwrap();
    let kernel = kernel();
    let (resident, keys, values) = cache.evaluated_dense_layer(0).unwrap();
    assert_eq!(resident, kv.tokens);
    let outputs = queries.map(|query| {
        let staged = cache.staged_reader_arguments(0).unwrap();
        let path = format!(
            "{:?}",
            kernel
                .planned_path(&staged.args(query, PackedMask::Causal))
                .unwrap()
        );
        let output = cache
            .dispatch_packed(0, query, PackedMask::Causal)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let output = host_readback::<f32>(&output)
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>();
        (output, path)
    });
    mlx_rs::memory::clear_cache();
    (keys, values, outputs)
}

/// SC-20669 fit-boundary investigation: compressed attention error versus history length and
/// append-chunk count, per code width, at the Llama-3.2-3B head geometry (bf16, D = 128,
/// GQA 3). For each length the dense fp64 oracle runs over the exact bf16 K/V the dense arm
/// holds; for each width the fp64 oracle runs over the stored representation's dequantized
/// readback (which must be bit-identical for every chunking: a group straddling an append
/// boundary quantizes exactly once, from the same rows). Reports, for the decode (per-row
/// split-KV) and multi-row (NAX/tiled) readers: `kernel` (reader vs fp64 over the stored
/// representation — kernel numerics only), `quant` (fp64 over the representation vs fp64
/// dense — quantization only), `total` (reader vs fp64 dense), and the dense bf16 SDPA's own
/// `total` as the floor. Asserts that the kernel error does not grow with length or chunking
/// and that every width's quantization error is ordered (8 < 4 < 2). At 131072 tokens the
/// default `SC20669_LONG_CONTEXT_KV_HEADS=2` (GQA 3 preserved) peaks near 1.3 GiB RSS; the full
/// Llama geometry (`=8`) near 3.8 GiB RSS and 14 GiB physical footprint (MLX buffers).
#[cfg(target_os = "macos")]
#[test]
#[ignore = "long-context investigation sweep (131072-token histories, minutes in debug); run with --ignored --nocapture"]
fn long_context_compressed_attention_error_is_flat_in_length_and_chunk_count() {
    let kv_heads = std::env::var("SC20669_LONG_CONTEXT_KV_HEADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2usize);
    let lengths = std::env::var("SC20669_LONG_CONTEXT_LENGTHS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|length| length.trim().parse().unwrap())
                .collect::<Vec<usize>>()
        })
        .unwrap_or_else(|| vec![4096, 32768, 131072]);
    let tile_rows = 16;
    let mut kernel_errors: Vec<(PackedCodeBits, usize, f64, f64)> = Vec::new();
    eprintln!(
            "S_kv\tappends\tbits\tdecode path\tdecode kernel\tdecode quant\tdecode total\tdense bf16 decode\ttile path\ttile kernel\ttile quant\ttile total\tdense bf16 tile"
        );
    for &tokens in &lengths {
        let kv = long_context_kv(kv_heads, kv_heads * 3, tokens, 128);
        let (decode_query, decode_exact) = long_context_query(&kv, 1);
        let (tile_query, tile_exact) = long_context_query(&kv, tile_rows);
        let dense_decode = long_context_oracle(&kv, &decode_exact, 1, &kv.keys, &kv.values);
        let dense_tile = long_context_oracle(&kv, &tile_exact, tile_rows, &kv.keys, &kv.values);
        let scale = (kv.width as f32).powf(-0.5);
        let bf16_sdpa = |query: &Array, mask| {
            let output = crate::primitives::attention::sdpa(
                query,
                &kv.key_array,
                &kv.value_array,
                scale,
                mask,
            )
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
            host_readback::<f32>(&output)
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>()
        };
        let dense_floor = [
            relative_l2(
                &bf16_sdpa(&decode_query, crate::primitives::attention::AttnMask::None),
                &dense_decode,
            ),
            relative_l2(
                &bf16_sdpa(&tile_query, crate::primitives::attention::AttnMask::Causal),
                &dense_tile,
            ),
        ];
        for bits in PackedCodeBits::ALL {
            let mut representation: Option<(u64, Vec<f64>, Vec<f64>)> = None;
            for (chunk, decode_tail) in [(0, 0), (2048, 0), (2000, 40)] {
                let ranges = long_context_fill(tokens, chunk, decode_tail);
                let (keys, values, [(decode, decode_path), (tile, tile_path)]) =
                    long_context_dispatch(&kv, bits, &ranges, [&decode_query, &tile_query]);
                let digest = {
                    use std::hash::{Hash, Hasher};
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    for value in keys.iter().chain(&values) {
                        value.to_bits().hash(&mut hasher);
                    }
                    hasher.finish()
                };
                let (stored, oracle_decode, oracle_tile) =
                    representation.get_or_insert_with(|| {
                        let decode = long_context_oracle(&kv, &decode_exact, 1, &keys, &values);
                        let tile = long_context_oracle(&kv, &tile_exact, tile_rows, &keys, &values);
                        // Element-level representation error (relative RMS against the
                        // bf16 originals): roughly 16x per two bits removed.
                        let relative_rms = |stored: &[f32], original: &[f32]| {
                            let (error, norm) = stored.iter().zip(original).fold(
                                (0.0f64, 0.0f64),
                                |(error, norm), (stored, original)| {
                                    (
                                        error + f64::from(stored - original).powi(2),
                                        norm + f64::from(*original).powi(2),
                                    )
                                },
                            );
                            (error / norm).sqrt()
                        };
                        eprintln!(
                            "{tokens}\trepresentation\t{}\tK rel rms {:.3e}\tV rel rms {:.3e}",
                            bits.bits(),
                            relative_rms(&keys, &kv.keys),
                            relative_rms(&values, &kv.values)
                        );
                        (digest, decode, tile)
                    });
                assert!(
                        digest == *stored,
                        "{bits:?} S={tokens}: {} appends store a different representation than one append",
                        ranges.len()
                    );
                let decode_kernel = relative_l2(&decode, oracle_decode);
                let tile_kernel = relative_l2(&tile, oracle_tile);
                eprintln!(
                        "{tokens}\t{}\t{}\t{decode_path}\t{decode_kernel:.2e}\t{:.2e}\t{:.2e}\t{:.2e}\t{tile_path}\t{tile_kernel:.2e}\t{:.2e}\t{:.2e}\t{:.2e}",
                        ranges.len(),
                        bits.bits(),
                        relative_l2(oracle_decode, &dense_decode),
                        relative_l2(&decode, &dense_decode),
                        dense_floor[0],
                        relative_l2(oracle_tile, &dense_tile),
                        relative_l2(&tile, &dense_tile),
                        dense_floor[1],
                    );
                kernel_errors.push((bits, tokens, decode_kernel, tile_kernel));
            }
            let (_, oracle_decode, oracle_tile) = representation.unwrap();
            let quant = relative_l2(&oracle_decode, &dense_decode)
                .max(relative_l2(&oracle_tile, &dense_tile));
            kernel_errors.push((bits, tokens, -quant, -quant));
        }
    }
    // Kernel numerics: bf16-output-level at every length, chunking, and width (no growth with
    // S_kv or append count), and each width's quantization error is ordered.
    for &(bits, tokens, decode, tile) in kernel_errors.iter().filter(|entry| entry.2 >= 0.0) {
        assert!(
            decode <= 8e-3 && tile <= 8e-3,
            "{bits:?} S={tokens}: kernel relative L2 decode {decode:.3e} tile {tile:.3e}"
        );
    }
    for &tokens in &lengths {
        let quant = |bits| {
            -kernel_errors
                .iter()
                .find(|entry| entry.0 == bits && entry.1 == tokens && entry.2 < 0.0)
                .unwrap()
                .2
        };
        assert!(
            quant(PackedCodeBits::Eight) < quant(PackedCodeBits::Four)
                && quant(PackedCodeBits::Four) < quant(PackedCodeBits::Two),
            "S={tokens}: quantization error is not ordered by width"
        );
    }
}
