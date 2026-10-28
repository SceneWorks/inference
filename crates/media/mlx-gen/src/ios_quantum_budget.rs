//! iOS quantum accounting primitives. Every figure is COUNTED from the ops of one quantum — how
//! many tensors of each shape are live at that quantum's peak — by its call site, and each call
//! site proves its counts with a CPU test that measures MLX's peak for a small shape
//! ([`measured_peak_bytes`], the stalled measurement) through [`assert_tracks_measured`], which
//! splits the proof in two:
//!
//! - **formula ≥ measured is a HARD failure** — the safety bound: a guard priced below what its
//!   quantum allocates admits a jetsam kill. Load only ever LOWERS a stalled CPU measurement (the
//!   CPU's in-flight window retains less of the graph when the worker keeps up), so this side
//!   never flakes.
//! - **formula ≤ 1.5× measured is REPORT-ONLY** — a ratio above it prints a greppable
//!   `PRICING-RATIO-HIGH <label> <ratio>` line and the test passes. The CPU measurement cannot
//!   observe the device's wider retention (up to ten in-flight command buffers of up to 20 ops /
//!   40 MB each on a phone) and swings with host load, so it cannot bound the high side; the
//!   LOW-END ACCURACY of every formula is calibrated against the device instead — the on-device
//!   anchor captures and the validation case (docs/ios-device-runbook.md, "Campaign protocol").
//!   There are no per-proof allowances: every ratio is printed and reported against the same 1.5.
//!
//! No blanket multipliers. These remain floors for one materialized quantum, not a process or
//! Metal-driver bound, and they never authorize production admission.

pub const PROCESS_RESERVE_BYTES: u64 = 512 * 1024 * 1024;

fn product(values: &[u64]) -> u64 {
    values.iter().fold(1u64, |a, b| a.saturating_mul(*b))
}
fn sum(values: &[u64]) -> u64 {
    values.iter().fold(0u64, |a, b| a.saturating_add(*b))
}

/// Bytes of one tensor of `dims` at `element_bytes` per element.
pub fn tensor_bytes(element_bytes: u64, dims: &[u64]) -> u64 {
    product(&[product(dims), element_bytes])
}

/// The allocator's page: MLX's Metal allocator (`backend/metal/allocator.cpp`, `malloc`) rounds
/// every allocation larger than `vm_page_size` (16 KiB on Apple Silicon and iOS) up to whole pages.
pub const ALLOCATOR_PAGE_BYTES: u64 = 16 * 1024;

/// Bytes ONE allocation of `bytes` actually occupies: page-rounded above one page, exact below.
pub fn allocated_bytes(bytes: u64) -> u64 {
    if bytes > ALLOCATOR_PAGE_BYTES {
        bytes
            .div_ceil(ALLOCATOR_PAGE_BYTES)
            .saturating_mul(ALLOCATOR_PAGE_BYTES)
    } else {
        bytes
    }
}

/// [`tensor_bytes`] as the allocator holds it: one page-rounded allocation.
pub fn allocated_tensor_bytes(element_bytes: u64, dims: &[u64]) -> u64 {
    allocated_bytes(tensor_bytes(element_bytes, dims))
}

/// How an attention's scores exist in memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scores {
    /// Fused SDPA (`mlx_rs::fast::scaled_dot_product_attention`): no score tensor materializes;
    /// only q/k/v/out, which the call site counts among its live hidden/context buffers.
    Fused,
    /// Scores materialized as `[B, H, query_rows, keys]` at `element_bytes`, `live` at once (e.g.
    /// the logits and their softmax), by an explicit-softmax or query-chunked path.
    Materialized { live: u64, element_bytes: u64 },
}

/// Score bytes of one attention over `keys` keys, `query_rows` queries at a time.
pub fn scores_bytes(scores: Scores, batch: u64, heads: u64, query_rows: u64, keys: u64) -> u64 {
    match scores {
        Scores::Fused => 0,
        Scores::Materialized {
            live,
            element_bytes,
        } => product(&[live, batch, heads, query_rows, keys, element_bytes]),
    }
}

/// One attention block (DiT, UNet spatial transformer, text-encoder layer, VAE mid-attention) from
/// loaded tensors, the request, and the call site's COUNTED live tensors at each phase's peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionBlockGeometry {
    pub batch: u64,
    /// Tokens attended over (image/latent tokens, or text tokens for a text layer).
    pub tokens: u64,
    /// Block width.
    pub dim: u64,
    pub heads: u64,
    /// Cross-attention context tokens (0 for self-attention only).
    pub context_tokens: u64,
    /// MLP hidden width (0 for an attention-only quantum).
    pub ff_hidden: u64,
    /// Query rows scored at once: `tokens` unchunked, the chunk under a bounded plan.
    pub query_rows: u64,
    /// Activation storage width in bytes (2 for f16/bf16 compute, 4 for f32).
    pub element_bytes: u64,
    /// Bytes of weight COPIES this quantum creates (dequantize/promote/cast); 0 when the resident
    /// weights are consumed in place.
    pub weight_bytes: u64,
    /// Live `[B, tokens, dim]` buffers at the attention phase's peak (residual, normed input,
    /// q/k/v/out as counted for this site).
    pub attn_hidden_live: u64,
    /// Live `[B, context_tokens, dim]` buffers at the attention phase's peak (cross k/v, ...).
    pub context_live: u64,
    pub scores: Scores,
    /// Live `[B, tokens, dim]` buffers at the MLP phase's peak.
    pub mlp_hidden_live: u64,
    /// Live `[B, tokens, ff_hidden]` buffers at the MLP phase's peak.
    pub ff_live: u64,
}

/// `weight_bytes + max(attention phase, MLP phase)`: the phases run one after the other, so their
/// intermediates are not live together.
pub fn attention_block_bytes(g: AttentionBlockGeometry) -> u64 {
    let hidden = tensor_bytes(g.element_bytes, &[g.batch, g.tokens, g.dim]);
    let attention = sum(&[
        product(&[g.attn_hidden_live, hidden]),
        product(&[
            g.context_live,
            tensor_bytes(g.element_bytes, &[g.batch, g.context_tokens, g.dim]),
        ]),
        scores_bytes(
            g.scores,
            g.batch,
            g.heads,
            g.query_rows,
            g.tokens.saturating_add(g.context_tokens),
        ),
    ]);
    let mlp = sum(&[
        product(&[g.mlp_hidden_live, hidden]),
        product(&[
            g.ff_live,
            tensor_bytes(g.element_bytes, &[g.batch, g.tokens, g.ff_hidden]),
        ]),
    ]);
    g.weight_bytes.saturating_add(attention.max(mlp))
}

/// One convolutional stage (resnet, up/downsample, conv in/out) at its spatial extent, with the
/// call site's COUNTED live input-width and output-width activations at its peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvStageGeometry {
    pub batch: u64,
    pub height: u64,
    pub width: u64,
    pub in_channels: u64,
    pub out_channels: u64,
    pub element_bytes: u64,
    /// Live `[B, H, W, in_channels]` buffers at the peak.
    pub in_live: u64,
    /// Live `[B, H, W, out_channels]` buffers at the peak.
    pub out_live: u64,
    /// Bytes of weight copies this quantum creates (promotion/cast); 0 when used in place.
    pub weight_bytes: u64,
}

pub fn conv_stage_bytes(g: ConvStageGeometry) -> u64 {
    let plane = product(&[g.batch, g.height, g.width, g.element_bytes]);
    sum(&[
        g.weight_bytes,
        product(&[g.in_live, plane, g.in_channels]),
        product(&[g.out_live, plane, g.out_channels]),
    ])
}

/// MLX's peak allocation while `run` builds and evaluates one quantum, above what was live before
/// it (inputs already materialized do not count; the outputs do). For host CPU proofs of a call
/// site's counted formula; the device enforces only the formula.
///
/// Measured STALLED: a long kernel (`STALL_EDGE`) is queued on the default stream first, so the
/// quantum's ops are encoded — their outputs allocated on the encoding thread, their operands held
/// by the queued tasks (so never donated) — before the first of them runs. That is the retention
/// worst case MLX's scheduler allows (it caps the work in flight, on the CPU at ten tasks, and the
/// device holds a command buffer's operands until it completes), and it is what every counted
/// price follows. Unstalled, the peak depends on how far THIS host's worker happens to lag the
/// encoder: the CI runner measured the same Anima output quantum at 68 KB in one run and 135 KB in
/// another (this Mac: 134 KB every time), so an unstalled measurement is a property of the
/// machine's load, not of the quantum.
#[cfg(not(target_os = "ios"))]
pub fn measured_peak_bytes(
    run: impl FnOnce() -> crate::Result<Vec<mlx_rs::Array>>,
) -> crate::Result<u64> {
    mlx_rs::memory::clear_cache();
    let stall = stall_kernel()?;
    let before = mlx_rs::memory::get_active_memory() as u64;
    mlx_rs::memory::reset_peak_memory();
    let outputs = run()?;
    mlx_rs::transforms::eval(outputs.iter())?;
    let peak = mlx_rs::memory::get_peak_memory() as u64;
    drop(outputs);
    mlx_rs::transforms::eval([&stall])?;
    drop(stall);
    Ok(peak.saturating_sub(before))
}

/// The stall kernel's edge: one `[E, E]` f32 product (~137 GFLOP at 4096; its one output is
/// allocated before the measurement starts) keeps the stream's worker busy far longer than encoding
/// any proof quantum's graph up to the scheduler's in-flight cap takes.
#[cfg(not(target_os = "ios"))]
const STALL_EDGE: i32 = 4096;

/// Queue `STALL_EDGE`'s product on the default stream and hand it back unevaluated-but-queued.
#[cfg(not(target_os = "ios"))]
fn stall_kernel() -> crate::Result<mlx_rs::Array> {
    thread_local! {
        static OPERAND: std::cell::RefCell<Option<mlx_rs::Array>> = const { std::cell::RefCell::new(None) };
    }
    let operand = OPERAND.with(|cell| -> crate::Result<mlx_rs::Array> {
        let mut cell = cell.borrow_mut();
        if cell.is_none() {
            let a = mlx_rs::ops::ones::<f32>(&[STALL_EDGE, STALL_EDGE])?;
            mlx_rs::transforms::eval([&a])?;
            *cell = Some(a);
        }
        Ok(cell.as_ref().expect("set above").clone())
    })?;
    let product = operand.matmul(&operand)?;
    mlx_rs::transforms::async_eval([&product])?;
    Ok(product)
}

/// The proof every counted formula carries (see the module doc for the split):
///
/// - **HARD**: the formula is never below the measured peak — it must not admit a quantum that
///   allocates more. Panics with both figures otherwise.
/// - **REPORT-ONLY**: above `max_ratio` times the measurement (1.5 everywhere) it writes
///   `PRICING-RATIO-HIGH <label> <ratio>` straight to the process's stderr (not through the test
///   harness's captured output, so a passing test still shows it) and returns
///   [`RatioReport::High`]. Low-end accuracy is calibrated on the device, not by this CPU bound.
pub fn assert_tracks_measured(
    label: &str,
    formula: u64,
    measured: u64,
    max_ratio: f64,
) -> RatioReport {
    assert!(
        formula >= measured,
        "{label}: formula {formula} B is BELOW the measured peak {measured} B"
    );
    let ratio = formula as f64 / measured.max(1) as f64;
    if ratio > max_ratio {
        use std::io::Write;
        let _ = writeln!(
            std::io::stderr(),
            "PRICING-RATIO-HIGH {label} {ratio:.3} (formula {formula} B, measured {measured} B, \
             report threshold {max_ratio}x)"
        );
        RatioReport::High(ratio)
    } else {
        RatioReport::Within(ratio)
    }
}

/// What [`assert_tracks_measured`] found on the report-only side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RatioReport {
    /// formula / measured, at or under the report threshold.
    Within(f64),
    /// formula / measured, over the report threshold (a `PRICING-RATIO-HIGH` line was written).
    High(f64),
}

/// Latent-sized elementwise work BETWEEN denoiser forwards (classifier-free-guidance combine and
/// the sampler update). It is lazy, so it materializes inside the NEXT guarded quantum, whose
/// scratch carries this allowance: CFG combine holds both branch predictions, their difference,
/// the guidance-scaled term and the sum (5 buffers); the sampler update holds the previous sample,
/// the model output, the scaled step, the noise and the new sample (5); plus up to 6 dtype-cast
/// copies = 16 f32 latent-sized buffers.
pub fn sampler_step_bytes(latent_elements: u64) -> u64 {
    product(&[16, latent_elements, 4])
}

/// Rotary tables built for `positions` at `head_dim`: f32 cos and sin plus their broadcast copies.
pub fn rope_table_bytes(positions: u64, head_dim: u64) -> u64 {
    product(&[4, positions, head_dim, 4])
}

/// `copies` f32 buffers of `elements` each: a small elementwise pass (mask multiply, positional
/// add, concatenation) folded into the guarded quantum that materializes it.
pub fn elementwise_bytes(elements: u64, copies: u64) -> u64 {
    product(&[copies, elements, 4])
}

pub fn check_headroom(
    available: Option<u64>,
    scratch: u64,
    reserve: u64,
) -> Result<(), &'static str> {
    let required = scratch
        .checked_add(reserve)
        .ok_or("iOS quantum accounting overflow")?;
    match available {
        Some(bytes) if bytes >= required => Ok(()),
        Some(_) => Err("iOS quantum refused before construction: insufficient current OS headroom"),
        None => Err("iOS quantum refused before construction: OS headroom unavailable"),
    }
}
pub fn guarded<T, E: From<&'static str>>(
    available: Option<u64>,
    scratch: u64,
    reserve: u64,
    build: impl FnOnce() -> Result<T, E>,
    materialize: impl FnOnce(&T) -> Result<(), E>,
    release: impl FnOnce(),
) -> Result<T, E> {
    check_headroom(available, scratch, reserve).map_err(E::from)?;
    let value = build()?;
    materialize(&value)?;
    release();
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn unavailable_insufficient_and_overflow_refuse_before_construction() {
        for available in [None, Some(0), Some(99)] {
            let constructed = Cell::new(false);
            let result = guarded::<(), &'static str>(
                available,
                64,
                36,
                || {
                    constructed.set(true);
                    Ok(())
                },
                |_| Ok(()),
                || {},
            );
            assert!(result.is_err());
            assert!(!constructed.get());
        }
        assert!(check_headroom(Some(u64::MAX), u64::MAX, 1).is_err());
        assert!(check_headroom(Some(100), 64, 36).is_ok());
    }

    #[test]
    fn layer_barrier_and_release_precede_the_next_layer() {
        let lazy_graph_depth = Cell::new(0);
        let releases = Cell::new(0);
        for _ in 0..26 {
            guarded::<(), &'static str>(
                Some(100),
                64,
                36,
                || {
                    lazy_graph_depth.set(lazy_graph_depth.get() + 1);
                    Ok(())
                },
                |_| {
                    assert_eq!(lazy_graph_depth.get(), 1);
                    lazy_graph_depth.set(0);
                    Ok(())
                },
                || {
                    assert_eq!(lazy_graph_depth.get(), 0);
                    releases.set(releases.get() + 1);
                },
            )
            .unwrap();
        }
        assert_eq!(releases.get(), 26);
        assert_eq!(lazy_graph_depth.get(), 0);
    }

    #[test]
    fn failed_materialization_never_returns_an_unevaluated_result() {
        let result = guarded::<(), &'static str>(
            Some(100),
            64,
            36,
            || Ok(()),
            |_| Err("eval failed"),
            || {},
        );
        assert_eq!(result, Err("eval failed"));
    }

    #[test]
    fn counted_attention_takes_the_larger_phase_and_fused_scores_cost_nothing() {
        let g = AttentionBlockGeometry {
            batch: 1,
            tokens: 64,
            dim: 32,
            heads: 4,
            context_tokens: 7,
            ff_hidden: 96,
            query_rows: 64,
            element_bytes: 2,
            weight_bytes: 1000,
            attn_hidden_live: 5,
            context_live: 2,
            scores: Scores::Fused,
            mlp_hidden_live: 2,
            ff_live: 2,
        };
        let attn = 5 * 64 * 32 * 2 + 2 * 7 * 32 * 2;
        let mlp = 2 * 64 * 32 * 2 + 2 * 64 * 96 * 2;
        assert_eq!(attention_block_bytes(g), 1000 + attn.max(mlp));
        let materialized = AttentionBlockGeometry {
            scores: Scores::Materialized {
                live: 2,
                element_bytes: 4,
            },
            query_rows: 16,
            ..g
        };
        assert_eq!(
            attention_block_bytes(materialized),
            1000 + (attn + 2 * 4 * 16 * 71 * 4).max(mlp)
        );
        assert_eq!(
            attention_block_bytes(AttentionBlockGeometry {
                batch: u64::MAX,
                ..g
            }),
            u64::MAX
        );
        let conv = ConvStageGeometry {
            batch: 1,
            height: 8,
            width: 8,
            in_channels: 4,
            out_channels: 6,
            element_bytes: 4,
            in_live: 2,
            out_live: 1,
            weight_bytes: 10,
        };
        assert_eq!(conv_stage_bytes(conv), 10 + 2 * 64 * 4 * 4 + 64 * 6 * 4);
    }

    #[test]
    fn allocations_above_a_page_round_to_whole_pages() {
        assert_eq!(allocated_bytes(100), 100);
        assert_eq!(allocated_bytes(16 * 1024), 16 * 1024);
        assert_eq!(allocated_bytes(16 * 1024 + 1), 32 * 1024);
        assert_eq!(allocated_tensor_bytes(4, &[4097]), 32 * 1024);
        assert_eq!(allocated_bytes(u64::MAX), u64::MAX);
    }

    /// The high side is report-only: ten times the measurement passes and reports.
    #[test]
    fn a_formula_ten_times_the_measurement_is_reported_not_failed() {
        assert_eq!(
            assert_tracks_measured("x", 1000, 100, 1.5),
            RatioReport::High(10.0)
        );
        assert_eq!(
            assert_tracks_measured("x", 150, 100, 1.5),
            RatioReport::Within(1.5)
        );
    }

    /// The low side stays a HARD failure: one byte under the measurement panics.
    #[test]
    #[should_panic(expected = "BELOW")]
    fn a_formula_under_the_measurement_fails_its_proof() {
        let _ = assert_tracks_measured("x", 99, 100, 1.5);
    }

    #[test]
    fn small_step_allowances_are_shape_derived_and_saturating() {
        // 1x4x64x64 latent: 16 f32 buffers.
        assert_eq!(sampler_step_bytes(4 * 64 * 64), 16 * 4 * 64 * 64 * 4);
        assert_eq!(rope_table_bytes(4096, 128), 4 * 4096 * 128 * 4);
        assert_eq!(elementwise_bytes(1000, 3), 12_000);
        assert_eq!(sampler_step_bytes(u64::MAX), u64::MAX);
    }
}
