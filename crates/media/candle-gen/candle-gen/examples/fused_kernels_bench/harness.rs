//! `fused_kernels_bench`'s harness: timing, peak device memory, the device label and the table.
//! Kernel modules only build inputs and hand [`Bench::compare`] a reference and a fused closure.

use std::time::Instant;

use candle_gen::candle_core::{Device, Result};
use candle_gen::cuda_mempool::MemPool;

/// A kernel module's entry point: its rows, one per shape.
pub type Suite = fn(&Bench) -> Result<Vec<Row>>;

/// One measured path.
#[derive(Clone, Copy, Debug)]
pub struct Measure {
    /// Wall milliseconds per iteration.
    pub ms: f64,
    /// Transient device-memory high-water above the pre-call live bytes, MiB.
    pub peak_mib: f64,
}

/// One table row: a kernel at one shape, reference vs fused.
#[derive(Clone, Debug)]
pub struct Row {
    pub kernel: String,
    pub shape: String,
    pub reference: Measure,
    pub fused: Measure,
}

/// The bench context every suite shares.
pub struct Bench {
    /// The CUDA device the rows run on.
    pub dev: Device,
    /// `sm_XY` of that device.
    pub device_label: String,
    warmup: usize,
    iters: usize,
    pool: MemPool,
}

fn env_count(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .filter(|n: &usize| *n > 0)
        .unwrap_or(default)
}

impl Bench {
    /// Open CUDA device `ordinal` and its current memory pool.
    pub fn new(ordinal: usize) -> Result<Self> {
        let dev = Device::new_cuda(ordinal)?;
        let cap = candle_gen::cuda_arch::device_compute_cap(&dev).ok_or_else(|| {
            candle_gen::candle_core::Error::Msg("cannot read the compute capability".into())
        })?;
        let pool = MemPool::device_current(ordinal as i32).ok_or_else(|| {
            candle_gen::candle_core::Error::Msg("no current CUDA memory pool".into())
        })?;
        // A bench row's "fused" column must time the fused kernel: override an inherited
        // `CANDLE_MEDIA_FUSED_KERNELS=0`. [`Bench::compare`] still verifies every fused window
        // (a per-kernel switch or a cached compile failure would fall back).
        candle_gen::fused_path::set_media_fused_kernels(Some(true));
        Ok(Self {
            dev,
            device_label: format!("sm_{cap}"),
            warmup: env_count("FUSED_BENCH_WARMUP", 3),
            iters: env_count("FUSED_BENCH_ITERS", 20),
            pool,
        })
    }

    /// Time `f` (after warm-up) and read its transient peak.
    pub fn measure(&self, f: &mut dyn FnMut() -> Result<()>) -> Result<Measure> {
        for _ in 0..self.warmup {
            f()?;
        }
        self.dev.synchronize()?;
        let baseline = self.pool.used().expect("pool USED_MEM_CURRENT");
        assert!(self.pool.reset_used_high_water(), "reset USED_MEM_HIGH");
        let start = Instant::now();
        for _ in 0..self.iters {
            f()?;
        }
        self.dev.synchronize()?;
        let ms = start.elapsed().as_secs_f64() * 1e3 / self.iters as f64;
        let high = self.pool.used_high().expect("pool USED_MEM_HIGH");
        Ok(Measure {
            ms,
            peak_mib: high.saturating_sub(baseline) as f64 / (1024.0 * 1024.0),
        })
    }

    /// One row: `reference` and `fused` measured back to back on the same inputs.
    ///
    /// The fused window is verified on the media fused-path counters: it must record fused
    /// calls and no reference call, or the row is refused with the counters it saw, so a fallback
    /// (a switch turned off, a cached compile failure, unsupported input) is never published as
    /// a fused timing.
    pub fn compare(
        &self,
        kernel: &str,
        shape: &str,
        reference: impl FnMut() -> Result<()>,
        fused: impl FnMut() -> Result<()>,
    ) -> Result<Row> {
        self.compare_with(
            kernel,
            shape,
            path_totals,
            |column, before, after| {
                if column == Column::Reference {
                    return Ok(());
                }
                let (fused_calls, reference_calls) = (after.0 - before.0, after.1 - before.1);
                if fused_calls == 0 || reference_calls != 0 {
                    return Err(format!(
                        "the fused column did not run only the fused kernel                          ({fused_calls} fused, {reference_calls} reference calls): {:?}",
                        candle_gen::fused_path::fused_path_counts()
                    ));
                }
                Ok(())
            },
            reference,
            fused,
        )
    }

    /// [`Bench::compare`] for a kernel whose path telemetry is not the media fused-path counters
    /// (the attention kernels count in `candle_gen::fused_attention::AttentionPathCounts`).
    ///
    /// `snapshot` reads the kernel's counters; `verify(column, before, after)` checks each
    /// measured window (warm-up included) and returns why it did not run only the intended path.
    /// A refusal fails the row with that reason, so a fallback is never published as a timing.
    pub fn compare_with<S>(
        &self,
        kernel: &str,
        shape: &str,
        snapshot: impl Fn() -> S,
        verify: impl Fn(Column, &S, &S) -> std::result::Result<(), String>,
        mut reference: impl FnMut() -> Result<()>,
        mut fused: impl FnMut() -> Result<()>,
    ) -> Result<Row> {
        let window = |column: Column, f: &mut dyn FnMut() -> Result<()>| -> Result<Measure> {
            let before = snapshot();
            let measure = self.measure(f)?;
            let after = snapshot();
            verify(column, &before, &after).map_err(|why| {
                candle_gen::candle_core::Error::Msg(format!("{kernel} {shape} {column:?}: {why}"))
            })?;
            Ok(measure)
        };
        let reference = window(Column::Reference, &mut reference)?;
        let fused = window(Column::Fused, &mut fused)?;
        Ok(Row {
            kernel: kernel.to_string(),
            shape: shape.to_string(),
            reference,
            fused,
        })
    }
}

/// The column of a row a [`Bench::compare_with`] verifier is checking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Column {
    Reference,
    Fused,
}

/// Process-wide (fused, reference) call totals over every media fused-path kernel.
fn path_totals() -> (u64, u64) {
    candle_gen::fused_path::fused_path_counts()
        .iter()
        .fold((0, 0), |(f, r), (_, c)| {
            (f + c.fused, r + c.reference_total())
        })
}

/// Print every row as one Markdown table.
pub fn print_table(bench: &Bench, rows: &[Row]) {
    println!("| kernel | shape | ref ms | fused ms | ref peak MiB | fused peak MiB | device |");
    println!("|---|---|---:|---:|---:|---:|---|");
    for r in rows {
        println!(
            "| {} | {} | {:.4} | {:.4} | {:.1} | {:.1} | {} |",
            r.kernel,
            r.shape,
            r.reference.ms,
            r.fused.ms,
            r.reference.peak_mib,
            r.fused.peak_mib,
            bench.device_label
        );
    }
    println!(
        "({} timed iterations after {} warm-up per cell)",
        bench.iters, bench.warmup
    );
}
