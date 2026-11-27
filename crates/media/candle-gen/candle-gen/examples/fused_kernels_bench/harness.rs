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
    pub fn compare(
        &self,
        kernel: &str,
        shape: &str,
        mut reference: impl FnMut() -> Result<()>,
        mut fused: impl FnMut() -> Result<()>,
    ) -> Result<Row> {
        let reference = self.measure(&mut reference)?;
        let fused = self.measure(&mut fused)?;
        Ok(Row {
            kernel: kernel.to_string(),
            shape: shape.to_string(),
            reference,
            fused,
        })
    }
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
