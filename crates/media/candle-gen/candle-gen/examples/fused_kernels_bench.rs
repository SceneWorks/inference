//! The shared fused-kernel bench (feature SW-F287, created by SW-6126): every fused CUDA kernel's
//! reference op chain against its fused launch, at representative shapes, in **one table**:
//!
//! ```text
//! kernel | shape | ref ms | fused ms | ref peak MiB | fused peak MiB | device
//! ```
//!
//! `ms` is wall time per iteration around a device synchronize; `peak MiB` is the transient
//! device-memory high-water above the live baseline, read from the CUDA stream-ordered pool
//! (`CU_MEMPOOL_ATTR_USED_MEM_HIGH`, [`candle_gen::cuda_mempool`]) — the memory one call allocates
//! on top of its inputs. `device` is the GPU's compute capability (`sm_120` on the dev box). Numbers
//! are per architecture: never extrapolate an `sm_120` row to `sm_89` (epic E11).
//!
//! ```text
//! (vcvars) cargo run --release -p candle-gen --features cuda --example fused_kernels_bench
//! (vcvars) cargo run --release -p candle-gen --features cuda --example fused_kernels_bench -- act_quant
//! ```
//!
//! The optional arguments name the suites to run (all by default). `FUSED_BENCH_ITERS` /
//! `FUSED_BENCH_WARMUP` override the 20 timed / 3 warm-up iterations.
//!
//! # Adding a kernel's rows
//!
//! Add one module under `examples/fused_kernels_bench/` exposing
//! `pub fn rows(bench: &Bench) -> Result<Vec<Row>>` (build inputs once, then one
//! [`Bench::compare`] per shape with a reference closure and a fused closure), and one line in
//! [`SUITES`]. The harness owns timing, memory, the device label and the table.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
#[path = "fused_kernels_bench/harness.rs"]
mod harness;

#[cfg(feature = "cuda")]
#[path = "fused_kernels_bench/act_quant.rs"]
mod act_quant;
#[cfg(feature = "cuda")]
#[path = "fused_kernels_bench/adaln.rs"]
mod adaln;
#[cfg(feature = "cuda")]
#[path = "fused_kernels_bench/rope.rs"]
mod rope;

/// Every suite: `(name, rows)`. One line per kernel module.
#[cfg(feature = "cuda")]
const SUITES: &[(&str, harness::Suite)] = &[
    ("act_quant", act_quant::rows),
    ("adaln", adaln::rows),
    ("rope", rope::rows),
];

#[cfg(feature = "cuda")]
fn main() -> candle_gen::candle_core::Result<()> {
    let wanted: Vec<String> = std::env::args().skip(1).collect();
    let bench = harness::Bench::new(0)?;
    let mut rows = Vec::new();
    for (name, suite) in SUITES {
        if wanted.is_empty() || wanted.iter().any(|w| w == name) {
            eprintln!("running {name} ...");
            rows.extend(suite(&bench)?);
        }
    }
    harness::print_table(&bench, &rows);
    Ok(())
}
