//! Candle-on-Metal reductions over rank > 4 tensors match the CPU backend.
//!
//! candle-core's Metal `reduce_op` moves the reduced dims to the end and, whenever the result is not
//! a contiguous trailing reduction of a contiguous tensor, launches candle-metal-kernels' `*_strided`
//! reduce kernels. Their `get_strided_index_t` (`reduce.metal`) decodes ranks 1–4 with a fixed-size
//! indexer and every larger rank through `get_strided_idx_fallback`, which at the pinned candle
//! revision peeled the outermost dims as if they were the innermost ones and read past the end of
//! the source buffer (upstream huggingface/candle#3873; provenance in
//! `crates/media/candle-gen/vendor/candle-metal-kernels/VENDORED.md`). Every reduce op (sum — and
//! so mean — min, max, argmin, argmax) is checked here at ranks 5 and 6 over every dim, a pair of
//! dims, and contiguous, permuted, narrowed and broadcast inputs.
//!
//! Metal-only: there is nothing to compare on a CPU-only build.

#![cfg(feature = "metal")]

use candle_gen::candle_core::{DType, Device, Result, Tensor};

/// Distinct small integers (exact in f32, so sums are exact and min/max/arg* have no ties): a
/// stride-coprime permutation of `0..n`, centred on zero.
fn distinct_values(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7919) % n) as f32 - (n / 2) as f32)
        .collect()
}

/// The ways an input reaches the reduce kernels: as built, and three non-contiguous views.
fn layouts(t: &Tensor) -> Result<Vec<(&'static str, Tensor)>> {
    let rank = t.rank();
    Ok(vec![
        ("contiguous", t.clone()),
        ("transposed", t.transpose(0, rank - 1)?),
        ("narrowed", t.narrow(rank - 2, 1, t.dim(rank - 2)? - 1)?),
        (
            "broadcast",
            t.narrow(1, 0, 1)?.broadcast_as({
                let mut dims = t.dims().to_vec();
                dims[1] = 3;
                dims
            })?,
        ),
    ])
}

/// One reduction, run on `x` (on whichever device it lives), read back to host f32.
fn reduce(op: &str, x: &Tensor, dims: &[usize]) -> Result<Vec<f32>> {
    let out = match (op, dims) {
        ("sum", _) => x.sum_keepdim(dims)?,
        ("mean", _) => x.mean_keepdim(dims)?,
        ("max", [d]) => x.max_keepdim(*d)?,
        ("min", [d]) => x.min_keepdim(*d)?,
        ("argmax", [d]) => x.argmax_keepdim(*d)?,
        ("argmin", [d]) => x.argmin_keepdim(*d)?,
        _ => unreachable!("{op} over {dims:?} is not a case"),
    };
    out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

#[test]
fn rank_5_and_6_reductions_on_metal_match_the_cpu() -> Result<()> {
    let metal = Device::new_metal(0)?;
    let mut checked = 0usize;
    let mut failures = Vec::new();
    for shape in [vec![2usize, 3, 4, 5, 6], vec![2, 3, 2, 4, 3, 5]] {
        let n = shape.iter().product();
        let cpu = Tensor::from_vec(distinct_values(n), shape.as_slice(), &Device::Cpu)?;
        let gpu = cpu.to_device(&metal)?;
        for ((layout, c), (_, g)) in layouts(&cpu)?.into_iter().zip(layouts(&gpu)?) {
            let rank = c.rank();
            // Every single dim (dim 0 and the trailing dim included), plus two-dim sums that
            // straddle the fallback / fixed-size indexer boundary.
            let mut cases: Vec<(&str, Vec<usize>)> = Vec::new();
            for d in 0..rank {
                for op in ["sum", "mean", "max", "min", "argmax", "argmin"] {
                    cases.push((op, vec![d]));
                }
            }
            for pair in [vec![0, 2], vec![1, rank - 2], vec![0, rank - 1]] {
                cases.push(("sum", pair.clone()));
                cases.push(("mean", pair));
            }
            for (op, dims) in cases {
                let want = reduce(op, &c, &dims)?;
                let got = reduce(op, &g, &dims)?;
                checked += 1;
                let worst = got
                    .iter()
                    .zip(&want)
                    .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                if got.len() != want.len() || !worst.is_finite() || worst > 1e-4 {
                    failures.push(format!(
                        "{shape:?} {layout} {op}{dims:?}: max|metal-cpu|={worst:e}"
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {checked} Metal reductions disagree with the CPU:\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}
