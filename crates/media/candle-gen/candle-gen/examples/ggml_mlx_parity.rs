//! Tensor-level parity of a Candle-native GGML component against the MLX-affine component it was
//! built from (SW-7285). CPU only, no forward pass. NOT part of CI: it reads real weights.
//!
//! ```text
//! cargo run --release --example ggml_mlx_parity -p candle-gen -- \
//!   --ggml <GGML component dir, e.g. out/<family>-candle/q4/transformer> \
//!   --mlx  <MLX component dir,  e.g. <mlx snapshot>/q4/transformer> [--group-size 64]
//! ```
//!
//! For every packed MLX Q4 projection (`{base}.weight` U32 + `.scales` + `.biases`):
//!
//! * **block bytes** -- the stored GGML `Q4_1` blocks are compared byte-for-byte with the blocks the
//!   production Windows path builds at load (`repack_mlx_q4_to_q4_1_gs`, the runtime repack). Equal
//!   bytes mean the two paths feed the matmul identical weights.
//! * **values** -- both dequantize to f32 and are compared with the exact MLX affine values
//!   (`code * scale + bias` in f32 from the bf16 `scales`/`biases`), reporting how far the f16
//!   storage of a `Q4_1` block's `d`/`m` moves any weight (zero when every scale/bias is
//!   f16-exact).
//!
//! Every other MLX tensor must appear in the GGML component with the same dtype, shape and bytes, and
//! the GGML component must carry nothing extra. Exits non-zero on any block-byte or dense mismatch.
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_layers::repack::repack_mlx_q4_to_q4_1_gs;
use candle_gen::candle_layers::stored_ggml::qtensor_from_stored;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

/// Every `*.safetensors` file directly under `dir`, sorted.
fn shards(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!("no .safetensors under {}", dir.display()).into());
    }
    Ok(files)
}

fn open(dir: &Path) -> Result<MmapedSafetensors> {
    // SAFETY: read-only mapping of immutable tier files for the duration of the comparison.
    Ok(unsafe { MmapedSafetensors::multi(&shards(dir)?)? })
}

fn names(st: &MmapedSafetensors) -> BTreeSet<String> {
    st.tensors().into_iter().map(|(name, _)| name).collect()
}

/// The exact MLX affine values of a packed Q4 triple, in f32: nibble `k` of each U32 word is the
/// code of column `8·word + k`, and one bf16 `scale`/`bias` pair covers `group_size` columns.
fn mlx_affine_f32(wq: &Tensor, scales: &Tensor, biases: &Tensor, group: usize) -> Result<Vec<f32>> {
    let (out_dim, words) = wq.dims2()?;
    let in_dim = words * 8;
    let wq = wq.flatten_all()?.to_vec1::<u32>()?;
    let scales = scales
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let biases = biases
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let groups = in_dim / group;
    let mut values = Vec::with_capacity(out_dim * in_dim);
    for row in 0..out_dim {
        for col in 0..in_dim {
            let word = wq[row * words + col / 8];
            let code = ((word >> (4 * (col % 8))) & 0xF) as f32;
            let g = row * groups + col / group;
            values.push(code * scales[g] + biases[g]);
        }
    }
    Ok(values)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let ggml_dir = PathBuf::from(arg(&args, "--ggml").ok_or("--ggml <component dir>")?);
    let mlx_dir = PathBuf::from(arg(&args, "--mlx").ok_or("--mlx <component dir>")?);
    let group: usize = arg(&args, "--group-size").map_or(Ok(64), |g| g.parse())?;
    let cpu = Device::Cpu;

    let ggml = open(&ggml_dir)?;
    let mlx = open(&mlx_dir)?;
    let ggml_names = names(&ggml);
    let mlx_names = names(&mlx);

    let mut seen = BTreeSet::new();
    let (mut packed, mut packed_byte_equal, mut dense, mut dense_equal) = (0, 0, 0, 0);
    let (mut value_diff_weights, mut value_weights) = (0usize, 0usize);
    let mut max_abs = 0f32;
    let mut failures = Vec::new();

    for name in &mlx_names {
        if let Some(base) = name.strip_suffix(".scales") {
            let weight = format!("{base}.weight");
            let wq = mlx.load(&weight, &cpu)?;
            let scales = mlx.load(name, &cpu)?;
            let biases = mlx.load(&format!("{base}.biases"), &cpu)?;
            seen.extend([weight.clone(), name.clone(), format!("{base}.biases")]);
            packed += 1;
            if !ggml_names.contains(&weight) {
                failures.push(format!("{base}: missing from the GGML component"));
                continue;
            }
            let stored = ggml.load(&weight, &cpu)?;
            let stored_q = qtensor_from_stored(&stored, &cpu)?;
            let runtime_q = repack_mlx_q4_to_q4_1_gs(&wq, &scales, &biases, group, &cpu)?;
            if stored_q.dtype() != runtime_q.dtype() || stored_q.shape() != runtime_q.shape() {
                failures.push(format!(
                    "{base}: stored {:?}{:?} vs runtime repack {:?}{:?}",
                    stored_q.dtype(),
                    stored_q.shape(),
                    runtime_q.dtype(),
                    runtime_q.shape()
                ));
                continue;
            }
            if stored_q.data()? == runtime_q.data()? {
                packed_byte_equal += 1;
            } else {
                failures.push(format!(
                    "{base}: stored blocks differ from the runtime repack"
                ));
            }
            let exact = mlx_affine_f32(&wq, &scales, &biases, group)?;
            let got = stored_q.dequantize(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
            value_weights += exact.len();
            for (a, b) in exact.iter().zip(&got) {
                if a.to_bits() != b.to_bits() {
                    value_diff_weights += 1;
                    max_abs = max_abs.max((a - b).abs());
                }
            }
        } else if name.ends_with(".biases")
            || (name.ends_with(".weight")
                && mlx_names.contains(&format!("{}.scales", name.trim_end_matches(".weight"))))
        {
            continue; // part of a packed triple, handled above
        } else {
            seen.insert(name.clone());
            dense += 1;
            if !ggml_names.contains(name) {
                failures.push(format!(
                    "{name}: dense tensor missing from the GGML component"
                ));
                continue;
            }
            let a = mlx.get(name)?;
            let b = ggml.get(name)?;
            if a.dtype() == b.dtype() && a.shape() == b.shape() && a.data() == b.data() {
                dense_equal += 1;
            } else {
                failures.push(format!(
                    "{name}: dense tensor differs ({:?}{:?} vs {:?}{:?})",
                    a.dtype(),
                    a.shape(),
                    b.dtype(),
                    b.shape()
                ));
            }
        }
    }
    for extra in ggml_names.difference(&seen) {
        failures.push(format!("{extra}: present only in the GGML component"));
    }

    println!(
        "ggml={} mlx={} group={group}\n\
         packed projections: {packed} ({packed_byte_equal} block-byte-identical to the runtime repack)\n\
         dense tensors:      {dense} ({dense_equal} byte-identical)\n\
         values vs exact MLX affine f32: {value_diff_weights} of {value_weights} weights differ, max |diff| {max_abs:e}",
        ggml_dir.display(),
        mlx_dir.display()
    );
    for failure in &failures {
        println!("MISMATCH {failure}");
    }
    if failures.is_empty() {
        println!("PARITY OK");
        Ok(())
    } else {
        Err(format!("{} mismatches", failures.len()).into())
    }
}
