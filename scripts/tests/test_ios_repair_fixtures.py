"""Compile/run the actual sparse fixture writer on CPU; validate its full file format/stat extent."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def sparse_writer():
    text = (ROOT / 'crates/media/mlx-gen/mlx-gen-minimax-h3/src/memory_strategy.rs').read_text(encoding='utf-8')
    start = text.index('    fn sparse_snapshot(')
    end = text.index('\n    /// Write the DiT', start)
    return text[start:end]


def validate(path, dense_witness=True):
    total = path.stat().st_size
    with path.open('rb') as source:
        size = int.from_bytes(source.read(8), 'little')
        if size > total - 8 or size > 65536:
            raise ValueError('header beyond file or fixture header budget')
        header = json.loads(source.read(size))
    intervals = []
    widths = dict(BF16=2, U8=1)
    for name, tensor in header.items():
        start, end = tensor['data_offsets']
        elements = 1
        for extent in tensor['shape']:
            elements *= extent
        if end - start != elements * widths[tensor['dtype']]:
            raise ValueError('wrong tensor byte extent')
        intervals.append((start, end))
    previous = 0
    for start, end in sorted(intervals):
        if start != previous:
            raise ValueError('payload gap/overlap')
        previous = end
    if previous + 8 + size != total:
        raise ValueError('unaccounted payload or truncated file')
    if dense_witness and not any(t['dtype'] == 'BF16' and t['shape'] == [1] for t in header.values()):
        raise ValueError('missing real dense tier witness')


class SparseFixtureFormat(unittest.TestCase):
    def compile_and_write(self, root, mutate=False):
        body = sparse_writer()
        if mutate:
            body = body.replace('file.write_all(header.as_bytes()).unwrap();',
                                'file.write_all(&vec![0_u8; header.len()]).unwrap();')
        source = root / 'fixture.rs'
        source.write_text('use std::path::Path;\n' + body + '\nfn main() {\n'
                          'let dir = std::env::args().nth(1).unwrap();\n'
                          'sparse_snapshot(Path::new(&dir), &[("even", 1024), ("odd", 1025)]);\n}\n',
                          encoding='utf-8')
        result = subprocess.run(['nice', '-n', '10', str(Path.home() / '.cargo/bin/rustc'),
                                 '--edition', '2021', '-C', 'debuginfo=0', str(source),
                                 '-o', str(root / 'fixture')], capture_output=True, text=True,
                                encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([str(root / 'fixture'), str(root / 'snapshot')], capture_output=True,
                                text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        return root / 'snapshot'

    def test_actual_writer_preserves_even_and_odd_stat_sizes_with_valid_dense_headers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.compile_and_write(Path(temp))
            for name, size in [('even', 1024), ('odd', 1025)]:
                path = root / name / 'model.safetensors'
                self.assertEqual(path.stat().st_size, size)
                validate(path)

    def test_compiling_zero_header_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.compile_and_write(Path(temp), mutate=True)
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                validate(root / 'even/model.safetensors')


class LensSparseGrowth(unittest.TestCase):
    """Compile the exact writer; inspect bounded headers, never the sparse payload."""
    def run_writer(self, root, mutation=False):
        text = (ROOT / 'crates/media/mlx-gen/mlx-gen-lens/src/registry.rs').read_text(encoding='utf-8')
        start = text.index('    fn write_sparse_footprint_shard(')
        body = text[start:text.index('    fn fixture_disk_bytes(', start)]
        if mutation:
            body = body.replace('file.set_len(total)', 'file.set_len(total + 1)')
        source = root / 'lens.rs'
        source.write_text(body + '\nfn main() { let args: Vec<_> = std::env::args().collect(); '
                          'write_sparse_footprint_shard(std::path::Path::new(&args[1]), args[2].parse().unwrap()); }',
                          encoding='utf-8')
        result = subprocess.run(['nice', '-n', '10', str(Path.home() / '.cargo/bin/rustc'),
                                 '--edition', '2021', '-C', 'debuginfo=0', str(source), '-o', str(root / 'lens')],
                                capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        path = root / 'model.safetensors'
        for size in [1024, 1025, 16385, 32_287_416_586]:
            result = subprocess.run([str(root / 'lens'), str(path), str(size)], capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stderr)
            if mutation:
                with self.assertRaisesRegex(ValueError, 'unaccounted payload'):
                    validate(path, dense_witness=False)
            else:
                self.assertEqual(path.stat().st_size, size)
                validate(path, dense_witness=False)

    def test_actual_writer_rewrites_small_odd_page_and_larger_than_calibration_extents(self):
        with tempfile.TemporaryDirectory() as temp:
            self.run_writer(Path(temp))

    def test_compiling_unrewritten_growth_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            self.run_writer(Path(temp), mutation=True)


def rust_function(text, name):
    start = text.index('fn ' + name + '(')
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


class SourceMathRepairs(unittest.TestCase):
    """Exact Rust math bodies with metadata stubs. NOT native allocator/kernel validation."""
    SDXL = ROOT / 'crates/media/mlx-gen/mlx-gen-sdxl/src'
    PRELUDE = '''
#[derive(Clone, Copy, Debug, PartialEq)] enum Dtype { Float16, Float32, Int32 }
impl Dtype { fn from_promoting_types(a: Self, b: Self) -> Self { if a == b { a } else { Self::Float32 } } }
fn dtype_bytes(d: Dtype) -> u64 { if d == Dtype::Float16 { 2 } else { 4 } }
fn tensor_bytes(width: u64, dims: &[u64]) -> u64 { dims.iter().product::<u64>() * width }
mod mlx_gen {
    pub mod ios_quantum { pub(crate) use crate::tensor_bytes; }
    pub mod adapters { pub mod facts {
        pub struct TensorFacts { pub shape: Vec<i32>, pub dtype: crate::Dtype, pub strides: Vec<i64> }
        impl TensorFacts { pub fn elements(&self) -> u64 { self.shape.iter().map(|v| *v as u64).product() } }
    } }
}
'''

    def compile_run(self, source, expected_success):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'math.rs').write_text(source, encoding='utf-8')
            build = subprocess.run(['nice', '-n', '10', str(Path.home() / '.cargo/bin/rustc'),
                                    '--edition', '2021', '-C', 'debuginfo=0', str(root / 'math.rs'),
                                    '-o', str(root / 'math')], capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run([str(root / 'math')], capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode == 0, expected_success, result.stderr)

    def norm_source(self, mutation=None):
        q = (self.SDXL / 'quantum.rs').read_text(encoding='utf-8')
        te = (self.SDXL / 'text_encoder.rs').read_text(encoding='utf-8')
        budget = (ROOT / 'crates/media/mlx-gen/src/ios_quantum_budget.rs').read_text(encoding='utf-8')
        norm = 'pub(crate) ' + rust_function(q, 'final_norm_bytes_from_metadata')
        pool = rust_function(te, 'clip_final_scratch')
        if mutation == 'scalars':
            norm = norm.replace('+ affine + scalars', '+ affine')
        if mutation == 'gather':
            pool = pool.replace('gather.saturating_add(', '0_u64.saturating_add(')
        prelude = self.PRELUDE.replace('fn tensor_bytes(', 'pub fn tensor_bytes(')
        prelude = prelude.replace('pub(crate) use crate::tensor_bytes;', 'pub(crate) use crate::tensor_bytes;\n'
                                 'pub const ALLOCATOR_PAGE_BYTES: u64 = 16384;\n' + rust_function(budget, 'allocated_bytes').replace('fn allocated_bytes', 'pub fn allocated_bytes'))
        return prelude + '''
mod quantum { use crate::{Dtype, dtype_bytes, tensor_bytes, mlx_gen};
    pub enum Backend { Cpu, Metal }
    fn backend() -> Backend { Backend::Cpu }
    fn u(v: i32) -> u64 { v.max(0) as u64 }
    pub fn dtype_bytes(d: Dtype) -> u64 { crate::dtype_bytes(d) }
    pub fn with_backend<T>(_: Backend, run: impl FnOnce() -> T) -> T { run() }
    pub struct LinearWork;
    impl LinearWork { pub fn operand_bytes(&self, _: &[i32], _: Dtype) -> u64 { 0 }
        pub fn output_dtype(&self, d: Dtype) -> Dtype { d } pub fn output_width(&self) -> u64 { 0 } }
''' .replace('Dtype, dtype_bytes, tensor_bytes', 'Dtype, tensor_bytes') + norm + '\n}\n' + pool + '\n' + rust_function(te, 'final_pool_prices_live_eos_and_native_fallback_scalar_operands') + '\nfn main() { final_pool_prices_live_eos_and_native_fallback_scalar_operands(); }'

    def test_actual_norm_and_eos_math(self):
        self.compile_run(self.norm_source(), True)

    def test_compiling_missing_fallback_scalars_is_rejected(self):
        self.compile_run(self.norm_source('scalars'), False)

    def test_compiling_missing_eos_lifetime_is_rejected(self):
        self.compile_run(self.norm_source('gather'), False)

    def ownership_source(self, mutation=False):
        q = (self.SDXL / 'quantum.rs').read_text(encoding='utf-8')
        body = rust_function(q, 'unique_allocation_requests')
        if mutation:
            body = body.replace('        owned\n', '        if input != compute { owned[4] = tensor(compute, cfg_rows); }\n        owned\n')
        return self.PRELUDE + '''
struct ForwardInputOwnership { state: Dtype, scaled: Option<Dtype>, cast_before_cfg: bool }
impl ForwardInputOwnership { ''' + body + ''' }
fn main() {
    for elements in [1, 256, 8193] {
        for rows in [1, 2] {
            let f = ForwardInputOwnership { state: Dtype::Float32, scaled: None, cast_before_cfg: false };
            let cfg = if rows == 2 { elements * rows * 4 } else { 0 };
            assert_eq!(f.unique_allocation_requests(elements, elements * 2, Dtype::Float16, rows),
                [elements * 8, 0, 0, cfg, 0]);
            let f = ForwardInputOwnership { state: Dtype::Float32, scaled: Some(Dtype::Float32), cast_before_cfg: true };
            let cfg = if rows == 2 { elements * rows * 2 } else { 0 };
            assert_eq!(f.unique_allocation_requests(elements, elements, Dtype::Float16, rows),
                [elements * 4, elements * 4, elements * 2, cfg, 0]);
        }
    }
}
'''

    def z_read_source(self, mutation=False):
        z = (ROOT / 'crates/media/mlx-gen/mlx-gen-z-image/src/quantum_tests.rs').read_text(encoding='utf-8')
        budget = (ROOT / 'crates/media/mlx-gen/src/ios_quantum_budget.rs').read_text(encoding='utf-8')
        projection = rust_function(z, 'expected_streamed_read_capacity')
        if mutation:
            projection = projection.replace('mlx_gen::ios_quantum::allocated_bytes(h.data_bytes)', 'h.data_bytes')
        return '''
mod mlx_gen {
    pub mod gen_core { pub mod weightsmeta {
        pub enum Dtype { BF16, F32, U8 }
        pub struct SafetensorsTensorHeader { pub name: String, pub dtype: Dtype, pub shape: Vec<usize>, pub data_bytes: u64 }
    } }
    pub mod ios_quantum { pub const ALLOCATOR_PAGE_BYTES: u64 = 16384; pub const PAGE_BYTES: u64 = ALLOCATOR_PAGE_BYTES;
''' + 'pub ' + rust_function(budget, 'allocated_bytes') + '\n} }\n' + projection + '\n' + rust_function(z, 'streamed_read_capacity_distinguishes_dtype_and_each_page_boundary') + '\nfn main() { streamed_read_capacity_distinguishes_dtype_and_each_page_boundary(); }'

    def test_actual_z_image_expected_read_projection(self):
        self.compile_run(self.z_read_source(), True)

    def test_compiling_raw_header_capacity_mutation_is_rejected(self):
        self.compile_run(self.z_read_source(True), False)

    def audio_source(self, mutation=False):
        text = (ROOT / 'crates/media/mlx-gen/mlx-gen-minimax-h3/src/memory_strategy.rs').read_text(encoding='utf-8')
        start = text.index('        let audio_f32: u64 =')
        expression = text[start:text.index('.sum();', start) + len('.sum();')]
        if mutation:
            expression = expression.replace('* 4)', '* 1)')
        return 'struct Header { shape: Vec<usize> }\nfn main() {\n' + \
            'let audio_headers = [Header { shape: vec![1] }, Header { shape: vec![127] }];\n' + expression + \
            '\nassert_eq!(audio_f32, 512, "BF16 parameter AND U8 padding materialize F32"); }'

    def test_actual_audio_fixture_f32_expectation(self):
        self.compile_run(self.audio_source(), True)

    def test_compiling_stored_width_audio_mutation_is_rejected(self):
        self.compile_run(self.audio_source(True), False)

    def coordinate_source(self, mutation=False):
        text = (self.SDXL / 'ios_pricing.rs').read_text(encoding='utf-8')
        body = rust_function(text, 'ancestral_body_coordinate')
        if mutation:
            body = body.replace('Some("first") => first', 'Some("first") => steady')
        return self.PRELUDE + '\n#[derive(Debug)] enum Error { Unsupported(String) }\n' + \
            'type Result<T> = std::result::Result<T, Error>;\n' + body + '\n' + \
            rust_function(text, 'ancestral_body_matching_preserves_precision_and_cfg_coordinates') + \
            '\nfn main() { ancestral_body_matching_preserves_precision_and_cfg_coordinates(); }'

    def test_actual_same_body_coordinate_math(self):
        self.compile_run(self.coordinate_source(), True)

    def test_compiling_mixed_precision_body_match_is_rejected(self):
        self.compile_run(self.coordinate_source(True), False)

    def test_actual_forward_owner_math(self):
        self.compile_run(self.ownership_source(), True)

    def test_compiling_fictitious_boundary_cast_is_rejected(self):
        self.compile_run(self.ownership_source(True), False)


if __name__ == '__main__':
    unittest.main()
