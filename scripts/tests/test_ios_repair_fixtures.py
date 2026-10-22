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


def validate(path):
    data = path.read_bytes()
    size = int.from_bytes(data[:8], 'little')
    if size > len(data) - 8:
        raise ValueError('header beyond file')
    header = json.loads(data[8:8 + size])
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
    if previous + 8 + size != len(data):
        raise ValueError('unaccounted payload or truncated file')
    if not any(t['dtype'] == 'BF16' and t['shape'] == [1] for t in header.values()):
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


if __name__ == '__main__':
    unittest.main()
