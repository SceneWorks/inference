"""CPU-only semantic tests. Synthetic shards are explicitly not native evidence."""
import contextlib, importlib.util, io, json, struct, tempfile, unittest
from pathlib import Path
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location("single_linear",HERE/"single_linear.py")
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
FIXTURE=HERE.parents[3]/"crates/media/mlx-gen/mlx-gen-qwen-image-2-1/tests/fixtures/single-linear-candidate2"

class PreparationTests(unittest.TestCase):
    def make_snapshot(self, root, bad_shape=False):
        snapshot=root/d.REV; component=snapshot/"q4/transformer";component.mkdir(parents=True)
        tensors={};payload=[];offset=0
        for k in d.KEYS:
            shape=[4096,512] if k.endswith(".weight") else [4096,64]
            if bad_shape and k.endswith(".weight"):shape=[4096,511]
            dtype="U32" if k.endswith(".weight") else "BF16"
            b=bytes(shape[0]*shape[1]*(4 if dtype=="U32" else 2))
            tensors[k]={"dtype":dtype,"shape":shape,"data_offsets":[offset,offset+len(b)]};payload.append(b);offset+=len(b)
        h=json.dumps(tensors).encode()
        (component/"synthetic.safetensors").write_bytes(struct.pack("<Q",len(h))+h+b"".join(payload))
        (component/"config.json").write_text(json.dumps({"quantization":{"bits":4,"group_size":64}}))
        return snapshot
    def test_actual_fixture_and_bit_exact_constructed_rows(self):
        with tempfile.TemporaryDirectory() as name, contextlib.redirect_stdout(io.StringIO()):
            root=Path(name);snapshot=self.make_snapshot(root);out=root/"prepared"
            d.prepare(FIXTURE,snapshot,out);p=d.read_json(out/"prepared.json")
            actual=(FIXTURE/p["capturedInput"]["file"]).read_bytes();replay=(out/p["input"]["file"]).read_bytes()
            self.assertEqual(p["input"]["shape"],[1,65,4096])
            for i in range(65):self.assertEqual(replay[i*16384:(i+1)*16384],actual[(i%3)*16384:(i%3+1)*16384])
            self.assertEqual(p["originalActivationDtype"],"Float32")
            with self.assertRaisesRegex(ValueError,"exclusive"):d.prepare(FIXTURE,snapshot,out)
    def test_selected_shape_refused_before_tensor_work(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);snapshot=self.make_snapshot(root,True)
            with self.assertRaisesRegex(ValueError,"shape/dtype"):d.prepare(FIXTURE,snapshot,root/"bad")
    def test_compiled_source_absence_is_a_refusal(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError,"compiled MLX source absent"):d.verify_build(Path(name),Path(name)/"proof.json")
    def test_bounds_reject_negative_and_nonfinite(self):
        import numpy as np
        r=np.ones((2,2)); b=np.full((2,2),.001)
        self.assertTrue(d.fits(r,r,b));self.assertFalse(d.fits(r+.1,r,b))
        self.assertFalse(d.fits(r,r,-b));self.assertFalse(d.fits(np.full((2,2),np.nan),r,b))
    def test_actual_input_factor_mutants(self):
        with contextlib.redirect_stdout(io.StringIO()):d.selftest(FIXTURE)

if __name__=="__main__":unittest.main()
