import json, subprocess, sys, tempfile, unittest
from pathlib import Path
SCRIPT = Path(__file__).parents[1] / "sc20686_cache_attribution.py"
H = "a" * 64
class AttributionTests(unittest.TestCase):
    def row(self, family, **kw):
        return {"producer":"sc20686-campaign-adapter-v1","family":family,"variant":"flux2_klein_9b_edit" if family.startswith("flux") else "wan2_2_ti2v_5b","coordinate_id":"a"*16,"arm":"normal","source_ref":"deadbeef","model_snapshot_sha256":H,"model_snapshot_bytes":1_000_000_000,"geometry":{"resolution":"512x512","reference_count":0,"frames":1,"prompt":"p","guidance":1,"layers":32,"heads":32,"head_dimension":128,"sq":1,"skv":1024,"dtype":"bf16","mask":"causal","rope":"4-axis"},"lifecycle":{"created":1,"reused":2,"invalidated":1},"allocator_samples":[{"peak_bytes":10*1024**3}],"process_samples":[{"peak_bytes":10*1024**3}],"raw_receipt_sha256":H,"raw_receipt_sidecar_sha256":H,"real_weights":True,"full_generation":True,"attention_kind":"cross","current_persistent_bytes":600*1024**2,"current_read_transient_bytes":100*1024**2,"candidate_persistent_bytes":100*1024**2,"candidate_read_transient_bytes":100*1024**2,"generation_duration_ms":1000,"cache_read_duration_ms":100,"reused_requests":2,**kw}
    def execute(self, rows):
        with tempfile.TemporaryDirectory() as d:
            i,o=Path(d)/"i.json",Path(d)/"o.json"; i.write_text(json.dumps(rows),encoding="utf-8"); p=subprocess.run([sys.executable,str(SCRIPT),str(i),str(o)]); return p.returncode, (json.loads(o.read_text()) if o.exists() else None)
    def test_separate_family_go_and_runtime_fraction(self):
        code,out=self.execute([self.row("flux2-klein"),self.row("wan")]); self.assertEqual(code,0); self.assertEqual(out["decisions"]["wan"]["decision"],"blocked")
    def test_self_missing_identity_and_duplicate_fail_closed(self):
        for rows in ([self.row("wan",attention_kind="self")],[self.row("wan",real_weights=False)],[self.row("wan"),self.row("wan")]): self.assertNotEqual(self.execute(rows)[0],0)
    def test_no_double_counting(self):
        code,out=self.execute([self.row("wan",current_persistent_bytes=600*1024**2,current_read_transient_bytes=400*1024**2,candidate_persistent_bytes=500*1024**2,candidate_read_transient_bytes=400*1024**2)]); self.assertEqual(code,0); self.assertEqual(out["decisions"]["wan"]["decision"],"blocked")
if __name__ == "__main__": unittest.main()
