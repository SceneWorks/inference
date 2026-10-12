"""CPU tests discriminate missing/failed device gates and exact build receipts."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from scripts.release import veda_s3_cuda as s3

ROOT=Path(__file__).resolve().parents[2]


def device_rows():
    return [dict(schema=s3.SCHEMA,kernel=s3.KERNEL,device_ordinal=0,dtype=d,
                 self_test_dimensions=[16,128],native_batch=2,native_heads=2,
                 sparse_calls=1,predictor_score_calls=4,forced_failure_dense=True,
                 unsupported_dtype_dense=True,intentional_dense_identity=True)
            for d in ("BF16","F16")]


def logs(out, stdout, stderr="", phase="test"):
    (out/f"{phase}.stdout.log").write_text(stdout,encoding="utf-8")
    (out/f"{phase}.stderr.log").write_text(stderr,encoding="utf-8")


class S3CudaTests(unittest.TestCase):
    def test_batch_commands_are_scoped_no_run_and_strict_lint(self):
        for phase in ("build","lint"):
            argv=s3.batch_command(phase)
            self.assertEqual(argv[:3],["cmd","/d","/c"])
            self.assertEqual(len(argv),4)
            script=(ROOT/argv[3].replace("\\","/")).read_text()
            self.assertIn('call "%VCVARS%" || exit /b 1',script)
            self.assertIn("--locked",script)
            self.assertIn("-j2",script)
            self.assertIn("--features cuda",script)
            self.assertNotIn("--example",script)
            self.assertNotIn("--ignored",script)
            self.assertIn("exit /b %errorlevel%",script)
        self.assertIn("--no-run --message-format=json",(ROOT/s3.batch_command("build")[3].replace("\\","/")).read_text())
        self.assertIn("-- -D warnings",(ROOT/s3.batch_command("lint")[3].replace("\\","/")).read_text())
        with self.assertRaises(ValueError):s3.batch_command("random")

    def test_exact_cargo_artifact_required_not_stale_glob(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);target=out/"target";target.mkdir()
            binary=target/"s3.exe";binary.write_bytes(b"exact binary")
            row=dict(reason="compiler-artifact",target=dict(name="candle_gen_minimax_h3"),profile=dict(test=True),executable=str(binary))
            for data,valid in [([row,dict(reason="build-finished",success=True)],True),([row],False),([row,row,dict(reason="build-finished",success=True)],False),([dict(row,executable=str(out/"other.exe")),dict(reason="build-finished",success=True)],False)]:
                logs(out,"vcvars noise\n"+"\n".join(map(json.dumps,data)),phase="build")
                if valid:self.assertEqual(s3.compiled_binary(out,target),binary.resolve())
                else:
                    with self.assertRaises(RuntimeError):s3.compiled_binary(out,target)

    def test_absent_device_requires_exact_expected_failure_not_skip_or_zero_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            failed=f"running 1 test\ntest {s3.TEST} ... FAILED\ntest result: FAILED. 0 passed; 1 failed; 0 ignored"
            error="S3 requires an exclusively admitted CUDA device"
            logs(out,failed,error)
            self.assertEqual(s3.verify_test_output(out,"test",dict(returncode=101,timeout=False),True),[])
            for stdout,stderr,code,timed in [(failed,error,0,False),("running 0 tests",error,101,False),(failed,"",101,False),(failed,error,101,True)]:
                logs(out,stdout,stderr)
                with self.assertRaises(RuntimeError):s3.verify_test_output(out,"test",dict(returncode=code,timeout=timed),True)

    def test_positive_gate_requires_two_native_receipts_and_no_ignored_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            def body(rows):
                return f"running 1 test\ntest {s3.TEST} ... "+"\n".join("S3_DEVICE_RECEIPT "+json.dumps(r) for r in rows)+"\ntest result: ok. 1 passed; 0 failed; 0 ignored"
            rows=device_rows();logs(out,body(rows))
            self.assertEqual(s3.verify_test_output(out,"test",dict(returncode=0,timeout=False),False),rows)
            for field,value in [("kernel","other"),("device_ordinal",1),("self_test_dimensions",[16]),("native_batch",1),("predictor_score_calls",0),("forced_failure_dense",False)]:
                changed=device_rows();changed[0][field]=value;logs(out,body(changed))
                with self.assertRaises(RuntimeError):s3.verify_test_output(out,"test",dict(returncode=0,timeout=False),False)
            logs(out,body(rows[:1]))
            with self.assertRaises(RuntimeError):s3.verify_test_output(out,"test",dict(returncode=0,timeout=False),False)
            logs(out,body(rows).replace("0 ignored","1 ignored"))
            with self.assertRaises(RuntimeError):s3.verify_test_output(out,"test",dict(returncode=0,timeout=False),False)

    def test_child_failure_and_hashes_are_persisted_before_gate_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);logs(out,"failed compiler", "compiler stderr")
            receipt=dict(results=[])
            with patch.object(s3.guard,"execute_owned",return_value=dict(returncode=17,timeout=False)):
                result=s3.append_result(receipt,out,"test",["binary"],time.monotonic()+30,Mock)
            persisted=json.loads((out/"receipt.json").read_text())
            self.assertEqual(result["returncode"],17)
            self.assertEqual(persisted["results"][0]["stdout_sha256"],s3.hash_file(out/"test.stdout.log"))
            self.assertEqual(persisted["results"][0]["stderr_sha256"],s3.hash_file(out/"test.stderr.log"))

    def test_device_gate_source_is_nonignored_and_requires_cuda(self):
        source=(ROOT/"crates/media/candle-gen/candle-gen-minimax-h3/src/veda.rs").read_text()
        before,gate=source.split("fn actual_cuda_kernel_gate()",1)
        self.assertNotIn("#[ignore",before[before.rfind("/// Coordinator"):])
        self.assertIn('Device::new_cuda(0).expect("S3 requires an exclusively admitted CUDA device")',gate)
        self.assertNotIn("else { return",gate)
        self.assertIn("S3_DEVICE_RECEIPT",gate)
        self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if absent else device["uuid"]',Path(s3.__file__).read_text())

    def test_workflow_is_manual_scoped_and_preserves_deadline_upload(self):
        import yaml
        workflow=yaml.safe_load((ROOT/".github/workflows/real-weights.yml").read_text())
        job=workflow["jobs"]["veda-s3-cuda"]
        self.assertEqual(job["if"],"github.event_name == 'workflow_dispatch' && inputs.profile == 'veda-s3-cuda'")
        self.assertEqual(job["timeout-minutes"],75)
        self.assertEqual(job["env"]["CARGO_BUILD_JOBS"],"2")
        self.assertNotIn("S0_Q4_DIT_DIR",job["env"])
        upload=job["steps"][-1]
        self.assertEqual(upload["if"],"always()")
        self.assertEqual(upload["with"]["if-no-files-found"],"error")
        self.assertEqual(upload["with"]["path"],"${{ runner.temp }}/veda-s3-cuda-${{ github.run_id }}-${{ github.run_attempt }}/")
        self.assertEqual(s3.guard.MAX_SECONDS,3600)


if __name__ == "__main__":unittest.main()
