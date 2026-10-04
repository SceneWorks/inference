"""The one-run HF hub override of kv-poc-campaign.yml (`hf_hub_override`, .github/kv-poc/hub.sh).

An override replaces /Volumes/Models/huggingface/hub for every job of a run, must be a plain
absolute path strictly under the runner user's home on its own volume, is created when missing,
is what the disk precheck measures, and is deleted by an always-run cleanup job behind the same
validation. Every deletion here runs with `rm` shimmed to refuse anything outside the test sandbox,
so even a broken cleanup cannot reach a real path.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
KV_POC = ROOT / ".github" / "kv-poc"
HUB_SH = KV_POC / "hub.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "kv-poc-campaign.yml"
DEFAULT_HUB = "/Volumes/Models/huggingface/hub"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_kv_poc_noise_floor import run_config  # noqa: E402


def syntax(path: str, home: str = "") -> tuple[int, str]:
    done = subprocess.run(
        ["bash", "-c", 'source "$1"; hub_override_syntax_problem "$2" "$3"', "_", str(HUB_SH), path, home],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    return done.returncode, done.stdout + done.stderr


class Sandbox:
    """A real temp dir as $HOME plus an `rm` on PATH that refuses any target outside it."""

    def __init__(self, tmp: str):
        self.root = Path(os.path.realpath(tmp))
        self.home = self.root / "home"
        self.home.mkdir()
        self.outside = self.root / "outside"
        self.outside.mkdir()
        (self.outside / "keep.bin").write_bytes(b"x")
        shim = self.root / "bin"
        shim.mkdir()
        self.rm_log = self.root / "rm.log"
        (shim / "rm").write_text(
            "#!/bin/bash\n"
            f'printf "%s\\n" "$*" >> "{self.rm_log}"\n'
            'for a in "$@"; do\n'
            '  case "$a" in -*) continue ;; esac\n'
            f'  case "$a" in "{self.home}"/?*) ;; *) echo "rm shim: refused $a" >&2; exit 97 ;; esac\n'
            "done\n"
            'exec /bin/rm "$@"\n',
            encoding="utf-8",
        )
        (shim / "rm").chmod(0o755)
        self.env = {**os.environ, "HOME": str(self.home), "PATH": f"{shim}:{os.environ['PATH']}"}

    def cleanup(self, path: str, home: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(HUB_SH), "cleanup", path, home or str(self.home)],
            env=self.env, capture_output=True, text=True, encoding="utf-8", check=False,
        )

    def common(self, override: str, script: str) -> subprocess.CompletedProcess:
        env = {
            **self.env,
            "INFERENCE_SHA": "a" * 40,
            "SCENEWORKS_SHA": "b" * 40,
            "KV_POC_ROOT": str(self.root / "kv-poc"),
            "KV_EXPECTED_RUNNER": "",
            "KV_HF_HUB_OVERRIDE": override,
        }
        return subprocess.run(
            ["bash", "-c", f'source "{KV_POC / "common.sh"}"; {script}'],
            env=env, capture_output=True, text=True, encoding="utf-8", check=False,
        )


class HubOverrideValidationTests(unittest.TestCase):
    def test_syntax_accepts_only_plain_paths_strictly_under_home(self):
        home = "/Users/MTrefry"
        for good in ("/Users/MTrefry/kv-poc-hub-20669", "/Users/MTrefry/a/b_c.d"):
            self.assertEqual(syntax(good, home), (0, ""), good)
            self.assertEqual(syntax(good)[0], 0, f"{good} (hosted config job: no home)")
        for bad in (
            "", "kv-poc-hub", "Users/MTrefry/x", "/Users/MTrefry", "/Users/MTrefry/",
            "/Users/MTrefryX/x", "/Users/other/x", "/tmp/x", DEFAULT_HUB, "/Volumes/Models",
            "/Users/MTrefry/../MTrefry/x", "/Users/MTrefry/a/../../../Volumes/Models",
            "/Users/MTrefry/./x", "/Users/MTrefry//x", "/Users/MTrefry/x/", "/Users/MTrefry/-rf",
            "/Users/MTrefry/a b", "/Users/MTrefry/$(id)", "/Users/MTrefry/x;rm", "/Users/MTrefry/*",
        ):
            self.assertNotEqual(syntax(bad, home)[0], 0, f"accepted {bad!r} under {home}")
        self.assertEqual(syntax("/Users/other/x")[0], 0, "hosted config: any /Users/<user>/x; the runner checks its HOME")
        self.assertNotEqual(syntax("/Volumes/Models/huggingface/hub")[0], 0)
        self.assertNotEqual(syntax("/Users/MTrefry")[0], 0, "a home itself is never an override")

    def test_config_resolves_validates_and_records_the_override(self):
        code, values, log = run_config("a2", DISPATCH_HF_HUB_OVERRIDE="/Users/MTrefry/kv-poc-hub-20669")
        self.assertEqual(code, 0, log)
        self.assertEqual(values["hf_hub_override"], "/Users/MTrefry/kv-poc-hub-20669")
        self.assertIn("| HF hub | `/Users/MTrefry/kv-poc-hub-20669` (one-run OVERRIDE", log)
        code, values, log = run_config("a2")
        self.assertEqual(code, 0, log)
        self.assertEqual(values["hf_hub_override"], "")
        self.assertIn(f"| HF hub | `{DEFAULT_HUB}` (default) |", log)
        for bad in (DEFAULT_HUB, "/Users/MTrefry/../x", "relative/x", "/Users/MTrefry"):
            code, values, log = run_config("a2", DISPATCH_HF_HUB_OVERRIDE=bad)
            self.assertNotEqual(code, 0, bad)
            self.assertIn("::error title=bad campaign parameter::hf_hub_override", log)
            self.assertNotIn("hf_hub_override", values)
        config = (KV_POC / "config.sh").read_text(encoding="utf-8")
        self.assertIn('hf_hub_override="$(read_key hf_hub_override "")"', config, "the run json carries it")

    def test_runner_refuses_symlinked_or_outside_paths_and_creates_a_valid_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            (box.home / "link").symlink_to(box.outside)
            for bad in (str(box.home / "link" / "hub"), str(box.home / "link"), str(box.outside / "hub"),
                        str(box.home), DEFAULT_HUB):
                done = box.common(bad, 'echo "$KV_HF_HUB"')
                self.assertNotEqual(done.returncode, 0, bad)
                self.assertIn("::error title=bad hf_hub_override::", done.stdout)
            self.assertFalse((box.outside / "hub").exists(), "a refused override is never created")
            override = str(box.home / "kv-poc-hub-20669")
            done = box.common(override, 'echo "$KV_HF_HUB"')
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertEqual(done.stdout.strip(), override)
            self.assertTrue(Path(override).is_dir(), "created when missing")
            done = box.common("", 'echo "$KV_HF_HUB"')
            self.assertEqual(done.stdout.strip(), DEFAULT_HUB, "no override = the default hub")

    def test_disk_precheck_measures_the_override_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            override = str(box.home / "kv-poc-hub-20669")
            for pins in ("models.tsv", "models-w2.tsv"):
                done = box.common(
                    override,
                    f'"{sys.executable}" "$KV_DIR/models.py" --report --hub "$KV_HF_HUB" '
                    f'--pins "$KV_DIR/{pins}" --reserve-gib 20',
                )
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(f"free on the hub volume ({override}):", done.stdout, pins)
        build = (KV_POC / "build.sh").read_text(encoding="utf-8")
        for stage in ("stage_models", "stage_w2_models"):
            body = build[build.index(f"{stage}() {{"):]
            body = body[: body.index("\n}\n")]
            self.assertIn('HF_HUB_CACHE="$KV_HF_HUB"', body, stage)
            self.assertIn('models.py" --hub "$KV_HF_HUB"', body, stage)


class HubOverrideCleanupTests(unittest.TestCase):
    def test_cleanup_deletes_only_the_validated_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            override = box.home / "kv-poc-hub-20669"
            blob = override / "models--org--x" / "blobs" / "abc"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"w")
            snap = override / "models--org--x" / "snapshots" / ("1" * 40)
            snap.mkdir(parents=True)
            (snap / "model.safetensors").symlink_to(blob)
            (snap / "escape").symlink_to(box.outside)
            done = box.cleanup(str(override))
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertFalse(override.exists())
            self.assertTrue((box.outside / "keep.bin").is_file(), "links inside are never followed")
            self.assertEqual(box.rm_log.read_text(encoding="utf-8").split(), ["-rf", "--", str(override)])
            done = box.cleanup(str(override))
            self.assertEqual(done.returncode, 0, "an already-deleted override is clean")

    def test_cleanup_refuses_everything_outside_the_validated_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = Sandbox(tmp)
            (box.home / "link").symlink_to(box.outside)
            for bad in (DEFAULT_HUB, "/Volumes/Models", str(box.outside), str(box.home),
                        str(box.home / "link"), str(box.home / "x" / ".." / "link"), "relative"):
                done = box.cleanup(bad)
                self.assertNotEqual(done.returncode, 0, bad)
                self.assertIn("::error title=hub override cleanup refused::", done.stdout, bad)
            self.assertFalse(box.rm_log.exists(), "a refused cleanup never reaches rm")
            self.assertTrue((box.outside / "keep.bin").is_file())
            self.assertTrue((box.home / "link").is_symlink())

    def test_workflow_wires_the_override_and_always_cleans_up(self):
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        self.assertIn("hf_hub_override", workflow[True]["workflow_dispatch"]["inputs"])
        jobs = workflow["jobs"]
        self.assertEqual(jobs["config"]["outputs"]["hf_hub_override"], "${{ steps.resolve.outputs.hf_hub_override }}")
        resolve = next(step for step in jobs["config"]["steps"] if step.get("id") == "resolve")
        self.assertEqual(resolve["env"]["DISPATCH_HF_HUB_OVERRIDE"], "${{ inputs.hf_hub_override }}")
        mac_jobs = [name for name, job in jobs.items() if name not in ("config", "cleanup")]
        for name in mac_jobs:
            self.assertEqual(jobs[name]["env"]["KV_HF_HUB_OVERRIDE"],
                             "${{ needs.config.outputs.hf_hub_override }}", name)
        cleanup = jobs["cleanup"]
        self.assertEqual(sorted(cleanup["needs"]), sorted(["config", *mac_jobs]), "runs after every job")
        self.assertTrue(cleanup["if"].startswith("${{ always() && "), cleanup["if"])
        self.assertIn("needs.config.outputs.hf_hub_override != ''", cleanup["if"])
        self.assertEqual(cleanup["runs-on"], "${{ fromJSON(needs.config.outputs.runs_on) }}")
        self.assertEqual(cleanup["env"]["KV_EXPECTED_RUNNER"], "${{ needs.config.outputs.runner_name }}")
        run = "\n".join(step.get("run", "") for step in cleanup["steps"])
        self.assertIn('bash .github/kv-poc/hub.sh cleanup "$KV_HF_HUB_OVERRIDE" "$HOME"', run)
        self.assertNotIn("rm ", run, "the only delete is hub.sh's guarded one")


if __name__ == "__main__":
    unittest.main()
