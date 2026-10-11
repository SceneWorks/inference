"""Delivery regression tests over real ffmpeg output, independent of model weights."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

spec=importlib.util.spec_from_file_location("acceptance_run",Path(__file__).with_name("run.py"))
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class DeliveryTests(unittest.TestCase):
    def test_complete_picture_and_source_audio_survive_mux(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            rgb=root/"result.rgb"
            # A full 39-frame picture independent of the audio stream's length.
            rgb.write_bytes(bytes([20,80,160])*(1024*576*39))
            for audio_duration in (None,0.4,1.625):
                with self.subTest(audio_duration=audio_duration):
                    source=root/"source.mp4"
                    args=["ffmpeg","-v","error","-y","-f","lavfi","-i","color=size=512x288:rate=24:duration=1.625"]
                    if audio_duration is not None:
                        args += ["-f","lavfi","-i",f"sine=frequency=440:sample_rate=32000:duration={audio_duration}"]
                    args += ["-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac",str(source)]
                    subprocess.run(args,check=True)
                    out=root/"delivered.mp4"
                    runner.mux(rgb,source,out)
                    if audio_duration is not None:
                        def soundtrack(path):
                            return subprocess.check_output(["ffmpeg","-v","error","-i",str(path),"-map","0:a:0","-f","hash","-hash","sha256","-"])
                        self.assertEqual(soundtrack(source),soundtrack(out))


if __name__=="__main__":
    unittest.main()
