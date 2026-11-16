import io
import pathlib
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ci"))
import flash_attn_prebuilt as packer
import fetch_flash_attn_prebuilt as fetcher


class FlashAttnPrebuiltTests(unittest.TestCase):
    def test_package_fetch_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            crate = root / "crate"
            (crate / "kernels").mkdir(parents=True)
            (crate / "build.rs").write_text("const FLAGS: &str = \"sm_80+sm_90+sm_120\";\n")
            (crate / "Cargo.toml").write_text("[package]\nname = \"candle-flash-attn\"\n")
            (crate / "kernels/flash.cu").write_text("// fixture\n")
            for name in packer.LICENSES:
                (crate / name).write_text("license fixture\n")
            source_hash = packer.source_sha256(crate)
            build = root / "build"
            build.mkdir()
            (build / "libflashattention.a").write_bytes(b"archive fixture")
            asset = root / "flash.tar.gz"
            target = "x86_64-unknown-linux-gnu"
            with patch.object(packer, "cuda_version", return_value="12.9"), patch.object(packer, "host_compiler_version", return_value="11.4.0"):
                packer.pack(crate, build, asset, target, "a" * 40)
            with tarfile.open(asset, "r:gz") as bundle:
                manifest = bundle.extractfile("flash-attn-prebuilt.txt").read().decode()
                self.assertIn(f"source_sha256={source_hash}\n", manifest)
            with patch.object(fetcher, "CRATE", crate), patch.object(fetcher.urllib.request, "urlopen", return_value=io.BytesIO(asset.read_bytes())):
                directory = fetcher.fetch(target, root / "cache")
            self.assertEqual((directory / "libflashattention.a").read_bytes(), b"archive fixture")
            (directory / "libflashattention.a").write_bytes(b"tampered")
            with patch.object(fetcher, "CRATE", crate):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    fetcher.fetch(target, root / "cache")

    def test_missing_release_falls_back(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(fetcher, "source_sha256", return_value="b" * 64), patch.object(fetcher.urllib.request, "urlopen", side_effect=fetcher.urllib.error.HTTPError("url", 404, "missing", {}, None)):
                self.assertIsNone(fetcher.fetch("x86_64-pc-windows-msvc", pathlib.Path(temp)))


if __name__ == "__main__":
    unittest.main()
