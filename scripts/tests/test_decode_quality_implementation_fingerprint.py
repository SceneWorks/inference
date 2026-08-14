import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "ci" / "decode_quality_implementation_fingerprint.py"
SPEC = importlib.util.spec_from_file_location("decode_quality_fingerprint", SCRIPT)
assert SPEC and SPEC.loader
fingerprint_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fingerprint_module)


class DecodeQualityImplementationFingerprintTests(unittest.TestCase):
    def test_embedded_decode_quality_source_closure_is_current(self) -> None:
        self.assertEqual(fingerprint_module.embedded(), fingerprint_module.fingerprint())
        self.assertEqual(len(fingerprint_module.fingerprint()), 64)


if __name__ == "__main__":
    unittest.main()
