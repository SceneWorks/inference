import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "ci" / "decode_quality_implementation_fingerprint.py"
SPEC = importlib.util.spec_from_file_location("decode_quality_fingerprint", SCRIPT)
assert SPEC and SPEC.loader
fingerprint_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fingerprint_module)


def test_embedded_decode_quality_source_closure_is_current() -> None:
    assert fingerprint_module.embedded() == fingerprint_module.fingerprint()
    assert len(fingerprint_module.fingerprint()) == 64
