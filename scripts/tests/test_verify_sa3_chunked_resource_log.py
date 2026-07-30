import tempfile
import unittest
from pathlib import Path

from scripts.reference.verify_sa3_chunked_resource_log import InvalidEvidence, parse


def record(case, mode, operation, peak_device, seconds):
    starts = 1 if mode == "direct" else 14
    processed = 1292 if mode == "direct" else 1792
    dim = 768 if case == "same_s" else 1536
    return (
        f"SA3_CHUNKED_RESOURCE case={case} mode={mode} operation={operation} "
        f"device=Metal(MetalDevice(DeviceId(0))) dtype=F32 latent_len=1292 "
        f"samples=5292032 dim={dim} chunk_size=128 overlap=32 starts={starts} "
        f"processed_latents={processed} load_seconds=0.5 "
        f"load_peak_rss_bytes=Some(7000000000) load_device_bytes=Some(1000000000) "
        f"elapsed_seconds={seconds} peak_rss_bytes=Some(9000000000) "
        f"peak_device_bytes=Some({peak_device}) "
        f"output_shape=(1, 2, 5292032) checksum=1\n"
    )


def valid_log():
    lines = []
    for case in ("same_s", "same_l"):
        for operation in ("encode", "decode"):
            lines.append(record(case, "direct", operation, 9_000_000_000, 10))
            lines.append(record(case, "chunked", operation, 7_500_000_000, 14))
    return "".join(lines)


class VerifySa3ChunkedResourceLogTests(unittest.TestCase):
    def parse_text(self, text):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resource.log"
            path.write_text(text, encoding="utf-8")
            return parse(path)

    def test_accepts_complete_reducing_metal_matrix(self):
        evidence = self.parse_text(valid_log())
        self.assertEqual(evidence["backend"], "Metal(MetalDevice(DeviceId(0)))")
        self.assertEqual(
            evidence["comparisons"]["same_l"]["decode"]["deviceReductionBytes"],
            1_500_000_000,
        )

    def test_rejects_missing_or_duplicate_records(self):
        lines = valid_log().splitlines(keepends=True)
        with self.assertRaisesRegex(InvalidEvidence, "matrix mismatch"):
            self.parse_text("".join(lines[:-1]))
        with self.assertRaisesRegex(InvalidEvidence, "duplicate"):
            self.parse_text(valid_log() + lines[0])

    def test_rejects_no_reduction_and_excess_wall_time(self):
        with self.assertRaisesRegex(InvalidEvidence, "did not reduce Metal allocation"):
            self.parse_text(
                valid_log().replace(
                    record("same_l", "chunked", "decode", 7_500_000_000, 14),
                    record("same_l", "chunked", "decode", 9_000_000_000, 14),
                )
            )
        with self.assertRaisesRegex(InvalidEvidence, "wall-time"):
            self.parse_text(
                valid_log().replace(
                    record("same_s", "chunked", "encode", 7_500_000_000, 14),
                    record("same_s", "chunked", "encode", 7_500_000_000, 21),
                )
            )


if __name__ == "__main__":
    unittest.main()
