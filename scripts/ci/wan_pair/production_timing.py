"""Fixed production-denoise comparison for the merged Wan CUDA 5B Heun implementation."""
import hashlib

from scripts.ci import video_tier1_evidence as evidence

NAME = "sw-6954-gpu0-wan-production-timing-v1"
CASE = "cuda-ti2v5b-t2v-heun"
# The baseline already contains the shared Linear spine and still uses the pre-migration sampler.
BASE = "e86b8d5fc493c16b2d9d852e0c9ecbdefee58294"
FEATURE = "f5c9ff80acd00334fd668e4ecc8d56be3347affd"
PRIOR = {
    "run": "44940165866", "attempt": 1, "artifact": 14722437756,
    "receipt_sha256": "b251bac88e9d53d930f592ce1d9eaa606777f14ec84abd06c7b1fe100823009b",
    "archive_sha256": "e9788eb9ab4930e05e0a5b0afca5257479b72600f8b0cc99499f59aa52d162b7",
    "timing_verdict": "inconclusive",
}
FIXTURE_AND_OUTPUT_SHA256 = "9d90853e107332dc45e18e96533c27f700ce63c66252ec33fde99e6ff4256a18"
CONDITIONING_SHA256 = "ebd34f75afe68c9bd01c6654ecd66c7353031bd48d0523edae5a0681ae9ca193"


def bind(previous, hardware, calls, pairs):
    row = previous["retained"][CASE]
    anchor = {key: row[key] for key in ("fixture", "initial_sha256", "terminal_sha256", "trace_sha256")}
    if evidence.canonical_hash(anchor) != FIXTURE_AND_OUTPUT_SHA256:
        raise ValueError("Wan fixture or bit/trace anchor differs from the audited GPU0 capture")
    return {
        "name": NAME, "retained": {CASE: anchor}, "previous_comparison": PRIOR,
        "conditioning_module_sha256": CONDITIONING_SHA256,
        "protocol": {"scope": "fresh timing and parity only", "fresh_hardware": hardware,
                     "baseline": BASE, "candidate": "exact merged workflow revision",
                     "pairs": pairs, "warmups": calls, "samples_per_timing_child": calls,
                     "order": "ten four-pair blocks: AB, BA, BA, AB (A=base, B=head)",
                     "timer": "CUDA synchronize, prepared native denoise, CUDA synchronize",
                     "interval_ranks": [14, 27], "ceiling": 1.02, "early_stopping": False,
                     "timing_previous_samples_pooled": False},
    }


def prepare_overlay(arm, source, original_digest, wrapper):
    if (hashlib.sha256(source).hexdigest() != original_digest
            or hashlib.sha256(wrapper).hexdigest() != CONDITIONING_SHA256
            or source.count(b"pair.measure(") != 1):
        raise ValueError("Wan timing requires the audited fixture and conditioning procedure")
    if arm == "head":
        # A fresh phase belongs to each render. The real resident path prepares it before
        # denoise, so its construction and destruction must fall outside the synchronized timer.
        sync = b"            device.synchronize()?;\n            let start = Instant::now();"
        call = b"crate::dense_curated_phase(&request, knobs.seed)?.as_ref()"
        if source.count(sync) != 1 or source.count(call) != 1:
            raise ValueError("Wan head lacks the audited native denoise timing boundary")
        source = source.replace(call, b"phase.as_ref()").replace(
            sync, b"            let phase = crate::dense_curated_phase(&request, knobs.seed)?;\n" + sync)
    return source.replace(b"pair.measure(", b"pair.measure_conditioned(") + b"\n" + wrapper
