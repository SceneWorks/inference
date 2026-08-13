import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "ci" / "collect_decode_quality_admission.py"
SPEC = importlib.util.spec_from_file_location("decode_quality_collector", SCRIPT)
assert SPEC and SPEC.loader
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def receipt(seed: int, error: int) -> dict:
    token = f"{seed:064x}"
    return {
        "family": "sdxl",
        "resolvedRoute": "realvisxl",
        "backend": "mlx",
        "tier": "q4",
        "mode": "text_to_image",
        "overlay": None,
        "geometry": {
            "width": 1024,
            "height": 1024,
            "batch": 1,
            "frames": 1,
            "referenceCount": 0,
        },
        "usePid": False,
        "tileEdge": 896,
        "overlap": 192,
        "metric": "max_abs_rgb_u8",
        "maximumError": 48,
        "seed": seed,
        "productionLatentProvenance": "realvisxl@abc q4 production prompt=p9-v1",
        "productionLatentSha256": token,
        "denseOutputSha256": token,
        "tiledOutputSha256": token,
        "observedError": error,
    }


def test_seals_a_multi_seed_coordinate_and_retains_failure_reason(tmp_path: Path) -> None:
    log = tmp_path / "quality.log"
    log.write_text(
        "noise\n"
        + "\n".join(
            f"test output DECODE_QUALITY_V1 {json.dumps(row)}"
            for row in (receipt(7, 47), receipt(99, 52))
        )
        + "\n",
        encoding="utf-8",
    )

    policies = collector.seal(collector.read_receipts([log]))

    assert len(policies) == 1
    policy = policies[0]
    assert [fixture["seed"] for fixture in policy["fixtures"]] == [7, 99]
    assert policy["disposition"] == {
        "kind": "refused",
        "reason": "max_abs_rgb_u8 exceeded 48: seed 99=52",
    }
    assert len(policy["productionEvidenceSha256"]) == 64
    assert policy["productionEvidenceSha256"] == collector.seal(
        [receipt(99, 52), receipt(7, 47)]
    )[0]["productionEvidenceSha256"]


def test_rejects_nonsemantic_fields_instead_of_absorbing_measurements(tmp_path: Path) -> None:
    row = receipt(7, 47)
    row["elapsedMs"] = 12
    log = tmp_path / "quality.log"
    log.write_text(
        f"DECODE_QUALITY_V1 {json.dumps(row)}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="semantic allowlist"):
        collector.read_receipts([log])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("usePid", 0, "usePid must be a boolean"),
        ("overlay", "bad=value", "overlay must be"),
        ("tileEdge", 128, "tileEdge must exceed"),
    ],
)
def test_rejects_malformed_semantic_coordinates(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    row = receipt(7, 47)
    if field == "tileEdge":
        row["overlap"] = 128
    row[field] = value
    log = tmp_path / "quality.log"
    log.write_text(f"DECODE_QUALITY_V1 {json.dumps(row)}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        collector.read_receipts([log])
