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
        "loadShape": "deferred_materialization",
        "artifact": {
            "repository": "SceneWorks/realvisxl-mlx",
            "revision": "a" * 40,
            "variant": "q4",
            "fingerprint": f"SceneWorks/realvisxl-mlx@{'a' * 40}:q4",
        },
        "implementationFingerprint": collector.IMPLEMENTATION_FINGERPRINT,
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
            f"test output DECODE_QUALITY_V2 {json.dumps(row)}"
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
        f"DECODE_QUALITY_V2 {json.dumps(row)}\n",
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
    log.write_text(f"DECODE_QUALITY_V2 {json.dumps(row)}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        collector.read_receipts([log])


def test_load_shape_artifact_implementation_and_tile_pair_are_exact_coordinates() -> None:
    rows = [receipt(7, 1), receipt(99, 2)]
    variants: list[dict] = []
    for mutate in (
        lambda row: row.update(loadShape="eager_materialization"),
        lambda row: row["artifact"].update(variant="q8"),
        lambda row: row.update(tileEdge=768, overlap=128),
    ):
        pair = [receipt(7, 1), receipt(99, 2)]
        for row in pair:
            mutate(row)
        variants.extend(pair)

    policies = collector.seal([*rows, *variants])
    assert len(policies) == 4
    coordinates = {
        (
            policy["loadShape"],
            policy["artifact"]["variant"],
            policy["implementationFingerprint"],
            policy["tileEdge"],
            policy["overlap"],
        )
        for policy in policies
    }
    assert len(coordinates) == 4


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update(loadShape="deferred"), "unsupported loadShape"),
        (
            lambda row: row["artifact"].update(revision="A" * 40),
            "artifact.revision",
        ),
        (
            lambda row: row["artifact"].update(extra="ambient"),
            "exact ABI-2 identity axes",
        ),
        (
            lambda row: row.update(implementationFingerprint="g" * 64),
            "implementationFingerprint",
        ),
        (
            lambda row: row.update(implementationFingerprint="f" * 64),
            "running inference source closure",
        ),
    ],
)
def test_rejects_malformed_load_and_source_identity(
    tmp_path: Path, mutate, message: str
) -> None:
    row = receipt(7, 1)
    mutate(row)
    log = tmp_path / "quality.log"
    log.write_text(f"DECODE_QUALITY_V2 {json.dumps(row)}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        collector.read_receipts([log])
