#!/usr/bin/env python3
"""Fail closed unless a fresh accelerator log proves SAME-S/L chunking resource bounds."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


RECORD = re.compile(
    r"SA3_CHUNKED_RESOURCE "
    r"case=(?P<case>same_[sl]) mode=(?P<mode>direct|chunked) "
    r"operation=(?P<operation>encode|decode) "
    r"device=(?P<device>\S+) dtype=F32 latent_len=(?P<latent_len>\d+) "
    r"samples=(?P<samples>\d+) dim=(?P<dim>\d+) "
    r"chunk_size=(?P<chunk_size>\d+) overlap=(?P<overlap>\d+) "
    r"starts=(?P<starts>\d+) processed_latents=(?P<processed_latents>\d+) "
    r"load_seconds=(?P<load_seconds>[0-9.]+) "
    r"load_peak_rss_bytes=Some\((?P<load_peak_rss_bytes>\d+)\) "
    r"load_device_bytes=Some\((?P<load_device_bytes>\d+)\) "
    r"elapsed_seconds=(?P<elapsed_seconds>[0-9.]+) "
    r"peak_rss_bytes=Some\((?P<peak_rss_bytes>\d+)\) "
    r"peak_device_bytes=Some\((?P<peak_device_bytes>\d+)\)"
)


class InvalidEvidence(RuntimeError):
    pass


def parse(path: Path) -> dict:
    records = {}
    for match in RECORD.finditer(path.read_text(encoding="utf-8")):
        record = match.groupdict()
        key = (record["case"], record["mode"], record["operation"])
        if key in records:
            raise InvalidEvidence(f"duplicate resource record: {key}")
        for field in (
            "latent_len",
            "samples",
            "dim",
            "chunk_size",
            "overlap",
            "starts",
            "processed_latents",
            "load_peak_rss_bytes",
            "load_device_bytes",
            "peak_rss_bytes",
            "peak_device_bytes",
        ):
            record[field] = int(record[field])
        for field in ("load_seconds", "elapsed_seconds"):
            record[field] = float(record[field])
        records[key] = record
    expected = {
        (case, mode, operation)
        for case in ("same_s", "same_l")
        for mode in ("direct", "chunked")
        for operation in ("encode", "decode")
    }
    if set(records) != expected:
        raise InvalidEvidence(
            f"resource record matrix mismatch: missing={expected - set(records)}, "
            f"extra={set(records) - expected}"
        )
    devices = {record["device"] for record in records.values()}
    if len(devices) != 1 or not next(iter(devices)).startswith("Metal"):
        raise InvalidEvidence(f"expected one asserted Metal backend, got {devices}")
    for record in records.values():
        if (
            record["latent_len"] != 1292
            or record["samples"] != 5_292_032
            or record["chunk_size"] != 128
            or record["overlap"] != 32
        ):
            raise InvalidEvidence(f"resource geometry mismatch: {record}")
        if record["mode"] == "direct":
            if record["starts"] != 1 or record["processed_latents"] != 1292:
                raise InvalidEvidence(f"direct geometry mismatch: {record}")
        elif record["starts"] != 14 or record["processed_latents"] != 1792:
            raise InvalidEvidence(f"chunked geometry mismatch: {record}")
    comparisons = {}
    for case in ("same_s", "same_l"):
        comparisons[case] = {}
        for operation in ("encode", "decode"):
            direct = records[(case, "direct", operation)]
            chunked = records[(case, "chunked", operation)]
            if chunked["peak_device_bytes"] >= direct["peak_device_bytes"]:
                raise InvalidEvidence(
                    f"{case} {operation} did not reduce Metal allocation: "
                    f"{chunked['peak_device_bytes']} >= {direct['peak_device_bytes']}"
                )
            if chunked["elapsed_seconds"] > direct["elapsed_seconds"] * 2.0:
                raise InvalidEvidence(
                    f"{case} {operation} exceeded the independent 2x wall-time guard"
                )
            comparisons[case][operation] = {
                "directLoadDeviceBytes": direct["load_device_bytes"],
                "chunkedLoadDeviceBytes": chunked["load_device_bytes"],
                "directPeakDeviceBytes": direct["peak_device_bytes"],
                "chunkedPeakDeviceBytes": chunked["peak_device_bytes"],
                "deviceReductionBytes": direct["peak_device_bytes"]
                - chunked["peak_device_bytes"],
                "deviceReductionFraction": 1
                - chunked["peak_device_bytes"] / direct["peak_device_bytes"],
                "directPeakRssBytes": direct["peak_rss_bytes"],
                "chunkedPeakRssBytes": chunked["peak_rss_bytes"],
                "directSeconds": direct["elapsed_seconds"],
                "chunkedSeconds": chunked["elapsed_seconds"],
                "wallTimeRatio": chunked["elapsed_seconds"]
                / direct["elapsed_seconds"],
            }
    return {
        "story": "sc-14540",
        "backend": next(iter(devices)),
        "dtype": "F32",
        "geometry": {
            "latentLength": 1292,
            "samples": 5_292_032,
            "chunkSize": 128,
            "overlap": 32,
            "directProcessedLatents": 1292,
            "chunkedProcessedLatents": 1792,
        },
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    evidence = parse(args.log)
    if args.json:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("SAME chunked resource evidence PASS")


if __name__ == "__main__":
    try:
        main()
    except (InvalidEvidence, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
