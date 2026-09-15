# SenseNova infographic contract refusal (sc-23234)

Eight installed-app jobs failed for the four infographic v2/v3 quality/fast routes,
all at bf16, 2048x2048, count 2. The request budget was 126 GiB: 128 GiB total,
28 bytes active, and 2 GiB reserved. Candidate construction succeeded, then the
selector refused the entire provider contract before comparing candidate peaks.

The production calibration fingerprint embedded the route slug `infographic-v2`
or `infographic-v3` and appended formula version `v1`. The shared evidence grammar
requires exactly one positive `vN` token. An installed infographic-v2 metadata test
reproduced the precise conformance error, as did a tiny-artifact unit regression.
The same naming defect exists in Candle production and fixture identities.

Use `infographic2` and `infographic3` only in calibration route slugs. Public model
IDs, repositories, artifact identity, and base quality/fast fingerprints are
unchanged. Validate actual alias-bound production contracts across both backends'
18 route/tier cells, plus Candle fixture contracts. Do not relax the shared grammar.

SceneWorks must update future capture plans and both adapter fingerprint builders.
Keep existing fixture filenames and historical captures/anchors unchanged. Those
records carry their original identities and must not be relabeled as new captures;
requests without matching current evidence use the existing estimate ladder.
