#!/usr/bin/env node
// Validate SC-22261's terminal StarVector admission plan and its one-run receipt. This tooling is
// CI/evidence-only; the shipping MLX and Candle providers remain native Rust and never invoke it.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

const SHA256 = /^[0-9a-f]{64}$/;
const REVISION = /^[0-9a-f]{40}$/;
const REQUIRED_LIMITS = ["complete_root", "eos", "token", "byte", "wall_time", "cancellation"];
const EXPECTED_SOURCES = new Map([
  ["starvector/svg-stack-simple", { revision: "1d2a96a17cc0c4c1f337b7631adc8c5885bc72ea", parquet_sha256: "ed6b73f3c92277e81b244c6ab3071d0831c4820178aed072182246bab402b004", rows: "a8b10c66cdc3135347112d998eddc981e7b41e17133c51bb53b0f8168312dba6" }],
  ["starvector/svg-icons-simple", { revision: "e1918a27ba6649e856e5db0710d8a6c7046762c1", parquet_sha256: "02cd88a8b8f5234975024a948c80a19f7a83092247685b1bef63c3db0b957227", rows: "0a53bf43bdc0a3ba4db43fffdbffceb4f1075accaf928b879ffa7d80d57eb287" }],
  ["starvector/svg-emoji-simple", { revision: "fa75b3617872ae57e6f3cb450aee65dbccbd69e0", parquet_sha256: "be192dfce16b45605df62eebffeb00db2f0b80e6f4919e10cea68c311af97807", rows: "d0744227718ab4f20be9f63c034e5509eacb39bf16c748163952a252174c7846" }],
  ["starvector/svg-fonts-simple", { revision: "453c739ea13ad2685127f721c333f14d99485299", parquet_sha256: "86db32ae45896a18b938baba088b69d797ae2d5f6d3d79742753a0e2ea89d86d", rows: "766de8a23620f100c6b9c3e7ab0bbf0627c9bdc4a6a597a06d03fb101a90f1d9" }],
]);
const EXPECTED_MODELS = new Map([
  ["1b", { key: "starvector-1b-im2svg", repository: "starvector/starvector-1b-im2svg", revision: "380ab95d25a8e9ab1dc825debe238b4953ae13b9" }],
  ["8b", { key: "starvector-8b-im2svg", repository: "starvector/starvector-8b-im2svg", revision: "518beea8dcb5f7a37c5911e92d1d62a76beee7f9" }],
]);
const EXPECTED_RUNS = new Map([
  ["mlx:1b", "mlx-starvector-1b"],
  ["mlx:8b", "mlx-starvector-8b"],
  ["candle-cuda:1b", "candle-starvector-1b"],
  ["candle-cuda:8b", "candle-starvector-8b"],
]);

function fail(message) { throw new Error(`starvector terminal evidence rejected: ${message}`); }
function readJson(path) { try { return JSON.parse(readFileSync(path, "utf8")); } catch (error) { fail(`cannot parse ${path}: ${error.message}`); } }
function hash(value) { return createHash("sha256").update(value).digest("hex"); }
function stable(value) {
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stable(value[key])}`).join(",")}}`;
  return JSON.stringify(value);
}
function onlyKeys(value, keys, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(`${label} must be an object`);
  const actual = Object.keys(value).sort(); const expected = [...keys].sort();
  if (actual.join("|") !== expected.join("|")) fail(`${label} keys differ: ${actual.join(",")}`);
}
function positiveInt(value, label) { if (!Number.isInteger(value) || value < 1) fail(`${label} must be a positive integer`); }
function unitInterval(value, label) { if (typeof value !== "number" || value < 0 || value > 1) fail(`${label} must be in [0, 1]`); }
function sha(value, label) { if (typeof value !== "string" || !SHA256.test(value)) fail(`${label} must be a lowercase SHA-256`); }
function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = sorted.length / 2;
  return sorted.length % 2 ? sorted[Math.floor(middle)] : (sorted[middle - 1] + sorted[middle]) / 2;
}
function p95(values) { return [...values].sort((left, right) => left - right)[Math.ceil(values.length * 0.95) - 1]; }
function pairedBootstrapLowerBound(pairs) {
  // A reproducible one-sided 95% paired bootstrap: 10,000 full-size resamples, Numerical
  // Recipes LCG seed 0x5a17c0de, statistic=(median(1B)-median(8B))/median(1B), 5th percentile.
  let state = 0x5a17c0de; const statistics = [];
  for (let iteration = 0; iteration < 10_000; iteration += 1) {
    const one = []; const eight = [];
    for (let draw = 0; draw < pairs.length; draw += 1) {
      state = (Math.imul(1664525, state) + 1013904223) >>> 0;
      const pair = pairs[state % pairs.length]; one.push(pair.one); eight.push(pair.eight);
    }
    const baseline = median(one);
    if (baseline <= 0) fail("paired 1B bootstrap median LPIPS is not positive");
    statistics.push((baseline - median(eight)) / baseline);
  }
  return statistics.sort((left, right) => left - right)[499];
}

export function validatePlan(corpus) {
  onlyKeys(corpus, ["schema_version", "purpose", "upstream_image_quality_cases", "deterministic_parity_cases", "sceneworks_owned_suites", "excluded_upstream_sources"], "corpus");
  if (corpus.schema_version !== 1) fail("unsupported corpus schema version");
  const quality = corpus.upstream_image_quality_cases;
  onlyKeys(quality, ["required_count", "selection_rule", "row_identity_sha256", "sources"], "image quality corpus");
  if (quality.required_count !== 120) fail("image quality corpus must select exactly 120 cases");
  sha(quality.row_identity_sha256, "image quality row identity");
  if (!Array.isArray(quality.sources) || quality.sources.length !== EXPECTED_SOURCES.size) fail("image quality corpus must declare each expected SVG-Bench source exactly once");
  let count = 0; const seen = new Set();
  for (const source of quality.sources) {
    onlyKeys(source, ["dataset", "revision", "split", "row_start", "row_count", "parquet_path", "parquet_sha256", "row_identity_sha256"], "upstream source");
    const expected = EXPECTED_SOURCES.get(source.dataset);
    if (!expected || seen.has(source.dataset)) fail(`unexpected or duplicate source ${source.dataset}`);
    seen.add(source.dataset);
    if (source.revision !== expected.revision || source.parquet_sha256 !== expected.parquet_sha256 || source.row_identity_sha256 !== expected.rows) fail(`immutable identity changed for ${source.dataset}`);
    if (source.split !== "test" || source.row_start !== 0 || source.row_count !== 30 || source.parquet_path !== "data/test-00000-of-00001.parquet") fail(`${source.dataset} selection is not the exact first thirty test rows`);
    count += source.row_count;
  }
  if (count !== quality.required_count) fail("source rows do not sum to exactly 120");
  if (quality.row_identity_sha256 !== "f9529c2e5a86bef6644054c909c4f621991f6384d9b33a029ad46ff2e6cd3b88") fail("image quality row identity aggregate changed");
  const parity = corpus.deterministic_parity_cases;
  onlyKeys(parity, ["required_count_per_backend", "selection_rule", "row_identity_sha256"], "parity corpus");
  if (parity.required_count_per_backend !== 20 || parity.row_identity_sha256 !== "2e6bc719f3e891ca6e464e6ece2355d2ecf31d607a0353a9ceff69ebdd6d7d15") fail("parity corpus must retain its exact twenty-row identity");
  const owned = corpus.sceneworks_owned_suites;
  onlyKeys(owned, ["hostile_sanitizer", "prompt_composition"], "SceneWorks-owned suites");
  if (owned.hostile_sanitizer.required_count !== 200 || owned.hostile_sanitizer.owner !== "SceneWorks" || owned.prompt_composition.required_count !== 60 || owned.prompt_composition.owner !== "SceneWorks") fail("SceneWorks suite boundary/count changed");
  if (!Array.isArray(corpus.excluded_upstream_sources) || corpus.excluded_upstream_sources.length !== 1 || corpus.excluded_upstream_sources[0].dataset !== "starvector/text2svg-stack" || corpus.excluded_upstream_sources[0].revision !== "c6f2bf0fffd8c1b69fcf748c97f4b0e7de6f2687") fail("text-only upstream corpus exclusion changed");
  return hash(stable(corpus));
}

function validateRun(run) {
  onlyKeys(run, ["backend", "provider_id", "tier", "device", "model", "image_quality", "deterministic_parity", "lifecycle", "limits"], "run");
  const runKey = `${run.backend}:${run.tier}`;
  if (EXPECTED_RUNS.get(runKey) !== run.provider_id || !run.device) fail(`unexpected provider identity ${runKey}`);
  const expectedModel = EXPECTED_MODELS.get(run.tier);
  onlyKeys(run.model, ["key", "repository", "revision", "inventory_sha256"], `${runKey} model`);
  if (run.model.key !== expectedModel.key || run.model.repository !== expectedModel.repository || run.model.revision !== expectedModel.revision) fail(`${runKey} model identity does not match its immutable pin`);
  sha(run.model.inventory_sha256, `${runKey} model inventory`);
  onlyKeys(run.image_quality, ["cases", "memory_headroom_percent"], `${runKey} image quality`);
  if (!Array.isArray(run.image_quality.cases) || run.image_quality.cases.length !== 120) fail(`${runKey} did not record all 120 ordered image cases`);
  const accepted = [];
  const latencies = [];
  for (const [index, record] of run.image_quality.cases.entries()) {
    onlyKeys(record, ["case_index", "accepted", "ssim", "lpips", "latency_seconds"], `${runKey} case ${index}`);
    if (record.case_index !== index || typeof record.accepted !== "boolean" || typeof record.latency_seconds !== "number" || record.latency_seconds < 0) fail(`${runKey} image-case order or latency is invalid`);
    if (!record.accepted) {
      if (record.ssim !== null || record.lpips !== null) fail(`${runKey} rejected case ${index} carries fabricated quality metrics`);
    } else {
      unitInterval(record.ssim, `${runKey} case ${index} SSIM`); unitInterval(record.lpips, `${runKey} case ${index} LPIPS`);
      accepted.push(record);
    }
    latencies.push(record.latency_seconds);
  }
  if (accepted.length < 114) fail(`${runKey} validity is below 95%`);
  const metrics = { validity_rate: accepted.length / 120, median_ssim: median(accepted.map((record) => record.ssim)), median_lpips: median(accepted.map((record) => record.lpips)), p95_latency_seconds: p95(latencies), cases: run.image_quality.cases };
  if (metrics.median_ssim < 0.85 || metrics.median_lpips > 0.20) fail(`${runKey} image-quality threshold failed`);
  if (metrics.p95_latency_seconds > 120) fail(`${runKey} p95 latency exceeds 120 seconds`);
  if (typeof run.image_quality.memory_headroom_percent !== "number" || run.image_quality.memory_headroom_percent < (run.tier === "1b" ? 15 : 10)) fail(`${runKey} memory-headroom threshold failed`);
  onlyKeys(run.deterministic_parity, ["case_count", "rendered_ssim"], `${runKey} parity`);
  if (run.deterministic_parity.case_count !== 20 || !Array.isArray(run.deterministic_parity.rendered_ssim) || run.deterministic_parity.rendered_ssim.length !== 20 || run.deterministic_parity.rendered_ssim.some((value) => typeof value !== "number" || value < 0.995 || value > 1)) fail(`${runKey} deterministic rendered-SSIM threshold failed`);
  onlyKeys(run.lifecycle, ["load", "unload", "reload", "memory_reported"], `${runKey} lifecycle`);
  if (Object.values(run.lifecycle).some((value) => value !== true)) fail(`${runKey} lifecycle evidence is incomplete`);
  onlyKeys(run.limits, REQUIRED_LIMITS, `${runKey} limits`);
  if (Object.values(run.limits).some((value) => value !== true)) fail(`${runKey} limit outcome evidence is incomplete`);
  return { runKey, inventory: run.model.inventory_sha256, metrics };
}

export function validateReceipt(receipt, corpusHash, inferenceRevision, sceneworksRevision) {
  onlyKeys(receipt, ["schema_version", "campaign_run_id", "inference_revision", "sceneworks_revision", "corpus_sha256", "runs"], "receipt");
  if (receipt.schema_version !== 1 || !receipt.campaign_run_id) fail("receipt version or run id is missing");
  if (receipt.inference_revision !== inferenceRevision || receipt.sceneworks_revision !== sceneworksRevision || !REVISION.test(receipt.inference_revision) || !REVISION.test(receipt.sceneworks_revision)) fail("receipt has a mixed inference/SceneWorks revision");
  if (receipt.corpus_sha256 !== corpusHash) fail("receipt corpus hash does not match the exact selected corpus");
  if (!Array.isArray(receipt.runs) || receipt.runs.length !== EXPECTED_RUNS.size) fail("receipt must include exactly four backend/tier runs");
  const seen = new Set(); const inventories = new Map(); const analyses = new Map();
  for (const run of receipt.runs) {
    const result = validateRun(run);
    if (seen.has(result.runKey)) fail(`duplicate backend/tier evidence ${result.runKey}`);
    seen.add(result.runKey);
    const prior = inventories.get(run.tier);
    if (prior && prior !== result.inventory) fail(`mixed snapshot inventory for ${run.tier}`);
    inventories.set(run.tier, result.inventory);
    analyses.set(result.runKey, result);
  }
  if (seen.size !== EXPECTED_RUNS.size || [...EXPECTED_RUNS].some(([key]) => !seen.has(key))) fail("missing required MLX/Candle run");
  for (const backend of ["mlx", "candle-cuda"]) {
    const one = analyses.get(`${backend}:1b`).metrics;
    const eight = analyses.get(`${backend}:8b`).metrics;
    if (one.median_lpips <= 0) fail(`8B ${backend} has no comparable 1B median LPIPS`);
    const improvement = (one.median_lpips - eight.median_lpips) / one.median_lpips;
    const validityDelta = eight.validity_rate - one.validity_rate;
    const pairs = one.cases.flatMap((record, index) => record.accepted && eight.cases[index].accepted ? [{ one: record.lpips, eight: eight.cases[index].lpips }] : []);
    if (pairs.length < 114 || improvement < 0.10 || validityDelta < -0.02 || pairedBootstrapLowerBound(pairs) <= 0) fail(`8B ${backend} LPIPS/validity/bootstrap threshold failed`);
  }
}

function option(name) { const index = process.argv.indexOf(name); if (index < 0 || !process.argv[index + 1]) fail(`missing ${name}`); return process.argv[index + 1]; }
function main() {
  const command = process.argv[2];
  if (command === "validate-plan") { console.log(`corpus_sha256=${validatePlan(readJson(option("--corpus")))}`); return; }
  if (command === "validate-receipt") {
    const corpusHash = validatePlan(readJson(option("--corpus")));
    validateReceipt(readJson(option("--receipt")), corpusHash, option("--inference-revision"), option("--sceneworks-revision"));
    console.log("starvector terminal receipt: OK"); return;
  }
  fail("usage: validate-plan --corpus FILE | validate-receipt --corpus FILE --receipt FILE --inference-revision SHA --sceneworks-revision SHA");
}
if (import.meta.url === `file://${process.argv[1]}`) { try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; } }
