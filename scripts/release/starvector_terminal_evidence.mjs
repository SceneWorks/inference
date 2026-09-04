#!/usr/bin/env node
// SC-22261 terminal evidence validator. CI-only; shipping providers stay native Rust.
import { createHash } from "node:crypto";
import { readFileSync, statSync } from "node:fs";

const SHA = /^[0-9a-f]{64}$/;
const REV = /^[0-9a-f]{40}$/;
const CAMPAIGN = /^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$/;
const DECIMAL_ID = /^[1-9][0-9]{0,15}$/;
const ARTIFACT_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$/;
const GITHUB_DIGEST = /^sha256:([0-9a-f]{64})$/;
const CURRENT_WORKFLOW_PATH = ".github/workflows/starvector-terminal.yml";
const MAX_SAFE_SIZE = Number.MAX_SAFE_INTEGER;
const MAX_ATTEMPT = 1000;
const MAX_LINEAGE_PREDECESSORS = 32;
const MAX_SOURCE_ARTIFACTS = 32;
const MAX_CONTENT_INVENTORY_ENTRIES = 20000;
const MAX_MANIFEST_ENTRIES = 100000;
export const MAX_RECEIPT_BYTES = 8 * 1024 * 1024;
const MAX_CORPUS_BYTES = 1024 * 1024;
const FAILURE_PHASES = new Set(["admission", "setup", "provisioning", "execution", "collection", "validation", "publication", "cleanup", "workflow"]);
const FAILURE_TUPLES = new Set(["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"]);
const NON_SUCCESS_CONCLUSIONS = new Set(["failure", "cancelled", "timed_out", "action_required", "stale", "startup_failure"]);
const LIMITS = ["complete_root", "eos", "token", "byte", "wall_time", "cancellation"];
const FINISH_REASONS = ["complete_root", "eos", "token_limit", "byte_limit", "wall_time_limit", "cancelled"];
const SOURCES = [
  ["starvector/svg-stack-simple", "1d2a96a17cc0c4c1f337b7631adc8c5885bc72ea", "ed6b73f3c92277e81b244c6ab3071d0831c4820178aed072182246bab402b004", "a8b10c66cdc3135347112d998eddc981e7b41e17133c51bb53b0f8168312dba6"],
  ["starvector/svg-icons-simple", "e1918a27ba6649e856e5db0710d8a6c7046762c1", "02cd88a8b8f5234975024a948c80a19f7a83092247685b1bef63c3db0b957227", "0a53bf43bdc0a3ba4db43fffdbffceb4f1075accaf928b879ffa7d80d57eb287"],
  ["starvector/svg-emoji-simple", "fa75b3617872ae57e6f3cb450aee65dbccbd69e0", "be192dfce16b45605df62eebffeb00db2f0b80e6f4919e10cea68c311af97807", "d0744227718ab4f20be9f63c034e5509eacb39bf16c748163952a252174c7846"],
  ["starvector/svg-fonts-simple", "453c739ea13ad2685127f721c333f14d99485299", "86db32ae45896a18b938baba088b69d797ae2d5f6d3d79742753a0e2ea89d86d", "766de8a23620f100c6b9c3e7ab0bbf0627c9bdc4a6a597a06d03fb101a90f1d9"],
];
const MODELS = new Map([["1b", ["starvector-1b-im2svg", "starvector/starvector-1b-im2svg", "380ab95d25a8e9ab1dc825debe238b4953ae13b9"]], ["8b", ["starvector-8b-im2svg", "starvector/starvector-8b-im2svg", "518beea8dcb5f7a37c5911e92d1d62a76beee7f9"]]]);
const RUNS = new Map([["mlx:1b", "mlx-starvector-1b"], ["mlx:8b", "mlx-starvector-8b"], ["candle-cuda:1b", "candle-starvector-1b"], ["candle-cuda:8b", "candle-starvector-8b"]]);
const HOSTILE = ["structure-variants", "doctype-entity", "pi-cdata", "script", "foreign-object", "event-handler", "css-import", "animation", "external-href", "data-href", "file-href", "use", "text", "byte-overrun", "node-overrun", "depth-overrun", "attribute-total-value-overrun", "path-command-number-overrun", "points-transform-overrun", "coordinate-dimension-viewbox-overrun"];
const PROMPTS = ["geometric badge", "isometric folder", "rounded calendar", "minimal rocket", "layered landscape", "abstract flower"];
const fail = (message) => { throw new Error(`starvector terminal evidence rejected: ${message}`); };
const hash = (value) => createHash("sha256").update(value).digest("hex");
const stable = (value) => Array.isArray(value) ? `[${value.map(stable).join(",")}]` : value && typeof value === "object" ? `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stable(value[key])}`).join(",")}}` : JSON.stringify(value);
const keys = (value, expected, label) => { if (!value || typeof value !== "object" || Array.isArray(value)) fail(`${label} must be object`); if (Object.keys(value).sort().join("|") !== [...expected].sort().join("|")) fail(`${label} keys differ`); };
const sha = (value, label) => { if (typeof value !== "string" || !SHA.test(value)) fail(`${label} must be SHA-256`); };
const positive = (value, label) => { if (!Number.isInteger(value) || value < 1) fail(`${label} must be positive integer`); };
const number = (value, label, min = 0, max = 1) => { if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) fail(`${label} out of range`); };
const median = (values) => { const sorted = [...values].sort((a, b) => a - b); return sorted.length % 2 ? sorted[Math.floor(sorted.length / 2)] : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2; };
const p95 = (values) => [...values].sort((a, b) => a - b)[Math.ceil(values.length * .95) - 1];
export function hostilePayload(index) { const n = index % 10, kind = HOSTILE[Math.floor(index / 10)], long = (char, length) => char.repeat(length + n); const payloads = { "structure-variants": [() => `noise-${n}<svg/>`, () => `<svg/>tail-${n}`, () => `<svg/><svg id="${n}"/>`, () => `<svg><path d="M${n}"`][n % 4](), "doctype-entity": n % 2 ? `<!DOCTYPE svg [<!ENTITY x "${n}">]><svg>&x;</svg>` : `<!DOCTYPE svg SYSTEM "https://invalid/${n}.dtd"><svg/>`, "pi-cdata": n % 2 ? `<?xml-stylesheet href="https://invalid/${n}"?><svg/>` : `<svg><![CDATA[<script>${n}</script>]]></svg>`, script: `<svg><script>x${n}()</script></svg>`, "foreign-object": `<svg><foreignObject>${n}</foreignObject></svg>`, "event-handler": `<svg onload="x${n}()"/>`, "css-import": `<svg><style>@import url(https://invalid/${n})</style></svg>`, animation: `<svg><animate attributeName="x" values="0;${n}"/></svg>`, "external-href": `<svg><a href="https://invalid/${n}"><path/></a></svg>`, "data-href": `<svg><image href="data:image/svg+xml,${n}"/></svg>`, "file-href": `<svg><image href="file:///tmp/${n}"/></svg>`, use: `<svg><use href="https://invalid/${n}.svg#x"/></svg>`, text: `<svg><text>${n}</text></svg>`, "byte-overrun": `<svg>${long("x", 262145)}</svg>`, "node-overrun": `<svg>${"<g/>".repeat(2001 + n)}</svg>`, "depth-overrun": `<svg>${"<g>".repeat(33 + n)}${"</g>".repeat(33 + n)}</svg>`, "attribute-total-value-overrun": n % 2 ? `<svg><path id="${long("x", 262145)}"/></svg>` : `<svg><path id="${long("i", 70000)}" class="${long("c", 70000)}" fill="${long("f", 70000)}" stroke="${long("s", 70000)}"/></svg>`, "path-command-number-overrun": n % 2 ? `<svg><path d="${"M0 0 ".repeat(100001 + n)}"/></svg>` : `<svg><path d="M${"1 ".repeat(200001 + n)}0"/></svg>`, "points-transform-overrun": n % 2 ? `<svg><polygon points="${"0,0 ".repeat(100001 + n)}"/></svg>` : `<svg><path transform="${"translate(1 1) ".repeat(100001 + n)}"/></svg>`, "coordinate-dimension-viewbox-overrun": [() => `<svg width="${1000000 + n}"/>`, () => `<svg height="${1000000 + n}"/>`, () => `<svg viewBox="0 0 ${1000000 + n} ${1000000 + n}"/>`, () => `<svg><path d="M${1000000 + n} ${1000000 + n}"/></svg>`, () => `<svg viewBox="${1000000 + n} ${1000000 + n} 10 10"/>`][n % 5]() }; return payloads[kind]; }
const promptPayload = (index) => `Create a ${PROMPTS[Math.floor(index / 10)]} vector illustration, variant ${index % 10}, with clear silhouette, balanced composition, and no text.`;
const OWNED_CASES = new Map();
function owned(kind) { if (!OWNED_CASES.has(kind)) { const count = kind === "hostile" ? HOSTILE.length * 10 : 60; const payload = kind === "hostile" ? hostilePayload : promptPayload; OWNED_CASES.set(kind, Array.from({ length: count }, (_, case_index) => ({ case_index, case_id: `${kind}-v1-${case_index}`, input_sha256: hash(payload(case_index)) }))); } return OWNED_CASES.get(kind); }
const ownedHash = (kind) => hash(owned(kind).map((entry) => entry.input_sha256).join("\n"));
const sourceFor = (index) => ({ dataset: SOURCES[Math.floor(index / 30)][0], revision: SOURCES[Math.floor(index / 30)][1], row_index: index % 30 });
function ref(refs, path, digest) { sha(digest, path); refs.push({ path, sha256: digest }); }
function boundedString(value, label, maximum) { if (typeof value !== "string" || value.length < 1 || value.length > maximum) fail(`${label} length invalid`); return value; }
function boundedInteger(value, label, minimum = 0, maximum = MAX_SAFE_SIZE) { if (!Number.isSafeInteger(value) || value < minimum || value > maximum) fail(`${label} must be bounded safe integer`); return value; }
function decimalId(value, label) { if (typeof value !== "string" || !DECIMAL_ID.test(value) || BigInt(value) > BigInt(Number.MAX_SAFE_INTEGER)) fail(`${label} must be canonical bounded decimal id`); return value; }
function campaignId(value, label) { if (typeof value !== "string" || !CAMPAIGN.test(value)) fail(`${label} must be safe campaign id`); return value; }
function canonicalPath(value, label) {
  if (typeof value !== "string" || !value || value.length > 1024 || value.startsWith("/") || value.endsWith("/") || value.includes("\\") || value.includes("//")) fail(`${label} must be safe canonical path`);
  const segments = value.split("/");
  if (segments.some((segment) => segment.length > 255 || segment === "." || segment === ".." || !/^[A-Za-z0-9._:-]+$/.test(segment))) fail(`${label} must be safe canonical path`);
  return value;
}
function sizedHash(value, label) { keys(value, ["path", "size", "sha256"], label); canonicalPath(value.path, `${label} path`); boundedInteger(value.size, `${label} size`, 1); sha(value.sha256, `${label} digest`); return value; }
function contentEntry(value, label) { keys(value, ["path", "byte_size", "sha256"], label); canonicalPath(value.path, `${label} path`); boundedInteger(value.byte_size, `${label} byte size`); sha(value.sha256, `${label} digest`); return value; }
function sortedEntries(entries) { return [...entries].sort((left, right) => left.path.localeCompare(right.path)); }
function sameEntries(left, right) { return JSON.stringify(left) === JSON.stringify(right); }
export function campaignLineageSha256(value) { return hash(stable(value)); }
export function pairedBootstrapLowerBound(pairs) { let state = 0x5a17c0de; const stats = []; for (let i = 0; i < 10000; i += 1) { const one = [], eight = []; for (let draw = 0; draw < pairs.length; draw += 1) { state = (Math.imul(1664525, state) + 1013904223) >>> 0; const pair = pairs[state % pairs.length]; one.push(pair.one); eight.push(pair.eight); } const base = median(one); if (base <= 0) fail("bootstrap baseline invalid"); stats.push((base - median(eight)) / base); } return stats.sort((a, b) => a - b)[499]; }

export function validatePlan(corpus) {
  keys(corpus, ["schema_version", "purpose", "upstream_image_quality_cases", "deterministic_parity_cases", "sceneworks_owned_suites", "excluded_upstream_sources"], "corpus");
  const quality = corpus.upstream_image_quality_cases; keys(quality, ["required_count", "selection_rule", "row_identity_sha256", "sources"], "quality corpus");
  if (corpus.schema_version !== 1 || quality.required_count !== 120 || quality.row_identity_sha256 !== "f9529c2e5a86bef6644054c909c4f621991f6384d9b33a029ad46ff2e6cd3b88" || !Array.isArray(quality.sources) || quality.sources.length !== 4) fail("quality corpus identity changed");
  quality.sources.forEach((source, index) => { keys(source, ["dataset", "revision", "split", "row_start", "row_count", "parquet_path", "parquet_sha256", "row_identity_sha256"], "source"); const expected = SOURCES[index]; if (!expected || source.dataset !== expected[0] || source.revision !== expected[1] || source.parquet_sha256 !== expected[2] || source.row_identity_sha256 !== expected[3] || source.split !== "test" || source.row_start !== 0 || source.row_count !== 30 || source.parquet_path !== "data/test-00000-of-00001.parquet") fail("immutable source identity changed"); });
  const parity = corpus.deterministic_parity_cases; keys(parity, ["required_count_per_backend", "selection_rule", "row_identity_sha256"], "parity corpus"); if (parity.required_count_per_backend !== 20 || parity.row_identity_sha256 !== "2e6bc719f3e891ca6e464e6ece2355d2ecf31d607a0353a9ceff69ebdd6d7d15") fail("parity corpus identity changed");
  const suites = corpus.sceneworks_owned_suites; keys(suites, ["hostile_sanitizer", "prompt_composition"], "owned suites"); for (const [name, kind, count, generator] of [["hostile_sanitizer", "hostile", 200, "starvector-terminal-hostile-v1"], ["prompt_composition", "prompt", 60, "starvector-terminal-prompt-v1"]]) { const suite = suites[name]; keys(suite, ["required_count", "owner", "purpose", "generator", "content_identity_sha256", "selection_rule"], `${name} suite`); if (suite.required_count !== count || suite.owner !== "SceneWorks" || suite.generator !== generator || suite.content_identity_sha256 !== ownedHash(kind)) fail(`${name} content identity changed`); }
  if (!Array.isArray(corpus.excluded_upstream_sources) || corpus.excluded_upstream_sources.length !== 1 || corpus.excluded_upstream_sources[0].dataset !== "starvector/text2svg-stack" || corpus.excluded_upstream_sources[0].revision !== "c6f2bf0fffd8c1b69fcf748c97f4b0e7de6f2687") fail("excluded source changed");
  return hash(stable(corpus));
}
function hardware(value, runKey, tier, refs) { keys(value, ["runner_name", "os", "arch", "system_memory_total_bytes", "baseline_available_bytes", "peak_process_rss_bytes", "accelerator"], `${runKey} hardware`); ["runner_name", "os", "arch"].forEach((key) => { if (typeof value[key] !== "string" || !value[key]) fail(`${runKey} missing ${key}`); }); ["system_memory_total_bytes", "baseline_available_bytes", "peak_process_rss_bytes"].forEach((key) => positive(value[key], `${runKey} ${key}`)); keys(value.accelerator, ["name", "uuid", "driver_runtime", "total_bytes", "baseline_free_bytes", "peak_used_bytes", "raw_probe_sha256"], `${runKey} accelerator`); if (typeof value.accelerator.name !== "string" || !value.accelerator.name || !(value.accelerator.uuid === null || typeof value.accelerator.uuid === "string") || typeof value.accelerator.driver_runtime !== "string" || !value.accelerator.driver_runtime) fail(`${runKey} accelerator identity invalid`); ["total_bytes", "baseline_free_bytes", "peak_used_bytes"].forEach((key) => positive(value.accelerator[key], `${runKey} ${key}`)); if (value.accelerator.baseline_free_bytes > value.accelerator.total_bytes || value.accelerator.peak_used_bytes > value.accelerator.total_bytes || ((value.accelerator.total_bytes - value.accelerator.peak_used_bytes) / value.accelerator.total_bytes) * 100 < (tier === "1b" ? 15 : 10)) fail(`${runKey} memory-headroom threshold failed`); ref(refs, `runs/${runKey}/hardware/raw-probe`, value.accelerator.raw_probe_sha256); }
function run(value, refs) {
  keys(value, ["backend", "provider_id", "tier", "device", "model", "hardware", "image_quality", "deterministic_parity", "lifecycle", "limits", "lifecycle_memory_transcript_sha256"], "run"); const runKey = `${value.backend}:${value.tier}`; if (RUNS.get(runKey) !== value.provider_id || typeof value.device !== "string" || !value.device) fail(`provider identity ${runKey}`); const model = MODELS.get(value.tier); keys(value.model, ["key", "repository", "revision", "inventory_sha256"], `${runKey} model`); if (!model || value.model.key !== model[0] || value.model.repository !== model[1] || value.model.revision !== model[2]) fail(`${runKey} model identity`); ref(refs, `runs/${runKey}/lifecycle-memory`, value.lifecycle_memory_transcript_sha256); hardware(value.hardware, runKey, value.tier, refs);
  keys(value.image_quality, ["cases"], `${runKey} quality`); if (!Array.isArray(value.image_quality.cases) || value.image_quality.cases.length !== 120) fail(`${runKey} needs 120 ordered cases`); const accepted = [], latencies = [];
  value.image_quality.cases.forEach((record, index) => { keys(record, ["case_index", "source", "source_svg_sha256", "input_png_sha256", "provider_transcript_sha256", "finish_reason", "canonical_svg_sha256", "preview_png_sha256", "accepted", "ssim", "lpips", "latency_seconds"], `${runKey} case`); const expected = sourceFor(index); keys(record.source, ["dataset", "revision", "row_index"], `${runKey} source`); if (!FINISH_REASONS.includes(record.finish_reason) || record.case_index !== index || record.source.dataset !== expected.dataset || record.source.revision !== expected.revision || record.source.row_index !== expected.row_index || typeof record.accepted !== "boolean") fail(`${runKey} case order/identity invalid`); ["source_svg_sha256", "input_png_sha256", "provider_transcript_sha256"].forEach((key) => ref(refs, `runs/${runKey}/cases/${index}/${key}`, record[key])); if (typeof record.latency_seconds !== "number" || record.latency_seconds < 0) fail(`${runKey} latency invalid`); if (record.accepted) { if (!["complete_root", "eos"].includes(record.finish_reason)) fail(`${runKey} accepted case has non-complete finish`); number(record.ssim, `${runKey} SSIM`); number(record.lpips, `${runKey} LPIPS`); ref(refs, `runs/${runKey}/cases/${index}/canonical`, record.canonical_svg_sha256); ref(refs, `runs/${runKey}/cases/${index}/preview`, record.preview_png_sha256); accepted.push(record); } else if (record.ssim !== null || record.lpips !== null || record.canonical_svg_sha256 !== null || record.preview_png_sha256 !== null) fail(`${runKey} rejected case carries output`); latencies.push(record.latency_seconds); });
  if (accepted.length < 114 || median(accepted.map((item) => item.ssim)) < .85 || median(accepted.map((item) => item.lpips)) > .20 || p95(latencies) > 120) fail(`${runKey} image threshold failed`); keys(value.deterministic_parity, ["case_count", "cases"], `${runKey} parity`); if (value.deterministic_parity.case_count !== 20 || !Array.isArray(value.deterministic_parity.cases) || value.deterministic_parity.cases.length !== 20) fail(`${runKey} parity count`); value.deterministic_parity.cases.forEach((record, index) => { keys(record, ["case_index", "seed", "first_preview_png_sha256", "second_preview_png_sha256", "rendered_ssim"], `${runKey} parity case`); if (record.case_index !== index || !Number.isInteger(record.seed) || record.seed < 0) fail(`${runKey} parity identity`); ref(refs, `runs/${runKey}/parity/${index}/first`, record.first_preview_png_sha256); ref(refs, `runs/${runKey}/parity/${index}/second`, record.second_preview_png_sha256); number(record.rendered_ssim, `${runKey} parity SSIM`); if (record.rendered_ssim < .995) fail(`${runKey} parity threshold`); }); keys(value.lifecycle, ["load", "unload", "reload", "memory_reported"], `${runKey} lifecycle`); if (Object.values(value.lifecycle).some((item) => item !== true)) fail(`${runKey} lifecycle incomplete`); keys(value.limits, LIMITS, `${runKey} limits`); if (Object.values(value.limits).some((item) => item !== true)) fail(`${runKey} limits incomplete`); return { runKey, inventory: value.model.inventory_sha256, validity: accepted.length / 120, median_lpips: median(accepted.map((item) => item.lpips)), cases: value.image_quality.cases };
}
function hostile(value, corpus, refs) { keys(value, ["corpus_sha256", "sanitizer_version", "cases"], "hostile"); const suite = corpus.sceneworks_owned_suites.hostile_sanitizer; if (value.corpus_sha256 !== suite.content_identity_sha256 || typeof value.sanitizer_version !== "string" || !value.sanitizer_version || !Array.isArray(value.cases) || value.cases.length !== 200) fail("hostile corpus/count invalid"); value.cases.forEach((record, index) => { keys(record, ["case_index", "case_id", "input_sha256", "expected_policy", "outcome", "error_code", "canonical_svg_sha256", "preview_png_sha256", "published_paths", "staging_residue", "result_contains_inline_svg"], "hostile case"); const expected = owned("hostile")[index]; if (record.case_index !== index || record.case_id !== expected.case_id || record.input_sha256 !== expected.input_sha256 || record.expected_policy !== "reject_or_sanitize_inert" || !["rejected", "sanitized_inert"].includes(record.outcome) || typeof record.error_code !== "string" || record.result_contains_inline_svg !== false || !Array.isArray(record.published_paths) || !Array.isArray(record.staging_residue)) fail("hostile evidence invalid"); ref(refs, `hostile/${index}/input`, record.input_sha256); if (record.outcome === "rejected") { if (record.canonical_svg_sha256 !== null || record.preview_png_sha256 !== null || record.published_paths.length !== 0 || record.staging_residue.length !== 0) fail("hostile rejected artifact"); } else { if (record.published_paths.join("|") !== "canonical.svg|preview.png" || record.staging_residue.length !== 0) fail("hostile inert publication invalid"); ref(refs, `hostile/${index}/canonical`, record.canonical_svg_sha256); ref(refs, `hostile/${index}/preview`, record.preview_png_sha256); } }); }
function prompt(value, corpus, refs) { keys(value, ["corpus_sha256", "raster_provider_id", "raster_model", "raster_revision", "raster_inventory_sha256", "clip_provider_id", "clip_model", "clip_revision", "clip_inventory_sha256", "metric_transcript_sha256", "cases"], "prompt"); const suite = corpus.sceneworks_owned_suites.prompt_composition; if (value.corpus_sha256 !== suite.content_identity_sha256 || !Array.isArray(value.cases) || value.cases.length !== 60) fail("prompt corpus/count invalid"); ["raster_provider_id", "raster_model", "raster_revision", "clip_provider_id", "clip_model", "clip_revision"].forEach((key) => { if (typeof value[key] !== "string" || !value[key]) fail(`prompt ${key} missing`); }); ["raster_inventory_sha256", "clip_inventory_sha256", "metric_transcript_sha256"].forEach((key) => ref(refs, `prompt/${key}`, value[key])); const accepted = []; value.cases.forEach((record, index) => { keys(record, ["case_index", "case_id", "prompt_sha256", "raster_png_sha256", "vector_provider_transcript_sha256", "canonical_svg_sha256", "preview_png_sha256", "accepted", "raster_prompt_cosine", "preview_prompt_cosine", "alignment_loss"], "prompt case"); const expected = owned("prompt")[index]; if (record.case_index !== index || record.case_id !== expected.case_id || record.prompt_sha256 !== expected.input_sha256 || typeof record.accepted !== "boolean") fail("prompt identity invalid"); ["prompt_sha256", "raster_png_sha256", "vector_provider_transcript_sha256"].forEach((key) => ref(refs, `prompt/${index}/${key}`, record[key])); if (record.accepted) { number(record.raster_prompt_cosine, "raster cosine", -1, 1); number(record.preview_prompt_cosine, "preview cosine", -1, 1); if (typeof record.alignment_loss !== "number" || record.alignment_loss < -2 || record.alignment_loss > 2 || Math.abs(record.alignment_loss - (record.raster_prompt_cosine - record.preview_prompt_cosine)) > 1e-12) fail("forged alignment loss"); ref(refs, `prompt/${index}/canonical`, record.canonical_svg_sha256); ref(refs, `prompt/${index}/preview`, record.preview_png_sha256); accepted.push(record); } else if (record.canonical_svg_sha256 !== null || record.preview_png_sha256 !== null || record.raster_prompt_cosine !== null || record.preview_prompt_cosine !== null || record.alignment_loss !== null) fail("rejected prompt carries metrics/output"); }); if (accepted.length < 57 || median(accepted.map((item) => item.alignment_loss)) > .02) fail("prompt median alignment threshold failed"); }
function metric(value, refs) { keys(value, ["rasterizer", "canvas", "ssim", "lpips", "metric_transcript_sha256"], "metric identity"); if (value.rasterizer !== "resvg-0.45") fail("rasterizer must be resvg-0.45"); keys(value.canvas, ["width", "height", "background", "colorspace"], "metric canvas"); if (value.canvas.width !== 512 || value.canvas.height !== 512 || value.canvas.background !== "white" || value.canvas.colorspace !== "srgb8") fail("metric canvas must be 512px white sRGB8"); keys(value.ssim, ["implementation", "package_version", "lock_sha256", "data_range", "channel_axis", "gaussian_weights", "sigma", "use_sample_covariance"], "SSIM identity"); if (value.ssim.implementation !== "skimage.metrics.structural_similarity" || typeof value.ssim.package_version !== "string" || !value.ssim.package_version || value.ssim.data_range !== 255 || value.ssim.channel_axis !== 2 || value.ssim.gaussian_weights !== true || value.ssim.sigma !== 1.5 || value.ssim.use_sample_covariance !== false) fail("SSIM identity/settings invalid"); keys(value.lpips, ["implementation", "package_version", "version", "net", "eval_mode", "rgb_normalization", "lock_sha256", "linear_weights_sha256", "alexnet_weights_sha256"], "LPIPS identity"); if (value.lpips.implementation !== "richzhang/lpips" || typeof value.lpips.package_version !== "string" || !value.lpips.package_version || value.lpips.version !== "0.1" || value.lpips.net !== "alex" || value.lpips.eval_mode !== true || value.lpips.rgb_normalization !== "[-1,1]" || value.lpips.linear_weights_sha256 !== "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0" || value.lpips.alexnet_weights_sha256 !== "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02") fail("LPIPS identity/settings invalid"); [value.ssim.lock_sha256, value.lpips.lock_sha256, value.lpips.linear_weights_sha256, value.lpips.alexnet_weights_sha256, value.metric_transcript_sha256].forEach((digest, index) => ref(refs, `metrics/${index}`, digest)); }
function preflight(value, revision, refs) { keys(value, ["workflow_run_id", "workflow_run_attempt", "head_sha", "inventory_artifacts", "hook_logs"], "preflight"); if (value.head_sha !== revision || !REV.test(value.head_sha) || typeof value.workflow_run_id !== "string" || !value.workflow_run_id || !Number.isInteger(value.workflow_run_attempt) || value.workflow_run_attempt < 1 || !Array.isArray(value.inventory_artifacts) || value.inventory_artifacts.length !== 2 || !Array.isArray(value.hook_logs) || value.hook_logs.length !== 4) fail("preflight invalid"); const tiers = new Set(), hooks = new Set(); value.inventory_artifacts.forEach((entry) => { keys(entry, ["tier", "sha256"], "inventory artifact"); if (!MODELS.has(entry.tier) || tiers.has(entry.tier)) fail("inventory artifact tier"); tiers.add(entry.tier); ref(refs, `preflight/inventory/${entry.tier}`, entry.sha256); }); value.hook_logs.forEach((entry) => { keys(entry, ["backend", "tier", "sha256"], "hook log"); const key = `${entry.backend}:${entry.tier}`; if (!RUNS.has(key) || hooks.has(key)) fail("hook log identity"); hooks.add(key); ref(refs, `preflight/hook/${key}`, entry.sha256); }); }
function manifest(value, id, refs) { keys(value, ["campaign_run_id", "entries", "aggregate_sha256"], "artifact manifest"); if (value.campaign_run_id !== id || !Array.isArray(value.entries)) fail("artifact manifest mixed run"); const entries = new Map(); value.entries.forEach((entry) => { keys(entry, ["path", "byte_size", "sha256"], "artifact entry"); if (typeof entry.path !== "string" || !entry.path || !Number.isInteger(entry.byte_size) || entry.byte_size < 0 || entries.has(entry.path)) fail("artifact entry invalid"); sha(entry.sha256, "artifact checksum"); entries.set(entry.path, entry.sha256); }); if (value.aggregate_sha256 !== hash(stable({ campaign_run_id: value.campaign_run_id, entries: value.entries }))) fail("artifact manifest checksum"); refs.forEach((entry) => { if (entries.get(entry.path) !== entry.sha256) fail(`artifact manifest missing/mixed ${entry.path}`); }); return value.aggregate_sha256; }
function currentEvidenceRefs(receipt, corpus) { const refs = []; metric(receipt.metric_identity, refs); preflight(receipt.inference_preflight, receipt.inference_revision, refs); ref(refs, "producer/transcript", receipt.producer.transcript_sha256); receipt.runs.forEach((entry) => run(entry, refs)); hostile(receipt.hostile_sanitizer, corpus, refs); prompt(receipt.prompt_composition, corpus, refs); return refs; }
function artifactWorkflowRun(value, artifact, label) {
  keys(value, ["id", "head_sha"], label);
  if (decimalId(value.id, `${label} id`) !== artifact.workflow_run_id || value.head_sha !== artifact.head_sha) fail(`${label} does not bind artifact workflow run`);
}
function githubArtifact(value, predecessor, label) {
  keys(value, ["role", "repository", "workflow_run_id", "workflow_run_attempt", "head_sha", "api_workflow_run", "id", "name", "size", "digest", "content_inventory"], label);
  if (!/^[a-z][a-z0-9-]{0,63}$/.test(boundedString(value.role, `${label} role`, 64)) || value.repository !== predecessor.workflow.repository || decimalId(value.workflow_run_id, `${label} workflow run id`) !== predecessor.workflow.run_id || boundedInteger(value.workflow_run_attempt, `${label} workflow attempt`, 1, MAX_ATTEMPT) !== predecessor.workflow.run_attempt || value.head_sha !== predecessor.workflow.head_sha || !ARTIFACT_NAME.test(boundedString(value.name, `${label} name`, 255))) fail(`${label} identity/provenance invalid`);
  decimalId(value.id, `${label} id`);
  boundedInteger(value.size, `${label} size`, 1);
  const digest = typeof value.digest === "string" ? GITHUB_DIGEST.exec(value.digest) : null;
  if (!digest) fail(`${label} digest invalid`);
  artifactWorkflowRun(value.api_workflow_run, value, `${label} API workflow run`);
  if (!Array.isArray(value.content_inventory) || value.content_inventory.length < 1 || value.content_inventory.length > MAX_CONTENT_INVENTORY_ENTRIES) fail(`${label} content inventory count invalid`);
  const inventory = value.content_inventory.map((entry, index) => contentEntry(entry, `${label} content ${index}`));
  if (!sameEntries(inventory, sortedEntries(inventory)) || new Set(inventory.map((entry) => entry.path)).size !== inventory.length) fail(`${label} content inventory must be complete, sorted, and unique`);
  return { ...value, content_inventory: inventory, sha256: digest[1] };
}
function workflowRun(value, predecessor) {
  keys(value, ["repository", "path", "run_id", "run_attempt", "head_sha", "conclusion"], "predecessor workflow");
  if (value.repository !== "SceneWorks/SceneWorks" || canonicalPath(value.path, "predecessor workflow path") !== value.path || decimalId(value.run_id, "predecessor workflow run id") !== value.run_id || boundedInteger(value.run_attempt, "predecessor workflow attempt", 1, MAX_ATTEMPT) !== value.run_attempt || value.head_sha !== predecessor.sceneworks_revision || !REV.test(value.head_sha) || !NON_SUCCESS_CONCLUSIONS.has(value.conclusion)) fail("predecessor workflow provenance invalid");
  return value;
}
function currentWorkflow(value, receipt) {
  keys(value, ["campaign_id", "inference_revision", "sceneworks_revision", "repository", "path", "run_id", "run_attempt", "head_sha"], "current workflow");
  if (campaignId(value.campaign_id, "current workflow campaign id") !== receipt.campaign_run_id || value.inference_revision !== receipt.inference_revision || value.sceneworks_revision !== receipt.sceneworks_revision || value.repository !== "SceneWorks/SceneWorks" || value.repository !== receipt.execution.repository || canonicalPath(value.path, "current workflow path") !== CURRENT_WORKFLOW_PATH || decimalId(value.run_id, "current workflow run id") !== receipt.execution.workflow_run_id || boundedInteger(value.run_attempt, "current workflow attempt", 1, MAX_ATTEMPT) !== receipt.execution.workflow_run_attempt || value.head_sha !== receipt.execution.head_sha || value.head_sha !== receipt.sceneworks_revision) fail("current workflow provenance invalid");
  return value;
}
function failureIdentity(value) {
  keys(value, ["code", "phase", "tuple"], "predecessor failure");
  if (!/^[a-z][a-z0-9._-]{0,127}$/.test(boundedString(value.code, "predecessor failure code", 128)) || !FAILURE_PHASES.has(value.phase) || !FAILURE_TUPLES.has(value.tuple)) fail("predecessor failure identity invalid");
}
function expectedQuarantineEntries(predecessor, artifacts) {
  const root = `quarantine/${predecessor.campaign_id}`;
  const entries = [
    { path: `${root}/markers/campaign/${predecessor.markers.campaign.path}`, size: predecessor.markers.campaign.size, sha256: predecessor.markers.campaign.sha256 },
    { path: `${root}/markers/tuple/${predecessor.markers.tuple.path}`, size: predecessor.markers.tuple.size, sha256: predecessor.markers.tuple.sha256 },
    ...artifacts.flatMap((artifact) => [
      { path: `${root}/source-artifacts/${artifact.role}/${artifact.id}/${artifact.name}`, size: artifact.size, sha256: artifact.sha256 },
      ...artifact.content_inventory.map((entry) => ({ path: `${root}/source-artifacts/${artifact.role}/${artifact.id}/extracted/${entry.path}`, size: entry.byte_size, sha256: entry.sha256 })),
    ]),
  ];
  const workflow = stable(predecessor.workflow);
  entries.push({ path: `${root}/workflow-run.json`, size: Buffer.byteLength(workflow), sha256: hash(workflow) });
  return sortedEntries(entries);
}
function validateCampaignLineage(receipt) {
  const lineage = receipt.campaign_lineage;
  keys(lineage, ["kind", "current_campaign_id", "current_workflow", "failed_predecessors", "supersession_records"], "campaign lineage");
  if (!new Set(["clean", "failed_campaign_supersession"]).has(lineage.kind) || campaignId(lineage.current_campaign_id, "current campaign id") !== receipt.campaign_run_id || !Array.isArray(lineage.failed_predecessors) || lineage.failed_predecessors.length > MAX_LINEAGE_PREDECESSORS || !Array.isArray(lineage.supersession_records) || lineage.supersession_records.length > MAX_LINEAGE_PREDECESSORS) fail("campaign lineage identity/count invalid");
  currentWorkflow(lineage.current_workflow, receipt);
  const lineageDigest = campaignLineageSha256(lineage);
  if (receipt.producer.campaign_lineage_sha256 !== lineageDigest) fail("producer lineage hash mismatch");
  const currentWorkflowPayload = stable(lineage.current_workflow);
  const refs = [
    { path: "lineage/campaign-lineage.json", byte_size: Buffer.byteLength(stable(lineage)), sha256: lineageDigest },
    { path: "lineage/current-workflow.json", byte_size: Buffer.byteLength(currentWorkflowPayload), sha256: hash(currentWorkflowPayload) },
  ];
  if (lineage.kind === "clean") {
    if (lineage.failed_predecessors.length !== 0 || lineage.supersession_records.length !== 0) fail("clean lineage must have empty history");
    return { refs, quarantinedDigests: new Set() };
  }
  if (lineage.failed_predecessors.length === 0 || lineage.supersession_records.length !== lineage.failed_predecessors.length) fail("failed-campaign lineage chain is incomplete");
  const campaignIds = new Set([receipt.campaign_run_id]);
  const workflowRuns = new Set([`${receipt.execution.repository}:${receipt.execution.workflow_run_id}:${receipt.execution.workflow_run_attempt}`]);
  const artifactIds = new Set(), paths = new Set(refs.map((entry) => entry.path)), historicalDigests = new Set(), authorityDigests = new Set();
  let previousRunId = null;
  const currentRunId = BigInt(lineage.current_workflow.run_id);
  const predecessors = lineage.failed_predecessors;
  predecessors.forEach((predecessor, index) => {
    keys(predecessor, ["campaign_id", "inference_revision", "sceneworks_revision", "workflow", "failure", "markers", "source_artifacts", "quarantine", "superseded_by"], `failed predecessor ${index}`);
    campaignId(predecessor.campaign_id, `failed predecessor ${index} campaign id`);
    campaignId(predecessor.superseded_by, `failed predecessor ${index} successor id`);
    if (campaignIds.has(predecessor.campaign_id) || !REV.test(predecessor.inference_revision) || !REV.test(predecessor.sceneworks_revision)) fail("duplicate/current predecessor or bad revision");
    campaignIds.add(predecessor.campaign_id);
    workflowRun(predecessor.workflow, predecessor);
    const runKey = `${predecessor.workflow.repository}:${predecessor.workflow.run_id}:${predecessor.workflow.run_attempt}`;
    if (workflowRuns.has(runKey)) fail("duplicate/replayed predecessor workflow run");
    workflowRuns.add(runKey);
    const numericRunId = BigInt(predecessor.workflow.run_id);
    if (previousRunId !== null && numericRunId <= previousRunId) fail("failed predecessors are not oldest-to-newest");
    previousRunId = numericRunId;
    failureIdentity(predecessor.failure);
    keys(predecessor.markers, ["campaign", "tuple"], "predecessor markers");
    sizedHash(predecessor.markers.campaign, "campaign marker");
    sizedHash(predecessor.markers.tuple, "tuple marker");
    if (!Array.isArray(predecessor.source_artifacts) || predecessor.source_artifacts.length === 0 || predecessor.source_artifacts.length > MAX_SOURCE_ARTIFACTS) fail("predecessor source artifact count invalid");
    const artifacts = predecessor.source_artifacts.map((artifact, artifactIndex) => githubArtifact(artifact, predecessor, `source artifact ${artifactIndex}`));
    for (const artifact of artifacts) {
      if (artifactIds.has(artifact.id)) fail("duplicate/replayed source artifact");
      artifactIds.add(artifact.id);
    }
    const expectedEntries = expectedQuarantineEntries(predecessor, artifacts);
    keys(predecessor.quarantine, ["root", "entries", "aggregate_sha256"], "predecessor quarantine");
    if (predecessor.quarantine.root !== `quarantine/${predecessor.campaign_id}` || !Array.isArray(predecessor.quarantine.entries) || predecessor.quarantine.entries.length > MAX_MANIFEST_ENTRIES) fail("quarantine root/entries invalid");
    const actualEntries = predecessor.quarantine.entries.map((entry, entryIndex) => sizedHash(entry, `quarantine entry ${entryIndex}`));
    if (JSON.stringify(actualEntries) !== JSON.stringify(expectedEntries)) fail("quarantine entries are not the exact sorted closure");
    const quarantineAggregate = hash(stable({ root: predecessor.quarantine.root, entries: actualEntries }));
    if (predecessor.quarantine.aggregate_sha256 !== quarantineAggregate) fail("quarantine aggregate invalid");
    for (const entry of actualEntries) {
      if (paths.has(entry.path)) fail("duplicate/replayed quarantine path");
      paths.add(entry.path); historicalDigests.add(entry.sha256); refs.push({ path: entry.path, byte_size: entry.size, sha256: entry.sha256 });
    }
    const aggregatePath = `${predecessor.quarantine.root}/aggregate.json`;
    if (paths.has(aggregatePath)) fail("duplicate/replayed quarantine aggregate path");
    paths.add(aggregatePath); historicalDigests.add(quarantineAggregate);
    const quarantinePayload = stable({ root: predecessor.quarantine.root, entries: actualEntries });
    refs.push({ path: aggregatePath, byte_size: Buffer.byteLength(quarantinePayload), sha256: hash(quarantinePayload) });

    const successor = index + 1 < predecessors.length ? predecessors[index + 1] : receipt;
    const record = lineage.supersession_records[index];
    keys(record, ["predecessor_campaign_id", "successor_campaign_id", "predecessor_inference_revision", "predecessor_sceneworks_revision", "successor_inference_revision", "successor_sceneworks_revision", "authority"], `supersession record ${index}`);
    sizedHash(record.authority, `supersession authority ${index}`);
    const expectedAuthorityPath = `lineage/supersession-records/${predecessor.campaign_id}-to-${successor.campaign_run_id ?? successor.campaign_id}.json`;
    if (predecessor.superseded_by !== (successor.campaign_run_id ?? successor.campaign_id) || record.predecessor_campaign_id !== predecessor.campaign_id || record.successor_campaign_id !== predecessor.superseded_by || record.predecessor_inference_revision !== predecessor.inference_revision || record.predecessor_sceneworks_revision !== predecessor.sceneworks_revision || record.successor_inference_revision !== successor.inference_revision || record.successor_sceneworks_revision !== successor.sceneworks_revision || record.authority.path !== expectedAuthorityPath) fail("supersession chain fork/head/pin mismatch");
    if (paths.has(record.authority.path) || authorityDigests.has(record.authority.sha256) || historicalDigests.has(record.authority.sha256)) fail("duplicate/replayed supersession authority");
    paths.add(record.authority.path); authorityDigests.add(record.authority.sha256); historicalDigests.add(record.authority.sha256);
    refs.push({ path: record.authority.path, byte_size: record.authority.size, sha256: record.authority.sha256 });
  });
  if (previousRunId >= currentRunId) fail("last predecessor workflow run must be older than current workflow run");
  return { refs, quarantinedDigests: historicalDigests };
}
function strictV2Manifest(value, expected) {
  keys(value, ["campaign_run_id", "entries", "aggregate_sha256"], "artifact manifest");
  if (value.campaign_run_id !== expected.campaign_run_id || !Array.isArray(value.entries) || value.entries.length > MAX_MANIFEST_ENTRIES) fail("artifact manifest mixed run/count");
  const seen = new Set();
  value.entries.forEach((entry) => { contentEntry(entry, "artifact entry"); if (seen.has(entry.path)) fail("artifact entry invalid"); seen.add(entry.path); });
  if (JSON.stringify(value.entries) !== JSON.stringify(sortedEntries(value.entries))) fail("artifact manifest entries are not sorted");
  if (!sameEntries(value.entries, expected.entries)) fail("artifact manifest is not exact V2 path/size/digest closure");
  if (value.aggregate_sha256 !== hash(stable({ campaign_run_id: value.campaign_run_id, entries: value.entries }))) fail("artifact manifest checksum");
  return value.aggregate_sha256;
}
export function buildArtifactManifest(receipt, corpus) {
  const refs = currentEvidenceRefs(receipt, corpus);
  if (receipt.schema_version === 1) { const entries = refs.map((entry) => ({ ...entry, byte_size: 1 })); return { campaign_run_id: receipt.campaign_run_id, entries, aggregate_sha256: hash(stable({ campaign_run_id: receipt.campaign_run_id, entries })) }; }
  if (receipt.schema_version !== 2) fail("unsupported receipt schema version");
  const lineage = validateCampaignLineage(receipt);
  const entries = sortedEntries([...refs.map((entry) => ({ ...entry, byte_size: 1 })), ...lineage.refs]);
  return { campaign_run_id: receipt.campaign_run_id, entries, aggregate_sha256: hash(stable({ campaign_run_id: receipt.campaign_run_id, entries })) };
}
function validateReceiptV1(receipt, corpusHash, inferenceRevision, sceneworksRevision, corpus) {
  keys(receipt, ["schema_version", "campaign_run_id", "inference_revision", "sceneworks_revision", "corpus_sha256", "execution", "producer", "metric_identity", "inference_preflight", "runs", "hostile_sanitizer", "prompt_composition", "artifact_manifest"], "receipt"); if (receipt.schema_version !== 1 || typeof receipt.campaign_run_id !== "string" || !receipt.campaign_run_id || receipt.inference_revision !== inferenceRevision || receipt.sceneworks_revision !== sceneworksRevision || !REV.test(receipt.inference_revision) || !REV.test(receipt.sceneworks_revision) || receipt.corpus_sha256 !== corpusHash) fail("receipt identity invalid"); keys(receipt.execution, ["repository", "workflow_run_id", "workflow_run_attempt", "head_sha", "started_at", "completed_at", "clean_tree"], "execution"); if (receipt.execution.repository !== "SceneWorks/SceneWorks" || receipt.execution.head_sha !== sceneworksRevision || !REV.test(receipt.execution.head_sha) || typeof receipt.execution.workflow_run_id !== "string" || !receipt.execution.workflow_run_id || !Number.isInteger(receipt.execution.workflow_run_attempt) || receipt.execution.workflow_run_attempt < 1 || receipt.execution.clean_tree !== true || Number.isNaN(Date.parse(receipt.execution.started_at)) || Number.isNaN(Date.parse(receipt.execution.completed_at)) || Date.parse(receipt.execution.completed_at) < Date.parse(receipt.execution.started_at)) fail("execution provenance invalid"); keys(receipt.producer, ["command", "artifact_name", "transcript_sha256", "artifact_manifest_sha256"], "producer"); if (typeof receipt.producer.command !== "string" || !receipt.producer.command || typeof receipt.producer.artifact_name !== "string" || !receipt.producer.artifact_name) fail("producer identity invalid"); sha(receipt.producer.transcript_sha256, "producer transcript"); sha(receipt.producer.artifact_manifest_sha256, "producer manifest"); const refs = []; metric(receipt.metric_identity, refs); preflight(receipt.inference_preflight, inferenceRevision, refs); ref(refs, "producer/transcript", receipt.producer.transcript_sha256); if (!Array.isArray(receipt.runs) || receipt.runs.length !== 4) fail("run count"); const seen = new Set(), inventory = new Map(), analysis = new Map(); receipt.runs.forEach((entry) => { const result = run(entry, refs); if (seen.has(result.runKey)) fail("duplicate run"); seen.add(result.runKey); if (inventory.has(entry.tier) && inventory.get(entry.tier) !== result.inventory) fail("mixed snapshot inventory"); inventory.set(entry.tier, result.inventory); analysis.set(result.runKey, result); }); if ([...RUNS.keys()].some((key) => !seen.has(key))) fail("required run missing"); hostile(receipt.hostile_sanitizer, corpus, refs); prompt(receipt.prompt_composition, corpus, refs); if (receipt.producer.artifact_manifest_sha256 !== manifest(receipt.artifact_manifest, receipt.campaign_run_id, refs)) fail("producer manifest does not bind receipt"); ["mlx", "candle-cuda"].forEach((backend) => { const one = analysis.get(`${backend}:1b`), eight = analysis.get(`${backend}:8b`); const improvement = (one.median_lpips - eight.median_lpips) / one.median_lpips; const pairs = one.cases.flatMap((entry, index) => entry.accepted && eight.cases[index].accepted ? [{ one: entry.lpips, eight: eight.cases[index].lpips }] : []); if (one.median_lpips <= 0 || pairs.length < 114 || improvement < .10 || eight.validity - one.validity < -.02 || pairedBootstrapLowerBound(pairs) <= 0) fail(`8B ${backend} threshold failed`); });
}
export function validateReceipt(receipt, corpusHash, inferenceRevision, sceneworksRevision, corpus) {
  if (receipt?.schema_version !== 2) return validateReceiptV1(receipt, corpusHash, inferenceRevision, sceneworksRevision, corpus);
  keys(receipt, ["schema_version", "campaign_run_id", "inference_revision", "sceneworks_revision", "corpus_sha256", "execution", "producer", "metric_identity", "inference_preflight", "runs", "hostile_sanitizer", "prompt_composition", "campaign_lineage", "artifact_manifest"], "receipt");
  campaignId(receipt.campaign_run_id, "receipt campaign id");
  keys(receipt.producer, ["command", "artifact_name", "transcript_sha256", "artifact_manifest_sha256", "campaign_lineage_sha256"], "producer");
  boundedString(receipt.producer.command, "producer command", 4096);
  boundedString(receipt.producer.artifact_name, "producer artifact name", 255);
  sha(receipt.producer.campaign_lineage_sha256, "producer lineage");
  if (!Array.isArray(receipt.artifact_manifest?.entries) || receipt.artifact_manifest.entries.length > MAX_MANIFEST_ENTRIES) fail("artifact manifest mixed run/count");
  decimalId(receipt.inference_preflight?.workflow_run_id, "preflight workflow run id");
  boundedInteger(receipt.inference_preflight?.workflow_run_attempt, "preflight workflow attempt", 1, MAX_ATTEMPT);
  const lineage = validateCampaignLineage(receipt);
  const legacy = { ...receipt, schema_version: 1, producer: { command: receipt.producer.command, artifact_name: receipt.producer.artifact_name, transcript_sha256: receipt.producer.transcript_sha256, artifact_manifest_sha256: receipt.producer.artifact_manifest_sha256 } };
  delete legacy.campaign_lineage;
  validateReceiptV1(legacy, corpusHash, inferenceRevision, sceneworksRevision, corpus);
  const current = currentEvidenceRefs(receipt, corpus);
  const producerDigests = [receipt.producer.artifact_manifest_sha256, receipt.producer.campaign_lineage_sha256];
  if (current.some((entry) => lineage.quarantinedDigests.has(entry.sha256)) || producerDigests.some((digest) => lineage.quarantinedDigests.has(digest))) fail("quarantined evidence cannot satisfy current campaign references");
  const expected = buildArtifactManifest(receipt, corpus);
  if (receipt.producer.artifact_manifest_sha256 !== strictV2Manifest(receipt.artifact_manifest, expected)) fail("producer manifest does not bind receipt");
}
function option(name) { const index = process.argv.indexOf(name); if (index < 0 || !process.argv[index + 1]) fail(`missing ${name}`); return process.argv[index + 1]; }
function readBoundedJson(path, label, maximumBytes) { const info = statSync(path); if (!info.isFile() || info.size > maximumBytes) fail(`${label} exceeds ${maximumBytes} byte input limit`); const bytes = readFileSync(path); if (bytes.byteLength > maximumBytes) fail(`${label} exceeds ${maximumBytes} byte input limit`); return JSON.parse(bytes.toString("utf8")); }
function main() { const command = process.argv[2]; if (command === "validate-plan") { console.log(`corpus_sha256=${validatePlan(readBoundedJson(option("--corpus"), "corpus", MAX_CORPUS_BYTES))}`); return; } if (command === "validate-receipt") { const corpus = readBoundedJson(option("--corpus"), "corpus", MAX_CORPUS_BYTES); validateReceipt(readBoundedJson(option("--receipt"), "receipt", MAX_RECEIPT_BYTES), validatePlan(corpus), option("--inference-revision"), option("--sceneworks-revision"), corpus); console.log("starvector terminal receipt: OK"); return; } fail("usage: validate-plan|validate-receipt"); }
if (import.meta.url === `file://${process.argv[1]}`) { try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; } }
