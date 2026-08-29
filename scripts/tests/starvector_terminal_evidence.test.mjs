import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { validatePlan, validateReceipt } from "../release/starvector_terminal_evidence.mjs";

const corpus = JSON.parse(readFileSync("release/starvector-terminal-corpus-v1.json", "utf8"));
const INFERENCE = "1".repeat(40);
const SCENEWORKS = "2".repeat(40);
const INVENTORY = "3".repeat(64);

function cases(tier) {
  return Array.from({ length: 120 }, (_, case_index) => ({
    case_index,
    accepted: true,
    ssim: 0.90,
    lpips: tier === "1b" ? 0.10 : 0.08,
    latency_seconds: 119,
  }));
}

function run(backend, tier) {
  const provider = {
    "mlx:1b": "mlx-starvector-1b",
    "mlx:8b": "mlx-starvector-8b",
    "candle-cuda:1b": "candle-starvector-1b",
    "candle-cuda:8b": "candle-starvector-8b",
  }[`${backend}:${tier}`];
  return {
    backend,
    provider_id: provider,
    tier,
    device: backend === "mlx" ? "Apple Metal" : "CUDA:0",
    model: {
      key: `starvector-${tier}-im2svg`,
      repository: `starvector/starvector-${tier}-im2svg`,
      revision: tier === "1b" ? "380ab95d25a8e9ab1dc825debe238b4953ae13b9" : "518beea8dcb5f7a37c5911e92d1d62a76beee7f9",
      inventory_sha256: INVENTORY,
    },
    image_quality: { cases: cases(tier), memory_headroom_percent: tier === "1b" ? 15 : 10 },
    deterministic_parity: { case_count: 20, rendered_ssim: Array(20).fill(0.996) },
    lifecycle: { load: true, unload: true, reload: true, memory_reported: true },
    limits: { complete_root: true, eos: true, token: true, byte: true, wall_time: true, cancellation: true },
  };
}

function receipt() {
  return {
    schema_version: 1,
    campaign_run_id: "single-terminal-run",
    inference_revision: INFERENCE,
    sceneworks_revision: SCENEWORKS,
    corpus_sha256: validatePlan(corpus),
    runs: [run("mlx", "1b"), run("mlx", "8b"), run("candle-cuda", "1b"), run("candle-cuda", "8b")],
  };
}

test("checked-in corpus has exact selected counts and immutable identities", () => {
  assert.match(validatePlan(corpus), /^[0-9a-f]{64}$/);
});

test("receipt accepts exactly the four native backend/tier runs", () => {
  validateReceipt(receipt(), validatePlan(corpus), INFERENCE, SCENEWORKS);
});

test("receipt rejects duplicate, mixed revision/inventory, and threshold mutations", () => {
  let mutated = receipt();
  mutated.runs[3] = { ...mutated.runs[2], tier: "1b" };
  assert.throws(() => validateReceipt(mutated, validatePlan(corpus), INFERENCE, SCENEWORKS), /duplicate/);

  mutated = receipt();
  mutated.runs[2].model.inventory_sha256 = "4".repeat(64);
  assert.throws(() => validateReceipt(mutated, validatePlan(corpus), INFERENCE, SCENEWORKS), /mixed snapshot inventory/);

  mutated = receipt();
  mutated.inference_revision = "0".repeat(40);
  assert.throws(() => validateReceipt(mutated, validatePlan(corpus), INFERENCE, SCENEWORKS), /mixed inference/);

  mutated = receipt();
  for (const record of mutated.runs[0].image_quality.cases.slice(0, 61)) record.ssim = 0.849;
  assert.throws(() => validateReceipt(mutated, validatePlan(corpus), INFERENCE, SCENEWORKS), /image-quality threshold/);

  mutated = receipt();
  for (const record of mutated.runs[0].image_quality.cases.slice(113)) record.latency_seconds = 120.001;
  assert.throws(() => validateReceipt(mutated, validatePlan(corpus), INFERENCE, SCENEWORKS), /p95 latency/);

  mutated = receipt();
  mutated.runs[0].deterministic_parity.rendered_ssim[19] = 0.994;
  assert.throws(() => validateReceipt(mutated, validatePlan(corpus), INFERENCE, SCENEWORKS), /rendered-SSIM/);

  mutated = receipt();
  for (const record of mutated.runs[1].image_quality.cases) record.lpips = 0.091;
  assert.throws(() => validateReceipt(mutated, validatePlan(corpus), INFERENCE, SCENEWORKS), /LPIPS/);

  mutated = receipt();
  for (const record of mutated.runs[1].image_quality.cases) record.lpips = 0.10;
  assert.throws(() => validateReceipt(mutated, validatePlan(corpus), INFERENCE, SCENEWORKS), /bootstrap/);

  mutated = receipt();
  for (const record of mutated.runs[1].image_quality.cases.slice(0, 7)) {
    record.accepted = false; record.ssim = null; record.lpips = null;
  }
  assert.throws(() => validateReceipt(mutated, validatePlan(corpus), INFERENCE, SCENEWORKS), /image-quality threshold|validity/);
});

test("receipt rejects aggregate spoofing and omitted or duplicated ordered cases", () => {
  let mutated = receipt();
  mutated.runs[0].image_quality.median_ssim = 0.99;
  assert.throws(() => validateReceipt(mutated, validatePlan(corpus), INFERENCE, SCENEWORKS), /keys differ/);

  mutated = receipt();
  mutated.runs[0].image_quality.cases.pop();
  assert.throws(() => validateReceipt(mutated, validatePlan(corpus), INFERENCE, SCENEWORKS), /120 ordered/);

  mutated = receipt();
  mutated.runs[0].image_quality.cases[119].case_index = 118;
  assert.throws(() => validateReceipt(mutated, validatePlan(corpus), INFERENCE, SCENEWORKS), /order/);

  mutated = receipt();
  mutated.eight_b_uplift = { bootstrap_confidence: 0.95, bootstrap_lower_bound: 1 };
  assert.throws(() => validateReceipt(mutated, validatePlan(corpus), INFERENCE, SCENEWORKS), /receipt keys differ/);
});

test("corpus rejects count and immutable-row mutations", () => {
  const mutated = structuredClone(corpus);
  mutated.upstream_image_quality_cases.sources[0].row_count = 29;
  assert.throws(() => validatePlan(mutated), /exact first thirty/);
  mutated.upstream_image_quality_cases.sources[0].row_count = 30;
  mutated.upstream_image_quality_cases.sources[0].parquet_sha256 = "0".repeat(64);
  assert.throws(() => validatePlan(mutated), /immutable identity/);
});
