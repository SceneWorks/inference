import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { validatePlan, validateReceipt } from "../release/starvector_terminal_evidence.mjs";

const corpus = JSON.parse(readFileSync("release/starvector-terminal-corpus-v1.json", "utf8"));
const INFERENCE = "1".repeat(40);
const SCENEWORKS = "2".repeat(40);
const INVENTORY = "3".repeat(64);

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
    image_quality: { case_count: 120, validity_rate: 0.97, median_ssim: 0.90, median_lpips: tier === "1b" ? 0.10 : 0.08, p95_latency_seconds: 119, memory_headroom_percent: tier === "1b" ? 15 : 10 },
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
    eight_b_uplift: {
      bootstrap_confidence: 0.95,
      by_backend: [
        { backend: "mlx", median_lpips_improvement: 0.2, validity_delta: 0, bootstrap_lower_bound: 0.001 },
        { backend: "candle-cuda", median_lpips_improvement: 0.2, validity_delta: 0, bootstrap_lower_bound: 0.001 },
      ],
    },
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
  mutated.runs[0].image_quality.median_ssim = 0.849;
  assert.throws(() => validateReceipt(mutated, validatePlan(corpus), INFERENCE, SCENEWORKS), /image-quality threshold/);

  mutated = receipt();
  mutated.runs[0].image_quality.p95_latency_seconds = 120.001;
  assert.throws(() => validateReceipt(mutated, validatePlan(corpus), INFERENCE, SCENEWORKS), /p95 latency/);

  mutated = receipt();
  mutated.runs[0].deterministic_parity.rendered_ssim[19] = 0.994;
  assert.throws(() => validateReceipt(mutated, validatePlan(corpus), INFERENCE, SCENEWORKS), /rendered-SSIM/);

  mutated = receipt();
  mutated.runs[1].image_quality.median_lpips = 0.091;
  mutated.eight_b_uplift.by_backend[0].median_lpips_improvement = 0.09;
  assert.throws(() => validateReceipt(mutated, validatePlan(corpus), INFERENCE, SCENEWORKS), /LPIPS/);

  mutated = receipt();
  mutated.eight_b_uplift.by_backend[0].bootstrap_lower_bound = 0;
  assert.throws(() => validateReceipt(mutated, validatePlan(corpus), INFERENCE, SCENEWORKS), /bootstrap/);

  mutated = receipt();
  mutated.runs[1].image_quality.validity_rate = 0.949;
  mutated.eight_b_uplift.by_backend[0].validity_delta = -0.021;
  assert.throws(() => validateReceipt(mutated, validatePlan(corpus), INFERENCE, SCENEWORKS), /image-quality threshold|validity/);
});

test("corpus rejects count and immutable-row mutations", () => {
  const mutated = structuredClone(corpus);
  mutated.upstream_image_quality_cases.sources[0].row_count = 29;
  assert.throws(() => validatePlan(mutated), /exact first thirty/);
  mutated.upstream_image_quality_cases.sources[0].row_count = 30;
  mutated.upstream_image_quality_cases.sources[0].parquet_sha256 = "0".repeat(64);
  assert.throws(() => validatePlan(mutated), /immutable identity/);
});
