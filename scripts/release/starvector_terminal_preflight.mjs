#!/usr/bin/env node
// Assemble the exact native-provider provenance consumed by SceneWorks' terminal campaign.
import { createHash } from "node:crypto";
import { lstat, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const REVISION = /^[0-9a-f]{40}$/;
const REQUIRED = [
  {
    collection: "inventory_artifacts",
    identity: { tier: "1b" },
    path: "inventory/starvector-1b-inventory.json",
  },
  {
    collection: "inventory_artifacts",
    identity: { tier: "8b" },
    path: "inventory/starvector-8b-inventory.json",
  },
  {
    collection: "hook_logs",
    identity: { backend: "mlx", tier: "1b" },
    path: "hooks/mlx-starvector-1b.log",
  },
  {
    collection: "hook_logs",
    identity: { backend: "mlx", tier: "8b" },
    path: "hooks/mlx-starvector-8b.log",
  },
  {
    collection: "hook_logs",
    identity: { backend: "candle-cuda", tier: "1b" },
    path: "hooks/candle-cuda-starvector-1b.log",
  },
  {
    collection: "hook_logs",
    identity: { backend: "candle-cuda", tier: "8b" },
    path: "hooks/candle-cuda-starvector-8b.log",
  },
];

function fail(message) {
  throw new Error(`StarVector terminal preflight: ${message}`);
}

function parseArguments(argv) {
  if (argv[0] !== "assemble") fail("expected the assemble command");
  const values = {};
  for (let index = 1; index < argv.length; index += 2) {
    const flag = argv[index];
    const value = argv[index + 1];
    if (!flag?.startsWith("--") || value === undefined) fail("arguments must be flag/value pairs");
    const key = flag.slice(2);
    if (values[key] !== undefined) fail(`duplicate --${key}`);
    values[key] = value;
  }
  const expected = ["head-sha", "root", "workflow-run-attempt", "workflow-run-id"];
  if (Object.keys(values).sort().join("|") !== expected.join("|")) {
    fail(`expected exactly ${expected.map((key) => `--${key}`).join(", ")}`);
  }
  return values;
}

async function checkedDigest(root, relative) {
  const candidate = path.join(root, ...relative.split("/"));
  const metadata = await lstat(candidate).catch(() => null);
  if (!metadata?.isFile() || metadata.isSymbolicLink() || metadata.size === 0) {
    fail(`required non-empty regular file is missing: ${relative}`);
  }
  return createHash("sha256").update(await readFile(candidate)).digest("hex");
}

export async function assemblePreflight({ root, headSha, workflowRunId, workflowRunAttempt }) {
  if (!path.isAbsolute(root)) fail("--root must be absolute");
  if (!REVISION.test(headSha)) fail("--head-sha must be a lowercase 40-hex revision");
  if (typeof workflowRunId !== "string" || !workflowRunId) fail("--workflow-run-id must be non-empty");
  const attempt = Number(workflowRunAttempt);
  if (!Number.isSafeInteger(attempt) || attempt < 1 || String(attempt) !== String(workflowRunAttempt)) {
    fail("--workflow-run-attempt must be a canonical positive integer");
  }

  const value = {
    workflow_run_id: workflowRunId,
    workflow_run_attempt: attempt,
    head_sha: headSha,
    inventory_artifacts: [],
    hook_logs: [],
  };
  for (const required of REQUIRED) {
    value[required.collection].push({
      ...required.identity,
      path: required.path,
      sha256: await checkedDigest(root, required.path),
    });
  }
  const output = path.join(root, "starvector-terminal-preflight.json");
  await writeFile(output, `${JSON.stringify(value, null, 2)}\n`, { encoding: "utf8", flag: "w" });
  return { output, value };
}

async function main() {
  const values = parseArguments(process.argv.slice(2));
  const { output } = await assemblePreflight({
    root: values.root,
    headSha: values["head-sha"],
    workflowRunId: values["workflow-run-id"],
    workflowRunAttempt: values["workflow-run-attempt"],
  });
  process.stdout.write(`StarVector terminal preflight: wrote ${output}\n`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  });
}
