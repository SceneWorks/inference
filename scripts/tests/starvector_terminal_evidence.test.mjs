import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, statSync, symlinkSync, truncateSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { artifactByteSizesFromFiles, buildArtifactManifest, campaignLineageSha256, currentArtifactReferences, hostilePayload, MAX_RECEIPT_BYTES, pairedBootstrapLowerBound, validatePlan, validateReceipt } from "../release/starvector_terminal_evidence.mjs";

const corpus = JSON.parse(readFileSync("release/starvector-terminal-corpus-v1.json", "utf8"));
const INFERENCE = "1".repeat(40), SCENEWORKS = "2".repeat(40), EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", h = (value) => createHash("sha256").update(value).digest("hex"), d = (label) => h(`fixture:${label}`);
const stable = (value) => Array.isArray(value) ? `[${value.map(stable).join(",")}]` : value && typeof value === "object" ? `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stable(value[key])}`).join(",")}}` : JSON.stringify(value);
const source = (index) => [["starvector/svg-stack-simple","1d2a96a17cc0c4c1f337b7631adc8c5885bc72ea"],["starvector/svg-icons-simple","e1918a27ba6649e856e5db0710d8a6c7046762c1"],["starvector/svg-emoji-simple","fa75b3617872ae57e6f3cb450aee65dbccbd69e0"],["starvector/svg-fonts-simple","453c739ea13ad2685127f721c333f14d99485299"]][Math.floor(index / 30)];
const prompts = ["geometric badge","isometric folder","rounded calendar","minimal rocket","layered landscape","abstract flower"];
const promptPayload = (index) => `Create a ${prompts[Math.floor(index / 10)]} vector illustration, variant ${index % 10}, with clear silhouette, balanced composition, and no text.`;
function imageCases(tier, backend) { return Array.from({ length: 120 }, (_, case_index) => ({ case_index, source: { dataset: source(case_index)[0], revision: source(case_index)[1], row_index: case_index % 30 }, source_svg_sha256: d(`${backend}-${tier}-source-${case_index}`), input_png_sha256: d(`${backend}-${tier}-png-${case_index}`), provider_transcript_sha256: d(`${backend}-${tier}-transcript-${case_index}`), finish_reason: "complete_root", canonical_svg_sha256: d(`${backend}-${tier}-svg-${case_index}`), preview_png_sha256: d(`${backend}-${tier}-preview-${case_index}`), accepted: true, ssim: .90, lpips: tier === "1b" ? .10 : .08, latency_seconds: 119 })); }
function run(backend, tier) { return { backend, provider_id: {"mlx:1b":"mlx-starvector-1b","mlx:8b":"mlx-starvector-8b","candle-cuda:1b":"candle-starvector-1b","candle-cuda:8b":"candle-starvector-8b"}[`${backend}:${tier}`], tier, device: backend === "mlx" ? "Apple Metal" : "CUDA:0", model: { key:`starvector-${tier}-im2svg`, repository:`starvector/starvector-${tier}-im2svg`, revision:tier === "1b" ? "380ab95d25a8e9ab1dc825debe238b4953ae13b9":"518beea8dcb5f7a37c5911e92d1d62a76beee7f9", inventory_sha256:d(`inventory-${tier}`) }, hardware: { runner_name:"fixture",os:"test",arch:"arm64",system_memory_total_bytes:1000,baseline_available_bytes:900,peak_process_rss_bytes:500,accelerator:{name:"fixture",uuid:null,driver_runtime:"fixture",total_bytes:1000,baseline_free_bytes:900,peak_used_bytes:tier === "1b" ? 850 : 880,raw_probe_sha256:d(`probe-${backend}-${tier}`)}}, image_quality:{cases:imageCases(tier,backend)}, deterministic_parity:{case_count:20,cases:Array.from({length:20},(_,case_index)=>({case_index,seed:case_index,first_preview_png_sha256:d(`${backend}-${tier}-first-${case_index}`),second_preview_png_sha256:d(`${backend}-${tier}-second-${case_index}`),rendered_ssim:.996}))}, lifecycle:{load:true,unload:true,reload:true,memory_reported:true},limits:{complete_root:true,eos:true,token:true,byte:true,wall_time:true,cancellation:true},lifecycle_memory_transcript_sha256:d(`lifecycle-${backend}-${tier}`) }; }
function hostile() { return { corpus_sha256:corpus.sceneworks_owned_suites.hostile_sanitizer.content_identity_sha256, sanitizer_version:"fixture", cases:Array.from({length:200},(_,case_index)=>({case_index,case_id:`hostile-v1-${case_index}`,input_sha256:h(hostilePayload(case_index)),expected_policy:"reject_or_sanitize_inert",outcome:"rejected",error_code:"rejected",canonical_svg_sha256:null,preview_png_sha256:null,published_paths:[],staging_residue:[],result_contains_inline_svg:false}))}; }
function prompt() { return { corpus_sha256:corpus.sceneworks_owned_suites.prompt_composition.content_identity_sha256,raster_provider_id:"fixture",raster_model:"fixture",raster_revision:"fixture",raster_inventory_sha256:d("raster-inventory"),clip_provider_id:"fixture",clip_model:"fixture",clip_revision:"fixture",clip_inventory_sha256:d("clip-inventory"),metric_transcript_sha256:d("prompt-metric"),cases:Array.from({length:60},(_,case_index)=>({case_index,case_id:`prompt-v1-${case_index}`,prompt_sha256:h(promptPayload(case_index)),raster_png_sha256:d(`raster-${case_index}`),vector_provider_transcript_sha256:d(`vector-${case_index}`),canonical_svg_sha256:d(`prompt-svg-${case_index}`),preview_png_sha256:d(`prompt-preview-${case_index}`),accepted:true,raster_prompt_cosine:.90,preview_prompt_cosine:.89,alignment_loss:.01}))}; }
const fixtureByteSize = (path) => Buffer.byteLength(`fixture artifact ${path}`);
function buildManifest(value, byteSizes = fixtureByteSize) { return buildArtifactManifest(value, corpus, value.schema_version === 2 ? byteSizes : undefined); }
function receipt() { const value={schema_version:1,campaign_run_id:"terminal-fixture",inference_revision:INFERENCE,sceneworks_revision:SCENEWORKS,corpus_sha256:validatePlan(corpus),execution:{repository:"SceneWorks/SceneWorks",workflow_run_id:"1",workflow_run_attempt:1,head_sha:SCENEWORKS,started_at:"2026-08-29T00:00:00Z",completed_at:"2026-08-29T00:01:00Z",clean_tree:true},producer:{command:"fixture",artifact_name:"fixture",transcript_sha256:d("producer"),artifact_manifest_sha256:d("pending")},metric_identity:{rasterizer:"resvg-0.45",canvas:{width:512,height:512,background:"white",colorspace:"srgb8"},ssim:{implementation:"skimage.metrics.structural_similarity",package_version:"fixture",lock_sha256:d("ssim-lock"),data_range:255,channel_axis:2,gaussian_weights:true,sigma:1.5,use_sample_covariance:false},lpips:{implementation:"richzhang/lpips",package_version:"fixture",version:"0.1",net:"alex",eval_mode:true,rgb_normalization:"[-1,1]",lock_sha256:d("lpips-lock"),linear_weights_sha256:"df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",alexnet_weights_sha256:"7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"},metric_transcript_sha256:d("metric")},inference_preflight:{workflow_run_id:"1",workflow_run_attempt:1,head_sha:INFERENCE,inventory_artifacts:[{tier:"1b",sha256:d("preflight-1b")},{tier:"8b",sha256:d("preflight-8b")}],hook_logs:["mlx:1b","mlx:8b","candle-cuda:1b","candle-cuda:8b"].map((key)=>{const [backend,tier]=key.split(":");return {backend,tier,sha256:d(`hook-${key}`)};})},runs:[run("mlx","1b"),run("mlx","8b"),run("candle-cuda","1b"),run("candle-cuda","8b")],hostile_sanitizer:hostile(),prompt_composition:prompt()}; value.artifact_manifest=buildManifest(value); value.producer.artifact_manifest_sha256=value.artifact_manifest.aggregate_sha256; return value; }
function quarantineEntries(predecessor) {
  const root = predecessor.quarantine.root;
  const entries = [
    { path:`${root}/markers/campaign/${predecessor.markers.campaign.path}`,size:predecessor.markers.campaign.size,sha256:predecessor.markers.campaign.sha256 },
    { path:`${root}/markers/tuple/${predecessor.markers.tuple.path}`,size:predecessor.markers.tuple.size,sha256:predecessor.markers.tuple.sha256 },
    ...predecessor.source_artifacts.flatMap((artifact)=>[
      {path:`${root}/source-artifacts/${artifact.role}/${artifact.id}/${artifact.name}`,size:artifact.size,sha256:artifact.digest.slice("sha256:".length)},
      ...artifact.content_inventory.map((entry)=>({path:`${root}/source-artifacts/${artifact.role}/${artifact.id}/extracted/${entry.path}`,size:entry.byte_size,sha256:entry.sha256})),
    ]),
    { path:`${root}/workflow-run.json`,size:Buffer.byteLength(stable(predecessor.workflow)),sha256:h(stable(predecessor.workflow)) },
  ];
  return entries.sort((left,right)=>left.path.localeCompare(right.path));
}
function resealQuarantine(predecessor) { predecessor.quarantine.entries=quarantineEntries(predecessor);predecessor.quarantine.aggregate_sha256=h(stable({root:predecessor.quarantine.root,entries:predecessor.quarantine.entries})); }
function rebuildEdges(value) {
  value.campaign_lineage.supersession_records=value.campaign_lineage.failed_predecessors.map((predecessor,index)=>{
    const successor=value.campaign_lineage.failed_predecessors[index+1]??value;
    const successorId=successor.campaign_id??successor.campaign_run_id;
    predecessor.superseded_by=successorId;
    return {predecessor_campaign_id:predecessor.campaign_id,successor_campaign_id:successorId,predecessor_inference_revision:predecessor.inference_revision,predecessor_sceneworks_revision:predecessor.sceneworks_revision,successor_inference_revision:successor.inference_revision,successor_sceneworks_revision:successor.sceneworks_revision,authority:{path:`lineage/supersession-records/${predecessor.campaign_id}-to-${successorId}.json`,size:200+index,sha256:d(`authority-${predecessor.campaign_id}-${successorId}`)}};
  });
}
function v2Receipt(predecessorCount=0) {
  const value=receipt();value.schema_version=2;value.campaign_run_id="terminal-current";value.execution.workflow_run_id="9000";value.producer.campaign_lineage_sha256=d("pending-lineage");
  for (const run of value.runs) {
    run.deterministic_parity = {case_count:20, upstream_reference:{implementation_repository:"https://github.com/joanrod/star-vector",implementation_revision:"0e083c1911760aa31bc576ca7f337a7f8ee605ec",checkpoint_repository:run.model.repository,checkpoint_revision:run.model.revision,checkpoint_inventory_sha256:run.model.inventory_sha256,config_sha256:d(`${run.tier}-config`),processor_sha256:d(`${run.tier}-processor`),transcript_sha256:d(`${run.tier}-oracle`)},cases:Array.from({length:20},(_,index)=>({case_index:index,seed:index,input_png_sha256:run.image_quality.cases[Math.floor(index/5)*30+index%5].input_png_sha256,native_preview_png_sha256:d(`${run.backend}-${run.tier}-native-${index}`),upstream_svg_sha256:d(`${run.tier}-oracle-svg-${index}`),upstream_preview_png_sha256:d(`${run.tier}-oracle-png-${index}`),rendered_ssim:.996}))};
  }
  const predecessors=Array.from({length:predecessorCount},(_,index)=>{
    const campaign_id=`terminal-failed-${index+1}`;
    const workflow={repository:"SceneWorks/SceneWorks",path:".github/workflows/server-candle-linux.yml",run_id:String(1000+index),run_attempt:1,head_sha:String(index+4).repeat(40),conclusion:index%2===0?"cancelled":"failure"};
    const predecessor={campaign_id,inference_revision:String(index+3).repeat(40),sceneworks_revision:String(index+4).repeat(40),workflow,failure:{code:index%2===0?"operator-cancelled":"tuple-failed",phase:"execution",tuple:index%2===0?"mlx:1b":"candle-cuda:8b"},markers:{campaign:{path:`markers/${campaign_id}.json`,size:100+index,sha256:d(`campaign-marker-${index}`)},tuple:{path:`markers/${campaign_id}-tuple.json`,size:110+index,sha256:d(`tuple-marker-${index}`)}},source_artifacts:[{role:"failed-campaign",repository:workflow.repository,workflow_run_id:workflow.run_id,workflow_run_attempt:workflow.run_attempt,head_sha:workflow.head_sha,api_workflow_run:{id:workflow.run_id,head_sha:workflow.head_sha},id:String(7000+index),name:`failed-campaign-${index+1}`,size:300+index,digest:`sha256:${d(`source-artifact-${index}`)}`,content_inventory:[{path:"campaign-summary.json",byte_size:50+index,sha256:d(`source-content-summary-${index}`)},{path:"tuple/raw-results.json",byte_size:60+index,sha256:d(`source-content-tuple-${index}`)}]}],quarantine:{root:`quarantine/${campaign_id}`,entries:[],aggregate_sha256:d("pending")},superseded_by:"pending"};
    resealQuarantine(predecessor);return predecessor;
  });
  value.campaign_lineage={kind:predecessorCount===0?"clean":"failed_campaign_supersession",current_campaign_id:value.campaign_run_id,current_workflow:{campaign_id:value.campaign_run_id,inference_revision:value.inference_revision,sceneworks_revision:value.sceneworks_revision,repository:value.execution.repository,path:".github/workflows/starvector-terminal.yml",run_id:value.execution.workflow_run_id,run_attempt:value.execution.workflow_run_attempt,head_sha:value.execution.head_sha},failed_predecessors:predecessors,supersession_records:[]};
  rebuildEdges(value);value.producer.campaign_lineage_sha256=campaignLineageSha256(value.campaign_lineage);value.artifact_manifest=buildManifest(value);value.producer.artifact_manifest_sha256=value.artifact_manifest.aggregate_sha256;return value;
}
function resealLineage(value) { value.producer.campaign_lineage_sha256=campaignLineageSha256(value.campaign_lineage); }
function resealManifest(value) { value.artifact_manifest.aggregate_sha256=h(stable({campaign_run_id:value.artifact_manifest.campaign_run_id,entries:value.artifact_manifest.entries}));value.producer.artifact_manifest_sha256=value.artifact_manifest.aggregate_sha256; }
function resealV2(value) { resealLineage(value);value.artifact_manifest=buildManifest(value);value.producer.artifact_manifest_sha256=value.artifact_manifest.aggregate_sha256;return value; }
function addHistoricalOwnedInputs(value, kind) {
  const predecessor=value.campaign_lineage.failed_predecessors[0],artifact=predecessor.source_artifacts[0];
  const cases=kind==="hostile"?value.hostile_sanitizer.cases:value.prompt_composition.cases;
  const payload=kind==="hostile"?hostilePayload:promptPayload,suffix=kind==="hostile"?"svg":"txt",width=kind==="hostile"?3:2;
  artifact.content_inventory.push(...cases.map((record,index)=>({path:`owned-inputs/${kind}/${String(index).padStart(width,"0")}.${suffix}`,byte_size:Buffer.byteLength(payload(index)),sha256:kind==="hostile"?record.input_sha256:record.prompt_sha256})));
  artifact.content_inventory.sort((left,right)=>left.path.localeCompare(right.path));
  resealQuarantine(predecessor);
  return resealV2(value);
}
function addAuthenticHistoricalInventory(value) {
  const predecessor=value.campaign_lineage.failed_predecessors[0],artifact=predecessor.source_artifacts[0];
  artifact.content_inventory.push(
    ...value.hostile_sanitizer.cases.map((record,index)=>({path:`owned-inputs/hostile/${String(index).padStart(3,"0")}.svg`,byte_size:Buffer.byteLength(hostilePayload(index)),sha256:record.input_sha256})),
    ...["controller-failure.json","controller-transcript.sha256","preflight-provenance.json","product-service-api.stdout.log","product-service-worker.stdout.log","product-service-logs.json","product-service-logs.sha256","route-command-transcript.json","tuple-controller.json"].map((path,index)=>({path,byte_size:100+index,sha256:d(`authentic-history-${path}`)})),
    {path:"product-service-api.stderr.log",byte_size:0,sha256:EMPTY_SHA256},
    {path:"product-service-worker.stderr.log",byte_size:0,sha256:EMPTY_SHA256},
  );
  artifact.content_inventory.sort((left,right)=>left.path.localeCompare(right.path));
  assert.equal(artifact.content_inventory.length,213);
  resealQuarantine(predecessor);
  return resealV2(value);
}
const validate = (value, byteSizes = fixtureByteSize)=>validateReceipt(value,validatePlan(corpus),INFERENCE,SCENEWORKS,corpus,value.schema_version === 2 ? byteSizes : undefined);
test("checked-in corpus carries exact selected and generated content identities",()=>assert.match(validatePlan(corpus),/^[0-9a-f]{64}$/));
test("coordinate/dimension hostile variants cover all terminal bounds",()=>{const variants=Array.from({length:10},(_,offset)=>hostilePayload(190+offset));assert(variants.some((value)=>value.includes("width=")));assert(variants.some((value)=>value.includes("height=")));assert(variants.some((value)=>value.includes("viewBox=\"0 0 ")));assert(variants.some((value)=>value.includes("<path d=\"M100000")));assert(variants.some((value)=>value.includes("viewBox=\"100000")));});
test("receipt accepts raw evidence for four providers, hostile suite, and prompt lineage",()=>validateReceipt(receipt(),validatePlan(corpus),INFERENCE,SCENEWORKS,corpus));
test("receipt rejects omitted/duplicated content, forged summaries, and false acceptance",()=>{let value=receipt();value.runs[0].image_quality.cases.pop();assert.throws(()=>validateReceipt(value,validatePlan(corpus),INFERENCE,SCENEWORKS,corpus),/120 ordered/);value=receipt();value.runs[0].image_quality.cases[119].case_index=118;assert.throws(()=>validateReceipt(value,validatePlan(corpus),INFERENCE,SCENEWORKS,corpus),/order/);value=receipt();value.runs[0].image_quality.median_ssim=.99;assert.throws(()=>validateReceipt(value,validatePlan(corpus),INFERENCE,SCENEWORKS,corpus),/keys differ/);value=receipt();value.hostile_sanitizer.cases[3].result_contains_inline_svg=true;assert.throws(()=>validateReceipt(value,validatePlan(corpus),INFERENCE,SCENEWORKS,corpus),/hostile evidence/);value=receipt();value.prompt_composition.cases[0].alignment_loss=.001;assert.throws(()=>validateReceipt(value,validatePlan(corpus),INFERENCE,SCENEWORKS,corpus),/forged alignment/);});
test("receipt rejects corpus drift, mixed artifacts/transcripts, and run provenance",()=>{let value=receipt();value.hostile_sanitizer.cases[0].input_sha256=d("drift");assert.throws(()=>validateReceipt(value,validatePlan(corpus),INFERENCE,SCENEWORKS,corpus),/hostile evidence/);value=receipt();value.runs[0].image_quality.cases[0].provider_transcript_sha256=d("foreign");assert.throws(()=>validateReceipt(value,validatePlan(corpus),INFERENCE,SCENEWORKS,corpus),/artifact manifest missing/);value=receipt();value.artifact_manifest.campaign_run_id="other";assert.throws(()=>validateReceipt(value,validatePlan(corpus),INFERENCE,SCENEWORKS,corpus),/mixed run/);value=receipt();value.execution.clean_tree=false;assert.throws(()=>validateReceipt(value,validatePlan(corpus),INFERENCE,SCENEWORKS,corpus),/execution provenance/);});
test("accepted image output requires a complete-root or EOS finish",()=>{const value=receipt();value.runs[0].image_quality.cases[0].finish_reason="token_limit";assert.throws(()=>validateReceipt(value,validatePlan(corpus),INFERENCE,SCENEWORKS,corpus),/non-complete finish/);});
test("prompt alignment permits an improved preview and paired bootstrap preserves correlation",()=>{const value=receipt();value.prompt_composition.cases[0].raster_prompt_cosine=.80;value.prompt_composition.cases[0].preview_prompt_cosine=.90;value.prompt_composition.cases[0].alignment_loss=-.10;value.artifact_manifest=buildManifest(value);value.producer.artifact_manifest_sha256=value.artifact_manifest.aggregate_sha256;validateReceipt(value,validatePlan(corpus),INFERENCE,SCENEWORKS,corpus);const paired=Array.from({length:120},(_,index)=>index<60?{one:.2,eight:.1}:{one:.8,eight:.7});const unpaired=paired.map((entry,index)=>({one:entry.one,eight:paired[(index+60)%paired.length].eight}));assert.notEqual(pairedBootstrapLowerBound(paired),pairedBootstrapLowerBound(unpaired));});

test("V1 remains valid without lineage fields and retains containment manifest semantics",()=>{const value=receipt();value.artifact_manifest.entries.push({path:"legacy/extra",byte_size:1,sha256:d("legacy-extra")});resealManifest(value);validate(value);});
test("V2 accepts clean, one-predecessor, and cross-pin multi-predecessor lineage",()=>{for(const count of [0,1,2]){const value=v2Receipt(count);assert.equal(value.campaign_lineage.failed_predecessors.length,count);if(count>0)assert.notEqual(value.campaign_lineage.failed_predecessors[0].inference_revision,value.inference_revision);validate(value);}});

test("V2 rejects omitted lineage and current-id cross-field mismatch",()=>{let value=v2Receipt(1);delete value.campaign_lineage.failed_predecessors[0].failure;resealLineage(value);assert.throws(()=>validate(value),/keys differ/);value=v2Receipt(1);value.campaign_lineage.current_campaign_id="another-current";resealLineage(value);assert.throws(()=>validate(value),/lineage identity/);});
test("V2 rejects supersession forks, cycles, and reordered predecessor history",()=>{let value=v2Receipt(2);value.campaign_lineage.failed_predecessors[0].superseded_by="forked-campaign";resealLineage(value);assert.throws(()=>validate(value),/fork\/head\/pin/);value=v2Receipt(2);const first=value.campaign_lineage.failed_predecessors[0],last=value.campaign_lineage.failed_predecessors[1],edge=value.campaign_lineage.supersession_records[1];last.superseded_by=first.campaign_id;edge.successor_campaign_id=first.campaign_id;edge.successor_inference_revision=first.inference_revision;edge.successor_sceneworks_revision=first.sceneworks_revision;edge.authority.path=`lineage/supersession-records/${last.campaign_id}-to-${first.campaign_id}.json`;resealLineage(value);assert.throws(()=>validate(value),/fork\/head\/pin/);value=v2Receipt(2);value.campaign_lineage.failed_predecessors.reverse();rebuildEdges(value);resealLineage(value);assert.throws(()=>validate(value),/oldest-to-newest/);});
test("V2 rejects duplicate campaigns and replayed workflow runs or GitHub artifacts",()=>{let value=v2Receipt(2);value.campaign_lineage.failed_predecessors[1].campaign_id=value.campaign_lineage.failed_predecessors[0].campaign_id;resealLineage(value);assert.throws(()=>validate(value),/duplicate\/current predecessor|fork\/head\/pin/);value=v2Receipt(2);value.campaign_lineage.failed_predecessors[1].workflow.run_id=value.campaign_lineage.failed_predecessors[0].workflow.run_id;resealQuarantine(value.campaign_lineage.failed_predecessors[1]);resealLineage(value);assert.throws(()=>validate(value),/duplicate\/replayed predecessor workflow/);value=v2Receipt(2);value.campaign_lineage.failed_predecessors[1].source_artifacts[0].id=value.campaign_lineage.failed_predecessors[0].source_artifacts[0].id;resealQuarantine(value.campaign_lineage.failed_predecessors[1]);resealLineage(value);assert.throws(()=>validate(value),/duplicate\/replayed source artifact/);});
test("V2 rejects successful predecessors and malformed workflow provenance",()=>{let value=v2Receipt(1);value.campaign_lineage.failed_predecessors[0].workflow.conclusion="success";resealQuarantine(value.campaign_lineage.failed_predecessors[0]);resealLineage(value);assert.throws(()=>validate(value),/workflow provenance/);value=v2Receipt(1);value.campaign_lineage.failed_predecessors[0].workflow.repository="Elsewhere/Repo";resealQuarantine(value.campaign_lineage.failed_predecessors[0]);resealLineage(value);assert.throws(()=>validate(value),/workflow provenance/);value=v2Receipt(1);value.campaign_lineage.failed_predecessors[0].workflow.path="../unsafe.yml";resealLineage(value);assert.throws(()=>validate(value),/safe canonical path/);});
test("V2 rejects predecessor head and supersession pin mismatches",()=>{let value=v2Receipt(1);value.campaign_lineage.failed_predecessors[0].workflow.head_sha="9".repeat(40);resealQuarantine(value.campaign_lineage.failed_predecessors[0]);resealLineage(value);assert.throws(()=>validate(value),/workflow provenance/);value=v2Receipt(1);value.campaign_lineage.supersession_records[0].predecessor_inference_revision="9".repeat(40);resealLineage(value);assert.throws(()=>validate(value),/fork\/head\/pin/);});
test("V2 rejects malformed artifact, marker, failure, and quarantine aggregate evidence",()=>{let value=v2Receipt(1);value.campaign_lineage.failed_predecessors[0].source_artifacts[0].digest=`sha256:${"A".repeat(64)}`;resealLineage(value);assert.throws(()=>validate(value),/artifact 0 digest/);value=v2Receipt(1);value.campaign_lineage.failed_predecessors[0].markers.tuple.path="../tuple.json";resealLineage(value);assert.throws(()=>validate(value),/safe canonical path/);value=v2Receipt(1);value.campaign_lineage.failed_predecessors[0].failure.tuple="unknown:1b";resealLineage(value);assert.throws(()=>validate(value),/failure identity/);value=v2Receipt(1);value.campaign_lineage.failed_predecessors[0].quarantine.aggregate_sha256=d("forged-aggregate");resealLineage(value);assert.throws(()=>validate(value),/quarantine aggregate/);});
test("V2 rejects incomplete quarantine copies and duplicate historical paths",()=>{let value=v2Receipt(1);value.campaign_lineage.failed_predecessors[0].quarantine.entries.pop();resealLineage(value);assert.throws(()=>validate(value),/exact sorted closure/);value=v2Receipt(1);const predecessor=value.campaign_lineage.failed_predecessors[0];predecessor.source_artifacts[0].content_inventory[1].path=predecessor.source_artifacts[0].content_inventory[0].path;resealLineage(value);assert.throws(()=>validate(value),/content inventory must be complete, sorted, and unique/);});

test("V2 manifest rejects extra, missing, duplicate, mixed, and unsorted entries",()=>{let value=v2Receipt(1);value.artifact_manifest.entries.push({path:"lineage/unreviewed.json",byte_size:1,sha256:d("extra")});value.artifact_manifest.entries.sort((a,b)=>a.path.localeCompare(b.path));resealManifest(value);assert.throws(()=>validate(value),/exact V2 path\/size\/digest closure/);value=v2Receipt(1);value.artifact_manifest.entries=value.artifact_manifest.entries.filter((entry)=>entry.path!=="lineage/campaign-lineage.json");resealManifest(value);assert.throws(()=>validate(value),/exact V2 path\/size\/digest closure/);value=v2Receipt(1);value.artifact_manifest.entries.push({...value.artifact_manifest.entries[0]});resealManifest(value);assert.throws(()=>validate(value),/artifact entry invalid/);value=v2Receipt(1);value.artifact_manifest.entries.find((entry)=>entry.path==="lineage/campaign-lineage.json").sha256=d("foreign-lineage");resealManifest(value);assert.throws(()=>validate(value),/exact V2 path\/size\/digest closure/);value=v2Receipt(1);value.artifact_manifest.entries.reverse();resealManifest(value);assert.throws(()=>validate(value),/not sorted/);});
test("V2 lineage mutation is rejected even when the producer lineage hash is recomputed",()=>{const value=v2Receipt(1);value.campaign_lineage.failed_predecessors[0].failure.code="different-failure";resealLineage(value);assert.throws(()=>validate(value),/exact V2 path\/size\/digest closure/);});
test("V2 permits matching deterministic current output bytes in historical artifact inventory",()=>{
  const value=v2Receipt(1), predecessor=value.campaign_lineage.failed_predecessors[0];
  predecessor.inference_revision=value.inference_revision;
  predecessor.source_artifacts[0].content_inventory.push({path:"corpus/input.png",byte_size:2048,sha256:value.runs[0].image_quality.cases[0].input_png_sha256},{path:"output/preview.png",byte_size:4096,sha256:value.runs[0].image_quality.cases[0].preview_png_sha256});
  predecessor.source_artifacts[0].content_inventory.sort((a,b)=>a.path.localeCompare(b.path));
  resealQuarantine(predecessor);rebuildEdges(value);resealV2(value);validate(value);
});
test("V2 permits all recomputed hostile inputs to recur in complete historical artifact content",()=>{const value=addHistoricalOwnedInputs(v2Receipt(1),"hostile");const inventory=value.campaign_lineage.failed_predecessors[0].source_artifacts[0].content_inventory.filter((entry)=>entry.path.startsWith("owned-inputs/hostile/"));assert.equal(inventory.length,200);validate(value);});
test("V2 permits recomputed deterministic prompt inputs to recur in complete historical artifact content",()=>{const value=addHistoricalOwnedInputs(v2Receipt(1),"prompt");const inventory=value.campaign_lineage.failed_predecessors[0].source_artifacts[0].content_inventory.filter((entry)=>entry.path.startsWith("owned-inputs/prompt/"));assert.equal(inventory.length,60);validate(value);});
test("V2 accepts an authentic 213-file historical inventory with exact empty service stderr logs",()=>{const value=addAuthenticHistoricalInventory(v2Receipt(1));const inventory=value.campaign_lineage.failed_predecessors[0].source_artifacts[0].content_inventory;const empty=inventory.filter((entry)=>entry.byte_size===0);assert.deepEqual(empty.map((entry)=>entry.path),["product-service-api.stderr.log","product-service-worker.stderr.log"]);assert(empty.every((entry)=>entry.sha256===EMPTY_SHA256));validate(value);});
test("V2 rejects zero-byte structural quarantine evidence",()=>{
  let value=v2Receipt(1),predecessor=value.campaign_lineage.failed_predecessors[0];
  predecessor.markers.campaign.size=0;resealQuarantine(predecessor);resealLineage(value);assert.throws(()=>validate(value),/bounded safe integer/);
  value=v2Receipt(1);predecessor=value.campaign_lineage.failed_predecessors[0];predecessor.source_artifacts[0].size=0;resealQuarantine(predecessor);resealLineage(value);assert.throws(()=>validate(value),/bounded safe integer/);
  value=v2Receipt(1);predecessor=value.campaign_lineage.failed_predecessors[0];predecessor.quarantine.entries.find((entry)=>entry.path.endsWith("/workflow-run.json")).size=0;predecessor.quarantine.aggregate_sha256=h(stable({root:predecessor.quarantine.root,entries:predecessor.quarantine.entries}));resealLineage(value);assert.throws(()=>validate(value),/zero-byte content is not bound/);
  value=v2Receipt(1);value.campaign_lineage.supersession_records[0].authority.size=0;resealLineage(value);assert.throws(()=>validate(value),/bounded safe integer/);
  value=v2Receipt(1);predecessor=value.campaign_lineage.failed_predecessors[0];value.artifact_manifest.entries.find((entry)=>entry.path===`${predecessor.quarantine.root}/aggregate.json`).byte_size=0;resealManifest(value);assert.throws(()=>validate(value),/exact V2 path\/size\/digest closure/);
});
test("V2 authentic zero-byte inventory requires exact content binding and complete quarantine closure",()=>{let value=addAuthenticHistoricalInventory(v2Receipt(1)),predecessor=value.campaign_lineage.failed_predecessors[0];predecessor.quarantine.entries.find((entry)=>entry.path.endsWith("product-service-api.stderr.log")).sha256=d("forged-empty-log");predecessor.quarantine.aggregate_sha256=h(stable({root:predecessor.quarantine.root,entries:predecessor.quarantine.entries}));resealLineage(value);assert.throws(()=>validate(value),/zero-byte content is not bound/);value=addAuthenticHistoricalInventory(v2Receipt(1));predecessor=value.campaign_lineage.failed_predecessors[0];predecessor.quarantine.entries=predecessor.quarantine.entries.filter((entry)=>!entry.path.endsWith("product-service-worker.stderr.log"));predecessor.quarantine.aggregate_sha256=h(stable({root:predecessor.quarantine.root,entries:predecessor.quarantine.entries}));resealLineage(value);assert.throws(()=>validate(value),/exact sorted closure/);});

test("V2 owned-input overlap still requires the exact complete quarantine closure",()=>{const value=addHistoricalOwnedInputs(v2Receipt(1),"hostile");value.campaign_lineage.failed_predecessors[0].quarantine.entries.pop();resealLineage(value);assert.throws(()=>validate(value),/exact sorted closure/);});

test("V2 seals the canonical current workflow and rejects noncanonical or non-newer run IDs",()=>{let value=v2Receipt(1);const entry=value.artifact_manifest.entries.find((candidate)=>candidate.path==="lineage/current-workflow.json");assert.equal(entry.byte_size,Buffer.byteLength(stable(value.campaign_lineage.current_workflow)));assert.equal(entry.sha256,h(stable(value.campaign_lineage.current_workflow)));value.campaign_lineage.current_workflow.repository="Elsewhere/Repo";resealLineage(value);assert.throws(()=>validate(value),/current workflow provenance/);value=v2Receipt(1);value.campaign_lineage.current_workflow.path=".github/workflows/other.yml";resealLineage(value);assert.throws(()=>validate(value),/current workflow provenance/);value=v2Receipt(1);value.execution.workflow_run_id="500";value.campaign_lineage.current_workflow.run_id="500";resealLineage(value);assert.throws(()=>validate(value),/older than current workflow run/);value=v2Receipt(1);value.execution.workflow_run_id="01000";value.campaign_lineage.current_workflow.run_id="01000";resealLineage(value);assert.throws(()=>validate(value),/canonical bounded decimal id/);value=v2Receipt(1);value.campaign_lineage.failed_predecessors[0].workflow.run_id="01000";resealLineage(value);assert.throws(()=>validate(value),/canonical bounded decimal id/);});

test("V2 binds each GitHub artifact to its workflow and complete extracted inventory",()=>{let value=v2Receipt(1);let artifact=value.campaign_lineage.failed_predecessors[0].source_artifacts[0];artifact.workflow_run_id="1001";artifact.api_workflow_run.id="1001";resealLineage(value);assert.throws(()=>validate(value),/identity\/provenance/);value=v2Receipt(1);artifact=value.campaign_lineage.failed_predecessors[0].source_artifacts[0];artifact.api_workflow_run.id="1001";resealLineage(value);assert.throws(()=>validate(value),/does not bind artifact workflow run/);value=v2Receipt(1);artifact=value.campaign_lineage.failed_predecessors[0].source_artifacts[0];artifact.id="07000";resealLineage(value);assert.throws(()=>validate(value),/canonical bounded decimal id/);value=v2Receipt(1);artifact=value.campaign_lineage.failed_predecessors[0].source_artifacts[0];artifact.head_sha="9".repeat(40);artifact.api_workflow_run.head_sha=artifact.head_sha;resealLineage(value);assert.throws(()=>validate(value),/identity\/provenance/);});

test("V2 quarantine aggregate and manifest share one canonical serialized payload",()=>{let value=v2Receipt(1);let predecessor=value.campaign_lineage.failed_predecessors[0];let payload=stable({root:predecessor.quarantine.root,entries:predecessor.quarantine.entries});let aggregate=value.artifact_manifest.entries.find((entry)=>entry.path===`${predecessor.quarantine.root}/aggregate.json`);assert.equal(aggregate.byte_size,Buffer.byteLength(payload));assert.equal(aggregate.sha256,h(payload));for(const field of ["byte_size","sha256","path"]){value=v2Receipt(1);predecessor=value.campaign_lineage.failed_predecessors[0];aggregate=value.artifact_manifest.entries.find((entry)=>entry.path===`${predecessor.quarantine.root}/aggregate.json`);if(field==="byte_size")aggregate.byte_size+=1;else if(field==="sha256")aggregate.sha256=d("mutated-quarantine-aggregate");else aggregate.path=`${predecessor.quarantine.root}/aggregate-mutated.json`;value.artifact_manifest.entries.sort((left,right)=>left.path.localeCompare(right.path));resealManifest(value);assert.throws(()=>validate(value),/exact V2 path\/size\/digest closure/);}});

test("V2 rejects unsafe numeric, string, path, and array bounds",()=>{let value=v2Receipt(1);const hugeId="9".repeat(10000);value.execution.workflow_run_id=hugeId;value.campaign_lineage.current_workflow.run_id=hugeId;resealLineage(value);assert.throws(()=>validate(value),/canonical bounded decimal id/);value=v2Receipt(1);value.execution.workflow_run_attempt=1001;value.campaign_lineage.current_workflow.run_attempt=1001;resealLineage(value);assert.throws(()=>validate(value),/bounded safe integer/);value=v2Receipt(1);value.campaign_lineage.failed_predecessors[0].markers.campaign.size=2**53;resealLineage(value);assert.throws(()=>validate(value),/bounded safe integer/);value=v2Receipt(1);value.campaign_lineage.failed_predecessors=Array.from({length:33},()=>({}));resealLineage(value);assert.throws(()=>validate(value),/identity\/count/);value=v2Receipt(1);value.campaign_lineage.failed_predecessors[0].source_artifacts=Array.from({length:33},()=>({}));resealLineage(value);assert.throws(()=>validate(value),/source artifact count/);value=v2Receipt(1);value.campaign_lineage.failed_predecessors[0].source_artifacts[0].content_inventory=Array.from({length:20001},()=>({path:"x",byte_size:1,sha256:d("x")}));resealLineage(value);assert.throws(()=>validate(value),/content inventory count/);value=v2Receipt(1);value.campaign_lineage.failed_predecessors[0].markers.campaign.path="x".repeat(1025);resealLineage(value);assert.throws(()=>validate(value),/safe canonical path/);value=v2Receipt(1);value.producer.command="x".repeat(4097);assert.throws(()=>validate(value),/producer command length/);value=v2Receipt(0);value.artifact_manifest.entries=Array.from({length:100001},()=>({path:"x",byte_size:1,sha256:d("x")}));assert.throws(()=>validate(value),/manifest mixed run\/count/);});

test("CLI rejects an oversized receipt before JSON parsing",()=>{const directory=mkdtempSync(join(tmpdir(),"starvector-receipt-"));try{const corpusPath=join(directory,"corpus.json"),receiptPath=join(directory,"receipt.json");writeFileSync(corpusPath,JSON.stringify(corpus));writeFileSync(receiptPath,"");truncateSync(receiptPath,MAX_RECEIPT_BYTES+1);const result=spawnSync(process.execPath,["scripts/release/starvector_terminal_evidence.mjs","validate-receipt","--corpus",corpusPath,"--receipt",receiptPath,"--inference-revision",INFERENCE,"--sceneworks-revision",SCENEWORKS],{cwd:process.cwd(),encoding:"utf8"});assert.notEqual(result.status,0);assert.match(result.stderr,/receipt exceeds 8388608 byte input limit/);}finally{rmSync(directory,{recursive:true,force:true});}});

test("V2 uses measured current artifact sizes with complete immutable overlap and empty historical logs",()=>{
  const directory=mkdtempSync(join(tmpdir(),"starvector-sized-current-"));
  try {
    const value=addAuthenticHistoricalInventory(v2Receipt(1));
    const sizes=new Map();
    for(const entry of currentArtifactReferences(value,corpus)) {
      const hostileMatch=entry.path.match(/^hostile\/(\d+)\/input$/),promptMatch=entry.path.match(/^prompt\/(\d+)\/prompt_sha256$/);
      const bytes=hostileMatch?hostilePayload(Number(hostileMatch[1])):promptMatch?promptPayload(Number(promptMatch[1])):`realistic current artifact ${entry.path}\n${"data ".repeat(29)}`;
      const file=join(directory,...entry.path.split("/"));mkdirSync(join(file,".."),{recursive:true});writeFileSync(file,bytes);sizes.set(entry.path,statSync(file).size);
    }
    value.artifact_manifest=buildManifest(value,sizes);value.producer.artifact_manifest_sha256=value.artifact_manifest.aggregate_sha256;
    assert(value.artifact_manifest.entries.filter(entry=>!entry.path.startsWith("lineage/")&&!entry.path.startsWith("quarantine/")).every(entry=>entry.byte_size>1));
    validate(value,sizes);
    const target=value.artifact_manifest.entries.find(entry=>entry.path==="producer/transcript");target.byte_size=1;resealManifest(value);
    assert.throws(()=>validate(value,sizes),/exact V2 path\/size\/digest closure/);
    const immutable=currentArtifactReferences(value,corpus).filter(entry=>entry.path.startsWith("hostile/")||entry.path.match(/^prompt\/\d+\/prompt_sha256$/));
    assert.equal(immutable.length,260);assert.deepEqual(artifactByteSizesFromFiles(directory,immutable),new Map(immutable.map(entry=>[entry.path,sizes.get(entry.path)])));
  } finally {rmSync(directory,{recursive:true,force:true});}
});
test("V2 rejects absent, incomplete, and invalid external current sizes",()=>{
  const value=v2Receipt();
  assert.throws(()=>buildArtifactManifest(value,corpus),/requires actual current artifact byte sizes/);
  assert.throws(()=>validateReceipt(value,validatePlan(corpus),INFERENCE,SCENEWORKS,corpus),/requires actual current artifact byte sizes/);
  for(const size of [undefined,-1,1.5,Number.MAX_SAFE_INTEGER+1]) assert.throws(()=>buildManifest(value,()=>size),/bounded safe integer/);
  assert.throws(()=>validate(value,new Map()),/bounded safe integer/);
});
test("canonical file verification rejects rehashed size lies, tampering, missing files, and symlink escapes",()=>{
  const directory=mkdtempSync(join(tmpdir(),"starvector-file-binding-"));
  try {
    const payload="real transcript\n".repeat(9000),entry={path:"producer/transcript",sha256:h(payload)};
    mkdirSync(join(directory,"producer"));writeFileSync(join(directory,entry.path),payload);
    assert.equal(artifactByteSizesFromFiles(directory,[entry]).get(entry.path),Buffer.byteLength(payload));
    assert.throws(()=>artifactByteSizesFromFiles(directory,[{...entry,byte_size:1}]),/byte size mismatch/);
    writeFileSync(join(directory,entry.path),payload.replace("real","fake"));assert.throws(()=>artifactByteSizesFromFiles(directory,[entry]),/digest mismatch/);
    rmSync(join(directory,entry.path));assert.throws(()=>artifactByteSizesFromFiles(directory,[entry]),/ENOENT/);
    writeFileSync(join(directory,"outside"),payload);symlinkSync(join(directory,"outside"),join(directory,entry.path));assert.throws(()=>artifactByteSizesFromFiles(directory,[entry]),/regular contained file/);
    rmSync(join(directory,"producer"),{recursive:true});mkdirSync(join(directory,"elsewhere"));writeFileSync(join(directory,"elsewhere","transcript"),payload);symlinkSync(join(directory,"elsewhere"),join(directory,"producer"));assert.throws(()=>artifactByteSizesFromFiles(directory,[entry]),/regular contained file/);
    assert.throws(()=>artifactByteSizesFromFiles(directory,[{...entry,path:"../outside"}]),/safe canonical path/);
  } finally {rmSync(directory,{recursive:true,force:true});}
});
test("CLI requires independent canonical evidence for V2 while V1 remains compatible",()=>{
  const directory=mkdtempSync(join(tmpdir(),"starvector-v2-cli-"));
  try {
    const receiptPath=join(directory,"receipt.json"),args=["scripts/release/starvector_terminal_evidence.mjs","validate-receipt","--corpus","release/starvector-terminal-corpus-v1.json","--receipt",receiptPath,"--inference-revision",INFERENCE,"--sceneworks-revision",SCENEWORKS];
    writeFileSync(receiptPath,JSON.stringify(v2Receipt(1)));let result=spawnSync(process.execPath,args,{encoding:"utf8"});assert.notEqual(result.status,0);assert.match(result.stderr,/missing --evidence-root/);
    writeFileSync(receiptPath,JSON.stringify(receipt()));result=spawnSync(process.execPath,args,{encoding:"utf8"});assert.equal(result.status,0,result.stderr);
  } finally {rmSync(directory,{recursive:true,force:true});}
});

test("V2 requires genuine frozen upstream parity and the four-source case mapping",()=>{
  for (const mutate of [
    value=>{delete value.runs[0].deterministic_parity.upstream_reference;},
    value=>{value.runs[0].deterministic_parity.upstream_reference.implementation_revision="a".repeat(40);},
    value=>{value.runs[0].deterministic_parity.upstream_reference.checkpoint_inventory_sha256=d("wrong-model");},
    value=>{value.runs[0].deterministic_parity.cases[5].input_png_sha256=value.runs[0].image_quality.cases[5].input_png_sha256;},
    value=>{value.runs[0].deterministic_parity.cases[0].rendered_ssim=.994;},
    value=>{value.runs[0].deterministic_parity.cases[0].second_preview_png_sha256=d("native-repeat");},
  ]) {const value=v2Receipt();mutate(value);assert.throws(()=>validate(value),/upstream|parity.*keys differ/);}
});
test("V2 applies the 120-second p95 requirement only to 1B while V1 remains historical",()=>{
  const value=v2Receipt();for(const run of value.runs.filter(run=>run.tier==="8b")) for(const record of run.image_quality.cases)record.latency_seconds=121;
  resealV2(value);validate(value);
  for(const record of value.runs[0].image_quality.cases)record.latency_seconds=121;
  assert.throws(()=>validate(value),/image threshold/);
  const legacy=receipt();for(const record of legacy.runs[1].image_quality.cases)record.latency_seconds=121;
  assert.throws(()=>validate(legacy),/image threshold/);
});

test("V2 admits the actual caller workflow and a newer attempt at the same pin",()=>{
  const value=v2Receipt(1),previous=value.campaign_lineage.failed_predecessors[0];
  previous.inference_revision=value.inference_revision;
  previous.workflow.run_id=value.execution.workflow_run_id;
  previous.source_artifacts.forEach(artifact=>{artifact.workflow_run_id=previous.workflow.run_id;artifact.api_workflow_run.id=previous.workflow.run_id;});
  value.execution.workflow_run_attempt=2;value.campaign_lineage.current_workflow.run_attempt=2;
  value.campaign_lineage.current_workflow.path=".github/workflows/server-candle-linux.yml";
  resealQuarantine(previous);rebuildEdges(value);resealV2(value);validate(value);
  value.execution.workflow_run_attempt=1;value.campaign_lineage.current_workflow.run_attempt=1;
  resealLineage(value);assert.throws(()=>validate(value),/duplicate\/replayed predecessor workflow/);
});

test("V2 canonical JSON key order is independent of producer object insertion order",()=>{
  const value=v2Receipt(1);
  // Recovery persists stable canonical JSON, while JS builders insert path/size/hash keys.
  // Parse those exact canonical bytes to exercise the production serialization boundary.
  validate(JSON.parse(stable(value)));
  const reordered=JSON.parse(stable(value));
  reordered.campaign_lineage.failed_predecessors[0].quarantine.entries.reverse();
  resealLineage(reordered);
  assert.throws(()=>validate(reordered),/exact sorted closure/);
});
