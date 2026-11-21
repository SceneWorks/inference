"""Candidate-owned, test-only SCAIL-2 probes built against each arm's own sources."""
from functools import partial
import hashlib
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE = "c7911db0e6cdcc41762326e2f8aad7e60634c2a1"
HERE = "scripts/ci/scail_pair/"
# SHA-256 of the unedited seeded-denoise span from each generate.rs at BASE.
BASE_DENOISE = {
    "mlx": "f6a0766d56c2ccb15b9a9fbd449b3900659f6184cf9168ac960ac2ec165b0407",
    "candle-cuda": "d5e271e2a7c9506d90534e2c9225923d02adc38a7dd7611e5544945f1e309f65",
}


def _git(revision, path):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT).decode()


def _span(source, start, end):
    if source.count(start) != 1:
        raise ValueError(f"expected one production anchor {start!r}")
    lo = source.index(start)
    return source[lo:source.index(end, lo + len(start))]


def _once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"expected one binding for {old!r}")
    return text.replace(old, new)


def _digest(source):
    return hashlib.sha256(source.encode()).hexdigest()


def _observed_model(model, mlx):
    # These methods are reachable only on the first forward of the untimed warmup.
    # Keep the original forward and cached RoPE method intact for the measured path.
    forward = _span(model, "    pub fn forward(&self, inp: &Scail2Inputs)", "\n    }\n}") + "\n    }"
    forward = _once(forward, "pub fn forward(&self, inp: &Scail2Inputs)",
                    "pub(crate) fn forward_observed(&self, inp: &Scail2Inputs, packing: &mut Packing)")
    # Record which input the native concatenation actually puts first inside
    # the one shared reference/video token block. Pointer identity avoids
    # reimplementing the content or guessing the order from the result shape.
    def observe_refvid(match):
        return ("let observed_inputs = " + match[2] + ";\n"
                '        for input in observed_inputs { packing.refvid.push(if std::ptr::eq(*input, &ref20) {'
                '"reference" } else { assert!(std::ptr::eq(*input, &x20)); "video" }); }\n'
                '        ' + match[1] + "observed_inputs")
    forward, n = re.subn(r"(let refvid = (?:concatenate_axis|Tensor::cat)\()(&\[&(?:ref20|x20), &(?:ref20|x20)\])",
                         observe_refvid, forward)
    if n != 1:
        raise ValueError("expected one native reference/video concatenation")
    for var, role in (("ar_emb", "additional_ref"), ("refvid_emb", "ref_video"), ("pose_emb", "pose")):
        push = f"tok_list.push({var});"
        forward = _once(forward, push, f'packing.tokens.push("{role}"); {push}')
    if mlx:
        chunks = forward
        start = "let mut addref_count = 0usize;"
    else:
        forward = _once(forward, "self.rope_tables(rope_t, rope_h, rope_w, addref_count, h_shift)",
                        "self.rope_tables_observed(rope_t, rope_h, rope_w, addref_count, h_shift, packing)")
        chunks = _span(model, "    fn rope_tables(", "\n    }\n") + "\n    }"
        chunks = _once(chunks, "fn rope_tables(", "fn rope_tables_observed(")
        chunks = _once(chunks, "h_shift: usize,", "h_shift: usize, packing: &mut Packing,")
        start = "let base_video_shift = 1usize;"
    signature = "grid, shift, pool" if mlx else "grid, shift, pool, dev: &Device"
    invoke = "grid, shift, pool" if mlx else "grid, shift, pool, dev"
    chunks, n = re.subn(r"self\s*\.rope\s*\.chunk\(", "observed_chunk(", chunks)
    if n != 4:
        raise ValueError("expected four native RoPE chunk bindings")
    chunks = _once(chunks, start, start + f"""
            let mut observed_chunk = |{signature}| {{
                packing.rope.push(serde_json::json!({{"grid":grid,"shift":shift,"pool":pool}}));
                self.rope.chunk({invoke})
            }};""")
    methods = chunks if mlx else forward + "\n" + chunks
    return ('\n#[cfg(test)] use crate::generate::scail_pair_fixture::Packing;\n'
            + '#[cfg(test)] impl Scail2Dit {\n' + methods + '\n}\n')


def render(read, params, backend, arm):
    mlx = backend == "mlx"
    if backend not in ("mlx", "candle-cuda") or arm not in ("base", "head"):
        raise ValueError("unknown SCAIL comparison binding")
    family = "mlx-gen" if mlx else "candle-gen"
    src = f"crates/media/{family}/{family}-scail2/src/"
    generate = read(src + "generate.rs")
    model = read(src + "model.rs")
    # The seed, real native scheduler, CFG, forward, clean-history operations and
    # progress dispatch are copied verbatim from the arm under test. The head calls
    # its actual denoise_segment helper. The base predates that seam.
    body = _span(generate, "        // Seeded init noise (per-segment", (
        "        // F-096: the bf16" if mlx else
        "        // --- decode this segment"))
    if arm == "base" and _digest(body) != BASE_DENOISE[backend]:
        raise ValueError("pre-migration seeded denoise differs from its frozen source digest")
    if arm == "head":
        if body.count("let latent = denoise_segment(") != 1:
            raise ValueError("candidate's owning entry no longer calls denoise_segment")
        prefix = _span(generate, "        let mut sched = make_scheduler(job.sampler" if mlx
                       else "        let mut sched = FlowScheduler::new(job.sampler, job.steps, job.shift);",
                       "        let dit = match &mut dit" if mlx else "        // This route loads")
        finish = _span(generate, "        if let Some(active) = active {",
                       "        // --- decode this segment") if mlx else ""
        body = prefix + body + finish
    elif "let mut latent = apply_clean_history(&noise" not in body:
        raise ValueError("base no longer owns the inline denoise loop")
    prep = _span(generate, "    let cfg_disabled = job.guidance", ";") + ";"
    module = params(HERE + ("mlx_probe.rs.in" if mlx else "cuda_probe.rs.in"))
    module = _once(module, "@COMMON@", params(HERE + "common.rs.in"))
    module = _once(module, "@RUN@", params(HERE + "run.rs.in"))
    if arm == "base":
        segments = "build_segments(job.driving_frames.len(), job.segment_len, job.segment_overlap)"
        windows = "for (seg_idx, &(seg_start, seg_end)) in segments.iter().enumerate() {"
    else:
        segments = "job_segments(job)?"
        windows = _span(generate, "    for segment in &segments {",
                        "        let mut sched =")
    module = _once(module, "@SEGMENTS@", segments)
    module = _once(module, "@WINDOWS@", windows)
    module = _once(module, "@JOB_FIELDS@", "memory: None," if mlx else "")
    bindings = """
        let ref_latent = &condition.ref_latent;
        let additional_ref_masks = &condition.additional_ref_masks;
        let additional_ref_latent = &condition.additional_ref_latent;
        let clip_fea = &condition.clip_fea;
        let context = &condition.context;
        let context_null = &condition.context_null;
    """ if mlx or arm == "base" else "let conditioning = condition;"
    if mlx:
        bindings += "let num_segments = segments.len()" + (" as u32;" if arm == "base" else ";")
    else:
        bindings += "let dev = resource; let comps = Components { dit }; let seg_count = segments.len();"
    if arm == "head":
        bindings += "let mut scope: Option<&mut dyn MemoryRequestScope> = None;"
    module = _once(module, "@BINDINGS@", bindings)
    module = _once(module, "@CFG@", prep)
    digest = _digest(body)
    # Capture the actual first scheduler sample (a retained handle only). On the
    # untimed diagnostic sweep inspect every subsequent scheduler sample as well:
    # the pre-forward clean-history rewrite cannot mask a missing post-step pin.
    bind = ("let x = apply_clean_history(&latent, history_latent.as_ref())?;" if arm == "base"
            else "|i, x, latent| {")
    capture = """
                if initial_sample.is_none() { initial_sample = Some(latent.clone()); }
                if OBSERVE {
                    if let Some(history) = &history_latent {
                        assert_prefix(&values(&latent)?, &values(history)?);
                    }
                }
    """
    body = _once(body, bind, capture + bind if arm == "base" else bind + capture)
    module = _once(module, "@NATIVE@", "// production generate.rs spans sha256 " + digest + "\n" + body)
    if not mlx:
        module = _once(module, "@POOL@", params("scripts/ci/wan_pair/cuda_pool.rs.in"))
    changes = {src+"generate.rs": generate + '\n#[cfg(test)]\n#[path = "scail_pair_fixture.rs"]\npub(crate) mod scail_pair_fixture;\n',
               src+"model.rs": model + _observed_model(model, mlx)}
    if not mlx:
        pipeline = src+"pipeline.rs"
        changes[pipeline] = read(pipeline) + '''
#[cfg(test)]
pub(crate) fn scail_pair_dit(path: PathBuf, cfg: &Scail2Config, dev: &Device) -> CResult<Scail2Dit> {
    Ok(Scail2Dit::new(cpu_cast_mmap_var_builder(&[path], DType::F32, dev)?, cfg)?)
}
'''
    return module.encode(), {k:v.encode() for k,v in changes.items()}


def _render(head, backend, arm):
    return render(partial(_git, BASE if arm == "base" else head), partial(_git, head), backend, arm)


def overlay_source(head, backend, arm):
    return _render(head, backend, arm)[0]


def overlay_edits(head, backend, arm):
    return _render(head, backend, arm)[1]
