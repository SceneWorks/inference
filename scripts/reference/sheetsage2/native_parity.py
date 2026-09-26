#!/usr/bin/env python3
"""Reference fixtures for the native SheetSage2 / MERT-v2 port (sc-22996, epic sc-22988).

Reference tooling only, like ``run_experiment.py`` next to it: it runs upstream's pinned remote
Python on the torch **CPU** device in fp32 to produce the oracles the native Candle port in
``crates/audio/candle-audio-sheetsage2`` is compared against. Nothing here is a shipped path (epic
requirement E3).

Commands (the environment is the one in README.md; ``--work`` must be outside the repository):

* ``tiny`` — builds a *tiny* model with the exact upstream architecture (MERT2 mel front end,
  ConvNeXt-v2 subsampler, Conformer blocks, rank-r LoRA merged in fp32, layer mix, projection, BART
  decoder, grammar-masked greedy decoding) from seeded random weights, runs it on a seeded input and
  writes the weights, every intermediate state and the greedy tokens into the crate's committed
  ``testdata/tiny`` directory. No pretrained weights are involved, so the committed fixture is a few
  hundred KB and carries no model licence. CI compares the native port with it.
* ``grammar`` — upstream's ``PromptGrammarState`` allowed-token sets along the committed real token
  oracles (``synth_full``, ``real_full``), written to the crate's ``testdata``.
* ``tables`` — ``mir_eval.chord`` tables the native export path needs (chord pitch sets for every
  label of the pinned vocabulary), written to the crate's ``testdata``.
* ``real`` — real weights (the pinned snapshots must already be in the local Hugging Face cache):
  every MERT2 hidden state, the layer mix, the decoder memory and the first decoder logits for the
  ``synth`` and ``nav_ssb`` model-input arrays, written into ``--work`` (hundreds of MB; never
  committed). The native real-weight test measures its tolerances against these.
* ``long`` — the >300 s multi-window path sc-23003 left untested: a deterministic 358.2 s array
  concatenated from the digest-pinned ``nav_ssb`` and ``synth`` model-input arrays, transcribed by
  upstream at the head code revision. The array and its outputs go to ``--work``; the small text/MIDI
  oracles are copied to ``artifacts/long_multiwindow``.
* ``silence`` — 12 s of digital silence transcribed by upstream (head code): the token oracle of the
  case upstream silently turns into a rest-only score. Oracles go to ``artifacts/silence``.

Every real-weight command asserts that the model-input arrays still match the SHA-256 digests in
``artifacts/fixtures.json`` and that every parameter is on the CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
REPO = HERE.parents[2]
CRATE_TESTDATA = REPO / "crates" / "audio" / "candle-audio-sheetsage2" / "testdata"

SHEETSAGE2_REPO = "m-a-p/SheetSage2"
SHEETSAGE2_REVISION = "eab522a8168e8b8b8c4856bf8609cd86198f01fe"
SHEETSAGE2_HEAD_REVISION = "4f89269db831bdc1880124164a00d4f9385cd129"

TINY_SEED = 22996
TINY_SAMPLE_RATE = 1600
TINY_WINDOW_SECONDS = 1.0
TINY_SIGNAL_SECONDS = 0.8


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def upstream_package(revision: str):
    """Import the pinned remote Python of ``revision`` as a package, straight from the local cache.

    The snapshot directory is located through ``huggingface_hub`` in offline mode; its ``*.py`` files
    are symlinked into a temporary package so their relative imports resolve.
    """
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(SHEETSAGE2_REPO, revision=revision, local_files_only=True))
    root = Path(tempfile.mkdtemp(prefix="ss2ref-"))
    package = root / "ss2ref"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for path in snapshot.glob("*.py"):
        if path.name != "__init__.py":
            (package / path.name).symlink_to(path.resolve())
    sys.path.insert(0, str(root))
    import importlib

    modules = {}
    for name in ("configuration_mert2", "modeling_mert2", "configuration_sheetsage2",
                 "modeling_sheetsage2", "tokenization_sheetsage2", "generation_sheetsage2"):
        modules[name] = importlib.import_module(f"ss2ref.{name}")
    return modules, snapshot


def load_arrays(work_fixtures: Path) -> dict:
    """The digest-pinned 24 kHz mono float32 model-input arrays of sc-23003."""
    import numpy as np

    committed = json.loads((ARTIFACTS / "fixtures.json").read_text(encoding="utf-8"))
    arrays = {}
    for name, entry in committed.items():
        path = work_fixtures / entry["model_input"]["file"]
        if not path.is_file():
            raise SystemExit(f"{path}: missing; run `run_experiment.py fixtures --work ...` first")
        if sha256_file(path) != entry["model_input"]["sha256"]:
            raise SystemExit(f"{path}: SHA-256 differs from artifacts/fixtures.json")
        arrays[name] = np.fromfile(str(path), dtype="<f4")
    return arrays


# ------------------------------------------------------------------------------------------ tiny

def tiny(out: Path) -> None:
    import numpy as np
    import torch
    from safetensors.torch import save_file

    modules, _ = upstream_package(SHEETSAGE2_HEAD_REVISION)
    MERT2Config = modules["configuration_mert2"].MERT2Config
    MERT2Model = modules["modeling_mert2"].MERT2Model
    SheetSage2Config = modules["configuration_sheetsage2"].SheetSage2Config
    SheetSage2Model = modules["modeling_sheetsage2"].SheetSage2Model
    Tokenizer = modules["tokenization_sheetsage2"].SheetSage2Tokenizer
    generation = modules["generation_sheetsage2"]

    torch.manual_seed(TINY_SEED)
    torch.set_num_threads(1)
    tokenizer = Tokenizer(TINY_WINDOW_SECONDS, 100, "v1")
    backbone = MERT2Config(
        hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
        num_mel_bins=16, sampling_rate=TINY_SAMPLE_RATE, n_fft=128, win_length=128, hop_length=16,
        subsampling_channels=[16, 24, 32], subsampling_depths=[1, 2, 1],
        conv_depthwise_kernel_size=7, variant="fs", context_seconds=TINY_WINDOW_SECONDS,
    )
    backbone._attn_implementation = "sdpa"
    backbone_dict = backbone.to_dict()
    for key in ("_name_or_path", "auto_map", "transformers_version", "architectures",
                "_attn_implementation_autoset"):
        backbone_dict.pop(key, None)
    config = SheetSage2Config(
        vocab_size=tokenizer.n_tokens, hidden_size=16, decoder_layers=2, num_attention_heads=4,
        intermediate_size=32, decoder_dropout=0.0, input_audio_length=TINY_WINDOW_SECONDS,
        max_output_seq_len=48, time_hz=100, sampling_rate=TINY_SAMPLE_RATE, lora_rank=4,
        lora_alpha=8, weights_format="merged", backbone_config=backbone_dict,
        tokenizer_fingerprint=tokenizer.vocab_fingerprint,
    )
    model = SheetSage2Model(config).eval()

    # Seeded weights with enough spread that greedy decoding is not a string of near-ties: the
    # native port has to reproduce every argmax exactly, and the committed margins show the
    # comparison is meaningful.
    generator = torch.Generator().manual_seed(TINY_SEED)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.endswith("layer_norm.weight") or name.endswith("layernorm_embedding.weight") \
                    or ".pointwise_block.0.weight" in name or ".resampling_layer.0.weight" in name \
                    or name.endswith("conv_block.4.1.weight"):
                value.copy_(1.0 + 0.2 * torch.randn(value.shape, generator=generator))
            elif value.ndim == 1 or name.endswith((".bias",)):
                value.copy_(0.2 * torch.randn(value.shape, generator=generator))
            else:
                fan_in = value[0].numel() if value.ndim > 1 else value.numel()
                value.copy_(torch.randn(value.shape, generator=generator) * (1.5 / fan_in ** 0.5))
        frontend = model.encoder.feature_extractor
        frontend.mel_mean.copy_(-20.0 + 5.0 * torch.randn(frontend.mel_mean.shape, generator=generator))
        frontend.mel_std.copy_(10.0 + 2.0 * torch.rand(frontend.mel_std.shape, generator=generator))
        model.layer_weight.copy_(torch.randn(model.layer_weight.shape, generator=generator))

    # Split the merged encoder into a "parent" checkpoint plus rank-r adapters, exactly the layout
    # the real snapshots have (MERT parent + SheetSage2 adapters), and merge them the upstream way.
    width, rank = backbone.hidden_size, config.lora_rank
    scale = config.lora_alpha / config.lora_rank
    adapters = {}
    parent_state = {k: v.detach().clone() for k, v in model.encoder.state_dict().items()}
    with torch.no_grad():
        for index, layer in enumerate(model.encoder.layers):
            for projection in ("query_proj", "key_proj", "value_proj", "out_proj"):
                a = torch.randn((rank, width), generator=generator) * 0.3
                b = torch.randn((width, rank), generator=generator) * 0.3
                prefix = f"adapter.layers.{index}.attn.{projection}"
                adapters[f"{prefix}.lora_A.weight"] = a
                adapters[f"{prefix}.lora_B.weight"] = b
                getattr(layer.attn, projection).weight.add_((b @ a) * scale)
    head_state = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if not k.startswith("encoder.") and k not in ("decoder.embed_tokens.weight",
                                                                 "output_projection.weight")}
    head_state.update(adapters)

    rng = np.random.default_rng(TINY_SEED)
    samples = int(round(TINY_SIGNAL_SECONDS * TINY_SAMPLE_RATE))
    t = np.arange(samples) / TINY_SAMPLE_RATE
    signal = (0.4 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 330.0 * t)
              + 0.05 * rng.standard_normal(samples)).astype(np.float32)
    waveform = torch.from_numpy(signal)[None]

    with torch.inference_mode():
        padded, _ = model._prepare_audio(waveform)
        mel = model.encoder.feature_extractor(padded)
        features = model.get_audio_features(waveform, output_hidden_states=True)
        margins, raw_first = [], None

        def capture(position, ids, logits, masked):
            nonlocal raw_first
            if raw_first is None:
                raw_first = logits[0].clone()
            top = torch.topk(masked[0], 2).values
            margins.append(float(top[0] - top[1]) if torch.isfinite(top[1]) else float("inf"))

        tokens = generation.constrained_prompt_generate(
            model, waveform, generation.FULL_TASK_PROMPTS, config.max_output_seq_len,
            autocast_dtype=None, stop_time_seconds=TINY_SIGNAL_SECONDS, memory=None,
            step_callback=capture,
        )
    tensors = {
        "input.waveform": waveform[0].contiguous(),
        "output.mel": mel[0].contiguous(),
        "output.input_hidden": features.input_hidden_state[0].contiguous(),
        "output.mixed": features.mixed_hidden_state[0].contiguous(),
        "output.memory": features.encoder_last_hidden_state[0].contiguous(),
        "output.first_logits": raw_first.contiguous(),
    }
    for index, state in enumerate(features.backbone_hidden_states):
        tensors[f"output.block.{index}"] = state[0].contiguous()
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in parent_state.items()}, str(out / "mert_parent.safetensors"))
    save_file({k: v.contiguous() for k, v in head_state.items()}, str(out / "sheetsage2_head.safetensors"))
    save_file(tensors, str(out / "reference.safetensors"))
    finite = [m for m in margins if m != float("inf")]
    write_json(out / "reference.json", {
        "generator": "scripts/reference/sheetsage2/native_parity.py tiny",
        "upstream_code": f"{SHEETSAGE2_REPO}@{SHEETSAGE2_HEAD_REVISION}",
        "seed": TINY_SEED,
        # The files are written in the pinned *adapter* layout (MERT parent + LoRA adapters), which
        # is what the native loader consumes; the in-memory model above was built merged only to
        # run upstream's forward pass.
        "sheetsage2_config": dict(
            {k: v for k, v in config.to_dict().items()
             if k not in ("transformers_version", "_name_or_path")},
            weights_format="adapter"),
        "tokenizer": {"n_tokens": tokenizer.n_tokens, "fingerprint": tokenizer.vocab_fingerprint,
                      "audio_length_seconds": TINY_WINDOW_SECONDS, "time_hz": 100},
        "signal_seconds": TINY_SIGNAL_SECONDS,
        "stop_time_seconds": TINY_SIGNAL_SECONDS,
        "max_sequence_length": config.max_output_seq_len,
        "tokens": [int(x) for x in tokens.tolist()],
        "tokens_described": [tokenizer.describe(int(x)) for x in tokens.tolist()],
        "greedy_min_margin": min(finite) if finite else None,
        "greedy_margins": margins,
    })
    print(f"tiny fixture: {len(tokens)} tokens, min greedy margin {min(finite):.4f}")


# ---------------------------------------------------------------------------------------- tables

def tables(out: Path) -> None:
    import mir_eval.chord
    import numpy as np

    modules, _ = upstream_package(SHEETSAGE2_HEAD_REVISION)
    tokenization = modules["tokenization_sheetsage2"]
    rows = {}
    for label in tokenization.FULL_CHORD_VOCABULARY:
        if label == "N":
            continue
        root, bitmap, bass = mir_eval.chord.encode(label, reduce_extended_chords=True)
        upper = [48 + root + int(i) for i in np.flatnonzero(bitmap > 0)]
        rows[label] = sorted(set([36 + (root + bass) % 12] + upper))
    write_json(out / "chord_pitches.json", {
        "generator": "scripts/reference/sheetsage2/native_parity.py tables",
        "source": "midi_sheetsage2.chord_pitches via mir_eval.chord.encode(reduce_extended_chords=True)",
        "pitches": rows,
    })
    print(f"chord table: {len(rows)} labels")


def _ranges(mask) -> list:
    """Compress a boolean vocabulary mask into half-open ``[start, end)`` runs."""
    runs, start = [], None
    for index, allowed in enumerate(mask):
        if allowed and start is None:
            start = index
        elif not allowed and start is not None:
            runs.append([start, index])
            start = None
    if start is not None:
        runs.append([start, len(mask)])
    return runs


def grammar(out: Path) -> None:
    """Upstream ``PromptGrammarState`` masks along committed real token streams.

    For each committed ``tokens.txt`` oracle it replays upstream's grammar state exactly as
    ``constrained_prompt_generate_batch`` does (prefix after ``<|out|>`` first, then one update per
    generated token) and records the allowed set *before* every generated token, as runs.
    """
    import torch

    modules, _ = upstream_package(SHEETSAGE2_HEAD_REVISION)
    tokenizer = modules["tokenization_sheetsage2"].SheetSage2Tokenizer(300.0, 100, "v1")
    generation = modules["generation_sheetsage2"]
    cases = {}
    for case in ("synth_full", "real_full"):
        lines = (ARTIFACTS / case / "tokens.txt").read_text(encoding="utf-8").splitlines()
        tokens = [int(line.split("\t")[1]) for line in lines if line and not line.startswith("#")]
        out_index = tokens.index(tokenizer.out_token)
        state = generation.PromptGrammarState(tokenizer)
        steps = []
        for token in tokens[out_index + 1:]:
            mask = state.allowed(torch.device("cpu")).tolist()
            steps.append({"token": token, "allowed": _ranges(mask)})
            if state.update(token):
                break
        cases[case] = {"prefix": tokens[:out_index + 1], "steps": steps}
    # Compact (one line): ~780 masks as runs; indentation would triple the committed size.
    (out / "grammar_masks.json").write_text(json.dumps({
        "generator": "scripts/reference/sheetsage2/native_parity.py grammar",
        "upstream_code": f"{SHEETSAGE2_REPO} generation_sheetsage2.PromptGrammarState",
        "cases": cases,
    }, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print({k: len(v["steps"]) for k, v in cases.items()})


# ------------------------------------------------------------------------------------------ real

def _load_model(revision: str):
    import torch
    from transformers import AutoModel

    os.environ["HF_HUB_OFFLINE"] = "1"
    model = AutoModel.from_pretrained(SHEETSAGE2_REPO, revision=revision, code_revision=revision,
                                      trust_remote_code=True, local_files_only=True).eval().to("cpu")
    if {p.device.type for p in model.parameters()} != {"cpu"}:
        raise SystemExit("CPU-only lane violated")
    if next(model.parameters()).dtype != torch.float32:
        raise SystemExit("reference must run in float32")
    return model


def real(work: Path, fixtures: Path, threads: int) -> None:
    import torch
    from safetensors.torch import save_file

    torch.set_num_threads(threads)
    arrays = load_arrays(fixtures)
    model = _load_model(SHEETSAGE2_REVISION)
    out = work / "native_parity"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in ("synth", "nav_ssb"):
        waveform = torch.from_numpy(arrays[name].copy())[None]
        started = time.monotonic()
        with torch.inference_mode():
            padded, _ = model._prepare_audio(waveform)
            mel = model.encoder.feature_extractor(padded)
            features = model.get_audio_features(waveform, output_hidden_states=True)
            prefix = model.tokenizer.prompt_prefix(
                modules_full_prompts := ("timestamp", "downbeat_meter", "structure", "key",
                                         "chord_full", "melody_full"))
            logits, _ = model.decode(features.encoder_last_hidden_state,
                                     torch.tensor([prefix], dtype=torch.long), use_cache=True)
        tensors = {
            "mel": mel[0].contiguous(),
            "input_hidden": features.input_hidden_state[0].contiguous(),
            "mixed": features.mixed_hidden_state[0].contiguous(),
            "memory": features.encoder_last_hidden_state[0].contiguous(),
            "prefix_logits": logits[0].contiguous(),
        }
        for index, state in enumerate(features.backbone_hidden_states):
            tensors[f"block.{index}"] = state[0].contiguous()
        save_file(tensors, str(out / f"{name}.safetensors"))
        summary[name] = {"seconds": round(time.monotonic() - started, 2),
                         "input_sha256": sha256_bytes(arrays[name].tobytes()),
                         "prompts": list(modules_full_prompts), "prefix": prefix}
        print(f"{name}: dumped in {summary[name]['seconds']} s")
    write_json(out / "summary.json", summary)


# ------------------------------------------------------------------------------------ transcribe

LONG_ORDER = ("nav_ssb", "synth", "nav_ssb", "synth", "nav_ssb", "synth", "nav_ssb")
ORACLE_SUFFIXES = {".abc", ".lab", ".txt", ".mid", ".json", ".tsv"}


def _transcribe_case(name: str, array, work: Path, threads: int) -> None:
    import numpy as np
    import torch

    torch.set_num_threads(threads)
    model = _load_model(SHEETSAGE2_HEAD_REVISION)
    case = work / "native_cases" / name
    if case.exists():
        shutil.rmtree(case)
    (case / "output").mkdir(parents=True)
    raw = case / f"{name}.model_input.24k_mono_f32le.raw"
    array.astype("<f4").tofile(str(raw))
    started = time.monotonic()
    result = model.transcribe(array.astype(np.float32), sampling_rate=24000,
                              output_dir=str(case / "output"), dtype="fp32", preset="default")
    seconds = round(time.monotonic() - started, 2)
    record = {
        "case": name,
        "generator": f"scripts/reference/sheetsage2/native_parity.py {name}",
        "upstream_code": f"{SHEETSAGE2_REPO}@{SHEETSAGE2_HEAD_REVISION}",
        "weights": f"{SHEETSAGE2_REPO}@{SHEETSAGE2_REVISION} (identical LFS object at both revisions)",
        "input": {"sha256": sha256_file(raw), "samples": int(len(array)), "sample_rate": 24000,
                  "file": raw.name},
        "transcribe_seconds": seconds,
        "result": {k: result.get(k) for k in ("duration_seconds", "warnings", "melody_notes",
                                               "vocal_notes", "instrumental_notes", "abc_measures",
                                               "abc_error", "diagnostics")},
        "windows": [{k: w[k] for k in ("start", "end", "accept_start", "accept_end", "prefix_end",
                                       "generation_stop", "prefix_tokens", "tokens", "events",
                                       "accepted_events")} for w in result["windows"]],
    }
    if name == "long_multiwindow":
        record["input"]["recipe"] = {"order": list(LONG_ORDER),
                                     "note": "concatenation of the digest-pinned sc-23003 arrays"}
    write_json(case / "case.json", record)
    target = ARTIFACTS / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for path in sorted((case / "output").rglob("*")):
        if path.is_file() and path.suffix in ORACLE_SUFFIXES and path.stat().st_size <= 400_000:
            destination = target / path.relative_to(case / "output")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    shutil.copy2(case / "case.json", target / "case.json")
    print(f"{name}: {len(result['windows'])} window(s), {seconds} s")


def long_case(work: Path, fixtures: Path, threads: int) -> None:
    import numpy as np

    arrays = load_arrays(fixtures)
    array = np.concatenate([arrays[name] for name in LONG_ORDER])
    _transcribe_case("long_multiwindow", array, work, threads)


def silence_case(work: Path, threads: int) -> None:
    import numpy as np

    _transcribe_case("silence", np.zeros(12 * 24000, dtype=np.float32), work, threads)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("tiny", "tables", "grammar", "real", "long", "silence"))
    parser.add_argument("--work", type=Path, help="scratch directory OUTSIDE the repository")
    parser.add_argument("--fixtures", type=Path,
                        help="directory holding the sc-23003 model-input arrays (default WORK/fixtures)")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.command == "tiny":
        tiny(CRATE_TESTDATA / "tiny")
        return 0
    if args.command == "tables":
        tables(CRATE_TESTDATA)
        return 0
    if args.command == "grammar":
        grammar(CRATE_TESTDATA)
        return 0
    if args.work is None:
        parser.error("--work is required for real-weight commands")
    work = args.work.resolve()
    if work == REPO or REPO in work.parents:
        parser.error("--work must be outside the repository")
    fixtures = (args.fixtures or work / "fixtures").resolve()
    if args.command == "real":
        real(work, fixtures, args.threads)
    elif args.command == "long":
        long_case(work, fixtures, args.threads)
    else:
        silence_case(work, args.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
