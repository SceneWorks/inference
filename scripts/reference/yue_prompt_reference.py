#!/usr/bin/env python3
"""Regenerate the YuE tokenizer / prompt-builder golden fixture (sc-19376, epic sc-19373).

`crates/audio/candle-audio-yue/src/tokenizer.rs` ports YuE's mm SentencePiece tokenizer
(`_MMSentencePieceTokenizer.tokenize`) and its stage-1 prompt builder (`get_prompt_texts`,
`get_first_segment_prompt`, `get_segment_prompt`, and the "smart context" `shorten_input`). This
script runs the **reference Python code itself** and writes what it produces to
`crates/audio/candle-audio-yue/tests/fixtures/yue_prompt_reference.json`:

* ``tokenize`` — text → ids from the upstream `_MMSentencePieceTokenizer` over the upstream
  `tokenizer.model` (English, Chinese, Japanese, Korean, byte-fallback, whitespace runs, embedded
  special-token strings, …).
* ``prompts`` — per case, the first-segment block and every later-segment block exactly as
  `Stage1Pipeline.get_first_segment_prompt` / `get_segment_prompt` build them, including both ICL
  modes (single-track `use_audio_prompt` and `use_dual_tracks_prompt`). Inputs go through the
  reference CLI's file read (`open(...).read().strip()`), so newline translation and stripping are
  part of the golden.
* ``shorten`` — `Stage1Pipeline.shorten_input` over real prompt blocks with synthetic audio: the
  block-drop case (one and several blocks) and the tail-truncation fallback.

Where the code comes from:

* YuE-v1 (`multimodal-art-projection/YuE` @ `YuE-v1`, commit
  ``6d4f0b1f8ce6a55fb2392e959394c46e07ee334d``) — `mmtokenizer.py`, `codecmanipulator.py`, the
  `tokenizer.model`, and `infer.py`'s `split_lyrics`, from the shared reference environment
  (``YUE_REF_DIR``, default ``~/.cache/sceneworks-yue-ref``; see its README). Each file's SHA-256
  is asserted.
* The `Stage1Pipeline` prompt methods come from `sgsdxzy/YuE-exllamav2` (the only upstream with the
  smart-context shortening), fetched at a pinned commit with its SHA-256 asserted. The class body is
  executed as-is; only its model/audio I/O (`load_audio_mono`, `encode_audio`, the codec load) is
  replaced by stubs that hand back fixed synthetic codec codes — the ICL *encoder* is a separate
  story (sc-19379); this fixture pins how its output is *wrapped*. `split_lyrics` from YuE-v1's
  `infer.py` is cross-checked against the exllamav2 copy on every case.

Nothing here loads model weights; it runs on CPU in a few seconds::

    ~/.cache/sceneworks-yue-ref/venv/bin/python scripts/reference/yue_prompt_reference.py
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT / "crates/audio/candle-audio-yue/tests/fixtures/yue_prompt_reference.json"
)

YUE_COMMIT = "6d4f0b1f8ce6a55fb2392e959394c46e07ee334d"
YUE_FILES = {
    "mmtokenizer.py": "1473edc703423b06ec404847f63fac4ceff0775c13d674e0d0b4d6e741c0cd05",
    "codecmanipulator.py": "0d0fcb26b38807ddfd7a012973b6a457258155bc7e13197320004d96fc99370e",
    "infer.py": "9e2fa795b1556d73e9a8c7a1c6c32bcb8b053d70ba7cf67b2c5ca9a4f510d47f",
    "mm_tokenizer_v0.2_hf/tokenizer.model": (
        "ee5c7cbf32da93989f14d9ba635e3e1d1ab2cc88a92908a5ed0f149375f6ee49"
    ),
}
EXL_COMMIT = "a644036251c96613e0c4bb192a9309bfc046dafd"
EXL_URL = (
    "https://raw.githubusercontent.com/sgsdxzy/YuE-exllamav2/"
    f"{EXL_COMMIT}/src/yue/infer_stage1.py"
)
EXL_SHA256 = "17d651039e600263d1e0b0f74c827ddad34eb9c5f18030555b7bc2d65cf08d70"

# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------

TOKENIZE_CASES = [
    "",
    "hello",
    "Hello, world!",
    "  leading and trailing  ",
    "multiple   spaces\tand\ttabs",
    "line one\nline two\n\nline four\n",
    "Generate music from the given lyrics segment by segment.\n[Genre] inspiring female pop",
    "[verse]\nStaring at the sunset, colors paint the sky\n",
    "[start_of_segment]",
    "[end_of_segment]",
    "[start_of_reference]",
    "[end_of_reference]",
    "<SOA><xcodec>",
    "text<SOA>more<EOA>tail",
    "<s>bos</s> eos <PAD> <unk> <CLS><SEP><MASK><EOD>",
    "<stage_1><stage_2><s_local><e_local><s_global><e_global><SOI><EOI><SOV><EOV>",
    "<<SOA>>",
    "<SOA",
    "我们一起唱歌，在夜空下闪耀。",
    "[chorus]\n月亮代表我的心\n你问我爱你有多深",
    "夜空に輝く星のように、君を想う。",
    "カタカナとひらがなとー長音",
    "사랑해요, 오늘 밤 별빛 아래서.",
    "[verse]\n하늘을 날아 꿈을 꾸네\n君と一緒に",
    "mixed English 中文 日本語 한국어 123",
    "emoji 🎵🎶 and byte fallback \u0007\u0001",
    "full width ＡＢＣ１２３ and ligature ﬁ",
    "combining é vs é, Thai สวัสดี",
    "carriage\r\nreturn\rlone",
    "non-breaking space and ideographic　space",
    "12345 67.89 -- ... !!! ???",
]

EN_GENRES = "inspiring female uplifting pop airy vocal electronic bright vocal vocal"
EN_LYRICS = """[verse]
Staring at the sunset, colors paint the sky
Thoughts of you keep swirling, can't deny
I know I let you down, I made mistakes
But I'm here to mend the heart I didn't break

[chorus]
Every road you take, I'll be one step behind
Every dream you chase, I'm reaching for the light
You can't fight this feeling now
I won't back down

[verse]
Moonlight on the water, whispers in the night
Holding on to memories, burning bright

[chorus]
Every road you take, I'll be one step behind

[outro]
I won't back down
"""

ZH_GENRES = "Pop, Mandarin, female vocal, piano, emotional"
ZH_LYRICS = """[verse]
月光洒在窗台上
思念像潮水一样
你的笑容在心上
轻轻地把我照亮

[chorus]
我愿意为你守候
直到天长地久
不管风雨多少
我都不会放手
"""

JPKR_GENRES = "J-pop K-pop female vocal synth upbeat"
JPKR_LYRICS = """[verse]
夜空に輝く星のように
君の笑顔が僕を照らす
[chorus]
사랑해요 오늘 밤
별빛 아래서 춤을 춰요
[bridge]
ずっと一緒にいたい
영원히 함께해요
"""

# Edge cases for the section split: text before the first label is dropped; a label with a space
# does not match `\\w+` (so `[verse 1]` is not a section and truncates the previous body); a
# Unicode label matches; empty bodies survive; `[start_of_segment]` inside a body is removed from
# the segment text (and, being `\\w+`, is itself a section label).
EDGE_GENRES = "  rock  guitar\n"
EDGE_LYRICS = (
    "intro text before any label\n"
    "[verse]\r\n  first body with CRLF  \r\n\r\n"
    "[verse 1]\nnot a section\n"
    "[副歌]\n副歌的歌词\n"
    "[bridge]\n"
    "[chorus_2]\nsecond chorus [end_of_segment] inline\n"
    "[outro]  \t tail  "
)

# Synthetic xcodec codebook-0 codes standing in for the ICL encoder's output (sc-19379 owns the
# encoder; this fixture pins how its codes are windowed into ids and wrapped).
SINGLE_CODES = [(i * 37 + 11) % 1024 for i in range(160)]  # 3.2 s at 50 fps
VOCAL_CODES = [(i * 53 + 7) % 1024 for i in range(120)]
INST_CODES = [(i * 29 + 101) % 1024 for i in range(120)]

PROMPT_CASES = [
    {"name": "en_cot", "genres": EN_GENRES, "lyrics": EN_LYRICS, "icl": None},
    {"name": "zh_cot", "genres": ZH_GENRES, "lyrics": ZH_LYRICS, "icl": None},
    {"name": "jp_kr_cot", "genres": JPKR_GENRES, "lyrics": JPKR_LYRICS, "icl": None},
    {"name": "edge_sections", "genres": EDGE_GENRES, "lyrics": EDGE_LYRICS, "icl": None},
    {
        "name": "en_icl_single",
        "genres": EN_GENRES,
        "lyrics": EN_LYRICS,
        "icl": {"mode": "single", "start": 0.5, "end": 2.5},
    },
    {
        "name": "zh_icl_dual",
        "genres": ZH_GENRES,
        "lyrics": ZH_LYRICS,
        "icl": {"mode": "dual", "start": 0.4, "end": 1.9},
    },
    {
        "name": "jp_kr_icl_single",
        "genres": JPKR_GENRES,
        "lyrics": JPKR_LYRICS,
        "icl": {"mode": "single", "start": 0.0, "end": 30.0},
    },
]

# --------------------------------------------------------------------------------------------
# Reference loading
# --------------------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference(ref_dir: Path):
    inf = ref_dir / "yue" / "inference"
    for rel, want in YUE_FILES.items():
        got = sha256(inf / rel)
        if got != want:
            sys.exit(f"{inf / rel}: sha256 {got} != pinned {want} (YuE-v1 @ {YUE_COMMIT})")
    os.chdir(inf)
    sys.path[:0] = [str(inf)]
    from codecmanipulator import CodecManipulator  # noqa: E402
    from mmtokenizer import _MMSentencePieceTokenizer  # noqa: E402

    infer_src = (inf / "infer.py").read_text(encoding="utf-8")
    split_lyrics_v1 = extract(infer_src, "split_lyrics", {"re": re})
    tok = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")
    return tok, CodecManipulator, split_lyrics_v1


def extract(source: str, name: str, namespace: dict):
    """Execute one top-level `def`/`class` from `source` in `namespace` and return it."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            code = compile(ast.Module(body=[node], type_ignores=[]), f"<{name}>", "exec")
            exec(code, namespace)
            return namespace[name]
    sys.exit(f"`{name}` not found in the reference source")


def fetch_exl(cache: Path | None) -> str:
    if cache is not None and cache.exists():
        data = cache.read_bytes()
    else:
        with urllib.request.urlopen(EXL_URL, timeout=60) as r:
            data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if got != EXL_SHA256:
        sys.exit(f"{EXL_URL}: sha256 {got} != pinned {EXL_SHA256}")
    return data.decode("utf-8")


class RecordingTokenizer:
    """Delegates to the upstream tokenizer and records every `tokenize(text)` call, so the fixture
    carries the exact text → ids table the prompt builder consumed (the Rust assembly test replays
    it without the 5 MB tokenizer; the real-tokenizer test checks the table itself)."""

    def __init__(self, tok):
        self._tok = tok
        self.log: dict[str, list[int]] = {}

    def tokenize(self, text):
        ids = self._tok.tokenize(text)
        self.log[text] = [int(i) for i in ids]
        return ids

    def __getattr__(self, name):
        return getattr(self._tok, name)


def build_pipeline(exl_src: str, tok, CodecManipulator, audio_codes: dict):
    import numpy as np
    import torch
    from einops import rearrange

    def load_audio_mono(path, sampling_rate=16000):  # stub: the "audio" is its path key
        return path

    def encode_audio(codec_model, audio_prompt, device, target_bw=0.5):
        # Stub for the xcodec encoder: the reference transposes to (batch, n_q, T) int16.
        return np.array([[audio_codes[audio_prompt]]], dtype=np.int16)

    ns = {
        "os": os,
        "re": re,
        "np": np,
        "torch": torch,
        "rearrange": rearrange,
        "CodecManipulator": CodecManipulator,
        "_MMSentencePieceTokenizer": type(tok),
        "load_audio_mono": load_audio_mono,
        "encode_audio": encode_audio,
    }
    cls = extract(exl_src, "Stage1Pipeline", ns)
    p = cls.__new__(cls)
    # `Stage1Pipeline.__init__` minus its path-relative tokenizer load and codec config.
    p.device = "cpu"
    p.codec_tool = CodecManipulator("xcodec", 0, 1)
    p.codec_model = object()  # `load_codec_model` returns early; `encode_audio` is stubbed
    p.mmtokenizer = RecordingTokenizer(tok)
    p.start_of_segment = tok.tokenize("[start_of_segment]")
    p.end_of_segment = tok.tokenize("[end_of_segment]")
    return p


# --------------------------------------------------------------------------------------------
# Fixture assembly
# --------------------------------------------------------------------------------------------


def read_like_cli(text: str) -> str:
    """The reference CLI reads genres/lyrics from text files: `open(path).read().strip()`."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        with open(path, encoding="utf-8") as f:  # text mode: universal newlines, as the CLI
            return f.read().strip()
    finally:
        os.unlink(path)


def prompt_case(p, split_lyrics_v1, case: dict) -> dict:
    import numpy as np

    p.mmtokenizer.log = {}
    genres = read_like_cli(case["genres"])
    lyrics_text = read_like_cli(case["lyrics"])
    lyrics, prompt_texts = p.get_prompt_texts(genres, lyrics_text)
    if split_lyrics_v1(lyrics_text) != lyrics:
        sys.exit(f"{case['name']}: YuE-v1 split_lyrics disagrees with the exllamav2 copy")
    labels = [re.match(r"\[(\w+)\]", seg).group(1) for seg in lyrics]

    icl = case["icl"]
    icl_ids = None
    kwargs = dict(
        use_dual_tracks_prompt=False,
        vocal_track_prompt_path="",
        instrumental_track_prompt_path="",
        use_audio_prompt=False,
        audio_prompt_path="",
        prompt_start_time=0,
        prompt_end_time=30,
    )
    if icl is not None:
        kwargs["prompt_start_time"] = icl["start"]
        kwargs["prompt_end_time"] = icl["end"]
        tool = p.codec_tool
        if icl["mode"] == "single":
            kwargs.update(use_audio_prompt=True, audio_prompt_path="mix")
            ids = tool.npy2ids(np.array([SINGLE_CODES]))
            icl_ids = ids[int(icl["start"] * 50) : int(icl["end"] * 50)]
        else:
            kwargs.update(
                use_dual_tracks_prompt=True,
                vocal_track_prompt_path="vocals",
                instrumental_track_prompt_path="instrumental",
            )
            v = tool.npy2ids(np.array([VOCAL_CODES]))
            i = tool.npy2ids(np.array([INST_CODES]))
            inter = [t for pair in zip(v, i) for t in pair]
            icl_ids = inter[int(icl["start"] * 50 * 2) : int(icl["end"] * 50 * 2)]

    blocks = [p.get_first_segment_prompt(prompt_texts[1], prompt_texts[0], **kwargs)]
    blocks += [p.get_segment_prompt(prompt_texts[i + 1]) for i in range(1, len(lyrics))]
    return {
        "name": case["name"],
        "genres": case["genres"],
        "lyrics": case["lyrics"],
        "icl_ids": icl_ids,
        "labels": labels,
        "segments": [[int(t) for t in b] for b in blocks],
        "encoded": p.mmtokenizer.log,
    }


def shorten_cases(p, prompts: list[dict]) -> list[dict]:
    import torch

    en = next(c for c in prompts if c["name"] == "en_cot")["segments"]
    audio = lambda n, k: [45334 + (j * 31 + k) % 1024 for j in range(n)]  # noqa: E731
    eoa = p.mmtokenizer.eoa

    # head+segment 0, audio, then segments 1.. each followed by audio (the last is the one being
    # prompted, so it carries no audio yet) — the sequence `shorten_input` sees before segment n.
    def sequence(n_segments: int, audio_len: int) -> list[int]:
        seq: list[int] = []
        for k, block in enumerate(en[:n_segments]):
            seq += block
            if k + 1 < n_segments:
                seq += audio(audio_len, k) + [eoa]
        return seq

    specs = [
        ("fits_untouched", 3, 40, None),
        ("drop_one_block", 3, 40, -30),
        ("drop_two_blocks", 5, 40, -130),
        ("fallback_two_markers", 2, 40, -10),
        ("fallback_after_drops", 4, 40, None),
    ]
    out = []
    for name, n, alen, delta in specs:
        seq = sequence(n, alen)
        if name == "fits_untouched":
            max_context = len(seq)
        elif name == "fallback_after_drops":
            # Dropping blocks until two markers remain leaves `head + last two blocks` (length
            # `kept`), which still does not fit; the tail truncation then reaches back into the
            # head, so the output differs from a plain tail cut of the original sequence.
            sos = p.start_of_segment
            marks = [i for i in range(len(seq)) if seq[i : i + len(sos)] == sos]
            kept = marks[0] + len(seq) - marks[-2]
            max_context = kept - 5
        else:
            max_context = len(seq) + delta
        got = p.shorten_input(torch.tensor([seq]), max_context)[0].tolist()
        out.append({"name": name, "max_context": max_context, "input": seq, "output": got})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--ref-dir",
        type=Path,
        default=Path(os.environ.get("YUE_REF_DIR", "~/.cache/sceneworks-yue-ref")).expanduser(),
    )
    ap.add_argument(
        "--exl-source",
        type=Path,
        default=None,
        help="a local copy of the pinned exllamav2 infer_stage1.py (sha256-checked); "
        "fetched from GitHub when omitted",
    )
    args = ap.parse_args()
    output = args.output.resolve()
    exl_src = fetch_exl(args.exl_source)
    tok, CodecManipulator, split_lyrics_v1 = load_reference(args.ref_dir)
    audio_codes = {"mix": SINGLE_CODES, "vocals": VOCAL_CODES, "instrumental": INST_CODES}
    p = build_pipeline(exl_src, tok, CodecManipulator, audio_codes)

    tokenize = [{"text": t, "ids": [int(i) for i in tok.tokenize(t)]} for t in TOKENIZE_CASES]
    prompts = [prompt_case(p, split_lyrics_v1, c) for c in PROMPT_CASES]
    fixture = {
        "provenance": {
            "producer": "scripts/reference/yue_prompt_reference.py",
            "yue_commit": YUE_COMMIT,
            "yue_files_sha256": YUE_FILES,
            "exllamav2_commit": EXL_COMMIT,
            "exllamav2_infer_stage1_sha256": EXL_SHA256,
        },
        "markers": {
            "start_of_segment": p.start_of_segment,
            "end_of_segment": p.end_of_segment,
            "start_of_reference": tok.tokenize("[start_of_reference]"),
            "end_of_reference": tok.tokenize("[end_of_reference]"),
            "soa": tok.soa,
            "eoa": tok.eoa,
            "xcodec_sep": p.codec_tool.sep_ids,
        },
        "tokenize": tokenize,
        "prompts": prompts,
        "shorten": shorten_cases(p, prompts),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, ensure_ascii=True, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
