#!/usr/bin/env python3
"""Regenerate the YuE2 text-tokenizer / request-protocol / symbolic-plan fixtures (sc-22990).

Everything here is evaluated by the **pinned upstream** (``yue2`` installed from
``multimodal-art-projection/YuE`` @ 92a73cc7652fcc1f937855e4b765e0a0edd7ff2e by
``setup_reference_env.sh``): ``yue2.tokenization_yue2.YuE2TextTokenizer``, ``yue2.protocol``
(``Sampling``, ``GenerationConfig``, ``SongRequest``, ``token_prefixes``, ``negative_prefix``,
``chunk_ranges``), ``yue2.nar.song_chunks``' context check, ``yue2.sampling.generate_tokens``'
budget checks, and ``yue2.pipeline.SymbolicPlan.save``. Nothing is re-implemented on this side
except the synthetic rank table below.

Outputs (``crates/audio/candle-audio-yue2/tests/fixtures/protocol/``):

* ``synthetic.tiktoken`` — a small byte-level BPE rank table **trained here** on this script's
  own corpus (256 single bytes + learned merges). ``qwen.tiktoken`` itself is never committed:
  the crate's licence policy gates redistribution of it (``license.rs``), and the repository is
  public. The synthetic table exercises the identical code path — upstream's own
  ``YuE2TextTokenizer`` is constructed on it (padded to the 151643 ordinary tokens the class
  requires with unreachable ``\\xff``-prefixed fillers), so the pre-tokenizer pattern, NFC,
  rank-ordered merging, the special-token table and every prefix are upstream's.
* ``tokenizer_synthetic.json`` — encode / decode / prefix cases on the synthetic table (CI).
* ``tokenizer_qwen.json`` — the same cases on the pinned ``qwen.tiktoken`` (only with
  ``--hub``); the Rust side checks it against the real file in an env-gated test.
* ``protocol_cases.json`` — accept/reject (and resolved values) for ``Sampling`` overrides,
  ``GenerationConfig.from_dict``, ``SongRequest``, ``chunk_ranges`` and the generation budget.
* ``plans/<name>/`` + ``plans.json`` — plans saved by upstream ``SymbolicPlan.save`` (synthetic
  tokenizer), which the Rust side must restore to the exact token IDs.

Run with the reference interpreter (no weights are loaded; peak RSS < 1 GB)::

    YUE2_HF_HUB=/path/to/huggingface/hub \\
        ~/.cache/sceneworks-yue2-ref/venv/bin/python scripts/reference/yue2/protocol_fixtures.py --hub
"""

from __future__ import annotations

import argparse
import base64
import collections
import dataclasses
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import unicodedata
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import regex  # noqa: E402
import tiktoken  # noqa: E402
import torch  # noqa: E402

import yue2  # noqa: E402
from yue2 import protocol  # noqa: E402
from yue2.nar import song_chunks  # noqa: E402
from yue2.pipeline import SymbolicPlan  # noqa: E402
from yue2.protocol import (  # noqa: E402
    GenerationConfig,
    Sampling,
    SongRequest,
    chunk_ranges,
    negative_prefix,
    resolve_sampling,
    token_prefixes,
)
from yue2.sampling import generate_tokens  # noqa: E402
from yue2.tokenization_yue2 import YuE2TextTokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "crates/audio/candle-audio-yue2/tests/fixtures/protocol"
YUE2_COMMIT = "92a73cc7652fcc1f937855e4b765e0a0edd7ff2e"
YUE2_3B = ("m-a-p/YuE2-3B", "1a96eca688d6ae5d7f0feb88573fec89920fcd19")
QWEN_TIKTOKEN_SHA256 = "b2b1b8dfb5cc5f024bafc373121c6aba3f66f9a5a0269e243470a1de16a33186"
ORDINARY = 151643
MERGES = 600

# The upstream pre-tokenizer, copied only to TRAIN the synthetic table (which merges exist); the
# fixtures themselves are always produced by upstream's own YuE2TextTokenizer. Guarded below.
PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)

ENGLISH_STYLE = (
    "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light "
    "drums, lyrical memorable melody, unhurried phrasing, 88 BPM"
)
ENGLISH_LYRICS = (
    "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\n\n[Chorus]\n"
    "Let the day come into view\nEvery road begins with you"
)
CHINESE_STYLE = "中文, 抒情流行, 温暖的女声, 钢琴与弦乐, 72 BPM"
CHINESE_LYRICS = "[Verse]\n夜色慢慢落在窗前\n我把思念写成诗篇\n\n[Chorus]\n等风吹过那条老街\n等你回来的那一天"
ABC_SCORE = (
    "X:1\nL:1/8\nQ:1/4=88\nM:4/4\nK:C\nV:Vocal\nV:Ins\n"
    '[V:Vocal] z2 E2 G2 A2 | c4 B2 A2 | G6 z2 |\n'
    '[V:Ins] "C" C,2 E,2 G,2 E,2 | "Am" A,,2 C,2 E,2 C,2 | "F" F,,4 A,,4 |\n'
)
ABC_MELODY = "X:1\nL:1/8\nM:4/4\nK:G\nV:Vocal\n[V:Vocal] D2 G2 B2 d2 | e4 d2 B2 | A6 z2 |\n"

CORPUS = [
    *protocol.INSTRUCTIONS.values(),
    "[Tags]\n[Lyrics]\n[Verse]\n[Chorus]\n[Bridge]\n[Outro]\n",
    ENGLISH_STYLE,
    ENGLISH_LYRICS,
    "Fold the night and leave it here\nMorning has a sky to clear\nHold a little room for light",
    CHINESE_STYLE,
    CHINESE_LYRICS,
    "我们一起唱歌 在春天的早晨 听见远方的声音",
    ABC_SCORE,
    ABC_MELODY,
]

# Text cases for ordinary-token encoding. Each exercises a distinct piece of the contract.
ENCODE_CASES = [
    ("empty", ""),
    ("english_style", ENGLISH_STYLE),
    ("english_lyrics", ENGLISH_LYRICS),
    ("english_unseen", "Quartz jukebox vows: 1,024 zebras!!\n\nWhy?"),
    ("chinese_lyrics", CHINESE_LYRICS),
    ("chinese_unseen", "鳳凰于飛，翽翽其羽。"),
    ("mixed_scripts", "Love 爱 amour любовь 사랑 حب"),
    ("abc_score", ABC_SCORE),
    ("abc_melody", ABC_MELODY),
    ("contractions", "I'M sure they'LL say it's what we'd've DONE, y'all"),
    ("case_fold_long_s", "itſ fine"),
    ("digits", "12345 3.14159 ٣٤٥ 0x1F"),
    ("whitespace_runs", "  lead\t\ttab   trail   \n\n\n  x  \r\n\r\n end  "),
    ("only_spaces", "     "),
    ("newline_runs", "\n\n\n"),
    ("punct_newlines", "!!!\n\n...\r\n?"),
    ("specials_as_text", "<|endoftext|><abc>X</abc><|im_start|><extra_3>"),
    ("nfd_latin", "Café naïve résumé"),
    ("nfd_hangul", "각 한"),
    ("nfc_reorder", "q̣̇ ậ"),
    ("compat_not_folded", "ﬁ ① Ａ"),
    # Unicode-version edges of the reference stack: Python 3.12's NFC data is Unicode 15.0 and
    # tiktoken 0.12's \p{L}/\p{N} tables are Unicode 16.0.
    ("nfc_unicode16_todhri", "\U000105d2̇ x\U000105c9"),
    ("nfc_unicode16_tulu", "\U00011382\U000113c9 \U00011383"),
    ("letters_unicode16", "a\U00010d50b \U0001e5d0"),
    ("letters_unicode17", "a\U00010940b \U00011db0"),
    ("control_separators", "a\x1cb\x1d c\x1e\x1fd"),
    ("ideographic_space", "你好　世界"),
    ("emoji", "sing 🎵🎶 along 👩‍🎤"),
]


def train_synthetic_ranks() -> dict[bytes, int]:
    """A deterministic byte-level BPE over CORPUS (ties broken by the smallest pair bytes)."""
    counts: collections.Counter[bytes] = collections.Counter()
    for text in CORPUS:
        for piece in regex.findall(PATTERN, unicodedata.normalize("NFC", text)):
            counts[piece.encode("utf-8")] += 1
    words = {piece: [bytes([b]) for b in piece] for piece in counts}
    ranks = {bytes([i]): i for i in range(256)}
    for _ in range(MERGES):
        pairs: collections.Counter[tuple[bytes, bytes]] = collections.Counter()
        for piece, parts in words.items():
            for a, b in zip(parts, parts[1:]):
                pairs[(a, b)] += counts[piece]
        if not pairs:
            break
        (a, b), best = min(pairs.items(), key=lambda kv: (-kv[1], kv[0]))
        merged = a + b
        if merged not in ranks:
            ranks[merged] = len(ranks)
        for piece, parts in words.items():
            out, i = [], 0
            while i < len(parts):
                if i + 1 < len(parts) and parts[i] == a and parts[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(parts[i])
                    i += 1
            words[piece] = out
    return ranks


def tiktoken_lines(ranks: dict[bytes, int]) -> bytes:
    return b"".join(
        base64.b64encode(token) + b" " + str(rank).encode() + b"\n"
        for token, rank in sorted(ranks.items(), key=lambda kv: kv[1])
    )


def upstream_on_synthetic(ranks: dict[bytes, int], scratch: Path) -> YuE2TextTokenizer:
    """Upstream's own class on the synthetic table, padded to the 151643 it insists on.

    Fillers start with 0xFF, which never occurs in UTF-8, so no piece or merge can reach them.
    """
    padded = dict(ranks)
    for rank in range(len(ranks), ORDINARY):
        padded[b"\xff\xfe" + rank.to_bytes(3, "big")] = rank
    path = scratch / "qwen.tiktoken"
    path.write_bytes(tiktoken_lines(padded))
    return YuE2TextTokenizer(path)


def guard_pattern_copy() -> None:
    source = Path(yue2.__file__).with_name("tokenization_yue2.py").read_text(encoding="utf-8")
    if f'pattern = r"{PATTERN}"' not in source:
        sys.exit("upstream pre-tokenizer pattern changed; update PATTERN")


def err(exc: BaseException) -> dict:
    return {"error": f"{type(exc).__name__}: {exc}"}


def attempt(fn):
    try:
        return {"ok": fn()}
    except (ValueError, TypeError) as exc:
        return err(exc)


def request_cases() -> list[tuple[str, dict]]:
    base_en = {"style": ENGLISH_STYLE, "lyrics": ENGLISH_LYRICS, "seed": 831001, "id": "city_lights"}
    base_zh = {"style": CHINESE_STYLE, "lyrics": CHINESE_LYRICS, "seed": 7, "id": "ye_se"}
    cases = []
    for lang, base in (("en", base_en), ("zh", base_zh)):
        for cot in ("off", "melody", "full"):
            cases.append((f"{lang}_{cot}", {**base, "cot": cot}))
        cases.append((f"{lang}_melody_external", {**base, "cot": "melody", "abc": ABC_MELODY}))
        cases.append((f"{lang}_full_external", {**base, "cot": "full", "abc": ABC_SCORE}))
    cases.append(("en_full_cfg", {**base_en, "cot": "full", "cfg_scale": 1.5}))
    cases.append(("en_off_cfg_one", {**base_en, "cot": "off", "cfg_scale": 1.0}))
    cases.append(("nfd_request", {**base_en, "style": "Café jazz", "cot": "full"}))
    return cases


def tokenizer_fixture(tok: YuE2TextTokenizer, provenance: dict) -> dict:
    encode = [{"name": name, "text": text, "ids": tok.encode(text)} for name, text in ENCODE_CASES]
    decode = [{"name": c["name"], "ids": c["ids"], "text": tok.decode(c["ids"])} for c in encode]
    chinese = tok.encode("夜色")
    decode += [
        {"name": name, "ids": ids, "text": tok.decode(ids)}
        for name, ids in (
            # Malformed UTF-8 decoded with errors="replace": a truncated sequence, a lone
            # continuation byte, an encoded surrogate, an overlong form, a code point past
            # U+10FFFF and a truncated 4-byte emoji.
            ("invalid_utf8_bytes", [tok._enc.encode_single_token(bytes([b])) for b in (
                0xE5, 0xA4, 0x41, 0x9C, 0xED, 0xA0, 0x80, 0x42, 0xC0, 0xAF, 0xF4, 0x90, 0x80,
                0x80, 0x43, 0xF0, 0x9F, 0x8E)]),
            ("specials", [ORDINARY, ORDINARY + 1, ORDINARY + 2, ORDINARY + 3, ORDINARY + 7,
                          ORDINARY + 8, ORDINARY + 203, protocol.ABC_START, protocol.ABC_END,
                          ORDINARY + 206, ORDINARY + 207]),
            ("past_vocab_dropped", chinese + [protocol.MUSIC_START, protocol.MUSIC_END,
                                              protocol.CODEC_OFFSET, 10**6] + chinese),
        )
    ]
    prefixes = []
    for name, data in request_cases():
        request = SongRequest(**data)
        abc_ids = tok.encode(ABC_MELODY if data["cot"] == "melody" else ABC_SCORE)
        case = {
            "name": name,
            "request": request.to_dict(),
            "text": request.text(),
            "guidance": request.guidance,
            "planner": attempt(lambda: token_prefixes(request, tok)),
            "negative_without_abc": attempt(lambda: negative_prefix(request, tok)),
        }
        if request.cot != "off":
            ids = tok.encode(request.abc) if request.abc is not None else abc_ids
            case["abc_ids"] = ids
            case["positive"] = attempt(lambda: token_prefixes(request, tok, ids))
            case["negative"] = attempt(lambda: negative_prefix(request, tok, ids))
        prefixes.append(case)
    en_full = SongRequest(**dict(request_cases())["en_full"])
    rejects = [
        {"name": "abc_id_is_eod", "request": en_full.to_dict(), "abc_ids": [5, ORDINARY],
         "positive": attempt(lambda: token_prefixes(en_full, tok, [5, ORDINARY])),
         "negative": attempt(lambda: negative_prefix(en_full, tok, [5, ORDINARY]))},
        {"name": "abc_id_is_abc_end", "request": en_full.to_dict(), "abc_ids": [protocol.ABC_END],
         "positive": attempt(lambda: token_prefixes(en_full, tok, [protocol.ABC_END])),
         "negative": attempt(lambda: negative_prefix(en_full, tok, [protocol.ABC_END]))},
        {"name": "empty_generated_abc", "request": en_full.to_dict(), "abc_ids": [],
         "positive": attempt(lambda: token_prefixes(en_full, tok, [])),
         "negative": attempt(lambda: negative_prefix(en_full, tok, []))},
    ]
    return {"provenance": provenance, "encode": encode, "decode": decode, "prefixes": prefixes,
            "abc_id_cases": rejects}


FLOATS = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def from_fixture(value):
    """Fixture JSON cannot hold NaN/inf; they are written as {"float": "nan"} and decoded here."""
    if isinstance(value, dict):
        if set(value) == {"float"}:
            return FLOATS[value["float"]]
        return {k: from_fixture(v) for k, v in value.items()}
    return value


def nf(name: str) -> dict:
    return {"float": name}


def sampling_cases() -> list:
    ok = [{}, {"temperature": 0}, {"temperature": 5}, {"temperature": 5.0}, {"top_p": 1},
          {"top_p": 1e-9}, {"top_k": 1}, {"top_k": 10**12}, {"repetition_penalty": 1e-9},
          {"penalty_window": 1}, {"penalty_window": 100}, {"min_tokens": 0, "max_tokens": 1},
          {"min_tokens": 9000, "max_tokens": 9000}, {"temperature": 2, "top_p": 0.5, "top_k": 7,
          "repetition_penalty": 1, "penalty_window": 64, "min_tokens": 10, "max_tokens": 20}]
    bad = [{"temperature": -0.01}, {"temperature": 5.0001}, {"temperature": nf("nan")},
           {"temperature": nf("inf")}, {"top_p": 0}, {"top_p": 1.0001}, {"top_p": nf("nan")},
           {"top_k": 0}, {"top_k": -3}, {"top_k": 30.0}, {"top_k": True}, {"top_k": "30"},
           {"repetition_penalty": 0}, {"repetition_penalty": -1}, {"repetition_penalty": nf("-inf")},
           {"penalty_window": 0}, {"penalty_window": 101}, {"penalty_window": 50.0},
           {"min_tokens": -1}, {"min_tokens": 10, "max_tokens": 9}, {"max_tokens": 0, "min_tokens": 0},
           {"max_tokens": 4096.0}, {"min_tokens": False}, {"temperature": "0.7"},
           {"temperature": None}, {"seed": 1}, {"top_n": 3}]
    out = []
    for phase, default in (("abc", GenerationConfig().abc), ("semantic", GenerationConfig().semantic)):
        for overrides in ok + bad:
            result = attempt(lambda: dataclasses.asdict(
                resolve_sampling(from_fixture(overrides), default)))
            out.append({"phase": phase, "overrides": overrides, **result})
    return out


def generation_config_cases() -> list:
    cases = [{}, {"ode_steps": 1}, {"ode_steps": 64}, {"ode_steps": 0}, {"ode_steps": -1},
             {"ode_steps": 32.0}, {"ode_steps": True}, {"ode_method": "euler"},
             {"ode_method": "midpoint"}, {"context": 24575}, {"context": 24576},
             {"context": 24577}, {"version": "yue2-native-v1"},
             {"abc": {"temperature": 0.5}}, {"semantic": {"max_tokens": 12000, "min_tokens": 0}},
             {"abc": {"top_k": 0}}, {"abc": {"bogus": 1}}, {"unknown": 1},
             json.loads(json.dumps(GenerationConfig().to_dict()))]
    return [{"input": c, **attempt(lambda: GenerationConfig.from_dict(c).to_dict())} for c in cases]


def song_request_cases() -> list:
    base = {"style": "pop", "lyrics": "[Verse]\nla la"}
    variants = [
        {}, {"cot": "off"}, {"cot": "melody"}, {"cot": "full"}, {"cot": "FULL"}, {"cot": ""},
        {"cot": None}, {"seed": 0}, {"seed": 2**63 - 1}, {"seed": 2**63}, {"seed": -1},
        {"seed": 1.0}, {"seed": True}, {"seed": "1"}, {"id": "song"}, {"id": "a" * 180},
        {"id": "a" * 181}, {"id": ".x"}, {"id": "_x"}, {"id": "a/b"}, {"id": "a b"},
        {"id": "café"}, {"id": "A.b-c_9"}, {"id": ""}, {"id": "x\n"},
        {"cot": "off", "abc": "X:1"}, {"cot": "melody", "abc": "X:1"}, {"cot": "full", "abc": ""},
        {"abc": "  \n\t"}, {"abc": "\x1c\x1d\x1e\x1f"}, {"abc": "　 "},
        {"abc": "​"}, {"abc": 5}, {"cfg_scale": 0}, {"cfg_scale": 20}, {"cfg_scale": 20.0001},
        {"cfg_scale": -0.0}, {"cfg_scale": -1}, {"cfg_scale": 1.5}, {"cfg_scale": nf("nan")},
        {"cfg_scale": nf("inf")}, {"cfg_scale": None}, {"cot": "off", "cfg_scale": None},
        {"style": 5}, {"lyrics": None}, {"extra": 1},
    ]
    out = []
    for variant in variants:
        data = {**base, **variant}

        def build(data=data):
            r = SongRequest(**from_fixture(data))
            return {"request": r.to_dict(), "guidance": r.guidance, "text": r.text()}

        out.append({"input": data, **attempt(build)})
    for missing in ("style", "lyrics"):
        data = {k: v for k, v in base.items() if k != missing}
        out.append({"input": data, **attempt(lambda data=data: SongRequest(**data).to_dict())})
    return out


def chunk_cases() -> list:
    out = []
    for frames, prefix, context in [(1, 100, 24576), (1500, 800, 24576), (30000, 100, 24576),
                                    (24000, 1, 24576), (5, 24571, 24576), (5, 24572, 24576),
                                    (5, 24573, 24576), (5, 30000, 24576), (0, 100, 24576),
                                    (10, 4, 11), (10, 4, 10), (10, 4, 24577), (10, 4, 0)]:
        def ranges(frames=frames, prefix=prefix, context=context):
            # song_chunks owns the context-range check; chunk_ranges the acoustic-context
            # arithmetic. Where both run, the chunk lengths song_chunks cuts must agree.
            if frames >= 1:
                chunks = song_chunks([1] * prefix, [0] * frames, 0, context)
            got = chunk_ranges(frames, prefix, context)
            assert [b - a for a, b in got] == [len(c.noise) for c in chunks]
            return [list(r) for r in got]

        out.append({"frames": frames, "prefix_tokens": prefix, "context": context,
                    **attempt(ranges)})
    return out


class _Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))


def budget_cases() -> list:
    """generate_tokens' pre-prefill checks; reaching the cancellation check means they passed."""
    out = []
    model = _Probe()
    for prefix, negative, max_tokens, cfg in [(100, None, 9000, 1.0), (15576, None, 9000, 1.0),
                                              (15577, None, 9000, 1.0), (24575, None, 1, 1.0),
                                              (24576, None, 1, 1.0), (100, None, 9000, 1.5),
                                              (100, 50, 9000, 1.5), (100, 15577, 9000, 1.5),
                                              (100, 15576, 9000, 1.01), (100, 50, 9000, 1.0),
                                              (100, 24576, 1, 1.0)]:
        sampling = Sampling(max_tokens=max_tokens, min_tokens=0)
        try:
            generate_tokens(model, [1] * prefix, sampling, 0, "semantic",
                            negative=None if negative is None else [1] * negative, cfg_scale=cfg,
                            cancelled=lambda: True)
            raise AssertionError("unreachable")
        except InterruptedError:
            result = {"ok": True}
        except ValueError as exc:
            result = err(exc)
        out.append({"prefix_tokens": prefix, "negative_tokens": negative,
                    "max_tokens": max_tokens, "cfg_scale": cfg, **result})
    return out


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def plans(tok: YuE2TextTokenizer, provenance: dict) -> dict:
    root = OUT / "plans"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    records = {}
    base = {"style": ENGLISH_STYLE, "lyrics": ENGLISH_LYRICS, "seed": 831001, "id": "city_lights"}

    def save(name, plan, kind):
        plan.save(root / name)
        records[name] = {"kind": kind, "request": plan.request.to_dict(), "abc": plan.abc,
                         "abc_ids": plan.abc_ids, "prefix": plan.prefix, "timing": plan.timing,
                         "truncated": plan.truncated}

    # A planner-generated score, as pipeline.plan builds it from sampled ids.
    full = SongRequest(**base, cot="full")
    ids = tok.encode(ABC_SCORE)
    save("full_generated", SymbolicPlan(full, tok.decode(ids), ids, token_prefixes(full, tok, ids),
                                        {"seconds": 1.25, "output_tokens": len(ids) + 1}, False),
         "generated")
    # A truncated plan whose decoded text carries U+FFFD (the ABC stopped mid-character).
    zh = SongRequest(style=CHINESE_STYLE, lyrics=CHINESE_LYRICS, cot="melody", seed=7, id="ye_se")
    zh_ids = tok.encode("X:1\nK:C\n夜") [:-1]
    save("melody_generated_truncated",
         SymbolicPlan(zh, tok.decode(zh_ids), zh_ids, token_prefixes(zh, tok, zh_ids),
                      {"seconds": 0.5, "output_tokens": len(zh_ids)}, True), "generated")
    # External ABC, exactly pipeline.plan's external branch.
    ext = SongRequest(**base, cot="melody", abc=ABC_MELODY, cfg_scale=1.5)
    ext_ids = tok.encode(ext.abc)
    save("melody_external", SymbolicPlan(ext, ext.abc, ext_ids, token_prefixes(ext, tok, ext_ids),
                                         {"seconds": 0., "output_tokens": 0,
                                          "external_prefix_tokens": len(ext_ids)}), "external")
    off = SongRequest(**base, cot="off")
    save("off", SymbolicPlan(off, None, [], token_prefixes(off, tok)), "off")
    for name in records:  # upstream restores its own saves exactly
        restored = SymbolicPlan.load(root / name)
        assert restored.abc_ids == records[name]["abc_ids"]
        assert restored.prefix == records[name]["prefix"]
    return {"provenance": provenance, "plans": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hub", action="store_true",
                        help="also write tokenizer_qwen.json from $YUE2_HF_HUB's pinned qwen.tiktoken")
    args = parser.parse_args()
    guard_pattern_copy()
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = {
        "generator": "scripts/reference/yue2/protocol_fixtures.py",
        "yue2_commit": YUE2_COMMIT,
        "python": sys.version.split()[0],
        "unicodedata": unicodedata.unidata_version,
        "tiktoken": tiktoken.__version__,
        "torch": torch.__version__,
    }

    ranks = train_synthetic_ranks()
    synthetic = tiktoken_lines(ranks)
    (OUT / "synthetic.tiktoken").write_bytes(synthetic)
    with tempfile.TemporaryDirectory() as scratch:
        tok = upstream_on_synthetic(ranks, Path(scratch))
        synth_prov = {**provenance, "tokenizer": "synthetic.tiktoken",
                      "synthetic_ranks": len(ranks),
                      "synthetic_sha256": hashlib.sha256(synthetic).hexdigest()}
        write_json(OUT / "tokenizer_synthetic.json", tokenizer_fixture(tok, synth_prov))
        write_json(OUT / "plans.json", plans(tok, synth_prov))

    write_json(OUT / "protocol_cases.json", {
        "provenance": provenance,
        "sampling": sampling_cases(),
        "generation_config": generation_config_cases(),
        "request": song_request_cases(),
        "chunk_ranges": chunk_cases(),
        "budget": budget_cases(),
        # SongRequest refuses external ABC for which `not abc.strip()`: Python's whitespace set.
        "python_isspace": [cp for cp in range(0x110000) if chr(cp).isspace()],
    })

    if args.hub:
        hub = Path(os.environ["YUE2_HF_HUB"])
        repo, revision = YUE2_3B
        path = hub / f"models--{repo.replace('/', '--')}" / "snapshots" / revision / "qwen.tiktoken"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != QWEN_TIKTOKEN_SHA256:
            sys.exit(f"{path}: sha256 {digest} is not the pinned {QWEN_TIKTOKEN_SHA256}")
        tok = YuE2TextTokenizer(path)
        write_json(OUT / "tokenizer_qwen.json",
                   tokenizer_fixture(tok, {**provenance, "tokenizer": "qwen.tiktoken",
                                           "repo": repo, "revision": revision,
                                           "sha256": QWEN_TIKTOKEN_SHA256}))


if __name__ == "__main__":
    main()
