import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROMPT_FILES = (
    ROOT / "crates/media/candle-gen/candle-gen-joycaption/src/prompt.rs",
    ROOT / "crates/media/mlx-gen/src/caption/joycaption.rs",
)
NOTICE_END = "//! JoyCaption caption **product policy**"
PROMPT_TABLE_START = "const PROMPT_TEMPLATES"
PROMPT_TABLE_END = "pub fn capabilities"
STRAIGHTFORWARD_MODIFICATION = (
    "Never mention what is absent, resolution, watermarks, signatures, compression artifacts, "
    "or unobservable details."
)
REQUIRED_MARKERS = (
    "//! ATTRIBUTION:",
    "https://github.com/fpgaminer/joycaption",
    "`gradio-app/app.py`'s `CAPTION_TYPE_MAP` and `NAME_OPTION`",
    "8445b2e55db7856d522e44ae84e7415fcf3413f6",
    "Apache License 2.0",
    "Copyright 2024 fpgaminer@bitcoin-mining.com",
    "`joycaption-source`",
    "//! MODIFIED BY SCENEWORKS (Apache-2.0 section 4(b)):",
    "ported from Python to Rust",
    "normalized prompt punctuation",
    "removed `very long`",
    "three Straightforward prompts",
    "watermark, signature, and",
    "compression-artifact mentions to forbidding them",
)


def attribution_notice(source: str) -> str:
    end = source.find(NOTICE_END)
    if end < 0:
        raise AssertionError("JoyCaption product-policy module header is missing")
    return source[:end]


def validate_notice(notice: str) -> None:
    for marker in REQUIRED_MARKERS:
        if marker not in notice:
            raise AssertionError(f"JoyCaption attribution notice is missing {marker!r}")


def prompt_table(source: str) -> str:
    start = source.find(PROMPT_TABLE_START)
    end = source.find(PROMPT_TABLE_END)
    if start < 0 or end < start:
        raise AssertionError("JoyCaption prompt table is missing")
    return source[start:end]


def validate_described_modifications(source: str) -> None:
    table = prompt_table(source)
    lengths_start = source.find("pub const CAPTION_LENGTHS")
    if lengths_start < 0:
        raise AssertionError("JoyCaption caption lengths are missing")
    caption_lengths = source[lengths_start : source.find(PROMPT_TABLE_START)]
    if '"very long"' in caption_lengths:
        raise AssertionError("JoyCaption caption lengths restored upstream's `very long` option")
    if table.count(STRAIGHTFORWARD_MODIFICATION) != 3:
        raise AssertionError("JoyCaption Straightforward modification is not present three times")
    for upstream_punctuation in (
        "—",
        "“",
        "”",
        "…",
        "`artist:`",
        "tags (if any)",
        "prefixed by 'artist:'",
    ):
        if upstream_punctuation in table:
            raise AssertionError(
                f"JoyCaption prompt table restored upstream punctuation {upstream_punctuation!r}"
            )


class JoyCaptionPromptAttributionTests(unittest.TestCase):
    def test_both_prompt_maps_carry_the_same_complete_notice(self):
        sources = [path.read_text(encoding="utf-8") for path in PROMPT_FILES]
        notices = [attribution_notice(source) for source in sources]
        for notice in notices:
            validate_notice(notice)
        self.assertEqual(notices[0], notices[1])
        self.assertEqual(prompt_table(sources[0]), prompt_table(sources[1]))
        for source in sources:
            validate_described_modifications(source)

    def test_guard_rejects_each_required_notice_marker_mutation(self):
        notice = attribution_notice(PROMPT_FILES[0].read_text(encoding="utf-8"))
        for marker in REQUIRED_MARKERS:
            with self.subTest(marker=marker):
                mutated = notice.replace(marker, "", 1)
                with self.assertRaisesRegex(
                    AssertionError, "JoyCaption attribution notice is missing"
                ):
                    validate_notice(mutated)

    def test_guard_rejects_each_described_source_mutation(self):
        source = PROMPT_FILES[0].read_text(encoding="utf-8")
        mutations = (
            source.replace('"long"];', '"long", "very long"];', 1),
            source.replace(STRAIGHTFORWARD_MODIFICATION, "Note any watermarks.", 1),
            source.replace("elements-people", "elements—people", 1),
            source.replace("artist:, copyright:", "`artist:`, copyright:", 1),
            source.replace("tags, if any,", "tags (if any),", 1),
            source.replace("prefixed by artist:", "prefixed by 'artist:'", 1),
        )
        for mutated in mutations:
            with self.subTest():
                with self.assertRaisesRegex(AssertionError, "JoyCaption"):
                    validate_described_modifications(mutated)


if __name__ == "__main__":
    unittest.main()
