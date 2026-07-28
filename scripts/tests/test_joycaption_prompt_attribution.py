import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROMPT_FILES = (
    ROOT / "crates/media/candle-gen/candle-gen-joycaption/src/prompt.rs",
    ROOT / "crates/media/mlx-gen/src/caption/joycaption.rs",
)
NOTICE_END = "//! JoyCaption caption **product policy**"
ATTRIBUTED_DATA_START = "pub const JOY_NAME_OPTION"
PROMPT_TABLE_START = "const PROMPT_TEMPLATES"
PROMPT_TABLE_END = "pub fn capabilities"
SELECTION_START = "pub fn build_prompt"
SELECTION_END = "#[cfg(test)]"
EXPECTED_NAME_OPTION = (
    "If there is a person/character in the image you must refer to them as {name}."
)
EXPECTED_CAPTION_TYPES = (
    "Descriptive",
    "Descriptive (Casual)",
    "Straightforward",
    "Stable Diffusion Prompt",
    "MidJourney",
    "Danbooru tag list",
    "e621 tag list",
    "Rule34 tag list",
    "Booru-like tag list",
    "Art Critic",
    "Product Listing",
    "Social Media Post",
)
EXPECTED_CAPTION_LENGTHS = ("any", "very short", "short", "medium-length", "long")
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


def attributed_prompt_surface(source: str) -> str:
    start = source.find(ATTRIBUTED_DATA_START)
    end = source.find(PROMPT_TABLE_END)
    if start < 0 or end < start:
        raise AssertionError("JoyCaption attributed prompt surface is missing")
    return source[start:end]


def rust_string_list(source: str, start_marker: str, end_marker: str) -> tuple[str, ...]:
    start = source.find(start_marker)
    end = source.find(end_marker, start)
    if start < 0 or end < start:
        raise AssertionError(f"JoyCaption string list {start_marker!r} is missing")
    return tuple(re.findall(r'"([^"]*)"', source[start:end]))


def selection_behavior(source: str) -> str:
    start = source.find(SELECTION_START)
    end = source.find(SELECTION_END, start)
    if start < 0 or end < start:
        raise AssertionError("JoyCaption prompt-selection implementation is missing")
    return source[start:end]


def validate_attributed_prompt_surface(source: str) -> None:
    attributed_prompt_surface(source)
    name_options = rust_string_list(source, ATTRIBUTED_DATA_START, "pub const CAPTION_TYPES")
    if name_options != (EXPECTED_NAME_OPTION,):
        raise AssertionError("JoyCaption name option diverged from the attributed source")
    caption_types = rust_string_list(source, "pub const CAPTION_TYPES", "pub const CAPTION_LENGTHS")
    if caption_types != EXPECTED_CAPTION_TYPES:
        raise AssertionError("JoyCaption caption taxonomy diverged from the attributed source")
    caption_lengths = rust_string_list(source, "pub const CAPTION_LENGTHS", PROMPT_TABLE_START)
    if caption_lengths != EXPECTED_CAPTION_LENGTHS:
        raise AssertionError("JoyCaption caption lengths diverged from the documented modification")


def validate_selection_behavior(source: str) -> None:
    selection = selection_behavior(source)
    for required_branch in (
        'caption_length == "any"',
        "!caption_length.is_empty() && caption_length.chars().all(|c| c.is_ascii_digit())",
        "templates_for(&options.caption_type)[template_index]",
        ".unwrap_or(&PROMPT_TEMPLATES[0].1)",
    ):
        if required_branch not in selection:
            raise AssertionError(
                f"JoyCaption prompt-selection behavior is missing {required_branch!r}"
            )


def validate_described_modifications(source: str) -> None:
    table = prompt_table(source)
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
        self.assertEqual(
            attributed_prompt_surface(sources[0]), attributed_prompt_surface(sources[1])
        )
        self.assertEqual(prompt_table(sources[0]), prompt_table(sources[1]))
        self.assertEqual(selection_behavior(sources[0]), selection_behavior(sources[1]))
        for source in sources:
            validate_attributed_prompt_surface(source)
            validate_selection_behavior(source)
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

    def test_guard_rejects_attributed_surface_and_selection_mutations(self):
        source = PROMPT_FILES[0].read_text(encoding="utf-8")
        surface_mutations = (
            (
                "name option",
                source.replace(EXPECTED_NAME_OPTION, f"{EXPECTED_NAME_OPTION} Changed.", 1),
            ),
            (
                "caption taxonomy",
                source.replace('"Social Media Post",', '"Social Media Thread",', 1),
            ),
            (
                "caption lengths",
                source.replace('"long"];', '"long", "very long"];', 1),
            ),
        )
        for mutation, mutated in surface_mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(AssertionError, "JoyCaption"):
                    validate_attributed_prompt_surface(mutated)

        empty_semantics_mutation = source.replace("!caption_length.is_empty() && ", "", 1)
        with self.assertRaisesRegex(AssertionError, "JoyCaption"):
            validate_selection_behavior(empty_semantics_mutation)


if __name__ == "__main__":
    unittest.main()
