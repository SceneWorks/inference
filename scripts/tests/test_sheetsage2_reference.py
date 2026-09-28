"""Model-free checks of the SheetSage2 reference experiment's evaluation helpers (sc-23003).

The experiment itself needs weights and a pinned Python environment; these helpers decide the
booleans committed in scripts/reference/sheetsage2/artifacts/evaluation.json and the fixture
digest guard, so they are pinned here, including the mutations that must flip them.
"""

import importlib.util
import json
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "reference" / "sheetsage2" / "run_experiment.py"
SPEC = importlib.util.spec_from_file_location("sheetsage2_run_experiment", SCRIPT)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)

HEADER = [
    "X:1", "T:", "M:3/4", "L:1/16", "Q:1/4=81",
    'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
    'V: Ins clef=treble name="Ins Melody" snm="Inst."', "K:C", "% verse",
]
FULL = "\n".join(HEADER + ["V: Vocal", '"C"g4e4c4|"G7"d4B4G4|', "V: Ins", "Z2|"]) + "\n"
MELODY = "\n".join(HEADER + ["V: Vocal", "g4e4c4|d4B4G4|", "V: Ins", "Z2|"]) + "\n"


class MelodyOnlyRelationTests(unittest.TestCase):
    def test_melody_only_equal_to_full_without_chords(self) -> None:
        relation = experiment.melody_only_relation(FULL, MELODY)
        self.assertFalse(relation["melody_only_has_chord_symbols"])
        self.assertTrue(relation["melody_only_equals_full_without_chord_symbols"])
        self.assertEqual(relation["melody_only_vs_full_without_chord_symbols_diff"], [])

    def test_trailing_newline_does_not_change_the_answer(self) -> None:
        relation = experiment.melody_only_relation(FULL.rstrip("\n"), MELODY)
        self.assertFalse(relation["melody_only_has_chord_symbols"])
        self.assertTrue(relation["melody_only_equals_full_without_chord_symbols"])

    def test_chord_symbol_left_in_melody_flips_has_chords(self) -> None:
        mutated = MELODY.replace("g4e4c4|", '"C"g4e4c4|')
        relation = experiment.melody_only_relation(FULL, mutated)
        self.assertTrue(relation["melody_only_has_chord_symbols"])

    def test_header_quotes_are_not_chord_symbols(self) -> None:
        relation = experiment.melody_only_relation(FULL, MELODY)
        self.assertFalse(relation["melody_only_has_chord_symbols"])  # V: lines carry quoted names

    def test_rest_merge_is_reported_as_a_difference(self) -> None:
        full = FULL.replace('"G7"d4B4G4|', 'z2"G7"z2B4G4|')
        melody = MELODY.replace("d4B4G4|", "z4B4G4|")
        relation = experiment.melody_only_relation(full, melody)
        self.assertFalse(relation["melody_only_equals_full_without_chord_symbols"])
        diff = relation["melody_only_vs_full_without_chord_symbols_diff"]
        self.assertIn("-g4e4c4|z2z2B4G4|", diff)
        self.assertIn("+g4e4c4|z4B4G4|", diff)


class FixtureDigestGuardTests(unittest.TestCase):
    def test_committed_fixtures_match_themselves(self) -> None:
        committed = json.loads((SCRIPT.parent / "artifacts" / "fixtures.json").read_text(encoding="utf-8"))
        self.assertEqual(experiment.check_against_committed(committed, committed), [])

    def test_changed_model_input_digest_is_reported(self) -> None:
        committed = json.loads((SCRIPT.parent / "artifacts" / "fixtures.json").read_text(encoding="utf-8"))
        regenerated = json.loads(json.dumps(committed))
        regenerated["synth"]["model_input"]["sha256"] = "0" * 64
        problems = experiment.check_against_committed(regenerated, committed)
        self.assertEqual(len(problems), 1)
        self.assertIn("synth model_input.sha256", problems[0])

    def test_every_fixture_pins_its_model_input_array(self) -> None:
        committed = json.loads((SCRIPT.parent / "artifacts" / "fixtures.json").read_text(encoding="utf-8"))
        for name, entry in committed.items():
            with self.subTest(fixture=name):
                self.assertEqual(len(entry["model_input"]["sha256"]), 64)
                self.assertEqual(entry["model_input"]["sample_rate"], 24000)


if __name__ == "__main__":
    unittest.main()
