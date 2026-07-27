"""Gate the SA3 listening panel's blinding, scoring and pre-registered analysis (sc-15178).

`scripts/audio/sa3_listening_blind.py` is the executable half of
`docs/migration/SC_15178_SA3_LISTENING_PROTOCOL.md`. Two properties in it are load-bearing and
would otherwise be prose:

* **The blinding actually blinds.** A playlist that leaks a variant name, a prompt, or the
  generator's variant-bearing filenames is not a blinded playlist, and a panel run on one measures
  expectation rather than audio.
* **A failed ABX suppresses the preference result.** ABX answers discriminability and MOS answers
  preference; sc-14545's acceptance asks the second, and the first gates it. Reporting a preference
  between two takes listeners cannot tell apart manufactures a finding out of noise. The tests below
  drive the analyzer with a simulated null panel and require `mos` to come back `None`.

The power numbers are re-derived here from an independent closed form rather than compared against
themselves, so a mutation to the analysis constants cannot pass by also mutating the expectation.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "audio" / "sa3_listening_blind.py"
PROTOCOL = ROOT / "docs" / "migration" / "SC_15178_SA3_LISTENING_PROTOCOL.md"
SA3_TESTS = ROOT / "crates" / "audio" / "candle-audio-stable-audio-3" / "tests"
GENERATOR = SA3_TESTS / "listening_stimuli.rs"


def load_module():
    spec = importlib.util.spec_from_file_location("sa3_listening_blind", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


blind = load_module()


def _join_rust_continuations(source: str) -> str:
    """Undo Rust's in-string line continuation (`\\` + newline + indent swallows both)."""
    return re.sub(r"\\\n\s*", "", source)


def _prose(text: str) -> str:
    """Flatten prose so a sentence needle survives line wrapping.

    Both files wrap: Markdown at the column, Rust behind `///`. A needle matched against the raw
    text would silently stop matching the moment a sentence rewrapped, which would make the count
    checks below quietly vacuous rather than loudly wrong.
    """
    stripped = re.sub(r"(?m)^\s*(///|//!)\s?", "", text)
    return re.sub(r"\s+", " ", stripped)


def parse_stimuli() -> list[dict]:
    """Read the pinned stimulus table straight out of the Rust generator.

    Parsed rather than duplicated: a copy of this table in Python could drift from the one the
    panel actually hears, and every cross-language invariant below would then be checking the copy.
    """
    source = _join_rust_continuations(GENERATOR.read_text(encoding="utf-8"))
    block = re.search(r"const STIMULI: &\[Stimulus\] = &\[(.*?)\n\];", source, re.S)
    assert block, "could not locate the STIMULI table in listening_stimuli.rs"
    entries = []
    for raw in re.finditer(
        r'Stimulus\s*\{\s*id:\s*"([^"]+)",\s*domain:\s*Domain::(\w+),\s*'
        r'prompt:\s*"((?:[^"\\]|\\.)*)",\s*held_out:\s*(true|false),\s*\}',
        block.group(1),
    ):
        entries.append({
            "id": raw.group(1),
            "domain": raw.group(2).lower(),
            "prompt": raw.group(3),
            "held_out": raw.group(4) == "true",
        })
    return entries


def parse_seeds() -> list[int]:
    source = GENERATOR.read_text(encoding="utf-8")
    block = re.search(r"const SEEDS: &\[u64\] = &\[([^\]]*)\];", source)
    assert block, "could not locate the SEEDS constant in listening_stimuli.rs"
    return [int(value.strip().replace("_", "")) for value in block.group(1).split(",") if value.strip()]


def parse_demo_cond_pool() -> list[tuple[str, list[str]]]:
    """Read the committed per-snapshot `demo_cond` pool out of the Rust generator.

    The pool itself is checked against the six on-disk `model_config.json` files by
    `the_committed_demo_cond_pool_matches_the_shipped_configs`, which needs the snapshots and is
    therefore `#[ignore]`d. What is checked *here*, weight-free, is everything derived from it.
    """
    source = _join_rust_continuations(GENERATOR.read_text(encoding="utf-8"))
    block = re.search(
        r"const DEMO_COND_POOL: &\[\(&str, &\[&str\]\)\] = &\[(.*?)\n\];", source, re.S
    )
    assert block, "could not locate the DEMO_COND_POOL table in listening_stimuli.rs"
    pool: list[tuple[str, list[str]]] = []
    for raw in re.finditer(r'\(\s*"([^"]+)",\s*&\[(.*?)\],?\s*\)', block.group(1), re.S):
        prompts = [m.group(1) for m in re.finditer(r'"((?:[^"\\]|\\.)*)"', raw.group(2))]
        pool.append((raw.group(1), prompts))
    return pool


def parse_sweep_prompts() -> dict[str, list[str]]:
    """Read `tests/provider.rs`'s `*_SWEEP_PROMPTS` side-ratio calibration constants."""
    source = _join_rust_continuations((SA3_TESTS / "provider.rs").read_text(encoding="utf-8"))
    sweeps = {}
    for raw in re.finditer(
        r"const (\w+_SWEEP_PROMPTS): &\[&str\] = &\[(.*?)\n\];", source, re.S
    ):
        sweeps[raw.group(1)] = [
            m.group(1) for m in re.finditer(r'"((?:[^"\\]|\\.)*)"', raw.group(2))
        ]
    return sweeps


STIMULI = parse_stimuli()
SEEDS = parse_seeds()
DEMO_COND_POOL = parse_demo_cond_pool()
SWEEP_PROMPTS = parse_sweep_prompts()


def specialist_for(domain: str) -> str:
    return "stable_audio_3_small_music" if domain == "music" else "stable_audio_3_small_sfx"


def fake_manifest() -> dict:
    """A manifest with the exact shape `tests/listening_stimuli.rs` writes."""
    takes = []
    for entry in STIMULI:
        stimulus, domain = entry["id"], entry["domain"]
        for seed in SEEDS:
            for variant in (blind.MEDIUM_VARIANT, specialist_for(domain)):
                takes.append({
                    "file": f"{stimulus}_{variant}_seed{seed}.wav",
                    "stimulus": stimulus,
                    "domain": domain,
                    "prompt": entry["prompt"],
                    "held_out": entry["held_out"],
                    "variant": variant,
                    "seed": seed,
                    "seconds": 30.0,
                    "steps": 8,
                    "sampler": "pingpong",
                    "sample_rate": 44_100,
                    "pre_lufs": -21.0,
                    "gain_linear": 0.8,
                    "gain_db": -1.94,
                    "post_lufs": -23.0,
                    "post_true_peak_dbtp": -4.0,
                    "pcm_sha256": "0" * 64,
                })
    return {"story": "sc-15178", "target_lufs": -23.0, "takes": takes}


def simulate(key: dict, abx_accuracy: float, medium_bonus: float, noise_seed: int = 99,
             null_accuracy: float = 0.5) -> dict:
    """Drive the analyzer with a synthetic panel at a known truth.

    `abx_accuracy` is the fraction of *contrast* trials answered correctly and `null_accuracy` the
    fraction of same-checkpoint trials; `medium_bonus` is the true per-screen rating advantage in
    points. Responses are laid down deterministically rather than sampled so a test asserting a
    p-value threshold is not flaky.
    """
    rng = blind.SplitMix64(noise_seed)
    abx_rows = []
    # Hits are laid down over the *pooled* trials, not per listener: the pre-registered ABX test is
    # pooled, so rounding per listener would make a requested 0.55 land somewhere else entirely.
    for kind, share in (("contrast", abx_accuracy), ("null", null_accuracy)):
        trials = [
            (listener, trial)
            for listener, design in sorted(key["listeners"].items())
            for trial in design["abx"] if trial["kind"] == kind
        ]
        hits = round(len(trials) * share)
        for index, (listener, trial) in enumerate(trials):
            correct = index < hits
            response = trial["x_is"] if correct else ("b" if trial["x_is"] == "a" else "a")
            abx_rows.append({"listener": listener, "trial": trial["trial"], "x_is": response})

    rating_rows = []
    for listener, design in sorted(key["listeners"].items()):
        # A per-listener offset spreads the ratings so the paired test sees a realistic sd.
        offset = (rng.below(21) - 10) / 2.0
        for screen in design["ratings"]:
            for index, slot in enumerate(screen["slots"], 1):
                base = 55.0 + offset
                if slot["variant"] == blind.MEDIUM_VARIANT:
                    base += medium_bonus
                # A small, sign-alternating jitter keeps the duplicate close to its original
                # (so nobody is excluded) without making the paired differences degenerate.
                base += (rng.below(5) - 2) * 0.5
                rating_rows.append({
                    "listener": listener,
                    "screen": screen["screen"],
                    "slot": str(index),
                    "rating": f"{min(max(base, 0.0), 100.0):.2f}",
                })
    return blind.unblind(key, abx_rows, rating_rows)


class PreRegistrationTests(unittest.TestCase):
    """The committed numbers, re-derived independently."""

    def test_abx_critical_value_matches_a_direct_enumeration(self) -> None:
        pre = blind.preregistration()
        n = pre["abx"]["pooled_trials"]
        self.assertEqual(n, blind.PANEL_SIZE * blind.CONTRAST_TRIALS_PER_LISTENER)
        # Independent: the smallest k whose upper tail under a fair coin is within alpha.
        expected = min(
            k for k in range(n + 1)
            if sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n <= blind.ABX_ALPHA
        )
        self.assertEqual(pre["abx"]["critical_count"], expected)
        self.assertEqual(expected, 134)

    def test_abx_is_powered_for_the_stated_effect(self) -> None:
        pre = blind.preregistration()["abx"]
        self.assertGreaterEqual(pre["power_at_effect_of_interest"], blind.TARGET_POWER)
        # And the design is honest about what it cannot see: the minimum detectable accuracy sits
        # strictly above chance, so "no effect found" never means "no effect was detectable".
        self.assertGreater(pre["minimum_detectable_accuracy"], blind.ABX_CHANCE)
        self.assertLess(pre["minimum_detectable_accuracy"], pre["effect_of_interest"])

    def test_mos_panel_size_reaches_target_power_at_the_stated_effect(self) -> None:
        pre = blind.preregistration()["mos"]
        self.assertAlmostEqual(pre["effect_dz"], blind.MOS_EFFECT_POINTS / blind.MOS_ASSUMED_SD)
        self.assertGreaterEqual(pre["power_at_effect"], blind.TARGET_POWER)
        # The panel is sized *by* the MOS half, so one listener fewer must drop below target --
        # otherwise the stated derivation is not the thing that produced the number.
        self.assertLess(
            blind.paired_t_power(blind.PANEL_SIZE - 1, pre["effect_dz"]),
            blind.TARGET_POWER,
        )

    def test_student_and_noncentral_t_agree_with_known_values(self) -> None:
        # t(0.975, 19) = 2.093 to three decimals; the two-sided critical value the MOS test uses.
        self.assertAlmostEqual(blind.student_critical(19, 0.05), 2.093, places=3)
        # A zero effect must land at exactly the nominal size, which is the sharpest available
        # check that the non-central integration is normalized.
        self.assertAlmostEqual(blind.paired_t_power(20, 0.0), 0.05, places=3)

    def test_null_control_band_brackets_chance(self) -> None:
        pre = blind.preregistration()["null_control"]
        low, high = pre["acceptance_counts"]
        n = pre["pooled_trials"]
        self.assertLess(low / n, blind.ABX_CHANCE)
        self.assertGreater(high / n, blind.ABX_CHANCE)

    def test_the_null_control_band_is_the_central_95_percent_and_is_symmetric(self) -> None:
        """Derived from the tail probabilities, not hardcoded.

        The previous version of this test asserted `[31, 50]` as a literal, which is how an
        off-by-one survived: `binomial_acceptance` returned the smallest count in the *upper
        rejection* region and used it as an inclusive bound, so the band was one count too wide at
        the top -- and that end is the leak direction the control exists to catch. Re-deriving both
        ends from an independent enumeration is what makes this assertion discriminate.
        """
        pre = blind.preregistration()["null_control"]
        low, high = pre["acceptance_counts"]
        n = pre["pooled_trials"]
        tail = 0.05 / 2.0

        def upper(k: int) -> float:
            return sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n

        def lower(k: int) -> float:
            return sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n

        # Both ends are the extreme *accepted* counts: one step outside the band is in a rejection
        # region, one step inside is not.
        self.assertLessEqual(lower(low - 1), tail)
        self.assertGreater(lower(low), tail)
        self.assertLessEqual(upper(high + 1), tail)
        self.assertGreater(upper(high), tail)

        # Symmetric about n/2 -- the binomial at p = 0.5 is, so an asymmetric band is a bug.
        self.assertEqual(low + high, n)
        self.assertAlmostEqual(lower(low - 1), upper(high + 1))

        # And the band really is the central 95%, not the 97.4% the inclusive-rejection-bound
        # version covered.
        coverage = 1.0 - lower(low - 1) - upper(high + 1)
        self.assertGreaterEqual(coverage, 0.95)
        self.assertLess(coverage, 0.97)

    def test_the_acceptance_band_never_returns_a_rejecting_count(self) -> None:
        """Across panel sizes, so the fix is a property rather than a fact about n = 80."""
        for n in (8, 16, 40, 80, 96, 120):
            low, high = blind.binomial_acceptance(n)
            self.assertLessEqual(low, high, f"n={n}")
            self.assertGreater(blind.binomial_sf(high, n, 0.5), 0.025, f"n={n}: high rejects")
            self.assertGreater(
                1.0 - blind.binomial_sf(low + 1, n, 0.5), 0.025, f"n={n}: low rejects"
            )

    def test_the_protocol_document_quotes_the_computed_numbers(self) -> None:
        """The doc's table and this script must not be able to drift apart."""
        text = PROTOCOL.read_text(encoding="utf-8")
        pre = blind.preregistration()
        for needle in (
            str(pre["panel_size"]),
            str(pre["abx"]["pooled_trials"]),
            str(pre["abx"]["critical_count"]),
            f"{pre['abx']['minimum_detectable_accuracy']:.3f}",
            f"{pre['mos']['minimum_detectable_points']:.1f}",
            str(pre["null_control"]["acceptance_counts"][0]),
            str(pre["null_control"]["acceptance_counts"][1]),
            str(pre["assignment_seed"]),
            str(pre["recruit_size"]),
        ):
            self.assertIn(needle, text, f"protocol document does not quote {needle!r}")

    def test_the_protocol_document_pins_the_abx_effect_of_interest_and_its_power(self) -> None:
        """`ABX_EFFECT_OF_INTEREST` sizes nothing else, so nothing above catches a change to it:
        0.65 -> 0.70 moves the power from 0.9986 to 1.0000 and leaves every other number alone.
        Both cells are bound to their table rows -- a bare `0.65` is a weak needle, and the
        `power at that effect` row also exists for the MOS half at a different value."""
        text = PROTOCOL.read_text(encoding="utf-8")
        abx = blind.preregistration()["abx"]
        for needle in (
            f"| effect worth acting on | {abx['effect_of_interest']:g} pooled accuracy |",
            f"| power at that effect | **{abx['power_at_effect_of_interest']:.3f}** |",
        ):
            self.assertIn(
                needle,
                text,
                f"protocol document's Q1 table does not quote {needle!r}",
            )

    def test_the_protocol_document_pins_the_committed_assignment_seed(self) -> None:
        """The bare-number needle above is weak for the seed -- `15178` is also the story id and a
        render seed, so it would be found in the document no matter what. This binds the seed to the
        sentence that pins it, so the constant cannot be changed with the suite still green."""
        text = PROTOCOL.read_text(encoding="utf-8")
        self.assertIn(
            f"committed randomization seed `{blind.DEFAULT_ASSIGNMENT_SEED}`",
            text,
            "protocol section 5 must pin the assignment seed the script actually uses",
        )

    def test_the_runbook_generates_the_recruited_number_of_playlists(self) -> None:
        """`RECRUIT_SIZE` states attrition headroom; a runbook that never passes it would yield
        exactly PANEL_SIZE playlists and no headroom at all."""
        text = PROTOCOL.read_text(encoding="utf-8")
        self.assertGreater(blind.RECRUIT_SIZE, blind.PANEL_SIZE)
        self.assertIn(f"--panel-size {blind.RECRUIT_SIZE}", text)


class StimulusHoldOutTests(unittest.TestCase):
    """The hold-out claim, re-derived from the sources rather than trusted.

    `listening_stimuli.rs` flags four of six prompts as held out from every committed test
    constant. That is a cross-file property the Rust file cannot check about itself, and the naive
    version of it is *false*: `tests/provider.rs` commits every shipped `demo_cond` prompt of every
    SA3 variant as a side-ratio calibration constant, so nothing in the `demo_cond` pool is held
    out at all. These tests are what stop that flag from becoming decoration.
    """

    @staticmethod
    def other_crate_sources() -> str:
        return "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((ROOT / "crates").rglob("*.rs"))
            if path != GENERATOR
        )

    def test_the_parser_found_the_real_table(self) -> None:
        self.assertEqual(len(STIMULI), 6)
        self.assertEqual(sorted({entry["domain"] for entry in STIMULI}), ["music", "sfx"])
        self.assertTrue(all(entry["prompt"] for entry in STIMULI))
        self.assertEqual(SEEDS, [15_178, 15_377])

    def test_held_out_prompts_appear_in_no_other_crate_source(self) -> None:
        sources = self.other_crate_sources()
        leaked = [e["id"] for e in STIMULI if e["held_out"] and e["prompt"] in sources]
        self.assertEqual(
            leaked,
            [],
            "these stimuli are flagged held out but are already committed elsewhere in crates/: "
            + ", ".join(leaked),
        )

    def test_the_scan_finds_the_anchors_so_it_is_not_vacuous(self) -> None:
        """The discriminating control: a scanner that finds nothing would pass the test above
        no matter what the held-out prompts were."""
        sources = self.other_crate_sources()
        anchors = [e["id"] for e in STIMULI if not e["held_out"]]
        self.assertTrue(anchors, "the set must keep at least one committed gate prompt as an anchor")
        for entry in STIMULI:
            if not entry["held_out"]:
                self.assertIn(
                    entry["prompt"],
                    sources,
                    f"{entry['id']} is flagged as a committed gate prompt but is not committed "
                    "anywhere; either the flag or the scan is wrong",
                )

    def test_at_least_half_the_set_is_held_out(self) -> None:
        held_out = sum(1 for entry in STIMULI if entry["held_out"])
        self.assertGreaterEqual(held_out / len(STIMULI), 0.5)

    def test_both_domains_and_the_load_bearing_sparse_case_are_present(self) -> None:
        by_domain = {domain: [e for e in STIMULI if e["domain"] == domain]
                     for domain in ("music", "sfx")}
        self.assertTrue(by_domain["music"], "medium serves music; the panel must cover it")
        self.assertTrue(by_domain["sfx"], "medium is the only SA3 checkpoint tagged for both")
        # sc-14545 comment #15161: medium's side ratio on this prompt collapses to ~1.2e-4 at two
        # of five seeds. It is where the checkpoints most plausibly differ in character.
        self.assertIn(
            "Dog barking next to a waterfall",
            [entry["prompt"] for entry in STIMULI],
            "the load-bearing sparse-SFX prompt was dropped from the stimulus set",
        )

    def test_the_generator_and_the_preregistration_agree_on_the_trial_count(self) -> None:
        """The cross-language invariant: the Rust table sizes the Python pre-registration."""
        self.assertEqual(len(STIMULI) * len(SEEDS), blind.CONTRAST_TRIALS_PER_LISTENER)
        self.assertEqual(len(STIMULI), blind.RATING_SCREENS_PER_LISTENER)

    def test_the_seeds_are_disjoint_from_the_crates_gate_seeds(self) -> None:
        """A panel scored on the draws a threshold was calibrated at inherits that luck."""
        self.assertEqual(set(SEEDS) & {42, 7, 2_026, 14_544, 31_337}, set())


class DemoCondPoolTests(unittest.TestCase):
    """Everything §3 and `listening_stimuli.rs` say about the shipped `demo_cond` pool, recomputed.

    Round-1 review asked for the blanket claim ("every prompt of every variant") to be replaced by
    counts. The replacement stated **14 distinct prompts**, which is the per-snapshot sum with only
    `small-sfx-base` deduplicated -- the correction introduced a derived number that nothing
    derived. The true figures are 11 distinct across 18 per-snapshot entries, and they are computed
    here from the committed table so no future edit can restate them by hand.
    """

    #: The two entries the sweep constants carry that are *not* `demo_cond` prompts: the neutral
    #: prompt the SFX divergence gate renders, and the prompt the music/medium per-run gates
    #: render. Both are named in the sweeps' own doc comments ("plus ...").
    GATE_ADDITIONS = frozenset({
        "a short bright transient followed by a decaying tail",
        "warm cinematic post-rock with bowed strings and restrained drums",
    })

    def by_snapshot(self) -> dict[str, list[str]]:
        return dict(DEMO_COND_POOL)

    def test_the_parsers_found_the_real_tables(self) -> None:
        """A parse that yields nothing would make every count below vacuously agree."""
        self.assertEqual(len(DEMO_COND_POOL), 6, "the pool must cover all six SA3 snapshots")
        self.assertEqual(
            sorted(SWEEP_PROMPTS),
            [
                "MEDIUM_SWEEP_PROMPTS",
                "MUSIC_BASE_SWEEP_PROMPTS",
                "MUSIC_SWEEP_PROMPTS",
                "SFX_SWEEP_PROMPTS",
            ],
        )
        self.assertTrue(all(SWEEP_PROMPTS.values()))

    def test_the_demo_cond_pool_counts_are_derived_from_the_committed_pool(self) -> None:
        entries = sum(len(prompts) for _, prompts in DEMO_COND_POOL)
        distinct = {prompt for _, prompts in DEMO_COND_POOL for prompt in prompts}
        by_snapshot = self.by_snapshot()
        overlap = set(by_snapshot["stable-audio-3-small-music"]) & set(
            by_snapshot["stable-audio-3-small-music-base"]
        )

        # Discriminating: if the pool ever stopped overlapping, the sentence below would be a
        # tautology and the two numbers would carry no information.
        self.assertLess(len(distinct), entries, "the pool no longer contains any duplicates")
        self.assertTrue(overlap, "small-music and small-music-base no longer overlap at all")

        sentence = f"**{len(distinct)} distinct prompts across {entries} per-snapshot entries**"
        share = f"**shares {len(overlap)} of its 4 with `small-music`**"
        for path in (PROTOCOL, GENERATOR):
            text = _prose(path.read_text(encoding="utf-8"))
            self.assertIn(
                sentence,
                text,
                f"{path.name} does not state the counts derived from DEMO_COND_POOL: {sentence!r}",
            )
            self.assertIn(
                share,
                text,
                f"{path.name} does not state the small-music overlap derived from "
                f"DEMO_COND_POOL: {share!r}",
            )

    def test_small_sfx_base_ships_the_music_base_list_not_an_sfx_one(self) -> None:
        """The other duplication the counts depend on, and the more surprising of the two."""
        by_snapshot = self.by_snapshot()
        self.assertEqual(
            by_snapshot["stable-audio-3-small-sfx-base"],
            by_snapshot["stable-audio-3-small-music-base"],
        )
        self.assertNotEqual(
            by_snapshot["stable-audio-3-small-sfx-base"],
            by_snapshot["stable-audio-3-small-sfx"],
        )

    def test_the_crate_commits_every_pooled_prompt(self) -> None:
        """§3's load-bearing claim: nothing in the `demo_cond` pool is held out, because
        `tests/provider.rs` calibrated a floor on all of it."""
        committed = {prompt for prompts in SWEEP_PROMPTS.values() for prompt in prompts}
        missing = sorted(
            {prompt for _, prompts in DEMO_COND_POOL for prompt in prompts} - committed
        )
        self.assertEqual(
            missing,
            [],
            "these shipped demo_cond prompts are in no *_SWEEP_PROMPTS constant, so §3's "
            "'the crate commits all of them' is false: " + "; ".join(missing),
        )

    def test_the_sweep_constants_add_only_the_two_gate_prompts(self) -> None:
        """The other direction. If a sweep grew an entry that is neither a shipped `demo_cond`
        prompt nor a named gate prompt, the test above would still pass while §3's account of what
        those constants contain had gone stale."""
        committed = {prompt for prompts in SWEEP_PROMPTS.values() for prompt in prompts}
        pooled = {prompt for _, prompts in DEMO_COND_POOL for prompt in prompts}
        self.assertEqual(sorted(committed - pooled), sorted(self.GATE_ADDITIONS))

    def test_medium_ships_exactly_two_and_both_are_in_its_sweep(self) -> None:
        """`MEDIUM_SWEEP_PROMPTS`'s doc comment reads like a selection; it is the whole list."""
        medium = self.by_snapshot()["stable-audio-3-medium"]
        self.assertEqual(len(medium), 2)
        for prompt in medium:
            self.assertIn(prompt, SWEEP_PROMPTS["MEDIUM_SWEEP_PROMPTS"])

    def test_medium_base_ships_none(self) -> None:
        self.assertEqual(self.by_snapshot()["stable-audio-3-medium-base"], [])


class RandomnessTests(unittest.TestCase):
    def test_splitmix64_is_the_reference_stream(self) -> None:
        # The published SplitMix64 output for state 0: seeding with 0 and stepping gamma once.
        rng = blind.SplitMix64(0)
        self.assertEqual(
            [rng.next_u64() for _ in range(3)],
            [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F],
        )

    def test_below_is_within_range_and_uses_every_value(self) -> None:
        rng = blind.SplitMix64(7)
        seen = {rng.below(5) for _ in range(400)}
        self.assertEqual(seen, {0, 1, 2, 3, 4})

    def test_shuffle_is_a_permutation(self) -> None:
        items = list(range(50))
        blind.SplitMix64(11).shuffle(items)
        self.assertEqual(sorted(items), list(range(50)))
        self.assertNotEqual(items, list(range(50)))


class AssignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = fake_manifest()
        self.playlists, self.key = blind.assign(self.manifest)

    def test_the_design_is_a_pure_function_of_the_committed_seed(self) -> None:
        again_playlists, again_key = blind.assign(self.manifest)
        self.assertEqual(self.playlists, again_playlists)
        self.assertEqual(self.key, again_key)

        other_playlists, other_key = blind.assign(
            self.manifest, seed=blind.DEFAULT_ASSIGNMENT_SEED + 1
        )
        # The *key* must move with the seed -- that is the randomization. The blinded playlist must
        # NOT: it carries only per-listener opaque handles, so its shape is deliberately
        # seed-independent and an operator cannot infer the design by diffing two playlists.
        self.assertNotEqual(self.key["listeners"], other_key["listeners"])
        self.assertEqual(self.playlists, other_playlists)

    def test_every_listener_gets_the_preregistered_trial_counts(self) -> None:
        self.assertEqual(len(self.key["listeners"]), blind.PANEL_SIZE)
        for design in self.key["listeners"].values():
            kinds = [trial["kind"] for trial in design["abx"]]
            self.assertEqual(kinds.count("contrast"), blind.CONTRAST_TRIALS_PER_LISTENER)
            self.assertEqual(kinds.count("null"), blind.NULL_TRIALS_PER_LISTENER)
            self.assertEqual(len(design["ratings"]), blind.RATING_SCREENS_PER_LISTENER)

    def test_contrast_trials_cover_every_stimulus_and_seed_exactly_once(self) -> None:
        expected = {(entry["id"], seed) for entry in STIMULI for seed in SEEDS}
        for design in self.key["listeners"].values():
            covered = [
                (trial["stimulus"], trial["seed"])
                for trial in design["abx"] if trial["kind"] == "contrast"
            ]
            self.assertEqual(sorted(covered), sorted(expected))

    def test_null_trials_present_the_same_file_three_times(self) -> None:
        """A null trial must carry no information, or it is not a chance-level control."""
        for design in self.key["listeners"].values():
            for trial in design["abx"]:
                if trial["kind"] != "null":
                    continue
                self.assertEqual(trial["a"], trial["b"])
                self.assertEqual(trial["a_variant"], trial["b_variant"])

    def test_ab_assignment_and_x_are_balanced_across_the_panel(self) -> None:
        contrast = [
            trial
            for design in self.key["listeners"].values()
            for trial in design["abx"] if trial["kind"] == "contrast"
        ]
        total = len(contrast)
        medium_first = sum(1 for t in contrast if t["a_variant"] == blind.MEDIUM_VARIANT)
        x_is_a = sum(1 for t in contrast if t["x_is"] == "a")
        # A systematic imbalance would let a listener with a position bias beat chance without
        # hearing anything. Three sd of a fair coin at n=240 is ~23 trials.
        self.assertLess(abs(medium_first - total / 2), 3 * math.sqrt(total) / 2)
        self.assertLess(abs(x_is_a - total / 2), 3 * math.sqrt(total) / 2)

    def test_rating_screens_carry_medium_the_specialist_and_one_duplicate(self) -> None:
        for design in self.key["listeners"].values():
            for screen in design["ratings"]:
                self.assertEqual(len(screen["slots"]), blind.RATING_SLOTS_PER_SCREEN)
                self.assertEqual(sum(1 for s in screen["slots"] if s["duplicate"]), 1)
                originals = [s["variant"] for s in screen["slots"] if not s["duplicate"]]
                self.assertEqual(len(set(originals)), 2)
                self.assertIn(blind.MEDIUM_VARIANT, originals)

    def test_the_blinded_playlists_leak_nothing(self) -> None:
        secrets = sorted(
            {take["variant"] for take in self.manifest["takes"]}
            | {take["prompt"] for take in self.manifest["takes"]}
            | {take["file"] for take in self.manifest["takes"]}
            | {str(seed) for seed in SEEDS}
        )
        self.assertEqual(blind.blinded_leaks(self.playlists, secrets), [])

    def test_the_leak_detector_would_catch_a_regression(self) -> None:
        """A detector that never fires is not evidence the playlists are clean."""
        leaky = {"L01": dict(self.playlists["L01"])}
        leaky["L01"]["abx"] = [{"trial": "t01", "a": f"{blind.MEDIUM_VARIANT}.wav"}]
        self.assertEqual(blind.blinded_leaks(leaky, [blind.MEDIUM_VARIANT]), [blind.MEDIUM_VARIANT])

    def test_a_manifest_of_the_wrong_shape_is_rejected(self) -> None:
        manifest = fake_manifest()
        manifest["takes"] = [t for t in manifest["takes"] if t["seed"] == SEEDS[0]]
        with self.assertRaises(ValueError):
            blind.assign(manifest)

    def test_materialize_copies_to_opaque_names(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "stimuli"
            source.mkdir()
            for take in self.manifest["takes"]:
                (source / take["file"]).write_text("wav", encoding="utf-8")
            playlists, key = blind.assign(self.manifest, panel=2)
            written = blind.materialize(playlists, key, source, root / "panel")
            self.assertEqual(written, 2 * (3 * 16 + 3 * 6))
            for name in (root / "panel" / "L01").iterdir():
                self.assertNotIn(blind.MEDIUM_VARIANT, name.name)


class SheetAndUnblindTests(unittest.TestCase):
    def setUp(self) -> None:
        self.playlists, self.key = blind.assign(fake_manifest(), panel=3)

    def test_sheets_have_one_row_per_trial_and_no_prefilled_answers(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            abx_path, rating_path = blind.write_sheets(self.playlists["L01"], out)
            abx_rows = blind.read_csv(abx_path, blind.ABX_COLUMNS)
            rating_rows = blind.read_csv(rating_path, blind.RATING_COLUMNS)
        self.assertEqual(
            len(abx_rows),
            blind.CONTRAST_TRIALS_PER_LISTENER + blind.NULL_TRIALS_PER_LISTENER,
        )
        self.assertEqual(
            len(rating_rows),
            blind.RATING_SCREENS_PER_LISTENER * blind.RATING_SLOTS_PER_SCREEN,
        )
        self.assertTrue(all(row["x_is"] == "" for row in abx_rows))

    def test_unblinding_scores_a_perfect_and_a_perfectly_wrong_sheet(self) -> None:
        design = self.key["listeners"]["L01"]["abx"]
        right = [{"listener": "L01", "trial": t["trial"], "x_is": t["x_is"]} for t in design]
        wrong = [
            {"listener": "L01", "trial": t["trial"], "x_is": "b" if t["x_is"] == "a" else "a"}
            for t in design
        ]
        self.assertTrue(all(r["correct"] for r in blind.unblind(self.key, right, [])["abx"]))
        self.assertFalse(any(r["correct"] for r in blind.unblind(self.key, wrong, [])["abx"]))

    def test_unblinding_rejects_malformed_responses(self) -> None:
        trial = self.key["listeners"]["L01"]["abx"][0]["trial"]
        for row in (
            {"listener": "L99", "trial": trial, "x_is": "a"},
            {"listener": "L01", "trial": "t99", "x_is": "a"},
            {"listener": "L01", "trial": trial, "x_is": "maybe"},
        ):
            with self.assertRaises(ValueError):
                blind.unblind(self.key, [row], [])

    def test_unblinding_rejects_an_out_of_scale_rating(self) -> None:
        screen = self.key["listeners"]["L01"]["ratings"][0]["screen"]
        with self.assertRaises(ValueError):
            blind.unblind(self.key, [], [
                {"listener": "L01", "screen": screen, "slot": "1", "rating": "101"},
            ])


class AnalysisTests(unittest.TestCase):
    """The pre-registered analysis, and above all the order it enforces."""

    def setUp(self) -> None:
        _, self.key = blind.assign(fake_manifest())

    def test_a_guessing_panel_produces_a_reported_null_and_no_preference(self) -> None:
        """The outcome the story says is likely, and the one most easily mis-reported.

        Listeners at chance, but a large true rating gap laid on top. A reader who only looked at
        the ratings would announce a 12-point preference. The pre-registered order forbids it.
        """
        report = blind.analyze(simulate(self.key, abx_accuracy=0.5, medium_bonus=12.0))
        self.assertTrue(report["validity"]["passed"])
        self.assertFalse(report["abx"]["discriminable"])
        self.assertIsNone(report["mos"], "a failed ABX must suppress the preference result")
        self.assertIn("NULL RESULT", report["conclusion"])
        self.assertGreater(report["abx"]["p_value"], blind.ABX_ALPHA)

    def test_a_barely_above_chance_panel_is_still_a_null(self) -> None:
        """0.55 is above chance and below the pre-registered threshold. It must not be reported
        as a finding just because the raw fraction exceeds 0.5."""
        report = blind.analyze(simulate(self.key, abx_accuracy=0.55, medium_bonus=12.0))
        self.assertFalse(report["abx"]["discriminable"])
        self.assertIsNone(report["mos"])

    def test_a_discriminating_panel_with_no_preference_says_different_not_better(self) -> None:
        report = blind.analyze(simulate(self.key, abx_accuracy=0.9, medium_bonus=0.0))
        self.assertTrue(report["abx"]["discriminable"])
        self.assertIsNotNone(report["mos"])
        self.assertFalse(report["mos"]["significant"])
        self.assertIn("NO PREFERENCE", report["conclusion"])

    def test_a_discriminating_panel_with_a_real_preference_reports_it(self) -> None:
        report = blind.analyze(simulate(self.key, abx_accuracy=0.9, medium_bonus=12.0))
        self.assertTrue(report["abx"]["discriminable"])
        self.assertIsNotNone(report["mos"])
        self.assertTrue(report["mos"]["significant"])
        self.assertTrue(report["mos"]["favours_medium"])
        self.assertLess(report["mos"]["wilcoxon"]["p"], blind.MOS_ALPHA)
        self.assertIn("PREFERENCE FOR MEDIUM", report["conclusion"])

    def test_a_preference_against_medium_is_reported_as_such(self) -> None:
        report = blind.analyze(simulate(self.key, abx_accuracy=0.9, medium_bonus=-12.0))
        self.assertTrue(report["mos"]["significant"])
        self.assertFalse(report["mos"]["favours_medium"])
        self.assertIn("PREFERENCE AGAINST MEDIUM", report["conclusion"])

    def test_a_leaked_panel_is_invalidated_before_anything_else_is_read(self) -> None:
        """Identical files scored well above chance means the presentation leaked."""
        report = blind.analyze(
            simulate(self.key, abx_accuracy=0.9, medium_bonus=12.0, null_accuracy=1.0)
        )
        self.assertFalse(report["validity"]["passed"])
        self.assertIsNone(report["abx"])
        self.assertIsNone(report["mos"])
        self.assertIn("PANEL INVALID", report["conclusion"])

    def test_a_panel_that_arrived_as_designed_reports_no_deviation(self) -> None:
        """The discriminating half of the pair below: a check that always fires is not a check."""
        report = blind.analyze(simulate(self.key, abx_accuracy=0.9, medium_bonus=12.0))
        self.assertIsNone(report["deviation"])
        self.assertNotIn("DEVIATION", report["conclusion"])

    def test_a_trimmed_panel_is_flagged_as_a_deviation_from_the_preregistration(self) -> None:
        """The garden-of-forking-paths hole the pre-registration exists to close.

        `analyze` recomputes the rejection threshold from the observed trial count, so dropping
        listeners moves it off the pre-registered 134/240. Silently recomputing is exactly what a
        post-hoc trim looks like, so the report must name it.
        """
        unblinded = simulate(self.key, abx_accuracy=0.9, medium_bonus=12.0)
        dropped = {"L01", "L02", "L03"}
        trimmed = dict(unblinded)
        trimmed["abx"] = [r for r in unblinded["abx"] if r["listener"] not in dropped]
        trimmed["ratings"] = [r for r in unblinded["ratings"] if r["listener"] not in dropped]

        report = blind.analyze(trimmed)
        self.assertIsNotNone(report["deviation"])
        self.assertIn("DEVIATION FROM PRE-REGISTRATION", report["conclusion"])
        joined = "; ".join(report["deviation"])
        self.assertIn("listeners", joined)
        self.assertIn("pooled contrast ABX trials", joined)
        self.assertIn("pooled null-control ABX trials", joined)
        # And the deviation is real, not cosmetic: the threshold genuinely moved.
        pre = unblinded["preregistration"]["abx"]
        self.assertNotEqual(report["abx"]["critical_count"], pre["critical_count"])

    def test_responses_carrying_no_preregistration_are_flagged_rather_than_trusted(self) -> None:
        unblinded = simulate(self.key, abx_accuracy=0.9, medium_bonus=12.0)
        stripped = {k: v for k, v in unblinded.items() if k != "preregistration"}
        report = blind.analyze(stripped)
        self.assertIsNotNone(report["deviation"])
        self.assertIn("DEVIATION FROM PRE-REGISTRATION", report["conclusion"])

    def test_the_key_embeds_the_preregistered_panel_size_not_the_playlist_count(self) -> None:
        """Generating spare playlists for attrition must not re-open the pre-registration."""
        _, key = blind.assign(fake_manifest(), panel=blind.RECRUIT_SIZE)
        self.assertEqual(len(key["listeners"]), blind.RECRUIT_SIZE)
        self.assertEqual(key["playlists_generated"], blind.RECRUIT_SIZE)
        self.assertEqual(key["panel_size"], blind.PANEL_SIZE)
        self.assertEqual(key["preregistration"]["panel_size"], blind.PANEL_SIZE)
        self.assertEqual(
            key["preregistration"]["abx"]["pooled_trials"],
            blind.PANEL_SIZE * blind.CONTRAST_TRIALS_PER_LISTENER,
        )

    def test_the_anchor_median_is_a_true_median_over_an_even_number_of_screens(self) -> None:
        """Six rating screens means the count is always even. The upper-middle value is not the
        median the protocol pre-registers, and it excludes listeners the rule does not."""
        self.assertEqual(blind.RATING_SCREENS_PER_LISTENER % 2, 0)
        unblinded = simulate(self.key, abx_accuracy=0.9, medium_bonus=0.0)
        # Lay down discrepancies whose true median straddles the threshold: half at 0, half well
        # above it. mean(0, 30) = 15 <= 20 retains; the upper-middle value 30 would exclude.
        rows = [r for r in unblinded["ratings"] if r["listener"] == "L01"]
        screens = sorted({r["screen"] for r in rows})
        for index, screen in enumerate(screens):
            here = [r for r in rows if r["screen"] == screen]
            duplicate = next(r for r in here if r["duplicate"])
            original = next(
                r for r in here if not r["duplicate"] and r["variant"] == duplicate["variant"]
            )
            offset = 0.0 if index < len(screens) // 2 else 30.0
            duplicate["rating"] = min(original["rating"] + offset, 100.0)
        report = blind.analyze(unblinded)
        self.assertNotIn(
            "L01",
            report["mos"]["listeners_excluded"],
            "the true median of {0,0,0,30,30,30} is 15, inside the 20-point rule",
        )

    def test_inconsistent_listeners_are_excluded_by_the_preregistered_rule(self) -> None:
        unblinded = simulate(self.key, abx_accuracy=0.9, medium_bonus=12.0)
        for row in unblinded["ratings"]:
            if row["listener"] == "L01" and row["duplicate"]:
                row["rating"] = min(row["rating"] + blind.MAX_ANCHOR_DISCREPANCY + 15.0, 100.0)
        report = blind.analyze(unblinded)
        self.assertIn("L01", report["mos"]["listeners_excluded"])
        self.assertEqual(report["mos"]["listeners_analyzed"], blind.PANEL_SIZE - 1)

    def _excluded_at_discrepancy(self, points: float) -> bool:
        """Give `L01` exactly `points` of duplicate-vs-original discrepancy on every screen."""
        unblinded = simulate(self.key, abx_accuracy=0.9, medium_bonus=12.0)
        rows = [r for r in unblinded["ratings"] if r["listener"] == "L01"]
        for screen in sorted({r["screen"] for r in rows}):
            here = [r for r in rows if r["screen"] == screen]
            duplicate = next(r for r in here if r["duplicate"])
            original = next(
                r for r in here if not r["duplicate"] and r["variant"] == duplicate["variant"]
            )
            # Move the *original* off the ceiling first so the offset is never clipped away.
            original["rating"] = 40.0
            duplicate["rating"] = 40.0 + points
        return "L01" in blind.analyze(unblinded)["mos"]["listeners_excluded"]

    def test_the_anchor_exclusion_threshold_is_the_pre_registered_twenty_points(self) -> None:
        """`MAX_ANCHOR_DISCREPANCY` is a pre-registered *post-hoc exclusion* threshold, quoted twice
        in §5 -- exactly the researcher degree of freedom §6 was written to close. Every other test
        that touches it phrases its fixture relative to the constant, so raising it to 25 left the
        whole suite green. This case is absolute: it flips at 20 and nowhere else."""
        self.assertFalse(
            self._excluded_at_discrepancy(18.0),
            "a listener whose median discrepancy is 18 points is inside the pre-registered rule and "
            "must be analyzed; the threshold has been tightened",
        )
        self.assertTrue(
            self._excluded_at_discrepancy(22.0),
            "a listener whose median discrepancy is 22 points exceeds the pre-registered 20-point "
            "rule and must be excluded; the threshold has been loosened",
        )
        # The pair above brackets the threshold; this pins the literal inside the bracket.
        self.assertEqual(blind.MAX_ANCHOR_DISCREPANCY, 20.0)

    def test_the_protocol_document_pins_the_anchor_exclusion_threshold(self) -> None:
        """The document quotes the threshold twice. Both are bound to their sentences here, because
        a bare `20` appears in this document as the panel size, a band width and a section number."""
        text = PROTOCOL.read_text(encoding="utf-8")
        points = f"{blind.MAX_ANCHOR_DISCREPANCY:g}"
        for needle in (
            f"discrepancy exceeds **{points} points**",
            f"(§5, {points} points)",
        ):
            self.assertIn(
                needle,
                text,
                f"protocol document does not pin the anchor-exclusion threshold: {needle!r}",
            )


class StatisticsTests(unittest.TestCase):
    def test_paired_t_matches_a_hand_computed_case(self) -> None:
        result = blind.paired_t_test([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(result["mean"], 2.5)
        self.assertAlmostEqual(result["sd"], math.sqrt(5.0 / 3.0))
        self.assertAlmostEqual(result["t"], 2.5 / (math.sqrt(5.0 / 3.0) / 2.0))

    def test_wilcoxon_matches_the_exact_enumeration_for_a_small_case(self) -> None:
        # n = 5, all positive: W = 0, and the two-sided exact p is 2/2^5.
        self.assertAlmostEqual(
            blind.wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0])["p"], 2.0 / 32.0
        )
        # A symmetric set cannot reject.
        self.assertGreater(blind.wilcoxon_signed_rank([1.0, -1.0, 2.0, -2.0])["p"], 0.5)

    def test_binomial_survival_is_exact_at_the_edges(self) -> None:
        self.assertAlmostEqual(blind.binomial_sf(0, 10, 0.5), 1.0)
        self.assertAlmostEqual(blind.binomial_sf(11, 10, 0.5), 0.0)
        self.assertAlmostEqual(blind.binomial_sf(10, 10, 0.5), 0.5**10)


if __name__ == "__main__":
    unittest.main()
