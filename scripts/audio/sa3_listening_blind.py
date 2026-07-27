#!/usr/bin/env python3
"""Blinding, scoring and the pre-registered analysis for the SA3 listening panel (sc-15178).

`docs/migration/SC_15178_SA3_LISTENING_PROTOCOL.md` is the protocol; this file is the executable
half of it. It owns four things that must not be decided per session:

1. **Randomized assignment** from a committed integer seed -- which take is presented as A, which
   as B, whether X duplicates A or B, and the trial order -- so two operators running the panel
   produce the same design and neither can nudge it.
2. **The blinded artefacts** -- a per-listener playlist that carries opaque handles only, and a
   private key held back until every response is collected.
3. **Unblinding**, which is a join and nothing else.
4. **The pre-registered analysis**, including the rule that makes the whole thing interpretable:
   ABX answers *discriminability* and MOS answers *preference*, and **a failed ABX makes the MOS
   run uninterpretable, so it is not reported**. That rule is enforced here rather than trusted to
   whoever runs the numbers. `analyze` also checks the counts that actually arrived against the
   pre-registration the key carries, and flags any difference as a `DEVIATION` -- its thresholds are
   recomputed from the observed counts, so a trimmed panel would otherwise be scored against a
   threshold that is quietly no longer the pre-registered one.

Everything is dependency-free (standard library only), matching the rest of `scripts/`. The
statistics are computed rather than tabulated: the exact binomial is `math.comb`, and the Student
and non-central t distributions are integrated numerically, so `power` reproduces every number the
protocol document commits to and `scripts/tests/test_sa3_listening_blind.py` re-derives them.

## What this deliberately does NOT do

It does not compute, and must never grow, an objective quality metric. CLAP, FAD, MR-STFT, SNR and
waveform cosine are *agreement* metrics with no valid reference across two post-trained checkpoints;
`crates/audio/candle-audio-stable-audio-3/tests/dtype_policy.rs` records why (same-seed F16-vs-F32
cosine 0.222 against 0.005 for two F32 renders at adjacent seeds). Substituting one for the panel
would be the exact false green sc-15178 exists to prevent.

## Usage

    # the committed pre-registration, recomputed
    sa3_listening_blind.py power

    # design the panel from the generator's manifest
    sa3_listening_blind.py assign --manifest stimuli/manifest.json \\
        --out-dir panel/ --source-dir stimuli/

    # blank response sheets
    sa3_listening_blind.py sheet --playlist panel/playlist_L01.json --out-dir panel/

    # after every response is collected
    sa3_listening_blind.py unblind --key panel/key.json \\
        --abx panel/abx_*.csv --ratings panel/ratings_*.csv --out panel/unblinded.json
    sa3_listening_blind.py analyze --unblinded panel/unblinded.json
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Pre-registration. Fixed before any listening; changing a value here after a
# run has started invalidates that run.
# ---------------------------------------------------------------------------

#: Listeners retained after screening. Set by the MOS half, which is the binding constraint --
#: the ABX half is over-powered at this size. The number comes from `preregistration()` below, not
#: from a standard; that it sits above the panel sizes ITU-R BS.1534 multi-stimulus tests are
#: typically run at is a sanity check on it, not its justification.
PANEL_SIZE = 20
#: Recruit this many to retain PANEL_SIZE after the screening and anchor-consistency exclusions.
RECRUIT_SIZE = 24

#: ABX contrast trials per listener: 6 stimuli x 2 seeds (the seeds are the within-listener
#: replicate).
CONTRAST_TRIALS_PER_LISTENER = 12
#: Same-checkpoint (identical-file) ABX trials per listener. A listener with no information scores
#: at chance on these by construction, so their pooled accuracy is the panel's validity check.
NULL_TRIALS_PER_LISTENER = 4

#: One-sided alpha for the ABX discriminability test. One-sided because below-chance ABX is not a
#: finding, it is noise.
ABX_ALPHA = 0.05
#: The null: a listener who cannot tell the checkpoints apart.
ABX_CHANCE = 0.5
#: The smallest pooled ABX accuracy worth acting on. Below this, "discriminable" would be a
#: statistically real difference too small to justify any product decision.
ABX_EFFECT_OF_INTEREST = 0.65

#: Rating screens per listener -- one per stimulus, at the first seed.
RATING_SCREENS_PER_LISTENER = 6
#: Takes presented per rating screen: medium, the specialist, and a hidden duplicate of one of them.
RATING_SLOTS_PER_SCREEN = 3
#: Continuous quality scale, ITU-R BS.1534 style, presented without a reference (there is no ground
#: truth for a text-to-audio render, so a true MUSHRA hidden reference does not exist here).
RATING_SCALE = (0, 100)
#: The preference difference worth acting on, in points: half the width of one labelled
#: band on the five-band 0-100 scale. Smaller than this is not worth a product decision.
MOS_EFFECT_POINTS = 10.0
#: Assumed standard deviation of the per-listener mean rating difference, in points. This is the
#: number most likely to be wrong; the protocol requires it to be re-estimated from the first run
#: and the pre-registration re-derived before any second run.
MOS_ASSUMED_SD = 15.0
#: Two-sided alpha for the preference test. Two-sided because medium being *worse* is a real and
#: reportable outcome.
MOS_ALPHA = 0.05

#: The power both halves are designed to reach at their stated effect size.
TARGET_POWER = 0.80

#: A listener whose hidden-duplicate ratings disagree by more than this many points (median over
#: their screens) is excluded before the preference analysis. Pre-registered, so it cannot be
#: tuned after seeing the result.
MAX_ANCHOR_DISCREPANCY = 20.0

#: The committed randomization seed. Assignment is a pure function of this plus the manifest.
DEFAULT_ASSIGNMENT_SEED = 15_178

#: The checkpoint under test. The specialist is whichever other variant the manifest carries for
#: that stimulus's domain.
MEDIUM_VARIANT = "stable_audio_3_medium"


def pooled_contrast_trials(panel: int = PANEL_SIZE) -> int:
    return panel * CONTRAST_TRIALS_PER_LISTENER


def pooled_null_trials(panel: int = PANEL_SIZE) -> int:
    return panel * NULL_TRIALS_PER_LISTENER


# ---------------------------------------------------------------------------
# Deterministic randomness.
# ---------------------------------------------------------------------------


_MASK64 = (1 << 64) - 1


class SplitMix64:
    """A committed, version-stable PRNG.

    The standard library's `random` would work, but its stream is an implementation detail of the
    interpreter; a protocol that pins "randomized from a committed seed" needs the assignment to be
    reproducible on a different Python. SplitMix64 is ten lines and fully specified, so a future
    reader can regenerate the exact panel design from the seed alone.
    """

    def __init__(self, seed: int) -> None:
        self.state = seed & _MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        return (z ^ (z >> 31)) & _MASK64

    def below(self, bound: int) -> int:
        """Uniform in `[0, bound)`, by rejection -- a modulo alone would bias low values."""
        if bound <= 0:
            raise ValueError("bound must be positive")
        limit = ((1 << 64) // bound) * bound
        while True:
            value = self.next_u64()
            if value < limit:
                return value % bound

    def shuffle(self, items: list) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]

    def coin(self) -> bool:
        return self.next_u64() & 1 == 1


# ---------------------------------------------------------------------------
# Statistics: exact binomial, Student t, non-central t. Standard library only.
# ---------------------------------------------------------------------------


def binomial_sf(k: int, n: int, p: float) -> float:
    """`P(X >= k)` for `X ~ Binomial(n, p)`, exactly."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(k, n + 1))


@functools.lru_cache(maxsize=None)
def binomial_critical(n: int, p0: float = ABX_CHANCE, alpha: float = ABX_ALPHA) -> int:
    """Smallest `k` with `P(X >= k | p0) <= alpha` -- the one-sided rejection threshold."""
    for k in range(n + 1):
        if binomial_sf(k, n, p0) <= alpha:
            return k
    return n + 1


def binomial_power(n: int, p1: float, p0: float = ABX_CHANCE, alpha: float = ABX_ALPHA) -> float:
    return binomial_sf(binomial_critical(n, p0, alpha), n, p1)


@functools.lru_cache(maxsize=None)
def binomial_mde(n: int, target: float = TARGET_POWER, p0: float = ABX_CHANCE,
                 alpha: float = ABX_ALPHA) -> float:
    """Smallest true accuracy (to 0.001) this many trials detects at `target` power."""
    step = 1000
    for numerator in range(int(p0 * step) + 1, step + 1):
        p1 = numerator / step
        if binomial_power(n, p1, p0, alpha) >= target:
            return p1
    return 1.0


@functools.lru_cache(maxsize=None)
def binomial_acceptance(n: int, p0: float = ABX_CHANCE, alpha: float = 0.05) -> tuple[int, int]:
    """Central `1 - alpha` acceptance interval on the count, for the same-checkpoint control.

    The control's purpose is inverted from the contrast test: a *pass* is landing inside this band,
    and the band is returned **inclusive at both ends**. A panel scoring above it on identical files
    is not a panel that heard something, it is a panel whose blinding leaked.

    Both bounds are the extreme *accepted* counts, not the extreme rejected ones. `low` is the
    smallest `k` with `P(X <= k - 1) <= alpha/2`, so `k - 1` is already in the lower rejection
    region. `high` is derived the same way from the upper tail and then stepped back by one: the
    `min` finds the smallest count in the *rejection* region, and returning it unchanged would widen
    the band by one count on the upper side only. That asymmetry is in the leak direction -- the one
    thing this control exists to catch -- so it is corrected here rather than left as slack.
    """
    tail = alpha / 2.0
    low = max(k for k in range(n + 1) if binomial_sf(k, n, p0) >= 1.0 - tail)
    # `default` covers the small-`n` case where even `n` correct is not significant at `tail`: the
    # whole range is then acceptance, and `min` over an empty sequence would raise instead.
    rejects_above = min((k for k in range(n + 1) if binomial_sf(k, n, p0) <= tail), default=n + 1)
    return low, rejects_above - 1


def _simpson(function, lo: float, hi: float, intervals: int = 4000) -> float:
    if intervals % 2:
        intervals += 1
    step = (hi - lo) / intervals
    total = function(lo) + function(hi)
    for i in range(1, intervals):
        total += function(lo + i * step) * (4 if i % 2 else 2)
    return total * step / 3.0


def _student_pdf(x: float, df: int) -> float:
    coefficient = math.exp(math.lgamma((df + 1) / 2.0) - math.lgamma(df / 2.0))
    coefficient /= math.sqrt(df * math.pi)
    return coefficient * (1.0 + x * x / df) ** (-(df + 1) / 2.0)


def student_sf(t: float, df: int) -> float:
    """`P(T > t)` for a central Student t, by symmetry plus numeric integration of the tail."""
    if t < 0:
        return 1.0 - student_sf(-t, df)
    # The tail beyond 60 sd-equivalents contributes below 1e-12 for every df used here.
    return _simpson(lambda x: _student_pdf(x, df), t, max(t + 1.0, 60.0), 6000)


@functools.lru_cache(maxsize=None)
def student_critical(df: int, alpha: float = MOS_ALPHA) -> float:
    """Two-sided critical value: `t` with `P(|T| > t) = alpha`."""
    lo, hi = 0.0, 60.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if student_sf(mid, df) > alpha / 2.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@functools.lru_cache(maxsize=None)
def paired_t_power(n: int, effect_dz: float, alpha: float = MOS_ALPHA) -> float:
    """Power of a two-sided one-sample (paired) t-test at standardized effect `effect_dz`.

    Computed from the non-central t directly: with `S = sqrt(chi2_df / df)`,
    `T = (Z + delta) / S`, so `P(T > t) = E_S[1 - Phi(t*S - delta)]`. Integrating over the density
    of `S` needs only `lgamma` and `erf`. The lower tail is included -- it is negligible at these
    effect sizes but omitting it would silently overstate a two-sided test as one-sided.
    """
    df = n - 1
    delta = effect_dz * math.sqrt(n)
    critical = student_critical(df, alpha)

    def density_s(s: float) -> float:
        if s <= 0.0:
            return 0.0
        w = df * s * s
        log_pdf = (df / 2.0 - 1.0) * math.log(w) - w / 2.0
        log_pdf -= (df / 2.0) * math.log(2.0) + math.lgamma(df / 2.0)
        return math.exp(log_pdf) * 2.0 * df * s

    upper = _simpson(lambda s: (1.0 - _normal_cdf(critical * s - delta)) * density_s(s), 1e-6, 5.0)
    lower = _simpson(lambda s: _normal_cdf(-critical * s - delta) * density_s(s), 1e-6, 5.0)
    return upper + lower


@functools.lru_cache(maxsize=None)
def paired_t_mde(n: int, target: float = TARGET_POWER, alpha: float = MOS_ALPHA) -> float:
    """Smallest standardized effect this many listeners detect at `target` power."""
    lo, hi = 0.0, 3.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if paired_t_power(n, mid, alpha) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def paired_t_test(differences: list[float]) -> dict:
    n = len(differences)
    if n < 2:
        raise ValueError("a paired t-test needs at least two listeners")
    mean = sum(differences) / n
    variance = sum((d - mean) ** 2 for d in differences) / (n - 1)
    sd = math.sqrt(variance)
    if sd == 0.0:
        t = math.inf if mean != 0.0 else 0.0
    else:
        t = mean / (sd / math.sqrt(n))
    p = 1.0 if t == 0.0 else 2.0 * student_sf(abs(t), n - 1)
    return {"n": n, "mean": mean, "sd": sd, "t": t, "df": n - 1, "p": min(p, 1.0)}


def wilcoxon_signed_rank(differences: list[float]) -> dict:
    """Exact two-sided signed-rank test, zeros dropped. The pre-registered robustness check.

    Exact rather than normal-approximated: the panel is 20 listeners, where the approximation is
    adequate but the exact enumeration is cheap and removes an argument about which to use.
    """
    values = [d for d in differences if d != 0.0]
    n = len(values)
    if n == 0:
        return {"n": 0, "w": 0.0, "p": 1.0}
    order = sorted(range(n), key=lambda i: abs(values[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(values[order[j + 1]]) == abs(values[order[i]]):
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    positive = sum(rank for rank, value in zip(ranks, values) if value > 0)
    total = sum(ranks)
    statistic = min(positive, total - positive)

    # Exact null: every sign assignment is equally likely.
    counts = {0.0: 1}
    for rank in ranks:
        updated: dict[float, int] = {}
        for value, count in counts.items():
            updated[value] = updated.get(value, 0) + count
            updated[value + rank] = updated.get(value + rank, 0) + count
        counts = updated
    space = 2**n
    at_or_below = sum(count for value, count in counts.items() if value <= statistic + 1e-9)
    return {"n": n, "w": statistic, "p": min(1.0, 2.0 * at_or_below / space)}


# ---------------------------------------------------------------------------
# Pre-registration report.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _preregistration_cached(panel: int) -> str:
    return json.dumps(_preregistration(panel), sort_keys=True)


def preregistration(panel: int = PANEL_SIZE) -> dict:
    """Cached wrapper -- the power integrals are deterministic and re-derived on every call."""
    return json.loads(_preregistration_cached(panel))


def _preregistration(panel: int) -> dict:
    """Every number the protocol document commits to, recomputed from the constants above."""
    contrast_n = pooled_contrast_trials(panel)
    null_n = pooled_null_trials(panel)
    low, high = binomial_acceptance(null_n)
    effect_dz = MOS_EFFECT_POINTS / MOS_ASSUMED_SD
    return {
        "panel_size": panel,
        "recruit_size": RECRUIT_SIZE,
        "assignment_seed": DEFAULT_ASSIGNMENT_SEED,
        "abx": {
            "trials_per_listener": CONTRAST_TRIALS_PER_LISTENER,
            "pooled_trials": contrast_n,
            "alpha_one_sided": ABX_ALPHA,
            "critical_count": binomial_critical(contrast_n),
            "critical_accuracy": binomial_critical(contrast_n) / contrast_n,
            "effect_of_interest": ABX_EFFECT_OF_INTEREST,
            "power_at_effect_of_interest": binomial_power(contrast_n, ABX_EFFECT_OF_INTEREST),
            "minimum_detectable_accuracy": binomial_mde(contrast_n),
        },
        "null_control": {
            "trials_per_listener": NULL_TRIALS_PER_LISTENER,
            "pooled_trials": null_n,
            "expected_accuracy": ABX_CHANCE,
            "acceptance_counts": [low, high],
            "acceptance_accuracy": [low / null_n, high / null_n],
        },
        "mos": {
            "screens_per_listener": RATING_SCREENS_PER_LISTENER,
            "slots_per_screen": RATING_SLOTS_PER_SCREEN,
            "scale": list(RATING_SCALE),
            "alpha_two_sided": MOS_ALPHA,
            "effect_points": MOS_EFFECT_POINTS,
            "assumed_sd_points": MOS_ASSUMED_SD,
            "effect_dz": effect_dz,
            "power_at_effect": paired_t_power(panel, effect_dz),
            "minimum_detectable_dz": paired_t_mde(panel),
            "minimum_detectable_points": paired_t_mde(panel) * MOS_ASSUMED_SD,
            "max_anchor_discrepancy": MAX_ANCHOR_DISCREPANCY,
        },
        "target_power": TARGET_POWER,
    }


# ---------------------------------------------------------------------------
# Assignment.
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def index_takes(manifest: dict) -> dict:
    """Group manifest takes by `(stimulus, seed)` -> `{variant: entry}`."""
    grouped: dict[tuple[str, int], dict[str, dict]] = {}
    for take in manifest["takes"]:
        grouped.setdefault((take["stimulus"], take["seed"]), {})[take["variant"]] = take
    return grouped


def listener_ids(panel: int) -> list[str]:
    return [f"L{index + 1:02d}" for index in range(panel)]


def assign(manifest: dict, seed: int = DEFAULT_ASSIGNMENT_SEED,
           panel: int = PANEL_SIZE) -> tuple[dict, dict]:
    """Build the blinded playlists and the private key.

    Returns `(playlists, key)`. `playlists` carries opaque handles only -- no variant name, no
    prompt, no seed -- because a listener who can see which take is medium is not blinded. The key
    carries the full mapping and is withheld until responses are collected.

    `panel` is **how many playlists to generate**, which is not the same number as the pre-registered
    panel. The runbook generates `RECRUIT_SIZE` of them so there is headroom for the attrition the
    protocol states; the pre-registration embedded in the key is always the `PANEL_SIZE` one, because
    it was fixed before any listening and generating spare playlists does not re-open it. Each
    listener's design comes from its own stream (`seed * 1_000_003 + position`), so the spare
    playlists are stable and independent -- a replacement listener does not perturb anyone else's.
    """
    grouped = index_takes(manifest)
    stimuli = sorted({stimulus for stimulus, _ in grouped})
    seeds = sorted({take_seed for _, take_seed in grouped})
    expected = len(stimuli) * len(seeds)
    if expected != CONTRAST_TRIALS_PER_LISTENER:
        raise ValueError(
            f"manifest yields {expected} contrast trials per listener "
            f"({len(stimuli)} stimuli x {len(seeds)} seeds); the pre-registration fixes "
            f"{CONTRAST_TRIALS_PER_LISTENER}"
        )

    playlists: dict[str, dict] = {}
    key: dict = {
        "story": "sc-15178",
        "assignment_seed": seed,
        "playlists_generated": panel,
        "panel_size": PANEL_SIZE,
        "preregistration": preregistration(),
        "listeners": {},
    }

    for position, listener in enumerate(listener_ids(panel)):
        # Per-listener stream, derived from the committed seed: one listener's design cannot be
        # inferred from another's, and re-running for a single replacement listener is stable.
        rng = SplitMix64(seed * 1_000_003 + position)
        trials = []

        for stimulus in stimuli:
            for take_seed in seeds:
                by_variant = grouped[(stimulus, take_seed)]
                specialist = next(v for v in by_variant if v != MEDIUM_VARIANT)
                medium_is_a = rng.coin()
                a_variant = MEDIUM_VARIANT if medium_is_a else specialist
                b_variant = specialist if medium_is_a else MEDIUM_VARIANT
                x_is_a = rng.coin()
                trials.append({
                    "kind": "contrast",
                    "stimulus": stimulus,
                    "domain": by_variant[MEDIUM_VARIANT]["domain"],
                    "seed": take_seed,
                    "a": by_variant[a_variant]["file"],
                    "b": by_variant[b_variant]["file"],
                    "a_variant": a_variant,
                    "b_variant": b_variant,
                    "x_is": "a" if x_is_a else "b",
                })

        # Same-checkpoint null trials: A, B and X are the *same* file, so a listener has no
        # information and scores at chance by construction. There is no honest alternative -- a
        # same-checkpoint pair at different seeds is different audio and therefore genuinely
        # discriminable, which would make the control measure the wrong thing.
        null_pool = list(stimuli)
        rng.shuffle(null_pool)
        for stimulus in null_pool[:NULL_TRIALS_PER_LISTENER]:
            take_seed = seeds[rng.below(len(seeds))]
            entry = grouped[(stimulus, take_seed)][MEDIUM_VARIANT]
            trials.append({
                "kind": "null",
                "stimulus": stimulus,
                "domain": entry["domain"],
                "seed": take_seed,
                "a": entry["file"],
                "b": entry["file"],
                "a_variant": MEDIUM_VARIANT,
                "b_variant": MEDIUM_VARIANT,
                "x_is": "a" if rng.coin() else "b",
            })

        rng.shuffle(trials)
        for number, trial in enumerate(trials, 1):
            trial["trial"] = f"t{number:02d}"

        # Rating screens: one per stimulus at the first seed, three slots -- medium, the
        # specialist, and a hidden duplicate of one of them as the per-listener consistency anchor.
        rating_seed = seeds[0]
        screens = []
        screen_stimuli = list(stimuli)
        rng.shuffle(screen_stimuli)
        for number, stimulus in enumerate(screen_stimuli[:RATING_SCREENS_PER_LISTENER], 1):
            by_variant = grouped[(stimulus, rating_seed)]
            specialist = next(v for v in by_variant if v != MEDIUM_VARIANT)
            duplicated = MEDIUM_VARIANT if rng.coin() else specialist
            slots = [
                {"variant": MEDIUM_VARIANT, "file": by_variant[MEDIUM_VARIANT]["file"],
                 "duplicate": False},
                {"variant": specialist, "file": by_variant[specialist]["file"],
                 "duplicate": False},
                {"variant": duplicated, "file": by_variant[duplicated]["file"],
                 "duplicate": True},
            ]
            rng.shuffle(slots)
            screens.append({
                "screen": f"s{number:02d}",
                "stimulus": stimulus,
                "domain": by_variant[MEDIUM_VARIANT]["domain"],
                "seed": rating_seed,
                "slots": slots,
            })

        playlists[listener] = {
            "listener": listener,
            "abx": [
                {
                    "trial": trial["trial"],
                    "a": f"{listener}-{trial['trial']}-a.wav",
                    "b": f"{listener}-{trial['trial']}-b.wav",
                    "x": f"{listener}-{trial['trial']}-x.wav",
                }
                for trial in trials
            ],
            "ratings": [
                {
                    "screen": screen["screen"],
                    "slots": [
                        f"{listener}-{screen['screen']}-{index + 1}.wav"
                        for index in range(RATING_SLOTS_PER_SCREEN)
                    ],
                }
                for screen in screens
            ],
        }
        key["listeners"][listener] = {"abx": trials, "ratings": screens}

    return playlists, key


def materialize(playlists: dict, key: dict, source_dir: Path, out_dir: Path) -> int:
    """Copy each take to its opaque per-listener filename.

    Copies rather than symlinks: a symlink's target is the variant-bearing filename, and an
    operator inspecting the presentation directory would be unblinded by `ls -l`.
    """
    written = 0
    for listener, playlist in playlists.items():
        listener_dir = out_dir / listener
        listener_dir.mkdir(parents=True, exist_ok=True)
        design = key["listeners"][listener]
        for blind, trial in zip(playlist["abx"], design["abx"]):
            source = {"a": trial["a"], "b": trial["b"], "x": trial[trial["x_is"]]}
            for slot in ("a", "b", "x"):
                shutil.copyfile(source_dir / source[slot], listener_dir / blind[slot])
                written += 1
        for blind, screen in zip(playlist["ratings"], design["ratings"]):
            for name, slot in zip(blind["slots"], screen["slots"]):
                shutil.copyfile(source_dir / slot["file"], listener_dir / name)
                written += 1
    return written


def blinded_leaks(playlists: dict, secrets: list[str]) -> list[str]:
    """Any occurrence of a secret token in the blinded artefacts. Should always be empty."""
    text = json.dumps(playlists)
    return [secret for secret in secrets if secret and secret in text]


# ---------------------------------------------------------------------------
# Response sheets and unblinding.
# ---------------------------------------------------------------------------


ABX_COLUMNS = ("listener", "trial", "x_is")
RATING_COLUMNS = ("listener", "screen", "slot", "rating")


def write_sheets(playlist: dict, out_dir: Path) -> tuple[Path, Path]:
    listener = playlist["listener"]
    abx_path = out_dir / f"abx_{listener}.csv"
    rating_path = out_dir / f"ratings_{listener}.csv"
    with abx_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ABX_COLUMNS)
        for trial in playlist["abx"]:
            writer.writerow([listener, trial["trial"], ""])
    with rating_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(RATING_COLUMNS)
        for screen in playlist["ratings"]:
            for slot in range(1, RATING_SLOTS_PER_SCREEN + 1):
                writer.writerow([listener, screen["screen"], slot, ""])
    return abx_path, rating_path


def read_csv(path: Path, columns: tuple[str, ...]) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        missing = [column for column in columns if column not in row]
        if missing:
            raise ValueError(f"{path}: missing column(s) {missing}")
    return rows


def unblind(key: dict, abx_rows: list[dict], rating_rows: list[dict]) -> dict:
    """Join responses onto the key. Scoring only -- no analysis decision is taken here."""
    abx: list[dict] = []
    for row in abx_rows:
        listener = row["listener"].strip()
        design = key["listeners"].get(listener)
        if design is None:
            raise ValueError(f"response for unknown listener {listener!r}")
        trial = next((t for t in design["abx"] if t["trial"] == row["trial"].strip()), None)
        if trial is None:
            raise ValueError(f"{listener}: unknown trial {row['trial']!r}")
        response = row["x_is"].strip().lower()
        if response not in {"a", "b"}:
            raise ValueError(f"{listener}/{trial['trial']}: x_is must be 'a' or 'b', got {response!r}")
        abx.append({
            "listener": listener,
            "trial": trial["trial"],
            "kind": trial["kind"],
            "stimulus": trial["stimulus"],
            "domain": trial["domain"],
            "seed": trial["seed"],
            "response": response,
            "truth": trial["x_is"],
            "correct": response == trial["x_is"],
        })

    ratings: list[dict] = []
    for row in rating_rows:
        listener = row["listener"].strip()
        design = key["listeners"].get(listener)
        if design is None:
            raise ValueError(f"rating for unknown listener {listener!r}")
        screen = next((s for s in design["ratings"] if s["screen"] == row["screen"].strip()), None)
        if screen is None:
            raise ValueError(f"{listener}: unknown rating screen {row['screen']!r}")
        slot_index = int(row["slot"]) - 1
        if not 0 <= slot_index < RATING_SLOTS_PER_SCREEN:
            raise ValueError(f"{listener}/{screen['screen']}: slot out of range")
        value = float(row["rating"])
        if not RATING_SCALE[0] <= value <= RATING_SCALE[1]:
            raise ValueError(f"{listener}/{screen['screen']}: rating {value} outside {RATING_SCALE}")
        slot = screen["slots"][slot_index]
        ratings.append({
            "listener": listener,
            "screen": screen["screen"],
            "stimulus": screen["stimulus"],
            "domain": screen["domain"],
            "slot": slot_index + 1,
            "variant": slot["variant"],
            "duplicate": slot["duplicate"],
            "rating": value,
        })

    return {
        "story": "sc-15178",
        "assignment_seed": key["assignment_seed"],
        "preregistration": key["preregistration"],
        "abx": abx,
        "ratings": ratings,
    }


# ---------------------------------------------------------------------------
# The pre-registered analysis.
# ---------------------------------------------------------------------------


def preregistration_deviations(unblinded: dict) -> list[str]:
    """Observed counts against the pre-registration the unblinded responses carry.

    This exists because every threshold below is recomputed from the **observed** trial count. That
    is the right thing to do arithmetically and the wrong thing to trust silently: drop trials or
    listeners and the rejection threshold moves with them, so an incomplete or post-hoc-trimmed
    panel would be scored against a threshold that is no longer the pre-registered 134/240 -- with
    nothing in the report saying so. Pre-registration exists to close exactly that forking path, and
    a file carrying the pre-registration is the place to enforce it.

    Returns one string per difference, empty when the panel arrived as designed.
    """
    pre = unblinded.get("preregistration")
    if not pre:
        return ["the unblinded responses carry no pre-registration to check against"]

    observed_contrast = sum(1 for row in unblinded["abx"] if row["kind"] == "contrast")
    observed_null = sum(1 for row in unblinded["abx"] if row["kind"] == "null")
    observed_listeners = len({row["listener"] for row in unblinded["abx"]})

    deviations = []
    for label, observed, expected in (
        ("pooled contrast ABX trials", observed_contrast, pre["abx"]["pooled_trials"]),
        ("pooled null-control ABX trials", observed_null, pre["null_control"]["pooled_trials"]),
        ("listeners", observed_listeners, pre["panel_size"]),
    ):
        if observed != expected:
            deviations.append(f"{label}: {observed} observed, {expected} pre-registered")
    return deviations


def analyze(unblinded: dict) -> dict:
    """Run the pre-registered analysis and check the panel against the pre-registration.

    The analysis itself is `_pre_registered_analysis`; this wrapper adds the enforcement half. A
    count that does not match what was pre-registered is reported in an explicit `deviation` field
    and prepended to the conclusion, so a trimmed panel cannot be read as a pre-registered result.
    The numbers are still computed -- refusing outright would hide a partial run that is worth
    looking at -- but they are labelled as no longer pre-registered, and that label is not
    overridable.
    """
    report = _pre_registered_analysis(unblinded)
    deviations = preregistration_deviations(unblinded)
    report["deviation"] = deviations or None
    if deviations:
        report["conclusion"] = (
            "DEVIATION FROM PRE-REGISTRATION: "
            + "; ".join(deviations)
            + ". Every threshold below was recomputed from the observed counts, so they are NOT "
            "the pre-registered ones and this is no longer a pre-registered analysis. Report the "
            "deviation with the result. --- " + report["conclusion"]
        )
    return report


def _pre_registered_analysis(unblinded: dict) -> dict:
    """Run the pre-registered analysis, in the pre-registered order.

    The order is the point:

    1. **Panel validity** on the same-checkpoint trials. Identical files carry no information, so
       an above-band score means the blinding leaked, not that listeners heard something. Nothing
       downstream is interpretable if this fails.
    2. **ABX discriminability**, one-sided exact binomial against chance. This is a *gate*, not a
       result to be traded off against the next step.
    3. **Preference**, and only if step 2 rejected chance. If listeners cannot tell the checkpoints
       apart, a preference between them is a preference between two things they cannot
       distinguish, and reporting it would manufacture a finding out of noise. This function
       returns `mos: None` with a reason in that case, and the reason is not overridable.
    """
    contrast = [row for row in unblinded["abx"] if row["kind"] == "contrast"]
    nulls = [row for row in unblinded["abx"] if row["kind"] == "null"]

    report: dict = {"validity": None, "abx": None, "mos": None, "conclusion": ""}

    null_n = len(nulls)
    null_correct = sum(1 for row in nulls if row["correct"])
    if null_n:
        low, high = binomial_acceptance(null_n)
        valid = low <= null_correct <= high
    else:
        low = high = 0
        valid = False
    report["validity"] = {
        "trials": null_n,
        "correct": null_correct,
        "accuracy": null_correct / null_n if null_n else 0.0,
        "acceptance_counts": [low, high],
        "passed": valid,
    }
    if not valid:
        report["conclusion"] = (
            "PANEL INVALID: the same-checkpoint control scored "
            f"{null_correct}/{null_n}, outside the [{low}, {high}] chance band. Identical files "
            "carry no information, so this is evidence about the presentation, not about the "
            "checkpoints. Neither the ABX nor the preference result is reportable."
        )
        return report

    contrast_n = len(contrast)
    contrast_correct = sum(1 for row in contrast if row["correct"])
    critical = binomial_critical(contrast_n)
    p_value = binomial_sf(contrast_correct, contrast_n, ABX_CHANCE)
    discriminable = contrast_correct >= critical
    report["abx"] = {
        "trials": contrast_n,
        "correct": contrast_correct,
        "accuracy": contrast_correct / contrast_n if contrast_n else 0.0,
        "critical_count": critical,
        "alpha_one_sided": ABX_ALPHA,
        "p_value": p_value,
        "discriminable": discriminable,
        "minimum_detectable_accuracy": binomial_mde(contrast_n),
        "by_domain": {
            domain: {
                "trials": sum(1 for row in contrast if row["domain"] == domain),
                "correct": sum(1 for row in contrast if row["domain"] == domain and row["correct"]),
            }
            for domain in sorted({row["domain"] for row in contrast})
        },
    }

    if not discriminable:
        report["conclusion"] = (
            f"NULL RESULT: pooled ABX accuracy {contrast_correct}/{contrast_n} "
            f"({contrast_correct / max(contrast_n, 1):.3f}), one-sided exact binomial "
            f"p = {p_value:.4f} against chance, below the {critical}/{contrast_n} rejection "
            f"threshold. The panel could not distinguish the two checkpoints at the "
            f"pre-registered power. The preference ratings are NOT reported: a preference between "
            "two takes a listener cannot tell apart is not a preference. This is a legitimate and "
            "expected outcome for two checkpoints of one family, and it retires the 'audibly "
            "higher quality' wording rather than substantiating it."
        )
        return report

    # Anchor consistency: a listener who rates the hidden duplicate far from its original is not
    # rating reliably. The rule and its threshold were fixed before any data existed.
    by_listener: dict[str, list[dict]] = {}
    for row in unblinded["ratings"]:
        by_listener.setdefault(row["listener"], []).append(row)

    excluded: list[str] = []
    differences: list[float] = []
    for listener, rows in sorted(by_listener.items()):
        discrepancies = []
        for screen in sorted({row["screen"] for row in rows}):
            here = [row for row in rows if row["screen"] == screen]
            duplicate = next((row for row in here if row["duplicate"]), None)
            if duplicate is None:
                continue
            original = next(
                (row for row in here
                 if not row["duplicate"] and row["variant"] == duplicate["variant"]),
                None,
            )
            if original is not None:
                discrepancies.append(abs(duplicate["rating"] - original["rating"]))
        # A true median, not the upper-middle value: there are six rating screens, so the count is
        # even and picking `sorted(...)[n // 2]` would silently pre-register one thing and compute
        # another -- in the exclusion-happy direction.
        if discrepancies:
            ordered = sorted(discrepancies)
            middle = len(ordered) // 2
            median = (
                ordered[middle]
                if len(ordered) % 2
                else (ordered[middle - 1] + ordered[middle]) / 2.0
            )
        else:
            median = 0.0
        if median > MAX_ANCHOR_DISCREPANCY:
            excluded.append(listener)
            continue
        # The duplicate slot is excluded from the preference contrast: averaging it in would weight
        # whichever checkpoint happened to be duplicated.
        scored = [row for row in rows if not row["duplicate"]]
        medium = [row["rating"] for row in scored if row["variant"] == MEDIUM_VARIANT]
        specialist = [row["rating"] for row in scored if row["variant"] != MEDIUM_VARIANT]
        if medium and specialist:
            differences.append(sum(medium) / len(medium) - sum(specialist) / len(specialist))

    if len(differences) < 2:
        report["conclusion"] = (
            "ABX rejected chance, but too few listeners survived the anchor-consistency "
            "exclusion to run the preference test."
        )
        return report

    parametric = paired_t_test(differences)
    robust = wilcoxon_signed_rank(differences)
    favours_medium = parametric["mean"] > 0
    significant = parametric["p"] < MOS_ALPHA
    report["mos"] = {
        "listeners_analyzed": len(differences),
        "listeners_excluded": excluded,
        "mean_difference_points": parametric["mean"],
        "sd_points": parametric["sd"],
        "t": parametric["t"],
        "df": parametric["df"],
        "p_value": parametric["p"],
        "alpha_two_sided": MOS_ALPHA,
        "wilcoxon": robust,
        "significant": significant,
        "favours_medium": favours_medium,
        "effect_points_of_interest": MOS_EFFECT_POINTS,
        "power_at_effect": paired_t_power(len(differences), MOS_EFFECT_POINTS / MOS_ASSUMED_SD),
    }

    if not significant:
        report["conclusion"] = (
            f"DISCRIMINABLE BUT NO PREFERENCE: ABX rejected chance "
            f"(p = {p_value:.4f}), so listeners can tell the checkpoints apart, but the rating "
            f"difference is {parametric['mean']:+.2f} points "
            f"(p = {parametric['p']:.4f}, n = {len(differences)}). Medium is different, not "
            "better. The 'audibly higher quality' wording is not substantiated."
        )
    elif favours_medium:
        report["conclusion"] = (
            f"PREFERENCE FOR MEDIUM: ABX p = {p_value:.4f}; rating difference "
            f"{parametric['mean']:+.2f} points (p = {parametric['p']:.4f}, n = "
            f"{len(differences)}). Compare against the {MOS_EFFECT_POINTS}-point effect the panel "
            "was sized for before deciding this justifies a product claim."
        )
    else:
        report["conclusion"] = (
            f"PREFERENCE AGAINST MEDIUM: ABX p = {p_value:.4f}; rating difference "
            f"{parametric['mean']:+.2f} points (p = {parametric['p']:.4f}, n = "
            f"{len(differences)}). Listeners preferred the specialist. Report this as-is."
        )
    return report


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("power", help="print the committed pre-registration, recomputed")

    assign_parser = sub.add_parser("assign", help="build blinded playlists and the private key")
    assign_parser.add_argument("--manifest", type=Path, required=True)
    assign_parser.add_argument("--out-dir", type=Path, required=True)
    assign_parser.add_argument("--source-dir", type=Path,
                               help="stimulus WAV directory; copies takes to opaque names")
    assign_parser.add_argument("--seed", type=int, default=DEFAULT_ASSIGNMENT_SEED)
    assign_parser.add_argument(
        "--panel-size", type=int, default=RECRUIT_SIZE,
        help=(f"how many playlists to generate; defaults to RECRUIT_SIZE ({RECRUIT_SIZE}) so there "
              f"is headroom for attrition above the {PANEL_SIZE} listeners the pre-registration is "
              "fixed at. The key's pre-registration is always the PANEL_SIZE one."))

    sheet_parser = sub.add_parser("sheet", help="emit blank response sheets for one playlist")
    sheet_parser.add_argument("--playlist", type=Path, required=True)
    sheet_parser.add_argument("--out-dir", type=Path, required=True)

    unblind_parser = sub.add_parser("unblind", help="join responses onto the key")
    unblind_parser.add_argument("--key", type=Path, required=True)
    unblind_parser.add_argument("--abx", type=Path, nargs="+", required=True)
    unblind_parser.add_argument("--ratings", type=Path, nargs="*", default=[])
    unblind_parser.add_argument("--out", type=Path, required=True)

    analyze_parser = sub.add_parser("analyze", help="run the pre-registered analysis")
    analyze_parser.add_argument("--unblinded", type=Path, required=True)
    analyze_parser.add_argument("--out", type=Path)

    args = parser.parse_args(argv)

    if args.command == "power":
        print(json.dumps(preregistration(), indent=2, sort_keys=True))
        return 0

    if args.command == "assign":
        manifest = load_manifest(args.manifest)
        playlists, key = assign(manifest, args.seed, args.panel_size)
        secrets = sorted({take["variant"] for take in manifest["takes"]}
                         | {take["prompt"] for take in manifest["takes"]}
                         | {take["file"] for take in manifest["takes"]})
        leaked = blinded_leaks(playlists, secrets)
        if leaked:
            raise SystemExit(f"blinded playlists leak {leaked}; refusing to write")
        for listener, playlist in playlists.items():
            _write_json(args.out_dir / f"playlist_{listener}.json", playlist)
        _write_json(args.out_dir / "key.json", key)
        if args.source_dir is not None:
            copied = materialize(playlists, key, args.source_dir, args.out_dir)
            print(f"materialized {copied} blinded takes under {args.out_dir}")
        print(f"wrote {len(playlists)} playlists and key.json to {args.out_dir}")
        print("KEEP key.json AWAY FROM LISTENERS AND OPERATORS UNTIL EVERY RESPONSE IS COLLECTED.")
        return 0

    if args.command == "sheet":
        playlist = json.loads(args.playlist.read_text(encoding="utf-8"))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for path in write_sheets(playlist, args.out_dir):
            print(path)
        return 0

    if args.command == "unblind":
        key = json.loads(args.key.read_text(encoding="utf-8"))
        abx_rows = [row for path in args.abx for row in read_csv(path, ABX_COLUMNS)]
        rating_rows = [row for path in args.ratings for row in read_csv(path, RATING_COLUMNS)]
        _write_json(args.out, unblind(key, abx_rows, rating_rows))
        print(f"unblinded {len(abx_rows)} ABX responses and {len(rating_rows)} ratings -> {args.out}")
        return 0

    if args.command == "analyze":
        unblinded = json.loads(args.unblinded.read_text(encoding="utf-8"))
        report = analyze(unblinded)
        if args.out is not None:
            _write_json(args.out, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        print()
        print(report["conclusion"])
        return 0

    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
