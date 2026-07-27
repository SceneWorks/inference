# SA3 medium-vs-small listening protocol (sc-15178)

A pinned, blinded ABX + preference protocol for the one question about Stable Audio 3 that no test
in this repository can answer: **does `stable_audio_3_medium` sound better than the small
specialists?**

This document is the protocol. Two artefacts execute it:

- [`crates/audio/candle-audio-stable-audio-3/tests/listening_stimuli.rs`](../../crates/audio/candle-audio-stable-audio-3/tests/listening_stimuli.rs)
  — the reproducible, level-matched stimulus generator.
- [`scripts/audio/sa3_listening_blind.py`](../../scripts/audio/sa3_listening_blind.py) — randomized
  assignment, the blinded response sheets, unblinding, and the pre-registered analysis.

**Status: designed, not run.** Executing it needs a panel of human listeners and is tracked as
**sc-15377**. Nothing in this repository currently claims medium is perceptually better, and nothing
should until that story reports a result — including the possibility that it reports a null.

---

## 1. Why a listening panel, and not a metric

The tempting shortcut is an objective number in a `cargo test`. It does not work here, and the
evidence is already committed.

**Agreement metrics have no valid reference across two post-trained checkpoints.** MR-STFT, SNR,
waveform cosine and their relatives measure how closely a candidate reproduces a *reference*. Two
different checkpoints rendering the same prompt are supposed to disagree; there is no reference for
either to be scored against.

**The sampler makes the disagreement structural, not incremental.** The registered sampler is
eight-step Pingpong, which re-injects noise at every step, so any perturbation selects a *different
trajectory* rather than nudging one.
[`tests/dtype_policy.rs`](../../crates/audio/candle-audio-stable-audio-3/tests/dtype_policy.rs)
records the measurement: medium's F16 render sits at waveform cosine **0.222** against its own F32
render *at the same seed*, while two F32 renders at *adjacent seeds* sit at **0.005** — a ~48x
ratio. sc-14545 comment #15252 reproduces this independently on CUDA (0.218 / 0.0045), agreeing to
three significant figures across two accelerators.

**Cross-checkpoint agreement sits in the same band as an unrelated take.** The measured
medium-vs-specialist cosines are 0.037–0.282
([`tests/variant_quality.rs`](../../crates/audio/candle-audio-stable-audio-3/tests/variant_quality.rs)).
Any threshold wide enough to admit honest output also admits noise. That file's divergence bound is
therefore a *mis-wiring detector* — it proves the two ids are not the same weights — and is
explicitly not a quality bar.

**Two further results say the metric is tracking the wrong thing.** F16 **passes** the shipped
envelope gate (`hf_emphasis` 0.091–0.122, inside the allowed 0.050–0.222), so the F32 compute-dtype
policy rests on the cosine evidence above rather than on F16 failing a quality gate. And
`variant_quality.rs` records the direction of surprise: the *larger* architectural gap
(medium vs `small_sfx`) produced the *higher* cosine, because the metric was responding to the
prompt's sparsity rather than to the weights.

**Conclusion, and the negative criterion this protocol pins:**

> No objective metric that fits inside a `cargo test` can carry a perceptual-superiority claim for
> this pair. **No test in this repository may be changed to assert perceptual superiority from an
> agreement metric.** A supplemental metric (CLAP, FAD) may be reported *alongside* a panel result,
> labelled supplemental, and may never stand in for one.

What sc-14545 *does* claim, and what remains true with or without this protocol, is the objective
**capability** difference: 380 s against 120 s, the 852M SAME-L decoder against the 108M SAME-S, and
both domains against one. All three are gated by tests today.

---

## 2. Two questions, and the order they must be answered in

These are different questions with different instruments, and conflating them is the failure mode
this section exists to prevent.

| | question | instrument | statistic |
|---|---|---|---|
| **Q1** | **Discriminability** — can a listener tell the two checkpoints apart *at all*? | ABX | pooled accuracy vs chance (0.5) |
| **Q2** | **Preference** — given a choice, which is rated higher? | multi-stimulus rating (MUSHRA-style, no reference) | paired difference in rating points |

sc-14545's acceptance wording — *"audibly higher quality than `small_music`"* — asks **Q2**.

> ### The gating rule
>
> **Q1 gates Q2. If ABX fails to reject chance, the preference run is uninterpretable and MUST NOT
> be reported.**
>
> A preference expressed between two takes a listener demonstrably cannot tell apart is not a
> preference; it is a coin flip with a rating scale attached, and reporting it manufactures a
> finding out of noise. This is not advisory. `analyze` in `sa3_listening_blind.py` returns
> `mos: null` with the reason attached whenever the ABX test does not reject, and
> `scripts/tests/test_sa3_listening_blind.py` drives it with a simulated chance-level panel carrying
> a large true rating gap and requires the preference result to be suppressed.

A null on Q1 is not a failed experiment. It is the answer, and it retires the "audibly higher
quality" wording in favour of the capability claim that is already gated (§7).

---

## 3. Stimuli

Six prompts, three per domain.

| id | domain | prompt | held out |
|---|---|---|---|
| `music-1` | music | Slow downtempo trip-hop with dusty vinyl crackle, muted trumpet and a walking upright bass | yes |
| `music-2` | music | Bright Afrobeat groove with interlocking guitars, talking drum and a horn section riff | yes |
| `music-3` | music | Meditative lo-fi ambient piano jazz, soft acoustic drum kit | no |
| `sfx-1` | sfx | A single wooden door creaking open in a large empty stone hall | yes |
| `sfx-2` | sfx | Distant thunder rolling over a quiet field with light rain on leaves | yes |
| `sfx-3` | sfx | Dog barking next to a waterfall | no |

**Both domains are covered because medium is the only released SA3 checkpoint tagged for music
*and* sound-effects.** Medium is contrasted against `stable_audio_3_small_music` on the music
prompts and against `stable_audio_3_small_sfx` on the SFX prompts.

### Why the held-out four are not `demo_cond` prompts

The obvious design is to draw the whole set from the checkpoints' own shipped `demo_cond` lists, so
each is heard on material its authors chose. **That design is unavailable, and discovering why is
part of this story's result.**

`tests/provider.rs` commits **every shipped `demo_cond` prompt of every SA3 variant** as a
side-ratio calibration constant — `SFX_SWEEP_PROMPTS`, `MUSIC_SWEEP_PROMPTS`,
`MEDIUM_SWEEP_PROMPTS`, `MUSIC_BASE_SWEEP_PROMPTS`. Those sweeps are what the shipped floors were
*tuned on*. There is therefore no held-out prompt anywhere in the `demo_cond` pool, and a set drawn
entirely from it would score the panel on the crate's own tuning data.

So the four held-out entries are **authored for this panel**, in the same idiom and the same two
domains, and appear nowhere else in the repository. Four of six is above the one-half minimum.

The two anchors (`music-3`, `sfx-3`) are the crate's existing gate prompts, kept deliberately: they
tie the panel to the operating point the real-weight lanes measure at, so a panel result and a gate
measurement are about the same renders.

None of this is asserted in prose alone. `scripts/tests/test_sa3_listening_blind.py` parses the
stimulus table out of the Rust generator, scans every other `.rs` source under `crates/` for each
prompt, and fails if a held-out prompt is committed elsewhere — with a discriminating control that
requires the same scan to *find* the two anchors, so a scanner that matched nothing could not pass.

### Why `sfx-3` is load-bearing

Medium's stereo side ratio on *"Dog barking next to a waterfall"* collapses to **~1.2e-4 at two of
five seeds** (sc-14545's two-domain sweep in `tests/provider.rs`; comment #15161). Sparse SFX is
where the two checkpoints are most likely to differ in **character** rather than in fidelity — and
character is precisely what a preference test can detect and an agreement metric provably cannot.
Dropping this prompt to make the set tidier would remove the case with the highest prior probability
of carrying a real difference. It is not optional, and its presence is asserted by the test suite.

`sfx-1` and `sfx-2` are authored to be sparse for the same reason, so the load-bearing condition is
not carried by a single prompt.

### Render controls

Fixed, and identical for every take:

| control | value | why |
|---|---|---|
| duration | 30.0 s | the operating point every real-weight gate in the crate measures at |
| steps | 8 | as above |
| sampler | `pingpong` | the registered sampler |
| sample rate | 44 100 Hz | the contract output rate |
| seeds | `15178`, `15377` | two per (prompt, checkpoint) — the within-listener replicate |

The seeds are deliberately **disjoint from the crate's gate seeds** (`42 / 7 / 2026`). A panel scored
on the exact draws a threshold was calibrated against would inherit that calibration's luck.

Six prompts × two seeds × two checkpoints = **24 takes**.

---

## 4. Level matching — the control everything else rests on

**An uncorrected loudness difference alone produces a preference.** Louder reliably reads as
"better" in an unlevelled comparison. A panel run on unmatched takes measures gain staging, and its
result would be indistinguishable from a real quality finding.

Matching is on **BS.1770-4 gated integrated loudness (LUFS)** via
[`candle_audio::harness::MetricSet`](../../crates/audio/candle-audio/src/harness.rs), **not RMS**.
RMS is frequency-blind: two takes at identical RMS but different spectral centres differ in
perceived loudness by many LU. The weight-free control
`level_matching_collapses_a_loudness_gap_that_rms_matching_leaves` measures that gap rather than
asserting it — an RMS-matched 60 Hz / 3 kHz pair sits **6.71 LU** apart, and LUFS matching collapses
it to 0.00 LU.

| rule | value |
|---|---|
| common target | `min(quietest take, -23.0 LUFS, -1.0 dBTP − worst peak-to-loudness ratio − 0.1 dB)` |
| post-match tolerance, per take and pairwise | **< 0.5 LU** |
| post-match true-peak ceiling | **≤ -1.0 dBTP** |
| applied gain, recorded per take | linear and dB, in `manifest.json` |

Taking the target at or below the **quietest** take makes every applied gain attenuating, so no take
can be pushed into clipping — which matters because the set spans a generalist and two specialists at
different operating levels. The generator asserts attenuation-only and fails if any take needs
make-up gain.

### Why the target is also peak-constrained

**Loudness normalization alone does not bound the peak, and on this set it demonstrably did not.**
The first real run of the generator failed, and the third term above is the fix.

Matched to the set's quietest integrated loudness (-23.149 LUFS), `small_sfx`'s *"Distant thunder
rolling over a quiet field"* take at seed 15377 landed at **+0.031 dBTP** — above full scale, from
PCM whose sample values were all inside [-1, 1]. Sparse, transient material has a very high
peak-to-loudness ratio; this take's is **~23.2 dB**, and 4×-oversampled true peak catches the
inter-sample overs that sample peak misses.

The two obvious repairs are both wrong. A limiter alters the audio being judged. Per-take attenuation
breaks the level matching, which is the control the whole protocol rests on. So the **target itself**
is pulled down by the set's worst peak-to-loudness ratio: true peak scales with the applied gain in
dB, so bounding the worst ratio bounds every take at once, and the set stays matched to a single
number. In the measured run this cost **1.131 dB** of overall level — nothing, since a listener sets
playback volume once and only the *relative* match matters.

The weight-free control
`the_common_target_is_pulled_down_by_the_sets_worst_peak_to_loudness_ratio` replays exactly those
measurements, asserting in both directions: a loudness-only target reproduces the +0.031 dBTP
failure, and the peak-constrained target clears the ceiling. Without the first half it would be a
test of a value that was never in danger.

The tolerance is checked **after re-measuring**, not inferred from the applied gain: BS.1770-4's
absolute -70 LUFS gate does not move with the signal, so a uniform gain does not shift gated
integrated loudness by exactly the gain in dB. The generator iterates up to three correction passes
and asserts convergence.

0.5 LU is roughly half the smallest loudness step listeners reliably report, so a residual at this
bound cannot be the thing a preference is built on.

---

## 5. Blinding, randomization and controls

Assignment is a pure function of the **committed randomization seed `15178`** and the generator's
manifest, computed by `sa3_listening_blind.py assign`. Two operators running the panel produce the
same design and neither can nudge it.

Randomized per listener:

- which checkpoint is presented as **A** and which as **B**, independently per trial;
- whether **X** duplicates A or B;
- trial order, with null trials interleaved;
- rating-screen order and slot order.

The randomization uses a **SplitMix64** implemented in the script rather than `random`, whose stream
is an implementation detail of the interpreter. A protocol that pins "randomized from a committed
seed" needs the design regenerable on a different Python.

### Blinding

`assign` emits a **blinded playlist** carrying opaque per-listener handles only (`L03-t07-a.wav`) and
a **private key** holding the mapping. The script refuses to write if any variant name, prompt or
generator filename appears in the blinded artefacts, and the test suite verifies both that the
playlists are clean and that the leak detector fires on a planted regression. `materialize` **copies**
takes to their opaque names rather than symlinking — a symlink's target is the variant-bearing
filename, and `ls -l` would unblind the operator.

The key must be withheld from listeners **and operators** until every response is collected.

> **Known limitation, stated rather than papered over.** In an ABX trial X is byte-identical to
> either A or B, so a listener with filesystem access could checksum the three files and score 100%
> without listening. Presentation must therefore be through a player that exposes neither filenames
> nor file contents. The same-checkpoint control (below) is what detects this failure if it happens.

### The same-checkpoint control

Four of each listener's sixteen ABX trials present **the same file as A, B and X**. A listener has
no information on these and scores at chance by construction, so their pooled accuracy is the
**panel's validity check** and is reported as such.

There is no honest alternative. A same-checkpoint pair at *different seeds* is different audio and
therefore genuinely discriminable, which would make the control measure the wrong thing.

Pooled over the panel that is **80 null trials**. The central 95% acceptance band is **31 to 50
correct** (0.388–0.625). A score above the band means the blinding leaked, not that listeners heard
something: `analyze` reports `PANEL INVALID` and refuses to read the contrast trials at all.

### The rating consistency anchor

Each rating screen presents **three** takes: medium, the specialist, and a **hidden duplicate** of
one of them. A listener whose median absolute duplicate-vs-original discrepancy exceeds **20 points**
is excluded before the preference analysis. The rule and its threshold are fixed here, before any
data exists, so they cannot be tuned after seeing the result. The duplicate slot is excluded from the
preference contrast itself — averaging it in would weight whichever checkpoint happened to be
duplicated.

---

## 6. Pre-registered panel size and analysis

**Fixed before any listening.** Every number below is recomputed by
`python3 scripts/audio/sa3_listening_blind.py power`, and
`scripts/tests/test_sa3_listening_blind.py` re-derives the critical values independently and asserts
this document quotes them. The statistics are exact (binomial via `math.comb`; Student and
non-central t by numeric integration), not tabulated.

### Panel

**20 listeners retained after screening; recruit 24.** The size is set by the **preference** half,
which is the binding constraint — the ABX half is over-powered at 20. The number comes from the
power calculation below, not from a standard; it sits comfortably above the post-screening panel
sizes ITU-R BS.1534 multi-stimulus tests are typically run at, which is a sanity check on it rather
than its justification.

Per listener: **12 contrast ABX trials** (6 prompts × 2 seeds) + **4 null ABX trials** + **6 rating
screens** × 3 slots. Run as three blocks with breaks; total session ≈ 50 minutes.

### Q1 — ABX discriminability (the gate)

| | |
|---|---|
| pooled trials | **240** (20 × 12) |
| null hypothesis | p = 0.5 (chance) |
| test | exact binomial, **one-sided**, α = 0.05 |
| rejection threshold | **≥ 134 / 240** correct (0.558) |
| effect worth acting on | 0.65 pooled accuracy |
| power at that effect | **0.999** |
| minimum detectable accuracy at 80% power | **0.584** |

One-sided because below-chance ABX is not a finding, it is noise.

The minimum detectable accuracy is the honest bound on a null: with 240 trials, "no effect found"
means "no effect at or above 58.4% pooled accuracy found". It does not mean no effect exists.

### Q2 — preference (only if Q1 rejects)

| | |
|---|---|
| unit of analysis | one number per listener: mean(medium rating) − mean(specialist rating) over their 6 screens |
| scale | continuous 0–100, MUSHRA-style, **no hidden reference** (a text-to-audio render has no ground truth, so a true MUSHRA reference does not exist) |
| test | two-sided paired t, α = 0.05, n = 20 |
| robustness check | exact Wilcoxon signed-rank, pre-registered, reported alongside |
| effect worth acting on | **10 points** — half the width of one labelled band on the five-band 0–100 scale |
| assumed SD of the paired difference | 15 points (`dz` = 0.667) |
| power at that effect | **0.807** |
| minimum detectable difference at 80% power | **9.9 points** |

Two-sided because medium being *worse* is a real and reportable outcome.

The assumed SD is the number most likely to be wrong. It is an assumption, flagged as one: the first
run must re-estimate it from the observed data, and any second run must re-derive this
pre-registration from the estimate before listening begins.

### Reporting

**A null result is a valid, useful and likely outcome, and must be reported.** Two post-trained
checkpoints of one family are more likely than not to be indistinguishable at 30 s / 8 steps. The
analysis emits one of five conclusions and none of them is a failure to report:

1. `PANEL INVALID` — the control leaked; nothing downstream is interpretable.
2. `NULL RESULT` — ABX did not reject chance. **Preference is not reported.** The "audibly higher
   quality" wording is retired.
3. `DISCRIMINABLE BUT NO PREFERENCE` — listeners can tell them apart; neither is rated higher.
   Medium is *different*, not *better*.
4. `PREFERENCE FOR MEDIUM` — compare the observed difference against the 10-point effect the panel
   was sized for before treating it as a product claim.
5. `PREFERENCE AGAINST MEDIUM` — reported as-is.

Results are recorded on **sc-15377**, together with the observed SD, the control score, the
exclusions, and the raw unblinded responses.

---

## 7. What happens to sc-14545's wording

sc-14545's acceptance asked for *"audibly higher quality than `small_music`"*. PR #275 declined to
claim it and said so in two places —
[`tests/variant_quality.rs`](../../crates/audio/candle-audio-stable-audio-3/tests/variant_quality.rs)
and [`SC_14545_MEDIUM_PROVIDER.md`](SC_14545_MEDIUM_PROVIDER.md). A sweep of `docs/`, `crates/audio/`,
`release/` and every descriptor found **no unsupported perceptual claim anywhere in the repository**;
both hits are disclaimers, not claims.

So there is nothing to retract. What changed with this story is that both disclaimers now **point at
this protocol** instead of describing an unspecified future piece of work, and this document names
the wording's disposition explicitly:

> The "audibly higher quality" wording is **retired** in favour of the capability claim that is
> already gated — **380 s against 120 s, SAME-L against SAME-S, both domains against one** — and is
> reinstated only if sc-15377 reports outcome 4 above with a difference at or above the 10-point
> effect this panel was sized for.

Until then, medium is the checkpoint that can do more, not the checkpoint that sounds better.

---

## 8. Running it

Generate the stimulus set (Metal, release; the snapshots must already be on disk — inference never
self-fetches):

```bash
SA3_MEDIUM_SNAPSHOT=...      \
SA3_SMALL_MUSIC_SNAPSHOT=... \
SA3_SMALL_SFX_SNAPSHOT=...   \
SA3_LISTENING_WAV_DIR=/path/to/stimuli \
  cargo test -p candle-audio-stable-audio-3 --features metal --release \
    --test listening_stimuli -- --ignored --nocapture
```

Each variant is loaded once and renders all of its takes in that single process — constructing a
generator costs ~43 s of cold start against a 10.4 GB (medium) or 3.45 GB (specialist) snapshot.
Output is 24 WAVs plus `manifest.json` recording the prompt, seed, applied gain, pre/post LUFS,
post true peak and PCM digest of every take.

**Measured run** (Metal, release, three on-disk snapshots, 157 s wall clock for all 24 takes):

| | |
|---|---|
| takes | 24 (6 prompts × 2 seeds × 2 checkpoints), 122 MB |
| pre-match integrated loudness | -10.546 … -23.149 LUFS (12.6 LU spread) |
| worst peak-to-loudness ratio | 23.181 dB (`sfx-2` / `small_sfx` / seed 15377) |
| common target | **-24.281 LUFS** |
| applied gain | -1.131 … -13.734 dB, all attenuating |
| widest pairwise post-match loudness delta | **0.0000 LU** |
| worst post-match true peak | **-1.100 dBTP** |
| distinct PCM digests | 24 of 24 |

The 12.6 LU pre-match spread is the measurement that makes §4 concrete: without level matching, the
loudest take in this set would sit more than twelve loudness units above the quietest, and the panel
would be reporting that.

Then design the panel, collect responses, and analyse:

```bash
python3 scripts/audio/sa3_listening_blind.py power          # the pre-registration, recomputed
python3 scripts/audio/sa3_listening_blind.py assign  --manifest stimuli/manifest.json \
                                                     --source-dir stimuli/ --out-dir panel/
python3 scripts/audio/sa3_listening_blind.py sheet   --playlist panel/playlist_L01.json \
                                                     --out-dir panel/
# ... listening happens here ...
python3 scripts/audio/sa3_listening_blind.py unblind --key panel/key.json \
                                                     --abx panel/abx_*.csv \
                                                     --ratings panel/ratings_*.csv \
                                                     --out panel/unblinded.json
python3 scripts/audio/sa3_listening_blind.py analyze --unblinded panel/unblinded.json
```

### CI

The generator's real-weight case is `#[ignore]`d — it is a generator, not a gate, and it runs in no
automatic lane by design. Its two weight-free cases (the LUFS-vs-RMS control and the peak-constraint
control) **are** named in the `Test Stable Audio 3 weight-free quality gates` step of
`.github/workflows/ci.yml`; `scripts/tests/test_sa3_ci_target_coverage.py` fails the build if that
target ever stops being named.

The Python half runs under `python3 -m unittest discover -s scripts/tests`, which also enforces the
cross-file hold-out claim and that this document quotes the numbers the analysis script computes.
