"""Self-consistency checks over the Mage-Flow goldens — sc-14036 (epic 14034).

Run after `dump_mage_flow_golden.py`. This does **not** re-run the reference; it asserts that
the dumped bundle hangs together, which catches the failure mode a parity golden is worst at
surfacing: a golden that is internally wrong but that every downstream test happily "matches".

Structure and metadata alone cannot do that — a bundle whose *payload* tensors were replaced by
shape-matched, moment-matched noise satisfies every shape/dtype/range assertion. So the checks
below are split in two:

* **structural** invariants (shapes, ranges, drop indices, schedule ladder), and
* **payload** invariants that recompute or cross-derive a tensor's actual numbers and would fail
  if it were replaced by noise, rescaled, or captured at the wrong point. These are weightless —
  they use only closed-form math (the msrope table, the Euler step, a reshape) or a *second,
  independently captured* tensor from another golden file.

The payload invariants, and the epic GAP answer each one pins:

* `traj_step1 == traj_step0 + Δσ·(unc + cfg·(cond − unc))` — the DiT velocity AND the CFG
  combine AND the Euler step, closed over the `dit` + `e2e` goldens (GAP 4: the schedule);
* the msrope table recomputed in numpy from `img_shapes` + `theta=10000` +
  `axes_dim=[16,56,56]` — GAP 3, including the confirmed `batch_cfg` **frame index 1** for the
  duplicated uncond half;
* `dit_in.txt == concat(gen_txt, neg_txt)`, bit-exact — GAP 1: the conditioning the DiT actually
  consumed IS the dumped final-post-RMSNorm TE output;
* the per-token RMS envelope of `*_hidden_full` — GAP 1 again, scale-anchored, because the
  penultimate layer and the final *pre*-norm state are orders of magnitude larger;
* `dec_from_latent` reconstructs `pixels` and `enc_mean`'s channels track that image — GAP 2:
  the VAE payload is a real encode/decode;
* `seq_step0`'s ref segment correlates with the `vae` golden's `enc_mean` — GAP 5: the ref
  segment really is the VAE encoding of the same image;
* `final_tokens` <-> `final_latent` reshape identity, plus channels tracking `image_u8` — the
  sampler output really produced the decoded image;
* cond/uncond halves of `dit_out` / `block_out.1` stay near-collinear and the block outputs are
  heavy-tailed — the DiT and block payloads are real activations, not Gaussian noise.

Every CFG-dependent invariant is gated on the branch count, NOT written as if the stream were
always doubled. Turbo's documented default is **cfg 1.0**, where the reference never builds the
negative branch (`pipeline.py:326`, `:535`) — the stream is not duplicated, `img_shapes` is not
concatenated and `_velocity` returns the raw transformer output. Assuming otherwise would reject
a correct Turbo golden wholesale, and at cfg exactly 1.0 the combine `unc + 1·(cond−unc)` is
algebraically `cond`, so the discrimination assertion would be self-defeating. The CFG-off path
falls back to the single-branch Euler identity — a real check, not a skip — scored against a
dropped step, a sigma off-by-one, a sign flip and a missing Δσ.

    python tools/verify_mage_flow_golden.py
    python tools/verify_mage_flow_golden.py --self-test   # prove the above DISCRIMINATE

`--self-test` corrupts a scratch copy of the bundle one tensor at a time — moment-matched, with
any structural invariant it would otherwise trip deliberately repaired — and asserts this script
rejects every one. A checker that has never been shown to fail is not evidence of anything.

Exits non-zero if any invariant fails, if an expected golden is **missing** (pass
`--allow-missing` to downgrade that to a skip), or if nothing was checked at all. Needs only
`numpy` + `safetensors` — no torch, no weights, no reference checkout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from _paths import fixture

GOLDEN = Path(fixture("tools/golden"))

# Every stage `dump_mage_flow_golden.py --stage all` writes. Absence is a FAILURE by default:
# the goldens are gitignored, so "no files" is the state of a fresh clone, and a checker that
# reports green there reports green exactly when it verified nothing.
_STEMS = ("noise", "vae", "te", "dit", "dit_block", "e2e", "edit")

# `MageFlowEmbedRope(theta=10000, axes_dim=params.axes_dim, scale_rope=True)`
# (`_vendor/mage_flow/models/mage_flow.py:72`); axes_dim = [16, 56, 56] sums to the 128-wide
# attention head dim, so the per-token table is 64 complex entries: 8 frame + 28 h + 28 w.
ROPE_THETA = 10000
ROPE_AXES_DIM = (16, 56, 56)

_FAILURES: list[str] = []
_CHECKS = 0


def _check(name: str, ok: bool, detail: str = "") -> None:
    global _CHECKS
    _CHECKS += 1
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        _FAILURES.append(name)


def _load(stem: str, allow_missing: bool) -> dict[str, np.ndarray] | None:
    path = GOLDEN / f"mage_flow_{stem}_golden.safetensors"
    if not path.is_file():
        if allow_missing:
            print(f"  [skip] {stem}: {path.name} not present (--allow-missing)")
            return None
        _check(
            f"{stem} golden is present",
            False,
            f"{path} missing — run `python tools/dump_mage_flow_golden.py --stage {stem}`",
        )
        return None
    return load_file(str(path))


def _require(bundle: dict[str, np.ndarray], *keys: str) -> bool:
    """True when every key is present; otherwise report the stale golden and skip the check."""
    missing = [k for k in keys if k not in bundle]
    if missing:
        _check(
            f"keys present: {', '.join(keys)}",
            False,
            f"missing {missing} — re-run dump_mage_flow_golden.py for this stage",
        )
        return False
    return True


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def _tokens_of(latent: np.ndarray) -> np.ndarray:
    """`[1, C, gh, gw]` -> `[gh*gw, C]`, the reference's `rearrange(x, "b c h w -> b (h w) c")`."""
    return latent[0].reshape(latent.shape[1], -1).T


def _cfg_branches(bundle: dict[str, np.ndarray]) -> int:
    """How many CFG branches the reference fused into the packed stream: 2 with CFG, else 1.

    `use_neg = cfg > 1.0 and any(neg_prompts…)` (`pipeline.py:326`) and `if cfg > 1.0:`
    (`:535`) — at or below cfg 1 the negative branch is never built, so the image stream is NOT
    duplicated, `img_shapes` is not concatenated, and `_velocity` returns the raw transformer
    output with no combine. **Turbo's documented default is cfg 1.0**, so every doubling
    assumption here has to be conditional or a correct Turbo golden gets rejected wholesale.
    Each caller asserts the structural consequence too, so this scalar cannot lie unchallenged.
    """
    return 2 if float(bundle["cfg"][0]) > 1.0 else 1


def _packed_segments(bundle: dict[str, np.ndarray], cu_key: str) -> int | None:
    """Segment count from a varlen `cu_seqlens` array (`len - 1`), or None if absent."""
    cu = bundle.get(cu_key)
    return int(len(cu) - 1) if cu is not None else None


# ------------------------------------------------------------------ payload discriminators


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over the flattened tensors — 0 for independent noise, ~1 for a match.

    The workhorse for "is this tensor still the thing it was captured as, or shape-matched
    noise": every use below pairs a tensor with an *independently captured* one it must track.
    """
    x = np.asarray(a, dtype=np.float64).ravel() - float(np.mean(a, dtype=np.float64))
    y = np.asarray(b, dtype=np.float64).ravel() - float(np.mean(b, dtype=np.float64))
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / denom) if denom > 0 else 0.0


def _tv_ratio(x: np.ndarray) -> float:
    """Mean |neighbour difference| over the last two (spatial) axes, in units of the std.

    Scale-free, so it survives any rescale of the tensor. White noise sits at ~1.13 (2−√2·…, the
    Gaussian mean absolute difference); anything with spatial structure — a decoded image, a VAE
    latent — sits far below it.
    """
    a = np.asarray(x, dtype=np.float64)
    std = float(a.std())
    if std == 0.0:
        return 0.0
    tv = float(np.abs(np.diff(a, axis=-1)).mean() + np.abs(np.diff(a, axis=-2)).mean()) / 2.0
    return tv / std


def _kurtosis(x: np.ndarray) -> float:
    """Non-excess kurtosis. Exactly 3 for a Gaussian; transformer activations are far heavier."""
    a = np.asarray(x, dtype=np.float64)
    var = float(a.var())
    return float(((a - a.mean()) ** 4).mean() / var**2) if var > 0 else 0.0


def _block_luma(image: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """`[H, W, 3]` u8 or `[1, 3, H, W]` float -> the `[gh, gw]` block-mean luminance map."""
    a = np.asarray(image, dtype=np.float64)
    plane = a.mean(axis=-1) if a.ndim == 3 else a[0].mean(axis=0)
    h, w = plane.shape
    return plane.reshape(gh, h // gh, gw, w // gw).mean(axis=(1, 3))


def _latent_image_channels(latent: np.ndarray, image: np.ndarray, threshold: float = 0.5) -> int:
    """How many of the latent's channels track the image's block-mean luminance.

    A `[1, C, gh, gw]` latent that really encodes `image` has a double-digit count here; an
    independent random tensor of the same shape has **zero** (its best channel peaks at ~0.2).
    """
    luma = _block_luma(image, latent.shape[2], latent.shape[3])
    return sum(1 for ch in latent[0] if abs(_corr(ch, luma)) > threshold)


def _rope_axis(index: np.ndarray, dim: int) -> np.ndarray:
    """`MageFlowEmbedRope.rope_params` — `e^{i·index·theta^(−2k/dim)}`, `dim//2` entries."""
    inv = 1.0 / np.power(float(ROPE_THETA), np.arange(0, dim, 2, dtype=np.float64) / dim)
    return np.exp(1j * np.outer(index.astype(np.float64), inv))


def _msrope_table(shapes: np.ndarray) -> np.ndarray:
    """Recompute the packed msrope table `[sum(f·h·w), 64]` from the segment `(frame, h, w)` list.

    A pure function of `img_shapes` + `theta` + `axes_dim` — no weights — so the golden's table
    can be checked against first principles rather than merely inspected for unit modulus.

    Mirrors `MageFlowEmbedRope._compute_video_freqs` (`mage_layers.py:187-209`) with
    `scale_rope=True`: the **frame** frequencies are indexed by the segment's position `idx` in
    the list, NOT by the row's frame count, and h/w are centered (`neg[-(n-n//2):] ++ pos[:n//2]`).
    That `idx` is what makes the `batch_cfg` duplicate rotate at frame 1 — see `_build_pack_ctx`'s
    `d_img_shapes = [img_shapes[0] + img_shapes[0]]` (`pipeline.py:161`).
    """
    pos_index = np.arange(4096)
    neg_index = np.arange(4096)[::-1] * -1 - 1
    pos = [_rope_axis(pos_index, d) for d in ROPE_AXES_DIM]
    neg = [_rope_axis(neg_index, d) for d in ROPE_AXES_DIM]

    segments = []
    for idx, row in enumerate(shapes):
        frame, height, width = (int(v) for v in row)
        f_freqs = np.broadcast_to(
            pos[0][idx : idx + frame].reshape(frame, 1, 1, -1),
            (frame, height, width, ROPE_AXES_DIM[0] // 2),
        )
        h_freqs = np.concatenate(
            [neg[1][-(height - height // 2) :], pos[1][: height // 2]], axis=0
        )
        h_freqs = np.broadcast_to(
            h_freqs.reshape(1, height, 1, -1), (frame, height, width, ROPE_AXES_DIM[1] // 2)
        )
        w_freqs = np.concatenate([neg[2][-(width - width // 2) :], pos[2][: width // 2]], axis=0)
        w_freqs = np.broadcast_to(
            w_freqs.reshape(1, 1, width, -1), (frame, height, width, ROPE_AXES_DIM[2] // 2)
        )
        segments.append(
            np.concatenate([f_freqs, h_freqs, w_freqs], axis=-1).reshape(frame * height * width, -1)
        )
    return np.concatenate(segments, axis=0)


# ------------------------------------------------------------------------------ stages


def check_noise(noise: dict[str, np.ndarray]) -> None:
    _check(
        "watermark is detectable in gs_noise",
        float(noise["detect_msg_acc"][0]) == 1.0 and float(noise["detect_raw_acc"][0]) > 0.99,
        f"msg_acc={float(noise['detect_msg_acc'][0]):.4f} raw_acc={float(noise['detect_raw_acc'][0]):.4f}",
    )
    # The whole point of the tensor: it must NOT be the plain randn the reference discards.
    _check(
        "gs_noise differs from the discarded plain randn",
        _max_abs(noise["gs_noise"], noise["plain_randn"]) > 0.1,
        f"max_abs={_max_abs(noise['gs_noise'], noise['plain_randn']):.4f}",
    )
    if _require(noise, "gs_noise_bf16"):
        _check(
            "gs_noise_bf16 is gs_noise at bf16 precision",
            0.0 < _max_abs(noise["gs_noise"], noise["gs_noise_bf16"]) < 0.05,
            f"max_abs={_max_abs(noise['gs_noise'], noise['gs_noise_bf16']):.6f}",
        )


def check_vae(vae: dict[str, np.ndarray], edit: dict[str, np.ndarray] | None) -> None:
    _check(
        "enc_latent is the posterior MEAN (sample_posterior=False branch)",
        _max_abs(vae["enc_latent"], vae["enc_mean"]) == 0.0,
    )
    _check(
        "enc_logvar is clamped to [-20, 10]",
        float(vae["enc_logvar"].min()) >= -20.0 and float(vae["enc_logvar"].max()) <= 10.0,
        f"[{vae['enc_logvar'].min():.3f}, {vae['enc_logvar'].max():.3f}]",
    )
    _check(
        "decode output is in [-1, 1] (pre-clamp reference range)",
        float(np.abs(vae["dec_from_latent"]).max()) < 1.5,
        f"max_abs={float(np.abs(vae['dec_from_latent']).max()):.4f}",
    )
    _check(
        "the two decodes differ (synthetic latent is not the encoded one)",
        _max_abs(vae["dec_from_latent"], vae["dec_from_synth"]) > 0.1,
    )

    # --- payload: the decode really is a RECONSTRUCTION of the encoded image ---------------
    # encode(x) -> decode round-trips a Mage-VAE at ~0.998 correlation. Shape-matched noise
    # (or a decode of the wrong latent — `dec_from_synth` scores 0.03) cannot reach this.
    recon = _corr(vae["dec_from_latent"], vae["pixels"])
    _check(
        "dec_from_latent RECONSTRUCTS pixels (encode->decode round-trip)",
        recon > 0.95,
        f"corr={recon:.4f} (dec_from_synth vs pixels: {_corr(vae['dec_from_synth'], vae['pixels']):.4f})",
    )
    for key in ("dec_from_latent", "dec_from_synth"):
        tv = _tv_ratio(vae[key])
        _check(
            f"{key} is a smooth image, not white noise (TV/std)",
            tv < 0.6,
            f"TV/std={tv:.4f} (white noise ~1.13)",
        )
    # --- payload: the posterior mean really encodes the input image ------------------------
    tracked = _latent_image_channels(vae["enc_mean"], vae["pixels"])
    _check(
        "enc_mean channels track the input image's luminance (it is an ENCODING, not noise)",
        tracked >= 4,
        f"{tracked}/128 channels |corr|>0.5 vs the 16x block-mean luma (random: 0)",
    )
    if edit is not None and "seq_step0" in edit and "target_tokens" in edit:
        # Independent corroboration from a SEPARATE run: the edit pipeline VAE-encodes the same
        # reference image at the same resolution (posterior SAMPLE, so ~equal, not identical).
        n_tgt = int(edit["target_tokens"][0])
        ref_seg = edit["seq_step0"][0, n_tgt : 2 * n_tgt]
        if ref_seg.shape == _tokens_of(vae["enc_mean"]).shape:
            agree = _corr(ref_seg, _tokens_of(vae["enc_mean"]))
            _check(
                "the edit golden's ref segment is the SAME VAE encoding as enc_mean",
                agree > 0.9,
                f"corr={agree:.4f} (edit samples the posterior, hence not bit-equal)",
            )


def check_te(te: dict[str, np.ndarray], dit: dict[str, np.ndarray] | None) -> None:
    drop = int(te["gen_drop_idx"][0])
    gen_len = int(te["gen_txt_len"][0])
    _check(
        f"gen_txt == gen_hidden_full[{drop}:{drop + gen_len}] (drop_idx {drop})",
        _max_abs(te["gen_txt"], te["gen_hidden_full"][drop : drop + gen_len]) == 0.0,
    )
    _check(
        "gen drop_idx is 34",
        drop == 34,
        f"drop_idx={drop}, {te['gen_hidden_full'].shape[0]} packed tokens -> {gen_len} conditioning tokens",
    )
    edrop = int(te["edit_drop_idx"][0])
    elen = int(te["edit_txt_len"][0])
    _check(
        f"edit_txt == edit_hidden_full[{edrop}:{edrop + elen}] (drop_idx {edrop})",
        _max_abs(te["edit_txt"], te["edit_hidden_full"][edrop : edrop + elen]) == 0.0,
    )
    _check("edit drop_idx is 64", edrop == 64, f"drop_idx={edrop} -> {elen} conditioning tokens")
    # The NEGATIVE prompt shares `gen_hidden_full`: `_encode_texts_packed` packs [pos, neg] into
    # ONE varlen forward, so the negative's post-drop slice is the tail of the same tensor. Worth
    # asserting separately — the positive slice sits at a fixed offset and can look right while
    # the tail is wrong (that is exactly how an MPS-dumped bundle fails; see tools/golden/README).
    if _require(te, "neg_txt", "neg_txt_len"):
        neg_len = int(te["neg_txt_len"][0])
        pos_seq = te["gen_hidden_full"].shape[0] - (drop + neg_len)
        tail = te["gen_hidden_full"][pos_seq + drop :]
        _check(
            f"neg_txt == gen_hidden_full[{pos_seq + drop}:] (the packed negative's post-drop tail)",
            tail.shape == te["neg_txt"].shape and _max_abs(te["neg_txt"], tail) == 0.0,
            f"max_abs={_max_abs(te['neg_txt'], tail):.4f}"
            if tail.shape == te["neg_txt"].shape
            else f"shape {tuple(tail.shape)} vs {tuple(te['neg_txt'].shape)}",
        )
    _check(
        "conditioning width is the Qwen3-VL LM hidden size (2560)",
        te["gen_txt"].shape[-1] == 2560 and te["edit_txt"].shape[-1] == 2560,
    )
    # A different template + a different image path must not produce the same vector.
    _check(
        "gen and edit conditioning are distinct",
        _max_abs(te["gen_vec"], te["edit_vec"]) > 1e-3,
    )

    # --- GAP 1, pinned end to end ----------------------------------------------------------
    # The slice relation above only says `gen_txt` is *a* window of `gen_hidden_full`; it says
    # nothing about WHICH hidden state was captured, so it survives any rescale applied to both.
    # This does not: `dit_in.txt` is the conditioning the transformer actually consumed, hooked
    # out of the e2e run, and under batch_cfg it is `cat(pos, neg)` (`_build_pack_ctx`'s `d_txt`).
    # It must be the dumped TE output BIT-FOR-BIT — which is only true if the TE golden captured
    # the final post-RMSNorm hidden state the pipeline feeds forward.
    # With CFG off the negative prompt is never encoded, so the DiT consumes the positive
    # conditioning alone — same bit-exact tie, one fewer segment.
    if dit is not None and "dit_in.txt" in dit and _require(te, "neg_txt"):
        packed = (
            np.concatenate([te["gen_txt"], te["neg_txt"]], axis=0)
            if _cfg_branches(dit) == 2
            else te["gen_txt"]
        )
        streams = "cat(gen_txt, neg_txt)" if _cfg_branches(dit) == 2 else "gen_txt (cfg <= 1)"
        consumed = dit["dit_in.txt"][0]
        _check(
            f"GAP 1: dit_in.txt IS {streams} bit-for-bit (final post-RMSNorm state)",
            consumed.shape == packed.shape and _max_abs(consumed, packed) == 0.0,
            f"dit_in.txt={tuple(consumed.shape)} vs packed={tuple(packed.shape)}"
            + (
                f", max_abs={_max_abs(consumed, packed):.4f}"
                if consumed.shape == packed.shape
                else ""
            ),
        )
    # `vec` is the MEAN of the post-drop tokens (`text_encoder.py:565`) — a second, independent
    # tie between the pooled vector and the token tensor (bf16 accumulation, hence not exact).
    for prefix, tokens in (("gen", "gen_txt"), ("neg", "neg_txt"), ("edit", "edit_txt")):
        if f"{prefix}_vec" in te and tokens in te:
            delta = _max_abs(te[f"{prefix}_vec"][0], te[tokens].astype(np.float64).mean(axis=0))
            _check(
                f"{prefix}_vec is the MEAN of {tokens} (bf16 accumulation)",
                delta < 1.0,
                f"max_abs={delta:.4f}",
            )
    # Scale anchor for the same GAP: an RMSNorm output has a checkpoint-fixed per-token scale
    # (`rms(y_t) ≈ rms(norm.weight)`, because the input is normalized to rms 1 first) and a
    # tightly concentrated one. The residual stream feeding that norm does not: the reference
    # probe measured max_abs 10433 at the penultimate layer and 4225 at the final PRE-norm state,
    # against 113 here. This rejects a wrong-layer capture even if it were uniformly rescaled to
    # keep the slice relation intact.
    medians = {}
    for prefix in ("gen", "edit"):
        key = f"{prefix}_hidden_full"
        if key not in te:
            continue
        rms = np.sqrt((te[key].astype(np.float64) ** 2).mean(axis=-1))
        p05, p50, p95 = (float(np.percentile(rms, p)) for p in (5, 50, 95))
        medians[prefix] = p50
        _check(
            f"{key} has the post-RMSNorm per-token scale (median RMS in [2, 6])",
            2.0 <= p50 <= 6.0,
            f"median={p50:.4f} (p5={p05:.4f}, p95={p95:.4f})",
        )
        _check(
            f"{key} per-token RMS is CONCENTRATED (normalized state, not a residual stream)",
            p95 / p50 < 1.6 and p05 / p50 > 0.5,
            f"p95/p50={p95 / p50:.3f}, p5/p50={p05 / p50:.3f}",
        )
    if len(medians) == 2:
        spread = abs(medians["gen"] - medians["edit"]) / max(medians.values())
        _check(
            "gen and edit hidden states share one norm scale (same final RMSNorm weight)",
            spread < 0.25,
            f"medians {medians['gen']:.4f} vs {medians['edit']:.4f} ({spread:.1%} apart)",
        )


def check_dit(dit: dict[str, np.ndarray], e2e: dict[str, np.ndarray] | None) -> None:
    _check(
        "dit_in.img carries the packed latent at 128 channels",
        dit["dit_in.img"].shape[-1] == 128,
        f"shape={tuple(dit['dit_in.img'].shape)}",
    )
    _check(
        "dit_in.txt is the 2560-wide conditioning",
        dit["dit_in.txt"].shape[-1] == 2560,
        f"shape={tuple(dit['dit_in.txt'].shape)}",
    )
    _check(
        "dit_out matches dit_in.img in shape (velocity, pre-CFG)",
        dit["dit_out"].shape == dit["dit_in.img"].shape,
    )
    branches = _cfg_branches(dit)
    segments = _packed_segments(dit, "dit_in.img_cu_seqlens")
    # The negative branch only EXISTS above cfg 1 (`use_neg = cfg > 1.0 and …`,
    # `pipeline.py:326`), so the packed stream is doubled iff CFG is on. Turbo's documented
    # default is cfg 1.0, so the un-doubled shape is a supported configuration, not a corruption.
    _check(
        "the stream is fused with a negative branch IFF cfg > 1 (pipeline.py:326)",
        segments is None or segments == branches,
        f"cfg={float(dit['cfg'][0]):g} -> {branches} branch(es), packed segments={segments}",
    )
    img = dit["dit_in.img"]
    half = img.shape[1] // branches
    if branches == 2:
        # batch_cfg packs [cond, uncond] into one varlen forward, and the two halves must DIFFER
        # on output (identical halves would mean the negative branch never applied).
        _check(
            "batch_cfg duplicated the image stream (cond half == uncond half on input)",
            _max_abs(img[:, :half], img[:, half:]) == 0.0,
        )
        _check(
            "…but the velocities differ (the negative branch actually conditions)",
            _max_abs(dit["dit_out"][:, :half], dit["dit_out"][:, half:]) > 1e-4,
            f"max_abs={_max_abs(dit['dit_out'][:, :half], dit['dit_out'][:, half:]):.5f}",
        )
        # …while still being nearly COLLINEAR: same latent, same timestep, only the text differs.
        # Independent noise in either half would score ~0 here.
        align = _corr(dit["dit_out"][0, :half], dit["dit_out"][0, half:])
        _check(
            "cond and uncond velocities are near-collinear (one latent, two prompts)",
            align > 0.9,
            f"corr={align:.4f}",
        )
    else:
        # CFG off: one segment, so there is nothing to compare halves against. Assert the shape
        # consequence instead of quietly dropping three checks.
        _check(
            "CFG off: the image stream is ONE undoubled latent grid",
            img.shape[1] == half and (segments in (None, 1)),
            f"{img.shape[1]} tokens, {segments} segment(s)",
        )
    if "img_shapes" in dit:
        _check(
            "img_shapes is one (frame, h, w) row per packed segment",
            dit["img_shapes"].ndim == 2
            and dit["img_shapes"].shape[1] == 3
            and dit["img_shapes"].shape[0] == branches,
            f"{dit['img_shapes'].tolist()}",
        )

    # --- payload: the Euler identity -------------------------------------------------------
    # `_velocity` combines the two halves as `unc + cfg*(cond - unc)` and the flow-match Euler
    # scheduler advances `x <- x + (σ_{i+1} - σ_i)·v`. Recomputing that from the *dit* golden's
    # velocity and the *e2e* golden's trajectory closes the loop over three files with no
    # weights, and is the check that a randomized `dit_out` (or `traj_step*`) cannot survive.
    if e2e is None or not {"traj_step0", "traj_step1", "cfg"} <= set(e2e):
        return
    steps = int(e2e["geometry"][2]) if "geometry" in e2e else 4
    sig_key = f"sigmas_{steps}"
    if sig_key not in e2e:
        return
    sigmas = e2e[sig_key].astype(np.float64)
    traj0, traj1 = e2e["traj_step0"], e2e["traj_step1"]
    if dit["dit_out"].shape[1] != traj0.shape[1]:
        return
    n = traj0.shape[1] // branches
    x0 = traj0[0, :n].astype(np.float64)
    x1 = traj1[0, :n].astype(np.float64)
    cfg = float(e2e["cfg"][0])
    d_sigma = float(sigmas[1] - sigmas[0])

    def residual(velocity: np.ndarray) -> float:
        return float(np.abs(x1 - (x0 + d_sigma * velocity)).max())

    if branches == 2:
        cond = dit["dit_out"][0, :n].astype(np.float64)
        unc = dit["dit_out"][0, n:].astype(np.float64)
        applied = unc + cfg * (cond - unc)
        label = f"(unc + {cfg:g}·(cond-unc))"
        # At cfg == 1 these degenerate onto the right answer algebraically, which is exactly why
        # the whole block is gated on `branches`.
        wrong = {
            "cond-only": residual(cond),
            "uncond-only": residual(unc),
            "swapped": residual(cond + cfg * (unc - cond)),
            "no-step": float(np.abs(x1 - x0).max()),
        }
    else:
        # CFG off: the velocity IS the transformer output, ungated. Same Euler step, and the
        # alternatives are the real port bugs available on this path — a dropped step, an
        # off-by-one into the sigma ladder, a sign error, and forgetting the Δσ scaling.
        applied = dit["dit_out"][0, :n].astype(np.float64)
        label = "dit_out (no CFG combine — cfg <= 1)"
        wrong = {
            "no-step": float(np.abs(x1 - x0).max()),
            "sigma off-by-one": float(
                np.abs(x1 - (x0 + float(sigmas[2] - sigmas[1]) * applied)).max()
            ),
            "sign-flipped": residual(-applied),
            "unscaled (Δσ dropped)": float(np.abs(x1 - (x0 + applied)).max()),
        }
    combined = residual(applied)
    # The reference runs bf16, so the step leaves a rounding residual (~0.009 on a std-0.95
    # tensor); every wrong alternative is an order of magnitude worse.
    _check(
        f"EULER: traj_step1 == traj_step0 + (σ1-σ0)·{label}",
        combined < 0.05,
        f"max_abs={combined:.5f} (bf16 step residual; x1 std={float(x1.std()):.3f})",
    )
    _check(
        "…and that is the ONLY velocity that fits (the check discriminates)",
        all(v > 5.0 * max(combined, 1e-9) for v in wrong.values()),
        ", ".join(f"{k}={v:.4f}" for k, v in wrong.items()),
    )


def check_dit_block(block: dict[str, np.ndarray], dit: dict[str, np.ndarray] | None) -> None:
    rope_re = block.get("block_in.image_rotary_emb_re")
    rope_im = block.get("block_in.image_rotary_emb_im")
    _check(
        "msrope table survived as COMPLEX (real + imag), not truncated to its real part",
        rope_re is not None and rope_im is not None,
    )
    if rope_re is not None and rope_im is not None:
        modulus = np.sqrt(rope_re.astype(np.float64) ** 2 + rope_im.astype(np.float64) ** 2)
        _check(
            "msrope entries have unit modulus (they are e^{i·theta})",
            float(np.abs(modulus - 1.0).max()) < 1e-5,
            f"max||z|-1|={float(np.abs(modulus - 1.0).max()):.2e}",
        )
        _check(
            "msrope half-width is head_dim/2 = 64",
            rope_re.shape[-1] == 64,
            f"shape={tuple(rope_re.shape)}",
        )
        _check(
            "the imaginary part is non-trivial (would be all-zero if wrongly cast)",
            float(np.abs(rope_im).max()) > 0.1,
        )
        _check_msrope_payload(block, dit, rope_re, rope_im)

    _check(
        "block output shapes match its inputs (txt, img)",
        block["block_out.0"].shape == block["block_in.encoder_hidden_states"].shape
        and block["block_out.1"].shape == block["block_in.hidden_states"].shape,
    )
    _check(
        "the block actually transformed the image stream",
        _max_abs(block["block_out.1"], block["block_in.hidden_states"]) > 1e-3,
    )
    # --- payload: the outputs are real activations -----------------------------------------
    out_img = block["block_out.1"]
    in_img = block["block_in.hidden_states"]
    branches = _cfg_branches(block)
    n = out_img.shape[1] // branches
    # The image tokens are a raster of the latent grid, so a real activation map varies smoothly
    # across neighbours; moment-matched noise sits at the Gaussian ~1.13. Works at ANY branch
    # count, so the CFG-off path keeps a payload discriminator on this tensor.
    if "geometry" in block:
        gh, gw = int(block["geometry"][0]) // 16, int(block["geometry"][1]) // 16
        if n == gh * gw:
            tv = _tv_ratio(out_img[0, :n].reshape(gh, gw, -1).transpose(2, 0, 1))
            _check(
                "block_out.1 varies smoothly across the latent grid (a real activation map)",
                tv < 0.7,
                f"TV/std={tv:.4f} (white noise ~1.13)",
            )
    # With CFG on, the two image-stream halves start identical and see the same timestep
    # embedding, so the output halves stay near-collinear; independent noise scores ~0.
    if branches == 2 and n > 0 and _max_abs(in_img[:, :n], in_img[:, n:]) == 0.0:
        align = _corr(out_img[0, :n], out_img[0, n:])
        _check(
            "block_out.1 cond/uncond halves are near-collinear (same input, different text)",
            align > 0.9,
            f"corr={align:.4f}",
        )
    # NR-MMDiT activations are massively heavy-tailed (outlier features); Gaussian noise is 3.0
    # by construction, so this rejects a moment-matched randomization of either stream.
    for key in ("block_out.0", "block_out.1"):
        kurt = _kurtosis(block[key])
        _check(
            f"{key} is a heavy-tailed activation tensor, not Gaussian noise",
            kurt > 10.0,
            f"kurtosis={kurt:.1f} (Gaussian = 3.0)",
        )


def _check_msrope_payload(
    block: dict[str, np.ndarray],
    dit: dict[str, np.ndarray] | None,
    rope_re: np.ndarray,
    rope_im: np.ndarray,
) -> None:
    """Recompute the whole msrope table in numpy and pin the batch_cfg frame index.

    Weightless: `img_shapes` + `theta=10000` + `axes_dim=[16,56,56]` determine every entry.
    """
    geometry = block.get("geometry")
    cu = block.get("block_in.img_cu_lens")
    if geometry is None or cu is None:
        return
    gh, gw = int(geometry[0]) // 16, int(geometry[1]) // 16
    seg_lens = np.diff(cu.astype(np.int64)).tolist()
    if not seg_lens or any(length != gh * gw for length in seg_lens):
        _check(
            "packed image segments are one gh*gw latent grid each",
            False,
            f"cu_lens={cu.tolist()} vs gh*gw={gh * gw}",
        )
        return
    shapes = np.array([[1, gh, gw]] * len(seg_lens), dtype=np.int32)
    _check(
        "the packed segment count matches the CFG branch count",
        len(seg_lens) == _cfg_branches(block),
        f"cfg={float(block['cfg'][0]):g} -> {_cfg_branches(block)} branch(es), "
        f"{len(seg_lens)} segment(s)",
    )
    if dit is not None and "img_shapes" in dit:
        _check(
            "block img_cu_lens agrees with the DiT golden's img_shapes",
            dit["img_shapes"].shape == shapes.shape
            and np.array_equal(dit["img_shapes"].astype(np.int64), shapes.astype(np.int64)),
            f"dit={dit['img_shapes'].tolist()} vs derived={shapes.tolist()}",
        )

    table = _msrope_table(shapes)
    if table.shape != rope_re.shape:
        _check(
            "msrope table recomputes to the golden's shape",
            False,
            f"recomputed {table.shape} vs golden {rope_re.shape}",
        )
        return
    delta = max(_max_abs(table.real, rope_re), _max_abs(table.imag, rope_im))
    _check(
        "msrope table RECOMPUTES from img_shapes + theta 10000 + axes_dim [16,56,56]",
        delta < 1e-5,
        f"max_abs={delta:.2e} (f32 rounding)",
    )

    # The CONFIRMED batch_cfg finding, stated as its own assertion: the frame frequencies come
    # from each segment's *position* in `d_img_shapes`, so segment j is rotated at frame index j.
    # Segment 0 is therefore exact identity and, WHEN CFG IS ON, the duplicated uncond segment is
    # rotated at 1. With CFG off there is one segment and the same rule still has content: the
    # sole segment must be exact identity, which is asserted below either way.
    frame_dim = ROPE_AXES_DIM[0]
    n_frame = frame_dim // 2
    inv = 1.0 / np.power(float(ROPE_THETA), np.arange(0, frame_dim, 2, dtype=np.float64) / frame_dim)
    ok, detail = True, []
    for idx, start in enumerate(np.cumsum([0, *seg_lens[:-1]])):
        rows = slice(int(start), int(start) + seg_lens[idx])
        want_re, want_im = np.cos(idx * inv), np.sin(idx * inv)
        d = max(
            float(np.abs(rope_re[rows, :n_frame] - want_re).max()),
            float(np.abs(rope_im[rows, :n_frame] - want_im).max()),
        )
        ok = ok and d < 1e-5
        detail.append(f"seg{idx}@frame{idx}: {d:.1e}")
    _check(
        "msrope rotates segment j at FRAME INDEX j"
        + (" (cond=identity, uncond dup=frame 1)" if len(seg_lens) >= 2 else " (CFG off: 1 segment)"),
        ok,
        ", ".join(detail),
    )
    _check(
        "…segment 0's frame slots are EXACTLY identity (1, 0)",
        float(np.abs(rope_re[: seg_lens[0], :n_frame] - 1.0).max()) == 0.0
        and float(np.abs(rope_im[: seg_lens[0], :n_frame]).max()) == 0.0,
    )
    # h/w are identical across the duplicated segments — only the frame slots move.
    if len(seg_lens) >= 2 and seg_lens[0] == seg_lens[1]:
        n0 = seg_lens[0]
        spatial = max(
            _max_abs(rope_re[:n0, n_frame:], rope_re[n0 : 2 * n0, n_frame:]),
            _max_abs(rope_im[:n0, n_frame:], rope_im[n0 : 2 * n0, n_frame:]),
        )
        _check(
            "…and the h/w slots are untouched by the duplication",
            spatial == 0.0,
            f"h/w max_abs={spatial:.4f}",
        )


def check_e2e(
    e2e: dict[str, np.ndarray], noise: dict[str, np.ndarray] | None, dit: dict[str, np.ndarray] | None
) -> None:
    sig4 = e2e["sigmas_4"]
    _check(
        "the 4-step schedule is the static-shift 6·s/(1+5·s) ladder + terminal 0",
        np.allclose(sig4, [1.0, 0.94736844, 0.85714287, 0.66666669, 0.0], atol=1e-6),
        f"{[round(float(v), 6) for v in sig4]}",
    )
    _check("sigmas end at 0", float(sig4[-1]) == 0.0)
    _check(
        "sigmas are strictly decreasing",
        bool(np.all(np.diff(sig4.astype(np.float64)) < 0)),
    )
    branches = _cfg_branches(e2e)
    if noise is not None and "traj_step0" in e2e and "gs_noise_bf16" in noise:
        traj = e2e["traj_step0"]
        gs = _tokens_of(noise["gs_noise_bf16"])
        _check(
            "the trajectory carries one latent per CFG branch",
            traj.shape[1] == branches * gs.shape[0],
            f"{traj.shape[1]} tokens = {branches} x {gs.shape[0]}",
        )
        start = traj[0, : traj.shape[1] // branches]
        _check(
            "the denoise STARTS from the Gaussian-Shading latent (not plain randn)",
            start.shape == gs.shape and _max_abs(start, gs) == 0.0,
            f"vs plain randn: max_abs={_max_abs(start, _tokens_of(noise['plain_randn'])):.4f}"
            if start.shape == gs.shape
            else f"shape {tuple(start.shape)} vs {tuple(gs.shape)}",
        )
    if "traj_step0" in e2e and "traj_step1" in e2e:
        _check(
            "step 1 differs from step 0 (the sampler advanced)",
            _max_abs(e2e["traj_step0"], e2e["traj_step1"]) > 1e-3,
        )
    _check(
        "the decoded image is not a blank refusal placeholder",
        int(e2e["image_u8"].min()) < 250 and float(e2e["image_u8"].std()) > 5.0,
        f"min={int(e2e['image_u8'].min())} std={float(e2e['image_u8'].std()):.1f}",
    )
    if dit is not None:
        _check(
            "the DiT golden was captured at e2e step 0 (same latent)",
            _max_abs(dit["dit_in.img"], e2e["traj_step0"]) == 0.0,
        )
    check_final_latent("e2e", e2e)


def check_final_latent(stage: str, bundle: dict[str, np.ndarray]) -> None:
    """`final_tokens` <-> `final_latent` reshape identity + "this latent made that image"."""
    if not {"final_tokens", "final_latent", "geometry", "image_u8"} <= set(bundle):
        return
    tokens, latent = bundle["final_tokens"], bundle["final_latent"]
    gh, gw = int(bundle["geometry"][0]) // 16, int(bundle["geometry"][1]) // 16
    rebuilt = np.ascontiguousarray(tokens.reshape(1, gh, gw, -1).transpose(0, 3, 1, 2))
    _check(
        f"{stage}: final_latent is final_tokens reshaped `(h w) c -> c h w` (bit-exact)",
        rebuilt.shape == latent.shape and _max_abs(rebuilt, latent) == 0.0,
        f"tokens={tuple(tokens.shape)} -> {tuple(rebuilt.shape)} vs {tuple(latent.shape)}",
    )
    tracked = _latent_image_channels(latent, bundle["image_u8"])
    _check(
        f"{stage}: final_latent is the latent that DECODED to image_u8",
        tracked >= 4,
        f"{tracked}/{latent.shape[1]} channels |corr|>0.5 vs the 16x block-mean luma (random: 0)",
    )
    tv = _tv_ratio(latent)
    _check(
        f"{stage}: final_latent has image-like spatial structure, not white noise",
        tv < 0.9,
        f"TV/std={tv:.4f} (white noise ~1.13)",
    )


def check_edit(edit: dict[str, np.ndarray], noise: dict[str, np.ndarray] | None) -> None:
    if "seq_step0" not in edit:
        return
    seq = edit["seq_step0"]
    n_tgt = int(edit["target_tokens"][0])
    branches = _cfg_branches(edit)
    half = seq.shape[1] // branches
    # `generate_edits` only builds the negative branch above cfg 1 (`pipeline.py:535`), so with
    # CFG off the stream is [target, ref…] once rather than twice.
    _check(
        f"the edit stream is [target, ref] per sample, x{branches} CFG branch(es)",
        half == 2 * n_tgt and seq.shape[1] == branches * half,
        f"per-branch tokens={half}, target tokens={n_tgt}, cfg={float(edit['cfg'][0]):g}",
    )
    if branches == 2:
        _check(
            "batch_cfg duplicated the edit stream exactly (cond half == uncond half)",
            _max_abs(seq[:, :half], seq[:, half:]) == 0.0,
        )
    if noise is not None and "gs_noise_bf16" in noise and half == 2 * n_tgt:
        target_slice = seq[0, :n_tgt]
        ref_slice = seq[0, n_tgt : 2 * n_tgt]
        _check(
            "segment 0 is the NOISY TARGET (== the Gaussian-Shading latent) — target is FIRST",
            _max_abs(target_slice, _tokens_of(noise["gs_noise_bf16"])) == 0.0,
        )
        _check(
            "segment 1 is a DIFFERENT tensor (the clean reference latent)",
            _max_abs(ref_slice, _tokens_of(noise["gs_noise_bf16"])) > 0.1,
            f"max_abs={_max_abs(ref_slice, _tokens_of(noise['gs_noise_bf16'])):.4f}",
        )
    if "seq_step1" in edit and half == 2 * n_tgt:
        _check(
            "refs stay CLEAN across steps while the target moves",
            _max_abs(edit["seq_step0"][0, n_tgt : 2 * n_tgt], edit["seq_step1"][0, n_tgt : 2 * n_tgt]) == 0.0
            and _max_abs(edit["seq_step0"][0, :n_tgt], edit["seq_step1"][0, :n_tgt]) > 1e-3,
        )
    if "img_shapes" in edit and "geometry" in edit:
        # GAP 5, asserted rather than merely printed. The rows are per-segment (frame COUNT,
        # h, w) — the msrope FRAME INDEX is the row's position, so under batch_cfg the segment
        # order is [target, ref…] then the duplicate, i.e. frames 0..1 then 2..3. With CFG off
        # there is no duplicate and the list is just [target, ref…] at frames 0..n_refs.
        shapes = edit["img_shapes"]
        gh, gw = int(edit["geometry"][0]) // 16, int(edit["geometry"][1]) // 16
        n_refs = half // n_tgt - 1 if n_tgt else 0
        n_rows = branches * (1 + n_refs)
        expect = np.array([[1, gh, gw]] * n_rows, dtype=np.int64)
        doubled = " doubled by batch_cfg" if branches == 2 else " (cfg <= 1, undoubled)"
        _check(
            f"img_shapes is [target, ref x{n_refs}]{doubled} — "
            f"{n_rows} rows of (1, {gh}, {gw})",
            shapes.shape == expect.shape and np.array_equal(shapes.astype(np.int64), expect),
            f"img_shapes={shapes.tolist()}",
        )
        _check(
            "…so the msrope frame INDEX runs 0..n-1 over that list (target 0, ref_j j)",
            shapes.shape[0] == n_rows and n_refs >= 1,
            f"frame indices={list(range(shapes.shape[0]))}",
        )
    _check(
        "the edited image is not a blank refusal placeholder",
        int(edit["image_u8"].min()) < 250 and float(edit["image_u8"].std()) > 5.0,
        f"min={int(edit['image_u8'].min())} std={float(edit['image_u8'].std()):.1f}",
    )
    check_final_latent("edit", edit)


# --------------------------------------------------------------------------------- self-test
#
# A checker that cannot be shown to REJECT a wrong bundle is worth nothing: 39 structural
# invariants used to pass against goldens whose payload tensors had been swapped for noise. So
# the discrimination claim is executable — `--self-test` corrupts a scratch copy one tensor at a
# time and asserts this script rejects every one. Same pattern as
# `candle-gen/scripts/check-gen-core-skew.sh --self-test`.
#
# Every mutation is moment-matched (identical shape, dtype, mean and std) and, where a structural
# invariant would otherwise trivially catch it, deliberately repaired to keep that invariant
# intact — `enc_latent` is kept equal to `enc_mean`, `final_latent` is re-derived from the
# corrupted `final_tokens` so the reshape identity still holds, the corrupted decode is clipped
# back into [-1, 1], the randomized msrope keeps unit modulus and matched spatial halves, and the
# rescaled TE keeps its `*_txt == *_hidden_full[drop:]` slice relation.


def _moment_matched(rng: np.random.Generator, a: np.ndarray) -> np.ndarray:
    x = rng.standard_normal(a.shape)
    x = (x - x.mean()) / x.std()
    return (x * float(a.std()) + float(a.mean())).astype(a.dtype)


def _mutations(rng: np.random.Generator) -> dict[str, tuple[str, object]]:
    def block_out(t):
        for key in ("block_out.0", "block_out.1"):
            t[key] = _moment_matched(rng, t[key])

    def dit_out(t):
        t["dit_out"] = _moment_matched(rng, t["dit_out"])

    def dec(t):
        t["dec_from_latent"] = np.clip(_moment_matched(rng, t["dec_from_latent"]), -1.0, 1.0)

    def enc(t):
        z = _moment_matched(rng, t["enc_mean"])
        t["enc_mean"], t["enc_latent"] = z, z.copy()

    def final(t):
        tokens = _moment_matched(rng, t["final_tokens"])
        gh, gw = int(t["geometry"][0]) // 16, int(t["geometry"][1]) // 16
        t["final_tokens"] = tokens
        t["final_latent"] = np.ascontiguousarray(
            tokens.reshape(1, gh, gw, -1).transpose(0, 3, 1, 2)
        )

    def msrope(t):
        n = t["block_in.image_rotary_emb_re"].shape[0] // 2
        width = t["block_in.image_rotary_emb_re"].shape[1]
        cond = rng.uniform(-np.pi, np.pi, size=(n, width))
        cond[:, :8] = 0.0  # frame slots left at identity in the cond half
        unc = cond.copy()  # spatial slots MATCHED across the halves
        unc[:, :8] = rng.uniform(0.05, 0.5, size=(n, 8))
        angles = np.concatenate([cond, unc])
        t["block_in.image_rotary_emb_re"] = np.cos(angles).astype(np.float32)
        t["block_in.image_rotary_emb_im"] = np.sin(angles).astype(np.float32)

    def rescale_te(factor: float):
        keys = ("gen_hidden_full", "gen_txt", "gen_vec", "neg_txt", "neg_vec",
                "edit_hidden_full", "edit_txt", "edit_vec")

        def apply(t):
            for key in keys:
                t[key] = (t[key].astype(np.float64) * factor).astype(np.float32)

        return apply

    return {
        "block_out.{0,1} -> noise": ("dit_block", block_out),
        "dit_out -> noise": ("dit", dit_out),
        "dec_from_latent -> noise": ("vae", dec),
        "enc_mean + enc_latent -> noise": ("vae", enc),
        "final_latent + final_tokens -> noise": ("e2e", final),
        "msrope angles -> random (unit modulus kept)": ("dit_block", msrope),
        "TE captured at a wrong layer (x3.7)": ("te", rescale_te(3.7)),
        "TE captured at a wrong layer (x1.3)": ("te", rescale_te(1.3)),
    }


def _self_test() -> int:
    import shutil
    import subprocess
    import tempfile

    from safetensors.numpy import save_file

    missing = [s for s in _STEMS if not (GOLDEN / f"mage_flow_{s}_golden.safetensors").is_file()]
    if missing:
        print(f"--self-test needs the full bundle; missing {missing} under {GOLDEN}")
        return 1

    def verdict(directory: Path) -> tuple[int, str]:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--golden", str(directory)],
            capture_output=True,
            text=True,
        )
        tail = next(
            (ln for ln in proc.stdout.splitlines() if ln.startswith(("FAILED ", "all "))), ""
        )
        return proc.returncode, tail

    rng = np.random.default_rng(20260724)
    rows: list[tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "bundle"

        def reset() -> None:
            shutil.rmtree(scratch, ignore_errors=True)
            scratch.mkdir(parents=True)
            for f in GOLDEN.glob("mage_flow_*.safetensors"):
                shutil.copy2(f, scratch / f.name)

        reset()
        code, tail = verdict(scratch)
        rows.append(("pristine copy (must PASS)", code == 0, tail))

        for name, (stem, mutate) in _mutations(rng).items():
            reset()
            path = scratch / f"mage_flow_{stem}_golden.safetensors"
            tensors = load_file(str(path))
            mutate(tensors)
            save_file(tensors, str(path))
            code, tail = verdict(scratch)
            rows.append((name, code != 0, tail))

        empty = Path(tmp) / "empty"
        empty.mkdir()
        rows.append(("empty directory (must FAIL)", verdict(empty)[0] != 0, ""))

    width = max(len(name) for name, _, _ in rows)
    for name, ok, tail in rows:
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name:{width}s}  {tail}")
    bad = sum(1 for _, ok, _ in rows if not ok)
    print()
    if bad:
        print(f"SELF-TEST FAILED: {bad}/{len(rows)} cases were not discriminated")
        return 1
    print(f"self-test: all {len(rows) - 1} corruptions rejected, the pristine bundle accepted")
    return 0


def main(argv: list[str] | None = None) -> int:
    global GOLDEN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden",
        default=None,
        help="directory holding the mage_flow_*_golden.safetensors bundle (default tools/golden)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="skip absent stages instead of failing (the goldens are gitignored)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="prove the invariants discriminate: corrupt a scratch copy and expect rejection",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        if args.golden:
            GOLDEN = Path(args.golden).expanduser()
        print(f"self-test against {GOLDEN}")
        return _self_test()
    if args.golden:
        GOLDEN = Path(args.golden).expanduser()
    print(f"golden dir: {GOLDEN}")

    bundles = {stem: _load(stem, args.allow_missing) for stem in _STEMS}
    noise, vae, te = bundles["noise"], bundles["vae"], bundles["te"]
    dit, block, e2e, edit = bundles["dit"], bundles["dit_block"], bundles["e2e"], bundles["edit"]

    for label, bundle, run in (
        ("noise", noise, lambda b: check_noise(b)),
        ("vae", vae, lambda b: check_vae(b, edit)),
        ("te", te, lambda b: check_te(b, dit)),
        ("dit", dit, lambda b: check_dit(b, e2e)),
        ("dit_block", block, lambda b: check_dit_block(b, dit)),
        ("e2e", e2e, lambda b: check_e2e(b, noise, dit)),
        ("edit", edit, lambda b: check_edit(b, noise)),
    ):
        if bundle is None:
            continue
        print(f"{label}:")
        run(bundle)

    print()
    if _FAILURES:
        print(f"FAILED {len(_FAILURES)}/{_CHECKS}: " + ", ".join(_FAILURES))
        return 1
    if _CHECKS == 0:
        print(f"NOTHING VERIFIED: no goldens found under {GOLDEN}. Run dump_mage_flow_golden.py.")
        return 1
    print(f"all {_CHECKS} golden invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
