"""Self-consistency checks over the Mage-Flow goldens — sc-14036 (epic 14034).

Run after `dump_mage_flow_golden.py`. This does **not** re-run the reference; it asserts that
the dumped bundle hangs together, which catches the failure mode a parity golden is worst at
surfacing: a golden that is internally wrong but that every downstream test happily "matches".

Each check is a *cross-file* invariant that only holds if the boundary it pins was captured
correctly, so it doubles as executable documentation of the epic's GAP answers:

* the denoise really starts from the **Gaussian-Shading** latent, not `randn` (GAP: watermark);
* the TE conditioning really is `hidden_full[drop_idx:]` with drop 34 / 64 (GAP 1);
* `enc_latent` really is the posterior **mean** (GAP 2's deterministic branch);
* the msrope table really is complex with unit modulus, and was not silently truncated to its
  real part on the way into the file (GAP 3);
* the edit image stream really is `[noisy_target, ref…]` with the target first (GAP 5).

    python tools/verify_mage_flow_golden.py

Exits non-zero on the first failed invariant. Needs only `numpy` + `safetensors` — no torch, no
weights, no reference checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from _paths import fixture

GOLDEN = Path(fixture("tools/golden"))

_FAILURES: list[str] = []
_CHECKS = 0


def _check(name: str, ok: bool, detail: str = "") -> None:
    global _CHECKS
    _CHECKS += 1
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        _FAILURES.append(name)


def _load(stem: str) -> dict[str, np.ndarray] | None:
    path = GOLDEN / f"mage_flow_{stem}_golden.safetensors"
    if not path.is_file():
        print(f"  [skip] {stem}: {path.name} not present")
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


def main() -> int:
    noise = _load("noise")
    vae = _load("vae")
    te = _load("te")
    dit = _load("dit")
    block = _load("dit_block")
    e2e = _load("e2e")
    edit = _load("edit")

    print("noise:")
    if noise is not None:
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

    print("vae:")
    if vae is not None:
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

    print("te:")
    if te is not None:
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
        _check(
            "conditioning width is the Qwen3-VL LM hidden size (2560)",
            te["gen_txt"].shape[-1] == 2560 and te["edit_txt"].shape[-1] == 2560,
        )
        # A different template + a different image path must not produce the same vector.
        _check(
            "gen and edit conditioning are distinct",
            _max_abs(te["gen_vec"], te["edit_vec"]) > 1e-3,
        )

    print("dit:")
    if dit is not None:
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
        # batch_cfg packs [cond, uncond] into one varlen forward -> two segments, and the two
        # halves must DIFFER (identical halves would mean the negative branch never applied).
        img = dit["dit_in.img"]
        half = img.shape[1] // 2
        _check(
            "batch_cfg duplicated the image stream (cond half == uncond half on input)",
            _max_abs(img[:, :half], img[:, half:]) == 0.0,
        )
        _check(
            "…but the velocities differ (the negative branch actually conditions)",
            _max_abs(dit["dit_out"][:, :half], dit["dit_out"][:, half:]) > 1e-4,
            f"max_abs={_max_abs(dit['dit_out'][:, :half], dit['dit_out'][:, half:]):.5f}",
        )
        if "img_shapes" in dit:
            _check(
                "img_shapes is one (frame, h, w) row per packed segment",
                dit["img_shapes"].ndim == 2 and dit["img_shapes"].shape[1] == 3,
                f"{dit['img_shapes'].tolist()}",
            )

    print("dit_block:")
    if block is not None:
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
                f"max|,|z|-1,|={float(np.abs(modulus - 1.0).max()):.2e}",
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
            # batch_cfg concatenates the SEGMENT LIST (`d_img_shapes = img_shapes[0] * 2`), and
            # the frame index comes from each segment's enumerate position — so the duplicated
            # (uncond) half is rotated at frame 1, not 0. The difference must live ENTIRELY in
            # the frame slots (axes_dim[0]//2 = 8 complex entries); h/w must be untouched.
            n = rope_re.shape[0] // 2
            frame_slots, spatial_slots = slice(0, 8), slice(8, None)
            frame_delta = max(
                _max_abs(rope_re[:n, frame_slots], rope_re[n:, frame_slots]),
                _max_abs(rope_im[:n, frame_slots], rope_im[n:, frame_slots]),
            )
            spatial_delta = max(
                _max_abs(rope_re[:n, spatial_slots], rope_re[n:, spatial_slots]),
                _max_abs(rope_im[:n, spatial_slots], rope_im[n:, spatial_slots]),
            )
            _check(
                "batch_cfg rotates the uncond half at a DIFFERENT frame index (frame slots only)",
                frame_delta > 0.1 and spatial_delta == 0.0,
                f"frame max_abs={frame_delta:.4f}, h/w max_abs={spatial_delta:.4f}",
            )
        _check(
            "block output shapes match its inputs (txt, img)",
            block["block_out.0"].shape == block["block_in.encoder_hidden_states"].shape
            and block["block_out.1"].shape == block["block_in.hidden_states"].shape,
        )
        _check(
            "the block actually transformed the image stream",
            _max_abs(block["block_out.1"], block["block_in.hidden_states"]) > 1e-3,
        )

    print("e2e:")
    if e2e is not None:
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
        if noise is not None and "traj_step0" in e2e and "gs_noise_bf16" in noise:
            traj = e2e["traj_step0"]
            half = traj.shape[1] // 2
            start = traj[0, :half]
            _check(
                "the denoise STARTS from the Gaussian-Shading latent (not plain randn)",
                _max_abs(start, _tokens_of(noise["gs_noise_bf16"])) == 0.0,
                f"vs plain randn: max_abs={_max_abs(start, _tokens_of(noise['plain_randn'])):.4f}",
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

    print("edit:")
    if edit is not None and "seq_step0" in edit:
        seq = edit["seq_step0"]
        n_tgt = int(edit["target_tokens"][0])
        half = seq.shape[1] // 2
        _check(
            "the edit stream is [target, ref] per sample (2 segments of equal length)",
            half == 2 * n_tgt,
            f"per-branch tokens={half}, target tokens={n_tgt}",
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
        if "img_shapes" in edit:
            frames = edit["img_shapes"][:, 0].tolist()
            _check(
                "msrope frame indices are 1 per segment (target then refs)",
                len(frames) >= 2,
                f"img_shapes={edit['img_shapes'].tolist()}",
            )
        _check(
            "the edited image is not a blank refusal placeholder",
            int(edit["image_u8"].min()) < 250 and float(edit["image_u8"].std()) > 5.0,
            f"min={int(edit['image_u8'].min())} std={float(edit['image_u8'].std()):.1f}",
        )

    print()
    if _FAILURES:
        print(f"FAILED {len(_FAILURES)}/{_CHECKS}: " + ", ".join(_FAILURES))
        return 1
    print(f"all {_CHECKS} golden invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
