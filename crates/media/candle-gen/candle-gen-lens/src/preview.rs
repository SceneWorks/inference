//! Lens per-step latent previews (epic 16948, sc-16955) — a thin re-export of the **already-committed**
//! FLUX.2 32-channel seam ([`candle_gen_flux2::preview`]), the exact shape [`crate::vae`] takes over
//! [`candle_gen_flux2::vae`].
//!
//! ## Why there is nothing to implement here
//!
//! The Lens latent space *is* the FLUX.2 one: 32-channel `AutoencoderKLFlux2`, 2×2 patchify into the
//! 128-channel transformer space, BatchNorm-stats normalization — and Lens loads that VAE into the very
//! same [`Flux2Vae`](candle_gen_flux2::vae::Flux2Vae) type. Its DiT emits the same packed
//! `[1, h·w, 128]` token sequence, and [`crate::vae::decode`] folds it onto the same `[1, 128, h, w]`
//! grid before the shared decode. So the preview projection — unpack, bn de-normalize, 2×2 unpatchify,
//! then the epic-16624 fit — is byte-for-byte the FLUX.2 one, and a Lens-local copy of those constants
//! would fork one latent space into two colour maps.
//!
//! The reuse is grounded in tensor bytes, not in the shared Rust type. The `vae/` file every Lens
//! snapshot publishes — `SceneWorks/Lens`, `Comfy-Org/Lens`, and every tier of the `lens-mlx` /
//! `lens-turbo-mlx` re-hosts — is one f32 container, SHA-256 `d64f3a68…c4b5`, 336,213,556 bytes, whose
//! **250** learned tensors (84,046,371 values) round exactly, round-to-nearest-even, onto the bf16
//! container the fit was measured against (`black-forest-labs/FLUX.2-klein-9B`, `ca70d220…0f04`). Two
//! widths of one checkpoint, not two fine-tunes. `tests/preview_real_weights.rs` re-derives that per
//! snapshot; the full record is `docs/migration/evidence/sc-16955-flux2-candle-preview.md`.
//!
//! ## What each route hands the hook
//!
//! Both shipped render lanes — the resident `render` and the sequential `render_sequential` — build a
//! [`hook`] per image over `(latent_h, latent_w)`, the same grid they hand [`crate::vae::decode`]. The
//! trainer's periodic sample render (`crate::training`) deliberately passes `None`: it drives the
//! sampler from a synthetic request with no `PreviewSink` and delivers a finished
//! `TrainingProgress::Sample` image rather than a live stream.
//!
//! CFG never reaches a frame: `Pipeline::denoise` fuses the `[cond, uncond]` batch **inside** its
//! predict closure and `cfg_rescale` blends it back to one velocity before returning, so the running
//! latent is batch 1 at every step.

pub use candle_gen_flux2::preview::{
    hook, project_packed_tokens, project_raw_latents, PACKED_LATENT_CHANNELS,
    PREVIEW_LATENT_CHANNELS,
};

#[cfg(test)]
mod tests {

    /// `candle_gen::run_flow_sampler`'s argument count before the predict closure. Pinned so a
    /// signature change — or a scanner mis-split — fails this inventory loudly instead of quietly
    /// shifting which argument "the one before the closure" names.
    const SAMPLER_ARGUMENTS_BEFORE_PREDICT: usize = 8;

    /// The arguments of every `run_flow_sampler` call in `source`, one entry per call site, covering
    /// the arguments **before** the predict closure — the window the `preview` argument sits in.
    ///
    /// Ported from sc-16950's Krea inventory. The window is bounded by the call's own bracket balance
    /// and ends at the first top-level `|`; it deliberately does not key off a closure parameter name,
    /// because a route naming that parameter something else would otherwise widen the window to the
    /// next call site (or to end of file) and let any `Some(&preview)` in the swallowed text — prose
    /// included — satisfy a route that was left dark. A missing bound is a failure, not a wider window.
    ///
    /// The match is textual, so writing the driver's name followed by an open paren in prose is read
    /// as a call site: name it without the paren in comments.
    fn sampler_call_sites(file: &str, source: &str) -> Vec<Vec<String>> {
        const CALL: &str = "run_flow_sampler(";
        let mut sites = Vec::new();
        let mut cursor = 0usize;
        while let Some(at) = source[cursor..].find(CALL) {
            let args_start = cursor + at + CALL.len();
            sites.push(sampler_call_arguments(
                file,
                sites.len(),
                &source[args_start..],
            ));
            cursor = args_start;
        }
        sites
    }

    /// The comma-separated top-level arguments of one call, given everything after its open paren.
    fn sampler_call_arguments(file: &str, index: usize, rest: &str) -> Vec<String> {
        let site = format!("{file}: run_flow_sampler call #{index}");
        let normalize = |text: &str| text.split_whitespace().collect::<Vec<_>>().join(" ");

        let mut args: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut depth = 1usize;
        let mut chars = rest.chars().peekable();
        while let Some(ch) = chars.next() {
            match ch {
                // Comments are not code: a `(` or a `|` inside one must not move the scan.
                '/' if chars.peek() == Some(&'/') => {
                    for c in chars.by_ref() {
                        if c == '\n' {
                            break;
                        }
                    }
                    current.push(' ');
                }
                '/' if chars.peek() == Some(&'*') => {
                    chars.next();
                    let (mut nesting, mut prev) = (1usize, '\0');
                    for c in chars.by_ref() {
                        match (prev, c) {
                            ('/', '*') => (nesting, prev) = (nesting + 1, '\0'),
                            ('*', '/') => {
                                nesting -= 1;
                                prev = '\0';
                                if nesting == 0 {
                                    break;
                                }
                            }
                            _ => prev = c,
                        }
                    }
                    assert_eq!(nesting, 0, "{site} has an unterminated block comment");
                    current.push(' ');
                }
                // Nor are string literals.
                '"' => {
                    current.push('"');
                    let mut escaped = false;
                    let mut closed = false;
                    for c in chars.by_ref() {
                        current.push(c);
                        if escaped {
                            escaped = false;
                        } else if c == '\\' {
                            escaped = true;
                        } else if c == '"' {
                            closed = true;
                            break;
                        }
                    }
                    assert!(closed, "{site} has an unterminated string literal");
                }
                '(' | '[' | '{' => {
                    depth += 1;
                    current.push(ch);
                }
                ')' | ']' | '}' => {
                    depth -= 1;
                    assert!(
                        depth > 0,
                        "{site} closes without a predict closure — the scan cannot bound its \
                         preview argument, so no assertion about that argument would mean anything"
                    );
                    current.push(ch);
                }
                ',' if depth == 1 => {
                    args.push(normalize(&current));
                    current.clear();
                }
                // The predict closure's parameter list: the argument window ends here, whatever that
                // parameter is called.
                '|' if depth == 1 => {
                    let trailing = normalize(&current);
                    assert!(
                        trailing.is_empty(),
                        "{site} has unparsed text {trailing:?} between its last argument and the \
                         predict closure — the scan cannot be trusted to have found the preview \
                         argument"
                    );
                    return args;
                }
                _ => current.push(ch),
            }
        }
        panic!("{site} is unterminated: no predict closure and no closing paren before end of file")
    }

    /// Lens has exactly ONE sampler site in its render surface — `Pipeline::denoise`, which both the
    /// resident (`render`) and sequential (`render_sequential`) lanes and the `denoise_for_parity`
    /// seam call — and it forwards its caller's hook.
    ///
    /// That the site is shared is why the *callers* are asserted separately below: a hook forwarded
    /// through one site says nothing about whether every caller supplies one.
    #[test]
    fn the_single_render_sampler_site_forwards_its_callers_hook() {
        let sites = sampler_call_sites("lib.rs", include_str!("lib.rs"));
        assert_eq!(
            sites.len(),
            1,
            "lib.rs: expected exactly 1 sampler call site, found {}. A new render route must pass a \
             preview hook and be named in this inventory (and in the catalog's).",
            sites.len()
        );
        let args = &sites[0];
        assert_eq!(
            args.len(),
            SAMPLER_ARGUMENTS_BEFORE_PREDICT,
            "lib.rs: expected {SAMPLER_ARGUMENTS_BEFORE_PREDICT} arguments before the predict \
             closure, parsed {args:?}"
        );
        // Positional, not `contains`: the preview is the argument immediately before the predict
        // closure, so this cannot be satisfied by the word appearing anywhere else.
        assert_eq!(
            args.last().map(String::as_str),
            Some("preview"),
            "lib.rs does not forward a preview hook: {args:?}"
        );
    }

    /// Both shipped render lanes build a hook; the parity seam deliberately does not.
    ///
    /// `Pipeline::denoise` is reached from three places and only two of them have a `PreviewSink` to
    /// emit into, so "the site forwards a hook" above is only half the fact — a lane that dropped its
    /// `Some(&preview)` for `None` would leave that site's assertion intact.
    #[test]
    fn both_render_lanes_build_a_hook_and_the_parity_seam_passes_none() {
        let source = include_str!("lib.rs");
        let hooks = source
            .matches(concat!("preview::", "hook(&req.preview,"))
            .count();
        assert_eq!(
            hooks, 2,
            "expected one preview hook per shipped render lane (resident + sequential), found \
             {hooks}"
        );

        // Every caller of `Pipeline::denoise`, classified by the argument it passes in the preview
        // slot. The declaration is skipped by name: it is `fn denoise(`, the calls are `.denoise(`.
        let callers: Vec<&str> = source
            .match_indices(".denoise(")
            .map(|(at, _)| {
                let tail = &source[at..];
                let end = tail
                    .find(")?;")
                    .unwrap_or_else(|| panic!("unterminated denoise call at byte {at}"));
                if tail[..end].contains("Some(&preview),") {
                    "hooked"
                } else {
                    "dark"
                }
            })
            .collect();
        assert_eq!(
            callers,
            ["hooked", "hooked", "dark"],
            "the two render lanes must pass a hook and the parity seam must pass None"
        );
    }

    /// The trainer's periodic sample render is the crate's only OTHER sampler site, and it is dark on
    /// purpose. Pinned here as well as in the catalog so the reason lives beside the code, and pinned
    /// as an exact count so a *second* trainer site could not appear unnoticed.
    #[test]
    fn the_trainer_sample_render_is_the_only_deliberately_dark_site() {
        let sites = sampler_call_sites("training.rs", include_str!("training.rs"));
        assert_eq!(
            sites.len(),
            1,
            "training.rs drives the sampler exactly once"
        );
        assert_eq!(
            sites[0].last().map(String::as_str),
            Some("None"),
            "the trainer sample render has no PreviewSink and must pass None: {:?}",
            sites[0]
        );
    }

    /// Lens introduces no fit of its own: the projection it reuses is `candle-gen-flux2`'s, over the
    /// 32-channel FLUX.2 space. A Lens-local copy of those constants would be a second source of truth
    /// for one latent space.
    #[test]
    fn lens_reuses_the_flux2_fit_rather_than_restating_it() {
        assert_eq!(super::PREVIEW_LATENT_CHANNELS, 32);
        assert_eq!(super::PACKED_LATENT_CHANNELS, 128);
        for (file, source) in [
            ("lib.rs", include_str!("lib.rs")),
            ("vae.rs", include_str!("vae.rs")),
            ("training.rs", include_str!("training.rs")),
        ] {
            assert!(
                !source.contains(concat!("RGB_", "FACTORS")) && !source.contains(concat!("RGB_", "BIAS")),
                "{file} holds a Lens-local copy of the FLUX.2 fit, forking one latent space into two \
                 colour maps"
            );
        }
    }

    /// Both registered Lens routes advertise the flag. Weights-free: descriptors only.
    #[test]
    fn both_registered_lens_routes_advertise_preview_support() {
        for descriptor in [crate::descriptor_base(), crate::descriptor_turbo()] {
            assert!(
                descriptor.capabilities.supports_preview,
                "{} must advertise preview support",
                descriptor.id
            );
        }
    }
}
