# SC-20686 persistent K/V campaign transport

The campaign adapter runs the one registered FLUX.2 Klein edit route and all five registered Wan
routes. For every normal and cancellation arm it creates a private `events.jsonl` file, passes that
path to the product entrypoint with `--sc20686-events`, and seals the exact event transcript as a
separate bundle artifact. Provider stdout and stderr are retained only as diagnostics: progress
output, including carriage-return updates, is never parsed as campaign evidence.

The Wan entrypoints are wired to the product-owned observer after each real route has bound its
snapshot-backed geometry. With observation off, the ownership hooks remain inactive and do not
allocate campaign evidence or retain cache ids. The adapter rejects a missing, non-JSONL, or
carriage-return-containing event transcript before reducing a campaign row.

FLUX.2 Klein edit is an evidence-based no-go for persistent-reference-K/V productization in this
campaign. Its `DoubleAttention` path projects reference K/V for each denoise evaluation and joins
it into dense attention; there is no persistent reference K/V boundary or packed reader to promote.
The observer records that transient reference-slice work so a live campaign can establish the
no-go without representing an ordinary attention allocation as a promotable cache.
