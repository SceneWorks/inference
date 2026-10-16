# The story-local real-weight evidence (sc-26234 AC2 + the native half of AC1): the pinned package
# through the port, each case under the external footprint guard
# (scripts/ci/iris_restore_footprint_guard.py; cap 64 GB, 30 min wall — the bf16 package is ~20 GB
# resident plus the 1024-squared attention/decoder activations). The component cases compare
# against the COMMITTED upstream CPU reference `flux3_real_golden_64.safetensors`
# (tools/dump_flux3_realweight.py at the pinned package); the render case is the ordinary catalog
# route (`provider_registry().load("flux3_klein")` -> one text-to-image render at the dispatched
# size / steps) and keeps the PNG + its JSON record in the evidence artifact. Report-only in the
# sense that no number gates anything else; the job is red when a case fails or does not run.
bin="$(cat "$RUNNER_TEMP/flux3-klein-test-binary")"
failed=0
run_guarded() { # $1 cap GB, $2 wall s, $3 test path
  local t="$3" log="$FLUX3_OUT/${3##*::}.log"
  python3.12 scripts/ci/iris_restore_footprint_guard.py "$1" "$2" \
    "$bin" "$t" --ignored --exact --nocapture --test-threads 1 > "$log" 2>&1 || failed=1
  cat "$log"
  grep -q "test result: ok. 1 passed" "$log" || { echo "::error::$t did not run exactly one passing test"; failed=1; }
}
export FLUX3_REAL_SIZE="${FLUX3_REAL_SIZE:-1024}"
export FLUX3_REAL_STEPS="${FLUX3_REAL_STEPS:-16}"
run_guarded 64 1800 real_weights::real_package_config_is_the_release
run_guarded 64 1800 real_weights::real_text_encoder_matches_upstream
run_guarded 64 1800 real_weights::real_autoencoder_decode_matches_upstream
run_guarded 64 1800 real_weights::real_dit_forward_matches_upstream
run_guarded 64 1800 real_weights::real_e2e_64_matches_upstream
run_guarded 64 1800 real_weights::real_distilled_text_to_image_renders
# sc-26238: the prompt upsampler alone (the 9B VLM resident, ~19 GB), two real rewrites.
run_guarded 64 1800 real_weights::real_prompt_upsampler_rewrites_the_readme_request
ls -la "$FLUX3_OUT"
exit "$failed"
