"""Source-only policy: no H3 complete-model/model-stack execution in ordinary CI.

This test reads the checked-in Rust sources and workflows. It runs no Cargo, builds
nothing, loads no model, and needs no accelerator. It is the enforcement point for
the rule that an H3 test which constructs or runs a complete DiT, a block stack, or
the production pipeline is:

* ``#[ignore]``d with a non-empty reason, so ordinary ``cargo test --lib --tests``
  compiles it but never executes it; and
* selected only by an explicit ``workflow_dispatch`` step that pins the exact number
  of executed tests plus every executed test name, so a rename, deletion, or filter
  change reds the route instead of reporting a vacuous green.

The manifest below is deliberately explicit. Adding a complete-model test means
adding it here and to one manual route; adding an executable H3 step to an ordinary
job reds ``test_ordinary_jobs_never_execute_h3_model_tests``.
"""

import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".github" / "workflows" / "ci.yml"
REAL_WEIGHTS = ROOT / ".github" / "workflows" / "real-weights.yml"
MLX_TESTS = ROOT / "crates/media/mlx-gen/mlx-gen-minimax-h3/tests"
CANDLE_TESTS = ROOT / "crates/media/candle-gen/candle-gen-minimax-h3/tests"
MLX_SRC = ROOT / "crates/media/mlx-gen/mlx-gen-minimax-h3/src"
CANDLE_SRC = ROOT / "crates/media/candle-gen/candle-gen-minimax-h3/src"
MINIMUM_IGNORE_REASON = 20

# Every H3 test that constructs or executes a complete model, a block stack, or the
# production pipeline on synthetic fixtures. `module` is the test-file stem, because
# the consolidated integration binary exposes each file as a module of that name.
GATED = {
    "mlx": {
        "adaln_cache": [
            "the_cache_reproduces_the_reference_modulation_table",
            "the_precomputed_path_reproduces_the_naive_per_step_run",
            "precompute_residual_bounds_the_mutation_floor",
            "a_cache_from_another_schedule_is_stale_and_observably_wrong",
            "a_step_off_by_one_in_the_remap_is_observable",
            "skipping_the_local_to_global_remap_is_observable",
            "the_cache_is_keyed_by_the_whole_schedule",
            "the_cache_size_tracks_the_schedule_and_nothing_else",
            "an_evicted_block_still_runs_but_cannot_project",
            "resident_residency_refuses_to_precompute_and_evict",
            "a_mis_shaped_timestep_embedding_is_rejected_before_eviction",
        ],
        "adaln_evict_memory": [
            "the_evicted_adaln_weights_leave_device_memory",
        ],
        "adapter_topology": [
            "native_pruned_turbo_adapter_residual_matches_reference",
            "body_deltas_are_identical_across_full_and_pruned_topologies",
            "full_topology_adaln_targets_are_refused_specifically_per_topology",
            "real_but_unaddressable_modules_name_the_exclusion",
            "an_adaln_only_file_with_no_body_targets_is_refused_specifically",
            "an_excluded_only_file_with_no_body_targets_is_refused_specifically",
            "an_adaln_only_lokr_with_no_body_targets_is_refused_specifically",
            "turbo_schedule_coordinates_drive_the_pruned_curve_without_a_thousand_scaling",
            "a_genuine_lokr_installs_on_the_pruned_curve_with_the_same_delta",
            "the_comfyui_converted_qkv_namespace_installs_identically_on_the_pruned_curve",
            "the_delta_is_independent_of_the_body_weights_it_folds_onto",
            "the_same_adapter_installs_on_every_partition_of_the_selected_checkpoint",
            "the_exact_dyadic_corpus_folds_exactly_on_both_topologies",
            "the_exact_corpus_is_bit_identical_across_partitions",
            "the_exact_lokr_corpus_folds_exactly_on_the_pruned_curve",
        ],
        "block_window": [
            "a_deferred_load_materializes_no_blocks",
            "every_window_opens_an_independent_view",
            "windowed_and_resident_stacks_agree",
            "the_two_residency_modes_are_mutually_exclusive",
            "the_windowed_adaln_precompute_matches_the_resident_one",
        ],
        "bounded_attention": [
            "a_bounded_plan_threads_the_whole_block_stack",
        ],
        "dit_io": [
            "the_model_embeds_the_context_before_refining_it",
            "a_resident_dit_verifies_every_source_group_before_the_schedule_consumes_it",
            "the_whole_model_reproduces_the_reference_velocity",
        ],
        "dit_parity": [
            "adaln_modulation_matches_the_reference",
            "transformer_block_matches_the_reference",
            "every_norm_epsilon_is_wired_through",
            "reading_the_ffn_halves_gate_first_breaks_parity",
            "the_modality_term_of_the_adaln_index_is_load_bearing",
            "the_adaln_table_is_reshaped_before_it_is_chunked",
            "every_block_weight_is_load_bearing",
            "parity_residuals_bound_the_mutation_floor",
            "weight_mapping_is_exhaustive_over_the_block_stack",
        ],
        "pruned_adaln": [
            "native_pruned_adaln_matches_pinned_reference",
            "full_model_live_coordinates_match_cache_and_refuse_foreign_basis",
            "full_model_detects_pruned_and_checks_pruning_identity",
            "manual_unit_patch_and_mixed_reference_full_forward_matches_resident_control",
        ],
        "turbo_lora": [
            "the_three_published_alphas_fold_at_one_and_one_sixteenth",
            "the_installed_fold_keeps_the_bf16_factor_dtype",
            "the_user_strength_multiplies_the_alpha_fold",
            "every_target_resolves_across_both_stacks",
            "an_unmatched_target_is_loud",
            "a_genuine_lokr_routes_to_the_shared_seam_and_the_turbo_files_do_not",
            "h3_trainer_metadata_cannot_route_mixed_lokr_factors_around_validation",
            "a_file_matching_nothing_names_the_expected_key_space",
            "every_adapter_file_is_forced_and_gpu_verified_at_its_load_boundary",
            "a_peft_blob_rank_disagreeing_with_the_factors_is_refused",
            "the_conversion_leaves_no_comfyui_module_behind",
        ],
    },
    "candle": {
        "adaln_evict_memory": [
            "the_evicted_adaln_weights_leave_memory",
        ],
        "adapter_topology": [
            "native_adapter_folds_on_the_pruned_curve",
            "body_deltas_are_identical_across_full_and_pruned_topologies",
            "full_topology_adaln_targets_are_refused_specifically_per_topology",
            "real_but_unaddressable_modules_name_the_exclusion",
            "an_adaln_only_file_with_no_body_targets_is_refused_specifically",
            "an_excluded_only_file_with_no_body_targets_is_refused_specifically",
            "an_adaln_only_lokr_with_no_body_targets_is_refused_specifically",
            "turbo_schedule_coordinates_drive_the_pruned_curve_without_a_thousand_scaling",
            "a_genuine_lokr_installs_on_the_pruned_curve_with_the_same_delta",
            "the_comfyui_converted_qkv_namespace_installs_identically_on_the_pruned_curve",
            "the_delta_is_independent_of_the_body_weights_it_folds_onto",
            "the_exact_dyadic_corpus_folds_exactly_on_both_topologies",
            "the_exact_lokr_corpus_folds_exactly_on_the_pruned_curve",
        ],
        "cross_backend": [
            "dit_agrees_with_the_mlx_lane",
            "the_dit_cross_backend_floor_is_also_the_mlx_lanes",
            "a_dit_gated_ffn_half_swap_is_loud_against_the_mlx_record",
        ],
        "dit_io": [
            "the_model_embeds_the_context_before_refining_it",
            "the_whole_model_reproduces_the_reference_velocity",
            "forward_packed_rejects_an_out_of_range_adaln_index",
            "the_cached_modulation_path_reproduces_the_resident_one",
        ],
        "dit_parity": [
            "adaln_modulation_matches_the_reference",
            "transformer_block_matches_the_reference",
            "every_norm_epsilon_is_wired_through",
            "reading_the_ffn_halves_gate_first_breaks_parity",
            "the_modality_term_of_the_adaln_index_is_load_bearing",
            "the_adaln_table_is_reshaped_before_it_is_chunked",
            "every_block_weight_is_load_bearing",
            "parity_residuals_bound_the_mutation_floor",
            "weight_mapping_is_exhaustive_over_the_block_stack",
        ],
        "pruned_adaln": [
            "native_pruned_adaln_matches_pinned_reference",
            "native_blocks_and_final_outputs_match_the_pinned_reference",
            "the_curve_cache_keys_on_the_schedule_and_the_basis",
            "a_mixed_basis_block_stack_is_refused_before_any_projection",
            "the_stock_path_is_untouched_and_refuses_curve_coordinates",
            "load_from_dir_builds_the_pruned_model_with_its_identity_at_stored_precision",
            "manual_unit_patch_and_mixed_reference_full_forward_matches_resident_control",
        ],
        "turbo_lora": [
            "every_enumerated_target_resolves_through_the_model_tree",
            "adaln_and_the_norms_are_not_addressable",
            "the_install_folds_at_the_published_strength",
            "the_installed_fold_keeps_the_factor_dtype",
            "a_file_that_folds_nothing_is_refused_per_spec_not_in_aggregate",
            "an_unmatched_target_is_surfaced_by_name",
            "lokr_matches_an_independent_dense_kron_and_rejects_mutant_results",
            "fused_qkv_and_mlp_lokr_cover_the_complete_block_surface_numerically",
            "lokr_rejects_unknown_partial_unmatched_and_false_kind_claims",
            "direct_lokr_reaches_every_trunk_and_token_refiner_adapter_target",
            "a_peft_blob_supplies_the_alpha_and_its_rank_is_only_a_cross_check",
            "the_conversion_leaves_no_comfyui_module_behind",
            "clay_raw_refiner_header_conversion_preserves_factors_alpha_and_strict_unknowns",
        ],
    },
}

# Library (`--lib`) unit tests that construct or run a complete DiT/JointDit. They are
# gated the same way as the integration suites, but execute through `cargo test --lib`
# on the same manual routes. `filter` is the libtest path prefix; `skip` is the
# explicit set of ignored non-model bodies sharing that filter (pinned).
LIB_GATED: dict[tuple[str, str], dict[str, object]] = {
    ("mlx", "veda_fixture"): {
        "file": MLX_SRC / "veda_fixture.rs",
        "filter": "veda_fixture::",
        "names": [
            "native_generator_learned_sparse_off_and_failure_paths",
            "native_veda_controls_preserve_carriers_and_capture_published_schedules",
        ],
        "skip": ["selected_kernel_actual_device_smoke"],
    },
    ("candle", "dit::model::veda_controls_tests"): {
        "file": CANDLE_SRC / "dit" / "model.rs",
        "filter": "dit::model::veda_controls_tests::",
        "names": [
            "native_controls_capture_every_mode_packed_tier_and_adapter_stack",
            "native_base_and_all_published_turbo_steps_reach_attention",
        ],
        "skip": [],
    },
}

LIB_ROUTES = [
    ("synthetic-w4a8-mlx", "mlx", "veda_fixture"),
    ("synthetic-w4a8-cuda", "candle", "dit::model::veda_controls_tests"),
]

# Deliberately ordinary: source-traced to metadata (`MiniMaxH3Dit::names`,
# `DitBlock::names`) or at most ONE transformer block — never the complete DiT, a
# block stack, or the pipeline. The test below re-checks that boundary from source,
# so one of these growing into a complete-model test reds this policy rather than
# quietly re-entering ordinary CI.
COMPONENT_ONLY = {
    ("mlx", "dit_io", "the_seventeen_load_at_the_published_geometry"): "metadata",
    ("candle", "dit_io", "the_seventeen_load_at_the_published_geometry"): "metadata",
    ("candle", "dit_io", "the_declared_names_partition_the_fixture"): "metadata",
}

# Route-only names: already-ignored native/real-fixture bodies that share a module
# with the synthetic suites above. They execute on the same manual route and are
# therefore counted and name-guarded too, but their `#[ignore]` reason is their own.
ROUTE_NAMES = {
    ("mlx", "w4a8_compare"): [
        "native_forward_matches_the_external_reference",
        "invalid_operands_are_refused_by_the_published_contract",
        "mixed_w4a8_and_convrot_layers_keep_their_own_arithmetic",
    ],
    ("candle", "w4a8_compare"): [
        "native_forward_matches_the_external_reference",
        "mixed_w4a8_and_convrot_layers_keep_their_own_arithmetic",
    ],
    ("candle", "convrot_int8"): ["native_projections_match_pinned_kitchen"],
    ("candle", "adaln_evict_memory"): ["the_evicted_adaln_weights_leave_the_cuda_pool"],
}

# Real-weight `#[ignore]`s that live in a module with a synthetic route. `--ignored`
# alone would conscript them, so each route SKIPS them by name; the skip set is
# pinned so a new skip (or a missing one) reds the policy.
SKIPS = {
    ("mlx", "turbo_lora"): [
        "published_turbo_files_resolve_to_dit_targets_at_the_measured_alphas",
        "the_real_turbo_lora_folds_onto_the_real_dit",
        "turbo_render_records_a_measured_clip",
        "exact_h3_trainer_file_receipt",
        "manual_metal_exact_h3_trainer_runtime_receipt",
    ],
    ("candle", "turbo_lora"): [
        "manual_cuda_fabricated_lokr_install_receipt",
        "exact_h3_trainer_file_receipt",
        "manual_cuda_exact_h3_trainer_runtime_receipt",
    ],
}

# (workflow job id, backend, module) -> the manual route that executes it.
ROUTES = [
    ("synthetic-w4a8-mlx", "mlx", "w4a8_compare"),
    ("synthetic-w4a8-mlx", "mlx", "adaln_cache"),
    ("synthetic-w4a8-mlx", "mlx", "adaln_evict_memory"),
    ("synthetic-w4a8-mlx", "mlx", "adapter_topology"),
    ("synthetic-w4a8-mlx", "mlx", "block_window"),
    ("synthetic-w4a8-mlx", "mlx", "bounded_attention"),
    ("synthetic-w4a8-mlx", "mlx", "dit_io"),
    ("synthetic-w4a8-mlx", "mlx", "dit_parity"),
    ("synthetic-w4a8-mlx", "mlx", "pruned_adaln"),
    ("synthetic-w4a8-mlx", "mlx", "turbo_lora"),
    ("synthetic-w4a8-cuda", "candle", "convrot_int8"),
    ("synthetic-w4a8-cuda", "candle", "adaln_evict_memory"),
    ("synthetic-w4a8-cuda", "candle", "cross_backend"),
    ("synthetic-w4a8-cuda", "candle", "dit_io"),
    ("synthetic-w4a8-cuda", "candle", "dit_parity"),
    ("synthetic-w4a8-cuda", "candle", "pruned_adaln"),
    ("synthetic-w4a8-cuda", "candle", "adapter_topology"),
    ("synthetic-w4a8-cuda", "candle", "turbo_lora"),
    ("synthetic-w4a8-cuda", "candle", "w4a8_compare"),
]


def _strip_comment_lines(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def jobs(workflow: str) -> dict[str, str]:
    """Split a workflow into top-level `jobs:` blocks, line-based."""
    lines = workflow.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.rstrip() == "jobs:")
    except StopIteration as exc:  # pragma: no cover - malformed workflow
        raise AssertionError("workflow has no top-level `jobs:` block") from exc
    starts = [
        (i, m.group(1))
        for i in range(start + 1, len(lines))
        if (m := re.match(r"^  ([A-Za-z0-9_-]+):\s*$", lines[i]))
    ]
    blocks = {}
    for idx, (line_no, name) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        blocks[name] = "\n".join(lines[line_no:end])
    return blocks


def steps(job_text: str) -> list[str]:
    parts = re.split(r"\n(?=      - name: )", job_text)
    return [p for p in parts if p.lstrip().startswith("- name:")]


def cargo_test_commands(step: str) -> list[str]:
    """Read complete logical commands, including the Metal turbo step's continuations."""
    commands = []
    pending = ""
    for raw in step.splitlines():
        line = raw.strip()
        if not pending and not line.startswith("cargo test "):
            continue
        continued = line.endswith("\\") or line.endswith("^")
        pending += (" " if pending else "") + (line[:-1].rstrip() if continued else line)
        if not continued:
            commands.append(pending)
            pending = ""
    if pending:
        raise AssertionError("unterminated Cargo command in H3 workflow")
    return commands


def rust_tests(path: Path) -> dict[str, bool]:
    """Map each `#[test]` function in a Rust file to whether it is `#[ignore]`d."""
    found: dict[str, bool] = {}
    pending: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("//") or line.startswith("/*") or line.startswith("*"):
            continue
        if line.startswith("#["):
            pending.append(line)
            continue
        match = re.match(r"fn\s+([A-Za-z0-9_]+)\s*[<(]", line)
        if match and any("#[test]" in attr for attr in pending):
            found[match.group(1)] = any("#[ignore" in attr for attr in pending)
            pending = []
            continue
        pending = []
    return found


def test_root(backend: str) -> Path:
    return MLX_TESTS if backend == "mlx" else CANDLE_TESTS


def ignore_reason(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"#\[ignore\s*=\s*\"([^\"]*)\"\]"
        rf"(?:\s*\n\s*#\[[^\n]*\])*\s*\n\s*fn\s+{re.escape(name)}\s*\(",
        text,
    )
    return match.group(1) if match else ""


# Dynamic discovery: these signals identify a test that CONSTRUCTS or RUNS a model or a
# full model block. Individual linear/AdaLN/packing/loader/denoise-loop-stub checks carry
# none of them and stay ordinary.
DISCOVERY_SIGNALS = [
    (r"MiniMaxH3Dit::from_weights", "complete"),
    (r"MiniMaxH3Dit::load", "complete"),
    (r"JointDit::new", "complete"),
    (r"forward_packed", "complete"),
    (r"MiniMaxH3Pipeline", "complete"),
    (r"\.generate\s*\(", "complete"),
    (r"veda_fixture::dit", "complete"),
    (r"DitBlock::from_weights", "block"),
    (r"DitBlockStream::new", "stream"),
    (r"\.open\s*\(", "stream_open"),
]

# Modules whose ignored model tests are deliberately reached by the privileged
# real-weight/generator lanes (`real-weights.yml`), not by the synthetic manual routes.
REAL_WEIGHT_MODULES = {
    ("mlx", "adaln_cold_start"),
    ("mlx", "adaln_evict_real_weights"),
    ("mlx", "audio_vae_encode_parity"),
    ("mlx", "audio_vae_parity"),
    ("mlx", "block_window_real"),
    ("mlx", "bounded_attention_real"),
    ("mlx", "cross_backend_record"),
    ("mlx", "duration_sweep_real"),
    ("mlx", "first_light"),
    ("mlx", "fl2va_first_light"),
    ("mlx", "quant_tiers_real"),
    ("mlx", "real_weights"),
    ("mlx", "ref2va_checkpoint"),
    ("mlx", "sequence_cost_real"),
    ("mlx", "staged_residency"),
    ("mlx", "streamed_generate_real"),
    ("mlx", "te_load_integrity"),
    ("mlx", "te_tier_generate_stages"),
    ("mlx", "te_tier_real_weights"),
    ("mlx", "veda_ab"),
    ("mlx", "vram_probe"),
    ("candle", "real_weights"),
    ("candle", "ref2va_checkpoint"),
    ("candle", "veda_ab"),
    ("candle", "vram_probe"),
}


def _strip_rust(text: str) -> str:
    """Blank comments and string/char literal contents, preserving byte offsets."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        elif c == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    break
                j += 1
            out.append('"' + " " * max(0, j - i - 1) + '"')
            i = j + 1
        elif c == "'":
            j = i + 1
            if j < n and text[j] == "\\":
                j += 2
            else:
                j += 1
            if j < n and text[j] == "'":
                out.append("'" + " " * (j - i - 1) + "'")
                i = j + 1
            else:
                out.append(c)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _rust_functions(text: str) -> list[tuple[str, int, str]]:
    clean = _strip_rust(text)
    found: list[tuple[str, int, str]] = []
    for match in re.finditer(r"\bfn\s+([A-Za-z0-9_]+)\s*[<(]", clean):
        brace = clean.find("{", match.end())
        if brace < 0 or ";" in clean[match.end() : brace]:
            continue
        depth = 0
        end = brace
        for end in range(brace, len(clean)):
            if clean[end] == "{":
                depth += 1
            elif clean[end] == "}":
                depth -= 1
                if depth == 0:
                    break
        found.append((match.group(1), match.start(), text[brace : end + 1]))
    return found


def _attributes_before(text: str, pos: int) -> list[str]:
    attrs: list[str] = []
    for line in reversed(text[max(0, pos - 1200) : pos].splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("//") or stripped.startswith("/*"):
            continue
        if stripped.startswith("#["):
            attrs.append(stripped)
        else:
            break
    return attrs


def model_execution_kind(path: Path, name: str) -> str | None:
    """Return 'complete'/'block' if the test reaches a model constructor, else None."""
    text = path.read_text(encoding="utf-8")
    functions = _rust_functions(text)
    direct: dict[str, set[str]] = {}
    calls: dict[str, set[str]] = {}
    for fn, _, body in functions:
        clean = _strip_rust(body)
        direct[fn] = {tag for pattern, tag in DISCOVERY_SIGNALS if re.search(pattern, clean)}
        calls[fn] = set(re.findall(r"\b([A-Za-z0-9_]+)\s*[<(]", clean))

    seen: set[str] = set()
    queue = [name]
    tags: set[str] = set()
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        tags |= direct.get(current, set())
        queue.extend(c for c in calls.get(current, set()) if c in direct)
    if "complete" in tags:
        return "complete"
    if "block" in tags:
        return "block"
    if "stream" in tags and "stream_open" in tags:
        return "block"
    return None


def h3_source_files() -> list[Path]:
    files: list[Path] = []
    for root in (MLX_TESTS, CANDLE_TESTS, MLX_SRC, CANDLE_SRC):
        files.extend(sorted(root.rglob("*.rs")))
    return files


def enclosing_rust_modules(path: Path, function: str) -> list[str]:
    """Resolve inline module ancestors without confusing comments/strings with Rust braces."""
    clean = _strip_rust(path.read_text(encoding="utf-8"))
    definition = re.search(r"\bfn\s+" + re.escape(function) + r"\s*\(", clean)
    if definition is None:
        raise AssertionError(f"{path}::{function} vanished")
    stack: list[str | None] = []
    for token in re.finditer(r"\bmod\s+([A-Za-z0-9_]+)\s*\{|[{}]", clean[:definition.start()]):
        if token.group(1):
            stack.append(token.group(1))
        elif token.group() == "{":
            stack.append(None)
        else:
            stack.pop()
    return [name for name in stack if name is not None]


def rust_test_body(path: Path, name: str) -> str:
    """Extract one test function body (comment lines removed), brace-matched."""
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"\n[ \t]*fn\s+{re.escape(name)}\s*[<(]", text)
    if match is None:
        raise AssertionError(f"{path.name}::{name} not found")
    brace = text.find("{", match.end())
    if brace < 0:
        raise AssertionError(f"{path.name}::{name} has no body")
    depth = 0
    end = brace
    for end in range(brace, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                break
    body = text[brace : end + 1]
    return "\n".join(
        line for line in body.splitlines() if not line.lstrip().startswith("//")
    )


class H3OrdinaryExecutionPolicyTest(unittest.TestCase):
    def test_actual_checkpoint_probe_is_ignored_and_unreachable_from_ordinary_ci(self) -> None:
        name = "actual_checkpoint_projections_match_pinned_cpu_oracle"
        # The only executor lives in the explicit SceneWorks terminal campaign. Native
        # PR/main test commands may compile this module, but must never opt it in.
        ordinary = _strip_comment_lines(CI.read_text(encoding="utf-8"))
        self.assertNotIn(f"artifact_probe::{name}", ordinary)
        self.assertNotIn("H3_ACTUAL_PROBE_MANIFEST", ordinary)
        for root in (MLX_TESTS, CANDLE_TESTS):
            source = root / "artifact_probe.rs"
            with self.subTest(source=source):
                self.assertTrue(rust_tests(source).get(name), "actual payload/device probe must stay ignored")
                self.assertGreaterEqual(len(ignore_reason(source, name)), MINIMUM_IGNORE_REASON)
                self.assertIn("mod artifact_probe;", (root / "main.rs").read_text(encoding="utf-8"))
                # Execute the source guard against an unignore mutant, not just a token check.
                mutant = re.sub(r'^\s*#\[ignore[^\n]*\]\s*$', '', source.read_text(encoding="utf-8"), flags=re.MULTILINE)
                with tempfile.TemporaryDirectory() as directory:
                    changed = Path(directory) / "artifact_probe.rs"
                    changed.write_text(mutant, encoding="utf-8")
                    self.assertIn(name, rust_tests(changed))
                    self.assertFalse(rust_tests(changed)[name], "unignore mutant escaped the policy")

    def test_h3_gpu0_and_main_gpu1_keep_distinct_physical_identity(self) -> None:
        blocks = jobs(CI.read_text(encoding="utf-8"))
        main = blocks["windows-cuda"]
        h3 = blocks["synthetic-w4a8-cuda"]
        for body, index in ((main, "1"), (h3, "0")):
            self.assertIn(f"nvidia-smi -i {index} --query-gpu=index,uuid,name,pci.bus_id", body)
            self.assertIn(f"$fields[0] -ne '{index}'", body)
            self.assertIn("$fields[1] -notmatch '^GPU-[0-9a-fA-F-]+$'", body)
            self.assertIn("CUDA_VISIBLE_DEVICES=$gpuUuid", body)
            self.assertIn("CUDA_DEVICE_ORDER=PCI_BUS_ID", body)
            self.assertIn(f"SCENEWORKS_GPU_ID={index}", body)
            self.assertNotIn("--query-compute-apps", body)
        self.assertNotIn("- name: Prove H3", main)
        self.assertIn("- name: Prove H3 pruned AdaLN", h3)
        self.assertIn("H3_GEOMETRY_AUDIO_FULL_FORWARD backend=cuda cases=11", h3)
        self.assertIn("inputs.synthetic_w4a8 == 'cuda'", h3)
        self.assertIn("github.event_name == 'workflow_dispatch'", h3)

    def test_h3_and_moss_manual_queue_keys_both_preserve_each_run(self) -> None:
        workflow = CI.read_text(encoding="utf-8")
        expression = re.search(r"^  group: inference-ci-\$\{\{ (.+) \}\}$", workflow, re.M).group(1)
        def key(event, synthetic, lane, run):
            values = {"github.event_name": event, "github.event.inputs.synthetic_w4a8": synthetic,
                      "inputs.lanes": lane, "github.run_id": run, "github.ref": "refs/pull/7/merge", "github.sha": "source-sha"}
            rendered = expression
            for name in sorted(values, key=len, reverse=True):
                rendered = rendered.replace(name, repr(values[name]))
            rendered = rendered.replace("&&", " and ").replace("||", " or ")
            self.assertNotRegex(rendered, r"[A-Za-z_]\w*(?:\.\w+)+")
            return eval(rendered, {"__builtins__": {}}, {"format": lambda template, value: template.format(value)})
        for mode, lane in (("cuda", "windows-cuda"), ("mlx", "all"), ("both", "all"),
                           ("none", "moss-edit-metal"), ("none", "moss-edit-cuda")):
            self.assertNotEqual(key("workflow_dispatch", mode, lane, 1), key("workflow_dispatch", mode, lane, 2))
        self.assertEqual(key("pull_request", "none", "all", 1), "refs/pull/7/merge")
        self.assertEqual(key("push", "none", "all", 1), "source-sha")
        self.assertEqual(key("workflow_dispatch", "none", "all", 1), "source-sha")
        self.assertIn("cancel-in-progress: ${{ github.event_name == 'pull_request' }}", workflow)
        library = (ROOT / "crates/contracts/gen-core/src/lib.rs").read_text(encoding="utf-8")
        for module in ("moss_image_edit", "pruned_adaln", "h3_adapter_memory", "h3_conditioning_memory", "h3_memory"):
            self.assertEqual(library.count(f"pub mod {module};"), 1, "both main and H3 contract exports must survive integration")

    def test_manual_geometry_proof_requires_actual_devices_and_all_cases(self) -> None:
        name = "manual_unit_patch_and_mixed_reference_full_forward_matches_resident_control"
        candle = rust_test_body(CANDLE_TESTS / "pruned_adaln.rs", name)
        metal = rust_test_body(MLX_TESTS / "pruned_adaln.rs", name)
        self.assertIn("Device::new_cuda(0)", candle)
        self.assertIn("manual_conditioning_model(&f, &cfg, &d)", candle)
        self.assertIn("output.0.device().is_cuda()", candle)
        self.assertIn("output.1.device().is_cuda()", candle)
        self.assertIn("assert!(d.is_cuda()", re.sub(r"\s+", "", candle))
        self.assertIn("mlx_rs::Device::try_default()", metal)
        self.assertIn("mlx_rs::DeviceType::Gpu", metal)
        for body in (candle, metal):
            compact = re.sub(r"\s+", "", body)
            self.assertIn("completed.set(completed.get()+1)", compact)
            self.assertIn("assert_eq!(completed.get(),11,", compact)
        workflow = CI.read_text(encoding="utf-8")
        for backend, body in (("cuda", candle), ("metal", metal)):
            marker = f"H3_GEOMETRY_AUDIO_FULL_FORWARD backend={backend} cases=11"
            self.assertIn(marker, body)
            self.assertIn(marker, workflow, "manual route must require the completed-device marker")

    def test_complete_model_tests_are_ignored_on_both_backends(self) -> None:
        for backend, modules in GATED.items():
            for module, names in modules.items():
                path = test_root(backend) / f"{module}.rs"
                tests = rust_tests(path)
                for name in names:
                    with self.subTest(backend=backend, module=module, test=name):
                        self.assertIn(name, rust_tests(path), f"{module}::{name} vanished")
                        self.assertTrue(
                            rust_tests(path)[name],
                            f"{backend}/{module}::{name} executes a complete model and must "
                            "carry #[ignore]",
                        )
                        self.assertGreaterEqual(
                            len(ignore_reason(path, name)),
                            MINIMUM_IGNORE_REASON,
                            f"{backend}/{module}::{name} needs a real #[ignore] reason",
                        )
        for (backend, key), spec in LIB_GATED.items():
            path = spec["file"]
            tests = rust_tests(path)
            for name in spec["names"]:
                with self.subTest(backend=backend, lib=key, test=name):
                    self.assertIn(name, tests)
                    self.assertTrue(tests[name], f"{backend}/lib {key}::{name} must be #[ignore]d")
                    self.assertGreaterEqual(
                        len(ignore_reason(path, name)),
                        MINIMUM_IGNORE_REASON,
                        f"{backend}/lib {key}::{name} needs a real #[ignore] reason",
                    )

    def test_component_only_tests_stay_component_level(self) -> None:
        for (backend, module, name), kind in COMPONENT_ONLY.items():
            with self.subTest(backend=backend, module=module, name=name):
                path = test_root(backend) / f"{module}.rs"
                tests = rust_tests(path)
                self.assertIn(name, tests)
                self.assertFalse(
                    tests[name],
                    f"{backend}/{module}::{name} is a component/metadata check and must "
                    "stay in ordinary CI",
                )
                body = rust_test_body(path, name)
                for forbidden in (
                    "MiniMaxH3Dit::from_weights",
                    "MiniMaxH3Dit::load",
                    "MiniMaxH3Dit::load_from_dir",
                    "forward_packed",
                ):
                    self.assertNotIn(
                        forbidden,
                        body,
                        f"{backend}/{module}::{name} now constructs or runs the "
                        f"complete model ({forbidden}); it must move to the gated manifest",
                    )
                # Every retained entry is metadata-only: `::names` over the published key
                # space, with no model construction. The dynamic discovery test below
                # independently rejects any test that reaches a model constructor.
                self.assertIn("::names(", body)

    def test_dynamic_discovery_covers_every_model_or_block_test(self) -> None:
        """New model/model-block tests must be gated and manifest-routed, not just static names."""
        covered: set[tuple[Path, str]] = set()
        for backend, modules in GATED.items():
            for module, names in modules.items():
                covered |= {(test_root(backend) / f"{module}.rs", n) for n in names}
        for (backend, module), names in ROUTE_NAMES.items():
            covered |= {(test_root(backend) / f"{module}.rs", n) for n in names}
        for spec in LIB_GATED.values():
            for n in list(spec["names"]) + list(spec["skip"]):
                covered.add((spec["file"], n))
        for (backend, module), names in SKIPS.items():
            covered |= {(test_root(backend) / f"{module}.rs", n) for n in names}

        for path in h3_source_files():
            backend = "mlx" if str(path).startswith(str(MLX_SRC)) or str(path).startswith(str(MLX_TESTS)) else "candle"
            tests = rust_tests(path)
            for name, ignored in tests.items():
                with self.subTest(file=str(path.relative_to(ROOT)), test=name):
                    kind = model_execution_kind(path, name)
                    if kind is None:
                        continue
                    if not ignored:
                        self.fail(
                            f"{path.relative_to(ROOT)}::{name} executes a {kind} model body "
                            "and is not #[ignore]d; ordinary CI must not run it"
                        )
                    if (path, name) in covered:
                        continue
                    module = path.stem if path.parent.name == "tests" else str(
                        path.relative_to(MLX_SRC if backend == "mlx" else CANDLE_SRC)
                    ).removesuffix(".rs").replace("/", "::")
                    self.assertIn(
                        (backend, module),
                        REAL_WEIGHT_MODULES,
                        f"{path.relative_to(ROOT)}::{name} is an ignored {kind} model body "
                        "with no synthetic route and is not a recorded real-weight module",
                    )

    def test_ordinary_jobs_never_execute_h3_model_tests(self) -> None:
        blocks = jobs(CI.read_text(encoding="utf-8"))
        for job, text in blocks.items():
            # Negative diagnostic exclusions mention workflow_dispatch too; they are NOT a
            # manual-only gate. Require its positive equality in the actual job/step if field.
            job_conditions = "\n".join(line for line in text.splitlines()
                if line.startswith("    if:") or line.startswith("      (github.event_name"))
            manual_job = "github.event_name == 'workflow_dispatch'" in job_conditions
            for step in steps(text):
                body = _strip_comment_lines(step)
                if "minimax-h3" not in body or "--ignored" not in body:
                    continue
                step_conditions = "\n".join(line for line in step.splitlines()
                    if line.startswith("        if:"))
                with self.subTest(job=job, step=step.splitlines()[0]):
                    self.assertTrue(manual_job or "github.event_name == 'workflow_dispatch'" in step_conditions,
                        "an H3 model step needs a positive workflow_dispatch gate; ordinary PR/main must never execute it")

    def test_hosted_pruned_precision_step_is_explicit_and_positive(self) -> None:
        block = jobs(CI.read_text(encoding="utf-8"))["macos-metal"]
        matching = [step for step in steps(block)
            if "- name: Check H3 precise pruned projection on hosted Metal (manual)" in step]
        self.assertEqual(len(matching), 1)
        step = matching[0]
        self.assertIn("if: github.event_name == 'workflow_dispatch'", step)
        self.assertIn("inputs.synthetic_w4a8 == 'mlx'", step)
        self.assertIn("inputs.synthetic_w4a8 == 'both'", step)
        self.assertIn("set -euo pipefail", step)
        self.assertIn("--ignored --exact --nocapture", step)
        self.assertIn("test result: ok. 1 passed; 0 failed; 0 ignored", step)
        self.assertIn("test pruned_adaln::native_pruned_adaln_matches_pinned_reference ...", step)
        self.assertIn("H3_PRUNED_ORACLE_DEVICE", step)
        self.assertIn("H3_PRUNED_ORACLE block 0 modulation: max_abs=", step)
        self.assertNotIn("continue-on-error", step)
        self.assertNotIn("|| true", step)
        self.assertIn('MACOSX_DEPLOYMENT_TARGET: "15.0"', block)

    def test_manual_h3_routes_pin_exact_counts_and_names(self) -> None:
        blocks = jobs(CI.read_text(encoding="utf-8"))
        for job, backend, module in ROUTES:
            with self.subTest(job=job, backend=backend, module=module):
                self.assertIn(job, blocks)
                matching = [
                    step
                    for step in steps(blocks[job])
                    if f"{module}::" in step and "minimax-h3" in step
                ]
                self.assertEqual(
                    1, len(matching), f"{job} must have exactly one {module} H3 step"
                )
                step = matching[0]
                body = _strip_comment_lines(step)
                self.assertIn("--ignored", body)
                self.assertNotIn("|| true", step, "an H3 route step cannot swallow failures")
                self.assertNotIn("set +e", step)
                if job.startswith("synthetic-w4a8-mlx"):
                    self.assertIn("set -euo pipefail", body)
                else:
                    self.assertIn('if not "%RC%"=="0" exit /b %RC%', body)

                skips = SKIPS.get((backend, module), [])
                commands = cargo_test_commands(body)
                self.assertEqual(len(commands), 2, "compiled-list preflight and execution are both required")
                for command in commands:
                    parsed_skips = re.findall(r"--skip\s+([A-Za-z0-9_]+)", command)
                    self.assertEqual(sorted(skips), sorted(parsed_skips),
                        f"{job}/{module}: metadata and execution must use the same real-weight exclusions")

                path = test_root(backend) / f"{module}.rs"
                ignored = [n for n, is_ignored in rust_tests(path).items() if is_ignored]
                expected = len(ignored) - len(skips)
                count = re.search(
                    r"test result: ok\. (\d+) passed; 0 failed; 0 ignored", step
                )
                self.assertIsNotNone(
                    count, f"{job}/{module} has no positive exact execution-count guard"
                )
                self.assertEqual(
                    expected,
                    int(count.group(1)),
                    f"{job}/{module} guard must equal its executable ignored bodies "
                    f"({len(ignored)} ignored - {len(skips)} skipped)",
                )
                self.assertGreaterEqual(expected, 1)

                guarded = list(GATED.get(backend, {}).get(module, []))
                guarded += ROUTE_NAMES.get((backend, module), [])
                for name in guarded:
                    if name in skips:
                        continue
                    with self.subTest(job=job, module=module, name=name):
                        self.assertIn(
                            f"test {module}::{name} ...",
                            step,
                            f"{job}/{module} does not name-guard {name}",
                        )

    def test_manual_lib_h3_routes_pin_exact_counts_and_names(self) -> None:
        blocks = jobs(CI.read_text(encoding="utf-8"))
        for job, backend, key in LIB_ROUTES:
            spec = LIB_GATED[(backend, key)]
            with self.subTest(job=job, backend=backend, lib=key):
                matching = [
                    step
                    for step in steps(blocks[job])
                    if "--lib" in step and str(spec["filter"]) in step and "minimax-h3" in step
                ]
                self.assertEqual(1, len(matching), f"{job} needs exactly one --lib {key} step")
                step = matching[0]
                body = _strip_comment_lines(step)
                self.assertIn("--ignored", body)
                self.assertNotIn("|| true", step)
                self.assertNotIn("set +e", step)
                if job.startswith("synthetic-w4a8-mlx"):
                    self.assertIn("set -euo pipefail", body)
                else:
                    self.assertIn('if not "%RC%"=="0" exit /b %RC%', body)

                skips = list(spec["skip"])
                commands = cargo_test_commands(body)
                self.assertEqual(len(commands), 2)
                for command in commands:
                    parsed = re.findall(r"--skip\s+([A-Za-z0-9_]+)", command)
                    self.assertEqual(sorted(skips), sorted(parsed), f"{job}/{key} skip set drifted")

                ignored = [n for n, v in rust_tests(spec["file"]).items() if v]
                expected = len(ignored) - len(skips)
                count = re.search(r"test result: ok\. (\d+) passed; 0 failed; 0 ignored", step)
                self.assertIsNotNone(count, f"{job}/{key} has no exact execution-count guard")
                self.assertEqual(expected, int(count.group(1)))
                self.assertGreaterEqual(expected, 1)

                prefix = str(spec["filter"])
                for name in spec["names"]:
                    self.assertIn(f"test {prefix}{name} ...", step)

    def test_library_filters_follow_real_rust_module_nesting(self) -> None:
        for job, backend, key in LIB_ROUTES:
            spec = LIB_GATED[(backend, key)]
            root = MLX_SRC if backend == "mlx" else CANDLE_SRC
            outer = list(spec["file"].relative_to(root).with_suffix("").parts)
            block = jobs(CI.read_text(encoding="utf-8"))[job]
            for name in spec["names"]:
                actual = "::".join(outer + enclosing_rust_modules(spec["file"], name) + [name])
                self.assertEqual(spec["filter"] + name, actual,
                    "workflow/policy prefixes must follow the actual Rust module, not an invented namespace")
                self.assertTrue(rust_tests(spec["file"])[name])
                self.assertIn(f'"{actual}: test"', block)
                self.assertIn(f'"test {actual} ..."', block)

    def test_compiled_metadata_preflights_bind_every_runtime_name_and_count(self) -> None:
        seen = 0
        for job, block in jobs(CI.read_text(encoding="utf-8")).items():
            for step in steps(block):
                commands = [line for line in cargo_test_commands(step)
                    if "minimax-h3" in line and "--ignored" in line]
                if not commands:
                    continue
                with self.subTest(job=job, step=step.splitlines()[0]):
                    self.assertEqual(len(commands), 2)
                    listed = [line for line in commands if "--list" in line]
                    executed = [line for line in commands if "--list" not in line]
                    self.assertEqual(len(listed), 1)
                    self.assertEqual(len(executed), 1)
                    self.assertIn("--list --color never", listed[0])
                    def command_arguments(command):
                        return command.split(" 2>&1", 1)[0].split(" >", 1)[0]
                    expected = command_arguments(executed[0]).replace(" --nocapture", "").replace(" --test-threads=1", "")
                    actual = command_arguments(listed[0]).replace(" --list --color never", "")
                    self.assertEqual(actual, expected, "metadata must list the exact executed filter/flags")
                    runtime_names = re.findall(r'"test ([A-Za-z0-9_:]+) \.\.\."', step)
                    metadata_names = re.findall(r'"([A-Za-z0-9_:]+): test"', step)
                    self.assertGreater(len(runtime_names), 0)
                    self.assertCountEqual(runtime_names, metadata_names)
                    count = re.search(r"test result: ok\. (\d+) passed; 0 failed; 0 ignored", step)
                    self.assertIsNotNone(count, "metadata preflight must never replace the execution count guard")
                    self.assertEqual(int(count[1]), len(runtime_names))
                    if "shell: cmd" in step:
                        self.assertIn("-File scripts\\ci\\h3-native-test-list.ps1", step)
                        self.assertIn(f"-ExpectedCount {len(runtime_names)}", step)
                        encoded = re.search(r'-ExpectedNames "([^"\n]+)"', step)
                        self.assertIsNotNone(encoded)
                        self.assertCountEqual(encoded[1].split("|"), runtime_names)
                        for name in runtime_names:
                            self.assertIn(f'findstr /L /C:"{name}: test"', step)
                        self.assertIn('type "%LIST%"', step)
                        self.assertNotIn("LIST_COUNT", step)
                        self.assertNotIn("findstr /R", step)
                        self.assertNotIn("findstr /L /X", step)
                    else:
                        self.assertIn(f' -eq {len(runtime_names)}', step)
                        for name in runtime_names:
                            self.assertIn(f'grep -Fxq "{name}: test"', step)
                        self.assertIn('cat "$list_log"', step)
                    seen += len(runtime_names)
        self.assertGreater(seen, 0)

    def test_windows_inventory_validator_is_exact_diagnostic_and_self_tested_before_build(self) -> None:
        script = (ROOT / "scripts/ci/h3-native-test-list.ps1").read_text(encoding="utf-8")
        for token in ("[IO.File]::ReadAllLines($path)", "[StringComparer]::Ordinal)",
                      "actual_count=$actual.Count", "expected_count=$count", "names=$actual",
                      "missing=$missing", "extra=$extra", "H3_COMPILED_TEST_LIST ",
                      "if ($actual.Count -ne $count)", "if ($missing.Count -ne 0 -or $extra.Count -ne 0)"):
            self.assertIn(token, script)
        self.assertLess(script.index("H3_COMPILED_TEST_LIST ", script.index("$actualSet=")), script.index("if ($actual.Count -ne $count)"))
        self.assertIn('foreach ($ending in @("`n","`r`n"))', script)
        for case in ("zero", "partial", "missing", "extra", "wrong-namespace", "substring", "trailing", "case", "duplicate", "nonpositive"):
            self.assertIn(f"label='{case}'", script)
        self.assertIn("Running tests\\main.rs", script)
        self.assertIn("Finished test profile", script)
        self.assertIn("Invoke-InventorySelfTest $ignoreMissing", script)
        self.assertIn("H3_COMPILED_TEST_LIST_MUTANT_REJECTED missing-name", script)
        self.assertIn("actual_count=$null", script)
        self.assertIn("read_error=$_.Exception.Message", script)
        self.assertIn("H3 compiled list self-test accepted an unreadable list", script)
        self.assertIn("self_test=[bool]$SelfTest", script)
        self.assertIn("Invoke-InventorySelfTest $correct", script)
        for job in ("windows-cuda-check", "synthetic-w4a8-cuda"):
            block = jobs(CI.read_text(encoding="utf-8"))[job]
            self.assertEqual(block.count("run: ./scripts/ci/h3-native-test-list.ps1 -SelfTest"), 1)
            first_build = re.search(r"(?m)^          cargo (?:build|test|clippy)", block)
            self.assertIsNotNone(first_build)
            self.assertLess(block.index("h3-native-test-list.ps1 -SelfTest"), first_build.start())
            if job == "windows-cuda-check":
                smoke = next(s for s in steps(block) if "- name: Verify native MOSS transport guards without GPU execution" in s)
                self.assertIn("timeout-minutes: 5", smoke)
                self.assertIn("-SelfTest -SelfTestPositiveTimeoutSeconds 60", smoke)
                self.assertLess(block.index("moss-native-terminal.ps1 -SelfTest"), first_build.start())

    def test_real_weights_routes_do_not_select_gated_synthetic_tests(self) -> None:
        body = _strip_comment_lines(REAL_WEIGHTS.read_text(encoding="utf-8"))
        for backend, modules in GATED.items():
            for module, names in modules.items():
                for name in names:
                    with self.subTest(backend=backend, module=module, name=name):
                        self.assertNotIn(
                            f"{module}::{name}",
                            body,
                            "a gated synthetic H3 test must not be reachable from the "
                            "scheduled/real-weight lane; the explicit manual H3 route is "
                            "its only home",
                        )
        for (backend, key), spec in LIB_GATED.items():
            for name in spec["names"]:
                with self.subTest(backend=backend, lib=key, name=name):
                    self.assertNotIn(
                        name,
                        body,
                        "a gated library model test must not be selectable from the "
                        "scheduled/real-weight lane",
                    )


if __name__ == "__main__":
    unittest.main()
