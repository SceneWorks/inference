call "%VCVARS%"
set "IRIS_FAILED=0"
rem The release REPA teacher against the committed upstream tokens, on the GPU.
call :run_one the_release_teacher_matches_upstream_on_real_weights curricula_parity::the_release_teacher_matches_upstream_on_real_weights --ignored || set "IRIS_FAILED=1"
rem Real multi-rank training on this box: rank r a process on cuda:r (the device inventory above
rem records the cards), over the host TCP transport (Windows has no NCCL). The runner listener
rem exposes one card (cuda:1 is an invalid ordinal), so the scenarios opt explicitly into
rem dist.world.share_device (several ranks on one card); without the opt-in such a rank is refused
rem typed, which the first check proves on one visible card.
call :run_one a_rank_without_its_cuda_device_is_refused_unless_sharing_is_opted_in train_distributed::a_rank_without_its_cuda_device_is_refused_unless_sharing_is_opted_in || set "IRIS_FAILED=1"
set "IRIS_DIST_SHARE_DEVICE=1"
call :run_one fsdp_full_and_grad_op_sharding_reproduce_ddp train_distributed::fsdp_full_and_grad_op_sharding_reproduce_ddp || set "IRIS_FAILED=1"
call :run_one a_sharded_checkpoint_resumes_exactly_and_reshards train_distributed::a_sharded_checkpoint_resumes_exactly_and_reshards || set "IRIS_FAILED=1"
call :run_one a_rank_that_exits_mid_run_fails_the_survivor_typed train_distributed::a_rank_that_exits_mid_run_fails_the_survivor_typed || set "IRIS_FAILED=1"
call :run_one a_rank_that_stalls_mid_run_fails_the_survivor_typed_within_the_timeout train_distributed::a_rank_that_stalls_mid_run_fails_the_survivor_typed_within_the_timeout || set "IRIS_FAILED=1"
rem sc-25690: ranks that resolve different resume checkpoints refuse typed (Windows sockets:
rem the non-blocking polled transport with TCP keepalive).
call :run_one ranks_that_see_different_checkpoints_refuse_to_resume train_distributed::ranks_that_see_different_checkpoints_refuse_to_resume || set "IRIS_FAILED=1"
rem A cancel whose rank-0 cancel-save / export outlasts the cancel grace still ends cleanly.
call :run_one a_cancel_with_a_writer_slower_than_the_grace_still_saves_and_exports train_distributed::a_cancel_with_a_writer_slower_than_the_grace_still_saves_and_exports || set "IRIS_FAILED=1"
set "IRIS_DIST_SHARE_DEVICE="
rem The miniature curricula lifecycle on the device (fp16 trains here: CUDA has half GEMM).
call :run_one fp16_trains_with_the_dynamic_loss_scaler curricula_lifecycle::fp16_trains_with_the_dynamic_loss_scaler || set "IRIS_FAILED=1"
call :run_one stage_presets_chain_256_to_512_with_repa_and_a_new_phase curricula_lifecycle::stage_presets_chain_256_to_512_with_repa_and_a_new_phase || set "IRIS_FAILED=1"
rem sc-25690: the checkpoint lineage on the device (cross-epoch cancel -> resume exact, a fresh
rem run beside another run's checkpoints, the whole-directory export; NTFS renames).
call :run_one a_cancel_at_a_new_epochs_first_batch_after_a_resume_resumes_exactly train_lineage::a_cancel_at_a_new_epochs_first_batch_after_a_resume_resumes_exactly || set "IRIS_FAILED=1"
call :run_one a_fresh_run_starts_its_own_lineage_beside_another_runs_checkpoints train_lineage::a_fresh_run_starts_its_own_lineage_beside_another_runs_checkpoints || set "IRIS_FAILED=1"
call :run_one a_full_model_export_replaces_its_directory_whole train_lineage::a_full_model_export_replaces_its_directory_whole || set "IRIS_FAILED=1"
call :run_one a_refused_resume_leaves_the_checkpoint_root_untouched train_lineage::a_refused_resume_leaves_the_checkpoint_root_untouched || set "IRIS_FAILED=1"
rem The bounded official curriculum on real weights (stage1_256 + REPA -> stage2_512 new phase).
call :run_one real_stage_presets_resume_256_to_512_with_repa curricula_real_weights::real_stage_presets_resume_256_to_512_with_repa --ignored || set "IRIS_FAILED=1"
if not "%IRIS_FAILED%"=="0" exit /b 1
exit /b 0

rem %1 = evidence log name, %2 = test path in the integration binary, %3 = --ignored or empty.
:run_one
set "log=%IRIS_OUT%\%~1.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader > "%IRIS_OUT%\%~1.vram-before.csv" 2>nul
set "TMPDIR=%RUNNER_TEMP%"
set "TEMP=%RUNNER_TEMP%"
cargo test --locked --release -p candle-gen-iris --features cuda --test integration %~2 -- %~3 --exact --nocapture --test-threads 1 > "%log%" 2>&1 || (type "%log%" & echo ::error::%~2 failed & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::%~2 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
exit /b 0
