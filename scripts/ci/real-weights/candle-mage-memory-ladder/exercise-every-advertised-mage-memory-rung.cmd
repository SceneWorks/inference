call "%VCVARS%"
set "MAGE_MEMORY_REFERENCE=%RUNNER_TEMP%\mage-memory-resident.rgb"
set "MAGE_MEMORY_ATTENTION_REFERENCE=%RUNNER_TEMP%\mage-memory-attention.rgb"
for %%R in (resident staged attention blocks) do (
  set "MAGE_MEMORY_RUNG=%%R"
  set "MAGE_MEMORY_OUT=%RUNNER_TEMP%\mage-memory-%%R.rgb"
  cargo test --locked --release -p candle-gen-mage --features cuda,testkit --test integration memory_ladder_real_weights::representative_route_exercises_advertised_rung -- --ignored --exact --nocapture || exit /b 1
)
