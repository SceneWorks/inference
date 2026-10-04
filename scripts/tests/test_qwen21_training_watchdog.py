"""Prevent diagnostics from racing the MLX allocator or changing its lifecycle."""

from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
PROVIDER = ROOT / "crates/media/mlx-gen/mlx-gen-qwen-image-2-1"


def background_violations(source):
    body = source.split("let handle = std::thread::spawn(move || {", 1)[1].split("        Self {", 1)[0]
    return re.findall(r"mlx_rs::|mlx_gen::|\bArray\b|\b(?:eval|async_eval|synchronize)\s*\(", body)


class TrainingWatchdogTests(unittest.TestCase):
    def test_background_sampler_uses_only_physical_and_host_counters(self):
        source = (PROVIDER / "tests/lora_real_weights.rs").read_text(encoding="utf-8")
        self.assertEqual(background_violations(source), [])
        self.assertIn('"background_physical_and_host_only_no_MLX_calls"', source)
        for injected in ["mlx_rs::memory::get_active_memory();", "mlx_rs::memory::get_cache_memory();",
                         "mlx_rs::memory::get_peak_memory();", "mlx_rs::memory::set_cache_limit(0);",
                         "mlx_rs::transforms::eval(arrays);", "synchronize();"]:
            with self.subTest(injected=injected):
                mutant = source.replace("let handle = std::thread::spawn(move || {",
                                        "let handle = std::thread::spawn(move || {" + injected, 1)
                self.assertTrue(background_violations(mutant))

    def test_foreground_trace_adds_no_evaluation_or_allocator_mutation(self):
        source = (PROVIDER / "src/training.rs").read_text(encoding="utf-8")
        trace = source.split("fn training_memory_trace(", 1)[1].split("impl Drop for TrainingPoolBound", 1)[0]
        self.assertEqual(re.findall(r"\b(?:eval|async_eval|synchronize|set_cache_limit|set_memory_limit|reset_peak_memory)\s*\(", trace), [])
        self.assertIn('"foreground_non_atomic_snapshot_no_added_eval_or_sync"', trace)
        self.assertIn('"caption_cache_cleared"', source)
        self.assertIn('"latent_cache_cleared"', source)
        self.assertIn('"step_{step}_gradients_returned"', source)
        self.assertIn('"step_{step}_before_progress"', source)


if __name__ == "__main__":
    unittest.main()
