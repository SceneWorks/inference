"""The H3 cache handoff must retain native checkpoint/projection adoption coverage."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import check_native_candle_media as guard


class H3CachedRouteTests(unittest.TestCase):
    def test_registered_production_routes_remain_covered(self):
        self.assertEqual(guard.check_candle_media(ROOT), [])

    def test_each_h3_cache_handoff_and_native_fallback_is_required(self):
        original = guard._production
        source = ROOT / "crates/media/candle-gen/candle-gen-minimax-h3/src"
        for entry in guard.ROUTES["minimax_h3"]:
            module, call = entry.split(":", 1)
            target = source / (module + ".rs")
            with self.subTest(module=module, call=call):
                def mutated(path):
                    text = original(path)
                    return text.replace(call, "removed_route(") if path == target else text
                with patch.object(guard, "_production", side_effect=mutated):
                    self.assertIn(
                        f"Candle media minimax_h3 no longer uses {module} / {call}",
                        guard.check_candle_media(ROOT),
                    )


if __name__ == "__main__":
    unittest.main()
