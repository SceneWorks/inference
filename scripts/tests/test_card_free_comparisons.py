"""A CUDA comparison request that names no card runs on whichever registered card fits.

Metadata only: no driver query, GPU claim or model run. Named-card requests keep their policy and
record no selection, so earlier named-GPU0 evidence verifies unchanged."""
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.ci import model_proof_resources as resources
from scripts.ci import sga_adapter_comparison as adapter
from scripts.ci import sga_terminal_comparison as terminal

GPU0, GPU1 = sorted(resources.REGISTERED_CUDA_CARDS, key=resources.REGISTERED_CUDA_CARDS.get)


def card(uuid, used_mib):
    return {'uuid': uuid, 'index': resources.REGISTERED_CUDA_CARDS[uuid], 'total_mib': 97887,
            'used_mib': used_mib, 'free_mib': 97887 - used_mib, 'utilization_percent': 3}


def select(payload, used):
    """GPU 0 holds `used[GPU0]` MiB of another process; choose the card as a producer would."""
    with TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
        return resources.choose_card(payload, Path(directory),
                                     reader=lambda uuid, _deadline: [card(uuid, used[uuid])])


class CardFreeComparisons(unittest.TestCase):
    def test_sga_adapter_request_without_a_card_selects_gpu1_when_gpu0_is_short(self):
        request = {"selection": "sga-adapter-tier1", "profile": next(p for b, p in adapter.PROFILES if b == "candle-cuda"),
                   "download_missing": True}
        adapter.request_value("candle-cuda", request)
        policy = adapter.policy_value("candle-cuda", request)
        self.assertTrue(resources.card_free_policy(policy))
        named = adapter.policy_value("candle-cuda", dict(request, cuda_device_uuid=GPU0))
        self.assertEqual(named["cudaDeviceUuid"], GPU0)
        payload, selection = select(policy, {GPU0: 90000, GPU1: 2000})
        self.assertEqual(payload, dict(policy, cudaDeviceUuid=GPU1))
        self.assertEqual(resources.executed_card_policy(policy, selection), payload)
        # A named-card record carries no selection and verifies against its own policy.
        self.assertEqual(resources.executed_card_policy(named, None), named)
        with self.assertRaises(ValueError):
            resources.executed_card_policy(named, selection)
        with self.assertRaises(ValueError):
            resources.executed_card_policy(policy, None)

    def test_sga_terminal_scored_request_may_omit_its_card_but_diagnostics_stay_named(self):
        request = {"selection": terminal.SELECTION, "profile": "ltx25", "variant": "dev", "download_missing": True}
        self.assertIsNotNone(terminal.profile_for("candle-cuda", request))
        revision = "1" * 40
        with (patch.object(terminal.proof, "repository_identity", return_value={"revision": revision}),
              patch.object(terminal.subprocess, "run", return_value=SimpleNamespace(returncode=0)),
              patch.object(terminal.paired, "git", return_value="2" * 40),
              patch.object(terminal, "terminal_overlay", return_value=(lambda *_: {}, {}))):
            contract = terminal.plan("candle-cuda", request, revision)
        self.assertNotIn("cudaDeviceUuid", contract["policy"])
        _, selection = select(contract["policy"], {GPU0: 1000, GPU1: 90000})
        ran = terminal.ran_contract(contract, selection)
        self.assertEqual((ran["policy"]["cudaDeviceUuid"], ran["request"]["cuda_device_uuid"]), (GPU0, GPU0))
        self.assertEqual(terminal.ran_contract(dict(contract, policy=dict(contract["policy"], cudaDeviceUuid=GPU1)), None)["policy"]["cudaDeviceUuid"], GPU1)
        with self.assertRaisesRegex(ValueError, "device binding|exact GPU|approved"):
            terminal.profile_for("candle-cuda", dict(request, diagnostic=terminal.WINDOWS_DEV_DIAGNOSTIC))

    def test_planners_validate_a_card_free_policy_with_a_registered_card(self):
        policy = {"backend": "windows-cuda", "deadlineSeconds": 60}
        self.assertIn(resources.planned_card_policy(policy)["cudaDeviceUuid"], resources.REGISTERED_CUDA_CARDS)
        named = dict(policy, cudaDeviceUuid=GPU1)
        self.assertIs(resources.planned_card_policy(named), named)


if __name__ == "__main__":
    unittest.main()
