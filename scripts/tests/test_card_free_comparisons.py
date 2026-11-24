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


    def test_a_selection_must_read_every_registered_card_or_exactly_the_named_one(self):
        policy = {"backend": "windows-cuda", "deadlineSeconds": 60, "gpuFreeReserveBytes": 2 * 1024**3,
                  "childGpuCapBytes": 12 * 1024**3, "schemaVersion": 1, "pollMillis": 250, "termGraceMillis": 2000,
                  "stdoutCapBytes": 4096, "stderrCapBytes": 4096, "eventCapBytes": 4096,
                  "childFootprintCapBytes": 8 * 1024**3, "hostFreeReserveBytes": 1024**3}
        _, selection = select(policy, {GPU0: 1000, GPU1: 3000})
        self.assertEqual(resources.executed_card_policy(policy, selection)["cudaDeviceUuid"], GPU0)
        # A record that read only the less-free card is not a most-free selection.
        partial = dict(selection, cards=[selection["cards"][1]], chosen_uuid=GPU1, chosen_index=1)
        with self.assertRaisesRegex(ValueError, "most-free"):
            resources.executed_card_policy(policy, partial)
        # A case that names one card reads exactly that card.
        with TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            _, named = resources.choose_card(policy, Path(directory), named=GPU1,
                                             reader=lambda uuid, _deadline: [card(uuid, 3000)])
        self.assertEqual(resources.executed_card_policy(policy, named, named=GPU1)["cudaDeviceUuid"], GPU1)
        with self.assertRaisesRegex(ValueError, "most-free"):
            resources.executed_card_policy(policy, named)

    def test_claim_and_spawn_estimates_are_checked_against_the_lane(self):
        from scripts.ci import flux1_conditional_comparison as conditional
        estimate = conditional.CASE_DEVICE_ESTIMATES["pulid-pid"]
        conditional.check_spawn_estimate({"staticFloorGpuBytes": estimate[1], "gpuEstimateSource": estimate[0]}, estimate)
        conditional.check_spawn_estimate({"staticFloorGpuBytes": None,
                                          "gpuEstimateSource": "child-footprint-cap-fallback"}, estimate)
        conditional.check_spawn_estimate({"staticFloorGpuBytes": None,
                                          "gpuEstimateSource": "child-footprint-cap-fallback"}, None)
        for bad in ({"staticFloorGpuBytes": estimate[1] - 1, "gpuEstimateSource": estimate[0]},
                    {"staticFloorGpuBytes": estimate[1], "gpuEstimateSource": "other"}):
            with self.assertRaisesRegex(ValueError, "different device estimate"):
                conditional.check_spawn_estimate(bad, estimate)
        executed = {"cudaDeviceUuid": GPU1}
        resources.require_claimed_card({"device_uuid": GPU1}, executed)
        with self.assertRaisesRegex(ValueError, "different card"):
            resources.require_claimed_card({"device_uuid": GPU0}, executed)
        with self.assertRaisesRegex(ValueError, "different device estimate"):
            resources.require_claimed_card({"device_uuid": GPU1, "claim_gpu_estimate": {
                "source": estimate[0], "estimate_bytes": estimate[1] + 1}}, executed, estimate)

    def test_sensenova_refuses_unregistered_cards_and_verifies_its_native_card(self):
        from scripts.ci import sensenova_loop_assets as assets, sensenova_loop_seal as seal
        self.assertEqual(assets.registered_index(GPU1), 1)
        with self.assertRaisesRegex(ValueError, "unregistered card"):
            assets.registered_index("GPU-12345678-1234-1234-1234-123456789abc")
        policy = {"backend": "windows-cuda", "deadlineSeconds": 60, "gpuFreeReserveBytes": 2 * 1024**3,
                  "childGpuCapBytes": 12 * 1024**3, "schemaVersion": 1, "pollMillis": 250, "termGraceMillis": 2000,
                  "stdoutCapBytes": 4096, "stderrCapBytes": 4096, "eventCapBytes": 4096,
                  "childFootprintCapBytes": 8 * 1024**3, "hostFreeReserveBytes": 1024**3}
        _, selection = select(policy, {GPU0: 90000, GPU1: 3000})
        claims = {"q8": {"supervision": {"device_claim": {"device_uuid": GPU1}}},
                  "q4": {"supervision": {"device_claim": {"device_uuid": GPU1}}}}
        seal.verify_native_card({"backend": "candle-cuda", "tiers": claims, "cuda_card_selection": selection}, policy)
        # A named-card native run (the accepted GPU 1 records) carries no selection.
        seal.verify_native_card({"backend": "candle-cuda", "tiers": claims}, dict(policy, cudaDeviceUuid=GPU1))
        claims["q4"]["supervision"]["device_claim"]["device_uuid"] = GPU0
        with self.assertRaisesRegex(ValueError, "different card"):
            seal.verify_native_card({"backend": "candle-cuda", "tiers": claims, "cuda_card_selection": selection}, policy)

if __name__ == "__main__":
    unittest.main()
