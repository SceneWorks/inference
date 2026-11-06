"""Run the exact allocator primitive under the existing CUDA lease and owned-child supervisor."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci.model_proof_resources import execution_claim
from scripts.media_campaign_supervisor import load_policy, run_guarded

FQN = 'cuda_proof_pool::constructor_binds_candle_to_an_enforced_pool'


def executable_from_artifacts(path: Path) -> Path:
    expected_source = ROOT / 'crates/media/candle-gen/candle-gen/tests/main.rs'
    selected = []
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        target = row.get('target', {})
        if (row.get('reason') == 'compiler-artifact' and row.get('executable')
                and target.get('name') == 'integration' and target.get('kind') == ['test']
                and Path(target.get('src_path', '')).resolve() == expected_source.resolve()
                and row.get('profile', {}).get('test') is True
                and {'cuda', 'testkit'} <= set(row.get('features', []))):
            selected.append(Path(row['executable']).resolve())
    if len(selected) != 1 or not selected[0].is_file():
        raise ValueError('one existing exact CUDA/testkit candle-gen integration executable required')
    return selected[0]


def run(output: Path, executable: Path, env: dict[str, str], *, claim=execution_claim,
        supervise=run_guarded, clock=time.monotonic) -> dict:
    uuid = env.get('CUDA_VISIBLE_DEVICES', '')
    if (not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', uuid)
            or env.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID' or env.get('SCENEWORKS_GPU_ID') != '1'):
        raise ValueError('explicit physical GPU1 UUID visibility and PCI_BUS_ID order required')
    if env.get('SCENEWORKS_CUDA_PROOF_POOL_BYTES') != '536870912':
        raise ValueError('the primitive requires its explicit 512 MiB pool cap')
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(schemaVersion=1, backend='windows-cuda', deadlineSeconds=1800,
                   pollMillis=100, termGraceMillis=5000, hostFreeReserveBytes=16 << 30,
                   childFootprintCapBytes=16 << 30, stdoutCapBytes=64 << 20,
                   stderrCapBytes=64 << 20, eventCapBytes=128 << 20,
                   cudaDeviceUuid=uuid, gpuFreeReserveBytes=2 << 30, childGpuCapBytes=8 << 30)
    policy_path = output / 'allocator-supervisor-policy.json'
    policy_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    policy = load_policy(policy_path)
    deadline = clock() + policy.deadline_seconds
    receipt = {'kind': 'explicit-allocator-primitive', 'accepted': False,
               'command': [str(executable), FQN, '--exact', '--ignored', '--nocapture', '--test-threads=1'],
               'claim_scope': 'physical GPU1 lease through owned child exit and post-execution verification'}
    try:
        with claim('candle-cuda', policy, output, env,
                   constraints={'required_cuda_physical_index': 1, 'required_cuda_device_uuid': uuid},
                   original_deadline_monotonic=deadline) as physical:
            receipt['physical_execution'] = physical
            remaining = deadline - clock()
            if remaining <= policy.term_grace_millis / 1000:
                raise TimeoutError('allocator proof deadline expired while waiting for its GPU lease')
            execution_policy = output / 'allocator-execution-policy.json'
            execution_policy.write_text(json.dumps({**payload, 'deadlineSeconds': remaining}, indent=2) + '\n', encoding='utf-8')
            result = supervise(receipt['command'], cwd=ROOT, env=env,
                               policy=load_policy(execution_policy),
                               stdout_path=output / 'allocator-proof.log',
                               stderr_path=output / 'allocator-proof.stderr.log',
                               event_path=output / 'allocator-proof.events.jsonl',
                               require_gpu_sample=False)
            receipt['supervision'] = asdict(result)
            if result.returncode != 0:
                raise RuntimeError(f'allocator primitive exited {result.returncode}')
            text = (output / 'allocator-proof.log').read_text(encoding='utf-8')
            if len(re.findall(r'test result: ok\. 1 passed; 0 failed; 0 ignored', text)) != 1:
                raise ValueError('exactly one passing non-skipped primitive required')
            markers = re.findall(r'CUDA_PROOF_POOL_EVIDENCE (\{[^\n]+\})', text)
            if len(markers) != 1:
                raise ValueError('exactly one positive allocator enforcement receipt required')
            evidence = json.loads(markers[0])
            count = evidence.get('successfulRecoveryCount')
            exercised = evidence.get('fragmentationRecoveryExercised')
            byte_fields = ('capBytes', 'heldBytes', 'usedHighBytes', 'reservedHighBytes')
            if (not all(evidence.get(key) is True for key in
                        ('overCapRefused', 'bothConstructors', 'cublasStreamResetRebound'))
                    or not all(type(evidence.get(key)) is int for key in byte_fields)
                    or evidence['capBytes'] != 536870912
                    or not evidence['capBytes'] // 2 <= evidence['heldBytes']
                    <= evidence['usedHighBytes'] <= evidence['reservedHighBytes'] <= evidence['capBytes']
                    or type(count) is not int or count < 0
                    or type(exercised) is not bool or exercised != (count > 0)):
                raise ValueError('allocator enforcement evidence is incomplete, inconsistent or outside its cap')
            receipt['allocator_evidence'] = evidence
        if not receipt['physical_execution'].get('post_execution_verified'):
            raise ValueError('physical execution release verification did not complete')
        receipt['accepted'] = True
        return receipt
    except BaseException as error:
        receipt['error'] = {'type': type(error).__name__, 'detail': str(error)}
        raise
    finally:
        (output / 'allocator-execution.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve(), executable_from_artifacts(args.artifacts), os.environ.copy())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
