"""Regenerate synthetic CPU-only MOSS Qwen cast-order fixtures (no model/weights).

Frozen Transformers5.5.3 Qwen3.5 expressions: modeling_qwen3_5.py
SHA256 aee59d55ee4e8ce0e50bf0e279796b85c2c66a28dfae55c3fcbb62fa9bcba048,
lines228–250/262–263 (Q),493 (log decay),720–725 (zero-centered norm).
The old-boundary alternatives are discriminators, not acceptance thresholds.
"""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F


def values(x):
    assert x.device.type == 'cpu'
    return x.float().flatten().tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    i = torch.arange(16, dtype=torch.float32, device='cpu')
    a = (((i * 17).remainder(113) - 56) / 16).bfloat16()
    bias = (((i * 7).remainder(29) - 14) / 32).bfloat16()
    log = (((i * 3).remainder(23) - 11) / 8).bfloat16()
    decay = (-log.float().exp() * F.softplus(a.float() + bias.float())).exp()
    q = (((torch.arange(128, dtype=torch.float32) * 13).remainder(101) - 50) / 32).bfloat16()
    # Both paths deliberately retain the same BF16 normalization; only scaling differs.
    qn = q * torch.rsqrt((q * q).sum(keepdim=True, dim=-1) + 1e-6)
    inv = torch.tensor(128 ** -0.5, dtype=torch.float32)
    x = (((i * 11).remainder(127) - 63) / 16).reshape(2,8).bfloat16()
    w = (((torch.arange(8, dtype=torch.float32) * 19).remainder(61) - 30) / 512).bfloat16()
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    expected_norm = (normalized * (1 + w.float())).bfloat16()
    old_norm = (normalized * (1 + w).float()).bfloat16()
    cases = {
        'decay': {'a':values(a), 'a_log':values(log), 'bias':values(bias), 'expected':values(decay), 'old':values(decay.bfloat16())},
        'query': {'normalized':values(qn), 'inverse_head_dim':float(inv), 'expected':values(qn.float()*inv), 'old':values(qn*inv.bfloat16())},
        'norm': {'x':values(x), 'weight':values(w), 'expected':values(expected_norm), 'old':values(old_norm)},
    }
    for name, c in cases.items():
        assert c['expected'] != c['old'], name
    output = Path(__file__).resolve().parents[3] / 'llm/mlx-llm/src/models/testdata/moss_qwen_precision.json'
    payload = json.dumps({'source':'Transformers5.5.3 Qwen3.5 frozen expressions; synthetic CPU only', 'torch':torch.__version__, 'cases':cases}, indent=2)+'\n'
    if args.check:
        assert output.read_text() == payload, 'CPU fixture differs; inspect runtime/source before regenerating'
    else:
        output.write_text(payload)
    print(output)


if __name__ == '__main__':
    main()
