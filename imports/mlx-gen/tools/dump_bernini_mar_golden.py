"""sc-5140: synthetic-fixture golden for the Bernini planner's MAR semantic-planning loop.

Composes a *tiny* but structurally-faithful Qwen2.5-VL text backbone + `MLPConnector` +
`SimpleMLPAdaLN` (DiffLoss_FM) and drives them through the **reference** `sample_vit_embed` loop
(`_vendor/bernini/bernini/pipeline.py` 724-884) + `feat_from_planner_to_renderer` (`bernini.py`),
dumping weights + the 3-stream inputs + the injected reveal order + per-step noise + the 4 output
streams + the filled `pred_vit_embed`.

Every sub-module's math is copied **verbatim** from the reference (the same copies used by the
qwen-backbone / clip-diff / handoff goldens), so the oracle is the reference. The two RNG consumers
are **injected** — the reveal permutation `order` (normally `np.random.shuffle`) and the per-step FM
base noise (normally `torch.randn` inside `DiffLoss_FM.sample`) — so the Rust port matches the
trajectory bit-for-bit. The chosen `order` exercises a single-token reveal, the reference's
`nonzero().sum()==0` skip (a lone token-0 reveal that stays masked), and two multi-token reveals.

Run:
  ~/Repos/mflux/.venv/bin/python tools/dump_bernini_mar_golden.py
Fixture -> mlx-gen-bernini/tests/fixtures/mar_golden.safetensors
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO_ROOT, "mlx-gen-bernini", "tests", "fixtures", "mar_golden.safetensors")

# --- tiny but structurally faithful dims ---
HIDDEN = 16          # planner hidden / connector in / clip-diff in+z channels
LAYERS = 2
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 8
INTER = 32
EPS = 1e-6
THETA = 1_000_000.0
MROPE = [1, 2, 1]    # sum*2 == HEAD_DIM
GEN = 12             # connector for_gen out (renderer prompt-embed width)
WIDTH = 24           # clip-diff model_channels
DEPTH = 2            # clip-diff res blocks
SHIFT = 2.0

PLANNING_STEP = 4
VIT_DENOISING_STEP = 2
VIT_TXT_CFG = 1.4
VIT_IMG_CFG = 1.2
N_QUERY = 6
ORDER = [3, 1, 5, 2, 4, 0]   # reveal: step0 -> {0} (skipped), step1 -> {4}, step2 -> {2,5}, step3 -> {1,3}


# ===== verbatim: qwen2.5-vl text backbone (modeling_qwen2_5_vl.py) =====
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def rotary_cos_sin(position_ids):
    half = HEAD_DIM // 2
    inv_freq = 1.0 / (THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    pos = position_ids[:, None, :].float()
    inv_exp = inv_freq[None, None, :, None].expand(3, 1, half, 1)
    pos_exp = pos[:, :, None, :]
    freqs = (inv_exp @ pos_exp).transpose(2, 3)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rms_norm(x, w):
    dt = x.dtype
    x = x.to(torch.float32)
    v = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(v + EPS)
    return w * x.to(dt)


def repeat_kv(h, n_rep):
    b, kvh, s, d = h.shape
    if n_rep == 1:
        return h
    h = h[:, :, None, :, :].expand(b, kvh, n_rep, s, d)
    return h.reshape(b, kvh * n_rep, s, d)


def qwen_penultimate(embeds, position_ids, mask, weights):
    """Reference forward; returns hidden_states[-2] (the penultimate residual stream)."""
    seq = embeds.shape[1]
    cos3, sin3 = rotary_cos_sin(position_ids)
    all_hidden = []
    hidden = embeds
    for i in range(LAYERS):
        all_hidden.append(hidden)
        p = f"model.layers.{i}"
        residual = hidden
        x = rms_norm(hidden, weights[f"{p}.input_layernorm.weight"])
        q = F.linear(x, weights[f"{p}.self_attn.q_proj.weight"], weights[f"{p}.self_attn.q_proj.bias"])
        k = F.linear(x, weights[f"{p}.self_attn.k_proj.weight"], weights[f"{p}.self_attn.k_proj.bias"])
        v = F.linear(x, weights[f"{p}.self_attn.v_proj.weight"], weights[f"{p}.self_attn.v_proj.bias"])
        q = q.view(1, seq, HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(1, seq, KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(1, seq, KV_HEADS, HEAD_DIM).transpose(1, 2)
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos3, sin3, MROPE)
        k = repeat_kv(k, HEADS // KV_HEADS)
        v = repeat_kv(v, HEADS // KV_HEADS)
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(HEAD_DIM)
        attn = attn + mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(1, seq, HEADS * HEAD_DIM)
        out = F.linear(out, weights[f"{p}.self_attn.o_proj.weight"])
        hidden = residual + out
        residual = hidden
        x = rms_norm(hidden, weights[f"{p}.post_attention_layernorm.weight"])
        gate = F.linear(x, weights[f"{p}.mlp.gate_proj.weight"])
        up = F.linear(x, weights[f"{p}.mlp.up_proj.weight"])
        x = F.linear(F.silu(gate) * up, weights[f"{p}.mlp.down_proj.weight"])
        hidden = residual + x
    final = rms_norm(hidden, weights["model.norm.weight"])
    all_hidden.append(final)
    return all_hidden[-2]


# ===== verbatim: bernini.py RMSNorm + MLPConnector =====
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


class MLPConnector(nn.Module):
    def __init__(self, in_dim, out_dim_for_gen, out_dim_for_vit):
        super().__init__()
        self.proj_gen = nn.Sequential(
            nn.Linear(in_dim, out_dim_for_gen), nn.GELU(),
            RMSNorm(out_dim_for_gen), nn.Linear(out_dim_for_gen, out_dim_for_gen),
        )
        self.pred_vit = nn.Sequential(
            nn.Linear(in_dim, out_dim_for_vit), nn.GELU(), nn.Linear(out_dim_for_vit, out_dim_for_vit),
            RMSNorm(out_dim_for_vit), nn.Linear(out_dim_for_vit, out_dim_for_vit),
        )

    def for_gen(self, x):
        return self.proj_gen(x)

    def for_vit(self, x):
        return self.pred_vit(x)


# ===== verbatim: diffloss_fm.py net + scheduler =====
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size).to(t.dtype))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(channels, channels, bias=True), nn.SiLU(), nn.Linear(channels, channels, bias=True))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        return x + gate_mlp * self.mlp(h)


class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class SimpleMLPAdaLN(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, z_channels, num_res_blocks):
        super().__init__()
        self.in_channels = in_channels
        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = FinalLayer(model_channels, out_channels)

    def forward(self, x, t, c):
        x = self.input_proj(x)
        y = self.time_embed(t) + self.cond_embed(c)
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x, y)

    def forward_with_txt_img_cfg(self, x, t, c, txt_cfg_scale, img_cfg_scale):
        part = x[: len(x) // 3]
        combined = torch.cat([part, part, part], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps, imgcond_eps = torch.split(eps, len(eps) // 3, dim=0)
        part_eps = uncond_eps + img_cfg_scale * (imgcond_eps - uncond_eps) + txt_cfg_scale * (cond_eps - imgcond_eps)
        eps = torch.cat([part_eps, part_eps, part_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class FlowMatchScheduler:
    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0,
                 sigma_max=1.0, sigma_min=0.003 / 1.002, extra_one_step=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, shift=None, dtype=torch.float32):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1, dtype=dtype)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps, dtype=dtype)
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

    def step(self, model_output, timestep, sample, to_final=False):
        tid = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[tid]
        sigma_ = 0 if (to_final or tid + 1 >= len(self.timesteps)) else self.sigmas[tid + 1]
        return sample + model_output * (sigma_ - sigma)


def sample_clip_diff(net, sched, z, noise_base, txt_cfg, img_cfg, num_steps):
    """DiffLoss_FM.sample triple-CFG path with **injected** base noise (replaces torch.randn)."""
    noise = torch.cat([noise_base, noise_base, noise_base], dim=0)
    sched.set_timesteps(num_steps)
    samples = noise.to(z.dtype)
    for t in sched.timesteps:
        timestep = t.unsqueeze(0).to(z.dtype)
        pred = net.forward_with_txt_img_cfg(samples, timestep, z, txt_cfg, img_cfg)
        samples = sched.step(pred, timestep, samples)
    return samples


# ===== verbatim: bernini.py feat_from_planner_to_renderer (inference branch) =====
def feat_from_planner_to_renderer(connector, hidden_states, visual_output_mask):
    pred_vit_mask = visual_output_mask.squeeze(0)
    txt_and_vit_mask = pred_vit_mask.logical_not()
    cond_embed_mask = txt_and_vit_mask | pred_vit_mask
    txt_mask = txt_and_vit_mask[cond_embed_mask]
    vit_mask = pred_vit_mask[cond_embed_mask]
    contexts = connector.for_gen(hidden_states[:, cond_embed_mask, :])
    return contexts, txt_mask, vit_mask


def gen_mask(length, positions):
    m = torch.zeros(1, length, dtype=torch.bool)
    for p in positions:
        m[0, p] = True
    return m


def causal_mask(seq):
    neg = torch.finfo(torch.float32).min
    return torch.triu(torch.full((seq, seq), neg), diagonal=1)[None, None]


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32)

    # ---- backbone weights ----
    weights = {}

    def lin(prefix, out_f, in_f, bias):
        weights[f"{prefix}.weight"] = rand(out_f, in_f) * 0.05
        if bias:
            weights[f"{prefix}.bias"] = rand(out_f) * 0.05

    weights["model.embed_tokens.weight"] = rand(HIDDEN, HIDDEN) * 0.05  # present; unused (embeds path)
    weights["model.norm.weight"] = torch.ones(HIDDEN) + rand(HIDDEN) * 0.02
    for i in range(LAYERS):
        p = f"model.layers.{i}"
        weights[f"{p}.input_layernorm.weight"] = torch.ones(HIDDEN) + rand(HIDDEN) * 0.02
        weights[f"{p}.post_attention_layernorm.weight"] = torch.ones(HIDDEN) + rand(HIDDEN) * 0.02
        lin(f"{p}.self_attn.q_proj", HEADS * HEAD_DIM, HIDDEN, True)
        lin(f"{p}.self_attn.k_proj", KV_HEADS * HEAD_DIM, HIDDEN, True)
        lin(f"{p}.self_attn.v_proj", KV_HEADS * HEAD_DIM, HIDDEN, True)
        lin(f"{p}.self_attn.o_proj", HIDDEN, HEADS * HEAD_DIM, False)
        lin(f"{p}.mlp.gate_proj", INTER, HIDDEN, False)
        lin(f"{p}.mlp.up_proj", INTER, HIDDEN, False)
        lin(f"{p}.mlp.down_proj", HIDDEN, INTER, False)

    conn = MLPConnector(HIDDEN, GEN, HIDDEN).to(torch.float32).eval()
    net = SimpleMLPAdaLN(HIDDEN, WIDTH, HIDDEN, HIDDEN, DEPTH).to(torch.float32).eval()
    sched = FlowMatchScheduler(num_inference_steps=VIT_DENOISING_STEP, shift=SHIFT, extra_one_step=True)
    mask_token = rand(1, 1, HIDDEN)  # mask_tokens[:, :1]

    # ---- 3 streams (different L / gen positions, same n_query) ----
    streams = {
        "cond":    dict(L=12, gen=[6, 7, 8, 9, 10, 11]),
        "uncond":  dict(L=10, gen=[4, 5, 6, 7, 8, 9]),
        "imgcond": dict(L=11, gen=[5, 6, 7, 8, 9, 10]),
    }
    for name, s in streams.items():
        L = s["L"]
        s["embeds"] = rand(1, L, HIDDEN)
        s["pos"] = torch.stack([torch.arange(L)] * 3, dim=0)  # text-style position ids (3, L)
        s["mask4d"] = causal_mask(L)
        s["vom"] = gen_mask(L, s["gen"])
        # post_process_input_embeds: set the gen slots to the mask_token (start fully masked).
        emb = s["embeds"].clone()
        emb[:, s["vom"].squeeze(0), :] = mask_token.expand(1, N_QUERY, HIDDEN)
        s["input"] = emb

    # ---- the MAR loop (verbatim pipeline.py 743-834) with injected order + per-step noise ----
    mask_ratio_gen = lambda step, totals: math.cos(math.pi / 2.0 * (step + 1) / totals)
    order = torch.tensor(ORDER).long()
    mask = torch.ones(N_QUERY)
    g_noise = torch.Generator().manual_seed(123)
    step_noise = {}  # step -> [np, HIDDEN] base noise actually consumed (placeholder for skips)

    in_embeds = {k: streams[k]["input"].clone() for k in streams}
    for step in range(PLANNING_STEP):
        hids = {k: qwen_penultimate(in_embeds[k], streams[k]["pos"], streams[k]["mask4d"], weights) for k in streams}
        pred_mllm = {k: conn.for_vit(hids[k][:, streams[k]["vom"].squeeze(0), :]) for k in streams}

        mask_ratio = mask_ratio_gen(step, PLANNING_STEP)
        mask_len = torch.tensor([math.floor(N_QUERY * mask_ratio)])
        mask_len = torch.maximum(torch.tensor([1.0]), torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))
        mask_next = torch.zeros_like(mask)
        mask_next = torch.scatter(mask_next, dim=-1, index=order[: mask_len.long().item()], src=torch.ones_like(mask)).bool()
        if step >= PLANNING_STEP - 1:
            mask_to_pred = mask.bool()
        else:
            mask_to_pred = torch.logical_xor(mask.bool(), mask_next)
        mask = mask_next.float()

        revealed = mask_to_pred.nonzero(as_tuple=True)[0]
        store = torch.zeros(1, HIDDEN)  # placeholder dumped for skipped steps (unused by the port)
        if revealed.sum() != 0:
            np_step = revealed.shape[0]
            noise_base = torch.randn(np_step, HIDDEN, generator=g_noise)
            store = noise_base
            cond_p = pred_mllm["cond"][:, revealed]
            uncond_p = pred_mllm["uncond"][:, revealed]
            imgcond_p = pred_mllm["imgcond"][:, revealed]
            z = torch.cat([cond_p, uncond_p, imgcond_p], dim=1)[0]  # [3*np, HIDDEN]
            sampled = sample_clip_diff(net, sched, z, noise_base, VIT_TXT_CFG, VIT_IMG_CFG, VIT_DENOISING_STEP)
            cur = sampled[: sampled.shape[0] // 3].unsqueeze(0)  # [1, np, HIDDEN]
            # write the revealed predictions back into all three streams' gen slots
            buf = in_embeds["cond"][:, streams["cond"]["vom"].squeeze(0), :].clone()
            buf[:, revealed] = cur
            for k in streams:
                in_embeds[k][:, streams[k]["vom"].squeeze(0), :] = buf
        step_noise[step] = store.contiguous()

    pred_vit_embed = in_embeds["cond"][:, streams["cond"]["vom"].squeeze(0), :]

    # ---- handoff: final cond + uncond forward -> feat_from_planner_to_renderer -> 4 streams ----
    cond_hidden = qwen_penultimate(in_embeds["cond"], streams["cond"]["pos"], streams["cond"]["mask4d"], weights)
    uncond_hidden = qwen_penultimate(in_embeds["uncond"], streams["uncond"]["pos"], streams["uncond"]["mask4d"], weights)
    cond_ctx, cond_txt, cond_vit = feat_from_planner_to_renderer(conn, cond_hidden, streams["cond"]["vom"])
    uncond_ctx, uncond_txt, _ = feat_from_planner_to_renderer(conn, uncond_hidden, streams["uncond"]["vom"])
    wtxt_wvit = cond_ctx
    wtxt_wovit = cond_ctx[:, cond_txt]
    wotxt_wvit = cond_ctx[:, cond_vit]
    wotxt_wovit = uncond_ctx[:, uncond_txt]

    # ---- dump ----
    out = {f"w.{k}": v.contiguous() for k, v in weights.items()}
    for k, v in conn.state_dict().items():
        out[f"conn.{k}"] = v.contiguous()
    for k, v in net.state_dict().items():
        out[f"net.{k}"] = v.contiguous()
    out["io.mask_token"] = mask_token.contiguous()
    out["io.order"] = order.to(torch.int32).contiguous()
    for name, s in streams.items():
        out[f"io.{name}.input"] = s["input"].contiguous()
        out[f"io.{name}.pos"] = s["pos"].to(torch.int32).contiguous()
        out[f"io.{name}.mask4d"] = s["mask4d"].contiguous()
        out[f"io.{name}.gen_idx"] = torch.tensor(s["gen"], dtype=torch.int32).contiguous()
    for step, v in step_noise.items():
        out[f"io.noise.{step}"] = v.contiguous()
    out["out.pred_vit_embed"] = pred_vit_embed.contiguous()
    out["out.wtxt_wvit"] = wtxt_wvit.contiguous()
    out["out.wtxt_wovit"] = wtxt_wovit.contiguous()
    out["out.wotxt_wvit"] = wotxt_wvit.contiguous()
    out["out.wotxt_wovit"] = wotxt_wovit.contiguous()

    meta = {
        "hidden": str(HIDDEN), "layers": str(LAYERS), "heads": str(HEADS), "kv_heads": str(KV_HEADS),
        "head_dim": str(HEAD_DIM), "intermediate": str(INTER), "mrope_section": ",".join(map(str, MROPE)),
        "gen": str(GEN), "width": str(WIDTH), "depth": str(DEPTH), "shift": repr(SHIFT),
        "planning_step": str(PLANNING_STEP), "vit_denoising_step": str(VIT_DENOISING_STEP),
        "vit_txt_cfg": repr(VIT_TXT_CFG), "vit_img_cfg": repr(VIT_IMG_CFG),
        "n_query": str(N_QUERY), "order": ",".join(map(str, ORDER)), "eps": repr(EPS), "theta": repr(THETA),
    }
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    save_file(out, FIXTURE, metadata=meta)
    print(f"wrote {FIXTURE}  ({len(out)} tensors)")
    print(f"  pred_vit_embed {tuple(pred_vit_embed.shape)}  wtxt_wvit {tuple(wtxt_wvit.shape)}  "
          f"wtxt_wovit {tuple(wtxt_wovit.shape)}  wotxt_wvit {tuple(wotxt_wvit.shape)}  "
          f"wotxt_wovit {tuple(wotxt_wovit.shape)}")
    # token 0 stays the mask_token (revealed alone in step0 -> nonzero().sum()==0 -> skipped)
    tok0 = pred_vit_embed[0, 0]
    print(f"  token0 == mask_token (skip quirk): {torch.allclose(tok0, mask_token[0, 0], atol=1e-6)}")


if __name__ == "__main__":
    main()
