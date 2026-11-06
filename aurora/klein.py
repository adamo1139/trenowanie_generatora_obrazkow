"""FLUX.2 [klein] base 4B as a frozen teacher, in this repo's conventions.

klein works in the same latent space as our denoiser: FLUX.2 VAE latents, 2x2-packed to 128 channels and normalised
with the VAE's batch-norm statistics (B, 128, h, w), rectified flow z_t = (1 - t) x + t eps with t = sigma, velocity
eps - x (diffusers' FlowMatchEulerDiscreteScheduler: x_next = x + (sigma_next - sigma) v). So its velocity can be used
as a target for ours directly; only the text side differs: Qwen3-4B with the chat template, the hidden states of
layers 9 / 18 / 27 stacked to width 7680, padded to 512 tokens (no mask; that is how klein was trained), and 4D (t, h,
w, l) positions for text and image tokens. Mirrors diffusers' Flux2KleinPipeline (0.42).
"""
import math

import torch

REPO = "black-forest-labs/FLUX.2-klein-base-4B"


def klein_sigmas(steps: int, tokens: int, device=None) -> torch.Tensor:
    """The pipeline's sampling schedule: linspace(1, 1/steps) shifted by its resolution / step-count dependent mu
    (compute_empirical_mu, exponential time shift), then a final 0. -> (steps + 1,)"""
    a1, b1, a2, b2 = 8.73809524e-05, 1.89833333, 0.00016927, 0.45666666
    if tokens > 4300:
        mu = a2 * tokens + b2
    else:
        m200, m10 = a2 * tokens + b2, a1 * tokens + b1
        a = (m200 - m10) / 190.0
        mu = a * steps + (m200 - 200.0 * a)
    s = torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64)
    s = math.exp(mu) / (math.exp(mu) + (1.0 / s - 1.0))
    return torch.cat([s, s.new_zeros(1)]).float().to(device)


def tokenizer(repo: str = REPO):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")


def tokenize(tok, prompts, max_len: int = 512):
    """Qwen3 chat template, padded / truncated to max_len, as the pipeline. -> (B, L) ids, (B, L) mask, int64 CPU."""
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
    return enc["input_ids"].long(), enc["attention_mask"].long()


class KleinTeacher:
    """Frozen klein transformer (+ its Qwen3 text encoder for encode()); plain object, never trained."""

    def __init__(self, device, repo: str = REPO, dtype=torch.bfloat16, text_encoder: bool = True):
        from diffusers import Flux2Transformer2DModel
        self.device, self.dtype = device, dtype
        self.transformer = Flux2Transformer2DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=dtype)
        self.transformer.to(device).eval().requires_grad_(False)
        self.te = self.tok = None
        if text_encoder:
            from transformers import Qwen3ForCausalLM
            self.tok = tokenizer(repo)
            self.te = Qwen3ForCausalLM.from_pretrained(repo, subfolder="text_encoder", torch_dtype=dtype)
            self.te.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, prompts, max_len: int = 512) -> torch.Tensor:
        """Prompts -> (B, max_len, 7680) text states, as Flux2KleinPipeline._get_qwen3_prompt_embeds."""
        ids, mask = tokenize(self.tok, prompts, max_len)
        return self.encode_ids(ids, mask)

    @torch.no_grad()
    def encode_ids(self, ids, mask, layers=(9, 18, 27), chunk: int = 16) -> torch.Tensor:
        """tokenize()'s (B, L) ids and attention mask -> (B, L, 7680) text states."""
        outs = []
        for i in range(0, len(ids), chunk):
            out = self.te(input_ids=ids[i:i + chunk].to(self.device), attention_mask=mask[i:i + chunk].to(self.device),
                          output_hidden_states=True, use_cache=False)
            h = torch.stack([out.hidden_states[k] for k in layers], dim=1)  # (b, 3, L, 2560)
            b, c, l, d = h.shape
            outs.append(h.permute(0, 2, 1, 3).reshape(b, l, c * d).to(self.dtype))
        return torch.cat(outs)

    @staticmethod
    def _ids(b, h, w, l_txt, device):
        img = torch.cartesian_prod(torch.arange(1), torch.arange(h), torch.arange(w), torch.arange(1))
        txt = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(l_txt))
        return img.to(device).expand(b, -1, -1), txt.to(device).expand(b, -1, -1)

    @torch.no_grad()
    def velocity(self, x, t, text, null_text=None, cfg: float = 4.0) -> torch.Tensor:
        """x: (B, 128, h, w) normalised latents at time t (B,) in [0, 1]; text: (B, L, 7680); null_text: (1 or B, L,
        7680) states of the empty prompt (the pipeline's negative prompt) for CFG. -> (B, 128, h, w) fp32 velocity
        eps - x (guided: v_null + cfg (v_text - v_null))."""
        b, c, h, w = x.shape
        z = x.to(self.dtype).reshape(b, c, h * w).transpose(1, 2)
        guided = null_text is not None and cfg != 1.0
        if guided:
            z = torch.cat([z, z])
            text = torch.cat([text, null_text.expand(b, -1, -1).to(text.dtype)])
            t = torch.cat([t, t])
        img_ids, txt_ids = self._ids(z.shape[0], h, w, text.shape[1], x.device)
        v = self.transformer(hidden_states=z, timestep=t.to(self.dtype), guidance=None, encoder_hidden_states=text,
                             txt_ids=txt_ids, img_ids=img_ids, return_dict=False)[0].float()
        if guided:
            v_text, v_null = v.chunk(2)
            v = v_null + cfg * (v_text - v_null)
        return v.transpose(1, 2).reshape(b, c, h, w).contiguous()

    @torch.no_grad()
    def sample(self, text, null_text, noise, steps: int = 50, cfg: float = 4.0) -> torch.Tensor:
        """Euler with the pipeline's schedule, from `noise` (B, 128, h, w). -> normalised latents."""
        sig = klein_sigmas(steps, noise.shape[2] * noise.shape[3], noise.device)
        z = noise.float()
        for k in range(steps):
            v = self.velocity(z, sig[k].expand(z.shape[0]), text, null_text, cfg)
            z = z + (sig[k + 1] - sig[k]) * v
        return z


# ---- teacher server (train_pp.py --opd): ranks holding klein, serving the training ranks' on-policy states ----------
# Protocol per request, control on the gloo group (any ready client), tensors on each pair's NCCL link:
#   gloo:  header [1, n_states, n_captions, h, w] (or [0] = stop), caption ids (n_captions, L), mask (n_captions, L)
#   NCCL:  states (n, 128, h, w) bf16, t (n,) fp32, caption index per state (n,) int64  ->  guided velocity (n, 128, h, w)
TAG_HDR, TAG_IDS, TAG_MASK = 21, 22, 23


def teacher_server(repo, clients, device, log, control_group, cfg: float = 4.0, micro: int = 32):
    import time
    import torch.distributed as dist
    teacher = KleinTeacher(device, repo)
    null = teacher.encode([""])
    log(f"teacher server (rank {dist.get_rank()}): {repo}, CFG {cfg}, for ranks {sorted(clients)}", flush=True)
    active, served, t0 = set(clients), 0, time.perf_counter()
    busy, win_t, win_n = 0.0, t0, 0
    while active:
        hdr = torch.empty(5, dtype=torch.long)
        r = dist.recv(hdr, group=control_group, tag=TAG_HDR)
        if hdr[0].item() == 0:
            active.remove(r)
            continue
        n, nc, h, w = (int(v) for v in hdr[1:])
        ids, mask = torch.empty(nc, 512, dtype=torch.long), torch.empty(nc, 512, dtype=torch.long)
        dist.recv(ids, src=r, group=control_group, tag=TAG_IDS)
        dist.recv(mask, src=r, group=control_group, tag=TAG_MASK)
        x = torch.empty(n, 128, h, w, dtype=torch.bfloat16, device=device)
        t = torch.empty(n, dtype=torch.float32, device=device)
        cap = torch.empty(n, dtype=torch.long, device=device)
        for buf in (x, t, cap):
            dist.recv(buf, src=r)
        start = time.perf_counter()
        text = teacher.encode_ids(ids, mask)
        v = torch.cat([teacher.velocity(x[i:i + micro].float(), t[i:i + micro], text[cap[i:i + micro]], null, cfg)
                       for i in range(0, n, micro)]).to(torch.bfloat16).contiguous()
        dist.send(v, dst=r)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        busy += time.perf_counter() - start
        served += n
        win_n += n
        if time.perf_counter() - win_t > 60:
            dt = time.perf_counter() - win_t
            log(f"teacher server (rank {dist.get_rank()}): busy {100 * busy / dt:.0f}%, {win_n / dt:.1f} states/s",
                flush=True)
            busy, win_t, win_n = 0.0, time.perf_counter(), 0
    log(f"teacher server (rank {dist.get_rank()}): {served:,} states in {time.perf_counter() - t0:.0f} s, done",
        flush=True)


class TeacherClient:
    """A training rank's side: request() splits the trajectories over all teacher ranks (each gets whole
    trajectories, so a caption is encoded once) and sends them without waiting; collect() returns the teachers'
    guided velocity (n, 128, h, w) fp32 in the request's order."""

    def __init__(self, servers, device, control_group):
        self.servers, self.device, self.group = list(servers), device, control_group

    def request(self, ids, mask, x, t, cap):
        """cap: caption index of every state, non-decreasing (states grouped by trajectory)."""
        import torch.distributed as dist
        n, _, h, w = x.shape
        parts = []
        bounds = torch.linspace(0, len(ids), len(self.servers) + 1).round().long().tolist()
        cap_cpu = cap.cpu()
        for server, c0, c1 in zip(self.servers, bounds[:-1], bounds[1:]):
            if c1 == c0:
                continue
            s0, s1 = int(torch.searchsorted(cap_cpu, c0)), int(torch.searchsorted(cap_cpu, c1))
            hdr = torch.tensor([1, s1 - s0, c1 - c0, h, w], dtype=torch.long)
            p_ids, p_mask = ids[c0:c1].contiguous(), mask[c0:c1].contiguous()
            reqs = [dist.isend(hdr, dst=server, group=self.group, tag=TAG_HDR),
                    dist.isend(p_ids, dst=server, group=self.group, tag=TAG_IDS),
                    dist.isend(p_mask, dst=server, group=self.group, tag=TAG_MASK)]
            bufs = [x[s0:s1].to(torch.bfloat16).contiguous(), t[s0:s1].float().contiguous(),
                    (cap[s0:s1] - c0).long().contiguous()]
            reqs += [dist.isend(b, dst=server) for b in bufs]
            out = torch.empty(s1 - s0, 128, h, w, dtype=torch.bfloat16, device=self.device)
            reqs.append(dist.irecv(out, src=server))
            parts.append(dict(reqs=reqs, keep=(hdr, p_ids, p_mask, bufs), out=out))
        return parts

    def collect(self, parts):
        for p in parts:
            for r in p["reqs"]:
                r.wait()
        return torch.cat([p["out"] for p in parts]).float()

    def stop(self):
        import torch.distributed as dist
        for server in self.servers:
            dist.send(torch.tensor([0, 0, 0, 0, 0]), dst=server, group=self.group, tag=TAG_HDR)
