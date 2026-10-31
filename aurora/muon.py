"""Muon (https://kellerjordan.github.io/posts/muon/) for 2D hidden weight matrices, sharded across DDP ranks.

Gradients arrive already all-reduced by DDP. Then, for every group of same-shape matrices:
  * every rank updates the momentum of all of them (cheap elementwise, keeps optimizer state identical on all ranks,
    so the rank-0 checkpoint is complete),
  * each rank orthogonalises only its contiguous slice of the group, stacked into one batched Newton-Schulz call,
  * the bf16 updates are all-gathered and every rank applies all of them -> weights stay identical.
Measured on the 63M model: per-matrix loop 41 ms/step, stacked 20 ms, stacked + 1/8 slice 4.5 ms.

Two DiT-specific variants (both off by default):
  * CMuon (arXiv:2608.02502, `splits`): fused matrices (qkv, cross-attention kv, SwiGLU gate/up) are cut into their
    functional row blocks and each block is orthogonalised on its own; orthogonalising the fused matrix couples the
    blocks' spectra, which the paper blames for Muon's late-stage plateau on DiTs. Each block keeps the fused matrix's
    shape scale (= the paper's sqrt(n_chunks) rescale), so --lr-muon carries over.
  * Periodic row-wise Muon (arXiv:2608.20818, `period` K, `gamma`): Newton-Schulz only every K-th step; in between,
    the momentum's vectors along the longer side are normalised to unit length and scaled by gamma (0.15 in the
    paper, K = 3). Cuts optimizer time ~2x at matched quality in their 1.3B-15B MMDiT runs.
"""
import math

import torch
import torch.distributed as dist


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalise the last two dims of G (works batched)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def rownorm(G: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Row-wise update of periodic row-wise Muon: in the wide orientation (rows <= cols), every row scaled to unit
    norm. Same Frobenius norm as an orthogonalised matrix (sqrt(min(rows, cols))). Works batched."""
    X = G.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / X.norm(dim=-1, keepdim=True).clamp_min(eps)
    if transposed:
        X = X.mT
    return X.bfloat16()


def _all_gather(out: torch.Tensor, local: torch.Tensor):
    if dist.get_backend() == "nccl":
        dist.all_gather_into_tensor(out, local)
    else:  # gloo (single-GPU tests) only gathers CPU tensors
        parts = list(torch.empty((out.shape[0],) + out.shape[1:], dtype=out.dtype).chunk(dist.get_world_size()))
        dist.all_gather(parts, local.cpu())
        out.copy_(torch.cat(parts))


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0, compile_ns=False,
                 shard=True, splits=None, period=1, gamma=0.15):
        """splits: {param: [row block sizes]} for CMuon; period / gamma: periodic row-wise Muon (period 1 = off)."""
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                                      weight_decay=weight_decay))
        # shard=False: every rank orthogonalises everything locally (DiLoCo, where ranks hold different weights)
        distributed = shard and dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if distributed else 0
        self.world = dist.get_world_size() if distributed else 1
        self.period, self.gamma, self._t = period, gamma, 0
        splits = splits or {}
        # buckets of same-shape (row block of a) matrices, in parameter order (identical on every rank); entries are
        # (param, first row, last row, shape scale). Kept outside param_groups so they are not serialised.
        self._buckets = []
        for group in self.param_groups:
            by_shape = {}
            for p in group["params"]:
                assert p.ndim == 2, "Muon is only for 2D weight matrices"
                rows, cols = p.shape
                scale = max(1.0, rows / cols) ** 0.5  # of the whole (fused) matrix
                sizes = splits.get(p, [rows])
                assert sum(sizes) == rows, (p.shape, sizes)
                r0 = 0
                for n in sizes:
                    by_shape.setdefault((n, cols), []).append((p, r0, r0 + n, scale))
                    r0 += n
            self._buckets.append(list(by_shape.values()))
        self._ns = torch.compile(zeropower_via_newtonschulz5) if compile_ns else zeropower_via_newtonschulz5

    @torch.no_grad()
    def step(self):
        refresh = self._t % self.period == 0  # Newton-Schulz step (always, unless periodic row-wise is on)
        self._t += 1
        for group, buckets in zip(self.param_groups, self._buckets):
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            ps = group["params"]
            grads = [p.grad for p in ps]
            assert all(g is not None for g in grads), "Muon expects a gradient for every matrix"
            for p in ps:
                if "momentum_buffer" not in self.state[p]:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p)
            torch._foreach_lerp_([self.state[p]["momentum_buffer"] for p in ps], grads, 1 - mom)
            torch._foreach_mul_(ps, 1 - lr * wd)
            for entries in buckets:
                rows, cols = entries[0][2] - entries[0][1], entries[0][0].shape[1]

                def direction(i):
                    p, r0, r1, _ = entries[i]
                    buf = self.state[p]["momentum_buffer"][r0:r1]
                    return p.grad[r0:r1].lerp(buf, mom) if group["nesterov"] else buf

                if refresh:
                    per = math.ceil(len(entries) / self.world)
                    lo, hi = min(len(entries), self.rank * per), min(len(entries), (self.rank + 1) * per)
                    local = torch.zeros(per, rows, cols, dtype=torch.bfloat16, device=entries[0][0].device)
                    if hi > lo:
                        local[: hi - lo] = self._ns(torch.stack([direction(i) for i in range(lo, hi)]),
                                                    group["ns_steps"])
                    if self.world > 1:
                        out = torch.empty(self.world * per, rows, cols, dtype=torch.bfloat16, device=local.device)
                        _all_gather(out, local)
                    else:
                        out = local
                    out = out[: len(entries)]
                else:  # row-wise step: cheap, every rank does all of it (momentum is identical across ranks)
                    out = rownorm(torch.stack([direction(i) for i in range(len(entries))])) * self.gamma
                updates = list(out.float().unbind(0))
                torch._foreach_mul_(updates, [e[3] for e in entries])
                torch._foreach_add_([p[r0:r1] for p, r0, r1, _ in entries], updates, alpha=-lr)
