"""Muon (https://kellerjordan.github.io/posts/muon/) for 2D hidden weight matrices, sharded across DDP ranks.

Gradients arrive already all-reduced by DDP. Then, for every group of same-shape matrices:
  * every rank updates the momentum of all of them (cheap elementwise, keeps optimizer state identical on all ranks,
    so the rank-0 checkpoint is complete),
  * each rank orthogonalises only its contiguous slice of the group, stacked into one batched Newton-Schulz call,
  * the bf16 updates are all-gathered and every rank applies all of them -> weights stay identical.
Measured on the 63M model: per-matrix loop 41 ms/step, stacked 20 ms, stacked + 1/8 slice 4.5 ms.
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


def _all_gather(out: torch.Tensor, local: torch.Tensor):
    if dist.get_backend() == "nccl":
        dist.all_gather_into_tensor(out, local)
    else:  # gloo (single-GPU tests) only gathers CPU tensors
        parts = list(torch.empty((out.shape[0],) + out.shape[1:], dtype=out.dtype).chunk(dist.get_world_size()))
        dist.all_gather(parts, local.cpu())
        out.copy_(torch.cat(parts))


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0, compile_ns=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                                      weight_decay=weight_decay))
        distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if distributed else 0
        self.world = dist.get_world_size() if distributed else 1
        # same-shape buckets, in parameter order (identical on every rank); kept outside param_groups so they are not
        # serialised into the optimizer state_dict
        self._buckets = []
        for group in self.param_groups:
            by_shape = {}
            for p in group["params"]:
                assert p.ndim == 2, "Muon is only for 2D weight matrices"
                by_shape.setdefault(tuple(p.shape), []).append(p)
            self._buckets.append(list(by_shape.values()))
        self._ns = torch.compile(zeropower_via_newtonschulz5) if compile_ns else zeropower_via_newtonschulz5

    @torch.no_grad()
    def step(self):
        for group, buckets in zip(self.param_groups, self._buckets):
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            for ps in buckets:
                grads = [p.grad for p in ps]
                assert all(g is not None for g in grads), "Muon expects a gradient for every matrix"
                for p in ps:
                    if "momentum_buffer" not in self.state[p]:
                        self.state[p]["momentum_buffer"] = torch.zeros_like(p)
                bufs = [self.state[p]["momentum_buffer"] for p in ps]
                torch._foreach_lerp_(bufs, grads, 1 - mom)

                per = math.ceil(len(ps) / self.world)
                lo, hi = min(len(ps), self.rank * per), min(len(ps), (self.rank + 1) * per)
                rows, cols = ps[0].shape
                local = torch.zeros(per, rows, cols, dtype=torch.bfloat16, device=ps[0].device)
                if hi > lo:
                    g = torch.stack([grads[i].lerp(bufs[i], mom) if group["nesterov"] else bufs[i]
                                     for i in range(lo, hi)])
                    local[: hi - lo] = self._ns(g, group["ns_steps"])
                if self.world > 1:
                    out = torch.empty(self.world * per, rows, cols, dtype=torch.bfloat16, device=local.device)
                    _all_gather(out, local)
                else:
                    out = local
                updates = list(out[: len(ps)].float().unbind(0))
                torch._foreach_mul_(ps, 1 - lr * wd)
                torch._foreach_add_(ps, updates, alpha=-lr * max(1.0, rows / cols) ** 0.5)
