"""Every trainable parameter must receive a nonzero gradient (catches dead branches, e.g. gate AND output both zero-init).

  python tests/check_grads.py

Gated branches legitimately get zero gradient on the very first step (gate = 0), so this checks the gradients after a few
optimizer steps.
"""
import sys
import torch

sys.path.insert(0, ".")
from aurora.model import SIZES, TE_SIZES, build_models, make_configs
from train import FlowModel, make_mlm

dev = "cuda" if torch.cuda.is_available() else "cpu"
for size, te_size in (("50m", "13m"), ("253m", "27m"), ("50m", "slayer149")):
    torch.manual_seed(0)
    te, den = build_models(*make_configs(size, te_size))
    te.to(dev), den.to(dev)
    lm = te_size == "slayer149"
    model = FlowModel(te, den, 0.0 if lm else 0.05)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for step in range(4):
        x0 = torch.randn(4, 128, 16, 16, device=dev)
        ids = torch.randint(6, 8192, (4, te.config.max_tokens), device=dev)
        ids[0] = torch.from_numpy(te.null_ids()).to(dev)  # the CFG null prompt must train too
        mlm = (None, None) if lm else make_mlm(ids, 0.15)
        loss = model(x0, ids, torch.rand(4, device=dev), torch.randn_like(x0), *mlm)[0]
        loss.backward()
        if step < 3:
            opt.step()
            opt.zero_grad()
    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().max().item() == 0.0)]
    print(f"{size} + TE {te_size}: {sum(1 for _ in model.parameters())} params, dead: {dead or 'none'}")
    assert not dead, dead
print("OK")
