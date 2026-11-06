"""CPU regression checks: python tests/check_training_fastpaths.py (no downloads)."""
import copy
import sys
import unittest
import torch
import torch.nn.functional as F
sys.path.insert(0, '.')
from aurora.model import Denoiser, DenoiserConfig, LMEncoderConfig
from aurora.slayer import LMTextEncoder, LM_NULL, LM_PAD
from aurora.muon import Muon, pack_directions, update_bf16_momentum
from train import weight_sq, WRMS_KEYS
from train_pp import Stage, checkpointed, lm_chunk_plan, text_widths, local_train_step


def make_stage(mode=None):
    cfg = DenoiserConfig(dim=32, depth=2, heads=4, mlp_hidden=64, text_dim=32, latent_size=2)
    tc = LMEncoderConfig(lm_dim=16, dim=32, depth=1, heads=4, kv_heads=2, head_dim=8,
                         mlp_hidden=64, max_tokens=128)
    den, te = Denoiser(cfg), LMTextEncoder(tc, with_lm=False)
    with torch.no_grad():
        for p in list(den.parameters()) + list(te.parameters()):
            p.normal_(0, .08)
    return Stage(den, cfg, 0, 2, True, True, te, act_checkpoint=mode)


class Fastpaths(unittest.TestCase):
    def test_trimmed_gradients_and_auxiliary_loss(self):
        for mode in (None, 'full', 'selective'):
            with self.subTest(checkpoint=mode):
                torch.manual_seed(8)
                a = make_stage(mode)
                a.repa_proj = torch.nn.Linear(32, 7)
                a.repa_block, a.repa_weight = 0, .5
                a.repa_targets = [torch.randn(1, 4, 7) for _ in range(2)]
                b = copy.deepcopy(a)
                # Checkpoint wrappers are closures: deepcopy does not rebind their captured modules.
                b.block_fns = [checkpointed(block, mode) if mode else block for block in b.blocks]
                ids = torch.full((4, 128), LM_PAD, dtype=torch.long)
                ids[0, :9], ids[1, 0], ids[2, :70], ids[3, 0] = 1, LM_NULL, 2, LM_NULL
                states = torch.randn(4, 128, 16)
                a.lm_states, b.lm_states = list(states.chunk(2)), list(states.chunk(2))
                z, t, target = torch.randn(4, 128, 2, 2), torch.rand(4), torch.randn(4, 128, 2, 2)
                ref = []
                for k in range(2):
                    a._mb = k
                    loss = F.mse_loss(a(z[k*2:k*2+2], t[k*2:k*2+2], ids[k*2:k*2+2]), target[k*2:k*2+2])
                    ref.append(loss.detach())
                    loss.backward()
                for p in a.parameters():
                    if p.grad is not None:
                        p.grad.div_(2)
                losses, widths = [], text_widths(ids, 2)
                self.assertEqual(widths, [64, 128])
                local_train_step(b, z, t, ids, target, 2, losses, widths)
                torch.testing.assert_close(torch.stack(ref), torch.stack(losses))
                for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
                    self.assertIsNotNone(q.grad, name)
                    torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-6, msg=name)

    def test_cpu_plans_null_and_partial_chunks(self):
        ids = torch.full((67, 128), LM_PAD, dtype=torch.long)
        ids[:, 0] = LM_NULL
        ids[1, :9], ids[34, :70], ids[66, :] = 1, 2, 3
        plans = lm_chunk_plan(ids)
        self.assertEqual([(i, rows.tolist(), list(sub.shape)) for i, rows, sub in plans],
                         [(0, [1], [1, 16]), (32, [2], [1, 72]), (64, [2], [1, 128])])
        ids[:, 0] = LM_NULL
        self.assertEqual(lm_chunk_plan(ids), [])

    def test_norm_and_weight_stats(self):
        torch.manual_seed(2)
        gs = [torch.randn(31, 17), torch.randn(7)]
        torch.testing.assert_close(torch.stack(torch._foreach_norm(gs)).float().square().sum(),
                                   sum(g.square().sum() for g in gs))
        p = torch.nn.Parameter(gs[0])
        ws = weight_sq([('den', 'blocks.0.attn.qkv.weight', p)], torch.device('cpu'))
        for kind in ('den_attn.qkv', 'hidden'):
            torch.testing.assert_close(ws[WRMS_KEYS.index(kind), 0], p.square().sum())

    def test_compiled_momentum_and_packing(self):
        torch.manual_seed(3)
        grad, old = torch.randn(64, 32), torch.randn(64, 32).bfloat16()
        expected = old.float().lerp(grad, .05)
        for fn in (update_bf16_momentum, torch.compile(update_bf16_momentum, backend='inductor')):
            b = old.clone()
            fn(b, grad, .95)
            lo = torch.nextafter(expected.bfloat16(), torch.full_like(b, -float('inf'))).float()
            hi = torch.nextafter(expected.bfloat16(), torch.full_like(b, float('inf'))).float()
            self.assertTrue(((b.float() >= lo) & (b.float() <= hi)).all())
        for nesterov in (True, False):
            gs, bs = [grad, grad.clone()], [old, old.clone()]
            ref = torch.stack([g.lerp(b.float(), .95) if nesterov else b for g, b in zip(gs, bs)])
            torch.testing.assert_close(pack_directions(gs, bs, .95, nesterov), ref)

    def test_bf16_muon_finite(self):
        for compile_ns in (False, True):
            p = torch.nn.Parameter(torch.randn(16, 8))
            opt = Muon([p], splits={p: [8, 8]}, shard=False, momentum_dtype=torch.bfloat16,
                       compile_ns=compile_ns)
            for _ in range(2):
                p.grad = torch.randn_like(p)
                opt.step()
            self.assertTrue(p.isfinite().all())
            self.assertEqual(opt.state[p]['momentum_buffer'].dtype, torch.bfloat16)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
