"""CPU protocol test: torchrun --standalone --nproc_per_node 3 tests/check_lm_server.py"""
import sys
from datetime import timedelta
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.nn as nn
sys.path.insert(0, '.')
from aurora.slayer import LM_NULL, LM_PAD, hf_lm_part
from train_pp import LMClient, lm_chunk_plan, lm_server


def make_lm():
    torch.manual_seed(1)
    lm = nn.Module()
    lm.model = nn.Module()
    lm.model.word_embeddings = nn.Embedding(50, 8)
    lm.model.layers = nn.ModuleList([])
    lm.model.norm = nn.Identity()
    lm.model.rotary_emb = lambda x, pos: None
    return lm.bfloat16()


def fake_vae(u8):  # (B, 3, H, H) uint8 -> (B, 4, H/4, H/4) float32, deterministic
    return u8.float()[:, :1, ::4, ::4].repeat(1, 4, 1, 1) / 255 + torch.arange(4.).view(1, 4, 1, 1)


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=timedelta(seconds=45))
    rank = dist.get_rank()
    control = dist.new_group(backend='gloo', timeout=timedelta(seconds=45))
    device, lm = torch.device('cpu'), make_lm()
    if rank == 2:
        lm_server(SimpleNamespace(repo='test'), [0, 1], device, print, control, lm=lm, vae=fake_vae)
    else:
        client = LMClient(2, 8, device, control)
        # Rank 0 cannot submit until rank 1 receives its result: round-robin would deadlock.
        if rank == 0:
            dist.recv(torch.empty(1, dtype=torch.long), src=1, group=control, tag=99)
        for iteration in range(3):
            ids = torch.full((67, 128), LM_PAD, dtype=torch.long)
            ids[:, 0] = LM_NULL
            if iteration != 1:
                ids[1, :9], ids[34, :70], ids[66, :] = 1, 2, 3
            # --vae-on-lm order of the training loop: the batch's images first, then its captions
            u8 = torch.randint(0, 256, (5 + rank, 3, 16, 16), dtype=torch.uint8)
            hv = client.request_vae(u8, (5 + rank, 4, 4, 4))
            hl = client.request(ids)
            torch.testing.assert_close(LMClient.vae_collect(hv), fake_vae(u8), rtol=0, atol=0)
            got = client.collect(hl)
            ref = torch.zeros(67, 128, 8, dtype=torch.bfloat16)
            for i, rows, sub in lm_chunk_plan(ids):
                h = hf_lm_part(lm, sub.masked_fill(sub.eq(LM_PAD), 0), True, True)
                ref[:, :sub.shape[1]].index_copy_(0, rows + i, h)
            torch.testing.assert_close(got, ref, rtol=0, atol=0)
            if rank == 1 and iteration == 0:
                dist.send(torch.ones(1, dtype=torch.long), dst=0, group=control, tag=99)
        client.stop()
    dist.barrier()
    if rank == 0:
        print('OK: ready-client scheduling, partial/null/repeated requests, VAE requests and shutdown')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
