"""CPU-only checks for opt-in trace lifecycle and GEMM/FLOP accounting."""
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aurora.model import make_configs
from aurora.profiling import TrainingProfile, export_profile
from profile_dit import block_flops, gemm_shapes


class ProfilingTests(unittest.TestCase):
    def test_cuda_annotations_do_not_double_count_kernels(self):
        prof = MagicMock()
        def event(name, annotation, us):
            return SimpleNamespace(name=name, device_type=torch.autograd.DeviceType.CUDA,
                                   is_user_annotation=annotation,
                                   time_range=SimpleNamespace(elapsed_us=lambda: us))
        prof.events.return_value = [event('train/forward_backward', True, 1000),
                                    event('gemm_kernel', False, 100)]
        prof.key_averages.return_value.table.return_value = ''
        with tempfile.TemporaryDirectory() as d:
            export_profile(prof, Path(d)/'test')
            rows = json.loads((Path(d)/'test.json').read_text())['kernels']
            self.assertEqual(rows, [dict(name='gemm_kernel', count=1, total_us=100)])

    def test_disabled_does_not_create_profiler(self):
        p = TrainingProfile('', cuda=False)
        p.begin_step(17)
        with p.phase('forward_backward'):
            pass
        p.end_step(17)
        self.assertIsNone(p.prof)
        self.assertFalse(p.active)

    def test_selected_rank_window_and_export(self):
        with tempfile.TemporaryDirectory() as d:
            other = TrainingProfile(d, rank=1, ranks=[0], start=2, steps=2, cuda=False)
            other.begin_step(2)
            self.assertFalse(other.active)
            p = TrainingProfile(d, rank=0, start=2, steps=2, cuda=False)
            p.begin_step(1)
            self.assertFalse(p.active)
            for step in [2,3]:
                p.begin_step(step)
                with p.phase('forward_backward'):
                    x = torch.randn(8,8,requires_grad=True)
                    (x@x).sum().backward()
                p.end_step(step)
            self.assertTrue(p.finished)
            payload=json.loads((Path(d)/'rank0_step2.json').read_text())
            self.assertTrue(any(r['name']=='train/forward_backward' for r in payload['operators']))
            p.begin_step(4)
            self.assertFalse(p.active)

    def test_flops_use_actual_text_width(self):
        cfg,te=make_configs('750m','poziomka',depth=12)
        small=gemm_shapes(cfg,te,8,64)
        big=gemm_shapes(cfg,te,8,256)
        self.assertEqual(small['self_qkv'],big['self_qkv'])
        self.assertEqual(big['cross_kv'][0],4*small['cross_kv'][0])
        self.assertLess(block_flops(cfg,8,64),block_flops(cfg,8,256))
        self.assertEqual(block_flops(cfg,16,64),2*block_flops(cfg,8,64))


if __name__ == '__main__':
    unittest.main()
