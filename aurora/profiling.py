"""Opt-in training traces. No CUDA events, synchronisation or profiler scopes when disabled."""
import json
import os
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch


def export_profile(prof, prefix, metadata=None):
    """Export a timeline, operator shapes and actual kernel names; timings are sums, not GPU utilisation."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(prefix) + '.trace.json')
    averages = prof.key_averages(group_by_input_shape=True)
    rows = []
    for e in averages:
        rows.append(dict(name=e.key, count=e.count, shapes=e.input_shapes,
                         cpu_total_us=e.cpu_time_total, cpu_self_us=e.self_cpu_time_total,
                         device_total_us=e.device_time_total, device_self_us=e.self_device_time_total))
    kernels = defaultdict(lambda: [0, 0.0])
    for e in prof.events():
        # CUDA user annotations span whole graphs/phases and overlap their kernels.
        # Including them here double-counts work (and can even dominate the table).
        if e.device_type == torch.autograd.DeviceType.CUDA and not e.is_user_annotation:
            kernels[e.name][0] += 1
            kernels[e.name][1] += e.time_range.elapsed_us()
    kernel_rows = [dict(name=k, count=v[0], total_us=v[1]) for k, v in kernels.items()]
    kernel_rows.sort(key=lambda r: r['total_us'], reverse=True)
    payload = dict(metadata=metadata or {}, operators=rows, kernels=kernel_rows,
                   timing_note='Operator totals are inclusive; kernels may overlap. Do not sum them as wall time.')
    Path(str(prefix) + '.json').write_text(json.dumps(payload, indent=2))
    tables = averages.table(sort_by='self_device_time_total' if kernels else 'self_cpu_time_total', row_limit=50)
    Path(str(prefix) + '.txt').write_text(tables)
    if os.environ.get("PROFILE_STACK") == "1":  # where the top device-time operators are called from
        stacks = prof.key_averages(group_by_stack_n=8).table(sort_by='self_device_time_total', row_limit=40)
        Path(str(prefix) + '.stack.txt').write_text(stacks)
    print(f'profile -> {prefix}.trace.json / .json / .txt', flush=True)


class TrainingProfile:
    def __init__(self, directory='', rank=0, ranks=(0,), start=17, steps=4, cuda=True, metadata=None):
        if start < 1 or steps < 1:
            raise ValueError('profile start and steps must be positive')
        self.enabled = bool(directory) and rank in ranks
        self.start, self.steps, self.rank = start, steps, rank
        self.directory, self.cuda, self.metadata = directory, cuda, metadata
        self.prof = None
        self.active = False
        self.finished = False

    def begin_step(self, step):
        if self.enabled and not self.finished and step >= self.start and self.prof is None:
            self.first = step
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.prof = torch.profiler.profile(activities=activities, record_shapes=True,
                                               profile_memory=False,
                                               with_stack=os.environ.get("PROFILE_STACK") == "1")
            self.prof.start()
            self.active = True
            print(f'profile rank {self.rank}: capturing steps {step}..{step + self.steps - 1}; '
                  'throughput during capture is not a benchmark', flush=True)

    def phase(self, name):
        return torch.profiler.record_function('train/' + name) if self.active else nullcontext()

    def end_step(self, step):
        if self.active:
            self.prof.step()
            if step >= self.first + self.steps - 1:
                self.close()

    def close(self):
        if self.active:
            self.prof.stop()
            self.active, self.finished = False, True
            export_profile(self.prof, Path(self.directory) / f'rank{self.rank}_step{self.first}', self.metadata)
            self.prof = None
