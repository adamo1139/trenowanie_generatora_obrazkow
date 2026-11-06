"""Profile real DiT shapes, without loading Poziomka or modifying a checkpoint.

  python tests/profile_dit.py --init runs/poziomka_stage2/merged.pt --depth 12 --out runs/profile_dit
  python tests/profile_dit.py --parts block --mb 8 16 24 --text-widths 64 128 256 \
      --compile-modes default max-autotune-no-cudagraphs --checkpoints none selective --trace
  python tests/profile_dit.py --device cpu --smoke --out /tmp/profile_dit_smoke

Synthetic activations, actual checkpoint architecture. Microbenchmarks are not end-to-end training throughput.
All timings follow warmup of the exact shape/backend. Optional traces are collected separately from timings.
"""
import argparse
import gc
import json
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aurora.model import CrossAttention, DenoiserBlock, DenoiserConfig, LMEncoderConfig, make_configs, te_config_from_dict, build_rope_2d
from aurora.profiling import export_profile
from train_pp import checkpointed

BACKENDS = {'flash': SDPBackend.FLASH_ATTENTION, 'efficient': SDPBackend.EFFICIENT_ATTENTION,
            'cudnn': SDPBackend.CUDNN_ATTENTION, 'math': SDPBackend.MATH}


def backend_context(name):
    return sdpa_kernel(BACKENDS[name]) if name in BACKENDS else nullcontext()


def configs(args):
    if args.init:
        # mmap leaves tensors on disk; only architecture and one block's weights are used.
        ck = torch.load(args.init, map_location='cpu', weights_only=False, mmap=True)
        den, te = DenoiserConfig(**ck['den_config']), te_config_from_dict(ck.get('te_config', {}))
    else:
        den, te = make_configs(args.size, args.te_size)
        ck = None
    if args.depth:
        den = replace(den, depth=args.depth)
    if args.smoke:
        den = replace(den, dim=32, depth=2, heads=4, mlp_hidden=64, text_dim=32, latent_size=2)
        te = LMEncoderConfig(dim=32, heads=4, kv_heads=2, head_dim=8, mlp_hidden=64, lm_dim=16)
        ck = None
    return den, te, ck


def lengths(batch, width, device):
    # Sorted captions with padding; one CFG null prompt. Reused across all compared backends.
    lens = torch.linspace(max(1, width // 2), width, batch).long()
    lens[0] = 1
    mask = torch.arange(width)[None, :] < lens[:, None]
    idx = mask.flatten().nonzero().flatten()
    cu = torch.cat([torch.zeros(1, dtype=torch.int32), lens.cumsum(0).int()])
    return mask.to(device), idx.to(device), cu.to(device)


def pack_attention(q, k, v, idx, cu_k, width_q, width_k):
    """Keep all queries, gather valid K/V. Index metadata is prepared on CPU, outside timed execution.

    Includes gather/scatter-layout cost in the measurement, unlike a benchmark of prepacked inputs.
    """
    from torch.nn.attention.varlen import varlen_attn
    batch, heads, _, dim = q.shape
    qp = q.transpose(1, 2).reshape(batch * width_q, heads, dim)
    kp = k.transpose(1, 2).reshape(batch * width_k, heads, dim).index_select(0, idx)
    vp = v.transpose(1, 2).reshape(batch * width_k, heads, dim).index_select(0, idx)
    cu_q = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * width_q
    out = varlen_attn(qp, kp, vp, cu_q, cu_k, width_q, width_k)
    return out.view(batch, width_q, heads, dim).transpose(1, 2)


def gemm_shapes(cfg, te, batch, width):
    """M, N, K for forward; benchmark autograd also measures dX and dW with the same weights."""
    t, d, h = cfg.latent_size ** 2, cfg.dim, cfg.mlp_hidden
    td = cfg.text_dim
    shapes = {'self_qkv': (batch*t, 3*d, d), 'self_out_cross_q_out': (batch*t, d, d),
              'cross_kv': (batch*width, 2*d, td), 'mlp_gate_up': (batch*t, 2*h, d),
              'mlp_down': (batch*t, d, h)}
    if isinstance(te, LMEncoderConfig):
        shapes.update(refiner_qkv=(batch*width, (te.heads+2*te.kv_heads)*te.head_dim, te.dim),
                      refiner_gate_up=(batch*width, 2*te.mlp_hidden, te.dim),
                      refiner_down=(batch*width, te.dim, te.mlp_hidden))
    return shapes


def block_flops(cfg, batch, width):
    t, d = cfg.latent_size**2, cfg.dim
    # Training GEMMs, self QK/AV and cross QK/AV. Excludes norm/activation, softmax, recomputation and optimizer.
    return 6*batch*(t*(6*d*d + 3*d*cfg.mlp_hidden) + width*2*cfg.text_dim*d) + 12*batch*d*(t*t+t*width)


def validate(fn, reference, inputs, amp):
    """Backend forward AND input-gradient comparison to dense SDPA math on the same tensors."""
    with amp():
        ref = reference(*inputs)
    grad = torch.randn_like(ref)
    ref_grads = torch.autograd.grad(ref, inputs, grad)
    with amp():
        got = fn(*inputs)
    grads = torch.autograd.grad(got, inputs, grad)
    def rel(a, b):
        return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-8)).item()
    errors = dict(output_rel=rel(got, ref), grad_rel=max(rel(a, b) for a, b in zip(grads, ref_grads)))
    if not torch.isfinite(got).all() or any(not torch.isfinite(g).all() for g in grads):
        raise ValueError('nonfinite attention output or gradient')
    # BF16 fused backends differ numerically; fail gross errors, retain exact error metrics in JSON.
    if errors['output_rel'] > .03 or errors['grad_rel'] > .05:
        raise ValueError(f'attention reference mismatch: {errors}')
    return errors


def measure(fn, inputs, params, amp, device, warmup, iterations):
    def clear():
        for p in (*inputs, *params):
            if p.requires_grad:
                p.grad = None
    grad = None
    def forward():
        with amp():
            return fn(*inputs)
    begin = time.perf_counter()
    for _ in range(warmup):
        clear()
        y = forward()
        if grad is None:
            grad = torch.randn_like(y)
        y.backward(grad)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    cold_ms = (time.perf_counter()-begin)*1000
    del y
    clear()
    base = torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        events = [[torch.cuda.Event(enable_timing=True) for _ in range(3)] for _ in range(iterations)]
    times = []
    start = time.perf_counter()
    for i in range(iterations):
        clear()
        if device.type == 'cuda':
            a,b,c = events[i]
            a.record()
            y = forward()
            b.record()
            y.backward(grad)
            c.record()
        else:
            a = time.perf_counter()
            y = forward()
            b = time.perf_counter()
            y.backward(grad)
            c = time.perf_counter()
            times.append(((b-a)*1000, (c-b)*1000))
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        times = [(a.elapsed_time(b), b.elapsed_time(c)) for a,b,c in events]
    wall_ms = (time.perf_counter()-start)*1000/iterations
    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    result = dict(warmup_ms=cold_ms, fwd_ms=statistics.median(a for a,b in times),
                  bwd_ms=statistics.median(b for a,b in times),
                  total_ms=statistics.median(a+b for a,b in times), wall_ms=wall_ms,
                  peak_allocated_gb=peak/2**30, peak_minus_base_gb=(peak-base)/2**30,
                  iterations=iterations)
    def trace_step():
        clear()
        forward().backward(grad)
    return result, trace_step


def build_case(part, cfg, te, ck, mb, width, backend, cp, device, dtype):
    rand = lambda *s: torch.randn(*s, device=device, dtype=dtype, requires_grad=True)
    mask, idx, cu = lengths(mb, width, device)
    t, d, heads = cfg.latent_size**2, cfg.dim, cfg.heads
    params, reference = [], None
    if part == 'block':
        mod = DenoiserBlock(d, heads, cfg.text_dim, cfg.mlp_hidden).to(device)
        for module in mod.modules():
            if isinstance(module, CrossAttention):
                module.sdpa_backend = backend
        if ck is not None:
            weights = {k[len('blocks.0.'):]: v for k,v in ck['den'].items() if k.startswith('blocks.0.')}
            mod.load_state_dict(weights)
        else:
            with torch.no_grad():
                mod.mod_table.normal_(0, .1)
        cos, sin = (v.to(device) for v in build_rope_2d(cfg.latent_size, d//heads))
        inputs = (rand(mb,t,d), rand(mb,6,d), rand(mb,width,cfg.text_dim))
        inner = checkpointed(mod, cp) if cp != 'none' else mod
        fn = lambda x, t6, text: inner(x,t6,text,mask,cos,sin)
        params = list(mod.parameters())
        flops = block_flops(cfg, mb, width)
    elif part.startswith('gemm:'):
        m,n,k = gemm_shapes(cfg,te,mb,width)[part.split(':')[1]]
        mod = torch.nn.Linear(k,n,bias=False).to(device)
        inputs = (rand(m,k),)
        fn, params, flops = mod, list(mod.parameters()), 6*m*n*k
    else:
        is_self = part == 'attention_self'
        is_refiner = part == 'attention_refiner'
        if is_refiner:
            heads, dim, tq = te.heads, te.head_dim, width
        else:
            dim, tq = d//heads, t
        tk = tq if is_self else width
        # SDPA core only, already projected Q/K/V; block and GEMM tests account for projections.
        inputs = (rand(mb,heads,tq,dim), rand(mb,heads,tk,dim), rand(mb,heads,tk,dim))
        bias = None if is_self else mask[:,None,None,:]
        def reference(q,k,v):
            with sdpa_kernel(SDPBackend.MATH):
                return F.scaled_dot_product_attention(q,k,v,attn_mask=bias)
        if backend == 'varlen':
            fn = lambda q,k,v: pack_attention(q,k,v,idx,cu,tq,tk)
        elif backend in BACKENDS:
            selected_backend = BACKENDS[backend]
            def fn(q, k, v):
                # Make backend selection explicit in the traced graph, rather than
                # relying on external global flags across compilation/cache reuse.
                with sdpa_kernel(selected_backend):
                    return F.scaled_dot_product_attention(q,k,v,attn_mask=bias)
        else:
            fn = lambda q,k,v: F.scaled_dot_product_attention(q,k,v,attn_mask=bias)
        flops = 12*mb*heads*dim*tq*tk
        if backend == 'varlen':
            # Query rows are dense; only valid KV tokens are multiplied.
            valid_tokens = int(mask.sum().cpu())
            flops = 12*heads*dim*tq*valid_tokens
    return fn, inputs, params, flops, reference


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--init', help='trusted training checkpoint, architecture + first denoiser block')
    p.add_argument('--size', default='750m')
    p.add_argument('--te-size', default='poziomka')
    p.add_argument('--depth', type=int, default=12)
    p.add_argument('--parts', nargs='+', choices=['block','gemm','attention'], default=['attention','gemm','block'])
    p.add_argument('--mb', nargs='+', type=int, default=[8,16])
    p.add_argument('--text-widths', nargs='+', type=int, default=[64,128])
    p.add_argument('--backends', nargs='+', choices=['auto',*BACKENDS,'varlen'], default=['auto','efficient','cudnn','varlen'])
    p.add_argument('--block-backends', nargs='+', choices=['auto','cudnn'], default=['auto'],
                   help='cross-attention backend inside the real denoiser block')
    p.add_argument('--compile-modes', nargs='+', choices=['eager','default','max-autotune-no-cudagraphs'], default=['default'])
    p.add_argument('--checkpoints', nargs='+', choices=['none','full','selective'], default=['selective'])
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--device', default='cuda')
    p.add_argument('--trace', action='store_true', help='separate profiler iteration per successful case')
    p.add_argument('--smoke', action='store_true', help='tiny CPU-testable shapes, not a performance benchmark')
    p.add_argument('--out', default='runs/profile_dit')
    args = p.parse_args()
    if min(args.mb+args.text_widths+[args.warmup,args.iterations]) < 1:
        p.error('sizes, warmup and iterations must be positive')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable. Run on the GPU host; --device cpu --smoke only checks the tool.')
    if args.smoke:
        args.mb, args.text_widths, args.compile_modes, args.backends = [2], [16], ['eager'], ['auto','math']
        args.warmup, args.iterations = 1, 2
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    import torch._inductor.config as ic
    ic.split_reductions = False  # match train_pp.py
    cfg, te, ck = configs(args)
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    metadata = dict(torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU smoke',
                    den_config=asdict(cfg), te_config=asdict(te), args=vars(args),
                    notes='Synthetic inputs; one block, not whole-model memory. Gradient buffers cleared per iteration. '
                          'FLOPs count useful GEMM/attention work, exclude checkpoint recomputation and optimizer.')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    results = []
    cases = []
    for mb in args.mb:
        for width in args.text_widths:
            if 'attention' in args.parts:
                for part in ('attention_self','attention_cross','attention_refiner'):
                    for backend in args.backends:
                        if part == 'attention_self' and backend == 'varlen':
                            continue
                        for mode in args.compile_modes:
                            cases.append((part,mb,width,backend,'none',mode))
            for mode in args.compile_modes:
                if 'gemm' in args.parts:
                    for name in gemm_shapes(cfg,te,mb,width):
                        cases.append(('gemm:'+name,mb,width,'auto','none',mode))
                if 'block' in args.parts:
                    for cp in args.checkpoints:
                        for backend in args.block_backends:
                            cases.append(('block',mb,width,backend,cp,mode))
    for number,(part,mb,width,backend,cp,mode) in enumerate(cases):
        label=f'{number:03d}_{part.replace(":","_")}_mb{mb}_t{width}_{backend}_{cp}_{mode}'
        row=dict(part=part,mb=mb,text_width=width,backend=backend,checkpoint=cp,compile_mode=mode)
        print(f'[{number+1}/{len(cases)}] {label}',flush=True)
        fn=inputs=params=trace_step=None
        try:
            torch._dynamo.reset()  # isolate compile variants and avoid recompile_limit across the sweep
            torch.manual_seed(123)
            fn,inputs,params,flops,reference=build_case(part,cfg,te,ck,mb,width,backend,cp,device,dtype)
            if mode != 'eager':
                fn=torch.compile(fn,dynamic=False,mode=mode)
            if reference is not None:
                with backend_context(backend):
                    row.update(validate(fn,reference,inputs,amp))
            # A block selects only its cross-attention backend internally. A global
            # context here would also switch self-attention, unlike train_pp.py.
            with backend_context(backend) if part != 'block' else nullcontext():
                timing,trace_step=measure(fn,inputs,params,amp,device,args.warmup,args.iterations)
                row.update(timing,status='ok',useful_flops=flops,
                           useful_tflops=flops/timing['total_ms']/1e9)
                if args.trace:
                    activities=[torch.profiler.ProfilerActivity.CPU]
                    if device.type == 'cuda': activities.append(torch.profiler.ProfilerActivity.CUDA)
                    with torch.profiler.profile(activities=activities,record_shapes=True) as prof:
                        trace_step()
                    export_profile(prof,out/label,row)
            print(f'  fwd {row["fwd_ms"]:.2f} ms | bwd {row["bwd_ms"]:.2f} ms | '
                  f'{row["useful_tflops"]:.1f} useful TFLOPS | peak {row["peak_allocated_gb"]:.2f} GiB',flush=True)
        except (RuntimeError, ValueError, ImportError, NotImplementedError) as e:
            row.update(status='error',error=f'{type(e).__name__}: {e}')
            print('  '+row['error'],flush=True)
        finally:
            results.append(row)
            (out/'results.json').write_text(json.dumps(results,indent=2))
            del fn,inputs,params,trace_step
            gc.collect()
            if device.type == 'cuda': torch.cuda.empty_cache()
    good=sum(r['status']=='ok' for r in results)
    print(f'{good}/{len(results)} successful cases -> {out}/results.json',flush=True)
    if not good: raise SystemExit(1)


if __name__ == '__main__':
    main()
