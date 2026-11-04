"""Grouped GEMM for MoE inference in one Triton launch: rows of `a` are sorted by expert (counts[e] rows each, in
expert order) and every group is multiplied by its own expert's weight, y[rows of e] = a[rows of e] @ w[e].

torch._grouped_mm runs this as one cuBLAS/CUTLASS call per expert on GPUs before Hopper; with 128 small experts
(~256 rows x 2048 x 320 each) that is ~6k tiny kernels per forward of Poziomka at ~27 TFLOPS. Here every program
takes one BLOCK_M slice of one expert's rows; the schedule (which expert, which rows) is built on the GPU from the
counts, so there is no host sync either.
"""
import torch
import triton
import triton.language as tl

BLOCK_M = 64  # fixed: the block schedule is computed before the launch


@triton.autotune(
    configs=[triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s)
             for bn, bk, w, s in [(64, 64, 4, 3), (128, 32, 4, 4), (128, 64, 8, 3), (64, 32, 4, 4),
                                  (128, 64, 4, 3), (256, 32, 8, 3)]],
    key=["N", "K", "M_BUCKET"],
)
@triton.jit
def _grouped_kernel(a_ptr, w_ptr, y_ptr, block_e_ptr, block_row_ptr, row_end_ptr, a_idx_ptr, y_idx_ptr, scale_ptr,
                    wscale_ptr, N, K, M_BUCKET, stride_am, stride_we, stride_wk, stride_ym,
                    GATHER_A: tl.constexpr, SCATTER_Y: tl.constexpr, SCALE: tl.constexpr, W_INT8: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(block_e_ptr + pid_m)
    if e < 0:  # unused tail of the upper-bound grid
        return
    row0 = tl.load(block_row_ptr + pid_m)
    row_end = tl.load(row_end_ptr + pid_m)
    rm = row0 + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    row_ok = rm < row_end
    a_row = tl.load(a_idx_ptr + rm, mask=row_ok, other=0) if GATHER_A else rm
    a_ptrs = a_ptr + a_row[:, None].to(tl.int64) * stride_am + rk[None, :]
    w_ptrs = w_ptr + e.to(tl.int64) * stride_we + rk[:, None] * stride_wk + rn[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=row_ok[:, None] & (rk[None, :] + k < K), other=0.0)
        w = tl.load(w_ptrs, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0)
        if W_INT8:  # weight-only int8: integers are exact in bf16, the per-column scale comes after the sum
            w = w.to(a.dtype)
        acc += tl.dot(a, w)
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * stride_wk
    if W_INT8:
        acc *= tl.load(wscale_ptr + e.to(tl.int64) * N + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
    if SCALE:
        acc *= tl.load(scale_ptr + rm, mask=row_ok, other=0.0).to(tl.float32)[:, None]
    y_row = tl.load(y_idx_ptr + rm, mask=row_ok, other=0) if SCATTER_Y else rm
    y_ptrs = y_ptr + y_row[:, None].to(tl.int64) * stride_ym + rn[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=row_ok[:, None] & (rn[None, :] < N))


def grouped_mm(a: torch.Tensor, w: torch.Tensor, counts: torch.Tensor, sched=None, a_idx=None, y_idx=None,
               scale=None, w_scale=None) -> torch.Tensor:
    """a (M, K) sorted by expert, w (E, K, N) contiguous, counts (E,) rows per expert (sum = M) -> (M, N).
    Fused extras for an MoE layer, all indexed by the sorted row r: a_idx[r] = the row of `a` to read (so `a` is the
    unsorted token matrix, no gathered copy), scale[r] multiplies output row r (routing weight), y_idx[r] = where
    output row r is written (back in token order, no scatter afterwards). w may be int8 with w_scale (E, N): the
    weight is then w * w_scale[e] (per expert and output column)."""
    M = a_idx.numel() if a_idx is not None else a.shape[0]
    K = a.shape[1]
    E, _, N = w.shape
    assert w.stride(2) == 1, "w must be (E, K, N) with contiguous rows"
    a = a.contiguous()
    if sched is None:
        sched = make_schedule(counts, M)
    block_e, block_row, row_end = sched
    y = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = lambda meta: (block_e.numel(), triton.cdiv(N, meta["BLOCK_N"]))
    dummy = block_e
    _grouped_kernel[grid](a, w, y, block_e, block_row, row_end,
                          dummy if a_idx is None else a_idx, dummy if y_idx is None else y_idx,
                          dummy if scale is None else scale, dummy if w_scale is None else w_scale, N, K, triton.next_power_of_2(max(1, M // E)),
                          a.stride(0), w.stride(0), w.stride(1), y.stride(0),
                          GATHER_A=a_idx is not None, SCATTER_Y=y_idx is not None, SCALE=scale is not None,
                          W_INT8=w_scale is not None,
                          BLOCK_M=BLOCK_M)
    return y


def make_schedule(counts: torch.Tensor, M: int):
    """Per M-block of BLOCK_M rows: (expert or -1, first row, end row of that expert). The grid is an upper bound,
    ceil(M / BLOCK_M) + E blocks, so nothing is read back to the host. Shared by the matmuls of one MoE layer."""
    E = counts.numel()
    starts = counts.cumsum(0) - counts
    nblk = (counts + BLOCK_M - 1) // BLOCK_M
    cum = nblk.cumsum(0)
    nb = triton.cdiv(M, BLOCK_M) + E
    b = torch.arange(nb, device=counts.device)
    e = torch.searchsorted(cum, b, right=True)
    valid = e < E
    ec = e.clamp(max=E - 1)
    block_row = starts[ec] + (b - (cum[ec] - nblk[ec])) * BLOCK_M
    row_end = starts[ec] + counts[ec]
    return (torch.where(valid, ec, -1).to(torch.int32), block_row.to(torch.int32), row_end.to(torch.int32))
