"""Newton-Schulz with symmetric matmuls (Triton), for Muon.

Two of the three matmuls of a quintic Newton-Schulz iteration produce symmetric matrices: A = X X^T and A A (A is
symmetric, so A A = A A^T). `syrk` computes only the lower-triangle tiles of such a product and writes each tile and its
mirror, so those two matmuls cost half; the third (B X) stays a cuBLAS baddbmm with the a X term fused in. That removes
~1/3 of the FLOPs on square matrices, ~29% on 2560 x 6912. b A + c A A is formed in fp32 in the kernel's epilogue.
Measured on the 12-block HomeIMG model's Muon matrices (RTX 5080): 869 -> 577 ms per step (1.51x); distance to an fp32
Newton-Schulz 0.9% vs 1.7-1.9% for the plain bf16 version.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({"BM": bm, "BK": bk}, num_warps=w, num_stages=s)
             for bm, bk, w, s in [(128, 32, 4, 4), (128, 64, 8, 3), (128, 32, 8, 4), (64, 64, 4, 4), (128, 64, 4, 3)]],
    key=["M", "K"],
)
@triton.jit
def _syrk_kernel(X, D, C, M, K, sxb, sxm, sxk, sdb, sdm, sdn, scb, scm, scn, alpha, beta,
                 HAS_D: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    i, j, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if j > i:  # upper triangle: written by its mirror tile
        return
    rm = i * BM + tl.arange(0, BM)
    rn = j * BM + tl.arange(0, BM)
    rk = tl.arange(0, BK)
    xa = X + b * sxb + rm[:, None] * sxm + rk[None, :] * sxk
    xb = X + b * sxb + rn[:, None] * sxm + rk[None, :] * sxk
    acc = tl.zeros((BM, BM), dtype=tl.float32)
    for k in range(0, K, BK):
        kmask = (k + rk)[None, :] < K
        a = tl.load(xa, mask=(rm[:, None] < M) & kmask, other=0.0)
        bb = tl.load(xb, mask=(rn[:, None] < M) & kmask, other=0.0)
        acc += tl.dot(a, tl.trans(bb))
        xa += BK * sxk
        xb += BK * sxk
    acc = acc * alpha
    mask = (rm[:, None] < M) & (rn[None, :] < M)
    if HAS_D:
        d = tl.load(D + b * sdb + rm[:, None] * sdm + rn[None, :] * sdn, mask=mask, other=0.0)
        acc += beta * d.to(tl.float32)
    out = acc.to(C.dtype.element_ty)
    tl.store(C + b * scb + rm[:, None] * scm + rn[None, :] * scn, out, mask=mask)
    if i != j:
        tl.store(C + b * scb + rn[:, None] * scm + rm[None, :] * scn, tl.trans(out),
                 mask=(rn[:, None] < M) & (rm[None, :] < M))


def syrk(x: torch.Tensor, d: torch.Tensor = None, alpha: float = 1.0, beta: float = 0.0) -> torch.Tensor:
    """(B, M, K) -> alpha x x^T + beta d, (B, M, M), exactly symmetric; d must be symmetric (only its lower tiles are
    read)."""
    assert x.ndim == 3
    B, M, K = x.shape
    c = torch.empty(B, M, M, device=x.device, dtype=x.dtype)
    dd = d if d is not None else c
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(M, meta["BM"]), B)
    _syrk_kernel[grid](x, dd, c, M, K, *x.stride(), *dd.stride(), *c.stride(), alpha, beta, HAS_D=d is not None)
    return c


def newtonschulz5_sym(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Same iteration as aurora.muon.zeropower_via_newtonschulz5, batched (B, m, n), with the symmetric products in
    `syrk`."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    squeeze = X.ndim == 2
    if squeeze:
        X = X.unsqueeze(0)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)).contiguous()
    for _ in range(steps):
        A = syrk(X)
        B = syrk(A, A, alpha=c, beta=b)  # b A + c A A
        X = torch.baddbmm(X, B, X, beta=a)
    if transposed:
        X = X.mT
    return X.squeeze(0) if squeeze else X
