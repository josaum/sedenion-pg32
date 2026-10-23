"""
Table 2 (algebraic columns) — measured, no training involved.

For every representation algebra A we draw B unit vectors u and report

  CLR_min(u) = min_{|v|=1} |u*v| / |u|  = sigma_min(L(u)) / |u|
      the smallest leakage *any* override vector can achieve. A division algebra
      has CLR_min == 1 identically (L(u)/|u| is orthogonal), so a contradiction can
      never be annihilated; only an algebra with zero divisors can reach 0.

  CLR_constr = |u*v| / (|u||v|) for the constructive partner v (Moreno row only).

  PSI(u,v,c) = |(uv)c - u(vc)| / (|(uv)c| + |u(vc)|)
      0 for associative algebras, 1 iff one bracketing vanishes and the other does not.

  softmax_bleed = weight an "overridden" slot keeps under softmax attention at a
      stated logit gap, in an 8-slot window — the Dense-R16 analogue of CLR. It is
      temperature-dependent, so the gap is reported with the number.
"""
import json, math, sys, pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sedenion_pg32 import tables, sed_mul, left_mult_matrix, MorenoConeProjection, moreno_nullspace_partner

B, SEED = 1024, 2026


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def sample(algebra, dtype):
    """Unit vectors in R^16 restricted to the sub-structure of each architecture row."""
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(B, 16, generator=g, dtype=dtype)
    if algebra == "Quat-H4":      # 4 independent quaternion channels: e0..e3 of 4 copies
        # product acts blockwise; model as block-diag L -> sample one quaternion per block
        return unit(x)
    if algebra == "Oct-O2":       # 2 independent octonion channels
        return unit(x)
    if algebra == "Sed-Free":
        return unit(x)
    if algebra == "Sed-Moreno":
        return MorenoConeProjection(True)(x) / math.sqrt(2)
    raise ValueError(algebra)


def left_mult(algebra, u, K, X):
    """L(u) for the row's product structure."""
    if algebra == "Quat-H4":
        blocks = [left_mult_matrix(torch.nn.functional.pad(u[..., 4 * i:4 * i + 4], (0, 12)), K, X)[..., :4, :4]
                  for i in range(4)]
        return torch.block_diag(*blocks) if u.dim() == 1 else torch.stack([torch.block_diag(*[b[r] for b in blocks]) for r in range(u.shape[0])])
    if algebra == "Oct-O2":
        blocks = [left_mult_matrix(torch.nn.functional.pad(u[..., 8 * i:8 * i + 8], (0, 8)), K, X)[..., :8, :8]
                  for i in range(2)]
        return torch.stack([torch.block_diag(*[b[r] for b in blocks]) for r in range(u.shape[0])])
    return torch.stack([left_mult_matrix(r, K, X) for r in u])


def product(algebra, u, v, K, X):
    if algebra == "Quat-H4":
        return torch.cat([sed_mul(torch.nn.functional.pad(u[..., 4*i:4*i+4], (0, 12)),
                                  torch.nn.functional.pad(v[..., 4*i:4*i+4], (0, 12)), K, X)[..., :4] for i in range(4)], -1)
    if algebra == "Oct-O2":
        return torch.cat([sed_mul(torch.nn.functional.pad(u[..., 8*i:8*i+8], (0, 8)),
                                  torch.nn.functional.pad(v[..., 8*i:8*i+8], (0, 8)), K, X)[..., :8] for i in range(2)], -1)
    return sed_mul(u, v, K, X)


def run(dtype):
    S, K, X = tables(dtype)
    out = {}
    g = torch.Generator().manual_seed(SEED + 1)
    for alg in ("Quat-H4", "Oct-O2", "Sed-Free", "Sed-Moreno"):
        u = sample(alg, dtype)
        L = left_mult(alg, u, K, X)
        sv = torch.linalg.svdvals(L.double())
        clr_min = (sv[:, -1] / u.double().norm(dim=-1))
        row = {"CLR_min_mean": clr_min.mean().item(), "CLR_min_min": clr_min.min().item(),
               # per-channel: sigma_min of a single block / that block's norm (Hurwitz => 1 for H, O)
               "CLR_min_per_channel": (torch.linalg.svdvals(left_mult(alg, u, K, X)[:, :(4 if alg == "Quat-H4" else 8 if alg == "Oct-O2" else 16),
                                                                                   :(4 if alg == "Quat-H4" else 8 if alg == "Oct-O2" else 16)].double())[:, -1]
                                        / u[:, :(4 if alg == "Quat-H4" else 8 if alg == "Oct-O2" else 16)].double().norm(dim=-1)).mean().item(),
               "rank_L_mode": int(torch.mode((sv > (1e-10 if dtype == torch.float64 else 1e-5) * sv[:, :1]).sum(1)).values)}
        if alg == "Sed-Moreno":
            v = moreno_nullspace_partner(u, torch.randn(B, 7, generator=g, dtype=dtype), K, X)
            c = unit(torch.randn(B, 16, generator=g, dtype=dtype))
            p = sed_mul(u, v, K, X)
            row["CLR_constr_mean"] = (p.norm(dim=-1) / (u.norm(dim=-1) * v.norm(dim=-1))).mean().item()
            row["CLR_constr_max"] = (p.norm(dim=-1) / (u.norm(dim=-1) * v.norm(dim=-1))).max().item()
        else:
            v = unit(torch.randn(B, 16, generator=g, dtype=dtype))
            c = unit(torch.randn(B, 16, generator=g, dtype=dtype))
        l = product(alg, product(alg, u, v, K, X), c, K, X)
        r = product(alg, u, product(alg, v, c, K, X), K, X)
        psi = (l - r).norm(dim=-1) / (l.norm(dim=-1) + r.norm(dim=-1) + (1e-300 if dtype == torch.float64 else 1e-38))
        row["PSI_mean"] = psi.mean().item()
        out[alg] = row
    # Dense-R16 softmax analogue
    for gap in (2.0, 4.0, 8.0):
        logits = torch.zeros(8, dtype=torch.float64); logits[0] = -gap
        out.setdefault("Dense-R16", {})[f"softmax_bleed_gap{int(gap)}"] = torch.softmax(logits, 0)[0].item()
    # Dense-R16: the only product is matrix multiplication, which is associative; measure it anyway
    g2 = torch.Generator().manual_seed(SEED + 2)
    A, Bm, C = (torch.randn(B, 16, 16, generator=g2, dtype=torch.float64) for _ in range(3))
    l = (A @ Bm) @ C; r = A @ (Bm @ C)
    out["Dense-R16"]["PSI_mean"] = ((l - r).flatten(1).norm(dim=-1) / (l.flatten(1).norm(dim=-1) + r.flatten(1).norm(dim=-1))).mean().item()
    return out


if __name__ == "__main__":
    res = {"B": B, "seed": SEED, "fp64": run(torch.float64), "fp32": run(torch.float32)}
    pathlib.Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=2))
    for prec in ("fp64", "fp32"):
        print(f"== {prec} ==")
        for alg, row in res[prec].items():
            print(f"{alg:11s} " + "  ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
