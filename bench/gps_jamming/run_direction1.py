"""
Direction 1, Experiment 7: is the sedenion-structured 16-parameter widely-linear subspace a GOOD subspace?

Filter: y = p^T L_q E(x), q in S (16 real parameters), fixed embedding E(x) = (Re x, Im x), fixed readout p.
H(q) = p^T L_q is a 2x16 real matrix, LINEAR in q, so {H(q)} is a 16-dimensional subspace of the
32-dimensional space of real-linear scalar beamformers.  We compare it, under IDENTICAL constraints,
against other 16-dimensional subspaces and against the full 32-dimensional (WL) space:

   sed_q(p)        {p^T L_q}                       16 params, sedenion left-regular representation
   complex_linear  {w^H x}                         16 params, the conventional subspace
   random_16 (x5)  random 16-dim subspaces of R^{2x16}
   full_WL         all 2x16                        32 params (ceiling)

Constraints (all methods): H e_s = (1,0), H J e_s = (0,1)  [satellite complex-linear gain 1, conjugate
gain 0 -> carrier preserved exactly];  H e_j = 0 for each jammer direction (rectilinear, known phase: 1 real
direction; circular/unknown phase: {e_j, J e_j}).  Objective: minimum noise gain ||H||_F / sqrt2 subject to
the constraints (the LCMV criterion).  Infeasible => reported as such.  Also: data-driven MVDR variants.
"""
import json, math, pathlib, sys, importlib.util
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("okc", HERE / "okc_revised_crpa_validation.py")
okc = importlib.util.module_from_spec(spec); spec.loader.exec_module(okc)
cd_mul, r16, c8, I16, J16 = okc.cd_mul, okc.r16, okc.c8, okc.I16, okc.J16


def L_basis():
    """B_k = L_{e_k} (16x16): left multiplication by basis unit e_k."""
    return [np.stack([cd_mul(I16[k], I16[i]) for i in range(16)], 1) for k in range(16)]
LB = L_basis()


def subspace_sed(p_rows):
    """16 matrices p^T L_{e_k}, each 2x16; p_rows = (i, j) readout coordinates."""
    P = np.stack([I16[p_rows[0]], I16[p_rows[1]]])
    return [P @ B for B in LB]

def subspace_complex():
    out = []
    for k in range(8):
        for part in (1, 1j):
            w = np.zeros(8, complex); w[k] = part; out.append(okc.w_to_h(w))
    return out

def subspace_random(rng):
    G = rng.standard_normal((32, 16)); Q, _ = np.linalg.qr(G)
    return [Q[:, k].reshape(2, 16) for k in range(16)]

def subspace_full():
    return [np.eye(32)[k].reshape(2, 16) for k in range(32)]


def solve_min_noise(basis, constraints):
    """min ||sum_k q_k B_k||_F  s.t.  (sum_k q_k B_k) c = t for (c, t) in constraints.  Returns H or None."""
    n = len(basis)
    G = np.array([[np.sum(Bk * Bl) for Bl in basis] for Bk in basis])        # n x n Gram
    A = np.stack([np.concatenate([Bk @ c for Bk in basis]) for c, _ in constraints], 0)  # rows: per constraint 2 eqs stacked below
    A = np.vstack([np.stack([(Bk @ c) for Bk in basis], 1) for c, _ in constraints])   # (2m) x n
    b = np.concatenate([t for _, t in constraints])
    # feasibility
    sol, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    if np.linalg.norm(A @ sol - b) > 1e-8: return None
    # KKT: min q^T G q  s.t. A q = b
    K = np.block([[2 * G, A.T], [A, np.zeros((A.shape[0], A.shape[0]))]])
    rhs = np.concatenate([np.zeros(n), b])
    z = np.linalg.lstsq(K, rhs, rcond=None)[0]; q = z[:n]
    if np.linalg.norm(A @ q - b) > 1e-6: return None
    return sum(qk * Bk for qk, Bk in zip(q, basis))


def noise_gain(H): return math.sqrt((H[0] @ H[0] + H[1] @ H[1]) / 2)


def trial(rng, J, circular, eps=1e-3):
    while True:
        ths = rng.uniform(-60, 60, J + 1)
        if all(abs(ths[i] - ths[k]) >= 8 for i in range(J + 1) for k in range(i)): break
    jam_c = [okc.steer(t) * np.exp(1j * rng.uniform(0, 2 * np.pi)) for t in ths[:J]]
    sat = r16(okc.steer(ths[J]))
    est = [r16(okc.estimate_with_error(z, eps, rng)) for z in jam_c]
    true = [r16(z) for z in jam_c]
    cons = [(sat, np.array([1.0, 0.0])), (J16 @ sat, np.array([0.0, 1.0]))]
    for e in est:
        cons.append((e, np.zeros(2)))
        if circular: cons.append((J16 @ e, np.zeros(2)))
    subs = {"sed_q (p=e0,e8)": subspace_sed((0, 8)), "sed_q (p=e1,e2)": subspace_sed((1, 2)),
            "sed_q (p=e1,e9)": subspace_sed((1, 9)), "complex_linear": subspace_complex(), "full_WL": subspace_full()}
    for r in range(3): subs[f"random_16 #{r}"] = subspace_random(rng)
    rows = {}
    phases = np.exp(1j * np.linspace(0, 2 * np.pi, 8, endpoint=False))
    for name, basis in subs.items():
        H = solve_min_noise(basis, cons)
        if H is None: rows[name] = None; continue
        if circular:
            leak = [abs(okc.apply_phase(H, c8(t), p)) for t in true for p in phases]
        else:
            leak = [abs(okc.h_to_z(H @ t)) for t in true]
        rows[name] = dict(noise_gain=noise_gain(H), null_db=-20 * math.log10(max(math.sqrt(np.mean(np.square(leak))), 1e-16)))
    return rows


def run(n, J, circular, rng):
    acc = {}
    infeas = {}
    for _ in range(n):
        for name, r in trial(rng, J, circular).items():
            if r is None: infeas[name] = infeas.get(name, 0) + 1; continue
            d = acc.setdefault(name, {"noise_gain": [], "null_db": []})
            for k in d: d[k].append(r[k])
    out = {}
    for name in set(list(acc) + list(infeas)):
        if name in acc:
            out[name] = dict(noise_gain=round(float(np.mean(acc[name]["noise_gain"])), 3),
                             noise_gain_db=round(float(20 * np.log10(np.mean(acc[name]["noise_gain"]))), 2),
                             null_db=round(float(np.mean(acc[name]["null_db"])), 1), infeasible=infeas.get(name, 0))
        else:
            out[name] = dict(infeasible=infeas[name])
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    rng = np.random.default_rng(2026); res = {}
    for J, circ in [(1, False), (2, False), (3, False), (5, False), (1, True), (2, True)]:
        key = f"J={J} {'circular (unknown phase)' if circ else 'rectilinear (known phase)'}"
        res[key] = run(n, J, circ, rng)
        print("==", key, f"[{4 + (2 if circ else 1) * 2 * J} real constraint equations]")
        for m in sorted(res[key], key=lambda m: res[key][m].get("noise_gain", 99)):
            print(f"   {m:20s} {res[key][m]}", flush=True)
    (HERE / "results_direction1.json").write_text(json.dumps(res, indent=1))
